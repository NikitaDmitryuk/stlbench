from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table
from shapely import affinity
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.fit import aabb_edge_lengths, compute_global_scale
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.mesh_repair import repair_cache_key, repair_report_step
from stlbench.core.overhang import find_stable_overhang_rotation
from stlbench.export.plate import export_plate_3mf
from stlbench.export.transform_log import (
    bounds_to_list,
    mesh_bounds,
    placement_transform_for_mesh,
    transform_bounds,
    transform_entry,
    transform_step,
    translation_matrix,
    uniform_scale_matrix,
    write_transform_log,
)
from stlbench.packing.bitmap_pack import (
    BitmapPackOptions,
    BitmapPackStats,
    pack_polygons_bitmap_single_plate,
)
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.polygon_footprint import mesh_to_packing_shadow
from stlbench.packing.polygon_pack import try_pack_polygons_single_plate
from stlbench.packing.rectpack_plate import PackedPlate, PackedRect, pack_rectangles_on_plates
from stlbench.pipeline.common import (
    finish_profile,
    load_named_meshes_with_repair,
    n_workers,
    repair_cache_dir_for_output,
    resolve_edge_margin,
    resolve_gap,
    resolve_orientation_policy,
    resolve_orientation_scale_tolerance,
    resolve_printer,
    resolve_repair_cache_enabled,
    resolve_repair_options,
    resolve_resin_orientation_options,
    resolve_settings,
    write_command_repair_report,
)
from stlbench.pipeline.progress import make_progress
from stlbench.profiling import ProfileOptions, make_profiler


@dataclass
class AutopackRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    post_fit_scale: float | None
    orient_on: bool
    orient_threshold_deg: float
    dry_run: bool
    recursive: bool
    any_rotation: bool = False
    maximize: bool = False
    orientation_policy: str | None = None
    orientation_scale_tolerance: float | None = None
    rotation_samples: int | None = None
    verbose: bool = False
    cleanup: bool = False
    repair: bool = False
    repair_cache: bool = True
    profile_options: ProfileOptions | None = None
    edge_margin_mm: float | None = None
    resin_balance: str | None = None
    autopack_pack_workers: int | str | None = None
    autopack_result_cache: bool = True
    autopack_attempt_cache: bool = True
    autopack_scale_tolerance: float | None = None
    autopack_packer: str | None = None
    autopack_bitmap_grid_mm: float | None = None
    autopack_bitmap_beam_width: int | None = None
    progress: bool = True


@dataclass(frozen=True)
class _PackAttempt:
    scale: float
    success: bool
    plate: PackedPlate | None
    duration_s: float
    cache_hit: bool = False


@dataclass
class _PackCacheStats:
    attempt_cache_hits: int = 0
    pack_attempts_run: int = 0
    pack_attempts_failed: int = 0
    result_cache_hit: bool = False
    bitmap_rasterize_s: float = 0.0
    bitmap_search_s: float = 0.0
    bitmap_candidates_tested: int = 0
    bitmap_fallback_scans: int = 0
    exact_validation_s: float = 0.0


@dataclass(frozen=True)
class _AutopackSearchResult:
    scale: float
    plate: PackedPlate | None
    stats: _PackCacheStats


def _try_pack_all(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
) -> PackedPlate | None:
    """Try to pack all polygons onto a single plate. Returns None on failure."""
    return try_pack_polygons_single_plate(polygons, bed_w, bed_h, gap_mm)


def _scale_polygons(polygons: list[BaseGeometry], scale: float) -> list[BaseGeometry]:
    return [affinity.scale(poly, xfact=scale, yfact=scale, origin=(0.0, 0.0)) for poly in polygons]


def _bisect_scale(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    s_upper: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> tuple[float, PackedPlate | None]:
    """Binary search for the maximum scale at which all parts fit on one plate."""
    lo, hi = 0.0, s_upper
    best_s = 0.0
    best_plate: PackedPlate | None = None

    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        scaled = _scale_polygons(base_polygons, mid)
        plate = _try_pack_all(scaled, bed_w, bed_h, gap_mm)
        if plate is not None:
            best_s = mid
            best_plate = plate
            lo = mid
        else:
            hi = mid

    return best_s, best_plate


def _default_pack_workers(n_parts: int) -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(n_parts, cpu, 4))


def _resolve_pack_workers(
    value: int | str | None, settings_value: int | str | None, n_parts: int
) -> int:
    source = settings_value if value is None else value
    if source is None or source == "auto":
        return _default_pack_workers(n_parts)
    out = int(source)
    if out < 1:
        raise ValueError("--autopack-pack-workers must be 'auto' or a positive integer.")
    return out


def _resolve_autopack_scale_tolerance(value: float | None, settings_value: float | None) -> float:
    out = float(settings_value if value is None and settings_value is not None else (value or 1e-4))
    if out <= 0:
        raise ValueError("--autopack-scale-tol must be > 0.")
    return out


def _resolve_autopack_packer(value: str | None, settings_value: str | None) -> str:
    out = value or settings_value or "auto"
    if out not in {"auto", "bitmap", "exact"}:
        raise ValueError("--autopack-packer must be auto, bitmap, or exact.")
    return out


def _resolve_bitmap_options(
    *,
    grid_mm: float | None,
    settings_grid_mm: float | None,
    beam_width: int | None,
    settings_beam_width: int | None,
) -> BitmapPackOptions:
    resolved_grid = float(
        settings_grid_mm if grid_mm is None and settings_grid_mm is not None else (grid_mm or 0.25)
    )
    resolved_beam = int(
        settings_beam_width
        if beam_width is None and settings_beam_width is not None
        else (beam_width or 16)
    )
    if resolved_grid <= 0:
        raise ValueError("--autopack-bitmap-grid-mm must be > 0.")
    if resolved_beam < 1:
        raise ValueError("--autopack-bitmap-beam-width must be >= 1.")
    return BitmapPackOptions(grid_mm=resolved_grid, beam_width=resolved_beam)


def _rect_to_json(rect: PackedRect) -> dict[str, float | int]:
    return {
        "part_index": rect.part_index,
        "x": rect.x,
        "y": rect.y,
        "width": rect.width,
        "height": rect.height,
        "rotation_deg": rect.rotation_deg,
    }


def _plate_to_json(plate: PackedPlate) -> dict[str, object]:
    return {"index": plate.index, "rects": [_rect_to_json(r) for r in plate.rects]}


def _plate_from_json(payload: object) -> PackedPlate | None:
    if not isinstance(payload, dict):
        return None
    rect_payloads = payload.get("rects")
    if not isinstance(rect_payloads, list):
        return None
    rects: list[PackedRect] = []
    for item in rect_payloads:
        if not isinstance(item, dict):
            return None
        try:
            rects.append(
                PackedRect(
                    part_index=int(item["part_index"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    width=float(item["width"]),
                    height=float(item["height"]),
                    rotation_deg=float(item.get("rotation_deg", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    try:
        index = int(payload.get("index", 0))
    except (TypeError, ValueError):
        return None
    return PackedPlate(index=index, rects=tuple(rects))


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _pack_cache_key(
    *,
    packer: str = "exact",
    bitmap_options: BitmapPackOptions | None = None,
    source_paths: list[Path],
    repair_cache_keys: list[str | None],
    footprint_keys: list[str],
    names: list[str],
    orientation_transforms: list[np.ndarray],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    geometric_upper: float,
    post_fit_scale: float,
    scale_tolerance: float,
) -> str:
    payload = {
        "version": 3,
        "packer": packer,
        "packer_version": _packer_version(packer),
        "bitmap_grid_mm": bitmap_options.grid_mm if bitmap_options is not None else None,
        "bitmap_beam_width": bitmap_options.beam_width if bitmap_options is not None else None,
        "source_paths": [str(p) for p in source_paths],
        "repair_cache_keys": repair_cache_keys,
        "footprint_keys": footprint_keys,
        "names": names,
        "orientation": [
            np.round(np.asarray(t, dtype=np.float64), 9).tolist() for t in orientation_transforms
        ],
        "bed_w": round(float(bed_w), 9),
        "bed_h": round(float(bed_h), 9),
        "gap_mm": round(float(gap_mm), 9),
        "geometric_upper": round(float(geometric_upper), 9),
        "post_fit_scale": round(float(post_fit_scale), 9),
        "scale_tolerance": round(float(scale_tolerance), 9),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _packer_version(packer: str) -> str:
    return "bitmap_pack_v3" if packer in {"auto", "bitmap"} else "polygon_pack_nfp_v1"


def _scale_key(scale: float) -> str:
    return f"{scale:.8f}"


def _attempt_path(cache_dir: Path, pack_key: str, scale: float) -> Path:
    return cache_dir / "attempts" / pack_key / f"{_scale_key(scale)}.json"


def _load_pack_attempt(
    cache_dir: Path | None,
    pack_key: str,
    scale: float,
    *,
    packer_version: str = "polygon_pack_nfp_v1",
) -> _PackAttempt | None:
    if cache_dir is None:
        return None
    path = _attempt_path(cache_dir, pack_key, scale)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("packer_version") != packer_version:
            return None
        plate = _plate_from_json(payload.get("plate")) if payload.get("success") else None
        return _PackAttempt(
            scale=float(payload["scale"]),
            success=bool(payload["success"]),
            plate=plate,
            duration_s=float(payload.get("duration_s", 0.0)),
            cache_hit=True,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _write_pack_attempt(
    cache_dir: Path | None,
    pack_key: str,
    attempt: _PackAttempt,
    *,
    packer_version: str = "polygon_pack_nfp_v1",
) -> None:
    if cache_dir is None:
        return
    payload: dict[str, object] = {
        "packer_version": packer_version,
        "scale": attempt.scale,
        "success": attempt.success,
        "duration_s": attempt.duration_s,
        "plate": _plate_to_json(attempt.plate) if attempt.plate is not None else None,
    }
    _write_json_atomic(_attempt_path(cache_dir, pack_key, attempt.scale), payload)


def _result_path(cache_dir: Path, pack_key: str) -> Path:
    return cache_dir / "results" / pack_key / "result.json"


def _load_autopack_result(
    cache_dir: Path | None,
    pack_key: str,
    *,
    packer_version: str = "polygon_pack_nfp_v1",
) -> tuple[float, PackedPlate] | None:
    if cache_dir is None:
        return None
    path = _result_path(cache_dir, pack_key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("packer_version") != packer_version:
            return None
        plate = _plate_from_json(payload.get("plate"))
        if plate is None:
            return None
        return float(payload["scale"]), plate
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _write_autopack_result(
    cache_dir: Path | None,
    pack_key: str,
    scale: float,
    plate: PackedPlate,
    *,
    packer_version: str = "polygon_pack_nfp_v1",
) -> None:
    if cache_dir is None:
        return
    payload: dict[str, object] = {
        "packer_version": packer_version,
        "scale": scale,
        "plate": _plate_to_json(plate),
    }
    _write_json_atomic(_result_path(cache_dir, pack_key), payload)


def _pack_attempt_worker(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    scale: float,
) -> _PackAttempt:
    start = time.perf_counter()
    scaled = _scale_polygons(base_polygons, scale)
    plate = _try_pack_all(scaled, bed_w, bed_h, gap_mm)
    return _PackAttempt(
        scale=scale,
        success=plate is not None,
        plate=plate,
        duration_s=time.perf_counter() - start,
    )


def _validate_plate_geometry(
    base_polygons: list[BaseGeometry],
    plate: PackedPlate,
    scale: float,
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    *,
    tolerance: float = 1e-3,
) -> bool:
    if not _plate_has_all_parts_once(plate, len(base_polygons)):
        return False
    bed = shapely_box(-tolerance, -tolerance, bed_w + tolerance, bed_h + tolerance)
    placed: list[BaseGeometry] = []
    for rect in plate.rects:
        if rect.part_index < 0 or rect.part_index >= len(base_polygons):
            return False
        poly = affinity.scale(
            base_polygons[rect.part_index],
            xfact=scale,
            yfact=scale,
            origin=(0.0, 0.0),
        )
        poly = affinity.rotate(poly, rect.rotation_deg, origin=(0.0, 0.0))
        poly = affinity.translate(poly, -poly.bounds[0], -poly.bounds[1])
        poly = affinity.translate(poly, rect.x, rect.y)
        if not bed.covers(poly):
            return False
        for other in placed:
            if gap_mm > 0 and poly.distance(other) + tolerance < gap_mm:
                return False
            if gap_mm <= 0 and poly.intersects(other) and poly.intersection(other).area > tolerance:
                return False
        placed.append(poly)
    return True


def _pack_at_scale_cached(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    scale: float,
    *,
    cache_dir: Path | None,
    pack_key: str,
    write_cache: bool,
    stats: _PackCacheStats,
) -> _PackAttempt:
    cached = _load_pack_attempt(cache_dir, pack_key, scale)
    if cached is not None:
        stats.attempt_cache_hits += 1
        return cached
    attempt = _pack_attempt_worker(base_polygons, bed_w, bed_h, gap_mm, scale)
    stats.pack_attempts_run += 1
    if not attempt.success:
        stats.pack_attempts_failed += 1
    if write_cache:
        _write_pack_attempt(cache_dir, pack_key, attempt)
    return attempt


def _bitmap_attempt_worker(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    scale: float,
    bitmap_options: BitmapPackOptions,
    stats: _PackCacheStats | None = None,
) -> _PackAttempt:
    start = time.perf_counter()
    validation_s = 0.0

    def _validator(candidate_plate: PackedPlate) -> bool:
        nonlocal validation_s
        validation_start = time.perf_counter()
        ok = _validate_plate_geometry(base_polygons, candidate_plate, scale, bed_w, bed_h, gap_mm)
        validation_s += time.perf_counter() - validation_start
        return ok

    bitmap_result = pack_polygons_bitmap_single_plate(
        base_polygons,
        bed_w,
        bed_h,
        gap_mm,
        scale,
        bitmap_options,
        validator=_validator,
    )
    if stats is not None:
        _merge_bitmap_stats(stats, bitmap_result.stats)
        stats.exact_validation_s += validation_s
    plate = bitmap_result.plate
    success = plate is not None
    return _PackAttempt(
        scale=scale,
        success=success,
        plate=plate,
        duration_s=time.perf_counter() - start,
    )


def _merge_bitmap_stats(stats: _PackCacheStats, bitmap_stats: BitmapPackStats) -> None:
    stats.bitmap_rasterize_s += bitmap_stats.rasterize_s
    stats.bitmap_search_s += bitmap_stats.search_s
    stats.bitmap_candidates_tested += bitmap_stats.candidates_tested
    stats.bitmap_fallback_scans += bitmap_stats.fallback_scans


def _pack_bitmap_at_scale_cached(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    scale: float,
    bitmap_options: BitmapPackOptions,
    *,
    cache_dir: Path | None,
    pack_key: str,
    write_cache: bool,
    stats: _PackCacheStats,
) -> _PackAttempt:
    packer_version = _packer_version("bitmap")
    cached = _load_pack_attempt(cache_dir, pack_key, scale, packer_version=packer_version)
    if cached is not None:
        stats.attempt_cache_hits += 1
        return cached
    attempt = _bitmap_attempt_worker(
        base_polygons,
        bed_w,
        bed_h,
        gap_mm,
        scale,
        bitmap_options,
        stats,
    )
    stats.pack_attempts_run += 1
    if not attempt.success:
        stats.pack_attempts_failed += 1
    if write_cache:
        _write_pack_attempt(cache_dir, pack_key, attempt, packer_version=packer_version)
    return attempt


def _search_scale_bitmap_cached(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    s_upper: float,
    *,
    tol: float,
    bitmap_options: BitmapPackOptions,
    cache_dir: Path | None,
    pack_key: str,
    read_result_cache: bool,
    write_result_cache: bool,
    write_attempt_cache: bool,
    max_iter: int = 50,
) -> _AutopackSearchResult:
    stats = _PackCacheStats()
    packer_version = _packer_version("bitmap")
    if read_result_cache:
        cached_result = _load_autopack_result(cache_dir, pack_key, packer_version=packer_version)
        if cached_result is not None:
            scale, plate = cached_result
            if scale > 0 and _validate_plate_geometry(
                base_polygons, plate, scale, bed_w, bed_h, gap_mm
            ):
                stats.result_cache_hit = True
                return _AutopackSearchResult(scale=scale, plate=plate, stats=stats)

    lo, hi = 0.0, s_upper
    best_s = 0.0
    best_plate: PackedPlate | None = None
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        attempt = _pack_bitmap_at_scale_cached(
            base_polygons,
            bed_w,
            bed_h,
            gap_mm,
            mid,
            bitmap_options,
            cache_dir=cache_dir,
            pack_key=pack_key,
            write_cache=write_attempt_cache,
            stats=stats,
        )
        if attempt.success and attempt.plate is not None:
            best_s = mid
            best_plate = attempt.plate
            lo = mid
        else:
            hi = mid

    if best_plate is not None and write_result_cache:
        _write_autopack_result(
            cache_dir,
            pack_key,
            best_s,
            best_plate,
            packer_version=packer_version,
        )
    return _AutopackSearchResult(scale=best_s, plate=best_plate, stats=stats)


def _rect_seed_scale(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    upper: float,
    tol: float,
) -> float:
    footprints = []
    for poly in base_polygons:
        minx, miny, maxx, maxy = poly.bounds
        footprints.append((float(maxx - minx), float(maxy - miny)))
    lo, hi = 0.0, upper
    best = 0.0
    for _ in range(24):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        scaled = [(w * mid, h * mid) for w, h in footprints]
        try:
            plates = pack_rectangles_on_plates(scaled, bed_w, bed_h, gap_mm, max_plates=1)
            ok = len(plates) == 1 and len(plates[0].rects) == len(base_polygons)
        except RuntimeError:
            ok = False
        if ok:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def _candidate_scales(lower_hint: float, upper: float, tol: float, workers: int) -> list[float]:
    if upper <= 0:
        return []
    fractions = (0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
    candidates = {min(upper, max(tol, upper * f)) for f in fractions}
    if lower_hint > 0:
        candidates.add(min(upper, lower_hint))
        span = max(upper - lower_hint, tol)
        for f in (0.25, 0.5, 0.75):
            candidates.add(min(upper, lower_hint + span * f))
    return sorted(candidates)[: max(6, workers * 3)]


def _run_candidate_batch(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    candidates: list[float],
    *,
    cache_dir: Path | None,
    pack_key: str,
    write_cache: bool,
    workers: int,
    stats: _PackCacheStats,
) -> list[_PackAttempt]:
    attempts: list[_PackAttempt] = []
    pending: list[float] = []
    for scale in candidates:
        cached = _load_pack_attempt(cache_dir, pack_key, scale)
        if cached is not None:
            stats.attempt_cache_hits += 1
            attempts.append(cached)
        else:
            pending.append(scale)
    if not pending:
        return attempts
    if workers <= 1 or len(pending) == 1:
        for scale in pending:
            attempt = _pack_attempt_worker(base_polygons, bed_w, bed_h, gap_mm, scale)
            attempts.append(attempt)
            stats.pack_attempts_run += 1
            if not attempt.success:
                stats.pack_attempts_failed += 1
            if write_cache:
                _write_pack_attempt(cache_dir, pack_key, attempt)
        return attempts
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_pack_attempt_worker, base_polygons, bed_w, bed_h, gap_mm, scale): scale
            for scale in pending
        }
        for future in as_completed(futures):
            attempt = future.result()
            attempts.append(attempt)
            stats.pack_attempts_run += 1
            if not attempt.success:
                stats.pack_attempts_failed += 1
            if write_cache:
                _write_pack_attempt(cache_dir, pack_key, attempt)
    return attempts


def _scale_plate_dimensions(plate: PackedPlate, factor: float) -> PackedPlate:
    if abs(factor - 1.0) <= 1e-12:
        return plate
    return PackedPlate(
        index=plate.index,
        rects=tuple(
            PackedRect(
                part_index=r.part_index,
                x=r.x,
                y=r.y,
                width=r.width * factor,
                height=r.height * factor,
                rotation_deg=r.rotation_deg,
            )
            for r in plate.rects
        ),
    )


def _search_scale_cached_parallel(
    base_polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    s_upper: float,
    *,
    tol: float,
    pack_workers: int,
    cache_dir: Path | None,
    pack_key: str,
    read_result_cache: bool,
    write_result_cache: bool,
    write_attempt_cache: bool,
    max_iter: int = 50,
) -> _AutopackSearchResult:
    stats = _PackCacheStats()
    if read_result_cache:
        cached_result = _load_autopack_result(cache_dir, pack_key)
        if cached_result is not None:
            scale, plate = cached_result
            if scale > 0 and _validate_plate_geometry(
                base_polygons, plate, scale, bed_w, bed_h, gap_mm
            ):
                stats.result_cache_hit = True
                return _AutopackSearchResult(scale=scale, plate=plate, stats=stats)

    lower_hint = _rect_seed_scale(base_polygons, bed_w, bed_h, gap_mm, s_upper, tol)
    candidates = _candidate_scales(lower_hint, s_upper, tol, pack_workers)
    attempts = _run_candidate_batch(
        base_polygons,
        bed_w,
        bed_h,
        gap_mm,
        candidates,
        cache_dir=cache_dir,
        pack_key=pack_key,
        write_cache=write_attempt_cache,
        workers=pack_workers,
        stats=stats,
    )
    successes = [a for a in attempts if a.success and a.plate is not None]
    failures = [a for a in attempts if not a.success]
    best_success = max(successes, key=lambda a: a.scale, default=None)
    lo = best_success.scale if best_success is not None else 0.0
    best_plate = best_success.plate if best_success is not None else None
    higher_failures = [a.scale for a in failures if a.scale > lo]
    hi = min(higher_failures) if higher_failures else s_upper

    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        attempt = _pack_at_scale_cached(
            base_polygons,
            bed_w,
            bed_h,
            gap_mm,
            mid,
            cache_dir=cache_dir,
            pack_key=pack_key,
            write_cache=write_attempt_cache,
            stats=stats,
        )
        if attempt.success and attempt.plate is not None:
            lo = attempt.scale
            best_plate = attempt.plate
        else:
            hi = attempt.scale

    if best_plate is not None and write_result_cache:
        _write_autopack_result(cache_dir, pack_key, lo, best_plate)
    return _AutopackSearchResult(scale=lo, plate=best_plate, stats=stats)


def _plate_has_all_parts_once(plate: PackedPlate, n_parts: int) -> bool:
    return sorted(rect.part_index for rect in plate.rects) == list(range(n_parts))


def _footprint_cache_key(
    *,
    source_path: Path,
    repair_cache_key: str | None,
    orientation: np.ndarray,
) -> str:
    payload = {
        "version": 1,
        "source_path": str(source_path),
        "repair_cache_key": repair_cache_key,
        "orientation": np.round(np.asarray(orientation, dtype=np.float64), 9).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_cached_footprint(cache_dir: Path, key: str) -> BaseGeometry | None:
    path = cache_dir / f"{key}.pkl"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            loaded = pickle.load(fh)  # noqa: S301 - trusted local cache keyed by source content.
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    return loaded if isinstance(loaded, BaseGeometry) else None


def _write_cached_footprint(cache_dir: Path, key: str, geometry: BaseGeometry) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / f"{key}.pkl"
    if final_path.exists():
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=cache_dir)
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "wb", closefd=True) as fh:
            pickle.dump(geometry, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def run_autopack(args: AutopackRunArgs) -> int:
    console = Console(stderr=True)
    profile_metadata: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "dry_run": args.dry_run,
        "any_rotation": args.any_rotation,
        "maximize": args.maximize,
    }
    profiler = make_profiler(
        command="autopack",
        output_base=args.output_dir,
        options=args.profile_options,
        metadata=profile_metadata,
    )
    profiler.start()
    st = resolve_settings(args.config_path)
    args.progress = args.progress and (st.ui.progress if st is not None else True)
    repair_options = resolve_repair_options(args.repair, st)
    repair_cache_dir = repair_cache_dir_for_output(
        args.output_dir,
        resolve_repair_cache_enabled(args.repair_cache, st) and not args.dry_run,
    )

    if args.maximize and not args.any_rotation:
        console.print("[red]--maximize requires --any-rotation[/red]")
        return finish_profile(profiler, console, 2)

    try:
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    gap = resolve_gap(args.gap_mm, st)
    edge_margin = resolve_edge_margin(args.edge_margin_mm, st)
    resin_options = resolve_resin_orientation_options(args.resin_balance, st)
    post_fit_scale = (
        float(args.post_fit_scale)
        if args.post_fit_scale is not None
        else (st.scaling.post_fit_scale if st else 1.0)
    )
    try:
        orientation_policy = resolve_orientation_policy(args.orientation_policy)
        scale_tolerance = resolve_orientation_scale_tolerance(args.orientation_scale_tolerance)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)
    try:
        pack_workers = _resolve_pack_workers(
            args.autopack_pack_workers,
            st.autopack.pack_workers if st is not None else None,
            1,
        )
        autopack_scale_tolerance = _resolve_autopack_scale_tolerance(
            args.autopack_scale_tolerance,
            st.autopack.scale_tolerance if st is not None else None,
        )
        autopack_packer = _resolve_autopack_packer(
            args.autopack_packer,
            st.autopack.packer if st is not None else None,
        )
        bitmap_options = _resolve_bitmap_options(
            grid_mm=args.autopack_bitmap_grid_mm,
            settings_grid_mm=st.autopack.bitmap_grid_mm if st is not None else None,
            beam_width=args.autopack_bitmap_beam_width,
            settings_beam_width=st.autopack.bitmap_beam_width if st is not None else None,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)
    rot_samples = (
        int(args.rotation_samples)
        if args.rotation_samples is not None
        else ORIENTATION_SAMPLES_DEFAULT
    )

    with profiler.stage("load meshes"):
        loaded = load_named_meshes_with_repair(
            args.input_dir,
            args.recursive,
            console,
            repair_options,
            repair_cache_dir,
        )
    if loaded is None:
        return finish_profile(profiler, console, 1)
    paths, names, meshes, repair_reports = loaded
    for path, report in zip(paths, repair_reports, strict=True):
        if repair_options.enabled and report.cache_key is None:
            report.cache_key = repair_cache_key(path, repair_options)
    try:
        pack_workers = _resolve_pack_workers(
            args.autopack_pack_workers,
            st.autopack.pack_workers if st is not None else None,
            len(names),
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    epx, epy, epz = px - 2.0 * edge_margin, py - 2.0 * edge_margin, pz

    # Find best print orientation per mesh and collect raw file dims in one pass.
    # --orient: minimise overhangs (support-optimised) → use oriented AABB dims.
    # default:  maximise scale factor (axis-permutation search).
    # Both paths must be consistent with the footprints passed to _bisect_scale.
    if args.orient_on:

        def _orient_one(
            m: trimesh.Trimesh,
        ) -> tuple[tuple[float, float, float], np.ndarray, tuple[float, float, float]]:
            file_d = aabb_edge_lengths(np.asarray(m.bounds))
            rotation, _, _metrics = find_stable_overhang_rotation(
                m,
                overhang_threshold_deg=args.orient_threshold_deg,
                printer_dims=(epx, epy, epz),
                resin_options=resin_options,
                source_up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )
            t4 = np.eye(4, dtype=np.float64)
            t4[:3, :3] = rotation
            m2 = m.copy()
            m2.apply_transform(t4)
            return file_d, t4, aabb_edge_lengths(np.asarray(m2.bounds))

    else:

        def _orient_one(  # noqa: F811
            m: trimesh.Trimesh,
        ) -> tuple[tuple[float, float, float], np.ndarray, tuple[float, float, float]]:
            file_d = aabb_edge_lengths(np.asarray(m.bounds))
            t4, ext = select_orientation_for_scale(
                m,
                epx,
                epy,
                epz,
                "sorted",
                any_rotation=args.any_rotation,
                maximize=args.maximize,
                random_samples=rot_samples,
                seed=ORIENTATION_SEED_DEFAULT,
                policy=orientation_policy,  # type: ignore[arg-type]
                scale_tolerance=scale_tolerance,
            )
            return file_d, t4, ext

    # Pre-populate trimesh lazy caches sequentially before threading.
    for mesh in meshes:
        _ = mesh.face_normals
        _ = mesh.area_faces

    _n = n_workers(len(meshes))
    if args.verbose:
        console.print(f"[dim]orient: {_n} workers for {len(meshes)} meshes[/dim]")
    with (
        profiler.stage("orientation search"),
        ThreadPoolExecutor(max_workers=_n) as pool,
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Finding orientations…", total=len(meshes))
        _results = []
        for result in profiler.map(pool, "autopack.orientation", _orient_one, meshes):
            _results.append(result)
            progress.advance(ptask)

    dims_list: list[tuple[float, float, float]] = [r[0] for r in _results]
    orient_transforms: list[np.ndarray] = [r[1] for r in _results]
    oriented_dims: list[tuple[float, float, float]] = [r[2] for r in _results]

    with profiler.stage("scale upper bound"):
        s_upper, _ = compute_global_scale((epx, epy, epz), oriented_dims, names, "sorted")
    geometric_upper = s_upper

    def _apply_orientation(m_t4: tuple[trimesh.Trimesh, np.ndarray]) -> trimesh.Trimesh:
        m, t4 = m_t4
        oriented = m.copy()
        oriented.apply_transform(t4)
        oriented.apply_translation(
            [
                -float(oriented.bounds[0][0]),
                -float(oriented.bounds[0][1]),
                -float(oriented.bounds[0][2]),
            ]
        )
        return oriented

    if args.verbose:
        console.print(f"[dim]apply orientation: {_n} workers for {len(meshes)} meshes[/dim]")
    with (
        profiler.stage("apply orientation"),
        ThreadPoolExecutor(max_workers=_n) as pool,
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Applying orientations…", total=len(meshes))
        oriented_meshes: list[trimesh.Trimesh] = []
        for oriented in profiler.map(
            pool,
            "autopack.apply_orientation",
            _apply_orientation,
            zip(meshes, orient_transforms, strict=True),
        ):
            oriented_meshes.append(oriented)
            progress.advance(ptask)

    if args.cleanup:
        with profiler.stage("cleanup"):
            for i, m in enumerate(oriented_meshes):
                cleaned, n_rem = remove_small_components(m)
                if n_rem:
                    oriented_meshes[i] = cleaned
                    console.print(
                        f"[dim]cleanup: {names[i]} — removed {n_rem} tiny component(s)[/dim]"
                    )

    footprint_cache_dir = (
        args.output_dir / "cache" / "footprints"
        if resolve_repair_cache_enabled(args.repair_cache, st) and not args.dry_run
        else None
    )
    with (
        profiler.stage("base footprint computation"),
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Computing footprints…", total=len(oriented_meshes))
        base_shadows = []
        footprint_cache_hits = 0
        footprint_keys: list[str] = []
        for i, m in enumerate(oriented_meshes):
            cache_key = _footprint_cache_key(
                source_path=paths[i],
                repair_cache_key=repair_reports[i].cache_key,
                orientation=orient_transforms[i],
            )
            footprint_keys.append(cache_key)
            cached_shadow = (
                _load_cached_footprint(footprint_cache_dir, cache_key)
                if footprint_cache_dir is not None
                else None
            )
            if cached_shadow is not None:
                footprint_cache_hits += 1
                base_shadows.append(cached_shadow)
                progress.advance(ptask)
                continue
            shadow = profiler.profiled_call("autopack.base_footprint", mesh_to_packing_shadow, m)
            base_shadows.append(shadow)
            if footprint_cache_dir is not None:
                _write_cached_footprint(footprint_cache_dir, cache_key, shadow)
            progress.advance(ptask)
        profile_metadata["footprint_cache"] = {
            "hits": footprint_cache_hits,
            "total": len(oriented_meshes),
        }

    autopack_cache_dir = args.output_dir / "cache" / "autopack"
    cache_settings_result = st.autopack.result_cache if st is not None else True
    cache_settings_attempt = st.autopack.attempt_cache if st is not None else True
    result_cache_enabled = bool(args.autopack_result_cache and cache_settings_result)
    attempt_cache_enabled = bool(args.autopack_attempt_cache and cache_settings_attempt)
    effective_packer = "bitmap" if autopack_packer == "auto" else autopack_packer
    pack_key = _pack_cache_key(
        packer=effective_packer,
        bitmap_options=bitmap_options if effective_packer == "bitmap" else None,
        source_paths=paths,
        repair_cache_keys=[report.cache_key for report in repair_reports],
        footprint_keys=footprint_keys,
        names=names,
        orientation_transforms=orient_transforms,
        bed_w=epx,
        bed_h=epy,
        gap_mm=gap,
        geometric_upper=geometric_upper,
        post_fit_scale=post_fit_scale,
        scale_tolerance=autopack_scale_tolerance,
    )
    autopack_cache_metadata: dict[str, object] = {
        "pack_key": pack_key,
        "result_cache_enabled": result_cache_enabled,
        "attempt_cache_enabled": attempt_cache_enabled,
        "pack_workers": pack_workers,
        "scale_tolerance": autopack_scale_tolerance,
        "packer": effective_packer,
        "bitmap_grid_mm": bitmap_options.grid_mm if effective_packer == "bitmap" else None,
        "bitmap_beam_width": bitmap_options.beam_width if effective_packer == "bitmap" else None,
    }
    profile_metadata["autopack_cache"] = autopack_cache_metadata

    with (
        profiler.stage("autopack scale search"),
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Searching single-plate scale…", total=1)
        if effective_packer == "bitmap":
            search_result = _search_scale_bitmap_cached(
                base_shadows,
                epx,
                epy,
                gap,
                geometric_upper,
                tol=autopack_scale_tolerance,
                bitmap_options=bitmap_options,
                cache_dir=autopack_cache_dir,
                pack_key=pack_key,
                read_result_cache=result_cache_enabled,
                write_result_cache=result_cache_enabled and not args.dry_run,
                write_attempt_cache=attempt_cache_enabled and not args.dry_run,
            )
        else:
            search_result = _search_scale_cached_parallel(
                base_shadows,
                epx,
                epy,
                gap,
                geometric_upper,
                tol=autopack_scale_tolerance,
                pack_workers=pack_workers,
                cache_dir=autopack_cache_dir,
                pack_key=pack_key,
                read_result_cache=result_cache_enabled,
                write_result_cache=result_cache_enabled and not args.dry_run,
                write_attempt_cache=attempt_cache_enabled and not args.dry_run,
            )
        s_pack, plate = search_result.scale, search_result.plate
        autopack_cache_metadata.update(
            {
                "autopack_result_cache_hit": search_result.stats.result_cache_hit,
                "pack_attempt_cache_hits": search_result.stats.attempt_cache_hits,
                "pack_attempts_run": search_result.stats.pack_attempts_run,
                "pack_attempts_failed": search_result.stats.pack_attempts_failed,
                "bitmap_rasterize_s": search_result.stats.bitmap_rasterize_s,
                "bitmap_search_s": search_result.stats.bitmap_search_s,
                "bitmap_candidates_tested": search_result.stats.bitmap_candidates_tested,
                "bitmap_fallback_scans": search_result.stats.bitmap_fallback_scans,
                "exact_validation_s": search_result.stats.exact_validation_s,
            }
        )
        progress.advance(ptask)

    s_best = s_pack * post_fit_scale
    if plate is not None and post_fit_scale > 1.0:
        if effective_packer == "bitmap":
            final_attempt = _pack_bitmap_at_scale_cached(
                base_shadows,
                epx,
                epy,
                gap,
                s_best,
                bitmap_options,
                cache_dir=autopack_cache_dir,
                pack_key=pack_key,
                write_cache=attempt_cache_enabled and not args.dry_run,
                stats=search_result.stats,
            )
        else:
            final_attempt = _pack_at_scale_cached(
                base_shadows,
                epx,
                epy,
                gap,
                s_best,
                cache_dir=autopack_cache_dir,
                pack_key=pack_key,
                write_cache=attempt_cache_enabled and not args.dry_run,
                stats=search_result.stats,
            )
        plate = final_attempt.plate
        autopack_cache_metadata.update(
            {
                "pack_attempt_cache_hits": search_result.stats.attempt_cache_hits,
                "pack_attempts_run": search_result.stats.pack_attempts_run,
                "pack_attempts_failed": search_result.stats.pack_attempts_failed,
                "exact_validation_s": search_result.stats.exact_validation_s,
            }
        )
    elif plate is not None and post_fit_scale != 1.0:
        plate = _scale_plate_dimensions(plate, post_fit_scale)

    if plate is None or s_best <= 0:
        console.print("[red]Cannot fit all parts on one plate at any scale.[/red]")
        return finish_profile(profiler, console, 1)
    if not _plate_has_all_parts_once(plate, len(names)):
        console.print("[red]Internal error: autopack did not place every part exactly once.[/red]")
        return finish_profile(profiler, console, 1)
    if not _validate_plate_geometry(base_shadows, plate, s_best, epx, epy, gap):
        console.print("[red]Internal error: autopack plate failed geometry validation.[/red]")
        return finish_profile(profiler, console, 1)

    profile_metadata["autopack_scale"] = {
        "geometric_upper": geometric_upper,
        "single_plate_max": s_pack,
        "post_fit_scale": post_fit_scale,
        "final_scale": s_best,
        "parts": len(names),
    }

    console.print(f"Optimal scale (all parts on one plate): {s_best:.6f}")
    console.print(f"Parts: {len(names)}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("original (mm)", justify="right")
    table.add_column("scaled (mm)", justify="right")
    for name, d, od in zip(names, dims_list, oriented_dims, strict=True):
        orig = f"{d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f}"
        sd = tuple(x * s_best for x in od)
        scaled_label = f"{sd[0]:.2f} x {sd[1]:.2f} x {sd[2]:.2f}"
        table.add_row(name, orig, scaled_label)
    console.print(table)

    if args.dry_run:
        return finish_profile(profiler, console, 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scaled_meshes: list[trimesh.Trimesh] = []
    for mesh in oriented_meshes:
        scaled = mesh.copy()
        scaled.apply_scale(s_best)
        scaled_meshes.append(scaled)

    with profiler.stage("export"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Exporting plate…", total=1)
        out_3mf = args.output_dir / "autopack_plate.3mf"
        out_json = args.output_dir / "autopack_plate.json"
        export_plate_3mf(scaled_meshes, plate, out_3mf, names=list(names), out_manifest=out_json)
        progress.advance(ptask)
    console.print(f"Wrote {out_3mf}")
    console.print(f"Wrote {out_json}")
    output_files = [out_3mf, out_json]

    transform_parts: list[dict] = []
    scale_matrix = uniform_scale_matrix(s_best)
    for rect in plate.rects:
        part_index = rect.part_index
        source_mesh = meshes[part_index]
        orientation = orient_transforms[part_index]
        oriented_bounds = transform_bounds(
            np.asarray(source_mesh.bounds, dtype=np.float64), orientation
        )
        normalize = translation_matrix(-oriented_bounds[0])
        source_to_oriented = normalize @ orientation
        source_to_scaled = scale_matrix @ source_to_oriented
        placement_transform, placement_steps = placement_transform_for_mesh(
            scaled_meshes[part_index], rect
        )
        source_to_export = placement_transform @ source_to_scaled
        transform_parts.append(
            transform_entry(
                index=part_index,
                plate_index=0,
                plate_file=out_3mf,
                source_path=paths[part_index],
                source_name=names[part_index],
                output_name=names[part_index],
                output_file=out_3mf,
                source_bounds_mm=mesh_bounds(source_mesh),
                final_bounds_mm=bounds_to_list(
                    transform_bounds(
                        np.asarray(source_mesh.bounds, dtype=np.float64), source_to_export
                    )
                ),
                source_to_export_matrix=source_to_export,
                steps=[
                    *(
                        [repair_report_step(repair_reports[part_index])]
                        if repair_reports[part_index].enabled
                        else []
                    ),
                    transform_step("autopack_orientation", matrix=orientation),
                    transform_step("normalize_to_origin", matrix=normalize),
                    transform_step(
                        "single_plate_scale",
                        matrix=scale_matrix,
                        params={"scale_factor": s_best},
                    ),
                    *placement_steps,
                ],
                scale_factor=s_best,
                plate_x_mm=rect.x,
                plate_y_mm=rect.y,
                rotation_deg=rect.rotation_deg,
            )
        )
    write_transform_log(
        args.output_dir / "transforms.json",
        command="autopack",
        output_files=output_files,
        parts=transform_parts,
        metadata={"scale": profile_metadata["autopack_scale"]},
    )
    write_command_repair_report(
        args.output_dir,
        command="autopack",
        reports=repair_reports,
        dry_run=args.dry_run,
    )
    return finish_profile(profiler, console, 0)
