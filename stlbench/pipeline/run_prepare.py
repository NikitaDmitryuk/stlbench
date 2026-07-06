"""Full preparation pipeline: scale → orient → layout.

Order rationale
---------------
1. **Scale first** – maximises the model size by finding the global scale factor
   that fits every part inside the build volume (free-orientation search, same
   as ``scale --orientation free``).
2. **Orient for minimum supports** – rotates each *already-scaled* part to
   minimise overhang area, subject to the constraint that the part still fits
   inside the build volume in the new orientation.
3. **Layout** – packs the oriented parts onto the minimum number of plates as
   evenly as possible.

Intermediate results
--------------------
When ``resume=True`` the pipeline checks ``output_dir/cache/meta.json`` for a
cached orient result.  If the input files' SHA-256 hashes match, steps 1–2 are
skipped and the previously oriented meshes are loaded from
``output_dir/cache/<stem>_oriented.stl``.  This lets you resume after a crash
without re-running the expensive orientation search.
"""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import pickle
import queue
import resource
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table
from scipy.spatial import QhullError
from shapely import affinity
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.fit import compute_global_scale
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.mesh_repair import RepairOptions, RepairReport, repair_report_step
from stlbench.core.overhang import (
    ResinOrientationOptions,
    apply_min_overhang_orientation,
    find_stable_overhang_rotation,
    find_stable_overhang_rotation_adaptive,
    find_stable_overhang_rotation_legacy,
    overhang_score,
    rotation_to_transform4,
)
from stlbench.export.plate import clear_mesh_cache, export_plate_3mf_lazy
from stlbench.export.transform_log import (
    bounds_to_list,
    placement_transform_for_bounds,
    transform_bounds,
    transform_entry,
    transform_step,
    translation_matrix,
    uniform_scale_matrix,
    write_transform_log,
)
from stlbench.packing.bitmap_pack import BitmapPackOptions, pack_polygons_bitmap_single_plate
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.polygon_footprint import mesh_to_packing_shadow
from stlbench.packing.polygon_pack import pack_polygons_on_plates
from stlbench.packing.rectpack_plate import PackedPlate, PackedRect
from stlbench.pipeline.common import (
    finish_profile,
    load_mesh_with_repair,
    repair_cache_dir_for_output,
    resolve_edge_margin,
    resolve_gap,
    resolve_printer,
    resolve_repair_cache_enabled,
    resolve_repair_options,
    resolve_resin_orientation_options,
    resolve_settings,
    write_command_repair_report,
)
from stlbench.pipeline.mesh_io import (
    SUPPORTED_EXTENSIONS,
    collect_mesh_paths,
    load_mesh,
)
from stlbench.pipeline.progress import make_progress
from stlbench.pipeline.resource_planner import (
    DEFAULT_MEMORY_BUDGET_FRACTION,
    choose_export_workers,
    make_prepare_worker_plan,
)
from stlbench.profiling import ProfileOptions, make_profiler

_IDENTITY3 = np.eye(3, dtype=np.float64)


@dataclass
class PrepareRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    post_fit_scale: float | None
    method: str | None
    overhang_threshold_deg: float
    n_orient_candidates: int
    dry_run: bool
    recursive: bool
    verbose: bool = False
    grid_step_mm: float = 2.0
    resume: bool = False
    cleanup: bool = False
    any_rotation: bool = False
    workers: str = "auto"
    profile_options: ProfileOptions | None = None
    edge_margin_mm: float | None = None
    resin_balance: str | None = None
    repair: bool = False
    repair_cache: bool = True
    packer: str | None = None
    bitmap_grid_mm: float | None = None
    bitmap_beam_width: int | None = None
    packing_result_cache: bool = True
    footprint_cache: bool = True
    progress: bool = True
    export_compression: str = "default"
    orientation_strategy: str = "auto"
    orientation_quality: str = "adaptive"


@dataclass(frozen=True)
class PreparedMeshRef:
    index: int
    name: str
    source_path: Path
    cache_path: Path
    dims: tuple[float, float, float]
    cache_bounds: np.ndarray | None = None
    source_bounds: np.ndarray | None = None
    source_to_cache_matrix: np.ndarray | None = None
    transform_steps: tuple[dict[str, object], ...] = ()
    source_transform_available: bool = True


@dataclass(frozen=True)
class _ScaleOrientJob:
    index: int
    path: Path
    px: float
    py: float
    pz: float
    method: str
    any_rotation: bool
    repair_options: RepairOptions
    name: str
    repair_cache_dir: Path | None
    orientation_strategy: str


@dataclass(frozen=True)
class _PrepareCacheJob:
    index: int
    path: Path
    name: str
    cache_dir: Path
    scale_transform: np.ndarray
    source_up: np.ndarray
    scale: float
    overhang_threshold_deg: float
    n_orient_candidates: int
    printer_xyz: tuple[float, float, float]
    cleanup: bool
    resin_options: ResinOrientationOptions
    repair_options: RepairOptions
    repair_cache_dir: Path | None
    orientation_strategy: str
    orientation_quality: str


@dataclass(frozen=True)
class _ExportPlateJob:
    plate: PackedPlate
    refs: tuple[PreparedMeshRef, ...]
    output_dir: Path
    names: tuple[str, ...]
    compression_mode: str


@dataclass(frozen=True)
class _FootprintJob:
    ref: PreparedMeshRef


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_META_NAME = "meta.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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
    return {"index": plate.index, "rects": [_rect_to_json(rect) for rect in plate.rects]}


def _plate_from_json(payload: object) -> PackedPlate | None:
    if not isinstance(payload, dict):
        return None
    raw_rects = payload.get("rects")
    if not isinstance(raw_rects, list):
        return None
    rects: list[PackedRect] = []
    for item in raw_rects:
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


def _footprint_cache_key(ref: PreparedMeshRef) -> str:
    stat = ref.cache_path.stat()
    return _json_hash(
        {
            "version": 1,
            "path": str(ref.cache_path),
            "size": stat.st_size,
            "sha256": _file_sha256(ref.cache_path),
            "index": ref.index,
        }
    )


def _load_cached_footprint(cache_dir: Path | None, key: str) -> BaseGeometry | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.pkl"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            loaded = pickle.load(fh)  # noqa: S301 - trusted local cache keyed by source content.
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    return loaded if isinstance(loaded, BaseGeometry) else None


def _write_cached_footprint(cache_dir: Path | None, key: str, geometry: BaseGeometry) -> None:
    if cache_dir is None:
        return
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


def _prepare_packer_version(packer: str) -> str:
    return "prepare_bitmap_v1" if packer == "bitmap" else "prepare_exact_v1"


def _packing_cache_key(
    *,
    footprint_keys: list[str],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    edge_margin_mm: float,
    part_heights: list[float],
    packer: str,
    bitmap_options: BitmapPackOptions | None,
    grid_step_mm: float,
) -> str:
    return _json_hash(
        {
            "version": 1,
            "footprint_keys": footprint_keys,
            "bed_w": round(float(bed_w), 9),
            "bed_h": round(float(bed_h), 9),
            "gap_mm": round(float(gap_mm), 9),
            "edge_margin_mm": round(float(edge_margin_mm), 9),
            "part_heights": (
                [round(float(h), 6) for h in part_heights] if packer == "exact" else None
            ),
            "packer": packer,
            "packer_version": _prepare_packer_version(packer),
            "bitmap_grid_mm": bitmap_options.grid_mm if bitmap_options is not None else None,
            "bitmap_beam_width": bitmap_options.beam_width if bitmap_options is not None else None,
            "grid_step_mm": round(float(grid_step_mm), 9),
        }
    )


def _load_packing_result(cache_dir: Path | None, key: str, packer: str) -> list[PackedPlate] | None:
    if cache_dir is None:
        return None
    path = cache_dir / "results" / key / "result.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("packer_version") != _prepare_packer_version(packer):
            return None
        raw_plates = payload.get("plates")
        if not isinstance(raw_plates, list):
            return None
        plates = [_plate_from_json(item) for item in raw_plates]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if any(plate is None for plate in plates):
        return None
    return [plate for plate in plates if plate is not None]


def _write_packing_result(
    cache_dir: Path | None,
    key: str,
    packer: str,
    plates: list[PackedPlate],
) -> None:
    if cache_dir is None:
        return
    _write_json_atomic(
        cache_dir / "results" / key / "result.json",
        {
            "packer_version": _prepare_packer_version(packer),
            "plates": [_plate_to_json(plate) for plate in plates],
        },
    )


def _renumber_plates(plates: list[PackedPlate]) -> list[PackedPlate]:
    return [
        PackedPlate(index=i, rects=tuple(plate.rects))
        for i, plate in enumerate(plates)
        if plate.rects
    ]


def _offset_plate(plate: PackedPlate, dx: float, dy: float, index: int) -> PackedPlate:
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12 and plate.index == index:
        return plate
    return PackedPlate(
        index=index,
        rects=tuple(
            PackedRect(
                part_index=rect.part_index,
                x=rect.x + dx,
                y=rect.y + dy,
                width=rect.width,
                height=rect.height,
                rotation_deg=rect.rotation_deg,
            )
            for rect in plate.rects
        ),
    )


def _validate_plate_geometry(
    polygons: list[BaseGeometry],
    plate: PackedPlate,
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    *,
    tolerance: float = 1e-3,
) -> bool:
    bed = shapely_box(-tolerance, -tolerance, bed_w + tolerance, bed_h + tolerance)
    placed: list[BaseGeometry] = []
    for rect in plate.rects:
        if rect.part_index < 0 or rect.part_index >= len(polygons):
            return False
        poly = affinity.rotate(polygons[rect.part_index], rect.rotation_deg, origin=(0.0, 0.0))
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


def _validate_layout_geometry(
    polygons: list[BaseGeometry],
    plates: list[PackedPlate],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
) -> bool:
    placed_indices = sorted(rect.part_index for plate in plates for rect in plate.rects)
    if placed_indices != list(range(len(polygons))):
        return False
    return all(_validate_plate_geometry(polygons, plate, bed_w, bed_h, gap_mm) for plate in plates)


def _pack_bitmap_multi_plate(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    edge_margin_mm: float,
    bitmap_options: BitmapPackOptions,
    metadata: dict[str, object],
    max_plates: int = 64,
) -> list[PackedPlate]:
    effective_w = bed_w - 2.0 * edge_margin_mm
    effective_h = bed_h - 2.0 * edge_margin_mm
    if effective_w <= 0 or effective_h <= 0:
        raise ValueError(
            f"edge_margin_mm={edge_margin_mm!r} leaves no printable bed area "
            f"inside {bed_w:.1f}×{bed_h:.1f} mm."
        )
    remaining = sorted(range(len(polygons)), key=lambda i: float(polygons[i].area), reverse=True)
    plates: list[PackedPlate] = []
    attempts: list[dict[str, object]] = []
    total_candidates = 0
    total_rasterize = 0.0
    total_search = 0.0
    while remaining:
        if len(plates) >= max_plates:
            raise RuntimeError(f"Exceeded max_plates={max_plates}; not all parts could be placed.")
        subset_polygons = [polygons[i] for i in remaining]
        subset_to_global = list(remaining)
        best_plate: PackedPlate | None = None
        best_count = 0
        for count in range(len(subset_polygons), 0, -1):
            candidate_polygons = subset_polygons[:count]
            mapping = tuple(subset_to_global)
            plate_index = len(plates)

            def _validator(
                local_plate: PackedPlate,
                mapping: tuple[int, ...] = mapping,
                plate_index: int = plate_index,
            ) -> bool:
                global_rects = tuple(
                    PackedRect(
                        part_index=mapping[rect.part_index],
                        x=rect.x,
                        y=rect.y,
                        width=rect.width,
                        height=rect.height,
                        rotation_deg=rect.rotation_deg,
                    )
                    for rect in local_plate.rects
                )
                global_plate = PackedPlate(index=plate_index, rects=global_rects)
                return _validate_plate_geometry(
                    polygons, global_plate, effective_w, effective_h, gap_mm
                )

            result = pack_polygons_bitmap_single_plate(
                candidate_polygons,
                effective_w,
                effective_h,
                gap_mm,
                scale=1.0,
                options=bitmap_options,
                validator=_validator,
            )
            total_candidates += result.stats.candidates_tested
            total_rasterize += result.stats.rasterize_s
            total_search += result.stats.search_s
            attempts.append(
                {
                    "plate": len(plates),
                    "try_parts": count,
                    "ok": result.plate is not None,
                    "candidates_tested": result.stats.candidates_tested,
                }
            )
            if result.plate is None:
                continue
            global_rects = tuple(
                PackedRect(
                    part_index=subset_to_global[rect.part_index],
                    x=rect.x,
                    y=rect.y,
                    width=rect.width,
                    height=rect.height,
                    rotation_deg=rect.rotation_deg,
                )
                for rect in result.plate.rects
            )
            best_plate = PackedPlate(index=len(plates), rects=global_rects)
            best_count = count
            break
        if best_plate is None or best_count <= 0:
            raise RuntimeError("bitmap packer could not place any remaining part on a new plate.")
        plates.append(_offset_plate(best_plate, edge_margin_mm, edge_margin_mm, len(plates)))
        placed = set(remaining[:best_count])
        remaining = [idx for idx in remaining if idx not in placed]
    metadata.update(
        {
            "strategy": "bitmap",
            "baseline_plates": len(plates),
            "final_plates": len(plates),
            "edge_margin_mm": edge_margin_mm,
            "effective_bed_mm": [effective_w, effective_h],
            "bitmap_grid_mm": bitmap_options.grid_mm,
            "bitmap_beam_width": bitmap_options.beam_width,
            "bitmap_candidates_tested": total_candidates,
            "bitmap_rasterize_s": total_rasterize,
            "bitmap_search_s": total_search,
            "attempts": attempts,
        }
    )
    return _renumber_plates(plates)


def _resolve_prepare_packer(value: str | None, settings_value: str | None) -> str:
    out = value or settings_value or "auto"
    if out not in {"auto", "bitmap", "exact"}:
        raise ValueError("--packer must be auto, bitmap, or exact.")
    return "exact" if out == "auto" else out


def _resolve_prepare_bitmap_options(
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
        raise ValueError("--bitmap-grid-mm must be > 0.")
    if resolved_beam < 1:
        raise ValueError("--bitmap-beam-width must be >= 1.")
    return BitmapPackOptions(grid_mm=resolved_grid, beam_width=resolved_beam)


def _cache_mesh_name(index: int, name: str) -> str:
    return f"{index:03d}_{Path(name).stem}_oriented.stl"


def _write_orient_cache_meta(
    cache_dir: Path,
    paths: list[Path],
    names: list[str],
    mesh_files: list[str],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "input_files": {str(p): _file_sha256(p) for p in paths},
        "names": names,
        "mesh_files": mesh_files,
    }
    (cache_dir / _CACHE_META_NAME).write_text(json.dumps(meta, indent=2))


def _load_orient_cache(
    cache_dir: Path,
    paths: list[Path],
    console: Console,
) -> tuple[list[str], list[PreparedMeshRef]] | None:
    meta_path = cache_dir / _CACHE_META_NAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    stored: dict[str, str] = meta.get("input_files", {})
    if set(stored.keys()) != {str(p) for p in paths}:
        return None
    for path in paths:
        if _file_sha256(path) != stored.get(str(path), ""):
            console.print("[dim]Cache miss: input file changed, re-running orient.[/dim]")
            return None

    names: list[str] = meta.get("names", [])
    mesh_files: list[str] = meta.get("mesh_files", [])
    refs: list[PreparedMeshRef] = []
    for index, (name, source_path, fname) in enumerate(zip(names, paths, mesh_files, strict=True)):
        stl_path = cache_dir / fname
        if not stl_path.exists():
            return None
        mesh = load_mesh(stl_path)
        bounds = np.asarray(mesh.bounds)
        dims = tuple(float(v) for v in bounds[1] - bounds[0])
        clear_mesh_cache(mesh)
        refs.append(
            PreparedMeshRef(
                index=index,
                name=name,
                source_path=source_path,
                cache_path=stl_path,
                dims=dims,  # type: ignore[arg-type]
                cache_bounds=bounds,
                source_transform_available=False,
            )
        )
        del mesh
    gc.collect()
    return names, refs


def _rotation_fits_vertices(
    vertices: np.ndarray,
    rotation: np.ndarray,
    printer_xyz: tuple[float, float, float],
    *,
    chunk_size: int = 250_000,
) -> bool:
    px, py, pz = printer_xyz
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, len(vertices), chunk_size):
        chunk = np.asarray(vertices[start : start + chunk_size], dtype=np.float64) @ rotation.T
        lo = np.minimum(lo, chunk.min(axis=0))
        hi = np.maximum(hi, chunk.max(axis=0))
    d = hi - lo
    xy_lo = min(float(d[0]), float(d[1]))
    xy_hi = max(float(d[0]), float(d[1]))
    bed_lo = min(px, py)
    bed_hi = max(px, py)
    return bool(d[2] <= pz + 1e-6 and xy_lo <= bed_lo + 1e-6 and xy_hi <= bed_hi + 1e-6)


def _find_support_orientation_for_prepare(
    mesh: trimesh.Trimesh,
    job: _PrepareCacheJob,
    timings: dict[str, Any],
) -> tuple[np.ndarray, float, Any]:
    support_fn = (
        find_stable_overhang_rotation_legacy
        if job.orientation_strategy == "legacy"
        else find_stable_overhang_rotation
    )
    if job.orientation_strategy != "legacy" and job.orientation_quality == "adaptive":
        support_fn = find_stable_overhang_rotation_adaptive  # type: ignore[assignment]
    try:
        result = support_fn(
            mesh,
            overhang_threshold_deg=job.overhang_threshold_deg,
            n_candidates=job.n_orient_candidates,
            printer_dims=job.printer_xyz,
            resin_options=job.resin_options,
            source_up=job.source_up,
        )
        if len(result) == 4:
            rotation, score_after, metrics, diagnostics = result
            timings.update(diagnostics)
        else:
            rotation, score_after, metrics = result
        metric_values = [
            metrics.overhang_score,
            metrics.height_mm,
            metrics.center_z_ratio,
            metrics.long_axis_z,
            metrics.long_axis_angle_from_bed_deg,
            metrics.pca_aspect,
            metrics.pca_line_ratio,
            metrics.stability_score,
            metrics.support_score_delta,
            metrics.xy_footprint_area_mm2,
            metrics.support_contact_proxy,
            metrics.surface_damage_proxy,
            metrics.salient_down_area_ratio,
            metrics.flat_safe_down_area_ratio,
            metrics.source_up_dot_build_up,
            metrics.upside_down_penalty,
            metrics.angle_band_penalty,
            metrics.vertical_penalty,
            metrics.horizontal_penalty,
        ]
        if not all(np.isfinite(v) for v in metric_values):
            raise ValueError("non-finite orientation metrics")
        if not _rotation_fits_vertices(
            np.asarray(mesh.vertices, dtype=np.float64), rotation, job.printer_xyz
        ):
            raise ValueError("selected orientation does not fit printer bounds")
        return rotation, score_after, metrics
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, QhullError) as exc:
        if job.orientation_strategy == "legacy":
            raise
        timings["fallback"] = True
        timings["fallback_reason"] = str(exc)
        timings["strategy_used"] = "legacy"
        timings["adaptive_enabled"] = False
        timings["adaptive_reason"] = "legacy_fallback"
        rotation, score_after, metrics = find_stable_overhang_rotation_legacy(
            mesh,
            overhang_threshold_deg=job.overhang_threshold_deg,
            n_candidates=job.n_orient_candidates,
            printer_dims=job.printer_xyz,
            resin_options=job.resin_options,
            source_up=job.source_up,
        )
        return rotation, score_after, metrics


def _scale_orientation_worker(
    job: _ScaleOrientJob,
) -> tuple[
    int,
    np.ndarray,
    tuple[float, float, float],
    bool,
    RepairReport,
    float,
    dict[str, float | str],
]:
    start = time.perf_counter()
    timings: dict[str, float | str] = {"part": job.name}
    load_start = time.perf_counter()
    m, repair_report = load_mesh_with_repair(
        job.path,
        job.repair_options,
        source_name=job.name,
        repair_cache_dir=job.repair_cache_dir,
    )
    timings["load_repair_s"] = time.perf_counter() - load_start
    has_multiple = False
    try:
        transform, dims = _compute_scale_orientation_loaded(m, job, timings)
    finally:
        clear_mesh_cache(m)
        del m
        gc.collect()
    timings["total_s"] = time.perf_counter() - start
    timings["strategy"] = job.orientation_strategy
    return (
        job.index,
        transform,
        dims,
        has_multiple,
        repair_report,
        time.perf_counter() - start,
        timings,
    )


def _compute_scale_orientation_loaded(
    mesh: trimesh.Trimesh,
    job: _ScaleOrientJob,
    timings: dict[str, float | str],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    search_start = time.perf_counter()
    transform, dims = select_orientation_for_scale(
        mesh,
        job.px,
        job.py,
        job.pz,
        job.method,  # type: ignore[arg-type]
        any_rotation=job.any_rotation,
        random_samples=ORIENTATION_SAMPLES_DEFAULT,
        seed=ORIENTATION_SEED_DEFAULT,
        compute_printability_metrics=False,
        use_fast_z=job.orientation_strategy != "legacy",
    )
    timings["search_s"] = time.perf_counter() - search_start
    return transform, dims


def _prepare_cache_worker(
    job: _PrepareCacheJob,
) -> tuple[
    int,
    float,
    float,
    float,
    PreparedMeshRef,
    dict[str, float | str],
    float,
    dict[str, float | str | bool],
]:
    start = time.perf_counter()
    timings: dict[str, float | str | bool] = {
        "part": job.name,
        "strategy_requested": job.orientation_strategy,
        "strategy_used": job.orientation_strategy,
        "orientation_quality": job.orientation_quality,
        "fallback": False,
        "adaptive_enabled": False,
        "adaptive_reason": "",
        "adaptive_accepted": False,
        "candidate_count_default": 0,
        "candidate_count_adaptive": 0,
    }
    load_start = time.perf_counter()
    mesh, repair_report = load_mesh_with_repair(
        job.path,
        job.repair_options,
        source_name=job.name,
        repair_cache_dir=job.repair_cache_dir,
    )
    timings["load_repair_s"] = time.perf_counter() - load_start
    try:
        source_bounds = np.asarray(mesh.bounds, dtype=np.float64).copy()
        source_to_cache = np.eye(4, dtype=np.float64)
        steps: list[dict[str, object]] = (
            [repair_report_step(repair_report)] if repair_report.enabled else []
        )
        scale_start = time.perf_counter()
        mesh.apply_transform(job.scale_transform)
        source_to_cache = job.scale_transform @ source_to_cache
        steps.append(transform_step("scale_orientation", matrix=job.scale_transform))
        scale_matrix = uniform_scale_matrix(job.scale)
        mesh.apply_scale(job.scale)
        source_to_cache = scale_matrix @ source_to_cache
        steps.append(
            transform_step("global_scale", matrix=scale_matrix, params={"scale_factor": job.scale})
        )
        normalize_before_orient = translation_matrix(
            [0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])]
        )
        mesh.apply_translation([0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])])
        source_to_cache = normalize_before_orient @ source_to_cache
        steps.append(transform_step("z_normalize", matrix=normalize_before_orient))
        timings["scale_transform_s"] = time.perf_counter() - scale_start
        support_start = time.perf_counter()
        sb = overhang_score(mesh, _IDENTITY3, job.overhang_threshold_deg)
        rotation, sa, metrics = _find_support_orientation_for_prepare(mesh, job, timings)
        timings["support_search_s"] = time.perf_counter() - support_start
        metrics_payload: dict[str, float | str] = {
            "overhang_score": metrics.overhang_score,
            "selected_height_mm": metrics.height_mm,
            "center_z_ratio": metrics.center_z_ratio,
            "long_axis_angle_from_bed_deg": metrics.long_axis_angle_from_bed_deg,
            "long_axis_z": metrics.long_axis_z,
            "pca_aspect": metrics.pca_aspect,
            "pca_line_ratio": metrics.pca_line_ratio,
            "stability_score": metrics.stability_score,
            "support_score_delta": metrics.support_score_delta,
            "xy_footprint_area_mm2": metrics.xy_footprint_area_mm2,
            "support_contact_proxy": metrics.support_contact_proxy,
            "surface_damage_proxy": metrics.surface_damage_proxy,
            "salient_down_area_ratio": metrics.salient_down_area_ratio,
            "flat_safe_down_area_ratio": metrics.flat_safe_down_area_ratio,
            "source_up_dot_build_up": metrics.source_up_dot_build_up,
            "upside_down_penalty": metrics.upside_down_penalty,
            "angle_band_penalty": metrics.angle_band_penalty,
            "vertical_penalty": metrics.vertical_penalty,
            "horizontal_penalty": metrics.horizontal_penalty,
            "selection_reason": metrics.selection_reason,
            "orientation_strategy": str(timings["strategy_used"]),
        }
        apply_start = time.perf_counter()
        pct = (sb - sa) / max(abs(sb), 1.0) * 100.0
        rotation4 = rotation_to_transform4(rotation)
        rotated_bounds = transform_bounds(np.asarray(mesh.bounds, dtype=np.float64), rotation4)
        normalize_after_orient = translation_matrix([0.0, 0.0, -float(rotated_bounds[0, 2])])
        oriented = apply_min_overhang_orientation(mesh, rotation)
        source_to_cache = normalize_after_orient @ rotation4 @ source_to_cache
        steps.extend(
            [
                transform_step("support_orientation", matrix=rotation4),
                transform_step("z_normalize", matrix=normalize_after_orient),
            ]
        )
        if job.cleanup:
            oriented, _n_removed = remove_small_components(oriented)
            if _n_removed:
                steps.append(
                    transform_step(
                        "cleanup",
                        params={"removed_components": _n_removed},
                        available=False,
                    )
                )
        bounds = np.asarray(oriented.bounds)
        dims_raw = bounds[1] - bounds[0]
        dims = (float(dims_raw[0]), float(dims_raw[1]), float(dims_raw[2]))
        out = job.cache_dir / _cache_mesh_name(job.index, job.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        oriented.export(str(out))
        clear_mesh_cache(oriented)
        timings["apply_export_cache_s"] = time.perf_counter() - apply_start
        ref = PreparedMeshRef(
            index=job.index,
            name=job.name,
            source_path=job.path,
            cache_path=out,
            dims=dims,
            cache_bounds=bounds.copy(),
            source_bounds=source_bounds,
            source_to_cache_matrix=source_to_cache,
            transform_steps=tuple(steps),
        )
        timings["total_s"] = time.perf_counter() - start
        return (
            job.index,
            sb,
            sa,
            pct,
            ref,
            metrics_payload,
            time.perf_counter() - start,
            timings,
        )
    finally:
        clear_mesh_cache(mesh)
        del mesh
        gc.collect()


def _prepare_cache_loaded_worker(
    job: _PrepareCacheJob,
    mesh: trimesh.Trimesh,
    repair_report: RepairReport,
    *,
    load_repair_s: float,
    retained: bool,
) -> tuple[
    int,
    float,
    float,
    float,
    PreparedMeshRef,
    dict[str, float | str],
    float,
    dict[str, float | str | bool],
]:
    start = time.perf_counter()
    timings: dict[str, float | str | bool] = {
        "part": job.name,
        "strategy_requested": job.orientation_strategy,
        "strategy_used": job.orientation_strategy,
        "orientation_quality": job.orientation_quality,
        "fallback": False,
        "adaptive_enabled": False,
        "adaptive_reason": "",
        "adaptive_accepted": False,
        "candidate_count_default": 0,
        "candidate_count_adaptive": 0,
        "retained_mesh": retained,
        "load_repair_s": load_repair_s,
    }
    try:
        source_bounds = np.asarray(mesh.bounds, dtype=np.float64).copy()
        source_to_cache = np.eye(4, dtype=np.float64)
        steps: list[dict[str, object]] = (
            [repair_report_step(repair_report)] if repair_report.enabled else []
        )
        scale_start = time.perf_counter()
        mesh.apply_transform(job.scale_transform)
        source_to_cache = job.scale_transform @ source_to_cache
        steps.append(transform_step("scale_orientation", matrix=job.scale_transform))
        scale_matrix = uniform_scale_matrix(job.scale)
        mesh.apply_scale(job.scale)
        source_to_cache = scale_matrix @ source_to_cache
        steps.append(
            transform_step("global_scale", matrix=scale_matrix, params={"scale_factor": job.scale})
        )
        normalize_before_orient = translation_matrix(
            [0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])]
        )
        mesh.apply_translation([0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])])
        source_to_cache = normalize_before_orient @ source_to_cache
        steps.append(transform_step("z_normalize", matrix=normalize_before_orient))
        timings["scale_transform_s"] = time.perf_counter() - scale_start
        support_start = time.perf_counter()
        sb = overhang_score(mesh, _IDENTITY3, job.overhang_threshold_deg)
        rotation, sa, metrics = _find_support_orientation_for_prepare(mesh, job, timings)
        timings["support_search_s"] = time.perf_counter() - support_start
        metrics_payload: dict[str, float | str] = {
            "overhang_score": metrics.overhang_score,
            "selected_height_mm": metrics.height_mm,
            "center_z_ratio": metrics.center_z_ratio,
            "long_axis_angle_from_bed_deg": metrics.long_axis_angle_from_bed_deg,
            "long_axis_z": metrics.long_axis_z,
            "pca_aspect": metrics.pca_aspect,
            "pca_line_ratio": metrics.pca_line_ratio,
            "stability_score": metrics.stability_score,
            "support_score_delta": metrics.support_score_delta,
            "xy_footprint_area_mm2": metrics.xy_footprint_area_mm2,
            "support_contact_proxy": metrics.support_contact_proxy,
            "surface_damage_proxy": metrics.surface_damage_proxy,
            "salient_down_area_ratio": metrics.salient_down_area_ratio,
            "flat_safe_down_area_ratio": metrics.flat_safe_down_area_ratio,
            "source_up_dot_build_up": metrics.source_up_dot_build_up,
            "upside_down_penalty": metrics.upside_down_penalty,
            "angle_band_penalty": metrics.angle_band_penalty,
            "vertical_penalty": metrics.vertical_penalty,
            "horizontal_penalty": metrics.horizontal_penalty,
            "selection_reason": metrics.selection_reason,
            "orientation_strategy": str(timings["strategy_used"]),
        }
        apply_start = time.perf_counter()
        pct = (sb - sa) / max(abs(sb), 1.0) * 100.0
        rotation4 = rotation_to_transform4(rotation)
        rotated_bounds = transform_bounds(np.asarray(mesh.bounds, dtype=np.float64), rotation4)
        normalize_after_orient = translation_matrix([0.0, 0.0, -float(rotated_bounds[0, 2])])
        oriented = apply_min_overhang_orientation(mesh, rotation)
        source_to_cache = normalize_after_orient @ rotation4 @ source_to_cache
        steps.extend(
            [
                transform_step("support_orientation", matrix=rotation4),
                transform_step("z_normalize", matrix=normalize_after_orient),
            ]
        )
        if job.cleanup:
            oriented, _n_removed = remove_small_components(oriented)
            if _n_removed:
                steps.append(
                    transform_step(
                        "cleanup",
                        params={"removed_components": _n_removed},
                        available=False,
                    )
                )
        bounds = np.asarray(oriented.bounds)
        dims_raw = bounds[1] - bounds[0]
        dims = (float(dims_raw[0]), float(dims_raw[1]), float(dims_raw[2]))
        out = job.cache_dir / _cache_mesh_name(job.index, job.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        oriented.export(str(out))
        clear_mesh_cache(oriented)
        timings["apply_export_cache_s"] = time.perf_counter() - apply_start
        ref = PreparedMeshRef(
            index=job.index,
            name=job.name,
            source_path=job.path,
            cache_path=out,
            dims=dims,
            cache_bounds=bounds.copy(),
            source_bounds=source_bounds,
            source_to_cache_matrix=source_to_cache,
            transform_steps=tuple(steps),
        )
        timings["total_s"] = time.perf_counter() - start
        return (
            job.index,
            sb,
            sa,
            pct,
            ref,
            metrics_payload,
            time.perf_counter() - start,
            timings,
        )
    finally:
        clear_mesh_cache(mesh)
        del mesh
        gc.collect()


def _orientation_actor_main(command_q: Any, result_q: Any) -> None:
    retained: dict[int, tuple[trimesh.Trimesh, RepairReport]] = {}
    pid = mp.current_process().pid
    while True:
        command = command_q.get()
        kind = command[0]
        if kind == "stop":
            for mesh, _report in retained.values():
                clear_mesh_cache(mesh)
            retained.clear()
            gc.collect()
            result_q.put(("stopped", -1, {"pid": pid, "max_rss_mb": _process_max_rss_mb()}))
            return
        try:
            if kind == "scale":
                _kind, job, retain_mesh = command
                start = time.perf_counter()
                timings: dict[str, float | str | bool] = {
                    "part": job.name,
                    "retained_mesh": bool(retain_mesh),
                }
                load_start = time.perf_counter()
                mesh, repair_report = load_mesh_with_repair(
                    job.path,
                    job.repair_options,
                    source_name=job.name,
                    repair_cache_dir=job.repair_cache_dir,
                )
                timings["load_repair_s"] = time.perf_counter() - load_start
                transform, dims = _compute_scale_orientation_loaded(mesh, job, timings)
                if retain_mesh:
                    retained[job.index] = (mesh, repair_report)
                else:
                    clear_mesh_cache(mesh)
                    del mesh
                    gc.collect()
                timings["total_s"] = time.perf_counter() - start
                timings["strategy"] = job.orientation_strategy
                timings["pid"] = str(pid)
                result_q.put(
                    (
                        "scale",
                        job.index,
                        (
                            transform,
                            dims,
                            False,
                            repair_report,
                            time.perf_counter() - start,
                            timings,
                            _process_max_rss_mb(),
                        ),
                    )
                )
            elif kind == "orient":
                _kind, job = command
                retained_payload = retained.pop(job.index, None)
                if retained_payload is not None:
                    mesh, repair_report = retained_payload
                    result = _prepare_cache_loaded_worker(
                        job,
                        mesh,
                        repair_report,
                        load_repair_s=0.0,
                        retained=True,
                    )
                else:
                    load_start = time.perf_counter()
                    mesh, repair_report = load_mesh_with_repair(
                        job.path,
                        job.repair_options,
                        source_name=job.name,
                        repair_cache_dir=job.repair_cache_dir,
                    )
                    result = _prepare_cache_loaded_worker(
                        job,
                        mesh,
                        repair_report,
                        load_repair_s=time.perf_counter() - load_start,
                        retained=False,
                    )
                result_q.put(("orient", job.index, (*result, _process_max_rss_mb())))
            else:
                raise ValueError(f"unknown orientation actor command: {kind!r}")
        except BaseException as exc:  # noqa: BLE001 - propagate child failure to parent
            result_q.put(
                (
                    "error",
                    getattr(command[1], "index", -1) if len(command) > 1 else -1,
                    {
                        "pid": pid,
                        "error": repr(exc),
                        "max_rss_mb": _process_max_rss_mb(),
                    },
                )
            )


def _stop_orientation_actors(
    actor_queues: list[Any],
    actor_processes: list[BaseProcess],
    actor_result_q: Any | None,
) -> None:
    if not actor_processes:
        return
    for command_q in actor_queues:
        with suppress(OSError, EOFError, BrokenPipeError):
            command_q.put(("stop",))
    stopped = 0
    if actor_result_q is not None:
        deadline = time.monotonic() + 5.0
        while stopped < len(actor_processes) and time.monotonic() < deadline:
            try:
                kind, _idx, _payload = actor_result_q.get(timeout=0.1)
            except queue.Empty:
                continue
            except (OSError, EOFError, BrokenPipeError):
                break
            if kind == "stopped":
                stopped += 1
    for proc in actor_processes:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)


def _export_plate_worker(job: _ExportPlateJob) -> tuple[Path, float]:
    start = time.perf_counter()
    refs_by_index = {ref.index: ref for ref in job.refs}
    out_3mf = job.output_dir / f"plate_{job.plate.index + 1:02d}.3mf"
    out_js = job.output_dir / f"plate_{job.plate.index + 1:02d}.json"

    def _load_part(part_index: int) -> trimesh.Trimesh:
        return load_mesh(refs_by_index[part_index].cache_path)

    export_plate_3mf_lazy(
        _load_part,
        job.plate,
        out_3mf,
        names=list(job.names),
        out_manifest=out_js,
        compression_mode=job.compression_mode,  # type: ignore[arg-type]
    )
    gc.collect()
    return out_3mf, time.perf_counter() - start


def _footprint_worker(job: _FootprintJob) -> tuple[int, BaseGeometry, float]:
    start = time.perf_counter()
    mesh = load_mesh(job.ref.cache_path)
    try:
        shadow = mesh_to_packing_shadow(mesh)
    finally:
        clear_mesh_cache(mesh)
        del mesh
        gc.collect()
    return job.ref.index, shadow, time.perf_counter() - start


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / (1024**3)
    if gib >= 1:
        return f"{gib:.1f} GiB"
    return f"{value / (1024**2):.1f} MiB"


def _rss_to_mb(raw: float) -> float:
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _process_max_rss_mb() -> float:
    return _rss_to_mb(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def _select_retained_indices(
    paths: list[Path],
    *,
    memory_budget_bytes: int | None,
    orient_workers: int,
) -> tuple[set[int], dict[str, object]]:
    sizes: list[int] = []
    for path in paths:
        try:
            sizes.append(path.stat().st_size)
        except OSError:
            sizes.append(0)
    largest = max(sizes, default=0)
    if memory_budget_bytes is None or largest <= 0 or orient_workers <= 0:
        return set(), {
            "enabled": False,
            "retained_indices": [],
            "estimated_retained_bytes": 0,
            "cap_bytes": 0,
        }
    cap = min(int(memory_budget_bytes * 0.10), int(orient_workers * largest * 18.0 * 0.10))
    retained: set[int] = set()
    used = 0
    for idx in sorted(range(len(paths)), key=lambda i: sizes[i], reverse=True):
        estimate = int(sizes[idx] * 4.0)
        if estimate <= 0:
            continue
        if len(retained) >= orient_workers:
            break
        if used + estimate <= cap:
            retained.add(idx)
            used += estimate
    return retained, {
        "enabled": bool(retained),
        "retained_indices": sorted(retained),
        "retained_names": [paths[i].name for i in sorted(retained)],
        "estimated_retained_bytes": used,
        "cap_bytes": cap,
        "estimate_multiplier": 4.0,
        "max_retained_parts": orient_workers,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_prepare(args: PrepareRunArgs) -> int:  # noqa: C901
    console = Console(stderr=True)
    profile_metadata: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "dry_run": args.dry_run,
        "resume": args.resume,
        "any_rotation": args.any_rotation,
    }
    profiler = make_profiler(
        command="prepare",
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

    try:
        px_raw, py_raw, pz_raw = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    post_fit_scale = (
        float(args.post_fit_scale)
        if args.post_fit_scale is not None
        else (st.scaling.post_fit_scale if st else 1.0)
    )
    gap = resolve_gap(args.gap_mm, st)
    edge_margin = resolve_edge_margin(args.edge_margin_mm, st)
    resin_options = resolve_resin_orientation_options(args.resin_balance, st)
    method: str = args.method or "sorted"
    requested_packer = args.packer or (st.autopack.packer if st else None) or "auto"
    try:
        packer = _resolve_prepare_packer(args.packer, st.autopack.packer if st else None)
        bitmap_options = _resolve_prepare_bitmap_options(
            grid_mm=args.bitmap_grid_mm,
            settings_grid_mm=st.autopack.bitmap_grid_mm if st else None,
            beam_width=args.bitmap_beam_width,
            settings_beam_width=st.autopack.bitmap_beam_width if st else None,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)
    if args.export_compression not in {"default", "fast", "store"}:
        console.print("[red]--export-compression must be default, fast, or store.[/red]")
        return finish_profile(profiler, console, 2)
    if args.orientation_strategy not in {"auto", "legacy"}:
        console.print("[red]--orientation-strategy must be auto or legacy.[/red]")
        return finish_profile(profiler, console, 2)
    if args.orientation_quality not in {"default", "adaptive"}:
        console.print("[red]--orientation-quality must be default or adaptive.[/red]")
        return finish_profile(profiler, console, 2)

    px, py, pz = px_raw, py_raw, pz_raw
    epx = px - 2.0 * edge_margin
    epy = py - 2.0 * edge_margin

    if st and st.printer.name:
        console.print(f"Printer: {st.printer.name}")
    console.print(f"Build volume: {px:.1f} × {py:.1f} × {pz:.1f} mm")
    console.print(
        f"Gap: {gap} mm  |  edge margin: {edge_margin} mm  |  post_fit_scale: {post_fit_scale}"
    )
    if packer == "bitmap" and requested_packer == "bitmap":
        console.print(
            "[yellow]Warning: prepare bitmap packing is a fast heuristic backend; "
            "use --packer auto/exact for quality-first plate count.[/yellow]"
        )

    with profiler.stage("collect inputs"):
        paths = collect_mesh_paths(args.input_dir, args.recursive)
    if not paths:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        console.print(f"[red]No mesh files ({exts}) found under {args.input_dir}[/red]")
        return finish_profile(profiler, console, 1)
    names = [
        str(p.relative_to(args.input_dir)) if p.is_relative_to(args.input_dir) else p.name
        for p in paths
    ]
    try:
        worker_plan = make_prepare_worker_plan(
            paths,
            requested_workers=args.workers,
            memory_budget_fraction=DEFAULT_MEMORY_BUDGET_FRACTION,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)
    profile_metadata["resource_plan"] = worker_plan.to_json()
    profile_metadata["orientation_options"] = {
        "strategy": args.orientation_strategy,
        "quality": args.orientation_quality,
        "resin_balance": resin_options.resin_balance,
        "long_part_target_angle_min_deg": resin_options.long_part_target_angle_min_deg,
        "long_part_target_angle_max_deg": resin_options.long_part_target_angle_max_deg,
        "long_part_low_angle_penalty_below_deg": resin_options.long_part_low_angle_penalty_below_deg,
        "long_part_high_angle_penalty_above_deg": resin_options.long_part_high_angle_penalty_above_deg,
    }
    profile_metadata["packing_options"] = {
        "requested_packer": requested_packer,
        "resolved_packer": packer,
        "packer": packer,
        "bitmap_grid_mm": bitmap_options.grid_mm if packer == "bitmap" else None,
        "bitmap_beam_width": bitmap_options.beam_width if packer == "bitmap" else None,
        "footprint_cache": args.footprint_cache,
        "packing_result_cache": args.packing_result_cache,
    }
    profile_metadata["export"] = {
        "compression": args.export_compression,
        "writer": "stlbench-direct",
    }
    if args.verbose:
        console.print(
            "[dim]workers: "
            f"scale={worker_plan.scale_workers} "
            f"orient={worker_plan.orient_workers} "
            f"footprint={worker_plan.footprint_workers} "
            f"export=auto  "
            f"RAM budget={_fmt_bytes(worker_plan.memory_budget_bytes)}  "
            f"largest model={_fmt_bytes(worker_plan.input.largest_bytes)}[/dim]"
        )

    temp_cache: tempfile.TemporaryDirectory[str] | None = None
    if args.dry_run and not args.resume:
        temp_cache = tempfile.TemporaryDirectory(prefix="stlbench-prepare-")
        cache_dir = Path(temp_cache.name)
    else:
        cache_dir = args.output_dir / "cache"

    # ──────────────────────────────────────────────────────────────────────────
    # Fast path: resume from orient cache
    # ──────────────────────────────────────────────────────────────────────────
    prepared_refs: list[PreparedMeshRef] | None = None
    repair_reports: list[RepairReport] = []
    if args.resume:
        with profiler.stage("cache load"):
            cached = _load_orient_cache(cache_dir, paths, console)
        if cached is not None:
            names, prepared_refs = cached
            console.print(
                f"[green]Resumed from cache[/green] ({len(prepared_refs)} meshes, "
                f"skipping scale + orient steps)."
            )

    if prepared_refs is None:
        # ──────────────────────────────────────────────────────────────────────
        # Step 1 – Scale
        # ──────────────────────────────────────────────────────────────────────
        console.print("\n[bold]1 / 3  Scale[/bold]")

        _n = worker_plan.scale_workers
        if args.verbose:
            console.print(f"[dim]scale-orient: {_n} workers for {len(paths)} meshes[/dim]")

        use_fused_orientation = args.orientation_strategy == "auto"
        actor_processes: list[BaseProcess] = []
        actor_queues: list[Any] = []
        actor_result_q: Any | None = None
        retained_indices: set[int] = set()
        retained_actor: dict[int, int] = {}
        if use_fused_orientation:
            retained_indices, retention_metadata = _select_retained_indices(
                paths,
                memory_budget_bytes=worker_plan.memory_budget_bytes,
                orient_workers=worker_plan.orient_workers,
            )
            profile_metadata["orientation_retention"] = retention_metadata
            ctx = mp.get_context("spawn")
            actor_result_q = ctx.Queue()
            actor_count = max(1, min(len(paths), worker_plan.orient_workers))
            for _actor_idx in range(actor_count):
                command_q = ctx.Queue()
                proc = ctx.Process(
                    target=_orientation_actor_main,
                    args=(command_q, actor_result_q),
                )
                proc.start()
                actor_queues.append(command_q)
                actor_processes.append(proc)
            for actor_idx, part_idx in enumerate(sorted(retained_indices)):
                retained_actor[part_idx] = actor_idx % actor_count

        _so: list = [None] * len(paths)
        with (
            profiler.stage("scale orientation search"),
            make_progress(console, enabled=args.progress) as progress,
        ):
            ptask = progress.add_task("Finding scale orientations…", total=len(paths))
            if use_fused_orientation:
                if actor_result_q is None:
                    raise RuntimeError("Orientation actor result queue was not initialised.")
                for idx, path in enumerate(paths):
                    actor_idx = retained_actor.get(idx, idx % len(actor_queues))
                    actor_queues[actor_idx].put(
                        (
                            "scale",
                            _ScaleOrientJob(
                                index=idx,
                                path=path,
                                px=epx,
                                py=epy,
                                pz=pz,
                                method=method,
                                any_rotation=args.any_rotation,
                                repair_options=repair_options,
                                name=names[idx],
                                repair_cache_dir=repair_cache_dir,
                                orientation_strategy=args.orientation_strategy,
                            ),
                            idx in retained_indices,
                        )
                    )
                scale_done = 0
                while scale_done < len(paths):
                    kind, idx, payload = actor_result_q.get()
                    if kind == "error":
                        profile_metadata["orientation_actor_fallback"] = {
                            "stage": "scale",
                            "part_index": idx,
                            "reason": payload,
                        }
                        _stop_orientation_actors(actor_queues, actor_processes, actor_result_q)
                        console.print(
                            f"[red]fused orientation scale failed for part {idx}: {payload}[/red]"
                        )
                        return finish_profile(profiler, console, 1)
                    if kind != "scale":
                        continue
                    (
                        transform,
                        dims,
                        has_multiple,
                        repair_report,
                        duration_s,
                        timing_payload,
                        worker_rss_mb,
                    ) = payload
                    timing_payload["worker_max_rss_mb"] = worker_rss_mb
                    profiler.record_worker(
                        "prepare.scale_orientation", duration_s, **timing_payload
                    )
                    _so[idx] = (transform, dims, repair_report)
                    if repair_report.enabled and repair_report.changed:
                        console.print(f"[dim]repair: {names[idx]} — mesh topology updated[/dim]")
                    if has_multiple:
                        console.print(
                            f"[yellow]Warning: {names[idx]!r} contains multiple surfaces — "
                            f"model may be broken (surfaces merged for processing).[/yellow]"
                        )
                    progress.update(ptask, advance=1, description=f"Scale orient: {names[idx]}")
                    scale_done += 1
            else:
                with ProcessPoolExecutor(max_workers=_n, max_tasks_per_child=1) as scale_pool:
                    futs_so = {
                        scale_pool.submit(
                            _scale_orientation_worker,
                            _ScaleOrientJob(
                                index=idx,
                                path=path,
                                px=epx,
                                py=epy,
                                pz=pz,
                                method=method,
                                any_rotation=args.any_rotation,
                                repair_options=repair_options,
                                name=names[idx],
                                repair_cache_dir=repair_cache_dir,
                                orientation_strategy=args.orientation_strategy,
                            ),
                        ): idx
                        for idx, path in enumerate(paths)
                    }
                    for fut_so in as_completed(futs_so):
                        idx = futs_so[fut_so]
                        try:
                            (
                                _idx,
                                transform,
                                dims,
                                has_multiple,
                                repair_report,
                                duration_s,
                                timing_payload,
                            ) = fut_so.result()
                        except (OSError, ValueError, TypeError) as e:
                            console.print(f"[red]Failed to load {paths[idx]}: {e}[/red]")
                            return finish_profile(profiler, console, 1)
                        profiler.record_worker(
                            "prepare.scale_orientation", duration_s, **timing_payload
                        )
                        _so[idx] = (transform, dims, repair_report)
                        if repair_report.enabled and repair_report.changed:
                            console.print(
                                f"[dim]repair: {names[idx]} — mesh topology updated[/dim]"
                            )
                        if has_multiple:
                            console.print(
                                f"[yellow]Warning: {names[idx]!r} contains multiple surfaces — "
                                f"model may be broken (surfaces merged for processing).[/yellow]"
                            )
                        progress.update(ptask, advance=1, description=f"Scale orient: {names[idx]}")

        scale_transforms = [r[0] for r in _so]
        oriented_dims = [r[1] for r in _so]
        repair_reports = [r[2] for r in _so]

        with profiler.stage("scale computation"):
            try:
                s_max, reports = compute_global_scale((px, py, pz), oriented_dims, names, method)  # type: ignore[arg-type]
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                _stop_orientation_actors(actor_queues, actor_processes, actor_result_q)
                return finish_profile(profiler, console, 1)

        s_final = s_max * post_fit_scale
        lim_name = reports[0].name
        console.print(f"s_max={s_max:.6f}  post_fit={post_fit_scale}  s_final={s_final:.6f}")
        console.print(f"Limiting part: {lim_name}")

        table = Table(show_header=True, header_style="bold")
        table.add_column("part", max_width=42)
        table.add_column("scaled (mm)", justify="right")
        for r in reports:
            sd = (r.dx * s_final, r.dy * s_final, r.dz * s_final)
            table.add_row(r.name, f"{sd[0]:.2f} × {sd[1]:.2f} × {sd[2]:.2f}")
        console.print(table)

        # ──────────────────────────────────────────────────────────────────────
        # Step 2 – Orient for minimum supports
        # ──────────────────────────────────────────────────────────────────────
        console.print(f"\n[bold]2 / 3  Orient[/bold]  (overhang ≥ {args.overhang_threshold_deg}°)")

        orient_table = Table(show_header=True, header_style="bold")
        orient_table.add_column("part", max_width=42)
        orient_table.add_column("before", justify="right")
        orient_table.add_column("after", justify="right")
        orient_table.add_column("Δ", justify="right")
        stability_table = Table(show_header=True, header_style="bold")
        stability_table.add_column("part", max_width=42)
        stability_table.add_column("height", justify="right")
        stability_table.add_column("center_z", justify="right")
        stability_table.add_column("axis angle", justify="right")
        stability_table.add_column("contact", justify="right")
        stability_table.add_column("damage", justify="right")
        stability_table.add_column("up", justify="right")
        stability_table.add_column("footprint", justify="right")
        stability_table.add_column("score", justify="right")
        stability_table.add_column("reason")

        cache_dir.mkdir(parents=True, exist_ok=True)

        _n2 = worker_plan.orient_workers
        if args.verbose:
            console.print(f"[dim]orient: {_n2} workers for {len(paths)} meshes[/dim]")

        _prepared: list[
            tuple[
                float,
                float,
                float,
                PreparedMeshRef,
                dict[str, float | str],
                dict[str, float | str | bool],
            ]
            | None
        ] = [None] * len(paths)
        with (
            profiler.stage("scale/orient/cache"),
            make_progress(console, enabled=args.progress) as progress,
        ):
            ptask = progress.add_task("Orienting parts…", total=len(paths))
            if use_fused_orientation:
                if actor_result_q is None:
                    raise RuntimeError("Orientation actor result queue was not initialised.")
                for idx, path in enumerate(paths):
                    actor_idx = retained_actor.get(idx, idx % len(actor_queues))
                    actor_queues[actor_idx].put(
                        (
                            "orient",
                            _PrepareCacheJob(
                                index=idx,
                                path=path,
                                name=names[idx],
                                cache_dir=cache_dir,
                                scale_transform=scale_transforms[idx],
                                source_up=scale_transforms[idx][:3, :3]
                                @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
                                scale=s_final,
                                overhang_threshold_deg=args.overhang_threshold_deg,
                                n_orient_candidates=args.n_orient_candidates,
                                printer_xyz=(epx, epy, pz),
                                cleanup=args.cleanup,
                                resin_options=resin_options,
                                repair_options=repair_options,
                                repair_cache_dir=repair_cache_dir,
                                orientation_strategy=args.orientation_strategy,
                                orientation_quality=args.orientation_quality,
                            ),
                        )
                    )
                orient_done = 0
                while orient_done < len(paths):
                    kind, idx, payload = actor_result_q.get()
                    if kind == "error":
                        profile_metadata["orientation_actor_fallback"] = {
                            "stage": "orient",
                            "part_index": idx,
                            "reason": payload,
                        }
                        _stop_orientation_actors(actor_queues, actor_processes, actor_result_q)
                        console.print(
                            f"[red]fused orientation failed for part {idx}: {payload}[/red]"
                        )
                        return finish_profile(profiler, console, 1)
                    if kind != "orient":
                        continue
                    (
                        _idx,
                        sb,
                        sa,
                        pct,
                        ref,
                        metrics_payload,
                        duration_s,
                        timing_payload,
                        worker_rss_mb,
                    ) = payload
                    timing_payload["worker_max_rss_mb"] = worker_rss_mb
                    profiler.record_worker(
                        "prepare.scale_orient_cache", duration_s, **timing_payload
                    )
                    _prepared[idx] = (sb, sa, pct, ref, metrics_payload, timing_payload)
                    progress.update(ptask, advance=1, description=f"Oriented: {names[idx]}")
                    orient_done += 1
            else:
                with ProcessPoolExecutor(max_workers=_n2, max_tasks_per_child=1) as orient_pool:
                    future_to_idx = {
                        orient_pool.submit(
                            _prepare_cache_worker,
                            _PrepareCacheJob(
                                index=idx,
                                path=path,
                                name=names[idx],
                                cache_dir=cache_dir,
                                scale_transform=scale_transforms[idx],
                                source_up=scale_transforms[idx][:3, :3]
                                @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
                                scale=s_final,
                                overhang_threshold_deg=args.overhang_threshold_deg,
                                n_orient_candidates=args.n_orient_candidates,
                                printer_xyz=(epx, epy, pz),
                                cleanup=args.cleanup,
                                resin_options=resin_options,
                                repair_options=repair_options,
                                repair_cache_dir=repair_cache_dir,
                                orientation_strategy=args.orientation_strategy,
                                orientation_quality=args.orientation_quality,
                            ),
                        ): idx
                        for idx, path in enumerate(paths)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx.pop(future)
                        try:
                            (
                                _idx,
                                sb,
                                sa,
                                pct,
                                ref,
                                metrics_payload,
                                duration_s,
                                timing_payload,
                            ) = future.result()
                        # Worker boundary: report the part name and fail the command.
                        except Exception as e:  # noqa: BLE001
                            if args.verbose:
                                console.print_exception()
                            console.print(
                                f"[red]orient failed for part {idx} ({names[idx]}): {e}[/red]"
                            )
                            return finish_profile(profiler, console, 1)
                        profiler.record_worker(
                            "prepare.scale_orient_cache", duration_s, **timing_payload
                        )
                        _prepared[idx] = (sb, sa, pct, ref, metrics_payload, timing_payload)
                        progress.update(ptask, advance=1, description=f"Oriented: {names[idx]}")

        _stop_orientation_actors(actor_queues, actor_processes, actor_result_q)

        prepared_refs = []
        mesh_files: list[str] = []
        orientation_metrics: list[dict[str, object]] = []
        orientation_timings: list[dict[str, object]] = []
        for name, row in zip(names, _prepared, strict=True):
            if row is None:
                console.print(
                    f"[red]Internal error: no prepared orientation result for {name}.[/red]"
                )
                return finish_profile(profiler, console, 1)
            sb, sa, pct, ref, metrics_payload, timing_payload = row
            orient_table.add_row(name, f"{sb:.1f}", f"{sa:.1f}", f"{pct:+.0f}%")
            stability_table.add_row(
                name,
                f"{float(metrics_payload['selected_height_mm']):.2f}",
                f"{float(metrics_payload['center_z_ratio']):.3f}",
                f"{float(metrics_payload['long_axis_angle_from_bed_deg']):.1f}°",
                f"{float(metrics_payload['support_contact_proxy']):.3f}",
                f"{float(metrics_payload['surface_damage_proxy']):.3f}",
                f"{float(metrics_payload['source_up_dot_build_up']):.2f}",
                f"{float(metrics_payload['xy_footprint_area_mm2']):.0f}",
                f"{float(metrics_payload['stability_score']):.3f}",
                str(metrics_payload["selection_reason"]),
            )
            orientation_metrics.append({"part": name, **metrics_payload})
            orientation_timings.append({"part": name, **timing_payload})
            prepared_refs.append(ref)
            mesh_files.append(ref.cache_path.name)
        profile_metadata["orientation_stability"] = orientation_metrics
        profile_metadata["orientation_timings"] = orientation_timings
        del _prepared
        gc.collect()

        console.print(orient_table)
        if args.verbose:
            console.print(stability_table)

        if not args.dry_run:
            try:
                with profiler.stage("cache metadata save"):
                    _write_orient_cache_meta(cache_dir, paths, names, mesh_files)
                console.print(f"[dim]Orient cache saved → {cache_dir}[/dim]")
            except (OSError, TypeError, ValueError) as exc:
                console.print(f"[yellow]Warning: could not save orient cache: {exc}[/yellow]")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 – Layout
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold]3 / 3  Layout[/bold]")
    if prepared_refs is None:
        console.print("[red]Internal error: layout started without prepared mesh references.[/red]")
        return finish_profile(profiler, console, 1)

    # Pre-check: each part must fit the bed in at least one orientation before packing.
    epx = px - 2.0 * edge_margin
    epy = py - 2.0 * edge_margin
    for ref in prepared_refs:
        dx, dy, _dz = ref.dims
        if not ((dx <= epx and dy <= epy) or (dy <= epx and dx <= epy)):
            console.print(
                f"[red]Part {ref.name!r} ({dx:.1f}×{dy:.1f} mm) does not fit "
                f"on bed {epx:.1f}×{epy:.1f} mm after edge margin.[/red]"
            )
            return finish_profile(profiler, console, 1)

    n_parts = len(prepared_refs)

    shadows: list[BaseGeometry | None] = [None] * n_parts
    footprint_keys = [_footprint_cache_key(ref) for ref in prepared_refs]
    footprint_cache_dir = (
        args.output_dir / "cache" / "footprints"
        if args.footprint_cache and not args.dry_run
        else (args.output_dir / "cache" / "footprints" if args.footprint_cache else None)
    )
    footprint_hits = 0
    with (
        profiler.stage("footprint computation"),
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Computing footprints…", total=n_parts)
        missing_refs: list[PreparedMeshRef] = []
        for ref, key in zip(prepared_refs, footprint_keys, strict=True):
            cached_shadow = _load_cached_footprint(footprint_cache_dir, key)
            if cached_shadow is not None:
                shadows[ref.index] = cached_shadow
                footprint_hits += 1
                progress.update(ptask, advance=1, description=f"Footprint cache: {ref.name}")
            else:
                missing_refs.append(ref)
        with ProcessPoolExecutor(
            max_workers=worker_plan.footprint_workers,
            max_tasks_per_child=1,
        ) as footprint_pool:
            footprint_futures = {
                footprint_pool.submit(_footprint_worker, _FootprintJob(ref=ref)): ref
                for ref in missing_refs
            }
            for footprint_future in as_completed(footprint_futures):
                ref = footprint_futures[footprint_future]
                idx, shadow, duration_s = footprint_future.result()
                profiler.record_worker("prepare.footprint", duration_s)
                shadows[idx] = shadow
                if not args.dry_run:
                    _write_cached_footprint(footprint_cache_dir, footprint_keys[idx], shadow)
                progress.update(ptask, advance=1, description=f"Footprint: {ref.name}")
    profile_metadata["footprint_cache"] = {"hits": footprint_hits, "total": n_parts}
    packed_shadows = [s for s in shadows if s is not None]
    part_heights = [ref.dims[2] for ref in prepared_refs]
    packing_metadata: dict[str, object] = {}
    packing_cache_dir = (
        args.output_dir / "cache" / "prepare_packing" if args.packing_result_cache else None
    )
    packing_key = _packing_cache_key(
        footprint_keys=footprint_keys,
        bed_w=px,
        bed_h=py,
        gap_mm=gap,
        edge_margin_mm=edge_margin,
        part_heights=part_heights,
        packer=packer,
        bitmap_options=bitmap_options if packer == "bitmap" else None,
        grid_step_mm=args.grid_step_mm,
    )

    with profiler.stage("packing"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Packing…", total=n_parts)
        cached_plates = _load_packing_result(packing_cache_dir, packing_key, packer)
        if cached_plates is not None and _validate_layout_geometry(
            packed_shadows, cached_plates, px, py, gap
        ):
            plates = cached_plates
            packing_metadata["cache_hit"] = True
            packing_metadata["strategy"] = packer
            packing_metadata["final_plates"] = len(plates)
            progress.update(ptask, completed=n_parts)
        else:
            packing_metadata["cache_hit"] = False
            if packer == "bitmap":
                plates = profiler.profiled_call(
                    "prepare.packing.bitmap",
                    _pack_bitmap_multi_plate,
                    packed_shadows,
                    px,
                    py,
                    gap,
                    edge_margin,
                    bitmap_options,
                    packing_metadata,
                )
                progress.update(ptask, completed=n_parts)
            else:
                plates = profiler.profiled_call(
                    "prepare.packing.exact",
                    pack_polygons_on_plates,
                    packed_shadows,
                    px,
                    py,
                    gap_mm=gap,
                    grid_step_mm=args.grid_step_mm,
                    on_placed=lambda: progress.advance(ptask),
                    part_heights=part_heights,
                    metadata=packing_metadata,
                    edge_margin_mm=edge_margin,
                )
            if not _validate_layout_geometry(packed_shadows, plates, px, py, gap):
                console.print(
                    "[red]Internal error: prepare layout failed geometry validation.[/red]"
                )
                return finish_profile(profiler, console, 1)
            if not args.dry_run:
                _write_packing_result(packing_cache_dir, packing_key, packer, plates)
        packing_metadata["cache_key"] = packing_key
        packing_metadata["requested_packer"] = requested_packer
        packing_metadata["resolved_packer"] = packer
        packing_metadata["packer"] = packer
        packing_metadata["final_plates"] = len(plates)
    profile_metadata["packing"] = packing_metadata

    console.print(f"Plates: {len(plates)}")
    refs_by_index = {ref.index: ref for ref in prepared_refs}
    for pl in plates:
        console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")
        plate_table = Table(show_header=True, header_style="bold")
        plate_table.add_column("part", max_width=48)
        plate_table.add_column("x", justify="right")
        plate_table.add_column("y", justify="right")
        plate_table.add_column("rot", justify="right")
        for rect in pl.rects:
            ref = refs_by_index[rect.part_index]
            plate_table.add_row(
                ref.name,
                f"{rect.x:.1f}",
                f"{rect.y:.1f}",
                f"{rect.rotation_deg:.0f}°",
            )
        console.print(plate_table)

    if args.dry_run:
        rc = finish_profile(profiler, console, 0)
        if temp_cache is not None:
            temp_cache.cleanup()
        return rc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    export_refs = tuple(prepared_refs)
    output_names = tuple(ref.name for ref in prepared_refs)
    ref_size_by_index = {ref.index: ref.cache_path.stat().st_size for ref in prepared_refs}
    plate_part_bytes = [
        sum(ref_size_by_index.get(rect.part_index, 0) for rect in plate.rects) for plate in plates
    ]
    export_workers = choose_export_workers(
        plate_part_bytes=plate_part_bytes,
        requested_workers=args.workers,
        memory_budget_bytes=worker_plan.memory_budget_bytes,
        cpu_cap=worker_plan.cpu_cap,
    )
    profile_metadata["resource_plan"] = {
        **worker_plan.to_json(),
        "export_workers": export_workers,
        "export_plate_bytes": plate_part_bytes,
    }
    if args.verbose:
        console.print(
            f"[dim]export: {export_workers} workers for {len(plates)} plates "
            f"(largest plate={_fmt_bytes(max(plate_part_bytes, default=0))})[/dim]"
        )

    with profiler.stage("export"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Exporting plates…", total=len(plates))
        output_files: list[Path] = []
        with ProcessPoolExecutor(
            max_workers=export_workers,
            max_tasks_per_child=1,
        ) as export_pool:
            export_futures = {
                export_pool.submit(
                    _export_plate_worker,
                    _ExportPlateJob(
                        plate=pl,
                        refs=export_refs,
                        output_dir=args.output_dir,
                        names=output_names,
                        compression_mode=args.export_compression,
                    ),
                ): pl
                for pl in plates
            }
            for export_future in as_completed(export_futures):
                out_path, duration_s = export_future.result()
                profiler.record_worker("prepare.export", duration_s)
                console.print(f"Wrote {out_path}")
                output_files.extend([out_path, out_path.with_suffix(".json")])
                progress.advance(ptask)

    with profiler.stage("transform log"):
        transform_parts: list[dict] = []
        refs_by_index = {ref.index: ref for ref in prepared_refs}
        for pl in plates:
            plate_file = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
            for rect in pl.rects:
                ref = refs_by_index[rect.part_index]
                cache_bounds = ref.cache_bounds
                if cache_bounds is None:
                    cached_mesh = load_mesh(ref.cache_path)
                    try:
                        cache_bounds = np.asarray(cached_mesh.bounds, dtype=np.float64)
                    finally:
                        clear_mesh_cache(cached_mesh)
                placement_transform, placement_steps = placement_transform_for_bounds(
                    cache_bounds, rect
                )
                source_to_export = (
                    placement_transform @ ref.source_to_cache_matrix
                    if ref.source_to_cache_matrix is not None
                    else None
                )
                source_bounds = ref.source_bounds
                if source_bounds is None:
                    source_mesh, _source_repair_report = load_mesh_with_repair(
                        ref.source_path,
                        repair_options,
                        source_name=ref.name,
                        repair_cache_dir=repair_cache_dir,
                    )
                    try:
                        source_bounds = np.asarray(source_mesh.bounds, dtype=np.float64)
                    finally:
                        clear_mesh_cache(source_mesh)
                source_bounds_mm = bounds_to_list(source_bounds)
                final_bounds_mm = (
                    bounds_to_list(transform_bounds(source_bounds, source_to_export))
                    if source_bounds is not None and source_to_export is not None
                    else None
                )
                transform_parts.append(
                    transform_entry(
                        index=ref.index,
                        plate_index=pl.index,
                        plate_file=plate_file,
                        source_path=ref.source_path,
                        source_name=ref.name,
                        output_name=ref.name,
                        output_file=plate_file,
                        source_bounds_mm=source_bounds_mm,
                        final_bounds_mm=final_bounds_mm,
                        source_to_export_matrix=source_to_export,
                        steps=[*ref.transform_steps, *placement_steps],
                        plate_x_mm=rect.x,
                        plate_y_mm=rect.y,
                        rotation_deg=rect.rotation_deg,
                        source_transform_available=ref.source_transform_available,
                    )
                )
        write_transform_log(
            args.output_dir / "transforms.json",
            command="prepare",
            output_files=output_files,
            parts=transform_parts,
            metadata={"resume": args.resume},
        )
    if repair_reports:
        write_command_repair_report(
            args.output_dir,
            command="prepare",
            reports=repair_reports,
            dry_run=args.dry_run,
        )

    return finish_profile(profiler, console, 0)
