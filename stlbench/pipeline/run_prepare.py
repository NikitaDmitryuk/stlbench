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
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from shapely.geometry.base import BaseGeometry

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.fit import compute_global_scale
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.overhang import (
    ResinOrientationOptions,
    apply_min_overhang_orientation,
    find_stable_overhang_rotation,
    overhang_score,
)
from stlbench.export.plate import clear_mesh_cache, export_plate_3mf_lazy
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.polygon_footprint import mesh_to_packing_shadow
from stlbench.packing.polygon_pack import pack_polygons_on_plates
from stlbench.packing.rectpack_plate import PackedPlate
from stlbench.pipeline.common import (
    finish_profile,
    resolve_edge_margin,
    resolve_gap,
    resolve_printer,
    resolve_resin_orientation_options,
    resolve_settings,
)
from stlbench.pipeline.mesh_io import (
    SUPPORTED_EXTENSIONS,
    collect_mesh_paths,
    load_mesh,
    load_mesh_with_info,
)
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


@dataclass(frozen=True)
class PreparedMeshRef:
    index: int
    name: str
    source_path: Path
    cache_path: Path
    dims: tuple[float, float, float]


@dataclass(frozen=True)
class _ScaleOrientJob:
    index: int
    path: Path
    px: float
    py: float
    pz: float
    method: str
    any_rotation: bool


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


@dataclass(frozen=True)
class _ExportPlateJob:
    plate: PackedPlate
    refs: tuple[PreparedMeshRef, ...]
    output_dir: Path
    names: tuple[str, ...]


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
            )
        )
        del mesh
    gc.collect()
    return names, refs


def _scale_orientation_worker(
    job: _ScaleOrientJob,
) -> tuple[int, np.ndarray, tuple[float, float, float], bool, float]:
    start = time.perf_counter()
    m, has_multiple = load_mesh_with_info(job.path)
    try:
        transform, dims = select_orientation_for_scale(
            m,
            job.px,
            job.py,
            job.pz,
            job.method,  # type: ignore[arg-type]
            any_rotation=job.any_rotation,
            random_samples=ORIENTATION_SAMPLES_DEFAULT,
            seed=ORIENTATION_SEED_DEFAULT,
            compute_printability_metrics=False,
        )
    finally:
        clear_mesh_cache(m)
        del m
        gc.collect()
    return job.index, transform, dims, has_multiple, time.perf_counter() - start


def _prepare_cache_worker(
    job: _PrepareCacheJob,
) -> tuple[int, float, float, float, PreparedMeshRef, dict[str, float | str], float]:
    start = time.perf_counter()
    mesh = load_mesh(job.path)
    try:
        mesh.apply_transform(job.scale_transform)
        mesh.apply_scale(job.scale)
        mesh.apply_translation([0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])])
        sb = overhang_score(mesh, _IDENTITY3, job.overhang_threshold_deg)
        rotation, sa, metrics = find_stable_overhang_rotation(
            mesh,
            overhang_threshold_deg=job.overhang_threshold_deg,
            n_candidates=job.n_orient_candidates,
            printer_dims=job.printer_xyz,
            resin_options=job.resin_options,
            source_up=job.source_up,
        )
        metrics_payload: dict[str, float | str] = {
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
        }
        pct = (sb - sa) / max(abs(sb), 1.0) * 100.0
        oriented = apply_min_overhang_orientation(mesh, rotation)
        if job.cleanup:
            oriented, _n_removed = remove_small_components(oriented)
        bounds = np.asarray(oriented.bounds)
        dims_raw = bounds[1] - bounds[0]
        dims = (float(dims_raw[0]), float(dims_raw[1]), float(dims_raw[2]))
        out = job.cache_dir / _cache_mesh_name(job.index, job.name)
        oriented.export(str(out))
        clear_mesh_cache(oriented)
        ref = PreparedMeshRef(
            index=job.index,
            name=job.name,
            source_path=job.path,
            cache_path=out,
            dims=dims,
        )
        return job.index, sb, sa, pct, ref, metrics_payload, time.perf_counter() - start
    finally:
        clear_mesh_cache(mesh)
        del mesh
        gc.collect()


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


# ---------------------------------------------------------------------------
# Progress bar factory
# ---------------------------------------------------------------------------


def _make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / (1024**3)
    if gib >= 1:
        return f"{gib:.1f} GiB"
    return f"{value / (1024**2):.1f} MiB"


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

    px, py, pz = px_raw, py_raw, pz_raw
    epx = px - 2.0 * edge_margin
    epy = py - 2.0 * edge_margin

    if st and st.printer.name:
        console.print(f"Printer: {st.printer.name}")
    console.print(f"Build volume: {px:.1f} × {py:.1f} × {pz:.1f} mm")
    console.print(
        f"Gap: {gap} mm  |  edge margin: {edge_margin} mm  |  post_fit_scale: {post_fit_scale}"
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
        "resin_balance": resin_options.resin_balance,
        "long_part_target_angle_min_deg": resin_options.long_part_target_angle_min_deg,
        "long_part_target_angle_max_deg": resin_options.long_part_target_angle_max_deg,
        "long_part_low_angle_penalty_below_deg": resin_options.long_part_low_angle_penalty_below_deg,
        "long_part_high_angle_penalty_above_deg": resin_options.long_part_high_angle_penalty_above_deg,
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

        _so: list = [None] * len(paths)
        with profiler.stage("scale orientation search"), _make_progress(console) as progress:
            ptask = progress.add_task("Finding scale orientations…", total=len(paths))
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
                        ),
                    ): idx
                    for idx, path in enumerate(paths)
                }
                for fut_so in as_completed(futs_so):
                    idx = futs_so[fut_so]
                    try:
                        _idx, transform, dims, has_multiple, duration_s = fut_so.result()
                    except (OSError, ValueError, TypeError) as e:
                        console.print(f"[red]Failed to load {paths[idx]}: {e}[/red]")
                        return finish_profile(profiler, console, 1)
                    profiler.record_worker("prepare.scale_orientation", duration_s)
                    _so[idx] = (transform, dims)
                    if has_multiple:
                        console.print(
                            f"[yellow]Warning: {names[idx]!r} contains multiple surfaces — "
                            f"model may be broken (surfaces merged for processing).[/yellow]"
                        )
                    progress.update(ptask, advance=1, description=f"Scale orient: {names[idx]}")

        scale_transforms = [r[0] for r in _so]
        oriented_dims = [r[1] for r in _so]

        with profiler.stage("scale computation"):
            try:
                s_max, reports = compute_global_scale((px, py, pz), oriented_dims, names, method)  # type: ignore[arg-type]
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
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
            tuple[float, float, float, PreparedMeshRef, dict[str, float | str]] | None
        ] = [None] * len(paths)
        with profiler.stage("scale/orient/cache"), _make_progress(console) as progress:
            ptask = progress.add_task("Orienting parts…", total=len(paths))
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
                        ),
                    ): idx
                    for idx, path in enumerate(paths)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx.pop(future)
                    try:
                        _idx, sb, sa, pct, ref, metrics_payload, duration_s = future.result()
                    except Exception as e:
                        if args.verbose:
                            console.print_exception()
                        console.print(
                            f"[red]orient failed for part {idx} ({names[idx]}): {e}[/red]"
                        )
                        return finish_profile(profiler, console, 1)
                    profiler.record_worker("prepare.scale_orient_cache", duration_s)
                    _prepared[idx] = (sb, sa, pct, ref, metrics_payload)
                    progress.update(ptask, advance=1, description=f"Oriented: {names[idx]}")

        prepared_refs = []
        mesh_files: list[str] = []
        orientation_metrics: list[dict[str, object]] = []
        for name, row in zip(names, _prepared, strict=True):
            assert row is not None
            sb, sa, pct, ref, metrics_payload = row
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
            prepared_refs.append(ref)
            mesh_files.append(ref.cache_path.name)
        profile_metadata["orientation_stability"] = orientation_metrics
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
            except Exception as exc:
                console.print(f"[yellow]Warning: could not save orient cache: {exc}[/yellow]")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 – Layout
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold]3 / 3  Layout[/bold]")
    assert prepared_refs is not None

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
    with profiler.stage("footprint computation"), _make_progress(console) as progress:
        ptask = progress.add_task("Computing footprints…", total=n_parts)
        with ProcessPoolExecutor(
            max_workers=worker_plan.footprint_workers,
            max_tasks_per_child=1,
        ) as footprint_pool:
            footprint_futures = {
                footprint_pool.submit(_footprint_worker, _FootprintJob(ref=ref)): ref
                for ref in prepared_refs
            }
            for footprint_future in as_completed(footprint_futures):
                ref = footprint_futures[footprint_future]
                idx, shadow, duration_s = footprint_future.result()
                profiler.record_worker("prepare.footprint", duration_s)
                shadows[idx] = shadow
                progress.update(ptask, advance=1, description=f"Footprint: {ref.name}")
    packed_shadows = [s for s in shadows if s is not None]
    part_heights = [ref.dims[2] for ref in prepared_refs]
    packing_metadata: dict[str, object] = {}

    with profiler.stage("packing"), _make_progress(console) as progress:
        ptask = progress.add_task("Packing…", total=n_parts)
        plates = profiler.profiled_call(
            "prepare.packing",
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
    profile_metadata["packing"] = packing_metadata

    console.print(f"Plates: {len(plates)}")
    for pl in plates:
        console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")

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

    with profiler.stage("export"), _make_progress(console) as progress:
        ptask = progress.add_task("Exporting plates…", total=len(plates))
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
                    ),
                ): pl
                for pl in plates
            }
            for export_future in as_completed(export_futures):
                out_path, duration_s = export_future.result()
                profiler.record_worker("prepare.export", duration_s)
                console.print(f"Wrote {out_path}")
                progress.advance(ptask)

    return finish_profile(profiler, console, 0)
