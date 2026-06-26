from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.fit import aabb_edge_lengths, compute_global_scale
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.overhang import find_stable_overhang_rotation
from stlbench.export.plate import export_plate_3mf, mesh_footprint_xy
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.polygon_footprint import mesh_to_packing_shadow
from stlbench.packing.polygon_pack import (
    footprints_to_box_polygons,
    pack_polygons_on_plates,
    try_pack_polygons_single_plate,
)
from stlbench.packing.rectpack_plate import PackedPlate
from stlbench.pipeline.common import (
    finish_profile,
    load_named_meshes,
    n_workers,
    resolve_edge_margin,
    resolve_gap,
    resolve_orientation_policy,
    resolve_orientation_scale_tolerance,
    resolve_printer,
    resolve_resin_orientation_options,
    resolve_settings,
)
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
    profile_options: ProfileOptions | None = None
    edge_margin_mm: float | None = None
    resin_balance: str | None = None


def _try_pack_all(
    footprints: list[tuple[float, float]],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
) -> PackedPlate | None:
    """Try to pack all footprints onto a single plate. Returns None on failure."""
    polygons = footprints_to_box_polygons(footprints)
    return try_pack_polygons_single_plate(polygons, bed_w, bed_h, gap_mm)


def _bisect_scale(
    base_footprints: list[tuple[float, float]],
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
        scaled = [(fw * mid, fh * mid) for fw, fh in base_footprints]
        plate = _try_pack_all(scaled, bed_w, bed_h, gap_mm)
        if plate is not None:
            best_s = mid
            best_plate = plate
            lo = mid
        else:
            hi = mid

    return best_s, best_plate


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
    rot_samples = (
        int(args.rotation_samples)
        if args.rotation_samples is not None
        else ORIENTATION_SAMPLES_DEFAULT
    )

    with profiler.stage("load meshes"):
        loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return finish_profile(profiler, console, 1)
    _paths, names, meshes = loaded

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
    with profiler.stage("orientation search"), ThreadPoolExecutor(max_workers=_n) as pool:
        _results = list(profiler.map(pool, "autopack.orientation", _orient_one, meshes))

    dims_list: list[tuple[float, float, float]] = [r[0] for r in _results]
    orient_transforms: list[np.ndarray] = [r[1] for r in _results]
    oriented_dims: list[tuple[float, float, float]] = [r[2] for r in _results]

    with profiler.stage("scale upper bound"):
        s_upper, _ = compute_global_scale((epx, epy, epz), oriented_dims, names, "sorted")
        s_upper *= post_fit_scale

    # (ex, ey) is the XY footprint after the orientation transform
    base_footprints: list[tuple[float, float]] = [(ex, ey) for ex, ey, _ez in oriented_dims]

    with profiler.stage("bbox pack bisection"):
        s_best, plate = _bisect_scale(base_footprints, epx, epy, gap, s_upper)

    if plate is None or s_best <= 0:
        console.print("[red]Cannot fit all parts on one plate at any scale.[/red]")
        return finish_profile(profiler, console, 1)

    console.print(f"Optimal scale (all parts on one plate): {s_best:.6f}")
    console.print(f"Parts: {len(names)}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("original (mm)", justify="right")
    table.add_column("scaled (mm)", justify="right")
    for name, d, od in zip(names, dims_list, oriented_dims, strict=True):
        orig = f"{d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f}"
        sd = tuple(x * s_best for x in od)
        scaled = f"{sd[0]:.2f} x {sd[1]:.2f} x {sd[2]:.2f}"
        table.add_row(name, orig, scaled)
    console.print(table)

    if args.dry_run:
        return finish_profile(profiler, console, 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Apply orientation transform + scale; export_plate_stl handles translation to origin.
    def _apply_scale(m_t4: tuple[trimesh.Trimesh, np.ndarray]) -> trimesh.Trimesh:
        m, t4 = m_t4
        s = m.copy()
        s.apply_transform(t4)
        s.apply_scale(s_best)
        return s

    if args.verbose:
        console.print(f"[dim]scale: {_n} workers for {len(meshes)} meshes[/dim]")
    with profiler.stage("apply scale"), ThreadPoolExecutor(max_workers=_n) as pool:
        scaled_meshes: list[trimesh.Trimesh] = list(
            profiler.map(
                pool,
                "autopack.apply_scale",
                _apply_scale,
                zip(meshes, orient_transforms, strict=True),
            )
        )

    if args.cleanup:
        with profiler.stage("cleanup"):
            for i, m in enumerate(scaled_meshes):
                cleaned, n_rem = remove_small_components(m)
                if n_rem:
                    scaled_meshes[i] = cleaned
                    console.print(
                        f"[dim]cleanup: {names[i]} — removed {n_rem} tiny component(s)[/dim]"
                    )

    # Repack with exact XY shadows for tighter nesting.
    # The bisection used bounding boxes (fast); this single pass uses exact shadows.
    with profiler.stage("exact footprint computation"):
        shadows = [
            profiler.profiled_call("autopack.exact_footprint", mesh_to_packing_shadow, m)
            for m in scaled_meshes
        ]
    with profiler.stage("exact packing"):
        part_heights = [mesh_footprint_xy(m)[2] for m in scaled_meshes]
        packing_metadata: dict[str, object] = {}
        exact_plates = profiler.profiled_call(
            "autopack.exact_packing",
            pack_polygons_on_plates,
            shadows,
            px,
            py,
            gap_mm=gap,
            part_heights=part_heights,
            metadata=packing_metadata,
            edge_margin_mm=edge_margin,
        )
        profile_metadata["packing"] = packing_metadata
        final_plate = exact_plates[0] if exact_plates else plate

    with profiler.stage("export"):
        out_3mf = args.output_dir / "autopack_plate.3mf"
        out_json = args.output_dir / "autopack_plate.json"
        export_plate_3mf(
            scaled_meshes, final_plate, out_3mf, names=list(names), out_manifest=out_json
        )
    console.print(f"Wrote {out_3mf}")
    console.print(f"Wrote {out_json}")
    return finish_profile(profiler, console, 0)
