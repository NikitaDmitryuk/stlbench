from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rectpack
import trimesh
from rich.console import Console

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.config.enums import ScaleFitMethod
from stlbench.core.fit import aabb_edge_lengths, compute_global_scale
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    rotation_to_transform4,
)
from stlbench.export.plate import mesh_footprint_xy
from stlbench.export.transform_log import (
    bounds_to_list,
    mesh_bounds,
    shared_geometry_placement_transform_for_mesh,
    transform_bounds,
    transform_entry,
    transform_step,
    translation_matrix,
    uniform_scale_matrix,
    write_transform_log,
)
from stlbench.packing.layout_orientation import select_layout_transform
from stlbench.packing.rectpack_plate import (
    PackedPlate,
    PackedRect,
)
from stlbench.pipeline.common import (
    finish_profile,
    load_mesh_with_repair,
    repair_cache_dir_for_output,
    resolve_gap,
    resolve_orientation_policy,
    resolve_orientation_scale_tolerance,
    resolve_printer,
    resolve_repair_cache_enabled,
    resolve_repair_options,
    resolve_settings,
    write_command_repair_report,
)
from stlbench.pipeline.mesh_io import collect_mesh_paths
from stlbench.pipeline.progress import make_progress
from stlbench.profiling import ProfileOptions, make_profiler


@dataclass
class FillRunArgs:
    input_file: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    scale: bool
    orient_on: bool
    orient_threshold_deg: float
    dry_run: bool
    cleanup: bool = False
    repair: bool = False
    any_rotation: bool = False
    orientation_policy: str | None = None
    orientation_scale_tolerance: float | None = None
    rotation_samples: int | None = None
    profile_options: ProfileOptions | None = None
    repair_cache: bool = True
    progress: bool = True


def _max_copies_on_plate(
    fw: float,
    fh: float,
    bed_w: float,
    bed_h: float,
    gap_mm: float,
) -> PackedPlate | None:
    """Pack as many identical rectangles (fw x fh) as possible onto one bin."""
    scale = 1000.0
    bw = max(1, int(np.floor((bed_w + gap_mm) * scale)))
    bh = max(1, int(np.floor((bed_h + gap_mm) * scale)))
    rw = max(1, int(np.ceil((fw + gap_mm) * scale)))
    rh = max(1, int(np.ceil((fh + gap_mm) * scale)))

    if not ((fw <= bed_w and fh <= bed_h) or (fh <= bed_w and fw <= bed_h)):
        return None

    max_possible = int((bw * bh) / max(1, rw * rh)) + 1
    max_possible = min(max_possible, 512)

    best_count = 0
    best_rects: list[PackedRect] = []

    lo, hi = 1, max_possible
    while lo <= hi:
        mid = (lo + hi) // 2
        packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline, rotation=True)
        packer.add_bin(bw, bh)
        for i in range(mid):
            packer.add_rect(rw, rh, rid=i)
        packer.pack()

        if len(packer) == 0:
            hi = mid - 1
            continue

        placed = []
        for r in packer[0]:
            rid = getattr(r, "rid", None)
            if rid is None:
                continue
            placed_w = int(round(r.width))
            placed_h = int(round(r.height))
            was_rotated = (placed_w == rh and placed_h == rw) and (rw != rh)
            actual_w = fh if was_rotated else fw
            actual_h = fw if was_rotated else fh
            placed.append(
                PackedRect(
                    part_index=0,
                    x=float(r.x) / scale,
                    y=float(r.y) / scale,
                    width=float(actual_w),
                    height=float(actual_h),
                    rotation_deg=90.0 if was_rotated else 0.0,
                )
            )

        if len(placed) >= mid:
            best_count = mid
            best_rects = placed
            lo = mid + 1
        else:
            if len(placed) > best_count:
                best_count = len(placed)
                best_rects = placed
            hi = mid - 1

    if best_count == 0:
        return None
    return PackedPlate(index=0, rects=tuple(best_rects[:best_count]))


def run_fill(args: FillRunArgs) -> int:
    console = Console(stderr=True)
    profiler = make_profiler(
        command="fill",
        output_base=args.output_dir,
        options=args.profile_options,
        metadata={"input": str(args.input_file), "dry_run": args.dry_run},
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
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    gap = resolve_gap(args.gap_mm, st)
    try:
        orientation_policy = resolve_orientation_policy(args.orientation_policy)
        scale_tolerance = resolve_orientation_scale_tolerance(args.orientation_scale_tolerance)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    inp = args.input_file
    if inp.is_dir():
        found = collect_mesh_paths(inp, recursive=False)
        if len(found) != 1:
            console.print(
                f"[red]fill expects exactly one mesh file (found {len(found)} in {inp}).[/red]"
            )
            return finish_profile(profiler, console, 2)
        inp = found[0]

    if not inp.is_file():
        console.print(f"[red]File not found: {inp}[/red]")
        return finish_profile(profiler, console, 2)

    with profiler.stage("load mesh"):
        try:
            mesh, repair_report = load_mesh_with_repair(
                inp,
                repair_options,
                source_name=inp.name,
                repair_cache_dir=repair_cache_dir,
            )
        except (OSError, ValueError, TypeError) as e:
            console.print(f"[red]Failed to load {inp}: {e}[/red]")
            return finish_profile(profiler, console, 1)
    if repair_report.enabled and repair_report.changed:
        console.print(f"[dim]repair: {inp.name} — mesh topology updated[/dim]")

    source_bounds_np = np.asarray(mesh.bounds, dtype=np.float64)
    source_bounds = mesh_bounds(mesh)
    pipeline_matrix = np.eye(4, dtype=np.float64)
    pipeline_steps: list[dict] = []
    if repair_report.enabled:
        from stlbench.core.mesh_repair import repair_report_step

        pipeline_steps.append(repair_report_step(repair_report))

    if args.cleanup:
        with profiler.stage("cleanup"):
            mesh, n_rem = remove_small_components(mesh)
            if n_rem:
                console.print(f"[dim]cleanup: {inp.name} — removed {n_rem} tiny component(s)[/dim]")
                pipeline_steps.append(
                    transform_step(
                        "cleanup",
                        params={"removed_components": n_rem},
                        available=False,
                    )
                )

    if args.scale:
        with profiler.stage("scale"):
            dims = aabb_edge_lengths(np.asarray(mesh.bounds))
            s, _ = compute_global_scale((px, py, pz), [dims], [inp.name], ScaleFitMethod.SORTED)
            pf = st.scaling.post_fit_scale if st else 1.0
            s_final = s * pf
            scale_matrix = uniform_scale_matrix(s_final)
            mesh.apply_scale(s_final)
            pipeline_matrix = scale_matrix @ pipeline_matrix
            pipeline_steps.append(
                transform_step("scale", matrix=scale_matrix, params={"scale_factor": s_final})
            )
        console.print(f"Scaled {inp.name} by {s_final:.6f}")

    if args.orient_on:
        with profiler.stage("overhang orientation"):
            rotation, score = find_min_overhang_rotation(
                mesh,
                overhang_threshold_deg=args.orient_threshold_deg,
                printer_dims=(px, py, pz),
            )
            rotation4 = rotation_to_transform4(rotation)
            rotated_bounds = transform_bounds(np.asarray(mesh.bounds, dtype=np.float64), rotation4)
            normalize = translation_matrix([0.0, 0.0, -float(rotated_bounds[0, 2])])
            mesh = apply_min_overhang_orientation(mesh, rotation)
            pipeline_matrix = normalize @ rotation4 @ pipeline_matrix
            pipeline_steps.extend(
                [
                    transform_step("support_orientation", matrix=rotation4),
                    transform_step("z_normalize", matrix=normalize),
                ]
            )
        console.print(f"Overhang score after orient: {score:.1f}")

    rot_samples = (
        int(args.rotation_samples)
        if args.rotation_samples is not None
        else ORIENTATION_SAMPLES_DEFAULT
    )
    rot_seed = ORIENTATION_SEED_DEFAULT

    with profiler.stage("layout orientation"):
        ok, transform, fw, fh = profiler.profiled_call(
            "fill.layout_orientation",
            select_layout_transform,
            mesh,
            px,
            py,
            pz,
            gap,
            random_samples=rot_samples,
            seed=rot_seed,
            any_rotation=args.any_rotation,
            policy=orientation_policy,
            scale_tolerance=scale_tolerance,
        )
    if not ok:
        console.print("[red]Part does not fit on the bed in any orientation.[/red]")
        return finish_profile(profiler, console, 1)

    mesh.apply_transform(transform)
    pipeline_matrix = transform @ pipeline_matrix
    pipeline_steps.append(transform_step("layout_orientation", matrix=transform))
    _, _, dz = mesh_footprint_xy(mesh)
    console.print(f"Part footprint: {fw:.2f} x {fh:.2f} mm, height: {dz:.2f} mm")

    with profiler.stage("packing"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Packing copies…", total=1)
        plate = profiler.profiled_call("fill.packing", _max_copies_on_plate, fw, fh, px, py, gap)
        progress.advance(ptask)
    if plate is None:
        console.print("[red]Part does not fit on the bed.[/red]")
        return finish_profile(profiler, console, 1)

    n = len(plate.rects)
    console.print(f"Copies that fit: {n}")

    if args.dry_run:
        return finish_profile(profiler, console, 0)

    with profiler.stage("export"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Exporting copies…", total=max(1, n))
        args.output_dir.mkdir(parents=True, exist_ok=True)

        out_3mf = args.output_dir / "fill_plate.3mf"
        out_json = args.output_dir / "fill_plate.json"
        scene = trimesh.Scene()
        base_mesh = mesh.copy()
        shared_normalize = translation_matrix(-np.asarray(base_mesh.bounds[0], dtype=np.float64))
        base_mesh.apply_transform(shared_normalize)
        geom_name = inp.stem
        scene.geometry[geom_name] = base_mesh
        for i, r in enumerate(plate.rects):
            node_name = f"copy_{i:02d}"
            node_transform, _steps = shared_geometry_placement_transform_for_mesh(base_mesh, r)
            scene.graph.update(
                frame_to=node_name,
                matrix=node_transform,
                geometry=geom_name,
                geometry_flags={"visible": True},
            )
            progress.advance(ptask)
        scene.export(str(out_3mf))

        manifest = {
            "source": inp.name,
            "copies": n,
            "bed_mm": [px, py, pz],
            "gap_mm": gap,
            "parts": [
                {
                    "copy": i,
                    "x_mm": r.x,
                    "y_mm": r.y,
                    "footprint_w_mm": r.width,
                    "footprint_h_mm": r.height,
                    "rotation_deg": r.rotation_deg,
                }
                for i, r in enumerate(plate.rects)
            ],
        }
        out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        transform_parts: list[dict] = []
        for i, r in enumerate(plate.rects):
            placement_transform, placement_steps = shared_geometry_placement_transform_for_mesh(
                base_mesh,
                r,
            )
            source_to_export = placement_transform @ shared_normalize @ pipeline_matrix
            transform_parts.append(
                transform_entry(
                    index=i,
                    plate_index=0,
                    plate_file=out_3mf,
                    source_path=inp,
                    source_name=inp.name,
                    output_name=f"copy_{i:02d}",
                    output_file=out_3mf,
                    source_bounds_mm=source_bounds,
                    final_bounds_mm=bounds_to_list(
                        transform_bounds(source_bounds_np, source_to_export)
                    ),
                    source_to_export_matrix=source_to_export,
                    steps=[
                        *pipeline_steps,
                        transform_step("shared_geometry_normalize", matrix=shared_normalize),
                        *placement_steps,
                    ],
                    plate_x_mm=r.x,
                    plate_y_mm=r.y,
                    rotation_deg=r.rotation_deg,
                )
            )
        write_transform_log(
            args.output_dir / "transforms.json",
            command="fill",
            output_files=[out_3mf, out_json],
            parts=transform_parts,
            metadata={"copies": n, "source": str(inp)},
        )
        write_command_repair_report(
            args.output_dir,
            command="fill",
            reports=[repair_report],
            dry_run=args.dry_run,
        )

    console.print(f"Wrote {out_3mf}  ({n} copies)")
    console.print(f"Wrote {out_json}")
    return finish_profile(profiler, console, 0)
