from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.mesh_repair import repair_report_step
from stlbench.export.plate import export_plate_3mf, mesh_footprint_xy
from stlbench.export.transform_log import (
    bounds_to_list,
    mesh_bounds,
    placement_transform_for_mesh,
    transform_bounds,
    transform_entry,
    transform_step,
    write_transform_log,
)
from stlbench.packing.layout_orientation import select_layout_transform
from stlbench.packing.polygon_footprint import mesh_to_packing_shadow
from stlbench.packing.polygon_pack import pack_polygons_on_plates
from stlbench.packing.rectpack_plate import int_bin_dims_mm
from stlbench.pipeline.common import (
    finish_profile,
    load_named_meshes_with_repair,
    repair_cache_dir_for_output,
    resolve_edge_margin,
    resolve_gap,
    resolve_orientation_policy,
    resolve_orientation_scale_tolerance,
    resolve_printer,
    resolve_repair_cache_enabled,
    resolve_repair_options,
    resolve_settings,
    write_command_repair_report,
)
from stlbench.pipeline.progress import make_progress
from stlbench.profiling import ProfileOptions, make_profiler


@dataclass
class LayoutRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    recursive: bool
    dry_run: bool
    cleanup: bool = False
    repair: bool = False
    repair_cache: bool = True
    any_rotation: bool = False
    orientation_policy: str | None = None
    orientation_scale_tolerance: float | None = None
    rotation_samples: int | None = None
    profile_options: ProfileOptions | None = None
    edge_margin_mm: float | None = None
    progress: bool = True


def run_layout(args: LayoutRunArgs) -> int:
    console = Console(stderr=True)
    profile_metadata: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "dry_run": args.dry_run,
        "any_rotation": args.any_rotation,
    }
    profiler = make_profiler(
        command="layout",
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
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    gap = resolve_gap(args.gap_mm, st)
    edge_margin = resolve_edge_margin(args.edge_margin_mm, st)
    epx = px - 2.0 * edge_margin
    epy = py - 2.0 * edge_margin
    try:
        orientation_policy = resolve_orientation_policy(args.orientation_policy)
        scale_tolerance = resolve_orientation_scale_tolerance(args.orientation_scale_tolerance)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

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

    n_parts = len(meshes)
    dims_list: list[tuple[float, float, float]] = []
    for m in meshes:
        dx, dy, dz = mesh_footprint_xy(m)
        dims_list.append((dx, dy, dz))

    rot_samples = (
        int(args.rotation_samples)
        if args.rotation_samples is not None
        else ORIENTATION_SAMPLES_DEFAULT
    )
    rot_seed = ORIENTATION_SEED_DEFAULT

    bw, bh = int_bin_dims_mm(epx, epy)
    bad_layout: list[tuple[str, float, float, float]] = []
    layout_plans: list[tuple[np.ndarray, float, float] | None] = []

    with (
        profiler.stage("orientation selection"),
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Finding orientations…", total=n_parts)
        for m, name in zip(meshes, names, strict=True):
            dx, dy, dz = mesh_footprint_xy(m)
            ok, t, fw, fh = profiler.profiled_call(
                "layout.select_orientation",
                select_layout_transform,
                m,
                epx,
                epy,
                pz,
                gap,
                random_samples=rot_samples,
                seed=rot_seed,
                any_rotation=args.any_rotation,
                policy=orientation_policy,
                scale_tolerance=scale_tolerance,
            )
            if not ok:
                bad_layout.append((name, dx, dy, dz))
                layout_plans.append(None)
            else:
                layout_plans.append((t, fw, fh))
            progress.update(ptask, advance=1, description=f"Orient: {name}")

    if bad_layout:
        console.print(
            "[red]No bed-fitting orientation for these parts: 90° permutations + "
            f"{rot_samples} random rotations (same idea as scale; seed={rot_seed}). "
            f"Bed {bw}x{bh} mm XY, Pz={pz:.2f} mm, gap={gap:.2f} mm.[/red]"
        )
        for name, dx, dy, dz in bad_layout:
            console.print(f"  [red]{name}[/red]: file-axis AABB {dx:.2f}x{dy:.2f}x{dz:.2f} mm")
        console.print(
            "[dim]Try smaller packing.gap_mm / scaling.post_fit_scale or split the model.[/dim]"
        )
        return finish_profile(profiler, console, 1)

    oriented_meshes: list[trimesh.Trimesh] = []
    for m, plan in zip(meshes, layout_plans, strict=True):
        assert plan is not None
        t, _fw, _fh = plan
        m2 = m.copy()
        m2.apply_transform(t)
        oriented_meshes.append(m2)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.cleanup:
        with profiler.stage("cleanup"):
            for i, m in enumerate(oriented_meshes):
                cleaned, n_rem = remove_small_components(m)
                if n_rem:
                    oriented_meshes[i] = cleaned
                    console.print(
                        f"[dim]cleanup: {names[i]} — removed {n_rem} tiny component(s)[/dim]"
                    )

    shadows = []
    with (
        profiler.stage("footprint computation"),
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Computing footprints…", total=n_parts)
        for i, m in enumerate(oriented_meshes):
            shadows.append(profiler.profiled_call("layout.footprint", mesh_to_packing_shadow, m))
            progress.update(ptask, advance=1, description=f"Footprint: {names[i]}")

    with profiler.stage("packing"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Packing…", total=n_parts)
        part_heights = [mesh_footprint_xy(m)[2] for m in oriented_meshes]
        packing_metadata: dict[str, object] = {}
        plates = profiler.profiled_call(
            "layout.packing",
            pack_polygons_on_plates,
            shadows,
            px,
            py,
            gap_mm=gap,
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
        return finish_profile(profiler, console, 0)

    def _export_plate(pl) -> Path:
        out_3mf = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
        out_js = args.output_dir / f"plate_{pl.index + 1:02d}.json"
        export_plate_3mf(oriented_meshes, pl, out_3mf, names=list(names), out_manifest=out_js)
        return out_3mf

    output_files: list[Path] = []
    with profiler.stage("export"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Exporting plates…", total=len(plates))
        for pl in plates:
            out_path = profiler.profiled_call("layout.export", _export_plate, pl)
            console.print(f"Wrote {out_path}")
            output_files.append(out_path)
            progress.advance(ptask)

    transform_parts: list[dict] = []
    for pl in plates:
        plate_file = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
        for rect in pl.rects:
            part_index = rect.part_index
            plan = layout_plans[part_index]
            assert plan is not None
            layout_transform = plan[0]
            placement_transform, placement_steps = placement_transform_for_mesh(
                oriented_meshes[part_index], rect
            )
            source_to_export = placement_transform @ layout_transform
            source_bounds = np.asarray(meshes[part_index].bounds, dtype=np.float64)
            transform_parts.append(
                transform_entry(
                    index=part_index,
                    plate_index=pl.index,
                    plate_file=plate_file,
                    source_path=paths[part_index],
                    source_name=names[part_index],
                    output_name=names[part_index],
                    output_file=plate_file,
                    source_bounds_mm=mesh_bounds(meshes[part_index]),
                    final_bounds_mm=bounds_to_list(
                        transform_bounds(source_bounds, source_to_export)
                    ),
                    source_to_export_matrix=source_to_export,
                    steps=[
                        *(
                            [repair_report_step(repair_reports[part_index])]
                            if repair_reports[part_index].enabled
                            else []
                        ),
                        transform_step("layout_orientation", matrix=layout_transform),
                        *placement_steps,
                    ],
                    plate_x_mm=rect.x,
                    plate_y_mm=rect.y,
                    rotation_deg=rect.rotation_deg,
                )
            )
    write_transform_log(
        args.output_dir / "transforms.json",
        command="layout",
        output_files=output_files,
        parts=transform_parts,
    )
    write_command_repair_report(
        args.output_dir,
        command="layout",
        reports=repair_reports,
        dry_run=args.dry_run,
    )

    return finish_profile(profiler, console, 0)
