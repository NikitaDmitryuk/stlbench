from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.schema import AppSettings
from stlbench.core.mesh_repair import repair_report_step
from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    overhang_score,
    rotation_to_transform4,
)
from stlbench.export.transform_log import (
    mesh_bounds,
    transform_bounds,
    transform_entry,
    transform_step,
    translation_matrix,
    write_transform_log,
)
from stlbench.pipeline.common import (
    finish_profile,
    load_named_meshes_with_repair,
    n_workers,
    repair_cache_dir_for_output,
    resolve_printer,
    resolve_repair_cache_enabled,
    resolve_repair_options,
    resolve_settings,
    write_command_repair_report,
)
from stlbench.pipeline.progress import make_progress
from stlbench.profiling import ProfileOptions, make_profiler

_IDENTITY = np.eye(3, dtype=np.float64)


@dataclass
class OrientRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    settings: AppSettings | None
    printer_xyz: tuple[float, float, float] | None
    overhang_threshold_deg: float
    n_candidates: int
    dry_run: bool
    recursive: bool
    suffix: str
    verbose: bool = False
    profile_options: ProfileOptions | None = None
    repair: bool = False
    repair_cache: bool = True
    progress: bool = True


def run_orient(args: OrientRunArgs) -> int:
    console = Console(stderr=True)
    profiler = make_profiler(
        command="orient",
        output_base=args.output_dir,
        options=args.profile_options,
        metadata={"input_dir": str(args.input_dir), "dry_run": args.dry_run},
    )
    profiler.start()
    st = args.settings or resolve_settings(args.config_path)
    args.progress = args.progress and (st.ui.progress if st is not None else True)
    repair_options = resolve_repair_options(args.repair, st)
    repair_cache_dir = repair_cache_dir_for_output(
        args.output_dir,
        resolve_repair_cache_enabled(args.repair_cache, st) and not args.dry_run,
    )

    # Printer dims are optional for orient: when provided, orientations that
    # don't fit the build volume receive a heavy penalty.
    printer_dims: tuple[float, float, float] | None = None
    if args.printer_xyz is not None or st is not None:
        try:
            px, py, pz = resolve_printer(args.printer_xyz, st)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return finish_profile(profiler, console, 2)
        printer_dims = (px, py, pz)
        if st and st.printer.name:
            console.print(f"Printer: {st.printer.name}")
        console.print(f"Build volume: {px:.1f} × {py:.1f} × {pz:.1f} mm")

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

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("score before", justify="right")
    table.add_column("score after", justify="right")
    table.add_column("improvement", justify="right")

    def _orient_one(mesh: trimesh.Trimesh) -> tuple[float, float, float, np.ndarray]:
        sb = overhang_score(mesh, _IDENTITY, args.overhang_threshold_deg)
        rotation, sa = find_min_overhang_rotation(
            mesh,
            overhang_threshold_deg=args.overhang_threshold_deg,
            n_candidates=args.n_candidates,
            printer_dims=printer_dims,
        )
        pct = (sb - sa) / max(abs(sb), 1.0) * 100.0
        return sb, sa, pct, rotation

    # Pre-populate trimesh lazy caches sequentially before threading.
    for mesh in meshes:
        _ = mesh.face_normals
        _ = mesh.area_faces

    _n = n_workers(len(meshes))
    if args.verbose:
        console.print(f"[dim]orient: {_n} workers for {len(meshes)} meshes[/dim]")
    with (
        profiler.stage("overhang search"),
        ThreadPoolExecutor(max_workers=_n) as pool,
        make_progress(console, enabled=args.progress) as progress,
    ):
        ptask = progress.add_task("Finding support orientations…", total=len(meshes))
        _or = []
        for result in profiler.map(pool, "orient.search", _orient_one, meshes):
            _or.append(result)
            progress.advance(ptask)

    results = []
    for path, name, mesh, (sb, sa, pct, rotation) in zip(paths, names, meshes, _or, strict=True):
        table.add_row(name, f"{sb:.1f}", f"{sa:.1f}", f"{pct:+.1f}%")
        results.append((path, name, mesh, rotation))

    console.print(table)
    console.print(f"Overhang threshold: {args.overhang_threshold_deg}°")

    if args.dry_run:
        return finish_profile(profiler, console, 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _export_one(
        item: tuple[Path, str, trimesh.Trimesh, np.ndarray, int],
    ) -> tuple[Path, dict]:
        path, name, mesh, rotation, idx = item
        if args.recursive:
            rel_parent = path.parent.relative_to(args.input_dir)
            out_sub = args.output_dir / rel_parent
            out_sub.mkdir(parents=True, exist_ok=True)
            out_dir = out_sub
        else:
            out_dir = args.output_dir
        out_path = out_dir / f"{path.stem + args.suffix}.stl"
        source_bounds = mesh_bounds(mesh)
        rotation4 = rotation_to_transform4(rotation)
        rotated_bounds = transform_bounds(np.asarray(mesh.bounds, dtype=np.float64), rotation4)
        normalize = translation_matrix([0.0, 0.0, -float(rotated_bounds[0, 2])])
        source_to_output = normalize @ rotation4
        oriented = apply_min_overhang_orientation(mesh, rotation)
        oriented.export(out_path)
        entry = transform_entry(
            source_path=path,
            source_name=name,
            output_name=out_path.name,
            output_file=out_path,
            source_bounds_mm=source_bounds,
            final_bounds_mm=mesh_bounds(oriented),
            source_to_export_matrix=source_to_output,
            steps=[
                *([repair_report_step(repair_reports[idx])] if repair_reports[idx].enabled else []),
                transform_step("support_orientation", matrix=rotation4),
                transform_step("z_normalize", matrix=normalize),
            ],
        )
        return out_path, entry

    with profiler.stage("export"), make_progress(console, enabled=args.progress) as progress:
        try:
            _ne = n_workers(len(results))
            if args.verbose:
                console.print(f"[dim]export: {_ne} workers for {len(results)} meshes[/dim]")
            output_files: list[Path] = []
            transform_parts: list[dict] = []
            ptask = progress.add_task("Exporting meshes…", total=len(results))
            export_items = [
                (path, name, mesh, rotation, idx)
                for idx, (path, name, mesh, rotation) in enumerate(results)
            ]
            with ThreadPoolExecutor(max_workers=_ne) as pool:
                for out_path, entry in profiler.map(
                    pool,
                    "orient.export",
                    _export_one,
                    export_items,
                ):
                    console.print(f"Wrote {out_path}")
                    output_files.append(out_path)
                    transform_parts.append(entry)
                    progress.advance(ptask)
            write_transform_log(
                args.output_dir / "transforms.json",
                command="orient",
                output_files=output_files,
                parts=transform_parts,
            )
            write_command_repair_report(
                args.output_dir,
                command="orient",
                reports=repair_reports,
                dry_run=args.dry_run,
            )
        except (OSError, ValueError) as e:
            if args.verbose:
                console.print_exception()
            console.print(f"[red]Failed to export: {e}[/red]")
            return finish_profile(profiler, console, 1)

    return finish_profile(profiler, console, 0)
