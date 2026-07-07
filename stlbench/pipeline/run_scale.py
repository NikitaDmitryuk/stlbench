from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.defaults import (
    ORIENTATION_SAMPLES_DEFAULT,
    ORIENTATION_SEED_DEFAULT,
)
from stlbench.config.enums import ScaleFitMethod, coerce_enum
from stlbench.config.schema import AppSettings
from stlbench.core.fit import (
    PartScaleReport,
    aabb_edge_lengths,
    compute_global_scale,
    limiting_part_index,
)
from stlbench.core.mesh_repair import repair_report_step
from stlbench.export.transform_log import (
    mesh_bounds,
    transform_entry,
    transform_step,
    translation_matrix,
    uniform_scale_matrix,
    write_transform_log,
)
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.pipeline.common import (
    finish_profile,
    load_mesh_with_repair,
    load_named_meshes_with_repair,
    n_workers,
    repair_cache_dir_for_output,
    resolve_orientation_policy,
    resolve_orientation_scale_tolerance,
    resolve_printer,
    resolve_repair_cache_enabled,
    resolve_repair_options,
    write_command_repair_report,
)
from stlbench.pipeline.progress import make_progress
from stlbench.profiling import ProfileOptions, make_profiler


@dataclass
class ScaleRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    settings: AppSettings | None
    printer_xyz: tuple[float, float, float] | None
    post_fit_scale: float | None
    method: str | None
    any_rotation: bool = False
    maximize: bool = False
    orientation_policy: str | None = None
    orientation_scale_tolerance: float | None = None
    scale_factor: float | None = None
    rotation_samples: int | None = None
    no_upscale: bool = False
    dry_run: bool = False
    recursive: bool = False
    suffix: str = ""
    verbose: bool = False
    profile_options: ProfileOptions | None = None
    repair: bool = False
    repair_cache: bool = True
    progress: bool = True


def run_scale(args: ScaleRunArgs) -> int:
    console = Console(stderr=True)
    profiler = make_profiler(
        command="scale",
        output_base=args.output_dir,
        options=args.profile_options,
        metadata={
            "input_dir": str(args.input_dir),
            "dry_run": args.dry_run,
            "any_rotation": args.any_rotation,
            "maximize": args.maximize,
        },
    )
    profiler.start()
    st = args.settings
    args.progress = args.progress and (st.ui.progress if st is not None else True)
    repair_options = resolve_repair_options(args.repair, st)
    repair_cache_dir = repair_cache_dir_for_output(
        args.output_dir,
        resolve_repair_cache_enabled(args.repair_cache, st) and not args.dry_run,
    )

    # ── Validation ────────────────────────────────────────────────────────────
    if args.maximize and not args.any_rotation:
        console.print(
            "[red]--maximize requires --any-rotation "
            "(full SO(3) search requires unrestricted orientation)[/red]"
        )
        return finish_profile(profiler, console, 2)
    try:
        orientation_policy = resolve_orientation_policy(args.orientation_policy)
        scale_tolerance = resolve_orientation_scale_tolerance(args.orientation_scale_tolerance)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    # ── Printer / post_fit_scale ─────────────────────────────────────────────
    # scale_factor bypasses fit-to-printer entirely, so printer dims are optional.
    if args.scale_factor is None:
        try:
            prx, pry, prz = resolve_printer(args.printer_xyz, st)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return finish_profile(profiler, console, 2)
    else:
        # Printer dims may still be provided (used only for display / validation).
        try:
            prx, pry, prz = resolve_printer(args.printer_xyz, st)
        except ValueError:
            prx, pry, prz = 0.0, 0.0, 0.0

    post_fit_scale = (
        float(args.post_fit_scale)
        if args.post_fit_scale is not None
        else (st.scaling.post_fit_scale if st is not None else 1.0)
    )

    try:
        method_s = coerce_enum(
            ScaleFitMethod,
            args.method or ScaleFitMethod.SORTED,
            "--method",
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return finish_profile(profiler, console, 2)

    rot_samples = (
        int(args.rotation_samples)
        if args.rotation_samples is not None
        else ORIENTATION_SAMPLES_DEFAULT
    )
    seed = ORIENTATION_SEED_DEFAULT

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.is_dir():
        console.print(f"[red]Input is not a directory: {input_dir}[/red]")
        return finish_profile(profiler, console, 2)

    with profiler.stage("load meshes"):
        loaded = load_named_meshes_with_repair(
            input_dir,
            args.recursive,
            console,
            repair_options,
            repair_cache_dir,
        )
    if loaded is None:
        return finish_profile(profiler, console, 1)
    paths, part_names, meshes, repair_reports = loaded
    del loaded

    px, py, pz = prx, pry, prz

    file_dims = [aabb_edge_lengths(np.asarray(m.bounds)) for m in meshes]

    # ── Rotation search ───────────────────────────────────────────────────────
    def _select_orient(mesh: trimesh.Trimesh) -> tuple[np.ndarray, tuple[float, float, float]]:
        return select_orientation_for_scale(
            mesh,
            px,
            py,
            pz,
            method_s,
            any_rotation=args.any_rotation,
            maximize=args.maximize,
            random_samples=rot_samples,
            seed=seed,
            policy=orientation_policy,
            scale_tolerance=scale_tolerance,
        )

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
        for result in profiler.map(pool, "scale.orientation", _select_orient, meshes):
            _results.append(result)
            progress.advance(ptask)
    transforms: list[np.ndarray] = [r[0] for r in _results]
    parts_dims: list[tuple[float, float, float]] = [r[1] for r in _results]

    del meshes  # free mesh data; export will re-load from disk

    file_dims_for_report = list(file_dims)

    # ── Scale computation ─────────────────────────────────────────────────────
    if args.scale_factor is not None:
        s_final = float(args.scale_factor) * post_fit_scale
        s_max = float(args.scale_factor)
        reports: list[PartScaleReport] = []
        limiter = "(explicit factor)"

        if st and st.printer.name:
            console.print(f"Printer profile: {st.printer.name}")
        console.print(f"Scale factor: {args.scale_factor:.6f}")
        console.print(f"post_fit_scale: {post_fit_scale:.6f}")
        console.print(f"s_final (applied): {s_final:.6f}")
    else:
        with profiler.stage("scale computation"):
            try:
                s_max, reports = compute_global_scale(
                    (px, py, pz),
                    parts_dims,
                    part_names,
                    method_s,
                    file_dims=file_dims_for_report,
                )
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                return finish_profile(profiler, console, 1)

        s_after = min(1.0, s_max) if args.no_upscale else s_max
        s_final = s_after * post_fit_scale
        lim_i = limiting_part_index(reports, s_max)
        limiter = reports[lim_i].name

        if st and st.printer.name:
            console.print(f"Printer profile: {st.printer.name}")
        console.print(f"Printer: {px:.4f} x {py:.4f} x {pz:.4f}")
        console.print(f"Method: {method_s.value}")
        console.print(
            f"Orientation policy: {orientation_policy.value}, tolerance: {scale_tolerance:.3f}"
        )
        if args.any_rotation:
            mode = "maximize" if args.maximize else "axis-permutations"
            console.print(f"Rotation: {mode}")
            if args.maximize:
                console.print(f"Rotation samples: {rot_samples}, seed: {seed}")
        else:
            console.print("Rotation: Z-only")
        console.print(f"s_max (geometry fit): {s_max:.6f}")
        console.print(f"post_fit_scale: {post_fit_scale:.6f}")
        console.print(f"s_final (applied): {s_final:.6f}")
        if args.no_upscale:
            console.print("(capped by --no-upscale before post_fit_scale)")
        console.print(f"Limiting part: {limiter}")

        table = Table(show_header=True, header_style="bold")
        table.add_column("part", max_width=42)
        table.add_column("dx", justify="right")
        table.add_column("dy", justify="right")
        table.add_column("dz", justify="right")
        table.add_column("s_part", justify="right")
        if args.any_rotation:
            table.add_column("fx")
            table.add_column("fy")
            table.add_column("fz")
        for r in reports:
            row = [
                r.name,
                f"{r.dx:.4f}",
                f"{r.dy:.4f}",
                f"{r.dz:.4f}",
                f"{r.s_limit:.6f}",
            ]
            if args.any_rotation and r.file_dx is not None:
                row.extend([f"{r.file_dx:.3f}", f"{r.file_dy:.3f}", f"{r.file_dz:.3f}"])
            elif args.any_rotation:
                row.extend(["", "", ""])
            table.add_row(*row)
        console.print(table)

    if args.dry_run:
        return finish_profile(profiler, console, 0)

    with profiler.stage("export"), make_progress(console, enabled=args.progress) as progress:
        ptask = progress.add_task("Exporting meshes…", total=len(paths))
        output_dir.mkdir(parents=True, exist_ok=True)
        transform_parts: list[dict] = []
        output_files: list[Path] = []

        for idx, path in enumerate(paths):
            if args.recursive:
                rel_parent = path.parent.relative_to(input_dir)
                out_sub = output_dir / rel_parent
                out_sub.mkdir(parents=True, exist_ok=True)
                out_dir = out_sub
            else:
                out_dir = output_dir
            stem = path.stem + args.suffix
            out_path = out_dir / f"{stem}.stl"

            if args.verbose:
                console.print(f"[dim]  [{idx + 1}/{len(paths)}] exporting {path.name}[/dim]")
            mesh, export_repair_report = load_mesh_with_repair(
                path,
                repair_options,
                source_name=part_names[idx],
                repair_cache_dir=repair_cache_dir,
            )
            source_bounds = mesh_bounds(mesh)
            source_to_output = np.eye(4, dtype=np.float64)
            steps = (
                [repair_report_step(export_repair_report)] if export_repair_report.enabled else []
            )
            t4 = transforms[idx]
            if not np.allclose(t4, np.eye(4)):
                mesh.apply_transform(t4)
                source_to_output = t4 @ source_to_output
                steps.append(transform_step("scale_orientation", matrix=t4))
            scale_matrix = uniform_scale_matrix(s_final)
            mesh.apply_scale(s_final)
            source_to_output = scale_matrix @ source_to_output
            steps.append(
                transform_step("scale", matrix=scale_matrix, params={"scale_factor": s_final})
            )
            if not np.allclose(t4, np.eye(4)):
                # Rotation around world origin shifts the mesh's XY position; restore
                # AABB min to origin so the output sits on the build plate as expected.
                b = np.asarray(mesh.bounds)
                if not np.allclose(b[0], np.zeros(3), atol=1e-6):
                    translate = translation_matrix(-b[0])
                    mesh.apply_translation(-b[0])
                    source_to_output = translate @ source_to_output
                    steps.append(transform_step("normalize_to_origin", matrix=translate))

            try:
                mesh.export(out_path)
            except (OSError, ValueError) as e:
                console.print(f"[red]Failed to export {out_path}: {e}[/red]")
                return finish_profile(profiler, console, 1)
            console.print(f"Wrote {out_path}")
            progress.advance(ptask)
            output_files.append(out_path)
            transform_parts.append(
                transform_entry(
                    index=idx,
                    source_path=path,
                    source_name=part_names[idx],
                    output_name=out_path.name,
                    output_file=out_path,
                    source_bounds_mm=source_bounds,
                    final_bounds_mm=mesh_bounds(mesh),
                    source_to_export_matrix=source_to_output,
                    steps=steps,
                    scale_factor=s_final,
                )
            )

        write_transform_log(
            output_dir / "transforms.json",
            command="scale",
            output_files=output_files,
            parts=transform_parts,
        )
        write_command_repair_report(
            output_dir,
            command="scale",
            reports=repair_reports,
            dry_run=args.dry_run,
        )

    return finish_profile(profiler, console, 0)
