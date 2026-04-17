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
from stlbench.config.schema import AppSettings
from stlbench.core.fit import (
    Method,
    PartScaleReport,
    aabb_edge_lengths,
    compute_global_scale,
    limiting_part_index,
)
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.pipeline.common import load_named_meshes, n_workers, resolve_printer
from stlbench.pipeline.mesh_io import load_mesh


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
    scale_factor: float | None = None
    rotation_samples: int | None = None
    no_upscale: bool = False
    dry_run: bool = False
    recursive: bool = False
    suffix: str = ""
    verbose: bool = False


def run_scale(args: ScaleRunArgs) -> int:
    console = Console(stderr=True)
    st = args.settings

    # ── Validation ────────────────────────────────────────────────────────────
    if args.maximize and not args.any_rotation:
        console.print(
            "[red]--maximize requires --any-rotation "
            "(full SO(3) search requires unrestricted orientation)[/red]"
        )
        return 2

    # ── Printer / post_fit_scale ─────────────────────────────────────────────
    # scale_factor bypasses fit-to-printer entirely, so printer dims are optional.
    if args.scale_factor is None:
        try:
            prx, pry, prz = resolve_printer(args.printer_xyz, st)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 2
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

    method_s: Method = args.method or "sorted"  # type: ignore[assignment]

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
        return 2

    loaded = load_named_meshes(input_dir, args.recursive, console)
    if loaded is None:
        return 1
    paths, part_names, meshes = loaded
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
        )

    _n = n_workers(len(meshes))
    if args.verbose:
        console.print(f"[dim]orient: {_n} workers for {len(meshes)} meshes[/dim]")
    with ThreadPoolExecutor(max_workers=_n) as pool:
        _results = list(pool.map(_select_orient, meshes))
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
            return 1

        s_after = min(1.0, s_max) if args.no_upscale else s_max
        s_final = s_after * post_fit_scale
        lim_i = limiting_part_index(reports, s_max)
        limiter = reports[lim_i].name

        if st and st.printer.name:
            console.print(f"Printer profile: {st.printer.name}")
        console.print(f"Printer: {px:.4f} x {py:.4f} x {pz:.4f}")
        console.print(f"Method: {method_s}")
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
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

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
        mesh = load_mesh(path)
        t4 = transforms[idx]
        if not np.allclose(t4, np.eye(4)):
            mesh.apply_transform(t4)
        mesh.apply_scale(s_final)
        if not np.allclose(t4, np.eye(4)):
            # Rotation around world origin shifts the mesh's XY position; restore
            # AABB min to origin so the output sits on the build plate as expected.
            b = np.asarray(mesh.bounds)
            if not np.allclose(b[0], np.zeros(3), atol=1e-6):
                mesh.apply_translation(-b[0])

        try:
            mesh.export(out_path)
        except (OSError, ValueError) as e:
            console.print(f"[red]Failed to export {out_path}: {e}[/red]")
            return 1
        console.print(f"Wrote {out_path}")

    return 0
