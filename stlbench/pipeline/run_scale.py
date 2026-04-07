from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.defaults import (
    ORIENTATION_MODE_DEFAULT,
    ORIENTATION_SAMPLES_DEFAULT,
    ORIENTATION_SEED_DEFAULT,
)
from stlbench.config.schema import AppSettings
from stlbench.core.fit import (
    Method,
    aabb_edge_lengths,
    compute_global_scale,
    limiting_part_index,
    printer_dims_with_margin,
)
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.pipeline.common import load_named_meshes, resolve_printer


@dataclass
class ScaleRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    settings: AppSettings | None
    printer_xyz: tuple[float, float, float] | None
    margin: float | None
    post_fit_scale: float | None
    method: str | None
    orientation: str | None
    rotation_samples: int | None
    no_upscale: bool
    dry_run: bool
    recursive: bool
    suffix: str


def run_scale(args: ScaleRunArgs) -> int:
    console = Console(stderr=True)
    st = args.settings

    try:
        prx, pry, prz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    margin = (
        float(args.margin)
        if args.margin is not None
        else (st.scaling.bed_margin if st is not None else 0.0)
    )
    post_fit_scale = (
        float(args.post_fit_scale)
        if args.post_fit_scale is not None
        else (st.scaling.post_fit_scale if st is not None else 1.0)
    )

    method_s: Method = args.method or "sorted"  # type: ignore[assignment]
    orient = args.orientation if args.orientation is not None else ORIENTATION_MODE_DEFAULT

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
    paths, _loaded_names, _loaded_meshes = loaded

    px, py, pz = printer_dims_with_margin(prx, pry, prz, margin)

    part_names: list[str] = list(_loaded_names)
    meshes: list[trimesh.Trimesh] = list(_loaded_meshes)
    parts_dims: list[tuple[float, float, float]] = []
    file_dims: list[tuple[float, float, float]] = []
    transforms: list[np.ndarray] = []

    for mesh in meshes:
        fb = np.asarray(mesh.bounds)
        file_d = aabb_edge_lengths(fb)
        file_dims.append(file_d)

        if orient == "free":
            t4, ext = select_orientation_for_scale(
                mesh,
                px,
                py,
                pz,
                method_s,
                random_samples=rot_samples,
                seed=seed,
            )
            parts_dims.append(ext)
            transforms.append(t4)
        else:
            parts_dims.append(file_d)
            transforms.append(np.eye(4, dtype=np.float64))

    file_dims_for_report = list(file_dims) if orient == "free" else None

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
    console.print(f"Printer (after margin): {px:.4f} x {py:.4f} x {pz:.4f}")
    console.print(f"Method: {method_s}, orientation: {orient}")
    if orient == "free":
        console.print(f"Rotation samples: {rot_samples}, seed: {seed}")
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
    if orient == "free":
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
        if orient == "free" and r.file_dx is not None:
            row.extend([f"{r.file_dx:.3f}", f"{r.file_dy:.3f}", f"{r.file_dz:.3f}"])
        elif orient == "free":
            row.extend(["", "", ""])
        table.add_row(*row)
    console.print(table)

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (path, mesh) in enumerate(zip(paths, meshes, strict=True)):
        if args.recursive:
            rel_parent = path.parent.relative_to(input_dir)
            out_sub = output_dir / rel_parent
            out_sub.mkdir(parents=True, exist_ok=True)
            out_dir = out_sub
        else:
            out_dir = output_dir
        stem = path.stem + args.suffix
        out_path = out_dir / f"{stem}.stl"
        scaled = mesh.copy()

        t4 = transforms[idx]
        if not np.allclose(t4, np.eye(4)):
            scaled.apply_transform(t4)

        scaled.apply_scale(s_final)

        try:
            scaled.export(out_path)
        except (OSError, ValueError) as e:
            console.print(f"[red]Failed to export {out_path}: {e}[/red]")
            return 1
        console.print(f"Wrote {out_path}")

    return 0
