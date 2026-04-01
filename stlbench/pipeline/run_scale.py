from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.schema import AppSettings
from stlbench.core.fit import (
    Method,
    aabb_edge_lengths,
    compute_global_scale,
    limiting_part_index,
    printer_dims_with_margin,
)
from stlbench.core.orientation import (
    best_orientation_for_conservative_fit,
    best_orientation_for_sorted_fit,
    mesh_vertices_for_orientation,
)
from stlbench.hollow.voxel_shell import hollow_mesh_voxel_shell
from stlbench.packing.shelf import build_packable_parts, greedy_shelf_plates
from stlbench.pipeline.common import load_named_meshes, resolve_printer, rotation_to_4x4


@dataclass
class ScaleRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    settings: AppSettings | None
    printer_xyz: tuple[float, float, float] | None
    margin: float | None
    supports_scale: float | None
    method: str | None
    orientation: str | None
    rotation_samples: int | None
    no_upscale: bool
    dry_run: bool
    recursive: bool
    suffix: str
    no_packing_report: bool
    hollow_override: bool | None


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
    supports_scale = (
        float(args.supports_scale)
        if args.supports_scale is not None
        else (st.scaling.supports_scale if st is not None else 1.0)
    )

    method_s: Method = args.method or "sorted"  # type: ignore[assignment]
    orient: str
    if st is not None and args.orientation is None:
        orient = st.orientation.mode
    elif args.orientation is not None:
        orient = args.orientation
    else:
        orient = "axis"

    if args.rotation_samples is not None:
        rot_samples = int(args.rotation_samples)
    elif st is not None:
        rot_samples = st.orientation.samples
    else:
        rot_samples = 2048

    seed = st.orientation.seed if st else 0

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
    p_sorted_calc: tuple[float, float, float] = tuple(sorted((px, py, pz)))  # type: ignore[assignment]

    part_names: list[str] = list(_loaded_names)
    meshes: list[trimesh.Trimesh] = list(_loaded_meshes)
    parts_dims: list[tuple[float, float, float]] = []
    file_dims: list[tuple[float, float, float]] = []
    rotations: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for mesh in meshes:
        fb = np.asarray(mesh.bounds)
        file_d = aabb_edge_lengths(fb)
        file_dims.append(file_d)

        if orient == "free":
            verts = mesh_vertices_for_orientation(mesh)
            if method_s == "sorted":
                res = best_orientation_for_sorted_fit(
                    verts,
                    p_sorted_calc,
                    rot_samples,
                    rng,
                    identity_baseline=file_d,
                )
            else:
                res = best_orientation_for_conservative_fit(
                    verts,
                    min(px, py, pz),
                    rot_samples,
                    rng,
                    identity_baseline=file_d,
                )
            parts_dims.append(res.extents)
            rotations.append(res.rotation)
        else:
            parts_dims.append(file_d)
            rotations.append(np.eye(3, dtype=np.float64))

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
    s_final = s_after * supports_scale
    lim_i = limiting_part_index(reports, s_max)
    limiter = reports[lim_i].name

    hollow_on = (
        (st.hollow.enabled if st else False)
        if args.hollow_override is None
        else args.hollow_override
    )
    if hollow_on and st and st.hollow.backend == "none":
        console.print("[yellow]hollow.enabled but backend=none; skipping hollow.[/yellow]")
        hollow_on = False
    if hollow_on and st is None:
        console.print(
            "[red]Для --hollow укажите --config с секцией [hollow] (толщина, voxel).[/red]"
        )
        return 2

    if st and st.printer.name:
        console.print(f"Printer profile: {st.printer.name}")
    console.print(f"Printer (after margin): {px:.4f} x {py:.4f} x {pz:.4f}")
    console.print(f"Method: {method_s}, orientation: {orient}")
    if orient == "free":
        console.print(f"Rotation samples: {rot_samples}, seed: {seed}")
    console.print(f"s_max (geometry fit): {s_max:.6f}")
    console.print(f"supports_scale: {supports_scale:.6f}")
    console.print(f"s_final (applied): {s_final:.6f}")
    if args.no_upscale:
        console.print("(capped by --no-upscale before supports_scale)")
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

    packing_report = True
    if args.no_packing_report:
        packing_report = False
    elif st is not None:
        packing_report = st.packing.report

    if packing_report:
        console.print()
        console.print("Группировка по столу (эвристика): наименьшее ребро AABB → Z; полки по XY.")
        scaled_dims = [(r.dx * s_final, r.dy * s_final, r.dz * s_final) for r in reports]
        packable, too_tall = build_packable_parts(
            [r.name for r in reports], scaled_dims, px, py, pz
        )
        if too_tall:
            console.print("Не помещаются по высоте/основанию (после s_final):")
            for n in too_tall:
                console.print(f"  - {n}")
        plates = greedy_shelf_plates(packable, px, py)
        if not plates and not too_tall:
            console.print("Нет деталей для группировки.")
        for i, group in enumerate(plates, start=1):
            console.print(f"Пластина {i}: {', '.join(group)}")

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

        rot = rotations[idx]
        if not np.allclose(rot, np.eye(3)):
            scaled.apply_transform(rotation_to_4x4(rot))

        scaled.apply_scale(s_final)

        if hollow_on and st and st.hollow.backend == "open3d_voxel":
            try:
                scaled = hollow_mesh_voxel_shell(
                    scaled,
                    wall_thickness_mm=st.hollow.wall_thickness_mm,
                    voxel_mm=st.hollow.voxel_mm,
                )
            except Exception as e:
                console.print(f"[red]Hollow failed for {path.name}: {e}[/red]")
                return 1

        try:
            scaled.export(out_path)
        except (OSError, ValueError) as e:
            console.print(f"[red]Failed to export {out_path}: {e}[/red]")
            return 1
        console.print(f"Wrote {out_path}")

    return 0
