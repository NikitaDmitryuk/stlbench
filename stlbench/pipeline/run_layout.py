from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console

from stlbench.config.loader import load_app_settings
from stlbench.config.schema import AppSettings
from stlbench.export.plate import export_plate_stl, mesh_footprint_xy
from stlbench.packing.layout_orientation import select_layout_transform
from stlbench.packing.rectpack_plate import int_bin_dims_mm, pack_rectangles_on_plates
from stlbench.packing.shelf import build_packable_parts, greedy_shelf_plates
from stlbench.pipeline.mesh_io import collect_stl_paths, load_mesh


@dataclass
class LayoutRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    algorithm: str | None
    recursive: bool
    dry_run: bool


def run_layout(args: LayoutRunArgs) -> int:
    console = Console(stderr=True)
    st: AppSettings | None = None
    if args.config_path is not None:
        st = load_app_settings(args.config_path)

    if args.printer_xyz is not None:
        px, py, pz = args.printer_xyz
    elif st is not None:
        px, py, pz = (
            st.printer.width_mm,
            st.printer.depth_mm,
            st.printer.height_mm,
        )
    else:
        console.print("[red]Нужны --printer или --config с [printer].[/red]")
        return 2

    gap = (
        float(args.gap_mm)
        if args.gap_mm is not None
        else (st.packing.gap_mm if st is not None else 2.0)
    )
    algo = (
        args.algorithm
        if args.algorithm is not None
        else (st.packing.algorithm if st is not None else "rectpack")
    )

    paths = collect_stl_paths(args.input_dir, args.recursive)
    if not paths:
        console.print("[red]Нет .stl[/red]")
        return 1

    names: list[str] = []
    meshes: list[trimesh.Trimesh] = []
    for p in paths:
        m = load_mesh(p)
        names.append(
            str(p.relative_to(args.input_dir)) if p.is_relative_to(args.input_dir) else p.name
        )
        meshes.append(m)

    dims_list: list[tuple[float, float, float]] = []
    for m in meshes:
        dx, dy, dz = mesh_footprint_xy(m)
        dims_list.append((dx, dy, dz))

    rot_samples = st.orientation.samples if st is not None else 4096
    rot_seed = st.orientation.seed if st is not None else 0

    bw, bh = int_bin_dims_mm(px, py)
    bad_layout: list[tuple[str, float, float, float]] = []
    layout_plans: list[tuple[np.ndarray, float, float] | None] = []
    for m, name in zip(meshes, names, strict=True):
        dx, dy, dz = mesh_footprint_xy(m)
        ok, t, fw, fh = select_layout_transform(
            m,
            px,
            py,
            pz,
            gap,
            random_samples=rot_samples,
            seed=rot_seed,
        )
        if not ok:
            bad_layout.append((name, dx, dy, dz))
            layout_plans.append(None)
        else:
            layout_plans.append((t, fw, fh))

    if bad_layout:
        console.print(
            "[red]Для этих деталей не нашлось ориентации под стол: дискретные 90° + "
            f"{rot_samples} случайных поворотов (как при расчёте масштаба; seed={rot_seed}). "
            f"Стол {bw}×{bh} мм по XY, Pz={pz:.2f} мм, gap={gap:.2f} мм.[/red]"
        )
        for name, dx, dy, dz in bad_layout:
            console.print(
                f"  [red]{name}[/red]: AABB в файле (по осям) {dx:.2f}×{dy:.2f}×{dz:.2f} мм"
            )
        console.print(
            "[dim]Уменьшите packing.gap_mm / scaling.supports_scale или разрежьте модель.[/dim]"
        )
        return 1

    oriented_meshes: list[trimesh.Trimesh] = []
    footprints: list[tuple[float, float]] = []
    for m, plan in zip(meshes, layout_plans, strict=True):
        assert plan is not None
        t, fw, fh = plan
        m2 = m.copy()
        m2.apply_transform(t)
        oriented_meshes.append(m2)
        footprints.append((fw, fh))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if algo == "shelf":
        packable, bad = build_packable_parts(names, dims_list, px, py, pz)
        if bad:
            console.print("Не влезают по эвристике:", ", ".join(bad))
        groups = greedy_shelf_plates(packable, px, py)
        for i, g in enumerate(groups, 1):
            console.print(f"Пластина {i} (shelf): {', '.join(g)}")
        console.print(
            '[dim]Экспорт одного STL для shelf не реализован — используйте packing.algorithm = "rectpack".[/dim]'
        )
        return 0

    plates = pack_rectangles_on_plates(footprints, px, py, gap_mm=gap)
    if args.dry_run:
        console.print(f"Пластин (rectpack): {len(plates)}")
        for pl in plates:
            console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")
        return 0

    for pl in plates:
        out_stl = args.output_dir / f"plate_{pl.index + 1:02d}.stl"
        out_js = args.output_dir / f"plate_{pl.index + 1:02d}.json"
        export_plate_stl(oriented_meshes, pl, out_stl, out_js)
        console.print(f"Wrote {out_stl}")
    return 0
