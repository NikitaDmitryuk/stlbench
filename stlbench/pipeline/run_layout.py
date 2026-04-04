from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.export.plate import export_plate_3mf, mesh_footprint_xy
from stlbench.packing.layout_orientation import select_layout_transform
from stlbench.packing.rectpack_plate import int_bin_dims_mm, pack_rectangles_on_plates
from stlbench.packing.shelf import build_packable_parts, greedy_shelf_plates
from stlbench.pipeline.common import (
    load_named_meshes,
    resolve_algorithm,
    resolve_gap,
    resolve_printer,
    resolve_settings,
)


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
    st = resolve_settings(args.config_path)

    try:
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    gap = resolve_gap(args.gap_mm, st)
    algo = resolve_algorithm(args.algorithm, st)

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    _paths, names, meshes = loaded

    dims_list: list[tuple[float, float, float]] = []
    for m in meshes:
        dx, dy, dz = mesh_footprint_xy(m)
        dims_list.append((dx, dy, dz))

    rot_samples = ORIENTATION_SAMPLES_DEFAULT
    rot_seed = ORIENTATION_SEED_DEFAULT

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
            "[red]No bed-fitting orientation for these parts: 90° permutations + "
            f"{rot_samples} random rotations (same idea as scale; seed={rot_seed}). "
            f"Bed {bw}x{bh} mm XY, Pz={pz:.2f} mm, gap={gap:.2f} mm.[/red]"
        )
        for name, dx, dy, dz in bad_layout:
            console.print(f"  [red]{name}[/red]: file-axis AABB {dx:.2f}x{dy:.2f}x{dz:.2f} mm")
        console.print(
            "[dim]Try smaller packing.gap_mm / scaling.post_fit_scale or split the model.[/dim]"
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
            console.print("Heuristic says these do not fit:", ", ".join(bad))
        groups = greedy_shelf_plates(packable, px, py)
        for i, g in enumerate(groups, 1):
            console.print(f"Plate {i} (shelf): {', '.join(g)}")
        console.print(
            "[dim]Shelf mode does not export STL; use: stlbench layout ... --algorithm rectpack[/dim]"
        )
        return 0

    plates = pack_rectangles_on_plates(footprints, px, py, gap_mm=gap)
    if args.dry_run:
        console.print(f"Plates (rectpack): {len(plates)}")
        for pl in plates:
            console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")
        return 0

    for pl in plates:
        out_3mf = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
        out_js = args.output_dir / f"plate_{pl.index + 1:02d}.json"
        export_plate_3mf(oriented_meshes, pl, out_3mf, names=list(names), out_manifest=out_js)
        console.print(f"Wrote {out_3mf}")
    return 0
