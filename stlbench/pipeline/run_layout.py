"""Layout parts on build plates using various packing algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.domain.part import Part
from stlbench.domain.plate import Plate
from stlbench.domain.printer import Printer
from stlbench.packing import make_packer
from stlbench.packing.layout_orientation import select_layout_transform
from stlbench.packing.rectpack_plate import int_bin_dims_mm
from stlbench.packing.shelf import build_packable_parts, greedy_shelf_plates
from stlbench.pipeline.common import (
    load_named_meshes,
    resolve_algorithm,
    resolve_gap,
    resolve_printer,
    resolve_settings,
)
from stlbench.steps.layout import LayoutStep

if TYPE_CHECKING:
    import numpy as np


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
    cleanup: bool = False


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

    printer = Printer.from_tuple((px, py, pz))
    if st and st.printer.name:
        printer = Printer(
            width_mm=printer.width_mm,
            depth_mm=printer.depth_mm,
            height_mm=printer.height_mm,
            name=st.printer.name,
        )

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    _paths, part_names, meshes = loaded

    # Create parts using the new domain object
    parts = [
        Part(name=name, mesh=mesh, source_path=path)
        for name, mesh, path in zip(part_names, meshes, _paths, strict=True)
    ]

    # Pre-check: every part must fit the bed in at least one orientation.
    dims_list: list[tuple[float, float, float]] = []
    layout_plans: list[tuple[np.ndarray, float, float] | None] = []
    bad_layout: list[tuple[str, float, float, float]] = []

    for part in parts:
        dx, dy, dz = part.extents
        dims_list.append((dx, dy, dz))
        ok, t, fw, fh = select_layout_transform(
            part.mesh,
            px,
            py,
            pz,
            gap,
            random_samples=4096,
            seed=0,
        )
        if not ok:
            bad_layout.append((part.name, dx, dy, dz))
            layout_plans.append(None)
        else:
            layout_plans.append((t, fw, fh))

    if bad_layout:
        bw, bh = int_bin_dims_mm(px, py)
        console.print(
            "[red]No bed-fitting orientation for these parts: 90° permutations + "
            f"4096 random rotations. "
            f"Bed {bw}x{bh} mm XY, Pz={pz:.2f} mm, gap={gap:.2f} mm.[/red]"
        )
        for name, dx, dy, dz in bad_layout:
            console.print(f"  [red]{name}[/red]: file-axis AABB {dx:.2f}x{dy:.2f}x{dz:.2f} mm")
        console.print(
            "[dim]Try smaller packing.gap_mm / scaling.post_fit_scale or split the model.[/dim]"
        )
        return 1

    # Apply layout transforms
    oriented_parts: list[Part] = []
    for part, plan in zip(parts, layout_plans, strict=True):
        assert plan is not None
        t, _fw, _fh = plan
        oriented_part = part.clone()
        oriented_part.apply_transform(t)
        oriented_parts.append(oriented_part)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if algo == "shelf":
        packable, bad = build_packable_parts(part_names, dims_list, px, py, pz)
        if bad:
            console.print("Heuristic says these do not fit:", ", ".join(bad))
        groups = greedy_shelf_plates(packable, px, py)
        for i, g in enumerate(groups, 1):
            console.print(f"Plate {i} (shelf): {', '.join(g)}")
        console.print(
            "[dim]Shelf mode does not export STL; use: stlbench layout ... --algorithm rectpack[/dim]"
        )
        return 0

    if args.cleanup:
        for i, part in enumerate(oriented_parts):
            cleaned_mesh, n_rem = remove_small_components(part.mesh)
            if n_rem:
                oriented_parts[i].mesh = cleaned_mesh
                console.print(
                    f"[dim]cleanup: {part_names[i]} — removed {n_rem} tiny component(s)[/dim]"
                )

    # Use the new packing strategy pattern
    try:
        packer = make_packer(algo, max_plates=64)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    # Create and run the layout step
    layout_step = LayoutStep(
        printer=printer,
        packer=packer,
        gap_mm=gap,
    )

    result = layout_step.process(oriented_parts)
    plates: list[Plate] = result.metadata.get("plates", [])

    if args.dry_run:
        console.print(f"Plates: {len(plates)}")
        for plate in plates:
            console.print(f"  Plate {plate.index + 1}: {plate.part_count} parts")
        return 0

    # Export plates using the new domain object
    for plate in plates:
        out_3mf = args.output_dir / f"plate_{plate.index + 1:02d}.3mf"
        out_js = args.output_dir / f"plate_{plate.index + 1:02d}.json"
        plate.export_3mf(out_3mf, out_js)
        console.print(f"Wrote {out_3mf}")

    return 0
