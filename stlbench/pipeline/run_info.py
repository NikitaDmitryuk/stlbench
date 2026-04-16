from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from stlbench.config.defaults import DEFAULT_PACKING_GAP_MM
from stlbench.core.fit import aabb_edge_lengths, compute_global_scale
from stlbench.pipeline.common import load_named_meshes, resolve_printer, resolve_settings
from stlbench.pipeline.run_fill import _max_copies_on_plate


@dataclass
class InfoRunArgs:
    input_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    recursive: bool


def run_info(args: InfoRunArgs) -> int:
    console = Console(stderr=True)
    st = resolve_settings(args.config_path)

    try:
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    _paths, names, meshes = loaded

    gap = st.packing.gap_mm if st else DEFAULT_PACKING_GAP_MM

    if st and st.printer.name:
        console.print(f"Printer: {st.printer.name}")
    console.print(f"Build volume: {px:.2f} x {py:.2f} x {pz:.2f} mm")
    console.print()

    dims_list: list[tuple[float, float, float]] = []
    for m in meshes:
        dims_list.append(aabb_edge_lengths(np.asarray(m.bounds)))

    table = Table(show_header=True, header_style="bold", title="Model parts")
    table.add_column("part", max_width=42)
    table.add_column("X (mm)", justify="right")
    table.add_column("Y (mm)", justify="right")
    table.add_column("Z (mm)", justify="right")
    table.add_column("volume (mm\u00b3)", justify="right")
    table.add_column("verts", justify="right")
    table.add_column("faces", justify="right")
    table.add_column("fits?", justify="center")
    table.add_column("s_max", justify="right")
    table.add_column("fill copies", justify="right")

    for name, m, d in zip(names, meshes, dims_list, strict=True):
        vol = float(abs(m.volume)) if hasattr(m, "volume") else 0.0
        n_verts = len(m.vertices)
        n_faces = len(m.faces)

        s, _ = compute_global_scale((px, py, pz), [d], [name], "sorted")
        fits = s >= 1.0 - 1e-9

        plate = _max_copies_on_plate(d[0], d[1], px, py, gap)
        copies = len(plate.rects) if plate else 0

        table.add_row(
            name,
            f"{d[0]:.2f}",
            f"{d[1]:.2f}",
            f"{d[2]:.2f}",
            f"{vol:.1f}",
            str(n_verts),
            str(n_faces),
            "[green]yes[/green]" if fits else "[red]no[/red]",
            f"{s:.4f}",
            str(copies),
        )

    console.print(table)

    if len(dims_list) > 1:
        console.print()
        s_all, _ = compute_global_scale((px, py, pz), dims_list, names, "sorted")
        pf = st.scaling.post_fit_scale if st else 1.0
        console.print(f"Global scale (all parts fit individually): {s_all:.6f}")
        console.print(f"With post_fit_scale ({pf}): {s_all * pf:.6f}")

    return 0
