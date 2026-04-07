from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from stlbench.config.schema import AppSettings
from stlbench.core.fit import printer_dims_with_margin
from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    overhang_score,
)
from stlbench.pipeline.common import load_named_meshes, resolve_printer, resolve_settings

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


def run_orient(args: OrientRunArgs) -> int:
    console = Console(stderr=True)
    st = args.settings or resolve_settings(args.config_path)

    # Printer dims are optional for orient: when provided, orientations that
    # don't fit the build volume receive a heavy penalty.
    printer_dims: tuple[float, float, float] | None = None
    if args.printer_xyz is not None or st is not None:
        try:
            px, py, pz = resolve_printer(args.printer_xyz, st)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 2
        margin = st.scaling.bed_margin if st is not None else 0.0
        px, py, pz = printer_dims_with_margin(px, py, pz, margin)
        printer_dims = (px, py, pz)
        if st and st.printer.name:
            console.print(f"Printer: {st.printer.name}")
        console.print(f"Build volume (after margin): {px:.1f} × {py:.1f} × {pz:.1f} mm")

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    paths, names, meshes = loaded

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("score before", justify="right")
    table.add_column("score after", justify="right")
    table.add_column("improvement", justify="right")

    results = []
    for path, name, mesh in zip(paths, names, meshes, strict=True):
        console.print(f"Analysing {name} …", highlight=False)

        score_before = overhang_score(mesh, _IDENTITY, args.overhang_threshold_deg)

        rotation, score_after = find_min_overhang_rotation(
            mesh,
            overhang_threshold_deg=args.overhang_threshold_deg,
            n_candidates=args.n_candidates,
            printer_dims=printer_dims,
        )

        improvement = score_before - score_after
        pct = (improvement / max(abs(score_before), 1.0)) * 100.0

        table.add_row(
            name,
            f"{score_before:.1f}",
            f"{score_after:.1f}",
            f"{pct:+.1f}%",
        )
        results.append((path, name, mesh, rotation))

    console.print(table)
    console.print(f"Overhang threshold: {args.overhang_threshold_deg}°")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path, _name, mesh, rotation in results:
        if args.recursive:
            rel_parent = path.parent.relative_to(args.input_dir)
            out_sub = args.output_dir / rel_parent
            out_sub.mkdir(parents=True, exist_ok=True)
            out_dir = out_sub
        else:
            out_dir = args.output_dir

        stem = path.stem + args.suffix
        out_path = out_dir / f"{stem}.stl"

        oriented = apply_min_overhang_orientation(mesh, rotation)
        try:
            oriented.export(out_path)
        except (OSError, ValueError) as e:
            console.print(f"[red]Failed to export {out_path}: {e}[/red]")
            return 1
        console.print(f"Wrote {out_path}")

    return 0
