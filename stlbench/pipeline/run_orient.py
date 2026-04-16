"""Orient parts to minimize overhang area (Tweaker-3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from stlbench.domain.part import Part
from stlbench.domain.printer import Printer
from stlbench.pipeline.common import (
    load_named_meshes,
    resolve_printer,
    resolve_settings,
)
from stlbench.reporting.tables import orient_table
from stlbench.steps.orient import OrientStep

if TYPE_CHECKING:
    from stlbench.config.schema import AppSettings


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


def run_orient(args: OrientRunArgs) -> int:
    console = Console(stderr=True)
    st = args.settings or resolve_settings(args.config_path)

    # Printer dims are optional for orient: when provided, orientations that
    # don't fit the build volume receive a heavy penalty.
    printer: Printer | None = None
    if args.printer_xyz is not None or st is not None:
        try:
            px, py, pz = resolve_printer(args.printer_xyz, st)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 2
        printer = Printer.from_tuple((px, py, pz))
        if st and st.printer.name:
            printer = Printer(
                width_mm=printer.width_mm,
                depth_mm=printer.depth_mm,
                height_mm=printer.height_mm,
                name=st.printer.name,
            )
        console.print(f"Printer: {printer}")

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    paths, part_names, meshes = loaded

    # Create parts using the new domain object
    parts = [
        Part(name=name, mesh=mesh, source_path=path)
        for name, mesh, path in zip(part_names, meshes, paths, strict=True)
    ]

    # Create and run the orient step
    orient_step = OrientStep(
        printer=printer or Printer(200.0, 150.0, 300.0),  # Default printer if none provided
        overhang_threshold_deg=args.overhang_threshold_deg,
        n_candidates=args.n_candidates,
    )

    result = orient_step.process(parts)
    oriented_parts = result.parts
    orient_stats = result.metadata.get("orient_stats", [])

    # Print results using the new reporting module
    if orient_stats:
        table = orient_table(orient_stats)
        console.print(table)
        console.print(f"Overhang threshold: {args.overhang_threshold_deg}°")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Export oriented parts
    for idx, (path, part) in enumerate(zip(paths, oriented_parts, strict=True)):
        if args.recursive:
            rel_parent = path.parent.relative_to(args.input_dir)
            out_sub = args.output_dir / rel_parent
            out_sub.mkdir(parents=True, exist_ok=True)
            out_dir = out_sub
        else:
            out_dir = args.output_dir
        out_path = out_dir / f"{path.stem + args.suffix}.stl"

        if args.verbose:
            console.print(f"[dim]  [{idx + 1}/{len(paths)}] exporting {path.name}[/dim]")

        try:
            part.mesh.export(out_path)
        except (OSError, ValueError) as e:
            console.print(f"[red]Failed to export {out_path}: {e}[/red]")
            return 1
        console.print(f"Wrote {out_path}")

    return 0
