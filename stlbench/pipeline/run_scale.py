"""Scale parts to fit printer volume."""

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
)
from stlbench.reporting.tables import scale_table
from stlbench.steps.scale import ScaleStep

if TYPE_CHECKING:
    from stlbench.config.schema import AppSettings


@dataclass
class ScaleRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    settings: AppSettings | None
    printer_xyz: tuple[float, float, float] | None
    post_fit_scale: float | None
    method: str | None
    allow_rotation: bool = False
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
    if args.maximize and not args.allow_rotation:
        console.print(
            "[red]--maximize requires --allow-rotation "
            "(cannot search for a better orientation without enabling rotation)[/red]"
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

    printer = Printer.from_tuple((prx, pry, prz))
    if st and st.printer.name:
        printer = Printer(
            width_mm=printer.width_mm,
            depth_mm=printer.depth_mm,
            height_mm=printer.height_mm,
            name=st.printer.name,
        )

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

    # Create parts using the new domain object
    parts = [
        Part(name=name, mesh=mesh, source_path=path)
        for name, mesh, path in zip(part_names, meshes, paths, strict=True)
    ]

    # Create and run the scale step
    rotation_samples = args.rotation_samples if args.rotation_samples is not None else 4096
    scale_step = ScaleStep(
        printer=printer,
        method=args.method or "sorted",  # type: ignore[arg-type]
        post_fit_scale=post_fit_scale,
        allow_rotation=args.allow_rotation,
        maximize=args.maximize,
        scale_factor=args.scale_factor,
        no_upscale=args.no_upscale,
        rotation_samples=rotation_samples,
    )

    result = scale_step.process(parts)
    scaled_parts = result.parts
    s_max = result.metadata.get("s_max", 1.0)
    s_final = result.metadata.get("s_final", 1.0)
    reports = result.metadata.get("reports", [])

    # Print results using the new reporting module
    if reports:
        if st and st.printer.name:
            console.print(f"Printer profile: {st.printer.name}")
        console.print(
            f"Printer: {printer.width_mm:.4f} x {printer.depth_mm:.4f} x {printer.height_mm:.4f}"
        )
        console.print(f"Method: {args.method or 'sorted'}")
        if args.allow_rotation:
            mode = "maximize" if args.maximize else "axis-permutations"
            console.print(f"Rotation: {mode}")
            if args.maximize and args.rotation_samples:
                console.print(f"Rotation samples: {args.rotation_samples}")
        else:
            console.print("Rotation: none")
        console.print(f"s_max (geometry fit): {s_max:.6f}")
        console.print(f"post_fit_scale: {post_fit_scale:.6f}")
        console.print(f"s_final (applied): {s_final:.6f}")
        if args.no_upscale:
            console.print("(capped by --no-upscale before post_fit_scale)")

        if reports:
            limiting_part = reports[0].name if reports else "unknown"
            console.print(f"Limiting part: {limiting_part}")

        table = scale_table(reports, s_final)
        console.print(table)

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export scaled parts
    for idx, (path, part) in enumerate(zip(paths, scaled_parts, strict=True)):
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

        try:
            part.mesh.export(out_path)
        except (OSError, ValueError) as e:
            console.print(f"[red]Failed to export {out_path}: {e}[/red]")
            return 1
        console.print(f"Wrote {out_path}")

    return 0
