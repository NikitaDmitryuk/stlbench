"""Typer CLI: ``stlbench`` / ``python -m stlbench``."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated

import typer

from stlbench.config.loader import load_app_settings
from stlbench.pipeline.run_autopack import AutopackRunArgs, run_autopack
from stlbench.pipeline.run_fill import FillRunArgs, run_fill
from stlbench.pipeline.run_info import InfoRunArgs, run_info
from stlbench.pipeline.run_layout import LayoutRunArgs, run_layout
from stlbench.pipeline.run_scale import ScaleRunArgs, run_scale

app = typer.Typer(
    no_args_is_help=True,
    help="STL preparation for resin 3D printing: scale, layout, fill, autopack, info.",
)


def _parse_printer_opt(value: str | None) -> tuple[float, float, float] | None:
    if value is None or not str(value).strip():
        return None
    parts = re.split(r"[\s,xX]+", str(value).strip())
    nums = [float(p) for p in parts if p]
    if len(nums) != 3:
        raise typer.BadParameter("Нужно ровно три числа: Px Py Pz или Px,Py,Pz.")
    return nums[0], nums[1], nums[2]


@app.command("scale")
def cmd_scale(
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", exists=True, file_okay=False, dir_okay=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, file_okay=True),
    ] = None,
    printer: Annotated[
        str | None,
        typer.Option(
            "--printer",
            "-p",
            help="Три числа: например 153.36,77.76,165",
        ),
    ] = None,
    margin: Annotated[float | None, typer.Option("--margin")] = None,
    supports_scale: Annotated[float | None, typer.Option("--supports-scale")] = None,
    method: Annotated[str | None, typer.Option("--method")] = None,
    orientation: Annotated[str | None, typer.Option("--orientation")] = None,
    rotation_samples: Annotated[int | None, typer.Option("--rotation-samples")] = None,
    no_upscale: Annotated[bool, typer.Option("--no-upscale")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    recursive: Annotated[bool, typer.Option("--recursive")] = False,
    suffix: Annotated[str, typer.Option("--suffix", show_default=False)] = "",
    no_packing_report: Annotated[bool, typer.Option("--no-packing-report")] = False,
    hollow: Annotated[bool | None, typer.Option("--hollow/--no-hollow")] = None,
) -> None:
    st = load_app_settings(config) if config else None
    pr = _parse_printer_opt(printer)
    raise typer.Exit(
        run_scale(
            ScaleRunArgs(
                input_dir=input_dir,
                output_dir=output_dir,
                config_path=config,
                settings=st,
                printer_xyz=pr,
                margin=margin,
                supports_scale=supports_scale,
                method=method,
                orientation=orientation,
                rotation_samples=rotation_samples,
                no_upscale=no_upscale,
                dry_run=dry_run,
                recursive=recursive,
                suffix=suffix,
                no_packing_report=no_packing_report,
                hollow_override=hollow,
            )
        )
    )


@app.command("layout")
def cmd_layout(
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", exists=True, file_okay=False, dir_okay=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, file_okay=True),
    ] = None,
    printer: Annotated[
        str | None,
        typer.Option("-p", "--printer", help="Px,Py,Pz"),
    ] = None,
    gap_mm: Annotated[float | None, typer.Option("--gap-mm")] = None,
    algorithm: Annotated[str | None, typer.Option("--algorithm")] = None,
    recursive: Annotated[bool, typer.Option("--recursive")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    pr = _parse_printer_opt(printer)
    raise typer.Exit(
        run_layout(
            LayoutRunArgs(
                input_dir=input_dir,
                output_dir=output_dir,
                config_path=config,
                printer_xyz=pr,
                gap_mm=gap_mm,
                algorithm=algorithm,
                recursive=recursive,
                dry_run=dry_run,
            )
        )
    )


@app.command("fill")
def cmd_fill(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input", "-i", exists=True, help="Single STL file or directory with one STL"
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, file_okay=True),
    ] = None,
    printer: Annotated[
        str | None,
        typer.Option("-p", "--printer", help="Px,Py,Pz"),
    ] = None,
    gap_mm: Annotated[float | None, typer.Option("--gap-mm")] = None,
    scale: Annotated[bool, typer.Option("--scale/--no-scale")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    pr = _parse_printer_opt(printer)
    raise typer.Exit(
        run_fill(
            FillRunArgs(
                input_file=input_file,
                output_dir=output_dir,
                config_path=config,
                printer_xyz=pr,
                gap_mm=gap_mm,
                scale=scale,
                dry_run=dry_run,
            )
        )
    )


@app.command("autopack")
def cmd_autopack(
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", exists=True, file_okay=False, dir_okay=True),
    ],
    output_dir: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, file_okay=True),
    ] = None,
    printer: Annotated[
        str | None,
        typer.Option("-p", "--printer", help="Px,Py,Pz"),
    ] = None,
    gap_mm: Annotated[float | None, typer.Option("--gap-mm")] = None,
    margin: Annotated[float | None, typer.Option("--margin")] = None,
    supports_scale: Annotated[float | None, typer.Option("--supports-scale")] = None,
    recursive: Annotated[bool, typer.Option("--recursive")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    pr = _parse_printer_opt(printer)
    raise typer.Exit(
        run_autopack(
            AutopackRunArgs(
                input_dir=input_dir,
                output_dir=output_dir,
                config_path=config,
                printer_xyz=pr,
                gap_mm=gap_mm,
                margin=margin,
                supports_scale=supports_scale,
                dry_run=dry_run,
                recursive=recursive,
            )
        )
    )


@app.command("info")
def cmd_info(
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", exists=True, file_okay=False, dir_okay=True),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, file_okay=True),
    ] = None,
    printer: Annotated[
        str | None,
        typer.Option("-p", "--printer", help="Px,Py,Pz"),
    ] = None,
    recursive: Annotated[bool, typer.Option("--recursive")] = False,
) -> None:
    pr = _parse_printer_opt(printer)
    raise typer.Exit(
        run_info(
            InfoRunArgs(
                input_dir=input_dir,
                config_path=config,
                printer_xyz=pr,
                recursive=recursive,
            )
        )
    )


@app.command("hollow")
def cmd_hollow_info() -> None:
    typer.echo("Configure [hollow] section in TOML and run: stlbench scale ... --hollow")


@app.command("supports")
def cmd_supports_info() -> None:
    typer.echo(
        "Supports are not generated by stlbench. Open exported STL in Lychee, Chitubox or another slicer."
    )


_KNOWN_COMMANDS = frozenset(
    {"scale", "layout", "fill", "autopack", "info", "hollow", "supports", "-h", "--help"}
)


def launch() -> None:
    if len(sys.argv) > 1 and sys.argv[1] not in _KNOWN_COMMANDS:
        sys.argv.insert(1, "scale")
    app()


def main(argv: list[str] | None = None) -> int:
    old = sys.argv
    try:
        if argv is not None:
            sys.argv = [old[0]] + list(argv)
        try:
            launch()
            return 0
        except SystemExit as e:
            code = e.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            return 1
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
