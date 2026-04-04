"""Typer CLI: ``stlbench`` / ``python -m stlbench``."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated

import typer

from stlbench.config.loader import load_app_settings
from stlbench.config.sample_config import render_sample_config_toml
from stlbench.pipeline.run_autopack import AutopackRunArgs, run_autopack
from stlbench.pipeline.run_fill import FillRunArgs, run_fill
from stlbench.pipeline.run_info import InfoRunArgs, run_info
from stlbench.pipeline.run_layout import LayoutRunArgs, run_layout
from stlbench.pipeline.run_scale import ScaleRunArgs, run_scale

_ROOT_HELP = """\
STL preparation for resin 3D printing: scale, layout, fill, autopack, info.

Typical commands (adjust paths and my_printer.toml):

\b
  # Printer profile — edit width_mm / depth_mm / height_mm for your machine
  stlbench config init -o my_printer.toml

  # Inspect parts: dimensions, fit, suggested scale, fill estimate
  stlbench info -i ./parts -c my_printer.toml

  # Scale all STLs to fit the build volume
  stlbench scale -i ./parts -o ./scaled -c my_printer.toml

  # Pack scaled parts onto build plates
  stlbench layout -i ./scaled -o ./plates -c my_printer.toml

  # Scale + pack so everything fits on one plate (if possible)
  stlbench autopack -i ./parts -o ./packed -c my_printer.toml

  # Fill the bed with copies of one STL (--scale fits part first, then packs)
  stlbench fill -i ./one_part.stl -o ./filled -c my_printer.toml --scale

  # Voxel hollow shell (optional) — needs [hollow] in TOML and --hollow
  stlbench scale -i ./parts -o ./scaled -c my_printer.toml --hollow

  # No config file: build volume as three numbers in mm (Px, Py, Pz)
  stlbench scale -i ./parts -o ./scaled -p "153.36,77.76,165"
"""

app = typer.Typer(no_args_is_help=True, help=_ROOT_HELP)

config_app = typer.Typer(help="Generate TOML configuration files.")
app.add_typer(config_app, name="config")


@config_app.command("init")
def cmd_config_init(
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Path to write (default: stlbench.toml in the current directory).",
        ),
    ] = Path("stlbench.toml"),
    stdout: Annotated[
        bool, typer.Option("--stdout", help="Print TOML to stdout; do not write a file.")
    ] = False,
    force: Annotated[
        bool, typer.Option("-f", "--force", help="Overwrite an existing file.")
    ] = False,
) -> None:
    text = render_sample_config_toml()
    if stdout:
        typer.echo(text, nl=False)
        raise typer.Exit(0)
    if output.exists() and not force:
        typer.secho(
            f"File already exists: {output}  (use --force to overwrite)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {output.resolve()}")
    raise typer.Exit(0)


def _parse_printer_opt(value: str | None) -> tuple[float, float, float] | None:
    if value is None or not str(value).strip():
        return None
    parts = re.split(r"[\s,xX]+", str(value).strip())
    nums = [float(p) for p in parts if p]
    if len(nums) != 3:
        raise typer.BadParameter("Need exactly three numbers: Px Py Pz or Px,Py,Pz.")
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
            help="Three numbers, e.g. 153.36,77.76,165",
        ),
    ] = None,
    margin: Annotated[float | None, typer.Option("--margin")] = None,
    post_fit_scale: Annotated[
        float | None,
        typer.Option(
            "--post-fit-scale",
            help="Multiplier after geometry fit (TOML: scaling.post_fit_scale).",
        ),
    ] = None,
    method: Annotated[str | None, typer.Option("--method")] = None,
    orientation: Annotated[str | None, typer.Option("--orientation")] = None,
    rotation_samples: Annotated[int | None, typer.Option("--rotation-samples")] = None,
    no_upscale: Annotated[bool, typer.Option("--no-upscale")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    recursive: Annotated[bool, typer.Option("--recursive")] = False,
    suffix: Annotated[str, typer.Option("--suffix", show_default=False)] = "",
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
                post_fit_scale=post_fit_scale,
                method=method,
                orientation=orientation,
                rotation_samples=rotation_samples,
                no_upscale=no_upscale,
                dry_run=dry_run,
                recursive=recursive,
                suffix=suffix,
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
    post_fit_scale: Annotated[
        float | None,
        typer.Option(
            "--post-fit-scale",
            help="Multiplier after geometry fit (TOML: scaling.post_fit_scale).",
        ),
    ] = None,
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
                post_fit_scale=post_fit_scale,
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
    typer.echo(
        "Hollowing runs only with: stlbench scale ... -c your.toml --hollow "
        "(set [hollow] wall_thickness_mm and voxel_mm in the TOML)."
    )


_KNOWN_COMMANDS = frozenset(
    {
        "scale",
        "layout",
        "fill",
        "autopack",
        "info",
        "hollow",
        "config",
        "-h",
        "--help",
    }
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
