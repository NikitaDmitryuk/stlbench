"""Sample TOML for `stlbench config init` — matches `configs/mars5_ultra.toml`.

Orientation mode, rotation sample count, and default layout algorithm live in code
(`stlbench.config.defaults` and CLI: `--orientation`, `layout --algorithm`).
"""

from __future__ import annotations

from stlbench.config.schema import (
    AppSettings,
    PackingSection,
    PrinterSection,
    ScalingSection,
)


def sample_app_settings() -> AppSettings:
    """Defaults for a new TOML profile (Mars 5 Ultra–style example)."""
    return AppSettings(
        printer=PrinterSection(
            name="Elegoo Mars 5 Ultra",
            width_mm=153.36,
            depth_mm=77.76,
            height_mm=165.0,
        ),
        scaling=ScalingSection(bed_margin=0.02, post_fit_scale=0.95),
        packing=PackingSection(gap_mm=5.0),
    )


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_number(v: float | int) -> str:
    if isinstance(v, bool):
        raise TypeError("use _toml_bool")
    if isinstance(v, int):
        return str(v)
    text = f"{v:.12f}".rstrip("0").rstrip(".")
    return text if text else "0"


def render_sample_config_toml() -> str:
    """Commented TOML text (same shape as the repo example)."""
    s = sample_app_settings()
    p = s.printer
    sc = s.scaling
    pk = s.packing

    lines = [
        "# Typical SLA build volume (example: ELEGOO Mars 5 Ultra,",
        "# ~6.04 x 3.06 x 6.49 in as mm). Replace with values from your printer manual.",
        "",
        "[printer]",
        f"name = {_toml_str(p.name)}",
        f"width_mm = {_toml_number(p.width_mm)}",
        f"depth_mm = {_toml_number(p.depth_mm)}",
        f"height_mm = {_toml_number(p.height_mm)}",
        "",
        "[scaling]",
        "# Per-axis margin on bed/height (same idea as --margin): 0.02 ~ 2% inset each side.",
        f"bed_margin = {_toml_number(sc.bed_margin)}",
        "# Multiplier after geometry fit (<1 leaves room for slicer/brim; 1.0 = none).",
        f"post_fit_scale = {_toml_number(sc.post_fit_scale)}",
        "",
        "[packing]",
        "# Gap between parts on the bed (layout, fill, autopack, info).",
        f"gap_mm = {_toml_number(pk.gap_mm)}",
        "",
    ]
    return "\n".join(lines)
