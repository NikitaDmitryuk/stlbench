"""Sample TOML generators for `stlbench config init` and `stlbench config job`.

Orientation mode, rotation sample count, and default layout algorithm live in code
(`stlbench.config.defaults` and CLI: `--orientation`, `layout --algorithm`).
"""

from __future__ import annotations

from stlbench.config.schema import (
    AppSettings,
    PackingSection,
    PipelineSection,
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
    """Printer profile TOML for `stlbench config init` (printer + scaling + packing)."""
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
        "# Per-axis margin on bed/height: 0.02 ~ 2% inset on each side.",
        f"bed_margin = {_toml_number(sc.bed_margin)}",
        "# Multiplier applied after geometry fit (<1 leaves room for slicer brim; 1.0 = none).",
        f"post_fit_scale = {_toml_number(sc.post_fit_scale)}",
        "",
        "[packing]",
        "# Surface-to-surface gap between parts on the bed (mm).",
        f"gap_mm = {_toml_number(pk.gap_mm)}",
        "",
    ]
    return "\n".join(lines)


def render_sample_job_toml() -> str:
    """Job-file TOML for `stlbench config job` (printer + pipeline + example parts)."""
    s = sample_app_settings()
    p = s.printer
    sc = s.scaling
    pk = s.packing
    pl = PipelineSection()
    steps_str = "[" + ", ".join(f'"{step.value}"' for step in pl.default_steps) + "]"

    lines = [
        "# stlbench job file — run with: stlbench job job.toml -o ./plates",
        "# Each [[parts]] entry can override 'steps'; omit to use default_steps.",
        "",
        "[printer]",
        f"name = {_toml_str(p.name)}",
        f"width_mm = {_toml_number(p.width_mm)}",
        f"depth_mm = {_toml_number(p.depth_mm)}",
        f"height_mm = {_toml_number(p.height_mm)}",
        "",
        "[scaling]",
        "# Per-axis margin on bed/height: 0.02 ~ 2% inset on each side.",
        f"bed_margin = {_toml_number(sc.bed_margin)}",
        "# Multiplier applied after geometry fit (<1 leaves room for slicer brim; 1.0 = none).",
        f"post_fit_scale = {_toml_number(sc.post_fit_scale)}",
        "",
        "[packing]",
        "# Surface-to-surface gap between parts on the bed (mm).",
        f"gap_mm = {_toml_number(pk.gap_mm)}",
        "",
        "[pipeline]",
        "# Default step sequence for parts that do not specify their own 'steps'.",
        "# Valid sequences (layout must always be last):",
        '#   ["scale", "orient", "layout"]  — SO(3) search → global scale → orient → pack  (default)',
        '#   ["orient", "scale", "layout"]  — orient first, then scale from oriented AABB',
        '#   ["scale", "layout"]            — scale but skip overhang orientation',
        '#   ["orient", "layout"]           — orient only, no scaling',
        '#   ["layout"]                     — pack as-is (model already prepared)',
        f"default_steps = {steps_str}",
        "",
        "# ---------------------------------------------------------------------------",
        "# Parts list — add one [[parts]] block per STL file.",
        "# ---------------------------------------------------------------------------",
        "",
        "[[parts]]",
        '# path = "models/part.stl"         # relative to this file',
        "# steps is omitted → inherits default_steps",
        "",
        "[[parts]]",
        '# path = "pre_oriented/sword.stl"  # already oriented + supported',
        '# steps = ["layout"]               # skip scale + orient, just pack',
        "",
    ]
    return "\n".join(lines)
