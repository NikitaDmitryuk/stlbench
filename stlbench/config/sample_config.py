"""Sample TOML generators for `stlbench config init` and `stlbench config job`."""

from __future__ import annotations

from stlbench.config.schema import (
    AppSettings,
    OrientationSection,
    PackingSection,
    PipelineSection,
    PrinterSection,
    RepairSection,
    ScalingSection,
    UISection,
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
        scaling=ScalingSection(post_fit_scale=0.95, any_rotation=True, maximize=True),
        packing=PackingSection(gap_mm=10.0),
        orientation=OrientationSection(),
        repair=RepairSection(),
        ui=UISection(),
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
    ori = s.orientation
    repair = s.repair
    autopack = s.autopack
    ui = s.ui

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
        "# Multiplier applied after geometry fit (<1 leaves room for slicer brim; 1.0 = none).",
        f"post_fit_scale = {_toml_number(sc.post_fit_scale)}",
        "# Allow any 3D orientation (all axis permutations) for scale fitting.",
        "# false = Z-axis rotation only (default).",
        f"any_rotation = {str(sc.any_rotation).lower()}",
        "# Full SO(3) random search (4096 samples) to maximise scale factor.",
        "# Requires any_rotation = true. Slow; produces arbitrary tilt angles.",
        f"maximize = {str(sc.maximize).lower()}",
        "",
        "[packing]",
        "# Surface-to-surface gap between parts on the bed (mm).",
        f"gap_mm = {_toml_number(pk.gap_mm)}",
        "# Keep parts away from the platform edge to leave room for support bases.",
        f"edge_margin_mm = {_toml_number(pk.edge_margin_mm)}",
        "# Optional: no more than this many prepare plates; prepare will reduce layout scale.",
        "# max_plates = 4",
        "",
        "[orientation]",
        '# Resin orientation balance: "balanced", "stability", or "compact".',
        f"resin_balance = {_toml_str(ori.resin_balance)}",
        '# Long-part angle policy: "thin_linear" (default), "linear", or "disabled".',
        f"long_part_angle_policy = {_toml_str(ori.long_part_angle_policy)}",
        '# Sacrificial assembly-side detection: "auto" or "disabled".',
        f"assembly_side_policy = {_toml_str(ori.assembly_side_policy)}",
        "# Preferred angle from the bed for parts matched by long_part_angle_policy.",
        f"long_part_target_angle_min_deg = {_toml_number(ori.long_part_target_angle_min_deg)}",
        f"long_part_target_angle_max_deg = {_toml_number(ori.long_part_target_angle_max_deg)}",
        "# Soft guardrails outside the preferred band.",
        f"long_part_low_angle_penalty_below_deg = {_toml_number(ori.long_part_low_angle_penalty_below_deg)}",
        f"long_part_high_angle_penalty_above_deg = {_toml_number(ori.long_part_high_angle_penalty_above_deg)}",
        "",
        "[repair]",
        "# Conservative mesh repair before scale/orient/layout/export.",
        f"enabled = {str(repair.enabled).lower()}",
        f"close_holes = {str(repair.close_holes).lower()}",
        f"max_hole_size_edges = {repair.max_hole_size_edges}",
        f"repair_non_manifold = {str(repair.repair_non_manifold).lower()}",
        f"remove_small_components = {str(repair.remove_small_components).lower()}",
        f"cache = {str(repair.cache).lower()}",
        "",
        "[autopack]",
        "# Cache exact polygon packing attempts/results and parallelise scale search.",
        f"packer = {_toml_str(autopack.packer)}",
        f"pack_workers = {_toml_str(autopack.pack_workers) if isinstance(autopack.pack_workers, str) else autopack.pack_workers}",
        f"result_cache = {str(autopack.result_cache).lower()}",
        f"attempt_cache = {str(autopack.attempt_cache).lower()}",
        f"scale_tolerance = {_toml_number(autopack.scale_tolerance)}",
        f"bitmap_grid_mm = {_toml_number(autopack.bitmap_grid_mm)}",
        f"bitmap_beam_width = {autopack.bitmap_beam_width}",
        "",
        "[ui]",
        "# Show interactive Rich progress bars when stderr is a terminal.",
        f"progress = {str(ui.progress).lower()}",
        "",
    ]
    return "\n".join(lines)


def render_sample_job_toml() -> str:
    """Job-file TOML for `stlbench config job` (printer + pipeline + example parts)."""
    s = sample_app_settings()
    p = s.printer
    sc = s.scaling
    pk = s.packing
    ori = s.orientation
    repair = s.repair
    autopack = s.autopack
    ui = s.ui
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
        "# Multiplier applied after geometry fit (<1 leaves room for slicer brim; 1.0 = none).",
        f"post_fit_scale = {_toml_number(sc.post_fit_scale)}",
        "# Allow any 3D orientation (all axis permutations) for scale fitting.",
        "# false = Z-axis rotation only (default).",
        f"any_rotation = {str(sc.any_rotation).lower()}",
        "# Full SO(3) random search (4096 samples) to maximise scale factor.",
        "# Requires any_rotation = true. Slow; produces arbitrary tilt angles.",
        f"maximize = {str(sc.maximize).lower()}",
        "",
        "[packing]",
        "# Surface-to-surface gap between parts on the bed (mm).",
        f"gap_mm = {_toml_number(pk.gap_mm)}",
        "# Keep parts away from the platform edge to leave room for support bases.",
        f"edge_margin_mm = {_toml_number(pk.edge_margin_mm)}",
        "# Optional: no more than this many prepare plates; prepare will reduce layout scale.",
        "# max_plates = 4",
        "",
        "[orientation]",
        '# Resin orientation balance: "balanced", "stability", or "compact".',
        f"resin_balance = {_toml_str(ori.resin_balance)}",
        '# Long-part angle policy: "thin_linear" (default), "linear", or "disabled".',
        f"long_part_angle_policy = {_toml_str(ori.long_part_angle_policy)}",
        '# Sacrificial assembly-side detection: "auto" or "disabled".',
        f"assembly_side_policy = {_toml_str(ori.assembly_side_policy)}",
        "# Preferred angle from the bed for parts matched by long_part_angle_policy.",
        f"long_part_target_angle_min_deg = {_toml_number(ori.long_part_target_angle_min_deg)}",
        f"long_part_target_angle_max_deg = {_toml_number(ori.long_part_target_angle_max_deg)}",
        f"long_part_low_angle_penalty_below_deg = {_toml_number(ori.long_part_low_angle_penalty_below_deg)}",
        f"long_part_high_angle_penalty_above_deg = {_toml_number(ori.long_part_high_angle_penalty_above_deg)}",
        "",
        "[repair]",
        "# Conservative mesh repair before scale/orient/layout/export.",
        f"enabled = {str(repair.enabled).lower()}",
        f"close_holes = {str(repair.close_holes).lower()}",
        f"max_hole_size_edges = {repair.max_hole_size_edges}",
        f"repair_non_manifold = {str(repair.repair_non_manifold).lower()}",
        f"remove_small_components = {str(repair.remove_small_components).lower()}",
        f"cache = {str(repair.cache).lower()}",
        "",
        "[autopack]",
        "# Cache exact polygon packing attempts/results and parallelise scale search.",
        f"packer = {_toml_str(autopack.packer)}",
        f"pack_workers = {_toml_str(autopack.pack_workers) if isinstance(autopack.pack_workers, str) else autopack.pack_workers}",
        f"result_cache = {str(autopack.result_cache).lower()}",
        f"attempt_cache = {str(autopack.attempt_cache).lower()}",
        f"scale_tolerance = {_toml_number(autopack.scale_tolerance)}",
        f"bitmap_grid_mm = {_toml_number(autopack.bitmap_grid_mm)}",
        f"bitmap_beam_width = {autopack.bitmap_beam_width}",
        "",
        "[ui]",
        "# Show interactive Rich progress bars when stderr is a terminal.",
        f"progress = {str(ui.progress).lower()}",
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
