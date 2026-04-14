"""Tests for the job-file pipeline (run_job.py + config/schema.py extensions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from stlbench.config.schema import AppSettings, PartSpec, PipelineSection, StepName
from stlbench.pipeline.run_job import JobRunArgs, run_job

# ---------------------------------------------------------------------------
# Paths to real STL assets shipped with the repo
# ---------------------------------------------------------------------------

_EXAMPLES = Path(__file__).parent.parent / "examples" / "gandalf"
_SCALED_DIR = _EXAMPLES / "scaled"
_SWORD = _SCALED_DIR / "sword.stl"
_STAFF = _SCALED_DIR / "staff.stl"
_FIGURE = _SCALED_DIR / "figure.stl"

# Printer profile that fits the already-scaled example parts without further scaling
_PRINTER_WIDE = (200.0, 200.0, 220.0)

# ---------------------------------------------------------------------------
# Schema-level unit tests (fast, no mesh loading)
# ---------------------------------------------------------------------------


def test_step_name_values():
    assert StepName.SCALE.value == "scale"
    assert StepName.ORIENT.value == "orient"
    assert StepName.LAYOUT.value == "layout"


def test_valid_step_sequences():
    valid = [
        ["scale", "orient", "layout"],
        ["orient", "scale", "layout"],
        ["scale", "layout"],
        ["orient", "layout"],
        ["layout"],
    ]
    for seq in valid:
        steps = [StepName(s) for s in seq]
        spec = PartSpec(path=Path("dummy.stl"), steps=steps)
        assert spec.steps == steps


def test_invalid_step_layout_not_last():
    with pytest.raises(ValueError, match="layout.*last"):
        PartSpec(
            path=Path("dummy.stl"),
            steps=[StepName.LAYOUT, StepName.SCALE],
        )


def test_invalid_step_missing_layout():
    with pytest.raises(ValueError, match="layout"):
        PartSpec(path=Path("dummy.stl"), steps=[StepName.SCALE, StepName.ORIENT])


def test_invalid_step_unknown_sequence():
    with pytest.raises(ValueError):
        PartSpec(
            path=Path("dummy.stl"),
            steps=[StepName.SCALE, StepName.SCALE, StepName.LAYOUT],
        )


def test_part_spec_inherits_default_steps():
    default = [StepName.SCALE, StepName.ORIENT, StepName.LAYOUT]
    spec = PartSpec(path=Path("dummy.stl"))  # steps=None
    assert spec.effective_steps(default) == default


def test_part_spec_overrides_steps():
    default = [StepName.SCALE, StepName.ORIENT, StepName.LAYOUT]
    override = [StepName.LAYOUT]
    spec = PartSpec(path=Path("dummy.stl"), steps=[StepName.LAYOUT])
    assert spec.effective_steps(default) == override


def test_pipeline_section_default():
    ps = PipelineSection()
    assert ps.default_steps == [StepName.SCALE, StepName.ORIENT, StepName.LAYOUT]


def test_app_settings_pipeline_field():
    """AppSettings must accept and preserve [pipeline] and [[parts]]."""
    s = AppSettings.model_validate(
        {
            "printer": {"width_mm": 100.0, "depth_mm": 100.0, "height_mm": 100.0},
            "pipeline": {"default_steps": ["layout"]},
            "parts": [
                {"path": "a.stl"},
                {"path": "b.stl", "steps": ["scale", "layout"]},
            ],
        }
    )
    assert s.pipeline.default_steps == [StepName.LAYOUT]
    assert len(s.parts) == 2
    assert s.parts[1].steps == [StepName.SCALE, StepName.LAYOUT]


def test_app_settings_backwards_compat():
    """Existing TOML files without [pipeline]/[[parts]] still load fine."""
    s = AppSettings.model_validate(
        {"printer": {"width_mm": 153.36, "depth_mm": 77.76, "height_mm": 165.0}}
    )
    assert s.pipeline.default_steps == [StepName.SCALE, StepName.ORIENT, StepName.LAYOUT]
    assert s.parts == []


# ---------------------------------------------------------------------------
# Job-runner integration tests (use real STL files; fast because dry_run=True)
# ---------------------------------------------------------------------------


def _make_job_toml(
    tmp_path: Path,
    parts: list[tuple[Path, list[str] | None]],
    default_steps: list[str] | None = None,
    printer: tuple[float, float, float] = _PRINTER_WIDE,
) -> Path:
    """Write a minimal job.toml and return its path."""
    lines = [
        "[printer]",
        f"width_mm  = {printer[0]}",
        f"depth_mm  = {printer[1]}",
        f"height_mm = {printer[2]}",
        "",
    ]
    if default_steps is not None:
        steps_str = "[" + ", ".join(f'"{s}"' for s in default_steps) + "]"
        lines += ["[pipeline]", f"default_steps = {steps_str}", ""]

    for path, steps in parts:
        lines.append("[[parts]]")
        lines.append(f'path = "{path}"')
        if steps is not None:
            steps_str = "[" + ", ".join(f'"{s}"' for s in steps) + "]"
            lines.append(f"steps = {steps_str}")
        lines.append("")

    job_path = tmp_path / "job.toml"
    job_path.write_text("\n".join(lines))
    return job_path


def _run(
    tmp_path: Path,
    parts: list[tuple[Path, list[str] | None]],
    default_steps: list[str] | None = None,
    dry_run: bool = True,
) -> int:
    job_path = _make_job_toml(tmp_path, parts, default_steps=default_steps)
    args = JobRunArgs(
        job_path=job_path,
        output_dir=tmp_path / "out",
        n_orient_candidates=10,  # fast for tests
        rotation_samples=32,  # fast for tests
        dry_run=dry_run,
    )
    return run_job(args)


@pytest.mark.skipif(not _SWORD.exists(), reason="Example STL assets not found")
def test_job_layout_only(tmp_path):
    """All parts with steps=['layout'] — no scale or orient."""
    rc = _run(
        tmp_path,
        parts=[
            (_SWORD, ["layout"]),
            (_STAFF, ["layout"]),
        ],
    )
    assert rc == 0


@pytest.mark.skipif(not _SWORD.exists(), reason="Example STL assets not found")
def test_job_full_pipeline(tmp_path):
    """Default full pipeline (scale → orient → layout)."""
    rc = _run(
        tmp_path,
        parts=[
            (_SWORD, None),
            (_STAFF, None),
        ],
        default_steps=["scale", "orient", "layout"],
    )
    assert rc == 0


@pytest.mark.skipif(not _SWORD.exists(), reason="Example STL assets not found")
def test_job_mixed(tmp_path):
    """Mix: one part needs full pipeline, another just layout."""
    rc = _run(
        tmp_path,
        parts=[
            (_FIGURE, None),  # default → scale + orient + layout
            (_SWORD, ["layout"]),  # already prepared, just pack
        ],
        default_steps=["scale", "orient", "layout"],
    )
    assert rc == 0


@pytest.mark.skipif(not _SWORD.exists(), reason="Example STL assets not found")
def test_job_scale_only_no_orient(tmp_path):
    """steps=['scale', 'layout'] — scale but skip overhang orientation."""
    rc = _run(
        tmp_path,
        parts=[(_SWORD, ["scale", "layout"])],
    )
    assert rc == 0


@pytest.mark.skipif(not _SWORD.exists(), reason="Example STL assets not found")
def test_job_orient_before_scale(tmp_path):
    """steps=['orient', 'scale', 'layout'] — orient first, then scale."""
    rc = _run(
        tmp_path,
        parts=[(_SWORD, ["orient", "scale", "layout"])],
    )
    assert rc == 0


@pytest.mark.skipif(not _SWORD.exists(), reason="Example STL assets not found")
def test_job_orient_only(tmp_path):
    """steps=['orient', 'layout'] — orient for overhangs, no scaling."""
    rc = _run(
        tmp_path,
        parts=[(_SWORD, ["orient", "layout"])],
    )
    assert rc == 0


@pytest.mark.skipif(not _SWORD.exists(), reason="Example STL assets not found")
def test_job_writes_files(tmp_path):
    """With dry_run=False the output 3MF and JSON files must be created."""
    rc = _run(
        tmp_path,
        parts=[
            (_SWORD, ["layout"]),
            (_STAFF, ["layout"]),
        ],
        dry_run=False,
    )
    assert rc == 0
    out_dir = tmp_path / "out"
    assert any(out_dir.glob("plate_*.3mf")), "No 3MF files written"
    assert any(out_dir.glob("plate_*.json")), "No JSON manifests written"


def test_job_missing_part_file(tmp_path):
    """Non-existent part path should return error code 2."""
    job_path = _make_job_toml(
        tmp_path,
        parts=[(Path("/nonexistent/missing.stl"), ["layout"])],
    )
    args = JobRunArgs(job_path=job_path, output_dir=tmp_path / "out", dry_run=True)
    assert run_job(args) == 2


def test_job_no_parts(tmp_path):
    """Job file with no [[parts]] should return error code 2."""
    job_path = tmp_path / "job.toml"
    job_path.write_text("[printer]\nwidth_mm = 100.0\ndepth_mm = 100.0\nheight_mm = 100.0\n")
    args = JobRunArgs(job_path=job_path, output_dir=tmp_path / "out", dry_run=True)
    assert run_job(args) == 2
