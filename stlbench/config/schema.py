from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class PrinterSection(BaseModel):
    name: str = ""
    width_mm: float = Field(gt=0)
    depth_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)


class ScalingSection(BaseModel):
    post_fit_scale: float = Field(default=1.0, gt=0.0)
    any_rotation: bool = False
    maximize: bool = False


class PackingSection(BaseModel):
    gap_mm: float = Field(default=2.0, ge=0.0)
    edge_margin_mm: float = Field(default=2.0, ge=0.0)


class OrientationSection(BaseModel):
    resin_balance: str = Field(default="balanced", pattern="^(balanced|stability|compact)$")
    long_part_target_angle_min_deg: float = Field(default=30.0, ge=0.0, le=90.0)
    long_part_target_angle_max_deg: float = Field(default=50.0, ge=0.0, le=90.0)
    long_part_low_angle_penalty_below_deg: float = Field(default=20.0, ge=0.0, le=90.0)
    long_part_high_angle_penalty_above_deg: float = Field(default=60.0, ge=0.0, le=90.0)

    @model_validator(mode="after")
    def _validate_angles(self) -> OrientationSection:
        if self.long_part_target_angle_min_deg > self.long_part_target_angle_max_deg:
            raise ValueError("orientation long part target min angle must be <= max angle")
        if self.long_part_low_angle_penalty_below_deg > self.long_part_target_angle_min_deg:
            raise ValueError("orientation low angle penalty threshold must be <= target min angle")
        if self.long_part_high_angle_penalty_above_deg < self.long_part_target_angle_max_deg:
            raise ValueError("orientation high angle penalty threshold must be >= target max angle")
        return self


# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------


class StepName(StrEnum):
    """Named pipeline step.

    Valid sequences (``layout`` must always be last):

    * ``["scale", "orient", "layout"]`` — global scale → Tweaker-3 orient → pack
    * ``["orient", "scale", "layout"]`` — Tweaker-3 orient → scale from oriented AABB → pack
    * ``["scale", "layout"]`` — global scale → pack
    * ``["orient", "layout"]`` — Tweaker-3 orient → pack
    * ``["layout"]`` — pack only (model already prepared)

    Rotation during the ``scale`` step is controlled by ``[scaling] any_rotation``
    and ``maximize`` in the job TOML (both default to ``false``).
    """

    SCALE = "scale"
    ORIENT = "orient"
    LAYOUT = "layout"


_VALID_STEP_SEQUENCES: frozenset[tuple[StepName, ...]] = frozenset(
    {
        (StepName.SCALE, StepName.ORIENT, StepName.LAYOUT),
        (StepName.ORIENT, StepName.SCALE, StepName.LAYOUT),
        (StepName.SCALE, StepName.LAYOUT),
        (StepName.ORIENT, StepName.LAYOUT),
        (StepName.LAYOUT,),
    }
)


def _validate_steps(steps: list[StepName]) -> list[StepName]:
    """Raise ValueError if *steps* is not a valid pipeline sequence."""
    if StepName.LAYOUT not in steps:
        raise ValueError("'layout' must be included in steps (and must be last)")
    if steps[-1] != StepName.LAYOUT:
        raise ValueError("'layout' must be the last step")
    key = tuple(steps)
    if key not in _VALID_STEP_SEQUENCES:
        valid = [list(s) for s in sorted(_VALID_STEP_SEQUENCES, key=len)]
        raise ValueError(
            f"Invalid step sequence {[s.value for s in steps]!r}. "
            f"Valid sequences: {[[s.value for s in seq] for seq in valid]}"
        )
    return steps


class PartSpec(BaseModel):
    """A single part entry in a job file.

    When *steps* is ``None`` the part inherits ``[pipeline].default_steps``.
    Resolve effective steps via :meth:`effective_steps`.
    """

    path: Path
    steps: list[StepName] | None = None

    @model_validator(mode="after")
    def _validate(self) -> PartSpec:
        if self.steps is not None:
            _validate_steps(self.steps)
        return self

    def effective_steps(self, default: list[StepName]) -> list[StepName]:
        return self.steps if self.steps is not None else default


class PipelineSection(BaseModel):
    """Global pipeline defaults applied to parts that don't specify their own steps."""

    default_steps: list[StepName] = [StepName.SCALE, StepName.ORIENT, StepName.LAYOUT]

    @model_validator(mode="after")
    def _validate(self) -> PipelineSection:
        _validate_steps(self.default_steps)
        return self


class AppSettings(BaseModel):
    printer: PrinterSection
    scaling: ScalingSection = Field(default_factory=ScalingSection)
    packing: PackingSection = Field(default_factory=PackingSection)
    orientation: OrientationSection = Field(default_factory=OrientationSection)
    pipeline: PipelineSection = Field(default_factory=PipelineSection)
    parts: list[PartSpec] = []
