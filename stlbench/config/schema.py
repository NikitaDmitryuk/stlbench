from __future__ import annotations

from pydantic import BaseModel, Field


class PrinterSection(BaseModel):
    name: str = ""
    width_mm: float = Field(gt=0)
    depth_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)


class ScalingSection(BaseModel):
    bed_margin: float = Field(default=0.0, ge=0.0, lt=1.0)
    post_fit_scale: float = Field(default=1.0, gt=0.0)


class PackingSection(BaseModel):
    gap_mm: float = Field(default=2.0, ge=0.0)


class HollowSection(BaseModel):
    """Voxel hollow parameters for ``scale --hollow``."""

    wall_thickness_mm: float = Field(default=2.0, gt=0.0)
    voxel_mm: float = Field(default=0.5, gt=0.0)


class AppSettings(BaseModel):
    printer: PrinterSection
    scaling: ScalingSection = Field(default_factory=ScalingSection)
    packing: PackingSection = Field(default_factory=PackingSection)
    hollow: HollowSection = Field(default_factory=HollowSection)
