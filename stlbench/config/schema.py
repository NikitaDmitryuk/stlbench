from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PrinterSection(BaseModel):
    name: str = ""
    width_mm: float = Field(gt=0)
    depth_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)


class ScalingSection(BaseModel):
    bed_margin: float = Field(default=0.0, ge=0.0, lt=1.0)
    supports_scale: float = Field(default=1.0, gt=0.0)


class OrientationSection(BaseModel):
    mode: Literal["axis", "free"] = "axis"
    samples: int = Field(default=2048, ge=1)
    seed: int = 0


class PackingSection(BaseModel):
    report: bool = True
    algorithm: Literal["shelf", "rectpack"] = "rectpack"
    gap_mm: float = Field(default=2.0, ge=0.0)


class HollowSection(BaseModel):
    enabled: bool = False
    backend: Literal["none", "open3d_voxel"] = "none"
    wall_thickness_mm: float = Field(default=2.0, gt=0.0)
    voxel_mm: float = Field(default=0.5, gt=0.0)


class SupportsSection(BaseModel):
    """
    Заглушка для совместимости TOML. Генерации опор в пакете нет — только слайсер.
    """

    enabled: bool = False
    backend: Literal["none", "external"] = "none"
    external_command_template: str = ""

    @field_validator("external_command_template", mode="before")
    @classmethod
    def _empty_str(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v)


class AppSettings(BaseModel):
    printer: PrinterSection
    scaling: ScalingSection = Field(default_factory=ScalingSection)
    orientation: OrientationSection = Field(default_factory=OrientationSection)
    packing: PackingSection = Field(default_factory=PackingSection)
    hollow: HollowSection = Field(default_factory=HollowSection)
    supports: SupportsSection = Field(default_factory=SupportsSection)
