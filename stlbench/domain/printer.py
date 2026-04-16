from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stlbench.config.schema import AppSettings


@dataclass(frozen=True)
class Printer:
    width_mm: float  # X
    depth_mm: float  # Y
    height_mm: float  # Z
    name: str = ""

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_tuple(cls, xyz: tuple[float, float, float], name: str = "") -> Printer:
        return cls(width_mm=xyz[0], depth_mm=xyz[1], height_mm=xyz[2], name=name)

    @classmethod
    def from_settings(cls, settings: AppSettings) -> Printer:
        p = settings.printer
        return cls(p.width_mm, p.depth_mm, p.height_mm, p.name)

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.width_mm, self.depth_mm, self.height_mm

    @property
    def xy(self) -> tuple[float, float]:
        return self.width_mm, self.depth_mm

    @property
    def min_dim(self) -> float:
        return min(self.width_mm, self.depth_mm, self.height_mm)

    @property
    def sorted_dims(self) -> tuple[float, float, float]:
        return tuple(sorted(self.xyz))  # type: ignore[return-value]

    @property
    def bed_area_mm2(self) -> float:
        return self.width_mm * self.depth_mm

    # ── Checks ──────────────────────────────────────────────────────────────

    def fits_xy(self, dx: float, dy: float, eps: float = 0.0) -> bool:
        """Check if a part fits on the bed (allowing 90° rotation)."""
        w, d = self.width_mm + eps, self.depth_mm + eps
        return (dx <= w and dy <= d) or (dy <= w and dx <= d)

    def fits_xyz(self, dx: float, dy: float, dz: float, eps: float = 0.0) -> bool:
        """Check if a part fits in the build volume."""
        return self.fits_xy(dx, dy, eps) and dz <= self.height_mm + eps

    def validate(self) -> None:
        """Raise ValueError if dimensions are invalid."""
        if any(x <= 0 for x in self.xyz):
            raise ValueError(f"Printer dimensions must be positive, got {self.xyz}")

    def __str__(self) -> str:
        parts = [f"{self.width_mm:.1f}×{self.depth_mm:.1f}×{self.height_mm:.1f} mm"]
        if self.name:
            parts.insert(0, self.name)
        return " ".join(parts)
