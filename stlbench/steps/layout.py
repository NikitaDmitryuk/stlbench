from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stlbench.domain.plate import Plate
from stlbench.domain.printer import Printer
from stlbench.packing.base import PackingStrategy
from stlbench.packing.polygon_footprint import mesh_to_xy_shadow
from stlbench.steps.base import PipelineStep, StepResult

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from stlbench.domain.part import Part


@dataclass
class LayoutStep(PipelineStep):
    """Layout parts on build plates."""

    printer: Printer
    packer: PackingStrategy
    gap_mm: float = 2.0

    def process(self, parts: list[Part]) -> StepResult:
        # Check that each part fits on the bed
        for p in parts:
            dx, dy, _ = p.extents
            if not self.printer.fits_xy(dx, dy):
                raise ValueError(
                    f"Part {p.name!r} ({dx:.1f}×{dy:.1f} mm) does not fit "
                    f"on bed {self.printer.width_mm:.1f}×{self.printer.depth_mm:.1f} mm"
                )

        # Compute XY shadows
        shadows: list[BaseGeometry] = [mesh_to_xy_shadow(p.mesh) for p in parts]

        # Pack
        packed_plates = self.packer.pack(
            shadows,
            self.printer.width_mm,
            self.printer.depth_mm,
            self.gap_mm,
        )

        # Wrap result in domain objects
        plates = [
            Plate(
                index=pp.index,
                parts=[parts[r.part_index] for r in pp.rects],
                names=[parts[r.part_index].name for r in pp.rects],
                packed=pp,
            )
            for pp in packed_plates
        ]

        return StepResult(
            parts=parts,  # original parts unchanged
            metadata={"plates": plates},
        )
