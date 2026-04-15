from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from stlbench.packing.rectpack_plate import PackedPlate


class PackingStrategy(ABC):
    """Interface for 2D packing algorithms.

    All implementations must accept XY shadows (Shapely polygons)
    and return a list of plates with placement coordinates.
    Part index in ``polygons`` corresponds to ``PackedRect.part_index``.
    """

    @abstractmethod
    def pack(
        self,
        polygons: list[BaseGeometry],
        bed_w: float,
        bed_h: float,
        gap_mm: float,
    ) -> list[PackedPlate]:
        """Pack polygons onto minimum number of plates.

        Args:
            polygons:  XY shadows of parts (one per part).
            bed_w:     Bed width along X in mm.
            bed_h:     Bed depth along Y in mm.
            gap_mm:    Minimum physical gap between parts.

        Returns:
            List of PackedPlate, one per used plate.
        """
        ...

    def try_pack_single_plate(
        self,
        polygons: list[BaseGeometry],
        bed_w: float,
        bed_h: float,
        gap_mm: float,
    ) -> PackedPlate | None:
        """Try to fit all parts on a single plate.

        Base implementation uses pack() and checks plate count.
        Packers may override for more efficient implementation.
        """
        plates = self.pack(polygons, bed_w, bed_h, gap_mm)
        if len(plates) == 1 and len(plates[0].rects) == len(polygons):
            return plates[0]
        return None


class PackingOptions:
    """Packing algorithm parameters passed when creating strategy."""

    grid_step_mm: float = 2.0
    max_plates: int = 64
    on_placed: Callable[[], None] | None = None
