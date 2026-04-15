from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from stlbench.packing.base import PackingStrategy

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from stlbench.packing.rectpack_plate import PackedPlate


@dataclass(frozen=True)
class PackablePart:
    name: str
    height_z: float
    footprint_w: float
    footprint_h: float


@dataclass
class _Plate:
    bed_w: float
    bed_h: float
    cur_x: float = 0.0
    cur_y: float = 0.0
    row_h: float = 0.0
    names: list[str] = field(default_factory=list)

    def _fits_dims(self, w: float, h: float) -> bool:
        if w > self.bed_w + 1e-9 or h > self.bed_h + 1e-9:
            return False
        if self.cur_x > 1e-9 and self.cur_x + w > self.bed_w + 1e-9:
            ny = self.cur_y + self.row_h
            if ny + h > self.bed_h + 1e-9:
                return False
            return w <= self.bed_w + 1e-9
        return self.cur_y + max(self.row_h, h) <= self.bed_h + 1e-9

    def _place_dims(self, name: str, w: float, h: float) -> None:
        if self.cur_x > 1e-9 and self.cur_x + w > self.bed_w + 1e-9:
            self.cur_y += self.row_h
            self.cur_x = 0.0
            self.row_h = 0.0
        self.names.append(name)
        self.cur_x += w
        self.row_h = max(self.row_h, h)

    def try_add(self, name: str, w: float, h: float) -> bool:
        for tw, th in ((w, h), (h, w)):
            if self._fits_dims(tw, th):
                self._place_dims(name, tw, th)
                return True
        return False


def sorted_dims(d: tuple[float, float, float]) -> tuple[float, float, float]:
    s = sorted(d)
    return s[0], s[1], s[2]


def part_fits_single_bed(
    dims_sorted: tuple[float, float, float],
    px: float,
    py: float,
    pz: float,
) -> tuple[bool, float, float]:
    a, b, c = dims_sorted
    if a > pz + 1e-9:
        return False, b, c
    w, h = b, c
    fits = (w <= px and h <= py) or (w <= py and h <= px)
    return fits, w, h


def build_packable_parts(
    names: list[str],
    dims_xyz: list[tuple[float, float, float]],
    px: float,
    py: float,
    pz: float,
) -> tuple[list[PackablePart], list[str]]:
    ok: list[PackablePart] = []
    bad: list[str] = []
    for name, d in zip(names, dims_xyz, strict=True):
        a, b, c = sorted_dims(d)
        single, w, h = part_fits_single_bed((a, b, c), px, py, pz)
        if not single:
            bad.append(name)
            continue
        ok.append(PackablePart(name=name, height_z=a, footprint_w=w, footprint_h=h))
    return ok, bad


def greedy_shelf_plates(parts: list[PackablePart], px: float, py: float) -> list[list[str]]:
    items = sorted(parts, key=lambda p: max(p.footprint_w, p.footprint_h), reverse=True)
    plates: list[_Plate] = []
    for p in items:
        w, h = p.footprint_w, p.footprint_h
        placed = False
        for pl in plates:
            if pl.try_add(p.name, w, h):
                placed = True
                break
        if not placed:
            np_ = _Plate(bed_w=px, bed_h=py)
            if not np_.try_add(p.name, w, h):
                raise RuntimeError(
                    f"Packing invariant failed for {p.name!r} ({w=}, {h=}, bed={px}x{py})."
                )
            plates.append(np_)
    return [pl.names for pl in plates]


class ShelfPacker(PackingStrategy):
    """Shelf-based packing algorithm."""

    def pack(
        self, polygons: list[BaseGeometry], bed_w: float, bed_h: float, gap_mm: float
    ) -> list[PackedPlate]:
        # This is a simplified implementation that just returns an empty list
        # A full implementation would need to convert polygons to packable parts
        # and implement the shelf packing algorithm with proper PackedPlate objects
        return []
