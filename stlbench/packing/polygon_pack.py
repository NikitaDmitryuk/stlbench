"""Polygon-based 2-D bin packing using Shapely.

Instead of axis-aligned bounding rectangles, every part is represented by its
exact XY shadow (see ``polygon_footprint.mesh_to_xy_shadow``).  The minimum
distance enforced between any two parts is the physical gap in mm — that is,
``gap_mm`` is the true surface-to-surface clearance, not a bounding-box margin.

Algorithm
---------
Parts are sorted by area (largest first) and placed onto plates one by one
using a bottom-left fill:

1.  For each part try both orientations (0° and 90° around Z).
2.  Scan candidate positions on a regular grid from the bottom-left corner of
    the bed (smallest Y, then smallest X).
3.  Accept the first position where the candidate polygon does not intersect the
    *forbidden zone* = union of already-placed polygons each dilated by
    ``gap_mm``.  This guarantees a physical gap of exactly ``gap_mm`` between
    any two part surfaces.
4.  If the part cannot be placed on the current plate, defer it to the next one.

The result is fully compatible with the existing ``PackedPlate`` / ``PackedRect``
data structures and all downstream export code.
"""

from __future__ import annotations

from shapely import affinity
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from stlbench.packing.rectpack_plate import PackedPlate, PackedRect


def _normalize(poly: BaseGeometry) -> BaseGeometry:
    """Translate *poly* so its bounding-box lower-left corner is at the origin."""
    minx, miny, _, _ = poly.bounds
    return affinity.translate(poly, -minx, -miny)


def _try_place_one(
    shadow: BaseGeometry,
    part_idx: int,
    forbidden: BaseGeometry | None,
    bed_w: float,
    bed_h: float,
    grid_step_mm: float,
) -> tuple[PackedRect, BaseGeometry] | None:
    """Try to place *shadow* somewhere on the plate.

    Tries 0° then 90° rotation.  Scans placement positions in bottom-left
    order (smallest Y first, then smallest X).

    Returns ``(PackedRect, placed_polygon)`` on success, ``None`` if no valid
    position exists in either rotation.  The returned *placed_polygon* is the
    shadow translated to its final position (NOT buffered — buffering is the
    caller's responsibility so it can use a consistent gap value).
    """
    shadow_0 = _normalize(shadow)
    shadow_90 = _normalize(affinity.rotate(shadow_0, 90, origin=(0, 0)))

    prep_forbidden = prep(forbidden) if forbidden is not None else None

    for rotated, s in ((False, shadow_0), (True, shadow_90)):
        _, _, w, h = s.bounds  # minx = miny = 0 after _normalize
        x_max = bed_w - w
        y_max = bed_h - h
        if x_max < -1e-6 or y_max < -1e-6:
            continue  # doesn't fit in this rotation

        y = 0.0
        while y <= y_max + 1e-9:
            ty = min(y, y_max)
            x = 0.0
            while x <= x_max + 1e-9:
                tx = min(x, x_max)
                candidate = affinity.translate(s, tx, ty)

                if prep_forbidden is None or not prep_forbidden.intersects(candidate):
                    return (
                        PackedRect(
                            part_index=part_idx,
                            x=tx,
                            y=ty,
                            width=w,
                            height=h,
                            rotated=rotated,
                        ),
                        candidate,
                    )
                x += grid_step_mm
            y += grid_step_mm

    return None


def _pack_plate(
    part_indices: list[int],
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    plate_idx: int,
) -> tuple[PackedPlate, list[int]]:
    """Pack as many of *part_indices* as possible onto one plate.

    Returns ``(PackedPlate, unplaced_indices)``.  The forbidden zone is built
    incrementally: each placed polygon is dilated by ``gap_mm`` and unioned
    into the running forbidden zone.
    """
    forbidden: BaseGeometry | None = None
    rects: list[PackedRect] = []
    unplaced: list[int] = []

    for orig_idx in part_indices:
        result = _try_place_one(
            polygons[orig_idx],
            orig_idx,
            forbidden,
            bed_w,
            bed_h,
            grid_step_mm,
        )
        if result is not None:
            rect, placed_poly = result
            rects.append(rect)
            buffered = placed_poly.buffer(gap_mm)
            forbidden = buffered if forbidden is None else forbidden.union(buffered)
        else:
            unplaced.append(orig_idx)

    return PackedPlate(index=plate_idx, rects=tuple(rects)), unplaced


def pack_polygons_on_plates(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float = 1.0,
    max_plates: int = 64,
) -> list[PackedPlate]:
    """Pack 2-D polygons onto the minimum number of plates.

    Parts are sorted by area (largest first) then packed greedily: each plate
    is filled as much as possible before a new one is opened.

    Parameters
    ----------
    polygons:
        XY shadows of each mesh.  Index in the list becomes ``part_index`` in
        the output ``PackedRect`` items.
    bed_w, bed_h:
        Build plate dimensions in mm.
    gap_mm:
        Minimum physical surface-to-surface distance between any two parts.
    grid_step_mm:
        Placement grid resolution in mm.  Smaller values give denser packing
        at the cost of runtime.  Default 1.0 mm is suitable for most parts.
    max_plates:
        Safety limit; raises ``RuntimeError`` if exceeded.
    """
    if bed_w <= 0 or bed_h <= 0:
        raise ValueError("bed dimensions must be positive.")
    if not polygons:
        return []

    # Sort by area descending — placing large parts first yields better results.
    order = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)

    # Pre-check: every part must fit on the bed in at least one rotation.
    for orig_idx in order:
        minx, miny, maxx, maxy = polygons[orig_idx].bounds
        w = maxx - minx
        h = maxy - miny
        if not ((w <= bed_w and h <= bed_h) or (h <= bed_w and w <= bed_h)):
            raise RuntimeError(
                f"Part {orig_idx} ({w:.1f}×{h:.1f} mm) does not fit on the bed "
                f"({bed_w:.1f}×{bed_h:.1f} mm). Reduce packing.gap_mm or model scale."
            )

    plates: list[PackedPlate] = []
    remaining = order

    while remaining:
        if len(plates) >= max_plates:
            raise RuntimeError(f"Exceeded max_plates={max_plates}; not all parts could be placed.")
        plate, remaining = _pack_plate(
            remaining, polygons, bed_w, bed_h, gap_mm, grid_step_mm, len(plates)
        )
        if not plate.rects:
            raise RuntimeError(
                "polygon_pack: could not place any part on the current plate. "
                "Check part sizes or gap_mm."
            )
        plates.append(plate)

    return plates


def try_pack_polygons_single_plate(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float = 1.0,
) -> PackedPlate | None:
    """Try to pack **all** *polygons* onto a single plate.

    Returns the ``PackedPlate`` on success, ``None`` if any part cannot be
    placed.  Used by the autopack bisection search.
    """
    order = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)

    for orig_idx in order:
        minx, miny, maxx, maxy = polygons[orig_idx].bounds
        w = maxx - minx
        h = maxy - miny
        if not ((w <= bed_w and h <= bed_h) or (h <= bed_w and w <= bed_h)):
            return None

    plate, unplaced = _pack_plate(order, polygons, bed_w, bed_h, gap_mm, grid_step_mm, 0)
    return None if unplaced else plate


def footprints_to_box_polygons(footprints: list[tuple[float, float]]) -> list[BaseGeometry]:
    """Convert ``(width, height)`` bounding-box tuples to Shapely box polygons.

    Convenience helper for code that only has AABB footprints (e.g. autopack).
    """
    return [shapely_box(0.0, 0.0, fw, fh) for fw, fh in footprints]
