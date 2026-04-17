"""Polygon-based 2-D bin packing using No-Fit Polygon (NFP) placement.

Each part is represented by its exact XY shadow. Placement uses the
No-Fit Polygon technique:

1. Sort parts by area (largest first).
2. For each part try _PLACEMENT_ROTATION_COUNT orientations evenly spaced over 360°.
3. For each orientation compute the valid placement region:
   - Inner NFP: rectangular bed constraint (shapely_box).
   - Outer NFP: MinkowskiSum(placed_buffered_i, reflect(new_part)) for each placed part.
   - valid = inner_nfp.difference(union(outer_nfps))
4. Place at the bottom-left vertex of the valid region.
5. Accept the rotation / position that minimises (y, x) across all angles.

This gives exact gap enforcement and sub-millimetre packing accuracy without
any grid discretisation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyclipper
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from stlbench.packing.rectpack_plate import PackedPlate, PackedRect

_PLACEMENT_ROTATION_COUNT = 36
_PLACEMENT_ANGLES: tuple[float, ...] = tuple(
    360.0 * i / _PLACEMENT_ROTATION_COUNT for i in range(_PLACEMENT_ROTATION_COUNT)
)

# Clipper integer scale: 1 unit = 1 µm → coordinates in range ±2^31
_CLIPPER_SCALE = 1_000


def _normalize(poly: BaseGeometry) -> BaseGeometry:
    """Translate *poly* so its bounding-box lower-left corner is at the origin."""
    minx, miny, _, _ = poly.bounds
    return affinity.translate(poly, -minx, -miny)


# ---------------------------------------------------------------------------
# NFP helpers
# ---------------------------------------------------------------------------


def _poly_to_int(poly: Polygon) -> list[tuple[int, int]]:
    """Exterior ring of *poly* → Clipper integer path (closing point removed)."""
    coords = np.asarray(poly.exterior.coords)[:-1]
    scaled = (coords * _CLIPPER_SCALE).astype(np.int64)
    return [(int(x), int(y)) for x, y in scaled]


def _int_paths_to_shapely(paths: list[list[tuple[int, int]]]) -> BaseGeometry | None:
    """Convert pyclipper result paths to a Shapely (Multi)Polygon."""
    polys: list[Polygon] = []
    for path in paths:
        if len(path) < 3:
            continue
        coords = [(x / _CLIPPER_SCALE, y / _CLIPPER_SCALE) for x, y in path]
        try:
            p = Polygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
        except Exception:
            continue
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return shapely.union_all(np.asarray(polys, dtype=object))


def _largest_poly(geom: BaseGeometry) -> Polygon | None:
    """Return the largest-area Polygon from *geom* (handles Multi / Collection)."""
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    candidates: list[Polygon] = []
    for g in getattr(geom, "geoms", []):
        if isinstance(g, Polygon) and not g.is_empty:
            candidates.append(g)
    return max(candidates, key=lambda g: g.area) if candidates else None


def _outer_nfp(fixed: BaseGeometry, moving: BaseGeometry) -> BaseGeometry | None:
    """Compute NFP(fixed, moving) via Minkowski sum.

    Returns the set of reference-point positions for *moving* that would cause
    overlap with *fixed*.  *fixed* should already be dilated by gap_mm.
    """
    fixed_poly = fixed if isinstance(fixed, Polygon) else _largest_poly(fixed)
    if fixed_poly is None:
        return None

    moving_poly = moving if isinstance(moving, Polygon) else _largest_poly(moving)
    if moving_poly is None:
        return None

    # Reflect moving through the origin: NFP(A, B) = A ⊕ (−B)
    neg_moving = affinity.scale(moving_poly, xfact=-1, yfact=-1, origin=(0, 0))

    fixed_int = _poly_to_int(fixed_poly)
    neg_moving_int = _poly_to_int(neg_moving)

    if len(fixed_int) < 3 or len(neg_moving_int) < 3:
        return None

    try:
        result = pyclipper.MinkowskiSum(fixed_int, neg_moving_int, True)
    except Exception:
        return None

    return _int_paths_to_shapely(result) if result else None


def _bottom_left_vertex(region: BaseGeometry) -> tuple[float, float] | None:
    """Return the bottom-left (min y, then min x) vertex of *region*."""
    coords: list[tuple[float, float]] = []
    if isinstance(region, Polygon):
        coords.extend((float(c[0]), float(c[1])) for c in region.exterior.coords)
    elif isinstance(region, MultiPolygon):
        for g in region.geoms:
            coords.extend((float(c[0]), float(c[1])) for c in g.exterior.coords)
    else:
        for g in getattr(region, "geoms", []):
            if isinstance(g, Polygon):
                coords.extend((float(c[0]), float(c[1])) for c in g.exterior.coords)
    if not coords:
        return None
    return min(coords, key=lambda c: (c[1], c[0]))


# ---------------------------------------------------------------------------
# Plate packer
# ---------------------------------------------------------------------------

_FORBIDDEN_SIMPLIFY_EVERY = 5
_FORBIDDEN_SIMPLIFY_TOL = 1.0


def _try_place_one(
    shadow: BaseGeometry,
    part_idx: int,
    placed_buffered: list[BaseGeometry],
    forbidden: BaseGeometry | None,
    bed_w: float,
    bed_h: float,
    _grid_step_mm: float,
) -> tuple[PackedRect, BaseGeometry] | None:
    """Find the best NFP bottom-left placement for *shadow*.

    Tries all rotation angles and returns the placement with the smallest
    (y, x) reference-point position, or ``None`` if the part cannot be placed.
    """
    prep_forbidden = prep(forbidden) if forbidden is not None else None

    best_pos: tuple[float, float] | None = None
    best_angle = 0.0
    best_w = 0.0
    best_h = 0.0
    best_candidate: BaseGeometry | None = None

    for angle in _PLACEMENT_ANGLES:
        s = _normalize(affinity.rotate(shadow, angle, origin=(0, 0)))
        _, _, w, h = s.bounds  # minx=miny=0 after _normalize
        x_max = bed_w - w
        y_max = bed_h - h
        if x_max < -1e-6 or y_max < -1e-6:
            continue

        # Clamp to zero; use a tiny epsilon to avoid degenerate Shapely boxes.
        x_max = max(x_max, 0.0)
        y_max = max(y_max, 0.0)
        inner_nfp = shapely_box(0.0, 0.0, max(x_max, 1e-6), max(y_max, 1e-6))

        if placed_buffered:
            outer_nfps: list[BaseGeometry] = []
            for buf in placed_buffered:
                nfp = _outer_nfp(buf, s)
                if nfp is not None and not nfp.is_empty:
                    outer_nfps.append(nfp)
            if outer_nfps:
                combined = shapely.union_all(np.asarray(outer_nfps, dtype=object))
                valid = inner_nfp.difference(combined)
            else:
                valid = inner_nfp
        else:
            valid = inner_nfp

        if valid.is_empty:
            continue

        bl = _bottom_left_vertex(valid)
        if bl is None:
            continue

        tx = max(0.0, min(float(bl[0]), x_max))
        ty = max(0.0, min(float(bl[1]), y_max))
        candidate = affinity.translate(s, tx, ty)

        # Guard against NFP numerical imprecision: allow boundary touching
        # (distance == gap_mm exactly) but reject interior overlap.
        if prep_forbidden is not None:
            inner = candidate.buffer(-1e-3)
            if not inner.is_empty and prep_forbidden.intersects(inner):
                continue

        if best_pos is None or (ty, tx) < best_pos:
            best_pos = (ty, tx)
            best_angle = angle
            best_w = w
            best_h = h
            best_candidate = candidate

    if best_pos is None or best_candidate is None:
        return None

    ty, tx = best_pos
    return (
        PackedRect(
            part_index=part_idx,
            x=tx,
            y=ty,
            width=best_w,
            height=best_h,
            rotation_deg=best_angle,
        ),
        best_candidate,
    )


def _pack_plate(
    part_indices: list[int],
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    plate_idx: int,
    on_placed: Callable[[], None] | None = None,
) -> tuple[PackedPlate, list[int]]:
    """Pack as many of *part_indices* as possible onto one plate.

    Returns ``(PackedPlate, unplaced_indices)``.
    """
    forbidden: BaseGeometry | None = None
    placed_buffered: list[BaseGeometry] = []
    rects: list[PackedRect] = []
    unplaced: list[int] = []

    for orig_idx in part_indices:
        result = _try_place_one(
            polygons[orig_idx],
            orig_idx,
            placed_buffered,
            forbidden,
            bed_w,
            bed_h,
            grid_step_mm,
        )
        if result is not None:
            rect, placed_poly = result
            rects.append(rect)
            buffered = placed_poly.buffer(gap_mm)
            placed_buffered.append(buffered)
            forbidden = buffered if forbidden is None else forbidden.union(buffered)
            if len(rects) % _FORBIDDEN_SIMPLIFY_EVERY == 0 and forbidden is not None:
                forbidden = forbidden.simplify(_FORBIDDEN_SIMPLIFY_TOL, preserve_topology=True)
            if on_placed is not None:
                on_placed()
        else:
            unplaced.append(orig_idx)

    return PackedPlate(index=plate_idx, rects=tuple(rects)), unplaced


def pack_polygons_on_plates(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float = 2.0,
    max_plates: int = 64,
    on_placed: Callable[[], None] | None = None,
) -> list[PackedPlate]:
    """Pack 2-D polygons onto the minimum number of plates using NFP placement.

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
        Kept for API compatibility; validated but not used by the NFP algorithm.
    max_plates:
        Safety limit; raises ``RuntimeError`` if exceeded.
    """
    if bed_w <= 0 or bed_h <= 0:
        raise ValueError("bed dimensions must be positive.")
    if grid_step_mm <= 0:
        raise ValueError(f"grid_step_mm must be positive, got {grid_step_mm!r}.")
    if not polygons:
        return []

    order = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)

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
            remaining, polygons, bed_w, bed_h, gap_mm, grid_step_mm, len(plates), on_placed
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
    grid_step_mm: float = 2.0,
) -> PackedPlate | None:
    """Try to pack **all** *polygons* onto a single plate.

    Returns the ``PackedPlate`` on success, ``None`` if any part cannot be
    placed.  Used by the autopack bisection search.
    """
    if grid_step_mm <= 0:
        raise ValueError(f"grid_step_mm must be positive, got {grid_step_mm!r}.")

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
    """Convert ``(width, height)`` bounding-box tuples to Shapely box polygons."""
    return [shapely_box(0.0, 0.0, fw, fh) for fw, fh in footprints]
