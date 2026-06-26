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

import math
import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pyclipper
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from stlbench.packing.rectpack_plate import PackedPlate, PackedRect, pack_rectangles_on_plates

_PLACEMENT_ROTATION_COUNT = 24
_PLACEMENT_ANGLES: tuple[float, ...] = tuple(
    360.0 * i / _PLACEMENT_ROTATION_COUNT for i in range(_PLACEMENT_ROTATION_COUNT)
)

# Clipper integer scale: 1 unit = 1 µm → coordinates in range ±2^31
_CLIPPER_SCALE = 1_000

# NFP computation may use simplified polygons to keep MinkowskiSum fast, but
# final candidate acceptance is checked against exact buffered placed polygons.
_NFP_SIMPLIFY_TOL: float = 0.1
# Buffer arc resolution (segments per quarter-circle) for the forbidden zone.
# Higher resolution avoids shaving clearance around curved/rotated corners.
_NFP_BUFFER_RESOLUTION: int = 8
_GAP_SAFETY_MARGIN_MM: float = 0.25


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
    scaled = np.rint(coords * _CLIPPER_SCALE).astype(np.int64)
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
    coords = _candidate_vertices(region)
    return coords[0] if coords else None


def _candidate_vertices(region: BaseGeometry) -> list[tuple[float, float]]:
    """Return placement candidate vertices sorted bottom-left first."""
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
    unique = {(round(x, 9), round(y, 9)) for x, y in coords}
    return sorted(unique, key=lambda c: (c[1], c[0]))


# ---------------------------------------------------------------------------
# Plate packer
# ---------------------------------------------------------------------------

_FORBIDDEN_SIMPLIFY_EVERY = 5
_FORBIDDEN_SIMPLIFY_TOL = 0.1
_DEFAULT_REFLOW_SWEEPS = 2
_DEFAULT_REFLOW_TIME_BUDGET_S = 30.0


def _axis_positions(max_value: float, step: float) -> list[float]:
    vals: list[float] = []
    cur = 0.0
    while cur <= max_value + 1e-9:
        vals.append(round(min(cur, max_value), 9))
        cur += step
    vals.append(round(max_value, 9))
    return sorted(set(vals))


def _bounded_axis_positions(
    max_value: float,
    size: float,
    existing: list[BaseGeometry],
    gap_mm: float,
    step: float,
    *,
    axis: int,
) -> list[float]:
    clearance = gap_mm + _GAP_SAFETY_MARGIN_MM
    values = {0.0, max_value}
    for geom in existing:
        minx, miny, maxx, maxy = geom.bounds
        lo = minx if axis == 0 else miny
        hi = maxx if axis == 0 else maxy
        for base in (hi + clearance, lo - size - clearance):
            for delta in (-step, 0.0, step):
                value = min(max(base + delta, 0.0), max_value)
                values.add(round(value, 9))
    return sorted(values)


def placed_polygon_from_rect(shadow: BaseGeometry, rect: PackedRect) -> BaseGeometry:
    """Recreate the placed polygon for *shadow* using the export/packer transform."""
    s = _normalize(affinity.rotate(shadow, rect.rotation_deg, origin=(0, 0)))
    return affinity.translate(s, rect.x, rect.y)


def _clearance_ok(
    candidate: BaseGeometry,
    placed_clearance: list[BaseGeometry],
    gap_mm: float,
) -> bool:
    min_clearance = gap_mm + _GAP_SAFETY_MARGIN_MM
    return all(
        candidate.distance(existing) >= min_clearance - 1e-6 for existing in placed_clearance
    )


def _within_bed(candidate: BaseGeometry, bed_w: float, bed_h: float) -> bool:
    minx, miny, maxx, maxy = candidate.bounds
    eps = 1e-6
    return bool(minx >= -eps and miny >= -eps and maxx <= bed_w + eps and maxy <= bed_h + eps)


def _centroid_xy(geom: BaseGeometry) -> tuple[float, float]:
    c = geom.centroid
    return float(c.x), float(c.y)


def _candidate_spread_score(candidate: BaseGeometry, existing: list[BaseGeometry]) -> float:
    if not existing:
        return 0.0
    distances = [float(candidate.distance(geom)) for geom in existing]
    cx, cy = _centroid_xy(candidate)
    ex = 0.0
    ey = 0.0
    for geom in existing:
        gx, gy = _centroid_xy(geom)
        ex += gx
        ey += gy
    ex /= len(existing)
    ey /= len(existing)
    centroid_distance = math.hypot(cx - ex, cy - ey)
    return float(
        min(distances) * 4.0 + (sum(distances) / len(distances)) + 0.05 * centroid_distance
    )


def _rect_for_candidate(
    part_idx: int,
    tx: float,
    ty: float,
    angle: float,
    width: float,
    height: float,
) -> PackedRect:
    return PackedRect(
        part_index=part_idx,
        x=tx,
        y=ty,
        width=width,
        height=height,
        rotation_deg=angle,
    )


def _grid_fallback_place(
    shadow: BaseGeometry,
    part_idx: int,
    placed_clearance: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    candidate_key: Callable[[BaseGeometry, float, float], tuple[float, ...]] | None = None,
) -> tuple[PackedRect, BaseGeometry] | None:
    """Find a strict-gap placement by bounded scanline search.

    This is intentionally used only after the NFP vertex heuristic fails.  It
    is slower, but catches obvious interior free-space placements that the
    convex-hull NFP acceleration can miss.
    """
    step = max(float(grid_step_mm), 0.1)
    best: tuple[tuple[float, ...], PackedRect, BaseGeometry] | None = None
    for angle in _PLACEMENT_ANGLES:
        s = _normalize(affinity.rotate(shadow, angle, origin=(0, 0)))
        _, _, w, h = s.bounds
        x_max = bed_w - w
        y_max = bed_h - h
        if x_max < -1e-6 or y_max < -1e-6:
            continue
        x_max = max(x_max, 0.0)
        y_max = max(y_max, 0.0)
        y_positions = _bounded_axis_positions(y_max, h, placed_clearance, gap_mm, step, axis=1)
        x_positions = _bounded_axis_positions(x_max, w, placed_clearance, gap_mm, step, axis=0)
        for ty in y_positions:
            for tx in x_positions:
                candidate = affinity.translate(s, tx, ty)
                if not _within_bed(candidate, bed_w, bed_h):
                    continue
                if not _clearance_ok(candidate, placed_clearance, gap_mm):
                    continue
                rect = _rect_for_candidate(part_idx, tx, ty, angle, w, h)
                if candidate_key is None:
                    return rect, candidate
                key = candidate_key(candidate, ty, tx)
                if best is None or key < best[0]:
                    best = (key, rect, candidate)

    return None if best is None else (best[1], best[2])


def _try_place_one(
    shadow: BaseGeometry,
    part_idx: int,
    placed_buffered: list[BaseGeometry],
    placed_clearance: list[BaseGeometry],
    forbidden: BaseGeometry | None,
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    candidate_key: Callable[[BaseGeometry, float, float], tuple[float, ...]] | None = None,
) -> tuple[PackedRect, BaseGeometry] | None:
    """Find the best NFP bottom-left placement for *shadow*.

    Tries all rotation angles and returns the placement with the smallest
    (y, x) reference-point position, or ``None`` if the part cannot be placed.
    """
    prep_forbidden = prep(forbidden) if forbidden is not None else None

    best_pos: tuple[float, float] | None = None
    best_key: tuple[float, ...] | None = None
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
            # Convex hull of the rotated shadow: 10-30 vertices instead of
            # 50-200, making MinkowskiSum ~200x faster. The hull overapproximates
            # the forbidden zone (conservative), but exact overlap is still checked
            # via prep_forbidden below.
            s_nfp = s.convex_hull
            outer_nfps: list[BaseGeometry] = []
            for buf in placed_buffered:
                nfp = _outer_nfp(buf, s_nfp)
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

        vertices = _candidate_vertices(valid)
        if not vertices:
            continue
        for vx, vy in vertices:
            tx = max(0.0, min(float(vx), x_max))
            ty = max(0.0, min(float(vy), y_max))
            candidate = affinity.translate(s, tx, ty)
            if not _within_bed(candidate, bed_w, bed_h):
                continue

            # Guard against NFP numerical imprecision: allow boundary touching
            # (distance == gap_mm exactly) but reject interior overlap.
            if prep_forbidden is not None:
                inner = candidate.buffer(-1e-3)
                if not inner.is_empty and prep_forbidden.intersects(inner):
                    continue
            if not _clearance_ok(candidate, placed_clearance, gap_mm):
                continue

            key = candidate_key(candidate, ty, tx) if candidate_key else (ty, tx)
            if best_key is None or key < best_key:
                best_pos = (ty, tx)
                best_key = key
                best_angle = angle
                best_w = w
                best_h = h
                best_candidate = candidate
            if candidate_key is None:
                break
        if candidate_key is None and best_pos is not None and best_pos[0] < 1e-6:
            break  # y=0 is optimal; no later rotation can do better

    if best_pos is None or best_candidate is None:
        return _grid_fallback_place(
            shadow,
            part_idx,
            placed_clearance,
            bed_w,
            bed_h,
            gap_mm,
            grid_step_mm,
            candidate_key,
        )

    ty, tx = best_pos
    return (
        _rect_for_candidate(part_idx, tx, ty, best_angle, best_w, best_h),
        best_candidate,
    )


def _add_placed_geometry(
    placed_poly: BaseGeometry,
    placed_clearance: list[BaseGeometry],
    placed_buffered: list[BaseGeometry],
    forbidden: BaseGeometry | None,
    gap_mm: float,
    placed_count: int,
) -> BaseGeometry | None:
    placed_clearance.append(placed_poly)
    # Simplify shadow before NFP buffering: fewer vertices → faster
    # MinkowskiSum in subsequent placements. This is only an acceleration
    # structure; placed_clearance above is the source of truth for gap.
    simple = placed_poly.simplify(_NFP_SIMPLIFY_TOL, preserve_topology=True)
    buffered = simple.buffer(gap_mm, quad_segs=_NFP_BUFFER_RESOLUTION)
    # Convex hull reduces vertex count from ~120 to ~20, making MinkowskiSum
    # much faster. Placement is conservative, but exact clearance is verified.
    placed_buffered.append(buffered.convex_hull)
    next_forbidden = buffered if forbidden is None else forbidden.union(buffered)
    if placed_count % _FORBIDDEN_SIMPLIFY_EVERY == 0:
        next_forbidden = next_forbidden.simplify(_FORBIDDEN_SIMPLIFY_TOL, preserve_topology=True)
    return next_forbidden


def _build_placement_state(
    rects: list[PackedRect],
    polygons: list[BaseGeometry],
    gap_mm: float,
) -> tuple[list[BaseGeometry], list[BaseGeometry], BaseGeometry | None]:
    placed_clearance: list[BaseGeometry] = []
    placed_buffered: list[BaseGeometry] = []
    forbidden: BaseGeometry | None = None
    for rect in rects:
        placed_poly = placed_polygon_from_rect(polygons[rect.part_index], rect)
        forbidden = _add_placed_geometry(
            placed_poly,
            placed_clearance,
            placed_buffered,
            forbidden,
            gap_mm,
            len(placed_clearance) + 1,
        )
    return placed_clearance, placed_buffered, forbidden


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
    placed_clearance: list[BaseGeometry] = []
    placed_buffered: list[BaseGeometry] = []
    rects: list[PackedRect] = []
    unplaced: list[int] = []

    for orig_idx in part_indices:
        result = _try_place_one(
            polygons[orig_idx],
            orig_idx,
            placed_buffered,
            placed_clearance,
            forbidden,
            bed_w,
            bed_h,
            gap_mm,
            grid_step_mm,
        )
        if result is not None:
            rect, placed_poly = result
            rects.append(rect)
            forbidden = _add_placed_geometry(
                placed_poly,
                placed_clearance,
                placed_buffered,
                forbidden,
                gap_mm,
                len(rects),
            )
            if on_placed is not None:
                on_placed()
        else:
            unplaced.append(orig_idx)

    return PackedPlate(index=plate_idx, rects=tuple(rects)), unplaced


def _compact_plates(
    plates: list[PackedPlate],
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
) -> list[PackedPlate]:
    if len(plates) <= 1:
        return plates

    plate_rects: list[list[PackedRect]] = [list(plate.rects) for plate in plates]
    moved = True
    while moved:
        moved = False
        for source_i in range(len(plate_rects) - 1, 0, -1):
            source_rects = plate_rects[source_i]
            for rect in list(source_rects):
                placed = False
                for target_i in range(source_i):
                    target_rects = plate_rects[target_i]
                    placed_clearance, placed_buffered, forbidden = _build_placement_state(
                        target_rects, polygons, gap_mm
                    )
                    result = _try_place_one(
                        polygons[rect.part_index],
                        rect.part_index,
                        placed_buffered,
                        placed_clearance,
                        forbidden,
                        bed_w,
                        bed_h,
                        gap_mm,
                        grid_step_mm,
                    )
                    if result is None:
                        continue
                    new_rect, _placed_poly = result
                    target_rects.append(new_rect)
                    source_rects.remove(rect)
                    moved = True
                    placed = True
                    break
                if placed:
                    continue

        if moved:
            plate_rects = [rects for rects in plate_rects if rects]

    return [PackedPlate(index=index, rects=tuple(rects)) for index, rects in enumerate(plate_rects)]


def _plate_height_range(rects: Sequence[PackedRect], part_heights: Sequence[float] | None) -> float:
    if part_heights is None or len(rects) <= 1:
        return 0.0
    heights = [float(part_heights[r.part_index]) for r in rects]
    return max(heights) - min(heights)


def _height_score(plates: Sequence[PackedPlate], part_heights: Sequence[float] | None) -> float:
    if part_heights is None:
        return 0.0
    ranges = [_plate_height_range(plate.rects, part_heights) for plate in plates]
    return max(ranges, default=0.0) * 4.0 + sum(ranges)


def _spread_score(plates: Sequence[PackedPlate], polygons: list[BaseGeometry]) -> float:
    total = 0.0
    for plate in plates:
        placed = [placed_polygon_from_rect(polygons[r.part_index], r) for r in plate.rects]
        if len(placed) <= 1:
            continue
        nearest_sum = 0.0
        min_distance = float("inf")
        for i, geom in enumerate(placed):
            distances = [geom.distance(other) for j, other in enumerate(placed) if i != j]
            nearest = min(distances)
            nearest_sum += nearest
            min_distance = min(min_distance, nearest)

        cells: set[tuple[int, int]] = set()
        minx = min(g.bounds[0] for g in placed)
        miny = min(g.bounds[1] for g in placed)
        maxx = max(g.bounds[2] for g in placed)
        maxy = max(g.bounds[3] for g in placed)
        span_x = max(maxx - minx, 1e-9)
        span_y = max(maxy - miny, 1e-9)
        for geom in placed:
            cx, cy = _centroid_xy(geom)
            gx = min(2, max(0, int(3 * (cx - minx) / span_x)))
            gy = min(2, max(0, int(3 * (cy - miny) / span_y)))
            cells.add((gx, gy))
        total += min_distance * 4.0 + nearest_sum / len(placed) + len(cells)
    return total


def _layout_score(
    plates: Sequence[PackedPlate],
    polygons: list[BaseGeometry],
    part_heights: Sequence[float] | None,
) -> tuple[int, float, float]:
    return (len(plates), _height_score(plates, part_heights), -_spread_score(plates, polygons))


def _rectangle_footprint(poly: BaseGeometry) -> tuple[float, float] | None:
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return None
    if abs(float(poly.area) - (w * h)) > max(1e-6, w * h * 1e-6):
        return None
    return float(w), float(h)


def _try_rectangular_fast_path(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    max_plates: int,
) -> list[PackedPlate] | None:
    footprints: list[tuple[float, float]] = []
    for poly in polygons:
        footprint = _rectangle_footprint(poly)
        if footprint is None:
            return None
        footprints.append(footprint)
    return pack_rectangles_on_plates(footprints, bed_w, bed_h, gap_mm, max_plates=max_plates)


def _renumber_plates(plates: Iterable[PackedPlate]) -> list[PackedPlate]:
    return [
        PackedPlate(index=i, rects=tuple(plate.rects))
        for i, plate in enumerate(plates)
        if plate.rects
    ]


def _offset_plates(plates: list[PackedPlate], dx: float, dy: float) -> list[PackedPlate]:
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return plates
    out: list[PackedPlate] = []
    for plate in plates:
        rects = tuple(
            PackedRect(
                part_index=r.part_index,
                x=r.x + dx,
                y=r.y + dy,
                width=r.width,
                height=r.height,
                rotation_deg=r.rotation_deg,
            )
            for r in plate.rects
        )
        out.append(PackedPlate(index=plate.index, rects=rects))
    return out


def _pack_order(
    order: list[int],
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    max_plates: int,
    on_placed: Callable[[], None] | None = None,
) -> list[PackedPlate]:
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

    return _compact_plates(plates, polygons, bed_w, bed_h, gap_mm, grid_step_mm)


def _ordering_candidates(
    order: list[int],
    polygons: list[BaseGeometry],
    part_heights: Sequence[float] | None,
) -> list[list[int]]:
    candidates: list[list[int]] = [order]
    if part_heights is not None:
        by_height_desc = sorted(order, key=lambda i: (-float(part_heights[i]), -polygons[i].area))
        by_height_asc = list(reversed(by_height_desc))
        by_height_area = sorted(
            order,
            key=lambda i: (round(float(part_heights[i]) / 5.0), -polygons[i].area),
        )
        candidates.extend([by_height_desc, by_height_asc, by_height_area])

    by_long_side = sorted(
        order,
        key=lambda i: (
            -max(
                polygons[i].bounds[2] - polygons[i].bounds[0],
                polygons[i].bounds[3] - polygons[i].bounds[1],
            ),
            -polygons[i].area,
        ),
    )
    candidates.append(by_long_side)

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _try_pack_order_into_k(
    order: list[int],
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    k: int,
    part_heights: Sequence[float] | None,
) -> list[PackedPlate] | None:
    rects_by_plate: list[list[PackedRect]] = [[] for _ in range(k)]

    for part_idx in order:
        best: tuple[tuple[float, float, int], int, PackedRect] | None = None
        for plate_idx in range(k):
            placed_clearance, placed_buffered, forbidden = _build_placement_state(
                rects_by_plate[plate_idx], polygons, gap_mm
            )
            result = _try_place_one(
                polygons[part_idx],
                part_idx,
                placed_buffered,
                placed_clearance,
                forbidden,
                bed_w,
                bed_h,
                gap_mm,
                grid_step_mm,
            )
            if result is None:
                continue
            rect, _placed = result
            trial_rects = [*rects_by_plate[plate_idx], rect]
            height_range = _plate_height_range(trial_rects, part_heights)
            area_load = sum(polygons[r.part_index].area for r in trial_rects)
            key = (height_range, area_load, plate_idx)
            if best is None or key < best[0]:
                best = (key, plate_idx, rect)
        if best is None:
            return None
        _key, plate_idx, rect = best
        rects_by_plate[plate_idx].append(rect)

    plates = _renumber_plates(
        PackedPlate(index=i, rects=tuple(rects)) for i, rects in enumerate(rects_by_plate)
    )
    return _compact_plates(plates, polygons, bed_w, bed_h, gap_mm, grid_step_mm)


def _try_pack_height_groups(
    order: list[int],
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    k: int,
    part_heights: Sequence[float] | None,
) -> list[PackedPlate] | None:
    if part_heights is None:
        return None
    by_height = sorted(order, key=lambda i: float(part_heights[i]), reverse=True)
    groups = [by_height[i::k] for i in range(k)]
    rects_by_plate: list[list[PackedRect]] = []
    overflow: list[int] = []
    for group in groups:
        group_order = sorted(group, key=lambda i: -polygons[i].area)
        plate, unplaced = _pack_plate(group_order, polygons, bed_w, bed_h, gap_mm, grid_step_mm, 0)
        rects_by_plate.append(list(plate.rects))
        overflow.extend(unplaced)

    if overflow:
        for part_idx in overflow:
            best: tuple[tuple[float, float, int], int, PackedRect] | None = None
            for plate_idx, rects in enumerate(rects_by_plate):
                placed_clearance, placed_buffered, forbidden = _build_placement_state(
                    rects, polygons, gap_mm
                )
                result = _try_place_one(
                    polygons[part_idx],
                    part_idx,
                    placed_buffered,
                    placed_clearance,
                    forbidden,
                    bed_w,
                    bed_h,
                    gap_mm,
                    grid_step_mm,
                )
                if result is None:
                    continue
                rect, _placed = result
                trial_rects = [*rects, rect]
                key = (
                    _plate_height_range(trial_rects, part_heights),
                    sum(polygons[r.part_index].area for r in trial_rects),
                    plate_idx,
                )
                if best is None or key < best[0]:
                    best = (key, plate_idx, rect)
            if best is None:
                return None
            _key, plate_idx, rect = best
            rects_by_plate[plate_idx].append(rect)

    plates = _renumber_plates(
        PackedPlate(index=i, rects=tuple(rects)) for i, rects in enumerate(rects_by_plate)
    )
    return _compact_plates(plates, polygons, bed_w, bed_h, gap_mm, grid_step_mm)


def _reflow_plate(
    plate: PackedPlate,
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    *,
    max_sweeps: int,
    deadline: float,
) -> PackedPlate:
    rects = list(plate.rects)
    if len(rects) <= 1:
        return plate

    for _sweep in range(max_sweeps):
        moved = False
        for old_rect in list(rects):
            if time.perf_counter() >= deadline:
                return PackedPlate(index=plate.index, rects=tuple(rects))
            current = placed_polygon_from_rect(polygons[old_rect.part_index], old_rect)
            others = [rect for rect in rects if rect is not old_rect]
            existing = [placed_polygon_from_rect(polygons[r.part_index], r) for r in others]
            old_score = _candidate_spread_score(current, existing)
            placed_clearance, placed_buffered, forbidden = _build_placement_state(
                others, polygons, gap_mm
            )

            def key(
                candidate: BaseGeometry,
                ty: float,
                tx: float,
                existing_geoms: list[BaseGeometry] = existing,
            ) -> tuple[float, float, float]:
                return (-_candidate_spread_score(candidate, existing_geoms), ty, tx)

            result = _try_place_one(
                polygons[old_rect.part_index],
                old_rect.part_index,
                placed_buffered,
                placed_clearance,
                forbidden,
                bed_w,
                bed_h,
                gap_mm,
                grid_step_mm,
                key,
            )
            if result is None:
                continue
            new_rect, new_poly = result
            if _candidate_spread_score(new_poly, existing) <= old_score + 1e-6:
                continue
            rects.remove(old_rect)
            rects.append(new_rect)
            moved = True
        if not moved:
            break
    return PackedPlate(index=plate.index, rects=tuple(rects))


def _reflow_layout(
    plates: list[PackedPlate],
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float,
    *,
    max_sweeps: int = _DEFAULT_REFLOW_SWEEPS,
    time_budget_s: float = _DEFAULT_REFLOW_TIME_BUDGET_S,
) -> list[PackedPlate]:
    deadline = time.perf_counter() + max(0.0, time_budget_s)
    reflowed = [
        _reflow_plate(
            plate,
            polygons,
            bed_w,
            bed_h,
            gap_mm,
            grid_step_mm,
            max_sweeps=max_sweeps,
            deadline=deadline,
        )
        for plate in plates
    ]
    return _renumber_plates(reflowed)


def pack_polygons_on_plates(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    grid_step_mm: float = 2.0,
    max_plates: int = 64,
    on_placed: Callable[[], None] | None = None,
    part_heights: Sequence[float] | None = None,
    strategy: str = "balanced",
    metadata: dict[str, object] | None = None,
    edge_margin_mm: float = 0.0,
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
    if edge_margin_mm < 0:
        raise ValueError(f"edge_margin_mm must be non-negative, got {edge_margin_mm!r}.")
    if part_heights is not None and len(part_heights) != len(polygons):
        raise ValueError(
            f"part_heights length ({len(part_heights)}) must match polygons length ({len(polygons)})."
        )
    if strategy not in {"balanced", "greedy"}:
        raise ValueError(f"Unknown packing strategy {strategy!r}; expected 'balanced' or 'greedy'.")
    if not polygons:
        return []

    effective_bed_w = bed_w - 2.0 * edge_margin_mm
    effective_bed_h = bed_h - 2.0 * edge_margin_mm
    if effective_bed_w <= 0 or effective_bed_h <= 0:
        raise ValueError(
            f"edge_margin_mm={edge_margin_mm!r} leaves no printable bed area "
            f"inside {bed_w:.1f}×{bed_h:.1f} mm."
        )

    order = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)

    for orig_idx in order:
        minx, miny, maxx, maxy = polygons[orig_idx].bounds
        w = maxx - minx
        h = maxy - miny
        if not (
            (w <= effective_bed_w and h <= effective_bed_h)
            or (h <= effective_bed_w and w <= effective_bed_h)
        ):
            raise RuntimeError(
                f"Part {orig_idx} ({w:.1f}×{h:.1f} mm) does not fit on the bed "
                f"({effective_bed_w:.1f}×{effective_bed_h:.1f} mm after edge margin). "
                "Reduce packing.gap_mm / packing.edge_margin_mm or model scale."
            )

    if part_heights is None:
        fast = _try_rectangular_fast_path(
            polygons, effective_bed_w, effective_bed_h, gap_mm, max_plates
        )
        if fast is not None:
            fast = _offset_plates(fast, edge_margin_mm, edge_margin_mm)
            if metadata is not None:
                metadata["strategy"] = "rectangular-fast-path"
                metadata["baseline_plates"] = len(fast)
                metadata["final_plates"] = len(fast)
                metadata["height_score"] = 0.0
                metadata["spread_score"] = _spread_score(fast, polygons)
                metadata["edge_margin_mm"] = edge_margin_mm
                metadata["effective_bed_mm"] = [effective_bed_w, effective_bed_h]
                metadata["attempts"] = [{"kind": "rectpack", "plates": len(fast)}]
            return fast

    baseline = _pack_order(
        order,
        polygons,
        effective_bed_w,
        effective_bed_h,
        gap_mm,
        grid_step_mm,
        max_plates,
        on_placed,
    )
    best = baseline
    best_score = _layout_score(best, polygons, part_heights)
    attempts: list[dict[str, object]] = [
        {"kind": "baseline", "plates": len(baseline), "score": list(best_score)}
    ]

    if strategy == "balanced" and part_heights is not None and len(polygons) > 1:
        if len(polygons) <= 12:
            lower_bound = max(
                1,
                math.ceil(
                    sum(p.area for p in polygons) / max(effective_bed_w * effective_bed_h, 1e-9)
                ),
            )
            orderings = _ordering_candidates(order, polygons, part_heights)
            deadline = time.perf_counter() + min(45.0, max(5.0, len(polygons) * 1.5))
            for k in range(lower_bound, min(len(best), max_plates) + 1):
                for order_idx, candidate_order in enumerate(orderings):
                    if time.perf_counter() >= deadline:
                        break
                    candidate = _try_pack_order_into_k(
                        candidate_order,
                        polygons,
                        effective_bed_w,
                        effective_bed_h,
                        gap_mm,
                        grid_step_mm,
                        k,
                        part_heights,
                    )
                    if candidate is None:
                        attempts.append(
                            {"kind": "fixed-k", "k": k, "order": order_idx, "ok": False}
                        )
                        continue
                    score = _layout_score(candidate, polygons, part_heights)
                    attempts.append(
                        {
                            "kind": "fixed-k",
                            "k": k,
                            "order": order_idx,
                            "ok": True,
                            "plates": len(candidate),
                            "score": list(score),
                        }
                    )
                    if score < best_score:
                        best = candidate
                        best_score = score

                if time.perf_counter() >= deadline:
                    break
                grouped = _try_pack_height_groups(
                    order,
                    polygons,
                    effective_bed_w,
                    effective_bed_h,
                    gap_mm,
                    grid_step_mm,
                    k,
                    part_heights,
                )
                if grouped is not None:
                    score = _layout_score(grouped, polygons, part_heights)
                    attempts.append(
                        {
                            "kind": "height-groups",
                            "k": k,
                            "ok": True,
                            "plates": len(grouped),
                            "score": list(score),
                        }
                    )
                    if score < best_score:
                        best = grouped
                        best_score = score

                if len(best) == k:
                    # Plate count is the first objective; no need to try higher k
                    # once the current k is feasible.
                    break
        else:
            attempts.append({"kind": "fixed-k", "skipped": "too_many_parts"})

        if len(polygons) <= 12:
            before_reflow_score = best_score
            reflowed = _reflow_layout(
                best,
                polygons,
                effective_bed_w,
                effective_bed_h,
                gap_mm,
                grid_step_mm,
                time_budget_s=min(5.0, max(1.0, len(polygons) * 0.25)),
            )
            reflowed_score = _layout_score(reflowed, polygons, part_heights)
            if reflowed_score <= best_score:
                best = reflowed
                best_score = reflowed_score
            attempts.append(
                {
                    "kind": "reflow",
                    "accepted": best_score <= before_reflow_score,
                    "score": list(reflowed_score),
                }
            )
        else:
            attempts.append({"kind": "reflow", "skipped": "too_many_parts"})

    if metadata is not None:
        metadata["strategy"] = strategy
        metadata["baseline_plates"] = len(baseline)
        metadata["final_plates"] = len(best)
        metadata["height_score"] = best_score[1]
        metadata["spread_score"] = -best_score[2]
        metadata["edge_margin_mm"] = edge_margin_mm
        metadata["effective_bed_mm"] = [effective_bed_w, effective_bed_h]
        metadata["attempts"] = attempts

    return _offset_plates(best, edge_margin_mm, edge_margin_mm)


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
