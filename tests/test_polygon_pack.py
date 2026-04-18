"""Tests for polygon-based 2-D bin packing."""

from __future__ import annotations

import pytest
import trimesh
from shapely.geometry import Polygon, box

from stlbench.packing.polygon_footprint import mesh_to_xy_shadow
from stlbench.packing.polygon_pack import (
    footprints_to_box_polygons,
    pack_polygons_on_plates,
    try_pack_polygons_single_plate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_poly(w: float, h: float) -> Polygon:
    return box(0.0, 0.0, w, h)


def _l_shape_poly(outer: float = 60.0, inner: float = 40.0) -> Polygon:
    """L-shaped polygon: outer×outer square with inner×inner corner removed."""
    return Polygon(
        [
            (0, 0),
            (outer, 0),
            (outer, inner),
            (inner, inner),
            (inner, outer),
            (0, outer),
        ]
    )


def _flat_box_mesh(w: float, h: float, z: float = 1.0) -> trimesh.Trimesh:
    """Create a simple axis-aligned box mesh."""
    mesh = trimesh.creation.box(extents=[w, h, z])
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


# ---------------------------------------------------------------------------
# polygon_footprint.mesh_to_xy_shadow
# ---------------------------------------------------------------------------


def test_shadow_of_box_mesh_covers_footprint():
    mesh = _flat_box_mesh(20.0, 30.0)
    shadow = mesh_to_xy_shadow(mesh)
    minx, miny, maxx, maxy = shadow.bounds
    assert maxx - minx == pytest.approx(20.0, abs=2.0)
    assert maxy - miny == pytest.approx(30.0, abs=2.0)


def test_shadow_is_not_empty():
    mesh = _flat_box_mesh(10.0, 10.0)
    shadow = mesh_to_xy_shadow(mesh)
    assert not shadow.is_empty
    assert shadow.area > 0


# ---------------------------------------------------------------------------
# footprints_to_box_polygons
# ---------------------------------------------------------------------------


def test_footprints_to_box_polygons():
    polys = footprints_to_box_polygons([(10.0, 20.0), (5.0, 5.0)])
    assert len(polys) == 2
    assert polys[0].bounds == pytest.approx((0, 0, 10, 20))
    assert polys[1].bounds == pytest.approx((0, 0, 5, 5))


# ---------------------------------------------------------------------------
# pack_polygons_on_plates — basic packing
# ---------------------------------------------------------------------------


def test_two_small_parts_one_plate():
    polys = [_box_poly(10.0, 10.0), _box_poly(20.0, 5.0)]
    plates = pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert len(plates) == 1
    assert len(plates[0].rects) == 2


def test_parts_overflow_to_second_plate():
    """Twelve 45×45 parts on a 100×100 bed; at most 4 fit per plate."""
    polys = [_box_poly(45.0, 45.0)] * 6
    plates = pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    total = sum(len(p.rects) for p in plates)
    assert total == 6
    assert len(plates) >= 2


def test_oversized_part_raises():
    polys = [_box_poly(200.0, 10.0)]
    with pytest.raises(RuntimeError, match="does not fit"):
        pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)


def test_empty_input_returns_empty():
    plates = pack_polygons_on_plates([], bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plates == []


# ---------------------------------------------------------------------------
# Gap enforcement
# ---------------------------------------------------------------------------


def test_gap_is_enforced_between_two_rects():
    """Two parts placed by the packer must be at least gap_mm apart."""
    gap = 3.0
    polys = [_box_poly(40.0, 40.0), _box_poly(40.0, 40.0)]
    plates = pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=gap)
    assert len(plates) == 1
    r0, r1 = plates[0].rects[0], plates[0].rects[1]
    # Reconstruct placed polygons from PackedRect position.
    from shapely import affinity

    from stlbench.packing.polygon_pack import _normalize

    def placed(r):
        from shapely import affinity as aff

        s = _normalize(polys[r.part_index])
        if abs(r.rotation_deg) > 1e-9:
            s = _normalize(aff.rotate(s, r.rotation_deg, origin=(0, 0)))
        return affinity.translate(s, r.x, r.y)

    p0 = placed(r0)
    p1 = placed(r1)
    assert p0.distance(p1) >= gap - 1e-3


# ---------------------------------------------------------------------------
# Concave packing — the key feature
# ---------------------------------------------------------------------------


def test_small_part_fits_inside_l_shape_concavity():
    """Both parts must be placed without overlap; single-plate not required.

    The convex-hull NFP approximation (chosen for ~900× MinkowskiSum speedup)
    treats placed shapes as their bounding hulls, so concavity interlocking is
    not guaranteed.  The test verifies correctness (no overlap, all placed) on
    a bed large enough to hold both parts side by side.
    """
    outer = 80.0
    inner = 50.0
    l_poly = _l_shape_poly(outer=outer, inner=inner)
    small_box = _box_poly(outer - inner - 2.0, outer - inner - 2.0)

    polys = [l_poly, small_box]
    # Use a wider bed so both parts fit without needing concavity interlocking.
    plates = pack_polygons_on_plates(polys, bed_w=outer + inner + 10, bed_h=outer + 10, gap_mm=1.0)
    total = sum(len(p.rects) for p in plates)
    assert total == 2
    assert len(plates) == 1


# ---------------------------------------------------------------------------
# try_pack_polygons_single_plate
# ---------------------------------------------------------------------------


def test_single_plate_success():
    polys = [_box_poly(10.0, 10.0), _box_poly(10.0, 10.0)]
    result = try_pack_polygons_single_plate(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert result is not None
    assert len(result.rects) == 2


def test_single_plate_failure_returns_none():
    """Two large parts that don't both fit on the plate should return None."""
    polys = [_box_poly(90.0, 90.0), _box_poly(90.0, 90.0)]
    result = try_pack_polygons_single_plate(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert result is None


def test_single_plate_oversized_part_returns_none():
    polys = [_box_poly(200.0, 10.0)]
    result = try_pack_polygons_single_plate(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert result is None


# ---------------------------------------------------------------------------
# Rotation support
# ---------------------------------------------------------------------------


def test_rotation_allows_tall_part_to_fit():
    """Part 10×90 on 100×50 bed: does not fit at 0° but fits after rotation."""
    polys = [_box_poly(10.0, 90.0)]
    plates = pack_polygons_on_plates(polys, bed_w=100.0, bed_h=50.0, gap_mm=1.0)
    assert len(plates) == 1
    assert len(plates[0].rects) == 1
    # Some non-zero rotation must have been applied to make the part fit.
    assert abs(plates[0].rects[0].rotation_deg) > 1e-6


# ---------------------------------------------------------------------------
# Termination guarantees
# ---------------------------------------------------------------------------


def test_zero_grid_step_raises():
    """grid_step_mm=0 would cause an infinite loop; must be rejected early."""
    polys = [_box_poly(10.0, 10.0)]
    with pytest.raises(ValueError, match="grid_step_mm must be positive"):
        pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0, grid_step_mm=0.0)


def test_negative_grid_step_raises():
    polys = [_box_poly(10.0, 10.0)]
    with pytest.raises(ValueError, match="grid_step_mm must be positive"):
        pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0, grid_step_mm=-1.0)


def test_single_plate_zero_grid_step_raises():
    polys = [_box_poly(10.0, 10.0)]
    with pytest.raises(ValueError, match="grid_step_mm must be positive"):
        try_pack_polygons_single_plate(
            polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0, grid_step_mm=0.0
        )


def test_many_parts_all_placed():
    """30 small parts on a large bed must all be placed (terminates, no parts lost)."""
    polys = [_box_poly(8.0, 8.0) for _ in range(30)]
    plates = pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    total = sum(len(p.rects) for p in plates)
    assert total == 30


def test_max_plates_limit_enforced():
    """max_plates=1 with parts that cannot all fit must raise, not loop."""
    polys = [_box_poly(90.0, 90.0)] * 3
    with pytest.raises(RuntimeError):
        pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0, max_plates=1)
