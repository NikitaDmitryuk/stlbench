"""Tests for polygon-based 2-D bin packing."""

from __future__ import annotations

import pytest
import trimesh
from shapely.geometry import Polygon, box

from stlbench.config.enums import ExactPackQuality
from stlbench.packing.polygon_footprint import mesh_to_packing_shadow, mesh_to_xy_shadow
from stlbench.packing.polygon_pack import (
    _compact_plates,
    _grid_fallback_place,
    _height_score,
    _poly_to_int,
    _spread_score,
    _within_bed,
    footprints_to_box_polygons,
    pack_polygons_on_plates,
    placed_polygon_from_rect,
    try_pack_polygons_single_plate,
)
from stlbench.packing.rectpack_plate import PackedPlate, PackedRect

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


def _placed_polygons(plate, polys):
    return [placed_polygon_from_rect(polys[r.part_index], r) for r in plate.rects]


def _min_distance_on_plate(plate, polys) -> float:
    placed = _placed_polygons(plate, polys)
    out = float("inf")
    for i, p0 in enumerate(placed):
        for p1 in placed[i + 1 :]:
            out = min(out, p0.distance(p1))
    return out


def _corner_plate_with_center_space() -> tuple[list[Polygon], PackedPlate]:
    polys = [
        _box_poly(20.0, 20.0),
        _box_poly(20.0, 20.0),
        _box_poly(20.0, 20.0),
        _box_poly(20.0, 20.0),
        _box_poly(8.0, 8.0),
    ]
    plate = PackedPlate(
        index=0,
        rects=(
            PackedRect(part_index=0, x=0.0, y=0.0, width=20.0, height=20.0),
            PackedRect(part_index=1, x=80.0, y=0.0, width=20.0, height=20.0),
            PackedRect(part_index=2, x=0.0, y=40.0, width=20.0, height=20.0),
            PackedRect(part_index=3, x=80.0, y=40.0, width=20.0, height=20.0),
        ),
    )
    return polys, plate


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


def test_packing_shadow_is_conservative_and_compact():
    mesh = _flat_box_mesh(20.0, 30.0)
    exact = mesh_to_xy_shadow(mesh, simplify_tol=0.0)
    packing = mesh_to_packing_shadow(mesh, simplify_tol=0.25)

    assert packing.area == pytest.approx(exact.area, abs=40.0)
    assert packing.bounds == pytest.approx(exact.bounds, abs=0.5)
    assert len(packing.exterior.coords) <= len(exact.exterior.coords) + 16


def test_packing_shadow_bounds_do_not_shrink_exact_mesh_bounds():
    mesh = _flat_box_mesh(20.0, 30.0)
    exact = mesh_to_xy_shadow(mesh, simplify_tol=0.0)
    packing = mesh_to_packing_shadow(mesh, simplify_tol=0.25)

    eminx, eminy, emaxx, emaxy = exact.bounds
    pminx, pminy, pmaxx, pmaxy = packing.bounds
    assert pminx <= eminx + 1e-9
    assert pminy <= eminy + 1e-9
    assert pmaxx >= emaxx - 1e-9
    assert pmaxy >= emaxy - 1e-9


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


def test_balanced_strategy_groups_similar_heights_without_more_plates():
    polys = [_box_poly(40.0, 20.0) for _ in range(4)]
    heights = [100.0, 10.0, 95.0, 8.0]

    greedy = pack_polygons_on_plates(
        polys,
        bed_w=90.0,
        bed_h=30.0,
        gap_mm=5.0,
        part_heights=heights,
        strategy="greedy",
    )
    balanced = pack_polygons_on_plates(
        polys,
        bed_w=90.0,
        bed_h=30.0,
        gap_mm=5.0,
        part_heights=heights,
    )

    assert len(balanced) == len(greedy) == 2
    assert _height_score(balanced, heights) < _height_score(greedy, heights)


def test_part_heights_none_keeps_old_api_shape():
    polys = [_box_poly(10.0, 10.0), _box_poly(20.0, 5.0)]

    plates = pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)

    assert len(plates) == 1
    assert sorted(r.part_index for r in plates[0].rects) == [0, 1]


def test_oversized_part_raises():
    polys = [_box_poly(200.0, 10.0)]
    with pytest.raises(RuntimeError, match="does not fit"):
        pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=1.0)


def test_empty_input_returns_empty():
    plates = pack_polygons_on_plates([], bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plates == []


def test_edge_margin_offsets_final_rects_inside_full_bed():
    polys = [_box_poly(20.0, 20.0), _box_poly(15.0, 10.0)]

    plates = pack_polygons_on_plates(
        polys,
        bed_w=60.0,
        bed_h=50.0,
        gap_mm=2.0,
        edge_margin_mm=2.0,
    )

    assert len(plates) == 1
    for placed in _placed_polygons(plates[0], polys):
        minx, miny, maxx, maxy = placed.bounds
        assert minx >= 2.0 - 1e-6
        assert miny >= 2.0 - 1e-6
        assert maxx <= 58.0 + 1e-6
        assert maxy <= 48.0 + 1e-6


def test_edge_margin_reduces_printable_area_for_fit_check():
    polys = [_box_poly(97.0, 20.0)]

    with pytest.raises(RuntimeError, match="after edge margin"):
        pack_polygons_on_plates(
            polys,
            bed_w=100.0,
            bed_h=50.0,
            gap_mm=1.0,
            edge_margin_mm=2.0,
        )


# ---------------------------------------------------------------------------
# Gap enforcement
# ---------------------------------------------------------------------------


def test_gap_is_enforced_between_two_rects():
    """Two parts placed by the packer must be at least gap_mm apart."""
    gap = 3.0
    polys = [_box_poly(40.0, 40.0), _box_poly(40.0, 40.0)]
    plates = pack_polygons_on_plates(polys, bed_w=100.0, bed_h=100.0, gap_mm=gap)
    assert len(plates) == 1
    assert _min_distance_on_plate(plates[0], polys) >= gap - 1e-3


def test_strict_gap_is_enforced_for_rotated_complex_polygons():
    gap = 5.0
    polys = [
        _l_shape_poly(outer=55.0, inner=30.0),
        Polygon([(0, 0), (42, 4), (38, 30), (6, 34)]),
        _box_poly(22.0, 18.0),
        Polygon([(0, 0), (30, 0), (22, 16), (4, 20)]),
    ]

    plates = pack_polygons_on_plates(polys, bed_w=120.0, bed_h=90.0, gap_mm=gap)

    for plate in plates:
        if len(plate.rects) > 1:
            assert _min_distance_on_plate(plate, polys) >= gap - 1e-3


def test_grid_fallback_finds_interior_free_space():
    polys, plate = _corner_plate_with_center_space()
    placed = _placed_polygons(plate, polys)

    result = _grid_fallback_place(
        polys[4],
        4,
        placed,
        bed_w=100.0,
        bed_h=60.0,
        gap_mm=10.0,
        grid_step_mm=2.0,
    )

    assert result is not None
    rect, placed_poly = result
    assert rect.part_index == 4
    assert all(placed_poly.distance(existing) >= 10.0 - 1e-3 for existing in placed)


def test_sparse_tail_plate_is_compacted_when_gap_allows():
    polys, first_plate = _corner_plate_with_center_space()
    second_plate = PackedPlate(
        index=1,
        rects=(PackedRect(part_index=4, x=0.0, y=0.0, width=8.0, height=8.0),),
    )

    compacted = _compact_plates(
        [first_plate, second_plate],
        polys,
        bed_w=100.0,
        bed_h=60.0,
        gap_mm=10.0,
        grid_step_mm=2.0,
    )

    assert len(compacted) == 1
    assert len(compacted[0].rects) == 5
    assert _min_distance_on_plate(compacted[0], polys) >= 10.0 - 1e-3


def test_reflow_preserves_gap_and_does_not_reduce_spread():
    polys = [_box_poly(10.0, 10.0) for _ in range(4)]

    greedy = pack_polygons_on_plates(
        polys,
        bed_w=80.0,
        bed_h=80.0,
        gap_mm=5.0,
        strategy="greedy",
    )
    balanced = pack_polygons_on_plates(
        polys,
        bed_w=80.0,
        bed_h=80.0,
        gap_mm=5.0,
        strategy="balanced",
    )

    assert len(balanced) == len(greedy)
    assert _spread_score(balanced, polys) >= _spread_score(greedy, polys)
    for plate in balanced:
        assert _min_distance_on_plate(plate, polys) >= 5.0 - 1e-3
        for placed in _placed_polygons(plate, polys):
            assert _within_bed(placed, bed_w=80.0, bed_h=80.0)


def test_placed_polygon_from_rect_matches_rect_transform():
    poly = _box_poly(10.0, 20.0)
    plates = pack_polygons_on_plates([poly], bed_w=100.0, bed_h=100.0, gap_mm=5.0)
    rect = plates[0].rects[0]

    placed = placed_polygon_from_rect(poly, rect)

    minx, miny, maxx, maxy = placed.bounds
    assert minx == pytest.approx(rect.x)
    assert miny == pytest.approx(rect.y)
    assert maxx - minx == pytest.approx(rect.width)
    assert maxy - miny == pytest.approx(rect.height)


def test_candidate_must_stay_inside_bed_bounds():
    assert _within_bed(_box_poly(10.0, 10.0), bed_w=10.0, bed_h=10.0)
    assert not _within_bed(box(0.0, 0.0, 10.002, 10.0), bed_w=10.0, bed_h=10.0)


def test_poly_to_int_rounds_instead_of_truncating():
    poly = Polygon([(0.0006, 0.0), (1.0004, 0.0), (0.0, 1.0006)])

    assert _poly_to_int(poly) == [(1, 0), (1000, 0), (0, 1001)]


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


def test_exact_feasibility_quality_skips_compaction():
    polys = [_l_shape_poly(), _box_poly(25.0, 25.0), _box_poly(20.0, 30.0)]
    metadata: dict[str, object] = {}

    plates = pack_polygons_on_plates(
        polys,
        bed_w=140.0,
        bed_h=100.0,
        gap_mm=2.0,
        part_heights=[10.0, 8.0, 6.0],
        metadata=metadata,
        quality=ExactPackQuality.FEASIBILITY,
    )

    assert plates
    assert metadata["quality"] == ExactPackQuality.FEASIBILITY.value
    assert metadata["exact_compact_s"] == 0.0


def test_exact_final_quality_records_compaction_timing():
    polys = [_l_shape_poly(), _box_poly(25.0, 25.0), _box_poly(20.0, 30.0)]
    metadata: dict[str, object] = {}

    plates = pack_polygons_on_plates(
        polys,
        bed_w=140.0,
        bed_h=100.0,
        gap_mm=2.0,
        part_heights=[10.0, 8.0, 6.0],
        metadata=metadata,
        quality=ExactPackQuality.FINAL,
    )

    assert plates
    assert metadata["quality"] == ExactPackQuality.FINAL.value
    assert metadata["exact_compact_s"] >= 0.0
