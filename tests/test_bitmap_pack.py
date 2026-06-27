from shapely.geometry import box

from stlbench.packing.bitmap_pack import (
    BitmapPackOptions,
    pack_polygons_bitmap_single_plate,
)


def test_bitmap_pack_places_rectangles_with_gap():
    polygons = [box(0.0, 0.0, 10.0, 10.0), box(0.0, 0.0, 10.0, 10.0)]

    result = pack_polygons_bitmap_single_plate(
        polygons,
        bed_w=25.0,
        bed_h=12.0,
        gap_mm=2.0,
        scale=1.0,
        options=BitmapPackOptions(grid_mm=0.5),
    )

    assert result.plate is not None
    assert len(result.plate.rects) == 2
    a, b = result.plate.rects
    assert abs(a.x - b.x) >= 12.0 or abs(a.y - b.y) >= 12.0


def test_bitmap_pack_fails_when_gap_cannot_fit():
    polygons = [box(0.0, 0.0, 10.0, 10.0), box(0.0, 0.0, 10.0, 10.0)]

    result = pack_polygons_bitmap_single_plate(
        polygons,
        bed_w=21.0,
        bed_h=10.0,
        gap_mm=2.0,
        scale=1.0,
        options=BitmapPackOptions(grid_mm=0.5),
    )

    assert result.plate is None


def test_bitmap_pack_handles_l_shapes():
    l_shape = box(0.0, 0.0, 10.0, 4.0).union(box(0.0, 0.0, 4.0, 10.0))

    result = pack_polygons_bitmap_single_plate(
        [l_shape, l_shape],
        bed_w=25.0,
        bed_h=15.0,
        gap_mm=1.0,
        scale=1.0,
        options=BitmapPackOptions(grid_mm=0.5),
    )

    assert result.plate is not None
    assert len(result.plate.rects) == 2
    assert result.stats.candidates_tested > 0
