import pytest

from stlbench.packing.rectpack_plate import pack_rectangles_on_plates


def test_pack_two_rects_one_plate():
    plates = pack_rectangles_on_plates(
        [(10.0, 10.0), (20.0, 5.0)],
        bed_w=100.0,
        bed_h=100.0,
        gap_mm=1.0,
        max_plates=8,
    )
    assert len(plates) >= 1
    total = sum(len(p.rects) for p in plates)
    assert total == 2


def test_pack_mixed_oversized_raises_clear_error():
    """Oversized parts should not leave rectpack with an empty bin; expect a clear error."""
    with pytest.raises(RuntimeError, match="none of the remaining"):
        pack_rectangles_on_plates(
            [(300.0, 10.0), (5.0, 5.0)],
            bed_w=100.0,
            bed_h=100.0,
            gap_mm=1.0,
            max_plates=8,
        )


def test_pack_rect_needs_rotation():
    """Part 10x90 on 100x50 bed: fits only when rotated to 90x10 (10 along Y <= 50)."""
    plates = pack_rectangles_on_plates(
        [(10.0, 90.0)],
        bed_w=100.0,
        bed_h=50.0,
        gap_mm=1.0,
        max_plates=8,
    )
    assert len(plates) >= 1
    total = sum(len(p.rects) for p in plates)
    assert total == 1


def test_large_gap_does_not_prevent_single_part():
    """Regression: gap_mm should only add space between parts, not at the leading edge.

    A part that fits the bed (fw <= bed_w) must not be rejected even when gap_mm
    is large (e.g. 5 mm).  Before the fix, the packing bin was constructed without
    the trailing-gap budget, so parts with fw > bed_w - gap_mm failed silently.
    """
    gap = 5.0
    # fw = bed_w - 1 < bed_w, so the part physically fits the bed
    plates = pack_rectangles_on_plates(
        [(195.0, 195.0)],
        bed_w=200.0,
        bed_h=200.0,
        gap_mm=gap,
    )
    assert len(plates) == 1
    assert len(plates[0].rects) == 1


def test_large_gap_two_parts_fit():
    """Two parts on separate plates when they don't both fit together with gap."""
    plates = pack_rectangles_on_plates(
        [(98.0, 98.0), (98.0, 98.0)],
        bed_w=200.0,
        bed_h=200.0,
        gap_mm=5.0,
    )
    total = sum(len(p.rects) for p in plates)
    assert total == 2
