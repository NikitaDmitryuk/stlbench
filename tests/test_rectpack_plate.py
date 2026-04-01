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
    """Непомещающиеся детали не должны ломать rectpack (len==0); даём понятную ошибку."""
    with pytest.raises(RuntimeError, match="ни одна из оставшихся"):
        pack_rectangles_on_plates(
            [(300.0, 10.0), (5.0, 5.0)],
            bed_w=100.0,
            bed_h=100.0,
            gap_mm=1.0,
            max_plates=8,
        )


def test_pack_rect_needs_rotation():
    """Деталь 10×90 на столе 100×50: влезает только повёрнутой (90×10 → 10 по Y ≤ 50)."""
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
