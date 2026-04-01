from stlbench.pipeline.run_fill import _max_copies_on_plate


def test_fill_single_small_rect():
    plate = _max_copies_on_plate(10.0, 10.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) > 1


def test_fill_large_part_one_copy():
    plate = _max_copies_on_plate(90.0, 90.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) == 1


def test_fill_oversized_returns_none():
    plate = _max_copies_on_plate(200.0, 200.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is None


def test_fill_exact_fit():
    plate = _max_copies_on_plate(48.0, 48.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) >= 4


def test_fill_respects_gap():
    plate_small_gap = _max_copies_on_plate(10.0, 10.0, 100.0, 100.0, gap_mm=0.5)
    plate_large_gap = _max_copies_on_plate(10.0, 10.0, 100.0, 100.0, gap_mm=5.0)
    assert plate_small_gap is not None
    assert plate_large_gap is not None
    assert len(plate_small_gap.rects) >= len(plate_large_gap.rects)
