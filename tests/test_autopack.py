from stlbench.pipeline.run_autopack import _bisect_scale, _try_pack_all


def test_try_pack_all_fits():
    footprints = [(10.0, 10.0), (20.0, 10.0)]
    plate = _try_pack_all(footprints, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) == 2


def test_try_pack_all_oversized():
    footprints = [(200.0, 10.0), (10.0, 10.0)]
    plate = _try_pack_all(footprints, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plate is None


def test_try_pack_all_tight():
    footprints = [(95.0, 95.0), (95.0, 95.0)]
    plate = _try_pack_all(footprints, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plate is None


def test_bisect_scale_finds_positive():
    base_footprints = [(10.0, 10.0), (20.0, 10.0)]
    s, plate = _bisect_scale(base_footprints, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=10.0)
    assert s > 0
    assert plate is not None
    assert len(plate.rects) == 2


def test_bisect_scale_respects_upper():
    base_footprints = [(5.0, 5.0)]
    s, plate = _bisect_scale(base_footprints, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=3.0)
    assert s > 0
    assert s <= 3.0 + 1e-4
    assert plate is not None


def test_bisect_scale_impossible():
    base_footprints = [(200.0, 200.0)]
    s, plate = _bisect_scale(base_footprints, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=0.4)
    assert s > 0
    assert plate is not None


def test_bisect_scale_multiple_parts():
    base_footprints = [(30.0, 30.0), (30.0, 30.0), (30.0, 30.0)]
    s, plate = _bisect_scale(base_footprints, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=5.0)
    assert s > 0
    assert plate is not None
    assert len(plate.rects) == 3
