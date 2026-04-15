import pytest

from stlbench.core.fit import (
    aabb_edge_lengths,
    compute_global_scale,
    s_max_for_part_conservative,
    s_max_for_part_printer_axes,
    s_max_for_part_sorted,
)


def test_aabb_edge_lengths():
    import numpy as np

    b = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 5.0]])
    assert aabb_edge_lengths(b) == (2.0, 3.0, 3.0)


def test_s_max_sorted_single():
    p_sorted = (10.0, 20.0, 30.0)
    d_sorted = (2.0, 3.0, 4.0)
    assert s_max_for_part_sorted(p_sorted, d_sorted) == pytest.approx(min(10 / 2, 20 / 3, 30 / 4))


def test_compute_global_scale_sorted_two_parts():
    # Part A: 10x1x1 -> sorted 1,1,10 -> s_A = min(10/1,10/1,10/10)=1
    # Part B: 5x5x5 -> s_B = 2
    s, reports = compute_global_scale(
        (10.0, 10.0, 10.0),
        [(10.0, 1.0, 1.0), (5.0, 5.0, 5.0)],
        ["a", "b"],
        "sorted",
    )
    assert s == pytest.approx(1.0)
    assert len(reports) == 2


def test_compute_global_scale_conservative():
    s, _ = compute_global_scale(
        (12.0, 15.0, 20.0),
        [(2.0, 3.0, 4.0), (6.0, 1.0, 1.0)],
        ["p1", "p2"],
        "conservative",
    )
    # p_min=12, limits: 12/4=3, 12/6=2 -> min 2
    assert s == pytest.approx(2.0)


def test_s_max_for_part_conservative():
    assert s_max_for_part_conservative(12.0, 2.0, 3.0, 4.0) == pytest.approx(3.0)


def test_s_max_for_part_printer_axes_z_limits():
    # Build height ez exceeds pz until scaled; XY is small on the bed.
    s, hint = s_max_for_part_printer_axes(100.0, 100.0, 50.0, 2.0, 2.0, 100.0)
    assert s == pytest.approx(0.5)
    assert hint == "z_build_height"


def test_s_max_for_part_printer_axes_xy_swap():
    # Long edge along Y in printer frame; 90° bed swap uses X for the long edge.
    s, hint = s_max_for_part_printer_axes(100.0, 100.0, 50.0, 2.0, 100.0, 2.0)
    assert s == pytest.approx(1.0)
    assert hint == "xy_bed_swapped"


def test_compute_global_scale_sorted_uses_printer_axes():
    s, _ = compute_global_scale(
        (100.0, 100.0, 50.0),
        [(2.0, 2.0, 100.0)],
        ["tall_z"],
        "sorted",
    )
    assert s == pytest.approx(0.5)


def test_compute_global_scale_empty():
    with pytest.raises(ValueError):
        compute_global_scale((1, 1, 1), [], [], "sorted")
