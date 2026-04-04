import numpy as np
import trimesh

from stlbench.packing.layout_orientation import select_layout_transform


def test_orientation_fits_only_with_axis_swap():
    """
    Box 30x100x10 mm (x,y,z): default pose height 10, footprint 30x100; 100 > bed depth 78.
    Y-up: height 100 exceeds pz. X-up: height 30, footprint 10x100 along X/Y fits 153x78.
    """
    mesh = trimesh.creation.box(extents=[30.0, 100.0, 10.0])
    ok, t, fw, fh = select_layout_transform(mesh, bed_x=153.0, bed_y=78.0, pz=50.0, gap_mm=2.0)
    assert ok
    assert t.shape == (4, 4)
    m2 = mesh.copy()
    m2.apply_transform(t)
    b = m2.bounds
    d = b[1] - b[0]
    assert float(d[2]) <= 50.0 + 1e-3
    assert abs(float(d[0]) - fw) < 1e-3
    assert abs(float(d[1]) - fh) < 1e-3


def test_footprint_swap_on_narrow_bed():
    """Long edge ends up along bed X (axis swap inside footprint_fits_bin_mm)."""
    mesh = trimesh.creation.box(extents=[20.0, 90.0, 15.0])
    ok, _, fw, fh = select_layout_transform(mesh, bed_x=153.0, bed_y=78.0, pz=80.0, gap_mm=1.0)
    assert ok
    assert max(fw, fh) <= 153.0 + 1e-3
    assert min(fw, fh) <= 78.0 + 1e-3


def test_impossible_height():
    mesh = trimesh.creation.box(extents=[50.0, 50.0, 200.0])
    ok, t, _, _ = select_layout_transform(mesh, bed_x=153.0, bed_y=78.0, pz=100.0, gap_mm=1.0)
    assert not ok
    assert np.allclose(t, np.eye(4))
