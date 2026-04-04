import numpy as np
import trimesh.creation

from stlbench.core.fit import aabb_edge_lengths, compute_global_scale


def test_box_scale_matches_formula():
    box = trimesh.creation.box(extents=(4.0, 6.0, 10.0))
    dims = aabb_edge_lengths(np.asarray(box.bounds))
    s, _ = compute_global_scale(
        (20.0, 30.0, 40.0),
        [dims],
        ["box"],
        "sorted",
    )
    scaled = box.copy()
    scaled.apply_scale(s)
    new_dims = aabb_edge_lengths(np.asarray(scaled.bounds))
    px, py, pz = (20.0, 30.0, 40.0)
    assert new_dims[0] <= px + 1e-5
    assert new_dims[1] <= py + 1e-5
    assert new_dims[2] <= pz + 1e-5
    p_sorted = sorted((px, py, pz))
    d_sorted = sorted(new_dims)
    assert d_sorted[0] <= p_sorted[0] + 1e-5
    assert d_sorted[1] <= p_sorted[1] + 1e-5
    assert d_sorted[2] <= p_sorted[2] + 1e-5
