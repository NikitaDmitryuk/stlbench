import numpy as np
import trimesh.creation

from stlbench.bbox_fit import aabb_edge_lengths
from stlbench.orientation import (
    best_aabb_extents_for_sorted_fit,
    score_sorted_fit,
)


def test_free_rotation_improves_diagonal_cylinder():
    cyl = trimesh.creation.cylinder(radius=0.15, height=12.0, sections=32)
    t = trimesh.transformations.rotation_matrix(np.deg2rad(35), [1, 1, 0])
    cyl.apply_transform(t)
    v = np.asarray(cyl.vertices, dtype=np.float64)
    fb = tuple(aabb_edge_lengths(np.asarray(cyl.bounds)))
    p_sorted = (20.0, 30.0, 40.0)
    rng = np.random.default_rng(0)
    best = best_aabb_extents_for_sorted_fit(
        v, p_sorted, n_samples=6000, rng=rng, identity_baseline=fb
    )
    assert score_sorted_fit(best, p_sorted) >= score_sorted_fit(fb, p_sorted) - 1e-9
    assert score_sorted_fit(best, p_sorted) > score_sorted_fit(fb, p_sorted) + 1e-6
