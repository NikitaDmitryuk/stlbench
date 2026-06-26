import numpy as np
import pytest
import trimesh

from stlbench.packing.layout_orientation import (
    HeuristicPrintabilityScorer,
    OrientationCandidate,
    generate_orientation_candidates,
    select_orientation_candidate,
)


def _box(extents: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


def test_generate_candidates_keeps_z_only_default_and_is_deterministic():
    mesh = _box((10.0, 20.0, 30.0))
    a = generate_orientation_candidates(mesh, (100.0, 100.0, 100.0), "sorted", seed=7)
    b = generate_orientation_candidates(mesh, (100.0, 100.0, 100.0), "sorted", seed=7)

    assert len(a) == 360
    assert np.allclose(a[0].transform, np.eye(4))
    assert [c.extents for c in a[:10]] == pytest.approx([c.extents for c in b[:10]])


def test_generate_candidates_full_search_count():
    mesh = _box((10.0, 20.0, 30.0))
    candidates = generate_orientation_candidates(
        mesh,
        (100.0, 100.0, 100.0),
        "sorted",
        any_rotation=True,
        maximize=True,
        random_samples=2,
        seed=7,
    )

    assert len(candidates) == 18


def test_max_scale_policy_selects_largest_scale_limit():
    mesh = _box((20.0, 60.0, 120.0))
    candidates = generate_orientation_candidates(
        mesh, (100.0, 80.0, 70.0), "sorted", any_rotation=True
    )

    selected = select_orientation_candidate(candidates, (100.0, 80.0, 70.0), policy="max-scale")

    assert selected.scale_limit == pytest.approx(max(c.scale_limit for c in candidates))


def test_printable_tolerance_filters_candidates_below_threshold():
    base = dict(
        transform=np.eye(4),
        rotation=np.eye(3),
        extents=(10.0, 10.0, 10.0),
        xy_area=100.0,
        height=10.0,
        pca_aspect=1.0,
        long_axis_z=0.0,
        down_area_ratio=0.0,
        center_z_ratio=0.5,
    )
    bad = OrientationCandidate(scale_limit=0.97, **base)
    good = OrientationCandidate(scale_limit=0.98, **base)
    best = OrientationCandidate(scale_limit=1.00, **{**base, "height": 100.0})

    selected = select_orientation_candidate(
        [bad, good, best],
        (100.0, 100.0, 100.0),
        policy="printable",
        scale_tolerance=0.98,
    )

    assert selected.scale_limit == pytest.approx(0.98)


def test_printable_policy_avoids_vertical_long_parts_within_scale_tolerance():
    mesh = _box((10.0, 10.0, 100.0))
    candidates = generate_orientation_candidates(
        mesh, (200.0, 200.0, 200.0), "sorted", any_rotation=True
    )

    selected = select_orientation_candidate(
        candidates,
        (200.0, 200.0, 200.0),
        policy="printable",
        scale_tolerance=0.98,
    )

    assert selected.pca_aspect >= 3.0
    assert selected.long_axis_z < 0.2


def test_overhang_proxy_penalizes_large_downward_faces():
    mesh = _box((40.0, 40.0, 5.0))
    candidates = generate_orientation_candidates(
        mesh, (100.0, 100.0, 100.0), "sorted", any_rotation=True
    )
    scorer = HeuristicPrintabilityScorer()

    flat = min(candidates, key=lambda c: c.height)
    upright = max(candidates, key=lambda c: c.height)
    flat_score = scorer.score(flat, (100.0, 100.0, 100.0))
    upright_score = scorer.score(upright, (100.0, 100.0, 100.0))

    assert flat.down_area_ratio > upright.down_area_ratio
    assert flat_score.down_area > upright_score.down_area
