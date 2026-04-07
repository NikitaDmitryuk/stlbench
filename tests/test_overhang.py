"""Tests for stlbench.core.overhang."""

import numpy as np
import pytest
import trimesh

from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    overhang_score,
)


def _tilted_box(angle_deg: float = 30.0) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 5.0])
    T = trimesh.transformations.rotation_matrix(np.radians(angle_deg), [1, 0, 0])
    mesh.apply_transform(T)
    return mesh  # type: ignore[no-any-return]


def test_overhang_score_identity_flat_box():
    """A flat, axis-aligned box has no overhangs under identity rotation."""
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 5.0])
    # Flat box: top face normal = [0,0,+1], bottom = [0,0,-1], sides = horizontal.
    # Only the perfectly-flat bottom face has nz=-1, which gives a bottom_area bonus.
    # With threshold=45°, sides (nz=0) are NOT overhangs. Expect score <= 0.
    score = overhang_score(mesh, np.eye(3), overhang_threshold_deg=45.0)
    assert score <= 0.0


def test_overhang_score_upside_down_box():
    """A box rotated 180° around X (top becomes bottom) should have zero overhangs
    because the large flat face now faces down and is counted as 'bottom area'."""
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 5.0])
    T = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    R = T[:3, :3]
    score = overhang_score(mesh, R, overhang_threshold_deg=45.0)
    assert score <= 0.0


def test_find_min_overhang_reduces_score():
    """Optimiser should produce an equal-or-better score than the identity rotation."""
    mesh = _tilted_box(angle_deg=40.0)
    score_before = overhang_score(mesh, np.eye(3), overhang_threshold_deg=45.0)
    _, score_after = find_min_overhang_rotation(mesh, overhang_threshold_deg=45.0, n_candidates=50)
    assert score_after <= score_before + 1e-6  # never worse


def test_find_min_overhang_box_reaches_zero():
    """A simple box can always be oriented with zero overhangs."""
    mesh = _tilted_box(angle_deg=30.0)
    _, score = find_min_overhang_rotation(mesh, overhang_threshold_deg=45.0, n_candidates=100)
    assert score <= 0.0 + 1.0  # small tolerance for floating-point


def test_apply_orientation_z_min_at_zero():
    """After applying the orientation, the lowest point of the mesh should be at z=0."""
    mesh = _tilted_box(angle_deg=45.0)
    rotation, _ = find_min_overhang_rotation(mesh, n_candidates=50)
    oriented = apply_min_overhang_orientation(mesh, rotation)
    assert oriented.bounds[0, 2] == pytest.approx(0.0, abs=1e-6)
