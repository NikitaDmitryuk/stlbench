"""Tests for stlbench.core.overhang."""

import numpy as np
import pytest
import trimesh

from stlbench.core.overhang import (
    ResinOrientationOptions,
    _face_saliency,
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    find_stable_overhang_rotation,
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


def test_stable_overhang_keeps_long_rod_in_resin_target_band():
    mesh = trimesh.creation.box(extents=[120.0, 6.0, 6.0])

    rotation, _, metrics = find_stable_overhang_rotation(
        mesh,
        n_candidates=80,
        printer_dims=(200.0, 200.0, 200.0),
        support_tolerance_ratio=0.5,
    )
    oriented = apply_min_overhang_orientation(mesh, rotation)

    assert 30.0 <= metrics.long_axis_angle_from_bed_deg <= 50.0
    assert oriented.extents[2] > 6.0
    assert metrics.selection_reason == "long_part_target_band"


def test_stable_overhang_rejects_horizontal_long_rod_when_target_exists():
    mesh = trimesh.creation.box(extents=[120.0, 6.0, 6.0])

    _rotation, _, metrics = find_stable_overhang_rotation(
        mesh,
        n_candidates=80,
        printer_dims=(200.0, 200.0, 200.0),
        support_tolerance_ratio=0.5,
        resin_options=ResinOrientationOptions(resin_balance="balanced"),
    )

    assert metrics.long_axis_angle_from_bed_deg >= 20.0
    assert metrics.horizontal_penalty == pytest.approx(0.0)


def test_stable_overhang_rejects_vertical_long_rod_when_target_exists():
    mesh = trimesh.creation.box(extents=[6.0, 6.0, 120.0])

    _rotation, _, metrics = find_stable_overhang_rotation(
        mesh,
        n_candidates=80,
        printer_dims=(200.0, 200.0, 200.0),
        support_tolerance_ratio=0.5,
        resin_options=ResinOrientationOptions(resin_balance="balanced"),
    )

    assert metrics.long_axis_angle_from_bed_deg <= 60.0
    assert metrics.vertical_penalty == pytest.approx(0.0)


def test_stable_overhang_penalizes_vertical_flat_plate():
    mesh = trimesh.creation.box(extents=[70.0, 4.0, 40.0])

    rotation, _, metrics = find_stable_overhang_rotation(
        mesh,
        n_candidates=80,
        printer_dims=(200.0, 200.0, 200.0),
        support_tolerance_ratio=0.5,
    )
    oriented = apply_min_overhang_orientation(mesh, rotation)

    assert metrics.long_axis_angle_from_bed_deg < 55.0
    assert oriented.extents[2] < 90.0


def test_source_up_preservation_rejects_upside_down_candidate():
    mesh = trimesh.creation.box(extents=[20.0, 20.0, 40.0])

    _rotation, _, metrics = find_stable_overhang_rotation(
        mesh,
        n_candidates=80,
        printer_dims=(100.0, 100.0, 100.0),
        support_tolerance_ratio=0.5,
        resin_options=ResinOrientationOptions(resin_balance="balanced"),
        source_up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    assert metrics.source_up_dot_build_up > 0.45
    assert metrics.upside_down_penalty == pytest.approx(0.0)
    assert metrics.selection_reason == "source_up_preserved"


def test_bumpy_surface_has_higher_saliency_than_flat_base():
    base = trimesh.creation.box(extents=[40.0, 20.0, 10.0])
    bump = trimesh.creation.box(extents=[8.0, 8.0, 6.0])
    bump.apply_translation([0.0, 0.0, 8.0])
    mesh = trimesh.util.concatenate([base, bump])

    saliency = _face_saliency(mesh)
    areas = np.asarray(mesh.area_faces)

    assert float(saliency.max()) > 0.5
    assert float(saliency[areas >= 200.0].mean()) < float(saliency[areas <= 32.0].mean())


def test_compact_balance_still_reports_surface_diagnostics():
    mesh = trimesh.creation.box(extents=[120.0, 6.0, 6.0])

    _, _, balanced = find_stable_overhang_rotation(
        mesh,
        n_candidates=80,
        printer_dims=(200.0, 200.0, 200.0),
        support_tolerance_ratio=0.5,
        resin_options=ResinOrientationOptions(resin_balance="balanced"),
    )
    _, _, compact = find_stable_overhang_rotation(
        mesh,
        n_candidates=80,
        printer_dims=(200.0, 200.0, 200.0),
        support_tolerance_ratio=0.5,
        resin_options=ResinOrientationOptions(resin_balance="compact"),
    )

    assert compact.surface_damage_proxy >= 0.0
    assert -1.0 <= compact.source_up_dot_build_up <= 1.0
    assert compact.upside_down_penalty >= 0.0
    assert compact.stability_score <= balanced.stability_score
