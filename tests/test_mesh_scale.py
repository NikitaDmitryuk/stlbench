import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh.creation

from stlbench.core.fit import aabb_edge_lengths, compute_global_scale
from stlbench.pipeline.run_scale import ScaleRunArgs, run_scale


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


def _make_scale_args(tmp: Path, **kwargs: Any) -> ScaleRunArgs:
    """Build a minimal ScaleRunArgs for a temp directory."""
    defaults: dict[str, Any] = dict(
        input_dir=tmp,
        output_dir=tmp / "out",
        config_path=None,
        settings=None,
        printer_xyz=(100.0, 100.0, 100.0),
        post_fit_scale=None,
        method="sorted",
        allow_rotation=False,
        maximize=False,
        scale_factor=None,
        rotation_samples=None,
        no_upscale=False,
        dry_run=False,
        recursive=False,
        suffix="",
    )
    defaults.update(kwargs)
    return ScaleRunArgs(**defaults)


def test_explicit_scale_factor():
    """--scale 2.0 doubles all dimensions, ignores printer fit."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        box.export(str(d / "box.stl"))

        rc = run_scale(_make_scale_args(d, scale_factor=2.0, printer_xyz=None))
        assert rc == 0

        out = trimesh.load(str(d / "out" / "box.stl"), force="mesh")
        dims = np.asarray(out.bounds)[1] - np.asarray(out.bounds)[0]
        assert all(abs(v - 20.0) < 1e-3 for v in dims), f"Expected 20³, got {dims}"


def test_explicit_scale_factor_with_post_fit():
    """--scale 2.0 --post-fit-scale 0.5 results in net factor 1.0."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        box.export(str(d / "box.stl"))

        rc = run_scale(_make_scale_args(d, scale_factor=2.0, post_fit_scale=0.5, printer_xyz=None))
        assert rc == 0

        out = trimesh.load(str(d / "out" / "box.stl"), force="mesh")
        dims = np.asarray(out.bounds)[1] - np.asarray(out.bounds)[0]
        assert all(abs(v - 10.0) < 1e-3 for v in dims), f"Expected 10³, got {dims}"


def test_no_rotation_by_default():
    """Default (allow_rotation=False): a non-cubic mesh is not reoriented."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Tall thin box: 5×5×80; place in a 100³ printer.
        # Without rotation, Z=80 is the limiting dim → s = 100/80 = 1.25
        # With rotation (laying it flat), longest dim goes to XY → s could be larger.
        box = trimesh.creation.box(extents=(5.0, 5.0, 80.0))
        box.export(str(d / "tall.stl"))

        rc = run_scale(_make_scale_args(d, allow_rotation=False))
        assert rc == 0

        out = trimesh.load(str(d / "out" / "tall.stl"), force="mesh")
        dims = sorted(np.asarray(out.bounds)[1] - np.asarray(out.bounds)[0])
        # Largest dim must equal 100 (Z was 80, scaled by 100/80)
        assert abs(dims[2] - 100.0) < 1e-2, f"Largest dim should be 100, got {dims[2]:.3f}"


def test_allow_rotation_no_maximize_uses_axis_permutations():
    """allow_rotation=True, maximize=False: tries 6 axis perms, returns a valid fit."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        box = trimesh.creation.box(extents=(5.0, 5.0, 80.0))
        box.export(str(d / "tall.stl"))

        rc = run_scale(_make_scale_args(d, allow_rotation=True, maximize=False))
        assert rc == 0

        out = trimesh.load(str(d / "out" / "tall.stl"), force="mesh")
        dims = np.asarray(out.bounds)[1] - np.asarray(out.bounds)[0]
        assert all(v <= 100.0 + 1e-3 for v in dims), f"Part exceeds bed: {dims}"


def test_maximize_requires_allow_rotation():
    """--maximize without --allow-rotation must return exit code 2."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        trimesh.creation.box(extents=(10.0, 10.0, 10.0)).export(str(d / "box.stl"))

        rc = run_scale(_make_scale_args(d, allow_rotation=False, maximize=True))
        assert rc == 2, f"Expected exit code 2, got {rc}"
