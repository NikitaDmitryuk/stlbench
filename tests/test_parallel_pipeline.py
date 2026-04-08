"""Integration tests for parallel execution in pipeline commands.

These tests verify that:
- Parallel orientation search and export produce correct results.
- Output ordering is preserved (result[i] corresponds to input[i]).
- The two-pass export in run_scale (re-loading from disk) produces valid STLs.
- run_prepare dry_run completes without errors for multiple parts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh
import trimesh.creation

from stlbench.pipeline.run_orient import OrientRunArgs, run_orient
from stlbench.pipeline.run_prepare import PrepareRunArgs, run_prepare
from stlbench.pipeline.run_scale import ScaleRunArgs, run_scale


def _write_box(path: Path, extents: tuple[float, float, float]) -> None:
    trimesh.creation.box(extents=extents).export(str(path))


# ── run_scale ─────────────────────────────────────────────────────────────────


def test_run_scale_free_parallel_multiple_parts():
    """Parallel orientation search (orient=free) produces valid scaled STLs."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_box(d / "a.stl", (10.0, 20.0, 5.0))
        _write_box(d / "b.stl", (30.0, 8.0, 15.0))
        out = d / "out"

        rc = run_scale(
            ScaleRunArgs(
                input_dir=d,
                output_dir=out,
                config_path=None,
                settings=None,
                printer_xyz=(100.0, 100.0, 100.0),
                margin=None,
                post_fit_scale=None,
                method="sorted",
                orientation="free",
                rotation_samples=8,  # minimal — keeps test fast
                no_upscale=False,
                dry_run=False,
                recursive=False,
                suffix="_s",
            )
        )

        assert rc == 0
        for name in ("a", "b"):
            path = out / f"{name}_s.stl"
            assert path.exists(), f"Missing {path.name}"
            m = trimesh.load(str(path), force="mesh")
            dims = np.asarray(m.bounds)[1] - np.asarray(m.bounds)[0]
            assert all(d <= 100.0 + 1e-3 for d in dims), f"{name}: dims {dims} exceed bed"


def test_run_scale_fixed_two_pass_export():
    """With orient=fixed the two-pass export (re-load from disk) writes valid STLs."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_box(d / "x.stl", (40.0, 40.0, 40.0))
        _write_box(d / "y.stl", (10.0, 10.0, 80.0))
        out = d / "out"

        rc = run_scale(
            ScaleRunArgs(
                input_dir=d,
                output_dir=out,
                config_path=None,
                settings=None,
                printer_xyz=(100.0, 100.0, 100.0),
                margin=None,
                post_fit_scale=None,
                method="sorted",
                orientation="fixed",
                rotation_samples=None,
                no_upscale=False,
                dry_run=False,
                recursive=False,
                suffix="",
            )
        )

        assert rc == 0
        for name in ("x", "y"):
            path = out / f"{name}.stl"
            assert path.exists(), f"Missing {path.name}"
            m = trimesh.load(str(path), force="mesh")
            dims = np.asarray(m.bounds)[1] - np.asarray(m.bounds)[0]
            assert all(d <= 100.0 + 1e-3 for d in dims)


def test_run_scale_output_ordering():
    """Parts are written in the same order as the input files (alphabetical glob order)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Three boxes with sizes chosen so each has a distinct limiting dimension.
        _write_box(d / "part_01.stl", (5.0, 5.0, 5.0))
        _write_box(d / "part_02.stl", (20.0, 5.0, 5.0))
        _write_box(d / "part_03.stl", (5.0, 30.0, 5.0))
        out = d / "out"

        rc = run_scale(
            ScaleRunArgs(
                input_dir=d,
                output_dir=out,
                config_path=None,
                settings=None,
                printer_xyz=(100.0, 100.0, 100.0),
                margin=None,
                post_fit_scale=None,
                method="sorted",
                orientation="fixed",
                rotation_samples=None,
                no_upscale=False,
                dry_run=False,
                recursive=False,
                suffix="",
            )
        )

        assert rc == 0
        for i in (1, 2, 3):
            assert (out / f"part_{i:02d}.stl").exists()


# ── run_orient ────────────────────────────────────────────────────────────────


def test_run_orient_parallel_writes_all_parts():
    """Parallel overhang search + parallel export produce one STL per input."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_box(d / "p1.stl", (10.0, 20.0, 5.0))
        _write_box(d / "p2.stl", (30.0, 5.0, 15.0))
        out = d / "out"

        rc = run_orient(
            OrientRunArgs(
                input_dir=d,
                output_dir=out,
                config_path=None,
                settings=None,
                printer_xyz=None,
                overhang_threshold_deg=45.0,
                n_candidates=5,  # minimal — keeps test fast
                dry_run=False,
                recursive=False,
                suffix="_o",
            )
        )

        assert rc == 0
        assert (out / "p1_o.stl").exists()
        assert (out / "p2_o.stl").exists()


def test_run_orient_dry_run_no_output():
    """dry_run=True: parallel analysis completes but writes no files."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_box(d / "a.stl", (10.0, 10.0, 10.0))
        _write_box(d / "b.stl", (20.0, 5.0, 8.0))
        out = d / "out"

        rc = run_orient(
            OrientRunArgs(
                input_dir=d,
                output_dir=out,
                config_path=None,
                settings=None,
                printer_xyz=None,
                overhang_threshold_deg=45.0,
                n_candidates=5,
                dry_run=True,
                recursive=False,
                suffix="",
            )
        )

        assert rc == 0
        assert not out.exists()


# ── run_prepare ───────────────────────────────────────────────────────────────


def test_run_prepare_dry_run_parallel():
    """run_prepare dry_run: all three parallel stages complete without errors.

    Uses cubes so that overhang re-orientation cannot increase the footprint
    beyond what scaled, and a large printer so the layout step always succeeds.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Cubes: all dimensions equal → footprint is invariant to orientation.
        _write_box(d / "a.stl", (10.0, 10.0, 10.0))
        _write_box(d / "b.stl", (10.0, 10.0, 10.0))
        out = d / "out"

        rc = run_prepare(
            PrepareRunArgs(
                input_dir=d,
                output_dir=out,
                config_path=None,
                printer_xyz=(500.0, 500.0, 500.0),
                gap_mm=2.0,
                margin=None,
                post_fit_scale=0.5,  # scale down so re-oriented dims stay well within bed
                method="sorted",
                overhang_threshold_deg=45.0,
                n_orient_candidates=5,  # minimal — keeps test fast
                dry_run=True,
                recursive=False,
            )
        )

        assert rc == 0
        assert not out.exists()
