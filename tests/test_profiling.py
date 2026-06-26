from __future__ import annotations

import json
import pstats
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import trimesh

from stlbench.pipeline.run_layout import LayoutRunArgs, run_layout
from stlbench.pipeline.run_prepare import PrepareRunArgs, run_prepare
from stlbench.pipeline.run_scale import ScaleRunArgs, run_scale
from stlbench.profiling import ProfileOptions, make_profiler


def _write_box(path: Path, extents: tuple[float, float, float]) -> None:
    trimesh.creation.box(extents=extents).export(str(path))


def _worker_hotspot(value: int) -> int:
    return sum(i * value for i in range(100))


def _assert_profile_artifacts(profile_dir: Path, command: str) -> dict[str, Any]:
    json_path = profile_dir / "profile.json"
    txt_path = profile_dir / "profile.txt"
    pstats_path = profile_dir / "profile.pstats"
    assert json_path.exists()
    assert txt_path.exists()
    assert pstats_path.exists()
    payload: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["command"] == command
    assert payload["status"] == "ok"
    assert payload["return_code"] == 0
    assert payload["stages"]
    assert payload["top_functions"]
    assert int(getattr(pstats.Stats(str(pstats_path)), "total_calls", 0)) > 0
    return payload


def test_execution_profiler_writes_artifacts_and_merges_worker_stats(tmp_path: Path):
    profile_dir = tmp_path / "profile"
    profiler = make_profiler(
        command="unit",
        output_base=tmp_path,
        options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
    )
    profiler.start()
    with (
        profiler.stage("outer"),
        profiler.stage("inner"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        list(profiler.map(pool, "worker.hotspot", _worker_hotspot, [1, 2, 3]))
    profiler.finish(status="ok", return_code=0)

    payload = _assert_profile_artifacts(profile_dir, "unit")
    assert payload["stages"][0]["name"] == "outer"
    assert payload["stages"][0]["children"][0]["name"] == "inner"
    assert any(w["name"] == "worker.hotspot" for w in payload["workers"])
    assert any(f["function"] == "_worker_hotspot" for f in payload["top_functions"])


def test_run_scale_dry_run_profile_creates_artifacts(tmp_path: Path):
    _write_box(tmp_path / "part.stl", (10.0, 20.0, 30.0))
    profile_dir = tmp_path / "scale-profile"

    rc = run_scale(
        ScaleRunArgs(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            config_path=None,
            settings=None,
            printer_xyz=(100.0, 100.0, 100.0),
            post_fit_scale=None,
            method="sorted",
            rotation_samples=4,
            dry_run=True,
            recursive=False,
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
        )
    )

    assert rc == 0
    payload = _assert_profile_artifacts(profile_dir, "scale")
    assert any(s["name"] == "orientation search" for s in payload["stages"])


def test_run_layout_dry_run_profile_creates_artifacts(tmp_path: Path):
    _write_box(tmp_path / "part.stl", (10.0, 20.0, 30.0))
    profile_dir = tmp_path / "layout-profile"

    rc = run_layout(
        LayoutRunArgs(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            config_path=None,
            printer_xyz=(100.0, 100.0, 100.0),
            gap_mm=1.0,
            recursive=False,
            dry_run=True,
            rotation_samples=4,
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
        )
    )

    assert rc == 0
    payload = _assert_profile_artifacts(profile_dir, "layout")
    assert any(s["name"] == "packing" for s in payload["stages"])


def test_run_prepare_dry_run_profile_creates_artifacts(tmp_path: Path):
    _write_box(tmp_path / "part.stl", (10.0, 20.0, 30.0))
    profile_dir = tmp_path / "prepare-profile"

    rc = run_prepare(
        PrepareRunArgs(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            config_path=None,
            printer_xyz=(100.0, 100.0, 100.0),
            gap_mm=1.0,
            post_fit_scale=None,
            method="sorted",
            overhang_threshold_deg=45.0,
            n_orient_candidates=8,
            dry_run=True,
            recursive=False,
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
        )
    )

    assert rc == 0
    payload = _assert_profile_artifacts(profile_dir, "prepare")
    assert any(s["name"] == "packing" for s in payload["stages"])
    assert payload["metadata"]["resource_plan"]["requested"] == "auto"
    assert payload["metadata"]["resource_plan"]["scale_workers"] >= 1
    assert payload["metadata"]["orientation_options"]["resin_balance"] == "balanced"
    orient = payload["metadata"]["orientation_stability"]
    assert len(orient) == 1
    assert "support_contact_proxy" in orient[0]
    assert "surface_damage_proxy" in orient[0]
    assert "source_up_dot_build_up" in orient[0]
    assert "upside_down_penalty" in orient[0]
    assert "xy_footprint_area_mm2" in orient[0]
    assert "selection_reason" in orient[0]


def test_run_prepare_writes_orient_cache_refs(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_box(input_dir / "part.stl", (10.0, 20.0, 30.0))
    out_dir = tmp_path / "out"

    rc = run_prepare(
        PrepareRunArgs(
            input_dir=input_dir,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(100.0, 100.0, 100.0),
            gap_mm=1.0,
            post_fit_scale=None,
            method="sorted",
            overhang_threshold_deg=45.0,
            n_orient_candidates=8,
            dry_run=False,
            recursive=False,
        )
    )

    assert rc == 0
    meta_path = out_dir / "cache" / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["names"] == ["part.stl"]
    assert len(payload["mesh_files"]) == 1
    assert (out_dir / "cache" / payload["mesh_files"][0]).exists()
