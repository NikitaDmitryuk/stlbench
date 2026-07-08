from __future__ import annotations

import json
import pstats
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from shapely.geometry import box

from stlbench.config.enums import PackerBackend
from stlbench.core.mesh_repair import RepairOptions
from stlbench.core.overhang import ResinOrientationOptions
from stlbench.packing.bitmap_pack import BitmapPackOptions
from stlbench.pipeline.run_layout import LayoutRunArgs, run_layout
from stlbench.pipeline.run_prepare import (
    PrepareRunArgs,
    _packing_cache_key,
    _prepare_cache_worker,
    _PrepareCacheJob,
    _resolve_prepare_packer,
    _scale_polygons,
    _search_prepare_layout_scale,
    _select_retained_indices,
    _validate_layout_geometry,
    run_prepare,
)
from stlbench.pipeline.run_scale import ScaleRunArgs, run_scale
from stlbench.profiling import ProfileOptions, make_profiler


def _write_box(path: Path, extents: tuple[float, float, float]) -> None:
    trimesh.creation.box(extents=extents).export(str(path))


def _prepare_args(
    input_dir: Path,
    out_dir: Path,
    *,
    packer: str | None = None,
    resume: bool = False,
    profile_options: ProfileOptions | None = None,
    export_compression: str = "default",
    orientation_strategy: str = "auto",
    orientation_quality: str = "adaptive",
) -> PrepareRunArgs:
    return PrepareRunArgs(
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
        packer=packer,
        resume=resume,
        profile_options=profile_options,
        export_compression=export_compression,
        orientation_strategy=orientation_strategy,
        orientation_quality=orientation_quality,
    )


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


def test_resolve_prepare_packer_returns_enum() -> None:
    assert _resolve_prepare_packer(None, None) is PackerBackend.EXACT
    assert _resolve_prepare_packer("auto", None) is PackerBackend.EXACT
    assert _resolve_prepare_packer("bitmap", None) is PackerBackend.BITMAP


def test_select_retained_indices_respects_cap_and_largest_first(tmp_path: Path):
    sizes = [100, 30, 20, 10]
    paths: list[Path] = []
    for idx, size in enumerate(sizes):
        path = tmp_path / f"part-{idx}.stl"
        path.write_bytes(b"x" * size)
        paths.append(path)

    retained, metadata = _select_retained_indices(
        paths,
        memory_budget_bytes=100_000,
        orient_workers=3,
    )

    assert retained == {0, 1}
    assert metadata["retained_indices"] == [0, 1]
    assert metadata["estimated_retained_bytes"] == 520
    assert metadata["cap_bytes"] == 540
    assert metadata["max_retained_parts"] == 3


def test_prepare_scale_search_shrinks_to_requested_plate_count():
    polygons = [box(0.0, 0.0, 60.0, 60.0) for _ in range(4)]

    scale, plates, metadata = _search_prepare_layout_scale(
        polygons,
        100.0,
        100.0,
        0.0,
        0.0,
        packer="exact",
        bitmap_options=BitmapPackOptions(),
        grid_step_mm=1.0,
        max_plates=1,
        part_heights=[10.0, 10.0, 10.0, 10.0],
        tolerance=1e-3,
    )

    assert plates is not None
    assert len(plates) == 1
    assert scale == pytest.approx(100.0 / 120.0, abs=0.02)
    assert metadata["max_plates"] == 1
    assert metadata["attempts_run"] > 0
    assert metadata["scale_search_version"] == "scale_search_v3"
    assert metadata["scale_attempts"]
    assert metadata["best_reused_as_final"] is True
    assert not any(attempt["kind"] == "final" for attempt in metadata["scale_attempts"])
    assert _validate_layout_geometry(_scale_polygons(polygons, scale), plates, 100.0, 100.0, 0.0)


def test_prepare_scale_search_keeps_full_scale_when_full_scale_fits_two_plates():
    polygons = [box(0.0, 0.0, 90.0, 90.0) for _ in range(2)]

    scale, plates, metadata = _search_prepare_layout_scale(
        polygons,
        100.0,
        100.0,
        1.0,
        0.0,
        packer="exact",
        bitmap_options=BitmapPackOptions(),
        grid_step_mm=1.0,
        max_plates=2,
        part_heights=[10.0, 10.0],
        tolerance=1e-3,
    )

    assert plates is not None
    assert len(plates) == 2
    assert scale >= 1.0 - 1e-3
    assert metadata["final_plates"] == 2


def test_prepare_scale_search_precheck_skips_impossible_scales_before_exact_pack():
    polygons = [box(0.0, 0.0, 50.0, 50.0) for _ in range(4)]

    scale, plates, metadata = _search_prepare_layout_scale(
        polygons,
        100.0,
        100.0,
        100.0,
        0.0,
        packer="exact",
        bitmap_options=BitmapPackOptions(),
        grid_step_mm=1.0,
        max_plates=1,
        part_heights=[10.0, 10.0, 10.0, 10.0],
        tolerance=1e-2,
    )

    assert scale >= 0.0
    assert plates is None or len(plates) <= 1
    assert metadata["skipped_by_precheck"] > 0
    assert any(
        attempt["kind"].endswith("-precheck") and attempt["fits"] is False
        for attempt in metadata["scale_attempts"]
    )


def test_prepare_packing_cache_key_includes_scale_search_options():
    kwargs = {
        "footprint_keys": ["a", "b"],
        "bed_w": 100.0,
        "bed_h": 100.0,
        "gap_mm": 1.0,
        "edge_margin_mm": 2.0,
        "part_heights": [10.0, 20.0],
        "packer": "exact",
        "bitmap_options": None,
        "grid_step_mm": 1.0,
    }

    base = _packing_cache_key(
        **kwargs,
        max_plates=1,
        scale_tolerance=1e-4,
        post_fit_scale=0.95,
    )

    assert base != _packing_cache_key(
        **kwargs,
        max_plates=2,
        scale_tolerance=1e-4,
        post_fit_scale=0.95,
    )
    assert base != _packing_cache_key(
        **kwargs,
        max_plates=1,
        scale_tolerance=1e-3,
        post_fit_scale=0.95,
    )
    assert base != _packing_cache_key(
        **kwargs,
        max_plates=1,
        scale_tolerance=1e-4,
        post_fit_scale=0.9,
    )
    assert base != _packing_cache_key(
        **{
            **kwargs,
            "packer": "bitmap",
            "bitmap_options": BitmapPackOptions(),
        },
        max_plates=1,
        scale_tolerance=1e-4,
        post_fit_scale=0.95,
    )


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


def test_run_layout_writes_transform_log(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_box(input_dir / "part.stl", (10.0, 20.0, 30.0))
    out_dir = tmp_path / "layout-out"

    rc = run_layout(
        LayoutRunArgs(
            input_dir=input_dir,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(100.0, 100.0, 100.0),
            gap_mm=1.0,
            recursive=False,
            dry_run=False,
            rotation_samples=4,
        )
    )

    assert rc == 0
    payload = json.loads((out_dir / "transforms.json").read_text(encoding="utf-8"))
    assert payload["command"] == "layout"
    assert len(payload["parts"]) == 1


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
    assert payload["metadata"]["orientation_options"]["quality"] == "adaptive"
    assert "orientation_retention" in payload["metadata"]
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
    log = json.loads((out_dir / "transforms.json").read_text(encoding="utf-8"))
    assert log["command"] == "prepare"
    assert len(log["parts"]) == 1
    part = log["parts"][0]
    assert part["source_bounds_mm"] is not None
    assert part["final_bounds_mm"] is not None
    assert part["source_to_export_matrix"] is not None
    assert part["export_to_source_matrix"] is not None


def test_run_prepare_fast_export_writes_loadable_3mf_and_profile_metadata(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_box(input_dir / "a.stl", (10.0, 20.0, 30.0))
    _write_box(input_dir / "b.stl", (15.0, 10.0, 20.0))
    out_dir = tmp_path / "out"
    profile_dir = tmp_path / "profile"

    rc = run_prepare(
        _prepare_args(
            input_dir,
            out_dir,
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
            export_compression="fast",
        )
    )

    assert rc == 0
    payload = _assert_profile_artifacts(profile_dir, "prepare")
    assert payload["metadata"]["export"]["compression"] == "fast"
    assert payload["metadata"]["orientation_options"]["strategy"] == "auto"
    assert payload["metadata"]["orientation_timings"]
    assert any(w["metadata"].get("part") for w in payload["workers"])
    assert any(s["name"] == "export" for s in payload["stages"])
    total_geometry = 0
    for plate_path in sorted(out_dir.glob("plate_*.3mf")):
        loaded = trimesh.load(plate_path, force="scene")
        assert isinstance(loaded, trimesh.Scene)
        total_geometry += len(loaded.geometry)
    assert total_geometry == 2


def test_run_prepare_max_plates_exports_one_plate_and_scale_metadata(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_box(input_dir / "a.stl", (60.0, 60.0, 60.0))
    _write_box(input_dir / "b.stl", (60.0, 60.0, 60.0))
    out_dir = tmp_path / "out"
    profile_dir = tmp_path / "profile"

    rc = run_prepare(
        PrepareRunArgs(
            input_dir=input_dir,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(100.0, 100.0, 100.0),
            gap_mm=0.0,
            edge_margin_mm=0.0,
            post_fit_scale=0.9,
            method="sorted",
            overhang_threshold_deg=45.0,
            n_orient_candidates=1,
            dry_run=False,
            recursive=False,
            packer=None,
            max_plates=1,
            scale_tolerance=1e-3,
            workers="1",
            progress=False,
            orientation_strategy="legacy",
            orientation_quality="default",
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
        )
    )

    assert rc == 0
    assert sorted(out_dir.glob("plate_*.3mf")) == [out_dir / "plate_01.3mf"]
    log = json.loads((out_dir / "transforms.json").read_text(encoding="utf-8"))
    assert log["metadata"]["max_plates"] == 1
    assert log["metadata"]["layout_pack_scale"] < 1.0
    assert log["metadata"]["post_fit_scale"] == pytest.approx(0.9)
    assert log["metadata"]["final_plate_count"] == 1
    assert all(
        {"layout_pack_scale", "post_fit_scale"}.issubset({step["name"] for step in part["steps"]})
        for part in log["parts"]
    )
    payload = _assert_profile_artifacts(profile_dir, "prepare")
    packing = payload["metadata"]["packing"]
    assert payload["metadata"]["packing_options"]["requested_packer"] == "auto"
    assert payload["metadata"]["packing_options"]["resolved_packer"] == "exact"
    assert packing["max_plates"] == 1
    assert packing["final_plates"] == 1
    assert packing["layout_pack_scale"] < 1.0
    assert packing["post_fit_scale"] == pytest.approx(0.9)
    assert packing["final_scale"] == pytest.approx(
        packing["s_max"] * packing["layout_pack_scale"] * packing["post_fit_scale"]
    )


def test_run_prepare_adaptive_orientation_quality_records_diagnostics(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_box(input_dir / "rod.stl", (80.0, 6.0, 6.0))
    out_dir = tmp_path / "out"
    profile_dir = tmp_path / "profile"

    rc = run_prepare(
        _prepare_args(
            input_dir,
            out_dir,
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
            orientation_quality="adaptive",
        )
    )

    assert rc == 0
    payload = _assert_profile_artifacts(profile_dir, "prepare")
    assert payload["metadata"]["orientation_options"]["quality"] == "adaptive"
    orient_workers = [w for w in payload["workers"] if w["name"] == "prepare.scale_orient_cache"]
    assert orient_workers
    metadata = orient_workers[0]["metadata"]
    assert metadata["orientation_quality"] == "adaptive"
    assert "adaptive_enabled" in metadata
    assert "candidate_count_default" in metadata
    assert "candidate_count_adaptive" in metadata


def test_prepare_cache_worker_auto_orientation_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    input_path = tmp_path / "part.stl"
    _write_box(input_path, (20.0, 20.0, 40.0))

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forced auto failure")

    monkeypatch.setattr("stlbench.pipeline.run_prepare.find_stable_overhang_rotation", _boom)

    (
        _idx,
        _sb,
        _sa,
        _pct,
        ref,
        metrics_payload,
        _duration_s,
        timing_payload,
    ) = _prepare_cache_worker(
        _PrepareCacheJob(
            index=0,
            path=input_path,
            name=input_path.name,
            cache_dir=tmp_path / "cache",
            scale_transform=np.eye(4, dtype=np.float64),
            source_up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            scale=1.0,
            overhang_threshold_deg=45.0,
            n_orient_candidates=20,
            printer_xyz=(100.0, 100.0, 100.0),
            cleanup=False,
            resin_options=ResinOrientationOptions(),
            repair_options=RepairOptions(),
            repair_cache_dir=None,
            orientation_strategy="auto",
            orientation_quality="default",
        )
    )

    assert ref.cache_path.exists()
    assert metrics_payload["orientation_strategy"] == "legacy"
    assert timing_payload["fallback"] is True
    assert timing_payload["strategy_used"] == "legacy"
    assert "forced auto failure" in str(timing_payload["fallback_reason"])


def test_run_prepare_reuses_footprint_and_packing_cache(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_box(input_dir / "a.stl", (10.0, 20.0, 30.0))
    _write_box(input_dir / "b.stl", (15.0, 10.0, 20.0))
    out_dir = tmp_path / "out"

    rc = run_prepare(_prepare_args(input_dir, out_dir, packer="bitmap"))
    assert rc == 0
    assert any((out_dir / "cache" / "footprints").glob("*.pkl"))
    assert any((out_dir / "cache" / "prepare_packing" / "results").glob("*/result.json"))

    profile_dir = tmp_path / "prepare-cache-profile"
    rc = run_prepare(
        _prepare_args(
            input_dir,
            out_dir,
            packer="bitmap",
            resume=True,
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir, limit=20),
        )
    )

    assert rc == 0
    payload = _assert_profile_artifacts(profile_dir, "prepare")
    assert payload["metadata"]["footprint_cache"]["hits"] == 2
    assert payload["metadata"]["packing"]["cache_hit"] is True


def test_run_prepare_auto_uses_exact_cache_not_bitmap_cache(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_box(input_dir / "a.stl", (10.0, 20.0, 30.0))
    _write_box(input_dir / "b.stl", (15.0, 10.0, 20.0))
    out_dir = tmp_path / "out"

    rc = run_prepare(_prepare_args(input_dir, out_dir, packer="bitmap"))
    assert rc == 0

    first_auto_profile = tmp_path / "prepare-auto-first-profile"
    rc = run_prepare(
        _prepare_args(
            input_dir,
            out_dir,
            packer="auto",
            resume=True,
            profile_options=ProfileOptions(enabled=True, profile_dir=first_auto_profile, limit=20),
        )
    )
    assert rc == 0
    first_payload = _assert_profile_artifacts(first_auto_profile, "prepare")
    assert first_payload["metadata"]["packing_options"]["requested_packer"] == "auto"
    assert first_payload["metadata"]["packing_options"]["resolved_packer"] == "exact"
    assert first_payload["metadata"]["packing"]["resolved_packer"] == "exact"
    assert first_payload["metadata"]["packing"]["cache_hit"] is False

    second_auto_profile = tmp_path / "prepare-auto-second-profile"
    rc = run_prepare(
        _prepare_args(
            input_dir,
            out_dir,
            packer="auto",
            resume=True,
            profile_options=ProfileOptions(enabled=True, profile_dir=second_auto_profile, limit=20),
        )
    )
    assert rc == 0
    second_payload = _assert_profile_artifacts(second_auto_profile, "prepare")
    assert second_payload["metadata"]["footprint_cache"]["hits"] == 2
    assert second_payload["metadata"]["packing"]["resolved_packer"] == "exact"
    assert second_payload["metadata"]["packing"]["cache_hit"] is True
