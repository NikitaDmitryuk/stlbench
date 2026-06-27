import importlib
import json
from pathlib import Path

import pytest
import trimesh
from shapely import affinity
from shapely.geometry import box

from stlbench.packing.polygon_pack import footprints_to_box_polygons
from stlbench.packing.rectpack_plate import PackedPlate, PackedRect
from stlbench.pipeline.run_autopack import (
    AutopackRunArgs,
    _attempt_path,
    _bisect_scale,
    _load_pack_attempt,
    _pack_cache_key,
    _PackAttempt,
    _search_scale_cached_parallel,
    _try_pack_all,
    _write_autopack_result,
    _write_pack_attempt,
    run_autopack,
)
from stlbench.profiling import ProfileOptions

run_autopack_module = importlib.import_module("stlbench.pipeline.run_autopack")


def _l_shape(size: float = 10.0, arm: float = 4.0):
    return box(0.0, 0.0, size, arm).union(box(0.0, 0.0, arm, size))


def _write_l_prism(path: Path, size: float = 10.0, arm: float = 4.0, height: float = 2.0) -> None:
    horiz = trimesh.creation.box(extents=(size, arm, height))
    horiz.apply_translation((size / 2.0, arm / 2.0, height / 2.0))
    vert = trimesh.creation.box(extents=(arm, size, height))
    vert.apply_translation((arm / 2.0, size / 2.0, height / 2.0))
    mesh = trimesh.util.concatenate([horiz, vert])
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def test_try_pack_all_fits():
    polygons = footprints_to_box_polygons([(10.0, 10.0), (20.0, 10.0)])
    plate = _try_pack_all(polygons, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) == 2


def test_try_pack_all_oversized():
    polygons = footprints_to_box_polygons([(200.0, 10.0), (10.0, 10.0)])
    plate = _try_pack_all(polygons, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plate is None


def test_try_pack_all_tight():
    polygons = footprints_to_box_polygons([(95.0, 95.0), (95.0, 95.0)])
    plate = _try_pack_all(polygons, bed_w=100.0, bed_h=100.0, gap_mm=1.0)
    assert plate is None


def test_bisect_scale_finds_positive():
    base_polygons = footprints_to_box_polygons([(10.0, 10.0), (20.0, 10.0)])
    s, plate = _bisect_scale(base_polygons, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=10.0)
    assert s > 0
    assert plate is not None
    assert len(plate.rects) == 2


def test_bisect_scale_respects_upper():
    base_polygons = footprints_to_box_polygons([(5.0, 5.0)])
    s, plate = _bisect_scale(base_polygons, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=3.0)
    assert s > 0
    assert s <= 3.0 + 1e-4
    assert plate is not None


def test_bisect_scale_impossible():
    base_polygons = footprints_to_box_polygons([(200.0, 200.0)])
    s, plate = _bisect_scale(base_polygons, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=0.4)
    assert s > 0
    assert plate is not None


def test_bisect_scale_multiple_parts():
    base_polygons = footprints_to_box_polygons([(30.0, 30.0), (30.0, 30.0), (30.0, 30.0)])
    s, plate = _bisect_scale(base_polygons, bed_w=100.0, bed_h=100.0, gap_mm=1.0, s_upper=5.0)
    assert s > 0
    assert plate is not None
    assert len(plate.rects) == 3


def test_bisect_scale_uses_concave_polygon_space_better_than_bbox():
    l1 = _l_shape()
    l2 = affinity.rotate(_l_shape(), 180.0, origin=(5.0, 5.0))
    bbox_polygons = footprints_to_box_polygons([(10.0, 10.0), (10.0, 10.0)])

    exact_s, exact_plate = _bisect_scale([l1, l2], bed_w=12.0, bed_h=12.0, gap_mm=0.0, s_upper=2.0)
    bbox_s, bbox_plate = _bisect_scale(
        bbox_polygons, bed_w=12.0, bed_h=12.0, gap_mm=0.0, s_upper=2.0
    )

    assert exact_plate is not None
    assert bbox_plate is not None
    assert exact_s > bbox_s * 1.15
    assert exact_s == pytest.approx(0.8445, abs=1e-3)


def test_autopack_pack_cache_key_changes_with_gap(tmp_path: Path):
    source = tmp_path / "part.stl"
    source.write_text("solid x\nendsolid x\n", encoding="utf-8")
    source_paths = [source]
    repair_cache_keys: list[str | None] = ["repair-a"]
    footprint_keys = ["footprint-a"]
    names = ["part.stl"]
    orientation_transforms = [trimesh.transformations.identity_matrix()]
    key_a = _pack_cache_key(
        source_paths=source_paths,
        repair_cache_keys=repair_cache_keys,
        footprint_keys=footprint_keys,
        names=names,
        orientation_transforms=orientation_transforms,
        bed_w=100.0,
        bed_h=50.0,
        gap_mm=1.0,
        geometric_upper=1.0,
        post_fit_scale=0.95,
        scale_tolerance=1e-4,
    )
    key_b = _pack_cache_key(
        source_paths=source_paths,
        repair_cache_keys=repair_cache_keys,
        footprint_keys=footprint_keys,
        names=names,
        orientation_transforms=orientation_transforms,
        bed_w=100.0,
        bed_h=50.0,
        gap_mm=2.0,
        geometric_upper=1.0,
        post_fit_scale=0.95,
        scale_tolerance=1e-4,
    )
    assert key_a != key_b


def test_autopack_attempt_cache_ignores_incomplete_temp(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    path = _attempt_path(cache_dir, "pack-key", 0.5)
    path.parent.mkdir(parents=True)
    (path.parent / f".{path.name}.broken.tmp").write_text("{}", encoding="utf-8")

    assert _load_pack_attempt(cache_dir, "pack-key", 0.5) is None


def test_autopack_attempt_cache_roundtrip(tmp_path: Path):
    plate = PackedPlate(
        index=0,
        rects=(PackedRect(part_index=0, x=1.0, y=2.0, width=3.0, height=4.0),),
    )
    attempt = _PackAttempt(scale=0.5, success=True, plate=plate, duration_s=1.25)

    _write_pack_attempt(tmp_path, "pack-key", attempt)
    loaded = _load_pack_attempt(tmp_path, "pack-key", 0.5)

    assert loaded is not None
    assert loaded.cache_hit
    assert loaded.success
    assert loaded.plate == plate


def test_run_autopack_dry_run_uses_exact_polygon_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    input_dir = tmp_path / "input"
    _write_l_prism(input_dir / "a.stl")
    _write_l_prism(input_dir / "b.stl")
    profile_dir = tmp_path / "profile"

    def _identity_orientation(mesh, *_args, **_kwargs):
        bounds = mesh.bounds
        dims = tuple(float(v) for v in bounds[1] - bounds[0])
        return trimesh.transformations.identity_matrix(), dims

    monkeypatch.setattr(run_autopack_module, "select_orientation_for_scale", _identity_orientation)

    rc = run_autopack(
        AutopackRunArgs(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            config_path=None,
            printer_xyz=(12.0, 12.0, 20.0),
            gap_mm=0.0,
            post_fit_scale=1.0,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=True,
            recursive=False,
            edge_margin_mm=0.0,
            orientation_policy="max-scale",
            autopack_packer="exact",
            profile_options=ProfileOptions(enabled=True, profile_dir=profile_dir),
        )
    )

    assert rc == 0
    payload = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    scale = payload["metadata"]["autopack_scale"]
    assert scale["parts"] == 2
    assert scale["final_scale"] == pytest.approx(0.8297, abs=1e-3)
    assert not (tmp_path / "out").exists()


def test_run_autopack_bitmap_does_not_call_exact_packer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    input_dir = tmp_path / "input"
    _write_l_prism(input_dir / "a.stl")
    _write_l_prism(input_dir / "b.stl")

    def _identity_orientation(mesh, *_args, **_kwargs):
        bounds = mesh.bounds
        dims = tuple(float(v) for v in bounds[1] - bounds[0])
        return trimesh.transformations.identity_matrix(), dims

    def _fail_exact(*_args, **_kwargs):
        raise AssertionError("bitmap scale search should not call exact NFP packer")

    monkeypatch.setattr(run_autopack_module, "select_orientation_for_scale", _identity_orientation)
    monkeypatch.setattr(run_autopack_module, "_try_pack_all", _fail_exact)

    rc = run_autopack(
        AutopackRunArgs(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            config_path=None,
            printer_xyz=(25.0, 25.0, 20.0),
            gap_mm=0.0,
            post_fit_scale=1.0,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=True,
            recursive=False,
            edge_margin_mm=0.0,
            orientation_policy="max-scale",
            autopack_packer="bitmap",
            autopack_bitmap_grid_mm=0.5,
        )
    )

    assert rc == 0


def test_run_autopack_writes_transform_log(tmp_path: Path):
    input_dir = tmp_path / "input"
    _write_l_prism(input_dir / "part.stl")
    out_dir = tmp_path / "out"

    rc = run_autopack(
        AutopackRunArgs(
            input_dir=input_dir,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(30.0, 30.0, 30.0),
            gap_mm=0.0,
            post_fit_scale=1.0,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=False,
            recursive=False,
            edge_margin_mm=0.0,
            orientation_policy="max-scale",
        )
    )

    assert rc == 0
    payload = json.loads((out_dir / "transforms.json").read_text(encoding="utf-8"))
    assert payload["command"] == "autopack"
    assert len(payload["parts"]) == 1


def test_run_autopack_uses_valid_result_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    _write_l_prism(input_dir / "part.stl")
    out_dir = tmp_path / "out"

    def _identity_orientation(mesh, *_args, **_kwargs):
        bounds = mesh.bounds
        dims = tuple(float(v) for v in bounds[1] - bounds[0])
        return trimesh.transformations.identity_matrix(), dims

    monkeypatch.setattr(run_autopack_module, "select_orientation_for_scale", _identity_orientation)

    rc = run_autopack(
        AutopackRunArgs(
            input_dir=input_dir,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(30.0, 30.0, 30.0),
            gap_mm=0.0,
            post_fit_scale=1.0,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=False,
            recursive=False,
            edge_margin_mm=0.0,
            orientation_policy="max-scale",
        )
    )
    assert rc == 0

    def _fail_search(*_args, **_kwargs):
        raise AssertionError("result cache should skip scale search")

    monkeypatch.setattr(run_autopack_module, "_run_candidate_batch", _fail_search)
    monkeypatch.setattr(run_autopack_module, "_pack_at_scale_cached", _fail_search)

    rc = run_autopack(
        AutopackRunArgs(
            input_dir=input_dir,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(30.0, 30.0, 30.0),
            gap_mm=0.0,
            post_fit_scale=1.0,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=True,
            recursive=False,
            edge_margin_mm=0.0,
            orientation_policy="max-scale",
        )
    )
    assert rc == 0


def test_autopack_search_ignores_invalid_result_cache(tmp_path: Path):
    base_polygons = [box(0.0, 0.0, 10.0, 10.0)]
    bad_plate = PackedPlate(
        index=0,
        rects=(PackedRect(part_index=0, x=100.0, y=100.0, width=10.0, height=10.0),),
    )
    _write_autopack_result(tmp_path, "pack-key", 1.0, bad_plate)

    result = _search_scale_cached_parallel(
        base_polygons,
        30.0,
        30.0,
        0.0,
        2.0,
        tol=1e-4,
        pack_workers=1,
        cache_dir=tmp_path,
        pack_key="pack-key",
        read_result_cache=True,
        write_result_cache=False,
        write_attempt_cache=False,
    )
    assert not result.stats.result_cache_hit
    assert result.plate is not None
    assert result.scale > 0
