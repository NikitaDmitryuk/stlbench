import json
from pathlib import Path

import numpy as np
import trimesh

from stlbench.pipeline.run_fill import FillRunArgs, _max_copies_on_plate, run_fill


def test_fill_single_small_rect():
    plate = _max_copies_on_plate(10.0, 10.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) > 1


def test_fill_large_part_one_copy():
    plate = _max_copies_on_plate(90.0, 90.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) == 1


def test_fill_oversized_returns_none():
    plate = _max_copies_on_plate(200.0, 200.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is None


def test_fill_exact_fit():
    plate = _max_copies_on_plate(48.0, 48.0, 100.0, 100.0, gap_mm=1.0)
    assert plate is not None
    assert len(plate.rects) >= 4


def test_fill_respects_gap():
    plate_small_gap = _max_copies_on_plate(10.0, 10.0, 100.0, 100.0, gap_mm=0.5)
    plate_large_gap = _max_copies_on_plate(10.0, 10.0, 100.0, 100.0, gap_mm=5.0)
    assert plate_small_gap is not None
    assert plate_large_gap is not None
    assert len(plate_small_gap.rects) >= len(plate_large_gap.rects)


def test_fill_packing_stays_inside_fractional_bed():
    plate = _max_copies_on_plate(4.93, 3.48, 153.36, 77.76, gap_mm=10.0)
    assert plate is not None
    assert plate.rects
    assert max(r.x + r.width for r in plate.rects) <= 153.36
    assert max(r.y + r.height for r in plate.rects) <= 77.76


def test_run_fill_writes_transform_log(tmp_path: Path):
    input_file = tmp_path / "part.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 10.0)).export(input_file)
    out_dir = tmp_path / "fill-out"

    rc = run_fill(
        FillRunArgs(
            input_file=input_file,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(30.0, 30.0, 30.0),
            gap_mm=1.0,
            scale=False,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=False,
            cleanup=False,
            repair=True,
            any_rotation=False,
            orientation_policy="max-scale",
        )
    )

    assert rc == 0
    payload = json.loads((out_dir / "transforms.json").read_text(encoding="utf-8"))
    assert payload["command"] == "fill"
    assert payload["metadata"]["copies"] == len(payload["parts"])
    assert payload["parts"][0]["steps"][0]["name"] == "repair"
    repair_payload = json.loads((out_dir / "repair_report.json").read_text(encoding="utf-8"))
    assert repair_payload["command"] == "fill"
    assert len(repair_payload["parts"]) == 1


def test_run_fill_3mf_scene_bounds_stay_inside_printer(tmp_path: Path):
    input_file = tmp_path / "part.stl"
    mesh = trimesh.creation.box(extents=(9.0, 4.0, 6.0))
    mesh.apply_translation([12.0, -7.0, 3.0])
    mesh.export(input_file)
    out_dir = tmp_path / "fill-out"

    rc = run_fill(
        FillRunArgs(
            input_file=input_file,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(30.0, 20.0, 30.0),
            gap_mm=2.0,
            scale=False,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=False,
            cleanup=False,
            repair=False,
            any_rotation=False,
            orientation_policy="max-scale",
        )
    )

    assert rc == 0
    scene = trimesh.load(out_dir / "fill_plate.3mf", force="scene")
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    assert bounds[0, 0] >= -1e-6
    assert bounds[0, 1] >= -1e-6
    assert bounds[0, 2] >= -1e-6
    assert bounds[1, 0] <= 30.0 + 1e-6
    assert bounds[1, 1] <= 20.0 + 1e-6
    assert bounds[1, 2] <= 30.0 + 1e-6


def test_run_fill_no_repair_omits_repair_step_and_report(tmp_path: Path):
    input_file = tmp_path / "part.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 10.0)).export(input_file)
    out_dir = tmp_path / "fill-out"

    rc = run_fill(
        FillRunArgs(
            input_file=input_file,
            output_dir=out_dir,
            config_path=None,
            printer_xyz=(30.0, 30.0, 30.0),
            gap_mm=1.0,
            scale=False,
            orient_on=False,
            orient_threshold_deg=45.0,
            dry_run=False,
            cleanup=False,
            repair=False,
            any_rotation=False,
            orientation_policy="max-scale",
        )
    )

    assert rc == 0
    payload = json.loads((out_dir / "transforms.json").read_text(encoding="utf-8"))
    assert all(step["name"] != "repair" for step in payload["parts"][0]["steps"])
    assert (out_dir / "repair_report.json").exists()
    report = json.loads((out_dir / "repair_report.json").read_text(encoding="utf-8"))
    assert report["parts"][0]["enabled"] is False
