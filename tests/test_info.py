from pathlib import Path

import trimesh.creation

from stlbench.pipeline.run_info import InfoRunArgs, run_info


def _write_box_stl(directory: Path, name: str, extents: tuple[float, float, float]) -> Path:
    box = trimesh.creation.box(extents=extents)
    p = directory / name
    box.export(p)
    return p


def test_info_runs_on_boxes(tmp_path: Path):
    _write_box_stl(tmp_path, "a.stl", (10.0, 20.0, 30.0))
    _write_box_stl(tmp_path, "b.stl", (5.0, 5.0, 5.0))

    rc = run_info(
        InfoRunArgs(
            input_dir=tmp_path,
            config_path=None,
            printer_xyz=(100.0, 100.0, 100.0),
            recursive=False,
        )
    )
    assert rc == 0


def test_info_empty_dir(tmp_path: Path):
    rc = run_info(
        InfoRunArgs(
            input_dir=tmp_path,
            config_path=None,
            printer_xyz=(100.0, 100.0, 100.0),
            recursive=False,
        )
    )
    assert rc != 0


def test_info_no_printer(tmp_path: Path):
    _write_box_stl(tmp_path, "a.stl", (10.0, 10.0, 10.0))
    rc = run_info(
        InfoRunArgs(
            input_dir=tmp_path,
            config_path=None,
            printer_xyz=None,
            recursive=False,
        )
    )
    assert rc == 2
