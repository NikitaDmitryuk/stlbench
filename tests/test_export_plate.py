from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from stlbench.config.enums import ExportCompressionMode
from stlbench.export.plate import _compression_options, export_plate_3mf_lazy
from stlbench.packing.rectpack_plate import PackedPlate, PackedRect


def test_export_plate_3mf_lazy_writes_manifest_and_loads_only_plate_parts(tmp_path: Path):
    meshes = [
        trimesh.creation.box(extents=(10.0, 20.0, 5.0)),
        trimesh.creation.box(extents=(8.0, 12.0, 4.0)),
        trimesh.creation.box(extents=(5.0, 5.0, 5.0)),
    ]
    calls: list[int] = []

    def load_part(part_index: int) -> trimesh.Trimesh:
        calls.append(part_index)
        copied = meshes[part_index].copy()
        assert isinstance(copied, trimesh.Trimesh)
        return copied

    plate = PackedPlate(
        index=2,
        rects=(
            PackedRect(part_index=1, x=3.0, y=4.0, width=12.0, height=8.0, rotation_deg=90.0),
            PackedRect(part_index=0, x=20.0, y=6.0, width=10.0, height=20.0, rotation_deg=0.0),
        ),
    )
    out_3mf = tmp_path / "plate.3mf"
    out_json = tmp_path / "plate.json"

    export_plate_3mf_lazy(
        load_part,
        plate,
        out_3mf,
        names=["a.stl", "b.stl", "c.stl"],
        out_manifest=out_json,
    )

    assert calls == [1, 0]
    assert out_3mf.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["plate_index"] == 2
    assert [p["index"] for p in payload["parts"]] == [1, 0]
    assert [p["name"] for p in payload["parts"]] == ["b.stl", "a.stl"]
    assert payload["parts"][0]["rotation_deg"] == 90.0


def test_compression_options_accept_enum_modes():
    assert _compression_options(ExportCompressionMode.DEFAULT) == (zipfile.ZIP_DEFLATED, 5)
    assert _compression_options(ExportCompressionMode.FAST) == (zipfile.ZIP_DEFLATED, 1)
    assert _compression_options(ExportCompressionMode.STORE) == (zipfile.ZIP_STORED, None)


def test_export_plate_3mf_lazy_welds_duplicate_vertices(tmp_path: Path):
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]]),
        process=False,
    )
    plate = PackedPlate(
        index=0,
        rects=(PackedRect(part_index=0, x=0.0, y=0.0, width=1.0, height=1.0),),
    )
    out_3mf = tmp_path / "plate.3mf"

    export_plate_3mf_lazy(lambda _part_index: mesh.copy(), plate, out_3mf)

    with zipfile.ZipFile(out_3mf) as zf:
        model_xml = zf.read("3D/3dmodel.model").decode("utf-8")
    assert model_xml.count("<vertex ") == 3


@pytest.mark.parametrize("compression_mode", ["default", "fast", "store"])
def test_export_plate_3mf_lazy_compression_modes_are_loadable(
    tmp_path: Path, compression_mode: str
):
    meshes = [
        trimesh.creation.box(extents=(10.0, 20.0, 5.0)),
        trimesh.creation.box(extents=(8.0, 12.0, 4.0)),
    ]
    plate = PackedPlate(
        index=0,
        rects=(
            PackedRect(part_index=0, x=0.0, y=0.0, width=10.0, height=20.0),
            PackedRect(part_index=1, x=20.0, y=0.0, width=8.0, height=12.0),
        ),
    )
    out_3mf = tmp_path / f"plate-{compression_mode}.3mf"
    out_json = tmp_path / f"plate-{compression_mode}.json"

    export_plate_3mf_lazy(
        lambda part_index: meshes[part_index].copy(),
        plate,
        out_3mf,
        names=["a.stl", "b.stl"],
        out_manifest=out_json,
        compression_mode=compression_mode,
    )

    loaded = trimesh.load(out_3mf, force="scene")
    assert isinstance(loaded, trimesh.Scene)
    assert len(loaded.geometry) == 2
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert [p["name"] for p in payload["parts"]] == ["a.stl", "b.stl"]


def test_export_plate_3mf_lazy_rejects_invalid_compression_mode(tmp_path: Path):
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 5.0))
    plate = PackedPlate(
        index=0,
        rects=(PackedRect(part_index=0, x=0.0, y=0.0, width=10.0, height=20.0),),
    )

    with pytest.raises(ValueError, match="compression_mode"):
        export_plate_3mf_lazy(
            lambda _part_index: mesh.copy(),
            plate,
            tmp_path / "plate.3mf",
            compression_mode="invalid",
        )


def test_export_plate_3mf_lazy_rejects_underestimated_footprint(tmp_path: Path):
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 5.0))
    plate = PackedPlate(
        index=0,
        rects=(PackedRect(part_index=0, x=0.0, y=0.0, width=9.9, height=20.0),),
    )

    with pytest.raises(ValueError, match="smaller than exported mesh bounds"):
        export_plate_3mf_lazy(
            lambda _part_index: mesh.copy(),
            plate,
            tmp_path / "plate.3mf",
            out_manifest=tmp_path / "plate.json",
        )
