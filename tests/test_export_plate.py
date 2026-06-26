from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from stlbench.export.plate import export_plate_3mf_lazy
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
