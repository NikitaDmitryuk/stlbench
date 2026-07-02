from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from stlbench.export.transform_log import (
    placement_transform_for_bounds,
    placement_transform_for_mesh,
    shared_geometry_placement_transform_for_mesh,
    transform_bounds,
    transform_entry,
    transform_step,
    translation_matrix,
    uniform_scale_matrix,
    write_transform_log,
)
from stlbench.packing.rectpack_plate import PackedRect


def test_transform_log_writes_schema_and_inverse(tmp_path):
    matrix = translation_matrix([1.0, 2.0, 3.0]) @ uniform_scale_matrix(2.0)
    entry = transform_entry(
        index=0,
        source_path="source.stl",
        source_name="source.stl",
        output_name="part",
        output_file="plate.3mf",
        source_bounds_mm=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        final_bounds_mm=[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        source_to_export_matrix=matrix,
        steps=[transform_step("scale", matrix=uniform_scale_matrix(2.0))],
        scale_factor=2.0,
    )

    out = tmp_path / "transforms.json"
    write_transform_log(out, command="unit", output_files=["plate.3mf"], parts=[entry])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["command"] == "unit"
    assert payload["units"] == "mm"
    assert np.asarray(payload["parts"][0]["source_to_export_matrix"]) == pytest.approx(matrix)
    inv = np.asarray(payload["parts"][0]["export_to_source_matrix"])
    assert inv @ matrix == pytest.approx(np.eye(4))


def test_placement_transform_matches_export_bounds():
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 5.0))
    rect = PackedRect(part_index=0, x=3.0, y=4.0, width=20.0, height=10.0, rotation_deg=90.0)

    matrix, steps = placement_transform_for_mesh(mesh, rect)
    placed_bounds = transform_bounds(np.asarray(mesh.bounds), matrix)

    assert [s["name"] for s in steps] == [
        "normalize_to_origin",
        "packer_z_rotation",
        "post_rotation_normalize",
        "plate_translation",
    ]
    assert placed_bounds[0] == pytest.approx([3.0, 4.0, 0.0])
    assert placed_bounds[1] == pytest.approx([23.0, 14.0, 5.0])


def test_placement_transform_for_bounds_matches_mesh_helper():
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 5.0))
    mesh.apply_translation([7.0, -3.0, 11.0])
    rect = PackedRect(part_index=0, x=3.0, y=4.0, width=20.0, height=10.0, rotation_deg=90.0)

    from_mesh, mesh_steps = placement_transform_for_mesh(mesh, rect)
    from_bounds, bounds_steps = placement_transform_for_bounds(np.asarray(mesh.bounds), rect)

    assert from_bounds == pytest.approx(from_mesh)
    assert bounds_steps == mesh_steps


def test_shared_geometry_placement_matches_full_placement():
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 5.0))
    mesh.apply_translation([7.0, -3.0, 11.0])
    rect = PackedRect(part_index=0, x=3.0, y=4.0, width=20.0, height=10.0, rotation_deg=90.0)

    full_matrix, _full_steps = placement_transform_for_mesh(mesh, rect)
    normalize = translation_matrix(-np.asarray(mesh.bounds)[0])
    normalized = mesh.copy()
    normalized.apply_transform(normalize)
    shared_matrix, shared_steps = shared_geometry_placement_transform_for_mesh(normalized, rect)

    assert [s["name"] for s in shared_steps] == [
        "packer_z_rotation",
        "post_rotation_normalize",
        "plate_translation",
    ]
    assert shared_matrix @ normalize == pytest.approx(full_matrix)
    placed_bounds = transform_bounds(np.asarray(normalized.bounds), shared_matrix)
    assert placed_bounds[0] == pytest.approx([3.0, 4.0, 0.0])
    assert placed_bounds[1] == pytest.approx([23.0, 14.0, 5.0])
