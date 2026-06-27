from __future__ import annotations

import json

import numpy as np
import trimesh

from stlbench.core.mesh_repair import (
    RepairOptions,
    load_repair_cache,
    mesh_health,
    repair_cache_key,
    repair_mesh,
    write_repair_cache,
    write_repair_report,
)


def test_repair_disabled_preserves_mesh_and_reports_disabled():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    repaired, report = repair_mesh(mesh, RepairOptions(enabled=False))

    assert repaired is mesh
    assert report.enabled is False
    assert report.changed is False
    assert report.before == report.after
    assert report.repair_strategy == "disabled"
    assert report.applied_filters == []


def test_full_repair_reports_pymeshlab_filters_for_duplicate_geometry():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.faces = np.vstack([mesh.faces.copy(), mesh.faces[:1]])

    repaired, report = repair_mesh(
        mesh,
        RepairOptions(enabled=True, close_holes=False),
        source_path="box.stl",
        source_name="box.stl",
    )

    assert report.enabled is True
    assert report.repair_strategy == "full"
    assert report.before["faces"] > report.after["faces"]
    assert len(repaired.faces) == report.after["faces"]
    assert "meshing_remove_duplicate_vertices" in report.applied_filters
    assert "meshing_remove_duplicate_faces" in report.applied_filters
    assert "meshing_remove_folded_faces" in report.applied_filters
    assert "meshing_remove_t_vertices" in report.applied_filters
    assert "meshing_snap_mismatched_borders" in report.applied_filters
    assert "trimesh.repair.fix_normals" in report.applied_filters
    assert report.after["slicer_safe"] is True


def test_mesh_health_flags_boundary_edges_as_not_slicer_safe():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    mesh.remove_unreferenced_vertices()

    health = mesh_health(mesh)

    assert health["watertight"] is False
    assert health["boundary_edges"] > 0
    assert health["slicer_safe"] is False
    assert "boundary_edges" in health["remaining_issues"]


def test_full_repair_reports_remaining_issues_when_filters_cannot_close_holes():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    mesh.remove_unreferenced_vertices()

    _repaired, report = repair_mesh(mesh, RepairOptions(enabled=True, close_holes=False))

    assert report.repair_strategy == "full"
    assert report.after["slicer_safe"] is False
    assert report.after["remaining_issues"]
    assert any("remaining_issues:" in warning for warning in report.warnings)


def test_write_repair_report_schema(tmp_path):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    _repaired, report = repair_mesh(mesh, RepairOptions(enabled=False), source_name="box")
    out = tmp_path / "repair_report.json"

    write_repair_report(out, command="unit", reports=[report])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["command"] == "unit"
    assert payload["units"] == "mm"
    assert payload["parts"][0]["source_name"] == "box"


def test_repair_cache_key_changes_with_options(tmp_path):
    source = tmp_path / "box.stl"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(source)

    key_enabled = repair_cache_key(source, RepairOptions(enabled=True))
    key_no_holes = repair_cache_key(source, RepairOptions(enabled=True, close_holes=False))

    assert key_enabled != key_no_holes


def test_repair_cache_hit_and_incomplete_entry_miss(tmp_path):
    source = tmp_path / "box.stl"
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.export(source)
    repaired, report = repair_mesh(mesh, RepairOptions(enabled=True), source_name="box")
    cache_dir = tmp_path / "cache"
    key = repair_cache_key(source, RepairOptions(enabled=True))

    assert load_repair_cache(cache_dir, key, source_name="box") is None
    (cache_dir / "incomplete").mkdir(parents=True)
    (cache_dir / "incomplete" / "report.json").write_text("{}", encoding="utf-8")
    assert load_repair_cache(cache_dir, "incomplete", source_name="box") is None

    write_repair_cache(cache_dir, key, repaired, report)
    cached = load_repair_cache(cache_dir, key, source_path=source, source_name="box")

    assert cached is not None
    cached_mesh, cached_report = cached
    assert len(cached_mesh.faces) == len(repaired.faces)
    assert cached_report.cache_hit is True
    assert cached_report.repair_strategy == "cached"
