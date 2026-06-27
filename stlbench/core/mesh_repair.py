from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from stlbench.core.mesh_cleanup import remove_small_components

REPAIR_STRATEGY_VERSION = "7"


@dataclass(frozen=True)
class RepairOptions:
    enabled: bool = False
    close_holes: bool = True
    max_hole_size_edges: int = 30
    repair_non_manifold: bool = True
    remove_small_components: bool = True


@dataclass
class RepairReport:
    source_path: str | None
    source_name: str | None
    enabled: bool
    changed: bool
    before: dict[str, Any]
    after: dict[str, Any]
    applied_filters: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repair_strategy: str = "disabled"
    timings_s: dict[str, float] = field(default_factory=dict)
    cache_key: str | None = None
    cache_hit: bool = False
    topology_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_name": self.source_name,
            "enabled": self.enabled,
            "changed": self.changed,
            "before": self.before,
            "after": self.after,
            "applied_filters": self.applied_filters,
            "warnings": self.warnings,
            "repair_strategy": self.repair_strategy,
            "timings_s": self.timings_s,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "topology_changed": self.topology_changed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RepairReport:
        return cls(
            source_path=payload.get("source_path"),
            source_name=payload.get("source_name"),
            enabled=bool(payload.get("enabled", True)),
            changed=bool(payload.get("changed", False)),
            before=dict(payload.get("before") or {}),
            after=dict(payload.get("after") or {}),
            applied_filters=list(payload.get("applied_filters") or []),
            warnings=list(payload.get("warnings") or []),
            repair_strategy=str(payload.get("repair_strategy") or "disabled"),
            timings_s={str(k): float(v) for k, v in dict(payload.get("timings_s") or {}).items()},
            cache_key=payload.get("cache_key"),
            cache_hit=bool(payload.get("cache_hit", False)),
            topology_changed=bool(payload.get("topology_changed", False)),
        )


def repair_report_step(report: RepairReport) -> dict[str, Any]:
    return {
        "name": "repair",
        "available": False,
        "params": {
            "enabled": report.enabled,
            "changed": report.changed,
            "applied_filters": report.applied_filters,
            "warnings": report.warnings,
            "before": report.before,
            "after": report.after,
            "repair_strategy": report.repair_strategy,
            "timings_s": report.timings_s,
            "cache_key": report.cache_key,
            "cache_hit": report.cache_hit,
            "topology_changed": report.topology_changed,
        },
    }


def _sorted_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.sort(np.asarray(mesh.faces, dtype=np.int64), axis=1)


def _edge_counts(mesh: trimesh.Trimesh) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.array([], dtype=np.int64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.asarray(counts, dtype=np.int64)


def _remaining_issues(health: dict[str, Any]) -> list[str]:
    checks = {
        "non_finite_bounds": not bool(health.get("finite_bounds", True)),
        "boundary_edges": int(health.get("boundary_edges") or 0) > 0,
        "non_manifold_edges": int(health.get("non_manifold_edges") or 0) > 0,
        "broken_faces": int(health.get("broken_faces") or 0) > 0,
        "duplicate_faces": int(health.get("duplicate_faces") or 0) > 0,
        "degenerate_faces": int(health.get("degenerate_faces") or 0) > 0,
        "unreferenced_vertices": int(health.get("unreferenced_vertices") or 0) > 0,
        "self_intersections": int(health.get("self_intersections") or 0) > 0,
        "non_manifold_vertices": int(health.get("non_manifold_vertices") or 0) > 0,
    }
    if not bool(health.get("watertight")):
        checks["not_watertight"] = True
    return [name for name, present in checks.items() if present]


def mesh_health(mesh: trimesh.Trimesh) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    finite_bounds = bounds.shape == (2, 3) and bool(np.isfinite(bounds).all())
    bounds_payload = [[float(v) for v in row] for row in bounds.tolist()] if finite_bounds else None
    try:
        if len(mesh.faces) == 0:
            components_count = 0
        elif len(mesh.face_adjacency) == 0:
            components_count = int(len(mesh.faces))
        else:
            components_count = int(len(trimesh.graph.connected_components(mesh.face_adjacency)))
    except (ValueError, IndexError):
        components_count = 0
    try:
        broken_faces_count = int(len(trimesh.repair.broken_faces(mesh)))
    except (ValueError, IndexError):
        broken_faces_count = 0

    sorted_faces = _sorted_faces(mesh)
    if len(sorted_faces) == 0:
        duplicate_faces = 0
    else:
        _unique_faces, face_counts = np.unique(sorted_faces, axis=0, return_counts=True)
        duplicate_faces = int(np.sum(np.maximum(face_counts - 1, 0)))
    try:
        degenerate_faces = int(np.sum(np.asarray(mesh.area_faces) <= 1e-12))
    except (ValueError, IndexError):
        degenerate_faces = 0
    if len(mesh.faces) == 0:
        unreferenced_vertices = int(len(mesh.vertices))
    else:
        referenced = np.unique(np.asarray(mesh.faces).reshape(-1))
        unreferenced_vertices = int(max(0, len(mesh.vertices) - len(referenced)))
    counts = _edge_counts(mesh)

    health: dict[str, Any] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_mm": bounds_payload,
        "finite_bounds": finite_bounds,
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "components": components_count,
        "broken_faces": broken_faces_count,
        "boundary_edges": int(np.sum(counts == 1)),
        "non_manifold_edges": int(np.sum(counts > 2)),
        "duplicate_faces": duplicate_faces,
        "degenerate_faces": degenerate_faces,
        "unreferenced_vertices": unreferenced_vertices,
        "self_intersections": None,
        "non_manifold_vertices": None,
    }
    issues = _remaining_issues(health)
    health["slicer_safe"] = not issues
    health["remaining_issues"] = issues
    return health


def _mesh_signature(mesh: trimesh.Trimesh) -> tuple[int, int, bytes]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return len(mesh.vertices), len(mesh.faces), np.round(bounds, 9).tobytes()


def _version_or_unknown(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _options_payload(options: RepairOptions) -> dict[str, Any]:
    return {
        "enabled": options.enabled,
        "close_holes": options.close_holes,
        "max_hole_size_edges": options.max_hole_size_edges,
        "repair_non_manifold": options.repair_non_manifold,
        "remove_small_components": options.remove_small_components,
        "strategy_version": REPAIR_STRATEGY_VERSION,
        "versions": {
            "stlbench": _version_or_unknown("stlbench"),
            "trimesh": getattr(trimesh, "__version__", "unknown"),
            "pymeshlab": _version_or_unknown("pymeshlab"),
        },
    }


def source_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repair_cache_key(source_path: Path, options: RepairOptions) -> str:
    h = hashlib.sha256()
    h.update(source_sha256(source_path).encode("ascii"))
    h.update(json.dumps(_options_payload(options), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _run_pymeshlab_filter(
    meshset: Any,
    name: str,
    kwargs: dict[str, Any],
    applied: list[str],
    warnings: list[str],
    timings: dict[str, float],
) -> None:
    start = time.perf_counter()
    try:
        getattr(meshset, name)(**kwargs)
        applied.append(name)
    except Exception as exc:  # pragma: no cover - PyMeshLab errors are data-dependent
        warnings.append(f"{name}: {exc}")
    finally:
        timings[name] = time.perf_counter() - start


def _full_pymeshlab_repair(
    mesh: trimesh.Trimesh,
    options: RepairOptions,
) -> tuple[trimesh.Trimesh, list[str], list[str], dict[str, float]]:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ),
        "mesh",
    )
    applied: list[str] = []
    warnings: list[str] = []
    timings: dict[str, float] = {}
    filters: list[tuple[str, dict[str, Any]]] = [
        ("meshing_remove_duplicate_vertices", {}),
        ("meshing_remove_duplicate_faces", {}),
        ("meshing_remove_null_faces", {}),
        ("meshing_remove_unreferenced_vertices", {}),
        ("meshing_remove_folded_faces", {}),
        ("meshing_remove_t_vertices", {}),
        ("meshing_snap_mismatched_borders", {}),
    ]
    if options.repair_non_manifold:
        filters.extend(
            [
                ("meshing_repair_non_manifold_vertices", {}),
                ("meshing_repair_non_manifold_edges", {}),
            ]
        )
    if options.close_holes:
        filters.append(
            (
                "meshing_close_holes",
                {
                    "maxholesize": int(options.max_hole_size_edges),
                    "newfaceselected": False,
                    "selfintersection": True,
                },
            )
        )
    filters.extend(
        [
            ("meshing_remove_unreferenced_vertices", {}),
            ("meshing_re_orient_faces_coherently", {}),
        ]
    )
    if options.repair_non_manifold:
        filters.extend(
            [
                ("meshing_repair_non_manifold_edges", {}),
                ("meshing_repair_non_manifold_vertices", {}),
            ]
        )

    for name, kwargs in filters:
        _run_pymeshlab_filter(ms, name, kwargs, applied, warnings, timings)

    out = ms.current_mesh()
    repaired = trimesh.Trimesh(
        vertices=np.asarray(out.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(out.face_matrix(), dtype=np.int64),
        process=False,
    )

    for name, func in [
        ("trimesh.repair.fix_winding", lambda: trimesh.repair.fix_winding(repaired)),
        ("trimesh.repair.fix_normals", lambda: trimesh.repair.fix_normals(repaired)),
        (
            "trimesh.repair.fix_inversion",
            lambda: trimesh.repair.fix_inversion(repaired, multibody=True),
        ),
    ]:
        start = time.perf_counter()
        try:
            func()
            applied.append(name)
        except Exception as exc:  # pragma: no cover - data-dependent repair failures
            warnings.append(f"{name}: {exc}")
        finally:
            timings[name] = time.perf_counter() - start
    return repaired, applied, warnings, timings


def repair_mesh(
    mesh: trimesh.Trimesh,
    options: RepairOptions | None = None,
    *,
    source_path: Path | str | None = None,
    source_name: str | None = None,
) -> tuple[trimesh.Trimesh, RepairReport]:
    opts = options or RepairOptions()
    before = mesh_health(mesh)
    if not opts.enabled:
        return mesh, RepairReport(
            source_path=None if source_path is None else str(source_path),
            source_name=source_name,
            enabled=False,
            changed=False,
            before=before,
            after=before,
            repair_strategy="disabled",
        )

    original_sig = _mesh_signature(mesh)
    start = time.perf_counter()
    repaired, applied, warnings, timings = _full_pymeshlab_repair(mesh, opts)
    timings["full_total"] = time.perf_counter() - start

    if opts.remove_small_components:
        start = time.perf_counter()
        repaired, removed = remove_small_components(repaired)
        timings["remove_small_components"] = time.perf_counter() - start
        if removed:
            applied.append("remove_small_components")
            warnings.append(f"removed {removed} small disconnected component(s)")

    after = mesh_health(repaired)
    remaining = list(after.get("remaining_issues") or [])
    if remaining:
        warnings.append(f"remaining_issues: {', '.join(remaining)}")
    changed = _mesh_signature(repaired) != original_sig
    return repaired, RepairReport(
        source_path=None if source_path is None else str(source_path),
        source_name=source_name,
        enabled=True,
        changed=changed,
        before=before,
        after=after,
        applied_filters=applied,
        warnings=warnings,
        repair_strategy="full",
        timings_s=timings,
    )


def load_repair_cache(
    cache_dir: Path,
    key: str,
    *,
    source_path: Path | str | None = None,
    source_name: str | None = None,
) -> tuple[trimesh.Trimesh, RepairReport] | None:
    entry_dir = cache_dir / key
    mesh_path = entry_dir / "mesh.ply"
    report_path = entry_dir / "report.json"
    if not mesh_path.is_file() or not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        report = RepairReport.from_dict(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    report.source_path = None if source_path is None else str(source_path)
    report.source_name = source_name
    report.repair_strategy = "cached"
    report.cache_hit = True
    report.cache_key = key
    return mesh, report


def write_repair_cache(
    cache_dir: Path,
    key: str,
    mesh: trimesh.Trimesh,
    report: RepairReport,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_dir = cache_dir / key
    if final_dir.exists():
        return
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_dir))
    try:
        mesh.export(tmp_dir / "mesh.ply")
        payload = report.to_dict()
        payload["cache_key"] = key
        (tmp_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            tmp_dir.replace(final_dir)
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def write_repair_report(
    path: Path,
    *,
    command: str,
    reports: list[RepairReport],
) -> None:
    payload = {
        "schema_version": 1,
        "command": command,
        "created_at": datetime.now(UTC).isoformat(),
        "units": "mm",
        "parts": [report.to_dict() for report in reports],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
