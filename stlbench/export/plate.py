from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import trimesh

from stlbench.packing.rectpack_plate import PackedPlate


class MeshLoader(Protocol):
    def __call__(self, part_index: int) -> trimesh.Trimesh: ...


def clear_mesh_cache(mesh: trimesh.Trimesh) -> None:
    """Drop trimesh lazy caches when a large mesh is about to be released."""
    cache = getattr(mesh, "_cache", None)
    clear = getattr(cache, "clear", None)
    if callable(clear):
        clear()


def _place_rect_local(m: trimesh.Trimesh, r: object) -> trimesh.Trimesh:
    """Move mesh so its bounding-box minimum is at the origin.

    Applies a Z rotation of ``r.rotation_deg`` degrees (if non-zero) then
    re-centres.  The plate-level (x, y) translation is NOT baked in — it goes
    into the scene item transform instead.
    """
    m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), -float(m.bounds[0][2])])
    angle_deg: float = float(getattr(r, "rotation_deg", 0.0))
    if abs(angle_deg) > 1e-9:
        rot = np.array(
            trimesh.transformations.rotation_matrix(np.radians(angle_deg), [0.0, 0.0, 1.0]),
            dtype=np.float64,
        )
        m.apply_transform(rot)
        m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), 0.0])
    return m


def _validate_rect_local_bounds(
    m: trimesh.Trimesh,
    r: object,
    name: str,
    *,
    epsilon_mm: float = 1e-3,
) -> tuple[float, float]:
    bounds = np.asarray(m.bounds, dtype=np.float64)
    actual_w = float(bounds[1, 0] - bounds[0, 0])
    actual_h = float(bounds[1, 1] - bounds[0, 1])
    rect_w = float(getattr(r, "width", actual_w))
    rect_h = float(getattr(r, "height", actual_h))
    if actual_w > rect_w + epsilon_mm or actual_h > rect_h + epsilon_mm:
        raise ValueError(
            f"Packed footprint for {name} is smaller than exported mesh bounds: "
            f"rect={rect_w:.4f}×{rect_h:.4f} mm, "
            f"mesh={actual_w:.4f}×{actual_h:.4f} mm."
        )
    return actual_w, actual_h


def _place_rect(m: trimesh.Trimesh, r: object) -> trimesh.Trimesh:
    """Translate mesh to origin, rotate by rotation_deg around Z, then place at rect (x, y).

    Used by the STL exporter where positions must be baked into vertices.
    """
    m = _place_rect_local(m, r)
    m.apply_translation([getattr(r, "x", 0.0), getattr(r, "y", 0.0), 0.0])
    return m


def export_plate_3mf(
    meshes: list[trimesh.Trimesh],
    plate: PackedPlate,
    out_3mf: Path,
    names: list[str] | None = None,
    out_manifest: Path | None = None,
) -> None:
    def _load_mesh(part_index: int) -> trimesh.Trimesh:
        return meshes[part_index]

    export_plate_3mf_lazy(
        _load_mesh,
        plate,
        out_3mf,
        names=names,
        out_manifest=out_manifest,
        copy_mesh=True,
    )


def export_plate_3mf_lazy(
    mesh_loader: MeshLoader,
    plate: PackedPlate,
    out_3mf: Path,
    names: list[str] | None = None,
    out_manifest: Path | None = None,
    *,
    copy_mesh: bool = False,
) -> None:
    """Export a packed plate as a 3MF with one independent object per part.

    Each mesh becomes a separate object in the 3MF file so that slicers
    (Elegoo SatelLite, Chitubox, Lychee, PrusaSlicer) can add supports and
    adjust orientation per-part after import.
    """
    out_meshes: list[trimesh.Trimesh] = []
    out_transforms: list[np.ndarray] = []
    out_names: list[str] = []
    manifest_parts: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for r in plate.rects:
        if r.part_index < 0:
            continue
        try:
            loaded = mesh_loader(r.part_index)
        except IndexError:
            continue
        m = loaded.copy() if copy_mesh else loaded
        m = _place_rect_local(m, r)

        base = (
            names[r.part_index]
            if names and r.part_index < len(names)
            else f"part_{r.part_index:02d}"
        )
        count = seen.get(base, 0)
        node_name = base if count == 0 else f"{base}_{count:02d}"
        seen[base] = count + 1
        _validate_rect_local_bounds(m, r, node_name)

        # Mesh bottom is already at z=0; only XY translation needed for placement.
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = r.x
        transform[1, 3] = r.y

        out_meshes.append(m)
        out_transforms.append(transform)
        out_names.append(node_name)

        manifest_parts.append(
            {
                "index": r.part_index,
                "name": node_name,
                "x_mm": r.x,
                "y_mm": r.y,
                "footprint_w_mm": r.width,
                "footprint_h_mm": r.height,
                "rotation_deg": r.rotation_deg,
            }
        )

    if not out_meshes:
        raise ValueError("No meshes to export for this plate.")

    scene = trimesh.Scene()
    for m, tf, name in zip(out_meshes, out_transforms, out_names, strict=True):
        scene.add_geometry(m, geom_name=name, transform=tf)
    out_3mf.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_3mf))
    for m in out_meshes:
        clear_mesh_cache(m)

    if out_manifest is not None:
        payload = {"plate_index": plate.index, "parts": manifest_parts}
        out_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_plate_stl(
    meshes: list[trimesh.Trimesh],
    plate: PackedPlate,
    out_stl: Path,
    out_manifest: Path | None = None,
) -> None:
    """Export a packed plate as a single combined STL (legacy; parts are merged)."""
    placed: list[trimesh.Trimesh] = []
    manifest_parts: list[dict[str, Any]] = []
    for r in plate.rects:
        if r.part_index < 0 or r.part_index >= len(meshes):
            continue
        m = _place_rect_local(meshes[r.part_index].copy(), r)
        _validate_rect_local_bounds(m, r, f"part_{r.part_index:02d}")
        m.apply_translation([getattr(r, "x", 0.0), getattr(r, "y", 0.0), 0.0])
        placed.append(m)
        manifest_parts.append(
            {
                "index": r.part_index,
                "x_mm": r.x,
                "y_mm": r.y,
                "footprint_w_mm": r.width,
                "footprint_h_mm": r.height,
                "rotation_deg": r.rotation_deg,
            }
        )
    if not placed:
        raise ValueError("No meshes to export for this plate.")
    combined = trimesh.util.concatenate(placed)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    combined.export(out_stl)
    if out_manifest is not None:
        payload = {"plate_index": plate.index, "parts": manifest_parts}
        out_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mesh_footprint_xy(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    b = np.asarray(mesh.bounds)
    d = b[1] - b[0]
    return float(d[0]), float(d[1]), float(d[2])
