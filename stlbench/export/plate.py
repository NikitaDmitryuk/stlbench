from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from stlbench.packing.rectpack_plate import PackedPlate

_ROT_Z_90 = np.array(
    trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0]),
    dtype=np.float64,
)


def _place_rect_local(m: trimesh.Trimesh, r: object, rot_z_90: np.ndarray) -> trimesh.Trimesh:
    """Move mesh so its bounding-box minimum is at the origin.

    Applies 90° Z rotation when ``r.rotated`` is True, then re-centres.
    The plate-level (x, y) translation is NOT baked in — it goes into the
    scene item transform instead.
    """
    m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), -float(m.bounds[0][2])])
    if getattr(r, "rotated", False):
        m.apply_transform(rot_z_90)
        m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), 0.0])
    return m


def _place_rect(m: trimesh.Trimesh, r: object, rot_z_90: np.ndarray) -> trimesh.Trimesh:
    """Translate mesh to origin, optionally rotate 90° Z, then place at rect (x, y).

    Used by the STL exporter where positions must be baked into vertices.
    """
    m = _place_rect_local(m, r, rot_z_90)
    m.apply_translation([getattr(r, "x", 0.0), getattr(r, "y", 0.0), 0.0])
    return m


def export_plate_3mf(
    meshes: list[trimesh.Trimesh],
    plate: PackedPlate,
    out_3mf: Path,
    names: list[str] | None = None,
    out_manifest: Path | None = None,
) -> None:
    """Export a packed plate as a 3MF scene with one independent object per part.

    Each mesh becomes a separate object in the 3MF file so that slicers
    (Elegoo SatelLite, Chitubox, Lychee, PrusaSlicer) can add supports and
    adjust orientation per-part after import.

    Uses trimesh.Scene for compliant 3MF generation (correct namespaces,
    p:UUID attributes, production extension).
    """
    scene = trimesh.Scene()
    manifest_parts: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for r in plate.rects:
        if r.part_index < 0 or r.part_index >= len(meshes):
            continue
        m = _place_rect_local(meshes[r.part_index].copy(), r, _ROT_Z_90)

        base = (
            names[r.part_index]
            if names and r.part_index < len(names)
            else f"part_{r.part_index:02d}"
        )
        count = seen.get(base, 0)
        node_name = base if count == 0 else f"{base}_{count:02d}"
        seen[base] = count + 1

        # Mesh bottom is already at z=0; only XY translation needed for placement.
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = r.x
        transform[1, 3] = r.y

        scene.add_geometry(m, geom_name=node_name, transform=transform)

        manifest_parts.append(
            {
                "index": r.part_index,
                "name": node_name,
                "x_mm": r.x,
                "y_mm": r.y,
                "footprint_w_mm": r.width,
                "footprint_h_mm": r.height,
                "rotated_90": r.rotated,
            }
        )

    if not scene.geometry:
        raise ValueError("No meshes to export for this plate.")

    out_3mf.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_3mf))

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
        m = _place_rect(meshes[r.part_index].copy(), r, _ROT_Z_90)
        placed.append(m)
        manifest_parts.append(
            {
                "index": r.part_index,
                "x_mm": r.x,
                "y_mm": r.y,
                "footprint_w_mm": r.width,
                "footprint_h_mm": r.height,
                "rotated_90": r.rotated,
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
