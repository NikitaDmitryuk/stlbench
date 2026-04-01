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


def export_plate_stl(
    meshes: list[trimesh.Trimesh],
    plate: PackedPlate,
    out_stl: Path,
    out_manifest: Path | None = None,
) -> None:
    placed: list[trimesh.Trimesh] = []
    manifest_parts: list[dict[str, Any]] = []
    for r in plate.rects:
        if r.part_index < 0 or r.part_index >= len(meshes):
            continue
        m = meshes[r.part_index].copy()

        m.apply_translation(
            [-float(m.bounds[0][0]), -float(m.bounds[0][1]), -float(m.bounds[0][2])]
        )

        if r.rotated:
            m.apply_transform(_ROT_Z_90)
            m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), 0.0])

        m.apply_translation([r.x, r.y, 0.0])
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
