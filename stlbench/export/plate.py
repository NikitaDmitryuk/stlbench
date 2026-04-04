from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import trimesh

from stlbench.packing.rectpack_plate import PackedPlate

# ── Minimal 3MF writer (no networkx / external 3MF libs needed) ─────────────
_3MF_CONTENT_TYPES = """\
<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_3MF_RELS = """\
<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" \
Target="/3D/3dmodel.model" Id="rel0"/>
</Relationships>"""


def _write_3mf(path: Path, objects: list[tuple[str, trimesh.Trimesh]]) -> None:
    """Write a 3MF file with one independent object per (name, mesh) pair.

    Uses only stdlib (zipfile + xml) — no networkx required.
    """
    chunks: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US"'
        ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
        "  <resources>",
    ]
    build_items: list[str] = []

    for obj_id, (name, mesh) in enumerate(objects, start=1):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        chunks.append(f'    <object id="{obj_id}" name="{_xml_escape(name)}" type="model">')
        chunks.append("      <mesh>")
        chunks.append("        <vertices>")
        for v in verts:
            chunks.append(f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
        chunks.append("        </vertices>")
        chunks.append("        <triangles>")
        for f in faces:
            chunks.append(f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>')
        chunks.append("        </triangles>")
        chunks.append("      </mesh>")
        chunks.append("    </object>")
        build_items.append(f'    <item objectid="{obj_id}"/>')

    chunks.append("  </resources>")
    chunks.append("  <build>")
    chunks.extend(build_items)
    chunks.append("  </build>")
    chunks.append("</model>")

    model_xml = "\n".join(chunks).encode("utf-8")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _3MF_CONTENT_TYPES)
        zf.writestr("_rels/.rels", _3MF_RELS)
        zf.writestr("3D/3dmodel.model", model_xml)


_ROT_Z_90 = np.array(
    trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0]),
    dtype=np.float64,
)


def _place_rect(m: trimesh.Trimesh, r: object, rot_z_90: np.ndarray) -> trimesh.Trimesh:
    """Translate mesh to origin, optionally rotate 90° Z, then place at rect (x, y)."""
    m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), -float(m.bounds[0][2])])
    if getattr(r, "rotated", False):
        m.apply_transform(rot_z_90)
        m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), 0.0])
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
    """
    objects: list[tuple[str, trimesh.Trimesh]] = []
    manifest_parts: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for r in plate.rects:
        if r.part_index < 0 or r.part_index >= len(meshes):
            continue
        m = _place_rect(meshes[r.part_index].copy(), r, _ROT_Z_90)

        base = (
            names[r.part_index]
            if names and r.part_index < len(names)
            else f"part_{r.part_index:02d}"
        )
        count = seen.get(base, 0)
        node_name = base if count == 0 else f"{base}_{count:02d}"
        seen[base] = count + 1

        objects.append((node_name, m))
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

    if not objects:
        raise ValueError("No meshes to export for this plate.")

    out_3mf.parent.mkdir(parents=True, exist_ok=True)
    _write_3mf(out_3mf, objects)
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
