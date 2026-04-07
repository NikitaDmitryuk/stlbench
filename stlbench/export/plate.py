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


def _mesh_to_xml(obj_id: int, name: str, mesh: trimesh.Trimesh) -> str:
    """Serialize one mesh object to 3MF XML (vertices in local coordinates).

    Vertices are kept in the mesh's own coordinate system (bounding-box min
    at origin).  The caller is responsible for encoding any plate-level
    placement in the ``<item transform>`` attribute.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)  # float32 = 3 sig-fig in mm
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Build vertex lines with numpy vectorisation (much faster than a Python loop)
    v_lines = np.empty(len(verts), dtype=object)
    xs = verts[:, 0]
    ys = verts[:, 1]
    zs = verts[:, 2]
    v_lines[:] = [
        f'<vertex x="{x:.3f}" y="{y:.3f}" z="{z:.3f}"/>' for x, y, z in zip(xs, ys, zs, strict=True)
    ]

    f_lines = np.empty(len(faces), dtype=object)
    v1s = faces[:, 0]
    v2s = faces[:, 1]
    v3s = faces[:, 2]
    f_lines[:] = [
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in zip(v1s, v2s, v3s, strict=True)
    ]

    vblock = "\n          ".join(v_lines)
    fblock = "\n          ".join(f_lines)

    return (
        f'    <object id="{obj_id}" name="{_xml_escape(name)}" type="model">\n'
        f"      <mesh>\n"
        f"        <vertices>\n"
        f"          {vblock}\n"
        f"        </vertices>\n"
        f"        <triangles>\n"
        f"          {fblock}\n"
        f"        </triangles>\n"
        f"      </mesh>\n"
        f"    </object>"
    )


def _item_transform(tx: float, ty: float) -> str:
    """3MF row-major 3×4 transform string: identity rotation + (tx, ty, 0) translation.

    The mesh orientation (including any 90° Z rotation) is already baked into
    the vertex coordinates by ``_place_rect_local``; the item transform only
    needs to position the object on the build plate.

    3MF layout: m00 m01 m02  m10 m11 m12  m20 m21 m22  tx ty tz
    """
    return f"1 0 0 0 1 0 0 0 1 {tx:.3f} {ty:.3f} 0"


def _write_3mf(
    path: Path,
    objects: list[tuple[str, trimesh.Trimesh]],
    item_transforms: list[tuple[float, float, bool]] | None = None,
) -> None:
    """Write a 3MF file with one independent object per (name, mesh) pair.

    Parameters
    ----------
    objects:
        ``(name, mesh)`` pairs.  Each mesh should have its bounding-box minimum
        at the origin so that the ``item_transforms`` correctly place it on the
        build plate.
    item_transforms:
        Per-object ``(tx_mm, ty_mm, rotate_90)`` placement on the build plate.
        When *None* every item gets an identity transform.

    Uses only stdlib (zipfile + xml) — no networkx required.
    """
    parts: list[str] = []
    build_items: list[str] = []

    for obj_id, (name, mesh) in enumerate(objects, start=1):
        parts.append(_mesh_to_xml(obj_id, name, mesh))
        if item_transforms is not None:
            tx, ty, _rot = item_transforms[obj_id - 1]
            t = _item_transform(tx, ty)
        else:
            t = "1 0 0 0 1 0 0 0 1 0 0 0"
        build_items.append(f'    <item objectid="{obj_id}" transform="{t}"/>')

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US"'
        ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        "  <resources>\n" + "\n".join(parts) + "\n  </resources>\n"
        "  <build>\n" + "\n".join(build_items) + "\n  </build>\n"
        "</model>"
    ).encode("utf-8")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _3MF_CONTENT_TYPES)
        zf.writestr("_rels/.rels", _3MF_RELS)
        zf.writestr("3D/3dmodel.model", model_xml)


_ROT_Z_90 = np.array(
    trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0]),
    dtype=np.float64,
)


def _place_rect_local(m: trimesh.Trimesh, r: object, rot_z_90: np.ndarray) -> trimesh.Trimesh:
    """Move mesh so its bounding-box minimum is at the origin.

    Applies 90° Z rotation when ``r.rotated`` is True, then re-centres.
    The plate-level (x, y) translation is NOT baked in — it goes into the
    ``<item transform>`` attribute instead.
    """
    m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), -float(m.bounds[0][2])])
    if getattr(r, "rotated", False):
        m.apply_transform(rot_z_90)
        m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), 0.0])
    return m


def _place_rect(m: trimesh.Trimesh, r: object, rot_z_90: np.ndarray) -> trimesh.Trimesh:
    """Translate mesh to origin, optionally rotate 90° Z, then place at rect (x, y).

    Used by the legacy STL exporter where positions must be baked into vertices.
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
    """
    objects: list[tuple[str, trimesh.Trimesh]] = []
    transforms: list[tuple[float, float, bool]] = []
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

        objects.append((node_name, m))
        transforms.append((r.x, r.y, r.rotated))
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
    _write_3mf(out_3mf, objects, item_transforms=transforms)
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
