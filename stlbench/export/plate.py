from __future__ import annotations

import json
import zipfile
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


def _write_3mf_compat(
    meshes: list[trimesh.Trimesh],
    transforms: list[np.ndarray],
    names: list[str],
    out_path: Path,
) -> None:
    """Write a minimal 3MF using only the core namespace.

    Trimesh ≥ 4.x adds production-extension attributes (p:UUID, partnumber)
    and re-declares namespaces on every <item> element.  Elegoo SatelLite and
    some other resin slicers reject those files.  This writer emits only the
    core 3MF 2015/02 namespace, matching the format those slicers expect.

    Each mesh must already be positioned at its local origin (Z-min = 0,
    XY-min = 0).  The plate-level XY translation is carried in *transforms*
    and written as the ``transform`` attribute of each ``<item>`` element.
    """
    _CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    _CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
    _REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
    _MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Types xmlns="{_CT_NS}">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        "</Types>\n"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Relationships xmlns="{_REL_NS}">\n'
        f'  <Relationship Type="{_MODEL_REL}" Target="/3D/3dmodel.model" Id="rel0"/>\n'
        "</Relationships>\n"
    )

    def _attr(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(out_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)

        with zf.open("3D/3dmodel.model", "w") as f:

            def w(s: str) -> None:
                f.write(s.encode("utf-8"))

            w('<?xml version="1.0" encoding="UTF-8"?>\n')
            w(f'<model unit="millimeter" xml:lang="en-US" xmlns="{_CORE_NS}">\n')
            w("  <resources>\n")

            for obj_id, (mesh, name) in enumerate(zip(meshes, names, strict=True), start=1):
                w(f'    <object id="{obj_id}" name="{_attr(name)}" type="model">\n')
                w("      <mesh>\n")
                w("        <vertices>\n")
                for v in mesh.vertices:
                    w(f'          <vertex x="{v[0]:.3f}" y="{v[1]:.3f}" z="{v[2]:.3f}"/>\n')
                w("        </vertices>\n")
                w("        <triangles>\n")
                for face in mesh.faces:
                    w(f'          <triangle v1="{face[0]}" v2="{face[1]}" v3="{face[2]}"/>\n')
                w("        </triangles>\n")
                w("      </mesh>\n")
                w("    </object>\n")

            w("  </resources>\n")
            w("  <build>\n")
            for obj_id, tf in enumerate(transforms, start=1):
                tx, ty = tf[0, 3], tf[1, 3]
                w(
                    f'    <item objectid="{obj_id}"'
                    f' transform="1 0 0 0 1 0 0 0 1 {tx:.3f} {ty:.3f} 0"/>\n'
                )
            w("  </build>\n")
            w("</model>\n")


def export_plate_3mf(
    meshes: list[trimesh.Trimesh],
    plate: PackedPlate,
    out_3mf: Path,
    names: list[str] | None = None,
    out_manifest: Path | None = None,
) -> None:
    """Export a packed plate as a 3MF with one independent object per part.

    Each mesh becomes a separate object in the 3MF file so that slicers
    (Elegoo SatelLite, Chitubox, Lychee, PrusaSlicer) can add supports and
    adjust orientation per-part after import.

    Uses a minimal core-only 3MF writer instead of trimesh.Scene.export()
    to avoid production-extension attributes (p:UUID, partnumber) that
    cause Elegoo SatelLite to refuse the file.
    """
    out_meshes: list[trimesh.Trimesh] = []
    out_transforms: list[np.ndarray] = []
    out_names: list[str] = []
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
                "rotated_90": r.rotated,
            }
        )

    if not out_meshes:
        raise ValueError("No meshes to export for this plate.")

    _write_3mf_compat(out_meshes, out_transforms, out_names, out_3mf)

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
