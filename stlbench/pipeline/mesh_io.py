from __future__ import annotations

from pathlib import Path
from typing import cast

import trimesh

# Formats loaded directly by trimesh (no extra runtime dependency).
# DAE/ZAE use trimesh's built-in Collada loader, which requires pycollada.
# FBX is handled separately via pymeshlab (see _load_mesh_pymeshlab).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".3mf", ".dae", ".zae", ".fbx"}
)


def collect_mesh_paths(input_dir: Path, recursive: bool) -> list[Path]:
    """Return sorted mesh files under *input_dir* with a supported extension.

    Extension matching is case-insensitive so ``.STL``, ``.Obj``, etc. are
    picked up on case-sensitive file-systems (Linux).
    """
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _load_mesh_pymeshlab(path: Path) -> trimesh.Trimesh:
    """Load *path* via pymeshlab and return a Trimesh.

    Used for formats trimesh cannot read natively (currently FBX).
    pymeshlab bundles MeshLab/assimp so no system library is required.
    """
    import pymeshlab  # noqa: PLC0415

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    m = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=m.vertex_matrix(),
        faces=m.face_matrix(),
        process=False,
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh, _ = load_mesh_with_info(path)
    return mesh


def load_mesh_with_info(path: Path) -> tuple[trimesh.Trimesh, bool]:
    """Load *path* and return ``(mesh, has_multiple_surfaces)``.

    *has_multiple_surfaces* is ``True`` when the file contained more than one
    geometry object (e.g. a multi-body or multi-part model).  The geometries
    are always merged into a single ``Trimesh`` for downstream use.
    """
    if path.suffix.lower() == ".fbx":
        return _load_mesh_pymeshlab(path), False

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geom = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geom:
            raise ValueError(f"No mesh geometry in scene: {path}")
        has_multiple = len(geom) > 1
        mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(geom)) if has_multiple else geom[0]
        return cast(trimesh.Trimesh, mesh), has_multiple
    elif isinstance(loaded, trimesh.Trimesh):
        return cast(trimesh.Trimesh, loaded), False
    else:
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)}")
