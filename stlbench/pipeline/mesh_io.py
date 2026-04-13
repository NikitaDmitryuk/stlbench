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
    if path.suffix.lower() == ".fbx":
        return _load_mesh_pymeshlab(path)

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geom = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geom:
            raise ValueError(f"No mesh geometry in scene: {path}")
        mesh = trimesh.util.concatenate(geom)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)}")
    return cast(trimesh.Trimesh, mesh)
