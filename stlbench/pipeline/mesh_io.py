from __future__ import annotations

from pathlib import Path
from typing import cast

import trimesh


def collect_stl_paths(input_dir: Path, recursive: bool) -> list[Path]:
    paths = sorted(input_dir.rglob("*.stl")) if recursive else sorted(input_dir.glob("*.stl"))
    return [p for p in paths if p.is_file()]


def load_mesh(path: Path) -> trimesh.Trimesh:
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
