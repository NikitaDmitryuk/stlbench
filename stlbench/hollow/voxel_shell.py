"""
Voxel hollow shell (trimesh + scipy).

For MeshLib-based workflows see `meshlib_note.md`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import trimesh


def hollow_mesh_voxel_shell(
    mesh: trimesh.Trimesh,
    wall_thickness_mm: float,
    voxel_mm: float,
) -> trimesh.Trimesh:
    if wall_thickness_mm <= 0 or voxel_mm <= 0:
        raise ValueError("wall_thickness_mm and voxel_mm must be positive.")

    try:
        from scipy import ndimage
    except ImportError as e:
        raise ImportError(
            "scipy is required for hollow (included with stlbench); "
            "reinstall the package or run: pip install scipy"
        ) from e

    pitch = float(voxel_mm)
    vg = mesh.voxelized(pitch=pitch)
    m = np.asarray(vg.filled, dtype=bool)
    if not np.any(m):
        raise ValueError("Empty voxel grid: increase voxel_mm or check the mesh.")

    filled = ndimage.binary_fill_holes(m)
    iters = max(1, int(round(wall_thickness_mm / pitch)))
    eroded = ndimage.binary_erosion(filled, iterations=iters)
    shell = filled & (~eroded)
    if not np.any(shell):
        raise ValueError("Shell is empty: walls too thick or voxel too small.")

    shell_vg = trimesh.voxel.VoxelGrid(shell, transform=vg.transform)
    out = shell_vg.marching_cubes
    if out is None or len(out.vertices) == 0:
        raise RuntimeError("marching_cubes returned no geometry.")
    return cast(trimesh.Trimesh, out)
