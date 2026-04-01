"""
Полая оболочка через воксели (trimesh + scipy).

Open3D не обязателен; см. [hollow] extra (scipy). Про MeshLib — `meshlib_note.md`.
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
        raise ImportError("Для hollow: pip install 'stlbench[hollow]' (scipy).") from e

    pitch = float(voxel_mm)
    vg = mesh.voxelized(pitch=pitch)
    m = np.asarray(vg.filled, dtype=bool)
    if not np.any(m):
        raise ValueError("Пустая воксельная сетка: увеличьте voxel_mm или проверьте меш.")

    filled = ndimage.binary_fill_holes(m)
    iters = max(1, int(round(wall_thickness_mm / pitch)))
    eroded = ndimage.binary_erosion(filled, iterations=iters)
    shell = filled & (~eroded)
    if not np.any(shell):
        raise ValueError("Оболочка пуста: слишком толстые стенки или слишком мелкий воксель.")

    shell_vg = trimesh.voxel.VoxelGrid(shell, transform=vg.transform)
    out = shell_vg.marching_cubes
    if out is None or len(out.vertices) == 0:
        raise RuntimeError("marching_cubes не вернул геометрию.")
    return cast(trimesh.Trimesh, out)
