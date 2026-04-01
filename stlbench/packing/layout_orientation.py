from __future__ import annotations

import numpy as np
import trimesh

from stlbench.core.orientation import (
    _random_rotation_matrix,
    aabb_extents_after_rotation,
    mesh_vertices_for_orientation,
)
from stlbench.packing.rectpack_plate import footprint_fits_bin_mm


def _axis_perm_4x4_list() -> list[np.ndarray]:
    """6 жёстких перестановок осей (какой размер AABB идёт в Z принтера, какие в X/Y)."""
    i4 = np.eye(4, dtype=np.float64)
    x_up = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    y_up = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    rz = np.asarray(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    out: list[np.ndarray] = []
    for a in (i4, x_up, y_up):
        out.append(a.copy())
        out.append(rz @ a)
    return out


def _perm_3x3_list() -> list[np.ndarray]:
    return [t[:3, :3].copy() for t in _axis_perm_4x4_list()]


def _rotation_to_4x4(r3: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.asarray(r3, dtype=np.float64)
    return t


def select_layout_transform(
    mesh: trimesh.Trimesh,
    bed_x: float,
    bed_y: float,
    pz: float,
    gap_mm: float,
    *,
    random_samples: int = 4096,
    seed: int = 0,
) -> tuple[bool, np.ndarray, float, float]:
    """
    Ищет ориентацию для печати: после поворота размеры AABB по осям принтера (X,Y,Z)
    должны укладываться в стол и Pz — с **перестановкой**, какая ось какому размеру принтера
    соответствует (согласовано с sorted-fit при расчёте масштаба).

    Для каждого базового поворота SO(3) (тождество + ``random_samples`` выборок, тот же RNG,
    что в конфиге) перебираются 6 перестановок осей ``P @ R``.

    Возвращает (успех, T 4×4, ширина по X стола, глубина по Y стола) — в финальных координатах
    принтера Z вверх, слои по Z.
    """
    verts = mesh_vertices_for_orientation(mesh)
    rng = np.random.default_rng(seed)
    perms = _perm_3x3_list()

    best: tuple[float, np.ndarray, float, float] | None = None

    bases: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for _ in range(max(0, random_samples)):
        bases.append(_random_rotation_matrix(rng))

    for r in bases:
        for p in perms:
            r_tot = p @ r
            ex, ey, ez = aabb_extents_after_rotation(verts, r_tot)
            if ez > pz + 1e-6:
                continue
            if not footprint_fits_bin_mm(ex, ey, bed_x, bed_y, gap_mm):
                continue
            area = ex * ey
            t4 = _rotation_to_4x4(r_tot)
            if best is None or area < best[0]:
                best = (area, t4.copy(), ex, ey)

    if best is None:
        return False, np.eye(4, dtype=np.float64), 0.0, 0.0
    _area, t, fw, fh = best
    return True, t, fw, fh
