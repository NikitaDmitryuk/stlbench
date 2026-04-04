from __future__ import annotations

import numpy as np
import trimesh

from stlbench.core.fit import Method, s_max_for_part_conservative, s_max_for_part_printer_axes
from stlbench.core.orientation import (
    _random_rotation_matrix,
    aabb_extents_after_rotation,
    mesh_vertices_for_orientation,
)
from stlbench.packing.rectpack_plate import footprint_fits_bin_mm


def _axis_perm_4x4_list() -> list[np.ndarray]:
    """Six rigid axis permutations (which AABB extent maps to printer Z vs X/Y)."""
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
    Find a print orientation: after rotation, printer-axis AABB (X,Y,Z) must fit the bed
    and Pz, allowing **axis permutation** (which model axis maps to which printer extent).
    Chooses the smallest XY footprint among valid orientations (for packing).

    For each base SO(3) rotation (identity plus ``random_samples`` draws, same RNG idea as
    scale) try six axis permutations ``P @ R``.

    Returns (ok, 4x4 transform, bed X width, bed Y depth) in printer coordinates (Z up).
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


def select_orientation_for_scale(
    mesh: trimesh.Trimesh,
    px: float,
    py: float,
    pz: float,
    method: Method,
    *,
    random_samples: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Same rotation/permutation search space as ``select_layout_transform`` (``P @ R``),
    but pick the candidate that **maximizes** the per-part scale limit in printer
    coordinates (``s_max_for_part_printer_axes`` for ``sorted``, conservative
    formula for ``conservative``). Tie-break: smaller XY footprint ``ex * ey``.
    """
    verts = mesh_vertices_for_orientation(mesh)
    rng = np.random.default_rng(seed)
    perms = _perm_3x3_list()

    best_key: tuple[float, float] | None = None
    best_t4: np.ndarray | None = None
    best_ext: tuple[float, float, float] | None = None

    bases: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for _ in range(max(0, random_samples)):
        bases.append(_random_rotation_matrix(rng))

    p_min = min(px, py, pz)
    for r in bases:
        for p in perms:
            r_tot = p @ r
            ex, ey, ez = aabb_extents_after_rotation(verts, r_tot)
            if method == "sorted":
                sc, _ = s_max_for_part_printer_axes(px, py, pz, ex, ey, ez)
            else:
                sc = s_max_for_part_conservative(p_min, ex, ey, ez)
            area = ex * ey
            key = (-sc, area)
            if best_key is None or key < best_key:
                best_key = key
                best_t4 = _rotation_to_4x4(r_tot)
                best_ext = (ex, ey, ez)

    assert best_t4 is not None and best_ext is not None
    return best_t4, best_ext
