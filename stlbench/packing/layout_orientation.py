from __future__ import annotations

import numpy as np
import trimesh

from stlbench.core.fit import Method, s_max_for_part_conservative, s_max_for_part_printer_axes
from stlbench.core.orientation import (
    _random_rotation_matrix,
    _z_rotation_candidates,
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
    any_rotation: bool = False,
) -> tuple[bool, np.ndarray, float, float]:
    """
    Find a print orientation for layout/packing.

    By default (``any_rotation=False``) only Z-axis rotations are considered
    (360 deterministic 1° steps).  The model is never flipped onto a different
    face — useful when supports are already placed.

    With ``any_rotation=True`` the full SO(3) is searched: ``random_samples``
    random rotations combined with all six canonical axis permutations, picking
    the smallest XY footprint among orientations that fit the build volume.

    Returns (ok, 4x4 transform, footprint_width, footprint_height).
    """
    verts = mesh_vertices_for_orientation(mesh)

    best: tuple[float, np.ndarray, float, float] | None = None

    if any_rotation:
        rng = np.random.default_rng(seed)
        perms = _perm_3x3_list()
        bases: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
        for _ in range(max(0, random_samples)):
            bases.append(_random_rotation_matrix(rng))
    else:
        perms = [np.eye(3, dtype=np.float64)]
        bases = _z_rotation_candidates()

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
    any_rotation: bool = False,
    maximize: bool = False,
    random_samples: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Search for a print orientation that optimises scale fit.

    By default (``any_rotation=False``) only Z-axis rotations are searched
    (360 deterministic 1° steps), keeping the model on its original face.
    This is appropriate when supports are already in place.

    With ``any_rotation=True`` all six canonical axis permutations are tried.
    Adding ``maximize=True`` (requires ``any_rotation=True``) also samples
    ``random_samples`` random SO(3) rotations on top of the canonical set.
    """
    verts = mesh_vertices_for_orientation(mesh)

    best_key: tuple[float, float] | None = None
    best_t4: np.ndarray | None = None
    best_ext: tuple[float, float, float] | None = None

    if any_rotation:
        rng = np.random.default_rng(seed)
        perms = _perm_3x3_list()
        bases: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
        if maximize:
            for _ in range(max(0, random_samples)):
                bases.append(_random_rotation_matrix(rng))
    else:
        perms = [np.eye(3, dtype=np.float64)]
        bases = _z_rotation_candidates()

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
