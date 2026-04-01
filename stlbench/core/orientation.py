from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrientationResult:
    extents: tuple[float, float, float]
    rotation: np.ndarray  # 3×3 SO(3)
    score: float


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q.astype(np.float64, copy=False)


def aabb_extents_after_rotation(
    vertices: np.ndarray, rotation: np.ndarray
) -> tuple[float, float, float]:
    if vertices.size == 0:
        raise ValueError("Empty vertex list.")
    r = (rotation @ vertices.T).T
    lo = r.min(axis=0)
    hi = r.max(axis=0)
    d = hi - lo
    return float(d[0]), float(d[1]), float(d[2])


def score_sorted_fit(
    extents_xyz: tuple[float, float, float], p_sorted: tuple[float, float, float]
) -> float:
    d_sorted = sorted(extents_xyz)
    p1, p2, p3 = p_sorted
    d1, d2, d3 = d_sorted[0], d_sorted[1], d_sorted[2]
    if d1 <= 0 or d2 <= 0 or d3 <= 0:
        return 0.0
    return min(p1 / d1, p2 / d2, p3 / d3)


def score_conservative_fit(extents_xyz: tuple[float, float, float], p_min: float) -> float:
    dmax = max(extents_xyz)
    if dmax <= 0:
        return 0.0
    return p_min / dmax


def best_orientation_for_conservative_fit(
    vertices: np.ndarray,
    p_min: float,
    n_samples: int,
    rng: np.random.Generator,
    *,
    identity_baseline: tuple[float, float, float] | None = None,
) -> OrientationResult:
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be (N, 3).")

    i3 = np.eye(3, dtype=np.float64)
    if identity_baseline is not None:
        best_extents = identity_baseline
        best_score = score_conservative_fit(identity_baseline, p_min)
    else:
        best_extents = aabb_extents_after_rotation(verts, i3)
        best_score = score_conservative_fit(best_extents, p_min)
    best_rot = i3.copy()

    for _ in range(max(0, n_samples)):
        r = _random_rotation_matrix(rng)
        ex = aabb_extents_after_rotation(verts, r)
        sc = score_conservative_fit(ex, p_min)
        if sc > best_score + 1e-15:
            best_score = sc
            best_extents = ex
            best_rot = r.copy()

    return OrientationResult(extents=best_extents, rotation=best_rot, score=best_score)


def best_orientation_for_sorted_fit(
    vertices: np.ndarray,
    p_sorted: tuple[float, float, float],
    n_samples: int,
    rng: np.random.Generator,
    *,
    identity_baseline: tuple[float, float, float] | None = None,
) -> OrientationResult:
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be (N, 3).")

    i3 = np.eye(3, dtype=np.float64)
    if identity_baseline is not None:
        best_extents = identity_baseline
        best_score = score_sorted_fit(identity_baseline, p_sorted)
    else:
        best_extents = aabb_extents_after_rotation(verts, i3)
        best_score = score_sorted_fit(best_extents, p_sorted)
    best_rot = i3.copy()

    for _ in range(max(0, n_samples)):
        r = _random_rotation_matrix(rng)
        ex = aabb_extents_after_rotation(verts, r)
        sc = score_sorted_fit(ex, p_sorted)
        if sc > best_score + 1e-15:
            best_score = sc
            best_extents = ex
            best_rot = r.copy()

    return OrientationResult(extents=best_extents, rotation=best_rot, score=best_score)


# backward-compat wrappers (used by old tests / layout_orientation)
def best_aabb_extents_for_sorted_fit(
    vertices: np.ndarray,
    p_sorted: tuple[float, float, float],
    n_samples: int,
    rng: np.random.Generator,
    *,
    identity_baseline: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    return best_orientation_for_sorted_fit(
        vertices, p_sorted, n_samples, rng, identity_baseline=identity_baseline
    ).extents


def best_aabb_extents_for_conservative_fit(
    vertices: np.ndarray,
    p_min: float,
    n_samples: int,
    rng: np.random.Generator,
    *,
    identity_baseline: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    return best_orientation_for_conservative_fit(
        vertices, p_min, n_samples, rng, identity_baseline=identity_baseline
    ).extents


def mesh_vertices_for_orientation(mesh: object, max_vertices: int = 80_000) -> np.ndarray:
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Expected trimesh.Trimesh.")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if v.shape[0] <= max_vertices:
        return v
    try:
        hull = mesh.convex_hull
        return np.asarray(hull.vertices, dtype=np.float64)
    except Exception:
        sel = np.linspace(0, v.shape[0] - 1, num=min(max_vertices, v.shape[0]), dtype=int)
        return np.asarray(v[sel], dtype=np.float64)
