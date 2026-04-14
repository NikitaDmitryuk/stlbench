"""Overhang analysis and support-minimizing orientation search.

Algorithm (Tweaker-3 inspired):
  1. Build a set of candidate "bottom" directions from the mesh's own face normals
     (largest faces first) plus a uniform icosphere sample.
  2. For each candidate direction, compute the overhang score of the mesh when
     that direction faces the build plate.
  3. Refine the best candidates with scipy.optimize.minimize (Nelder-Mead).

Score = overhang_area - 0.5 * bottom_area   (lower is better)

  overhang_area – sum of face areas whose downward-facing angle exceeds the
                  overhang threshold (default 45°).
  bottom_area   – sum of face areas that are nearly flat (≤ 5° from horizontal)
                  and face down; large flat bases stabilise the print and need
                  no support pillars.
"""

from __future__ import annotations

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _rotation_from_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix R such that R @ src ≈ dst (unit vectors)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    v = np.cross(src, dst)
    s = float(np.linalg.norm(v))
    c = float(np.dot(src, dst))

    if s < 1e-10:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180° rotation around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * (1.0 - c) / (s * s)


def _angles_to_down(theta: float, phi: float) -> np.ndarray:
    """Spherical-coordinate unit vector; used as the candidate bottom direction."""
    return np.array(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        dtype=np.float64,
    )


def _rotation_from_angles(theta: float, phi: float) -> np.ndarray:
    """Rotation that places the (theta, phi) direction toward [0, 0, -1]."""
    down = np.array([0.0, 0.0, -1.0])
    direction = _angles_to_down(theta, phi)
    return _rotation_from_to(direction, down)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_DOWN = np.array([0.0, 0.0, -1.0])


def overhang_score(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    overhang_threshold_deg: float = 45.0,
    *,
    cos_t: float | None = None,
) -> float:
    """Compute overhang score for mesh under the given 3×3 rotation.

    Parameters
    ----------
    mesh:
        Source mesh (unchanged).
    rotation:
        3×3 rotation matrix to apply to face normals.
    overhang_threshold_deg:
        Faces whose downward angle exceeds this threshold need support.
        Typical values: 45° (FDM/resin default), 30° (conservative resin).
    cos_t:
        Pre-computed ``cos(radians(overhang_threshold_deg))``.  Pass this
        when calling the function in a tight loop to avoid recomputing the
        transcendental function on every call.

    Returns
    -------
    float
        Lower score ↔ fewer supports.
    """
    normals = mesh.face_normals @ rotation.T  # (N, 3)
    nz = normals[:, 2]
    areas = mesh.area_faces

    if cos_t is None:
        cos_t = float(np.cos(np.radians(overhang_threshold_deg)))

    # "True bottom": nearly-flat faces (within ~8°) that rest directly on the
    # build plate / FEP — cured in the first layers, no printed support needed.
    bottom_mask = nz < -0.99
    bottom_area = float(areas[bottom_mask].sum())

    # Overhang faces: downward enough to need support, but NOT the flat bottom.
    overhang_mask = (nz < -cos_t) & ~bottom_mask
    overhang_area = float(areas[overhang_mask].sum())

    # Lower score = fewer / smaller supports.
    # Bonus for large flat bases stabilises the print and avoids pillars.
    return overhang_area - 0.5 * bottom_area


def _batch_overhang_scores(
    face_normals: np.ndarray,
    area_faces: np.ndarray,
    rotations: np.ndarray,
    cos_t: float,
    chunk_size: int = 128,
) -> np.ndarray:
    """Batch overhang scoring for K candidate rotations without Python loop overhead.

    Parameters
    ----------
    face_normals : (N, 3) array
    area_faces   : (N,) array
    rotations    : (K, 3, 3) array of candidate rotation matrices
    cos_t        : pre-computed cos(radians(overhang_threshold_deg))
    chunk_size   : rotations processed per NumPy batch (controls peak memory:
                   chunk_size × N × 8 bytes)

    Returns
    -------
    (K,) array of overhang scores (lower = better)
    """
    K = len(rotations)
    scores = np.empty(K, dtype=np.float64)
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        R_chunk = rotations[start:end]  # (C, 3, 3)
        # nz[c, n] = R_chunk[c, 2, :] · face_normals[n, :]
        nz = R_chunk[:, 2, :] @ face_normals.T  # (C, N)
        bottom_mask = nz < -0.99
        overhang_mask = (nz < -cos_t) & ~bottom_mask
        bottom_areas = (area_faces * bottom_mask).sum(axis=1)
        overhang_areas = (area_faces * overhang_mask).sum(axis=1)
        scores[start:end] = overhang_areas - 0.5 * bottom_areas
    return scores


# ---------------------------------------------------------------------------
# Printer-fit check
# ---------------------------------------------------------------------------


def _fits_printer(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    px: float,
    py: float,
    pz: float,
) -> bool:
    """Return True if the mesh in *rotation* fits within (px, py, pz).

    XY dimensions may be swapped (the model can be placed in any XY orientation
    on the build plate).  Z must not exceed pz.

    Uses convex-hull vertices instead of all mesh vertices: the AABB of the
    convex hull equals the AABB of the full mesh, but with far fewer points.
    """
    verts = mesh.convex_hull.vertices @ rotation.T
    dx = float(verts[:, 0].max() - verts[:, 0].min())
    dy = float(verts[:, 1].max() - verts[:, 1].min())
    dz = float(verts[:, 2].max() - verts[:, 2].min())
    xy_lo = min(dx, dy)
    xy_hi = max(dx, dy)
    bed_lo = min(px, py)
    bed_hi = max(px, py)
    tol = 1e-6
    return dz <= pz + tol and xy_lo <= bed_lo + tol and xy_hi <= bed_hi + tol


# ---------------------------------------------------------------------------
# Candidate directions
# ---------------------------------------------------------------------------


def _build_candidates(mesh: trimesh.Trimesh, n_mesh_candidates: int) -> np.ndarray:
    """Return (K, 3) array of candidate 'bottom' unit directions.

    Combines:
      - Top-N face normals by area (likely stable resting faces).
      - Icosphere normals for uniform coverage (subdivisions=1 → 80 faces).
    """
    normals = mesh.face_normals  # (M, 3)
    areas = mesh.area_faces

    k = min(n_mesh_candidates, len(normals))
    top_idx = np.argpartition(areas, -k)[-k:]
    mesh_cands = normals[top_idx]

    ico = trimesh.creation.icosphere(subdivisions=1)  # 80 faces (was 320 at subdivisions=2)
    combined = np.vstack([mesh_cands, ico.face_normals, -ico.face_normals])

    # Normalise and deduplicate (round to 3 decimals for hashing)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.where(norms > 0, norms, 1.0)
    _, unique_idx = np.unique(np.round(combined, 3), axis=0, return_index=True)
    return np.asarray(combined[unique_idx])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_min_overhang_rotation(
    mesh: trimesh.Trimesh,
    overhang_threshold_deg: float = 45.0,
    n_candidates: int = 200,
    printer_dims: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Find the 3×3 rotation that minimises overhang area for *mesh*.

    Parameters
    ----------
    mesh:
        Input mesh (not modified).
    overhang_threshold_deg:
        Overhang angle threshold (45° is standard).
    n_candidates:
        How many of the mesh's largest faces to include as candidate
        bottom directions (in addition to icosphere samples).
    printer_dims:
        ``(px, py, pz)`` build-volume in mm (after margin).  When given,
        orientations where the model does not fit receive a heavy penalty so
        the search strongly prefers in-bounds solutions.

    Returns
    -------
    rotation : np.ndarray, shape (3, 3)
        Best rotation found (apply with ``mesh.apply_transform(R4)``).
    score : float
        Overhang score of that rotation (penalty excluded).
    """
    from scipy.optimize import minimize  # required dependency

    _PRINTER_PENALTY = 1e9
    cos_t = float(np.cos(np.radians(overhang_threshold_deg)))

    def _penalised_score(R: np.ndarray) -> float:
        s = overhang_score(mesh, R, cos_t=cos_t)
        if printer_dims is not None and not _fits_printer(mesh, R, *printer_dims):
            s += _PRINTER_PENALTY
        return s

    candidates = _build_candidates(mesh, n_candidates)

    # Phase 1: batch-evaluate all candidates in one vectorised pass ----------
    rotations = np.array([_rotation_from_to(cand, _DOWN) for cand in candidates])  # (K, 3, 3)
    raw_scores = _batch_overhang_scores(mesh.face_normals, mesh.area_faces, rotations, cos_t)
    n_top = min(3, len(raw_scores))
    top_idx = np.argpartition(raw_scores, n_top - 1)[:n_top]
    top_candidates = [(raw_scores[i], rotations[i]) for i in top_idx]
    top_candidates.sort(key=lambda x: x[0])

    # Phase 2: Nelder-Mead refinement around each top candidate -------------
    best_penalised = float("inf")
    best_R = rotations[top_idx[0]]

    for _, init_R in top_candidates:
        down_in_orig = init_R.T @ _DOWN
        phi0 = float(np.arccos(np.clip(down_in_orig[2], -1.0, 1.0)))
        theta0 = float(np.arctan2(down_in_orig[1], down_in_orig[0]))

        def _objective(angles: np.ndarray) -> float:
            R = _rotation_from_angles(angles[0], angles[1])
            return _penalised_score(R)

        result = minimize(
            _objective,
            x0=np.array([theta0, phi0]),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": mesh.area * 1e-4, "maxiter": 150},
        )

        if result.fun < best_penalised:
            best_penalised = result.fun
            best_R = _rotation_from_angles(result.x[0], result.x[1])

    return best_R, overhang_score(mesh, best_R, cos_t=cos_t)


def rotation_to_transform4(rotation: np.ndarray) -> np.ndarray:
    """Embed a 3×3 rotation into a 4×4 homogeneous transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    return T


def apply_min_overhang_orientation(mesh: trimesh.Trimesh, rotation: np.ndarray) -> trimesh.Trimesh:
    """Apply rotation and translate the mesh so its lowest point sits at z = 0."""
    m = mesh.copy()
    m.apply_transform(rotation_to_transform4(rotation))
    m.apply_translation([0.0, 0.0, -float(m.bounds[0, 2])])
    return m
