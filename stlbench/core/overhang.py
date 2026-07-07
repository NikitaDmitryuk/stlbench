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

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import trimesh

from stlbench.config.defaults import (
    DEFAULT_ASSEMBLY_SIDE_POLICY,
    DEFAULT_LONG_PART_ANGLE_POLICY,
    DEFAULT_LONG_PART_HIGH_ANGLE_PENALTY_ABOVE_DEG,
    DEFAULT_LONG_PART_LOW_ANGLE_PENALTY_BELOW_DEG,
    DEFAULT_LONG_PART_TARGET_ANGLE_MAX_DEG,
    DEFAULT_LONG_PART_TARGET_ANGLE_MIN_DEG,
    DEFAULT_RESIN_BALANCE,
)
from stlbench.config.enums import (
    AssemblySidePolicy,
    CandidateProfile,
    LongPartAnglePolicy,
    ResinBalance,
    coerce_enum,
)

LEGACY_LINEAR_MIN_PCA_ASPECT = 3.0
LEGACY_LINEAR_MAX_PCA_LINE_RATIO = 0.55
THIN_LINEAR_MIN_PCA_ASPECT = 6.0
THIN_LINEAR_MAX_PCA_LINE_RATIO = 0.35

LONG_VERTICAL_Z_START = 0.57
LONG_VERTICAL_ASPECT_DIVISOR = 1.5
LONG_VERTICAL_MAX_ASPECT_MULTIPLIER = 10.0
FLAT_PART_RATIO_THRESHOLD = 0.35
FLAT_PART_MIN_PCA_ASPECT = 2.0
FLAT_PART_VERTICAL_Z_START = 0.45
FLAT_VERTICAL_PENALTY_WEIGHT = 4.0

NON_LINEAR_FOOTPRINT_WEIGHT_MULTIPLIER = 0.30
NON_LINEAR_HEIGHT_WEIGHT_MULTIPLIER = 3.0
NON_LINEAR_CENTER_WEIGHT_MULTIPLIER = 1.8
NON_LINEAR_CONTACT_WEIGHT_MULTIPLIER = 0.50
NON_LINEAR_SURFACE_WEIGHT_MULTIPLIER = 1.35
NON_LINEAR_UPSIDE_WEIGHT_MULTIPLIER = 1.35
COMPACT_LINEAR_FOOTPRINT_WEIGHT_MULTIPLIER = 8.0
COMPACT_LINEAR_HEIGHT_WEIGHT_MULTIPLIER = 0.50
NON_LINEAR_CONTACT_MULTIPLIER = 0.35

ADAPTIVE_SUPPORT_DELTA_TRIGGER = 0.16
ADAPTIVE_SUPPORT_FIT_MARGIN_TRIGGER = 0.12
ADAPTIVE_NEAR_BOUNDS_FIT_MARGIN = 0.08
ADAPTIVE_CLOSE_PHASE1_DELTA_TRIGGER = 0.01

ASSEMBLY_LOW_SALIENCY_MAX = 0.12
ASSEMBLY_FLAT_CLUSTER_COS = 0.965
ASSEMBLY_MIN_CLUSTER_AREA_RATIO = 0.075
ASSEMBLY_MAX_LOW_SALIENCY_AREA_RATIO = 0.96
ASSEMBLY_SOURCE_TOP_EXCLUSION_DOT = 0.55
ASSEMBLY_MIN_CONFIDENCE = 0.55
ASSEMBLY_SUPPORT_REWARD_WEIGHT = 4.0
ASSEMBLY_VISIBLE_SUPPORT_WEIGHT = 2.20
ASSEMBLY_ALIGNMENT_REWARD_WEIGHT = 1.5
ASSEMBLY_SOURCE_UP_RELIEF = 0.65
ASSEMBLY_TILTED_DOWN_Z = -0.72
ASSEMBLY_Z_SPIN_STEPS = 6


@dataclass(frozen=True)
class StabilityScoreWeights:
    support: float
    footprint: float
    height: float
    center: float
    contact: float
    angle: float
    hard_angle: float
    surface: float
    upside: float


STABILITY_SCORE_WEIGHTS: dict[ResinBalance, StabilityScoreWeights] = {
    ResinBalance.STABILITY: StabilityScoreWeights(
        support=1.00,
        footprint=0.35,
        height=0.35,
        center=0.35,
        contact=0.80,
        angle=2.8,
        hard_angle=2.4,
        surface=1.70,
        upside=1.60,
    ),
    ResinBalance.COMPACT: StabilityScoreWeights(
        support=0.65,
        footprint=1.60,
        height=0.20,
        center=0.20,
        contact=0.60,
        angle=1.6,
        hard_angle=1.5,
        surface=0.75,
        upside=0.75,
    ),
    ResinBalance.BALANCED: StabilityScoreWeights(
        support=0.80,
        footprint=1.10,
        height=0.25,
        center=0.25,
        contact=0.90,
        angle=2.2,
        hard_angle=2.0,
        surface=1.20,
        upside=1.10,
    ),
}


@dataclass(frozen=True)
class ResinOrientationOptions:
    resin_balance: ResinBalance | str = DEFAULT_RESIN_BALANCE
    long_part_angle_policy: LongPartAnglePolicy | str = DEFAULT_LONG_PART_ANGLE_POLICY
    assembly_side_policy: AssemblySidePolicy | str = DEFAULT_ASSEMBLY_SIDE_POLICY
    long_part_target_angle_min_deg: float = DEFAULT_LONG_PART_TARGET_ANGLE_MIN_DEG
    long_part_target_angle_max_deg: float = DEFAULT_LONG_PART_TARGET_ANGLE_MAX_DEG
    long_part_low_angle_penalty_below_deg: float = DEFAULT_LONG_PART_LOW_ANGLE_PENALTY_BELOW_DEG
    long_part_high_angle_penalty_above_deg: float = DEFAULT_LONG_PART_HIGH_ANGLE_PENALTY_ABOVE_DEG

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resin_balance",
            coerce_enum(ResinBalance, self.resin_balance, "resin_balance"),
        )
        object.__setattr__(
            self,
            "long_part_angle_policy",
            coerce_enum(
                LongPartAnglePolicy,
                self.long_part_angle_policy,
                "long_part_angle_policy",
            ),
        )
        object.__setattr__(
            self,
            "assembly_side_policy",
            coerce_enum(AssemblySidePolicy, self.assembly_side_policy, "assembly_side_policy"),
        )
        if self.long_part_target_angle_min_deg > self.long_part_target_angle_max_deg:
            raise ValueError("long part target min angle must be <= max angle.")
        if self.long_part_low_angle_penalty_below_deg > self.long_part_target_angle_min_deg:
            raise ValueError("long part low angle threshold must be <= target min angle.")
        if self.long_part_high_angle_penalty_above_deg < self.long_part_target_angle_max_deg:
            raise ValueError("long part high angle threshold must be >= target max angle.")


@dataclass(frozen=True)
class OrientationStabilityMetrics:
    overhang_score: float
    height_mm: float
    center_z_ratio: float
    long_axis_z: float
    long_axis_angle_from_bed_deg: float
    pca_aspect: float
    pca_line_ratio: float
    stability_score: float
    support_score_delta: float
    xy_footprint_area_mm2: float
    support_contact_proxy: float
    surface_damage_proxy: float
    salient_down_area_ratio: float
    flat_safe_down_area_ratio: float
    sacrificial_support_ratio: float
    visible_support_ratio: float
    assembly_side_alignment: float
    assembly_side_confidence: float
    source_up_dot_build_up: float
    upside_down_penalty: float
    angle_band_penalty: float
    vertical_penalty: float
    horizontal_penalty: float
    selection_reason: str


@dataclass(frozen=True)
class OrientationSearchData:
    cos_t: float
    face_normals: np.ndarray
    face_areas: np.ndarray
    face_saliency: np.ndarray
    rotations: np.ndarray
    raw_scores: np.ndarray
    fits: np.ndarray
    phase1_scores: np.ndarray
    convex_hull_vertices: np.ndarray | None
    candidate_profile: CandidateProfile


@dataclass(frozen=True)
class AssemblySideData:
    normal: np.ndarray
    mask: np.ndarray
    confidence: float
    area_ratio: float


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


def _batch_fits_printer(
    ch_verts: np.ndarray,
    rotations: np.ndarray,
    px: float,
    py: float,
    pz: float,
) -> np.ndarray:
    """Batch printer-fit check for K candidate rotations.

    Parameters
    ----------
    ch_verts  : (V, 3) convex-hull vertices of the mesh (cached by trimesh)
    rotations : (K, 3, 3) candidate rotation matrices

    Returns
    -------
    (K,) bool array — True where the rotated mesh fits inside (px, py, pz).
    """
    # rotated[k, v, xyz] = rotations[k] @ ch_verts[v]
    # (K, 3, 3) @ (3, V) → (K, 3, V) → transpose → (K, V, 3)
    rotated = (rotations @ ch_verts.T).transpose(0, 2, 1)
    dx = rotated[:, :, 0].max(axis=1) - rotated[:, :, 0].min(axis=1)
    dy = rotated[:, :, 1].max(axis=1) - rotated[:, :, 1].min(axis=1)
    dz = rotated[:, :, 2].max(axis=1) - rotated[:, :, 2].min(axis=1)
    xy_lo = np.minimum(dx, dy)
    xy_hi = np.maximum(dx, dy)
    bed_lo = min(px, py)
    bed_hi = max(px, py)
    tol = 1e-6
    fits: np.ndarray = (dz <= pz + tol) & (xy_lo <= bed_lo + tol) & (xy_hi <= bed_hi + tol)
    return fits


def _fits_printer_vertices(
    vertices: np.ndarray,
    rotation: np.ndarray,
    px: float,
    py: float,
    pz: float,
) -> bool:
    verts = np.asarray(vertices, dtype=np.float64) @ rotation.T
    dx = float(verts[:, 0].max() - verts[:, 0].min())
    dy = float(verts[:, 1].max() - verts[:, 1].min())
    dz = float(verts[:, 2].max() - verts[:, 2].min())
    xy_lo = min(dx, dy)
    xy_hi = max(dx, dy)
    bed_lo = min(px, py)
    bed_hi = max(px, py)
    tol = 1e-6
    return dz <= pz + tol and xy_lo <= bed_lo + tol and xy_hi <= bed_hi + tol


def _mesh_vertices_for_stability(mesh: trimesh.Trimesh, max_vertices: int = 80_000) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) <= max_vertices:
        return vertices
    sel = np.linspace(0, len(vertices) - 1, num=max_vertices, dtype=int)
    return np.asarray(vertices[sel], dtype=np.float64)


def _unit_or_default(value: np.ndarray | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        out = np.array(default, dtype=np.float64)
    else:
        out = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-12:
        return np.array(default, dtype=np.float64)
    return out / norm


def _face_saliency(mesh: trimesh.Trimesh) -> np.ndarray:
    """Cheap geometry-only proxy for visible/detail-rich faces.

    High values mean the local surface normal varies from adjacent faces, which
    catches relief/curvature better than area alone. Large flat faces remain
    low-saliency and are therefore safer places for support scars.
    """
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    saliency = np.zeros(len(normals), dtype=np.float64)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return saliency
    dots = np.einsum("ij,ij->i", normals[adjacency[:, 0]], normals[adjacency[:, 1]])
    variation = 1.0 - np.clip(np.abs(dots), 0.0, 1.0)
    np.maximum.at(saliency, adjacency[:, 0], variation)
    np.maximum.at(saliency, adjacency[:, 1], variation)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    median_area = float(np.median(areas)) if len(areas) else 0.0
    area_scale = median_area / np.maximum(areas, 1e-12) if median_area > 0 else 1.0
    return np.clip(saliency * 0.3 * np.minimum(area_scale, 2.0), 0.0, 1.0)


def _subsample_surface_data(
    mesh: trimesh.Trimesh,
    max_faces: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    saliency = _face_saliency(mesh)
    n_faces = len(normals)
    if max_faces >= n_faces:
        return normals, areas, saliency

    total_area = float(areas.sum())
    probs = areas / total_area
    rng = np.random.default_rng(0)
    idx = rng.choice(n_faces, max_faces, replace=False, p=probs)
    sub_areas = areas[idx]
    scale = total_area / float(sub_areas.sum())
    return normals[idx], sub_areas * scale, saliency[idx]


def _principal_axis_stats(vertices: np.ndarray) -> tuple[np.ndarray, float, float]:
    centered = vertices - vertices.mean(axis=0)
    if centered.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 1.0, 1.0
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    axis = vecs[:, -1]
    norm = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64) if norm <= 1e-12 else axis / norm
    aspect = float(np.sqrt(vals[-1] / max(vals[0], 1e-12))) if vals[-1] > 0 else 1.0
    line_ratio = float(np.sqrt(vals[-2] / max(vals[-1], 1e-12))) if vals[-1] > 0 else 1.0
    return axis.astype(np.float64, copy=False), max(1.0, aspect), line_ratio


def _principal_axis(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    axis, aspect, _line_ratio = _principal_axis_stats(vertices)
    return axis, aspect


def _detect_assembly_side(
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    face_saliency: np.ndarray,
    options: ResinOrientationOptions,
    source_up: np.ndarray | None,
) -> AssemblySideData | None:
    if options.assembly_side_policy is AssemblySidePolicy.DISABLED:
        return None
    total_area = float(np.sum(face_areas))
    if total_area <= 1e-9 or len(face_normals) == 0:
        return None

    source_up_vec = _unit_or_default(source_up, (0.0, 0.0, 1.0))
    source_top_weight = face_normals @ source_up_vec
    eligible = (face_saliency <= ASSEMBLY_LOW_SALIENCY_MAX) & (
        source_top_weight <= ASSEMBLY_SOURCE_TOP_EXCLUSION_DOT
    )
    low_area_ratio = float(np.sum(face_areas[eligible]) / total_area)
    if (
        low_area_ratio < ASSEMBLY_MIN_CLUSTER_AREA_RATIO
        or low_area_ratio > ASSEMBLY_MAX_LOW_SALIENCY_AREA_RATIO
    ):
        return None

    order = np.argsort(face_areas)[::-1]
    best: AssemblySideData | None = None
    used = np.zeros(len(face_normals), dtype=bool)
    for idx in order:
        if used[idx] or not bool(eligible[idx]):
            continue
        normal = face_normals[idx]
        cluster = eligible & ((face_normals @ normal) >= ASSEMBLY_FLAT_CLUSTER_COS)
        used |= cluster
        cluster_area = float(np.sum(face_areas[cluster]))
        area_ratio = cluster_area / total_area
        if area_ratio < ASSEMBLY_MIN_CLUSTER_AREA_RATIO:
            continue
        mean_saliency = float(
            np.sum(face_areas[cluster] * face_saliency[cluster]) / max(cluster_area, 1e-9)
        )
        source_top_ratio = float(
            np.sum(face_areas[cluster] * np.clip(source_top_weight[cluster], 0.0, 1.0))
            / max(cluster_area, 1e-9)
        )
        confidence = float(
            np.clip(
                (area_ratio / 0.18)
                * (1.0 - mean_saliency / max(ASSEMBLY_LOW_SALIENCY_MAX, 1e-9))
                * (1.0 - source_top_ratio),
                0.0,
                1.0,
            )
        )
        if confidence < ASSEMBLY_MIN_CONFIDENCE:
            continue
        unit_normal = _unit_or_default(normal, (0.0, 0.0, 1.0))
        candidate = AssemblySideData(
            normal=unit_normal,
            mask=cluster,
            confidence=confidence,
            area_ratio=area_ratio,
        )
        if best is None or (
            candidate.confidence,
            candidate.area_ratio,
        ) > (
            best.confidence,
            best.area_ratio,
        ):
            best = candidate
    return best


def _rotation_around_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit_or_default(axis, (0.0, 0.0, 1.0))
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _assembly_side_rotations(assembly_side: AssemblySideData | None) -> list[np.ndarray]:
    if assembly_side is None:
        return []
    normal = assembly_side.normal
    tilted_y = float(np.sqrt(max(0.0, 1.0 - ASSEMBLY_TILTED_DOWN_Z**2)))
    targets = [
        _DOWN,
        np.array([0.0, tilted_y, ASSEMBLY_TILTED_DOWN_Z], dtype=np.float64),
        np.array([0.0, -tilted_y, ASSEMBLY_TILTED_DOWN_Z], dtype=np.float64),
    ]
    rotations: list[np.ndarray] = []
    for target in targets:
        base = _rotation_from_to(normal, target)
        rotations.append(base)
        for spin_idx in range(ASSEMBLY_Z_SPIN_STEPS):
            angle = (spin_idx + 1) * (2.0 * np.pi / ASSEMBLY_Z_SPIN_STEPS)
            rotations.append(_rotation_around_axis(_DOWN, angle) @ base)
    return rotations


def _is_linear_shape(pca_aspect: float, pca_line_ratio: float) -> bool:
    return (
        pca_aspect >= LEGACY_LINEAR_MIN_PCA_ASPECT
        and pca_line_ratio <= LEGACY_LINEAR_MAX_PCA_LINE_RATIO
    )


def _uses_long_part_angle_policy(
    options: ResinOrientationOptions,
    pca_aspect: float,
    pca_line_ratio: float,
) -> bool:
    if options.long_part_angle_policy is LongPartAnglePolicy.DISABLED:
        return False
    if options.long_part_angle_policy is LongPartAnglePolicy.LINEAR:
        return _is_linear_shape(pca_aspect, pca_line_ratio)
    return (
        pca_aspect >= THIN_LINEAR_MIN_PCA_ASPECT
        and pca_line_ratio <= THIN_LINEAR_MAX_PCA_LINE_RATIO
    )


def _stability_metrics(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    overhang_value: float,
    best_overhang_value: float,
    printer_dims: tuple[float, float, float] | None,
    vertices: np.ndarray,
    principal_axis: np.ndarray,
    pca_aspect: float,
    pca_line_ratio: float,
    centroid: np.ndarray,
    total_area: float,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    face_saliency: np.ndarray,
    cos_t: float,
    options: ResinOrientationOptions,
    source_up: np.ndarray | None,
    assembly_side: AssemblySideData | None,
) -> OrientationStabilityMetrics:
    rotated = vertices @ rotation.T
    lo = rotated.min(axis=0)
    hi = rotated.max(axis=0)
    extents = hi - lo
    height = float(extents[2])
    pz = printer_dims[2] if printer_dims is not None else max(height, 1e-9)
    height_ratio = height / max(float(pz), 1e-9)
    c = rotation @ centroid
    center_z_ratio = float(np.clip((c[2] - lo[2]) / max(height, 1e-9), 0.0, 1.0))
    long_axis = rotation @ principal_axis
    long_axis_z = float(abs(long_axis[2]) / max(np.linalg.norm(long_axis), 1e-12))
    long_axis_angle = float(np.degrees(np.arcsin(np.clip(long_axis_z, 0.0, 1.0))))
    xy_footprint_area = float(extents[0] * extents[1])
    support_norm = overhang_value / max(total_area, 1e-9)
    nz = face_normals @ rotation[2]
    support_contact_proxy = float(
        np.sum(face_areas[(nz < -0.2) & (nz > -0.99)]) / max(total_area, 1e-9)
    )
    source_up_vec = _unit_or_default(source_up, (0.0, 0.0, 1.0))
    final_source_up = rotation @ source_up_vec
    source_up_dot_build_up = float(np.clip(final_source_up[2], -1.0, 1.0))
    upside_down_penalty = max(0.0, 0.5 - source_up_dot_build_up) ** 2
    support_need = np.clip((-nz - 0.2) / 0.8, 0.0, 1.0)
    source_top_weight = np.clip(face_normals @ source_up_vec, 0.0, 1.0)
    surface_weight = 0.25 + 1.50 * face_saliency + 1.00 * source_top_weight
    surface_damage_proxy = float(
        np.sum(face_areas * support_need * surface_weight) / max(total_area, 1e-9)
    )
    salient_down_area_ratio = float(
        np.sum(face_areas * support_need * face_saliency) / max(total_area, 1e-9)
    )
    flat_safe_mask = (nz < -0.99) & (face_saliency < 0.15) & (source_top_weight < 0.25)
    flat_safe_down_area_ratio = float(np.sum(face_areas[flat_safe_mask]) / max(total_area, 1e-9))
    if assembly_side is None:
        sacrificial_support_ratio = 0.0
        visible_support_ratio = float(
            np.sum(
                face_areas
                * support_need
                * np.maximum(face_saliency, np.clip(source_top_weight, 0.0, 1.0))
            )
            / max(total_area, 1e-9)
        )
        assembly_side_alignment = 0.0
        assembly_side_confidence = 0.0
    else:
        sacrificial_mask = assembly_side.mask
        visible_weight = np.maximum(
            face_saliency,
            np.clip(source_top_weight, 0.0, 1.0),
        )
        visible_weight = np.where(sacrificial_mask, 0.0, visible_weight)
        sacrificial_support_ratio = float(
            np.sum(face_areas[sacrificial_mask] * support_need[sacrificial_mask])
            / max(total_area, 1e-9)
        )
        visible_support_ratio = float(
            np.sum(face_areas * support_need * visible_weight) / max(total_area, 1e-9)
        )
        assembly_side_alignment = float(np.clip(-(assembly_side.normal @ rotation[2]), 0.0, 1.0))
        assembly_side_confidence = assembly_side.confidence

    is_long = _is_linear_shape(pca_aspect, pca_line_ratio)
    uses_angle_policy = _uses_long_part_angle_policy(options, pca_aspect, pca_line_ratio)
    target_min = options.long_part_target_angle_min_deg
    target_max = options.long_part_target_angle_max_deg
    low_cut = options.long_part_low_angle_penalty_below_deg
    high_cut = options.long_part_high_angle_penalty_above_deg
    horizontal_penalty = 0.0
    vertical_penalty = 0.0
    angle_band_penalty = 0.0
    if uses_angle_policy:
        if long_axis_angle < target_min:
            angle_band_penalty = ((target_min - long_axis_angle) / max(target_min, 1.0)) ** 2
        elif long_axis_angle > target_max:
            angle_band_penalty = ((long_axis_angle - target_max) / max(90.0 - target_max, 1.0)) ** 2
        if long_axis_angle < low_cut:
            horizontal_penalty = ((low_cut - long_axis_angle) / max(low_cut, 1.0)) ** 2
        if long_axis_angle > high_cut:
            vertical_penalty = ((long_axis_angle - high_cut) / max(90.0 - high_cut, 1.0)) ** 2

    long_penalty = 0.0
    if is_long:
        long_penalty = max(0.0, long_axis_z - LONG_VERTICAL_Z_START) ** 2 * min(
            LONG_VERTICAL_MAX_ASPECT_MULTIPLIER,
            pca_aspect / LONG_VERTICAL_ASPECT_DIVISOR,
        )

    dims_sorted = np.sort(extents)
    flat_ratio = float(dims_sorted[0] / max(dims_sorted[2], 1e-9))
    flat_vertical_penalty = 0.0
    if flat_ratio < FLAT_PART_RATIO_THRESHOLD and pca_aspect >= FLAT_PART_MIN_PCA_ASPECT:
        flat_vertical_penalty = (
            max(0.0, long_axis_z - FLAT_PART_VERTICAL_Z_START) ** 2 * FLAT_VERTICAL_PENALTY_WEIGHT
        )

    footprint_norm = xy_footprint_area / max(
        printer_dims[0] * printer_dims[1] if printer_dims is not None else xy_footprint_area,
        1e-9,
    )
    weights = STABILITY_SCORE_WEIGHTS[cast(ResinBalance, options.resin_balance)]
    support_w = weights.support
    footprint_w = weights.footprint
    height_w = weights.height
    center_w = weights.center
    contact_w = weights.contact
    angle_w = weights.angle
    hard_angle_w = weights.hard_angle
    surface_w = weights.surface
    upside_w = weights.upside
    if not is_long:
        footprint_w *= NON_LINEAR_FOOTPRINT_WEIGHT_MULTIPLIER
        height_w *= NON_LINEAR_HEIGHT_WEIGHT_MULTIPLIER
        center_w *= NON_LINEAR_CENTER_WEIGHT_MULTIPLIER
        contact_w *= NON_LINEAR_CONTACT_WEIGHT_MULTIPLIER
        surface_w *= NON_LINEAR_SURFACE_WEIGHT_MULTIPLIER
        upside_w *= NON_LINEAR_UPSIDE_WEIGHT_MULTIPLIER
    elif options.resin_balance is ResinBalance.COMPACT:
        footprint_w *= COMPACT_LINEAR_FOOTPRINT_WEIGHT_MULTIPLIER
        height_w *= COMPACT_LINEAR_HEIGHT_WEIGHT_MULTIPLIER

    long_contact = support_contact_proxy * (1.0 if is_long else NON_LINEAR_CONTACT_MULTIPLIER)
    if assembly_side_confidence >= ASSEMBLY_MIN_CONFIDENCE and assembly_side_alignment > 0.35:
        upside_w *= 1.0 - ASSEMBLY_SOURCE_UP_RELIEF * assembly_side_confidence

    assembly_reward = (
        ASSEMBLY_SUPPORT_REWARD_WEIGHT * assembly_side_confidence * sacrificial_support_ratio
        + ASSEMBLY_ALIGNMENT_REWARD_WEIGHT
        * assembly_side_confidence
        * assembly_side_alignment
        * assembly_side.area_ratio
        if assembly_side is not None
        else 0.0
    )
    assembly_visible_penalty = (
        ASSEMBLY_VISIBLE_SUPPORT_WEIGHT * assembly_side_confidence * visible_support_ratio
    )
    stability_score = (
        support_w * support_norm
        + footprint_w * footprint_norm
        + height_w * height_ratio
        + center_w * center_z_ratio
        + contact_w * long_contact
        + angle_w * angle_band_penalty
        + hard_angle_w * (horizontal_penalty + vertical_penalty)
        + surface_w * surface_damage_proxy
        + upside_w * upside_down_penalty
        + long_penalty
        + flat_vertical_penalty
        + assembly_visible_penalty
        - assembly_reward
    )
    selection_reason = "pure_overhang"
    if uses_angle_policy and target_min <= long_axis_angle <= target_max:
        selection_reason = "long_part_target_band"
    elif uses_angle_policy and (
        angle_band_penalty > 0 or horizontal_penalty > 0 or vertical_penalty > 0
    ):
        selection_reason = "balanced_long_part"
    elif flat_vertical_penalty > 0:
        selection_reason = "flat_plate_stability"
    elif (
        assembly_side_confidence >= ASSEMBLY_MIN_CONFIDENCE
        and assembly_side_alignment > 0.35
        and sacrificial_support_ratio >= visible_support_ratio
    ):
        selection_reason = "assembly_side_support"
    elif upside_down_penalty <= 1e-6 and source_up_dot_build_up > 0.35:
        selection_reason = "source_up_preserved"
    elif surface_damage_proxy < 0.20:
        selection_reason = "surface_damage_avoided"
    elif upside_down_penalty > 0.0 or surface_damage_proxy > 0.45:
        selection_reason = "support_overrode_surface"
    return OrientationStabilityMetrics(
        overhang_score=overhang_value,
        height_mm=height,
        center_z_ratio=center_z_ratio,
        long_axis_z=long_axis_z,
        long_axis_angle_from_bed_deg=long_axis_angle,
        pca_aspect=pca_aspect,
        pca_line_ratio=pca_line_ratio,
        stability_score=stability_score,
        support_score_delta=overhang_value - best_overhang_value,
        xy_footprint_area_mm2=xy_footprint_area,
        support_contact_proxy=support_contact_proxy,
        surface_damage_proxy=surface_damage_proxy,
        salient_down_area_ratio=salient_down_area_ratio,
        flat_safe_down_area_ratio=flat_safe_down_area_ratio,
        sacrificial_support_ratio=sacrificial_support_ratio,
        visible_support_ratio=visible_support_ratio,
        assembly_side_alignment=assembly_side_alignment,
        assembly_side_confidence=assembly_side_confidence,
        source_up_dot_build_up=source_up_dot_build_up,
        upside_down_penalty=upside_down_penalty,
        angle_band_penalty=angle_band_penalty,
        vertical_penalty=vertical_penalty,
        horizontal_penalty=horizontal_penalty,
        selection_reason=selection_reason,
    )


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
    return _fits_printer_vertices(mesh.convex_hull.vertices, rotation, px, py, pz)


# ---------------------------------------------------------------------------
# Candidate directions
# ---------------------------------------------------------------------------


def _build_candidates(
    mesh: trimesh.Trimesh,
    n_mesh_candidates: int,
    *,
    candidate_profile: CandidateProfile | str = CandidateProfile.DEFAULT,
) -> np.ndarray:
    candidate_profile = coerce_enum(CandidateProfile, candidate_profile, "candidate_profile")
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

    if candidate_profile is CandidateProfile.ADAPTIVE:
        ico_subdivisions = 2
    elif candidate_profile is CandidateProfile.DEFAULT:
        ico_subdivisions = 1
    ico = trimesh.creation.icosphere(subdivisions=ico_subdivisions)
    # Six axis-aligned directions guarantee at least one Phase-1 candidate for
    # meshes that only fit in a narrow angular window (e.g. tall/thin parts).
    axis_dirs = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    combined = np.vstack([mesh_cands, ico.face_normals, -ico.face_normals, axis_dirs])

    # Normalise and deduplicate (round to 3 decimals for hashing)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.where(norms > 0, norms, 1.0)
    _, unique_idx = np.unique(np.round(combined, 3), axis=0, return_index=True)
    return np.asarray(combined[unique_idx])


# ---------------------------------------------------------------------------
# Face-normal subsampling
# ---------------------------------------------------------------------------

_MAX_FACES_ORIENT = 50_000  # beyond this, subsample for speed


def _subsample_normals(
    face_normals: np.ndarray,
    area_faces: np.ndarray,
    max_faces: int = _MAX_FACES_ORIENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample face normals/areas for faster overhang scoring.

    Faces are chosen **with probability proportional to their area**, so large
    (influential) faces are preferentially retained.  The returned areas are
    scaled so their total equals the original, preserving the absolute
    magnitude of overhang scores.

    For meshes with ≤ *max_faces* faces, the original arrays are returned
    unchanged (no copy, no allocation).

    Parameters
    ----------
    face_normals : (N, 3)
    area_faces   : (N,)
    max_faces    : target sample size (default 50 000)

    Returns
    -------
    face_normals_sub : (min(N, max_faces), 3)
    area_faces_sub   : (min(N, max_faces),) — scaled to preserve total area
    """
    N = len(face_normals)
    if max_faces >= N:
        return face_normals, area_faces

    total_area = float(area_faces.sum())
    probs = area_faces / total_area
    # Seed 0 → deterministic: same mesh always gives same subsample.
    rng = np.random.default_rng(0)
    idx = rng.choice(N, max_faces, replace=False, p=probs)
    sub_areas = area_faces[idx]
    scale = total_area / float(sub_areas.sum())
    return face_normals[idx], sub_areas * scale


def _build_orientation_search_data(
    mesh: trimesh.Trimesh,
    overhang_threshold_deg: float,
    n_candidates: int,
    printer_dims: tuple[float, float, float] | None,
    *,
    candidate_profile: CandidateProfile | str = CandidateProfile.DEFAULT,
) -> OrientationSearchData:
    candidate_profile = coerce_enum(CandidateProfile, candidate_profile, "candidate_profile")
    _PRINTER_PENALTY = 1e9
    cos_t = float(np.cos(np.radians(overhang_threshold_deg)))
    fn, af, face_saliency = _subsample_surface_data(mesh)
    candidates = _build_candidates(mesh, n_candidates, candidate_profile=candidate_profile)
    rotations = np.array([_rotation_from_to(cand, _DOWN) for cand in candidates])
    raw_scores = _batch_overhang_scores(fn, af, rotations, cos_t)
    if printer_dims is not None:
        ch_verts = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
        fits = _batch_fits_printer(ch_verts, rotations, *printer_dims)
        phase1_scores = raw_scores + np.where(fits, 0.0, _PRINTER_PENALTY)
    else:
        ch_verts = None
        fits = np.ones(len(rotations), dtype=bool)
        phase1_scores = raw_scores
    return OrientationSearchData(
        cos_t=cos_t,
        face_normals=fn,
        face_areas=af,
        face_saliency=face_saliency,
        rotations=rotations,
        raw_scores=raw_scores,
        fits=fits,
        phase1_scores=phase1_scores,
        convex_hull_vertices=ch_verts,
        candidate_profile=candidate_profile,
    )


def _fast_overhang_from_data(data: OrientationSearchData, rotation: np.ndarray) -> float:
    nz = data.face_normals @ rotation[2]
    bottom_mask = nz < -0.99
    overhang_mask = (nz < -data.cos_t) & ~bottom_mask
    return float((data.face_areas * overhang_mask).sum()) - 0.5 * float(
        (data.face_areas * bottom_mask).sum()
    )


def _find_min_overhang_rotation_from_data(
    mesh: trimesh.Trimesh,
    data: OrientationSearchData,
    printer_dims: tuple[float, float, float] | None,
) -> tuple[np.ndarray, float]:
    from scipy.optimize import minimize  # required dependency

    _PRINTER_PENALTY = 1e9
    if len(data.phase1_scores) == 0:
        raise ValueError("No orientation candidates.")

    def _penalised_score(rotation: np.ndarray) -> float:
        score = _fast_overhang_from_data(data, rotation)
        if printer_dims is not None:
            if data.convex_hull_vertices is None:
                raise RuntimeError("Printer fit check requires convex hull vertices.")
            if not _fits_printer_vertices(data.convex_hull_vertices, rotation, *printer_dims):
                score += _PRINTER_PENALTY
        return score

    n_top = min(3, len(data.phase1_scores))
    top_idx = np.argpartition(data.phase1_scores, n_top - 1)[:n_top]
    top_candidates = [(data.phase1_scores[i], data.rotations[i]) for i in top_idx]
    top_candidates.sort(key=lambda x: x[0])

    fitting_mask = data.fits & (data.phase1_scores < 0.5 * _PRINTER_PENALTY)
    if fitting_mask.any():
        best_phase1_idx = int(np.argmin(np.where(fitting_mask, data.raw_scores, np.inf)))
        fallback_R: np.ndarray = data.rotations[best_phase1_idx]
    else:
        fallback_R = data.rotations[top_idx[0]]

    best_penalised = float("inf")
    best_R = fallback_R
    for _, init_R in top_candidates:
        down_in_orig = init_R.T @ _DOWN
        phi0 = float(np.arccos(np.clip(down_in_orig[2], -1.0, 1.0)))
        theta0 = float(np.arctan2(down_in_orig[1], down_in_orig[0]))

        def _objective(angles: np.ndarray) -> float:
            rotation = _rotation_from_angles(angles[0], angles[1])
            return _penalised_score(rotation)

        result = minimize(
            _objective,
            x0=np.array([theta0, phi0]),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": mesh.area * 1e-4, "maxiter": 150},
        )
        if result.fun < best_penalised:
            best_penalised = float(result.fun)
            best_R = _rotation_from_angles(result.x[0], result.x[1])

    if best_penalised >= 0.5 * _PRINTER_PENALTY:
        best_R = fallback_R

    return best_R, overhang_score(mesh, best_R, cos_t=data.cos_t)


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

    # Sub-sample face normals/areas for speed on high-poly meshes.
    # Faces are weighted by area so the approximation is unbiased.
    # Meshes with ≤ _MAX_FACES_ORIENT faces are used as-is (no copy).
    fn, af = _subsample_normals(mesh.face_normals, mesh.area_faces)

    def _fast_overhang(R: np.ndarray) -> float:
        """Overhang score using subsampled normals — cheap for NM iterations.

        Uses matrix-vector (fn @ R[2]) instead of matrix-matrix multiply,
        which is another ~3× speedup over the generic overhang_score path.
        R[2] is the third row of the rotation matrix, i.e. the "down" axis
        expressed in the original mesh frame.
        """
        nz = fn @ R[2]  # (N_sub,)
        bottom_mask = nz < -0.99
        overhang_mask = (nz < -cos_t) & ~bottom_mask
        return float((af * overhang_mask).sum()) - 0.5 * float((af * bottom_mask).sum())

    def _penalised_score(R: np.ndarray) -> float:
        s = _fast_overhang(R)
        if printer_dims is not None and not _fits_printer(mesh, R, *printer_dims):
            s += _PRINTER_PENALTY
        return s

    candidates = _build_candidates(mesh, n_candidates)

    # Phase 1: batch-evaluate all candidates in one vectorised pass ----------
    rotations = np.array([_rotation_from_to(cand, _DOWN) for cand in candidates])  # (K, 3, 3)
    raw_scores = _batch_overhang_scores(fn, af, rotations, cos_t)

    # Apply printer-fit penalty in Phase 1 so only fitting orientations are
    # selected as starting points for the Nelder-Mead refinement.
    if printer_dims is not None:
        ch_verts = mesh.convex_hull.vertices  # (V, 3), cached by trimesh
        fits = _batch_fits_printer(ch_verts, rotations, *printer_dims)
        phase1_scores = raw_scores + np.where(fits, 0.0, _PRINTER_PENALTY)
    else:
        phase1_scores = raw_scores
        fits = np.ones(len(rotations), dtype=bool)

    n_top = min(3, len(phase1_scores))
    top_idx = np.argpartition(phase1_scores, n_top - 1)[:n_top]
    top_candidates = [(phase1_scores[i], rotations[i]) for i in top_idx]
    top_candidates.sort(key=lambda x: x[0])

    # Keep the best fitting Phase 1 candidate as an emergency fallback in case
    # Nelder-Mead fails to escape the non-fitting region for all starting points.
    fitting_mask = fits & (phase1_scores < 0.5 * _PRINTER_PENALTY)
    if fitting_mask.any():
        best_phase1_idx = int(np.argmin(np.where(fitting_mask, raw_scores, np.inf)))
        fallback_R: np.ndarray = rotations[best_phase1_idx]
    else:
        fallback_R = rotations[top_idx[0]]

    # Phase 2: Nelder-Mead refinement around each top candidate -------------
    best_penalised = float("inf")
    best_R = fallback_R

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

    # Fallback: if Phase 2 converged to a non-fitting orientation despite the
    # penalty, use the best fitting candidate from Phase 1 instead.
    if best_penalised >= 0.5 * _PRINTER_PENALTY:
        best_R = fallback_R

    return best_R, overhang_score(mesh, best_R, cos_t=cos_t)


def find_stable_overhang_rotation_legacy(
    mesh: trimesh.Trimesh,
    overhang_threshold_deg: float = 45.0,
    n_candidates: int = 200,
    printer_dims: tuple[float, float, float] | None = None,
    *,
    support_tolerance_ratio: float = 0.20,
    resin_options: ResinOrientationOptions | None = None,
    source_up: np.ndarray | None = None,
) -> tuple[np.ndarray, float, OrientationStabilityMetrics]:
    """Find an overhang-good orientation with a stability-aware tie-break.

    The pure overhang optimum is used as the support baseline.  Among candidate
    orientations whose support score is within ``support_tolerance_ratio`` of
    total mesh area, choose the one with lower height, lower center of mass,
    and less-vertical principal axis for long/thin parts.
    """
    options = resin_options or ResinOrientationOptions()
    cos_t = float(np.cos(np.radians(overhang_threshold_deg)))
    best_overhang_R, best_overhang = find_min_overhang_rotation(
        mesh,
        overhang_threshold_deg=overhang_threshold_deg,
        n_candidates=n_candidates,
        printer_dims=printer_dims,
    )

    fn, af, face_saliency = _subsample_surface_data(mesh)
    total_area = float(np.sum(mesh.area_faces))
    assembly_side = _detect_assembly_side(fn, af, face_saliency, options, source_up)
    candidates = _build_candidates(mesh, n_candidates)
    rotations = np.array([_rotation_from_to(cand, _DOWN) for cand in candidates])
    raw_scores = _batch_overhang_scores(fn, af, rotations, cos_t)

    all_rotations = [best_overhang_R]
    all_scores = [best_overhang]
    for rotation, score in zip(rotations, raw_scores, strict=True):
        score_f = float(score)
        if score_f <= best_overhang + support_tolerance_ratio * max(total_area, 1.0) and (
            printer_dims is None or _fits_printer(mesh, rotation, *printer_dims)
        ):
            all_rotations.append(rotation)
            all_scores.append(score_f)
    for rotation in _assembly_side_rotations(assembly_side):
        if printer_dims is not None and not _fits_printer(mesh, rotation, *printer_dims):
            continue
        score_f = overhang_score(mesh, rotation, cos_t=cos_t)
        if score_f <= best_overhang + support_tolerance_ratio * max(total_area, 1.0):
            all_rotations.append(rotation)
            all_scores.append(score_f)

    vertices = _mesh_vertices_for_stability(mesh)
    principal_axis, pca_aspect, pca_line_ratio = _principal_axis_stats(vertices)
    centroid = np.asarray(mesh.centroid, dtype=np.float64)
    metrics = [
        _stability_metrics(
            mesh,
            rotation,
            score,
            best_overhang,
            printer_dims,
            vertices,
            principal_axis,
            pca_aspect,
            pca_line_ratio,
            centroid,
            total_area,
            fn,
            af,
            face_saliency,
            cos_t,
            options,
            source_up,
            assembly_side,
        )
        for rotation, score in zip(all_rotations, all_scores, strict=True)
    ]
    in_target_band = [
        i
        for i, m in enumerate(metrics)
        if (
            _uses_long_part_angle_policy(options, m.pca_aspect, m.pca_line_ratio)
            and options.long_part_target_angle_min_deg
            <= m.long_axis_angle_from_bed_deg
            <= options.long_part_target_angle_max_deg
        )
    ]
    if in_target_band and options.resin_balance is ResinBalance.BALANCED:
        # For long parts the target angle band exists to avoid both failure modes:
        # fully horizontal "support along the whole shaft" and near-vertical fragile tips.
        candidates_idx = in_target_band
    else:
        candidates_idx = list(range(len(all_rotations)))
    best_idx = min(
        candidates_idx,
        key=lambda i: (
            metrics[i].stability_score,
            metrics[i].overhang_score,
            metrics[i].height_mm,
        ),
    )
    return all_rotations[best_idx], metrics[best_idx].overhang_score, metrics[best_idx]


def _orientation_quality_tuple(metrics: OrientationStabilityMetrics) -> tuple[float, float, float]:
    return metrics.stability_score, metrics.overhang_score, metrics.height_mm


def _metrics_not_worse(
    candidate: OrientationStabilityMetrics,
    baseline: OrientationStabilityMetrics,
    *,
    atol: float = 1e-6,
) -> bool:
    return all(
        cand <= base + atol
        for cand, base in zip(
            _orientation_quality_tuple(candidate), _orientation_quality_tuple(baseline), strict=True
        )
    )


def _fit_margin_ratio(
    vertices: np.ndarray,
    rotation: np.ndarray,
    printer_dims: tuple[float, float, float] | None,
) -> float:
    if printer_dims is None or len(vertices) == 0:
        return 1.0
    rotated = np.asarray(vertices, dtype=np.float64) @ rotation.T
    extents = rotated.max(axis=0) - rotated.min(axis=0)
    px, py, pz = printer_dims
    xy_lo = min(float(extents[0]), float(extents[1]))
    xy_hi = max(float(extents[0]), float(extents[1]))
    bed_lo = min(px, py)
    bed_hi = max(px, py)
    margins = [
        (bed_lo - xy_lo) / max(bed_lo, 1e-9),
        (bed_hi - xy_hi) / max(bed_hi, 1e-9),
        (pz - float(extents[2])) / max(pz, 1e-9),
    ]
    return float(min(margins))


def _phase1_second_delta_ratio(data: OrientationSearchData, total_area: float) -> float:
    finite = np.asarray(data.phase1_scores[np.isfinite(data.phase1_scores)], dtype=np.float64)
    if len(finite) < 2:
        return float("inf")
    idx = np.argpartition(finite, 1)[:2]
    best_two = np.sort(finite[idx])
    return float((best_two[1] - best_two[0]) / max(total_area, 1.0))


def _adaptive_trigger_reason(
    metrics: OrientationStabilityMetrics,
    data: OrientationSearchData,
    total_area: float,
    rotation: np.ndarray,
    printer_dims: tuple[float, float, float] | None,
) -> str:
    reasons: list[str] = []
    fit_vertices = (
        data.convex_hull_vertices if data.convex_hull_vertices is not None else np.empty((0, 3))
    )
    fit_margin = _fit_margin_ratio(fit_vertices, rotation, printer_dims)
    is_linear_part = _is_linear_shape(metrics.pca_aspect, metrics.pca_line_ratio)
    if is_linear_part:
        reasons.append("linear_part")
    band_delta = metrics.support_score_delta / max(total_area, 1.0)
    if band_delta >= ADAPTIVE_SUPPORT_DELTA_TRIGGER and (
        is_linear_part or fit_margin <= ADAPTIVE_SUPPORT_FIT_MARGIN_TRIGGER
    ):
        reasons.append("support_tolerance_edge")
    if fit_margin <= ADAPTIVE_NEAR_BOUNDS_FIT_MARGIN:
        reasons.append("near_printer_bounds")
    if is_linear_part and metrics.selection_reason in {
        "balanced_long_part",
        "support_overrode_surface",
        "flat_plate_stability",
    }:
        reasons.append(f"selection_{metrics.selection_reason}")
    if _phase1_second_delta_ratio(data, total_area) <= ADAPTIVE_CLOSE_PHASE1_DELTA_TRIGGER and (
        is_linear_part or fit_margin <= ADAPTIVE_NEAR_BOUNDS_FIT_MARGIN
    ):
        reasons.append("close_phase1_candidates")
    return ",".join(reasons)


def _stable_overhang_rotation_with_diagnostics(
    mesh: trimesh.Trimesh,
    overhang_threshold_deg: float = 45.0,
    n_candidates: int = 200,
    printer_dims: tuple[float, float, float] | None = None,
    *,
    support_tolerance_ratio: float = 0.20,
    resin_options: ResinOrientationOptions | None = None,
    source_up: np.ndarray | None = None,
    candidate_profile: CandidateProfile | str = CandidateProfile.DEFAULT,
) -> tuple[np.ndarray, float, OrientationStabilityMetrics, dict[str, float | str | bool | int]]:
    candidate_profile = coerce_enum(CandidateProfile, candidate_profile, "candidate_profile")
    options = resin_options or ResinOrientationOptions()
    effective_candidates = (
        n_candidates * 2 if candidate_profile is CandidateProfile.ADAPTIVE else n_candidates
    )
    data = _build_orientation_search_data(
        mesh,
        overhang_threshold_deg=overhang_threshold_deg,
        n_candidates=effective_candidates,
        printer_dims=printer_dims,
        candidate_profile=candidate_profile,
    )
    best_overhang_R, best_overhang = _find_min_overhang_rotation_from_data(
        mesh,
        data,
        printer_dims,
    )

    total_area = float(np.sum(mesh.area_faces))
    assembly_side = _detect_assembly_side(
        data.face_normals,
        data.face_areas,
        data.face_saliency,
        options,
        source_up,
    )
    all_rotations = [best_overhang_R]
    all_scores = [best_overhang]
    max_allowed = best_overhang + support_tolerance_ratio * max(total_area, 1.0)
    for rotation, score, fits in zip(data.rotations, data.raw_scores, data.fits, strict=True):
        score_f = float(score)
        if score_f <= max_allowed and (printer_dims is None or bool(fits)):
            all_rotations.append(rotation)
            all_scores.append(score_f)
    for rotation in _assembly_side_rotations(assembly_side):
        if printer_dims is not None:
            if data.convex_hull_vertices is None:
                raise RuntimeError("Printer fit check requires convex hull vertices.")
            if not _fits_printer_vertices(data.convex_hull_vertices, rotation, *printer_dims):
                continue
        score_f = _fast_overhang_from_data(data, rotation)
        if score_f <= max_allowed:
            all_rotations.append(rotation)
            all_scores.append(score_f)

    vertices = _mesh_vertices_for_stability(mesh)
    principal_axis, pca_aspect, pca_line_ratio = _principal_axis_stats(vertices)
    centroid = np.asarray(mesh.centroid, dtype=np.float64)
    metrics = [
        _stability_metrics(
            mesh,
            rotation,
            score,
            best_overhang,
            printer_dims,
            vertices,
            principal_axis,
            pca_aspect,
            pca_line_ratio,
            centroid,
            total_area,
            data.face_normals,
            data.face_areas,
            data.face_saliency,
            data.cos_t,
            options,
            source_up,
            assembly_side,
        )
        for rotation, score in zip(all_rotations, all_scores, strict=True)
    ]
    in_target_band = [
        i
        for i, m in enumerate(metrics)
        if (
            _uses_long_part_angle_policy(options, m.pca_aspect, m.pca_line_ratio)
            and options.long_part_target_angle_min_deg
            <= m.long_axis_angle_from_bed_deg
            <= options.long_part_target_angle_max_deg
        )
    ]
    if in_target_band and options.resin_balance is ResinBalance.BALANCED:
        candidates_idx = in_target_band
    else:
        candidates_idx = list(range(len(all_rotations)))
    best_idx = min(
        candidates_idx,
        key=lambda i: (
            metrics[i].stability_score,
            metrics[i].overhang_score,
            metrics[i].height_mm,
        ),
    )
    rotation = all_rotations[best_idx]
    selected_metrics = metrics[best_idx]
    diagnostics: dict[str, float | str | bool | int] = {
        "candidate_profile": candidate_profile.value,
        "candidate_count": int(len(data.rotations)),
        "assembly_side_candidate_count": int(len(_assembly_side_rotations(assembly_side))),
        "assembly_side_detected": assembly_side is not None,
        "assembly_side_confidence": assembly_side.confidence if assembly_side is not None else 0.0,
        "assembly_side_area_ratio": assembly_side.area_ratio if assembly_side is not None else 0.0,
        "fitting_candidate_count": int(np.count_nonzero(data.fits)),
        "phase1_second_delta_ratio": _phase1_second_delta_ratio(data, total_area),
        "fit_margin_ratio": _fit_margin_ratio(
            (
                data.convex_hull_vertices
                if data.convex_hull_vertices is not None
                else np.empty((0, 3))
            ),
            rotation,
            printer_dims,
        ),
    }
    diagnostics["adaptive_trigger_reason"] = _adaptive_trigger_reason(
        selected_metrics,
        data,
        total_area,
        rotation,
        printer_dims,
    )
    return rotation, selected_metrics.overhang_score, selected_metrics, diagnostics


def find_stable_overhang_rotation(
    mesh: trimesh.Trimesh,
    overhang_threshold_deg: float = 45.0,
    n_candidates: int = 200,
    printer_dims: tuple[float, float, float] | None = None,
    *,
    support_tolerance_ratio: float = 0.20,
    resin_options: ResinOrientationOptions | None = None,
    source_up: np.ndarray | None = None,
) -> tuple[np.ndarray, float, OrientationStabilityMetrics]:
    """Find a stable support orientation while reusing candidate scoring data."""
    rotation, score, metrics, _diagnostics = _stable_overhang_rotation_with_diagnostics(
        mesh,
        overhang_threshold_deg=overhang_threshold_deg,
        n_candidates=n_candidates,
        printer_dims=printer_dims,
        support_tolerance_ratio=support_tolerance_ratio,
        resin_options=resin_options,
        source_up=source_up,
        candidate_profile=CandidateProfile.DEFAULT,
    )
    return rotation, score, metrics


def find_stable_overhang_rotation_adaptive(
    mesh: trimesh.Trimesh,
    overhang_threshold_deg: float = 45.0,
    n_candidates: int = 200,
    printer_dims: tuple[float, float, float] | None = None,
    *,
    support_tolerance_ratio: float = 0.20,
    resin_options: ResinOrientationOptions | None = None,
    source_up: np.ndarray | None = None,
) -> tuple[np.ndarray, float, OrientationStabilityMetrics, dict[str, float | str | bool | int]]:
    baseline_start = time.perf_counter()
    baseline_rotation, baseline_score, baseline_metrics, baseline_diag = (
        _stable_overhang_rotation_with_diagnostics(
            mesh,
            overhang_threshold_deg=overhang_threshold_deg,
            n_candidates=n_candidates,
            printer_dims=printer_dims,
            support_tolerance_ratio=support_tolerance_ratio,
            resin_options=resin_options,
            source_up=source_up,
            candidate_profile=CandidateProfile.DEFAULT,
        )
    )
    baseline_support_s = time.perf_counter() - baseline_start
    trigger = str(baseline_diag["adaptive_trigger_reason"])
    diagnostics: dict[str, float | str | bool | int] = {
        "adaptive_enabled": bool(trigger),
        "adaptive_reason": trigger,
        "baseline_support_s": baseline_support_s,
        "adaptive_support_s": 0.0,
        "candidate_count_default": int(baseline_diag["candidate_count"]),
        "candidate_count_adaptive": 0,
        "adaptive_accepted": False,
        "accepted": False,
    }
    if not trigger:
        return baseline_rotation, baseline_score, baseline_metrics, diagnostics

    adaptive_start = time.perf_counter()
    adaptive_rotation, adaptive_score, adaptive_metrics, adaptive_diag = (
        _stable_overhang_rotation_with_diagnostics(
            mesh,
            overhang_threshold_deg=overhang_threshold_deg,
            n_candidates=n_candidates,
            printer_dims=printer_dims,
            support_tolerance_ratio=support_tolerance_ratio,
            resin_options=resin_options,
            source_up=source_up,
            candidate_profile=CandidateProfile.ADAPTIVE,
        )
    )
    diagnostics["adaptive_support_s"] = time.perf_counter() - adaptive_start
    diagnostics["candidate_count_adaptive"] = int(adaptive_diag["candidate_count"])
    if _metrics_not_worse(adaptive_metrics, baseline_metrics):
        diagnostics["adaptive_accepted"] = True
        diagnostics["accepted"] = True
        return adaptive_rotation, adaptive_score, adaptive_metrics, diagnostics
    return baseline_rotation, baseline_score, baseline_metrics, diagnostics


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
