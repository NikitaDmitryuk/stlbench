from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import trimesh

from stlbench.core.fit import Method, s_max_for_part_conservative, s_max_for_part_printer_axes
from stlbench.core.orientation import (
    _random_rotation_matrix,
    _z_rotation_candidates,
    mesh_vertices_for_orientation,
)
from stlbench.packing.rectpack_plate import footprint_fits_bin_mm

OrientationPolicy = Literal["max-scale", "printable"]

DEFAULT_ORIENTATION_SCALE_TOLERANCE = 0.98
OVERHANG_ANGLE_DEG = 45.0
LONG_PART_ASPECT = 3.0


@dataclass(frozen=True)
class ScoreComponents:
    height: float
    xy_area: float
    down_area: float
    center_z: float
    long_axis_vertical: float

    @property
    def total(self) -> float:
        return self.height + self.xy_area + self.down_area + self.center_z + self.long_axis_vertical


@dataclass(frozen=True)
class OrientationCandidate:
    transform: np.ndarray
    rotation: np.ndarray
    extents: tuple[float, float, float]
    scale_limit: float
    xy_area: float
    height: float
    pca_aspect: float
    long_axis_z: float
    down_area_ratio: float
    center_z_ratio: float
    score: ScoreComponents | None = None


class OrientationScorer(Protocol):
    def score(
        self,
        candidate: OrientationCandidate,
        printer_xyz: tuple[float, float, float],
    ) -> ScoreComponents: ...


class HeuristicPrintabilityScorer:
    """Fast resin-printability proxy; future support-aware scoring can plug in here."""

    def score(
        self,
        candidate: OrientationCandidate,
        printer_xyz: tuple[float, float, float],
    ) -> ScoreComponents:
        px, py, pz = printer_xyz
        bed_area = max(px * py, 1e-9)
        height_ratio = candidate.height / max(pz, 1e-9)
        xy_ratio = candidate.xy_area / bed_area
        center_ratio = candidate.center_z_ratio

        long_vertical = 0.0
        if candidate.pca_aspect >= LONG_PART_ASPECT:
            # Starts gently around 45° from horizontal and becomes strong near vertical.
            long_vertical = max(0.0, candidate.long_axis_z - 0.7) ** 2 * 12.0

        return ScoreComponents(
            height=0.80 * height_ratio,
            xy_area=0.70 * xy_ratio,
            down_area=1.40 * candidate.down_area_ratio,
            center_z=0.35 * center_ratio,
            long_axis_vertical=long_vertical,
        )


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
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )
    y_up4 = np.eye(4, dtype=np.float64)
    y_up4[:3, :3] = y_up
    rz = np.asarray(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    out: list[np.ndarray] = []
    for a in (i4, x_up, y_up4):
        out.append(a.copy())
        out.append(rz @ a)
    return out


def _perm_3x3_list() -> list[np.ndarray]:
    return [t[:3, :3].copy() for t in _axis_perm_4x4_list()]


def _rotation_to_4x4(r3: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.asarray(r3, dtype=np.float64)
    return t


def _principal_axis(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    centered = vertices - vertices.mean(axis=0)
    if centered.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 1.0
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    axis = vecs[:, -1]
    norm = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64) if norm <= 1e-12 else axis / norm
    aspect = float(np.sqrt(vals[-1] / max(vals[0], 1e-12))) if vals[-1] > 0 else 1.0
    return axis.astype(np.float64, copy=False), max(1.0, aspect)


def _face_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    areas = getattr(mesh, "area_faces", None)
    if areas is None:
        return np.zeros((len(mesh.faces),), dtype=np.float64)
    return np.asarray(areas, dtype=np.float64)


def _rotated_extents_chunked(
    vertices: np.ndarray,
    rotation: np.ndarray,
    *,
    chunk_size: int = 250_000,
) -> tuple[float, float, float]:
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start : start + chunk_size] @ rotation.T
        lo = np.minimum(lo, chunk.min(axis=0))
        hi = np.maximum(hi, chunk.max(axis=0))
    d = hi - lo
    return float(d[0]), float(d[1]), float(d[2])


def _candidate_rotations(
    *,
    any_rotation: bool,
    maximize: bool,
    random_samples: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if any_rotation:
        rng = np.random.default_rng(seed)
        perms = _perm_3x3_list()
        bases: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
        if maximize:
            for _ in range(max(0, random_samples)):
                bases.append(_random_rotation_matrix(rng))
        return bases, perms
    return _z_rotation_candidates(), [np.eye(3, dtype=np.float64)]


def generate_orientation_candidates(
    mesh: trimesh.Trimesh,
    printer_xyz: tuple[float, float, float],
    method: Method,
    *,
    any_rotation: bool = False,
    maximize: bool = False,
    random_samples: int = 4096,
    seed: int = 0,
    compute_printability_metrics: bool = True,
) -> list[OrientationCandidate]:
    verts = mesh_vertices_for_orientation(mesh)
    bases, perms = _candidate_rotations(
        any_rotation=any_rotation,
        maximize=maximize,
        random_samples=random_samples,
        seed=seed,
    )
    principal_axis, pca_aspect = _principal_axis(verts)
    normals = np.empty((0, 3), dtype=np.float64)
    areas = np.empty((0,), dtype=np.float64)
    total_area = 0.0
    center = verts.mean(axis=0)
    if compute_printability_metrics:
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        areas = _face_areas(mesh)
        total_area = float(np.sum(areas))
        center = np.asarray(mesh.centroid, dtype=np.float64)

    px, py, pz = printer_xyz
    p_min = min(px, py, pz)
    overhang_cos = float(np.cos(np.deg2rad(OVERHANG_ANGLE_DEG)))
    out: list[OrientationCandidate] = []

    for r in bases:
        for p in perms:
            r_tot = p @ r
            rverts = (r_tot @ verts.T).T
            lo = rverts.min(axis=0)
            hi = rverts.max(axis=0)
            d = hi - lo
            ex, ey, ez = float(d[0]), float(d[1]), float(d[2])
            if ex <= 0 or ey <= 0 or ez <= 0:
                continue
            if method == "sorted":
                scale_limit, _ = s_max_for_part_printer_axes(px, py, pz, ex, ey, ez)
            else:
                scale_limit = s_max_for_part_conservative(p_min, ex, ey, ez)

            down_area_ratio = 0.0
            if total_area > 0 and normals.size and areas.size == normals.shape[0]:
                rnormals = (r_tot @ normals.T).T
                down_weight = np.clip(
                    (-rnormals[:, 2] - overhang_cos) / (1.0 - overhang_cos),
                    0.0,
                    1.0,
                )
                down_area_ratio = float(np.sum(areas * down_weight) / total_area)

            c = r_tot @ center
            center_z_ratio = float(np.clip((c[2] - lo[2]) / max(ez, 1e-9), 0.0, 1.0))
            long_axis = r_tot @ principal_axis
            long_axis_z = float(abs(long_axis[2]) / max(np.linalg.norm(long_axis), 1e-12))

            out.append(
                OrientationCandidate(
                    transform=_rotation_to_4x4(r_tot),
                    rotation=r_tot.copy(),
                    extents=(ex, ey, ez),
                    scale_limit=scale_limit,
                    xy_area=ex * ey,
                    height=ez,
                    pca_aspect=pca_aspect,
                    long_axis_z=long_axis_z,
                    down_area_ratio=down_area_ratio,
                    center_z_ratio=center_z_ratio,
                )
            )

    return out


def _with_score(
    candidate: OrientationCandidate,
    scorer: OrientationScorer,
    printer_xyz: tuple[float, float, float],
) -> OrientationCandidate:
    return OrientationCandidate(
        transform=candidate.transform,
        rotation=candidate.rotation,
        extents=candidate.extents,
        scale_limit=candidate.scale_limit,
        xy_area=candidate.xy_area,
        height=candidate.height,
        pca_aspect=candidate.pca_aspect,
        long_axis_z=candidate.long_axis_z,
        down_area_ratio=candidate.down_area_ratio,
        center_z_ratio=candidate.center_z_ratio,
        score=scorer.score(candidate, printer_xyz),
    )


def select_orientation_candidate(
    candidates: list[OrientationCandidate],
    printer_xyz: tuple[float, float, float],
    *,
    policy: OrientationPolicy = "printable",
    scale_tolerance: float = DEFAULT_ORIENTATION_SCALE_TOLERANCE,
    scorer: OrientationScorer | None = None,
) -> OrientationCandidate:
    if not candidates:
        raise ValueError("No orientation candidates.")
    if not (0 < scale_tolerance <= 1):
        raise ValueError("scale_tolerance must be in (0, 1].")

    max_scale = max(c.scale_limit for c in candidates)
    if policy == "max-scale":
        return min(candidates, key=lambda c: (-c.scale_limit, c.xy_area, c.height))
    if policy != "printable":
        raise ValueError(f"Unknown orientation policy: {policy}")

    threshold = max_scale * scale_tolerance
    eligible = [c for c in candidates if c.scale_limit >= threshold]
    if not eligible:
        eligible = candidates
    use_scorer = scorer or HeuristicPrintabilityScorer()
    scored = [_with_score(c, use_scorer, printer_xyz) for c in eligible]
    return min(scored, key=lambda c: (c.score.total if c.score else 0.0, -c.scale_limit, c.xy_area))


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
    policy: OrientationPolicy = "printable",
    scale_tolerance: float = DEFAULT_ORIENTATION_SCALE_TOLERANCE,
) -> tuple[bool, np.ndarray, float, float]:
    candidates = generate_orientation_candidates(
        mesh,
        (bed_x, bed_y, pz),
        "sorted",
        any_rotation=any_rotation,
        maximize=True,
        random_samples=random_samples,
        seed=seed,
    )
    valid = [
        c
        for c in candidates
        if c.height <= pz + 1e-6
        and footprint_fits_bin_mm(c.extents[0], c.extents[1], bed_x, bed_y, gap_mm)
    ]
    if not valid:
        return False, np.eye(4, dtype=np.float64), 0.0, 0.0
    selected = select_orientation_candidate(
        valid,
        (bed_x, bed_y, pz),
        policy=policy,
        scale_tolerance=scale_tolerance,
    )
    return True, selected.transform.copy(), selected.extents[0], selected.extents[1]


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
    policy: OrientationPolicy = "printable",
    scale_tolerance: float = DEFAULT_ORIENTATION_SCALE_TOLERANCE,
    compute_printability_metrics: bool = True,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    candidates = generate_orientation_candidates(
        mesh,
        (px, py, pz),
        method,
        any_rotation=any_rotation,
        maximize=maximize,
        random_samples=random_samples,
        seed=seed,
        compute_printability_metrics=compute_printability_metrics,
    )
    selected = select_orientation_candidate(
        candidates,
        (px, py, pz),
        policy=policy,
        scale_tolerance=scale_tolerance,
    )
    exact_extents = _rotated_extents_chunked(
        np.asarray(mesh.vertices, dtype=np.float64),
        selected.rotation,
    )
    return selected.transform.copy(), exact_extents
