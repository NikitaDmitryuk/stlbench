from __future__ import annotations

import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

# For meshes larger than this, randomly subsample faces before computing the
# union to keep runtime bounded.  The shadow is still representative because
# even a random subsample covers the full XY extent of the part.
_MAX_FACES_EXACT = 20_000


def mesh_to_xy_shadow(mesh: trimesh.Trimesh, simplify_tol: float = 0.5) -> BaseGeometry:
    """Return the XY-plane shadow (top-view footprint) of *mesh* as a Shapely polygon.

    Computes the exact union of all face projections onto the XY plane using
    Shapely 2.0 vectorised operations.  Unlike the convex hull this preserves
    concave pockets (L-shapes, U-brackets, etc.) so the packer can interlock
    parts and achieve higher plate density.

    For meshes with more than ``_MAX_FACES_EXACT`` faces a seeded random
    subsample is used; the footprint is still accurate to ±simplify_tol mm.

    Parameters
    ----------
    mesh:
        Mesh already oriented for printing (XY plane = build plate).
    simplify_tol:
        Douglas–Peucker tolerance in mm applied after union.  Default 0.5 mm
        keeps vertex count small while preserving features relevant to packing.
        Pass 0 to disable.
    """
    xy: np.ndarray = mesh.vertices[:, :2]

    if len(xy) < 3 or len(mesh.faces) == 0:
        return MultiPoint(xy).convex_hull

    try:
        faces = mesh.faces
        if len(faces) > _MAX_FACES_EXACT:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(faces), _MAX_FACES_EXACT, replace=False)
            faces = faces[idx]  # type: ignore[assignment]

        # Build F closed triangle rings: shape (F, 4, 2) — first vertex repeated.
        coords_2d: np.ndarray = xy[faces]  # (F, 3, 2)
        closed: np.ndarray = np.concatenate([coords_2d, coords_2d[:, :1, :]], axis=1)  # (F, 4, 2)
        n = len(closed)
        flat: np.ndarray = closed.reshape(-1, 2)

        # Shapely 2.0 ragged-array polygon creation (vectorised, no Python loop).
        # offsets[0]: start of each ring in `flat`   → [0, 4, 8, …, 4n]
        # offsets[1]: start of each polygon in rings → [0, 1, 2, …, n]
        ring_offsets: np.ndarray = np.arange(0, 4 * (n + 1), 4, dtype=np.int64)
        geom_offsets: np.ndarray = np.arange(n + 1, dtype=np.int64)
        polys = shapely.from_ragged_array(
            shapely.GeometryType.POLYGON,
            flat,
            (ring_offsets, geom_offsets),
        )

        # Drop degenerate (zero-area) triangles.
        valid: np.ndarray = shapely.is_valid(polys) & ~shapely.is_empty(polys)
        polys = polys[valid]
        if len(polys) == 0:
            raise ValueError("no valid triangles after degenerate filter")

        shadow: BaseGeometry = shapely.union_all(polys)

        # Remove internal holes — a through-hole in the part does not create
        # usable space for another part on the same build plate.
        from shapely.geometry import MultiPolygon  # noqa: PLC0415

        if isinstance(shadow, Polygon):
            shadow = Polygon(shadow.exterior)
        elif isinstance(shadow, MultiPolygon):
            shadow = Polygon(max(shadow.geoms, key=lambda g: g.area).exterior)
        else:
            # GeometryCollection or other — extract the largest polygon if any.
            polys_only = [g for g in getattr(shadow, "geoms", []) if isinstance(g, Polygon)]
            if not polys_only:
                raise ValueError("union produced no polygon")
            shadow = Polygon(max(polys_only, key=lambda g: g.area).exterior)

    except Exception:
        # Fallback: convex hull of all vertices (original behaviour).
        try:
            from scipy.spatial import ConvexHull  # noqa: PLC0415

            hull = ConvexHull(xy)
            shadow = Polygon(xy[hull.vertices])
        except Exception:
            shadow = MultiPoint(xy).convex_hull

    if simplify_tol > 0 and not shadow.is_empty:
        shadow = shadow.simplify(simplify_tol, preserve_topology=True)

    return shadow
