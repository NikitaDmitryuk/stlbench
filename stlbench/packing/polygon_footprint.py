from __future__ import annotations

import numpy as np
import shapely
import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

# Threshold above which we skip the union-of-triangles approach and use the
# convex hull of all XY vertices instead.  Random face subsampling fails for
# large meshes with non-uniform face density: e.g. a standing figure has most
# faces on nearly-vertical surfaces that project to zero-area 2D triangles,
# so the sampled union covers only a tiny patch instead of the full footprint.
_MAX_FACES_UNION = 20_000


def mesh_to_xy_shadow(mesh: trimesh.Trimesh, simplify_tol: float = 0.5) -> BaseGeometry:
    """Return the XY-plane shadow (top-view footprint) of *mesh* as a Shapely polygon.

    Computes the exact union of all face projections onto the XY plane using
    Shapely 2.0 vectorised operations.  Unlike the convex hull this preserves
    concave pockets (L-shapes, U-brackets, etc.) so the packer can interlock
    parts and achieve higher plate density.

    For meshes with more than ``_MAX_FACES_UNION`` faces, the convex hull of
    all XY vertices is used instead (random face subsampling gives incorrect
    results when face density is non-uniform).

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

    # Large meshes: use convex hull of ALL XY vertices.  Always correct and
    # O(n log n); the union-of-triangles path fails for non-uniform meshes.
    if len(mesh.faces) > _MAX_FACES_UNION:
        try:
            from scipy.spatial import ConvexHull  # noqa: PLC0415

            hull = ConvexHull(xy)
            shadow: BaseGeometry = Polygon(xy[hull.vertices])
        except Exception:
            shadow = MultiPoint(xy).convex_hull
        if simplify_tol > 0 and not shadow.is_empty:
            shadow = shadow.simplify(simplify_tol, preserve_topology=True)
        return shadow

    try:
        faces = mesh.faces

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

        shadow = shapely.union_all(polys)

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
