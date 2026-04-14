from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry


def mesh_to_xy_shadow(mesh: trimesh.Trimesh, simplify_tol: float = 1.0) -> BaseGeometry:
    """Return the XY-plane shadow (top-view footprint) of *mesh* as a Shapely polygon.

    Computes the **convex hull** of all vertex projections onto the XY plane.
    This is O(V log V) and handles meshes with millions of faces in milliseconds,
    versus the old triangle-union approach which was O(F) and took tens of seconds.

    Trade-off: the result is the convex hull of the footprint, so concave pockets
    (e.g. the inside of an L-shape) are not represented.  For typical resin-print
    parts this rarely matters — the convex envelope determines how much bed space
    a part occupies from above.

    Parameters
    ----------
    mesh:
        Mesh already oriented for printing (XY plane = build plate).
    simplify_tol:
        Douglas–Peucker tolerance in mm applied to the hull polygon.  Reduces
        vertex count without meaningful shape change.  Default 1.0 mm is
        appropriate for typical print beds (150–300 mm).  Pass 0 to disable.
    """
    xy: np.ndarray = mesh.vertices[:, :2]

    if len(xy) < 3:
        return MultiPoint(xy).convex_hull

    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415

        hull = ConvexHull(xy)
        shadow: BaseGeometry = Polygon(xy[hull.vertices])
    except Exception:
        # Degenerate geometry (e.g. all points collinear) — fall back to Shapely.
        shadow = MultiPoint(xy).convex_hull

    if simplify_tol > 0 and not shadow.is_empty:
        shadow = shadow.simplify(simplify_tol, preserve_topology=True)

    return shadow
