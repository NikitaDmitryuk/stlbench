from __future__ import annotations

import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def mesh_to_xy_shadow(mesh: trimesh.Trimesh, simplify_tol: float = 0.5) -> BaseGeometry:
    """Return the XY-plane shadow of *mesh* as a Shapely geometry.

    Projects every triangular face onto the XY plane and takes the union.
    The result is the exact footprint (not a convex hull), so concave shapes
    pack correctly — e.g. a small part can fit inside the concave pocket of
    an L-shaped model.

    Parameters
    ----------
    mesh:
        Mesh already oriented for printing (the XY plane is the build plate).
    simplify_tol:
        Douglas–Peucker tolerance in mm.  Reduces vertex count without
        significant shape distortion.  Pass 0 to disable.
    """
    tris = [Polygon(mesh.vertices[face, :2]) for face in mesh.faces]
    valid = [t for t in tris if t.is_valid and not t.is_empty]
    shadow = unary_union(valid) if valid else MultiPoint(mesh.vertices[:, :2]).convex_hull
    if simplify_tol > 0:
        shadow = shadow.simplify(simplify_tol, preserve_topology=True)
    return shadow
