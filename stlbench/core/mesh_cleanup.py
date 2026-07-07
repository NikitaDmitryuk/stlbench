"""Mesh cleanup utilities for slicer-oriented mesh repair."""

from __future__ import annotations

import numpy as np
import trimesh


def remove_degenerate_faces(
    mesh: trimesh.Trimesh,
    area_epsilon: float = 1e-12,
) -> tuple[trimesh.Trimesh, int]:
    """Remove zero-area and near-zero-area triangles from *mesh*.

    Resin slicer hollowing can fail on triangles whose vertices repeat or are
    collinear.  This cleanup is deterministic and deliberately narrow: valid
    faces are preserved, removed faces are dropped, and newly unreferenced
    vertices are compacted away.
    """
    if len(mesh.faces) == 0:
        return mesh, 0

    try:
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
    except (ValueError, IndexError):
        return mesh, 0
    keep = np.isfinite(areas) & (areas > float(area_epsilon))
    removed = int(np.sum(~keep))
    if removed == 0:
        return mesh, 0

    cleaned = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[keep],
        process=False,
    )
    cleaned.remove_unreferenced_vertices()
    return cleaned, removed


def remove_small_components(
    mesh: trimesh.Trimesh,
    min_faces: int = 0,
) -> tuple[trimesh.Trimesh, int]:
    """Remove disconnected face components smaller than *min_faces*.

    If *min_faces* is 0 (default), the threshold is computed automatically as
    ``max(50, largest_component_faces // 1000)``.  This removes tiny floating
    artefacts (stray triangles, zero-volume double-faces, seam caps) that some
    mesh exporters leave behind and that can cause resin slicers to fail when
    trying to compute mesh volume.

    Parameters
    ----------
    mesh:
        Input mesh (not modified in-place).
    min_faces:
        Keep only components with at least this many faces.  Pass 0 for
        auto-threshold.

    Returns
    -------
    tuple[trimesh.Trimesh, int]
        ``(cleaned_mesh, n_removed)`` — cleaned mesh and the number of
        components that were dropped.
    """
    comps = trimesh.graph.connected_components(mesh.face_adjacency)
    if not comps:
        return mesh, 0

    sizes = [len(c) for c in comps]
    threshold = min_faces if min_faces > 0 else max(50, max(sizes) // 1000)

    largest_index = int(np.argmax(sizes))
    keep = [
        np.asarray(c)
        for i, (c, s) in enumerate(zip(comps, sizes, strict=True))
        if s >= threshold or i == largest_index
    ]
    n_removed = sum(1 for s in sizes if s < threshold)

    if n_removed == 0:
        return mesh, 0

    keep_face_ids = np.concatenate(keep)
    cleaned = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[keep_face_ids],
        process=False,
    )
    cleaned.remove_unreferenced_vertices()
    return cleaned, n_removed
