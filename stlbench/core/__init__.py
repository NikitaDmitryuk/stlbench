"""Core scaling and orientation math."""

from stlbench.core.fit import (
    Method,
    PartScaleReport,
    aabb_edge_lengths,
    compute_global_scale,
    limiting_part_index,
    printer_dims_with_margin,
    s_max_for_part_conservative,
    s_max_for_part_sorted,
)
from stlbench.core.orientation import (
    best_aabb_extents_for_conservative_fit,
    best_aabb_extents_for_sorted_fit,
    mesh_vertices_for_orientation,
)

__all__ = [
    "Method",
    "PartScaleReport",
    "aabb_edge_lengths",
    "compute_global_scale",
    "limiting_part_index",
    "printer_dims_with_margin",
    "s_max_for_part_conservative",
    "s_max_for_part_sorted",
    "best_aabb_extents_for_conservative_fit",
    "best_aabb_extents_for_sorted_fit",
    "mesh_vertices_for_orientation",
]
