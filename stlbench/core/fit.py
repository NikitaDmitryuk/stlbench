from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Method = Literal["sorted", "conservative"]


@dataclass(frozen=True)
class PartScaleReport:
    name: str
    dx: float
    dy: float
    dz: float
    s_limit: float
    limiting_axis: str | None
    file_dx: float | None = None
    file_dy: float | None = None
    file_dz: float | None = None


class FitCalculator:
    """Calculate optimal scaling factors for 3D printing."""

    def __init__(self, printer_xyz: tuple[float, float, float]):
        self.printer_x, self.printer_y, self.printer_z = printer_xyz
        self._validate_printer_dims()

    def _validate_printer_dims(self) -> None:
        """Validate that printer dimensions are positive."""
        if min(self.printer_x, self.printer_y, self.printer_z) <= 0:
            raise ValueError("Printer dimensions must be positive.")

    def aabb_edge_lengths(self, bounds: np.ndarray) -> tuple[float, float, float]:
        """bounds: (2, 3) min/max corners."""
        if bounds.shape != (2, 3):
            raise ValueError("bounds must have shape (2, 3).")
        d = bounds[1] - bounds[0]
        return float(d[0]), float(d[1]), float(d[2])

    def _require_positive_dims(self, dims: tuple[float, float, float], label: str) -> None:
        if any(x <= 0 for x in dims):
            raise ValueError(f"{label}: all AABB edge lengths must be positive, got {dims}.")

    def s_max_for_part_sorted(
        self, p_sorted: tuple[float, float, float], d_sorted: tuple[float, float, float]
    ) -> float:
        self._require_positive_dims(d_sorted, "Part")
        p1, p2, p3 = p_sorted
        d1, d2, d3 = d_sorted
        return min(p1 / d1, p2 / d2, p3 / d3)

    def s_max_for_part_conservative(self, p_min: float, dx: float, dy: float, dz: float) -> float:
        dmax = max(dx, dy, dz)
        if dmax <= 0:
            raise ValueError(f"Part max dimension must be positive, got {dmax}.")
        return p_min / dmax

    def s_max_for_part_printer_axes(self, ex: float, ey: float, ez: float) -> tuple[float, str]:
        """
        Largest uniform scale *s* such that the axis-aligned AABB (ex,ey,ez) fits the
        printer box (px,py,pz): build height along *pz*/*ez*, and the XY footprint may
        be rotated 90° on the bed (swap which extent aligns with X vs Y).
        """
        self._require_positive_dims((ex, ey, ez), "Part")
        s_z = self.printer_z / ez
        s_a = min(self.printer_x / ex, self.printer_y / ey)
        s_b = min(self.printer_x / ey, self.printer_y / ex)
        s_xy = max(s_a, s_b)
        s_full = min(s_xy, s_z)
        eps = 1e-9
        xy_label = "xy_bed_swapped" if abs(s_xy - s_b) < eps and s_b >= s_a - eps else "xy_bed"
        if s_z < s_xy - eps:
            return s_full, "z_build_height"
        if s_xy < s_z - eps:
            return s_full, xy_label
        return s_full, f"{xy_label}_and_z"

    def s_max_for_part_with_z_rotation(self, ex: float, ey: float, ez: float) -> tuple[float, str]:
        """
        Largest uniform scale *s* considering both orientations around Z axis:
        1. Original orientation: ex along X, ey along Y
        2. Rotated 90° around Z: ey along X, ex along Y
        Returns the better scaling factor and orientation info.
        """
        self._require_positive_dims((ex, ey, ez), "Part")

        # Original orientation
        s_z1 = self.printer_z / ez
        s_xy1 = min(self.printer_x / ex, self.printer_y / ey)
        s1 = min(s_xy1, s_z1)

        # Rotated 90° around Z
        s_z2 = self.printer_z / ez
        s_xy2 = min(self.printer_x / ey, self.printer_y / ex)
        s2 = min(s_xy2, s_z2)

        # Choose better scale
        eps = 1e-9
        if s2 > s1 + eps:
            # Better with rotation
            s_full = s2
            xy_label = "xy_bed_swapped"
            if s_z2 < s_xy2 - eps:
                return s_full, "z_build_height"
            return s_full, xy_label
        else:
            # Better without rotation or equal
            s_full = s1
            xy_label = "xy_bed"
            if s_z1 < s_xy1 - eps:
                return s_full, "z_build_height"
            return s_full, xy_label

    def s_max_for_part_arbitrary_z_rotation(
        self, mesh_vertices: np.ndarray, samples: int = 360
    ) -> tuple[float, float, str]:
        """
        Find the optimal scale by rotating the part around Z axis with arbitrary angles.

        Args:
            mesh_vertices: Array of mesh vertices (n, 3)
            samples: Number of angle samples to test (default 360 for 1° resolution)

        Returns:
            tuple of (scale_factor, rotation_angle_rad, limiting_axis)
        """
        from ..core.orientation import aabb_extents_after_rotation

        best_scale = 0.0
        best_angle = 0.0
        best_axis_label = "xy_bed"

        # Test multiple rotation angles around Z axis
        for i in range(samples):
            angle_rad = (2.0 * np.pi * i) / samples
            # Create rotation matrix around Z axis
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array(
                [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )

            # Get extents after rotation
            ex, ey, ez = aabb_extents_after_rotation(mesh_vertices, rotation_matrix)

            # Calculate scale for this orientation
            s_z = self.printer_z / ez
            s_xy = min(self.printer_x / ex, self.printer_y / ey)
            s_current = min(s_xy, s_z)

            # Update best if this is better
            if s_current > best_scale:
                best_scale = s_current
                best_angle = angle_rad
                eps = 1e-9
                best_axis_label = "z_build_height" if s_z < s_xy - eps else "xy_bed"

        return best_scale, best_angle, best_axis_label

    def find_optimal_z_rotation_transform(
        self, mesh_vertices: np.ndarray, samples: int = 360
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """
        Find the optimal rotation around Z axis and return the transformation matrix and resulting extents.

        Args:
            mesh_vertices: Array of mesh vertices (n, 3)
            samples: Number of angle samples to test (default 360 for 1° resolution)

        Returns:
            tuple of (4x4 transformation matrix, (ex, ey, ez) extents)
        """
        from ..core.orientation import aabb_extents_after_rotation

        best_scale = 0.0
        best_angle = 0.0
        best_extents = (0.0, 0.0, 0.0)

        # Test multiple rotation angles around Z axis
        for i in range(samples):
            angle_rad = (2.0 * np.pi * i) / samples
            # Create rotation matrix around Z axis
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array(
                [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )

            # Get extents after rotation
            ex, ey, ez = aabb_extents_after_rotation(mesh_vertices, rotation_matrix)

            # Calculate scale for this orientation
            s_z = self.printer_z / ez
            s_xy = min(self.printer_x / ex, self.printer_y / ey)
            s_current = min(s_xy, s_z)

            # Update best if this is better
            if s_current > best_scale:
                best_scale = s_current
                best_angle = angle_rad
                best_extents = (ex, ey, ez)

        # Create transformation matrix for the best angle
        cos_a, sin_a = np.cos(best_angle), np.sin(best_angle)
        transform = np.array(
            [
                [cos_a, -sin_a, 0.0, 0.0],
                [sin_a, cos_a, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return transform, best_extents

    def compute_global_scale(
        self,
        parts_dims: list[tuple[float, float, float]],
        part_names: list[str],
        method: Method,
        file_dims: list[tuple[float, float, float]] | None = None,
        use_arbitrary_rotation: bool = False,
        mesh_vertices_list: list[np.ndarray] | None = None,
    ) -> tuple[float, list[PartScaleReport]]:
        """Uniform scale across parts.

        For ``method="sorted"``, each ``parts_dims`` triple is **(dx, dy, dz) along printer
        X, Y, Z** (build height along Z). The per-part limit allows a 90° rotation of the
        XY footprint on the bed (see ``s_max_for_part_printer_axes``).

        If use_arbitrary_rotation is True and mesh_vertices_list is provided, uses arbitrary
        rotation around Z axis for better fitting.
        """
        if len(parts_dims) != len(part_names):
            raise ValueError("parts_dims and part_names length mismatch.")
        if not parts_dims:
            raise ValueError("No parts to scale.")
        if file_dims is not None and len(file_dims) != len(parts_dims):
            raise ValueError("file_dims length must match parts_dims.")
        if (
            use_arbitrary_rotation
            and mesh_vertices_list is not None
            and len(mesh_vertices_list) != len(parts_dims)
        ):
            raise ValueError("mesh_vertices_list length must match parts_dims.")

        reports: list[PartScaleReport] = []
        limits: list[float] = []

        if method == "sorted":
            for i, (name, dims) in enumerate(zip(part_names, parts_dims, strict=True)):
                self._require_positive_dims(dims, name)
                if use_arbitrary_rotation and mesh_vertices_list is not None:
                    # Use arbitrary rotation around Z axis
                    s_lim, angle_rad, axis = self.s_max_for_part_arbitrary_z_rotation(
                        mesh_vertices_list[i]
                    )
                    # For reporting, we still use the original dims but note that rotation was applied
                    limits.append(s_lim)
                    fx = fy = fz = None
                    if file_dims is not None:
                        fx, fy, fz = file_dims[i]
                    reports.append(
                        PartScaleReport(
                            name=name,
                            dx=dims[0],
                            dy=dims[1],
                            dz=dims[2],
                            s_limit=s_lim,
                            limiting_axis=f"z_rotation_{angle_rad:.3f}rad",
                            file_dx=fx,
                            file_dy=fy,
                            file_dz=fz,
                        )
                    )
                else:
                    # Always check both Z-axis orientations even without allow_rotation flag
                    s_lim, axis = self.s_max_for_part_with_z_rotation(*dims)
                    limits.append(s_lim)
                    fx = fy = fz = None
                    if file_dims is not None:
                        fx, fy, fz = file_dims[i]
                    reports.append(
                        PartScaleReport(
                            name=name,
                            dx=dims[0],
                            dy=dims[1],
                            dz=dims[2],
                            s_limit=s_lim,
                            limiting_axis=axis,
                            file_dx=fx,
                            file_dy=fy,
                            file_dz=fz,
                        )
                    )
        elif method == "conservative":
            p_min = min(self.printer_x, self.printer_y, self.printer_z)
            for i, (name, dims) in enumerate(zip(part_names, parts_dims, strict=True)):
                s_lim = self.s_max_for_part_conservative(p_min, *dims)
                limits.append(s_lim)
                fx = fy = fz = None
                if file_dims is not None:
                    fx, fy, fz = file_dims[i]
                reports.append(
                    PartScaleReport(
                        name=name,
                        dx=dims[0],
                        dy=dims[1],
                        dz=dims[2],
                        s_limit=s_lim,
                        limiting_axis="max_extent_vs_min_printer",
                        file_dx=fx,
                        file_dy=fy,
                        file_dz=fz,
                    )
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        s_max = min(limits)
        return s_max, reports

    def limiting_part_index(self, reports: list[PartScaleReport], s_max: float) -> int:
        eps = 1e-12
        for i, r in enumerate(reports):
            if r.s_limit <= s_max + eps:
                return i
        return 0


# Backward compatibility functions
def aabb_edge_lengths(bounds: np.ndarray) -> tuple[float, float, float]:
    """bounds: (2, 3) min/max corners."""
    if bounds.shape != (2, 3):
        raise ValueError("bounds must have shape (2, 3).")
    d = bounds[1] - bounds[0]
    return float(d[0]), float(d[1]), float(d[2])


def _require_positive_dims(dims: tuple[float, float, float], label: str) -> None:
    if any(x <= 0 for x in dims):
        raise ValueError(f"{label}: all AABB edge lengths must be positive, got {dims}.")


def s_max_for_part_sorted(
    p_sorted: tuple[float, float, float], d_sorted: tuple[float, float, float]
) -> float:
    _require_positive_dims(d_sorted, "Part")
    p1, p2, p3 = p_sorted
    d1, d2, d3 = d_sorted
    return min(p1 / d1, p2 / d2, p3 / d3)


def s_max_for_part_conservative(p_min: float, dx: float, dy: float, dz: float) -> float:
    dmax = max(dx, dy, dz)
    if dmax <= 0:
        raise ValueError(f"Part max dimension must be positive, got {dmax}.")
    return p_min / dmax


def s_max_for_part_printer_axes(
    px: float, py: float, pz: float, ex: float, ey: float, ez: float
) -> tuple[float, str]:
    """
    Largest uniform scale *s* such that the axis-aligned AABB (ex,ey,ez) fits the
    printer box (px,py,pz): build height along *pz*/*ez*, and the XY footprint may
    be rotated 90° on the bed (swap which extent aligns with X vs Y).
    """
    _require_positive_dims((ex, ey, ez), "Part")
    s_z = pz / ez
    s_a = min(px / ex, py / ey)
    s_b = min(px / ey, py / ex)
    s_xy = max(s_a, s_b)
    s_full = min(s_xy, s_z)
    eps = 1e-9
    xy_label = "xy_bed_swapped" if abs(s_xy - s_b) < eps and s_b >= s_a - eps else "xy_bed"
    if s_z < s_xy - eps:
        return s_full, "z_build_height"
    if s_xy < s_z - eps:
        return s_full, xy_label
    return s_full, f"{xy_label}_and_z"


def compute_global_scale(
    printer_xyz: tuple[float, float, float],
    parts_dims: list[tuple[float, float, float]],
    part_names: list[str],
    method: Method,
    file_dims: list[tuple[float, float, float]] | None = None,
    use_arbitrary_rotation: bool = False,
    mesh_vertices_list: list[np.ndarray] | None = None,
) -> tuple[float, list[PartScaleReport]]:
    """Uniform scale across parts.

    For ``method="sorted"``, each ``parts_dims`` triple is **(dx, dy, dz) along printer
    X, Y, Z** (build height along Z). The per-part limit allows a 90° rotation of the
    XY footprint on the bed (see ``s_max_for_part_printer_axes``).

    If use_arbitrary_rotation is True and mesh_vertices_list is provided, uses arbitrary
    rotation around Z axis for better fitting.
    """
    calculator = FitCalculator(printer_xyz)
    return calculator.compute_global_scale(
        parts_dims, part_names, method, file_dims, use_arbitrary_rotation, mesh_vertices_list
    )


def limiting_part_index(reports: list[PartScaleReport], s_max: float) -> int:
    eps = 1e-12
    for i, r in enumerate(reports):
        if r.s_limit <= s_max + eps:
            return i
    return 0
