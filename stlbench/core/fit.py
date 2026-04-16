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

    def compute_global_scale(
        self,
        parts_dims: list[tuple[float, float, float]],
        part_names: list[str],
        method: Method,
        file_dims: list[tuple[float, float, float]] | None = None,
    ) -> tuple[float, list[PartScaleReport]]:
        """Uniform scale across parts.

        For ``method="sorted"``, each ``parts_dims`` triple is **(dx, dy, dz) along printer
        X, Y, Z** (build height along Z). The per-part limit allows a 90° rotation of the
        XY footprint on the bed (see ``s_max_for_part_printer_axes``).
        """
        if len(parts_dims) != len(part_names):
            raise ValueError("parts_dims and part_names length mismatch.")
        if not parts_dims:
            raise ValueError("No parts to scale.")
        if file_dims is not None and len(file_dims) != len(parts_dims):
            raise ValueError("file_dims length must match parts_dims.")

        reports: list[PartScaleReport] = []
        limits: list[float] = []

        if method == "sorted":
            for i, (name, dims) in enumerate(zip(part_names, parts_dims, strict=True)):
                self._require_positive_dims(dims, name)
                s_lim, axis = self.s_max_for_part_printer_axes(*dims)
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
) -> tuple[float, list[PartScaleReport]]:
    """Uniform scale across parts.

    For ``method="sorted"``, each ``parts_dims`` triple is **(dx, dy, dz) along printer
    X, Y, Z** (build height along Z). The per-part limit allows a 90° rotation of the
    XY footprint on the bed (see ``s_max_for_part_printer_axes``).
    """
    calculator = FitCalculator(printer_xyz)
    return calculator.compute_global_scale(parts_dims, part_names, method, file_dims)


def limiting_part_index(reports: list[PartScaleReport], s_max: float) -> int:
    eps = 1e-12
    for i, r in enumerate(reports):
        if r.s_limit <= s_max + eps:
            return i
    return 0
