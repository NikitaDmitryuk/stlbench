from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage
from shapely import affinity
from shapely.geometry.base import BaseGeometry

from stlbench.packing.rectpack_plate import PackedPlate, PackedRect

_BITMAP_ROTATION_COUNT = 24
_BITMAP_ANGLES: tuple[float, ...] = tuple(
    360.0 * i / _BITMAP_ROTATION_COUNT for i in range(_BITMAP_ROTATION_COUNT)
)


@dataclass(frozen=True)
class BitmapPackOptions:
    grid_mm: float = 0.25
    beam_width: int = 16


@dataclass(frozen=True)
class BitmapOrientation:
    part_index: int
    angle_deg: float
    mask: np.ndarray
    inflated_mask: np.ndarray
    pad_cells: int
    width_mm: float
    height_mm: float
    width_cells: int
    height_cells: int
    area_cells: int


@dataclass
class BitmapPackStats:
    grid_mm: float
    bed_cols: int
    bed_rows: int
    candidates_tested: int = 0
    fallback_scans: int = 0
    rasterize_s: float = 0.0
    search_s: float = 0.0
    orders_tried: int = 0
    placed_parts: int = 0
    scale_loss_estimate_mm: float = 0.0


@dataclass(frozen=True)
class BitmapPackResult:
    plate: PackedPlate | None
    stats: BitmapPackStats


@dataclass(frozen=True)
class _PlacedBitmap:
    part_index: int
    x_cell: int
    y_cell: int
    width_cells: int
    height_cells: int
    rect: PackedRect


def _normalize(poly: BaseGeometry) -> BaseGeometry:
    minx, miny, _, _ = poly.bounds
    return affinity.translate(poly, -minx, -miny)


def _disk_structure(radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=bool)
    ys, xs = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    return np.asarray((xs * xs + ys * ys) <= radius_cells * radius_cells, dtype=bool)


def _rasterize_polygon(poly: BaseGeometry, grid_mm: float) -> tuple[np.ndarray, float, float]:
    minx, miny, maxx, maxy = poly.bounds
    width = max(float(maxx - minx), grid_mm)
    height = max(float(maxy - miny), grid_mm)
    cols = max(1, int(math.ceil(width / grid_mm)))
    rows = max(1, int(math.ceil(height / grid_mm)))
    normalized = _normalize(poly)
    # Buffer by half a cell diagonal so a center-sampled mask conservatively
    # covers thin edges and sub-cell geometry.
    conservative = normalized.buffer(grid_mm * math.sqrt(2.0) * 0.5)
    xs = (np.arange(cols, dtype=np.float64) + 0.5) * grid_mm
    ys = (np.arange(rows, dtype=np.float64) + 0.5) * grid_mm
    xx, yy = np.meshgrid(xs, ys)
    mask = np.asarray(shapely.contains_xy(conservative, xx, yy), dtype=bool)
    if not mask.any():
        mask = np.ones((rows, cols), dtype=bool)
    return mask, width, height


def _rasterize_orientations(
    polygons: list[BaseGeometry],
    scale: float,
    options: BitmapPackOptions,
) -> list[list[BitmapOrientation]]:
    grid = options.grid_mm
    pad_cells = max(0, int(math.ceil(0.0 / grid)))
    out: list[list[BitmapOrientation]] = []
    for part_index, poly in enumerate(polygons):
        scaled = affinity.scale(poly, xfact=scale, yfact=scale, origin=(0.0, 0.0))
        part_orientations: list[BitmapOrientation] = []
        seen_shapes: set[tuple[int, int, int]] = set()
        for angle in _BITMAP_ANGLES:
            rotated = _normalize(affinity.rotate(scaled, angle, origin=(0.0, 0.0)))
            mask, width_mm, height_mm = _rasterize_polygon(rotated, grid)
            shape_key = (mask.shape[0], mask.shape[1], int(mask.sum()))
            if shape_key in seen_shapes:
                continue
            seen_shapes.add(shape_key)
            inflated_mask = mask
            part_orientations.append(
                BitmapOrientation(
                    part_index=part_index,
                    angle_deg=angle,
                    mask=mask,
                    inflated_mask=inflated_mask,
                    pad_cells=pad_cells,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    width_cells=mask.shape[1],
                    height_cells=mask.shape[0],
                    area_cells=int(mask.sum()),
                )
            )
        out.append(part_orientations)
    return out


def _inflate_orientations(
    orientations: list[list[BitmapOrientation]],
    gap_mm: float,
    grid_mm: float,
) -> list[list[BitmapOrientation]]:
    out: list[list[BitmapOrientation]] = []
    for part_orientations in orientations:
        inflated_part: list[BitmapOrientation] = []
        for orient in part_orientations:
            inflated_part.append(
                BitmapOrientation(
                    part_index=orient.part_index,
                    angle_deg=orient.angle_deg,
                    mask=orient.mask,
                    inflated_mask=orient.mask,
                    pad_cells=0,
                    width_mm=orient.width_mm,
                    height_mm=orient.height_mm,
                    width_cells=orient.width_cells,
                    height_cells=orient.height_cells,
                    area_cells=orient.area_cells,
                )
            )
        out.append(inflated_part)
    return out


def _part_orders(
    polygons: list[BaseGeometry], orientations: list[list[BitmapOrientation]]
) -> list[list[int]]:
    indices = list(range(len(polygons)))
    by_area = sorted(indices, key=lambda i: (-float(polygons[i].area), i))
    by_long_side = sorted(
        indices,
        key=lambda i: (
            -max(
                polygons[i].bounds[2] - polygons[i].bounds[0],
                polygons[i].bounds[3] - polygons[i].bounds[1],
            ),
            -float(polygons[i].area),
            i,
        ),
    )
    by_bitmap = sorted(
        indices,
        key=lambda i: (-max(o.area_cells for o in orientations[i]), i),
    )
    by_area_asc = list(reversed(by_area))
    by_long_side_asc = list(reversed(by_long_side))
    interleaved: list[int] = []
    left = 0
    right = len(by_area) - 1
    while left <= right:
        interleaved.append(by_area[left])
        if left != right:
            interleaved.append(by_area[right])
        left += 1
        right -= 1
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for order in (by_area, by_long_side, by_bitmap, interleaved, by_area_asc, by_long_side_asc):
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            out.append(order)
    return out


def _candidate_positions(
    orient: BitmapOrientation,
    placed: list[_PlacedBitmap],
    bed_cols: int,
    bed_rows: int,
    gap_cells: int,
    *,
    fallback_stride: int | None = None,
) -> list[tuple[int, int]]:
    max_x = bed_cols - orient.width_cells
    max_y = bed_rows - orient.height_cells
    if max_x < 0 or max_y < 0:
        return []
    if fallback_stride is not None:
        stride = max(1, fallback_stride)
        xs = list(range(0, max_x + 1, stride))
        ys = list(range(0, max_y + 1, stride))
        if xs[-1] != max_x:
            xs.append(max_x)
        if ys[-1] != max_y:
            ys.append(max_y)
        return [(x, y) for y in ys for x in xs]

    x_values = {0, max_x}
    y_values = {0, max_y}
    for item in placed:
        x_values.update(
            {
                item.x_cell,
                item.x_cell + item.width_cells + gap_cells,
                item.x_cell - orient.width_cells - gap_cells,
                item.x_cell + item.width_cells - orient.width_cells,
            }
        )
        y_values.update(
            {
                item.y_cell,
                item.y_cell + item.height_cells + gap_cells,
                item.y_cell - orient.height_cells - gap_cells,
                item.y_cell + item.height_cells - orient.height_cells,
            }
        )
    valid_x = sorted(x for x in x_values if 0 <= x <= max_x)
    valid_y = sorted(y for y in y_values if 0 <= y <= max_y)
    return [(x, y) for y in valid_y for x in valid_x]


def _mask_overlaps(occupied: np.ndarray, mask: np.ndarray, x_cell: int, y_cell: int) -> bool:
    h, w = mask.shape
    region = occupied[y_cell : y_cell + h, x_cell : x_cell + w]
    return bool(np.any(region & mask))


def _mask_has_clearance(
    distance_field: np.ndarray,
    mask: np.ndarray,
    x_cell: int,
    y_cell: int,
    gap_cells: int,
) -> bool:
    if gap_cells <= 0:
        return True
    h, w = mask.shape
    region = distance_field[y_cell : y_cell + h, x_cell : x_cell + w]
    if region.shape != mask.shape:
        return False
    return bool(np.all(region[mask] >= gap_cells))


def _stamp_inflated(
    occupied: np.ndarray, orient: BitmapOrientation, x_cell: int, y_cell: int
) -> None:
    pad = orient.pad_cells
    start_x = x_cell - pad
    start_y = y_cell - pad
    end_x = start_x + orient.inflated_mask.shape[1]
    end_y = start_y + orient.inflated_mask.shape[0]
    dst_x0 = max(0, start_x)
    dst_y0 = max(0, start_y)
    dst_x1 = min(occupied.shape[1], end_x)
    dst_y1 = min(occupied.shape[0], end_y)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return
    src_x0 = dst_x0 - start_x
    src_y0 = dst_y0 - start_y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    occupied[dst_y0:dst_y1, dst_x0:dst_x1] |= orient.inflated_mask[src_y0:src_y1, src_x0:src_x1]


def _try_place_part(
    part_index: int,
    orientations: list[list[BitmapOrientation]],
    occupied: np.ndarray,
    distance_field: np.ndarray,
    placed: list[_PlacedBitmap],
    grid_mm: float,
    gap_cells: int,
    stats: BitmapPackStats,
) -> tuple[BitmapOrientation, int, int] | None:
    bed_rows, bed_cols = occupied.shape
    best: tuple[tuple[int, int, int, int], BitmapOrientation, int, int] | None = None
    for orient in orientations[part_index]:
        for x_cell, y_cell in _candidate_positions(orient, placed, bed_cols, bed_rows, gap_cells):
            stats.candidates_tested += 1
            if _mask_overlaps(occupied, orient.mask, x_cell, y_cell):
                continue
            if not _mask_has_clearance(distance_field, orient.mask, x_cell, y_cell, gap_cells):
                continue
            score = (y_cell, x_cell, orient.height_cells, orient.width_cells)
            if best is None or score < best[0]:
                best = (score, orient, x_cell, y_cell)
                break
        if best is not None and best[0][0] == 0:
            break
    if best is not None:
        return best[1], best[2], best[3]

    fallback_stride = max(1, int(round(2.0 / grid_mm)))
    stats.fallback_scans += 1
    for orient in orientations[part_index]:
        for x_cell, y_cell in _candidate_positions(
            orient, placed, bed_cols, bed_rows, gap_cells, fallback_stride=fallback_stride
        ):
            stats.candidates_tested += 1
            if not _mask_overlaps(occupied, orient.mask, x_cell, y_cell) and _mask_has_clearance(
                distance_field, orient.mask, x_cell, y_cell, gap_cells
            ):
                return orient, x_cell, y_cell
    return None


def _pack_order(
    order: list[int],
    orientations: list[list[BitmapOrientation]],
    bed_cols: int,
    bed_rows: int,
    grid_mm: float,
    gap_cells: int,
    stats: BitmapPackStats,
) -> PackedPlate | None:
    occupied = np.zeros((bed_rows, bed_cols), dtype=bool)
    distance_field = np.full((bed_rows, bed_cols), float("inf"), dtype=np.float64)
    placed: list[_PlacedBitmap] = []
    rects: list[PackedRect] = []
    for part_index in order:
        result = _try_place_part(
            part_index,
            orientations,
            occupied,
            distance_field,
            placed,
            grid_mm,
            gap_cells,
            stats,
        )
        if result is None:
            return None
        orient, x_cell, y_cell = result
        _stamp_inflated(occupied, orient, x_cell, y_cell)
        rect = PackedRect(
            part_index=part_index,
            x=x_cell * grid_mm,
            y=y_cell * grid_mm,
            width=orient.width_mm,
            height=orient.height_mm,
            rotation_deg=orient.angle_deg,
        )
        rects.append(rect)
        if gap_cells > 0:
            distance_field = ndimage.distance_transform_edt(~occupied)
        placed.append(
            _PlacedBitmap(
                part_index=part_index,
                x_cell=x_cell,
                y_cell=y_cell,
                width_cells=orient.width_cells,
                height_cells=orient.height_cells,
                rect=rect,
            )
        )
    stats.placed_parts = max(stats.placed_parts, len(rects))
    return PackedPlate(index=0, rects=tuple(rects))


def pack_polygons_bitmap_single_plate(
    polygons: list[BaseGeometry],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    scale: float,
    options: BitmapPackOptions | None = None,
    validator: Callable[[PackedPlate], bool] | None = None,
) -> BitmapPackResult:
    options = options or BitmapPackOptions()
    if options.grid_mm <= 0:
        raise ValueError("bitmap grid_mm must be > 0")
    if options.beam_width < 1:
        raise ValueError("bitmap beam_width must be >= 1")
    grid = float(options.grid_mm)
    bed_cols = max(1, int(math.floor(bed_w / grid)))
    bed_rows = max(1, int(math.floor(bed_h / grid)))
    stats = BitmapPackStats(
        grid_mm=grid,
        bed_cols=bed_cols,
        bed_rows=bed_rows,
        scale_loss_estimate_mm=grid * 2.0,
    )
    start = time.perf_counter()
    orientations = _rasterize_orientations(polygons, scale, options)
    orientations = _inflate_orientations(orientations, gap_mm, grid)
    stats.rasterize_s = time.perf_counter() - start
    gap_cells = max(0, int(math.ceil(gap_mm / grid)))
    search_start = time.perf_counter()
    best_plate: PackedPlate | None = None
    for order in _part_orders(polygons, orientations)[: options.beam_width]:
        stats.orders_tried += 1
        plate = _pack_order(order, orientations, bed_cols, bed_rows, grid, gap_cells, stats)
        if plate is not None:
            if validator is not None and not validator(plate):
                continue
            best_plate = plate
            break
    stats.search_s = time.perf_counter() - search_start
    return BitmapPackResult(plate=best_plate, stats=stats)
