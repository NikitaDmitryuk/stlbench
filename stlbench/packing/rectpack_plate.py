from __future__ import annotations

from dataclasses import dataclass

import rectpack


@dataclass(frozen=True)
class PackedRect:
    part_index: int
    x: float
    y: float
    width: float
    height: float
    rotated: bool = False


@dataclass(frozen=True)
class PackedPlate:
    index: int
    rects: tuple[PackedRect, ...]


def int_bin_dims_mm(bed_w: float, bed_h: float, gap_mm: float = 0.0) -> tuple[int, int]:
    """Convert bed dimensions to integer pixel units, adding one gap to each axis.

    The extra ``gap_mm`` on each axis is the "trailing" gap budget: a model that
    exactly fills the bed (fw == bed_w) still has room for its trailing-edge gap,
    so the bin needs to be ``bed_w + gap_mm`` wide.
    """
    return max(1, int(round(bed_w + gap_mm))), max(1, int(round(bed_h + gap_mm)))


def int_rect_dims_mm(fw: float, fh: float, gap_mm: float) -> tuple[int, int]:
    """Rectangle size including one trailing gap on each axis.

    The gap is added only to the *trailing* side of each rectangle.  When two
    rectangles are packed adjacent to each other the trailing gap of the first
    becomes the space between them, giving exactly ``gap_mm`` clearance between
    any two parts.  At the leading edge (bin boundary) no gap is consumed, so
    parts can sit flush against the bed edge.
    """
    return (
        max(1, int(round(fw + gap_mm))),
        max(1, int(round(fh + gap_mm))),
    )


def footprint_fits_bin_mm(fw: float, fh: float, bed_w: float, bed_h: float, gap_mm: float) -> bool:
    bw, bh = int_bin_dims_mm(bed_w, bed_h, gap_mm)
    w, h = int_rect_dims_mm(fw, fh, gap_mm)
    return (w <= bw and h <= bh) or (w <= bh and h <= bw)


def _pack_subset(
    indices: list[int],
    orig_int_dims: dict[int, tuple[int, int]],
    bw: int,
    bh: int,
    gap_mm: float,
    plate_idx: int,
) -> tuple[PackedPlate, list[int]]:
    """Pack a set of parts onto one bin. Returns (plate, unplaced_indices)."""
    packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline, rotation=True)
    packer.add_bin(bw, bh)
    for idx in indices:
        w, h = orig_int_dims[idx]
        packer.add_rect(w, h, rid=idx)
    packer.pack()

    g = gap_mm
    placed_ids: set[int] = set()
    rects: list[PackedRect] = []

    if len(packer) > 0:
        for r in packer[0]:
            rid = getattr(r, "rid", None)
            if rid is None:
                continue
            idx = int(rid)
            placed_ids.add(idx)
            ow, oh = orig_int_dims[idx]
            placed_w = int(round(r.width))
            placed_h = int(round(r.height))
            was_rotated = (placed_w == oh and placed_h == ow) and (ow != oh)
            rects.append(
                PackedRect(
                    part_index=idx,
                    x=float(r.x),
                    y=float(r.y),
                    width=float(r.width) - g,
                    height=float(r.height) - g,
                    rotated=was_rotated,
                )
            )

    unplaced = [i for i in indices if i not in placed_ids]
    return PackedPlate(index=plate_idx, rects=tuple(rects)), unplaced


def _count_greedy_plates(
    fittable: list[int],
    orig_int_dims: dict[int, tuple[int, int]],
    bw: int,
    bh: int,
    gap_mm: float,
    max_plates: int,
) -> int:
    """Count the minimum number of plates needed using greedy fill-first packing."""
    pending = list(fittable)
    count = 0
    while pending:
        if count >= max_plates:
            raise RuntimeError(f"Exceeded max_plates={max_plates}.")
        plate, pending = _pack_subset(pending, orig_int_dims, bw, bh, gap_mm, count)
        if not plate.rects:
            raise RuntimeError(
                "rectpack: could not place any part on the plate; check part sizes or gap_mm."
            )
        count += 1
    return count


def _assign_balanced(
    indices: list[int],
    orig_int_dims: dict[int, tuple[int, int]],
    n_plates: int,
) -> list[list[int]]:
    """Assign parts to n_plates plates using LPT: largest footprint area first,
    each part goes to the plate with the smallest current total area load."""
    sorted_parts = sorted(
        indices,
        key=lambda i: orig_int_dims[i][0] * orig_int_dims[i][1],
        reverse=True,
    )
    plate_load = [0] * n_plates
    assignments: list[list[int]] = [[] for _ in range(n_plates)]
    for idx in sorted_parts:
        p = min(range(n_plates), key=lambda i: plate_load[i])
        w, h = orig_int_dims[idx]
        plate_load[p] += w * h
        assignments[p].append(idx)
    return assignments


def pack_rectangles_on_plates(
    footprints: list[tuple[float, float]],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    max_plates: int = 64,
) -> list[PackedPlate]:
    """Pack parts onto the minimum number of plates, distributing them evenly.

    Uses a two-phase approach:
    1. Greedy fill-first to determine the minimum number of plates needed.
    2. LPT (largest-first) balanced assignment across that many plates,
       then pack each plate independently with rectpack.
    """
    if bed_w <= 0 or bed_h <= 0:
        raise ValueError("bed dimensions must be positive.")
    g = gap_mm
    bw, bh = int_bin_dims_mm(bed_w, bed_h)

    orig_int_dims: dict[int, tuple[int, int]] = {}
    for idx, (fw, fh) in enumerate(footprints):
        orig_int_dims[idx] = int_rect_dims_mm(fw, fh, g)

    fittable = [
        idx
        for idx in range(len(footprints))
        if footprint_fits_bin_mm(footprints[idx][0], footprints[idx][1], bed_w, bed_h, g)
    ]
    if len(fittable) < len(footprints):
        raise RuntimeError(
            "rectpack: none of the remaining parts fit on the bed in XY "
            f"(bin {bw}x{bh} mm at gap {g} mm). Reduce packing.gap_mm "
            "or scaling.post_fit_scale in the config."
        )
    if not fittable:
        return []

    # Phase 1: find minimum N with greedy fill-first
    n_min = _count_greedy_plates(fittable, orig_int_dims, bw, bh, g, max_plates)

    if n_min == 1:
        plate, _ = _pack_subset(fittable, orig_int_dims, bw, bh, g, 0)
        return [plate] if plate.rects else []

    # Phase 2: distribute parts evenly across n_min plates using LPT
    assignments = _assign_balanced(fittable, orig_int_dims, n_min)

    # Phase 3: pack each assigned group independently
    plates_out: list[PackedPlate] = []
    overflow: list[int] = []

    for parts in assignments:
        if not parts:
            continue
        plate, unplaced = _pack_subset(parts, orig_int_dims, bw, bh, g, len(plates_out))
        overflow.extend(unplaced)
        if plate.rects:
            plates_out.append(plate)

    # Overflow (rare: only if LPT assigned too many large parts to one plate)
    while overflow:
        if len(plates_out) >= max_plates:
            raise RuntimeError(f"Exceeded max_plates={max_plates}; not all parts could be placed.")
        plate, overflow = _pack_subset(overflow, orig_int_dims, bw, bh, g, len(plates_out))
        if not plate.rects:
            raise RuntimeError(
                "rectpack: could not place overflow parts; check part sizes or gap_mm."
            )
        plates_out.append(plate)

    # Ensure sequential plate indices
    return [PackedPlate(index=i, rects=pl.rects) for i, pl in enumerate(plates_out)]
