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


def int_bin_dims_mm(bed_w: float, bed_h: float) -> tuple[int, int]:
    return max(1, int(round(bed_w))), max(1, int(round(bed_h)))


def int_rect_dims_mm(fw: float, fh: float, gap_mm: float) -> tuple[int, int]:
    g = gap_mm
    return (
        max(1, int(round(fw + 2 * g))),
        max(1, int(round(fh + 2 * g))),
    )


def footprint_fits_bin_mm(fw: float, fh: float, bed_w: float, bed_h: float, gap_mm: float) -> bool:
    bw, bh = int_bin_dims_mm(bed_w, bed_h)
    w, h = int_rect_dims_mm(fw, fh, gap_mm)
    return (w <= bw and h <= bh) or (w <= bh and h <= bw)


def pack_rectangles_on_plates(
    footprints: list[tuple[float, float]],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    max_plates: int = 64,
) -> list[PackedPlate]:
    if bed_w <= 0 or bed_h <= 0:
        raise ValueError("bed dimensions must be positive.")
    g = gap_mm
    bw, bh = int_bin_dims_mm(bed_w, bed_h)

    orig_int_dims: dict[int, tuple[int, int]] = {}
    for idx, (fw, fh) in enumerate(footprints):
        orig_int_dims[idx] = int_rect_dims_mm(fw, fh, g)

    pending: list[int] = list(range(len(footprints)))
    plates_out: list[PackedPlate] = []
    plate_idx = 0

    while pending:
        if plate_idx >= max_plates:
            raise RuntimeError(f"Превышен max_plates={max_plates}; не все детали уместились.")
        packer = rectpack.newPacker(
            mode=rectpack.PackingMode.Offline,
            rotation=True,
        )
        packer.add_bin(bw, bh)
        fittable = [
            idx
            for idx in pending
            if footprint_fits_bin_mm(footprints[idx][0], footprints[idx][1], bed_w, bed_h, g)
        ]
        if not fittable:
            raise RuntimeError(
                "rectpack: ни одна из оставшихся деталей не помещается на стол по XY "
                f"(bin {bw}×{bh} мм при зазоре {g} мм). Уменьшите packing.gap_mm "
                "или scaling.supports_scale в конфиге."
            )
        for idx in fittable:
            w, h = orig_int_dims[idx]
            packer.add_rect(w, h, rid=idx)
        packer.pack()

        if len(packer) == 0:
            raise RuntimeError(
                "rectpack: нет bin после pack() — сообщите об этом кейсе "
                f"(bin {bw}×{bh}, деталей в очереди: {len(fittable)})."
            )

        b0 = packer[0]
        rects: list[PackedRect] = []
        placed_ids: set[int] = set()
        for r in b0:
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
                    x=float(r.x) + g,
                    y=float(r.y) + g,
                    width=float(r.width) - 2 * g,
                    height=float(r.height) - 2 * g,
                    rotated=was_rotated,
                )
            )

        if not placed_ids:
            raise RuntimeError(
                "Не удалось разместить ни одну деталь на пластине; проверьте габариты или gap_mm."
            )

        plates_out.append(PackedPlate(index=plate_idx, rects=tuple(rects)))
        plate_idx += 1
        pending = [i for i in pending if i not in placed_ids]

    return plates_out
