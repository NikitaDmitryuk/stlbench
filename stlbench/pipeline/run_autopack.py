from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rectpack
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.fit import aabb_edge_lengths, compute_global_scale, printer_dims_with_margin
from stlbench.export.plate import export_plate_3mf
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.rectpack_plate import (
    PackedPlate,
    PackedRect,
    footprint_fits_bin_mm,
    int_bin_dims_mm,
    int_rect_dims_mm,
)
from stlbench.pipeline.common import (
    load_named_meshes,
    resolve_gap,
    resolve_printer,
    resolve_settings,
)


@dataclass
class AutopackRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    margin: float | None
    post_fit_scale: float | None
    dry_run: bool
    recursive: bool


def _try_pack_all(
    footprints: list[tuple[float, float]],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
) -> PackedPlate | None:
    """Try to pack all footprints onto a single plate. Returns None on failure."""
    bw, bh = int_bin_dims_mm(bed_w, bed_h)
    g = gap_mm

    for fw, fh in footprints:
        if not footprint_fits_bin_mm(fw, fh, bed_w, bed_h, g):
            return None

    packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline, rotation=True)
    packer.add_bin(bw, bh)
    int_dims: dict[int, tuple[int, int]] = {}
    for idx, (fw, fh) in enumerate(footprints):
        w, h = int_rect_dims_mm(fw, fh, g)
        int_dims[idx] = (w, h)
        packer.add_rect(w, h, rid=idx)
    packer.pack()

    if len(packer) == 0:
        return None

    placed_ids: set[int] = set()
    rects: list[PackedRect] = []
    for r in packer[0]:
        rid = getattr(r, "rid", None)
        if rid is None:
            continue
        idx = int(rid)
        placed_ids.add(idx)
        ow, oh = int_dims[idx]
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

    if len(placed_ids) < len(footprints):
        return None

    return PackedPlate(index=0, rects=tuple(rects))


def _bisect_scale(
    base_footprints: list[tuple[float, float]],
    bed_w: float,
    bed_h: float,
    gap_mm: float,
    s_upper: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> tuple[float, PackedPlate | None]:
    """Binary search for the maximum scale at which all parts fit on one plate."""
    lo, hi = 0.0, s_upper
    best_s = 0.0
    best_plate: PackedPlate | None = None

    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        scaled = [(fw * mid, fh * mid) for fw, fh in base_footprints]
        plate = _try_pack_all(scaled, bed_w, bed_h, gap_mm)
        if plate is not None:
            best_s = mid
            best_plate = plate
            lo = mid
        else:
            hi = mid

    return best_s, best_plate


def run_autopack(args: AutopackRunArgs) -> int:
    console = Console(stderr=True)
    st = resolve_settings(args.config_path)

    try:
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    gap = resolve_gap(args.gap_mm, st)
    margin = (
        float(args.margin) if args.margin is not None else (st.scaling.bed_margin if st else 0.0)
    )
    post_fit_scale = (
        float(args.post_fit_scale)
        if args.post_fit_scale is not None
        else (st.scaling.post_fit_scale if st else 1.0)
    )

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    _paths, names, meshes = loaded

    epx, epy, epz = printer_dims_with_margin(px, py, pz, margin)

    # Raw file dims (for display)
    dims_list: list[tuple[float, float, float]] = []
    for m in meshes:
        dims_list.append(aabb_edge_lengths(np.asarray(m.bounds)))

    # Find best print orientation per mesh (maximises scale, considers axis permutations).
    # This must be consistent with the footprints passed to _bisect_scale so that the
    # export uses the same orientation that the packer assumed.
    orient_transforms: list[np.ndarray] = []
    oriented_dims: list[tuple[float, float, float]] = []
    for m in meshes:
        t4, ext = select_orientation_for_scale(
            m,
            epx,
            epy,
            epz,
            "sorted",
            random_samples=ORIENTATION_SAMPLES_DEFAULT,
            seed=ORIENTATION_SEED_DEFAULT,
        )
        orient_transforms.append(t4)
        oriented_dims.append(ext)  # (ex, ey, ez) in printer coordinates

    s_upper, _ = compute_global_scale((epx, epy, epz), oriented_dims, names, "sorted")
    s_upper *= post_fit_scale

    # (ex, ey) is the XY footprint after the orientation transform
    base_footprints: list[tuple[float, float]] = [(ex, ey) for ex, ey, _ez in oriented_dims]

    s_best, plate = _bisect_scale(base_footprints, epx, epy, gap, s_upper)

    if plate is None or s_best <= 0:
        console.print("[red]Cannot fit all parts on one plate at any scale.[/red]")
        return 1

    console.print(f"Optimal scale (all parts on one plate): {s_best:.6f}")
    console.print(f"Parts: {len(names)}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("original (mm)", justify="right")
    table.add_column("scaled (mm)", justify="right")
    for name, d, od in zip(names, dims_list, oriented_dims, strict=True):
        orig = f"{d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f}"
        sd = tuple(x * s_best for x in od)
        scaled = f"{sd[0]:.2f} x {sd[1]:.2f} x {sd[2]:.2f}"
        table.add_row(name, orig, scaled)
    console.print(table)

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Apply orientation transform + scale; export_plate_stl handles translation to origin.
    scaled_meshes: list[trimesh.Trimesh] = []
    for m, t4 in zip(meshes, orient_transforms, strict=True):
        s = m.copy()
        s.apply_transform(t4)
        s.apply_scale(s_best)
        scaled_meshes.append(s)

    out_3mf = args.output_dir / "autopack_plate.3mf"
    out_json = args.output_dir / "autopack_plate.json"
    export_plate_3mf(scaled_meshes, plate, out_3mf, names=list(names), out_manifest=out_json)
    console.print(f"Wrote {out_3mf}")
    console.print(f"Wrote {out_json}")
    return 0
