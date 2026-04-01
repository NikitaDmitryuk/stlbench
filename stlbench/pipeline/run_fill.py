from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rectpack
import trimesh
from rich.console import Console

from stlbench.core.fit import aabb_edge_lengths, compute_global_scale, printer_dims_with_margin
from stlbench.export.plate import mesh_footprint_xy
from stlbench.packing.layout_orientation import select_layout_transform
from stlbench.packing.rectpack_plate import (
    PackedPlate,
    PackedRect,
    footprint_fits_bin_mm,
    int_bin_dims_mm,
    int_rect_dims_mm,
)
from stlbench.pipeline.common import resolve_gap, resolve_printer, resolve_settings
from stlbench.pipeline.mesh_io import load_mesh


@dataclass
class FillRunArgs:
    input_file: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    scale: bool
    dry_run: bool


def _max_copies_on_plate(
    fw: float,
    fh: float,
    bed_w: float,
    bed_h: float,
    gap_mm: float,
) -> PackedPlate | None:
    """Pack as many identical rectangles (fw x fh) as possible onto one bin."""
    bw, bh = int_bin_dims_mm(bed_w, bed_h)
    rw, rh = int_rect_dims_mm(fw, fh, gap_mm)

    if not footprint_fits_bin_mm(fw, fh, bed_w, bed_h, gap_mm):
        return None

    max_possible = int((bw * bh) / max(1, rw * rh)) + 1
    max_possible = min(max_possible, 512)

    best_count = 0
    best_rects: list[PackedRect] = []

    lo, hi = 1, max_possible
    while lo <= hi:
        mid = (lo + hi) // 2
        packer = rectpack.newPacker(mode=rectpack.PackingMode.Offline, rotation=True)
        packer.add_bin(bw, bh)
        for i in range(mid):
            packer.add_rect(rw, rh, rid=i)
        packer.pack()

        if len(packer) == 0:
            hi = mid - 1
            continue

        placed = []
        for r in packer[0]:
            rid = getattr(r, "rid", None)
            if rid is None:
                continue
            placed_w = int(round(r.width))
            placed_h = int(round(r.height))
            was_rotated = (placed_w == rh and placed_h == rw) and (rw != rh)
            placed.append(
                PackedRect(
                    part_index=0,
                    x=float(r.x) + gap_mm,
                    y=float(r.y) + gap_mm,
                    width=float(r.width) - 2 * gap_mm,
                    height=float(r.height) - 2 * gap_mm,
                    rotated=was_rotated,
                )
            )

        if len(placed) >= mid:
            best_count = mid
            best_rects = placed
            lo = mid + 1
        else:
            if len(placed) > best_count:
                best_count = len(placed)
                best_rects = placed
            hi = mid - 1

    if best_count == 0:
        return None
    return PackedPlate(index=0, rects=tuple(best_rects[:best_count]))


_ROT_Z_90 = np.array(
    trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 0.0, 1.0]),
    dtype=np.float64,
)


def run_fill(args: FillRunArgs) -> int:
    console = Console(stderr=True)
    st = resolve_settings(args.config_path)

    try:
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    gap = resolve_gap(args.gap_mm, st)

    inp = args.input_file
    if inp.is_dir():
        stls = sorted(inp.glob("*.stl"))
        if len(stls) != 1:
            console.print(
                f"[red]fill expects exactly one STL file (found {len(stls)} in {inp}).[/red]"
            )
            return 2
        inp = stls[0]

    if not inp.is_file():
        console.print(f"[red]File not found: {inp}[/red]")
        return 2

    try:
        mesh = load_mesh(inp)
    except (OSError, ValueError, TypeError) as e:
        console.print(f"[red]Failed to load {inp}: {e}[/red]")
        return 1

    if args.scale:
        margin = st.scaling.bed_margin if st else 0.0
        epx, epy, epz = printer_dims_with_margin(px, py, pz, margin)
        dims = aabb_edge_lengths(np.asarray(mesh.bounds))
        s, _ = compute_global_scale((epx, epy, epz), [dims], [inp.name], "sorted")
        supports = st.scaling.supports_scale if st else 1.0
        s_final = s * supports
        mesh.apply_scale(s_final)
        console.print(f"Scaled {inp.name} by {s_final:.6f}")

    rot_samples = st.orientation.samples if st else 4096
    rot_seed = st.orientation.seed if st else 0

    ok, transform, fw, fh = select_layout_transform(
        mesh, px, py, pz, gap, random_samples=rot_samples, seed=rot_seed
    )
    if not ok:
        console.print("[red]Part does not fit on the bed in any orientation.[/red]")
        return 1

    mesh.apply_transform(transform)
    _, _, dz = mesh_footprint_xy(mesh)
    console.print(f"Part footprint: {fw:.2f} x {fh:.2f} mm, height: {dz:.2f} mm")

    plate = _max_copies_on_plate(fw, fh, px, py, gap)
    if plate is None:
        console.print("[red]Part does not fit on the bed.[/red]")
        return 1

    n = len(plate.rects)
    console.print(f"Copies that fit: {n}")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    placed: list[trimesh.Trimesh] = []
    for r in plate.rects:
        m = mesh.copy()
        m.apply_translation(
            [-float(m.bounds[0][0]), -float(m.bounds[0][1]), -float(m.bounds[0][2])]
        )
        if r.rotated:
            m.apply_transform(_ROT_Z_90)
            m.apply_translation([-float(m.bounds[0][0]), -float(m.bounds[0][1]), 0.0])
        m.apply_translation([r.x, r.y, 0.0])
        placed.append(m)

    combined = trimesh.util.concatenate(placed)
    out_stl = args.output_dir / "fill_plate.stl"
    out_json = args.output_dir / "fill_plate.json"
    combined.export(out_stl)

    manifest = {
        "source": inp.name,
        "copies": n,
        "bed_mm": [px, py, pz],
        "gap_mm": gap,
        "parts": [
            {
                "copy": i,
                "x_mm": r.x,
                "y_mm": r.y,
                "footprint_w_mm": r.width,
                "footprint_h_mm": r.height,
                "rotated_90": r.rotated,
            }
            for i, r in enumerate(plate.rects)
        ],
    }
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    console.print(f"Wrote {out_stl}  ({n} copies)")
    console.print(f"Wrote {out_json}")
    return 0
