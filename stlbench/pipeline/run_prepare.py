"""Full preparation pipeline: scale → orient → layout, all in memory.

Order rationale
---------------
1. **Scale first** – maximises the model size by finding the global scale factor
   that fits every part inside the build volume (free-orientation search, same
   as ``scale --orientation free``).
2. **Orient for minimum supports** – rotates each *already-scaled* part to
   minimise overhang area, subject to the constraint that the part still fits
   inside the build volume in the new orientation.
3. **Layout** – packs the oriented parts onto the minimum number of plates as
   evenly as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.fit import (
    compute_global_scale,
    printer_dims_with_margin,
)
from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    overhang_score,
)
from stlbench.export.plate import export_plate_3mf
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.rectpack_plate import footprint_fits_bin_mm, pack_rectangles_on_plates
from stlbench.pipeline.common import (
    load_named_meshes,
    resolve_gap,
    resolve_printer,
    resolve_settings,
)

_IDENTITY3 = np.eye(3, dtype=np.float64)


@dataclass
class PrepareRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    margin: float | None
    post_fit_scale: float | None
    method: str | None
    overhang_threshold_deg: float
    n_orient_candidates: int
    dry_run: bool
    recursive: bool


def run_prepare(args: PrepareRunArgs) -> int:  # noqa: C901
    console = Console(stderr=True)
    st = resolve_settings(args.config_path)

    try:
        px_raw, py_raw, pz_raw = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    margin = (
        float(args.margin) if args.margin is not None else (st.scaling.bed_margin if st else 0.0)
    )
    post_fit_scale = (
        float(args.post_fit_scale)
        if args.post_fit_scale is not None
        else (st.scaling.post_fit_scale if st else 1.0)
    )
    gap = resolve_gap(args.gap_mm, st)
    method: str = args.method or "sorted"

    px, py, pz = printer_dims_with_margin(px_raw, py_raw, pz_raw, margin)

    if st and st.printer.name:
        console.print(f"Printer: {st.printer.name}")
    console.print(f"Build volume (after margin): {px:.1f} × {py:.1f} × {pz:.1f} mm")
    console.print(f"Gap: {gap} mm  |  post_fit_scale: {post_fit_scale}")

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    _paths, names, meshes = loaded

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 – Scale
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold]1 / 3  Scale[/bold]")

    scale_transforms: list[np.ndarray] = []
    oriented_dims: list[tuple[float, float, float]] = []
    for m in meshes:
        t4, ext = select_orientation_for_scale(
            m,
            px,
            py,
            pz,
            method,  # type: ignore[arg-type]
            random_samples=ORIENTATION_SAMPLES_DEFAULT,
            seed=ORIENTATION_SEED_DEFAULT,
        )
        scale_transforms.append(t4)
        oriented_dims.append(ext)

    try:
        s_max, reports = compute_global_scale((px, py, pz), oriented_dims, names, method)  # type: ignore[arg-type]
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    s_final = s_max * post_fit_scale
    lim_name = reports[0].name  # limiting part (sorted by s_limit ascending)
    console.print(f"s_max={s_max:.6f}  post_fit={post_fit_scale}  s_final={s_final:.6f}")
    console.print(f"Limiting part: {lim_name}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("scaled (mm)", justify="right")
    for r in reports:
        sd = (r.dx * s_final, r.dy * s_final, r.dz * s_final)
        table.add_row(r.name, f"{sd[0]:.2f} × {sd[1]:.2f} × {sd[2]:.2f}")
    console.print(table)

    scaled_meshes: list[trimesh.Trimesh] = []
    for m, t4 in zip(meshes, scale_transforms, strict=True):
        m2 = m.copy()
        m2.apply_transform(t4)
        m2.apply_scale(s_final)
        m2.apply_translation([0.0, 0.0, -float(np.asarray(m2.bounds)[0, 2])])
        scaled_meshes.append(m2)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 – Orient for minimum supports
    # ──────────────────────────────────────────────────────────────────────────
    console.print(f"\n[bold]2 / 3  Orient[/bold]  (overhang ≥ {args.overhang_threshold_deg}°)")

    orient_table = Table(show_header=True, header_style="bold")
    orient_table.add_column("part", max_width=42)
    orient_table.add_column("before", justify="right")
    orient_table.add_column("after", justify="right")
    orient_table.add_column("Δ", justify="right")

    oriented_meshes: list[trimesh.Trimesh] = []
    for name, mesh in zip(names, scaled_meshes, strict=True):
        console.print(f"  {name} …", highlight=False)
        score_before = overhang_score(mesh, _IDENTITY3, args.overhang_threshold_deg)
        rotation, score_after = find_min_overhang_rotation(
            mesh,
            overhang_threshold_deg=args.overhang_threshold_deg,
            n_candidates=args.n_orient_candidates,
            printer_dims=(px, py, pz),
        )
        pct = (score_before - score_after) / max(abs(score_before), 1.0) * 100.0
        orient_table.add_row(name, f"{score_before:.1f}", f"{score_after:.1f}", f"{pct:+.0f}%")
        oriented_meshes.append(apply_min_overhang_orientation(mesh, rotation))

    console.print(orient_table)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 – Layout
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold]3 / 3  Layout[/bold]")

    footprints: list[tuple[float, float]] = []
    for name, m in zip(names, oriented_meshes, strict=True):
        b = np.asarray(m.bounds)
        dx = float(b[1, 0] - b[0, 0])
        dy = float(b[1, 1] - b[0, 1])
        if not footprint_fits_bin_mm(dx, dy, px, py, gap):
            console.print(
                f"[red]Part {name!r} ({dx:.1f}×{dy:.1f} mm) does not fit "
                f"on bed {px:.1f}×{py:.1f} mm (gap {gap} mm).[/red]"
            )
            return 1
        footprints.append((dx, dy))

    plates = pack_rectangles_on_plates(footprints, px, py, gap_mm=gap)

    console.print(f"Plates: {len(plates)}")
    for pl in plates:
        console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pl in plates:
        out_3mf = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
        out_js = args.output_dir / f"plate_{pl.index + 1:02d}.json"
        export_plate_3mf(oriented_meshes, pl, out_3mf, names=list(names), out_manifest=out_js)
        console.print(f"Wrote {out_3mf}")

    return 0
