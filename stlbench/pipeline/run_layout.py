from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.export.plate import export_plate_3mf, mesh_footprint_xy
from stlbench.packing.layout_orientation import select_layout_transform
from stlbench.packing.polygon_footprint import mesh_to_xy_shadow
from stlbench.packing.polygon_pack import pack_polygons_on_plates
from stlbench.packing.rectpack_plate import int_bin_dims_mm
from stlbench.pipeline.common import (
    load_named_meshes,
    n_workers,
    resolve_gap,
    resolve_printer,
    resolve_settings,
)


@dataclass
class LayoutRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    recursive: bool
    dry_run: bool
    cleanup: bool = False
    any_rotation: bool = False


def _make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def run_layout(args: LayoutRunArgs) -> int:
    console = Console(stderr=True)
    st = resolve_settings(args.config_path)

    try:
        px, py, pz = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    gap = resolve_gap(args.gap_mm, st)

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    _paths, names, meshes = loaded

    n_parts = len(meshes)
    dims_list: list[tuple[float, float, float]] = []
    for m in meshes:
        dx, dy, dz = mesh_footprint_xy(m)
        dims_list.append((dx, dy, dz))

    rot_samples = ORIENTATION_SAMPLES_DEFAULT
    rot_seed = ORIENTATION_SEED_DEFAULT

    bw, bh = int_bin_dims_mm(px, py)
    bad_layout: list[tuple[str, float, float, float]] = []
    layout_plans: list[tuple[np.ndarray, float, float] | None] = []

    with _make_progress(console) as progress:
        ptask = progress.add_task("Finding orientations…", total=n_parts)
        for m, name in zip(meshes, names, strict=True):
            dx, dy, dz = mesh_footprint_xy(m)
            ok, t, fw, fh = select_layout_transform(
                m,
                px,
                py,
                pz,
                gap,
                random_samples=rot_samples,
                seed=rot_seed,
                any_rotation=args.any_rotation,
            )
            if not ok:
                bad_layout.append((name, dx, dy, dz))
                layout_plans.append(None)
            else:
                layout_plans.append((t, fw, fh))
            progress.update(ptask, advance=1, description=f"Orient: {name}")

    if bad_layout:
        console.print(
            "[red]No bed-fitting orientation for these parts: 90° permutations + "
            f"{rot_samples} random rotations (same idea as scale; seed={rot_seed}). "
            f"Bed {bw}x{bh} mm XY, Pz={pz:.2f} mm, gap={gap:.2f} mm.[/red]"
        )
        for name, dx, dy, dz in bad_layout:
            console.print(f"  [red]{name}[/red]: file-axis AABB {dx:.2f}x{dy:.2f}x{dz:.2f} mm")
        console.print(
            "[dim]Try smaller packing.gap_mm / scaling.post_fit_scale or split the model.[/dim]"
        )
        return 1

    oriented_meshes: list[trimesh.Trimesh] = []
    for m, plan in zip(meshes, layout_plans, strict=True):
        assert plan is not None
        t, _fw, _fh = plan
        m2 = m.copy()
        m2.apply_transform(t)
        oriented_meshes.append(m2)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.cleanup:
        for i, m in enumerate(oriented_meshes):
            cleaned, n_rem = remove_small_components(m)
            if n_rem:
                oriented_meshes[i] = cleaned
                console.print(f"[dim]cleanup: {names[i]} — removed {n_rem} tiny component(s)[/dim]")

    shadows = []
    with _make_progress(console) as progress:
        ptask = progress.add_task("Computing footprints…", total=n_parts)
        for i, m in enumerate(oriented_meshes):
            shadows.append(mesh_to_xy_shadow(m))
            progress.update(ptask, advance=1, description=f"Footprint: {names[i]}")

    with _make_progress(console) as progress:
        ptask = progress.add_task("Packing…", total=n_parts)
        plates = pack_polygons_on_plates(
            shadows,
            px,
            py,
            gap_mm=gap,
            on_placed=lambda: progress.advance(ptask),
        )

    console.print(f"Plates: {len(plates)}")
    for pl in plates:
        console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")

    if args.dry_run:
        return 0

    def _export_plate(pl) -> Path:
        out_3mf = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
        out_js = args.output_dir / f"plate_{pl.index + 1:02d}.json"
        export_plate_3mf(oriented_meshes, pl, out_3mf, names=list(names), out_manifest=out_js)
        return out_3mf

    _np = n_workers(len(plates))
    with _make_progress(console) as progress:
        ptask = progress.add_task("Exporting plates…", total=len(plates))
        with ThreadPoolExecutor(max_workers=_np) as pool:
            futs = [pool.submit(_export_plate, pl) for pl in plates]
            for fut in as_completed(futs):
                out_path = fut.result()
                console.print(f"Wrote {out_path}")
                progress.advance(ptask)

    return 0
