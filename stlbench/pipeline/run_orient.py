from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console
from rich.table import Table

from stlbench.config.schema import AppSettings
from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    overhang_score,
)
from stlbench.pipeline.common import load_named_meshes, n_workers, resolve_printer, resolve_settings

_IDENTITY = np.eye(3, dtype=np.float64)


@dataclass
class OrientRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    settings: AppSettings | None
    printer_xyz: tuple[float, float, float] | None
    overhang_threshold_deg: float
    n_candidates: int
    dry_run: bool
    recursive: bool
    suffix: str
    verbose: bool = False


def run_orient(args: OrientRunArgs) -> int:
    console = Console(stderr=True)
    st = args.settings or resolve_settings(args.config_path)

    # Printer dims are optional for orient: when provided, orientations that
    # don't fit the build volume receive a heavy penalty.
    printer_dims: tuple[float, float, float] | None = None
    if args.printer_xyz is not None or st is not None:
        try:
            px, py, pz = resolve_printer(args.printer_xyz, st)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 2
        printer_dims = (px, py, pz)
        if st and st.printer.name:
            console.print(f"Printer: {st.printer.name}")
        console.print(f"Build volume: {px:.1f} × {py:.1f} × {pz:.1f} mm")

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    paths, names, meshes = loaded

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("score before", justify="right")
    table.add_column("score after", justify="right")
    table.add_column("improvement", justify="right")

    def _orient_one(mesh: trimesh.Trimesh) -> tuple[float, float, float, np.ndarray]:
        sb = overhang_score(mesh, _IDENTITY, args.overhang_threshold_deg)
        rotation, sa = find_min_overhang_rotation(
            mesh,
            overhang_threshold_deg=args.overhang_threshold_deg,
            n_candidates=args.n_candidates,
            printer_dims=printer_dims,
        )
        pct = (sb - sa) / max(abs(sb), 1.0) * 100.0
        return sb, sa, pct, rotation

    # Pre-populate trimesh lazy caches sequentially before threading.
    for mesh in meshes:
        _ = mesh.face_normals
        _ = mesh.area_faces

    _n = n_workers(len(meshes))
    if args.verbose:
        console.print(f"[dim]orient: {_n} workers for {len(meshes)} meshes[/dim]")
    with ThreadPoolExecutor(max_workers=_n) as pool:
        _or = list(pool.map(_orient_one, meshes))

    results = []
    for path, name, mesh, (sb, sa, pct, rotation) in zip(paths, names, meshes, _or, strict=True):
        table.add_row(name, f"{sb:.1f}", f"{sa:.1f}", f"{pct:+.1f}%")
        results.append((path, name, mesh, rotation))

    console.print(table)
    console.print(f"Overhang threshold: {args.overhang_threshold_deg}°")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _export_one(item: tuple[Path, str, trimesh.Trimesh, np.ndarray]) -> Path:
        path, _name, mesh, rotation = item
        if args.recursive:
            rel_parent = path.parent.relative_to(args.input_dir)
            out_sub = args.output_dir / rel_parent
            out_sub.mkdir(parents=True, exist_ok=True)
            out_dir = out_sub
        else:
            out_dir = args.output_dir
        out_path = out_dir / f"{path.stem + args.suffix}.stl"
        apply_min_overhang_orientation(mesh, rotation).export(out_path)
        return out_path

    try:
        _ne = n_workers(len(results))
        if args.verbose:
            console.print(f"[dim]export: {_ne} workers for {len(results)} meshes[/dim]")
        with ThreadPoolExecutor(max_workers=_ne) as pool:
            for out_path in pool.map(_export_one, results):
                console.print(f"Wrote {out_path}")
    except (OSError, ValueError) as e:
        if args.verbose:
            console.print_exception()
        console.print(f"[red]Failed to export: {e}[/red]")
        return 1

    return 0
