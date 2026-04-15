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

Intermediate results
--------------------
When ``resume=True`` the pipeline checks ``output_dir/cache/meta.json`` for a
cached orient result.  If the input files' SHA-256 hashes match, steps 1–2 are
skipped and the previously oriented meshes are loaded from
``output_dir/cache/<stem>_oriented.stl``.  This lets you resume after a crash
without re-running the expensive orientation search.
"""

from __future__ import annotations

import hashlib
import json
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
from rich.table import Table

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.core.fit import compute_global_scale
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    overhang_score,
)
from stlbench.export.plate import export_plate_3mf
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.polygon_footprint import mesh_to_xy_shadow
from stlbench.packing.polygon_pack import pack_polygons_on_plates
from stlbench.pipeline.common import (
    load_named_meshes,
    n_workers,
    resolve_gap,
    resolve_printer,
    resolve_settings,
)
from stlbench.pipeline.mesh_io import load_mesh

_IDENTITY3 = np.eye(3, dtype=np.float64)


@dataclass
class PrepareRunArgs:
    input_dir: Path
    output_dir: Path
    config_path: Path | None
    printer_xyz: tuple[float, float, float] | None
    gap_mm: float | None
    post_fit_scale: float | None
    method: str | None
    overhang_threshold_deg: float
    n_orient_candidates: int
    dry_run: bool
    recursive: bool
    verbose: bool = False
    grid_step_mm: float = 2.0
    resume: bool = False
    cleanup: bool = False


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_META_NAME = "meta.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_orient_cache(
    cache_dir: Path,
    paths: list[Path],
    names: list[str],
    oriented_meshes: list[trimesh.Trimesh],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "input_files": {str(p): _file_sha256(p) for p in paths},
        "names": names,
    }
    mesh_files: list[str] = []
    for name, mesh in zip(names, oriented_meshes, strict=True):
        stem = Path(name).stem
        out = cache_dir / f"{stem}_oriented.stl"
        mesh.export(str(out))
        mesh_files.append(out.name)
    meta["mesh_files"] = mesh_files
    (cache_dir / _CACHE_META_NAME).write_text(json.dumps(meta, indent=2))


def _load_orient_cache(
    cache_dir: Path,
    paths: list[Path],
    console: Console,
) -> tuple[list[str], list[trimesh.Trimesh]] | None:
    meta_path = cache_dir / _CACHE_META_NAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    stored: dict[str, str] = meta.get("input_files", {})
    if set(stored.keys()) != {str(p) for p in paths}:
        return None
    for path in paths:
        if _file_sha256(path) != stored.get(str(path), ""):
            console.print("[dim]Cache miss: input file changed, re-running orient.[/dim]")
            return None

    names: list[str] = meta.get("names", [])
    mesh_files: list[str] = meta.get("mesh_files", [])
    oriented: list[trimesh.Trimesh] = []
    for fname in mesh_files:
        stl_path = cache_dir / fname
        if not stl_path.exists():
            return None
        oriented.append(load_mesh(stl_path))
    return names, oriented


# ---------------------------------------------------------------------------
# Progress bar factory
# ---------------------------------------------------------------------------


def _make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_prepare(args: PrepareRunArgs) -> int:  # noqa: C901
    console = Console(stderr=True)
    st = resolve_settings(args.config_path)

    try:
        px_raw, py_raw, pz_raw = resolve_printer(args.printer_xyz, st)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    post_fit_scale = (
        float(args.post_fit_scale)
        if args.post_fit_scale is not None
        else (st.scaling.post_fit_scale if st else 1.0)
    )
    gap = resolve_gap(args.gap_mm, st)
    method: str = args.method or "sorted"

    px, py, pz = px_raw, py_raw, pz_raw

    if st and st.printer.name:
        console.print(f"Printer: {st.printer.name}")
    console.print(f"Build volume: {px:.1f} × {py:.1f} × {pz:.1f} mm")
    console.print(f"Gap: {gap} mm  |  post_fit_scale: {post_fit_scale}")

    loaded = load_named_meshes(args.input_dir, args.recursive, console)
    if loaded is None:
        return 1
    _paths, names, meshes = loaded

    cache_dir = args.output_dir / "cache"

    # ──────────────────────────────────────────────────────────────────────────
    # Fast path: resume from orient cache
    # ──────────────────────────────────────────────────────────────────────────
    oriented_meshes: list[trimesh.Trimesh] | None = None
    if args.resume:
        cached = _load_orient_cache(cache_dir, _paths, console)
        if cached is not None:
            names, oriented_meshes = cached
            console.print(
                f"[green]Resumed from cache[/green] ({len(oriented_meshes)} meshes, "
                f"skipping scale + orient steps)."
            )

    if oriented_meshes is None:
        # ──────────────────────────────────────────────────────────────────────
        # Step 1 – Scale
        # ──────────────────────────────────────────────────────────────────────
        console.print("\n[bold]1 / 3  Scale[/bold]")

        def _select_orient(m: trimesh.Trimesh) -> tuple[np.ndarray, tuple[float, float, float]]:
            return select_orientation_for_scale(
                m,
                px,
                py,
                pz,
                method,  # type: ignore[arg-type]
                random_samples=ORIENTATION_SAMPLES_DEFAULT,
                seed=ORIENTATION_SEED_DEFAULT,
            )

        _n = n_workers(len(meshes))
        if args.verbose:
            console.print(f"[dim]scale-orient: {_n} workers for {len(meshes)} meshes[/dim]")

        _so: list = [None] * len(meshes)
        with _make_progress(console) as progress:
            ptask = progress.add_task("Finding scale orientations…", total=len(meshes))
            with ThreadPoolExecutor(max_workers=_n) as pool:
                futs_so = {pool.submit(_select_orient, m): i for i, m in enumerate(meshes)}
                for fut_so in as_completed(futs_so):
                    idx = futs_so[fut_so]
                    _so[idx] = fut_so.result()
                    progress.update(ptask, advance=1, description=f"Scale orient: {names[idx]}")

        scale_transforms = [r[0] for r in _so]
        oriented_dims = [r[1] for r in _so]

        try:
            s_max, reports = compute_global_scale((px, py, pz), oriented_dims, names, method)  # type: ignore[arg-type]
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 1

        s_final = s_max * post_fit_scale
        lim_name = reports[0].name
        console.print(f"s_max={s_max:.6f}  post_fit={post_fit_scale}  s_final={s_final:.6f}")
        console.print(f"Limiting part: {lim_name}")

        table = Table(show_header=True, header_style="bold")
        table.add_column("part", max_width=42)
        table.add_column("scaled (mm)", justify="right")
        for r in reports:
            sd = (r.dx * s_final, r.dy * s_final, r.dz * s_final)
            table.add_row(r.name, f"{sd[0]:.2f} × {sd[1]:.2f} × {sd[2]:.2f}")
        console.print(table)

        def _apply_scale(m_t4: tuple[trimesh.Trimesh, np.ndarray]) -> trimesh.Trimesh:
            m, t4 = m_t4
            m2 = m.copy()
            m2.apply_transform(t4)
            m2.apply_scale(s_final)
            m2.apply_translation([0.0, 0.0, -float(np.asarray(m2.bounds)[0, 2])])
            return m2

        scaled_meshes_list: list = [None] * len(meshes)
        with _make_progress(console) as progress:
            ptask = progress.add_task("Applying scale…", total=len(meshes))
            with ThreadPoolExecutor(max_workers=_n) as pool:
                futs_sc = {
                    pool.submit(_apply_scale, (m, t)): i
                    for i, (m, t) in enumerate(zip(meshes, scale_transforms, strict=True))
                }
                for fut_sc in as_completed(futs_sc):
                    idx = futs_sc[fut_sc]
                    scaled_meshes_list[idx] = fut_sc.result()
                    progress.update(ptask, advance=1, description=f"Scaling: {names[idx]}")
        scaled_meshes: list[trimesh.Trimesh] = scaled_meshes_list
        del meshes

        # ──────────────────────────────────────────────────────────────────────
        # Step 2 – Orient for minimum supports
        # ──────────────────────────────────────────────────────────────────────
        console.print(f"\n[bold]2 / 3  Orient[/bold]  (overhang ≥ {args.overhang_threshold_deg}°)")

        orient_table = Table(show_header=True, header_style="bold")
        orient_table.add_column("part", max_width=42)
        orient_table.add_column("before", justify="right")
        orient_table.add_column("after", justify="right")
        orient_table.add_column("Δ", justify="right")

        def _orient_one(mesh: trimesh.Trimesh) -> tuple[float, float, float, trimesh.Trimesh]:
            sb = overhang_score(mesh, _IDENTITY3, args.overhang_threshold_deg)
            rotation, sa = find_min_overhang_rotation(
                mesh,
                overhang_threshold_deg=args.overhang_threshold_deg,
                n_candidates=args.n_orient_candidates,
                printer_dims=(px, py, pz),
            )
            pct = (sb - sa) / max(abs(sb), 1.0) * 100.0
            return sb, sa, pct, apply_min_overhang_orientation(mesh, rotation)

        # Pre-populate trimesh lazy caches sequentially to prevent concurrent
        # cache-initialization races and to reduce peak memory during threading.
        for mesh in scaled_meshes:
            _ = mesh.face_normals
            _ = mesh.area_faces

        _n2 = n_workers(len(scaled_meshes))
        if args.verbose:
            console.print(f"[dim]orient: {_n2} workers for {len(scaled_meshes)} meshes[/dim]")
        _total = len(scaled_meshes)
        _or: list = [None] * _total

        with _make_progress(console) as progress:
            ptask = progress.add_task("Orienting parts…", total=_total)
            with ThreadPoolExecutor(max_workers=_n2) as pool:
                future_to_idx = {
                    pool.submit(_orient_one, scaled_meshes[i]): i for i in range(_total)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx.pop(future)
                    try:
                        _or[idx] = future.result()
                    except Exception as e:
                        if args.verbose:
                            console.print_exception()
                        console.print(
                            f"[red]orient failed for part {idx} ({names[idx]}): {e}[/red]"
                        )
                        return 1
                    scaled_meshes[idx] = None  # type: ignore[call-overload]  # release as soon as done
                    progress.update(ptask, advance=1, description=f"Oriented: {names[idx]}")
        del scaled_meshes

        oriented_meshes = []
        for name, (sb, sa, pct, oriented) in zip(names, _or, strict=True):
            orient_table.add_row(name, f"{sb:.1f}", f"{sa:.1f}", f"{pct:+.0f}%")
            oriented_meshes.append(oriented)
        del _or

        console.print(orient_table)

        # Save intermediate results so --resume can skip steps 1–2 next run.
        if not args.dry_run:
            try:
                _save_orient_cache(cache_dir, _paths, names, oriented_meshes)
                console.print(f"[dim]Orient cache saved → {cache_dir}[/dim]")
            except Exception as exc:
                console.print(f"[yellow]Warning: could not save orient cache: {exc}[/yellow]")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 – Layout
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold]3 / 3  Layout[/bold]")

    # Pre-check: each part must fit the bed in at least one orientation before packing.
    for name, m in zip(names, oriented_meshes, strict=True):
        b = np.asarray(m.bounds)
        dx = float(b[1, 0] - b[0, 0])
        dy = float(b[1, 1] - b[0, 1])
        if not ((dx <= px and dy <= py) or (dy <= px and dx <= py)):
            console.print(
                f"[red]Part {name!r} ({dx:.1f}×{dy:.1f} mm) does not fit "
                f"on bed {px:.1f}×{py:.1f} mm.[/red]"
            )
            return 1

    n_parts = len(oriented_meshes)

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
            grid_step_mm=args.grid_step_mm,
            on_placed=lambda: progress.advance(ptask),
        )

    console.print(f"Plates: {len(plates)}")
    for pl in plates:
        console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _export_plate(pl) -> Path:
        out_3mf = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
        out_js = args.output_dir / f"plate_{pl.index + 1:02d}.json"
        export_plate_3mf(oriented_meshes, pl, out_3mf, names=list(names), out_manifest=out_js)
        return out_3mf

    _np = n_workers(len(plates))
    if args.verbose:
        console.print(f"[dim]export: {_np} workers for {len(plates)} plates[/dim]")

    with _make_progress(console) as progress:
        ptask = progress.add_task("Exporting plates…", total=len(plates))
        with ThreadPoolExecutor(max_workers=_np) as pool:
            futs_ex = [pool.submit(_export_plate, pl) for pl in plates]
            for fut_ex in as_completed(futs_ex):
                out_path = fut_ex.result()
                console.print(f"Wrote {out_path}")
                progress.advance(ptask)

    return 0
