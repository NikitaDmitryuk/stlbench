"""Job-file pipeline: per-part configurable steps (scale / orient / layout).

Each part in the job TOML can specify its own ``steps`` list, or inherit the
``[pipeline].default_steps`` setting.  All parts—regardless of what steps they
ran—are packed together onto the same build plates.

Algorithm overview
------------------
Pass 1 — scale preparation (parallel)
    For parts that include the ``scale`` step:
    * If ``scale`` comes before ``orient``: search for the best Z-axis
      orientation by default (controlled by ``[scaling] any_rotation`` and
      ``maximize`` in the job TOML; ``any_rotation=true`` enables full 3D
      search), then read the AABB extents of the (possibly rotated) mesh.
    * If ``orient`` comes before ``scale``: run ``find_min_overhang_rotation()``
      first, apply that orientation, then read the AABB extents of the oriented
      mesh.  No additional rotation search is needed.

Pass 2 — global scale
    Compute a single ``s_final`` that fits every scale-part inside the build
    volume.  Parts with ``steps=["layout"]`` keep their original size.

Pass 3 — apply pipeline (parallel)
    For each part apply its steps in order using the pre-computed data from
    Pass 1 and ``s_final`` from Pass 2.

Pass 4 — pack and export
    Compute XY footprints, pack onto plates, write 3MF + JSON manifests.
"""

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
from rich.table import Table

from stlbench.config.defaults import ORIENTATION_SAMPLES_DEFAULT, ORIENTATION_SEED_DEFAULT
from stlbench.config.loader import load_app_settings
from stlbench.config.schema import PartSpec, StepName
from stlbench.core.fit import compute_global_scale
from stlbench.core.mesh_cleanup import remove_small_components
from stlbench.core.overhang import apply_min_overhang_orientation, find_min_overhang_rotation
from stlbench.export.plate import export_plate_3mf
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.packing.polygon_footprint import mesh_to_xy_shadow
from stlbench.packing.polygon_pack import pack_polygons_on_plates
from stlbench.pipeline.common import n_workers
from stlbench.pipeline.mesh_io import load_mesh


@dataclass
class JobRunArgs:
    job_path: Path
    output_dir: Path
    n_orient_candidates: int = 200
    overhang_threshold_deg: float = 45.0
    rotation_samples: int = ORIENTATION_SAMPLES_DEFAULT
    grid_step_mm: float = 2.0
    verbose: bool = False
    dry_run: bool = False
    cleanup: bool = False


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _PartWork:
    """Working state for one part across all passes."""

    spec: PartSpec
    abs_path: Path
    steps: list[StepName]
    label: str  # display name (stem)

    # Filled by Pass 1 (for parts with scale step)
    pass1_mesh: trimesh.Trimesh | None = None  # mesh in scale/orient state after pass 1
    pass1_dims: tuple[float, float, float] | None = None  # AABB (dx, dy, dz) for s_max

    # Filled by Pass 3
    final_mesh: trimesh.Trimesh | None = None

    @property
    def has_scale(self) -> bool:
        return StepName.SCALE in self.steps

    @property
    def has_orient(self) -> bool:
        return StepName.ORIENT in self.steps

    @property
    def scale_before_orient(self) -> bool:
        """True when scale step precedes orient step (or orient is absent)."""
        if not self.has_scale:
            return False
        if not self.has_orient:
            return True
        return self.steps.index(StepName.SCALE) < self.steps.index(StepName.ORIENT)


# ---------------------------------------------------------------------------
# Progress helper
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
# Pass-1 workers
# ---------------------------------------------------------------------------


def _pass1_scale_first(
    pw: _PartWork,
    px: float,
    py: float,
    pz: float,
    method: str,
    any_rotation: bool,
    maximize: bool,
    rotation_samples: int,
) -> _PartWork:
    """Prepare mesh for global scale: search for best orientation (Z-only by default)."""
    mesh = load_mesh(pw.abs_path)
    t4, dims = select_orientation_for_scale(
        mesh,
        px,
        py,
        pz,
        method,  # type: ignore[arg-type]
        any_rotation=any_rotation,
        maximize=maximize,
        random_samples=rotation_samples,
        seed=ORIENTATION_SEED_DEFAULT,
    )
    mesh.apply_transform(t4)
    mesh.apply_translation([0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])])
    pw.pass1_mesh = mesh
    pw.pass1_dims = dims
    return pw


def _pass1_orient_first(
    pw: _PartWork,
    px: float,
    py: float,
    pz: float,
    overhang_threshold_deg: float,
    n_candidates: int,
) -> _PartWork:
    """Tweaker-3 orient, then derive scale dims from the oriented AABB."""
    mesh = load_mesh(pw.abs_path)
    _ = mesh.face_normals  # warm caches before any concurrent use
    _ = mesh.area_faces
    rotation, _ = find_min_overhang_rotation(
        mesh,
        overhang_threshold_deg=overhang_threshold_deg,
        n_candidates=n_candidates,
        printer_dims=(px, py, pz),
    )
    oriented = apply_min_overhang_orientation(mesh, rotation)
    b = np.asarray(oriented.bounds)
    dims = (float(b[1, 0] - b[0, 0]), float(b[1, 1] - b[0, 1]), float(b[1, 2] - b[0, 2]))
    pw.pass1_mesh = oriented  # already at z=0 from apply_min_overhang_orientation
    pw.pass1_dims = dims
    return pw


# ---------------------------------------------------------------------------
# Pass-3 workers
# ---------------------------------------------------------------------------


def _pass3_apply(
    pw: _PartWork,
    s_final: float,
    px: float,
    py: float,
    pz: float,
    overhang_threshold_deg: float,
    n_candidates: int,
) -> _PartWork:
    """Apply the remaining pipeline steps to produce the final mesh."""
    if pw.has_scale and pw.scale_before_orient:
        # Pass 1 already applied scale-orientation transform; apply scale now.
        mesh = pw.pass1_mesh
        assert mesh is not None
        mesh.apply_scale(s_final)
        mesh.apply_translation([0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])])
        if pw.has_orient:
            _ = mesh.face_normals
            _ = mesh.area_faces
            rotation, _ = find_min_overhang_rotation(
                mesh,
                overhang_threshold_deg=overhang_threshold_deg,
                n_candidates=n_candidates,
                printer_dims=(px, py, pz),
            )
            mesh = apply_min_overhang_orientation(mesh, rotation)

    elif pw.has_orient and not pw.scale_before_orient:
        # Pass 1 already applied orient; apply scale now if needed.
        mesh = pw.pass1_mesh
        assert mesh is not None
        if pw.has_scale:
            mesh.apply_scale(s_final)
            mesh.apply_translation([0.0, 0.0, -float(np.asarray(mesh.bounds)[0, 2])])

    else:
        # steps == ["orient", "layout"] or ["layout"]
        # orient-only: pass1 already ran orient; layout-only: load fresh
        mesh = pw.pass1_mesh if pw.pass1_mesh is not None else load_mesh(pw.abs_path)

    pw.final_mesh = mesh
    return pw


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_job(args: JobRunArgs) -> int:  # noqa: C901
    console = Console(stderr=True)

    # ── Load job file ───────────────────────────────────────────────────────
    try:
        settings = load_app_settings(args.job_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        return 2

    if not settings.parts:
        console.print("[red]Job file has no [[parts]] entries.[/red]")
        return 2

    px_raw = settings.printer.width_mm
    py_raw = settings.printer.depth_mm
    pz_raw = settings.printer.height_mm
    post_fit_scale = settings.scaling.post_fit_scale
    any_rotation = settings.scaling.any_rotation
    maximize = settings.scaling.maximize
    gap_mm = settings.packing.gap_mm
    method = "sorted"

    px, py, pz = px_raw, py_raw, pz_raw

    if settings.printer.name:
        console.print(f"Printer: {settings.printer.name}")
    console.print(f"Build volume: {px:.1f} × {py:.1f} × {pz:.1f} mm")
    console.print(f"Gap: {gap_mm} mm  |  post_fit_scale: {post_fit_scale}")

    default_steps = settings.pipeline.default_steps
    base_dir = args.job_path.parent

    # Build _PartWork list ─────────────────────────────────────────────────
    works: list[_PartWork] = []
    for spec in settings.parts:
        abs_path = (base_dir / spec.path).resolve()
        if not abs_path.exists():
            console.print(f"[red]Part not found: {abs_path}[/red]")
            return 2
        steps = spec.effective_steps(default_steps)
        works.append(
            _PartWork(
                spec=spec,
                abs_path=abs_path,
                steps=steps,
                label=abs_path.stem,
            )
        )

    console.print(f"\nParts: {len(works)}")
    for pw in works:
        step_str = " → ".join(s.value for s in pw.steps)
        console.print(f"  {pw.label}: [{step_str}]")

    # ── Pass 1: scale preparation (parallel) ───────────────────────────────
    scale_works = [pw for pw in works if pw.has_scale]
    orient_only_works = [pw for pw in works if pw.has_orient and not pw.has_scale]

    if scale_works or orient_only_works:
        console.print("\n[bold]Pass 1  Orientation / scale search[/bold]")

    all_pass1 = scale_works + orient_only_works
    if all_pass1:
        _n = n_workers(len(all_pass1))
        if args.verbose:
            console.print(f"[dim]pass1: {_n} workers for {len(all_pass1)} parts[/dim]")

        def _submit_pass1(pw: _PartWork):
            if pw.has_scale and pw.scale_before_orient:
                return _pass1_scale_first(
                    pw, px, py, pz, method, any_rotation, maximize, args.rotation_samples
                )
            else:
                # orient-first (either scale-after-orient or orient-only)
                return _pass1_orient_first(
                    pw, px, py, pz, args.overhang_threshold_deg, args.n_orient_candidates
                )

        with _make_progress(console) as progress:
            ptask = progress.add_task("Pass 1…", total=len(all_pass1))
            with ThreadPoolExecutor(max_workers=_n) as pool:
                futs = {pool.submit(_submit_pass1, pw): pw for pw in all_pass1}
                for fut in as_completed(futs):
                    try:
                        fut.result()  # result written in-place to pw
                    except Exception as e:
                        orig = futs[fut]
                        if args.verbose:
                            console.print_exception()
                        console.print(f"[red]Pass 1 failed for {orig.label!r}: {e}[/red]")
                        return 1
                    progress.advance(ptask)

    # ── Pass 2: global scale ────────────────────────────────────────────────
    s_final = 1.0
    if scale_works:
        console.print("\n[bold]Pass 2  Global scale[/bold]")
        scale_dims = [pw.pass1_dims for pw in scale_works]
        scale_names = [pw.label for pw in scale_works]
        assert all(d is not None for d in scale_dims)

        try:
            s_max, reports = compute_global_scale(
                (px, py, pz),
                scale_dims,  # type: ignore[arg-type]
                scale_names,
                method,  # type: ignore[arg-type]
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 1

        s_final = s_max * post_fit_scale
        console.print(f"s_max={s_max:.6f}  ×  post_fit={post_fit_scale}  →  s_final={s_final:.6f}")
        console.print(f"Limiting part: {reports[0].name}")

        table = Table(show_header=True, header_style="bold")
        table.add_column("part", max_width=42)
        table.add_column("scaled (mm)", justify="right")
        for r in reports:
            sd = (r.dx * s_final, r.dy * s_final, r.dz * s_final)
            table.add_row(r.name, f"{sd[0]:.2f} × {sd[1]:.2f} × {sd[2]:.2f}")
        console.print(table)

    # ── Pass 3: apply pipeline (parallel) ──────────────────────────────────
    console.print("\n[bold]Pass 3  Apply pipeline[/bold]")
    _n3 = n_workers(len(works))
    if args.verbose:
        console.print(f"[dim]pass3: {_n3} workers for {len(works)} parts[/dim]")

    def _submit_pass3(pw: _PartWork) -> _PartWork:
        return _pass3_apply(
            pw, s_final, px, py, pz, args.overhang_threshold_deg, args.n_orient_candidates
        )

    with _make_progress(console) as progress:
        ptask = progress.add_task("Applying pipeline…", total=len(works))
        with ThreadPoolExecutor(max_workers=_n3) as pool:
            futs3 = {pool.submit(_submit_pass3, pw): pw for pw in works}
            for fut3 in as_completed(futs3):
                try:
                    fut3.result()
                except Exception as e:
                    orig = futs3[fut3]
                    if args.verbose:
                        console.print_exception()
                    console.print(f"[red]Pass 3 failed for {orig.label!r}: {e}[/red]")
                    return 1
                progress.advance(ptask)

    final_meshes: list[trimesh.Trimesh] = [pw.final_mesh for pw in works]  # type: ignore[misc]
    final_names: list[str] = [pw.label for pw in works]

    if args.cleanup:
        for i, m in enumerate(final_meshes):
            cleaned, n_rem = remove_small_components(m)
            if n_rem:
                final_meshes[i] = cleaned
                console.print(
                    f"[dim]cleanup: {final_names[i]} — removed {n_rem} tiny component(s)[/dim]"
                )

    # ── Pass 4: pack and export ─────────────────────────────────────────────
    console.print("\n[bold]Pass 4  Layout[/bold]")

    # Pre-check: every part must fit the bed in at least one orientation.
    # Use a small tolerance (0.1 mm) to absorb floating-point drift after scaling.
    _FIT_EPS = 0.1
    for name, m in zip(final_names, final_meshes, strict=True):
        b = np.asarray(m.bounds)
        dx = float(b[1, 0] - b[0, 0])
        dy = float(b[1, 1] - b[0, 1])
        if not (
            (dx <= px + _FIT_EPS and dy <= py + _FIT_EPS)
            or (dy <= px + _FIT_EPS and dx <= py + _FIT_EPS)
        ):
            console.print(
                f"[red]Part {name!r} ({dx:.1f}×{dy:.1f} mm) does not fit "
                f"on bed {px:.1f}×{py:.1f} mm.[/red]"
            )
            return 1

    shadows = []
    with _make_progress(console) as progress:
        ptask = progress.add_task("Computing footprints…", total=len(final_meshes))
        for i, m in enumerate(final_meshes):
            shadows.append(mesh_to_xy_shadow(m))
            progress.update(ptask, advance=1, description=f"Footprint: {final_names[i]}")

    with _make_progress(console) as progress:
        ptask = progress.add_task("Packing…", total=len(final_meshes))
        plates = pack_polygons_on_plates(
            shadows,
            px,
            py,
            gap_mm=gap_mm,
            grid_step_mm=args.grid_step_mm,
            on_placed=lambda: progress.advance(ptask),
        )

    console.print(f"Plates: {len(plates)}")
    for pl in plates:
        console.print(f"  Plate {pl.index + 1}: {len(pl.rects)} parts")

    if args.dry_run:
        console.print("[dim]Dry-run: no files written.[/dim]")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def _export_plate(pl) -> Path:
        out_3mf = args.output_dir / f"plate_{pl.index + 1:02d}.3mf"
        out_js = args.output_dir / f"plate_{pl.index + 1:02d}.json"
        export_plate_3mf(final_meshes, pl, out_3mf, names=final_names, out_manifest=out_js)
        return out_3mf

    _np = n_workers(len(plates))
    with _make_progress(console) as progress:
        ptask = progress.add_task("Exporting plates…", total=len(plates))
        with ThreadPoolExecutor(max_workers=_np) as pool:
            futs_ex = [pool.submit(_export_plate, pl) for pl in plates]
            for fut_ex in as_completed(futs_ex):
                out_path = fut_ex.result()
                console.print(f"Wrote {out_path}")
                progress.advance(ptask)

    return 0
