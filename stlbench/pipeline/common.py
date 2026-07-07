from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console

from stlbench.config.defaults import (
    DEFAULT_EDGE_MARGIN_MM,
    DEFAULT_PACKING_GAP_MM,
    DEFAULT_RESIN_BALANCE,
    ORIENTATION_POLICY_DEFAULT,
    ORIENTATION_SCALE_TOLERANCE_DEFAULT,
)
from stlbench.config.enums import OrientationPolicy, ResinBalance, coerce_enum
from stlbench.config.loader import load_app_settings
from stlbench.config.schema import AppSettings
from stlbench.core.mesh_repair import (
    RepairOptions,
    RepairReport,
    load_repair_cache,
    repair_cache_key,
    repair_mesh,
    write_repair_cache,
    write_repair_report,
)
from stlbench.core.overhang import ResinOrientationOptions
from stlbench.pipeline.mesh_io import SUPPORTED_EXTENSIONS, collect_mesh_paths, load_mesh_with_info
from stlbench.profiling import Profiler


def n_workers(n_items: int) -> int:
    """Number of ThreadPoolExecutor workers: min(n_items, ⌊cpu_count * 2/3⌋, 1).

    Caps at two-thirds of available logical CPUs so the system remains
    responsive while heavy geometry work is running.
    """
    cpu = os.cpu_count() or 2
    cap = max(1, int(cpu * 2 / 3))
    return min(n_items, cap)


def resolve_printer(
    printer_xyz: tuple[float, float, float] | None,
    settings: AppSettings | None,
) -> tuple[float, float, float]:
    if printer_xyz is not None:
        return printer_xyz
    if settings is not None:
        p = settings.printer
        return p.width_mm, p.depth_mm, p.height_mm
    raise ValueError("Set --printer Px,Py,Pz or use --config with a [printer] section.")


def resolve_settings(config_path: Path | None) -> AppSettings | None:
    if config_path is not None:
        return load_app_settings(config_path)
    return None


def resolve_repair_options(
    enabled: bool,
    settings: AppSettings | None,
) -> RepairOptions:
    if settings is None:
        return RepairOptions(enabled=enabled)
    source = settings.repair
    return RepairOptions(
        enabled=bool(enabled),
        close_holes=source.close_holes,
        max_hole_size_edges=source.max_hole_size_edges,
        repair_non_manifold=source.repair_non_manifold,
        remove_small_components=source.remove_small_components,
    )


def resolve_repair_cache_enabled(enabled: bool, settings: AppSettings | None) -> bool:
    return bool(enabled and (settings.repair.cache if settings is not None else True))


def resolve_gap(gap_mm: float | None, settings: AppSettings | None) -> float:
    if gap_mm is not None:
        return float(gap_mm)
    if settings is not None:
        return settings.packing.gap_mm
    return DEFAULT_PACKING_GAP_MM


def resolve_edge_margin(edge_margin_mm: float | None, settings: AppSettings | None) -> float:
    if edge_margin_mm is not None:
        return float(edge_margin_mm)
    if settings is not None:
        return settings.packing.edge_margin_mm
    return DEFAULT_EDGE_MARGIN_MM


def resolve_resin_orientation_options(
    resin_balance: str | None,
    settings: AppSettings | None,
) -> ResinOrientationOptions:
    if settings is not None:
        source = settings.orientation
        return ResinOrientationOptions(
            resin_balance=resin_balance or source.resin_balance,
            long_part_angle_policy=source.long_part_angle_policy,
            assembly_side_policy=source.assembly_side_policy,
            long_part_target_angle_min_deg=source.long_part_target_angle_min_deg,
            long_part_target_angle_max_deg=source.long_part_target_angle_max_deg,
            long_part_low_angle_penalty_below_deg=source.long_part_low_angle_penalty_below_deg,
            long_part_high_angle_penalty_above_deg=source.long_part_high_angle_penalty_above_deg,
        )
    return ResinOrientationOptions(
        resin_balance=coerce_enum(
            ResinBalance,
            resin_balance or DEFAULT_RESIN_BALANCE,
            "--resin-balance",
        )
    )


def resolve_orientation_policy(policy: str | None) -> OrientationPolicy:
    out = policy or ORIENTATION_POLICY_DEFAULT
    return coerce_enum(OrientationPolicy, out, "--orientation-policy")


def resolve_orientation_scale_tolerance(value: float | None) -> float:
    out = ORIENTATION_SCALE_TOLERANCE_DEFAULT if value is None else float(value)
    if not (0 < out <= 1):
        raise ValueError("--orientation-scale-tolerance must be in (0, 1].")
    return out


def rotation_to_4x4(r3: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.asarray(r3, dtype=np.float64)
    return t


def load_named_meshes(
    input_dir: Path,
    recursive: bool,
    console: Console,
    repair_options: RepairOptions | None = None,
) -> tuple[list[Path], list[str], list[trimesh.Trimesh]] | None:
    """Load all mesh files from *input_dir* and return (paths, names, meshes).

    Returns ``None`` and prints an error when no files are found or a file
    fails to load.
    """
    paths = collect_mesh_paths(input_dir, recursive)
    if not paths:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        console.print(f"[red]No mesh files ({exts}) found under {input_dir}[/red]")
        return None

    names: list[str] = []
    meshes: list[trimesh.Trimesh] = []
    for p in paths:
        try:
            m, has_multiple = load_mesh_with_info(p)
            if repair_options is not None:
                name = str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else p.name
                m, report = repair_mesh(
                    m,
                    repair_options,
                    source_path=p,
                    source_name=name,
                )
                if report.enabled and report.changed:
                    console.print(f"[dim]repair: {name} — mesh topology updated[/dim]")
        except (OSError, ValueError, TypeError) as e:
            console.print(f"[red]Failed to load {p}: {e}[/red]")
            return None
        name = str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else p.name
        if has_multiple:
            console.print(
                f"[yellow]Warning: {name!r} contains multiple surfaces — "
                f"model may be broken (surfaces merged for processing).[/yellow]"
            )
        names.append(name)
        meshes.append(m)
    return paths, names, meshes


def load_named_meshes_with_repair(
    input_dir: Path,
    recursive: bool,
    console: Console,
    repair_options: RepairOptions,
    repair_cache_dir: Path | None = None,
) -> tuple[list[Path], list[str], list[trimesh.Trimesh], list[RepairReport]] | None:
    paths = collect_mesh_paths(input_dir, recursive)
    if not paths:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        console.print(f"[red]No mesh files ({exts}) found under {input_dir}[/red]")
        return None

    names: list[str] = []
    meshes: list[trimesh.Trimesh] = []
    reports: list[RepairReport] = []
    for p in paths:
        name = str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else p.name
        try:
            cached = (
                load_repair_cache(
                    repair_cache_dir,
                    repair_cache_key(p, repair_options),
                    source_path=p,
                    source_name=name,
                )
                if repair_options.enabled and repair_cache_dir is not None
                else None
            )
            if cached is not None:
                m, report = cached
                has_multiple = False
            else:
                m, has_multiple = load_mesh_with_info(p)
                m, report = repair_mesh(m, repair_options, source_path=p, source_name=name)
                if repair_options.enabled and repair_cache_dir is not None:
                    key = repair_cache_key(p, repair_options)
                    report.cache_key = key
                    write_repair_cache(repair_cache_dir, key, m, report)
        except (OSError, ValueError, TypeError) as e:
            console.print(f"[red]Failed to load {p}: {e}[/red]")
            return None
        if has_multiple:
            console.print(
                f"[yellow]Warning: {name!r} contains multiple surfaces — "
                f"model may be broken (surfaces merged for processing).[/yellow]"
            )
        if report.cache_hit:
            console.print(f"[dim]repair cache: {name}[/dim]")
        elif report.enabled and report.changed:
            console.print(f"[dim]repair: {name} — mesh topology updated[/dim]")
        names.append(name)
        meshes.append(m)
        reports.append(report)
    return paths, names, meshes, reports


def load_mesh_with_repair(
    path: Path,
    repair_options: RepairOptions,
    *,
    source_name: str | None = None,
    repair_cache_dir: Path | None = None,
) -> tuple[trimesh.Trimesh, RepairReport]:
    if repair_options.enabled and repair_cache_dir is not None:
        key = repair_cache_key(path, repair_options)
        cached = load_repair_cache(
            repair_cache_dir,
            key,
            source_path=path,
            source_name=source_name or path.name,
        )
        if cached is not None:
            return cached
    mesh, _has_multiple = load_mesh_with_info(path)
    repaired, report = repair_mesh(
        mesh,
        repair_options,
        source_path=path,
        source_name=source_name or path.name,
    )
    if repair_options.enabled and repair_cache_dir is not None:
        key = repair_cache_key(path, repair_options)
        report.cache_key = key
        write_repair_cache(repair_cache_dir, key, repaired, report)
    return repaired, report


def repair_cache_dir_for_output(output_dir: Path, enabled: bool) -> Path | None:
    if not enabled:
        return None
    return output_dir / "cache" / "repair"


def write_command_repair_report(
    output_dir: Path,
    *,
    command: str,
    reports: list[RepairReport],
    dry_run: bool,
) -> None:
    if dry_run:
        return
    write_repair_report(output_dir / "repair_report.json", command=command, reports=reports)


def finish_profile(profiler: Profiler, console: Console, return_code: int) -> int:
    status = "ok" if return_code == 0 else "error"
    profiler.finish(status=status, return_code=return_code, console=console)
    return return_code
