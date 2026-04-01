from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from rich.console import Console

from stlbench.config.loader import load_app_settings
from stlbench.config.schema import AppSettings
from stlbench.pipeline.mesh_io import collect_stl_paths, load_mesh


def resolve_printer(
    printer_xyz: tuple[float, float, float] | None,
    settings: AppSettings | None,
) -> tuple[float, float, float]:
    if printer_xyz is not None:
        return printer_xyz
    if settings is not None:
        p = settings.printer
        return p.width_mm, p.depth_mm, p.height_mm
    raise ValueError("Укажите printer или --config с [printer].")


def resolve_settings(config_path: Path | None) -> AppSettings | None:
    if config_path is not None:
        return load_app_settings(config_path)
    return None


def resolve_gap(gap_mm: float | None, settings: AppSettings | None) -> float:
    if gap_mm is not None:
        return float(gap_mm)
    if settings is not None:
        return settings.packing.gap_mm
    return 2.0


def resolve_algorithm(algorithm: str | None, settings: AppSettings | None) -> str:
    if algorithm is not None:
        return algorithm
    if settings is not None:
        return settings.packing.algorithm
    return "rectpack"


def rotation_to_4x4(r3: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.asarray(r3, dtype=np.float64)
    return t


def load_named_meshes(
    input_dir: Path,
    recursive: bool,
    console: Console,
) -> tuple[list[Path], list[str], list[trimesh.Trimesh]] | None:
    """Load all STL files from *input_dir* and return (paths, names, meshes).

    Returns ``None`` and prints an error when no files are found or a file
    fails to load.
    """
    paths = collect_stl_paths(input_dir, recursive)
    if not paths:
        console.print(f"[red]No .stl files under {input_dir}[/red]")
        return None

    names: list[str] = []
    meshes: list[trimesh.Trimesh] = []
    for p in paths:
        try:
            m = load_mesh(p)
        except (OSError, ValueError, TypeError) as e:
            console.print(f"[red]Failed to load {p}: {e}[/red]")
            return None
        names.append(str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else p.name)
        meshes.append(m)
    return paths, names, meshes
