from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from stlbench.domain.printer import Printer


@dataclass
class Part:
    name: str
    mesh: trimesh.Trimesh
    source_path: Path | None = None
    _transforms: list[np.ndarray] = field(default_factory=list, repr=False)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path, name: str | None = None) -> Part:
        from stlbench.pipeline.mesh_io import load_mesh

        mesh = load_mesh(path)
        return cls(
            name=name or path.stem,
            mesh=mesh,
            source_path=path,
        )

    @classmethod
    def load_dir(
        cls,
        directory: Path,
        recursive: bool = False,
    ) -> list[Part]:
        from stlbench.pipeline.mesh_io import collect_mesh_paths, load_mesh_with_info

        paths = collect_mesh_paths(directory, recursive)
        parts = []
        for p in paths:
            mesh, _ = load_mesh_with_info(p)
            name = str(p.relative_to(directory)) if p.is_relative_to(directory) else p.name
            parts.append(cls(name=name, mesh=mesh, source_path=p))
        return parts

    # ── Geometry ─────────────────────────────────────────────────────────────

    @property
    def bounds(self) -> np.ndarray:
        """shape (2,3): [[xmin,ymin,zmin],[xmax,ymax,zmax]]"""
        return np.asarray(self.mesh.bounds)

    @property
    def extents(self) -> tuple[float, float, float]:
        """AABB (dx, dy, dz) in mm."""
        b = self.bounds
        d = b[1] - b[0]
        return float(d[0]), float(d[1]), float(d[2])

    @property
    def footprint_xy(self) -> tuple[float, float]:
        dx, dy, _ = self.extents
        return dx, dy

    @property
    def volume_mm3(self) -> float:
        return float(self.mesh.volume)

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces)

    # ── Transforms ────────────────────────────────────────────────────────────

    def apply_transform(self, t4: np.ndarray) -> Part:
        """Apply transform in-place and record in history. Returns self."""
        self.mesh.apply_transform(t4)
        self._transforms.append(t4.copy())
        return self

    def apply_scale(self, s: float) -> Part:
        self.mesh.apply_scale(s)
        return self

    def floor_z(self) -> Part:
        """Move part so its bottom is at Z=0."""
        z_min = float(self.bounds[0, 2])
        self.mesh.apply_translation([0.0, 0.0, -z_min])
        return self

    def clone(self) -> Part:
        """Deep copy — mesh and transform history."""
        new = Part(
            name=self.name,
            mesh=self.mesh.copy(),
            source_path=self.source_path,
            _transforms=list(self._transforms),
        )
        return new

    # ── Introspection ──────────────────────────────────────────────────────────

    def fits_printer(self, printer: Printer, eps: float = 0.0) -> bool:
        dx, dy, dz = self.extents
        return printer.fits_xyz(dx, dy, dz, eps)

    def warm_caches(self) -> None:
        """Initialize trimesh lazy caches before multithreaded work."""
        _ = self.mesh.face_normals
        _ = self.mesh.area_faces

    def __repr__(self) -> str:
        dx, dy, dz = self.extents
        return f"Part({self.name!r}, {dx:.1f}×{dy:.1f}×{dz:.1f} mm)"
