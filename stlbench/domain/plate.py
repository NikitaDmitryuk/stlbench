from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stlbench.packing.rectpack_plate import PackedPlate

    from .part import Part


@dataclass
class Plate:
    """Result of packing one plate with meshes and metadata."""

    index: int
    parts: list[Part]  # meshes in final position
    names: list[str]
    packed: PackedPlate  # placement coordinates

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def part_count(self) -> int:
        return len(self.parts)

    # ── Export ───────────────────────────────────────────────────────────────

    def export_3mf(self, path: Path, manifest_path: Path | None = None) -> None:
        from stlbench.export.plate import export_plate_3mf

        meshes = [p.mesh for p in self.parts]
        export_plate_3mf(meshes, self.packed, path, names=self.names, out_manifest=manifest_path)

    def export_stl(self, path: Path, manifest_path: Path | None = None) -> None:
        from stlbench.export.plate import export_plate_stl

        meshes = [p.mesh for p in self.parts]
        export_plate_stl(meshes, self.packed, path, out_manifest=manifest_path)
