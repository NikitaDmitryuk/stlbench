"""Session-scoped fixtures: synthetic STL assets for job-runner tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh


def _write_box_stl(path: Path, extents: tuple[float, float, float]) -> None:
    mesh = trimesh.creation.box(extents=list(extents))
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


@pytest.fixture(scope="session")
def stl_assets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Return paths to three synthetic STL files (box meshes).

    sword  — 10×5×90 mm  (long, thin; tests rotation/orient)
    staff  — 8×8×150 mm  (very tall; tests height constraint)
    figure — 40×30×100 mm (compact block; tests full pipeline)

    All fit inside the 200×200×220 mm test printer volume.
    """
    base = tmp_path_factory.mktemp("stl_assets")
    assets: dict[str, Path] = {
        "sword": base / "sword.stl",
        "staff": base / "staff.stl",
        "figure": base / "figure.stl",
    }
    _write_box_stl(assets["sword"], (10.0, 5.0, 90.0))
    _write_box_stl(assets["staff"], (8.0, 8.0, 150.0))
    _write_box_stl(assets["figure"], (40.0, 30.0, 100.0))
    return assets
