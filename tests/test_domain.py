from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stlbench.domain.part import Part
from stlbench.domain.printer import Printer
from stlbench.pipeline.mesh_io import load_mesh


@pytest.fixture
def sample_printer() -> Printer:
    return Printer(width_mm=200.0, depth_mm=150.0, height_mm=300.0, name="Test Printer")


@pytest.fixture
def sample_mesh_path() -> Path:
    # Using a simple cube mesh from the test assets
    return Path(__file__).parent / "assets" / "cube.stl"


@pytest.fixture
def sample_part(sample_mesh_path: Path) -> Part:
    mesh = load_mesh(sample_mesh_path)
    return Part(name="test_cube", mesh=mesh, source_path=sample_mesh_path)


class TestPrinter:
    def test_from_tuple(self):
        p = Printer.from_tuple((200, 150, 300), name="Saturn 4")
        assert p.width_mm == 200
        assert p.depth_mm == 150
        assert p.height_mm == 300
        assert p.name == "Saturn 4"
        assert str(p) == "Saturn 4 200.0×150.0×300.0 mm"

    def test_xyz_property(self):
        p = Printer(100, 80, 150)
        assert p.xyz == (100, 80, 150)

    def test_xy_property(self):
        p = Printer(100, 80, 150)
        assert p.xy == (100, 80)

    def test_sorted_dims(self):
        p = Printer(100, 150, 80)
        assert p.sorted_dims == (80, 100, 150)

    def test_bed_area_mm2(self):
        p = Printer(100, 80, 150)
        assert p.bed_area_mm2 == 8000

    def test_fits_xy_direct(self, sample_printer: Printer):
        # Fits directly
        assert sample_printer.fits_xy(190, 140)

    def test_fits_xy_rotated(self, sample_printer: Printer):
        # Fits when rotated 90°
        assert sample_printer.fits_xy(140, 190)

    def test_fits_xy_too_large(self, sample_printer: Printer):
        # Doesn't fit either way
        assert not sample_printer.fits_xy(210, 140)
        assert not sample_printer.fits_xy(140, 210)

    def test_fits_xyz(self, sample_printer: Printer):
        # Fits in all dimensions
        assert sample_printer.fits_xyz(190, 140, 290)
        # Too tall
        assert not sample_printer.fits_xyz(190, 140, 310)

    def test_validate_valid(self):
        p = Printer(100, 80, 150)
        # Should not raise
        p.validate()

    def test_validate_invalid(self):
        p = Printer(0, 80, 150)
        with pytest.raises(ValueError, match="Printer dimensions must be positive"):
            p.validate()


class TestPart:
    def test_load(self, sample_mesh_path: Path):
        part = Part.load(sample_mesh_path, name="custom_name")
        assert part.name == "custom_name"
        assert part.source_path == sample_mesh_path
        assert part.mesh is not None

    def test_extents(self, sample_part: Part):
        # Check that we get reasonable extents
        dx, dy, dz = sample_part.extents
        # All dimensions should be positive
        assert dx > 0
        assert dy > 0
        assert dz > 0
        # For our unit cube, dimensions should be around 1.0
        assert dx >= 1.0
        assert dy >= 1.0
        assert dz >= 1.0
        # Should be reasonable sizes for a 3D model (not huge)
        assert dx < 1000
        assert dy < 1000
        assert dz < 1000

    def test_footprint_xy(self, sample_part: Part):
        fw, fh = sample_part.footprint_xy
        dx, dy, _ = sample_part.extents
        assert fw == dx
        assert fh == dy

    def test_clone(self, sample_part: Part):
        clone = sample_part.clone()
        assert clone.name == sample_part.name
        assert clone.source_path == sample_part.source_path
        # Mesh should be a copy, not the same object
        assert clone.mesh is not sample_part.mesh
        # But should have the same vertices
        np.testing.assert_array_equal(clone.mesh.vertices, sample_part.mesh.vertices)

    def test_apply_transform(self, sample_part: Part):
        original_vertices = sample_part.mesh.vertices.copy()
        transform = np.eye(4)
        transform[0, 3] = 10  # Translate 10mm in X

        result = sample_part.apply_transform(transform)

        # Should return self
        assert result is sample_part
        # Should have recorded the transform
        assert len(sample_part._transforms) == 1
        np.testing.assert_array_equal(sample_part._transforms[0], transform)
        # Vertices should be different
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(sample_part.mesh.vertices, original_vertices)

    def test_apply_scale(self, sample_part: Part):
        original_vertices = sample_part.mesh.vertices.copy()

        result = sample_part.apply_scale(2.0)

        # Should return self
        assert result is sample_part
        # Vertices should be scaled
        np.testing.assert_array_almost_equal(sample_part.mesh.vertices, original_vertices * 2.0)

    def test_floor_z(self, sample_part: Part):
        # Get initial bounds
        initial_min_z = sample_part.bounds[0, 2]

        # Apply a transform to move it away from Z=0
        transform = np.eye(4)
        transform[2, 3] = 50  # Move 50mm up in Z
        sample_part.apply_transform(transform)

        # Get min Z before flooring (should be initial + 50)
        z_min_before = sample_part.bounds[0, 2]
        assert abs(z_min_before - (initial_min_z + 50)) < 0.1

        result = sample_part.floor_z()

        # Should return self
        assert result is sample_part
        # Min Z should now be close to 0
        z_min_after = sample_part.bounds[0, 2]
        assert abs(z_min_after) < 0.1

    def test_fits_printer(self, sample_part: Part, sample_printer: Printer):
        # Unit cube should fit in our test printer
        assert sample_part.fits_printer(sample_printer)

        # Apply a huge scale to make it not fit
        large_part = sample_part.clone()
        large_part.apply_scale(300)  # Scale to 300mm in each dimension

        assert not large_part.fits_printer(sample_printer)
