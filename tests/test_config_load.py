from pathlib import Path

import pytest
from pydantic import ValidationError

from stlbench.config.enums import (
    AssemblySidePolicy,
    LongPartAnglePolicy,
    PackerBackend,
    ResinBalance,
)
from stlbench.config.loader import load_app_settings, load_config


def test_load_mars5_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "mars5_ultra.toml")
    assert "Mars" in cfg.printer_name
    assert cfg.width_mm > 0
    assert cfg.post_fit_scale == 0.95
    assert cfg.settings.packing.gap_mm == 10.0
    assert cfg.settings.orientation.long_part_angle_policy is LongPartAnglePolicy.THIN_LINEAR


def test_toml_mode_strings_parse_to_enums(tmp_path: Path):
    config_path = tmp_path / "profile.toml"
    config_path.write_text(
        """
[printer]
width_mm = 100
depth_mm = 100
height_mm = 100

[orientation]
resin_balance = "compact"
long_part_angle_policy = "linear"
assembly_side_policy = "disabled"

[autopack]
packer = "bitmap"
""".strip(),
        encoding="utf-8",
    )

    settings = load_app_settings(config_path)

    assert settings.orientation.resin_balance is ResinBalance.COMPACT
    assert settings.orientation.long_part_angle_policy is LongPartAnglePolicy.LINEAR
    assert settings.orientation.assembly_side_policy is AssemblySidePolicy.DISABLED
    assert settings.autopack.packer is PackerBackend.BITMAP


def test_invalid_assembly_side_policy_is_rejected(tmp_path: Path):
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[printer]
width_mm = 100
depth_mm = 100
height_mm = 100

[orientation]
assembly_side_policy = "always"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_app_settings(config_path)


def test_invalid_long_part_angle_policy_is_rejected(tmp_path: Path):
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[printer]
width_mm = 100
depth_mm = 100
height_mm = 100

[orientation]
long_part_angle_policy = "always"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_app_settings(config_path)
