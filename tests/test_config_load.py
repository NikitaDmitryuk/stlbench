from pathlib import Path

from stlbench.config.loader import load_config


def test_load_mars5_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "mars5_ultra.toml")
    assert "Mars" in cfg.printer_name
    assert cfg.width_mm > 0
    assert cfg.post_fit_scale == 0.95
    assert cfg.settings.packing.gap_mm == 0.5
