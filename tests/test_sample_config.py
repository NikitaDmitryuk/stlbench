from pathlib import Path

from stlbench.config.loader import load_app_settings
from stlbench.config.sample_config import render_sample_config_toml, sample_app_settings


def test_sample_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "profile.toml"
    path.write_text(render_sample_config_toml(), encoding="utf-8")
    loaded = load_app_settings(path)
    assert loaded == sample_app_settings()


def test_repo_mars5_matches_sample() -> None:
    root = Path(__file__).resolve().parents[1]
    mars5 = load_app_settings(root / "configs" / "mars5_ultra.toml")
    assert mars5 == sample_app_settings()
