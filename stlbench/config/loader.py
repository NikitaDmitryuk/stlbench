from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from stlbench.config.schema import AppSettings


def load_app_settings(path: Path) -> AppSettings:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    printer = raw.get("printer") or {}
    if "width_mm" not in printer:
        raise ValueError("config [printer] must define width_mm, depth_mm, height_mm.")
    data = {
        "printer": printer,
        "scaling": raw.get("scaling") or {},
        "orientation": raw.get("orientation") or {},
        "packing": raw.get("packing") or {},
        "hollow": raw.get("hollow") or {},
        "supports": raw.get("supports") or {},
    }
    return cast(AppSettings, AppSettings.model_validate(data))


# Backward-compatible flat view for legacy code paths
class AppConfig:
    """Плоский доступ к полям (совместимость со старым API)."""

    def __init__(self, s: AppSettings) -> None:
        self._s = s

    @property
    def printer_name(self) -> str:
        return self._s.printer.name

    @property
    def width_mm(self) -> float:
        return self._s.printer.width_mm

    @property
    def depth_mm(self) -> float:
        return self._s.printer.depth_mm

    @property
    def height_mm(self) -> float:
        return self._s.printer.height_mm

    @property
    def bed_margin(self) -> float:
        return self._s.scaling.bed_margin

    @property
    def supports_scale(self) -> float:
        return self._s.scaling.supports_scale

    @property
    def orientation_mode(self) -> str:
        return self._s.orientation.mode

    @property
    def orientation_samples(self) -> int:
        return self._s.orientation.samples

    @property
    def orientation_seed(self) -> int:
        return self._s.orientation.seed

    @property
    def packing_report(self) -> bool:
        return self._s.packing.report

    @property
    def settings(self) -> AppSettings:
        return self._s


def load_config(path: Path) -> AppConfig:
    return AppConfig(load_app_settings(path))
