from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

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
        "packing": raw.get("packing") or {},
        "pipeline": raw.get("pipeline") or {},
        "parts": raw.get("parts") or [],
    }
    return cast(AppSettings, AppSettings.model_validate(data))


class AppConfig:
    """Flat read-only view over :class:`AppSettings` (legacy helper API)."""

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
    def post_fit_scale(self) -> float:
        return self._s.scaling.post_fit_scale

    @property
    def settings(self) -> AppSettings:
        return self._s


def load_config(path: Path) -> AppConfig:
    return AppConfig(load_app_settings(path))
