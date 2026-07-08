from __future__ import annotations

from enum import StrEnum
from typing import TypeVar


class ResinBalance(StrEnum):
    BALANCED = "balanced"
    STABILITY = "stability"
    COMPACT = "compact"


class LongPartAnglePolicy(StrEnum):
    THIN_LINEAR = "thin_linear"
    LINEAR = "linear"
    DISABLED = "disabled"


class AssemblySidePolicy(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"


class ScaleFitMethod(StrEnum):
    SORTED = "sorted"
    CONSERVATIVE = "conservative"


class OrientationPolicy(StrEnum):
    MAX_SCALE = "max-scale"
    PRINTABLE = "printable"


class PackerBackend(StrEnum):
    AUTO = "auto"
    BITMAP = "bitmap"
    EXACT = "exact"


class ExactPackQuality(StrEnum):
    FINAL = "final"
    FEASIBILITY = "feasibility"


class ExportCompressionMode(StrEnum):
    DEFAULT = "default"
    FAST = "fast"
    STORE = "store"


class PrepareOrientationStrategy(StrEnum):
    AUTO = "auto"
    LEGACY = "legacy"


class PrepareOrientationQuality(StrEnum):
    DEFAULT = "default"
    ADAPTIVE = "adaptive"


class CandidateProfile(StrEnum):
    DEFAULT = "default"
    ADAPTIVE = "adaptive"


class ProfileSortKey(StrEnum):
    CUMULATIVE = "cumulative"
    TOTAL_TIME = "tottime"
    CALLS = "calls"


EnumT = TypeVar("EnumT", bound=StrEnum)


def enum_values(enum_cls: type[EnumT]) -> str:
    return ", ".join(item.value for item in enum_cls)


def coerce_enum(enum_cls: type[EnumT], value: EnumT | str, label: str) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be {enum_values(enum_cls)}; got {value!r}.") from exc
