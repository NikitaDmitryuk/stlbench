from __future__ import annotations

from typing import TYPE_CHECKING

from stlbench.packing.base import PackingStrategy
from stlbench.packing.polygon_pack import PolygonPacker
from stlbench.packing.rectpack_plate import RectPacker
from stlbench.packing.shelf import ShelfPacker

if TYPE_CHECKING:
    pass

_REGISTRY: dict[str, type[PackingStrategy]] = {
    "polygon": PolygonPacker,
    "rectpack": RectPacker,
    "shelf": ShelfPacker,
}


def make_packer(algorithm: str, **kwargs) -> PackingStrategy:
    """Create a packer by algorithm name.

    Args:
        algorithm: "polygon" | "rectpack" | "shelf"
        **kwargs:  Constructor parameters (grid_step_mm, max_plates, etc.)
    """
    cls = _REGISTRY.get(algorithm)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown packing algorithm {algorithm!r}. Available: {available}")
    return cls(**kwargs)
