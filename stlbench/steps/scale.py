from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from stlbench.core.fit import Method, compute_global_scale
from stlbench.domain.part import Part
from stlbench.domain.printer import Printer
from stlbench.packing.layout_orientation import select_orientation_for_scale
from stlbench.steps.base import PipelineStep, StepResult

if TYPE_CHECKING:
    pass


@dataclass
class ScaleStep(PipelineStep):
    """Scale parts to maximum size for the printer.

    By default, orientation is not changed (only scaling).
    allow_rotation=True enables orientation search to maximize scale.
    """

    printer: Printer
    method: Method = "sorted"
    post_fit_scale: float = 1.0
    allow_rotation: bool = False
    maximize: bool = False
    scale_factor: float | None = None  # explicit factor, bypasses computation
    no_upscale: bool = False
    rotation_samples: int = 4096
    seed: int = 0

    def process(self, parts: list[Part]) -> StepResult:
        if self.scale_factor is not None:
            return self._apply_explicit_scale(parts)
        return self._compute_and_apply(parts)

    def _compute_and_apply(self, parts: list[Part]) -> StepResult:
        if self.allow_rotation:
            transforms, dims = self._find_orientations(parts)
        else:
            transforms = [None] * len(parts)
            dims = [p.extents for p in parts]

        s_max, reports = compute_global_scale(
            self.printer.xyz,
            dims,
            [p.name for p in parts],
            self.method,
        )

        s = min(1.0, s_max) if self.no_upscale else s_max
        s_final = s * self.post_fit_scale

        result_parts = []
        for part, t4 in zip(parts, transforms, strict=True):
            p = part.clone()
            if t4 is not None:
                p.apply_transform(t4)
            p.apply_scale(s_final).floor_z()
            result_parts.append(p)

        return StepResult(
            parts=result_parts,
            metadata={"s_max": s_max, "s_final": s_final, "reports": reports},
        )

    def _find_orientations(
        self, parts: list[Part]
    ) -> tuple[list[np.ndarray | None], list[tuple[float, float, float]]]:
        """Parallel orientation search for scaling."""
        from stlbench.pipeline.common import n_workers

        n = n_workers(len(parts))
        results: list[tuple[np.ndarray, tuple[float, float, float]] | None] = [None] * len(parts)

        def _work(idx: int):
            return select_orientation_for_scale(
                parts[idx].mesh,
                *self.printer.xyz,
                self.method,
                maximize=self.maximize,
                random_samples=self.rotation_samples,
                seed=self.seed,
            )

        # Sequential fallback for small numbers of parts
        if n <= 1:
            for i in range(len(parts)):
                results[i] = _work(i)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n) as pool:
                futs = {pool.submit(_work, i): i for i in range(len(parts))}
                for fut in as_completed(futs):
                    i = futs[fut]
                    results[i] = fut.result()

        transforms: list[np.ndarray | None] = [r[0] if r is not None else None for r in results]
        dims: list[tuple[float, float, float]] = [
            r[1] if r is not None else (0.0, 0.0, 0.0) for r in results
        ]
        return transforms, dims

    def _apply_explicit_scale(self, parts: list[Part]) -> StepResult:
        assert self.scale_factor is not None
        s_final = self.scale_factor * self.post_fit_scale
        result_parts = [p.clone().apply_scale(s_final).floor_z() for p in parts]
        return StepResult(
            parts=result_parts,
            metadata={"s_final": s_final, "s_max": self.scale_factor},
        )
