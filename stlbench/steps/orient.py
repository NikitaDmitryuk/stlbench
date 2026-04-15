from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from stlbench.core.overhang import (
    apply_min_overhang_orientation,
    find_min_overhang_rotation,
    overhang_score,
)
from stlbench.domain.part import Part
from stlbench.domain.printer import Printer
from stlbench.steps.base import PipelineStep, StepResult

if TYPE_CHECKING:
    pass


@dataclass
class OrientStep(PipelineStep):
    """Rotate parts to minimize overhang area (Tweaker-3)."""

    printer: Printer
    overhang_threshold_deg: float = 45.0
    n_candidates: int = 200

    def process(self, parts: list[Part]) -> StepResult:
        # Warm trimesh caches before multithreaded work (prevents race conditions)
        for p in parts:
            p.warm_caches()

        from stlbench.pipeline.common import n_workers

        n = n_workers(len(parts))
        results = [None] * len(parts)

        def _work(idx: int):
            mesh = parts[idx].mesh
            sb = overhang_score(mesh, np.eye(3), self.overhang_threshold_deg)
            rot, sa = find_min_overhang_rotation(
                mesh,
                overhang_threshold_deg=self.overhang_threshold_deg,
                n_candidates=self.n_candidates,
                printer_dims=self.printer.xyz,
            )
            oriented_mesh = apply_min_overhang_orientation(mesh, rot)
            pct = (sb - sa) / max(abs(sb), 1.0) * 100.0
            return oriented_mesh, sb, sa, pct

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

        result_parts = []
        orient_stats = []
        for part, result in zip(parts, results, strict=True):
            if result is None:
                # Fallback - keep original part if orientation failed
                result_parts.append(part.clone())
                orient_stats.append(
                    {"name": part.name, "before": 0.0, "after": 0.0, "delta_pct": 0.0}
                )
            else:
                oriented_mesh, sb, sa, pct = result
                p = Part(name=part.name, mesh=oriented_mesh, source_path=part.source_path)
                result_parts.append(p)
                orient_stats.append(
                    {"name": part.name, "before": sb, "after": sa, "delta_pct": pct}
                )

        return StepResult(
            parts=result_parts,
            metadata={"orient_stats": orient_stats},
        )
