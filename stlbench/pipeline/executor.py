from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stlbench.steps.base import PipelineStep, StepResult

if TYPE_CHECKING:
    from stlbench.domain.part import Part


@dataclass
class Pipeline:
    """Sequential processing pipeline.

    Each step receives parts from the previous step's result.
    All step metadata is collected into a single dict.
    """

    steps: list[PipelineStep]

    def run(self, parts: list[Part]) -> StepResult:
        current_parts = parts
        combined_meta: dict = {}

        for step in self.steps:
            result = step.process(current_parts)
            current_parts = result.parts
            combined_meta.update(result.metadata)

        return StepResult(parts=current_parts, metadata=combined_meta)
