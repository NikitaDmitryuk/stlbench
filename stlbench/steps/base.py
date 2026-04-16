from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stlbench.domain.part import Part


@dataclass
class StepResult:
    """Result of one pipeline step."""

    parts: list[Part]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PipelineStep(ABC):
    """One step in the processing pipeline."""

    @abstractmethod
    def process(self, parts: list[Part]) -> StepResult:
        """Process a list of parts and return the result.

        Implementations may mutate the provided Part objects or create new ones.
        The order of parts in the result should match the input order.
        """
        ...

    @property
    def name(self) -> str:
        return type(self).__name__
