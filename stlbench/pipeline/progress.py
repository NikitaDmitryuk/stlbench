from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


def progress_enabled(console: Console, requested: bool) -> bool:
    if not requested:
        return False
    if os.environ.get("CI") or os.environ.get("NO_COLOR") or os.environ.get("STLBENCH_NO_PROGRESS"):
        return False
    return bool(console.is_terminal)


def progress_columns() -> tuple[Any, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


class NullProgress(AbstractContextManager["NullProgress"]):
    def __enter__(self) -> NullProgress:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def add_task(self, _description: str, total: float | None = None, **_kwargs: Any) -> TaskID:
        return TaskID(0)

    def update(self, _task_id: TaskID, **_kwargs: Any) -> None:
        return None

    def advance(self, _task_id: TaskID, advance: float = 1) -> None:
        return None


@dataclass
class PipelineProgress(AbstractContextManager["PipelineProgress"]):
    console: Console
    command: str
    stages: tuple[str, ...]
    requested: bool = True

    def __post_init__(self) -> None:
        self.enabled = progress_enabled(self.console, self.requested)
        self._stage_index = 0
        self._live: Live | None = None
        self._overall: Progress | None = None
        self._detail: Progress | None = None
        self._overall_task: TaskID | None = None

    def __enter__(self) -> PipelineProgress:
        if not self.enabled:
            return self
        self._overall = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._detail = Progress(*progress_columns(), console=self.console, transient=False)
        self._overall_task = self._overall.add_task(
            f"{self.command}: starting", total=max(1, len(self.stages))
        )
        self._live = Live(
            Group(self._overall, Text(""), self._detail),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.stop()
        return None

    def stage(
        self, name: str, *, total: float | None = None, description: str | None = None
    ) -> StageProgress:
        return StageProgress(self, name=name, total=total, description=description or name)

    def _start_stage(
        self, name: str, total: float | None, description: str
    ) -> Progress | NullProgress:
        if not self.enabled or self._overall is None or self._detail is None:
            return NullProgress()
        self._stage_index += 1
        label = f"{self.command}: {self._stage_index}/{max(1, len(self.stages))} {name}"
        self._overall.update(self._overall_task, description=label)  # type: ignore[arg-type]
        return self._detail

    def _finish_stage(self) -> None:
        if not self.enabled or self._overall is None:
            return
        self._overall.advance(self._overall_task)  # type: ignore[arg-type]


class StageProgress(AbstractContextManager[Progress | NullProgress]):
    def __init__(
        self,
        owner: PipelineProgress,
        *,
        name: str,
        total: float | None,
        description: str,
    ) -> None:
        self.owner = owner
        self.name = name
        self.total = total
        self.description = description
        self.progress: Progress | NullProgress | None = None

    def __enter__(self) -> Progress | NullProgress:
        self.progress = self.owner._start_stage(self.name, self.total, self.description)
        return self.progress

    def __exit__(self, *exc: object) -> None:
        self.owner._finish_stage()
        return None


def make_progress(console: Console, *, enabled: bool = True) -> Progress | NullProgress:
    if not progress_enabled(console, enabled):
        return NullProgress()
    return Progress(*progress_columns(), console=console)
