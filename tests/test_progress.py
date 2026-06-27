from __future__ import annotations

import io

import pytest
from rich.console import Console

from stlbench.pipeline.progress import PipelineProgress, make_progress, progress_enabled


def test_noop_progress_accepts_task_updates() -> None:
    console = Console(file=io.StringIO(), force_terminal=False)

    assert progress_enabled(console, requested=True) is False
    with make_progress(console, enabled=True) as progress:
        task = progress.add_task("Working…", total=2)
        progress.advance(task)
        progress.update(task, completed=2, description="Done")


def test_pipeline_progress_closes_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("STLBENCH_NO_PROGRESS", raising=False)
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    ui = PipelineProgress(console, "unit", ("One",), requested=True)

    with pytest.raises(RuntimeError), ui, ui.stage("One") as progress:
        task = progress.add_task("Doing work…", total=1)
        progress.advance(task)
        raise RuntimeError("boom")

    assert ui._live is not None
    assert not ui._live.is_started
