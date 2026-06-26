from __future__ import annotations

from pathlib import Path

import pytest

from stlbench.pipeline import resource_planner as rp


def test_total_system_memory_detection_does_not_raise():
    value = rp.total_system_memory_bytes()
    assert value is None or value > 0


def test_choose_workers_uses_one_worker_when_memory_is_tight():
    workers = rp.choose_workers(
        n_items=8,
        requested_workers="auto",
        memory_budget_bytes=1_000,
        cpu_cap=8,
        estimated_worker_rss_bytes=10_000,
    )
    assert workers == 1


def test_choose_workers_allows_parallelism_when_memory_allows():
    workers = rp.choose_workers(
        n_items=8,
        requested_workers="auto",
        memory_budget_bytes=40_000,
        cpu_cap=8,
        estimated_worker_rss_bytes=10_000,
    )
    assert workers == 4


def test_choose_workers_honors_manual_override():
    workers = rp.choose_workers(
        n_items=8,
        requested_workers="3",
        memory_budget_bytes=1_000,
        cpu_cap=8,
        estimated_worker_rss_bytes=10_000,
    )
    assert workers == 3


def test_invalid_worker_override_raises():
    with pytest.raises(ValueError, match="--workers"):
        rp.parse_worker_override("nope")


def test_prepare_worker_plan_uses_input_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = []
    for idx, size in enumerate((100, 200, 300)):
        path = tmp_path / f"part_{idx}.stl"
        path.write_bytes(b"x" * size)
        paths.append(path)

    monkeypatch.setattr(rp, "total_system_memory_bytes", lambda: 8 * 1024**3)
    plan = rp.make_prepare_worker_plan(paths, requested_workers="auto")

    assert plan.input.count == 3
    assert plan.input.total_bytes == 600
    assert plan.input.largest_bytes == 300
    assert plan.memory_budget_bytes == int(8 * 1024**3 * 0.70)
    assert plan.scale_workers >= 1
