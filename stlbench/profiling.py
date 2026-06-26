from __future__ import annotations

import cProfile
import io
import json
import pstats
import resource
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from rich.console import Console

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ProfileOptions:
    enabled: bool = False
    profile_dir: Path | None = None
    sort: str = "cumulative"
    limit: int = 50


@dataclass
class StageRecord:
    name: str
    started_at: float
    ended_at: float | None = None
    duration_s: float = 0.0
    rss_start_mb: float = 0.0
    rss_end_mb: float = 0.0
    rss_peak_mb: float = 0.0
    children: list[StageRecord] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_s": self.duration_s,
            "rss_start_mb": self.rss_start_mb,
            "rss_end_mb": self.rss_end_mb,
            "rss_peak_mb": self.rss_peak_mb,
            "children": [c.to_json() for c in self.children],
        }


@dataclass
class WorkerRecord:
    name: str
    duration_s: float


def _rss_to_mb(raw: float) -> float:
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def current_max_rss_mb() -> float:
    """Return max RSS in MiB across this process and finished children."""
    self_rss = _rss_to_mb(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    child_rss = _rss_to_mb(float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    return max(self_rss, child_rss)


def memory_snapshot() -> dict[str, float]:
    self_rss = _rss_to_mb(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    child_rss = _rss_to_mb(float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    return {
        "parent_max_rss_mb": self_rss,
        "children_max_rss_mb": child_rss,
        "max_rss_mb": max(self_rss, child_rss),
    }


class ExecutionProfiler:
    def __init__(
        self,
        *,
        command: str,
        output_base: Path,
        options: ProfileOptions,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.command = command
        self.options = options
        self.metadata = metadata or {}
        self.started_iso = datetime.now(UTC).isoformat()
        self.started_perf = time.perf_counter()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = options.profile_dir or (output_base / "profiles" / stamp)
        self._roots: list[StageRecord] = []
        self._stack: list[StageRecord] = []
        self._profiles: list[cProfile.Profile] = []
        self._workers: list[WorkerRecord] = []
        self._lock = threading.Lock()
        self._profile_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.options.enabled

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        rec = StageRecord(
            name=name,
            started_at=time.perf_counter(),
            rss_start_mb=current_max_rss_mb(),
        )
        if self._stack:
            self._stack[-1].children.append(rec)
        else:
            self._roots.append(rec)
        self._stack.append(rec)
        try:
            yield
        finally:
            rec.ended_at = time.perf_counter()
            rec.duration_s = rec.ended_at - rec.started_at
            rec.rss_end_mb = current_max_rss_mb()
            rec.rss_peak_mb = max(rec.rss_start_mb, rec.rss_end_mb)
            self._stack.pop()

    def profiled_call(self, name: str, fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        if not self.enabled:
            return fn(*args, **kwargs)
        profile: cProfile.Profile | None = None
        start = time.perf_counter()
        acquired = self._profile_lock.acquire(blocking=False)
        try:
            if acquired:
                profile = cProfile.Profile()
                return profile.runcall(fn, *args, **kwargs)
            return fn(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start
            if profile is not None:
                self._add_profile(profile)
            if acquired:
                self._profile_lock.release()
            with self._lock:
                self._workers.append(WorkerRecord(name=name, duration_s=duration))

    def record_worker(self, name: str, duration_s: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._workers.append(WorkerRecord(name=name, duration_s=duration_s))

    def map(
        self,
        pool: Any,
        name: str,
        fn: Callable[[T], R],
        items: Iterable[T],
    ) -> Iterable[R]:
        if not self.enabled:
            return cast(Iterable[R], pool.map(fn, items))

        def wrapped(item: T) -> R:
            return self.profiled_call(name, fn, item)

        return cast(Iterable[R], pool.map(wrapped, items))

    def submit(self, pool: Any, name: str, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return pool.submit(fn, *args, **kwargs)

        def wrapped() -> R:
            return self.profiled_call(name, fn, *args, **kwargs)

        return pool.submit(wrapped)

    def finish(
        self, *, status: str, return_code: int | None, console: Console | None = None
    ) -> None:
        if not self.enabled:
            return
        self.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats = self._merged_stats()
        pstats_path = self.output_dir / "profile.pstats"
        stats.dump_stats(str(pstats_path))
        top_functions = self._top_functions(stats)
        finished_iso = datetime.now(UTC).isoformat()
        total_s = time.perf_counter() - self.started_perf
        payload = {
            "command": self.command,
            "metadata": self.metadata,
            "status": status,
            "return_code": return_code,
            "started_at": self.started_iso,
            "finished_at": finished_iso,
            "duration_s": total_s,
            "memory": memory_snapshot(),
            "stages": [r.to_json() for r in self._roots],
            "workers": [asdict(w) for w in self._workers],
            "sort": self.options.sort,
            "limit": self.options.limit,
            "top_functions": top_functions,
            "artifacts": {
                "json": "profile.json",
                "text": "profile.txt",
                "pstats": "profile.pstats",
            },
        }
        (self.output_dir / "profile.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        (self.output_dir / "profile.txt").write_text(
            self._text_summary(payload, stats), encoding="utf-8"
        )
        if console is not None:
            console.print(f"[dim]Profile written → {self.output_dir}[/dim]")

    def _add_profile(self, profile: cProfile.Profile) -> None:
        with self._lock:
            self._profiles.append(profile)

    def _merged_stats(self) -> pstats.Stats:
        if not self._profiles:
            profile = cProfile.Profile()
            profile.runcall(lambda: None)
            self._profiles.append(profile)
        stats = pstats.Stats(self._profiles[0], stream=io.StringIO())
        for profile in self._profiles[1:]:
            stats.add(profile)
        return stats

    def _sort_key(self) -> str:
        if self.options.sort not in {"cumulative", "tottime", "calls"}:
            return "cumulative"
        return self.options.sort

    def _top_functions(self, stats: pstats.Stats) -> list[dict[str, Any]]:
        stats.sort_stats(self._sort_key())
        rows: list[dict[str, Any]] = []
        fcn_list = cast(list[Any], getattr(stats, "fcn_list", []) or [])
        stats_any = cast(Any, stats)
        stats_data = cast(dict[Any, tuple[int, int, float, float, dict[Any, Any]]], stats_any.stats)
        for func in fcn_list[: self.options.limit]:
            cc, nc, tt, ct, _callers = stats_data[func]
            filename, line, name = func
            rows.append(
                {
                    "file": filename,
                    "line": line,
                    "function": name,
                    "primitive_calls": cc,
                    "calls": nc,
                    "tottime_s": tt,
                    "cumulative_s": ct,
                }
            )
        return rows

    def _text_summary(self, payload: dict[str, Any], stats: pstats.Stats) -> str:
        out = io.StringIO()
        out.write(f"stlbench profile: {self.command}\n")
        out.write(f"status: {payload['status']}  return_code: {payload['return_code']}\n")
        out.write(f"duration: {payload['duration_s']:.6f}s\n\n")
        memory = payload.get("memory", {})
        if memory:
            out.write(
                f"max_rss: {memory.get('max_rss_mb', 0.0):.1f} MiB"
                f"  parent={memory.get('parent_max_rss_mb', 0.0):.1f} MiB"
                f"  children={memory.get('children_max_rss_mb', 0.0):.1f} MiB\n\n"
            )
        resource_plan = payload.get("metadata", {}).get("resource_plan")
        if resource_plan:
            out.write("Resource plan:\n")
            for key in (
                "requested",
                "cpu_cap",
                "memory_budget_fraction",
                "total_ram_bytes",
                "memory_budget_bytes",
                "scale_workers",
                "orient_workers",
                "footprint_workers",
                "export_workers",
            ):
                if key in resource_plan:
                    out.write(f"  {key}: {resource_plan[key]}\n")
            out.write("\n")
        out.write("Stages:\n")
        for stage in payload["stages"]:
            self._write_stage(out, stage, 0)
        out.write("\nSlowest worker tasks:\n")
        for worker in sorted(payload["workers"], key=lambda w: w["duration_s"], reverse=True)[
            : self.options.limit
        ]:
            out.write(f"  {worker['duration_s']:.6f}s  {worker['name']}\n")
        out.write("\nTop functions:\n")
        stats_any = cast(Any, stats)
        stats_any.stream = out
        stats.sort_stats(self._sort_key()).print_stats(self.options.limit)
        return out.getvalue()

    def _write_stage(self, out: io.StringIO, stage: dict[str, Any], depth: int) -> None:
        out.write(
            f"{'  ' * depth}{stage['duration_s']:.6f}s  {stage['name']}"
            f"  rss={stage.get('rss_end_mb', 0.0):.1f} MiB\n"
        )
        for child in stage["children"]:
            self._write_stage(out, child, depth + 1)


class NullProfiler:
    enabled = False
    output_dir: Path | None = None

    def start(self) -> None:
        pass

    def finish(
        self,
        *,
        status: str,
        return_code: int | None,
        console: Console | None = None,
    ) -> None:
        pass

    def stage(self, name: str):
        return nullcontext()

    def profiled_call(self, name: str, fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        return fn(*args, **kwargs)

    def record_worker(self, name: str, duration_s: float) -> None:
        pass

    def map(
        self,
        pool: Any,
        name: str,
        fn: Callable[[T], R],
        items: Iterable[T],
    ) -> Iterable[R]:
        return cast(Iterable[R], pool.map(fn, items))

    def submit(self, pool: Any, name: str, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Any:
        return pool.submit(fn, *args, **kwargs)


Profiler = ExecutionProfiler | NullProfiler


def make_profiler(
    *,
    command: str,
    output_base: Path,
    options: ProfileOptions | None,
    metadata: dict[str, Any] | None = None,
) -> Profiler:
    if options is None or not options.enabled:
        return NullProfiler()
    return ExecutionProfiler(
        command=command,
        output_base=output_base,
        options=options,
        metadata=metadata,
    )
