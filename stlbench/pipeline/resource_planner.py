from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

_MIB = 1024 * 1024
DEFAULT_MEMORY_BUDGET_FRACTION = 0.70


@dataclass(frozen=True)
class InputSizeStats:
    count: int
    total_bytes: int
    largest_bytes: int

    def to_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerPlan:
    requested: str
    cpu_cap: int
    memory_budget_fraction: float
    total_ram_bytes: int | None
    memory_budget_bytes: int | None
    input: InputSizeStats
    scale_workers: int
    orient_workers: int
    footprint_workers: int
    export_workers: int

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input"] = self.input.to_json()
        return payload


def total_system_memory_bytes() -> int | None:
    """Return physical RAM in bytes using only stdlib/platform tools."""
    meminfo = Path("/proc/meminfo")
    try:
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (OSError, ValueError, AttributeError):
        pass

    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        value = int(raw)
        return value if value > 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def input_size_stats(paths: list[Path]) -> InputSizeStats:
    sizes: list[int] = []
    for path in paths:
        try:
            sizes.append(path.stat().st_size)
        except OSError:
            sizes.append(0)
    return InputSizeStats(
        count=len(paths),
        total_bytes=sum(sizes),
        largest_bytes=max(sizes, default=0),
    )


def _cpu_cap() -> int:
    cpu = os.cpu_count() or 2
    return max(1, int(cpu * 2 / 3))


def parse_worker_override(value: str | None) -> int | None:
    if value is None or value == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("--workers must be 'auto' or a positive integer.") from exc
    if parsed <= 0:
        raise ValueError("--workers must be 'auto' or a positive integer.")
    return parsed


def _estimate_mesh_worker_rss_bytes(largest_input_bytes: int, *, multiplier: float) -> int:
    # STL text/binary loading, numpy arrays, trimesh caches and Shapely buffers
    # can expand a mesh many times over its file size. Keep this conservative.
    return max(int(largest_input_bytes * multiplier), 512 * _MIB)


def choose_workers(
    *,
    n_items: int,
    requested_workers: str | None,
    memory_budget_bytes: int | None,
    cpu_cap: int,
    estimated_worker_rss_bytes: int,
) -> int:
    if n_items <= 0:
        return 1
    override = parse_worker_override(requested_workers)
    base_cap = min(n_items, cpu_cap)
    if override is not None:
        return max(1, min(base_cap, override))
    if memory_budget_bytes is None or estimated_worker_rss_bytes <= 0:
        return max(1, base_cap)
    memory_cap = max(1, memory_budget_bytes // estimated_worker_rss_bytes)
    return max(1, min(base_cap, int(memory_cap)))


def make_prepare_worker_plan(
    paths: list[Path],
    *,
    requested_workers: str | None = "auto",
    memory_budget_fraction: float = DEFAULT_MEMORY_BUDGET_FRACTION,
) -> WorkerPlan:
    stats = input_size_stats(paths)
    total_ram = total_system_memory_bytes()
    budget = int(total_ram * memory_budget_fraction) if total_ram is not None else None
    cpu_cap = _cpu_cap()
    requested = requested_workers or "auto"

    scale_estimate = _estimate_mesh_worker_rss_bytes(stats.largest_bytes, multiplier=14.0)
    orient_estimate = _estimate_mesh_worker_rss_bytes(stats.largest_bytes, multiplier=18.0)
    footprint_estimate = _estimate_mesh_worker_rss_bytes(stats.largest_bytes, multiplier=12.0)

    return WorkerPlan(
        requested=requested,
        cpu_cap=cpu_cap,
        memory_budget_fraction=memory_budget_fraction,
        total_ram_bytes=total_ram,
        memory_budget_bytes=budget,
        input=stats,
        scale_workers=choose_workers(
            n_items=stats.count,
            requested_workers=requested,
            memory_budget_bytes=budget,
            cpu_cap=cpu_cap,
            estimated_worker_rss_bytes=scale_estimate,
        ),
        orient_workers=choose_workers(
            n_items=stats.count,
            requested_workers=requested,
            memory_budget_bytes=budget,
            cpu_cap=cpu_cap,
            estimated_worker_rss_bytes=orient_estimate,
        ),
        footprint_workers=choose_workers(
            n_items=stats.count,
            requested_workers=requested,
            memory_budget_bytes=budget,
            cpu_cap=cpu_cap,
            estimated_worker_rss_bytes=footprint_estimate,
        ),
        export_workers=1,
    )


def choose_export_workers(
    *,
    plate_part_bytes: list[int],
    requested_workers: str | None,
    memory_budget_bytes: int | None,
    cpu_cap: int,
) -> int:
    if not plate_part_bytes:
        return 1
    largest_plate = max(plate_part_bytes)
    export_estimate = max(int(largest_plate * 8.0), 768 * _MIB)
    return choose_workers(
        n_items=len(plate_part_bytes),
        requested_workers=requested_workers,
        memory_budget_bytes=memory_budget_bytes,
        cpu_cap=cpu_cap,
        estimated_worker_rss_bytes=export_estimate,
    )
