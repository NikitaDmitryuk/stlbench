from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.table import Table

    from stlbench.core.fit import PartScaleReport


def scale_table(reports: list[PartScaleReport], s_final: float) -> Table:
    """Create a Rich table showing scale results."""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("dx", justify="right")
    table.add_column("dy", justify="right")
    table.add_column("dz", justify="right")
    table.add_column("s_part", justify="right")
    table.add_column("scaled (mm)", justify="right")

    for r in reports:
        sd = (r.dx * s_final, r.dy * s_final, r.dz * s_final)
        table.add_row(
            r.name,
            f"{r.dx:.4f}",
            f"{r.dy:.4f}",
            f"{r.dz:.4f}",
            f"{r.s_limit:.6f}",
            f"{sd[0]:.2f} × {sd[1]:.2f} × {sd[2]:.2f}",
        )
    return table


def orient_table(orient_stats: list[dict]) -> Table:
    """Create a Rich table showing orientation results."""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("part", max_width=42)
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("Δ", justify="right")

    for s in orient_stats:
        table.add_row(
            s["name"], f"{s['before']:.1f}", f"{s['after']:.1f}", f"{s['delta_pct']:+.0f}%"
        )
    return table
