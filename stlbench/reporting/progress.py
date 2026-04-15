from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


@contextmanager
def make_progress(console: Console):
    """Context manager for Rich progress bar. Single place where it's created."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        yield progress
