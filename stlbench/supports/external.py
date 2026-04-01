from __future__ import annotations

from pathlib import Path


def run_external_support_stub(command_template: str, work_dir: Path) -> None:
    """
    Зарезервировано под свой subprocess. По умолчанию опоры ставят в Lychee / Chitubox / Prusa.
    """
    if not command_template.strip():
        raise ValueError("external_command_template пуст.")
    raise NotImplementedError(
        f"Внешние опоры не подключены: {command_template!r}, {work_dir}. "
        "Добавьте свой вызов или используйте слайсер."
    )
