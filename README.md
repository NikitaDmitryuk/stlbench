# stlbench

Конвейер подготовки STL к фотополимерной печати: **единый масштаб**, отчёт по группам на столе, команда **`layout`** (укладка в `plate_XX.stl` + JSON), опционально **полые оболочки** (воксели + scipy). **Опоры** в пакете не строятся — их добавляют в **Lychee, Chitubox, Prusa** и т.п. после экспорта; множитель `scaling.supports_scale` в конфиге задаёт только **запас по масштабу** под слайсерные опоры и кайму.

Размеры задаются в **одних единицах** (обычно мм).

Части модели Кратоса лежат в **`models/kratos/base/`**. Результаты пайплайна — рядом: **`models/kratos/print_scaled/`** и **`models/kratos/print_plates/`** (в `.gitignore`, не коммитятся).

## Модель Кратоса — итоговые команды (скопировать целиком)

Запускайте из **корня клона** этого репозитория (рядом должны быть каталоги `stlbench/` с кодом пакета и `models/`).

**Первый раз** — установка зависимостей и полный прогон:

```bash
poetry install --with dev --extras hollow
poetry run stlbench scale -i models/kratos/base -o models/kratos/print_scaled -c configs/mars5_ultra.toml
poetry run stlbench layout -i models/kratos/print_scaled -o models/kratos/print_plates -c configs/mars5_ultra.toml
```

**Повторно** (после того как `poetry install` уже делали):

```bash
poetry run stlbench scale -i models/kratos/base -o models/kratos/print_scaled -c configs/mars5_ultra.toml
poetry run stlbench layout -i models/kratos/print_scaled -o models/kratos/print_plates -c configs/mars5_ultra.toml
```

**Тот же сценарий через переменную** (подставьте свой путь к клону вместо примера):

```bash
export STLBENCH_ROOT="$HOME/Documents/myprojects/stlbench"
cd "$STLBENCH_ROOT"
poetry install --with dev --extras hollow
poetry run stlbench scale -i "$STLBENCH_ROOT/models/kratos/base" -o "$STLBENCH_ROOT/models/kratos/print_scaled" -c configs/mars5_ultra.toml
poetry run stlbench layout -i "$STLBENCH_ROOT/models/kratos/print_scaled" -o "$STLBENCH_ROOT/models/kratos/print_plates" -c configs/mars5_ultra.toml
```

Результат: **`models/kratos/print_plates/plate_01.stl`** (при необходимости `plate_02.stl`, …) и рядом **`plate_01.json`** с позициями. Опоры — в слайсере.

## Установка (Poetry)

```bash
poetry install --with dev --extras hollow
```

- `--extras hollow` — `scipy` для полых оболочек (если включите `--hollow`). Полный цикл для Кратоса — в разделе выше.

## Установка без Poetry (pip)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[hollow]"
pip install pytest   # только для тестов
```

Сборка идёт через `poetry-core` из этого `pyproject.toml`; extras `dev` из Poetry group в `pip` не подтягиваются — тесты ставьте отдельно.

## CLI (Typer)

Совместимость: `python -m stlbench -i ... -o ...` автоматически трактуется как подкоманда **`scale`**.

```bash
python -m stlbench scale -i ./parts -o ./out -c configs/mars5_ultra.toml
# или явно:
stlbench scale --input ./parts --output ./out --config configs/mars5_ultra.toml
```

Принтер в одной строке:

```bash
stlbench scale -i ./parts -o ./out -p "153.36,77.76,165"
```

### Команда `layout`

Укладка уже подготовленных STL по прямоугольникам AABB в XY (`rectpack`), экспорт **`plate_01.stl`** и **`plate_01.json`** (позиции). Высота по Z не меняется — убедитесь, что детали ориентированы «стоя» на стол.

```bash
stlbench layout -i ./scaled -o ./plates -c configs/mars5_ultra.toml --dry-run
```

`packing.algorithm = "shelf"` в конфиге даёт только **текстовый** отчёт по группам без экспорта STL; для пластин используйте **`rectpack`** (по умолчанию в схеме).

### Полые оболочки

В [configs/mars5_ultra.toml](configs/mars5_ultra.toml) секция `[hollow]`. Включение в файле + при необходимости флаг:

```bash
stlbench scale ... -c configs/mars5_ultra.toml --hollow
```

- **Hollow**: backend `open3d_voxel` в конфиге — фактически **scipy + trimesh voxel** (имя историческое). Нужен extra `[hollow]`.

### Опоры

Команда `stlbench supports` выводит напоминание: опоры ставятся в слайсере. Секция `[supports]` в TOML оставлена для совместимости (`backend = "none"`); шаблон `external_command_template` зарезервирован под свой вызов в `supports/external.py` (пока не реализован).

Про промышленный offset см. [stlbench/hollow/meshlib_note.md](stlbench/hollow/meshlib_note.md).

## Структура пакета

Исходники Python — пакет **`stlbench`** (каталог `stlbench/` в корне репозитория).

| Модуль | Назначение |
|--------|------------|
| `stlbench.core` | Расчёт масштаба и ориентации bbox |
| `stlbench.config` | Pydantic-схема + загрузка TOML |
| `stlbench.packing` | Полка (`shelf`) и `rectpack` |
| `stlbench.export` | Склейка пластины и manifest |
| `stlbench.hollow` | Воксельная оболочка |
| `stlbench.supports` | Заглушка под внешние опоры (без геометрии в CLI) |
| `stlbench.pipeline` | Сценарии `run_scale`, `run_layout` |
| `stlbench.cli` (модуль `cli.py`) | Typer-приложение |

Вспомогательные модули `stlbench.bbox_fit`, `stlbench.orientation`, `stlbench.plate_groups`, `stlbench.config_load` — прокси для обратной совместимости.

## Ограничения

Boolean и воксели чувствительны к **дырявым** STL. Полости в пакете — упрощение; для сложных моделей и опор используйте слайсер.
