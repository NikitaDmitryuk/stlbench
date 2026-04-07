# stlbench

[![PyPI version](https://img.shields.io/pypi/v/stlbench.svg)](https://pypi.org/project/stlbench/)
[![Python versions](https://img.shields.io/pypi/pyversions/stlbench.svg)](https://pypi.org/project/stlbench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**STL preparation toolkit for resin 3D printing.**

stlbench takes STL files and prepares them for SLA/DLP printers: automatic
support-minimising orientation, uniform scaling to fit the build volume, packing
parts onto rectangular print plates, filling the bed with copies, and combined
scale-and-pack in one step. Support generation and hollowing are **not** performed --
use your slicer (Lychee, Chitubox, PrusaSlicer, etc.) after export.

## Installation

```bash
pip install stlbench
```

### Development install

```bash
git clone https://github.com/NikitaDmitryuk/stlbench.git
cd stlbench
poetry install --with dev
```

## Quick Start

Run **`stlbench --help`** for the same command cheatsheet (copy-paste friendly).

```bash
# Full pipeline in one command: scale → orient → layout (recommended)
stlbench prepare -i ./parts -o ./plates -c configs/mars5_ultra.toml

# Or step by step:
stlbench info   -i ./parts                 -c configs/mars5_ultra.toml  # inspect
stlbench orient -i ./parts -o ./oriented   -c configs/mars5_ultra.toml  # minimise supports
stlbench scale  -i ./oriented -o ./scaled  -c configs/mars5_ultra.toml  # fit to build volume
stlbench layout -i ./scaled -o ./plates    -c configs/mars5_ultra.toml  # pack on plates

# Scale + pack all on one plate in one step
stlbench autopack -i ./parts -o ./packed -c configs/mars5_ultra.toml

# Fill the bed with copies of a single part
stlbench fill -i ./part.stl -o ./filled -c configs/mars5_ultra.toml
```

Or specify the printer inline without a config file:

```bash
stlbench prepare -i ./parts -o ./plates -p "153.36,77.76,165"
```

## Commands

### `prepare` -- Full pipeline: scale → orient → layout

```bash
stlbench prepare -i ./parts -o ./plates -c configs/mars5_ultra.toml
```

Runs the three preparation steps in optimal order:

1. **Scale** — finds the largest scale factor that fits every part inside the build volume
   (using an orientation-free search), then applies a uniform scale to all parts.
2. **Orient** — rotates each *already-scaled* part to minimise overhanging surface area,
   subject to the constraint that the part still fits the build volume in the new orientation.
3. **Layout** — packs the oriented parts onto the minimum number of plates, distributed
   as evenly as possible.

Exports one `plate_NN.3mf` + `plate_NN.json` per plate.

Key options: `--overhang-angle` (default 45°), `--orient-candidates`, `--gap-mm`,
`--post-fit-scale`, `--dry-run`, `--recursive`.

### `info` -- Analyze models (read-only)

```bash
stlbench info -i ./parts -c configs/mars5_ultra.toml
```

Displays a table with AABB dimensions, volume, vertex/face counts, whether each
part fits the bed, maximum scale factor, and how many copies would fit (`fill`).
No files are written.

### `scale` -- Uniform scaling

```bash
stlbench scale -i ./parts -o ./out -c configs/mars5_ultra.toml
```

Computes a single scale factor so that **every** part fits inside the printer
build volume. The largest part determines the factor; all parts share the same
scale. Supports two methods: `sorted` (default) and `conservative`.

Key options: `--dry-run`, `--no-upscale`, `--method`, `--orientation free`,
`--post-fit-scale`, `--suffix`, `--recursive`.

### `layout` -- Pack parts onto plates

```bash
stlbench layout -i ./scaled -o ./plates -c configs/mars5_ultra.toml
```

Arranges already-scaled STL files onto rectangular print plates using `rectpack`.
Exports `plate_01.stl` + `plate_01.json` with positions. Multiple plates are
created if parts do not fit on one.

Key options: `--dry-run`, `--gap-mm`, `--algorithm shelf|rectpack`.

### `fill` -- Maximum copies of one part

```bash
stlbench fill -i ./part.stl -o ./filled -c configs/mars5_ultra.toml
```

Packs as many copies of a single STL file as possible onto one plate. Useful for
batch printing identical parts.

Key options: `--scale` (scale the part to fit before filling),
`--orient/--no-orient` (minimise supports before filling),
`--overhang-angle`, `--dry-run`, `--gap-mm`.

### `orient` -- Minimise support structures

```bash
stlbench orient -i ./parts -o ./oriented -c configs/mars5_ultra.toml
```

For each STL file, searches for the rotation that minimises the total area of
overhanging surfaces (faces whose downward angle exceeds the threshold). Uses a
two-phase search: discrete evaluation of ~600+ candidate orientations derived from
the model's own face normals and a uniform icosphere, followed by Nelder-Mead local
refinement via `scipy.optimize`. The result is written as a new STL with the
bottom at z = 0, ready for `scale` / `layout`.

When `--config` or `--printer` is supplied, the search is constrained to orientations
that fit inside the build volume (non-fitting orientations receive a heavy penalty).

Key options: `--config`/`-c`, `--printer`/`-p`, `--overhang-angle` (default 45°),
`--candidates` (default 200), `--dry-run`, `--suffix`, `--recursive`.

### `autopack` -- Scale + layout on one plate

```bash
stlbench autopack -i ./parts -o ./packed -c configs/mars5_ultra.toml
```

Binary-searches for the maximum scale factor at which **all** parts fit onto a
single plate simultaneously. Combines `scale` and `layout` into one step with a
different goal: all parts on one plate, not each part fitting individually.

Key options: `--orient/--no-orient` (minimise supports before packing),
`--overhang-angle`, `--dry-run`, `--gap-mm`, `--margin`, `--post-fit-scale`.

### `config init` -- Create a starter TOML

```bash
stlbench config init -o my_printer.toml
```

Writes a commented profile with the same defaults as [`configs/mars5_ultra.toml`](configs/mars5_ultra.toml).
Use `--stdout` to print without saving, or `--force` to overwrite an existing file.

## Configuration

Printer profiles are TOML files. See [`configs/mars5_ultra.toml`](configs/mars5_ultra.toml)
for a complete example (ELEGOO Mars 5 Ultra), or generate one with `stlbench config init`.

Key sections:

| Section         | Purpose                                          |
|-----------------|--------------------------------------------------|
| `[printer]`     | Build volume: `width_mm`, `depth_mm`, `height_mm`|
| `[scaling]`     | `bed_margin`, `post_fit_scale`                   |
| `[packing]`     | `gap_mm` between parts on the bed                |

Orientation (`axis` / `free`) and rotation sample count are **not** in TOML: use
`scale --orientation` and `scale --rotation-samples`. With `free`, orientation matches the
same printer-axis search as `layout` (permutation × random rotations), but scale picks the
candidate that **maximizes** the group scale factor (layout still minimizes XY footprint for
packing). Plate placement is only in `layout`. Default `layout` algorithm (`rectpack` vs
`shelf`) is set via `layout --algorithm`.

## Examples

See [`examples/README.md`](examples/README.md) for a full walkthrough using the
included Gendalf model (3 parts tracked via Git LFS).

## Package Structure

| Module             | Purpose                                               |
|--------------------|-------------------------------------------------------|
| `stlbench.cli`     | Typer CLI application                                 |
| `stlbench.core`    | Scale factor, orientation, overhang analysis          |
| `stlbench.config`  | Pydantic schema + TOML loader                         |
| `stlbench.packing` | Shelf and rectpack algorithms                         |
| `stlbench.export`  | Plate STL assembly and JSON manifest                  |
| `stlbench.pipeline`| Command runners (orient, scale, layout, fill, etc.)   |

## Limitations

- Boolean operations are sensitive to non-manifold STL. For complex models use a
  mesh repair tool first.
- Supports and hollowing are not generated — use your slicer after export.

## License

[MIT](LICENSE)
