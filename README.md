# stlbench

[![PyPI version](https://img.shields.io/pypi/v/stlbench.svg)](https://pypi.org/project/stlbench/)
[![Python versions](https://img.shields.io/pypi/pyversions/stlbench.svg)](https://pypi.org/project/stlbench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**STL preparation toolkit for resin 3D printing.**

stlbench takes STL files and prepares them for SLA/DLP printers: uniform scaling to
fit the build volume, packing parts onto rectangular print plates, filling the bed with
copies, and combined scale-and-pack in one step. Supports are **not** generated --
use your slicer (Lychee, Chitubox, PrusaSlicer, etc.) after export.

## Installation

```bash
pip install stlbench
```

(`scipy` is a required dependency for voxel hollowing with `scale --hollow`; install uses
binary wheels on supported Python versions — see [PyPI](https://pypi.org/project/scipy/).)

### Development install

```bash
git clone https://github.com/NikitaDmitryuk/stlbench.git
cd stlbench
poetry install --with dev
```

## Quick Start

Run **`stlbench --help`** for the same command cheatsheet (copy-paste friendly).

```bash
# Inspect model parts
stlbench info -i ./parts -c configs/mars5_ultra.toml

# Scale all parts to fit the printer
stlbench scale -i ./parts -o ./scaled -c configs/mars5_ultra.toml

# Pack scaled parts onto plates
stlbench layout -i ./scaled -o ./plates -c configs/mars5_ultra.toml

# Scale + pack all on one plate in one step
stlbench autopack -i ./parts -o ./packed -c configs/mars5_ultra.toml

# Fill the bed with copies of a single part
stlbench fill -i ./part.stl -o ./filled -c configs/mars5_ultra.toml
```

Or specify the printer inline without a config file:

```bash
stlbench scale -i ./parts -o ./scaled -p "153.36,77.76,165"
```

## Commands

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
`--hollow`, `--post-fit-scale`.

### `layout` -- Pack parts onto plates

```bash
stlbench layout -i ./scaled -o ./plates -c configs/mars5_ultra.toml
```

Arranges already-scaled STL files onto rectangular print plates using `rectpack`.
Exports `plate_01.stl` + `plate_01.json` with positions. Multiple plates are
created if parts do not fit on one.

Key options: `--dry-run`, `--gap-mm`, `--algorithm shelf|rectpack`.

### `autopack` -- Scale + layout on one plate

```bash
stlbench autopack -i ./parts -o ./packed -c configs/mars5_ultra.toml
```

Binary-searches for the maximum scale factor at which **all** parts fit onto a
single plate simultaneously. Combines `scale` and `layout` into one step with a
different goal: all parts on one plate, not each part fitting individually.

Key options: `--dry-run`, `--gap-mm`, `--margin`, `--post-fit-scale`.

### `fill` -- Maximum copies of one part

```bash
stlbench fill -i ./part.stl -o ./filled -c configs/mars5_ultra.toml
```

Packs as many copies of a single STL file as possible onto one plate. Useful for
batch printing identical parts.

Key options: `--scale` (scale the part to fit before filling), `--dry-run`, `--gap-mm`.

### `hollow`

`stlbench hollow` prints a short reminder: hollowing is optional and runs only with
`scale ... --hollow` plus `[hollow]` in the TOML.

### `config init` -- Create a starter TOML

```bash
stlbench config init -o my_printer.toml
```

Writes a commented profile with the same defaults as [`configs/mars5_ultra.toml`](configs/mars5_ultra.toml)
(example resin printer sizes, scaling, gap, hollow params). Use `--stdout` to print
without saving, or `--force` to overwrite an existing file.

## Configuration

Printer profiles are TOML files. See [`configs/mars5_ultra.toml`](configs/mars5_ultra.toml)
for a complete example (ELEGOO Mars 5 Ultra), or generate one with `stlbench config init`.

Key sections:

| Section         | Purpose                                          |
|-----------------|--------------------------------------------------|
| `[printer]`     | Build volume: `width_mm`, `depth_mm`, `height_mm`|
| `[scaling]`     | `bed_margin`, `post_fit_scale`                   |
| `[packing]`     | `gap_mm` between parts on the bed                |
| `[hollow]`      | `wall_thickness_mm`, `voxel_mm` for `scale --hollow` |

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

| Module             | Purpose                                    |
|--------------------|--------------------------------------------|
| `stlbench.cli`     | Typer CLI application                      |
| `stlbench.core`    | Scale factor computation and orientation   |
| `stlbench.config`  | Pydantic schema + TOML loader              |
| `stlbench.packing` | Shelf and rectpack algorithms              |
| `stlbench.export`  | Plate STL assembly and JSON manifest       |
| `stlbench.hollow`  | Voxel shell hollowing (`scale --hollow`)   |
| `stlbench.pipeline`| Command runners (scale, layout, fill, etc.)|

## Limitations

- Boolean and voxel operations are sensitive to non-manifold STL. For complex
  models use a mesh repair tool first.
- Hollow shells in this package are a simplified voxel approach; for production
  use your slicer's built-in hollowing.
- Very new Python releases may lack a `scipy` wheel yet; use a version listed for your
  platform on PyPI or wait for upstream wheels.

## License

[MIT](LICENSE)
