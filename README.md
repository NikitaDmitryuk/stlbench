# stlbench

[![PyPI version](https://img.shields.io/pypi/v/stlbench.svg)](https://pypi.org/project/stlbench/)
[![Python versions](https://img.shields.io/pypi/pyversions/stlbench.svg)](https://pypi.org/project/stlbench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**STL preparation toolkit for resin 3D printing.**

stlbench takes STL files and prepares them for SLA/DLP printers: automatic
support-minimising orientation, uniform scaling to fit the build volume, packing
parts onto rectangular print plates, filling the bed with copies, and a flexible
job-file pipeline for mixing raw and pre-prepared models. Support generation and
hollowing are **not** performed — use your slicer (Lychee, Chitubox,
PrusaSlicer, Elegoo SatelLite, etc.) after export.

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

Run **`stlbench --help`** for the command cheatsheet. The most common workflows:

```bash
# 1a. Generate a printer profile (edit width_mm / depth_mm / height_mm for your machine)
stlbench config init -o my_printer.toml

# 1b. Or generate a job-file template (includes [pipeline] and [[parts]] sections)
stlbench config job -o job.toml

# 2a. Full pipeline in one command: scale → orient → layout (recommended)
stlbench prepare -i ./parts -o ./plates -c my_printer.toml

# 2b. Mix raw and pre-prepared parts using a job file
stlbench job job.toml -o ./plates

# 3. Or run individual steps manually:
stlbench info    -i ./parts                -c my_printer.toml   # inspect dimensions
stlbench orient  -i ./parts -o ./oriented  -c my_printer.toml   # minimise supports
stlbench scale   -i ./oriented -o ./scaled -c my_printer.toml   # fit to build volume
stlbench layout  -i ./scaled -o ./plates   -c my_printer.toml   # pack onto plates

# Scale + pack all on one plate (no separate scale/layout steps)
stlbench autopack -i ./parts -o ./packed -c my_printer.toml

# Fill the bed with copies of a single part
stlbench fill -i ./part.stl -o ./filled -c my_printer.toml --scale
```

No config file? Specify the build volume inline as three numbers (X, Y, Z) in mm:

```bash
stlbench prepare -i ./parts -o ./plates -p "153.36,77.76,165"
```

## Commands

### `prepare` — Full pipeline: scale → orient → layout

```bash
stlbench prepare -i ./parts -o ./plates -c my_printer.toml
```

Runs the three preparation steps in the optimal order for resin printing:

1. **Scale** — finds the largest scale factor that fits every part inside the build
   volume (SO(3) orientation search), then applies a uniform scale to all parts.
2. **Orient** — rotates each scaled part to minimise overhanging surface area,
   constrained so the part still fits the build volume.
3. **Layout** — packs the oriented parts onto the minimum number of plates,
   distributed as evenly as possible.

Exports one `plate_NN.3mf` + `plate_NN.json` manifest per plate.

Key options: `--overhang-angle` (default 45°), `--orient-candidates` (default 200),
`--gap-mm`, `--post-fit-scale`, `--dry-run`, `--recursive`, `--resume`.

---

### `job` — Per-part configurable pipeline from a job file

```bash
stlbench job job.toml -o ./plates
```

Runs a flexible pipeline where each part can have its own set of steps. Useful when
some models are already oriented and supported (they only need packing) while others
need the full scale → orient → layout treatment. All parts end up on the same plates.

**Example `job.toml`:**

```toml
[printer]
width_mm  = 153.36
depth_mm  = 77.76
height_mm = 165.0

[scaling]
bed_margin     = 0.02
post_fit_scale = 0.95

[packing]
gap_mm = 2.0

[pipeline]
default_steps = ["scale", "orient", "layout"]   # default for parts without explicit steps

[[parts]]
path = "models/gandalf.stl"          # uses default_steps

[[parts]]
path = "models/staff.stl"
steps = ["scale", "layout"]          # scale but skip orient

[[parts]]
path = "supported/sword.stl"
steps = ["layout"]                   # already prepared — pack only
```

Valid step sequences (`layout` must always be last):

| `steps` | What happens |
|---------|-------------|
| `["scale", "orient", "layout"]` | SO(3) search → global scale → Tweaker-3 orient → pack |
| `["orient", "scale", "layout"]` | Tweaker-3 orient → scale from oriented AABB → pack |
| `["scale", "layout"]` | SO(3) search → global scale → pack |
| `["orient", "layout"]` | Tweaker-3 orient → pack |
| `["layout"]` | Pack only (model already prepared) |

Global scale is computed once across **all** parts that include the `scale` step, so
they all receive exactly the same scale factor.

Key options: `--candidates`, `--overhang-angle`, `--rotation-samples`,
`--grid-step`, `--dry-run`, `--verbose`.

---

### `info` — Inspect models (read-only)

```bash
stlbench info -i ./parts -c my_printer.toml
```

Displays a table with AABB dimensions, volume, vertex/face counts, whether each part
fits the bed, maximum scale factor, and how many copies would fit using `fill`.
No files are written.

---

### `scale` — Uniform scaling

```bash
stlbench scale -i ./parts -o ./scaled -c my_printer.toml
```

Computes a single scale factor so that **every** part fits inside the printer build
volume. The largest part determines the factor; all parts share the same scale.

Key options: `--dry-run`, `--no-upscale`, `--method sorted|conservative`,
`--post-fit-scale`, `--suffix`, `--recursive`.

---

### `orient` — Minimise support structures

```bash
stlbench orient -i ./parts -o ./oriented -c my_printer.toml
```

For each STL file, searches for the rotation that minimises overhanging surface area
(faces whose downward angle exceeds the threshold). Uses a two-phase search: discrete
evaluation of candidate orientations derived from the model's face normals, followed
by Nelder-Mead local refinement. The result is written with the bottom at z = 0.

When `--config` or `--printer` is supplied, the search is constrained to orientations
that fit inside the build volume.

Key options: `--overhang-angle` (default 45°), `--candidates` (default 200),
`--dry-run`, `--suffix`, `--recursive`.

---

### `layout` — Pack parts onto plates

```bash
stlbench layout -i ./scaled -o ./plates -c my_printer.toml
```

Arranges already-scaled STL files onto rectangular print plates. Exports
`plate_NN.3mf` + `plate_NN.json` with part positions. Multiple plates are created
if all parts do not fit on one.

Key options: `--dry-run`, `--gap-mm`, `--algorithm`.

---

### `autopack` — Scale + layout in one step

```bash
stlbench autopack -i ./parts -o ./packed -c my_printer.toml
```

Binary-searches for the maximum scale factor at which **all** parts fit onto a single
plate simultaneously. Combines `scale` and `layout` into one step with the goal of
keeping everything on one plate.

Key options: `--orient/--no-orient`, `--overhang-angle`, `--dry-run`, `--gap-mm`,
`--margin`, `--post-fit-scale`.

---

### `fill` — Maximum copies of one part

```bash
stlbench fill -i ./part.stl -o ./filled -c my_printer.toml
```

Packs as many copies of a single STL as possible onto one plate. Add `--scale` to
fit the part to the bed first, `--orient` to minimise supports before filling.

Key options: `--scale/--no-scale`, `--orient/--no-orient`, `--overhang-angle`,
`--dry-run`, `--gap-mm`.

---

### `config init` — Create a printer profile

```bash
stlbench config init -o my_printer.toml
```

Writes a printer profile TOML with `[printer]`, `[scaling]`, and `[packing]` sections.
Use `--stdout` to print without saving, or `--force` to overwrite.

### `config job` — Create a job-file template

```bash
stlbench config job -o job.toml
```

Writes a job-file template with all sections pre-filled: `[printer]`, `[scaling]`,
`[packing]`, `[pipeline]`, and two commented-out `[[parts]]` examples. Edit the file,
fill in your STL paths, then run `stlbench job job.toml -o ./plates`.

## Configuration

Printer profiles are TOML files. Generate a template with `stlbench config init` or
see [`configs/mars5_ultra.toml`](configs/mars5_ultra.toml) for a complete example
(ELEGOO Mars 5 Ultra).

| Section      | Keys                                                  | Purpose                              |
|--------------|-------------------------------------------------------|--------------------------------------|
| `[printer]`  | `name`, `width_mm`, `depth_mm`, `height_mm`           | Build volume (required)              |
| `[scaling]`  | `bed_margin` (0–1), `post_fit_scale` (>0)             | Margin and post-scale multiplier     |
| `[packing]`  | `gap_mm`                                              | Surface-to-surface gap between parts |
| `[pipeline]` | `default_steps`                                       | Default step list for `job` command  |
| `[[parts]]`  | `path`, `steps`                                       | Per-part entries for `job` command   |

## Output Files

All commands that write files produce **3MF** output (except `scale` and `orient`,
which write scaled/rotated STL files).

| Command    | Output                                           |
|------------|--------------------------------------------------|
| `prepare`  | `plate_NN.3mf`, `plate_NN.json` per plate        |
| `job`      | `plate_NN.3mf`, `plate_NN.json` per plate        |
| `scale`    | `*.stl` (one per input part, in-place scaled)    |
| `orient`   | `*.stl` (one per input part, rotated)            |
| `layout`   | `plate_NN.3mf`, `plate_NN.json` per plate        |
| `autopack` | `plate_NN.3mf`, `plate_NN.json` per plate        |
| `fill`     | `fill_plate.3mf`, `fill_plate.json`              |
| `info`     | Console output only                              |

The 3MF files use only the core 3MF 2015/02 namespace and are compatible with
Elegoo SatelLite, Chitubox, Lychee, and PrusaSlicer.

## Examples

See [`examples/README.md`](examples/README.md) for a step-by-step walkthrough
using the included Gandalf model (3 parts tracked via Git LFS).

## Package Structure

| Module              | Purpose                                                 |
|---------------------|---------------------------------------------------------|
| `stlbench.cli`      | Typer CLI — all commands and argument parsing           |
| `stlbench.config`   | Pydantic schema (`AppSettings`, `PartSpec`) + TOML loader |
| `stlbench.core`     | Scale factor computation, overhang analysis, orientation search |
| `stlbench.packing`  | 2D polygon packing onto plates (Shapely + custom grid)  |
| `stlbench.export`   | 3MF and JSON manifest writers                           |
| `stlbench.pipeline` | Command runners (`run_prepare`, `run_job`, `run_scale`, etc.) |

## Limitations

- Non-manifold meshes may produce incorrect AABB or scale results. Repair first
  with a tool such as Meshmixer or Microsoft 3D Builder.
- Supports and hollowing are not generated — open the exported 3MF in your slicer.

## License

[MIT](LICENSE)
