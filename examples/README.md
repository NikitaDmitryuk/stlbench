# stlbench Examples

## Gandalf (3-part model)

The `gandalf/` directory contains a three-part model of Gandalf stored via Git LFS:
`figure.stl`, `staff.stl`, and `sword.stl`. The `gandalf/scaled/` subdirectory holds
pre-scaled versions of these parts.

> **Git LFS:** Run `git lfs pull` if the STL files appear as text pointer files.

All commands below are run from the **repository root** (next to `stlbench/` and
`examples/`), and use the included ELEGOO Mars 5 Ultra config.

---

## 1. `info` — Inspect parts before doing anything

```bash
stlbench info -i examples/gandalf/main -c configs/mars5_ultra.toml
```

Prints a table for each STL file showing AABB dimensions, volume, vertex/face counts,
whether the part fits the bed at 1:1 scale, the maximum scale factor, and how many
copies would fit using `fill`. No files are written.

---

## 2. `prepare` — Full pipeline in one command

```bash
stlbench prepare \
  -i examples/gandalf/main \
  -o examples/gandalf/plates \
  -c configs/mars5_ultra.toml
```

Runs scale → orient → layout in one step. Writes `plate_01.3mf` + `plate_01.json`
(and additional plates if needed) to `examples/gandalf/plates/`.

Add `--dry-run` to see what would happen without writing any files.

---

## 3. `job` — Mix raw and pre-prepared parts

Create a `job.toml` that packs the full-pipeline figure alongside the already-scaled
sword (which only needs to be placed on the plate):

```toml
[printer]
width_mm  = 153.36
depth_mm  = 77.76
height_mm = 165.0

[pipeline]
default_steps = ["scale", "orient", "layout"]

[[parts]]
path = "examples/gandalf/main/figure.stl"

[[parts]]
path = "examples/gandalf/scaled/sword.stl"
steps = ["layout"]
```

```bash
stlbench job job.toml -o examples/gandalf/job_out
```

Writes `plate_NN.3mf` + `plate_NN.json` with all parts packed together.

---

## 4. `scale` — Scale to fit the build volume

```bash
stlbench scale \
  -i examples/gandalf/main \
  -o examples/gandalf/scaled \
  -c configs/mars5_ultra.toml
```

Computes a single scale factor so that every part fits inside the printer volume.
Writes scaled STL files to `examples/gandalf/scaled/`.

Useful options:
- `--dry-run` — show the scale factor without writing files
- `--no-upscale` — cap scale at 1.0 (never enlarge)
- `--post-fit-scale 0.95` — apply an extra 5% safety margin

---

## 5. `orient` — Minimise support structures

```bash
stlbench orient \
  -i examples/gandalf/main \
  -o examples/gandalf/oriented \
  -c configs/mars5_ultra.toml
```

Rotates each part to minimise overhanging surface area. Writes re-oriented STL files
with the bottom at z = 0, ready for the next step.

Useful options:
- `--overhang-angle 45` — surfaces steeper than this need support (default 45°)
- `--candidates 200` — number of face-normal candidates to evaluate

---

## 6. `layout` — Pack parts onto print plates

```bash
stlbench layout \
  -i examples/gandalf/scaled \
  -o examples/gandalf/plates \
  -c configs/mars5_ultra.toml
```

Arranges the scaled STL files onto rectangular plates. Exports `plate_01.3mf` +
`plate_01.json` (and further plates if needed).

Useful options:
- `--dry-run` — report plate count without exporting
- `--gap-mm 1.0` — override gap between parts

---

## 7. `autopack` — Scale + pack all parts onto one plate

```bash
stlbench autopack \
  -i examples/gandalf/main \
  -o examples/gandalf/autopack \
  -c configs/mars5_ultra.toml
```

Finds the maximum scale factor at which **all** parts fit onto a single plate.
Exports `plate_01.3mf` + `plate_01.json`.

---

## 8. `fill` — Maximum copies of a single part

```bash
stlbench fill \
  -i examples/gandalf/scaled/sword.stl \
  -o examples/gandalf/fill_sword \
  -c configs/mars5_ultra.toml
```

Packs as many copies of `sword.stl` as possible onto one plate. Exports
`fill_plate.3mf` + `fill_plate.json`.

Add `--scale` to scale the part down to fit the bed first:

```bash
stlbench fill \
  -i examples/gandalf/main/sword.stl \
  -o examples/gandalf/fill_sword_scaled \
  -c configs/mars5_ultra.toml \
  --scale
```

---

## Output file summary

| Command    | Output directory                    | Key files                              |
|------------|-------------------------------------|----------------------------------------|
| `info`     | *(console only)*                    |                                        |
| `prepare`  | `examples/gandalf/plates/`          | `plate_NN.3mf`, `plate_NN.json`        |
| `job`      | `examples/gandalf/job_out/`         | `plate_NN.3mf`, `plate_NN.json`        |
| `scale`    | `examples/gandalf/scaled/`          | `*.stl` (one per input part)           |
| `orient`   | `examples/gandalf/oriented/`        | `*.stl` (one per input part)           |
| `layout`   | `examples/gandalf/plates/`          | `plate_NN.3mf`, `plate_NN.json`        |
| `autopack` | `examples/gandalf/autopack/`        | `plate_01.3mf`, `plate_01.json`        |
| `fill`     | `examples/gandalf/fill_sword/`      | `fill_plate.3mf`, `fill_plate.json`    |

All output directories are listed in `.gitignore` and are not committed to the repo.

---

## Notes

- **Supports are not generated by stlbench.** After exporting, open the 3MF in
  Elegoo SatelLite, Lychee, Chitubox, or another slicer to add supports.
- The 3MF files use only the core 3MF 2015/02 namespace and open correctly in all
  major resin slicers.
- The Gandalf model files are tracked with **Git LFS**. Run `git lfs pull` if the
  STL files appear as pointer text files.
