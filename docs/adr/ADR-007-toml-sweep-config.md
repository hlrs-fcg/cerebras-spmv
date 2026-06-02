# ADR-007: TOML Sweep Configuration (replacing CSV + argparse + Makefile)

**Status:** Accepted  
**Date:** 2026-06-02

## Context

Before this ADR, sweep configuration was split across three places:

1. **CSV files** in `bench/configs/` declared the matrix parameters and PE counts for each run
2. **argparse flags** (`--work-dir`, `--output`, `--start-index`, `--max-runs`) carried infrastructure parameters
3. **Makefile targets** were the only way to rerun a named sweep with its original settings

This split caused two problems. First, a CSV row cannot express a sweep *shape* (e.g. "weak scaling at 2 points per octave up to fabric max"); the CSV had to be pre-expanded by a separate generator script and then committed alongside the config. Second, reproducing a result required recovering both the CSV and the exact CLI invocation; neither alone was sufficient.

There was also no place to attach metadata (description, date, intent) to a sweep run.

## Decision

Use a **single TOML file per sweep** that owns all parameters: matrix spec, sweep shape, PE count sequence, and output paths. The CLI accepts one or more TOML paths as positional arguments plus three infrastructure-only flags (`--start-index`, `--max-runs`, `-v`). Nothing else.

### File format

A sweep TOML has three sections: `[meta]`, `[output]`, and either `[sweep]` for a parametric sweep or one or more `[[run]]` entries for an explicit list.

#### Parametric sweep (`[sweep]`)

```toml
[meta]
name        = "weak-scaling"
description = "NNZ per PE held constant as PE count grows"

[output]
work_dir = "cluster_work/sweeps/weak-scaling"
results  = "cluster/results/data/{date}_{name}.csv"

[sweep]
type       = "weak_scaling"   # or "strong_scaling"
density    = 0.01
seed       = 42
nnz_per_pe = 2000             # weak_scaling: rows = nnz_per_pe * num_pes / density

[sweep.num_pes]
pow2_up_to        = "fabric_max"  # or an integer ceiling
points_per_octave = 2             # 1 = powers of 2 only; 2 = also 2^(n+0.5) = 2^n * sqrt(2)
```

For `type = "strong_scaling"`, replace `nnz_per_pe` with `rows` and `cols`:

```toml
[sweep]
type    = "strong_scaling"
rows    = 10_000_000
cols    = 10_000_000
density = 0.001
seed    = 42

[sweep.num_pes]
pow2_up_to        = "fabric_max"
points_per_octave = 2
```

#### Explicit run list (`[[run]]`)

For ad-hoc or one-off configurations, use a list of run tables:

```toml
[[run]]
rows    = 10_000
cols    = 10_000
density = 0.01
seed    = 42

[[run]]
rows    = 100_000
cols    = 100_000
density = 0.001
seed    = 42
num_pes_target = 4096
```

Any field accepted by `SpMVConfig` is valid in a `[[run]]` block. `num_pes_target` is optional; omitting it uses the full fabric.

### PE count sequence

`pow2_up_to = "fabric_max"` resolves to 857,010 (the WSE-3 usable fabric: $742 \times 1155$). With `points_per_octave = 2` the sequence is:

$$\{ \lfloor 2^{n/2} \rceil \mid n \in \mathbb{N},\; \lfloor 2^{n/2} \rceil \leq P_\text{max} \}$$

starting from $n = 12$ (giving $2^6 = 64$). This produces two points per octave:

| $n$ | value |
|---|---|
| 12 | 64 |
| 13 | 91 |
| 14 | 128 |
| 15 | 181 |
| ... | ... |
| 38 | 524,288 |
| 39 | 741,455 |

28 data points from 64 to 741,455 for the full WSE-3 range.

### Output path interpolation

`{date}` expands to `YYYY-MM-DD` at run time; `{name}` expands to `meta.name`. This makes the results file self-describing without manual renaming.

### CLI

```
uv run python run.py run  --rows N --cols N --density F [--num-pes N] [--work-dir DIR] [-v]
uv run python run.py sweep <config.toml> [<config.toml> ...] [--start-index N] [--max-runs N] [-v]
```

`run` compiles and executes a single inline configuration; `sweep` loads one or more TOML files and runs their expanded config lists. Both share the same compile/run path. Example sweep configs live in `bench/configs/`.

## Rationale

- **Single source of truth**: one file contains everything needed to reproduce a sweep. No external generator script or Makefile target is needed.
- **Self-describing sweep shapes**: `weak_scaling` with `nnz_per_pe` and `strong_scaling` with `rows/cols` express intent directly. The CSV approach required pre-expanding the shape and losing that information.
- **Pow2 sequence is a derived quantity**: expressing it as `pow2_up_to` + `points_per_octave` rather than as an enumerated list avoids the list going stale when the fabric size or desired density changes.
- **No two-level configuration**: argparse flags previously duplicated parameters that were logically part of the experiment (output path, work dir). Moving them to TOML means the CLI can stay minimal and stable.
- **TOML over JSON/YAML**: `tomllib` is Python 3.11 stdlib (no dependency); TOML supports inline comments, integer literals with underscores (`10_000_000`), and is more readable than JSON for nested tables.

## Consequences

- `cluster/run_sweep.py` is deleted; `cluster/run.py` is the new entry point.
- Root-level `compile.py` and `run.py` are deleted; single-config compilation is done via `cluster/run.py sweep` with a one-entry `[[run]]` TOML or via direct Python API.
- All existing CSV sweep configs in `bench/configs/` are superseded. The files are removed; their content can be reconstructed from the parametric TOML equivalents.
- `{date}` interpolation in `results` means the same TOML file re-run on a different day produces a new output file automatically, which is the desired behaviour for repeat measurements.
- `tomllib` is read-only (stdlib); no third-party TOML writer dependency is needed.
