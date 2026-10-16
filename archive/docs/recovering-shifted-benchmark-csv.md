# Recovering a positionally-shifted benchmark CSV

**Status:** the cause is fixed; this note exists to salvage files written before
the fix. One such file is known: `cluster/results/data/single_114000x114000.csv`
on the cluster, holding the 65,536-PE channels sweep.

## What went wrong

`sdk/launcher.py._download_benchmark` appended the rows it downloaded from the
worker using **the downloaded file's** fieldnames, into a destination file whose
header had been written under an older, narrower schema.

`csv.DictWriter` does not compare against the existing header. It just writes
the values in the order of the `fieldnames` it was handed. So every appended
line carried the **wide** column count in **wide** order underneath a **narrow**
header, and each write reported success.

Reading the file back maps `wide_value[i]` onto `narrow_header[i]`. Everything
after the first inserted column lands under the wrong key.

Eight columns were added across two changes:

| Added | Change |
|---|---|
| `kx`, `ky`, `num_col_stripes`, `tile_extract_time`, `h2d_bytes_total` | streaming rewrite |
| `channels`, `width_west_buf`, `width_east_buf` | memcpy I/O configurability |

They were inserted mid-schema, not appended, so the shift starts early and the
displacement varies by column.

## Recognising it

- a duration column holding an absurd magnitude — `compute_time_hw_avg` reading
  `112086016.0` seconds was the tell here
- the value you expected showing up in a *different* column a few positions
  along (the real `0.039398 s` had landed in `compute_hw_cycles_min`)
- columns you know were being swept missing from the header entirely — the
  channels sweep recorded no `channels` column at all
- data lines with more fields than the header

Quick check without repairing anything:

```bash
python bench/repair_csv.py cluster/results/data/single_114000x114000.csv --dry-run
```

## Repairing

Nothing is lost. The values are all present and in wide order; only the keys are
wrong, so the repair is exact — re-zip each over-long line against the current
schema.

```bash
# look first
python bench/repair_csv.py cluster/results/data/single_114000x114000.csv --dry-run

# repair in place (keeps a .bak)
python bench/repair_csv.py cluster/results/data/single_114000x114000.csv

# or leave the original untouched
python bench/repair_csv.py cluster/results/data/single_114000x114000.csv -o repaired.csv
```

Rows are classified by field count, which is unambiguous while the two schemas
differ in width (they differ by 8):

- `len(row) == len(header)` — written under the old schema, already aligned;
  backfilled from the `BenchmarkResult` defaults
- `len(row) == len(current schema)` — positionally appended, re-keyed
- anything else — reported and left out, never guessed at

Backfilling old rows from the dataclass defaults is correct for them: every row
predating these columns was measured at `channels=1` with no edge buffers.

The helper refuses to run if the old header happens to have the same width as
the current schema, since row width could not then discriminate the two.

## Expected values for `single_114000x114000.csv`

Seven rows at **65,536 PEs**, all validated at relative error **1.49e-07**.
Use these to confirm the repair landed:

| Row | channels | compute time |
|---|---:|---:|
| earlier run | 1 | 39.308 ms |
| sweep | 1 | 39.398 ms |
| sweep | 2 | 44.878 ms |
| sweep | 4 | 56.228 ms |
| sweep | 8 | 62.224 ms |
| sweep | 11 | 64.154 ms |
| sweep | 16 | 62.697 ms |

Note the `channels` column cannot be recovered from the file — it was never
written, because the header lacked the column. After repairing the alignment,
set it by hand from the order above.

Result: more channels is monotonically **worse** at this size, plateauing around
62–64 ms. `channels=1` remains the default.

## Why it cannot happen again

`bench/benchmark.py:append_rows()` is now the single writer for both
`save_benchmark()` and `_download_benchmark()`, and it compares the destination
header against the fieldnames it is about to write:

- header identical → plain append
- header is a strict subset → migrate the file to the wider schema, backfilling
  old rows from the dataclass defaults, keeping a `.bak`, swapping atomically
- header otherwise incompatible → write to a `<stem>.schemaN.csv` sibling and log
  an error; the original is left untouched
- no file → create it with the header

Positional append is not reachable from either path.
