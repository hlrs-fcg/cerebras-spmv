# Archive

Material that is no longer part of the SpMV code but is kept for the record. Nothing here is run
by the tests or imported by the code, and it is not maintained.

| Path | What it was |
|---|---|
| `docs/streaming-rewrite-status.md` | The 2026-09/10 streaming rewrite log and fix plan (superseded by `docs/adr/` and the results in the paper) |
| `docs/sketch-streaming-spmv.md` | The first design sketch of streaming SpMV (superseded by ADR-001..005) |
| `docs/recovering-shifted-benchmark-csv.md`, `bench/repair_csv.py` | Salvaging CSVs written by a positional append bug (fixed in `bench/benchmark.py:append_rows`) |
| `bench/analyze_h2d.py`, `bench/configs/h2d-*.toml` | The 2026-10-01/03 H2D shape/order/channel experiments; their question moved to the microbenchmarks (cs3-microbench) and the rules to ADR-008 |

The 2026-10 development result CSVs are on the branch `archive/dev-results-2026-10` of the
private development repository; the removed `--x-per-pass` experiment is on `archive/x-per-pass`.
