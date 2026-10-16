# v1.0 results (ISC26 poster, January 2026)

The raw CSVs behind the ISC26 poster, measured with the v1.0 code (tag `v1.0`, ADR-000), kept as they
were written. 123 rows in 17 files.

- **Provenance.** `git_commit` is `unknown` in every row: the v1.0 launcher recorded the commit of
  the working directory on the cluster, which had no `.git`. Read them as "tag `v1.0`". Since v2.0 every
  row names its commit, whether the tree was clean, and `git describe` (`sdk/build_info.py`).
- **Schema.** 44 to 50 columns, the v1.0 `BenchmarkResult`. Timings are host-return times, which the
  v2.0 work showed to be lower bounds for streamed transfers (`docs/host-timing-barrier.md`); the
  v1.0 kernel resident-A numbers are per-PE cycle maxima, unsynced. In the v1.0 files the
  `compute_time_hw_*` columns are in milliseconds despite the name, so `gflops_hw` is 1000x too low.
- **Pairs.** The `2026-01-15_*` and `2026-01-18_*` files are two runs of the same sweeps (duplicates; the weak-scaling file of
  `2026-01-19` is a third).
- **Units.** `compute_time_hw_*` is in milliseconds although the name says seconds, so `gflops_hw_*` is 1000x too low. Seconds are
  `compute_hw_cycles_* / 0.85e9`. `total_run_time` is H2D + compute + D2H + gather as the host measured them; the per-call partitioning
  (`distribute_time`, `prepare_matched_pairs_time`) is separate.
- **Sizes.** No full-wafer run: the largest grid is 524,288 PEs (`2026-01-20_strong-scaling-large.csv`, a 40M-NNZ matrix); the 65,536-PE
  weak-scaling point has 65.5M NNZ (1000 per PE).
- **Not comparable** with the v2.0 benchmarks (different kernel, timing basis and tiling), so do not
  mix the two directories in one plot.

New results are written to `cluster/results/data/` by default; commit the ones you want kept.
