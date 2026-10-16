# SpMV barrier re-measure: summary (2026-10-07)

Scope: the SpMV streaming results only. The host↔wafer microbenchmarks M1–M10 and their notebooks are in
**[hlrs-fcg/cs3-microbench](https://github.com/hlrs-fcg/cs3-microbench)** (private repository) (`analysis/SUMMARY.md` there).

All times are barrier-closed (a host-return time is never used for a rate), 65,536 PEs (256×256), 114,000² at
d=1e-3 (13.0 M non-zeros), medians of n=3 unless stated. Notebook: `spmv_barrier_remeasure.ipynb`; data:
`cluster/results/data/2026-10-07_barrier-remeasure.csv` and `2026-10-07_s1-column-matrix.csv`.

| Question | Headline |
|---|---|
| valid timing | host-return never above barrier time (0/28 rows); full-mode tail 82% at 1 ch (header) |
| header vs valtask | valtask 2.6× / 1.9× / 1.8× faster than header (1 / 8 / 16 ch, medians; the 16-ch ranges overlap) |
| full vs column | column mode 107 / 59 / 62–66 ms; valtask column 41 / 46 ms (n=2 / 3, partial S1 file, interim) |
| setup copy | col-major +30% at 16 channels |

What the microbenchmarks add for the SpMV (details in cs3-microbench): H2D reaches 1.13 / 5.15 / 10.3 GB/s at
1 / 8 / 16 channels when a transfer is sent as many narrow rectangles, and a single full rectangle (the SpMV
setup and full-mode streaming) gets 3.5× less; host-side times understate a transfer by about 29% at 8 and 16
channels; on the wafer the SpMV's east-west reduction chain beats the vendor allreduce library's row reduction.
The rules this project adopts are in `docs/adr/ADR-008-host-wafer-transfer-and-timing.md`.

Open: whether the SpMV kernel itself keeps up with narrow-strip streaming; the cause of the val/valtask hangs
(the microbenchmarks reproduced none).
