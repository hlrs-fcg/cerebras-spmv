# Microbenchmarks (moved)

The host↔wafer microbenchmarks M1–M10 (copy and streaming H2D/D2H, overlap, per-call overhead,
sync latency, on-wafer collectives, edge buffers, a noise probe), their kernels, the planner and
launcher (`python -m bench.micro`), the raw results and the analysis notebooks now live in their own
repository, **[hlrs-fcg/cs3-microbench](https://github.com/hlrs-fcg/cs3-microbench)** (private). Its `docs/microbenchmarks.md` is the
design, commands and risks; its `analysis/SUMMARY.md` is the result in one page. Lessons from them that
this project acts on are in `docs/adr/ADR-008-host-wafer-transfer-and-timing.md`.
