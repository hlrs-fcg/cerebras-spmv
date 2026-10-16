# ADR-008: Host↔Wafer Transfer and Timing Rules

**Status:** Accepted
**Date:** 2026-10-08

## Context

Until 2026-10-07, every transfer time in this project was a host-return time. A blocking `launch` or memcpy returns before the wafer is done, so those numbers were lower bounds and the rates built on them were too high. The channel and shape conclusions drawn from them turned out wrong ("consumption-bound", "flat in channels"). We then measured the interface directly with ten microbenchmarks, M1–M10, at 65,536 PEs. Every time ends at a completion barrier or uses the synced on-wafer clock. The code, data, notebooks and vendor quotes now live in the private repo `hlrs-fcg/cs3-microbench` (`analysis/SUMMARY.md` there). SDK 2.10.0, CS-3 at EPCC.

## Decision

All host↔wafer code in this project follows these rules.

1. **Timing.** Never time with a call's return. Close every host bracket with a completion barrier: a small blocking D2H that can only finish after the work, minus its idle-repeat cost (`docs/host-timing-barrier.md`). Or use the synced on-wafer clock. At 8 and 16 channels a blocking H2D returns with ~29% of the transfer still in flight; a blocking `launch` returns after ~13 µs regardless of the device.
2. **H2D shape.** Send many narrow rectangles (k ≈ 2 × channels or more), issued one after another, never one full W×H rectangle. Ceiling: 1.13 / 5.15 / 10.3 GB/s at 1 / 8 / 16 channels. A full rectangle gets 3.5× less, and is the worst shape in every mode. Row vs col order doesn't matter for narrow rectangles.
3. **D2H shape.** The opposite: one full rectangle in col-major order (or 1-wide columns). Ceiling: 0.89 / 3.8 / 6.3 GB/s. Never one-PE-tall rows (~70× slower).
4. **Channels.** Use 16 (SdkRuntime's maximum); the ceiling is still rising there.
5. **Mode.** Streaming and copy mode reach the same ceiling (M2 = M1). Choose by algorithm, not speed. A bulk DSD receive is as fast as a per-word data task (M3).
6. **Calls.** More, smaller calls are faster for the same bytes (M5). The per-call cost (31–57 µs) only matters below ~4 words per PE.
7. **Synchronisation.** Synchronise with a D2H read (167 µs for one word, 235–470 µs for all PEs), never a launch. Prefer on-wafer collectives: a scalar allreduce over 256×256 costs ~31 µs.
8. **Edge buffers.** `--width-west-buf` / `--width-east-buf` have no measurable effect for an idle core; leave them at 0. If used, `--fabric-offsets` x must be 4 + `width_west_buf` (a cslc requirement).
9. **Statistics.** Report n ≥ 3 with min / median / max and the system name. The first repetition can be 15–130% slow. Results from a system that has failed a trivial job are discarded.

## Rationale

- Each rule rests on a barrier-verified microbenchmark with n ≥ 3, reproduced across three systems. Serial reruns match the parallel sweeps within noise.
- The vendor's own bandwidth example agrees with rule 1: it says host timing "may not show the true timing" and times on the device (`csl-examples/benchmarks/bandwidth-test/run.appliance.py:57–60`). The SDK guide's "If blocking mode (`nonblock=False`) is selected, the API waits until the operation is done" is contradicted by rule 1's measurements.
- The vendor's col-major advice holds for copy mode (rule 3, and +30% for the SpMV's full-rectangle setup copy) but not for streaming into our kernel. Its example's "around 7 Gbps" per channel is exceeded at 1 channel and reached only to 74% at 8 and 16.

## Consequences

- ADR-001 (A stays resident for CG), ADR-003 (valtask), ADR-004 (transfer shape, setup in strips) and ADR-005 (on-wafer dot products, D2H shape, sync) carry 2026-10-08 amendments built on these rules.
- The SpMV setup copy should be split into narrow strips (not yet done). The streaming H2D already uses column mode, which matches rule 2.
- Open: why D2H stays at 61% of H2D at 16 channels; whether our SpMV kernel keeps up with the narrow-strip ceiling; the val/valtask hang cause; and whether the in-flight share saturates at 1 channel beyond 2.4 GB.

## Amendment (2026-10-16): the launch floor and the call from the host (v2.0)

- **Subtract the launch floor.** Every synced window contains a launch overhead that does not depend on the work. Run a null twin (`--null-spmv`: same program and launch, empty SpMV body) and report net = window minus floor. Floors are 448 / 1,485 / 4,845 cycles at 16 / 65K / full wafer (0.5 / 1.75 / 5.7 µs at 850 MHz). It matters at the full wafer (29.6 µs window, 23.9 µs net) and for small matrices.
- **A host-visible call costs 0.86 ms (65K) to 0.93 ms (full wafer)**, 30 to 50 times the on-wafer window. The resident SpMV beats the host CPU on the window, but a loop of single host calls does not; the comparison holds only if the iteration loop runs on the wafer (ADR-009, ADR-005).
- Setup costs for break-even arithmetic: program load 90 s (65K) and 216 to 221 s (full wafer); streaming valtask setup 46.9 ms (65K) and 422 ms (full wafer).
