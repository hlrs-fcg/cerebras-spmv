# Synced on-wafer timing (`--sync-timing`, `--loop-count`)

**Date:** 2026-10-06 · **Branch:** `fix/val-column-and-tsc` · **State:** host side
and models tested, **CSL never compiled** (the SDK is Linux-only). Default off.

## Why

The existing tsc columns (`compute_hw_cycles_*`) come from one stamp pair per PE:
first streamed word (header) or DSD arm (val) → `chain_done`. Those counters are
**unsynchronised**: every PE's tsc has its own offset, so `max(end) − min(start)`
across PEs means nothing, and the columns can only be read as a per-PE spread.
They are also single-shot, so there is no variance. The host figure
(`compute_time_hw_*`) includes the runtime's command and transfer latency.

## What it does

Cerebras' own pattern (`csl-examples/benchmarks/bandwidth-test`, `sync/`):

```
f_sync                  once: P0.0's broadcast reaches every PE; each stamps ref
for pass in loop_count:
    f_tic               each PE stamps tic
    spmv → stream → wait_ew (nonblocking)
    f_toc (blocking)    each PE stamps toc right after its own wait_ew releases,
                        packs {tic, toc, ref} into sync_pack[5]
    completion barrier  1-word D2H of y_result[0] from column 0, timed, then
                        repeated on the idle device (docs/host-timing-barrier.md)
    D2H y_result, D2H sync_pack        (outside the bracket)
```

**The completion barrier is inside every pass.** The host brackets (`memcpy_h2d_time`,
`compute_time_hw_*`) end when a blocking call returns, and that can be before the
device is done. So each pass is closed by its own barrier, and `device_done_time`,
`barrier_d2h_*` and `stream_time_corrected` are per-pass means, not a single
measurement of the last pass. The barrier runs after `f_toc` has returned, so it
never enters the tic/toc bracket: `wafer_*` is unaffected. `loop_count = 1` gives
exactly the single-pass figures of the barrier commit.

Per pass the host computes `ref' = ref − (px + py)` (the broadcast takes one
cycle per hop from P0.0, the same correction as bandwidth-test's `run.py`), puts
every stamp on that clock, and records **`max(toc − ref') − min(tic − ref')`**:
the time from the first PE starting the pass to the last PE finishing it, on one
clock. Over `loop_count` passes it reports min / avg / max / std.

**Sync once, not per pass.** The reference stays valid as long as the tsc
free-runs. The sync module enables it at startup, and under `SYNC_TIMING` the
kernels never call `enable_tsc()`/`disable_tsc()` again (each kernel's
`chain_done` disables it when the flag is off, as before). One sync is exactly
the upstream usage, so the protocol is used as Cerebras ships it.

**Unsynced path unchanged.** With the flag off, the pass is the old
`spmv → stream → wait_ew (blocking)`. With `loop_count = 1`, every existing
column gets the same value as before.

## Where it lives

| piece | file |
|---|---|
| sync PE module, vendored from csl-examples @ 162e0f3; `pe_off.csl` is the flag-off stand-in | `src/csl/sync/{pe,pe_off}.csl` (the vendored `sync/layout.csl` is gone, see SDK 2.10 below) |
| `SYNC_TIMING` param, sync colors/tasks, gated exports | `src/csl/layout.csl` |
| `f_sync` / `f_tic` / `f_toc`, `sync_pack`, tsc gating | `src/csl/pe.csl`, `src/csl/pe_valtask.csl`, `src/csl/pe_resident.csl` (identical blocks) |
| pass loop, `unpack_sync_pack`, `synced_pass_cycles`, `pass_stats` | `cluster/worker_device.py`, `cluster/worker_timing.py` |
| flags | `SpMVConfig.sync_timing` (compile-time), `.loop_count` (runtime) |

The vendored module has two local edits, both so that it is **inert when
off**. Every comptime block (binds, the STARTUP activation, WSE-3 queue init,
routes) is wrapped in `if (enabled)`, and the six fabric DSDs are `const`. A
test diffs it against the upstream checkout.

**Resources.** The sync module needs 5 colors, 3 input/output queues and 4
local tasks, all chosen to stay clear of what the kernels already use:

| | sync | already used |
|---|---|---|
| colors | 1, 3, 4, 5, 6 | 0 (MEMCPYH2D_DATA_1), 7/8 (EW); 2 left for a D2H stream |
| queues | in/out 3, 4, 5 | in 2 (H2D stream), in/out 6 (EW); 0/1 memcpy |
| local tasks | 15–18 (bandwidth-test's) | 10, 11 |

The module has no routing for a 1-wide or 1-tall grid. `SpMVConfig` refuses
such grids under `sync_timing`, and `layout.csl` asserts the same at compile
time. SRAM: +11 words per PE, counted by the guard when the flag is on.

## Multi-pass correctness

`loop_count > 1` re-sends the same A each pass, and **every pass's y is read
back and validated**. `relative_error` is the worst pass, and one bad pass
fails the run. Every kernel resets per pass. `pe.csl`'s `spmv()` clears
all KY rows of `y_result` and resets the state machine. `pe_valtask.csl`'s
`local_spmv()` writes every row, and its data task overwrites `vals[0..n)` every pass. `tests/test_sync_timing.py` runs five
passes on stateful models of both kernels: garbage initial `y_result`/`vals`,
persistent queues, repeats, then a new x, then new values. It covers three
geometries in column and full mode. It requires bit-identical repeats and
bitwise header/val agreement on every pass. A mutation (dropping the
`y_result` clear) is caught.

## CSV

| column | meaning |
|---|---|
| `loop_count` | passes run (1 for every older row) |
| `sync_timing` | flag (False for older rows) |
| `wafer_cycles_{min,avg,max,std}` | per-pass synced span over passes, cycles |
| `wafer_time_{min,avg,max,std}` | the same at 850 MHz, seconds |
| `compute_time_hw_*` | now real per-pass host statistics (identical to before at 1 pass) |
| `device_done_time`, `barrier_d2h_time`, `barrier_d2h_repeat_time`, `stream_time_corrected` | means over passes of the per-pass barrier-closed brackets (identical at 1 pass); `compute_time_hw_*` stay host-return times, a lower bound |
| `memcpy_h2d_time`, `memcpy_d2h_time` | per-pass means |
| `h2d_bytes_total` | bytes of **one** pass (unchanged meaning at 1 pass) |
| `host_pass_time_{avg,std}` | host time per pass from before `f_tic`/`spmv` to the completion barrier returning: the true upper bound |
| `wafer_host_ratio_max` | max over passes of `wafer_time / host_pass_time`; must be ≤ 1 |

`wafer_*` stay 0 unless every pass read back cleanly. A PE that never stamped
is an error, not a number, and the worker logs a greppable `RECORDED AS 0` line.

**Reading it.** `wafer_time_avg` vs `compute_time_hw_avg` separates on-wafer
time from host/runtime overhead. Unlike the per-PE span, it is comparable
across the two encodings, because both are bracketed by the same tic/toc.
Two caveats. First, tic is stamped when each PE's command stream reaches
`f_tic`, so the span includes command dispatch to the earliest PE. That is the
same definition bandwidth-test uses. Second, the span includes the stream's
arrival, by design, because the stream is what we are measuring.

## Cross-check against host wall time

**The 88.7M-cycle val span is not impossible. The host figure under-measures
it.** At 65,536 PEs in full mode at channels 1, pe_val's per-PE tsc span read
88.7M cycles (104 ms at 850 MHz) against 21.99 ms of `compute_time_hw_avg`.
Every recorded row where the span exceeds the host figure has an inflated y
D2H right after it:

| run (65,536 PEs) | compute+barrier | y D2H | span max @ 850 MHz |
|---|---|---|---|
| val full ch1 | 22.0 ms | **193.0 ms** | 104.4 ms |
| header column ch1 | 39.4 ms | **155.6 ms** | 86.3 ms |
| header column ch2 | 44.9 ms | 92.8 ms | 72.4 ms |
| header full ch8 | 19.8 ms | 49.9 ms | 18.9 ms |
| header column ch8 | 62.2 ms | 1.1 ms | 54.9 ms |

The span always fits inside compute+barrier+D2H, and it exceeds
compute+barrier exactly when the D2H is long. So `launch("wait_ew",
nonblock=False)` can return before the device has finished the pass. The
remainder is absorbed by the next blocking call, the y D2H, which cannot
complete before the device has. The tsc and the stamp placement are
consistent with that. The span is uniform across PEs for val (88.705M–88.713M)
because pe_val stamps `tsc_start` when `spmv()` arms the DSD, which is nearly
simultaneous everywhere. **Consequence beyond synced timing:** when the y D2H is
long, `compute_time_hw_*` (and rates derived from it) understate the pass. The
val full/ch1 point looked like 22 ms and was really ≥ 104 ms.

**Does the synced path inherit it? No.** Its stamps are device-side and read
by a blocking D2H queued behind `f_toc`, so when the host returns does not
matter. It also never calls `enable_tsc()`/`disable_tsc()` per pass: the sync
module enables the tsc once, and under `SYNC_TIMING` the kernels' calls are
compiled out. Its numbers do inherit the 850 MHz assumption, like everything
else.

**The cross-check, built in.** Per pass, the host bracket starts *before*
`f_tic` is issued (moved for this) and ends when the completion barrier D2H
returns (it was the y D2H before the barrier was in the pass loop; the barrier
is tighter, as it adds neither the y read nor the idle repeat)
(`host_pass_time_*`). Every device stamp lies inside it, so
`wafer_host_ratio = wafer_time / host_pass_time` must be ≤ 1. Each pass logs
the ratio, `> 1` logs a WARNING that the wafer numbers are untrustworthy, and
the CSV records `wafer_host_ratio_max`. The per-PE span gets the same check
against the last pass's bracket. Separately, a warning says when the span
exceeds compute+barrier but fits the full bracket, i.e. when
`compute_time_hw_*` under-measured the run. The recorded rows above are
pinned in `tests/test_sync_timing.py`. All of them pass the check, and two
fail the naive compute+barrier comparison.

## Risks: only cslc and hardware can settle these

1. **Compile: SDK 2.10.0 (found 2026-10-07).** The first cslc run on the cluster
   rejected `sync/layout.csl:38:31`, the `comptime_struct` return type of
   `get_params`, with "use of undeclared identifier". The SDK 2.10.0 release removed
   `comptime_struct` and `@concat_structs`; the csl-examples we vendored from target
   SDK 1.4.0. Evidence: Cerebras' migration guide
   (https://sdk.cerebras.net/csl/comptime-struct-migration: "removed in SDK 2.10.0",
   replace by declared `struct` types). The fix avoids struct types altogether:
   `layout.csl` computes the per-PE sync colors and flags inline (same assignment as
   the old `get_params`) and passes them as 13 flat typed params, and each kernel
   hands them to the sync module as an anonymous struct literal, as it already does
   for `memcpy_params`. The kernels' own `param memcpy_params;` is untyped and compiled
   on main with 2.10.0. A test fails if `comptime_struct` or `@concat_structs`
   appears in any `.csl` here.
   **Flag off now imports nothing from `sync/`** except the tiny stub
   `sync/pe_off.csl`: each kernel picks the module path with a comptime function
   (`sync_pe_file()`, the pattern csl-examples' `stencil_3d_7pts/pe.csl` uses for
   `get_stencil_module()`), so a problem in the real sync module cannot break the
   normal kernels. Still unverified by cslc: the path-selecting function, an
   `if` expression over colors in the tile params, a module `var` read through
   `sync_mod.tscRefBuffer`.
2. **Flag-off SRAM.** The inert module and the tic/toc buffers may still be
   allocated (about 11 words), depending on dead-code elimination. That does
   not affect timing.
3. **Sync and the EW chain.** The sync colors use routes on every PE, as EW
   does. They are distinct colors, so they should not interact, but this has
   not been exercised together.
4. **Artifacts.** `SYNC_TIMING:0` is now in every option string, so existing
   artifacts recompile, as happened with `ENCODING`.

## Commands for bench

Run these in order. Each compile takes ~70 s at these sizes. Stop at the first
failure.

```bash
# 1. regression, flag OFF: both kernels must reproduce the known 16-PE result (5.32e-08)
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding valtask --h2d-mode full
# 2. smoke, flag ON, 5 passes, both kernels
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --sync-timing --loop-count 5
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding valtask --h2d-mode full --sync-timing --loop-count 5
# 3. the A/B point, synced: 65,536 PEs, full / row / 16 channels, 5 passes
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --h2d-mode full --memcpy-order row --channels 16 --encoding header --sync-timing --loop-count 5
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --h2d-mode full --memcpy-order row --channels 16 --encoding valtask --sync-timing --loop-count 5
```

Step 2 can be tried on the simulator first by adding `--simulator`. It
exercises the sync routing and the command sequence without the wafer.

**Check in the worker log:** a `Pass k/5: ... on-wafer N cycles = X ms (r of
host-through-barrier ...)` line for every pass, every pass `PASSED`, no `RECORDED
AS 0` and no `EXCEEDS` warning. Expected: `wafer_host_ratio_max` ≤ 1, and
`wafer_time_std` small relative to the mean. `wafer_time_avg` may exceed
`compute_time_hw_avg`; that is the under-measurement above, not an error. Four extra passes add only milliseconds of compute
plus a y and sync_pack readback each, so job times should stay at the usual
history: 16 PEs 22–24 s, 65,536 PEs 145–153 s. A job running well beyond ~2×
that is a hang. Diagnose it in the simulator at 16 PEs, not on the wafer.
