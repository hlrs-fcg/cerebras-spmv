# Host timing lower bounds and the completion barrier

> **Canonical copy:** this document now lives in the `cs3-microbench` repo (`docs/host-timing-barrier.md`); this copy is kept because the ADRs cite it and is not updated further.

**Date:** 2026-10-07 · **Branch:** `fix/completion-barrier` · host-only (no
kernel change, so artifact hashes do not move).

## The problem

Every streaming time recorded since 10-01 (`memcpy_h2d_time`,
`compute_time_hw_*`) ends when `runner.launch("wait_ew", nonblock=False)`
returns. That call can return **before the device has finished the pass**, and
the next blocking call, the 0.46 MB y D2H, then absorbs the rest:

| row | y D2H |
|---|---|
| column / row / 8 (10-01) | 1.0 ms |
| full / row / 8 | 50 ms |
| full / row / 16 | 27 ms |
| channels 1 | 193 ms / 371 ms |

A 0.46 MB copy that takes 1 ms on an idle device cannot take 50–371 ms. The
extra time is the pass still running. The per-PE tsc spans agree: val full/ch1
read 104 ms on the device against 22 ms of host compute+barrier, and the span
always fits inside compute+barrier+D2H. **`memcpy_h2d_time` and
`compute_time_hw_*` are lower bounds, and every rate derived from them is an
upper bound.**

## The fix

Right after `wait_ew`, the worker issues a **completion barrier**: a blocking
copy-mode D2H of one word, `y_result[0]` from each column-0 PE (rectangle
(0,0), width 1, height H).

- Column 0 ends every row's EW chain, so a column-0 PE finishes only after its
  whole row has.
- Its command stream is held by `wait_ew` until that chain is done, and the
  barrier D2H is queued behind it. The y D2H soaking up the time is direct
  evidence that a copy-mode D2H queued there waits.
- So the barrier cannot complete before every PE has finished the pass.

It is timed (`barrier_d2h_time`), and then **the same D2H is issued again** on
the now-idle device (`barrier_d2h_repeat_time`) to price the barrier itself:

```
device_done_time      = start of the H2D bracket -> barrier returned
stream_time_corrected = device_done_time - barrier_d2h_repeat_time
```

`memcpy_h2d_time` is kept unchanged so old and new rows stay comparable, and
the logs now call it the **host-return** time.

**Why this signal.** csl-examples never relies on host return times. Its
benchmarks (bandwidth-test, gemv) bracket the work on the device with tsc
stamps and read them back afterwards; synced timing (`docs/synced-timing.md`) brings that here. Without a
kernel change, a D2H that must wait for the last chain is the strongest host
signal available, and reading one word from H PEs keeps its own cost small and
measurable.

## The setup phase is doubtful too, so it gets the same pattern

The copy-mode setup transfers (`x_vec`, `col_indices`, [`val_meta`],
`row_ptrs`) end in a blocking `memcpy_h2d`. For an H2D, "blocking" can only
promise that the host is done with its buffer. Nothing in the SDK semantics we
rely on says that every PE has written the data. The `wait_ew` evidence shows
that a blocking call can return before the device is done, so
`setup_h2d_time` is treated as doubtful too. After the last setup H2D the
worker reads **one word of `row_ptrs` (the last array sent) from every PE**:
each PE answers only after the writes queued before it, so the read cannot
complete until all of setup has landed. That read is repeated to price it:

```
setup_done_time          = setup start -> setup barrier returned
setup_h2d_time_corrected = setup_done_time - setup_barrier_d2h_repeat_time
```

The copy-mode link rate (the retired `analyze_h2d.py`'s "copy" column) used the
corrected time. If it turns out that the setup H2D was already complete on
return, the barrier is cheap and the two columns agree. That agreement would
itself be the evidence.


**Setup in strips (2026-10-08).** The setup copy is now sent as `--setup-strips` vertical strips of
the grid per array (default 2 x channels, 1 = the old single rectangle), issued one after another:
all but the last strip of an array block, the final strip of the last array (`row_ptrs`) blocks, and
the barrier above follows it unchanged, so `setup_h2d_time_corrected` still brackets the whole of
setup. The strip count is a CSV column (`setup_strips`; 1 for rows written before it existed).
Setup rates at different strip counts are therefore different transfers: compare at equal
`setup_strips`.

## CSV

| column | meaning |
|---|---|
| `memcpy_h2d_time`, `compute_time_hw_*`, `setup_h2d_time` | host-return times, **lower bounds** (unchanged) |
| `device_done_time` | H2D bracket start → pass barrier returned |
| `barrier_d2h_time` / `barrier_d2h_repeat_time` | the barrier, then the same D2H on an idle device |
| `stream_time_corrected` | `device_done_time − barrier_d2h_repeat_time`: **use for streaming rates** |
| `setup_done_time`, `setup_barrier_d2h_time`, `setup_barrier_d2h_repeat_time` | the same for setup |
| `setup_h2d_time_corrected` | **use for copy-mode rates** |

All are 0 on rows written before the barrier existed.

The analysis notebooks use the corrected columns where present. The retired
`analyze_h2d.py` (`archive/bench/`) marked every row timed by host return as `LB` in a
`basis` column. **Conclusions drawn from LB rows alone need re-measuring.**
That includes 4040625's "streaming is consumption-bound" (8 → 16 channels
gave streaming +1.7%), which rested on host-return times whose hidden
remainder changed with the channel count (50 ms vs 27 ms of D2H).

## Commands for bench

Re-measure the points the conclusions rest on. Each recompiles, because the
main tree moved since their artifacts were built:

```bash
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --h2d-mode full
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --h2d-mode full --memcpy-order row --channels 8
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --h2d-mode full --memcpy-order row --channels 16
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --h2d-mode full --memcpy-order row --channels 1
```

**Check in the worker log:** a `Stream+compute: host-return X ms
(memcpy_h2d_time), device-done Y ms (barrier B ms, idle repeat R ms) ->
stream_time_corrected Z ms` line. R should be small, comparable to the 1.0 ms
idle y D2H or less, since it reads H words rather than H·KY. Where B ≫ R, the
host-return time was hiding device time. The setup line has the same shape.
