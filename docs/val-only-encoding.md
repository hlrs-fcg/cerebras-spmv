# Val-only stream encoding (`--encoding valtask`)

> **v2.0 status.** `valtask` is the current streaming encoding (the default; ADR-003). The
> bulk `val` kernel described in the first half of this note (`pe_val.csl`, one armed `@fmovs`
> per pass) was **removed**: it hung on hardware whenever streams overlapped and a PE got more
> words than the fabric buffers, and `valtask` (a per-word data task) replaced it. The text
> below is kept as the design history; its `--encoding val` commands no longer exist. The
> ENCODING compile param id 1 stays retired so earlier artifacts keep their meaning.

**Original date:** 2026-10-06. Everything about the bulk kernel below was reviewed against SDK
examples, not built, when it was written.

## Why

Commit 4040625: streaming H2D runs at ~5.9 GB/s and does not respond to channels
(8 → 16: +1.7%, while copy mode gains +97.8%). So widening the link buys nothing,
and removing words is the lever. The header stream at the A/B point
(114,000² @ d=1e-3, 256×256, KX = KY = 446, 198 NNZ/PE avg, 259 max) is:

| | words | MB as sent (full mode) | over NNZ |
|---|---|---|---|
| NNZ | 12,996,000 | — | — |
| header, unpadded | 23,547,662 | — | +81% |
| header, full mode | 29,884,416 | **119.5** (matches the hardware log) | +130% |
| **val-only, full mode** | 16,973,824 | **67.9** | +31% (padding only) |
| val-only, column mode | 15,700,480 | 62.8 | +21% |

The brief's "+116%" is neither figure. Unpadded it is +81%: 10.5M non-empty row
slices for 13.0M NNZ. As sent in full mode it is +130%. Val-only full mode sends
**1.76× fewer bytes**. All of its remaining overhead is padding up to the busiest
PE (259 NNZ), and the kernel never reads that padding.

## Design

**No new per-NNZ preload is needed.** The header kernel already keeps
`col_indices[TOTAL_NNZ]` and the CSR `row_ptrs[KY+1]` resident from the copy-mode
setup phase. `row_ptrs` already gives every row's start and end, so a per-NNZ
row-id array would duplicate it. The only new preload is **one word per PE**:
`val_meta[0]`, the number of words that PE receives per pass. ADR-003 rejected
val-only because of a "40 MB row_offsets overhead". That overhead was already
being paid by `row_ptrs` before this change, so the objection no longer applies.

**Kernel decision: no per-word task.** CSL DSDs are affine (`base + stride·i`).
None of them can do the indexed accumulate `y[row[k]] += v[k]·x[col[k]]` in one
op. Cerebras' own `benchmarks/spmv-hypersparse` does that step as a scalar loop
too. The kernel therefore splits the work in two (`src/csl/pe_val.csl`):

1. `spmv()` arms **one async `@fmovs(vals_dsd, fabin_dsd)`** on the H2D input
   queue, with its length set from `val_meta[0]` via `@set_dsd_length`. The SDK
   uses the same pattern: `tutorials/pipeline-01-basic` runs an async DSD op on a
   `fabin_dsd` over `MEMCPYH2D_1` / input queue 2 on WSE-3, and the WSE-3 stencil
   lib calls `@set_dsd_length` on fabin DSDs. The queue is *not* bound to a data
   task, because on WSE-3 a queue feeds either a task or a microthread.
2. When it completes, it activates `ew_start`. That task runs the local CSR loop
   over `vals` / `col_indices` / `row_ptrs` / `x_vec`, writes all KY rows (so no
   clearing pass is needed), and then starts the EW reduction, unchanged.

**Cycles per word (estimates; nothing here is measured yet):**

| | stream path, per word | off the stream path |
|---|---|---|
| header `stream_word`, value word | task dispatch + 3 flag tests + `col_indices` load + gathered `x` load + `y` load/FMAC/store + 2 counter updates: ~20–30 cycles | — |
| header `stream_word`, header word | dispatch + decode + `row_ptrs` load (+81% more of these words) | — |
| val `@fmovs` fabin → mem | ~1 cycle (DSD engine, no task, no branch) | CSR loop: ~446 row iterations + ≤259 NNZ × ~10 cycles ≈ 5k cycles ≈ 6 µs per PE, all PEs in parallel |

**A caveat on the premise, from the recorded data.** The header runs' per-PE
tsc spans (first word → `chain_done`) at full/row/8 are 8.07M–16.1M cycles for
456 words per PE. That is **≥17,700 cycles between words at every PE**, so each
PE's compute engine is idle more than 99% of the transfer. "The per-word task
is the limit" therefore cannot mean that a PE is saturated by its own words.
If the task is the limit, the cause must be indirect, e.g. a slow-draining
input queue backpressuring the shared H2D color for PEs further along the row
(head-of-line blocking). Otherwise the limit is per-word somewhere else in the
streaming path. Val-only helps in both cases, by different amounts, and the A/B
can tell them apart (see "What the A/B tells us").

**Termination is by count, not sentinel.** A values-only stream has no reserved
word, since `0x00000000` is `+0.0f`. Each PE consumes exactly `val_meta[0]`
words: its own values plus any padding the rectangle memcpy delivers. The
padding lands in `vals[]` past `row_ptrs[KY]` and is never read. This makes the
host–device contract **exact**. If a PE is sent fewer words than it expects,
its `@fmovs` never completes and the run hangs. If it is sent more, the extra
words strand in the queue (a hang, or garbage on the next pass). Both numbers
come from one place, `coo_tiling.plan_val_transfers()`. `val_meta` is computed
by summing the very rectangles the worker sends, so they agree by construction.

A PE with zero words never arms a DSD and activates `ew_start` directly. This
is the val-only counterpart of the old "words exactly fill the buffer, so no
sentinel arrives" deadlock. A column whose PEs all have zero NNZ is left out of
the plan, so no zero-length memcpy is ever issued.

**Modelled over multiple passes.** `tests/test_val_encoding.py` keeps the
device state across passes, as the hardware does. Each PE's input queue
persists, so unconsumed words carry into the next pass, and `vals`/`y_result`
start as garbage. Four passes run through one setup and one plan: a repeat (bit
identical), new values with the same sparsity, and a new `x`. They run on the
three geometries that broke the header kernel: 64×64 d=0.5 on 1×1 (the
recorded 2,048-NNZ crash config), 8×8 dense on 2×2 (every PE exactly full) and
512×512 d=0.4 on 16×16, each in column and full mode. A transfer sent twice
surfaces as stranded words rather than being silently absorbed.

**Header kernel untouched.** The val path lives in its own `pe_val.csl`, and
`layout.csl` picks the tile code from the `ENCODING` compile param. That keeps
`pe.csl` byte-identical to the hardware-validated kernel. `val_meta` is
exported only when `ENCODING == 1`.

**SRAM.** `pe_val.csl` adds `vals[TOTAL_NNZ]` f32 + `val_meta[1]`, i.e.
`4·(TOTAL_NNZ + 1)` bytes. That is 1.26 KB at the A/B point (TOTAL_NNZ = 314).
`SpMVConfig`'s guard counts this term under `--encoding val` and adds nothing
under `header`. At the ceilings this is exactly the "A resident" column in
`archive/docs/streaming-rewrite-status.md` §"Ceilings at full wafer". Val-only costs
0.2–2.3% of the max N at PDE sparsity (7–100 NNZ/row) and ~16% at 1000
NNZ/row. The DSD length is u16, so the plan refuses any PE that would receive
more than 65,535 words per pass. SRAM runs out long before that.

**What it costs in the tsc columns.** No per-word task means nothing can stamp
the first word, so `pe_val.csl` stamps `tsc_start` when `spmv()` arms the DSD.
Under `val`, `compute_hw_cycles_*` include the wait for the stream to arrive,
so they are **not comparable to header rows**. The `encoding` CSV column tells
the two apart.

## Flag usage

`--encoding {header,val}` (default `header`) works on `run.py run` and
`run.py sweep`, and as `encoding = "val"` in a TOML `[sweep]` or `[[run]]`.
It is **compile-time**: it reaches cslc as `ENCODING:0|1`, so the two kernels
never share an artifact. Since `ENCODING:0` is now part of every option string,
header artifacts compiled before this branch are not reused. The flag is
threaded through `SpMVConfig.encoding` → `sdk/compiler.py` params →
`sdk/launcher.py --encoding` → the worker. The worker refuses to start if its
`--encoding` disagrees with `out.json`'s `ENCODING`, because a mismatch would
hang rather than fail. The CSV gains an `encoding` column; older rows read
back as `header`. The sweep config puts the key in a new `KERNEL_KEYS` group:
compile-time like `IO_KEYS`, but string-valued and with no fabric-geometry
warning.

## The bulk kernel's A/B (removed)

The val-vs-header A/B (`bench/configs/val-encoding-ab.toml`, 2026-10-07) is in the git history. Its
only lasting result: the bulk kernel hung under concurrent streams, so `valtask` carries the
stream. The hypotheses it tested are in the commit that added it.

## `valtask`: the val stream through a per-word data task (`--encoding valtask`)

> **Update 2026-10-07:** valtask is *not* overlap-tolerant in column mode; it
> hung there and is now serialised like val (below, and Risk 7).

**Why.** The A/B's val/8 point **hung** at 65,536 PEs in full mode with
channels 8: 5 min 47 s DURATION, against ~2.5 min for header/8. With
channels > 1 the SDK splits the one full rectangle across I/O channels, so the
streams into the PEs overlap. That is the same condition as the column-mode
hang (Risk 7): overlapping streams plus more words per PE than the fabric
buffers. The header kernel's per-word data task tolerates it, and
`pe_val.csl`'s single armed async `@fmovs`, which waits for an exact count,
does not.

**What.** `src/csl/pe_valtask.csl` (`ENCODING == 2`) keeps everything about
val except how the stream is consumed:

- **Same as val:** the 4 B/NNZ stream, `plan_val_transfers`, `val_meta`,
  `vals[TOTAL_NNZ]` and the SRAM cost.
- **Consumption:** a data task bound to the H2D queue (pe.csl's WSE-3
  `@get_data_task_id(h2d_iq)` pattern) does `vals[n] = word; n += 1`. At
  `n == val_meta[0]` it activates `ew_start`.
- **Unchanged after the stream:** `local_spmv`, `ew_start`, `chain_done` and
  `wait_ew` are verbatim `pe_val.csl`, and a test pins that.
- **Words past the count** are swallowed (`draining`, as in pe.csl), so stray
  padding can never back up the queue. The plan sends exactly the count, so in
  practice nothing is swallowed.
- **`val_meta[0] == 0`** goes straight to `ew_start`, as in pe_val.
- **Per-pass reset:** `spmv()` resets the counters before `@unblock`-ing the
  data task, which starts blocked.
- **tsc:** `tsc_start` is stamped on the first word, as in pe.csl, so valtask's
  cycle columns are comparable to header rows (not to val's).
- **Serialisation:** `column_rects_serialized` auto-serialises column mode for
  `val` **and** `valtask`. This kernel was built on the theory that a per-word
  data task tolerates overlapping streams like header; hardware refuted that
  for column mode (see Risk 7: 16 PEs, 4x4, d=0.01, cancelled at 2m54s with
  overlapping rects). Header alone stays overlapping unless
  `--serialize-column-rects` is passed. The comment block at the top of
  `pe_valtask.csl` still says no serialisation is needed; it was left
  untouched so the kernel source, and with it the artifact hash, does not move.

`pe.csl` and `pe_val.csl` are byte-identical to c0f3444, and a test checks
that against git. `layout.csl` picks the kernel by `ENCODING` and exports
`val_meta` for 1 or 2.

**What it costs.** Header-style dispatch, one task per word, but with fewer
words (val's 1.76× fewer than header in full mode at the A/B point) and a
cheaper body: one store and a compare, no decode, no gather, no FMAC. So it
tests whether "fewer words through the same per-word task" is enough by itself.

**Modelled** in `tests/test_valtask_encoding.py` by a stateful device:
garbage initial `vals`/`y_result`, persistent queues, words arriving in random
chunks, and words that beat `spmv()` to the PE waiting in the blocked queue.
It covers round trips in both modes and orders, the `val_meta == 0` /
empty-column rung, full-mode padding, stray words, under-delivery (a hang),
bitwise agreement with val, and five passes on the three geometries that broke
the header kernel.

**Risk, multi-pass only.** Once a pass reaches its count, the data task stays
unblocked and draining. If a later pass's first words reached a PE before
that pass's `launch("spmv")` did, they would be swallowed and the PE would
hang. pe.csl's `draining` has the same property. Today's worker runs one
pass per run, where the task starts blocked, so this cannot happen. A
multi-pass loop should watch for it.

### Commands for bench (valtask)

Smoke at 16 PEs, full mode, channels 1. Expect 5.32e-08:

```bash
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding valtask --h2d-mode full --channels 1
```

Then the A/B point at 65,536 PEs, full mode, row order, at channels 8 (where
val hung) and 16. Expect a relative error near the header's 1.49e-07:

```bash
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --h2d-mode full --memcpy-order row --channels 8 --encoding valtask
uv run python run.py run --rows 114000 --cols 114000 --density 0.001 --num-pes 65536 --h2d-mode full --memcpy-order row --channels 16 --encoding valtask
```

The column geometry that hung val, and hung valtask when its rects overlapped.
valtask is now serialised automatically, so this should pass; add
`--h2d-mode column` explicitly if the default ever changes:

```bash
uv run python run.py run --rows 1000 --cols 1000 --density 0.01 --num-pes 16 --encoding valtask
```

## Risks — only hardware can settle these

1. **It may not compile.** These constructs are new to this repo, and none has
   been through cslc: an unbound H2D input queue driving a microthreaded
   `@fmovs`, `@set_dsd_length` on a fabin DSD, `@as(u16, i32)`, a comptime `if`
   around `@set_tile_code` and `@export_name`, and a `const pe_params = .{...}`
   struct reused across both `@set_tile_code` branches. The smoke rung catches
   all of them cheaply. One precedent to know: the WSE-3 stencil lib drives its
   fabin DSDs through DSRs (`@load_to_dsr`) and keeps a plain async `@mov32`
   commented out. Our EW `@fadds` on a fabin DSD already works on hardware, and
   `pipeline-01-basic` uses the plain form, but the DSR route is the fallback.
2. **A count mismatch hangs instead of failing.** Host and device agree by
   construction and the model tests check it, but the model cannot show how the
   memcpy runtime actually splits a rectangle. If a full-mode transfer delivered
   anything other than exactly `n_words` per PE, the run would hang at
   `wait_ew`. The edge rung is the cheapest place to find out.
3. **Microthread or queue clash.** The stream `@fmovs` (queue 2) and the EW ops
   (queue 6) never overlap in time, but nothing here assigns `.ut_id`
   explicitly. If cslc rejects the default, give the stream op a distinct ID.
4. **Bitwise agreement with header is modelled, not guaranteed.** In the numpy
   model both kernels sum each row in the same order and agree bit for bit. On
   hardware the codegen may contract `acc += v*x` and `y[r] += v*x` differently
   (FMA versus multiply then add), which would show up as noise around 1e-7.
5. **The tsc spans change meaning** under `val` (see above). Compare kernels
   with `compute_time_hw_avg`, not with the cycle columns.
6. **Not streaming in ADR-001's sense.** Values now land in SRAM, the same way
   `col_indices` already did. This changes nothing about what fits (the guard
   accounts for it), but a future multi-pass or column-striped design would
   have to size `vals[]` per pass.
7. **Column mode: overlapping column transfers hang on hardware, so val
   serialises them.** *Resolved 2026-10-06.*

   **Hardware evidence (4×4 unless noted).**

   | words/PE | column rects | result |
   |---|---|---|
   | ~650–674 (1000² d=0.01) | overlapping (all `nonblock=True`) | **HANG**, twice: 112 s and 103 s DURATION (a passing job takes 22 s) |
   | ~650–674 (1000² d=0.01) | serialised | PASS, 5.32e-08, 22 s, **same artifact hash** as the hangs |
   | ≤ 76 (1000² d=0.001) | overlapping | PASS, 4.14e-08 |
   | ≤ 2 (64 PEs) | overlapping | PASS |

   Full mode (one rectangle) passes. **The simulator passes either way**
   (5.32e-08), so it cannot reproduce the hang. An earlier "hang" at ~65
   words/PE was a false positive: that run was cancelled on age, not stuck. A
   col-order run was inconclusive and is left out.

   **Cause.** The hang needs **both** overlapping nonblocking column rects
   **and** more words per PE than the fabric can buffer. That threshold lies
   somewhere between 76 and ~650 words/PE and has not been located more
   precisely. In column mode a pass is W back-to-back 1-wide streaming
   `memcpy_h2d` calls. `--serialize-column-rects` sends every one but the last
   with `nonblock=False`, so only one is in flight at a time, and the last
   stays nonblocking so `wait_ew` is still queued behind it. Serialised, the
   hanging configuration passes on the same artifact. Host-side serialisation
   fixes it at any word count, and the kernel is unchanged. Why the header
   kernel tolerates the overlap is not known. One plausible difference is
   that its per-word data task drains the input queue continuously, while
   val's single `@fmovs` waits for an exact count, but that has not been
   tested.

   **Now.** `cluster/coo_tiling.py: column_rects_serialized` turns
   serialisation on for val in column mode, **unconditionally**: it does not
   look at word counts, because the threshold is unknown and serialising
   costs nothing in correctness. The worker logs it once. The earlier
   refusal guard is gone. The CSV column
   `serialize_column_rects` records what ran, so it is True on every val
   column-mode row. The header kernel's column path is unchanged (overlapping)
   unless `--serialize-column-rects` is passed explicitly. That opt-in exists to
   test whether header column mode also gains from it.

   **valtask hangs on overlap too (hardware, 2026-10-07).** `--encoding
   valtask`, column mode, 16 PEs (4x4), d=0.01, overlapping rects: still
   running at 2m54s (a passing job takes ~22 s), cancelled. So the per-word
   data task does not rescue overlap; the "header tolerates it" observation
   is unexplained, not a property of per-word consumption. `column_rects_serialized`
   therefore covers both `VAL_ENCODINGS`. The serialised valtask run has not
   been measured yet. Full mode at channels > 1 is not touched by this rule:
   valtask/8 at 65K PEs passed (1.49e-07) without it.

   **Full mode stays the recommended path.** Serialised column mode works, but
   its per-PE tsc span is about 4× full mode's at the 16-PE rung: 160.8K vs
   39.3K cycles. The "val-only, column mode"
   row in the table under "Why" is the byte count; its time is not
   competitive.
