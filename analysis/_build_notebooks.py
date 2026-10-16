"""Generate the SpMV barrier re-measure notebook's source (unexecuted). Run, then execute with nbconvert.

(The microbenchmark notebooks M1-M10 moved to the cs3-microbench repo with their generators.)

    uv run --no-project --python 3.11 --with nbformat python analysis/_build_notebooks.py

The notebooks are generated so the shared structure (Hypothesis / Result / What that means /
What the vendor says) stays identical. Edit the text here, not in the .ipynb, then re-execute.
"""
from pathlib import Path
import sys

import nbformat as nbf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

PRE = '''import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import lib
lib.setup_plots()
pd.set_option("display.width", 200, "display.max_columns", 30)'''

VENDOR_SRC = """Sources: **[SDK]** https://sdk.cerebras.ai/tensor-streaming ·
**[BW]** `csl-examples/benchmarks/bandwidth-test/run.appliance.py` (csl-examples
targets SDK 1.4.0; the cluster runs 2.10.0) · **[CLI]** `--channels` / `--width-*-buf` help,
cerebras-sdk 2.10.0 · **[VCM]** `docs/vendor-claims-vs-measurements.md`.
[SDK]:N cites a line of `analysis/vendor_sources/2026-10-08_tensor-streaming.md`, a snapshot of
https://sdk.cerebras.ai/tensor-streaming.md taken 2026-10-08; every [SDK] quote was checked against it with `grep -F`."""


def notebook(title, intro, sections, extra_cells=()):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(PRE)]
    cells += list(extra_cells)
    for i, s in enumerate(sections, 1):
        cells.append(nbf.v4.new_markdown_cell(f"---\n## {i}. {s['q']}"))
        cells.append(nbf.v4.new_markdown_cell("### Hypothesis\n\n" + s["hyp"]))
        cells.append(nbf.v4.new_markdown_cell("### Result"))
        cells.append(nbf.v4.new_code_cell(s["code"]))
        cells.append(nbf.v4.new_markdown_cell("### What that means\n\n" + s["means"]))
        cells.append(nbf.v4.new_markdown_cell("### What the vendor says\n\n" + s["vendor"] + "\n\n" + VENDOR_SRC))
    nb.cells = cells
    return nb


def save(name, nb):
    nbf.write(nb, HERE / name)


# =====================================================================================
# spmv_barrier_remeasure
# =====================================================================================
spmv_sections = [
    dict(
        q="Timing basis: how far below the truth are host-return times?",
        hyp="""**Claim under test** (docs/host-timing-barrier.md): `memcpy_h2d_time` (host return) is a lower
bound on the true streaming time, because `wait_ew(nonblock=False)` returns before the device is done and the
next blocking call absorbs the rest. `stream_time_corrected` (barrier-closed) is the valid time.

**Prediction.** Corrected ≥ host-return on every row. The gap (the *tail*) is largest where the device is
slowest to drain: streaming at 1 channel (the 10-07 log showed a 97 ms host return against 535 ms corrected),
and smallest in column mode, where the barrier re-measure found host-return and corrected nearly equal.
**Falsifier:** corrected < host-return on any row would mean the barrier is not a valid completion signal.""",
        code='''raw = lib.load_spmv()
df, ex = lib.valid_spmv(raw)
print("timing basis: stream_time_corrected (barrier-closed) vs memcpy_h2d_time (host-return, lower bound)")
print(f"{len(raw)} rows loaded from {raw['source'].nunique() if len(raw) else 0} file(s) |", ex, f"| usable: {len(df)}")
if df.empty:
    lib.no_data("2026-10-07_barrier-remeasure.csv")
else:
    df["tail_ms"] = (df.stream_time_corrected - df.memcpy_h2d_time) * 1e3
    df["tail_frac"] = df.tail_ms / (df.stream_time_corrected * 1e3)
    print("rows where corrected < host-return:", int((df.tail_ms < -0.05).sum()), "(tolerance 0.05 ms)")
    t = df.groupby(["encoding", "h2d_mode", "channels"]).agg(
        n=("tail_ms", "size"), host_ms_med=("memcpy_h2d_time", lambda s: s.median() * 1e3),
        corr_ms_med=("stream_time_corrected", lambda s: s.median() * 1e3),
        tail_ms_min=("tail_ms", "min"), tail_ms_med=("tail_ms", "median"), tail_ms_max=("tail_ms", "max"),
        tail_frac_med=("tail_frac", "median")).round(2)
    display(t)
    fig, axs = lib.panels(2, sharey=True, sharex=True)
    for ax, hm in zip(axs, ["full", "column"]):
        for enc, g in df[df.h2d_mode == hm].groupby("encoding"):
            st = lib.style("encoding", enc)
            ax.scatter(g.memcpy_h2d_time * 1e3, g.stream_time_corrected * 1e3, s=40, color=st["color"], marker=st["marker"],
                       alpha=0.9, edgecolors="white", linewidths=0.6, label=enc, zorder=3)
        ax.plot([5, 700], [5, 700], color=lib.INK, lw=1.2, ls=(0, (1, 3)), label="corrected = host-return", zorder=2)
        lib.finish(ax, "host-return time, a lower bound (ms)", "barrier-closed time (ms)", f"{hm} mode, 65,536 PEs",
                   legend="encoding", logx=True, logy=True, xticks=[10, 30, 100, 300], yticks=[10, 30, 100, 300, 1000])
    plt.show()''',
        means="""- The barrier-closed time is never below the host-return time (0 of 28 rows), so the check that could have broken the
  barrier's premise did not fire.
- In **full** mode the host returns long before the device is done: the tail is 82% of the time for header at 1 channel (438 ms of 535), 94% for
  valtask/val at 1 channel, 41% / 33% at 8 and ~13% at 16 channels. In **column** mode at 8 and 16 channels the two agree to ≤0.1 ms: nothing is
  left buffered when the call returns. At 1 channel column mode still has a tail (65 ms of 107, n=1).
- **Not established:** that the barrier is *exact*. It bounds the true time from above by at most its own
  cost (0.2 ms idle); an independent on-wafer tsc check is still held.""",
        vendor="""**Agrees.** [BW] lines 57–58: "To record the elapsed time on host may not show the true timing because the runtime
may not start the transaction when the user calls a H2D/D2H command." The vendor's own benchmark therefore times
on the device ("Instead, this bandwidhTest samples the timing on the device.", [BW]:60).

**Agrees.** [SDK]:257 "If blocking mode (`nonblock=False`) is selected, the API waits until the operation is done."
and, for nonblocking mode, "the function returns before the operation even starts" ([VCM] §4 quotes the full
sentence). Our finding is the narrower point that "done" for a *launched function* is not "device finished".""",
    ),
    dict(
        q="Header vs valtask (full mode, row-major): does dropping the header words pay?",
        hyp="""**Reasoning** (docs/val-only-encoding.md): header streams 29.9 M words (119.5 MB) per pass and valtask
16.97 M (67.9 MB), 1.76× fewer; valtask also removes per-word decode work. If the transfer is **link-bound** the
speed-up should track the byte ratio, ≈1.76×. If it is **consumption-bound** (task cost per word) it could be larger.

**Prediction:** valtask faster than header at every channel count, by ≥1.76× at 1 channel.
**Bulk `val` (one `@fmovs`)** was predicted to equal valtask on bytes but removes the per-word task, hence faster;
the concurrency hang seen in hardware is a separate matter.
**Falsifier:** a speed-up well below 1.76× at 1 channel, or valtask slower than header.""",
        code='''if not df.empty:
    f = df[(df.h2d_mode == "full") & (df.memcpy_order == "row") & (df.setup_order == "row")]
    t = lib.summarize(f, ["encoding", "channels"], "stream_time_corrected", 1e3, 1)
    t = t.rename(columns={"min": "min ms", "median": "median ms", "max": "max ms"})
    display(t)
    piv = t.pivot(index="channels", columns="encoding", values="median ms")
    if {"header", "valtask"} <= set(piv.columns):
        piv["header/valtask (median)"] = (piv["header"] / piv["valtask"]).round(2)
    display(piv)
    fig, axs = lib.panels(1)
    lib.strip_plot(axs[0], f.assign(ms=f.stream_time_corrected * 1e3), "channels", "ms", "encoding", "encoding")
    lib.finish(axs[0], "I/O channels (count)", "barrier-closed stream time (ms)", "full / row-major: every run, line = median",
               legend="encoding", logx=True, logy=True, xticks=[1, 8, 16], yticks=[30, 100, 300, 1000])
    plt.show()''',
        means="""- **valtask beats header at every channel count on the median** (2.6× at 1 channel, 1.9× at 8, 1.8× at 16 on medians of 180 vs 101 ms;
  min against min at 16 channels is only 1.1×, 85.9 vs 76.7 ms, and the ranges 86–268 and 77–123 ms overlap). Row-major setup only,
  so the rows are one session's. At 1 channel 2.6× exceeds the 1.76× byte
  ratio, so header was *not* purely link-bound there: per-word work matters too.
- At 16 channels the spread (3× between identical header runs) is as large as the encoding effect: n=3 per side is a hint, not a ranking.
- Bulk `val` is measured at 1 channel only (194 ms, ~5% under valtask) because it hangs when streams overlap.
- **Not established:** a 16-channel ranking. n=3 with 3× noise is not enough; that is what S2 (n≥3, alternating) is for.""",
        vendor="""**Silent.** No vendor text compares stream encodings. Their closest code is a scalar loop for the indexed
accumulate, which docs/val-only-encoding.md ("Kernel decision") says `benchmarks/spmv-hypersparse` also does as a scalar loop
(not re-checked here; the kernel is under `benchmarks/spmv-hypersparse/src/hypersparse_spmv/pe.csl`). [BW]:44 on aggregation: "runtime can aggregate multiple nonblocking H2D/D2H commands implicitly."
Nothing in the vendor material says fewer words per nonzero is faster; our 2.6× is our own result.""",
    ),
    dict(
        q="Do more I/O channels speed up streaming?",
        hyp="""**Prediction:** yes. Copy-mode setup scaled 1→16 channels (0.33 → 5.2 GB/s row-major) and nothing in streaming
should cap that unless the PE-side task is the bottleneck. The 2026-10-03 conclusion ("consumption-bound,
channel-flat") rested on host-return times and was predicted to disappear under the barrier.
**Falsifier:** corrected times flat between 8 and 16 channels with spread smaller than the effect.""",
        code='''if not df.empty:
    f = df[(df.h2d_mode == "full") & (df.memcpy_order == "row") & (df.setup_order == "row") & df.encoding.isin(["header", "valtask"])]
    t = lib.summarize(f, ["encoding", "channels"], "stream_time_corrected", 1e3, 1)
    base = t[t.channels == 1].set_index("encoding")["median"]
    t["speed-up vs 1 ch (median)"] = [round(base[e] / m, 2) for e, m in zip(t.encoding, t["median"])]
    t["GB/s (median, 1 pass)"] = [round(f[(f.encoding == e) & (f.channels == c)].h2d_bytes_total.iloc[0] / (m * 1e-3) / 1e9, 2)
                                  for e, c, m in zip(t.encoding, t.channels, t["median"])]
    display(t.rename(columns={"min": "min ms", "median": "median ms", "max": "max ms"}))''',
        means="""- Streaming scales with channels (row-major setup, full mode): header 535 → 152 → 180 ms (16 ch: n=3, min 86), valtask 205 → 81 → 101 ms
  (medians; n=1 at 1 and 8 channels) for 1 / 8 / 16 channels. The 1→16 speed-up is 6.2× on the best header run (3.0× on the median) and 2.0× on the valtask median.
- 8 → 16 channels is not reliably faster for valtask (81 ms vs a 77–123 ms range): the flat-looking top end is
  noise-dominated, not proven flat.
- The "channel-flat" reading of 10-03 is withdrawn (`docs/results.md`). **Not established:** where scaling stops.
  Streaming is at 0.3–1.6 GB/s here against 5–7 GB/s in copy mode, so some other limit remains (see M2).""",
        vendor="""**Agrees with direction.** [BW]:25: "The more channels, the higher bandwidth". [BW]:18–19: "The maximum bandwidth of a
single channel is around 7Gbps (Giga bit per second)." (= 0.875 GB/s). This is a comment in the vendor's example code (SDK 1.4.0 examples), not SDK documentation.

**Contradicts in magnitude.** Our verified 1-channel copy rate is 0.33 GB/s (2.6 Gbps), and streaming 0.22–0.33 GB/s,
below a third of the quoted per-channel maximum. The vendor figure may refer to different hardware or SDK (the
file targets SDK 1.4.0); we cannot tell which.

**Unsourced claim withdrawn.** "Performance improvements are minimal past 8 channels" is attributed to the docs in our
commits `24d995a` and `4040625`; no such sentence exists in [SDK] or [CLI] ([VCM] §3).""",
    ),
    dict(
        q="Full vs column H2D mode (header kernel) and the effect of setup order",
        hyp="""**Reasoning.** Column mode sends one 1-wide rectangle per grid column with exact per-column lengths (112.1 MB vs
119.5 MB in full mode, −6%). docs/streaming-rewrite-status.md argued a 1-wide rectangle cannot use more than one
channel, which predicts column mode **does not scale with channels** and loses to full mode at high channel counts.
Setup order col-major was predicted to help copy-mode setup (vendor claim).
**Falsifier:** column mode faster than full at 16 channels despite 1-wide rectangles.""",
        code='''if not df.empty:
    f = df[(df.encoding == "header") & (df.memcpy_order == "row")]
    t = lib.summarize(f, ["h2d_mode", "channels", "setup_order", "serialize_column_rects"], "stream_time_corrected", 1e3, 1)
    display(t.rename(columns={"min": "min ms", "median": "median ms", "max": "max ms"}))
    fig, axs = lib.panels(1)
    lib.strip_plot(axs[0], f.assign(ms=f.stream_time_corrected * 1e3), "channels", "ms", "h2d_mode", "h2d_mode")
    lib.finish(axs[0], "I/O channels (count)", "barrier-closed stream time (ms)", "header, row-major stream order",
               legend="H2D mode", logx=True, logy=True, xticks=[1, 8, 16], yticks=[30, 100, 300, 1000])
    plt.show()
    mo = df[(df.encoding == "header") & (df.h2d_mode == "full") & (df.channels == 16)]
    print("header, full, 16 ch by memory order (stream order of the data):")
    display(lib.summarize(mo, ["memcpy_order", "setup_order"], "stream_time_corrected", 1e3, 1))
    v = df[(df.encoding == "valtask") & (df.h2d_mode == "column")]
    if len(v):
        print("valtask in column mode (S1 file, serialized rects):")
        display(lib.summarize(v, ["channels", "setup_order"], "stream_time_corrected", 1e3, 1))''',
        means="""- **The prediction is false.** Column mode is faster than full at every channel count (header, medians: 107 vs 535 ms at 1 ch,
  59 vs 152 ms at 8 (n=3 vs 1), 62–66 vs 86–268 ms at 16) and far steadier at 16 channels (61.4–69.4 ms over 6 column runs, setup order and
  serialization mixed; full mode varies 3×).
  Column mode also carries only 6% fewer bytes, so the gain is not payload. Serializing the rectangles (n=1, 69.4 ms) did not help header at 16 channels.
- Column mode does *not* improve from 8 to 16 channels (59 → 62–66 ms), consistent with 1-wide rectangles not spreading
  over channels, but at 8–16 it is already faster than every full-mode run.
- Valtask in column mode (from the S1 file, rects serialized) is the fastest streaming seen: about 40–49 ms.
  Those rows come from the later S1 session; compare to the first session's rows with the drift caveat (`docs/results.md`, "Timing basis").
- Memory order for streaming (header, full, 16 ch): the one col-major run (245 ms) lies inside the row-major range (86 / 180 / 268 ms, n=3): col-major is not better, and one run cannot say more.
- **Not established:** why. M4 (strips k = 1 … W, serial vs all) isolates concurrency/granularity from layout.""",
        vendor="""**Silent on column mode for streaming.** [SDK]:313: "Column-major delivers better bandwidth than row-major."
It concerns the memory order of the data buffer, not the rectangle shape, and the vendor benchmark measures copy mode only
(`streaming=False` throughout, e.g. [BW]:344, 360, 384, 404). Our column H2D *mode* (many narrow rectangles) has no vendor counterpart.
[BW]:44: "runtime can aggregate multiple nonblocking H2D/D2H commands implicitly." is the nearest statement; it
suggests many nonblocking calls need not cost more than one.""",
    ),
    dict(
        q="Setup copy: row-major vs col-major, by channels",
        hyp="""**Prediction (vendor):** col-major setup is faster than row-major in copy mode. We measured +43% on 10-06 using
host-return times, which overstate rates; the barrier-closed gain is predicted to be smaller.
**Falsifier:** no difference, or col-major slower, on `setup_h2d_time_corrected`.""",
        code='''if not df.empty:
    f = df.copy()
    f["setup_gbps"] = f.setup_h2d_bytes / f.setup_h2d_time_corrected / 1e9
    t = lib.summarize(f, ["channels", "setup_order", "source"], "setup_gbps", 1.0, 2)
    t = t.rename(columns={"min": "min GB/s", "median": "median GB/s", "max": "max GB/s"})
    print("timing basis: setup_h2d_time_corrected (barrier-closed); 302 MB copy-mode setup")
    display(t)
    f["stream_gbps"] = f.h2d_bytes_total / f.stream_time_corrected / 1e9
    fig, axs = lib.panels(3, sharey=True, sharex=True)
    lib.strip_plot(axs[0], f, "channels", "setup_gbps", "setup_order", "order", labels={"row": "row-major setup", "col": "col-major setup"}, endlabel=False)
    lib.draw_claims_vs_channels(axs[0])
    lib.finish(axs[0], "I/O channels (count)", "setup copy rate (GB/s), line = median", "setup copy, 302 MB, barrier-closed",
               legend="setup order", logx=True, logy=True, xticks=[1, 8, 16], yticks=[0.3, 1, 3, 10])
    for ax, hm in zip(axs[1:], ["full", "column"]):
        lib.strip_plot(ax, f[f.h2d_mode == hm], "channels", "stream_gbps", "encoding", "encoding", endlabel=False)
        lib.draw_claims_vs_channels(ax)
        lib.finish(ax, "I/O channels (count)", "streaming rate, one pass (GB/s), line = median", f"streaming, {hm} mode, barrier-closed",
                   legend="encoding", logx=True, logy=True, xticks=[1, 8, 16], yticks=[0.3, 1, 3, 10])
    lib.place_claims(fig); plt.show()
    display(lib.aggregate_note(max(f.setup_gbps.max(), f.stream_gbps.max())))''',
        means="""- At 16 channels col-major setup runs at a median of 6.66 GB/s (4.85–7.21, n=9) against 5.12 (3.72–5.20, n=8) row-major: **about +30% on medians**, below
  the +43% from host-return times. The ranges overlap and mix sessions (see the `source` split in the table): the session-to-session spread
  of setup rate is itself ~±20%, so +30% is an estimate, not a constant.
- At 8 channels col-major (n=2, 5.36 GB/s) is far above row-major (median 3.06, n=5), a larger gain than at 16; 1 channel has row-major only.
- Setup scales with channels (0.37 → 3.06 → 5.12 GB/s row-major medians), the same shape as streaming.
- **Not established:** the col/row gain at 1 channel, and its size at 8 (n=2).""",
        vendor="""**Agrees.** [SDK]:313: "Column-major delivers better bandwidth than row-major." Our gain is +30% (barrier-verified, 16
channels, 302 MB). Mismatch of detail: [SDK] prose says for column-major "w is the fastest varying dimension" while the vendor
code ([VCM] §2, `hwl_2_oned_colmajor`, `run.appliance.py:178`) makes **h** fastest; hardware validation follows the code.
**Contradicts** the 10.35 GB/s figure from 10-06 only in that it was a host-return rate and is an overestimate.""",
    ),
]

nb = notebook(
    "SpMV barrier re-measure (2026-10-07)",
    """Barrier-verified results for the streaming SpMV: 65,536 PEs (256×256), 114,000² at d=1e-3 (13.0 M non-zeros),
header vs valtask encoding, full vs column H2D mode, 1 / 8 / 16 I/O channels. Data:
`2026-10-07_barrier-remeasure.csv` and `2026-10-07_s1-column-matrix.csv` under `lib.DATA`. Only barrier-closed times
(`stream_time_corrected`, `setup_h2d_time_corrected`) are used for conclusions; host-return times appear only to
show their gap. These CSVs have no `system` column, so a bad system cannot be filtered by name; the one known bad-system run
never produced a row. Compare within a session only (session drift, `docs/results.md`).

Hypotheses were written down per question before the tables were looked at *for this notebook*, but the team already
knew the headline numbers (`docs/results.md`); the predictions are those recorded in `docs/val-only-encoding.md` and
`docs/streaming-rewrite-status.md`, and where they were wrong the text says so.""",
    spmv_sections,
)
save("spmv_barrier_remeasure.ipynb", nb)
print("written")


# =====================================================================================
# spmv_scaling  (setup strips, synced timing, weak scaling; 2026-10-09)
# =====================================================================================
D107_LOAD = nbf.v4.new_code_cell('''F = {"weak": "2026-10-09_weak-scaling.csv", "d56": "2026-10-09_dan56.csv", "strips": "2026-10-09_setup-strips.csv",
     "xsys": "2026-10-09_xsys-check.csv", "fw": "2026-10-08_fullwafer.csv",
     "old": "2026-10-07_barrier-remeasure.csv", "s1": "2026-10-07_s1-column-matrix.csv"}
raw = {k: lib.load_spmv([v]) for k, v in F.items()}

def ok(d):
    """Validated, SUCCESS, barrier-closed stream time present."""
    if d.empty:
        return d
    keep = d.validation_passed.astype(bool) & d.exit_reason.eq("SUCCESS") & (d.stream_time_corrected.fillna(0) > 0)
    return d[keep].copy()

D = {k: ok(v) for k, v in raw.items()}
for k, v in raw.items():
    print(f"{F[k]:38s} {len(v):3d} rows, {len(D[k]):3d} usable")
for d in D.values():
    if not d.empty:
        d["sync"] = d.get("sync_timing", pd.Series(False, index=d.index)).astype(str).eq("True")
        d["strips"] = d.get("setup_strips", pd.Series(1, index=d.index)).fillna(1).astype(int)
        d["stream_ms"] = d.stream_time_corrected * 1e3
        d["setup_ms"] = d.setup_h2d_time_corrected * 1e3
        d["stream_GBps"] = d.h2d_bytes_total / d.stream_time_corrected / 1e9
        d["setup_GBps"] = d.setup_h2d_bytes / d.setup_h2d_time_corrected / 1e9
# The 65K weak-scaling point is the flag-off synced-timing runs: identical config, reuse noted in commit 0cbc1c4.
ws = pd.concat([D["weak"], D["d56"][~D["d56"].sync]], ignore_index=True).sort_values("num_pes")
# Systems are not in these CSVs; from the run reports: system A for every 10-09 run up to 428K run 1,
# system B for 428K runs 2-3, the full wafer and the 65K cross-check.''')

d107_sections = [
    dict(
        q="Setup copy in 32 narrow strips vs one rectangle (65K PEs)",
        hyp="""**Prediction (plan page, 2026-10-09, before the runs):** M1/M4 reached 10.3 GB/s at 16 channels with narrow strips issued
serially, against 6.3–7.2 GB/s for one rectangle in S1. Expected 8–9.5 GB/s for 32 strips (65K has only 4.6 KB/PE, so per-strip
cost should show), hoped ~10. Stream time should not move: only the setup copy changes.
**Falsifier:** 32 strips no faster than 1, or stream time changing by more than run-to-run noise.""",
        code='''s = pd.concat([D["strips"].assign(system="system A"), D["xsys"].assign(system="system B")], ignore_index=True)
if s.empty:
    lib.no_data("2026-10-09_setup-strips.csv")
else:
    display(lib.summarize(s, ["strips", "system"], "setup_ms", 1, 1).rename(columns=lambda c: c + " ms" if c in ("min", "median", "max") else c))
    display(lib.summarize(s, ["strips", "system"], "setup_GBps", 1, 2).rename(columns=lambda c: c + " GB/s" if c in ("min", "median", "max") else c))
    display(lib.summarize(s, ["strips"], "stream_ms", 1, 1).rename(columns=lambda c: c + " stream ms" if c in ("min", "median", "max") else c))
    m = s[s.system == "system A"].groupby("strips").setup_ms.median()
    print(f"speed-up 1 -> 32 strips (median, system A): {m[1] / m[32]:.2f}x")
    fig, axs = lib.panels(1, height=4.6)
    ax = axs[0]
    for i, (k, g) in enumerate(s.groupby("strips")):
        st = lib.style("index", i)
        ax.scatter(g.strips, g.setup_GBps, s=60, color=st["color"], marker=st["marker"], edgecolors="white", linewidths=0.6,
                   label=f"{k} strip{'s' if k > 1 else ''}", zorder=3)
        m76 = g[g.system == "system A"].setup_GBps.median()
        ax.annotate(f"median {m76:.1f} GB/s (system A)", (k, m76), xytext=(10, 0),
                    textcoords="offset points", color=st["color"], va="center")
    lib.draw_claims(ax, 16)
    lib.finish(ax, "setup strips (count)", "setup copy, barrier-closed (GB/s)", "65,536 PEs, 302 MB setup, 16 channels: every run",
               logx=True, xticks=[1, 32])
    ax.set_ylim(0, 16)
    lib.place_claims(fig); plt.show()''',
        means="""*(Written 2026-10-13 from the rows above.)*
- **32 strips set up 1.48× faster** than one rectangle: 27.7 vs 41.0 ms median, 10.9 vs 7.4 GB/s. Every 32-strip run beats every 1-strip run.
  That beats the hope, and is at or above the 10.3 GB/s the microbenchmarks reported.
- The single system B run (26.0 ms) falls inside the system A range, so there's no sign of a system effect.
- Stream time doesn't move (46.5 vs 45.9 ms medians), as predicted.
- **Not established:** why strips help. Likely it's more transfers in flight across the 16 channels. That was not isolated here.""",
        vendor="""**Silent on strips.** The guide's only shape advice is about order: [SDK]:313 "Column-major delivers better bandwidth than
row-major." **Context:** [BW]:18–19 "The maximum bandwidth of a single channel is around 7Gbps", i.e. 0.875 GB/s per channel,
14 GB/s at 16 channels (red line). 32 strips reach 78% of that comment's 16-channel figure.""",
    ),
    dict(
        q="Does an independent on-wafer clock agree with the host barrier?",
        hyp="""**Prediction (plan page, 2026-10-09):** the synced on-wafer window (`wafer_time_*`, max toc − min tic across PEs) is shorter
than the barrier-closed host time, by the launch and barrier costs. The flag-off build's stream time is within ±10% of S1
(43.5–48.9 ms). I also hoped the window would show a compute time of a few ms. That hope was mis-framed (see below).
**Falsifier:** window ≥ barrier time (the reference clock or the barrier is wrong), or a ratio that varies a lot between passes.""",
        code='''d = D["d56"]
if d.empty:
    lib.no_data("2026-10-09_dan56.csv")
else:
    d = d.assign(arm=np.where(d.sync, "sync, loop 5", "flag off"))
    display(lib.summarize(d, ["arm"], "stream_ms", 1, 1).rename(columns=lambda c: c + " barrier ms" if c in ("min", "median", "max") else c))
    on = d[d.sync].assign(wafer_ms=lambda x: x.wafer_time_avg * 1e3, host_ms=lambda x: x.host_pass_time_avg * 1e3)
    display(on[["wafer_ms", "host_ms", "wafer_host_ratio_max"]].assign(wafer_min_ms=on.wafer_time_min * 1e3,
            wafer_max_ms=on.wafer_time_max * 1e3).round(2))
    print(f"median on-wafer window {on.wafer_ms.median():.1f} ms vs flag-off barrier {d[~d.sync].stream_ms.median():.1f} ms "
          f"-> ratio {on.wafer_ms.median() / d[~d.sync].stream_ms.median():.3f}")
    fig, axs = lib.panels(1, height=4.2)
    ax = axs[0]
    rows = [("flag off: barrier-closed", d[~d.sync].stream_ms, 0), ("sync: barrier-closed", on.stream_ms, 1),
            ("sync: on-wafer window", on.wafer_ms, 2)]
    for y, (lab, v, i) in enumerate(rows):
        st = lib.style("index", i)
        ax.scatter(v, [y] * len(v), s=60, color=st["color"], marker=st["marker"], edgecolors="white", zorder=3)
        ax.annotate(f"median {v.median():.1f} ms", (v.max(), y), xytext=(10, 0), textcoords="offset points", color=st["color"], va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_ylim(-0.6, 2.6)
    lib.finish(ax, "time per SpMV pass (ms)", "", "65,536 PEs, n=3 runs per arm (5 passes each for sync)")
    ax.set_xlim(35, 52); plt.show()''',
        means="""*(Written 2026-10-13.)*
- **The clocks agree.** The on-wafer window is 40.7 / 40.8 / 41.4 ms. That's **0.88 of the sync build's own barrier time in every one
  of the 15 passes**, and 0.90 of the flag-off build's 45.2 ms. The barrier overstates on-wafer time by a steady 11–13%: launch,
  start/stop skew across PEs, and the barrier itself.
- The sync build costs about +1 ms on the host metric (46.2 vs 45.2 ms).
- **The compute-time hope was mis-framed.** valtask computes while the values arrive, so the window spans the transfer. It can't
  isolate compute. That needs a resident-A run, the CG design.
- **Consequence:** the barrier-closed numbers from 10-07 onward can be trusted as an upper bound within 11–13%.""",
        vendor="""**Agrees.** [BW]:57–58 "To record the elapsed time on host may not show the true timing because the runtime may not start the
transaction when the user calls a H2D/D2H command", and the vendor benchmark "samples the timing on the device" ([BW]:60).
Synced timing vendors that sync module (csl-examples, `src/csl/sync/`).""",
    ),
    dict(
        q="Weak scaling of the stream: what limits it?",
        hyp="""**Prediction (plan page):** bytes grow with PEs (~1 KB/PE), so time grows roughly linearly at 1.36–2.0 GB/s (the 65K and 10-08
full-wafer rates). Below 65K, fixed costs dominate. Full wafer 430–470 ms. Hope: the rate reaches ≥2 GB/s and flattens.
**Model tested afterwards (not pre-registered):** valtask serializes its W column transfers, so
T = W · (c + bytes_per_column / B). This was fitted once on all runs.
**Falsifier for the model:** residuals beyond run-to-run noise (~5–10%), or a size-dependent drift in them.""",
        code='''if ws.empty:
    lib.no_data("2026-10-09_weak-scaling.csv")
else:
    ws["W"] = ws.grid_width
    ws["col_bytes"] = ws.h2d_bytes_total / ws.W
    ws["col_s"] = ws.stream_time_corrected / ws.W
    A = np.c_[np.ones(len(ws)), ws.col_bytes]
    (c, inv), *_ = np.linalg.lstsq(A, ws.col_s, rcond=None)
    B = 1 / inv
    ws["model_ms"] = ws.W * (c + ws.col_bytes / B) * 1e3
    print(f"fit over {len(ws)} runs: per-column fixed cost c = {c * 1e6:.1f} us, per-column bandwidth B = {B / 1e9:.2f} GB/s")
    t = ws.groupby("num_pes").agg(grid=("grid_width", lambda s: f"{s.iloc[0]}x{ws.loc[s.index[0], 'grid_height']}"),
        n=("stream_ms", "size"), MB=("h2d_bytes_total", lambda s: s.iloc[0] / 1e6), min_ms=("stream_ms", "min"),
        median_ms=("stream_ms", "median"), max_ms=("stream_ms", "max"), GBps=("stream_GBps", "median"), model_ms=("model_ms", "first"))
    t["resid %"] = 100 * (t.model_ms / t.median_ms - 1)
    t["fixed-cost share %"] = 100 * (t.index.map(ws.groupby("num_pes").W.first()) * c * 1e3) / t.median_ms
    display(t.round(2))
    fig, axs = lib.panels(2, sharex=False, sharey=False)
    ax = axs[0]
    ax.scatter(ws.col_bytes / 1e3, ws.col_s * 1e3, s=40, color=lib.BLUE, edgecolors="white", zorder=3, label="each run")
    xs = np.linspace(0, ws.col_bytes.max() * 1.05, 50)
    ax.plot(xs / 1e3, (c + xs / B) * 1e3, color=lib.ORANGE, ls="--", label=f"fit: {c * 1e6:.1f} µs + bytes / {B / 1e9:.2f} GB/s")
    for p, g in ws.groupby("num_pes"):
        ax.annotate(f"{p:,}", (g.col_bytes.iloc[0] / 1e3, g.col_s.median() * 1e3), xytext=(6, -12), textcoords="offset points", fontsize=10)
    lib.finish(ax, "bytes per column transfer (KB)", "time per column transfer (ms)", "one serialized column transfer")
    ax = axs[1]
    lib.draw_series(ax, ws, "num_pes", "stream_ms", "index", 0, label="measured (median line)")
    m = ws.groupby("num_pes").model_ms.first()
    ax.plot(m.index, m.values, color=lib.ORANGE, ls="--", marker="s", label="model")
    lib.finish(ax, "PEs", "stream time, barrier-closed (ms)", "weak scaling, ~200 NNZ/PE, d=1e-3", logx=True, logy=True,
               xticks=[1024, 4096, 16384, 65536, 262144, 857010], yticks=[1, 3, 10, 30, 100, 300, 1000])
    plt.show()''',
        means="""*(Written 2026-10-13.)*
- **Every point validated, and the predictions held:** 262K 158 ms, 428K 249 ms, full wafer **458.8 ms** (417.7–480.1; 10-08's
  single run of 443 ms falls inside that range).
- **The rate rises with scale** (0.37 → 1.86 GB/s) and does not flatten at ≥2 GB/s. 13.1× the PEs from 65K costs 10.2× the time.
- **The model fits all 7 sizes within ~3%** (fit over all 21 runs) with one fixed cost per column (~57 µs) and one per-column bandwidth (~2.07 GB/s).
  The fixed cost explains the low small-size rates: it's 82% of the time at 1K, 32% at 65K and 9% at the full wafer.
  The ceiling the rate approaches is the per-column bandwidth, not the 12 GB/s the link delivers for copies (section 4).
- **What follows:** the lever is overlapping the column transfers. Header tolerates overlap. For valtask it's untested on hardware,
  and the simulator does not reproduce the hangs. Fewer grid columns would cut the fixed share.
- **Not established:** what sets the 2 GB/s per column (channel mapping of a 1-wide rectangle vs per-PE consumption). The model is
  descriptive, fitted after seeing the data.""",
        vendor="""**Partly comparable.** [BW]:19–20 "the overhead of TCP is about 200 us, a non-negligible cost when the transaction is small."
Our fixed cost per column transfer is ~57 µs, under that figure, but of the same kind. [SDK]:112 "You can block and unblock the input
tensor colors to overlap computation and communication": that's on-wafer overlap. **Silent** on overlapping host transfers, and on what a
1-column-wide streaming transfer can reach.""",
    ),
    dict(
        q="Setup copy across scale: where is the ceiling?",
        hyp="""**Prediction:** setup copy rises with transfer size and saturates near the M1/M4 figure of 10.3 GB/s at 16 channels.
**Falsifier:** saturation clearly above or below 10.3 GB/s.""",
        code='''if ws.empty:
    lib.no_data("2026-10-09_weak-scaling.csv")
else:
    display(lib.summarize(ws, ["num_pes"], "setup_GBps", 1, 2).assign(MB=ws.groupby("num_pes").setup_h2d_bytes.first().values / 1e6))
    big = ws[ws.num_pes >= 65536]
    A = np.c_[np.ones(len(big)), big.setup_h2d_bytes]
    (c, inv), *_ = np.linalg.lstsq(A, big.setup_h2d_time_corrected, rcond=None)
    print(f"fit >= 65K PEs: setup = {c * 1e3:.2f} ms + bytes / {1 / inv / 1e9:.2f} GB/s, "
          f"max residual {100 * np.max(np.abs((A @ [c, inv]) / big.setup_h2d_time_corrected - 1)):.1f}%")
    fig, axs = lib.panels(1)
    ax = axs[0]
    lib.draw_series(ax, ws, "num_pes", "setup_GBps", "index", 0, label="setup copy, 32 strips")
    ax.axhline(10.3, color=lib.INK, ls=(0, (2, 3)), lw=1.2, label="M1/M4 figure, 10.3 GB/s")
    lib.draw_claims(ax, 16)
    lib.finish(ax, "PEs", "setup copy, barrier-closed (GB/s)", "16 channels, col-major, 32 strips", logx=True,
               xticks=[1024, 4096, 16384, 65536, 262144, 857010])
    ax.set_ylim(0, 16)
    lib.place_claims(fig); plt.show()''',
        means="""*(Written 2026-10-13.)*
- **Falsified, upward:** setup saturates at **12.0 GB/s** from 262K on, not 10.3. The fit from 65K up is ~1.3 ms + bytes / 12.0 GB/s, within ~5% (the 65K runs scatter most).
- 12.0 GB/s is **96% of one 100 GbE link** (12.5 GB/s). The copy is bandwidth-bound, probably at the link itself.
- The 10.3 GB/s figure was measured on 65K-sized transfers. It's a size-dependent rate, not a ceiling. The paper corrections list was updated (f140bfd).
- **Not established:** whether more than one link is in use. The rate stopping just under one link's line rate suggests one.""",
        vendor="""**Consistent.** [BW]:17 "The host connects the device via 100Gbps ethernets." That's one link = 12.5 GB/s (red line; the plural
leaves the link count open). [BW]:18–19 "around 7Gbps" per channel: 14 GB/s at 16 channels, which we don't reach. The binding limit is
nearer the link than the channel sum.""",
    ),
    dict(
        q="Did the wall time get better? Before vs after, by phase",
        hyp="""**Prediction:** the optimizations (valtask, column mode, col-major setup, 16 channels, strips) cut wafer I/O by an order of
magnitude at 65K. Wall time improves much less, because program load (~40 s) and host tiling (~10 s) don't change.
**Falsifier:** wall time tracking wafer I/O.""",
        code='''timed = ["matrix_generation_time", "vector_generation_time", "tile_extract_time", "sdk_init_time", "program_load_time",
         "setup_done_time", "setup_barrier_d2h_time", "setup_barrier_d2h_repeat_time", "device_done_time", "barrier_d2h_time",
         "barrier_d2h_repeat_time", "memcpy_d2h_time", "gather_results_time", "validation_time"]
def phases(d, label):
    d = d.copy()
    d["encode_s"] = d.total_time - d[[c for c in timed if c in d]].fillna(0).sum(axis=1)   # untimed: host stream building
    d["io_s"] = d.setup_h2d_time_corrected + d.stream_time_corrected
    r = d[["program_load_time", "tile_extract_time", "encode_s", "io_s", "stream_time_corrected", "setup_h2d_time_corrected", "total_time"]].median()
    r["n"] = len(d); r.name = label
    return r
o, s1, fw = (D[k][D[k].num_pes == p] if not D[k].empty else D[k] for k, p in (("old", 65536), ("s1", 65536), ("fw", 857010)))
rows = []
if not o.empty:
    rows.append(phases(o[(o.encoding == "header") & (o.h2d_mode == "full") & (o.channels == 1)], "65K old defaults: header/full/1ch/row (10-07)"))
if not s1.empty:
    rows.append(phases(s1[(s1.encoding == "valtask") & (s1.channels == 16)], "65K valtask/column/16ch/col, 1 strip (S1 10-07)"))
rows.append(phases(D["d56"][~D["d56"].sync], "65K new defaults + 32 strips (10-09)"))
if not fw.empty:
    rows.append(phases(fw[fw.encoding == "header"], "857K header/column/16ch/col, 1 strip (10-08)"))
rows.append(phases(D["weak"][D["weak"].num_pes == 857010], "857K new defaults + 32 strips (10-09)"))
t = pd.DataFrame(rows)
t.columns = ["load s", "tiling s", "encode s (untimed)", "wafer I/O s", "stream s", "setup s", "total s", "n"]
display(t.round(3))
print(f"65K wafer I/O old -> new: {t.iloc[0]['wafer I/O s'] / t.iloc[2]['wafer I/O s']:.1f}x; "
      f"wall {t.iloc[0]['total s'] / t.iloc[2]['total s']:.2f}x; full wafer wall header -> new: {t.iloc[3]['total s'] / t.iloc[4]['total s']:.2f}x")
fig, axs = lib.panels(2, sharex=False, sharey=False, height=3.4)
for ax, sel, title in [(axs[0], [0, 1, 2], "65,536 PEs"), (axs[1], [3, 4], "857,010 PEs (full wafer)")]:
    sub = t.iloc[sel]
    left = np.zeros(len(sub))
    for col, color, lab in [("load s", lib.BLUE, "program load"), ("tiling s", lib.ORANGE, "host tiling"),
                            ("encode s (untimed)", lib.GREEN, "host stream encoding (untimed)"), ("wafer I/O s", lib.INK, "wafer I/O")]:
        ax.barh(range(len(sub)), sub[col], left=left, color=color, edgecolor="white", label=lab)
        left += sub[col].values
    for i, v in enumerate(sub["total s"]):
        ax.annotate(f"{v:,.0f} s", (v, i), xytext=(6, 0), textcoords="offset points", va="center")
    ax.set_yticks(range(len(sub)), [s.split(" (")[0] for s in sub.index]); ax.invert_yaxis()
    lib.finish(ax, "wall time per run (s)", "", title)
    ax.get_legend().remove()
axs[1].legend(loc="lower right", ncol=1)
fig.tight_layout()
plt.show()''',
        means="""*(Written 2026-10-13.)*
- **Wafer I/O got ~20× faster at 65K** (1.44 s → 0.074 s; stream 11.6×, setup 33×). At the full wafer, against the header run, it's 1.45×.
- **The biggest wall-time win was on the host:** header needs a Python stream-building step (`build_stream_buffer`, untimed in the CSV,
  so it appears as the gap). That costs ~96 s at 65K and **~1,210 s at the full wafer**. valtask's equivalent is ~1 s / ~8 s. So the
  full-wafer wall time fell **4.1×** (1,593 → 390 s).
- **At 65K the wall time improved only 1.4×,** because **program load doubled** (37–46 s → ~90 s). It was already bimodal within S1 on 10-07
  for the same config, so it's an appliance effect, not ours. At the old load time the 65K run would be ~58 s, 2.5× faster than the old defaults.
- **What remains:** program load (217 s) and host tiling (159 s) are 96% of a full-wafer run. Wafer I/O is under 0.3% of the wall time.
- **Not established:** that the whole gap is encoding (it also holds small untimed steps). The header/valtask difference, ~95 s at 65K
  and ~1,200 s at the full wafer, is far larger than any of those. Add a timer around `build_*_stream_buffer`.""",
        vendor="""**Silent.** [SDK]:7 says `SdkRuntime` is used to "load programs, launch kernels, and stream tensor data", but gives no load-time figures.
Host-side data preparation is the user's code.""",
    ),
    dict(
        q="Accuracy across scale",
        hyp="""**Prediction:** fp32 accumulation error grows roughly like √n (summing ~n·d random-sign terms per row); all runs stay below 1e-6.
**Falsifier:** growth faster than linear in n, or any run above 1e-6.""",
        code='''if ws.empty:
    lib.no_data("weak scaling")
else:
    e = ws.groupby("matrix_rows").relative_error.agg(["size", "min", "max"])
    e["max / sqrt(n)"] = e["max"] / np.sqrt(e.index)
    display(e)
    k = e["max / sqrt(n)"].median()
    fig, axs = lib.panels(1, height=4.4)
    ax = axs[0]
    ax.scatter(ws.matrix_rows, ws.relative_error, s=50, color=lib.BLUE, edgecolors="white", zorder=3, label="each run")
    xs = np.array([1.2e4, 5e5]); ax.plot(xs, k * np.sqrt(xs), color=lib.ORANGE, ls="--", label=f"{k:.2e} · √n")
    lib.finish(ax, "matrix dimension n", "relative error vs fp64", "all 18 weak-scaling runs + 65K", logx=True, logy=True,
               xticks=[14300, 57200, 114000, 229000, 414000], yticks=[5e-8, 1e-7, 2e-7, 3e-7])
    plt.show()''',
        means="""*(Written 2026-10-13.)*
- Error grows close to √n (the ratio max/√n stays within 4.3–5.2e-10) and peaks at **2.75e-07 at the full wafer**. Repeat runs at the
  same size give identical errors: the reduction order is deterministic.
- The 16-channel default, which clamps to the grid height below 16 rows (`4d7d244`), validated at every size.""",
        vendor="""**Silent.** No vendor figure for fp32 SpMV accuracy.""",
    ),
]

nb = notebook(
    "SpMV scaling: setup strips, synced timing, weak scaling to the full wafer (2026-10-09)",
    """The 2026-10-09 CS-3 window: 65K setup-strips A/B synced on-wafer timing, and weak scaling at ~200 NNZ/PE, d=1e-3,
from 1,024 to 857,010 PEs (n=3 per point), all with the new defaults: valtask, column mode, col-major setup, 16 channels
(clamped to the grid height), 32 setup strips. Wall-time comparisons reach back to the 10-07 and 10-08 runs. Only barrier-closed
times count. Predictions were written down before the runs; where a model was fitted afterwards, the text says so.""",
    d107_sections,
    extra_cells=[D107_LOAD],
)
save("spmv_scaling.ipynb", nb)
print("written spmv_scaling")


# =====================================================================================
# v1_vs_v2  (twelve cells: what changed from v1.0 to v2.0; every number is read from the CSVs)
# =====================================================================================
import _cells_v1_vs_v2 as C  # noqa: E402

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [
    nbf.v4.new_markdown_cell(C.INTRO),
    nbf.v4.new_markdown_cell(C.S1_MD),
    nbf.v4.new_code_cell(C.SETUP + "\n\n" + C.S1_CODE),
    nbf.v4.new_markdown_cell(C.S2_MD),
    nbf.v4.new_code_cell(C.S2_CODE),
    nbf.v4.new_markdown_cell(C.S3_MD),
    nbf.v4.new_code_cell(C.S3_CODE),
    nbf.v4.new_markdown_cell(C.S4_MD),
    nbf.v4.new_code_cell(C.S4_CODE),
    nbf.v4.new_markdown_cell(C.S5_MD),
    nbf.v4.new_code_cell(C.S5_CODE),
    nbf.v4.new_markdown_cell(C.NEXT_MD),
]
save("v1_vs_v2.ipynb", nb)
print("written v1_vs_v2")
