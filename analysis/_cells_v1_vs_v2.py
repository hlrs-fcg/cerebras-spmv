"""Cell sources of analysis/v1_vs_v2.ipynb (assembled by _build_notebooks.py; edit the text here, not in the .ipynb).

Every number the notebook shows is computed from the CSVs, except the host-CPU times (docs/results.md, lib.HOST_CPU_S), the vendor
figures (lib.GBPS_PER_CHANNEL, lib.LINK_GBPS) and the pre-registered prediction bands of section 3 (docs/resident-spmv.md, quoted and
labelled as such). Each section is a markdown cell with the Hypothesis, then a code cell whose output is the Result, with What that
means (computed from the data) and What the vendor says printed under it.
"""

INTRO = """# SpMV on the Cerebras CS-3: v2.0 against v1.0

A five-minute walk through what changed. **v1.0** (the ISC26 poster, June 2026) sent every matrix entry to the wafer for every
product and timed the host's blocking calls. **v2.0** keeps the matrix on the wafer (*resident*), streams it only when it does not
fit, and times everything with a completion barrier.

**Timing basis.** v1.0 numbers are *host-return* times, which stop when the host is done with a call, not when the device is
(`docs/host-timing-barrier.md`): lower bounds for what the user waited. v2.0 numbers are barrier-closed, or the on-wafer window of
synced timing (`docs/synced-timing.md`). **Data.** v1.0: `cluster/results/v1.0-data/`. v2.0 release candidate RC1 (commit
`4739dc4`): `cluster/results/v2.0-data/`. Earlier *development* runs that RC1 does not repeat (streaming at 65K and the full wafer,
the first resident kernels) are copies in `analysis/data/dev/`, marked `dev` below and replaced by RC1 rows when those exist.

Run: `uv run --no-project --python 3.11 --with numpy --with pandas --with matplotlib --with jupyter jupyter lab analysis/v1_vs_v2.ipynb`."""

SETUP = '''import sys, pathlib
for here in (pathlib.Path.cwd(), pathlib.Path.cwd() / "analysis"):          # run from the repo root or from analysis/
    if (here / "lib.py").exists():
        sys.path.insert(0, str(here)); break
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import lib
lib.setup_plots()
pd.set_option("display.width", 200, "display.max_columns", 30)

v1 = lib.load_v1()
v2all = lib.good(lib.load_v2())
v2 = v2all[v2all.matrix == "random"] if not v2all.empty else v2all          # the generator matrices (sections 1-4)
mtx = v2all[v2all.matrix != "random"] if not v2all.empty else v2all         # real SuiteSparse matrices (section 5)
ok = lambda df: df[df["null_spmv"].astype(str) != "True"]          # rows that computed y (not null twins)
pending = lambda what: display(Markdown(f"*pending: {what}; this cell reads it from `cluster/results/v2.0-data/` when it appears.*"))

inv = pd.DataFrame([
    ("v1.0 (ISC26)", "cluster/results/v1.0-data", len(v1)),
    ("v2.0 RC1", "cluster/results/v2.0-data", 0 if v2.empty else int((v2.origin == "RC1").sum())),
    ("development runs", "analysis/data/dev", 0 if v2.empty else int((v2.origin == "dev").sum())),
    ("real-matrix runs", "cluster/results/v2.0-data", len(mtx)),
], columns=["data", "directory", "valid rows"])
display(inv)
if v2.empty or not ((v2.origin == "RC1") & (v2.encoding == "valtask") & (v2.num_pes > 20_000)).any():
    pending("valtask streaming at 65K and the full wafer from RC1 (section 1 uses the development runs for it)")'''

S1_MD = """---
## 1. One SpMV, v1.0 against v2.0

### Hypothesis
- **v1.0** moves the whole matrix over the host link per product, one blocking transfer per PE: seconds per SpMV, set by the
  link and the call count, not by the wafer.
- **v2.0 streaming** moves it as 4 bytes per NNZ in a few hundred large transfers: bounded by the link rate, tens to hundreds of
  milliseconds. That still loses to the host CPU, which reads the same bytes from DRAM.
- **v2.0 resident** moves nothing per product: microseconds on the wafer, about a millisecond through the host runtime.
- The streaming rate should sit well under the vendor's per-channel figure, because a stream is one rectangle at a time."""

S1_CODE = '''# ---- the series, per PE count: series -> {pes: (times of the runs, NNZ of that matrix, where the rows come from)}
S = {}
def put(name, pes, times, nnz, origin):
    if len(times):
        S.setdefault(name, {})[pes] = (np.asarray(times, float), float(nnz), origin)

if not v1.empty:
    for scaling, pes in (("weak", 65_536), ("strong", int(v1.num_pes.max()))):
        r = v1[v1.scaling.str.startswith(scaling) & (v1.num_pes == pes)]
        put("v1.0 per-PE transfers", pes, r.call_s, r.total_nnz.iloc[0] if len(r) else 0, "v1.0")
if not v2.empty:
    for pes in (65_536, lib.FULL_WAFER):
        st = v2[(v2.encoding == "valtask") & (v2.num_pes == pes) & ~v2.sync_timing.astype(bool)]
        if len(st):
            org = "RC1" if (st.origin == "RC1").any() else "dev"
            st = st[st.origin == org]
            put("v2.0 streaming (valtask)", pes, st.stream_time_corrected, st.total_nnz.iloc[0], org)
        rs = ok(v2[(v2.encoding == "resident") & (v2.num_pes == pes) & (v2.origin == "RC1")])
        if len(rs):
            put("v2.0 resident, on-wafer window", pes, rs.window_s, rs.total_nnz.iloc[0], "RC1")
            put("v2.0 resident, host-visible call", pes, rs.stream_time_corrected, rs.total_nnz.iloc[0], "RC1")
    for pes, t in lib.HOST_CPU_S.items():
        nz = v2[(v2.num_pes == pes) & (v2.origin == "RC1")].total_nnz
        if len(nz):
            put("host CPU, 32 threads", pes, [t], nz.iloc[0], "docs/results.md")

table = pd.DataFrame([dict(series=s, PEs=f"{p:,}", origin=o, runs=len(t), min=lib.fmt_s(t.min()), median=lib.fmt_s(np.median(t)),
                           max=lib.fmt_s(t.max()), NNZ=f"{nz / 1e6:.1f}M", ns_per_NNZ=f"{np.median(t) / nz * 1e9:.3g}")
                      for s, d in S.items() for p, (t, nz, o) in sorted(d.items())])
display(table)

if len(table):
    style = {"v1.0 per-PE transfers": lib.style("index", 1), "v2.0 streaming (valtask)": lib.style("index", 0),
             "v2.0 resident, on-wafer window": lib.style("index", 2),
             "v2.0 resident, host-visible call": dict(lib.style("index", 2), ls="-", marker="v"),
             "host CPU, 32 threads": dict(color=lib.INK, marker="D", ls="--")}
    fig, axs = lib.panels(2, sharey=False, sharex=True)
    for ax, per_nnz in zip(axs, (False, True)):
        scale = lambda p, d: (1e9 / d[p][1]) if per_nnz else 1.0
        for s, d in S.items():
            xs = sorted(d)
            ys = [np.median(d[p][0]) * scale(p, d) for p in xs]
            st = style[s]
            for p in xs:
                ax.scatter([p] * len(d[p][0]), d[p][0] * scale(p, d), s=40, color=st["color"], marker=st["marker"], alpha=0.9,
                           edgecolors="white", linewidths=0.6, zorder=2)
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.2, ms=7, mec="white", mew=0.6, label=s, zorder=3)
            ax.annotate(s, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", color=st["color"], fontsize=11,
                        va="center", annotation_clip=False)
        lib.finish(ax, "PEs", "ns per NNZ" if per_nnz else "time per SpMV (s)", logx=True, logy=True,
                   title="the same, per non-zero (the matrices differ in size)" if per_nnz else "time of one SpMV",
                   xticks=[65_536, 524_288, lib.FULL_WAFER])
        ax.set_xlim(40_000, 3_000_000)
    fig.tight_layout(); plt.show()
else:
    pending("v1.0 or v2.0 rows for this comparison")

# ---- streaming rate against the vendor's figure (effective host-to-wafer bytes per second)
ws = v2[(v2.encoding == "valtask") & (v2.h2d_bytes_total > 0) & ~v2.sync_timing.astype(bool)] if not v2.empty else v2
if len(ws):
    ws = ws.assign(rate=ws.h2d_bytes_total / ws.stream_time_corrected / 1e9)
    fig, (ax,) = lib.panels(1, sharey=False, sharex=False)
    lib.draw_series(ax, ws, "num_pes", "rate", "index", 0, label="v2.0 streaming (valtask), 16 channels")
    lib.draw_claims(ax, 16)
    lib.finish(ax, "PEs", "streamed A, GB/s (barrier-closed)", logx=True, logy=True,
               title="how fast A actually streams", xticks=[1024, 16384, 262144, lib.FULL_WAFER])
    lib.place_claims(fig); plt.show()
    best = ws.rate.max()

# ---- what that means (computed)
def t(series, pes):
    return np.median(S[series][pes][0]) if series in S and pes in S[series] else float("nan")
lines = []
for pes in (65_536, lib.FULL_WAFER):
    stream, win, hostv, cpu = (t("v2.0 streaming (valtask)", pes), t("v2.0 resident, on-wafer window", pes),
                               t("v2.0 resident, host-visible call", pes), t("host CPU, 32 threads", pes))
    if np.isfinite(stream * win * hostv * cpu):
        lines.append(f"- **{pes:,} PEs:** streaming {lib.fmt_s(stream)} per SpMV against the host CPU's {lib.fmt_s(cpu)} "
                     f"({stream / cpu:.1f}x **slower**); resident window {lib.fmt_s(win)} ({cpu / win:,.0f}x faster than the CPU), "
                     f"{lib.fmt_s(hostv)} when the call goes through the host runtime ({cpu / hostv:.1f}x).")
v1_65, st_65 = t("v1.0 per-PE transfers", 65_536), t("v2.0 streaming (valtask)", 65_536)
if np.isfinite(v1_65) and np.isfinite(st_65):
    nz1, nz2 = S["v1.0 per-PE transfers"][65_536][1], S["v2.0 streaming (valtask)"][65_536][1]
    lines.append(f"- **v1.0 to v2.0 streaming at 65K:** {lib.fmt_s(v1_65)} ({nz1 / 1e6:.0f}M NNZ) to {lib.fmt_s(st_65)} "
                 f"({nz2 / 1e6:.0f}M NNZ): {v1_65 / nz1 / (st_65 / nz2):,.0f}x less time per NNZ.")
if len(ws):
    lines.append(f"- Streaming reaches at best **{best:.2f} GB/s**; the vendor example's comment gives {lib.GBPS_PER_CHANNEL:.3f} GB/s per "
                 f"channel ({lib.GBPS_PER_CHANNEL * 16:.0f} GB/s at 16) and one 100 GbE link is {lib.LINK_GBPS} GB/s (red dotted): "
                 f"we get {100 * best / lib.LINK_GBPS:.0f}% of the link.")
vendor = ("`csl-examples` bandwidth-test: \\"around 7Gbps\\" per I/O channel (`run.appliance.py:18`, a comment in an SDK 1.4.0 example; "
          "the cluster runs 2.10.0). "
          + (f"At 16 channels that is {lib.GBPS_PER_CHANNEL * 16:.0f} GB/s against the {best:.2f} GB/s a streamed matrix gets here "
             f"({lib.GBPS_PER_CHANNEL * 16 / best:.0f}x lower): " if len(ws) else "")
          + "the rate is one narrow rectangle at a time (`docs/adr/ADR-008-host-wafer-transfer-and-timing.md`). "
          "Nothing from the vendor on SpMV time per call.")
display(Markdown("### What that means\\n" + "\\n".join(lines) + """
- v1.0 had no full-wafer run (its largest grid is the second orange point, a different matrix); v1.0 times are host-return lower bounds
  and exclude the per-call partitioning, so the real gap is larger.
- Streaming A every call cannot beat the host CPU; **keeping A resident can**, but only if the loop runs on the wafer: through the
  host runtime every call costs about a millisecond.

### What the vendor says
""" + vendor))'''

S2_MD = """---
## 2. Three resident kernels, one winner

### Hypothesis
The first resident kernel ran the local SpMV as a CSR row loop. On a hypersparse tile (well under one non-zero per row) most rows
are empty, so per-row work should dominate and a **COO scatter** (no row loop) should win **by more the sparser the tile is**; a
CSR loop on 16-bit loop state should help everywhere a little. Rule for choosing: the lowest valid window at scale."""

S2_CODE = '''names = {0: "kernel 0: CSR loop", 1: "kernel 1: CSR on u16", 2: "kernel 2: COO scatter (ships)"}
grids = [("16 PEs", 16), ("65K PEs", 65_536), ("full wafer", lib.FULL_WAFER)]
K = {}                                                          # (kernel, pes) -> windows (cycles), nnz_max / ky, origin
def kput(k, pes, df, origin):
    df = ok(df)
    if len(df):
        K[(k, pes)] = (df.window_cycles.to_numpy(float), float(df.max_nnz_per_pe.iloc[0] / df.ky.iloc[0]), origin)

if not v2.empty:
    src = lambda pat: v2[v2.source.str.contains(pat, regex=False)]
    kput(0, 16, src("kernelcheck16_hw_k0.csv"), "dev"); kput(1, 16, src("kernelcheck16_hw_k1.csv"), "dev")
    kput(2, 16, src("kernelcheck16_hw_k2.csv"), "dev")
    kput(0, 65_536, src("resident-k0-65k"), "dev"); kput(1, 65_536, src("kernelcheck65_hw_k1"), "dev")
    rc65 = v2[(v2.origin == "RC1") & (v2.num_pes == 65_536) & (v2.encoding == "resident")]
    kput(2, 65_536, rc65 if len(rc65) else src("kernelcheck65_hw_k2"), "RC1" if len(rc65) else "dev")
    kput(0, lib.FULL_WAFER, src("resident-k0-full"), "dev")
    kput(2, lib.FULL_WAFER, v2[(v2.origin == "RC1") & (v2.num_pes == lib.FULL_WAFER) & (v2.encoding == "resident")], "RC1")

rows = [dict(kernel=names[k], grid=lbl, origin=K[(k, p)][2], runs=len(K[(k, p)][0]),
             cycles=int(round(np.median(K[(k, p)][0]))), us=round(np.median(K[(k, p)][0]) / lib.CLOCK_HZ * 1e6, 1),
             nnz_per_row=round(K[(k, p)][1], 2))
        for lbl, p in grids for k in (0, 1, 2) if (k, p) in K]
display(pd.DataFrame(rows))

def w(k, p):
    return np.median(K[(k, p)][0]) if (k, p) in K else float("nan")
def vs(other, mine):                                           # "1.55x faster" / "1.09x slower" from two windows
    r = other / mine
    return f"{r:.2f}x faster" if r >= 1 else f"{1 / r:.2f}x slower"

lines = []
if rows:
    fig, axs = lib.panels(2, sharey=False, sharex=False)
    ax = axs[0]
    for k in (0, 1, 2):
        pts = [(i, w(k, p)) for i, (lbl, p) in enumerate(grids) if (k, p) in K]
        if not pts:
            continue
        st = lib.style("index", k)
        ax.plot([i for i, _ in pts], [y for _, y in pts], color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.2, ms=8,
                mec="white", mew=0.6, label=names[k])
        ax.annotate(names[k], pts[-1], xytext=(8, 0), textcoords="offset points", color=st["color"], fontsize=11, va="center",
                    annotation_clip=False)
    ax.set_xticks(range(3), [g for g, _ in grids]); ax.set_xlim(-0.2, 2.9)
    lib.finish(ax, "grid", "window of one SpMV (cycles at 0.85 GHz)", logy=True, title="the window, per kernel and size",
               yticks=[10_000, 20_000, 30_000, 40_000])
    ax = axs[1]
    for k in (1, 2):
        pts = sorted((K[(k, p)][1], w(0, p) / w(k, p)) for lbl, p in grids if (k, p) in K and (0, p) in K)
        if pts:
            st = lib.style("index", k)
            ax.plot(*zip(*pts), color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.2, ms=8, mec="white", mew=0.6,
                    label=names[k])
            ax.annotate(names[k], pts[-1], xytext=(8, 9 if k == 1 else -9), textcoords="offset points", color=st["color"],
                        fontsize=11, va="center", annotation_clip=False)
    ax.axhline(1.0, color=lib.INK, lw=1.0, ls=":")
    lib.finish(ax, "NNZ per row on the busiest PE (sparser to the left)", "speed-up over kernel 0", logx=True,
               title="the sparser the tile, the more the COO scatter gains", xticks=[0.6, 1, 2, 3])
    ax.set_xlim(0.45, 6); fig.tight_layout(); plt.show()
    for lbl, p in grids:
        if (0, p) in K and (2, p) in K:
            lines.append(f"- **{lbl}:** kernel 2 is {vs(w(0, p), w(2, p))} than kernel 0"
                         + (f" and {vs(w(1, p), w(2, p))} than kernel 1" if (1, p) in K else "")
                         + f" (busiest PE: {K[(2, p)][1]:.2f} non-zeros per row).")
    best16 = min((k for k in (0, 1, 2) if (k, 16) in K), key=lambda k: w(k, 16), default=None)
    if best16 is not None and best16 != 2:
        lines.append(f"- At 16 PEs the fastest is **{names[best16]}** ({w(best16, 16):,.0f} cycles), not kernel 2 ({w(2, 16):,.0f}): "
                     "dense tiles (several non-zeros per row) do not pay for the scatter.")
else:
    pending("the kernel-check rows")
display(Markdown("### What that means\\n" + "\\n".join(lines) + """
- The hypothesis holds: the COO scatter's gain **grows as tiles get sparser**, which is the regime of real sparse matrices at scale.
  Kernel 2 ships because the rule is the lowest valid window at scale, not at 16 PEs. Its price is one more array (`row_ids`) per PE.
- Kernel 1 was not run at the full wafer; it was not the winner at 65K, so it was dropped without that run.

### What the vendor says
Cerebras' own hypersparse SpMV (`csl-examples/benchmarks/spmv-hypersparse`) is a scalar loop on 16-bit (`u16`) indices, the idea of
kernel 1. **Agrees** that integer width matters (kernel 1 beats kernel 0 everywhere), but it is not the biggest lever here."""))'''

S3_MD = """---
## 3. The launch floor and a timing model that predicts

### Hypothesis
The synced window of one SpMV is **floor + chain + compute**: the launch machinery that a no-op kernel (the *null twin*) measures,
the east-west reduction that costs a fixed number of cycles per PE column, and the local scatter at a fixed cost per non-zero of the
busiest PE. If that is right, **two sizes fix the constants and predict the third**, and the floor is a straight line in grid
width + height. Predictions were written down before the runs (`docs/resident-spmv.md`, "Scorecard")."""

S3_CODE = '''# ---- the floor: null twins (spmv() is a no-op), cycles of the window, against grid width + height
nulls = v2[(v2.null_spmv.astype(str) == "True")] if not v2.empty else v2
lines = []
if len(nulls):
    nulls = nulls.assign(wh=nulls.grid_width + nulls.grid_height)
    fl = nulls.groupby("num_pes").agg(wh=("wh", "first"), floor=("window_cycles", "median"), runs=("window_cycles", "size"),
                                       W=("grid_width", "first"), H=("grid_height", "first")).reset_index()
    b, a = np.polyfit(fl.wh, fl.floor, 1)
    display(fl.round(1))

# ---- the model: window = floor + c*W + e*NNZ_max, fitted on two sizes, predicting the third (kernel 2, RC1 rows)
rc = ok(v2[(v2.origin == "RC1") & (v2.encoding == "resident")]) if not v2.empty else v2
pts = []
if len(rc) and len(nulls):
    for pes, g in rc.groupby("num_pes"):
        fr = nulls[nulls.num_pes == pes].window_cycles
        if len(fr):
            pts.append(dict(pes=pes, W=int(g.grid_width.iloc[0]), N=int(g.max_nnz_per_pe.iloc[0]),
                            window=float(g.window_cycles.median()), floor=float(fr.median())))
P = pd.DataFrame(pts)
if len(P) >= 3:
    P = P.sort_values("pes").reset_index(drop=True)
    P["net"] = P.window - P.floor
    pred = []
    for i in range(len(P)):
        rest = P.drop(index=i)
        c, e = np.linalg.lstsq(np.column_stack([rest.W, rest.N]), rest.net, rcond=None)[0]
        pred.append(P.floor[i] + c * P.W[i] + e * P.N[i])
    P["predicted"] = pred
    P["error_%"] = 100 * (P.predicted / P.window - 1)
    c_all, e_all = np.linalg.lstsq(np.column_stack([P.W, P.N]), P.net, rcond=None)[0]
    display(P.round(1))

    # the predictions made before the runs (quoted from docs/resident-spmv.md, "Scorecard"; claims, not data)
    bands = {1: (15_700, 16_900), 2: (24_800, 26_000)}              # position -> (low, high) of the window band
    labels = [f"{int(p):,} PEs" if p < 100_000 else "full wafer" for p in P.pes]
    fig, axs = lib.panels(2, sharey=False, sharex=False)
    ax = axs[0]
    for _, r in nulls.iterrows():
        ax.scatter(r.wh, r.window_cycles, s=40, color=lib.ORANGE, marker="s", edgecolors="white", linewidths=0.6, zorder=2)
    xs = np.array([fl.wh.min(), fl.wh.max()])
    ax.plot(fl.wh, fl.floor, color=lib.ORANGE, marker="s", ls="--", lw=2.2, ms=8, mec="white", mew=0.6,
            label="null twin (measured)", zorder=3)
    ax.plot(xs, a + b * xs, color=lib.INK, lw=1.0, ls=":", label=f"line: {a:,.0f} + {b:.2f} x (W+H) cycles", zorder=1)
    ax.annotate("launch floor", (fl.wh.iloc[-1], fl.floor.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                color=lib.ORANGE, fontsize=11, va="center", annotation_clip=False)
    lib.finish(ax, "grid width + height (PEs)", "launch floor (cycles)", logx=True, logy=True,
               title="a no-op kernel still costs a window, growing with the grid", xticks=[8, 100, 500, 1900],
               yticks=[300, 1000, 3000])
    ax = axs[1]
    xpos = np.arange(len(P))
    for i, (lo, hi) in bands.items():
        if i < len(P):
            ax.fill_between([i - 0.28, i + 0.28], lo, hi, color=lib.ORANGE, alpha=0.35, zorder=1, label="pre-registered band")
    ax.vlines(xpos, np.minimum(P.window, P.predicted), np.maximum(P.window, P.predicted), color=lib.INK, lw=1.0, zorder=2)
    ax.scatter(xpos, P.window, s=80, color=lib.GREEN, marker="^", edgecolors="white", linewidths=0.6, zorder=3,
               label="measured window")
    ax.scatter(xpos, P.predicted, s=70, color=lib.BLUE, marker="o", edgecolors="white", linewidths=0.6, zorder=3,
               label="model fitted on the other two sizes")
    for i, r in P.iterrows():
        ax.annotate(f"{r['error_%']:+.1f}%", (i, max(r.window, r.predicted)), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=11, color=lib.BLUE)
    ax.set_xticks(xpos, labels); ax.set_xlim(-0.6, len(P) - 0.4)
    h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h))
    lib.finish(ax, "grid", "window of one SpMV (cycles)", title="predictions against measurements (kernel 2, RC1)")
    ax.set_ylim(P.window.min() * 0.85, P.window.max() * 1.15)
    ax.legend(seen.values(), seen.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=10)
    fig.tight_layout(); plt.show()

    lines.append(f"- Fitted on all three sizes: **{c_all:.1f} cycles per PE column** (the east-west chain) and **{e_all:.1f} cycles per "
                 f"non-zero** of the busiest PE; with the measured floor, a model fitted on the other two sizes predicts each window "
                 f"to within {P['error_%'].abs().max():.1f}%.")
    lines.append(f"- The floor is a line in width + height ({a:,.0f} + {b:.2f} x (W+H) cycles), so a no-op takes "
                 f"{lib.fmt_s(P.floor.iloc[-1] / lib.CLOCK_HZ)} at the full wafer: **{100 * P.floor.iloc[-1] / P.window.iloc[-1]:.0f}% of the "
                 f"full-wafer window is the launch itself**.")
else:
    pending("the RC1 kernel-2 runs and their null twins at three sizes")
display(Markdown("### What that means\\n" + "\\n".join(lines) + """
- The two orange bands were written down before the runs: the 65K run landed just **below** its band (a miss, in the favourable
  direction, recorded as a miss) and the full wafer **inside** its band (a hit). A model with two constants and a measured floor predicts
  the third size, so the cost is understood, not only measured.
- The chain grows with the grid width and the floor with width + height: at the full wafer much of the window is the price of being big,
  which is why per-call speed needs the loop on the wafer.

### What the vendor says
Silent on a floor of this kind. The sync module the window is read with is vendored from `csl-examples/benchmarks/bandwidth-test/src/sync`
(`NOTICE`)."""))'''

S4_MD = """---
## 4. Wall time, and when does the wafer pay off?

### Hypothesis
A run is dominated by getting the program onto the wafer and the matrix tiled, not by SpMV. The resident loop pays that back after
**thousands** of products: the break-even is the fixed cost divided by what one host SpMV costs (`fixed / host time`). Because a host round
trip (about a millisecond) is tiny against the host CPU's tens of milliseconds, **the break-even should hardly depend on whether the loop
runs on the wafer**; what the on-wafer loop changes is the time per product *after* the break-even."""

S4_CODE = '''rc = ok(v2[(v2.origin == "RC1") & (v2.encoding == "resident") & (v2.num_pes > 20_000)]) if not v2.empty else v2
lines = []
if len(rc):
    phases = [("program load", "program_load_time", lib.BLUE), ("host tiling and buffers", "host_prep_time", lib.ORANGE),
              ("setup copy", "setup_h2d_time_corrected", lib.GREEN)]
    agg = rc.groupby("num_pes").median(numeric_only=True)
    agg["rest"] = agg.total_time - sum(agg[c] for _, c, _ in phases)
    agg["fixed"] = agg.program_load_time + agg.host_prep_time + agg.setup_h2d_time_corrected
    display(agg[["total_time", "program_load_time", "host_prep_time", "setup_h2d_time_corrected", "rest"]].round(2))

    fig, axs = lib.panels(2, sharey=False, sharex=False)
    ax = axs[0]
    for i, (pes, r) in enumerate(agg.iterrows()):
        left = 0.0
        for name, col, colr in phases + [("everything else", "rest", "#b8b8b8")]:
            ax.barh(i, r[col], left=left, color=colr, edgecolor="white", label=name if i == 0 else None)
            if r[col] > 0.03 * agg.total_time.max():
                ax.text(left + r[col] / 2, i, f"{r[col]:.3g} s", ha="center", va="center",
                        color=lib.INK if colr == "#b8b8b8" else "white", fontsize=11)
            left += r[col]
    ax.set_yticks(range(len(agg)), [f"{int(p):,} PEs" for p in agg.index]); ax.grid(axis="y", visible=False)
    lib.finish(ax, "seconds of one run", "", title="where the wall time goes")

    ax = axs[1]
    pes = lib.FULL_WAFER if lib.FULL_WAFER in agg.index else agg.index[-1]
    r = agg.loc[pes]; host = lib.HOST_CPU_S[int(pes)]
    n = np.logspace(0, 5.5, 200)
    curves = [("host CPU", n * host, dict(color=lib.INK, ls="--")),
              ("wafer, loop on the wafer", r.fixed + n * r.window_s, dict(color=lib.GREEN, ls="-")),
              ("wafer, each call via the host", r.fixed + n * r.stream_time_corrected, dict(color=lib.BLUE, ls=":"))]
    for name, y, st in curves:
        ax.plot(n, y, color=st["color"], ls=st["ls"], lw=2.2, label=name, zorder=3)
        ax.annotate(name, (n[-1], y[-1]), xytext=(8, 0), textcoords="offset points", color=st["color"], fontsize=11, va="center",
                    annotation_clip=False)
    be = {"loop": r.fixed / (host - r.window_s), "host": r.fixed / (host - r.stream_time_corrected)}
    ax.scatter([be["loop"]], [be["loop"] * host], s=90, color=lib.GREEN, marker="^", edgecolors=lib.INK, linewidths=1.0, zorder=4)
    ax.scatter([be["host"]], [be["host"] * host], s=70, color=lib.BLUE, marker="o", edgecolors=lib.INK, linewidths=1.0, zorder=4)
    ax.annotate(f"break-even: {be['loop']:,.0f} products (loop on the wafer),\\n{be['host']:,.0f} (each call via the host)",
                (be["loop"], be["loop"] * host), xytext=(-14, 22), textcoords="offset points", ha="right", fontsize=11, color=lib.INK)
    lib.finish(ax, "number of SpMVs", "time since the run started (s)", logx=True, logy=True,
               title=f"break-even against the host CPU at {int(pes):,} PEs", xticks=[1, 10, 100, 1000, 10_000, 100_000],
               yticks=[1, 10, 100, 1000, 10_000])
    ax.set_xlim(0.8, 3_000_000); fig.tight_layout(); plt.show()

    lines += [f"- At {int(pes):,} PEs the fixed cost is **{r.fixed:.0f} s** (program load {r.program_load_time:.0f} s, tiling "
              f"{r.host_prep_time:.1f} s, setup copy {r.setup_h2d_time_corrected:.2f} s) against **{lib.fmt_s(host)}** for the host CPU's SpMV: "
              f"break-even after **{be['loop']:,.0f} SpMVs** with the loop on the wafer ({lib.fmt_s(r.window_s)} each), "
              f"**{be['host']:,.0f}** if every call goes through the host runtime ({lib.fmt_s(r.stream_time_corrected)} each).",
              f"- The two break-evens differ by {100 * abs(be['loop'] - be['host']) / be['loop']:.1f}%: the fixed cost dominates. After it, a "
              f"product costs {lib.fmt_s(r.window_s)} on the wafer and {lib.fmt_s(r.stream_time_corrected)} through the host: "
              f"**{r.stream_time_corrected / r.window_s:,.0f}x** apart."]
    full_dev = v2[(v2.encoding == "valtask") & (v2.num_pes == lib.FULL_WAFER) & (v2.origin == "dev")]
    if len(full_dev) and lib.FULL_WAFER in agg.index:
        lines.append(f"- Host tiling of the full-wafer matrix took {full_dev.tile_extract_time.median():.0f} s in the development streaming "
                     f"runs and {rc[rc.num_pes == lib.FULL_WAFER].tile_extract_time.median():.1f} s in RC1 (a fused numba / ahead-of-time "
                     f"tiler): **program load is now {100 * agg.loc[lib.FULL_WAFER].program_load_time / agg.loc[lib.FULL_WAFER].total_time:.0f}% "
                     f"of a run**.")
else:
    pending("the RC1 resident runs")
display(Markdown("### What that means\\n" + "\\n".join(lines) + """
- A solver with thousands of iterations amortises program load and tiling; a single product never does.

### What the vendor says
Silent on program-load time. Whether a loaded program can stay on the wafer between jobs is a question for the appliance team (next cell)."""))'''

S5_MD = """---
## 5. Real matrices against the host CPU

### Hypothesis
The wafer's window is NNZ-bound (about 37 cycles per NNZ on the busiest PE) plus a grid-size floor of about 1.8 to 6 µs, so **tiny
matrices lose** (the floor alone exceeds what the host needs for a few hundred non-zeros) and the wafer **wins once the matrix is big
enough to hide the floor**, and more the more PEs share it. Expected: nos6 (3.3K NNZ) and bcsstk01 (400) at best level with one host
core; bcsstk14 (63K NNZ) ahead; parabolic_fem (3.7M NNZ, full wafer) far ahead. The host's own time is noisy at small sizes (32 threads
on a shared VM): the ratio is given against the best and the single-thread host."""

S5_CODE = '''host = lib.read_host_txt(lib.V2_DIR / "2026-10-15_host-cpu-mtx.txt")
if mtx.empty or host.empty:
    pending("real-matrix runs and the host baseline (2026-10-15_host-cpu-mtx.txt)")
else:
    rows = []
    for (name, pes), g in ok(mtx).groupby(["matrix", "num_pes"]):
        h = host[host.matrix == name].set_index("method").steady_s
        win = lib.med(g.window_s)
        rows.append(dict(matrix=name, PEs=f"{pes:,}", NNZ=int(g.total_nnz.iloc[0]), runs=len(g), window=lib.fmt_s(win),
                         **{"host best": lib.fmt_s(h.min()), "scipy 1 thread": lib.fmt_s(h["scipy 1 thread"]),
                            "host / wafer (best)": round(h.min() / win, 2), "host / wafer (scipy 1)": round(h["scipy 1 thread"] / win, 2)}))
    table = pd.DataFrame(rows).sort_values(["NNZ", "PEs"])
    display(table)
    best = table.sort_values("PEs").groupby("matrix").last()                # the largest grid each matrix ran on
    lines = [f"- **{m}** ({r.NNZ:,} NNZ, {r.PEs} PEs): window {r.window}, host best {r['host best']}: "
             + (f"wafer **{r['host / wafer (best)']:.1f}x faster**" if r["host / wafer (best)"] >= 1.15 else
                f"wafer **{1 / r['host / wafer (best)']:.1f}x slower**" if r["host / wafer (best)"] <= 0.87 else "level")
             + f" ({r['host / wafer (scipy 1)']:.1f}x against one core)" for m, r in best.sort_values("NNZ").iterrows()]
    hr = lib.read_host_txt(lib.V2_DIR / "2026-10-15_host-cpu-random.txt")
    if not hr.empty:
        for n, label, key in (("n=114,000", "65K size", 65_536), ("n=414,000", "full-wafer size", lib.FULL_WAFER)):
            r = hr[(hr.matrix == n) & (hr.method == "numba 32 threads")]
            if len(r):
                lines.append(f"- host numba, 32 threads, {label}: median {lib.fmt_s(r.median_s.iloc[0])} "
                             f"(min {lib.fmt_s(r.min_s.iloc[0])}, max {lib.fmt_s(r.max_s.iloc[0])}) in this run, against the "
                             f"{lib.fmt_s(lib.HOST_CPU_S[key])} the other cells use.")
    display(Markdown("### What that means\\n" + "\\n".join(lines) + """
- The wafer's window is a floor plus NNZ-bound work, so it loses on the tiny matrices and wins when the matrix fills the PEs it runs on;
  `parabolic_fem` on the full wafer is the row to read for a real matrix that fills it, and the random 171M-NNZ matrix (section 1) is the
  same effect at the largest size.
- Host numbers are the steady state of a loop of calls; the first call after a pause also pays thread launch (see the `per call` column in
  `2026-10-15_host-cpu-mtx.txt`), and 32-thread numba is slower than one thread below a few thousand NNZ.

### What the vendor says
Nothing on SpMV time per call."""))
'''

NEXT_MD = """---
## 6. What is next

- **v3.0: conjugate gradient on the resident SpMV**, in this repository. The SpMV is ready; the iteration is not.
- **On-wafer y to x.** The kernel returns y distributed by PE row but needs x distributed by PE column. Until that transpose runs on the
  wafer, every iteration goes through the host (the "each call via the host" curve above). The design is in ADR-005; it is the one piece
  of CG that is neither built nor measured.
- **The loaded-program question.** Program load dominates a run (section 4). Can a loaded program stay on the wafer between jobs, or can
  load be cut? A question for the appliance team, and the cheapest way to move the break-even.
- **Real matrices.** `run.py --mtx-file` sizes a run from a MatrixMarket file and refuses it before compiling if it does not fit. With the block
  tiling, only balanced matrices fit (`docs/results.md`, "Supported matrices"); a cyclic or permuted distribution for skewed ones is planned.
- **Data in this release:** RC1 (commit `4739dc4`) hardware runs, a simulator smoke from `580e41f`, and the host baselines, all under
  `cluster/results/v2.0-data/`; each cell reads its CSV from there."""
