"""Shared loaders, filters and plot helpers for the analysis notebooks.

Data is read from the checkout's results directory (`cluster/results/data`), never copied. Override the location with ``SPMV_DATA``. Every loader returns an
empty frame when nothing exists yet, so a notebook runs before its data arrives.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("SPMV_DATA", Path(__file__).resolve().parent.parent / "cluster" / "results" / "data")).expanduser()
SPMV_PES = 65_536


class Excluded(dict):
    """Counts of rows dropped, by reason, for the Result subsection."""

    def __str__(self) -> str:
        if not any(self.values()):
            return "excluded: none"
        return "excluded: " + ", ".join(f"{v} {k}" for k, v in self.items() if v)


# ---------------------------------------------------------------- SpMV
SPMV_FILES = ("2026-10-07_barrier-remeasure.csv", "2026-10-07_s1-column-matrix.csv")


def load_spmv(names=SPMV_FILES) -> pd.DataFrame:
    """Concatenate the SpMV result CSVs that exist; ``source`` names the file."""
    frames = []
    for name in names:
        f = DATA / name
        if f.exists():
            df = pd.read_csv(f)
            df["source"] = name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def valid_spmv(df: pd.DataFrame, pes: int = SPMV_PES) -> tuple[pd.DataFrame, Excluded]:
    """Barrier-verified rows only: validated, SUCCESS, corrected time present, ``pes`` PEs.

    The 10-07 CSV has no ``system`` column, so a bad system cannot be filtered here: the only
    known bad-system row was cancelled and never written.
    """
    ex = Excluded()
    if df.empty:
        return df, ex
    small = df["num_pes"] != pes
    failed = (~df["validation_passed"].astype(bool) | df["exit_reason"].ne("SUCCESS")) & ~small
    nocorr = (df["stream_time_corrected"].fillna(0) <= 0) & ~small & ~failed
    ex[f"not {pes} PEs (smokes)"] = int(small.sum())
    ex["failed validation / not SUCCESS"] = int(failed.sum())
    ex["no corrected time (pre-barrier)"] = int(nocorr.sum())
    out = df[~(small | failed | nocorr)].copy()
    out["chunk"] = out["encoding"] + "/" + out["h2d_mode"]
    return out.reset_index(drop=True), ex


# ---------------------------------------------------------------- statistics
def summarize(df: pd.DataFrame, by, col: str, scale: float = 1.0, digits: int = 3) -> pd.DataFrame:
    """n / min / median / max of ``col`` per group; never just the best run."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(by, observed=True)[col]
    out = pd.DataFrame({"n": g.size(), "min": g.min() * scale, "median": g.median() * scale,
                        "max": g.max() * scale})
    out[["min", "median", "max"]] = out[["min", "median", "max"]].round(digits)
    return out.reset_index()


def no_data(what: str) -> None:
    from IPython.display import Markdown, display
    display(Markdown(f"**No data yet** for {what}. Expected under `{DATA}`. "
                     "Re-run this notebook when it arrives."))


# ---------------------------------------------------------------- plotting
# Presentation rules (orchestrator, 2026-10-08): SVG output, solid white background, near-black ink,
# at most 3 hues per axes in a fixed order (validated colour-blind-safe), and never colour alone:
# every series has its own marker and line style, a direct end label and a legend.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
HUES = [BLUE, ORANGE, GREEN]
MARKERS = ["o", "s", "^"]
LINES = ["-", "--", ":"]
INK, GRID = "#1a1a1a", "#d9d9d9"
PANEL = (11, 5.2)  # inches per panel

# Same entity = same colour/marker/line in every plot and notebook.
_ENTITY = {
    "mode": ["copy", "stream"],                 # copy blue, stream orange
    "order": ["row", "col"],                    # row blue, col orange
    "encoding": ["header", "valtask", "val"],
    "channels": [1, 8, 16],
    "nonblock": ["all", "serial", "block"],
    "h2d_mode": ["full", "column"],
    "basis": ["tail", "corrected"],
}


def style(kind: str, key) -> dict:
    """Colour, marker and line style for one entity (``kind`` in ``_ENTITY``) or, for a free index, ``kind='index'``."""
    i = key if kind == "index" else _ENTITY[kind].index(key)
    return {"color": HUES[i % 3], "marker": MARKERS[i % 3], "ls": LINES[i % 3]}


def setup_plots() -> None:
    import matplotlib as mpl
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("svg")
    except ImportError:  # plain Python, no notebook
        pass
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.transparent": False, "savefig.dpi": 200, "figure.figsize": PANEL,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "axes.grid": True, "grid.color": GRID, "grid.alpha": 1.0, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": GRID, "legend.framealpha": 1.0,
        "lines.linewidth": 2.2, "lines.markersize": 7, "svg.fonttype": "path",
    })


def panels(n: int, sharey: bool = True, sharex: bool = True, height: float | None = None):
    """``n`` panels stacked vertically, ~(11, 5.2) each (``height`` overrides), so type stays legible in a notebook."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(n, 1, figsize=(PANEL[0], (height or PANEL[1]) * n), sharey=sharey, sharex=sharex, squeeze=False)
    return fig, list(axs[:, 0])


def _plain(v, _=None):
    return f"{v:g}"


def _bytes(v, _=None):
    for unit, f in (("GB", 1e9), ("MB", 1e6), ("KB", 1024)):
        if v >= f:
            return f"{v / f:g} {unit}"
    return f"{v:g} B"


def plain_log(ax, axis: str = "y", kind: str = "plain", ticks=None) -> None:
    """Log axis with plain tick labels (0.3, 1, 3, 10; or 1 KB, 4 KB) instead of 10^x."""
    import matplotlib.ticker as mt
    ax_ = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    if ticks is None:
        ticks = [m * 10.0 ** e for e in range(-4, 6) for m in (1, 3)]
        ticks = [t for t in ticks if lo / 1.3 <= t <= hi * 1.3]
    if kind == "bytes":  # drop a tick that sits within 30% of the next one (36 KB vs 40 KB), keeping the larger
        ticks = sorted(ticks)
        ticks = [t for i, t in enumerate(ticks) if i == len(ticks) - 1 or ticks[i + 1] / t > 1.3]
    ax_.set_major_locator(mt.FixedLocator(ticks))
    ax_.set_major_formatter(mt.FuncFormatter(_bytes if kind == "bytes" else _plain))
    ax_.set_minor_locator(mt.NullLocator())


def draw_series(ax, df: pd.DataFrame, x: str, y: str, kind: str, key, label: str | None = None,
                points: bool = True, endlabel: bool = True, label_text: str | None = None) -> None:
    """Points for every run (size ~40, alpha 0.9, thin white edge) and a median line on top."""
    st = style(kind, key)
    label = str(key) if label is None else label
    if points:
        ax.scatter(df[x], df[y], s=40, color=st["color"], marker=st["marker"], alpha=0.9,
                   edgecolors="white", linewidths=0.6, zorder=2)
    med = df.groupby(x)[y].median().sort_index()
    ax.plot(med.index, med.values, color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.2, ms=7,
            mec="white", mew=0.6, label=label, zorder=3)
    if endlabel and len(med):
        ax.annotate(label_text or label, (med.index[-1], med.values[-1]), xytext=(7, 0), textcoords="offset points",
                    color=st["color"], fontsize=11, va="center", annotation_clip=False)


def strip_plot(ax, df: pd.DataFrame, x: str, y: str, hue: str, kind: str, labels: dict | None = None,
               endlabel: bool = True) -> None:
    """One series per value of ``hue`` (<=3), styled by ``kind``; every run shown, line = median."""
    for key, sub in df.groupby(hue, observed=True):
        draw_series(ax, sub, x, y, kind, key, label=(labels or {}).get(key), endlabel=endlabel)


def finish(ax, xlabel: str, ylabel: str, title: str | None = None, legend: str | None = None, logx=False, logy=False,
           xkind: str = "plain", xticks=None, yticks=None) -> None:
    """Axis labels with units, plain log tick labels, a legend, and room on the right for end labels."""
    if logx:
        ax.margins(x=0.08)
    if logy:
        ax.margins(y=0.08)
    if logx:
        plain_log(ax, "x", xkind, xticks)
    elif xticks is not None:
        ax.set_xticks(xticks, [_bytes(t) if xkind == "bytes" else _plain(t) for t in xticks])
    if logy:
        plain_log(ax, "y", "plain", yticks)
    ax.tick_params(labelbottom=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title=legend, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, borderaxespad=0)
    ax.figure.tight_layout()


# ---------------------------------------------------------------- vendor / example claims
# Dark red is reserved for claims and never used for a data series. One source of truth, cited like the Vendor sections.
CLAIM = "#c0392b"
GBPS_PER_CHANNEL = 7 / 8   # "around 7Gbps" per channel = 0.875 GB/s
LINK_GBPS = 12.5           # one 100 GbE link
AGGREGATE_GBPS = 150.0     # 1.2 Tb/s system I/O, not drawn


def vendor_claims(channels: int) -> list[tuple[float, str, str]]:
    """(GB/s, short label, source) for the numeric claims that apply at ``channels`` I/O channels."""
    ch = int(channels)
    v = GBPS_PER_CHANNEL * ch
    return [
        (v, f"7 Gbps/ch × {ch} = {v:.3g} GB/s [BW]:18",
         "csl-examples benchmarks/bandwidth-test/run.appliance.py:18–19, a comment in example code (SDK 1.4.0 examples)"),
        (LINK_GBPS, "one 100 GbE link = 12.5 GB/s [SYS] unpinned",
         "third-party CS-3 spec summaries; docs/vendor-claims-vs-measurements.md §5; not pinned to a CS-3 datasheet"),
    ]


def _include_y(ax, y: float) -> None:
    """Make the autoscaled y range include ``y`` without freezing the limits (shared axes keep autoscaling)."""
    ax.update_datalim([(ax.dataLim.x0, y)])
    ax.autoscale_view(scalex=False)


def _add_claim_label(ax, v: float, label: str) -> None:
    ax._claims = getattr(ax, "_claims", []) + [(v, label)]


def draw_claims(ax, channels, link: bool = True) -> None:
    """Red dotted reference lines behind the data; widens the y range to include them.

    ``channels``: an int (panel for that channel count) or an iterable (shared panel: one per-channel line each).
    Call after the data is drawn. Labels go in the right margin when ``place_claims(fig)`` runs, just before showing.
    """
    chs = [channels] if isinstance(channels, (int, np.integer)) else list(channels)
    lines = [vendor_claims(c)[0] for c in chs]
    if link:
        lines.append(vendor_claims(chs[0])[1])
    top = 0.0
    for i, (v, label, _src) in enumerate(lines):
        ax.axhline(v, color=CLAIM, ls=":", lw=1.6, zorder=1, label="vendor / example claim (red dotted)" if i == 0 else None)
        _add_claim_label(ax, v, label)
        top = max(top, v * 1.7)
    _include_y(ax, top)


def draw_claims_vs_channels(ax, link: bool = True) -> None:
    """For a plot with I/O channels on x: the per-channel claim as a line through (ch, 0.875·ch), plus the link line."""
    xs = np.array([1, 8, 16])
    ax.plot(xs, GBPS_PER_CHANNEL * xs, color=CLAIM, ls=":", lw=1.6, zorder=1, label="vendor / example claim (red dotted)")
    _add_claim_label(ax, GBPS_PER_CHANNEL * 16, "7 Gbps/ch × ch (14 GB/s at 16 ch) [BW]:18")
    if link:
        ax.axhline(LINK_GBPS, color=CLAIM, ls=":", lw=1.6, zorder=1)
        _add_claim_label(ax, LINK_GBPS, vendor_claims(1)[1][1])
    _include_y(ax, GBPS_PER_CHANNEL * 16 * 1.7)


def place_claims(fig, fontsize: float = 10) -> None:
    """Write the claim labels in the right margin of every axes, nudged apart vertically (>= 1.2 line heights),
    with a thin leader line back to the true y. Call once, after all panels are finished and before showing."""
    fig.tight_layout()
    for ax in fig.axes:
        claims = sorted(getattr(ax, "_claims", []))
        if not claims:
            continue
        ax.get_ylim()  # settle autoscale
        gap = 1.2 * fontsize * fig.dpi / 72
        ys = [ax.transData.transform((0, v))[1] for v, _ in claims]
        adj = list(ys)
        for k in range(1, len(adj)):
            adj[k] = max(adj[k], adj[k - 1] + gap)
        shift = (sum(adj) - sum(ys)) / len(adj) / 2          # re-centre the pushed block a little
        adj = [a - shift if a > y else a for a, y in zip(adj, ys)]
        for k in range(1, len(adj)):
            adj[k] = max(adj[k], adj[k - 1] + gap)
        for (v, label), y0, y1 in zip(claims, ys, adj):
            y_data = ax.transData.inverted().transform((0, y1))[1]
            tr = ax.get_yaxis_transform()
            ax.plot([1.0, 1.012], [v, y_data], transform=tr, color=CLAIM, lw=0.8, clip_on=False, zorder=1)
            ax.text(1.015, y_data, label, transform=tr, ha="left", va="center", fontsize=fontsize, color=CLAIM, clip_on=False)
    fig.tight_layout()


def aggregate_note(best_gbps: float):
    """Markdown line printed under a bandwidth plot (the 150 GB/s aggregate is off the scale)."""
    from IPython.display import Markdown
    return Markdown(f"*CS-3 aggregate 150 GB/s (1.2 Tb/s) not shown [SYS], unpinned third-party figure; "
                    f"best measured here = {best_gbps:.2f} GB/s = {100 * best_gbps / AGGREGATE_GBPS:.1f}% of it.*")


# ---------------------------------------------------------------- v1.0 vs v2.0 (analysis/v1_vs_v2.ipynb)
ROOT = Path(__file__).resolve().parent.parent
V1_DIR = ROOT / "cluster" / "results" / "v1.0-data"
V2_DIR = ROOT / "cluster" / "results" / "v2.0-data"
DEV_DIR = ROOT / "analysis" / "data" / "dev"
CLOCK_HZ = 0.85e9                       # WSE-3 tsc rate used everywhere in the repo (docs/synced-timing.md)
# Host CPU, 32-thread numba CSR on the same matrices (docs/results.md, "Against the host CPU"): the only typed figures.
HOST_CPU_S = {65_536: 3.48e-3, 857_010: 61.70e-3}
FULL_WAFER = 857_010


def read_dir(directory: Path, pattern: str = "*.csv", skip: tuple = ("_sim_",)) -> pd.DataFrame:
    """All hardware CSVs of a directory, concatenated, with ``source`` (file name); empty when there are none yet."""
    frames = []
    for f in sorted(Path(directory).glob(pattern)):
        if any(s in f.name for s in skip):
            continue
        frames.append(pd.read_csv(f).copy().assign(source=f.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_v2() -> pd.DataFrame:
    """v2.0 rows (release candidate, ``origin == "RC1"``) and the development runs they rest on (``"dev"``)."""
    rc, dev = read_dir(V2_DIR), read_dir(DEV_DIR)
    parts = [x.assign(origin=o) for x, o in ((rc, "RC1"), (dev, "dev")) if not x.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).copy()
    df["matrix"] = df["source"].str.extract(r"_mtx_(.+?)_(?:16|65k|full)")[0].fillna("random")   # real matrix, or the generator
    # min over passes == median in every run (the window is deterministic to a few cycles)
    # the window is deterministic to a few cycles, with an occasional outlier pass: take the mean of a run whose passes agree
    # (std <= 5 cycles, which is what the docs quote) and the minimum (= the median pass) of a run that has outliers
    cyc = df["wafer_cycles_avg"].where(df["wafer_cycles_std"] <= 5, df["wafer_cycles_min"])
    return df.assign(window_cycles=cyc, window_s=cyc / CLOCK_HZ)


_UNIT = {"us": 1e-6, "ms": 1e-3}


def read_host_txt(path: Path) -> pd.DataFrame:
    """The host-CPU baseline outputs (bench/host_spmv_baseline.py: ``--mtx`` mode and the random sizes) as one row per
    (matrix, method): ``per_call_s`` / ``steady_s`` (``--mtx`` only), ``median_s`` / ``min_s`` / ``max_s`` (random sizes)."""
    import re
    rows, cur = [], None
    if not Path(path).exists():
        return pd.DataFrame()
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+\.mtx) \d+x\d+ ", line) or re.match(r"(n=[\d,]+) nnz=", line)
        if m:
            cur = m.group(1).replace(".mtx", "")
            continue
        m = re.match(r"\s+(numpy bincount|scipy CSR|numba CSR fp32, (\d+) threads)\S*.*?"
                     r"(?:per call\s+([\d.]+) (us|ms) \(min\s+([\d.]+) (?:us|ms), max\s+([\d.]+) (?:us|ms)\)\s+steady\s+([\d.]+) (us|ms)"
                     r"|median\s+([\d.]+) ms\s+\(min\s+([\d.]+), max\s+([\d.]+)\))", line)
        if m and cur:
            method = ("numba %s threads" % m.group(2)) if m.group(2) else ("numpy reference" if m.group(1).startswith("numpy") else "scipy 1 thread")
            if m.group(3):
                rows.append(dict(matrix=cur, method=method, per_call_s=float(m.group(3)) * _UNIT[m.group(4)],
                                 steady_s=float(m.group(7)) * _UNIT[m.group(8)]))
            else:
                rows.append(dict(matrix=cur, method=method, median_s=float(m.group(9)) * 1e-3, min_s=float(m.group(10)) * 1e-3,
                                 max_s=float(m.group(11)) * 1e-3))
    return pd.DataFrame(rows)


def good(df: pd.DataFrame) -> pd.DataFrame:
    """Validated, successful rows (null twins are 'SUCCESS_NULL_SPMV')."""
    if df.empty:
        return df
    return df[df["validation_passed"].astype(bool) & df["exit_reason"].astype(str).str.startswith("SUCCESS")]


def load_v1() -> pd.DataFrame:
    """The v1.0 (ISC26) rows used here: weak scaling (1000 NNZ/PE) and the largest strong-scaling grid.

    ``call_s`` = ``total_run_time``: H2D + compute + D2H + gather as the v1.0 host measured them (host-return times, lower
    bounds, docs/host-timing-barrier.md). The per-call partitioning (``distribute_time`` + ``prepare_matched_pairs_time``) is
    separate. ``compute_hw_cycles_max`` is the per-PE tsc span, so cycles / 0.85 GHz is seconds; the ``compute_time_hw_*``
    columns are in milliseconds despite their name (cluster/results/v1.0-data/README.md)."""
    frames = []
    for name, tag in (("2026-01-19_comprehensive-03-weak-scaling.csv", "weak, 1000 NNZ/PE"),
                      ("2026-01-20_strong-scaling-large.csv", "strong, 40M NNZ")):
        f = V1_DIR / name
        if f.exists():
            frames.append(pd.read_csv(f).copy().assign(scaling=tag, source=name))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["validation_passed"].astype(bool)]
    return df.assign(call_s=df["total_run_time"], compute_s=df["compute_hw_cycles_max"] / CLOCK_HZ,
                     partition_s=df["distribute_time"] + df["prepare_matched_pairs_time"])


def med(series) -> float:
    return float(np.median(np.asarray(series, dtype=float))) if len(series) else float("nan")


def fmt_s(t: float) -> str:
    """A time with a unit that fits its size (ns / µs / ms / s)."""
    if not np.isfinite(t):
        return "n/a"
    for unit, f in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if t >= f:
            return f"{t / f:.3g} {unit}"
    return f"{t / 1e-9:.3g} ns"
