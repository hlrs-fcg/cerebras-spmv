#!/usr/bin/env python3
"""Read the H2D transfer experiments and say which hypothesis survives.

Three questions, in the order they have to be answered:

1. SHAPE OR LINK.  'column' mode issues W one-column-wide memcpy_h2d calls, and
   a one-column transfer cannot be split across I/O channels, so extra channels
   only add routing overhead.  'full' sends the same stream as one padded WxH
   rectangle, which the hardware can spread over channels.  If 'full' improves
   with channels and 'column' does not, the ceiling was shape.

2. MEMORY ORDER.  https://sdk.cerebras.ai/tensor-streaming states column-major
   delivers better bandwidth than row-major, and Cerebras' own bandwidth
   benchmark uses COL_MAJOR for the transfer it measures.  Holding mode and
   channels fixed isolates it.

3. LINK OR KERNEL.  The decisive one.  Our streaming transfer runs a kernel task
   per word (header decode + FMAC in stream_word), so its rate is link PLUS
   kernel cost.  The one-time setup transfer (x_vec, col_indices, row_ptrs) goes
   streaming=False -- copy mode, DMA straight to PE memory, no compute-engine
   involvement -- so it measures the link alone.  Cerebras' bandwidth benchmark
   measures exclusively in that mode.  The ratio between the two is the kernel's
   share, and it decides whether to optimise the transfer or the encoding.

TIMING BASIS.  A blocking launch / memcpy_h2d can return before the device is
done, so compute_time_hw_avg and setup_h2d_time are host-return LOWER bounds
and rates built on them are UPPER bounds.  Rows since the completion barrier
carry stream_time_corrected / setup_h2d_time_corrected; those are used where
present, and every rate on a host-return time is marked 'LB' in the 'basis'
column.  Conclusions drawn from LB rows alone (e.g. 4040625's "consumption-
bound") need re-measuring on the corrected columns.

Rates are reported on three bases because they answer different questions:
  raw    bytes actually transferred / time   (what the link carried)
  useful column-equivalent bytes / time      (cross-mode, strips 'full' padding)
  copy   setup bytes / setup time            (the link without kernel cost)
"""

import argparse
import csv
import sys
from pathlib import Path

# The channels=1 / column / row-major point recorded before these experiments
# existed, under different code.  Reproducing it is the comparability check.
HISTORICAL_BASELINE_MS = 39.398


def num(value, default=0.0) -> float:
    """CSV scalars are strings, and legacy rows wrote ints as '0.0'."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stream_time(row: dict) -> tuple[float, bool]:
    """(seconds, corrected?) for one pass: stream_time_corrected when the row
    has it, else the host-return compute_time_hw_avg (a lower bound)."""
    corr = num(row.get("stream_time_corrected"))
    if corr > 0:
        return corr, True
    return num(row.get("compute_time_hw_avg")), False


def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("exit_reason") == "SUCCESS"]
    rows = [r for r in rows if num(r.get("h2d_bytes_total")) > 0
            and stream_time(r)[0] > 0]
    if not rows:
        sys.exit(f"no rows in {path} carry both h2d_bytes_total and a timing")
    return rows


def key(row: dict) -> tuple[str, str, int]:
    return (row.get("h2d_mode", "column"),
            row.get("memcpy_order", "row"),
            int(num(row.get("channels"), 1)))


def copy_rate(row: dict):
    """Copy-mode link rate from the setup H2D, GB/s, or None on legacy rows.
    Uses setup_h2d_time_corrected when present, else the host-return time."""
    b = num(row.get("setup_h2d_bytes"))
    t = num(row.get("setup_h2d_time_corrected")) or num(row.get("setup_h2d_time"))
    return b / t / 1e9 if b > 0 and t > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="sweep results CSV")
    args = ap.parse_args()

    rows = load(args.csv)
    by = {key(r): r for r in rows}
    orders = sorted({r.get("memcpy_order", "row") for r in rows})
    channels = sorted({int(num(r.get("channels"), 1)) for r in rows})

    # Useful bytes = what 'column' would have sent for the same matrix, so modes
    # compare without 'full' being flattered for its padding.
    col_bytes = {}
    for (mode, _o, _c), r in by.items():
        if mode == "column":
            col_bytes[int(num(r["num_pes"], 1))] = num(r["h2d_bytes_total"])

    r0 = rows[0]
    print(f"\n{args.csv}")
    print(f"{int(num(r0['num_pes'], 1)):,} PEs "
          f"({int(num(r0['grid_width'], 1))}x{int(num(r0['grid_height'], 1))})  "
          f"{int(num(r0['matrix_rows'])):,}^2 @ d={r0['density']}  "
          f"NNZ={int(num(r0['total_nnz'])):,}  commit={r0['git_commit']}\n")

    hdr = (f"{'mode':8s} {'ord':>4s} {'ch':>3s} {'time ms':>9s} {'MB sent':>9s} "
           f"{'raw':>7s} {'useful':>7s} {'copy':>7s} {'x copy':>7s} {'rel err':>10s} "
           f"{'basis':>5s}")
    print(hdr)
    print("-" * len(hdr))

    table = {}
    any_lb = False
    for mode in ("column", "full"):
        for order in orders:
            for ch in channels:
                r = by.get((mode, order, ch))
                if r is None:
                    continue
                t, corrected = stream_time(r)
                any_lb = any_lb or not corrected
                sent = num(r["h2d_bytes_total"])
                useful = col_bytes.get(int(num(r["num_pes"], 1)), sent)
                raw, use = sent / t / 1e9, useful / t / 1e9
                cp = copy_rate(r)
                table[(mode, order, ch)] = (t, raw, use, cp)
                print(f"{mode:8s} {order:>4s} {ch:>3d} {t * 1e3:>9.1f} "
                      f"{sent / 1e6:>9.1f} {raw:>7.2f} {use:>7.2f} "
                      f"{(f'{cp:.2f}' if cp else '-'):>7s} "
                      f"{(f'{raw / cp:.2f}' if cp else '-'):>7s} "
                      f"{num(r['relative_error']):>10.2e} "
                      f"{'corr' if corrected else 'LB':>5s}")

    if any_lb:
        print("\nLB = host-return time (no completion barrier): a LOWER bound on the "
              "pass, so its rates are UPPER bounds.")

    missing = [k for k in [(m, o, c) for m in ("column", "full")
                           for o in orders for c in channels] if k not in table]
    if missing:
        print(f"\nnot measured: {', '.join('/'.join(map(str, k)) for k in missing)}")

    # --- comparability ------------------------------------------------------
    base = table.get(("column", "row", 1))
    if base:
        drift = 100 * (base[0] * 1e3 - HISTORICAL_BASELINE_MS) / HISTORICAL_BASELINE_MS
        verdict = "comparable" if abs(drift) < 10 else "NOT comparable"
        print(f"\ncolumn/row/1 vs recorded {HISTORICAL_BASELINE_MS} ms: "
              f"{base[0] * 1e3:.1f} ms ({drift:+.1f}%) -- {verdict}")

    # --- 1. channels, within each mode and order ----------------------------
    lines = []
    for mode in ("column", "full"):
        for order in orders:
            pts = sorted((ch, v) for (m, o, ch), v in table.items()
                         if m == mode and o == order)
            if len(pts) < 2:
                continue
            (lo_ch, lo), (hi_ch, hi) = pts[0], pts[-1]
            d = 100 * (hi[2] - lo[2]) / lo[2]
            arrow = "improves" if d > 5 else ("degrades" if d < -5 else "flat")
            lines.append(f"  {mode:8s} {order}-major  ch{lo_ch} -> ch{hi_ch}: "
                         f"{lo[2]:.2f} -> {hi[2]:.2f} GB/s  ({d:+.1f}%, {arrow})")
    if lines:
        print("\n1. channel scaling (useful GB/s):")
        print("\n".join(lines))

    # --- 2. memory order, at fixed mode and channels ------------------------
    if "row" in orders and "col" in orders:
        lines = []
        for mode in ("column", "full"):
            for ch in channels:
                a, b = table.get((mode, "row", ch)), table.get((mode, "col", ch))
                if not (a and b):
                    continue
                d = 100 * (b[1] - a[1]) / a[1]
                v = ("col-major wins" if d > 5 else
                     "row-major wins" if d < -5 else "no difference")
                lines.append(f"  {mode:8s} ch{ch}: row {a[1]:.2f} -> col {b[1]:.2f} "
                             f"GB/s  ({d:+.1f}%, {v})")
        if lines:
            print("\n2. memory order at fixed mode and channels (raw GB/s):")
            print("\n".join(lines))

    # --- 3. link vs kernel --------------------------------------------------
    measured = [(k, v) for k, v in table.items() if v[3]]
    if measured:
        best_k, best_v = max(measured, key=lambda kv: kv[1][1])
        cp = best_v[3]
        frac = best_v[1] / cp
        print("\n3. copy mode (streaming=False) vs streaming -- link vs kernel:")
        print(f"  copy-mode link rate : {cp:.2f} GB/s")
        print(f"  best streaming rate : {best_v[1]:.2f} GB/s "
              f"({'/'.join(map(str, best_k))})")
        print(f"  streaming reaches   : {100 * frac:.0f}% of copy mode")

        # The ratio alone is a TRAP.  What discriminates is how each mode
        # RESPONDS TO CHANNELS: a link-bound transfer must speed up when the
        # link is widened.  If streaming stays flat across channel counts while
        # copy mode scales, streaming is not limited by the link at all, however
        # close the two rates happen to sit.
        def scaling(sel):
            pts = sorted((ch, v) for (m, o, ch), v in table.items() if sel(m, o, ch))
            if len(pts) < 2 or not pts[0][1][3]:
                return None
            return pts
        stream_pts = scaling(lambda m, o, c: m == "full" and o == "row")
        stream_delta = copy_delta = None
        if stream_pts and len(stream_pts) >= 2:
            lo, hi = stream_pts[0][1], stream_pts[-1][1]
            stream_delta = 100 * (hi[1] - lo[1]) / lo[1]
            if lo[3] and hi[3]:
                copy_delta = 100 * (hi[3] - lo[3]) / lo[3]

        if stream_delta is not None and copy_delta is not None:
            print(f"\n  response to channels: streaming {stream_delta:+.1f}%, "
                  f"copy {copy_delta:+.1f}%")
            if copy_delta > 20 and stream_delta < 10:
                print("\n  => CONSUMPTION-BOUND, not link-bound. Copy mode speeds up")
                print("     when the link widens; streaming does not, so the link is")
                print("     not what limits streaming -- the per-word kernel task is")
                print("     (stream_word: header decode + FMAC). Note the raw ratio")
                print("     above looks 'close to the link' and is misleading here.")
                print("     Widening the interface buys streaming nothing; removing")
                print("     words does. ADR-003's val-only encoding is the lever.")
            elif frac < 0.7:
                print("\n  => KERNEL-BOUND on the ratio as well as the scaling.")
            else:
                print("\n  => LINK-BOUND. Streaming tracks copy mode as the link widens.")
        elif frac < 0.7:
            print("\n  => KERNEL-BOUND. The per-word task dominates, not the link.")
        else:
            print("\n  => inconclusive: need >=2 channel counts with copy-mode timing")
            print("     to tell a link ceiling from a consumption ceiling.")
    else:
        print("\n3. copy mode: not measured on these rows "
              "(setup_h2d_bytes/time absent -- pre-dates the column)")

    # --- shape verdict ------------------------------------------------------
    full_pts = sorted(((o, ch), v) for (m, o, ch), v in table.items() if m == "full")
    if len(full_pts) >= 2:
        d = 100 * (full_pts[-1][1][2] - full_pts[0][1][2]) / full_pts[0][1][2]
        print()
        if d > 5:
            print("=> SHAPE, not link: 'full' gains with channels, so the original")
            print("   ~1.3-1.5 GB/s ceiling was the one-column transfer shape.")
        else:
            print("=> 'full' does not gain with channels either; the ceiling is not")
            print("   transfer shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
