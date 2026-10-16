"""Cycle and barrier timing: per-PE tsc spans, the synced on-wafer window, completion barriers.
Pure numpy (plus the runner passed in); staged flat next to worker_runtime.py."""

import time

import numpy as np


# --------------------------------------------------------------------------
# Per-PE cycle-timestamp reconstruction
# --------------------------------------------------------------------------
# A 48-bit tsc count is stored as three little-endian u16 words.  These two
# helpers are a direct transcription of the reference benchmarks shipped with
# the Cerebras SDK examples:
#   csl-examples/benchmarks/7pt-stencil-spmv/run.py:92
#   csl-examples/benchmarks/single-tile-matvec/run.py:49
# Those examples define their own make_u48 / sub_ts and deliberately do NOT use
# cerebras.sdk.sdk_utils.calculate_cycles — its contract differs (an earlier
# cross-check against it produced a value ~93x off), so nothing here depends on
# that SDK helper.

def make_u48(words: np.ndarray) -> np.ndarray:
    """words[..., 0:3] little-endian u16 -> 48-bit integer (int64)."""
    w = np.asarray(words, dtype=np.int64)
    return w[..., 0] + (w[..., 1] << 16) + (w[..., 2] << 32)


def sub_ts(words: np.ndarray) -> np.ndarray:
    """words[..., 0:6] = [start_lo, start_mid, start_hi, end_lo, end_mid, end_hi]
    -> (end - start) in cycles.  Matches csl-examples' sub_ts()."""
    return make_u48(words[..., 3:6]) - make_u48(words[..., 0:3])


def unpack_tsc_cycles(tsc_pack: np.ndarray, H: int, W: int):
    """Reconstruct per-PE (end - start) cycle spans from the packed f32 buffer.

    ``tsc_pack`` is ``H*W*3`` float32, D2H'd from device symbol ``tsc_pack``.
    Each PE's ``chain_done`` (src/csl/pe.csl) packed two 48-bit tsc counts —
    ``start`` stamped on the first streamed word, ``end`` when its EW chain
    finished — into 3 f32 with this exact bit layout:

        pack[0] = start_word[1] << 16 | start_word[0]
        pack[1] = end_word[0]   << 16 | start_word[2]
        pack[2] = end_word[2]   << 16 | end_word[1]

    The host splits each packed word into ``lo = w & 0xFFFF`` / ``hi = w >> 16``
    and feeds the six words to ``sub_ts`` in the order
    ``single-tile-matvec/run.py`` uses.  Bit-for-bit; no float arithmetic.

    Returns ``(pe_cycles, start48)``, both shape ``(H*W,)`` int64.  ``pe_cycles``
    for a PE that never stamped (``start48 == 0``) is meaningless — callers
    filter on ``start48``.
    """
    u = np.ascontiguousarray(tsc_pack, dtype=np.float32).reshape(H, W, 3)
    u = u.view(np.uint32).astype(np.int64)
    lo = u & 0xFFFF
    hi = (u >> 16) & 0xFFFF
    words6 = np.stack(
        [lo[..., 0], hi[..., 0], lo[..., 1],   # start_lo, start_mid, start_hi
         hi[..., 1], lo[..., 2], hi[..., 2]],  # end_lo,   end_mid,   end_hi
        axis=-1,
    )
    start48 = make_u48(words6[..., 0:3]).ravel()
    pe_cycles = sub_ts(words6).ravel()   # == make_u48(end words) - start48
    return pe_cycles, start48


def summarise_tsc_cycles(pe_cycles: np.ndarray, start48: np.ndarray) -> dict:
    """Aggregate per-PE tsc spans, excluding PEs that never stamped.

    A PE that never ran ``get_timestamp(&tsc_start)`` reports ``start48 == 0``
    (also the value a stale / failed D2H leaves).  Those PEs are dropped before
    min/max/mean/std so a partial readback cannot masquerade as "0 cycles".

    Returns a dict with ``n_total``, ``n_stamped``, ``n_zero`` always present,
    ``min``/``max``/``mean``/``std`` present only when ``n_stamped > 0``, plus
    ``reason`` — ``"ok"`` or a human string saying why the stats are absent —
    and small ``sample_*`` slices for logging.  Never raises on shape/values;
    callers log ``reason`` loudly whenever it is not ``"ok"``.
    """
    pe_cycles = np.asarray(pe_cycles).ravel()
    start48 = np.asarray(start48).ravel()
    n_total = int(start48.size)
    stamped = start48 != 0
    n_stamped = int(stamped.sum())
    out = {
        "n_total": n_total,
        "n_stamped": n_stamped,
        "n_zero": n_total - n_stamped,
        "sample_start48": start48[:8].tolist(),
        "sample_cycles": pe_cycles[:8].tolist(),
    }
    if n_total == 0:
        out["reason"] = "tsc buffer was empty (0 PEs)"
        return out
    if n_stamped == 0:
        out["reason"] = (
            "every PE reports start48==0 — tsc never stamped on device, or the "
            "D2H returned an all-zero / unwritten buffer"
        )
        return out
    recorded = pe_cycles[stamped]
    out["min"] = int(recorded.min())
    out["max"] = int(recorded.max())
    out["mean"] = float(recorded.mean())
    out["std"] = float(recorded.std())
    out["reason"] = "ok"
    return out


# --------------------------------------------------------------------------
# Synced on-wafer timing (docs/synced-timing.md)
# --------------------------------------------------------------------------
# csl-examples' bandwidth-test pattern: f_sync once (every PE samples a
# reference clock broadcast from P0.0), then per pass f_tic ... f_toc.  Each
# PE's f_toc packs {tic, toc, ref} into sync_pack[5]:
#   [0..2]  tic / toc, the same bit layout as tsc_pack (start = tic, end = toc)
#   [3]     ref[1] << 16 | ref[0]        (bandwidth-test's time_ref[0])
#   [4]     ref[2]                       (time_ref[1], high half 0)

WSE_CLOCK_HZ = 0.85e9
"""Cycles -> seconds, as every csl-example converts (850 MHz)."""


def unpack_sync_pack(sync_pack: np.ndarray, H: int, W: int):
    """``H*W*5`` float32 (ROW_MAJOR D2H of ``sync_pack``) -> (tic, toc, ref).

    Each is an ``(H, W)`` int64 array of 48-bit tsc counts.  Bit-for-bit, no
    float arithmetic, like unpack_tsc_cycles.
    """
    u = np.ascontiguousarray(sync_pack, dtype=np.float32).reshape(H, W, 5)
    u = u.view(np.uint32).astype(np.int64)
    lo = u & 0xFFFF
    hi = (u >> 16) & 0xFFFF
    tic = make_u48(np.stack([lo[..., 0], hi[..., 0], lo[..., 1]], axis=-1))
    toc = make_u48(np.stack([hi[..., 1], lo[..., 2], hi[..., 2]], axis=-1))
    ref = make_u48(np.stack([lo[..., 3], hi[..., 3], lo[..., 4]], axis=-1))
    return tic, toc, ref


def synced_pass_cycles(tic: np.ndarray, toc: np.ndarray, ref: np.ndarray) -> int:
    """On-wafer cycles for one pass: max(toc) - min(tic) on the common clock.

    Each PE's tsc has its own offset, so raw stamps from different PEs cannot be
    compared.  All PEs stamped ``ref`` when P0.0's sync broadcast reached them,
    which happens ``px + py`` hops after P0.0 (bandwidth-test/run.py: "adjust
    the reference clock by the propagation delay"), so ``ref - (px + py)`` is
    the same global instant on every PE, and shifting by it puts all stamps on
    one clock.  Raises ValueError if any PE never stamped (a zero count),
    rather than reporting a span built from garbage.
    """
    tic, toc, ref = (np.asarray(a, dtype=np.int64) for a in (tic, toc, ref))
    for name, a in (("tic", tic), ("toc", toc), ("ref", ref)):
        n_zero = int((a == 0).sum())
        if n_zero:
            raise ValueError(f"{n_zero}/{a.size} PEs report {name}==0 (never stamped)")
    H, W = ref.shape
    py, px = np.indices((H, W))
    ref_adj = ref - (px + py)
    return int((toc - ref_adj).max() - (tic - ref_adj).min())


def wafer_host_ratio(cycles: int, host_s: float) -> float:
    """Synced on-wafer time over the host time that must contain it.

    ``host_s`` runs from before f_tic is issued to the return of the completion
    barrier D2H after the pass.  Every stamp is taken on the device
    after its command was issued and before that D2H -- queued behind f_toc --
    can complete, so a correct reading is <= 1.  > 1 means the tsc rate or a
    stamp is wrong.  Do NOT compare against compute_time_hw_*: launch(...,
    nonblock=False) can return before the device has finished (the recorded
    val full/ch1 run: 22 ms host vs 104 ms per-PE span, with the remainder
    landing in a 193 ms y D2H), so that bracket is not an upper bound.
    """
    if host_s <= 0:
        return float("inf")
    return cycles / WSE_CLOCK_HZ / host_s


def pass_stats(values) -> dict:
    """min / avg / max / std (population) over per-pass values; zeros if none."""
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"min": 0.0, "avg": 0.0, "max": 0.0, "std": 0.0}
    return {"min": float(v.min()), "avg": float(v.mean()),
            "max": float(v.max()), "std": float(v.std())}


# --------------------------------------------------------------------------
# Completion barriers
# --------------------------------------------------------------------------
# A blocking launch / memcpy_h2d returns when the HOST is done with the call,
# which can be before the DEVICE is: the y D2H after launch("wait_ew",
# nonblock=False) took 1.0 ms in the 10-01 column/8 row but 50 ms after
# full/row/8, 27 ms after full/row/16 and 193 / 371 ms at ch1 -- it was soaking
# up the rest of the pass.  So every host bracket that ends in a blocking call
# is closed instead by a tiny copy-mode D2H queued behind it, which cannot
# complete until the device has executed everything before it, and the D2H's
# own cost is measured by issuing it again on the now-idle device.
# docs/host-timing-barrier.md.

def timed_d2h(runner, sym, w: int, h: int, dtype, order) -> float:
    """One-word-per-PE blocking D2H of ``sym`` over the (0,0)-anchored w x h
    rectangle; returns its wall time in seconds."""
    buf = np.zeros(w * h, dtype=np.float32)
    t = time.perf_counter()
    runner.memcpy_d2h(buf, sym, 0, 0, w, h, 1, streaming=False,
                      data_type=dtype, order=order, nonblock=False)
    return time.perf_counter() - t


def corrected_time(done_s: float, barrier_repeat_s: float) -> float:
    """Bracket closed by a barrier D2H, minus that D2H's own idle-device cost."""
    return done_s - barrier_repeat_s
