"""Round-trip tests for the per-PE cycle-timestamp pack/unpack (pure numpy).

Device side — src/csl/pe.csl, task ``chain_done`` — packs two 48-bit tsc counts
into three f32 words:

    pack[0] = start_word[1] << 16 | start_word[0]
    pack[1] = end_word[0]   << 16 | start_word[2]
    pack[2] = end_word[2]   << 16 | end_word[1]

where a 48-bit count C is stored little-endian as
``(C & 0xFFFF, (C >> 16) & 0xFFFF, (C >> 32) & 0xFFFF)`` (the ``get_timestamp``
layout, tsc_size_words == 3).

Host side — ``worker_runtime.unpack_tsc_cycles`` via ``make_u48`` / ``sub_ts`` —
is a transcription of the reference benchmarks in the Cerebras SDK examples
(``csl-examples/benchmarks/single-tile-matvec/run.py:49`` and
``.../7pt-stencil-spmv/run.py:92``).  These tests pin that the host reverses the
CSL layout exactly, and that it agrees with a literal transcription of the
examples' ``make_u48`` / ``sub_ts`` on random input.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("cluster", "bench"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from worker_runtime import (  # noqa: E402
    make_u48,
    sub_ts,
    summarise_tsc_cycles,
    unpack_tsc_cycles,
)

U48_MAX = (1 << 48) - 1


# --------------------------------------------------------------------------
# Models: the CSL pack, and the examples' make_u48 / sub_ts, both literal
# --------------------------------------------------------------------------

def _count_to_words(c: int) -> tuple[int, int, int]:
    """48-bit int -> (lo, mid, hi) u16 — the get_timestamp storage layout."""
    assert 0 <= c <= U48_MAX, c
    return (c & 0xFFFF, (c >> 16) & 0xFFFF, (c >> 32) & 0xFFFF)


def device_pack(start48: int, end48: int) -> np.ndarray:
    """Literal transcription of src/csl/pe.csl chain_done -> 3 float32.

    The ``@bitcast(f32, ...)`` + f32 D2H is modelled by writing the u32 bytes and
    reading them back as float32 (bit-exact, NaN payloads included).
    """
    s = _count_to_words(start48)
    e = _count_to_words(end48)
    p0 = ((s[1] << 16) | s[0]) & 0xFFFFFFFF
    p1 = ((e[0] << 16) | s[2]) & 0xFFFFFFFF
    p2 = ((e[2] << 16) | e[1]) & 0xFFFFFFFF
    return np.frombuffer(struct.pack("<3I", p0, p1, p2), dtype=np.float32).copy()


def ref_make_u48(words) -> int:
    """csl-examples/.../run.py: words[0] + (words[1] << 16) + (words[2] << 32)."""
    return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)


def ref_sub_ts(words) -> int:
    """csl-examples/.../run.py: make_u48(words[3:]) - make_u48(words[0:3])."""
    return ref_make_u48(words[3:6]) - ref_make_u48(words[0:3])


def ref_six_words_from_pack(pack_f32: np.ndarray) -> np.ndarray:
    """single-tile-matvec/run.py unpack loop: split each packed f32 lo/hi."""
    hexes = np.ascontiguousarray(pack_f32, dtype=np.float32).view(np.uint32)
    w = np.zeros(6, dtype=np.uint64)
    w[0] = hexes[0] & 0xFFFF
    w[1] = (hexes[0] >> 16) & 0xFFFF
    w[2] = hexes[1] & 0xFFFF
    w[3] = (hexes[1] >> 16) & 0xFFFF
    w[4] = hexes[2] & 0xFFFF
    w[5] = (hexes[2] >> 16) & 0xFFFF
    return w


# --------------------------------------------------------------------------
# Single-PE round trip: every 16-bit boundary and sign case
# --------------------------------------------------------------------------

PAIRS = [
    pytest.param(0, 0, id="never-stamped"),
    pytest.param(100, 350, id="small-positive"),
    pytest.param(350, 100, id="end-before-start-negative"),
    pytest.param(0, 12_345, id="zero-start-nonzero-end"),
    pytest.param(0xFFFF, 0x1_0000, id="carry-word0-to-word1"),
    pytest.param(0xFFFF_FFFF, 0x1_0000_0000, id="carry-word1-to-word2-above-2^32"),
    pytest.param(0x1234_5678_9ABC, 0x1234_5678_9ABD, id="bits-in-all-3-words-diff-1"),
    pytest.param(0x0000_FEDC_BA98, 0x0001_2345_6789, id="generic-48bit"),
    pytest.param(U48_MAX, U48_MAX, id="max-48bit-diff-0"),
    pytest.param(U48_MAX - 7, U48_MAX, id="max-48bit-diff-7"),
    pytest.param(0x9ABC, 0x0001_0000_7FC0, id="pack1-is-nan-f32-bitpattern"),
    pytest.param(0x0007_8000_0001, 0x0007_8000_0000, id="high-word-set-tiny-negative"),
]


@pytest.mark.parametrize("start48, end48", PAIRS)
def test_single_pe_round_trip(start48, end48):
    pack = device_pack(start48, end48)
    pe_cycles, start_out = unpack_tsc_cycles(pack, H=1, W=1)

    assert pe_cycles.shape == (1,) and start_out.shape == (1,)
    assert int(start_out[0]) == start48
    assert int(pe_cycles[0]) == end48 - start48        # signed, exact
    # agrees with the examples' sub_ts on the same packed buffer
    assert int(pe_cycles[0]) == ref_sub_ts(ref_six_words_from_pack(pack))


def test_end_before_start_stays_negative_not_wrapped():
    pack = device_pack(1_000_000, 999_000)
    pe_cycles, _ = unpack_tsc_cycles(pack, H=1, W=1)
    assert int(pe_cycles[0]) == -1000
    assert pe_cycles.dtype == np.int64


# --------------------------------------------------------------------------
# A PE that never stamped: start48 == 0, filtered by the caller
# --------------------------------------------------------------------------

def test_never_stamped_is_zero_and_filterable():
    # 4 PEs: two stamped, two never (all-zero pack).
    S = [0, 5_000, 0, 40_000]
    E = [0, 9_000, 0, 41_500]
    buf = np.concatenate([device_pack(s, e) for s, e in zip(S, E)])
    pe_cycles, start48 = unpack_tsc_cycles(buf, H=1, W=4)

    assert list((start48 == 0)) == [True, False, True, False]
    recorded = pe_cycles[start48 != 0]
    assert sorted(recorded.tolist()) == [1_500, 4_000]


# --------------------------------------------------------------------------
# Grid: per-PE triples stay together, row-major PE order in .ravel()
# --------------------------------------------------------------------------

def test_grid_recovers_every_pe_in_row_major_order():
    H, W = 3, 4
    rng = np.random.default_rng(0)
    starts = rng.integers(0, U48_MAX, size=H * W, endpoint=True).astype(object)
    diffs = rng.integers(1, 500_000, size=H * W)
    ends = [int(s) + int(d) for s, d in zip(starts, diffs)]
    ends = [min(e, U48_MAX) for e in ends]

    buf = np.concatenate([device_pack(int(s), int(e)) for s, e in zip(starts, ends)])
    pe_cycles, start48 = unpack_tsc_cycles(buf, H=H, W=W)

    assert pe_cycles.shape == (H * W,)
    for i in range(H * W):
        assert int(start48[i]) == int(starts[i])
        assert int(pe_cycles[i]) == int(ends[i]) - int(starts[i])


# --------------------------------------------------------------------------
# Agreement with the examples' helpers on random input
# --------------------------------------------------------------------------

def test_make_u48_matches_reference_random():
    rng = np.random.default_rng(1)
    w = rng.integers(0, 0x1_0000, size=(4000, 3), dtype=np.int64)
    got = make_u48(w)
    exp = np.array([ref_make_u48(row) for row in w], dtype=np.int64)
    assert np.array_equal(got, exp)


def test_sub_ts_matches_reference_random():
    rng = np.random.default_rng(2)
    w = rng.integers(0, 0x1_0000, size=(4000, 6), dtype=np.int64)
    got = sub_ts(w)
    exp = np.array([ref_sub_ts(row) for row in w], dtype=np.int64)
    assert np.array_equal(got, exp)


def test_unpack_matches_reference_on_random_pairs():
    rng = np.random.default_rng(3)
    starts = rng.integers(0, U48_MAX, size=3000, endpoint=True)
    ends = rng.integers(0, U48_MAX, size=3000, endpoint=True)  # incl. end < start
    for s, e in zip(starts.tolist(), ends.tolist()):
        pack = device_pack(s, e)
        pe_cycles, start48 = unpack_tsc_cycles(pack, H=1, W=1)
        assert int(start48[0]) == s
        assert int(pe_cycles[0]) == e - s
        assert int(pe_cycles[0]) == ref_sub_ts(ref_six_words_from_pack(pack))


# --------------------------------------------------------------------------
# Filtering + aggregation (summarise_tsc_cycles) at realistic PE counts
# --------------------------------------------------------------------------

def device_pack_grid(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Vectorised device_pack over N PEs -> (N*3,) float32, PE-major then word.

    Reshapes to (H, W, 3) row-major, i.e. PE index == py*W + px, matching
    unpack_tsc_cycles.
    """
    s = np.asarray(starts, dtype=np.uint64)
    e = np.asarray(ends, dtype=np.uint64)
    s0, s1, s2 = s & 0xFFFF, (s >> 16) & 0xFFFF, (s >> 32) & 0xFFFF
    e0, e1, e2 = e & 0xFFFF, (e >> 16) & 0xFFFF, (e >> 32) & 0xFFFF
    p0 = ((s1 << 16) | s0).astype(np.uint32)
    p1 = ((e0 << 16) | s2).astype(np.uint32)
    p2 = ((e2 << 16) | e1).astype(np.uint32)
    return np.stack([p0, p1, p2], axis=-1).reshape(-1).view(np.float32).copy()


def test_device_pack_grid_matches_scalar_device_pack():
    pairs = [(0, 0), (100, 350), (350, 100), (U48_MAX, U48_MAX),
             (0x1234_5678_9ABC, 0x1234_5678_9ABD), (0x9ABC, 0x0001_0000_7FC0)]
    starts = np.array([p[0] for p in pairs], dtype=object)
    ends = np.array([p[1] for p in pairs], dtype=object)
    grid = device_pack_grid([int(x) for x in starts], [int(x) for x in ends])
    for i, (s, e) in enumerate(pairs):
        assert np.array_equal(grid.view(np.uint32)[i * 3:(i + 1) * 3],
                              device_pack(s, e).view(np.uint32))


@pytest.mark.parametrize("H, W", [(1, 16), (256, 256), (1155, 742)])
def test_all_zero_buffer_reports_no_stamped_pe_without_raising(H, W):
    """A D2H that returns an unwritten/zeroed buffer must not look like real 0s."""
    buf = np.zeros(H * W * 3, dtype=np.float32)
    pe_cycles, start48 = unpack_tsc_cycles(buf, H, W)
    summ = summarise_tsc_cycles(pe_cycles, start48)

    assert summ["n_total"] == H * W
    assert summ["n_stamped"] == 0
    assert summ["n_zero"] == H * W
    assert summ["reason"] != "ok" and "start48==0" in summ["reason"]
    assert "min" not in summ and "max" not in summ   # no fake stats


@pytest.mark.parametrize("H, W, frac_unstamped", [
    (256, 256, 0.0),      # 65,536 PEs, all healthy
    (256, 256, 0.10),     # 10% never stamped
    (1155, 742, 0.001),   # full wafer, a handful never stamped
])
def test_aggregation_uses_only_stamped_pes(H, W, frac_unstamped):
    n = H * W
    rng = np.random.default_rng(H * W + int(frac_unstamped * 1e6))
    # realistic span range: tens of thousands to a few hundred thousand cycles
    starts = rng.integers(1_000, 5_000_000, size=n, dtype=np.int64)
    spans = rng.integers(40_000, 320_000, size=n, dtype=np.int64)
    ends = starts + spans

    n_unstamped = int(round(frac_unstamped * n))
    if n_unstamped:
        idx = rng.choice(n, size=n_unstamped, replace=False)
        starts[idx] = 0          # never stamped -> start48 == 0
        ends[idx] = 0

    buf = device_pack_grid(starts, ends)
    pe_cycles, start48 = unpack_tsc_cycles(buf, H, W)
    summ = summarise_tsc_cycles(pe_cycles, start48)

    stamped = starts != 0
    ref = (ends - starts)[stamped]

    assert summ["reason"] == "ok"
    assert summ["n_total"] == n
    assert summ["n_stamped"] == int(stamped.sum())
    assert summ["n_zero"] == n_unstamped
    assert summ["min"] == int(ref.min())
    assert summ["max"] == int(ref.max())
    assert summ["mean"] == pytest.approx(float(ref.mean()))
    assert summ["std"] == pytest.approx(float(ref.std()))
    # unstamped PEs must not drag min toward 0
    assert summ["min"] >= 40_000


def test_single_stamped_pe_gives_zero_std():
    starts = np.zeros(64, dtype=np.int64)
    ends = np.zeros(64, dtype=np.int64)
    starts[7], ends[7] = 123_456, 123_456 + 99_999
    buf = device_pack_grid(starts, ends)
    pe_cycles, start48 = unpack_tsc_cycles(buf, H=8, W=8)
    summ = summarise_tsc_cycles(pe_cycles, start48)

    assert summ["n_stamped"] == 1
    assert summ["min"] == summ["max"] == 99_999
    assert summ["std"] == 0.0
    assert summ["reason"] == "ok"


def test_summarise_handles_empty_and_negative_spans():
    assert summarise_tsc_cycles(np.array([]), np.array([]))["reason"] != "ok"

    # end < start on a stamped PE: kept, negative, not wrapped
    starts = np.array([10, 5_000], dtype=np.int64)
    ends = np.array([10, 4_000], dtype=np.int64)   # PE0 span 0, PE1 span -1000
    buf = device_pack_grid(starts, ends)
    pe_cycles, start48 = unpack_tsc_cycles(buf, H=1, W=2)
    summ = summarise_tsc_cycles(pe_cycles, start48)
    assert summ["n_stamped"] == 2
    assert summ["min"] == -1000 and summ["max"] == 0
