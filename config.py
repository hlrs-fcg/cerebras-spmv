import math
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

# Simulating the full wafer is intractable, so a simulator run sizes the fabric
# to wrap its PE grid exactly and caps each grid dimension.
SIM_MAX_GRID_DIM = 100
SIM_MAX_PES = SIM_MAX_GRID_DIM * SIM_MAX_GRID_DIM

# WSE-3: 48 KB SRAM per PE; runtime/code takes ~10 KB, leaving ~38 KB.
PE_SRAM_BUDGET_BYTES = 38 * 1024

# Per-PE NNZ tail bound, calibrated against the measured hardware runs in
# cluster/results/data/.  For the block tiling in cluster/coo_tiling.py the
# observed per-tile maximum is max = avg + k*sqrt(avg), and k tracks
# sqrt(2*ln(num_pes)) from below across five orders of magnitude of PE count:
#
#     num_pes         4     16    1024   65536   428505
#     measured k   1.02   1.96    3.15    4.31     4.96
#     sqrt(2 lnP)  1.67   2.35    3.72    4.71     5.09
#
# So the margin is sqrt(2*avg*ln(num_pes)) with a safety factor, plus a small
# absolute floor to keep tiny-avg and single-PE cases sane.  Against those runs
# the result clears the observed maximum with at least 1.77x headroom.
#
# This is calibrated for the *current* tiling only.  cluster/results/v1.0-data/
# used the old histogram/matched-pairs distribution and shows k up to 25, which
# this bound would not cover.  The backstop is the runtime check in
# cluster/worker_runtime.py, which aborts when max_nnz > TOTAL_NNZ rather than
# letting the setup memcpy overrun col_indices.
NNZ_TAIL_SAFETY = 1.5
NNZ_ABS_FLOOR = 16


def total_nnz_estimate(avg_nnz_per_pe: int, num_pes: int) -> int:
    """Compile-time size for col_indices[TOTAL_NNZ] on one PE.

    col_indices is a *structure* array — one i32 per non-zero in this PE's tile
    — so it is sized from the per-tile NNZ maximum alone.  It deliberately
    carries no KY term: KY belongs to max_stream_words = max_nnz + KY, which is
    the length of the value *stream*, and the PE never buffers the stream
    (src/csl/pe.csl consumes it a word at a time and the host now writes an
    explicit sentinel, so no padding allowance is needed either).
    """
    avg = max(avg_nnz_per_pe, 0)
    tail = NNZ_TAIL_SAFETY * math.sqrt(2 * avg * math.log(max(num_pes, 2)))
    return math.ceil(avg + tail) + NNZ_ABS_FLOOR


# memcpy I/O configuration.  These defaults reproduce what sdk/compiler.py
# hardcoded before they were configurable, so existing results stay comparable.
#
# --channels, per the SDK's own argument help in
# cerebras/sdk/client/sdk_appliance_utils.py (default=0):
#
#     "Number of memcpy I/O channels to use when memcpy support is compiled
#      with this program. If this argument is not present, or is 0, then the
#      previous single-I/O channel version is compiled."
#
# The boundary is therefore at ZERO, not one: absent or 0 selects the legacy
# single-I/O-channel implementation, and any N >= 1 selects the multi-channel
# implementation configured with N channels.  Every result recorded before
# 2026-10-08 was measured at 1 channel (1.38 GB/s / 11.0 Gbps H2D on hardware).
# The default is now 16, the SDK maximum, because the microbenchmarks (cs3-microbench M1/M2)
# scale 1.13 -> 5.15 -> 10.3 GB/s over 1 / 8 / 16 channels; pass --channels 1 for the old
# configuration (its cslc option string is unchanged).
#
# --width-west-buf / --width-east-buf are documented as "width of west/east
# buffer (default is zero, i.e. no buffer to mitigate slow input/output)".
#
# That argument parser documents no maximum for --channels; the limit is
# enforced downstream by the compiler, so the usable ceiling has to be found
# empirically on hardware.
DEFAULT_CHANNELS = 16
DEFAULT_FABRIC_OFFSETS = (4, 1)
DEFAULT_WIDTH_WEST_BUF = 0
DEFAULT_WIDTH_EAST_BUF = 0

# --h2d-mode selects the SHAPE of the streaming A-values H2D transfer.  Unlike
# the three settings above it is a pure runtime knob: it reaches
# cluster/worker_runtime.py, never cslc, and so changes no fabric geometry.
#
#   "column"  W separate memcpy_h2d calls, one per PE column, each a 1-wide x
#             H-tall rectangle of that column's exact word count.  A 1-column
#             transfer cannot be split across I/O channels, which is the
#             leading explanation for the channels sweep degrading monotonically
#             (39.4 ms at channels=1 -> 62-64 ms at 8-16) instead of improving.
#   "full"    one memcpy_h2d over the whole WxH rectangle, every column padded
#             to max(col_words).  A few percent more bytes, but a shape the
#             hardware can actually spread over channels.  Safe only because
#             each PE stream carries an explicit 0 sentinel and the kernel
#             drains trailing padding (src/csl/pe.csl, stream_word "draining").
#
# "column" is the default because it is what every result recorded so far was
# measured under.  Crossing this with --channels is the two-factor experiment
# that separates a genuine ~1.3-1.5 GB/s link ceiling from a transfer-shape
# artifact; see archive/docs/streaming-rewrite-status.md.
H2D_MODES = ("column", "full")
DEFAULT_H2D_MODE = "column"

# --memcpy-order selects the host-side memory order of the streaming A transfer.
# Runtime-only, like h2d_mode: it reaches the worker, never cslc.
#
# The SDK guide states "Column-major delivers better bandwidth than row-major"
# (https://sdk.cerebras.ai/tensor-streaming), and Cerebras' own bandwidth
# benchmark uses COL_MAJOR for the transfer it measures and ROW_MAJOR only for
# small timestamp readback -- stronger evidence than the prose.
#
# BEWARE: that prose is wrong about which axis is fastest.  It says column-major
# makes `w` fastest-varying.  The shipped code disagrees: hwl_2_oned_colmajor in
# csl-examples/benchmarks/bandwidth-test/run.appliance.py:178 loops
# `for l / for w / for h` with h INNERMOST, so h is fastest.  Confirmed from the
# other side by that example's ROW_MAJOR timestamp readback reshaping as
# (height, width, 3) order="C", i.e. ROW_MAJOR is [h][w][l] with l fastest --
# which is what our transfers already did.  See _flatten_whl() in
# cluster/worker_runtime.py for the resulting host layouts.
MEMCPY_ORDERS = ("row", "col")
DEFAULT_MEMCPY_ORDER = "row"
# Memory order of the copy-mode setup transfers.  "col" since 2026-10-08: the fastest verified
# setup copy (6.39-6.75 GB/s at 16 channels against 5.2 row-major, full wafer 8.4-8.7 as strips).
DEFAULT_SETUP_ORDER = "col"
# The setup copy is split into this many vertical strips PER CHANNEL, issued one after another
# (ADR-008 rule 2; cs3-microbench M1/M4: strips reach 10.3 GB/s at 16 channels, one full
# rectangle 5.2-6.75).  --setup-strips 1 is the old single rectangle.
SETUP_STRIPS_PER_CHANNEL = 2

# --encoding selects the A-values stream format, and with it the KERNEL.  Unlike
# h2d_mode / memcpy_order it is compile-time: it reaches cslc as the ENCODING
# param, src/csl/layout.csl picks the kernel from it, and the worker must be told the
# same value (it cross-checks against out.json).
#
#   "header"  ADR-003 as accepted: one (local_row, count) header per non-empty
#             row, then raw values, 0-terminated.  Consumed by a per-word data
#             task (header decode + FMAC).  A comparison mode since v2.0.
#   "valtask" values only (4 B/NNZ), one plan_val_transfers() stream + val_meta,
#             consumed by a per-word data task that stores each word into vals[]
#             and counts to val_meta[0], then a local CSR loop and the EW reduction
#             (src/csl/pe_valtask.csl).  docs/val-only-encoding.md.
#   "resident" A lives on the wafer: setup copies col_indices, row_ptrs AND the values
#             (vals[TOTAL_NNZ]) once, and a call is one launched function: local CSR over
#             the resident arrays, then the EW chain (src/csl/pe_resident.csl).  Nothing
#             streams per call.  The CG case, and the one where the wafer can beat the
#             host CPU (docs/resident-spmv.md).
#
# The ENCODING compile param is the ID below, not the position: id 1 was the bulk "val"
# kernel (removed in v2.0, it hung under concurrent streams), and the ids are kept so artifacts
# compiled before stay valid.  "valtask" is the default since 2026-10-08 (S1 n=3, and the full
# wafer: 443 ms against header's 670 ms at 857K PEs; ADR-003).
ENCODING_IDS = {"header": 0, "valtask": 2, "resident": 3}
ENCODINGS = tuple(ENCODING_IDS)
DEFAULT_ENCODING = "valtask"
# Encodings that stream values only: they share plan_val_transfers, val_meta
# and vals[TOTAL_NNZ].
VAL_ENCODINGS = ("valtask",)
# Encodings with A resident on the wafer: no per-call A stream, so --h2d-mode,
# --memcpy-order and --serialize-column-rects have nothing to act on.
RESIDENT_ENCODINGS = ("resident",)
# --loop-count default for a resident run (run.py): one SpMV is microseconds, so one pass
# says nothing about the spread.  Every other encoding keeps DEFAULT_LOOP_COUNT.
RESIDENT_LOOP_COUNT = 20
# ENCODING param of --null-spmv: pe_resident.csl with spmv() a no-op (layout.csl).  Not in
# ENCODINGS (the encoding is still "resident"); only the compile param differs.
NULL_SPMV_ENCODING_ID = 4
# The resident kernel has one local_spmv, a COO scatter over a row_ids array (setup copies it, TOTAL_NNZ words of
# SRAM). It was chosen on hardware over a CSR loop and a u16 CSR loop (docs/resident-spmv.md); there is no flag.

# --sync-timing (docs/synced-timing.md) compiles the csl-examples
# sync module in (SYNC_TIMING param) so all PEs share a reference clock, and the
# worker brackets every pass with f_tic / f_toc.  It reports the per-pass
# on-wafer time max(toc) - min(tic) across PEs, which a per-PE span cannot give.
# --loop-count repeats the pass (runtime only).  Both default off / 1, so
# existing configs and numbers are unchanged.  The sync module's routing has
# no 1-wide or 1-tall case, so those grids are refused under sync.
DEFAULT_LOOP_COUNT = 1
# 32-bit words sync_timing adds per PE: sync_pack[5] + tic/toc 2x3 u16 (3) +
# the module's buf[1] and 3 u16 reference clock (2).
SYNC_TIMING_EXTRA_WORDS = 5 + 3 + 1 + 2

# --encoding valtask with --h2d-mode column SERIALISES their column rectangles
# automatically, at any word count (cluster/coo_tiling.py:
# column_rects_serialized).  With the W rects all in flight at once
# (nonblock=True) it hangs on hardware once a PE gets more words than the
# fabric buffers; sent one at a time it passes.  See docs/val-only-encoding.md.

# I/O channels and edge buffers consume fabric columns at the west and east
# edges.  No rule relating channel count to required fabric_offsets is
# documented in the SDK argument parser — the limit is enforced downstream by
# the compiler — so it has to be found empirically on hardware.  What this
# repo's own channels=1 artifact shows (cluster_work/out/{west,east}/) is a
# 3-column west halo and a 2-column east halo, consistent with
# fabric_offsets=(4, 1), and width_west_buf / width_east_buf appearing directly
# as halo module params.
#
# So these three fields pass through to cslc untouched and the grid is NOT
# shrunk automatically.  Instead, a config that asks for a non-default I/O
# setup *and* consumes the full usable fabric width warns, because that is the
# case where extra halo columns would overflow the fabric.  Raise
# fabric_offsets/padding to buy the room once the real limit is known.
_IO_GEOMETRY_UNVERIFIED = (
    "channels/width_west_buf/width_east_buf change how many fabric columns the "
    "memcpy halo needs, and the SDK documents no maximum — the compiler enforces "
    "it downstream. This grid already uses the full usable width {width}, so the "
    "halo may not fit. Increase fabric_offsets or padding, or reduce the grid, "
    "if compilation fails on the appliance."
)


def grid_shape(num_pes: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Squarest (width, height) covering num_pes inside a max_w x max_h box.

    Prefers an exact factor pair so width*height == num_pes; a near-square grid
    keeps KY = ceil(rows/height) small, which is what actually has to fit in PE
    SRAM.  Falls back to rounding the grid up when no square-ish factorisation
    fits the box.  Raises rather than silently clamping to a smaller grid.
    """
    if num_pes < 1:
        raise ValueError(f"num_pes_target must be >= 1, got {num_pes}")
    if num_pes > max_w * max_h:
        raise ValueError(
            f"num_pes_target={num_pes:,} exceeds the largest usable grid "
            f"{max_w}x{max_h} = {max_w * max_h:,} PEs"
        )

    # Largest divisor <= sqrt(num_pes); any smaller one only widens the grid.
    # Only taken when it is actually square-ish: a prime target factors as
    # num_pes x 1, and a 1-high grid puts every matrix row on every PE.
    root = math.isqrt(num_pes)
    for h in range(root, 0, -1):
        if num_pes % h == 0:
            w = num_pes // h
            if 2 * h >= root and w <= max_w and h <= max_h:
                return w, h
            break

    h = min(max(root, -(-num_pes // max_w)), max_h)
    w = -(-num_pes // h)
    if w > max_w:
        raise ValueError(
            f"cannot fit num_pes_target={num_pes:,} into a {max_w}x{max_h} grid"
        )
    return w, h


@dataclass
class SpMVConfig:
    # matrix spec
    rows: int = 0
    cols: int = 0
    density: float = 0.0
    seed: int = 42
    mtx_file: Optional[Path] = None

    # hardware — always uses the full usable fabric
    fabric_dims: tuple = (762, 1172)
    fabric_offsets: tuple = DEFAULT_FABRIC_OFFSETS
    padding: tuple = (16, 16)
    arch: str = "wse3"
    simulator: bool = False
    disable_version_check: bool = True

    out_path: Path = Path("cluster_work/out")
    comment: str = ""

    # optional PE count / grid shape; None = use full fabric
    num_pes_target: Optional[int] = None
    # H; None = derive a near-square grid from num_pes_target, else full fabric
    grid_height_target: Optional[int] = None

    # column-stripe streaming: S passes over A so SRAM shrinks ~S×
    num_col_stripes: int = 1

    # memcpy I/O configuration; defaults reproduce the previous hardcoded flags
    channels: int = DEFAULT_CHANNELS
    width_west_buf: int = DEFAULT_WIDTH_WEST_BUF
    width_east_buf: int = DEFAULT_WIDTH_EAST_BUF

    # H2D transfer shape; runtime-only, passed through to the worker
    h2d_mode: str = DEFAULT_H2D_MODE

    # host-side memory order of the streaming A transfer; runtime-only
    memcpy_order: str = DEFAULT_MEMCPY_ORDER

    # A-values stream encoding; COMPILE-TIME (selects the kernel), see ENCODINGS
    encoding: str = DEFAULT_ENCODING
    # memory order of the copy-mode setup transfers; runtime-only
    setup_order: str = DEFAULT_SETUP_ORDER
    # vertical strips of the copy-mode setup transfers, issued serially; runtime-only.
    # None: SETUP_STRIPS_PER_CHANNEL x channels (see resolved_setup_strips); 1: one rectangle.
    setup_strips: Optional[int] = None
    # Explicitly serialise the streaming column rects: every one but the last
    # nonblock=False, so no two streaming memcpys are in flight at once.  valtask
    # in column mode does this regardless (it hangs otherwise); for header it
    # is opt-in.  Runtime-only (worker); never reaches cslc or the artifact.
    serialize_column_rects: bool = False

    # synced on-wafer timing: COMPILE-TIME (SYNC_TIMING param), see above
    sync_timing: bool = False
    # passes over the same A and x; runtime-only (worker)
    loop_count: int = DEFAULT_LOOP_COUNT
    # resident only, COMPILE-TIME (NULL_SPMV param): spmv() computes nothing and releases the
    # command stream at once, so the synced window is the launch machinery alone.  The
    # worker skips validation (y is never written).  docs/resident-spmv.md.
    null_spmv: bool = False

    # grid — derived from fabric + targets in __post_init__, not overridable
    width: int = field(init=False)
    height: int = field(init=False)
    num_pes: int = field(init=False)

    # CSL array sizes — estimated from matrix params, can be overridden for real matrices
    total_nnz_per_pe: Optional[int] = None

    # set by sdk/compiler.compile() after CSL compilation
    artifact_id: str = field(init=False, default="")
    compile_time: float = field(init=False, default=0.0)

    # set by sdk/launcher.run() from the login-node repo before staging
    git_commit: str = field(init=False, default="unknown")

    def __post_init__(self):
        # Explicit [[run]] entries in a sweep TOML are splatted in as raw TOML
        # scalars (cluster/sweep_config.py: expand), so path fields arrive as
        # str and .mkdir()/.name blow up downstream.  The [sweep] path never hit
        # this because it passes an already-built Path.  Coerce here rather than
        # trusting every call site.
        self.out_path = Path(self.out_path)
        if self.mtx_file is not None:
            self.mtx_file = Path(self.mtx_file)

        for name in ("channels", "width_west_buf", "width_east_buf"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")

        # Reject an unknown mode here rather than on the appliance: the worker's
        # argparse would also catch it, but only after a ~70 s compile.
        if self.h2d_mode not in H2D_MODES:
            raise ValueError(
                f"h2d_mode must be one of {H2D_MODES}, got {self.h2d_mode!r}"
            )

        if self.memcpy_order not in MEMCPY_ORDERS:
            raise ValueError(
                f"memcpy_order must be one of {MEMCPY_ORDERS}, got {self.memcpy_order!r}"
            )

        if self.encoding not in ENCODINGS:
            raise ValueError(
                f"encoding must be one of {ENCODINGS}, got {self.encoding!r}"
            )

        if self.setup_strips is not None and (
                not isinstance(self.setup_strips, int) or isinstance(self.setup_strips, bool)
                or self.setup_strips < 1):
            raise ValueError(f"setup_strips must be an int >= 1 or None, got {self.setup_strips!r}")

        if self.setup_order not in MEMCPY_ORDERS:
            raise ValueError(
                f"setup_order must be one of {MEMCPY_ORDERS}, got {self.setup_order!r}"
            )

        if (not isinstance(self.loop_count, int) or isinstance(self.loop_count, bool)
                or self.loop_count < 1):
            raise ValueError(f"loop_count must be an int >= 1, got {self.loop_count!r}")
        if not isinstance(self.sync_timing, bool):
            raise ValueError(f"sync_timing must be a bool, got {self.sync_timing!r}")
        if not isinstance(self.null_spmv, bool):
            raise ValueError(f"null_spmv must be a bool, got {self.null_spmv!r}")
        if self.null_spmv and self.encoding not in RESIDENT_ENCODINGS:
            raise ValueError("null_spmv is a resident-kernel path: use --encoding resident")

        # cslc's memcpy generator requires --fabric-offsets=4+width_west_buf,1 (its own error,
        # SDK 2.10.0: "The core must have --fabric-offsets=4+width_west_buf,1").  Applied to
        # the default offsets only, so an explicit fabric_offsets is respected (and a config
        # reloaded from its own to_dict(), which already holds the derived value, is stable).
        # Zero buffers leave the offsets, and so the cslc option string, untouched.
        if self.width_west_buf > 0 and tuple(self.fabric_offsets) == DEFAULT_FABRIC_OFFSETS:
            self.fabric_offsets = (DEFAULT_FABRIC_OFFSETS[0] + self.width_west_buf,
                                   DEFAULT_FABRIC_OFFSETS[1])

        # The simulator shapes the grid first and wraps the fabric around it;
        # hardware fits the grid into the fabric it was given.
        if self.simulator:
            max_w = max_h = SIM_MAX_GRID_DIM
        else:
            max_w = self.fabric_dims[0] - self.fabric_offsets[0] - self.padding[0]
            max_h = self.fabric_dims[1] - self.fabric_offsets[1] - self.padding[1]

        if self.grid_height_target is not None:
            h = self.grid_height_target
            if h > max_h:
                raise ValueError(
                    f"grid_height_target={h} exceeds the largest usable height {max_h}"
                )
            if self.num_pes_target is not None:
                w = -(-self.num_pes_target // h)
                if w > max_w:
                    raise ValueError(
                        f"num_pes_target={self.num_pes_target:,} at height {h} needs "
                        f"width {w}, which exceeds the largest usable width {max_w}"
                    )
            else:
                w = max_w
        elif self.num_pes_target is not None:
            w, h = grid_shape(self.num_pes_target, max_w, max_h)
        else:
            w, h = max_w, max_h

        self.width = w
        self.height = h
        self.num_pes = w * h

        # cslc refuses more memcpy channels than kernel rows ("Can not have more
        # memcpy channels than kernel rows", _generate_memcpy), which the
        # 16-channel default hits on any grid under 16 rows (4x4 smoke, 2026-10-09).
        # Clamp rather than fail after a compile; the CSV records the clamped value.
        # Only an explicit request is worth a warning; the default clamps silently.
        if self.channels > h:
            if self.channels != DEFAULT_CHANNELS:
                warnings.warn(
                    f"channels={self.channels} exceeds the grid height {h}; "
                    f"clamped to {h} (cslc allows at most one channel per kernel row)",
                    stacklevel=3,
                )
            self.channels = h

        if self.sync_timing and (w < 2 or h < 2):
            raise ValueError(
                f"sync_timing needs a grid at least 2x2, got {w}x{h}: the "
                f"csl-examples sync module has no 1-wide / 1-tall routing case"
            )

        if self.uses_non_default_io and w == max_w:
            warnings.warn(_IO_GEOMETRY_UNVERIFIED.format(width=w), stacklevel=3)

        if self.simulator:
            # Size the fabric to wrap the grid plus the memcpy halo and padding.
            self.fabric_dims = (
                w + self.fabric_offsets[0] + self.padding[0],
                h + self.fabric_offsets[1] + self.padding[1],
            )

        if self.total_nnz_per_pe is None and self.rows > 0:
            avg_nnz = int(self.rows * self.cols * self.density / self.num_pes)
            avg_nnz_per_stripe = avg_nnz // max(self.num_col_stripes, 1)
            self.total_nnz_per_pe = total_nnz_estimate(avg_nnz_per_stripe, self.num_pes)

        # Resident per-PE arrays in src/csl/pe.csl, all 32-bit:
        #   x_vec[KX] + col_indices[TOTAL_NNZ] + row_ptrs[KY+1]
        #   + y_result[KY] + ew_buf[KY]
        # A is streamed in a word at a time, so there is no a_stream buffer.
        # With TOTAL_NNZ free of its old KY term this is KX + 3*KY per PE plus
        # the NNZ terms, i.e. 1/W + 3/H words per unknown rather than 1/W + 4/H.
        # The val-only kernel (pe_valtask.csl) adds vals[TOTAL_NNZ] + val_meta[1].
        if self.rows > 0 and self.total_nnz_per_pe is not None:
            _words = (
                self.kx
                + self.total_nnz_per_pe
                + (self.ky + 1)
                + self.ky
                + self.ky
                + self.val_only_extra_words
                + (SYNC_TIMING_EXTRA_WORDS if self.sync_timing else 0)
            )
            _bytes = _words * 4
            if _bytes > PE_SRAM_BUDGET_BYTES:
                raise ValueError(
                    f"Estimated PE data usage {_bytes / 1024:.1f} KB exceeds "
                    f"{PE_SRAM_BUDGET_BYTES // 1024} KB usable SRAM per PE "
                    f"(KX={self.kx} KY={self.ky} TOTAL_NNZ={self.total_nnz_per_pe}, "
                    f"encoding={self.encoding}, sync_timing={self.sync_timing}). "
                    f"Use more PEs or a smaller matrix."
                )

    @property
    def uses_non_default_io(self) -> bool:
        """True when the memcpy I/O setup differs from the historical default."""
        return (
            self.channels != min(DEFAULT_CHANNELS, self.height)
            or self.width_west_buf != DEFAULT_WIDTH_WEST_BUF
            or self.width_east_buf != DEFAULT_WIDTH_EAST_BUF
        )

    @property
    def val_only_extra_words(self) -> int:
        """32-bit words pe_valtask.csl holds beyond pe.csl:
        vals[TOTAL_NNZ] + val_meta[1].  pe_resident.csl has vals[TOTAL_NNZ] and row_ids[TOTAL_NNZ], no val_meta."""
        if self.total_nnz_per_pe is None:
            return 0
        if self.encoding in RESIDENT_ENCODINGS:
            # vals[TOTAL_NNZ] and row_ids[TOTAL_NNZ] (the COO scatter)
            return self.total_nnz_per_pe * 2
        if self.encoding not in VAL_ENCODINGS:
            return 0
        return self.total_nnz_per_pe + 1

    @property
    def resolved_setup_strips(self) -> int:
        """Strips the worker splits each setup transfer into: setup_strips, or 2 x channels
        when unset, never more than the grid is wide (a strip is at least one PE column)."""
        k = self.setup_strips if self.setup_strips is not None \
            else SETUP_STRIPS_PER_CHANNEL * max(self.channels, 1)
        return max(1, min(k, self.width))

    @property
    def encoding_id(self) -> int:
        """The ENCODING compile param: the id in ENCODING_IDS; 4 for the resident kernel with
        spmv() compiled as a no-op (null_spmv), so no existing option string changes."""
        if self.null_spmv:
            return NULL_SPMV_ENCODING_ID
        return ENCODING_IDS[self.encoding]

    @property
    def kx(self) -> int:
        """x elements per PE column = ceil(cols / width)."""
        return (self.cols + self.width - 1) // self.width if self.cols > 0 else 1

    @property
    def kx_stripe(self) -> int:
        """Compile-time KX: x elements per stripe = ceil(kx / num_col_stripes)."""
        s = max(self.num_col_stripes, 1)
        return (self.kx + s - 1) // s

    @property
    def ky(self) -> int:
        """Rows per PE row = ceil(rows / height)."""
        return (self.rows + self.height - 1) // self.height if self.rows > 0 else 1

    @property
    def fabric_dims_str(self) -> str:
        return f"{self.fabric_dims[0]},{self.fabric_dims[1]}"

    @property
    def fabric_offsets_str(self) -> str:
        return f"{self.fabric_offsets[0]},{self.fabric_offsets[1]}"

    def to_dict(self) -> dict:
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Path):
                v = str(v)
            elif isinstance(v, tuple):
                v = list(v)
            d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SpMVConfig":
        init_names = {f.name for f in fields(cls) if f.init}
        kwargs = {}
        for f in fields(cls):
            if f.name not in init_names or f.name not in d:
                continue
            v = d[f.name]
            if f.name in ("mtx_file", "out_path") and v is not None:
                v = Path(v)
            elif f.name in ("fabric_dims", "fabric_offsets", "padding") and isinstance(
                v, list
            ):
                v = tuple(v)
            kwargs[f.name] = v
        cfg = cls(**kwargs)
        if "artifact_id" in d:
            object.__setattr__(cfg, "artifact_id", d["artifact_id"])
        if "git_commit" in d:
            object.__setattr__(cfg, "git_commit", d["git_commit"])
        if "compile_time" in d:
            object.__setattr__(cfg, "compile_time", d["compile_time"])
        return cfg

    def __str__(self) -> str:
        lines = [f"{f.name}: {getattr(self, f.name)}" for f in fields(self)]
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = SpMVConfig(rows=100_000, cols=100_000, density=0.001)
    print(cfg)
    roundtrip = SpMVConfig.from_dict(cfg.to_dict())
    assert str(cfg) == str(roundtrip), "Roundtrip failed"
    print("\nRoundtrip OK")
