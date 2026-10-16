#!/usr/bin/env python3
"""
Worker Runtime Script
Runs on worker node via SdkLauncher. Executes SpMV on Cerebras hardware.
Self-contained: no imports from login-node utils/ or sdk/.

This file is the entry point and the order of the stages; the work is in the sibling modules
SdkLauncher stages next to it (sdk/launcher.py:STAGED_FILES, all flat):

    worker_cli       the command line
    worker_host      geometry, matrix, tiled host buffers, the artifact's out.json
    worker_device    connect, setup copy, timed passes, per-PE cycle readback
    worker_modes     the hooks that differ per mode:  worker_stream (header / val / valtask),
                     worker_resident (A resident; a future CG mode goes beside it)
    worker_report    summary, validation, the CSV row
    worker_common / worker_layout / worker_timing   shared helpers

The helper names that tests and docs imported from here are re-exported below.
"""

import logging
import sys
import time

# Before anything that imports numba: the appliance's Python has none, so a staged zip of numba +
# llvmlite (--vendor, built by sdk/vendor_pack.py) is unpacked onto sys.path first.
import worker_env
worker_env.activate_vendor(sys.argv)

import tile_cache  # noqa: F401  (re-exported)
import tile_kernels
from benchmark import BenchmarkResult, save_benchmark  # noqa: F401  (re-exported)
from coo_tiling import (  # noqa: F401  (ENCODINGS ... spmv_reference are re-exported)
    ENCODING_IDS,
    ENCODINGS,
    NULL_SPMV_ENCODING_ID,
    RESIDENT_ENCODINGS,
    VAL_ENCODINGS,
    build_row_ids,
    column_rects_serialized,
    generate_random_coo,
    spmv_reference,
)
from host_prep import prepare_host_buffers  # noqa: F401  (re-exported)
from worker_cli import parse_args
from worker_common import (  # noqa: F401  (re-exported)
    Run,
    SdkRuntimeWrapper,
    artifact_encoding,
    artifact_local_kernel,
    artifact_null_spmv,
    artifact_sync_timing,
    make_progress,
    validate_results,
)
from worker_device import run_device
from worker_host import load_matrix, prepare_host, read_compile_data, set_geometry
from worker_layout import (  # noqa: F401  (re-exported)
    _flatten_pe_major,
    _flatten_strip,
    _flatten_whl,
    gather_y,
    rect_nonblock_flags,
    setup_strip_bounds,
    setup_strip_calls,
)
from worker_report import finish
from worker_resident import (  # noqa: F401  (re-exported)
    ResidentMode,
    resident_pass_commands,
)
from worker_stream import StreamMode
from worker_timing import (  # noqa: F401  (re-exported)
    WSE_CLOCK_HZ,
    corrected_time,
    make_u48,
    pass_stats,
    sub_ts,
    summarise_tsc_cycles,
    synced_pass_cycles,
    timed_d2h,
    unpack_sync_pack,
    unpack_tsc_cycles,
    wafer_host_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    resident = args.encoding in RESIDENT_ENCODINGS

    # Streaming column rects one at a time?  Always for val + column (W
    # overlapping nonblocking rects hung on hardware; serialised they pass),
    # opt-in otherwise.  Decided and logged once, used by every send below.
    serialize_rects = column_rects_serialized(args.encoding, args.h2d_mode,
                                              args.serialize_column_rects)
    if resident:
        serialize_rects = False   # nothing streams: --serialize-column-rects does nothing
    if serialize_rects and args.h2d_mode == "column":
        logger.info(
            "H2D column mode: column rects SERIALISED (all but the last "
            "nonblock=False) -- "
            + ("requested (--serialize-column-rects)" if args.serialize_column_rects
               else "automatic for --encoding valtask: overlapping column "
                    "transfers hang on hardware (docs/val-only-encoding.md)")
        )

    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info(f"Artifact: {args.name}  System: {args.cmaddr}")
    logger.info("=" * 60)
    env_info = worker_env.diagnostics(tile_kernels)
    logger.info("Worker: " + "; ".join(f"{k}={v}" for k, v in env_info.items()))

    total_start = time.perf_counter()
    if args.progress_file:
        open(args.progress_file, "w").close()      # a fresh file per run
    progress = make_progress(args.progress_file or None, total_start)
    progress(f"start {args.encoding} {args.grid_width}x{args.grid_height} "
             f"loop_count={args.loop_count} sync={args.sync_timing} "
             f"null_spmv={args.null_spmv}")

    r = Run(args, total_start, progress, env_info)
    r.resident = resident
    r.serialize_rects = serialize_rects

    set_geometry(r)
    for stage in (load_matrix, prepare_host, read_compile_data):
        code = stage(r)
        if code:
            return code

    mode = ResidentMode() if resident else StreamMode()
    code = run_device(r, mode, (SdkRuntime, MemcpyDataType, MemcpyOrder))
    if code:
        return code
    return finish(r, mode)


if __name__ == "__main__":
    sys.exit(main())
