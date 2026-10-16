"""CSL compilation via the Cerebras SDK appliance (SdkCompiler).

Compilation runs on the appliance, not locally: the CS-3 user node has neither
`cslc` nor `cs_python` on PATH, so there is no local-subprocess path.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

from config import SpMVConfig
from utils.logging import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# src/csl/pe.csl gates the east-west reduction on this; 1 = reduce partial y
# across the PE row, which is what cluster/worker_runtime.py's column-0 D2H
# assumes.
EW_ENABLED = 1

# What cslc needs from the repo: the CSL sources (layout.csl imports pe*.csl and sync/pe.csl by
# relative path).  stage_csl copies exactly these into a temporary directory for the compile.
CSL_DIRS = ("src/csl",)
CSL_MAIN = "src/csl/layout.csl"

# Streaming H2D color id.  0 selects MEMCPYH2D_1, the color pe.csl binds its
# data task to via @initialize_queue(h2d_iq, .{.color = MEMCPYH2D_1}).
MEMCPYH2D_DATA_1_ID = 0


def stage_csl(csl_dirs, dest: Path, root: Path = _PROJECT_ROOT) -> Path:
    """Copy every ``*.csl`` under ``root/<csl_dirs>`` into ``dest/<root.name>/``, keeping the
    relative paths, and return that directory.

    SdkCompiler packages whatever directory it is given.  Handed the repo root it reads
    ``.git`` too, so a commit or merge in a synced tree while it ran failed a compile with
    ``FileNotFoundError: .../.git/objects/..``.  Staging exactly the CSL sources (cslc imports
    nothing else) into a private temporary directory, under the same directory name and the
    same relative paths, takes the compile off the live tree.
    """
    app = dest / root.name
    for d in csl_dirs:
        for f in sorted((root / d).rglob("*.csl")):
            if ".git" in f.relative_to(root).parts:
                continue                    # belt and braces for a csl_dirs that is too wide
            target = app / f.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
    return app


def sdk_compile(csl_dirs, csl_main: str, options: str, out_path: str,
                disable_version_check: bool, root: Path = _PROJECT_ROOT):
    """SdkCompiler.compile on a staged copy of ``csl_dirs``; returns the artifact id (or None)."""
    # Imported here, not at module scope: the SDK is Linux-only and run.py must
    # stay importable (--help, tests) on machines without it.
    from cerebras.sdk.client import SdkCompiler

    with tempfile.TemporaryDirectory(prefix="csl-stage-") as tmp:
        app = stage_csl(csl_dirs, Path(tmp), root)
        with SdkCompiler(resource_cpu=24000, resource_mem=64 << 30,
                         disable_version_check=disable_version_check) as compiler:
            return compiler.compile(app_path=str(app), csl_main=csl_main, options=options,
                                    out_path=out_path)


def compile_options(cfg: SpMVConfig) -> str:
    """cslc option string for cfg.

    Every param declared in src/csl/layout.csl must appear in --params or cslc
    fails: width, height, KX, KY, TOTAL_NNZ, EW_ENABLED, MEMCPYH2D_DATA_1_ID,
    ENCODING, SYNC_TIMING.
    """
    # KX, not cfg.kx_stripe: column striping is not implemented end to end
    # (cluster/worker_runtime.py rejects --num-stripes > 1), so the PE still
    # sizes x_vec for the whole column.
    params = ",".join([
        f"width:{cfg.width}",
        f"height:{cfg.height}",
        f"KX:{cfg.kx}",
        f"KY:{cfg.ky}",
        f"TOTAL_NNZ:{cfg.total_nnz_per_pe}",
        f"EW_ENABLED:{EW_ENABLED}",
        f"MEMCPYH2D_DATA_1_ID:{MEMCPYH2D_DATA_1_ID}",
        # Selects pe.csl (0, row-header), pe_valtask.csl (2) or
        # pe_resident.csl (3, A resident) in layout.csl.  Part of the option string,
        # so the kernels never share an artifact.
        f"ENCODING:{cfg.encoding_id}",
        # 1 compiles in the sync module and f_sync / f_tic / f_toc
        # (docs/synced-timing.md); 0 leaves both kernels as they were.
        f"SYNC_TIMING:{int(cfg.sync_timing)}",
    ])
    # --channels selects the memcpy I/O implementation: per the SDK's argument
    # help (default=0), absent or 0 compiles "the previous single-I/O channel
    # version", while any N >= 1 compiles the multi-channel version with N
    # channels.  Always emitted so the value is explicit in the compile log.
    # The width-*-buf flags are emitted only when non-zero, which keeps the
    # option string byte-identical to the pre-change one at default settings.
    io_flags = [f"--channels={cfg.channels}"]
    if cfg.width_west_buf:
        io_flags.append(f"--width-west-buf={cfg.width_west_buf}")
    if cfg.width_east_buf:
        io_flags.append(f"--width-east-buf={cfg.width_east_buf}")

    return (
        f"--fabric-dims={cfg.fabric_dims[0]},{cfg.fabric_dims[1]} "
        f"--fabric-offsets={cfg.fabric_offsets[0]},{cfg.fabric_offsets[1]} "
        f"--memcpy {' '.join(io_flags)} "
        f"--arch={cfg.arch} "
        f"--params={params}"
    )


def compile(cfg: SpMVConfig) -> None:
    """Compile CSL for cfg and save artifact.json.

    Sets cfg.artifact_id and cfg.compile_time on success.
    """
    logger.info(f"Compiling for {cfg.width}×{cfg.height} grid ({cfg.num_pes:,} PEs)")
    logger.info(f"  KX={cfg.kx}, KY={cfg.ky}, TOTAL_NNZ={cfg.total_nnz_per_pe}, "
                f"encoding={cfg.encoding}")

    cfg.out_path.mkdir(parents=True, exist_ok=True)
    out_path = str(cfg.out_path.resolve())

    options = compile_options(cfg)
    logger.debug(f"cslc options: {options}")

    t0 = time.time()
    artifact_id = sdk_compile(CSL_DIRS, CSL_MAIN, options, out_path, cfg.disable_version_check)
    compile_time = time.time() - t0

    if artifact_id is None:
        raise RuntimeError("CSL compilation failed — no artifact produced")

    cfg.artifact_id = artifact_id
    cfg.compile_time = compile_time
    logger.info(f"Compiled in {compile_time:.2f}s — artifact: {artifact_id}")

    with open(cfg.out_path / "artifact.json", "w", encoding="utf8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logger.info(f"Artifact saved to {cfg.out_path / 'artifact.json'}")
