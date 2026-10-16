"""TOML sweep configuration: load, expand, and generate sweep configs."""

import math
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

from cluster import mtx_sizing
from config import (
    DEFAULT_CHANNELS,
    DEFAULT_ENCODING,
    DEFAULT_H2D_MODE,
    DEFAULT_LOOP_COUNT,
    DEFAULT_MEMCPY_ORDER,
    DEFAULT_SETUP_ORDER,
    DEFAULT_WIDTH_EAST_BUF,
    DEFAULT_WIDTH_WEST_BUF,
    ENCODINGS,
    H2D_MODES,
    MEMCPY_ORDERS,
    SpMVConfig,
)

# [sweep] keys that configure memcpy I/O rather than the parameter sweep itself.
# These three are compile-time: they reach cslc and change fabric geometry.
IO_KEYS = ("channels", "width_west_buf", "width_east_buf")

# [sweep] keys that configure the run but are resolved at runtime by the worker,
# not by cslc.  Kept apart from IO_KEYS because they are strings rather than
# non-negative ints and because they trigger no fabric-geometry warning.
RUNTIME_KEYS = ("h2d_mode", "memcpy_order", "setup_order")

# [sweep] keys that select the KERNEL.  Compile-time like IO_KEYS (they reach
# cslc as params), but strings, and they change no fabric geometry, so they
# trigger no halo warning either.
KERNEL_KEYS = ("encoding",)

# [sweep] keys that are booleans.  serialize_column_rects is runtime (worker
# only): automatic for val in column mode, opt-in for header; sync_timing
# is compile-time (SYNC_TIMING param, docs/synced-timing.md); null_spmv is compile-time,
# resident only (docs/resident-spmv.md).
BOOL_KEYS = ("serialize_column_rects", "sync_timing", "null_spmv")

# [sweep] keys that are runtime ints: passes per run (docs/synced-timing.md).
LOOP_KEYS = ("loop_count",)

# [sweep] key: strips of the copy-mode setup transfers (runtime int; unset = 2 x channels).
SETUP_KEYS = ("setup_strips",)

# String-valued overrides, whichever phase consumes them.
STRING_KEYS = (*RUNTIME_KEYS, *KERNEL_KEYS)

# Everything a TOML [sweep] block or a CLI flag may override per run.
OVERRIDE_KEYS = (*IO_KEYS, *RUNTIME_KEYS, *KERNEL_KEYS, *BOOL_KEYS, *LOOP_KEYS, *SETUP_KEYS)


def _coerce(key: str, value):
    """A TOML or CLI value for override key, as SpMVConfig wants it."""
    if key in STRING_KEYS:
        return str(value)
    if key in BOOL_KEYS:
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false, got {value!r}")
        return value
    return int(value)

FABRIC_MAX_PES = 742 * 1155  # 857,010


def _pow2_sequence(up_to: int, points_per_octave: int) -> list[int]:
    """Return rounded 2^(n/ppo) values <= up_to, starting from the first >= 64."""
    ppo = points_per_octave
    values = []
    n = 0
    while True:
        v = round(2 ** (n / ppo))
        if v > up_to:
            break
        if v >= 64:
            values.append(v)
        n += 1
    # deduplicate while preserving order (rounding can collide at low n)
    seen = set()
    result = []
    for v in values:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


@dataclass
class NumPesSpec:
    pow2_up_to: int  # resolved ceiling
    points_per_octave: int = 1

    def sequence(self) -> list[int]:
        return _pow2_sequence(self.pow2_up_to, self.points_per_octave)

    @classmethod
    def from_dict(cls, d: dict) -> "NumPesSpec":
        raw = d.get("pow2_up_to", "fabric_max")
        ceiling = FABRIC_MAX_PES if raw == "fabric_max" else int(raw)
        return cls(pow2_up_to=ceiling, points_per_octave=int(d.get("points_per_octave", 1)))


@dataclass
class SweepConfig:
    name: str
    description: str
    work_dir: Path
    results_template: str  # may contain {date} and {name}

    # exactly one of sweep_* or explicit_runs is populated
    sweep_type: Optional[str] = None        # "weak_scaling" | "strong_scaling"
    sweep_params: dict = field(default_factory=dict)
    num_pes_spec: Optional[NumPesSpec] = None

    explicit_runs: list[dict] = field(default_factory=list)
    
    # simulator mode
    simulator: bool = False

    # memcpy I/O configuration applied to every run in the sweep
    channels: int = DEFAULT_CHANNELS
    width_west_buf: int = DEFAULT_WIDTH_WEST_BUF
    width_east_buf: int = DEFAULT_WIDTH_EAST_BUF

    # H2D transfer shape applied to every run in the sweep
    h2d_mode: str = DEFAULT_H2D_MODE

    # host-side memory order applied to every run in the sweep
    memcpy_order: str = DEFAULT_MEMCPY_ORDER

    # A-values stream encoding (selects the kernel) applied to every run
    encoding: str = DEFAULT_ENCODING
    # memory order of the copy-mode setup transfers
    setup_order: str = DEFAULT_SETUP_ORDER
    # strips per setup transfer; None = 2 x channels
    setup_strips: Optional[int] = None
    # serialise the streaming column rects (automatic for val + column)
    serialize_column_rects: bool = False
    # synced on-wafer timing and passes per run
    sync_timing: bool = False
    loop_count: int = DEFAULT_LOOP_COUNT
    # resident only, compile-time: spmv() is a no-op (launch-latency baseline)
    null_spmv: bool = False

    @property
    def io_settings(self) -> dict:
        """Compile-time memcpy I/O settings only."""
        return {k: getattr(self, k) for k in IO_KEYS}

    @property
    def run_settings(self) -> dict:
        """Every per-run override, compile-time and runtime, for SpMVConfig."""
        return {k: getattr(self, k) for k in OVERRIDE_KEYS}

    @classmethod
    def from_toml(
        cls,
        path: Path,
        cli_override_simulator: bool = False,
        cli_override_io: Optional[dict] = None,
    ) -> "SweepConfig":
        with open(path, "rb") as f:
            doc = tomllib.load(f)

        meta = doc.get("meta", {})
        out = doc.get("output", {})

        name = meta.get("name", path.stem)
        description = meta.get("description", "")
        work_dir = Path(out.get("work_dir", "cluster_work/sweeps"))
        results_template = out.get("results", "cluster/results/data/{date}_{name}.csv")

        sweep_type = None
        sweep_params: dict = {}
        num_pes_spec = None
        explicit_runs: list[dict] = []
        simulator = False
        settings = {
            "channels": DEFAULT_CHANNELS,
            "width_west_buf": DEFAULT_WIDTH_WEST_BUF,
            "width_east_buf": DEFAULT_WIDTH_EAST_BUF,
            "h2d_mode": DEFAULT_H2D_MODE,
            "memcpy_order": DEFAULT_MEMCPY_ORDER,
            "encoding": DEFAULT_ENCODING,
            "setup_order": DEFAULT_SETUP_ORDER,
            "setup_strips": None,
            "serialize_column_rects": False,
            "sync_timing": False,
            "loop_count": DEFAULT_LOOP_COUNT,
            "null_spmv": False,
        }

        if "sweep" in doc:
            s = doc["sweep"]
            sweep_type = s["type"]
            num_pes_spec = NumPesSpec.from_dict(s.get("num_pes", {}))
            reserved = ("type", "num_pes", "simulator", *OVERRIDE_KEYS)
            sweep_params = {k: v for k, v in s.items() if k not in reserved}
            simulator = s.get("simulator", False)
            for key in IO_KEYS:
                if key in s:
                    settings[key] = int(s[key])
            for key in (*STRING_KEYS, *BOOL_KEYS, *LOOP_KEYS, *SETUP_KEYS):
                if key in s:
                    settings[key] = _coerce(key, s[key])

        if "run" in doc:
            explicit_runs = doc["run"]

        # CLI override takes precedence over the TOML, which overrides defaults
        if cli_override_simulator:
            simulator = True
        for key, value in (cli_override_io or {}).items():
            if key not in OVERRIDE_KEYS:
                raise ValueError(f"unknown per-run override {key!r}")
            settings[key] = _coerce(key, value)

        if settings["h2d_mode"] not in H2D_MODES:
            raise ValueError(
                f"h2d_mode must be one of {H2D_MODES}, got {settings['h2d_mode']!r}"
            )

        for key in ("memcpy_order", "setup_order"):
            if settings[key] not in MEMCPY_ORDERS:
                raise ValueError(
                    f"{key} must be one of {MEMCPY_ORDERS}, got {settings[key]!r}"
                )

        if settings["encoding"] not in ENCODINGS:
            raise ValueError(
                f"encoding must be one of {ENCODINGS}, got {settings['encoding']!r}"
            )

        return cls(
            name=name,
            description=description,
            work_dir=work_dir,
            results_template=results_template,
            sweep_type=sweep_type,
            sweep_params=sweep_params,
            num_pes_spec=num_pes_spec,
            explicit_runs=explicit_runs,
            simulator=simulator,
            **settings,
        )

    def results_path(self) -> Path:
        today = date.today().isoformat()
        return Path(self.results_template.format(date=today, name=self.name))

    def expand(self) -> Iterator[SpMVConfig]:
        """Yield one SpMVConfig per run."""
        if self.explicit_runs:
            for i, r in enumerate(self.explicit_runs, 1):
                if r.get("mtx_file"):
                    # rows, cols, density and TOTAL_NNZ come from the file; refused here if it cannot fit, which is
                    # before run.cmd_sweep starts any run (it expands all of them first)
                    try:
                        r = mtx_sizing.size_run(r)
                    except (ValueError, OSError) as e:
                        raise ValueError(f"[[run]] #{i} ({r.get('comment') or r['mtx_file']}): {e}") from e
                yield SpMVConfig(**{k: v for k, v in r.items()})
            return

        if self.sweep_type == "weak_scaling":
            yield from self._expand_weak_scaling()
        elif self.sweep_type == "strong_scaling":
            yield from self._expand_strong_scaling()
        else:
            raise ValueError(f"Unknown sweep type: {self.sweep_type!r}")

    def _expand_weak_scaling(self) -> Iterator[SpMVConfig]:
        p = self.sweep_params
        nnz_per_pe: int = p["nnz_per_pe"]
        density: float = p["density"]
        seed: int = p.get("seed", 42)
        comment: str = p.get("comment", "")

        for num_pes in self.num_pes_spec.sequence():
            total_nnz = nnz_per_pe * num_pes
            # rows = cols = sqrt(total_nnz / density), rounded to nearest int
            side = round(math.sqrt(total_nnz / density))
            yield SpMVConfig(
                rows=side,
                cols=side,
                density=density,
                seed=seed,
                num_pes_target=num_pes,
                out_path=self.work_dir,
                comment=comment or f"weak_scaling num_pes={num_pes}",
                simulator=self.simulator,
                **self.run_settings,
            )

    def _expand_strong_scaling(self) -> Iterator[SpMVConfig]:
        p = self.sweep_params
        rows: int = p["rows"]
        cols: int = p["cols"]
        density: float = p["density"]
        seed: int = p.get("seed", 42)
        comment: str = p.get("comment", "")

        for num_pes in self.num_pes_spec.sequence():
            yield SpMVConfig(
                rows=rows,
                cols=cols,
                density=density,
                seed=seed,
                num_pes_target=num_pes,
                out_path=self.work_dir,
                comment=comment or f"strong_scaling num_pes={num_pes}",
                simulator=self.simulator,
                **self.run_settings,
            )


