"""TOML sweep configuration: load, expand, and generate sweep configs."""

import math
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

from config import SpMVConfig

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

    @classmethod
    def from_toml(cls, path: Path) -> "SweepConfig":
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

        if "sweep" in doc:
            s = doc["sweep"]
            sweep_type = s["type"]
            num_pes_spec = NumPesSpec.from_dict(s.get("num_pes", {}))
            sweep_params = {k: v for k, v in s.items() if k not in ("type", "num_pes")}

        if "run" in doc:
            explicit_runs = doc["run"]

        return cls(
            name=name,
            description=description,
            work_dir=work_dir,
            results_template=results_template,
            sweep_type=sweep_type,
            sweep_params=sweep_params,
            num_pes_spec=num_pes_spec,
            explicit_runs=explicit_runs,
        )

    def results_path(self) -> Path:
        today = date.today().isoformat()
        return Path(self.results_template.format(date=today, name=self.name))

    def expand(self) -> Iterator[SpMVConfig]:
        """Yield one SpMVConfig per run."""
        if self.explicit_runs:
            for r in self.explicit_runs:
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
            )


