from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


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
    fabric_offsets: tuple = (4, 1)
    padding: tuple = (16, 16)
    arch: str = "wse3"
    simulator: bool = False
    disable_version_check: bool = True

    out_path: Path = Path("cluster_work/out")
    comment: str = ""

    # optional PE count limit; None = use full fabric
    num_pes_target: Optional[int] = None

    # grid — derived from fabric + num_pes_target in __post_init__, not overridable
    width: int = field(init=False)
    height: int = field(init=False)
    num_pes: int = field(init=False)

    # CSL array sizes — estimated from matrix params, can be overridden for real matrices
    total_nnz_per_pe: Optional[int] = None
    max_rows_per_pe: Optional[int] = None

    # set by sdk/compiler.compile() after CSL compilation
    artifact_id: str = field(init=False, default="")

    def __post_init__(self):
        max_w = self.fabric_dims[0] - self.fabric_offsets[0] - self.padding[0]
        max_h = self.fabric_dims[1] - self.fabric_offsets[1] - self.padding[1]

        if self.num_pes_target is not None:
            w = min(self.num_pes_target, max_w)
            h = min((self.num_pes_target + w - 1) // w, max_h)
            self.width = w
            self.height = h
        else:
            self.width = max_w
            self.height = max_h

        self.num_pes = self.width * self.height

        if self.total_nnz_per_pe is None and self.rows > 0:
            avg_nnz = int(self.rows * self.cols * self.density / self.num_pes)
            self.total_nnz_per_pe = max(avg_nnz + 5, 10)

        if self.max_rows_per_pe is None and self.rows > 0:
            self.max_rows_per_pe = self.rows // self.num_pes + 2

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
            elif f.name in ("fabric_dims", "fabric_offsets", "padding") and isinstance(v, list):
                v = tuple(v)
            kwargs[f.name] = v
        cfg = cls(**kwargs)
        if "artifact_id" in d:
            object.__setattr__(cfg, "artifact_id", d["artifact_id"])
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
