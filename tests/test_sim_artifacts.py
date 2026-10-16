"""Simulator evidence comes back even when the run fails.

A simulator deadlock raises out of SdkLauncher.run(), and sim.log, sim_stats.json
and the simfab traces are what diagnose it, so sdk/launcher.run downloads them
in a finally.  Traces are requested only on simulator runs: hardware never sees
--sim-trace, so SdkRuntime never gets a kwarg it might reject.
"""

import sys
import types
from pathlib import Path

import pytest

from config import SpMVConfig
from sdk import launcher as launcher_mod
from sdk.launcher import SIM_ARTIFACTS, build_worker_command

from tests.worker_src import worker_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "cluster" / "worker_runtime.py"


def _cfg(tmp_path, simulator):
    return SpMVConfig(rows=1000, cols=1000, density=0.01, num_pes_target=16,
                      simulator=simulator, out_path=tmp_path / "work")


class FakeLauncher:
    """Stands in for cerebras.sdk.client.SdkLauncher."""

    def __init__(self, raise_in_run, present):
        self.raise_in_run = raise_in_run
        self.present = set(present)
        self.downloads = []

    def __call__(self, *args, **kwargs):      # SdkLauncher(artifact_id, ...)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stage(self, path):
        pass

    def run(self, cmd):
        self.cmd = cmd
        if self.raise_in_run:
            raise RuntimeError("Stopped due to idleness")
        return "ok"

    def download_artifact(self, name, out_path):
        self.downloads.append(name)
        if name not in self.present:
            raise FileNotFoundError(name)
        Path(out_path).write_text("x" if name != "benchmark_results.csv"
                                  else "timestamp\n")


@pytest.fixture
def fake_sdk(monkeypatch):
    def install(**kw):
        fake = FakeLauncher(**kw)
        client = types.ModuleType("cerebras.sdk.client")
        client.SdkLauncher = fake
        for name, mod in (("cerebras", types.ModuleType("cerebras")),
                          ("cerebras.sdk", types.ModuleType("cerebras.sdk")),
                          ("cerebras.sdk.client", client)):
            monkeypatch.setitem(sys.modules, name, mod)
        return fake
    return install


def test_simulator_runs_request_traces_and_hardware_never_does(tmp_path):
    assert " --sim-trace" in build_worker_command(_cfg(tmp_path, True))
    assert "--sim-trace" not in build_worker_command(_cfg(tmp_path, False))


def test_worker_accepts_the_flag_and_only_then_unsuppresses_traces():
    src = worker_source()
    assert '"--sim-trace", action="store_true"' in src
    assert 'if args.sim_trace:\n            runtime_kwargs["suppress_simfab_trace"] = False' in src
    assert "SdkRuntimeWrapper(SdkRuntime, args.name, **runtime_kwargs)" in src


def test_artifacts_download_even_when_the_run_raises(tmp_path, fake_sdk):
    fake = fake_sdk(raise_in_run=True,
                    present={"sim.log", "sim_stats.json", "out/simfab_traces"})
    cfg = _cfg(tmp_path, True)
    with pytest.raises(RuntimeError, match="idleness"):
        launcher_mod.run(cfg, download_path=tmp_path / "res.csv")
    sim = tmp_path / "work" / "sim"
    assert (sim / "sim.log").exists()
    assert (sim / "sim_stats.json").exists()
    assert (sim / "out_simfab_traces").exists()
    # The benchmark download was attempted too, and its absence did not mask
    # the original error.
    assert "benchmark_results.csv" in fake.downloads


def test_every_artifact_is_tried_in_both_locations(tmp_path, fake_sdk):
    fake = fake_sdk(raise_in_run=False, present=set())
    launcher_mod.run(_cfg(tmp_path, True), download_path=tmp_path / "res.csv")
    for name in SIM_ARTIFACTS:
        assert name in fake.downloads and f"out/{name}" in fake.downloads


def test_hardware_runs_fetch_no_sim_artifacts(tmp_path, fake_sdk):
    fake = fake_sdk(raise_in_run=False, present=set())
    launcher_mod.run(_cfg(tmp_path, False), download_path=tmp_path / "res.csv")
    assert fake.downloads == ["progress.log", "benchmark_results.csv"]   # no sim artifacts
    assert "--sim-trace" not in fake.cmd
