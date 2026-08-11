from __future__ import annotations

import json
import time
from itertools import pairwise
from pathlib import Path

import pytest

import sigma_theory_compiler.cpu_symbolic_overlap_benchmark as campaign

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cpu_symbolic_overlap_benchmark.json"
ARTIFACT = ROOT / "runs/benchmarks/cpu-real-formula-overlap-15-16.json"


def _fake_stage(workers: int, *, peak: float = 85.0) -> dict:
    counts = {"reject": campaign.UNIQUE_FORMULAS, "pass": 0, "ambiguous": 0}
    body = {
        "workers": workers,
        "batch_count": campaign.FIXED_SHARD_COUNT,
        "fixed_shard_count": campaign.FIXED_SHARD_COUNT,
        "fixed_shard_size": campaign.FIXED_SHARD_SIZE,
        "backend": "cpu_numpy",
        "gpu_workers": 0,
        "interval": {"start": campaign.START, "stop": campaign.STOP},
        "unique_formula_count": campaign.UNIQUE_FORMULAS,
        "candidate_grid_evaluations": campaign.CANDIDATE_GRID_EVALUATIONS,
        "signed_term_hessian_accumulations": campaign.SIGNED_TERM_HESSIAN_ACCUMULATIONS,
        "counts": counts,
        "ambiguous_are_not_passes": True,
        "exact_gap_overlap_duplicate_free_coverage": True,
        "all_evaluator_controls_passed": True,
        "elapsed_seconds": 10.0,
        "unique_formulas_per_second": campaign.UNIQUE_FORMULAS / 10,
        "candidate_grid_evaluations_per_second": campaign.CANDIDATE_GRID_EVALUATIONS / 10,
        "partition_independent_status_root_sha256": "a" * 64,
        "fixed_shard_manifest_root_sha256": "b" * 64,
        "reported_margin_minimum": -1.0,
        "worker_cpu_seconds": 100.0,
        "worker_cpu_capacity_percent": 41.67,
        "cpu_sample_count": 4,
        "cpu_sampling_contract": "fixed_interval_blocking_device_wide_psutil",
        "cpu_sample_interval_seconds": 0.1,
        "cpu_percent_mean": 82.0,
        "cpu_percent_median": 82.0,
        "cpu_percent_peak": peak,
        "minimum_available_ram_mib": 40_000,
        "cpu_target_percent": 80,
        "cpu_target_met_by_median": True,
        "backoff_threshold_exceeded": peak > 92,
        "hard_deadline_enforced": True,
        "hard_deadline_triggered": False,
        "owned_worker_termination_count": 0,
        "wall_bound_exceeded": False,
    }
    return {**body, "content_sha256": campaign._content_sha(body)}


def test_config_and_artifact_validate() -> None:
    config, _ = campaign.load_config(CONFIG)
    assert config["allowlisted_evaluator"] == campaign.EVALUATOR
    assert config["gpu_workers"] == 0
    assert config["start_ordinal"] == 1_000_000_000
    assert config["stop_ordinal_exclusive"] == 1_000_065_536
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    campaign.validate_artifact(result, CONFIG)
    assert result["coverage"]["unique_formula_count"] == 65_536
    assert result["coverage"]["candidate_grid_evaluations"] == 22_478_848
    assert result["coverage"]["signed_term_hessian_accumulations"] == 134_873_088
    assert result["scientific_pass"] is False
    assert result["resource_policy_promoted"] is False


def test_fixed_shards_are_gap_overlap_and_worker_partition_independent() -> None:
    shards = campaign._fixed_shards()
    assert len(shards) == campaign.FIXED_SHARD_COUNT
    assert shards[0][0] == campaign.START
    assert shards[-1][1] == campaign.STOP
    assert sum(stop - start for start, stop in shards) == campaign.UNIQUE_FORMULAS
    assert all(left[1] == right[0] for left, right in pairwise(shards))
    assert all(stop - start == campaign.FIXED_SHARD_SIZE for start, stop in shards)


def test_small_direct_real_cpu_evaluator_replay() -> None:
    config, root = campaign.load_config(CONFIG)
    generator = root / config["bindings"]["generator_config"]["path"]
    result = campaign._formula_batch_job(
        0,
        campaign.START,
        campaign.START + 8,
        str(generator),
        config["bindings"]["generator_config"]["file_sha256"],
        config["ambiguity_guard"],
    )
    assert result["backend"] == "cpu_numpy"
    assert sum(result["counts"].values()) == 8
    assert result["batch"] == {
        "start_ordinal": campaign.START,
        "end_ordinal_exclusive": campaign.START + 8,
        "candidate_count": 8,
    }
    assert result["data_eligibility"]["passed"] is True


def test_stage_16_is_admitted_only_after_stage_15_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[int] = []

    def fake_run(_config: dict, _root: Path, workers: int, _deadline: float) -> dict:
        called.append(workers)
        return _fake_stage(workers)

    monkeypatch.setattr(campaign, "_run_stage", fake_run)
    monkeypatch.setattr(
        campaign,
        "_resource_sample",
        lambda *_args: {"cpu_percent": 70.0, "available_ram_mib": 40_000},
    )
    result = campaign.execute_campaign(CONFIG)
    assert called == [15, 16]
    assert result["stage_16_admitted"] is True
    assert result["cpu_target_met"] is True


def test_stage_16_backs_off_above_92_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[int] = []

    def fake_run(_config: dict, _root: Path, workers: int, _deadline: float) -> dict:
        called.append(workers)
        return _fake_stage(workers, peak=93.0)

    monkeypatch.setattr(campaign, "_run_stage", fake_run)
    result = campaign.execute_campaign(CONFIG)
    assert called == [15]
    assert result["stage_16_admitted"] is False
    assert result["stage_16_blocker"] == "stage_15_control_or_resource_guard"


def test_stage_16_racing_preflight_backoff_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(_config: dict, _root: Path, workers: int, _deadline: float) -> dict:
        if workers == 16:
            raise RuntimeError("CPU formula overlap preflight backoff threshold exceeded")
        return _fake_stage(workers)

    monkeypatch.setattr(campaign, "_run_stage", fake_run)
    monkeypatch.setattr(
        campaign,
        "_resource_sample",
        lambda *_args: {"cpu_percent": 70.0, "available_ram_mib": 40_000},
    )
    result = campaign.execute_campaign(CONFIG)
    assert [stage["workers"] for stage in result["stages"]] == [15]
    assert result["stage_16_admitted"] is False
    assert result["stage_16_blocker"] == "cpu_backoff"


def test_ram_floor_fails_before_pool_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = campaign.load_config(CONFIG)
    monkeypatch.setattr(
        campaign,
        "_resource_sample",
        lambda *_args: {"cpu_percent": 10.0, "available_ram_mib": 32_767},
    )
    with pytest.raises(RuntimeError, match="RAM floor"):
        campaign._run_stage(config, root, 15)


def _rehash(result: dict, stage_index: int | None = None) -> dict:
    if stage_index is not None:
        result["stages"][stage_index]["content_sha256"] = campaign._content_sha(
            result["stages"][stage_index]
        )
    result["content_sha256"] = campaign._content_sha(result)
    return result


@pytest.mark.parametrize(
    "mutator,stage_index",
    [
        (lambda row: row.__setitem__("theory_pass", True), None),
        (lambda row: row.__setitem__("interpretation", "proves theory and observations pass"), None),
        (lambda row: row["coverage"]["interval"].__setitem__("start", 1), None),
        (
            lambda row: row["contract"].__setitem__("allowlisted_evaluator", "synthetic_cpu"),
            None,
        ),
        (
            lambda row: row.__setitem__("cpu_target_met", not row["cpu_target_met"]),
            None,
        ),
        (lambda row: row.__setitem__("decision", "promoted"), None),
        (
            lambda row: row["stages"][0].__setitem__("candidate_grid_evaluations", 1),
            0,
        ),
        (
            lambda row: row["stages"][0].__setitem__("worker_cpu_capacity_percent", 100),
            0,
        ),
        (lambda row: row["stages"][0].__setitem__("cpu_sample_count", 99), 0),
        (lambda row: row["stages"][0].__setitem__("cpu_percent_mean", 1.0), 0),
        (
            lambda row: row["stages"][0].__setitem__("hard_deadline_enforced", False),
            0,
        ),
        (
            lambda row: row["stages"][-1]["counts"].__setitem__("reject", 65_535),
            -1,
        ),
        (
            lambda row: row["stages"][-1].__setitem__(
                "partition_independent_status_root_sha256", "f" * 64
            ),
            -1,
        ),
        (
            lambda row: row["hardware_attestation"].__setitem__("logical_processors", 23),
            None,
        ),
    ],
)
def test_rehashed_semantic_tampering_fails_closed(mutator, stage_index) -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    mutator(result)
    _rehash(result, stage_index)
    with pytest.raises(ValueError):
        campaign.validate_artifact(result, CONFIG)


def test_artifact_claim_tamper_and_config_evaluator_injection_fail(tmp_path: Path) -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    result["scientific_pass"] = True
    _rehash(result)
    with pytest.raises(ValueError, match="forbidden claim"):
        campaign.validate_artifact(result, CONFIG)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["allowlisted_evaluator"] = "synthetic_cpu"
    bad = tmp_path / "configs/cpu_symbolic_overlap_benchmark.json"
    bad.parent.mkdir()
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        campaign.load_config(bad)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["ambiguity_guard"] = 1e-6
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        campaign.load_config(bad)


def test_hard_deadline_terminates_only_owned_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = campaign.load_config(CONFIG)
    bounded = dict(config)
    bounded.update(
        {
            "maximum_stage_seconds": 0.15,
            "shutdown_reserve_seconds": 0.05,
            "sample_interval_seconds": 0.01,
            "minimum_available_ram_mib": 0,
            "cpu_backoff_above_percent": 100,
        }
    )

    class FakeProcess:
        alive = True
        terminated = False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False
            self.terminated = True

        def join(self, timeout: float) -> None:
            assert timeout <= 0.5

    process = FakeProcess()

    class FakeExecutor:
        def __init__(self, **_kwargs) -> None:
            self._processes = {1: process}

        def submit(self, *_args):
            from concurrent.futures import Future

            return Future()

        def shutdown(self, **_kwargs) -> None:
            return None

    def sample(interval: float = 0.0) -> dict:
        time.sleep(interval)
        return {"cpu_percent": 10.0, "available_ram_mib": 40_000}

    monkeypatch.setattr(campaign.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(campaign, "_resource_sample", sample)
    monkeypatch.setattr(
        campaign,
        "_hardware_attestation",
        lambda *_args: {"logical_processors": 24},
    )
    stage = campaign._run_stage(bounded, root, 15)
    assert stage["hard_deadline_triggered"] is True
    assert stage["owned_worker_termination_count"] == 1
    assert stage["all_evaluator_controls_passed"] is False
    assert stage["elapsed_seconds"] <= bounded["maximum_stage_seconds"]
    assert process.terminated is True


def test_source_has_no_sqlite_gpu_external_signal_or_runtime_injection_surface() -> None:
    source = (ROOT / "src/sigma_theory_compiler/cpu_symbolic_overlap_benchmark.py").read_text()
    lowered = source.lower()
    assert "import sqlite" not in lowered
    assert "import cupy" not in lowered
    assert "torch.cuda" not in lowered
    assert "os.kill" not in lowered
    assert source.count("process.terminate()") == 1
    assert "popen(" not in lowered
    assert "executor=" not in lowered
    config = json.loads(CONFIG.read_text())
    assert config["gpu_workers"] == 0
    assert config["seals"]["sqlite_access"] is False
    assert config["seals"]["existing_process_signaled"] is False
