import json
from pathlib import Path

import pytest

from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.real_formula_execution import (
    FiniteFormulaQueueRefill,
    configure_real_evaluators,
    cpu_formula_batch_evaluator,
    cuda_available,
    gpu_formula_batch_evaluator,
    run_finite_formula_search,
    validate_formula_batch_result,
)

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
ADAPTER = ROOT / "configs" / "real_formula_execution_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution() -> dict:
    config = _load(EXECUTION)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 16,
        "lease_seconds": 10,
        "checkpoint_every_completions": 2,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": 16,
        "maximum_wall_seconds": 60,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 1}
    config["gpu"] = {**config["gpu"], "maximum_batch_candidates": 4096}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 1,
        "gpu_workers": 1,
        "worker_poll_seconds": 0.01,
        "telemetry_interval_seconds": 0.02,
        "maximum_wall_seconds_per_run": 15,
        "shutdown_grace_seconds": 2,
    }
    return config


def _adapter(stop: int = 512) -> dict:
    config = _load(ADAPTER)
    config["generator_config_path"] = str(GENERATOR)
    config["stop_ordinal_exclusive"] = stop
    config["target_pending_batches"] = 4
    config["cpu_batch_candidates"] = 128
    config["gpu_batch_candidates"] = 128
    config["lane_cycle"] = ["cpu", "gpu"]
    return config


def _lease(coordinator: PersistentParallelSearch, lane: str):
    lease = coordinator.claim(lane, f"direct-{lane}", lease_seconds=30)
    assert lease
    return lease


def test_real_cpu_batch_is_deterministic_and_schema_validated(tmp_path: Path) -> None:
    execution = _execution()
    execution["supervisor"]["gpu_workers"] = 0
    coordinator = PersistentParallelSearch(tmp_path / "cpu.sqlite", execution, _load(PROFILE))
    adapter = _adapter(stop=64)
    adapter["lane_cycle"] = ["cpu"]
    adapter["cpu_batch_candidates"] = 64
    refill = FiniteFormulaQueueRefill(coordinator, adapter)
    assert refill.refill()["accepted_batches"] == 1
    lease = _lease(coordinator, "cpu")
    first = cpu_formula_batch_evaluator(lease)
    second = cpu_formula_batch_evaluator(lease)
    validate_formula_batch_result(first, lease.payload)
    assert first["status_root_sha256"] == second["status_root_sha256"]
    assert first["counts"] == second["counts"]
    assert sum(first["counts"].values()) == 64
    assert first["data_eligibility"]["passed"]


def test_real_gpu_kernel_matches_cpu_status_root_when_cuda_available(tmp_path: Path) -> None:
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    execution = _execution()
    coordinator = PersistentParallelSearch(tmp_path / "gpu.sqlite", execution, _load(PROFILE))
    adapter = _adapter(stop=4096)
    adapter["lane_cycle"] = ["gpu"]
    adapter["gpu_batch_candidates"] = 4096
    refill = FiniteFormulaQueueRefill(coordinator, adapter)
    assert refill.refill()["accepted_batches"] == 1
    lease = _lease(coordinator, "gpu")
    gpu = gpu_formula_batch_evaluator(lease)
    cpu = cpu_formula_batch_evaluator(lease)
    assert gpu["status_root_sha256"] == cpu["status_root_sha256"]
    assert gpu["counts"] == cpu["counts"]
    assert gpu["backend"] == "gpu_cupy_raw_kernel"
    assert gpu["candidates_per_second"] > 0


def test_finite_generator_auto_refills_real_cpu_and_gpu_supervisor(tmp_path: Path) -> None:
    available, _ = cuda_available()
    adapter = _adapter(stop=256)
    execution = _execution()
    if not available:
        adapter["lane_cycle"] = ["cpu"]
        execution["supervisor"]["gpu_workers"] = 0
    report = run_finite_formula_search(
        tmp_path / "end-to-end.sqlite",
        execution,
        _load(PROFILE),
        adapter,
        tmp_path / "formula-telemetry.jsonl",
        maximum_waves=4,
    )
    assert report["generator_exhausted"]
    assert report["all_work_succeeded"]
    assert report["processed_candidates"] == 256
    assert report["valid_result_batches"] == 2
    assert report["paid_llm_calls_made"] == 0
    assert report["data_eligibility_passed"]
    assert set(report["backend_batch_counts"]) <= {
        "cpu_numpy",
        "gpu_cupy_raw_kernel",
    }


def test_generator_refill_rejects_ineligible_data_contract(tmp_path: Path) -> None:
    execution = _execution()
    coordinator = PersistentParallelSearch(tmp_path / "reject.sqlite", execution, _load(PROFILE))
    adapter = _adapter(stop=16)
    adapter["data_eligibility"]["dark_matter_or_halo_inputs"] = True
    with pytest.raises(ValueError, match="eligibility"):
        FiniteFormulaQueueRefill(coordinator, adapter)

    bad_generator = _load(GENERATOR)
    bad_generator["observational_data_opened"] = True
    path = tmp_path / "opened.json"
    path.write_text(json.dumps(bad_generator), encoding="utf-8")
    adapter = _adapter(stop=16)
    adapter["generator_config_path"] = str(path)
    with pytest.raises(ValueError, match="observational data"):
        FiniteFormulaQueueRefill(coordinator, adapter)


def test_real_evaluator_configuration_does_not_spawn_unused_lane_owners() -> None:
    execution = _execution()
    gpu_only = _adapter(stop=16)
    gpu_only["lane_cycle"] = ["gpu"]
    assert configure_real_evaluators(execution, gpu_only)["supervisor"]["cpu_workers"] == 0
    cpu_only = _adapter(stop=16)
    cpu_only["lane_cycle"] = ["cpu"]
    assert configure_real_evaluators(execution, cpu_only)["supervisor"]["gpu_workers"] == 0
