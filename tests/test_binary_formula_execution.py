import json
from pathlib import Path

import pytest

from sigma_theory_compiler.binary_formula_execution import (
    BinaryBlockQueueRefill,
    cpu_binary_block_evaluator,
    gpu_binary_block_evaluator,
    run_binary_block_search,
    validate_binary_result,
)
from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.real_formula_execution import cuda_available

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
ADAPTER = ROOT / "configs" / "binary_formula_execution_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
MANIFEST = ROOT / "runs" / "knowledge-base" / "survivor-export-smoke.json"
SURVIVORS = ROOT / "runs" / "knowledge-base" / "survivors-smoke"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution() -> dict:
    config = _load(EXECUTION)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 4,
        "lease_seconds": 30,
        "checkpoint_every_completions": 2,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": 4,
        "maximum_wall_seconds": 60,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 1}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 1,
        "gpu_workers": 1,
    }
    return config


def _adapter(stop: int = 1, lanes: list[str] | None = None) -> dict:
    config = _load(ADAPTER)
    config["manifest_path"] = str(MANIFEST)
    config["survivor_directory"] = str(SURVIVORS)
    config["generator_config_path"] = str(GENERATOR)
    config["stop_export_block_exclusive"] = stop
    config["target_pending_blocks"] = stop
    config["lane_cycle"] = lanes or ["cpu"]
    config["equivalence_samples_per_block"] = 24
    return config


def _lease(coordinator: PersistentParallelSearch, lane: str):
    lease = coordinator.claim(lane, f"direct-{lane}", lease_seconds=30)
    assert lease
    return lease


def test_rust_binary_block_is_restart_safe_and_ordinal_equivalent(tmp_path: Path) -> None:
    execution = _execution()
    execution["supervisor"]["gpu_workers"] = 0
    coordinator = PersistentParallelSearch(tmp_path / "binary.sqlite", execution, _load(PROFILE))
    refill = BinaryBlockQueueRefill(coordinator, _adapter())
    report = refill.refill()
    assert report["accepted_blocks"] == 1
    assert report["cursor"]["exhausted"]

    # This recreates the only unsafe interruption window: enqueue committed but
    # the cursor update did not. The deterministic duplicate advances on restart.
    with coordinator.connect() as connection:
        connection.execute(
            "UPDATE binary_block_cursor SET next_position=0,sequence=0,lane_index=0 "
            "WHERE source_id=?",
            (refill.source_id,),
        )
    resumed = BinaryBlockQueueRefill(coordinator, _adapter())
    lease = _lease(coordinator, "cpu")
    first = cpu_binary_block_evaluator(lease)
    second = cpu_binary_block_evaluator(lease)
    assert coordinator.finish(lease, "direct-cpu", first)
    duplicate = resumed.refill()
    assert duplicate["duplicate_blocks"] == 1
    assert duplicate["cursor"]["exhausted"]

    validate_binary_result(first, lease.payload)
    assert first["ordinal_equivalence"]["checked"] == 24
    assert first["ordinal_equivalence"]["all_equal"]
    assert first["status_root_sha256"] == second["status_root_sha256"]
    assert first["counts"] == second["counts"]
    assert sum(first["counts"].values()) == 3272
    assert first["data_eligibility"]["paid_llm_calls"] is False


def test_cached_gpu_binary_path_matches_cpu_exactly(tmp_path: Path) -> None:
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    coordinator = PersistentParallelSearch(
        tmp_path / "gpu-binary.sqlite", _execution(), _load(PROFILE)
    )
    refill = BinaryBlockQueueRefill(coordinator, _adapter(lanes=["gpu"]))
    assert refill.refill()["accepted_blocks"] == 1
    lease = _lease(coordinator, "gpu")
    first_gpu = gpu_binary_block_evaluator(lease)
    second_gpu = gpu_binary_block_evaluator(lease)
    cpu = cpu_binary_block_evaluator(lease)
    assert first_gpu["status_root_sha256"] == cpu["status_root_sha256"]
    assert first_gpu["counts"] == cpu["counts"]
    assert second_gpu["cuda_assets_reused"] is True
    assert second_gpu["status_root_sha256"] == first_gpu["status_root_sha256"]
    assert second_gpu["records_per_second"] > 0


def test_hash_tamper_and_ineligible_source_fail_closed(tmp_path: Path) -> None:
    execution = _execution()
    execution["supervisor"]["gpu_workers"] = 0
    coordinator = PersistentParallelSearch(tmp_path / "tamper.sqlite", execution, _load(PROFILE))
    adapter = _adapter()
    refill = BinaryBlockQueueRefill(coordinator, adapter)
    refill.refill()
    lease = _lease(coordinator, "cpu")
    original = Path(lease.payload["block_path"])
    copied = tmp_path / original.name
    copied.write_bytes(original.read_bytes() + b"tampered")
    lease.payload["block_path"] = str(copied)
    with pytest.raises(ValueError, match="size mismatch"):
        cpu_binary_block_evaluator(lease)

    adapter = _adapter()
    adapter["data_eligibility"]["redshift_distance_inputs"] = True
    with pytest.raises(ValueError, match="eligibility"):
        BinaryBlockQueueRefill(coordinator, adapter)


def test_persistent_gpu_owner_consumes_real_binary_blocks(tmp_path: Path) -> None:
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    execution = _execution()
    execution["supervisor"] = {
        **execution["supervisor"],
        "cpu_workers": 0,
        "worker_poll_seconds": 0.01,
        "telemetry_interval_seconds": 0.02,
        "maximum_wall_seconds_per_run": 15,
        "shutdown_grace_seconds": 2,
    }
    report = run_binary_block_search(
        tmp_path / "persistent-gpu.sqlite",
        execution,
        _load(PROFILE),
        _adapter(stop=2, lanes=["gpu"]),
        tmp_path / "binary-telemetry.jsonl",
    )
    assert report["cursor"]["exhausted"]
    assert report["all_work_succeeded"]
    assert report["valid_result_blocks"] == 2
    assert report["processed_records"] == 3650
    assert report["backend_block_counts"] == {"gpu_cupy_binary_cached": 2}
    assert report["supervisor"]["process_starts"] == 1
    assert report["paid_llm_calls_made"] == 0
