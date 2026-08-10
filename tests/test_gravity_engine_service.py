import json
import time
from pathlib import Path

import pytest

from sigma_theory_compiler.cli import main as cli_main
from sigma_theory_compiler.gravity_engine_service import (
    _hardware_telemetry,
    export_service,
    initialize_service,
    resume_service,
    run_service_worker,
    service_status,
    start_service,
)
from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.real_formula_execution import FiniteFormulaQueueRefill, cuda_available

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
REAL = ROOT / "configs" / "real_formula_execution_5090.json"
BINARY = ROOT / "configs" / "binary_formula_execution_5090.json"
MANIFEST = ROOT / "runs" / "knowledge-base" / "survivor-export-smoke.json"
SURVIVORS = ROOT / "runs" / "knowledge-base" / "survivors-smoke"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_hardware_telemetry_distinguishes_sensor_load_from_lane_occupancy() -> None:
    sample = _hardware_telemetry()
    assert "distinct from lease occupancy" in sample["semantics"]
    assert isinstance(sample["cpu"]["available"], bool)
    assert isinstance(sample["gpu"]["available"], bool)
    if sample["cpu"]["available"]:
        assert 0 <= sample["cpu"]["utilization_percent"] <= 100
        assert sample["cpu"]["logical_processors"] >= 1
    if sample["gpu"]["available"]:
        assert 0 <= sample["gpu"]["utilization_percent"] <= 100
        assert sample["gpu"]["memory_used_bytes"] <= sample["gpu"]["memory_total_bytes"]


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _execution(tmp_path: Path, *, gpu: bool = False) -> Path:
    config = _load(EXECUTION)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 4,
        "lease_seconds": 2,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": 8,
        "maximum_wall_seconds": 30,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 1}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 0 if gpu else 1,
        "gpu_workers": 1 if gpu else 0,
        "worker_poll_seconds": 0.01,
        "refill_interval_seconds": 0.01,
        "telemetry_interval_seconds": 0.02,
        "maximum_wall_seconds_per_run": 20,
        "shutdown_grace_seconds": 2,
        "maximum_telemetry_bytes": 4 * 1024 * 1024,
    }
    return _write(tmp_path / f"execution-{'gpu' if gpu else 'cpu'}.json", config)


def _real_adapter(tmp_path: Path, *, stop: int = 384) -> Path:
    config = _load(REAL)
    config["generator_config_path"] = str(GENERATOR)
    config["start_ordinal"] = 0
    config["stop_ordinal_exclusive"] = stop
    config["target_pending_batches"] = 1
    config["cpu_batch_candidates"] = 128
    config["gpu_batch_candidates"] = 128
    config["lane_cycle"] = ["cpu"]
    return _write(tmp_path / "real.json", config)


def _binary_adapter(tmp_path: Path) -> Path:
    config = _load(BINARY)
    config["manifest_path"] = str(MANIFEST)
    config["survivor_directory"] = str(SURVIVORS)
    config["generator_config_path"] = str(GENERATOR)
    config["start_export_block"] = 0
    config["stop_export_block_exclusive"] = 1
    config["target_pending_blocks"] = 1
    config["lane_cycle"] = ["gpu"]
    return _write(tmp_path / "binary.json", config)


def test_real_service_auto_refills_exports_and_builds_dashboard(tmp_path: Path) -> None:
    service = tmp_path / "real-service"
    result = start_service(
        service,
        _execution(tmp_path),
        PROFILE,
        _real_adapter(tmp_path),
        mode="real",
        foreground=True,
        maximum_tasks=4,
        maximum_wall_seconds=20,
        maximum_disk_bytes=64 * 1024 * 1024,
    )
    assert result["run"]["stop_reason"] == "queue_drained"
    assert result["run"]["refill_calls"] > 1
    status = service_status(service)
    assert status["state"] == "completed"
    assert status["source"]["exhausted"]
    assert status["execution"]["counts"]["succeeded"] == 3
    assert status["execution"]["paid_llm_calls_enabled"] is False
    assert status["cost_budget"]["maximum_paid_llm_spend_usd"] == 0
    assert (service / "dashboard.html").is_file()
    report = export_service(service, tmp_path / "engine-export.json")
    assert report["results"]["succeeded_work_items"] == 3
    assert report["results"]["processed_candidates"] == 384
    assert report["results"]["candidate_counts_by_lane"] == {"cpu": 384}
    assert sum(report["results"]["status_counts"].values()) == 384
    assert report["data_eligibility"]["dark_matter_or_halo_inputs"] is False
    assert cli_main(["engine-status", "--service-dir", str(service)]) == 0
    assert (
        cli_main(
            [
                "engine-export",
                "--service-dir",
                str(service),
                "--output",
                str(tmp_path / "cli-export.json"),
            ]
        )
        == 0
    )


def test_stop_request_then_resume_uses_same_database_and_cursor(tmp_path: Path) -> None:
    service = tmp_path / "resumable-service"
    initialize_service(
        service,
        _execution(tmp_path),
        PROFILE,
        _real_adapter(tmp_path, stop=256),
        mode="real",
        maximum_tasks=2,
        maximum_wall_seconds=20,
        maximum_disk_bytes=64 * 1024 * 1024,
    )
    (service / "stop.request").write_text("test\n", encoding="utf-8")
    stopped = run_service_worker(service)
    assert stopped["stop_reason"] == "external_stop_requested"
    assert service_status(service)["state"] == "stopped"

    resumed = resume_service(service, foreground=True)
    assert resumed["run"]["stop_reason"] == "queue_drained"
    assert resumed["status"]["state"] == "completed"
    assert resumed["status"]["execution"]["counts"]["succeeded"] == 2
    assert resumed["status"]["execution"]["checkpoint_sequence"] >= 2


def test_live_cuda_binary_service_is_bounded_and_rooted(tmp_path: Path) -> None:
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    service = tmp_path / "cuda-service"
    result = start_service(
        service,
        _execution(tmp_path, gpu=True),
        PROFILE,
        _binary_adapter(tmp_path),
        mode="binary",
        foreground=True,
        maximum_tasks=1,
        maximum_wall_seconds=20,
        maximum_disk_bytes=64 * 1024 * 1024,
    )
    assert result["run"]["stop_reason"] == "queue_drained"
    assert result["run"]["utilization"]["gpu"]["peak"] == 1.0
    exported = export_service(service, tmp_path / "cuda-export.json")
    assert exported["results"]["backend_counts"] == {"gpu_cupy_binary_cached": 1}
    assert exported["results"]["processed_candidates"] == 3_272
    assert sum(exported["results"]["status_counts"].values()) == 3_272
    assert exported["results"]["status_roots_root_sha256"] != "0" * 64
    assert result["status"]["execution"]["lanes"]["gpu"]["capacity"] == 1


def test_service_rejects_ineligible_adapter_before_database_creation(tmp_path: Path) -> None:
    adapter = _load(REAL)
    adapter["generator_config_path"] = str(GENERATOR)
    adapter["data_eligibility"]["redshift_distance_inputs"] = True
    adapter_path = _write(tmp_path / "ineligible.json", adapter)
    service = tmp_path / "rejected"
    with pytest.raises(ValueError, match="eligibility"):
        initialize_service(
            service,
            _execution(tmp_path),
            PROFILE,
            adapter_path,
            mode="real",
        )
    assert not (service / "engine.sqlite").exists()


def test_service_resume_recovers_an_expired_worker_lease(tmp_path: Path) -> None:
    service = tmp_path / "lease-recovery"
    initialize_service(
        service,
        _execution(tmp_path),
        PROFILE,
        _real_adapter(tmp_path, stop=256),
        mode="real",
        maximum_tasks=2,
        maximum_wall_seconds=20,
        maximum_disk_bytes=64 * 1024 * 1024,
    )
    execution = _load(service / "execution-config.json")
    resource = _load(service / "resource-profile.json")
    adapter = _load(service / "adapter-config.json")
    coordinator = PersistentParallelSearch(service / "engine.sqlite", execution, resource)
    source = FiniteFormulaQueueRefill(coordinator, adapter)
    assert source.refill()["accepted_batches"] == 1
    abandoned = coordinator.claim("cpu", "crashed-worker", lease_seconds=-1)
    assert abandoned and abandoned.attempt == 1

    resumed = resume_service(service, foreground=True)
    assert resumed["run"]["stop_reason"] == "queue_drained"
    assert resumed["status"]["execution"]["recovered_leases"] == 1
    with coordinator.connect() as connection:
        row = connection.execute(
            "SELECT attempt,state FROM work WHERE ordinal=0 AND lane='cpu'"
        ).fetchone()
    assert dict(row) == {"attempt": 2, "state": "succeeded"}


def test_detached_service_process_reaches_completed_state(tmp_path: Path) -> None:
    service = tmp_path / "detached"
    started = start_service(
        service,
        _execution(tmp_path),
        PROFILE,
        _real_adapter(tmp_path, stop=64),
        mode="real",
        foreground=False,
        maximum_tasks=1,
        maximum_wall_seconds=20,
        maximum_disk_bytes=64 * 1024 * 1024,
    )
    assert started["pid"] > 0
    deadline = time.monotonic() + 15
    status = service_status(service, write_artifacts=False)
    while status["state"] not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(0.05)
        status = service_status(service, write_artifacts=False)
    assert status["state"] == "completed", (service / "service.log").read_text(
        encoding="utf-8"
    )
    assert status["execution"]["counts"]["succeeded"] == 1
