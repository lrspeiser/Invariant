from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import sigma_theory_compiler.kastner_schlatter_deferred_gpu_handoff_service as handoff

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_deferred_gpu_handoff_service.json"
READINESS = ROOT / "runs/engine/kastner-schlatter-deferred-gpu-handoff-readiness.json"


def _sample(utilization: int, free_mib: int) -> dict:
    return {
        "device_index": 0,
        "device_name": "NVIDIA GeForce RTX 5090",
        "gpu_utilization_percent": utilization,
        "memory_used_mib": 32607 - free_mib,
        "memory_free_mib": free_mib,
        "memory_total_mib": 32607,
        "power_watts": 100.0,
        "sampled_utc": "synthetic-control",
        "scope": "deterministic fake NVML control",
    }


def _scheduler_result(*_: object, **__: object) -> dict:
    config, root = handoff.load_config(CONFIG)
    scheduler = json.loads((root / config["bindings"]["scheduler_config"]["path"]).read_text())
    runtime = root / scheduler["runtime_directory"]
    return {
        "enqueue": {"inserted": 1, "deduplicated": 0},
        "supervisor": {
            "schema_version": "sigma-parallel-supervisor-run-1.0",
            "stop_reason": "queue_drained",
            "final_telemetry": {
                "counts": {"succeeded": 1},
                "queue": {"pending": 0},
            },
        },
        "database": str(runtime / scheduler["database_name"]),
        "telemetry": str(runtime / scheduler["telemetry_name"]),
        "immutable_artifact_written": False,
        "live_campaign_sqlite_accessed": False,
    }


def _run(
    tmp_path: Path,
    *,
    cycles: int = 1,
    polls: int = 3,
) -> dict:
    return handoff.run_service(
        CONFIG,
        runtime_override=tmp_path / "service",
        gpu_owner_runtime_override=tmp_path / "owner",
        maximum_cycles_override=cycles,
        maximum_wait_polls_override=polls,
    )


def test_readiness_validates_without_starting_service() -> None:
    result = json.loads(READINESS.read_text(encoding="utf-8"))
    handoff.validate_readiness(result, CONFIG)
    audit = result["current_runtime_audit"]
    assert audit["service_started_by_readiness"] is False
    assert audit["gpu_owner_reserved_by_readiness"] is False
    assert audit["scheduler_started_by_readiness"] is False
    assert result["handoff_contract"]["gpu_workers"] == 1
    assert result["handoff_contract"]["cpu_workers"] == 0
    assert result["handoff_contract"]["post_reservation_nvml_safe_recheck"] is True
    assert result["handoff_contract"]["completed_queue_resume_is_idempotent"] is True
    assert result["handoff_contract"]["atomic_checkpoint"] == "handoff-service-checkpoint.json"


def test_fake_nvml_three_safe_samples_handoff_to_fixed_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = iter([_sample(5, 9000)] * 4)
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(handoff.deferred, "sample_nvml", lambda: next(samples))
    monkeypatch.setattr(handoff.time, "sleep", lambda _: None)

    def fake_scheduler(*args: object, **kwargs: object) -> dict:
        calls.append((args, kwargs))
        return _scheduler_result()

    monkeypatch.setattr(handoff, "run_scheduler", fake_scheduler)
    result = _run(tmp_path)
    assert result["state"] == "completed"
    assert result["attempted_cycles"] == 1
    assert result["executed_cycles"] == 1
    assert len(calls) == 1
    assert calls[0][1] == {"maximum_wall_seconds": 120.0}
    assert result["last_scheduler_receipt"]["reviewed_workload_succeeded"] is True
    assert not (tmp_path / "service/handoff-service.lease.json").exists()
    assert not (tmp_path / "owner/deferred-gpu-owner.lease.json").exists()
    assert not list(tmp_path.rglob("*.sqlite*"))


def test_unsafe_sample_resets_consecutive_safety_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = iter(
        [
            _sample(5, 9000),
            _sample(99, 9000),
            _sample(5, 9000),
            _sample(5, 9000),
            _sample(5, 9000),
            _sample(5, 9000),
        ]
    )
    monkeypatch.setattr(handoff.deferred, "sample_nvml", lambda: next(samples))
    monkeypatch.setattr(handoff.time, "sleep", lambda _: None)
    monkeypatch.setattr(handoff, "run_scheduler", _scheduler_result)
    result = _run(tmp_path, polls=5)
    assert result["state"] == "completed"
    assert result["polls_in_cycle"] == 5


def test_busy_device_times_out_without_owner_or_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False
    monkeypatch.setattr(handoff.deferred, "sample_nvml", lambda: _sample(99, 8000))
    monkeypatch.setattr(handoff.time, "sleep", lambda _: None)

    def forbidden(*_: object, **__: object) -> dict:
        nonlocal called
        called = True
        raise AssertionError("scheduler must remain sealed")

    monkeypatch.setattr(handoff, "run_scheduler", forbidden)
    result = _run(tmp_path)
    assert result["state"] == "stopped"
    assert result["attempted_cycles"] == 1
    assert result["executed_cycles"] == 0
    assert called is False
    assert not (tmp_path / "owner").exists()


def test_stop_request_is_observed_before_nvml_or_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "service"
    runtime.mkdir()
    (runtime / "handoff-service.stop.request").write_text("stop\n")
    monkeypatch.setattr(
        handoff.deferred,
        "sample_nvml",
        lambda: pytest.fail("NVML must not be sampled after stop"),
    )
    monkeypatch.setattr(handoff, "run_scheduler", lambda *_a, **_k: pytest.fail("executor opened"))
    result = _run(tmp_path)
    assert result["state"] == "stopped"
    assert result["attempted_cycles"] == 0
    assert result["executed_cycles"] == 0


def test_checkpoint_resume_keeps_cycle_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handoff.deferred, "sample_nvml", lambda: _sample(99, 8000))
    monkeypatch.setattr(handoff.time, "sleep", lambda _: None)
    first = _run(tmp_path)
    assert first["attempted_cycles"] == 1
    samples = iter([_sample(1, 10_000)] * 4)
    monkeypatch.setattr(handoff.deferred, "sample_nvml", lambda: next(samples))
    monkeypatch.setattr(handoff, "run_scheduler", _scheduler_result)
    resumed = _run(tmp_path, cycles=2)
    assert resumed["state"] == "completed"
    assert resumed["attempted_cycles"] == 2
    assert resumed["executed_cycles"] == 1
    assert resumed["started_utc"] == first["started_utc"]

    monkeypatch.setattr(
        handoff.deferred,
        "sample_nvml",
        lambda: pytest.fail("completed queue must resume without NVML"),
    )
    monkeypatch.setattr(
        handoff,
        "run_scheduler",
        lambda *_a, **_k: pytest.fail("completed queue must be idempotent"),
    )
    idempotent = _run(tmp_path, cycles=2)
    assert idempotent["state"] == "completed"
    assert idempotent["attempted_cycles"] == 2
    assert idempotent["executed_cycles"] == 1


def test_duplicate_service_rejected_and_stale_owner_recovered(tmp_path: Path) -> None:
    config, _ = handoff.load_config(CONFIG)
    lease_path, owned = handoff._acquire_service_lease(tmp_path, config, 0)
    with pytest.raises(RuntimeError, match="already active"):
        handoff._acquire_service_lease(tmp_path, config, 0)
    handoff._release_service_lease(lease_path, owned, int(config["maximum_state_bytes"]))
    stale_body = {
        "schema_version": handoff.LEASE_SCHEMA,
        "service_id": config["service_id"],
        "service_epoch": config["service_epoch"],
        "pid": 2_147_483_647,
        "process_argv_sha256": "0" * 64,
        "attempted_cycles": 7,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    stale = {**stale_body, "content_sha256": handoff._content_sha(stale_body)}
    lease_path.write_text(json.dumps(stale), encoding="utf-8")
    recovered_path, recovered = handoff._acquire_service_lease(tmp_path, config, 7)
    assert (tmp_path / config["service_recovery_name"]).exists()
    handoff._release_service_lease(recovered_path, recovered, int(config["maximum_state_bytes"]))


def test_tamper_oversize_and_bound_widening_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = json.loads(READINESS.read_text(encoding="utf-8"))
    result["scientific_test_pass"] = True
    result["content_sha256"] = handoff._content_sha(result)
    with pytest.raises(ValueError, match="claim"):
        handoff.validate_readiness(result, CONFIG)
    config, _ = handoff.load_config(CONFIG)
    lease = tmp_path / config["service_lease_name"]
    lease.write_bytes(b"x" * (int(config["maximum_state_bytes"]) + 1))
    with pytest.raises(RuntimeError, match="size bound"):
        handoff._acquire_service_lease(tmp_path, config, 0)
    lease.unlink()
    monkeypatch.setattr(handoff.deferred, "sample_nvml", lambda: _sample(99, 8000))
    with pytest.raises(ValueError, match="widens"):
        handoff.run_service(
            CONFIG,
            runtime_override=tmp_path / "service",
            gpu_owner_runtime_override=tmp_path / "owner",
            maximum_cycles_override=25,
            maximum_wait_polls_override=3,
        )


def test_source_and_config_have_no_direct_sqlite_signal_or_injection_surface() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/kastner_schlatter_deferred_gpu_handoff_service.py"
    ).read_text()
    lowered = source.lower()
    assert "import sqlite" not in lowered
    assert "import subprocess" not in lowered
    assert "os.kill" not in lowered
    assert "terminate(" not in lowered
    assert "popen(" not in lowered
    assert "executor=" not in lowered
    config = json.loads(CONFIG.read_text())
    assert not {"callable", "command", "argv", "module", "function"}.intersection(config)
    assert config["seals"]["handoff_service_direct_sqlite_access"] is False
    assert config["seals"]["existing_process_signaled"] is False
