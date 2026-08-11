from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import sigma_theory_compiler.kastner_schlatter_set_indexed_gpu_scheduler_adapter as adapter
from sigma_theory_compiler.persistent_parallel_search import WorkLease

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_set_indexed_gpu_scheduler_adapter.json"
READINESS = ROOT / "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-adapter-readiness.json"
IMMUTABLE = ROOT / "runs/engine/kastner-schlatter-set-indexed-cuda-falsification-campaign.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_config_tree(tmp_path: Path, config: dict) -> Path:
    for binding in config["bindings"].values():
        source = ROOT / binding["path"]
        target = tmp_path / binding["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    target_config = tmp_path / "configs" / CONFIG.name
    target_config.parent.mkdir(parents=True, exist_ok=True)
    target_config.write_text(json.dumps(config), encoding="utf-8")
    return target_config


def _lease(config: dict, attempt: int = 1) -> WorkLease:
    payload = adapter._reviewed_payload(config)
    payload["adapter_config_path"] = str(CONFIG.resolve())
    payload["adapter_config_file_sha256"] = adapter._file_sha(CONFIG)
    return WorkLease("PSW-reviewed", 0, "gpu", 123, attempt, 3, payload)


def test_readiness_validates_and_did_not_start_scheduler() -> None:
    result = json.loads(READINESS.read_text(encoding="utf-8"))
    adapter.validate_readiness(result, CONFIG)
    assert result["scheduler_contract"]["gpu_owner_count"] == 1
    assert result["scheduler_contract"]["cpu_worker_count"] == 0
    assert result["scheduler_contract"]["arbitrary_callable_or_subprocess_surface"] is False
    assert result["execution_state"] == {
        "runtime_created_by_readiness": False,
        "scheduler_started_by_readiness": False,
        "worker_result_created_by_readiness": False,
    }


def test_idempotent_enqueue_and_durable_result(tmp_path: Path) -> None:
    coordinator, config, _, _ = adapter.create_coordinator(CONFIG, runtime_override=tmp_path)
    first = adapter.enqueue_reviewed_workload(coordinator, config, CONFIG)
    second = adapter.enqueue_reviewed_workload(coordinator, config, CONFIG)
    assert first == {"accepted": 1, "duplicate": 0, "backpressured": 0, "budget_rejected": 0}
    assert second == {"accepted": 0, "duplicate": 1, "backpressured": 0, "budget_rejected": 0}
    lease = coordinator.claim("gpu", "test-owner")
    assert lease is not None
    result = {"workload_id": adapter.REVIEWED_WORKLOAD_ID, "immutable_artifact_written": False}
    assert coordinator.finish(lease, "test-owner", result)
    resumed, _, _, _ = adapter.create_coordinator(CONFIG, runtime_override=tmp_path)
    durable = adapter.durable_results(resumed)
    assert durable[0]["state"] == "succeeded"
    assert durable[0]["result"] == result


def test_expired_lease_is_recovered_after_restart(tmp_path: Path) -> None:
    coordinator, config, _, _ = adapter.create_coordinator(CONFIG, runtime_override=tmp_path)
    adapter.enqueue_reviewed_workload(coordinator, config, CONFIG)
    lease = coordinator.claim("gpu", "crashed-owner")
    assert lease is not None
    with coordinator.connect() as connection:
        connection.execute(
            "UPDATE work SET lease_expires_utc=? WHERE work_id=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), lease.work_id),
        )
    resumed, _, _, _ = adapter.create_coordinator(CONFIG, runtime_override=tmp_path)
    assert resumed.recover_expired() == {"recovered": 1, "failed": 0}
    recovered = resumed.claim("gpu", "replacement-owner")
    assert recovered is not None
    assert recovered.work_id == lease.work_id
    assert recovered.attempt == 2


def test_fixed_evaluator_has_no_immutable_write(monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = adapter.load_adapter_config(CONFIG)
    before = IMMUTABLE.read_bytes()
    synthetic_result = {
        "scenario_results": [],
        "counts": {"scenario_cells": 0},
        "crosscheck": {"all_heldout_decisions_byte_equal": True},
    }
    monkeypatch.setattr(adapter, "execute_gpu_workload", lambda _: synthetic_result)
    result = adapter.gpu_reviewed_workload_evaluator(_lease(config))
    assert result["compute_result"] == synthetic_result
    assert result["immutable_artifact_written"] is False
    assert result["live_campaign_sqlite_accessed"] is False
    assert IMMUTABLE.read_bytes() == before


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("gpu_evaluator", "evil.module:callable", "single-owner closed"),
        ("gpu_workers", 2, "single-owner closed"),
        ("cpu_workers", 1, "single-owner closed"),
    ],
)
def test_callable_and_owner_tamper_fail(
    tmp_path: Path, key: str, value: object, match: str
) -> None:
    config = _config()
    config["persistent_config"]["supervisor"][key] = value
    bad = _write_config_tree(tmp_path, config)
    with pytest.raises(ValueError, match=match):
        adapter.load_adapter_config(bad)


def test_payload_injection_and_hash_tamper_fail(tmp_path: Path) -> None:
    config, _ = adapter.load_adapter_config(CONFIG)
    payload = adapter._reviewed_payload(config)
    payload["command"] = "arbitrary.exe"
    with pytest.raises(ValueError, match="tampered"):
        adapter._validate_payload(payload, config)
    bad = _write_config_tree(tmp_path, copy.deepcopy(config))
    source = tmp_path / config["bindings"]["workload_source"]["path"]
    source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="workload_source file hash mismatch"):
        adapter.load_adapter_config(bad)


def test_live_campaign_runtime_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="live campaign runtime override"):
        adapter.create_coordinator(CONFIG, runtime_override=ROOT / "runs/campaigns/forbidden")


def test_supervisor_construction_has_exactly_one_gpu_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_run(self, maximum_wall_seconds=None):
        captured["specs"] = [(spec.lane, spec.evaluator, spec.slot) for spec in self.specs]
        return {"stop_reason": "test_control", "maximum_wall_seconds": maximum_wall_seconds}

    monkeypatch.setattr(adapter.PersistentParallelSupervisor, "run", fake_run)
    report = adapter.run_scheduler(CONFIG, runtime_override=tmp_path, maximum_wall_seconds=0.1)
    assert captured["specs"] == [("gpu", adapter.FIXED_EVALUATOR, 0)]
    assert report["live_campaign_sqlite_accessed"] is False
    assert report["immutable_artifact_written"] is False


def test_readiness_claim_tamper_fails() -> None:
    result = json.loads(READINESS.read_text(encoding="utf-8"))
    result["scientific_test_pass"] = True
    result["content_sha256"] = adapter._content_sha(result)
    with pytest.raises(ValueError, match="readiness claim"):
        adapter.validate_readiness(result, CONFIG)
