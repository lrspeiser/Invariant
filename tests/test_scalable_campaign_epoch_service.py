from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.scalable_campaign_epoch_service import (
    ScalableCampaignEpochService,
    _sha,
    load_scalable_campaign_epoch_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scalable_campaign_epoch_service.json"


def _enabled() -> dict:
    config = load_scalable_campaign_epoch_config(CONFIG)
    config["execution_enabled"] = True
    return config


def test_bounded_epoch_runs_replays_and_preserves_exact_decisions(tmp_path: Path) -> None:
    config = _enabled()
    service = ScalableCampaignEpochService(tmp_path / "service", config, ROOT)
    assert service.enqueue() == {
        "accepted": 10,
        "duplicate": 0,
        "backpressured": 0,
        "budget_rejected": 0,
    }
    assert service.run_ready() == 10
    first = service.status()
    assert first["sealed_epoch_counts"] == {
        "parameter_cells": 256,
        "unique_candidates": 163,
        "aliases": 93,
        "decisions": {"blocked": 158, "pass": 3, "reject": 2},
    }
    assert first["queue"]["states"] == {"succeeded": 10}
    assert first["next_epoch_readiness"] == {
        "state": "blocked",
        "blockers": [
            "missing_hash_bound_future_manifest_chunk",
            "missing_reviewed_future_compiler_adapter",
        ],
        "admitted_future_cells": 0,
        "scientific_compilation_started": False,
    }

    resumed = ScalableCampaignEpochService(tmp_path / "service", config, ROOT)
    assert resumed.enqueue()["duplicate"] == 10
    assert resumed.run_ready() == 0
    second = resumed.status()
    assert second["stage_registry_root_sha256"] == first["stage_registry_root_sha256"]
    assert second["source_export"] == first["source_export"]
    assert second["sealed_epoch_counts"] == first["sealed_epoch_counts"]


def test_lease_recovery_and_changed_config_refusal(tmp_path: Path) -> None:
    config = _enabled()
    service = ScalableCampaignEpochService(tmp_path / "service", config, ROOT)
    service.enqueue()
    lease = service.coordinator.claim("cpu", "crashed", lease_seconds=1)
    assert lease is not None
    with service.coordinator.connect() as connection:
        connection.execute(
            "UPDATE work SET lease_expires_utc='2000-01-01T00:00:00+00:00' WHERE work_id=?",
            (lease.work_id,),
        )
    resumed = ScalableCampaignEpochService(tmp_path / "service", config, ROOT)
    assert resumed.recovered_on_start == {"recovered": 1, "failed": 0}
    assert resumed.run_ready() == 10

    changed = json.loads(json.dumps(config))
    changed["budget"]["maximum_wall_seconds"] = 121
    with pytest.raises(ValueError, match="different execution config"):
        ScalableCampaignEpochService(tmp_path / "service", changed, ROOT)


def test_hash_tamper_and_disabled_execution_fail_closed(tmp_path: Path) -> None:
    config = load_scalable_campaign_epoch_config(CONFIG)
    service = ScalableCampaignEpochService(tmp_path / "disabled", config, ROOT)
    with pytest.raises(PermissionError, match="disabled"):
        service.enqueue()

    tampered = json.loads(json.dumps(config))
    tampered["evidence_export"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        ScalableCampaignEpochService(tmp_path / "tampered", tampered, ROOT)

    artifact = json.loads(
        (ROOT / "runs/engine/scalable-campaign-staged-epoch-status.json").read_text()
    )
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert artifact == service.status()


def test_future_chunk_is_bounded_hash_bound_and_only_admitted(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    config = _enabled()
    paths = {
        config["evidence_export_config"]["path"],
        config["evidence_export"]["path"],
        config["coordinator_config"]["path"],
        config["resource_profile"]["path"],
    }
    export_config = json.loads((ROOT / config["evidence_export_config"]["path"]).read_text())
    for value in export_config.values():
        if isinstance(value, dict) and "path" in value:
            paths.add(value["path"])
    source_rel = "src/sigma_theory_compiler/scalable_campaign_epoch_service.py"
    paths.add(source_rel)
    for rel in paths:
        target = fixture / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / rel).read_bytes())

    future_body = {
        "schema_version": "sigma-scalable-campaign-future-manifest-chunk-1.0",
        "chunk_id": "fixture-next-epoch-0001",
        "parent_epoch_content_sha256": config["evidence_export"]["content_sha256"],
        "parameter_cells": [
            {"parameter_cell_id": "FUTURE-000", "parameter_cell_lineage_sha256": "1" * 64},
            {"parameter_cell_id": "FUTURE-001", "parameter_cell_lineage_sha256": "2" * 64},
        ],
        "data_eligibility": config["data_eligibility"],
        "external_paid_llm_calls": False,
    }
    future = {**future_body, "content_sha256": _sha(future_body)}
    future_path = fixture / "fixtures" / "future.json"
    future_path.parent.mkdir(parents=True)
    future_path.write_text(json.dumps(future, sort_keys=True), encoding="utf-8")
    source_hash = hashlib.sha256((fixture / source_rel).read_bytes()).hexdigest()
    adapter_body = {
        "schema_version": "sigma-reviewed-future-compiler-adapter-descriptor-1.0",
        "reviewed": True,
        "task_type": "reviewed_future_manifest_chunk_admission",
        "next_task_type": "reviewed_future_candidate_compilation",
        "parent_epoch_content_sha256": config["evidence_export"]["content_sha256"],
        "callback_entrypoint": "sigma_theory_compiler.scalable_campaign_epoch_service:reviewed_future_manifest_admission_adapter",
        "callback_source_path": source_rel,
        "callback_source_file_sha256": source_hash,
        "data_eligibility": config["data_eligibility"],
        "external_paid_llm_calls": False,
    }
    adapter = {**adapter_body, "content_sha256": _sha(adapter_body)}
    adapter_path = fixture / "fixtures" / "adapter.json"
    adapter_path.write_text(json.dumps(adapter, sort_keys=True), encoding="utf-8")
    config["future_manifest_chunk"] = {
        "path": "fixtures/future.json",
        "file_sha256": hashlib.sha256(future_path.read_bytes()).hexdigest(),
        "content_sha256": future["content_sha256"],
    }
    config["reviewed_future_compiler_adapter"] = {
        "path": "fixtures/adapter.json",
        "file_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "content_sha256": adapter["content_sha256"],
    }
    service = ScalableCampaignEpochService(tmp_path / "future-service", config, fixture)
    assert service.enqueue()["accepted"] == 11
    assert service.run_ready() == 11
    status = service.status()
    assert status["next_epoch_readiness"] == {
        "state": "ready_for_reviewed_compiler_queue_admission",
        "blockers": [],
        "admitted_future_cells": 2,
        "scientific_compilation_started": False,
    }

    future["parameter_cells"].append(future["parameter_cells"][0])
    future_path.write_text(json.dumps(future, sort_keys=True), encoding="utf-8")
    config["future_manifest_chunk"]["file_sha256"] = hashlib.sha256(
        future_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="invalid future_manifest_chunk content"):
        ScalableCampaignEpochService(tmp_path / "bad-future", config, fixture)
