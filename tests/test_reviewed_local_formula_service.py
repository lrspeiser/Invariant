from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.campaign import CampaignStore
from sigma_theory_compiler.reviewed_local_formula_service import (
    TASK_TYPE,
    ReviewedLocalServiceError,
    build_readiness_artifact,
    export_service,
    resume_service,
    start_service,
    status_service,
    stop_service,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reviewed_local_formula_service.json"
ARTIFACT = ROOT / "runs/engine/reviewed-local-formula-service-readiness.json"


def _enabled_config(tmp_path: Path) -> Path:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["execution_enabled"] = True
    path = tmp_path / "enabled-service.json"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_detached_start_stop_resume_and_status(tmp_path: Path) -> None:
    config = _enabled_config(tmp_path)
    root = tmp_path / "service"
    started = start_service(ROOT, root, config, allow_bounded_test_harness=True)
    assert started["service_state"] == "ready"
    assert started["task_status"] == "queued"
    stopped = stop_service(ROOT, root, config)
    assert stopped["service_state"] == "stopped"
    completed = resume_service(ROOT, root, config)
    assert completed["service_state"] == "complete"
    assert completed["task_status"] == "succeeded"
    assert completed["result_core_sha256"] is not None
    assert status_service(ROOT, root, config)["service_state"] == "complete"


def test_expired_worker_lease_recovers_and_export_is_deterministic(tmp_path: Path) -> None:
    config = _enabled_config(tmp_path)
    root = tmp_path / "recovery-service"
    started = start_service(ROOT, root, config, allow_bounded_test_harness=True)
    checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    store = CampaignStore(root / "service.sqlite")
    crashed = store.claim_task(
        checkpoint["campaign_id"], "crashed-service-worker", -1, allowed_task_types={TASK_TYPE}
    )
    assert crashed and crashed.attempt == 1
    completed = resume_service(ROOT, root, config)
    assert completed["service_state"] == "complete"
    with store.connect() as connection:
        attempt = connection.execute(
            "SELECT attempt FROM tasks WHERE task_id=?", (checkpoint["task_id"],)
        ).fetchone()[0]
    assert attempt == 2

    first = export_service(ROOT, root, config, tmp_path / "first.json")
    second = export_service(ROOT, root, config, tmp_path / "second.json")
    assert first == second
    assert first["epoch_status"]["lineage_preserved"] is True
    assert first["epoch_status"]["paid_spend_usd"] == "0.000000"
    assert first["epoch_status"]["network_calls"] == 0
    persisted = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    for forbidden in (
        b"bounded-local-mock-capability",
        b"covariant_action(EH_R)",
        b"EH_R+ALIEN_TERM",
        b"sqrt(-g)",
    ):
        assert forbidden not in persisted
    assert started["callback_registry_sha256"] == completed["callback_registry_sha256"]


def test_changed_config_and_checkpoint_tamper_are_refused(tmp_path: Path) -> None:
    config = _enabled_config(tmp_path)
    root = tmp_path / "tamper-service"
    start_service(ROOT, root, config, allow_bounded_test_harness=True)
    changed = json.loads(config.read_text(encoding="utf-8"))
    changed["budgets"]["maximum_wall_seconds"] = 121
    config.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ReviewedLocalServiceError, match="config changed"):
        status_service(ROOT, root, config)

    clean_config = _enabled_config(tmp_path)
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["service_state"] = "complete"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ReviewedLocalServiceError, match="checkpoint hash mismatch"):
        status_service(ROOT, root, clean_config)


def test_checked_in_config_disabled_and_readiness_matches() -> None:
    with pytest.raises(ReviewedLocalServiceError, match="execution is disabled"):
        start_service(ROOT, ROOT / "never-created-service", CONFIG)
    built = build_readiness_artifact(ROOT, CONFIG)
    assert built["default_execution_enabled"] is False
    assert built["budgets"]["maximum_tasks"] == 1
    assert built["network_allowed"] is False
    assert built["paid_spend_usd"] == "0.000000"
    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == built
