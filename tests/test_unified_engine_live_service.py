from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.unified_engine_live_service import (
    _checkpoint_payload,
    _load_json,
    _validate_checkpoint,
    build_readiness,
    load_live_config,
    refresh_once,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/unified_engine_live_service.json"


def test_checked_config_is_bounded_portable_and_fail_closed() -> None:
    config = load_live_config(REPO, CONFIG)
    assert config["enabled"] is True
    assert config["refresh_interval_seconds"] == 300
    assert config["maximum_refreshes"] == 4032
    assert config["maximum_consecutive_failures"] == 12
    assert config["data_seals"] == {
        "observations_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_calls": False,
    }
    assert not Path(config["runtime_directory"]).is_absolute()


def test_one_live_refresh_writes_hash_bound_snapshot_and_human_dashboard(
    tmp_path: Path,
) -> None:
    config = load_live_config(REPO, CONFIG)
    runtime = REPO / "tmp" / f"live-service-test-{tmp_path.name}"
    config = {**config, "runtime_directory": runtime.relative_to(REPO).as_posix()}
    try:
        result = refresh_once(REPO, config)
        snapshot_path = runtime / "unified-engine-status-live.json"
        dashboard_path = runtime / "dashboard.html"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        dashboard = dashboard_path.read_text(encoding="utf-8")
        assert result["core_content_sha256"] == snapshot["core_content_sha256"]
        assert (
            result["snapshot_file_sha256"] == hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        )
        assert (
            result["dashboard_file_sha256"]
            == hashlib.sha256(dashboard_path.read_bytes()).hexdigest()
        )
        assert snapshot["core"]["data_seals"]["observations_opened"] is False
        assert "Theory formula" in dashboard
        assert "Proof and test hierarchy" in dashboard
        assert snapshot["core_content_sha256"] in dashboard
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


def test_checkpoint_tamper_and_config_rebinding_fail_closed(tmp_path: Path) -> None:
    config_sha = "a" * 64
    checkpoint = {
        "schema_version": "sigma-unified-engine-live-service-checkpoint-1.0",
        "config_file_sha256": config_sha,
        "state": "running",
        "pid": 1,
        "refresh_count": 2,
        "consecutive_failures": 0,
        "last_error": None,
        "last_refresh": None,
        "stop_reason": None,
        "updated_utc": "2026-08-11T00:00:00+00:00",
    }
    path = tmp_path / "checkpoint.json"
    path.write_bytes(_checkpoint_payload(checkpoint))
    stored = _load_json(path)
    _validate_checkpoint(stored, config_sha)
    stored["refresh_count"] = 3
    with pytest.raises(ValueError, match="content hash"):
        _validate_checkpoint(stored, config_sha)
    stored = _load_json(path)
    with pytest.raises(ValueError, match="config binding"):
        _validate_checkpoint(stored, "b" * 64)


def test_path_escape_and_opened_seal_are_rejected(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["runtime_directory"] = "../escape"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_live_config(REPO, path)
    config["runtime_directory"] = "runs/engine/live"
    config["data_seals"]["observations_opened"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="data seals"):
        load_live_config(REPO, path)


def test_readiness_is_exactly_reproducible_and_database_read_only() -> None:
    readiness = build_readiness(REPO, CONFIG)
    stored = json.loads(
        (REPO / "runs/engine/unified-engine-live-service-readiness.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness == stored
    assert readiness["decision"] == "ready_enabled_read_only_bounded"
    assert readiness["writes_live_campaign_database"] is False
    assert readiness["immutable_snapshot_overwritten"] is False
