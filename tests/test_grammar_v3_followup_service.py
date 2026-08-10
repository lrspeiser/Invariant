from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_followup_service import (
    GrammarV3FollowupService,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_followup_service.json"
COORDINATOR_CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
RESOURCE_PROFILE = ROOT / "configs" / "resource_profile_5090.json"
PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-followup-service-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _coordinator_config(service_config: dict) -> dict:
    config = _load(COORDINATOR_CONFIG)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 10,
        "maximum_attempts": 3,
        "lease_seconds": 2,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        "maximum_tasks": 10,
        "maximum_wall_seconds": service_config["budget"]["maximum_wall_seconds"],
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 2}
    return config


def _service(directory: Path, config: dict | None = None) -> GrammarV3FollowupService:
    config = config or _load(CONFIG)
    return GrammarV3FollowupService(
        directory,
        config,
        _coordinator_config(config),
        _load(RESOURCE_PROFILE),
        ROOT,
    )


def _start_stop_resume(directory: Path) -> dict:
    service = _service(directory)
    started = service.start()
    assert started["admission"]["accepted"] == 6
    assert started["executed"] == 6
    assert started["deferred_accepted"] == 4
    before_root = started["status"]["completed_work_records_root_sha256"]
    service.stop()
    resumed_service = _service(directory)
    resumed = resumed_service.resume()
    assert resumed["admission"]["accepted"] == 0
    assert resumed["admission"]["duplicate"] == 6
    assert resumed["executed"] == 0
    assert resumed["deferred_accepted"] == 0
    assert resumed["deferred_duplicate"] == 4
    assert resumed["status"]["completed_work_records_root_sha256"] == before_root
    return resumed_service.export()


def test_service_lifecycle_defers_missing_evaluators_and_restarts(tmp_path: Path) -> None:
    directory = tmp_path / "service"
    service = _service(directory)
    started = service.start()
    status = started["status"]
    assert status["lifecycle"] == "waiting_for_reviewed_evaluators"
    assert status["cycle_count"] == 1
    assert status["admitted_count"] == 6
    assert status["processed_count"] == 6
    assert status["deferred_count"] == 4
    assert status["packet_state_counts"] == {
        "deferred_missing_evaluator": 4,
        "succeeded": 6,
    }
    assert status["reviewed_evaluator_invocation_count"] == 6
    assert status["missing_evaluator_executions"] == 0
    assert status["candidate_scientific_decisions_changed"] == 0
    assert {row["task_type"].split("_")[0] for row in status["deferred_packets"]} == {
        "g3",
        "g4",
    }
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["paid_llm_spend_usd"] == 0.0

    stopped = service.stop()
    assert stopped["lifecycle"] == "stopped"
    completed_root = stopped["completed_work_records_root_sha256"]
    deferred_root = stopped["deferred_packet_root_sha256"]

    restarted = _service(directory)
    resumed = restarted.resume()
    assert resumed["admission"]["accepted"] == 0
    assert resumed["admission"]["duplicate"] == 6
    assert resumed["executed"] == 0
    assert resumed["deferred_duplicate"] == 4
    assert resumed["status"]["cycle_count"] == 2
    assert resumed["status"]["completed_work_records_root_sha256"] == completed_root
    assert resumed["status"]["deferred_packet_root_sha256"] == deferred_root
    assert resumed["status"]["coordinator_checkpoint_sequence"] == 5


def test_report_evaluator_budget_and_live_target_tamper_fail_closed(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["reviewed_report_revisions"][0]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="report revision file"):
        _service(tmp_path / "report", config)

    config = _load(CONFIG)
    config["evaluator_descriptor_allowlist"][0]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptors differ|descriptor hash"):
        _service(tmp_path / "descriptor", config)

    config = _load(CONFIG)
    config["budget"]["maximum_service_bytes"] = 4096
    with pytest.raises(ValueError, match="original queue budgets"):
        _service(tmp_path / "disk", config)

    with pytest.raises(ValueError, match="live campaign watchdog"):
        _service(tmp_path / "campaign-v1-live.sqlite")


def test_cycle_budget_and_start_semantics_are_fail_closed(tmp_path: Path) -> None:
    service = _service(tmp_path / "cycles")
    service.start()
    with pytest.raises(ValueError, match="new lifecycle"):
        service.start()
    service.resume()
    service.resume()
    with pytest.raises(RuntimeError, match="cycle budget"):
        service.resume()


def test_portable_service_export_is_exact(tmp_path: Path) -> None:
    assert _start_stop_resume(tmp_path / "portable") == _load(PORTABLE)
