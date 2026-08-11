from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_g4_solar_reviewed_execution import (
    ACTION_SHA256,
    BLOCKER,
    CANDIDATE_ID,
    DESCRIPTOR_FIELD,
    GrammarV3G4SolarReviewedExecution,
)
from sigma_theory_compiler.persistent_parallel_search import WorkLease
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.reviewed_g4_candidate_solar_evaluator import (
    REQUIRED_REGISTRATION_HASHES,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_g4_solar_reviewed_execution.json"
PORTABLE = (
    ROOT / "runs" / "engine" / "grammar-v3-g4-solar-reviewed-execution-status.json"
)
DESCRIPTOR = ROOT / "configs" / "reviewed_g4_candidate_solar_evaluator.json"
EXECUTOR = (
    ROOT
    / "src"
    / "sigma_theory_compiler"
    / "grammar_v3_g4_solar_reviewed_execution.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution(directory: Path, config: dict | None = None):
    return GrammarV3G4SolarReviewedExecution(
        directory,
        _load(CONFIG) if config is None else config,
        ROOT,
    )


def test_single_candidate_callback_executes_and_preserves_exact_blocker(
    tmp_path: Path,
) -> None:
    execution = _execution(tmp_path / "execution")
    assert execution.enqueue() == {
        "accepted": 1,
        "duplicate": 0,
        "backpressured": 0,
        "budget_rejected": 0,
        "requested": 1,
    }
    run = execution.run_bounded()
    assert run["processed"] == 1
    status = run["status"]
    assert status["candidate_id"] == CANDIDATE_ID
    assert status["action_sha256"] == ACTION_SHA256
    assert status["task_count"] == 1
    assert status["work_state_counts"] == {"succeeded": 1}
    assert status["decision_counts"] == {"blocked": 1}
    assert status["reviewed_evaluator_invocation_count"] == 1
    assert status["filled_registration_hash_count"] == 1
    assert status["missing_registration_hash_count"] == 16
    assert status["missing_registration_hashes"] == sorted(
        set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD}
    )
    assert status["candidate_use_authorized"] is False
    assert status["observational_opening_authorized"] is False
    assert status["observational_data_opened"] is False
    assert status["primary_record_access_count"] == 0
    assert status["tracking_target_values_opened"] is False
    assert status["paid_llm_spend_usd"] == 0.0
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}

    with execution.coordinator.connect() as connection:
        result = json.loads(
            connection.execute("SELECT result_json FROM work").fetchone()[0]
        )
    assert result["decision"] == "blocked"
    assert result["blocker"] == BLOCKER
    assert result["filled_registration_hash_count"] == 1
    assert result["observational_data_opened"] is False


def test_restart_replay_and_expired_lease_recovery_are_deterministic(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "restart"
    first = _execution(directory)
    assert first.enqueue()["accepted"] == 1
    abandoned = first.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None

    restarted = _execution(directory)
    assert restarted.recovered_on_start == {"recovered": 1, "failed": 0}
    completed = restarted.run_bounded()["status"]
    assert completed["work_records"][0]["attempt"] == 2
    assert completed["recovered_leases"] == 1
    root = completed["work_records_root_sha256"]

    replay = _execution(directory)
    assert replay.enqueue()["duplicate"] == 1
    assert replay.run_bounded()["processed"] == 0
    assert replay.status()["work_records_root_sha256"] == root


def test_allowlist_executor_predecessor_and_authorization_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    mutations = [
        ("allowlist_file", "artifact changed"),
        ("allowlist_binding", "not exactly allowlisted"),
        ("callback_source", "not exactly allowlisted"),
        ("executor_source", "execution source changed"),
        ("queue_lineage", "content changed"),
        ("authorization", "forbidden input"),
        ("missing_authorization", "config is invalid"),
    ]
    for index, (mutation, message) in enumerate(mutations):
        config = _load(CONFIG)
        if mutation == "allowlist_file":
            config["evaluator_descriptor_allowlist"][0]["file_sha256"] = "0" * 64
        elif mutation == "allowlist_binding":
            config["evaluator_descriptor_allowlist"][0][
                "descriptor_binding_sha256"
            ] = "0" * 64
        elif mutation == "callback_source":
            config["evaluator_descriptor_allowlist"][0][
                "callback_source_sha256"
            ] = "0" * 64
        elif mutation == "executor_source":
            config["executor_source"]["file_sha256"] = "0" * 64
        elif mutation == "queue_lineage":
            config["predecessor_followup_queue"]["content_sha256"] = "0" * 64
        elif mutation == "authorization":
            config["observational_opening_authorized"] = True
        else:
            del config["observational_opening_authorized"]
        with pytest.raises(ValueError, match=message):
            _execution(tmp_path / f"tamper-{index}", config)


def test_lease_and_stored_result_tampering_are_rejected(tmp_path: Path) -> None:
    execution = _execution(tmp_path / "lease")
    execution.enqueue()
    lease = execution.coordinator.claim("cpu", "manual")
    assert lease is not None
    bad_payload = copy.deepcopy(lease.payload)
    bad_payload["observational_opening_authorized"] = True
    bad = WorkLease(
        lease.work_id,
        lease.ordinal,
        lease.lane,
        lease.seed,
        lease.attempt,
        lease.max_attempts,
        bad_payload,
    )
    with pytest.raises(ValueError, match="lease or lineage changed"):
        execution.execute_lease(bad)

    execution.coordinator.fail(lease, "manual", "negative fixture")
    execution.run_bounded()
    with execution.coordinator.connect() as connection:
        result = json.loads(
            connection.execute("SELECT result_json FROM work").fetchone()[0]
        )
        result["observational_data_opened"] = True
        connection.execute(
            "UPDATE work SET result_json=?",
            (json.dumps(result, sort_keys=True, separators=(",", ":")),),
        )
    with pytest.raises(ValueError, match="stored reviewed G4 Solar result changed"):
        execution.status()


def test_descriptor_and_executor_files_are_exactly_bound() -> None:
    config = _load(CONFIG)
    allow = config["evaluator_descriptor_allowlist"]
    assert len(allow) == 1
    assert allow[0]["file_sha256"] == _file_sha(DESCRIPTOR)
    assert allow[0]["callback_source_sha256"] == _load(DESCRIPTOR)[
        "artifact_sha256"
    ]
    assert config["executor_source"]["file_sha256"] == _file_sha(EXECUTOR)
    assert config["candidate"] == {
        "candidate_id": CANDIDATE_ID,
        "role": "generated_candidate",
        "action_sha256": ACTION_SHA256,
    }


def test_portable_status_is_exact_rebuild(tmp_path: Path) -> None:
    execution = _execution(tmp_path / "portable")
    assert execution.enqueue()["accepted"] == 1
    execution.run_bounded()
    assert execution.export() == _load(PORTABLE)
