from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_followup_queue import (
    GrammarV3FollowupQueue,
    portable_followup_status,
)
from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_followup_queue.json"
COORDINATOR_CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
RESOURCE_PROFILE = ROOT / "configs" / "resource_profile_5090.json"
PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-followup-queue-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _coordinator_config(config: dict) -> dict:
    coordinator = _load(COORDINATOR_CONFIG)
    coordinator["queue"] = {
        **coordinator["queue"],
        "maximum_pending_work": 10,
        "maximum_attempts": 3,
        "lease_seconds": 2,
        "checkpoint_every_completions": 1,
    }
    coordinator["budget"] = {
        "maximum_tasks": 10,
        "maximum_wall_seconds": config["budget"]["maximum_wall_seconds"],
    }
    coordinator["cpu"] = {**coordinator["cpu"], "maximum_workers": 2}
    return coordinator


def _adapter(database: Path, config: dict | None = None) -> GrammarV3FollowupQueue:
    config = config or _load(CONFIG)
    coordinator = PersistentParallelSearch(
        database, _coordinator_config(config), _load(RESOURCE_PROFILE)
    )
    return GrammarV3FollowupQueue(coordinator, config, ROOT)


def _run(database: Path) -> dict:
    adapter = _adapter(database)
    assert adapter.enqueue()["accepted"] == 10
    execution = adapter.run_bounded()
    assert execution["executed"] == 10
    return portable_followup_status(execution["status"])


def test_packets_preserve_pareto_axes_blockers_and_reviewed_types(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path / "queue.sqlite")
    assert len(adapter.work_packets) == 10
    assert Counter(packet["task_type"] for packet in adapter.work_packets) == {
        "aether_nonlinear_twist_energy": 2,
        "g2_global_boundary_dirac_contract": 2,
        "g2_global_positive_mass": 2,
        "g3_uniform_interval_cell": 1,
        "g3_global_lapse_dirac_contract": 1,
        "g4_global_lapse_invertibility": 1,
        "g4_global_positive_energy": 1,
    }
    aether_packets = [
        packet
        for packet in adapter.work_packets
        if packet["task_type"] == "aether_nonlinear_twist_energy"
    ]
    g2_packets = [
        packet
        for packet in adapter.work_packets
        if packet["task_type"]
        in {"g2_global_boundary_dirac_contract", "g2_global_positive_mass"}
    ]
    missing_packets = [
        packet
        for packet in adapter.work_packets
        if packet["task_type"].startswith(("g3_", "g4_"))
    ]
    assert len(aether_packets) == 2
    assert len(g2_packets) == 4
    assert all(packet["reviewed_evaluator_binding_sha256"] for packet in aether_packets)
    assert all(packet["reviewed_evaluator_binding_sha256"] for packet in g2_packets)
    assert all(packet["reviewed_evaluator_binding_sha256"] is None for packet in missing_packets)
    assert all(packet["pareto_axes"] == adapter.report["priority_axes"] for packet in adapter.work_packets)
    assert all(packet["target_blockers"] for packet in adapter.work_packets)
    assert all(packet["data_eligibility"] == ELIGIBILITY for packet in adapter.work_packets)
    assert all("truth_score" not in packet for packet in adapter.work_packets)

    admission = adapter.enqueue()
    assert admission["accepted"] == 10
    execution = adapter.run_bounded()
    status = execution["status"]
    assert execution["executed"] == 10
    assert status["work_state_counts"] == {"succeeded": 10}
    assert status["followup_decision_counts"] == {"blocked": 10}
    assert status["reviewed_evaluator_invocation_count"] == 6
    assert status["missing_evaluator_count"] == 4
    assert status["candidate_scientific_decisions_changed"] == 0
    assert status["observational_data_opened"] is False
    assert status["paid_llm_spend_usd"] == 0.0
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    with adapter.coordinator.connect() as connection:
        rows = connection.execute(
            "SELECT priority,result_json FROM work ORDER BY ordinal"
        ).fetchall()
    assert all(float(row["priority"]) == 0.0 for row in rows)
    results = [json.loads(row["result_json"]) for row in rows]
    invoked = [result for result in results if result["evaluator_invoked"]]
    missing = [result for result in results if not result["evaluator_invoked"]]
    assert len(invoked) == 6
    assert len(missing) == 4
    aether_results = [
        result
        for result in invoked
        if result["task_type"] == "aether_nonlinear_twist_energy"
    ]
    g2_boundary_results = [
        result
        for result in invoked
        if result["task_type"] == "g2_global_boundary_dirac_contract"
    ]
    g2_mass_results = [
        result
        for result in invoked
        if result["task_type"] == "g2_global_positive_mass"
    ]
    assert all(
        result["blocker"] == "complete_generic_twisting_reduced_hamiltonian"
        and result["decision"] == "blocked"
        and result["reviewed_evidence"]["negative_energy_mode_found"] is False
        and result["reviewed_evidence"]["scientific_candidate_decision_changed"] is False
        for result in aether_results
    )
    assert len(g2_boundary_results) == 2
    assert all(
        result["blocker"] == "complete_distributed_dirac_boundary_contract"
        and result["reviewed_evidence"]["negative_total_energy_counterexample_found"] is False
        and result["reviewed_evidence"]["scientific_candidate_decision_changed"] is False
        for result in g2_boundary_results
    )
    assert len(g2_mass_results) == 2
    assert all(
        result["blocker"] == "hash_bound_general_nonmaximal_positive_mass_theorem"
        and result["reviewed_evidence"]["negative_total_energy_counterexample_found"] is False
        and result["reviewed_evidence"]["scientific_candidate_decision_changed"] is False
        for result in g2_mass_results
    )
    assert all(result["blocker"] == "reviewed_evaluator_missing" for result in missing)

    resumed = _adapter(tmp_path / "queue.sqlite")
    replay = resumed.enqueue()
    assert replay["accepted"] == 0
    assert replay["duplicate"] == 10
    assert resumed.status()["work_records_root_sha256"] == status[
        "work_records_root_sha256"
    ]


def test_expired_lease_recovery_is_deterministic(tmp_path: Path) -> None:
    database = tmp_path / "recovery.sqlite"
    adapter = _adapter(database)
    assert adapter.enqueue()["accepted"] == 10
    abandoned = adapter.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = _adapter(database)
    execution = resumed.run_bounded()
    assert execution["recovered"] == {"recovered": 1, "failed": 0}
    assert execution["status"]["work_state_counts"] == {"succeeded": 10}
    assert execution["status"]["followup_decision_counts"] == {"blocked": 10}
    assert max(record["attempt"] for record in execution["status"]["work_records"]) == 2


def test_report_evaluator_disk_and_live_database_tamper_fail_closed(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["pareto_report"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="report file mismatch"):
        _adapter(tmp_path / "report.sqlite", config)

    config = _load(CONFIG)
    config["reviewed_evaluators"]["g4_global_positive_energy"] = {
        "descriptor_path": "arbitrary",
        "descriptor_file_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="evaluator allowlist"):
        _adapter(tmp_path / "evaluator.sqlite", config)

    config = _load(CONFIG)
    config["reviewed_evaluators"]["aether_nonlinear_twist_energy"][
        "descriptor_file_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="descriptor hash"):
        _adapter(tmp_path / "descriptor.sqlite", config)

    config = _load(CONFIG)
    config["budget"]["maximum_database_bytes"] = 4096
    disk_limited = _adapter(tmp_path / "disk.sqlite", config)
    with pytest.raises(RuntimeError, match="disk budget"):
        disk_limited.enqueue()

    with pytest.raises(ValueError, match="live campaign watchdog"):
        _adapter(tmp_path / "campaign-v1-live.sqlite")


def test_portable_followup_status_is_exact(tmp_path: Path) -> None:
    assert _run(tmp_path / "portable.sqlite") == _load(PORTABLE)
