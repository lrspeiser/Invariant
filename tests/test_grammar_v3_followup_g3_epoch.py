from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_followup_queue import portable_followup_status
from sigma_theory_compiler.grammar_v3_followup_service import GrammarV3FollowupService
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR_CONFIG = ROOT / "configs" / "grammar_v3_followup_service.json"
EPOCH_CONFIG = ROOT / "configs" / "grammar_v3_followup_service_g3_epoch.json"
COORDINATOR_CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
RESOURCE_PROFILE = ROOT / "configs" / "resource_profile_5090.json"
PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-followup-service-g3-epoch-status.json"
QUEUE_PORTABLE = (
    ROOT / "runs" / "engine" / "grammar-v3-followup-queue-g3-epoch-status.json"
)


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


def _service(directory: Path, config: dict) -> GrammarV3FollowupService:
    return GrammarV3FollowupService(
        directory,
        config,
        _coordinator_config(config),
        _load(RESOURCE_PROFILE),
        ROOT,
    )


def _predecessor_epoch(directory: Path) -> None:
    config = _load(PREDECESSOR_CONFIG)
    service = _service(directory, config)
    assert service.start()["executed"] == 6
    service.stop()
    resumed = _service(directory, config).resume()
    assert resumed["executed"] == 0
    assert resumed["status"]["completed_work_records_root_sha256"] == (
        "594a872c481bdaf5c0343c928384db127dec66c453dbaa2ecb87721927b72309"
    )
    assert resumed["status"]["deferred_packet_root_sha256"] == (
        "6c19289f49c45cea10f790c2f83ff290d698cfd122c4e0186c4818e609fc4cc7"
    )


def _run_epoch(directory: Path) -> tuple[GrammarV3FollowupService, dict]:
    _predecessor_epoch(directory)
    epoch = _service(directory, _load(EPOCH_CONFIG))
    migrated = epoch.status()
    assert migrated["processed_count"] == 6
    assert migrated["deferred_count"] == 4
    resumed = epoch.resume()
    assert resumed["admission"] == {
        "accepted": 2,
        "duplicate": 6,
        "backpressured": 0,
        "budget_rejected": 0,
    }
    assert resumed["executed"] == 2
    assert resumed["deferred_resolved"] == 2
    assert resumed["deferred_accepted"] == 0
    assert resumed["deferred_duplicate"] == 2
    return epoch, epoch.export()


def test_epoch_resumes_two_g3_packets_and_leaves_only_g4_deferred(tmp_path: Path) -> None:
    epoch, status = _run_epoch(tmp_path / "service")
    assert status["lifecycle"] == "waiting_for_reviewed_evaluators"
    assert status["cycle_count"] == 3
    assert status["admitted_count"] == 8
    assert status["processed_count"] == 8
    assert status["deferred_count"] == 2
    assert status["packet_state_counts"] == {
        "deferred_missing_evaluator": 2,
        "succeeded": 8,
    }
    assert status["reviewed_evaluator_invocation_count"] == 8
    assert status["missing_evaluator_executions"] == 0
    assert status["candidate_scientific_decisions_changed"] == 0
    assert {packet["task_type"] for packet in status["deferred_packets"]} == {
        "g4_global_lapse_invertibility",
        "g4_global_positive_energy",
    }
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["paid_llm_spend_usd"] == 0.0

    with epoch.coordinator.connect() as connection:
        rows = connection.execute(
            "SELECT result_json FROM work WHERE result_json IS NOT NULL"
        ).fetchall()
    results = [json.loads(row[0]) for row in rows]
    g3 = {result["task_type"]: result for result in results if result["task_type"].startswith("g3_")}
    assert set(g3) == {"g3_uniform_interval_cell", "g3_global_lapse_dirac_contract"}
    interval = g3["g3_uniform_interval_cell"]["reviewed_evidence"]
    assert interval["blocker"] == "candidate_specific_full_Delta_N_operator"
    assert interval["componentwise_domain_sha256"] == (
        "bcf4c8b42f3e81cb7359c999f538b9c2a6da819cbb12023eaa29e98e6cb4a131"
    )
    lapse = g3["g3_global_lapse_dirac_contract"]["reviewed_evidence"]
    assert lapse["blocker"] == "asymptotically_flat_or_global_energy_domain"
    assert lapse["coercivity_sha256"] == (
        "308aaed19457d0caa98174809fe4c7c13cebdf47c6e9e197ce011fe0bde41ad3"
    )
    assert lapse["global_energy_status"] == "blocked"
    assert all(result["decision"] == "blocked" for result in g3.values())

    restarted = _service(tmp_path / "service", _load(EPOCH_CONFIG))
    assert restarted.export() == status


def test_epoch_replay_roots_are_deterministic(tmp_path: Path) -> None:
    _, first = _run_epoch(tmp_path / "first")
    _, second = _run_epoch(tmp_path / "second")
    for key in (
        "queue_registry_root_sha256",
        "completed_work_records_root_sha256",
        "deferred_packet_root_sha256",
        "content_sha256",
    ):
        assert first[key] == second[key]


def test_epoch_predecessor_and_descriptor_tamper_fail_closed(tmp_path: Path) -> None:
    _predecessor_epoch(tmp_path / "predecessor")
    config = _load(EPOCH_CONFIG)
    config["predecessor_service_epoch"]["deferred_packet_root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predecessor roots"):
        _service(tmp_path / "predecessor", config)

    config = _load(EPOCH_CONFIG)
    config["evaluator_descriptor_allowlist"][-1]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptors differ|descriptor hash"):
        _service(tmp_path / "descriptor", config)


def test_portable_epoch_export_is_exact(tmp_path: Path) -> None:
    epoch, exported = _run_epoch(tmp_path / "portable")
    assert exported == _load(PORTABLE)
    assert portable_followup_status(epoch.queue.status()) == _load(QUEUE_PORTABLE)
