from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_followup_queue import portable_followup_status
from sigma_theory_compiler.grammar_v3_followup_service import GrammarV3FollowupService
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "grammar_v3_followup_service.json"
G3_CONFIG = ROOT / "configs" / "grammar_v3_followup_service_g3_epoch.json"
G4_CONFIG = ROOT / "configs" / "grammar_v3_followup_service_g4_final.json"
COORDINATOR_CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
RESOURCE_PROFILE = ROOT / "configs" / "resource_profile_5090.json"
PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-followup-service-g4-final-status.json"
QUEUE_PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-followup-queue-g4-final-status.json"


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


def _run_final(directory: Path) -> tuple[GrammarV3FollowupService, dict]:
    base_config = _load(BASE_CONFIG)
    base = _service(directory, base_config)
    assert base.start()["executed"] == 6
    base.stop()
    assert _service(directory, base_config).resume()["executed"] == 0

    g3 = _service(directory, _load(G3_CONFIG))
    g3_cycle = g3.resume()
    assert g3_cycle["executed"] == 2
    assert g3_cycle["status"]["deferred_count"] == 2

    final = _service(directory, _load(G4_CONFIG))
    final_cycle = final.resume()
    assert final_cycle["admission"] == {
        "accepted": 2,
        "duplicate": 8,
        "backpressured": 0,
        "budget_rejected": 0,
    }
    assert final_cycle["executed"] == 2
    assert final_cycle["deferred_resolved"] == 2
    assert final_cycle["deferred_accepted"] == 0
    assert final_cycle["deferred_duplicate"] == 0
    return final, final.export()


def test_final_epoch_reviews_g4_and_records_the_exact_formal_pass(tmp_path: Path) -> None:
    final, status = _run_final(tmp_path / "service")
    assert status["lifecycle"] == "idle"
    assert status["cycle_count"] == 4
    assert status["admitted_count"] == 10
    assert status["processed_count"] == 10
    assert status["deferred_count"] == 0
    assert status["deferred_packets"] == []
    assert status["packet_state_counts"] == {
        "deferred_missing_evaluator": 0,
        "succeeded": 10,
    }
    assert status["reviewed_evaluator_invocation_count"] == 10
    assert status["missing_evaluator_executions"] == 0
    assert status["candidate_scientific_decisions_changed"] == 1
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["paid_llm_spend_usd"] == 0.0

    with final.coordinator.connect() as connection:
        rows = connection.execute(
            "SELECT result_json FROM work WHERE result_json IS NOT NULL"
        ).fetchall()
    results = [json.loads(row[0]) for row in rows]
    g4 = {
        result["task_type"]: result for result in results if result["task_type"].startswith("g4_")
    }
    assert set(g4) == {
        "g4_global_lapse_invertibility",
        "g4_global_positive_energy",
    }
    lapse = g4["g4_global_lapse_invertibility"]["reviewed_evidence"]
    energy = g4["g4_global_positive_energy"]["reviewed_evidence"]
    assert lapse["blocker"] is None
    assert lapse["global_nonunitary_lapse_status"] == (
        "pass_in_Einstein_frame_generalized_harmonic_gauge"
    )
    assert energy["blocker"] is None
    assert energy["positive_energy_status"] == (
        "pass_from_hash_bound_predecessor_in_global_equivalent_frame"
    )
    assert energy["remaining_formal_blocker"] is None
    assert all(result["decision"] == "pass" for result in g4.values())

    restarted = _service(tmp_path / "service", _load(G4_CONFIG))
    assert restarted.export() == status


def test_final_epoch_replay_roots_are_deterministic(tmp_path: Path) -> None:
    _, first = _run_final(tmp_path / "first")
    _, second = _run_final(tmp_path / "second")
    for key in (
        "queue_registry_root_sha256",
        "completed_work_records_root_sha256",
        "deferred_packet_root_sha256",
        "content_sha256",
    ):
        assert first[key] == second[key]


def test_final_epoch_predecessor_and_descriptor_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "predecessor"
    base_config = _load(BASE_CONFIG)
    base = _service(directory, base_config)
    base.start()
    base.stop()
    _service(directory, base_config).resume()
    g3 = _service(directory, _load(G3_CONFIG))
    g3.resume()

    config = _load(G4_CONFIG)
    config["predecessor_service_epoch"]["deferred_packet_root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predecessor roots"):
        _service(directory, config)

    config = _load(G4_CONFIG)
    config["evaluator_descriptor_allowlist"][-1]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptors differ|descriptor hash"):
        _service(tmp_path / "descriptor", config)


def test_final_epoch_portable_exports_are_exact(tmp_path: Path) -> None:
    final, exported = _run_final(tmp_path / "portable")
    assert exported == _load(PORTABLE)
    assert portable_followup_status(final.queue.status()) == _load(QUEUE_PORTABLE)
