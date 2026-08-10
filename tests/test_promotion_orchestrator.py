import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.promotion_orchestrator import (
    ELIGIBILITY,
    EVIDENCE_SCHEMA,
    PromotionOrchestrator,
    evaluator_binding,
    validate_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
MANIFEST = ROOT / "runs" / "knowledge-base" / "survivor-export-smoke.json"
SURVIVORS = ROOT / "runs" / "knowledge-base" / "survivors-smoke"


def pass_evaluator(candidate: dict, context: dict) -> dict:
    return {
        "decision": "pass",
        "candidate_id": candidate["candidate_id"],
        "input_lineage_sha256": context["input_lineage_sha256"],
        "check": "deterministic_test_pass",
        "data_eligibility": dict(ELIGIBILITY),
    }


def reject_evaluator(candidate: dict, context: dict) -> dict:
    return {
        "decision": "reject",
        "candidate_id": candidate["candidate_id"],
        "input_lineage_sha256": context["input_lineage_sha256"],
        "check": "deterministic_negative_control",
        "data_eligibility": dict(ELIGIBILITY),
    }


def blocked_evaluator(candidate: dict, context: dict) -> dict:
    return {
        "decision": "blocked",
        "blocker": "missing_exact_covariant_adapter",
        "candidate_id": candidate["candidate_id"],
        "input_lineage_sha256": context["input_lineage_sha256"],
        "data_eligibility": dict(ELIGIBILITY),
    }


def _descriptor(evaluator_id: str, callback_name: str) -> dict:
    artifact = Path(__file__).resolve()
    return {
        "evaluator_id": evaluator_id,
        "version": "test-1.0",
        "callback": f"{__name__}:{callback_name}",
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "data_eligibility": dict(ELIGIBILITY),
    }


def _pipeline(descriptor: dict | None = None) -> dict:
    binding = evaluator_binding(descriptor) if descriptor else None
    return {
        "schema_version": "sigma-promotion-pipeline-1.0",
        "external_paid_llm_calls": False,
        "maximum_evaluator_attempts": 2,
        "data_eligibility": dict(ELIGIBILITY),
        "stages": [
            {
                "name": "sampled_static",
                "category": "cheap",
                "evaluator_id": None,
                "required_evaluator_binding_sha256": None,
            },
            {
                "name": "symbolic_health",
                "category": "symbolic",
                "evaluator_id": descriptor["evaluator_id"] if descriptor else None,
                "required_evaluator_binding_sha256": binding,
            },
            {
                "name": "formal_health",
                "category": "formal",
                "evaluator_id": None,
                "required_evaluator_binding_sha256": None,
            },
            {
                "name": "direct_observable_controls",
                "category": "observational",
                "evaluator_id": None,
                "required_evaluator_binding_sha256": None,
            },
        ],
    }


def _candidate(identifier: str = "candidate-1") -> dict:
    return {
        "candidate_id": identifier,
        "ordinal": 42,
        "formula": "R+epsilon*X",
        "data_eligibility": dict(ELIGIBILITY),
    }


def _evidence(identifier: str = "candidate-1") -> dict:
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "candidate_id": identifier,
        "ordinal": 42,
        "status": "pass",
        "source_result_sha256": "a" * 64,
        "status_root_sha256": "b" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def test_missing_gate_blocks_then_exact_evaluator_reopens_and_hash_chains(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor("symbolic-pass", "pass_evaluator")
    pipeline = _pipeline(descriptor)
    database = tmp_path / "promotion.sqlite"
    orchestrator = PromotionOrchestrator(database, pipeline)
    initial = orchestrator.register_candidate(_candidate(), _evidence())
    before = orchestrator.status()
    assert before["candidates"][0]["stage_name"] == "symbolic_health"
    assert before["candidates"][0]["state"] == "blocked"
    assert before["candidates"][0]["blocker"] == "hash_bound_evaluator_not_registered"

    binding = orchestrator.register_evaluator(descriptor)
    assert binding == pipeline["stages"][1]["required_evaluator_binding_sha256"]
    outcome = orchestrator.run_ready()
    assert outcome == {"evaluated": 1, "passed": 1, "rejected": 0, "blocked": 0}
    after = orchestrator.status()
    assert after["candidates"][0]["stage_name"] == "formal_health"
    assert after["candidates"][0]["state"] == "blocked"
    assert after["candidates"][0]["blocker"] == "unimplemented_gate_fail_closed"
    assert after["unimplemented_gates_fail_closed"] == [
        "formal_health",
        "direct_observable_controls",
    ]
    with orchestrator.connect() as connection:
        row = connection.execute(
            "SELECT input_lineage_sha256,result_sha256,output_lineage_sha256 "
            "FROM candidate_stages WHERE candidate_id='candidate-1' AND stage_index=1"
        ).fetchone()
    assert initial != row["input_lineage_sha256"]
    assert all(
        len(row[key]) == 64
        for key in ("input_lineage_sha256", "result_sha256", "output_lineage_sha256")
    )

    # Candidate/evaluator replay and process restart are idempotent.
    content_hash = after["content_sha256"]
    assert orchestrator.register_candidate(_candidate(), _evidence()) == initial
    assert orchestrator.register_evaluator(descriptor) == binding
    resumed = PromotionOrchestrator(database, pipeline)
    assert resumed.run_ready()["evaluated"] == 0
    assert resumed.status()["content_sha256"] == content_hash
    path = resumed.write_status(tmp_path / "promotion-status.json")
    assert json.loads(path.read_text())["content_sha256"] == content_hash


def test_rejection_stops_every_downstream_gate(tmp_path: Path) -> None:
    descriptor = _descriptor("symbolic-reject", "reject_evaluator")
    orchestrator = PromotionOrchestrator(tmp_path / "reject.sqlite", _pipeline(descriptor))
    orchestrator.register_evaluator(descriptor)
    orchestrator.register_candidate(_candidate("negative"), _evidence("negative"))
    outcome = orchestrator.run_ready()
    assert outcome["rejected"] == 1
    with orchestrator.connect() as connection:
        rows = connection.execute(
            "SELECT stage_index,state,blocker FROM candidate_stages "
            "WHERE candidate_id='negative' ORDER BY stage_index"
        ).fetchall()
    assert [row["state"] for row in rows] == ["passed", "rejected", "blocked", "blocked"]
    assert rows[2]["blocker"] == rows[3]["blocker"] == "upstream_rejected"


def test_unresolved_evaluator_result_blocks_without_rejecting_candidate(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor("symbolic-blocked", "blocked_evaluator")
    orchestrator = PromotionOrchestrator(tmp_path / "blocked.sqlite", _pipeline(descriptor))
    orchestrator.register_evaluator(descriptor)
    orchestrator.register_candidate(_candidate("unresolved"), _evidence("unresolved"))
    assert orchestrator.run_ready() == {
        "evaluated": 1,
        "passed": 0,
        "rejected": 0,
        "blocked": 1,
    }
    status = orchestrator.status()
    candidate = status["candidates"][0]
    assert candidate["state"] == "blocked"
    assert candidate["blocker"] == "missing_exact_covariant_adapter"
    with orchestrator.connect() as connection:
        row = connection.execute(
            "SELECT result_sha256,output_lineage_sha256 FROM candidate_stages "
            "WHERE candidate_id='unresolved' AND stage_index=1"
        ).fetchone()
    assert len(row["result_sha256"]) == len(row["output_lineage_sha256"]) == 64


def test_interrupted_running_gate_recovers_and_replays_once(tmp_path: Path) -> None:
    descriptor = _descriptor("symbolic-recovery", "pass_evaluator")
    pipeline = _pipeline(descriptor)
    database = tmp_path / "recovery.sqlite"
    orchestrator = PromotionOrchestrator(database, pipeline)
    orchestrator.register_evaluator(descriptor)
    orchestrator.register_candidate(_candidate(), _evidence())
    with orchestrator.connect() as connection:
        connection.execute(
            "UPDATE candidate_stages SET state='running' WHERE candidate_id='candidate-1' "
            "AND stage_index=1"
        )
    resumed = PromotionOrchestrator(database, pipeline)
    assert resumed.run_ready()["passed"] == 1
    with resumed.connect() as connection:
        row = connection.execute(
            "SELECT state,attempt,result_sha256 FROM candidate_stages "
            "WHERE candidate_id='candidate-1' AND stage_index=1"
        ).fetchone()
    assert row["state"] == "passed"
    assert row["attempt"] == 1
    assert len(row["result_sha256"]) == 64


def test_hash_or_data_unsealing_never_registers(tmp_path: Path) -> None:
    descriptor = _descriptor("symbolic-pass", "pass_evaluator")
    orchestrator = PromotionOrchestrator(tmp_path / "negative.sqlite", _pipeline(descriptor))
    tampered = {**descriptor, "artifact_sha256": "0" * 64}
    with pytest.raises(ValueError, match="artifact hash"):
        orchestrator.register_evaluator(tampered)

    opened = json.loads(json.dumps(_pipeline(descriptor)))
    opened["data_eligibility"]["observational_data_opened"] = True
    with pytest.raises(ValueError, match="eligibility"):
        validate_pipeline(opened)
    candidate = _candidate()
    candidate["data_eligibility"]["dark_matter_or_halo_inputs"] = True
    with pytest.raises(ValueError, match="eligibility"):
        orchestrator.register_candidate(candidate, _evidence())


def test_bounded_real_rust_survivors_import_idempotently_and_stop_closed(tmp_path: Path) -> None:
    pipeline = json.loads(
        (ROOT / "configs" / "promotion_pipeline_fail_closed.json").read_text(encoding="utf-8")
    )
    orchestrator = PromotionOrchestrator(tmp_path / "rust-import.sqlite", pipeline)
    first = orchestrator.import_rust_survivors(
        MANIFEST, GENERATOR, SURVIVORS, maximum_candidates=3
    )
    second = orchestrator.import_rust_survivors(
        MANIFEST, GENERATOR, SURVIVORS, maximum_candidates=3
    )
    assert first == {"accepted": 3, "duplicates": 0, "limit": 3}
    assert second == {"accepted": 0, "duplicates": 3, "limit": 3}
    status = orchestrator.status()
    assert status["candidate_count"] == 3
    assert status["stages"]["sampled_static"]["counts"] == {"passed": 3}
    assert status["stages"]["covariant_symbolic_health"]["counts"] == {"blocked": 3}
    assert all(
        candidate["blocker"] == "hash_bound_evaluator_not_registered"
        for candidate in status["candidates"]
    )
