import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.promotion_dossier import (
    build_promotion_dossiers,
    write_promotion_dossiers,
)
from sigma_theory_compiler.promotion_orchestrator import (
    ELIGIBILITY,
    EVIDENCE_SCHEMA,
    PromotionOrchestrator,
    evaluator_binding,
)


def pass_evaluator(candidate: dict, context: dict) -> dict:
    return {
        "decision": "pass",
        "check": "exact_symbolic_identity",
        "candidate_id": candidate["candidate_id"],
        "input_lineage_sha256": context["input_lineage_sha256"],
        "data_eligibility": dict(ELIGIBILITY),
    }


def reject_evaluator(candidate: dict, context: dict) -> dict:
    return {
        "decision": "reject",
        "failure": "negative_kinetic_eigenvalue",
        "candidate_id": candidate["candidate_id"],
        "input_lineage_sha256": context["input_lineage_sha256"],
        "data_eligibility": dict(ELIGIBILITY),
    }


def _descriptor(evaluator_id: str, callback: str) -> dict:
    artifact = Path(__file__).resolve()
    return {
        "evaluator_id": evaluator_id,
        "version": "test-1.0",
        "callback": f"{__name__}:{callback}",
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "data_eligibility": dict(ELIGIBILITY),
    }


def _pipeline(symbolic: dict, formal: dict) -> dict:
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
                "evaluator_id": symbolic["evaluator_id"],
                "required_evaluator_binding_sha256": evaluator_binding(symbolic),
            },
            {
                "name": "formal_health",
                "category": "formal",
                "evaluator_id": formal["evaluator_id"],
                "required_evaluator_binding_sha256": evaluator_binding(formal),
            },
            {
                "name": "direct_observable_controls",
                "category": "observational",
                "evaluator_id": None,
                "required_evaluator_binding_sha256": None,
            },
        ],
    }


def _candidate(identifier: str, ordinal: int, terms: list[int]) -> dict:
    return {
        "candidate_id": identifier,
        "ordinal": ordinal,
        "term_ids": terms,
        "correction_expression": "+".join(f"t{term}" for term in terms),
        "data_eligibility": dict(ELIGIBILITY),
    }


def _evidence(identifier: str, ordinal: int) -> dict:
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "candidate_id": identifier,
        "ordinal": ordinal,
        "status": "pass",
        "source_result_sha256": "a" * 64,
        "status_root_sha256": "b" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _built_ledger(tmp_path: Path) -> tuple[Path, dict]:
    symbolic = _descriptor("symbolic-pass", "pass_evaluator")
    formal = _descriptor("formal-reject", "reject_evaluator")
    pipeline = _pipeline(symbolic, formal)
    database = tmp_path / "promotion.sqlite"
    orchestrator = PromotionOrchestrator(database, pipeline)
    orchestrator.register_evaluator(symbolic)
    orchestrator.register_candidate(_candidate("blocked-simple", 1, [1]), _evidence("blocked-simple", 1))
    orchestrator.register_candidate(_candidate("blocked-complex", 2, [1, 2]), _evidence("blocked-complex", 2))
    assert orchestrator.run_ready(maximum_tasks=10)["passed"] == 2
    orchestrator.register_evaluator(formal)
    orchestrator.register_candidate(_candidate("rejected", 3, [3]), _evidence("rejected", 3))
    outcome = orchestrator.run_ready(maximum_tasks=10)
    assert outcome["rejected"] == 3
    return database, pipeline


def test_dossier_verifies_lineage_explains_rejections_and_pareto_queue(tmp_path: Path) -> None:
    database, pipeline = _built_ledger(tmp_path)
    report = build_promotion_dossiers(database)
    assert report["pipeline_sha256"] == hashlib.sha256(
        json.dumps(pipeline, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert report["candidate_count"] == 3
    assert report["terminal_rejection_count"] == 3
    assert report["work_queue_count"] == 0
    assert all(
        item["disposition"] == "terminally_rejected"
        for item in report["candidate_dossiers"]
    )
    rejected = next(item for item in report["candidate_dossiers"] if item["candidate_id"] == "rejected")
    assert rejected["first_nonpass"]["stage_name"] == "formal_health"
    assert "failure=negative_kinetic_eigenvalue" in rejected["first_nonpass"]["reasons"]
    body = {key: value for key, value in report.items() if key != "content_sha256"}
    assert report["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_blocked_candidates_are_ranked_for_work_not_truth(tmp_path: Path) -> None:
    symbolic = _descriptor("symbolic-pass", "pass_evaluator")
    formal = _descriptor("formal-pass", "pass_evaluator")
    pipeline = _pipeline(symbolic, formal)
    database = tmp_path / "promotion.sqlite"
    orchestrator = PromotionOrchestrator(database, pipeline)
    orchestrator.register_evaluator(symbolic)
    for identifier, ordinal, terms in (
        ("simple", 1, [1]),
        ("complex", 2, [1, 2, 3]),
    ):
        orchestrator.register_candidate(_candidate(identifier, ordinal, terms), _evidence(identifier, ordinal))
    orchestrator.run_ready(maximum_tasks=10)
    report = build_promotion_dossiers(database)
    assert report["terminal_rejection_count"] == 0
    assert report["work_queue_count"] == 2
    queue = {item["candidate_id"]: item for item in report["pareto_work_queue"]}
    assert queue["simple"]["pareto_front"] == 1
    assert queue["complex"]["pareto_front"] == 2
    assert queue["simple"]["next_gate"] == "formal_health"
    assert report["data_eligibility"]["passed"] is True
    assert "not a truth probability" in report["interpretation"]
    output = tmp_path / "dossiers.json"
    write_promotion_dossiers(database, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_dossier_rejects_self_inconsistent_result_lineage(tmp_path: Path) -> None:
    database, _ = _built_ledger(tmp_path)
    import sqlite3

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE candidate_stages SET result_sha256=? "
            "WHERE candidate_id='rejected' AND stage_name='formal_health'",
            ("0" * 64,),
        )
    with pytest.raises(ValueError, match="gate-result lineage is invalid"):
        build_promotion_dossiers(database)


def test_promotion_dossier_cli_writes_verified_report(tmp_path: Path) -> None:
    from sigma_theory_compiler.cli import main

    database, _ = _built_ledger(tmp_path)
    output = tmp_path / "cli-dossier.json"
    assert main(["promotion-dossier", "--database", str(database), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_count"] == 3
