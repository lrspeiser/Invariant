import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.adm_dirac_promotion_evaluator import (
    GENERATED_REJECT_BUNDLE,
    KNOWN_ANSWER_BUNDLE,
    adm_dirac_principal_health_evaluator,
    bundle_binding,
)
from sigma_theory_compiler.high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
)
from sigma_theory_compiler.promotion_orchestrator import (
    ELIGIBILITY,
    EVIDENCE_SCHEMA,
    PromotionOrchestrator,
    evaluator_binding,
)
from sigma_theory_compiler.survivors import iter_survivors

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "configs" / "promotion_pipeline_fail_closed.json"
STATIC_DESCRIPTOR = ROOT / "configs" / "promotion_static_lift_evaluator.json"
FORMAL_DESCRIPTOR = ROOT / "configs" / "promotion_adm_dirac_principal_evaluator.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
MANIFEST = ROOT / "runs" / "knowledge-base" / "survivor-export-smoke.json"
SURVIVORS = ROOT / "runs" / "knowledge-base" / "survivors-smoke"
STATUS = ROOT / "runs" / "engine" / "adm-dirac-principal-evaluator-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor(path: Path) -> dict:
    descriptor = _load(path)
    descriptor["artifact_path"] = str((ROOT / descriptor["artifact_path"]).resolve())
    return descriptor


def _context() -> dict:
    return {
        "stage_name": "adm_dirac_principal_health",
        "category": "formal",
        "attempt": 1,
        "input_lineage_sha256": "a" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _provenance(bundle: dict) -> dict:
    return {
        "bundle_id": bundle["bundle_id"],
        "bundle_binding_sha256": bundle_binding(bundle),
        "input_action_sha256": bundle["input_action_sha256"],
    }


def _known_answer_candidate() -> dict:
    return {
        "candidate_id": KNOWN_ANSWER_BUNDLE["candidate_id"],
        "ordinal": 0,
        "correction_expression": "x",
        "covariant_action_provenance": _provenance(KNOWN_ANSWER_BUNDLE),
        "data_eligibility": dict(ELIGIBILITY),
    }


def _generated_reject_candidate() -> dict:
    return {
        "candidate_id": GENERATED_REJECT_BUNDLE["candidate_id"],
        "ordinal": GENERATED_REJECT_BUNDLE["ordinal"],
        "correction_expression": GENERATED_REJECT_BUNDLE["correction_expression"],
        "covariant_action_provenance": _provenance(GENERATED_REJECT_BUNDLE),
        "data_eligibility": dict(ELIGIBILITY),
    }


def _real_survivor_candidates(limit: int = 3) -> list[dict]:
    generator = _load(GENERATOR)
    basis = build_basis(int(generator["basis_count"]))
    candidates = []
    for index, survivor in enumerate(iter_survivors(MANIFEST, SURVIVORS)):
        if index >= limit:
            break
        decoded = decode_ordinal(
            int(generator["basis_count"]),
            int(generator["max_action_terms"]),
            int(survivor["ordinal"]),
        )
        candidates.append(
            {
                "candidate_id": candidate_id(str(generator["protocol_version"]), decoded),
                "ordinal": int(survivor["ordinal"]),
                "term_ids": list(decoded["term_ids"]),
                "signs": list(decoded["signs"]),
                "correction_expression": correction_expression(decoded, basis),
                "data_eligibility": dict(ELIGIBILITY),
            }
        )
    return candidates


def test_hash_bound_known_answer_passes_and_generated_candidate_rejects() -> None:
    control = adm_dirac_principal_health_evaluator(_known_answer_candidate(), _context())
    assert control["decision"] == "pass"
    assert set(control["gate_statuses"].values()) == {"pass"}
    assert control["bundle_binding_sha256"] == bundle_binding(KNOWN_ANSWER_BUNDLE)

    candidate = adm_dirac_principal_health_evaluator(
        _generated_reject_candidate(), _context()
    )
    assert candidate["decision"] == "reject"
    assert candidate["rejected_gates"] == ["higher_jet_regularity"]
    assert candidate["gate_statuses"]["generated_dirac_ir"] == "unresolved"
    assert candidate["gate_statuses"]["generated_principal_ir"] == "unresolved"
    assert control["data_eligibility"] == candidate["data_eligibility"] == ELIGIBILITY


def test_real_stream_survivors_block_without_inventing_covariant_dynamics() -> None:
    candidates = _real_survivor_candidates()
    assert len(candidates) == 3
    results = [
        adm_dirac_principal_health_evaluator(candidate, _context())
        for candidate in candidates
    ]
    assert [result["decision"] for result in results] == ["blocked"] * 3
    assert {result["blocker"] for result in results} == {
        "missing_exact_candidate_to_covariant_action_map"
    }
    assert all(result["input_action_sha256"] is None for result in results)


def test_provenance_tampering_and_control_reuse_fail_closed() -> None:
    tampered = _generated_reject_candidate()
    tampered["covariant_action_provenance"]["input_action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        adm_dirac_principal_health_evaluator(tampered, _context())

    reused_control = _known_answer_candidate()
    reused_control["candidate_id"] = "STC2-false-control-reuse"
    with pytest.raises(ValueError, match="cannot be attached"):
        adm_dirac_principal_health_evaluator(reused_control, _context())


def test_pipeline_binding_and_formal_gate_execute_end_to_end(tmp_path: Path) -> None:
    pipeline = _load(PIPELINE)
    static_descriptor = _descriptor(STATIC_DESCRIPTOR)
    formal_descriptor = _descriptor(FORMAL_DESCRIPTOR)
    assert evaluator_binding(formal_descriptor) == pipeline["stages"][2][
        "required_evaluator_binding_sha256"
    ]
    orchestrator = PromotionOrchestrator(tmp_path / "formal-promotion.sqlite", pipeline)
    orchestrator.register_evaluator(static_descriptor)
    orchestrator.register_evaluator(formal_descriptor)
    candidate = _known_answer_candidate()
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "ordinal": candidate["ordinal"],
        "status": "pass",
        "source_result_sha256": "b" * 64,
        "status_root_sha256": "c" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }
    orchestrator.register_candidate(candidate, evidence)
    assert orchestrator.run_ready(maximum_tasks=3) == {
        "evaluated": 2,
        "passed": 2,
        "rejected": 0,
        "blocked": 0,
    }
    status = orchestrator.status()
    assert status["stages"]["adm_dirac_principal_health"]["counts"] == {"passed": 1}
    assert status["candidates"][0]["stage_name"] == "solar_known_answer_controls"
    assert status["candidates"][0]["blocker"] == "unimplemented_gate_fail_closed"


def test_review_status_artifact_matches_actual_evaluator_decisions() -> None:
    artifact = _load(STATUS)
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert artifact["known_answer_control"] == adm_dirac_principal_health_evaluator(
        _known_answer_candidate(), _context()
    )
    assert artifact["generated_candidate_negative_control"] == (
        adm_dirac_principal_health_evaluator(_generated_reject_candidate(), _context())
    )
    expected_real = [
        adm_dirac_principal_health_evaluator(candidate, _context())
        for candidate in _real_survivor_candidates()
    ]
    assert artifact["real_survivor_decisions"] == expected_real
    assert artifact["data_eligibility"] == ELIGIBILITY
