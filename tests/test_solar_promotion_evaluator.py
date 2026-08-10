import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.adm_dirac_promotion_evaluator import (
    KNOWN_ANSWER_BUNDLE as EA_BUNDLE,
)
from sigma_theory_compiler.adm_dirac_promotion_evaluator import (
    bundle_binding as adm_bundle_binding,
)
from sigma_theory_compiler.promotion_orchestrator import (
    ELIGIBILITY,
    EVIDENCE_SCHEMA,
    PromotionOrchestrator,
    evaluator_binding,
)
from sigma_theory_compiler.solar_promotion_evaluator import (
    GR_SOLAR_BUNDLE,
    _golden_status,
    _load_bundle,
    bundle_binding,
    solar_known_answer_evaluator,
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "configs" / "promotion_pipeline_fail_closed.json"
STATIC_DESCRIPTOR = ROOT / "configs" / "promotion_static_lift_evaluator.json"
FORMAL_DESCRIPTOR = ROOT / "configs" / "promotion_adm_dirac_principal_evaluator.json"
SOLAR_DESCRIPTOR = ROOT / "configs" / "promotion_solar_known_answer_evaluator.json"
STATUS = ROOT / "runs" / "engine" / "solar-known-answer-evaluator-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor(path: Path) -> dict:
    descriptor = _load(path)
    descriptor["artifact_path"] = str((ROOT / descriptor["artifact_path"]).resolve())
    return descriptor


def _context() -> dict:
    return {
        "stage_name": "solar_known_answer_controls",
        "category": "observational",
        "attempt": 1,
        "input_lineage_sha256": "a" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _solar_provenance() -> dict:
    return {
        "bundle_id": GR_SOLAR_BUNDLE["bundle_id"],
        "bundle_binding_sha256": bundle_binding(GR_SOLAR_BUNDLE),
        "input_action_sha256": GR_SOLAR_BUNDLE["input_action_sha256"],
    }


def _gr_candidate() -> dict:
    return {
        "candidate_id": GR_SOLAR_BUNDLE["candidate_id"],
        "ordinal": 0,
        "correction_expression": "GR",
        "solar_control_provenance": _solar_provenance(),
        "data_eligibility": dict(ELIGIBILITY),
    }


def _ea_candidate() -> dict:
    return {
        "candidate_id": EA_BUNDLE["candidate_id"],
        "ordinal": 0,
        "correction_expression": "x",
        "covariant_action_provenance": {
            "bundle_id": EA_BUNDLE["bundle_id"],
            "bundle_binding_sha256": adm_bundle_binding(EA_BUNDLE),
            "input_action_sha256": EA_BUNDLE["input_action_sha256"],
        },
        "data_eligibility": dict(ELIGIBILITY),
    }


def test_exact_gr_solar_known_answer_bundle_passes_without_opening_data() -> None:
    result = solar_known_answer_evaluator(_gr_candidate(), _context())
    assert result["decision"] == "pass"
    assert set(result["golden_statuses"].values()) == {"pass"}
    assert len(result["golden_statuses"]) == 5
    assert result["input_action_sha256"] == GR_SOLAR_BUNDLE["input_action_sha256"]
    assert result["data_eligibility"] == ELIGIBILITY
    assert "no candidate observation is opened" in result["scope"]


def test_discovery_candidate_blocks_and_known_answer_provenance_cannot_be_reused() -> None:
    blocked = solar_known_answer_evaluator(_ea_candidate(), _context())
    assert blocked["decision"] == "blocked"
    assert blocked["blocker"] == "missing_exact_action_bound_solar_control_bundle"

    reused = _gr_candidate()
    reused["candidate_id"] = "STC2-false-solar-control-reuse"
    with pytest.raises(ValueError, match="cannot be attached"):
        solar_known_answer_evaluator(reused, _context())

    tampered = _gr_candidate()
    tampered["solar_control_provenance"]["input_action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        solar_known_answer_evaluator(tampered, _context())


def test_golden_accounting_negative_and_sealed_prerequisite_are_executable() -> None:
    reference = _load_bundle(GR_SOLAR_BUNDLE)["reference"]
    corrupted = copy.deepcopy(reference)
    corrupted["golden_checks"][1]["status"] = "fail"
    statuses = _golden_status(corrupted)
    assert statuses["gr_ppn_recovery"] == "fail"
    corrupted["counts"]["passed"] = 4
    corrupted["counts"]["failed"] = 1
    with pytest.raises(ValueError, match="accounting mismatch"):
        _golden_status(corrupted)


def test_pipeline_binding_reaches_solar_and_blocks_unmapped_einstein_aether(
    tmp_path: Path,
) -> None:
    pipeline = _load(PIPELINE)
    descriptors = [
        _descriptor(STATIC_DESCRIPTOR),
        _descriptor(FORMAL_DESCRIPTOR),
        _descriptor(SOLAR_DESCRIPTOR),
    ]
    assert evaluator_binding(descriptors[-1]) == pipeline["stages"][3][
        "required_evaluator_binding_sha256"
    ]
    orchestrator = PromotionOrchestrator(tmp_path / "solar-promotion.sqlite", pipeline)
    for descriptor in descriptors:
        orchestrator.register_evaluator(descriptor)
    candidate = _ea_candidate()
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
    assert orchestrator.run_ready(maximum_tasks=4) == {
        "evaluated": 3,
        "passed": 2,
        "rejected": 0,
        "blocked": 1,
    }
    status = orchestrator.status()
    assert status["stages"]["solar_known_answer_controls"]["counts"] == {"blocked": 1}
    assert status["candidates"][0]["blocker"] == (
        "missing_exact_action_bound_solar_control_bundle"
    )


def test_review_status_artifact_matches_actual_evaluator_decisions() -> None:
    artifact = _load(STATUS)
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert artifact["known_answer_control"] == solar_known_answer_evaluator(
        _gr_candidate(), _context()
    )
    assert artifact["unmapped_candidate_control"] == solar_known_answer_evaluator(
        _ea_candidate(), _context()
    )
    assert artifact["data_eligibility"] == ELIGIBILITY
