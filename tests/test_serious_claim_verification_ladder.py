from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler import serious_claim_verification_ladder as L
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / L.OUTPUT_PATH


def _reseal_stage(stage: dict) -> None:
    body = {key: item for key, item in stage.items() if key != "content_sha256"}
    stage["content_sha256"] = canonical_sha256(body)


def _reseal_chain(chain: dict) -> None:
    body = {key: item for key, item in chain.items() if key != "content_sha256"}
    chain["content_sha256"] = canonical_sha256(body)


def _reseal_linked_stages(chain: dict) -> None:
    previous = None
    for stage in chain["stages"]:
        stage["previous_stage_sha256"] = previous
        _reseal_stage(stage)
        previous = stage["content_sha256"]
    _reseal_chain(chain)


def _reseal_receipt(receipt: dict) -> None:
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = canonical_sha256(body)


def test_stored_ladder_is_candidate_bound_ordered_and_fail_closed() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    L.validate_receipt(value, ROOT)
    chain = value["known_control_chain"]
    assert [stage["backend"] for stage in chain["stages"]] == list(L.REQUIRED_STAGES)
    assert chain["stages"][0]["previous_stage_sha256"] is None
    for left, right in zip(chain["stages"], chain["stages"][1:]):
        assert right["previous_stage_sha256"] == left["content_sha256"]
    assert value["summary"] == {
        "backend_mathematical_mutations_rejected": 10,
        "known_control_candidates": 2,
        "lean_kernel_mutation_artifact_bound": True,
        "negative_controls_blocked": 2,
        "required_stage_order": list(L.REQUIRED_STAGES),
        "structural_mutations_rejected": 5,
        "status": "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION",
    }
    assert value["release_gate"]["serious_claims_released"] == 0
    assert not value["claims"]["serious_claim_released"]


def test_first_four_backends_rerun_positives_and_reject_candidate_specific_wrong_formulas() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    stages = {stage["backend"]: stage for stage in value["known_control_chain"]["stages"]}
    expected_results = {
        "exact_arithmetic": "MISMATCH_WITNESS",
        "cas": "NONZERO_NORMAL_FORM",
        "smt": "SAT_COUNTERMODEL",
        "interval": "ZERO_EXCLUDED",
    }
    for backend, expected_result in expected_results.items():
        evidence = stages[backend]["evidence"]
        assert len(evidence["controls"]) == 2
        assert all(row["independent_positive_reexecution"] for row in evidence["controls"])
        assert len(evidence["mathematical_mutation_controls"]) == 2
        for mutation in evidence["mathematical_mutation_controls"]:
            assert mutation["backend"] == backend
            assert mutation["backend_result"] == expected_result
            assert mutation["mutation_operator"] == "add_exact_unit"
            assert mutation["witness"]["residual"] == "1/1"
            assert mutation["wrong_formula_rejected"] is True


def test_lean_wrong_formula_kernel_artifact_binds_two_ci_rejections() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    lean = value["known_control_chain"]["stages"][-1]["evidence"]
    control = lean["wrong_formula_kernel_control"]
    assert control["artifact_bound"] is True
    assert control["required_for_serious_claim"] is True
    assert control["status"] == "PASS_CI_KERNEL_REJECTION_ARTIFACT_BOUND"
    assert control["artifact_content_sha256"] == (
        "491926a5f4f1613b698fd612d347c92cecf4298edcf5629af06c15f90b2af004"
    )
    assert control["artifact_registry_binding"]["artifact_id"] == 9505070169
    assert control["artifact_registry_binding"]["run_id"] == 32683958812
    assert control["artifact_registry_binding"]["head_sha"] == (
        "58bbcb40b644c2cf613e20ae4d50f2e051b134d4"
    )
    assert len(control["controls"]) == 2
    assert all(item["outcome"] == "REJECTED_BY_LEAN_KERNEL" for item in control["controls"])
    assert value["claims"]["all_five_backend_mutations_complete"] is True
    assert value["release_gate"]["lean_wrong_formula_artifact_required"] is True
    assert value["release_gate"]["serious_claims_released"] == 0


def test_lean_ci_registry_substitution_fails_against_downloaded_artifact() -> None:
    config = deepcopy(L.load_config(ROOT))
    config["mathematical_wrong_formula_control"]["lean_artifact_binding"]["head_sha"] = "0" * 40
    with pytest.raises(L.SeriousClaimVerificationError, match="registry binding"):
        L._lean_ci_artifact(ROOT, config)


def test_semantically_resealed_wrong_formula_witness_tamper_fails_against_sources() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    witness = value["known_control_chain"]["stages"][0]["evidence"][
        "mathematical_mutation_controls"
    ][0]["witness"]
    witness["original_output"] = "101/1"
    witness["mutated_output"] = "102/1"
    _reseal_linked_stages(value["known_control_chain"])
    _reseal_receipt(value)
    with pytest.raises(L.SeriousClaimVerificationError, match="evidence changed"):
        L.validate_receipt(value, ROOT)


def test_bounded_unknowns_do_not_acquire_a_partial_ladder_release() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert {item["benchmark_id"] for item in value["negative_controls"]} == {
        "external.authority-oeis-005132",
        "external.authority-oeis-002858",
    }
    for item in value["negative_controls"]:
        assert {"cas", "smt", "interval", "lean"}.issubset(item["missing_or_failed_backends"])
        assert item["status"] == "BLOCKED_INCOMPLETE_BACKEND_LADDER"
        assert item["serious_claim_released"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_stage",
        "reordered_stages",
        "candidate_scope_substitution",
        "broken_previous_stage_link",
        "backend_unavailable",
    ],
)
def test_semantically_resealed_chain_mutations_fail(mutation: str) -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    chain = value["known_control_chain"]
    if mutation == "missing_stage":
        del chain["stages"][2]
    elif mutation == "reordered_stages":
        chain["stages"][1], chain["stages"][2] = chain["stages"][2], chain["stages"][1]
    elif mutation == "candidate_scope_substitution":
        chain["stages"][3]["candidate_scope_sha256"] = "0" * 64
        _reseal_stage(chain["stages"][3])
    elif mutation == "broken_previous_stage_link":
        chain["stages"][4]["previous_stage_sha256"] = "0" * 64
        _reseal_stage(chain["stages"][4])
    elif mutation == "backend_unavailable":
        chain["stages"][4]["backend_available"] = False
        _reseal_stage(chain["stages"][4])
    _reseal_chain(chain)
    _reseal_receipt(value)
    with pytest.raises(L.SeriousClaimVerificationError):
        L.validate_receipt(value)


def test_current_sources_rebuild_the_stored_receipt_exactly() -> None:
    stored = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert L.build_receipt(ROOT) == stored
