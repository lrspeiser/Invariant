from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
)
from sigma_theory_compiler.future_aether_weighted_ift_contract_gate import (
    BLOCKER,
    REQUIRED_CONTRACT_FIELDS,
    build_future_aether_weighted_ift_contract_gate,
    evaluate_quantitative_ift_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_weighted_ift_contract_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-weighted-ift-contract-gate.json"
INVERSE_PATH = ROOT / "runs/engine/future-aether-regular-adm-inverse-margin-gate.json"
WEAK_PATH = ROOT / "runs/engine/future-aether-weak-field-ae-constraint-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_weighted_ift_contract_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        BLOCKER: 3,
        CHARACTERISTIC_BLOCKER: 11,
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0


def test_both_predecessor_action_and_record_bindings_are_exact(rebuilt: dict) -> None:
    inverse = json.loads(INVERSE_PATH.read_text(encoding="utf-8"))
    weak = json.loads(WEAK_PATH.read_text(encoding="utf-8"))
    inverse_records = {item["candidate_id"]: item for item in inverse["candidate_records"]}
    weak_records = {item["candidate_id"]: item for item in weak["candidate_records"]}
    for item in rebuilt["candidate_records"]:
        source_inverse = inverse_records[item["candidate_id"]]
        source_weak = weak_records[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == source_inverse["typed_action_ir_sha256"]
        assert item["typed_action_ir_sha256"] == source_weak["typed_action_ir_sha256"]
        assert item["source_inverse_margin_record_sha256"] == source_inverse["content_sha256"]
        assert item["source_weak_field_record_sha256"] == source_weak["content_sha256"]


def test_eleven_characteristic_candidates_are_preserved_without_ift_inference(
    rebuilt: dict,
) -> None:
    forced = [
        item
        for item in rebuilt["candidate_records"]
        if item["first_blocker"] == CHARACTERISTIC_BLOCKER
    ]
    assert len(forced) == 11
    for item in forced:
        assert item["weighted_ift_contract_certificate"] is None
        assert (
            item["gate_ledger"]["reference_conformal_York_and_Aether_blocks"]["status"]
            == "not_reached"
        )


def test_three_regular_records_expose_controls_and_exact_missing_contract(rebuilt: dict) -> None:
    regular = [item for item in rebuilt["candidate_records"] if item["first_blocker"] == BLOCKER]
    assert len(regular) == 3
    assert rebuilt["reference_conformal_York_Aether_block_control_count"] == 3
    assert rebuilt["typed_weighted_operator_contract_complete_count"] == 0
    assert rebuilt["missing_contract_field_counts"] == {key: 3 for key in REQUIRED_CONTRACT_FIELDS}
    for item in regular:
        certificate = item["weighted_ift_contract_certificate"]
        controls = certificate["available_exact_controls"]
        contract = certificate["typed_weighted_operator_contract"]
        assert controls["scalar_conformal_reference_solution_exists"] is True
        assert controls["vector_York_reference_solution_exists"] is True
        assert Fraction(controls["uniform_Aether_Legendre_block_inverse_bound"]) > 0
        assert Fraction(controls["strict_negative_source_energy_margin_over_pi"]) > 0
        assert contract["required_fields"] == list(REQUIRED_CONTRACT_FIELDS)
        assert contract["missing_fields"] == {
            key: "not_registered" for key in REQUIRED_CONTRACT_FIELDS
        }
        assert contract["complete"] is False


def test_exact_positive_control_closes_all_three_ift_inequalities() -> None:
    result = evaluate_quantitative_ift_contract(
        {
            "reference_inverse_norm": "2",
            "operator_perturbation_norm": "1/8",
            "seed_nonlinear_constraint_residual_norm": "1/64",
            "nonlinear_second_derivative_majorant": "1/64",
            "completed_boundary_first_derivative_bound": "1/16",
            "completed_boundary_second_derivative_bound": "1/16",
            "negative_boundary_energy_margin": "1",
        }
    )
    assert result["neumann_product"] == "1/4"
    assert result["full_inverse_bound"] == "8/3"
    assert result["solution_radius_bound"] == "1/12"
    assert result["all_conditions_pass"] is True
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    assert result["content_sha256"] == _sha(body)


def test_exact_negative_control_stops_at_neumann_obstruction() -> None:
    result = evaluate_quantitative_ift_contract(
        {
            "reference_inverse_norm": "2",
            "operator_perturbation_norm": "1/2",
            "seed_nonlinear_constraint_residual_norm": "0",
            "nonlinear_second_derivative_majorant": "0",
            "completed_boundary_first_derivative_bound": "0",
            "completed_boundary_second_derivative_bound": "0",
            "negative_boundary_energy_margin": "1",
        }
    )
    assert result["neumann_product"] == "1"
    assert result["neumann_inverse_pass"] is False
    assert result["full_inverse_bound"] is None
    assert result["all_conditions_pass"] is False


def test_no_full_operator_remainder_boundary_or_rejection_overclaim(rebuilt: dict) -> None:
    assert rebuilt["full_weighted_operator_isomorphism_pass_count"] == 0
    assert rebuilt["nonlinear_remainder_bound_pass_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0
    for item in rebuilt["candidate_records"]:
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_weighted_ift_contract_gate_completed"] is True
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)
        assert item["data_eligibility"] == rebuilt["data_eligibility"]
        assert item["observational_data_opened"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                data_eligibility={
                    **config["data_eligibility"],
                    "observational_data_opened": True,
                }
            ),
            "eligibility is open",
        ),
        (lambda config: config.update(observational_authorization=True), "opened observations"),
        (lambda config: config.update(external_paid_llm_calls=True), "enabled paid LLM calls"),
        (
            lambda config: config["budget"].update(maximum_contract_fields=10),
            "budget is not exact",
        ),
        (
            lambda config: config["source_inverse_margin_artifact"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_budget_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_weighted_ift_contract_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_inverse_margin_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_weighted_ift_contract_gate(config, ROOT)
