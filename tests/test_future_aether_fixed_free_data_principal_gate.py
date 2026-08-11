from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_fixed_free_data_principal_gate import (
    BLOCKER,
    build_future_aether_fixed_free_data_principal_gate,
    exact_augmented_symbol_control,
)
from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_fixed_free_data_principal_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-fixed-free-data-principal-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-weighted-reference-operator-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_fixed_free_data_principal_gate(_config(), ROOT)


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


def test_candidate_action_and_predecessor_bindings_are_preserved(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert item["compilation_receipt_sha256"] == predecessor["compilation_receipt_sha256"]
        assert item["source_weighted_reference_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_augmented_Aether_unknown_is_an_exact_nonelliptic_negative_control() -> None:
    control = exact_augmented_symbol_control()
    assert control["rank"] == 4
    assert control["column_count"] == 7
    assert control["right_kernel_dimension"] == 3
    assert control["Aether_second_order_columns_zero"] is True
    assert control["augmented_square_isomorphism_possible"] is False
    assert all(row[-3:] == ["0", "0", "0"] for row in control["second_order_symbol_matrix"])
    body = {key: value for key, value in control.items() if key != "content_sha256"}
    assert control["content_sha256"] == _sha(body)


def test_three_regular_candidates_close_the_Aether_column_classification_only(
    rebuilt: dict,
) -> None:
    regular = [item for item in rebuilt["candidate_records"] if item["first_blocker"] == BLOCKER]
    assert len(regular) == 3
    assert rebuilt["positive_unit_branch_constraint_variable_classification_count"] == 3
    assert rebuilt["zero_dimensional_Aether_constraint_diagonal_block_count"] == 3
    assert rebuilt["zero_Aether_second_order_off_diagonal_columns_count"] == 3
    assert rebuilt["augmented_Aether_unknown_nonelliptic_negative_control_count"] == 3
    for item in regular:
        certificate = item["fixed_free_data_principal_certificate"]
        variables = certificate["reduced_positive_unit_branch_constraint_variables"]
        jets = certificate["reviewed_spatial_jet_order_derivation"]
        positive = certificate["fixed_Aether_free_data_reference_positive_control"]
        assert variables["elliptic_solve_variable_count"] == 4
        assert variables["independent_Aether_secondary_constraint_variable_count"] == 0
        assert variables["Aether_role"] == "prescribed_free_data_for_the_four_secondary_constraints"
        assert jets["maximum_Aether_free_data_spatial_derivative_order_in_constraints"] == 1
        assert jets["Aether_second_order_principal_column"] == "zero"
        assert positive["metric_reference_symbol_spectrum"] == ["2", "2", "8/3", "4"]
        assert positive["status"] == "pass_reference_only"


def test_eleven_characteristic_candidates_remain_separate_and_not_reached(rebuilt: dict) -> None:
    forced = [
        item
        for item in rebuilt["candidate_records"]
        if item["first_blocker"] == CHARACTERISTIC_BLOCKER
    ]
    assert len(forced) == 11
    for item in forced:
        assert item["fixed_free_data_principal_certificate"] is None
        assert (
            item["gate_ledger"]["positive_unit_branch_constraint_variable_classification"]["status"]
            == "not_reached"
        )


def test_finite_tilt_Fredholm_nonlinear_and_rejection_claims_remain_closed(rebuilt: dict) -> None:
    assert rebuilt["finite_tilt_metric_York_principal_symbol_pass_count"] == 0
    assert rebuilt["fixed_free_data_full_principal_ellipticity_pass_count"] == 0
    assert rebuilt["lower_order_coefficient_bound_pass_count"] == 0
    assert rebuilt["weighted_Fredholm_isomorphism_pass_count"] == 0
    assert rebuilt["computable_full_inverse_norm_count"] == 0
    assert rebuilt["nonlinear_remainder_bound_pass_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    for item in rebuilt["candidate_records"]:
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False


def test_hash_provenance_reviewed_control_and_all_seals(rebuilt: dict) -> None:
    assert (
        rebuilt["reviewed_ADM_constraint_control_binding"]["content_sha256"]
        == "d8ed883d59f2fd5db01fb340104b077bf8f697278a5225e0975f993387ae7f49"
    )
    assert rebuilt["bounded_fixed_free_data_principal_gate_completed"] is True
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
            lambda config: config["budget"].update(maximum_augmented_symbol_columns=8),
            "budget is not exact",
        ),
        (
            lambda config: config["source_weighted_reference_artifact"].update(
                content_sha256="0" * 64
            ),
            "content hash mismatch",
        ),
        (
            lambda config: config["reviewed_adm_constraint_control"].update(
                content_sha256="0" * 64
            ),
            "reviewed ADM control changed",
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
        build_future_aether_fixed_free_data_principal_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_weighted_reference_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_fixed_free_data_principal_gate(config, ROOT)
