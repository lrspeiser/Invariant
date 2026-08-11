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
from sigma_theory_compiler.future_aether_weighted_reference_operator_gate import (
    BLOCKER,
    build_future_aether_weighted_reference_operator_gate,
    exact_reference_symbol,
    exact_ungauged_hamiltonian_negative_control,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_weighted_reference_operator_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-weighted-reference-operator-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-weighted-ift-contract-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_weighted_reference_operator_gate(_config(), ROOT)


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
        assert item["source_weighted_ift_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_exact_reference_symbol_positive_controls_are_direction_independent() -> None:
    axial = exact_reference_symbol((Fraction(1), Fraction(0), Fraction(0)))
    diagonal = exact_reference_symbol((Fraction(2, 3), Fraction(2, 3), Fraction(1, 3)))
    for result in (axial, diagonal):
        assert result["York_symbol_eigenvalues"] == ["2", "2", "8/3"]
        assert result["combined_symbol_eigenvalues"] == ["2", "2", "8/3", "4"]
        assert result["combined_symbol_determinant"] == "128/3"
        assert result["combined_principal_ellipticity_margin"] == "2"
        assert result["principal_symbol_invertible"] is True
        body = {key: value for key, value in result.items() if key != "content_sha256"}
        assert result["content_sha256"] == _sha(body)


def test_nonunit_symbol_input_fails_closed() -> None:
    with pytest.raises(ValueError, match="exact unit norm"):
        exact_reference_symbol((Fraction(1), Fraction(1), Fraction(0)))


def test_ungauged_pure_diffeomorphism_is_an_exact_negative_control() -> None:
    control = exact_ungauged_hamiltonian_negative_control()
    assert control["metric_perturbation_h_ij"] == [
        ["0", "1", "0"],
        ["1", "0", "0"],
        ["0", "0", "0"],
    ]
    assert control["linearized_scalar_curvature_symbol"] == "0"
    assert control["nonzero_pure_gauge_symbol_kernel_witness"] is True
    assert control["ungauged_scalar_symbol_injective"] is False


def test_three_regular_candidates_close_only_the_metric_reference_layer(rebuilt: dict) -> None:
    regular = [item for item in rebuilt["candidate_records"] if item["first_blocker"] == BLOCKER]
    assert len(regular) == 3
    assert rebuilt["declared_metric_weighted_contract_count"] == 3
    assert rebuilt["metric_reference_principal_ellipticity_pass_count"] == 3
    assert rebuilt["metric_reference_trivial_kernel_pass_count"] == 3
    assert rebuilt["registered_compact_source_right_inverse_count"] == 3
    assert rebuilt["ungauged_pure_diffeomorphism_negative_control_count"] == 3
    for item in regular:
        certificate = item["weighted_reference_operator_certificate"]
        contract = certificate["declared_metric_weighted_contract"]
        assert contract["weight_delta"] == "-1/2"
        assert contract["domain"].startswith("H^2_-1/2")
        assert contract["codomain"].startswith("L^2_-5/2")
        assert "integral_R3" in contract["norm_convention"]
        assert certificate["principal_symbol_certificates"]["ellipticity_margin"] == "2"
        assert certificate["decaying_kernel_certificate"]["reference_metric_kernel_trivial"]
        assert (
            Fraction(certificate["carried_candidate_controls"]["Aether_Legendre_inverse_bound"]) > 0
        )


def test_eleven_characteristic_candidates_remain_not_reached(rebuilt: dict) -> None:
    forced = [
        item
        for item in rebuilt["candidate_records"]
        if item["first_blocker"] == CHARACTERISTIC_BLOCKER
    ]
    assert len(forced) == 11
    for item in forced:
        assert item["weighted_reference_operator_certificate"] is None
        assert (
            item["gate_ledger"]["declared_metric_weighted_domain_codomain_and_gauge"]["status"]
            == "not_reached"
        )


def test_no_full_coupled_fredholm_norm_remainder_or_rejection_overclaim(rebuilt: dict) -> None:
    assert rebuilt["candidate_Aether_constraint_principal_block_pass_count"] == 0
    assert rebuilt["full_coupled_Fredholm_operator_defined_count"] == 0
    assert rebuilt["full_weighted_operator_isomorphism_pass_count"] == 0
    assert rebuilt["computable_full_inverse_norm_count"] == 0
    assert rebuilt["nonlinear_remainder_bound_pass_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    for item in rebuilt["candidate_records"]:
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_weighted_reference_operator_gate_completed"] is True
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
            lambda config: config["weighted_contract"].update(weight_delta="-1"),
            "contract is not exact",
        ),
        (
            lambda config: config["source_weighted_ift_artifact"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_contract_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_weighted_reference_operator_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_weighted_ift_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_weighted_reference_operator_gate(config, ROOT)
