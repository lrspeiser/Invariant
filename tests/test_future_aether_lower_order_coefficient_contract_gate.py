from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_finite_tilt_york_symbol_gate import (
    YORK_SHELL_BLOCKER,
)
from sigma_theory_compiler.future_aether_lower_order_coefficient_contract_gate import (
    BLOCKER,
    build_future_aether_lower_order_coefficient_contract_gate,
    exact_compact_profile_jet_control,
    exact_profile_regularity_negative_control,
)
from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_lower_order_coefficient_contract_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-lower-order-coefficient-contract-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-principal-inverse-fredholm-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_lower_order_coefficient_contract_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
        CHARACTERISTIC_BLOCKER: 11,
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0


def test_candidate_action_and_predecessor_binding_is_exact(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert item["source_principal_inverse_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_exact_compact_profile_positive_jet_control() -> None:
    result = exact_compact_profile_jet_control(Fraction(145475033, 5963776))
    assert result["sup_gradient_A_squared"] == "106051299057/1199078608"
    assert result["gradient_maximizer_radius_squared"] == "1/7"
    assert result["sup_component_Hessian_A_squared_upper"] == "1018325231/13312"
    assert result["sup_component_third_derivative_A_squared_upper"] == "9164927079/3328"
    assert result["sup_gradient_tilt_squared"] == "15427816230009943881/1787759056125952"
    assert result["weighted_local_bounds_available"] is True
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    assert result["content_sha256"] == _sha(body)


def test_profile_regularity_negative_and_invalid_amplitude_controls() -> None:
    control = exact_profile_regularity_negative_control()
    assert control["mutated_regularity"] == "C2_compact_support"
    assert control["required_regularity"] == "C3_compact_support"
    assert control["coefficient_contract_admissible"] is False
    with pytest.raises(ValueError, match="amplitude must be positive"):
        exact_compact_profile_jet_control(Fraction(0))


def test_sole_candidate_has_typed_interface_and_sharp_canonical_blocker(rebuilt: dict) -> None:
    active = [
        item
        for item in rebuilt["candidate_records"]
        if item["lower_order_coefficient_contract_certificate"] is not None
    ]
    assert len(active) == 1
    item = active[0]
    certificate = item["lower_order_coefficient_contract_certificate"]
    interface = certificate["declared_lower_order_coefficient_interface"]
    canonical = certificate["canonical_background_point_audit"]
    dag = certificate["distributed_constraint_DAG_audit"]
    assert item["candidate_id"].startswith("G3A-5e9f")
    assert item["first_blocker"] == BLOCKER
    assert certificate["compact_profile_C3_weighted_jet_bounds_proven"] is True
    assert interface["required_B_tensor_shape"] == [3, 4, 4]
    assert interface["required_C_tensor_shape"] == [4, 4]
    assert canonical["A_i"] == "registered_compact_profile"
    assert canonical["pi^ij"] == "not_registered_after_finite_tilt_Legendre_transform"
    assert canonical["p_A^i"] == "not_registered_after_finite_tilt_Legendre_transform"
    assert canonical["complete"] is False
    assert dag["linearized_B_order_one_coefficients"] == "not_registered"
    assert dag["linearized_C_order_zero_coefficients"] == "not_registered"
    assert dag["complete"] is False


def test_prior_characteristic_and_York_shell_blockers_are_preserved(rebuilt: dict) -> None:
    assert (
        sum(
            item["first_blocker"] == CHARACTERISTIC_BLOCKER for item in rebuilt["candidate_records"]
        )
        == 11
    )
    assert (
        sum(item["first_blocker"] == YORK_SHELL_BLOCKER for item in rebuilt["candidate_records"])
        == 2
    )
    for item in rebuilt["candidate_records"]:
        if item["first_blocker"] != BLOCKER:
            assert item["lower_order_coefficient_contract_certificate"] is None
            assert item["gate_ledger"]["compact_profile_weighted_C3_jet_bounds"]["status"] == (
                "not_reached"
            )


def test_full_coefficient_Fredholm_nonlinear_and_rejection_claims_remain_closed(
    rebuilt: dict,
) -> None:
    assert rebuilt["compact_profile_C3_weighted_jet_bound_pass_count"] == 1
    assert rebuilt["lower_order_coefficient_contract_declared_count"] == 1
    assert rebuilt["full_canonical_background_point_registered_count"] == 0
    assert rebuilt["distributed_lower_order_coefficient_registry_complete_count"] == 0
    assert rebuilt["weighted_relative_lower_order_bound_pass_count"] == 0
    assert rebuilt["weighted_Fredholm_isomorphism_pass_count"] == 0
    assert rebuilt["full_operator_inverse_norm_pass_count"] == 0
    assert rebuilt["nonlinear_remainder_bound_pass_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    for item in rebuilt["candidate_records"]:
        assert item["decision"] == "blocked"
        assert item["candidate_rejection_authorized"] is False
        assert item["constraint_satisfying_negative_total_energy_datum_proven"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_lower_order_coefficient_contract_gate_completed"] is True
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
            lambda config: config["coefficient_contract"].update(required_background_jet_order=2),
            "coefficient contract is not exact",
        ),
        (
            lambda config: config["budget"].update(maximum_seed_jet_order=2),
            "budget is not exact",
        ),
        (
            lambda config: config["source_principal_inverse_artifact"].update(
                content_sha256="0" * 64
            ),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_contract_budget_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_lower_order_coefficient_contract_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_principal_inverse_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_lower_order_coefficient_contract_gate(config, ROOT)
