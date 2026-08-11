from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_weak_field_ae_constraint_gate import (
    build_future_aether_weak_field_ae_constraint_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_weak_field_ae_constraint_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-weak-field-ae-constraint-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-pure-twist-ae-no-go-audit.json"
BLOCKER = (
    "finite_amplitude_candidate_bound_nonlinear_AE_coupled_constraint_solution_"
    "with_negative_completed_boundary_energy_beyond_positive_weak_field_quadratic_regime"
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_weak_field_ae_constraint_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {BLOCKER: 14}
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0


def test_source_candidate_action_and_record_bindings_are_preserved(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    assert len(expected) == 14
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert (
            item["action_density_equivalence_sha256"]
            == predecessor["action_density_equivalence_sha256"]
        )
        assert item["compilation_receipt_sha256"] == predecessor["compilation_receipt_sha256"]
        assert item["source_no_go_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_exact_compact_energy_identity_and_controls(rebuilt: dict) -> None:
    control = rebuilt["symbolic_weak_field_control"]
    density = control["quadratic_static_Aether_density"]
    assert density["integrated_formula"] == ("E_2=(1/2)*integral[c1*|grad a|^2+(c2+c3)*(div a)^2]")
    assert density["full_Hamiltonian_source"] == "S_H^(2)=rho_2+partial_i(Q_i^(2))"
    assert density["compact_monopole"] == "integral S_H^(2)=integral rho_2=E_2"
    assert density["c4_order"] == "quartic_in_epsilon"
    toroidal = control["positive_control_compact_toroidal_seed"]
    assert toroidal["divergence"] == "0"
    assert toroidal["integral_gradient_norm_squared"] == "524288*pi/2909907"
    assert toroidal["integral_crossed_gradient"] == "0"
    negative = control["negative_control_excluded_coupling"]
    assert negative["same_toroidal_seed_energy"] == "-262144*pi/2909907"
    assert (
        control["boundary_control_noncompact_affine_rotation"][
            "integration_by_parts_identity_authorized"
        ]
        is False
    )


def test_linearized_constraint_and_boundary_completion_is_exactly_scoped(
    rebuilt: dict,
) -> None:
    assert rebuilt["weak_field_linearized_constraint_completion_count"] == 14
    assert rebuilt["strictly_positive_compact_quadratic_energy_count"] == 14
    assert rebuilt["weak_field_negative_completed_energy_direction_count"] == 0
    assert rebuilt["finite_amplitude_nonlinear_constraint_completion_count"] == 0
    control = rebuilt["symbolic_weak_field_control"]
    assert control["linearized_Hamiltonian_completion"] == {
        "constraint": "4*M2*Delta(phi)+S_H^(2)=0",
        "green_solution": ("phi(x)=(1/(16*pi*M2))*integral S_H^(2)(y)/|x-y| d^3y"),
        "asymptotic_coefficient": "phi=E_2/(16*pi*M2*r)+O(r^-2)",
        "ADM_boundary_energy_coefficient": "M_ADM^(2)=E_2",
    }
    momentum = control["linearized_momentum_completion"]
    assert momentum["symbol_determinant"] == "4*(k1**2 + k2**2 + k3**2)**3/3"
    assert (
        control["completed_Aether_boundary_energy"]["compact_seed_boundary_term"]
        == "zero_through_order_epsilon_squared"
    )
    assert "not a full nonlinear" in control["proved_scope"]


def test_all_candidate_coefficients_are_coercive_and_remain_blocked(rebuilt: dict) -> None:
    assert rebuilt["c2_plus_c3_counts"] == {
        "0": 2,
        "1/16": 3,
        "1/32": 1,
        "1/4": 1,
        "1/8": 3,
        "3/16": 2,
        "3/32": 1,
        "5/32": 1,
    }
    for item in rebuilt["candidate_records"]:
        certificate = item["weak_field_AE_constraint_certificate"]
        assert Fraction(certificate["c1"]) == Fraction(1, 32)
        assert Fraction(certificate["c2_plus_c3"]) >= 0
        assert Fraction(certificate["coercive_gradient_coefficient"]) == Fraction(1, 64)
        assert certificate["strictly_positive_for_every_nonzero_compact_seed"] is True
        assert certificate["linearized_Hamiltonian_constraint_completed"] is True
        assert certificate["linearized_momentum_constraint_completed"] is True
        assert certificate["linearized_completed_boundary_energy_negative"] is False
        assert certificate["finite_amplitude_nonlinear_constraint_solution_proven"] is False
        assert item["decision"] == "blocked"
        assert item["first_blocker"] == BLOCKER
        assert item["formal_pass"] is False
        assert item["candidate_rejection_authorized"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_weak_field_AE_constraint_gate_completed"] is True
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == _config()["data_eligibility"]
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)
        record_provenance = item["provenance"]
        record_body = {
            key: value for key, value in record_provenance.items() if key != "binding_sha256"
        }
        assert record_provenance["binding_sha256"] == _sha(record_body)
        assert item["observational_data_opened"] is False
        assert item["solar_bundle_generated"] is False


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
        (
            lambda config: config.update(observational_authorization=True),
            "opened observations",
        ),
        (
            lambda config: config.update(external_paid_llm_calls=True),
            "enabled paid LLM calls",
        ),
        (
            lambda config: config["budget"].update(maximum_symbolic_polynomial_terms=499),
            "budget is not exact",
        ),
        (
            lambda config: config["source_no_go_artifact"].update(content_sha256="0" * 64),
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
        build_future_aether_weak_field_ae_constraint_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_no_go_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_weak_field_ae_constraint_gate(config, ROOT)
