from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g3_asymptotically_flat_transition_audit import (
    _sha,
    _validate_domain,
    _validate_target,
    build_g3_asymptotically_flat_transition_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g3_asymptotically_flat_transition_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g3-asymptotically-flat-transition-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g3_asymptotically_flat_transition_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "71705e817b79942c5574fb39d3fd4482a650374bd62fa86895517049f317a289"
    )


def test_radial_profile_connects_center_to_integrable_AF_tail(rebuilt: dict) -> None:
    profile = rebuilt["candidate_records"][0]["radial_profile_certificate"]
    assert profile["endpoint_values"] == {
        "X_at_r_zero": "1/2",
        "X_strictly_positive_at_finite_r": True,
        "X_limit_at_infinity": "0",
    }
    assert profile["interior_connection"]["matches_certified_center_at_r_zero"] is True
    hessian = profile["mixed_hessian_norm"]
    assert hessian["global_maximum_location"] == "r=L"
    assert hessian["global_maximum"] == "1/(sqrt(2)*L)"
    assert hessian["stationary_residual_at_r_equals_L"] == "0"
    assert hessian["second_derivative_squared_at_r_equals_L"] == "-6/L**4"
    assert hessian["below_predecessor_component_bound_1_over_100"] is True
    assert profile["falloff"] == {
        "v": "L^2/r^2+O(r^-6)",
        "X": "L^4/(2*r^4)+O(r^-8)",
        "d_v_d_r": "-2*L^2/r^3+O(r^-7)",
        "canonical_G2_energy_tail_integrable": True,
    }


def test_principal_and_common_cone_remain_uniform_through_X_zero_limit(
    rebuilt: dict,
) -> None:
    principal = rebuilt["candidate_records"][0]["principal_common_cone_certificate"]
    assert principal["uniform_bounds_X_in_0_to_half"] == {
        "P00_upper": "-1",
        "spatial_eigenvalue_lower": "39999/40000",
        "time_space_norm_upper": "1/(5000*sqrt(2))",
        "time_space_norm_upper_squared": "1/50000000",
        "characteristic_discriminant_lower": "39999/40000",
        "BSSN_sigma": "1",
        "slicing_cone_polynomial_upper": "-2499/2500",
    }
    assert principal["direction_sphere_method"].endswith("no_sampling")
    assert principal["status"] == "pass_on_complete_radial_reference_profile_including_X_limit_zero"
    assert principal["scope"].endswith("not an Einstein-constraint solution")


def test_annulus_sequence_proves_no_bounded_AF_lapse_inverse(rebuilt: dict) -> None:
    obstruction = rebuilt["candidate_records"][0]["lapse_crossing_obstruction"]
    assert obstruction["full_multiplier"] == (
        "Delta_N(r)=v(r)^3+(3/2)*beta^2*v(r)^7"
    )
    assert obstruction["pointwise_properties"] == {
        "positive_at_every_finite_r": True,
        "pointwise_kernel": "none",
        "limit_at_infinity": "0",
        "asymptotic": "L^6/r^6+O(r^-14)",
    }
    annulus = obstruction["annulus_approximate_zero_modes"]
    assert annulus["bound_limit"] == "0"
    assert annulus["conclusion"] == (
        "zero_lies_in_approximate_spectrum_and_inverse_is_unbounded"
    )
    assert obstruction["Dirac_operator_status"] == (
        "blocked_not_boundedly_invertible_on_L2_R3"
    )
    assert obstruction["exact_obstruction"] == (
        "uniform_timelike_clock_margin_is_lost_as_X_tends_to_zero"
    )


def test_AF_bridge_is_blocked_without_false_rejection_or_constraint_claim(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    gates = record["gate_ledger"]
    assert gates["explicit_AF_decaying_gradient_profile"]["status"] == "pass"
    assert gates["uniform_principal_and_common_cone_on_reference_profile"]["status"] == "pass"
    assert gates["uniform_lapse_Dirac_invertibility"]["status"] == "blocked"
    assert gates["Einstein_constraint_solution"]["status"] == "blocked"
    assert gates["global_hamiltonian_energy"]["status"] == "blocked"
    assert record["decision"] == "blocked"
    assert record["first_missing_premise"] == (
        "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain"
    )
    assert record["negative_energy_counterexample_found"] is False


def test_external_gates_remain_sealed(rebuilt: dict) -> None:
    assert rebuilt["AF_principal_common_cone_profile_pass_count"] == 1
    assert rebuilt["AF_lapse_Dirac_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_domain_and_action_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config["transition_domain"])
    changed["function_space"]["lapse_multiplier"] = "unbound"
    with pytest.raises(ValueError, match="transition domain changed"):
        _validate_domain(changed)

    predecessor = json.loads((ROOT / config["predecessor"]["path"]).read_text(encoding="utf-8"))
    record = predecessor["candidate_records"][0]
    target = copy.deepcopy(config["target"])
    target["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)
