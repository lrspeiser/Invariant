from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.g4_source_class_scalar_uniqueness_audit import (
    _sha,
    _validate_source_class,
    build_g4_source_class_scalar_uniqueness_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g4_source_class_scalar_uniqueness_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g4-source-class-scalar-uniqueness-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g4_source_class_scalar_uniqueness_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "53cc6473950764ad222299420db70d71aa4236673442415aabf3fed5a146e6a7"
    )


def test_candidate_coupling_has_global_one_over_50_lipschitz_bound(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0][
        "global_coupling_lipschitz_certificate"
    ]
    y = sp.Symbol("y")
    gap = sp.sympify(certificate["one_over_50_minus_abs_derivative"])
    assert sp.factor(gap - y * (392 * y + 19375) / (100 * (14 * y + 625) ** 2)) == 0
    assert certificate["global_result"] == (
        "abs(d_alpha/d_chi)<=1/50_for_all_real_phi"
    )
    assert certificate["integrated_result"] == (
        "abs(alpha(chi))<=abs(chi)/50_with_alpha(0)=0"
    )
    assert certificate["role"] == (
        "candidate_specific_global_nonlinear_bound_not_only_beta0"
    )


def test_exact_interval_hardy_margin_is_strictly_coercive(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0][
        "source_class_coercivity_certificate"
    ]
    bound = certificate["dimensionless_relative_form_bound"]
    eta = Fraction(bound["eta_strict_upper"])
    margin = Fraction(bound["one_minus_eta_lower"])
    assert Fraction(bound["geometry_ratio"]) == Fraction(1_020_100, 970_299)
    assert eta == Fraction(81_608, 77_182_875)
    assert margin == Fraction(77_101_267, 77_182_875)
    assert eta + margin == 1
    assert 0 < eta < 1
    assert bound["eta_below_one"] is True
    profile = certificate["resolved_profile_Birman_Schwinger_route"]
    assert profile["operator"] == "B=sqrt(W)*L0^(-1)*sqrt(W)"
    assert profile["coercivity_rule"] == "kappa<1 implies Q>=(1-kappa)*Q0"
    assert profile["status"] == "exact_conditional_profile_criterion"
    assert "without a pointwise rho_trace_max" in profile["advantage"]


def test_theorem_is_nonlinear_static_and_shape_independent_within_class(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0][
        "source_class_coercivity_certificate"
    ]
    result = certificate["nonlinear_static_uniqueness"]
    assert result["result"] == (
        "chi=0_is_the_only_static_D1,2_solution_in_the_entire_source_class"
    )
    assert result["uses_linearization_only"] is False
    assert result["allows_arbitrary_source_shape_inside_B_R"] is True
    assert "if_the_declared_intervals_hold" in result[
        "allows_self_consistent_metric_and_trace"
    ]
    assert certificate["status"] == (
        "pass_source_class_nonlinear_static_uniqueness_and_linear_scalar_stability"
    )


def test_linear_scalar_zero_and_tachyonic_modes_are_excluded_with_scoped_claim(
    rebuilt: dict,
) -> None:
    stability = rebuilt["candidate_records"][0]["source_class_coercivity_certificate"][
        "linear_scalar_stability"
    ]
    assert stability["negative_eigenvalue"] == "excluded"
    assert stability["D1,2_zero_mode"] == "excluded"
    assert stability["tachyonic_scalarization_mode"] == "excluded_on_the_source_class"
    assert stability["metric_and_material_perturbation_decoupling"] == (
        "pass_because_alpha_0=0"
    )
    assert "not nonlinear dynamical stability" in stability["scope"]


def test_mass_and_radius_alone_fail_by_exact_concentration_control(rebuilt: dict) -> None:
    control = rebuilt["candidate_records"][0]["mass_radius_only_negative_control"]
    assert control["center_integral_tau_over_distance"] == "3*M/(2*epsilon)"
    assert control["epsilon_limit"] == "infinity_as_epsilon->0"
    assert "mass and outer radius alone cannot bound" in control["lesson"]
    assert control["candidate_rejection"] is False


def test_minimal_future_source_facts_are_explicit_and_not_claimed(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    contract = record["minimal_real_source_instantiation_contract"]
    fact_ids = {item["id"] for item in contract["required_registered_facts"]}
    assert fact_ids == {
        "source_support_radius_upper",
        "total_mass_and_compactness",
        "trace_density_or_concentration_upper",
        "pressure_trace_sign",
        "static_geometry_intervals",
        "scalar_boundary_and_topology",
    }
    assert "total mass plus visible radius without a concentration bound" in contract[
        "facts_not_sufficient_by_themselves"
    ]
    assert contract["current_registration_status"] == (
        "missing_no_real_source_facts_opened"
    )
    assert record["physics_side_real_source_blocker"] == {
        "theorem_side": "closed_for_every_source_satisfying_the_explicit_class",
        "real_Sun_instantiation": "blocked_until_registered_facts_instantiate_the_class",
    }
    assert record["real_solar_bundle_admissible"] is False


def test_decisions_seals_and_negative_inference_count_are_honest(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["gate_status_counts"] == {"pass": 4, "reject": 1, "blocked": 1}
    assert rebuilt["source_class_theorem_pass_count"] == 1
    assert rebuilt["real_source_instantiation_pass_count"] == 0
    assert record["decision"] == "blocked"
    assert record["source_class_theorem_decision"] == "pass"
    assert record["gate_ledger"]["mass_radius_sufficiency"]["status"] == "reject"
    assert record["candidate_rejection_found"] is False
    assert record["first_missing_premise"] == "registered_real_source_interval_certificate"
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_source_class_and_action_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    weakened = copy.deepcopy(config["source_class"])
    weakened["matter"]["dimensionless_trace_radius_bound"] = "unbounded"
    with pytest.raises(ValueError, match="source-class contract changed"):
        _validate_source_class(weakened)

    tampered = copy.deepcopy(config)
    tampered["target"]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predecessor target"):
        build_g4_source_class_scalar_uniqueness_audit(tampered, ROOT)
