from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.g3_full_lapse_dirac_operator_audit import (
    _derive_full_delta,
    _sha,
    _validate_domain,
    _validate_target,
    build_g3_full_lapse_dirac_operator_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g3_full_lapse_dirac_operator_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g3-full-lapse-dirac-operator-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g3_full_lapse_dirac_operator_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "8f92e1e3e8ee81cb1ddf31e2f00979843e40ca8887e2b388295c506f623ef9f8"
    )


def test_unitary_adm_and_fixed_momentum_delta_are_exact(rebuilt: dict) -> None:
    derivation = rebuilt["candidate_records"][0]["full_lapse_operator_derivation"]
    assert derivation["cubic_integration_by_parts"]["reduced_ADM_density_over_sqrt_h"] == (
        "K*beta/(3*N**2)"
    )
    assert derivation["cubic_integration_by_parts"]["spatial_lapse_derivative_terms"] == "none"
    assert derivation["canonical_reduction"]["differentiation_contract"] == (
        "Delta_N=-d^2 H/dN^2 at fixed canonical momentum"
    )
    assert derivation["full_Delta_N"]["differential_order"] == 0
    assert derivation["full_Delta_N"]["operator_type"] == "real_multiplication_operator"
    assert derivation["exact_residuals"] == {
        "fixed_momentum_Delta": "0",
        "momentum_to_curvature_Delta": "0",
    }

    lapse, beta, momentum, curvature = sp.symbols("N beta q K", positive=True)
    symbols = {"N": lapse, "beta": beta, "q": momentum, "K": curvature}
    fixed = sp.sympify(derivation["full_Delta_N"]["fixed_momentum"], locals=symbols)
    velocity = sp.sympify(derivation["full_Delta_N"]["on_velocity_cell"], locals=symbols)
    assert sp.factor(
        fixed
        - (
            lapse**-3
            - 2 * beta * momentum * lapse**-4
            + sp.Rational(5, 2) * beta**2 * lapse**-7
        )
    ) == 0
    assert sp.factor(
        velocity
        - (
            lapse**-3
            + 2 * beta * curvature * lapse**-4
            + sp.Rational(3, 2) * beta**2 * lapse**-7
        )
    ) == 0


def test_predecessor_box_implies_unitary_trace_and_lapse_bounds(rebuilt: dict) -> None:
    coercivity = rebuilt["candidate_records"][0]["coercivity_certificate"]
    trace = coercivity["unitary_foliation_trace_bound"]
    assert trace["raw_abs_K_upper"] == "196513/3764768"
    assert Fraction(trace["raw_abs_K_upper"]) < Fraction(trace["chosen_abs_K_envelope"])
    assert trace["chosen_abs_K_envelope"] == "3/50"
    lower = coercivity["Delta_N_lower_bound"]
    assert lower["unitary_lapse_interval"] == ["50/51", "50/49"]
    assert lower["BSSN_lapse_not_identified_with_unitary_lapse"] is True
    assert lower["N_derivative_sufficient_gap"] == "31199/10625"
    assert lower["attained_bound_endpoint"] == "N=50/49,K=-3/50"


def test_full_delta_is_coercive_and_has_no_periodic_zero_mode(rebuilt: dict) -> None:
    result = rebuilt["candidate_records"][0]["coercivity_certificate"]
    lower = result["Delta_N_lower_bound"]
    assert lower["exact"] == "14690865266218547/15625000000000000"
    assert Fraction(lower["exact"]) > Fraction(94, 100)
    assert lower["strictly_positive"] is True
    assert lower["upper_at_N_50_over_51_K_3_over_50"] == (
        "16604362835033553/15625000000000000"
    )
    function_space = result["function_space_result"]
    assert function_space["operator"] == "M_Delta:L2(T3)->L2(T3)"
    assert function_space["kernel"] == "{0}"
    assert function_space["boundary_zero_modes"] == "excluded_by_strict_pointwise_lower_bound"
    assert function_space["status"] == "pass"


def test_lapse_dirac_gate_passes_but_af_energy_remains_blocked(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    gates = record["gate_ledger"]
    assert gates["full_candidate_Delta_N_derivation"]["status"] == "pass"
    assert gates["Delta_N_coercivity_and_zero_mode_exclusion"]["status"] == "pass"
    assert gates["distributed_Dirac_on_periodic_cell"]["status"] == "pass"
    assert gates["asymptotically_flat_extension"]["status"] == "blocked"
    assert "nondecaying canonical G2=X stress" in gates["asymptotically_flat_extension"][
        "reason"
    ]
    assert gates["global_hamiltonian_energy"]["status"] == "blocked"
    assert record["decision"] == "blocked"
    assert record["first_missing_premise"] == "asymptotically_flat_or_global_energy_domain"
    assert record["necessary_condition_rejection_found"] is False


def test_external_gates_stay_sealed(rebuilt: dict) -> None:
    assert rebuilt["full_lapse_dirac_pass_count"] == 1
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_weakened_domain_and_tampered_action_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    weakened = copy.deepcopy(config["operator_domain"])
    weakened["spatial_boundary_terms"] = "uncontrolled"
    with pytest.raises(ValueError, match="function-space or boundary domain changed"):
        _validate_domain(weakened)

    predecessor = json.loads((ROOT / config["predecessor"]["path"]).read_text(encoding="utf-8"))
    record = predecessor["candidate_records"][0]
    target = copy.deepcopy(config["target"])
    target["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)


def test_wrong_beta_changes_exact_delta_binding() -> None:
    quarter = _derive_full_delta(Fraction(1, 4))
    assert quarter["full_Delta_N"]["beta_specialization"] == "1/4"
    assert quarter["content_sha256"] != _derive_full_delta(Fraction(1, 100))[
        "content_sha256"
    ]
