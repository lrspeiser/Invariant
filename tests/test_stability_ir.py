from __future__ import annotations

import copy
from pathlib import Path

from sigma_theory_compiler.action_ir import (
    compile_action_file,
    compile_action_spec,
    load_action_grammar,
)
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.dirac_ir import compile_dirac_ir
from sigma_theory_compiler.formal_backend import load_field_contract
from sigma_theory_compiler.legendre_ir import compile_legendre_ir
from sigma_theory_compiler.stability_ir import compile_stability_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT_PATH = ROOT / "configs" / "covariant_field_contract.json"
GRAMMAR = load_action_grammar(GRAMMAR_PATH)
CONTRACT = load_field_contract(CONTRACT_PATH)


def _controls() -> dict[str, bool]:
    return {
        name: True
        for name in (
            "cadabra_adm_spatial_curvature_variation",
            "nonlinear_adm_hamiltonian_constraint_algebra",
            "canonical_metric_diffeomorphism_algebra",
            "canonical_metric_dewitt_kinetic_covariance",
            "spatial_curvature_density_diffeomorphism_covariance",
            "canonical_scalar",
            "canonical_scalar_noether_identity",
            "canonical_scalar_gravity_cross_constraint_identities",
            "three_spatial_dimensional_smeared_brackets",
            "proca_adm_dirac",
            "proca_divergence_identity",
            "proca_stress_noether_identity",
            "proca_reduced_smeared_constraint_algebra",
            "einstein_aether_generic_3plus1_legendre",
            "einstein_aether_generic_lapse_shift_constraint_seeds",
            "einstein_aether_spatial_diffeomorphism_algebra",
            "einstein_aether_generic_dh_covariance",
            "einstein_aether_generic_hh_deformation_kinematics",
            "einstein_aether_arbitrary_background_4d_noether",
            "regular_holonomic_multiplier_dirac_theorem",
            "unit_timelike_vector_dirac_chain",
            "einstein_hilbert_linearized_adm",
            "principal_symbol_controls",
            "curved_background_principal_controls",
            "einstein_aether_linearized_physical_energy",
            "einstein_aether_restricted_nonlinear_total_energy",
            "einstein_aether_reduced_five_mode_principal_domain",
            "einstein_aether_global_tilt_legendre_strata",
            "einstein_aether_covariant_arbitrary_background_hyperbolicity",
        )
    }


def _compile_action(action: dict) -> tuple[dict, dict]:
    controls = _controls()
    adm = compile_adm_ir(action, controls)
    legendre = compile_legendre_ir(action, adm)
    dirac = compile_dirac_ir(action, adm, legendre, controls)
    return dirac, compile_stability_ir(action, dirac, controls)


def _compile_file(name: str) -> tuple[dict, dict]:
    action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR_PATH, CONTRACT_PATH)
    return action, _compile_action(action)[1]


def _scalar_spec(coefficient: str, *, positive_c_x: bool) -> dict:
    domain = {"positive": ["M_Pl", "Lambda_phi", "m_phi"]}
    if positive_c_x:
        domain["positive"].append("c_X")
    return {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": ["g_mu_nu", "phi"],
        "matter_metric": "g_mu_nu",
        "terms": ["EH_R", "SCALAR_X", "SCALAR_MASS"],
        "coefficients": {"SCALAR_X": coefficient},
        "universal_constants": ["M_Pl", "Lambda_phi", "m_phi", "c_X"],
        "parameter_domain": domain,
        "static_dictionary_status": "derived",
    }


def test_healthy_eh_scalar_and_proca_controls_pass_stability_certificate() -> None:
    for name in (
        "einstein_hilbert_control.json",
        "canonical_scalar_control.json",
        "proca_control.json",
    ):
        action, result = _compile_file(name)
        assert result == compile_stability_ir(
            action,
            _compile_action(action)[0],
            _controls(),
        )
        assert result["status"] == "pass"
        assert result["condition_certificate"]["status"] == "pass"
        assert result["physical_hamiltonian"]["status"] == "pass"
        assert result["principal_symbol"]["status"] == "pass"
        assert result["input_action_sha256"] == action["content_sha256"]


def test_aether_domain_passes_principal_conditions_but_not_generic_energy() -> None:
    _, result = _compile_file("einstein_aether_control.json")
    assert result["status"] == "unresolved"
    assert result["condition_certificate"]["status"] == "pass"
    assert all(
        item["status"] == "pass" for item in result["condition_certificate"]["conditions"]
    )
    assert result["physical_hamiltonian"]["status"] == "unresolved"
    assert result["principal_symbol"]["status"] == "pass"
    assert result["derived_effective_parameters"]["c13_effective"] == "c1 + c3"
    assert "spin2_speed_squared" in result["principal_symbol"][
        "characteristic_speed_squared"
    ]


def test_wrong_sign_scalar_is_rejected_even_when_family_controls_pass() -> None:
    action = compile_action_spec(_scalar_spec("-1", positive_c_x=False), GRAMMAR, CONTRACT)
    assert action["valid"], action["errors"]
    _, result = _compile_action(action)
    assert result["status"] == "reject"
    conditions = {item["name"]: item for item in result["condition_certificate"]["conditions"]}
    assert conditions["positive_scalar_kinetic_and_gradient"]["status"] == "reject"
    assert result["physical_hamiltonian"]["status"] == "reject"
    assert result["principal_symbol"]["status"] == "reject"


def test_symbolic_scalar_requires_a_hash_bound_positive_domain() -> None:
    unspecified = compile_action_spec(
        _scalar_spec("c_X", positive_c_x=False), GRAMMAR, CONTRACT
    )
    _, unresolved = _compile_action(unspecified)
    assert unresolved["condition_certificate"]["status"] == "unresolved"
    assert unresolved["status"] == "unresolved"

    declared = compile_action_spec(_scalar_spec("c_X", positive_c_x=True), GRAMMAR, CONTRACT)
    _, passed = _compile_action(declared)
    assert passed["condition_certificate"]["status"] == "pass"
    assert passed["status"] == "pass"
    assert unspecified["content_sha256"] != declared["content_sha256"]


def test_stability_ir_fails_closed_on_hash_or_control_failure() -> None:
    action = compile_action_file(
        ROOT / "configs" / "actions" / "einstein_hilbert_control.json",
        GRAMMAR_PATH,
        CONTRACT_PATH,
    )
    dirac, _ = _compile_action(action)
    wrong = copy.deepcopy(dirac)
    wrong["input_action_sha256"] = "0" * 64
    rejected = compile_stability_ir(action, wrong, _controls())
    assert rejected["status"] == "reject"
    assert rejected["errors"] == ["Dirac IR belongs to a different action hash"]

    unresolved = compile_stability_ir(action, dirac, {})
    assert unresolved["status"] == "unresolved"
    assert unresolved["physical_hamiltonian"]["missing_or_failed_controls"] == [
        "einstein_hilbert_linearized_adm"
    ]
