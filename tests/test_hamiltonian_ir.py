from __future__ import annotations

import copy
from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.dirac_ir import compile_dirac_ir
from sigma_theory_compiler.hamiltonian_ir import compile_physical_hamiltonian_ir
from sigma_theory_compiler.legendre_ir import compile_legendre_ir
from sigma_theory_compiler.principal_ir import compile_physical_principal_ir
from sigma_theory_compiler.stability_ir import compile_stability_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


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
            "quartic_horndeski_covariant_adm_degeneracy",
            "quartic_horndeski_unitary_flrw_dirac_chain",
            "quartic_horndeski_unitary_distributed_dirac_closure",
            "quartic_horndeski_timelike_flat_principal_symbol",
            "quartic_horndeski_arbitrary_curvature_scalar_principal",
            "quartic_horndeski_coupled_formulation_hyperbolicity",
            "quartic_horndeski_full_local_principal_extraction",
            "quartic_horndeski_global_timelike_gradient_no_go",
            "quartic_horndeski_timelike_flat_physical_hamiltonian",
        )
    }


def _compile(name: str) -> tuple[dict, dict, dict, dict, dict]:
    controls = _controls()
    action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
    adm = compile_adm_ir(action, controls)
    legendre = compile_legendre_ir(action, adm)
    dirac = compile_dirac_ir(action, adm, legendre, controls)
    stability = compile_stability_ir(action, dirac, controls)
    principal = compile_physical_principal_ir(action, dirac, stability)
    hamiltonian = compile_physical_hamiltonian_ir(
        action, dirac, stability, principal
    )
    return action, dirac, stability, principal, hamiltonian


def test_eh_scalar_and_proca_physical_hamiltonians_are_exact_and_positive() -> None:
    expected_modes = {
        "einstein_hilbert_control.json": 2,
        "canonical_scalar_control.json": 3,
        "proca_control.json": 5,
    }
    for name, modes in expected_modes.items():
        action, dirac, stability, principal, result = _compile(name)
        assert result == compile_physical_hamiltonian_ir(
            action, dirac, stability, principal
        )
        assert result["status"] == "pass"
        assert result["physical_mode_count"] == modes
        assert result["constraint_surface_physical_dof"] == modes
        assert result["gauge_reduction_status"] == "pass"
        assert result["legendre_transform_residual"] == "0"
        assert result["positivity_certificate"]["status"] == "pass"
        assert result["input_principal_ir_sha256"] == principal["content_sha256"]


def test_scalar_mass_and_proca_second_class_energy_are_present() -> None:
    *_, scalar = _compile("canonical_scalar_control.json")
    assert "m_phi**2" in scalar["coordinate_hessian"]
    assert "potential Hessian" in scalar["construction"]["mass_or_constraint_reduction"]

    *_, proca = _compile("proca_control.json")
    assert "k**2/m_A**2" in proca["momentum_hessian"]
    assert "div p" in proca["construction"]["mass_or_constraint_reduction"]
    assert proca["construction"]["gauge_kinetic_amplitude"] == "1"
    assert proca["construction"]["mass_energy_amplitude"] == "1"


def test_aether_quadratic_energy_passes_but_generic_nonlinear_energy_stays_open() -> None:
    *_, result = _compile("einstein_aether_control.json")
    assert result["status"] == "unresolved"
    assert result["physical_mode_count"] == 5
    assert result["legendre_transform_residual"] == "0"
    assert result["positivity_certificate"]["status"] == "pass"
    assert result["positivity_certificate"]["source_reduced_energy_status"] == "unresolved"
    assert result["generic_nonlinear_total_energy"]["status"] == "unresolved"
    assert "twisting-Aether" in result["generic_nonlinear_total_energy"]["scope"]


def test_horndeski_patchwise_quadratic_energy_passes_while_global_energy_stays_open() -> None:
    *_, result = _compile("quartic_horndeski_control.json")
    assert result["status"] == "unresolved"
    assert result["physical_mode_count"] == 3
    assert result["constraint_surface_physical_dof"] == 3
    assert result["gauge_reduction_status"] == "pass"
    assert result["legendre_transform_residual"] == "0"
    assert (
        result["positivity_certificate"]["declared_background_patch_status"]
        == "pass"
    )
    assert result["generic_nonlinear_total_energy"]["status"] == "unresolved"
    assert "Horndeski energy" in result["generic_nonlinear_total_energy"]["scope"]


def test_hamiltonian_ir_rejects_wrong_sign_certificate_and_hash_mismatch() -> None:
    action, dirac, stability, principal, _ = _compile("canonical_scalar_control.json")
    wrong_sign = copy.deepcopy(stability)
    wrong_sign["condition_certificate"]["status"] = "reject"
    wrong_sign["physical_hamiltonian"]["status"] = "reject"
    rejected = compile_physical_hamiltonian_ir(
        action, dirac, wrong_sign, principal
    )
    assert rejected["status"] == "reject"
    assert rejected["positivity_certificate"]["status"] == "reject"

    broken = copy.deepcopy(principal)
    broken["input_stability_ir_sha256"] = "0" * 64
    hash_reject = compile_physical_hamiltonian_ir(
        action, dirac, stability, broken
    )
    assert hash_reject["status"] == "reject"
    assert hash_reject["errors"] == [
        "principal IR belongs to a different stability hash"
    ]


def test_hamiltonian_ir_stays_unresolved_without_physical_dirac_reduction() -> None:
    action, dirac, stability, principal, _ = _compile("einstein_hilbert_control.json")
    unresolved_dirac = copy.deepcopy(dirac)
    unresolved_dirac["status"] = "unresolved"
    unresolved_dirac["content_sha256"] = "unresolved-dirac"
    unresolved_stability = copy.deepcopy(stability)
    unresolved_stability["input_dirac_ir_sha256"] = "unresolved-dirac"
    unresolved_stability["content_sha256"] = "unresolved-stability"
    unresolved_principal = copy.deepcopy(principal)
    unresolved_principal["input_dirac_ir_sha256"] = "unresolved-dirac"
    unresolved_principal["input_stability_ir_sha256"] = "unresolved-stability"
    unresolved_principal["content_sha256"] = "unresolved-principal"
    result = compile_physical_hamiltonian_ir(
        action, unresolved_dirac, unresolved_stability, unresolved_principal
    )
    assert result["status"] == "unresolved"
    assert result["gauge_reduction_status"] == "unresolved"
