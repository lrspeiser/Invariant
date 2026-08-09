from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.scalar_tensor_pack import (
    SCHEMA_VERSION,
    compile_scalar_tensor_pack,
    generic_g2_variation_noether_control,
    generic_g3_variation_noether_control,
    generic_g4_phi_variation_noether_control,
    generic_g4_scalar_variation_control,
)

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "configs/operator_packs/horndeski_l2_l4_polynomial.json"


def test_horndeski_function_pack_derives_l4_completion_without_enumeration() -> None:
    result = compile_scalar_tensor_pack(json.loads(PACK.read_text()))
    assert result["status"] == "compiled_formal_adapters_unresolved", result
    assert result["errors"] == []
    assert result["derived_function_derivatives"]["g4_x"] == "a10 + 2*a20*x"
    assert result["derived_function_derivatives"]["g4_xx"] == "2*a20"
    assert result["derivative_override_residuals"] == {"g4_x": "0"}
    assert result["l4_differential_completion"]["independent_choice_forbidden"]
    assert result["mutation_space"]["declared_cardinality"] == 135
    assert not result["mutation_space"]["enumerated"]
    assert result["capability_status"]["typed_normalized_covariant_family"] == "pass"
    assert result["capability_status"]["generic_g2_variation_and_noether"] == "pass"
    assert result["capability_status"]["generic_g3_variation_and_noether"] == "pass"
    assert result["capability_status"]["generic_g4_phi_only_adapter"] == "available"
    assert result["capability_status"]["generic_g4_fixed_metric_scalar_variation"] == "pass"
    assert result["capability_status"]["generic_g4_flat_metric_noether"] == "pass"
    assert result["capability_status"]["generic_g4_curved_exact_witnesses"] == "pass"
    assert result["capability_status"]["generic_g4_curved_all_jet_theorem"] == "pass"
    assert result["capability_status"]["generic_adm_kinetic_primary_constraint"] == "pass"
    assert (
        result["capability_status"]["generic_g4_independent_backend_metric_variation"]
        == "unresolved"
    )
    assert result["capability_status"]["generic_covariant_variation"] == "unresolved"
    assert result["capability_status"]["generic_noether_identity"] == "pass"
    assert (
        result["capability_status"]["compiled_g4_phi_only_variation_and_noether"]
        == "not_applicable_x_dependent"
    )
    assert (
        result["capability_status"]["generic_adm_dirac"]
        == "pass_on_regular_lapse_hessian_patches"
    )
    assert (
        result["capability_status"]["generic_tensor_hamiltonian"]
        == "pass_on_F_T_and_G_T_positive_patches"
    )
    assert (
        result["capability_status"]["generic_tensor_principal_symbol"]
        == "pass_on_F_T_and_G_T_positive_patches"
    )
    assert result["capability_status"]["generic_flrw_scalar_reduction"] == (
        "pass_with_background_sign_proof_required"
    )
    assert result["capability_status"]["generic_flrw_scalar_hamiltonian"] == (
        "pass_on_Theta_nonzero_F_S_and_G_S_positive_patches"
    )
    assert result["capability_status"]["generic_flrw_scalar_principal_symbol"] == (
        "pass_on_Theta_nonzero_F_S_and_G_S_positive_patches"
    )
    assert result["capability_status"]["generic_hamiltonian"] == (
        "pass_on_flrw_healthy_patches_global_unresolved"
    )
    assert result["capability_status"]["generic_principal_symbol"] == (
        "pass_on_flrw_healthy_patches_inhomogeneous_unresolved"
    )
    assert result["capability_status"]["flrw_interval_background_certificate"] == (
        "available_requires_run_config"
    )
    assert result["capability_status"][
        "generic_kessence_effective_metric_and_hamiltonian"
    ] == "pass"
    assert result["capability_status"]["generic_weak_field_generalized_harmonic"] == (
        "partition_by_canonical_G3_and_G4_X_zero"
    )
    assert result["capability_status"]["generic_weak_field_modified_harmonic"] == (
        "conditional_requires_candidate_weak_coupling_cone_and_symmetrizer_bounds"
    )
    assert result["capability_status"][
        "generic_cubic_horndeski_bssn_hyperbolicity"
    ] == "conditional_requires_candidate_uniform_weak_field_and_cone_bounds"
    formulation = result["formulation_classification"]
    assert formulation["canonical_G3"] == "d10*x"
    assert formulation["G4_X"] == "a10 + 2*a20*x"
    partition = formulation["mutation_axis_partition"]
    assert partition["status"] == "exact_axis_partition"
    assert partition["generalized_harmonic_eligible"] == 3
    assert partition["modified_harmonic_required"] == 132
    assert partition["obstruction_class_counts"] == {
        "generalized_harmonic_kessence": 3,
        "modified_harmonic_G3_only": 6,
        "modified_harmonic_G4_X_only": 42,
        "modified_harmonic_G3_and_G4_X": 84,
    }
    assert partition["proof_subclass_counts"] == {
        "generalized_harmonic_kessence": 3,
        "cubic_G3_only": 6,
        "quartic_linear_X_G4_only_G2_linear_X": 4,
        "quartic_linear_X_G4_only_G2_nonlinear_X": 8,
        "quartic_nonlinear_X_G4_only": 30,
        "mixed_G3_linear_X_G4": 24,
        "mixed_G3_nonlinear_X_G4": 60,
    }
    assert len(partition["assignment_classifications"]) == 135
    assert sum(
        item["proof_route"] == "cubic_horndeski_bssn_or_ccz4_weak_field"
        for item in partition["assignment_classifications"]
    ) == 6
    assert sum(
        item["proof_route"]
        == "linear_X_quartic_full_symbol_requires_phi_coefficients_fixed_zero"
        for item in partition["assignment_classifications"]
    ) == 4
    assert sum(
        item["proof_route"]
        == "linear_X_quartic_plus_kessence_full_symbol_requires_phi_coefficients_fixed_zero"
        for item in partition["assignment_classifications"]
    ) == 8
    assert all(
        item["proof_route_fixed_coefficient_requirements"]
        == {"c11": "0", "c02": "0", "d01": "0", "a01": "0"}
        for item in partition["assignment_classifications"]
        if item["proof_subclass"].startswith("quartic_linear_X_G4_only")
    )
    assert partition["count_residual"] == 0
    assert result["compiled_kessence_kinetic"] != "0"
    assert result["compiled_kessence_gradient"] != "0"
    assert result["compiled_adm_regularity_factor"] != "0"
    assert result["compiled_g2_unitary_lapse_hessian_factor"] != "0"
    assert result["compiled_tensor_G_T"] != "0"
    assert result["compiled_tensor_F_T"] != "0"
    assert result["compiled_tensor_speed_squared"] != "0"
    assert result["compiled_scalar_Theta"] != "0"
    assert result["compiled_scalar_G_S"] != "0"
    assert result["compiled_scalar_F_S"] != "0"
    assert result["compiled_scalar_speed_squared"] != "0"
    assert result["compiled_flrw_background_variables"]["on_shell_required"]
    background = result["compiled_flrw_background_system"]
    assert background["evolution_unknowns"] == ["h_tau", "x_tau"]
    assert background["evolution_reconstruction_residual"] == (
        "Matrix([[0], [0]])"
    )
    assert background["evolution_determinant"] != "0"
    assert result["generic_horndeski_l2_l4_tensor_stability_control"][
        "legendre_residual"
    ] == "0"
    assert result["capability_status"]["observations"] == "sealed"


def test_generic_g2_variation_and_noether_identity_has_a_sign_negative_control() -> None:
    passed, evidence = generic_g2_variation_noether_control()
    assert passed, evidence
    assert evidence["local_jet_residuals"] == ["0", "0", "0", "0"]
    assert evidence["corrupted_sign_rejected"]
    assert any(
        residual != "0"
        for residual in evidence["corrupted_metric_pressure_sign_residuals"]
    )


def test_generic_g3_variation_cancels_third_derivatives_and_closes_noether() -> None:
    passed, evidence = generic_g3_variation_noether_control()
    assert passed, evidence
    assert evidence["third_derivative_cancellation_residual"] == "0"
    assert evidence["flat_arbitrary_third_jet_noether_residuals"] == [
        "0",
        "0",
        "0",
        "0",
    ]
    assert evidence["noether_residuals"] == ["0", "0", "0", "0"]
    assert evidence["omitted_braiding_stress_rejected"]
    assert evidence["omitted_ricci_commutator_rejected"]
    assert "ricci_pp" in evidence["field_euler_second_order"]


def test_generic_g4_phi_variation_closes_noether_and_rejects_omissions() -> None:
    passed, evidence = generic_g4_phi_variation_noether_control()
    assert passed, evidence
    assert evidence["noether_residuals"] == ["0", "0", "0", "0"]
    assert evidence["omitted_metric_completion_rejected"]
    assert evidence["wrong_scalar_sign_rejected"]
    assert evidence["constant_F_limit"]["reduces_to_einstein_hilbert"]


def test_phi_only_g4_pack_activates_the_exact_nonminimal_adapter() -> None:
    spec = {
        "schema_version": SCHEMA_VERSION,
        "normalization": {
            "u": "phi/Lambda_phi",
            "x": "-nabla_phi_squared/(2*Lambda_phi**4)",
            "Lambda_phi_positive": True,
        },
        "coefficients": ["m2", "f1", "f2"],
        "functions": {
            "g2": "x",
            "g3": "0",
            "g4": "m2/2+f1*u+f2*u**2",
        },
        "derivative_overrides": {"g4_x": "0"},
        "mutation_axes": [],
    }
    result = compile_scalar_tensor_pack(spec)
    assert result["errors"] == []
    assert result["derived_function_derivatives"]["g4_x"] == "0"
    assert (
        result["capability_status"]["compiled_g4_phi_only_variation_and_noether"]
        == "pass"
    )
    assert result["capability_status"]["generic_covariant_variation"] == "pass"
    assert result["capability_status"]["generic_noether_identity"] == "pass"


def test_generic_g4_scalar_current_cancels_every_flat_third_derivative() -> None:
    passed, evidence = generic_g4_scalar_variation_control()
    assert passed, evidence
    flat = evidence["flat_arbitrary_third_jet"]
    assert flat["independent_symmetric_third_derivatives"] == 20
    assert flat["surviving_third_derivatives"] == {}
    assert flat["second_order"]
    flat_noether = evidence["flat_arbitrary_metric_scalar_noether"]
    assert flat_noether["metric_symmetry_residuals"] == ["0"] * 6
    assert flat_noether["combined_noether_residuals"] == ["0"] * 4
    assert flat_noether["omitted_G4_XX_q_mu_q_nu_rejected"]
    curved = evidence["curved_linear_x_reduction"]
    assert curved["reduction_residual"] == "0"
    assert curved["fourth_derivative_coefficient"] == "0"
    assert curved["curvature_gradient_coefficient"] == "0"
    assert curved["wrong_completion_rejected"]


def test_linear_g4_reduces_to_the_existing_named_control_relation() -> None:
    spec = {
        "schema_version": SCHEMA_VERSION,
        "normalization": {
            "u": "phi/Lambda_phi",
            "x": "-nabla_phi_squared/(2*Lambda_phi**4)",
            "Lambda_phi_positive": True,
        },
        "coefficients": ["m2", "alpha"],
        "functions": {"g2": "x", "g3": "0", "g4": "m2/2 + alpha*x"},
        "derivative_overrides": {"g4_x": "alpha"},
        "mutation_axes": [],
    }
    result = compile_scalar_tensor_pack(spec)
    assert result["errors"] == []
    assert result["functions"]["g4"] == "(2*alpha*x + m2)/2"
    assert result["derived_function_derivatives"]["g4_x"] == "alpha"
    assert result["derived_function_derivatives"]["g4_xx"] == "0"
    assert result["compiled_adm_regularity_factor"] == "-(2*alpha*x - m2)/2"


def test_inconsistent_independent_g4_x_choice_is_rejected() -> None:
    spec = json.loads(PACK.read_text())
    spec["derivative_overrides"]["g4_x"] = "a10"
    result = compile_scalar_tensor_pack(spec)
    assert result["status"] == "reject"
    assert result["derivative_override_residuals"]["g4_x"] == "-2*a20*x"
    assert "inconsistent with its parent function" in " ".join(result["errors"])


def test_undeclared_function_symbol_is_rejected() -> None:
    spec = json.loads(PACK.read_text())
    spec["functions"]["g3"] = "secret*x"
    result = compile_scalar_tensor_pack(spec)
    assert result["status"] == "reject"
    assert "undeclared scalar-tensor symbol" in " ".join(result["errors"])
