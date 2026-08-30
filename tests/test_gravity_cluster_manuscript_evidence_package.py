from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_manuscript_evidence_package as package

ROOT = Path(__file__).resolve().parents[1]


def test_development_evidence_is_packaged_without_paper_overclaim() -> None:
    receipt = package.build_receipt(ROOT)
    assert receipt["decision"] == "DEVELOPMENT_MANUSCRIPT_EVIDENCE_PACKAGED_NOT_PAPER_READY"
    assert set(receipt["completed_goal_evidence"]) == {
        "CP12.2",
        "CP12.4",
        "CP12.5",
        "CP12.7",
        "CP12.8",
        "CP12.9",
    }
    assert set(receipt["blocked_goal_evidence"]) == {
        "CP12.1",
        "CP12.3",
        "CP12.6",
        "CP12.10",
        "CP12.11",
        "CP12.12",
    }
    assert receipt["claims"]["development_evidence"] is True
    assert receipt["claims"]["independent_replication"] is False
    assert receipt["claims"]["bounded_paper_ready"] is False


def test_all_candidate_rows_and_absolute_relative_summaries_are_retained() -> None:
    receipt = package.build_receipt(ROOT)
    assert receipt["counts"]["per_row_candidate_predictions"] == 233
    rows = receipt["per_row_candidate_predictions"]
    assert len({row["row_id"] for row in rows}) == 233
    assert {row["split"] for row in rows} == {
        "development_train",
        "development_holdout",
        "confirmation",
    }
    assert all(
        {"observed", "predicted", "error", "log_residual", "standardized_square"} <= set(row)
        for row in rows
    )
    assert set(receipt["split_summaries"]) == {
        "development_train",
        "development_holdout",
        "confirmation",
    }


def test_access_ledger_keeps_same_release_and_independent_access_separate() -> None:
    receipt = package.build_receipt(ROOT)
    access = receipt["access_ledger"]
    assert access["confirmation_response_files_opened_after_freeze"] == 8
    assert access["same_release_confirmation_rows"] == 77
    assert access["direct_lensing_likelihood_evaluations"] == 0
    assert access["inferred_total_mass_rows"] == 0
    assert access["independent_target_rows_opened"] == 0
    assert access["independent_observational_authorization"] is False


def test_negative_uncertainty_prior_art_and_claim_boundaries_are_all_present() -> None:
    receipt = package.build_receipt(ROOT)
    assert receipt["comparators_and_ablations"]["ablations"]
    assert receipt["negative_and_numerical_controls"]["synthetic_recovery"]
    assert receipt["negative_and_numerical_controls"]["false_selection"]
    assert receipt["uncertainty_and_alternative_cause_boundary"]["source_covariance_blockers"]
    calibration = receipt["quotient_sampler_calibration_and_newtonian_boundary"]
    assert calibration["v1_passed"] is False
    assert calibration["v2_passed"] is False
    assert calibration["v3_synthetic_sbc_passed"] is True
    assert calibration["newtonian_control_unlock"] is True
    assert calibration["candidate_production_unlock"] is False
    assert calibration["newtonian_external_approval_present"] is False
    covariance = receipt["development_pressure_covariance_boundary"]
    assert covariance["scoring_decision"] == "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS"
    assert covariance["reconstructed_matrices"] == 8
    assert covariance["scored_pressure_rows"] == 54
    assert covariance["CP5_2_through_CP5_6_complete"] is False
    assert receipt["prior_art_boundary"]["closest_behavioral_neighbor"]["source_id"] == (
        "PENNER_MODIFIED_GRAS_AQUAL_2026"
    )
    assert set(receipt["claim_tracks"]) == {
        "bounded_empirical_publication",
        "physical_mechanism",
        "universal_theory",
    }


def test_new_cross_scale_group_and_strata_evidence_keeps_claim_ceilings() -> None:
    receipt = package.build_receipt(ROOT)
    ben = receipt["shared_ben_synthetic_and_real_boundary"]
    assert ben["synthetic_raw_candidates"] == 240
    assert ben["synthetic_equivalence_classes"] == 60
    assert ben["synthetic_grammar_mechanics_validated"] is True
    assert ben["synthetic_recovery_is_scientific_evidence"] is False
    assert ben["local_sparc_confirmation_sealed_for_descendant"] is False
    assert ben["v2_blocked_before_payload_load"] is True
    assert ben["xcop_predictor_output_mapping_ready"] is False
    assert ben["v2_payload_loader_present"] is False
    assert ben["v2_real_scoring_executed"] is False
    ben_v4 = receipt["shared_ben_development_executor_v4_boundary"]
    assert ben_v4["canonical_full_classes"] == 60
    assert ben_v4["registered_ablations"] == 180
    assert ben_v4["unique_asts_across_full_and_ablations"] == 78
    assert ben_v4["production_executed"] is False
    assert ben_v4["target_files_opened"] == 0
    assert ben_v4["target_rows_read"] == 0
    assert ben_v4["scores_computed"] == 0
    assert ben_v4["comparison_operator"] == "binary64_numerical_indifference_band"
    assert ben_v4["reference_runtime_is_fully_frozen"] is False
    assert ben_v4["indifference_band_removes_all_runtime_variation"] is False
    assert ben_v4["terminal_success_marker_required"] is True
    assert ben_v4["publication_ready"] is False
    group = receipt["group_scale_source_boundary"]
    assert group["candidate_lanes"] == 3
    assert group["ready_lanes"] == 0
    assert group["CP10_1_complete"] is False
    assert group["CP10_2_complete"] is False
    assert group["scientific_result_emitted"] is False
    assert group["v3_candidate_lanes"] == 11
    assert group["v3_ready_lanes"] == 0
    assert group["v3_partial_lanes"] == 4
    assert group["v3_blocked_lanes"] == 7
    assert group["v3_preferred_lane"] == "XCLASS_LOWZ_155"
    assert group["v3_backup_lane"] == "EFEDS_542_RAW_REDUCTION"
    assert group["v3_accept_author_sample"] == 239
    assert group["v3_accept_current_table_rows"] == 240
    assert group["v3_accept_count_resolved"] is False
    assert group["v3_scientific_rows_opened"] == 0
    strata = receipt["cluster_strata_boundary"]
    assert strata["development_clusters"] == 8
    assert strata["CP5_11_predictor_strata_frozen"] is True
    assert strata["candidate_absolute_gate_passed"] is False
    assert strata["candidate_cluster_wins"] == 4
    assert strata["minimum_cluster_wins"] == 5
    assert strata["candidate_object_win_gate_passed"] is False
    assert strata["frozen_stratum_explains_covariance_flips"] is False
    assert strata["CP5_13_complete"] is False
    assert strata["causal_variable_identified"] is False
    assert strata["scientific_claim_allowed"] is False
    shape_missing = receipt["predictor_shape_and_missing_variable_boundary"]
    assert shape_missing["shape_production_authorized"] is False
    assert shape_missing["shape_real_scoring_executed"] is False
    assert shape_missing["shape_absolute_prediction_established"] is False
    assert shape_missing["defined_proxy_contracts"] == 4
    assert shape_missing["continuous_measurement_ready_rows"] == 0
    assert shape_missing["source_blocked_applicable_rows"] == 16
    acquisition = receipt["group_and_act_acquisition_boundary"]
    assert acquisition["group_alias_rows_opened"] == 0
    assert acquisition["group_scientific_payload_rows_opened"] == 0
    assert acquisition["group_ready_science_lanes"] == 0
    assert acquisition["act_catalog_rows_opened"] == 0
    assert acquisition["act_population_gate_evaluated"] is False
    assert acquisition["act_executor_authorized"] is False
    assert acquisition["act_executor_execution_started"] is False
    assert acquisition["act_executor_network_calls"] == 0
    assert acquisition["act_executor_catalog_rows_opened"] == 0
    assert acquisition["act_executor_overlap_count_computed"] is False
    assert acquisition["act_executor_xcop_exclusions_computed"] is False
    assert acquisition["act_executor_minimum_192_rule_evaluated"] is False
    assert acquisition["xclass_executor_authorized"] is False
    assert acquisition["xclass_executor_get_attempts"] == 0
    assert acquisition["xclass_executor_network_bytes"] == 0
    assert acquisition["xclass_executor_identity_rows"] == 0
    assert acquisition["xclass_executor_scientific_values"] == 0
    assert acquisition["xclass_executor_obsid_mapping_available"] is False
    assert acquisition["xclass_executor_xcop_overlap_known"] is False
    assert acquisition["xclass_executor_five_object_pilot_unlocked"] is False
    theory = receipt["matter_lensing_theory_boundary"]
    assert theory["template_level_gates_passed"] == 1
    assert theory["health_gates_total"] == 10
    assert theory["health_gates_blocked"] == 9
    assert theory["healthy_action_completed"] is False
    assert theory["symbolic_checks_passed"] == 20
    assert theory["independent_numeric_checks_passed"] == 6
    assert theory["full_H2_passed"] is False
    assert theory["H3_scalar_external_metric"].startswith("PARTIAL_MACHINE_DERIVED")
    assert theory["H4_constant_coefficient"].startswith("PARTIAL_MACHINE_DERIVED")
    assert theory["full_H3_passed"] is False
    assert theory["full_H4_passed"] is False
    assert theory["designed_u_above_one_third_failure_preserved"] is True
    assert theory["u_above_one_third_gate_contribution"] == ("the X_chi contribution is negative")
    assert theory["metric_constraints_derived"] is False
    assert theory["on_shell_backgrounds_established"] is False
    assert theory["global_strong_hyperbolicity_established"] is False
    assert theory["conditional_timelike_mixing_no_go"] == (
        "PASS_MACHINE_DERIVED_UNDER_FROZEN_HYPOTHESES"
    )
    assert theory["bounded_domain_nonnegative_examples_exist"] is True
    assert theory["full_determinant_no_go"] is False
    assert theory["unconditional_action_no_go_established"] is False
    assert theory["kinetic_gate_observational_files_opened"] == 0
    assert theory["kinetic_gate_observational_support"] is False
    assert theory["restricted_static_source_bound_established"] is True
    assert theory["physical_source_law_established"] is False
    assert theory["physical_on_shell_solution_established"] is False
    assert theory["universal_conformal_source_identity_established"] is True
    assert theory["physical_source_profile_established"] is False
    assert theory["metric_backreaction_established"] is False
    assert theory["solar_necessary_conditions_established"] is True
    assert theory["solar_gate_passed"] is False
    assert theory["gw_gate_passed"] is False
    assert theory["restricted_flat_flrw_equations_established"] is True
    assert theory["flrw_gate_limit_obstruction_derived"] is True
    assert theory["healthy_late_time_history_exists"] is False
    assert theory["perturbation_stability_established"] is False
    assert theory["cosmological_fit_performed"] is False
    assert theory["covariant_scalar_stress_and_exchange_established"] is True
    assert theory["formal_same_action_field_equation_contract_established"] is True
    assert theory["einstein_hilbert_curvature_variation_machine_verified"] is False
    assert theory["covariant_full_H2"] is False
    assert theory["covariant_ADM_constraints_derived"] is False
    assert theory["covariant_metric_backreaction_solved"] is False
    assert theory["CP11_3_complete"] is True
    assert theory["energy_momentum_exchange_and_constraint_propagation_established"] is True
    assert theory["hamiltonian_constraint_derived"] is True
    assert theory["momentum_constraint_derived"] is True
    assert theory["constraint_principal_subsystem_symmetric_hyperbolic"] is True
    assert theory["standard_adm_representative_only"] is True
    assert theory["adm_full_H2"] is False
    assert theory["adm_full_H3"] is False
    assert theory["adm_full_H4"] is False
    assert theory["full_metric_scalar_matter_system_strongly_hyperbolic"] is False
    assert theory["physical_hamiltonian_positive"] is False
    assert theory["constraint_preserving_boundary_conditions_instantiated"] is False
    assert theory["global_constraint_propagation"] is False
    assert theory["adm_lensing_prediction"] is False
    assert theory["adm_observational_support"] is False
    assert theory["scalar_hamiltonian_decision"].startswith("PARTIAL_SCALAR_ADM_HAMILTONIAN")
    assert theory["restricted_scalar_canonical_hamiltonian_derived"] is True
    assert theory["necessary_legendre_and_slice_health_conditions_derived"] is True
    assert theory["homogeneous_gate_energy_obstruction_derived"] is True
    assert theory["positive_principal_negative_energy_case_preserved"] is True
    assert theory["invalid_adm_time_slice_case_preserved"] is True
    assert theory["scalar_hamiltonian_CP11_4_complete"] is False
    assert theory["scalar_physical_hamiltonian_positive"] is False
    assert theory["scalar_full_no_ghost_result"] is False
    assert theory["scalar_full_gradient_stability"] is False
    assert theory["scalar_full_hyperbolicity"] is False
    assert theory["scalar_causality_established"] is False
    assert theory["deep_aqual_transition_decision"].startswith(
        "CONDITIONAL_EXACT_DEEP_AQUAL_TRANSITION_NO_GO"
    )
    assert theory["conditional_exact_transition_no_go_established"] is True
    assert theory["exact_deep_aqual_transition_is_C2"] is False
    assert theory["exact_deep_aqual_transition_is_uniformly_nondegenerate"] is False
    assert theory["positive_floor_regulator_removes_transition_degeneracy"] is True
    assert theory["positive_floor_regulator_preserves_exact_low_gradient_aqual"] is False
    assert theory["regulated_example_is_subluminal_relative_to_conformal_matter_cone"] is False
    assert theory["regulated_example_has_global_unbounded_domain_lower_bound"] is False
    assert theory["deep_aqual_transition_CP11_4_complete"] is False
    assert theory["deep_aqual_transition_healthy_action"] is False
    assert theory["formula_kinetic_reconstruction_decision"].startswith(
        "MINIMAL_FORMULA_TO_KINETIC_RECONSTRUCTION_DERIVED"
    )
    assert theory["formula_to_minimal_kinetic_map_derived"] is True
    assert theory["formula_registry_classes_classified"] == 60
    assert theory["formula_source_only_classes"] == 3
    assert theory["formula_auxiliary_dependent_classes"] == 57
    assert theory["quadrature_minimal_map_single_valued_and_locally_positive"] is True
    assert theory["quadrature_minimal_map_causal_relative_to_conformal_matter_cone"] is False
    assert theory["quadrature_minimal_map_has_global_regular_unbounded_domain"] is False
    assert theory["rar_like_minimal_map_single_valued_globally"] is False
    assert theory["rar_like_minimal_map_gradient_stable_globally"] is False
    assert theory["formula_full_covariant_bridge_derived"] is False
    assert theory["formula_surviving_physical_candidate_selected"] is False
    assert theory["formula_kinetic_CP11_1_complete"] is False
    assert theory["formula_kinetic_CP11_4_complete"] is False
    assert theory["quadrature_action_decision"].startswith(
        "RESTRICTED_QUADRATURE_UNIVERSAL_CONFORMAL_ACTION_DERIVED"
    )
    assert theory["restricted_quadrature_action_defined"] is True
    assert theory["quadrature_motion_law_recovered_exactly"] is True
    assert theory["quadrature_universal_matter_photon_metric_defined"] is True
    assert theory["quadrature_separate_photon_adjustment_present"] is False
    assert theory["quadrature_scalar_stress_tensor_derived"] is True
    assert theory["quadrature_direct_conformal_lensing_shift_cancels"] is True
    assert theory["quadrature_quantitative_lensing_solution_derived"] is False
    assert theory["quadrature_local_static_energy_density_positive"] is True
    assert theory["quadrature_scalar_cone_causal"] is False
    assert theory["quadrature_low_gradient_transition_nondegenerate"] is False
    assert theory["quadrature_finite_gradient_endpoint_regular"] is False
    assert theory["quadrature_timelike_cosmological_branch_defined"] is False
    assert theory["quadrature_action_CP11_1_complete"] is False
    assert theory["quadrature_action_CP11_4_complete"] is False
    assert theory["quadrature_action_CP11_8_complete"] is False
    assert theory["quadrature_lensing_decision"].startswith(
        "RESTRICTED_QUADRATURE_LENSING_BACKREACTION_DERIVED"
    )
    assert theory["quadrature_restricted_lensing_backreaction_derived"] is True
    assert theory["quadrature_scalar_stress_lensing_source_nonzero"] is True
    assert theory["quadrature_lensing_backreaction_compactness_suppressed"] is True
    assert theory["quadrature_asymptotic_motion_lensing_match"] is False
    assert theory["quadrature_finite_isolated_scalar_energy"] is False
    assert theory["quadrature_standard_finite_ADM_mass_established"] is False
    assert theory["quadrature_global_quantitative_lensing_success"] is False
    assert theory["quadrature_lensing_CP11_8_complete"] is False
    assert theory["quadrature_lensing_CP11_10_complete"] is False
    assert theory["quadrature_vector_metric_decision"].startswith(
        "RESTRICTED_QUADRATURE_UNIVERSAL_VECTOR_METRIC_ACTION_DERIVED"
    )
    assert theory["quadrature_vector_metric_same_action_architecture"] is True
    assert theory["quadrature_vector_metric_universal_matter_photon_metric"] is True
    assert theory["quadrature_vector_metric_separate_photon_adjustment"] is False
    assert theory["quadrature_vector_metric_leading_motion_lensing_relation"] is True
    assert theory["quadrature_vector_metric_fixed_aether_scalar_causal"] is True
    assert theory["quadrature_vector_metric_full_causality"] is False
    assert theory["quadrature_vector_metric_quantitative_lensing"] is False
    assert theory["quadrature_vector_metric_gw_physical_gate"] is False
    assert theory["quadrature_vector_metric_Solar_System_complete"] is False
    assert theory["quadrature_vector_metric_cosmology_viable"] is False
    assert theory["quadrature_vector_metric_CP11_1_complete"] is False
    assert theory["quadrature_vector_metric_CP11_4_complete"] is False
    assert theory["quadrature_vector_metric_CP11_6_complete"] is False
    assert theory["quadrature_vector_metric_CP11_8_complete"] is False
    assert theory["quadrature_vector_metric_CP11_10_complete"] is False
    assert theory["quadrature_aether_mode_decision"].startswith(
        "RESTRICTED_AETHER_MODE_AND_PPN_CONDITIONS_DERIVED"
    )
    assert theory["quadrature_aether_five_mode_formulas_rechecked"] is True
    assert theory["quadrature_aether_finite_luminal_locus_exists"] is True
    assert theory["quadrature_aether_exact_gw_ppn_zero_regular"] is False
    assert theory["quadrature_aether_uniform_kinetic_margin"] is False
    assert theory["quadrature_aether_full_coupled_health"] is False
    assert theory["quadrature_aether_Solar_gate"] is False
    assert theory["quadrature_aether_GW_gate"] is False
    assert theory["quadrature_reduced_factorization_decision"].startswith(
        "RESTRICTED_STATIC_BRANCH_REDUCED_PRINCIPAL_FACTORIZATION_DERIVED"
    )
    assert theory["quadrature_reduced_six_mode_factorization"] is True
    assert theory["quadrature_reduced_six_mode_local_causality"] is True
    assert theory["quadrature_reduced_principal_scalar_mixing_present"] is False
    assert theory["quadrature_reduced_nonzero_W_factorization"] is False
    assert theory["quadrature_reduced_unreduced_constraint_hyperbolicity"] is False
    assert theory["quadrature_reduced_healthy_action"] is False
    assert theory["quadrature_combined_decision"].startswith(
        "RESTRICTED_STATIC_W_ZERO_COMBINED_TETRAD_SCALAR_SYMMETRIC_HYPERBOLICITY"
    )
    assert theory["quadrature_combined_symmetric_hyperbolicity"] is True
    assert theory["quadrature_combined_common_Cauchy_time"] is True
    assert theory["quadrature_combined_aether_necessary_bounds"] is True
    assert theory["quadrature_combined_all_mode_cherenkov_safety"] is False
    assert theory["quadrature_combined_full_health"] is False
    assert theory["quadrature_cherenkov_decision"].startswith(
        "RESTRICTED_STATIC_W_ZERO_QUADRATURE_SCALAR_CHERENKOV"
    )
    assert theory["quadrature_cherenkov_phase_space"] is True
    assert theory["quadrature_cherenkov_ultrarelativistic_source_nonzero"] is True
    assert theory["quadrature_cherenkov_fixed_s_alpha_decoupling"] is False
    assert theory["quadrature_cherenkov_rate_derived"] is False
    assert theory["quadrature_cherenkov_observational_exclusion"] is False
    assert theory["quadrature_cherenkov_cutoff_rate_decision"].startswith(
        "RESTRICTED_STATIONARY_W_ZERO_QUADRATURE_SCALAR_CHERENKOV_CUTOFF_RATE"
    )
    assert theory["quadrature_cherenkov_restricted_stationary_rate_derived"] is True
    assert theory["quadrature_cherenkov_conditional_cutoff_bound_derived"] is True
    assert theory["quadrature_cherenkov_physical_cutoff_established"] is False
    assert theory["quadrature_cherenkov_formation_recoil_rate_derived"] is False
    assert theory["quadrature_cherenkov_background_established"] is False
    assert theory["quadrature_cherenkov_survival_passed"] is False
    assert theory["quadrature_cherenkov_cutoff_rate_observational_exclusion"] is False
    assert receipt["counts"]["adm_constraint_symbolic_checks_passed"] == 18
    assert receipt["counts"]["adm_constraint_numeric_cases_passed"] == 3
    assert receipt["counts"]["scalar_hamiltonian_symbolic_checks_passed"] == 24
    assert receipt["counts"]["scalar_hamiltonian_numeric_cases_passed"] == 4
    assert receipt["counts"]["scalar_hamiltonian_designed_failures_preserved"] == 2
    assert receipt["counts"]["deep_aqual_transition_symbolic_checks_passed"] == 24
    assert receipt["counts"]["deep_aqual_transition_numeric_cases_passed"] == 4
    assert receipt["counts"]["formula_kinetic_symbolic_checks_passed"] == 16
    assert receipt["counts"]["formula_kinetic_quadrature_numeric_probes"] == 5
    assert receipt["counts"]["formula_kinetic_rar_witness_points"] == 2
    assert receipt["counts"]["quadrature_action_symbolic_checks_passed"] == 24
    assert receipt["counts"]["quadrature_action_numeric_branch_probes_passed"] == 4
    assert receipt["counts"]["quadrature_lensing_symbolic_checks_passed"] == 16
    assert receipt["counts"]["quadrature_lensing_numeric_probes_passed"] == 4
    assert receipt["counts"]["quadrature_vector_metric_symbolic_checks_passed"] == 21
    assert receipt["counts"]["quadrature_vector_metric_numeric_cases_passed"] == 4
    assert receipt["counts"]["quadrature_combined_symbolic_checks_passed"] == 37
    assert receipt["counts"]["quadrature_combined_numeric_cases_passed"] == 4
    assert receipt["counts"]["quadrature_cherenkov_symbolic_checks_passed"] == 22
    assert receipt["counts"]["quadrature_cherenkov_numeric_cases_passed"] == 4
    assert receipt["counts"]["quadrature_cherenkov_cutoff_rate_symbolic_checks_passed"] == 25
    assert receipt["counts"]["quadrature_cherenkov_cutoff_rate_numeric_cases_passed"] == 4
    assert receipt["counts"]["quadrature_aether_symbolic_checks_passed"] == 25
    assert receipt["counts"]["quadrature_aether_epsilon_cases_passed"] == 3
    assert receipt["counts"]["quadrature_reduced_factorization_symbolic_checks_passed"] == 22
    assert receipt["counts"]["quadrature_reduced_factorization_numeric_cases_passed"] == 4
    assert theory["scientific_claim_allowed"] is False


@pytest.mark.parametrize(
    "source_id,mutation,match",
    [
        (
            "xcop_shape_bridge_preflight",
            lambda value: value["claims"].__setitem__("real_scoring_executed", True),
            "shape bridge",
        ),
        (
            "missing_variable_preflight",
            lambda value: value["counts"].__setitem__("continuous_measurement_ready_rows", 1),
            "missing-variable",
        ),
        (
            "group_scale_bridge_acquisition_v2",
            lambda value: value["counts"].__setitem__("scientific_payload_rows_opened", 1),
            "group acquisition",
        ),
        (
            "act_erass_overlap_preflight",
            lambda value: value["population_gate"].__setitem__("rule_evaluated", True),
            "ACT/eRASS",
        ),
        (
            "act_erass_overlap_executor_v2",
            lambda value: value["claims"].__setitem__("overlap_count_computed", True),
            "ACT/eRASS executor",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value.__setitem__("scores_computed", 1),
            r"B\+E\+N V4 executor",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value["candidate_and_ablation_accounting"].__setitem__(
                "unique_asts_across_full_and_ablations", 79
            ),
            r"B\+E\+N V4 executor",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value["runtime_environment_contract"].__setitem__(
                "comparison_operator", "exact_binary64"
            ),
            r"B\+E\+N V4 executor",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value["result_validation_contract"].__setitem__(
                "terminal_success_marker_required_after_runtime_restoration", False
            ),
            r"B\+E\+N V4 executor",
        ),
        (
            "matter_lensing_theory_preflight",
            lambda value: value["claim_boundary"].__setitem__("healthy_action_completed", True),
            r"matter\+lensing theory",
        ),
        (
            "matter_lensing_symbolic_derivation",
            lambda value: value["claim_boundary"].__setitem__("full_H2_passed", True),
            "bounded symbolic",
        ),
        (
            "matter_lensing_external_metric_principal_symbol",
            lambda value: value["claim_boundary"].__setitem__("full_H4_passed", True),
            "external-metric principal-symbol",
        ),
        (
            "matter_lensing_kinetic_gate_conditional_no_go",
            lambda value: value["claim_boundary"].__setitem__(
                "unconditional_action_no_go_established", True
            ),
            "conditional kinetic-gate",
        ),
        (
            "matter_lensing_kinetic_gate_conditional_no_go",
            lambda value: value["counts"].__setitem__("observational_files_opened", 1),
            "conditional kinetic-gate",
        ),
        (
            "group_scale_source_audit_v3",
            lambda value: value["counts"].__setitem__("ready_science_lanes", 1),
            "group-scale V3",
        ),
        (
            "group_scale_xclass_identity_executor_v1",
            lambda value: value["execution_accounting"].__setitem__("get_attempts", 1),
            "guarded X-CLASS",
        ),
        (
            "matter_lensing_split_gate_source_bound",
            lambda value: value["claim_boundary"].__setitem__(
                "physical_source_law_established", True
            ),
            "source-bound",
        ),
        (
            "matter_lensing_universal_conformal_source",
            lambda value: value["claim_boundary"].__setitem__(
                "metric_backreaction_established", True
            ),
            "conformal-source",
        ),
        (
            "matter_lensing_solar_gw_necessary_conditions",
            lambda value: value["gate_adjudication"].__setitem__("solar_gate_passed", True),
            "Solar/GW",
        ),
        (
            "matter_lensing_flrw_necessary_conditions",
            lambda value: value["adjudication"].__setitem__(
                "healthy_late_time_history_exists", True
            ),
            "FLRW",
        ),
        (
            "matter_lensing_covariant_field_equations",
            lambda value: value["claim_boundary"].__setitem__("lensing_success_established", True),
            "covariant field-equation",
        ),
        (
            "matter_lensing_adm_constraint_propagation",
            lambda value: value["claim_boundary"].__setitem__(
                "full_characteristic_system_established", True
            ),
            "ADM constraint-propagation",
        ),
        (
            "matter_lensing_scalar_hamiltonian_necessary_conditions",
            lambda value: value["claim_boundary"].__setitem__(
                "full_no_ghost_result_established", True
            ),
            "scalar Hamiltonian",
        ),
        (
            "matter_lensing_deep_aqual_transition_tradeoff",
            lambda value: value["claim_boundary"].__setitem__("healthy_action_established", True),
            "deep-AQUAL transition",
        ),
        (
            "shared_formula_scalar_kinetic_reconstruction",
            lambda value: value["claim_boundary"].__setitem__(
                "surviving_physical_candidate_selected", True
            ),
            "formula kinetic reconstruction",
        ),
        (
            "shared_quadrature_covariant_action",
            lambda value: value["claim_boundary"].__setitem__(
                "quantitative_lensing_prediction_established", True
            ),
            "quadrature action",
        ),
        (
            "shared_quadrature_lensing_backreaction",
            lambda value: value["adjudication"].__setitem__(
                "same_action_lensing_matches_scalar_motion_enhancement_asymptotically", True
            ),
            "quadrature lensing",
        ),
        (
            "shared_quadrature_universal_vector_metric",
            lambda value: value["adjudication"].__setitem__("gw_physical_gate_passed", True),
            "quadrature vector-metric",
        ),
        (
            "shared_quadrature_aether_mode_conditions",
            lambda value: value["claim_boundary"].__setitem__(
                "full_covariant_health_established", True
            ),
            "quadrature aether-mode",
        ),
        (
            "shared_quadrature_reduced_principal_factorization",
            lambda value: value["claim_boundary"].__setitem__("healthy_action_established", True),
            "quadrature reduced-principal",
        ),
        (
            "shared_quadrature_combined_tetrad_hyperbolicity",
            lambda value: value["claim_boundary"].__setitem__(
                "all_mode_cherenkov_safety_established", True
            ),
            "quadrature combined symmetric-hyperbolicity",
        ),
        (
            "shared_quadrature_scalar_cherenkov_obstruction",
            lambda value: value["claim_boundary"].__setitem__(
                "observational_scalar_cherenkov_exclusion_established", True
            ),
            "quadrature scalar Cherenkov obstruction",
        ),
        (
            "shared_quadrature_scalar_cherenkov_cutoff_rate",
            lambda value: value["claim_boundary"].__setitem__("physical_cutoff_established", True),
            "quadrature scalar Cherenkov cutoff-rate",
        ),
        (
            "shared_quadrature_scalar_local_cutoff_ceiling",
            lambda value: value["claim_boundary"].__setitem__(
                "strong_coupling_scale_established", True
            ),
            "quadrature scalar local-cutoff coefficient ceiling",
        ),
    ],
)
def test_new_source_semantics_fail_closed(source_id: str, mutation: object, match: str) -> None:
    sources = package._load_sources(ROOT, package.load_config(ROOT))
    changed = copy.deepcopy(sources)
    mutation(changed[source_id])  # type: ignore[operator]
    with pytest.raises(package.GravityClusterManuscriptPackageError, match=match):
        package._validate_new_source_semantics(changed)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["claim_boundary"].__setitem__("bounded_paper_ready", True),
            "claim boundary",
        ),
        (
            lambda value: value["environment_freeze"].__setitem__("numpy", "latest"),
            "environment",
        ),
        (
            lambda value: value["source_bindings"][0].__setitem__("content_sha256", "0" * 64),
            "content changed",
        ),
    ],
)
def test_claim_environment_and_evidence_mutations_fail_closed(mutation: object, match: str) -> None:
    config = copy.deepcopy(package.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    if match == "content changed":
        with pytest.raises(package.GravityClusterManuscriptPackageError, match=match):
            package._load_sources(ROOT, config)
    else:
        with pytest.raises(package.GravityClusterManuscriptPackageError, match=match):
            package.validate_config(config)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / package.OUTPUT_PATH).read_text(encoding="utf-8"))
    package.validate_receipt(stored, ROOT)
    assert stored == package.build_receipt(ROOT)
