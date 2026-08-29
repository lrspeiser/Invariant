from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_shared_target_blind_ben_xcop_shape_preflight_v3 as preflight,
)


def config() -> dict:
    return json.loads((preflight.ROOT / preflight.CONFIG_PATH).read_text(encoding="utf-8"))


def test_v3_identity_and_committed_lineage_are_frozen() -> None:
    frozen = config()
    preflight.validate_config(frozen)
    assert frozen["lineage"]["v2_freeze_commit"] == ("4c779f00d20745b1d454ee3e8465c30319a4de75")
    assert frozen["lineage"]["v2_candidate_registry_content_sha256"] == (
        "45966eae73d7641ea982a7eea47aad883a9ff344baf121b91b901c32ef819f19"
    )
    assert frozen["lineage"]["v2_raw_candidates"] == 240
    assert frozen["lineage"]["v2_canonical_candidates"] == 60
    assert frozen["lineage"]["v3_changes_candidate_registry"] is False


def test_all_local_sparc_remains_development_only_and_restricted() -> None:
    sparc = config()["populations"]["sparc"]
    assert sparc["all_locally_accessible_rows_role"] == ("development_only_for_this_descendant")
    assert sparc["local_confirmation_role_exists"] is False
    assert sparc["local_confirmation_claim_allowed"] is False
    assert sparc["planned_score_subset_objects"] == 139
    assert sparc["planned_score_subset_rows"] == 2720
    assert sparc["rows_outside_planned_subset_scored"] == 0


def test_exact_eight_xcop_development_objects_and_holdout_exclusion() -> None:
    xcop = config()["populations"]["xcop"]
    assert xcop["objects"] == preflight.EXPECTED_XCOP_OBJECTS
    assert xcop["predictor_density_rows_metadata_count"] == 521
    assert xcop["pressure_temperature_response_rows_metadata_count"] == 184
    assert xcop["independent_item59_holdouts_forbidden"] == [
        "A2029",
        "A3158",
        "A644",
        "RXC1825",
    ]


def test_shape_mapping_is_predictor_only_and_dimensionally_closed() -> None:
    frozen = config()
    inputs = frozen["predictor_only_input_mapping"]
    mapping = frozen["shape_output_mapping"]
    assert inputs["required_density_columns"] == ["RW_X", "NE"]
    assert inputs["response_fields_used_to_construct_candidate_input_or_shape"] == []
    assert inputs["candidate_input_order"] == ["q", "x", "s", "1"]
    assert all(
        inputs[key] is True
        for key in (
            "no_r500_required",
            "no_p500_required",
            "no_t500_required",
            "no_outer_pressure_required",
        )
    )
    assert mapping["physical_equation"] == ("dP_e/dr=-mu_gas_particle*m_p*n_e(r)*g_c(r)")
    assert mapping["composition_coefficient"] == {
        "symbol": "mu_gas_particle",
        "value": 0.61,
        "definition": "mean mass per thermal gas particle in proton-mass units, not mean mass per electron",
        "electron_pressure_derivation": "P_gas=(mu_e/mu_gas_particle)*P_e and rho_gas=mu_e*m_p*n_e, so dP_e/dr=-mu_gas_particle*m_p*n_e*g",
        "shape_effect": "constant coefficient is absorbed by b_P and b_T and cannot change H_c curvature",
    }
    assert mapping["dimensionless_drop_equation"] == ("H_c(x)=integral_from_x_to_1 q(u)*f_c(u) du")
    assert mapping["shared_shape_equation"] == (
        "phi_c(x;zeta)=zeta+(1-zeta)*H_c(x), with 0<=zeta<=1"
    )
    assert mapping["pressure_shape_equation"] == "P_SZ_hat(x)=b_P*phi_c(x;zeta)"
    assert mapping["temperature_shape_equation"] == ("T_X_hat(x)=b_T*phi_c(x;zeta)/q(x)")
    assert set(mapping["dimensions"].values()) == {
        "dimensionless",
        "dimensionless released P_SZ units",
        "dimensionless released T_X units",
        "dimensionless profiled coupled-shape nuisances",
    }
    assert mapping["mapping_ready"] is True
    assert mapping["H_c_is_target_independent"] is True
    assert mapping["observable_normalization_and_boundary_are_response_profiled_at_scoring"] is True
    assert mapping["profiled_nuisances_do_not_enter_H_c_or_candidate_generation"] is True


def test_physical_scales_and_measured_boundary_are_never_mapping_inputs() -> None:
    mapping = config()["shape_output_mapping"]
    assert mapping["mapping_never_uses"] == [
        "P500",
        "T500",
        "outermost_P_SZ_as_anchor",
        "R500",
        "hydrostatic_mass",
        "lensing",
        "inferred_total_mass",
    ]
    assert mapping["absolute_amplitude_identifiable"] is False
    assert mapping["absolute_boundary_pressure_identifiable"] is False
    assert mapping["pressure_temperature_cross_calibration_identifiable"] is False


def test_xcop_comparators_are_exact_predictor_only_and_domain_specific() -> None:
    comparators = config()["xcop_comparator_contract"]
    assert comparators["frozen_before_response_access"] is True
    assert comparators["cross_domain_comparator_identity_differs"] is True
    assert comparators["sparc_comparators_unchanged"] == [
        "newtonian_baryons",
        "empirical_rar",
    ]
    assert comparators["xcop_comparators"] == [
        "gas_only_newtonian_shape",
        "uniform_acceleration_shape_control",
    ]
    gas = comparators["gas_only_newtonian_shape"]
    assert gas["inner_continuation"] == "q(u)=q(x_min) for 0<=u<x_min"
    assert gas["enclosed_source_equation"] == (
        "m_q(x_min)=q(x_min)*x_min^3/3; for x>=x_min, "
        "m_q(x)=m_q(x_min)+integral_from_x_min_to_x q(u)*u^2 du"
    )
    assert gas["raw_acceleration_shape"] == "f_N_raw(x)=m_q(x)/x^2"
    assert gas["normalization"] == "f_N=f_N_raw/max(f_N_raw) on predictor nodes"
    assert "gas-only" in gas["stellar_handling"]
    uniform = comparators["uniform_acceleration_shape_control"]
    assert uniform["equation"] == "f_U(x)=1 on every predictor node"
    assert uniform["normalization"] == "already max-normalized"
    assert "absolute y=g_bar/a0" in comparators["why_empirical_rar_is_not_an_xcop_control"]
    assert comparators["post_response_comparator_choice_allowed"] is False


def test_nuisance_profile_is_fixed_symmetric_and_nonconfirmatory() -> None:
    nuisance = config()["nuisance_contract"]
    assert nuisance["nuisances_per_cluster_per_candidate_or_comparator"] == [
        "zeta",
        "b_P",
        "b_T",
    ]
    assert nuisance["shared_shape_basis"] == "phi=zeta+(1-zeta)*H_c"
    assert nuisance["pressure_design_column"] == "phi"
    assert nuisance["temperature_design_column"] == "phi/q"
    assert nuisance["zeta_grid_points"] == 4097
    assert nuisance["zeta_grid_step"] == 1 / 4096
    assert "one-column nonnegative weighted least-squares" in nuisance["fit_method"]
    assert nuisance["nuisances_are_never_candidate_inputs"] is True
    assert nuisance["nuisances_are_never_reused_between_candidates_or_comparators"] is True
    assert nuisance["nuisance_fit_is_development_only_not_confirmation"] is True
    assert "zeta=1" in nuisance["shared_boundary_ratio_diagnostic"]
    assert nuisance["parameter_count_per_cluster"] == 3
    assert nuisance["parameter_count_eight_clusters"] == 24
    assert nuisance["aggregate_response_bins_metadata_count"] == 184
    assert nuisance["aggregate_nominal_residual_degrees_of_freedom"] == 160
    assert nuisance["four_independent_affine_nuisance_control_scored"] is False
    assert nuisance["matched_control_rule"] == (
        "every X-COP full candidate, X-COP ablation, gas-only Newtonian shape control, "
        "and uniform-acceleration shape control receives exactly the same three-parameter "
        "coupled profile; empirical RAR remains SPARC-only"
    )
    jacobian = nuisance["response_node_jacobian"]
    assert jacobian["parameter_order"] == ["zeta", "b_P", "b_T"]
    assert jacobian["pressure_row"] == ["b_P*(1-H)", "phi", "0"]
    assert jacobian["temperature_row"] == ["b_T*(1-H)/q", "0", "phi/q"]
    assert jacobian["rank_required"] == 3
    assert jacobian["relative_singular_value_floor"] == 2**-26
    assert jacobian["maximum_condition_number"] == 2**26
    assert "response-fitted" in jacobian["evaluation_point"]
    assert "not response-independent" in jacobian["weighting"]
    assert "generic rank possibility" in jacobian["generic_structural_rank_precheck"]
    assert "only after authorized scoring" in jacobian["conditioning_evidence_timing"]
    assert "neither actual rank nor empirical conditioning" in jacobian["pre_score_claim"]
    assert "three pressure and three temperature" in jacobian["bin_requirements"]
    assert "max(H)-min(H)" in jacobian["variation_requirement"]


def test_item61_score_and_no_single_veto_policy_are_preserved() -> None:
    score = config()["score_contract"]
    assert score["primary_metric_id"] == ("equal_object_mean_squared_standardized_residual")
    assert score["comparators_by_domain"] == {
        "sparc": ["newtonian_baryons", "empirical_rar"],
        "xcop": [
            "gas_only_newtonian_shape",
            "uniform_acceleration_shape_control",
        ],
    }
    assert score["weighting"] == "equal object then equal observable then equal row"
    assert score["numeric_improvement_threshold"] is None
    assert score["single_counterexample_terminal"] is False
    assert score["counterexample_count_alone_terminal"] is False
    assert score["finite_sample_may_prune_formula_family"] is False
    assert score["row_level_pooling_for_selection"] is False
    assert score["no_new_folds"] is True
    assert score["xcop_primary_is_joint_coupled_shape_score"] is True
    assert score["not_absolute_joint_physical_prediction"] is True


def test_candidate_selection_and_ablations_are_frozen_before_targets() -> None:
    selection = config()["selection_and_ablation"]
    assert selection["fixed_ablations_per_candidate"] == [
        "N_zero_ablation",
        "B_unity_gate_ablation",
        "A_off_nuisance_ablation",
    ]
    assert "ascending class_id first" in selection["selection_rule"]
    assert selection["post_response_candidate_generation_allowed"] is False
    assert selection["post_response_formula_repair_allowed"] is False
    assert selection["threshold_tuning_from_rows_allowed"] is False
    assert selection["single_object_veto_allowed"] is False
    assert "H_nonnegative_outer_zero_negative_derivative_gate" in selection["eligibility"]
    assert "pressure_nonincreasing_outward_gate" in selection["eligibility"]
    assert "response_node_jacobian_rank3_condition_gate" in selection["eligibility"]
    gates = selection["shape_structure_gates"]
    assert gates["integration_identity"] == "H(x)=integral_x^1 q(u)f(u)du"
    assert "H>=-tol_H" in gates["nonnegative"]
    assert "H(x_outer)" in gates["outer_boundary"]
    assert "=-0.5*(q[j]*f[j]+q[j+1]*f[j+1])<0" in gates["derivative"]
    assert "nonincreasing outward" in gates["pressure_slope"]
    assert "no fixed sign" in gates["temperature_slope"]


def test_compute_ceiling_is_exact_and_zero_cost() -> None:
    ceiling = config()["compute_ceiling"]
    assert ceiling["ablation_variants"] == 3 * 60
    assert ceiling["formula_domain_batches_per_backend"] == (60 + 180) * 2 + 4
    assert ceiling["sparc_formula_row_cells_per_backend"] == 242 * 2720
    assert ceiling["xcop_formula_row_cells_per_backend"] == 242 * 521
    assert ceiling["total_formula_row_cells_per_backend"] == 784_322
    assert ceiling["total_formula_row_cells_both_backends"] == 1_568_644
    assert ceiling["xcop_coupled_three_parameter_nuisance_fits"] == 242 * 8
    assert ceiling["xcop_zeta_objective_evaluations"] == 242 * 8 * 4097
    assert ceiling["xcop_analytic_scale_solves"] == 2 * 242 * 8 * 4097
    assert ceiling["maximum_object_score_reductions"] == 242 * (139 + 8)
    assert ceiling["maximum_response_row_score_terms"] == 242 * (2720 + 184)
    assert ceiling["maximum_payload_file_opens"] == 1 + 8 * 3
    assert ceiling["network_calls"] == 0
    assert ceiling["model_calls"] == 0
    assert ceiling["paid_calls"] == 0
    assert ceiling["maximum_api_spend_usd"] == 0.0


def test_zero_access_chronology_is_exact() -> None:
    chronology = config()["zero_access_chronology"]
    assert chronology["v3_contract_frozen_before_payload_access"] is True
    for key, value in chronology.items():
        if key != "v3_contract_frozen_before_payload_access":
            assert value == 0


def test_current_manifest_is_unauthorized_and_rejected_before_payload() -> None:
    frozen = config()
    authorization = preflight.read_json(preflight.ROOT / preflight.AUTHORIZATION_PATH)
    preflight.validate_authorization(authorization, frozen, require_authorized=False)
    assert authorization["authorized"] is False
    assert authorization["sparc_objects"] == 0
    assert authorization["sparc_rows"] == 0
    assert authorization["xcop_objects"] == []
    assert authorization["xcop_predictor_rows"] == 0
    assert authorization["xcop_response_rows"] == 0
    assert all(value == 0 or value == 0.0 for value in authorization["compute_ceiling"].values())
    with pytest.raises(
        preflight.BENXCOPShapePreflightV3Error,
        match="UNAUTHORIZED_BEFORE_PAYLOAD_LOAD",
    ):
        preflight.production_preflight(preflight.ROOT / preflight.AUTHORIZATION_PATH)


def test_exact_future_authorization_only_returns_metadata_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = config()
    receipt = preflight.read_json(preflight.ROOT / preflight.RECEIPT_PATH)
    authorized = {
        "schema_version": preflight.AUTHORIZATION_SCHEMA,
        "authorization_id": "future-explicit-approval",
        "authorized": True,
        "approved_by": "future-approver",
        "approved_at": "2099-01-01T00:00:00Z",
        "config_file_sha256": preflight.CONFIG_FILE_SHA256,
        "preflight_receipt_file_sha256": preflight.file_sha256(
            preflight.ROOT / preflight.RECEIPT_PATH
        ),
        "preflight_receipt_content_sha256": receipt["content_sha256"],
        "v2_freeze_commit": frozen["lineage"]["v2_freeze_commit"],
        "candidate_registry_content_sha256": frozen["lineage"][
            "v2_candidate_registry_content_sha256"
        ],
        "sparc_role": "development_only",
        "sparc_objects": 139,
        "sparc_rows": 2720,
        "xcop_role": "development_only",
        "xcop_objects": preflight.EXPECTED_XCOP_OBJECTS,
        "xcop_predictor_rows": 521,
        "xcop_response_rows": 184,
        "compute_ceiling": frozen["compute_ceiling"],
        "claim_acknowledgements": preflight.EXPECTED_ACKNOWLEDGEMENTS,
    }
    preflight.validate_authorization(authorized, frozen, require_authorized=True)
    original = preflight.read_json

    def fake_read(path: Path) -> dict:
        if path == Path("authorized-v3.json"):
            return authorized
        return original(path)

    monkeypatch.setattr(preflight, "read_json", fake_read)
    result = preflight.production_preflight(Path("authorized-v3.json"))
    assert result == {
        "metadata_ready": True,
        "authorization_id": "future-explicit-approval",
        "payload_loader_present": False,
        "successor_executor_required": True,
        "payload_rows_read": 0,
        "real_candidates_scored": 0,
    }


@pytest.mark.parametrize(
    "section",
    [
        "source_bindings",
        "lineage",
        "populations",
        "predictor_only_input_mapping",
        "shape_output_mapping",
        "xcop_comparator_contract",
        "nuisance_contract",
        "score_contract",
        "selection_and_ablation",
        "compute_ceiling",
        "zero_access_chronology",
        "production_gate",
        "approval_schema",
        "claim_boundary",
    ],
)
def test_section_mutations_fail_closed(section: str) -> None:
    mutated = copy.deepcopy(config())
    mutated[section][next(iter(mutated[section]))] = "tampered"
    with pytest.raises(preflight.BENXCOPShapePreflightV3Error, match=f"frozen {section}"):
        preflight.validate_config(mutated)


def test_authorization_mutation_fails_closed() -> None:
    frozen = config()
    authorization = preflight.read_json(preflight.ROOT / preflight.AUTHORIZATION_PATH)
    authorization["xcop_response_rows"] = 1
    with pytest.raises(preflight.BENXCOPShapePreflightV3Error):
        preflight.validate_authorization(authorization, frozen, require_authorized=False)


def test_atomic_writer_refuses_overwrite_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    target = tmp_path / "receipt.json"
    preflight.write_json_no_clobber(target, {"first": True})
    original = target.read_bytes()
    with pytest.raises(preflight.BENXCOPShapePreflightV3Error, match="no-clobber"):
        preflight.write_json_no_clobber(target, {"first": False})
    assert target.read_bytes() == original
    assert json.loads(original) == {"first": True}


def test_v3_source_has_no_payload_or_scoring_loader() -> None:
    source = (
        preflight.ROOT
        / "src/sigma_theory_compiler/gravity_shared_target_blind_ben_xcop_shape_preflight_v3.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "fits.open",
        "import astropy",
        "import numpy",
        "import pandas",
        "sparc_rotation_curves_full_v1",
        "_density_L1.fits",
        "_pressure.fits",
        "_temperature.fits",
    ):
        assert forbidden not in source
    assert "payload_loader_present_in_v3" in source
    assert "scoring_executor_present_in_v3" in source


def test_claim_ceiling_is_shape_only_and_nonconfirmatory() -> None:
    claims = config()["claim_boundary"]
    assert claims["predictor_only_xcop_shape_basis_frozen"] is True
    assert claims["parameter_free_target_independent_observable_mapping"] is False
    assert claims["response_profiled_nuisance_required_at_scoring"] is True
    assert claims["xcop_newtonian_control_is_gas_only"] is True
    assert claims["xcop_empirical_rar_control_defined"] is False
    assert claims["xcop_comparators_frozen_predictor_only"] is True
    assert claims["response_node_identifiability_gate_frozen"] is True
    assert claims["absolute_pressure_or_temperature_prediction"] is False
    assert claims["shared_pressure_temperature_calibration_tested"] is False
    assert claims["hydrostatic_equilibrium_proven"] is False
    assert claims["nonthermal_support_modeled"] is False
    assert claims["production_authorized"] is False
    assert claims["real_scoring_executed"] is False
    assert claims["scientific_claim_allowed_now"] is False
    assert (
        "development-only coupled radial shape compatibility" in claims["future_pass_claim_ceiling"]
    )


def test_frozen_receipt_reconstructs_exactly() -> None:
    result = preflight.check()
    assert result["valid"] is True
    assert result["decision"] == preflight.DECISION
    assert result["mapping_ready"] is True
    assert result["authorized"] is False
    assert result["payload_rows_read"] == 0
    assert result["real_candidates_scored"] == 0
