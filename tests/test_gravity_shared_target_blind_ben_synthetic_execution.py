from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    gravity_shared_target_blind_ben_synthetic_execution as ben,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ben.CONFIG_PATH
RECEIPT = ROOT / ben.RECEIPT_PATH


def raw_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_contract() -> dict[str, object]:
    return ben.load_config(CONFIG, ben.file_sha256(CONFIG))


def registry() -> dict[str, object]:
    return ben.build_candidate_registry(load_contract())


def test_contract_binds_frozen_shared_evaluator_and_lead_receipts() -> None:
    config = load_contract()
    assert config["source_bindings"] == ben.EXPECTED_SOURCE_BINDINGS
    assert config["generation_packet_sha256"] == (
        "7bfae7b2d7cfe615bb5f80e2f375e01b72f249237a126c592364eb1dfeaf2785"
    )
    for binding in config["source_bindings"].values():
        assert ben.file_sha256(ROOT / binding["path"]) == binding["file_sha256"]


def test_finite_grammar_roles_and_exclusions_are_exact() -> None:
    assert {role: len(rows) for role, rows in ben.COMPONENT_GRAMMAR.items()} == {
        "E_local_base": 4,
        "B_continuous_gate": 4,
        "N_additive_channel": 5,
        "A_nuisance": 3,
    }
    assert ben.sha256_value(ben.COMPONENT_GRAMMAR) == (
        "c319ad7ca481025d15cc5a0500bb724c5f0408e0eb95f59faa63843b12c5dd11"
    )
    assert ben.SAME_ACTION_BOUNDARY["independent_photon_multiplier_allowed"] is False
    assert ben.SAME_ACTION_BOUNDARY["real_lensing_score_allowed"] is False
    assert ben.CLAIM_BOUNDARY["same_action_derived"] is False


def test_raw_and_exact_equivalence_accounting() -> None:
    candidates = registry()
    ben.validate_registry(candidates)
    assert candidates["raw_candidate_count"] == 240
    assert candidates["equivalence_class_count"] == 60
    assert candidates["equivalence_classes_by_provenance"] == {
        "known_rewrite": 6,
        "known_combination": 18,
        "potentially_new_synthesis": 36,
    }
    assert sum(row["raw_member_count"] for row in candidates["equivalence_classes"]) == 240


def test_identity_aliases_and_null_channel_collapse_exactly() -> None:
    candidates = registry()
    raw = {row["raw_id"]: row for row in candidates["raw_candidates"]}
    base = "raw.E.newtonian.B.low_acceleration.N.radial_tail.A.off"
    e_alias = "raw.E.newtonian_identity_alias.B.low_acceleration.N.radial_tail.A.off"
    b_alias = "raw.E.newtonian.B.low_acceleration_identity_alias.N.radial_tail.A.off"
    n_alias = "raw.E.newtonian.B.low_acceleration.N.radial_tail_identity_alias.A.off"
    a_alias = "raw.E.newtonian.B.low_acceleration.N.radial_tail.A.off_identity_alias"
    hashes = {
        raw[name]["canonical_expression_sha256"]
        for name in (base, e_alias, b_alias, n_alias, a_alias)
    }
    assert len(hashes) == 1
    null_low = "raw.E.newtonian.B.low_acceleration.N.null_ablation.A.off"
    null_state = "raw.E.newtonian.B.state_weighted.N.null_ablation.A.off"
    null_geometry = "raw.E.newtonian.B.geometry_weighted.N.null_ablation.A.off"
    assert (
        len(
            {
                raw[name]["canonical_expression_sha256"]
                for name in (null_low, null_state, null_geometry)
            }
        )
        == 1
    )


def test_equivalence_is_structural_and_raw_parity_is_exact() -> None:
    control = ben.raw_to_canonical_parity(registry())
    assert control["raw_candidates_checked"] == 240
    assert control["maximum_raw_to_canonical_abs_difference"] == ("0.000000000000e+00")
    assert control["all_raw_to_canonical_parity_pass"] is True


def test_dimension_limits_and_channel_roles_pass() -> None:
    controls = ben.dimension_and_limit_controls(registry())
    assert controls["all_intermediates_dimensionless"] is True
    assert controls["all_probe_predictions_finite_nonnegative"] is True
    assert controls["high_source_local_limit_pass"] is True
    assert controls["high_source_additive_suppression_pass"] is True
    assert controls["A_nuisance_bounded_calibration_pass"] is True
    assert controls["A_nuisance_minimum"] == "9.750000000000e-01"
    assert controls["A_nuisance_maximum"] == "1.025000000000e+00"
    assert controls["object_survey_or_class_switches"] == 0
    assert controls["M_temporal_phase_operator_occurrences"] == 0


def test_all_four_adapters_recover_all_injected_provenance_classes() -> None:
    result = ben.execute(load_contract())
    assert set(result["synthetic_domains"]) == set(ben.DOMAINS)
    for domain, row in result["synthetic_domains"].items():
        assert row["domain"] == domain
        assert row["synthetic_rows"] == 64
        assert row["real_rows_read"] == 0
        assert row["real_target_fields_read"] == []
        assert row["real_scores_computed"] == 0
        assert row["all_recovery_pass"] is True
        assert row["all_wrong_law_controls_pass"] is True
        assert set(row["recovery"]) == set(ben.PROVENANCE_LABELS)
        for label, recovery in row["recovery"].items():
            assert recovery["injected_provenance_label"] == label
            assert recovery["injected_class_recovered"] is True
            assert recovery["minimum_mse"] == "0.000000000000e+00"
            assert recovery["wrong_law_rejected"] is True
    xcop = result["synthetic_domains"]["xcop_thermodynamic"]
    assert xcop["recovery"]["known_combination"]["data_induced_tie_count"] == 2
    assert xcop["recovery"]["known_combination"]["unique_exact_recovery"] is False


def test_all_four_domain_ablations_are_nontrivial_and_additive() -> None:
    result = ben.execute(load_contract())
    for row in result["synthetic_domains"].values():
        ablations = row["channel_ablations"]
        assert ablations["additive_reconstruction_max_abs_error"] == ("0.000000000000e+00")
        assert ablations["N_channel_nonzero"] is True
        assert ablations["B_gate_replaced_by_constant_changes_output"] is True
        assert ablations["A_nuisance_off_changes_output"] is True
        assert ablations["N_zero_equals_base_only"] is True
        assert ablations["each_role_separately_evaluated"] is True


def test_same_action_metric_interface_is_synthetic_and_fail_closed() -> None:
    interface = ben.execute(load_contract())["same_action_metric_interface"]
    assert interface["synthetic_projection_parity_pass"] is True
    assert interface["synthetic_projection_max_abs_error"] == "0.000000000000e+00"
    assert interface["action_or_field_equations_frozen"] is False
    assert interface["synthetic_projection_is_physical_derivation"] is False
    assert interface["independent_photon_multiplier_allowed"] is False
    assert interface["real_lensing_score_allowed"] is False
    assert interface["real_lensing_unlock"] is False
    assert interface["empirical_score_key_present"] is False
    assert interface["empirical_rows_read"] == 0
    assert "empirical_score" not in interface


def test_candidate_generation_registry_contains_no_answer_or_domain_leakage() -> None:
    encoded = ben.canonical_json(registry()).lower()
    for token in (
        "object_id",
        "object_name",
        "survey",
        "galaxy_rotation",
        "xcop_thermodynamic",
        "group_bridge",
        "lensing_metric",
        "class_label",
        "observed",
        "response",
        "target_coefficient",
        "inferred_total_mass",
        "sealed_row",
    ):
        assert token not in encoded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("component_grammar_sha256", "0" * 64),
        ("rows_per_domain", 65),
        ("generation_packet_sha256", "0" * 64),
    ],
)
def test_config_contract_mutations_fail_closed(field: str, value: object) -> None:
    changed = copy.deepcopy(raw_config())
    changed[field] = value
    with pytest.raises(ben.BENSyntheticExecutionError, match="frozen contract changed"):
        ben.validate_config_contract(changed)


def test_claim_same_action_and_source_binding_mutations_fail_closed() -> None:
    for mutation in ("claim", "same_action", "binding"):
        changed = copy.deepcopy(raw_config())
        if mutation == "claim":
            changed["claim_boundary"]["ben_child_empirically_works"] = True
        elif mutation == "same_action":
            changed["same_action_boundary"]["real_lensing_score_allowed"] = True
        else:
            changed["source_bindings"]["shared_receipt"]["file_sha256"] = "0" * 64
        with pytest.raises(ben.BENSyntheticExecutionError, match="frozen contract changed"):
            ben.validate_config_contract(changed)


def test_ast_rejects_untyped_variable_and_invalid_dimensions() -> None:
    predictors = np.ones((3, 4), dtype=np.float64)
    with pytest.raises(ben.BENSyntheticExecutionError, match="not typed"):
        ben.evaluate_ast({"var": "outcome"}, predictors)
    with pytest.raises(ben.BENSyntheticExecutionError, match="operator changed"):
        ben.ast_dimension({"op": "fit_target", "args": [{"var": "x_source"}]})


def test_receipt_reconstructs_with_exact_accounting_and_claim_ceiling() -> None:
    checked = ben.check_receipt(CONFIG, ben.file_sha256(CONFIG), RECEIPT)
    assert checked["valid"] is True
    assert checked["decision"] == ben.DECISION
    assert checked["raw_candidate_count"] == 240
    assert checked["equivalence_class_count"] == 60
    assert checked["all_synthetic_controls_pass"] is True
    assert checked["real_scientific_evaluation_unlocked"] is False
    assert checked["same_action_derived"] is False
    assert checked["scientific_claim_allowed"] is False
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    accounting = receipt["compute_accounting"]
    assert accounting["total_formula_vector_evaluation_calls"] == 828
    assert accounting["candidate_registry_vector_evaluation_calls"] == 240
    assert accounting["candidate_score_comparisons"] == 720
    assert accounting["candidate_score_row_comparisons"] == 46_080
    assert accounting["real_formula_evaluation_calls"] == 0


def test_receipt_no_clobber_preserves_existing_bytes() -> None:
    before = RECEIPT.read_bytes()
    with pytest.raises(ben.BENSyntheticExecutionError, match="no-clobber"):
        ben.write_receipt(CONFIG, ben.file_sha256(CONFIG), RECEIPT)
    assert RECEIPT.read_bytes() == before
