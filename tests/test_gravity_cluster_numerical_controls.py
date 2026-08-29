from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_numerical_controls as controls

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stored() -> dict[str, object]:
    return json.loads((ROOT / controls.OUTPUT_PATH).read_text(encoding="utf-8"))


def test_stored_controls_rebuild_and_pass_without_opening_new_targets(
    stored: dict[str, object],
) -> None:
    controls.validate_receipt(stored, ROOT)
    assert stored["decision"] == "NUMERICAL_CONTROLS_PASS_DEVELOPMENT_ONLY"
    assert stored["claims"]["all_CP6_tasks_complete"] is True
    assert stored["claims"]["development_numerical_control_gate_passed"] is True
    assert stored["claims"]["independent_replication"] is False
    assert stored["sample"]["target_rows_opened"] == 0
    assert all(stored["task_pass"].values())
    assert set(stored["completed_goal_evidence"]) == set(controls.CP6_TASKS)


def test_all_injected_laws_recover_and_wrong_law_cannot_make_a_physical_claim(
    stored: dict[str, object],
) -> None:
    recovery = stored["synthetic_recovery"]
    assert recovery["passed"] is True
    assert recovery["library_variants"] == 2027
    assert [row["selected_class"] for row in recovery["injections"]] == [
        "newtonian_baryons",
        "empirical_rar",
        "cross_scale_boundary",
        "GR_PLUS_NFW",
        "WRONG_REVERSED_NFW",
    ]
    wrong = recovery["injections"][-1]
    assert wrong["physical_claim_allowed"] is False
    assert wrong["wrong_law_physical_claim_rejected"] is True


def test_full_grammar_null_rate_and_gpu_agreement_pass_frozen_thresholds(
    stored: dict[str, object],
) -> None:
    false = stored["false_selection"]
    assert false["trials"] == 4096
    assert false["search_variants"] == 2025
    assert false["qualifying_false_selections"] == 70
    assert false["false_selection_fraction"] == pytest.approx(0.01708984375)
    assert false["passed"] is True
    agreement = stored["implementation_agreement"]
    assert agreement["gpu_device"] == "NVIDIA GeForce RTX 5090"
    assert agreement["cpu_gpu_maximum_absolute_score_difference"] < 1e-8
    assert agreement["separate_direct_scorer_maximum_absolute_difference"] < 1e-8
    assert agreement["cpu_gpu_selected_variant_ids_match"] is True
    assert agreement["passed"] is True


def test_controls_retain_outer_radius_loss_and_underpowered_120_cluster_cap(
    stored: dict[str, object],
) -> None:
    folds = stored["fold_controls"]
    assert folds["candidate_beats_nfw_all_leave_one_out"] is True
    assert folds["candidate_beats_nfw_both_instrument_observable_strata"] is True
    assert folds["candidate_beats_nfw_all_radial_blocks"] is False
    assert folds["radial_blocks"][-1]["candidate_beats_nfw"] is False
    power = stored["prospective_power_and_stopping"]
    assert power["calculated_required_clusters"] == 192
    assert power["planned_independent_clusters"] == 120
    assert power["maximum_sample_sufficient"] is False
    assert power["planned_approximate_power"] < power["target_power"]


def test_target_access_threshold_weakening_and_optional_stopping_fail_closed() -> None:
    config = controls.load_config(ROOT)
    opened = copy.deepcopy(config)
    opened["sample_contract"]["target_rows_opened"] = 1
    with pytest.raises(controls.GravityClusterControlError, match="sample seal"):
        controls.validate_config(opened, ROOT)

    weakened = copy.deepcopy(config)
    weakened["implementation_agreement"]["maximum_absolute_score_difference"] = 1e-3
    with pytest.raises(controls.GravityClusterControlError, match="agreement"):
        controls.validate_config(weakened, ROOT)

    stopping = copy.deepcopy(config)
    stopping["power_and_stopping"]["optional_stopping"] = True
    with pytest.raises(controls.GravityClusterControlError, match="stopping"):
        controls.validate_config(stopping, ROOT)
