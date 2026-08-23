from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import uncertain_invariant_discovery as U
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _receipt() -> dict[str, object]:
    return U.build_receipt(ROOT)


def _by_id() -> dict[str, dict[str, object]]:
    return {result["problem_id"]: result for result in _receipt()["results"]}


def test_uncertain_controls_retain_nine_training_candidates_and_five_survivors() -> None:
    receipt = _receipt()
    assert receipt["summary"] == {
        "censored_controls": 1,
        "controls": 5,
        "dependent_joint_controls": 1,
        "deployment_failed_candidates": 4,
        "deployment_surviving_candidates": 5,
        "marginal_false_positive_candidates_rejected": 3,
        "missingness_controls": 1,
        "noisy_controls": 1,
        "status": "PASS_COUPLED_UNCERTAIN_INVARIANT_BRANCH_CONTROLS",
        "target_blind_controls": 5,
        "training_candidates_retained": 9,
        "unit_hypothesis_branches_retained": 2,
        "unit_uncertainty_controls": 1,
    }
    assert all(
        result["target_access"]["target_visible_to_learner"] is False
        for result in receipt["results"]
    )


def test_noisy_intervals_retain_compatible_set_before_exact_deployment_filter() -> None:
    result = _by_id()["control.noisy-ratio-action"]
    assert [item["expression"] for item in result["training_candidates"]] == [
        "a*b/c",
        "a**2/b",
        "a*c/b**2",
    ]
    assert result["deployment"]["failed_candidates"] == 2
    assert result["creative_brief"]["deployment_surviving_coordinates"] == ["a*b/c"]
    assert result["independent_evaluators"]["finite_bounded_corner_checks"] == 12


def test_missing_active_variables_are_unresolved_without_pruning_coordinate() -> None:
    result = _by_id()["control.missing-ratio-action"]
    candidate = result["training_candidates"][0]
    assert candidate["expression"] == "a*b/c"
    assert candidate["evaluable_training_transformations"] == 2
    assert candidate["missing_training_transformations"] == 2
    assert [replay["status"] for replay in candidate["training_replays"]].count(
        "UNRESOLVED_MISSING_ACTIVE_VARIABLE"
    ) == 2
    assert result["deployment"]["surviving_candidates"] == 1


def test_one_sided_censoring_stays_set_valued_until_exact_deployment() -> None:
    result = _by_id()["control.censored-ratio-action"]
    assert [item["expression"] for item in result["training_candidates"]] == [
        "a*b/c",
        "a*c/b**2",
    ]
    assert any(
        replay["upper"] == "infinity"
        for candidate in result["training_candidates"]
        for replay in candidate["training_replays"]
    )
    assert result["deployment"]["failed_candidates"] == 1
    assert result["creative_brief"]["deployment_surviving_coordinates"] == ["a*b/c"]


def test_joint_support_rejects_false_candidates_admitted_by_marginal_factorization() -> None:
    result = _by_id()["control.dependent-joint-ratio-action"]
    assert [item["expression"] for item in result["training_candidates"]] == ["a*b/c"]
    assert result["independent_evaluators"] == {
        "agreement": True,
        "finite_bounded_corner_checks": 0,
        "global_unit_hypothesis_branches": 0,
        "interval_monotonicity_replays": 0,
        "joint_atom_replays": 392,
        "marginal_false_positive_candidates_rejected": 3,
    }
    replay = result["training_candidates"][0]["training_replays"][0]
    assert replay["status"] == "COMPATIBLE_JOINT_ATOM"
    assert replay["marginal_envelope_status"] == "COMPATIBLE_CONTAINS_ONE"
    assert result["creative_brief"]["dependence_semantics"] == (
        "finite_joint_support_without_marginal_factorization"
    )


def test_unit_hypotheses_remain_global_formula_branches_until_deployment() -> None:
    result = _by_id()["control.unit-uncertain-ratio-action"]
    assert [item["expression"] for item in result["training_candidates"]] == [
        "a*b/c",
        "a**2/b",
    ]
    assert [
        (candidate["expression"], branch["hypothesis_id"])
        for candidate in result["training_candidates"]
        for branch in candidate["compatible_unit_hypotheses"]
    ] == [("a*b/c", "b_three_c_two"), ("a**2/b", "b_half")]
    assert result["deployment"]["failed_candidates"] == 1
    assert result["creative_brief"]["deployment_surviving_coordinates"] == ["a*b/c"]
    assert result["creative_brief"]["retained_evidence_branches"] == [
        {"branch_ids": ["b_three_c_two"], "expression": "a*b/c"},
        {"branch_ids": ["b_half"], "expression": "a**2/b"},
    ]


def test_unit_hypothesis_cannot_switch_between_training_transformations() -> None:
    config, _ = U.load_config(ROOT)
    problem = copy.deepcopy(config["problems"][4])
    problem["training_transformations"][0]["observations"]["b"]["value"] = "16"
    with pytest.raises(U.UncertainInvariantError, match="candidate set is empty"):
        U.learn_problem(problem, config["policy"])


def test_deployment_mutation_changes_filter_without_erasing_training_set() -> None:
    config, _ = U.load_config(ROOT)
    problem = copy.deepcopy(config["problems"][0])
    problem["deployment_transformations"][0]["observations"]["c"]["value"] = "124"
    changed = U.learn_problem(problem, config["policy"])
    assert len(changed["training_candidates"]) == 3
    assert changed["deployment"]["surviving_candidates"] == 0
    assert changed["deployment"]["failed_candidates"] == 3


def test_sealed_target_changes_cannot_change_uncertain_learning() -> None:
    config, targets = U.load_config(ROOT)
    problem = config["problems"][0]
    learned_before = U.learn_problem(problem, config["policy"])
    changed_targets = copy.deepcopy(targets)
    changed_targets["controls"][0]["expected_training_candidate_set"] = [[1, 0, 0]]
    assert changed_targets != targets
    assert U.learn_problem(problem, config["policy"]) == learned_before


def test_resealed_receipt_cannot_turn_ambiguity_into_unique_formula() -> None:
    changed = copy.deepcopy(_receipt())
    changed["claims"]["unique_formula_identified"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(U.UncertainInvariantError, match="claim boundary"):
        U.validate_receipt(changed, ROOT)


def test_stored_receipt_reproduces_current_sources() -> None:
    stored = json.loads((ROOT / U.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert U.validate_receipt(stored, ROOT) == stored
