from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import learned_invariant_discovery as L
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _receipt() -> dict[str, object]:
    return L.build_receipt(ROOT)


def _by_id() -> dict[str, dict[str, object]]:
    return {result["problem_id"]: result for result in _receipt()["results"]}


def test_three_learning_outcomes_retain_all_coordinate_branches() -> None:
    receipt = _receipt()
    assert receipt["summary"] == {
        "deployment_repaired_coordinates": 6,
        "identified_passes": 1,
        "problems": 3,
        "shift_rejections": 1,
        "status": "PASS_LEARNED_MULTI_INVARIANT_CONTROLS",
        "training_coordinates_retained": 7,
        "underdetermined_controls": 1,
    }
    assert {result["status"] for result in receipt["results"]} == {
        "PASS_LEARNED_INVARIANT_BASIS",
        "REJECT_TRAIN_ONLY_INVARIANT_SPACE",
        "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE",
    }
    assert all(
        result["target_access"]["target_visible_to_learner"] is False
        for result in receipt["results"]
    )


def test_drag_basis_is_learned_from_ratios_and_survives_deployment() -> None:
    drag = _by_id()["control.learned-drag-multi-coordinate"]
    assert drag["search"]["training_nullity"] == 2
    assert drag["search"]["augmented_nullity"] == 2
    assert len(drag["training_coordinates"]) == 2
    assert drag["deployment"]["coordinate_failures"] == 0
    assert all(
        replay["invariant_on_all_deployment_transformations"]
        for replay in drag["deployment"]["replays"]
    )
    evaluators = drag["independent_evaluators"]
    assert evaluators["agreement"] is True
    assert evaluators["training_fraction_rank"] == evaluators["training_sympy_rank"] == 4
    assert evaluators["augmented_fraction_rank"] == evaluators["augmented_sympy_rank"] == 4
    assert drag["sealed_control_evaluation"] == {
        "expected_status_matched": True,
        "sealed_subspace_matched": True,
        "target_visible_to_learner": False,
    }


def test_hidden_deployment_action_rejects_but_repairs_training_coordinates() -> None:
    shifted = _by_id()["control.shifted-hidden-action"]
    assert shifted["search"]["training_nullity"] == 2
    assert shifted["search"]["augmented_nullity"] == 1
    assert shifted["deployment"]["coordinate_failures"] == 2
    assert shifted["creative_brief"]["candidate_invariant_coordinates"]
    assert shifted["creative_brief"]["deployment_repaired_coordinates"] == [
        "output*input*scale*nuisance"
    ]
    assert shifted["status"] == "REJECT_TRAIN_ONLY_INVARIANT_SPACE"


def test_underidentified_actions_keep_three_coordinates_without_forced_choice() -> None:
    unresolved = _by_id()["control.underidentified-action"]
    assert unresolved["search"]["training_nullity"] == 3
    assert unresolved["search"]["maximum_identifiable_nullity"] == 2
    assert len(unresolved["training_coordinates"]) == 3
    assert unresolved["status"] == "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE"
    assert "requires more transformations" in unresolved["creative_brief"][
        "constraint_statement"
    ]


def test_shift_mutation_that_hides_the_new_action_changes_the_verdict() -> None:
    config, _ = L.load_config(ROOT)
    problem = copy.deepcopy(
        next(
            problem
            for problem in config["problems"]
            if problem["problem_id"] == "control.shifted-hidden-action"
        )
    )
    problem["deployment_transformations"][0]["ratios"] = {
        "output": "5",
        "input": "1/5",
        "scale": "1",
        "nuisance": "1",
    }
    changed = L.learn_problem(problem, config["policy"])
    assert changed["status"] == "PASS_LEARNED_INVARIANT_BASIS"
    assert changed["deployment"]["coordinate_failures"] == 0


def test_sealed_target_changes_cannot_change_target_blind_learning() -> None:
    config, targets = L.load_config(ROOT)
    problem = config["problems"][0]
    learned_before = L.learn_problem(problem, config["policy"])
    changed_targets = copy.deepcopy(targets)
    changed_targets["controls"][0]["expected_invariant_subspace_basis"] = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ]
    assert changed_targets != targets
    assert L.learn_problem(problem, config["policy"]) == learned_before
    assert learned_before["target_access"]["target_visible_to_learner"] is False


def test_resealed_receipt_cannot_promote_invariance_to_a_law() -> None:
    changed = copy.deepcopy(_receipt())
    changed["claims"]["empirical_law_discovered"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(L.LearnedInvariantError, match="claim boundary"):
        L.validate_receipt(changed, ROOT)


def test_stored_receipt_reproduces_current_public_and_sealed_sources() -> None:
    stored = json.loads((ROOT / L.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert L.validate_receipt(stored, ROOT) == stored
