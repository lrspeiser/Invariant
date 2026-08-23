from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import state_pair_invariant_discovery as S
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _receipt() -> dict[str, object]:
    return S.build_receipt(ROOT)


def _by_id() -> dict[str, dict[str, object]]:
    return {result["problem_id"]: result for result in _receipt()["results"]}


def test_three_target_blind_matrix_and_nonlinear_controls_pass() -> None:
    receipt = _receipt()
    assert receipt["summary"] == {
        "algebraically_independent_coordinates": 4,
        "controls": 3,
        "deployment_failures": 0,
        "matrix_action_controls": 2,
        "nonlinear_action_controls": 1,
        "status": "PASS_EXACT_MATRIX_AND_NONLINEAR_STATE_PAIR_CONTROLS",
        "target_blind_controls": 3,
        "training_linear_invariant_coordinates": 5,
    }
    assert all(result["status"] == S.PASS_STATUS for result in receipt["results"])
    assert all(
        result["target_access"]["target_visible_to_learner"] is False
        for result in receipt["results"]
    )


def test_orthogonal_pairs_learn_squared_radius_without_action_matrix() -> None:
    result = _by_id()["control.orthogonal-plane-state-pairs"]
    coordinates = [
        row["expression"]
        for row in result["training_coordinates"]
        if row["algebraically_independent"]
    ]
    assert coordinates == ["x**2 + y**2"]
    assert result["search"]["training_rank"] == 4
    assert result["search"]["training_nullity"] == 1
    assert result["deployment"]["coordinate_failures"] == 0


def test_matrix_conjugation_pairs_learn_trace_and_determinant_coordinates() -> None:
    result = _by_id()["control.matrix-conjugation-2x2"]
    assert [row["expression"] for row in result["training_coordinates"]] == [
        "a + d",
        "a*d - b*c",
        "a**2 + 2*a*d + d**2",
    ]
    assert [
        row["expression"]
        for row in result["training_coordinates"]
        if row["algebraically_independent"]
    ] == ["a + d", "a*d - b*c"]
    assert result["search"]["training_nullity"] == 3
    assert result["search"]["training_algebraically_independent_coordinates"] == 2
    assert result["independent_evaluators"]["agreement"] is True
    assert result["deployment"]["coordinate_failures"] == 0


def test_nonlinear_pairs_learn_parabolic_coordinate() -> None:
    result = _by_id()["control.nonlinear-parabolic-shear"]
    assert [row["expression"] for row in result["training_coordinates"]] == ["y - x**2"]
    assert result["action_kind"] == "nonlinear_polynomial"
    assert result["deployment"]["coordinate_failures"] == 0


def test_hidden_deployment_mutation_rejects_but_retains_training_coordinate() -> None:
    config, _ = S.load_config(ROOT)
    problem = copy.deepcopy(
        next(
            problem
            for problem in config["problems"]
            if problem["problem_id"] == "control.nonlinear-parabolic-shear"
        )
    )
    problem["deployment_pairs"][0]["after"]["y"] = "0"
    changed = S.learn_problem(problem, config["policy"])
    assert changed["status"] == S.REJECT_STATUS
    assert changed["deployment"]["coordinate_failures"] == 1
    assert changed["training_coordinates"][0]["expression"] == "y - x**2"


def test_sealed_target_changes_cannot_change_target_blind_learning() -> None:
    config, targets = S.load_config(ROOT)
    problem = config["problems"][1]
    learned_before = S.learn_problem(problem, config["policy"])
    changed_targets = copy.deepcopy(targets)
    changed_targets["controls"][1]["expected_invariant_subspace_basis"] = [
        [1] + [0] * 13,
        [0, 1] + [0] * 12,
        [0, 0, 1] + [0] * 11,
    ]
    assert changed_targets != targets
    assert S.learn_problem(problem, config["policy"]) == learned_before
    assert learned_before["target_access"]["target_visible_to_learner"] is False


def test_resealed_receipt_cannot_promote_pair_invariance_to_a_theorem() -> None:
    changed = copy.deepcopy(_receipt())
    changed["claims"]["theorem_proved"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(S.StatePairInvariantError, match="claim boundary"):
        S.validate_receipt(changed, ROOT)


def test_stored_receipt_reproduces_current_public_and_sealed_sources() -> None:
    stored = json.loads((ROOT / S.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert S.validate_receipt(stored, ROOT) == stored
