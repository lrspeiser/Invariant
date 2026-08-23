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


def test_six_target_blind_typed_state_pair_controls_pass() -> None:
    receipt = _receipt()
    assert receipt["summary"] == {
        "algebraically_independent_coordinates": 7,
        "controls": 6,
        "deployment_failures": 0,
        "feature_grammar_kinds": [
            "laurent_monomials",
            "logarithmic_coordinates",
            "polynomial_monomials",
        ],
        "higher_degree_controls": 1,
        "matrix_action_controls": 2,
        "nonlinear_action_controls": 2,
        "rational_action_controls": 1,
        "status": "PASS_EXACT_TYPED_STATE_PAIR_INVARIANT_CONTROLS",
        "target_blind_controls": 6,
        "training_linear_invariant_coordinates": 9,
        "transcendental_action_controls": 1,
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


def test_degree_three_pairs_learn_cubic_coordinate_without_target_coefficients() -> None:
    result = _by_id()["control.cubic-shear-state-pairs"]
    assert [row["expression"] for row in result["training_coordinates"]] == ["y - x**3"]
    assert result["search"]["feature_grammar"] == {
        "kind": "polynomial_monomials",
        "maximum_total_degree": 3,
    }
    assert result["search"]["features"] == 9
    assert result["search"]["training_rank"] == 8
    assert result["target_access"]["target_visible_to_learner"] is False


def test_laurent_pairs_learn_rational_basis_and_remove_algebraic_duplicate() -> None:
    result = _by_id()["control.laurent-inversion-state-pairs"]
    assert [row["expression"] for row in result["training_coordinates"]] == [
        "x + 1/x",
        "x**2 + 1/x**2",
    ]
    assert [
        row["expression"]
        for row in result["training_coordinates"]
        if row["algebraically_independent"]
    ] == ["x + 1/x"]
    assert result["search"]["training_nullity"] == 2
    assert result["deployment"]["coordinate_failures"] == 0


def test_laurent_domain_guard_rejects_zero_before_division() -> None:
    config, _ = S.load_config(ROOT)
    problem = copy.deepcopy(config["problems"][4])
    problem["deployment_pairs"][0]["before"]["x"] = "0"
    with pytest.raises(S.StatePairInvariantError, match="division by zero"):
        S.learn_problem(problem, config["policy"])


def test_formal_log_pairs_use_two_prime_valuation_evaluators() -> None:
    result = _by_id()["control.logarithmic-scaling-state-pairs"]
    assert [row["expression"] for row in result["training_coordinates"]] == [
        "2*log(x) - log(y)"
    ]
    assert result["independent_evaluators"]["training_feature_evaluation"] == {
        "agreement": True,
        "constraint_rows": 4,
        "evaluator_pair": ["fraction_trial_division", "sympy_factorint"],
        "prime_support": [2, 3, 5, 7],
    }
    assert result["independent_evaluators"]["deployment_feature_evaluation"][
        "prime_support"
    ] == [11, 13]
    assert result["deployment"]["replays"][0]["differences"] == [{}, {}]


def test_logarithmic_domain_guard_rejects_nonpositive_state() -> None:
    config, _ = S.load_config(ROOT)
    problem = copy.deepcopy(config["problems"][5])
    problem["deployment_pairs"][0]["after"]["x"] = "-77"
    with pytest.raises(S.StatePairInvariantError, match="positive domain"):
        S.learn_problem(problem, config["policy"])


def test_formal_log_evaluator_disagreement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = S.load_config(ROOT)
    problem = config["problems"][5]
    monkeypatch.setattr(S, "_sympy_prime_valuations", lambda _value: {})
    with pytest.raises(S.StatePairInvariantError, match="formal-log evaluators disagree"):
        S.learn_problem(problem, config["policy"])


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
