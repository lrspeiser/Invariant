"""Learn exact polynomial invariants from target-blind before/after state pairs.

The learner receives rational state pairs and a bounded polynomial degree.  It does not receive
transformation matrices, group parameters, named formulas, or target coefficients.  It constructs
the monomial difference matrix, solves its nullspace independently with Fraction elimination and
SymPy, replays every learned coordinate on held-out pairs, and retains both failed training
coordinates and a separately repaired deployment basis when the action changes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from sympy import Matrix

from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/state_pair_invariant_discovery.json"
TARGETS_PATH = "configs/state_pair_invariant_discovery_targets.json"
OUTPUT_PATH = "runs/math/state-pair-invariant-discovery/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/state_pair_invariant_discovery.py"
TEST_PATH = "tests/test_state_pair_invariant_discovery.py"
CONFIG_SCHEMA = "invariant-state-pair-invariant-discovery-config-1.0"
TARGETS_SCHEMA = "invariant-state-pair-invariant-discovery-targets-1.0"
RECEIPT_SCHEMA = "invariant-state-pair-invariant-discovery-receipt-1.0"
CAMPAIGN_ID = "state-pair-invariant-controls-2026-08-23-001"
PASS_STATUS = "PASS_EXACT_STATE_PAIR_INVARIANT_BASIS"
REJECT_STATUS = "REJECT_TRAIN_ONLY_STATE_PAIR_INVARIANTS"
UNDERDETERMINED_STATUS = "UNDERDETERMINED_STATE_PAIR_INVARIANT_SPACE"
_EXPECTED_PROBLEMS = {
    "control.orthogonal-plane-state-pairs": ("geometry", "matrix_orthogonal"),
    "control.matrix-conjugation-2x2": ("linear_algebra", "matrix_conjugation"),
    "control.nonlinear-parabolic-shear": ("nonlinear_geometry", "nonlinear_polynomial"),
}
_ORIGIN_LABELS = [
    "known_rewrite",
    "cross_domain_synthesis",
    "proposed_new_construction",
    "uncertain",
]


class StatePairInvariantError(ValueError):
    """A state-pair declaration, inference, replay, seal, or claim boundary failed."""


def _strict(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StatePairInvariantError(f"{label} keys changed")
    return value


def _bound_path(root: Path, relative: str | Path, label: str) -> tuple[str, Path]:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as error:
        raise StatePairInvariantError(f"{label} escaped the repository root") from error
    return normalized, path


def _normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _rational(value: Any, label: str) -> Fraction:
    if not isinstance(value, str):
        raise StatePairInvariantError(f"{label} is not a rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise StatePairInvariantError(f"{label} is not rational") from error


def _integer_vector(value: Any, width: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise StatePairInvariantError(f"{label} is not an integer vector of length {width}")
    return tuple(value)


def _validate_pairs(value: Any, variables: Sequence[str], minimum: int, label: str) -> None:
    if not isinstance(value, list) or len(value) < minimum:
        raise StatePairInvariantError(f"{label} lacks state pairs")
    identifiers: set[str] = set()
    for pair in value:
        _strict(pair, {"after", "before", "pair_id"}, label)
        identifier = pair["pair_id"]
        before = pair["before"]
        after = pair["after"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or set(before) != set(variables)
            or set(after) != set(variables)
        ):
            raise StatePairInvariantError(f"{label} identity or state coverage changed")
        before_values = [_rational(before[name], f"{identifier} before {name}") for name in variables]
        after_values = [_rational(after[name], f"{identifier} after {name}") for name in variables]
        if before_values == after_values:
            raise StatePairInvariantError(f"{label} contains a trivial pair")
        identifiers.add(identifier)


def _monomials(width: int, maximum_total_degree: int) -> list[tuple[int, ...]]:
    values = range(maximum_total_degree + 1)
    monomials = [
        exponents
        for exponents in itertools.product(values, repeat=width)
        if 1 <= sum(exponents) <= maximum_total_degree
    ]
    return sorted(
        monomials,
        key=lambda item: (sum(item), tuple(-exponent for exponent in item)),
    )


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict(value, {"campaign_id", "policy", "problems", "schema_version"}, "state-pair config")
    if value["schema_version"] != CONFIG_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise StatePairInvariantError("state-pair config identity changed")
    policy = _strict(
        value["policy"],
        {
            "maximum_algebraic_coordinates",
            "maximum_linear_invariant_dimension",
            "maximum_monomials",
            "maximum_total_degree",
            "minimum_deployment_pairs",
            "minimum_training_pairs",
        },
        "state-pair policy",
    )
    if policy != {
        "maximum_algebraic_coordinates": 4,
        "maximum_linear_invariant_dimension": 4,
        "maximum_monomials": 14,
        "maximum_total_degree": 2,
        "minimum_deployment_pairs": 2,
        "minimum_training_pairs": 4,
    }:
        raise StatePairInvariantError("state-pair policy changed")
    problems = value["problems"]
    if not isinstance(problems, list) or len(problems) != len(_EXPECTED_PROBLEMS):
        raise StatePairInvariantError("state-pair problem coverage changed")
    observed: dict[str, tuple[str, str]] = {}
    for problem in problems:
        _strict(
            problem,
            {
                "action_kind",
                "deployment_pairs",
                "domain",
                "maximum_total_degree",
                "problem_id",
                "training_pairs",
                "variables",
            },
            "state-pair problem",
        )
        problem_id = problem["problem_id"]
        domain = problem["domain"]
        action_kind = problem["action_kind"]
        variables = problem["variables"]
        degree = problem["maximum_total_degree"]
        if (
            _EXPECTED_PROBLEMS.get(problem_id) != (domain, action_kind)
            or problem_id in observed
            or not isinstance(variables, list)
            or not 2 <= len(variables) <= 4
            or len(variables) != len(set(variables))
            or any(not isinstance(name, str) or not name for name in variables)
            or not isinstance(degree, int)
            or isinstance(degree, bool)
            or not 1 <= degree <= policy["maximum_total_degree"]
            or len(_monomials(len(variables), degree)) > policy["maximum_monomials"]
        ):
            raise StatePairInvariantError("state-pair problem identity or search space changed")
        _validate_pairs(
            problem["training_pairs"],
            variables,
            policy["minimum_training_pairs"],
            f"{problem_id} training",
        )
        _validate_pairs(
            problem["deployment_pairs"],
            variables,
            policy["minimum_deployment_pairs"],
            f"{problem_id} deployment",
        )
        observed[problem_id] = (domain, action_kind)
    if observed != _EXPECTED_PROBLEMS:
        raise StatePairInvariantError("state-pair problem set changed")
    return dict(value)


def validate_targets(value: Mapping[str, Any], widths: Mapping[str, int]) -> dict[str, Any]:
    _strict(value, {"campaign_id", "controls", "schema_version"}, "state-pair targets")
    if value["schema_version"] != TARGETS_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise StatePairInvariantError("state-pair target identity changed")
    controls = value["controls"]
    if not isinstance(controls, list) or len(controls) != len(widths):
        raise StatePairInvariantError("state-pair target coverage changed")
    observed: set[str] = set()
    for control in controls:
        _strict(
            control,
            {
                "expected_algebraically_independent_coordinates",
                "expected_augmented_nullity",
                "expected_augmented_rank",
                "expected_invariant_subspace_basis",
                "expected_status",
                "problem_id",
            },
            "state-pair target",
        )
        problem_id = control["problem_id"]
        width = widths.get(problem_id)
        basis = control["expected_invariant_subspace_basis"]
        nullity = control["expected_augmented_nullity"]
        if (
            width is None
            or problem_id in observed
            or control["expected_status"] != PASS_STATUS
            or not isinstance(control["expected_augmented_rank"], int)
            or not isinstance(nullity, int)
            or control["expected_augmented_rank"] + nullity != width
            or not isinstance(control["expected_algebraically_independent_coordinates"], int)
            or not 1 <= control["expected_algebraically_independent_coordinates"] <= nullity
            or not isinstance(basis, list)
            or len(basis) != nullity
        ):
            raise StatePairInvariantError("state-pair target shape changed")
        for index, vector in enumerate(basis):
            if not any(_integer_vector(vector, width, f"{problem_id} target {index}")):
                raise StatePairInvariantError("state-pair target contains a trivial vector")
        observed.add(problem_id)
    if observed != set(widths):
        raise StatePairInvariantError("state-pair target problem set changed")
    return dict(value)


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _, config_path = _bound_path(root, CONFIG_PATH, "state-pair config")
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    widths = {
        problem["problem_id"]: len(
            _monomials(len(problem["variables"]), problem["maximum_total_degree"])
        )
        for problem in config["problems"]
    }
    _, targets_path = _bound_path(root, TARGETS_PATH, "state-pair targets")
    targets = validate_targets(json.loads(targets_path.read_text(encoding="utf-8")), widths)
    return config, targets


def _canonical_integer_vector(values: Sequence[Any]) -> tuple[int, ...]:
    fractions = [Fraction(str(value)) for value in values]
    if not fractions or not any(fractions):
        raise StatePairInvariantError("polynomial coefficient vector is trivial")
    denominator_lcm = math.lcm(*(item.denominator for item in fractions))
    integers = [item.numerator * (denominator_lcm // item.denominator) for item in fractions]
    divisor = math.gcd(*(abs(item) for item in integers if item))
    integers = [item // divisor for item in integers]
    first = next(item for item in integers if item)
    if first < 0:
        integers = [-item for item in integers]
    return tuple(integers)


def _rref_fraction(
    rows: Sequence[Sequence[Fraction]], width: int
) -> tuple[list[list[Fraction]], list[int]]:
    if any(len(row) != width for row in rows):
        raise StatePairInvariantError("state-pair constraint matrix is ragged")
    matrix = [[Fraction(item) for item in row] for row in rows]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (index for index in range(pivot_row, len(matrix)) if matrix[index][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][column]
        matrix[pivot_row] = [item / pivot_value for item in matrix[pivot_row]]
        for row_index, row in enumerate(matrix):
            if row_index == pivot_row or not row[column]:
                continue
            factor = row[column]
            matrix[row_index] = [
                item - factor * pivot_item
                for item, pivot_item in zip(row, matrix[pivot_row], strict=True)
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return matrix, pivots


def _fraction_rank(rows: Sequence[Sequence[Fraction]], width: int) -> int:
    return len(_rref_fraction(rows, width)[1])


def _fraction_nullspace(
    rows: Sequence[Sequence[Fraction]], width: int
) -> list[tuple[int, ...]]:
    rref, pivots = _rref_fraction(rows, width)
    free_columns = [column for column in range(width) if column not in pivots]
    basis = []
    for free in free_columns:
        vector = [Fraction(0) for _ in range(width)]
        vector[free] = Fraction(1)
        for row_index, pivot in enumerate(pivots):
            vector[pivot] = -rref[row_index][free]
        basis.append(_canonical_integer_vector(vector))
    return basis


def _sympy_rank_and_basis(
    rows: Sequence[Sequence[Fraction]], width: int
) -> tuple[int, list[tuple[int, ...]]]:
    matrix = Matrix(rows) if rows else Matrix.zeros(0, width)
    basis = [_canonical_integer_vector(vector) for vector in matrix.nullspace()]
    return int(matrix.rank()), basis


def _span_rank(vectors: Sequence[Sequence[int]], width: int) -> int:
    return _fraction_rank([[Fraction(item) for item in vector] for vector in vectors], width)


def _same_span(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]], width: int
) -> bool:
    left_rank = _span_rank(left, width)
    right_rank = _span_rank(right, width)
    return left_rank == right_rank == _span_rank([*left, *right], width)


def _evaluate_monomial(state: Sequence[Fraction], exponents: Sequence[int]) -> Fraction:
    value = Fraction(1)
    for coordinate, exponent in zip(state, exponents, strict=True):
        value *= coordinate**exponent
    return value


def _state(pair: Mapping[str, Any], side: str, variables: Sequence[str]) -> list[Fraction]:
    return [
        _rational(pair[side][name], f"{pair['pair_id']} {side} {name}")
        for name in variables
    ]


def _constraint_rows(
    pairs: Sequence[Mapping[str, Any]],
    variables: Sequence[str],
    monomials: Sequence[Sequence[int]],
) -> list[list[Fraction]]:
    rows = []
    for pair in pairs:
        before = _state(pair, "before", variables)
        after = _state(pair, "after", variables)
        rows.append(
            [
                _evaluate_monomial(after, exponents)
                - _evaluate_monomial(before, exponents)
                for exponents in monomials
            ]
        )
    return rows


def _monomial_expression(variables: Sequence[str], exponents: Sequence[int]) -> str:
    factors = []
    for name, exponent in zip(variables, exponents, strict=True):
        if exponent:
            factors.append(name if exponent == 1 else f"{name}**{exponent}")
    return "*".join(factors)


def _polynomial_expression(
    variables: Sequence[str],
    monomials: Sequence[Sequence[int]],
    coefficients: Sequence[int],
) -> str:
    terms: list[tuple[int, str]] = []
    for coefficient, exponents in zip(coefficients, monomials, strict=True):
        if not coefficient:
            continue
        monomial = _monomial_expression(variables, exponents)
        magnitude = abs(coefficient)
        term = monomial if magnitude == 1 else f"{magnitude}*{monomial}"
        terms.append((1 if coefficient > 0 else -1, term))
    if not terms:
        raise StatePairInvariantError("polynomial expression is trivial")
    sign, first = terms[0]
    text = first if sign > 0 else f"-{first}"
    for sign, term in terms[1:]:
        text += f" {'+' if sign > 0 else '-'} {term}"
    return text


def _basis_sort_key(
    vector: Sequence[int], monomials: Sequence[Sequence[int]]
) -> tuple[Any, ...]:
    active = [
        (coefficient, exponents)
        for coefficient, exponents in zip(vector, monomials, strict=True)
        if coefficient
    ]
    return (
        max(sum(exponents) for _, exponents in active),
        len(active),
        sum(abs(coefficient) for coefficient, _ in active),
        tuple(vector),
    )


def _gradient_at(
    coefficients: Sequence[int],
    monomials: Sequence[Sequence[int]],
    point: Sequence[Fraction],
) -> list[Fraction]:
    gradient = []
    for variable_index in range(len(point)):
        value = Fraction(0)
        for coefficient, exponents in zip(coefficients, monomials, strict=True):
            exponent = exponents[variable_index]
            if not coefficient or not exponent:
                continue
            term = Fraction(coefficient * exponent)
            for index, (coordinate, power) in enumerate(zip(point, exponents, strict=True)):
                term *= coordinate ** (power - (index == variable_index))
            value += term
        gradient.append(value)
    return gradient


def _select_algebraically_independent(
    basis: Sequence[tuple[int, ...]],
    monomials: Sequence[Sequence[int]],
    variable_count: int,
    maximum: int,
) -> list[tuple[int, ...]]:
    witnesses = [
        [Fraction((seed + 2) * (index + 2) + index) for index in range(variable_count)]
        for seed in range(8)
    ]
    selected: list[tuple[int, ...]] = []
    for candidate in basis:
        trial = [*selected, candidate]
        if any(
            _fraction_rank(
                [_gradient_at(vector, monomials, witness) for vector in trial],
                variable_count,
            )
            == len(trial)
            for witness in witnesses
        ):
            selected.append(candidate)
        if len(selected) == maximum:
            break
    return selected


def _coordinate_rows(
    variables: Sequence[str],
    monomials: Sequence[Sequence[int]],
    basis: Sequence[tuple[int, ...]],
    algebraically_independent: Sequence[tuple[int, ...]],
) -> list[dict[str, Any]]:
    independent = set(algebraically_independent)
    return [
        {
            "algebraically_independent": vector in independent,
            "coefficient_vector": list(vector),
            "coordinate_id": f"phi_{index}",
            "expression": _polynomial_expression(variables, monomials, vector),
        }
        for index, vector in enumerate(basis, start=1)
    ]


def _evaluate_polynomial(
    coefficients: Sequence[int],
    monomials: Sequence[Sequence[int]],
    state: Sequence[Fraction],
) -> Fraction:
    return sum(
        Fraction(coefficient) * _evaluate_monomial(state, exponents)
        for coefficient, exponents in zip(coefficients, monomials, strict=True)
    )


def _replay_basis(
    pairs: Sequence[Mapping[str, Any]],
    variables: Sequence[str],
    monomials: Sequence[Sequence[int]],
    basis: Sequence[tuple[int, ...]],
) -> tuple[list[dict[str, Any]], int]:
    replays = []
    failures = 0
    for index, vector in enumerate(basis, start=1):
        differences = []
        for pair in pairs:
            before = _state(pair, "before", variables)
            after = _state(pair, "after", variables)
            differences.append(
                _evaluate_polynomial(vector, monomials, after)
                - _evaluate_polynomial(vector, monomials, before)
            )
        passes = all(not value for value in differences)
        failures += not passes
        replays.append(
            {
                "coordinate_id": f"phi_{index}",
                "differences": [str(value) for value in differences],
                "invariant_on_all_deployment_pairs": passes,
            }
        )
    return replays, failures


def _learn_basis(
    rows: Sequence[Sequence[Fraction]],
    monomials: Sequence[Sequence[int]],
) -> tuple[int, list[tuple[int, ...]], dict[str, Any]]:
    width = len(monomials)
    fraction_rank = _fraction_rank(rows, width)
    fraction_basis = _fraction_nullspace(rows, width)
    sympy_rank, sympy_basis = _sympy_rank_and_basis(rows, width)
    if fraction_rank != sympy_rank or not _same_span(fraction_basis, sympy_basis, width):
        raise StatePairInvariantError("independent state-pair nullspace evaluators disagree")
    basis = sorted(fraction_basis, key=lambda vector: _basis_sort_key(vector, monomials))
    return fraction_rank, basis, {
        "agreement": True,
        "fraction_nullspace_basis": [list(vector) for vector in fraction_basis],
        "fraction_rank": fraction_rank,
        "sympy_nullspace_basis": [list(vector) for vector in sympy_basis],
        "sympy_rank": sympy_rank,
    }


def learn_problem(problem: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Infer polynomial invariant coordinates without accepting a sealed target."""

    variables = list(problem["variables"])
    monomials = _monomials(len(variables), problem["maximum_total_degree"])
    training_pairs = problem["training_pairs"]
    deployment_pairs = problem["deployment_pairs"]
    training_rows = _constraint_rows(training_pairs, variables, monomials)
    deployment_rows = _constraint_rows(deployment_pairs, variables, monomials)
    augmented_rows = [*training_rows, *deployment_rows]

    training_rank, training_basis, training_evaluator = _learn_basis(training_rows, monomials)
    augmented_rank, repaired_basis, augmented_evaluator = _learn_basis(augmented_rows, monomials)
    training_independent = _select_algebraically_independent(
        training_basis,
        monomials,
        len(variables),
        policy["maximum_algebraic_coordinates"],
    )
    repaired_independent = _select_algebraically_independent(
        repaired_basis,
        monomials,
        len(variables),
        policy["maximum_algebraic_coordinates"],
    )
    training_coordinates = _coordinate_rows(
        variables, monomials, training_basis, training_independent
    )
    repaired_coordinates = _coordinate_rows(
        variables, monomials, repaired_basis, repaired_independent
    )
    deployment_replays, deployment_failures = _replay_basis(
        deployment_pairs, variables, monomials, training_basis
    )
    training_nullity = len(monomials) - training_rank
    augmented_nullity = len(monomials) - augmented_rank
    if training_nullity > policy["maximum_linear_invariant_dimension"]:
        status = UNDERDETERMINED_STATUS
    elif deployment_failures or augmented_nullity != training_nullity:
        status = REJECT_STATUS
    else:
        status = PASS_STATUS

    return {
        "action_kind": problem["action_kind"],
        "creative_brief": {
            "action_kind": problem["action_kind"],
            "candidate_invariant_coordinates": [
                row["expression"] for row in training_coordinates if row["algebraically_independent"]
            ],
            "constraint_statement": (
                "Treat the algebraically independent coordinates as first-principles building "
                "blocks. Retain the full linear invariant basis, deployment failures, and repaired "
                "basis as distinct branches; exact pair invariance is not a truth or novelty claim."
            ),
            "deployment_repaired_coordinates": [
                row["expression"] for row in repaired_coordinates if row["algebraically_independent"]
            ],
            "identifiability_status": status,
            "llm_origin_assessment_labels": list(_ORIGIN_LABELS),
            "novelty_caution": (
                "These are known-answer calibration actions. A learned invariant basis establishes "
                "neither a new theorem nor literature novelty."
            ),
            "retained_linear_invariant_basis": [
                row["expression"] for row in training_coordinates
            ],
        },
        "deployment": {
            "coordinate_failures": deployment_failures,
            "pairs": len(deployment_pairs),
            "replays": deployment_replays,
        },
        "deployment_repaired_coordinates": repaired_coordinates,
        "domain": problem["domain"],
        "independent_evaluators": {
            "agreement": True,
            "augmented": augmented_evaluator,
            "training": training_evaluator,
        },
        "monomial_basis": [
            {
                "exponents": list(exponents),
                "expression": _monomial_expression(variables, exponents),
            }
            for exponents in monomials
        ],
        "problem_id": problem["problem_id"],
        "search": {
            "augmented_nullity": augmented_nullity,
            "augmented_rank": augmented_rank,
            "maximum_total_degree": problem["maximum_total_degree"],
            "monomials": len(monomials),
            "training_algebraically_independent_coordinates": len(training_independent),
            "training_nullity": training_nullity,
            "training_rank": training_rank,
        },
        "status": status,
        "target_access": {
            "learner_input_sha256": canonical_sha256(problem),
            "target_visible_to_learner": False,
        },
        "training_coordinates": training_coordinates,
        "variables": variables,
    }


def _evaluate_target(
    result: Mapping[str, Any],
    target: Mapping[str, Any],
    augmented_rows: Sequence[Sequence[Fraction]],
) -> dict[str, Any]:
    width = len(result["monomial_basis"])
    expected_basis = [
        _integer_vector(vector, width, f"{result['problem_id']} sealed basis")
        for vector in target["expected_invariant_subspace_basis"]
    ]
    expected_nullity = target["expected_augmented_nullity"]
    subspace_matches = (
        _span_rank(expected_basis, width) == expected_nullity
        and all(
            not sum(Fraction(a) * b for a, b in zip(row, vector, strict=True))
            for vector in expected_basis
            for row in augmented_rows
        )
        and result["search"]["augmented_nullity"] == expected_nullity
    )
    if (
        result["status"] != target["expected_status"]
        or result["search"]["augmented_rank"] != target["expected_augmented_rank"]
        or result["search"]["training_algebraically_independent_coordinates"]
        != target["expected_algebraically_independent_coordinates"]
        or not subspace_matches
    ):
        raise StatePairInvariantError(f"{result['problem_id']} sealed control failed")
    return {
        "expected_algebraic_coordinate_count_matched": True,
        "expected_status_matched": True,
        "sealed_subspace_matched": True,
        "target_visible_to_learner": False,
    }


def _source_binding(root: Path, relative: str) -> dict[str, str]:
    path_text, path = _bound_path(root, relative, "state-pair source binding")
    if not path.is_file():
        raise StatePairInvariantError(f"state-pair source binding is missing: {path_text}")
    return {"normalized_sha256": _normalized_sha256(path), "path": path_text}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config, targets = load_config(root)
    target_by_problem = {control["problem_id"]: control for control in targets["controls"]}
    results = []
    for problem in config["problems"]:
        learned = learn_problem(problem, config["policy"])
        monomials = [tuple(item["exponents"]) for item in learned["monomial_basis"]]
        augmented_rows = _constraint_rows(
            [*problem["training_pairs"], *problem["deployment_pairs"]],
            problem["variables"],
            monomials,
        )
        learned["sealed_control_evaluation"] = _evaluate_target(
            learned,
            target_by_problem[problem["problem_id"]],
            augmented_rows,
        )
        results.append(learned)
    body: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "claims": {
            "action_generator_recovered": False,
            "empirical_law_discovered": False,
            "literature_novelty_established": False,
            "named_formula_recovered": False,
            "target_used_for_learning": False,
            "theorem_proved": False,
        },
        "release_gate": {
            "creative_context_ready": True,
            "serious_claim_released": False,
            "status": "PASS_STATE_PAIR_CONTROLS_NO_THEOREM_OR_NOVELTY_CLAIM",
        },
        "results": results,
        "schema_version": RECEIPT_SCHEMA,
        "source_bindings": {
            "config": _source_binding(root, CONFIG_PATH),
            "source": _source_binding(root, SOURCE_PATH),
            "targets": _source_binding(root, TARGETS_PATH),
            "test": _source_binding(root, TEST_PATH),
        },
        "summary": {
            "algebraically_independent_coordinates": sum(
                result["search"]["training_algebraically_independent_coordinates"]
                for result in results
            ),
            "controls": len(results),
            "deployment_failures": sum(
                result["deployment"]["coordinate_failures"] for result in results
            ),
            "matrix_action_controls": sum(
                result["action_kind"].startswith("matrix_") for result in results
            ),
            "nonlinear_action_controls": sum(
                result["action_kind"] == "nonlinear_polynomial" for result in results
            ),
            "status": "PASS_EXACT_MATRIX_AND_NONLINEAR_STATE_PAIR_CONTROLS",
            "target_blind_controls": sum(
                result["target_access"]["target_visible_to_learner"] is False
                for result in results
            ),
            "training_linear_invariant_coordinates": sum(
                len(result["training_coordinates"]) for result in results
            ),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _strict(
        receipt,
        {
            "campaign_id",
            "claims",
            "content_sha256",
            "release_gate",
            "results",
            "schema_version",
            "source_bindings",
            "summary",
        },
        "state-pair receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA or receipt["campaign_id"] != CAMPAIGN_ID:
        raise StatePairInvariantError("state-pair receipt identity changed")
    claims = _strict(
        receipt["claims"],
        {
            "action_generator_recovered",
            "empirical_law_discovered",
            "literature_novelty_established",
            "named_formula_recovered",
            "target_used_for_learning",
            "theorem_proved",
        },
        "state-pair claims",
    )
    if any(claims.values()):
        raise StatePairInvariantError("state-pair claim boundary changed")
    expected = build_receipt(root)
    if receipt != expected:
        raise StatePairInvariantError("state-pair receipt no longer reproduces")
    return dict(receipt)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", type=Path, default=Path.cwd())
        command_parser.add_argument(
            "--output" if command == "build" else "--receipt",
            type=Path,
            default=Path(OUTPUT_PATH),
        )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "build":
        receipt = build_receipt(root)
        _, output = _bound_path(root, args.output, "state-pair receipt output")
        _write_json(output, receipt)
    else:
        _, receipt_path = _bound_path(root, args.receipt, "state-pair receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "controls": receipt["summary"]["controls"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
