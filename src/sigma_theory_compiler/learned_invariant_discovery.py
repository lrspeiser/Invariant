"""Learn exact multiplicative invariant coordinates from paired transformations.

The learner sees positive rational before/after ratios only.  It does not receive dimension labels,
action weights, target exponents, or named formulas.  Prime valuations turn the observed ratios into
an exact integer action matrix; bounded primitive nullspace coordinates are then learned on the
training split and replayed on deployment transformations.  Failed coordinates remain visible
beside a repaired deployment basis, and an oversized nullspace is reported as underdetermined.
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

CONFIG_PATH = "configs/learned_invariant_discovery.json"
TARGETS_PATH = "configs/learned_invariant_discovery_targets.json"
OUTPUT_PATH = "runs/math/learned-invariant-discovery/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/learned_invariant_discovery.py"
TEST_PATH = "tests/test_learned_invariant_discovery.py"
CONFIG_SCHEMA = "invariant-learned-invariant-discovery-config-1.0"
TARGETS_SCHEMA = "invariant-learned-invariant-discovery-targets-1.0"
RECEIPT_SCHEMA = "invariant-learned-invariant-discovery-receipt-1.0"
CAMPAIGN_ID = "learned-invariant-controls-2026-08-23-001"
_EXPECTED_PROBLEMS = {
    "control.learned-drag-multi-coordinate": "aerodynamics",
    "control.shifted-hidden-action": "synthetic_shift",
    "control.underidentified-action": "synthetic_identifiability",
}
_EXPECTED_STATUSES = {
    "PASS_LEARNED_INVARIANT_BASIS",
    "REJECT_TRAIN_ONLY_INVARIANT_SPACE",
    "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE",
}
_ORIGIN_LABELS = [
    "known_rewrite",
    "cross_domain_synthesis",
    "proposed_new_construction",
    "uncertain",
]


class LearnedInvariantError(ValueError):
    """A learned-action declaration, inference, holdout, seal, or claim boundary failed."""


def _strict(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LearnedInvariantError(f"{label} keys changed")
    return value


def _bound_path(root: Path, relative: str | Path, label: str) -> tuple[str, Path]:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as error:
        raise LearnedInvariantError(f"{label} escaped the repository root") from error
    return normalized, path


def _normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _positive_fraction(value: Any, label: str) -> Fraction:
    if not isinstance(value, str):
        raise LearnedInvariantError(f"{label} is not a rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise LearnedInvariantError(f"{label} is not rational") from error
    if parsed <= 0:
        raise LearnedInvariantError(f"{label} must be positive")
    return parsed


def _integer_vector(value: Any, length: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise LearnedInvariantError(f"{label} is not an integer vector of length {length}")
    return tuple(value)


def _validate_transformations(
    value: Any,
    variables: Sequence[str],
    minimum: int,
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) < minimum:
        raise LearnedInvariantError(f"{label} lacks transformations")
    identifiers: set[str] = set()
    for transformation in value:
        _strict(transformation, {"ratios", "transformation_id"}, label)
        identifier = transformation["transformation_id"]
        ratios = transformation["ratios"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(ratios, Mapping)
            or set(ratios) != set(variables)
        ):
            raise LearnedInvariantError(f"{label} identity or ratio coverage changed")
        parsed = [_positive_fraction(ratios[name], f"{identifier} {name} ratio") for name in variables]
        if all(item == 1 for item in parsed):
            raise LearnedInvariantError(f"{label} contains a trivial transformation")
        identifiers.add(identifier)


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict(value, {"campaign_id", "policy", "problems", "schema_version"}, "learner config")
    if value["schema_version"] != CONFIG_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise LearnedInvariantError("learner config identity changed")
    policy = _strict(
        value["policy"],
        {
            "maximum_absolute_exponent",
            "maximum_identifiable_nullity",
            "maximum_l1_norm",
            "minimum_deployment_transformations",
            "minimum_training_transformations",
        },
        "learner policy",
    )
    if policy != {
        "maximum_absolute_exponent": 3,
        "maximum_identifiable_nullity": 2,
        "maximum_l1_norm": 8,
        "minimum_deployment_transformations": 1,
        "minimum_training_transformations": 1,
    }:
        raise LearnedInvariantError("learner policy changed")
    problems = value["problems"]
    if not isinstance(problems, list) or len(problems) != len(_EXPECTED_PROBLEMS):
        raise LearnedInvariantError("learner problem coverage changed")
    observed: dict[str, str] = {}
    for problem in problems:
        _strict(
            problem,
            {
                "deployment_transformations",
                "domain",
                "problem_id",
                "training_transformations",
                "variables",
            },
            "learner problem",
        )
        problem_id = problem["problem_id"]
        domain = problem["domain"]
        variables = problem["variables"]
        if (
            _EXPECTED_PROBLEMS.get(problem_id) != domain
            or problem_id in observed
            or not isinstance(variables, list)
            or not 4 <= len(variables) <= 6
            or len(variables) != len(set(variables))
            or any(not isinstance(name, str) or not name for name in variables)
        ):
            raise LearnedInvariantError("learner problem identity or variables changed")
        _validate_transformations(
            problem["training_transformations"],
            variables,
            policy["minimum_training_transformations"],
            f"{problem_id} training",
        )
        _validate_transformations(
            problem["deployment_transformations"],
            variables,
            policy["minimum_deployment_transformations"],
            f"{problem_id} deployment",
        )
        observed[problem_id] = domain
    if observed != _EXPECTED_PROBLEMS:
        raise LearnedInvariantError("learner problem set changed")
    return dict(value)


def validate_targets(value: Mapping[str, Any], problems: Mapping[str, int]) -> dict[str, Any]:
    _strict(value, {"campaign_id", "controls", "schema_version"}, "learner targets")
    if value["schema_version"] != TARGETS_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise LearnedInvariantError("learner target identity changed")
    controls = value["controls"]
    if not isinstance(controls, list) or len(controls) != len(problems):
        raise LearnedInvariantError("learner target coverage changed")
    observed: set[str] = set()
    for control in controls:
        _strict(
            control,
            {
                "expected_augmented_nullity",
                "expected_augmented_rank",
                "expected_invariant_subspace_basis",
                "expected_status",
                "problem_id",
            },
            "learner target",
        )
        problem_id = control["problem_id"]
        width = problems.get(problem_id)
        basis = control["expected_invariant_subspace_basis"]
        if (
            width is None
            or problem_id in observed
            or control["expected_status"] not in _EXPECTED_STATUSES
            or not isinstance(control["expected_augmented_rank"], int)
            or not isinstance(control["expected_augmented_nullity"], int)
            or control["expected_augmented_rank"] + control["expected_augmented_nullity"] != width
            or not isinstance(basis, list)
            or len(basis) != control["expected_augmented_nullity"]
        ):
            raise LearnedInvariantError("learner target shape changed")
        for index, vector in enumerate(basis):
            if not any(_integer_vector(vector, width, f"{problem_id} target {index}")):
                raise LearnedInvariantError("learner target contains a trivial vector")
        observed.add(problem_id)
    if observed != set(problems):
        raise LearnedInvariantError("learner target problem set changed")
    return dict(value)


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _, config_path = _bound_path(root, CONFIG_PATH, "learner config")
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    widths = {problem["problem_id"]: len(problem["variables"]) for problem in config["problems"]}
    _, targets_path = _bound_path(root, TARGETS_PATH, "learner targets")
    targets = validate_targets(json.loads(targets_path.read_text(encoding="utf-8")), widths)
    return config, targets


def _factor_integer(value: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            value //= divisor
        divisor += 1 if divisor == 2 else 2
    if value > 1:
        factors[value] = factors.get(value, 0) + 1
    return factors


def _valuations(value: Fraction) -> dict[int, int]:
    result = _factor_integer(value.numerator)
    for prime, exponent in _factor_integer(value.denominator).items():
        result[prime] = result.get(prime, 0) - exponent
    return {prime: exponent for prime, exponent in result.items() if exponent}


def _canonical_integer_vector(values: Sequence[Fraction | int]) -> tuple[int, ...]:
    fractions = [Fraction(value) for value in values]
    if not fractions or not any(fractions):
        raise LearnedInvariantError("integer vector is trivial")
    denominator_lcm = math.lcm(*(item.denominator for item in fractions))
    integers = [item.numerator * (denominator_lcm // item.denominator) for item in fractions]
    divisor = math.gcd(*(abs(item) for item in integers if item))
    integers = [item // divisor for item in integers]
    first = next(item for item in integers if item)
    if first < 0:
        integers = [-item for item in integers]
    return tuple(integers)


def _constraint_rows(
    transformations: Sequence[Mapping[str, Any]], variables: Sequence[str]
) -> list[tuple[int, ...]]:
    rows: set[tuple[int, ...]] = set()
    for transformation in transformations:
        valuations = {
            name: _valuations(
                _positive_fraction(
                    transformation["ratios"][name],
                    f"{transformation['transformation_id']} {name} ratio",
                )
            )
            for name in variables
        }
        primes = sorted({prime for item in valuations.values() for prime in item})
        for prime in primes:
            row = tuple(valuations[name].get(prime, 0) for name in variables)
            if any(row):
                rows.add(_canonical_integer_vector(row))
    return sorted(rows)


def _rank_fraction(rows: Sequence[Sequence[int]]) -> int:
    if not rows:
        return 0
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise LearnedInvariantError("rank matrix is empty or ragged")
    matrix = [[Fraction(item) for item in row] for row in rows]
    rank = 0
    for column in range(width):
        pivot = next(
            (index for index in range(rank, len(matrix)) if matrix[index][column]), None
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [item / pivot_value for item in matrix[rank]]
        for row_index, row in enumerate(matrix):
            if row_index == rank or not row[column]:
                continue
            factor = row[column]
            matrix[row_index] = [
                item - factor * pivot_item
                for item, pivot_item in zip(row, matrix[rank], strict=True)
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _sympy_rank_and_basis(rows: Sequence[Sequence[int]], width: int) -> tuple[int, list[tuple[int, ...]]]:
    matrix = Matrix(rows) if rows else Matrix.zeros(0, width)
    basis = [
        _canonical_integer_vector([Fraction(str(item)) for item in vector])
        for vector in matrix.nullspace()
    ]
    return int(matrix.rank()), basis


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _enumerate_invariants(
    rows: Sequence[Sequence[int]],
    width: int,
    maximum_absolute_exponent: int,
    maximum_l1_norm: int,
) -> list[tuple[int, ...]]:
    results: set[tuple[int, ...]] = set()
    values = range(-maximum_absolute_exponent, maximum_absolute_exponent + 1)
    for candidate in itertools.product(values, repeat=width):
        if not any(candidate) or sum(abs(item) for item in candidate) > maximum_l1_norm:
            continue
        if candidate != _canonical_integer_vector(candidate):
            continue
        if not any(_dot(row, candidate) for row in rows):
            results.add(candidate)
    return sorted(results, key=lambda item: (sum(abs(value) for value in item), item))


def _select_basis(candidates: Sequence[tuple[int, ...]], nullity: int) -> list[tuple[int, ...]]:
    selected: list[tuple[int, ...]] = []
    for candidate in candidates:
        if _rank_fraction([*selected, candidate]) > len(selected):
            selected.append(candidate)
        if len(selected) == nullity:
            break
    if len(selected) != nullity:
        raise LearnedInvariantError("bounded search did not span the learned nullspace")
    return selected


def _expression(names: Sequence[str], exponents: Sequence[int]) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for name, exponent in zip(names, exponents, strict=True):
        if not exponent:
            continue
        factor = name if abs(exponent) == 1 else f"{name}**{abs(exponent)}"
        (numerator if exponent > 0 else denominator).append(factor)
    numerator_text = "*".join(numerator) or "1"
    if not denominator:
        return numerator_text
    denominator_text = "*".join(denominator)
    if len(denominator) > 1:
        denominator_text = f"({denominator_text})"
    return f"{numerator_text}/{denominator_text}"


def _coordinate_rows(names: Sequence[str], basis: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    return [
        {
            "coordinate_id": f"pi_{index}",
            "exponents": list(vector),
            "expression": _expression(names, vector),
        }
        for index, vector in enumerate(basis, start=1)
    ]


def _replay_coordinate(
    transformation: Mapping[str, Any], names: Sequence[str], exponents: Sequence[int]
) -> Fraction:
    result = Fraction(1)
    for name, exponent in zip(names, exponents, strict=True):
        result *= _positive_fraction(
            transformation["ratios"][name],
            f"{transformation['transformation_id']} {name} ratio",
        ) ** exponent
    return result


def learn_problem(problem: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Infer a basis from public transformations only; no sealed target is accepted."""

    names = list(problem["variables"])
    training = problem["training_transformations"]
    deployment = problem["deployment_transformations"]
    training_rows = _constraint_rows(training, names)
    deployment_rows = _constraint_rows(deployment, names)
    augmented_rows = sorted(set(training_rows) | set(deployment_rows))

    training_fraction_rank = _rank_fraction(training_rows)
    training_sympy_rank, training_sympy_basis = _sympy_rank_and_basis(training_rows, len(names))
    augmented_fraction_rank = _rank_fraction(augmented_rows)
    augmented_sympy_rank, augmented_sympy_basis = _sympy_rank_and_basis(
        augmented_rows, len(names)
    )
    if (
        training_fraction_rank != training_sympy_rank
        or augmented_fraction_rank != augmented_sympy_rank
    ):
        raise LearnedInvariantError("independent learned-action rank evaluators disagree")
    training_nullity = len(names) - training_fraction_rank
    augmented_nullity = len(names) - augmented_fraction_rank
    if len(training_sympy_basis) != training_nullity or len(augmented_sympy_basis) != augmented_nullity:
        raise LearnedInvariantError("learned-action nullspace dimension changed")

    training_candidates = _enumerate_invariants(
        training_rows,
        len(names),
        policy["maximum_absolute_exponent"],
        policy["maximum_l1_norm"],
    )
    training_basis = _select_basis(training_candidates, training_nullity)
    augmented_candidates = _enumerate_invariants(
        augmented_rows,
        len(names),
        policy["maximum_absolute_exponent"],
        policy["maximum_l1_norm"],
    )
    repaired_basis = _select_basis(augmented_candidates, augmented_nullity)

    deployment_replays = []
    deployment_failures = 0
    for coordinate_index, vector in enumerate(training_basis, start=1):
        values = [
            _replay_coordinate(transformation, names, vector) for transformation in deployment
        ]
        passes = all(value == 1 for value in values)
        deployment_failures += not passes
        deployment_replays.append(
            {
                "coordinate_id": f"pi_{coordinate_index}",
                "invariant_on_all_deployment_transformations": passes,
                "replay_values": [str(value) for value in values],
            }
        )

    if training_nullity > policy["maximum_identifiable_nullity"]:
        status = "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE"
    elif deployment_failures or augmented_nullity != training_nullity:
        status = "REJECT_TRAIN_ONLY_INVARIANT_SPACE"
    else:
        status = "PASS_LEARNED_INVARIANT_BASIS"

    training_coordinates = _coordinate_rows(names, training_basis)
    repaired_coordinates = _coordinate_rows(names, repaired_basis)
    return {
        "creative_brief": {
            "candidate_invariant_coordinates": [
                item["expression"] for item in training_coordinates
            ],
            "constraint_statement": (
                "Treat training coordinates as retained hypotheses. Keep deployment failures as "
                "repair branches and the repaired coordinates as distinct proposals; an "
                "underdetermined subspace requires more transformations, not forced selection."
            ),
            "deployment_repaired_coordinates": [
                item["expression"] for item in repaired_coordinates
            ],
            "identifiability_status": status,
            "llm_origin_assessment_labels": list(_ORIGIN_LABELS),
            "novelty_caution": (
                "Exact transformation invariance is neither a discovered scientific law nor "
                "evidence of literature novelty."
            ),
        },
        "deployment": {
            "coordinate_failures": deployment_failures,
            "replays": deployment_replays,
            "transformations": len(deployment),
        },
        "domain": problem["domain"],
        "independent_evaluators": {
            "agreement": True,
            "augmented_fraction_rank": augmented_fraction_rank,
            "augmented_sympy_nullspace_basis": [list(item) for item in augmented_sympy_basis],
            "augmented_sympy_rank": augmented_sympy_rank,
            "training_fraction_rank": training_fraction_rank,
            "training_sympy_nullspace_basis": [list(item) for item in training_sympy_basis],
            "training_sympy_rank": training_sympy_rank,
        },
        "learned_action_rows": {
            "augmented": [list(row) for row in augmented_rows],
            "deployment": [list(row) for row in deployment_rows],
            "training": [list(row) for row in training_rows],
        },
        "problem_id": problem["problem_id"],
        "search": {
            "augmented_bounded_primitive_invariants": len(augmented_candidates),
            "augmented_nullity": augmented_nullity,
            "maximum_identifiable_nullity": policy["maximum_identifiable_nullity"],
            "training_bounded_primitive_invariants": len(training_candidates),
            "training_nullity": training_nullity,
        },
        "status": status,
        "target_access": {
            "learner_input_sha256": canonical_sha256(problem),
            "target_visible_to_learner": False,
        },
        "training_coordinates": training_coordinates,
        "deployment_repaired_coordinates": repaired_coordinates,
        "variables": names,
    }


def _evaluate_target(result: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    width = len(result["variables"])
    expected_basis = [
        _integer_vector(vector, width, f"{result['problem_id']} sealed basis")
        for vector in target["expected_invariant_subspace_basis"]
    ]
    augmented_rows = result["learned_action_rows"]["augmented"]
    expected_rank = _rank_fraction(expected_basis)
    expected_nullity = target["expected_augmented_nullity"]
    subspace_matches = (
        expected_rank == expected_nullity
        and all(not _dot(row, vector) for vector in expected_basis for row in augmented_rows)
        and result["search"]["augmented_nullity"] == expected_nullity
    )
    if (
        result["status"] != target["expected_status"]
        or result["independent_evaluators"]["augmented_fraction_rank"]
        != target["expected_augmented_rank"]
        or not subspace_matches
    ):
        raise LearnedInvariantError(f"{result['problem_id']} sealed control failed")
    return {
        "expected_status_matched": True,
        "sealed_subspace_matched": True,
        "target_visible_to_learner": False,
    }


def _source_binding(root: Path, relative: str) -> dict[str, str]:
    path_text, path = _bound_path(root, relative, "learner source binding")
    if not path.is_file():
        raise LearnedInvariantError(f"learner source binding is missing: {path_text}")
    return {"normalized_sha256": _normalized_sha256(path), "path": path_text}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config, targets = load_config(root)
    target_by_problem = {control["problem_id"]: control for control in targets["controls"]}
    results = []
    for problem in config["problems"]:
        result = learn_problem(problem, config["policy"])
        result["sealed_control_evaluation"] = _evaluate_target(
            result, target_by_problem[problem["problem_id"]]
        )
        results.append(result)
    body: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "claims": {
            "empirical_law_discovered": False,
            "invariance_implies_truth": False,
            "literature_novelty_established": False,
            "named_formula_recovered": False,
            "target_used_for_learning": False,
        },
        "release_gate": {
            "creative_context_ready": True,
            "serious_claim_released": False,
            "status": "PASS_LEARNED_CONTROLS_NO_LAW_OR_NOVELTY_CLAIM",
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
            "deployment_repaired_coordinates": sum(
                len(result["deployment_repaired_coordinates"]) for result in results
            ),
            "identified_passes": sum(
                result["status"] == "PASS_LEARNED_INVARIANT_BASIS" for result in results
            ),
            "problems": len(results),
            "shift_rejections": sum(
                result["status"] == "REJECT_TRAIN_ONLY_INVARIANT_SPACE" for result in results
            ),
            "status": "PASS_LEARNED_MULTI_INVARIANT_CONTROLS",
            "training_coordinates_retained": sum(
                len(result["training_coordinates"]) for result in results
            ),
            "underdetermined_controls": sum(
                result["status"] == "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE"
                for result in results
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
        "learner receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA or receipt["campaign_id"] != CAMPAIGN_ID:
        raise LearnedInvariantError("learner receipt identity changed")
    claims = _strict(
        receipt["claims"],
        {
            "empirical_law_discovered",
            "invariance_implies_truth",
            "literature_novelty_established",
            "named_formula_recovered",
            "target_used_for_learning",
        },
        "learner claims",
    )
    if any(claims.values()):
        raise LearnedInvariantError("learner claim boundary changed")
    expected = build_receipt(root)
    if receipt != expected:
        raise LearnedInvariantError("learner receipt no longer reproduces")
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
        _, output = _bound_path(root, args.output, "learner receipt output")
        _write_json(output, receipt)
    else:
        _, receipt_path = _bound_path(root, args.receipt, "learner receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "problems": receipt["summary"]["problems"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
