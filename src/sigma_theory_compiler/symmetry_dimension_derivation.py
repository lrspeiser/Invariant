"""Derive exact invariant coordinates forced by dimensions and declared symmetries.

This is a bounded first-principles control engine, not a law-discovery oracle.  It computes
integer nullspace coordinates with two independent exact evaluators, checks a bounded primitive
enumeration, and records the arbitrary functional freedom that dimensional and symmetry arguments
cannot remove.
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

CONFIG_PATH = "configs/symmetry_dimension_derivation.json"
OUTPUT_PATH = "runs/math/symmetry-dimension-derivation/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/symmetry_dimension_derivation.py"
TEST_PATH = "tests/test_symmetry_dimension_derivation.py"
CONFIG_SCHEMA = "invariant-symmetry-dimension-derivation-config-1.1"
RECEIPT_SCHEMA = "invariant-symmetry-dimension-derivation-receipt-1.1"
CAMPAIGN_ID = "first-principles-d4-controls-2026-08-23-002"
_EXPECTED_PROBLEMS = {
    "control.drag-similarity": "aerodynamics",
    "control.diffusion-similarity": "transport",
    "control.kepler-similarity": "orbital_dynamics",
    "control.reynolds-similarity": "fluid_dynamics",
    "control.simple-pendulum-scaling": "mechanics",
}
_ORIGIN_LABELS = [
    "known_rewrite",
    "cross_domain_synthesis",
    "proposed_new_construction",
    "uncertain",
]
_HEX = frozenset("0123456789abcdef")


class SymmetryDimensionError(ValueError):
    """A declaration, derivation, seal, or conservative claim boundary failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SymmetryDimensionError(f"{label} keys changed")


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _bound_path(root: Path, value: str | Path, label: str) -> tuple[str, Path]:
    root = root.resolve()
    candidate = (root / value).resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise SymmetryDimensionError(f"{label} escaped the repository root") from error
    return relative, candidate


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SymmetryDimensionError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _integer_vector(value: Any, length: int, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise SymmetryDimensionError(f"{label} is not an integer vector of length {length}")
    return value


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict(
        value,
        {
            "campaign_id",
            "engine_principal_id",
            "problems",
            "schema_version",
            "search_policy",
        },
        "symmetry/dimension config",
    )
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["campaign_id"] != CAMPAIGN_ID
        or value["engine_principal_id"] != "invariant.discovery-engine"
    ):
        raise SymmetryDimensionError("symmetry/dimension config identity changed")
    policy = value["search_policy"]
    _strict(
        policy,
        {
            "maximum_absolute_exponent",
            "maximum_l1_norm",
            "minimum_controls",
            "require_independent_rank_evaluators",
            "require_symmetry_nuisance_mutation",
        },
        "symmetry/dimension search policy",
    )
    if policy != {
        "maximum_absolute_exponent": 3,
        "maximum_l1_norm": 8,
        "minimum_controls": 5,
        "require_independent_rank_evaluators": True,
        "require_symmetry_nuisance_mutation": True,
    }:
        raise SymmetryDimensionError("symmetry/dimension search policy weakened")
    problems = value["problems"]
    if not isinstance(problems, list) or len(problems) != policy["minimum_controls"]:
        raise SymmetryDimensionError("symmetry/dimension control count changed")
    observed: dict[str, str] = {}
    for problem in problems:
        _strict(
            problem,
            {
                "dimension_axes",
                "domain",
                "expected_control_basis",
                "problem_id",
                "symmetry_actions",
                "variables",
            },
            "symmetry/dimension problem",
        )
        problem_id = problem["problem_id"]
        domain = problem["domain"]
        if _EXPECTED_PROBLEMS.get(problem_id) != domain or problem_id in observed:
            raise SymmetryDimensionError("symmetry/dimension problem identity changed")
        axes = problem["dimension_axes"]
        variables = problem["variables"]
        actions = problem["symmetry_actions"]
        if (
            not isinstance(axes, list)
            or len(axes) not in {2, 3}
            or len(axes) != len(set(axes))
            or any(not isinstance(axis, str) or not axis for axis in axes)
            or not isinstance(variables, list)
            or len(variables) not in {4, 5, 6}
            or not isinstance(actions, list)
            or len(actions) != 1
        ):
            raise SymmetryDimensionError("symmetry/dimension declaration shape changed")
        names: set[str] = set()
        for variable in variables:
            _strict(variable, {"dimensions", "name"}, "declared variable")
            name = variable["name"]
            if not isinstance(name, str) or not name or name in names:
                raise SymmetryDimensionError("declared variable names are missing or duplicated")
            _integer_vector(variable["dimensions"], len(axes), f"{name} dimensions")
            names.add(name)
        for action in actions:
            _strict(action, {"name", "weights"}, "declared symmetry action")
            if not isinstance(action["name"], str) or not action["name"]:
                raise SymmetryDimensionError("declared symmetry name is missing")
            weights = _integer_vector(
                action["weights"], len(variables), f"{problem_id} symmetry weights"
            )
            if not any(weights):
                raise SymmetryDimensionError("declared symmetry action is trivial")
        expected_basis = problem["expected_control_basis"]
        if (
            not isinstance(expected_basis, list)
            or not 1 <= len(expected_basis) < len(variables)
        ):
            raise SymmetryDimensionError("expected invariant basis shape changed")
        for index, vector in enumerate(expected_basis):
            expected = _integer_vector(
                vector,
                len(variables),
                f"{problem_id} expected basis vector {index}",
            )
            if not any(expected) or tuple(expected) != _canonical_integer_vector(expected):
                raise SymmetryDimensionError("expected invariant basis is not primitive canonical")
        observed[problem_id] = domain
    if observed != _EXPECTED_PROBLEMS:
        raise SymmetryDimensionError("symmetry/dimension domain coverage changed")
    return dict(value)


def load_config(root: Path, config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    _, path = _bound_path(root, config_path, "symmetry/dimension config")
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_config(value)


def _matrix_rows(problem: Mapping[str, Any], *, include_symmetry: bool = True) -> list[list[int]]:
    axes = problem["dimension_axes"]
    variables = problem["variables"]
    rows = [
        [variable["dimensions"][axis_index] for variable in variables]
        for axis_index in range(len(axes))
    ]
    if include_symmetry:
        rows.extend(list(action["weights"]) for action in problem["symmetry_actions"])
    return rows


def _rank_fraction(rows: Sequence[Sequence[int]]) -> int:
    if not rows:
        return 0
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise SymmetryDimensionError("rank matrix is empty or ragged")
    matrix = [[Fraction(item) for item in row] for row in rows]
    rank = 0
    for column in range(width):
        pivot = next(
            (index for index in range(rank, len(matrix)) if matrix[index][column] != 0),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][column]
        matrix[rank] = [item / pivot_value for item in matrix[rank]]
        for row_index, row in enumerate(matrix):
            if row_index == rank or row[column] == 0:
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


def _canonical_integer_vector(values: Sequence[Fraction | int]) -> tuple[int, ...]:
    fractions = [Fraction(value) for value in values]
    if not fractions or not any(fractions):
        raise SymmetryDimensionError("nullspace vector is trivial")
    denominator_lcm = math.lcm(*(value.denominator for value in fractions))
    integers = [value.numerator * (denominator_lcm // value.denominator) for value in fractions]
    divisor = math.gcd(*(abs(value) for value in integers if value))
    integers = [value // divisor for value in integers]
    first = next(value for value in integers if value)
    if first < 0:
        integers = [-value for value in integers]
    return tuple(integers)


def _sympy_rank_and_basis(rows: Sequence[Sequence[int]]) -> tuple[int, list[tuple[int, ...]]]:
    matrix = Matrix(rows)
    basis = [
        _canonical_integer_vector([Fraction(str(item)) for item in vector])
        for vector in matrix.nullspace()
    ]
    return int(matrix.rank()), basis


def _dot(row: Sequence[int], vector: Sequence[int]) -> int:
    return sum(left * right for left, right in zip(row, vector, strict=True))


def enumerate_primitive_invariants(
    rows: Sequence[Sequence[int]],
    *,
    variable_count: int,
    maximum_absolute_exponent: int,
    maximum_l1_norm: int,
) -> list[tuple[int, ...]]:
    invariants: set[tuple[int, ...]] = set()
    exponent_range = range(-maximum_absolute_exponent, maximum_absolute_exponent + 1)
    for candidate in itertools.product(exponent_range, repeat=variable_count):
        if not any(candidate) or sum(abs(item) for item in candidate) > maximum_l1_norm:
            continue
        primitive = _canonical_integer_vector(candidate)
        if candidate != primitive or any(_dot(row, candidate) for row in rows):
            continue
        invariants.add(candidate)
    return sorted(invariants, key=lambda item: (sum(abs(value) for value in item), item))


def _expression(names: Sequence[str], exponents: Sequence[int]) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for name, exponent in zip(names, exponents, strict=True):
        if exponent == 0:
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


def evaluate_problem(problem: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    names = [variable["name"] for variable in problem["variables"]]
    expected_basis = [tuple(vector) for vector in problem["expected_control_basis"]]
    dimension_rows = _matrix_rows(problem, include_symmetry=False)
    combined_rows = _matrix_rows(problem, include_symmetry=True)
    fraction_dimension_rank = _rank_fraction(dimension_rows)
    fraction_combined_rank = _rank_fraction(combined_rows)
    sympy_dimension_rank, sympy_dimension_basis = _sympy_rank_and_basis(dimension_rows)
    sympy_combined_rank, sympy_combined_basis = _sympy_rank_and_basis(combined_rows)
    if (
        fraction_dimension_rank != sympy_dimension_rank
        or fraction_combined_rank != sympy_combined_rank
    ):
        raise SymmetryDimensionError("independent exact rank evaluators disagree")
    nullity = len(names) - fraction_combined_rank
    declared_basis_rank = _rank_fraction(expected_basis)
    if (
        nullity != len(expected_basis)
        or len(sympy_combined_basis) != nullity
        or declared_basis_rank != nullity
        or any(_dot(row, vector) for vector in expected_basis for row in combined_rows)
    ):
        raise SymmetryDimensionError(f"{problem['problem_id']} exact nullspace changed")
    enumerated = enumerate_primitive_invariants(
        combined_rows,
        variable_count=len(names),
        maximum_absolute_exponent=policy["maximum_absolute_exponent"],
        maximum_l1_norm=policy["maximum_l1_norm"],
    )
    if any(vector not in enumerated for vector in expected_basis):
        raise SymmetryDimensionError(f"{problem['problem_id']} bounded invariant changed")

    offset_mutations = []
    for coordinate_index, expected in enumerate(expected_basis, start=1):
        offset = list(expected)
        offset[0] += 1
        offset_residuals = [_dot(row, offset) for row in combined_rows]
        if not any(offset_residuals):
            raise SymmetryDimensionError("dimension-breaking exponent mutation was admitted")
        offset_mutations.append(
            {
                "coordinate_id": f"pi_{coordinate_index}",
                "mutated_exponents": offset,
                "rejected": True,
                "row_residuals": offset_residuals,
            }
        )

    collapsed_basis = [list(vector) for vector in expected_basis]
    collapsed_basis[-1] = (
        list(expected_basis[0]) if nullity > 1 else [0 for _ in names]
    )
    collapsed_rank = _rank_fraction(collapsed_basis)
    if collapsed_rank >= nullity:
        raise SymmetryDimensionError("dependent invariant-basis mutation was admitted")

    dimension_only_nullity = len(names) - fraction_dimension_rank
    nuisance = tuple(1 if index == len(names) - 1 else 0 for index in range(len(names)))
    dimension_only_enumeration = enumerate_primitive_invariants(
        dimension_rows,
        variable_count=len(names),
        maximum_absolute_exponent=policy["maximum_absolute_exponent"],
        maximum_l1_norm=policy["maximum_l1_norm"],
    )
    if (
        dimension_only_nullity != nullity + 1
        or nuisance not in dimension_only_enumeration
        or nuisance not in sympy_dimension_basis
        or any(_dot(row, nuisance) for row in dimension_rows)
        or not any(_dot(row, nuisance) for row in combined_rows)
    ):
        raise SymmetryDimensionError("nuisance-symmetry mutation did not open the search space")

    expressions = [_expression(names, vector) for vector in expected_basis]
    invariants = [
        {
            "coordinate_id": f"pi_{index}",
            "exponents": list(vector),
            "expression": expression,
            "primitive": True,
        }
        for index, (vector, expression) in enumerate(
            zip(expected_basis, expressions, strict=True), start=1
        )
    ]
    return {
        "creative_brief": {
            "candidate_invariant_coordinates": expressions,
            "constraint_statement": (
                "Generate hypotheses only through the declared dimension and symmetry quotient; "
                "retain multiple mechanisms until exact falsification."
            ),
            "llm_origin_assessment_labels": _ORIGIN_LABELS,
            "novelty_caution": (
                "Dimensional and symmetry admissibility is neither empirical truth nor literature novelty."
            ),
        },
        "dimensions": {
            "axes": list(problem["dimension_axes"]),
            "rank": fraction_dimension_rank,
        },
        "domain": problem["domain"],
        "forced_form": {
            "free_function_arity": nullity,
            "free_function_determined": False,
            "statement": f"F({','.join(expressions)}) = 0",
        },
        "independent_evaluators": {
            "agreement": True,
            "declared_basis_rank": declared_basis_rank,
            "fraction_gaussian_elimination_rank": fraction_combined_rank,
            "sympy_exact_nullspace_basis": [list(vector) for vector in sympy_combined_basis],
            "sympy_rank": sympy_combined_rank,
        },
        "invariant_coordinates": invariants,
        "mutations": {
            "basis_collapse": {
                "baseline_rank": nullity,
                "mutated_basis": collapsed_basis,
                "mutated_rank": collapsed_rank,
                "rejected": True,
            },
            "dimension_exponent_offsets": offset_mutations,
            "drop_nuisance_symmetry": {
                "baseline_nullity": nullity,
                "mutated_nullity": dimension_only_nullity,
                "nuisance_coordinate_admitted": True,
                "nuisance_exponents": list(nuisance),
                "rejected": True,
            },
        },
        "problem_id": problem["problem_id"],
        "search": {
            "bounded_primitive_invariants": len(enumerated),
            "combined_rank": fraction_combined_rank,
            "nullity": nullity,
        },
        "symmetry_actions": [action["name"] for action in problem["symmetry_actions"]],
        "variables": names,
    }


def _source_binding(root: Path, relative: str) -> dict[str, str]:
    _, path = _bound_path(root, relative, "D4 source binding")
    if not path.is_file():
        raise SymmetryDimensionError(f"D4 source binding is missing: {relative}")
    return {"normalized_sha256": _normalized_file_sha256(path), "path": relative}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    results = [evaluate_problem(problem, config["search_policy"]) for problem in config["problems"]]
    if len(results) != config["search_policy"]["minimum_controls"]:
        raise SymmetryDimensionError("D4 control coverage changed")
    body: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "claims": {
            "dimensionless_implies_true": False,
            "empirical_fit_established": False,
            "literature_novelty_established": False,
            "specific_law_discovered": False,
            "symmetry_dimension_constraints_determine_free_function": False,
        },
        "engine_principal_id": config["engine_principal_id"],
        "evaluators": {
            "agreement_required": True,
            "implementations": ["fraction_gaussian_elimination", "sympy_exact_nullspace"],
        },
        "release_gate": {
            "d4_controls_ready": True,
            "multiple_invariant_coordinates_ready": True,
            "serious_claim_released": False,
            "status": "PASS_D4_MULTI_COORDINATE_CONTROLS_NO_LAW_OR_NOVELTY_CLAIM",
        },
        "results": results,
        "schema_version": RECEIPT_SCHEMA,
        "source_bindings": {
            "config": _source_binding(root, CONFIG_PATH),
            "source": _source_binding(root, SOURCE_PATH),
            "test": _source_binding(root, TEST_PATH),
        },
        "summary": {
            "basis_collapse_mutations_rejected": sum(
                result["mutations"]["basis_collapse"]["rejected"] for result in results
            ),
            "controls_passed": len(results),
            "dimension_mutations_rejected": sum(
                sum(
                    mutation["rejected"]
                    for mutation in result["mutations"]["dimension_exponent_offsets"]
                )
                for result in results
            ),
            "invariant_coordinates": sum(result["search"]["nullity"] for result in results),
            "multi_coordinate_controls": sum(
                result["search"]["nullity"] > 1 for result in results
            ),
            "status": "PASS_SYMMETRY_DIMENSION_MULTI_COORDINATE_DERIVATION",
            "symmetry_mutations_rejected": sum(
                result["mutations"]["drop_nuisance_symmetry"]["rejected"]
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
            "engine_principal_id",
            "evaluators",
            "release_gate",
            "results",
            "schema_version",
            "source_bindings",
            "summary",
        },
        "symmetry/dimension receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA or receipt["campaign_id"] != CAMPAIGN_ID:
        raise SymmetryDimensionError("symmetry/dimension receipt identity changed")
    claims = receipt["claims"]
    _strict(
        claims,
        {
            "dimensionless_implies_true",
            "empirical_fit_established",
            "literature_novelty_established",
            "specific_law_discovered",
            "symmetry_dimension_constraints_determine_free_function",
        },
        "symmetry/dimension claims",
    )
    if any(claims.values()):
        raise SymmetryDimensionError("symmetry/dimension claim boundary was promoted")
    _sha(receipt["content_sha256"], "symmetry/dimension content seal")
    expected = build_receipt(root)
    if receipt != expected:
        raise SymmetryDimensionError("symmetry/dimension receipt no longer reproduces")
    return dict(receipt)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, default=Path.cwd())
        subparser.add_argument(
            "--output" if command == "build" else "--receipt",
            type=Path,
            default=Path(OUTPUT_PATH),
        )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "build":
        receipt = build_receipt(root)
        _, output = _bound_path(root, args.output, "symmetry/dimension receipt output")
        _write_json(output, receipt)
    else:
        _, receipt_path = _bound_path(root, args.receipt, "symmetry/dimension receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "controls": receipt["summary"]["controls_passed"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
