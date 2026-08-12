"""Deterministic bounded equality saturation over a closed exact rewrite registry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .math_expression_ir import Expression, add, expression_to_data, literal, multiply
from .sigma_core import ArtifactKind, CandidateArtifact, ProvenanceRecord, canonical_sha256

RESULT_SCHEMA = "sigma-egraph-candidate-generator-result-1.0"
REGISTERED_RULE_IDS = (
    "add_associative_flatten",
    "add_commutative_sort",
    "add_identity",
    "additive_inverse",
    "constant_add",
    "constant_multiply",
    "constant_negation",
    "distributive_expand",
    "distributive_factor_common_left",
    "double_negation",
    "multiply_associative_flatten",
    "multiply_commutative_sort",
    "multiply_identity",
    "multiply_zero",
)
_RULE_STATEMENTS = {
    "add_associative_flatten": "nested finite addition may be flattened",
    "add_commutative_sort": "finite addition arguments may be permuted",
    "add_identity": "adding exact zero preserves a value",
    "additive_inverse": "a value plus its syntactic negation is exact zero",
    "constant_add": "a finite sum of exact rational literals may be evaluated",
    "constant_multiply": "a finite product of exact rational literals may be evaluated",
    "constant_negation": "the negation of an exact rational literal may be evaluated",
    "distributive_expand": "finite multiplication distributes over finite addition",
    "distributive_factor_common_left": "a common finite product factor may be extracted",
    "double_negation": "two exact additive negations cancel",
    "multiply_associative_flatten": "nested finite multiplication may be flattened",
    "multiply_commutative_sort": "finite multiplication arguments may be permuted",
    "multiply_identity": "multiplying by exact one preserves a value",
    "multiply_zero": "multiplying by exact zero yields exact zero",
}
_ALLOWED_OPERATIONS = {"literal", "symbol", "add", "multiply", "negate"}


class EGraphBoundaryError(ValueError):
    """Raised when saturation leaves the closed exact boundary."""


@dataclass(frozen=True, slots=True)
class SaturationLimits:
    maximum_nodes: int = 512
    maximum_iterations: int = 12
    maximum_work_units: int = 50_000

    def __post_init__(self) -> None:
        for name in ("maximum_nodes", "maximum_iterations", "maximum_work_units"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise EGraphBoundaryError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_nodes": self.maximum_nodes,
            "maximum_iterations": self.maximum_iterations,
            "maximum_work_units": self.maximum_work_units,
        }


DEFAULT_LIMITS = SaturationLimits()


def _expression_sha(expression: Expression) -> str:
    return canonical_sha256(expression_to_data(expression))


def _is_exact_zero(expression: Expression) -> bool:
    return expression.operation == "literal" and Fraction(expression.value) == 0


def _is_exact_one(expression: Expression) -> bool:
    return expression.operation == "literal" and Fraction(expression.value) == 1


def _validate_expression(expression: Expression) -> None:
    if not isinstance(expression, Expression):
        raise EGraphBoundaryError("seed is not a Math Expression IR node")
    if expression.operation not in _ALLOWED_OPERATIONS:
        raise EGraphBoundaryError(
            f"operation is outside the closed rewrite domain: {expression.operation}"
        )
    if expression.math_type is not None:
        raise EGraphBoundaryError(
            "typed expressions require a separately proved typed rewrite registry"
        )
    if expression.operation == "literal" and (
        isinstance(expression.value, bool) or not isinstance(expression.value, (int, Fraction))
    ):
        raise EGraphBoundaryError("only exact integer and rational literals are admitted")
    for argument in expression.arguments:
        _validate_expression(argument)


def _paths(expression: Expression) -> tuple[tuple[int, ...], ...]:
    paths = [()]
    for index, argument in enumerate(expression.arguments):
        paths.extend((index, *child) for child in _paths(argument))
    return tuple(paths)


def _at(expression: Expression, path: tuple[int, ...]) -> Expression:
    current = expression
    for index in path:
        current = current.arguments[index]
    return current


def _rebuild(expression: Expression, arguments: tuple[Expression, ...]) -> Expression:
    return Expression(expression.operation, arguments, expression.value, expression.math_type)


def _replace(expression: Expression, path: tuple[int, ...], replacement: Expression) -> Expression:
    if not path:
        return replacement
    index = path[0]
    arguments = list(expression.arguments)
    arguments[index] = _replace(arguments[index], path[1:], replacement)
    return _rebuild(expression, tuple(arguments))


def _sorted(arguments: Sequence[Expression]) -> tuple[Expression, ...]:
    return tuple(sorted(arguments, key=_expression_sha))


def _product(arguments: Sequence[Expression]) -> Expression:
    if not arguments:
        return literal(1)
    if len(arguments) == 1:
        return arguments[0]
    return multiply(*arguments)


def _sum(arguments: Sequence[Expression]) -> Expression:
    if not arguments:
        return literal(0)
    if len(arguments) == 1:
        return arguments[0]
    return add(*arguments)


def _apply_rule(rule_id: str, expression: Expression) -> Expression | None:
    arguments = expression.arguments
    if rule_id == "add_associative_flatten" and expression.operation == "add":
        flattened = tuple(
            child
            for argument in arguments
            for child in (argument.arguments if argument.operation == "add" else (argument,))
        )
        return add(*flattened) if flattened != arguments else None
    if rule_id == "multiply_associative_flatten" and expression.operation == "multiply":
        flattened = tuple(
            child
            for argument in arguments
            for child in (argument.arguments if argument.operation == "multiply" else (argument,))
        )
        return multiply(*flattened) if flattened != arguments else None
    if rule_id == "add_commutative_sort" and expression.operation == "add":
        ordered = _sorted(arguments)
        return add(*ordered) if ordered != arguments else None
    if rule_id == "multiply_commutative_sort" and expression.operation == "multiply":
        ordered = _sorted(arguments)
        return multiply(*ordered) if ordered != arguments else None
    if rule_id == "add_identity" and expression.operation == "add":
        kept = tuple(argument for argument in arguments if not _is_exact_zero(argument))
        return _sum(kept) if kept != arguments else None
    if rule_id == "multiply_identity" and expression.operation == "multiply":
        kept = tuple(argument for argument in arguments if not _is_exact_one(argument))
        return _product(kept) if kept != arguments else None
    if (
        rule_id == "multiply_zero"
        and expression.operation == "multiply"
        and any(_is_exact_zero(argument) for argument in arguments)
    ):
        return literal(0)
    if (
        rule_id == "double_negation"
        and expression.operation == "negate"
        and arguments[0].operation == "negate"
    ):
        return arguments[0].arguments[0]
    if rule_id == "constant_negation" and expression.operation == "negate":
        argument = arguments[0]
        if argument.operation == "literal":
            return literal(-Fraction(argument.value))
    if (
        rule_id == "constant_add"
        and expression.operation == "add"
        and all(argument.operation == "literal" for argument in arguments)
    ):
        return literal(sum((Fraction(argument.value) for argument in arguments), Fraction(0)))
    if (
        rule_id == "constant_multiply"
        and expression.operation == "multiply"
        and all(argument.operation == "literal" for argument in arguments)
    ):
        value = Fraction(1)
        for argument in arguments:
            value *= Fraction(argument.value)
        return literal(value)
    if rule_id == "additive_inverse" and expression.operation == "add":
        hashes = {_expression_sha(argument): argument for argument in arguments}
        for index, argument in enumerate(arguments):
            if argument.operation == "negate":
                positive = argument.arguments[0]
                positive_hash = _expression_sha(positive)
                if positive_hash in hashes:
                    remaining = list(arguments)
                    remaining.pop(index)
                    for other_index, other in enumerate(remaining):
                        if _expression_sha(other) == positive_hash:
                            remaining.pop(other_index)
                            return _sum((*remaining, literal(0)))
    if rule_id == "distributive_expand" and expression.operation == "multiply":
        for index, argument in enumerate(arguments):
            if argument.operation == "add":
                terms = [
                    _product((*arguments[:index], term, *arguments[index + 1 :]))
                    for term in argument.arguments
                ]
                return _sum(terms)
    if (
        rule_id == "distributive_factor_common_left"
        and expression.operation == "add"
        and len(arguments) == 2
    ):
        left = list(
            arguments[0].arguments if arguments[0].operation == "multiply" else (arguments[0],)
        )
        right = list(
            arguments[1].arguments if arguments[1].operation == "multiply" else (arguments[1],)
        )
        common_hashes = sorted(
            {_expression_sha(item) for item in left} & {_expression_sha(item) for item in right}
        )
        if common_hashes:
            selected = common_hashes[0]
            left_index = next(i for i, item in enumerate(left) if _expression_sha(item) == selected)
            right_index = next(
                i for i, item in enumerate(right) if _expression_sha(item) == selected
            )
            common = left.pop(left_index)
            right.pop(right_index)
            return multiply(common, add(_product(left), _product(right)))
    return None


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        low, high = sorted((left_root, right_root))
        self.parent[high] = low
        return True


def _subtree_closure(expression: Expression) -> dict[str, Expression]:
    result: dict[str, Expression] = {}

    def visit(node: Expression) -> None:
        digest = _expression_sha(node)
        result.setdefault(digest, node)
        for argument in node.arguments:
            visit(argument)

    visit(expression)
    return result


def _value_key(expression: Expression) -> Any:
    if expression.operation == "literal":
        value = Fraction(expression.value)
        return (value.numerator, value.denominator)
    return expression.value


def _congruence_signature(expression: Expression, union: _UnionFind) -> tuple[Any, ...]:
    return (
        expression.operation,
        _value_key(expression),
        tuple(union.find(_expression_sha(argument)) for argument in expression.arguments),
    )


def _cost(expression: Expression) -> tuple[int, int, str]:
    child_costs = [_cost(argument) for argument in expression.arguments]
    nodes = 1 + sum(item[0] for item in child_costs)
    depth = 1 + max((item[1] for item in child_costs), default=0)
    return nodes, depth, canonical_sha256(expression_to_data(expression))


def _registered_rules(rule_ids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(rule_ids)
    unknown = sorted(set(values) - set(REGISTERED_RULE_IDS))
    if unknown:
        raise EGraphBoundaryError(f"unregistered or unproved rewrite rule: {unknown[0]}")
    if not values or values != tuple(sorted(set(values))):
        raise EGraphBoundaryError("rewrite rule IDs must be nonempty, sorted, and unique")
    return values


def _rule_manifest(rule_ids: Sequence[str]) -> list[dict[str, str]]:
    return [
        {
            "rule_id": rule_id,
            "statement": _RULE_STATEMENTS[rule_id],
            "content_sha256": canonical_sha256(
                {"rule_id": rule_id, "statement": _RULE_STATEMENTS[rule_id]}
            ),
        }
        for rule_id in rule_ids
    ]


def saturate_expressions(
    seeds: Sequence[Expression],
    *,
    rule_ids: Sequence[str] = REGISTERED_RULE_IDS,
    limits: SaturationLimits = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Saturate exact IR expressions under registered rules and deterministic work caps."""

    if not isinstance(limits, SaturationLimits):
        raise EGraphBoundaryError("limits must be SaturationLimits")
    rules = _registered_rules(rule_ids)
    if not seeds:
        raise EGraphBoundaryError("at least one seed expression is required")
    for seed in seeds:
        _validate_expression(seed)

    expressions: dict[str, Expression] = {}
    union = _UnionFind()

    def add_closure(expression: Expression) -> bool:
        closure = _subtree_closure(expression)
        new_hashes = sorted(set(closure) - set(expressions))
        if len(expressions) + len(new_hashes) > limits.maximum_nodes:
            return False
        for digest in new_hashes:
            expressions[digest] = closure[digest]
            union.add(digest)
        return True

    seed_hashes = tuple(_expression_sha(seed) for seed in seeds)
    for seed in seeds:
        if not add_closure(seed):
            raise EGraphBoundaryError("seed expression closure exceeds maximum_nodes")

    lineage: list[dict[str, Any]] = []
    work_units = 0
    deduplicated_results = 0
    direct_rewrites = 0
    congruence_merges = 0
    iterations_completed = 0
    decision = "bounded_iteration_cap"
    complete = False
    stopped = False

    for iteration in range(1, limits.maximum_iterations + 1):
        changed = False
        snapshot = tuple(sorted(expressions))
        for source_sha in snapshot:
            source = expressions[source_sha]
            for path in _paths(source):
                target_node = _at(source, path)
                for rule_id in rules:
                    if work_units >= limits.maximum_work_units:
                        decision = "bounded_work_unit_cap"
                        stopped = True
                        break
                    work_units += 1
                    replacement = _apply_rule(rule_id, target_node)
                    if replacement is None:
                        continue
                    target = _replace(source, path, replacement)
                    target_sha = _expression_sha(target)
                    existed = target_sha in expressions
                    if not add_closure(target):
                        decision = "bounded_node_cap"
                        stopped = True
                        break
                    if existed:
                        deduplicated_results += 1
                    if union.union(source_sha, target_sha):
                        direct_rewrites += 1
                        changed = True
                        lineage.append(
                            {
                                "sequence": len(lineage),
                                "iteration": iteration,
                                "kind": "registered_rewrite",
                                "rule_id": rule_id,
                                "source_expression_sha256": source_sha,
                                "result_expression_sha256": target_sha,
                                "path": list(path),
                            }
                        )
                if stopped:
                    break
            if stopped:
                break
        if not stopped:
            while True:
                groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
                for digest in sorted(expressions):
                    groups[_congruence_signature(expressions[digest], union)].append(digest)
                merged = False
                for members in sorted(groups.values(), key=lambda rows: rows[0]):
                    for other in members[1:]:
                        if union.union(members[0], other):
                            congruence_merges += 1
                            changed = merged = True
                            lineage.append(
                                {
                                    "sequence": len(lineage),
                                    "iteration": iteration,
                                    "kind": "congruence_closure",
                                    "rule_id": "registered_congruence_inference",
                                    "source_expression_sha256": members[0],
                                    "result_expression_sha256": other,
                                    "path": [],
                                }
                            )
                if not merged:
                    break
        iterations_completed = iteration
        if stopped:
            break
        if not changed:
            complete = True
            decision = "saturated_registered_rules_fixed_point"
            break

    classes: defaultdict[str, list[str]] = defaultdict(list)
    for digest in sorted(expressions):
        classes[union.find(digest)].append(digest)
    eclasses = []
    for members in sorted(classes.values(), key=lambda rows: rows[0]):
        representative = min(members)
        extracted = min(members, key=lambda digest: _cost(expressions[digest]))
        cost = _cost(expressions[extracted])
        eclasses.append(
            {
                "eclass_id": f"eclass-{representative[:24]}",
                "member_expression_sha256s": members,
                "extracted_expression_sha256": extracted,
                "extraction_cost": {"nodes": cost[0], "depth": cost[1]},
            }
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "fixed_point_complete": complete,
        "limits": limits.to_dict(),
        "registered_rules": _rule_manifest(rules),
        "seed_expression_sha256s": list(seed_hashes),
        "expressions": [
            {"expression_sha256": digest, "expression": expression_to_data(expressions[digest])}
            for digest in sorted(expressions)
        ],
        "eclasses": eclasses,
        "rewrite_lineage": lineage,
        "rewrite_lineage_root_sha256": canonical_sha256(lineage),
        "counts": {
            "seed_inputs": len(seeds),
            "unique_seed_hashes": len(set(seed_hashes)),
            "unique_expression_nodes": len(expressions),
            "canonical_eclasses": len(eclasses),
            "iterations_completed": iterations_completed,
            "work_units_consumed": work_units,
            "direct_rewrite_merges": direct_rewrites,
            "congruence_merges": congruence_merges,
            "deduplicated_rewrite_results": deduplicated_results,
        },
        "claims": {
            "equivalence_scope": "registered_exact_rewrites_and_congruence_only",
            "unregistered_equivalence_claimed": False,
            "novelty_claimed": False,
            "promotion_authorized": False,
            "time_based_termination_used": False,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    validate_saturation_result(result)
    return result


def _exact_number_from_data(value: Mapping[str, Any]) -> int | Fraction:
    if value.get("kind") == "integer" and set(value) == {"kind", "value"}:
        integer = value["value"]
        if isinstance(integer, int) and not isinstance(integer, bool):
            return integer
    if value.get("kind") == "rational" and set(value) == {"kind", "numerator", "denominator"}:
        numerator = value["numerator"]
        denominator = value["denominator"]
        if all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in (numerator, denominator)
        ):
            return Fraction(numerator, denominator)
    raise EGraphBoundaryError("serialized expression literal is not exact rational data")


def _expression_from_data(value: Mapping[str, Any]) -> Expression:
    if not isinstance(value, Mapping) or set(value) - {"operation", "arguments", "value"}:
        raise EGraphBoundaryError("serialized expression keys changed")
    operation = value.get("operation")
    arguments = value.get("arguments")
    if operation not in _ALLOWED_OPERATIONS or not isinstance(arguments, list):
        raise EGraphBoundaryError("serialized expression operation changed")
    children = tuple(_expression_from_data(child) for child in arguments)
    if operation == "literal":
        return literal(_exact_number_from_data(value.get("value", {})))
    if operation == "symbol":
        if set(value) != {"operation", "arguments", "value"}:
            raise EGraphBoundaryError("serialized symbol keys changed")
        return Expression("symbol", value=value["value"])
    if set(value) != {"operation", "arguments"}:
        raise EGraphBoundaryError("serialized operator keys changed")
    return Expression(operation, children)


def validate_saturation_result(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "decision",
        "fixed_point_complete",
        "limits",
        "registered_rules",
        "seed_expression_sha256s",
        "expressions",
        "eclasses",
        "rewrite_lineage",
        "rewrite_lineage_root_sha256",
        "counts",
        "claims",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise EGraphBoundaryError("saturation result keys changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["schema_version"] != RESULT_SCHEMA or value["content_sha256"] != canonical_sha256(
        body
    ):
        raise EGraphBoundaryError("saturation result seal changed")
    limits = SaturationLimits(**value["limits"])
    rules = tuple(row.get("rule_id") for row in value["registered_rules"])
    _registered_rules(rules)
    if value["registered_rules"] != _rule_manifest(rules):
        raise EGraphBoundaryError("registered rewrite manifest changed")
    expression_rows = value["expressions"]
    if not isinstance(expression_rows, list):
        raise EGraphBoundaryError("expressions must be an array")
    expressions: dict[str, Expression] = {}
    for row in expression_rows:
        if not isinstance(row, Mapping) or set(row) != {"expression_sha256", "expression"}:
            raise EGraphBoundaryError("expression row keys changed")
        expression = _expression_from_data(row["expression"])
        digest = _expression_sha(expression)
        if row["expression_sha256"] != digest or digest in expressions:
            raise EGraphBoundaryError("expression content hash or dedup changed")
        expressions[digest] = expression
    if list(expressions) != sorted(expressions) or len(expressions) > limits.maximum_nodes:
        raise EGraphBoundaryError("expression ordering or node cap changed")
    eclasses = value["eclasses"]
    membership: dict[str, str] = {}
    for row in eclasses:
        if set(row) != {
            "eclass_id",
            "member_expression_sha256s",
            "extracted_expression_sha256",
            "extraction_cost",
        }:
            raise EGraphBoundaryError("e-class keys changed")
        members = row["member_expression_sha256s"]
        if (
            not members
            or members != sorted(set(members))
            or any(item not in expressions for item in members)
        ):
            raise EGraphBoundaryError("e-class membership changed")
        if row["eclass_id"] != f"eclass-{min(members)[:24]}":
            raise EGraphBoundaryError("canonical e-class ID changed")
        extracted = min(members, key=lambda digest: _cost(expressions[digest]))
        cost = _cost(expressions[extracted])
        if row["extracted_expression_sha256"] != extracted or row["extraction_cost"] != {
            "nodes": cost[0],
            "depth": cost[1],
        }:
            raise EGraphBoundaryError("canonical extraction changed")
        for member in members:
            if member in membership:
                raise EGraphBoundaryError("expression appears in multiple e-classes")
            membership[member] = row["eclass_id"]
    if set(membership) != set(expressions):
        raise EGraphBoundaryError("e-classes do not partition expressions")
    lineage = value["rewrite_lineage"]
    if value["rewrite_lineage_root_sha256"] != canonical_sha256(lineage):
        raise EGraphBoundaryError("rewrite lineage root changed")
    for sequence, event in enumerate(lineage):
        if (
            set(event)
            != {
                "sequence",
                "iteration",
                "kind",
                "rule_id",
                "source_expression_sha256",
                "result_expression_sha256",
                "path",
            }
            or event["sequence"] != sequence
        ):
            raise EGraphBoundaryError("rewrite lineage event keys or sequence changed")
        source_sha = event["source_expression_sha256"]
        result_sha = event["result_expression_sha256"]
        if source_sha not in expressions or result_sha not in expressions:
            raise EGraphBoundaryError("rewrite lineage references unknown expression")
        if membership[source_sha] != membership[result_sha]:
            raise EGraphBoundaryError("rewrite lineage crosses final e-classes")
        if event["kind"] == "registered_rewrite":
            if event["rule_id"] not in rules or not isinstance(event["path"], list):
                raise EGraphBoundaryError("rewrite lineage uses an unregistered rule")
            try:
                path = tuple(event["path"])
                replacement = _apply_rule(event["rule_id"], _at(expressions[source_sha], path))
                rebuilt = (
                    None
                    if replacement is None
                    else _replace(expressions[source_sha], path, replacement)
                )
            except (IndexError, TypeError):
                rebuilt = None
            if rebuilt is None or _expression_sha(rebuilt) != result_sha:
                raise EGraphBoundaryError("rewrite lineage step is not replayable")
        elif event["kind"] == "congruence_closure":
            if event["rule_id"] != "registered_congruence_inference" or event["path"] != []:
                raise EGraphBoundaryError("congruence lineage contract changed")
            left = expressions[source_sha]
            right = expressions[result_sha]
            if (
                left.operation != right.operation
                or _value_key(left) != _value_key(right)
                or len(left.arguments) != len(right.arguments)
                or any(
                    membership[_expression_sha(a)] != membership[_expression_sha(b)]
                    for a, b in zip(left.arguments, right.arguments, strict=True)
                )
            ):
                raise EGraphBoundaryError("congruence inference is not justified")
        else:
            raise EGraphBoundaryError("rewrite lineage kind changed")
    counts = value["counts"]
    if (
        set(counts)
        != {
            "seed_inputs",
            "unique_seed_hashes",
            "unique_expression_nodes",
            "canonical_eclasses",
            "iterations_completed",
            "work_units_consumed",
            "direct_rewrite_merges",
            "congruence_merges",
            "deduplicated_rewrite_results",
        }
        or counts["unique_expression_nodes"] != len(expressions)
        or counts["canonical_eclasses"] != len(eclasses)
        or counts["unique_seed_hashes"] != len(set(value["seed_expression_sha256s"]))
        or counts["work_units_consumed"] > limits.maximum_work_units
        or counts["iterations_completed"] > limits.maximum_iterations
        or counts["direct_rewrite_merges"]
        != sum(event["kind"] == "registered_rewrite" for event in lineage)
        or counts["congruence_merges"]
        != sum(event["kind"] == "congruence_closure" for event in lineage)
    ):
        raise EGraphBoundaryError("saturation counts changed")
    expected_claims = {
        "equivalence_scope": "registered_exact_rewrites_and_congruence_only",
        "unregistered_equivalence_claimed": False,
        "novelty_claimed": False,
        "promotion_authorized": False,
        "time_based_termination_used": False,
    }
    if value["claims"] != expected_claims:
        raise EGraphBoundaryError("saturation claim boundary changed")
    if value["fixed_point_complete"] is True:
        if value["decision"] != "saturated_registered_rules_fixed_point":
            raise EGraphBoundaryError("fixed-point decision changed")
    elif value["decision"] not in {
        "bounded_iteration_cap",
        "bounded_node_cap",
        "bounded_work_unit_cap",
    }:
        raise EGraphBoundaryError("bounded decision changed")


def validate_replay(
    value: Mapping[str, Any],
    seeds: Sequence[Expression],
    *,
    rule_ids: Sequence[str] = REGISTERED_RULE_IDS,
    limits: SaturationLimits = DEFAULT_LIMITS,
) -> None:
    validate_saturation_result(value)
    if dict(value) != saturate_expressions(seeds, rule_ids=rule_ids, limits=limits):
        raise EGraphBoundaryError("saturation deterministic replay mismatch")


def extract_candidate_artifacts(
    result: Mapping[str, Any], provenance: ProvenanceRecord
) -> tuple[CandidateArtifact, ...]:
    """Emit one Sigma Core identity candidate per distinct seed e-class."""

    validate_saturation_result(result)
    if not isinstance(provenance, ProvenanceRecord):
        raise EGraphBoundaryError("candidate provenance must be a Sigma Core ProvenanceRecord")
    expression_rows = {row["expression_sha256"]: row["expression"] for row in result["expressions"]}
    class_by_member = {
        member: row for row in result["eclasses"] for member in row["member_expression_sha256s"]
    }
    selected = {
        class_by_member[seed]["eclass_id"]: class_by_member[seed]
        for seed in result["seed_expression_sha256s"]
    }
    candidates = []
    for eclass_id, row in sorted(selected.items()):
        extracted_sha = row["extracted_expression_sha256"]
        candidates.append(
            CandidateArtifact.create(
                ArtifactKind.IDENTITY,
                f"Canonical representative of {eclass_id} under the registered exact rewrite set",
                {
                    "generator_schema_version": RESULT_SCHEMA,
                    "saturation_result_sha256": result["content_sha256"],
                    "rewrite_lineage_root_sha256": result["rewrite_lineage_root_sha256"],
                    "eclass_id": eclass_id,
                    "member_expression_sha256s": row["member_expression_sha256s"],
                    "extracted_expression_sha256": extracted_sha,
                    "extracted_expression": expression_rows[extracted_sha],
                    "registered_rule_ids": [item["rule_id"] for item in result["registered_rules"]],
                    "fixed_point_complete": result["fixed_point_complete"],
                },
                provenance,
                assumptions=(
                    "Equivalence is limited to the content-hash-bound registered rewrite manifest.",
                ),
                claims=("registered_rewrite_equivalence",),
            )
        )
    unique = {candidate.content_sha256: candidate for candidate in candidates}
    return tuple(unique[digest] for digest in sorted(unique))
