"""Sigma Core adapter for exact, fail-closed mathematical formula evaluation.

The primitive Math Pack modules deliberately do not depend on :mod:`sigma_core`.
This adapter is the narrow integration boundary: it turns their deterministic
checks into typed Sigma stage and promotion-gate outcomes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import sympy as sp

from .math_canonicalizer import canonical_sha256 as formula_sha256
from .math_canonicalizer import to_sympy
from .math_counterexample import SearchStatus, SearchStrategy, find_counterexample
from .math_expression_ir import (
    Equation,
    Expression,
    Formula,
    Inequality,
    InequalityRelation,
    Recurrence,
    add,
    call,
    formula_to_data,
    literal,
    multiply,
    negate,
    power,
    symbol,
)
from .math_types import ComplexType, IntegerType, MathType, RationalType, RealType
from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    CheckResult,
    DomainPackDescriptor,
    GateDefinition,
    GateOutcome,
    OutcomeStatus,
    StageDefinition,
    StageOutcome,
)

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = tuple(
    sorted(
        (
            ArtifactKind.CONJECTURE,
            ArtifactKind.FORMULA,
            ArtifactKind.IDENTITY,
            ArtifactKind.THEOREM,
        ),
        key=lambda item: item.value,
    )
)
_STAGE_IDS = (
    "typed",
    "canonicalized",
    "counterexample_screened",
    "exactly_verified",
    "prior_art_checked",
)


class MathPackBoundaryError(ValueError):
    """Raised when a mathematical candidate crosses an undeclared boundary."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MathPackBoundaryError(f"{label} keys changed")


def _fraction_data(value: int | Fraction) -> dict[str, int]:
    rational = Fraction(value)
    return {"numerator": rational.numerator, "denominator": rational.denominator}


def _fraction_from_data(value: Mapping[str, Any], label: str) -> Fraction:
    _exact_keys(value, {"numerator", "denominator"}, label)
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise MathPackBoundaryError(f"{label} is not a canonical rational")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise MathPackBoundaryError(f"{label} is not reduced")
    return result


def _type_data(value: MathType) -> dict[str, Any]:
    if isinstance(value, IntegerType):
        return {"kind": "integer", "minimum": value.minimum, "maximum": value.maximum}
    if isinstance(value, RationalType):
        return {
            "kind": "rational",
            "minimum": None if value.minimum is None else _fraction_data(value.minimum),
            "maximum": None if value.maximum is None else _fraction_data(value.maximum),
        }
    if isinstance(value, RealType):
        if isinstance(value.minimum, float) or isinstance(value.maximum, float):
            raise MathPackBoundaryError("real bounds must be exact in Sigma candidates")
        return {
            "kind": "real",
            "minimum": None if value.minimum is None else _fraction_data(value.minimum),
            "maximum": None if value.maximum is None else _fraction_data(value.maximum),
        }
    if isinstance(value, ComplexType):
        return {"kind": "complex"}
    raise MathPackBoundaryError("automatic Math Pack evaluation supports scalar types only")


def _type_from_data(value: Mapping[str, Any]) -> MathType:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise MathPackBoundaryError("variable type must be an object with a kind")
    kind = value["kind"]
    if kind == "integer":
        _exact_keys(value, {"kind", "minimum", "maximum"}, "integer type")
        bounds = (value["minimum"], value["maximum"])
        if any(
            item is not None and (isinstance(item, bool) or not isinstance(item, int))
            for item in bounds
        ):
            raise MathPackBoundaryError("integer bounds must be integers or null")
        return IntegerType(*bounds)
    if kind in {"rational", "real"}:
        _exact_keys(value, {"kind", "minimum", "maximum"}, f"{kind} type")
        bounds = tuple(
            None if value[name] is None else _fraction_from_data(value[name], f"{kind} {name}")
            for name in ("minimum", "maximum")
        )
        return RationalType(*bounds) if kind == "rational" else RealType(*bounds)
    if kind == "complex":
        _exact_keys(value, {"kind"}, "complex type")
        return ComplexType()
    raise MathPackBoundaryError("unregistered scalar mathematical type")


def _literal_from_data(value: Mapping[str, Any]) -> int | Fraction:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise MathPackBoundaryError("literal payload is malformed")
    if value["kind"] == "integer":
        _exact_keys(value, {"kind", "value"}, "integer literal")
        number = value["value"]
        if isinstance(number, bool) or not isinstance(number, int):
            raise MathPackBoundaryError("integer literal is not an integer")
        return number
    if value["kind"] == "rational":
        _exact_keys(value, {"kind", "numerator", "denominator"}, "rational literal")
        return _fraction_from_data(
            {"numerator": value["numerator"], "denominator": value["denominator"]},
            "rational literal",
        )
    raise MathPackBoundaryError("only exact real rational literals are supported")


def _expression_from_data(value: Mapping[str, Any]) -> Expression:
    if not isinstance(value, Mapping):
        raise MathPackBoundaryError("expression must be an object")
    allowed = {"operation", "arguments", "value"}
    if not set(value) <= allowed or not {"operation", "arguments"} <= set(value):
        raise MathPackBoundaryError("expression keys changed")
    operation, raw_arguments = value["operation"], value["arguments"]
    if not isinstance(operation, str) or not isinstance(raw_arguments, list):
        raise MathPackBoundaryError("expression operation/arguments are malformed")
    arguments = tuple(_expression_from_data(item) for item in raw_arguments)
    if operation == "literal":
        if set(value) != allowed or arguments:
            raise MathPackBoundaryError("literal expression shape changed")
        return literal(_literal_from_data(value["value"]))
    if operation == "symbol":
        if set(value) != allowed or arguments or not isinstance(value["value"], str):
            raise MathPackBoundaryError("symbol expression shape changed")
        return symbol(value["value"])
    if "value" in value:
        if operation != "call" or not isinstance(value["value"], str):
            raise MathPackBoundaryError("unexpected expression value")
        return call(value["value"], *arguments)
    constructors = {
        "add": lambda: add(*arguments),
        "multiply": lambda: multiply(*arguments),
        "power": lambda: power(*arguments),
        "negate": lambda: negate(*arguments),
    }
    if operation not in constructors:
        raise MathPackBoundaryError("unregistered expression operation")
    return constructors[operation]()


def _formula_from_data(value: Mapping[str, Any]) -> Formula:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise MathPackBoundaryError("formula must be an object with a kind")
    kind = value["kind"]
    if kind == "equation":
        _exact_keys(value, {"kind", "left", "right"}, "equation")
        return Equation(_expression_from_data(value["left"]), _expression_from_data(value["right"]))
    if kind == "inequality":
        _exact_keys(value, {"kind", "left", "right", "relation"}, "inequality")
        try:
            relation = InequalityRelation(value["relation"])
        except ValueError as error:
            raise MathPackBoundaryError("inequality relation is unregistered") from error
        return Inequality(
            _expression_from_data(value["left"]),
            _expression_from_data(value["right"]),
            relation,
        )
    if kind == "recurrence":
        _exact_keys(
            value,
            {"kind", "sequence", "index", "order", "equation", "initial_conditions"},
            "recurrence",
        )
        if not isinstance(value["initial_conditions"], list):
            raise MathPackBoundaryError("recurrence initial conditions must be an array")
        equation = _formula_from_data(value["equation"])
        if not isinstance(equation, Equation):
            raise MathPackBoundaryError("recurrence body must be an equation")
        initial = []
        for item in value["initial_conditions"]:
            _exact_keys(item, {"index", "value"}, "initial condition")
            index = item["index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise MathPackBoundaryError("initial-condition index must be an integer")
            initial.append((index, _expression_from_data(item["value"])))
        return Recurrence(
            str(value["sequence"]),
            _expression_from_data(value["index"]),
            value["order"],
            equation,
            tuple(initial),
        )
    raise MathPackBoundaryError("unregistered formula kind")


def _symbols(expression: Expression) -> set[str]:
    result = {str(expression.value)} if expression.operation == "symbol" else set()
    for argument in expression.arguments:
        result.update(_symbols(argument))
    return result


def _formula_symbols(formula: Formula) -> set[str]:
    if isinstance(formula, (Equation, Inequality)):
        return _symbols(formula.left) | _symbols(formula.right)
    if isinstance(formula, Recurrence):
        result = _symbols(formula.index) | _formula_symbols(formula.equation)
        for _, expression in formula.initial_conditions:
            result.update(_symbols(expression))
        return result
    raise MathPackBoundaryError("unregistered formula node")


def _exact_value_data(value: int | Fraction) -> int | dict[str, int]:
    return value if isinstance(value, int) else _fraction_data(value)


def _exact_value_from_data(value: Any) -> int | Fraction:
    if isinstance(value, bool):
        raise MathPackBoundaryError("boolean assignment is not a scalar")
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return _fraction_from_data(value, "assignment rational")
    raise MathPackBoundaryError("assignment value must be integer or rational")


def math_candidate_representation(
    formula: Formula,
    variables: Mapping[str, MathType],
    *,
    exact_assignments: Sequence[Mapping[str, int | Fraction]] = (),
    random_trials: int = 64,
    adversarial_limit: int = 256,
    seed: int = 0,
    proof_method: str = "sympy_exact_identity",
    prior_art_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact closed representation consumed by :class:`MathDomainPack`."""

    if any(not isinstance(name, str) or not name.isidentifier() for name in variables):
        raise MathPackBoundaryError("variable names must be identifiers")
    return {
        "schema_version": "sigma-math-candidate-1.0",
        "formula": formula_to_data(formula),
        "variables": {name: _type_data(variables[name]) for name in sorted(variables)},
        "canonical_formula_sha256": formula_sha256(formula),
        "counterexample_plan": {
            "exact_assignments": [
                {name: _exact_value_data(assignment[name]) for name in sorted(assignment)}
                for assignment in exact_assignments
            ],
            "strategies": ["exact", "adversarial", "random"],
            "random_trials": random_trials,
            "adversarial_limit": adversarial_limit,
            "seed": seed,
        },
        "proof_contract": {"method": proof_method},
        "prior_art_contract": {
            "receipt_sha256": prior_art_receipt_sha256,
            "comparison_after_proof_only": True,
        },
    }


def math_pack_descriptor() -> DomainPackDescriptor:
    stages = tuple(
        StageDefinition(
            stage_id,
            ordinal,
            _KINDS,
            tuple(sorted(_STAGE_IDS[:ordinal])),
        )
        for ordinal, stage_id in enumerate(_STAGE_IDS)
    )
    gates = tuple(
        sorted(
            (
                GateDefinition(
                    f"admit_{stage.stage_id}",
                    None if stage.ordinal == 0 else _STAGE_IDS[stage.ordinal - 1],
                    stage.stage_id,
                    _KINDS,
                    tuple(sorted(_STAGE_IDS[: stage.ordinal + 1])),
                )
                for stage in stages
            ),
            key=lambda item: item.gate_id,
        )
    )
    return DomainPackDescriptor("sigma.math", "1.0.0", _KINDS, stages, gates)


class MathDomainPack:
    """First exact Math Pack implementation for scalar formula artifacts."""

    def __init__(self) -> None:
        self._descriptor = math_pack_descriptor()

    @property
    def descriptor(self) -> DomainPackDescriptor:
        return self._descriptor

    @staticmethod
    def _decode(
        artifact: CandidateArtifact,
    ) -> tuple[Formula, dict[str, MathType], Mapping[str, Any]]:
        representation = artifact.representation
        _exact_keys(
            representation,
            {
                "schema_version",
                "formula",
                "variables",
                "canonical_formula_sha256",
                "counterexample_plan",
                "proof_contract",
                "prior_art_contract",
            },
            "math candidate",
        )
        if representation["schema_version"] != "sigma-math-candidate-1.0":
            raise MathPackBoundaryError("math candidate schema version changed")
        if not isinstance(representation["variables"], Mapping):
            raise MathPackBoundaryError("variables must be an object")
        variables = {
            name: _type_from_data(value)
            for name, value in sorted(representation["variables"].items())
            if isinstance(name, str) and name.isidentifier()
        }
        if set(variables) != set(representation["variables"]):
            raise MathPackBoundaryError("variable names must be identifiers")
        formula = _formula_from_data(representation["formula"])
        return formula, variables, representation

    def evaluate_stage(
        self,
        artifact: CandidateArtifact,
        stage: StageDefinition,
        prior_outcomes: Mapping[str, StageOutcome],
    ) -> StageOutcome:
        del prior_outcomes
        formula, variables, representation = self._decode(artifact)
        if stage.stage_id == "typed":
            undeclared = sorted(_formula_symbols(formula) - set(variables))
            passed = not undeclared
            check = CheckResult.create(
                "typed_formula",
                passed,
                {"declared_variables": sorted(variables), "undeclared_symbols": undeclared},
            )
            return StageOutcome.create(
                stage.stage_id,
                artifact.ref,
                OutcomeStatus.PASS if passed else OutcomeStatus.REJECT,
                (check,),
                reason_codes=() if passed else ("undeclared_symbols",),
            )
        if stage.stage_id == "canonicalized":
            expected = formula_sha256(formula)
            passed = representation["canonical_formula_sha256"] == expected
            check = CheckResult.create(
                "canonical_formula_identity",
                passed,
                {"expected_sha256": expected},
            )
            return StageOutcome.create(
                stage.stage_id,
                artifact.ref,
                OutcomeStatus.PASS if passed else OutcomeStatus.REJECT,
                (check,),
                reason_codes=() if passed else ("canonical_identity_mismatch",),
            )
        if stage.stage_id == "counterexample_screened":
            plan = representation["counterexample_plan"]
            _exact_keys(
                plan,
                {
                    "exact_assignments",
                    "strategies",
                    "random_trials",
                    "adversarial_limit",
                    "seed",
                },
                "counterexample plan",
            )
            if not isinstance(plan["exact_assignments"], list) or not isinstance(
                plan["strategies"], list
            ):
                raise MathPackBoundaryError("counterexample plan arrays are malformed")
            try:
                strategies = tuple(SearchStrategy(item) for item in plan["strategies"])
            except ValueError as error:
                raise MathPackBoundaryError("counterexample strategy is unregistered") from error
            assignments = tuple(
                {name: _exact_value_from_data(value) for name, value in item.items()}
                for item in plan["exact_assignments"]
                if isinstance(item, Mapping)
            )
            if len(assignments) != len(plan["exact_assignments"]):
                raise MathPackBoundaryError("exact assignment is not an object")
            report = find_counterexample(
                formula,
                variables,
                exact_assignments=assignments,
                strategies=strategies,
                random_trials=plan["random_trials"],
                adversarial_limit=plan["adversarial_limit"],
                seed=plan["seed"],
            )
            passed = report.status is SearchStatus.INCONCLUSIVE_WITHIN_BUDGET
            check = CheckResult.create(
                "bounded_counterexample_search",
                passed,
                {
                    "status": report.status.value,
                    "trials_run": report.trials_run,
                    "proves_formula": report.proves_formula,
                },
            )
            return StageOutcome.create(
                stage.stage_id,
                artifact.ref,
                OutcomeStatus.PASS
                if passed
                else OutcomeStatus.REJECT
                if report.status is SearchStatus.COUNTEREXAMPLE_FOUND
                else OutcomeStatus.BLOCK,
                (check,),
                reason_codes=()
                if passed
                else (
                    "counterexample_found"
                    if report.status is SearchStatus.COUNTEREXAMPLE_FOUND
                    else "counterexample_search_unsupported",
                ),
            )
        if stage.stage_id == "exactly_verified":
            proof = representation["proof_contract"]
            _exact_keys(proof, {"method"}, "proof contract")
            supported = proof["method"] == "sympy_exact_identity" and isinstance(formula, Equation)
            proved = supported and sp.cancel(to_sympy(formula.left) - to_sympy(formula.right)) == 0
            check = CheckResult.create(
                "exact_proof",
                proved,
                {"method": proof["method"], "supported": supported},
            )
            return StageOutcome.create(
                stage.stage_id,
                artifact.ref,
                OutcomeStatus.PASS if proved else OutcomeStatus.BLOCK,
                (check,),
                reason_codes=() if proved else ("exact_proof_not_closed",),
            )
        if stage.stage_id == "prior_art_checked":
            contract = representation["prior_art_contract"]
            _exact_keys(
                contract,
                {"receipt_sha256", "comparison_after_proof_only"},
                "prior-art contract",
            )
            receipt = contract["receipt_sha256"]
            source = next(
                (item for item in artifact.provenance.sources if item.role == "prior_art_receipt"),
                None,
            )
            passed = (
                contract["comparison_after_proof_only"] is True
                and isinstance(receipt, str)
                and _HASH.fullmatch(receipt) is not None
                and source is not None
                and source.file_sha256 == receipt
            )
            check = CheckResult.create(
                "post_proof_prior_art_receipt",
                passed,
                {"comparison_after_proof_only": contract["comparison_after_proof_only"]},
            )
            return StageOutcome.create(
                stage.stage_id,
                artifact.ref,
                OutcomeStatus.PASS if passed else OutcomeStatus.BLOCK,
                (check,),
                evidence=() if source is None else (source,),
                reason_codes=() if passed else ("prior_art_receipt_missing_or_unbound",),
            )
        raise MathPackBoundaryError("unregistered Math Pack stage")

    def evaluate_gate(
        self,
        artifact: CandidateArtifact,
        gate: GateDefinition,
        stage_outcomes: Mapping[str, StageOutcome],
    ) -> GateOutcome:
        expected = tuple(stage_outcomes[name].ref for name in sorted(stage_outcomes))
        check = CheckResult.create(
            "stage_receipts_bound",
            True,
            {"gate_id": gate.gate_id, "stages": sorted(stage_outcomes)},
        )
        return GateOutcome.create(
            gate.gate_id,
            artifact.ref,
            OutcomeStatus.PASS,
            expected,
            (check,),
        )


__all__ = [
    "MathDomainPack",
    "MathPackBoundaryError",
    "math_candidate_representation",
    "math_pack_descriptor",
]
