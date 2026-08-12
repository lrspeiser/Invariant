"""Exact, independently checkable proof certificates for Math Pack v1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef

from .math_canonicalizer import canonical_data, to_sympy
from .math_expression_ir import Equation, Expression, Recurrence, formula_to_data

IDENTITY_SCHEMA = "invariant-math-rational-identity-certificate-1.0"
INDUCTION_SCHEMA = "invariant-math-induction-certificate-1.0"

_IDENTITY_KEYS = {
    "canonical_statement_sha256",
    "certificate_kind",
    "content_sha256",
    "decision",
    "limitations",
    "schema_version",
    "scope",
    "statement_sha256",
    "variables",
    "witness",
}
_INDUCTION_KEYS = {
    "base_index",
    "certificate_kind",
    "content_sha256",
    "decision",
    "index_symbol",
    "limitations",
    "obligations",
    "recurrence_sha256",
    "schema_version",
    "scope",
    "statement_sha256",
}
_WITNESS_KEYS = {
    "cleared_denominator_sha256",
    "cleared_numerator_sha256",
    "cleared_numerator_zero",
    "method",
}


class UnsupportedProof(ValueError):
    """Raised when a claim falls outside the deliberately bounded proof algebra."""


class ProofFailure(ValueError):
    """Raised when an exact proof obligation has a nonzero residual."""


class ProofValidationError(ValueError):
    """Raised when a certificate is malformed, altered, or bound to another claim."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_data(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha_expression(value: sp.Expr) -> str:
    return hashlib.sha256(sp.srepr(value).encode("utf-8")).hexdigest()


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _sha_data(result)
    return result


def _verify_seal(certificate: Mapping[str, Any]) -> None:
    supplied = certificate.get("content_sha256")
    body = {key: value for key, value in certificate.items() if key != "content_sha256"}
    if not isinstance(supplied, str) or supplied != _sha_data(body):
        raise ProofValidationError("certificate content hash changed")


def _guard_rational_expression(
    expression: Expression,
    *,
    allowed_sequence: str | None = None,
    allowed_call: sp.Expr | None = None,
) -> None:
    """Admit only rational operations and an optional exact sequence call."""

    if expression.operation == "literal":
        if isinstance(expression.value, bool) or not isinstance(expression.value, (int, Fraction)):
            raise UnsupportedProof("proof algebra accepts only exact integer/rational literals")
        return
    if expression.operation == "symbol":
        return
    if expression.operation == "call":
        if allowed_sequence is None or expression.value != allowed_sequence:
            raise UnsupportedProof(
                "uninterpreted functions are unsupported in this proof obligation"
            )
        if len(expression.arguments) != 1:
            raise UnsupportedProof("induction sequence calls must have exactly one index")
        for argument in expression.arguments:
            _guard_rational_expression(argument)
        if allowed_call is not None and to_sympy(expression) != allowed_call:
            raise UnsupportedProof("induction recurrence contains an unsupported sequence shift")
        return
    if expression.operation == "power":
        exponent = expression.arguments[1]
        if exponent.operation != "literal" or not isinstance(exponent.value, int):
            raise UnsupportedProof("proof algebra permits only integer powers")
    for argument in expression.arguments:
        _guard_rational_expression(
            argument,
            allowed_sequence=allowed_sequence,
            allowed_call=allowed_call,
        )


def _guard_equation(equation: Equation) -> None:
    if not isinstance(equation, Equation):
        raise UnsupportedProof("proof backend accepts equations only")
    _guard_rational_expression(equation.left)
    _guard_rational_expression(equation.right)


def _cleared_witness(residual: sp.Expr, *, method: str) -> dict[str, Any]:
    if residual.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise UnsupportedProof("proof residual contains a non-finite symbolic value")
    together = sp.together(residual)
    numerator, denominator = sp.fraction(together)
    numerator = sp.expand(numerator)
    denominator = sp.expand(denominator)
    if denominator == 0:
        raise UnsupportedProof("proof residual has an identically zero denominator")
    if numerator != 0:
        raise ProofFailure(f"exact proof obligation has nonzero numerator: {numerator}")
    return {
        "method": method,
        "cleared_numerator_zero": True,
        "cleared_numerator_sha256": _sha_expression(numerator),
        "cleared_denominator_sha256": _sha_expression(denominator),
    }


def _statement_sha(statement: Equation) -> str:
    return _sha_data(formula_to_data(statement))


def _identity_certificate(statement: Equation) -> dict[str, Any]:
    _guard_equation(statement)
    left = to_sympy(statement.left)
    right = to_sympy(statement.right)
    witness = _cleared_witness(
        left - right,
        method="sympy_together_expand_exact_numerator_zero",
    )
    variables = sorted(str(item) for item in (left - right).free_symbols)
    return _sealed(
        {
            "schema_version": IDENTITY_SCHEMA,
            "certificate_kind": "exact_rational_identity",
            "decision": "proved_exact_rational_identity_on_regular_domain",
            "statement_sha256": _statement_sha(statement),
            "canonical_statement_sha256": _sha_data(canonical_data(statement)),
            "variables": variables,
            "witness": witness,
            "scope": "exact equality after rational denominator clearing",
            "limitations": [
                "denominator_nonvanishing_domain_is_not_certified",
                "no_analytic_branch_or_limit_claim",
                "no_floating_point_literals",
            ],
        }
    )


def prove_rational_identity(statement: Equation) -> dict[str, Any]:
    """Return a sealed exact certificate for a rational identity."""

    return _identity_certificate(statement)


def validate_rational_identity_certificate(
    certificate: Mapping[str, Any], statement: Equation
) -> None:
    """Recompute and validate every field of a rational-identity certificate."""

    if set(certificate) != _IDENTITY_KEYS:
        raise ProofValidationError("rational identity certificate schema changed")
    if not isinstance(certificate.get("witness"), Mapping) or set(certificate["witness"]) != (
        _WITNESS_KEYS
    ):
        raise ProofValidationError("rational identity witness schema changed")
    _verify_seal(certificate)
    try:
        expected = _identity_certificate(statement)
    except (ProofFailure, UnsupportedProof) as error:
        raise ProofValidationError("bound rational identity no longer proves") from error
    if dict(certificate) != expected:
        raise ProofValidationError("rational identity certificate does not match its statement")


def _sequence_call(name: str, argument: sp.Expr) -> sp.Expr:
    return sp.Function(name)(argument)


def _induction_certificate(
    statement: Equation,
    recurrence: Recurrence,
    base_index: int,
) -> dict[str, Any]:
    if isinstance(base_index, bool) or not isinstance(base_index, int):
        raise UnsupportedProof("induction base index must be an integer")
    if recurrence.order != 1:
        raise UnsupportedProof("Math Pack v1 induction supports first-order recurrences only")
    if recurrence.index.operation != "symbol":
        raise UnsupportedProof("induction index must be a symbol")

    index_name = str(recurrence.index.value)
    index = to_sympy(recurrence.index)
    current_call = _sequence_call(recurrence.sequence, index)
    successor_call = _sequence_call(recurrence.sequence, index + 1)

    _guard_rational_expression(
        statement.left,
        allowed_sequence=recurrence.sequence,
        allowed_call=current_call,
    )
    _guard_rational_expression(statement.right)
    if to_sympy(statement.left) != current_call:
        raise UnsupportedProof(
            "induction statement must have the current sequence term on the left"
        )

    _guard_rational_expression(
        recurrence.equation.left,
        allowed_sequence=recurrence.sequence,
        allowed_call=successor_call,
    )
    if to_sympy(recurrence.equation.left) != successor_call:
        raise UnsupportedProof("recurrence must solve the immediate successor term on the left")
    _guard_rational_expression(
        recurrence.equation.right,
        allowed_sequence=recurrence.sequence,
        allowed_call=current_call,
    )

    initial = [value for position, value in recurrence.initial_conditions if position == base_index]
    if len(initial) != 1:
        raise UnsupportedProof(
            "recurrence must contain exactly one initial value at the base index"
        )
    _guard_rational_expression(initial[0])

    statement_left = to_sympy(statement.left)
    statement_right = to_sympy(statement.right)
    initial_value = to_sympy(initial[0])
    base_residual = (
        (statement_left - statement_right)
        .subs(index, base_index)
        .xreplace({_sequence_call(recurrence.sequence, sp.Integer(base_index)): initial_value})
    )
    base_witness = _cleared_witness(
        base_residual,
        method="initial_condition_substitution_then_exact_numerator_zero",
    )

    recurrence_right = to_sympy(recurrence.equation.right)
    successor_residual = (statement_left - statement_right).subs(index, index + 1)
    successor_residual = successor_residual.xreplace({successor_call: recurrence_right})
    successor_residual = successor_residual.xreplace({current_call: statement_right})
    if successor_residual.atoms(AppliedUndef):
        raise UnsupportedProof("successor obligation retains an uninterpreted sequence term")
    successor_witness = _cleared_witness(
        successor_residual,
        method="recurrence_and_induction_hypothesis_substitution_then_exact_numerator_zero",
    )

    return _sealed(
        {
            "schema_version": INDUCTION_SCHEMA,
            "certificate_kind": "exact_first_order_induction",
            "decision": "proved_by_base_and_symbolic_successor_identity",
            "statement_sha256": _statement_sha(statement),
            "recurrence_sha256": _sha_data(formula_to_data(recurrence)),
            "index_symbol": index_name,
            "base_index": base_index,
            "obligations": {
                "base": {
                    "initial_condition_index": base_index,
                    "witness": base_witness,
                },
                "successor": {
                    "recurrence_rule_sha256": _statement_sha(recurrence.equation),
                    "hypothesis_sha256": _statement_sha(statement),
                    "witness": successor_witness,
                },
            },
            "scope": (
                "all integer indices reachable from the base by repeated regular successor steps"
            ),
            "limitations": [
                "first_order_solved_recurrences_only",
                "no_backward_induction",
                "recurrence_and_statement_denominators_must_remain_nonzero",
                "no_convergence_or_closed_form_discovery_claim",
                "no_floating_point_literals",
            ],
        }
    )


def prove_induction(
    statement: Equation,
    recurrence: Recurrence,
    *,
    base_index: int,
) -> dict[str, Any]:
    """Prove a closed form from a first-order recurrence and one initial value."""

    return _induction_certificate(statement, recurrence, base_index)


def validate_induction_certificate(
    certificate: Mapping[str, Any],
    statement: Equation,
    recurrence: Recurrence,
) -> None:
    """Independently recompute base and successor obligations and validate the seal."""

    if set(certificate) != _INDUCTION_KEYS:
        raise ProofValidationError("induction certificate schema changed")
    obligations = certificate.get("obligations")
    if not isinstance(obligations, Mapping) or set(obligations) != {"base", "successor"}:
        raise ProofValidationError("induction obligations schema changed")
    base = obligations["base"]
    successor = obligations["successor"]
    if (
        not isinstance(base, Mapping)
        or set(base) != {"initial_condition_index", "witness"}
        or not isinstance(base.get("witness"), Mapping)
        or set(base["witness"]) != _WITNESS_KEYS
        or not isinstance(successor, Mapping)
        or set(successor) != {"hypothesis_sha256", "recurrence_rule_sha256", "witness"}
        or not isinstance(successor.get("witness"), Mapping)
        or set(successor["witness"]) != _WITNESS_KEYS
    ):
        raise ProofValidationError("induction obligation witness schema changed")
    _verify_seal(certificate)
    base_index = certificate.get("base_index")
    try:
        expected = _induction_certificate(statement, recurrence, base_index)
    except (ProofFailure, UnsupportedProof) as error:
        raise ProofValidationError("bound induction obligations no longer prove") from error
    if dict(certificate) != expected:
        raise ProofValidationError("induction certificate does not match its bound claim")
