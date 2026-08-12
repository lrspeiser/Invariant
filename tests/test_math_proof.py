from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from fractions import Fraction

import pytest

from sigma_theory_compiler.math_expression_ir import Equation, Recurrence, call, literal, symbol
from sigma_theory_compiler.math_proof import (
    ProofFailure,
    ProofValidationError,
    UnsupportedProof,
    prove_induction,
    prove_rational_identity,
    validate_induction_certificate,
    validate_rational_identity_certificate,
)
from sigma_theory_compiler.math_types import INTEGER


def _reseal(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    value["content_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_exact_rational_identity_certificate_is_reproducible_and_checkable() -> None:
    x = symbol("x")
    statement = Equation((x**2 - 1) / (x - 1), x + 1)

    first = prove_rational_identity(statement)
    second = prove_rational_identity(statement)
    validate_rational_identity_certificate(first, statement)

    assert first == second
    assert first["decision"] == "proved_exact_rational_identity_on_regular_domain"
    assert first["witness"]["cleared_numerator_zero"] is True
    assert first["variables"] == ["x"]
    assert "denominator_nonvanishing_domain_is_not_certified" in first["limitations"]


def test_false_and_unsupported_rational_claims_fail_closed() -> None:
    x = symbol("x")
    with pytest.raises(ProofFailure, match="nonzero numerator"):
        prove_rational_identity(Equation(x + 1, x))
    with pytest.raises(UnsupportedProof, match="integer/rational"):
        prove_rational_identity(Equation(x + 0.1, x + 0.1))
    with pytest.raises(UnsupportedProof, match="uninterpreted functions"):
        prove_rational_identity(Equation(call("f", x), call("f", x)))


def test_identity_certificate_rejects_resealed_and_unknown_mutations() -> None:
    x = symbol("x")
    statement = Equation((x + 1) ** 2, x**2 + 2 * x + 1)
    certificate = prove_rational_identity(statement)

    changed = deepcopy(certificate)
    changed["decision"] = "proved_everywhere"
    _reseal(changed)
    with pytest.raises(ProofValidationError):
        validate_rational_identity_certificate(changed, statement)

    extra = deepcopy(certificate)
    extra["extra"] = False
    with pytest.raises(ProofValidationError, match="schema"):
        validate_rational_identity_certificate(extra, statement)


def _triangular_induction() -> tuple[Equation, Recurrence]:
    n = symbol("n", INTEGER)
    recurrence = Recurrence(
        sequence="triangular",
        index=n,
        order=1,
        equation=Equation(
            call("triangular", n + 1),
            call("triangular", n) + n + 1,
        ),
        initial_conditions=((0, literal(0)),),
    )
    statement = Equation(call("triangular", n), n * (n + 1) * Fraction(1, 2))
    return statement, recurrence


def test_induction_certificate_checks_base_and_symbolic_successor() -> None:
    statement, recurrence = _triangular_induction()

    certificate = prove_induction(statement, recurrence, base_index=0)
    validate_induction_certificate(certificate, statement, recurrence)

    assert certificate["decision"] == "proved_by_base_and_symbolic_successor_identity"
    assert certificate["obligations"]["base"]["witness"]["cleared_numerator_zero"]
    assert certificate["obligations"]["successor"]["witness"]["cleared_numerator_zero"]
    assert certificate["scope"] == (
        "all integer indices reachable from the base by repeated regular successor steps"
    )
    assert "recurrence_and_statement_denominators_must_remain_nonzero" in certificate["limitations"]


def test_induction_rejects_bad_base_bad_successor_and_higher_order() -> None:
    statement, recurrence = _triangular_induction()
    bad_base = Recurrence(
        recurrence.sequence,
        recurrence.index,
        recurrence.order,
        recurrence.equation,
        ((0, literal(1)),),
    )
    with pytest.raises(ProofFailure, match="nonzero numerator"):
        prove_induction(statement, bad_base, base_index=0)

    n = recurrence.index
    bad_step = Recurrence(
        recurrence.sequence,
        n,
        1,
        Equation(call("triangular", n + 1), call("triangular", n) + n + 2),
        recurrence.initial_conditions,
    )
    with pytest.raises(ProofFailure, match="nonzero numerator"):
        prove_induction(statement, bad_step, base_index=0)

    higher_order = Recurrence(
        recurrence.sequence,
        n,
        2,
        recurrence.equation,
        recurrence.initial_conditions,
    )
    with pytest.raises(UnsupportedProof, match="first-order"):
        prove_induction(statement, higher_order, base_index=0)


def test_induction_certificate_rejects_tampering_and_claim_mismatch() -> None:
    statement, recurrence = _triangular_induction()
    certificate = prove_induction(statement, recurrence, base_index=0)

    changed = deepcopy(certificate)
    changed["obligations"]["base"]["initial_condition_index"] = 1
    _reseal(changed)
    with pytest.raises(ProofValidationError):
        validate_induction_certificate(changed, statement, recurrence)

    n = recurrence.index
    different = Equation(call("triangular", n), n * (n + 1) * Fraction(1, 3))
    with pytest.raises(ProofValidationError):
        validate_induction_certificate(certificate, different, recurrence)
