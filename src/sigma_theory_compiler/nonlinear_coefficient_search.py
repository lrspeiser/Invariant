"""B2 — exact nonlinear coefficient search.

B1 searches bases whose coefficients enter *linearly*, so the solve is exact rational
linear algebra.  That excludes an entire class of ordinary structure: anything with a
parameter inside an exponent, a denominator, or a nested argument.  `a*b^n` is not a
linear-basis problem in `(a, b)`; neither is `(a*n+b)/(n+d)`.

This module solves those exactly.  Each declared model is a closed-form template with
unknown parameters.  Rows are turned into a polynomial system over the rationals, solved
by exact elimination, and filtered to exact rational parameter assignments.  Irrational
and complex roots are discarded rather than rounded: a result is exact or it is absent.

The acceptance discipline is inherited from B1 unchanged, because it is the part that
makes a result mean something:

* consume only the minimum rows needed to determine the parameters;
* every remaining row is untouched holdout the solution must predict exactly;
* fewer than `min_confirmations` holdout rows is refused as fitting, not discovery;
* models are Occam-ordered and the accepted model carries exact rejections of every
  strictly simpler one.

Claim boundary: the model list is declared and finite.  A BLOCK means "not in this
declared model set", never "no closed form exists".
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .basis_synthesis import BasisSynthesisError, _fraction_data, _parse_rows
from .sigma_core import canonical_json_bytes, canonical_sha256

MODEL_SCHEMA = "invariant-nonlinear-model-set-1.0"
RESULT_SCHEMA = "invariant-nonlinear-coefficient-search-result-1.0"

SYSTEM_CAPS = {
    "max_rows": 64,
    "max_parameters": 4,
    "max_abs_point": 64,
    "min_confirmations": 2,
    "max_candidate_solutions": 64,
    "exponent_search_bound": 6,
}

CLAIMS = {
    "approximate_or_rounded_roots_accepted": False,
    "corpus_absence_establishes_novelty": False,
    "holdout_confirmation_required": True,
    "interpolation_accepted_as_discovery": False,
    "model_set_is_declared_and_finite": True,
    "unbounded_functional_search": False,
}


class NonlinearSearchError(ValueError):
    """Raised on malformed input, cap violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Declared model set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Model:
    model_id: str
    rank: int
    parameters: tuple[str, ...]
    template: str

    @property
    def arity(self) -> int:
        return len(self.parameters)


def _model_expression(model: Model, point: sp.Expr, values: Mapping[str, sp.Expr]) -> sp.Expr:
    """Symbolic value of `model` at `point` under a parameter assignment."""

    get = values.__getitem__
    if model.model_id == "pure_geometric":
        return get("a") * get("b") ** point
    if model.model_id == "shifted_geometric":
        return get("a") * get("b") ** point + get("c")
    if model.model_id == "geometric_linear_argument":
        return get("a") * get("b") ** point * (point + get("c"))
    if model.model_id == "linear_fractional":
        return (get("a") * point + get("b")) / (point + get("d"))
    if model.model_id == "reciprocal_affine":
        return get("a") / (get("b") * point + get("c"))
    raise NonlinearSearchError(f"unsupported model: {model.model_id}")


#: Ordered by (arity, rank).  Ordering is the Occam preference and must stay frozen.
MODELS: tuple[Model, ...] = tuple(
    sorted(
        (
            Model("pure_geometric", 0, ("a", "b"), "a*b^n"),
            Model("shifted_geometric", 1, ("a", "b", "c"), "a*b^n + c"),
            Model("geometric_linear_argument", 2, ("a", "b", "c"), "a*b^n*(n + c)"),
            Model("linear_fractional", 3, ("a", "b", "d"), "(a*n + b)/(n + d)"),
            Model("reciprocal_affine", 4, ("a", "b", "c"), "a/(b*n + c)"),
        ),
        key=lambda model: (model.arity, model.rank, model.model_id),
    )
)

#: Power-law exponents are searched over a declared bounded integer range rather than
#: solved transcendentally, so every reported exponent is exact.
POWER_LAW_EXPONENTS = tuple(
    exponent
    for exponent in range(-SYSTEM_CAPS["exponent_search_bound"], SYSTEM_CAPS["exponent_search_bound"] + 1)
    if exponent not in (0, 1)
)


# ---------------------------------------------------------------------------
# Exact solving
# ---------------------------------------------------------------------------


def _exact_rational(value: sp.Expr) -> Fraction | None:
    """Return an exact Fraction, or None for irrational/complex/undefined values."""

    if value is None or value.free_symbols:
        return None
    try:
        simplified = sp.nsimplify(sp.simplify(value), rational=True)
    except (TypeError, ValueError, sp.SympifyError):
        return None
    if not simplified.is_number or not simplified.is_real:
        return None
    if not isinstance(simplified, sp.Rational):
        return None
    return Fraction(int(simplified.p), int(simplified.q))


def _solve_model(
    model: Model, rows: Sequence[tuple[int, Fraction]]
) -> list[dict[str, Fraction]]:
    """Exact rational parameter assignments consistent with the leading rows."""

    symbols = {name: sp.Symbol(name) for name in model.parameters}
    equations = []
    for point, value in rows[: model.arity]:
        expression = _model_expression(model, sp.Integer(point), symbols)
        equations.append(sp.Eq(expression, sp.Rational(value.numerator, value.denominator)))
    try:
        raw = sp.solve(
            equations, list(symbols.values()), dict=True, rational=True, manual=False
        )
    except (NotImplementedError, sp.SympifyError, TypeError, ValueError, ZeroDivisionError):
        return []
    if not isinstance(raw, list):
        return []
    assignments: list[dict[str, Fraction]] = []
    for solution in raw[: SYSTEM_CAPS["max_candidate_solutions"]]:
        if not isinstance(solution, dict):
            continue
        exact: dict[str, Fraction] = {}
        for name, symbol in symbols.items():
            value = solution.get(symbol)
            if value is None:
                break
            rational = _exact_rational(value)
            if rational is None:
                break
            exact[name] = rational
        if len(exact) == model.arity:
            assignments.append(exact)
    # Canonical ordering keeps the receipt byte-stable across sympy versions.
    assignments.sort(key=lambda item: [(item[name].numerator, item[name].denominator) for name in model.parameters])
    return assignments


def _evaluate(model: Model, assignment: Mapping[str, Fraction], point: int) -> Fraction | None:
    """Exact value of the instantiated model, or None where it is undefined."""

    get = assignment.__getitem__
    try:
        if model.model_id == "pure_geometric":
            return get("a") * get("b") ** point
        if model.model_id == "shifted_geometric":
            return get("a") * get("b") ** point + get("c")
        if model.model_id == "geometric_linear_argument":
            return get("a") * get("b") ** point * (Fraction(point) + get("c"))
        if model.model_id == "linear_fractional":
            denominator = Fraction(point) + get("d")
            return None if denominator == 0 else (get("a") * point + get("b")) / denominator
        if model.model_id == "reciprocal_affine":
            denominator = get("b") * point + get("c")
            return None if denominator == 0 else get("a") / denominator
    except (ZeroDivisionError, ValueError):
        return None
    raise NonlinearSearchError(f"unsupported model: {model.model_id}")


def _confirm(
    model: Model,
    assignment: Mapping[str, Fraction],
    rows: Sequence[tuple[int, Fraction]],
) -> dict[str, Any]:
    """Check the assignment against every row, separating fit rows from holdout."""

    fit_points = {point for point, _ in rows[: model.arity]}
    confirmations = 0
    for point, value in rows:
        predicted = _evaluate(model, assignment, point)
        if predicted is None:
            return {"status": "REJECT", "reason": "model_undefined_on_public_row", "point": point}
        if predicted != value:
            return {
                "status": "REJECT",
                "reason": "holdout_counterexample",
                "counterexample": {
                    "point": point,
                    "predicted": _fraction_data(predicted),
                    "observed": _fraction_data(value),
                    "residual": _fraction_data(predicted - value),
                },
            }
        if point not in fit_points:
            confirmations += 1
    if confirmations < SYSTEM_CAPS["min_confirmations"]:
        return {
            "status": "BLOCK",
            "reason": "insufficient_holdout_confirmations",
            "confirmations": confirmations,
        }
    return {"status": "PASS", "confirmations": confirmations}


def _render(model: Model, assignment: Mapping[str, Fraction]) -> str:
    rendered = model.template
    for name in model.parameters:
        rendered = rendered.replace(name, f"({assignment[name]})")
    return rendered


def _try_power_law(rows: Sequence[tuple[int, Fraction]]) -> dict[str, Any]:
    """`a*n^p` over a declared bounded integer exponent range."""

    outcome = {"model_id": "power_law", "arity": 2, "template": "a*n^p"}
    for exponent in POWER_LAW_EXPONENTS:
        usable = [
            (point, value)
            for point, value in rows
            if not (point == 0 and exponent < 0) and not (point == 0 and exponent > 0 and value != 0)
        ]
        if len(usable) < 1 + SYSTEM_CAPS["min_confirmations"]:
            continue
        anchor_point, anchor_value = usable[0]
        basis = Fraction(anchor_point) ** exponent
        if basis == 0:
            continue
        scale = anchor_value / basis
        confirmations = 0
        matched = True
        for point, value in usable[1:]:
            if scale * Fraction(point) ** exponent != value:
                matched = False
                break
            confirmations += 1
        if matched and confirmations >= SYSTEM_CAPS["min_confirmations"]:
            return {
                **outcome,
                "status": "PASS",
                "reason": "unique_exact_solution_confirmed_on_holdout",
                "parameters": {"a": _fraction_data(scale), "p": exponent},
                "confirmations": confirmations,
                "expression": f"({scale})*n^{exponent}",
            }
    return {**outcome, "status": "REJECT", "reason": "no_exact_bounded_integer_exponent"}


def search_nonlinear(rows: Any) -> dict[str, Any]:
    """Search the declared model set for the simplest exactly-confirmed model."""

    try:
        parsed = _parse_rows(rows)
    except BasisSynthesisError as error:  # one error contract for this module
        raise NonlinearSearchError(str(error)) from error
    rejected: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None

    for model in MODELS:
        outcome: dict[str, Any] = {
            "model_id": model.model_id,
            "arity": model.arity,
            "template": model.template,
        }
        if len(parsed) < model.arity + SYSTEM_CAPS["min_confirmations"]:
            rejected.append({**outcome, "status": "SKIP", "reason": "insufficient_rows_for_model"})
            continue
        assignments = _solve_model(model, parsed)
        if not assignments:
            rejected.append(
                {**outcome, "status": "REJECT", "reason": "no_exact_rational_parameter_solution"}
            )
            continue
        best: dict[str, Any] | None = None
        for assignment in assignments:
            verdict = _confirm(model, assignment, parsed)
            if verdict["status"] == "PASS":
                best = {
                    **outcome,
                    "status": "PASS",
                    "reason": "unique_exact_solution_confirmed_on_holdout",
                    "parameters": {
                        name: _fraction_data(assignment[name]) for name in model.parameters
                    },
                    "confirmations": verdict["confirmations"],
                    "expression": _render(model, assignment),
                }
                break
            best = {**outcome, **verdict}
        if best and best.get("status") == "PASS":
            accepted = best
            break
        rejected.append(best or {**outcome, "status": "REJECT", "reason": "unresolved"})

    if accepted is None:
        power_law = _try_power_law(parsed)
        if power_law["status"] == "PASS":
            accepted = power_law
        else:
            rejected.append(power_law)

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "counts": {
            "declared_models": len(MODELS) + 1,
            "models_rejected_before_acceptance": len(rejected),
            "public_rows": len(parsed),
        },
        "decision": "PASS" if accepted else "BLOCK",
        "first_blocker": None if accepted else "no_qualifying_model_in_declared_set",
        "minimality_certificate": {
            "ordering": "arity_then_model_rank_then_model_id",
            "strictly_simpler_models_rejected": rejected,
        },
        "model_schema": MODEL_SCHEMA,
        "public_rows": [
            {"point": point, "value": _fraction_data(value)} for point, value in parsed
        ],
        "result": accepted,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Exact nonlinear parameter recovery over a declared finite model set with "
            "bounded integer exponent search. PASS means the returned model is the "
            "simplest declared model with an exact rational parameter assignment that "
            "predicts every untouched holdout row. It does not establish novelty, "
            "scientific significance, or that no closed form exists outside this set."
        ),
        "system_caps": SYSTEM_CAPS,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_result(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RESULT_SCHEMA:
        raise NonlinearSearchError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise NonlinearSearchError("result seal changed")
    rows = [
        {"point": row["point"], "value": row["value"]} for row in value.get("public_rows", [])
    ]
    if dict(value) != search_nonlinear(rows):
        raise NonlinearSearchError("result exact replay changed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact nonlinear coefficient search (B2).")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, Mapping) else payload
    if args.validate_checked:
        validate_result(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = search_nonlinear(rows)
    if args.output:
        encoded = canonical_json_bytes(result) + b"\n"
        path = Path(args.output)
        if path.exists() and path.read_bytes() != encoded:
            raise NonlinearSearchError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
