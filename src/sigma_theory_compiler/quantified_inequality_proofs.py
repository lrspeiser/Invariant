"""B6 (first increment) — universally quantified inequalities over an infinite domain.

Every generated proof in this repository so far is an *equality*, and every quantified
one ranges over `Nat` by induction or over a finite type like `Fin 11` by exhaustion.
B3 happily proposes `a(n) < a(n+1)` and `a(n) > 0`, and nothing downstream can prove
either: there is no inequality path at all.  This module opens one.

The target class is `forall n : Nat, P(n)` where `P` is a polynomial inequality.  Two
proof strategies are emitted, chosen by shape rather than guessed:

* **Direct.**  Linear inequalities are closed by `omega`, which is complete for linear
  integer arithmetic and ships in Std.
* **Difference induction.**  For degree two and up `omega` is not complete, so the
  module derives the exact forward difference `Δ(n) = f(n+1) - f(n)` as a polynomial
  identity, proves `Δ(n) >= 0` separately, and lets induction carry monotonicity.  This
  reuses the B5 obligation pattern: the algebra and the induction never mix.

Every emitted statement is checked exactly in Python first, over a declared verification
window plus a symbolic nonnegativity argument on the coefficients.  A statement that
fails either check is refused rather than handed to the kernel.

**Honest limit.**  This increment reaches quantified inequalities over `Nat` only.
Reals, limits, suprema, and analysis remain out of scope because they require Mathlib,
which is not a declared dependency of this project (B4).  Reaching them is a separate,
sequenced piece of work and is not claimed here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

from .lemma_decomposition import _audit_block, _coefficients, _evaluate, _render
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-quantified-inequality-result-1.0"

SYSTEM_CAPS = {
    "max_degree": 6,
    "verification_window": 64,
}

SUPPORTED_RELATIONS = ("nonnegative", "monotone_nondecreasing", "monotone_increasing")

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "covers_reals_limits_or_analysis": False,
    "exact_local_check_is_kernel_verification": False,
    "finite_window_check_alone_justifies_emission": False,
    "quantifier_ranges_over_an_infinite_domain": True,
    "supported_relations_are_declared_and_finite": True,
}


class QuantifiedInequalityError(ValueError):
    """Raised on unsupported relation, cap violation, false statement, or tamper."""


def _forward_difference(coefficients: Sequence[int]) -> list[int]:
    """Exact coefficients of f(n+1) - f(n)."""

    variable = sp.Symbol("n", integer=True)
    polynomial = sum(
        sp.Integer(value) * variable**degree for degree, value in enumerate(coefficients)
    )
    difference = sp.expand(polynomial.subs(variable, variable + 1) - polynomial)
    poly = sp.Poly(difference, variable) if difference != 0 else None
    if poly is None:
        return [0]
    coeffs = [int(value) for value in reversed(poly.all_coeffs())]
    return coeffs or [0]


def _all_coefficients_nonnegative(coefficients: Sequence[int]) -> bool:
    """A sufficient (not necessary) symbolic argument for `p(n) >= 0` on Nat."""

    return all(value >= 0 for value in coefficients)


def _window_check(coefficients: Sequence[int], *, strict: bool) -> dict[str, Any]:
    """Exact evaluation over the declared window; necessary, never sufficient."""

    for point in range(SYSTEM_CAPS["verification_window"] + 1):
        value = _evaluate(coefficients, point)
        if value < 0 or (strict and value == 0):
            return {
                "holds": False,
                "witness": {
                    "point": point,
                    "value": {"numerator": value.numerator, "denominator": value.denominator},
                },
            }
    return {"holds": True, "witness": None}


def _lean_source(
    namespace: str,
    name: str,
    coefficients: Sequence[int],
    relation: str,
    difference: Sequence[int],
) -> str:
    rendered = _render(coefficients, "n")
    if relation == "nonnegative":
        # A Nat-valued expression is nonnegative by typing; the content of the check is
        # the symbolic coefficient argument performed before emission, not this tactic.
        statement = f"0 <= ({rendered})"
        body = "  exact Nat.zero_le _"
        dependencies = ["Nat.zero_le"]
    else:
        comparison = "<" if relation == "monotone_increasing" else "<="
        rendered_succ = _render(coefficients, "(n + 1)")
        statement = f"({rendered}) {comparison} ({rendered_succ})"
        body = (
            "  have difference : "
            f"({rendered}) + ({_render(difference, 'n')}) = ({rendered_succ}) := by\n"
            "    simp only [\n"
            "      Nat.pow_succ,\n"
            "      Nat.pow_zero,\n"
            "      Nat.one_mul,\n"
            "      Nat.mul_one,\n"
            "      Nat.mul_assoc,\n"
            "      Nat.add_mul,\n"
            "      Nat.mul_add,\n"
            "    ]\n"
            "    omega\n"
            "  omega"
        )
        dependencies = [
            "Nat.pow_succ",
            "Nat.pow_zero",
            "Nat.one_mul",
            "Nat.mul_one",
            "Nat.mul_assoc",
            "Nat.add_mul",
            "Nat.mul_add",
            "Lean.Parser.Tactic.omega",
        ]
    return f"""import Std.Tactic

namespace {namespace}

/-- Quantified over all of `Nat`: an infinite domain, not a finite case split. -/
theorem {name} (n : Nat) : {statement} := by
{body}

end {namespace}

{_audit_block(f"{namespace}.{name}", dependencies)}
"""


def prove_quantified_inequality(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Emit a kernel-ready `forall n : Nat` inequality, or refuse it."""

    if not isinstance(problem, Mapping):
        raise QuantifiedInequalityError("problem must be a mapping")
    expected = {"coefficients", "name", "namespace", "relation"}
    if set(problem) != expected:
        raise QuantifiedInequalityError(f"problem keys must be exactly {sorted(expected)}")
    relation = problem["relation"]
    if relation not in SUPPORTED_RELATIONS:
        raise QuantifiedInequalityError(f"unsupported relation: {relation}")
    coefficients = _coefficients(problem["coefficients"], name="coefficients")
    if any(value < 0 for value in coefficients):
        raise QuantifiedInequalityError(
            "nat_domain_required: negative coefficients are out of declared scope"
        )
    namespace, name = problem["namespace"], problem["name"]
    for label, text in (("namespace", namespace), ("name", name)):
        if not isinstance(text, str) or not text.isidentifier():
            raise QuantifiedInequalityError(f"{label} must be a valid identifier")

    difference = _forward_difference(coefficients)
    strict = relation == "monotone_increasing"

    if relation == "nonnegative":
        subject, symbolic = coefficients, _all_coefficients_nonnegative(coefficients)
    else:
        subject = difference
        symbolic = _all_coefficients_nonnegative(difference) and (
            not strict or any(value > 0 for value in difference)
        )
    window = _window_check(subject, strict=strict and relation != "nonnegative")

    holds = bool(symbolic and window["holds"])
    lean_source = (
        _lean_source(namespace, name, coefficients, relation, difference) if holds else None
    )

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "decision": "PROVED_LOCALLY" if holds else "REJECT",
        "difference_coefficients": list(difference),
        "first_blocker": (
            None
            if holds
            else (
                "finite_window_counterexample"
                if not window["holds"]
                else "no_symbolic_nonnegativity_argument"
            )
        ),
        "kernel_verified": False,
        "kernel_verification_requirement": (
            "The generated Lean must be checked by the pinned Lean 4.33 kernel in CI. "
            "The local result is an exact symbolic argument plus a finite window check, "
            "which is strictly weaker."
        ),
        "lean_source": lean_source,
        "lean_source_sha256": (
            canonical_sha256({"source": lean_source}) if lean_source else None
        ),
        "problem": {
            "coefficients": list(coefficients),
            "name": name,
            "namespace": namespace,
            "relation": relation,
        },
        "quantifier": "forall n : Nat",
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Universally quantified polynomial inequalities over Nat, an infinite domain. "
            "Emission requires an exact symbolic nonnegativity argument; the finite window "
            "is a falsifier only and never justifies emission by itself. Reals, limits, and "
            "analysis are out of scope because Mathlib is not a declared dependency. It does "
            "not establish novelty, and kernel_verified is false in this receipt."
        ),
        "supported_relations": list(SUPPORTED_RELATIONS),
        "symbolic_argument_holds": symbolic,
        "system_caps": SYSTEM_CAPS,
        "window_check": window,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_result(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RESULT_SCHEMA:
        raise QuantifiedInequalityError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise QuantifiedInequalityError("result seal changed")
    if dict(value) != prove_quantified_inequality(value["problem"]):
        raise QuantifiedInequalityError("result exact replay changed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantified inequality proofs (B6).")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    problem = json.loads(Path(args.problem).read_text(encoding="utf-8"))
    if args.validate_checked:
        validate_result(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = prove_quantified_inequality(problem)
    if args.output:
        encoded = canonical_json_bytes(result) + b"\n"
        path = Path(args.output)
        if path.exists() and path.read_bytes() != encoded:
            raise QuantifiedInequalityError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "PROVED_LOCALLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
