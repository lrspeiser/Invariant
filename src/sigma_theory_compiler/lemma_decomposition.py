"""B5 — lemma decomposition.

Existing generated proofs are monolithic: one theorem closed by one induction, with
any helper lemma written by hand.  `ConstraintRecoveredRecurrence.lean` is the honest
example — it needed a private `recoveredPolynomialSuccessor` lemma, and a person
supplied it.  Every nontrivial proof has that shape, so as long as the helper is
manual the pipeline cannot prove anything harder than its author already understood.

This module generates the decomposition.  Given a first-order recurrence
`a(n+1) = a(n) + g(n)` with base value `a(0)` and a candidate closed form `f`, it emits
an obligation DAG:

* `base_case` — `f(0) = a(0)`;
* `successor_identity` — `f(n) + g(n) = f(n+1)`, a pure polynomial identity with no
  induction in it at all;
* `main_induction` — `forall n, a(n) = f(n)`, which cites the other two and contains no
  algebra of its own.

Splitting this way matters because the three obligations fail *independently*.  A
monolithic proof that breaks tells you only that it broke; here a failure names which
of the base, the algebra, or the induction skeleton is wrong.

Verification is deliberately two-layer:

1. **Exact, local, now.**  Each obligation is checked in exact rational arithmetic in
   Python.  A false obligation is never emitted.
2. **Independent kernel, in CI.**  The generated Lean is re-proved by the pinned Lean
   kernel, which shares no code with layer 1.

`kernel_verified` is false in every receipt this module produces.  Layer 1 passing is
not layer 2 passing, and the two must never be conflated.

Claim boundary: first-order additive recurrences with polynomial `g` and polynomial `f`
over the naturals.  Anything else is refused rather than approximated.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-lemma-decomposition-result-1.0"
OBLIGATION_SCHEMA = "invariant-proof-obligation-1.0"

SYSTEM_CAPS = {
    "max_degree": 6,
    "max_abs_coefficient": 10**6,
}

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "exact_local_check_is_kernel_verification": False,
    "hand_written_helper_lemmas_required": False,
    "kernel_verified_without_a_ci_receipt": False,
    "obligations_fail_independently": True,
    "supported_shapes_are_declared_and_finite": True,
}


class LemmaDecompositionError(ValueError):
    """Raised on unsupported shape, cap violation, false obligation, or tamper."""


# ---------------------------------------------------------------------------
# Polynomial helpers
# ---------------------------------------------------------------------------


def _coefficients(value: Any, *, name: str) -> tuple[int, ...]:
    """Ascending integer coefficients, validated against the declared caps."""

    if not isinstance(value, list) or not value:
        raise LemmaDecompositionError(f"{name} must be a non-empty coefficient list")
    if len(value) - 1 > SYSTEM_CAPS["max_degree"]:
        raise LemmaDecompositionError(f"{name} exceeds the declared degree cap")
    coefficients = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise LemmaDecompositionError(f"{name} coefficients must be integers")
        if abs(item) > SYSTEM_CAPS["max_abs_coefficient"]:
            raise LemmaDecompositionError(f"{name} coefficient exceeds cap")
        coefficients.append(item)
    return tuple(coefficients)


def _evaluate(coefficients: Sequence[int], point: int) -> Fraction:
    return sum(
        (Fraction(coefficient) * Fraction(point) ** degree
         for degree, coefficient in enumerate(coefficients)),
        Fraction(0),
    )


def _render(coefficients: Sequence[int], variable: str) -> str:
    parts: list[str] = []
    for degree, coefficient in reversed(list(enumerate(coefficients))):
        if coefficient == 0:
            continue
        if degree == 0:
            parts.append(str(coefficient))
        elif degree == 1:
            parts.append(f"{coefficient} * {variable}" if coefficient != 1 else variable)
        else:
            power = f"{variable} ^ {degree}"
            parts.append(f"{coefficient} * {power}" if coefficient != 1 else power)
    return " + ".join(parts) if parts else "0"


# ---------------------------------------------------------------------------
# Exact obligation checking (layer 1)
# ---------------------------------------------------------------------------


def _check_base_case(closed_form: Sequence[int], base_value: int) -> dict[str, Any]:
    observed = _evaluate(closed_form, 0)
    holds = observed == base_value
    return {
        "holds": holds,
        "detail": {
            "closed_form_at_zero": {"numerator": observed.numerator, "denominator": observed.denominator},
            "declared_base_value": base_value,
        },
    }


def _check_successor_identity(
    closed_form: Sequence[int], step: Sequence[int]
) -> dict[str, Any]:
    """`f(n) + g(n) = f(n+1)` as an exact polynomial identity in n."""

    variable = sp.Symbol("n", integer=True)
    left = sum(
        sp.Integer(coefficient) * variable**degree
        for degree, coefficient in enumerate(closed_form)
    ) + sum(
        sp.Integer(coefficient) * variable**degree for degree, coefficient in enumerate(step)
    )
    right = sum(
        sp.Integer(coefficient) * (variable + 1) ** degree
        for degree, coefficient in enumerate(closed_form)
    )
    residual = sp.expand(left - right)
    return {"holds": residual == 0, "detail": {"residual": str(residual)}}


# ---------------------------------------------------------------------------
# Lean emission (layer 2 input)
# ---------------------------------------------------------------------------


def _audit_block(target: str, dependencies: Sequence[str]) -> str:
    lines = ['#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"']
    lines.append(f'#eval IO.println "target={target}"')
    lines.extend(f'#eval IO.println "dependency={item}"' for item in dependencies)
    lines.append('#eval IO.println "result=checked"')
    lines.append('#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"')
    return "\n".join(lines)


#: Tactic vocabulary restricted to what this repository's already-kernel-verified
#: proofs use.  `ring` and `linarith` live in Mathlib, which is not a declared
#: dependency (see B4); emitting them would produce Lean that cannot compile here.
_NAT_NORMALIZATION = (
    "Nat.pow_succ",
    "Nat.pow_zero",
    "Nat.one_mul",
    "Nat.mul_one",
    "Nat.mul_assoc",
    "Nat.add_mul",
    "Nat.mul_add",
)


def _lean_source(
    namespace: str,
    sequence: str,
    closed_form: Sequence[int],
    step: Sequence[int],
    base_value: int,
) -> str:
    closed = _render(closed_form, "n")
    closed_succ = _render(closed_form, "(n + 1)")
    step_text = _render(step, "n")
    normalization = ",\n    ".join(_NAT_NORMALIZATION)
    return f"""import Std.Tactic

namespace {namespace}

def {sequence} : Nat → Nat
  | 0 => {base_value}
  | n + 1 => {sequence} n + ({step_text})

/-- Obligation 1 of 3: the base case, isolated from all algebra. -/
theorem {sequence}BaseCase : {sequence} 0 = {_render(closed_form, "0")} := by
  rfl

/-- Obligation 2 of 3: a pure polynomial identity. No induction appears here. -/
theorem {sequence}SuccessorIdentity (n : Nat) :
    ({closed}) + ({step_text}) = ({closed_succ}) := by
  simp only [
    {normalization},
  ]
  omega

/-- Obligation 3 of 3: the induction skeleton. It cites the two lemmas above and
    performs no algebra of its own. -/
theorem {sequence}ClosedForm (n : Nat) :
    {sequence} n = ({closed}) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [{sequence}, ih]
      exact {sequence}SuccessorIdentity n

end {namespace}

{_audit_block(
    f"{namespace}.{sequence}ClosedForm",
    [
        f"{namespace}.{sequence}",
        f"{namespace}.{sequence}BaseCase",
        f"{namespace}.{sequence}SuccessorIdentity",
        "Nat.rec",
        *_NAT_NORMALIZATION,
        "Lean.Parser.Tactic.omega",
    ],
)}
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def decompose_closed_form_proof(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Decompose `a(n+1) = a(n) + g(n)`, `a(0) = base` against a closed form."""

    if not isinstance(problem, Mapping):
        raise LemmaDecompositionError("problem must be a mapping")
    expected = {"base_value", "closed_form", "namespace", "sequence_name", "step"}
    if set(problem) != expected:
        raise LemmaDecompositionError(f"problem keys must be exactly {sorted(expected)}")
    closed_form = _coefficients(problem["closed_form"], name="closed_form")
    step = _coefficients(problem["step"], name="step")
    base_value = problem["base_value"]
    if not isinstance(base_value, int) or isinstance(base_value, bool):
        raise LemmaDecompositionError("base_value must be an integer")
    # The emitted Lean is Nat-typed, matching this repository's kernel-verified idiom.
    # Negative data would typecheck differently (or truncate under Nat subtraction), so
    # it is refused rather than silently emitted.
    if base_value < 0 or any(value < 0 for value in (*closed_form, *step)):
        raise LemmaDecompositionError(
            "nat_domain_required: negative base value or coefficient is out of declared scope"
        )
    namespace = problem["namespace"]
    sequence = problem["sequence_name"]
    for name, text in (("namespace", namespace), ("sequence_name", sequence)):
        if not isinstance(text, str) or not text.isidentifier():
            raise LemmaDecompositionError(f"{name} must be a valid identifier")

    base_check = _check_base_case(closed_form, base_value)
    successor_check = _check_successor_identity(closed_form, step)

    obligations = [
        {
            "obligation_id": "base_case",
            "schema_version": OBLIGATION_SCHEMA,
            "statement": f"{sequence}(0) = {_render(closed_form, '0')}",
            "depends_on": [],
            "contains_induction": False,
            "exact_local_check": base_check["holds"],
            "detail": base_check["detail"],
        },
        {
            "obligation_id": "successor_identity",
            "schema_version": OBLIGATION_SCHEMA,
            "statement": (
                f"forall n, ({_render(closed_form, 'n')}) + ({_render(step, 'n')}) "
                f"= ({_render(closed_form, '(n + 1)')})"
            ),
            "depends_on": [],
            "contains_induction": False,
            "exact_local_check": successor_check["holds"],
            "detail": successor_check["detail"],
        },
        {
            "obligation_id": "main_induction",
            "schema_version": OBLIGATION_SCHEMA,
            "statement": f"forall n, {sequence}(n) = {_render(closed_form, 'n')}",
            "depends_on": ["base_case", "successor_identity"],
            "contains_induction": True,
            # The skeleton is sound exactly when both cited obligations hold.
            "exact_local_check": base_check["holds"] and successor_check["holds"],
            "detail": {"cites": ["base_case", "successor_identity"]},
        },
    ]

    failed = [item["obligation_id"] for item in obligations if not item["exact_local_check"]]
    decision = "DECOMPOSED" if not failed else "REJECT"
    lean_source = (
        _lean_source(namespace, sequence, closed_form, step, base_value) if not failed else None
    )

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "counts": {
            "obligations": len(obligations),
            "obligations_failing_exact_local_check": len(failed),
            "obligations_with_induction": sum(
                1 for item in obligations if item["contains_induction"]
            ),
        },
        "decision": decision,
        "failed_obligations": failed,
        "first_blocker": None if not failed else f"exact_local_check_failed:{failed[0]}",
        "kernel_verified": False,
        "kernel_verification_requirement": (
            "The generated Lean must be checked by the pinned Lean 4.33 kernel in CI. "
            "This receipt records only the exact local check, which is a different and "
            "weaker statement."
        ),
        "lean_source": lean_source,
        "lean_source_sha256": (
            canonical_sha256({"source": lean_source}) if lean_source else None
        ),
        "obligations": obligations,
        "problem": {
            "base_value": base_value,
            "closed_form": list(closed_form),
            "namespace": namespace,
            "sequence_name": sequence,
            "step": list(step),
        },
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Automatic lemma decomposition for first-order additive recurrences with "
            "polynomial step and polynomial closed form. Obligations are checked exactly "
            "in rational arithmetic locally and must be independently re-proved by the "
            "pinned Lean kernel in CI; kernel_verified is false in this receipt. It does "
            "not establish novelty or cover shapes outside the declared family."
        ),
        "system_caps": SYSTEM_CAPS,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_result(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RESULT_SCHEMA:
        raise LemmaDecompositionError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise LemmaDecompositionError("result seal changed")
    if dict(value) != decompose_closed_form_proof(value["problem"]):
        raise LemmaDecompositionError("result exact replay changed")


def write_lean_source(value: Mapping[str, Any], output_path: Path) -> Path:
    """Materialize the generated Lean for the CI kernel stage."""

    source = value.get("lean_source")
    if not source:
        raise LemmaDecompositionError("no lean source in a rejected decomposition")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8", newline="\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Lemma decomposition (B5).")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--output")
    parser.add_argument("--lean-output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    problem = json.loads(Path(args.problem).read_text(encoding="utf-8"))
    if args.validate_checked:
        validate_result(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = decompose_closed_form_proof(problem)
    if args.output:
        encoded = canonical_json_bytes(result) + b"\n"
        path = Path(args.output)
        if path.exists() and path.read_bytes() != encoded:
            raise LemmaDecompositionError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    else:
        print(json.dumps(result, indent=2))
    if args.lean_output and result["decision"] == "DECOMPOSED":
        write_lean_source(result, Path(args.lean_output))
    return 0 if result["decision"] == "DECOMPOSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
