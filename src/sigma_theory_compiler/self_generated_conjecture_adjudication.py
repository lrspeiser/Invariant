"""C3 -- self-generated conjectures carrying their own verification obligations.

``sparse_region_relations`` is the only module in the engine that generates its own
targets: 2,826 constants built from their own definitions rather than taken off a list a
human wrote.  It stops one step short of conjecture, though.  It *searches* for relations
among its self-generated constants at 400 decimal digits and reports the survivors as
unreviewed; it never emits a statement together with the obligation that would settle it,
and its arithmetic is floating-point, so nothing it produces can sit on a certificate
path.  ``conjecture_generation`` (B3) closes the other half -- it emits typed falsifiable
statements -- but its input rows are supplied by a caller who has already chosen the
object, and its ``proved`` field is false by construction because no obligation is ever
discharged.

This module joins the two halves on an exact integer path:

1. **The universe generates itself.**  A declared finite box of seed polynomials
   ``f(k) = k^p + a*k + b`` and a repeat count for the partial-sum operator
   ``S[f](n) = sum_{k=0}^{n} f(k)`` enumerates every object.  Nobody names a target.  Each
   object is carried as an exact integer numerator polynomial over a positive integer
   denominator, ``a(n) = N(n)/D``, and is admitted only if that closed form reproduces the
   directly summed integer value at every index of the observation window.

2. **The engine proposes statements.**  Two statement schemas are enumerated over a
   declared lattice, and instances are found by looking at a *prefix* of each object only:

   * ``residue_class_congruence`` -- ``for all n >= 0 with n = j (mod q): a(n) = r (mod m)``.
     The modulus is read off the prefix as a gcd of differences, so the statement is risky:
     a prefix gcd that overstates the true modulus produces a false conjecture.
   * ``polynomial_relation`` -- ``for all n >= 0: X(n) = alpha*Y(n) + beta*Z(n)`` for three
     *distinct* self-generated objects and a coefficient pair from a declared lattice.
     Instances are found by matching the combination against the pool on a short window,
     so a relation that happens to hold on that window and nowhere else is emitted and then
     killed.

3. **Every statement carries its own obligation, and the obligation is discharged.**
   A conjecture record contains an ``obligation`` block naming the decision procedure, its
   completeness scope, and every input needed to re-run it, plus an ``adjudication`` block
   holding the verdict and either a sealed certificate or an exact witness.

   * A congruence reduces, exactly, to ``K(t) = (a(q*t + j) - r)/m`` being an integer for
     every ``t >= 0``, and ``K`` is a rational polynomial.  A rational polynomial is
     integer-valued on the integers **iff** its finite-difference (binomial-basis)
     coefficients ``b_i = Delta^i K(0)`` are all integers -- necessary because each ``b_i``
     is an integer combination of ``K(0..i)``, sufficient because ``K(t) = sum_i b_i
     C(t,i)``.  That makes the test a *complete decision procedure*, not a sample, and it
     bounds the least counterexample by ``deg K``.  The reconstruction identity
     ``sum_i b_i*C(t,i) = K(t)`` -- the step the whole verdict rests on -- is proved by
     ``math_proof.prove_rational_identity`` and the returned certificate is re-validated.
   * A polynomial relation is an exact rational identity: ``math_proof`` proves it or
     ``math_counterexample.find_counterexample`` returns the integer that kills it.

4. **A generator that only restates its input is refused.**  Five admission gates run on
   every emitted candidate before adjudication.  ``restatement_control`` is a second,
   deliberately degenerate generator that echoes its input back in four disguises -- the
   observed value table, ``X = 1*X + 0*Z``, a congruence whose modulus is big enough to
   carry the observed values, and ``a(n) = 0 (mod 1)`` -- and it is run through the *same*
   gates.  The run aborts unless every one of its candidates is refused.

   Two further controls stop the gate and the adjudicator from passing by being vacuous:
   the gates must refuse some of the *honest* generator's candidates too, and the
   adjudicator must return both verdicts.  A run in which everything is admitted, or in
   which everything is PROVED, is void.

Exactness.  Every number on the certificate path is ``int`` or ``Fraction``.  There is no
float anywhere in a receipt; ``validate_receipt`` rejects one.

Claim boundary.  ``PROVED`` means: proved for every non-negative integer index by the
declared decision procedure, on an object this module generated.  It does not mean new, and
it does not mean interesting.  The objects are elementary by construction -- polynomial
partial sums -- and what is being demonstrated is the *loop*: statement proposed by the
engine, obligation attached by the engine, verdict returned by machinery that already
existed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any

from . import math_counterexample as ce
from . import math_proof as proof
from .math_canonicalizer import canonical_sha256 as formula_sha256
from .math_expression_ir import Equation, Expression, add, literal, multiply, power, symbol
from .math_types import IntegerType
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-self-generated-conjecture-adjudication-1.0"

#: Sealed honesty claims.  A receipt that drops or flips any of these is invalid.
CLAIMS = {
    "targets_are_self_generated_not_caller_supplied": True,
    "conjectures_carry_their_own_verification_obligation": True,
    "obligations_are_discharged_by_existing_machinery": True,
    "verdict_is_exact_no_floating_point_on_certificate_path": True,
    "restating_generator_must_be_refused": True,
    "proved_means_novel": False,
    "pool_membership_establishes_interest": False,
}


class ConjectureAdjudicationError(ValueError):
    """Raised when the generation/adjudication loop violates a declared invariant."""


# ---------------------------------------------------------------------------
# Exact polynomial arithmetic.  Coefficients ascend: ``coeffs[i]`` multiplies ``x**i``.
# ---------------------------------------------------------------------------

Poly = tuple[Fraction, ...]


def poly_trim(coeffs: Sequence[Fraction]) -> Poly:
    items = list(coeffs)
    while len(items) > 1 and items[-1] == 0:
        items.pop()
    return tuple(items)


def poly_eval(coeffs: Sequence[Fraction], x: int | Fraction) -> Fraction:
    total = Fraction(0)
    for coefficient in reversed(coeffs):
        total = total * x + coefficient
    return total


def poly_add(left: Sequence[Fraction], right: Sequence[Fraction]) -> Poly:
    size = max(len(left), len(right))
    return poly_trim(
        [
            (left[i] if i < len(left) else Fraction(0))
            + (right[i] if i < len(right) else Fraction(0))
            for i in range(size)
        ]
    )


def poly_scale(coeffs: Sequence[Fraction], factor: Fraction) -> Poly:
    return poly_trim([factor * coefficient for coefficient in coeffs])


def poly_mul(left: Sequence[Fraction], right: Sequence[Fraction]) -> Poly:
    out = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a == 0:
            continue
        for j, b in enumerate(right):
            out[i + j] += a * b
    return poly_trim(out)


def poly_compose_linear(coeffs: Sequence[Fraction], q: int, j: int) -> Poly:
    """Coefficients of ``P(q*t + j)`` in ``t``, exactly."""

    inner: Poly = (Fraction(j), Fraction(q))
    result: Poly = (Fraction(0),)
    powered: Poly = (Fraction(1),)
    for coefficient in coeffs:
        if coefficient != 0:
            result = poly_add(result, poly_scale(powered, coefficient))
        powered = poly_mul(powered, inner)
    return poly_trim(result)


def forward_differences(values: Sequence[Fraction]) -> tuple[Fraction, ...]:
    """``(Delta^0 v(0), Delta^1 v(0), ...)`` -- the binomial-basis coefficients."""

    row = [Fraction(value) for value in values]
    out: list[Fraction] = []
    while row:
        out.append(row[0])
        row = [row[k + 1] - row[k] for k in range(len(row) - 1)]
    return tuple(out)


def binomial_poly(index: int) -> Poly:
    """Coefficients of ``C(x, index) = x(x-1)...(x-index+1)/index!``."""

    result: Poly = (Fraction(1),)
    for shift in range(index):
        result = poly_mul(result, (Fraction(-shift), Fraction(1)))
    factor = Fraction(1, 1)
    for step in range(2, index + 1):
        factor /= step
    return poly_scale(result, factor)


def poly_from_values(values: Sequence[int | Fraction], degree: int) -> Poly:
    """Exact interpolation through ``(0, v0), ..., (degree, v_degree)``."""

    if len(values) <= degree:
        raise ConjectureAdjudicationError("not enough values to interpolate the declared degree")
    differences = forward_differences([Fraction(value) for value in values[: degree + 1]])
    result: Poly = (Fraction(0),)
    for index, coefficient in enumerate(differences):
        if coefficient != 0:
            result = poly_add(result, poly_scale(binomial_poly(index), coefficient))
    return poly_trim(result)


def clear_denominators(coeffs: Sequence[Fraction]) -> tuple[tuple[int, ...], int]:
    """Return ``(N, D)`` with integer ``N``, ``D > 0`` and ``P = N/D`` in lowest terms."""

    denominator = 1
    for coefficient in coeffs:
        step = coefficient.denominator
        denominator = denominator * step // gcd(denominator, step)
    numerator = [int(coefficient * denominator) for coefficient in coeffs]
    content = 0
    for value in numerator:
        content = gcd(content, abs(value))
    if content:
        common = gcd(content, denominator)
        if common > 1:
            numerator = [value // common for value in numerator]
            denominator //= common
    return tuple(numerator), denominator


def poly_expression(coeffs: Sequence[Fraction], variable: str) -> Expression:
    """Render a polynomial as rational IR -- integer powers and exact literals only."""

    var = symbol(variable)
    terms: list[Expression] = []
    for index, coefficient in enumerate(coeffs):
        if coefficient == 0:
            continue
        scalar = literal(int(coefficient) if coefficient.denominator == 1 else coefficient)
        if index == 0:
            terms.append(scalar)
        elif index == 1:
            terms.append(multiply(scalar, var))
        else:
            terms.append(multiply(scalar, power(var, literal(index))))
    if not terms:
        return literal(0)
    return terms[0] if len(terms) == 1 else add(*terms)


def fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


# ---------------------------------------------------------------------------
# Component 1 -- the universe generates itself.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoolConfig:
    """Declared parameter box.  Everything the pool contains follows from this record."""

    seed_powers: tuple[int, ...] = (1, 2, 3)
    seed_box: int = 2
    sum_depths: tuple[int, ...] = (0, 1, 2)
    window: int = 32
    prefix: int = 10
    relation_window: int = 4

    def as_json(self) -> dict[str, Any]:
        return {
            "seed_family": "f(k) = k^p + a*k + b",
            "seed_powers": list(self.seed_powers),
            "seed_box": f"a, b in [-{self.seed_box}, {self.seed_box}]",
            "sum_depths": list(self.sum_depths),
            "sum_operator": "S[f](n) = sum_{k=0}^{n} f(k)",
            "observation_window": f"n in [0, {self.window})",
            "congruence_formation_window": f"n in [0, {self.prefix})",
            "relation_formation_window": f"n in [0, {self.relation_window})",
        }


def seed_values(power: int, a: int, b: int, count: int) -> list[int]:
    return [k**power + a * k + b for k in range(count)]


def partial_sum(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    running = 0
    for value in values:
        running += value
        out.append(running)
    return out


def build_pool(config: PoolConfig) -> dict[str, Any]:
    """Enumerate the declared box.  No target is named; the box is the whole input."""

    members: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    box = config.seed_box
    for depth in config.sum_depths:
        for exponent in config.seed_powers:
            for a in range(-box, box + 1):
                for b in range(-box, box + 1):
                    values = seed_values(exponent, a, b, config.window)
                    for _ in range(depth):
                        values = partial_sum(values)
                    degree = exponent + depth
                    coeffs = poly_from_values(values, degree)
                    numerator, denominator = clear_denominators(coeffs)
                    reproduces = all(
                        poly_eval([Fraction(c) for c in numerator], n) == denominator * values[n]
                        for n in range(config.window)
                    )
                    record = {
                        "object_id": f"S{depth}:p{exponent}:a{a}:b{b}",
                        "definition": (
                            f"a(n) = (S^{depth} f)(n) with f(k) = k^{exponent} "
                            f"{'+' if a >= 0 else '-'} {abs(a)}*k "
                            f"{'+' if b >= 0 else '-'} {abs(b)}"
                        ),
                        "seed": {"power": exponent, "a": a, "b": b, "depth": depth},
                        "numerator": list(numerator),
                        "denominator": denominator,
                        "degree": len(numerator) - 1,
                        "values": list(values),
                    }
                    if not reproduces:
                        dropped.append(
                            {"object_id": record["object_id"], "reason": "closed_form_mismatch"}
                        )
                        continue
                    members.append(record)

    seen: dict[tuple[int, ...], str] = {}
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for record in members:
        key = tuple(record["values"])
        if key in seen:
            duplicates += 1
            continue
        seen[key] = record["object_id"]
        unique.append(record)
    unique.sort(key=lambda record: record["object_id"])
    return {
        "config": config.as_json(),
        "enumerated": len(members) + len(dropped),
        "dropped": dropped,
        "duplicate_value_vectors_removed": duplicates,
        "objects": unique,
    }


# ---------------------------------------------------------------------------
# Component 2 -- statement schemas.  Declared before the data, finite, and small.
# ---------------------------------------------------------------------------

#: Residue-class schema lattice.  A predicate outside this lattice cannot be emitted, which
#: is what stops a "conjecture" from smuggling the observed value table into its parameters.
CONGRUENCE_PERIODS = (1, 2, 3, 4, 6)
MAX_MODULUS = 12
MIN_CLASS_SAMPLES = 2

#: Relation schema lattice.
RELATION_COEFFICIENTS = tuple(value for value in range(-3, 4) if value != 0)
RELATION_PAIR_SAMPLE = 900
RELATION_INSTANCES_PER_SCHEMA = 6

ADMISSION_GATES = (
    "claim_extends_beyond_formation_window",
    "predicate_within_declared_lattice",
    "sides_share_no_object_slot",
    "pool_share_at_least_two",
    "pool_separates_at_least_one",
)

STATEMENT_KINDS = ("residue_class_congruence", "polynomial_relation")


def congruence_holds_on_window(values: Sequence[int], q: int, j: int, m: int, r: int) -> bool:
    return all(value % m == r % m for n, value in enumerate(values) if n % q == j)


def largest_admissible_modulus(spread: int) -> int | None:
    """The strongest modulus inside the declared lattice that the prefix supports."""

    if spread == 0:
        return None
    spread = abs(spread)
    for candidate in range(min(MAX_MODULUS, spread), 1, -1):
        if spread % candidate == 0:
            return candidate
    return None


def emit_congruences(pool: Sequence[Mapping[str, Any]], config: PoolConfig) -> list[dict[str, Any]]:
    """Propose residue-class congruences from the prefix alone."""

    emitted: list[dict[str, Any]] = []
    for record in pool:
        prefix = record["values"][: config.prefix]
        for q in CONGRUENCE_PERIODS:
            for j in range(q):
                sample = [value for n, value in enumerate(prefix) if n % q == j]
                if len(sample) < MIN_CLASS_SAMPLES:
                    continue
                spread = 0
                for value in sample[1:]:
                    spread = gcd(spread, value - sample[0])
                modulus = largest_admissible_modulus(spread)
                if modulus is None:
                    continue
                residue = sample[0] % modulus
                emitted.append(
                    {
                        "kind": "residue_class_congruence",
                        "object_id": record["object_id"],
                        "parameters": {"q": q, "j": j, "m": modulus, "r": residue},
                        "statement": (
                            f"for all n >= 0 with n = {j} (mod {q}): "
                            f"a(n) = {residue} (mod {modulus})  [a = {record['object_id']}]"
                        ),
                        "formation_window": [0, config.prefix],
                        "formation_samples": len(sample),
                        "claim_range": "n >= 0",
                    }
                )
    return emitted


def _sampled_pairs(order: Sequence[str]) -> list[tuple[str, str]]:
    """A deterministic, duplicate-free walk over ordered object pairs."""

    size = len(order)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for offset in (1, 7, 13, 31, 53, 97):
        for start in range(size):
            left = order[start]
            right = order[(start * 3 + offset) % size]
            if left == right:
                continue
            pair = (left, right)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= RELATION_PAIR_SAMPLE:
                return pairs
    return pairs


def emit_relations(pool: Sequence[Mapping[str, Any]], config: PoolConfig) -> list[dict[str, Any]]:
    """Propose three-object linear relations that match on the short relation window."""

    width = config.relation_window
    index: dict[tuple[int, ...], list[str]] = {}
    for record in pool:
        index.setdefault(tuple(record["values"][:width]), []).append(record["object_id"])
    by_id = {record["object_id"]: record for record in pool}
    pairs = _sampled_pairs([record["object_id"] for record in pool])

    emitted: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    per_schema: dict[tuple[int, int], int] = {}
    for alpha in RELATION_COEFFICIENTS:
        for beta in RELATION_COEFFICIENTS:
            for left_id, right_id in pairs:
                if per_schema.get((alpha, beta), 0) >= RELATION_INSTANCES_PER_SCHEMA:
                    break
                left = by_id[left_id]["values"][:width]
                right = by_id[right_id]["values"][:width]
                target = tuple(alpha * a + beta * b for a, b in zip(left, right, strict=True))
                for target_id in index.get(target, ()):
                    key = (target_id, left_id, right_id, alpha, beta)
                    if target_id in (left_id, right_id) or key in seen:
                        continue
                    seen.add(key)
                    per_schema[(alpha, beta)] = per_schema.get((alpha, beta), 0) + 1
                    emitted.append(
                        {
                            "kind": "polynomial_relation",
                            "object_id": target_id,
                            "parameters": {
                                "alpha": alpha,
                                "beta": beta,
                                "left_id": left_id,
                                "right_id": right_id,
                            },
                            "statement": (
                                f"for all n >= 0: {target_id}(n) = "
                                f"{alpha}*{left_id}(n) + {beta}*{right_id}(n)"
                            ),
                            "formation_window": [0, width],
                            "formation_samples": width,
                            "claim_range": "n >= 0",
                        }
                    )
                    break
    return emitted


# ---------------------------------------------------------------------------
# Component 3 -- admission gates.  This is where a restating generator dies.
# ---------------------------------------------------------------------------


def relation_schema_counts(
    pool: Sequence[Mapping[str, Any]], config: PoolConfig
) -> dict[tuple[int, int], dict[str, int]]:
    """Per coefficient pair: how many pool triples satisfy the schema, and how many do not.

    Measured over exactly the pair sample the emitter walks, so the content figure a gate
    reads is the same population the conjecture was drawn from.
    """

    width = config.relation_window
    by_id = {record["object_id"]: record for record in pool}
    index: dict[tuple[int, ...], list[str]] = {}
    for record in pool:
        index.setdefault(tuple(record["values"][:width]), []).append(record["object_id"])
    pairs = _sampled_pairs([record["object_id"] for record in pool])

    counts: dict[tuple[int, int], dict[str, int]] = {}
    for alpha in RELATION_COEFFICIENTS:
        for beta in RELATION_COEFFICIENTS:
            share = 0
            separate = 0
            for left_id, right_id in pairs:
                left = by_id[left_id]["values"][:width]
                right = by_id[right_id]["values"][:width]
                target = tuple(alpha * a + beta * b for a, b in zip(left, right, strict=True))
                hit = [
                    object_id
                    for object_id in index.get(target, ())
                    if object_id not in (left_id, right_id)
                ]
                if hit:
                    share += len(hit)
                else:
                    separate += 1
            counts[(alpha, beta)] = {"share": share, "separate": separate}
    return counts


def admit(
    candidate: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    config: PoolConfig,
    schema_counts: Mapping[tuple[int, int], Mapping[str, int]],
) -> dict[str, Any]:
    """Run the five gates.  ``admitted`` is the conjunction; refusals name the gate."""

    kind = candidate.get("kind")
    parameters = candidate.get("parameters", {})
    checks: dict[str, bool] = {}

    window = candidate.get("formation_window") or [0, 0]
    checks["claim_extends_beyond_formation_window"] = (
        candidate.get("claim_range") == "n >= 0" and int(window[1]) < config.window
    )

    if kind == "residue_class_congruence":
        q = parameters.get("q")
        j = parameters.get("j")
        m = parameters.get("m")
        r = parameters.get("r")
        checks["predicate_within_declared_lattice"] = (
            q in CONGRUENCE_PERIODS
            and isinstance(j, int)
            and 0 <= j < q
            and isinstance(m, int)
            and 2 <= m <= MAX_MODULUS
            and isinstance(r, int)
            and 0 <= r < m
        )
        checks["sides_share_no_object_slot"] = True
        # Content is measured whenever the predicate is *evaluable*, not only when it is
        # inside the lattice, so an out-of-lattice candidate is still refused on real counts
        # rather than on a zero the gate never computed.
        if (
            isinstance(q, int)
            and q >= 1
            and isinstance(j, int)
            and 0 <= j < q
            and (isinstance(m, int) and m >= 1 and isinstance(r, int))
        ):
            share = sum(
                1 for record in pool if congruence_holds_on_window(record["values"], q, j, m, r)
            )
            separate = len(pool) - share
        else:
            share = 0
            separate = 0
        checks["pool_share_at_least_two"] = share >= 2
        checks["pool_separates_at_least_one"] = separate >= 1
        content = {"share": share, "separate": separate}
    elif kind == "polynomial_relation":
        alpha = parameters.get("alpha")
        beta = parameters.get("beta")
        slots = (candidate.get("object_id"), parameters.get("left_id"), parameters.get("right_id"))
        checks["predicate_within_declared_lattice"] = (
            alpha in RELATION_COEFFICIENTS and beta in RELATION_COEFFICIENTS
        )
        checks["sides_share_no_object_slot"] = all(slots) and len(set(slots)) == 3
        counts = schema_counts.get((alpha, beta), {"share": 0, "separate": 0})
        checks["pool_share_at_least_two"] = int(counts["share"]) >= 2
        checks["pool_separates_at_least_one"] = int(counts["separate"]) >= 1
        content = {"share": int(counts["share"]), "separate": int(counts["separate"])}
    else:
        for gate in ADMISSION_GATES[1:]:
            checks[gate] = False
        content = {"share": 0, "separate": 0}

    refused = [gate for gate in ADMISSION_GATES if not checks.get(gate, False)]
    return {"checks": checks, "content": content, "admitted": not refused, "refused_gates": refused}


# ---------------------------------------------------------------------------
# Component 4 -- obligations, and their discharge by existing machinery.
# ---------------------------------------------------------------------------

CONGRUENCE_ROUTE = "sigma_theory_compiler.math_proof.prove_rational_identity"
RELATION_REFUTE_ROUTE = "sigma_theory_compiler.math_counterexample.find_counterexample"
RELATION_SEARCH_BOUND = 64


def congruence_quotient(record: Mapping[str, Any], q: int, j: int, m: int, r: int) -> Poly:
    """``K(t) = (a(q*t + j) - r) / m`` as an exact rational polynomial in ``t``."""

    numerator = [Fraction(value) for value in record["numerator"]]
    denominator = Fraction(record["denominator"])
    shifted = poly_compose_linear(numerator, q, j)
    return poly_scale(
        poly_add(shifted, (Fraction(-r) * denominator,)), Fraction(1, m) / denominator
    )


def congruence_obligation(
    record: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    q = int(parameters["q"])
    j = int(parameters["j"])
    m = int(parameters["m"])
    r = int(parameters["r"])
    quotient = congruence_quotient(record, q, j, m, r)
    body = {
        "obligation_kind": "integer_valued_rational_polynomial",
        "decision_procedure": (
            "K(t) = (a(q*t + j) - r)/m is integer-valued on t >= 0 iff every "
            "finite-difference coefficient b_i = Delta^i K(0) is an integer; necessity holds "
            "because b_i is an integer combination of K(0..i), sufficiency because "
            "K(t) = sum_i b_i*C(t,i)"
        ),
        "completeness": "complete_over_all_nonnegative_integers",
        "least_counterexample_bound": "deg K",
        "routed_to": [CONGRUENCE_ROUTE],
        "inputs": {
            "object_numerator": list(record["numerator"]),
            "object_denominator": int(record["denominator"]),
            "q": q,
            "j": j,
            "m": m,
            "r": r,
            "quotient_polynomial": [fraction_json(value) for value in quotient],
        },
    }
    body["obligation_sha256"] = canonical_sha256(body)
    return body


def relation_obligation(
    target: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    alpha: int,
    beta: int,
) -> dict[str, Any]:
    body = {
        "obligation_kind": "exact_rational_identity",
        "decision_procedure": (
            "clear denominators of alpha*Y(n) + beta*Z(n) - X(n) and require an identically "
            "zero numerator; a nonzero numerator of degree d has at most d integer roots, so "
            "an exact scan of n in [0, bound] returns a witness"
        ),
        "completeness": "complete_for_polynomials_of_degree_at_most_the_search_bound",
        "routed_to": [CONGRUENCE_ROUTE, RELATION_REFUTE_ROUTE],
        "inputs": {
            "target_numerator": list(target["numerator"]),
            "target_denominator": int(target["denominator"]),
            "left_numerator": list(left["numerator"]),
            "left_denominator": int(left["denominator"]),
            "right_numerator": list(right["numerator"]),
            "right_denominator": int(right["denominator"]),
            "alpha": alpha,
            "beta": beta,
            "search_bound": RELATION_SEARCH_BOUND,
        },
    }
    body["obligation_sha256"] = canonical_sha256(body)
    return body


def adjudicate_congruence(
    record: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Decide the congruence completely, with the reconstruction step externally proved."""

    q = int(parameters["q"])
    j = int(parameters["j"])
    m = int(parameters["m"])
    r = int(parameters["r"])
    quotient = congruence_quotient(record, q, j, m, r)
    degree = len(quotient) - 1
    samples = [poly_eval(quotient, t) for t in range(degree + 1)]
    coefficients = forward_differences(samples)

    reconstruction: Poly = (Fraction(0),)
    for index, coefficient in enumerate(coefficients):
        if coefficient != 0:
            reconstruction = poly_add(reconstruction, poly_scale(binomial_poly(index), coefficient))
    statement = Equation(
        poly_expression(reconstruction, "t"),
        poly_expression(quotient, "t"),
    )
    try:
        certificate = proof.prove_rational_identity(statement)
        proof.validate_rational_identity_certificate(certificate, statement)
    except (proof.ProofFailure, proof.UnsupportedProof, proof.ProofValidationError) as error:
        return {
            "verdict": "OPEN",
            "reason": f"binomial reconstruction was not certified: {error}",
            "route": CONGRUENCE_ROUTE,
        }

    integral = [value.denominator == 1 for value in coefficients]
    result: dict[str, Any] = {
        "route": CONGRUENCE_ROUTE,
        "reconstruction_statement_sha256": formula_sha256(statement),
        "reconstruction_certificate": certificate,
        "binomial_coefficients": [fraction_json(value) for value in coefficients],
    }
    if all(integral):
        result["verdict"] = "PROVED"
        result["scope"] = "every integer n >= 0 in the residue class"
        return result

    witness_t = next(t for t in range(degree + 1) if poly_eval(quotient, t).denominator != 1)
    index = q * witness_t + j
    value_numerator = poly_eval([Fraction(c) for c in record["numerator"]], index)
    value = value_numerator / Fraction(record["denominator"])
    if value.denominator != 1:
        return {
            "verdict": "OPEN",
            "reason": "object value at the witness index is not an integer",
            "route": CONGRUENCE_ROUTE,
        }
    result["verdict"] = "REFUTED"
    result["witness"] = {
        "t": witness_t,
        "n": index,
        "a_of_n": int(value),
        "residue": int(value) % m,
        "claimed_residue": r,
    }
    return result


def adjudicate_relation(
    target: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    alpha: int,
    beta: int,
) -> dict[str, Any]:
    """Prove the identity with ``math_proof`` or kill it with ``math_counterexample``."""

    def rational(record: Mapping[str, Any]) -> Poly:
        return poly_scale(
            tuple(Fraction(value) for value in record["numerator"]),
            Fraction(1, int(record["denominator"])),
        )

    combination = poly_add(
        poly_scale(rational(left), Fraction(alpha)),
        poly_scale(rational(right), Fraction(beta)),
    )
    statement = Equation(
        poly_expression(combination, "n"),
        poly_expression(rational(target), "n"),
    )
    try:
        certificate = proof.prove_rational_identity(statement)
        proof.validate_rational_identity_certificate(certificate, statement)
    except proof.ProofFailure:
        report = ce.find_counterexample(
            statement,
            {"n": IntegerType(minimum=0, maximum=RELATION_SEARCH_BOUND)},
            exact_assignments=[{"n": k} for k in range(RELATION_SEARCH_BOUND + 1)],
            strategies=(ce.SearchStrategy.EXACT,),
        )
        if report.status is not ce.SearchStatus.COUNTEREXAMPLE_FOUND:
            return {
                "verdict": "OPEN",
                "reason": f"identity failed but no witness within bound: {report.status.value}",
                "route": RELATION_REFUTE_ROUTE,
            }
        assignment = dict(report.counterexample.assignment)
        index = int(assignment["n"])
        return {
            "verdict": "REFUTED",
            "route": RELATION_REFUTE_ROUTE,
            "witness": {
                "n": index,
                "target": int(poly_eval(rational(target), index)),
                "combination": int(poly_eval(combination, index)),
                "trials_run": report.trials_run,
            },
        }
    except (proof.UnsupportedProof, proof.ProofValidationError) as error:
        return {"verdict": "OPEN", "reason": str(error), "route": CONGRUENCE_ROUTE}

    return {
        "verdict": "PROVED",
        "route": CONGRUENCE_ROUTE,
        "scope": "every integer n >= 0",
        "statement_sha256": formula_sha256(statement),
        "certificate": certificate,
    }


# ---------------------------------------------------------------------------
# Component 5 -- controls.  Positives are worthless without these.
# ---------------------------------------------------------------------------


def restating_generator(
    pool: Sequence[Mapping[str, Any]], config: PoolConfig
) -> list[dict[str, Any]]:
    """The degeneracy this module exists to refuse: a generator that echoes its input.

    Four disguises, each a different way of saying nothing, each aimed at a different gate:

    * ``value_table`` -- reports the rows it was handed and claims nothing outside them.
    * ``self_relation`` -- ``X(n) = 1*X(n) + 0*Z(n)``: the object on both sides.
    * ``unbounded_modulus_echo`` -- a congruence whose modulus is large enough to carry the
      observed values themselves, so the "predicate" is the data in disguise.
    * ``tautology`` -- ``a(n) = 0 (mod 1)``, true of every object in the pool.

    Control objects are the first twelve pool members whose prefix takes at least three
    distinct values; on a constant object the echo disguise is not constructible, because a
    small modulus already summarises the prefix and the statement stops being an echo.
    """

    varied = [
        record
        for record in pool
        if len(set(record["values"][: config.prefix])) >= 3
        and max(abs(value) for value in record["values"][: config.prefix]) > MAX_MODULUS
    ]
    candidates: list[dict[str, Any]] = []
    for record in varied[:12]:
        prefix = record["values"][: config.prefix]
        candidates.append(
            {
                "kind": "observed_value_table",
                "disguise": "value_table",
                "object_id": record["object_id"],
                "parameters": {"values": list(prefix)},
                "statement": f"a(n) = {prefix} for n in [0, {config.prefix})",
                "formation_window": [0, config.prefix],
                "formation_samples": len(prefix),
                "claim_range": f"0 <= n < {config.prefix}",
            }
        )
        candidates.append(
            {
                "kind": "polynomial_relation",
                "disguise": "self_relation",
                "object_id": record["object_id"],
                "parameters": {
                    "alpha": 1,
                    "beta": 0,
                    "left_id": record["object_id"],
                    "right_id": pool[0]["object_id"],
                },
                "statement": f"for all n >= 0: {record['object_id']}(n) = 1*{record['object_id']}(n)",
                "formation_window": [0, config.relation_window],
                "formation_samples": config.relation_window,
                "claim_range": "n >= 0",
            }
        )
        candidates.append(
            {
                "kind": "residue_class_congruence",
                "disguise": "unbounded_modulus_echo",
                "object_id": record["object_id"],
                "parameters": {
                    "q": 1,
                    "j": 0,
                    "m": max(abs(value) for value in prefix) + 1,
                    "r": prefix[0] % (max(abs(value) for value in prefix) + 1),
                },
                "statement": "a(n) = its own prefix values, dressed as a congruence",
                "formation_window": [0, config.prefix],
                "formation_samples": config.prefix,
                "claim_range": "n >= 0",
            }
        )
        candidates.append(
            {
                "kind": "residue_class_congruence",
                "disguise": "tautology",
                "object_id": record["object_id"],
                "parameters": {"q": 1, "j": 0, "m": 1, "r": 0},
                "statement": "for all n >= 0: a(n) = 0 (mod 1)",
                "formation_window": [0, config.prefix],
                "formation_samples": config.prefix,
                "claim_range": "n >= 0",
            }
        )
    return candidates


def restatement_control(
    pool: Sequence[Mapping[str, Any]],
    config: PoolConfig,
    schema_counts: Mapping[tuple[int, int], Mapping[str, int]],
) -> dict[str, Any]:
    """Run the degenerate generator through the *same* gates.  Nothing may get through."""

    candidates = restating_generator(pool, config)
    rows: list[dict[str, Any]] = []
    by_disguise: dict[str, set[str]] = {}
    for candidate in candidates:
        verdict = admit(candidate, pool, config, schema_counts)
        rows.append(
            {
                "disguise": candidate["disguise"],
                "object_id": candidate["object_id"],
                "admitted": verdict["admitted"],
                "refused_gates": verdict["refused_gates"],
            }
        )
        by_disguise.setdefault(candidate["disguise"], set()).update(verdict["refused_gates"])
    admitted = [row for row in rows if row["admitted"]]
    return {
        "control": "restating_generator_must_be_refused",
        "candidates": len(rows),
        "admitted": len(admitted),
        "passed": not admitted,
        "refusal_gates_by_disguise": {
            key: sorted(value) for key, value in sorted(by_disguise.items())
        },
        "sample": rows[:8],
    }


def planted_false_control(pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A congruence built to hold at n = 0 and fail later must come back REFUTED."""

    record = next(item for item in pool if item["object_id"] == "S1:p2:a0:b0")
    parameters = {"q": 4, "j": 0, "m": 10, "r": 0}
    verdict = adjudicate_congruence(record, parameters)
    return {
        "control": "planted_false_congruence_must_be_refuted",
        "object_id": record["object_id"],
        "parameters": parameters,
        "verdict": verdict.get("verdict"),
        "witness": verdict.get("witness"),
        "passed": verdict.get("verdict") == "REFUTED" and verdict.get("witness") is not None,
    }


def planted_true_control(pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``n odd  =>  n^2 = 1 (mod 8)`` must come back PROVED."""

    record = next(item for item in pool if item["object_id"] == "S0:p2:a0:b0")
    parameters = {"q": 2, "j": 1, "m": 8, "r": 1}
    verdict = adjudicate_congruence(record, parameters)
    return {
        "control": "planted_true_congruence_must_be_proved",
        "object_id": record["object_id"],
        "parameters": parameters,
        "verdict": verdict.get("verdict"),
        "passed": verdict.get("verdict") == "PROVED",
    }


def planted_false_relation_control(pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A relation that is false as a polynomial must be killed with an exact witness."""

    target = next(item for item in pool if item["object_id"] == "S0:p2:a0:b0")
    left = next(item for item in pool if item["object_id"] == "S0:p1:a0:b0")
    right = next(item for item in pool if item["object_id"] == "S0:p1:a0:b1")
    verdict = adjudicate_relation(target, left, right, 1, 1)
    return {
        "control": "planted_false_relation_must_be_refuted",
        "verdict": verdict.get("verdict"),
        "witness": verdict.get("witness"),
        "passed": verdict.get("verdict") == "REFUTED" and verdict.get("witness") is not None,
    }


def _contains_float(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(item) for item in value)
    return False


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


def _stride_sample(items: Sequence[Any], limit: int) -> list[Any]:
    """Deterministic even coverage of the candidate list rather than its first block."""

    if limit <= 0 or len(items) <= limit:
        return list(items)
    picked = sorted({index * len(items) // limit for index in range(limit)})
    return [items[index] for index in picked]


def run_loop(
    config: PoolConfig | None = None,
    *,
    max_congruences: int = 400,
    max_relations: int = 120,
) -> dict[str, Any]:
    """Generate the universe, propose statements, attach obligations, adjudicate."""

    config = config or PoolConfig()
    pool = build_pool(config)
    objects = pool["objects"]
    by_id = {record["object_id"]: record for record in objects}
    schema_counts = relation_schema_counts(objects, config)

    controls = [
        restatement_control(objects, config, schema_counts),
        planted_true_control(objects),
        planted_false_control(objects),
        planted_false_relation_control(objects),
    ]

    congruence_candidates = _stride_sample(emit_congruences(objects, config), max_congruences)
    relation_candidates = _stride_sample(emit_relations(objects, config), max_relations)
    candidates = congruence_candidates + relation_candidates

    conjectures: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for candidate in candidates:
        gate = admit(candidate, objects, config, schema_counts)
        record = dict(candidate)
        record["admission"] = gate
        if not gate["admitted"]:
            refused.append(
                {
                    "kind": candidate["kind"],
                    "object_id": candidate["object_id"],
                    "statement": candidate["statement"],
                    "refused_gates": gate["refused_gates"],
                }
            )
            continue
        if candidate["kind"] == "residue_class_congruence":
            source = by_id[candidate["object_id"]]
            record["obligation"] = congruence_obligation(source, candidate["parameters"])
            record["adjudication"] = adjudicate_congruence(source, candidate["parameters"])
        else:
            parameters = candidate["parameters"]
            target = by_id[candidate["object_id"]]
            left = by_id[parameters["left_id"]]
            right = by_id[parameters["right_id"]]
            record["obligation"] = relation_obligation(
                target, left, right, int(parameters["alpha"]), int(parameters["beta"])
            )
            record["adjudication"] = adjudicate_relation(
                target, left, right, int(parameters["alpha"]), int(parameters["beta"])
            )
        conjectures.append(record)

    verdicts: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for record in conjectures:
        verdict = record["adjudication"]["verdict"]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        bucket = by_kind.setdefault(record["kind"], {})
        bucket[verdict] = bucket.get(verdict, 0) + 1

    # A gate that refuses nothing real, and an adjudicator that returns one verdict for
    # everything, would both let the loop "succeed" while doing no work.  Both are refused.
    controls.append(
        {
            "control": "gates_bite_on_the_honest_generator_too",
            "admitted": len(conjectures),
            "refused": len(refused),
            "passed": bool(conjectures) and bool(refused),
        }
    )
    controls.append(
        {
            "control": "adjudicator_is_not_a_constant_function",
            "proved": verdicts.get("PROVED", 0),
            "refuted": verdicts.get("REFUTED", 0),
            "passed": verdicts.get("PROVED", 0) > 0 and verdicts.get("REFUTED", 0) > 0,
        }
    )
    failed = [entry for entry in controls if not entry["passed"]]

    result = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "pool": {
            "config": pool["config"],
            "enumerated": pool["enumerated"],
            "dropped": pool["dropped"],
            "duplicate_value_vectors_removed": pool["duplicate_value_vectors_removed"],
            "objects_admitted": len(objects),
            "sample": [
                {
                    "object_id": record["object_id"],
                    "definition": record["definition"],
                    "numerator": record["numerator"],
                    "denominator": record["denominator"],
                    "first_values": record["values"][:8],
                }
                for record in objects[:6]
            ],
        },
        "statement_kinds": list(STATEMENT_KINDS),
        "admission_gates": list(ADMISSION_GATES),
        "controls": controls,
        "controls_passed": not failed,
        "counts": {
            "candidates_emitted": len(candidates),
            "candidates_emitted_by_kind": {
                "residue_class_congruence": len(congruence_candidates),
                "polynomial_relation": len(relation_candidates),
            },
            "refused_by_admission_gates": len(refused),
            "adjudicated": len(conjectures),
            "verdicts": verdicts,
            "verdicts_by_kind": by_kind,
        },
        "conjectures": conjectures,
        "refused_sample": refused[:12],
        "limitations": [
            "objects are polynomial partial sums; the pool is elementary by construction",
            "PROVED is a statement about a self-generated object, never a novelty claim",
            "the statement schema lattice is declared and finite; absence means absent here",
            (
                "the declared box contains degenerate members (f(k) = k - k + 0 is the zero "
                "sequence), so a share of the proved relations are trivial once unfolded"
            ),
            "relation refutation is complete only for degrees within the declared search bound",
        ],
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Independently re-check a receipt: seal, claims, controls, exactness, verdicts."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise ConjectureAdjudicationError("unexpected schema version")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    # Exactness first: the canonical encoder refuses floats outright, so this check has to
    # run before the seal is recomputed or the run-ending error would come from elsewhere.
    if _contains_float(body):
        raise ConjectureAdjudicationError("receipt carries a floating-point value")
    if value.get("content_sha256") != canonical_sha256(body):
        raise ConjectureAdjudicationError("receipt content hash changed")
    if dict(value.get("claims") or {}) != CLAIMS:
        raise ConjectureAdjudicationError("sealed claims were altered")
    if not value.get("controls_passed"):
        raise ConjectureAdjudicationError("a control failed; the run is void")
    for entry in value.get("controls") or ():
        if not entry.get("passed"):
            raise ConjectureAdjudicationError(f"control did not pass: {entry.get('control')}")
    counts = value.get("counts") or {}
    conjectures = value.get("conjectures") or []
    if counts.get("adjudicated") != len(conjectures):
        raise ConjectureAdjudicationError("adjudicated count disagrees with the conjecture list")
    if not conjectures:
        raise ConjectureAdjudicationError("no conjecture reached adjudication")
    if not counts.get("refused_by_admission_gates"):
        raise ConjectureAdjudicationError("the admission gates refused nothing; they are vacuous")
    declared = dict(counts.get("verdicts") or {})
    if not declared.get("PROVED") or not declared.get("REFUTED"):
        raise ConjectureAdjudicationError("the adjudicator returned one verdict for everything")
    tally: dict[str, int] = {}
    for record in conjectures:
        if not record.get("admission", {}).get("admitted"):
            raise ConjectureAdjudicationError("an unadmitted candidate reached adjudication")
        obligation = record.get("obligation") or {}
        sealed = {key: item for key, item in obligation.items() if key != "obligation_sha256"}
        if obligation.get("obligation_sha256") != canonical_sha256(sealed):
            raise ConjectureAdjudicationError("obligation hash changed")
        verdict = (record.get("adjudication") or {}).get("verdict")
        if verdict not in {"PROVED", "REFUTED", "OPEN"}:
            raise ConjectureAdjudicationError(f"unknown verdict: {verdict}")
        if verdict == "REFUTED" and not (record["adjudication"].get("witness")):
            raise ConjectureAdjudicationError("a refutation carries no witness")
        tally[verdict] = tally.get(verdict, 0) + 1
    if tally != dict(counts.get("verdicts") or {}):
        raise ConjectureAdjudicationError("verdict tally disagrees with the conjecture list")


def write_receipt(result: Mapping[str, Any], output: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(result) + b"\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="write the sealed receipt here")
    parser.add_argument("--max-congruences", type=int, default=400)
    parser.add_argument("--max-relations", type=int, default=120)
    parser.add_argument("--summary", action="store_true", help="print counts only")
    args = parser.parse_args(argv)

    result = run_loop(
        max_congruences=args.max_congruences,
        max_relations=args.max_relations,
    )
    validate_receipt(result)
    if args.output:
        write_receipt(result, args.output)
    payload = result["counts"] if args.summary else result
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
