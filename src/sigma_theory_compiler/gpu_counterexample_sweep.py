"""M7 — GPU counterexample sweep with exact CPU witness verification.

Conjecture generation (B3) ends with statements that merely *survived* a holdout: a
handful of rows never contradicted them.  Nothing in the stack then goes hunting for the
counterexample at scale.  This module is that hunter.  It takes one declared,
universally quantified integer statement, a declared half-open range ``[lo, hi)``, and
sweeps every integer in that range in chunks — on GPU when available — looking for a
witness that kills the statement.

Three-bucket honesty is the core discipline.  Every swept index lands in exactly one of

  * decided-and-holding,
  * decided-and-violating (a **counterexample candidate**), or
  * ``undecided_at_step_cap`` — a bounded-iteration sequence (the Collatz total
    stopping time) ran out of its declared step budget.

The third bucket is fail-closed: a lane the kernel could not finish is *never* counted
as a pass and *never* reported as a counterexample.  If any lane is undecided and no
counterexample is confirmed, the whole receipt says ``UNDECIDED_STEP_CAP_HIT``.

The GPU layer is a screen, nothing more.  Every violation the screen flags is
re-verified on the CPU with pure-Python arbitrary-precision integer arithmetic (sympy
is used for primality only) before it may enter the receipt as a witness.  A screen
violation the exact layer cannot reproduce is an integrity failure and raises instead
of being silently dropped.  Collatz lanes whose trajectory would overflow uint64 are
not guessed at: the kernel marks them and the CPU settles them exactly.  Statement
configurations whose check arithmetic cannot be proven to fit in int64 do not run on
the vectorized layer at all — they fall back to exact Python integers end to end.

Statement kinds (declared and finite): ``divisibility``, ``congruence``,
``index_scaling_relation`` (rational alpha/beta), and ``monotonicity`` over a small
registry of built-in integer sequences computable termwise on GPU, plus two
pure-arithmetic kinds needing no sequence: ``goldbach_even_sum_of_two_primes`` and
``polynomial_positivity``.  The Collatz total stopping time is *not* GPU-friendly
termwise (the loop is unbounded), so it runs under a bounded-iteration kernel with a
declared cap of 10,000 steps.  Goldbach uses a CPU-sieved prime bitmap uploaded to the
device once (memory cap: ``hi <= 10^9``, i.e. a 1 GB host boolean sieve packed to a
125 MB device bitmap); the GPU scans the smallest primes and any even number the scan
misses is completed *exhaustively* on the CPU with sympy before any claim is made.

Claim boundary.  A finite sweep proves nothing about integers outside the range, and
for classical statements already verified far beyond any feasible sweep it adds no new
bound at all: Goldbach is verified in the literature for all even numbers up to
4*10^18 (Oliveira e Silva, Herzog & Pardi 2014), so a sweep below that bound is a
**mechanism receipt** — ``claims.exceeds_literature_bound`` is false and the receipt
says so.  The new-knowledge lane is sweeping the engine's *own* surviving conjectures
(e.g. the discovered relation sigma_collatz(2n) = sigma_collatz(n) + 1), which carry
no literature bound.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from fractions import Fraction
from math import isqrt, lcm
from pathlib import Path
from typing import Any

import numpy as np

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-gpu-counterexample-sweep-result-1.0"

DECISION_NO_COUNTEREXAMPLE = "NO_COUNTEREXAMPLE_IN_RANGE"
DECISION_COUNTEREXAMPLE = "COUNTEREXAMPLE"
DECISION_UNDECIDED = "UNDECIDED_STEP_CAP_HIT"
_DECISIONS = (DECISION_NO_COUNTEREXAMPLE, DECISION_COUNTEREXAMPLE, DECISION_UNDECIDED)

_STATUS_OK = 0
_STATUS_STEP_CAP = 1
_STATUS_OVERFLOW = 2

_INT64_LIMIT = 2**63 - 1
#: Largest v for which 3*v + 1 still fits in uint64; beyond it the kernel refuses to
#: guess and hands the lane to the exact CPU layer instead of wrapping silently.
_COLLATZ_UINT64_GUARD = (2**64 - 2) // 3

SYSTEM_CAPS = {
    "collatz_step_cap_default": 10000,
    "collatz_step_cap_max": 10000,
    #: Goldbach prime bitmap memory cap: the CPU sieve holds `hi` bytes of host RAM
    #: (1.0 GB at the cap) and the packed device bitmap holds `hi / 8` bytes
    #: (125 MB at the cap), uploaded once per sweep.
    "goldbach_max_hi": 10**9,
    #: The vectorized layer scans all primes below this; any even number it cannot
    #: resolve is completed exhaustively on the CPU with sympy.
    "goldbach_scan_prime_limit": 10000,
    "max_cpu_escalations": 100000,
    "undecided_sample_cap": 100,
    "violation_adjudication_batch": 256,
}

STATEMENT_KINDS = (
    "divisibility",
    "congruence",
    "index_scaling_relation",
    "monotonicity",
    "goldbach_even_sum_of_two_primes",
    "polynomial_positivity",
)

#: Built-in integer sequences computable termwise on the vectorized layer.
SEQUENCE_REGISTRY = {
    "affine": {"parameters": ("a", "b"), "bounded_iteration": False, "min_lo": None},
    "collatz_total_stopping_time": {"parameters": (), "bounded_iteration": True, "min_lo": 1},
    "digit_sum_base10": {"parameters": (), "bounded_iteration": False, "min_lo": 0},
    "popcount_base2": {"parameters": (), "bounded_iteration": False, "min_lo": 0},
    "triangular_number": {"parameters": (), "bounded_iteration": False, "min_lo": 0},
}

LITERATURE_BOUNDS = {
    "goldbach_even_sum_of_two_primes": {
        "verified_below": 4 * 10**18,
        "citation": (
            "Oliveira e Silva, Herzog & Pardi, Math. Comp. 83 (2014) 2033-2060: "
            "the Goldbach conjecture is verified for all even numbers up to 4e18"
        ),
    },
}

_SCOPE = (
    "Exhaustive counterexample sweep of one declared universally quantified integer "
    "statement over the half-open range [lo, hi). The GPU (or vectorized CPU) layer is a "
    "screen only: every reported witness is re-verified with exact pure-Python integer "
    "arithmetic (sympy for primality) before it enters the receipt, and a screen violation "
    "the exact layer cannot reproduce raises instead of being dropped. Bounded-iteration "
    "sequences run under a declared step cap and any lane exceeding it lands in a "
    "fail-closed third bucket (UNDECIDED_STEP_CAP_HIT), never in pass or fail. A finite "
    "sweep proves nothing outside the range; for statements with a literature bound "
    "(Goldbach: 4e18) a sweep below that bound adds no new bound and this receipt is a "
    "mechanism receipt. The new-knowledge lane is the engine's own surviving conjectures, "
    "which carry no literature bound."
)


class CounterexampleSweepError(ValueError):
    """Raised on malformed input, unsound configuration, integrity failure, or tamper."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CounterexampleSweepError(message)


def _plain_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CounterexampleSweepError(f"{label} must be a plain integer")
    return value


def _host_ints(array: Any) -> list[int]:
    if hasattr(array, "get"):
        array = array.get()
    return [int(item) for item in np.asarray(array).tolist()]


# ---------------------------------------------------------------------------
# Sequence registry: exact CPU term, vectorized term, and value bounds
# ---------------------------------------------------------------------------


def _collatz_exact(n: int, cap: int) -> int | None:
    """Total stopping time of n, or None when it exceeds the declared step cap."""

    v = n
    for steps in range(cap + 1):
        if v == 1:
            return steps
        if steps == cap:
            return None
        v = 3 * v + 1 if v & 1 else v >> 1
    return None


def _collatz_terms_vectorized(xp: Any, ns: Any, cap: int) -> tuple[Any, Any]:
    """Masked-loop evaluation with the same semantics as the bounded GPU kernel."""

    v = ns.astype(xp.uint64)
    value = xp.full(ns.shape, -1, dtype=xp.int64)
    status = xp.full(ns.shape, _STATUS_STEP_CAP, dtype=xp.int8)
    active = xp.ones(ns.shape, dtype=bool)
    for steps in range(cap + 1):
        finished = active & (v == xp.uint64(1))
        value = xp.where(finished, xp.int64(steps), value)
        status = xp.where(finished, xp.int8(_STATUS_OK), status)
        active = active & ~finished
        if steps == cap or not bool(active.any()):
            break
        odd = (v & xp.uint64(1)) != 0
        overflow = active & odd & (v > xp.uint64(_COLLATZ_UINT64_GUARD))
        status = xp.where(overflow, xp.int8(_STATUS_OVERFLOW), status)
        active = active & ~overflow
        v = xp.where(active & odd, v * xp.uint64(3) + xp.uint64(1), v)
        v = xp.where(active & ~odd, v >> xp.uint64(1), v)
    return value.astype(xp.int64), status.astype(xp.int8)


_COLLATZ_KERNEL = None


def _collatz_kernel(cp: Any) -> Any:
    """Bounded-iteration kernel: one thread walks one trajectory, capped at `cap` steps."""

    global _COLLATZ_KERNEL
    if _COLLATZ_KERNEL is None:
        _COLLATZ_KERNEL = cp.ElementwiseKernel(
            "uint64 n, int32 cap",
            "int64 value, int8 status",
            """
            unsigned long long v = n;
            long long out_value = -1;
            signed char out_status = 1;
            for (int steps = 0; steps <= cap; steps++) {
                if (v == 1ULL) { out_value = steps; out_status = 0; break; }
                if (steps == cap) { break; }
                if (v & 1ULL) {
                    if (v > 6148914691236517204ULL) { out_status = 2; break; }
                    v = 3ULL * v + 1ULL;
                } else {
                    v >>= 1;
                }
            }
            value = out_value;
            status = out_status;
            """,
            "sigma_collatz_total_stopping_time",
        )
    return _COLLATZ_KERNEL


def _digit_sum_terms(xp: Any, ns: Any) -> Any:
    v = ns.astype(xp.int64)
    total = xp.zeros(ns.shape, dtype=xp.int64)
    for _ in range(19):  # int64 has at most 19 decimal digits
        total = total + v % 10
        v = v // 10
    return total


def _popcount_terms(xp: Any, ns: Any) -> Any:
    v = ns.astype(xp.uint64)
    total = xp.zeros(ns.shape, dtype=xp.int64)
    for _ in range(64):
        total = total + (v & xp.uint64(1)).astype(xp.int64)
        v = v >> xp.uint64(1)
    return total


def _triangular_terms(xp: Any, ns: Any) -> Any:
    # Parity split keeps every intermediate at or below n*(n+1)/2, so the value
    # bound proven in the range plan also bounds the intermediates.
    even = ns % 2 == 0
    return xp.where(even, (ns // 2) * (ns + 1), ns * ((ns + 1) // 2))


def _digit_sum_exact(n: int) -> int:
    total = 0
    while n:
        total += n % 10
        n //= 10
    return total


def _sequence_tools(name: str, params: Mapping[str, int], step_cap: int) -> dict[str, Any]:
    """Closures for one registry sequence: vectorized terms, exact term, value bound."""

    if name == "affine":
        a, b = params["a"], params["b"]
        return {
            "terms": lambda xp, ns: (a * ns + b, None),
            "exact": lambda n: a * n + b,
            "value_bound": lambda m: abs(a) * m + abs(b),
            "describe": f"a(n) = ({a})*n + ({b})",
        }
    if name == "collatz_total_stopping_time":

        def terms(xp: Any, ns: Any) -> tuple[Any, Any]:
            if xp is np:
                return _collatz_terms_vectorized(np, ns, step_cap)
            value, status = _collatz_kernel(xp)(ns.astype(xp.uint64), np.int32(step_cap))
            return value, status

        return {
            "terms": terms,
            "exact": lambda n: _collatz_exact(n, step_cap),
            "value_bound": lambda m: step_cap,
            "describe": (
                f"a(n) = total stopping time of the Collatz map from n (step cap {step_cap})"
            ),
        }
    if name == "digit_sum_base10":
        return {
            "terms": lambda xp, ns: (_digit_sum_terms(xp, ns), None),
            "exact": _digit_sum_exact,
            "value_bound": lambda m: 9 * 19,
            "describe": "a(n) = sum of the base-10 digits of n",
        }
    if name == "popcount_base2":
        return {
            "terms": lambda xp, ns: (_popcount_terms(xp, ns), None),
            "exact": lambda n: n.bit_count(),
            "value_bound": lambda m: 64,
            "describe": "a(n) = number of set bits in the base-2 expansion of n",
        }
    if name == "triangular_number":
        return {
            "terms": lambda xp, ns: (_triangular_terms(xp, ns), None),
            "exact": lambda n: n * (n + 1) // 2,
            "value_bound": lambda m: m * (m + 1) // 2,
            "describe": "a(n) = n*(n+1)/2",
        }
    raise CounterexampleSweepError(f"unknown sequence: {name}")


# ---------------------------------------------------------------------------
# Statement normalization and compilation
# ---------------------------------------------------------------------------


def _rational(value: Any, label: str) -> Fraction:
    if isinstance(value, (bool, float)):
        raise CounterexampleSweepError(f"{label} must be an exact rational, not a float or bool")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise CounterexampleSweepError(f"{label} is not a valid rational: {value}") from error
    if isinstance(value, Mapping):
        if set(value) != {"numerator", "denominator"}:
            raise CounterexampleSweepError(f"{label} mapping keys changed")
        numerator = _plain_int(value["numerator"], f"{label}.numerator")
        denominator = _plain_int(value["denominator"], f"{label}.denominator")
        _require(denominator != 0, f"{label} denominator must be nonzero")
        return Fraction(numerator, denominator)
    raise CounterexampleSweepError(f"{label} has an unsupported type")


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _sequence_fields(statement: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the sequence, its parameters, and the step cap."""

    name = statement.get("sequence")
    if name not in SEQUENCE_REGISTRY:
        raise CounterexampleSweepError(f"unknown sequence: {name!r}")
    meta = SEQUENCE_REGISTRY[name]
    raw_params = statement.get("sequence_params", {})
    if not isinstance(raw_params, Mapping):
        raise CounterexampleSweepError("sequence_params must be a mapping")
    if set(raw_params) != set(meta["parameters"]):
        raise CounterexampleSweepError(
            f"sequence {name} takes exactly the parameters {sorted(meta['parameters'])}"
        )
    params = {key: _plain_int(raw_params[key], f"sequence_params.{key}") for key in raw_params}
    fields: dict[str, Any] = {"sequence": name, "sequence_params": params}
    if meta["bounded_iteration"]:
        cap = statement.get("step_cap", SYSTEM_CAPS["collatz_step_cap_default"])
        cap = _plain_int(cap, "step_cap")
        _require(
            1 <= cap <= SYSTEM_CAPS["collatz_step_cap_max"],
            f"step_cap must be in [1, {SYSTEM_CAPS['collatz_step_cap_max']}]",
        )
        fields["step_cap"] = cap
    else:
        _require("step_cap" not in statement, "step_cap only applies to bounded sequences")
    return fields


_KIND_KEYS = {
    "divisibility": {"sequence", "sequence_params", "step_cap", "divisor"},
    "congruence": {"sequence", "sequence_params", "step_cap", "modulus", "residue"},
    "index_scaling_relation": {"sequence", "sequence_params", "step_cap", "scale", "alpha", "beta"},
    "monotonicity": {"sequence", "sequence_params", "step_cap", "direction"},
    "goldbach_even_sum_of_two_primes": set(),
    "polynomial_positivity": {"coefficients"},
}


def _normalize_statement(statement: Any) -> dict[str, Any]:
    """Canonical statement echo: exact integers only, deterministic derived text."""

    if not isinstance(statement, Mapping):
        raise CounterexampleSweepError("statement must be a mapping")
    kind = statement.get("kind")
    if kind not in STATEMENT_KINDS:
        raise CounterexampleSweepError(f"unknown statement kind: {kind!r}")
    allowed = _KIND_KEYS[kind] | {"kind", "text"}
    unknown = set(statement) - allowed
    _require(not unknown, f"unknown statement keys for {kind}: {sorted(unknown)}")

    norm: dict[str, Any] = {"kind": kind}
    if kind == "goldbach_even_sum_of_two_primes":
        text = "every even n >= 4 in the declared range is the sum of two primes"
    elif kind == "polynomial_positivity":
        coefficients = statement.get("coefficients")
        _require(
            isinstance(coefficients, (list, tuple)) and len(coefficients) >= 1,
            "coefficients must be a nonempty list of integers (ascending powers)",
        )
        coefficients = [
            _plain_int(item, f"coefficients[{index}]") for index, item in enumerate(coefficients)
        ]
        norm["coefficients"] = coefficients
        rendered = " + ".join(
            f"({coefficient})" if power == 0 else f"({coefficient})*n^{power}"
            for power, coefficient in enumerate(coefficients)
        )
        text = f"for all n in the declared range: P(n) > 0, where P(n) = {rendered}"
    else:
        norm.update(_sequence_fields(statement))
        tools = _sequence_tools(
            norm["sequence"], norm["sequence_params"], norm.get("step_cap", 0)
        )
        described = tools["describe"]
        if kind == "divisibility":
            divisor = _plain_int(statement.get("divisor"), "divisor")
            _require(divisor >= 2, "divisor must be at least 2")
            norm["divisor"] = divisor
            text = f"for all n in the declared range: {divisor} divides a(n), where {described}"
        elif kind == "congruence":
            modulus = _plain_int(statement.get("modulus"), "modulus")
            _require(modulus >= 2, "modulus must be at least 2")
            residue = _plain_int(statement.get("residue"), "residue") % modulus
            norm["modulus"] = modulus
            norm["residue"] = residue
            text = (
                f"for all n in the declared range: a(n) = {residue} (mod {modulus}), "
                f"where {described}"
            )
        elif kind == "index_scaling_relation":
            scale = _plain_int(statement.get("scale", 2), "scale")
            _require(scale >= 2, "scale must be at least 2")
            alpha = _rational(statement.get("alpha"), "alpha")
            beta = _rational(statement.get("beta"), "beta")
            norm["scale"] = scale
            norm["alpha"] = _fraction_data(alpha)
            norm["beta"] = _fraction_data(beta)
            text = (
                f"for all n in the declared range: a({scale}*n) = ({alpha})*a(n) + ({beta}), "
                f"where {described}"
            )
        else:  # monotonicity
            direction = statement.get("direction")
            _require(
                direction in ("increasing", "decreasing"),
                "direction must be 'increasing' or 'decreasing'",
            )
            norm["direction"] = direction
            comparison = "<" if direction == "increasing" else ">"
            text = (
                f"for all n in the declared range: a(n) {comparison} a(n+1), where {described}"
            )
    if "text" in statement:
        _require(statement["text"] == text, "statement text does not match its parameters")
    norm["text"] = text
    return norm


def _compile_statement(norm: Mapping[str, Any]) -> dict[str, Any]:
    """Executable form of a normalized statement (Fractions, closures, scaled integers)."""

    kind = norm["kind"]
    spec: dict[str, Any] = {"kind": kind, "norm": dict(norm)}
    if kind == "polynomial_positivity":
        spec["coefficients"] = list(norm["coefficients"])
        return spec
    if kind == "goldbach_even_sum_of_two_primes":
        return spec
    spec["tools"] = _sequence_tools(
        norm["sequence"], norm["sequence_params"], norm.get("step_cap", 0)
    )
    spec["min_lo"] = SEQUENCE_REGISTRY[norm["sequence"]]["min_lo"]
    if kind == "divisibility":
        spec["divisor"] = norm["divisor"]
    elif kind == "congruence":
        spec["modulus"] = norm["modulus"]
        spec["residue"] = norm["residue"]
    elif kind == "monotonicity":
        spec["direction"] = norm["direction"]
    else:  # index_scaling_relation
        alpha = Fraction(norm["alpha"]["numerator"], norm["alpha"]["denominator"])
        beta = Fraction(norm["beta"]["numerator"], norm["beta"]["denominator"])
        common = lcm(alpha.denominator, beta.denominator)
        spec["scale"] = norm["scale"]
        spec["alpha"] = alpha
        spec["beta"] = beta
        # a(s*n) == alpha*a(n) + beta  <=>  common*a(s*n) == A*a(n) + B, all integers.
        spec["_common"] = common
        spec["_alpha_scaled"] = alpha.numerator * (common // alpha.denominator)
        spec["_beta_scaled"] = beta.numerator * (common // beta.denominator)
    return spec


def _range_plan(spec: Mapping[str, Any], lo: int, hi: int) -> dict[str, Any]:
    """Domain checks plus an a-priori int64 soundness proof for the vectorized layer."""

    _require(lo < hi, "empty range: lo must be strictly below hi")
    kind = spec["kind"]
    if kind == "goldbach_even_sum_of_two_primes":
        _require(lo >= 0, "goldbach range must be non-negative")
        _require(
            hi <= SYSTEM_CAPS["goldbach_max_hi"],
            "goldbach sweep exceeds the declared prime-bitmap memory cap "
            f"(hi <= {SYSTEM_CAPS['goldbach_max_hi']})",
        )
        return {"int64_sound": True}
    if kind == "polynomial_positivity":
        magnitude = max(abs(lo), abs(hi - 1))
        bound = 0
        for coefficient in reversed(spec["coefficients"]):
            bound = bound * magnitude + abs(coefficient)
        return {"int64_sound": bound <= _INT64_LIMIT}
    indices = [lo, hi - 1]
    if kind == "monotonicity":
        indices.append(hi)
    if kind == "index_scaling_relation":
        indices.extend([spec["scale"] * lo, spec["scale"] * (hi - 1)])
    if spec["min_lo"] is not None:
        _require(
            min(indices) >= spec["min_lo"],
            f"sequence domain starts at {spec['min_lo']} for this statement",
        )
    max_abs_index = max(abs(index) for index in indices)
    if max_abs_index > _INT64_LIMIT:
        return {"int64_sound": False}
    value_bound = spec["tools"]["value_bound"](max_abs_index)
    if kind == "index_scaling_relation":
        need = max(
            spec["_common"] * value_bound,
            abs(spec["_alpha_scaled"]) * value_bound + abs(spec["_beta_scaled"]),
        )
    else:
        need = value_bound
    return {"int64_sound": need <= _INT64_LIMIT}


# ---------------------------------------------------------------------------
# Exact CPU check (pure Python integers; sympy for primality only)
# ---------------------------------------------------------------------------


def _goldbach_decomposition(n: int) -> tuple[int, int] | None:
    """Smallest-p decomposition n = p + q with both prime, or None (exhaustive)."""

    import sympy

    for p in sympy.primerange(2, n // 2 + 1):
        if sympy.isprime(n - p):
            return int(p), n - int(p)
    return None


def _cpu_check_statement(spec: Mapping[str, Any], n: int) -> tuple[bool | None, dict[str, Any]]:
    """Decide the statement at one index exactly.  None means the step cap bound it."""

    kind = spec["kind"]
    if kind == "goldbach_even_sum_of_two_primes":
        if n < 4 or n % 2 != 0:
            return True, {"vacuous": True}
        decomposition = _goldbach_decomposition(n)
        detail: dict[str, Any] = {
            "decomposition": None if decomposition is None else list(decomposition),
            "search": "exhaustive primes p <= n/2 (sympy primality)",
        }
        return decomposition is not None, detail
    if kind == "polynomial_positivity":
        value = 0
        for coefficient in reversed(spec["coefficients"]):
            value = value * n + coefficient
        return value > 0, {"value": value}
    exact = spec["tools"]["exact"]
    if kind in ("divisibility", "congruence"):
        term = exact(n)
        if term is None:
            return None, {"reason": "step_cap"}
        if kind == "divisibility":
            remainder = term % spec["divisor"]
            return remainder == 0, {"sequence_value": term, "remainder": remainder}
        observed = term % spec["modulus"]
        return observed == spec["residue"], {"sequence_value": term, "observed_residue": observed}
    if kind == "monotonicity":
        left, right = exact(n), exact(n + 1)
        if left is None or right is None:
            return None, {"reason": "step_cap"}
        holds = left < right if spec["direction"] == "increasing" else left > right
        return holds, {"value_at_n": left, "value_at_next": right}
    # index_scaling_relation
    base, image = exact(n), exact(spec["scale"] * n)
    if base is None or image is None:
        return None, {"reason": "step_cap"}
    predicted = spec["alpha"] * base + spec["beta"]
    return Fraction(image) == predicted, {
        "value_at_n": base,
        "scaled_index": spec["scale"] * n,
        "value_at_scaled_index": image,
        "predicted": _fraction_data(predicted),
    }


# ---------------------------------------------------------------------------
# Vectorized chunk evaluation (shared numpy/cupy code path)
# ---------------------------------------------------------------------------


def _term_bundle(xp: Any, tools: Mapping[str, Any], ns: Any) -> tuple[Any, Any]:
    values, status = tools["terms"](xp, ns)
    if status is None:
        status = xp.zeros(ns.shape, dtype=xp.int8)
    return values, status


def _mask_outcome(
    xp: Any, ns: Any, violation: Any, statuses: list[Any] | None, checked: int
) -> dict[str, Any]:
    """Split one chunk into the three buckets and pull only what the CPU needs."""

    sample_cap = SYSTEM_CAPS["undecided_sample_cap"]
    outcome: dict[str, Any] = {
        "checked": checked,
        "undecided_count": 0,
        "undecided_sample": [],
        "candidates": [],
    }
    if statuses:
        undecided = xp.zeros(ns.shape, dtype=bool)
        overflow = xp.zeros(ns.shape, dtype=bool)
        for status in statuses:
            undecided = undecided | (status == _STATUS_STEP_CAP)
            overflow = overflow | (status == _STATUS_OVERFLOW)
        overflow = overflow & ~undecided
        violation = violation & ~(undecided | overflow)
        outcome["undecided_count"] = int(undecided.sum())
        outcome["undecided_sample"] = _host_ints(ns[undecided][:sample_cap])
        overflow_count = int(overflow.sum())
        _require(
            overflow_count <= SYSTEM_CAPS["max_cpu_escalations"],
            "uint64 overflow escalations exceed the declared CPU cap",
        )
        overflow_candidates = [
            (value, "overflow_escalation") for value in _host_ints(ns[overflow])
        ]
    else:
        overflow_candidates = []
    outcome["violation_count"] = int(violation.sum())
    batch = SYSTEM_CAPS["violation_adjudication_batch"]
    screen_candidates = [
        (value, "screen_violation") for value in _host_ints(ns[violation][:batch])
    ]
    outcome["candidates"] = sorted(screen_candidates + overflow_candidates)
    return outcome


def _goldbach_chunk(xp: Any, resources: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    ns = xp.arange(start, stop, dtype=xp.int64)
    evens = ns[(ns % 2 == 0) & (ns >= 4)]
    checked = int(evens.size)
    outcome: dict[str, Any] = {
        "checked": checked,
        "undecided_count": 0,
        "undecided_sample": [],
        "candidates": [],
        "violation_count": 0,
    }
    if checked == 0:
        return outcome
    bitmap = resources["bitmap"]
    unresolved = evens
    for prime in resources["scan_primes"]:
        if int(unresolved.size) == 0:
            break
        partner = unresolved - prime
        legal = partner >= 2
        index = xp.maximum(partner, 0)
        bits = (bitmap[index >> 3].astype(xp.int64) >> (index & 7)) & 1
        unresolved = unresolved[~(legal & (bits == 1))]
    misses = _host_ints(unresolved)
    _require(
        len(misses) <= SYSTEM_CAPS["max_cpu_escalations"],
        "goldbach scan misses exceed the declared CPU completion cap",
    )
    outcome["violation_count"] = len(misses)
    outcome["candidates"] = [(value, "goldbach_scan_miss") for value in misses]
    return outcome


def _vector_chunk(
    xp: Any, spec: Mapping[str, Any], resources: Any, start: int, stop: int
) -> dict[str, Any]:
    kind = spec["kind"]
    if kind == "goldbach_even_sum_of_two_primes":
        return _goldbach_chunk(xp, resources, start, stop)
    ns = xp.arange(start, stop, dtype=xp.int64)
    checked = stop - start
    if kind == "polynomial_positivity":
        value = xp.zeros(ns.shape, dtype=xp.int64)
        for coefficient in reversed(spec["coefficients"]):
            value = value * ns + coefficient
        return _mask_outcome(xp, ns, value <= 0, None, checked)
    tools = spec["tools"]
    if kind in ("divisibility", "congruence"):
        values, status = _term_bundle(xp, tools, ns)
        if kind == "divisibility":
            violation = values % spec["divisor"] != 0
        else:
            violation = values % spec["modulus"] != spec["residue"]
        return _mask_outcome(xp, ns, violation, [status], checked)
    if kind == "monotonicity":
        left, left_status = _term_bundle(xp, tools, ns)
        right, right_status = _term_bundle(xp, tools, ns + 1)
        if spec["direction"] == "increasing":
            violation = left >= right
        else:
            violation = left <= right
        return _mask_outcome(xp, ns, violation, [left_status, right_status], checked)
    # index_scaling_relation
    base, base_status = _term_bundle(xp, tools, ns)
    image, image_status = _term_bundle(xp, tools, ns * spec["scale"])
    violation = spec["_common"] * image != spec["_alpha_scaled"] * base + spec["_beta_scaled"]
    return _mask_outcome(xp, ns, violation, [base_status, image_status], checked)


def _bigint_chunk(spec: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Exact fallback when the check arithmetic cannot be proven to fit in int64."""

    outcome: dict[str, Any] = {
        "checked": 0,
        "undecided_count": 0,
        "undecided_sample": [],
        "candidates": [],
        "violation_count": 0,
    }
    sample_cap = SYSTEM_CAPS["undecided_sample_cap"]
    for n in range(start, stop):
        holds, _ = _cpu_check_statement(spec, n)
        outcome["checked"] += 1
        if holds is None:
            outcome["undecided_count"] += 1
            if len(outcome["undecided_sample"]) < sample_cap:
                outcome["undecided_sample"].append(n)
        elif holds is False:
            outcome["violation_count"] += 1
            outcome["candidates"] = [(n, "exact_violation")]
            break
    return outcome


# ---------------------------------------------------------------------------
# Goldbach resources: CPU-sieved prime bitmap, uploaded once
# ---------------------------------------------------------------------------


def _prime_bitmap(hi: int) -> np.ndarray:
    """Little-endian packed primality bits for [0, hi), spot-checked against sympy."""

    import sympy

    flags = np.zeros(max(hi, 4), dtype=bool)
    flags[2:] = True
    for p in range(2, isqrt(len(flags) - 1) + 1):
        if flags[p]:
            flags[p * p :: p] = False
    rng = np.random.default_rng(0x5EED)
    for index in rng.integers(0, hi, size=min(64, hi)):
        if bool(flags[index]) != bool(sympy.isprime(int(index))):
            raise CounterexampleSweepError(f"prime bitmap disagrees with sympy at {int(index)}")
    return np.packbits(flags[:hi], bitorder="little")


def _scan_primes() -> list[int]:
    import sympy

    return [int(p) for p in sympy.primerange(2, SYSTEM_CAPS["goldbach_scan_prime_limit"])]


def _prepare_resources(spec: Mapping[str, Any], xp: Any, hi: int) -> dict[str, Any] | None:
    if spec["kind"] != "goldbach_even_sum_of_two_primes" or xp is None:
        return None
    packed = _prime_bitmap(hi)
    bitmap = packed if xp is np else xp.asarray(packed)
    return {"bitmap": bitmap, "scan_primes": _scan_primes()}


# ---------------------------------------------------------------------------
# Claims and the literature-bound honesty rule
# ---------------------------------------------------------------------------


def _literature_entry(kind: str) -> dict[str, Any] | None:
    entry = LITERATURE_BOUNDS.get(kind)
    return None if entry is None else dict(entry)


def _build_claims(kind: str, hi: int) -> dict[str, bool]:
    entry = LITERATURE_BOUNDS.get(kind)
    has_bound = entry is not None
    exceeds = bool(has_bound and hi > entry["verified_below"])
    return {
        "corpus_absence_establishes_novelty": False,
        "decision_is_proof_of_universal_statement": False,
        "exceeds_literature_bound": exceeds,
        "mechanism_receipt_below_literature_bound": bool(has_bound and not exceeds),
        "scalar_truth_or_probability_score": False,
        "screen_layer_trusted_without_exact_witness_check": False,
        "statement_has_declared_literature_bound": has_bound,
        "step_cap_lanes_counted_as_pass_or_fail": False,
    }


# ---------------------------------------------------------------------------
# The sweep driver
# ---------------------------------------------------------------------------


def sweep(
    statement: Mapping[str, Any],
    lo: int,
    hi: int,
    chunk: int = 1 << 24,
    use_gpu: bool = True,
) -> dict[str, Any]:
    """Sweep [lo, hi) for a counterexample to `statement` and seal a receipt."""

    norm = _normalize_statement(statement)
    spec = _compile_statement(norm)
    lo = _plain_int(lo, "lo")
    hi = _plain_int(hi, "hi")
    chunk = _plain_int(chunk, "chunk")
    _require(chunk >= 1, "chunk must be positive")
    plan = _range_plan(spec, lo, hi)

    if not plan["int64_sound"]:
        xp = None
        device = "cpu-python"
        arithmetic_path = "python-bigint"
    elif use_gpu:
        import cupy

        xp = cupy
        device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        arithmetic_path = "int64"
    else:
        xp = np
        device = "cpu-numpy"
        arithmetic_path = "int64"

    started = time.perf_counter()
    resources = _prepare_resources(spec, xp, hi)
    counters = {
        "screen_violation_lanes": 0,
        "overflow_escalations_resolved": 0,
        "goldbach_scan_misses_resolved": 0,
        "witness_cpu_checks": 0,
    }
    checked = 0
    undecided_count = 0
    undecided_sample: list[int] = []
    sample_cap = SYSTEM_CAPS["undecided_sample_cap"]
    witness: dict[str, Any] | None = None
    chunks = 0
    scanned_up_to = lo

    for start in range(lo, hi, chunk):
        stop = min(start + chunk, hi)
        chunks += 1
        if xp is None:
            outcome = _bigint_chunk(spec, start, stop)
        else:
            outcome = _vector_chunk(xp, spec, resources, start, stop)
        checked += outcome["checked"]
        counters["screen_violation_lanes"] += outcome["violation_count"]

        adjudication_undecided: list[int] = []
        for candidate, reason in outcome["candidates"]:
            counters["witness_cpu_checks"] += 1
            holds, detail = _cpu_check_statement(spec, candidate)
            if holds is False:
                witness = {"n": candidate, "exact_check": detail}
                break
            if holds is None:
                adjudication_undecided.append(candidate)
            elif reason == "goldbach_scan_miss":
                counters["goldbach_scan_misses_resolved"] += 1
            elif reason == "overflow_escalation":
                counters["overflow_escalations_resolved"] += 1
            else:
                raise CounterexampleSweepError(
                    "integrity failure: screen violation at "
                    f"n={candidate} was not reproduced by the exact CPU check"
                )

        undecided_count += outcome["undecided_count"] + len(adjudication_undecided)
        if len(undecided_sample) < sample_cap:
            merged = sorted(outcome["undecided_sample"] + adjudication_undecided)
            undecided_sample.extend(merged[: sample_cap - len(undecided_sample)])
        scanned_up_to = stop
        if witness is not None:
            break
    elapsed = time.perf_counter() - started

    if witness is not None:
        decision = DECISION_COUNTEREXAMPLE
    elif undecided_count > 0:
        decision = DECISION_UNDECIDED
    else:
        decision = DECISION_NO_COUNTEREXAMPLE

    counts = {
        "checked": checked,
        "scanned_up_to": scanned_up_to,
        "undecided_step_cap": undecided_count,
        **counters,
    }
    body: dict[str, Any] = {
        "arithmetic_path": arithmetic_path,
        "chunk_size": chunk,
        "chunks": chunks,
        "claims": _build_claims(spec["kind"], hi),
        "counts": counts,
        "decision": decision,
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "literature": _literature_entry(spec["kind"]),
        "range": {"lo": lo, "hi": hi},
        "schema_version": RESULT_SCHEMA,
        "scope": _SCOPE,
        "statement": norm,
        "system_caps": SYSTEM_CAPS,
        "throughput_per_second": int(checked / elapsed) if elapsed > 0 else None,
        "undecided": {"count": undecided_count, "sample": undecided_sample},
        "witness": witness,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, schema, coherence, claims-recompute, and exact witness re-verification."""

    if not isinstance(value, Mapping):
        raise CounterexampleSweepError("receipt must be a mapping")
    if value.get("schema_version") != RESULT_SCHEMA:
        raise CounterexampleSweepError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CounterexampleSweepError("receipt seal changed")
    norm = _normalize_statement(value.get("statement"))
    if norm != value.get("statement"):
        raise CounterexampleSweepError("statement echo is not canonical")
    spec = _compile_statement(norm)
    declared_range = value.get("range")
    if not isinstance(declared_range, Mapping) or set(declared_range) != {"lo", "hi"}:
        raise CounterexampleSweepError("range block changed")
    lo = _plain_int(declared_range["lo"], "range.lo")
    hi = _plain_int(declared_range["hi"], "range.hi")
    _require(lo < hi, "range is empty")
    if value.get("claims") != _build_claims(norm["kind"], hi):
        raise CounterexampleSweepError("claims block does not match the statement and range")
    if value.get("literature") != _literature_entry(norm["kind"]):
        raise CounterexampleSweepError("literature block does not match the statement")
    decision = value.get("decision")
    if decision not in _DECISIONS:
        raise CounterexampleSweepError(f"unknown decision: {decision!r}")
    undecided = value.get("undecided")
    if not isinstance(undecided, Mapping) or set(undecided) != {"count", "sample"}:
        raise CounterexampleSweepError("undecided block changed")
    count = _plain_int(undecided["count"], "undecided.count")
    sample = undecided["sample"]
    _require(isinstance(sample, list), "undecided sample must be a list")
    _require(
        len(sample) == min(count, SYSTEM_CAPS["undecided_sample_cap"]),
        "undecided sample length does not match its count and cap",
    )
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in sample)
        and sample == sorted(sample),
        "undecided sample must be ascending integers",
    )
    counts = value.get("counts")
    if not isinstance(counts, Mapping):
        raise CounterexampleSweepError("counts block changed")
    scanned_up_to = _plain_int(counts.get("scanned_up_to"), "counts.scanned_up_to")
    witness = value.get("witness")
    if decision == DECISION_COUNTEREXAMPLE:
        _require(isinstance(witness, Mapping), "a COUNTEREXAMPLE receipt must carry a witness")
    else:
        _require(witness is None, f"a {decision} receipt must not carry a witness")
        _require(scanned_up_to == hi, f"a {decision} receipt must have scanned the full range")
        if decision == DECISION_NO_COUNTEREXAMPLE:
            _require(count == 0, "NO_COUNTEREXAMPLE_IN_RANGE requires zero undecided lanes")
        else:
            _require(count > 0, "UNDECIDED_STEP_CAP_HIT requires at least one undecided lane")
    if witness is not None:
        if set(witness) != {"n", "exact_check"}:
            raise CounterexampleSweepError("witness block changed")
        n = _plain_int(witness["n"], "witness.n")
        _require(lo <= n < hi, "witness index is outside the declared range")
        holds, detail = _cpu_check_statement(spec, n)
        if holds is not False:
            raise CounterexampleSweepError("witness does not violate the statement on exact recheck")
        if detail != witness["exact_check"]:
            raise CounterexampleSweepError("witness exact-check data changed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU counterexample sweep (M7).")
    parser.add_argument("--statement", help="path to a JSON statement file")
    parser.add_argument("--lo", type=int)
    parser.add_argument("--hi", type=int)
    parser.add_argument("--chunk", type=int, default=1 << 24)
    parser.add_argument("--output")
    parser.add_argument("--cpu", action="store_true", help="force the numpy path")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        if not args.output:
            parser.error("--validate-checked requires --output")
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    if args.statement is None or args.lo is None or args.hi is None:
        parser.error("--statement, --lo, and --hi are required to run a sweep")
    statement = json.loads(Path(args.statement).read_text(encoding="utf-8"))
    receipt = sweep(statement, args.lo, args.hi, chunk=args.chunk, use_gpu=not args.cpu)
    if args.output:
        path = Path(args.output)
        encoded = canonical_json_bytes(receipt) + b"\n"
        if path.exists() and path.read_bytes() != encoded:
            raise CounterexampleSweepError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "range": receipt["range"],
                "checked": receipt["counts"]["checked"],
                "undecided": receipt["undecided"]["count"],
                "witness_n": None if receipt["witness"] is None else receipt["witness"]["n"],
                "elapsed_seconds": receipt["elapsed_seconds"],
                "throughput_per_second": receipt["throughput_per_second"],
                "device": receipt["device"],
            },
            indent=2,
        )
    )
    if receipt["decision"] == DECISION_COUNTEREXAMPLE:
        return 3
    if receipt["decision"] == DECISION_UNDECIDED:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
