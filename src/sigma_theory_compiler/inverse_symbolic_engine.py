"""Inverse symbolic engine: brute-force-backwards formula discovery for named constants.

Forward symbolic work starts from a formula and computes its value.  This engine runs the
arrow backwards at scale: enumerate an enormous declared formula family on the GPU, evaluate
every member numerically, and ask which members land on a target constant.  The honesty core
is the digit holdout, the direct analog of the project's row-holdout rule for data: a match
found at fit precision is interpolation, never discovery.  The only signal this engine is
allowed to promote is *survival at far higher precision than the match was found at* — and
even a full survivor is published as a conjecture, not a theorem.  Every receipt carries
``match_at_fit_precision_is_not_discovery`` and
``survival_at_verify_precision_is_conjecture_not_proof`` as sealed claims.

Two lanes:

**PSLQ lane** (exact integer-relation detection).  For a target constant ``t`` and the
declared basis ``[1, pi, e, ln2, ln3, sqrt2, sqrt3, zeta3, EulerGamma, Catalan]`` computed
to 120 decimal digits, ``mpmath.pslq`` searches for integer relations
``c_0*t + sum(c_i * b_i) = 0`` under a declared coefficient bound.  Every hit is re-verified
at 200 digits before it may be reported as an exact rational-linear conjecture.  A found
relation is exact if true, but the search is bounded: "no relation" always means "no relation
with coefficients under the declared bound", never nonexistence.

**CF enumeration lane** (the Ramanujan-style structural lane).  An ordinal-indexed grammar of
generalized continued fractions ``CF(a, b) = a_0 + b_1/(a_1 + b_2/(a_2 + ...))`` with integer
polynomial patterns ``a_n = alpha0 + alpha1*n + alpha2*n^2`` and
``b_n = beta0 + beta1*n + beta2*n^2``, all six coefficients in {-4..4} (9^6 = 531,441 shapes),
crossed with a deduplicated Moebius wrap ``x -> (p*x + q)/(r*x + s)`` with p,q,r,s in {-2..2}
(identity included; determinant zero excluded; scalar multiples collapsed), for
531,441 x 224 = 119,042,784 ordinals.  Depth-40 convergents are evaluated in fp64 on the GPU
and compared against the declared target list inside a 1e-12 window; matches are re-evaluated
with mpmath at depth 2000 / 60 digits (survival bar 1e-50), and those survivors again at
depth 8000 / 120 digits (survival bar 1e-100).  Survivors are conjecture receipts with the
formula rendered in text and MathML-ready LaTeX, the full digit-survival trail, and a
prior-art label against a small built-in table of classical continued fractions:
``KNOWN_REDISCOVERED`` or ``NOT_IN_BUILTIN_TABLE`` — never "novel".  Absence from a finite
table establishes nothing about the literature (``corpus_absence_establishes_novelty: false``).

Expected controls: the classical golden-ratio and e continued fractions must be rediscovered.
Anything unlabeled that survives 120 digits is a candidate formula nobody in the built-in
table wrote down, published strictly as a conjecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA_PSLQ = "invariant-inverse-symbolic-pslq-result-1.0"
RESULT_SCHEMA_CF = "invariant-inverse-symbolic-cf-enumeration-result-1.0"

POLY_COEFFICIENT_VALUES = tuple(range(-4, 5))  # 9 values per slot
POLY_SLOTS = 6  # alpha0..alpha2, beta0..beta2
SHAPE_COUNT = len(POLY_COEFFICIENT_VALUES) ** POLY_SLOTS  # 531,441
MOBIUS_COEFFICIENT_RANGE = (-2, 2)
DEPTH_FP64 = 40
MATCH_WINDOW = 1e-12

#: Mandatory honesty claims shared by both lanes.  A receipt that drops or flips any of
#: these is invalid regardless of its numerical content.
SHARED_CLAIMS = {
    "match_at_fit_precision_is_not_discovery": True,
    "survival_at_verify_precision_is_conjecture_not_proof": True,
    "corpus_absence_establishes_novelty": False,
}

PSLQ_CLAIMS = {
    **SHARED_CLAIMS,
    "relation_is_exact_rational_linear_conjecture": True,
    "no_relation_means_none_under_declared_bound_only": True,
}

CF_CLAIMS = {
    **SHARED_CLAIMS,
    "builtin_table_absence_establishes_novelty": False,
    "survivors_are_conjectures_not_theorems": True,
    "enumeration_exhaustive_over_declared_family": True,
}

PSLQ_CONFIG: dict[str, Any] = {
    "basis": [
        {"name": "one", "definition": "1"},
        {"name": "pi", "definition": "pi"},
        {"name": "e", "definition": "exp(1)"},
        {"name": "ln2", "definition": "log(2)"},
        {"name": "ln3", "definition": "log(3)"},
        {"name": "sqrt2", "definition": "sqrt(2)"},
        {"name": "sqrt3", "definition": "sqrt(3)"},
        {"name": "zeta3", "definition": "zeta(3)"},
        {"name": "euler_gamma", "definition": "EulerGamma"},
        {"name": "catalan", "definition": "Catalan"},
    ],
    "fit_dps": 120,
    "verify_dps": 200,
    "pslq_tol": "1e-100",
    "max_coefficient": 1000000,
    "max_steps": 10000,
    "verify_normalized_residual_threshold": "1e-180",
    "negative_control_seed": "invariant-inverse-symbolic-negative-control-v1",
    "targets": [
        {"name": "atan_one", "definition": "atan(1)", "expectation": "known_relation"},
        {"name": "log_six", "definition": "log(6)", "expectation": "known_relation"},
        {
            "name": "three_plus_two_pi_over_seven",
            "definition": "(3 + 2*pi)/7",
            "expectation": "constructed_control_relation",
        },
        {
            "name": "gamma_plus_two_catalan",
            "definition": "EulerGamma + 2*Catalan",
            "expectation": "constructed_control_relation",
        },
        {
            "name": "sha256_pseudorandom",
            "definition": "0.<130 digits from iterated sha256 of the declared seed>",
            "expectation": "no_relation_under_bound",
        },
    ],
}

CF_CONFIG: dict[str, Any] = {
    "grammar": (
        "CF(a, b) = a_0 + b_1/(a_1 + b_2/(a_2 + ...)); "
        "a_n = alpha0 + alpha1*n + alpha2*n^2; b_n = beta0 + beta1*n + beta2*n^2"
    ),
    "polynomial_coefficient_range": [-4, 4],
    "shape_count": SHAPE_COUNT,
    "shape_digit_order": "base-9 little-endian: alpha0, alpha1, alpha2, beta0, beta1, beta2",
    "mobius_coefficient_range": [-2, 2],
    "mobius_dedup": (
        "exclude determinant zero; divide by gcd; flip sign so the first nonzero of "
        "(p, q, r, s) is positive; identity map included"
    ),
    "ordinal_layout": "ordinal = shape_index * mobius_class_count + mobius_index",
    "fp64_depth": DEPTH_FP64,
    "fp64_match_window": "1e-12",
    "fp64_convergence_window": "1e-12",
    "verify_stages": [
        {"stage": "mpmath_depth2000_dps60", "depth": 2000, "dps": 60, "threshold": "1e-50"},
        {"stage": "mpmath_depth8000_dps120", "depth": 8000, "dps": 120, "threshold": "1e-100"},
    ],
    "value_render_dps": 100,
    "crosscheck_sample": 1024,
    "targets": [
        {"name": "pi", "definition": "pi"},
        {"name": "e", "definition": "exp(1)"},
        {"name": "sqrt2", "definition": "sqrt(2)"},
        {"name": "sqrt3", "definition": "sqrt(3)"},
        {"name": "ln2", "definition": "log(2)"},
        {"name": "zeta3", "definition": "zeta(3)"},
        {"name": "catalan", "definition": "Catalan"},
        {"name": "euler_gamma", "definition": "EulerGamma"},
        {"name": "phi", "definition": "(1 + sqrt(5))/2"},
        {"name": "e_pi", "definition": "exp(pi)"},
    ],
}

QUADRATIC_SURD_TARGETS = frozenset({"sqrt2", "sqrt3", "phi"})

#: Classical continued fractions inside the declared family.  Entries with a wrap and target
#: are rediscovery controls; entries with target null document classical shapes (the pi
#: family) whose value is in the family but whose wrap to the target needs coefficients
#: outside [-2, 2].  The table is finite and explicit; absence from it is never novelty.
BUILTIN_KNOWN_TABLE: tuple[dict[str, Any], ...] = (
    {
        "id": "phi_simple_cf",
        "alpha": [1, 0, 0],
        "beta": [1, 0, 0],
        "mobius": [1, 0, 0, 1],
        "target": "phi",
        "value": "phi",
        "source_note": "simple continued fraction [1; 1, 1, 1, ...] of the golden ratio",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "silver_ratio_sqrt2_cf",
        "alpha": [2, 0, 0],
        "beta": [1, 0, 0],
        "mobius": [1, -1, 0, 1],
        "target": "sqrt2",
        "value": "1 + sqrt(2)",
        "source_note": "periodic CF x = 2 + 1/x (silver ratio); sqrt(2) = x - 1",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "sqrt3_periodic_cf",
        "alpha": [2, 0, 0],
        "beta": [2, 0, 0],
        "mobius": [1, -1, 0, 1],
        "target": "sqrt3",
        "value": "1 + sqrt(3)",
        "source_note": "periodic CF x = 2 + 2/x; sqrt(3) = x - 1",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "sqrt3_from_4_minus_1_cf",
        "alpha": [4, 0, 0],
        "beta": [-1, 0, 0],
        "mobius": [1, -2, 0, 1],
        "target": "sqrt3",
        "value": "2 + sqrt(3)",
        "source_note": "periodic CF x = 4 - 1/x; sqrt(3) = x - 2",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "sqrt2_from_4_minus_2_cf",
        "alpha": [4, 0, 0],
        "beta": [-2, 0, 0],
        "mobius": [1, -2, 0, 1],
        "target": "sqrt2",
        "value": "2 + sqrt(2)",
        "source_note": "periodic CF x = 4 - 2/x; sqrt(2) = x - 2",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "phi_squared_cf",
        "alpha": [3, 0, 0],
        "beta": [-1, 0, 0],
        "mobius": [1, -1, 0, 1],
        "target": "phi",
        "value": "1 + phi = phi^2",
        "source_note": "periodic CF x = 3 - 1/x; phi = x - 1",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "euler_e_tail_cf",
        "alpha": [1, 1, 0],
        "beta": [0, 1, 0],
        "mobius": [2, 1, 1, 0],
        "target": "e",
        "value": "1/(e - 2)",
        "source_note": "Euler 1737: e = 2 + 1/(1 + 1/(2 + 2/(3 + 3/(4 + ...)))); tail form",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "euler_e_alternating_cf",
        "alpha": [3, 1, 0],
        "beta": [0, -1, 0],
        "mobius": [1, 0, 0, 1],
        "target": "e",
        "value": "e",
        "source_note": "classical alternating form e = 3 - 1/(4 - 2/(5 - 3/(6 - ...)))",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "lambert_coth_half_cf",
        "alpha": [2, 4, 0],
        "beta": [1, 0, 0],
        "mobius": [1, 1, 1, -1],
        "target": "e",
        "value": "coth(1/2) = (e + 1)/(e - 1)",
        "source_note": "Lambert/Euler: coth(1/2) = 2 + 1/(6 + 1/(10 + 1/(14 + ...)))",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "e_over_e_minus_one_cf",
        "alpha": [2, 1, 0],
        "beta": [0, -1, 0],
        "mobius": [1, 0, 1, -1],
        "target": "e",
        "value": "e/(e - 1)",
        "source_note": "Euler-family hypergeometric ratio: x = 2 - 1/(3 - 2/(4 - 3/(5 - ...)))",
        "convergence": "fast",
        "value_decimal_approx": None,
    },
    {
        "id": "brouncker_4_over_pi_shifted_cf",
        "alpha": [2, 0, 0],
        "beta": [1, -4, 4],
        "mobius": None,
        "target": None,
        "value": "1 + 4/pi",
        "source_note": (
            "Brouncker 1655: 4/pi = 1 + 1^2/(2 + 3^2/(2 + 5^2/(2 + ...))); the constant-2 "
            "form equals 1 + 4/pi; no Moebius wrap with coefficients in [-2, 2] maps it to pi"
        ),
        "convergence": "slow",
        "value_decimal_approx": "2.27323954",
    },
    {
        "id": "gauss_4_over_pi_cf",
        "alpha": [1, 2, 0],
        "beta": [0, 0, 1],
        "mobius": None,
        "target": None,
        "value": "4/pi",
        "source_note": (
            "Euler/Gauss: 4/pi = 1 + 1/(3 + 4/(5 + 9/(7 + 16/(9 + ...)))); reaching pi "
            "itself needs a wrap coefficient of 4, outside [-2, 2]"
        ),
        "convergence": "fast",
        "value_decimal_approx": "1.27323954",
    },
    {
        "id": "lange_pi_plus_3_cf",
        "alpha": [6, 0, 0],
        "beta": [1, -4, 4],
        "mobius": None,
        "target": None,
        "value": "pi + 3",
        "source_note": (
            "classical form pi = 3 + 1^2/(6 + 3^2/(6 + 5^2/(6 + ...))); the constant-6 "
            "family member equals pi + 3; the wrap to pi needs q = -3, outside [-2, 2]"
        ),
        "convergence": "slow",
        "value_decimal_approx": "6.14159265",
    },
    {
        "id": "lambert_cot1_cf",
        "alpha": [1, 2, 0],
        "beta": [-1, 0, 0],
        "mobius": None,
        "target": None,
        "value": "cot(1)",
        "source_note": "Lambert 1761 tangent CF at z = 1: cot(1) = 1 - 1/(3 - 1/(5 - ...))",
        "convergence": "fast",
        "value_decimal_approx": "0.64209262",
    },
    {
        "id": "lambert_coth1_cf",
        "alpha": [1, 2, 0],
        "beta": [1, 0, 0],
        "mobius": None,
        "target": None,
        "value": "coth(1)",
        "source_note": "Lambert hyperbolic tangent CF at z = 1: coth(1) = 1 + 1/(3 + 1/(5 + ...))",
        "convergence": "fast",
        "value_decimal_approx": "1.31303529",
    },
)


class InverseSymbolicError(ValueError):
    """Raised on malformed input, codec violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Constant registry (exact mpmath definitions; evaluated at the caller's dps)
# ---------------------------------------------------------------------------


def constant_value(name: str) -> mp.mpf:
    """Exact named constant at the current mpmath working precision."""

    if name == "one":
        return mp.mpf(1)
    if name == "pi":
        return +mp.pi
    if name == "e":
        return +mp.e
    if name == "ln2":
        return mp.log(2)
    if name == "ln3":
        return mp.log(3)
    if name == "sqrt2":
        return mp.sqrt(2)
    if name == "sqrt3":
        return mp.sqrt(3)
    if name == "zeta3":
        return mp.zeta(3)
    if name == "euler_gamma":
        return +mp.euler
    if name == "catalan":
        return +mp.catalan
    if name == "phi":
        return +mp.phi
    if name == "e_pi":
        return mp.exp(mp.pi)
    raise InverseSymbolicError(f"unknown constant: {name}")


def _pseudorandom_decimal_digits(seed: str, count: int) -> str:
    """Deterministic transcendental-looking digit stream from iterated SHA-256."""

    state = seed.encode("utf-8")
    digits: list[str] = []
    while len(digits) < count:
        state = hashlib.sha256(state).digest()
        digits.extend(str(byte % 10) for byte in state)
    return "".join(digits[:count])


def pslq_target_value(name: str) -> mp.mpf:
    """Exact PSLQ-lane target at the current working precision."""

    if name == "atan_one":
        return mp.atan(1)
    if name == "log_six":
        return mp.log(6)
    if name == "three_plus_two_pi_over_seven":
        return (3 + 2 * mp.pi) / 7
    if name == "gamma_plus_two_catalan":
        return mp.euler + 2 * mp.catalan
    if name == "sha256_pseudorandom":
        digits = _pseudorandom_decimal_digits(PSLQ_CONFIG["negative_control_seed"], 130)
        return mp.mpf("0." + digits)
    raise InverseSymbolicError(f"unknown pslq target: {name}")


# ---------------------------------------------------------------------------
# Moebius wrap table and ordinal codec
# ---------------------------------------------------------------------------


def _canonical_mobius(p: int, q: int, r: int, s: int) -> tuple[int, int, int, int] | None:
    """Canonical class representative, or None for a degenerate (non-invertible) map."""

    if p * s - q * r == 0:
        return None
    divisor = math.gcd(math.gcd(abs(p), abs(q)), math.gcd(abs(r), abs(s)))
    entries = (p // divisor, q // divisor, r // divisor, s // divisor)
    for value in entries:
        if value != 0:
            if value < 0:
                entries = tuple(-item for item in entries)
            break
    return entries  # type: ignore[return-value]


def build_mobius_table() -> tuple[tuple[int, int, int, int], ...]:
    """All distinct invertible Moebius maps with coefficients in {-2..2}, sorted."""

    low, high = MOBIUS_COEFFICIENT_RANGE
    classes: set[tuple[int, int, int, int]] = set()
    for p in range(low, high + 1):
        for q in range(low, high + 1):
            for r in range(low, high + 1):
                for s in range(low, high + 1):
                    canonical = _canonical_mobius(p, q, r, s)
                    if canonical is not None:
                        classes.add(canonical)
    return tuple(sorted(classes))


MOBIUS_TABLE = build_mobius_table()
MOBIUS_COUNT = len(MOBIUS_TABLE)
MOBIUS_INDEX = {entry: index for index, entry in enumerate(MOBIUS_TABLE)}
TOTAL_ORDINALS = SHAPE_COUNT * MOBIUS_COUNT
CF_TARGET_NAMES = tuple(item["name"] for item in CF_CONFIG["targets"])


def decode_shape(shape_index: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Shape index -> (alpha, beta) polynomial coefficient triples."""

    if not 0 <= shape_index < SHAPE_COUNT:
        raise InverseSymbolicError(f"shape index out of range: {shape_index}")
    digits: list[int] = []
    value = shape_index
    for _ in range(POLY_SLOTS):
        digits.append(value % 9 - 4)
        value //= 9
    return (digits[0], digits[1], digits[2]), (digits[3], digits[4], digits[5])


def encode_shape(alpha: Sequence[int], beta: Sequence[int]) -> int:
    """Inverse of :func:`decode_shape`."""

    if len(alpha) != 3 or len(beta) != 3:
        raise InverseSymbolicError("alpha and beta must each have three coefficients")
    index = 0
    for digit in reversed((*alpha, *beta)):
        if digit not in POLY_COEFFICIENT_VALUES:
            raise InverseSymbolicError(f"polynomial coefficient out of range: {digit}")
        index = index * 9 + (digit + 4)
    return index


def decode_ordinal(ordinal: int) -> dict[str, Any]:
    """Ordinal -> {shape_index, mobius_index, alpha, beta, mobius}."""

    if not 0 <= ordinal < TOTAL_ORDINALS:
        raise InverseSymbolicError(f"ordinal out of range: {ordinal}")
    shape_index, mobius_index = divmod(ordinal, MOBIUS_COUNT)
    alpha, beta = decode_shape(shape_index)
    return {
        "ordinal": ordinal,
        "shape_index": shape_index,
        "mobius_index": mobius_index,
        "alpha": list(alpha),
        "beta": list(beta),
        "mobius": list(MOBIUS_TABLE[mobius_index]),
    }


def encode_ordinal(alpha: Sequence[int], beta: Sequence[int], mobius: Sequence[int]) -> int:
    """(alpha, beta, mobius) -> ordinal; the wrap must be a canonical table entry."""

    canonical = _canonical_mobius(*(int(v) for v in mobius))
    if canonical is None or canonical not in MOBIUS_INDEX:
        raise InverseSymbolicError(f"not a canonical invertible wrap in range: {tuple(mobius)}")
    return encode_shape(alpha, beta) * MOBIUS_COUNT + MOBIUS_INDEX[canonical]


# ---------------------------------------------------------------------------
# Vectorized fp64 evaluation (shared numpy/cupy code path)
# ---------------------------------------------------------------------------


def _shape_coefficients(xp: Any, shape_indices: Any) -> tuple[Any, ...]:
    """Mixed-radix decode: shape indices -> six fp64 coefficient vectors."""

    value = shape_indices.astype(xp.int64)
    columns = []
    for _ in range(POLY_SLOTS):
        columns.append((value % 9 - 4).astype(xp.float64))
        value = value // 9
    return tuple(columns)


def _cf_convergent(xp: Any, coefficients: tuple[Any, ...], depth: int) -> Any:
    """Depth-``depth`` convergent by backward recurrence, vectorized over shapes."""

    a0, a1, a2, b0, b1, b2 = coefficients
    x = a0 + a1 * depth + a2 * (depth * depth)
    for n in range(depth - 1, -1, -1):
        m = n + 1
        x = (a0 + a1 * n + a2 * (n * n)) + (b0 + b1 * m + b2 * (m * m)) / x
    return x


def _target_floats() -> np.ndarray:
    with mp.workdps(40):
        return np.array([float(constant_value(name)) for name in CF_TARGET_NAMES])


def enumerate_cf_matches(
    xp: Any,
    *,
    shape_start: int = 0,
    shape_stop: int = SHAPE_COUNT,
    mobius_block: int = 32,
) -> list[dict[str, Any]]:
    """fp64 sweep of shapes x wraps x targets.  Returns sorted raw match records.

    A match is an ordinal whose wrapped depth-40 and depth-39 convergents are both finite,
    agree within the convergence window, and land within the match window of a target.
    """

    shape_indices = xp.arange(shape_start, shape_stop, dtype=xp.int64)
    with np.errstate(all="ignore"):
        coefficients = _shape_coefficients(xp, shape_indices)
        deep = _cf_convergent(xp, coefficients, DEPTH_FP64)
        shallow = _cf_convergent(xp, coefficients, DEPTH_FP64 - 1)
    targets = xp.asarray(_target_floats())
    wraps = xp.asarray(np.array(MOBIUS_TABLE, dtype=np.float64))
    matches: list[dict[str, Any]] = []
    for block_start in range(0, MOBIUS_COUNT, mobius_block):
        block = wraps[block_start : block_start + mobius_block]
        p, q = block[:, 0:1], block[:, 1:2]
        r, s = block[:, 2:3], block[:, 3:4]
        with np.errstate(all="ignore"):
            wrapped_deep = (p * deep[None, :] + q) / (r * deep[None, :] + s)
            wrapped_shallow = (p * shallow[None, :] + q) / (r * shallow[None, :] + s)
            stable = (
                xp.isfinite(wrapped_deep)
                & xp.isfinite(wrapped_shallow)
                & (xp.abs(wrapped_deep - wrapped_shallow) < MATCH_WINDOW)
            )
            for target_index in range(len(CF_TARGET_NAMES)):
                error = xp.abs(wrapped_deep - targets[target_index])
                hit = stable & (error < MATCH_WINDOW)
                wrap_rows, shape_columns = xp.nonzero(hit)
                if wrap_rows.shape[0] == 0:
                    continue
                errors = error[wrap_rows, shape_columns]
                if xp is not np:
                    wrap_rows = wrap_rows.get()
                    shape_columns = shape_columns.get()
                    errors = errors.get()
                for wrap_row, shape_column, fp64_error in zip(
                    wrap_rows, shape_columns, errors, strict=True
                ):
                    shape_index = shape_start + int(shape_column)
                    mobius_index = block_start + int(wrap_row)
                    matches.append(
                        {
                            "shape_index": shape_index,
                            "mobius_index": mobius_index,
                            "target": CF_TARGET_NAMES[int(target_index)],
                            "fp64_abs_error": float(fp64_error),
                        }
                    )
    matches.sort(key=lambda item: (item["target"], item["shape_index"], item["mobius_index"]))
    return matches


def decide_ordinals(xp: Any, ordinals: Any) -> Any:
    """Per-ordinal target decisions, shape (n, targets).  Used by the CPU/GPU crosscheck."""

    ordinals = ordinals.astype(xp.int64)
    shape_indices = ordinals // MOBIUS_COUNT
    mobius_indices = ordinals % MOBIUS_COUNT
    with np.errstate(all="ignore"):
        coefficients = _shape_coefficients(xp, shape_indices)
        deep = _cf_convergent(xp, coefficients, DEPTH_FP64)
        shallow = _cf_convergent(xp, coefficients, DEPTH_FP64 - 1)
        wraps = xp.asarray(np.array(MOBIUS_TABLE, dtype=np.float64))[mobius_indices]
        p, q, r, s = wraps[:, 0], wraps[:, 1], wraps[:, 2], wraps[:, 3]
        wrapped_deep = (p * deep + q) / (r * deep + s)
        wrapped_shallow = (p * shallow + q) / (r * shallow + s)
        stable = (
            xp.isfinite(wrapped_deep)
            & xp.isfinite(wrapped_shallow)
            & (xp.abs(wrapped_deep - wrapped_shallow) < MATCH_WINDOW)
        )
        targets = xp.asarray(_target_floats())
        return stable[:, None] & (xp.abs(wrapped_deep[:, None] - targets[None, :]) < MATCH_WINDOW)


# ---------------------------------------------------------------------------
# Exact mpmath layer: the digit holdout
# ---------------------------------------------------------------------------


def cf_value_mp(alpha: Sequence[int], beta: Sequence[int], depth: int) -> mp.mpf:
    """Depth-``depth`` convergent at the current working precision (NaN on 0-division)."""

    a0, a1, a2 = (int(v) for v in alpha)
    b0, b1, b2 = (int(v) for v in beta)
    x = mp.mpf(a0 + a1 * depth + a2 * depth * depth)
    for n in range(depth - 1, -1, -1):
        m = n + 1
        if x == 0:
            return mp.mpf("nan")
        x = (a0 + a1 * n + a2 * n * n) + mp.mpf(b0 + b1 * m + b2 * m * m) / x
    return x


def _wrapped_mp(value: mp.mpf, mobius: Sequence[int]) -> mp.mpf:
    p, q, r, s = (int(v) for v in mobius)
    denominator = r * value + s
    if denominator == 0 or not mp.isfinite(value):
        return mp.mpf("nan")
    return (p * value + q) / denominator


def _error_string(error: mp.mpf, dps: int) -> str:
    if not mp.isfinite(error):
        return "nan"
    if error == 0:
        return f"<1e-{dps}"
    return mp.nstr(error, 3)


def survival_trail(
    value_at_depth: Callable[[int], mp.mpf],
    target_name: str,
    fp64_abs_error: float,
) -> tuple[list[dict[str, Any]], str]:
    """Run the digit holdout: fp64 match -> 60-digit stage -> 120-digit stage.

    ``value_at_depth`` must produce the candidate value at the requested depth using the
    *current* mpmath working precision; it is called inside each stage's ``workdps``.  The
    return is (trail, status) where status is ``SURVIVED_ALL_STAGES`` or ``DIED_AT_<stage>``.
    """

    trail = [
        {
            "stage": f"fp64_depth{DEPTH_FP64}",
            "abs_error": format(fp64_abs_error, ".3e"),
            "threshold": CF_CONFIG["fp64_match_window"],
            "passed": True,
        }
    ]
    for stage in CF_CONFIG["verify_stages"]:
        with mp.workdps(stage["dps"]):
            error = abs(value_at_depth(stage["depth"]) - constant_value(target_name))
            threshold = mp.mpf(stage["threshold"])
            passed = bool(mp.isfinite(error) and error < threshold)
            trail.append(
                {
                    "stage": stage["stage"],
                    "abs_error": _error_string(error, stage["dps"]),
                    "threshold": stage["threshold"],
                    "passed": passed,
                }
            )
        if not passed:
            return trail, f"DIED_AT_{stage['stage'].upper()}"
    return trail, "SURVIVED_ALL_STAGES"


class _ShapeValueCache:
    """Cache of exact CF values keyed by (alpha, beta, depth, dps)."""

    def __init__(self) -> None:
        self._values: dict[tuple[Any, ...], mp.mpf] = {}

    def value(self, alpha: Sequence[int], beta: Sequence[int], depth: int) -> mp.mpf:
        key = (tuple(alpha), tuple(beta), depth, mp.mp.dps)
        if key not in self._values:
            self._values[key] = cf_value_mp(alpha, beta, depth)
        return self._values[key]


# ---------------------------------------------------------------------------
# Prior-art classification against the built-in table
# ---------------------------------------------------------------------------


def classify_prior_art(
    alpha: Sequence[int], beta: Sequence[int], target_name: str
) -> dict[str, Any]:
    """KNOWN_REDISCOVERED versus NOT_IN_BUILTIN_TABLE (never a novelty claim)."""

    alpha = tuple(int(v) for v in alpha)
    beta = tuple(int(v) for v in beta)
    for entry in BUILTIN_KNOWN_TABLE:
        if entry["target"] == target_name and (alpha, beta) == (
            tuple(entry["alpha"]),
            tuple(entry["beta"]),
        ):
            return {
                "label": "KNOWN_REDISCOVERED",
                "basis": "exact_builtin_shape",
                "builtin_id": entry["id"],
            }
    for entry in BUILTIN_KNOWN_TABLE:
        if entry["target"] != target_name:
            continue
        base_alpha = tuple(entry["alpha"])
        base_beta = tuple(entry["beta"])
        for scale in (-1, 2, -2):
            scaled = (
                tuple(scale * v for v in base_alpha),
                tuple(scale * scale * v for v in base_beta),
            )
            if (alpha, beta) == scaled:
                return {
                    "label": "KNOWN_REDISCOVERED",
                    "basis": f"constant_equivalence_scaling_c={scale}",
                    "builtin_id": entry["id"],
                }
    if target_name in QUADRATIC_SURD_TARGETS:
        return {
            "label": "KNOWN_REDISCOVERED",
            "basis": "quadratic_surd_classical_theory",
            "builtin_id": None,
        }
    return {
        "label": "NOT_IN_BUILTIN_TABLE",
        "basis": "no_structural_match_in_builtin_table",
        "builtin_id": None,
    }


# ---------------------------------------------------------------------------
# Rendering (text and MathML-ready LaTeX)
# ---------------------------------------------------------------------------

_TARGET_TEXT = {
    "pi": ("pi", r"\pi"),
    "e": ("e", "e"),
    "sqrt2": ("sqrt(2)", r"\sqrt{2}"),
    "sqrt3": ("sqrt(3)", r"\sqrt{3}"),
    "ln2": ("ln(2)", r"\ln 2"),
    "ln3": ("ln(3)", r"\ln 3"),
    "zeta3": ("zeta(3)", r"\zeta(3)"),
    "catalan": ("Catalan", "G"),
    "euler_gamma": ("EulerGamma", r"\gamma"),
    "phi": ("phi", r"\varphi"),
    "e_pi": ("e^pi", r"e^{\pi}"),
    "one": ("1", "1"),
}


def _poly_text(coefficients: Sequence[int], latex: bool) -> str:
    c0, c1, c2 = (int(v) for v in coefficients)
    terms: list[str] = []
    square = "n^{2}" if latex else "n^2"
    for coefficient, symbol in ((c2, square), (c1, "n"), (c0, "")):
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        if symbol and magnitude == 1:
            body = symbol
        elif symbol:
            body = f"{magnitude}{symbol}" if latex else f"{magnitude}*{symbol}"
        else:
            body = str(magnitude)
        if not terms:
            terms.append(body if coefficient > 0 else f"-{body}")
        else:
            terms.append(f"+ {body}" if coefficient > 0 else f"- {body}")
    return " ".join(terms) if terms else "0"


def _linear_in_x(p: int, q: int, latex: bool) -> str:
    variable = "x"
    terms: list[str] = []
    if p != 0:
        if abs(p) == 1:
            head = variable
        else:
            head = f"{abs(p)}{variable}" if latex else f"{abs(p)}*{variable}"
        terms.append(head if p > 0 else f"-{head}")
    if q != 0 or not terms:
        if not terms:
            terms.append(str(q))
        else:
            terms.append(f"+ {abs(q)}" if q > 0 else f"- {abs(q)}")
    return " ".join(terms)


def _wrap_text(mobius: Sequence[int], latex: bool) -> str:
    p, q, r, s = (int(v) for v in mobius)
    numerator = _linear_in_x(p, q, latex)
    denominator = _linear_in_x(r, s, latex)
    if (r, s) == (0, 1):
        return numerator
    if latex:
        return rf"\frac{{{numerator}}}{{{denominator}}}"
    return f"({numerator})/({denominator})"


def _cf_layers_text(alpha: Sequence[int], beta: Sequence[int], layers: int, latex: bool) -> str:
    a0, a1, a2 = (int(v) for v in alpha)
    b0, b1, b2 = (int(v) for v in beta)
    a_values = [a0 + a1 * n + a2 * n * n for n in range(layers + 1)]
    b_values = [b0 + b1 * n + b2 * n * n for n in range(1, layers + 1)]
    if latex:
        tail = rf"{a_values[layers]} + \ddots"
        for n in range(layers - 1, 0, -1):
            sign = "+" if b_values[n] >= 0 else "-"
            tail = rf"{a_values[n]} {sign} \cfrac{{{abs(b_values[n])}}}{{{tail}}}"
        sign = "+" if b_values[0] >= 0 else "-"
        return rf"{a_values[0]} {sign} \cfrac{{{abs(b_values[0])}}}{{{tail}}}"
    tail = f"{a_values[layers]} + ..."
    for n in range(layers - 1, 0, -1):
        sign = "+" if b_values[n] >= 0 else "-"
        tail = f"{a_values[n]} {sign} {abs(b_values[n])}/({tail})"
    sign = "+" if b_values[0] >= 0 else "-"
    return f"{a_values[0]} {sign} {abs(b_values[0])}/({tail})"


def render_cf_conjecture(
    alpha: Sequence[int], beta: Sequence[int], mobius: Sequence[int], target_name: str
) -> dict[str, str]:
    """Human text and MathML-ready LaTeX for one continued-fraction conjecture."""

    text_target, latex_target = _TARGET_TEXT[target_name]
    text = (
        f"{text_target} =? {_wrap_text(mobius, latex=False)} where "
        f"x = {_cf_layers_text(alpha, beta, 4, latex=False)} with "
        f"a_n = {_poly_text(alpha, latex=False)}, b_n = {_poly_text(beta, latex=False)}"
    )
    latex = (
        rf"{latex_target} \stackrel{{?}}{{=}} {_wrap_text(mobius, latex=True)}, \quad "
        rf"x = {_cf_layers_text(alpha, beta, 3, latex=True)}, \quad "
        rf"a_n = {_poly_text(alpha, latex=True)},\; b_n = {_poly_text(beta, latex=True)}"
    )
    return {"text": text, "latex": latex}


# ---------------------------------------------------------------------------
# PSLQ lane
# ---------------------------------------------------------------------------


def relation_normalized_residual(
    target_value: mp.mpf, coefficients: Sequence[int]
) -> tuple[mp.mpf, mp.mpf]:
    """(|residual|, normalized residual) of c_0*t + sum(c_i*b_i) at the current dps."""

    names = [item["name"] for item in PSLQ_CONFIG["basis"]]
    if len(coefficients) != 1 + len(names):
        raise InverseSymbolicError("coefficient vector length does not match target + basis")
    values = [target_value] + [constant_value(name) for name in names]
    residual = mp.mpf(0)
    norm = mp.mpf(0)
    for coefficient, value in zip(coefficients, values, strict=True):
        residual += coefficient * value
        norm += abs(coefficient * value)
    if norm == 0:
        raise InverseSymbolicError("all-zero relation")
    return abs(residual), abs(residual) / norm


def _solved_relation(coefficients: Sequence[int]) -> dict[str, Any] | None:
    """Solve c_0*t + sum(c_i*b_i) = 0 for t as an exact rational combination."""

    target_coefficient = int(coefficients[0])
    if target_coefficient == 0:
        return None
    names = [item["name"] for item in PSLQ_CONFIG["basis"]]
    parts: dict[str, Fraction] = {}
    for name, coefficient in zip(names, coefficients[1:], strict=True):
        if coefficient:
            parts[name] = Fraction(-int(coefficient), target_coefficient)
    return parts


def _render_solution(target_definition: str, parts: Mapping[str, Fraction]) -> dict[str, str]:
    text_terms: list[str] = []
    latex_terms: list[str] = []
    for name, fraction in parts.items():
        text_name, latex_name = _TARGET_TEXT[name]
        magnitude = abs(fraction)
        if name == "one":
            text_body, latex_body = str(magnitude), rf"\tfrac{{{magnitude.numerator}}}{{{magnitude.denominator}}}"
            if magnitude.denominator == 1:
                latex_body = str(magnitude.numerator)
        elif magnitude == 1:
            text_body, latex_body = text_name, latex_name
        elif magnitude.denominator == 1:
            text_body = f"{magnitude.numerator}*{text_name}"
            latex_body = f"{magnitude.numerator}{latex_name}"
        else:
            text_body = f"({magnitude})*{text_name}"
            latex_body = rf"\tfrac{{{magnitude.numerator}}}{{{magnitude.denominator}}}{latex_name}"
        if not text_terms:
            text_terms.append(text_body if fraction > 0 else f"-{text_body}")
            latex_terms.append(latex_body if fraction > 0 else f"-{latex_body}")
        else:
            text_terms.append(f"+ {text_body}" if fraction > 0 else f"- {text_body}")
            latex_terms.append(f"+ {latex_body}" if fraction > 0 else f"- {latex_body}")
    if not text_terms:
        text_terms, latex_terms = ["0"], ["0"]
    return {
        "text": f"{target_definition} = " + " ".join(text_terms),
        "latex": " ".join(latex_terms),
    }


def run_pslq_lane() -> dict[str, Any]:
    """Run the PSLQ lane over the declared targets and seal a receipt."""

    started = time.perf_counter()
    fit_dps = PSLQ_CONFIG["fit_dps"]
    verify_dps = PSLQ_CONFIG["verify_dps"]
    verify_threshold = mp.mpf(PSLQ_CONFIG["verify_normalized_residual_threshold"])
    basis_names = [item["name"] for item in PSLQ_CONFIG["basis"]]
    target_reports: list[dict[str, Any]] = []
    relations_found = 0
    relations_verified = 0
    negative_controls_clean = 0
    for target in PSLQ_CONFIG["targets"]:
        name = target["name"]
        with mp.workdps(fit_dps):
            fit_value = pslq_target_value(name)
            vector = [fit_value] + [constant_value(basis) for basis in basis_names]
            relation = mp.pslq(
                vector,
                tol=mp.mpf(PSLQ_CONFIG["pslq_tol"]),
                maxcoeff=PSLQ_CONFIG["max_coefficient"],
                maxsteps=PSLQ_CONFIG["max_steps"],
            )
            fit_value_text = mp.nstr(fit_value, fit_dps - 5)
        report: dict[str, Any] = {
            "name": name,
            "definition": target["definition"],
            "expectation": target["expectation"],
            "fit_value": fit_value_text,
            "relation": None,
        }
        if relation is None:
            report["status"] = "NO_RELATION_FOUND_UNDER_BOUND"
            report["searched_bound"] = {
                "max_coefficient": PSLQ_CONFIG["max_coefficient"],
                "max_steps": PSLQ_CONFIG["max_steps"],
                "tol": PSLQ_CONFIG["pslq_tol"],
                "note": (
                    "absence of a relation under this bound is a bounded-search fact, "
                    "not a proof that no relation exists"
                ),
            }
            if target["expectation"] == "no_relation_under_bound":
                negative_controls_clean += 1
        else:
            relations_found += 1
            coefficients = [int(v) for v in relation]
            with mp.workdps(fit_dps):
                _, fit_normalized = relation_normalized_residual(
                    pslq_target_value(name), coefficients
                )
                fit_residual_text = mp.nstr(fit_normalized, 3)
            with mp.workdps(verify_dps):
                _, verify_normalized = relation_normalized_residual(
                    pslq_target_value(name), coefficients
                )
                verified = bool(verify_normalized < verify_threshold)
                verify_residual_text = _error_string(verify_normalized, verify_dps)
            relations_verified += int(verified)
            solved = _solved_relation(coefficients)
            rendered = (
                _render_solution(target["definition"], solved) if solved is not None else None
            )
            report["status"] = (
                "RELATION_FOUND_AND_VERIFIED" if verified else "RELATION_DIED_AT_VERIFY"
            )
            report["relation"] = {
                "coefficient_on_target": coefficients[0],
                "coefficients_on_basis": {
                    basis: coefficient
                    for basis, coefficient in zip(basis_names, coefficients[1:], strict=True)
                    if coefficient
                },
                "solved_form": rendered["text"] if rendered else None,
                "solved_form_latex": rendered["latex"] if rendered else None,
                "normalized_residual_at_fit": fit_residual_text,
                "normalized_residual_at_verify": verify_residual_text,
                "verified_at_verify_dps": verified,
            }
        target_reports.append(report)
    elapsed = time.perf_counter() - started
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_PSLQ,
        "lane": "pslq",
        "claims": PSLQ_CLAIMS,
        "config": PSLQ_CONFIG,
        "config_sha256": canonical_sha256(PSLQ_CONFIG),
        "targets": target_reports,
        "counts": {
            "targets": len(target_reports),
            "relations_found": relations_found,
            "relations_verified_at_200dps": relations_verified,
            "negative_controls_clean": negative_controls_clean,
        },
        "scope": (
            "Bounded integer-relation search over a declared 10-constant basis at 120 "
            "digits with 200-digit re-verification. Found relations are exact rational-"
            "linear conjectures; a missing relation is only missing under the declared "
            "coefficient bound. Nothing here is a proof and nothing here is a novelty claim."
        ),
    }
    core = canonical_sha256(body)
    body["result_core_sha256"] = core
    body["measurement"] = {"elapsed_seconds": format(elapsed, ".3f")}
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# CF enumeration lane
# ---------------------------------------------------------------------------


def _verify_matches(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Run the full digit-holdout ladder on every fp64 match (shape values cached)."""

    cache = _ShapeValueCache()
    verified: list[dict[str, Any]] = []
    group_sizes: dict[tuple[int, str], int] = {}
    for match in matches:
        key = (match["shape_index"], match["target"])
        group_sizes[key] = group_sizes.get(key, 0) + 1
    for match in matches:
        alpha, beta = decode_shape(match["shape_index"])
        mobius = MOBIUS_TABLE[match["mobius_index"]]

        def value_at_depth(
            depth: int,
            alpha: tuple[int, int, int] = alpha,
            beta: tuple[int, int, int] = beta,
            mobius: tuple[int, int, int, int] = mobius,
        ) -> mp.mpf:
            return _wrapped_mp(cache.value(alpha, beta, depth), mobius)

        trail, status = survival_trail(value_at_depth, match["target"], match["fp64_abs_error"])
        record: dict[str, Any] = {
            "ordinal": match["shape_index"] * MOBIUS_COUNT + match["mobius_index"],
            "shape_index": match["shape_index"],
            "mobius_index": match["mobius_index"],
            "alpha": list(alpha),
            "beta": list(beta),
            "mobius": list(mobius),
            "target": match["target"],
            "digit_trail": trail,
            "status": status,
            "shape_target_group_size": group_sizes[(match["shape_index"], match["target"])],
        }
        if status == "SURVIVED_ALL_STAGES":
            with mp.workdps(CF_CONFIG["verify_stages"][-1]["dps"]):
                stage_depth = CF_CONFIG["verify_stages"][-1]["depth"]
                value_text = mp.nstr(
                    cache.value(alpha, beta, stage_depth), CF_CONFIG["value_render_dps"]
                )
            rendered = render_cf_conjecture(alpha, beta, mobius, match["target"])
            record["cf_value_at_final_stage"] = value_text
            record["formula_text"] = rendered["text"]
            record["formula_latex"] = rendered["latex"]
            record["prior_art"] = classify_prior_art(alpha, beta, match["target"])
        verified.append(record)
    return verified


def run_cf_lane(
    *,
    use_gpu: bool = True,
    shape_limit: int | None = None,
    mobius_block: int = 32,
) -> dict[str, Any]:
    """Enumerate the declared CF family, verify matches, and seal a receipt."""

    if use_gpu:
        import cupy as xp

        device = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        xp = np
        device = "cpu-numpy"
    shape_stop = SHAPE_COUNT if shape_limit is None else min(shape_limit, SHAPE_COUNT)
    started = time.perf_counter()
    matches = enumerate_cf_matches(xp, shape_stop=shape_stop, mobius_block=mobius_block)
    if use_gpu:
        xp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    evaluated_ordinals = shape_stop * MOBIUS_COUNT

    verified = _verify_matches(matches)
    survivors = [item for item in verified if item["status"] == "SURVIVED_ALL_STAGES"]
    casualties = [item for item in verified if item["status"] != "SURVIVED_ALL_STAGES"]
    stage2_id = CF_CONFIG["verify_stages"][0]["stage"].upper()
    per_target = {
        name: {
            "fp64_matches": sum(1 for item in verified if item["target"] == name),
            "stage3_survivors": sum(1 for item in survivors if item["target"] == name),
        }
        for name in CF_TARGET_NAMES
    }
    prior_art_counts = {
        "known_rediscovered": sum(
            1 for item in survivors if item["prior_art"]["label"] == "KNOWN_REDISCOVERED"
        ),
        "not_in_builtin_table": sum(
            1 for item in survivors if item["prior_art"]["label"] == "NOT_IN_BUILTIN_TABLE"
        ),
    }

    crosscheck: dict[str, Any] = {"performed": False}
    if use_gpu:
        rng = np.random.default_rng(20260816)
        sample_size = min(CF_CONFIG["crosscheck_sample"], evaluated_ordinals)
        sample = rng.choice(evaluated_ordinals, size=sample_size, replace=False)
        anchors = [
            encode_ordinal((1, 0, 0), (1, 0, 0), (1, 0, 0, 1)),
            encode_ordinal((2, 0, 0), (1, 0, 0), (1, -1, 0, 1)),
            encode_ordinal((3, 1, 0), (0, -1, 0), (1, 0, 0, 1)),
        ]
        sample = np.unique(
            np.concatenate(
                [sample, [a for a in anchors if a < evaluated_ordinals]]
            ).astype(np.int64)
        )
        import cupy as cp

        gpu_decisions = decide_ordinals(cp, cp.asarray(sample)).get()
        cpu_decisions = decide_ordinals(np, sample)
        crosscheck = {
            "performed": True,
            "sample_ordinals": int(sample.shape[0]),
            "decision_disagreements": int((gpu_decisions != cpu_decisions).sum()),
        }

    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_CF,
        "lane": "cf-enumeration",
        "claims": CF_CLAIMS,
        "config": CF_CONFIG,
        "config_sha256": canonical_sha256(CF_CONFIG),
        "family": {
            "shape_count": SHAPE_COUNT,
            "mobius_class_count": MOBIUS_COUNT,
            "total_ordinals": TOTAL_ORDINALS,
            "evaluated_shape_count": shape_stop,
            "evaluated_ordinals": evaluated_ordinals,
            "exhaustive_this_run": shape_stop == SHAPE_COUNT,
        },
        "builtin_known_table": [dict(entry) for entry in BUILTIN_KNOWN_TABLE],
        "counts": {
            "fp64_matches": len(verified),
            "shape_target_groups": len(
                {(item["shape_index"], item["target"]) for item in verified}
            ),
            "stage2_survivors": sum(
                1
                for item in verified
                if item["status"] == "SURVIVED_ALL_STAGES"
                or not item["status"].endswith(stage2_id)
            ),
            "stage3_survivors": len(survivors),
            "died_in_verification": len(casualties),
            "per_target": per_target,
            "prior_art": prior_art_counts,
        },
        "crosscheck": crosscheck,
        "survivors": survivors,
        "died": casualties,
        "negative_results": {
            "targets_with_zero_matches": [
                name for name in CF_TARGET_NAMES if per_target[name]["fp64_matches"] == 0
            ]
        },
        "scope": (
            "Exhaustive fp64 GPU enumeration of a declared ordinal-indexed continued-"
            "fraction family with Moebius wraps, matched against declared constants at "
            "1e-12 and promoted only through mpmath digit-holdout stages at 60 and 120 "
            "digits. Survivors are conjectures. The prior-art label compares against a "
            "finite built-in table only and never asserts novelty."
        ),
    }
    core = canonical_sha256(body)
    body["result_core_sha256"] = core
    body["measurement"] = {
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "ordinals_per_second": int(evaluated_ordinals / elapsed) if elapsed > 0 else None,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------


def _check_seals(value: Mapping[str, Any], schema: str) -> None:
    if value.get("schema_version") != schema:
        raise InverseSymbolicError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise InverseSymbolicError("receipt seal changed")
    core_body = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    if value.get("result_core_sha256") != canonical_sha256(core_body):
        raise InverseSymbolicError("deterministic core seal changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise InverseSymbolicError("config binding changed")


def _check_claims(value: Mapping[str, Any], expected: Mapping[str, bool]) -> None:
    if value.get("claims") != expected:
        raise InverseSymbolicError("claims block changed")


def validate_pslq_receipt(value: Mapping[str, Any]) -> None:
    """Seal, schema, claims, and 200-digit residual replay for every reported relation."""

    _check_seals(value, RESULT_SCHEMA_PSLQ)
    _check_claims(value, PSLQ_CLAIMS)
    threshold = mp.mpf(str(value["config"]["verify_normalized_residual_threshold"]))
    basis_names = [item["name"] for item in value["config"]["basis"]]
    for report in value.get("targets", []):
        relation = report.get("relation")
        if relation is None:
            if report.get("status") != "NO_RELATION_FOUND_UNDER_BOUND":
                raise InverseSymbolicError("relation-free target must record the bounded search")
            continue
        coefficients = [int(relation["coefficient_on_target"])] + [
            int(relation["coefficients_on_basis"].get(name, 0)) for name in basis_names
        ]
        with mp.workdps(int(value["config"]["verify_dps"])):
            _, normalized = relation_normalized_residual(
                pslq_target_value(report["name"]), coefficients
            )
            replay_verified = bool(normalized < threshold)
        if replay_verified != relation["verified_at_verify_dps"]:
            raise InverseSymbolicError(f"relation replay changed for {report['name']}")


def validate_cf_receipt(value: Mapping[str, Any], *, replay_survivors: int = 6) -> None:
    """Seal, schema, claims, count consistency, and a bounded stage-2 survivor replay."""

    _check_seals(value, RESULT_SCHEMA_CF)
    _check_claims(value, CF_CLAIMS)
    survivors = value.get("survivors", [])
    counts = value.get("counts", {})
    if counts.get("stage3_survivors") != len(survivors):
        raise InverseSymbolicError("survivor count changed")
    if counts.get("fp64_matches") != len(survivors) + len(value.get("died", [])):
        raise InverseSymbolicError("match count changed")
    labels = {"known_rediscovered": 0, "not_in_builtin_table": 0}
    for survivor in survivors:
        replayed = classify_prior_art(
            survivor["alpha"], survivor["beta"], survivor["target"]
        )
        if replayed != survivor.get("prior_art"):
            raise InverseSymbolicError("prior-art label changed")
        labels[replayed["label"].lower()] += 1
    if counts.get("prior_art") != labels:
        raise InverseSymbolicError("prior-art counts changed")
    stage2 = value["config"]["verify_stages"][0]
    for survivor in survivors[:replay_survivors]:
        with mp.workdps(int(stage2["dps"])):
            exact = _wrapped_mp(
                cf_value_mp(survivor["alpha"], survivor["beta"], int(stage2["depth"])),
                survivor["mobius"],
            )
            error = abs(exact - constant_value(survivor["target"]))
            if not error < mp.mpf(str(stage2["threshold"])):
                raise InverseSymbolicError(
                    f"stage-2 replay failed for ordinal {survivor['ordinal']}"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise InverseSymbolicError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inverse symbolic engine: PSLQ lane and GPU CF enumeration lane."
    )
    parser.add_argument("--lane", choices=("pslq", "cf"), required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="force the numpy path (cf lane)")
    parser.add_argument("--shape-limit", type=int, default=None, help="cf lane shape prefix")
    parser.add_argument("--mobius-block", type=int, default=32)
    args = parser.parse_args()
    if args.validate_checked:
        loaded = json.loads(Path(args.output).read_text(encoding="utf-8"))
        if args.lane == "pslq":
            validate_pslq_receipt(loaded)
        else:
            validate_cf_receipt(loaded)
        print(json.dumps({"validated": True, "lane": args.lane, "output": args.output}))
        return 0
    if args.lane == "pslq":
        result = run_pslq_lane()
        summary = {
            "lane": "pslq",
            "targets": result["counts"]["targets"],
            "relations_found": result["counts"]["relations_found"],
            "relations_verified_at_200dps": result["counts"]["relations_verified_at_200dps"],
            "negative_controls_clean": result["counts"]["negative_controls_clean"],
            "elapsed_seconds": result["measurement"]["elapsed_seconds"],
        }
    else:
        result = run_cf_lane(
            use_gpu=not args.cpu, shape_limit=args.shape_limit, mobius_block=args.mobius_block
        )
        summary = {
            "lane": "cf-enumeration",
            "evaluated_ordinals": result["family"]["evaluated_ordinals"],
            "fp64_matches": result["counts"]["fp64_matches"],
            "stage3_survivors": result["counts"]["stage3_survivors"],
            "known_rediscovered": result["counts"]["prior_art"]["known_rediscovered"],
            "not_in_builtin_table": result["counts"]["prior_art"]["not_in_builtin_table"],
            "ordinals_per_second": result["measurement"]["ordinals_per_second"],
            "device": result["measurement"]["device"],
            "elapsed_seconds": result["measurement"]["elapsed_seconds"],
        }
    if args.output:
        _write_receipt(result, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
