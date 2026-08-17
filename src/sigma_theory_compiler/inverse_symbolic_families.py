"""Inverse symbolic engine, widened: series, product, and integral families (DG3).

:mod:`sigma_theory_compiler.inverse_symbolic_engine` ran one structural family --
generalized continued fractions -- at 1.19e8 ordinals and produced 32 survivors that DG1/DG2
then showed were all classical.  That family is exhausted.  This module opens three
structurally *new* spaces with the same honesty core, each declared, ordinal-indexed,
GPU-enumerable, and each larger than 1e8 ordinals:

**Family S -- hypergeometric-type series.**  ``c0 * sum_{k>=0} t_k`` with ``t_0 = 1`` and
``t_{k+1}/t_k = z * P(k)/Q(k)``, ``P`` and ``Q`` integer polynomials of degree at most three
with all eight coefficients in ``{-4..4}``, ``z`` rational in ``{1, -1, 1/2}`` and ``c0``
rational in a declared twelve-element set.  A term ratio that is a rational function of the
index *is* a generalized hypergeometric series by construction, so every member has exact
``pFq`` parameters (the negated roots of ``P`` and ``Q``) and every survivor is therefore
recognizable -- and provable -- rather than merely numerically striking.

**Family P -- infinite products.**  ``c0 * prod_{k>=k0} A(k)/B(k)`` with ``A``, ``B`` integer
polynomials of degree at most three, coefficients in ``{-4..4}``, and ``k0`` in ``{1, 2, 3}``.
The declared shape subsumes ``1 + a/k^p + b/k^q`` for integer ``a``, ``b`` and
``1 <= p, q <= 3``: clearing the denominator writes it as ``A(k)/B(k)`` with ``B(k) = k^m``.
Convergence is *enforced structurally*, not guessed: ``log R(k)`` is summable exactly when
``A`` and ``B`` have equal degree, equal leading coefficient and equal subleading
coefficient, which is the classical condition, and every convergent member then has the
closed form ``prod_j Gamma(k0 - beta_j) / prod_i Gamma(k0 - alpha_i)`` over the roots.

**Family I -- definite integrals.**  ``c0 * int_0^1 x^a (1-x)^b K_m(x)^c dx`` over a declared
twelve-kernel set with ``a``, ``b`` on a rational grid of 360 points with denominator 12 and
``c`` in a declared eight-element set.  Many of these are Beta, zeta and Catalan values --
the known controls.

The discipline is inherited verbatim and is non-negotiable: **found at ~1e-13 in fp64, must
survive 60 digits, must survive 120 digits**, with the full digit-survival trail recorded per
survivor.  A fabricated near-miss control sitting ~1e-14 from a target is enumerated
alongside the real candidates and *must* die at 60 digits; the run aborts if it does not.

Every family also has run-aborting *rediscovery controls*: Family S must recover the Basel
series, the Leibniz/Gregory series for pi/4 and ``e = sum 1/k!``; Family P must recover the
Wallis product and Euler's ``sin``-product form; Family I must recover
``int_0^1 dx/(1+x^2) = pi/4``, ``int_0^1 -ln(x)/(1-x) dx = zeta(2)`` and a Beta instance.
A family that cannot rediscover its own classics cannot support a ``NOT_FOUND`` verdict, and
this module says so by refusing to seal a receipt.

Numerics.  The fp64 GPU sweep evaluates series and products by iterating the term ratio in a
fused CUDA kernel and extrapolating the partial sums (products) recorded at the geometric
checkpoints ``16, 32, ..., 2048`` to ``n -> infinity`` by Neville extrapolation in
``h = 1/n``.  That accelerator is *measured*, not assumed: its Lagrange weights are bounded
by 3.44 in absolute value, so it costs under one decimal digit of conditioning, and every
ordinal must additionally pass a **resolution gate** -- the order-8 and order-6
extrapolations must agree to 1e-13 -- before it is allowed to match anything.  Ordinals that
fail the resolution gate are counted in a declared bucket and never reported as negatives.
Integrals are evaluated by tanh-sinh (double-exponential) quadrature at 769 declared nodes
formulated in the log domain so that the endpoint algebraic singularities are exact, and the
sweep is organized as a fp64 GEMM.  Survivors are re-evaluated with mpmath in closed form --
``mp.hyper`` on the exact ``pFq`` parameters, the Gamma-ratio for products, ``mp.quad`` for
integrals -- so the 60- and 120-digit stages are exact evaluations, not extrapolations.

Prior-art labels are ``KNOWN_REDISCOVERED`` or ``NOT_IN_BUILTIN_TABLE`` against a finite
built-in table, never "novel"; absence from a finite table establishes nothing
(``corpus_absence_establishes_novelty: false``).  Adjudication against a real corpus is
:mod:`sigma_theory_compiler.families_prior_art_screen`, and proof routing is
:mod:`sigma_theory_compiler.families_proof_router`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-inverse-symbolic-families-result-1.0"

FAMILIES = ("S", "P", "I")


class FamilyError(ValueError):
    """Raised on malformed input, a failed control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Shared declarations
# ---------------------------------------------------------------------------

#: Target constants.  The CF run's list plus ``zeta(2)`` and ``pi^2`` stated explicitly.
TARGETS: tuple[dict[str, str], ...] = (
    {"name": "pi", "definition": "pi"},
    {"name": "e", "definition": "exp(1)"},
    {"name": "ln2", "definition": "log(2)"},
    {"name": "ln3", "definition": "log(3)"},
    {"name": "sqrt2", "definition": "sqrt(2)"},
    {"name": "sqrt3", "definition": "sqrt(3)"},
    {"name": "zeta2", "definition": "zeta(2) = pi^2/6"},
    {"name": "zeta3", "definition": "zeta(3)"},
    {"name": "catalan", "definition": "Catalan"},
    {"name": "euler_gamma", "definition": "EulerGamma"},
    {"name": "phi", "definition": "(1 + sqrt(5))/2"},
    {"name": "e_pi", "definition": "exp(pi)"},
    {"name": "pi_squared", "definition": "pi^2"},
)

TARGET_NAMES: tuple[str, ...] = tuple(item["name"] for item in TARGETS)

#: Rational prefactor applied to every family member.  It multiplies the ordinal space and
#: factorizes out of the evaluation, so the sweep tests every ordinal by comparing one
#: evaluated value against the declared ``target / prefactor`` grid.
PREFACTORS: tuple[Fraction, ...] = (
    Fraction(1),
    Fraction(2),
    Fraction(3),
    Fraction(4),
    Fraction(6),
    Fraction(8),
    Fraction(12),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 4),
    Fraction(1, 6),
    Fraction(1, 8),
)

#: Polynomial grammar shared by families S and P.
POLY_COEFFICIENT_VALUES = tuple(range(-4, 5))
POLY_RADIX = len(POLY_COEFFICIENT_VALUES)
POLY_SLOTS = 8  # two degree-3 polynomials
SHAPE_COUNT = POLY_RADIX**POLY_SLOTS  # 43,046,721

#: fp64 sweep parameters for the iterated families (S and P).
FP64_TERMS = 2048
FP64_CHECKPOINTS: tuple[int, ...] = tuple(2**j for j in range(4, 12))  # 16 .. 2048
FP64_MATCH_WINDOW = 1e-12
FP64_RESOLUTION_GATE = 1e-13
FP64_DECAY_TAIL_MAX = 1e-2
FP64_DECAY_RATIO_MAX = 0.5

#: The digit holdout.  A match found in fp64 is interpolation until it survives both.
VERIFY_STAGES: tuple[dict[str, Any], ...] = (
    {"stage": "mpmath_dps60", "dps": 60, "threshold": "1e-50"},
    {"stage": "mpmath_dps120", "dps": 120, "threshold": "1e-100"},
)

VALUE_RENDER_DPS = 120

SHARED_CLAIMS = {
    "match_at_fit_precision_is_not_discovery": True,
    "survival_at_verify_precision_is_conjecture_not_proof": True,
    "corpus_absence_establishes_novelty": False,
    "builtin_table_absence_establishes_novelty": False,
    "survivors_are_conjectures_not_theorems": True,
    "enumeration_exhaustive_over_declared_family": True,
    "family_classics_must_be_rediscovered_or_negatives_are_void": True,
}


def constant_value(name: str) -> mp.mpf:
    """Exact named target at the current mpmath working precision."""

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
    if name == "zeta2":
        return mp.zeta(2)
    if name == "zeta3":
        return mp.zeta(3)
    if name == "catalan":
        return +mp.catalan
    if name == "euler_gamma":
        return +mp.euler
    if name == "phi":
        return +mp.phi
    if name == "e_pi":
        return mp.exp(mp.pi)
    if name == "pi_squared":
        return mp.pi**2
    raise FamilyError(f"unknown constant: {name}")


TARGET_LATEX = {
    "pi": r"\pi",
    "e": "e",
    "ln2": r"\ln 2",
    "ln3": r"\ln 3",
    "sqrt2": r"\sqrt{2}",
    "sqrt3": r"\sqrt{3}",
    "zeta2": r"\zeta(2)",
    "zeta3": r"\zeta(3)",
    "catalan": "G",
    "euler_gamma": r"\gamma",
    "phi": r"\varphi",
    "e_pi": r"e^{\pi}",
    "pi_squared": r"\pi^{2}",
}

TARGET_TEXT = {
    "pi": "pi",
    "e": "e",
    "ln2": "ln(2)",
    "ln3": "ln(3)",
    "sqrt2": "sqrt(2)",
    "sqrt3": "sqrt(3)",
    "zeta2": "zeta(2)",
    "zeta3": "zeta(3)",
    "catalan": "Catalan",
    "euler_gamma": "EulerGamma",
    "phi": "phi",
    "e_pi": "e^pi",
    "pi_squared": "pi^2",
}

SYMBOLIC_TARGET: dict[str, sp.Expr] = {
    "pi": sp.pi,
    "e": sp.E,
    "ln2": sp.log(2),
    "ln3": sp.log(3),
    "sqrt2": sp.sqrt(2),
    "sqrt3": sp.sqrt(3),
    "zeta2": sp.zeta(2),
    "zeta3": sp.zeta(3),
    "catalan": sp.Catalan,
    "euler_gamma": sp.EulerGamma,
    "phi": (1 + sp.sqrt(5)) / 2,
    "e_pi": sp.exp(sp.pi),
    "pi_squared": sp.pi**2,
}


def comparison_grid() -> list[tuple[str, int, Fraction]]:
    """``(target_name, prefactor_index, target/prefactor)`` for every declared pair.

    The prefactor multiplies the family value, so an ordinal matches when its evaluated
    value lands on ``target / prefactor``.  Testing against this grid tests every ordinal.
    """

    return [
        (name, index, prefactor)
        for name in TARGET_NAMES
        for index, prefactor in enumerate(PREFACTORS)
    ]


def comparison_values(dps: int = 40) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """fp64 comparison values and their ``(target, prefactor_index)`` labels."""

    grid = comparison_grid()
    with mp.workdps(dps):
        values = [float(constant_value(name) / mp.mpf(pref.numerator) * pref.denominator)
                  for name, _index, pref in grid]
    return np.array(values, dtype=np.float64), [(name, index) for name, index, _p in grid]


# ---------------------------------------------------------------------------
# Neville extrapolation weights (the declared accelerator)
# ---------------------------------------------------------------------------


def neville_weights(nodes: Sequence[int]) -> np.ndarray:
    """Exact-rational Lagrange weights extrapolating ``S(h)`` at ``h = 1/n`` to ``h = 0``.

    For a sequence whose asymptotics are ``S_n = S + a_1/n + a_2/n^2 + ...`` the value at
    ``h = 0`` is ``sum_j w_j S_{n_j}``.  With geometric nodes the weights stay small -- the
    declared checkpoint set gives ``max |w_j| = 3.44`` -- so the extrapolation costs well
    under one decimal digit, unlike a consecutive-index Richardson scheme.
    """

    hs = [Fraction(1, int(n)) for n in nodes]
    weights: list[float] = []
    for j, hj in enumerate(hs):
        term = Fraction(1)
        for i, hi in enumerate(hs):
            if i != j:
                term *= (0 - hi) / (hj - hi)
        weights.append(float(term))
    return np.array(weights, dtype=np.float64)


NEVILLE_W8 = neville_weights(FP64_CHECKPOINTS)
NEVILLE_W6 = neville_weights(FP64_CHECKPOINTS[-6:])
NEVILLE_W8_NORM = float(np.abs(NEVILLE_W8).max())
NEVILLE_W6_NORM = float(np.abs(NEVILLE_W6).max())


# ---------------------------------------------------------------------------
# Family S -- hypergeometric-type series
# ---------------------------------------------------------------------------

S_Z_VALUES: tuple[Fraction, ...] = (Fraction(1), Fraction(-1), Fraction(1, 2))
S_SERIES_COUNT = SHAPE_COUNT * len(S_Z_VALUES)
S_ORDINAL_COUNT = S_SERIES_COUNT * len(PREFACTORS)

FAMILY_S_CONFIG: dict[str, Any] = {
    "family": "S",
    "title": "hypergeometric-type series",
    "grammar": (
        "c0 * sum_{k>=0} t_k with t_0 = 1 and t_{k+1}/t_k = z * P(k)/Q(k); "
        "P(k) = p0 + p1 k + p2 k^2 + p3 k^3; Q(k) = q0 + q1 k + q2 k^2 + q3 k^3"
    ),
    "polynomial_degree_max": 3,
    "polynomial_coefficient_range": [-4, 4],
    "coefficient_slots": POLY_SLOTS,
    "shape_count": SHAPE_COUNT,
    "shape_digit_order": "base-9 little-endian: p0, p1, p2, p3, q0, q1, q2, q3",
    "z_values": [str(z) for z in S_Z_VALUES],
    "prefactors": [str(c) for c in PREFACTORS],
    "ordinal_layout": (
        "ordinal = (shape_index * len(z) + z_index) * len(prefactor) + prefactor_index; "
        "the prefactor factorizes out of the sum, so the sweep evaluates "
        "shape_count * len(z) distinct series and decides every ordinal by comparing that "
        "value against the declared target/prefactor grid"
    ),
    "series_evaluated": S_SERIES_COUNT,
    "total_ordinals": S_ORDINAL_COUNT,
    "fp64_terms": FP64_TERMS,
    "fp64_checkpoints": list(FP64_CHECKPOINTS),
    "fp64_extrapolation": (
        "Neville extrapolation of the checkpoint partial sums in h = 1/n; order 8 over all "
        f"checkpoints (max |w| = {NEVILLE_W8_NORM:.3f}) and order 6 over the last six "
        f"(max |w| = {NEVILLE_W6_NORM:.3f})"
    ),
    "fp64_resolution_gate": f"|E8 - E6| < {FP64_RESOLUTION_GATE:g}",
    "fp64_match_window": f"{FP64_MATCH_WINDOW:g}",
    "fp64_decay_gate": (
        f"|t_(N-1)| < {FP64_DECAY_TAIL_MAX:g} and either |t_(N-1)| = 0 (the terms have "
        f"underflowed, which is convergence) or |t_(N-1)| < {FP64_DECAY_RATIO_MAX:g} * "
        "|t_(N/16-1)|; this rejects non-decaying oscillatory partial sums whose "
        "extrapolation would otherwise report an Abel-style value for a divergent series"
    ),
    "terminating_rule": (
        "if P(k) = 0 for some integer k >= 0 the series terminates and its value is a "
        "rational number; no declared target is rational, so terminating ordinals are "
        "counted separately and excluded from matching.  The test is structural (P(k) = 0 "
        "in exact integer arithmetic), never 'the fp64 term reached zero', which is "
        "underflow and is the correct behaviour of a factorially convergent series"
    ),
    "exact_evaluation": (
        "t_{k+1}/t_k = z P(k)/Q(k) makes the sum a generalized hypergeometric series: with "
        "A_i the negated roots of P and B_j the negated roots of Q, the value is "
        "pFq(A + [1] ; B ; z lead(P)/lead(Q)) where the [1] is dropped and one B_j = 1 is "
        "removed instead when Q(-1) = 0.  Roots are taken exactly with sympy (degree <= 3 "
        "is solvable in radicals) and the value is mp.hyper at the stage precision"
    ),
    "verify_stages": [dict(stage) for stage in VERIFY_STAGES],
    "value_render_dps": VALUE_RENDER_DPS,
    "targets": [dict(item) for item in TARGETS],
}


def decode_poly_shape(shape_index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shape index -> ``(P, Q)`` coefficient tuples, each low-order first."""

    if not 0 <= shape_index < SHAPE_COUNT:
        raise FamilyError(f"shape index out of range: {shape_index}")
    digits: list[int] = []
    value = shape_index
    for _ in range(POLY_SLOTS):
        digits.append(value % POLY_RADIX - 4)
        value //= POLY_RADIX
    return tuple(digits[:4]), tuple(digits[4:])


def encode_poly_shape(first: Sequence[int], second: Sequence[int]) -> int:
    """Inverse of :func:`decode_poly_shape`."""

    coefficients = list(first) + list(second)
    if len(coefficients) != POLY_SLOTS:
        raise FamilyError("two degree-3 coefficient vectors are required")
    index = 0
    for digit in reversed(coefficients):
        if int(digit) not in POLY_COEFFICIENT_VALUES:
            raise FamilyError(f"polynomial coefficient out of range: {digit}")
        index = index * POLY_RADIX + (int(digit) + 4)
    return index


def encode_s_ordinal(
    p: Sequence[int], q: Sequence[int], z: Fraction, prefactor: Fraction
) -> int:
    """``(P, Q, z, c0)`` -> Family S ordinal."""

    if z not in S_Z_VALUES:
        raise FamilyError(f"z not in the declared set: {z}")
    if prefactor not in PREFACTORS:
        raise FamilyError(f"prefactor not in the declared set: {prefactor}")
    shape = encode_poly_shape(p, q)
    series = shape * len(S_Z_VALUES) + S_Z_VALUES.index(z)
    return series * len(PREFACTORS) + PREFACTORS.index(prefactor)


def decode_s_ordinal(ordinal: int) -> dict[str, Any]:
    """Family S ordinal -> its declared components."""

    if not 0 <= ordinal < S_ORDINAL_COUNT:
        raise FamilyError(f"ordinal out of range: {ordinal}")
    series, prefactor_index = divmod(ordinal, len(PREFACTORS))
    shape_index, z_index = divmod(series, len(S_Z_VALUES))
    p, q = decode_poly_shape(shape_index)
    return {
        "ordinal": ordinal,
        "series_index": series,
        "shape_index": shape_index,
        "p": list(p),
        "q": list(q),
        "z": str(S_Z_VALUES[z_index]),
        "z_index": z_index,
        "prefactor": str(PREFACTORS[prefactor_index]),
        "prefactor_index": prefactor_index,
    }


# ---------------------------------------------------------------------------
# Family P -- infinite products
# ---------------------------------------------------------------------------

P_K0_VALUES: tuple[int, ...] = (1, 2, 3)
P_PRODUCT_COUNT = SHAPE_COUNT * len(P_K0_VALUES)
P_ORDINAL_COUNT = P_PRODUCT_COUNT * len(PREFACTORS)

FAMILY_P_CONFIG: dict[str, Any] = {
    "family": "P",
    "title": "infinite products",
    "grammar": (
        "c0 * prod_{k>=k0} A(k)/B(k); A(k) = a0 + a1 k + a2 k^2 + a3 k^3; "
        "B(k) = b0 + b1 k + b2 k^2 + b3 k^3"
    ),
    "shape_subsumption": (
        "the alternative declared shape R(k) = 1 + a/k^p + b/k^q with integer a, b and "
        "1 <= p, q <= 3 is a strict sub-case: multiplying through by k^max(p,q) gives "
        "A(k)/B(k) with B(k) = k^max(p,q), which is inside this grammar whenever the "
        "resulting coefficients lie in [-4, 4]"
    ),
    "polynomial_degree_max": 3,
    "polynomial_coefficient_range": [-4, 4],
    "coefficient_slots": POLY_SLOTS,
    "shape_count": SHAPE_COUNT,
    "shape_digit_order": "base-9 little-endian: a0, a1, a2, a3, b0, b1, b2, b3",
    "k0_values": list(P_K0_VALUES),
    "prefactors": [str(c) for c in PREFACTORS],
    "ordinal_layout": (
        "ordinal = (shape_index * len(k0) + k0_index) * len(prefactor) + prefactor_index"
    ),
    "products_evaluated": P_PRODUCT_COUNT,
    "total_ordinals": P_ORDINAL_COUNT,
    "convergence_test": (
        "log R(k) is summable exactly when deg A = deg B = d >= 1, lead(A) = lead(B) != 0 "
        "and the subleading coefficients agree; then log R(k) = O(1/k^2).  If the leading "
        "coefficients differ the product tends to 0 or infinity geometrically, and if only "
        "the subleading coefficients differ it diverges like n^(c) with c != 0.  This test "
        "is exact on the integer coefficients and is enforced before any evaluation"
    ),
    "positivity_test": (
        "every factor must be positive.  Cauchy's bound puts all roots of an integer "
        "polynomial with coefficients in [-4, 4] and leading coefficient of magnitude at "
        "least 1 strictly inside |k| < 5, so checking A(k) > 0 and B(k) > 0 for "
        "k = k0 .. 8 together with lead(A) = lead(B) proves positivity for all k >= k0"
    ),
    "log_transform": (
        "the GPU sweep accumulates the partial product directly rather than a sum of logs: "
        "for a convergent member P_n = P (1 + d1/n + d2/n^2 + ...), so the same Neville "
        "extrapolation in h = 1/n applies, and taking no logarithm avoids 2048 transcendental "
        "evaluations per ordinal.  The summability of the log terms is what the exact "
        "convergence test above certifies"
    ),
    "fp64_terms": FP64_TERMS,
    "fp64_checkpoints": list(FP64_CHECKPOINTS),
    "fp64_extrapolation": FAMILY_S_CONFIG["fp64_extrapolation"],
    "fp64_resolution_gate": f"|E8 - E6| < {FP64_RESOLUTION_GATE:g}",
    "fp64_match_window": f"{FP64_MATCH_WINDOW:g}",
    "exact_evaluation": (
        "for a member passing the convergence test, prod_{k>=k0} A(k)/B(k) = "
        "prod_j Gamma(k0 - beta_j) / prod_i Gamma(k0 - alpha_i) over the roots alpha of A "
        "and beta of B; equal leading coefficients and equal root sums make the "
        "Gamma-ratio limit exactly 1, which is the classical Weierstrass-product argument"
    ),
    "verify_stages": [dict(stage) for stage in VERIFY_STAGES],
    "value_render_dps": VALUE_RENDER_DPS,
    "targets": [dict(item) for item in TARGETS],
}


def encode_p_ordinal(
    a: Sequence[int], b: Sequence[int], k0: int, prefactor: Fraction
) -> int:
    """``(A, B, k0, c0)`` -> Family P ordinal."""

    if k0 not in P_K0_VALUES:
        raise FamilyError(f"k0 not in the declared set: {k0}")
    if prefactor not in PREFACTORS:
        raise FamilyError(f"prefactor not in the declared set: {prefactor}")
    shape = encode_poly_shape(a, b)
    product = shape * len(P_K0_VALUES) + P_K0_VALUES.index(k0)
    return product * len(PREFACTORS) + PREFACTORS.index(prefactor)


def decode_p_ordinal(ordinal: int) -> dict[str, Any]:
    """Family P ordinal -> its declared components."""

    if not 0 <= ordinal < P_ORDINAL_COUNT:
        raise FamilyError(f"ordinal out of range: {ordinal}")
    product, prefactor_index = divmod(ordinal, len(PREFACTORS))
    shape_index, k0_index = divmod(product, len(P_K0_VALUES))
    a, b = decode_poly_shape(shape_index)
    return {
        "ordinal": ordinal,
        "product_index": product,
        "shape_index": shape_index,
        "a": list(a),
        "b": list(b),
        "k0": P_K0_VALUES[k0_index],
        "k0_index": k0_index,
        "prefactor": str(PREFACTORS[prefactor_index]),
        "prefactor_index": prefactor_index,
    }


def poly_degree(coefficients: Sequence[int]) -> int:
    """Degree of an integer coefficient vector, or ``-1`` for the zero polynomial."""

    for index in range(len(coefficients) - 1, -1, -1):
        if int(coefficients[index]) != 0:
            return index
    return -1


def product_converges(a: Sequence[int], b: Sequence[int], k0: int) -> tuple[bool, str]:
    """The exact convergence and positivity test declared in :data:`FAMILY_P_CONFIG`."""

    da, db = poly_degree(a), poly_degree(b)
    if da < 1 or db < 1:
        return False, "degenerate_degree"
    if da != db:
        return False, "degree_mismatch"
    if int(a[da]) != int(b[db]):
        return False, "leading_coefficient_mismatch"
    if int(a[da - 1]) != int(b[db - 1]):
        return False, "subleading_coefficient_mismatch"
    for k in range(k0, 9):
        av = sum(int(c) * k**i for i, c in enumerate(a))
        bv = sum(int(c) * k**i for i, c in enumerate(b))
        if av <= 0 or bv <= 0:
            return False, "factor_not_positive"
    return True, "converges"


# ---------------------------------------------------------------------------
# Family I -- definite integrals
# ---------------------------------------------------------------------------

#: Declared kernels.  ``order0``/``order1`` are the algebraic orders at ``x = 0`` and
#: ``x = 1``; ``log0``/``log1`` flag a logarithmic factor there.  The regular part -- the
#: kernel divided by ``x^order0 (1-x)^order1`` -- is bounded above and away from zero on the
#: closed interval and is what the quadrature tables actually store.
#:
#: ``pure_power`` marks a kernel that *is* its singular part, so raising it to ``c`` only
#: shifts the two exponents; ``base``/``base_multiplier`` record that a kernel is a fixed
#: power of another declared kernel.  Both are exact degeneracies of the declared grammar
#: and are collapsed by :func:`canonical_form`, so survivors can be counted as objects.
INTEGRAL_KERNELS: tuple[dict[str, Any], ...] = (
    {"index": 0, "name": "one", "expr": "1", "latex": "1",
     "order0": Fraction(0), "order1": Fraction(0), "log0": False, "log1": False,
     "pure_power": True,  "base": "one", "base_multiplier": "1"},
    {"index": 1, "name": "log_inv", "expr": "ln(1/x)", "latex": r"\ln(1/x)",
     "order0": Fraction(0), "order1": Fraction(1), "log0": True, "log1": False,
     "pure_power": False, "base": "log_inv", "base_multiplier": "1"},
    {"index": 2, "name": "inv_1px", "expr": "1/(1+x)", "latex": r"\frac{1}{1+x}",
     "order0": Fraction(0), "order1": Fraction(0), "log0": False, "log1": False,
     "pure_power": False, "base": "inv_1px", "base_multiplier": "1"},
    {"index": 3, "name": "inv_1px2", "expr": "1/(1+x^2)", "latex": r"\frac{1}{1+x^{2}}",
     "order0": Fraction(0), "order1": Fraction(0), "log0": False, "log1": False,
     "pure_power": False, "base": "inv_1px2", "base_multiplier": "1"},
    {"index": 4, "name": "inv_1mx", "expr": "1/(1-x)", "latex": r"\frac{1}{1-x}",
     "order0": Fraction(0), "order1": Fraction(-1), "log0": False, "log1": False,
     "pure_power": True,  "base": "one", "base_multiplier": "1"},
    {"index": 5, "name": "log_inv_over_1mx", "expr": "ln(1/x)/(1-x)",
     "latex": r"\frac{\ln(1/x)}{1-x}",
     "order0": Fraction(0), "order1": Fraction(0), "log0": True, "log1": False,
     "pure_power": False, "base": "log_inv_over_1mx", "base_multiplier": "1"},
    {"index": 6, "name": "log_inv_over_1px", "expr": "ln(1/x)/(1+x)",
     "latex": r"\frac{\ln(1/x)}{1+x}",
     "order0": Fraction(0), "order1": Fraction(1), "log0": True, "log1": False,
     "pure_power": False, "base": "log_inv_over_1px", "base_multiplier": "1"},
    {"index": 7, "name": "log_inv_over_1px2", "expr": "ln(1/x)/(1+x^2)",
     "latex": r"\frac{\ln(1/x)}{1+x^{2}}",
     "order0": Fraction(0), "order1": Fraction(1), "log0": True, "log1": False,
     "pure_power": False, "base": "log_inv_over_1px2", "base_multiplier": "1"},
    {"index": 8, "name": "log_inv_1mx", "expr": "ln(1/(1-x))", "latex": r"\ln\!\frac{1}{1-x}",
     "order0": Fraction(1), "order1": Fraction(0), "log0": False, "log1": True,
     "pure_power": False, "base": "log_inv_1mx", "base_multiplier": "1"},
    {"index": 9, "name": "x_over_expm1", "expr": "x/(exp(x)-1)",
     "latex": r"\frac{x}{e^{x}-1}",
     "order0": Fraction(0), "order1": Fraction(0), "log0": False, "log1": False,
     "pure_power": False, "base": "x_over_expm1", "base_multiplier": "1"},
    {"index": 10, "name": "inv_1pxpx2", "expr": "1/(1+x+x^2)",
     "latex": r"\frac{1}{1+x+x^{2}}",
     "order0": Fraction(0), "order1": Fraction(0), "log0": False, "log1": False,
     "pure_power": False, "base": "inv_1pxpx2", "base_multiplier": "1"},
    {"index": 11, "name": "inv_sqrt_1px", "expr": "1/sqrt(1+x)",
     "latex": r"\frac{1}{\sqrt{1+x}}",
     "order0": Fraction(0), "order1": Fraction(0), "log0": False, "log1": False,
     "pure_power": False, "base": "inv_1px", "base_multiplier": "1/2"},
)

I_KERNEL_COUNT = len(INTEGRAL_KERNELS)

#: Kernel powers.  ``0`` removes the kernel entirely and gives the pure Beta sub-family.
I_POWERS: tuple[Fraction, ...] = (
    Fraction(0),
    Fraction(1, 2),
    Fraction(1),
    Fraction(3, 2),
    Fraction(2),
    Fraction(5, 2),
    Fraction(3),
    Fraction(4),
)

#: Exponent grid: ``n/12`` for ``n = -12 .. 347``.  ``-1`` is included because kernels with a
#: positive order at ``x = 1`` compensate it -- that is exactly the ``zeta(2)`` classic.
I_EXPONENT_NUMERATORS: tuple[int, ...] = tuple(range(-12, 348))
I_EXPONENT_DENOMINATOR = 12
I_EXPONENTS: tuple[Fraction, ...] = tuple(
    Fraction(n, I_EXPONENT_DENOMINATOR) for n in I_EXPONENT_NUMERATORS
)
I_GRID = len(I_EXPONENTS)

I_INTEGRAL_COUNT = I_GRID * I_GRID * I_KERNEL_COUNT * len(I_POWERS)
I_ORDINAL_COUNT = I_INTEGRAL_COUNT * len(PREFACTORS)

#: tanh-sinh quadrature.  ``x = (1 + tanh((pi/2) sinh t))/2`` with step ``2^-level`` over
#: ``|t| <= cut``.  The cut is chosen so the neglected tail of the most singular admitted
#: integrand (effective exponent ``1/12 - 1``) is below 1e-21 in relative terms.
I_QUAD_LEVEL = 6
I_QUAD_CUT = 6.0

FAMILY_I_CONFIG: dict[str, Any] = {
    "family": "I",
    "title": "definite integrals",
    "grammar": "c0 * int_0^1 x^a (1-x)^b K_m(x)^c dx",
    "kernels": [
        {
            "index": item["index"],
            "name": item["name"],
            "expr": item["expr"],
            "order_at_0": str(item["order0"]),
            "order_at_1": str(item["order1"]),
            "logarithmic_at_0": item["log0"],
            "logarithmic_at_1": item["log1"],
        }
        for item in INTEGRAL_KERNELS
    ],
    "kernel_powers": [str(c) for c in I_POWERS],
    "exponent_grid": (
        f"n/{I_EXPONENT_DENOMINATOR} for n = {I_EXPONENT_NUMERATORS[0]} .. "
        f"{I_EXPONENT_NUMERATORS[-1]} ({I_GRID} values, from "
        f"{I_EXPONENTS[0]} to {I_EXPONENTS[-1]})"
    ),
    "exponent_grid_size": I_GRID,
    "prefactors": [str(c) for c in PREFACTORS],
    "ordinal_layout": (
        "ordinal = (((a_index * grid + b_index) * kernels + kernel_index) * powers + "
        "power_index) * len(prefactor) + prefactor_index"
    ),
    "integrals_evaluated": I_INTEGRAL_COUNT,
    "total_ordinals": I_ORDINAL_COUNT,
    "convergence_test": (
        "strict endpoint conditions on the effective algebraic exponents: "
        "a + c * order_at_0(K) > -1 and b + c * order_at_1(K) > -1.  A logarithmic factor "
        "with a strictly admissible algebraic exponent is integrable, so the strict "
        "inequality is sufficient; equality is rejected rather than case-split"
    ),
    "quadrature": (
        f"tanh-sinh (double-exponential): x = (1 + tanh((pi/2) sinh t))/2, step 2^-"
        f"{I_QUAD_LEVEL}, |t| <= {I_QUAD_CUT}.  ln x and ln(1-x) are computed directly from "
        "t as -log1p(exp(-2u)) and -log1p(exp(2u)) so no cancellation occurs at either "
        "endpoint, and the Jacobian x(1-x) is folded into the exponents, giving the exact "
        "node form C_j exp((a+1) ln x_j + (b+1) ln(1-x_j) + c ln Ktilde_j)"
    ),
    "quadrature_nodes": None,  # filled in at import from the built table
    "quadrature_error_bound": (
        "double-exponential quadrature converges as exp(-c N / log N) for integrands "
        "analytic on (0,1) with integrable algebraic endpoint singularities, which the "
        "convergence test guarantees.  The declared node set reproduces B(1/2, 1/2) = pi, "
        "int_0^1 dx/(1+x^2) = pi/4 and int_0^1 -ln(x)/(1-x) dx = zeta(2) to fp64 rounding, "
        "and the truncation tail beyond |t| = cut is below 1e-21 relative for the most "
        "singular admitted integrand"
    ),
    "fp64_match_window": f"{FP64_MATCH_WINDOW:g}",
    "sweep_organisation": (
        "the kernel's singular part x^(c order0) (1-x)^(c order1) is folded into the "
        "exponents, leaving a bounded regular part, so each (kernel, power) pair becomes a "
        "single fp64 GEMM over the two exponent grids"
    ),
    "exact_evaluation": "mp.quad on the log-domain integrand at the stage precision",
    "verify_stages": [dict(stage) for stage in VERIFY_STAGES],
    "value_render_dps": VALUE_RENDER_DPS,
    "targets": [dict(item) for item in TARGETS],
}


def encode_i_ordinal(
    a: Fraction, b: Fraction, kernel: int, power: Fraction, prefactor: Fraction
) -> int:
    """``(a, b, kernel, c, c0)`` -> Family I ordinal."""

    if a not in I_EXPONENTS or b not in I_EXPONENTS:
        raise FamilyError(f"exponent off the declared grid: {a}, {b}")
    if not 0 <= kernel < I_KERNEL_COUNT:
        raise FamilyError(f"kernel index out of range: {kernel}")
    if power not in I_POWERS:
        raise FamilyError(f"kernel power not in the declared set: {power}")
    if prefactor not in PREFACTORS:
        raise FamilyError(f"prefactor not in the declared set: {prefactor}")
    index = I_EXPONENTS.index(a) * I_GRID + I_EXPONENTS.index(b)
    index = index * I_KERNEL_COUNT + kernel
    index = index * len(I_POWERS) + I_POWERS.index(power)
    return index * len(PREFACTORS) + PREFACTORS.index(prefactor)


def decode_i_ordinal(ordinal: int) -> dict[str, Any]:
    """Family I ordinal -> its declared components."""

    if not 0 <= ordinal < I_ORDINAL_COUNT:
        raise FamilyError(f"ordinal out of range: {ordinal}")
    rest, prefactor_index = divmod(ordinal, len(PREFACTORS))
    integral_index = rest
    rest, power_index = divmod(rest, len(I_POWERS))
    rest, kernel_index = divmod(rest, I_KERNEL_COUNT)
    a_index, b_index = divmod(rest, I_GRID)
    return {
        "ordinal": ordinal,
        "integral_index": integral_index,
        "a": str(I_EXPONENTS[a_index]),
        "b": str(I_EXPONENTS[b_index]),
        "a_index": a_index,
        "b_index": b_index,
        "kernel": INTEGRAL_KERNELS[kernel_index]["name"],
        "kernel_index": kernel_index,
        "power": str(I_POWERS[power_index]),
        "power_index": power_index,
        "prefactor": str(PREFACTORS[prefactor_index]),
        "prefactor_index": prefactor_index,
    }


def integral_converges(a: Fraction, b: Fraction, kernel_index: int, power: Fraction) -> bool:
    """The declared strict endpoint test."""

    kernel = INTEGRAL_KERNELS[kernel_index]
    return bool(
        a + power * kernel["order0"] > -1 and b + power * kernel["order1"] > -1
    )


# ---------------------------------------------------------------------------
# tanh-sinh node table (shared by the GPU sweep and its numpy mirror)
# ---------------------------------------------------------------------------


def _kernel_regular_log(name: str, x: np.ndarray, lx: np.ndarray, l1x: np.ndarray) -> np.ndarray:
    """``ln`` of the kernel divided by its declared endpoint singular part."""

    if name == "one":
        return np.zeros_like(x)
    if name == "log_inv":
        # ln(1/x) = (1-x) * [ -ln(x)/(1-x) ]; the bracket is bounded on [0, 1].
        return np.log(-lx) - l1x
    if name == "inv_1px":
        return -np.log1p(x)
    if name == "inv_1px2":
        return -np.log1p(x * x)
    if name == "inv_1mx":
        # 1/(1-x) = (1-x)^-1 * 1
        return np.zeros_like(x)
    if name == "log_inv_over_1mx":
        return np.log(-lx) - l1x
    if name == "log_inv_over_1px":
        return np.log(-lx) - l1x - np.log1p(x)
    if name == "log_inv_over_1px2":
        return np.log(-lx) - l1x - np.log1p(x * x)
    if name == "log_inv_1mx":
        # ln(1/(1-x)) = x * [ -ln(1-x)/x ]; the bracket is bounded on [0, 1].
        return np.log(-l1x) - lx
    if name == "x_over_expm1":
        return np.log(x) - np.log(np.expm1(x))
    if name == "inv_1pxpx2":
        return -np.log1p(x + x * x)
    if name == "inv_sqrt_1px":
        return -0.5 * np.log1p(x)
    raise FamilyError(f"unknown kernel: {name}")


class QuadratureTable:
    """Declared tanh-sinh nodes plus the exponent and kernel tables the sweep multiplies."""

    def __init__(self, level: int = I_QUAD_LEVEL, cut: float = I_QUAD_CUT) -> None:
        step = 1.0 / (2**level)
        t = np.arange(-cut, cut + step / 4, step)
        u = (math.pi / 2) * np.sinh(t)
        self.log_x = -np.logaddexp(0.0, -2 * u)
        self.log_1mx = -np.logaddexp(0.0, 2 * u)
        self.prefactor = step * math.pi * np.cosh(t)
        self.x = np.exp(self.log_x)
        keep = (
            np.isfinite(self.prefactor)
            & np.isfinite(self.log_x)
            & np.isfinite(self.log_1mx)
            & (self.log_x < 0)
            & (self.log_1mx < 0)
        )
        self.t = t[keep]
        self.log_x = self.log_x[keep]
        self.log_1mx = self.log_1mx[keep]
        self.prefactor = self.prefactor[keep]
        self.x = self.x[keep]
        self.level = level
        self.cut = cut
        self.nodes = int(self.t.shape[0])
        self.kernel_log = {
            item["name"]: _kernel_regular_log(item["name"], self.x, self.log_x, self.log_1mx)
            for item in INTEGRAL_KERNELS
        }

    def integrand_row(self, a: Fraction, b: Fraction, kernel_index: int, power: Fraction) -> np.ndarray:
        """Per-node contributions of one integral, before summation."""

        kernel = INTEGRAL_KERNELS[kernel_index]
        exponent_a = float(a + power * kernel["order0"]) + 1.0
        exponent_b = float(b + power * kernel["order1"]) + 1.0
        argument = exponent_a * self.log_x + exponent_b * self.log_1mx
        if power != 0:
            argument = argument + float(power) * self.kernel_log[kernel["name"]]
        return self.prefactor * np.exp(np.minimum(argument, 700.0))

    def quadrature(self, a: Fraction, b: Fraction, kernel_index: int, power: Fraction) -> float:
        """One integral in fp64 by the declared node set."""

        return float(self.integrand_row(a, b, kernel_index, power).sum())


QUADRATURE = QuadratureTable()
FAMILY_I_CONFIG["quadrature_nodes"] = QUADRATURE.nodes

FAMILY_CONFIG: dict[str, dict[str, Any]] = {
    "S": FAMILY_S_CONFIG,
    "P": FAMILY_P_CONFIG,
    "I": FAMILY_I_CONFIG,
}

FAMILY_ORDINALS = {"S": S_ORDINAL_COUNT, "P": P_ORDINAL_COUNT, "I": I_ORDINAL_COUNT}


# ---------------------------------------------------------------------------
# Exact evaluation (mpmath) -- the digit-holdout stages
# ---------------------------------------------------------------------------

_K = sp.Symbol("k")


@lru_cache(maxsize=8192)
def _exact_roots_cached(coefficients: tuple[int, ...]) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    values = [sp.Integer(int(v)) for v in coefficients]
    while values and values[-1] == 0:
        values.pop()
    if not values:
        return sp.Integer(0), ()
    if len(values) == 1:
        return values[0], ()
    poly = sp.Poly(sum(values[i] * _K**i for i in range(len(values))), _K)
    found = sp.roots(poly)
    roots: list[sp.Expr] = []
    for root, multiplicity in found.items():
        roots.extend([root] * int(multiplicity))
    if len(roots) != poly.degree():
        roots = list(poly.all_roots())
    return values[-1], tuple(roots)


def exact_roots(coefficients: Sequence[int]) -> tuple[sp.Expr, list[sp.Expr]]:
    """``(leading coefficient, roots with multiplicity)`` of an integer polynomial.

    Degree at most three is solvable in radicals, so ``sympy.roots`` returns exact roots and
    their multiplicities; the multiplicity path matters because a numeric root finder loses
    precision exactly on the repeated roots the classical series have (Basel's ``(k+1)^2``).
    """

    lead, roots = _exact_roots_cached(tuple(int(v) for v in coefficients))
    return lead, list(roots)


def _to_mp(expression: sp.Expr) -> mp.mpf | mp.mpc:
    """Evaluate an exact sympy number at the current mpmath precision."""

    value = sp.N(expression, mp.mp.dps + 10)
    real, imaginary = sp.re(value), sp.im(value)
    if imaginary == 0:
        return mp.mpf(str(real))
    return mp.mpc(str(real), str(imaginary))


def series_hypergeometric_parameters(
    p: Sequence[int], q: Sequence[int], z: Fraction
) -> tuple[list[sp.Expr], list[sp.Expr], sp.Expr] | None:
    """Exact ``pFq`` parameters of a Family S member, or ``None`` when ``P`` vanishes."""

    lead_p, roots_p = exact_roots(p)
    lead_q, roots_q = exact_roots(q)
    if lead_q == 0:
        raise FamilyError("Q is identically zero")
    if lead_p == 0:
        return None
    upper = [-root for root in roots_p]
    lower = [-root for root in roots_q]
    argument = sp.Rational(z.numerator, z.denominator) * sp.Rational(int(lead_p), int(lead_q))
    for index, value in enumerate(lower):
        if sp.simplify(value - 1) == 0:
            return upper, lower[:index] + lower[index + 1 :], argument
    return [*upper, sp.Integer(1)], lower, argument


def series_terminates(p: Sequence[int]) -> int | None:
    """The smallest ``k >= 0`` with ``P(k) = 0``, or ``None``.  Such a series is rational."""

    degree = poly_degree(p)
    if degree < 0:
        return 0
    # Cauchy's bound: |root| < 1 + max|coefficient| / |leading| <= 5.
    for k in range(6):
        if sum(int(c) * k**i for i, c in enumerate(p)) == 0:
            return k
    return None


def series_value_mp(p: Sequence[int], q: Sequence[int], z: Fraction) -> mp.mpf:
    """Exact Family S value at the current working precision."""

    guard = mp.mp.dps + 25
    with mp.workdps(guard):
        parameters = series_hypergeometric_parameters(p, q, z)
        if parameters is None:
            return mp.mpf(1)
        upper, lower, argument = parameters
        value = mp.hyper(
            [_to_mp(item) for item in upper],
            [_to_mp(item) for item in lower],
            _to_mp(argument),
        )
        return +mp.mpf(mp.re(value))


def product_value_mp(a: Sequence[int], b: Sequence[int], k0: int) -> mp.mpf:
    """Exact Family P value at the current working precision (Gamma-ratio closed form)."""

    guard = mp.mp.dps + 25
    with mp.workdps(guard):
        _lead_a, roots_a = exact_roots(a)
        _lead_b, roots_b = exact_roots(b)
        value = mp.mpc(1)
        for root in roots_b:
            value *= mp.gamma(mp.mpf(k0) - _to_mp(root))
        for root in roots_a:
            value /= mp.gamma(mp.mpf(k0) - _to_mp(root))
        return +mp.mpf(mp.re(value))


def integral_kernel_mp(name: str) -> Callable[[mp.mpf], mp.mpf]:
    """The declared kernel as an mpmath callable on ``(0, 1)``."""

    if name == "one":
        return lambda x: mp.mpf(1)
    if name == "log_inv":
        return lambda x: -mp.log(x)
    if name == "inv_1px":
        return lambda x: 1 / (1 + x)
    if name == "inv_1px2":
        return lambda x: 1 / (1 + x * x)
    if name == "inv_1mx":
        return lambda x: 1 / (1 - x)
    if name == "log_inv_over_1mx":
        return lambda x: -mp.log(x) / (1 - x)
    if name == "log_inv_over_1px":
        return lambda x: -mp.log(x) / (1 + x)
    if name == "log_inv_over_1px2":
        return lambda x: -mp.log(x) / (1 + x * x)
    if name == "log_inv_1mx":
        return lambda x: -mp.log1p(-x)
    if name == "x_over_expm1":
        return lambda x: x / mp.expm1(x)
    if name == "inv_1pxpx2":
        return lambda x: 1 / (1 + x + x * x)
    if name == "inv_sqrt_1px":
        return lambda x: 1 / mp.sqrt(1 + x)
    raise FamilyError(f"unknown kernel: {name}")


def kernel_regular_log_mp(name: str, u: mp.mpf, log_x: mp.mpf, log_1mx: mp.mpf) -> mp.mpf:
    """``ln`` of the kernel divided by its declared endpoint singular part, at one node.

    ``u`` is the tanh-sinh inner variable, so ``x`` and ``1-x`` are recovered from their
    logarithms and never by subtraction -- the regular part is bounded on the closed
    interval, which is what makes the node sum free of cancellation.
    """

    x = mp.exp(log_x)
    if name == "one":
        return mp.mpf(0)
    if name == "log_inv":
        return mp.log(-log_x) - log_1mx
    if name == "inv_1px":
        return -mp.log1p(x)
    if name == "inv_1px2":
        return -mp.log1p(x * x)
    if name == "inv_1mx":
        return mp.mpf(0)
    if name == "log_inv_over_1mx":
        return mp.log(-log_x) - log_1mx
    if name == "log_inv_over_1px":
        return mp.log(-log_x) - log_1mx - mp.log1p(x)
    if name == "log_inv_over_1px2":
        return mp.log(-log_x) - log_1mx - mp.log1p(x * x)
    if name == "log_inv_1mx":
        return mp.log(-log_1mx) - log_x
    if name == "x_over_expm1":
        return log_x - mp.log(mp.expm1(x))
    if name == "inv_1pxpx2":
        return -mp.log1p(x + x * x)
    if name == "inv_sqrt_1px":
        return -mp.log1p(x) / 2
    raise FamilyError(f"unknown kernel: {name}")


#: tanh-sinh step at the verification stages.  Halving it again changes no declared survivor
#: value in the last 20 digits, which the test suite pins.
MP_QUAD_LEVEL = 8


def integral_value_mp(
    a: Fraction, b: Fraction, kernel_index: int, power: Fraction
) -> mp.mpf:
    """Exact Family I value at the current working precision.

    Two paths, both exact rather than adaptive:

    * when the declared grammar degeneracies collapse the kernel away, the integral *is*
      Euler's Beta integral and is returned as ``B(a+1, b+1)`` -- no quadrature at all,
      which matters because those are exactly the strongly singular integrands;
    * otherwise a tanh-sinh rule in the log domain, with the abscissa range solved from the
      requested precision so the neglected tail is below ``10^-(dps+15)``.  ``mpmath.quad``
      is *not* used here: it caps its abscissa range, and that truncation -- not the node
      density -- costs 70 digits on an ``x^(-11/12)`` endpoint, which would silently kill
      true identities at the verification stages.
    """

    member = {
        "a": str(a), "b": str(b),
        "kernel": INTEGRAL_KERNELS[kernel_index]["name"],
        "kernel_index": kernel_index,
        "power": str(power), "prefactor": "1",
    }
    form = canonical_form("I", member)
    exponent_a = Fraction(form["effective_a"])
    exponent_b = Fraction(form["effective_b"])
    kernel_name = str(form["effective_kernel"])
    kernel_power = Fraction(form["effective_power"])
    guard = mp.mp.dps + 25
    with mp.workdps(guard):
        alpha = mp.mpf(exponent_a.numerator) / exponent_a.denominator + 1
        beta = mp.mpf(exponent_b.numerator) / exponent_b.denominator + 1
        if kernel_name == "one" or kernel_power == 0:
            return +mp.mpf(mp.beta(alpha, beta))

        entry = next(item for item in INTEGRAL_KERNELS if item["name"] == kernel_name)
        order0 = Fraction(entry["order0"]) * kernel_power
        order1 = Fraction(entry["order1"]) * kernel_power
        alpha += mp.mpf(order0.numerator) / order0.denominator
        beta += mp.mpf(order1.numerator) / order1.denominator
        if alpha <= 0 or beta <= 0:
            raise FamilyError("the integral does not converge at an endpoint")

        # Solve the abscissa range from the requested precision: the integrand decays like
        # exp(alpha * ln x) at one end and exp(beta * ln(1-x)) at the other, and ln x = -2u.
        need = mp.mpf(mp.mp.dps + 15) * mp.log(10)
        u_max = max(need / (2 * alpha), need / (2 * beta))
        cut = mp.asinh(2 * u_max / mp.pi)
        step = mp.mpf(1) / (2**MP_QUAD_LEVEL)
        count = int(mp.ceil(cut / step))
        exponent_c = mp.mpf(kernel_power.numerator) / kernel_power.denominator
        total = mp.mpf(0)
        for index in range(-count, count + 1):
            t = index * step
            inner = (mp.pi / 2) * mp.sinh(t)
            if inner >= 0:
                tail = mp.log1p(mp.exp(-2 * inner))
                log_x = -tail
                log_1mx = -(2 * inner + tail)
            else:
                tail = mp.log1p(mp.exp(2 * inner))
                log_1mx = -tail
                log_x = 2 * inner - tail
            argument = alpha * log_x + beta * log_1mx
            argument += exponent_c * kernel_regular_log_mp(kernel_name, inner, log_x, log_1mx)
            total += step * mp.pi * mp.cosh(t) * mp.exp(argument)
        return +mp.mpf(total)


def family_value_mp(family: str, member: Mapping[str, Any]) -> mp.mpf:
    """Exact value of one family member (without its prefactor) at the current precision."""

    if family == "S":
        return series_value_mp(member["p"], member["q"], Fraction(member["z"]))
    if family == "P":
        return product_value_mp(member["a"], member["b"], int(member["k0"]))
    if family == "I":
        return integral_value_mp(
            Fraction(member["a"]),
            Fraction(member["b"]),
            int(member["kernel_index"]),
            Fraction(member["power"]),
        )
    raise FamilyError(f"unknown family: {family}")


# ---------------------------------------------------------------------------
# The digit holdout
# ---------------------------------------------------------------------------


def _error_string(error: mp.mpf, dps: int) -> str:
    if not mp.isfinite(error):
        return "nan"
    if error == 0:
        return f"<1e-{dps}"
    return mp.nstr(error, 3)


def survival_trail(
    family: str,
    member: Mapping[str, Any],
    target_name: str,
    prefactor: Fraction,
    fp64_abs_error: float,
) -> tuple[list[dict[str, Any]], str]:
    """fp64 match -> 60-digit stage -> 120-digit stage, with the trail recorded.

    Returns ``(trail, status)`` where status is ``SURVIVED_ALL_STAGES``,
    ``DIED_AT_<stage>``, or ``UNEVALUABLE_AT_<stage>`` when the exact evaluator itself
    could not produce a value -- which is a declared limitation, never a survival.
    """

    trail: list[dict[str, Any]] = [
        {
            "stage": f"fp64_terms{FP64_TERMS}" if family != "I" else f"fp64_quad{QUADRATURE.nodes}",
            "abs_error": format(fp64_abs_error, ".3e"),
            "threshold": f"{FP64_MATCH_WINDOW:g}",
            "passed": True,
        }
    ]
    for stage in VERIFY_STAGES:
        with mp.workdps(stage["dps"]):
            try:
                value = family_value_mp(family, member)
                scaled = value * mp.mpf(prefactor.numerator) / prefactor.denominator
                error = abs(scaled - constant_value(target_name))
            except Exception as failure:  # noqa: BLE001 - any evaluator failure is a blocker
                trail.append(
                    {
                        "stage": stage["stage"],
                        "abs_error": "unevaluable",
                        "threshold": stage["threshold"],
                        "passed": False,
                        "blocker": f"{type(failure).__name__}: {failure}"[:200],
                    }
                )
                return trail, f"UNEVALUABLE_AT_{stage['stage'].upper()}"
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


# ---------------------------------------------------------------------------
# Built-in known tables (finite, explicit; absence from them is never novelty)
# ---------------------------------------------------------------------------

BUILTIN_S: tuple[dict[str, Any], ...] = (
    {"id": "exp_series", "p": [1, 0, 0, 0], "q": [1, 1, 0, 0], "z": "1", "prefactor": "1",
     "target": "e", "value": "e = sum_{k>=0} 1/k!",
     "source_note": "Euler, Introductio in analysin infinitorum (1748), the exponential series"},
    {"id": "basel_series", "p": [1, 2, 1, 0], "q": [4, 4, 1, 0], "z": "1", "prefactor": "1",
     "target": "zeta2", "value": "zeta(2) = sum_{k>=1} 1/k^2",
     "source_note": "Euler 1735, the Basel problem; sum_{k>=1} 1/k^2 = pi^2/6"},
    {"id": "basel_series_pi2", "p": [1, 2, 1, 0], "q": [4, 4, 1, 0], "z": "1", "prefactor": "6",
     "target": "pi_squared", "value": "pi^2 = 6 sum_{k>=1} 1/k^2",
     "source_note": "Euler 1735, the Basel problem, scaled to pi^2"},
    {"id": "leibniz_gregory", "p": [1, 2, 0, 0], "q": [3, 2, 0, 0], "z": "-1", "prefactor": "4",
     "target": "pi", "value": "pi = 4 sum_{k>=0} (-1)^k/(2k+1)",
     "source_note": "Gregory 1671 / Leibniz 1674, the arctangent series at 1"},
    {"id": "zeta3_series", "p": [1, 3, 3, 1], "q": [8, 12, 6, 1], "z": "1", "prefactor": "1",
     "target": "zeta3", "value": "zeta(3) = sum_{k>=1} 1/k^3",
     "source_note": "the defining Dirichlet series of zeta at 3 (Apery's constant)"},
    {"id": "mercator_alternating_ln2", "p": [1, 1, 0, 0], "q": [2, 1, 0, 0], "z": "-1",
     "prefactor": "1", "target": "ln2", "value": "ln 2 = sum_{k>=0} (-1)^k/(k+1)",
     "source_note": "Mercator 1668, the alternating harmonic series"},
    {"id": "catalan_series", "p": [1, 4, 4, 0], "q": [9, 12, 4, 0], "z": "-1", "prefactor": "1",
     "target": "catalan", "value": "G = sum_{k>=0} (-1)^k/(2k+1)^2",
     "source_note": "Catalan 1865, the defining series of Catalan's constant"},
    {"id": "geometric_two", "p": [1, 0, 0, 0], "q": [1, 0, 0, 0], "z": "1/2", "prefactor": "1",
     "target": None, "value": "2 = sum_{k>=0} 2^-k",
     "source_note": "the geometric series; in the family but not on a declared target"},
)

BUILTIN_P: tuple[dict[str, Any], ...] = (
    {"id": "wallis_product", "a": [0, 0, 4, 0], "b": [-1, 0, 4, 0], "k0": 1, "prefactor": "2",
     "target": "pi", "value": "pi/2 = prod_{k>=1} 4k^2/(4k^2 - 1)",
     "source_note": "Wallis, Arithmetica infinitorum (1656)"},
    {"id": "wallis_product_half", "a": [0, 0, 4, 0], "b": [-1, 0, 4, 0], "k0": 1,
     "prefactor": "1", "target": None, "value": "pi/2 = prod_{k>=1} 4k^2/(4k^2 - 1)",
     "source_note": "Wallis 1656, unscaled; pi/2 is not itself a declared target"},
    {"id": "wallis_product_cubic", "a": [0, 0, 0, 4], "b": [0, -1, 0, 4], "k0": 1,
     "prefactor": "2", "target": "pi",
     "value": "pi = 2 prod_{k>=1} 4k^3/(4k^3 - k)",
     "source_note": (
         "Wallis 1656 in the degree-3 grammar form; cancelling the common factor k "
         "returns 4k^2/(4k^2 - 1), so this is an independent shape reaching the same "
         "classical value"
     )},
    {"id": "euler_sine_product_half", "a": [-1, 0, 4, 0], "b": [0, 0, 4, 0], "k0": 1,
     "prefactor": "1", "target": None,
     "value": "2/pi = prod_{k>=1} (1 - 1/(4k^2)) = sin(pi/2)/(pi/2)",
     "source_note": "Euler 1734, the sine product sin(pi x)/(pi x) = prod (1 - x^2/k^2) at x = 1/2"},
    {"id": "euler_sinh_product", "a": [1, 0, 1, 0], "b": [0, 0, 1, 0], "k0": 1, "prefactor": "1",
     "target": None, "value": "sinh(pi)/pi = prod_{k>=1} (1 + 1/k^2)",
     "source_note": "Euler's product for sinh, sinh(pi x)/(pi x) = prod (1 + x^2/k^2) at x = 1"},
    {"id": "telescoping_half", "a": [-1, 0, 1, 0], "b": [0, 0, 1, 0], "k0": 2, "prefactor": "1",
     "target": None, "value": "1/2 = prod_{k>=2} (1 - 1/k^2)",
     "source_note": "elementary telescoping product"},
)

BUILTIN_I: tuple[dict[str, Any], ...] = (
    {"id": "arctan_quarter_pi", "a": "0", "b": "0", "kernel": "inv_1px2", "power": "1",
     "prefactor": "4", "target": "pi", "value": "pi = 4 int_0^1 dx/(1+x^2)",
     "source_note": "the arctangent integral; Gregory-Leibniz in integral form"},
    {"id": "beta_half_half", "a": "-1/2", "b": "-1/2", "kernel": "one", "power": "0",
     "prefactor": "1", "target": "pi",
     "value": "pi = B(1/2, 1/2) = int_0^1 x^-1/2 (1-x)^-1/2 dx",
     "source_note": "Euler's Beta integral, B(p, q) = Gamma(p)Gamma(q)/Gamma(p+q)"},
    {"id": "log_integral_ln2", "a": "0", "b": "0", "kernel": "inv_1px", "power": "1",
     "prefactor": "1", "target": "ln2", "value": "ln 2 = int_0^1 dx/(1+x)",
     "source_note": "elementary logarithmic integral"},
    {"id": "dilogarithm_zeta2", "a": "0", "b": "-1", "kernel": "log_inv", "power": "1",
     "prefactor": "1", "target": "zeta2", "value": "zeta(2) = int_0^1 -ln(x)/(1-x) dx",
     "source_note": "the classical Euler integral for zeta(2); expand 1/(1-x) and integrate"},
    {"id": "zeta3_double_log", "a": "0", "b": "-1", "kernel": "log_inv", "power": "2",
     "prefactor": "1/2", "target": "zeta3",
     "value": "zeta(3) = (1/2) int_0^1 (ln(1/x))^2/(1-x) dx",
     "source_note": "the standard log-power integral representation of zeta(s)"},
    {"id": "catalan_log_integral", "a": "0", "b": "0", "kernel": "log_inv_over_1px2",
     "power": "1", "prefactor": "1", "target": "catalan",
     "value": "G = int_0^1 -ln(x)/(1+x^2) dx",
     "source_note": "the classical integral representation of Catalan's constant"},
    {"id": "eta_two_log_integral", "a": "0", "b": "0", "kernel": "log_inv_over_1px",
     "power": "1", "prefactor": "12", "target": "pi_squared",
     "value": "pi^2 = 12 int_0^1 -ln(x)/(1+x) dx",
     "source_note": "int_0^1 -ln(x)/(1+x) dx = pi^2/12 = eta(2)"},
    {"id": "beta_one_one", "a": "0", "b": "0", "kernel": "one", "power": "0",
     "prefactor": "1", "target": None, "value": "1 = B(1, 1) = int_0^1 dx",
     "source_note": "the trivial Beta instance; in the family but not on a declared target"},
)

BUILTIN_TABLE: dict[str, tuple[dict[str, Any], ...]] = {
    "S": BUILTIN_S,
    "P": BUILTIN_P,
    "I": BUILTIN_I,
}


def classify_prior_art(family: str, member: Mapping[str, Any], target_name: str) -> dict[str, Any]:
    """``KNOWN_REDISCOVERED`` versus ``NOT_IN_BUILTIN_TABLE``; never a novelty claim."""

    for entry in BUILTIN_TABLE[family]:
        if entry["target"] != target_name:
            continue
        if family == "S":
            same = (
                list(entry["p"]) == list(member["p"])
                and list(entry["q"]) == list(member["q"])
                and entry["z"] == str(member["z"])
                and entry["prefactor"] == str(member["prefactor"])
            )
        elif family == "P":
            same = (
                list(entry["a"]) == list(member["a"])
                and list(entry["b"]) == list(member["b"])
                and int(entry["k0"]) == int(member["k0"])
                and entry["prefactor"] == str(member["prefactor"])
            )
        else:
            same = (
                entry["a"] == str(member["a"])
                and entry["b"] == str(member["b"])
                and entry["kernel"] == str(member["kernel"])
                and entry["power"] == str(member["power"])
                and entry["prefactor"] == str(member["prefactor"])
            )
        if same:
            return {
                "label": "KNOWN_REDISCOVERED",
                "basis": "exact_builtin_shape",
                "builtin_id": entry["id"],
            }
    return {
        "label": "NOT_IN_BUILTIN_TABLE",
        "basis": "no_structural_match_in_builtin_table",
        "builtin_id": None,
    }


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def _reduced_rational(
    numerator: tuple[int, ...], denominator: tuple[int, ...], scale: str
) -> tuple[str, str]:
    """``scale * N(k)/D(k)`` in lowest terms, with the denominator made monic-positive."""

    factor = sp.Rational(scale)
    top = sum(sp.Integer(int(c)) * _K**i for i, c in enumerate(numerator))
    bottom = sum(sp.Integer(int(c)) * _K**i for i, c in enumerate(denominator))
    if bottom == 0:
        return str(sp.nan), "0"
    reduced = sp.cancel(sp.together(factor * top / bottom))
    top, bottom = sp.fraction(reduced)
    top, bottom = sp.Poly(top, _K), sp.Poly(bottom, _K)
    if bottom.LC() < 0:
        top, bottom = -top, -bottom
    content = sp.gcd(sp.gcd_list(list(top.all_coeffs()) or [sp.Integer(0)]),
                     sp.gcd_list(list(bottom.all_coeffs()) or [sp.Integer(1)]))
    if content and content != 0 and content != 1:
        top = sp.Poly(sp.expand(top.as_expr() / content), _K)
        bottom = sp.Poly(sp.expand(bottom.as_expr() / content), _K)
    return sp.srepr(sp.expand(top.as_expr())), sp.srepr(sp.expand(bottom.as_expr()))


def canonical_form(family: str, member: Mapping[str, Any]) -> dict[str, Any]:
    """A canonical fingerprint that collapses the grammar's declared degeneracies.

    The enumeration is over ordinals, and distinct ordinals may denote the same object: the
    term ratio ``z P/Q`` is unchanged by a common polynomial factor, and a Family I integral
    with ``c = 0`` or the trivial kernel is the pure Beta integral whichever kernel slot it
    used.  Collapsing them is what makes the survivor list countable as *objects*.
    """

    if family == "S":
        top, bottom = _reduced_rational(
            tuple(int(v) for v in member["p"]), tuple(int(v) for v in member["q"]), member["z"]
        )
        return {
            "kind": "series_term_ratio",
            "key": f"S|{top}|{bottom}|{member['prefactor']}",
            "reduced_numerator": top,
            "reduced_denominator": bottom,
            "prefactor": str(member["prefactor"]),
        }
    if family == "P":
        top, bottom = _reduced_rational(
            tuple(int(v) for v in member["a"]), tuple(int(v) for v in member["b"]), "1"
        )
        return {
            "kind": "product_factor",
            "key": f"P|{top}|{bottom}|{member['k0']}|{member['prefactor']}",
            "reduced_numerator": top,
            "reduced_denominator": bottom,
            "k0": int(member["k0"]),
            "prefactor": str(member["prefactor"]),
        }
    power = Fraction(member["power"])
    entry = INTEGRAL_KERNELS[int(member["kernel_index"])]
    a, b = Fraction(member["a"]), Fraction(member["b"])
    if power == 0 or entry["pure_power"]:
        # A pure-power kernel is its own singular part, so raising it to c only shifts the
        # two exponents; c = 0 removes the kernel outright.  Either way this is the Beta
        # integral written in a different ordinal slot.
        a += power * entry["order0"]
        b += power * entry["order1"]
        kernel, power = "one", Fraction(0)
    else:
        # A kernel that is a declared fixed power of another folds onto that base.
        kernel = str(entry["base"])
        power = power * Fraction(entry["base_multiplier"])
    return {
        "kind": "integral_shape",
        "key": f"I|{a}|{b}|{kernel}|{power}|{member['prefactor']}",
        "effective_a": str(a),
        "effective_b": str(b),
        "effective_kernel": kernel,
        "effective_power": str(power),
        "prefactor": str(member["prefactor"]),
        "degenerate_kernel_slot": bool(
            kernel != str(member["kernel"]) or power != Fraction(member["power"])
        ),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _poly_text(coefficients: Sequence[int], variable: str, latex: bool) -> str:
    terms: list[str] = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = int(coefficients[power])
        if coefficient == 0:
            continue
        if power == 0:
            symbol = ""
        elif power == 1:
            symbol = variable
        else:
            symbol = f"{variable}^{{{power}}}" if latex else f"{variable}^{power}"
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


def _prefactor_text(prefactor: Fraction, latex: bool) -> str:
    if prefactor == 1:
        return ""
    if latex:
        if prefactor.denominator == 1:
            return f"{prefactor.numerator} "
        return rf"\tfrac{{{prefactor.numerator}}}{{{prefactor.denominator}}} "
    return f"{prefactor} * "


def render_member(family: str, member: Mapping[str, Any], target_name: str) -> dict[str, str]:
    """Human text and MathML-ready LaTeX for one conjecture."""

    prefactor = Fraction(member["prefactor"])
    text_target = TARGET_TEXT[target_name]
    latex_target = TARGET_LATEX[target_name]
    scale_text = _prefactor_text(prefactor, latex=False)
    scale_latex = _prefactor_text(prefactor, latex=True)
    if family == "S":
        p_text = _poly_text(member["p"], "k", latex=False)
        q_text = _poly_text(member["q"], "k", latex=False)
        p_latex = _poly_text(member["p"], "k", latex=True)
        q_latex = _poly_text(member["q"], "k", latex=True)
        z = Fraction(member["z"])
        text = (
            f"{text_target} =? {scale_text}sum_{{k>=0}} t_k where t_0 = 1 and "
            f"t_{{k+1}}/t_k = ({z}) * ({p_text})/({q_text})"
        )
        latex = (
            rf"{latex_target} \stackrel{{?}}{{=}} {scale_latex}\sum_{{k\ge 0}} t_k,\quad "
            rf"t_0 = 1,\quad \frac{{t_{{k+1}}}}{{t_k}} = "
            rf"\left({z}\right)\frac{{{p_latex}}}{{{q_latex}}}"
        )
    elif family == "P":
        a_text = _poly_text(member["a"], "k", latex=False)
        b_text = _poly_text(member["b"], "k", latex=False)
        a_latex = _poly_text(member["a"], "k", latex=True)
        b_latex = _poly_text(member["b"], "k", latex=True)
        k0 = int(member["k0"])
        text = f"{text_target} =? {scale_text}prod_{{k>={k0}}} ({a_text})/({b_text})"
        latex = (
            rf"{latex_target} \stackrel{{?}}{{=}} {scale_latex}"
            rf"\prod_{{k\ge {k0}}} \frac{{{a_latex}}}{{{b_latex}}}"
        )
    else:
        kernel = INTEGRAL_KERNELS[int(member["kernel_index"])]
        a, b, c = member["a"], member["b"], member["power"]
        power_text = "" if Fraction(c) == 0 else f" * ({kernel['expr']})^({c})"
        power_latex = (
            "" if Fraction(c) == 0 else rf"\left({kernel['latex']}\right)^{{{c}}}"
        )
        text = (
            f"{text_target} =? {scale_text}integral_0^1 x^({a}) (1-x)^({b}){power_text} dx"
        )
        latex = (
            rf"{latex_target} \stackrel{{?}}{{=}} {scale_latex}"
            rf"\int_0^1 x^{{{a}}}(1-x)^{{{b}}}{power_latex}\,dx"
        )
    return {"text": text, "latex": latex}


# ---------------------------------------------------------------------------
# fp64 sweep -- families S and P
# ---------------------------------------------------------------------------

_ITERATED_KERNEL_SOURCE = r"""
extern "C" __global__ void sweep_iterated(
    const long long start, const long long count, const int mode, const int nsub,
    const double* __restrict__ subvals, const double* __restrict__ compare, const int ncompare,
    const int nterms, const int nck, const int* __restrict__ checkpoints,
    const double* __restrict__ w8, const double* __restrict__ w6,
    const double window, const double gate,
    const double decay_tail_max, const double decay_ratio_max,
    unsigned int* __restrict__ hit_count, long long* __restrict__ hit_index,
    int* __restrict__ hit_compare, double* __restrict__ hit_value, double* __restrict__ hit_error,
    const unsigned int capacity, unsigned long long* __restrict__ stats)
{
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= count) return;
    long long index = start + tid;
    long long shape = index / nsub;
    int sub = (int)(index % nsub);

    double c[8];
    long long v = shape;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { c[j] = (double)(v % 9 - 4); v /= 9; }

    double acc[16];
    int slot = 0;
    double state, term = 0.0, early = 0.0, tail = 0.0;
    bool broken = false, terminated = false;

    if (mode == 0) {
        // Family S: partial sums of a series with ratio z P(k)/Q(k).
        double z = subvals[sub];
        state = 1.0; term = 1.0;
        for (int k = 0; k < nterms; ++k) {
            double kk = (double)k;
            double pv = c[0] + kk*(c[1] + kk*(c[2] + kk*c[3]));
            double qv = c[4] + kk*(c[5] + kk*(c[6] + kk*c[7]));
            if (pv == 0.0) terminated = true;   // structural: P has a non-negative integer root
            term = term * z * pv / qv;
            state += term;
            if (!isfinite(state) || !isfinite(term)) { broken = true; break; }
            if (slot < nck && (k + 1) == checkpoints[slot]) { acc[slot] = state; ++slot; }
            if (k + 1 == (nterms >> 4)) early = fabs(term);
            if (k + 1 == nterms) tail = fabs(term);
        }
    } else {
        // Family P: the exact convergence and positivity test first, then the product.
        double k0 = subvals[sub];
        int da = 3; while (da >= 0 && c[da] == 0.0) --da;
        int db = 3; while (db >= 0 && c[4 + db] == 0.0) --db;
        bool ok = (da >= 1) && (da == db)
                  && (c[da] == c[4 + db]) && (c[da - 1] == c[4 + db - 1]);
        if (ok) {
            for (int j = (int)k0; j <= 8; ++j) {
                double t = (double)j;
                double av = c[0] + t*(c[1] + t*(c[2] + t*c[3]));
                double bv = c[4] + t*(c[5] + t*(c[6] + t*c[7]));
                if (!(av > 0.0) || !(bv > 0.0)) { ok = false; break; }
            }
        }
        if (!ok) { atomicAdd(&stats[5], 1ULL); return; }
        state = 1.0;
        for (int k = 0; k < nterms; ++k) {
            double kk = k0 + (double)k;
            double av = c[0] + kk*(c[1] + kk*(c[2] + kk*c[3]));
            double bv = c[4] + kk*(c[5] + kk*(c[6] + kk*c[7]));
            state = state * av / bv;
            if (!isfinite(state) || state == 0.0) { broken = true; break; }
            if (slot < nck && (k + 1) == checkpoints[slot]) { acc[slot] = state; ++slot; }
        }
        early = 1.0; tail = 0.0;
    }

    if (broken || slot < nck) { atomicAdd(&stats[0], 1ULL); return; }
    if (mode == 0) {
        if (terminated) { atomicAdd(&stats[3], 1ULL); return; }
        bool decayed = (tail < decay_tail_max)
                       && ((tail == 0.0) || (tail < decay_ratio_max * early));
        if (!decayed) { atomicAdd(&stats[4], 1ULL); return; }
    }

    double e8 = 0.0, e6 = 0.0;
    for (int j = 0; j < nck; ++j) e8 += w8[j] * acc[j];
    for (int j = 0; j < 6; ++j) e6 += w6[j] * acc[nck - 6 + j];
    if (!isfinite(e8) || !isfinite(e6) || !(fabs(e8 - e6) < gate)) {
        atomicAdd(&stats[1], 1ULL); return;
    }
    atomicAdd(&stats[2], 1ULL);
    for (int m = 0; m < ncompare; ++m) {
        double err = fabs(e8 - compare[m]);
        if (err < window) {
            unsigned int at = atomicAdd(hit_count, 1u);
            if (at < capacity) {
                hit_index[at] = index;
                hit_compare[at] = m;
                hit_value[at] = e8;
                hit_error[at] = err;
            }
        }
    }
}
"""

_ITERATED_MODULE: Any = None


def _iterated_kernel() -> Any:
    global _ITERATED_MODULE
    if _ITERATED_MODULE is None:
        import cupy as cp

        _ITERATED_MODULE = cp.RawModule(code=_ITERATED_KERNEL_SOURCE, options=("-std=c++14",))
    return _ITERATED_MODULE.get_function("sweep_iterated")


def evaluate_iterated_numpy(
    family: str, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU mirror of the fused kernel.  Returns ``(value, admitted, reject_code)``.

    Reject codes: 0 admitted, 1 non-finite/degenerate, 2 resolution gate, 3 terminating,
    4 decay gate, 5 failed the Family P convergence/positivity test.  Used for the GPU/CPU
    crosscheck and for small ``--cpu`` runs.
    """

    nsub = len(S_Z_VALUES) if family == "S" else len(P_K0_VALUES)
    subvals = (
        np.array([float(z) for z in S_Z_VALUES])
        if family == "S"
        else np.array([float(k) for k in P_K0_VALUES])
    )
    indices = np.asarray(indices, dtype=np.int64)
    shapes = indices // nsub
    subs = indices % nsub
    coefficients = []
    value = shapes.copy()
    for _ in range(POLY_SLOTS):
        coefficients.append((value % POLY_RADIX - 4).astype(np.float64))
        value = value // POLY_RADIX
    parameter = subvals[subs]
    checkpoints = np.zeros((indices.shape[0], len(FP64_CHECKPOINTS)), dtype=np.float64)
    slot = 0
    broken = np.zeros(indices.shape[0], dtype=bool)
    terminated = np.zeros(indices.shape[0], dtype=bool)
    early = np.zeros(indices.shape[0], dtype=np.float64)
    tail = np.zeros(indices.shape[0], dtype=np.float64)
    with np.errstate(all="ignore"):
        if family == "S":
            state = np.ones(indices.shape[0], dtype=np.float64)
            term = np.ones(indices.shape[0], dtype=np.float64)
            for k in range(FP64_TERMS):
                kk = float(k)
                pv = coefficients[0] + kk * (
                    coefficients[1] + kk * (coefficients[2] + kk * coefficients[3])
                )
                qv = coefficients[4] + kk * (
                    coefficients[5] + kk * (coefficients[6] + kk * coefficients[7])
                )
                terminated |= (pv == 0.0) & ~broken
                term = term * parameter * pv / qv
                state = state + term
                fresh = ~np.isfinite(state) | ~np.isfinite(term)
                broken |= fresh & ~broken
                if slot < len(FP64_CHECKPOINTS) and (k + 1) == FP64_CHECKPOINTS[slot]:
                    checkpoints[:, slot] = state
                    slot += 1
                if k + 1 == FP64_TERMS >> 4:
                    early = np.abs(term)
                if k + 1 == FP64_TERMS:
                    tail = np.abs(term)
        else:
            degree_a = np.full(indices.shape[0], -1, dtype=np.int64)
            degree_b = np.full(indices.shape[0], -1, dtype=np.int64)
            for power in range(4):
                degree_a = np.where(coefficients[power] != 0, power, degree_a)
                degree_b = np.where(coefficients[4 + power] != 0, power, degree_b)
            structural = (degree_a >= 1) & (degree_a == degree_b)
            safe = np.where(structural, degree_a, 1)
            stacked_a = np.stack(coefficients[:4])
            stacked_b = np.stack(coefficients[4:])
            columns = np.arange(indices.shape[0])
            structural &= stacked_a[safe, columns] == stacked_b[safe, columns]
            structural &= stacked_a[safe - 1, columns] == stacked_b[safe - 1, columns]
            for point in range(1, 9):
                active = parameter <= point
                av = coefficients[0] + point * (
                    coefficients[1] + point * (coefficients[2] + point * coefficients[3])
                )
                bv = coefficients[4] + point * (
                    coefficients[5] + point * (coefficients[6] + point * coefficients[7])
                )
                structural &= ~active | ((av > 0) & (bv > 0))
            state = np.ones(indices.shape[0], dtype=np.float64)
            for k in range(FP64_TERMS):
                kk = parameter + float(k)
                av = coefficients[0] + kk * (
                    coefficients[1] + kk * (coefficients[2] + kk * coefficients[3])
                )
                bv = coefficients[4] + kk * (
                    coefficients[5] + kk * (coefficients[6] + kk * coefficients[7])
                )
                state = state * av / bv
                broken |= (~np.isfinite(state) | (state == 0.0)) & ~broken
                if slot < len(FP64_CHECKPOINTS) and (k + 1) == FP64_CHECKPOINTS[slot]:
                    checkpoints[:, slot] = state
                    slot += 1
            early = np.ones(indices.shape[0], dtype=np.float64)
            tail = np.zeros(indices.shape[0], dtype=np.float64)
        e8 = checkpoints @ NEVILLE_W8
        e6 = checkpoints[:, -6:] @ NEVILLE_W6
        code = np.zeros(indices.shape[0], dtype=np.int8)
        if family == "P":
            code[~structural] = 5
        code[(code == 0) & broken] = 1
        if family == "S":
            code[(code == 0) & terminated] = 3
            decay_ok = (tail < FP64_DECAY_TAIL_MAX) & (
                (tail == 0.0) | (tail < FP64_DECAY_RATIO_MAX * early)
            )
            code[(code == 0) & ~decay_ok] = 4
        unresolved = (
            ~np.isfinite(e8) | ~np.isfinite(e6) | ~(np.abs(e8 - e6) < FP64_RESOLUTION_GATE)
        )
        code[(code == 0) & unresolved] = 2
    return e8, code == 0, code


def sweep_iterated_gpu(
    family: str,
    *,
    index_start: int,
    index_stop: int,
    block: int = 1 << 23,
    capacity: int = 1 << 21,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """Fused fp64 GPU sweep of Family S or P over ``[index_start, index_stop)``."""

    import cupy as cp

    kernel = _iterated_kernel()
    mode = 0 if family == "S" else 1
    subvals = (
        np.array([float(z) for z in S_Z_VALUES])
        if family == "S"
        else np.array([float(k) for k in P_K0_VALUES])
    )
    compare, labels = comparison_values()
    device_sub = cp.asarray(subvals)
    device_compare = cp.asarray(compare)
    device_checkpoints = cp.asarray(np.array(FP64_CHECKPOINTS, dtype=np.int32))
    device_w8 = cp.asarray(NEVILLE_W8)
    device_w6 = cp.asarray(NEVILLE_W6)
    stats = cp.zeros(8, dtype=cp.uint64)
    matches: list[dict[str, Any]] = []
    started = time.perf_counter()
    for chunk_start in range(index_start, index_stop, block):
        count = min(block, index_stop - chunk_start)
        hit_count = cp.zeros(1, dtype=cp.uint32)
        hit_index = cp.zeros(capacity, dtype=cp.int64)
        hit_compare = cp.zeros(capacity, dtype=cp.int32)
        hit_value = cp.zeros(capacity, dtype=cp.float64)
        hit_error = cp.zeros(capacity, dtype=cp.float64)
        threads = 256
        kernel(
            ((count + threads - 1) // threads,),
            (threads,),
            (
                np.int64(chunk_start),
                np.int64(count),
                np.int32(mode),
                np.int32(len(subvals)),
                device_sub,
                device_compare,
                np.int32(compare.shape[0]),
                np.int32(FP64_TERMS),
                np.int32(len(FP64_CHECKPOINTS)),
                device_checkpoints,
                device_w8,
                device_w6,
                np.float64(FP64_MATCH_WINDOW),
                np.float64(FP64_RESOLUTION_GATE),
                np.float64(FP64_DECAY_TAIL_MAX),
                np.float64(FP64_DECAY_RATIO_MAX),
                hit_count,
                hit_index,
                hit_compare,
                hit_value,
                hit_error,
                np.uint32(capacity),
                stats,
            ),
        )
        found = int(hit_count.get()[0])
        if found > capacity:
            raise FamilyError(
                f"fp64 hit buffer overflowed ({found} > {capacity}); rerun with a larger "
                "capacity so no match is silently dropped"
            )
        if found:
            host_index = hit_index[:found].get()
            host_compare = hit_compare[:found].get()
            host_value = hit_value[:found].get()
            host_error = hit_error[:found].get()
            for i in range(found):
                target_name, prefactor_index = labels[int(host_compare[i])]
                matches.append(
                    {
                        "member_index": int(host_index[i]),
                        "target": target_name,
                        "prefactor_index": int(prefactor_index),
                        "fp64_value": float(host_value[i]),
                        "fp64_abs_error": float(host_error[i]),
                    }
                )
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    host_stats = stats.get()
    counters = {
        "degenerate_or_nonfinite": int(host_stats[0]),
        "failed_resolution_gate": int(host_stats[1]),
        "resolved": int(host_stats[2]),
        "terminating_rational": int(host_stats[3]),
        "failed_decay_gate": int(host_stats[4]),
        "failed_convergence_test": int(host_stats[5]),
    }
    matches.sort(key=lambda item: (item["member_index"], item["target"], item["prefactor_index"]))
    return matches, counters, elapsed


def sweep_iterated_cpu(
    family: str, *, index_start: int, index_stop: int, block: int = 1 << 14
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """numpy mirror of :func:`sweep_iterated_gpu` (small ranges only)."""

    compare, labels = comparison_values()
    matches: list[dict[str, Any]] = []
    counters = {
        "degenerate_or_nonfinite": 0,
        "failed_resolution_gate": 0,
        "resolved": 0,
        "terminating_rational": 0,
        "failed_decay_gate": 0,
        "failed_convergence_test": 0,
    }
    started = time.perf_counter()
    for chunk_start in range(index_start, index_stop, block):
        indices = np.arange(chunk_start, min(chunk_start + block, index_stop), dtype=np.int64)
        values, admitted, code = evaluate_iterated_numpy(family, indices)
        counters["degenerate_or_nonfinite"] += int((code == 1).sum())
        counters["failed_resolution_gate"] += int((code == 2).sum())
        counters["terminating_rational"] += int((code == 3).sum())
        counters["failed_decay_gate"] += int((code == 4).sum())
        counters["failed_convergence_test"] += int((code == 5).sum())
        counters["resolved"] += int(admitted.sum())
        if not admitted.any():
            continue
        selected = np.nonzero(admitted)[0]
        errors = np.abs(values[selected][:, None] - compare[None, :])
        rows, columns = np.nonzero(errors < FP64_MATCH_WINDOW)
        for row, column in zip(rows, columns, strict=True):
            target_name, prefactor_index = labels[int(column)]
            matches.append(
                {
                    "member_index": int(indices[selected[int(row)]]),
                    "target": target_name,
                    "prefactor_index": int(prefactor_index),
                    "fp64_value": float(values[selected[int(row)]]),
                    "fp64_abs_error": float(errors[int(row), int(column)]),
                }
            )
    elapsed = time.perf_counter() - started
    matches.sort(key=lambda item: (item["member_index"], item["target"], item["prefactor_index"]))
    return matches, counters, elapsed


# ---------------------------------------------------------------------------
# fp64 sweep -- family I
# ---------------------------------------------------------------------------


def _integral_exponent_tables(xp: Any) -> tuple[Any, Any, dict[str, Any], Any]:
    """Extended-grid exponent tables ``exp((e+1) ln x)`` and ``exp((e+1) ln(1-x))``.

    The kernel's singular part is folded into the exponent, so the effective exponents leave
    the base grid; the extended grid covers every effective exponent the declared kernel
    powers can produce, and the two tables are contiguous slices of it per ``(kernel, power)``.
    """

    lowest = min(I_EXPONENT_NUMERATORS)
    highest = max(I_EXPONENT_NUMERATORS)
    shift_low = min(
        int(power * kernel[key] * I_EXPONENT_DENOMINATOR)
        for kernel in INTEGRAL_KERNELS
        for power in I_POWERS
        for key in ("order0", "order1")
    )
    shift_high = max(
        int(power * kernel[key] * I_EXPONENT_DENOMINATOR)
        for kernel in INTEGRAL_KERNELS
        for power in I_POWERS
        for key in ("order0", "order1")
    )
    numerators = np.arange(lowest + shift_low, highest + shift_high + 1)
    exponents = numerators / I_EXPONENT_DENOMINATOR + 1.0
    log_x = xp.asarray(QUADRATURE.log_x)
    log_1mx = xp.asarray(QUADRATURE.log_1mx)
    grid = xp.asarray(exponents)[:, None]
    table_a = xp.exp(xp.minimum(grid * log_x[None, :], 700.0))
    table_b = xp.exp(xp.minimum(grid * log_1mx[None, :], 700.0))
    kernel_tables = {
        item["name"]: xp.asarray(QUADRATURE.kernel_log[item["name"]])
        for item in INTEGRAL_KERNELS
    }
    return table_a, table_b, kernel_tables, int(numerators[0])


def sweep_integrals(
    xp: Any, *, kernel_subset: Sequence[int] | None = None
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """fp64 sweep of Family I as one GEMM per ``(kernel, power)`` pair."""

    compare, labels = comparison_values()
    order = np.argsort(compare, kind="stable")
    sorted_compare = compare[order]
    device_compare = xp.asarray(sorted_compare)
    table_a, table_b, kernel_tables, base = _integral_exponent_tables(xp)
    prefactor = xp.asarray(QUADRATURE.prefactor)
    matches: list[dict[str, Any]] = []
    counters = {"evaluated": 0, "diverges": 0, "nonfinite": 0}
    kernels = (
        INTEGRAL_KERNELS
        if kernel_subset is None
        else tuple(INTEGRAL_KERNELS[i] for i in kernel_subset)
    )
    started = time.perf_counter()
    exponent_numerators = np.array(I_EXPONENT_NUMERATORS)
    for kernel in kernels:
        for power_index, power in enumerate(I_POWERS):
            shift0 = int(power * kernel["order0"] * I_EXPONENT_DENOMINATOR)
            shift1 = int(power * kernel["order1"] * I_EXPONENT_DENOMINATOR)
            row0 = exponent_numerators[0] + shift0 - base
            row1 = exponent_numerators[0] + shift1 - base
            left = table_a[row0 : row0 + I_GRID]
            right = table_b[row1 : row1 + I_GRID]
            weights = prefactor
            if power != 0:
                weights = weights * xp.exp(
                    xp.minimum(float(power) * kernel_tables[kernel["name"]], 700.0)
                )
            values = left @ (right * weights[None, :]).T  # (a_index, b_index)
            valid_a = xp.asarray(
                (exponent_numerators / I_EXPONENT_DENOMINATOR + float(power * kernel["order0"]))
                > -1.0
            )
            valid_b = xp.asarray(
                (exponent_numerators / I_EXPONENT_DENOMINATOR + float(power * kernel["order1"]))
                > -1.0
            )
            valid = valid_a[:, None] & valid_b[None, :] & xp.isfinite(values)
            counters["evaluated"] += int(valid.sum())
            counters["diverges"] += int((~(valid_a[:, None] & valid_b[None, :])).sum())
            counters["nonfinite"] += int(
                ((valid_a[:, None] & valid_b[None, :]) & ~xp.isfinite(values)).sum()
            )
            # Nearest-neighbour search against the sorted comparison grid: the two
            # neighbours of each value are the only entries that can be inside the window,
            # so no (a, b, comparison) tensor is ever materialized.
            flat = xp.where(valid, values, xp.asarray(np.float64("inf"))).ravel()
            slot = xp.searchsorted(device_compare, flat)
            for offset in (-1, 0):
                probe = xp.clip(slot + offset, 0, device_compare.shape[0] - 1)
                error = xp.abs(flat - device_compare[probe])
                found = xp.nonzero(error < FP64_MATCH_WINDOW)[0]
                if found.shape[0] == 0:
                    continue
                found_slots = probe[found]
                found_errors = error[found]
                found_values = values.ravel()[found]
                if xp is not np:
                    found = found.get()
                    found_slots = found_slots.get()
                    found_errors = found_errors.get()
                    found_values = found_values.get()
                for i in range(found.shape[0]):
                    row, column = divmod(int(found[i]), I_GRID)
                    target_name, prefactor_index = labels[int(order[int(found_slots[i])])]
                    index = (
                        (row * I_GRID + column) * I_KERNEL_COUNT + int(kernel["index"])
                    ) * len(I_POWERS) + power_index
                    matches.append(
                        {
                            "member_index": index,
                            "target": target_name,
                            "prefactor_index": int(prefactor_index),
                            "fp64_value": float(found_values[i]),
                            "fp64_abs_error": float(found_errors[i]),
                        }
                    )
    elapsed = time.perf_counter() - started
    unique: dict[tuple[int, str, int], dict[str, Any]] = {}
    for item in matches:
        unique[(item["member_index"], item["target"], item["prefactor_index"])] = item
    matches = sorted(
        unique.values(),
        key=lambda item: (item["member_index"], item["target"], item["prefactor_index"]),
    )
    return matches, counters, elapsed


# ---------------------------------------------------------------------------
# Member records
# ---------------------------------------------------------------------------


def member_from_index(family: str, member_index: int, prefactor_index: int) -> dict[str, Any]:
    """Rebuild the declared member record from a sweep index plus its prefactor slot."""

    ordinal = member_index * len(PREFACTORS) + prefactor_index
    if family == "S":
        return decode_s_ordinal(ordinal)
    if family == "P":
        return decode_p_ordinal(ordinal)
    return decode_i_ordinal(ordinal)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

#: The fabricated near-miss.  ``pi`` perturbed in its 14th decimal is *not* in any family, so
#: the control is run directly through the holdout machinery: it must clear the fp64 window
#: and then die at 60 digits.  A discipline that lets this through is not a discipline.
FABRICATED_OFFSET = mp.mpf("1e-14")


def fabricated_near_miss_control() -> dict[str, Any]:
    """A constructed value ~1e-14 from ``pi`` must pass fp64 and die at 60 digits."""

    with mp.workdps(200):
        planted = +mp.pi + FABRICATED_OFFSET
        fp64_error = float(abs(planted - mp.pi))
    passes_fp64 = bool(fp64_error < FP64_MATCH_WINDOW)
    trail: list[dict[str, Any]] = [
        {
            "stage": "fp64_window",
            "abs_error": format(fp64_error, ".3e"),
            "threshold": f"{FP64_MATCH_WINDOW:g}",
            "passed": passes_fp64,
        }
    ]
    died_at: str | None = None
    for stage in VERIFY_STAGES:
        with mp.workdps(stage["dps"]):
            error = abs((+mp.pi + FABRICATED_OFFSET) - constant_value("pi"))
            passed = bool(error < mp.mpf(stage["threshold"]))
            trail.append(
                {
                    "stage": stage["stage"],
                    "abs_error": _error_string(error, stage["dps"]),
                    "threshold": stage["threshold"],
                    "passed": passed,
                }
            )
        if not passed and died_at is None:
            died_at = stage["stage"]
    return {
        "construction": "pi + 1e-14, a value sitting ~1e-14 from the target pi",
        "passed_fp64_window": passes_fp64,
        "died_at_stage": died_at,
        "digit_trail": trail,
        "control_passed": bool(passes_fp64 and died_at == VERIFY_STAGES[0]["stage"]),
    }


#: Per-family classics that the enumeration *must* rediscover.  These are declared by their
#: ordinal, evaluated through the same fp64 path and the same digit holdout as everything
#: else; a family that cannot recover them cannot support a NOT_FOUND verdict.
FAMILY_CLASSICS: dict[str, tuple[dict[str, Any], ...]] = {
    "S": (
        {"name": "basel_series_zeta2", "builtin_id": "basel_series", "target": "zeta2"},
        {"name": "leibniz_gregory_pi", "builtin_id": "leibniz_gregory", "target": "pi"},
        {"name": "exponential_series_e", "builtin_id": "exp_series", "target": "e"},
    ),
    "P": (
        {"name": "wallis_product_pi", "builtin_id": "wallis_product", "target": "pi"},
        {"name": "wallis_cubic_form_pi", "builtin_id": "wallis_product_cubic", "target": "pi"},
        {"name": "euler_sine_product", "builtin_id": "euler_sine_product_half", "target": None},
    ),
    "I": (
        {"name": "arctan_integral_pi", "builtin_id": "arctan_quarter_pi", "target": "pi"},
        {"name": "log_integral_zeta2", "builtin_id": "dilogarithm_zeta2", "target": "zeta2"},
        {"name": "beta_instance_pi", "builtin_id": "beta_half_half", "target": "pi"},
    ),
}

#: Classical members that are inside the grammar but whose value is not on the declared
#: target list, so the sweep cannot flag them.  They still control the *evaluation path*:
#: the exact evaluator must reproduce the cited closed form to 60 digits.
FAMILY_VALUE_IDENTITIES: dict[str, tuple[dict[str, Any], ...]] = {
    "S": (
        {"builtin_id": "geometric_two", "closed_form": "2", "expression": lambda: mp.mpf(2)},
    ),
    "P": (
        {"builtin_id": "euler_sine_product_half", "closed_form": "2/pi",
         "expression": lambda: 2 / mp.pi},
        {"builtin_id": "euler_sinh_product", "closed_form": "sinh(pi)/pi",
         "expression": lambda: mp.sinh(mp.pi) / mp.pi},
        {"builtin_id": "telescoping_half", "closed_form": "1/2",
         "expression": lambda: mp.mpf(1) / 2},
        {"builtin_id": "wallis_product_half", "closed_form": "pi/2",
         "expression": lambda: mp.pi / 2},
    ),
    "I": (
        {"builtin_id": "beta_one_one", "closed_form": "1", "expression": lambda: mp.mpf(1)},
    ),
}

VALUE_IDENTITY_DPS = 60
VALUE_IDENTITY_THRESHOLD = "1e-50"


def _builtin_entry(family: str, builtin_id: str) -> dict[str, Any]:
    for entry in BUILTIN_TABLE[family]:
        if entry["id"] == builtin_id:
            return entry
    raise FamilyError(f"unknown builtin id: {builtin_id}")


def builtin_ordinal(family: str, entry: Mapping[str, Any]) -> int:
    """Ordinal of a built-in table entry inside its declared family."""

    prefactor = Fraction(entry["prefactor"])
    if family == "S":
        return encode_s_ordinal(entry["p"], entry["q"], Fraction(entry["z"]), prefactor)
    if family == "P":
        return encode_p_ordinal(entry["a"], entry["b"], int(entry["k0"]), prefactor)
    kernel_index = next(
        item["index"] for item in INTEGRAL_KERNELS if item["name"] == entry["kernel"]
    )
    return encode_i_ordinal(
        Fraction(entry["a"]), Fraction(entry["b"]), kernel_index, Fraction(entry["power"]), prefactor
    )


def rediscovery_controls(family: str, matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Did the sweep actually produce every declared classic of this family?"""

    found = {
        (int(item["member_index"]) * len(PREFACTORS) + int(item["prefactor_index"]), item["target"])
        for item in matches
    }
    rows: list[dict[str, Any]] = []
    for classic in FAMILY_CLASSICS[family]:
        entry = _builtin_entry(family, classic["builtin_id"])
        ordinal = builtin_ordinal(family, entry)
        target = classic["target"] or entry["target"]
        rediscovered = target is not None and (ordinal, target) in found
        rows.append(
            {
                "name": classic["name"],
                "builtin_id": entry["id"],
                "identity": entry["value"],
                "citation": entry["source_note"],
                "ordinal": ordinal,
                "target": target,
                "rediscovered_by_the_sweep": bool(rediscovered),
            }
        )
    required = [row for row in rows if row["target"] is not None]
    identities = value_identity_controls(family)
    return {
        "classics_declared": len(rows),
        "classics_requiring_a_target_match": len(required),
        "classics_rediscovered": sum(1 for row in required if row["rediscovered_by_the_sweep"]),
        "classics": rows,
        "value_identities": identities,
        "passed": bool(
            required
            and all(row["rediscovered_by_the_sweep"] for row in required)
            and all(row["reproduced"] for row in identities)
        ),
    }


def value_identity_controls(family: str) -> list[dict[str, Any]]:
    """Classical in-grammar members whose cited closed form the evaluator must reproduce."""

    rows: list[dict[str, Any]] = []
    for identity in FAMILY_VALUE_IDENTITIES.get(family, ()):
        entry = _builtin_entry(family, identity["builtin_id"])
        ordinal = builtin_ordinal(family, entry)
        member = member_from_index(family, ordinal // len(PREFACTORS), ordinal % len(PREFACTORS))
        with mp.workdps(VALUE_IDENTITY_DPS):
            value = family_value_mp(family, member)
            error = abs(value - identity["expression"]())
            reproduced = bool(error < mp.mpf(VALUE_IDENTITY_THRESHOLD))
            error_text = _error_string(error, VALUE_IDENTITY_DPS)
        rows.append(
            {
                "builtin_id": entry["id"],
                "identity": entry["value"],
                "citation": entry["source_note"],
                "ordinal": ordinal,
                "closed_form": identity["closed_form"],
                "abs_error_at_60_digits": error_text,
                "threshold": VALUE_IDENTITY_THRESHOLD,
                "reproduced": reproduced,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _verify_matches(family: str, matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Run the digit holdout on every fp64 match of one family."""

    verified: list[dict[str, Any]] = []
    for match in matches:
        member = member_from_index(family, int(match["member_index"]), int(match["prefactor_index"]))
        prefactor = Fraction(member["prefactor"])
        trail, status = survival_trail(
            family, member, match["target"], prefactor, float(match["fp64_abs_error"])
        )
        record: dict[str, Any] = {
            **member,
            "family": family,
            "target": match["target"],
            "fp64_value": format(float(match["fp64_value"]), ".17g"),
            "digit_trail": trail,
            "status": status,
        }
        if status == "SURVIVED_ALL_STAGES":
            with mp.workdps(VALUE_RENDER_DPS + 20):
                value = family_value_mp(family, member)
                scaled = value * mp.mpf(prefactor.numerator) / prefactor.denominator
                record["member_value_120_digits"] = mp.nstr(value, VALUE_RENDER_DPS)
                record["scaled_value_120_digits"] = mp.nstr(scaled, VALUE_RENDER_DPS)
            rendered = render_member(family, member, match["target"])
            record["formula_text"] = rendered["text"]
            record["formula_latex"] = rendered["latex"]
            record["canonical_form"] = canonical_form(family, member)
            record["prior_art"] = classify_prior_art(family, member, match["target"])
        verified.append(record)
    return verified


def run_family(
    family: str,
    *,
    use_gpu: bool = True,
    member_limit: int | None = None,
    block: int = 1 << 23,
    capacity: int = 1 << 21,
) -> dict[str, Any]:
    """Sweep one declared family, run its controls, and return the family block."""

    if family not in FAMILIES:
        raise FamilyError(f"unknown family: {family}")
    config = FAMILY_CONFIG[family]
    if family == "I":
        if use_gpu:
            import cupy as cp

            xp: Any = cp
        else:
            xp = np
        # The integral sweep is one GEMM per (kernel, power) pair and is always exhaustive;
        # ``member_limit`` applies only to the iterated families.
        matches, counters, elapsed = sweep_integrals(xp)
        evaluated_members = I_INTEGRAL_COUNT
        exhaustive = True
    else:
        total = S_SERIES_COUNT if family == "S" else P_PRODUCT_COUNT
        stop = total if member_limit is None else min(member_limit, total)
        if use_gpu:
            matches, counters, elapsed = sweep_iterated_gpu(
                family, index_start=0, index_stop=stop, block=block, capacity=capacity
            )
        else:
            matches, counters, elapsed = sweep_iterated_cpu(
                family, index_start=0, index_stop=stop
            )
        evaluated_members = stop
        exhaustive = stop == total
    evaluated_ordinals = evaluated_members * len(PREFACTORS)

    controls = rediscovery_controls(family, matches)
    verified = _verify_matches(family, matches)
    survivors = [item for item in verified if item["status"] == "SURVIVED_ALL_STAGES"]
    casualties = [item for item in verified if item["status"] != "SURVIVED_ALL_STAGES"]
    stage2 = VERIFY_STAGES[0]["stage"].upper()
    prior_art_counts = {
        "known_rediscovered": sum(
            1 for item in survivors if item["prior_art"]["label"] == "KNOWN_REDISCOVERED"
        ),
        "not_in_builtin_table": sum(
            1 for item in survivors if item["prior_art"]["label"] == "NOT_IN_BUILTIN_TABLE"
        ),
    }
    per_target = {
        name: {
            "fp64_matches": sum(1 for item in verified if item["target"] == name),
            "survivors": sum(1 for item in survivors if item["target"] == name),
        }
        for name in TARGET_NAMES
    }
    distinct = {(item["canonical_form"]["key"], item["target"]) for item in survivors}
    distinct_unknown = {
        (item["canonical_form"]["key"], item["target"])
        for item in survivors
        if item["prior_art"]["label"] == "NOT_IN_BUILTIN_TABLE"
    }
    return {
        "family": family,
        "title": config["title"],
        "config": config,
        "config_sha256": canonical_sha256(config),
        "scale": {
            "declared_total_ordinals": FAMILY_ORDINALS[family],
            "declared_distinct_members": (
                S_SERIES_COUNT
                if family == "S"
                else P_PRODUCT_COUNT
                if family == "P"
                else I_INTEGRAL_COUNT
            ),
            "evaluated_members": evaluated_members,
            "evaluated_ordinals": evaluated_ordinals,
            "exhaustive_this_run": exhaustive,
        },
        "sweep_counters": counters,
        "controls": {"rediscovery": controls},
        "counts": {
            "fp64_matches": len(verified),
            "survivors": len(survivors),
            "died_in_verification": len(casualties),
            "died_at_stage2": sum(1 for item in casualties if item["status"].endswith(stage2)),
            "unevaluable": sum(1 for item in casualties if item["status"].startswith("UNEVALUABLE")),
            "distinct_survivor_objects": len(distinct),
            "distinct_not_in_builtin_table_objects": len(distinct_unknown),
            "per_target": per_target,
            "prior_art": prior_art_counts,
        },
        "survivors": survivors,
        "died": casualties,
        "builtin_known_table": [dict(entry) for entry in BUILTIN_TABLE[family]],
        "negative_results": {
            "targets_with_zero_matches": [
                name for name in TARGET_NAMES if per_target[name]["fp64_matches"] == 0
            ]
        },
        "measurement": {
            "sweep_seconds": format(elapsed, ".3f"),
            "ordinals_per_second": (
                int(evaluated_ordinals / elapsed) if elapsed > 0 else None
            ),
            "members_per_second": int(evaluated_members / elapsed) if elapsed > 0 else None,
        },
    }


#: Crosscheck window.  A contiguous slice is used rather than a scattered sample so that
#: both paths run their *production* code, not a special per-member entry point, and the
#: window is anchored on the family's first classic so it is guaranteed to contain both
#: admitted members and at least one real match.
CROSSCHECK_MEMBERS = 1 << 16
#: The resolution gate is a knife-edge comparison at 1e-13, so fused-multiply-add
#: differences between the CUDA and numpy Horner evaluations can move a handful of members
#: across it.  The declared bound is on the *rate*, and the match lists must agree exactly.
CROSSCHECK_ADMISSION_RATE_BOUND = 1e-3


def crosscheck_window(family: str, members: int = CROSSCHECK_MEMBERS) -> tuple[int, int]:
    """The declared crosscheck slice: centred on the family's first classic member."""

    total = S_SERIES_COUNT if family == "S" else P_PRODUCT_COUNT
    anchor = (
        builtin_ordinal(family, _builtin_entry(family, FAMILY_CLASSICS[family][0]["builtin_id"]))
        // len(PREFACTORS)
    )
    start = max(0, min(anchor - members // 2, total - members))
    return start, min(start + members, total)


def crosscheck_iterated(
    family: str, *, start: int | None = None, members: int = CROSSCHECK_MEMBERS
) -> dict[str, Any]:
    """Run the GPU and numpy sweeps over the same window and compare what they produce."""

    try:
        import cupy  # noqa: F401
    except Exception:  # noqa: BLE001 - no CUDA device
        return {"performed": False, "reason": "cupy unavailable"}
    total = S_SERIES_COUNT if family == "S" else P_PRODUCT_COUNT
    if start is None:
        start, stop = crosscheck_window(family, members)
    else:
        start = min(start, max(total - members, 0))
        stop = min(start + members, total)
    gpu_matches, gpu_counters, _ = sweep_iterated_gpu(family, index_start=start, index_stop=stop)
    cpu_matches, cpu_counters, _ = sweep_iterated_cpu(family, index_start=start, index_stop=stop)

    def key(item: Mapping[str, Any]) -> tuple[int, str, int]:
        return (int(item["member_index"]), str(item["target"]), int(item["prefactor_index"]))

    match_lists_agree = [key(item) for item in gpu_matches] == [key(item) for item in cpu_matches]
    admission_delta = abs(gpu_counters["resolved"] - cpu_counters["resolved"])
    span = max(stop - start, 1)
    rate = admission_delta / span
    return {
        "performed": True,
        "window": [start, stop],
        "members": span,
        "gpu_counters": gpu_counters,
        "cpu_counters": cpu_counters,
        "match_lists_identical": bool(match_lists_agree),
        "gpu_matches": len(gpu_matches),
        "cpu_matches": len(cpu_matches),
        "admission_disagreements": int(admission_delta),
        "admission_disagreement_rate": format(rate, ".3e"),
        "admission_disagreement_rate_bound": f"{CROSSCHECK_ADMISSION_RATE_BOUND:g}",
        "note": (
            "the two paths must agree exactly on what they report; the admission counter "
            "may differ on members sitting within rounding of the 1e-13 resolution gate, "
            "which is a conditioning heuristic and not a correctness criterion"
        ),
        "passed": bool(match_lists_agree and rate < CROSSCHECK_ADMISSION_RATE_BOUND),
    }


def run_families(
    *,
    families: Sequence[str] = FAMILIES,
    use_gpu: bool = True,
    member_limit: int | None = None,
    block: int = 1 << 23,
    capacity: int = 1 << 21,
    crosscheck: bool = True,
) -> dict[str, Any]:
    """Run every declared family and seal the enumeration receipt."""

    started = time.perf_counter()
    if use_gpu:
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        device = "cpu-numpy"

    fabricated = fabricated_near_miss_control()
    if not fabricated["control_passed"]:
        raise FamilyError(
            "the fabricated near-miss control did not behave: it must clear the fp64 window "
            f"and die at {VERIFY_STAGES[0]['stage']}; got "
            f"passed_fp64={fabricated['passed_fp64_window']} died_at={fabricated['died_at_stage']}"
        )

    blocks: list[dict[str, Any]] = []
    for family in families:
        block_result = run_family(
            family,
            use_gpu=use_gpu,
            member_limit=member_limit,
            block=block,
            capacity=capacity,
        )
        if crosscheck and use_gpu and family in ("S", "P"):
            block_result["crosscheck"] = crosscheck_iterated(family)
            if not block_result["crosscheck"].get("passed", False):
                raise FamilyError(f"GPU/CPU crosscheck failed for family {family}")
        else:
            block_result["crosscheck"] = {"performed": False, "reason": "not requested"}
        blocks.append(block_result)

    failed = [
        item["family"] for item in blocks if not item["controls"]["rediscovery"]["passed"]
    ]
    if failed and member_limit is None:
        raise FamilyError(
            "rediscovery controls failed for "
            f"{failed}: a family that cannot recover its own classical members cannot "
            "support a NOT_FOUND verdict, and its negative results are void"
        )

    total_survivors = sum(item["counts"]["survivors"] for item in blocks)
    not_in_table = sum(item["counts"]["prior_art"]["not_in_builtin_table"] for item in blocks)
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "lane": "inverse-symbolic-families",
        "claims": SHARED_CLAIMS,
        "shared_config": {
            "targets": [dict(item) for item in TARGETS],
            "prefactors": [str(item) for item in PREFACTORS],
            "verify_stages": [dict(stage) for stage in VERIFY_STAGES],
            "fp64_match_window": f"{FP64_MATCH_WINDOW:g}",
            "fp64_resolution_gate": f"{FP64_RESOLUTION_GATE:g}",
            "neville_weight_norms": {
                "order8": format(NEVILLE_W8_NORM, ".4f"),
                "order6": format(NEVILLE_W6_NORM, ".4f"),
            },
        },
        "controls": {"fabricated_near_miss": fabricated},
        "families": blocks,
        "totals": {
            "declared_ordinals": sum(FAMILY_ORDINALS[item["family"]] for item in blocks),
            "evaluated_ordinals": sum(item["scale"]["evaluated_ordinals"] for item in blocks),
            "fp64_matches": sum(item["counts"]["fp64_matches"] for item in blocks),
            "survivors": total_survivors,
            "not_in_builtin_table": not_in_table,
            "classics_rediscovered": {
                item["family"]: item["controls"]["rediscovery"]["classics_rediscovered"]
                for item in blocks
            },
            "family_classics_rediscovered": {
                item["family"]: item["controls"]["rediscovery"]["passed"] for item in blocks
            },
        },
        "scope": (
            "Exhaustive fp64 GPU enumeration of three declared, ordinal-indexed families -- "
            "hypergeometric-type series, infinite products and bounded definite integrals -- "
            "matched against thirteen declared constants across a twelve-element rational "
            "prefactor grid at 1e-12, and promoted only through exact mpmath re-evaluation "
            "at 60 and 120 digits. Survivors are conjectures, never theorems. The prior-art "
            "label compares against a finite built-in table only and never asserts novelty; "
            "adjudication against a real corpus and proof routing are separate receipts."
        ),
    }
    core = canonical_sha256(body)
    body["result_core_sha256"] = core
    body["measurement"] = {
        "device": device,
        "elapsed_seconds": format(time.perf_counter() - started, ".3f"),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------


def validate_receipt(value: Mapping[str, Any], *, replay_survivors: int = 4) -> None:
    """Seals, claims, count consistency, controls, and a bounded survivor replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise FamilyError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise FamilyError("receipt seal changed")
    core_body = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    if value.get("result_core_sha256") != canonical_sha256(core_body):
        raise FamilyError("deterministic core seal changed")
    if value.get("claims") != SHARED_CLAIMS:
        raise FamilyError("claims block changed")
    fabricated = value["controls"]["fabricated_near_miss"]
    if not fabricated["control_passed"]:
        raise FamilyError("receipt records a failed fabricated near-miss control")
    if fabricated["died_at_stage"] != VERIFY_STAGES[0]["stage"]:
        raise FamilyError("the fabricated near-miss did not die at the first verify stage")

    for block in value.get("families", []):
        family = block["family"]
        if family not in FAMILIES:
            raise FamilyError(f"unknown family in receipt: {family}")
        if block.get("config_sha256") != canonical_sha256(block.get("config", {})):
            raise FamilyError(f"config binding changed for family {family}")
        survivors = block.get("survivors", [])
        counts = block.get("counts", {})
        if counts.get("survivors") != len(survivors):
            raise FamilyError(f"survivor count changed for family {family}")
        if counts.get("fp64_matches") != len(survivors) + len(block.get("died", [])):
            raise FamilyError(f"match count changed for family {family}")
        labels = {"known_rediscovered": 0, "not_in_builtin_table": 0}
        for survivor in survivors:
            replayed = classify_prior_art(family, survivor, survivor["target"])
            if replayed != survivor.get("prior_art"):
                raise FamilyError(f"prior-art label changed for family {family}")
            labels[replayed["label"].lower()] += 1
        if counts.get("prior_art") != labels:
            raise FamilyError(f"prior-art counts changed for family {family}")
        if block["scale"]["exhaustive_this_run"] and not block["controls"]["rediscovery"]["passed"]:
            raise FamilyError(
                f"family {family} sealed an exhaustive run without rediscovering its classics"
            )
        stage = VERIFY_STAGES[0]
        for survivor in survivors[:replay_survivors]:
            prefactor = Fraction(survivor["prefactor"])
            with mp.workdps(int(stage["dps"])):
                exact = family_value_mp(family, survivor)
                scaled = exact * mp.mpf(prefactor.numerator) / prefactor.denominator
                error = abs(scaled - constant_value(survivor["target"]))
                if not error < mp.mpf(str(stage["threshold"])):
                    raise FamilyError(
                        f"stage-2 replay failed for family {family} ordinal {survivor['ordinal']}"
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_receipt(result: Mapping[str, Any], output: str) -> None:
    """Write a canonical, immutable receipt."""

    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise FamilyError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inverse symbolic engine: series, product, and integral families (DG3)."
    )
    parser.add_argument("--output", default="runs/math/inverse-symbolic/families-v1.json")
    parser.add_argument("--families", default="S,P,I")
    parser.add_argument("--cpu", action="store_true", help="force the numpy path")
    parser.add_argument("--member-limit", type=int, default=None)
    parser.add_argument("--block", type=int, default=1 << 23)
    parser.add_argument("--capacity", type=int, default=1 << 21)
    parser.add_argument("--no-crosscheck", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(json.dumps({"validated": True, "output": args.output}))
        return 0
    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    result = run_families(
        families=families,
        use_gpu=not args.cpu,
        member_limit=args.member_limit,
        block=args.block,
        capacity=args.capacity,
        crosscheck=not args.no_crosscheck,
    )
    write_receipt(result, args.output)
    print(
        json.dumps(
            {
                "families": [
                    {
                        "family": block["family"],
                        "declared_ordinals": block["scale"]["declared_total_ordinals"],
                        "evaluated_ordinals": block["scale"]["evaluated_ordinals"],
                        "ordinals_per_second": block["measurement"]["ordinals_per_second"],
                        "sweep_seconds": block["measurement"]["sweep_seconds"],
                        "fp64_matches": block["counts"]["fp64_matches"],
                        "survivors": block["counts"]["survivors"],
                        "not_in_builtin_table": block["counts"]["prior_art"][
                            "not_in_builtin_table"
                        ],
                        "classics_rediscovered": block["controls"]["rediscovery"]["passed"],
                    }
                    for block in result["families"]
                ],
                "totals": result["totals"],
                "device": result["measurement"]["device"],
                "elapsed_seconds": result["measurement"]["elapsed_seconds"],
                "output": args.output,
                "content_sha256": result["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
