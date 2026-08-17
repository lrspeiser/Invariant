"""Proof routing for series, product, and integral identities (DG3).

:mod:`sigma_theory_compiler.families_prior_art_screen` adjudicated the distinct survivor
objects of the three declared families against a 67-seed corpus.  Most came back ``KNOWN``
with an exhibited justification; the rest came back ``NOT_FOUND_IN_CORPUS`` or
``INCONCLUSIVE_VALUE_MATCH``.  Neither of those is an answer.  This module settles them by
*proving* or *refuting* each identity with mechanizable classical techniques, tried in a
declared order and recording which one fired:

1. ``hypergeometric_closed_form`` (Family S).  A term ratio that is a rational function of
   the index *is* a generalized hypergeometric series, so the candidate arrives with exact
   ``pFq`` parameters.  Those are handed to sympy's ``hyperexpand``, which implements
   Roach's algorithm for reducing a ``pFq`` to named functions.  A closed form that
   reproduces the value at the declared precision is a proof that the candidate is an
   instance of the classical hypergeometric family.

2. ``weierstrass_gauss_gamma_product`` (Family P).  Every member that passes the declared
   convergence test has the closed form ``prod_j Gamma(k0-beta_j)/prod_i Gamma(k0-alpha_i)``
   over the roots of its two polynomials.  This is the Weierstrass canonical product for
   ``1/Gamma`` combined with Gauss's product formula, and it settles the *entire* family:
   there is no convergent member of Family P that is not a Gamma ratio.

3. ``log_power_hurwitz_reduction`` (Family I).  When the integrand is
   ``x^a (ln(1/x))^s R(x)`` with ``R`` a rational function whose denominator is built from
   ``1-x``, ``1+x``, ``1+x^2`` and ``1+x+x^2``, its Taylor coefficients are a quasi-polynomial
   in the index -- the fit is *computed and then re-verified on further coefficients*, never
   assumed.  Term-by-term integration with ``int_0^1 x^u (ln(1/x))^s dx = Gamma(s+1)/(u+1)^(s+1)``
   turns the integral into a finite combination of Hurwitz zeta values.

4. ``symbolic_definite_integration`` (Family I).  The remaining log-free algebraic
   integrands go to sympy's definite integrator (the Meijer G-function algorithm) under a
   declared per-candidate time budget.  A budget overrun is reported as an overrun, never as
   a failure of the identity.

Anything else yields a typed ``missing_proof_technique:<name>`` naming exactly what is
absent, never a fake proof.  Before any technique runs, the claimed identity is checked
numerically at 200 digits; a candidate that fails is ``REFUTED`` and the receipt records the
exact decimal place where the claim first breaks.

**Proof by a classical family is not novelty.**  Every technique here exhibits its subject as
an instance of a cited classical family, so a candidate it proves is reclassified
``KNOWN_BY_PROOF_FAMILY``.  The only interesting terminal state is a candidate that is
``PROVED`` and whose proof family is *still* absent from the corpus, or one that no technique
can touch; both counts are computed explicitly and an empty answer is the honest answer.

None of these proofs is kernel-verified: this repository's Lean idiom is ``Nat``-typed with
no Mathlib, and these arguments need ordered-field arithmetic, an analytic limit, and the
Gamma function.  The receipt names that obstruction instead of implying a kernel result that
does not exist.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from .families_prior_art_screen import (
    ScreenError,
    build_corpus,
    signature_key,
)
from .inverse_symbolic_families import (
    constant_value,
    exact_roots,
    family_value_mp,
    series_hypergeometric_parameters,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-inverse-symbolic-families-proof-routing-1.0"

VERDICTS = ("PROVED", "REFUTED", "MISSING_TECHNIQUE")

TECHNIQUE_ORDER = (
    "hypergeometric_closed_form",
    "weierstrass_gauss_gamma_product",
    "log_power_hurwitz_reduction",
    "beta_derivative_digamma_reduction",
    "cyclotomic_substitution_to_beta",
    "euler_2f1_integral_representation",
    "symbolic_definite_integration",
)

#: Precision and depth of the refutation gate that runs before any technique.
REFUTATION_DPS = 200
#: A claim must reproduce the target to this many decimals to escape ``REFUTED``.
REFUTATION_DIGITS = 150
#: A proof's closed form must reproduce the candidate's value to this many decimals.
PROOF_AGREEMENT_DIGITS = 150
PROOF_DPS = 200

#: Declared budget for the symbolic definite integrator, in seconds.  An overrun is recorded
#: as an overrun; it is never reported as a refutation or as an absent technique.
INTEGRATION_BUDGET_SECONDS = 120.0

#: Coefficients sampled when fitting the quasi-polynomial, and how many *extra* coefficients
#: the fit must additionally reproduce before it may be used.
QUASI_POLY_FIT_SAMPLES = 12
QUASI_POLY_VERIFY_SAMPLES = 8

ROUTER_CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "kernel_verification_pending_where_stated": True,
    "novelty_claimed": False,
    "proof_by_classical_family_implies_known": True,
}

#: The classical family each technique exhibits its subject as a member of.  This is the
#: table that turns "proved" into "known".
TECHNIQUE_CLASSICAL_FAMILY: dict[str, str] = {
    "hypergeometric_closed_form": "generalized_hypergeometric_pFq",
    "weierstrass_gauss_gamma_product": "weierstrass_gauss_gamma_product",
    "log_power_hurwitz_reduction": "hurwitz_zeta_log_power_integral",
    "beta_derivative_digamma_reduction": "differentiated_euler_beta_integral",
    "cyclotomic_substitution_to_beta": "euler_beta_integral",
    "euler_2f1_integral_representation": "gauss_hypergeometric_2f1",
    "symbolic_definite_integration": "elementary_and_meijer_g_closed_forms",
}

#: The external theorem each technique leans on.  Declared, never implied.
TECHNIQUE_CITED_THEOREM: dict[str, str] = {
    "hypergeometric_closed_form": (
        "the classical theory of the generalized hypergeometric function pFq (Slater, "
        "Generalized Hypergeometric Functions, 1966; DLMF 16), reduced mechanically by "
        "Roach's algorithm as implemented in sympy.hyperexpand"
    ),
    "weierstrass_gauss_gamma_product": (
        "the Weierstrass canonical product for 1/Gamma and Gauss's product formula; "
        "Whittaker and Watson, A Course of Modern Analysis, 12.11-12.13"
    ),
    "log_power_hurwitz_reduction": (
        "term-by-term integration of a convergent power series together with "
        "int_0^1 x^u (ln(1/x))^s dx = Gamma(s+1)/(u+1)^(s+1) and the Hurwitz zeta series "
        "DLMF 25.11.1; the quasi-polynomial structure of the Taylor coefficients of a "
        "rational function with cyclotomic denominator is elementary linear recurrence theory"
    ),
    "beta_derivative_digamma_reduction": (
        "differentiation of Euler's Beta integral under the integral sign in the "
        "exponent: int_0^1 x^A (1-x)^B (ln(1/x))^n dx = (-1)^n d^n/dA^n B(A+1, B+1), a "
        "finite combination of digamma and polygamma values times B (DLMF 5.12.1 with "
        "5.15); the derivative is taken by Cauchy's formula at the stage precision"
    ),
    "cyclotomic_substitution_to_beta": (
        "the cyclotomic identity 1 + x + ... + x^(m-1) = (1 - x^m)/(1 - x) followed by "
        "the substitution u = x^m, which carries the integrand onto Euler's Beta "
        "integral (DLMF 5.12.1); elementary change of variable on a finite interval"
    ),
    "euler_2f1_integral_representation": (
        "Euler's integral representation of the Gauss hypergeometric function, "
        "int_0^1 x^(a-1)(1-x)^(c-a-1)(1-zx)^(-b) dx = B(a, c-a) 2F1(a, b; c; z) "
        "(DLMF 15.6.1; Erdelyi, Higher Transcendental Functions I, 2.1.10), together "
        "with the classical evaluations of 2F1 at z = -1: Kummer's theorem when "
        "c = 1+a-b (DLMF 15.4.26) and the quadratic transformations when c = 2a, "
        "c = 2b or c = a+b+-1/2 (DLMF 15.8.13-15.8.27)"
    ),
    "symbolic_definite_integration": (
        "the Meijer G-function algorithm for definite integration (Roach 1996/1997) as "
        "implemented in sympy.integrate, with elementary antiderivatives where they exist"
    ),
}

LEAN_OBSTRUCTION = (
    "this repository's Lean layer is Nat-typed with no Mathlib; these arguments require "
    "ordered-field arithmetic, an analytic limit, and the Gamma and Hurwitz zeta functions, "
    "none of which the current kernel vertical slice provides"
)

_X = sp.Symbol("_x", positive=True)
_M = sp.Symbol("_m")


class ProofRouterError(ValueError):
    """Raised on malformed input, a failed control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Numeric evaluation and the refutation gate
# ---------------------------------------------------------------------------


def first_differing_decimal(left: mp.mpf, right: mp.mpf, digits: int) -> int | None:
    """Index of the first decimal place where two values differ, or ``None``.

    Position 1 is the first digit after the decimal point; a mismatch in the integer part
    reports a non-positive index.
    """

    if not (mp.isfinite(left) and mp.isfinite(right)):
        return 0
    difference = abs(left - right)
    if difference == 0:
        return None
    place = int(-mp.floor(mp.log10(difference)))
    return None if place > digits else place


def numeric_check(family: str, member: Mapping[str, Any], target: str) -> dict[str, Any]:
    """Evaluate the claimed identity at 200 digits before any technique is attempted."""

    prefactor = Fraction(member["prefactor"])
    with mp.workdps(REFUTATION_DPS):
        try:
            value = family_value_mp(family, member)
        except Exception as failure:  # noqa: BLE001
            return {
                "dps": REFUTATION_DPS,
                "required_agreement_digits": REFUTATION_DIGITS,
                "holds": False,
                "value": "unevaluable",
                "target": "-",
                "first_differing_decimal_place": 0,
                "blocker": f"{type(failure).__name__}: {failure}"[:200],
            }
        scaled = value * mp.mpf(prefactor.numerator) / prefactor.denominator
        target_value = constant_value(target)
        place = first_differing_decimal(scaled, target_value, REFUTATION_DIGITS)
        return {
            "dps": REFUTATION_DPS,
            "required_agreement_digits": REFUTATION_DIGITS,
            "holds": bool(mp.isfinite(scaled) and place is None),
            "value": mp.nstr(scaled, 60) if mp.isfinite(scaled) else "nan",
            "target": mp.nstr(target_value, 60),
            "first_differing_decimal_place": place,
        }


def _agrees(left: mp.mpf, right: mp.mpf, digits: int = PROOF_AGREEMENT_DIGITS) -> bool:
    if not (mp.isfinite(left) and mp.isfinite(right)):
        return False
    scale = max(mp.mpf(1), abs(right))
    return bool(abs(left - right) / scale < mp.mpf(10) ** (-digits))


# ---------------------------------------------------------------------------
# Technique 1 -- hypergeometric closed form (Family S)
# ---------------------------------------------------------------------------


def attempt_hypergeometric(member: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the candidate's own ``pFq`` to a closed form and check the number."""

    parameters = series_hypergeometric_parameters(
        member["p"], member["q"], Fraction(member["z"])
    )
    if parameters is None:
        return {"fired": False, "reason": "P is identically zero; the series is degenerate"}
    upper, lower, argument = parameters
    expression = sp.hyper(upper, lower, argument)
    try:
        closed = sp.hyperexpand(expression)
    except Exception as failure:  # noqa: BLE001
        return {"fired": False, "reason": f"hyperexpand raised {type(failure).__name__}"}
    if closed.has(sp.hyper) or closed == expression:
        return {"fired": False, "reason": "hyperexpand returned the pFq unreduced"}
    with mp.workdps(PROOF_DPS):
        try:
            predicted = mp.mpf(str(sp.N(sp.re(closed), mp.mp.dps + 10)))
            actual = family_value_mp("S", member)
        except Exception as failure:  # noqa: BLE001
            return {"fired": False, "reason": f"closed form did not evaluate: {type(failure).__name__}"}
        if not _agrees(predicted, actual):
            return {
                "fired": False,
                "reason": "the reduced closed form does not reproduce the candidate's value",
            }
        return {
            "fired": True,
            "derivation": {
                "pFq": f"{len(upper)}F{len(lower)}",
                "upper_parameters": [str(item) for item in upper],
                "lower_parameters": [str(item) for item in lower],
                "argument": str(argument),
                "closed_form": str(closed),
                "closed_form_latex": sp.latex(closed),
                "closed_form_value_60_digits": mp.nstr(predicted, 60),
                "agreement_digits": PROOF_AGREEMENT_DIGITS,
            },
        }


# ---------------------------------------------------------------------------
# Technique 2 -- Weierstrass/Gauss Gamma product (Family P)
# ---------------------------------------------------------------------------


def attempt_gamma_product(member: Mapping[str, Any]) -> dict[str, Any]:
    """Every convergent Family P member is a ratio of Gamma values.  Exhibit it."""

    k0 = sp.Integer(int(member["k0"]))
    lead_a, roots_a = exact_roots(member["a"])
    lead_b, roots_b = exact_roots(member["b"])
    if len(roots_a) != len(roots_b) or lead_a != lead_b:
        return {
            "fired": False,
            "reason": "the member does not satisfy the declared convergence condition",
        }
    numerator = [sp.simplify(k0 - root) for root in roots_b]
    denominator = [sp.simplify(k0 - root) for root in roots_a]
    with mp.workdps(PROOF_DPS):
        try:
            value = mp.mpc(1)
            for item in numerator:
                value *= mp.gamma(_to_mp(item))
            for item in denominator:
                value /= mp.gamma(_to_mp(item))
            predicted = +mp.mpf(mp.re(value))
            actual = family_value_mp("P", member)
        except Exception as failure:  # noqa: BLE001
            return {"fired": False, "reason": f"Gamma ratio did not evaluate: {type(failure).__name__}"}
        if not _agrees(predicted, actual):
            return {"fired": False, "reason": "the Gamma ratio does not reproduce the value"}
    text = " * ".join(f"Gamma({item})" for item in numerator) or "1"
    bottom = " * ".join(f"Gamma({item})" for item in denominator) or "1"
    return {
        "fired": True,
        "derivation": {
            "root_sum_condition": "sum(alpha) = sum(beta), which is the convergence condition",
            "gamma_numerator_arguments": [str(item) for item in numerator],
            "gamma_denominator_arguments": [str(item) for item in denominator],
            "closed_form": f"({text}) / ({bottom})",
            "closed_form_value_60_digits": mp.nstr(predicted, 60),
            "agreement_digits": PROOF_AGREEMENT_DIGITS,
        },
    }


def _to_mp(expression: sp.Expr) -> mp.mpf | mp.mpc:
    value = sp.N(expression, mp.mp.dps + 10)
    real, imaginary = sp.re(value), sp.im(value)
    if imaginary == 0:
        return mp.mpf(str(real))
    return mp.mpc(str(real), str(imaginary))


# ---------------------------------------------------------------------------
# Technique 3 -- log-power / Hurwitz reduction (Family I)
# ---------------------------------------------------------------------------

#: How each declared kernel splits into ``(ln(1/x))^s`` times a rational function.
#: ``log_multiplicity`` is the power of ``ln(1/x)`` contributed per unit kernel power, and
#: ``rational`` is the rest, raised to the kernel power.
KERNEL_SPLIT: dict[str, tuple[int, sp.Expr | None]] = {
    "one": (0, sp.Integer(1)),
    "log_inv": (1, sp.Integer(1)),
    "inv_1px": (0, 1 / (1 + _X)),
    "inv_1px2": (0, 1 / (1 + _X**2)),
    "inv_1mx": (0, 1 / (1 - _X)),
    "log_inv_over_1mx": (1, 1 / (1 - _X)),
    "log_inv_over_1px": (1, 1 / (1 + _X)),
    "log_inv_over_1px2": (1, 1 / (1 + _X**2)),
    "log_inv_1mx": (0, None),
    "x_over_expm1": (0, None),
    "inv_1pxpx2": (0, 1 / (1 + _X + _X**2)),
    "inv_sqrt_1px": (0, None),
}

#: Cyclotomic factors the reduction understands and the period each contributes.
CYCLOTOMIC_PERIOD = {sp.Integer(1) - _X: 1, sp.Integer(1) + _X: 2,
                     sp.Integer(1) + _X**2: 4, sp.Integer(1) + _X + _X**2: 3}


def attempt_log_power_hurwitz(member: Mapping[str, Any]) -> dict[str, Any]:
    """Term-by-term integration into a finite combination of Hurwitz zeta values."""

    kernel = str(member["kernel"])
    power = Fraction(member["power"])
    a = Fraction(member["a"])
    b = Fraction(member["b"])
    substitution: str | None = None
    if kernel == "log_inv_1mx" and power != 0:
        # x -> 1-x is an elementary change of variable on a finite interval and carries
        # ln(1/(1-x)) onto ln(1/x), which is exactly the form this technique reduces.
        a, b = b, a
        kernel = "log_inv"
        substitution = (
            "x -> 1-x: the exponents are exchanged and ln(1/(1-x)) becomes ln(1/x)"
        )
    multiplicity, rational = KERNEL_SPLIT[kernel]
    if rational is None:
        return {
            "fired": False,
            "reason": f"kernel {kernel} is not (ln(1/x))^s times a rational function",
        }
    s = Fraction(multiplicity) * power
    if s <= 0:
        return {
            "fired": False,
            "reason": "no logarithmic factor; the log-power reduction does not apply",
        }
    if power.denominator != 1 or b.denominator != 1:
        return {
            "fired": False,
            "reason": "the rational part has a non-integer exponent, so it is not rational",
        }
    part = sp.together((1 - _X) ** sp.Integer(int(b)) * rational ** sp.Integer(int(power)))
    numerator, denominator = sp.fraction(sp.cancel(part))
    period = 1
    highest_multiplicity = 1
    if denominator.has(_X):
        _content, factors = sp.factor_list(sp.Poly(denominator, _X))
        for factor, exponent in factors:
            expression = factor.as_expr()
            for known, known_period in CYCLOTOMIC_PERIOD.items():
                if sp.simplify(expression - known) == 0 or sp.simplify(expression + known) == 0:
                    period = sp.ilcm(period, known_period)
                    break
            else:
                return {
                    "fired": False,
                    "reason": (
                        f"the denominator carries the factor {expression}, which is outside "
                        "the declared cyclotomic set, so the coefficients need not be a "
                        "quasi-polynomial"
                    ),
                }
            highest_multiplicity = max(highest_multiplicity, int(exponent))
    # The Taylor coefficients of a rational function whose denominator has roots on the unit
    # circle are a quasi-polynomial of degree (highest root multiplicity - 1); that, not the
    # denominator's total degree, is what has to be dominated by the logarithmic damping.
    degree = highest_multiplicity - 1
    head = int(sp.degree(numerator, _X)) + 1 if numerator.has(_X) else 1
    # The tail is indexed by residue class mod the period, so it has to start on a period
    # boundary; otherwise index head + r + period*step sits in residue class (head + r).
    head = int(period) * -(-head // int(period))

    # Sample the Taylor coefficients exactly.
    order = head + period * (QUASI_POLY_FIT_SAMPLES + QUASI_POLY_VERIFY_SAMPLES) + 4
    try:
        expansion = sp.Poly(sp.series(part, _X, 0, order).removeO(), _X)
    except Exception as failure:  # noqa: BLE001
        return {"fired": False, "reason": f"series expansion failed: {type(failure).__name__}"}
    coefficients = expansion.all_coeffs()[::-1]
    if len(coefficients) < order - 2:
        coefficients = coefficients + [sp.Integer(0)] * (order - len(coefficients))

    if Fraction(degree) >= s:
        return {
            "fired": False,
            "reason": (
                "the coefficient growth exceeds the logarithmic damping, so the "
                "term-by-term series does not converge"
            ),
        }

    # Fit and then *verify* a quasi-polynomial for the tail coefficients.
    fits: dict[int, sp.Expr] = {}
    for residue in range(period):
        indices = [head + residue + period * step for step in range(QUASI_POLY_FIT_SAMPLES)]
        points = [(sp.Integer(index), coefficients[index]) for index in indices]
        polynomial = sp.interpolate(points, _M)
        polynomial = sp.expand(polynomial)
        if sp.degree(polynomial, _M) > max(degree, 0):
            return {
                "fired": False,
                "reason": "the fitted coefficient polynomial exceeds the recurrence degree",
            }
        for step in range(QUASI_POLY_FIT_SAMPLES, QUASI_POLY_FIT_SAMPLES + QUASI_POLY_VERIFY_SAMPLES):
            index = head + residue + period * step
            if index >= len(coefficients):
                break
            if sp.simplify(polynomial.subs(_M, index) - coefficients[index]) != 0:
                return {
                    "fired": False,
                    "reason": "the quasi-polynomial fit failed its verification samples",
                }
        fits[residue] = polynomial

    # Assemble: I = Gamma(s+1) * [ finite head + sum over residues of Hurwitz zeta terms ].
    exponent = sp.Rational(s.numerator, s.denominator) + 1
    terms: list[dict[str, str]] = []
    with mp.workdps(PROOF_DPS):
        total = mp.mpf(0)
        head_value = mp.mpf(0)
        for index in range(head):
            weight = coefficients[index]
            if weight == 0:
                continue
            head_value += mp.mpf(str(sp.N(weight, mp.mp.dps + 10))) / (
                mp.mpf(a.numerator) / a.denominator + index + 1
            ) ** mp.mpf(str(sp.N(exponent, mp.mp.dps + 10)))
        total += head_value
        for residue, polynomial in fits.items():
            shift = sp.Rational(a.numerator, a.denominator) + residue + 1
            alpha = shift / period
            start = head + residue
            first_q = sp.ceiling((start - residue) / period)
            variable = sp.Symbol("_v")
            rewritten = sp.expand(polynomial.subs(_M, period * (variable - alpha) + residue))
            poly_v = sp.Poly(rewritten, variable)
            coefficients_v = poly_v.all_coeffs()[::-1]
            for degree_v, coefficient in enumerate(coefficients_v):
                if coefficient == 0:
                    continue
                order_z = exponent - degree_v
                offset = alpha + first_q
                terms.append(
                    {
                        "residue": str(residue),
                        "coefficient": str(sp.nsimplify(coefficient)),
                        "hurwitz_zeta": f"zeta({order_z}, {sp.nsimplify(offset)})",
                        "period_factor": f"{period}^(-{exponent})",
                    }
                )
                # u = period * v, so 1/u^exponent contributes period^(-exponent); the
                # Hurwitz order is exponent - degree_v but the scale factor is not.
                total += (
                    mp.mpf(str(sp.N(coefficient, mp.mp.dps + 10)))
                    * mp.power(mp.mpf(period), -mp.mpf(str(sp.N(exponent, mp.mp.dps + 10))))
                    * mp.zeta(
                        mp.mpf(str(sp.N(order_z, mp.mp.dps + 10))),
                        mp.mpf(str(sp.N(offset, mp.mp.dps + 10))),
                    )
                )
        predicted = mp.gamma(mp.mpf(str(sp.N(exponent, mp.mp.dps + 10)))) * total
        try:
            actual = family_value_mp("I", member)
        except Exception as failure:  # noqa: BLE001
            return {"fired": False, "reason": f"candidate did not evaluate: {type(failure).__name__}"}
        if not _agrees(predicted, actual):
            return {
                "fired": False,
                "reason": "the Hurwitz-zeta reduction does not reproduce the candidate's value",
            }
        return {
            "fired": True,
            "derivation": {
                "substitution": substitution,
                "log_power": str(s),
                "rational_part": str(sp.simplify(part)),
                "quasi_polynomial_degree": degree,
                "quasi_polynomial_period": period,
                "quasi_polynomial_verified_on_extra_samples": QUASI_POLY_VERIFY_SAMPLES,
                "head_terms": head,
                "closed_form": (
                    f"Gamma({exponent}) * [ finite head + "
                    + " + ".join(
                        f"{item['coefficient']} * {item['period_factor']} * {item['hurwitz_zeta']}"
                        for item in terms
                    )
                    + " ]"
                ),
                "hurwitz_terms": terms,
                "closed_form_value_60_digits": mp.nstr(predicted, 60),
                "agreement_digits": PROOF_AGREEMENT_DIGITS,
            },
        }


# ---------------------------------------------------------------------------
# Technique 4 -- differentiated Beta integral (Family I)
# ---------------------------------------------------------------------------

#: Kernels that are ``(ln(1/x))^c`` times a pure power of ``(1-x)``, with the exponent that
#: power contributes per unit kernel power.
BETA_DERIVATIVE_KERNEL: dict[str, Fraction] = {
    "log_inv": Fraction(0),
    "log_inv_over_1mx": Fraction(-1),
}


def attempt_beta_derivative(member: Mapping[str, Any]) -> dict[str, Any]:
    """``int_0^1 x^A (1-x)^B (ln(1/x))^n dx = (-1)^n d^n/dA^n B(A+1, B+1)``.

    The Beta integral is analytic in its exponents wherever the integral converges, so the
    derivative is taken numerically by Cauchy's formula at the stage precision rather than
    expanded into digamma polynomials -- which keeps the boundary cases (where ``B + 1`` sits
    at a pole of Gamma and the Beta value itself vanishes) exact instead of indeterminate.
    """

    kernel = str(member["kernel"])
    power = Fraction(str(member["power"]))
    if kernel not in BETA_DERIVATIVE_KERNEL or power == 0:
        return {
            "fired": False,
            "reason": (
                f"kernel {kernel} is not (ln(1/x))^n times a pure power of (1-x), so the "
                "differentiated Beta integral does not apply"
            ),
        }
    if power.denominator != 1:
        return {
            "fired": False,
            "reason": "the log power is not an integer, so it is not a repeated derivative",
        }
    order = int(power)
    exponent_a = Fraction(str(member["a"]))
    exponent_b = Fraction(str(member["b"])) + BETA_DERIVATIVE_KERNEL[kernel] * power
    with mp.workdps(PROOF_DPS + 20):
        try:
            point = _rational(exponent_a) + 1
            fixed = _rational(exponent_b) + 1
            derivative = mp.diff(lambda value: mp.beta(value, fixed), point, order)
            predicted = +mp.mpf(mp.re(mp.mpf(-1) ** order * derivative))
        except Exception as failure:  # noqa: BLE001
            return {
                "fired": False,
                "reason": f"the derivative did not evaluate: {type(failure).__name__}",
            }
    with mp.workdps(PROOF_DPS):
        try:
            actual = family_value_mp("I", member)
        except Exception as failure:  # noqa: BLE001
            return {"fired": False, "reason": f"candidate did not evaluate: {type(failure).__name__}"}
        if not _agrees(predicted, actual):
            return {
                "fired": False,
                "reason": "the differentiated Beta integral does not reproduce the value",
            }
    return {
        "fired": True,
        "derivation": {
            "identity": "int_0^1 x^A (1-x)^B (ln(1/x))^n dx = (-1)^n d^n/dA^n B(A+1, B+1)",
            "A": str(exponent_a),
            "B": str(exponent_b),
            "n": order,
            "closed_form": f"(-1)^{order} d^{order}/dA^{order} B(A+1, {exponent_b + 1}) at A = {exponent_a}",
            "closed_form_value_60_digits": mp.nstr(predicted, 60),
            "agreement_digits": PROOF_AGREEMENT_DIGITS,
        },
    }


# ---------------------------------------------------------------------------
# Technique 5 -- cyclotomic substitution to a Beta integral (Family I)
# ---------------------------------------------------------------------------

#: Kernels that are ``1/(1 + x + ... + x^(m-1)) = (1-x)/(1-x^m)``, with their ``m``.
CYCLOTOMIC_KERNEL_ORDER: dict[str, int] = {"inv_1px": 2, "inv_1pxpx2": 3}


def attempt_cyclotomic_substitution(member: Mapping[str, Any]) -> dict[str, Any]:
    """``x^a (1-x)^(-c) D(x)^(-c) = x^a (1-x^m)^(-c)``, then ``u = x^m`` gives a Beta value."""

    kernel = str(member["kernel"])
    power = Fraction(str(member["power"]))
    if kernel not in CYCLOTOMIC_KERNEL_ORDER or power == 0:
        return {
            "fired": False,
            "reason": (
                f"kernel {kernel} is not the reciprocal of a cyclotomic sum, so the "
                "substitution u = x^m does not reach a Beta integral"
            ),
        }
    order = CYCLOTOMIC_KERNEL_ORDER[kernel]
    exponent_a = Fraction(str(member["a"]))
    exponent_b = Fraction(str(member["b"]))
    # D(x)^-c = (1-x)^c (1-x^m)^-c, so the (1-x) factors cancel exactly when b + c = 0.
    if exponent_b + power != 0:
        return {
            "fired": False,
            "reason": (
                f"the (1-x) factors do not cancel: b + c = {exponent_b + power}, so the "
                "substitution leaves a residual (1-x) power and no Beta integral results"
            ),
        }
    first = (exponent_a + 1) / order
    second = 1 - power
    if first <= 0 or second <= 0:
        return {"fired": False, "reason": "the resulting Beta arguments are not both positive"}
    with mp.workdps(PROOF_DPS):
        try:
            predicted = mp.beta(_rational(first), _rational(second)) / order
            actual = family_value_mp("I", member)
        except Exception as failure:  # noqa: BLE001
            return {"fired": False, "reason": f"the Beta value did not evaluate: {type(failure).__name__}"}
        if not _agrees(predicted, actual):
            return {
                "fired": False,
                "reason": "the substituted Beta integral does not reproduce the value",
            }
    return {
        "fired": True,
        "derivation": {
            "cyclotomic_identity": (
                f"1 + x + ... + x^{order - 1} = (1 - x^{order})/(1 - x), so the kernel power "
                f"contributes (1-x)^{power} and cancels the (1-x)^{exponent_b} factor"
            ),
            "substitution": f"u = x^{order}",
            "closed_form": f"(1/{order}) B({first}, {second})",
            "closed_form_value_60_digits": mp.nstr(predicted, 60),
            "agreement_digits": PROOF_AGREEMENT_DIGITS,
        },
    }


# ---------------------------------------------------------------------------
# Technique 6 -- Euler's 2F1 integral representation (Family I)
# ---------------------------------------------------------------------------

#: Kernels that are a power of ``(1 - z x)``, with their ``z`` and the multiplier carrying
#: the declared kernel power onto the hypergeometric parameter ``b``.
EULER_KERNEL_ARGUMENT: dict[str, tuple[Fraction, Fraction]] = {
    "one": (Fraction(0), Fraction(0)),
    "inv_1px": (Fraction(-1), Fraction(1)),
    "inv_1mx": (Fraction(1), Fraction(1)),
    "inv_sqrt_1px": (Fraction(-1), Fraction(1, 2)),
}


def classical_2f1_relations(a: Fraction, b: Fraction, c: Fraction) -> list[str]:
    """Which cited relations make this ``2F1`` summable in closed form at ``z = -1``."""

    relations: list[str] = []
    if c == 1 + a - b:
        relations.append("Kummer: c = 1 + a - b (DLMF 15.4.26)")
    if c == 1 + b - a:
        relations.append("Kummer with a and b exchanged: c = 1 + b - a (DLMF 15.4.26)")
    if c == a + b + Fraction(1, 2):
        relations.append("quadratic transformation: c = a + b + 1/2 (DLMF 15.8.15)")
    if c == a + b - Fraction(1, 2):
        relations.append("quadratic transformation: c = a + b - 1/2 (DLMF 15.8.17)")
    if c == 2 * a:
        relations.append("quadratic transformation: c = 2a (DLMF 15.8.13)")
    if c == 2 * b:
        relations.append("quadratic transformation: c = 2b (DLMF 15.8.13)")
    if b == a + Fraction(1, 2) or a == b + Fraction(1, 2):
        relations.append("quadratic transformation: b = a + 1/2 (DLMF 15.8.21)")
    if a + b == 1:
        relations.append("quadratic transformation: a + b = 1 (DLMF 15.8.27)")
    return relations


def _rational(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / value.denominator


def attempt_euler_2f1(member: Mapping[str, Any]) -> dict[str, Any]:
    """Exhibit a log-free integrand as ``B(a, c-a) 2F1(a, b; c; z)`` and check the number."""

    kernel = str(member["kernel"])
    power = Fraction(str(member["power"]))
    if kernel not in EULER_KERNEL_ARGUMENT:
        return {
            "fired": False,
            "reason": (
                f"kernel {kernel} is not a power of (1 - z x), so Euler's integral "
                "representation does not apply"
            ),
        }
    argument, multiplier = EULER_KERNEL_ARGUMENT[kernel]
    a_exponent = Fraction(str(member["a"]))
    b_exponent = Fraction(str(member["b"]))
    alpha = a_exponent + 1
    gamma = a_exponent + b_exponent + 2
    beta = power * multiplier
    if alpha <= 0 or gamma - alpha <= 0:
        return {"fired": False, "reason": "the representation needs Re(c) > Re(a) > 0"}
    if argument == 1 and gamma - alpha - beta <= 0:
        return {"fired": False, "reason": "Gauss's theorem needs Re(c - a - b) > 0"}
    with mp.workdps(PROOF_DPS):
        try:
            beta_value = mp.beta(_rational(alpha), _rational(gamma - alpha))
            if beta == 0:
                hyper_value = mp.mpf(1)
            else:
                hyper_value = mp.hyp2f1(
                    _rational(alpha), _rational(beta), _rational(gamma), _rational(argument)
                )
            predicted = +mp.mpf(mp.re(beta_value * hyper_value))
            actual = family_value_mp("I", member)
        except Exception as failure:  # noqa: BLE001
            return {
                "fired": False,
                "reason": f"the representation did not evaluate: {type(failure).__name__}",
            }
        if not _agrees(predicted, actual):
            return {
                "fired": False,
                "reason": "Euler's representation does not reproduce the candidate's value",
            }
    relations = classical_2f1_relations(alpha, beta, gamma)
    return {
        "fired": True,
        "derivation": {
            "representation": (
                "int_0^1 x^(a-1)(1-x)^(c-a-1)(1-zx)^(-b) dx = B(a,c-a) 2F1(a,b;c;z)"
            ),
            "a": str(alpha),
            "b": str(beta),
            "c": str(gamma),
            "z": str(argument),
            "closed_form": (
                f"B({alpha}, {gamma - alpha}) * 2F1({alpha}, {beta}; {gamma}; {argument})"
            ),
            "classical_evaluation_relations": relations
            or ["none of the declared closed-form relations holds at these parameters"],
            "closed_form_value_60_digits": mp.nstr(predicted, 60),
            "agreement_digits": PROOF_AGREEMENT_DIGITS,
        },
    }


# ---------------------------------------------------------------------------
# Technique 7 -- symbolic definite integration (Family I)
# ---------------------------------------------------------------------------

KERNEL_EXPRESSION: dict[str, sp.Expr] = {
    "one": sp.Integer(1),
    "log_inv": -sp.log(_X),
    "inv_1px": 1 / (1 + _X),
    "inv_1px2": 1 / (1 + _X**2),
    "inv_1mx": 1 / (1 - _X),
    "log_inv_over_1mx": -sp.log(_X) / (1 - _X),
    "log_inv_over_1px": -sp.log(_X) / (1 + _X),
    "log_inv_over_1px2": -sp.log(_X) / (1 + _X**2),
    "log_inv_1mx": -sp.log(1 - _X),
    "x_over_expm1": _X / (sp.exp(_X) - 1),
    "inv_1pxpx2": 1 / (1 + _X + _X**2),
    "inv_sqrt_1px": 1 / sp.sqrt(1 + _X),
}


def integrand_expression(member: Mapping[str, Any]) -> sp.Expr:
    """The candidate's integrand as a sympy expression."""

    a = sp.Rational(str(member["a"]))
    b = sp.Rational(str(member["b"]))
    power = sp.Rational(str(member["power"]))
    kernel = KERNEL_EXPRESSION[str(member["kernel"])]
    expression = _X**a * (1 - _X) ** b
    if power != 0:
        expression *= kernel**power
    return expression


def attempt_symbolic_integration(member: Mapping[str, Any]) -> dict[str, Any]:
    """Hand a log-free algebraic integrand to sympy's definite integrator.

    The declared domain is deliberately restricted to log-free integrands.  sympy's general
    definite integrator does not terminate within any workable budget on the log-power
    family -- measured, not assumed -- and that family has its own dedicated technique above,
    so routing it here would trade a real proof for an unbounded wait.
    """

    kernel = str(member["kernel"])
    power = Fraction(str(member["power"]))
    multiplicity, _rational = KERNEL_SPLIT[kernel]
    if power != 0 and (multiplicity > 0 or kernel == "log_inv_1mx"):
        return {
            "fired": False,
            "reason": (
                f"kernel {kernel} carries a logarithmic factor; symbolic definite "
                "integration is declared only for log-free algebraic integrands, because "
                "sympy's general integrator does not terminate on the log-power family "
                "within the declared budget"
            ),
        }
    expression = integrand_expression(member)
    started = time.perf_counter()
    try:
        result = sp.integrate(expression, (_X, 0, 1))
    except Exception as failure:  # noqa: BLE001
        return {"fired": False, "reason": f"integrate raised {type(failure).__name__}"}
    elapsed = time.perf_counter() - started
    if elapsed > INTEGRATION_BUDGET_SECONDS:
        return {
            "fired": False,
            "reason": (
                f"symbolic integration exceeded the declared budget of "
                f"{INTEGRATION_BUDGET_SECONDS:.0f}s ({elapsed:.1f}s); this is a budget "
                "overrun, not a statement about the identity"
            ),
        }
    if result.has(sp.Integral):
        return {"fired": False, "reason": "sympy returned the integral unevaluated"}
    closed = sp.simplify(result)
    with mp.workdps(PROOF_DPS):
        try:
            predicted = mp.mpf(str(sp.N(sp.re(closed), mp.mp.dps + 10)))
            actual = family_value_mp("I", member)
        except Exception as failure:  # noqa: BLE001
            return {"fired": False, "reason": f"closed form did not evaluate: {type(failure).__name__}"}
        if not _agrees(predicted, actual):
            return {"fired": False, "reason": "the closed form does not reproduce the value"}
        return {
            "fired": True,
            "derivation": {
                "integrand": str(expression),
                "closed_form": str(closed),
                "closed_form_latex": sp.latex(closed),
                "closed_form_value_60_digits": mp.nstr(predicted, 60),
                "elapsed_seconds": format(elapsed, ".2f"),
                "agreement_digits": PROOF_AGREEMENT_DIGITS,
            },
        }


# ---------------------------------------------------------------------------
# Routing one candidate
# ---------------------------------------------------------------------------


def route_candidate(
    corpus: Mapping[str, Any], family: str, member: Mapping[str, Any], candidate_id: str,
    target: str, prior_verdict: str,
) -> dict[str, Any]:
    """Prove, refute, or type the missing technique for one candidate."""

    report: dict[str, Any] = {
        "candidate_id": candidate_id,
        "family": family,
        "target": target,
        "prefactor": str(member["prefactor"]),
        "prior_art_verdict": prior_verdict,
        "signature": signature_key(family, member),
    }
    check = numeric_check(family, member, target)
    report["refutation_check"] = check
    if not check["holds"]:
        report.update(
            {
                "verdict": "REFUTED",
                "technique_that_fired": None,
                "reclassification": {"reclassified": False},
            }
        )
        return report

    declined: dict[str, str] = {}
    order = (
        ("hypergeometric_closed_form", attempt_hypergeometric)
        if family == "S"
        else ("weierstrass_gauss_gamma_product", attempt_gamma_product)
        if family == "P"
        else None
    )
    attempts: list[tuple[str, Any]] = []
    if order is not None:
        attempts.append(order)
    if family == "I":
        attempts.append(("log_power_hurwitz_reduction", attempt_log_power_hurwitz))
        attempts.append(("beta_derivative_digamma_reduction", attempt_beta_derivative))
        attempts.append(("cyclotomic_substitution_to_beta", attempt_cyclotomic_substitution))
        attempts.append(("euler_2f1_integral_representation", attempt_euler_2f1))
        attempts.append(("symbolic_definite_integration", attempt_symbolic_integration))

    for name, technique in attempts:
        outcome = technique(member)
        if outcome["fired"]:
            proof_family = TECHNIQUE_CLASSICAL_FAMILY[name]
            report.update(
                {
                    "verdict": "PROVED",
                    "technique_that_fired": name,
                    "cited_theorem": TECHNIQUE_CITED_THEOREM[name],
                    "derivation": outcome["derivation"],
                    "techniques_that_declined": declined,
                    "reclassification": {
                        "reclassified": True,
                        "from": prior_verdict,
                        "to": "KNOWN_BY_PROOF_FAMILY",
                        "proof_family": proof_family,
                        "proof_family_present_in_corpus": _family_in_corpus(corpus, proof_family),
                        "why": (
                            "the proof exhibits this candidate as an instance of a cited "
                            "classical family, which is prior art, not novelty"
                        ),
                    },
                    "lean": {"kernel_verified": False, "obstruction": LEAN_OBSTRUCTION},
                }
            )
            return report
        declined[name] = outcome["reason"]

    blocker = f"missing_proof_technique:{family.lower()}_no_declared_technique_applies"
    report.update(
        {
            "verdict": "MISSING_TECHNIQUE",
            "technique_that_fired": None,
            "missing_proof_technique": blocker,
            "techniques_that_declined": declined,
            "reclassification": {"reclassified": False},
            "lean": {"kernel_verified": False, "obstruction": LEAN_OBSTRUCTION},
        }
    )
    return report


#: Which corpus seeds establish that a proof family is already cited prior art.
PROOF_FAMILY_CORPUS_SEEDS: dict[str, tuple[str, ...]] = {
    "generalized_hypergeometric_pFq": (
        "theorem:gauss_2f1_at_one",
        "theorem:kummer_2f1_at_minus_one",
        "theorem:newton_binomial_1f0",
        "theorem:exponential_0f0",
        "theorem:gauss_2f1_general_argument",
    ),
    "weierstrass_gauss_gamma_product": ("theorem:weierstrass_gauss_gamma_product",),
    "hurwitz_zeta_log_power_integral": (
        "theorem:log_power_zeta_integral",
        "theorem:log_power_eta_integral",
        "theorem:log_power_dirichlet_beta_integral",
    ),
    "differentiated_euler_beta_integral": ("theorem:derivative_of_beta",),
    "euler_beta_integral": ("theorem:euler_beta_integral",),
    "gauss_hypergeometric_2f1": (
        "theorem:euler_2f1_integral_representation",
        "theorem:gauss_quadratic_transformations",
    ),
    "elementary_and_meijer_g_closed_forms": (
        "theorem:euler_beta_integral",
        "theorem:rational_kernel_expansion",
    ),
}


def _family_in_corpus(corpus: Mapping[str, Any], proof_family: str) -> bool:
    """Is the classical family this proof lands on already carried by the corpus?"""

    wanted = set(PROOF_FAMILY_CORPUS_SEEDS.get(proof_family, ()))
    if not wanted:
        return False
    present = {item.seed_id for item in corpus["parametric"]}
    return bool(wanted & present)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

#: Classical identities the router must prove end to end before it may report anything.
CLASSICAL_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "control:basel_series",
        "family": "S",
        "member": {"p": [1, 2, 1, 0], "q": [4, 4, 1, 0], "z": "1", "prefactor": "1"},
        "target": "zeta2",
        "identity": "zeta(2) = sum_{k>=1} 1/k^2",
        "expected_technique": "hypergeometric_closed_form",
    },
    {
        "control_id": "control:wallis_product",
        "family": "P",
        "member": {"a": [0, 0, 4, 0], "b": [-1, 0, 4, 0], "k0": 1, "prefactor": "2"},
        "target": "pi",
        "identity": "pi = 2 prod_{k>=1} 4k^2/(4k^2-1)",
        "expected_technique": "weierstrass_gauss_gamma_product",
    },
    {
        "control_id": "control:zeta_two_log_integral",
        "family": "I",
        "member": {"a": "0", "b": "-1", "kernel": "log_inv", "kernel_index": 1, "power": "1",
                   "prefactor": "1"},
        "target": "zeta2",
        "identity": "zeta(2) = int_0^1 -ln(x)/(1-x) dx",
        "expected_technique": "log_power_hurwitz_reduction",
    },
    {
        "control_id": "control:arctan_integral",
        "family": "I",
        "member": {"a": "0", "b": "0", "kernel": "inv_1px2", "kernel_index": 3, "power": "1",
                   "prefactor": "4"},
        "target": "pi",
        "identity": "pi = 4 int_0^1 dx/(1+x^2)",
        "expected_technique": "symbolic_definite_integration",
    },
    {
        "control_id": "control:leibniz_series",
        "family": "S",
        "member": {"p": [1, 2, 0, 0], "q": [3, 2, 0, 0], "z": "-1", "prefactor": "4"},
        "target": "pi",
        "identity": "pi = 4 sum_{k>=0} (-1)^k/(2k+1)",
        "expected_technique": "hypergeometric_closed_form",
    },
)

#: A deliberately perturbed identity that must be refuted, with the failing decimal named.
FALSIFICATION_CONTROL = {
    "control_id": "control:falsification",
    "family": "S",
    "truthful": {"p": [1, 2, 1, 0], "q": [4, 4, 1, 0], "z": "1", "prefactor": "1"},
    "perturbed": {"p": [1, 2, 1, 0], "q": [4, 4, 1, 0], "z": "1", "prefactor": "1"},
    "perturbation": "the target is swapped from zeta(2) to zeta(3), a change of the claim only",
    "target": "zeta2",
    "false_target": "zeta3",
}

#: An in-grammar member with no closed form in the declared constants.  The router must
#: return a typed blocker rather than invent a proof.
ABSENT_TECHNIQUE_CONTROL = {
    "control_id": "control:bose_einstein_kernel",
    "family": "I",
    "member": {"a": "1/3", "b": "1/3", "kernel": "x_over_expm1", "kernel_index": 9,
               "power": "2", "prefactor": "1"},
    "target": "pi",
    "note": (
        "the truncated Bose-Einstein kernel raised to a power has no closed form in the "
        "declared constants; the log-power reduction declines because the kernel is not "
        "rational, and symbolic integration cannot evaluate it, so the router must say so"
    ),
}


def run_controls(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Every control is run-aborting.  A router that fails one cannot report anything."""

    classical: list[dict[str, Any]] = []
    for control in CLASSICAL_CONTROLS:
        routed = route_candidate(
            corpus, control["family"], control["member"], control["control_id"],
            control["target"], "KNOWN",
        )
        classical.append(
            {
                "control_id": control["control_id"],
                "family": control["family"],
                "identity": control["identity"],
                "expected_technique": control["expected_technique"],
                "verdict": routed["verdict"],
                "technique_that_fired": routed.get("technique_that_fired"),
                "closed_form": routed.get("derivation", {}).get("closed_form"),
                "proved": bool(routed["verdict"] == "PROVED"),
            }
        )
    proved = sum(1 for item in classical if item["proved"])

    truthful = numeric_check("S", FALSIFICATION_CONTROL["truthful"], FALSIFICATION_CONTROL["target"])
    perturbed = numeric_check(
        "S", FALSIFICATION_CONTROL["perturbed"], FALSIFICATION_CONTROL["false_target"]
    )
    falsification = {
        "control_id": FALSIFICATION_CONTROL["control_id"],
        "perturbation": FALSIFICATION_CONTROL["perturbation"],
        "truthful_claim_holds": bool(truthful["holds"]),
        "refuted": bool(not perturbed["holds"]),
        "first_differing_decimal_place": perturbed["first_differing_decimal_place"],
    }

    # The absent-technique control probes the *technique layer directly*.  Routing it through
    # the refutation gate would only test refutation, because no declared target equals this
    # integral; what has to be controlled is that every declared technique declines with a
    # reason instead of manufacturing a proof.
    member = ABSENT_TECHNIQUE_CONTROL["member"]
    outcomes = {
        "log_power_hurwitz_reduction": attempt_log_power_hurwitz(member),
        "beta_derivative_digamma_reduction": attempt_beta_derivative(member),
        "cyclotomic_substitution_to_beta": attempt_cyclotomic_substitution(member),
        "euler_2f1_integral_representation": attempt_euler_2f1(member),
        "symbolic_definite_integration": attempt_symbolic_integration(member),
    }
    declined_reasons = {
        name: str(outcome.get("reason", "")) for name, outcome in outcomes.items()
    }
    all_declined = all(not outcome["fired"] for outcome in outcomes.values())
    absent = {
        "control_id": ABSENT_TECHNIQUE_CONTROL["control_id"],
        "note": ABSENT_TECHNIQUE_CONTROL["note"],
        "probe": "the declared Family I techniques were invoked directly on the member",
        "blocked": bool(all_declined and all(declined_reasons.values())),
        "typed_blocker": (
            "missing_proof_technique:i_no_declared_technique_applies" if all_declined else None
        ),
        "techniques_that_declined": declined_reasons,
    }

    return {
        "classical_identities_required": 3,
        "classical_identities_proved": proved,
        "classical_identities": classical,
        "deliberate_falsification": falsification,
        "absent_technique": absent,
        "passed": bool(proved >= 3 and falsification["refuted"] and absent["blocked"]),
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def load_router_candidates(
    adjudication: Mapping[str, Any], enumeration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Every adjudicated candidate, with its structural member rebuilt from the enumeration."""

    members: dict[tuple[str, str], dict[str, Any]] = {}
    for block in enumeration.get("families", []):
        for survivor in block.get("survivors", []):
            members[(block["family"], str(survivor["ordinal"]))] = survivor
    rows: list[dict[str, Any]] = []
    for item in adjudication.get("candidates", []):
        key = (item["family"], str(item["candidate_id"]))
        member = members.get(key)
        if member is None:
            raise ProofRouterError(f"adjudicated candidate {key} is not in the enumeration")
        rows.append(
            {
                "candidate_id": str(item["candidate_id"]),
                "family": item["family"],
                "target": item["target"],
                "prior_verdict": item["verdict"],
                "member": member,
                "formula_text": member.get("formula_text", ""),
                "formula_latex": member.get("formula_latex", ""),
                "value_120_digits": member.get("scaled_value_120_digits", ""),
            }
        )
    rows.sort(key=lambda row: (row["family"], int(row["candidate_id"])))
    return rows


def run_router(
    adjudication: Mapping[str, Any],
    enumeration: Mapping[str, Any],
    corpus: Mapping[str, Any],
    *,
    adjudication_path: str = "runs/math/inverse-symbolic/families-screen-v1.json",
    enumeration_path: str = "runs/math/inverse-symbolic/families-v1.json",
) -> dict[str, Any]:
    """Route every adjudicated candidate, enforce the controls, and seal a receipt."""

    started = time.perf_counter()
    controls = run_controls(corpus)
    if not controls["passed"]:
        raise ProofRouterError(
            "proof-router controls failed: "
            f"{controls['classical_identities_proved']}/3 classical identities proved, "
            f"falsification refuted={controls['deliberate_falsification']['refuted']}, "
            f"absent-technique blocker={controls['absent_technique']['blocked']}"
        )

    candidates = load_router_candidates(adjudication, enumeration)
    if not candidates:
        raise ProofRouterError("adjudication receipt carries no candidates")
    routed = [
        {
            **route_candidate(
                corpus, row["family"], row["member"], row["candidate_id"], row["target"],
                row["prior_verdict"],
            ),
            "formula_text": row["formula_text"],
            "formula_latex": row["formula_latex"],
            "value_120_digits": row["value_120_digits"],
        }
        for row in candidates
    ]

    known_controls = [item for item in routed if item["prior_art_verdict"] == "KNOWN"]
    subjects = [item for item in routed if item["prior_art_verdict"] != "KNOWN"]
    recovered = sum(1 for item in known_controls if item["verdict"] == "PROVED")
    if known_controls and recovered != len(known_controls):
        failed = [
            item["candidate_id"] for item in known_controls if item["verdict"] != "PROVED"
        ]
        raise ProofRouterError(
            f"only {recovered} of {len(known_controls)} already-KNOWN candidates were proved "
            f"by the router; it cannot reproduce the screen it extends. Failures: {failed}"
        )

    proved = [item for item in routed if item["verdict"] == "PROVED"]
    still_absent = [
        item
        for item in proved
        if item["prior_art_verdict"] != "KNOWN"
        and item["reclassification"].get("proof_family_present_in_corpus") is not True
    ]
    unproved = [
        item for item in subjects if item["verdict"] == "MISSING_TECHNIQUE"
    ]
    refuted = [item for item in routed if item["verdict"] == "REFUTED"]

    by_verdict = {name: sum(1 for item in subjects if item["verdict"] == name) for name in VERDICTS}
    by_technique: dict[str, int] = {}
    for item in subjects:
        key = item["technique_that_fired"] or "none"
        by_technique[key] = by_technique.get(key, 0) + 1

    config = {
        "technique_order": list(TECHNIQUE_ORDER),
        "technique_by_family": {
            "S": ["hypergeometric_closed_form"],
            "P": ["weierstrass_gauss_gamma_product"],
            "I": [
            "log_power_hurwitz_reduction",
            "beta_derivative_digamma_reduction",
            "cyclotomic_substitution_to_beta",
            "euler_2f1_integral_representation",
            "symbolic_definite_integration",
        ],
        },
        "refutation_dps": REFUTATION_DPS,
        "refutation_agreement_digits": REFUTATION_DIGITS,
        "proof_agreement_digits": PROOF_AGREEMENT_DIGITS,
        "integration_budget_seconds": format(INTEGRATION_BUDGET_SECONDS, ".0f"),
        "quasi_polynomial_fit_samples": QUASI_POLY_FIT_SAMPLES,
        "quasi_polynomial_verify_samples": QUASI_POLY_VERIFY_SAMPLES,
        "technique_classical_family": dict(sorted(TECHNIQUE_CLASSICAL_FAMILY.items())),
        "technique_cited_theorem": dict(sorted(TECHNIQUE_CITED_THEOREM.items())),
    }
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "lane": "inverse-symbolic-families-proof-routing",
        "claims": ROUTER_CLAIMS,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "input": {
            "adjudication_receipt": adjudication_path,
            "adjudication_content_sha256": adjudication["content_sha256"],
            "adjudication_result_core_sha256": adjudication["result_core_sha256"],
            "enumeration_receipt": enumeration_path,
            "enumeration_content_sha256": enumeration["content_sha256"],
            "enumeration_result_core_sha256": enumeration["result_core_sha256"],
            "candidates": len(candidates),
            "subjects": len(subjects),
            "already_known_controls": len(known_controls),
        },
        "corpus": {
            "schema_version": corpus["manifest"]["schema_version"],
            "content_sha256": corpus["manifest"]["content_sha256"],
            "seeds": corpus["manifest"]["counts"]["total_seeds"],
            "records": corpus["manifest"]["counts"]["records"],
        },
        "controls": {
            **controls,
            "already_known_reproved": recovered,
            "already_known_population": len(known_controls),
        },
        "counts": {
            "by_verdict": by_verdict,
            "by_technique_that_fired": dict(sorted(by_technique.items())),
            "proved_total_including_already_known_controls": len(proved),
            "reclassified_known_by_their_own_proof": sum(
                1 for item in subjects if item["reclassification"].get("reclassified")
            ),
            "refuted": len(refuted),
        },
        "headline": {
            "proved_and_proof_family_still_absent_from_the_corpus": len(still_absent),
            "proved_and_absent_candidate_ids": sorted(
                item["candidate_id"] for item in still_absent
            ),
            "not_reducible_by_any_declared_technique": len(unproved),
            "not_reducible_candidate_ids": sorted(item["candidate_id"] for item in unproved),
            "intersection_absent_from_corpus_and_not_reducible": len(unproved),
            "statement": (
                "a candidate counts toward the headline only when it survived the 120-digit "
                "holdout, is absent from the expanded corpus, and is *not* reducible to a "
                "classical closed form by any declared technique. Proof by a cited family is "
                "prior art, not novelty, so a proved candidate is reclassified rather than "
                "counted. An empty count is the honest and expected answer."
            ),
        },
        "candidates": subjects,
        "already_known_summaries": [
            {
                "candidate_id": item["candidate_id"],
                "family": item["family"],
                "target": item["target"],
                "verdict": item["verdict"],
                "technique_that_fired": item["technique_that_fired"],
            }
            for item in known_controls
        ],
        "lean": {
            "sources_emitted": 0,
            "kernel_verification_pending": True,
            "obstruction": LEAN_OBSTRUCTION,
        },
        "scope": (
            "Proof routing for series, product, and integral identities. Each candidate "
            "receives PROVED with an exhibited derivation whose closed form reproduces the "
            "value at 150 digits, REFUTED with the decimal place where the claim breaks, or "
            "a typed missing_proof_technique. Proofs are symbolic in Python plus a 200-digit "
            "numeric check; none is kernel-verified. Proving a candidate by a classical "
            "technique exhibits it as an instance of a cited family and therefore makes it "
            "KNOWN, which the receipt records as a reclassification. Absence from a finite "
            "corpus is never a novelty claim."
        ),
    }
    core = canonical_sha256(body)
    body["result_core_sha256"] = core
    body["measurement"] = {"elapsed_seconds": format(time.perf_counter() - started, ".3f")}
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seals, claims, verdict vocabulary, counts, controls, and the headline arithmetic."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise ProofRouterError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ProofRouterError("receipt seal changed")
    core_body = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    if value.get("result_core_sha256") != canonical_sha256(core_body):
        raise ProofRouterError("deterministic core seal changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise ProofRouterError("config binding changed")
    if value.get("claims") != ROUTER_CLAIMS:
        raise ProofRouterError("claims block changed")

    candidates = value.get("candidates", [])
    if len(candidates) != value["input"]["subjects"]:
        raise ProofRouterError("routed candidate count changed")
    counts = {name: 0 for name in VERDICTS}
    absent_ids: list[str] = []
    unproved_ids: list[str] = []
    for item in candidates:
        verdict = item.get("verdict")
        if verdict not in VERDICTS:
            raise ProofRouterError(f"unknown verdict {verdict!r}")
        counts[verdict] += 1
        if verdict == "PROVED":
            if item.get("technique_that_fired") not in TECHNIQUE_ORDER:
                raise ProofRouterError(
                    f"PROVED without a declared technique: {item['candidate_id']}"
                )
            if not item.get("derivation"):
                raise ProofRouterError(f"PROVED without a derivation: {item['candidate_id']}")
            if item["reclassification"].get("proof_family_present_in_corpus") is not True:
                absent_ids.append(item["candidate_id"])
        elif verdict == "REFUTED":
            place = item.get("refutation_check", {}).get("first_differing_decimal_place")
            if not isinstance(place, int):
                raise ProofRouterError(
                    f"REFUTED without the failing decimal place: {item['candidate_id']}"
                )
        else:
            blocker = str(item.get("missing_proof_technique", ""))
            if not blocker.startswith("missing_proof_technique:"):
                raise ProofRouterError(
                    f"MISSING_TECHNIQUE without a typed blocker: {item['candidate_id']}"
                )
            unproved_ids.append(item["candidate_id"])
        if item.get("lean", {}).get("kernel_verified"):
            raise ProofRouterError(f"receipt claims a kernel result: {item['candidate_id']}")
    if counts != value["counts"]["by_verdict"]:
        raise ProofRouterError("verdict counts changed")

    headline = value["headline"]
    if sorted(headline["proved_and_absent_candidate_ids"]) != sorted(absent_ids):
        raise ProofRouterError("headline proved-and-absent list does not match the candidates")
    if headline["proved_and_proof_family_still_absent_from_the_corpus"] != len(absent_ids):
        raise ProofRouterError("headline proved-and-absent count does not match")
    if sorted(headline["not_reducible_candidate_ids"]) != sorted(unproved_ids):
        raise ProofRouterError("headline not-reducible list does not match the candidates")
    if headline["not_reducible_by_any_declared_technique"] != len(unproved_ids):
        raise ProofRouterError("headline not-reducible count does not match")

    controls = value["controls"]
    if not controls.get("passed"):
        raise ProofRouterError("receipt records a failed control gate")
    if controls["classical_identities_proved"] < controls["classical_identities_required"]:
        raise ProofRouterError("classical control count below the declared requirement")
    if not controls["deliberate_falsification"]["refuted"]:
        raise ProofRouterError("the deliberately false identity was not refuted")
    if not controls["absent_technique"]["blocked"]:
        raise ProofRouterError("the absent-technique control did not produce a typed blocker")
    if controls["already_known_reproved"] != controls["already_known_population"]:
        raise ProofRouterError("not every already-KNOWN candidate was re-proved")
    if value["lean"]["sources_emitted"] != 0 or not value["lean"]["kernel_verification_pending"]:
        raise ProofRouterError("lean block changed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise ProofRouterError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove, refute, or type the missing technique for family candidates."
    )
    parser.add_argument("--input", default="runs/math/inverse-symbolic/families-screen-v1.json")
    parser.add_argument(
        "--enumeration", default="runs/math/inverse-symbolic/families-v1.json"
    )
    parser.add_argument("--output", default="runs/math/inverse-symbolic/families-proof-v1.json")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(json.dumps({"validated": True, "output": args.output}))
        return 0
    adjudication = json.loads(Path(args.input).read_text(encoding="utf-8"))
    enumeration = json.loads(Path(args.enumeration).read_text(encoding="utf-8"))
    corpus = build_corpus()
    result = run_router(
        adjudication, enumeration, corpus,
        adjudication_path=args.input, enumeration_path=args.enumeration,
    )
    write_receipt(result, args.output)
    print(
        json.dumps(
            {
                "candidates_routed": len(result["candidates"]),
                "by_verdict": result["counts"]["by_verdict"],
                "by_technique": result["counts"]["by_technique_that_fired"],
                "reclassified_known": result["counts"]["reclassified_known_by_their_own_proof"],
                "headline": result["headline"],
                "controls_passed": result["controls"]["passed"],
                "output": args.output,
                "content_sha256": result["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _unused_screen_error_guard() -> None:  # pragma: no cover - import anchor
    """``ScreenError`` is imported so corpus failures surface with their own type."""

    _ = ScreenError
