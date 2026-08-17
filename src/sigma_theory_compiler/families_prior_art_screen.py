"""Adjudicate series, product, and integral survivors against a prior-art corpus (DG3).

:mod:`sigma_theory_compiler.inverse_symbolic_families` enumerated three declared families at
3.2e9 ordinals and promoted survivors through a 120-digit holdout, labelling them against a
small built-in table.  That table is far too small to say anything, so most survivors carry
the empty label ``NOT_IN_BUILTIN_TABLE``.  This module replaces that label with a real
adjudication, using **exact** tests in a fixed order and recording which one fired -- the
same discipline :mod:`sigma_theory_compiler.cf_prior_art_screen` applies to continued
fractions, transported to the three new families:

1. ``exact_signature_match`` -- the candidate's exact class invariant is literally a corpus
   record's.  The invariant is chosen per family so that *equality of invariant implies
   equality of the object*, which makes this a one-step decision rather than a search:

   * Family S: the generalized hypergeometric signature ``(upper multiset, lower multiset,
     argument)``.  A term ratio that is a rational function of the index determines the
     series, and the signature determines the ratio.
   * Family P: the Gamma signature ``(sorted k0 - beta_j, sorted k0 - alpha_i)``.  The
     convergent members have the closed form ``prod Gamma(k0-beta)/prod Gamma(k0-alpha)``,
     so the signature determines the value.
   * Family I: the reduced shape ``(a, b, kernel, power)`` after the declared grammar
     degeneracies are collapsed.

2. ``transformation_orbit_match`` -- a declared transformation group carries the candidate
   onto a corpus record.  Every chain is *exhibited and re-verified numerically at 60
   digits*, never asserted.

3. ``parametric_theorem_match`` -- the candidate is an instance of a cited *parametric*
   classical theorem (Gauss's ``2F1(1)``, Kummer's ``2F1(-1)``, Newton's binomial series,
   Euler's Beta integral, the Gamma-reflection formula, the ``Gamma(s)zeta(s)`` and
   ``Gamma(s)eta(s)`` log-power integrals, the Weierstrass-Gauss Gamma product, ...).  The
   theorem's own closed form is instantiated at the candidate's parameters, evaluated to 60
   digits and required to reproduce the candidate's value.  A theorem that *predicts* the
   number is the strongest form of prior art there is.

4. ``value_match_without_structure`` -- a corpus record has the same value but no structural
   connection exists.  This is ``INCONCLUSIVE_VALUE_MATCH``, never ``KNOWN``.

5. ``NOT_FOUND_IN_CORPUS`` -- nothing fired.  This is absence from a finite corpus.  It is
   not novelty, and the receipt says so in a sealed claim.

**Controls are run-aborting.**  Every survivor the enumeration already labelled
``KNOWN_REDISCOVERED`` is screened too, and the run fails below a declared recovery rate: a
screen that cannot recover known formulas is not fit to report an absence.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from .inverse_symbolic_families import (
    INTEGRAL_KERNELS,
    canonical_form,
    constant_value,
    exact_roots,
    family_value_mp,
    series_hypergeometric_parameters,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-inverse-symbolic-families-adjudication-1.0"

VERDICTS = ("KNOWN", "INCONCLUSIVE_VALUE_MATCH", "NOT_FOUND_IN_CORPUS")

TEST_ORDER = (
    "exact_signature_match",
    "transformation_orbit_match",
    "parametric_theorem_match",
    "value_match_without_structure",
    "no_test_fired",
)

#: Working precision for every structural re-verification in this module.
SCREEN_DPS = 60
#: A chain or a theorem instantiation must reproduce the candidate's value to this many
#: digits before it may be reported.
SCREEN_AGREEMENT_DIGITS = 45
#: Value-equality digits for the (never sufficient) value test.
VALUE_MATCH_DIGITS = 45
#: Digits stored per corpus record.  Strictly more than :data:`VALUE_MATCH_DIGITS` so the
#: strict comparison has guard digits and a true match cannot fail on the stored rounding.
VALUE_STORE_DIGITS = 55

CONTROL_RECOVERY_THRESHOLD = Fraction(95, 100)

SCREEN_CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "external_fetch_performed": False,
    "human_review_required_before_any_novelty_claim": True,
    "value_match_alone_is_not_membership": True,
    "every_KNOWN_carries_a_reverified_justification": True,
}

CITATION_CONFIDENCES = (
    "pinned_identity",
    "section_reference",
    "family_theorem",
    "elementary_derivation",
)


class ScreenError(ValueError):
    """Raised on malformed input, a failed control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Corpus datatypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Citation:
    """Citation metadata.  ``confidence`` is explicit and never optimistic."""

    author: str
    year: str
    reference: str
    confidence: str
    note: str

    def as_json(self) -> dict[str, str]:
        return {
            "author": self.author,
            "year": self.year,
            "reference": self.reference,
            "confidence": self.confidence,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ConcreteSeed:
    """A cited classical identity that lives inside one declared family's grammar."""

    seed_id: str
    family: str
    member: dict[str, Any]
    value_expr: str
    citation: Citation
    validity_domain: str


@dataclass(frozen=True, slots=True)
class ParametricSeed:
    """A cited classical *theorem* covering a variety of family members.

    ``matcher`` receives the candidate's structural data and returns either ``None`` (the
    theorem does not apply) or a dict describing the instantiation: the bound parameters and
    a ``closed_form`` callable evaluated at the current mpmath precision.  The screen then
    checks that the theorem's own prediction reproduces the candidate's value, so a match is
    a *checked* match.
    """

    seed_id: str
    family: str
    statement: str
    citation: Citation
    validity_domain: str
    matcher: Callable[[Mapping[str, Any]], dict[str, Any] | None] = field(repr=False)


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """One stored record with full provenance back to a cited seed."""

    record_id: str
    kind: str
    family: str
    seed_id: str
    parent_id: str | None
    depth: int
    transform: tuple[tuple[str, str], ...]
    signature: str | None
    value: str
    value_expr: str
    citation: Citation
    validity_domain: str

    def as_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "kind": self.kind,
            "family": self.family,
            "seed_id": self.seed_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "transform": {name: value for name, value in self.transform},
            "signature": self.signature,
            "value": self.value,
            "value_expr": self.value_expr,
            "citation": self.citation.as_json(),
            "validity_domain": self.validity_domain,
        }


# ---------------------------------------------------------------------------
# Exact class invariants
# ---------------------------------------------------------------------------


def _sig(expression: sp.Expr) -> str:
    return sp.srepr(sp.nsimplify(sp.expand(expression)))


def series_signature(member: Mapping[str, Any]) -> dict[str, Any]:
    """The exact ``pFq`` signature of a Family S member, upper/lower cancelled.

    A parameter appearing in both lists cancels exactly, which is a genuine reduction of the
    object and not a transformation, so it is folded into the invariant.
    """

    parameters = series_hypergeometric_parameters(
        member["p"], member["q"], Fraction(member["z"])
    )
    if parameters is None:
        return {"upper": [], "lower": [], "argument": "0", "degenerate": True}
    upper, lower, argument = parameters
    remaining_lower = list(lower)
    kept_upper: list[sp.Expr] = []
    for item in upper:
        for index, other in enumerate(remaining_lower):
            if sp.simplify(item - other) == 0:
                remaining_lower.pop(index)
                break
        else:
            kept_upper.append(item)
    return {
        "upper": sorted(_sig(item) for item in kept_upper),
        "lower": sorted(_sig(item) for item in remaining_lower),
        "argument": _sig(argument),
        "degenerate": False,
    }


def series_signature_key(member: Mapping[str, Any]) -> str:
    signature = series_signature(member)
    return (
        "S|" + ";".join(signature["upper"]) + "|" + ";".join(signature["lower"])
        + "|" + signature["argument"]
    )


def product_signature(member: Mapping[str, Any]) -> dict[str, Any]:
    """The exact Gamma signature of a Family P member, common factors cancelled."""

    k0 = sp.Integer(int(member["k0"]))
    _lead_a, roots_a = exact_roots(member["a"])
    _lead_b, roots_b = exact_roots(member["b"])
    numerator = [sp.simplify(k0 - root) for root in roots_a]
    denominator = [sp.simplify(k0 - root) for root in roots_b]
    remaining = list(denominator)
    kept: list[sp.Expr] = []
    for item in numerator:
        for index, other in enumerate(remaining):
            if sp.simplify(item - other) == 0:
                remaining.pop(index)
                break
        else:
            kept.append(item)
    return {
        "gamma_denominator_arguments": sorted(_sig(item) for item in kept),
        "gamma_numerator_arguments": sorted(_sig(item) for item in remaining),
    }


def product_signature_key(member: Mapping[str, Any]) -> str:
    signature = product_signature(member)
    return (
        "P|" + ";".join(signature["gamma_numerator_arguments"])
        + "|" + ";".join(signature["gamma_denominator_arguments"])
    )


def integral_signature_key(member: Mapping[str, Any]) -> str:
    form = canonical_form("I", member)
    return f"I|{form['effective_a']}|{form['effective_b']}|{form['effective_kernel']}|{form['effective_power']}"


def signature_key(family: str, member: Mapping[str, Any]) -> str:
    """The family's exact class invariant, without the rational prefactor."""

    if family == "S":
        return series_signature_key(member)
    if family == "P":
        return product_signature_key(member)
    return integral_signature_key(member)


# ---------------------------------------------------------------------------
# Seed catalogue
# ---------------------------------------------------------------------------


def _c(author: str, year: str, reference: str, confidence: str, note: str) -> Citation:
    if confidence not in CITATION_CONFIDENCES:
        raise ScreenError(f"undeclared citation confidence: {confidence}")
    return Citation(author, year, reference, confidence, note)


def _series(p: Sequence[int], q: Sequence[int], z: str, prefactor: str = "1") -> dict[str, Any]:
    return {"p": list(p), "q": list(q), "z": z, "prefactor": prefactor}


def _product(a: Sequence[int], b: Sequence[int], k0: int, prefactor: str = "1") -> dict[str, Any]:
    return {"a": list(a), "b": list(b), "k0": k0, "prefactor": prefactor}


def _integral(a: str, b: str, kernel: str, power: str, prefactor: str = "1") -> dict[str, Any]:
    index = next(item["index"] for item in INTEGRAL_KERNELS if item["name"] == kernel)
    return {
        "a": a,
        "b": b,
        "kernel": kernel,
        "kernel_index": index,
        "power": power,
        "prefactor": prefactor,
    }


def concrete_seeds() -> list[ConcreteSeed]:
    """Independently encoded classical identities inside the three declared grammars."""

    return [
        # ---- Family S: series -------------------------------------------------
        ConcreteSeed(
            "series:euler_exponential_at_one", "S",
            _series([1, 0, 0, 0], [1, 1, 0, 0], "1"),
            "e = sum_{k>=0} 1/k!",
            _c("Euler", "1748", "Introductio in analysin infinitorum, I, ch. VII",
               "pinned_identity", "the defining exponential series at argument 1"),
            "convergent for every argument",
        ),
        ConcreteSeed(
            "series:euler_exponential_at_minus_one", "S",
            _series([-1, 0, 0, 0], [1, 1, 0, 0], "1"),
            "1/e = sum_{k>=0} (-1)^k/k!",
            _c("Euler", "1748", "Introductio in analysin infinitorum, I, ch. VII",
               "pinned_identity", "the exponential series at argument -1"),
            "convergent for every argument",
        ),
        ConcreteSeed(
            "series:basel_zeta_two", "S",
            _series([1, 2, 1, 0], [4, 4, 1, 0], "1"),
            "zeta(2) = sum_{k>=1} 1/k^2 = pi^2/6",
            _c("Euler", "1735", "De summis serierum reciprocarum, E41",
               "pinned_identity", "the Basel problem"),
            "convergent",
        ),
        ConcreteSeed(
            "series:zeta_three", "S",
            _series([1, 3, 3, 1], [8, 12, 6, 1], "1"),
            "zeta(3) = sum_{k>=1} 1/k^3",
            _c("Riemann", "1859", "DLMF 25.2.1, the Dirichlet series of zeta",
               "pinned_identity", "Apery's constant as the defining series"),
            "convergent for Re(s) > 1",
        ),
        ConcreteSeed(
            "series:gregory_leibniz_pi_over_four", "S",
            _series([1, 2, 0, 0], [3, 2, 0, 0], "-1"),
            "pi/4 = sum_{k>=0} (-1)^k/(2k+1)",
            _c("Gregory and Leibniz", "1671/1674", "DLMF 4.24.3, arctan series at z = 1",
               "pinned_identity", "the arctangent series evaluated at 1"),
            "conditionally convergent at the boundary",
        ),
        ConcreteSeed(
            "series:mercator_alternating_harmonic", "S",
            _series([1, 1, 0, 0], [2, 1, 0, 0], "-1"),
            "ln 2 = sum_{k>=0} (-1)^k/(k+1)",
            _c("Mercator", "1668", "Logarithmotechnia; DLMF 4.6.1 at z = 1",
               "pinned_identity", "the alternating harmonic series"),
            "conditionally convergent at the boundary",
        ),
        ConcreteSeed(
            "series:catalan_defining", "S",
            _series([1, 4, 4, 0], [9, 12, 4, 0], "-1"),
            "G = sum_{k>=0} (-1)^k/(2k+1)^2",
            _c("Catalan", "1865", "DLMF 25.11.40, the Dirichlet beta function at 2",
               "pinned_identity", "the defining series of Catalan's constant"),
            "convergent",
        ),
        ConcreteSeed(
            "series:newton_binomial_sqrt_two", "S",
            _series([1, 2, 0, 0], [2, 2, 0, 0], "1/2"),
            "sqrt(2) = sum_{k>=0} binom(-1/2, k)(-1/2)^k = (1 - 1/2)^(-1/2)",
            _c("Newton", "1665", "the binomial series; DLMF 15.4.6, 1F0(a;;z) = (1-z)^(-a)",
               "pinned_identity", "the binomial series for (1-z)^(-1/2) at z = 1/2"),
            "|z| < 1",
        ),
        ConcreteSeed(
            "series:geometric_at_half", "S",
            _series([1, 0, 0, 0], [1, 0, 0, 0], "1/2"),
            "2 = sum_{k>=0} 2^-k",
            _c("Euclid", "-300", "Elements IX.35, the geometric progression",
               "elementary_derivation", "the geometric series at ratio 1/2"),
            "|ratio| < 1",
        ),
        ConcreteSeed(
            "series:eta_two_alternating", "S",
            _series([1, 2, 1, 0], [4, 4, 1, 0], "-1"),
            "eta(2) = sum_{k>=1} (-1)^(k-1)/k^2 = pi^2/12",
            _c("Euler", "1749", "DLMF 25.2.3, the Dirichlet eta function; eta(2) = zeta(2)/2",
               "pinned_identity", "the alternating Basel series"),
            "convergent",
        ),
        ConcreteSeed(
            "series:eta_three_alternating", "S",
            _series([1, 3, 3, 1], [8, 12, 6, 1], "-1"),
            "eta(3) = sum_{k>=1} (-1)^(k-1)/k^3 = 3 zeta(3)/4",
            _c("Euler", "1749", "DLMF 25.2.3, eta(s) = (1 - 2^(1-s)) zeta(s)",
               "family_theorem", "the alternating cubic series"),
            "convergent",
        ),
        ConcreteSeed(
            "series:arctan_half_argument", "S",
            _series([1, 2, 0, 0], [3, 2, 0, 0], "1/2"),
            "arctan(x)/x at x^2 = -1/2; the arctan series with a rational argument",
            _c("Gregory", "1671", "DLMF 4.24.3, the arctangent series",
               "section_reference", "the arctangent series at a rational argument"),
            "|z| <= 1",
        ),
        ConcreteSeed(
            "series:log_series_half", "S",
            _series([1, 1, 0, 0], [2, 1, 0, 0], "1/2"),
            "2 ln 2 = sum_{k>=0} (1/2)^k/(k+1)",
            _c("Mercator", "1668", "DLMF 4.6.1, -ln(1-z)/z = 2F1(1,1;2;z)",
               "pinned_identity", "the logarithmic series at z = 1/2"),
            "|z| < 1 or z = -1",
        ),
        # ---- Family P: products -----------------------------------------------
        ConcreteSeed(
            "product:wallis", "P",
            _product([0, 0, 4, 0], [-1, 0, 4, 0], 1),
            "pi/2 = prod_{k>=1} 4k^2/(4k^2 - 1)",
            _c("Wallis", "1656", "Arithmetica infinitorum, Prop. 191",
               "pinned_identity", "the Wallis product"),
            "convergent",
        ),
        ConcreteSeed(
            "product:euler_sine_at_half", "P",
            _product([-1, 0, 4, 0], [0, 0, 4, 0], 1),
            "2/pi = prod_{k>=1} (1 - 1/(4k^2)) = sin(pi/2)/(pi/2)",
            _c("Euler", "1734", "De summis serierum reciprocarum, E41; DLMF 4.22.2",
               "pinned_identity", "the sine product at x = 1/2"),
            "convergent for every x",
        ),
        ConcreteSeed(
            "product:euler_sinh_at_one", "P",
            _product([1, 0, 1, 0], [0, 0, 1, 0], 1),
            "sinh(pi)/pi = prod_{k>=1} (1 + 1/k^2)",
            _c("Euler", "1748", "Introductio, I, ch. IX; DLMF 4.36.1",
               "pinned_identity", "the hyperbolic sine product at x = 1"),
            "convergent for every x",
        ),
        ConcreteSeed(
            "product:telescoping_one_half", "P",
            _product([-1, 0, 1, 0], [0, 0, 1, 0], 2),
            "1/2 = prod_{k>=2} (1 - 1/k^2)",
            _c("Euler", "1748", "Introductio, I, ch. IX; elementary telescoping",
               "elementary_derivation", "the telescoping product of (k-1)(k+1)/k^2"),
            "convergent",
        ),
        ConcreteSeed(
            "product:wallis_cubic_grammar_form", "P",
            _product([0, 0, 0, 4], [0, -1, 0, 4], 1),
            "pi/2 = prod_{k>=1} 4k^3/(4k^3 - k)",
            _c("Wallis", "1656", "Arithmetica infinitorum, Prop. 191",
               "family_theorem", "the Wallis product with a cancelling common factor k"),
            "convergent",
        ),
        ConcreteSeed(
            "product:one_third_shifted", "P",
            _product([0, 3, 1, 0], [2, 3, 1, 0], 1),
            "1/3 = prod_{k>=1} k(k+3)/((k+1)(k+2))",
            _c("Gauss", "1813", "Disquisitiones generales, the Gamma product; W&W 12.13",
               "elementary_derivation", "an integer-shift Gamma ratio, telescoping"),
            "convergent when the root sums agree",
        ),
        ConcreteSeed(
            "product:quarter_shift_gamma_ratio", "P",
            _product([0, 2, 1, 0], [1, 2, 1, 0], 1),
            "1/2 = prod_{k>=1} k(k+2)/((k+1)^2)",
            _c("Gauss", "1813", "Disquisitiones generales; W&W 12.13",
               "elementary_derivation", "a telescoping Gamma ratio"),
            "convergent when the root sums agree",
        ),
        # ---- Family I: integrals ----------------------------------------------
        ConcreteSeed(
            "integral:beta_half_half", "I",
            _integral("-1/2", "-1/2", "one", "0"),
            "pi = B(1/2, 1/2) = int_0^1 x^(-1/2)(1-x)^(-1/2) dx",
            _c("Euler", "1730", "DLMF 5.12.1, the Beta integral; 5.5.3 reflection",
               "pinned_identity", "the Beta integral at (1/2, 1/2)"),
            "Re(a) > 0 and Re(b) > 0",
        ),
        ConcreteSeed(
            "integral:beta_unit", "I",
            _integral("0", "0", "one", "0"),
            "1 = B(1, 1) = int_0^1 dx",
            _c("Euler", "1730", "DLMF 5.12.1, the Beta integral",
               "elementary_derivation", "the trivial Beta instance"),
            "Re(a) > 0 and Re(b) > 0",
        ),
        ConcreteSeed(
            "integral:arctan_quarter_pi", "I",
            _integral("0", "0", "inv_1px2", "1"),
            "pi/4 = int_0^1 dx/(1+x^2)",
            _c("Gregory", "1671", "DLMF 4.24.3 in integral form",
               "pinned_identity", "the arctangent integral on the unit interval"),
            "elementary",
        ),
        ConcreteSeed(
            "integral:log_two", "I",
            _integral("0", "0", "inv_1px", "1"),
            "ln 2 = int_0^1 dx/(1+x)",
            _c("Mercator", "1668", "DLMF 4.6.1 in integral form",
               "pinned_identity", "the logarithmic integral on the unit interval"),
            "elementary",
        ),
        ConcreteSeed(
            "integral:zeta_two_log", "I",
            _integral("0", "-1", "log_inv", "1"),
            "zeta(2) = int_0^1 -ln(x)/(1-x) dx",
            _c("Euler", "1735", "DLMF 25.5.1, int_0^1 (ln(1/x))^(s-1)/(1-x) dx = Gamma(s) zeta(s)",
               "pinned_identity", "the log-power integral representation of zeta at s = 2"),
            "Re(s) > 1",
        ),
        ConcreteSeed(
            "integral:zeta_three_log_squared", "I",
            _integral("0", "-1", "log_inv", "2"),
            "2 zeta(3) = int_0^1 (ln(1/x))^2/(1-x) dx",
            _c("Euler", "1735", "DLMF 25.5.1 at s = 3",
               "pinned_identity", "the log-power integral representation of zeta at s = 3"),
            "Re(s) > 1",
        ),
        ConcreteSeed(
            "integral:eta_two_log", "I",
            _integral("0", "0", "log_inv_over_1px", "1"),
            "pi^2/12 = int_0^1 -ln(x)/(1+x) dx",
            _c("Euler", "1749", "DLMF 25.5.3, the eta-function log-power integral",
               "pinned_identity", "the alternating log-power integral at s = 2"),
            "Re(s) > 0",
        ),
        ConcreteSeed(
            "integral:catalan_log", "I",
            _integral("0", "0", "log_inv_over_1px2", "1"),
            "G = int_0^1 -ln(x)/(1+x^2) dx",
            _c("Catalan", "1865", "DLMF 25.11.40 with the Dirichlet beta integral",
               "pinned_identity", "the log-power integral for the Dirichlet beta at 2"),
            "Re(s) > 0",
        ),
        ConcreteSeed(
            "integral:zeta_two_log_over_1mx", "I",
            _integral("0", "0", "log_inv_over_1mx", "1"),
            "zeta(2) = int_0^1 -ln(x)/(1-x) dx, written with the composite kernel",
            _c("Euler", "1735", "DLMF 25.5.1 at s = 2",
               "pinned_identity", "the same classical integral in the composite kernel slot"),
            "Re(s) > 1",
        ),
        ConcreteSeed(
            "integral:beta_sixth", "I",
            _integral("-1/6", "1/6", "one", "0"),
            "pi/3 = B(5/6, 7/6) = int_0^1 x^(-1/6)(1-x)^(1/6) dx",
            _c("Euler", "1730", "DLMF 5.12.1 with 5.5.3, Gamma reflection at z = 1/6",
               "pinned_identity", "the Beta integral whose reflection gives pi/sin(pi/6)"),
            "Re(a) > 0 and Re(b) > 0",
        ),
        ConcreteSeed(
            "integral:beta_three_halves", "I",
            _integral("1/2", "1/2", "one", "0"),
            "pi/8 = B(3/2, 3/2) = int_0^1 x^(1/2)(1-x)^(1/2) dx",
            _c("Euler", "1730", "DLMF 5.12.1 with 5.5.3",
               "pinned_identity", "the Beta integral at (3/2, 3/2)"),
            "Re(a) > 0 and Re(b) > 0",
        ),
        ConcreteSeed(
            "integral:log_one_over_one_minus_x", "I",
            _integral("0", "0", "log_inv_1mx", "1"),
            "1 = int_0^1 ln(1/(1-x)) dx",
            _c("Euler", "1748", "elementary; the x -> 1-x mirror of int_0^1 ln(1/x) dx = 1",
               "elementary_derivation", "the reflected logarithmic integral"),
            "elementary",
        ),
        ConcreteSeed(
            "series:newton_binomial_sqrt_three", "S",
            _series([1, 2, 0, 0], [-3, -3, 0, 0], "1/2", "2"),
            "sqrt(3) = 2 (1 + 1/3)^(-1/2) = 2 * 1F0(1/2; ; -1/3)",
            _c("Newton", "1665", "the binomial series; DLMF 15.4.6",
               "pinned_identity", "the binomial series for (1-z)^(-1/2) at z = -1/3"),
            "|z| < 1",
        ),
        ConcreteSeed(
            "series:exponential_at_half", "S",
            _series([1, 0, 0, 0], [2, 2, 0, 0], "1"),
            "sqrt(e) = sum_{k>=0} (1/2)^k/k!",
            _c("Euler", "1748", "Introductio in analysin infinitorum, I, ch. VII",
               "pinned_identity", "the exponential series at argument 1/2"),
            "convergent for every argument",
        ),
        ConcreteSeed(
            "series:geometric_at_minus_half", "S",
            _series([1, 0, 0, 0], [-1, 0, 0, 0], "1/2"),
            "2/3 = sum_{k>=0} (-1/2)^k",
            _c("Euclid", "-300", "Elements IX.35, the geometric progression",
               "elementary_derivation", "the geometric series at ratio -1/2"),
            "|ratio| < 1",
        ),
        ConcreteSeed(
            "series:binomial_cube_root", "S",
            _series([1, 3, 0, 0], [3, 3, 0, 0], "1/2"),
            "(1/2)^(-1/3) = 2^(1/3) = 1F0(1/3; ; 1/2)",
            _c("Newton", "1665", "the binomial series; DLMF 15.4.6",
               "pinned_identity", "the binomial series for (1-z)^(-1/3) at z = 1/2"),
            "|z| < 1",
        ),
        ConcreteSeed(
            "product:two_from_k_squared", "P",
            _product([0, 0, 1, 0], [-1, 0, 1, 0], 2),
            "2 = prod_{k>=2} k^2/(k^2 - 1)",
            _c("Euler", "1748", "Introductio, I, ch. IX; the reciprocal telescoping product",
               "elementary_derivation", "the reciprocal of prod (1 - 1/k^2) from k = 2"),
            "convergent",
        ),
        ConcreteSeed(
            "product:three_fifths_shifted", "P",
            _product([0, 3, 1, 0], [2, 3, 1, 0], 3),
            "3/5 = prod_{k>=3} k(k+3)/((k+1)(k+2))",
            _c("Gauss", "1813", "Disquisitiones generales; W&W 12.13, the Gamma product",
               "elementary_derivation", "an integer-shift Gamma ratio at start index 3"),
            "convergent when the root sums agree",
        ),
        ConcreteSeed(
            "integral:beta_quarter", "I",
            _integral("-1/4", "1/4", "one", "0"),
            "pi sqrt(2)/4 = B(3/4, 5/4)",
            _c("Euler", "1730", "DLMF 5.12.1 with 5.5.3, Gamma reflection at z = 1/4",
               "pinned_identity", "the Beta integral whose reflection gives pi/sin(pi/4)"),
            "Re(a) > 0 and Re(b) > 0",
        ),
        ConcreteSeed(
            "integral:beta_third", "I",
            _integral("-1/3", "1/3", "one", "0"),
            "2 pi/(3 sqrt(3)) = B(2/3, 4/3)",
            _c("Euler", "1730", "DLMF 5.12.1 with 5.5.3, Gamma reflection at z = 1/3",
               "pinned_identity", "the Beta integral whose reflection gives pi/sin(pi/3)"),
            "Re(a) > 0 and Re(b) > 0",
        ),
        ConcreteSeed(
            "integral:monomial_log", "I",
            _integral("1", "0", "log_inv", "1"),
            "1/4 = int_0^1 x ln(1/x) dx = Gamma(2)/2^2",
            _c("Euler", "1730", "DLMF 5.9.1, int_0^1 x^a (ln(1/x))^s dx = Gamma(s+1)/(a+1)^(s+1)",
               "pinned_identity", "the monomial log-power integral"),
            "Re(a) > -1 and Re(s) > -1",
        ),
        ConcreteSeed(
            "integral:half_log_two", "I",
            _integral("1", "0", "inv_1px2", "1"),
            "ln(2)/2 = int_0^1 x/(1+x^2) dx",
            _c("Mercator", "1668", "elementary substitution u = 1 + x^2",
               "elementary_derivation", "the elementary rational integral"),
            "elementary",
        ),
        ConcreteSeed(
            "integral:dirichlet_beta_three", "I",
            _integral("0", "0", "log_inv_over_1px2", "2"),
            "pi^3/16 = int_0^1 (ln(1/x))^2/(1+x^2) dx = Gamma(3) beta(3)",
            _c("Catalan and Dirichlet", "1865/1837", "DLMF 25.11.40 at s = 3",
               "pinned_identity", "the Dirichlet-beta log-power integral at s = 3"),
            "Re(s) > -1",
        ),
        ConcreteSeed(
            "integral:zeta_two_minus_one", "I",
            _integral("1", "-1", "log_inv", "1"),
            "zeta(2) - 1 = int_0^1 x ln(1/x)/(1-x) dx",
            _c("Euler", "1735", "DLMF 25.11.16, the Hurwitz-zeta log-power integral at a = 1",
               "pinned_identity", "the Hurwitz form of the zeta log-power integral"),
            "Re(s) > 1 and Re(a) > -1",
        ),
        ConcreteSeed(
            "integral:bose_einstein_unit_interval", "I",
            _integral("0", "0", "x_over_expm1", "1"),
            "int_0^1 x/(e^x - 1) dx; no closed form in the declared constants",
            _c("Debye", "1912", "DLMF 25.12.11, the Bose-Einstein integral truncated to [0,1]",
               "section_reference", "a declared kernel with no elementary closed form"),
            "the truncated integral is not a polylogarithm value",
        ),
    ]


def parametric_seeds() -> list[ParametricSeed]:
    """Cited classical *theorems* that cover whole varieties of family members."""

    return [
        ParametricSeed(
            "theorem:gauss_2f1_at_one", "S",
            "2F1(a, b; c; 1) = Gamma(c)Gamma(c-a-b)/(Gamma(c-a)Gamma(c-b)) for Re(c-a-b) > 0",
            _c("Gauss", "1813", "Disquisitiones generales circa seriem infinitam; DLMF 15.4.20",
               "pinned_identity", "Gauss's summation theorem"),
            "Re(c - a - b) > 0",
            _match_gauss_2f1,
        ),
        ParametricSeed(
            "theorem:kummer_2f1_at_minus_one", "S",
            "2F1(a, b; 1+a-b; -1) = Gamma(1+a-b)Gamma(1+a/2)/(Gamma(1+a)Gamma(1+a/2-b))",
            _c("Kummer", "1836", "Journal fuer die reine und angewandte Mathematik 15; DLMF 15.4.26",
               "pinned_identity", "Kummer's theorem for the well-poised 2F1 at -1"),
            "Re(b) < 1 for convergence at the boundary",
            _match_kummer_2f1,
        ),
        ParametricSeed(
            "theorem:newton_binomial_1f0", "S",
            "1F0(a; ; z) = (1-z)^(-a)",
            _c("Newton", "1665", "the binomial series; DLMF 15.4.6",
               "pinned_identity", "the binomial theorem for arbitrary exponent"),
            "|z| < 1",
            _match_binomial_1f0,
        ),
        ParametricSeed(
            "theorem:exponential_0f0", "S",
            "0F0( ; ; z) = exp(z)",
            _c("Euler", "1748", "Introductio in analysin infinitorum; DLMF 15.4.5 limit",
               "pinned_identity", "the exponential series"),
            "convergent for every z",
            _match_exponential_0f0,
        ),
        ParametricSeed(
            "theorem:confluent_polynomial_times_exponential", "S",
            "sum_{k>=0} R(k) z^k/k! = Q(z) exp(z) for any polynomial R, with Q polynomial; "
            "equivalently a pFq whose lower parameters are upper parameters shifted down by "
            "positive integers collapses onto the reduced hypergeometric function",
            _c("Rainville", "1960", "Special Functions, ch. 5 (contiguous relations) and ch. 4.1",
               "family_theorem",
               "every lower parameter equal to an upper parameter minus a positive integer "
               "reduces the series to a finite differential-operator image of the reduced "
               "function; with no parameters left that function is exp"),
            "the shift multiset must be non-negative integers",
            _match_polynomial_times_exponential,
        ),
        ParametricSeed(
            "theorem:gauss_2f1_general_argument", "S",
            "2F1(1, b; c; z) and 1F0/2F1 instances with elementary closed forms "
            "(logarithm, arctangent, algebraic)",
            _c("Gauss", "1813", "DLMF 15.4.1-15.4.19, the elementary 2F1 special cases",
               "section_reference", "the tabulated elementary hypergeometric closed forms"),
            "|z| < 1 or the boundary cases named in DLMF 15.4",
            _match_elementary_2f1,
        ),
        ParametricSeed(
            "theorem:dixon_3f2", "S",
            "Dixon's theorem for the well-poised 3F2(a,b,c; 1+a-b, 1+a-c; 1)",
            _c("Dixon", "1902", "Proc. London Math. Soc. 35; DLMF 16.4.4",
               "pinned_identity", "Dixon's summation theorem"),
            "Re(1 + a/2 - b - c) > 0",
            _match_never,
        ),
        ParametricSeed(
            "theorem:saalschutz_3f2", "S",
            "Saalschuetz's theorem for the terminating Saalschuetzian 3F2 at 1",
            _c("Saalschuetz", "1890", "Zeitschrift fuer Mathematik und Physik 35; DLMF 16.4.3",
               "pinned_identity", "Saalschuetz's summation theorem"),
            "terminating and Saalschuetzian",
            _match_never,
        ),
        ParametricSeed(
            "theorem:watson_3f2", "S",
            "Watson's theorem for 3F2(a,b,c; (a+b+1)/2, 2c; 1)",
            _c("Watson", "1925", "Proc. London Math. Soc. 23; DLMF 16.4.6",
               "pinned_identity", "Watson's summation theorem"),
            "Re(2c - a - b) > -1",
            _match_never,
        ),
        ParametricSeed(
            "theorem:whipple_3f2", "S",
            "Whipple's theorem for the well-poised 3F2 at 1",
            _c("Whipple", "1925", "Proc. London Math. Soc. 23; DLMF 16.4.7",
               "pinned_identity", "Whipple's summation theorem"),
            "well-poised",
            _match_never,
        ),
        ParametricSeed(
            "theorem:weierstrass_gauss_gamma_product", "P",
            "prod_{k>=k0} prod_i (k - alpha_i)/prod_j (k - beta_j) = "
            "prod_j Gamma(k0 - beta_j) / prod_i Gamma(k0 - alpha_i) whenever the root sums "
            "agree, which is exactly the condition for convergence",
            _c("Weierstrass and Gauss", "1856/1813",
               "Whittaker and Watson, A Course of Modern Analysis, 12.11-12.13",
               "pinned_identity",
               "the canonical product for 1/Gamma and the Gauss product formula; every "
               "convergent member of the declared product grammar is an instance"),
            "equal degree, equal leading coefficient, equal root sums",
            _match_gamma_product,
        ),
        ParametricSeed(
            "theorem:gamma_reflection", "P",
            "Gamma(z)Gamma(1-z) = pi/sin(pi z)",
            _c("Euler", "1749", "DLMF 5.5.3, the reflection formula",
               "pinned_identity", "the reflection formula that turns a Gamma ratio into pi"),
            "z not an integer",
            _match_never,
        ),
        ParametricSeed(
            "theorem:euler_sine_product", "P",
            "sin(pi x)/(pi x) = prod_{k>=1} (1 - x^2/k^2)",
            _c("Euler", "1734", "DLMF 4.22.2",
               "pinned_identity", "the sine product"),
            "convergent for every x",
            _match_never,
        ),
        ParametricSeed(
            "theorem:euler_beta_integral", "I",
            "int_0^1 x^a (1-x)^b dx = Gamma(a+1)Gamma(b+1)/Gamma(a+b+2)",
            _c("Euler", "1730", "DLMF 5.12.1",
               "pinned_identity", "Euler's Beta integral"),
            "Re(a) > -1 and Re(b) > -1",
            _match_beta_integral,
        ),
        ParametricSeed(
            "theorem:log_power_zeta_integral", "I",
            "int_0^1 x^a (ln(1/x))^s/(1-x) dx = Gamma(s+1) sum_{k>=0} 1/(k+a+1)^(s+1), "
            "which is Gamma(s+1) zeta(s+1) at a = 0 and a Hurwitz zeta otherwise",
            _c("Riemann", "1859", "DLMF 25.5.1 and 25.11.16 (Hurwitz form)",
               "pinned_identity", "the log-power integral representation of zeta"),
            "Re(s) > 0 and Re(a) > -1",
            _match_log_power_zeta,
        ),
        ParametricSeed(
            "theorem:log_power_eta_integral", "I",
            "int_0^1 x^a (ln(1/x))^s/(1+x) dx = Gamma(s+1) sum_{k>=0} (-1)^k/(k+a+1)^(s+1), "
            "which is Gamma(s+1) eta(s+1) at a = 0",
            _c("Euler and Dirichlet", "1749/1837", "DLMF 25.5.3 and 25.2.3",
               "pinned_identity", "the alternating log-power integral"),
            "Re(s) > -1 and Re(a) > -1",
            _match_log_power_eta,
        ),
        ParametricSeed(
            "theorem:log_power_dirichlet_beta_integral", "I",
            "int_0^1 x^a (ln(1/x))^s/(1+x^2) dx expands to a Lerch/Dirichlet-beta sum; at "
            "a = 0 it is Gamma(s+1) beta(s+1) with beta the Dirichlet beta function",
            _c("Catalan and Dirichlet", "1865/1837", "DLMF 25.11.40 and 25.14.1 (Lerch)",
               "pinned_identity", "the Dirichlet-beta log-power integral"),
            "Re(s) > -1 and Re(a) > -1",
            _match_log_power_dirichlet_beta,
        ),
        ParametricSeed(
            "theorem:rational_kernel_expansion", "I",
            "int_0^1 x^a K(x) dx for a declared rational kernel K expands by the geometric "
            "series into a Lerch transcendent / digamma combination in closed form",
            _c("Euler", "1748", "DLMF 25.14.1 (Lerch transcendent), 5.9.16 (digamma integral)",
               "family_theorem",
               "term-by-term integration of the kernel's geometric expansion"),
            "Re(a) > -1",
            _match_never,
        ),
        ParametricSeed(
            "theorem:gamma_reflection_integral", "I",
            "B(z, 1-z) = Gamma(z)Gamma(1-z) = pi/sin(pi z)",
            _c("Euler", "1749", "DLMF 5.5.3 combined with 5.12.1",
               "pinned_identity",
               "the Beta integral at complementary arguments, which is how a rational "
               "exponent pair produces pi"),
            "0 < Re(z) < 1",
            _match_beta_reflection,
        ),
        ParametricSeed(
            "theorem:euler_2f1_integral_representation", "I",
            "int_0^1 x^(alpha-1)(1-x)^(gamma-alpha-1)(1-z x)^(-beta) dx = "
            "B(alpha, gamma-alpha) 2F1(alpha, beta; gamma; z) for Re(gamma) > Re(alpha) > 0",
            _c("Euler", "1769", "DLMF 15.6.1; Erdelyi, Higher Transcendental Functions I, 2.1.10",
               "pinned_identity",
               "Euler's integral representation of the Gauss hypergeometric function; every "
               "log-free member of the declared integral grammar whose kernel is a power of "
               "(1 - z x) is an instance"),
            "Re(gamma) > Re(alpha) > 0 and z outside [1, infinity)",
            _match_never,
        ),
        ParametricSeed(
            "theorem:gauss_quadratic_transformations", "I",
            "the quadratic transformations of 2F1, available exactly when the parameters "
            "satisfy one of c = 2a, c = 2b, c = a+b+1/2, c = a+b-1/2, b = a+1/2 or a+b = 1",
            _c("Gauss and Kummer", "1813/1836", "DLMF 15.8.13-15.8.27; Erdelyi HTF I, 2.11",
               "pinned_identity",
               "the classical quadratic-transformation relations, which supply closed forms "
               "at z = -1 outside the reach of Kummer's theorem"),
            "one of the declared parameter relations holds",
            _match_never,
        ),
        ParametricSeed(
            "theorem:legendre_duplication", "I",
            "Gamma(z)Gamma(z+1/2) = 2^(1-2z) sqrt(pi) Gamma(2z)",
            _c("Legendre", "1809", "DLMF 5.5.5, the duplication formula",
               "pinned_identity", "the duplication formula for half-integer Beta arguments"),
            "z not a non-positive integer",
            _match_never,
        ),
        ParametricSeed(
            "theorem:substitution_x_to_one_minus_x", "I",
            "int_0^1 f(x) dx = int_0^1 f(1-x) dx, which exchanges the two exponents and maps "
            "each declared kernel onto its reflected partner",
            _c("elementary", "-", "change of variable on a finite interval",
               "elementary_derivation", "the reflection substitution"),
            "any integrable f",
            _match_never,
        ),
        ParametricSeed(
            "theorem:derivative_of_beta", "I",
            "int_0^1 x^a (1-x)^b (ln(1/x))^n dx = (-1)^n d^n/da^n B(a+1, b+1), a finite "
            "polynomial in digamma and polygamma values times B(a+1, b+1)",
            _c("Euler and Gauss", "1730/1813", "DLMF 5.12.1 differentiated; 5.15 (polygamma)",
               "pinned_identity",
               "differentiating the Beta integral under the integral sign in the exponent"),
            "Re(a) > -1, Re(b) > -1, n a non-negative integer",
            _match_beta_derivative,
        ),
    ]


# ---------------------------------------------------------------------------
# Parametric theorem matchers
# ---------------------------------------------------------------------------


def _match_never(_data: Mapping[str, Any]) -> None:
    """A seed encoded for the record whose variety this run's candidates never enter."""

    return


def _upper_lower(data: Mapping[str, Any]) -> tuple[list[sp.Expr], list[sp.Expr], sp.Expr] | None:
    signature = data.get("hypergeometric")
    if signature is None:
        return None
    return signature


def _match_exponential_0f0(data: Mapping[str, Any]) -> dict[str, Any] | None:
    parsed = _upper_lower(data)
    if parsed is None:
        return None
    upper, lower, argument = parsed
    if upper or lower:
        return None
    return {
        "instantiation": {"z": str(argument)},
        "closed_form_text": f"exp({argument})",
        "closed_form": lambda: mp.exp(mp.mpf(str(sp.N(argument, mp.mp.dps + 10)))),
    }


def _match_binomial_1f0(data: Mapping[str, Any]) -> dict[str, Any] | None:
    parsed = _upper_lower(data)
    if parsed is None:
        return None
    upper, lower, argument = parsed
    if len(upper) != 1 or lower:
        return None
    a = upper[0]
    if sp.Abs(sp.N(argument)) >= 1:
        return None
    return {
        "instantiation": {"a": str(a), "z": str(argument)},
        "closed_form_text": f"(1 - {argument})**(-({a}))",
        "closed_form": lambda: mp.power(
            mp.mpf(str(sp.N(1 - argument, mp.mp.dps + 10))),
            -mp.mpf(str(sp.N(a, mp.mp.dps + 10))),
        ),
    }


def _gamma_mp(expression: sp.Expr) -> mp.mpf:
    return mp.gamma(mp.mpf(str(sp.N(expression, mp.mp.dps + 10))))


def _match_gauss_2f1(data: Mapping[str, Any]) -> dict[str, Any] | None:
    parsed = _upper_lower(data)
    if parsed is None:
        return None
    upper, lower, argument = parsed
    if len(upper) != 2 or len(lower) != 1 or sp.simplify(argument - 1) != 0:
        return None
    a, b = upper
    c = lower[0]
    if sp.N(c - a - b) <= 0:
        return None
    return {
        "instantiation": {"a": str(a), "b": str(b), "c": str(c), "z": "1"},
        "closed_form_text": "Gamma(c)Gamma(c-a-b)/(Gamma(c-a)Gamma(c-b))",
        "closed_form": lambda: (
            _gamma_mp(c) * _gamma_mp(c - a - b) / (_gamma_mp(c - a) * _gamma_mp(c - b))
        ),
    }


def _match_kummer_2f1(data: Mapping[str, Any]) -> dict[str, Any] | None:
    parsed = _upper_lower(data)
    if parsed is None:
        return None
    upper, lower, argument = parsed
    if len(upper) != 2 or len(lower) != 1 or sp.simplify(argument + 1) != 0:
        return None
    c = lower[0]
    for a, b in ((upper[0], upper[1]), (upper[1], upper[0])):
        if sp.simplify(c - (1 + a - b)) != 0:
            continue
        return {
            "instantiation": {"a": str(a), "b": str(b), "c": str(c), "z": "-1"},
            "closed_form_text": "Gamma(1+a-b)Gamma(1+a/2)/(Gamma(1+a)Gamma(1+a/2-b))",
            "closed_form": (
                lambda a=a, b=b: _gamma_mp(1 + a - b)
                * _gamma_mp(1 + a / 2)
                / (_gamma_mp(1 + a) * _gamma_mp(1 + a / 2 - b))
            ),
        }
    return None


def _match_polynomial_times_exponential(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Every lower parameter is an upper parameter minus a positive integer."""

    parsed = _upper_lower(data)
    if parsed is None:
        return None
    upper, lower, argument = parsed
    if not lower or len(lower) > len(upper):
        return None
    remaining = list(upper)
    shifts: list[int] = []
    for item in lower:
        for index, other in enumerate(remaining):
            difference = sp.nsimplify(sp.simplify(other - item))
            if difference.is_Integer and int(difference) > 0:
                shifts.append(int(difference))
                remaining.pop(index)
                break
        else:
            return None
    if remaining:
        return None
    paired = list(zip(lower, shifts, strict=True))
    return {
        "instantiation": {
            "upper": [str(item) for item in upper],
            "lower": [str(item) for item in lower],
            "downward_shifts": shifts,
            "z": str(argument),
            "term_polynomial": str(_shift_reduction_polynomial(paired)),
        },
        "closed_form_text": (
            "R(k) = prod_j (k + b_j)_{m_j} is a polynomial, t_k = R(k) z^k / (R(0) k!), and "
            "sum_k k^p z^k/k! = exp(z) sum_j S(p, j) z^j with S the Stirling numbers of the "
            "second kind, so the sum is Q(z) exp(z) / R(0) for an explicit polynomial Q"
        ),
        "closed_form": lambda: _polynomial_times_exponential_value(paired, argument),
        "reduction": "polynomial_times_exponential",
    }


def _shift_reduction_polynomial(paired: Sequence[tuple[sp.Expr, int]]) -> sp.Expr:
    """``R(k) = prod_j (k + b_j)(k + b_j + 1)...(k + b_j + m_j - 1)``."""

    symbol = sp.Symbol("_k")
    polynomial = sp.Integer(1)
    for base, shift in paired:
        for step in range(shift):
            polynomial *= symbol + base + step
    return sp.expand(polynomial)


def _polynomial_times_exponential_value(
    paired: Sequence[tuple[sp.Expr, int]], argument: sp.Expr
) -> mp.mpf:
    """Evaluate the exact finite reduction ``Q(z) exp(z) / R(0)``."""

    symbol = sp.Symbol("_k")
    polynomial = _shift_reduction_polynomial(paired)
    at_zero = sp.simplify(polynomial.subs(symbol, 0))
    if at_zero == 0:
        raise ScreenError("the reduction polynomial vanishes at k = 0")
    coefficients = sp.Poly(polynomial, symbol).all_coeffs()[::-1]
    z = sp.Symbol("_z")
    total = sp.Integer(0)
    for power, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        inner = sum(
            sp.functions.combinatorial.numbers.stirling(power, j) * z**j
            for j in range(power + 1)
        )
        total += coefficient * inner
    closed = sp.simplify(total * sp.exp(z) / at_zero)
    return mp.mpf(str(sp.N(closed.subs(z, argument), mp.mp.dps + 10)))


def _match_elementary_2f1(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """``2F1(1, b; b+1; z)`` and its relatives: the tabulated elementary closed forms."""

    parsed = _upper_lower(data)
    if parsed is None:
        return None
    upper, lower, argument = parsed
    if len(upper) != 2 or len(lower) != 1:
        return None
    b_candidates = [(upper[0], upper[1]), (upper[1], upper[0])]
    c = lower[0]
    for one, b in b_candidates:
        if sp.simplify(one - 1) != 0:
            continue
        if sp.simplify(c - (b + 1)) != 0:
            continue
        # 2F1(1, b; b+1; z) = b * (-z)^(-b) * B_{z}(b, 0)-type incomplete form; the DLMF
        # elementary cases are z-hypergeometric with a logarithm or arctangent value.
        return {
            "instantiation": {"b": str(b), "c": str(c), "z": str(argument)},
            "closed_form_text": "b * Phi(z, 1, b) (Lerch), the elementary DLMF 15.4 family",
            "closed_form": lambda b=b: mp.mpf(str(sp.N(b, mp.mp.dps + 10)))
            * mp.lerchphi(
                mp.mpf(str(sp.N(argument, mp.mp.dps + 10))),
                mp.mpf(1),
                mp.mpf(str(sp.N(b, mp.mp.dps + 10))),
            ),
        }
    return None


def _match_gamma_product(data: Mapping[str, Any]) -> dict[str, Any] | None:
    signature = data.get("gamma_signature")
    if signature is None:
        return None
    numerator, denominator = signature
    if not numerator and not denominator:
        return None
    return {
        "instantiation": {
            "gamma_numerator_arguments": [str(item) for item in numerator],
            "gamma_denominator_arguments": [str(item) for item in denominator],
        },
        "closed_form_text": "prod Gamma(k0 - beta_j) / prod Gamma(k0 - alpha_i)",
        "closed_form": lambda: _gamma_ratio(numerator, denominator),
    }


def _gamma_ratio(numerator: Sequence[sp.Expr], denominator: Sequence[sp.Expr]) -> mp.mpf:
    value = mp.mpc(1)
    for item in numerator:
        value *= mp.gamma(_to_mp_number(item))
    for item in denominator:
        value /= mp.gamma(_to_mp_number(item))
    return +mp.mpf(mp.re(value))


def _to_mp_number(expression: sp.Expr) -> mp.mpf | mp.mpc:
    value = sp.N(expression, mp.mp.dps + 10)
    real, imaginary = sp.re(value), sp.im(value)
    if imaginary == 0:
        return mp.mpf(str(real))
    return mp.mpc(str(real), str(imaginary))


def _match_beta_integral(data: Mapping[str, Any]) -> dict[str, Any] | None:
    shape = data.get("integral_shape")
    if shape is None:
        return None
    a, b, kernel, power = shape
    if kernel != "one" or power != 0:
        return None
    return {
        "instantiation": {"a": str(a), "b": str(b)},
        "closed_form_text": "Gamma(a+1)Gamma(b+1)/Gamma(a+b+2)",
        "closed_form": lambda: mp.beta(
            mp.mpf(a.numerator) / a.denominator + 1, mp.mpf(b.numerator) / b.denominator + 1
        ),
    }


def _match_beta_reflection(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """The Beta integral at complementary exponents, which is exactly ``pi/sin(pi z)``."""

    shape = data.get("integral_shape")
    if shape is None:
        return None
    a, b, kernel, power = shape
    if kernel != "one" or power != 0:
        return None
    if (a + 1) + (b + 1) != 1:
        return None
    z = a + 1
    return {
        "instantiation": {"z": str(z)},
        "closed_form_text": "pi/sin(pi z)",
        "closed_form": lambda: mp.pi / mp.sin(mp.pi * mp.mpf(z.numerator) / z.denominator),
    }


def _lerch_log_power(a: Fraction, s: Fraction, sign: int, step: int) -> mp.mpf:
    """``Gamma(s+1) sum_{k>=0} sign^k / (k + a + 1)^(s+1)`` over the declared step."""

    exponent = mp.mpf(s.numerator) / s.denominator + 1
    offset = mp.mpf(a.numerator) / a.denominator + 1
    total = mp.nsum(
        lambda k: mp.mpf(sign) ** k / (mp.mpf(step) * k + offset) ** exponent,
        [0, mp.inf],
    )
    return mp.gamma(exponent) * total


def _match_log_power_zeta(data: Mapping[str, Any]) -> dict[str, Any] | None:
    shape = data.get("integral_shape")
    if shape is None:
        return None
    a, b, kernel, power = shape
    if kernel not in ("log_inv", "log_inv_over_1mx"):
        return None
    if power <= 0:
        return None
    # kernel log_inv with b = -1 is (ln(1/x))^c/(1-x); the composite kernel carries the
    # 1/(1-x) itself and then needs b = 0.
    if kernel == "log_inv" and b != -1:
        return None
    if kernel == "log_inv_over_1mx" and (b != 0 or power != 1):
        # (ln(1/x)/(1-x))^c is not the zeta integral for c != 1; only c = 1 is the
        # composite spelling of (ln(1/x))^1/(1-x).
        return None
    exponent = power
    return {
        "instantiation": {"a": str(a), "s": str(exponent), "kernel": kernel},
        "closed_form_text": "Gamma(s+1) * sum_{k>=0} (k+a+1)^(-(s+1))  (Hurwitz zeta)",
        "closed_form": lambda: _lerch_log_power(a, exponent, 1, 1),
    }


def _match_log_power_eta(data: Mapping[str, Any]) -> dict[str, Any] | None:
    shape = data.get("integral_shape")
    if shape is None:
        return None
    a, b, kernel, power = shape
    if kernel not in ("inv_1px", "log_inv_over_1px"):
        return None
    if kernel == "inv_1px":
        if b != 0 or power != 1:
            return None
        exponent = Fraction(0)
    else:
        if b != 0 or power != 1:
            return None
        exponent = Fraction(1)
    return {
        "instantiation": {"a": str(a), "s": str(exponent), "kernel": kernel},
        "closed_form_text": "Gamma(s+1) * sum_{k>=0} (-1)^k (k+a+1)^(-(s+1))  (Lerch/eta)",
        "closed_form": lambda: _lerch_log_power(a, exponent, -1, 1),
    }


def _match_log_power_dirichlet_beta(data: Mapping[str, Any]) -> dict[str, Any] | None:
    shape = data.get("integral_shape")
    if shape is None:
        return None
    a, b, kernel, power = shape
    if kernel not in ("inv_1px2", "log_inv_over_1px2"):
        return None
    if b != 0:
        return None
    if kernel == "inv_1px2":
        if power != 1:
            return None
        exponent = Fraction(0)
    else:
        if power != 1:
            return None
        exponent = Fraction(1)
    # int_0^1 x^a (ln(1/x))^s/(1+x^2) dx = Gamma(s+1) sum (-1)^k (2k+a+1)^-(s+1).
    return {
        "instantiation": {"a": str(a), "s": str(exponent), "kernel": kernel},
        "closed_form_text": "Gamma(s+1) * sum_{k>=0} (-1)^k (2k+a+1)^(-(s+1))  (Dirichlet beta)",
        "closed_form": lambda: _lerch_log_power(a, exponent, -1, 2),
    }


def _match_beta_derivative(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """``x^a (1-x)^b (ln(1/x))^n`` with integer ``n``: the differentiated Beta integral."""

    shape = data.get("integral_shape")
    if shape is None:
        return None
    a, b, kernel, power = shape
    if kernel != "log_inv" or power.denominator != 1 or power <= 0:
        return None
    if b <= -1:
        return None
    order = int(power)
    symbol = sp.Symbol("_alpha")
    expression = sp.gamma(symbol + 1) * sp.gamma(sp.Rational(b) + 1) / sp.gamma(
        symbol + sp.Rational(b) + 2
    )
    derivative = sp.diff(expression, symbol, order) * (-1) ** order
    point = sp.Rational(a)
    return {
        "instantiation": {"a": str(a), "b": str(b), "n": order},
        "closed_form_text": f"(-1)^{order} d^{order}/da^{order} B(a+1, b+1)",
        "closed_form": lambda: mp.mpf(
            str(sp.N(derivative.subs(symbol, point), mp.mp.dps + 10))
        ),
    }


# ---------------------------------------------------------------------------
# Candidate structure passed to the matchers
# ---------------------------------------------------------------------------


def candidate_structure(family: str, member: Mapping[str, Any]) -> dict[str, Any]:
    """The exact structural data a matcher may inspect."""

    data: dict[str, Any] = {"family": family}
    if family == "S":
        parameters = series_hypergeometric_parameters(
            member["p"], member["q"], Fraction(member["z"])
        )
        if parameters is not None:
            upper, lower, argument = parameters
            remaining = list(lower)
            kept: list[sp.Expr] = []
            for item in upper:
                for index, other in enumerate(remaining):
                    if sp.simplify(item - other) == 0:
                        remaining.pop(index)
                        break
                else:
                    kept.append(item)
            data["hypergeometric"] = (kept, remaining, argument)
    elif family == "P":
        signature = product_signature(member)
        k0 = sp.Integer(int(member["k0"]))
        _lead_a, roots_a = exact_roots(member["a"])
        _lead_b, roots_b = exact_roots(member["b"])
        numerator = [sp.simplify(k0 - root) for root in roots_b]
        denominator = [sp.simplify(k0 - root) for root in roots_a]
        data["gamma_signature"] = (numerator, denominator)
        data["gamma_signature_text"] = signature
    else:
        form = canonical_form("I", member)
        data["integral_shape"] = (
            Fraction(form["effective_a"]),
            Fraction(form["effective_b"]),
            form["effective_kernel"],
            Fraction(form["effective_power"]),
        )
    return data


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

#: Declared transformation groups, per family.  Every generator is value-preserving up to an
#: explicit factor that the expansion records, and every derived record keeps a parent edge
#: back to its cited seed.
DECLARED_TRANSFORMATIONS: dict[str, tuple[str, ...]] = {
    "S": ("index_shift",),
    "P": ("index_shift", "reciprocal"),
    "I": ("substitution_x_to_one_minus_x", "beta_contiguous_shift"),
}

#: Rewrites that leave the class invariant *unchanged*.  They are folded into the invariant
#: itself rather than used to grow the corpus, which is why the invariant is a one-step
#: decision: a common polynomial factor, a cancelling upper/lower parameter pair, a
#: permutation of parameters, or a kernel written as a power of another all denote the
#: same object and collapse before any comparison happens.
CANONICAL_REDUCTIONS: dict[str, tuple[str, ...]] = {
    "S": ("common_polynomial_factor", "upper_lower_parameter_cancellation", "parameter_permutation"),
    "P": ("common_polynomial_factor", "root_multiset_cancellation"),
    "I": ("pure_power_kernel_fold", "base_kernel_fold", "zero_power_kernel_collapse"),
}

#: Common polynomial factors used by the expansion: multiply both polynomials by these and
#: the object is unchanged, which is exactly the redundancy the enumeration's grammar has.
COMMON_FACTORS: tuple[tuple[int, ...], ...] = (
    (2, 0, 0, 0),
    (3, 0, 0, 0),
    (4, 0, 0, 0),
    (-1, 0, 0, 0),
    (-2, 0, 0, 0),
    (1, 1, 0, 0),
    (2, 1, 0, 0),
    (1, 0, 1, 0),
)


def _multiply(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...] | None:
    result = [0] * 4
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if a == 0 or b == 0:
                continue
            if i + j > 3:
                return None
            result[i + j] += a * b
    if any(abs(value) > 4 for value in result):
        return None
    return tuple(result)


def _shift_polynomial(coefficients: Sequence[int], shift: int) -> tuple[int, ...] | None:
    """``p(k) -> p(k + shift)`` inside the declared coefficient range, or ``None``."""

    symbol = sp.Symbol("_s")
    expression = sp.expand(
        sum(sp.Integer(int(c)) * (symbol + shift) ** i for i, c in enumerate(coefficients))
    )
    poly = sp.Poly(expression, symbol) if expression != 0 else None
    values = [0, 0, 0, 0]
    if poly is not None:
        for power, coefficient in zip(
            range(len(poly.all_coeffs()) - 1, -1, -1), poly.all_coeffs(), strict=True
        ):
            if power > 3:
                return None
            values[power] = int(coefficient)
    if any(abs(value) > 4 for value in values):
        return None
    return tuple(values)


_KERNEL_REFLECTION = {
    "one": "one",
    "log_inv": "log_inv_1mx",
    "log_inv_1mx": "log_inv",
    "inv_1mx": "inv_1mx",
}


def build_corpus(*, verbose: bool = False) -> dict[str, Any]:
    """Encode the seeds, self-certify them, and expand by the declared transformations."""

    started = time.perf_counter()
    records: list[CorpusRecord] = []
    seeds = concrete_seeds()
    parametrics = parametric_seeds()
    failures: list[str] = []

    for seed in seeds:
        with mp.workdps(SCREEN_DPS):
            try:
                value = family_value_mp(seed.family, seed.member)
                value_text = mp.nstr(value, VALUE_STORE_DIGITS)
            except Exception as failure:  # noqa: BLE001 - a seed that cannot evaluate is a bug
                failures.append(f"{seed.seed_id}: {type(failure).__name__}: {failure}")
                continue
        records.append(
            CorpusRecord(
                record_id=seed.seed_id,
                kind="seed",
                family=seed.family,
                seed_id=seed.seed_id,
                parent_id=None,
                depth=0,
                transform=(),
                signature=signature_key(seed.family, seed.member),
                value=value_text,
                value_expr=seed.value_expr,
                citation=seed.citation,
                validity_domain=seed.validity_domain,
            )
        )
    if failures:
        raise ScreenError("seed self-certification failed: " + "; ".join(failures))

    seed_records = list(records)
    for record, seed in zip(seed_records, seeds, strict=True):
        records.extend(_expand_seed(record, seed))

    by_id = {item.record_id: item for item in records}
    if len(by_id) != len(records):
        duplicates = len(records) - len(by_id)
        raise ScreenError(f"{duplicates} duplicate record ids in the corpus")
    closure = verify_forest_closure(records)

    signatures: dict[str, list[CorpusRecord]] = {}
    for record in records:
        if record.signature:
            signatures.setdefault(record.signature, []).append(record)
    values: dict[str, list[CorpusRecord]] = {}
    for record in records:
        values.setdefault(record.value[:30], []).append(record)

    manifest = {
        "schema_version": "invariant-inverse-symbolic-families-corpus-1.0",
        "counts": {
            "concrete_seeds": len(seeds),
            "parametric_theorem_seeds": len(parametrics),
            "total_seeds": len(seeds) + len(parametrics),
            "records": len(records),
            "distinct_signatures": len(signatures),
            "by_family": {
                family: sum(1 for item in records if item.family == family)
                for family in ("S", "P", "I")
            },
        },
        "declared_transformations": {
            family: list(value) for family, value in DECLARED_TRANSFORMATIONS.items()
        },
        "provenance_forest": closure,
        "claims": {
            "external_fetch_performed": False,
            "every_record_resolves_to_a_cited_seed": True,
            "seeds_self_certified_numerically_at_60_digits": True,
        },
        "citation_confidences": {
            confidence: sum(
                1 for item in records if item.citation.confidence == confidence
            )
            for confidence in CITATION_CONFIDENCES
        },
        "build_seconds": format(time.perf_counter() - started, ".3f"),
    }
    manifest["records_sha256"] = canonical_sha256([item.as_json() for item in records])
    manifest["content_sha256"] = canonical_sha256(manifest)
    if verbose:
        print(json.dumps(manifest["counts"], indent=2))
    return {
        "records": records,
        "by_id": by_id,
        "signatures": signatures,
        "values": values,
        "parametric": parametrics,
        "manifest": manifest,
    }


def _expand_seed(record: CorpusRecord, seed: ConcreteSeed) -> list[CorpusRecord]:
    """One declared expansion pass around a seed."""

    derived: list[CorpusRecord] = []
    seen = {record.signature}

    def add(member: Mapping[str, Any], transformation: str, detail: str) -> None:
        try:
            key = signature_key(seed.family, member)
        except Exception:  # noqa: BLE001 - an out-of-domain image is simply not added
            return
        if key in seen:
            return
        seen.add(key)
        with mp.workdps(SCREEN_DPS):
            try:
                value_text = mp.nstr(family_value_mp(seed.family, member), VALUE_STORE_DIGITS)
            except Exception:  # noqa: BLE001
                return
        derived.append(
            CorpusRecord(
                record_id=f"{record.record_id}#{transformation}:{len(derived)}",
                kind="derived",
                family=seed.family,
                seed_id=seed.seed_id,
                parent_id=record.record_id,
                depth=1,
                transform=(("transformation", transformation), ("detail", detail)),
                signature=key,
                value=value_text,
                value_expr=f"{seed.value_expr}  [{transformation}: {detail}]",
                citation=seed.citation,
                validity_domain=seed.validity_domain,
            )
        )

    if seed.family == "S":
        for shift in (1, 2):
            p = _shift_polynomial(seed.member["p"], shift)
            q = _shift_polynomial(seed.member["q"], shift)
            if p is None or q is None:
                continue
            add({**seed.member, "p": list(p), "q": list(q)}, "index_shift",
                f"P(k), Q(k) -> P(k+{shift}), Q(k+{shift}): the series tail after "
                f"{shift} term(s), related to the seed by an exact finite correction")
    elif seed.family == "P":
        add({**seed.member, "a": seed.member["b"], "b": seed.member["a"]},
            "reciprocal", "numerator and denominator exchanged")
        for k0 in (1, 2, 3):
            if k0 != seed.member["k0"]:
                add({**seed.member, "k0": k0}, "index_shift",
                    f"start index moved to {k0}, which multiplies the seed by the explicit "
                    "finite ratio of the omitted or added factors")
    else:
        a, b = Fraction(seed.member["a"]), Fraction(seed.member["b"])
        kernel = str(seed.member["kernel"])
        reflected = _KERNEL_REFLECTION.get(kernel)
        if reflected is not None:
            index = next(
                item["index"] for item in INTEGRAL_KERNELS if item["name"] == reflected
            )
            add(
                {**seed.member, "a": str(b), "b": str(a), "kernel": reflected,
                 "kernel_index": index},
                "substitution_x_to_one_minus_x",
                f"x -> 1-x, exponents swapped and kernel {kernel} -> {reflected}",
            )
        if kernel == "one" and Fraction(seed.member["power"]) == 0:
            for da, db in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (-1, 0), (0, -1)):
                if a + da <= -1 or b + db <= -1:
                    continue
                add(
                    {**seed.member, "a": str(a + da), "b": str(b + db)},
                    "beta_contiguous_shift",
                    f"(a, b) -> (a+{da}, b+{db}) by the Beta recurrence "
                    "B(a+1, b) = B(a, b) a/(a+b), DLMF 5.12.1",
                )
    return derived


def verify_forest_closure(records: Sequence[CorpusRecord]) -> dict[str, Any]:
    """Prove every non-seed record resolves to a cited seed by a declared transformation."""

    by_id = {item.record_id: item for item in records}
    longest = 0
    for record in records:
        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = record.record_id
        while current is not None:
            if current in seen:
                raise ScreenError(f"provenance cycle at {current}")
            seen.add(current)
            entry = by_id.get(current)
            if entry is None:
                raise ScreenError(f"dangling provenance edge to {current}")
            chain.append(current)
            if entry.kind == "seed":
                break
            name = dict(entry.transform).get("transformation")
            if name not in DECLARED_TRANSFORMATIONS[entry.family]:
                raise ScreenError(f"undeclared transformation {name!r} on {current}")
            current = entry.parent_id
        else:
            raise ScreenError(f"record {record.record_id} never reaches a seed")
        if by_id[chain[-1]].seed_id != record.seed_id:
            raise ScreenError(f"seed attribution mismatch for {record.record_id}")
        longest = max(longest, len(chain) - 1)
    return {
        "records": len(records),
        "max_chain_length": longest,
        "all_records_resolve_to_a_cited_seed": True,
    }


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def _values_agree(left: mp.mpf, right: mp.mpf, digits: int) -> bool:
    if not (mp.isfinite(left) and mp.isfinite(right)):
        return False
    scale = max(mp.mpf(1), abs(right))
    return bool(abs(left - right) / scale < mp.mpf(10) ** (-digits))


def _record_summary(corpus: Mapping[str, Any], record: CorpusRecord) -> dict[str, Any]:
    chain: list[dict[str, str]] = []
    current = record
    while current.parent_id is not None:
        detail = dict(current.transform)
        chain.append(
            {
                "record_id": current.record_id,
                "transformation": detail.get("transformation", ""),
                "detail": detail.get("detail", ""),
            }
        )
        current = corpus["by_id"][current.parent_id]
    return {
        "record_id": record.record_id,
        "family": record.family,
        "seed_id": record.seed_id,
        "identity": record.value_expr,
        "value": record.value,
        "citation": record.citation.as_json(),
        "validity_domain": record.validity_domain,
        "provenance": list(reversed(chain)),
    }


def screen_candidate(
    corpus: Mapping[str, Any], family: str, member: Mapping[str, Any], candidate_id: str,
    target: str, enumeration_label: str, member_value: str,
) -> dict[str, Any]:
    """Adjudicate one survivor.  Returns a fully explained verdict record."""

    with mp.workdps(SCREEN_DPS):
        prefactor = Fraction(member["prefactor"])
        value = family_value_mp(family, member)
        scaled = value * mp.mpf(prefactor.numerator) / prefactor.denominator
        target_value = constant_value(target)
        signature = signature_key(family, member)
        report: dict[str, Any] = {
            "candidate_id": candidate_id,
            "family": family,
            "target": target,
            "prefactor": str(prefactor),
            "signature": signature,
            "canonical_key": canonical_form(family, member)["key"],
            "enumeration_label": enumeration_label,
            "member_value_45_digits": mp.nstr(value, 45),
            "reproduces_target_at_45_digits": _values_agree(
                scaled, target_value, SCREEN_AGREEMENT_DIGITS
            ),
        }

        # Test 1 -- exact class invariant.
        for record in corpus["signatures"].get(signature, []):
            if _values_agree(mp.mpf(record.value), value, VALUE_MATCH_DIGITS):
                report.update(
                    {
                        "verdict": "KNOWN",
                        "test_that_fired": "exact_signature_match",
                        "matched_record": _record_summary(corpus, record),
                        "justification_verified": True,
                        "justification": (
                            "the candidate's exact class invariant is a corpus record's, and "
                            "equality of that invariant implies equality of the object"
                        ),
                    }
                )
                return report

        # Test 2 -- one step of the declared transformation group.
        declined_orbit_images: dict[str, str] = {}
        orbit = _orbit_step(family, member)
        for moved, step in orbit:
            # An orbit image can leave the declared grammar (a shifted exponent off the grid,
            # a polynomial that no longer factors over the declared set).  That is not an
            # error: the image simply is not a candidate for membership, so it is skipped and
            # the remaining generators are still tried.
            try:
                moved_signature = signature_key(family, moved)
            except Exception as failure:  # noqa: BLE001
                declined_orbit_images[step["transformation"]] = (
                    f"image left the declared grammar: {type(failure).__name__}"
                )
                continue
            for record in corpus["signatures"].get(moved_signature, []):
                with mp.workdps(SCREEN_DPS):
                    moved_value = family_value_mp(family, moved)
                if _values_agree(mp.mpf(record.value), moved_value, VALUE_MATCH_DIGITS):
                    report.update(
                        {
                            "verdict": "KNOWN",
                            "test_that_fired": "transformation_orbit_match",
                            "matched_record": _record_summary(corpus, record),
                            "transformation_chain": [step],
                            "justification_verified": True,
                            "justification": (
                                "a declared transformation carries the candidate onto the "
                                "matched record, and the image was re-evaluated at 60 digits"
                            ),
                        }
                    )
                    return report

        # Test 3 -- instance of a cited parametric theorem.
        structure = candidate_structure(family, member)
        declined: dict[str, str] = {}
        for theorem in corpus["parametric"]:
            if theorem.family != family:
                continue
            try:
                instance = theorem.matcher(structure)
            except Exception as failure:  # noqa: BLE001 - a matcher error declines, never matches
                declined[theorem.seed_id] = f"matcher error: {type(failure).__name__}"
                continue
            if instance is None:
                declined[theorem.seed_id] = "the candidate is outside this theorem's variety"
                continue
            predicted: mp.mpf | None = None
            if instance.get("closed_form") is not None:
                with mp.workdps(SCREEN_DPS):
                    try:
                        predicted = mp.mpf(mp.re(instance["closed_form"]()))
                    except Exception as failure:  # noqa: BLE001
                        declined[theorem.seed_id] = (
                            f"closed form did not evaluate: {type(failure).__name__}"
                        )
                        continue
                if not _values_agree(predicted, value, SCREEN_AGREEMENT_DIGITS):
                    declined[theorem.seed_id] = (
                        "the theorem's closed form does not reproduce the candidate's value"
                    )
                    continue
            else:
                declined[theorem.seed_id] = "no checkable closed form was produced"
                continue
            report.update(
                {
                    "verdict": "KNOWN",
                    "test_that_fired": "parametric_theorem_match",
                    "matched_record": {
                        "record_id": theorem.seed_id,
                        "family": theorem.family,
                        "seed_id": theorem.seed_id,
                        "identity": theorem.statement,
                        "value": mp.nstr(predicted, 45) if predicted is not None else "reduction",
                        "citation": theorem.citation.as_json(),
                        "validity_domain": theorem.validity_domain,
                        "provenance": [],
                    },
                    "theorem_instantiation": instance["instantiation"],
                    "theorem_closed_form": instance["closed_form_text"],
                    "justification_verified": True,
                    "justification": (
                        "the cited theorem was instantiated at the candidate's own exact "
                        "parameters and its closed form reproduces the candidate's value to "
                        f"{SCREEN_AGREEMENT_DIGITS} digits"
                    ),
                    "theorems_that_declined": declined,
                }
            )
            return report

        # Test 4 -- value equality, which is never membership on its own.
        by_value = [
            record
            for record in corpus["values"].get(mp.nstr(value, VALUE_STORE_DIGITS)[:30], [])
            if _values_agree(mp.mpf(record.value), value, VALUE_MATCH_DIGITS)
        ]
        reasons = {
            "exact_signature_match": "no corpus record carries this exact class invariant",
            "transformation_orbit_match": (
                f"none of the {len(orbit)} one-step images under the declared transformation "
                "group of this family lands on a corpus invariant"
            ),
            "orbit_images_outside_the_declared_grammar": declined_orbit_images,
            "parametric_theorem_match": declined,
        }
        if by_value:
            report.update(
                {
                    "verdict": "INCONCLUSIVE_VALUE_MATCH",
                    "test_that_fired": "value_match_without_structure",
                    "value_matches": [_record_summary(corpus, item) for item in by_value[:3]],
                    "why_no_structural_match": reasons,
                    "note": (
                        "a corpus record has the same value, but no declared transformation "
                        "and no cited theorem connects the two objects; two different "
                        "formulas can share a value"
                    ),
                }
            )
            return report

        report.update(
            {
                "verdict": "NOT_FOUND_IN_CORPUS",
                "test_that_fired": "no_test_fired",
                "why_no_structural_match": reasons,
                "note": (
                    "absence from this finite corpus; this is not a novelty claim and "
                    "requires human prior-art review before any such claim"
                ),
            }
        )
        return report


def _orbit_step(family: str, member: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    """One step of every declared non-canonical generator."""

    out: list[tuple[dict[str, Any], dict[str, str]]] = []
    if family == "S":
        for factor in COMMON_FACTORS:
            p = _multiply(member["p"], factor)
            q = _multiply(member["q"], factor)
            if p is None or q is None:
                continue
            out.append(
                (
                    {**member, "p": list(p), "q": list(q)},
                    {
                        "transformation": "common_polynomial_factor",
                        "detail": f"P and Q multiplied by {list(factor)}",
                    },
                )
            )
    elif family == "P":
        out.append(
            (
                {**member, "a": member["b"], "b": member["a"]},
                {"transformation": "reciprocal", "detail": "numerator and denominator exchanged"},
            )
        )
        for k0 in (1, 2, 3):
            if k0 != int(member["k0"]):
                out.append(
                    (
                        {**member, "k0": k0},
                        {"transformation": "index_shift", "detail": f"start index moved to {k0}"},
                    )
                )
        for factor in COMMON_FACTORS:
            a = _multiply(member["a"], factor)
            b = _multiply(member["b"], factor)
            if a is None or b is None:
                continue
            out.append(
                (
                    {**member, "a": list(a), "b": list(b)},
                    {
                        "transformation": "common_polynomial_factor",
                        "detail": f"A and B multiplied by {list(factor)}",
                    },
                )
            )
    else:
        form = canonical_form("I", member)
        kernel = form["effective_kernel"]
        reflected = _KERNEL_REFLECTION.get(kernel)
        if reflected is not None:
            index = next(
                item["index"] for item in INTEGRAL_KERNELS if item["name"] == reflected
            )
            out.append(
                (
                    {
                        "a": form["effective_b"],
                        "b": form["effective_a"],
                        "kernel": reflected,
                        "kernel_index": index,
                        "power": form["effective_power"],
                        "prefactor": member["prefactor"],
                    },
                    {
                        "transformation": "substitution_x_to_one_minus_x",
                        "detail": f"x -> 1-x; exponents swapped, kernel {kernel} -> {reflected}",
                    },
                )
            )
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def load_candidates(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One entry per *distinct object*, keeping the lowest ordinal as the representative."""

    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for block in receipt.get("families", []):
        family = block["family"]
        for survivor in block.get("survivors", []):
            key = (family, survivor["canonical_form"]["key"], survivor["target"])
            existing = seen.get(key)
            if existing is None or int(survivor["ordinal"]) < int(existing["ordinal"]):
                seen[key] = {**survivor, "family": family}
    return sorted(seen.values(), key=lambda item: (item["family"], int(item["ordinal"])))


def run_screen(
    receipt: Mapping[str, Any],
    corpus: Mapping[str, Any],
    *,
    receipt_path: str = "runs/math/inverse-symbolic/families-v1.json",
) -> dict[str, Any]:
    """Screen every distinct survivor object, enforce the controls, and seal a receipt."""

    started = time.perf_counter()
    candidates = load_candidates(receipt)
    if not candidates:
        raise ScreenError("input receipt carries no survivors")
    adjudications = [
        screen_candidate(
            corpus,
            item["family"],
            item,
            str(item["ordinal"]),
            str(item["target"]),
            str(item["prior_art"]["label"]),
            str(item.get("member_value_120_digits", "")),
        )
        for item in candidates
    ]

    controls = [item for item in adjudications if item["enumeration_label"] == "KNOWN_REDISCOVERED"]
    subjects = [item for item in adjudications if item["enumeration_label"] != "KNOWN_REDISCOVERED"]
    if not controls:
        raise ScreenError("no KNOWN_REDISCOVERED controls in the input receipt")
    recovered = sum(1 for item in controls if item["verdict"] == "KNOWN")
    rate = Fraction(recovered, len(controls))
    control_block = {
        "labelled_known_rediscovered": len(controls),
        "screened_KNOWN": recovered,
        "recovery_rate": f"{float(rate):.4f}",
        "threshold": f"{float(CONTROL_RECOVERY_THRESHOLD):.2f}",
        "per_family": {
            family: {
                "controls": sum(1 for item in controls if item["family"] == family),
                "recovered": sum(
                    1 for item in controls if item["family"] == family and item["verdict"] == "KNOWN"
                ),
            }
            for family in ("S", "P", "I")
        },
        "passed": bool(rate >= CONTROL_RECOVERY_THRESHOLD),
    }
    if not control_block["passed"]:
        raise ScreenError(
            f"control recovery rate {control_block['recovery_rate']} below the declared "
            f"threshold {control_block['threshold']}: the screen cannot recover known "
            "formulas and is therefore not fit to report any absence"
        )

    by_verdict = {name: sum(1 for item in subjects if item["verdict"] == name) for name in VERDICTS}
    by_test: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for item in subjects:
        by_test[item["test_that_fired"]] = by_test.get(item["test_that_fired"], 0) + 1
        bucket = by_family.setdefault(item["family"], {name: 0 for name in VERDICTS})
        bucket[item["verdict"]] += 1

    config = {
        "test_order": list(TEST_ORDER),
        "screen_dps": SCREEN_DPS,
        "agreement_digits": SCREEN_AGREEMENT_DIGITS,
        "value_match_digits": VALUE_MATCH_DIGITS,
        "control_recovery_threshold": f"{float(CONTROL_RECOVERY_THRESHOLD):.2f}",
        "class_invariants": {
            "S": "the pFq signature (upper multiset, lower multiset, argument) after cancellation",
            "P": "the Gamma signature (sorted k0 - beta_j, sorted k0 - alpha_i) after cancellation",
            "I": "the reduced shape (a, b, kernel, power) after the grammar degeneracies collapse",
        },
        "declared_transformations": {
            family: list(value) for family, value in DECLARED_TRANSFORMATIONS.items()
        },
        "candidate_deduplication": (
            "one entry per distinct canonical object per target; the lowest ordinal is the "
            "representative, so a grammar redundancy is never counted as a separate finding"
        ),
    }
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "lane": "inverse-symbolic-families-adjudication",
        "claims": SCREEN_CLAIMS,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "input": {
            "receipt": receipt_path,
            "content_sha256": receipt["content_sha256"],
            "result_core_sha256": receipt["result_core_sha256"],
            "survivors_in_receipt": sum(
                block["counts"]["survivors"] for block in receipt["families"]
            ),
            "distinct_objects_screened": len(candidates),
            "labelled_known_rediscovered": len(controls),
            "labelled_not_in_builtin_table": len(subjects),
        },
        "corpus": corpus["manifest"],
        "controls": control_block,
        "counts": {
            "by_verdict": by_verdict,
            "by_test_that_fired": dict(sorted(by_test.items())),
            "by_family": dict(sorted(by_family.items())),
        },
        "candidates": subjects,
        "control_summaries": [
            {
                "candidate_id": item["candidate_id"],
                "family": item["family"],
                "target": item["target"],
                "verdict": item["verdict"],
                "test_that_fired": item["test_that_fired"],
                "matched_record_id": item.get("matched_record", {}).get("record_id"),
                "citation_reference": item.get("matched_record", {})
                .get("citation", {})
                .get("reference"),
            }
            for item in controls
        ],
        "scope": (
            "Exact adjudication of series, product, and integral survivors against a corpus "
            "built from independently encoded classical identities, their declared "
            "transformation orbits, and cited parametric theorems. KNOWN requires either an "
            "exact class-invariant match, an exhibited transformation chain, or a cited "
            "theorem instantiated at the candidate's own parameters whose closed form "
            "reproduces the candidate's value at 45 digits. Value equality alone yields "
            "INCONCLUSIVE_VALUE_MATCH. NOT_FOUND_IN_CORPUS is absence from a finite corpus "
            "and is never a novelty claim."
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
    """Seals, claims, verdict vocabulary, count consistency, and the control gate."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise ScreenError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ScreenError("receipt seal changed")
    core_body = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    if value.get("result_core_sha256") != canonical_sha256(core_body):
        raise ScreenError("deterministic core seal changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise ScreenError("config binding changed")
    if value.get("claims") != SCREEN_CLAIMS:
        raise ScreenError("claims block changed")
    candidates = value.get("candidates", [])
    if len(candidates) != value["input"]["labelled_not_in_builtin_table"]:
        raise ScreenError("adjudicated candidate count changed")
    counts = {name: 0 for name in VERDICTS}
    for item in candidates:
        if item["verdict"] not in VERDICTS:
            raise ScreenError(f"unknown verdict {item['verdict']!r}")
        counts[item["verdict"]] += 1
        if item["verdict"] == "KNOWN":
            record = item.get("matched_record") or {}
            if not record.get("citation", {}).get("reference"):
                raise ScreenError(f"KNOWN verdict without a citation: {item['candidate_id']}")
            if not item.get("justification_verified"):
                raise ScreenError(
                    f"KNOWN verdict without a verified justification: {item['candidate_id']}"
                )
        elif not item.get("why_no_structural_match"):
            raise ScreenError(f"non-KNOWN verdict without a reason: {item['candidate_id']}")
    if counts != value["counts"]["by_verdict"]:
        raise ScreenError("verdict counts changed")
    controls = value["controls"]
    if controls["screened_KNOWN"] > controls["labelled_known_rediscovered"]:
        raise ScreenError("control recovery exceeds the control population")
    rate = Fraction(controls["screened_KNOWN"], controls["labelled_known_rediscovered"])
    if bool(rate >= CONTROL_RECOVERY_THRESHOLD) != controls["passed"]:
        raise ScreenError("control gate result changed")
    if not controls["passed"]:
        raise ScreenError("receipt records a failed control gate")
    if value["corpus"]["counts"]["total_seeds"] < 60:
        raise ScreenError("corpus carries fewer than the declared 60 seeds")
    if not value["corpus"]["provenance_forest"]["all_records_resolve_to_a_cited_seed"]:
        raise ScreenError("corpus provenance forest is not closed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise ScreenError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adjudicate family survivors against the prior-art corpus."
    )
    parser.add_argument("--input", default="runs/math/inverse-symbolic/families-v1.json")
    parser.add_argument("--output", default="runs/math/inverse-symbolic/families-screen-v1.json")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(json.dumps({"validated": True, "output": args.output}))
        return 0
    receipt = json.loads(Path(args.input).read_text(encoding="utf-8"))
    corpus = build_corpus()
    result = run_screen(receipt, corpus, receipt_path=args.input)
    write_receipt(result, args.output)
    print(
        json.dumps(
            {
                "distinct_objects_screened": result["input"]["distinct_objects_screened"],
                "candidates_adjudicated": len(result["candidates"]),
                "by_verdict": result["counts"]["by_verdict"],
                "by_test": result["counts"]["by_test_that_fired"],
                "by_family": result["counts"]["by_family"],
                "control_recovery_rate": result["controls"]["recovery_rate"],
                "corpus_seeds": result["corpus"]["counts"]["total_seeds"],
                "corpus_records": result["corpus"]["counts"]["records"],
                "output": args.output,
                "content_sha256": result["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
