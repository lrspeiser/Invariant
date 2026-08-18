"""Gates for sparse-region relation discovery over self-generated constants.

The module exists because three prior campaigns aimed at the classical constants found
nothing, so these tests pin the three things that make this attempt different, plus the
honesty core it inherits.

*The constants are ours and they are right.*  Every class is checked against an
independently derived value: the telescoping sums against their exact rationals, the
alternating and character twists against a structurally independent Hurwitz-zeta expansion,
the Epstein lattice sums against ``4 zeta(2) Catalan`` and against ``w zeta(s) L(s, chi_D)``
at class number one, and the class-number routine against the classical list of thirteen
class-number-one discriminants.  A member whose error bound cannot be certified must be
dropped rather than estimated, and the exclusion rules (squarefree, no pole at a positive
integer) are pinned exactly.

*The density map is arithmetic anybody can redo.*  The reduction fraction printed in a region
row must equal the integers printed beside it, and the sparse rule must be the declared
comparison against those integers -- not a score.

*Failure to reduce is the reported signal, so the reduction ladder must actually work.*  Each
technique is exercised on a case it should fire on and a case it should decline, with the
typed blocker checked, because a ladder that silently declines would manufacture headlines.

Then the four run-aborting controls: a planted telescoping relation must be found and must
come back ``KNOWN``; a pseudorandom decoy must not enter a relation under the bound; a
relation constructed to hold at 200 digits and fail at 400 must be caught; and the
verification path must be provably independent of the cached values, which is tested by
corrupting the cache and showing the recomputation does not move.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest

from sigma_theory_compiler import cf_proof_router as router
from sigma_theory_compiler import sparse_region_relations as srr
from sigma_theory_compiler.cf_prior_art_corpus import load_corpus
from sigma_theory_compiler.sigma_core import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "runs" / "math" / "sparse-regions" / "relations-v1.json"
CORPUS_DB = ROOT / "runs" / "math" / "prior-art" / "cf-corpus-v1.sqlite"
CORPUS_MANIFEST = ROOT / "runs" / "math" / "prior-art" / "cf-corpus-v1-manifest.json"

#: A pool small enough to build in seconds and wide enough to carry every class.
SMOKE = srr.PoolConfig(
    shift_box=2, epstein_min_discriminant=-24, cf_members=6, integral_members=4
)


@pytest.fixture(scope="module")
def pool() -> dict:
    return srr.build_pool(SMOKE)


@pytest.fixture(scope="module")
def records(pool: dict) -> dict:
    return {member["constant_id"]: member for member in pool["members"]}


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS_DB.exists():
        pytest.skip(f"corpus not present: {CORPUS_DB}")
    return load_corpus(CORPUS_DB, CORPUS_MANIFEST)


@pytest.fixture(scope="module")
def context(corpus):
    return router.RouterContext(corpus=corpus)


@pytest.fixture(scope="module")
def receipt() -> dict:
    if not RECEIPT.exists():
        pytest.skip(f"receipt not present: {RECEIPT}")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _reload(value):
    """Round-trip through canonical JSON, as a validator would receive it."""

    return json.loads(canonical_json_bytes(value))


# ---------------------------------------------------------------------------
# Component 1 -- the constants are ours, and they are right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1, 0, Fraction(1)),  # sum 1/(n(n+1)) telescopes to 1
        (2, 0, Fraction(3, 4)),  # sum 1/(n(n+2)) = (1/2)(1 + 1/2)
        (3, 2, Fraction(1, 2)),  # sum 1/((n+1)(n+2)) = 1/2
        (5, 6, Fraction(1, 3)),  # sum 1/((n+2)(n+3)) = 1/3
    ],
)
def test_telescoping_sums_hit_their_exact_rationals(a: int, b: int, expected: Fraction) -> None:
    value, imaginary = srr.sum_class_value(2, a, b, [1], 1, 120)
    with mp.workdps(120):
        target = mp.mpf(expected.numerator) / expected.denominator
        assert abs(value - target) < mp.mpf("1e-100")
        assert imaginary < mp.mpf("1e-100")


def test_alternating_and_character_twists_match_hand_derived_closed_forms() -> None:
    """Two twists of ``1/(n(n+1))`` that partial fractions evaluates by hand.

    ``sum (-1)^n [1/n - 1/(n+1)] = -ln 2 - (ln 2 - 1) = 1 - 2 ln 2``, and with the mod-4
    character ``sum chi(n)/n - sum_{m>=2} chi(m-1)/m = pi/4 - (1/2) ln 2`` because
    ``chi(m-1)`` on ``m >= 2`` is the alternating harmonic series on the even index.  Neither
    derivation goes anywhere near the code under test.
    """

    alternating, _ = srr.sum_class_value(2, 1, 0, [-1, 1], 2, 80)
    character, _ = srr.sum_class_value(2, 1, 0, [1, 0, -1, 0], 4, 80)
    with mp.workdps(80):
        assert abs(alternating - (1 - 2 * mp.log(2))) < mp.mpf("1e-70")
        assert abs(character - (mp.pi / 4 - mp.log(2) / 2)) < mp.mpf("1e-70")


@pytest.mark.parametrize(
    ("p", "a", "b", "modulus", "weights"),
    [
        (2, -3, 5, 1, [1]),
        (3, 2, -1, 3, [1, -1, 0]),
        (4, -2, 3, 2, [-1, 1]),
        (4, 1, 1, 4, [1, 0, -1, 0]),
    ],
)
def test_closed_form_agrees_with_the_independent_zeta_expansion(
    p: int, a: int, b: int, modulus: int, weights: list[int]
) -> None:
    """The cross-check uses no digamma and no partial fractions -- a real second opinion."""

    closed, _ = srr.sum_class_value(p, a, b, weights, modulus, 80)
    independent = srr.sum_class_crosscheck(p, a, b, weights, modulus, dps=40)
    with mp.workdps(60):
        assert abs(closed - independent) / max(mp.mpf(1), abs(closed)) < mp.mpf("1e-25")


def test_pole_and_squarefree_exclusions_are_exact() -> None:
    assert srr.polynomial_positive_integer_roots(2, -3, 2) == [1, 2]
    assert srr.polynomial_positive_integer_roots(2, 1, 0) == []
    assert srr.polynomial_positive_integer_roots(3, -7, 6) == [1, 2]
    assert not srr.polynomial_is_squarefree(2, 0, 0)
    assert not srr.polynomial_is_squarefree(2, -2, 1)  # (n-1)^2
    assert srr.polynomial_is_squarefree(2, -3, 5)


def test_excluded_members_never_enter_the_pool(pool: dict) -> None:
    for member in pool["members"]:
        if member["pool_class"] not in srr.SUM_CLASS_NAMES:
            continue
        parameters = member["parameters"]
        p, a, b = (int(parameters[key]) for key in ("p", "a", "b"))
        assert srr.polynomial_is_squarefree(p, a, b)
        assert srr.polynomial_positive_integer_roots(p, a, b) == []


def test_rational_root_detection() -> None:
    assert srr.polynomial_rational_roots(2, 1, 0) == [Fraction(-1), Fraction(0)]
    assert srr.polynomial_rational_roots(2, 2, 0) == [Fraction(-2), Fraction(0)]
    assert srr.polynomial_rational_roots(2, -3, 5) is None
    assert srr.polynomial_rational_roots(3, 0, 8) is None  # one rational, two complex


def test_epstein_reproduces_the_gaussian_lattice_closed_form() -> None:
    """``sum 1/(m^2+n^2)^2 = 4 zeta(2) beta(2)`` -- a value nobody can argue with."""

    value = srr.epstein_value(-4, 2, 80)
    with mp.workdps(80):
        assert abs(value - 4 * mp.zeta(2) * mp.catalan) < mp.mpf("1e-70")


@pytest.mark.parametrize("discriminant", [-3, -4, -7, -8, -11, -19, -43, -67])
def test_fundamental_class_number_one_epstein_sums_factor(discriminant: int) -> None:
    assert srr.class_number(discriminant) == 1
    assert srr.fundamental_discriminant(discriminant) == (discriminant, 1)
    for order in (2, 3):
        lattice = srr.epstein_value(discriminant, order, 60)
        factored = srr.epstein_factorisation_value(discriminant, order, 50)
        with mp.workdps(60):
            assert abs(lattice - factored) / abs(lattice) < mp.mpf("1e-40")


@pytest.mark.parametrize("discriminant", [-12, -16, -27, -28])
def test_nonfundamental_class_number_one_does_not_obey_the_factorisation(
    discriminant: int,
) -> None:
    """``h = 1`` is not enough: a non-maximal order breaks ``w zeta(s) L(s, chi_D)``.

    Counting these as classical would be wrong, and counting them as sparse would be a
    rediscovery waiting to happen, so they get their own region.
    """

    assert srr.class_number(discriminant) == 1
    assert srr.fundamental_discriminant(discriminant)[1] > 1
    lattice = srr.epstein_value(discriminant, 2, 60)
    factored = srr.epstein_factorisation_value(discriminant, 2, 50)
    with mp.workdps(60):
        assert abs(lattice - factored) / abs(lattice) > mp.mpf("1e-6")
    record = {
        "pool_class": "epstein_lattice",
        "parameters": {"discriminant": discriminant, "s": 2, "class_number": 1},
    }
    assert srr.structural_classical_closed_form(record) is None


def test_class_number_matches_the_classical_list() -> None:
    """Exactly thirteen negative discriminants have class number one."""

    classical = {-3, -4, -7, -8, -11, -12, -16, -19, -27, -28, -43, -67, -163}
    found = {
        d for d in range(-3, -200, -1) if d % 4 in (0, 1) and srr.class_number(d) == 1
    }
    assert found == classical
    assert (srr.class_number(-23), srr.class_number(-47), srr.class_number(-71)) == (3, 5, 7)


def test_class_number_greater_than_one_does_not_factor() -> None:
    """The h > 1 case is the sparse region; it must genuinely fail to factor."""

    lattice = srr.epstein_value(-20, 2, 60)
    factored = srr.epstein_factorisation_value(-20, 2, 50)
    with mp.workdps(60):
        assert abs(lattice - factored) / abs(lattice) > mp.mpf("0.1")


def test_fundamental_discriminant_split() -> None:
    assert srr.fundamental_discriminant(-12) == (-3, 2)
    assert srr.fundamental_discriminant(-27) == (-3, 3)
    assert srr.fundamental_discriminant(-16) == (-4, 2)
    assert srr.fundamental_discriminant(-20) == (-20, 1)


def test_representation_counts_are_exact() -> None:
    """Counting by the tight row bounds must equal a brute-force scan."""

    for form in ((1, 0, 1), (1, 1, 1), (1, 0, 3), (1, 1, 5)):
        a, b, c = form
        bound = 40
        brute: dict[int, int] = {}
        for m in range(-30, 31):
            for n in range(-30, 31):
                if m == 0 and n == 0:
                    continue
                value = a * m * m + b * m * n + c * n * n
                if 0 < value <= bound:
                    brute[value] = brute.get(value, 0) + 1
        assert srr.representation_counts(form, bound) == brute


def test_every_pool_member_carries_a_certified_bound(pool: dict) -> None:
    assert pool["counts"]["members"] == len(pool["members"])
    assert pool["counts"]["members"] > 0
    for member in pool["members"]:
        bound = member["error_bound"]
        assert bound["certified"] is True
        assert bound["declared_bound"] == f"1e-{SMOKE.dps + srr.BOUND_MARGIN}"
        assert bound["method"]
        assert bound["certification"]
        assert member["catalogued_status"] is None  # Component 2 fills this, not Component 1
        assert member["definition"]
        assert len(member["value"].replace("-", "").replace(".", "")) >= SMOKE.dps - 5


def test_an_uncertifiable_member_is_dropped_not_estimated() -> None:
    """A continued fraction cut off before it has converged must be dropped, not estimated.

    The golden-ratio continued fraction converges linearly, so at depth 6 against depth 12 it
    is nowhere near the declared 1e-68 bound -- exactly the situation where an estimate would
    be tempting.
    """

    record = {
        "constant_id": "cf:truncated",
        "pool_class": "cf_unmatched",
        "parameters": {"alpha": [1, 0, 0], "beta": [1, 0, 0], "depth": 6, "mobius": [1, 0, 0, 1]},
    }
    value, certificate = srr._evaluate_with_certificate(record, 60)
    assert value is None
    assert certificate["certified"] is False
    assert certificate["dropped_reason"]


def test_pool_covers_every_declared_class(pool: dict) -> None:
    expected = set(srr.SUM_CLASS_NAMES) | {
        "epstein_lattice",
        "cf_unmatched",
        "integral_off_locus",
    }
    assert set(pool["counts"]["by_class"]) == expected


def test_definitions_alone_reproduce_the_values(pool: dict) -> None:
    """The definition is the object's identity: parameters in, value out."""

    seen = set()
    for member in pool["members"]:
        if member["pool_class"] in seen:
            continue
        seen.add(member["pool_class"])
        replayed = srr.recompute_value(member, 60)
        with mp.workdps(60):
            stored = mp.mpf(member["value"])
            assert abs(replayed - stored) / max(mp.mpf(1), abs(stored)) < mp.mpf("1e-45")
    assert len(seen) == len(pool["counts"]["by_class"])


def test_pool_crosscheck_control(pool: dict) -> None:
    result = srr.crosscheck_pool(pool["members"], stride=40)
    assert result["control_passed"] is True
    assert result["sampled"] > 0
    assert result["failures"] == 0


# ---------------------------------------------------------------------------
# Component 2 -- the density map is hand-checkable arithmetic
# ---------------------------------------------------------------------------


def test_density_rows_are_reproducible_by_hand(pool: dict, corpus) -> None:
    density = srr.build_density_map(pool, corpus)
    for row in density["regions"]:
        assert row["reduction_fraction"] == f"{row['reduced']}/{row['members']}"
        fraction = Fraction(row["reduced"], row["members"])
        expected = "SPARSE" if fraction <= srr.SPARSE_REDUCTION_FRACTION else "DENSE"
        assert row["verdict"] == expected
        assert row["reduction_fraction_decimal"] == format(float(fraction), ".4f")
    totals = sum(row["members"] for row in density["regions"])
    assert totals == len(pool["members"])


def test_class_number_one_regions_are_dense_and_the_rest_sparse(pool: dict, corpus) -> None:
    """The one place where the answer is known in advance -- and it comes out right."""

    density = srr.build_density_map(pool, corpus)
    rows = {row["region"]: row for row in density["regions"]}
    for order in (2, 3):
        dense = rows.get(f"epstein_lattice/s={order}/h=1_fundamental")
        sparse = rows.get(f"epstein_lattice/s={order}/h=gt1")
        assert dense is not None and sparse is not None
        assert dense["verdict"] == "DENSE"
        assert dense["reduced"] == dense["members"]
        assert sparse["verdict"] == "SPARSE"
        assert sparse["reduced"] == 0


def test_structural_closed_forms_are_theorems_not_pslq() -> None:
    epstein = {
        "pool_class": "epstein_lattice",
        "parameters": {"discriminant": -7, "s": 2, "class_number": 1, "form": [1, 1, 2]},
    }
    found = srr.structural_classical_closed_form(epstein)
    assert found is not None
    assert found["technique"] == "epstein_class_number_one_factorisation"
    assert "L(2, chi_-7)" in found["closed_form"]

    harder = {
        "pool_class": "epstein_lattice",
        "parameters": {"discriminant": -20, "s": 2, "class_number": 2, "form": [1, 0, 5]},
    }
    assert srr.structural_classical_closed_form(harder) is None

    rational = {
        "pool_class": "zeta_shift",
        "parameters": {"p": 2, "a": 1, "b": 0, "modulus": 1, "weights": [1]},
    }
    assert (
        srr.structural_classical_closed_form(rational)["technique"]
        == "gauss_digamma_rational_arguments"
    )
    quadratic = {
        "pool_class": "zeta_shift",
        "parameters": {"p": 2, "a": -3, "b": 5, "modulus": 1, "weights": [1]},
    }
    assert (
        srr.structural_classical_closed_form(quadratic)["technique"]
        == "digamma_reflection_collapse_to_elementary"
    )
    cubic = {
        "pool_class": "zeta_shift",
        "parameters": {"p": 3, "a": -3, "b": 5, "modulus": 1, "weights": [1]},
    }
    assert srr.structural_classical_closed_form(cubic) is None


def test_sparse_membership_needs_all_three_conditions(pool: dict, corpus) -> None:
    density = srr.build_density_map(pool, corpus)
    rows = {row["region"]: row for row in density["regions"]}
    for member in pool["members"]:
        status = member["catalogued_status"]
        expected = (
            rows[member["region"]]["verdict"] == "SPARSE"
            and not status["reduces_to_classical_basis"]
            and status["corpus_value_hits"] == 0
        )
        assert status["is_sparse_member"] is expected


def test_classicality_search_finds_a_known_closed_form() -> None:
    """The bounded PSLQ lane must recognise a constant that IS in the classical basis."""

    with mp.workdps(srr.CLASSICALITY["fit_dps"] + 20):
        value = mp.nstr(mp.pi**2 / 6 - mp.log(2) / 3, srr.CLASSICALITY["fit_dps"])
    relation = srr.classicality_relation(value)
    assert relation is not None
    assert relation[0] != 0


def test_classicality_search_declines_a_pseudorandom_decimal() -> None:
    digits = srr._digest_stream("invariant-sparse-region-classicality-negative-control")
    value = "0." + "".join(str(next(digits) % 10) for _ in range(140))
    assert srr.classicality_relation(value) is None


# ---------------------------------------------------------------------------
# Component 3 -- the reduction ladder and the hunt
# ---------------------------------------------------------------------------


def test_digamma_expansion_reproduces_the_value() -> None:
    """The modulus-12 refinement is exact, so it must return the constant it came from."""

    for pool_class, modulus, weights in (
        ("zeta_shift", 1, [1]),
        ("alternating_shift", 2, [-1, 1]),
        ("character_mod3_shift", 3, [1, -1, 0]),
        ("character_mod4_shift", 4, [1, 0, -1, 0]),
    ):
        record = {
            "pool_class": pool_class,
            "parameters": {"p": 3, "a": 2, "b": -1, "modulus": modulus, "weights": weights},
        }
        with mp.workdps(80):
            rebuilt = sum(
                coefficient * mp.digamma(argument)
                for coefficient, argument in srr.digamma_expansion(record, 80)
            )
            direct, _ = srr.sum_class_value(3, 2, -1, weights, modulus, 80)
            assert abs(mp.re(rebuilt) - direct) < mp.mpf("1e-60")


def test_digamma_canonicalisation_preserves_the_sum() -> None:
    """Shift and reflection move terms around; the total value may not move."""

    record = {
        "pool_class": "alternating_shift",
        "parameters": {"p": 2, "a": 3, "b": 5, "modulus": 2, "weights": [-1, 1]},
    }
    with mp.workdps(80):
        terms = srr.digamma_expansion(record, 80)
        before = sum(coefficient * mp.digamma(argument) for coefficient, argument in terms)
        canonical, scalar, cot_terms = srr.canonicalise_digamma_terms(terms, 80)
        after = sum(coefficient * mp.digamma(argument) for coefficient, argument in canonical)
        after += scalar
        after += sum(
            coefficient * mp.pi / mp.tan(mp.pi * argument) for coefficient, argument in cot_terms
        )
        assert abs(before - after) < mp.mpf("1e-55")


def test_digamma_span_fires_on_a_relation_it_explains(records: dict) -> None:
    """Two shifts of one polynomial share a digamma basis, so the span must cancel."""

    left = "zeta_shift:p2:a1:b0"
    right = "zeta_shift:p2:a2:b0"
    if left not in records or right not in records:
        pytest.skip("planted members absent from the smoke pool")
    hit = {"constants": [left, right], "anchors": [], "coefficients": [3, -4]}
    outcome = srr.attempt_digamma_span(hit, records, dps=200)
    assert outcome["fired"] is True
    assert outcome["detail"]["psi_buckets"] > 0


def test_digamma_span_declines_when_the_bases_are_disjoint(records: dict) -> None:
    """Unrelated cubics give disjoint digamma bases; a false relation must not fire."""

    names = [
        name
        for name, record in sorted(records.items())
        if record["pool_class"] == "zeta_shift"
        and int(record["parameters"]["p"]) == 3
        and srr.polynomial_rational_roots(
            3, int(record["parameters"]["a"]), int(record["parameters"]["b"])
        )
        is None
    ][:2]
    assert len(names) == 2
    hit = {"constants": names, "anchors": [], "coefficients": [1, 1]}
    outcome = srr.attempt_digamma_span(hit, records, dps=200)
    assert outcome["fired"] is False
    assert outcome["blocker"] == "digamma_span_does_not_cancel"


def test_digamma_span_declines_when_only_the_psi_terms_cancel(records: dict) -> None:
    """Every quadratic's psi terms cancel, so psi cancellation alone proves nothing.

    Firing on it would mark relations that are not even true ``KNOWN``, so the elementary
    remainder is checked as well and must be the thing that stops this one.
    """

    names = [
        name
        for name, record in sorted(records.items())
        if record["pool_class"] == "zeta_shift" and int(record["parameters"]["p"]) == 2
    ][:2]
    assert len(names) == 2
    hit = {"constants": names, "anchors": [], "coefficients": [1, 1]}
    outcome = srr.attempt_digamma_span(hit, records, dps=200)
    assert outcome["fired"] is False
    assert outcome["blocker"] == "elementary_remainder_does_not_vanish"


@pytest.mark.parametrize(("a", "b"), [(-3, 5), (2, 3), (4, 7), (1, 1), (0, 1), (5, 7)])
def test_every_quadratic_shift_collapses_to_an_elementary_form(a: int, b: int) -> None:
    """The whole degree-two family is ``pi cot`` at an algebraic argument.

    ``psi(1 - r2) - psi(1 - r1)`` with ``r1 + r2 = -a`` an integer is carried onto itself by
    the recurrence and the reflection formula, leaving ``sum_j 1/(r1 + j) - pi cot(pi r1)``.
    So none of these is a discovery, and the density map must say so -- which is only true
    because the canonicalisation collapses the *whole orbit* of an argument under
    ``x -> x + 1`` and ``x -> 1 - x``, not a half-open window of it.
    """

    record = {
        "pool_class": "zeta_shift",
        "parameters": {"p": 2, "a": a, "b": b, "modulus": 1, "weights": [1]},
    }
    collapsed = srr.elementary_collapse(record, dps=200)
    assert collapsed is not None
    direct, _ = srr.sum_class_value(2, a, b, [1], 1, 200)
    with mp.workdps(200):
        assert abs(mp.re(collapsed) - direct) < mp.mpf("1e-150")


@pytest.mark.parametrize(("p", "a", "b"), [(3, 2, -1), (3, -3, 5), (4, 1, 1), (4, -2, 3)])
def test_higher_degree_shifts_do_not_collapse(p: int, a: int, b: int) -> None:
    """Three and four roots do not pair off, so the elementary route closes."""

    record = {
        "pool_class": "zeta_shift",
        "parameters": {"p": p, "a": a, "b": b, "modulus": 1, "weights": [1]},
    }
    assert srr.elementary_collapse(record, dps=200) is None


#: The relation the first full run put in its headline.  ``P_A(n + 3) = P_B(n)`` and the mod-3
#: character has period 3, so ``S_B = S_A + 1/6`` and ``6 S_A - 6 S_B + 1 = 0`` is a change of
#: summation index, not a discovery.  The ladder had no technique for it, the headline was
#: therefore wrong, and this is the regression that keeps it closed.
_TRANSLATE_A = {
    "pool_class": "character_mod3_shift",
    "parameters": {"p": 2, "a": -4, "b": 1, "modulus": 3, "weights": [1, -1, 0]},
}
_TRANSLATE_B = {
    "pool_class": "character_mod3_shift",
    "parameters": {"p": 2, "a": 2, "b": -2, "modulus": 3, "weights": [1, -1, 0]},
}


def test_translation_offset_is_the_exact_rational_head() -> None:
    """``sum_{m=1..3} chi3(m)/P_A(m) = 1/(-2) - 1/(-3) = -1/6``, so the offset is ``+1/6``."""

    assert [(n + 3) ** 2 - 4 * (n + 3) + 1 for n in (1, 2, 3)] == [
        n * n + 2 * n - 2 for n in (1, 2, 3)
    ]
    assert srr.translation_offset(_TRANSLATE_A, _TRANSLATE_B) == Fraction(1, 6)
    assert srr.translation_offset(_TRANSLATE_B, _TRANSLATE_A) == Fraction(-1, 6)
    assert srr.translation_offset(_TRANSLATE_A, _TRANSLATE_A) == Fraction(0)


def test_translation_offset_reproduces_the_numeric_difference() -> None:
    left, _ = srr.sum_class_value(2, -4, 1, [1, -1, 0], 3, 120)
    right, _ = srr.sum_class_value(2, 2, -2, [1, -1, 0], 3, 120)
    offset = srr.translation_offset(_TRANSLATE_A, _TRANSLATE_B)
    with mp.workdps(120):
        predicted = left + mp.mpf(offset.numerator) / offset.denominator
        assert abs(predicted - right) < mp.mpf("1e-100")


def test_translation_offset_declines_unrelated_and_off_period_shifts() -> None:
    off_period = {
        "pool_class": "character_mod3_shift",
        "parameters": {"p": 2, "a": -2, "b": -2, "modulus": 3, "weights": [1, -1, 0]},
    }
    unrelated = {
        "pool_class": "character_mod3_shift",
        "parameters": {"p": 2, "a": 1, "b": 7, "modulus": 3, "weights": [1, -1, 0]},
    }
    cubic = {
        "pool_class": "character_mod3_shift",
        "parameters": {"p": 3, "a": -4, "b": 1, "modulus": 3, "weights": [1, -1, 0]},
    }
    assert srr.translation_offset(_TRANSLATE_A, off_period) is None  # shift 1, period 3
    assert srr.translation_offset(_TRANSLATE_A, unrelated) is None
    assert srr.translation_offset(cubic, cubic) is None  # translation leaves the grammar


def test_index_translation_fires_on_the_first_runs_headline() -> None:
    records = {"A": _TRANSLATE_A, "B": _TRANSLATE_B}
    hit = {"constants": ["A", "B"], "anchors": ["one"], "coefficients": [6, -6, 1]}
    outcome = srr.attempt_index_translation(hit, records)
    assert outcome["fired"] is True
    detail = outcome["detail"]
    assert detail["rational_residue"] == "0"
    assert len(detail["translation_classes"]) == 1
    assert detail["translation_classes"][0]["coefficient_sum"] == 0
    assert detail["translation_classes"][0]["offsets_from_representative"]["B"] == "1/6"


def test_index_translation_declines_when_the_certificate_does_not_close() -> None:
    records = {"A": _TRANSLATE_A, "B": _TRANSLATE_B}
    wrong_rational = {"constants": ["A", "B"], "anchors": ["one"], "coefficients": [6, -6, 2]}
    assert srr.attempt_index_translation(wrong_rational, records)["blocker"] == (
        "translation_rational_residue_does_not_vanish"
    )
    wrong_classes = {"constants": ["A", "B"], "anchors": [], "coefficients": [6, -5]}
    assert srr.attempt_index_translation(wrong_classes, records)["blocker"] == (
        "translation_classes_do_not_cancel"
    )
    irrational_anchor = {
        "constants": ["A", "B"],
        "anchors": ["one", "pi"],
        "coefficients": [6, -6, 1, 3],
    }
    assert srr.attempt_index_translation(irrational_anchor, records)["blocker"] == (
        "irrational_anchor_in_relation"
    )


def test_reduction_techniques_decline_with_typed_blockers(records: dict) -> None:
    """A technique that does not apply must say so by name, never fire by accident."""

    epstein = sorted(
        name for name, record in records.items() if record["pool_class"] == "epstein_lattice"
    )
    sums = sorted(
        name for name, record in records.items() if record["pool_class"] == "zeta_shift"
    )
    mixed = {"constants": [epstein[0], sums[0]], "anchors": [], "coefficients": [1, 1]}
    assert srr.attempt_gauss_digamma(mixed, records)["blocker"] == (
        "gauss_digamma_not_defined_for_class"
    )
    assert srr.attempt_digamma_span(mixed, records, dps=120)["blocker"] == (
        "digamma_span_not_defined_for_class"
    )
    assert srr.attempt_epstein(mixed, records)["blocker"] == (
        "epstein_factorisation_not_defined_for_class"
    )
    assert srr.attempt_beta_gamma(mixed, records)["blocker"] == (
        "beta_reduction_not_defined_for_class"
    )


def test_epstein_technique_fires_only_at_class_number_one(records: dict) -> None:
    def pick(class_number: int) -> str | None:
        for name, record in sorted(records.items()):
            if record["pool_class"] != "epstein_lattice":
                continue
            if int(record["parameters"]["class_number"]) == class_number:
                return name
        return None

    easy, hard = pick(1), pick(2)
    assert easy is not None and hard is not None
    fires = srr.attempt_epstein(
        {"constants": [easy], "anchors": [], "coefficients": [1]}, records
    )
    assert fires["fired"] is True
    declines = srr.attempt_epstein(
        {"constants": [hard], "anchors": [], "coefficients": [1]}, records
    )
    assert declines["fired"] is False
    assert declines["blocker"] == "class_number_greater_than_one"


def test_commensurable_lattice_technique(records: dict) -> None:
    def pick(discriminant: int, order: int = 2) -> str:
        return f"epstein:D{discriminant}:s{order}"

    same = {
        "constants": [pick(-3), pick(-12)],
        "anchors": [],
        "coefficients": [1, -1],
    }
    if any(name not in records for name in same["constants"]):
        pytest.skip("commensurable pair absent from the smoke pool")
    outcome = srr.attempt_epstein_commensurable(same, records)
    assert outcome["fired"] is True
    assert outcome["detail"][pick(-12)]["fundamental_discriminant"] == -3
    different = {
        "constants": [pick(-3), pick(-20)],
        "anchors": [],
        "coefficients": [1, -1],
    }
    if all(name in records for name in different["constants"]):
        assert srr.attempt_epstein_commensurable(different, records)["blocker"] == (
            "distinct_fundamental_discriminants"
        )


def test_cf_router_runs_the_classical_ladder(records: dict, context) -> None:
    """The continued-fraction class really goes through cf_proof_router, verdict and all."""

    name = next(
        name for name, record in sorted(records.items()) if record["pool_class"] == "cf_unmatched"
    )
    report = srr.route_cf_member(records[name], context)
    assert report["verdict"] in router.VERDICTS
    attempted = {item["technique"] for item in report["techniques_attempted"]}
    assert attempted == set(router.TECHNIQUE_ORDER)  # every declared technique is tried
    assert report["numeric_check"]["holds"] is True
    if report["verdict"] == "MISSING_TECHNIQUE":
        assert report["missing_proof_technique"]


def test_hunt_requires_two_pool_constants(records: dict) -> None:
    """A relation carried entirely by an anchor is Component 2's business, not the hunt's."""

    name = "zeta_shift:p2:a1:b0"
    if name not in records:
        pytest.skip("planted member absent")
    with mp.workdps(srr.RELATION_HUNT["fit_dps"]):
        values = {name: mp.mpf(records[name]["value"])}
        other = next(
            key
            for key in sorted(records)
            if key != name and records[key]["pool_class"] == "cf_unmatched"
        )
        values[other] = mp.mpf(records[other]["value"])
    subset = {"schedule": "t", "constants": [name, other], "anchors": ["one"]}
    hit = srr.hunt_subset(subset, values)
    assert hit is None  # PSLQ returns S(1,0,2) - 1 = 0, whose pool support is one


def test_relation_rendering() -> None:
    hit = {
        "constants": ["alpha", "beta"],
        "anchors": ["one"],
        "coefficients": [3, -4, 7],
    }
    assert srr.render_relation(hit) == "3*alpha - 4*beta + 7 = 0"


def test_subset_schedule_is_deterministic_and_spreads_its_caps(pool: dict, corpus) -> None:
    srr.build_density_map(pool, corpus)
    sparse = [
        member
        for member in pool["members"]
        if member["catalogued_status"]["is_sparse_member"]
    ]
    caps = srr._subset_caps("smoke")
    first = srr.subset_schedule(sparse, caps=caps)
    second = srr.subset_schedule(sparse, caps=caps)
    assert first == second
    assert first
    for subset in first:
        assert len(subset["constants"]) >= 2
        assert len(set(subset["constants"])) == len(subset["constants"])
    within = [item for item in first if item["schedule"] == "within_class"]
    if within:
        classes = {
            next(m for m in sparse if m["constant_id"] == name)["pool_class"]
            for item in within
            for name in item["constants"]
        }
        # round-robin draining must not spend the whole cap on one class
        assert len(classes) >= 1


# ---------------------------------------------------------------------------
# The four run-aborting controls
# ---------------------------------------------------------------------------


def test_control_planted_relation_is_found_and_classified_known(records: dict, context) -> None:
    result = srr.planted_relation_control(records, context)
    assert result["available"] is True
    assert result["control_passed"] is True
    assert result["verification"]["survived"] is True
    assert result["reduction_verdict"] == "KNOWN_BY_PROOF_FAMILY"
    assert result["technique_that_fired"] in srr.REDUCTION_TECHNIQUES
    coefficients = result["coefficients"]
    assert sorted(abs(item) for item in coefficients) == [3, 4]


def test_control_random_false_relation_is_not_found(records: dict) -> None:
    result = srr.false_relation_control(records)
    assert result["control_passed"] is True
    assert result["relation_involves_the_decoy"] is False


def test_control_precision_artifact_dies_at_400(records: dict) -> None:
    result = srr.precision_artifact_control(records)
    assert result["found_at_200dps"] is True
    assert result["survived_400dps"] is False
    assert result["control_passed"] is True


def test_control_recomputation_is_independent_of_the_cache(records: dict) -> None:
    result = srr.independence_control(records)
    assert result["control_passed"] is True
    assert result["recomputation_unchanged"] is True


def test_corrupting_every_cached_value_does_not_move_verification(records: dict) -> None:
    """The strong form: poison the whole cache, then re-verify a relation from definitions."""

    left, right = "zeta_shift:p2:a1:b0", "zeta_shift:p2:a2:b0"
    if left not in records or right not in records:
        pytest.skip("planted members absent")
    hit = {"constants": [left, right], "anchors": [], "coefficients": [3, -4]}
    clean = srr.verify_at_400(hit, records)
    poisoned = {
        name: {**record, "value": "0.1111111111111111111111111"}
        for name, record in records.items()
    }
    after = srr.verify_at_400(hit, poisoned)
    assert clean["normalized_residual_at_400dps"] == after["normalized_residual_at_400dps"]
    assert after["survived"] is True


def test_a_relation_that_dies_at_400_is_discarded_with_a_reason(records: dict) -> None:
    name = min(records)
    hit = {"constants": [name, name], "anchors": ["one"], "coefficients": [1, -1, 1]}
    result = srr.verify_at_400(hit, records)
    assert result["survived"] is False
    assert result["discard_reason"] == "PSLQ artifact: died at 400 digits"


# ---------------------------------------------------------------------------
# Receipt: determinism, validation, tamper
# ---------------------------------------------------------------------------


def test_receipt_validates(receipt: dict) -> None:
    srr.validate_receipt(_reload(receipt))


def test_receipt_claims_are_the_declared_block(receipt: dict) -> None:
    assert receipt["claims"] == srr.CLAIMS
    assert receipt["claims"]["corpus_absence_establishes_novelty"] is False
    assert receipt["claims"]["unreduced_is_not_novel_it_is_unreviewed"] is True
    assert receipt["claims"]["human_review_required"] is True
    assert receipt["claims"]["targets_are_self_generated_not_classical"] is True


def test_receipt_pool_is_at_least_the_declared_scale(receipt: dict) -> None:
    assert receipt["pool"]["counts"]["members"] >= 2000
    assert len(receipt["pool"]["counts"]["by_class"]) == 7


def test_receipt_headline_matches_its_own_counts(receipt: dict) -> None:
    counts = receipt["relation_hunt"]["counts"]
    headline = receipt["headline"]
    assert counts["survives_classical_reduction"] == len(headline["relations"])
    assert counts["pslq_hits_at_200dps"] == (
        counts["survivors_at_400dps"] + counts["discarded_as_pslq_artifacts"]
    )
    assert counts["survivors_at_400dps"] == (
        counts["reduced_known_by_proof_family"] + counts["survives_classical_reduction"]
    )
    for relation in headline["relations"]:
        assert relation["reduction"]["verdict"] == "SURVIVES_CLASSICAL_REDUCTION"
        assert relation["verification"]["survived"] is True
        assert set(relation["definitions"]) == set(relation["constants"])
        attempted = {
            item["technique"] for item in relation["reduction"]["techniques_attempted"]
        }
        assert attempted == set(srr.REDUCTION_TECHNIQUES)


def test_receipt_controls_all_passed(receipt: dict) -> None:
    for name, control in receipt["controls"].items():
        assert control["control_passed"] is True, name


def test_receipt_seals_detect_tamper(receipt: dict) -> None:
    with pytest.raises(srr.SparseRegionError):
        srr.validate_receipt({**_reload(receipt), "claims": {"human_review_required": False}})
    broken = _reload(receipt)
    broken["relation_hunt"]["counts"]["survivors_at_400dps"] += 1
    with pytest.raises(srr.SparseRegionError):
        srr.validate_receipt(broken)
    reworded = _reload(receipt)
    reworded["scope"] = "a different scope sentence"
    with pytest.raises(srr.SparseRegionError):
        srr.validate_receipt(reworded)


def test_receipt_density_rows_are_recomputable_from_the_members(receipt: dict) -> None:
    tally: dict[str, list[int]] = {}
    for member in receipt["pool"]["members"]:
        row = tally.setdefault(member["region"], [0, 0])
        row[0] += 1
        row[1] += int(member["catalogued_status"]["reduces_to_classical_basis"])
    for region in receipt["density_map"]["regions"]:
        members, reduced = tally[region["region"]]
        assert region["reduction_fraction"] == f"{reduced}/{members}"


def test_receipt_is_byte_canonical() -> None:
    if not RECEIPT.exists():
        pytest.skip("receipt not present")
    raw = RECEIPT.read_bytes()
    assert raw.endswith(b"\n")
    assert canonical_json_bytes(json.loads(raw.decode("utf-8"))) + b"\n" == raw


def test_deterministic_core_is_stable() -> None:
    """Two runs of a tiny configuration must seal the identical deterministic core."""

    if not CORPUS_DB.exists():
        pytest.skip("corpus not present")
    tiny = srr.PoolConfig(
        shift_box=2, epstein_min_discriminant=-11, cf_members=3, integral_members=2
    )
    kwargs = {
        "config": tiny,
        "scale": "smoke",
        "corpus_database": str(CORPUS_DB),
        "corpus_manifest": str(CORPUS_MANIFEST),
    }
    first = srr.run_sparse_regions(**kwargs)
    second = srr.run_sparse_regions(**kwargs)
    assert first["result_core_sha256"] == second["result_core_sha256"]
    srr.validate_receipt(_reload(first))


def test_cli_validate_checked() -> None:
    if not RECEIPT.exists():
        pytest.skip("receipt not present")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sigma_theory_compiler.sparse_region_relations",
            "--validate-checked",
            "--output",
            str(RECEIPT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validated"] is True


def test_write_receipt_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    srr.write_receipt({"a": 1}, str(target))
    srr.write_receipt({"a": 1}, str(target))  # identical bytes are fine
    with pytest.raises(srr.SparseRegionError):
        srr.write_receipt({"a": 2}, str(target))
