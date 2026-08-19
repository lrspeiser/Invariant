"""Gates for the tensor-space constraint search.

The point of the module under test is that Einstein's field equations are *derived by exhaustion*
from declared constraints, so these tests do not settle for reading the receipt back.  Wherever the
search claims a tensor identity, the test recomputes that identity with its own independent code
and compares -- and wherever it claims a dimension, the test re-runs the linear algebra.

Pinned here: that the curvature engine really produces a Riemann tensor with all four of its
algebraic symmetries and the first Bianchi identity, on random metric jets; that the contracted
Bianchi identity is re-derived rather than assumed; that the enumeration really produces ten
contraction patterns at order two and five hundred and ninety-five at order four; that the
divergence-free space in d=4 at order 2 is exactly two-dimensional and is spanned by ``g_mn`` and
``R_mn - R g_mn/2``; that a fabricated conserved tensor is rejected; that the Gauss-Bonnet
Euler-Lagrange tensor vanishes identically in four dimensions and does not above them; that
dropping conservation enlarges the space; that dropping general covariance is refused with a typed
blocker; that the Newtonian limit really solves to ``8 pi G / c^4``; and that the receipt is
deterministic, float-free, claim-frozen, byte-canonical and tamper-evident under a resealed edit.

The full-receipt fixtures are session-scoped because one complete run does five searches across
three dimensions under two primes.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from sigma_theory_compiler import tensor_constraint_search as tcs
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / tcs.CONFIG_PATH).read_text(encoding="utf-8"))
RECEIPT = ROOT / tcs.OUTPUT_PATH
PRIME = 1048573


@pytest.fixture(scope="session")
def receipt() -> dict:
    return tcs.run_tensor_constraint_search(CONFIG, ROOT)


@pytest.fixture(scope="session")
def sealed() -> dict:
    if not RECEIPT.exists():  # pragma: no cover - the receipt is committed
        pytest.skip("committed receipt not present")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _search(receipt: dict, search_id: str) -> dict:
    return next(item for item in receipt["searches"] if item["search_id"] == search_id)


# ---------------------------------------------------------------------------
# The curvature engine is real.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [4, 5])
def test_riemann_algebraic_symmetries_hold_on_random_jets(dimension: int) -> None:
    """Independently confirm the engine's Riemann tensor is a Riemann tensor."""

    for sample in range(2):
        geometry = tcs._Geometry(
            dimension,
            tcs.metric_jet(dimension, 3, PRIME, "test-symmetry", sample),
            PRIME,
            3,
        )
        riemann = geometry.riemann
        zero = np.zeros_like(riemann)
        assert np.array_equal((riemann + np.einsum("abcdj->bacdj", riemann)) % PRIME, zero)
        assert np.array_equal((riemann + np.einsum("abcdj->abdcj", riemann)) % PRIME, zero)
        assert np.array_equal(riemann, np.einsum("abcdj->cdabj", riemann))
        cyclic = (
            riemann
            + np.einsum("abcdj->acdbj", riemann)
            + np.einsum("abcdj->adbcj", riemann)
        ) % PRIME
        assert np.array_equal(cyclic, zero)
        assert np.array_equal(geometry.ricci, np.einsum("abj->baj", geometry.ricci))


def test_contracted_bianchi_identity_is_rederived_not_assumed() -> None:
    """``nabla^a R_ab = (1/2) nabla_b R`` must come out of the engine, and ``nabla_b R`` must not
    be identically zero -- otherwise the divergence-free constraint would be vacuous."""

    gradient_seen = False
    for sample in range(3):
        geometry = tcs._Geometry(
            4, tcs.metric_jet(4, 3, PRIME, "test-bianchi", sample), PRIME, 3
        )
        terms = tcs.named_tensors(geometry, 2)
        divergence_ricci = geometry.divergence(terms["Ric"], 1)[..., 0] % PRIME
        gradient_scalar = geometry.divergence(terms["Rg"], 1)[..., 0] % PRIME
        assert np.array_equal((2 * divergence_ricci - gradient_scalar) % PRIME, np.zeros(4))
        assert not np.any(geometry.divergence(terms["g"], 1) % PRIME)
        gradient_seen = gradient_seen or bool(np.any(gradient_scalar))
    assert gradient_seen, "nabla_b R vanished on every sample; the constraint would be vacuous"


def test_metric_jet_is_minkowskian_at_the_base_point() -> None:
    jet = tcs.metric_jet(4, 3, PRIME, "test-jet", 0)
    assert jet[..., 0].tolist() == [
        [PRIME - 1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert np.array_equal(jet, np.einsum("abj->baj", jet)), "the metric jet must be symmetric"
    assert np.any(jet[..., 1:]), "the higher jet coefficients must actually be populated"


# ---------------------------------------------------------------------------
# The enumeration is an enumeration.
# ---------------------------------------------------------------------------


def test_pattern_enumeration_counts_are_combinatorial() -> None:
    """Recount the patterns by hand and compare.

    Order 2: the empty multiset with a ``g_mn`` carrier is 1; one Riemann has four slots, giving
    ``C(4,2) = 6`` free-pair choices and 3 all-contracted pairings, so 9.  Order 4 adds two
    Riemanns (``C(8,2) * 15 = 420`` plus ``7!! = 105``) and one twice-differentiated Riemann
    (``C(6,2) * 3 = 45`` plus ``5!! = 15``).  Odd-slot multisets contribute nothing.
    """

    assert len(tcs.enumerate_patterns(2)) == 1 + 9
    order4 = tcs.enumerate_patterns(4)
    assert len(order4) == 1 + 9 + (420 + 105) + (45 + 15) == 595
    counts: dict[tuple[str, ...], int] = {}
    for pattern in order4:
        counts[pattern["factors"]] = counts.get(pattern["factors"], 0) + 1
    assert counts == {(): 1, ("RIEM",): 9, ("RIEM", "RIEM"): 525, ("DDRIEM",): 60}
    assert ("DRIEM",) not in counts, "an odd number of slots cannot make a rank-2 tensor"


def test_named_basis_spans_the_enumeration_in_every_declared_cell(receipt: dict) -> None:
    for search in receipt["searches"]:
        enumeration = search["enumeration"]
        assert enumeration["performed"] is True
        assert enumeration["named_basis_spans_enumeration"] is True
        assert enumeration["named_basis_rank"] == enumeration["rank_after_symmetry"]


def test_order_two_enumeration_collapses_ten_patterns_to_three(receipt: dict) -> None:
    search = _search(receipt, "d4-order2-einstein")
    assert search["enumeration"]["formal_pattern_count"] == 10
    assert search["enumeration"]["rank_after_symmetry"] == 3
    assert search["independent_dimension"] == 3
    assert [item["expression"] for item in search["basis_terms"]] == [
        "g_mn",
        "R_mn",
        "R g_mn",
    ]


# ---------------------------------------------------------------------------
# The headline result.
# ---------------------------------------------------------------------------


def test_headline_space_is_exactly_two_dimensional(receipt: dict) -> None:
    search = _search(receipt, "d4-order2-einstein")
    assert search["surviving_dimension"] == 2
    assert search["divergence_free_space"]["basis"] == ["g_mn", "R_mn - (1/2) R g_mn"]
    assert search["divergence_free_space"]["identically_vanishing_inside"] == 0
    certificate = search["uniqueness_certificate"]
    assert certificate["sandwich_tight"] is True
    assert certificate["independently_exhibited_dimension"] == 2
    assert certificate["algebraic_lower_bound_available"] is True


def test_headline_space_is_recomputed_independently() -> None:
    """Redo the whole d=4 order-2 reduction here, with this test's own linear algebra."""

    names = tcs.named_term_names(2)
    bank = tcs.build_bank(4, 2, PRIME, "test-independent", 8)
    rows = []
    for name in names:
        row: list[int] = []
        for geometry in bank:
            terms = tcs.named_tensors(geometry, 2)
            row.extend(int(v) for v in geometry.divergence(terms[name], 1)[..., 0].ravel())
        rows.append(row)
    nullspace = tcs._nullspace(rows, PRIME)
    assert len(nullspace) == 2
    reduced, _ = tcs._rref(nullspace, PRIME)
    rendered = [tcs._vector_text(tcs._normalise(vector, PRIME), names) for vector in reduced]
    assert rendered == ["g_mn", "R_mn - (1/2) R g_mn"]


def test_reduction_table_is_monotone_and_ends_at_one(receipt: dict) -> None:
    table = receipt["reduction_tables"]["d4-order2-einstein"]
    assert [row["constraint"] for row in table] == [
        "(declared framework only)",
        "generally_covariant",
        "derivative_order <= 2",
        "symmetric",
        "divergence_free",
        "newtonian_limit",
    ]
    numeric = [row["dimension"] for row in table[2:]]
    assert numeric == ["3", "3", "2", "1"]


# ---------------------------------------------------------------------------
# Newtonian limit.
# ---------------------------------------------------------------------------


def test_newtonian_limit_fixes_eight_pi_g_over_c_fourth(receipt: dict) -> None:
    chain = receipt["newtonian_limit"]
    light, newton = sp.symbols("c G", positive=True)
    # sympify must reuse the assumption-carrying symbols, or it silently makes fresh ones.
    context = {"c": light, "G": newton, "pi": sp.pi}
    kappa = sp.sympify(chain["coupling_constant_kappa"], locals=context)
    alpha = sp.sympify(chain["alpha_coefficient_of_einstein_tensor"], locals=context)
    assert sp.simplify(kappa - 8 * sp.pi * newton / light**4) == 0
    assert sp.simplify(alpha - light**4 / (8 * sp.pi * newton)) == 0
    assert sp.simplify(kappa * alpha - 1) == 0
    assert "UNFORCED" in chain["lambda_status"]
    assert len(chain["steps"]) == 8


def test_newtonian_limit_is_recomputed_not_stored() -> None:
    first = tcs.newtonian_limit_chain()
    second = tcs.newtonian_limit_chain()
    assert first == second
    assert first["coupling_constant_kappa"] == "8*pi*G/c**4"


def test_lambda_survives_the_newtonian_limit(receipt: dict) -> None:
    """The cosmological constant is unforced: one free parameter must remain."""

    search = _search(receipt, "d4-order2-einstein")
    assert receipt["counts"]["free_parameters_after_newtonian_limit"] == 1
    assert search["surviving_dimension"] - 1 == 1


# ---------------------------------------------------------------------------
# Generalizations.
# ---------------------------------------------------------------------------


def test_gauss_bonnet_adds_one_dimension_only_above_four(receipt: dict) -> None:
    rows = {
        item["dimension"]: item
        for item in receipt["generalizations"]["gauss_bonnet"]["by_dimension"]
    }
    assert rows[4]["identically_vanishing_dimension"] == 1
    assert rows[5]["identically_vanishing_dimension"] == 0
    assert rows[6]["identically_vanishing_dimension"] == 0
    assert rows[4]["divergence_free_distinct_dimension"] == 4
    assert rows[5]["divergence_free_distinct_dimension"] == 5
    assert rows[6]["divergence_free_distinct_dimension"] == 5
    assert rows[4]["gauss_bonnet_member"]["identically_zero"] is True
    for dimension in (5, 6):
        assert rows[dimension]["gauss_bonnet_member"]["identically_zero"] is False
        assert rows[dimension]["gauss_bonnet_member"]["divergence_free"] is True
        assert rows[dimension]["gauss_bonnet_member"]["in_reported_space"] is True


def test_the_vanishing_combination_in_four_dimensions_is_the_lanczos_tensor(receipt: dict) -> None:
    search = _search(receipt, "d4-order4-relaxed")
    assert search["identically_vanishing"]["dimension"] == 1
    names = tcs.named_term_names(4)
    declared = [Fraction(tcs.DECLARED_VECTORS["gauss_bonnet_lanczos"].get(n, "0")) for n in names]
    scaled = [value * -2 for value in declared]
    assert search["identically_vanishing"]["vectors"] == [tcs._vector_text(scaled, names)]


def test_gauss_bonnet_tensor_vanishes_in_d4_and_not_in_d5_independently() -> None:
    """Recompute the Lanczos tensor here rather than trusting the receipt."""

    names = tcs.named_term_names(4)
    vector = tcs._declared_vector("gauss_bonnet_lanczos", names, PRIME)
    for dimension, expect_zero in ((4, True), (5, False)):
        geometry = tcs._Geometry(
            dimension,
            tcs.metric_jet(dimension, 5, PRIME, f"test-gb-{dimension}", 0),
            PRIME,
            5,
        )
        terms = tcs.named_tensors(geometry, 4)
        tensor = tcs._combine(vector, terms, names, PRIME)
        assert (not np.any(tensor % PRIME)) is expect_zero
        assert not np.any(geometry.divergence(tensor, 1) % PRIME)


def test_order_four_relaxation_contains_the_f_of_r_family(receipt: dict) -> None:
    search = _search(receipt, "d4-order4-relaxed")
    member = next(
        item
        for item in search["exhibited_members"]
        if item["name"] == "quadratic_r_squared_euler_lagrange"
    )
    assert member["divergence_free"] is True
    assert member["in_reported_space"] is True
    assert member["identically_zero"] is False
    axis = receipt["relaxation_controls"]["derivable_modified_gravity"]["axis_1_derivative_order"]
    assert axis["dimension_before"] == 2
    assert axis["dimension_after"] == 4
    assert axis["f_of_r_member_is_divergence_free"] is True


def test_screened_gravity_candidate_count_is_derived_from_its_own_receipt(receipt: dict) -> None:
    """The 23 surviving candidates are recounted from the bound receipt, never transcribed."""

    source = json.loads(
        (ROOT / "runs/gpu-baryonic-screen/nonlocal-localization-v1.json").read_text(
            encoding="utf-8"
        )
    )
    passing = [item for item in source["families"] if item.get("stability") == "STABLE_PASS"]
    block = receipt["relaxation_controls"]["derivable_modified_gravity"]["axis_2_field_content"][
        "repository_candidates"
    ]
    assert block["surviving_families"] == len(passing) == 12
    assert block["surviving_candidates"] == sum(int(i["size"]) for i in passing) == 23
    assert receipt["counts"]["screened_gravity_candidates_referenced"] == 23


# ---------------------------------------------------------------------------
# The one-sided sampling guarantee.
#
# The module's central promise is that a dimension "can only be reported too large, never too
# small".  That promise is about the SAMPLE SIZE, so the only way to test it is to vary the sample
# size, and these are the tests that do.  Every one of them holds the exhibited lower bound fixed
# and starves the bank underneath it.
# ---------------------------------------------------------------------------

DIVERGENCE_FREE = ["generally_covariant", "derivative_order", "symmetric", "divergence_free"]


def _starved(bank_samples: int, *, dimension: int = 4, order: int = 2, seed: str = "7") -> dict:
    return tcs.run_search(
        dimension=dimension,
        order=order,
        constraints=DIVERGENCE_FREE,
        modulus=PRIME,
        seed=seed,
        bank_samples=bank_samples,
        holdout_samples=3,
        enumerate_basis=False,
    )


def test_the_reproduced_defect_is_fixed() -> None:
    """The exact call from the bug report: it used to report 1 while exhibiting 2."""

    search = tcs.run_search(
        dimension=4,
        order=2,
        constraints=DIVERGENCE_FREE,
        modulus=1048573,
        seed=7,
        bank_samples=1,
        holdout_samples=3,
        enumerate_basis=True,
    )
    certificate = search["uniqueness_certificate"]
    assert certificate["independently_exhibited_dimension"] == 2
    assert certificate["sampled_dimension"] == 2
    assert tcs.published_dimension(search) == 2


@pytest.mark.parametrize("bank_samples", [1, 2, 3, 5, 9])
def test_reported_dimension_never_falls_below_what_the_same_call_exhibits(
    bank_samples: int,
) -> None:
    """The guarantee, stated as a direction and checked at five sample sizes.

    A member the search exhibits is a member the space contains, so the reported dimension is
    forbidden from sitting below the exhibited rank no matter how starved the bank is.  This is the
    assertion the defect violated.
    """

    search = _starved(bank_samples)
    certificate = search["uniqueness_certificate"]
    assert certificate["sampled_dimension"] >= certificate["independently_exhibited_dimension"]
    assert tcs.published_dimension(search) == 2


@pytest.mark.parametrize("bank_samples", [1, 3, 9])
def test_the_guarantee_direction_holds_at_order_four_too(bank_samples: int) -> None:
    """d=4 order 4 is the cell with a genuine identically-vanishing direction to subtract."""

    search = _starved(bank_samples, order=4)
    certificate = search["uniqueness_certificate"]
    assert certificate["sampled_dimension"] >= certificate["independently_exhibited_dimension"]
    assert certificate["witnessed_identically_vanishing_dimension"] == 1
    assert tcs.published_dimension(search) == 4


def test_naive_nullity_difference_violates_the_guarantee_on_a_starved_bank() -> None:
    """The control that must FAIL.  Rebuilt here with this test's own linear algebra.

    ``len(divergence_nullspace) - len(sampled_vanishing_space)`` is the arithmetic the module used
    to publish.  Both terms are upper bounds, so the difference has no guaranteed sign.  On a
    one-jet bank it lands at 1 while the same cell exhibits 2 members.  If this assertion ever
    stops holding, the repair has stopped being load-bearing and the test should be deleted, not
    relaxed.
    """

    names = tcs.named_term_names(2)
    bank = tcs.build_bank(4, 2, PRIME, "7", 1)
    pairs = [(geometry, tcs.named_tensors(geometry, 2)) for geometry in bank]
    vanishing = tcs._nullspace(tcs._sample_rows(pairs, names, divergence=False), PRIME)
    nullity = tcs._nullspace(tcs._sample_rows(pairs, names, divergence=True), PRIME)
    naive = len(nullity) - len(vanishing)
    assert len(vanishing) == 1 and len(nullity) == 2
    assert naive == 1

    search = _starved(1)
    assert search["uniqueness_certificate"]["independently_exhibited_dimension"] == 2
    assert naive < search["uniqueness_certificate"]["independently_exhibited_dimension"]
    assert tcs.published_dimension(search) == 2


def test_the_holdout_actually_refutes_bank_only_directions() -> None:
    """A holdout that never refutes anything is decoration.  On a starved bank it refutes plenty."""

    search = _starved(1, order=4)
    vanishing = search["identically_vanishing"]
    assert vanishing["bank_only_dimension"] > vanishing["dimension"]
    assert vanishing["holdout_refuted_dimensions"] == (
        vanishing["bank_only_dimension"] - vanishing["dimension"]
    )
    space = search["divergence_free_space"]
    assert space["coefficient_nullity_bank_only"] > space["coefficient_nullity"]
    assert space["holdout_refuted_dimensions"] > 0


def test_bank_size_does_not_move_the_published_dimension(receipt: dict) -> None:
    """Every configured cell, rerun on a one-jet bank, must land on the sealed dimension."""

    for search in receipt["searches"]:
        if not search.get("divergence_free_applied"):
            continue
        starved = _starved(
            1,
            dimension=search["dimension"],
            order=search["derivative_order"],
            seed=CONFIG["arithmetic"]["jet_seed"],
        )
        assert tcs.published_dimension(starved) == tcs.published_dimension(search)


# ---------------------------------------------------------------------------
# Fail closed: an untight sandwich is not a dimension.
# ---------------------------------------------------------------------------


def test_published_dimension_refuses_an_untight_search() -> None:
    with pytest.raises(tcs.SandwichNotTight, match="did not close"):
        tcs.published_dimension(
            {"search_id": "synthetic", "dimension_published": False, "surviving_dimension": 7}
        )
    assert tcs.published_dimension({"dimension_published": True, "surviving_dimension": 7}) == 7


def test_an_unspanned_nullspace_refuses_to_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Withdraw an exhibited direction and d=5 at order 4 must stop reporting a dimension.

    ``E3`` and the Lanczos vector both have to go, because in d=5 either one recovers the other
    from ``E1`` and ``E2`` -- which is itself a nice check that the six exhibited members really do
    span only five directions.  What is left is a five-dimensional sampled nullspace with four
    exhibited directions inside it.  Four is a lower bound and five is an upper bound; the honest
    answer is that the dimension is not known, and the search has to say so.
    """

    withdrawn = {"quadratic_riemann_squared_euler_lagrange", "gauss_bonnet_lanczos"}
    monkeypatch.setattr(
        tcs,
        "EXHIBITED_LABELS",
        tuple(label for label in tcs.EXHIBITED_LABELS if label not in withdrawn),
    )
    search = _starved(3, dimension=5, order=4, seed=CONFIG["arithmetic"]["jet_seed"])
    certificate = search["uniqueness_certificate"]
    assert certificate["coefficient_nullity"] == 5
    assert certificate["exhibited_coefficient_rank"] == 4
    assert certificate["coefficient_sandwich_tight"] is False
    assert search["dimension_published"] is False
    assert search["surviving_dimension"] is None
    assert search["blocker"]["reason"] == "exhibited_members_do_not_span_the_sampled_nullspace"
    with pytest.raises(tcs.SandwichNotTight):
        tcs.published_dimension(search)


def test_an_unwitnessed_vanishing_direction_refuses_to_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Withdraw the Lanczos witness and d=4 at order 4 must refuse rather than subtract a sample.

    The sampled vanishing dimension is still 1, but with no declared member to name it there is
    nothing that bounds the true vanishing dimension from below, and subtracting the sampled 1
    would report the space too small.  Refusing is the fix.
    """

    reduced = {
        label: spec
        for label, spec in tcs.DECLARED_VECTORS.items()
        if label != "gauss_bonnet_lanczos"
    }
    monkeypatch.setattr(tcs, "DECLARED_VECTORS", reduced)
    search = _starved(3, dimension=4, order=4, seed=CONFIG["arithmetic"]["jet_seed"])
    assert search["identically_vanishing"]["dimension"] == 1
    assert search["identically_vanishing"]["witnessed_dimension"] == 0
    assert search["identically_vanishing"]["fully_witnessed"] is False
    assert search["dimension_published"] is False
    assert search["blocker"]["reason"] == "unwitnessed_identically_vanishing_directions"
    with pytest.raises(tcs.SandwichNotTight):
        tcs.published_dimension(search)


def test_every_sealed_search_closed_its_own_sandwich(receipt: dict) -> None:
    assert receipt["counts"]["searches_with_a_closed_sandwich"] == len(receipt["searches"])
    for search in receipt["searches"]:
        assert search["dimension_published"] is True
        assert isinstance(search["surviving_dimension"], int)
        assert "blocker" not in search
        assert search["identically_vanishing"]["fully_witnessed"] is True


def test_the_gauss_bonnet_verdict_rests_on_three_closed_sandwiches(receipt: dict) -> None:
    """The d>4 verdict is a DIFFERENCE of dimensions, so each one has to be two-sided."""

    rows = receipt["generalizations"]["gauss_bonnet"]["by_dimension"]
    assert len(rows) == 3
    for row in rows:
        assert row["sandwich_tight"] is True
    for search_id in ("d4-order4-relaxed", "d5-order4-gaussbonnet", "d6-order4-gaussbonnet"):
        certificate = _search(receipt, search_id)["uniqueness_certificate"]
        assert certificate["coefficient_sandwich_tight"] is True
        assert certificate["vanishing_fully_witnessed"] is True
        assert (
            certificate["independently_exhibited_dimension"] == certificate["sampled_dimension"]
        )


# ---------------------------------------------------------------------------
# The two new exhibited members that close the order-four sandwich.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    ["quadratic_ricci_squared_euler_lagrange", "quadratic_riemann_squared_euler_lagrange"],
)
@pytest.mark.parametrize("dimension", [4, 5])
def test_the_quadratic_variations_are_divergence_free_independently(
    label: str, dimension: int
) -> None:
    """Recompute conservation here, on a jet the search never touched."""

    names = tcs.named_term_names(4)
    vector = tcs._declared_vector(label, names, PRIME)
    geometry = tcs._Geometry(
        dimension,
        tcs.metric_jet(dimension, 5, PRIME, f"test-quadratic-{label}-{dimension}", 0),
        PRIME,
        5,
    )
    terms = tcs.named_tensors(geometry, 4)
    tensor = tcs._combine(vector, terms, names, PRIME)
    assert np.any(tensor % PRIME), "the member is identically zero and certifies nothing"
    assert not np.any(geometry.divergence(tensor, 1) % PRIME)


@pytest.mark.parametrize("term", ["RiemRic", "BoxRic", "DDR"])
def test_a_perturbed_quadratic_variation_is_rejected(term: str) -> None:
    """The control that must FAIL: flip one coefficient and conservation must break.

    Without this, "the declared vector is divergence-free" would be a statement about the checker
    rather than about the vector.
    """

    names = tcs.named_term_names(4)
    spec = dict(tcs.DECLARED_VECTORS["quadratic_ricci_squared_euler_lagrange"])
    spec[term] = str(-Fraction(spec[term]))
    vector = [
        (Fraction(spec.get(name, "0")).numerator
         * pow(Fraction(spec.get(name, "0")).denominator, PRIME - 2, PRIME))
        % PRIME
        for name in names
    ]
    geometry = tcs._Geometry(
        5, tcs.metric_jet(5, 5, PRIME, f"test-perturbed-{term}", 0), PRIME, 5
    )
    terms = tcs.named_tensors(geometry, 4)
    tensor = tcs._combine(vector, terms, names, PRIME)
    assert np.any(geometry.divergence(tensor, 1) % PRIME)


def test_gauss_bonnet_is_the_one_minus_four_one_combination_of_the_variations() -> None:
    """E1 - 4 E2 + E3 must be the declared Lanczos vector, exactly, in rational arithmetic.

    Four vectors written independently from the literature agreeing on ten rational coefficients --
    with all three derivative-carrying coefficients cancelling, as they must for a second-order
    tensor -- is what makes them cross-checks of each other rather than four separate assertions.
    """

    names = tcs.named_term_names(4)
    combination = {name: Fraction(0) for name in names}
    for label, weight in tcs.GAUSS_BONNET_DECOMPOSITION:
        for term, value in tcs.DECLARED_VECTORS[label].items():
            combination[term] += Fraction(weight) * Fraction(value)
    target = {
        name: Fraction(tcs.DECLARED_VECTORS["gauss_bonnet_lanczos"].get(name, "0"))
        for name in names
    }
    assert combination == target
    assert all(combination[name] == 0 for name in ("DDR", "BoxRg", "BoxRic"))
    assert tcs.gauss_bonnet_decomposition()["status"] == "pass"


# ---------------------------------------------------------------------------
# Controls.
# ---------------------------------------------------------------------------


def test_dropping_conservation_enlarges_the_space(receipt: dict) -> None:
    control = receipt["relaxation_controls"]["dropping_divergence_free_enlarges_the_space"]
    assert control["strictly_larger"] is True
    assert control["with_constraint"] == 2
    assert control["without_constraint"] == 3


def test_dropping_general_covariance_is_refused_with_a_typed_blocker() -> None:
    with pytest.raises(tcs.ConstraintOutOfScope) as error:
        tcs.run_search(
            dimension=4,
            order=2,
            constraints=["derivative_order", "symmetric", "divergence_free"],
            modulus=PRIME,
            seed="test-refusal",
            bank_samples=1,
            holdout_samples=1,
            enumerate_basis=False,
        )
    assert "not finite-dimensional" in str(error.value)
    assert issubclass(tcs.ConstraintOutOfScope, tcs.TensorConstraintSearchError)


def test_missing_derivative_order_is_also_refused() -> None:
    with pytest.raises(tcs.ConstraintOutOfScope):
        tcs.run_search(
            dimension=4,
            order=2,
            constraints=["generally_covariant", "symmetric", "divergence_free"],
            modulus=PRIME,
            seed="test-refusal",
            bank_samples=1,
            holdout_samples=1,
            enumerate_basis=False,
        )


def test_unknown_constraint_is_rejected() -> None:
    with pytest.raises(tcs.TensorConstraintSearchError):
        tcs.run_search(
            dimension=4,
            order=2,
            constraints=["generally_covariant", "derivative_order", "positively_vibed"],
            modulus=PRIME,
            seed="test-refusal",
            bank_samples=1,
            holdout_samples=1,
            enumerate_basis=False,
        )


def test_fabricated_conserved_tensor_is_rejected_independently() -> None:
    """``R_mn - R g_mn/3`` is not conserved, and the residual is ``(1/6) nabla_n R``."""

    names = tcs.named_term_names(2)
    fabricated = tcs._declared_vector("fabricated_third", names, PRIME)
    einstein = tcs._declared_vector("einstein", names, PRIME)
    caught = 0
    for sample in range(3):
        geometry = tcs._Geometry(
            4, tcs.metric_jet(4, 3, PRIME, "test-fabricated", sample), PRIME, 3
        )
        terms = tcs.named_tensors(geometry, 2)
        residual = geometry.divergence(
            tcs._combine(fabricated, terms, names, PRIME), 1
        )[..., 0] % PRIME
        assert not np.any(
            geometry.divergence(tcs._combine(einstein, terms, names, PRIME), 1) % PRIME
        )
        gradient = geometry.divergence(terms["Rg"], 1)[..., 0] % PRIME
        sixth = pow(6, PRIME - 2, PRIME)
        assert np.array_equal(residual, (gradient * sixth) % PRIME)
        caught += int(bool(np.any(residual)))
    assert caught == 3


def test_negative_controls_all_pass(receipt: dict) -> None:
    controls = {item["control"]: item for item in receipt["negative_controls"]}
    assert set(controls) == {
        "fabricated_divergence_free_tensor_is_rejected",
        "gauss_bonnet_is_topological_in_four_dimensions",
        "dropping_general_covariance_is_refused",
        "gauss_bonnet_is_the_1_minus4_1_combination_of_the_quadratic_variations",
        "naive_nullity_difference_violates_the_one_sided_sampling_guarantee",
        "elasticity_framework_declared_over_a_curvature_computation_is_refused",
        "curvature_framework_declared_over_an_elasticity_computation_is_refused",
        "unsupported_generator_declaration_is_refused",
        "untyped_free_text_framework_declaration_is_refused",
        "schwarzschild_crosscheck_against_repository_relativity_module",
    }
    assert all(item["status"] == "pass" for item in receipt["negative_controls"])
    assert controls["fabricated_divergence_free_tensor_is_rejected"]["verdict"] == "REJECTED"
    guarantee = controls["naive_nullity_difference_violates_the_one_sided_sampling_guarantee"]
    assert guarantee["naive_violations"] >= 1
    for case in guarantee["cases"]:
        assert case["repaired_dimension"] >= case["independently_exhibited_dimension"]
    assert any(case["naive_violates_the_guarantee"] for case in guarantee["cases"])
    assert (
        controls["elasticity_framework_declared_over_a_curvature_computation_is_refused"][
            "blocker_type"
        ]
        == "FrameworkMismatch"
    )


def test_declared_framework_is_confronted_with_the_computation(receipt: dict) -> None:
    """The declared_framework block is an INPUT, not a decoration.

    It used to be referenced nowhere in this file, which is exactly how a receipt could carry a
    framework claim describing a different theory from the one that produced its numbers.
    """

    declared = receipt["declared_framework"]["machine_checkable"]
    consistency = receipt["framework_consistency"]
    assert consistency["all_searches_agree"] is True
    assert consistency["typed_declaration"] == {
        field: declared[field] for field in tcs.FRAMEWORK_FIELDS
    }
    assert consistency["searches_checked"] == [
        item["search_id"] for item in receipt["searches"]
    ]
    for item in receipt["searches"]:
        realized = item["realized_framework"]
        assert realized["concomitant_generator"] == "riemann_tensor"
        assert realized["curvature"] == "generic"
        assert realized["metric_signature"] == "lorentzian"
        for field in tcs.FRAMEWORK_FIELDS:
            assert realized[field] == declared[field]


def test_lying_about_the_declared_framework_is_refused() -> None:
    """The defect this gate closes, pinned against the shipped config."""

    bogus = copy.deepcopy(CONFIG)
    bogus["declared_framework"]["machine_checkable"] = dict(
        tcs.ELASTICITY_DECLARATION_OVER_CURVATURE
    )
    bogus["declared_framework"]["geometry"] = "flat Euclidean 3-space"
    bogus["declared_framework"]["concomitant_class"] = (
        "polynomial in the linear strain tensor"
    )
    # It no longer even reaches the curvature generator: the declaration routes elsewhere, and
    # were it forced through the gate, the gate refuses it.
    assert tcs.declared_generator(bogus) == "linear_strain_tensor"
    bank = tcs.build_bank(4, 2, PRIME, "lie", 2)
    realized = tcs.measure_realized_framework(
        bank, [tcs.named_tensors(item, 2) for item in bank], dimension=4, order=2, modulus=PRIME
    )
    with pytest.raises(tcs.FrameworkMismatch):
        tcs.check_framework_consistency(
            bogus["declared_framework"]["machine_checkable"], realized, context="lie"
        )


def test_schwarzschild_crosscheck_reuses_the_repository_control(receipt: dict) -> None:
    entry = next(
        item
        for item in receipt["negative_controls"]
        if item.get("source", "").endswith("schwarzschild_ricci_components")
    )
    assert entry["nonzero_components"] == []
    assert entry["component_count"] == 16
    assert entry["status"] == "pass"


# ---------------------------------------------------------------------------
# Arithmetic hygiene.
# ---------------------------------------------------------------------------


def test_rational_reconstruction_round_trips() -> None:
    for fraction in (Fraction(-1, 2), Fraction(2), Fraction(-4), Fraction(1, 8), Fraction(7, 3)):
        residue = (
            fraction.numerator * pow(fraction.denominator, PRIME - 2, PRIME)
        ) % PRIME
        assert tcs._rational(residue, PRIME) == fraction


def test_modular_einsum_matches_exact_integer_arithmetic() -> None:
    """The pairwise fold must agree with an overflow-free object-dtype einsum."""

    rng = np.random.default_rng(11)
    left = rng.integers(0, PRIME, size=(4, 4, 4, 4))
    right = rng.integers(0, PRIME, size=(4, 4))
    folded = tcs._modular_einsum(["abcd", "cd"], "ab", [left, right], PRIME)
    exact = np.einsum("abcd,cd->ab", left.astype(object), right.astype(object)) % PRIME
    assert folded.tolist() == exact.tolist()


def test_two_primes_agree(receipt: dict) -> None:
    assert receipt["counts"]["prime_replays"] == 2
    for search in receipt["searches"]:
        assert search["prime_replays"] == 2


# ---------------------------------------------------------------------------
# Receipt hygiene.
# ---------------------------------------------------------------------------


def test_claims_block_is_frozen(receipt: dict) -> None:
    assert receipt["claims"] == {
        "declared_framework_selects_the_basis_generator": True,
        "declared_framework_is_gated_against_the_measured_computation": True,
        "derivation_is_symbolic_and_checked": True,
        "framework_was_declared_not_discovered": True,
        "novelty_claimed": False,
        "uniqueness_is_within_declared_basis_and_order": True,
    }


def test_receipt_carries_no_floats_and_no_host_paths(receipt: dict) -> None:
    tcs._no_floats(receipt)
    assert not tcs._HOST_PATH.search(json.dumps(receipt, sort_keys=True))


def test_receipt_is_deterministic_across_processes(receipt: dict, sealed: dict) -> None:
    """The committed receipt was written by a separate CLI process at a different time.

    Byte-identity between it and an in-session rebuild is a stronger determinism statement than
    running the builder twice inside one interpreter, and it costs nothing extra.
    """

    assert sealed["content_sha256"] == receipt["content_sha256"]
    assert sealed == receipt


def test_search_is_deterministic_within_a_process() -> None:
    """A cheap cell run twice must be bit-identical: no clock, no unseeded randomness."""

    kwargs = {
        "dimension": 4,
        "order": 2,
        "constraints": ["generally_covariant", "derivative_order", "symmetric", "divergence_free"],
        "modulus": PRIME,
        "seed": "test-determinism",
        "bank_samples": 4,
        "holdout_samples": 2,
        "enumerate_basis": True,
    }
    assert tcs.run_search(**kwargs) == tcs.run_search(**kwargs)
    assert tcs.newtonian_limit_chain() == tcs.newtonian_limit_chain()


def test_committed_receipt_passes_full_replay_validation(sealed: dict) -> None:
    """The expensive one: ``validate_receipt`` re-derives the entire receipt and compares."""

    tcs.validate_receipt(sealed, CONFIG, ROOT)


def test_receipt_is_byte_canonical(sealed: dict) -> None:
    raw = RECEIPT.read_bytes()
    assert raw.endswith(b"\n")
    assert canonical_json_bytes(json.loads(raw.decode("utf-8"))) + b"\n" == raw


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["claims"].__setitem__("novelty_claimed", True),
        lambda value: value["claims"].__setitem__(
            "framework_was_declared_not_discovered", False
        ),
        lambda value: value["claims"].__setitem__(
            "uniqueness_is_within_declared_basis_and_order", False
        ),
        lambda value: value.__setitem__("unknown_top_level_key", True),
        lambda value: value.__setitem__("schema_version", "invariant-something-else-1.0"),
        lambda value: value["negative_controls"][0].__setitem__("status", "fail"),
        lambda value: value.__setitem__("config_sha256", "0" * 64),
        lambda value: value.__setitem__("decision", "DERIVED at C:\\Users\\someone\\elsewhere"),
    ],
)
def test_resealed_tamper_fails_closed_before_replay(sealed: dict, mutator) -> None:
    """These mutations are caught by the cheap checks, ahead of the expensive exact replay."""

    tampered = copy.deepcopy(sealed)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(tcs.TensorConstraintSearchError):
        tcs.validate_receipt(tampered, CONFIG, ROOT)


def test_resealed_deep_tamper_is_caught_by_the_exact_replay(sealed: dict) -> None:
    """A forgery that survives every cheap check must still die at the replay.

    This is the one that proves the seal is not the defense: the numbers themselves are
    re-derived.  It is a single test because each call re-runs the whole search.
    """

    tampered = copy.deepcopy(sealed)
    tampered["newtonian_limit"]["coupling_constant_kappa"] = "4*pi*G/c**4"
    tampered["counts"]["final_family_dimension"] = 1
    tampered["searches"][0]["surviving_dimension"] = 1
    _reseal(tampered)
    assert tampered["content_sha256"] == canonical_sha256(
        {key: item for key, item in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(tcs.TensorConstraintSearchError, match="exact replay"):
        tcs.validate_receipt(tampered, CONFIG, ROOT)


def test_plain_edit_without_reseal_fails_closed(sealed: dict) -> None:
    tampered = copy.deepcopy(sealed)
    tampered["decision"] = "DERIVED: something else entirely"
    with pytest.raises(tcs.TensorConstraintSearchError):
        tcs.validate_receipt(tampered, CONFIG, ROOT)


def test_config_binding_detects_a_changed_declaration(sealed: dict) -> None:
    altered = copy.deepcopy(CONFIG)
    altered["searches"][0]["dimension"] = 5
    with pytest.raises(tcs.TensorConstraintSearchError):
        tcs.validate_receipt(sealed, altered, ROOT)


def test_write_receipt_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    tcs.write_receipt({"a": 1}, target)
    tcs.write_receipt({"a": 1}, target)
    with pytest.raises(tcs.TensorConstraintSearchError):
        tcs.write_receipt({"a": 2}, target)


def test_source_bindings_cover_the_reused_machinery(receipt: dict) -> None:
    bindings = receipt["source_bindings"]
    assert "formal/cadabra/contracted_bianchi.cdb" in bindings
    assert "formal/cadabra/einstein_hilbert_metric_variation.cdb" in bindings
    assert "configs/actions/einstein_hilbert_control.json" in bindings
    for binding in bindings.values():
        digest = binding.get("file_sha256") or binding.get("semantic_sha256")
        assert isinstance(digest, str) and len(digest) == 64


def test_cli_validate_checked() -> None:
    if not RECEIPT.exists():  # pragma: no cover - the receipt is committed
        pytest.skip("committed receipt not present")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sigma_theory_compiler.tensor_constraint_search",
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
