"""Tests for the continued-fraction proof router.

The load-bearing checks are the ones that could let a wrong answer through:

* a rational Riccati solution must be substituted back and verified, and a recurrence with no
  hypergeometric solution (Lambert's ``coth``, whose recurrence is Bessel's) must come back
  with a typed blocker rather than a proof;
* a closed form for a hypergeometric series must agree with the series *summed directly* --
  outside its disc of convergence the Meijer-G closed form is an analytic continuation and
  the identity would be false;
* a deliberately falsified identity must be refuted, and the receipt must carry the decimal
  place where it breaks;
* proving a candidate by a classical technique must *reclassify it as known*, because a proof
  that exhibits a continued fraction as an instance of a cited family is prior art;
* the headline -- proved and still absent from the corpus -- must be recomputed from the
  per-candidate reclassifications, so it cannot be asserted independently of them.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from sigma_theory_compiler.cf_prior_art_corpus import (
    CFPattern,
    Poly,
    Rat,
    SeqSpec,
    load_corpus,
    seq_from_poly,
)
from sigma_theory_compiler.cf_proof_router import (
    ABSENT_TECHNIQUE_CONTROL,
    CLASSICAL_CONTROLS,
    LEAN_OBSTRUCTION,
    ROUTER_CLAIMS,
    TECHNIQUE_ORDER,
    VERDICTS,
    ProofRouterError,
    RouterContext,
    _attempt_pincherle,
    _control_candidate,
    _falsification_control,
    euler_minding_factorisation,
    evaluate_hypergeometric,
    first_differing_decimal,
    gauss_parameters,
    hypergeometric_series_value,
    lean_emission,
    load_router_candidates,
    numeric_check,
    pfq_parameters,
    pincherle_value,
    rational_riccati_solutions,
    route_candidate,
    run_controls,
    run_router,
    series_numeric_sum,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

REPO = Path(__file__).resolve().parents[1]
DATABASE = REPO / "runs/math/prior-art/cf-corpus-v1.sqlite"
MANIFEST = REPO / "runs/math/prior-art/cf-corpus-v1-manifest.json"
ADJUDICATION = REPO / "runs/math/prior-art/cf-adjudication-v1.json"
ROUTING = REPO / "runs/math/prior-art/cf-proof-routing-v1.json"

_N = sp.Symbol("n")
_K = sp.Symbol("k")
_M = sp.Symbol("m")


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DATABASE, MANIFEST)


@pytest.fixture(scope="module")
def adjudication():
    return json.loads(ADJUDICATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def routing_receipt():
    return json.loads(ROUTING.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Technique 4 primitives: Riccati, Pincherle
# ---------------------------------------------------------------------------


def test_riccati_solutions_are_substituted_back_and_verified() -> None:
    a = _N + 2
    b = -_N
    solutions = rational_riccati_solutions(a, b)
    assert solutions
    for solution in solutions:
        residual = sp.simplify(
            solution * solution.subs(_N, _N - 1) - a * solution.subs(_N, _N - 1) - b
        )
        assert residual == 0
    assert any(sp.simplify(item - (_N + 1)) == 0 for item in solutions)


def test_bessel_recurrence_has_no_hypergeometric_solution() -> None:
    # Lambert's coth continued fraction: y_n = (2n + 1) y_{n-1} + y_{n-2}.
    assert rational_riccati_solutions(2 * _N + 1, sp.Integer(1)) == []


def test_pincherle_recovers_a_classical_e_continued_fraction() -> None:
    value, steps = pincherle_value((2, 1, 0), (0, -1, 0))
    assert value is not None
    assert sp.simplify(value - sp.E / (sp.E - 1)) == 0
    assert steps["casoratian"].startswith("C_n = ")
    assert "pincherle_identity" in steps


def test_pincherle_names_its_blocker_when_no_technique_applies() -> None:
    value, steps = pincherle_value((1, 2, 0), (1, 0, 0))
    assert value is None
    assert steps["blocker"].startswith("missing_proof_technique:")


# ---------------------------------------------------------------------------
# Technique 2 primitives: the Euler-Minding factorisation
# ---------------------------------------------------------------------------


def test_euler_minding_factorisation_is_an_exact_polynomial_identity() -> None:
    alpha, beta = (-4, -3, 0), (0, -2, -2)
    factorisation = euler_minding_factorisation(alpha, beta)
    assert factorisation is not None
    p, q = factorisation
    a = sum(sp.Integer(c) * _N**i for i, c in enumerate(alpha))
    b = sum(sp.Integer(c) * _N**i for i, c in enumerate(beta))
    assert sp.expand(p + q - a) == 0
    assert sp.expand(p * q.subs(_N, _N - 1) + b) == 0


def test_euler_minding_factorisation_declines_when_none_exists() -> None:
    # a_n = 3n + 3, b_n = -2n^2 has no (p, q) over the rationals; the candidate is still
    # provable, but only after an equivalence transformation, which technique 4 supplies.
    assert euler_minding_factorisation((3, 3, 0), (0, 0, -2)) is None


# ---------------------------------------------------------------------------
# Hypergeometric evaluation and its guard
# ---------------------------------------------------------------------------


def test_pfq_parameters_are_read_off_the_term_ratio() -> None:
    parameters = pfq_parameters(sp.cancel((_M + 1) / (2 * (_M + 3))))
    assert parameters is not None
    a_list, b_list, argument = parameters
    assert sorted(map(str, a_list)) == ["1", "1"]
    assert list(map(str, b_list)) == ["3"]
    assert argument == sp.Rational(1, 2)


def test_euler_integral_reduction_evaluates_a_3f2_that_hyperexpand_cannot() -> None:
    a_list = (sp.Integer(1), sp.Integer(1), sp.Integer(1))
    b_list = (sp.Rational(3, 2), sp.Integer(3))
    value, reduction = evaluate_hypergeometric(a_list, b_list, sp.Rational(1, 2))
    assert reduction == "euler_integral_reduction"
    assert sp.simplify(value - (sp.pi - 2)) == 0


def test_shift_orbit_recombination_sums_the_e_family_series() -> None:
    # T_{k+1}/T_k = (k^2+k+1)/((k+2)(k^2+5k+7)): an irreducible quadratic and its shift by 2.
    ratio = sp.cancel((_K**2 + _K + 1) / ((_K + 2) * (_K**2 + 5 * _K + 7)))
    value, reduction, detail = hypergeometric_series_value(ratio, 0)
    assert value is not None
    assert reduction.startswith("shift_orbit_recombination")
    assert sp.simplify(value - 3 * (sp.E - 2) / 2) == 0
    assert detail["shift_orbit"]["irreducible_factor"] == "k**2 + k + 1"


def test_series_guard_rejects_an_analytic_continuation_outside_convergence() -> None:
    # ratio -> 2, so the series diverges even though 1F0(1;;2) "evaluates" to -1.
    diverging = sp.Integer(2) * (_K + 1) / (_K + 1)
    assert series_numeric_sum(diverging, 1) is None
    value, reduction, _ = hypergeometric_series_value(diverging, 1)
    assert value is None
    assert "disagrees_with_the_summed_series" in reduction


def test_summed_series_matches_the_closed_form_it_certifies() -> None:
    ratio = sp.cancel((_K) / (2 * (_K + 2)))
    value, _, _ = hypergeometric_series_value(ratio, 1)
    assert value is not None
    with mp.workdps(40):
        summed = series_numeric_sum(ratio, 1)
        assert abs(summed - mp.mpf(str(sp.N(value, 40)))) < mp.mpf("1e-30")


# ---------------------------------------------------------------------------
# Technique 3: Gauss's continued fraction, solved parametrically
# ---------------------------------------------------------------------------


def _gauss_pattern(a: Fraction, b: Fraction, c: Fraction, z: Fraction) -> CFPattern:
    even = Rat.of(
        Poly.linear(Fraction(1, 2), a - 1) * Poly.linear(Fraction(1, 2), c - b - 1),
        Poly.linear(1, c - 2) * Poly.linear(1, c - 1),
    )
    odd = Rat.of(
        Poly.linear(Fraction(1, 2), b - Fraction(1, 2))
        * Poly.linear(Fraction(1, 2), c - a - Fraction(1, 2)),
        Poly.linear(1, c - 2) * Poly.linear(1, c - 1),
    )
    numerators = SeqSpec.build(
        2, (Rat.constant(-z) * even, Rat.constant(-z) * odd), {1: Fraction(1)}
    )
    return CFPattern(seq_from_poly(Poly.constant(1), {0: Fraction(0)}), numerators)


@pytest.mark.parametrize(
    ("a", "b", "c", "z"),
    [
        (Fraction(1), Fraction(1), Fraction(2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(3, 2), Fraction(-1)),
        (Fraction(2), Fraction(1), Fraction(3), Fraction(1, 3)),
    ],
)
def test_gauss_parameters_are_solved_not_looked_up(a, b, c, z) -> None:
    solved = gauss_parameters(_gauss_pattern(a, b, c, z))
    assert solved is not None
    assert (solved["a"], solved["b"], solved["c"], solved["z"]) == (
        sp.Rational(a),
        sp.Rational(b),
        sp.Rational(c),
        sp.Rational(z),
    )


def test_gauss_matcher_declines_a_period_one_pattern() -> None:
    pattern = CFPattern(seq_from_poly(Poly.of(4, 3, 0)), seq_from_poly(Poly.of(0, -2, -2)))
    assert gauss_parameters(pattern) is None


# ---------------------------------------------------------------------------
# Refutation
# ---------------------------------------------------------------------------


def test_first_differing_decimal_locates_the_break() -> None:
    assert first_differing_decimal(mp.mpf("1.2345"), mp.mpf("1.2345"), 50) is None
    assert first_differing_decimal(mp.mpf("1.2345"), mp.mpf("1.2445"), 50) == 2


def test_a_deliberately_false_identity_is_refuted_with_a_digit() -> None:
    control = _falsification_control()
    assert control["truthful_claim_holds"] is True
    assert control["refuted"] is True
    assert isinstance(control["first_differing_decimal_place"], int)
    assert control["first_differing_decimal_place"] >= 0


def test_numeric_check_passes_the_truthful_claim() -> None:
    candidate = _control_candidate(CLASSICAL_CONTROLS[0])
    assert numeric_check(candidate)["holds"] is True


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_classical_identities_are_proved_end_to_end(corpus) -> None:
    controls = run_controls(RouterContext(corpus=corpus))
    assert controls["passed"] is True
    assert controls["classical_identities_proved"] >= controls["classical_identities_required"]
    for item in controls["classical_identities"]:
        assert item["proved"] is True, item
        assert item["closed_form"]


def test_absent_technique_control_yields_a_typed_blocker() -> None:
    candidate = _control_candidate(ABSENT_TECHNIQUE_CONTROL)
    outcome = _attempt_pincherle(candidate)
    assert outcome["fired"] is False
    assert outcome["summary"]["blocker"].startswith("missing_proof_technique:")


# ---------------------------------------------------------------------------
# The routed receipt
# ---------------------------------------------------------------------------


def test_every_inconclusive_candidate_receives_a_terminal_verdict(routing_receipt) -> None:
    candidates = routing_receipt["candidates"]
    assert len(candidates) == 12
    for item in candidates:
        assert item["verdict"] in VERDICTS
        assert item["prior_art_verdict"] == "INCONCLUSIVE_VALUE_MATCH"
        if item["verdict"] == "PROVED":
            assert item["technique_that_fired"] in TECHNIQUE_ORDER
            assert item["derivation"]
            assert item["cited_theorem"]
        elif item["verdict"] == "MISSING_TECHNIQUE":
            assert item["missing_proof_technique"].startswith("missing_proof_technique:")
        else:
            assert isinstance(item["refutation"]["first_differing_decimal_place"], int)


def test_every_already_known_candidate_is_reproved_by_equivalence(routing_receipt) -> None:
    controls = routing_receipt["controls"]
    assert controls["already_known_population"] == 20
    assert controls["already_known_reproved_by_equivalence"] == 20
    for item in routing_receipt["already_known_summaries"]:
        assert item["verdict"] == "PROVED"
        assert item["technique_that_fired"] == "equivalence_transformation"
        assert item["matched_record_id"]


def test_technique_order_is_respected(routing_receipt) -> None:
    for item in routing_receipt["candidates"]:
        attempted = [entry["technique"] for entry in item["techniques_attempted"]]
        assert attempted == list(TECHNIQUE_ORDER[: len(attempted)])
        fired = [entry for entry in item["techniques_attempted"] if entry["fired"]]
        assert len(fired) <= 1
        for entry in item["techniques_attempted"][:-1]:
            assert entry["fired"] is False
            assert entry["blocker"]


def test_a_proof_by_a_classical_family_reclassifies_the_candidate_as_known(
    routing_receipt,
) -> None:
    proved = [item for item in routing_receipt["candidates"] if item["verdict"] == "PROVED"]
    assert proved
    for item in proved:
        reclassification = item["reclassification"]
        assert reclassification["prior_art_verdict_before"] == "INCONCLUSIVE_VALUE_MATCH"
        if reclassification["proof_family_present_in_corpus"]:
            assert reclassification["prior_art_verdict_after"] == "KNOWN_BY_PROOF_FAMILY"
            assert reclassification["reclassified"] is True
            assert reclassification["proof_family_exemplar"]["citation"]["reference"]


def test_headline_is_recomputed_from_the_reclassifications(routing_receipt) -> None:
    absent = [
        item["candidate_id"]
        for item in routing_receipt["candidates"]
        if item["verdict"] == "PROVED"
        and item["reclassification"]["proof_family_present_in_corpus"] is not True
    ]
    headline = routing_receipt["headline"]
    assert headline["proved_and_still_absent_from_the_corpus"] == len(absent)
    assert sorted(headline["candidate_ids"]) == sorted(absent)


def test_no_novelty_is_claimed(routing_receipt) -> None:
    assert routing_receipt["claims"] == ROUTER_CLAIMS
    assert routing_receipt["claims"]["novelty_claimed"] is False
    assert routing_receipt["claims"]["corpus_absence_establishes_novelty"] is False
    assert routing_receipt["claims"]["proof_by_classical_family_implies_known"] is True


def test_no_lean_source_is_emitted_and_the_obstruction_is_named(routing_receipt) -> None:
    assert routing_receipt["lean"]["sources_emitted"] == 0
    assert routing_receipt["lean"]["kernel_verification_pending"] is True
    assert routing_receipt["lean"]["obstruction"] == LEAN_OBSTRUCTION
    for item in routing_receipt["candidates"]:
        assert item["lean"]["lean_source"] is None
        assert item["lean"]["kernel_verified"] is False
        assert item["lean"]["kernel_verification_pending"] is True


def test_lean_emission_is_computed_from_the_final_step() -> None:
    emitted = lean_emission({"final_step": "closed_form_of_a_hypergeometric_series"})
    assert emitted["kernel_verified"] is False
    assert emitted["obstruction"] == LEAN_OBSTRUCTION
    with pytest.raises(ProofRouterError):
        lean_emission({"final_step": "nat_induction_over_a_nonnegative_closed_form"})


def test_receipt_binds_the_adjudication_and_corpus_artifacts(
    routing_receipt, adjudication, corpus
) -> None:
    assert routing_receipt["input"]["content_sha256"] == adjudication["content_sha256"]
    assert routing_receipt["input"]["result_core_sha256"] == adjudication["result_core_sha256"]
    assert routing_receipt["corpus"]["content_sha256"] == corpus.manifest["content_sha256"]


def test_receipt_validates(routing_receipt) -> None:
    validate_receipt(routing_receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.__setitem__("schema_version", "other"), id="schema"),
        pytest.param(
            lambda r: r["claims"].__setitem__("novelty_claimed", True), id="claims"
        ),
        pytest.param(
            lambda r: r["headline"].__setitem__("proved_and_still_absent_from_the_corpus", 3),
            id="headline",
        ),
        pytest.param(
            lambda r: r["candidates"][0].__setitem__("verdict", "PROVED_MAYBE"), id="verdict"
        ),
        pytest.param(
            lambda r: r["candidates"][0]["lean"].__setitem__("kernel_verified", True),
            id="kernel-claim",
        ),
        pytest.param(
            lambda r: r["controls"]["deliberate_falsification"].__setitem__("refuted", False),
            id="falsification",
        ),
        pytest.param(
            lambda r: r["controls"].__setitem__("already_known_reproved_by_equivalence", 1),
            id="known-controls",
        ),
        pytest.param(lambda r: r["config"].__setitem__("refutation_dps", 10), id="config"),
    ],
)
def test_receipt_tamper_is_detected(routing_receipt, mutate) -> None:
    mutated = copy.deepcopy(routing_receipt)
    mutate(mutated)
    with pytest.raises(ProofRouterError):
        validate_receipt(mutated)


def _reseal(receipt: dict) -> dict:
    """Re-seal a mutated receipt so validation must fail on *semantics*, not on the hash."""

    receipt["result_core_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"content_sha256", "result_core_sha256", "measurement"}
        }
    )
    receipt["content_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    return receipt


def test_a_proved_verdict_without_a_derivation_is_rejected(routing_receipt) -> None:
    mutated = copy.deepcopy(routing_receipt)
    subject = next(item for item in mutated["candidates"] if item["verdict"] == "PROVED")
    subject["derivation"] = {}
    with pytest.raises(ProofRouterError, match="derivation"):
        validate_receipt(_reseal(mutated))


def test_a_resealed_kernel_claim_is_still_rejected(routing_receipt) -> None:
    mutated = copy.deepcopy(routing_receipt)
    mutated["candidates"][0]["lean"]["kernel_verified"] = True
    with pytest.raises(ProofRouterError, match="kernel"):
        validate_receipt(_reseal(mutated))


def test_a_resealed_headline_inflation_is_still_rejected(routing_receipt) -> None:
    mutated = copy.deepcopy(routing_receipt)
    subject = next(item for item in mutated["candidates"] if item["verdict"] == "PROVED")
    subject["reclassification"]["proof_family_present_in_corpus"] = False
    with pytest.raises(ProofRouterError, match="headline"):
        validate_receipt(_reseal(mutated))


def test_router_is_deterministic(corpus, adjudication) -> None:
    trimmed = {
        **adjudication,
        "candidates": adjudication["candidates"][:2],
        "control_summaries": [],
    }
    first = run_router(trimmed, corpus)
    second = run_router(trimmed, corpus)
    assert first["result_core_sha256"] == second["result_core_sha256"]


def test_router_refuses_an_empty_input(corpus, adjudication) -> None:
    with pytest.raises(ProofRouterError, match="no candidates"):
        run_router({**adjudication, "candidates": [], "control_summaries": []}, corpus)


def test_router_reproduces_a_single_candidate_from_the_receipt(
    corpus, adjudication, routing_receipt
) -> None:
    context = RouterContext(corpus=corpus)
    candidate = next(
        item
        for item in load_router_candidates(adjudication)
        if item.prior_verdict == "INCONCLUSIVE_VALUE_MATCH"
    )
    routed = route_candidate(context, candidate)
    stored = next(
        item
        for item in routing_receipt["candidates"]
        if item["candidate_id"] == candidate.candidate_id
    )
    assert routed["verdict"] == stored["verdict"]
    assert routed["technique_that_fired"] == stored["technique_that_fired"]
