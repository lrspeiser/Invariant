"""Gates for the C1 consumer: a search that cannot report an uncertified null.

The falsifier this file exists to keep closed is stated in the spec in one sentence: *any
exhaustive search reported as a negative result without a reachability certificate for what it
was looking for*.  Four things have to be true for it to stay closed, and each has its own
section below.

*The headline test.*  A search over a grammar that provably cannot express its target must come
back ``UNINFORMATIVE_NULL`` and must not be publishable as a negative.  It is run against all
three independent exclusion arguments -- field closure, the height ladder, and exact image
exhaustion -- because a gate that only fires on one of them is a gate with a hole.

*The control that must fail.*  A test suite in which everything returns ``UNINFORMATIVE`` would
pass while proving nothing, so the same fragment is searched for a target it *does* express and
must come back ``REAL_NEGATIVE`` instead, on a certificate carrying a constructive derivation.
The two searches differ only in their target.

*The search is a search.*  The enumerator visits every token sequence of the fragment at the
real mode C program length, and its structurally valid count is compared against the closed-form
tree count for that fragment.  Witness ordinals are re-encoded into the real 23^9 mode C space
and decoded back, so what was found is a member of the space the GPU campaign enumerates and not
of a toy alongside it.

*The label cannot be forged.*  Twelve doctored adjudications, headed by a ``REAL_NEGATIVE``
resealed onto a null over a fragment proved not to contain its target, must every one be
rejected -- and the receipt-level gate must refuse a compositional-search receipt that reports
zero candidates without a reachability block at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.certified_null_search import (
    ADJUDICATION_SCHEMA,
    DEMO_FRAGMENT,
    DEMO_KNOWN_DERIVATION,
    DEMO_REACHABLE_TARGET,
    DEMO_SEARCHES,
    EXPECTED_DEMO_OUTCOMES,
    MAX_ENUMERATED_SEQUENCES,
    MODE_C_WITNESSES,
    MODE_F_FULL,
    MODE_F_WITNESSES,
    NO_KNOWN_CLOSED_FORM,
    OUTCOME_INCONSISTENT,
    OUTCOME_POSITIVE,
    OUTCOME_REAL_NEGATIVE,
    OUTCOME_UNINFORMATIVE,
    REASON_OUTSIDE_SPACE,
    REASON_REACHABILITY_UNRESOLVED,
    RECEIPT_SCHEMA,
    SERIES_INDEX,
    SearchDeclaration,
    _outcome,
    adjudicate_campaign_nulls,
    build_receipt,
    campaign_reachability_block,
    campaign_reachability_certificates,
    certificate_for,
    certified_null_search_controls,
    certified_search,
    exhaustive_fragment_search,
    main,
    reduce_series_or_limit,
    run_demo_searches,
    series_limit_reachability_certificate,
    validate_receipt,
    verify_adjudication,
    verify_campaign_reachability_block,
    verify_series_limit_certificate,
)
from sigma_theory_compiler.compositional_expression_search import (
    CLAIMS,
    MODE_CONFIG,
    REAL_TARGETS,
    RESULT_SCHEMA,
    SYMBOLIC_TARGETS,
    CompositionalSearchError,
    decode_ordinal,
    program_status,
    to_sympy,
)
from sigma_theory_compiler.compositional_expression_search import (
    validate_receipt as validate_campaign_receipt,
)
from sigma_theory_compiler.reachability_certificate import (
    MODE_C_FULL,
    MODE_C_RATIONAL,
    VERDICT_OUTSIDE,
    VERDICT_REACHABLE,
    VERDICT_UNRESOLVED,
    Fragment,
    tokens_from_rpn,
    tree_counts_depth_free,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adjudications() -> dict[str, dict]:
    return {
        str(item["search"]["declaration"]["name"]): item for item in run_demo_searches()
    }


@pytest.fixture(scope="module")
def controls() -> dict:
    return certified_null_search_controls()


@pytest.fixture(scope="module")
def campaign_block() -> dict:
    return campaign_reachability_block({"C": {}, "F": {}})


# ---------------------------------------------------------------------------
# The headline test: a grammar that cannot express the target returns UNINFORMATIVE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "target_kind", "argument"),
    [
        ("sqrt2", "symbolic", "field_closure"),
        ("1/9973", "rational", "height_bound"),
        ("355/113", "rational", "exact_image_exhaustion"),
    ],
)
def test_search_over_a_grammar_that_cannot_express_the_target_is_uninformative(
    target: str, target_kind: str, argument: str
) -> None:
    """The whole point of C1, on a real exhaustive search, three exclusion arguments deep.

    Each of these targets is provably outside the searched fragment for a different structural
    reason.  In every case the search really runs -- all 262,144 token sequences, 7,168 valid
    programs -- finds nothing, and is refused permission to call that a negative result.
    """

    declaration = SearchDeclaration(
        name=f"cannot_express_{argument}",
        fragment=DEMO_FRAGMENT,
        target=target,
        target_kind=target_kind,
        objective="any program of the fragment whose exact value is the target",
    )
    adjudication = certified_search(declaration)

    assert adjudication["outcome"] == OUTCOME_UNINFORMATIVE
    assert adjudication["outcome"] != OUTCOME_REAL_NEGATIVE
    assert adjudication["uninformative_reason"] == REASON_OUTSIDE_SPACE
    assert adjudication["publishable_as_a_negative"] is False
    assert adjudication["certificate_verdict"] == VERDICT_OUTSIDE
    assert adjudication["certificate_verified"] is True

    # The exclusion is the *declared* one, not whichever one happened to fire.
    certificate = adjudication["certificate"]
    kind = certificate.get("argument", {}).get("kind") or certificate.get("proof", {}).get("kind")
    assert kind == argument

    # And the search really was exhaustive and really found nothing.
    search = adjudication["search"]
    assert search["coverage"]["exhaustive"] is True
    assert search["coverage"]["coverage_argument_holds"] is True
    assert search["witness_count"] == 0
    assert search["accepted_count"] == 0

    verify_adjudication(adjudication)


def test_an_open_irrationality_is_unresolved_rather_than_negative() -> None:
    """Catalan's constant is not known to be irrational, so nothing may be concluded.

    This is the third way a null can be uninformative and it is the least comfortable: the
    search cannot even decide whether one of its own values equals the target.  Reporting a
    zero here would be reporting a fact about sympy's assumption database.
    """

    declaration = SearchDeclaration(
        name="catalan_is_open",
        fragment=DEMO_FRAGMENT,
        target="catalan",
        target_kind="symbolic",
        objective="any program of the fragment whose exact value is the target",
    )
    adjudication = certified_search(declaration)
    assert SYMBOLIC_TARGETS["catalan"].is_rational is None
    assert adjudication["outcome"] == OUTCOME_UNINFORMATIVE
    assert adjudication["uninformative_reason"] == REASON_REACHABILITY_UNRESOLVED
    assert adjudication["certificate_verdict"] == VERDICT_UNRESOLVED
    assert adjudication["search"]["witness_count"] is None
    assert adjudication["search"]["comparison_decided"] is False
    verify_adjudication(adjudication)


# ---------------------------------------------------------------------------
# The control that must fail: a reachable target must NOT come back uninformative
# ---------------------------------------------------------------------------


def test_a_reachable_target_yields_a_real_negative_not_an_uninformative_null() -> None:
    """Same fragment, same enumerator, same predicate; only the target differs.

    Without this, a module that returned ``UNINFORMATIVE_NULL`` unconditionally would pass every
    other test in this file.  ``1/16`` is inside the fragment, the certificate exhibits the
    derivation, and the null is therefore about the mathematics and publishable.
    """

    declaration = SearchDeclaration(
        name="new_derivation_of_a_reachable_target",
        fragment=DEMO_FRAGMENT,
        target=DEMO_REACHABLE_TARGET,
        target_kind="rational",
        objective="a derivation of the target that is not already catalogued",
        known_derivations=(DEMO_KNOWN_DERIVATION,),
    )
    adjudication = certified_search(declaration)

    assert adjudication["outcome"] == OUTCOME_REAL_NEGATIVE
    assert adjudication["outcome"] != OUTCOME_UNINFORMATIVE
    assert adjudication["uninformative_reason"] is None
    assert adjudication["publishable_as_a_negative"] is True
    assert adjudication["certificate_verdict"] == VERDICT_REACHABLE
    assert adjudication["certificate"]["derivation"]["equals_target"] is True

    search = adjudication["search"]
    assert search["witness_count"] == 1
    assert search["accepted_count"] == 0
    assert search["witnesses"][0]["rpn"] == DEMO_KNOWN_DERIVATION
    assert search["witnesses"][0]["already_catalogued"] is True
    verify_adjudication(adjudication)


def test_dropping_the_catalogue_turns_the_same_null_into_a_positive() -> None:
    """The predicate is doing real work: the only thing that changed is what counts as new."""

    declaration = SearchDeclaration(
        name="uncatalogued",
        fragment=DEMO_FRAGMENT,
        target=DEMO_REACHABLE_TARGET,
        target_kind="rational",
        objective="a derivation of the target that is not already catalogued",
    )
    adjudication = certified_search(declaration)
    assert adjudication["outcome"] == OUTCOME_POSITIVE
    assert adjudication["search"]["accepted_count"] == 1


def test_the_demonstration_set_covers_every_outcome_the_algebra_can_produce(
    adjudications: dict[str, dict],
) -> None:
    produced = {item["outcome"] for item in adjudications.values()}
    assert produced == {OUTCOME_POSITIVE, OUTCOME_REAL_NEGATIVE, OUTCOME_UNINFORMATIVE}
    for name, expected in EXPECTED_DEMO_OUTCOMES.items():
        assert adjudications[name]["outcome"] == expected


# ---------------------------------------------------------------------------
# The search is a real, counted, exhaustive search
# ---------------------------------------------------------------------------


def test_the_coverage_argument_is_a_count_not_a_spot_check() -> None:
    """Nothing was skipped, and that is compared against a closed form rather than asserted."""

    declaration = DEMO_SEARCHES[0]
    search = exhaustive_fragment_search(declaration)
    coverage = search["coverage"]
    alphabet = len(DEMO_FRAGMENT.token_indices)
    length = DEMO_FRAGMENT.program_length
    assert length == int(MODE_CONFIG["C"]["program_length"]) == 9
    assert coverage["token_sequences_enumerated"] == alphabet**length == 262_144
    assert coverage["structurally_valid_measured"] == 7_168
    assert coverage["structurally_valid_closed_form"] == tree_counts_depth_free(DEMO_FRAGMENT)[-1]
    assert coverage["coverage_argument_holds"] is True
    assert coverage["programs_with_a_value"] <= coverage["structurally_valid_measured"]


def test_witnesses_are_members_of_the_real_mode_c_ordinal_space(
    adjudications: dict[str, dict],
) -> None:
    """A find here is a find in the space the 1.9e12-candidate campaign enumerates."""

    search = adjudications["real_negative_new_derivation_of_1_16"]["search"]
    space = int(MODE_CONFIG["C"]["space_size"])
    for witness in search["witnesses"]:
        ordinal = int(witness["ordinal_in_mode_space"])
        assert 0 <= ordinal < space
        tokens = decode_ordinal(ordinal, "C")
        assert program_status(tokens) == "ok"
        assert tokens == tokens_from_rpn(str(witness["rpn"]), DEMO_FRAGMENT)
        assert Fraction(str(witness["exact_value"])) == Fraction(DEMO_REACHABLE_TARGET)


def test_the_reachable_target_really_has_exactly_one_derivation() -> None:
    """The catalogue is complete, so the REAL_NEGATIVE is not an artefact of a short listing."""

    declaration = SearchDeclaration(
        name="count_them_all",
        fragment=DEMO_FRAGMENT,
        target=DEMO_REACHABLE_TARGET,
        target_kind="rational",
        objective="every derivation of the target",
    )
    search = exhaustive_fragment_search(declaration)
    assert search["witness_count"] == 1
    assert search["witnesses_listed"] == 1
    assert search["witnesses"][0]["rpn"] == DEMO_KNOWN_DERIVATION


def test_a_fragment_too_large_to_exhaust_is_refused_by_name_not_truncated() -> None:
    """A truncated sweep cannot support a negative, so there is no partial mode."""

    declaration = SearchDeclaration(
        name="too_big",
        fragment=MODE_C_RATIONAL,
        target="1/2",
        target_kind="rational",
        objective="any program of the fragment whose exact value is the target",
    )
    space = len(MODE_C_RATIONAL.token_indices) ** MODE_C_RATIONAL.program_length
    assert space > MAX_ENUMERATED_SEQUENCES
    with pytest.raises(CompositionalSearchError, match=str(MAX_ENUMERATED_SEQUENCES)):
        exhaustive_fragment_search(declaration)


def test_a_non_rational_fragment_is_refused_by_the_exact_enumerator() -> None:
    with pytest.raises(CompositionalSearchError, match="rationally closed"):
        exhaustive_fragment_search(
            SearchDeclaration(
                name="not_q",
                fragment=MODE_C_FULL,
                target="1/2",
                target_kind="rational",
                objective="any program",
            )
        )


# ---------------------------------------------------------------------------
# The algebra itself
# ---------------------------------------------------------------------------


def test_the_outcome_table_is_total_and_has_no_fifth_answer() -> None:
    assert _outcome(VERDICT_REACHABLE, 3, 1)[0] == OUTCOME_POSITIVE
    assert _outcome(VERDICT_OUTSIDE, 0, 0)[0] == OUTCOME_UNINFORMATIVE
    assert _outcome(VERDICT_UNRESOLVED, 0, 0)[0] == OUTCOME_UNINFORMATIVE
    assert _outcome(VERDICT_UNRESOLVED, 0, 1)[0] == OUTCOME_POSITIVE
    assert _outcome(VERDICT_REACHABLE, 1, 0)[0] == OUTCOME_REAL_NEGATIVE
    assert _outcome(VERDICT_REACHABLE, 0, 0)[0] == OUTCOME_INCONSISTENT
    assert _outcome(VERDICT_OUTSIDE, 2, 0)[0] == OUTCOME_INCONSISTENT
    assert _outcome(VERDICT_UNRESOLVED, None, None)[0] == OUTCOME_UNINFORMATIVE


def test_a_contradiction_is_decided_before_anything_is_called_a_find() -> None:
    """A find that contradicts the certificate is a disagreement, not a success.

    Ordering matters here and it is the kind of ordering that is easy to get wrong: if the
    positive branch ran first, a search producing a program for a target its own certificate
    excludes would report POSITIVE and the disagreement would be buried under a success.
    """

    assert _outcome(VERDICT_OUTSIDE, 2, 1)[0] == OUTCOME_INCONSISTENT
    assert _outcome(VERDICT_OUTSIDE, 2, 2)[0] == OUTCOME_INCONSISTENT
    assert _outcome(VERDICT_REACHABLE, 0, 1)[0] == OUTCOME_INCONSISTENT


def test_a_disagreement_between_the_two_computations_fails_closed() -> None:
    """The certificate closes over values; the search enumerates programs.  If they differ,
    one of them is wrong and neither result may be published."""

    declaration = SearchDeclaration(
        name="disagreement",
        fragment=DEMO_FRAGMENT,
        target=DEMO_REACHABLE_TARGET,
        target_kind="rational",
        objective="any program of the fragment whose exact value is the target",
    )
    search = exhaustive_fragment_search(declaration)
    doctored = json.loads(json.dumps(search))
    doctored["witness_count"] = 0
    doctored["accepted_count"] = 0
    from sigma_theory_compiler.certified_null_search import adjudicate_null

    adjudication = adjudicate_null(declaration, doctored, certificate_for(declaration))
    assert adjudication["outcome"] == OUTCOME_INCONSISTENT
    with pytest.raises(CompositionalSearchError, match="two independent computations disagree"):
        raise CompositionalSearchError(adjudication["statement"])


def test_a_broken_coverage_argument_cannot_support_any_verdict() -> None:
    declaration = DEMO_SEARCHES[0]
    search = json.loads(json.dumps(exhaustive_fragment_search(declaration)))
    search["coverage"]["coverage_argument_holds"] = False
    from sigma_theory_compiler.certified_null_search import adjudicate_null

    with pytest.raises(CompositionalSearchError, match="coverage argument"):
        adjudicate_null(declaration, search, certificate_for(declaration))


# ---------------------------------------------------------------------------
# The mode F series/limit lane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("target", "witness"), sorted(MODE_F_WITNESSES.items()))
def test_mode_f_witnesses_are_proved_exactly_not_matched_to_digits(
    target: str, witness: tuple[str, str]
) -> None:
    rpn, submode = witness
    certificate = series_limit_reachability_certificate(rpn, target, submode, MODE_F_FULL)
    assert certificate["verdict"] == VERDICT_REACHABLE
    assert certificate["proof"]["kind"] == "series_limit_identity"
    assert certificate["program"]["mentions_index_variable"] is True
    assert certificate["program"]["codec_roundtrip"] is True
    ordinal = int(certificate["program"]["ordinal"])
    assert 0 <= ordinal < int(MODE_CONFIG["F"]["space_size"])
    report = verify_series_limit_certificate(certificate)
    assert report["accepted"] is True
    closed = sp.sympify(certificate["reduction"]["closed_form_printed"])
    assert sp.simplify(closed - SYMBOLIC_TARGETS[target]) == 0


def test_the_series_lane_reduces_with_the_search_modules_own_index_symbol() -> None:
    assert SERIES_INDEX == sp.Symbol("k", positive=True)
    tokens = tokens_from_rpn(MODE_F_WITNESSES["zeta2"][0], MODE_F_FULL)
    assert reduce_series_or_limit(to_sympy(tokens), "series") == sp.pi**2 / 6


def test_a_program_whose_reduction_does_not_close_claims_nothing() -> None:
    """An unevaluated ``Sum`` is not a value, and a certificate must not treat one as one.

    The program is a perfectly well-formed member of the mode F space -- valid, length eight,
    mentions ``k``, round-trips through the codec.  What it is not is *summable*: sympy leaves
    ``Sum(csc(k)**2, (k, 1, oo))`` unevaluated, and the certificate reports that rather than
    dressing an unevaluated object up as a value.
    """

    certificate = series_limit_reachability_certificate(
        "k sin recip sqr 1 mul 1 mul", "zeta2", "series", MODE_F_FULL
    )
    assert certificate["program"]["program_status"] == "ok"
    assert certificate["program"]["codec_roundtrip"] is True
    assert certificate["program"]["mentions_index_variable"] is True
    assert certificate["reduction"]["closed_form_sympy"] is None
    assert certificate["verdict"] == VERDICT_UNRESOLVED
    assert certificate["proof"]["kind"] == "none"
    assert "did not close" in certificate["proof"]["statement"]
    assert verify_series_limit_certificate(certificate)["accepted"] is True


def test_a_malformed_mode_f_program_is_unresolved_for_a_different_reason() -> None:
    certificate = series_limit_reachability_certificate(
        "k sin recip 1 mul 1 mul 1", "zeta2", "series", MODE_F_FULL
    )
    assert certificate["program"]["program_status"] != "ok"
    assert certificate["verdict"] == VERDICT_UNRESOLVED
    assert "well-formed" in certificate["proof"]["statement"]


def test_a_mode_f_certificate_cannot_be_tampered_and_resealed() -> None:
    from sigma_theory_compiler.reachability_certificate import seal_certificate

    rpn, submode = MODE_F_WITNESSES["zeta3"]
    honest = series_limit_reachability_certificate(rpn, "zeta3", submode, MODE_F_FULL)
    forged = json.loads(json.dumps(honest))
    forged["target"] = {
        "kind": "symbolic",
        "name": "zeta2",
        "sympy": sp.srepr(SYMBOLIC_TARGETS["zeta2"]),
    }
    with pytest.raises(CompositionalSearchError, match="certificate rejected"):
        verify_series_limit_certificate(seal_certificate(forged))

    unsealed = json.loads(json.dumps(honest))
    unsealed["verdict"] = VERDICT_OUTSIDE
    with pytest.raises(CompositionalSearchError, match="seal does not match"):
        verify_series_limit_certificate(unsealed)


def test_a_mode_c_program_is_refused_by_the_mode_f_lane() -> None:
    with pytest.raises(CompositionalSearchError, match="mode F"):
        series_limit_reachability_certificate(
            MODE_C_WITNESSES["pi"], "pi", "series", MODE_C_FULL
        )


# ---------------------------------------------------------------------------
# The campaign lane
# ---------------------------------------------------------------------------


def test_every_declared_target_of_every_mode_gets_a_row(campaign_block: dict) -> None:
    """A complete account of the campaign's silence: no target may be quietly left out."""

    declared = {str(row["name"]) for row in REAL_TARGETS}
    assert len(declared) == 12
    rows = campaign_block["null_adjudication"]
    assert len(rows) == 24
    assert {(row["mode"], row["target"]) for row in rows} == {
        (mode, target) for mode in ("C", "F") for target in declared
    }
    assert campaign_block["summary"]["every_null_carries_a_verdict"] is True
    assert campaign_block["summary"]["reachable_cells"] == 12
    assert campaign_block["summary"]["real_negative"] == 12
    assert campaign_block["summary"]["uninformative_null"] == 12


def test_mode_c_reachability_is_proved_for_every_declared_witness(
    campaign_block: dict,
) -> None:
    reachable = {
        row["target"]
        for row in campaign_block["null_adjudication"]
        if row["mode"] == "C" and row["certificate_verdict"] == VERDICT_REACHABLE
    }
    assert reachable == set(MODE_C_WITNESSES)
    assert len(reachable) == 9
    for record in campaign_block["certificates"]:
        if record["mode"] == "C" and record["certificate"] is not None:
            assert record["certificate_verified"] is True
            assert record["certificate"]["proof"]["kind"] == "symbolic_identity"


def test_targets_with_no_closed_form_are_named_and_stay_uninformative(
    campaign_block: dict,
) -> None:
    """`zeta3`, `catalan` and `euler_gamma` have no mode C witness, and the block says so."""

    unresolved_c = {
        row["target"]
        for row in campaign_block["null_adjudication"]
        if row["mode"] == "C" and row["certificate_verdict"] == VERDICT_UNRESOLVED
    }
    assert unresolved_c == {"zeta3", "catalan", "euler_gamma"}
    for row in campaign_block["null_adjudication"]:
        if row["certificate_verdict"] == VERDICT_UNRESOLVED:
            assert row["outcome"] == OUTCOME_UNINFORMATIVE
            assert row["publishable_as_a_negative"] is False
    assert set(NO_KNOWN_CLOSED_FORM) <= unresolved_c


def test_a_campaign_finding_flips_the_cell_to_positive() -> None:
    block = campaign_reachability_block({"C": {"pi": 2}, "F": {}})
    row = next(
        item
        for item in block["null_adjudication"]
        if item["mode"] == "C" and item["target"] == "pi"
    )
    assert row["findings"] == 2
    assert row["outcome"] == OUTCOME_POSITIVE
    verify_campaign_reachability_block(block)


def test_the_campaign_block_reverifies_every_certificate(campaign_block: dict) -> None:
    report = verify_campaign_reachability_block(campaign_block)
    assert report["accepted"] is True
    assert report["cells"] == 24
    assert report["certificates_verified"] == 12


def test_a_doctored_campaign_block_is_rejected(campaign_block: dict) -> None:
    from sigma_theory_compiler.reachability_certificate import seal_certificate

    forged = json.loads(json.dumps(campaign_block))
    row = next(
        item
        for item in forged["null_adjudication"]
        if item["certificate_verdict"] == VERDICT_UNRESOLVED
    )
    row["outcome"] = OUTCOME_REAL_NEGATIVE
    row["publishable_as_a_negative"] = True
    with pytest.raises(CompositionalSearchError, match="reachability block rejected"):
        verify_campaign_reachability_block(seal_certificate(forged))


def test_an_unverified_reachable_claim_is_refused_in_the_campaign_lane() -> None:
    records = json.loads(json.dumps(campaign_reachability_certificates()))
    target = next(item for item in records if item["verdict"] == VERDICT_REACHABLE)
    target["certificate_verified"] = False
    with pytest.raises(CompositionalSearchError, match="did not verify"):
        adjudicate_campaign_nulls(records, {"C": {}, "F": {}})


# ---------------------------------------------------------------------------
# Forgery controls
# ---------------------------------------------------------------------------


def test_all_honest_accepted_and_all_probes_rejected(controls: dict) -> None:
    assert controls["all_honest_accepted"] is True
    assert controls["all_probes_rejected"] is True
    assert controls["honest_count"] == len(DEMO_SEARCHES)
    assert controls["probe_count"] == 12


EXPECTED_PROBES = (
    "real_negative_for_a_target_outside_the_space",
    "uninformative_for_a_reachable_target",
    "positive_claimed_with_no_accepted_witness",
    "certificate_swapped_for_a_different_target",
    "certificate_swapped_for_a_different_fragment",
    "search_witness_count_inflated",
    "coverage_argument_broken_but_verdict_kept",
    "known_derivation_quietly_dropped",
    "certificate_seal_tampered",
    "unsealed_outcome_flip",
    "positive_downgraded_to_a_publishable_negative",
    "unresolved_null_relabelled_real",
)


def test_every_named_probe_is_present_and_rejected(controls: dict) -> None:
    seen = {row["probe"]: row for row in controls["probes"]}
    assert set(seen) == set(EXPECTED_PROBES)
    for name in EXPECTED_PROBES:
        assert seen[name]["rejected"] is True, name
        assert seen[name]["reason"]


def test_the_headline_forgery_is_the_lie_c1_exists_to_stop(
    adjudications: dict[str, dict],
) -> None:
    """A null over a fragment proved not to contain its target, relabelled and resealed."""

    from sigma_theory_compiler.reachability_certificate import seal_certificate

    honest = adjudications["uninformative_sqrt2_is_irrational"]
    forged = json.loads(json.dumps(honest))
    forged["outcome"] = OUTCOME_REAL_NEGATIVE
    forged["uninformative_reason"] = None
    forged["publishable_as_a_negative"] = True
    forged = seal_certificate(forged)
    # The seal is valid: the forgery is in the claim, not in the bytes.
    payload = {key: item for key, item in forged.items() if key != "content_sha256"}
    assert canonical_sha256(payload) == forged["content_sha256"]
    with pytest.raises(CompositionalSearchError, match="does not match a regeneration"):
        verify_adjudication(forged)


# ---------------------------------------------------------------------------
# The receipt-level gate on the real compositional search
# ---------------------------------------------------------------------------


def _seal_campaign_receipt(body: dict) -> dict:
    payload = {key: item for key, item in body.items() if key != "content_sha256"}
    core = {
        key: item
        for key, item in payload.items()
        if key not in {"result_core_sha256", "measurement"}
    }
    payload["result_core_sha256"] = canonical_sha256(core)
    return {**payload, "content_sha256": canonical_sha256(payload)}


def _null_campaign_receipt(**extra) -> dict:
    body = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "decoy_calibration": {"totals": {"decoy_post_gate_survivors": 0}},
        "chance_gate": {
            "C": {
                "per_target": [
                    {"mode": "C", "submode": "constant", "target": name, "role": "real"}
                    for name in ("pi", "e", "sqrt2")
                ]
            },
            "F": {"per_target": []},
        },
        "controls": {},
        "codec_validity": [],
        "headline": {"count": 0, "entries": []},
        **extra,
    }
    return _seal_campaign_receipt(body)


def test_a_campaign_receipt_reporting_a_null_without_a_certificate_is_refused() -> None:
    """This is the falsifier, wired to the gate that now refuses to publish it."""

    with pytest.raises(CompositionalSearchError, match="C1"):
        validate_campaign_receipt(_null_campaign_receipt())


def test_the_same_null_with_a_reachability_block_is_accepted(campaign_block: dict) -> None:
    validate_campaign_receipt(_null_campaign_receipt(reachability=campaign_block))


def test_a_receipt_whose_block_adjudicates_a_different_campaign_is_refused() -> None:
    block = campaign_reachability_block({"C": {"pi": 1}, "F": {}})
    with pytest.raises(CompositionalSearchError, match="headline carries"):
        validate_campaign_receipt(_null_campaign_receipt(reachability=block))


def test_a_receipt_missing_an_adjudication_for_a_searched_target_is_refused(
    campaign_block: dict,
) -> None:
    receipt = json.loads(json.dumps(_null_campaign_receipt(reachability=campaign_block)))
    receipt["chance_gate"]["C"]["per_target"].append(
        {"mode": "C", "submode": "constant", "target": "not_a_declared_target", "role": "real"}
    )
    with pytest.raises(CompositionalSearchError, match="no reachability adjudication"):
        validate_campaign_receipt(_seal_campaign_receipt(receipt))


def test_run_campaign_attaches_the_block_and_binds_it_into_the_core_hash() -> None:
    """The wiring is in the sealed body, not in a side file: the core hash covers it."""

    import inspect

    from sigma_theory_compiler import compositional_expression_search as search_module

    source = inspect.getsource(search_module.run_campaign)
    assert "campaign_reachability_block" in source
    assert '"reachability": reachability,' in source
    core_source = inspect.getsource(search_module.validate_receipt)
    assert "_validate_reachability" in core_source


REPLAY_RECEIPT = (
    REPO_ROOT / "runs" / "math" / "compositional" / "c1-wiring-replay-reduced-budget.json"
)


@pytest.mark.skipif(not REPLAY_RECEIPT.exists(), reason="no C1 wiring replay in the tree")
def test_a_real_campaign_receipt_carries_the_block_and_adjudicates_every_target() -> None:
    """Evidence that the wiring runs, on the real 1.9e12-candidate sweep rather than a fixture.

    This receipt is a *reduced-budget* replay -- corpus screening skipped, verification budgets
    cut to 8 and 4 -- so its headline is not comparable to the campaign of record and it is not
    a replacement for ``search-v1.json``.  What it does establish is the only thing this test
    claims: an exhaustive sweep of the whole declared space produced a receipt whose nulls are
    each labelled from a verified certificate, and the receipt passes its own validator.
    """

    receipt = json.loads(REPLAY_RECEIPT.read_text(encoding="utf-8"))
    validate_campaign_receipt(receipt)

    assert receipt["campaign"]["exhaustive_over_declared_space"] == {"C": True, "F": True}
    assert receipt["campaign"]["enumerated_total"] >= 10**12
    assert receipt["corpus"]["available"] is False, "the replay is reduced-fidelity by design"

    block = receipt["reachability"]
    verify_campaign_reachability_block(block)
    rows = block["null_adjudication"]
    assert len(rows) == 24
    assert {row["outcome"] for row in rows} <= {
        OUTCOME_POSITIVE,
        OUTCOME_REAL_NEGATIVE,
        OUTCOME_UNINFORMATIVE,
    }
    # Every cell reporting nothing carries a verdict, and only the reachable ones may be
    # published as negatives.
    for row in rows:
        assert row["certificate_verdict"] in {VERDICT_REACHABLE, VERDICT_UNRESOLVED}
        if row["findings"] == 0:
            expected = (
                OUTCOME_REAL_NEGATIVE
                if row["certificate_verdict"] == VERDICT_REACHABLE
                else OUTCOME_UNINFORMATIVE
            )
            assert row["outcome"] == expected, row
    assert block["summary"]["real_negative"] >= 1
    assert block["summary"]["uninformative_null"] >= 1


@pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "math" / "compositional" / "search-v1.json").exists(),
    reason="no campaign receipt in the tree",
)
def test_the_pinned_campaign_receipt_would_be_refused_if_its_headline_were_empty() -> None:
    """The pinned receipt predates the wiring; it publishes a positive, so the gate lets it
    through.  Emptying its headline turns it into exactly the thing C1 forbids, and the gate
    then refuses it -- on the real receipt, not a synthetic one."""

    path = REPO_ROOT / "runs" / "math" / "compositional" / "search-v1.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_campaign_receipt(receipt)
    assert "reachability" not in receipt
    assert receipt["headline"]["count"] == 1

    emptied = json.loads(json.dumps(receipt))
    emptied["headline"] = {"count": 0, "entries": [], "meaning": receipt["headline"]["meaning"]}
    with pytest.raises(CompositionalSearchError, match="C1"):
        validate_campaign_receipt(_seal_campaign_receipt(emptied))


# ---------------------------------------------------------------------------
# Exactness, determinism, receipt, CLI
# ---------------------------------------------------------------------------


def _walk(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    else:
        yield path, value


def test_no_adjudication_carries_a_float(adjudications: dict[str, dict], campaign_block) -> None:
    for adjudication in adjudications.values():
        for path, item in _walk(adjudication):
            assert not isinstance(item, float), path
    for path, item in _walk(campaign_block):
        assert not isinstance(item, float), path


def test_adjudications_are_deterministic() -> None:
    first = run_demo_searches()
    second = run_demo_searches()
    assert [item["content_sha256"] for item in first] == [
        item["content_sha256"] for item in second
    ]
    assert (
        campaign_reachability_block({"C": {}, "F": {}})["content_sha256"]
        == campaign_reachability_block({"C": {}, "F": {}})["content_sha256"]
    )


def test_receipt_seals_validates_and_decides() -> None:
    receipt = build_receipt()
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["decision"] == "CERTIFIED"
    assert receipt["demonstration_outcomes"] == EXPECTED_DEMO_OUTCOMES
    validate_receipt(receipt)

    tampered = json.loads(json.dumps(receipt))
    tampered["decision"] = "CERTIFIED_ANYWAY"
    with pytest.raises(CompositionalSearchError, match="seal"):
        validate_receipt(tampered)


def test_adjudication_schema_is_pinned(adjudications: dict[str, dict]) -> None:
    for adjudication in adjudications.values():
        assert adjudication["schema_version"] == ADJUDICATION_SCHEMA


def test_cli_writes_a_sealed_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert main(["--output", str(output)]) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    assert receipt["decision"] == "CERTIFIED"


def test_module_runs_as_a_script() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "sigma_theory_compiler.certified_null_search", "--summary"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["decision"] == "CERTIFIED"
    assert payload["campaign_summary"]["cells"] == 24


def test_a_fragment_declaration_binds_the_certificate_to_the_searched_space() -> None:
    """A true certificate about a *different* fragment is the cheapest forgery there is."""

    from sigma_theory_compiler.certified_null_search import adjudicate_null
    from sigma_theory_compiler.reachability_certificate import (
        rational_reachability_certificate,
    )

    declaration = SearchDeclaration(
        name="binding",
        fragment=DEMO_FRAGMENT,
        target="355/113",
        target_kind="rational",
        objective="any program of the fragment whose exact value is the target",
    )
    search = exhaustive_fragment_search(declaration)
    wider = Fragment("tiny_neg_recip_div", "C", ("1", "2", "neg", "recip", "div"), "Q")
    with pytest.raises(CompositionalSearchError, match="different fragment"):
        adjudicate_null(
            declaration, search, rational_reachability_certificate("355/113", wider)
        )
