"""Partial credit must add an ordering without adding a claim.

The tests split into three jobs.  First, that the ledger reads the certificates honestly: the
same obligations met, the same ones failed, with the shortfalls the sources actually publish.
Second, that the ordering is *earned* -- every strict step re-verified from the receipt with
integer arithmetic alone, and cross-checked against an independent high-precision oracle that
never touches the certificate path.  Third, that nothing about a rank can be turned into a pass.
"""

from __future__ import annotations

import copy
import json
import random
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from sigma_theory_compiler.exact_real_bracket import LESS, compare_expressions
from sigma_theory_compiler.quartic_global_h7_partial_credit import (
    BLOCK_GATE_PATH,
    FINITE_SOBOLEV_PATH,
    GLOBAL_H7_PATH,
    OBLIGATION_LEDGER,
    RANKING_KEYS,
    PartialCreditError,
    build_from_root,
    build_partial_credit,
    ledger_sha256,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "physics-language" / "quartic-global-h7-partial-credit" / "campaign.json"

#: The four coefficient classes the ledger is expected to separate, nearest first.  Written down
#: here so a change in the ordering is a test failure rather than a quietly different receipt.
EXPECTED_TIERS = (
    ("quartic-symbol-50f184dfe1a814bf", "quartic-symbol-e4a6a9193316a6ff"),
    (
        "quartic-symbol-06e267a9215345b6",
        "quartic-symbol-076dc0ba965ab63a",
        "quartic-symbol-561de1410d6cb21f",
        "quartic-symbol-f31a234e2bf7b97f",
    ),
    ("quartic-symbol-9e65901e5299a514", "quartic-symbol-fb5c20c15ce6d778"),
    (
        "quartic-symbol-317e5395817a432b",
        "quartic-symbol-5455cad9e42a0dbc",
        "quartic-symbol-8fd254934d778c28",
        "quartic-symbol-ef832e4c3b71ee42",
    ),
)


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sources() -> tuple[dict, dict, dict]:
    return _load(GLOBAL_H7_PATH), _load(FINITE_SOBOLEV_PATH), _load(BLOCK_GATE_PATH)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_from_root(ROOT)


def _by_id(result: dict) -> dict[str, dict]:
    return {record["candidate_id"]: record for record in result["candidate_records"]}


def _oracle(expression: str) -> mp.mpf:
    """An independent evaluation of a closed form; never used on the certificate path."""

    return mp.mpf(str(sp.N(sp.sympify(expression), 60)))


def _rational(text: str) -> Fraction:
    return Fraction(text)


# --------------------------------------------------------------------------------------
# The ledger reads the sources honestly
# --------------------------------------------------------------------------------------


def test_the_verdict_is_still_block_and_no_claim_is_opened(receipt: dict) -> None:
    assert receipt["decision"] == "BLOCK"
    assert receipt["counts"]["passed"] == 0
    assert receipt["counts"]["promoted"] == 0
    assert receipt["counts"]["blocked"] == 12
    assert not any(receipt["claims"].values())
    for record in receipt["candidate_records"]:
        assert record["decision"] == "BLOCK"
        assert record["rank_is_a_proof"] is False


def test_every_candidate_meets_the_same_five_and_fails_the_same_eight(receipt: dict) -> None:
    met = {tuple(record["obligations_met"]) for record in receipt["candidate_records"]}
    failed = {
        tuple(item["key"] for item in record["obligations_failed"])
        for record in receipt["candidate_records"]
    }
    assert len(met) == 1
    assert len(failed) == 1
    assert len(next(iter(met))) == 5
    assert len(next(iter(failed))) == 8
    assert set(next(iter(met))) | set(next(iter(failed))) <= {
        obligation.key for obligation in OBLIGATION_LEDGER
    }


def test_the_discrete_shortfalls_are_the_ones_the_sources_publish(receipt: dict) -> None:
    record = receipt["candidate_records"][0]
    shortfalls = {item["key"]: item["shortfall"] for item in record["obligations_failed"]}
    assert shortfalls["lower_DF_entries"] == "594"
    assert shortfalls["universal_affine_split"] == "11"
    assert shortfalls["full_direction_completion_obstruction"] == "15"
    assert record["discrete_shortfall_total"] == str(
        sum(int(value) for value in shortfalls.values())
    )
    certificate = next(
        item
        for item in _load(GLOBAL_H7_PATH)["certificates"]
        if item["candidate_id"] == record["candidate_id"]
    )
    assert certificate["good_unknown_and_source"]["lower_DF_entries_missing"] == 594


def test_the_discrete_layer_alone_would_see_twelve_identical_candidates(receipt: dict) -> None:
    audit = receipt["gradient_audit"]
    assert audit["discrete_layer_separates"] is False
    assert audit["discrete_layer_distinct_signatures"] == 1
    signatures = {
        (
            record["discrete_shortfall_total"],
            tuple(record["discrete_shortfall_vector"]),
        )
        for record in receipt["candidate_records"]
    }
    assert len(signatures) == 1


def test_the_magnitude_layer_is_what_carries_the_gradient(receipt: dict) -> None:
    audit = receipt["gradient_audit"]
    assert audit["gradient_present"] is True
    assert audit["magnitude_layer_separates"] is True
    assert audit["magnitude_layer_distinct_signatures"] == 4
    assert audit["distinct_tiers"] == 4
    assert audit["unseparated_pairs"] == []
    assert set(audit["separating_keys"]) == {
        "certified_linear_energy_growth",
        "uncancelled_slice_growth_multiplier",
    }


def test_each_margin_names_the_unmet_obligation_it_quantifies(receipt: dict) -> None:
    record = receipt["candidate_records"][0]
    attachments = {key: value["attached_to"] for key, value in record["margins"].items()}
    assert attachments["certified_linear_energy_growth"] == "closed_global_H7_inequality"
    assert attachments["unresolved_remainder_coefficient"] == "paralinearization_remainder_bound"
    assert attachments["uncancelled_slice_growth_multiplier"] == "full_tensor_cancellation"
    failed = {item["key"]: item["margins"] for item in record["obligations_failed"]}
    assert failed["closed_global_H7_inequality"] == ["certified_linear_energy_growth"]
    assert failed["full_tensor_cancellation"] == ["uncancelled_slice_growth_multiplier"]


# --------------------------------------------------------------------------------------
# The ordering is earned
# --------------------------------------------------------------------------------------


def test_the_twelve_land_in_the_expected_four_tiers(receipt: dict) -> None:
    assert tuple(tuple(tier) for tier in receipt["gradient_audit"]["tiers"]) == EXPECTED_TIERS
    records = _by_id(receipt)
    assert sorted(record["rank"] for record in records.values()) == [
        1, 1, 3, 3, 3, 3, 7, 7, 9, 9, 9, 9
    ]
    for tier_index, tier in enumerate(EXPECTED_TIERS, start=1):
        for candidate_id in tier:
            assert records[candidate_id]["tier"] == tier_index
            assert records[candidate_id]["tier_size"] == len(tier)


def test_the_gate_is_blind_to_the_sign_of_a10(receipt: dict) -> None:
    records = _by_id(receipt)
    for tier in EXPECTED_TIERS:
        magnitudes = {abs(sp.Rational(records[cid]["coefficients"]["a10"])) for cid in tier}
        signs = {sp.sign(sp.Rational(records[cid]["coefficients"]["a10"])) for cid in tier}
        assert len(magnitudes) == 1, "a tier must fix |a10|"
        assert signs == {1, -1}, "both signs of a10 sit in the same tier"


def test_every_strict_step_is_re_verifiable_from_the_receipt_with_integers_alone(
    receipt: dict,
) -> None:
    witnesses = receipt["gradient_audit"]["separation_witnesses"]
    assert len(witnesses) == 3
    for witness in witnesses:
        separation = witness["separation"]
        if "left_bracket" in separation:
            assert _rational(separation["left_bracket"]["hi"]) < _rational(
                separation["right_bracket"]["lo"]
            )
        else:
            assert int(separation["left"]) < int(separation["right"])


def test_receipt_brackets_actually_contain_the_values_they_claim_to(receipt: dict) -> None:
    mp.mp.dps = 60
    for record in receipt["candidate_records"]:
        for key, margin in record["margins"].items():
            if margin["kind"] != "magnitude":
                continue
            truth = _oracle(margin["exact"])
            low = _rational(margin["lo"])
            high = _rational(margin["hi"])
            assert mp.mpf(low.numerator) / mp.mpf(low.denominator) <= truth, key
            assert truth <= mp.mpf(high.numerator) / mp.mpf(high.denominator), key


def test_the_ranking_agrees_with_an_independent_high_precision_oracle(receipt: dict) -> None:
    mp.mp.dps = 60
    records = _by_id(receipt)
    ordered = sorted(records.values(), key=lambda item: (item["rank"], item["candidate_id"]))
    keyed = [
        (
            int(item["margins"]["uncancelled_slice_growth_multiplier"]["exact"]),
            _oracle(item["margins"]["certified_linear_energy_growth"]["exact"]),
        )
        for item in ordered
    ]
    assert keyed == sorted(keyed)


def test_the_lifespan_ceiling_is_largest_for_the_top_tier(receipt: dict) -> None:
    records = _by_id(receipt)
    ceilings = {}
    for tier_index, tier in enumerate(EXPECTED_TIERS, start=1):
        brackets = {
            (
                records[cid]["conditional_lifespan_ceiling"]["lo"],
                records[cid]["conditional_lifespan_ceiling"]["hi"],
            )
            for cid in tier
        }
        assert len(brackets) == 1
        ceilings[tier_index] = _rational(next(iter(brackets))[0])
        assert records[tier[0]]["conditional_lifespan_ceiling"]["expression"] == (
            "2*log(2)/A_known"
        )
    # A smaller certified growth constant is a larger already-proved lifespan ceiling, so tier 1
    # has the most room and tier 4 the least.
    assert ceilings[1] > ceilings[2] > ceilings[3] > ceilings[4]


def test_the_ordering_does_not_depend_on_which_magnitude_key_is_declared_first(
    receipt: dict,
) -> None:
    robustness = receipt["gradient_audit"]["key_order_robustness"]
    assert robustness["ranking_invariant_under_key_swap"] is True
    assert [tuple(tier) for tier in robustness["tiers_under_swapped_key_order"]] == list(
        EXPECTED_TIERS
    )


def test_the_upstream_drift_at_head_is_representational_not_numeric() -> None:
    """The annular campaign has been rewritten in a different algebraic form since the global-H7
    receipt was sealed, which is why the upstream chain no longer replays at HEAD.  The values
    the ranking rests on are unaffected: at 1024 bits the two forms cannot be separated, so the
    drift cannot have moved any candidate between tiers.  (A bracket can refute equality, never
    prove it, so this is a necessary condition and is stated as one.)
    """

    annular = _load("runs/physics-language/quartic-annular-k55-c6-campaign/campaign.json")
    global_h7 = _load(GLOBAL_H7_PATH)
    published = {
        item["candidate_id"]: item["principal_anti_wick_composition_constant"]["exact"]
        for item in annular["certificates"]
    }
    embedded = {
        item["candidate_id"]: item["summed_certified_terms"]["annular_C6_principal_composition"]
        for item in global_h7["certificates"]
    }
    assert set(published) == set(embedded)
    for candidate_id, expression in embedded.items():
        assert (
            compare_expressions(
                expression, published[candidate_id], ladder=(64, 256, 1024)
            ).verdict
            != LESS
        )
        assert not compare_expressions(
            published[candidate_id], expression, ladder=(64, 256, 1024)
        ).separated()


def test_the_ranking_is_not_a_relabelling_of_the_candidate_identity(receipt: dict) -> None:
    records = _by_id(receipt)
    by_ledger = [
        cid for cid in sorted(records, key=lambda item: (records[item]["rank"], item))
    ]
    assert by_ledger != sorted(records), "ranking by id must not reproduce the ledger order"


def test_the_ranking_is_deterministic_under_input_permutation() -> None:
    """Reordering the source lists changes their seals but must not change the ranking."""

    global_h7, finite_sobolev, block_gate = _sources()
    baseline = build_partial_credit(global_h7, finite_sobolev, block_gate)
    shuffled_global = copy.deepcopy(global_h7)
    shuffled_finite = copy.deepcopy(finite_sobolev)
    shuffled_gate = copy.deepcopy(block_gate)
    generator = random.Random(20260818)
    generator.shuffle(shuffled_global["certificates"])
    generator.shuffle(shuffled_finite["candidate_records"])
    generator.shuffle(shuffled_gate["candidate_records"])
    for document in (shuffled_global, shuffled_finite, shuffled_gate):
        document["content_sha256"] = _reseal(document)
    permuted = build_partial_credit(shuffled_global, shuffled_finite, shuffled_gate)
    for key in ("gradient_audit", "candidate_records", "counts", "negative_controls"):
        assert permuted[key] == baseline[key], key
    assert permuted["upstream_sha256"] != baseline["upstream_sha256"]


def test_the_committed_artifact_matches_a_fresh_rebuild(receipt: dict) -> None:
    stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert stored["content_sha256"] == receipt["content_sha256"]
    assert stored == receipt
    validate_receipt(stored)


# --------------------------------------------------------------------------------------
# Controls: each one must fail
# --------------------------------------------------------------------------------------


def test_the_declared_controls_all_fired(receipt: dict) -> None:
    controls = receipt["negative_controls"]
    assert all(control["rejected"] is True for control in controls.values())
    assert controls["identical_magnitudes_yield_no_gradient"]["tiers_when_flattened"] == "1"
    starved = controls["starved_precision_yields_no_full_order"]
    assert starved["applicable"] is True
    assert starved["tiers_at_declared_precision"] == "4"
    assert starved["tiers_at_starved_precision"] == "2"


def test_flattening_the_magnitudes_destroys_the_ordering() -> None:
    """The gradient detector must be falsifiable: identical inputs, no tiers."""

    global_h7, finite_sobolev, block_gate = _sources()
    reference = global_h7["certificates"][0]
    for certificate in global_h7["certificates"]:
        certificate["strongest_global_differential_inequality"] = copy.deepcopy(
            reference["strongest_global_differential_inequality"]
        )
        certificate["global_energy"] = copy.deepcopy(reference["global_energy"])
    for record in finite_sobolev["candidate_records"]:
        record["absolute_growth_multiplier"] = finite_sobolev["candidate_records"][0][
            "absolute_growth_multiplier"
        ]
    for document in (global_h7, finite_sobolev):
        document["content_sha256"] = _reseal(document)
    flattened = build_partial_credit(global_h7, finite_sobolev, block_gate)
    assert flattened["gradient_audit"]["distinct_tiers"] == 1
    assert flattened["gradient_audit"]["gradient_present"] is False
    assert {record["rank"] for record in flattened["candidate_records"]} == {1}


def test_a_starved_precision_budget_refuses_to_finish_the_order() -> None:
    """Sixteen bits cannot see the tight pairs, so the ranker must leave them tied."""

    global_h7, finite_sobolev, block_gate = _sources()
    starved = build_partial_credit(global_h7, finite_sobolev, block_gate, ladder=(16,))
    assert starved["gradient_audit"]["distinct_tiers"] == 2
    assert starved["gradient_audit"]["unseparated_pairs"]
    full = build_from_root(ROOT)
    assert starved["gradient_audit"]["distinct_tiers"] < full["gradient_audit"]["distinct_tiers"]


@pytest.mark.parametrize(
    "mutation",
    [
        "lifespan_proved",
        "inequality_closed",
        "gate_record_passes",
        "gate_claim_opened",
        "candidate_dropped",
        "seal_broken",
        "identity_misaligned",
    ],
)
def test_partial_credit_is_refused_on_a_tampered_source(mutation: str) -> None:
    global_h7, finite_sobolev, block_gate = _sources()
    if mutation == "lifespan_proved":
        global_h7["certificates"][0]["nonlinear_lifespan_proved"] = True
        global_h7["content_sha256"] = _reseal(global_h7)
    elif mutation == "inequality_closed":
        global_h7["certificates"][3]["global_H7_differential_inequality_closed"] = True
        global_h7["content_sha256"] = _reseal(global_h7)
    elif mutation == "gate_record_passes":
        block_gate["candidate_records"][0]["decision"] = "PASS"
        block_gate["content_sha256"] = _reseal(block_gate)
    elif mutation == "gate_claim_opened":
        block_gate["claims"]["lifespan_proved"] = True
        block_gate["content_sha256"] = _reseal(block_gate)
    elif mutation == "candidate_dropped":
        global_h7["certificates"].pop()
        global_h7["content_sha256"] = _reseal(global_h7)
    elif mutation == "seal_broken":
        global_h7["certificates"][0]["candidate_id"] = "quartic-symbol-forged0000000000"
    elif mutation == "identity_misaligned":
        finite_sobolev["candidate_records"][0]["candidate_id"] = "quartic-symbol-0000000000000000"
        finite_sobolev["content_sha256"] = _reseal(finite_sobolev)
    with pytest.raises(PartialCreditError):
        build_partial_credit(global_h7, finite_sobolev, block_gate)


def test_a_receipt_cannot_be_edited_into_a_pass(receipt: dict) -> None:
    forged = copy.deepcopy(receipt)
    forged["decision"] = "PASS"
    with pytest.raises(PartialCreditError):
        validate_receipt(forged)

    forged = copy.deepcopy(receipt)
    forged["candidate_records"][0]["decision"] = "PASS"
    forged["content_sha256"] = _reseal(forged)
    with pytest.raises(PartialCreditError):
        validate_receipt(forged)

    forged = copy.deepcopy(receipt)
    forged["claims"]["lifespan_proved"] = True
    forged["content_sha256"] = _reseal(forged)
    with pytest.raises(PartialCreditError):
        validate_receipt(forged)

    forged = copy.deepcopy(receipt)
    forged["counts"]["passed"] = 1
    forged["content_sha256"] = _reseal(forged)
    with pytest.raises(PartialCreditError):
        validate_receipt(forged)


def test_the_declared_ledger_and_key_order_are_sealed(receipt: dict) -> None:
    assert receipt["obligation_ledger_sha256"] == ledger_sha256()
    assert tuple(receipt["ranking_keys"]) == RANKING_KEYS
    forged = copy.deepcopy(receipt)
    forged["ranking_keys"] = list(reversed(RANKING_KEYS))
    forged["content_sha256"] = _reseal(forged)
    with pytest.raises(PartialCreditError):
        validate_receipt(forged)
    forged = copy.deepcopy(receipt)
    forged["obligation_ledger_sha256"] = "0" * 64
    forged["content_sha256"] = _reseal(forged)
    with pytest.raises(PartialCreditError):
        validate_receipt(forged)


def test_a_forged_separation_witness_does_not_survive_re_verification(receipt: dict) -> None:
    """The receipt's own witnesses are checkable; a swapped one stops separating."""

    witness = next(
        item
        for item in receipt["gradient_audit"]["separation_witnesses"]
        if "left_bracket" in item["separation"]
    )["separation"]
    assert _rational(witness["left_bracket"]["hi"]) < _rational(witness["right_bracket"]["lo"])
    swapped_left = witness["right_bracket"]
    swapped_right = witness["left_bracket"]
    assert not _rational(swapped_left["hi"]) < _rational(swapped_right["lo"])


def test_the_exact_comparator_reproduces_the_receipt_order(receipt: dict) -> None:
    records = _by_id(receipt)
    for nearer_tier, farther_tier in pairwise(EXPECTED_TIERS):
        nearer = records[nearer_tier[0]]["margins"]
        farther = records[farther_tier[0]]["margins"]
        nearer_multiplier = int(nearer["uncancelled_slice_growth_multiplier"]["exact"])
        farther_multiplier = int(farther["uncancelled_slice_growth_multiplier"]["exact"])
        if nearer_multiplier != farther_multiplier:
            assert nearer_multiplier < farther_multiplier
            continue
        comparison = compare_expressions(
            nearer["certified_linear_energy_growth"]["exact"],
            farther["certified_linear_energy_growth"]["exact"],
        )
        assert comparison.verdict == LESS


def _reseal(document: dict) -> str:
    import hashlib

    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
