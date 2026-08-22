"""SELFGEN -- the widened self-generated conjecture loop must stay honest and exact.

The tests split into four groups.  The first pins the *widening* itself: every admitted
object carries a proof that it is not a polynomial of degree at most five, and the C3 pool --
whose members are exactly such polynomials -- is refused wholesale.  The second pins the
decision procedure: the orbit table is complete over ``n >= 0``, its closure identity
replays, and known classical facts about Fibonacci come back with the classical answers.
The third pins the honesty machinery: a receipt is sealed, its claims cannot be flipped, a
refutation must carry a witness, and a proved statement must carry a prior-art triage.  The
fourth pins the census as an exhaustion rather than a sample: the three outcome counts must
sum to the declared box size.
"""

from __future__ import annotations

import json

import pytest

from sigma_theory_compiler import self_generated_conjecture_widening as widening
from sigma_theory_compiler.self_generated_conjecture_adjudication import (
    PoolConfig,
)
from sigma_theory_compiler.self_generated_conjecture_adjudication import (
    build_pool as build_c3_pool,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

SMALL = widening.WideningConfig(orders=(2,), coefficient_box=2, initial_box=1)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return widening.run_loop(SMALL, max_per_kind=24)


# ---------------------------------------------------------------------------
# 1 -- the widening is real: these objects are outside the elementary universe.
# ---------------------------------------------------------------------------


def test_every_admitted_object_carries_a_non_polynomial_certificate() -> None:
    pool = widening.build_pool(SMALL)
    assert pool["objects"]
    for record in pool["objects"]:
        certificate = record["non_polynomial_certificate"]
        degree = int(certificate["refuted_polynomial_degree_at_most"])
        assert degree >= widening.ELEMENTARY_DEGREE_CEILING
        # The certificate is a single number and it must be the number it claims to be.
        row = widening.forward_difference_order(
            record["values"], int(certificate["difference_order"])
        )
        assert row[int(certificate["witness_index"])] == int(certificate["witness_value"]) != 0


def test_the_c3_pool_is_refused_by_the_widening_gate_object_by_object() -> None:
    c3 = build_c3_pool(PoolConfig(window=SMALL.window))
    assert len(c3["objects"]) > 100
    for member in c3["objects"]:
        certificate = widening.non_polynomial_certificate(member["values"])
        degree = (
            -1 if certificate is None else int(certificate["refuted_polynomial_degree_at_most"])
        )
        assert degree < widening.ELEMENTARY_DEGREE_CEILING, member["object_id"]


def test_the_elementary_closed_form_gate_names_the_index_where_it_breaks() -> None:
    pool = widening.build_pool(SMALL)
    record = pool["objects"][0]
    refusal = widening.elementary_procedure_fails(record)
    assert refusal
    index = int(refusal["first_disagreement_index"])
    assert int(refusal["object_value"]) == record["values"][index]
    assert int(refusal["polynomial_value"]) != record["values"][index]


# ---------------------------------------------------------------------------
# 2 -- the decision procedure is complete, and it reproduces classical facts.
# ---------------------------------------------------------------------------


def test_orbit_table_predicts_every_index_it_was_never_shown() -> None:
    coefficients, initial, modulus = [1, 1], [0, 1], 7
    table = widening.residue_table(coefficients, initial, modulus, cap=10_000)
    assert table is not None
    widening.validate_residue_table(coefficients, initial, table)
    direct = widening.recurrence_values(coefficients, initial, 400)
    for index, value in enumerate(direct):
        assert table.at(index) == value % modulus
    assert table.mu + table.lam <= modulus ** len(coefficients)


def test_pisano_period_of_the_fibonacci_object_matches_wall_1960() -> None:
    # D. D. Wall, Amer. Math. Monthly 67 (1960) 525-532: pi(2)=3, pi(3)=8, pi(5)=20,
    # pi(7)=16, pi(11)=10.  The orbit procedure has to agree with the published table.
    for modulus, published in ((2, 3), (3, 8), (5, 20), (7, 16), (11, 10)):
        table = widening.residue_table([1, 1], [0, 1], modulus, cap=100_000)
        assert table is not None
        assert table.mu == 0
        assert table.lam == published, (modulus, table.lam, published)


def test_classical_fibonacci_divisibility_comes_back_proved_and_a_wrong_one_refuted() -> None:
    record = {
        "object_id": "fibonacci",
        "order": 2,
        "coefficients": [1, 1],
        "initial": [0, 1],
        "values": widening.recurrence_values([1, 1], [0, 1], SMALL.window),
    }
    # 2 | F(n) iff 3 | n, 3 | F(n) iff 4 | n, 5 | F(n) iff 5 | n -- all classical.
    for modulus, period in ((2, 3), (3, 4), (5, 5)):
        _, verdict = widening.adjudicate_divisibility(
            record, {"m": modulus, "q": period, "j": 0}, SMALL
        )
        assert verdict["verdict"] == "PROVED", (modulus, period, verdict)
    _, wrong = widening.adjudicate_divisibility(record, {"m": 3, "q": 3, "j": 0}, SMALL)
    assert wrong["verdict"] == "REFUTED"
    witness = wrong["witness"]
    assert record["values"][int(witness["n"])] % 3 != 0 or int(witness["n"]) % 3 != 0


def test_zero_free_routes_are_sound_and_neither_is_allowed_to_overclaim() -> None:
    def planted(coefficients: list[int], initial: list[int]) -> dict:
        return {
            "object_id": "planted",
            "order": len(coefficients),
            "coefficients": coefficients,
            "initial": initial,
            "values": widening.recurrence_values(coefficients, initial, SMALL.window),
        }

    _, refuted = widening.adjudicate_zero_free(planted([0, 1], [1, 0]), SMALL)
    assert refuted["verdict"] == "REFUTED" and refuted["witness"]["n"] == 1

    _, grown = widening.adjudicate_zero_free(planted([1, 1], [1, 1]), SMALL)
    assert grown["verdict"] == "PROVED" and grown["proof_route"] == "positivity_induction"

    _, obstructed = widening.adjudicate_zero_free(planted([-2, -2], [-1, -1]), SMALL)
    assert obstructed["verdict"] == "PROVED" and obstructed["proof_route"] == "local_obstruction"
    table = obstructed["residue_table"]
    assert 0 not in table["values"]

    # No zero anywhere the search reached, and still OPEN: neither route arrives.
    _, unreached = widening.adjudicate_zero_free(planted([-1, -1, -1, 1], [-1, 1, -1, 1]), SMALL)
    assert unreached["verdict"] == "OPEN"


def test_a_zero_free_certificate_is_rechecked_from_its_inputs_alone() -> None:
    record = {
        "object_id": "planted",
        "order": 2,
        "coefficients": [-2, -2],
        "initial": [-1, -1],
        "values": widening.recurrence_values([-2, -2], [-1, -1], SMALL.window),
    }
    _, verdict = widening.adjudicate_zero_free(record, SMALL)
    assert verdict["proof_route"] == "local_obstruction"
    widening.recheck_zero_free_certificate([-2, -2], [-1, -1], verdict)
    tampered = json.loads(json.dumps(verdict))
    tampered["residue_table"]["lam"] += 1
    with pytest.raises(widening.ConjectureWideningError):
        widening.recheck_zero_free_certificate([-2, -2], [-1, -1], tampered)


def test_a_positivity_certificate_is_rejected_when_its_twist_is_wrong() -> None:
    certificate = widening.positivity_certificate([1, 1], [1, 1], max_start=6, window=24)
    assert certificate is not None
    adjudication = {"proof_route": "positivity_induction", "positivity_certificate": certificate}
    widening.recheck_zero_free_certificate([1, 1], [1, 1], adjudication)
    broken = json.loads(json.dumps(adjudication))
    broken["positivity_certificate"]["class_proofs"][0]["twisted_coefficients"] = [-1, 1]
    with pytest.raises(widening.ConjectureWideningError):
        widening.recheck_zero_free_certificate([1, 1], [1, 1], broken)


def test_decimation_reproduces_the_subsequence_recurrence_exactly() -> None:
    # Cayley-Hamilton: u(pt + r) obeys the recurrence read off the characteristic polynomial
    # of M**p.  For Fibonacci and p = 2 that is the classical x(t+2) = 3x(t+1) - x(t).
    matrix = widening.companion_matrix([1, 1])
    assert widening.characteristic_recurrence(widening.matrix_power(matrix, 2)) == [3, -1]
    for coefficients, initial in (
        ([1, 1], [0, 1]),
        ([-1, -2, 1], [1, -1, 2]),
        ([2, 0, 0, -1], [1, 1, 0, 2]),
    ):
        order = len(coefficients)
        values = widening.recurrence_values(coefficients, initial, 200)
        base = widening.companion_matrix(coefficients)
        for period in (1, 2, 3, 4, 5):
            rule = widening.characteristic_recurrence(widening.matrix_power(base, period))
            for residue in range(period):
                decimated = values[residue::period]
                for index in range(len(decimated) - order):
                    predicted = sum(
                        rule[offset] * decimated[index + order - 1 - offset]
                        for offset in range(order)
                    )
                    assert predicted == decimated[index + order]


def test_a_positivity_certificate_must_cover_every_residue_class() -> None:
    certificate = widening.positivity_certificate(
        [1, 1], [1, 1], max_start=6, window=24, decimations=(2,)
    )
    assert certificate is not None and certificate["decimation"] == 2
    adjudication = {"proof_route": "positivity_induction", "positivity_certificate": certificate}
    widening.recheck_zero_free_certificate([1, 1], [1, 1], adjudication)
    broken = json.loads(json.dumps(adjudication))
    broken["positivity_certificate"]["class_proofs"].pop()
    with pytest.raises(widening.ConjectureWideningError):
        widening.recheck_zero_free_certificate([1, 1], [1, 1], broken)


# ---------------------------------------------------------------------------
# 3 -- the honesty machinery.
# ---------------------------------------------------------------------------


def test_the_receipt_validates_and_its_controls_all_passed(receipt: dict) -> None:
    widening.validate_receipt(receipt)
    assert receipt["controls_passed"]
    names = {entry["control"] for entry in receipt["controls"]}
    assert "elementary_objects_must_be_refused_by_the_widening_gate" in names
    assert "restating_generator_must_be_refused" in names
    assert "planted_zero_free_verdicts_must_split_four_ways" in names
    assert all(entry["passed"] for entry in receipt["controls"])


def test_sealed_claims_cannot_be_flipped(receipt: dict) -> None:
    assert receipt["claims"]["proved_means_novel"] is False
    assert receipt["claims"]["prior_art_absence_establishes_novelty"] is False
    tampered = json.loads(json.dumps(receipt))
    tampered["claims"]["proved_means_novel"] = True
    with pytest.raises(widening.ConjectureWideningError):
        widening.validate_receipt(tampered)


def test_the_seal_covers_the_body(receipt: dict) -> None:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)
    tampered = json.loads(json.dumps(receipt))
    tampered["counts"]["adjudicated"] += 1
    with pytest.raises(widening.ConjectureWideningError):
        widening.validate_receipt(tampered)


def test_no_float_reaches_the_certificate_path(receipt: dict) -> None:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert not widening._contains_float(body)
    tampered = json.loads(json.dumps(receipt))
    tampered["counts"]["candidates_emitted"] = 1.0
    with pytest.raises(widening.ConjectureWideningError):
        widening.validate_receipt(tampered)


def test_every_refutation_carries_a_witness_and_every_proof_a_triage(receipt: dict) -> None:
    proved = 0
    refuted = 0
    for record in receipt["conjectures"]:
        verdict = record["adjudication"]["verdict"]
        if verdict == "REFUTED":
            refuted += 1
            assert record["adjudication"]["witness"]
        elif verdict == "PROVED":
            proved += 1
            bucket = record["prior_art"]["bucket"]
            assert bucket in {"proved_and_known", "proved_and_prior_art_not_found"}
            if bucket == "proved_and_prior_art_not_found":
                assert record["prior_art"]["search_record"]["standing_disclaimer"]
                assert record["prior_art"]["novelty_claim"].startswith("none")
    assert proved and refuted


def test_the_gates_refuse_both_the_restating_generator_and_some_honest_candidates(
    receipt: dict,
) -> None:
    assert receipt["counts"]["refused_by_admission_gates"] > 0
    control = next(
        entry
        for entry in receipt["controls"]
        if entry["control"] == "restating_generator_must_be_refused"
    )
    assert control["candidates"] > 0 and control["admitted"] == 0


def test_prior_art_entries_all_carry_an_attribution() -> None:
    for entry in widening.PRIOR_ART:
        assert entry["attribution"].strip()
        assert entry["confidence"] in {
            "family_theorem",
            "pinned_identity",
            "section_reference",
            "elementary_derivation",
        }
    assert widening.PRIOR_ART_SEARCH_RECORD["not_searched"]


def test_object_specific_prior_art_does_not_cover_objects_it_cannot_reach() -> None:
    lucas = next(
        entry for entry in widening.PRIOR_ART if entry["key"] == "lucas_rank_of_apparition"
    )
    assert widening._entry_applies(lucas, {"coefficients": [1, 1], "initial": [0, 1]})
    assert not widening._entry_applies(lucas, {"coefficients": [1, 1, 1], "initial": [0, 1, 1]})


# ---------------------------------------------------------------------------
# 4 -- the census is an exhaustion, not a sample.
# ---------------------------------------------------------------------------


def test_the_census_partitions_its_declared_box_exactly() -> None:
    config = widening.CensusConfig(order=3, coefficient_box=1, initial_box=1, zero_scan=60)
    sweep = widening.census(config)
    counts = sweep["counts"]
    assert sweep["visited"] == config.box_size() == 729
    # The order-3 box is settled outright: every member is either killed by an explicit zero
    # or proved zero-free.  A regression that weakened either route would show up here.
    assert counts["unsettled"] == 0
    assert (
        counts["has_zero_with_explicit_witness"]
        + counts["zero_free_with_modulus_certificate"]
        + counts["unsettled"]
        == counts["total"]
        == config.box_size()
    )
    assert sweep["partition_is_exact"]


def test_a_census_attached_to_a_receipt_is_revalidated(receipt: dict) -> None:
    result = widening.run_loop(
        SMALL,
        max_per_kind=24,
        census_config=widening.CensusConfig(
            order=2, coefficient_box=1, initial_box=1, zero_scan=40
        ),
    )
    widening.validate_receipt(result)
    assert result["skolem_census"]["partition_is_exact"]
    tampered = json.loads(json.dumps(result))
    tampered["skolem_census"]["counts"]["unsettled"] += 1
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(widening.ConjectureWideningError):
        widening.validate_receipt(tampered)
