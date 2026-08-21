from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from sigma_theory_compiler.pancyclic_chord_search import (
    CLAIMS,
    HEURISTIC_SEARCH_LOG,
    RECEIPT_SCHEMA,
    SEALED_TABLE,
    SEALED_TABLE_LAST_N,
    WITNESSES,
    PancyclicError,
    candidate_chords,
    counting_lower_bound_chords,
    cycle_spectrum,
    exhaustive_search,
    minimum_size_by_search,
    normalise_chords,
    record_gate,
    verify_witness,
    witness_receipt,
)


# --------------------------------------------------------------------------------------
# An independent cycle enumerator.  cycle_spectrum() walks the GF(2) cycle space; this one
# grows simple paths from a fixed smallest vertex.  The two share no code and no idea.
# --------------------------------------------------------------------------------------
def brute_force_cycle_lengths(n: int, chords: Sequence[tuple[int, int]]) -> list[int]:
    adjacency: dict[int, set[int]] = {v: set() for v in range(n)}
    for v in range(n):
        adjacency[v].add((v + 1) % n)
        adjacency[(v + 1) % n].add(v)
    for u, v in chords:
        adjacency[u].add(v)
        adjacency[v].add(u)
    lengths: list[int] = []
    for root in range(n):
        stack = [(root, [root], {root})]
        while stack:
            current, path, seen = stack.pop()
            for nxt in sorted(adjacency[current]):
                if nxt == root and len(path) >= 3:
                    # count each cycle once: fix the direction by the second vertex
                    if path[1] < path[-1]:
                        lengths.append(len(path))
                    continue
                if nxt in seen or nxt < root:
                    continue
                stack.append((nxt, path + [nxt], seen | {nxt}))
    return sorted(lengths)


def test_counting_lower_bound_is_the_bondy_shi_count() -> None:
    assert counting_lower_bound_chords(3) == 0
    assert counting_lower_bound_chords(5) == 1
    assert counting_lower_bound_chords(6) == 2
    assert counting_lower_bound_chords(9) == 2
    assert counting_lower_bound_chords(10) == 3
    assert counting_lower_bound_chords(18) == 4
    assert counting_lower_bound_chords(33) == 4
    assert counting_lower_bound_chords(34) == 5
    assert counting_lower_bound_chords(38) == 5
    assert counting_lower_bound_chords(65) == 5
    assert counting_lower_bound_chords(66) == 6
    for n in range(3, 200):
        k = counting_lower_bound_chords(n)
        assert (1 << (k + 1)) - 1 >= n - 2
        assert k == 0 or (1 << k) - 1 < n - 2
    with pytest.raises(PancyclicError):
        counting_lower_bound_chords(2)


def test_published_table_never_falls_below_the_counting_bound() -> None:
    for n, (k, size) in SEALED_TABLE.items():
        assert size == n + k
        assert k >= counting_lower_bound_chords(n)
    assert max(SEALED_TABLE) == SEALED_TABLE_LAST_N


def test_cycle_spectrum_on_hand_checkable_graphs() -> None:
    plain = cycle_spectrum(7, ())
    assert plain["spectrum"] == [7]
    assert plain["elements_examined"] == plain["elements_expected"] == 1
    assert plain["cycles_found"] == 1

    one_chord = cycle_spectrum(5, ((0, 2),))
    assert one_chord["spectrum"] == [3, 4, 5]
    assert one_chord["elements_examined"] == 3
    assert one_chord["cycles_found"] == 3

    # C_6 with the long diagonal: two 4-cycles of the same length plus the hexagon
    diagonal = cycle_spectrum(6, ((0, 3),))
    assert diagonal["spectrum"] == [4, 6]
    assert diagonal["cycles_found"] == 3


def test_cycle_space_enumeration_agrees_with_path_search() -> None:
    cases = [
        (5, ((0, 2),)),
        (6, ((0, 3),)),
        (8, ((0, 2), (1, 4))),
        (9, ((0, 2), (1, 3), (1, 5))),
        (14, ((0, 2), (1, 4), (1, 10))),
        (12, ((0, 6), (2, 8), (3, 9))),
    ]
    for n, chords in cases:
        by_space = cycle_spectrum(n, chords)
        by_path = brute_force_cycle_lengths(n, chords)
        assert by_space["cycles_found"] == len(by_path)
        assert by_space["spectrum"] == sorted(set(by_path))


def test_witness_cycle_counts_match_an_independent_path_search() -> None:
    for n, chords in WITNESSES.items():
        by_space = cycle_spectrum(n, chords)
        by_path = brute_force_cycle_lengths(n, chords)
        assert by_space["cycles_found"] == len(by_path)
        assert by_space["spectrum"] == sorted(set(by_path))


def test_shi_bound_holds_on_every_case_examined() -> None:
    for n, chords in WITNESSES.items():
        report = cycle_spectrum(n, chords)
        assert report["cycles_found"] <= (1 << (len(chords) + 1)) - 1
        assert report["cycles_found"] >= n - 2


def test_normalise_chords_refuses_malformed_input() -> None:
    assert normalise_chords(8, [(4, 1), (0, 2)]) == ((0, 2), (1, 4))
    with pytest.raises(PancyclicError):
        normalise_chords(8, [(0, 1)])       # already an edge of the cycle
    with pytest.raises(PancyclicError):
        normalise_chords(8, [(7, 0)])       # the wrap-around edge
    with pytest.raises(PancyclicError):
        normalise_chords(8, [(3, 3)])       # loop
    with pytest.raises(PancyclicError):
        normalise_chords(8, [(0, 2), (2, 0)])   # repeated chord
    with pytest.raises(PancyclicError):
        normalise_chords(8, [(0, 9)])       # out of range
    with pytest.raises(PancyclicError):
        normalise_chords(8, [(0, True)])    # bool is not an endpoint
    with pytest.raises(PancyclicError):
        normalise_chords(8, [(0, 2, 4)])    # not a pair


def test_candidate_chords_excludes_exactly_the_cycle_edges() -> None:
    for n in (5, 8, 13):
        chords = candidate_chords(n)
        assert len(chords) == n * (n - 1) // 2 - n
        assert len(set(chords)) == len(chords)
        for u, v in chords:
            assert (v - u) not in (1, n - 1)


def test_exhaustive_search_reproduces_the_published_table_up_to_fourteen() -> None:
    for n in range(3, 15):
        published_k, published_size = SEALED_TABLE[n]
        outcome = minimum_size_by_search(n, max_chords=published_k)
        assert outcome.witness is not None, n
        assert outcome.chord_count == published_k, n
        assert n + outcome.chord_count == published_size, n
        certificate = verify_witness(n, outcome.witness)
        assert certificate["pancyclic"], n
        assert record_gate(certificate)["verdict"] == "reproduces_the_published_value"


def test_the_counting_bound_is_not_tight_at_fifteen() -> None:
    # First order where the elementary count allows 3 chords but no 3-chord graph exists.
    assert counting_lower_bound_chords(15) == 3
    assert SEALED_TABLE[15] == (4, 19)
    exhausted = exhaustive_search(15, 3)
    assert exhausted.witness is None
    assert exhausted.configurations == 18018
    found = exhaustive_search(15, 4)
    assert found.witness is not None
    assert verify_witness(15, found.witness)["pancyclic"]


def test_witnesses_determine_values_past_the_published_table() -> None:
    assert set(WITNESSES) == {38, 39, 40}
    for n, chords in sorted(WITNESSES.items()):
        certificate = verify_witness(n, chords)
        assert certificate["order"] == n
        assert certificate["chord_count"] == 5
        assert certificate["edges"] == n + 5
        assert certificate["cycle_space_dimension"] == 6
        assert certificate["elements_examined"] == 63
        assert certificate["elements_expected"] == 63
        assert certificate["spectrum"] == list(range(3, n + 1))
        assert certificate["missing_lengths"] == []
        assert certificate["pancyclic"] is True
        assert certificate["counting_lower_bound_chords"] == 5
        assert certificate["counting_lower_bound_edges"] == n + 5
        assert certificate["matches_counting_lower_bound"] is True
        gate = record_gate(certificate)
        assert gate["sealed_edges"] is None
        assert gate["verdict"] == "determines_a_value_past_the_published_table"
        assert n > SEALED_TABLE_LAST_N


def test_every_chord_of_every_witness_is_load_bearing() -> None:
    for n, chords in WITNESSES.items():
        for dropped in range(len(chords)):
            reduced = tuple(c for i, c in enumerate(chords) if i != dropped)
            assert verify_witness(n, reduced)["pancyclic"] is False


def test_a_perturbed_witness_stops_being_pancyclic() -> None:
    n = 38
    chords = list(WITNESSES[n])
    chords[0] = (11, 14)
    certificate = verify_witness(n, chords)
    assert certificate["pancyclic"] is False
    assert certificate["missing_lengths"]
    assert record_gate(certificate)["verdict"] == "not_pancyclic"


def test_record_gate_reports_a_reproduction_inside_the_table() -> None:
    certificate = verify_witness(14, ((0, 2), (1, 4), (1, 10)))
    assert certificate["pancyclic"] is True
    gate = record_gate(certificate)
    assert gate["sealed_edges"] == 17
    assert gate["verdict"] == "reproduces_the_published_value"


def test_record_gate_reports_a_worse_object_as_worse() -> None:
    certificate = verify_witness(14, ((0, 2), (1, 4), (1, 10), (3, 8)))
    assert certificate["pancyclic"] is True
    assert record_gate(certificate)["verdict"] == "worse_than_the_published_value"


def test_search_log_never_stands_in_for_a_proof() -> None:
    # A recorded hit rate is a property of a search budget, not of the mathematics; the only
    # thing it may imply is the existence direction, and only where a witness backs it.
    for order, (restarts, hits) in HEURISTIC_SEARCH_LOG.items():
        assert 0 <= hits <= restarts
        if hits > 0:
            assert order <= max(WITNESSES) or order in SEALED_TABLE
    assert HEURISTIC_SEARCH_LOG[41] == (27, 0)
    assert 41 not in WITNESSES
    assert 41 not in SEALED_TABLE
    assert counting_lower_bound_chords(41) == 5


def test_receipt_is_deterministic_and_binds_the_witnesses() -> None:
    first = witness_receipt()
    second = witness_receipt()
    assert first == second
    assert first["schema"] == RECEIPT_SCHEMA
    assert first["claims"] == dict(sorted(CLAIMS.items()))
    assert len(first["entries"]) == len(WITNESSES)
    assert json.loads(json.dumps(first)) == first
    subset = witness_receipt([38])
    assert len(subset["entries"]) == 1
    assert subset["receipt_sha256"] != first["receipt_sha256"]


def test_receipt_carries_no_floats() -> None:
    floats: list[object] = []

    def walk(value: object) -> None:
        if isinstance(value, float):
            floats.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(witness_receipt())
    assert floats == []
