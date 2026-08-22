"""Tests for the cyclic Ramsey lower bound search.

The load-bearing test in this file is :func:`test_pruned_search_agrees_with_brute_force`.
Every other claim the module makes rests on the pruned enumerator being *complete*, and the
only way to know that is to run the unpruned one -- a literal loop over all
``2 ** floor(n / 2)`` symmetric connection sets -- and check the two agree set for set.
"""

from __future__ import annotations

import json
import sys
from math import gcd

import pytest

from sigma_theory_compiler.cyclic_ramsey_lower_bounds import (
    SEALED_CYCLIC_ORDERS,
    SEALED_RECORD,
    adjacency_independence_number,
    adjacency_is_triangle_free,
    build_adjacency,
    compare_to_record,
    connection_from_residues,
    difference_classes,
    dihedral_adjacency,
    dihedral_search,
    enumerate_naive,
    exhaustive_search,
    has_independent_set,
    independence_number,
    independent_set_witness,
    is_symmetric,
    is_triangle_free,
    main,
    max_cyclic_order,
    scan_orders,
    seal_certificate,
    triangle_witness,
    verify_certificate,
    verify_colouring,
)

sys.setrecursionlimit(100000)


# --------------------------------------------------------------------------------------
# Representation.
# --------------------------------------------------------------------------------------


def test_difference_classes_partition_the_nonzero_residues() -> None:
    for n in range(2, 30):
        classes = difference_classes(n)
        assert len(classes) == n // 2
        flat = [d for cls in classes for d in cls]
        assert sorted(flat) == list(range(1, n))
        for cls in classes:
            for d in cls:
                assert (n - d) % n in cls


def test_difference_classes_rejects_nonpositive_order() -> None:
    with pytest.raises(ValueError):
        difference_classes(0)


def test_connection_sets_are_symmetric() -> None:
    for n in (7, 12, 13):
        for residues in ([1], [2, 3], [1, 2, 3]):
            mask = connection_from_residues(n, residues)
            assert is_symmetric(n, mask)


def test_connection_from_residues_rejects_zero() -> None:
    with pytest.raises(ValueError):
        connection_from_residues(9, [0])
    with pytest.raises(ValueError):
        connection_from_residues(9, [9])


# --------------------------------------------------------------------------------------
# Exhaustive clique / independence primitives.
# --------------------------------------------------------------------------------------


def _brute_force_independence(n: int, connection: int) -> int:
    """Independence number by looking at every one of the 2**n vertex subsets."""

    adj = build_adjacency(n, connection)
    best = 0
    for bits in range(1 << n):
        ok = True
        rest = bits
        while rest:
            low = rest & -rest
            v = low.bit_length() - 1
            if adj[v] & bits:
                ok = False
                break
            rest &= ~low
        if ok:
            best = max(best, bits.bit_count())
    return best


def _brute_force_triangle_free(n: int, connection: int) -> bool:
    adj = build_adjacency(n, connection)
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if (adj[a] >> b) & 1 and (adj[a] >> c) & 1 and (adj[b] >> c) & 1:
                    return False
    return True


@pytest.mark.parametrize("n", [5, 7, 8, 9, 10, 11, 12, 13])
def test_independence_number_matches_full_subset_enumeration(n: int) -> None:
    classes = difference_classes(n)
    for bits in range(1 << len(classes)):
        connection = 0
        for i, cls in enumerate(classes):
            if (bits >> i) & 1:
                for d in cls:
                    connection |= 1 << d
        assert independence_number(n, connection) == _brute_force_independence(n, connection)


@pytest.mark.parametrize("n", [5, 7, 9, 11, 13, 14])
def test_triangle_test_matches_full_triple_enumeration(n: int) -> None:
    classes = difference_classes(n)
    for bits in range(1 << len(classes)):
        connection = 0
        for i, cls in enumerate(classes):
            if (bits >> i) & 1:
                for d in cls:
                    connection |= 1 << d
        expected = _brute_force_triangle_free(n, connection)
        assert is_triangle_free(n, connection) is expected
        assert (triangle_witness(n, connection) is None) is expected


def test_triangle_witness_is_a_real_triangle() -> None:
    n, connection = 9, connection_from_residues(9, [1, 2])
    tri = triangle_witness(n, connection)
    assert tri is not None
    adj = build_adjacency(n, connection)
    a, b, c = tri
    assert len({a, b, c}) == 3
    assert (adj[a] >> b) & 1 and (adj[a] >> c) & 1 and (adj[b] >> c) & 1


def test_independent_set_witness_is_independent_and_maximum() -> None:
    n = 13
    connection = connection_from_residues(n, [4, 6])
    alpha = independence_number(n, connection)
    witness = independent_set_witness(n, build_adjacency(n, connection), alpha)
    assert witness is not None and len(witness) == alpha
    adj = build_adjacency(n, connection)
    for u in witness:
        for v in witness:
            if u != v:
                assert not (adj[u] >> v) & 1
    assert independent_set_witness(n, adj, alpha + 1) is None


def test_has_independent_set_is_monotone_downward() -> None:
    n = 17
    connection = connection_from_residues(n, [1, 2, 4, 8])
    adj = build_adjacency(n, connection)
    alpha = independence_number(n, connection)
    for size in range(alpha + 1):
        assert has_independent_set(n, adj, size)
    assert not has_independent_set(n, adj, alpha + 1)


def test_five_cycle_is_the_classical_R33_witness() -> None:
    checks = verify_colouring(5, [1, 4], 3)
    assert checks["valid"] is True
    assert checks["triangle_free"] is True
    assert checks["independence_number"] == 2
    assert checks["certifies"] == "R(3,3) >= 6"


# --------------------------------------------------------------------------------------
# Completeness of the pruned search -- the test everything else depends on.
# --------------------------------------------------------------------------------------


def _orbit(n: int, connection: int) -> set[int]:
    orbit = set()
    for u in range(1, n):
        if gcd(u, n) != 1:
            continue
        image = 0
        rest = connection
        while rest:
            low = rest & -rest
            x = low.bit_length() - 1
            image |= 1 << ((u * x) % n)
            rest &= ~low
        orbit.add(image)
    return orbit


@pytest.mark.parametrize("n", list(range(3, 21)))
@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_pruned_search_agrees_with_brute_force(n: int, k: int) -> None:
    """The pruned walk must find a representative of every witness the naive loop finds,
    and must never report a set the naive loop rejects."""

    naive = set(enumerate_naive(n, k))
    outcome = exhaustive_search(n, k, collect_all=True)
    pruned = {connection_from_residues(n, w.connection) for w in outcome.witnesses}

    # nothing invented
    assert pruned <= naive
    # nothing lost: every naive witness has an orbit member among the canonical survivors
    for connection in naive:
        assert _orbit(n, connection) & pruned, (n, k, connection)
    # existence agrees
    assert bool(pruned) == bool(naive)


@pytest.mark.parametrize("n", [6, 9, 12, 15, 16, 18])
def test_witness_properties_hold_for_every_returned_colouring(n: int) -> None:
    for k in (4, 5, 6):
        outcome = exhaustive_search(n, k, collect_all=True)
        for w in outcome.witnesses:
            connection = connection_from_residues(n, w.connection)
            assert is_symmetric(n, connection)
            assert is_triangle_free(n, connection)
            assert independence_number(n, connection) == w.independence
            assert w.independence <= k - 1
            assert w.degree == connection.bit_count()
            # fact 1 of the module docstring
            assert w.degree <= k - 1


def test_degree_window_can_be_vacuous() -> None:
    outcome = exhaustive_search(40, 3)
    assert outcome.vacuous is True
    assert outcome.witnesses == []
    assert "degree window empty" in outcome.reason


def test_search_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        exhaustive_search(0, 3)
    with pytest.raises(ValueError):
        exhaustive_search(10, 1)


def test_stop_at_first_and_collect_all_agree_on_existence() -> None:
    for n in range(3, 26):
        first = exhaustive_search(n, 5, stop_at_first=True)
        every = exhaustive_search(n, 5, collect_all=True)
        assert bool(first.witnesses) == bool(every.witnesses)
        assert len(first.witnesses) <= 1


# --------------------------------------------------------------------------------------
# Reproduction of the published cyclic sequence.
# --------------------------------------------------------------------------------------

# Upper end of each scan is chosen well past the last hit; existence is not monotone in n,
# so the whole interval is walked rather than stopping at the first miss.
_SCAN_LIMIT = {3: 10, 4: 14, 5: 20, 6: 24, 7: 30, 8: 34, 9: 42, 10: 44}


@pytest.mark.parametrize("k", sorted(_SCAN_LIMIT))
def test_reproduces_published_cyclic_orders(k: int) -> None:
    """OEIS A000789: maximal order of a triangle-free cyclic graph with no independent set
    of size k.  a(10) onward is credited to Harborth and Krause."""

    best, witness = max_cyclic_order(k, 1, _SCAN_LIMIT[k])
    assert best == SEALED_CYCLIC_ORDERS[k]
    assert witness is not None
    checks = verify_colouring(best, list(witness.connection), k)
    assert checks["valid"] is True
    assert checks["independence_number"] == k - 1


def test_cyclic_orders_stay_below_the_published_ramsey_bound() -> None:
    """a(k) <= R(3,k) - 1, since a cyclic witness is in particular a witness."""

    for k, order in SEALED_CYCLIC_ORDERS.items():
        if k in SEALED_RECORD:
            assert order <= SEALED_RECORD[k][0] - 1


def test_cyclic_family_is_strictly_weaker_at_k_equals_six() -> None:
    """R(3,6) = 18 needs a 17-vertex witness, but no cyclic graph on 17 vertices works --
    the standard demonstration that exhausting this family is not the same as solving the
    Ramsey problem."""

    assert SEALED_CYCLIC_ORDERS[6] == 16
    assert SEALED_RECORD[6][0] - 1 == 17
    assert exhaustive_search(17, 6).witnesses == []


# --------------------------------------------------------------------------------------
# Frontier witnesses found by this module.
# --------------------------------------------------------------------------------------

FRONTIER = {
    13: (57, [8, 9, 11, 13, 15, 25, 32, 42, 44, 46, 48, 49]),
    14: (63, [5, 6, 16, 18, 26, 30, 33, 37, 45, 47, 57, 58]),
    15: (72, [3, 10, 12, 14, 18, 33, 34, 38, 39, 54, 58, 60, 62, 69]),
}


@pytest.mark.parametrize("k", sorted(FRONTIER))
def test_frontier_witnesses_verify_exactly(k: int) -> None:
    order, residues = FRONTIER[k]
    checks = verify_colouring(order, residues, k)
    assert checks["valid"] is True
    assert checks["triangle_free"] is True
    assert checks["triangle_witness"] is None
    assert checks["independence_number"] == k - 1
    assert checks["certifies"] == f"R(3,{k}) >= {order + 1}"


@pytest.mark.parametrize("k", sorted(FRONTIER))
def test_frontier_witnesses_do_not_beat_the_published_bound(k: int) -> None:
    """The honest expected outcome, written down as an assertion so a future improvement
    breaks this test loudly instead of passing quietly."""

    order, _ = FRONTIER[k]
    verdict = compare_to_record(k, order)
    assert verdict["beats_published_lower_bound"] is False
    assert verdict["gap_to_published"] == SEALED_RECORD[k][0] - (order + 1)
    assert verdict["gap_to_published"] > 0


def test_record_gate_would_fire_on_a_larger_order() -> None:
    """The gate is a real integer comparison, not a constant False."""

    verdict = compare_to_record(13, SEALED_RECORD[13][0])
    assert verdict["beats_published_lower_bound"] is True


def test_no_cyclic_witness_at_the_orders_that_would_improve_the_record() -> None:
    """Improving R(3,13) needs a cyclic triangle-free graph on 61 vertices with alpha <= 12."""

    for order in (58, 59, 60, 61):
        assert exhaustive_search(order, 13).witnesses == []


# --------------------------------------------------------------------------------------
# Certificates.
# --------------------------------------------------------------------------------------


def test_certificate_round_trip_and_tamper_detection() -> None:
    outcomes = list(scan_orders(5, 1, 16))
    cert = seal_certificate(5, 1, 16, outcomes)
    assert cert["max_cyclic_order"] == 13
    assert cert["exhaustive"] is True
    assert cert["sampling_used"] is False
    assert verify_certificate(cert)["valid"] is True

    tampered = dict(cert)
    tampered["max_cyclic_order"] = 14
    assert verify_certificate(tampered)["valid"] is False

    forged = dict(cert)
    witness = dict(cert["witness"])  # type: ignore[arg-type]
    witness["independence_number"] = 3
    forged["witness"] = witness
    report = verify_certificate(forged)
    assert report["valid"] is False


def test_certificate_carries_no_floats() -> None:
    outcomes = list(scan_orders(4, 1, 12))
    cert = seal_certificate(4, 1, 12, outcomes)

    def walk(value: object) -> None:
        assert not isinstance(value, float), value
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(cert)
    json.dumps(cert)


def test_seal_requires_a_contiguous_scan() -> None:
    outcomes = [exhaustive_search(n, 4) for n in (1, 2, 4)]
    with pytest.raises(ValueError):
        seal_certificate(4, 1, 4, outcomes)


def test_verify_colouring_rejects_broken_input() -> None:
    assert verify_colouring(9, [1, 2], 3)["valid"] is False           # has a triangle
    assert verify_colouring(9, [1], 3)["valid"] is False              # not symmetric
    assert verify_colouring(9, [12], 3)["valid"] is False             # residue out of range
    assert verify_colouring(11, [1, 10], 3)["valid"] is False         # alpha too large


def test_cli_emits_a_verifiable_certificate(tmp_path, capsys) -> None:
    out = tmp_path / "cert.json"
    assert main(["--k", "4", "--lo", "1", "--hi", "12", "--output", str(out)]) == 0
    cert = json.loads(out.read_text(encoding="utf-8"))
    assert cert["max_cyclic_order"] == 8
    assert verify_certificate(cert)["valid"] is True
    assert "Radziszowski" in cert["record_comparison"]["record_source"]
    capsys.readouterr()


# --------------------------------------------------------------------------------------
# The dihedral family.
# --------------------------------------------------------------------------------------


def _brute_force_dihedral(m: int, k: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every (A, B) pair with no reduction at all: all symmetric A, all 2**m subsets B."""

    n = 2 * m
    classes = difference_classes(m)
    found = []
    for abits in range(1 << len(classes)):
        rotations: list[int] = []
        for i, cls in enumerate(classes):
            if (abits >> i) & 1:
                rotations.extend(cls)
        for bbits in range(1 << m):
            reflections = [j for j in range(m) if (bbits >> j) & 1]
            adj = dihedral_adjacency(m, rotations, reflections)
            if not adjacency_is_triangle_free(n, adj):
                continue
            if adjacency_independence_number(n, adj) <= k - 1:
                found.append((tuple(sorted(rotations)), tuple(reflections)))
    return found


@pytest.mark.parametrize("m", [5, 7, 9])
@pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8])
def test_dihedral_search_agrees_with_brute_force_on_existence(m: int, k: int) -> None:
    """The conjugation reduction ("0 in B without loss") and the degree window must not
    lose a witness -- checked against a loop over every one of the 2**m reflection sets."""

    brute = _brute_force_dihedral(m, k)
    found = dihedral_search(m, k, stop_at_first=False)
    assert bool(found["witnesses"]) == bool(brute), (m, k)


@pytest.mark.parametrize("m", [5, 7, 9, 11])
def test_dihedral_witnesses_are_triangle_free_with_the_claimed_independence(m: int) -> None:
    for k in (6, 7, 8):
        found = dihedral_search(m, k, stop_at_first=False)
        for w in found["witnesses"]:
            adj = dihedral_adjacency(m, w["rotations"], w["reflections"])
            assert adjacency_is_triangle_free(2 * m, adj)
            assert adjacency_independence_number(2 * m, adj) == w["independence_number"]
            assert w["independence_number"] <= k - 1
            assert w["degree"] == len(w["rotations"]) + len(w["reflections"])


def test_dihedral_adjacency_is_a_symmetric_regular_graph() -> None:
    m = 11
    adj = dihedral_adjacency(m, [1, 10, 3, 8], [0, 2, 5])
    n = 2 * m
    degree = adj[0].bit_count()
    for u in range(n):
        assert adj[u].bit_count() == degree
        assert not (adj[u] >> u) & 1
        for v in range(n):
            assert ((adj[u] >> v) & 1) == ((adj[v] >> u) & 1)
    assert degree == 4 + 3


def test_dihedral_adjacency_rejects_bad_connection_sets() -> None:
    with pytest.raises(ValueError):
        dihedral_adjacency(7, [0], [1])
    with pytest.raises(ValueError):
        dihedral_adjacency(7, [1], [1])          # not closed under inversion
    with pytest.raises(ValueError):
        dihedral_search(8, 5)                    # even m breaks the conjugation reduction


def test_dihedral_cosets_induce_the_circulant() -> None:
    """Both cosets of the rotation subgroup induce C_m(A); that is what bounds alpha below."""

    m, rotations = 13, [1, 12, 5, 8]
    adj = dihedral_adjacency(m, rotations, [0, 3])
    circ = build_adjacency(m, connection_from_residues(m, rotations))
    for i in range(m):
        assert adj[i] & ((1 << m) - 1) == circ[i]
        assert (adj[m + i] >> m) == circ[i]
