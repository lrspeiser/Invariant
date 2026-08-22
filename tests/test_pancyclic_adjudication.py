"""Independent adjudication of the m(38)/m(39)/m(40) minimal-pancyclic claim.

This file is deliberately adversarial towards
:mod:`sigma_theory_compiler.pancyclic_chord_search`.  It shares no code with the
cycle-space enumerator it is checking: the witnesses are re-typed here from the claim
text rather than imported, and the cycle counts are recomputed by two enumerators that
work on completely different principles.

What is checked
---------------
1.  **The record.**  ``SEALED_TABLE`` is compared entry by entry against Griffin,
    "Minimal Pancyclicity", arXiv:1312.0274 (2013), Table 1, transcribed here from the
    paper itself, and against the twenty terms of OEIS A105206 (which stops at n = 22).
    The published table of exact values ends at n = 37 with m(37) = 42.  Separately, the
    module under audit understates the record: there *is* a published upper bound past
    n = 37.  Sridharan (1978), as restated by George, Marr and Wallis (JCMCC 86, 2013,
    p.126), gives m(v) <= v + 6 for 37 <= v <= 52.  Together with the Bondy/Shi lower
    bound m(v) >= v + 5 for 34 <= v <= 65 this brackets m(38), m(39) and m(40) within a
    single edge, and each witness lands on the lower end -- so the witnesses improve the
    published upper bound by one edge rather than filling a vacuum.
2.  **The upper bound.**  Each witness is re-verified pancyclic by (a) a depth-first
    simple-cycle enumerator rooted at the smallest vertex and (b) an exhaustive
    in/out assignment of every edge of ``C_n`` for every one of the ``2^k`` chord
    subsets.  Neither touches GF(2).  All arithmetic is integer.
3.  **The lower bound.**  ``m(n) >= n + 5`` for ``34 <= n <= 65`` is re-derived from the
    Bondy/Shi count without any search, and re-checked directly against the definition
    of ``counting_lower_bound_chords``.
4.  **Rejection of invalid objects.**  Non-pancyclic graphs, chord-deleted witnesses and
    perturbed witnesses must all be reported as not pancyclic.
5.  **A defect in the search reduction, pinned as a test.**  ``_rotation_reduced_sets``
    is *not* orbit-complete: it emits only chord sets with exactly one chord incident to
    vertex 0, so it silently skips every chord set in which each chord-endpoint vertex
    carries at least two chords.  This does not touch the three witnesses -- a
    construction is verified by cycle enumeration, not by the search that found it --
    but it does mean the module's ``exhaustive_search`` is not exhaustive.  The skipped
    class is enumerated here directly and shown to contain no pancyclic graph at the
    orders where the module draws a negative conclusion, so the conclusions survive.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest

from sigma_theory_compiler.pancyclic_chord_search import (
    SEALED_TABLE,
    SEALED_TABLE_LAST_N,
    WITNESSES,
    _rotation_reduced_sets,
    candidate_chords,
    counting_lower_bound_chords,
    cycle_spectrum,
    record_gate,
    verify_witness,
)

# Witnesses re-typed from the claim, not imported, so that a corrupted WITNESSES table
# cannot make this file agree with itself.
CLAIMED_WITNESSES = {
    38: ((12, 14), (13, 31), (14, 27), (23, 30), (28, 31)),
    39: ((1, 36), (3, 36), (19, 34), (29, 38), (35, 37)),
    40: ((2, 30), (21, 28), (22, 29), (27, 31), (27, 32)),
}
CLAIMED_CYCLE_COUNTS = {38: 41, 39: 45, 40: 48}

# Griffin, arXiv:1312.0274, Table 1, transcribed from the paper: n -> m(n).
GRIFFIN_TABLE_1 = {
    3: 3, 4: 5, 5: 6, 6: 8, 7: 9, 8: 10, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16,
    14: 17, 15: 19, 16: 20, 17: 21, 18: 22, 19: 23, 20: 24, 21: 25, 22: 26, 23: 27,
    24: 28, 25: 30, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38,
    34: 39, 35: 40, 36: 41, 37: 42,
}

# OEIS A105206 DATA as served by oeis.org, offset 3: m(3), m(4), ..., m(22).
OEIS_A105206 = (3, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26)

# The published UPPER bound at orders past Griffin's table.  M. R. Sridharan, "On an
# extremal problem concerning pancyclic graphs", J. Math. Phys. Sci. 12 (1978), 297-306,
# gives constructions which J. C. George, A. Marr and W. D. Wallis, "Minimal Pancyclic
# Graphs", JCMCC 86 (2013), 125-133, p.126 restate as
#     21 <= v <= 36 : m(v) <= v + 5
#     37 <= v <= 52 : m(v) <= v + 6
#     53 <= v <= 84 : m(v) <= v + 7
# so the best published upper bound at v = 38, 39, 40 is v + 6.  Griffin's 5-chord
# construction improves this only at v = 37, where it gives m(37) = 42 = 37 + 5.
SRIDHARAN_EXTRA_EDGES = {(21, 36): 5, (37, 52): 6, (53, 84): 7}

# GMW 2013 determine m(v) only for v <= 22 -- "the cases v >= 23 remain open" (p.132),
# and their sequence is OEIS A105206.  Griffin 2013 is the frontier, and it ends at 37.
GMW_LAST_DETERMINED_ORDER = 22


def _sridharan_upper_bound(v: int) -> int:
    for (low, high), extra in SRIDHARAN_EXTRA_EDGES.items():
        if low <= v <= high:
            return v + extra
    raise AssertionError(f"no Sridharan range covers v={v}")


# ---------------------------------------------------------------------------------
# Two enumerators that share nothing with cycle_spectrum().
# ---------------------------------------------------------------------------------
def _edges_of(n: int, chords) -> set[frozenset[int]]:
    edges = {frozenset((v, (v + 1) % n)) for v in range(n)}
    assert len(edges) == n
    for u, v in chords:
        e = frozenset((u, v))
        assert len(e) == 2 and e not in edges, f"chord {(u, v)} is not a new simple edge"
        edges.add(e)
    return edges


def cycles_by_dfs(n: int, chords) -> list[frozenset[frozenset[int]]]:
    """Every simple cycle, found by growing paths from the cycle's smallest vertex."""

    adj: dict[int, set[int]] = {v: set() for v in range(n)}
    for e in _edges_of(n, chords):
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    found: list[frozenset[frozenset[int]]] = []

    def walk(root: int, cur: int, path: list[int], seen: set[int]) -> None:
        for nxt in adj[cur]:
            if nxt == root:
                if len(path) >= 3 and path[1] < path[-1]:
                    size = len(path)
                    found.append(frozenset(
                        frozenset((path[i], path[(i + 1) % size])) for i in range(size)))
                continue
            if nxt < root or nxt in seen:
                continue
            path.append(nxt)
            seen.add(nxt)
            walk(root, nxt, path, seen)
            path.pop()
            seen.discard(nxt)

    for root in range(n):
        walk(root, root, [root], {root})
    assert len(set(found)) == len(found), "path enumerator emitted a duplicate cycle"
    return found


def cycles_by_edge_assignment(n: int, chords) -> list[frozenset[frozenset[int]]]:
    """Every simple cycle, found by deciding each edge of ``C_n`` in or out, per chord subset."""

    cyc = [frozenset((v, (v + 1) % n)) for v in range(n)]
    found: list[frozenset[frozenset[int]]] = []
    for r in range(len(chords) + 1):
        for picked in combinations(range(len(chords)), r):
            chosen = [frozenset(chords[i]) for i in picked]
            deg: Counter[int] = Counter()
            for e in chosen:
                for v in e:
                    deg[v] += 1
            if any(d > 2 for d in deg.values()):
                continue
            found.extend(_assign(n, cyc, chosen, deg))
    assert len(set(found)) == len(found), "assignment enumerator emitted a duplicate cycle"
    return found


def _assign(n, cyc, chosen, base_deg):
    results = []
    use = [False] * n
    deg = Counter(base_deg)

    def rec(i: int) -> None:
        if i == n:
            if any(deg[v] not in (0, 2) for v in range(n)):
                return
            edgeset = set(chosen) | {cyc[j] for j in range(n) if use[j]}
            if not edgeset:
                return
            adj: dict[int, list[int]] = {}
            for e in edgeset:
                a, b = tuple(e)
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            if any(len(x) != 2 for x in adj.values()):
                return
            start = next(iter(adj))
            prev, cur, steps = None, start, 0
            while True:
                a, b = adj[cur]
                nxt = a if a != prev else b
                prev, cur = cur, nxt
                steps += 1
                if cur == start:
                    break
            if steps == len(edgeset):
                results.append(frozenset(edgeset))
            return
        a, b = i, (i + 1) % n
        for choice in (False, True):
            if choice:
                deg[a] += 1
                deg[b] += 1
            ok = deg[a] <= 2 and deg[b] <= 2 and (i == 0 or deg[i] in (0, 2))
            if ok:
                use[i] = choice
                rec(i + 1)
                use[i] = False
            if choice:
                deg[a] -= 1
                deg[b] -= 1

    rec(0)
    return results


# ---------------------------------------------------------------------------------
# 1. The record
# ---------------------------------------------------------------------------------
def test_sealed_table_is_griffin_table_1_verbatim() -> None:
    assert {n: size for n, (_, size) in SEALED_TABLE.items()} == GRIFFIN_TABLE_1
    assert SEALED_TABLE_LAST_N == 37
    assert GRIFFIN_TABLE_1[37] == 42
    assert 38 not in GRIFFIN_TABLE_1


def test_sealed_table_agrees_with_oeis_a105206_where_oeis_reaches() -> None:
    for offset, value in enumerate(OEIS_A105206):
        n = 3 + offset
        assert SEALED_TABLE[n][1] == value, n
    # OEIS stops at n = 22; it is behind the published table, not ahead of it.
    assert 3 + len(OEIS_A105206) - 1 == 22


# ---------------------------------------------------------------------------------
# 2. The upper bound: the witnesses, by two unrelated enumerators
# ---------------------------------------------------------------------------------
def test_module_witnesses_are_the_claimed_witnesses() -> None:
    assert {n: tuple(sorted(c)) for n, c in WITNESSES.items()} == {
        n: tuple(sorted(c)) for n, c in CLAIMED_WITNESSES.items()
    }


@pytest.mark.parametrize("n", sorted(CLAIMED_WITNESSES))
def test_witness_is_pancyclic_under_two_independent_enumerators(n: int) -> None:
    chords = CLAIMED_WITNESSES[n]
    edges = _edges_of(n, chords)
    assert len(edges) == n + 5, "witness must have exactly n + 5 edges"

    by_dfs = cycles_by_dfs(n, chords)
    by_assign = cycles_by_edge_assignment(n, chords)

    # identical cycles, not merely identical counts
    assert set(by_dfs) == set(by_assign)
    assert len(by_dfs) == CLAIMED_CYCLE_COUNTS[n]
    assert len(by_dfs) <= (1 << 6) - 1, "Shi bound"

    spectrum = sorted({len(c) for c in by_dfs})
    assert spectrum == list(range(3, n + 1))

    # and the module under audit must say the same thing
    certificate = verify_witness(n, chords)
    assert certificate["pancyclic"] is True
    assert certificate["edges"] == n + 5
    assert certificate["cycles_found"] == len(by_dfs)
    assert certificate["spectrum"] == spectrum
    assert certificate["elements_examined"] == certificate["elements_expected"] == 63
    assert record_gate(certificate)["sealed_edges"] is None


# ---------------------------------------------------------------------------------
# 3. The lower bound, with no search anywhere in it
# ---------------------------------------------------------------------------------
def test_counting_bound_forces_five_chords_from_34_to_65() -> None:
    for n in range(34, 66):
        assert (1 << 5) - 1 < n - 2, n          # four chords cannot supply n-2 cycles
        assert (1 << 6) - 1 >= n - 2, n         # five chords are not excluded by counting
        assert counting_lower_bound_chords(n) == 5, n
    assert counting_lower_bound_chords(33) == 4
    assert counting_lower_bound_chords(66) == 6


def test_the_three_orders_are_pinned_exactly() -> None:
    for n, expected in ((38, 43), (39, 44), (40, 45)):
        assert n + counting_lower_bound_chords(n) == expected      # lower bound
        assert verify_witness(n, CLAIMED_WITNESSES[n])["edges"] == expected  # upper bound
        assert n > SEALED_TABLE_LAST_N


def test_the_witnesses_improve_the_published_upper_bound_by_one_edge() -> None:
    """There WAS a published upper bound at these orders, and each witness beats it by 1.

    Sridharan's construction gives m(v) <= v + 6 for 37 <= v <= 52, and Griffin improves
    that only at v = 37.  So before these witnesses the published brackets were
    m(38) in {43,44}, m(39) in {44,45}, m(40) in {45,46} -- each undetermined by exactly
    one edge.  Each witness lands on the lower end and closes its bracket.
    """
    for n in (38, 39, 40):
        published_upper = _sridharan_upper_bound(n)
        published_lower = n + counting_lower_bound_chords(n)
        assert published_upper == n + 6
        assert published_lower == n + 5
        assert published_upper - published_lower == 1, "bracket was one edge wide"

        witnessed = verify_witness(n, CLAIMED_WITNESSES[n])
        assert witnessed["pancyclic"] is True
        assert witnessed["edges"] == published_lower < published_upper

    # Sanity: the same published bound reproduces, not beats, the table where it overlaps.
    assert _sridharan_upper_bound(36) == SEALED_TABLE[36][1] == 41
    # and it is one edge LOOSE at v = 37, which is exactly what Griffin's construction fixed.
    assert _sridharan_upper_bound(37) == 43
    assert SEALED_TABLE[37][1] == 42


def test_george_marr_wallis_stop_at_order_22() -> None:
    """GMW 2013 determine m(v) only to v = 22; Griffin 2013 is the frontier, ending at 37."""
    assert GMW_LAST_DETERMINED_ORDER == 3 + len(OEIS_A105206) - 1 == 22
    assert SEALED_TABLE_LAST_N == 37 > GMW_LAST_DETERMINED_ORDER
    for offset, value in enumerate(OEIS_A105206):
        assert SEALED_TABLE[3 + offset][1] == value


# ---------------------------------------------------------------------------------
# 4. Invalid objects must be rejected
# ---------------------------------------------------------------------------------
def test_invalid_objects_are_rejected() -> None:
    bare = verify_witness(38, ())
    assert bare["pancyclic"] is False and bare["spectrum"] == [38]
    assert record_gate(bare)["verdict"] == "not_pancyclic"

    bunched = verify_witness(38, ((0, 2), (1, 3), (2, 4), (3, 5), (4, 6)))
    assert bunched["pancyclic"] is False
    assert len(bunched["missing_lengths"]) == 27

    for n, chords in CLAIMED_WITNESSES.items():
        for dropped in range(5):
            reduced = tuple(c for i, c in enumerate(chords) if i != dropped)
            assert verify_witness(n, reduced)["pancyclic"] is False, (n, dropped)

    # a four-chord graph at n = 38 is below the proven bound and cannot be pancyclic
    four = verify_witness(38, ((0, 2), (5, 20), (10, 30), (15, 35)))
    assert four["pancyclic"] is False
    assert four["counting_lower_bound_chords"] == 5


def test_a_moved_chord_breaks_the_n40_witness_every_time() -> None:
    n = 40
    chords = list(CLAIMED_WITNESSES[n])
    survivors = 0
    tried = 0
    for i in range(5):
        for delta in (-2, -1, 1, 2):
            for end in (0, 1):
                moved = list(chords)
                u, v = moved[i]
                moved[i] = ((u + delta) % n, v) if end == 0 else (u, (v + delta) % n)
                a, b = moved[i]
                if a == b or (b - a) % n in (1, n - 1):
                    continue
                if len({frozenset(c) for c in moved}) != 5:
                    continue
                tried += 1
                if verify_witness(n, moved)["pancyclic"]:
                    survivors += 1
    assert tried == 38
    assert survivors == 0


# ---------------------------------------------------------------------------------
# 5. The search reduction is NOT orbit-complete.  Pinned, with the damage bounded.
# ---------------------------------------------------------------------------------
def _orbit(n, chords):
    out = set()
    for shift in range(n):
        for sign in (1, -1):
            img = []
            for u, v in chords:
                a, b = (sign * u + shift) % n, (sign * v + shift) % n
                img.append((min(a, b), max(a, b)))
            out.add(tuple(sorted(img)))
    return out


def _all_orbit_representatives(n, k):
    seen: set[tuple] = set()
    reps = []
    for chords in combinations(candidate_chords(n), k):
        chords = tuple(sorted(chords))
        if chords in seen:
            continue
        seen |= _orbit(n, chords)
        reps.append(chords)
    return reps


@pytest.mark.parametrize(
    "n,k,expected_orbits,expected_missed",
    [(9, 2, 27, 0), (9, 3, 182, 3), (10, 3, 365, 4), (11, 3, 644, 5)],
)
def test_rotation_reduction_skips_orbits(n, k, expected_orbits, expected_missed) -> None:
    generated = {tuple(sorted(s)) for s in _rotation_reduced_sets(n, k)}
    reps = _all_orbit_representatives(n, k)
    assert len(reps) == expected_orbits
    missed = [s for s in reps if not (_orbit(n, s) & generated)]
    assert len(missed) == expected_missed, missed
    # every missed orbit has the same shape: no chord-endpoint vertex of chord-degree 1,
    # so no rotation leaves exactly one chord at vertex 0, which is all the generator emits.
    for chords in missed:
        deg: Counter[int] = Counter()
        for u, v in chords:
            deg[u] += 1
            deg[v] += 1
        assert min(deg.values()) >= 2
    for chords in generated:
        incident_to_zero = sum(1 for u, v in chords if u == 0 or v == 0)
        assert incident_to_zero == 1


def _skipped_class(n, k):
    """Chord k-sets every one of whose endpoint vertices carries >= 2 chords."""
    out: set[tuple] = set()
    for span in range(3, k + 1):
        for verts in combinations(range(n), span):
            pairs = [
                (a, b) for a, b in combinations(verts, 2)
                if (b - a) % n not in (1, n - 1)
            ]
            if len(pairs) < k:
                continue
            for chords in combinations(pairs, k):
                deg: Counter[int] = Counter()
                for u, v in chords:
                    deg[u] += 1
                    deg[v] += 1
                if min(deg.values()) >= 2:
                    out.add(tuple(sorted(chords)))
    return sorted(out)


def test_the_skipped_class_contains_no_pancyclic_graph_where_it_matters() -> None:
    """The module concludes 'no pancyclic graph' at (15,3) and at (25,4)/(26,4).

    Its own search never looked at the skipped orbits.  Enumerate them here and confirm
    the conclusions happen to survive -- by inspection, not by the module's search.
    """
    for n, k, expected in ((15, 3, 275), (25, 4, 26375), (26, 4, 31668)):
        skipped = _skipped_class(n, k)
        assert len(skipped) == expected, (n, k, len(skipped))
        required = set(range(3, n + 1))
        pancyclic = [
            chords for chords in skipped
            if required <= set(cycle_spectrum(n, chords)["spectrum"])
        ]
        assert pancyclic == [], (n, k, pancyclic[:3])


def test_a_fully_unreduced_search_at_n15_k3_confirms_m15_is_19() -> None:
    """No symmetry reduction at all: every one of the C(90,3) chord triples."""
    chords_available = candidate_chords(15)
    assert len(chords_available) == 90
    required = set(range(3, 16))
    examined = 0
    hits = 0
    for chords in combinations(chords_available, 3):
        examined += 1
        if required <= set(cycle_spectrum(15, chords)["spectrum"]):
            hits += 1
    assert examined == 90 * 89 * 88 // 6 == 117480
    assert hits == 0
    assert SEALED_TABLE[15] == (4, 19)


def test_no_float_anywhere_in_this_adjudication() -> None:
    assert not isinstance(counting_lower_bound_chords(38), float)
    certificate = verify_witness(38, CLAIMED_WITNESSES[38])

    def walk(value) -> None:
        assert not isinstance(value, float), value
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(certificate)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
