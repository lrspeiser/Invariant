from __future__ import annotations

import itertools
import json
import pathlib

import pytest

from sigma_theory_compiler.weak_sidon_exhaustion import (
    A345731_PUBLISHED,
    CoverageCertificate,
    enumerate_prefix_cells,
    is_weak_sidon,
    min_diameter_lower_bound,
    search_within,
    witness_diameter,
)

# Published witnesses quoted in the OEIS entries, used as fixed reference data.
A345731_16_WITNESS = (0, 3, 5, 6, 32, 49, 59, 68, 93, 106, 118, 126, 130, 134, 141, 148)
A004133_17_WITNESS = (
    0, 5, 9, 10, 11, 43, 62, 75, 88, 112, 115, 129, 136, 143, 151, 159, 171,
)


def _ladder(nmax: int, engine: str) -> list[int]:
    """Rebuild A345731(2..nmax) from nothing but the definition."""

    w = [0, 0]
    for n in range(2, nmax + 1):
        L = max(w[n - 1], min_diameter_lower_bound(n))
        w.append(L)
        while True:
            w[n] = L
            found, nodes, wit = search_within(n, L, w, engine=engine)
            assert nodes >= 0, "node cap must not fire in tests"
            if found:
                assert is_weak_sidon(wit)
                assert witness_diameter(wit) == L
                break
            L += 1
        w[n] = L
    return w


def test_reference_predicate_matches_brute_force_definition() -> None:
    for n in range(2, 6):
        for combo in itertools.combinations(range(12), n):
            sums = [a + b for a, b in itertools.combinations(combo, 2)]
            assert is_weak_sidon(combo) == (len(set(sums)) == len(sums))


def test_published_witnesses_are_weak_sidon_with_the_stated_diameter() -> None:
    assert is_weak_sidon(A345731_16_WITNESS)
    assert witness_diameter(A345731_16_WITNESS) == A345731_PUBLISHED[16] == 148
    assert is_weak_sidon(A004133_17_WITNESS)
    assert witness_diameter(A004133_17_WITNESS) == 171
    # A004133(17) = 330 is the max pair sum of that 17-element set.
    assert A004133_17_WITNESS[-1] + A004133_17_WITNESS[-2] == 330


def test_three_term_progressions_are_allowed_but_two_are_not() -> None:
    # A weak Sidon set may contain a 3-term AP (unlike a Sidon set) ...
    assert is_weak_sidon((0, 1, 2))
    # ... but no element may be the midpoint of two of them, since that repeats
    # a pair sum.  This is what min_diameter_lower_bound rests on.
    assert not is_weak_sidon((0, 1, 2, 3, 4))
    assert not is_weak_sidon((1, 3, 5, 7))


def test_min_diameter_lower_bound_never_exceeds_the_published_value() -> None:
    for n, value in A345731_PUBLISHED.items():
        assert min_diameter_lower_bound(n) <= value


@pytest.mark.parametrize("engine", ["python", "numba"])
def test_ladder_reproduces_published_terms(engine: str) -> None:
    pytest.importorskip("numba") if engine == "numba" else None
    nmax = 10 if engine == "numba" else 8
    w = _ladder(nmax, engine)
    for n in range(2, nmax + 1):
        assert w[n] == A345731_PUBLISHED[n], (n, w[n], A345731_PUBLISHED[n])


def test_both_engines_agree_node_for_node() -> None:
    pytest.importorskip("numba")
    w = [0, 0, 1, 2, 4, 7, 12, 18, 24, 34, 45]
    for n in range(5, 10):
        for lmax in range(w[n] - 2, w[n] + 1):
            wt = list(w)
            wt[n] = lmax
            a = search_within(n, lmax, wt, engine="python")
            b = search_within(n, lmax, wt, engine="numba")
            assert a[0] == b[0], (n, lmax, a, b)
            assert a[1] == b[1], (n, lmax, a, b)


def test_prefix_cells_partition_the_declared_space_exactly() -> None:
    w = [0, 0, 1, 2, 4, 7, 12, 18, 24, 34, 45, 57, 71, 86, 105, 126, 148, 149]
    n, lmax = 17, 160
    cells = enumerate_prefix_cells(n, lmax, w)
    # Independent recount of the same box, written a different way.
    expected = sum(
        max(0, (lmax - w[n - 2]) - max(a1 + 1, w[3]) + 1)
        for a1 in range(max(1, w[2]), lmax - w[n - 1] + 1)
    )
    assert len(cells) == expected
    assert len({(c.a1, c.a2) for c in cells}) == len(cells)
    for c in cells:
        assert c.a1 < c.a2
        assert c.a1 >= w[2] and c.a1 <= lmax - w[n - 1]
        assert c.a2 >= w[3] and c.a2 <= lmax - w[n - 2]


def test_cell_partition_loses_no_solution() -> None:
    pytest.importorskip("numba")
    w = [0, 0, 1, 2, 4, 7, 12, 18, 24, 34, 45]
    n, lmax = 9, 34
    whole = search_within(n, lmax, w, engine="numba")
    assert whole[0]
    cells = enumerate_prefix_cells(n, lmax, w)
    hits = 0
    nodes = 0
    for c in cells:
        found, nd, wit = search_within(
            n, lmax, w, a1_range=(c.a1, c.a1), a2_range=(c.a2, c.a2), engine="numba"
        )
        nodes += nd
        if found:
            hits += 1
            assert is_weak_sidon(wit)
            assert witness_diameter(wit) <= lmax
    assert hits >= 1
    assert nodes > 0


def test_certificate_completeness_is_a_cardinality_identity() -> None:
    good = CoverageCertificate(
        n=9, lmax=33, wtab=(0, 0, 1), declared_cells=120, traversed_cells=120,
        nodes=5, exhaustive=True, found=False, witness=(),
    )
    assert good.is_complete()
    assert good.as_dict()["declared_cells"] == "120"
    short = CoverageCertificate(
        n=9, lmax=33, wtab=(0, 0, 1), declared_cells=120, traversed_cells=119,
        nodes=5, exhaustive=True, found=False, witness=(),
    )
    assert not short.is_complete()
    capped = CoverageCertificate(
        n=9, lmax=33, wtab=(0, 0, 1), declared_cells=120, traversed_cells=120,
        nodes=5, exhaustive=False, found=False, witness=(),
    )
    assert not capped.is_complete()


def test_node_cap_marks_a_traversal_as_non_exhaustive() -> None:
    pytest.importorskip("numba")
    w = [0, 0, 1, 2, 4, 7, 12, 18, 24, 34, 45, 57]
    found, nodes, _ = search_within(11, 56, w, node_cap=1000, engine="numba")
    assert not found
    assert nodes < 0, "an aborted traversal must report a negative node count"


# --------------------------------------------------------------------------
# recorded coverage certificates
# --------------------------------------------------------------------------

CERT_DIR = pathlib.Path(__file__).resolve().parents[1] / "work" / "weak_sidon"


def _load(name: str) -> list[dict]:
    path = CERT_DIR / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_recorded_chain_agrees_with_the_published_sequence() -> None:
    settled = {}
    for rec in _load("verified_chain.jsonl"):
        assert rec["exhaustive"], rec
        assert rec["declared_cells"] == rec["traversed_cells"], rec
        if rec["found"]:
            settled[int(rec["n"])] = int(rec["lmax"])
    assert settled, "chain log must contain at least one settled stage"
    for n, value in settled.items():
        assert value == A345731_PUBLISHED[n], (n, value)
    # the chain is contiguous from n = 2 and self-contained: every stage that
    # is settled only ever used pruning values for strictly smaller sizes.
    assert sorted(settled) == list(range(2, max(settled) + 1))


def test_open_case_certificates_are_complete_traversals() -> None:
    recs = _load("open_case_17.jsonl")
    assert recs, "expected recorded levels for the open case n = 17"
    for rec in recs:
        n, lmax = int(rec["n"]), int(rec["lmax"])
        assert n == 17
        wtab = [int(v) for v in rec["wtab"]]
        # the declared cell count is reproducible from the module, not trusted
        cells = enumerate_prefix_cells(n, lmax, wtab, depth=3)
        assert len(cells) == int(rec["declared_cells"]), (lmax, len(cells))
        assert rec["traversed_cells"] == rec["declared_cells"], rec
        assert rec["exhaustive"] is True, rec
        assert rec["found"] is False, rec
        assert int(rec["nodes"]) > 0
        # the pruning table must be a valid lower-bound table
        for m in range(2, n):
            assert wtab[m] <= A345731_PUBLISHED[m]


def test_open_case_lower_bound_follows_from_the_certificates() -> None:
    levels = sorted(int(r["lmax"]) for r in _load("open_case_17.jsonl") if not r["found"])
    assert levels == list(range(levels[0], levels[0] + len(levels))), levels
    assert levels[0] <= A345731_PUBLISHED[16] + 1, "must start no higher than the trivial bound"
    bound = levels[-1] + 1
    assert bound > A345731_PUBLISHED[16] + 1, "certificates must improve on monotonicity"
    # the published 16-element optimum is 148, so monotonicity alone gives 149
    assert bound == 151


def test_per_cell_log_matches_its_level_certificate() -> None:
    rows = _load("cells_17_150.jsonl")
    level = next(r for r in _load("open_case_17.jsonl") if r["lmax"] == "150")
    keys = {(r["a1"], r["a2"], r["a3"]) for r in rows}
    assert len(keys) == len(rows) == int(level["declared_cells"])
    assert all(int(r["nodes"]) >= 0 for r in rows), "no cell may have aborted"
    assert not any(r["found"] for r in rows)
    assert sum(abs(int(r["nodes"])) for r in rows) == int(level["nodes"])
