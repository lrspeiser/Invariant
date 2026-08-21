"""An exhaustion is only worth as much as its claim to have missed nothing.

Two things can go wrong and neither shows up as an error: the scoring can be wrong, so the
wrong threshold is refuted; or the pruning can be unsound, so a solution is skipped and
"infeasible" is reported for something feasible.  The tests below attack both -- the score
against hand-checkable triangles, the pruning by re-running searches with the symmetry
reduction switched off and demanding the same verdict, and the whole pipeline against every
published term of A248866 that runs fast enough to sit in a test suite.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from sigma_theory_compiler.discrete_heilbronn_grid import (
    ESTABLISHED_TERMS,
    PUBLISHED_TERMS,
    TERMS_SOURCE,
    _orbit_minimal,
    _pair_masks,
    branch_verdict,
    certify_min_double_area,
    double_area,
    enumerate_branches,
    exists_configuration,
    grid_points,
    maximal_min_double_area,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

FAST_TERMS = [3, 4, 5, 6, 7, 8, 9]


def test_double_area_is_twice_the_area_and_vanishes_on_collinear_points() -> None:
    assert double_area((0, 0), (2, 0), (0, 2)) == 4      # right triangle, area 2
    assert double_area((0, 0), (0, 2), (2, 0)) == 4      # orientation does not matter
    assert double_area((0, 0), (1, 1), (2, 2)) == 0      # collinear
    assert double_area((0, 0), (3, 0), (0, 3)) == 9      # matches the a(4) witness scale


def test_scoring_rejects_selections_that_are_not_legal() -> None:
    assert certify_min_double_area([(0, 0), (0, 2), (2, 0)], 3) == 4
    with pytest.raises(ValueError):
        certify_min_double_area([(0, 0), (0, 2), (3, 0)], 3)      # off the grid
    with pytest.raises(ValueError):
        certify_min_double_area([(0, 0), (0, 0), (2, 0)], 3)      # repeated point
    with pytest.raises(ValueError):
        certify_min_double_area([(0, 0), (0, 2)], 3)              # not enough for a triangle
    with pytest.raises(TypeError):
        certify_min_double_area([(0.0, 0), (0, 2), (2, 0)], 3)


def test_pair_masks_agree_with_a_direct_triple_scan() -> None:
    n, threshold = 5, 6
    points, masks = _pair_masks(n, threshold)
    size = len(points)
    for i, j in combinations(range(size), 2):
        expected = 0
        for k in range(size):
            if k in (i, j):
                continue
            if double_area(points[i], points[j], points[k]) >= threshold:
                expected |= 1 << k
        assert masks[i][j] == expected
        assert masks[j][i] == expected


def test_orbit_minimal_points_are_a_genuine_eighth_of_the_grid() -> None:
    for n in (5, 8, 13):
        points = grid_points(n)
        minimal = [p for p in points if _orbit_minimal(p, n)]
        # Every point's orbit contains exactly one minimal representative.
        m = n - 1
        for x, y in points:
            orbit = {(x, y), (y, x), (m - x, y), (x, m - y),
                     (m - x, m - y), (y, m - x), (m - y, x), (m - y, m - x)}
            assert len(orbit & set(minimal)) == 1
        assert len(minimal) * 8 >= len(points)


@pytest.mark.parametrize("n", [5, 6, 7])
def test_symmetry_reduction_never_changes_a_verdict(n: int) -> None:
    """The eightfold cut is the only unsound-looking step; run without it and compare."""
    published = PUBLISHED_TERMS[n]
    for threshold in (published - 1, published, published + 1):
        reduced = exists_configuration(n, threshold, use_symmetry=True)
        complete = exists_configuration(n, threshold, use_symmetry=False)
        assert reduced.feasible == complete.feasible
        assert reduced.nodes <= complete.nodes


def test_feasibility_is_monotone_in_the_threshold() -> None:
    n = 8
    seen_false = False
    for threshold in range(1, 9):
        verdict = exists_configuration(n, threshold)
        if seen_false:
            assert not verdict.feasible, "feasibility must not return once it has stopped"
        seen_false = seen_false or not verdict.feasible


@pytest.mark.parametrize("n", FAST_TERMS)
def test_exhaustion_reproduces_the_published_term(n: int) -> None:
    value, witness, refutation = maximal_min_double_area(n)
    assert value == PUBLISHED_TERMS[n]
    assert witness.feasible and witness.threshold == value
    assert witness.witness is not None and len(witness.witness) == n
    # The witness is re-scored from scratch, not taken on the search's word.
    assert certify_min_double_area(list(witness.witness), n) == value
    # And the load-bearing half: nothing at all exists one step higher.
    assert refutation.threshold == value + 1
    assert refutation.feasible is False
    assert refutation.witness is None


def test_verdict_payload_is_canonical_and_claims_no_novelty() -> None:
    verdict = exists_configuration(5, 6)
    payload = verdict.to_payload()
    assert canonical_sha256(payload) == verdict.digest()
    assert payload["enumeration"].startswith("complete over all")
    assert payload["absence_establishes_novelty"] is False


def test_published_terms_are_transcribed_with_their_source() -> None:
    assert "A248866" in TERMS_SOURCE
    assert "oeis.org/A248866" in TERMS_SOURCE
    assert sorted(PUBLISHED_TERMS) == list(range(3, 14))
    assert PUBLISHED_TERMS[4] == 9  # the one term larger than its neighbours
    assert all(isinstance(v, int) and v > 0 for v in PUBLISHED_TERMS.values())


def test_root_splitting_partitions_the_same_search() -> None:
    """Splitting by first point is how the work parallelises; it must not change the answer."""
    n, threshold = 7, 5
    roots = [i for i, p in enumerate(grid_points(n)) if _orbit_minimal(p, n)]
    whole = exists_configuration(n, threshold)
    pieces = [exists_configuration(n, threshold, roots=[r]) for r in roots]
    assert whole.feasible == any(piece.feasible for piece in pieces)

    n, threshold = 7, 6  # infeasible, so every piece must independently agree
    whole = exists_configuration(n, threshold)
    pieces = [exists_configuration(n, threshold, roots=[r]) for r in roots]
    assert whole.feasible is False
    assert not any(piece.feasible for piece in pieces)
    assert sum(piece.nodes for piece in pieces) == whole.nodes


def test_terms_established_here_have_re_verifiable_witnesses() -> None:
    """The lower half of each self-settled term is cheap, so CI re-proves it every run."""
    for n, term in ESTABLISHED_TERMS.items():
        assert term.n == n
        assert len(term.witness) == n
        assert len({p for p in term.witness}) == n
        # Re-scored from the raw coordinates; the stored value is not consulted.
        assert certify_min_double_area(list(term.witness), n) == term.value
        # The refutation is one step above and is what makes the term exact.
        assert term.refutation_threshold == term.value + 1
        assert term.refutation_nodes > 10**8
        assert term.beyond_published == (n > max(PUBLISHED_TERMS))
        if n in PUBLISHED_TERMS:
            assert term.value == PUBLISHED_TERMS[n]


def test_established_terms_do_not_overwrite_the_published_ones() -> None:
    """Two tables, kept apart on purpose: what OEIS prints, and what this repo computed."""
    assert set(ESTABLISHED_TERMS) & set(PUBLISHED_TERMS) == set()
    assert all(term.beyond_published for term in ESTABLISHED_TERMS.values())


def test_branch_split_reconstructs_the_whole_sweep_exactly() -> None:
    """The parallel driver decides one (first, second) prefix per task; the pieces must
    reassemble into the unsplit search, verdict and node count alike."""
    n, threshold = 7, 6            # infeasible, so every branch must agree independently
    branches = enumerate_branches(n)
    whole = exists_configuration(n, threshold)
    pieces = [branch_verdict(n, threshold, prefix) for prefix in branches]
    assert whole.feasible is False
    assert not any(piece.feasible for piece in pieces)
    # The split is conservative rather than identical: the unsplit search can abandon a whole
    # tail of second points at once when even taking all of them cannot reach n, whereas the
    # driver hands each of those out as its own (trivially dead) task.  So the shard total
    # brackets the unsplit count -- it may not exceed it, and may fall short by at most the
    # one node per branch that the unsplit search spent choosing that second point.
    shard_total = sum(piece.nodes for piece in pieces)
    assert shard_total <= whole.nodes <= shard_total + len(branches)

    n, threshold = 7, 5            # feasible: at least one branch must carry a witness
    branches = enumerate_branches(n)
    pieces = [branch_verdict(n, threshold, prefix) for prefix in branches]
    assert any(piece.feasible for piece in pieces)
    for piece in pieces:
        if piece.witness is not None:
            assert certify_min_double_area(list(piece.witness), n) >= threshold


def test_branch_enumeration_counts_what_a_complete_sweep_must_cover() -> None:
    for n in (7, 14):
        branches = enumerate_branches(n)
        roots = [i for i, p in enumerate(grid_points(n)) if _orbit_minimal(p, n)]
        assert len(branches) == sum(n * n - r - 1 for r in roots)
        assert len(set(branches)) == len(branches)
    assert len(enumerate_branches(14)) == 4564   # the count the a(14) sweep reported


def test_branch_verdict_rejects_a_malformed_prefix() -> None:
    with pytest.raises(ValueError):
        branch_verdict(7, 5, [4, 2])
    with pytest.raises(ValueError):
        branch_verdict(7, 5, [3, 3])
