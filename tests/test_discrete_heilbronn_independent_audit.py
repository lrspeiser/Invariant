"""The second procedure is only useful if it is calibrated against answers it cannot fake.

So it is pinned three ways here.  Against literal brute force, which looks at every subset
and cannot be wrong about small ``n``.  Against the eleven witnesses OEIS actually publishes,
scored from their own printed pictures.  And against ``discrete_heilbronn_grid``, which it
must agree with everywhere the two are both cheap enough to run -- because the whole point of
writing it was to have something that could disagree.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import pytest

from sigma_theory_compiler.discrete_heilbronn_grid import (
    ESTABLISHED_TERMS,
    certify_min_double_area,
    enumerate_branches,
    exists_configuration,
)
from sigma_theory_compiler.discrete_heilbronn_independent_audit import (
    OEIS_PUBLISHED,
    REPRODUCED,
    brute_force_maximum,
    decide,
    maximum,
    min_twice_area,
    shard_list,
    twice_area,
)

# The pictures printed in the OEIS worked-examples file a248866.txt, read as (row, column).
# These are the published objects, not this repository's: they are the calibration.
OEIS_WITNESSES: dict[int, tuple[tuple[int, int], ...]] = {
    3: ((0, 0), (0, 2), (2, 0)),
    4: ((0, 0), (0, 3), (3, 0), (3, 3)),
    5: ((0, 0), (0, 3), (2, 4), (3, 0), (4, 2)),
    6: ((0, 0), (0, 2), (3, 0), (3, 5), (5, 2), (5, 5)),
    7: ((0, 0), (0, 5), (1, 3), (4, 2), (5, 0), (5, 6), (6, 5)),
    8: ((0, 0), (0, 7), (1, 3), (3, 1), (4, 6), (6, 4), (7, 0), (7, 7)),
    9: ((0, 0), (0, 3), (2, 1), (2, 6), (5, 8), (7, 0), (7, 6), (8, 1), (8, 8)),
    10: ((0, 0), (0, 3), (2, 7), (4, 2), (4, 5), (6, 0), (6, 9), (7, 5), (9, 3), (9, 9)),
    11: ((0, 0), (0, 7), (1, 4), (1, 10), (4, 1), (4, 9), (7, 0), (9, 4), (9, 10),
         (10, 1), (10, 9)),
    12: ((0, 0), (0, 3), (2, 8), (2, 11), (4, 1), (5, 4), (5, 10), (9, 6), (10, 0),
         (10, 11), (11, 1), (11, 10)),
    13: ((0, 0), (0, 4), (2, 7), (2, 11), (4, 1), (6, 5), (6, 8), (8, 0), (8, 12),
         (10, 1), (10, 7), (12, 5), (12, 11)),
}


def test_twice_area_is_hand_checkable() -> None:
    assert twice_area((0, 0), (2, 0), (0, 2)) == 4
    assert twice_area((0, 0), (0, 2), (2, 0)) == 4        # unsigned
    assert twice_area((0, 0), (1, 1), (2, 2)) == 0        # collinear
    assert twice_area((0, 0), (3, 0), (0, 3)) == 9


@pytest.mark.parametrize("n", sorted(OEIS_WITNESSES))
def test_scoring_reproduces_every_published_witness(n: int) -> None:
    """Eleven objects whose scores were published in 2015; the scorer must agree with all."""
    witness = OEIS_WITNESSES[n]
    assert len(witness) == n
    assert min_twice_area(witness, n) == OEIS_PUBLISHED[n]


def test_the_two_scorers_agree_on_the_settled_witnesses() -> None:
    for n, term in ESTABLISHED_TERMS.items():
        assert min_twice_area(term.witness, n) == term.value
        assert certify_min_double_area(list(term.witness), n) == term.value


def test_scoring_refuses_illegal_selections_and_exposes_a_lying_witness() -> None:
    term = ESTABLISHED_TERMS[14]
    good = list(term.witness)
    assert min_twice_area(good, 14) == 6
    with pytest.raises(ValueError):
        min_twice_area(good[:-1] + [(14, 0)], 14)             # off the grid
    with pytest.raises(ValueError):
        min_twice_area(good[:-1] + [good[0]], 14)             # repeated
    with pytest.raises(TypeError):
        min_twice_area([(0.0, 0), *good[1:]], 14)             # not an integer
    with pytest.raises(ValueError):
        min_twice_area(good[:2], 14)                          # no triangle
    # Move one point and the claimed 6 has to collapse, or the scorer is not scoring.
    assert min_twice_area([(0, 1), *good[1:]], 14) < 6
    assert min_twice_area([(0, 0), (1, 1), (2, 2), *good[3:]], 14) == 0


@pytest.mark.parametrize("n", [3, 4, 5])
def test_brute_force_over_every_subset_agrees_with_oeis(n: int) -> None:
    """No pruning, no symmetry, no cleverness -- every C(n^2, n) subset, counted."""
    value, seen = brute_force_maximum(n)
    assert seen == comb(n * n, n)
    assert value == OEIS_PUBLISHED[n]


@pytest.mark.parametrize("n", [5, 6])
def test_shards_partition_the_whole_space(n: int) -> None:
    """Completeness is arithmetic here: every subset lands in exactly one shard."""
    shards = shard_list(n)
    assert len(shards) == comb(n * n, 2) == len(set(shards))
    index = set(shards)
    landed = {shard: 0 for shard in shards}
    for subset in combinations(range(n * n), n):
        key = (subset[0], subset[1])
        assert key in index
        landed[key] += 1
    assert sum(landed.values()) == comb(n * n, n)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_independent_sweep_reproduces_the_published_term(n: int) -> None:
    value, witness = maximum(n)
    assert value == OEIS_PUBLISHED[n]
    assert min_twice_area(witness, n) == value
    # The half that carries a non-existence claim: nothing one step up, whole space swept.
    empty, _, done = decide(n, value + 1)
    assert empty is None
    assert done == len(shard_list(n))


@pytest.mark.parametrize("n", [5, 6, 7])
def test_the_two_procedures_agree_including_where_one_uses_symmetry(n: int) -> None:
    """One reduces by the square's eight symmetries, the other sweeps everything."""
    for threshold in (OEIS_PUBLISHED[n] - 1, OEIS_PUBLISHED[n], OEIS_PUBLISHED[n] + 1):
        theirs = exists_configuration(n, threshold)
        mine, _, done = decide(n, threshold)
        assert theirs.feasible == (mine is not None)
        assert done == len(shard_list(n)) or mine is not None


@pytest.mark.parametrize("n", [5, 6, 7, 14, 15])
def test_the_symmetry_reduced_branch_list_is_a_subset_of_the_full_shard_list(n: int) -> None:
    """Whatever the eightfold cut keeps, the full sweep also ran -- so it can be compared."""
    assert set(enumerate_branches(n)) <= set(shard_list(n))


def test_the_reproduction_record_matches_what_it_claims_to_reproduce() -> None:
    for n, record in REPRODUCED.items():
        term = ESTABLISHED_TERMS[n]
        assert record.threshold == term.refutation_threshold
        assert record.feasible is False
        assert record.shards == len(shard_list(n))
        # Restricted to their branches, this module's node count came out identical.
        assert record.nodes_on_reduced_branches == term.refutation_nodes
        # And the shards their symmetry cut never visited held nothing.
        assert record.solutions_in_skipped_shards == 0
        assert (
            record.shards_skipped_by_symmetry
            == len(shard_list(n)) - len(enumerate_branches(n))
        )
        assert record.nodes > record.nodes_on_reduced_branches
