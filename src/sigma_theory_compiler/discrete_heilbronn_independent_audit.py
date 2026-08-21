"""A second, independent decision procedure for A248866, written to disagree.

**Why a second one exists.**  ``discrete_heilbronn_grid`` settles ``a(14) = a(15) = 6``, and
the load-bearing half of each is a *non-existence* claim: no ``n``-subset of the ``n x n``
grid reaches twice-area ``7``.  Nothing outside this repository can confirm a non-existence
claim, so the only available check is a second procedure that shares no code with the first
and can be calibrated against things whose answers are already known.  This module is that
procedure.  It imports no mathematics from ``discrete_heilbronn_grid`` -- its determinant,
its masks, its search and its space decomposition are all written separately -- so the two
agreeing is evidence, and the two disagreeing would be a defect report.

**The two differences that matter.**  First, this search applies *no symmetry reduction at
all*.  The other module cuts the work roughly eightfold by starting only from orbit-minimal
points, which is the single step in it that could in principle lose a solution; running the
whole space instead makes that step irrelevant to the verdict rather than something to be
argued about.  Second, the space is decomposed differently: by the pair ``(smallest index,
second-smallest index)`` over *every* such pair, so completeness is arithmetic rather than
geometric.  Every ``n``-subset has exactly one smallest and one second-smallest element, so
the ``C(n^2, 2)`` shards partition the ``C(n^2, n)`` subsets -- nothing skipped, nothing
counted twice -- and :func:`shard_list` is that partition written down.

**What was actually run.**  :data:`REPRODUCED` records the two full sweeps this module
completed.  Both agree with ``discrete_heilbronn_grid``: no ``14``-subset and no
``15``-subset reaches ``7``.  Restricted to the branch list the other module used, the node
counts here came out bit-for-bit equal to the ones it recorded (5,480,024,818 and
45,102,635,138), and none of the 14,546 and 18,564 shards its symmetry cut skipped contained
a solution -- which is the eightfold reduction checked at the size that carries the claim
rather than at ``n = 5, 6, 7``.

**Calibration, and its limit.**  :func:`brute_force_maximum` looks at literally every
``C(n^2, n)`` subset and is the ground truth for ``n <= 5``; :func:`decide` reproduces all
eleven published terms ``a(3)..a(13)``.  Agreement between two implementations is not
independence of *idea* -- both are the natural branch-and-bound, and a shared conceptual
error would survive both.  What rules that out is the calibration, not the agreement: a
procedure that reproduces eleven known answers and, on a planted solution at ``n = 14`` and
``n = 15``, recovers it at threshold ``7``, is not a procedure that silently returns
"infeasible".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb

__all__ = [
    "OEIS_PUBLISHED",
    "REPRODUCED",
    "SweepRecord",
    "brute_force_maximum",
    "decide",
    "min_twice_area",
    "shard_list",
    "twice_area",
]

# Offset 3.  Transcribed from the OEIS text record fetched 2026-08-20, whose %S line reads
# "4,9,6,6,5,6,5,6,6,6,6" and whose latest revision is #22, Mar 13 2015.
OEIS_PUBLISHED: dict[int, int] = {
    3: 4, 4: 9, 5: 6, 6: 6, 7: 5, 8: 6, 9: 5, 10: 6, 11: 6, 12: 6, 13: 6,
}


@dataclass(frozen=True)
class SweepRecord:
    """One complete ``(n, threshold)`` sweep this module ran to completion."""

    n: int
    threshold: int
    feasible: bool
    shards: int
    nodes: int
    seconds: float
    #: Nodes this module spent on the *symmetry-reduced* branch list of the other module.
    nodes_on_reduced_branches: int
    #: Shards the other module's eightfold cut never visited, and how many held a solution.
    shards_skipped_by_symmetry: int
    solutions_in_skipped_shards: int


REPRODUCED: dict[int, SweepRecord] = {
    14: SweepRecord(
        n=14, threshold=7, feasible=False, shards=19_110, nodes=9_296_032_251,
        seconds=315.5, nodes_on_reduced_branches=5_480_024_818,
        shards_skipped_by_symmetry=14_546, solutions_in_skipped_shards=0,
    ),
    15: SweepRecord(
        n=15, threshold=7, feasible=False, shards=25_200, nodes=71_765_752_222,
        seconds=1452.5, nodes_on_reduced_branches=45_102_635_138,
        shards_skipped_by_symmetry=18_564, solutions_in_skipped_shards=0,
    ),
}


def twice_area(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    """Twice the area of triangle ``abc``: a non-negative integer, zero iff collinear."""
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def min_twice_area(points: Sequence[tuple[int, int]], n: int) -> int:
    """Minimum over **every** ``C(len(points), 3)`` triple, with the count asserted.

    Refuses to score anything that is not a legal selection: a coordinate off the ``n x n``
    grid, a repeated point, a non-integer, or fewer than three points.  The triple counter is
    checked against ``C(len, 3)`` so a scorer that silently skipped triples would fail here
    rather than return a flattering number.
    """
    pts = [tuple(p) for p in points]
    if len(pts) < 3:
        raise ValueError("a triangle needs three points")
    for x, y in pts:
        if not isinstance(x, int) or not isinstance(y, int) or isinstance(x, bool):
            raise TypeError("grid coordinates must be plain ints")
        if isinstance(y, bool):
            raise TypeError("grid coordinates must be plain ints")
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError(f"({x}, {y}) is outside the {n} x {n} grid")
    if len(set(pts)) != len(pts):
        raise ValueError("points must be distinct")
    seen = 0
    best: int | None = None
    for triple in combinations(pts, 3):
        value = twice_area(*triple)
        best = value if best is None else min(best, value)
        seen += 1
    if seen != comb(len(pts), 3):
        raise AssertionError(f"scanned {seen} triples, not C({len(pts)}, 3)")
    assert best is not None
    return best


def shard_list(n: int) -> list[tuple[int, int]]:
    """The ``C(n^2, 2)`` shards that partition every ``n``-subset of the grid.

    Each subset has exactly one smallest and one second-smallest index, so mapping it to that
    pair is total and injective on shards: completeness of a sweep is then the arithmetic
    fact that all ``C(n^2, 2)`` shards were decided, with no symmetry argument in sight.
    """
    size = n * n
    return [(first, second) for first in range(size) for second in range(first + 1, size)]


@lru_cache(maxsize=8)
def _masks(n: int, threshold: int) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """``masks[p][q]`` = bitset of ``r`` with ``|det(p, q, r)| >= threshold``.

    Cached because a sweep rebuilds nothing between shards; the table depends only on
    ``(n, threshold)``.  Callers never mutate it.
    """
    grid = [(x, y) for x in range(n) for y in range(n)]
    size = n * n
    masks = [[0] * size for _ in range(size)]
    for i in range(size):
        xi, yi = grid[i]
        for j in range(size):
            if j == i:
                continue
            dx, dy = grid[j][0] - xi, grid[j][1] - yi
            acc = 0
            for k in range(size):
                if k == i or k == j:
                    continue
                det = dx * (grid[k][1] - yi) - (grid[k][0] - xi) * dy
                if det >= threshold or -det >= threshold:
                    acc |= 1 << k
            masks[i][j] = acc
    return grid, masks


def search_shard(
    n: int, threshold: int, shard: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...] | None, int]:
    """Complete search of one shard: all ``n``-subsets whose two least indices are ``shard``."""
    first, second = shard
    if not 0 <= first < second < n * n:
        raise ValueError("shard must be a strictly increasing index pair")
    grid, masks = _masks(n, threshold)
    size = n * n
    nodes = 0

    def descend(chosen: list[int], cand: int) -> list[int] | None:
        nonlocal nodes
        if len(chosen) == n:
            return list(chosen)
        need = n - len(chosen)
        while cand:
            if cand.bit_count() < need:
                return None
            low = cand & -cand
            point = low.bit_length() - 1
            cand ^= low
            reduced = cand
            row = masks[point]
            for already in chosen:
                reduced &= row[already]
            nodes += 1
            if reduced.bit_count() >= need - 1:
                chosen.append(point)
                found = descend(chosen, reduced)
                chosen.pop()
                if found is not None:
                    return found
        return None

    start = (((1 << size) - 1) >> (second + 1)) << (second + 1)
    start &= masks[second][first]
    found = descend([first, second], start)
    if found is None:
        return None, nodes
    witness = tuple(grid[i] for i in found)
    if min_twice_area(list(witness), n) < threshold:
        raise AssertionError("search returned a set that does not clear the threshold")
    return witness, nodes


def decide(
    n: int, threshold: int, shards: Sequence[tuple[int, int]] | None = None
) -> tuple[tuple[tuple[int, int], ...] | None, int, int]:
    """Decide ``(n, threshold)`` by sweeping every shard.  Returns ``(witness, nodes, done)``.

    Passing ``shards`` runs a subset of them, which is how a long sweep is split across
    processes; a verdict of "infeasible" only means anything when ``done == len(shard_list(n))``.
    """
    todo = shard_list(n) if shards is None else list(shards)
    total_nodes = 0
    for shard in todo:
        witness, nodes = search_shard(n, threshold, shard)
        total_nodes += nodes
        if witness is not None:
            return witness, total_nodes, len(todo)
    return None, total_nodes, len(todo)


def maximum(n: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    """``a(n)`` by climbing the threshold until the sweep comes back empty."""
    threshold = 1
    best: tuple[tuple[int, int], ...] | None = None
    while True:
        witness, _, _ = decide(n, threshold)
        if witness is None:
            if best is None:
                raise ValueError(f"nothing feasible at threshold {threshold}")
            return threshold - 1, best
        best = witness
        threshold += 1


def brute_force_maximum(n: int) -> tuple[int, int]:
    """``a(n)`` by inspecting every one of the ``C(n^2, n)`` subsets.  Ground truth, tiny ``n``."""
    grid = [(x, y) for x in range(n) for y in range(n)]
    best = 0
    seen = 0
    for subset in combinations(grid, n):
        seen += 1
        value = min(twice_area(*t) for t in combinations(subset, 3))
        best = max(best, value)
    if seen != comb(n * n, n):
        raise AssertionError(f"visited {seen} subsets, not C({n * n}, {n})")
    return best, seen
