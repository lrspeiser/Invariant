"""Exhaustive solution of the discrete Heilbronn problem on the ``n x n`` integer lattice.

**The quantity.**  Choose ``n`` of the ``n^2`` lattice points ``{0..n-1}^2`` and look at the
smallest triangle any three of them span.  ``a(n)`` is the largest that smallest triangle can
be made, measured as *twice* its area -- which for lattice points is the integer

    ``det = (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i)``.

So ``a(n) = max over n-subsets S of min over triples of |det|``, an integer optimum over a
finite set.  This is OEIS `A248866 <https://oeis.org/A248866>`_, contributed by Gordon
Hamilton in March 2015 and extended by Hiroaki Yamanouchi to ``a(13)`` a few days later.  The
entry still carries the ``more`` keyword: past ``a(13)`` the sequence is unpublished.

**Why this is settleable rather than merely searchable.**  Everything is integral and the
domain is finite, so ``a(n) = T`` is two exact statements: a witness achieving ``T``, and the
*non-existence* of any ``n``-subset achieving ``T + 1``.  :func:`exists_configuration`
decides existence by complete enumeration, so the second half is a proof and not a failure to
find.  ``feasible(n, T)`` is monotone decreasing in ``T`` -- a set whose triples all clear
``T + 1`` clears ``T`` too -- so refuting ``T + 1`` refutes every larger threshold at once.

**How the enumeration stays finite in practice.**  Subsets are built in increasing index
order, with a bitset ``cand`` of the points still admissible.  For every pair ``(p, q)``,
``_pair_masks`` precomputes the set of ``r`` with ``|det(p, q, r)| >= T``; extending a partial
set by ``p`` intersects ``cand`` with ``mask[p][q]`` for each ``q`` already chosen, so a
partial set is only ever extended by points consistent with *every* pair in it, and a branch
dies as soon as ``|chosen| + popcount(cand) < n``.

**The one symmetry argument.**  The square's eight symmetries map solutions to solutions, so
if any exists, one exists that is lexicographically smallest among its eight images; the
smallest-index point of that representative is necessarily smallest in its own orbit.  The
enumeration therefore starts only from orbit-minimal points, cutting the work about eightfold
without losing a single solution -- and :func:`exists_configuration` accepts
``use_symmetry=False`` so the two searches can be compared directly.

**Honest limits.**  A verdict here is exact and complete for the ``(n, T)`` it was run on, and
says nothing about any other ``n``.  Reproducing ``a(3)..a(13)`` shows the enumeration agrees
with the published terms; it is not evidence about terms nobody has computed, and absence of
a published ``a(14)`` is not a claim that none exists somewhere.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

__all__ = [
    "ESTABLISHED_TERMS",
    "PUBLISHED_TERMS",
    "EstablishedTerm",
    "GridVerdict",
    "certify_min_double_area",
    "exists_configuration",
    "grid_points",
    "maximal_min_double_area",
]

TERMS_SOURCE = (
    "OEIS A248866, 'Discrete Heilbronn Triangle Problem: a(n) is twice the maximal area of "
    "the smallest triangle defined by three vertices that are a subset of n points on an "
    "n X n square lattice', Gordon Hamilton (Mar 2015), a(5)/a(7)/a(9) corrected and "
    "a(10)-a(13) added by Hiroaki Yamanouchi (Mar 2015); keyword 'more'; "
    "https://oeis.org/A248866 (read 2026-08-20)"
)

# Offset 3.  These are the published terms and the whole of them: the entry stops at a(13).
PUBLISHED_TERMS: dict[int, int] = {
    3: 4, 4: 9, 5: 6, 6: 6, 7: 5, 8: 6, 9: 5, 10: 6, 11: 6, 12: 6, 13: 6,
}
LAST_PUBLISHED_N = max(PUBLISHED_TERMS)


@dataclass(frozen=True)
class EstablishedTerm:
    """A term this repository settled by its own complete enumeration.

    The witness is carried so the *lower* half re-verifies in milliseconds on every CI run.
    The *upper* half -- that nothing reaches ``value + 1`` -- is a completed sweep whose cost
    is recorded here rather than repeated: it is reproduced with

        ``python -m sigma_theory_compiler.discrete_heilbronn_grid --n N --threshold V+1``

    which must report ``feasible: false``.  ``refutation_nodes`` is what that sweep visited,
    so a rerun that finishes suspiciously fast is visibly not the same computation.
    """

    n: int
    value: int
    witness: tuple[tuple[int, int], ...]
    refutation_threshold: int
    refutation_nodes: int
    refutation_seconds: float
    beyond_published: bool


ESTABLISHED_TERMS: dict[int, EstablishedTerm] = {
    14: EstablishedTerm(
        n=14,
        value=6,
        witness=(
            (0, 0), (0, 4), (2, 8), (2, 12), (4, 1), (4, 4), (6, 8),
            (6, 11), (8, 0), (10, 6), (10, 12), (11, 1), (13, 6), (13, 11),
        ),
        refutation_threshold=7,
        refutation_nodes=5_480_025_159,
        refutation_seconds=242.2,
        beyond_published=True,
    ),
}


def grid_points(n: int) -> list[tuple[int, int]]:
    """The ``n^2`` lattice points, indexed so that index ``x * n + y`` is lexicographic."""
    if n < 3:
        raise ValueError("need at least three points")
    return [(x, y) for x in range(n) for y in range(n)]


def double_area(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    """Twice the triangle area, as a non-negative integer.  Zero exactly when collinear."""
    value = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    return -value if value < 0 else value


def certify_min_double_area(points: Sequence[tuple[int, int]], n: int) -> int:
    """Exhaustive minimum of ``|det|`` over all ``C(len(points), 3)`` triples.

    Rejects anything that is not a legal selection -- points off the grid, or repeated -- so
    a witness cannot be scored unless it really is a subset of ``{0..n-1}^2``.
    """
    size = len(points)
    if size < 3:
        raise ValueError("a triangle needs at least three points")
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("grid coordinates must be Python ints")
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError("point outside the n x n grid")
    if len({tuple(p) for p in points}) != size:
        raise ValueError("points must be distinct")
    return min(double_area(*triple) for triple in combinations(points, 3))


def _pair_masks(n: int, threshold: int) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """``mask[p][q]`` = bitset of points ``r`` with ``|det(p, q, r)| >= threshold``."""
    points = grid_points(n)
    size = len(points)
    masks = [[0] * size for _ in range(size)]
    for i in range(size):
        xi, yi = points[i]
        for j in range(i + 1, size):
            dxj = points[j][0] - xi
            dyj = points[j][1] - yi
            value = 0
            for k in range(size):
                if k == i or k == j:
                    continue
                det = dxj * (points[k][1] - yi) - (points[k][0] - xi) * dyj
                if det >= threshold or -det >= threshold:
                    value |= 1 << k
            masks[i][j] = value
            masks[j][i] = value
    return points, masks


def _orbit_minimal(point: tuple[int, int], n: int) -> bool:
    """Is this point the least-indexed member of its orbit under the square's symmetries?"""
    x, y = point
    m = n - 1
    orbit = (
        (x, y), (y, x), (m - x, y), (x, m - y),
        (m - x, m - y), (y, m - x), (m - y, x), (m - y, m - x),
    )
    return min(a * n + b for a, b in orbit) == x * n + y


@dataclass(frozen=True)
class GridVerdict:
    """The outcome of one complete ``(n, threshold)`` enumeration."""

    n: int
    threshold: int
    feasible: bool
    nodes: int
    witness: tuple[tuple[int, int], ...] | None
    used_symmetry: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "problem": "discrete_heilbronn_grid_min_double_area",
            "terms_source": TERMS_SOURCE,
            "n": self.n,
            "threshold": self.threshold,
            "feasible": self.feasible,
            "search_nodes": self.nodes,
            "witness": None if self.witness is None else [list(p) for p in self.witness],
            "symmetry_reduction_used": self.used_symmetry,
            "enumeration": "complete over all n-subsets of the n x n grid, exact integers",
            "absence_establishes_novelty": False,
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_payload())


def _branch_roots(n: int, use_symmetry: bool) -> Iterator[int]:
    points = grid_points(n)
    for index, point in enumerate(points):
        if not use_symmetry or _orbit_minimal(point, n):
            yield index


def exists_configuration(
    n: int, threshold: int, *, use_symmetry: bool = True, roots: Sequence[int] | None = None
) -> GridVerdict:
    """Decide, by complete enumeration, whether ``n`` grid points can all clear ``threshold``.

    ``roots`` restricts the first (lowest-index) chosen point, which is how the work is split
    across processes; passing it does not change what a full sweep of roots decides.
    """
    if threshold < 1:
        raise ValueError("threshold must be positive")
    points, masks = _pair_masks(n, threshold)
    size = n * n
    nodes = 0

    def descend(chosen: list[int], cand: int, depth: int) -> list[int] | None:
        nonlocal nodes
        if depth == n:
            return list(chosen)
        while cand:
            if depth + cand.bit_count() < n:
                return None
            low = cand & -cand
            p = low.bit_length() - 1
            cand ^= low
            reduced = cand  # only indices above p remain selectable
            row = masks[p]
            for q in chosen:
                reduced &= row[q]
            nodes += 1
            if depth + 1 + reduced.bit_count() >= n:
                chosen.append(p)
                found = descend(chosen, reduced, depth + 1)
                chosen.pop()
                if found is not None:
                    return found
        return None

    candidates = list(_branch_roots(n, use_symmetry)) if roots is None else list(roots)
    full = (1 << size) - 1
    for first in candidates:
        found = descend([first], (full >> (first + 1)) << (first + 1), 1)
        if found is not None:
            witness = tuple(points[i] for i in found)
            assert certify_min_double_area(list(witness), n) >= threshold
            return GridVerdict(n, threshold, True, nodes, witness, use_symmetry)
    return GridVerdict(n, threshold, False, nodes, None, use_symmetry)


def maximal_min_double_area(n: int, *, start: int = 1) -> tuple[int, GridVerdict, GridVerdict]:
    """Return ``(a(n), witness verdict, refutation verdict)`` by climbing the threshold.

    The refutation verdict is the load-bearing half: it is a completed enumeration finding no
    ``n``-subset at ``a(n) + 1``, and monotonicity in the threshold carries that to every
    larger value.
    """
    best = None
    threshold = start
    while True:
        verdict = exists_configuration(n, threshold)
        if not verdict.feasible:
            if best is None:
                raise ValueError(f"no configuration even at threshold {threshold}")
            return threshold - 1, best, verdict
        best = verdict
        threshold += 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--threshold", type=int, default=None,
                        help="decide one threshold; omit to climb to a(n)")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument(
        "--roots", type=str, default=None,
        help="comma-separated first-point indices, for reproducing one shard of a split sweep",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    roots = None
    if args.roots is not None:
        roots = [int(part) for part in args.roots.split(",") if part.strip()]

    if args.threshold is not None:
        verdict = exists_configuration(
            args.n, args.threshold, use_symmetry=not args.no_symmetry, roots=roots
        )
        payload = verdict.to_payload()
        payload["roots_restricted_to"] = roots
    else:
        value, witness, refutation = maximal_min_double_area(args.n)
        payload = {
            "n": args.n,
            "a_n": value,
            "published": PUBLISHED_TERMS.get(args.n),
            "agrees_with_published": PUBLISHED_TERMS.get(args.n) == value,
            "beyond_published_range": args.n > LAST_PUBLISHED_N,
            "witness": witness.to_payload(),
            "refutation": refutation.to_payload(),
        }
    payload["digest"] = canonical_sha256(payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
