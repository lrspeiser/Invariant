"""Ramsey lower bounds by exhaustive search over the cyclic (circulant) family.

A lower bound ``R(s, t) > n`` is settled by exhibiting one 2-colouring of ``K_n`` with no
red ``K_s`` and no blue ``K_t``.  The coloured graph *is* the proof, so the only thing that
can go wrong is the verification, and verification here is exhaustive clique enumeration in
integer arithmetic.  This module attacks the case ``s = 3``: red is a triangle-free graph,
blue is its complement, and "no blue ``K_t``" means the red graph has independence number at
most ``t - 1``.

**The declared family.**  A *cyclic* (circulant) colouring of ``K_n`` is determined by a set
``S`` of residues with ``S = -S`` and ``0 not in S``: vertices are ``Z_n`` and ``u ~ v`` iff
``u - v in S``.  So the search is over subsets of the ``floor(n / 2)`` difference classes
``{d, n - d}`` rather than over all ``2 ** C(n, 2)`` edge colourings.  Everything below is
exhaustive *over this family* and says nothing about non-cyclic colourings.

**Why the family is worth exhausting.**  Radziszowski's dynamic survey (DS1, revision #18,
24 April 2026), item 2.1(i), records that Harborth and Krause "presented all best lower
bounds up to 102 from cyclic graphs avoiding complete graphs.  In particular, no lower bound
in Table Ia can be improved with a cyclic graph on less than 102 vertices, except possibly
for R(3, k) for k >= 13."  That sentence names exactly one open door in the whole of Table Ia
for this construction family, and :data:`SEALED_RECORD` reproduces the numbers it refers to.

**Four exact facts do all the pruning.**  Each is an integer statement about the graph, so
none of them can silently discard a witness.

1.  ``S = N(0)``, and in a triangle-free graph a neighbourhood is independent, so
    ``alpha >= |S|``.  Hence ``alpha <= k - 1`` forces ``degree <= k - 1``.
2.  Greedy independence: ``alpha >= ceil(n / (degree + 1))``.  Hence
    ``degree >= ceil(n / (k - 1)) - 1``.
3.  ``{0, 1, ..., min(S) - 1}`` is independent, so ``alpha >= min(S)`` and the smallest
    difference in ``S`` is at most ``k - 1``.
4.  Triangle-freeness of ``C_n(S)`` is exactly sum-freeness: a triangle ``{0, a, b}`` needs
    ``a, b, b - a in S``, and since ``S = -S`` the condition is ``(S + S) and S`` disjoint.
    The enumeration maintains ``M = S + S`` incrementally and rejects on ``S and M``.

Facts 1-3 are *necessary conditions on the final graph*, and the enumeration only ever
discards a partial set when no completion of it can satisfy them, so the walk is complete.
:func:`enumerate_naive` is the same search with every one of them switched off -- a literal
loop over all ``2 ** floor(n / 2)`` subsets -- and the tests cross-check the two enumerators
agree set-for-set on every ``n`` small enough to brute force.

**Nothing is sampled.**  Independence is decided by :func:`has_independent_set`, a complete
depth-first search that uses vertex-transitivity to fix vertex ``0`` (every independent set of
a circulant can be rotated to contain ``0``) and prunes only on the exact counting bound
"fewer candidates remain than vertices still needed".  Greedy colouring is used only to
*reject*: exhibiting an independent set of size ``k`` proves ``alpha >= k``, so a rejection is
a proof, never an estimate.  A candidate that greedy fails to reject goes to the complete
search.

**Multiplier symmetry is applied only after rejection.**  For ``u`` a unit mod ``n``, the map
``x -> u x`` is a graph isomorphism ``C_n(S) -> C_n(uS)``, so ``alpha`` is constant on
multiplier orbits.  The enumeration still visits every set; the expensive exact test is run
only on the lexicographically least member of each orbit.  Skipping a non-canonical set can
therefore never skip a witness -- its canonical partner is enumerated too.

**Honest limits, stated as claims.**  Absence of a cyclic witness at order ``n`` is a
statement about circulant graphs and nothing else; the best known ``R(3, k)`` witnesses for
several ``k`` are provably not cyclic (``k = 6`` is the smallest such ``k``).  The record gate
:func:`compare_to_record` is an integer comparison against :data:`SEALED_RECORD`, and every
entry there is a published bound, so the honest expected outcome is that it does not fire.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from math import gcd

from .sigma_core import canonical_sha256

__all__ = [
    "SEALED_RECORD",
    "CyclicColouring",
    "SearchOutcome",
    "adjacency_independence_number",
    "adjacency_is_triangle_free",
    "build_adjacency",
    "compare_to_record",
    "difference_classes",
    "dihedral_adjacency",
    "dihedral_search",
    "enumerate_naive",
    "exhaustive_search",
    "has_independent_set",
    "independence_number",
    "is_triangle_free",
    "main",
    "scan_orders",
    "seal_certificate",
    "verify_certificate",
]


# --------------------------------------------------------------------------------------
# Sealed published record.  Source is named on every entry; nothing here is inferred.
# --------------------------------------------------------------------------------------

RECORD_SOURCE = (
    "S. P. Radziszowski, Small Ramsey Numbers, Electronic Journal of Combinatorics, "
    "Dynamic Survey DS1, revision #18, 24 April 2026, Table Ia."
)

CYCLIC_SOURCE = (
    "H. Harborth and S. Krause, Ramsey Numbers for Circulant Colorings, Congressus "
    "Numerantium 161 (2003) 139-150; terms a(10), a(11), a(12) of OEIS A000789 "
    "(maximal order of a triangle-free cyclic graph with no independent set of size n) "
    "are credited to that paper by J. Backelin, 2016."
)

# Published two-colour bounds for R(3, k):  k -> (lower, upper), DS1.18 Table Ia.
SEALED_RECORD: dict[int, tuple[int, int]] = {
    3: (6, 6),
    4: (9, 9),
    5: (14, 14),
    6: (18, 18),
    7: (23, 23),
    8: (28, 28),
    9: (36, 36),
    10: (40, 41),
    11: (47, 50),
    12: (53, 59),
    13: (61, 68),
    14: (67, 77),
    15: (74, 87),
}

# OEIS A000789: maximal order of a triangle-free cyclic graph with no independent set of
# size k.  Published terms only -- the sequence carries keyword "more" and stops at k = 12.
SEALED_CYCLIC_ORDERS: dict[int, int] = {
    2: 2,
    3: 5,
    4: 8,
    5: 13,
    6: 16,
    7: 21,
    8: 26,
    9: 35,
    10: 38,
    11: 45,
    12: 48,
}


# --------------------------------------------------------------------------------------
# Circulant graphs as integer bitmasks.
# --------------------------------------------------------------------------------------


def difference_classes(n: int) -> list[tuple[int, ...]]:
    """The ``floor(n / 2)`` classes ``{d, n - d}`` that a symmetric ``S`` is built from."""

    if n < 1:
        raise ValueError("order must be positive")
    out: list[tuple[int, ...]] = []
    for d in range(1, n // 2 + 1):
        out.append((d,) if d == n - d else (d, n - d))
    return out


def _rotate(mask: int, shift: int, n: int, full: int) -> int:
    if shift == 0:
        return mask
    return ((mask << shift) | (mask >> (n - shift))) & full


def build_adjacency(n: int, connection: int) -> list[int]:
    """Row ``v`` is the bitmask of neighbours of ``v`` in ``C_n(S)``."""

    full = (1 << n) - 1
    return [_rotate(connection, v, n, full) for v in range(n)]


def connection_from_residues(n: int, residues: Sequence[int]) -> int:
    mask = 0
    for r in residues:
        r %= n
        if r == 0:
            raise ValueError("0 is not a difference")
        mask |= 1 << r
        mask |= 1 << (n - r)
    return mask


def residues_of(n: int, connection: int) -> list[int]:
    return [i for i in range(n) if (connection >> i) & 1]


def is_symmetric(n: int, connection: int) -> bool:
    if connection & 1:
        return False
    for i in range(1, n):
        if ((connection >> i) & 1) != ((connection >> (n - i)) & 1):
            return False
    return True


def is_triangle_free(n: int, connection: int) -> bool:
    """Exhaustive: every edge is examined and every common neighbour of its ends found.

    A triangle contains three edges, so scanning all edges finds every triangle; the test
    is a complete search over vertex triples, not a sample of them.
    """

    adj = build_adjacency(n, connection)
    for a in range(n):
        rest = adj[a] >> (a + 1)
        b = a + 1
        while rest:
            if rest & 1 and adj[a] & adj[b]:
                return False
            rest >>= 1
            b += 1
    return True


def triangle_witness(n: int, connection: int) -> tuple[int, int, int] | None:
    adj = build_adjacency(n, connection)
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            if common:
                c = (common & -common).bit_length() - 1
                return (a, b, c)
    return None


# --------------------------------------------------------------------------------------
# Exact independence.
# --------------------------------------------------------------------------------------


def has_independent_set(n: int, adj: Sequence[int], target: int) -> bool:
    """Complete decision procedure for "does ``C_n(S)`` contain ``target`` pairwise
    non-adjacent vertices?".

    Vertex-transitivity of a circulant lets us fix vertex ``0``: rotating any independent
    set by ``-v`` for one of its members ``v`` gives an independent set containing ``0``.
    The recursion then branches on every remaining candidate and prunes only when the number
    of surviving candidates is smaller than the number of vertices still required, which
    cannot discard a solution.
    """

    if target <= 0:
        return True
    if target == 1:
        return n >= 1
    full = (1 << n) - 1
    start = full & ~adj[0] & ~1

    def rec(cand: int, need: int) -> bool:
        if need == 0:
            return True
        if cand.bit_count() < need:
            return False
        rest = cand
        while rest:
            if rest.bit_count() < need:
                return False
            low = rest & -rest
            v = low.bit_length() - 1
            if rec(rest & ~adj[v] & ~low, need - 1):
                return True
            rest &= ~low
        return False

    return rec(start, target - 1)


def independent_set_witness(n: int, adj: Sequence[int], target: int) -> list[int] | None:
    if target <= 0:
        return []
    full = (1 << n) - 1
    chosen: list[int] = [0]

    def rec(cand: int, need: int) -> bool:
        if need == 0:
            return True
        if cand.bit_count() < need:
            return False
        rest = cand
        while rest:
            if rest.bit_count() < need:
                return False
            low = rest & -rest
            v = low.bit_length() - 1
            chosen.append(v)
            if rec(rest & ~adj[v] & ~low, need - 1):
                return True
            chosen.pop()
            rest &= ~low
        return False

    if target == 1:
        return [0]
    if rec(full & ~adj[0] & ~1, target - 1):
        return sorted(chosen)
    return None


def independence_number(n: int, connection: int) -> int:
    adj = build_adjacency(n, connection)
    size = 0
    while has_independent_set(n, adj, size + 1):
        size += 1
    return size


def _greedy_lowest(n: int, adj: Sequence[int], cap: int) -> int:
    """Greedy independent set taking the lowest-indexed candidate; returns min(size, cap).

    The returned value is the size of an actual independent set, so it is a valid lower
    bound on ``alpha`` and using it to reject a candidate is a proof, not a heuristic.
    """

    cand = (1 << n) - 1
    size = 0
    while cand:
        low = cand & -cand
        v = low.bit_length() - 1
        size += 1
        if size >= cap:
            return size
        cand &= ~adj[v] & ~low
    return size


def _greedy_min_degree(n: int, adj: Sequence[int], cap: int) -> int:
    cand = (1 << n) - 1
    size = 0
    while cand:
        best_v, best_d = -1, n + 1
        rest = cand
        while rest:
            low = rest & -rest
            v = low.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg < best_d:
                best_d, best_v = deg, v
                if deg == 0:
                    break
            rest &= ~low
        size += 1
        if size >= cap:
            return size
        cand &= ~adj[best_v] & ~(1 << best_v)
    return size


# --------------------------------------------------------------------------------------
# Colouring record.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CyclicColouring:
    """A verified circulant witness for ``R(3, k) > n``."""

    order: int
    connection: tuple[int, ...]
    independence: int
    degree: int

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "connection_set": list(self.connection),
            "difference_classes": sorted({min(d, self.order - d) for d in self.connection}),
            "degree": self.degree,
            "independence_number": self.independence,
            "clique_number_bound": 2,
            "certifies": f"R(3,{self.independence + 1}) >= {self.order + 1}",
        }


@dataclass
class SearchOutcome:
    order: int
    clique_target: int
    witnesses: list[CyclicColouring] = field(default_factory=list)
    nodes: int = 0
    candidates_tested: int = 0
    rejected_by_greedy: int = 0
    decided_exactly: int = 0
    vacuous: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "clique_target": self.clique_target,
            "witness_count": len(self.witnesses),
            "witnesses": [w.to_dict() for w in self.witnesses],
            "search_nodes": self.nodes,
            "candidates_tested": self.candidates_tested,
            "rejected_by_greedy": self.rejected_by_greedy,
            "decided_by_complete_search": self.decided_exactly,
            "vacuous": self.vacuous,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------------------
# The two enumerators.
# --------------------------------------------------------------------------------------


def enumerate_naive(n: int, k: int) -> list[int]:
    """Reference enumerator: literally every one of the ``2 ** floor(n / 2)`` symmetric
    connection sets, with no pruning of any kind.  Exponential and only usable for small
    ``n``; the tests use it to certify that :func:`exhaustive_search` discards nothing."""

    classes = difference_classes(n)
    found: list[int] = []
    for bits in range(1 << len(classes)):
        connection = 0
        for i, cls in enumerate(classes):
            if (bits >> i) & 1:
                for d in cls:
                    connection |= 1 << d
        if connection == 0:
            continue
        if not is_triangle_free(n, connection):
            continue
        if independence_number(n, connection) <= k - 1:
            found.append(connection)
    return sorted(found)


def exhaustive_search(
    n: int, k: int, stop_at_first: bool = True, collect_all: bool = False
) -> SearchOutcome:
    """Complete search over circulant triangle-free graphs on ``Z_n`` with ``alpha <= k-1``.

    ``stop_at_first`` returns as soon as one witness is verified, which is all a lower bound
    needs.  ``collect_all`` walks the whole space and returns every canonical witness.
    """

    if n < 1:
        raise ValueError("order must be positive")
    if k < 2:
        raise ValueError("clique target must be at least 2")
    if collect_all:
        stop_at_first = False

    out = SearchOutcome(order=n, clique_target=k)
    full = (1 << n) - 1
    max_degree = k - 1                        # fact 1: alpha >= degree
    min_degree = -(-n // (k - 1)) - 1         # fact 2: alpha >= ceil(n / (degree + 1))
    if min_degree > max_degree:
        out.vacuous = True
        out.reason = (
            f"degree window empty: alpha <= {k - 1} forces degree <= {k - 1} and "
            f"degree >= {min_degree}"
        )
        return out

    classes = difference_classes(n)
    units = [u for u in range(2, n) if gcd(u, n) == 1]

    def is_canonical(connection: int) -> bool:
        for u in units:
            image = 0
            rest = connection
            while rest:
                low = rest & -rest
                x = low.bit_length() - 1
                image |= 1 << ((u * x) % n)
                rest &= ~low
                if image > connection:
                    break
            if image < connection:
                return False
        return True

    def consider(connection: int) -> None:
        out.candidates_tested += 1
        adj = build_adjacency(n, connection)
        if _greedy_lowest(n, adj, k) >= k:
            out.rejected_by_greedy += 1
            return
        if _greedy_min_degree(n, adj, k) >= k:
            out.rejected_by_greedy += 1
            return
        if not is_canonical(connection):
            return
        out.decided_exactly += 1
        if has_independent_set(n, adj, k):
            return
        alpha = independence_number(n, connection)
        out.witnesses.append(
            CyclicColouring(
                order=n,
                connection=tuple(residues_of(n, connection)),
                independence=alpha,
                degree=connection.bit_count(),
            )
        )

    def walk(index: int, connection: int, sums: int, degree: int) -> None:
        out.nodes += 1
        if degree >= min_degree and degree >= 1:
            consider(connection)
            if out.witnesses and stop_at_first:
                return
        remaining = len(classes) - index
        if degree + 2 * remaining < min_degree:
            return
        # fact 3: the smallest difference in S is at most k - 1, so the first class taken
        # is bounded; later classes are unconstrained beyond the degree window.
        limit = len(classes) if connection else min(len(classes), max_degree)
        for j in range(index, limit):
            cls = classes[j]
            x = cls[0]
            y = n - x
            step = len(cls)
            if degree + step > max_degree:
                continue
            if (sums >> x) & 1:
                continue
            doubled = (2 * x) % n
            if (connection >> doubled) & 1 or doubled == x or doubled == y:
                continue
            grown = connection | (1 << x) | (1 << y % n)
            grown_sums = (
                sums
                | _rotate(connection, x, n, full)
                | _rotate(connection, y % n, n, full)
                | (1 << doubled)
                | (1 << ((2 * y) % n))
                | 1
            )
            if grown & grown_sums:           # fact 4: sum-free <=> triangle-free
                continue
            walk(j + 1, grown, grown_sums, degree + step)
            if out.witnesses and stop_at_first:
                return

    walk(0, 0, 0, 0)
    out.reason = (
        f"complete walk over symmetric connection sets with degree in "
        f"[{min_degree}, {max_degree}]"
    )
    return out


def scan_orders(k: int, lo: int, hi: int) -> Iterator[SearchOutcome]:
    for n in range(lo, hi + 1):
        yield exhaustive_search(n, k)


def max_cyclic_order(k: int, lo: int, hi: int) -> tuple[int, CyclicColouring | None]:
    """Largest ``n`` in ``[lo, hi]`` carrying a circulant witness for ``R(3, k) > n``.

    Existence is *not* monotone in ``n``, so the whole interval is walked; the caller is
    responsible for choosing an interval that provably contains every feasible order.
    """

    best, witness = 0, None
    for outcome in scan_orders(k, lo, hi):
        if outcome.witnesses:
            best, witness = outcome.order, outcome.witnesses[0]
    return best, witness


# --------------------------------------------------------------------------------------
# Independent re-verification and sealing.
# --------------------------------------------------------------------------------------


def verify_colouring(n: int, residues: Sequence[int], k: int) -> dict[str, object]:
    """Re-derive every claim about a colouring from the residue list alone."""

    connection = 0
    for r in residues:
        if not isinstance(r, int) or not 0 < r < n:
            return {"valid": False, "reason": f"residue {r!r} outside 1..{n - 1}"}
        connection |= 1 << r
    checks: dict[str, object] = {"order": n, "residues": sorted(residues)}
    checks["symmetric"] = is_symmetric(n, connection)
    if not checks["symmetric"]:
        return {"valid": False, "reason": "connection set is not closed under negation", **checks}
    tri = triangle_witness(n, connection)
    checks["triangle_free"] = tri is None
    checks["triangle_witness"] = list(tri) if tri else None
    alpha = independence_number(n, connection)
    checks["independence_number"] = alpha
    checks["degree"] = connection.bit_count()
    checks["edge_count"] = n * connection.bit_count() // 2
    witness = independent_set_witness(n, build_adjacency(n, connection), alpha)
    checks["largest_independent_set"] = witness
    checks["valid"] = bool(tri is None and alpha <= k - 1)
    checks["certifies"] = f"R(3,{k}) >= {n + 1}" if checks["valid"] else None
    return checks


def compare_to_record(k: int, order: int) -> dict[str, object]:
    """Integer comparison against the sealed published bound.  The only gate that could
    ever say "new", and it is expected not to fire."""

    published_lower, published_upper = SEALED_RECORD[k]
    attained = order + 1
    return {
        "clique_target": k,
        "cyclic_order": order,
        "cyclic_lower_bound": attained,
        "published_lower_bound": published_lower,
        "published_upper_bound": published_upper,
        "gap_to_published": published_lower - attained,
        "beats_published_lower_bound": attained > published_lower,
        "record_source": RECORD_SOURCE,
    }


# --------------------------------------------------------------------------------------
# The dihedral family: the only non-cyclic Cayley graphs at the frontier orders.
# --------------------------------------------------------------------------------------
#
# Improving the Table Ia entry for R(3,k) needs a witness on at least ``lower_bound``
# vertices, so the smallest order that could improve R(3,13), R(3,14), R(3,15) is 61, 67, 74.
# 61 and 67 are prime, so Z_p is the only group of that order and the circulant search above
# already covers every Cayley graph on them.  74 = 2 * 37 admits exactly two groups, Z_74 and
# the dihedral group D_37, so D_37 is the entire remaining Cayley family at the frontier.
#
# Write D_m = <r, s | r^m, s^2, s r s = r^-1>, elements r^i (index i) and s r^i (index m + i).
# A symmetric connection set is a pair (A, B) with A = -A a set of rotation exponents and B a
# set of reflection exponents -- every reflection is an involution, so B is unconstrained.
# Multiplying out the four cases, Cay(D_m, A u B) is triangle-free exactly when
#
#     A is sum-free in Z_m          (no triangle inside either coset), and
#     (B - B) and A are disjoint    (no triangle straddling the two cosets),
#
# and the second condition says precisely that B is an independent set of C_m(A).  Both cosets
# induce a copy of C_m(A), so alpha(Cay) >= alpha(C_m(A)) -- which is what makes the search
# small: A must already be a good circulant before B is chosen at all.
#
# Conjugation by r^t fixes every rotation and sends s r^i to s r^(i - 2t).  For odd m the
# map t -> 2t is a bijection, so a non-empty B may be translated to contain 0 without loss;
# that is an exact m-fold reduction, not a heuristic.


def dihedral_adjacency(m: int, rotations: Sequence[int], reflections: Sequence[int]) -> list[int]:
    """Adjacency bitmasks of ``Cay(D_m, {r^a : a in A} u {s r^b : b in B})`` on ``2m`` vertices."""

    n = 2 * m
    rot = {a % m for a in rotations}
    ref = {b % m for b in reflections}
    if 0 in rot:
        raise ValueError("the identity is not a connection element")
    if any((m - a) % m not in rot for a in rot):
        raise ValueError("rotation part must be closed under inversion")
    adj = [0] * n
    for i in range(m):
        for a in rot:
            adj[i] |= 1 << ((i + a) % m)                    # r^i ~ r^(i+a)
            adj[m + i] |= 1 << (m + (i + a) % m)            # s r^i ~ s r^(i+a)
        for b in ref:
            # r^i ~ s r^j  iff  i + j in B
            adj[i] |= 1 << (m + (b - i) % m)
            adj[m + i] |= 1 << ((b - i) % m)
    return adj


def adjacency_is_triangle_free(n: int, adj: Sequence[int]) -> bool:
    """Exhaustive over triples: every edge is examined for a common neighbour."""

    for a in range(n):
        rest = adj[a] >> (a + 1)
        b = a + 1
        while rest:
            if rest & 1 and adj[a] & adj[b]:
                return False
            rest >>= 1
            b += 1
    return True


def adjacency_independence_number(n: int, adj: Sequence[int]) -> int:
    size = 0
    while has_independent_set(n, adj, size + 1):
        size += 1
    return size


def _independent_sets(m: int, adj: Sequence[int], cap: int) -> Iterator[tuple[int, ...]]:
    """Every independent set of ``C_m(A)`` of size at most ``cap`` that contains 0, plus the
    empty set.  Complete: the recursion branches on both including and excluding each vertex."""

    yield ()
    if cap <= 0:
        return
    chosen = [0]

    def grow(start: int, banned: int) -> Iterator[tuple[int, ...]]:
        yield tuple(chosen)
        if len(chosen) >= cap:
            return
        for v in range(start, m):
            if (banned >> v) & 1:
                continue
            chosen.append(v)
            yield from grow(v + 1, banned | adj[v])
            chosen.pop()

    yield from grow(1, adj[0] | 1)


def dihedral_search(m: int, k: int, stop_at_first: bool = True) -> dict[str, object]:
    """Complete search over Cayley graphs of ``D_m`` (order ``2m``) with alpha <= k-1.

    ``m`` must be odd, which is the case at every frontier order this module targets, so the
    conjugation reduction "0 in B" is available.
    """

    if m % 2 == 0:
        raise ValueError("m must be odd for the conjugation reduction to be exact")
    n = 2 * m
    max_degree = k - 1
    min_degree = -(-n // (k - 1)) - 1
    classes = difference_classes(m)
    stats = {"rotation_sets": 0, "pairs_tested": 0, "decided_exactly": 0}
    witnesses: list[dict[str, object]] = []

    def rotation_sets() -> Iterator[tuple[int, int]]:
        """Symmetric sum-free A subsets of Z_m, as (mask, degree)."""

        full = (1 << m) - 1

        def walk(index: int, mask: int, sums: int, degree: int) -> Iterator[tuple[int, int]]:
            yield mask, degree
            for j in range(index, len(classes)):
                x = classes[j][0]
                y = m - x
                step = len(classes[j])
                if degree + step > max_degree:
                    continue
                grown = mask | (1 << x) | (1 << (y % m))
                grown_sums = (
                    sums
                    | _rotate(mask, x, m, full)
                    | _rotate(mask, y % m, m, full)
                    | (1 << ((2 * x) % m))
                    | (1 << ((2 * y) % m))
                    | 1
                )
                if grown & grown_sums:
                    continue
                yield from walk(j + 1, grown, grown_sums, degree + step)

        yield from walk(0, 0, 0, 0)

    for mask, degree_a in rotation_sets():
        stats["rotation_sets"] += 1
        coset_adj = build_adjacency(m, mask)
        # both cosets induce C_m(A), so this is a lower bound on alpha of the whole graph
        if adjacency_independence_number(m, coset_adj) > k - 1:
            continue
        budget = max_degree - degree_a
        rotations = residues_of(m, mask)
        for reflections in _independent_sets(m, coset_adj, budget):
            degree = degree_a + len(reflections)
            if degree < min_degree or degree > max_degree:
                continue
            stats["pairs_tested"] += 1
            adj = dihedral_adjacency(m, rotations, reflections)
            if _greedy_lowest(n, adj, k) >= k or _greedy_min_degree(n, adj, k) >= k:
                continue
            stats["decided_exactly"] += 1
            if has_independent_set(n, adj, k):
                continue
            if not adjacency_is_triangle_free(n, adj):     # belt and braces; should not fire
                continue
            witnesses.append(
                {
                    "group": f"D_{m}",
                    "order": n,
                    "rotations": rotations,
                    "reflections": list(reflections),
                    "degree": degree,
                    "independence_number": adjacency_independence_number(n, adj),
                }
            )
            if stop_at_first:
                return {"order": n, "clique_target": k, "witnesses": witnesses, **stats}
    return {"order": n, "clique_target": k, "witnesses": witnesses, **stats}


def seal_certificate(
    k: int, lo: int, hi: int, outcomes: Sequence[SearchOutcome]
) -> dict[str, object]:
    orders = [o.order for o in outcomes]
    if orders != list(range(lo, hi + 1)):
        raise ValueError("outcomes must cover the scanned interval contiguously")
    hits = [o for o in outcomes if o.witnesses]
    best = max((o.order for o in hits), default=0)
    best_witness = next((o.witnesses[0] for o in outcomes if o.order == best), None)
    body: dict[str, object] = {
        "problem": "R(3,k) lower bounds from cyclic (circulant) colourings",
        "clique_target": k,
        "scanned_orders": {"lo": lo, "hi": hi, "count": hi - lo + 1},
        "orders_with_witness": [o.order for o in hits],
        "max_cyclic_order": best,
        "witness": best_witness.to_dict() if best_witness else None,
        "search_nodes_total": sum(o.nodes for o in outcomes),
        "candidates_tested_total": sum(o.candidates_tested for o in outcomes),
        "decided_by_complete_search_total": sum(o.decided_exactly for o in outcomes),
        "exhaustive": True,
        "sampling_used": False,
        "arithmetic": "integer bitmask only",
        "record_comparison": compare_to_record(k, best) if k in SEALED_RECORD else None,
        "published_cyclic_order": SEALED_CYCLIC_ORDERS.get(k),
        "cyclic_source": CYCLIC_SOURCE,
    }
    body["result_core_sha256"] = canonical_sha256(body)
    return {**body, "content_sha256": canonical_sha256(body)}


def verify_certificate(cert: dict[str, object]) -> dict[str, object]:
    body = {kk: vv for kk, vv in cert.items() if kk != "content_sha256"}
    problems: list[str] = []
    if cert.get("content_sha256") != canonical_sha256(body):
        problems.append("content_sha256 mismatch")
    core = {kk: vv for kk, vv in body.items() if kk != "result_core_sha256"}
    if body.get("result_core_sha256") != canonical_sha256(core):
        problems.append("result_core_sha256 mismatch")
    witness = cert.get("witness")
    if witness:
        k = int(cert["clique_target"])  # type: ignore[arg-type]
        checks = verify_colouring(
            int(witness["order"]), list(witness["connection_set"]), k  # type: ignore[index]
        )
        if not checks["valid"]:
            problems.append(f"witness fails re-verification: {checks.get('reason')}")
        if checks.get("independence_number") != witness.get("independence_number"):  # type: ignore[union-attr]
            problems.append("independence number disagrees with the sealed value")
    return {"valid": not problems, "problems": problems}


# --------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, required=True, help="avoid an independent set of this size")
    parser.add_argument("--lo", type=int, default=1)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    outcomes = list(scan_orders(args.k, args.lo, args.hi))
    cert = seal_certificate(args.k, args.lo, args.hi, outcomes)
    text = json.dumps(cert, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
