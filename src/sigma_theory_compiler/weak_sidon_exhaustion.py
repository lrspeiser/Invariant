"""Exhaustive search for minimal-diameter weak Sidon sets (OEIS A345731).

A finite set ``A = {a_0 < a_1 < ... < a_{n-1}}`` of integers is a *weak Sidon
set* (equivalently a *weak B_2 set*, or a *well-spread sequence*) when all
``C(n, 2)`` pairwise sums ``a_i + a_j`` with ``i < j`` are pairwise distinct.

``A345731(n)`` is the least ``L`` such that an ``n``-element weak Sidon set is
contained in ``[0, L]``; equivalently the minimum diameter of an ``n``-element
weak Sidon set.  Published values run ``n = 2..16``; ``A345731(17)`` is open.

Everything here is exact integer arithmetic.  No floating point, no sampling:
the search either traverses a declared space completely or reports that it did
not.

Structure of the search
-----------------------
``search_within(n, lmax, wtab)`` decides whether *any* ``n``-element weak Sidon
set fits inside ``[0, lmax]``.  Sets are built left to right starting from
``a_0 = 0`` (translation invariance).  Two sound bounds prune:

* prefix bound   ``a_k >= wtab[k + 1]``      -- ``{a_0..a_k}`` is a weak Sidon
  set of size ``k + 1`` and diameter ``a_k``.
* suffix bound   ``lmax - a_k >= wtab[n - k]`` -- ``{a_k..a_{n-1}}`` is a weak
  Sidon set of size ``n - k`` and diameter ``a_{n-1} - a_k <= lmax - a_k``.

Both require only that ``wtab[m]`` be a *valid lower bound* on the minimum
diameter of an ``m``-element weak Sidon set, so the search stays sound when
``wtab`` holds lower bounds rather than exact values.

One symmetry is quotiented out: ``A`` and its reflection ``a_{n-1} - A`` are
both weak Sidon sets with the same diameter, so it suffices to examine sets
whose first gap does not exceed their last gap.  Every reflection orbit keeps a
representative, so the reduction loses nothing.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

# Published terms of A345731, offset 2 (n = 2..16).  a(17) was deleted from the
# OEIS entry on 2025-07-27 when a(16) was corrected to 148 by Zhao Hui Du, so
# n = 17 is the first open case.
A345731_PUBLISHED: dict[int, int] = {
    2: 1, 3: 2, 4: 4, 5: 7, 6: 12, 7: 18, 8: 24, 9: 34,
    10: 45, 11: 57, 12: 71, 13: 86, 14: 105, 15: 126, 16: 148,
}

__all__ = [
    "A345731_PUBLISHED",
    "CoverageCertificate",
    "PrefixCell",
    "enumerate_prefix_cells",
    "is_weak_sidon",
    "min_diameter_lower_bound",
    "search_within",
    "witness_diameter",
]


def is_weak_sidon(values: object) -> bool:
    """Exact reference test: are all pairwise sums of distinct elements distinct?

    Deliberately written as the naive definition so that it can referee the
    incremental machinery used by the search.
    """

    items = sorted(int(v) for v in values)  # type: ignore[union-attr]
    if len(set(items)) != len(items):
        return False
    sums = [a + b for a, b in itertools.combinations(items, 2)]
    return len(set(sums)) == len(sums)


def witness_diameter(values: object) -> int:
    items = sorted(int(v) for v in values)  # type: ignore[union-attr]
    return items[-1] - items[0]


def min_diameter_lower_bound(n: int) -> int:
    """A proof-only lower bound on the minimum diameter of an n-element weak Sidon set.

    Each element of a weak Sidon set is the midpoint of at most one 3-term
    arithmetic progression inside the set: two progressions ``{a-d, a, a+d}``
    and ``{a-e, a, a+e}`` with ``d != e`` would give disjoint pairs
    ``{a-d, a+d}`` and ``{a-e, a+e}`` of equal sum ``2a``.  A difference can
    therefore repeat at most once, and only as such a progression, so at least
    ``C(n, 2) - (n - 2)`` distinct positive differences occur, each at most the
    diameter.
    """

    if n < 2:
        return 0
    return n * (n - 1) // 2 - max(n - 2, 0)


# --------------------------------------------------------------------------
# core search
# --------------------------------------------------------------------------


def _search_python(
    n: int,
    lmax: int,
    wtab: list[int],
    a1_lo: int,
    a1_hi: int,
    a2_lo: int,
    a2_hi: int,
    a3_lo: int,
    a3_hi: int,
    node_cap: int,
) -> tuple[bool, int, tuple[int, ...]]:
    """Reference (slow, dependency-free) implementation of :func:`search_within`."""

    chosen = [0] * n
    used: set[int] = set()
    stack_new: list[list[int]] = [[] for _ in range(n)]
    nodes = 0

    def add(k: int, x: int) -> bool:
        fresh: list[int] = []
        for j in range(k):
            s = chosen[j] + x
            if s in used:
                for s2 in fresh:
                    used.discard(s2)
                return False
            used.add(s)
            fresh.append(s)
        stack_new[k] = fresh
        return True

    def drop(k: int) -> None:
        for s in stack_new[k]:
            used.discard(s)
        stack_new[k] = []

    def rec(k: int) -> tuple[bool, tuple[int, ...]]:
        nonlocal nodes
        if k == n:
            if chosen[1] <= chosen[n - 1] - chosen[n - 2]:
                return True, tuple(chosen)
            return False, ()
        lo = max(chosen[k - 1] + 1, wtab[k + 1])
        hi = lmax - wtab[n - k]
        if k == 1:
            lo, hi = max(lo, a1_lo), min(hi, a1_hi)
        elif k == 2:
            lo, hi = max(lo, a2_lo), min(hi, a2_hi)
        elif k == 3:
            lo, hi = max(lo, a3_lo), min(hi, a3_hi)
        for x in range(lo, hi + 1):
            if not add(k, x):
                continue
            chosen[k] = x
            nodes += 1
            if nodes >= node_cap:
                drop(k)
                return False, ()
            ok, wit = rec(k + 1)
            if ok:
                return True, wit
            drop(k)
        return False, ()

    found, witness = rec(1)
    return found, nodes, witness


try:  # pragma: no cover - exercised only when numba is installed
    import numpy as _np
    from numba import njit as _njit

    _HAVE_NUMBA = True
except Exception:  # noqa: BLE001  # pragma: no cover - optional accelerator
    _HAVE_NUMBA = False


if _HAVE_NUMBA:  # pragma: no cover - compiled path, refereed by the python path

    @_njit(cache=True, nogil=True)
    def _search_numba(n, lmax, wtab, a1_lo, a1_hi, a2_lo, a2_hi, a3_lo, a3_hi, node_cap):
        chosen = _np.zeros(n + 1, _np.int64)
        pos = _np.zeros(n + 2, _np.int64)
        used = _np.zeros(2 * lmax + 4, _np.uint8)
        nodes = 0
        k = 1
        pos[1] = 0
        while k >= 1:
            lo = chosen[k - 1] + 1
            lo = max(lo, wtab[k + 1])
            hi = lmax - wtab[n - k]
            if k == 1:
                lo = max(lo, a1_lo)
                hi = min(hi, a1_hi)
            elif k == 2:
                lo = max(lo, a2_lo)
                hi = min(hi, a2_hi)
            elif k == 3:
                lo = max(lo, a3_lo)
                hi = min(hi, a3_hi)
            if k == n - 1:
                # reflection canonical form: first gap <= last gap
                lo2 = chosen[n - 2] + chosen[1]
                lo = max(lo, lo2)
            x = pos[k]
            x = max(x, lo)
            placed = -1
            while x <= hi:
                good = True
                j = 0
                while j < k:
                    s = chosen[j] + x
                    if used[s] != 0:
                        good = False
                        break
                    used[s] = 1
                    j += 1
                if good:
                    placed = x
                    break
                i = 0
                while i < j:
                    used[chosen[i] + x] = 0
                    i += 1
                x += 1
            if placed < 0:
                k -= 1
                if k < 1:
                    break
                c = chosen[k]
                for j in range(k):
                    used[chosen[j] + c] = 0
                pos[k] = c + 1
                continue
            chosen[k] = placed
            nodes += 1
            if nodes >= node_cap:
                return False, -nodes, chosen
            if k + 1 == n:
                return True, nodes, chosen.copy()
            k += 1
            pos[k] = placed + 1
        return False, nodes, chosen


def search_within(
    n: int,
    lmax: int,
    wtab: list[int],
    *,
    a1_range: tuple[int, int] | None = None,
    a2_range: tuple[int, int] | None = None,
    a3_range: tuple[int, int] | None = None,
    node_cap: int = 1 << 62,
    engine: str = "auto",
) -> tuple[bool, int, tuple[int, ...]]:
    """Is there an ``n``-element weak Sidon set inside ``[0, lmax]``?

    Returns ``(found, nodes, witness)``.  ``nodes`` is negative when the node
    cap aborted the traversal, in which case the search is *not* exhaustive and
    ``found=False`` carries no information.
    """

    if n < 2:
        raise ValueError("n must be at least 2")
    if len(wtab) <= n:
        raise ValueError("wtab must be indexable up to n")
    a1_lo, a1_hi = a1_range if a1_range else (1, lmax)
    a2_lo, a2_hi = a2_range if a2_range else (1, lmax)
    a3_lo, a3_hi = a3_range if a3_range else (1, lmax)
    use = engine
    if use == "auto":
        use = "numba" if _HAVE_NUMBA else "python"
    if use == "numba":
        if not _HAVE_NUMBA:
            raise RuntimeError("numba engine requested but numba is unavailable")
        w = _np.asarray(list(wtab) + [0] * (n + 2), dtype=_np.int64)
        found, nodes, arr = _search_numba(
            n, lmax, w, a1_lo, a1_hi, a2_lo, a2_hi, a3_lo, a3_hi, node_cap
        )
        return bool(found), int(nodes), tuple(int(v) for v in arr[:n]) if found else ()
    return _search_python(
        n, lmax, list(wtab), a1_lo, a1_hi, a2_lo, a2_hi, a3_lo, a3_hi, node_cap
    )


# --------------------------------------------------------------------------
# coverage certificate
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrefixCell:
    """One cell of the partition of the search space by ``(a_1, a_2)`` or ``(a_1, a_2, a_3)``."""

    a1: int
    a2: int
    a3: int | None = None

    def as_dict(self) -> dict[str, str]:
        d = {"a1": str(self.a1), "a2": str(self.a2)}
        if self.a3 is not None:
            d["a3"] = str(self.a3)
        return d


def enumerate_prefix_cells(
    n: int, lmax: int, wtab: list[int], depth: int = 2
) -> list[PrefixCell]:
    """Every ``(a_1, a_2)`` a solution could possibly start with.

    Independently recomputable by inspection: ``a_1`` ranges over
    ``[wtab[2], lmax - wtab[n-1]]``, and given ``a_1`` the element ``a_2``
    ranges over ``[max(a_1 + 1, wtab[3]), lmax - wtab[n-2]]``, with the pair
    ``{a_1, a_2}`` required to keep ``{0, a_1, a_2}`` weak Sidon (which for a
    3-element set means only ``a_2 != 2 * a_1``... in fact ``{0, x, y}`` has
    sums ``x, y, x + y``, all distinct whenever ``0 < x < y``, so no pair is
    excluded at this depth).
    """

    cells: list[PrefixCell] = []
    lo1 = max(1, wtab[2])
    hi1 = lmax - wtab[n - 1]
    for a1 in range(lo1, hi1 + 1):
        lo2 = max(a1 + 1, wtab[3])
        hi2 = lmax - wtab[n - 2]
        for a2 in range(lo2, hi2 + 1):
            if depth < 3:
                cells.append(PrefixCell(a1, a2))
                continue
            lo3 = max(a2 + 1, wtab[4])
            hi3 = lmax - wtab[n - 3]
            for a3 in range(lo3, hi3 + 1):
                cells.append(PrefixCell(a1, a2, a3))
    return cells


@dataclass(frozen=True, slots=True)
class CoverageCertificate:
    """Everything needed to check that a declared space was covered exactly."""

    n: int
    lmax: int
    wtab: tuple[int, ...]
    declared_cells: int
    traversed_cells: int
    nodes: int
    exhaustive: bool
    found: bool
    witness: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "n": str(self.n),
            "lmax": str(self.lmax),
            "wtab": [str(v) for v in self.wtab],
            "declared_cells": str(self.declared_cells),
            "traversed_cells": str(self.traversed_cells),
            "nodes": str(self.nodes),
            "exhaustive": self.exhaustive,
            "found": self.found,
            "witness": [str(v) for v in self.witness],
        }

    def is_complete(self) -> bool:
        return self.exhaustive and self.declared_cells == self.traversed_cells
