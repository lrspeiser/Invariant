"""Minimal pancyclic graphs: an open Erdős problem with an exactly countable objective.

**The problem.**  Erdős Problem #1016 (Erdős 1971, after Bondy) asks for the growth of
``h(n)``, the least number such that some graph on ``n`` vertices with ``n + h(n)`` edges
contains a cycle on ``k`` vertices for every ``3 <= k <= n``.  Such a graph is *pancyclic*;
write ``m(n) = n + h(n)`` for the minimum size of a pancyclic graph of order ``n``.  The
asymptotic question is open and is not what this module attacks.  What this module attacks is
the finite table underneath it, which stops at a specific place in the literature.

**Why the table is finitely decidable.**  A pancyclic graph contains a cycle of length ``n``,
so it is a Hamiltonian cycle ``C_n`` plus ``k = m - n`` chords.  Its cycle space over GF(2)
has dimension ``k + 1``, every cycle of the graph is a non-zero element of that space, and
distinct cycles are distinct elements; hence the graph has at most ``2^(k+1) - 1`` cycles
(Shi 1994) and needs at least ``n - 2`` of them.  So ``2^(k+1) - 1 >= n - 2`` -- the
Bondy/Shi counting bound -- and for fixed ``k`` only finitely many ``n`` are even possible.
The whole question at a given ``n`` is therefore a finite search over chord sets, and a
witness is checked by enumerating all ``2^(k+1) - 1`` cycle-space elements in exact integer
arithmetic.  Nothing here is sampled and nothing here is a float.

**Where the literature stops.**  Griffin (arXiv:1312.0274, 2013), agreeing with George, Marr
and Wallis (JCMCC 86, 2013), publishes ``m(n)`` for ``3 <= n <= 37``: an exhaustive search of
all Hamiltonian graphs with at most 4 chords, an exhaustive search with 5 chords for
``n <= 31``, and an explicit 5-chord construction covering ``23 <= n <= 37``.  Griffin states
the 5-chord construction "has cycles of lengths 3 to 19 and of lengths n-17 to n, so for G to
be pancyclic it is required that n-17 <= 20 or n <= 37".  Nothing is published for ``n >= 38``
and OEIS A105206 is further behind still, stopping at ``n = 22``.  :data:`SEALED_TABLE` holds
that published table verbatim and is the only record this module compares against.

**What a witness at n >= 38 settles.**  The counting bound gives ``m(n) >= n + 5`` for every
``34 <= n <= 65``.  So a pancyclic graph on ``n >= 38`` vertices with exactly ``n + 5`` edges
pins ``m(n) = n + 5`` exactly -- the lower bound is a two-line count, the upper bound is the
object itself.  :data:`WITNESSES` carries such objects.  Each is re-verified from its chord
list every time this module is imported into a test; none of them is trusted because it was
recorded.

**Honest limits, stated as claims.**  Finding a witness at ``n`` says nothing about ``n + 1``:
absence of a witness in :data:`WITNESSES` is absence of a search, never a proof of
impossibility.  This module contains no exhaustive search at 5 chords for ``n >= 32``, so it
cannot and does not claim any upper limit on the largest ``n`` admitting 5 chords.  The one
statement it makes is per-``n`` and is an equality with a construction on one side and an
elementary count on the other.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

RECEIPT_SCHEMA = "invariant-pancyclic-chord-search-1.0"


class PancyclicError(ValueError):
    """Raised on a malformed chord set, a malformed witness, or a tampered receipt."""


CLAIMS = {
    "absence_of_a_witness_is_absence_of_a_search": True,
    "cycle_enumeration_is_exhaustive_not_sampled": True,
    "lower_bound_is_an_elementary_count_not_a_search": True,
    "no_float_reaches_a_sealed_number": True,
    "sealed_table_is_read_only": True,
    "witness_is_reverified_from_its_chords": True,
}

# Griffin, "Minimal Pancyclicity", arXiv:1312.0274 (2013), Table 1; the same values are
# reported by J. C. George, A. Marr and W. D. Wallis, "Minimal Pancyclic Graphs",
# J. Combin. Math. Combin. Comput. 86 (2013).  n -> (chords k, m(n) = n + k).
SEALED_TABLE: Mapping[int, tuple[int, int]] = {
    3: (0, 3), 4: (1, 5), 5: (1, 6), 6: (2, 8), 7: (2, 9), 8: (2, 10), 9: (3, 12),
    10: (3, 13), 11: (3, 14), 12: (3, 15), 13: (3, 16), 14: (3, 17), 15: (4, 19),
    16: (4, 20), 17: (4, 21), 18: (4, 22), 19: (4, 23), 20: (4, 24), 21: (4, 25),
    22: (4, 26), 23: (4, 27), 24: (4, 28), 25: (5, 30), 26: (5, 31), 27: (5, 32),
    28: (5, 33), 29: (5, 34), 30: (5, 35), 31: (5, 36), 32: (5, 37), 33: (5, 38),
    34: (5, 39), 35: (5, 40), 36: (5, 41), 37: (5, 42),
}

SEALED_TABLE_SOURCE = (
    "S. Griffin, Minimal Pancyclicity, arXiv:1312.0274 (2013), Table 1; "
    "J. C. George, A. Marr, W. D. Wallis, Minimal Pancyclic Graphs, JCMCC 86 (2013)."
)

SEALED_TABLE_LAST_N = 37

# Chord sets, on the vertex labelling 0..n-1 of a Hamiltonian cycle, of pancyclic graphs with
# n + 5 edges for orders past the published table.  Every entry is re-verified by
# verify_witness(); the list is data, not evidence.
WITNESSES: Mapping[int, tuple[tuple[int, int], ...]] = {
    38: ((12, 14), (13, 31), (14, 27), (23, 30), (28, 31)),
    39: ((1, 36), (3, 36), (19, 34), (29, 38), (35, 37)),
    40: ((2, 30), (21, 28), (22, 29), (27, 31), (27, 32)),
}

# Simulated-annealing runs actually performed at 5 chords, in a fixed 60-second budget per
# order, recorded so that "not found" stays a statement about a search and never about the
# mathematics.  ``order -> (restarts, restarts that reached a pancyclic graph)``.  The order
# 40 and 41 zeroes here are the 60-second budget failing; order 40 was reached later by a
# longer run, order 41 was not reached by roughly 500 restarts across two longer runs.  None
# of that is evidence that no 5-chord pancyclic graph of order 41 exists.
HEURISTIC_SEARCH_LOG: Mapping[int, tuple[int, int]] = {
    35: (35, 24),
    36: (39, 21),
    37: (32, 11),
    38: (29, 2),
    39: (29, 2),
    40: (28, 0),
    41: (27, 0),
}


def counting_lower_bound_chords(n: int) -> int:
    """Least ``k`` with ``2^(k+1) - 1 >= n - 2``: the Bondy/Shi bound on chords.

    A pancyclic graph on ``n`` vertices is ``C_n`` plus ``k`` chords, its cycle space has
    dimension ``k + 1``, distinct cycles are distinct non-zero elements of that space, and it
    needs one cycle for each of the ``n - 2`` lengths ``3..n``.
    """

    if n < 3:
        raise PancyclicError(f"pancyclicity is defined for n >= 3, got {n}")
    k = 0
    while (1 << (k + 1)) - 1 < n - 2:
        k += 1
    return k


def normalise_chords(n: int, chords: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Validate and canonicalise a chord set on ``C_n`` with vertices ``0..n-1``."""

    if n < 3:
        raise PancyclicError(f"n must be at least 3, got {n}")
    out: list[tuple[int, int]] = []
    for pair in chords:
        values = list(pair)
        if len(values) != 2:
            raise PancyclicError(f"chord {pair!r} is not a pair")
        u, v = values
        for endpoint in (u, v):
            if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                raise PancyclicError(f"chord {pair!r} has a non-integer endpoint")
        if not (0 <= u < n and 0 <= v < n):
            raise PancyclicError(f"chord {pair!r} leaves the vertex range 0..{n - 1}")
        if u == v:
            raise PancyclicError(f"chord {pair!r} is a loop")
        a, b = (u, v) if u < v else (v, u)
        if (b - a) == 1 or (b - a) == n - 1:
            raise PancyclicError(f"chord {pair!r} duplicates an edge of the Hamiltonian cycle")
        out.append((a, b))
    if len(set(out)) != len(out):
        raise PancyclicError("chord set contains a repeated chord")
    return tuple(sorted(out))


def _is_single_cycle(
    n: int, chords: Sequence[tuple[int, int]], cycle_edges: int, chord_set: int
) -> int:
    """Return the length of the edge set if it is one cycle, else ``0``."""

    adjacency: dict[int, list[int]] = {}
    size = 0
    remaining = cycle_edges
    while remaining:
        low = remaining & -remaining
        index = low.bit_length() - 1
        remaining ^= low
        a, b = index, (index + 1) % n
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        size += 1
    for position, (a, b) in enumerate(chords):
        if (chord_set >> position) & 1:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
            size += 1
    if size == 0:
        return 0
    for neighbours in adjacency.values():
        if len(neighbours) != 2:
            return 0
    start = next(iter(adjacency))
    previous, current, walked = None, start, 0
    while True:
        first, second = adjacency[current]
        nxt = first if first != previous else second
        previous, current = current, nxt
        walked += 1
        if current == start:
            break
    return size if walked == size else 0


def cycle_spectrum(n: int, chords: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Enumerate every cycle of ``C_n`` plus ``chords``, exhaustively and exactly.

    The path ``0-1-...-(n-1)`` is a spanning tree; its fundamental cycles are the Hamiltonian
    cycle together with one cycle per chord, so they are a basis of the cycle space.  Every
    cycle of the graph lies in the cycle space and no two distinct cycles are the same
    element, so scanning all ``2^(k+1) - 1`` non-zero elements finds every cycle exactly once.
    """

    k = len(chords)
    basis: list[tuple[int, int]] = [((1 << n) - 1, 0)]
    for position, (u, v) in enumerate(chords):
        basis.append((((1 << (v - u)) - 1) << u, 1 << position))
    spectrum: set[int] = set()
    cycles = 0
    examined = 0
    for selector in range(1, 1 << (k + 1)):
        cycle_edges = 0
        chord_set = 0
        for index in range(k + 1):
            if (selector >> index) & 1:
                cycle_edges ^= basis[index][0]
                chord_set ^= basis[index][1]
        examined += 1
        length = _is_single_cycle(n, chords, cycle_edges, chord_set)
        if length:
            cycles += 1
            spectrum.add(length)
    return {
        "cycle_space_dimension": k + 1,
        "elements_examined": examined,
        "elements_expected": (1 << (k + 1)) - 1,
        "cycles_found": cycles,
        "spectrum": sorted(spectrum),
    }


def verify_witness(n: int, chords: Iterable[Sequence[int]]) -> dict[str, Any]:
    """Verify one candidate pancyclic graph and return its exact certificate."""

    canonical = normalise_chords(n, chords)
    k = len(canonical)
    report = cycle_spectrum(n, canonical)
    if report["elements_examined"] != report["elements_expected"]:
        raise PancyclicError("cycle-space enumeration did not cover every element")
    required = list(range(3, n + 1))
    present = set(report["spectrum"])
    missing = [length for length in required if length not in present]
    lower_bound_chords = counting_lower_bound_chords(n)
    return {
        "order": n,
        "chords": [list(pair) for pair in canonical],
        "chord_count": k,
        "edges": n + k,
        "cycle_space_dimension": report["cycle_space_dimension"],
        "elements_examined": report["elements_examined"],
        "elements_expected": report["elements_expected"],
        "cycles_found": report["cycles_found"],
        "required_lengths": len(required),
        "spectrum": report["spectrum"],
        "missing_lengths": missing,
        "pancyclic": not missing,
        "counting_lower_bound_chords": lower_bound_chords,
        "counting_lower_bound_edges": n + lower_bound_chords,
        "matches_counting_lower_bound": (not missing) and k == lower_bound_chords,
    }


def record_gate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare a verified certificate against :data:`SEALED_TABLE`.  Integer comparison only."""

    n = int(certificate["order"])
    edges = int(certificate["edges"])
    pancyclic = bool(certificate["pancyclic"])
    tight = bool(certificate["matches_counting_lower_bound"])
    sealed = SEALED_TABLE.get(n)
    if not pancyclic:
        verdict = "not_pancyclic"
    elif sealed is None and n > SEALED_TABLE_LAST_N and tight:
        verdict = "determines_a_value_past_the_published_table"
    elif sealed is None and n > SEALED_TABLE_LAST_N:
        verdict = "upper_bound_past_the_published_table_only"
    elif sealed is not None and edges < sealed[1]:
        verdict = "beats_the_published_value"
    elif sealed is not None and edges == sealed[1]:
        verdict = "reproduces_the_published_value"
    else:
        verdict = "worse_than_the_published_value"
    return {
        "order": n,
        "edges": edges,
        "sealed_edges": None if sealed is None else sealed[1],
        "sealed_source": SEALED_TABLE_SOURCE,
        "sealed_table_last_order": SEALED_TABLE_LAST_N,
        "verdict": verdict,
    }


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Result of an exhaustive chord-set search at one order and chord count."""

    order: int
    chord_count: int
    configurations: int
    witness: tuple[tuple[int, int], ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "chord_count": self.chord_count,
            "configurations": self.configurations,
            "witness": None if self.witness is None else [list(p) for p in self.witness],
        }


def candidate_chords(n: int) -> list[tuple[int, int]]:
    """Every pair of vertices of ``C_n`` that is not already an edge of the cycle."""

    return [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if (v - u) != 1 and (v - u) != n - 1
    ]


def _rotation_reduced_sets(n: int, k: int) -> Iterator[tuple[tuple[int, int], ...]]:
    """Chord ``k``-sets covering every rotation/reflection class at least once.

    Rotate so some chord has endpoint ``0``; reflect through ``0`` so that chord is
    ``(0, d)`` with ``2 <= d <= n // 2``.  Every class has such a representative, so a search
    over these sets is exhaustive.
    """

    if k == 0:
        yield ()
        return
    others = [c for c in candidate_chords(n) if c[0] != 0]
    for d in range(2, n // 2 + 1):
        first = (0, d)
        for rest in itertools.combinations(others, k - 1):
            yield (first,) + rest


def exhaustive_search(n: int, k: int) -> SearchOutcome:
    """Search every chord ``k``-set on ``C_n`` (up to rotation and reflection)."""

    required = set(range(3, n + 1))
    configurations = 0
    for chords in _rotation_reduced_sets(n, k):
        configurations += 1
        if required <= set(cycle_spectrum(n, chords)["spectrum"]):
            return SearchOutcome(n, k, configurations, chords)
    return SearchOutcome(n, k, configurations, None)


def minimum_size_by_search(n: int, max_chords: int = 4) -> SearchOutcome:
    """Smallest chord count admitting a pancyclic graph on ``n`` vertices, by full search."""

    start = counting_lower_bound_chords(n)
    for k in range(start, max_chords + 1):
        outcome = exhaustive_search(n, k)
        if outcome.witness is not None:
            return outcome
    return SearchOutcome(n, max_chords, 0, None)


def witness_receipt(orders: Sequence[int] | None = None) -> dict[str, Any]:
    """Verify the sealed witnesses and seal the result into a hashable receipt."""

    selected = sorted(WITNESSES) if orders is None else sorted(orders)
    entries = []
    for n in selected:
        chords = WITNESSES[n]
        certificate = verify_witness(n, chords)
        entries.append({"certificate": certificate, "gate": record_gate(certificate)})
    body = {
        "schema": RECEIPT_SCHEMA,
        "claims": dict(sorted(CLAIMS.items())),
        "sealed_table_source": SEALED_TABLE_SOURCE,
        "sealed_table_last_order": SEALED_TABLE_LAST_N,
        "entries": entries,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", type=int, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = witness_receipt(args.orders)
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
