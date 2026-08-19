"""A measure of creativity that counts behaviours, not spellings.

The obvious measure -- how many different programs did it write -- is worse than useless here,
and one measured run proves it.  A live model asked for laws fitting a blinded rotation curve
returned six syntactically distinct programs::

    u / (1 - u/2 + u*u/6)
    u * (1 + u)**(-0.25)
    2*u / (2 + u)
    u * (1 - u/3 + u*u/12)

Every one of them is the identity map to within 5e-7 over the declared domain, because the
domain runs from 1e-12 to 1e-6 and each correction term is negligible on it.  Source diversity
was high.  Functional diversity was exactly one.  A measure that cannot tell those apart will
reward a proposer for finding new ways to spell the seed.

So every quantity here is computed from a program's OUTPUT VECTOR on the declared probe points,
which is the only thing about a program that can matter.  Two programs are the same behaviour
when their outputs agree to the declared tolerance, however differently they are written; two
programs are different behaviours when their outputs differ, however similar the source.

Three numbers answer the question "did this change make the search more creative, or did it
narrow onto what it already knows?"

``effective_novel_behaviours``
    The headline.  The effective number of distinct behaviours that are NOT matches of a known
    solution family -- ``exp`` of the Shannon entropy of the cluster occupancies, so a population
    dominated by one behaviour scores near 1 no matter how long its tail.  This is what should go
    up if a change is worth keeping.

``wasted_variation_ratio``
    ``distinct_sources / distinct_behaviours``.  1.0 means every new program does something new.
    The live run above scored 16.0: sixteen spellings of one function.  Rising values mean the
    proposer is working hard and exploring nothing.

``known_collapse_fraction``
    The share of programs that matched a declared known family.  This is the "narrowing onto what
    it knows" term.  It is reported SEPARATELY and never folded into the headline, because
    recovering a known law is a real success for a blind search and only stops being interesting
    if it is all the search does.

What the measure deliberately does not do: judge whether a behaviour is *good*.  Quality is
already measured by the evaluator.  Creativity here is spread, not correctness, and a run can
honestly be very creative and entirely wrong.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .sigma_core import canonical_sha256

SCHEMA = "invariant-creativity-measure-1.0"

#: Two output vectors are the same behaviour when every coordinate agrees to this log-relative
#: tolerance.  It is declared rather than tuned: 1e-9 is far below any difference a search could
#: act on and far above float replay noise.
DEFAULT_TOLERANCE = 1e-9

#: A program counts as novel when its multiplier is at least this.  Zero means it matched a
#: declared known family exactly.
NOVELTY_FLOOR = 1e-6


def _vector(outputs: Sequence[str]) -> tuple[float, ...] | None:
    try:
        return tuple(float(value) for value in outputs)
    except (TypeError, ValueError):
        return None


def log_relative_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """``max_i |ln(left_i / right_i)|``, the scale-free distance between two behaviours.

    Scale-free is the point: on a domain spanning six decades, an absolute difference says
    almost nothing.  Non-finite or non-positive coordinates make the pair incomparable and
    return infinity, which keeps them in separate clusters rather than silently merging them.
    """

    if len(left) != len(right):
        return math.inf
    worst = 0.0
    for a, b in zip(left, right):
        if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
            if a != b:
                return math.inf
            continue
        worst = max(worst, abs(math.log(a / b)))
    return worst


def cluster_behaviours(
    vectors: Sequence[tuple[float, ...]], tolerance: float = DEFAULT_TOLERANCE
) -> list[list[int]]:
    """Greedy clustering of output vectors: indices grouped by behavioural identity."""

    clusters: list[list[int]] = []
    reps: list[tuple[float, ...]] = []
    for index, vector in enumerate(vectors):
        for slot, rep in enumerate(reps):
            if log_relative_distance(vector, rep) <= tolerance:
                clusters[slot].append(index)
                break
        else:
            clusters.append([index])
            reps.append(vector)
    return clusters


def _effective_number(sizes: Sequence[int]) -> float:
    """``exp(H)`` over occupancies: the effective number of behaviours actually present.

    A raw count says 17 when a population is one behaviour repeated sixteen times plus one
    other.  The effective number says about 2, which is what a reader means by "how many
    different things did it try".
    """

    total = sum(sizes)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for size in sizes:
        if size <= 0:
            continue
        share = size / total
        entropy -= share * math.log(share)
    return math.exp(entropy)


def measure_creativity(
    programs: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    novelty_floor: float = NOVELTY_FLOOR,
    origin: str | None = "proposed",
) -> dict[str, Any]:
    """Behavioural creativity of one population of sealed programs.

    ``origin`` restricts the population to programs the search actually produced; seeds and
    planted probes are not the proposer's work and would flatter or penalise it unfairly.
    """

    selected = [
        program
        for program in programs
        if origin is None or program.get("origin") == origin
    ]
    usable: list[Mapping[str, Any]] = []
    vectors: list[tuple[float, ...]] = []
    for program in selected:
        vector = _vector(program.get("outputs") or ())
        if vector:
            usable.append(program)
            vectors.append(vector)

    sources = {program.get("source", "") for program in usable}
    clusters = cluster_behaviours(vectors, tolerance)
    sizes = [len(cluster) for cluster in clusters]

    def _novelty(program: Mapping[str, Any]) -> float:
        block = program.get("novelty") or {}
        try:
            return float(block.get("novelty_multiplier", 0.0))
        except (TypeError, ValueError):
            return 0.0

    novel_clusters = [
        cluster
        for cluster in clusters
        if max(_novelty(usable[i]) for i in cluster) >= novelty_floor
    ]
    known_hits = sum(1 for program in usable if _novelty(program) < novelty_floor)

    reps = [vectors[cluster[0]] for cluster in clusters]
    spans: list[float] = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            distance = log_relative_distance(reps[i], reps[j])
            if math.isfinite(distance):
                spans.append(distance)
    spans.sort()
    span = spans[len(spans) // 2] if spans else 0.0

    distinct_behaviours = len(clusters)
    payload = {
        "schema_version": SCHEMA,
        "declared": {
            "tolerance": format(tolerance, ".3g"),
            "novelty_floor": format(novelty_floor, ".3g"),
            "origin_filter": origin,
            "distance": "max_i |ln(a_i / b_i)| over the declared probe points",
        },
        "population": {
            "programs": len(usable),
            "distinct_sources": len(sources),
            "distinct_behaviours": distinct_behaviours,
        },
        # Decimal strings, not floats: canonical_sha256 forbids floats so a receipt can never
        # carry cross-runtime serialization drift.
        "effective_novel_behaviours": format(
            _effective_number([len(cluster) for cluster in novel_clusters]), ".6f"
        ),
        "effective_behaviours": format(_effective_number(sizes), ".6f"),
        "wasted_variation_ratio": format(
            len(sources) / distinct_behaviours if distinct_behaviours else 0.0, ".6f"
        ),
        "known_collapse_fraction": format(known_hits / len(usable) if usable else 0.0, ".6f"),
        "behavioural_span_median": format(span, ".6g"),
        "claims": {
            "measures_spread_not_correctness": True,
            "syntactic_variation_alone_cannot_raise_it": True,
            "known_collapse_is_reported_not_penalised": True,
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def compare(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """A/B two measurements and say which way each number moved."""

    def _get(block: Mapping[str, Any], key: str) -> float:
        return float(block.get(key, 0.0))

    rows = []
    for key, better in (
        ("effective_novel_behaviours", "up"),
        ("effective_behaviours", "up"),
        ("wasted_variation_ratio", "down"),
        ("known_collapse_fraction", "report"),
    ):
        left, right = _get(before, key), _get(after, key)
        delta = right - left
        if better == "report":
            verdict = "reported"
        elif better == "up":
            verdict = "better" if delta > 0 else ("worse" if delta < 0 else "unchanged")
        else:
            verdict = "better" if delta < 0 else ("worse" if delta > 0 else "unchanged")
        rows.append(
            {
                "metric": key,
                "before": format(left, ".6f"),
                "after": format(right, ".6f"),
                "delta": format(delta, ".6f"),
                "direction_that_is_better": better,
                "verdict": verdict,
            }
        )
    headline = next(r for r in rows if r["metric"] == "effective_novel_behaviours")
    return {
        "schema_version": SCHEMA,
        "rows": rows,
        "verdict": headline["verdict"],
        "rule": (
            "keep the change when effective_novel_behaviours rises: the search tried more "
            "genuinely different things that are not already-known answers"
        ),
    }
