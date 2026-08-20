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

import decimal
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from .sigma_core import canonical_sha256

SCHEMA = "invariant-creativity-measure-1.0"

#: Every number this module *reports* is computed in this context and emitted as a decimal
#: string.  ``Decimal.ln`` and ``Decimal.exp`` are correctly rounded to the context precision,
#: so the reported digits are a function of the inputs alone and not of whichever libm the host
#: happens to ship -- which is the whole point once these numbers are sealed into a receipt and
#: a replay on another machine has to reproduce them byte for byte.  40 digits against a
#: 6-decimal report is margin, not precision theatre.
REPORT_CONTEXT = decimal.Context(prec=40)

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


#: One coordinate of a behaviour, prepared once: its exact natural logarithm when the value is
#: usable, and always the raw value so an unusable pair can still be compared for equality.
_Coordinate = tuple[Decimal | None, float]


def _prepared(vector: Sequence[float]) -> tuple[_Coordinate, ...]:
    """Take the logarithm of every coordinate ONCE.

    The distance is ``max_i |ln a_i - ln b_i|``, so a population of *n* behaviours needs *n*
    logarithm evaluations, not one per pair.  Doing it per pair is what makes an exact distance
    look expensive; doing it per coordinate makes the pairwise loop pure subtraction.
    """

    prepared: list[_Coordinate] = []
    for value in vector:
        if math.isfinite(value) and value > 0.0:
            prepared.append((REPORT_CONTEXT.ln(Decimal(value)), value))
        else:
            prepared.append((None, value))
    return tuple(prepared)


def _exact_distance(
    left: Sequence[_Coordinate], right: Sequence[_Coordinate]
) -> Decimal | None:
    """The distance between two prepared behaviours, or ``None`` when they are incomparable."""

    if len(left) != len(right):
        return None
    worst = Decimal(0)
    for (log_a, raw_a), (log_b, raw_b) in zip(left, right):
        if log_a is None or log_b is None:
            if raw_a != raw_b:
                return None
            continue
        worst = max(worst, abs(log_a - log_b))
    return worst


def log_relative_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """``max_i |ln(left_i / right_i)|``, the scale-free distance between two behaviours.

    Scale-free is the point: on a domain spanning six decades, an absolute difference says
    almost nothing.  Non-finite or non-positive coordinates make the pair incomparable and
    return infinity, which keeps them in separate clusters rather than silently merging them.
    """

    distance = _exact_distance(_prepared(left), _prepared(right))
    return math.inf if distance is None else float(distance)


def cluster_behaviours(
    vectors: Sequence[tuple[float, ...]], tolerance: float = DEFAULT_TOLERANCE
) -> list[list[int]]:
    """Greedy clustering of output vectors: indices grouped by behavioural identity."""

    limit = Decimal(tolerance)
    clusters: list[list[int]] = []
    reps: list[tuple[_Coordinate, ...]] = []
    for index, vector in enumerate(vectors):
        prepared = _prepared(vector)
        for slot, rep in enumerate(reps):
            distance = _exact_distance(prepared, rep)
            if distance is not None and distance <= limit:
                clusters[slot].append(index)
                break
        else:
            clusters.append([index])
            reps.append(prepared)
    return clusters


def _effective_number(sizes: Sequence[int]) -> Decimal:
    """``exp(H)`` over occupancies: the effective number of behaviours actually present.

    A raw count says 17 when a population is one behaviour repeated sixteen times plus one
    other.  The effective number says about 2, which is what a reader means by "how many
    different things did it try".

    The occupancies are integers, so the shares are exact rationals and the only inexact step
    is the logarithm -- taken here at a declared precision that rounds correctly, so the sealed
    digits do not depend on the host's libm.
    """

    total = sum(sizes)
    if total <= 0:
        return Decimal(0)
    entropy = Decimal(0)
    for size in sizes:
        if size <= 0:
            continue
        share = REPORT_CONTEXT.divide(Decimal(size), Decimal(total))
        entropy -= REPORT_CONTEXT.multiply(share, REPORT_CONTEXT.ln(share))
    return REPORT_CONTEXT.exp(entropy)


def measure_creativity(
    programs: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    novelty_floor: float = NOVELTY_FLOOR,
    origin: str | None = "proposed",
    best_quality: float | None = None,
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

    reps = [_prepared(vectors[cluster[0]]) for cluster in clusters]
    spans: list[Decimal] = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            distance = _exact_distance(reps[i], reps[j])
            if distance is not None:
                spans.append(distance)
    spans.sort()
    span = spans[len(spans) // 2] if spans else Decimal(0)

    # A high waste ratio means opposite things at opposite ends of the quality range, and reading
    # it alone gets the verdict backwards.  A measured sequence run wrote 44 distinct sources that
    # were one behaviour -- identical to the stuck run's signature -- except the behaviour was the
    # exact answer at quality 1.0.  Forty-four spellings of a correct solution is convergence and
    # the search is finished; forty-four spellings of a wrong one is a search that cannot move.
    distinct_behaviours = len(clusters)
    if best_quality is None:
        regime = "unknown_no_quality_supplied"
    elif best_quality >= 0.99:
        regime = "converged"
    elif distinct_behaviours <= 1 and len(usable) > 2:
        regime = "stuck"
    else:
        regime = "exploring"
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
        # Exact rationals: counts over counts, divided at the declared precision rather than
        # in binary floating point, so the reported digits are the true digits.
        "wasted_variation_ratio": format(
            REPORT_CONTEXT.divide(Decimal(len(sources)), Decimal(distinct_behaviours))
            if distinct_behaviours
            else Decimal(0),
            ".6f",
        ),
        "known_collapse_fraction": format(
            REPORT_CONTEXT.divide(Decimal(known_hits), Decimal(len(usable)))
            if usable
            else Decimal(0),
            ".6f",
        ),
        "behavioural_span_median": format(span, ".6g"),
        "regime": regime,
        "best_quality": format(best_quality, ".9f") if best_quality is not None else None,
        "regime_rule": (
            "converged when best quality >= 0.99, whatever the waste ratio; stuck when the whole "
            "population is one behaviour and quality is not; otherwise exploring"
        ),
        "claims": {
            "measures_spread_not_correctness": True,
            "syntactic_variation_alone_cannot_raise_it": True,
            "known_collapse_is_reported_not_penalised": True,
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


#: The metrics an A/B report walks, and which way is an improvement.  ``report`` means the
#: number is published and never scored: recovering a known law is a real success for a blind
#: search, so it must not be able to make a change look bad.
COMPARED_METRICS: tuple[tuple[str, str], ...] = (
    ("effective_novel_behaviours", "up"),
    ("effective_behaviours", "up"),
    ("wasted_variation_ratio", "down"),
    ("known_collapse_fraction", "report"),
)


def compare(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """A/B two measurements and say which way each number moved.

    The measurements arrive as decimal strings and the deltas stay decimal: subtracting two
    sealed six-decimal numbers is exact arithmetic, and doing it in binary floating point would
    put a rounding artefact on a certificate path for no reason at all.
    """

    def _get(block: Mapping[str, Any], key: str) -> Decimal:
        raw = block.get(key, "0")
        try:
            return Decimal(str(raw))
        except decimal.InvalidOperation:
            return Decimal(0)

    rows = []
    for key, better in COMPARED_METRICS:
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
