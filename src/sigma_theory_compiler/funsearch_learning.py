"""Does the FunSearch loop *learn*, or is its score channel merely open?

C6 says a sweep must be steered by what survived the last one, and its falsifier is an
enumeration whose proposal distribution is identical at both ends.  An earlier version of
this module claimed to measure that.  Its verifier refuted it, and the refutation is the
reason this file exists in its present form, so it is stated before anything else.

What went wrong the first time
------------------------------

The v1 statistic was a difference in differences: how much closer the late proposals sat to
the top-scoring pole of the mid-campaign population than to the bottom-scoring pole of the
same population.  The null was a **permutation of generations** -- relabel which generations
count as early and which as late, recompute, and ask how often a random relabelling beats
the true ordering.  ``runs/math/funsearch/learning-v1.json`` reported ``p = 1/1000`` on its
live arm, ``367/500`` on its ablation, and the verdict ``learned``.

That null is anti-conservative, and it is anti-conservative for a reason that has nothing to
do with the arithmetic.  Permuting generations destroys the *time* ordering while holding
the *axis* fixed.  So it asks "does the proposal distribution drift?" and not "does it drift
along the direction score picks out".  A proposer whose output wanders monotonically in any
direction whatever will project some non-zero component onto whatever axis it is handed, and
the true early/late labelling will then beat almost every relabelling of it.  The score axis
is never put at risk.  The verifier that refused to merge v1 re-ran one of its arms against a
null that randomises the *poles* instead of the generations and watched it move from
``p = 0.0060`` to ``p = 0.3632``: the drift was real, its alignment with score was not.

The distinction those two nulls separate is exactly the one C6 cares about.  A campaign in
which ``final_score`` is computed, recorded and threaded through the loop has an **open score
channel**; a campaign in which the surviving programs measurably pull later proposals toward
themselves is **learning**.  v1 measured the first and reported the second.

The three nulls this module runs
--------------------------------

One statistic, three nulls, and a verdict that only the honest two can support.

``T`` is the normalised projection of the proposal drift onto the score axis.  Write
``p(v)`` for the distribution a count vector induces.  The drift is
``D = p(end) - p(start)``, the score axis is ``A = p(elite) - p(foil)``, and

    T = <D, A> / ||A||_1 .

``T > 0`` means late proposals moved toward the high-scoring pole *relative to* the
low-scoring pole of the same population.  Dividing by ``||A||_1`` -- which is exactly
``2 * TV(elite, foil)`` -- is what makes a randomised pole pair comparable to the real one:
random poles drawn from a mixed pool sit closer together than the two score extremes do, and
an unnormalised statistic would call the real pair extreme for that reason alone.  ``T`` is
bounded by ``max_s |D_s| <= 1``.

**N1, generation permutation.**  The v1 null, kept because deleting it would hide the
finding.  It is marked inadmissible in the receipt: it may not support a positive verdict.

**N2, pole randomisation.**  Hold the observed windows fixed; permute the ``final_score``
labels across the mid-campaign pool and re-form the two poles from the permuted ranking.
Under the null "score carries no information about which features propagate" the observed
axis is exchangeable with the drawn ones, so this is an exact conditional permutation test.

**N3, generation-stratified pole randomisation.**  N2 alone still has a confound: in a
campaign that improves, the top-scoring programs sit *late* in the middle section, so the
elite pole is temporally nearer the end window than a uniformly drawn pole would be, and
proximity in time can masquerade as alignment with score.  N3 removes it by construction --
each drawn pole takes exactly as many programs from each generation as the real pole did, so
the drawn and observed axes have identical temporal profiles and differ only in *which*
programs within a generation were called elite.

**The ablation.**  :func:`funsearch_loop.run_problem` takes ``selection_pressure=False``,
which severs every path from ``final_score`` to the proposer and changes nothing else.  A
statistic that fires there is measuring drift, and vetoes the run.

Calibration and power, both measured rather than asserted
---------------------------------------------------------

:func:`null_calibration` replaces the score axis with a score-blind axis of the same
construction and counts how often each null fires anyway.  It is run on the campaign's own
data, so it is a false-positive rate for this instrument on this run, not a simulation.  Its
output is the evidence for the paragraph above and it is sealed into the receipt.

:func:`detection_floor` asks the opposite question, and a negative result is worthless
without it.  The alternative hypothesis is *built*: the leading programs of each early
generation are replaced by foil-pole programs and of each late generation by elite-pole
programs, one more per generation at each rung, until both admissible nulls fire.  The poles
and therefore both null reference distributions are untouched, so the only thing that moves
is the drift.  The smallest firing count is the detection floor, in the units the campaign
supplies: proposals per generation.  A null result above a known floor is a measurement; a
null result from an instrument that fires at no planted level is an instrument failure
wearing the same word, which is exactly the C1 distinction between a real negative and an
uninformative one.

A campaign with no floor at all is *not* evidence that the search does not learn, and
:func:`verdict_of` refuses to report it as such: it returns ``uninformative_null``, which is
the statistical form of the reachability certificate C1 demands before a null result may be
published.  Half the campaigns in the sealed sweep land there, and the reason is worth
stating because it is a property of the loop and not of the test: in a campaign that
improves, the top-scoring mid-campaign programs *are* the late ones, so the score axis and
the clock become collinear and no amount of arithmetic can attribute the drift to one rather
than the other.

The pole construction and the campaign geometry were chosen on that floor and on nothing
else.  Balanced per-generation poles and pole sizes of 12, 20, 25, 40 and 60 were measured;
so were campaigns of 60 and 80 generations at 5 and 8 proposals per call.  None beat the v1
geometry's floor and one had no floor at all, so the v1 geometry is kept -- which also keeps
the comparison against v1 a comparison of nulls rather than of two different experiments.
Selecting on the floor cannot manufacture the headline: a more sensitive instrument that
still reports no shift is a stronger negative, not a weaker one.

Exact arithmetic
----------------

``T`` is one :class:`~fractions.Fraction` built from two integers.  With ``S = sum(start)``,
``E = sum(end)``, ``P = sum(elite)``, ``Q = sum(foil)``,

    d_s = end_s * S - start_s * E      (numerator of D_s over S*E)
    a_s = elite_s * Q - foil_s * P     (numerator of A_s over P*Q)
    T   = sum_s d_s a_s / (S * E * sum_s |a_s|)

and the ``P*Q`` cancels.  Every intermediate is a Python integer; there is one division and
it is rational.  p-values are ``(hits + 1) / (draws + 1)``, exact by construction, compared
against a declared rational alpha.  ``final_score`` is a float and is used only to *order*
programs into poles -- an ordering, never an arithmetic step, and never on the certificate.

What was actually measured
--------------------------

Reported in ``runs/math/funsearch/learning-v2.json`` and regenerated by ``--out``, not
summarised here, because a docstring cannot be re-derived from a hash.  The headline is in
:data:`FINDING`.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import funsearch_loop as fl
from .sigma_core import canonical_sha256

RECEIPT_SCHEMA = "invariant-funsearch-learning-measurement-2.0"

#: The one-line result, kept next to the code that produces it so the two cannot drift.
FINDING = (
    "The v1 generation-permutation null is anti-conservative and is demoted to "
    "inadmissible. Under the two pole-randomised nulls the shift the v1 receipt reported "
    "is not established. A campaign where the instrument is separately shown able to see a "
    "planted score-aligned shift and does not see one is a real negative; a campaign where "
    "it is not is reported as an uninformative null and not as a negative."
)

CLAIMS = {
    "a_negative_is_reported_only_with_a_measured_detection_floor": True,
    "a_shift_under_the_ablation_would_void_the_result": True,
    "an_open_score_channel_is_not_the_same_as_learning": True,
    "distribution_distance_is_exact_rational_arithmetic": True,
    "generation_permutation_alone_can_support_a_positive_verdict": False,
    "null_false_positive_rates_are_measured_not_assumed": True,
    "proposal_distribution_is_measured_not_modelled": True,
}

SCOPE = (
    "Two campaigns of the declared FunSearch loop that differ in exactly one field: "
    "whether final_score is allowed to reach the proposer. For each, the drift of the "
    "pooled feature-count distribution from the first window of generations to the last is "
    "projected onto the axis separating the top-scoring from the bottom-scoring pole of the "
    "mid-campaign population, and that projection is tested against three nulls: a "
    "permutation of generations (inadmissible, retained only because it is the null the "
    "first version of this measurement shipped with), a permutation of the score labels "
    "over the mid-campaign pool, and the same permutation stratified so each drawn pole "
    "matches the real pole's per-generation composition. A positive verdict requires the "
    "live arm to clear both admissible nulls and the ablated arm to clear none. It would be "
    "a statement about this loop, this proposer, this problem and this seed; it is not a "
    "claim that the programs found are correct, novel, or good."
)


# ---------------------------------------------------------------------------
# 1. The feature alphabet: a closed, declared symbol table
# ---------------------------------------------------------------------------

#: Binary operators counted individually.  Anything else becomes ``op:other``.
_BINARY_OPS = ("Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow")
#: Unary operators.  Anything else becomes ``unary:other``.
_UNARY_OPS = ("USub", "UAdd", "Not", "Invert")
#: Comparison operators.  Anything else becomes ``cmp:other``.
_COMPARE_OPS = ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn")
#: Statement and expression node kinds carried as structure.  Anything else is not counted.
_NODE_KINDS = (
    "Assign",
    "AugAssign",
    "BoolOp",
    "Call",
    "Compare",
    "For",
    "If",
    "IfExp",
    "List",
    "Return",
    "Subscript",
    "Tuple",
    "While",
)
#: Identifiers counted individually: the mutation-bank variables plus the builtins the
#: sandbox allows.  Everything else becomes ``name:other``, so the alphabet stays closed
#: however the proposer names its temporaries.
_NAMES = (
    "abs",
    "float",
    "i",
    "int",
    "k",
    "len",
    "math",
    "max",
    "min",
    "n",
    "pow",
    "range",
    "sum",
    "u",
    "v",
    "x",
)
#: Attribute names counted individually, i.e. the ``math`` surface.  Else ``attr:other``.
_ATTRS = ("cos", "exp", "fabs", "floor", "log", "pow", "sin", "sqrt", "tanh")

FEATURE_ALPHABET: tuple[str, ...] = tuple(
    sorted(
        {f"op:{name}" for name in _BINARY_OPS}
        | {f"unary:{name}" for name in _UNARY_OPS}
        | {f"cmp:{name}" for name in _COMPARE_OPS}
        | {f"node:{name}" for name in _NODE_KINDS}
        | {f"name:{name}" for name in _NAMES}
        | {f"attr:{name}" for name in _ATTRS}
        | {
            "attr:other",
            "cmp:other",
            "const:bool",
            "const:float",
            "const:int",
            "const:other",
            "name:other",
            "op:other",
            "unary:other",
        }
    )
)

_INDEX = {symbol: position for position, symbol in enumerate(FEATURE_ALPHABET)}

#: A count vector aligned to :data:`FEATURE_ALPHABET`.
Vector = tuple[int, ...]

_ZERO: Vector = tuple(0 for _ in FEATURE_ALPHABET)


class LearningError(ValueError):
    """A measurement that cannot be made honestly is refused, never approximated."""


def _symbols_of(node: ast.AST) -> tuple[str, ...]:
    """Every alphabet symbol emitted by one AST node.  Total: never raises, never escapes."""

    if isinstance(node, ast.BinOp):
        name = type(node.op).__name__
        return (f"op:{name}" if name in _BINARY_OPS else "op:other",)
    if isinstance(node, ast.UnaryOp):
        name = type(node.op).__name__
        return (f"unary:{name}" if name in _UNARY_OPS else "unary:other",)
    if isinstance(node, ast.Compare):
        out = ["node:Compare"]
        for operator in node.ops:
            name = type(operator).__name__
            out.append(f"cmp:{name}" if name in _COMPARE_OPS else "cmp:other")
        return tuple(out)
    if isinstance(node, ast.Name):
        return (f"name:{node.id}" if node.id in _NAMES else "name:other",)
    if isinstance(node, ast.Attribute):
        return (f"attr:{node.attr}" if node.attr in _ATTRS else "attr:other",)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):  # bool before int: bool is a subclass of int
            return ("const:bool",)
        if isinstance(value, int):
            return ("const:int",)
        if isinstance(value, float):
            return ("const:float",)
        return ("const:other",)
    name = type(node).__name__
    return (f"node:{name}",) if name in _NODE_KINDS else ()


def program_features(source: str) -> Vector | None:
    """Integer counts over :data:`FEATURE_ALPHABET` for one program's syntax.

    Returns ``None`` for source that does not parse or that emits no declared symbol, so a
    caller can count those cases rather than silently folding them into a distribution.
    """

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    counts = [0] * len(FEATURE_ALPHABET)
    for node in ast.walk(tree):
        for symbol in _symbols_of(node):
            counts[_INDEX[symbol]] += 1
    return tuple(counts) if any(counts) else None


def pooled(vectors: Iterable[Vector]) -> Vector:
    """Elementwise integer sum.  The pooled distribution of a set of programs."""

    total = [0] * len(FEATURE_ALPHABET)
    for vector in vectors:
        for position, value in enumerate(vector):
            total[position] += value
    return tuple(total)


# ---------------------------------------------------------------------------
# 2. The statistic: normalised projection of the drift onto the score axis
# ---------------------------------------------------------------------------


def total_variation(left: Vector, right: Vector) -> Fraction:
    """Total variation distance between the distributions the two count vectors induce.

    ``TV(a/A, b/B) = (1/2) sum_s |a_s/A - b_s/B| = sum_s |a_s B - b_s A| / (2 A B)``.  The
    right-hand form is evaluated: integer cross-multiplication, one integer sum, one
    :class:`~fractions.Fraction`.  Reported alongside the statistic as the *separation* of
    the two poles, which is what the projection is normalised by.
    """

    a_total = sum(left)
    b_total = sum(right)
    if a_total <= 0 or b_total <= 0:
        raise LearningError("total variation is undefined for an empty count vector")
    numerator = sum(
        abs(a_value * b_total - b_value * a_total) for a_value, b_value in zip(left, right)
    )
    return Fraction(numerator, 2 * a_total * b_total)


def drift_numerator(start: Vector, end: Vector) -> tuple[list[int], int]:
    """``(d, S*E)`` with ``D_s = d_s / (S*E)``: the proposal drift, as integers."""

    s_total = sum(start)
    e_total = sum(end)
    if s_total <= 0 or e_total <= 0:
        raise LearningError("a window with no declared features has no distribution")
    return [
        end_value * s_total - start_value * e_total for start_value, end_value in zip(start, end)
    ], s_total * e_total


def axis_numerator(elite: Vector, foil: Vector) -> list[int]:
    """``a`` with ``A_s = a_s / (P*Q)``: the score axis, as integers.

    The ``P*Q`` never has to be formed: the statistic divides ``<D, A>`` by ``||A||_1`` and
    the two carry the same denominator, so it cancels exactly.
    """

    p_total = sum(elite)
    q_total = sum(foil)
    if p_total <= 0 or q_total <= 0:
        raise LearningError("a pole with no declared features has no distribution")
    return [
        elite_value * q_total - foil_value * p_total for elite_value, foil_value in zip(elite, foil)
    ]


def projection_from_parts(drift: Sequence[int], scale: int, axis: Sequence[int]) -> Fraction:
    """``T = sum_s d_s a_s / (scale * sum_s |a_s|)``.  One division, and it is rational."""

    weight = sum(abs(value) for value in axis)
    if weight == 0:
        raise LearningError("the two poles induce the same distribution: the axis is empty")
    return Fraction(
        sum(d * a for d, a in zip(drift, axis)),
        scale * weight,
    )


def projection(start: Vector, end: Vector, elite: Vector, foil: Vector) -> Fraction:
    """How far the proposal distribution moved along the elite-minus-foil axis.

    Positive means late proposals sit further toward the high-scoring pole, *relative to* the
    low-scoring pole of the same mid-campaign population, than early proposals did.  Anything
    that carries proposals along generically -- programs getting longer, the mutation grammar
    favouring a bank term, the population wandering -- is orthogonal to the score axis only
    on average, which is precisely why the nulls below randomise the axis and not the clock.
    """

    drift, scale = drift_numerator(start, end)
    return projection_from_parts(drift, scale, axis_numerator(elite, foil))


# ---------------------------------------------------------------------------
# 3. The pool, its poles, and the three nulls over them
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoolProgram:
    """One mid-campaign proposal: where it came from, what it scored, what it looks like."""

    generation: int
    program_sha256: str
    final_score: float
    features: Vector


def rank_pool(pool: Sequence[PoolProgram]) -> list[PoolProgram]:
    """Order by descending ``final_score``, ties broken by hash so the order is total.

    ``final_score`` is a float.  It is compared here and nowhere else; no arithmetic is done
    on it and none of it reaches the certificate.
    """

    return sorted(pool, key=lambda item: (-item.final_score, item.program_sha256))


def poles_of(pool: Sequence[PoolProgram], size: int) -> tuple[list[PoolProgram], list[PoolProgram]]:
    """The top ``size`` and bottom ``size`` of the ranked pool, which must be disjoint."""

    if size < 1:
        raise LearningError("a pole needs at least one program")
    if len(pool) < 2 * size:
        raise LearningError(
            f"the pool carries {len(pool)} programs and cannot hold two disjoint poles of {size}"
        )
    ranked = rank_pool(pool)
    return ranked[:size], ranked[-size:]


@dataclass(frozen=True, slots=True)
class NullResult:
    """One null: what it randomises, what it therefore tests, and the exact p-value."""

    null_id: str
    randomises: str
    admissible: bool
    statistic: Fraction
    draws: int
    at_least_as_extreme: int
    p_value: Fraction
    alpha: Fraction
    fires: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "null_id": self.null_id,
            "randomises": self.randomises,
            "admissible": self.admissible,
            "statistic": _rational(self.statistic),
            "statistic_decimal_not_certificate": format(float(self.statistic), ".9f"),
            "draws": self.draws,
            "at_least_as_extreme": self.at_least_as_extreme,
            "p_value": _rational(self.p_value),
            "p_value_decimal_not_certificate": format(float(self.p_value), ".9f"),
            "alpha": _rational(self.alpha),
            "fires": self.fires,
        }


def _rational(value: Fraction) -> str:
    """The certificate rendering: numerator over denominator, never a decimal."""

    return f"{value.numerator}/{value.denominator}"


def _seal(
    null_id: str,
    randomises: str,
    admissible: bool,
    observed: Fraction,
    drawn: Sequence[Fraction],
    alpha: Fraction,
) -> NullResult:
    hits = sum(1 for value in drawn if value >= observed)
    p_value = Fraction(hits + 1, len(drawn) + 1)
    return NullResult(
        null_id=null_id,
        randomises=randomises,
        admissible=admissible,
        statistic=observed,
        draws=len(drawn),
        at_least_as_extreme=hits,
        p_value=p_value,
        alpha=alpha,
        fires=bool(observed > 0 and p_value <= alpha),
    )


def generation_permutation_draws(
    start_groups: Sequence[Vector],
    end_groups: Sequence[Vector],
    axis: Sequence[int],
    *,
    draws: int,
    seed: int,
) -> list[Fraction]:
    """N1.  Relabel which generations are early and which are late; hold the axis fixed.

    Retained, and inadmissible.  It randomises the clock, so it tests whether the proposal
    distribution *drifted*; it never puts the score axis at risk, and a drifting proposer
    therefore clears it whatever axis it is handed.  :func:`null_calibration` measures how
    badly.
    """

    if not start_groups or not end_groups:
        raise LearningError("both windows must contain at least one generation")
    groups = [*start_groups, *end_groups]
    cut = len(start_groups)
    rng = random.Random(seed)
    out: list[Fraction] = []
    for _ in range(draws):
        order = list(groups)
        rng.shuffle(order)
        try:
            drift, scale = drift_numerator(pooled(order[:cut]), pooled(order[cut:]))
        except LearningError:  # a relabelling that emptied a window contributes nothing
            continue
        out.append(projection_from_parts(drift, scale, axis))
    if not out:
        raise LearningError("every generation relabelling emptied a window")
    return out


def _random_axes(
    pool: Sequence[PoolProgram], size: int, *, draws: int, seed: int
) -> list[list[int]]:
    """N2's axes: permute the score labels, then re-form top-``size`` and bottom-``size``.

    Permuting the labels and taking the two ends of the permuted ranking is the same thing as
    drawing two disjoint ``size``-subsets uniformly, which is what is done here.  Under the
    null that ``final_score`` says nothing about which features propagate, the observed axis
    is one draw from exactly this distribution.
    """

    if len(pool) < 2 * size:
        raise LearningError(
            f"a pool of {len(pool)} cannot supply two disjoint poles of {size}: the drawn "
            "poles would share programs and the axis would be shrunk toward zero"
        )
    ordered = sorted(pool, key=lambda item: item.program_sha256)
    rng = random.Random(seed)
    out: list[list[int]] = []
    for _ in range(draws):
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        elite = pooled(item.features for item in shuffled[:size])
        foil = pooled(item.features for item in shuffled[-size:])
        out.append(axis_numerator(elite, foil))
    return out


def _stratified_axes(
    pool: Sequence[PoolProgram],
    elite_items: Sequence[PoolProgram],
    foil_items: Sequence[PoolProgram],
    *,
    draws: int,
    seed: int,
) -> list[list[int]]:
    """N3's axes: as N2, but each drawn pole matches the real pole's generation profile.

    In a campaign that improves, high scores cluster late, so the real elite pole is
    temporally nearer the end window than a uniform draw would be -- and nearness in time can
    look exactly like alignment with score.  Here the score labels are permuted *within* each
    generation: a generation that contributed three programs to the elite pole and one to the
    foil pole contributes three and one to every drawn pair, so the drawn axes carry the same
    temporal profile as the observed one and differ only in which programs were called elite.
    """

    by_generation: dict[int, list[PoolProgram]] = {}
    for item in sorted(pool, key=lambda value: value.program_sha256):
        by_generation.setdefault(item.generation, []).append(item)
    elite_need = Counter(item.generation for item in elite_items)
    foil_need = Counter(item.generation for item in foil_items)
    for generation in set(elite_need) | set(foil_need):
        need = elite_need[generation] + foil_need[generation]
        available = len(by_generation.get(generation, ()))
        if available < need:
            raise LearningError(
                f"generation {generation} holds {available} pool programs but the observed "
                f"poles took {need} from it"
            )
    rng = random.Random(seed)
    out: list[list[int]] = []
    for _ in range(draws):
        elite_pick: list[Vector] = []
        foil_pick: list[Vector] = []
        for generation in sorted(set(elite_need) | set(foil_need)):
            candidates = list(by_generation[generation])
            rng.shuffle(candidates)
            take = elite_need[generation]
            drawn = candidates[: take + foil_need[generation]]
            elite_pick.extend(item.features for item in drawn[:take])
            foil_pick.extend(item.features for item in drawn[take:])
        out.append(axis_numerator(pooled(elite_pick), pooled(foil_pick)))
    return out


def _axis_draws(drift: Sequence[int], scale: int, axes: Sequence[Sequence[int]]) -> list[Fraction]:
    """The statistic under a set of drawn axes, with the observed windows held fixed."""

    out: list[Fraction] = []
    for axis in axes:
        try:
            out.append(projection_from_parts(drift, scale, axis))
        except LearningError:  # a drawn pair whose poles coincide carries no axis
            continue
    if not out:
        raise LearningError("every drawn axis was empty")
    return out


# ---------------------------------------------------------------------------
# 4. Measured false-positive rate of each null on this campaign's own data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """How often a null fires when the axis it is handed carries no score information."""

    null_id: str
    trials: int
    fired: int
    alpha: Fraction

    @property
    def rate(self) -> Fraction:
        return Fraction(self.fired, self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "null_id": self.null_id,
            "trials": self.trials,
            "fired": self.fired,
            "false_positive_rate": _rational(self.rate),
            "false_positive_rate_decimal_not_certificate": format(float(self.rate), ".9f"),
            "alpha": _rational(self.alpha),
        }


def null_calibration(
    start_groups: Sequence[Vector],
    end_groups: Sequence[Vector],
    pool: Sequence[PoolProgram],
    elite_items: Sequence[PoolProgram],
    foil_items: Sequence[PoolProgram],
    *,
    trials: int,
    draws: int,
    seed: int,
    alpha: Fraction,
) -> list[CalibrationResult]:
    """Hand each null a score-blind axis ``trials`` times and count how often it fires.

    The axis is drawn from the null's own reference distribution, so by construction it
    carries no information about ``final_score``; every firing is a false positive.  A
    well-behaved null fires at about ``alpha``.  N1 does not, and this is where that is
    measured rather than argued.

    The observed windows are held fixed throughout, so N1's reference draws depend only on
    the axis through a linear form and are computed once and reused.
    """

    drift, scale = drift_numerator(pooled(start_groups), pooled(end_groups))
    size = len(elite_items)
    axes_flat = _random_axes(pool, size, draws=trials, seed=seed ^ 0x5EED1)
    axes_strat = _stratified_axes(pool, elite_items, foil_items, draws=trials, seed=seed ^ 0x5EED2)

    groups = [*start_groups, *end_groups]
    cut = len(start_groups)
    rng = random.Random(seed ^ 0x5EED3)
    relabelled: list[tuple[list[int], int]] = []
    for _ in range(draws):
        order = list(groups)
        rng.shuffle(order)
        try:
            relabelled.append(drift_numerator(pooled(order[:cut]), pooled(order[cut:])))
        except LearningError:
            continue

    fired_n1 = 0
    for axis in axes_flat:
        try:
            observed = projection_from_parts(drift, scale, axis)
        except LearningError:
            continue
        drawn = [
            projection_from_parts(other_drift, other_scale, axis)
            for other_drift, other_scale in relabelled
        ]
        if _seal("n1", "", False, observed, drawn, alpha).fires:
            fired_n1 += 1

    fired_n2 = 0
    reference_flat = _random_axes(pool, size, draws=draws, seed=seed ^ 0x5EED4)
    for axis in axes_flat:
        try:
            observed = projection_from_parts(drift, scale, axis)
        except LearningError:
            continue
        if _seal("n2", "", True, observed, _axis_draws(drift, scale, reference_flat), alpha).fires:
            fired_n2 += 1

    fired_n3 = 0
    reference_strat = _stratified_axes(
        pool, elite_items, foil_items, draws=draws, seed=seed ^ 0x5EED5
    )
    for axis in axes_strat:
        try:
            observed = projection_from_parts(drift, scale, axis)
        except LearningError:
            continue
        if _seal("n3", "", True, observed, _axis_draws(drift, scale, reference_strat), alpha).fires:
            fired_n3 += 1

    return [
        CalibrationResult("n1_generation_permutation", trials, fired_n1, alpha),
        CalibrationResult("n2_pole_randomisation", trials, fired_n2, alpha),
        CalibrationResult("n3_stratified_pole_randomisation", trials, fired_n3, alpha),
    ]


# ---------------------------------------------------------------------------
# 5. Power: could this instrument have seen the thing it did not see?
# ---------------------------------------------------------------------------


def plant(
    program_groups: Sequence[Sequence[Vector]], donors: Sequence[Vector], count: int
) -> Vector:
    """Overwrite the leading ``count`` programs of each generation with pole programs.

    Donors are cycled in a fixed order and the replaced slots are the leading ones, so the
    construction is deterministic and replays exactly.  Returns the pooled window.
    """

    if count < 0:
        raise LearningError("a planted count cannot be negative")
    if not donors:
        raise LearningError("planting needs at least one donor program")
    out: list[Vector] = []
    cursor = 0
    for group in program_groups:
        keep = list(group)
        for position in range(min(count, len(keep))):
            keep[position] = donors[cursor % len(donors)]
            cursor += 1
        out.extend(keep)
    if not out:
        raise LearningError("the planted window carried no programs")
    return pooled(out)


@dataclass(frozen=True, slots=True)
class PowerResult:
    """The detection floor, and the whole ladder that established it.

    A negative result from a null that fires at no planted level is not a conservative
    measurement, it is a dead instrument, and "the search does not learn" said by a dead
    instrument is the uninformative null C1 exists to distinguish from a real one.  So the
    alternative hypothesis is *built* and fed to the same two admissible nulls: the leading
    ``count`` programs of each early generation are replaced by foil-pole programs and the
    leading ``count`` of each late generation by elite-pole programs.  At ``count = 0`` this
    is the observed campaign.  At the top rung the campaign has been replaced by a drift that
    runs exactly from the low-scoring pole to the high-scoring one -- learning by
    construction, and nothing else about the test changed, since the poles and therefore the
    axis and both null reference distributions are untouched.

    The smallest ``count`` at which both admissible nulls fire is the detection floor, in the
    units the campaign itself supplies: proposals per generation.
    """

    programs_per_generation: int
    ladder: tuple[tuple[int, Fraction, Fraction, Fraction, bool], ...]
    floor: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": (
                "early-window programs replaced by foil-pole programs and late-window "
                "programs by elite-pole programs, count per generation; poles, axis and "
                "both null reference distributions unchanged"
            ),
            "max_programs_per_generation": self.programs_per_generation,
            "detection_floor_planted_per_generation": self.floor,
            "detection_floor_found": self.floor is not None,
            "ladder": [
                {
                    "planted_per_generation": planted,
                    "statistic": _rational(statistic),
                    "p_n2": _rational(p_two),
                    "p_n3": _rational(p_three),
                    "both_admissible_nulls_fire": fires,
                }
                for planted, statistic, p_two, p_three, fires in self.ladder
            ],
        }


def detection_floor(
    start: WindowCapture,
    end: WindowCapture,
    elite: Sequence[Vector],
    foil: Sequence[Vector],
    axes_flat: Sequence[Sequence[int]],
    axes_stratified: Sequence[Sequence[int]],
    observed_axis: Sequence[int],
    *,
    alpha: Fraction,
) -> PowerResult:
    """Walk the planted ladder from the observed campaign to learning-by-construction.

    The drawn axes are handed in rather than redrawn: they depend only on the pool and the
    poles, neither of which planting touches, so every rung is compared against the same
    reference distribution the real campaign was compared against.  The walk stops at the
    first rung that fires, because the floor is what is wanted and the rungs above it cost
    ``draws`` projections each.
    """

    widest = max((len(group) for group in (*start.program_groups, *end.program_groups)), default=0)
    ladder: list[tuple[int, Fraction, Fraction, Fraction, bool]] = []
    floor: int | None = None
    for count in range(widest + 1):
        drift, scale = drift_numerator(
            plant(start.program_groups, list(foil), count),
            plant(end.program_groups, list(elite), count),
        )
        statistic = projection_from_parts(drift, scale, observed_axis)
        two = _seal("n2", "", True, statistic, _axis_draws(drift, scale, axes_flat), alpha)
        three = _seal("n3", "", True, statistic, _axis_draws(drift, scale, axes_stratified), alpha)
        fires = two.fires and three.fires
        ladder.append((count, statistic, two.p_value, three.p_value, fires))
        if fires and floor is None:
            floor = count
            break
    return PowerResult(programs_per_generation=widest, ladder=tuple(ladder), floor=floor)


# ---------------------------------------------------------------------------
# 6. Capturing a campaign
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowCapture:
    """One end of the campaign: which generations, how many programs, what they looked like."""

    label: str
    generations: tuple[int, ...]
    program_groups: tuple[tuple[Vector, ...], ...]
    programs: int
    unparsed: int

    @property
    def groups(self) -> tuple[Vector, ...]:
        """One pooled vector per generation: what N1 permutes."""

        return tuple(pooled(group) if group else _ZERO for group in self.program_groups)

    @property
    def counts(self) -> Vector:
        return pooled(self.groups)

    def to_dict(self) -> dict[str, Any]:
        counts = self.counts
        tokens = sum(counts)
        return {
            "label": self.label,
            "generations": list(self.generations),
            "programs_measured": self.programs,
            "programs_without_declared_features": self.unparsed,
            "feature_tokens": tokens,
            "distribution": {
                symbol: f"{counts[position]}/{tokens}"
                for position, symbol in enumerate(FEATURE_ALPHABET)
                if counts[position]
            },
        }


@dataclass(frozen=True, slots=True)
class ArmMeasurement:
    """One campaign arm, measured against all three nulls."""

    arm: str
    selection_pressure: bool
    problem_id: str
    seed: int
    generations_observed: int
    start: WindowCapture
    end: WindowCapture
    pool_size: int
    pole_size: int
    elite_mean_final_score: float
    foil_mean_final_score: float
    pole_separation: Fraction
    nulls: tuple[NullResult, ...]
    calibration: tuple[CalibrationResult, ...]
    power: PowerResult
    best_final_score: float

    @property
    def statistic(self) -> Fraction:
        return self.nulls[0].statistic

    def fires_under(self, null_id: str) -> bool:
        for null in self.nulls:
            if null.null_id == null_id:
                return null.fires
        raise LearningError(f"no null named {null_id}")

    @property
    def fires_under_every_admissible_null(self) -> bool:
        admissible = [null for null in self.nulls if null.admissible]
        return bool(admissible) and all(null.fires for null in admissible)

    @property
    def fires_under_any_admissible_null(self) -> bool:
        return any(null.fires for null in self.nulls if null.admissible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "selection_pressure": self.selection_pressure,
            "problem_id": self.problem_id,
            "seed": self.seed,
            "generations_observed": self.generations_observed,
            "window_start": self.start.to_dict(),
            "window_end": self.end.to_dict(),
            "statistic": _rational(self.statistic),
            "statistic_decimal_not_certificate": format(float(self.statistic), ".9f"),
            "poles": {
                "drawn_from": "mid-campaign proposals, disjoint from both windows",
                "pool_programs": self.pool_size,
                "pole_programs_each": self.pole_size,
                "elite_mean_final_score": format(self.elite_mean_final_score, ".9f"),
                "foil_mean_final_score": format(self.foil_mean_final_score, ".9f"),
                "separation_total_variation": _rational(self.pole_separation),
            },
            "nulls": [null.to_dict() for null in self.nulls],
            "null_calibration": [item.to_dict() for item in self.calibration],
            "power": self.power.to_dict(),
            "fires_under_every_admissible_null": self.fires_under_every_admissible_null,
            "best_final_score": format(self.best_final_score, ".9f"),
        }


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """Everything that has to be equal between the two arms except the one ablated field."""

    problem_id: str = "blinded_response_law"
    generations: int = 60
    islands: int = 4
    proposals_per_call: int = 5
    window: int = 10
    pole_size: int = 12
    seed: int = 7
    sweep_seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    draws: int = 999
    null_seed: int = 20260819
    calibration_trials: int = 100
    calibration_draws: int = 199
    alpha_numerator: int = 1
    alpha_denominator: int = 20
    sandbox_wall_seconds: float = 4.0

    @property
    def alpha(self) -> Fraction:
        return Fraction(self.alpha_numerator, self.alpha_denominator)

    def loop_config(self) -> fl.LoopConfig:
        return fl.LoopConfig(
            generations=self.generations,
            islands=self.islands,
            proposals_per_call=self.proposals_per_call,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "generations": self.generations,
            "islands": self.islands,
            "proposals_per_call": self.proposals_per_call,
            "window_generations_each_end": self.window,
            "pole_size": self.pole_size,
            "seed": self.seed,
            "sweep_seeds": list(self.sweep_seeds),
            "draws": self.draws,
            "null_seed": self.null_seed,
            "calibration_trials": self.calibration_trials,
            "calibration_draws": self.calibration_draws,
            "alpha": f"{self.alpha_numerator}/{self.alpha_denominator}",
            "sandbox_wall_seconds": format(self.sandbox_wall_seconds, ".3f"),
            "feature_alphabet_size": len(FEATURE_ALPHABET),
            "feature_alphabet_sha256": canonical_sha256(list(FEATURE_ALPHABET)),
        }


def _window_capture(label: str, records: Sequence[tuple[int, Sequence[Any]]]) -> WindowCapture:
    groups: list[tuple[Vector, ...]] = []
    programs = 0
    unparsed = 0
    for _, proposals in records:
        vectors: list[Vector] = []
        for item in proposals:
            features = program_features(item.source)
            if features is None:
                unparsed += 1
                continue
            vectors.append(features)
            programs += 1
        groups.append(tuple(vectors))
    return WindowCapture(
        label=label,
        generations=tuple(generation for generation, _ in records),
        program_groups=tuple(groups),
        programs=programs,
        unparsed=unparsed,
    )


def run_campaign(
    config: CampaignConfig, *, selection_pressure: bool, ledger_path: str | Path, corpus: Any = None
) -> tuple[list[tuple[int, tuple[Any, ...]]], float]:
    """Run one arm and return its productive generations plus the best score it reached.

    ``ledger_path`` is mandatory and must point at a scratch file.  This measurement makes
    thousands of proposal calls; charging them to the repository's declared LLM ledger would
    be a lie, because the proposer is :class:`funsearch_loop.MockMutationProposer` -- the free
    deterministic mutator -- and no model is called.
    """

    problem = fl.declared_problems()[config.problem_id]
    problem = fl.replace(
        problem, sandbox=fl.replace(problem.sandbox, wall_seconds=config.sandbox_wall_seconds)
    )
    loop = config.loop_config()
    records: list[Mapping[str, Any]] = []
    governor = fl.SpendGovernor(Path(ledger_path), loop.generations + 8, 10**9, 0)
    proposer = fl.MockMutationProposer(loop.seed, problem.mutation_bank)
    block = fl.run_problem(
        problem,
        loop,
        proposer,
        governor,
        corpus=corpus,
        selection_pressure=selection_pressure,
        observer=records.append,
    )
    productive = [
        (int(record["generation"]), tuple(record["proposed"]))
        for record in records
        if record["proposed"]
    ]
    headline = block["headline"]["best_by_final_score"]
    return productive, float(headline["final_score"]) if headline else 0.0


def measure_arm(
    config: CampaignConfig,
    *,
    selection_pressure: bool,
    ledger_path: str | Path,
    corpus: Any = None,
) -> ArmMeasurement:
    """Run one arm and test its proposal drift against all three nulls."""

    productive, best_final_score = run_campaign(
        config, selection_pressure=selection_pressure, ledger_path=ledger_path, corpus=corpus
    )
    return measure_records(
        config,
        productive,
        selection_pressure=selection_pressure,
        best_final_score=best_final_score,
    )


def measure_records(
    config: CampaignConfig,
    productive: Sequence[tuple[int, Sequence[Any]]],
    *,
    selection_pressure: bool,
    best_final_score: float,
) -> ArmMeasurement:
    """The measurement proper, separated from the run so it can be tested on fixtures."""

    if len(productive) < 2 * config.window + 1:
        raise LearningError(
            "a campaign needs more productive generations than two windows and a middle: "
            f"got {len(productive)}, need {2 * config.window + 1}"
        )
    start_records = list(productive[: config.window])
    end_records = list(productive[-config.window :])
    window_generations = {generation for generation, _ in start_records}
    window_generations |= {generation for generation, _ in end_records}

    pool: list[PoolProgram] = []
    for generation, proposals in productive:
        if generation in window_generations:
            continue
        for item in proposals:
            features = program_features(item.source)
            if features is None:
                continue
            pool.append(
                PoolProgram(
                    generation=generation,
                    program_sha256=item.program_sha256,
                    final_score=float(item.final_score),
                    features=features,
                )
            )
    elite_items, foil_items = poles_of(pool, config.pole_size)
    elite = pooled(item.features for item in elite_items)
    foil = pooled(item.features for item in foil_items)

    start = _window_capture("campaign_start", start_records)
    end = _window_capture("campaign_end", end_records)
    drift, scale = drift_numerator(start.counts, end.counts)
    axis = axis_numerator(elite, foil)
    observed = projection_from_parts(drift, scale, axis)
    alpha = config.alpha

    axes_flat = _random_axes(
        pool, config.pole_size, draws=config.draws, seed=config.null_seed ^ 0xA1
    )
    axes_stratified = _stratified_axes(
        pool, elite_items, foil_items, draws=config.draws, seed=config.null_seed ^ 0xA2
    )
    nulls = (
        _seal(
            "n1_generation_permutation",
            "which generations count as early and which as late, holding the axis fixed",
            False,
            observed,
            generation_permutation_draws(
                start.groups, end.groups, axis, draws=config.draws, seed=config.null_seed
            ),
            alpha,
        ),
        _seal(
            "n2_pole_randomisation",
            "the final_score labels over the mid-campaign pool, holding the windows fixed",
            True,
            observed,
            _axis_draws(drift, scale, axes_flat),
            alpha,
        ),
        _seal(
            "n3_stratified_pole_randomisation",
            "the final_score labels within each generation, so the drawn poles carry the "
            "observed poles' temporal profile",
            True,
            observed,
            _axis_draws(drift, scale, axes_stratified),
            alpha,
        ),
    )
    power = detection_floor(
        start,
        end,
        [item.features for item in elite_items],
        [item.features for item in foil_items],
        axes_flat,
        axes_stratified,
        axis,
        alpha=alpha,
    )
    calibration = null_calibration(
        start.groups,
        end.groups,
        pool,
        elite_items,
        foil_items,
        trials=config.calibration_trials,
        draws=config.calibration_draws,
        seed=config.null_seed ^ 0xC0,
        alpha=alpha,
    )

    return ArmMeasurement(
        arm="selection_on" if selection_pressure else "selection_off_ablation",
        selection_pressure=selection_pressure,
        problem_id=config.problem_id,
        seed=config.seed,
        generations_observed=len(productive),
        start=start,
        end=end,
        pool_size=len(pool),
        pole_size=config.pole_size,
        elite_mean_final_score=sum(item.final_score for item in elite_items) / len(elite_items),
        foil_mean_final_score=sum(item.final_score for item in foil_items) / len(foil_items),
        pole_separation=total_variation(elite, foil),
        nulls=nulls,
        calibration=tuple(calibration),
        power=power,
        best_final_score=best_final_score,
    )


# ---------------------------------------------------------------------------
# 6. The report
# ---------------------------------------------------------------------------


def verdict_of(live: ArmMeasurement, ablated: ArmMeasurement) -> str:
    """The four outcomes, in the order that decides them.

    ``uninformative_null`` is the one that matters and the one v1 had no way to reach.  The
    goals document is explicit that "zero survivors and no reachability certificate" is not a
    result but an admission that the engine cannot tell whether it is working, and the same
    logic applies to a statistical null: a campaign whose instrument does not fire even on a
    fully planted score-aligned drift has not measured the absence of learning, it has failed
    to measure anything.  Reporting that as ``no_measurable_shift`` would be the same mistake
    v1 made, pointed the other way.
    """

    if ablated.fires_under_any_admissible_null:
        return "statistic_is_measuring_drift"
    if live.fires_under_every_admissible_null:
        return "learned"
    if live.power.floor is None:
        return "uninformative_null"
    return "no_measurable_shift"


VERDICT_RULE = (
    "learned requires a positive statistic clearing every admissible null on the "
    "selection_on arm and no admissible null firing on the ablated arm; the "
    "n1_generation_permutation null is inadmissible and cannot support a positive verdict; "
    "a live arm that clears no admissible null and has no detection floor is "
    "uninformative_null, not no_measurable_shift"
)


def sweep(
    config: CampaignConfig, *, scratch_ledger: str | Path | None = None
) -> dict[int, tuple[ArmMeasurement, ArmMeasurement]]:
    """Both arms of one campaign per declared seed.

    A single seed decides nothing here in either direction.  A test at alpha ``1/20`` fires
    on about one ablated campaign in twenty by construction, and a live campaign that fails
    to clear the threshold is a statement about that campaign.  Only the rate over seeds is
    a statement about the loop, so the receipt carries the rate.
    """

    if config.seed not in config.sweep_seeds:
        raise LearningError(
            f"the headline seed {config.seed} is not in the declared sweep "
            f"{list(config.sweep_seeds)}: the headline would not be one of the rows"
        )
    out: dict[int, tuple[ArmMeasurement, ArmMeasurement]] = {}
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch_ledger) if scratch_ledger is not None else Path(scratch)
        root.mkdir(parents=True, exist_ok=True)
        for seed in config.sweep_seeds:
            per_seed = replace(config, seed=seed)
            out[seed] = (
                measure_arm(
                    per_seed,
                    selection_pressure=True,
                    ledger_path=root / f"on-{seed}.json",
                ),
                measure_arm(
                    per_seed,
                    selection_pressure=False,
                    ledger_path=root / f"off-{seed}.json",
                ),
            )
    return out


def learning_report(
    config: CampaignConfig | None = None, *, scratch_ledger: str | Path | None = None
) -> dict[str, Any]:
    """Run the whole sweep, test every arm against all three nulls, and seal the comparison."""

    settings = config or CampaignConfig()
    measured = sweep(settings, scratch_ledger=scratch_ledger)
    live, ablated = measured[settings.seed]
    return seal_report(settings, live, ablated, measured)


def sweep_block(
    measured: Mapping[int, tuple[ArmMeasurement, ArmMeasurement]],
) -> dict[str, Any]:
    """Per-campaign rows and the fire rate of each null in each arm."""

    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {
        "selection_on": {"campaigns": 0, "power_qualified": 0, "n1": 0, "n2": 0, "n3": 0},
        "selection_off_ablation": {
            "campaigns": 0,
            "power_qualified": 0,
            "n1": 0,
            "n2": 0,
            "n3": 0,
        },
    }
    for seed in sorted(measured):
        for arm in measured[seed]:
            fires = {
                "n1": arm.nulls[0].fires,
                "n2": arm.nulls[1].fires,
                "n3": arm.nulls[2].fires,
            }
            rows.append(
                {
                    "problem_id": arm.problem_id,
                    "seed": seed,
                    "selection_pressure": arm.selection_pressure,
                    "arm": arm.arm,
                    "power_qualified": arm.power.floor is not None,
                    "statistic": _rational(arm.statistic),
                    "p_values": {
                        "n1": _rational(arm.nulls[0].p_value),
                        "n2": _rational(arm.nulls[1].p_value),
                        "n3": _rational(arm.nulls[2].p_value),
                    },
                    "fires": fires,
                    "detection_floor_planted_per_generation": arm.power.floor,
                    "null_false_positive_rates": {
                        "n1": _rational(arm.calibration[0].rate),
                        "n2": _rational(arm.calibration[1].rate),
                        "n3": _rational(arm.calibration[2].rate),
                    },
                }
            )
            bucket = totals[arm.arm]
            bucket["campaigns"] += 1
            bucket["power_qualified"] += int(arm.power.floor is not None)
            for key, value in fires.items():
                bucket[key] += int(value)
    verdicts = [{"seed": seed, "verdict": verdict_of(*measured[seed])} for seed in sorted(measured)]
    counts = {
        name: sum(1 for item in verdicts if item["verdict"] == name)
        for name in (
            "learned",
            "no_measurable_shift",
            "uninformative_null",
            "statistic_is_measuring_drift",
        )
    }
    return {
        "campaigns": len(rows),
        "rows": rows,
        "totals": totals,
        "verdicts": verdicts,
        "verdict_counts": counts,
    }


def seal_report(
    settings: CampaignConfig,
    live: ArmMeasurement,
    ablated: ArmMeasurement,
    measured: Mapping[int, tuple[ArmMeasurement, ArmMeasurement]],
) -> dict[str, Any]:
    """The receipt body plus its content hash.  Never hand-edited: regenerate it."""

    verdict = verdict_of(live, ablated)
    seeds = sweep_block(measured)
    totals = seeds["totals"]
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "lane": "funsearch-learning-measurement",
        "supersedes": {
            "schema_version": "invariant-funsearch-learning-measurement-1.0",
            "receipt": "runs/math/funsearch/learning-v1.json",
            "reason": (
                "its only null permuted generations while holding the score axis fixed, so "
                "it tested whether the proposal distribution drifted and never put the "
                "score axis at risk; that null is retained here as n1 and marked "
                "inadmissible, and its measured false-positive rate on a score-blind axis "
                "is sealed in every arm's null_calibration block"
            ),
        },
        "claims": CLAIMS,
        "finding": FINDING,
        "config": settings.to_dict(),
        "config_sha256": canonical_sha256(settings.to_dict()),
        "ablation": {
            "field": "selection_pressure",
            "removed_paths": [
                "example sampling weighted by final_score becomes uniform",
                "island truncation by final_score becomes truncation by program_sha256",
                "the periodic island reset that reseeds weak islands from strong is skipped",
            ],
            "unchanged": [
                "the problem, its evaluator and its sealed answer",
                "the sandbox and its budget",
                "the proposer, its seed and its mutation grammar",
                "scoring itself, which still runs and is still sealed",
            ],
        },
        "statistic": {
            "name": "normalised projection of proposal drift onto the score axis",
            "definition": "T = <p(end) - p(start), p(elite) - p(foil)> / ||p(elite) - p(foil)||_1",
            "evaluated_as": "sum_s d_s a_s / (S*E*sum_s |a_s|) with d_s = end_s*S - start_s*E "
            "and a_s = elite_s*Q - foil_s*P",
            "bound": "|T| <= max_s |p(end)_s - p(start)_s| <= 1",
            "float_on_certificate_path": False,
        },
        "arms": [live.to_dict(), ablated.to_dict()],
        "seed_sweep": seeds,
        "verdict": verdict,
        "verdict_rule": VERDICT_RULE,
        "headline": {
            "verdict": verdict,
            "selection_on_statistic": _rational(live.statistic),
            "selection_on_p_n1_inadmissible": _rational(live.nulls[0].p_value),
            "selection_on_p_n2": _rational(live.nulls[1].p_value),
            "selection_on_p_n3": _rational(live.nulls[2].p_value),
            "ablation_statistic": _rational(ablated.statistic),
            "ablation_p_n1_inadmissible": _rational(ablated.nulls[0].p_value),
            "ablation_p_n2": _rational(ablated.nulls[1].p_value),
            "ablation_p_n3": _rational(ablated.nulls[2].p_value),
            "selection_on_best_final_score": format(live.best_final_score, ".9f"),
            "ablation_best_final_score": format(ablated.best_final_score, ".9f"),
            "selection_on_detection_floor": live.power.floor,
            "ablation_detection_floor": ablated.power.floor,
            "sweep_n3_fires_selection_on": (
                f"{totals['selection_on']['n3']}/{totals['selection_on']['campaigns']}"
            ),
            "sweep_n2_fires_selection_on": (
                f"{totals['selection_on']['n2']}/{totals['selection_on']['campaigns']}"
            ),
            "sweep_n1_fires_selection_on_inadmissible": (
                f"{totals['selection_on']['n1']}/{totals['selection_on']['campaigns']}"
            ),
            "sweep_campaigns_with_a_detection_floor_selection_on": (
                f"{totals['selection_on']['power_qualified']}/{totals['selection_on']['campaigns']}"
            ),
            "sweep_verdict_counts": seeds["verdict_counts"],
            "sweep_n2_fires_ablated": (
                f"{totals['selection_off_ablation']['n2']}"
                f"/{totals['selection_off_ablation']['campaigns']}"
            ),
            "sweep_n3_fires_ablated": (
                f"{totals['selection_off_ablation']['n3']}"
                f"/{totals['selection_off_ablation']['campaigns']}"
            ),
        },
        "scope": SCOPE,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _parse_rational(text: str) -> Fraction:
    numerator, _, denominator = text.partition("/")
    return Fraction(int(numerator), int(denominator))


def validate_report(value: Mapping[str, Any]) -> None:
    """Refuse a report whose seal, claims, verdict rule or exact arithmetic has moved."""

    if value.get("schema_version") != RECEIPT_SCHEMA:
        raise LearningError("report schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise LearningError("report seal changed")
    if value.get("claims") != CLAIMS:
        raise LearningError("claims block changed")
    if value.get("finding") != FINDING:
        raise LearningError("the finding text does not match the module that produced it")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise LearningError("config binding changed")
    if value.get("verdict_rule") != VERDICT_RULE:
        raise LearningError("verdict rule changed")
    arms = value.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise LearningError("a report must carry exactly two arms")
    if [arm["selection_pressure"] for arm in arms] != [True, False]:
        raise LearningError("the two arms must be the live run and its ablation")
    expected_ids = [
        "n1_generation_permutation",
        "n2_pole_randomisation",
        "n3_stratified_pole_randomisation",
    ]
    for arm in arms:
        nulls = arm["nulls"]
        if [null["null_id"] for null in nulls] != expected_ids:
            raise LearningError("an arm does not carry the three declared nulls")
        if [null["admissible"] for null in nulls] != [False, True, True]:
            raise LearningError("the admissibility of the nulls changed")
        statistic = _parse_rational(arm["statistic"])
        for null in nulls:
            if _parse_rational(null["statistic"]) != statistic:
                raise LearningError("the three nulls do not test the same statistic")
            expected_p = Fraction(null["at_least_as_extreme"] + 1, null["draws"] + 1)
            if _parse_rational(null["p_value"]) != expected_p:
                raise LearningError("a p-value does not equal (hits + 1) / (draws + 1)")
            alpha = _parse_rational(null["alpha"])
            if null["fires"] != bool(statistic > 0 and expected_p <= alpha):
                raise LearningError("a fires flag disagrees with its own p-value")
        admissible = [null for null in nulls if null["admissible"]]
        if arm["fires_under_every_admissible_null"] != all(null["fires"] for null in admissible):
            raise LearningError("an arm's admissible-null summary disagrees with its nulls")
        calibration = arm["null_calibration"]
        if [item["null_id"] for item in calibration] != [
            "n1_generation_permutation",
            "n2_pole_randomisation",
            "n3_stratified_pole_randomisation",
        ]:
            raise LearningError("a calibration block does not cover the three nulls")
        for item in calibration:
            if _parse_rational(item["false_positive_rate"]) != Fraction(
                item["fired"], item["trials"]
            ):
                raise LearningError("a false-positive rate is not fired/trials")
        power = arm["power"]
        ladder = power["ladder"]
        if not ladder or ladder[0]["planted_per_generation"] != 0:
            raise LearningError("a power ladder must start from the unplanted campaign")
        if [step["planted_per_generation"] for step in ladder] != list(range(len(ladder))):
            raise LearningError("a power ladder must step one planted program at a time")
        if _parse_rational(ladder[0]["statistic"]) != statistic:
            raise LearningError(
                "the unplanted rung of the power ladder is not the observed statistic"
            )
        # Rung zero is the observed campaign against the same drawn axes the arm's own
        # admissible nulls used, so the two must agree exactly.  If they do not, the ladder
        # was computed against a different reference distribution and proves nothing about
        # this arm.
        for key, index in (("p_n2", 1), ("p_n3", 2)):
            if _parse_rational(ladder[0][key]) != _parse_rational(nulls[index]["p_value"]):
                raise LearningError(
                    "the unplanted rung disagrees with the arm's own admissible null"
                )
        for step in ladder:
            expected_fires = (
                all(
                    _parse_rational(step[key]) <= _parse_rational(nulls[index]["alpha"])
                    for key, index in (("p_n2", 1), ("p_n3", 2))
                )
                and _parse_rational(step["statistic"]) > 0
            )
            if step["both_admissible_nulls_fire"] != expected_fires:
                raise LearningError("a power rung disagrees with its own p-values")
        firing = [
            step["planted_per_generation"] for step in ladder if step["both_admissible_nulls_fire"]
        ]
        expected_floor = firing[0] if firing else None
        if power["detection_floor_planted_per_generation"] != expected_floor:
            raise LearningError("the detection floor is not the first firing rung")
        if power["detection_floor_found"] != (expected_floor is not None):
            raise LearningError("the detection-floor flag disagrees with the ladder")
        if expected_floor == 0 and not arm["fires_under_every_admissible_null"]:
            raise LearningError(
                "the unplanted rung fires but the arm does not: the ladder and the nulls "
                "disagree about the same campaign"
            )
    alpha = _parse_rational(arms[0]["nulls"][1]["alpha"])
    sweep_value = value.get("seed_sweep")
    if not isinstance(sweep_value, dict):
        raise LearningError("a report must carry its seed sweep")
    rows = sweep_value["rows"]
    if sweep_value["campaigns"] != len(rows):
        raise LearningError("the sweep campaign count does not match its rows")
    if len(rows) < 2:
        raise LearningError("a sweep of fewer than two campaigns is not a sweep")
    recomputed: dict[str, dict[str, int]] = {
        "selection_on": {"campaigns": 0, "power_qualified": 0, "n1": 0, "n2": 0, "n3": 0},
        "selection_off_ablation": {
            "campaigns": 0,
            "power_qualified": 0,
            "n1": 0,
            "n2": 0,
            "n3": 0,
        },
    }
    for row in rows:
        bucket = recomputed[row["arm"]]
        bucket["campaigns"] += 1
        if row["power_qualified"] != (row["detection_floor_planted_per_generation"] is not None):
            raise LearningError("a sweep row's power flag disagrees with its detection floor")
        bucket["power_qualified"] += int(row["power_qualified"])
        for key in ("n1", "n2", "n3"):
            bucket[key] += int(row["fires"][key])
        for key in ("n1", "n2", "n3"):
            declared = _parse_rational(row["p_values"][key])
            expected_fire = bool(_parse_rational(row["statistic"]) > 0 and declared <= alpha)
            if row["fires"][key] != expected_fire:
                raise LearningError("a sweep row's fire flag disagrees with its p-value")
    if sweep_value["totals"] != recomputed:
        raise LearningError("the sweep totals are not the tally of its own rows")
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    for item in sweep_value["verdicts"]:
        pair = by_seed[item["seed"]]
        live_row = pair["selection_on"]
        ablated_row = pair["selection_off_ablation"]
        if ablated_row["fires"]["n2"] or ablated_row["fires"]["n3"]:
            expected_seed_verdict = "statistic_is_measuring_drift"
        elif live_row["fires"]["n2"] and live_row["fires"]["n3"]:
            expected_seed_verdict = "learned"
        elif not live_row["power_qualified"]:
            expected_seed_verdict = "uninformative_null"
        else:
            expected_seed_verdict = "no_measurable_shift"
        if item["verdict"] != expected_seed_verdict:
            raise LearningError("a sweep verdict does not follow from that seed's two arms")
    tallied = {
        name: sum(1 for item in sweep_value["verdicts"] if item["verdict"] == name)
        for name in sweep_value["verdict_counts"]
    }
    if sweep_value["verdict_counts"] != tallied:
        raise LearningError("the sweep verdict counts are not the tally of its own verdicts")
    headline_seed = value["config"]["seed"]
    for arm_name, arm in zip(("selection_on", "selection_off_ablation"), arms, strict=True):
        match = [row for row in rows if row["seed"] == headline_seed and row["arm"] == arm_name]
        if len(match) != 1:
            raise LearningError("the headline arm is not exactly one row of the sweep")
        if match[0]["statistic"] != arm["statistic"]:
            raise LearningError("the headline arm disagrees with its own sweep row")

    live, ablated = arms
    if any(null["fires"] for null in ablated["nulls"] if null["admissible"]):
        expected = "statistic_is_measuring_drift"
    elif live["fires_under_every_admissible_null"]:
        expected = "learned"
    elif not live["power"]["detection_floor_found"]:
        expected = "uninformative_null"
    else:
        expected = "no_measurable_shift"
    if value.get("verdict") != expected:
        raise LearningError("the verdict does not follow from the two arms")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    # ``CampaignConfig`` uses slots, so the class attributes are descriptors rather than
    # values; the defaults have to come from an instance.
    fallback = CampaignConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problem", default=fallback.problem_id)
    parser.add_argument("--generations", type=int, default=fallback.generations)
    parser.add_argument("--window", type=int, default=fallback.window)
    parser.add_argument("--seed", type=int, default=fallback.seed)
    parser.add_argument("--draws", type=int, default=fallback.draws)
    parser.add_argument("--calibration-trials", type=int, default=fallback.calibration_trials)
    parser.add_argument(
        "--sweep-seeds",
        default=",".join(str(value) for value in fallback.sweep_seeds),
        help="comma-separated proposer seeds; the headline seed must be one of them",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = learning_report(
        CampaignConfig(
            problem_id=args.problem,
            generations=args.generations,
            window=args.window,
            seed=args.seed,
            sweep_seeds=tuple(int(value) for value in args.sweep_seeds.split(",") if value),
            draws=args.draws,
            calibration_trials=args.calibration_trials,
        )
    )
    validate_report(report)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" explicitly: the receipt's bytes must not depend on the platform
        # that produced them, and Path.write_text would translate on Windows.
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    print(json.dumps(report["headline"], indent=2, sort_keys=True))
    print(f"verdict: {report['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
