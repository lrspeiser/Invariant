"""Model-guided ORDERING of a declared-framework lattice sweep.  Ordering only, never elimination.

The problem
-----------
``configs/tensor_constraint_search.json`` declares a framework -- ``field_content``, the connection,
``tensor_rank``, ``tensor_symmetry``, ``metric_signature``, ``curvature``,
``concomitant_generator`` -- and then, per search, a ``dimension``, a ``derivative_order`` and a
constraint list.  Every one of those is handed to the engine, and the engine derives only what they
force.  Making the declaration searchable means sweeping a *lattice* of such declarations instead of
running the five hand-written cells the config happens to name.  That lattice is large and each cell
costs a modular-jet rank computation, so the order the cells are visited in is worth money.

What this module is and is not
------------------------------
It is a **scheduler**.  It asks a proposer -- the FunSearch proposer of
:mod:`sigma_theory_compiler.funsearch_loop`, charged through that module's
:class:`~sigma_theory_compiler.funsearch_loop.SpendGovernor` -- for a *priority function* over
declared framework fields, fits that function against the dimensions already measured, and visits
the remaining cells in the order it induces.

It is **not** a prune.  The soundness rule of this search outranks speed: a prune that can discard a
non-empty cell is a bug, not an optimisation.  Two prunes are sound -- monotonicity of the dimension
along the constraint sublattice, and the one-sided sampling bound that makes a cheap low-sample zero
a *proof* of zero -- and neither of them is model-guided.  A model may reorder the sweep.  It may
never eliminate a cell, because nothing a model says is evidence about a nullspace.

That restriction is not a comment here, it is a checked invariant.  :func:`sweep` re-derives the
multiset of pending cell identifiers from whatever the scheduler hands back and raises
:class:`OrderChangedTheSet` if a single cell was dropped, duplicated or invented.  A scheduler that
tries to prune therefore *cannot* complete a sweep, and
:class:`EliminatingScheduler` is carried in this module purely so the invariant has something that
must fail.

The cell interface
------------------
A cell is a dict of framework fields plus a cost estimate -- :class:`LatticeCell`, constructed from
and serialised back to exactly that mapping.  Its identity is the canonical SHA-256 of the framework
fields *alone*: the cost estimate is a scheduling hint and must not be able to change what a cell
is.  Two schedulers handed the same lattice therefore agree on cell identity by construction, which
is what makes the order-invariance control a statement about results rather than about labels.

The synthetic lattice
---------------------
The real lattice belongs to the lattice sweep.  So this module ships a *declared synthetic lattice*
-- :func:`synthetic_lattice` -- of 96 cells over three spacetime dimensions, two derivative orders,
four field contents and the four nested constraint prefixes of the declared constraint chain, with a
dimension oracle that is deliberately built to have the two properties the real search has:

* non-increasing as constraints are added, and
* non-decreasing in field content.

:func:`assert_lattice_monotone` measures both off the oracle rather than trusting them, and
:func:`non_monotone_oracle` is the variant that must fail that measurement.

What the ablation reports
-------------------------
:func:`ablation` runs the same lattice under guided, fixed lexicographic, cost-ascending and thirty
one seeded random orders, and reports cells-to-first-hit and cells-to-half-the-hits for each.  The
verdict is whatever comes out.  If guidance does not beat random the receipt says so; a scheduler
that is no better than shuffling is a thing worth knowing before anyone builds on it.

Exactness
---------
Nothing on a certificate path is a float.  Priority scores come back from the sandbox as ``.17g``
decimal text and are lifted to :class:`~fractions.Fraction` before they are ever compared.  That
lift does not recover the exact binary double -- seventeen significant digits round-trip a double
uniquely but do not equal it -- and it does not need to: an injective map composed with a monotone
rounding is strictly monotone, so the order induced on the rationals is exactly the order of the
doubles, and rationals compare without a tolerance, a NaN or a platform-dependent tie.
:func:`priority_scores` states that argument where it is used.  Retrospective fitness is an integer
count.  Every rational in the receipt is emitted as ``"p/q"`` text, because
:func:`~sigma_theory_compiler.sigma_core.canonical_sha256` rejects floats outright.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from .funsearch_loop import (
    NoveltyVerdict,
    ProposalCall,
    SandboxBudget,
    SandboxOutcome,
    ScoredProgram,
    SpendGovernor,
    guard_prompt,
    run_in_sandbox,
)
from .sigma_core import canonical_sha256

#: Schemas.  Bump only with a receipt-shape change.
RESULT_SCHEMA = "invariant-guided-lattice-order-result-1.0"
CELL_SCHEMA = "invariant-guided-lattice-cell-1.0"

#: Repository-relative paths this module binds to itself.
SOURCE_PATH = "src/sigma_theory_compiler/guided_lattice_order.py"
TEST_PATH = "tests/test_guided_lattice_order.py"

#: Claims block.  Frozen; any change changes the receipt hash and therefore the claim.  Note what
#: is absent: whether guidance beats random is a MEASUREMENT and appears in the ablation block, not
#: here.  A claim is a thing the module promises; a measurement is a thing it reports.
CLAIMS: dict[str, bool] = {
    "guidance_reorders_and_never_eliminates": True,
    "order_cannot_change_the_result_set": True,
    "cell_identity_excludes_the_cost_estimate": True,
    "priority_scores_are_exact_rationals": True,
    "proposal_calls_are_charged_to_the_spend_governor": True,
    "guidance_quality_is_measured_not_claimed": True,
}


class GuidedOrderError(ValueError):
    """Base class for the typed refusals of this module."""


class OrderChangedTheSet(GuidedOrderError):
    """A scheduler returned something that was not a permutation of the pending cells.

    This is the soundness failure the whole module exists to make impossible.  Dropping a cell is
    an unsound prune; duplicating one double-charges the evaluator and can make an order-dependent
    evaluator look consistent; inventing one evaluates a framework nobody declared.  All three are
    the same bug and all three are refused here.
    """


class LatticeDeclarationError(GuidedOrderError):
    """A cell was declared with a missing, extra or ill-typed framework field."""


# ---------------------------------------------------------------------------
# 1. The cell: a dict of framework fields plus a cost estimate.
# ---------------------------------------------------------------------------

#: The framework fields that make up a cell.  These are exactly the things
#: ``configs/tensor_constraint_search.json`` DECLARES and hands to the engine: the framework block's
#: ``field_content``, ``tensor_rank``, ``tensor_symmetry``, ``metric_signature``, ``curvature`` and
#: ``concomitant_generator``, plus the per-search ``dimension``, ``derivative_order`` and
#: ``constraints``.  The Levi-Civita connection is not a separate field: it is fixed by
#: ``concomitant_generator``, which is where the config hides it inside the ``geometry`` prose.
CELL_FRAMEWORK_FIELDS: tuple[str, ...] = (
    "concomitant_generator",
    "constraints",
    "curvature",
    "derivative_order",
    "dimension",
    "field_content",
    "metric_signature",
    "tensor_rank",
    "tensor_symmetry",
)

#: The full cell mapping is the framework fields plus the cost estimate.
CELL_FIELDS: tuple[str, ...] = tuple(sorted(CELL_FRAMEWORK_FIELDS + ("cost_estimate",)))

_SEQUENCE_FIELDS = frozenset({"constraints", "field_content"})
_INT_FIELDS = frozenset({"derivative_order", "dimension", "tensor_rank"})
_TEXT_FIELDS = frozenset(
    {"concomitant_generator", "curvature", "metric_signature", "tensor_symmetry"}
)


@dataclass(frozen=True, slots=True)
class LatticeCell:
    """One point of the declared-framework lattice.

    Identity is the canonical SHA-256 of the framework fields alone.  ``cost_estimate`` is excluded
    on purpose: it is a scheduling hint, and a hint that could change a cell's identity would let a
    re-estimate silently turn one cell into two and break the order-invariance control.
    """

    dimension: int
    derivative_order: int
    field_content: tuple[str, ...]
    constraints: tuple[str, ...]
    tensor_rank: int
    tensor_symmetry: str
    metric_signature: str
    curvature: str
    concomitant_generator: str
    cost_estimate: int

    def __post_init__(self) -> None:
        for name in _INT_FIELDS | {"cost_estimate"}:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise LatticeDeclarationError(f"cell field {name!r} must be a positive int")
        for name in _TEXT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise LatticeDeclarationError(f"cell field {name!r} must be a non-empty string")
        for name in _SEQUENCE_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value:
                raise LatticeDeclarationError(f"cell field {name!r} must be a non-empty tuple")
            if any(not isinstance(item, str) or not item for item in value):
                raise LatticeDeclarationError(f"cell field {name!r} must hold non-empty strings")
            if tuple(sorted(set(value))) != value:
                raise LatticeDeclarationError(
                    f"cell field {name!r} must be sorted and free of duplicates so that two "
                    "spellings of one declaration cannot become two cells"
                )

    # -- the dict interface -----------------------------------------------------------

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LatticeCell:
        """Build a cell from a plain dict of framework fields plus ``cost_estimate``.

        Nothing is coerced.  ``int(True)`` is ``1`` and ``str(None)`` is ``"None"``, so a coercing
        constructor would launder an ill-typed declaration into a well-typed cell and the validation
        in ``__post_init__`` would never see it.  The only conversion is list to tuple, which JSON
        forces and which cannot hide a type error.
        """

        missing = sorted(set(CELL_FIELDS) - set(value))
        extra = sorted(set(value) - set(CELL_FIELDS))
        if missing or extra:
            raise LatticeDeclarationError(
                f"cell mapping must carry exactly {list(CELL_FIELDS)}; missing {missing}, "
                f"unexpected {extra}"
            )
        for name in _SEQUENCE_FIELDS:
            if isinstance(value[name], (str, bytes)) or not isinstance(value[name], Sequence):
                raise LatticeDeclarationError(f"cell field {name!r} must be a sequence of strings")
        return cls(
            dimension=value["dimension"],
            derivative_order=value["derivative_order"],
            field_content=tuple(value["field_content"]),
            constraints=tuple(value["constraints"]),
            tensor_rank=value["tensor_rank"],
            tensor_symmetry=value["tensor_symmetry"],
            metric_signature=value["metric_signature"],
            curvature=value["curvature"],
            concomitant_generator=value["concomitant_generator"],
            cost_estimate=value["cost_estimate"],
        )

    def framework(self) -> dict[str, Any]:
        """The declared framework fields, canonical-JSON ready, WITHOUT the cost estimate."""

        return {
            "concomitant_generator": self.concomitant_generator,
            "constraints": list(self.constraints),
            "curvature": self.curvature,
            "derivative_order": self.derivative_order,
            "dimension": self.dimension,
            "field_content": list(self.field_content),
            "metric_signature": self.metric_signature,
            "tensor_rank": self.tensor_rank,
            "tensor_symmetry": self.tensor_symmetry,
        }

    def to_mapping(self) -> dict[str, Any]:
        """The full cell: framework fields plus the cost estimate."""

        return {**self.framework(), "cost_estimate": self.cost_estimate}

    @property
    def cell_id(self) -> str:
        return canonical_sha256({"schema_version": CELL_SCHEMA, "framework": self.framework()})

    @property
    def label(self) -> str:
        """A short human handle.  Never used for identity or ordering -- only for reading."""

        codes = "".join(sorted(name[0] for name in self.field_content))
        return f"d{self.dimension}-k{self.derivative_order}-{codes}-c{len(self.constraints)}"

    def sort_key(self) -> tuple[Any, ...]:
        """The FIXED lexicographic key.  Declared here so the control arm is reproducible."""

        return (
            self.dimension,
            self.derivative_order,
            self.field_content,
            self.constraints,
            self.cell_id,
        )


def canonical_lattice(cells: Sequence[LatticeCell]) -> tuple[LatticeCell, ...]:
    """The declared lattice in lexicographic order, refusing duplicate cell identities."""

    ordered = tuple(sorted(cells, key=LatticeCell.sort_key))
    seen = [cell.cell_id for cell in ordered]
    if len(set(seen)) != len(seen):
        raise LatticeDeclarationError("lattice carries two cells with the same framework")
    return ordered


# ---------------------------------------------------------------------------
# 2. Features: what a priority function is allowed to see.
# ---------------------------------------------------------------------------

#: Every feature is a function of the DECLARATION.  Nothing measured, nothing from the oracle: a
#: priority function that could see the answer would be scoring itself.
FEATURE_NAMES: tuple[str, ...] = (
    "dimension",
    "derivative_order",
    "field_count",
    "constraint_count",
    "cost_estimate",
    "vector_field",
    "scalar_field",
    "divergence_free",
)

PRIORITY_ENTRY = "priority"


def feature_row(cell: LatticeCell) -> tuple[float, ...]:
    """The declared feature vector of a cell, in :data:`FEATURE_NAMES` order."""

    return (
        float(cell.dimension),
        float(cell.derivative_order),
        float(len(cell.field_content)),
        float(len(cell.constraints)),
        float(cell.cost_estimate),
        1.0 if any("vector" in name for name in cell.field_content) else 0.0,
        1.0 if any("scalar" in name for name in cell.field_content) else 0.0,
        1.0 if "divergence_free" in cell.constraints else 0.0,
    )


def priority_signature() -> str:
    return f"def {PRIORITY_ENTRY}({', '.join(FEATURE_NAMES)}) -> float"


def priority_source(body: str) -> str:
    """A priority function with ``body`` as its return statement.  The declared source shape."""

    head = f"def {PRIORITY_ENTRY}(\n    " + ",\n    ".join(FEATURE_NAMES) + ",\n):"
    return f"{head}\n    {body}\n"


#: The seed priority function is deliberately UNINFORMATIVE.  A hand-written prior -- "prefer more
#: field content, prefer fewer constraints" -- would make the guided arm win for a reason that has
#: nothing to do with the proposer, and the ablation would be measuring the author's taste.  Every
#: bit of structure the guided arm has, it got from the proposer fitted against measured dimensions.
SEED_PRIORITY_SOURCE = priority_source("return 0.0")

#: Expression fragments the deterministic mock proposer may splice in.  This is the mutation
#: grammar's vocabulary, and it is the declared features plus small constants -- nothing else, so a
#: mutant can only ever be a function of the declaration.
MUTATION_BANK: tuple[str, ...] = FEATURE_NAMES + ("1.0", "2.0", "0.5")

#: The execution envelope for a priority function.  ``math`` is not on the allowlist: a priority
#: function needs arithmetic on eight numbers and nothing else, and a narrower jail is a better one.
PRIORITY_BUDGET = SandboxBudget(wall_seconds=4.0, import_allowlist=())

#: Tokens that must never reach a proposer prompt.  These name the SYNTHETIC lattice's internal
#: generating rule.  The proposer is entitled to see the declaration and the dimensions already
#: measured -- that is the scheduler's own information -- and entitled to see nothing about the rule
#: that produced them, or the ablation would be measuring a leak.
FORBIDDEN_GUIDANCE_VOCABULARY: tuple[str, ...] = (
    "oracle",
    "supply",
    "demand",
    "synthetic",
    "weights",
)

_GUIDANCE_INSTRUCTION = (
    "You are given a scoring function over the declared fields of a search cell, and a table of "
    "cells that have already been evaluated together with the integer each one measured. Write ONE "
    "new implementation of the same signature that would give a strictly larger score to the cells "
    "that measured greater than zero than to the cells that measured zero. Return only Python "
    "source for a single function. Use no imports, no input or output, and no attribute names "
    "beginning with an underscore. The score is used only to ORDER the remaining cells; it never "
    "decides whether a cell is evaluated, so a confident wrong answer costs ordering and nothing "
    "else."
)


def build_ordering_prompt(
    evaluated: Sequence[tuple[LatticeCell, int]],
    examples: Sequence[ScoredProgram],
) -> str:
    """Assemble the guidance prompt, then refuse it if a forbidden term got in."""

    lines = [
        _GUIDANCE_INSTRUCTION,
        "",
        f"signature: {priority_signature()}",
        f"features, in order: {', '.join(FEATURE_NAMES)}",
        "",
        "# cells already evaluated, as (feature row) -> measured value:",
    ]
    for cell, measured in evaluated:
        row = ", ".join(format(value, ".6g") for value in feature_row(cell))
        lines.append(f"#   ({row})  ->  {measured}")
    lines.append("")
    for index, example in enumerate(examples):
        lines.append(f"# example {index} placed {format(example.final_score, '.0f')} of them first")
        lines.append(example.source)
        lines.append("")
    lines.append("# your implementation:")
    return guard_prompt("\n".join(lines), FORBIDDEN_GUIDANCE_VOCABULARY)


# ---------------------------------------------------------------------------
# 3. Running a priority function, exactly.
# ---------------------------------------------------------------------------


def priority_scores(
    source: str,
    cells: Sequence[LatticeCell],
    budget: SandboxBudget = PRIORITY_BUDGET,
) -> tuple[SandboxOutcome, dict[str, Fraction] | None]:
    """Execute one priority function over ``cells`` and lift its outputs to exact rationals.

    The sandbox emits ``.17g`` decimal text.  That is NOT the exact binary value of the double the
    child computed -- ``0.4`` prints as ``0.40000000000000002``, and the rational built from that
    decimal differs from ``Fraction(0.4)``.  What seventeen significant digits *does* guarantee is
    that the map from double to decimal is injective, and rounding to a fixed number of significant
    digits is monotone; an injective monotone map is strictly monotone, so the order this function
    induces on the rationals is exactly the order of the doubles the child produced.  Ordering is
    the only thing these numbers are ever used for, and the rationals compare without a tolerance,
    without a NaN and without a platform-dependent tie, which the doubles would not.
    """

    if not cells:
        return SandboxOutcome(True, ()), {}
    outcome = run_in_sandbox(source, PRIORITY_ENTRY, [feature_row(c) for c in cells], budget)
    if not outcome.ok or len(outcome.outputs) != len(cells):
        if outcome.ok:
            outcome = SandboxOutcome(False, reason="wrong_output_shape", detail="row count")
        return outcome, None
    try:
        values = [Fraction(Decimal(text)) for text in outcome.outputs]
    except (ArithmeticError, ValueError):  # pragma: no cover - the child already rejects these
        return SandboxOutcome(False, reason="output_unparseable", detail="not decimal"), None
    return outcome, {cell.cell_id: value for cell, value in zip(cells, values, strict=True)}


def order_by_scores(
    cells: Sequence[LatticeCell], scores: Mapping[str, Fraction]
) -> tuple[LatticeCell, ...]:
    """Descending score, ties broken by cell identity.  Total, exact and reproducible."""

    return tuple(sorted(cells, key=lambda cell: (-scores[cell.cell_id], cell.cell_id)))


def retrospective_fitness(scores: Mapping[str, Fraction], measured: Mapping[str, int]) -> int:
    """How many of the already-measured non-zero cells this function puts in the leading block.

    An integer, so it is exact and totally ordered.  The leading block is as long as there are
    non-zero cells, which makes a perfect score equal to the number of them and makes the metric
    insensitive to how a function ranks the cells it got wrong -- the sweep only ever consumes a
    prefix, so only the prefix is scored.
    """

    hits = [cell_id for cell_id, value in measured.items() if value > 0]
    if not hits:
        return 0
    ranked = sorted(measured, key=lambda cell_id: (-scores[cell_id], cell_id))
    leading = set(ranked[: len(hits)])
    return sum(1 for cell_id in hits if cell_id in leading)


def as_example(source: str, fitness: int) -> ScoredProgram:
    """Wrap a priority function as the ``ScoredProgram`` the FunSearch proposers consume."""

    return ScoredProgram(
        program_sha256=canonical_sha256({"source": source}),
        source=source,
        origin="priority",
        generation=0,
        island=0,
        sandbox=SandboxOutcome(True, ()),
        outputs=(),
        quality=float(fitness),
        quality_detail={"retrospective_fitness": fitness},
        novelty=NoveltyVerdict(1.0, "not_screened", {}),
        final_score=float(fitness),
    )


# ---------------------------------------------------------------------------
# 4. Schedulers.  Every one of them returns a PERMUTATION of the pending cells.
# ---------------------------------------------------------------------------


class Scheduler(Protocol):
    """Ordering only.  ``order`` must return a permutation of ``pending``; :func:`sweep` checks."""

    scheduler_id: str

    def order(
        self, pending: Sequence[LatticeCell], measured: Mapping[str, int]
    ) -> Sequence[LatticeCell]: ...


@dataclass
class LexicographicScheduler:
    """The fixed control arm: the declared lexicographic key, ignoring everything measured."""

    scheduler_id: str = "lexicographic"

    def order(
        self, pending: Sequence[LatticeCell], measured: Mapping[str, int]
    ) -> Sequence[LatticeCell]:
        return sorted(pending, key=LatticeCell.sort_key)

    def diagnostics(self) -> dict[str, Any]:
        return {"scheduler_id": self.scheduler_id, "uses_measurements": False}


@dataclass
class CostScheduler:
    """The cheapest-first arm: the obvious engineering baseline, and still not a model."""

    scheduler_id: str = "cost_ascending"

    def order(
        self, pending: Sequence[LatticeCell], measured: Mapping[str, int]
    ) -> Sequence[LatticeCell]:
        return sorted(pending, key=lambda cell: (cell.cost_estimate, cell.cell_id))

    def diagnostics(self) -> dict[str, Any]:
        return {"scheduler_id": self.scheduler_id, "uses_measurements": False}


@dataclass
class RandomScheduler:
    """The shuffle arm.  Seeded, so a random order is as reproducible as a fixed one."""

    seed: int
    scheduler_id: str = "random"

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def order(
        self, pending: Sequence[LatticeCell], measured: Mapping[str, int]
    ) -> Sequence[LatticeCell]:
        ordered = sorted(pending, key=LatticeCell.sort_key)
        self._random.shuffle(ordered)
        return ordered

    def diagnostics(self) -> dict[str, Any]:
        return {"scheduler_id": self.scheduler_id, "seed": self.seed, "uses_measurements": False}


@dataclass
class EliminatingScheduler:
    """A scheduler that PRUNES, carried here so the soundness invariant has something to catch.

    It is exactly the optimisation the rules forbid: score the pending cells and drop the tail.  On
    this lattice it would even look like it worked -- the discarded cells are mostly zero.  It never
    completes a sweep, because :func:`sweep` refuses a returned multiset that is not the pending
    one, and a control that cannot fail is worse than none.
    """

    keep_fraction: Fraction = Fraction(1, 2)
    scheduler_id: str = "eliminating_control"

    def order(
        self, pending: Sequence[LatticeCell], measured: Mapping[str, int]
    ) -> Sequence[LatticeCell]:
        ordered = sorted(pending, key=LatticeCell.sort_key)
        keep = max(1, int(len(ordered) * self.keep_fraction))
        return ordered[:keep]


@dataclass
class DuplicatingScheduler:
    """The other half of the invariant's teeth: a cell offered twice."""

    scheduler_id: str = "duplicating_control"

    def order(
        self, pending: Sequence[LatticeCell], measured: Mapping[str, int]
    ) -> Sequence[LatticeCell]:
        ordered = sorted(pending, key=LatticeCell.sort_key)
        return [ordered[0], *ordered] if len(ordered) > 1 else ordered


@dataclass
class GuidedScheduler:
    """Ask the proposer for a priority function, fit it to what has been measured, reorder.

    The loop is FunSearch's, narrowed to one output: the population is priority functions, the
    fitness is :func:`retrospective_fitness` against the cells already evaluated, and the champion
    is what orders the pending set.  Every proposal call is charged to the
    :class:`~sigma_theory_compiler.funsearch_loop.SpendGovernor` first, so a campaign that runs out
    of budget degrades to its last champion and then to lexicographic -- it never degrades to
    skipping cells, because skipping is not in the interface.
    """

    proposer: Any
    governor: SpendGovernor
    warmup_cells: int = 12
    refit_period: int = 8
    proposals_per_call: int = 6
    example_count: int = 3
    budget: SandboxBudget = PRIORITY_BUDGET
    scheduler_id: str = "guided"

    def __post_init__(self) -> None:
        if self.warmup_cells < 1 or self.refit_period < 1 or self.proposals_per_call < 1:
            raise GuidedOrderError("guided scheduler cadence must be positive")
        self._lattice: tuple[LatticeCell, ...] = ()
        self._champion: str = SEED_PRIORITY_SOURCE
        self._champion_scores: dict[str, Fraction] = {}
        self._bank: list[tuple[int, str]] = []
        self._calls: list[ProposalCall] = []
        self._refits: list[dict[str, Any]] = []
        self._sandbox_failures: dict[str, int] = {}
        self._next_refit: int = self.warmup_cells
        self._fallbacks: int = 0

    # -- lifecycle ---------------------------------------------------------------------

    def begin(self, lattice: Sequence[LatticeCell]) -> None:
        """Bind the declared lattice.  The champion is scored over it once per champion change."""

        self._lattice = tuple(lattice)
        self._rescore_champion()

    def _rescore_champion(self) -> None:
        outcome, scores = priority_scores(self._champion, self._lattice, self.budget)
        if scores is None:
            self._note_failure(outcome)
            self._champion_scores = {}
            return
        self._champion_scores = scores

    def _note_failure(self, outcome: SandboxOutcome) -> None:
        reason = outcome.reason or "unknown"
        self._sandbox_failures[reason] = self._sandbox_failures.get(reason, 0) + 1

    # -- the ordering itself -----------------------------------------------------------

    def order(
        self, pending: Sequence[LatticeCell], measured: Mapping[str, int]
    ) -> Sequence[LatticeCell]:
        if len(measured) < self.warmup_cells:
            # Declared warm-up: with nothing measured there is nothing to fit, so the guided arm
            # runs the SAME fixed order as the control arm.  Charging the warm-up to guidance is
            # the honest accounting -- a scheduler does not get to start its clock late.
            return sorted(pending, key=LatticeCell.sort_key)
        if len(measured) >= self._next_refit:
            self._next_refit = len(measured) + self.refit_period
            self._refit(measured)
        if not all(cell.cell_id in self._champion_scores for cell in pending):
            self._fallbacks += 1
            return sorted(pending, key=LatticeCell.sort_key)
        return order_by_scores(pending, self._champion_scores)

    # -- one governed proposal call ----------------------------------------------------

    def _refit(self, measured: Mapping[str, int]) -> None:
        by_id = {cell.cell_id: cell for cell in self._lattice}
        evaluated = [(by_id[cell_id], value) for cell_id, value in sorted(measured.items())]
        incumbent = (
            retrospective_fitness(self._champion_scores, measured)
            if all(cell_id in self._champion_scores for cell_id in measured)
            else -1
        )
        record: dict[str, Any] = {
            "evaluated_cells": len(measured),
            "incumbent_fitness": incumbent,
            "non_zero_cells": sum(1 for value in measured.values() if value > 0),
            "proposals": 0,
            "usable_proposals": 0,
            "champion_replaced": False,
        }
        if not self.governor.may_call():
            record["halt_reason"] = self.governor.halt_reason
            self._refits.append(record)
            return
        examples = [as_example(self._champion, max(incumbent, 0))]
        for fitness, source in sorted(self._bank, reverse=True)[: self.example_count - 1]:
            examples.append(as_example(source, fitness))
        prompt = build_ordering_prompt(evaluated, examples)
        self.governor.charge()
        call = self.proposer.propose(prompt, examples, self.proposals_per_call)
        self._calls.append(call)
        if not call.ok:
            # Degrade, never eliminate.  A dead proposer costs ordering quality and nothing else.
            record["proposer_failure"] = call.reason
            self._refits.append(record)
            return
        best_fitness, best_source, best_scores = incumbent, None, None
        for source in self.proposer.programs():
            record["proposals"] += 1
            outcome, scores = priority_scores(source, self._lattice, self.budget)
            if scores is None:
                self._note_failure(outcome)
                continue
            record["usable_proposals"] += 1
            fitness = retrospective_fitness(scores, measured)
            self._bank.append((fitness, source))
            if fitness > best_fitness:
                best_fitness, best_source, best_scores = fitness, source, scores
        if best_source is not None and best_scores is not None:
            # Strictly greater only.  A tie keeps the incumbent, which keeps the champion path
            # deterministic and stops a run from drifting between equally good functions.
            self._champion, self._champion_scores = best_source, best_scores
            record["champion_replaced"] = True
        self._bank = sorted(self._bank, reverse=True)[: 4 * self.example_count]
        record["champion_fitness"] = best_fitness
        self._refits.append(record)

    # -- what the receipt records ------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "scheduler_id": self.scheduler_id,
            "uses_measurements": True,
            "proposer_id": getattr(self.proposer, "proposer_id", "unknown"),
            "warmup_cells": self.warmup_cells,
            "refit_period": self.refit_period,
            "proposals_per_call": self.proposals_per_call,
            "seed_priority_sha256": canonical_sha256({"source": SEED_PRIORITY_SOURCE}),
            "champion_priority_sha256": canonical_sha256({"source": self._champion}),
            "champion_is_still_the_seed": self._champion == SEED_PRIORITY_SOURCE,
            "champion_source": self._champion,
            "refits": list(self._refits),
            "proposal_calls": [call.to_dict() for call in self._calls],
            "lexicographic_fallbacks": self._fallbacks,
            "sandbox_failures": dict(sorted(self._sandbox_failures.items())),
            "spend_governor": self.governor.to_dict(),
        }


# ---------------------------------------------------------------------------
# 5. The sweep, and the invariant that makes ordering safe.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One complete pass over the lattice: what was visited, in what order, and what it measured."""

    scheduler_id: str
    visit_order: tuple[str, ...]
    results: dict[str, int]
    evaluations: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def results_sha256(self) -> str:
        """The ORDER-FREE digest of the sweep: cell identity to measured value, sorted."""

        return canonical_sha256(
            {
                "schema_version": RESULT_SCHEMA,
                "results": {cell_id: value for cell_id, value in sorted(self.results.items())},
            }
        )

    def order_sha256(self) -> str:
        """The order-SENSITIVE digest.  Two arms agreeing here would mean guidance did nothing."""

        return canonical_sha256({"visit_order": list(self.visit_order)})

    def hits(self) -> int:
        return sum(1 for value in self.results.values() if value > 0)


def _check_permutation(
    returned: Sequence[LatticeCell], pending: Sequence[LatticeCell], scheduler_id: str
) -> None:
    """The soundness gate.  Ordering only means: same multiset out as in, or the sweep stops."""

    got = sorted(cell.cell_id for cell in returned)
    want = sorted(cell.cell_id for cell in pending)
    if got == want:
        return
    dropped = sorted(set(want) - set(got))
    invented = sorted(set(got) - set(want))
    duplicated = len(got) != len(set(got))
    raise OrderChangedTheSet(
        f"scheduler {scheduler_id!r} returned {len(got)} cells for {len(want)} pending: "
        f"{len(dropped)} dropped, {len(invented)} invented, duplicates={duplicated}.  A model may "
        "reorder this sweep and may never eliminate a cell from it -- nothing a model says is "
        "evidence about a nullspace, and a prune that can discard a non-empty cell is a bug."
    )


def sweep(
    lattice: Sequence[LatticeCell],
    scheduler: Any,
    evaluate: Callable[[LatticeCell], int],
) -> SweepResult:
    """Visit every cell of ``lattice`` exactly once, in whatever order ``scheduler`` asks for."""

    cells = canonical_lattice(lattice)
    begin = getattr(scheduler, "begin", None)
    if callable(begin):
        begin(cells)
    scheduler_id = getattr(scheduler, "scheduler_id", type(scheduler).__name__)
    pending: list[LatticeCell] = list(cells)
    visit_order: list[str] = []
    results: dict[str, int] = {}
    while pending:
        proposed = list(scheduler.order(tuple(pending), dict(results)))
        _check_permutation(proposed, pending, scheduler_id)
        cell = proposed[0]
        value = evaluate(cell)
        if not isinstance(value, int) or isinstance(value, bool):
            raise GuidedOrderError("an evaluator must return an int dimension")
        results[cell.cell_id] = value
        visit_order.append(cell.cell_id)
        pending = proposed[1:]
    if len(visit_order) != len(cells) or set(visit_order) != {cell.cell_id for cell in cells}:
        raise OrderChangedTheSet(  # pragma: no cover - defended twice on purpose
            f"scheduler {scheduler_id!r} completed a sweep that did not cover the lattice"
        )
    diagnostics = getattr(scheduler, "diagnostics", None)
    return SweepResult(
        scheduler_id=scheduler_id,
        visit_order=tuple(visit_order),
        results=results,
        evaluations=len(visit_order),
        diagnostics=diagnostics() if callable(diagnostics) else {},
    )


class CountingEvaluator:
    """An evaluator wrapper that counts calls and refuses to answer the same cell twice.

    The refusal matters: a scheduler that re-offered a cell would otherwise show up as extra cost
    rather than as the correctness failure it is, and the order-invariance control would be
    comparing two sweeps that did different amounts of work.
    """

    def __init__(self, inner: Callable[[LatticeCell], int]) -> None:
        self._inner = inner
        self.calls = 0
        self._seen: set[str] = set()

    def __call__(self, cell: LatticeCell) -> int:
        if cell.cell_id in self._seen:
            raise OrderChangedTheSet(f"cell {cell.label} was offered for evaluation twice")
        self._seen.add(cell.cell_id)
        self.calls += 1
        return self._inner(cell)


# ---------------------------------------------------------------------------
# 6. The declared synthetic lattice.
# ---------------------------------------------------------------------------

SYNTHETIC_LATTICE_ID = "guided-order-synthetic-v1"

#: The constraint CHAIN, declared in the order ``configs/tensor_constraint_search.json`` declares
#: it.  A cell's constraint set is a prefix of this chain, which is what makes the constraint axis a
#: chain in the sublattice rather than an unordered pile.
SYNTHETIC_CONSTRAINT_CHAIN: tuple[str, ...] = (
    "generally_covariant",
    "derivative_order",
    "symmetric",
    "divergence_free",
    "newtonian_limit",
)

#: The field contents swept.  Every one contains the metric, because a search with no metric has no
#: connection and the enumeration is not defined -- the same refusal the real module makes.
SYNTHETIC_FIELD_CONTENTS: tuple[tuple[str, ...], ...] = (
    ("metric",),
    ("metric", "scalar"),
    ("metric", "unit_timelike_vector"),
    ("metric", "scalar", "unit_timelike_vector"),
)

SYNTHETIC_DIMENSIONS: tuple[int, ...] = (4, 5, 6)
SYNTHETIC_DERIVATIVE_ORDERS: tuple[int, ...] = (2, 4)
SYNTHETIC_CONSTRAINT_DEPTHS: tuple[int, ...] = (2, 3, 4, 5)

#: How much coefficient space each declared field brings.  Non-negative, which is what makes the
#: oracle non-decreasing in field content.
_FIELD_WEIGHT: dict[str, int] = {"metric": 1, "scalar": 2, "unit_timelike_vector": 4}

#: How much each constraint takes away.  Non-negative, which is what makes the oracle non-increasing
#: as constraints are added.
_CONSTRAINT_COST: dict[str, int] = {
    "generally_covariant": 2,
    "derivative_order": 3,
    "symmetric": 3,
    "divergence_free": 6,
    "newtonian_limit": 1,
}


def synthetic_cost(dimension: int, derivative_order: int, field_count: int) -> int:
    """A plausible cost estimate: jets grow with dimension, order and field content."""

    return dimension * derivative_order * derivative_order * (3**field_count)


def synthetic_lattice() -> tuple[LatticeCell, ...]:
    """The declared 96-cell stand-in lattice, in canonical order."""

    cells: list[LatticeCell] = []
    for dimension in SYNTHETIC_DIMENSIONS:
        for order in SYNTHETIC_DERIVATIVE_ORDERS:
            for content in SYNTHETIC_FIELD_CONTENTS:
                for depth in SYNTHETIC_CONSTRAINT_DEPTHS:
                    cells.append(
                        LatticeCell.from_mapping(
                            {
                                "dimension": dimension,
                                "derivative_order": order,
                                "field_content": tuple(sorted(content)),
                                "constraints": tuple(sorted(SYNTHETIC_CONSTRAINT_CHAIN[:depth])),
                                "tensor_rank": 2,
                                "tensor_symmetry": "symmetric",
                                "metric_signature": "lorentzian",
                                "curvature": "generic",
                                "concomitant_generator": "riemann_tensor",
                                "cost_estimate": synthetic_cost(dimension, order, len(content)),
                            }
                        )
                    )
    return canonical_lattice(cells)


def synthetic_dimension(cell: LatticeCell) -> int:
    """The stand-in for ``published_dimension(run_search(...))``.

    Built to carry the two structural facts the real search has and nothing else: the surviving
    space is non-decreasing in field content and non-increasing as constraints are added.  It is NOT
    a model of Lovelock's theorem and no number it returns means anything about gravity.
    """

    available = sum(_FIELD_WEIGHT[name] for name in cell.field_content)
    available *= cell.derivative_order // 2
    available += cell.dimension - min(SYNTHETIC_DIMENSIONS)
    removed = sum(_CONSTRAINT_COST[name] for name in cell.constraints)
    return max(0, available - removed)


def non_monotone_oracle(cell: LatticeCell) -> int:
    """The variant that must FAIL :func:`assert_lattice_monotone`.

    It rewards ``divergence_free`` instead of charging for it, so the constraint axis stops being
    non-increasing.  Carried so the monotonicity measurement has something it can catch; a check
    that no input can fail is not a check.
    """

    base = synthetic_dimension(cell)
    return base + 9 if "divergence_free" in cell.constraints else base


def assert_lattice_monotone(
    lattice: Sequence[LatticeCell], evaluate: Callable[[LatticeCell], int]
) -> dict[str, Any]:
    """Measure both monotonicity directions off ``evaluate``; raise on the first violation.

    This is the property the two SOUND prunes rest on.  It is measured here rather than asserted so
    that a stand-in lattice cannot quietly stop being a faithful stand-in.
    """

    index = {
        (cell.dimension, cell.derivative_order, cell.field_content, cell.constraints): cell
        for cell in lattice
    }
    # Every STRICT SUBSET pair, not just the adjacent one.  Comparing a cell only against the
    # constraint set with its last element removed checks a single edge of the chain -- and because
    # a constraint set is stored sorted rather than in chain order, the removed element is not even
    # reliably the last constraint declared.  On this lattice that version measured 24 of the 144
    # comparisons and let a deliberately non-monotone oracle through.
    constraint_pairs = 0
    for key, cell in index.items():
        for other_key, weaker in index.items():
            if key[:3] != other_key[:3] or not set(other_key[3]) < set(key[3]):
                continue
            constraint_pairs += 1
            if evaluate(cell) > evaluate(weaker):
                raise GuidedOrderError(
                    f"monotonicity violated on the constraint axis at {cell.label}: adding "
                    f"{sorted(set(key[3]) - set(other_key[3]))} raised the dimension from "
                    f"{evaluate(weaker)} to {evaluate(cell)}"
                )
    content_pairs = 0
    for key, cell in index.items():
        for other_key, other in index.items():
            if key[:2] != other_key[:2] or key[3] != other_key[3]:
                continue
            if not set(key[2]) < set(other_key[2]):
                continue
            content_pairs += 1
            if evaluate(other) < evaluate(cell):
                raise GuidedOrderError(
                    f"monotonicity violated on the field-content axis: {other.label} carries a "
                    f"superset of the fields of {cell.label} but measured less"
                )
    return {
        "cells": len(lattice),
        "constraint_comparisons": constraint_pairs,
        "field_content_comparisons": content_pairs,
        "constraint_axis_non_increasing": True,
        "field_content_axis_non_decreasing": True,
    }


# ---------------------------------------------------------------------------
# 7. The ablation.
# ---------------------------------------------------------------------------

#: Odd on purpose, so the median over the random arm is an integer and no rounding rule is needed.
RANDOM_SEEDS: tuple[int, ...] = tuple(range(1, 32))


def cells_to_hit_count(result: SweepResult, target: int) -> int | None:
    """One-based evaluation index at which the ``target``-th non-zero cell was reached."""

    if target < 1:
        raise GuidedOrderError("hit target must be at least one")
    seen = 0
    for index, cell_id in enumerate(result.visit_order, start=1):
        if result.results[cell_id] > 0:
            seen += 1
            if seen >= target:
                return index
    return None


def hits_within(result: SweepResult, prefix: int) -> int:
    """Non-zero cells found in the first ``prefix`` evaluations.

    Precision-at-k, and the one metric here that is symmetric across arms at a FIXED budget.  The
    two headline metrics both measure how long an arm took to reach a hit count, which lets a long
    warm-up dominate them; this one asks the question the other way round and is worth reading
    beside them.
    """

    return sum(1 for cell_id in result.visit_order[:prefix] if result.results[cell_id] > 0)


def sweep_metrics(result: SweepResult, *, quarter: int | None = None) -> dict[str, Any]:
    total = result.hits()
    half = -(-total // 2)  # ceil, in integers
    prefix = quarter if quarter is not None else -(-result.evaluations // 4)
    return {
        "scheduler_id": result.scheduler_id,
        "evaluations": result.evaluations,
        "non_zero_cells": total,
        "cells_to_first_hit": cells_to_hit_count(result, 1) if total else None,
        "cells_to_half_the_hits": cells_to_hit_count(result, half) if total else None,
        "half_target": half,
        "quarter_budget": prefix,
        "hits_in_the_first_quarter": hits_within(result, prefix),
    }


def _fraction_text(value: Fraction) -> str:
    """Exact rational as text.  ``canonical_sha256`` rejects floats; this never produces one."""

    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _spread(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:  # pragma: no cover - the seed list is non-empty by construction
        raise GuidedOrderError("cannot summarise an empty arm")
    median = Fraction(ordered[count // 2]) if count % 2 else Fraction(
        ordered[count // 2 - 1] + ordered[count // 2], 2
    )
    return {
        "runs": count,
        "minimum": ordered[0],
        "median": _fraction_text(median),
        "mean": _fraction_text(Fraction(sum(ordered), count)),
        "maximum": ordered[-1],
    }


def _beats(guided: int | None, others: Sequence[int | None]) -> dict[str, Any]:
    """Strictly-fewer-evaluations comparison, reported as a count rather than a verdict."""

    usable = [value for value in others if value is not None]
    if guided is None or not usable:
        return {"comparable_runs": len(usable), "guided_strictly_better": None}
    wins = sum(1 for value in usable if guided < value)
    return {
        "comparable_runs": len(usable),
        "guided_strictly_better_in": wins,
        "guided_strictly_worse_in": sum(1 for value in usable if guided > value),
        "guided_beats_the_majority": bool(2 * wins > len(usable)),
    }


@dataclass(frozen=True, slots=True)
class AblationArms:
    """The four orders, run over one lattice with one evaluator.  Nothing summarised yet."""

    guided: SweepResult
    fixed: SweepResult
    cost: SweepResult
    randoms: tuple[SweepResult, ...]
    random_seeds: tuple[int, ...]
    cells: int
    warmup_cells: int

    def every(self) -> tuple[SweepResult, ...]:
        return (self.guided, self.fixed, self.cost, *self.randoms)


def run_arms(
    lattice: Sequence[LatticeCell],
    evaluate: Callable[[LatticeCell], int],
    guided_scheduler: Any,
    *,
    random_seeds: Sequence[int] = RANDOM_SEEDS,
) -> AblationArms:
    """Run guided, fixed, cost-ascending and every seeded random order over the same lattice.

    Every arm gets a fresh :class:`CountingEvaluator` wrapping the SAME evaluator over the SAME
    cells, so the arms differ in order and in nothing else.
    """

    cells = canonical_lattice(lattice)
    return AblationArms(
        guided=sweep(cells, guided_scheduler, CountingEvaluator(evaluate)),
        fixed=sweep(cells, LexicographicScheduler(), CountingEvaluator(evaluate)),
        cost=sweep(cells, CostScheduler(), CountingEvaluator(evaluate)),
        randoms=tuple(
            sweep(cells, RandomScheduler(seed=seed), CountingEvaluator(evaluate))
            for seed in random_seeds
        ),
        random_seeds=tuple(random_seeds),
        cells=len(cells),
        warmup_cells=int(getattr(guided_scheduler, "warmup_cells", 0)),
    )


def ablation_report(arms: AblationArms) -> dict[str, Any]:
    """Summarise the arms, refusing first if they did not agree on the per-cell results.

    The seeds come off ``arms`` rather than off a parameter: a report that could be handed a
    different seed list than the one that was actually swept would be a receipt describing a run
    that never happened.

    The refusal is the load-bearing part.  Comparing cells-to-first-hit between two orders is only
    meaningful if the two orders measured the same thing, so the digests are confronted before any
    comparison is reported rather than after.
    """

    guided, fixed, cost, randoms = arms.guided, arms.fixed, arms.cost, list(arms.randoms)
    digests = {run.results_sha256() for run in arms.every()}
    if len(digests) != 1:
        raise OrderChangedTheSet(
            "the ablation arms disagreed on the per-cell results, so the evaluator is order "
            "dependent and no comparison between the arms means anything"
        )

    quarter = -(-arms.cells // 4)
    guided_metrics = sweep_metrics(guided, quarter=quarter)
    fixed_metrics = sweep_metrics(fixed, quarter=quarter)
    cost_metrics = sweep_metrics(cost, quarter=quarter)
    random_metrics = [sweep_metrics(run, quarter=quarter) for run in randoms]
    first = [item["cells_to_first_hit"] for item in random_metrics]
    half = [item["cells_to_half_the_hits"] for item in random_metrics]
    front = [item["hits_in_the_first_quarter"] for item in random_metrics]

    beats_random_first = _beats(guided_metrics["cells_to_first_hit"], first)
    beats_random_half = _beats(guided_metrics["cells_to_half_the_hits"], half)
    verdict = _verdict(
        guided_metrics,
        fixed_metrics,
        beats_random_first,
        beats_random_half,
        warmup=arms.warmup_cells,
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "lattice_id": SYNTHETIC_LATTICE_ID,
        "cells": arms.cells,
        "non_zero_cells": guided_metrics["non_zero_cells"],
        "results_sha256": guided.results_sha256(),
        "arms": {
            "guided": guided_metrics,
            "fixed_lexicographic": fixed_metrics,
            "cost_ascending": cost_metrics,
            "random": {
                "seeds": list(arms.random_seeds),
                "cells_to_first_hit": _spread([v for v in first if v is not None]),
                "cells_to_half_the_hits": _spread([v for v in half if v is not None]),
                "hits_in_the_first_quarter": _spread(front),
            },
        },
        "guided_versus_random": {
            "cells_to_first_hit": beats_random_first,
            "cells_to_half_the_hits": beats_random_half,
        },
        "order_digests": {
            "guided": guided.order_sha256(),
            "fixed_lexicographic": fixed.order_sha256(),
            "cost_ascending": cost.order_sha256(),
        },
        "guided_order_differs_from_fixed": guided.order_sha256() != fixed.order_sha256(),
        "guidance": guided.diagnostics,
        "verdict": verdict,
        "claims": dict(CLAIMS),
    }


def ablation(
    lattice: Sequence[LatticeCell],
    evaluate: Callable[[LatticeCell], int],
    guided_scheduler: Any,
    *,
    random_seeds: Sequence[int] = RANDOM_SEEDS,
) -> dict[str, Any]:
    """Guided against fixed, cost-ascending and seeded random, on one lattice, then summarised."""

    return ablation_report(run_arms(lattice, evaluate, guided_scheduler, random_seeds=random_seeds))


def _verdict(
    guided: Mapping[str, Any],
    fixed: Mapping[str, Any],
    first: Mapping[str, Any],
    half: Mapping[str, Any],
    *,
    warmup: int,
) -> dict[str, Any]:
    """The honest reading, assembled from the counts rather than chosen."""

    beat_random_first = bool(first.get("guided_beats_the_majority"))
    beat_random_half = bool(half.get("guided_beats_the_majority"))
    beat_fixed_first = _strictly_better(guided["cells_to_first_hit"], fixed["cells_to_first_hit"])
    beat_fixed_half = _strictly_better(
        guided["cells_to_half_the_hits"], fixed["cells_to_half_the_hits"]
    )
    parts: list[str] = []
    if beat_fixed_half:
        parts.append(
            f"guidance reached half the non-zero cells in {guided['cells_to_half_the_hits']} "
            f"evaluations against {fixed['cells_to_half_the_hits']} for the fixed order"
        )
    else:
        parts.append(
            f"guidance did NOT beat the fixed order to half the non-zero cells "
            f"({guided['cells_to_half_the_hits']} against {fixed['cells_to_half_the_hits']})"
        )
    if beat_random_half:
        parts.append(
            f"and beat the random order on that metric in "
            f"{half.get('guided_strictly_better_in')} of {half.get('comparable_runs')} seeds -- a "
            "majority, but a thin one, and not a margin anyone should build on"
        )
    else:
        parts.append(
            f"and did NOT beat the random order on that metric, winning "
            f"{half.get('guided_strictly_better_in')} of {half.get('comparable_runs')} seeds"
        )
    if not beat_random_first:
        # This is structural, not a tuning failure, and saying so is the point of reporting it.
        parts.append(
            f"; on cells-to-first-hit guidance loses to random ({first.get('guided_strictly_better_in')}"
            f" of {first.get('comparable_runs')} seeds) and cannot do otherwise: for its first "
            f"{warmup} evaluations it has no measured dimension to fit and runs the fixed order, so "
            "its first-hit number is the fixed order's number whenever the fixed order finds a hit "
            "inside the warm-up.  A guided scheduler with an uninformative prior can only pay off "
            "AFTER the first hit"
        )
    return {
        "guided_beats_random_on_first_hit": beat_random_first,
        "guided_beats_random_on_half_the_hits": beat_random_half,
        "guided_beats_random_on_both_metrics": beat_random_first and beat_random_half,
        "guided_beats_fixed_on_first_hit": beat_fixed_first,
        "guided_beats_fixed_on_half_the_hits": beat_fixed_half,
        "warmup_cells": warmup,
        "summary": "".join(part if part.startswith(";") else f" {part}" for part in parts).strip(),
    }


def _strictly_better(guided: int | None, other: int | None) -> bool | None:
    if guided is None or other is None:
        return None
    return guided < other


# ---------------------------------------------------------------------------
# 8. Campaign entry point.
# ---------------------------------------------------------------------------


def default_governor(ledger_path: str | Path, *, max_calls: int = 12) -> SpendGovernor:
    """The declared envelope for a guided sweep: a hard call cap and a hard dollar cap.

    ``charge_per_call_hundredths`` is one cent, which is the repo's convention for a small
    structured call.  With the deterministic mock proposer nothing is actually spent, and the ledger
    still moves -- the point of a governor is that the accounting does not depend on which proposer
    happens to be wired in.
    """

    return SpendGovernor(
        ledger_path=Path(ledger_path),
        max_calls=max_calls,
        max_dollars_hundredths=max_calls,
        charge_per_call_hundredths=1,
    )


def campaign_arms(proposer: Any, governor: SpendGovernor, **kwargs: Any) -> AblationArms:
    """Run every arm of the declared campaign once.  Sweeping is the expensive part; do it here."""

    guided = GuidedScheduler(proposer=proposer, governor=governor, **kwargs)
    return run_arms(synthetic_lattice(), synthetic_dimension, guided)


def campaign_report(arms: AblationArms) -> dict[str, Any]:
    """Summarise the arms and seal the receipt.  No float ever reaches this dictionary."""

    report = ablation_report(arms)
    report["lattice_monotonicity"] = assert_lattice_monotone(
        synthetic_lattice(), synthetic_dimension
    )
    report["source_path"] = SOURCE_PATH
    report["test_path"] = TEST_PATH
    report["receipt_sha256"] = canonical_sha256(report)
    return report


def run_campaign(proposer: Any, governor: SpendGovernor, **kwargs: Any) -> dict[str, Any]:
    """The whole thing: sweep every arm, measure monotonicity, seal the report."""

    return campaign_report(campaign_arms(proposer, governor, **kwargs))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI shell
    from .funsearch_loop import MockMutationProposer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", default="runs/math/guided-lattice-order/ledger.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-calls", type=int, default=12)
    arguments = parser.parse_args(argv)

    proposer = MockMutationProposer(seed=arguments.seed, bank=MUTATION_BANK)
    governor = default_governor(arguments.ledger, max_calls=arguments.max_calls)
    report = run_campaign(proposer, governor)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(report["verdict"]["summary"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
