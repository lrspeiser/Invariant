"""More than one model in the loop, with attribution, so the ensemble can be audited.

:class:`~.funsearch_loop.ClaudeCliProposer` takes one model, fixed for a whole campaign.
AlphaEvolve reports that this is the wrong shape: a fast model maximises the *breadth* of
ideas explored, a stronger model supplies *depth*, and the ensemble beats either alone.  This
engine could not express that at all -- there was one ``--model`` flag and no way to say
"three quarters cheap, one quarter expensive", let alone to ask afterwards whether the
expensive quarter paid for itself.

So this module adds three things, and the third is the one that matters.

**A declared ensemble.**  :class:`ModelEnsemble` is a tuple of :class:`ModelSlot` -- a model
name, an integer weight, an integer cost in units of the governor's base charge, and an
optional per-model proposal count.  Weights are integers because the draw schedule is then
exact: :meth:`ModelEnsemble.cycle` is a largest-remainder allocation over integers, so a
campaign of ``k`` calls gives model ``i`` a count within one of ``w_i * k / W`` with no
floating point anywhere, and a full cycle gives it *exactly* ``w_i``.  A weighted-random
policy is available and seeded, but the deterministic cycle is the default because a
realized allocation that provably equals the declared one is what makes an ablation clean.

**Attribution on every sealed program.**  Each record carries which model was first to seal
that exact source, which other models later returned the identical source, and how many times
each returned it.  Attribution is by *return*, not by novelty of authorship: if two models
both hand back the same program, both are credited with the behaviour, because the question
the attribution has to answer is "what would be missing if this model were not here".

**Per-model yield, and the ablation it makes possible.**  For each model: calls, spend,
programs returned, programs it was first to seal, distinct behaviours contributed, behaviours
*no other model contributed*, and best quality reached.  Then a leave-one-model-out
counterfactual over the sealed population: what the campaign's behaviour set and best quality
would have been without that model, and whether its share of unique behaviour is at least its
share of spend.  That is Tier 6 L2 -- "removing the LLM must measurably degrade yield, or it
is decoration" -- asked of one model inside the ensemble instead of the lane as a whole.

**What the numbers are not.**  Two honest limits, both declared in the receipt.

*The counterfactual is on the sealed population, not a re-run.*  Deleting a model would change
the search trajectory, so "best quality without this model" is the best quality among the
programs the other models actually returned, not the best quality they would have reached with
the freed budget.  It is a lower bound on what the survivors could do and it is stated as one.

*Behaviour clustering is generous.*  Behaviours are clusters of output vectors under
:func:`~.creativity_measure.cluster_behaviours`, which refuses to merge vectors with a
non-positive coordinate.  It can therefore split one behaviour into two but never merge two
into one, so ``behaviours_unique_to_this_model`` is an **upper** bound.  A model that scores
``earned_its_cost`` under a generous measure has a weak claim; a model that scores
``did_not_earn_its_cost`` under a generous measure has failed a test tilted in its favour,
and that verdict is the strong one.  The negative control in the test suite is exactly this
case: two models with identical behaviour, where the honest answer is that neither model
contributed anything the other did not.

**What it measured the first time it was run.**  On the demonstration ensemble -- three
quarters of the calls to a cheap wide-bank model, one quarter to an eight-times-dearer model
whose bank alone can express the sealed rule's residual term -- the expensive model took
``8/11`` of the spend and returned ``5/47`` of the behaviours only it contributed, so its
``cost_verdict`` is ``did_not_earn_its_cost``.  Against the equal-*dollar* single-model
control it is worse still: the cheap model alone, given the same 132 hundredths and therefore
132 calls instead of 48, reached quality 0.993890818 against the ensemble's 0.982271003.  So
the first thing this module did was refuse the assumption it was built to test.  That is the
intended use.

Cost goes through the existing :class:`~.funsearch_loop.SpendGovernor` with its caps
untouched; the only addition is that a call may cost more than one unit of the base charge.
When the cap binds, the campaign halts -- it does not skip the expensive model and carry on,
because that would silently rewrite the declared weights into something nobody declared.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .creativity_measure import DEFAULT_TOLERANCE, cluster_behaviours
from .funsearch_loop import (
    HIGH_QUALITY_THRESHOLD,
    ClaudeCliProposer,
    LoopConfig,
    MockMutationProposer,
    ProblemSpec,
    ProposalCall,
    ProposerCallFailed,
    ScoredProgram,
    SpendGovernor,
    build_prompt,
    classify_failure,
    declared_problems,
    reset_islands,
    sample_examples,
    score_program,
    stable_hash,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-model-ensemble-proposer-1.0"

#: The declared draw policies.  Both are reproducible from the seed; they differ in whether
#: the realized allocation is guaranteed to match the declared weights or only to match them
#: in expectation.
DRAW_POLICIES: tuple[str, ...] = ("deterministic_cycle", "weighted_random")

#: The deterministic cycle has one entry per unit of total weight and the receipt carries it
#: whole, so the declaration is checkable by reading it.  Beyond this the cycle stops being
#: something a person can check and starts being a wall of text: weights are *shares*, and
#: 512 of them is far finer than any ensemble anyone can justify.
MAX_TOTAL_WEIGHT = 512

CLAIMS = {
    "attribution_is_by_return_not_by_authorship": True,
    "behaviour_clustering_can_split_never_merge": True,
    "counterfactual_is_on_the_sealed_population_not_a_rerun": True,
    "every_call_is_charged_before_it_is_made": True,
    "model_identity_never_reaches_the_prompt": True,
    "weights_are_integers_so_the_schedule_is_exact": True,
}

SCOPE = (
    "A proposer that draws from a declared ensemble of models with integer weights, records "
    "on every sealed program which model returned it, and reports per-model yield: calls, "
    "spend, programs returned, programs first sealed, distinct behaviours contributed, "
    "behaviours no other model contributed, and best quality reached. The leave-one-model-out "
    "figures are a counterfactual over the sealed population, not a re-run, and behaviour "
    "counts are an upper bound because the clustering can split a behaviour but never merge "
    "two. A 'did_not_earn_its_cost' verdict is therefore strong and 'earned_its_cost' is weak."
)


class EnsembleError(ValueError):
    """Typed blocker: the ensemble declaration or its receipt is not well formed."""


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSlot:
    """One model in the ensemble, with everything about it declared as integers.

    ``weight`` is a share of the call schedule.  ``cost_units`` is the price of one call in
    units of the governor's ``charge_per_call_hundredths``, so a model eight times the price
    of the cheap one costs eight units and the receipt can divide spend by yield exactly.
    ``proposals_per_call`` overrides the loop's setting when positive, which is how "the fast
    model explores breadth" is said in the declaration rather than asserted in prose.
    """

    name: str
    weight: int
    cost_units: int = 1
    proposals_per_call: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise EnsembleError("a model slot needs a name")
        for field_name in ("weight", "cost_units"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise EnsembleError(f"{field_name} must be an integer of at least 1")
        if (
            not isinstance(self.proposals_per_call, int)
            or isinstance(self.proposals_per_call, bool)
            or self.proposals_per_call < 0
        ):
            raise EnsembleError("proposals_per_call must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "cost_units": self.cost_units,
            "proposals_per_call": self.proposals_per_call,
        }


@dataclass(frozen=True, slots=True)
class ModelEnsemble:
    """A weighted set of models and the rule that turns it into a call schedule."""

    slots: tuple[ModelSlot, ...]
    draw_policy: str = "deterministic_cycle"
    seed: int = 20260819

    def __post_init__(self) -> None:
        if not self.slots:
            raise EnsembleError("an ensemble needs at least one model")
        names = [slot.name for slot in self.slots]
        if len(set(names)) != len(names):
            raise EnsembleError(f"model names must be distinct: {sorted(names)}")
        if self.draw_policy not in DRAW_POLICIES:
            raise EnsembleError(f"undeclared draw policy: {self.draw_policy}")
        total = sum(slot.weight for slot in self.slots)
        if total > MAX_TOTAL_WEIGHT:
            raise EnsembleError(
                f"total weight {total} exceeds the declared maximum {MAX_TOTAL_WEIGHT}; "
                "reduce the weights to their common ratio"
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.slots)

    @property
    def total_weight(self) -> int:
        return sum(slot.weight for slot in self.slots)

    def slot(self, name: str) -> ModelSlot:
        for item in self.slots:
            if item.name == name:
                return item
        raise EnsembleError(f"undeclared model: {name}")

    def cycle(self) -> tuple[str, ...]:
        """One full period of the deterministic schedule: exactly ``weight`` calls each.

        Largest-remainder allocation, done in integers.  At step ``k`` the model chosen is the
        one with the largest ``weight_i * k - allocated_i * W``; ties break on declaration
        order.  Nothing here is floating point, and the invariant it buys is checkable:
        after any ``k`` draws, ``|allocated_i * W - weight_i * k| < W``.
        """

        total = self.total_weight
        allocated = [0] * len(self.slots)
        order: list[str] = []
        for step in range(1, total + 1):
            best = max(
                range(len(self.slots)),
                key=lambda i: (self.slots[i].weight * step - allocated[i] * total, -i),
            )
            allocated[best] += 1
            order.append(self.slots[best].name)
        return tuple(order)

    def schedule(self, calls: int) -> tuple[str, ...]:
        """The first ``calls`` model draws under the declared policy."""

        if calls < 0:
            raise EnsembleError("a schedule length cannot be negative")
        if self.draw_policy == "deterministic_cycle":
            period = self.cycle()
            return tuple(period[index % len(period)] for index in range(calls))
        rng = random.Random(self.seed)
        weights = [slot.weight for slot in self.slots]
        return tuple(
            rng.choices(self.names, weights=weights, k=1)[0] for _ in range(calls)
        )

    def allocation_is_exact(self, counts: Mapping[str, int]) -> bool:
        """Does a realized call count honour the declared weights to the schedule's bound?

        Integer arithmetic throughout: for every model, ``|c_i * W - w_i * k| < W``.  Under
        ``weighted_random`` no such bound exists and this returns ``True`` vacuously, which
        is why the deterministic cycle is the default for anything an ablation rests on.
        """

        if self.draw_policy != "deterministic_cycle":
            return True
        total_calls = sum(counts.get(name, 0) for name in self.names)
        total = self.total_weight
        return all(
            abs(counts.get(slot.name, 0) * total - slot.weight * total_calls) < total
            for slot in self.slots
        )

    def to_dict(self) -> dict[str, Any]:
        total = self.total_weight
        return {
            "draw_policy": self.draw_policy,
            "seed": self.seed,
            "total_weight": total,
            "models": [
                {
                    **slot.to_dict(),
                    "declared_share": _ratio(slot.weight, total),
                }
                for slot in self.slots
            ],
            "cycle": list(self.cycle()),
        }


def _ratio(numerator: int, denominator: int) -> str:
    """An exact rational as ``"a/b"``.  Never a float: canonical_sha256 refuses those."""

    if denominator == 0:
        return "0/1"
    value = Fraction(int(numerator), int(denominator))
    return f"{value.numerator}/{value.denominator}"


def _per(numerator: int, denominator: int) -> str:
    """Cost per unit of yield.  ``"undefined"`` when the yield is zero, never a silent 0."""

    return _ratio(numerator, denominator) if denominator else "undefined"


# ---------------------------------------------------------------------------
# 2. The proposer
# ---------------------------------------------------------------------------


class EnsembleProposer:
    """Routes each call to one declared model and remembers who answered.

    It is a drop-in for the single-model proposers: ``propose`` and ``programs`` have the same
    shapes, so :func:`~.funsearch_loop.run_problem` accepts one.  What that loop cannot do is
    *record* the attribution, which is why :func:`run_ensemble_problem` exists.

    The prompt is passed through byte-identically.  The model's identity never enters it --
    there is no "you are the fast model" preamble -- so a prompt is a function of the problem
    and the examples alone and two models drawn on the same examples see the same bytes.  That
    is also what keeps the blindness guard meaningful: it screens one prompt, not one per model.
    """

    proposer_id = "model_ensemble"

    def __init__(
        self,
        ensemble: ModelEnsemble,
        delegates: Mapping[str, Any],
        *,
        max_calls: int = 100000,
    ) -> None:
        missing = sorted(set(ensemble.names) - set(delegates))
        if missing:
            raise EnsembleError(f"no delegate supplied for declared models: {missing}")
        extra = sorted(set(delegates) - set(ensemble.names))
        if extra:
            raise EnsembleError(f"delegates supplied for undeclared models: {extra}")
        self.ensemble = ensemble
        self._delegates = dict(delegates)
        self._schedule = ensemble.schedule(max_calls)
        self._index = 0
        self._last: tuple[str, ...] = ()
        self._last_model = ""
        self.calls_by_model: dict[str, int] = {name: 0 for name in ensemble.names}

    def peek(self) -> ModelSlot:
        """The slot the next call would use, without consuming it.

        The governor is charged *before* the call is made (L4), and it cannot be charged the
        right amount without knowing which model is about to answer.  This is that lookahead,
        and it is deterministic, so charging and calling can never disagree.
        """

        if self._index >= len(self._schedule):
            raise EnsembleError("the declared call schedule is exhausted")
        return self.ensemble.slot(self._schedule[self._index])

    def propose(self, prompt: str, examples: Sequence[ScoredProgram], count: int) -> ProposalCall:
        slot = self.peek()
        self._index += 1
        self._last_model = slot.name
        self.calls_by_model[slot.name] += 1
        wanted = slot.proposals_per_call or count
        call = self._delegates[slot.name].propose(prompt, examples, wanted)
        self._last = tuple(self._delegates[slot.name].programs())
        return ProposalCall(
            f"{self.proposer_id}[{slot.name}]",
            call.prompt_sha256,
            call.prompt_bytes,
            call.returned,
            call.ok,
            call.reason,
            call.detail,
        )

    def programs(self) -> tuple[str, ...]:
        return self._last

    @property
    def last_model(self) -> str:
        return self._last_model


def build_claude_ensemble(
    ensemble: ModelEnsemble, executable: str = "claude", timeout: float = 180.0
) -> dict[str, Any]:
    """One :class:`~.funsearch_loop.ClaudeCliProposer` per slot, each pinned to its model."""

    return {
        slot.name: ClaudeCliProposer(executable, model=slot.name, timeout=timeout)
        for slot in ensemble.slots
    }


def build_mock_ensemble(
    ensemble: ModelEnsemble, banks: Mapping[str, Sequence[str]], seed: int = 20260819
) -> dict[str, Any]:
    """One deterministic mutator per slot, each with its own declared token bank.

    The banks are what make two mock models behave differently, and differently in a way that
    is declared rather than emergent: a model whose bank cannot express a term can never reach
    a behaviour that needs it, whatever its budget.  This is what the test suite uses, so it
    spends nothing.
    """

    missing = sorted(set(ensemble.names) - set(banks))
    if missing:
        raise EnsembleError(f"no mutation bank declared for: {missing}")
    return {
        slot.name: MockMutationProposer(seed + index * 977, banks[slot.name])
        for index, slot in enumerate(ensemble.slots)
    }


# ---------------------------------------------------------------------------
# 3. Attribution
# ---------------------------------------------------------------------------


def _vector(outputs: Sequence[str]) -> tuple[float, ...] | None:
    try:
        return tuple(float(value) for value in outputs)
    except (TypeError, ValueError):
        return None


@dataclass
class _Attribution:
    """Who returned this exact source, and how often."""

    proposed_by: str = ""
    first_seen_call: int = -1
    first_seen_generation: int = -1
    first_seen_island: int = -1
    returns: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, call: int, generation: int, island: int) -> None:
        self.returns[model] = self.returns.get(model, 0) + 1
        if not self.proposed_by:
            self.proposed_by = model
            self.first_seen_call = call
            self.first_seen_generation = generation
            self.first_seen_island = island

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_by": self.proposed_by,
            "also_proposed_by": sorted(
                name for name in self.returns if name != self.proposed_by
            ),
            "first_seen_call": self.first_seen_call,
            "first_seen_generation": self.first_seen_generation,
            "first_seen_island": self.first_seen_island,
            "times_returned": sum(self.returns.values()),
            "times_returned_by_model": dict(sorted(self.returns.items())),
        }


def _empty_attribution() -> dict[str, Any]:
    """A founder or a probe: nobody in the ensemble proposed it, and that is recorded."""

    return {
        "proposed_by": "",
        "also_proposed_by": [],
        "first_seen_call": -1,
        "first_seen_generation": -1,
        "first_seen_island": -1,
        "times_returned": 0,
        "times_returned_by_model": {},
    }


# ---------------------------------------------------------------------------
# 4. The loop
# ---------------------------------------------------------------------------


def run_ensemble_problem(
    problem: ProblemSpec,
    config: LoopConfig,
    proposer: EnsembleProposer,
    governor: SpendGovernor,
    *,
    corpus: Any = None,
    seeded_programs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """One problem, one ensemble, one block of the receipt with attribution on every program.

    The selection rules are :func:`~.funsearch_loop.sample_examples` and
    :func:`~.funsearch_loop.reset_islands` -- imported, not restated, so this loop and the
    single-model loop cannot drift into being two different searches.

    A transient failure retries against the **next** model in the schedule rather than the
    one that just failed.  Two reasons, and both are load-bearing: the thing that failed is
    usually the thing that will fail again, and a retry that re-drew the same model would put
    an unscheduled extra call on it, which is precisely the bookkeeping the declared weights
    exist to make checkable.  Consuming the schedule keeps the allocation bound true over
    every call actually made, retries included.
    """

    rng = random.Random(config.seed ^ stable_hash(problem.problem_id))
    founders = [score_program(problem, problem.seed_program, origin="seed", corpus=corpus)]
    for label, source in seeded_programs:
        founders.append(score_program(problem, source, origin=label, corpus=corpus))
    islands: list[list[ScoredProgram]] = [list(founders) for _ in range(config.islands)]
    sealed: dict[str, ScoredProgram] = {item.program_sha256: item for item in founders}
    attribution: dict[str, _Attribution] = {}
    calls: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    charged_by_model: dict[str, int] = {name: 0 for name in proposer.ensemble.names}
    history: list[dict[str, Any]] = []
    for item in founders:
        if not item.sandbox.ok:
            incidents.append(
                {"program_sha256": item.program_sha256, **item.sandbox.classification()}
            )
    halt_reason = "generations_exhausted"

    steps = [
        (generation, island)
        for generation in range(config.generations)
        for island in (
            range(config.islands) if config.sweep_islands else (generation % config.islands,)
        )
    ]
    for generation, index in steps:
        slot = proposer.peek()
        if not governor.may_call(slot.cost_units):
            halt_reason = governor.halt_reason
            break
        examples = sample_examples(
            islands[index], config.examples_per_prompt, config.temperature, rng
        )
        prompt = build_prompt(problem, examples)
        governor.charge(slot.cost_units)
        charged_by_model[slot.name] += slot.cost_units * governor.charge_per_call_hundredths
        call = proposer.propose(prompt, examples, config.proposals_per_call)
        calls.append(
            {
                **call.to_dict(),
                "model": slot.name,
                "cost_units": slot.cost_units,
                "charged_hundredths": slot.cost_units * governor.charge_per_call_hundredths,
                "generation": generation,
                "island": index,
            }
        )
        attempt = 0
        while not call.ok:
            kind = classify_failure(call)
            if kind == "persistent" or attempt >= config.transient_retries:
                raise ProposerCallFailed(
                    f"proposal call to model {slot.name!r} failed at generation "
                    f"{generation}, island {index}: {call.reason} / {call.detail[:200]}.  "
                    f"Classified {kind}.  Stopping the run rather than continuing with an "
                    "empty generation and sealing a receipt that looks like a full campaign."
                )
            attempt += 1
            time.sleep(config.retry_backoff_seconds * attempt)
            retry_slot = proposer.peek()
            if not governor.may_call(retry_slot.cost_units):
                halt_reason = governor.halt_reason
                break
            governor.charge(retry_slot.cost_units)
            charged_by_model[retry_slot.name] += (
                retry_slot.cost_units * governor.charge_per_call_hundredths
            )
            call = proposer.propose(prompt, examples, config.proposals_per_call)
            calls.append(
                {
                    **call.to_dict(),
                    "model": retry_slot.name,
                    "cost_units": retry_slot.cost_units,
                    "charged_hundredths": (
                        retry_slot.cost_units * governor.charge_per_call_hundredths
                    ),
                    "generation": generation,
                    "island": index,
                }
            )
            slot = retry_slot
        if not call.ok:
            break
        call_index = len(calls) - 1
        produced: list[ScoredProgram] = []
        for source in proposer.programs():
            scored = score_program(
                problem,
                source,
                origin="proposed",
                generation=generation,
                island=index,
                corpus=corpus,
            )
            produced.append(scored)
            sealed.setdefault(scored.program_sha256, scored)
            record = attribution.setdefault(scored.program_sha256, _Attribution())
            record.record(proposer.last_model, call_index, generation, index)
            if not scored.sandbox.ok:
                incidents.append(
                    {"program_sha256": scored.program_sha256, **scored.sandbox.classification()}
                )
        islands[index] = sorted(
            {item.program_sha256: item for item in [*islands[index], *produced]}.values(),
            key=lambda item: (-item.final_score, item.program_sha256),
        )[: config.island_capacity]
        history.append(
            {
                "generation": generation,
                "island": index,
                "model": slot.name,
                "proposed": len(produced),
                "best_quality_in_island": format(
                    max((item.quality for item in islands[index]), default=0.0), ".9f"
                ),
            }
        )
        if config.reset_period and (generation + 1) % config.reset_period == 0:
            islands = reset_islands(islands, rng)

    ordered = sorted(
        sealed.values(), key=lambda item: (-item.final_score, -item.quality, item.program_sha256)
    )
    records = [
        {
            **item.to_dict(),
            "attribution": (
                attribution[item.program_sha256].to_dict()
                if item.program_sha256 in attribution
                else _empty_attribution()
            ),
        }
        for item in ordered
    ]
    block: dict[str, Any] = {
        "problem": problem.to_dict(),
        "loop": config.to_dict(),
        "ensemble": proposer.ensemble.to_dict(),
        "generations_run": len(history),
        "halt_reason": halt_reason,
        "population_history": history,
        "proposal_calls": calls,
        "sandbox_incidents": incidents,
        "sealed_programs": records,
        "charged_hundredths_by_model": dict(sorted(charged_by_model.items())),
    }
    block["model_yield"] = per_model_yield(block)
    block["ablation"] = leave_one_model_out(block)
    block["headline"] = _block_headline(block)
    return block


# ---------------------------------------------------------------------------
# 5. Yield and ablation
# ---------------------------------------------------------------------------


def _behaviour_clusters(
    records: Sequence[Mapping[str, Any]], tolerance: float = DEFAULT_TOLERANCE
) -> list[list[int]]:
    """Cluster the executable proposed programs by output vector, returning record indices."""

    positions: list[int] = []
    vectors: list[tuple[float, ...]] = []
    for index, record in enumerate(records):
        if record["attribution"]["proposed_by"] == "":
            continue
        vector = _vector(record.get("outputs") or ())
        if vector:
            positions.append(index)
            vectors.append(vector)
    return [[positions[j] for j in group] for group in cluster_behaviours(vectors, tolerance)]


def _decimal(value: str) -> Decimal:
    return Decimal(value)


def _best_quality(records: Sequence[Mapping[str, Any]]) -> Decimal:
    return max((_decimal(item["quality"]) for item in records), default=Decimal(0))


def per_model_yield(block: Mapping[str, Any]) -> dict[str, Any]:
    """Programs proposed, distinct behaviours contributed, best quality reached -- per model."""

    ensemble = block["ensemble"]
    names = [model["name"] for model in ensemble["models"]]
    records = block["sealed_programs"]
    calls = block["proposal_calls"]
    charged = block["charged_hundredths_by_model"]

    clusters = _behaviour_clusters(records)
    contributors: list[set[str]] = []
    for cluster in clusters:
        owners: set[str] = set()
        for position in cluster:
            owners.update(records[position]["attribution"]["times_returned_by_model"])
        contributors.append(owners)

    total_calls = len(calls)
    total_charged = sum(charged.values())
    total_weight = ensemble["total_weight"]

    rows: list[dict[str, Any]] = []
    for model in ensemble["models"]:
        name = model["name"]
        mine = [
            record
            for record in records
            if name in record["attribution"]["times_returned_by_model"]
        ]
        returned = sum(
            record["attribution"]["times_returned_by_model"].get(name, 0) for record in records
        )
        first = [r for r in records if r["attribution"]["proposed_by"] == name]
        executed = [r for r in mine if r["sandbox"]["ok"]]
        contributed = [i for i, owners in enumerate(contributors) if name in owners]
        unique = [i for i in contributed if contributors[i] == {name}]
        tuples = {tuple(r["outputs"]) for r in executed if r["outputs"]}
        model_calls = sum(1 for call in calls if call["model"] == name)
        spend = charged.get(name, 0)
        rows.append(
            {
                "model": name,
                "declared_weight": model["weight"],
                "declared_share": _ratio(model["weight"], total_weight),
                "cost_units_per_call": model["cost_units"],
                "calls": model_calls,
                "realized_share": _ratio(model_calls, total_calls),
                "charged_hundredths": spend,
                "spend_share": _ratio(spend, total_charged),
                "programs_returned": returned,
                "programs_sealed_first": len(first),
                "programs_executed": len(executed),
                "distinct_behaviours_contributed": len(contributed),
                "behaviours_unique_to_this_model": len(unique),
                "distinct_output_tuples_contributed": len(tuples),
                "best_quality_reached": format(_best_quality(mine), "f") if mine else "0",
                "best_final_score_reached": (
                    format(
                        max(_decimal(item["final_score"]) for item in mine), "f"
                    )
                    if mine
                    else "0"
                ),
                "high_quality_programs": sum(
                    1
                    for item in mine
                    if _decimal(item["quality"]) >= Decimal(str(HIGH_QUALITY_THRESHOLD))
                ),
                "hundredths_per_distinct_behaviour": _per(spend, len(contributed)),
                "hundredths_per_unique_behaviour": _per(spend, len(unique)),
            }
        )
    rows.sort(key=lambda row: row["model"])
    counts = {name: sum(1 for call in calls if call["model"] == name) for name in names}
    return {
        "declared": {
            "behaviour_key": (
                "clusters of the output vector under creativity_measure.cluster_behaviours, "
                "tolerance " + format(DEFAULT_TOLERANCE, ".3g")
            ),
            "behaviour_counts_are_an_upper_bound": True,
            "attribution_is_by_return_not_by_first_seal": True,
        },
        "totals": {
            "calls": total_calls,
            "charged_hundredths": total_charged,
            "programs_sealed": len(records),
            "programs_proposed_by_a_model": sum(
                1 for r in records if r["attribution"]["proposed_by"]
            ),
            "distinct_behaviours": len(clusters),
            "best_quality": format(_best_quality(records), "f") if records else "0",
        },
        "realized_allocation": dict(sorted(counts.items())),
        "allocation_honours_declared_weights": _allocation_ok(ensemble, counts),
        "rows": rows,
    }


def _allocation_ok(ensemble: Mapping[str, Any], counts: Mapping[str, int]) -> bool:
    if ensemble["draw_policy"] != "deterministic_cycle":
        return True
    total = ensemble["total_weight"]
    drawn = sum(counts.values())
    return all(
        abs(counts.get(model["name"], 0) * total - model["weight"] * drawn) < total
        for model in ensemble["models"]
    )


def leave_one_model_out(block: Mapping[str, Any]) -> dict[str, Any]:
    """What the campaign would have held without each model -- on the sealed population.

    This is the ablation Tier 6 L2 asks for, scoped to one model rather than the whole lane.
    It is a counterfactual over what was actually returned: the survivors do not get the
    removed model's budget back, so ``best_quality_without_this_model`` is a lower bound on
    what they would have reached, and ``quality_lost_by_removal`` is correspondingly an upper
    bound.  Both are stated, neither is smuggled.
    """

    records = block["sealed_programs"]
    clusters = _behaviour_clusters(records)
    contributors: list[set[str]] = []
    for cluster in clusters:
        owners: set[str] = set()
        for position in cluster:
            owners.update(records[position]["attribution"]["times_returned_by_model"])
        contributors.append(owners)
    charged = block["charged_hundredths_by_model"]
    total_charged = sum(charged.values())
    best_all = _best_quality(records)

    rows: list[dict[str, Any]] = []
    for model in block["ensemble"]["models"]:
        name = model["name"]
        lost = [i for i, owners in enumerate(contributors) if owners == {name}]
        survivors = [
            record
            for record in records
            if set(record["attribution"]["times_returned_by_model"]) - {name}
            or record["attribution"]["proposed_by"] == ""
        ]
        best_without = _best_quality(survivors)
        spend = charged.get(name, 0)
        spend_share = Fraction(spend, total_charged) if total_charged else Fraction(0)
        unique_share = Fraction(len(lost), len(clusters)) if clusters else Fraction(0)
        quality_lost = best_all - best_without
        contributed = len(lost) > 0 or quality_lost > 0
        efficient = unique_share >= spend_share
        rows.append(
            {
                "model": name,
                "spend_share": _ratio(spend, total_charged),
                "unique_behaviour_share": _ratio(len(lost), len(clusters)),
                "behaviours_lost_by_removal": len(lost),
                "behaviours_retained": len(clusters) - len(lost),
                "best_quality_with_every_model": format(best_all, "f"),
                "best_quality_without_this_model": format(best_without, "f"),
                "quality_lost_by_removal": format(quality_lost, "f"),
                "contribution_verdict": (
                    "contributed" if contributed else "contributed_nothing"
                ),
                "unique_share_at_least_spend_share": efficient,
                "cost_verdict": (
                    "earned_its_cost"
                    if contributed and efficient
                    else "did_not_earn_its_cost"
                ),
            }
        )
    rows.sort(key=lambda row: row["model"])
    return {
        "rule": (
            "two questions, kept apart because they have different answers. "
            "contribution_verdict is 'contributed' when removing the model from the sealed "
            "population would cost the campaign at least one behaviour or lower its best "
            "quality -- that is the cheap bar and almost anything clears it. cost_verdict is "
            "'earned_its_cost' only when the model also took no larger a share of the spend "
            "than it returned of the behaviours only it contributed. Both are exact: shares "
            "are rationals compared by cross-multiplication, qualities are decimals. The "
            "behaviour count is an upper bound, so 'did_not_earn_its_cost' is the strong "
            "verdict and 'earned_its_cost' is the weak one."
        ),
        "counterfactual_is_on_the_sealed_population_not_a_rerun": True,
        "rows": rows,
    }


def _block_headline(block: Mapping[str, Any]) -> dict[str, Any]:
    yields = block["model_yield"]
    ablation = block["ablation"]
    return {
        "models": len(block["ensemble"]["models"]),
        "calls": yields["totals"]["calls"],
        "charged_hundredths": yields["totals"]["charged_hundredths"],
        "programs_sealed": yields["totals"]["programs_sealed"],
        "distinct_behaviours": yields["totals"]["distinct_behaviours"],
        "best_quality": yields["totals"]["best_quality"],
        "allocation_honours_declared_weights": yields["allocation_honours_declared_weights"],
        "models_that_contributed_nothing": sorted(
            row["model"]
            for row in ablation["rows"]
            if row["contribution_verdict"] != "contributed"
        ),
        "models_that_earned_their_cost": sorted(
            row["model"]
            for row in ablation["rows"]
            if row["cost_verdict"] == "earned_its_cost"
        ),
        "models_that_did_not_earn_their_cost": sorted(
            row["model"]
            for row in ablation["rows"]
            if row["cost_verdict"] != "earned_its_cost"
        ),
    }


# ---------------------------------------------------------------------------
# 6. The campaign
# ---------------------------------------------------------------------------

#: The demonstration ensemble.  A cheap model with three quarters of the calls and a wide
#: shallow bank, and an expensive model with one quarter of the calls at eight times the
#: price and the one bank token that can express the sealed rule's residual term.  The point
#: is not that this is how haiku and opus differ; it is that the difference is DECLARED, so
#: the receipt's attribution can be checked against something known rather than believed.
DEMO_ENSEMBLE = ModelEnsemble(
    slots=(
        ModelSlot(name="broad", weight=3, cost_units=1, proposals_per_call=6),
        ModelSlot(name="deep", weight=1, cost_units=8, proposals_per_call=3),
    )
)

#: Token banks for the deterministic stand-ins.  ``broad`` cannot write ``n % 4``, so the
#: residual term of the sealed sequence rule is outside its reach whatever its budget.
DEMO_BANKS: dict[str, tuple[str, ...]] = {
    "broad": ("n", "n * n", "n % 2", "n % 3", "1", "2", "3", "4", "5"),
    "deep": ("n", "n * n", "n % 4", "1", "2", "3"),
}


def run_ensemble_campaign(
    *,
    config: LoopConfig | None = None,
    ensemble: ModelEnsemble | None = None,
    ledger_path: str | Path = "runs/math/model-ensemble/spend-ledger.json",
    max_calls: int = 240,
    max_dollars_hundredths: int = 1000,
    charge_per_call_hundredths: int = 1,
    problem_key: str = "blinded_sequence_rule",
    proposer_kind: str = "mock",
    claude_executable: str = "claude",
) -> dict[str, Any]:
    """Two arms, one receipt: the ensemble, and the cheap model alone at the same spend.

    The single-model arm is the control the ablation needs.  It is the same problem, the same
    loop configuration and the same spend cap, differing in exactly one declared field -- the
    ensemble -- so a difference in yield between the arms cannot come from anywhere else.
    """

    started = time.perf_counter()
    settings = config or LoopConfig(
        islands=4, generations=48, proposals_per_call=6, reset_period=12, seed=20260819
    )
    declared = ensemble or DEMO_ENSEMBLE
    problem = declared_problems()[problem_key]

    if proposer_kind not in ("mock", "claude"):
        raise EnsembleError(f"undeclared proposer kind: {proposer_kind}")

    def delegates_for(spec: ModelEnsemble, salt: int) -> dict[str, Any]:
        if proposer_kind == "claude":
            return build_claude_ensemble(spec, claude_executable)
        banks = {
            name: DEMO_BANKS.get(name, problem.mutation_bank) for name in spec.names
        }
        return build_mock_ensemble(spec, banks, seed=settings.seed + salt)

    blocks: list[dict[str, Any]] = []
    governor = SpendGovernor(
        Path(ledger_path), max_calls, max_dollars_hundredths, charge_per_call_hundredths
    )
    ensemble_block = run_ensemble_problem(
        problem,
        settings,
        EnsembleProposer(declared, delegates_for(declared, 0), max_calls=max_calls + 8),
        governor,
    )
    ensemble_block["run_label"] = "ensemble"
    blocks.append(ensemble_block)

    # The control arm.  One model, and -- this is the point -- the SAME NUMBER OF DOLLARS,
    # not the same number of calls.  Equal calls would be a rigged comparison: the ensemble
    # spent eleven hundredths per cycle of four calls and the cheap model spends four, so an
    # equal-call control asks whether more money buys more, which it obviously does.  Equal
    # spend asks the question the ablation actually needs: for this budget, was a quarter of
    # it better spent on the expensive model, or on nearly three times as many cheap calls?
    spent = governor.charged_hundredths
    cheapest = min(declared.slots, key=lambda slot: (slot.cost_units, slot.name))
    affordable = spent // max(1, cheapest.cost_units * charge_per_call_hundredths)
    solo = ModelEnsemble(
        slots=(replace(cheapest, weight=1),),
        draw_policy=declared.draw_policy,
        seed=declared.seed,
    )
    solo_settings = replace(settings, generations=max(1, affordable))
    solo_governor = SpendGovernor(
        Path(ledger_path), max(1, affordable), max(1, spent), charge_per_call_hundredths
    )
    solo_block = run_ensemble_problem(
        problem,
        solo_settings,
        EnsembleProposer(solo, delegates_for(solo, 0), max_calls=affordable + 8),
        solo_governor,
    )
    solo_block["run_label"] = "single_model_control"
    solo_block["control_note"] = (
        "no ensemble was drawn here. The cheapest declared model ran alone on the same "
        "problem under the same island rules, given exactly the hundredths the ensemble arm "
        "spent and therefore more calls. The loop configuration differs only in generations, "
        "which is what 'buy more cheap calls with the same money' means."
    )
    blocks.append(solo_block)

    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "lane": "model-ensemble-proposer",
        "claims": CLAIMS,
        "config": settings.to_dict(),
        "config_sha256": canonical_sha256(settings.to_dict()),
        "ensemble": declared.to_dict(),
        "ensemble_sha256": canonical_sha256(declared.to_dict()),
        "proposer": {
            "kind": proposer_kind,
            "delegate": (
                "claude_cli_oauth per model" if proposer_kind == "claude" else "mock per model"
            ),
            "note": (
                "the delegate supplies source code only; every number in this receipt was "
                "produced by executing a program, and the model that produced it is recorded "
                "on the program rather than asserted in prose"
            ),
        },
        "budget": {
            "ensemble_arm": governor.to_dict(),
            "single_model_arm": solo_governor.to_dict(),
            # A mock campaign still charges the governor -- the caps are part of what is being
            # tested -- but those hundredths are simulated, and a ledger that cannot tell
            # simulated dollars from real ones is the drift this repository has already been
            # bitten by once. The CLI keeps the two in separate files; this flag says which.
            "charges_are_simulated": proposer_kind != "claude",
        },
        "problems": blocks,
        "headline": _campaign_headline(blocks),
        "scope": SCOPE,
    }
    body["result_core_sha256"] = canonical_sha256(body)
    body["measurement"] = {"elapsed_seconds": format(time.perf_counter() - started, ".3f")}
    return {**body, "content_sha256": canonical_sha256(body)}


def _campaign_headline(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label = {block["run_label"]: block for block in blocks}
    lines = {
        block["run_label"]: block["headline"] for block in blocks
    }
    ensemble = by_label.get("ensemble")
    solo = by_label.get("single_model_control")
    comparison: dict[str, Any] = {"available": False}
    if ensemble is not None and solo is not None:
        left = ensemble["model_yield"]["totals"]
        right = solo["model_yield"]["totals"]
        comparison = {
            "available": True,
            "ensemble_distinct_behaviours": left["distinct_behaviours"],
            "single_model_distinct_behaviours": right["distinct_behaviours"],
            "ensemble_best_quality": left["best_quality"],
            "single_model_best_quality": right["best_quality"],
            "ensemble_calls": left["calls"],
            "single_model_calls": right["calls"],
            "ensemble_charged_hundredths": left["charged_hundredths"],
            "single_model_charged_hundredths": right["charged_hundredths"],
            "spend_is_equal": left["charged_hundredths"] == right["charged_hundredths"],
            "quality_delta": format(
                _decimal(left["best_quality"]) - _decimal(right["best_quality"]), "f"
            ),
            "behaviour_delta": (
                left["distinct_behaviours"] - right["distinct_behaviours"]
            ),
            "note": (
                "equal dollars, not equal calls. The control is the cheapest model alone -- "
                "what the engine did before this module existed -- given the hundredths the "
                "ensemble spent and therefore more calls. A positive quality_delta means the "
                "expensive model bought something the extra cheap calls did not."
            ),
        }
    lines["ensemble_versus_single_model"] = comparison
    return lines


# ---------------------------------------------------------------------------
# 7. Receipt validation
# ---------------------------------------------------------------------------


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seals, claims, and every arithmetic relation the attribution rests on."""

    if value.get("schema_version") != SCHEMA:
        raise EnsembleError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise EnsembleError("receipt seal changed")
    core = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    if value.get("result_core_sha256") != canonical_sha256(core):
        raise EnsembleError("deterministic core seal changed")
    if value.get("claims") != CLAIMS:
        raise EnsembleError("claims block changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise EnsembleError("config binding changed")
    if value.get("ensemble_sha256") != canonical_sha256(value.get("ensemble", {})):
        raise EnsembleError("ensemble binding changed")
    labels = [block["run_label"] for block in value["problems"]]
    if "ensemble" not in labels:
        raise EnsembleError("the receipt carries no ensemble arm")
    if "single_model_control" not in labels:
        raise EnsembleError("the receipt carries no single-model control arm")
    for block in value["problems"]:
        validate_block(block)


def validate_block(block: Mapping[str, Any]) -> None:
    """One arm: attribution, allocation, spend and yield must all agree with each other."""

    declared = {model["name"] for model in block["ensemble"]["models"]}
    if not declared:
        raise EnsembleError("a block declares no models")

    proposed = 0
    for record in block["sealed_programs"]:
        attribution = record["attribution"]
        by = attribution["proposed_by"]
        returns = attribution["times_returned_by_model"]
        if by:
            proposed += 1
            if by not in declared:
                raise EnsembleError(f"a program is attributed to an undeclared model: {by}")
            if by not in returns:
                raise EnsembleError(f"a program's first sealer is not among its returners: {by}")
        elif returns:
            raise EnsembleError("a program has returners but no first sealer")
        unknown = sorted(set(returns) - declared)
        if unknown:
            raise EnsembleError(f"a program is attributed to undeclared models: {unknown}")
        if sum(returns.values()) != attribution["times_returned"]:
            raise EnsembleError("an attribution's return counts do not sum to its total")
        if sorted(name for name in returns if name != by) != attribution["also_proposed_by"]:
            raise EnsembleError("an attribution's also_proposed_by disagrees with its counts")
        if record["origin"] == "proposed" and not by:
            # The converse is deliberately NOT required.  A mutator can regenerate the seed,
            # and when it does the record keeps its founder origin and gains a returner, which
            # is the honest thing to write down rather than an inconsistency to hide.
            raise EnsembleError("a proposed program carries no model attribution")

    yields = block["model_yield"]
    if yields["totals"]["programs_proposed_by_a_model"] != proposed:
        raise EnsembleError("the yield table counts a different number of proposed programs")
    if sum(row["programs_sealed_first"] for row in yields["rows"]) != proposed:
        raise EnsembleError("first-seal counts do not partition the proposed programs")
    if {row["model"] for row in yields["rows"]} != declared:
        raise EnsembleError("the yield table does not cover exactly the declared models")

    calls = block["proposal_calls"]
    if sum(row["calls"] for row in yields["rows"]) != len(calls):
        raise EnsembleError("per-model call counts do not sum to the calls made")
    if yields["totals"]["calls"] != len(calls):
        raise EnsembleError("the yield total disagrees with the recorded calls")
    charged = block["charged_hundredths_by_model"]
    if sum(charged.values()) != sum(call["charged_hundredths"] for call in calls):
        raise EnsembleError("per-model spend does not sum to the charged calls")
    for row in yields["rows"]:
        if row["charged_hundredths"] != charged.get(row["model"], 0):
            raise EnsembleError(f"the yield table misreports spend for {row['model']}")
        if row["programs_returned"] > 0 and row["charged_hundredths"] <= 0:
            raise EnsembleError(
                f"model {row['model']} returned programs and was charged nothing"
            )
        if row["behaviours_unique_to_this_model"] > row["distinct_behaviours_contributed"]:
            raise EnsembleError("a model claims more unique behaviours than it contributed")
    if not yields["allocation_honours_declared_weights"]:
        raise EnsembleError(
            "the realized call allocation does not honour the declared weights"
        )
    if not _allocation_ok(block["ensemble"], yields["realized_allocation"]):
        raise EnsembleError("the recorded allocation fails the declared weight bound")
    for name, count in yields["realized_allocation"].items():
        if count != sum(1 for call in calls if call["model"] == name):
            raise EnsembleError(f"the realized allocation misreports calls for {name}")

    ablation = block["ablation"]
    total_behaviours = yields["totals"]["distinct_behaviours"]
    if {row["model"] for row in ablation["rows"]} != declared:
        raise EnsembleError("the ablation does not cover exactly the declared models")
    for row in ablation["rows"]:
        if row["behaviours_lost_by_removal"] + row["behaviours_retained"] != total_behaviours:
            raise EnsembleError("an ablation row does not account for every behaviour")
        lost = _decimal(row["best_quality_with_every_model"]) - _decimal(
            row["best_quality_without_this_model"]
        )
        if lost != _decimal(row["quality_lost_by_removal"]):
            raise EnsembleError("an ablation row's quality arithmetic does not close")
        if lost < 0:
            raise EnsembleError("removing a model cannot raise the best quality")
        contributed = row["behaviours_lost_by_removal"] > 0 or lost > 0
        expected = "contributed" if contributed else "contributed_nothing"
        if row["contribution_verdict"] != expected:
            raise EnsembleError(
                f"the contribution verdict for {row['model']} does not follow its own rule"
            )
        efficient = Fraction(
            *(int(part) for part in row["unique_behaviour_share"].split("/"))
        ) >= Fraction(*(int(part) for part in row["spend_share"].split("/")))
        if row["unique_share_at_least_spend_share"] != efficient:
            raise EnsembleError(
                f"the cost-share comparison for {row['model']} does not match its own shares"
            )
        cost = "earned_its_cost" if contributed and efficient else "did_not_earn_its_cost"
        if row["cost_verdict"] != cost:
            raise EnsembleError(
                f"the cost verdict for {row['model']} does not follow its own rule"
            )


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def write_receipt(result: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(result) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A FunSearch proposer that draws from a weighted ensemble of models."
    )
    parser.add_argument("--output", default="runs/math/model-ensemble/ensemble-v1.json")
    #: Left unset, the ledger follows the proposer: real dollars for a live campaign,
    #: a separate simulated ledger for the mock. Mixing them is how a spend ledger drifts.
    parser.add_argument("--ledger", default="")
    parser.add_argument("--generations", type=int, default=48)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-calls", type=int, default=240)
    parser.add_argument("--max-dollars-hundredths", type=int, default=1000)
    parser.add_argument("--proposer", choices=("mock", "claude"), default="mock")
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--problem", default="blinded_sequence_rule")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)

    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(json.dumps({"validated": True, "output": args.output}))
        return 0

    ledger = args.ledger or (
        "runs/math/model-ensemble/spend-ledger.json"
        if args.proposer == "claude"
        else "runs/math/model-ensemble/spend-ledger-simulated.json"
    )
    result = run_ensemble_campaign(
        config=LoopConfig(
            islands=args.islands,
            generations=args.generations,
            proposals_per_call=6,
            reset_period=12,
            seed=args.seed,
        ),
        ledger_path=ledger,
        max_calls=args.max_calls,
        max_dollars_hundredths=args.max_dollars_hundredths,
        problem_key=args.problem,
        proposer_kind=args.proposer,
        claude_executable=args.claude,
    )
    validate_receipt(result)
    write_receipt(result, args.output)
    print(
        json.dumps(
            {
                "headline": result["headline"],
                "output": args.output,
                "content_sha256": result["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


__all__ = [
    "CLAIMS",
    "DEMO_BANKS",
    "DEMO_ENSEMBLE",
    "DRAW_POLICIES",
    "MAX_TOTAL_WEIGHT",
    "SCHEMA",
    "EnsembleError",
    "EnsembleProposer",
    "ModelEnsemble",
    "ModelSlot",
    "build_claude_ensemble",
    "build_mock_ensemble",
    "leave_one_model_out",
    "main",
    "per_model_yield",
    "run_ensemble_campaign",
    "run_ensemble_problem",
    "validate_block",
    "validate_receipt",
    "write_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
