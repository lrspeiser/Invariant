"""A5 — the declared, frozen registry of every discovery lane.

The scheduler used to derive work from one fixed stage list per machine-form kind.  That
list was a *convenience*, not a claim: it decided which capabilities ever touched a
problem, and every capability it omitted vanished without trace.  A lane that never ran
looks exactly like a lane that ran and found nothing, and a system that cannot tell those
apart cannot honestly say what it has tried.

This module fixes that boundary.  Every landed discovery lane is declared here once —
what it accepts, what it costs, what resource it needs, what upstream artifact it
requires, and the exact vocabulary of typed blockers it is allowed to emit.
:func:`applicable_lanes` answers, for one queue entry, which lanes can run **and why each
of the others cannot**, with a typed reason drawn from a frozen list.

Three rules keep the registry trustworthy.

**A skip is a fact, never an absence.**  :func:`applicable_lanes` returns a decision for
*every* declared lane.  An inapplicable lane carries a typed
:data:`SKIP_REASONS` code and a hand-checkable detail string; nothing is silently
dropped, so "we never tried X on Y" is recoverable from the record.

**Applicability is decided from declared data, never from a trial run.**  A lane accepts
a set of ``machine_form.kind`` values and/or a named shape predicate over the entry's own
declared fields (row budget, equation roster, target constant).  Every predicate is a
pure function of the sealed queue entry, so the same entry always yields the same
decision and the decision can be re-derived by hand from the queue file.

**The registry is frozen.**  :data:`REGISTRY_CONTENT_SHA256` binds the whole declared
table under canonical JSON.  Adding, removing, or re-scoping a lane changes that digest
and fails the pin test — registering a capability is a deliberate act, not a side effect.

Claim boundary: registry membership asserts that a lane *exists and declares what it
accepts*.  It asserts nothing about whether the lane will succeed, whether its output is
correct, and nothing at all about the mathematics.  ``awaiting_problem_kind`` records
that a landed lane has no target in the current queue — a gap in the *queue*, stated
plainly rather than hidden by deleting the lane.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .sigma_core import canonical_sha256

REGISTRY_SCHEMA = "invariant-lane-registry-1.0"

#: Row cap of the scheduler's built-in generators.  Mirrors
#: ``discovery_scheduler.GENERATOR_CAPS["max_rows"]``; a test pins the two together so
#: the registry can never promise a lane more rows than the host can produce.
EFFECTIVE_ROW_CAP = 64

#: Every reason a lane can be skipped.  Exhaustive and frozen: :func:`applicable_lanes`
#: may return no other code, and a test proves each one is reachable from the sealed
#: queue, so the vocabulary is neither over- nor under-specified.
SKIP_REASONS: tuple[str, ...] = (
    "equation_not_in_sweeper_roster",
    "insufficient_rows",
    "kind_mismatch",
    "needs_bounded_coloring_statement",
    "needs_target_constant",
    "not_in_declared_roster",
)

#: Resource tags.  ``gpu`` lanes take the scheduler's single GPU lease; ``cpu`` lanes run
#: in the process pool.
RESOURCES: tuple[str, ...] = ("cpu", "gpu")

#: Declared cost classes.  Ordering only; no lane carries a predicted runtime.
COSTS: tuple[str, ...] = ("fast", "medium", "slow")

#: Problem ids the exponent-Diophantine sweeper is built for.  Declared, not inferred:
#: ``exponent_diophantine_sweeper.MODE_PROBLEM_IDS`` names three targets and
#: ``erdos_straus`` shares the Erdos-Straus box with its sweeper target.
DIOPHANTINE_SWEEPER_ROSTER: tuple[str, ...] = (
    "beal_conjecture",
    "erdos_straus",
    "erdos_straus_sweeper_target",
    "fermat_catalan",
)

#: The twelve targets of the unsolved-progress campaign
#: (``dozen_unsolved_progress_campaign.DOZEN_IDS``), restated here so the registry is
#: readable on its own; a test pins the two tuples together.
UNSOLVED_DOZEN_ROSTER: tuple[str, ...] = (
    "brocard_problem",
    "erdos_moser",
    "gilbreath_conjecture",
    "giuga_conjecture",
    "lehmer_totient",
    "lychrel_196",
    "odd_perfect_number",
    "odd_untouchable",
    "recaman_coverage",
    "singmaster_conjecture",
    "twin_prime_infinitude",
    "ulam_sequence_structure",
)


class LaneRegistryError(ValueError):
    """Raised when the registry is malformed or an unknown lane/predicate is named."""


# ---------------------------------------------------------------------------
# Declared shape predicates over a sealed queue entry
# ---------------------------------------------------------------------------


def declared_row_budget(entry: Mapping[str, Any]) -> int | None:
    """Rows this entry can supply, or ``None`` when the kind supplies no rows.

    ``sequence_rows`` declares ``max_point`` and ``integer_trajectory`` declares
    ``max_steps``; both are capped by :data:`EFFECTIVE_ROW_CAP`, because that is what the
    host's built-in generators will actually emit.
    """

    machine_form = entry["machine_form"]
    kind = machine_form["kind"]
    if kind == "sequence_rows":
        return min(int(machine_form["max_point"]), EFFECTIVE_ROW_CAP)
    if kind == "integer_trajectory":
        return min(int(machine_form["max_steps"]), EFFECTIVE_ROW_CAP)
    return None


def _predicate_target_constant(entry: Mapping[str, Any]) -> bool:
    """True when the entry declares a named target constant to invert."""

    return entry["machine_form"]["kind"] == "target_constant"


def _predicate_bounded_coloring(entry: Mapping[str, Any]) -> bool:
    """True when the entry is an M10-shaped bounded coloring statement."""

    return entry["machine_form"]["kind"] == "bounded_combinatorial_coloring"


def _predicate_diophantine_sweeper_roster(entry: Mapping[str, Any]) -> bool:
    return entry["id"] in DIOPHANTINE_SWEEPER_ROSTER


def _predicate_unsolved_dozen_roster(entry: Mapping[str, Any]) -> bool:
    return entry["id"] in UNSOLVED_DOZEN_ROSTER


#: predicate id -> (skip reason when it fails, prose statement, pure test).
SHAPE_PREDICATES: dict[str, tuple[str, str, Callable[[Mapping[str, Any]], bool]]] = {
    "target_constant": (
        "needs_target_constant",
        "the entry declares a named target constant to invert",
        _predicate_target_constant,
    ),
    "bounded_coloring_statement": (
        "needs_bounded_coloring_statement",
        "the entry declares a bounded combinatorial coloring statement (n, k)",
        _predicate_bounded_coloring,
    ),
    "in_diophantine_sweeper_roster": (
        "equation_not_in_sweeper_roster",
        f"the entry id is one of the declared sweeper targets {list(DIOPHANTINE_SWEEPER_ROSTER)}",
        _predicate_diophantine_sweeper_roster,
    ),
    "in_unsolved_dozen_roster": (
        "not_in_declared_roster",
        f"the entry id is one of the twelve campaign targets {list(UNSOLVED_DOZEN_ROSTER)}",
        _predicate_unsolved_dozen_roster,
    ),
}


# ---------------------------------------------------------------------------
# The lane specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneSpec:
    """One declared discovery lane.

    ``accepts_kinds`` is the set of ``machine_form.kind`` values the lane serves; empty
    means "decided by ``requires_shape`` alone".  ``requires_shape`` names predicates from
    :data:`SHAPE_PREDICATES`, each of which supplies the typed skip reason when it fails.
    ``min_rows`` is the lane's own declared row floor, read off the module's caps.
    ``emits_blockers`` is the *complete* vocabulary the lane may produce — a blocker
    outside it is a defect, not a discovery.
    """

    lane_id: str
    module: str
    entry_point: str
    stage: str
    accepts_kinds: tuple[str, ...]
    requires_shape: tuple[str, ...]
    resource: str
    typical_cost: str
    emits_blockers: tuple[str, ...]
    preconditions: tuple[str, ...]
    min_rows: int | None = None
    awaiting_problem_kind: bool = False
    note: str = ""

    def as_declared(self) -> dict[str, Any]:
        """Canonical-JSON-safe view of the declaration, used for the registry seal."""

        return {
            "accepts_kinds": list(self.accepts_kinds),
            "awaiting_problem_kind": self.awaiting_problem_kind,
            "emits_blockers": list(self.emits_blockers),
            "entry_point": self.entry_point,
            "lane_id": self.lane_id,
            "min_rows": self.min_rows,
            "module": self.module,
            "note": self.note,
            "preconditions": list(self.preconditions),
            "requires_shape": list(self.requires_shape),
            "resource": self.resource,
            "stage": self.stage,
            "typical_cost": self.typical_cost,
        }


@dataclass(frozen=True)
class LaneDecision:
    """Why one lane will or will not be attempted on one problem."""

    lane_id: str
    applicable: bool
    skip_reason: str | None
    detail: str

    def as_record(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "detail": self.detail,
            "lane_id": self.lane_id,
            "skip_reason": self.skip_reason,
        }


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_ROW_KINDS = ("integer_trajectory", "sequence_rows")

LANES: tuple[LaneSpec, ...] = (
    LaneSpec(
        lane_id="row_generation",
        module="sigma_theory_compiler.discovery_scheduler",
        entry_point="_stage_generate_rows",
        stage="generate_rows",
        accepts_kinds=_ROW_KINDS,
        requires_shape=(),
        resource="cpu",
        typical_cost="fast",
        emits_blockers=(
            "generator_cap_exceeded:*",
            "generator_cap_truncated:*",
            "missing_generator:*",
        ),
        preconditions=("a row generator registered in discovery_scheduler.GENERATOR_REGISTRY",),
        min_rows=1,
        note="Produces the exact rows every downstream row lane consumes.",
    ),
    LaneSpec(
        lane_id="conjecture_generation",
        module="sigma_theory_compiler.conjecture_generation",
        entry_point="generate_conjectures",
        stage="conjecture",
        accepts_kinds=_ROW_KINDS,
        requires_shape=(),
        resource="cpu",
        typical_cost="fast",
        emits_blockers=("upstream_blocked:generate_rows",),
        preconditions=("a COMPLETED row_generation receipt with >= 7 rows",),
        min_rows=7,
        note="B3: proposes typed falsifiable statements; needs a 4/3 prefix/holdout split.",
    ),
    LaneSpec(
        lane_id="basis_synthesis",
        module="sigma_theory_compiler.basis_synthesis",
        entry_point="synthesize_basis",
        stage="basis_synthesis",
        accepts_kinds=_ROW_KINDS,
        requires_shape=(),
        resource="cpu",
        typical_cost="fast",
        emits_blockers=(
            "first_blocker:no_qualifying_basis_in_declared_ladder",
            "upstream_blocked:generate_rows",
        ),
        preconditions=("a COMPLETED row_generation receipt with >= 3 rows",),
        min_rows=3,
        note="B1: the declared closed-form ladder, minimality certificate included.",
    ),
    LaneSpec(
        lane_id="nonlinear_coefficient_search",
        module="sigma_theory_compiler.nonlinear_coefficient_search",
        entry_point="search_nonlinear",
        stage="nonlinear_search",
        accepts_kinds=_ROW_KINDS,
        requires_shape=(),
        resource="cpu",
        typical_cost="fast",
        emits_blockers=(
            "first_blocker:no_qualifying_model_in_declared_set",
            "upstream_blocked:generate_rows",
        ),
        preconditions=("a COMPLETED row_generation receipt with >= 3 rows",),
        min_rows=3,
        note="B2: geometric/linear-fractional models B1's linear ladder cannot express.",
    ),
    LaneSpec(
        lane_id="structural_repair",
        module="sigma_theory_compiler.structural_repair",
        entry_point="repair_structure",
        stage="structural_repair",
        accepts_kinds=_ROW_KINDS,
        requires_shape=(),
        resource="cpu",
        typical_cost="medium",
        emits_blockers=(
            "first_blocker:no_declared_repair_recovered_structure",
            "upstream_blocked:generate_rows",
        ),
        preconditions=("a COMPLETED row_generation receipt with >= 5 rows",),
        min_rows=5,
        note="B7: basis unions and declared sub-domain restrictions after B1 blocks.",
    ),
    LaneSpec(
        lane_id="holonomic_guesser",
        module="sigma_theory_compiler.holonomic_guesser",
        entry_point="guess_receipt",
        stage="holonomic_guess",
        accepts_kinds=_ROW_KINDS,
        requires_shape=(),
        resource="cpu",
        typical_cost="medium",
        emits_blockers=("upstream_blocked:generate_rows",),
        preconditions=(
            "a COMPLETED row_generation receipt with >= 11 rows at consecutive points",
        ),
        min_rows=11,
        note="P-recursive annihilator ladder; the first cell needs order + width + 6 terms.",
    ),
    LaneSpec(
        lane_id="spectral_signal_scan",
        module="sigma_theory_compiler.spectral_signal_scan",
        entry_point="scan_receipt",
        stage="spectral_scan",
        accepts_kinds=_ROW_KINDS,
        requires_shape=(),
        resource="gpu",
        typical_cost="medium",
        emits_blockers=("upstream_blocked:generate_rows",),
        preconditions=(
            (
                "a COMPLETED row_generation receipt with >= 24 exact integer rows "
                "(16 prefix + 8 holdout)"
            ),
        ),
        min_rows=24,
        note="Discharges statement_kinds_too_weak: the quasi-periodic kind no algebraic "
        "statement kind can express.  fp64 grid is a prefilter; mpmath adjudicates.",
    ),
    LaneSpec(
        lane_id="lemma_decomposition",
        module="sigma_theory_compiler.lemma_decomposition",
        entry_point="decompose_closed_form_proof",
        stage="lemma_decomposition",
        accepts_kinds=("integer_trajectory", "module_target", "sequence_rows"),
        requires_shape=(),
        resource="cpu",
        typical_cost="fast",
        emits_blockers=(
            "missing_prover:closed_form",
            "missing_prover:linear_recurrence",
            "upstream_blocked:conjecture",
        ),
        preconditions=(
            (
                "a COMPLETED conjecture receipt carrying closed_form or "
                "linear_recurrence survivors, plus a Nat-domain polynomial closed form "
                "from basis_synthesis"
            ),
        ),
        note="B5: induction-shaped Lean obligations for closed forms.",
    ),
    LaneSpec(
        lane_id="quantified_inequality_proofs",
        module="sigma_theory_compiler.quantified_inequality_proofs",
        entry_point="prove_quantified_inequality",
        stage="quantified_inequality",
        accepts_kinds=("integer_trajectory", "module_target", "sequence_rows"),
        requires_shape=(),
        resource="cpu",
        typical_cost="fast",
        emits_blockers=(
            "missing_prover:monotonicity",
            "missing_prover:sign",
            "upstream_blocked:conjecture",
        ),
        preconditions=(
            (
                "a COMPLETED conjecture receipt carrying monotonicity or sign "
                "survivors, plus a Nat-domain polynomial closed form from basis_synthesis"
            ),
        ),
        note="B6: forall n : Nat inequalities over the Nat domain only.",
    ),
    LaneSpec(
        lane_id="gpu_counterexample_sweep",
        module="sigma_theory_compiler.gpu_counterexample_sweep",
        entry_point="sweep",
        stage="sweep",
        accepts_kinds=("diophantine_family", "integer_trajectory", "sequence_rows"),
        requires_shape=(),
        resource="gpu",
        typical_cost="slow",
        emits_blockers=(
            "missing_sweeper:*",
            "missing_sweeper:diophantine_family",
            "upstream_blocked:conjecture",
        ),
        preconditions=(
            (
                "a COMPLETED conjecture receipt with a sweepable survivor (divisibility, "
                "congruence, index_scaling_relation) and a sequence in "
                "discovery_scheduler.SWEEP_SEQUENCES"
            ),
        ),
        note="M7: exhaustive counterexample search over a declared finite range.",
    ),
    LaneSpec(
        lane_id="exponent_diophantine_sweeper",
        module="sigma_theory_compiler.exponent_diophantine_sweeper",
        entry_point="run_erdos_straus_sweep|run_beal_sweep|run_fermat_catalan_sweep",
        stage="diophantine_sweep",
        accepts_kinds=("diophantine_family",),
        requires_shape=("in_diophantine_sweeper_roster",),
        resource="gpu",
        typical_cost="slow",
        emits_blockers=(),
        preconditions=("a sealed problem queue binding the target's diophantine_family form",),
        note="Discharges missing_sweeper:diophantine_family for its declared boxes; the "
        "fp64 log-space screen is admitted only behind an a-priori resolution proof.",
    ),
    LaneSpec(
        lane_id="sat_certificate_lane",
        module="sigma_theory_compiler.sat_certificate_lane",
        entry_point="statement_from_machine_form|decide",
        stage="sat_certificate",
        accepts_kinds=(),
        requires_shape=("bounded_coloring_statement",),
        resource="cpu",
        typical_cost="medium",
        emits_blockers=(
            "CAP_TRIPPED:max_clauses",
            "CAP_TRIPPED:max_seconds",
            "CAP_TRIPPED:max_vars",
        ),
        preconditions=("a bounded_combinatorial_coloring machine form with n and k",),
        awaiting_problem_kind=True,
        note="M10: landed and queue-adapted, but problem_queue_v3 declares no "
        "bounded_combinatorial_coloring entry, so the gap is in the queue, not the lane.",
    ),
    LaneSpec(
        lane_id="inverse_symbolic_engine",
        module="sigma_theory_compiler.inverse_symbolic_engine",
        entry_point="run_pslq_lane|run_cf_lane",
        stage="inverse_symbolic",
        accepts_kinds=(),
        requires_shape=("target_constant",),
        resource="gpu",
        typical_cost="slow",
        emits_blockers=(),
        preconditions=("a named target constant with a declared verify precision",),
        awaiting_problem_kind=True,
        note="Runs the arrow backwards from a constant; problem_queue_v3 declares no "
        "target_constant machine form, so no queue entry can reach it yet.",
    ),
    LaneSpec(
        lane_id="dozen_unsolved_progress_campaign",
        module="sigma_theory_compiler.dozen_unsolved_progress_campaign",
        entry_point="build_receipt",
        stage="unsolved_progress",
        accepts_kinds=("diophantine_family", "integer_trajectory", "sequence_rows"),
        requires_shape=("in_unsolved_dozen_roster",),
        resource="cpu",
        typical_cost="medium",
        emits_blockers=(
            "bounded_multiplicity_not_expressible",
            "constant_rows_carry_no_signal",
            "criterion_dependency",
            "factorial_growth",
            "infinitude_not_sweepable",
            "literature_bound_unreachable",
            "no_termination_lane",
            "reduces_to_strong_goldbach",
            "sequential_map",
            "sieve_memory_wall",
            "statement_kinds_too_weak",
        ),
        preconditions=("the sealed queue entry, non-control and non-synthetic",),
        note="The named-gap source: one declared first_blocker per target, table-fixed "
        "so a run can never quietly rename its own obstruction.",
    ),
    LaneSpec(
        lane_id="gpu_campaign_receipt_binding",
        module="sigma_theory_compiler.discovery_scheduler",
        entry_point="_stage_note_campaigns",
        stage="note_gpu_campaign_receipts",
        accepts_kinds=("dataset_law_fit",),
        requires_shape=(),
        resource="cpu",
        typical_cost="fast",
        emits_blockers=("missing_campaign_receipts:gpu-baryonic-screen",),
        preconditions=("sealed receipts under runs/gpu-baryonic-screen",),
        note="Binds the physics screen campaigns by hash so a dataset_law_fit target is "
        "never laneless; the screens themselves are launched outside the scheduler.",
    ),
)

LANE_IDS: tuple[str, ...] = tuple(lane.lane_id for lane in LANES)

_BY_ID: dict[str, LaneSpec] = {lane.lane_id: lane for lane in LANES}

CLAIMS = {
    "lane_membership_establishes_capability_quality": False,
    "scalar_truth_or_probability_score": False,
    "skips_are_recorded_never_silent": True,
    "unattempted_lane_is_a_recorded_fact": True,
}

SCOPE = (
    "The frozen declaration of every landed discovery lane: what machine-form kinds and "
    "data shapes it accepts, which resource pool it needs, what upstream artifact it "
    "requires, and the complete vocabulary of typed blockers it may emit. Applicability "
    "is decided from the sealed queue entry alone. Registry membership asserts that a "
    "lane exists and declares its interface; it asserts nothing about the lane's output, "
    "its correctness, or the mathematics."
)


def _registry_body() -> dict[str, Any]:
    return {
        "claims": CLAIMS,
        "effective_row_cap": EFFECTIVE_ROW_CAP,
        "lanes": [lane.as_declared() for lane in LANES],
        "resources": list(RESOURCES),
        "schema_version": REGISTRY_SCHEMA,
        "scope": SCOPE,
        "shape_predicates": {
            name: {"skip_reason": reason, "statement": statement}
            for name, (reason, statement, _) in sorted(SHAPE_PREDICATES.items())
        },
        "skip_reasons": list(SKIP_REASONS),
        "typical_costs": list(COSTS),
    }


#: Content seal over the whole declaration.  Changing any lane changes this digest.
REGISTRY_CONTENT_SHA256 = canonical_sha256(_registry_body())


def registry_declaration() -> dict[str, Any]:
    """The sealed declaration, for embedding in receipts."""

    body = _registry_body()
    return {**body, "content_sha256": REGISTRY_CONTENT_SHA256}


def lane(lane_id: str) -> LaneSpec:
    """The declared lane, or a hard failure.  Never invents a lane."""

    try:
        return _BY_ID[lane_id]
    except KeyError:
        raise LaneRegistryError(f"unknown lane: {lane_id!r}") from None


def lanes_by_resource(resource: str) -> tuple[LaneSpec, ...]:
    if resource not in RESOURCES:
        raise LaneRegistryError(f"unknown resource: {resource!r}")
    return tuple(item for item in LANES if item.resource == resource)


def lane_for_stage(stage: str) -> LaneSpec | None:
    """The single lane that owns a stage name, or ``None``."""

    for item in LANES:
        if item.stage == stage:
            return item
    return None


def lanes_emitting(gap_id: str) -> tuple[str, ...]:
    """Lane ids whose declared vocabulary covers ``gap_id``.

    A declaration ending in ``:*`` covers every subject of that blocker kind, which is how
    ``missing_generator:*`` covers ``missing_generator:ulam_u_1_2`` without the registry
    having to enumerate generators it does not have.
    """

    found: list[str] = []
    for item in LANES:
        for declared in item.emits_blockers:
            if declared == gap_id:
                found.append(item.lane_id)
                break
            if declared.endswith(":*") and gap_id.startswith(declared[:-1]):
                found.append(item.lane_id)
                break
    return tuple(sorted(set(found)))


def declared_blocker_vocabulary() -> tuple[str, ...]:
    """Every blocker pattern any lane declares it may emit."""

    return tuple(sorted({item for spec in LANES for item in spec.emits_blockers}))


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


def _decide(spec: LaneSpec, entry: Mapping[str, Any]) -> LaneDecision:
    kind = entry["machine_form"]["kind"]
    if spec.accepts_kinds and kind not in spec.accepts_kinds:
        return LaneDecision(
            spec.lane_id,
            False,
            "kind_mismatch",
            f"lane accepts {list(spec.accepts_kinds)}; this entry declares kind {kind!r}",
        )
    for predicate_id in spec.requires_shape:
        reason, statement, test = SHAPE_PREDICATES[predicate_id]
        if not test(entry):
            return LaneDecision(
                spec.lane_id,
                False,
                reason,
                f"lane requires that {statement}; entry {entry['id']!r} does not satisfy it",
            )
    if spec.min_rows is not None:
        budget = declared_row_budget(entry)
        if budget is None:
            return LaneDecision(
                spec.lane_id,
                False,
                "kind_mismatch",
                f"lane needs rows; kind {kind!r} declares no row budget",
            )
        if budget < spec.min_rows:
            return LaneDecision(
                spec.lane_id,
                False,
                "insufficient_rows",
                f"lane needs >= {spec.min_rows} rows; the entry declares {budget} "
                f"(capped at {EFFECTIVE_ROW_CAP} by the host generators)",
            )
    return LaneDecision(spec.lane_id, True, None, "declared shape and row budget accepted")


def lane_decisions(entry: Mapping[str, Any]) -> tuple[LaneDecision, ...]:
    """A decision for **every** declared lane on this queue entry, lane order fixed."""

    return tuple(_decide(spec, entry) for spec in LANES)


def applicable_lanes(entry: Mapping[str, Any]) -> tuple[LaneSpec, ...]:
    """The lanes that can run on this queue entry, in declaration order."""

    return tuple(
        _BY_ID[decision.lane_id] for decision in lane_decisions(entry) if decision.applicable
    )


def skipped_lanes(entry: Mapping[str, Any]) -> tuple[LaneDecision, ...]:
    """The recorded skips: every lane that will not be attempted, and why."""

    return tuple(decision for decision in lane_decisions(entry) if not decision.applicable)


def fanout_plan(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The full (problem x lane) plan for a queue, applicable and skipped alike."""

    problems: dict[str, Any] = {}
    for entry in entries:
        decisions = lane_decisions(entry)
        problems[entry["id"]] = {
            "applicable_lanes": [d.lane_id for d in decisions if d.applicable],
            "kind": entry["machine_form"]["kind"],
            "skipped": [d.as_record() for d in decisions if not d.applicable],
        }
    counts = {
        "attempts_planned": sum(len(item["applicable_lanes"]) for item in problems.values()),
        "lanes": len(LANES),
        "problems": len(problems),
        "skips_recorded": sum(len(item["skipped"]) for item in problems.values()),
    }
    return {
        "counts": counts,
        "problems": problems,
        "registry_content_sha256": REGISTRY_CONTENT_SHA256,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def render_table() -> str:
    """Deterministic ``lane -> accepts -> resource`` table."""

    header = "| lane_id | module | accepts | resource | cost | stage |"
    rule = "| --- | --- | --- | --- | --- | --- |"
    rows = []
    for spec in LANES:
        accepts = ", ".join(spec.accepts_kinds) or "(by shape predicate)"
        if spec.requires_shape:
            accepts += " + " + ", ".join(spec.requires_shape)
        if spec.min_rows:
            accepts += f" + rows>={spec.min_rows}"
        rows.append(
            f"| `{spec.lane_id}` | `{spec.module.split('.')[-1]}` | {accepts} | "
            f"{spec.resource} | {spec.typical_cost} | `{spec.stage}` |"
        )
    return "\n".join([header, rule, *rows])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Declared discovery lane registry (A5).")
    parser.add_argument("--queue", default=None, help="sealed problem queue to plan against")
    parser.add_argument("--table", action="store_true", help="print the markdown lane table")
    parser.add_argument(
        "--validate-checked", action="store_true", help="verify the frozen registry seal"
    )
    args = parser.parse_args(argv)
    if args.validate_checked:
        recomputed = canonical_sha256(_registry_body())
        if recomputed != REGISTRY_CONTENT_SHA256:
            print(f"INVALID registry seal: {recomputed} != {REGISTRY_CONTENT_SHA256}")
            return 1
        print(f"VALID lanes={len(LANES)} content_sha256={REGISTRY_CONTENT_SHA256}")
        return 0
    if args.table:
        print(render_table())
        return 0
    if args.queue:
        from .problem_queue import load_queue

        plan = fanout_plan(load_queue(args.queue)["entries"])
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    print(json.dumps(registry_declaration(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
