"""Model-guided ordering of a framework-lattice sweep: soundness first, then the ablation.

The soundness rule outranks speed, so it is tested first and hardest.  A model may REORDER this
sweep; it may never eliminate a cell from it.  Every positive claim below has a control that must
fail beside it -- an eliminating scheduler, a duplicating one, an inventing one, an order-dependent
evaluator, a non-monotone oracle, a prompt carrying the generating rule -- because a check no input
can fail is not a check.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.funsearch_loop import (
    FunSearchError,
    MockMutationProposer,
    ProposalCall,
    SpendGovernor,
    read_ledger,
)
from sigma_theory_compiler.guided_lattice_order import (
    CLAIMS,
    FORBIDDEN_GUIDANCE_VOCABULARY,
    MUTATION_BANK,
    RANDOM_SEEDS,
    SEED_PRIORITY_SOURCE,
    AblationArms,
    CountingEvaluator,
    DuplicatingScheduler,
    EliminatingScheduler,
    GuidedScheduler,
    LatticeCell,
    LatticeDeclarationError,
    LexicographicScheduler,
    OrderChangedTheSet,
    RandomScheduler,
    ablation_report,
    as_example,
    assert_lattice_monotone,
    build_ordering_prompt,
    campaign_arms,
    campaign_report,
    canonical_lattice,
    cells_to_hit_count,
    default_governor,
    feature_row,
    non_monotone_oracle,
    order_by_scores,
    priority_scores,
    priority_source,
    retrospective_fitness,
    sweep,
    sweep_metrics,
    synthetic_dimension,
    synthetic_lattice,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

SEED = 20260818

#: The lattice is 96 cells; these are structural facts about the declaration, not measurements.
EXPECTED_CELLS = 96
EXPECTED_NON_ZERO_CELLS = 24

#: The arms that involve no floating-point arithmetic at all -- a fixed sort key, a cost sort key
#: and a seeded shuffle -- are pinned exactly.  The guided arm runs proposer-written arithmetic
#: through the sandbox, so it is asserted by inequality and structure rather than by an exact
#: number: pinning it would make an unrelated libm difference read as a soundness failure.
EXPECTED_FIXED_FIRST_HIT = 11
EXPECTED_FIXED_HALF = 60
EXPECTED_COST_FIRST_HIT = 22
EXPECTED_COST_HALF = 76
EXPECTED_RANDOM_FIRST_HIT_MEDIAN = "2"
EXPECTED_RANDOM_HALF_MEDIAN = "49"


def _cell(**overrides: object) -> LatticeCell:
    base: dict[str, object] = {
        "dimension": 4,
        "derivative_order": 2,
        "field_content": ("metric",),
        "constraints": ("derivative_order", "generally_covariant"),
        "tensor_rank": 2,
        "tensor_symmetry": "symmetric",
        "metric_signature": "lorentzian",
        "curvature": "generic",
        "concomitant_generator": "riemann_tensor",
        "cost_estimate": 48,
    }
    base.update(overrides)
    return LatticeCell.from_mapping(base)


@pytest.fixture(scope="module")
def arms(tmp_path_factory: pytest.TempPathFactory) -> AblationArms:
    """One guided sweep for the whole module.  It is the only expensive thing here."""

    ledger = tmp_path_factory.mktemp("guided-order") / "ledger.json"
    proposer = MockMutationProposer(seed=SEED, bank=MUTATION_BANK)
    return campaign_arms(proposer, default_governor(ledger, max_calls=12))


@pytest.fixture(scope="module")
def report(arms: AblationArms) -> dict:
    return campaign_report(arms)


# ---------------------------------------------------------------------------
# 1. The cell interface: a dict of framework fields plus a cost estimate.
# ---------------------------------------------------------------------------


def test_a_cell_round_trips_through_its_mapping() -> None:
    cell = _cell()
    assert LatticeCell.from_mapping(cell.to_mapping()) == cell
    assert set(cell.to_mapping()) == set(cell.framework()) | {"cost_estimate"}
    assert "cost_estimate" not in cell.framework()


def test_cell_identity_is_the_framework_and_never_the_cost_estimate() -> None:
    """A re-estimate of cost must not be able to turn one cell into two."""

    cheap = _cell(cost_estimate=48)
    expensive = _cell(cost_estimate=48_000)
    assert cheap.cell_id == expensive.cell_id
    assert cheap.cost_estimate != expensive.cost_estimate

    # ... and every declared framework field DOES move it, or the lattice would collapse cells
    # that declare different searches into one.
    for name, value in (
        ("dimension", 5),
        ("derivative_order", 4),
        ("field_content", ("metric", "scalar")),
        ("constraints", ("derivative_order", "generally_covariant", "symmetric")),
        ("tensor_rank", 4),
        ("tensor_symmetry", "antisymmetric"),
        ("metric_signature", "riemannian"),
        ("curvature", "flat"),
        ("concomitant_generator", "unit_timelike_vector"),
    ):
        assert _cell(**{name: value}).cell_id != cheap.cell_id, name


def test_a_lattice_refuses_two_cells_with_the_same_framework() -> None:
    """The control for the identity claim: same framework, different cost, must be refused."""

    with pytest.raises(LatticeDeclarationError, match="same framework"):
        canonical_lattice([_cell(cost_estimate=48), _cell(cost_estimate=96)])


@pytest.mark.parametrize(
    "overrides",
    [
        {"dimension": 0},
        {"derivative_order": -2},
        {"tensor_rank": True},
        {"tensor_symmetry": ""},
        {"field_content": ()},
        {"field_content": ("scalar", "metric")},
        {"constraints": ("symmetric", "symmetric")},
        {"cost_estimate": 0},
    ],
)
def test_an_ill_declared_cell_is_refused(overrides: dict) -> None:
    with pytest.raises(LatticeDeclarationError):
        _cell(**overrides)


def test_a_cell_mapping_with_a_missing_or_extra_key_is_refused() -> None:
    mapping = _cell().to_mapping()
    del mapping["curvature"]
    with pytest.raises(LatticeDeclarationError, match="missing"):
        LatticeCell.from_mapping(mapping)
    mapping = _cell().to_mapping()
    mapping["jet_seed"] = "smuggled"
    with pytest.raises(LatticeDeclarationError, match="unexpected"):
        LatticeCell.from_mapping(mapping)


# ---------------------------------------------------------------------------
# 2. The synthetic lattice and the two monotonicity directions.
# ---------------------------------------------------------------------------


def test_the_synthetic_lattice_has_the_declared_shape() -> None:
    cells = synthetic_lattice()
    assert len(cells) == EXPECTED_CELLS
    assert len({cell.cell_id for cell in cells}) == EXPECTED_CELLS
    assert sum(1 for cell in cells if synthetic_dimension(cell) > 0) == EXPECTED_NON_ZERO_CELLS
    # Non-zero cells exist at every constraint depth including the deepest, so "reach the
    # non-trivial cells" is not secretly "reach the under-constrained corner".
    depths = {len(cell.constraints) for cell in cells if synthetic_dimension(cell) > 0}
    assert depths == {2, 3, 4, 5}


def test_the_synthetic_oracle_is_monotone_in_both_declared_directions() -> None:
    measured = assert_lattice_monotone(synthetic_lattice(), synthetic_dimension)
    assert measured["constraint_axis_non_increasing"] is True
    assert measured["field_content_axis_non_decreasing"] is True
    assert measured["constraint_comparisons"] > 0
    assert measured["field_content_comparisons"] > 0


def test_the_monotonicity_measurement_can_fail() -> None:
    """The control.  An oracle that rewards a constraint must be caught, or the check is scenery."""

    with pytest.raises(Exception, match="monotonicity violated on the constraint axis"):
        assert_lattice_monotone(synthetic_lattice(), non_monotone_oracle)


# ---------------------------------------------------------------------------
# 3. The soundness rule: ordering only, never elimination.
# ---------------------------------------------------------------------------


def test_order_cannot_change_the_set_of_results(arms: AblationArms) -> None:
    """The headline control: the same lattice under four different orders, one set of results."""

    cells = synthetic_lattice()
    identifiers = {cell.cell_id for cell in cells}
    digests = set()
    for run in arms.every():
        assert run.evaluations == EXPECTED_CELLS
        assert len(run.visit_order) == EXPECTED_CELLS
        assert set(run.visit_order) == identifiers
        assert len(set(run.visit_order)) == EXPECTED_CELLS
        assert run.results == arms.fixed.results
        digests.add(run.results_sha256())
    assert len(digests) == 1

    # ... and the truth the equality is being measured against, computed without any scheduler.
    assert arms.fixed.results == {cell.cell_id: synthetic_dimension(cell) for cell in cells}


def test_the_orders_really_were_different(arms: AblationArms) -> None:
    """Teeth for the test above.  If guidance never moved a cell, the equality proves nothing."""

    assert arms.guided.order_sha256() != arms.fixed.order_sha256()
    assert arms.cost.order_sha256() != arms.fixed.order_sha256()
    assert arms.randoms[0].order_sha256() != arms.fixed.order_sha256()
    assert arms.guided.diagnostics["champion_is_still_the_seed"] is False
    # The guided arm diverges from the fixed order only after the declared warm-up.
    warmup = arms.guided.diagnostics["warmup_cells"]
    assert arms.guided.visit_order[:warmup] == arms.fixed.visit_order[:warmup]
    assert arms.guided.visit_order[warmup:] != arms.fixed.visit_order[warmup:]


def test_a_scheduler_that_prunes_cannot_complete_a_sweep() -> None:
    """The control the whole module exists for: an unsound prune must not be servable."""

    with pytest.raises(OrderChangedTheSet, match="dropped"):
        sweep(synthetic_lattice(), EliminatingScheduler(), synthetic_dimension)


def test_a_scheduler_that_duplicates_a_cell_cannot_complete_a_sweep() -> None:
    with pytest.raises(OrderChangedTheSet, match="duplicates=True"):
        sweep(synthetic_lattice(), DuplicatingScheduler(), synthetic_dimension)


def test_a_scheduler_that_invents_a_cell_cannot_complete_a_sweep() -> None:
    foreign = _cell(dimension=11, cost_estimate=1)

    class Inventing:
        scheduler_id = "inventing_control"

        def order(self, pending, measured):
            return [foreign, *sorted(pending, key=LatticeCell.sort_key)[:-1]]

    with pytest.raises(OrderChangedTheSet, match="invented"):
        sweep(synthetic_lattice(), Inventing(), synthetic_dimension)


def test_the_evaluator_refuses_to_answer_the_same_cell_twice() -> None:
    counter = CountingEvaluator(synthetic_dimension)
    cell = synthetic_lattice()[0]
    assert counter(cell) == synthetic_dimension(cell)
    with pytest.raises(OrderChangedTheSet, match="twice"):
        counter(cell)


def test_the_order_invariance_control_can_itself_fail() -> None:
    """Without this, `results are identical` might just mean `nothing was measured`.

    An evaluator whose answer depends on how many cells came before it is exactly the bug the
    invariance control is supposed to notice.  It must notice.
    """

    state = {"seen": 0}

    def order_dependent(cell: LatticeCell) -> int:
        state["seen"] += 1
        return synthetic_dimension(cell) + state["seen"] % 2

    cells = synthetic_lattice()
    guided_like = sweep(cells, RandomScheduler(seed=7), CountingEvaluator(order_dependent))
    state["seen"] = 0
    fixed = sweep(cells, LexicographicScheduler(), CountingEvaluator(order_dependent))
    assert guided_like.results_sha256() != fixed.results_sha256()

    state["seen"] = 0
    faulty = AblationArms(
        guided=guided_like,
        fixed=fixed,
        cost=fixed,
        randoms=(),
        random_seeds=(),
        cells=len(cells),
        warmup_cells=0,
    )
    with pytest.raises(OrderChangedTheSet, match="order\ndependent|order dependent"):
        ablation_report(faulty)


# ---------------------------------------------------------------------------
# 4. Priority functions: exact, sandboxed, and never able to skip a cell.
# ---------------------------------------------------------------------------


def test_a_priority_function_returns_rationals_that_order_exactly_as_the_doubles_did() -> None:
    """Not the exact binary double -- the exact ORDER of them, which is all ordering needs.

    Seventeen significant digits round-trip a double uniquely, so the lift is injective, and
    rounding to a fixed digit count is monotone; injective plus monotone is strictly monotone.
    """

    cells = synthetic_lattice()[:12]
    outcome, scores = priority_scores(priority_source("return 0.1 * dimension - cost_estimate"), cells)
    assert outcome.ok, outcome.reason
    assert scores is not None

    doubles = {cell.cell_id: 0.1 * cell.dimension - cell.cost_estimate for cell in cells}
    for cell in cells:
        value = scores[cell.cell_id]
        assert isinstance(value, Fraction)
        assert float(value) == doubles[cell.cell_id]  # injective: it round-trips
    assert len(set(scores.values())) == len(set(doubles.values()))  # no two doubles collided
    assert [c.cell_id for c in order_by_scores(cells, scores)] == [
        cell.cell_id
        for cell in sorted(cells, key=lambda c: (-doubles[c.cell_id], c.cell_id))
    ]


def test_the_seed_priority_function_is_uninformative() -> None:
    """If the seed already knew where the answers were, the ablation would measure the author."""

    cells = synthetic_lattice()
    outcome, scores = priority_scores(SEED_PRIORITY_SOURCE, cells)
    assert outcome.ok
    assert scores is not None
    assert set(scores.values()) == {Fraction(0)}
    # A constant score leaves the order fixed by the identity tie-break: total and reproducible.
    assert order_by_scores(cells, scores) == tuple(sorted(cells, key=lambda c: c.cell_id))


def test_a_priority_function_that_misbehaves_is_a_typed_failure_not_an_exception() -> None:
    cells = synthetic_lattice()[:4]
    for body, reason in (
        ("return dimension / 0", "runtime_error"),
        ("return 'high'", "wrong_output_shape"),
        ("return open('x')", "static_screen_denied"),
        ("return 1e308 * 1e308", "non_finite_output"),
    ):
        outcome, scores = priority_scores(priority_source(body), cells)
        assert not outcome.ok and scores is None
        assert outcome.reason == reason, body


def test_features_are_a_function_of_the_declaration_only() -> None:
    cell = _cell(field_content=("metric", "unit_timelike_vector"), cost_estimate=144)
    row = feature_row(cell)
    assert row == (4.0, 2.0, 2.0, 2.0, 144.0, 1.0, 0.0, 0.0)
    # Same declaration, same features -- there is no channel from the measured dimension into them.
    assert feature_row(cell) == feature_row(LatticeCell.from_mapping(cell.to_mapping()))


def test_retrospective_fitness_counts_the_leading_block() -> None:
    measured = {"a": 3, "b": 0, "c": 5, "d": 0}
    perfect = {"a": Fraction(9), "c": Fraction(8), "b": Fraction(1), "d": Fraction(0)}
    assert retrospective_fitness(perfect, measured) == 2
    inverted = {"a": Fraction(0), "c": Fraction(1), "b": Fraction(9), "d": Fraction(8)}
    assert retrospective_fitness(inverted, measured) == 0
    assert retrospective_fitness(perfect, {"b": 0, "d": 0}) == 0


# ---------------------------------------------------------------------------
# 5. The prompt guard.
# ---------------------------------------------------------------------------


def test_the_guidance_prompt_carries_the_declaration_and_the_measurements() -> None:
    cells = synthetic_lattice()[:3]
    prompt = build_ordering_prompt([(cell, synthetic_dimension(cell)) for cell in cells], [])
    assert "signature:" in prompt
    for name in ("dimension", "constraint_count", "divergence_free"):
        assert name in prompt


def test_the_prompt_carries_no_declaration_free_text_at_all() -> None:
    """The first line of defence is that the prompt is NUMERIC: nothing to leak, not a filter."""

    cell = synthetic_lattice()[0]
    prompt = build_ordering_prompt([(cell, synthetic_dimension(cell))], [])
    for text in (cell.concomitant_generator, cell.metric_signature, cell.curvature, cell.label):
        assert text not in prompt


def test_the_guidance_prompt_refuses_the_generating_rule() -> None:
    """The control the guard exists for: an example program echoed back into the next prompt.

    Example sources ARE embedded verbatim, so they are the one channel by which text -- rather than
    a feature row -- reaches the proposer.  A source naming the rule that produced the measurements
    must take the prompt down rather than be forwarded.
    """

    leaking = as_example(priority_source("return dimension * supply_weight"), 1)
    with pytest.raises(FunSearchError, match="forbidden vocabulary"):
        build_ordering_prompt([(synthetic_lattice()[0], 3)], [leaking])
    assert "supply" in FORBIDDEN_GUIDANCE_VOCABULARY

    # ... and the guard does not fire on an honest example, or it would take every run down.
    honest = as_example(priority_source("return dimension - constraint_count"), 1)
    assert "constraint_count" in build_ordering_prompt([(synthetic_lattice()[0], 3)], [honest])


# ---------------------------------------------------------------------------
# 6. The spend governor, and what happens when it binds.
# ---------------------------------------------------------------------------


def test_every_proposal_call_is_charged_to_the_governor(report: dict) -> None:
    governor = report["guidance"]["spend_governor"]
    calls = report["guidance"]["proposal_calls"]
    assert governor["ledger_schema"] == "invariant-llm-spend-ledger-1.0"
    assert governor["calls"] == len(calls) >= 1
    assert governor["charged_dollars_hundredths"] == governor["calls"]
    assert governor["closing_ledger_hundredths"] == governor["calls"]
    assert read_ledger(governor["ledger_path"]) == governor["calls"]
    assert {call["proposer_id"] for call in calls} == {"deterministic_mock_mutator"}
    assert all(call["ok"] for call in calls)


def test_a_bound_spend_cap_costs_ordering_and_never_a_cell(tmp_path) -> None:
    """When the money runs out the sweep must still be complete.  Skipping is not a fallback."""

    cells = synthetic_lattice()
    starved = GuidedScheduler(
        proposer=MockMutationProposer(seed=SEED, bank=MUTATION_BANK),
        governor=default_governor(tmp_path / "ledger.json", max_calls=1),
        warmup_cells=12,
        refit_period=8,
        proposals_per_call=2,
    )
    run = sweep(cells, starved, CountingEvaluator(synthetic_dimension))
    assert run.evaluations == EXPECTED_CELLS
    assert run.results == {cell.cell_id: synthetic_dimension(cell) for cell in cells}
    halts = [record.get("halt_reason") for record in run.diagnostics["refits"]]
    assert "call_cap_reached" in halts
    assert run.diagnostics["spend_governor"]["calls"] == 1


def test_a_dead_proposer_degrades_to_the_fixed_order(tmp_path) -> None:
    """A proposer that never answers must cost ordering quality and nothing else."""

    class DeadProposer:
        proposer_id = "dead_control"

        def propose(self, prompt, examples, count):
            return ProposalCall(self.proposer_id, "", 0, 0, False, "provider_unavailable", "")

        def programs(self):
            return ()

    cells = synthetic_lattice()
    scheduler = GuidedScheduler(
        proposer=DeadProposer(),
        governor=default_governor(tmp_path / "ledger.json", max_calls=12),
    )
    run = sweep(cells, scheduler, CountingEvaluator(synthetic_dimension))
    fixed = sweep(cells, LexicographicScheduler(), CountingEvaluator(synthetic_dimension))
    assert run.results_sha256() == fixed.results_sha256()
    assert run.diagnostics["champion_is_still_the_seed"] is True
    assert {record.get("proposer_failure") for record in run.diagnostics["refits"]} == {
        "provider_unavailable"
    }


def test_the_governor_refuses_an_impossible_envelope(tmp_path) -> None:
    with pytest.raises(FunSearchError, match="caps must be positive"):
        SpendGovernor(
            ledger_path=tmp_path / "ledger.json",
            max_calls=0,
            max_dollars_hundredths=1,
            charge_per_call_hundredths=1,
        )


# ---------------------------------------------------------------------------
# 7. The ablation.
# ---------------------------------------------------------------------------


def test_the_fixed_and_cost_arms_are_pinned_exactly(arms: AblationArms) -> None:
    """No floating point anywhere in these two orders, so they are pinned to the digit."""

    fixed = sweep_metrics(arms.fixed, quarter=24)
    cost = sweep_metrics(arms.cost, quarter=24)
    assert fixed["cells_to_first_hit"] == EXPECTED_FIXED_FIRST_HIT
    assert fixed["cells_to_half_the_hits"] == EXPECTED_FIXED_HALF
    assert cost["cells_to_first_hit"] == EXPECTED_COST_FIRST_HIT
    assert cost["cells_to_half_the_hits"] == EXPECTED_COST_HALF
    assert fixed["non_zero_cells"] == cost["non_zero_cells"] == EXPECTED_NON_ZERO_CELLS
    assert fixed["half_target"] == 12


def test_the_random_arm_is_pinned_exactly(report: dict) -> None:
    random_arm = report["arms"]["random"]
    assert random_arm["seeds"] == list(RANDOM_SEEDS)
    assert len(RANDOM_SEEDS) % 2 == 1, "an odd seed count keeps the median an integer"
    assert random_arm["cells_to_first_hit"]["median"] == EXPECTED_RANDOM_FIRST_HIT_MEDIAN
    assert random_arm["cells_to_half_the_hits"]["median"] == EXPECTED_RANDOM_HALF_MEDIAN
    assert random_arm["cells_to_first_hit"]["runs"] == len(RANDOM_SEEDS)


def test_the_ablation_reports_both_metrics_for_all_three_arms(report: dict) -> None:
    arms_block = report["arms"]
    assert set(arms_block) == {"guided", "fixed_lexicographic", "cost_ascending", "random"}
    for name in ("guided", "fixed_lexicographic", "cost_ascending"):
        block = arms_block[name]
        assert block["evaluations"] == EXPECTED_CELLS
        assert block["non_zero_cells"] == EXPECTED_NON_ZERO_CELLS
        assert isinstance(block["cells_to_first_hit"], int)
        assert isinstance(block["cells_to_half_the_hits"], int)
    for metric in ("cells_to_first_hit", "cells_to_half_the_hits"):
        spread = arms_block["random"][metric]
        assert spread["minimum"] <= spread["maximum"]
        assert set(spread) == {"runs", "minimum", "median", "mean", "maximum"}


def test_the_measured_verdict_matches_the_measured_numbers(report: dict) -> None:
    """The verdict is assembled from the counts.  Here it is confronted with them again."""

    guided = report["arms"]["guided"]
    fixed = report["arms"]["fixed_lexicographic"]
    verdict = report["verdict"]
    assert verdict["guided_beats_fixed_on_half_the_hits"] == (
        guided["cells_to_half_the_hits"] < fixed["cells_to_half_the_hits"]
    )
    assert verdict["guided_beats_fixed_on_first_hit"] == (
        guided["cells_to_first_hit"] < fixed["cells_to_first_hit"]
    )
    comparison = report["guided_versus_random"]["cells_to_half_the_hits"]
    assert verdict["guided_beats_random_on_half_the_hits"] == (
        2 * comparison["guided_strictly_better_in"] > comparison["comparable_runs"]
    )
    assert verdict["guided_beats_random_on_both_metrics"] == (
        verdict["guided_beats_random_on_first_hit"]
        and verdict["guided_beats_random_on_half_the_hits"]
    )


def test_the_measured_outcome_on_this_lattice(report: dict) -> None:
    """The honest answer, pinned in the direction it came out.

    Guidance beats the fixed lexicographic scan comfortably and beats a random shuffle by a margin
    too thin to build on: it loses cells-to-first-hit outright, and the front-loading metric --
    non-zero cells found inside a fixed quarter-lattice budget -- puts it BELOW the random median.
    That is the measurement; if a change to the proposer or the fitness moves it, this assertion is
    what will say so.
    """

    guided = report["arms"]["guided"]
    fixed = report["arms"]["fixed_lexicographic"]
    verdict = report["verdict"]

    assert guided["cells_to_half_the_hits"] < fixed["cells_to_half_the_hits"]
    assert verdict["guided_beats_random_on_first_hit"] is False
    assert verdict["guided_beats_random_on_both_metrics"] is False
    # The first-hit loss is structural, not a tuning accident: the warm-up runs the fixed order.
    assert guided["cells_to_first_hit"] == fixed["cells_to_first_hit"]
    assert verdict["warmup_cells"] >= fixed["cells_to_first_hit"]
    # Front-loading, at equal budget, against the random median.
    front = guided["hits_in_the_first_quarter"]
    assert front > fixed["hits_in_the_first_quarter"]
    assert Fraction(front) < Fraction(report["arms"]["random"]["hits_in_the_first_quarter"]["median"])
    assert "thin one" in verdict["summary"] or "did NOT beat the random order" in verdict["summary"]


def test_cells_to_hit_count_reads_the_visit_order(arms: AblationArms) -> None:
    run = arms.fixed
    assert cells_to_hit_count(run, 1) == EXPECTED_FIXED_FIRST_HIT
    assert cells_to_hit_count(run, EXPECTED_NON_ZERO_CELLS) <= EXPECTED_CELLS
    assert cells_to_hit_count(run, EXPECTED_NON_ZERO_CELLS + 1) is None
    with pytest.raises(Exception, match="at least one"):
        cells_to_hit_count(run, 0)


# ---------------------------------------------------------------------------
# 8. The receipt.
# ---------------------------------------------------------------------------


def test_the_receipt_is_canonical_and_carries_no_floats(report: dict) -> None:
    """``canonical_sha256`` raises on a float, so sealing the receipt IS the exactness check."""

    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    assert canonical_sha256(body) == report["receipt_sha256"]
    assert report["claims"] == CLAIMS
    assert report["claims"]["guidance_quality_is_measured_not_claimed"] is True
    assert "guided_beats_random" not in report["claims"]
    assert report["lattice_monotonicity"]["constraint_axis_non_increasing"] is True
    assert report["guided_order_differs_from_fixed"] is True


def test_the_receipt_is_reproducible(tmp_path) -> None:
    """Same seed, same ledger start, same bytes.  Two independent campaigns are compared."""

    def once(name: str) -> dict:
        proposer = MockMutationProposer(seed=SEED, bank=MUTATION_BANK)
        governor = default_governor(tmp_path / f"{name}.json", max_calls=3)
        return campaign_report(
            campaign_arms(proposer, governor, warmup_cells=12, refit_period=24, proposals_per_call=2)
        )

    first = once("a")
    second = once("b")
    assert first["arms"] == second["arms"]
    assert first["order_digests"] == second["order_digests"]
    assert first["guidance"]["champion_priority_sha256"] == (
        second["guidance"]["champion_priority_sha256"]
    )


def test_the_module_binds_its_own_paths(report: dict) -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / report["source_path"]).is_file()
    assert (root / report["test_path"]).is_file()
    assert report["test_path"].endswith(Path(__file__).name)
