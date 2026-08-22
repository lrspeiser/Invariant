"""Gates for a measurement whose first version was refuted by its own verifier.

The claim under test is not "the FunSearch loop learns".  It is narrower and it is what C6
actually asks for: *if* the surviving programs steer the proposer, this instrument sees it,
and if they do not, it says so.  Four things could make that vacuous, and there is a gate
for each.

*The null could be anti-conservative.*  It was.  v1's only null permuted generations while
holding the score axis fixed, so it tested whether the proposal distribution drifted and
never put the score axis at risk.  :func:`test_the_v1_null_is_anti_conservative_and_the_new
_ones_are_not` builds a stream that drifts hard with scores assigned at random -- so every
firing is by construction a false positive -- and measures the rate for all three nulls.
N1's rate must be far above alpha and N2's and N3's must not be.

*A null could be fooled by a covariate.*  In a campaign that improves, high scores cluster
late, and nearness in time can look exactly like alignment with score.
:func:`test_n3_removes_a_temporal_confound_that_n2_falls_for` builds a pool where features
depend only on generation and score depends only on generation: N2 fires, and N3, which
randomises the score labels *within* each generation, must not.

*The instrument could simply be dead.*  A null that never fires reports "no learning" on
every input, and that is an uninformative null, not a negative result.  The power gates
build the alternative hypothesis -- an early window replaced by the low-scoring pole and a
late window by the high-scoring one -- and require both admissible nulls to fire on it.

*The arithmetic could be floating point wearing a certificate's clothes.*  The statistic is
one :class:`~fractions.Fraction` built from two integers; the gates check it against
hand-computed rationals, check its bound and its exact antisymmetry under swapping the
poles, and walk the sealed receipt asserting no float ever reaches it.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import random
import tempfile
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import funsearch_learning as L
from sigma_theory_compiler import funsearch_loop as fl

#: Small enough to run in a test, large enough to carry two windows and a middle.
FAST = L.CampaignConfig(
    problem_id="blinded_response_law",
    generations=60,
    islands=4,
    proposals_per_call=5,
    window=10,
    pole_size=12,
    seed=7,
    sweep_seeds=(7, 8),
    draws=499,
    calibration_trials=40,
    calibration_draws=99,
    sandbox_wall_seconds=4.0,
)

RECEIPT = Path("runs/math/funsearch/learning-v2.json")
ALPHA = Fraction(1, 20)
WIDTH = len(L.FEATURE_ALPHABET)


def vec(**counts: int) -> L.Vector:
    """A count vector written by symbol name, so a test reads as arithmetic."""

    out = [0] * WIDTH
    for symbol, value in counts.items():
        out[L.FEATURE_ALPHABET.index(symbol.replace("__", ":"))] = value
    return tuple(out)


@pytest.fixture(scope="module")
def arms() -> tuple[L.ArmMeasurement, L.ArmMeasurement]:
    """Both arms of one campaign.  Expensive, so the whole module shares it."""

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        live = L.measure_arm(FAST, selection_pressure=True, ledger_path=root / "on.json")
        ablated = L.measure_arm(FAST, selection_pressure=False, ledger_path=root / "off.json")
    return live, ablated


# ---------------------------------------------------------------------------
# The feature alphabet is closed
# ---------------------------------------------------------------------------


def test_the_alphabet_is_sorted_unique_and_indexable() -> None:
    assert len(set(L.FEATURE_ALPHABET)) == len(L.FEATURE_ALPHABET)
    assert list(L.FEATURE_ALPHABET) == sorted(L.FEATURE_ALPHABET)
    assert len(L.FEATURE_ALPHABET) > 40


def test_no_program_can_emit_a_symbol_outside_the_alphabet() -> None:
    """A distribution is only a distribution if the symbol table is closed."""

    sources = [
        "def f(u):\n    return u @ u\n",
        "def f(u):\n    return u if u > 0 else -u\n",
        "def f(u):\n    return [z for z in range(3)]\n",
        "def f(u):\n    return {'a': u}\n",
        "def f(u):\n    import os\n    return os.getpid()\n",
        "async def f(u):\n    return await u\n",
        "def f(u):\n    return b'bytes'\n",
        "def f(u):\n    return u is not None\n",
        "def f(u):\n    yield u\n",
        "def f(u):\n    lam = lambda q: q ** 2\n    return lam(u)\n",
    ]
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            for symbol in L._symbols_of(node):
                assert symbol in L._INDEX, f"{symbol} escaped the alphabet"


def test_unparseable_and_featureless_source_is_refused_not_folded_in() -> None:
    assert L.program_features("def f(:\n") is None
    assert L.program_features("") is None
    assert L.program_features("pass\n") is None
    assert L.program_features("def f(u):\n    return u\n") is not None


def test_features_are_counts_of_what_is_actually_there() -> None:
    features = L.program_features("def f(u):\n    return u * u + math.sqrt(u)\n")
    assert features is not None
    counted = {
        L.FEATURE_ALPHABET[position]: value for position, value in enumerate(features) if value
    }
    assert counted["op:Mult"] == 1
    assert counted["op:Add"] == 1
    # the parameter is an ast.arg, not an ast.Name, so only the three uses are counted
    assert counted["name:u"] == 3
    assert counted["name:math"] == 1
    assert counted["attr:sqrt"] == 1
    assert counted["node:Call"] == 1
    assert counted["node:Return"] == 1


# ---------------------------------------------------------------------------
# The statistic is exact, bounded, and means what it says
# ---------------------------------------------------------------------------


def test_total_variation_is_an_exact_rational_and_hits_both_endpoints() -> None:
    left = vec(op__Add=3, op__Mult=1)
    assert L.total_variation(left, left) == Fraction(0)
    assert L.total_variation(vec(op__Add=1), vec(op__Mult=1)) == Fraction(1)
    value = L.total_variation(vec(op__Add=3, op__Mult=1), vec(op__Add=1, op__Mult=1))
    assert isinstance(value, Fraction)
    # |3*2 - 1*4| + |1*2 - 1*4| = 2 + 2 = 4, over 2*4*2 = 16.
    assert value == Fraction(4, 16)


def test_the_projection_matches_the_declared_formula_by_hand() -> None:
    start = vec(op__Add=3, op__Mult=1)  # S = 4
    end = vec(op__Add=1, op__Mult=3)  # E = 4
    elite = vec(op__Add=1, op__Mult=3)  # P = 4
    foil = vec(op__Add=3, op__Mult=1)  # Q = 4
    # d = end*S - start*E = (4 - 12, 12 - 4) = (-8, 8); a = elite*Q - foil*P = (-8, 8).
    # T = (64 + 64) / (4*4 * 16) = 128 / 256 = 1/2.
    assert L.projection(start, end, elite, foil) == Fraction(1, 2)


def test_the_projection_is_antisymmetric_in_the_two_poles() -> None:
    start, end = vec(op__Add=3, op__Mult=1), vec(op__Add=1, op__Mult=3)
    elite, foil = vec(op__Add=1, op__Mult=5), vec(op__Add=7, op__Mult=1)
    forward = L.projection(start, end, elite, foil)
    assert L.projection(start, end, foil, elite) == -forward


def test_the_projection_is_invariant_to_the_size_of_a_pole() -> None:
    """The normalisation is what makes a drawn pole pair comparable to the real one.

    Doubling every count in a pole leaves the distribution it induces unchanged, so the
    statistic must not move.  An unnormalised statistic would.
    """

    start, end = vec(op__Add=3, op__Mult=1), vec(op__Add=1, op__Mult=3)
    elite, foil = vec(op__Add=1, op__Mult=5), vec(op__Add=7, op__Mult=1)
    doubled = tuple(2 * value for value in elite)
    assert L.projection(start, end, doubled, foil) == L.projection(start, end, elite, foil)


def test_a_drift_orthogonal_to_the_score_axis_scores_exactly_zero() -> None:
    """Generic drift only counts when it points somewhere score cares about.

    ``start`` and ``end`` differ -- the distribution genuinely moved -- but they moved
    equally on the two coordinates the score axis separates, so the projection is exactly
    zero rather than nearly zero.  This is the property the whole design rests on, and it is
    checkable in integers.
    """

    start = vec(op__Add=2, op__Mult=2, name__u=4)  # p = (1/4, 1/4, 1/2)
    end = vec(op__Add=3, op__Mult=3, name__u=2)  # p = (3/8, 3/8, 1/4)
    elite = vec(op__Add=3, op__Mult=1)  # p = (3/4, 1/4, 0)
    foil = vec(op__Add=1, op__Mult=3)  # p = (1/4, 3/4, 0)
    assert L.total_variation(start, end) > 0
    # D = (1/8, 1/8, -1/4) and A = (1/2, -1/2, 0), so <D, A> = 0 exactly.
    assert L.projection(start, end, elite, foil) == Fraction(0)
    assert L.projection(start, start, elite, foil) == Fraction(0)


def test_the_projection_is_bounded_by_the_largest_coordinate_of_the_drift() -> None:
    rng = random.Random(4242)
    for _ in range(200):

        def draw() -> L.Vector:
            return tuple(rng.randrange(0, 6) for _ in range(WIDTH))

        start, end, elite, foil = draw(), draw(), draw(), draw()
        if min(sum(start), sum(end), sum(elite), sum(foil)) <= 0:
            continue
        drift, scale = L.drift_numerator(start, end)
        axis = L.axis_numerator(elite, foil)
        if sum(abs(value) for value in axis) == 0:
            continue
        statistic = L.projection_from_parts(drift, scale, axis)
        bound = Fraction(max(abs(value) for value in drift), scale)
        assert abs(statistic) <= bound <= 1


def test_the_statistic_refuses_a_degenerate_input_rather_than_returning_a_number() -> None:
    good = vec(op__Add=1)
    empty = tuple(0 for _ in range(WIDTH))
    with pytest.raises(L.LearningError):
        L.drift_numerator(empty, good)
    with pytest.raises(L.LearningError):
        L.axis_numerator(good, empty)
    with pytest.raises(L.LearningError):
        L.projection(good, good, good, good)  # both poles equal: the axis is empty


# ---------------------------------------------------------------------------
# The nulls: calibration, and the finding that produced this file
# ---------------------------------------------------------------------------


def _synthetic_stream(
    *,
    generations: int,
    per_generation: int,
    drift_per_generation: int,
    noise_seed: int,
) -> list[tuple[int, list[L.Vector]]]:
    """A stream whose distribution walks steadily along one coordinate."""

    rng = random.Random(noise_seed)
    out: list[tuple[int, list[L.Vector]]] = []
    for generation in range(generations):
        group: list[L.Vector] = []
        for _ in range(per_generation):
            group.append(
                vec(
                    op__Add=20 + drift_per_generation * generation + rng.randrange(0, 5),
                    op__Mult=20
                    - drift_per_generation * generation
                    + rng.randrange(0, 5)
                    + drift_per_generation * generations,
                    name__u=10 + rng.randrange(0, 5),
                )
            )
        out.append((generation, group))
    return out


def _fire_rates_on_score_blind_labels(trials: int) -> dict[str, int]:
    """Run all three nulls on a hard-drifting stream whose scores are assigned at random.

    Scores carry no information about features by construction, so every firing is a false
    positive and the count is a false-positive rate for the null that produced it.
    """

    stream = _synthetic_stream(
        generations=40, per_generation=5, drift_per_generation=2, noise_seed=99
    )
    start_groups = [L.pooled(group) for _, group in stream[:8]]
    end_groups = [L.pooled(group) for _, group in stream[-8:]]
    drift, scale = L.drift_numerator(L.pooled(start_groups), L.pooled(end_groups))
    fired = {"n1": 0, "n2": 0, "n3": 0}
    for trial in range(trials):
        label = random.Random(1000 + trial)
        pool = [
            L.PoolProgram(
                generation=generation,
                program_sha256=f"{generation:03d}{index:03d}",
                final_score=label.random(),  # independent of features, by construction
                features=features,
            )
            for generation, group in stream[8:-8]
            for index, features in enumerate(group)
        ]
        elite_items, foil_items = L.poles_of(pool, 12)
        axis = L.axis_numerator(
            L.pooled(item.features for item in elite_items),
            L.pooled(item.features for item in foil_items),
        )
        statistic = L.projection_from_parts(drift, scale, axis)
        results = {
            "n1": L._seal(
                "n1",
                "",
                False,
                statistic,
                L.generation_permutation_draws(
                    start_groups, end_groups, axis, draws=199, seed=7 + trial
                ),
                ALPHA,
            ),
            "n2": L._seal(
                "n2",
                "",
                True,
                statistic,
                L._axis_draws(drift, scale, L._random_axes(pool, 12, draws=199, seed=17 + trial)),
                ALPHA,
            ),
            "n3": L._seal(
                "n3",
                "",
                True,
                statistic,
                L._axis_draws(
                    drift,
                    scale,
                    L._stratified_axes(pool, elite_items, foil_items, draws=199, seed=27 + trial),
                ),
                ALPHA,
            ),
        }
        for key, result in results.items():
            fired[key] += int(result.fires)
    return fired


def test_the_v1_null_is_anti_conservative_and_the_two_new_ones_are_not() -> None:
    """The finding this module exists to record, as an assertion.

    A stream that drifts hard, with ``final_score`` assigned by a generator that never looks
    at a feature.  There is nothing here for a null to find, so the whole of N1's firing rate
    is error.  The control that must fail is on the same line: N2 and N3 see the identical
    data and must stay at their declared alpha.
    """

    trials = 60
    fired = _fire_rates_on_score_blind_labels(trials)
    assert fired["n1"] >= trials // 4, (
        "N1 was expected to be anti-conservative on a drifting score-blind stream; "
        f"it fired {fired['n1']}/{trials}"
    )
    assert fired["n2"] <= trials // 5, f"N2 fired {fired['n2']}/{trials}"
    assert fired["n3"] <= trials // 5, f"N3 fired {fired['n3']}/{trials}"
    assert fired["n1"] > 3 * max(fired["n2"], fired["n3"], 1)


def _temporal_confound_pool() -> tuple[list[L.Vector], list[L.Vector], list[L.PoolProgram], int]:
    """A pool where features depend only on generation and score depends only on generation.

    Within a generation, ``final_score`` and features are unrelated: the score ordering is
    the program's index and the feature perturbation is a fixed shuffle that is not that
    index.  So there is no within-generation score signal to find, and any null that reports
    one is reading the clock.
    """

    noise = [(0, 0), (2, 1), (1, 3), (3, 0), (0, 2), (1, 1)]
    per_generation = 6

    def features_of(generation: int, index: int) -> L.Vector:
        extra = noise[(generation * 5 + index * 3) % len(noise)]
        return vec(
            op__Add=40 - generation + extra[0],
            op__Mult=10 + generation + extra[1],
            name__u=12,
        )

    start_groups = [
        [features_of(generation, index) for index in range(per_generation)]
        for generation in range(8)
    ]
    end_groups = [
        [features_of(generation, index) for index in range(per_generation)]
        for generation in range(32, 40)
    ]
    pool = [
        L.PoolProgram(
            generation=generation,
            program_sha256=f"{generation:03d}{index:03d}",
            # score rises with generation; within a generation it is the index, which the
            # feature perturbation above does not follow.
            final_score=float(generation) + 0.001 * index,
            features=features_of(generation, index),
        )
        for generation in range(8, 32)
        for index in range(per_generation)
    ]
    return (
        [L.pooled(group) for group in start_groups],
        [L.pooled(group) for group in end_groups],
        pool,
        21,
    )


def test_n3_removes_a_temporal_confound_that_n2_falls_for() -> None:
    """Nearness in time is not alignment with score, and only N3 can tell them apart."""

    start_groups, end_groups, pool, size = _temporal_confound_pool()
    drift, scale = L.drift_numerator(L.pooled(start_groups), L.pooled(end_groups))
    elite_items, foil_items = L.poles_of(pool, size)
    axis = L.axis_numerator(
        L.pooled(item.features for item in elite_items),
        L.pooled(item.features for item in foil_items),
    )
    statistic = L.projection_from_parts(drift, scale, axis)
    flat = L._seal(
        "n2",
        "",
        True,
        statistic,
        L._axis_draws(drift, scale, L._random_axes(pool, size, draws=499, seed=3)),
        ALPHA,
    )
    stratified = L._seal(
        "n3",
        "",
        True,
        statistic,
        L._axis_draws(
            drift,
            scale,
            L._stratified_axes(pool, elite_items, foil_items, draws=499, seed=4),
        ),
        ALPHA,
    )
    assert flat.fires, "the confound has to be present for the gate to mean anything"
    assert not stratified.fires, (
        "N3 randomises the score labels inside each generation, so a pool whose only "
        f"signal is the clock must not clear it; p = {stratified.p_value}"
    )


def test_the_stratified_draw_holds_the_generation_profile_exactly_fixed() -> None:
    """N3's whole claim, checked in integers rather than argued.

    Give every program in a generation the same features.  Then *which* programs a
    within-generation permutation picks cannot matter, and every stratified axis must equal
    the observed axis exactly -- which is only true if the draw really does take the same
    number from each generation that the observed poles took.  The unstratified draw is free
    to take from anywhere, and on the same pool it does not reproduce the axis.
    """

    per_generation = 4
    pool = [
        L.PoolProgram(
            generation=generation,
            program_sha256=f"{generation:03d}{index:03d}",
            final_score=float(generation) + 0.001 * index,
            features=vec(op__Add=30 - generation, op__Mult=5 + generation, name__u=9),
        )
        for generation in range(20)
        for index in range(per_generation)
    ]
    elite_items, foil_items = L.poles_of(pool, 14)
    observed = L.axis_numerator(
        L.pooled(item.features for item in elite_items),
        L.pooled(item.features for item in foil_items),
    )
    stratified = L._stratified_axes(pool, elite_items, foil_items, draws=40, seed=11)
    assert all(list(axis) == list(observed) for axis in stratified)
    flat = L._random_axes(pool, 14, draws=40, seed=11)
    assert any(list(axis) != list(observed) for axis in flat)


def test_the_stratified_draw_refuses_a_pool_that_cannot_supply_the_profile() -> None:
    pool = [
        L.PoolProgram(
            generation=index // 2,
            program_sha256=f"{index:03d}",
            final_score=float(index),
            features=vec(op__Add=1 + index, op__Mult=2),
        )
        for index in range(8)
    ]
    elite_items, foil_items = L.poles_of(pool, 3)
    truncated = [item for item in pool if item.generation != elite_items[0].generation]
    with pytest.raises(L.LearningError):
        L._stratified_axes(truncated, elite_items, foil_items, draws=2, seed=1)


def test_p_value_is_exactly_hits_plus_one_over_draws_plus_one() -> None:
    drawn = [Fraction(index, 100) for index in range(-40, 60)]
    result = L._seal("x", "", True, Fraction(1, 4), drawn, ALPHA)
    assert result.at_least_as_extreme == sum(1 for value in drawn if value >= Fraction(1, 4))
    assert result.p_value == Fraction(result.at_least_as_extreme + 1, len(drawn) + 1)
    assert isinstance(result.p_value, Fraction)


def test_a_negative_statistic_never_fires_however_small_the_p_value() -> None:
    drawn = [Fraction(-index, 10) for index in range(1, 100)]
    result = L._seal("x", "", True, Fraction(-1, 1000), drawn, ALPHA)
    assert result.p_value <= ALPHA
    assert not result.fires


def test_a_null_is_refused_rather_than_faked_on_degenerate_input() -> None:
    groups = [vec(op__Add=1), vec(op__Mult=1)]
    with pytest.raises(L.LearningError):
        L.generation_permutation_draws([], groups, [1] * WIDTH, draws=5, seed=1)
    with pytest.raises(L.LearningError):
        L._axis_draws([1] * WIDTH, 1, [[0] * WIDTH])


# ---------------------------------------------------------------------------
# Power: a negative from a dead instrument is not a negative
# ---------------------------------------------------------------------------


def test_planting_is_deterministic_and_replaces_the_leading_slots() -> None:
    groups = [[vec(op__Add=1), vec(op__Add=1)], [vec(op__Add=1)]]
    donor = vec(op__Mult=10)
    assert L.plant(groups, [donor], 0) == L.pooled([vec(op__Add=1)] * 3)
    once = L.plant(groups, [donor], 1)
    assert once == L.pooled([donor, vec(op__Add=1), donor])
    assert L.plant(groups, [donor], 5) == L.pooled([donor] * 3)
    assert L.plant(groups, [donor], 1) == once
    with pytest.raises(L.LearningError):
        L.plant(groups, [], 1)
    with pytest.raises(L.LearningError):
        L.plant(groups, [donor], -1)


def test_the_instrument_fires_when_learning_is_true_by_construction(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    """The gate that makes a negative publishable.

    The top rung of the ladder is a campaign whose early window *is* the low-scoring pole and
    whose late window *is* the high-scoring pole.  If both admissible nulls fail to fire
    there, this module cannot detect learning at all and its verdict means nothing.
    """

    for arm in arms:
        ladder = arm.power.ladder
        assert ladder[0][0] == 0
        assert ladder[0][1] == arm.statistic, "rung zero must be the observed campaign"
        assert arm.power.floor is not None, (
            f"{arm.arm}: no planted level fired, so the instrument is dead and its "
            "negative verdict is uninformative"
        )
        if arm.power.floor == 0:
            assert arm.fires_under_every_admissible_null
        assert ladder[-1][4] is True
        assert arm.power.floor <= arm.power.programs_per_generation


def test_campaign_poles_have_exchangeable_labels_within_each_generation() -> None:
    """N3 must have a reachable label permutation everywhere the observed axis is built."""

    pool: list[L.PoolProgram] = []
    for generation in range(4):
        for position in range(5):
            pool.append(
                L.PoolProgram(
                    generation=generation,
                    program_sha256=f"{generation}-{position}",
                    final_score=float(5 - position + generation),
                    features=vec(op__Add=position + 1, op__Mult=6 - position),
                )
            )
    elite, foil = L.paired_poles_of(pool, 6)
    elite_profile = {generation: 0 for generation in range(4)}
    foil_profile = {generation: 0 for generation in range(4)}
    for item in elite:
        elite_profile[item.generation] += 1
    for item in foil:
        foil_profile[item.generation] += 1
    assert elite_profile == foil_profile
    assert all(high.final_score >= low.final_score for high, low in zip(elite, foil, strict=True))


def test_the_detection_floor_is_the_first_firing_rung(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    for arm in arms:
        firing = [planted for planted, _, _, _, fires in arm.power.ladder if fires]
        assert arm.power.floor == (firing[0] if firing else None)
        for planted, _, p_two, p_three, fires in arm.power.ladder:
            assert fires == (p_two <= ALPHA and p_three <= ALPHA)
            assert planted <= arm.power.programs_per_generation


# ---------------------------------------------------------------------------
# The ablation changes exactly one thing
# ---------------------------------------------------------------------------


def test_the_ablation_switch_left_on_reproduces_the_live_loop_bit_for_bit(
    tmp_path: Path,
) -> None:
    """``selection_pressure=True`` must be the code that was already there."""

    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=12, islands=3, proposals_per_call=3, seed=5)

    def run(**kwargs: object) -> dict:
        governor = fl.SpendGovernor(tmp_path / f"l{len(kwargs)}.json", 500, 5000, 1)
        proposer = fl.MockMutationProposer(config.seed, problem.mutation_bank)
        return fl.run_problem(problem, config, proposer, governor, **kwargs)  # type: ignore[arg-type]

    assert json.dumps(run(), sort_keys=True) == json.dumps(
        run(selection_pressure=True), sort_keys=True
    )


def test_the_ablation_actually_changes_the_run(tmp_path: Path) -> None:
    """A switch that changed nothing would make the control vacuously pass."""

    problem = fl.declared_problems()["blinded_response_law"]
    config = fl.LoopConfig(generations=24, islands=3, proposals_per_call=4, seed=5)

    def run(pressure: bool) -> dict:
        governor = fl.SpendGovernor(tmp_path / f"l{pressure}.json", 500, 5000, 1)
        proposer = fl.MockMutationProposer(config.seed, problem.mutation_bank)
        return fl.run_problem(problem, config, proposer, governor, selection_pressure=pressure)

    live = run(True)
    ablated = run(False)
    assert live["population_history"] != ablated["population_history"]
    assert float(live["headline"]["best_by_final_score"]["final_score"]) > float(
        ablated["headline"]["best_by_final_score"]["final_score"]
    )


def test_the_blind_sampler_never_reads_a_score() -> None:
    """Uniform means uniform: a score-ordered island must not bias what comes out."""

    class Probe:
        def __init__(self, digest: str, score: float) -> None:
            self.program_sha256 = digest
            self.final_score = score

    island = [Probe(f"{index:02d}", float(index)) for index in range(8)]
    seen: dict[str, int] = {}
    for seed in range(4000):
        for item in fl._sample_examples_blind(island, 3, random.Random(seed)):
            seen[item.program_sha256] = seen.get(item.program_sha256, 0) + 1
    assert len(seen) == 8
    counts = sorted(seen.values())
    # 12000 draws over 8 members: uniform gives 1500 each.  A score-weighted sampler at the
    # loop's temperature would put essentially everything on the top member.
    assert counts[0] > 1200 and counts[-1] < 1800


def test_the_measurement_does_not_touch_the_declared_spend_ledger(tmp_path: Path) -> None:
    """The mutator is free.  A measurement that charged the LLM ledger would be a lie."""

    real = Path("runs/math/funsearch/spend-ledger.json")
    before = fl.read_ledger(real) if real.is_file() else None
    scratch = tmp_path / "scratch.json"
    config = L.CampaignConfig(
        problem_id="blinded_response_law",
        generations=30,
        islands=3,
        proposals_per_call=4,
        window=6,
        pole_size=6,
        seed=5,
        draws=99,
        calibration_trials=8,
        calibration_draws=49,
    )
    L.measure_arm(config, selection_pressure=True, ledger_path=scratch)
    assert fl.read_ledger(scratch) == 0, "the measurement must charge nothing"
    if before is not None:
        assert fl.read_ledger(real) == before, "the declared ledger moved"
        assert before == 600, "the declared funsearch ledger is pinned at 600 hundredths"


# ---------------------------------------------------------------------------
# The measurement on a real campaign
# ---------------------------------------------------------------------------


def test_the_two_windows_are_disjoint_and_the_poles_come_from_neither(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    for arm in arms:
        assert not set(arm.start.generations) & set(arm.end.generations)
        assert len(arm.start.generations) == FAST.window
        assert len(arm.end.generations) == FAST.window
        assert arm.pool_size >= 2 * FAST.pole_size
        assert arm.pole_size == FAST.pole_size


def test_the_poles_are_separated_by_score(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    for arm in arms:
        assert arm.elite_mean_final_score > arm.foil_mean_final_score
        assert arm.pole_separation > 0


def test_every_arm_carries_the_three_declared_nulls_on_one_statistic(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    for arm in arms:
        assert [null.null_id for null in arm.nulls] == [
            "n1_generation_permutation",
            "n2_pole_randomisation",
            "n3_stratified_pole_randomisation",
        ]
        assert [null.admissible for null in arm.nulls] == [False, True, True]
        assert len({null.statistic for null in arm.nulls}) == 1


def test_the_ablation_shows_no_shift_under_either_admissible_null(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    """The headline control.  If the ablation fires, the whole instrument is void."""

    _, ablated = arms
    assert not ablated.fires_under_any_admissible_null, (
        "selection_pressure=False severs every path from final_score to the proposer; a "
        "statistic that still fires there is reading drift"
    )


def test_the_measured_calibration_reproduces_the_finding(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    """On the campaign's own data, N1's false-positive rate must exceed the other two."""

    for arm in arms:
        rates = {item.null_id: item.rate for item in arm.calibration}
        assert set(rates) == {
            "n1_generation_permutation",
            "n2_pole_randomisation",
            "n3_stratified_pole_randomisation",
        }
        assert rates["n1_generation_permutation"] > Fraction(1, 20)
        assert rates["n1_generation_permutation"] > rates["n2_pole_randomisation"]
        assert rates["n1_generation_permutation"] > rates["n3_stratified_pole_randomisation"]
        for key in ("n2_pole_randomisation", "n3_stratified_pole_randomisation"):
            assert rates[key] <= Fraction(1, 4), f"{arm.arm}/{key} is {rates[key]}"


def test_every_number_on_the_certificate_path_is_a_fraction(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    for arm in arms:
        assert isinstance(arm.statistic, Fraction)
        assert isinstance(arm.pole_separation, Fraction)
        for null in arm.nulls:
            assert isinstance(null.statistic, Fraction)
            assert isinstance(null.p_value, Fraction)
            assert isinstance(null.alpha, Fraction)
            assert isinstance(null.at_least_as_extreme, int)
        for item in arm.calibration:
            assert isinstance(item.rate, Fraction)
        for _, statistic, p_two, p_three, _ in arm.power.ladder:
            assert isinstance(statistic, Fraction)
            assert isinstance(p_two, Fraction)
            assert isinstance(p_three, Fraction)


def test_the_measurement_is_reproducible(tmp_path: Path) -> None:
    config = L.CampaignConfig(
        problem_id="blinded_sequence_rule",
        generations=30,
        islands=3,
        proposals_per_call=4,
        window=6,
        pole_size=6,
        seed=11,
        draws=99,
        calibration_trials=6,
        calibration_draws=49,
    )
    first = L.measure_arm(config, selection_pressure=True, ledger_path=tmp_path / "a.json")
    second = L.measure_arm(config, selection_pressure=True, ledger_path=tmp_path / "b.json")
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_a_campaign_too_short_to_measure_is_refused_not_guessed(tmp_path: Path) -> None:
    config = L.CampaignConfig(generations=6, window=6, pole_size=3, draws=9)
    with pytest.raises(L.LearningError):
        L.measure_arm(config, selection_pressure=True, ledger_path=tmp_path / "l.json")


def test_the_serialised_window_distribution_is_a_rational_histogram(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    for arm in arms:
        for window in (arm.start.to_dict(), arm.end.to_dict()):
            tokens = window["feature_tokens"]
            total = 0
            for symbol, text in window["distribution"].items():
                assert symbol in L._INDEX
                numerator, _, denominator = text.partition("/")
                assert int(denominator) == tokens
                total += int(numerator)
            assert total == tokens


# ---------------------------------------------------------------------------
# The report and its seal
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(arms: tuple[L.ArmMeasurement, L.ArmMeasurement]) -> dict:
    live, ablated = arms
    return L.seal_report(FAST, live, ablated, {FAST.seed: (live, ablated)})


def test_the_report_seals_validates_and_reaches_a_verdict(report: dict) -> None:
    L.validate_report(report)
    assert report["verdict"] in {
        "learned",
        "no_measurable_shift",
        "uninformative_null",
        "statistic_is_measuring_drift",
    }
    assert report["schema_version"] == L.RECEIPT_SCHEMA
    assert report["supersedes"]["receipt"] == "runs/math/funsearch/learning-v1.json"


def test_the_report_refuses_a_broken_seal(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    tampered["verdict"] = "learned"
    with pytest.raises(L.LearningError):
        L.validate_report(tampered)


def test_a_forged_p_value_is_caught_even_after_resealing(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    tampered["arms"][0]["nulls"][1]["p_value"] = "1/1000"
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    from sigma_theory_compiler.sigma_core import canonical_sha256

    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(L.LearningError):
        L.validate_report(tampered)


def test_a_verdict_resting_on_the_inadmissible_null_alone_is_refused(report: dict) -> None:
    """The whole point.  N1 firing may never buy a positive verdict."""

    tampered = json.loads(json.dumps(report))
    for arm in tampered["arms"]:
        for null in arm["nulls"]:
            null["fires"] = null["null_id"] == "n1_generation_permutation"
        arm["fires_under_every_admissible_null"] = False
    tampered["verdict"] = "learned"
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    from sigma_theory_compiler.sigma_core import canonical_sha256

    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(L.LearningError):
        L.validate_report(tampered)


def test_promoting_the_inadmissible_null_is_refused(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    tampered["arms"][0]["nulls"][0]["admissible"] = True
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    from sigma_theory_compiler.sigma_core import canonical_sha256

    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(L.LearningError):
        L.validate_report(tampered)


def test_a_forged_power_ladder_is_caught(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    ladder = tampered["arms"][0]["power"]["ladder"]
    ladder[0]["both_admissible_nulls_fire"] = True
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    from sigma_theory_compiler.sigma_core import canonical_sha256

    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(L.LearningError):
        L.validate_report(tampered)


def test_the_verdict_follows_the_declared_rule() -> None:
    assert L.VERDICT_RULE.startswith("learned requires")
    assert "inadmissible" in L.VERDICT_RULE


# ---------------------------------------------------------------------------
# The sealed receipt on disk
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RECEIPT.is_file(), reason="receipt not present in this checkout")
def test_the_sealed_receipt_validates_and_says_what_the_module_says() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    L.validate_report(value)
    assert value["finding"] == L.FINDING
    assert value["claims"] == L.CLAIMS
    assert value["verdict_rule"] == L.VERDICT_RULE
    for arm in value["arms"]:
        assert arm["power"]["detection_floor_found"], (
            "a receipt reporting no shift is only a result if the instrument that failed to "
            "see one is shown able to see a planted one"
        )
        rates = {item["null_id"]: item for item in arm["null_calibration"]}
        n1 = Fraction(
            rates["n1_generation_permutation"]["fired"],
            rates["n1_generation_permutation"]["trials"],
        )
        assert n1 > Fraction(1, 20), "the sealed receipt must carry the v1 finding"


@pytest.mark.skipif(not RECEIPT.is_file(), reason="receipt not present in this checkout")
def test_the_sealed_receipt_shows_the_ablation_is_not_firing_systematically() -> None:
    """The ablation is allowed to fire at alpha.  It is not allowed to fire as a rule."""

    totals = json.loads(RECEIPT.read_text(encoding="utf-8"))["seed_sweep"]["totals"]
    ablated = totals["selection_off_ablation"]
    for null_id in ("n2", "n3"):
        assert Fraction(ablated[null_id], ablated["campaigns"]) <= Fraction(1, 4), (
            f"the ablation cleared {null_id} on {ablated[null_id]} of "
            f"{ablated['campaigns']} campaigns; the statistic is reading drift"
        )


@pytest.mark.skipif(not RECEIPT.is_file(), reason="receipt not present in this checkout")
def test_the_sealed_receipt_carries_the_gap_between_the_old_null_and_the_new_ones() -> None:
    """Over the whole sweep, N1 must fire more often than the admissible nulls do.

    This is the finding stated as a rate rather than a single campaign: the same statistic,
    the same data, and only the null changed.
    """

    totals = json.loads(RECEIPT.read_text(encoding="utf-8"))["seed_sweep"]["totals"]
    live = totals["selection_on"]
    assert live["n1"] > live["n2"]
    assert live["n1"] > live["n3"]


@pytest.mark.skipif(not RECEIPT.is_file(), reason="receipt not present in this checkout")
def test_the_sealed_receipt_carries_a_seed_sweep_that_is_not_one_lucky_run() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    sweep = value["seed_sweep"]
    assert sweep["campaigns"] >= 12, "one campaign is not evidence in either direction"
    for row in sweep["rows"]:
        assert set(row) >= {"problem_id", "seed", "selection_pressure", "fires"}
    live_fires = sum(1 for row in sweep["rows"] if row["selection_pressure"] and row["fires"]["n3"])
    ablated_fires = sum(
        1 for row in sweep["rows"] if not row["selection_pressure"] and row["fires"]["n3"]
    )
    assert sweep["totals"]["selection_on"]["n3"] == live_fires
    assert sweep["totals"]["selection_off_ablation"]["n3"] == ablated_fires


def test_a_sweep_whose_headline_seed_is_not_a_row_is_refused() -> None:
    """The headline has to be one of the campaigns the sweep reports, or it is cherry-picked."""

    with pytest.raises(L.LearningError):
        L.sweep(L.CampaignConfig(seed=7, sweep_seeds=(1, 2)))


def test_a_forged_sweep_total_is_caught(report: dict) -> None:
    tampered = json.loads(json.dumps(report))
    tampered["seed_sweep"]["totals"]["selection_on"]["n3"] += 1
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    from sigma_theory_compiler.sigma_core import canonical_sha256

    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(L.LearningError):
        L.validate_report(tampered)


def test_a_negative_from_an_instrument_with_no_floor_is_not_called_a_negative(
    arms: tuple[L.ArmMeasurement, L.ArmMeasurement],
) -> None:
    """The verdict that C1's logic requires, exercised on both branches.

    Two campaigns can produce the identical p-values and mean opposite things: one where a
    planted score-aligned drift would have been seen, and one where it would not.  Only the
    first is a measurement, and the verdict has to separate them.
    """

    live, ablated = arms
    assert not ablated.fires_under_any_admissible_null
    assert not live.fires_under_every_admissible_null
    with_floor = dataclasses.replace(live, power=dataclasses.replace(live.power, floor=3))
    without_floor = dataclasses.replace(live, power=dataclasses.replace(live.power, floor=None))
    assert L.verdict_of(with_floor, ablated) == "no_measurable_shift"
    assert L.verdict_of(without_floor, ablated) == "uninformative_null"
    # and a firing ablation still outranks both
    fired = dataclasses.replace(
        ablated,
        nulls=(
            ablated.nulls[0],
            dataclasses.replace(ablated.nulls[1], fires=True),
            ablated.nulls[2],
        ),
    )
    assert L.verdict_of(without_floor, fired) == "statistic_is_measuring_drift"


@pytest.mark.skipif(not RECEIPT.is_file(), reason="receipt not present in this checkout")
def test_the_sealed_receipt_separates_real_negatives_from_uninformative_ones() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    sweep = value["seed_sweep"]
    counts = sweep["verdict_counts"]
    assert sum(counts.values()) == len(sweep["verdicts"])
    assert counts["statistic_is_measuring_drift"] == 0, (
        "an ablated arm clearing an admissible null voids the instrument"
    )
    qualified = sweep["totals"]["selection_on"]["power_qualified"]
    assert counts["no_measurable_shift"] + counts["learned"] == qualified, (
        "every campaign with a detection floor must reach a substantive verdict, and every "
        "campaign without one must not"
    )
