"""Gates for the behavioural creativity measure.

The measure exists because source diversity is not creativity, so the controls that matter are
the ones that try to inflate it dishonestly: rewriting one function many ways, padding a
population with a repeated behaviour, and rediscovering a known answer.  Each must fail to move
the headline.
"""

from __future__ import annotations

import math

import pytest

from sigma_theory_compiler import creativity_measure as cm


def _program(source: str, outputs, novelty: float = 1.0, origin: str = "proposed"):
    return {
        "source": source,
        "origin": origin,
        "outputs": [repr(float(value)) for value in outputs],
        "novelty": {"novelty_multiplier": str(novelty)},
    }


# ---------------------------------------------------------------------------
# The measure counts behaviours, not spellings
# ---------------------------------------------------------------------------


def test_many_spellings_of_one_function_are_one_behaviour() -> None:
    """The failure this measure exists for: high source variety, zero functional variety."""

    points = [1e-12, 1e-9, 1e-6]
    identity = [_program(f"def rule(u):  # variant {i}\n    return u * 1.0", points)
                for i in range(8)]
    result = cm.measure_creativity(identity)
    assert result["population"]["distinct_sources"] == 8
    assert result["population"]["distinct_behaviours"] == 1
    assert float(result["effective_behaviours"]) == pytest.approx(1.0, abs=1e-9)
    assert float(result["wasted_variation_ratio"]) == pytest.approx(8.0)


def test_genuinely_different_behaviours_are_counted() -> None:
    points = [1e-12, 1e-9, 1e-6]
    population = [
        _program("a", points),
        _program("b", [p * 2 for p in points]),
        _program("c", [p * 10 for p in points]),
        _program("d", [math.sqrt(p) for p in points]),
    ]
    result = cm.measure_creativity(population)
    assert result["population"]["distinct_behaviours"] == 4
    assert float(result["effective_behaviours"]) == pytest.approx(4.0, abs=1e-6)
    assert float(result["wasted_variation_ratio"]) == pytest.approx(1.0)


def test_rewriting_a_program_cannot_raise_the_measure() -> None:
    """The anti-gaming control. Adding spellings of an existing behaviour must not help."""

    points = [1e-12, 1e-9, 1e-6]
    base = [_program("one", points), _program("two", [p * 3 for p in points])]
    padded = base + [
        _program(f"one rewritten {i}", points) for i in range(20)
    ]
    lean = cm.measure_creativity(base)
    fat = cm.measure_creativity(padded)
    assert fat["population"]["distinct_behaviours"] == lean["population"]["distinct_behaviours"]
    assert float(fat["effective_novel_behaviours"]) <= float(lean["effective_novel_behaviours"]) + 1e-9
    assert float(fat["wasted_variation_ratio"]) > float(lean["wasted_variation_ratio"])


def test_a_dominated_population_does_not_score_as_many_behaviours() -> None:
    """Sixteen of one behaviour plus one other is about two, not seventeen."""

    points = [1e-12, 1e-9, 1e-6]
    population = [_program(f"same {i}", points) for i in range(16)]
    population.append(_program("other", [p * 5 for p in points]))
    result = cm.measure_creativity(population)
    assert result["population"]["distinct_behaviours"] == 2
    assert float(result["effective_behaviours"]) < 2.0
    assert float(result["effective_behaviours"]) > 1.0


# ---------------------------------------------------------------------------
# Known answers are reported, never folded into the headline
# ---------------------------------------------------------------------------


def test_known_family_matches_are_excluded_from_the_headline() -> None:
    points = [1e-12, 1e-9, 1e-6]
    population = [
        _program("known", points, novelty=0.0),
        _program("also known", [p * 2 for p in points], novelty=0.0),
        _program("novel", [p * 7 for p in points], novelty=1.0),
    ]
    result = cm.measure_creativity(population)
    assert result["population"]["distinct_behaviours"] == 3
    assert float(result["effective_novel_behaviours"]) == pytest.approx(1.0, abs=1e-9)
    assert float(result["known_collapse_fraction"]) == pytest.approx(2 / 3)


def test_recovering_only_known_answers_scores_zero_novel_behaviours() -> None:
    points = [1e-12, 1e-9, 1e-6]
    population = [
        _program("k1", points, novelty=0.0),
        _program("k2", [p * 2 for p in points], novelty=0.0),
    ]
    result = cm.measure_creativity(population)
    assert float(result["effective_novel_behaviours"]) == pytest.approx(0.0)
    assert float(result["known_collapse_fraction"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Distance and hygiene
# ---------------------------------------------------------------------------


def test_distance_is_scale_free() -> None:
    """A factor of two is the same distance at 1e-12 as at 1e-6."""

    small = cm.log_relative_distance([1e-12], [2e-12])
    large = cm.log_relative_distance([1e-6], [2e-6])
    assert small == pytest.approx(large)
    assert small == pytest.approx(math.log(2.0))


def test_incomparable_vectors_do_not_merge() -> None:
    assert cm.log_relative_distance([1.0, 2.0], [1.0]) == math.inf
    assert cm.log_relative_distance([float("nan")], [1.0]) == math.inf
    assert cm.log_relative_distance([-1.0], [1.0]) == math.inf


def test_only_proposed_programs_are_measured() -> None:
    """Seeds and planted probes are not the proposer's work."""

    points = [1e-12, 1e-9, 1e-6]
    population = [
        _program("seed", points, origin="seed"),
        _program("probe", [p * 2 for p in points], origin="probe_known_canonical"),
        _program("mine", [p * 3 for p in points], origin="proposed"),
    ]
    result = cm.measure_creativity(population)
    assert result["population"]["programs"] == 1
    assert result["population"]["distinct_behaviours"] == 1


def test_an_empty_population_is_zero_not_an_error() -> None:
    result = cm.measure_creativity([])
    assert float(result["effective_novel_behaviours"]) == pytest.approx(0.0)
    assert float(result["wasted_variation_ratio"]) == pytest.approx(0.0)


def test_the_comparison_states_which_way_is_better() -> None:
    points = [1e-12, 1e-9, 1e-6]
    before = cm.measure_creativity([_program(f"s{i}", points) for i in range(6)])
    after = cm.measure_creativity(
        [_program(f"s{i}", [p * (i + 1) for p in points]) for i in range(6)]
    )
    report = cm.compare(before, after)
    assert report["verdict"] == "better"
    waste = next(r for r in report["rows"] if r["metric"] == "wasted_variation_ratio")
    assert waste["direction_that_is_better"] == "down"
    assert waste["verdict"] == "better"


def test_waste_means_converged_when_quality_is_high() -> None:
    """Forty-four spellings of the right answer is convergence, not paralysis.

    A measured live run on the sequence problem produced exactly this shape: 44 distinct
    sources, one behaviour, quality 1.0. Reading the waste ratio alone would have condemned a
    solved problem.
    """

    points = [1.0, 2.0, 3.0]
    solved = [_program(f"spelling {i}", points) for i in range(44)]
    stuck = [_program(f"spelling {i}", points) for i in range(44)]
    assert cm.measure_creativity(solved, best_quality=1.0)["regime"] == "converged"
    assert cm.measure_creativity(stuck, best_quality=0.06)["regime"] == "stuck"
    assert cm.measure_creativity(stuck)["regime"] == "unknown_no_quality_supplied"


def test_a_spread_population_is_exploring() -> None:
    points = [1.0, 2.0, 3.0]
    spread = [_program(f"s{i}", [p * (i + 1) for p in points]) for i in range(5)]
    assert cm.measure_creativity(spread, best_quality=0.4)["regime"] == "exploring"
