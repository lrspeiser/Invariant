"""The verification path is the deliverable here, so it is what the tests attack.

Every claim this module can make reduces to one sentence -- *this explicit rational
configuration has minimum triangle area exactly A, and A beats the published record* -- so
the tests check the three ways that sentence can be false: the area is computed wrongly, an
illegal configuration is accepted, or a tie is reported as a win.
"""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from sigma_theory_compiler.heilbronn_square_search import (
    PUBLISHED_RECORDS,
    RECORD_SOURCE,
    _RealRefiner,
    build_layout,
    certify,
    decimal_floor,
    enumerate_layouts,
    exact_min_double_area,
    lattice_polish,
    orbit_sizes,
    search_configuration,
    search_symmetric,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

UNIT_SQUARE_CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def _independent_min_area(points, denominator):
    """Recompute the objective by a deliberately different route: exact Fraction areas."""
    best = None
    for a, b, c in combinations(points, 3):
        pa = (Fraction(a[0], denominator), Fraction(a[1], denominator))
        pb = (Fraction(b[0], denominator), Fraction(b[1], denominator))
        pc = (Fraction(c[0], denominator), Fraction(c[1], denominator))
        area = abs(
            (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1])
        ) / 2
        best = area if best is None else min(best, area)
    return best


def test_decimal_floor_truncates_and_never_rounds_up() -> None:
    assert decimal_floor(Fraction(1, 3), 5) == "0.33333"
    assert decimal_floor(Fraction(2, 3), 5) == "0.66666"  # not .66667
    assert decimal_floor(Fraction(1, 2), 3) == "0.500"
    assert decimal_floor(Fraction(7, 341), 8) == "0.02052785"
    assert decimal_floor(Fraction(5, 1), 0) == "5"
    assert decimal_floor(Fraction(-2, 3), 3) == "-0.667"  # floors, so still a lower bound
    with pytest.raises(ValueError):
        decimal_floor(Fraction(1, 2), -1)


def test_minimum_double_area_matches_an_independent_exact_computation() -> None:
    minimum, attaining = exact_min_double_area(UNIT_SQUARE_CORNERS)
    assert (minimum, attaining) == (1, 4)

    rng = np.random.default_rng(20260820)
    denominator = 512
    for _ in range(12):
        points = [
            (int(x), int(y))
            for x, y in rng.integers(0, denominator + 1, size=(9, 2))
        ]
        if len({p for p in points}) != len(points):
            continue
        minimum, _ = exact_min_double_area(points)
        expected = _independent_min_area(points, denominator)
        assert Fraction(minimum, 2 * denominator * denominator) == expected


def test_certified_area_is_exact_and_the_payload_carries_no_floats() -> None:
    certificate = certify(UNIT_SQUARE_CORNERS, 1)
    assert certificate.area == Fraction(1, 2)
    assert certificate.area_decimal.startswith("0.5000")
    assert certificate.triples_checked == 4
    assert certificate.attaining_triples == 4
    # canonical_sha256 raises on any float anywhere in the structure.
    assert canonical_sha256(certificate.to_payload()) == certificate.digest()
    assert json.loads(json.dumps(certificate.to_payload()))["n"] == 4


def test_matching_an_exactly_known_record_is_not_beating_it() -> None:
    """n = 4's record is exactly 1/2 and the corners attain it; that is a tie, not a win."""
    certificate = certify(UNIT_SQUARE_CORNERS, 1)
    assert certificate.record_strict is True
    assert certificate.area == PUBLISHED_RECORDS[4].lower_bound_fraction
    assert certificate.beats_record is False
    assert certificate.shortfall_ratio_decimal == "1.000000000"


def test_truncated_records_demand_a_full_ulp_of_improvement() -> None:
    record = PUBLISHED_RECORDS[27]
    assert record.strict is False
    assert record.lower_bound == "0.006790"
    assert record.beat_threshold == "0.0067910"
    assert record.is_beaten_by(Fraction("0.006790")) is False
    assert record.is_beaten_by(Fraction("0.0067909")) is False
    assert record.is_beaten_by(Fraction("0.006791")) is True


def test_illegal_configurations_are_refused_rather_than_scored() -> None:
    with pytest.raises(ValueError):  # outside the square
        certify([(0, 0), (1, 0), (1, 1), (2, 1)], 1)
    with pytest.raises(ValueError):  # repeated point
        certify([(0, 0), (0, 0), (1, 1), (0, 1)], 1)
    with pytest.raises(ValueError):  # three collinear, area zero
        certify([(0, 0), (1, 1), (2, 2), (0, 2)], 2)
    with pytest.raises(TypeError):  # float coordinates cannot be exact
        certify([(0.0, 0.0), (1, 0), (1, 1), (0, 1)], 1)
    # A perfectly legal 36-point configuration (a parabola arc, so no three are collinear)
    # still cannot be scored, because no record for n = 36 has been transcribed.
    parabola = [(35 * i, i * i) for i in range(36)]
    assert exact_min_double_area(parabola)[0] > 0
    with pytest.raises(KeyError):
        certify(parabola, 1225)


def test_transcribed_record_table_is_internally_consistent() -> None:
    assert "erich-friedman.github.io" in RECORD_SOURCE
    assert set(PUBLISHED_RECORDS) == set(range(3, 36))
    previous = None
    for n in range(3, 36):
        record = PUBLISHED_RECORDS[n]
        assert record.n == n
        assert record.beat_threshold_fraction >= record.lower_bound_fraction
        # Deleting a point cannot shrink the smallest triangle, so the true optimum is
        # non-increasing in n; a transcription typo would almost certainly break this.
        if previous is not None:
            assert record.lower_bound_fraction <= previous
        previous = record.lower_bound_fraction


def test_determinant_gradients_agree_with_finite_differences() -> None:
    refiner = _RealRefiner(5)
    rng = np.random.default_rng(7)
    points = rng.random((5, 2))
    base = refiner.dets(points)
    grad = refiner.gradients(points)
    step = 1e-6
    for row, (i, j, k) in enumerate(combinations(range(5), 3)):
        slots = [(i, 0), (i, 1), (j, 0), (j, 1), (k, 0), (k, 1)]
        for slot, (point, axis) in enumerate(slots):
            moved = points.copy()
            moved[point, axis] += step
            numeric = (refiner.dets(moved)[row] - base[row]) / step
            assert numeric == pytest.approx(grad[row, slot], abs=1e-5)


def test_lattice_polish_lands_on_the_lattice_and_never_loses_ground() -> None:
    rng = np.random.default_rng(3)
    real = rng.random((6, 2))
    denominator = 1 << 16
    points, minimum = lattice_polish(real, denominator, seed=1)
    grid = np.rint(real * denominator).astype(np.int64)
    rounded = [(int(x), int(y)) for x, y in grid]
    assert minimum >= exact_min_double_area(rounded)[0]
    assert exact_min_double_area(points)[0] == minimum
    assert all(0 <= x <= denominator and 0 <= y <= denominator for x, y in points)


def _square_maps():
    return (
        lambda x, y: (x, y),
        lambda x, y: (1 - y, x),
        lambda x, y: (1 - x, 1 - y),
        lambda x, y: (y, 1 - x),
        lambda x, y: (1 - x, y),
        lambda x, y: (x, 1 - y),
        lambda x, y: (y, x),
        lambda x, y: (1 - y, 1 - x),
    )


def test_layout_points_stay_in_the_square_and_the_orbit_really_is_invariant() -> None:
    layout = build_layout("d4", ("generic",))
    assert layout.n == 8
    assert layout.dimension == 2
    rng = np.random.default_rng(5)
    for _ in range(6):
        points = layout.points(rng.random(2))
        assert points.min() >= 0.0 and points.max() <= 1.0
        original = {(round(x, 9), round(y, 9)) for x, y in points}
        for transform in _square_maps():
            image = {
                (round(a, 9), round(b, 9))
                for a, b in (transform(x, y) for x, y in points)
            }
            assert image == original


def test_enumerate_layouts_says_no_when_a_symmetry_cannot_hold() -> None:
    # 8a + 4b + (0 or 1) never equals 27 or 30, so no D4-symmetric set of that size exists.
    assert enumerate_layouts("d4", 27) == []
    assert enumerate_layouts("d4", 30) == []
    assert ("generic", "generic", "generic") in enumerate_layouts("d4", 24)
    # C2 pairs points up, so it is available for every n; odd n needs the fixed centre.
    for n in (19, 27, 30):
        combos = enumerate_layouts("c2", n)
        assert combos
        for combo in combos:
            assert sum(orbit_sizes("c2")[spec] for spec in combo) == n
        if n % 2:
            assert all("centre" in combo for combo in combos)


def test_symmetric_search_returns_a_certifiable_configuration() -> None:
    found = search_symmetric(12, "d2", seconds=4.0, seed=2)
    assert found is not None
    points, value = found
    assert points.shape == (12, 2)
    assert value > 0
    lattice, minimum = lattice_polish(points, 1 << 18, seed=4)
    assert certify(lattice, 1 << 18).min_double_area == minimum
    assert search_symmetric(27, "d4", seconds=1.0, seed=0) is None


@pytest.mark.parametrize("n", [5, 6])
def test_short_search_returns_a_configuration_that_certifies(n: int) -> None:
    denominator = 1 << 18
    points, minimum = search_configuration(
        n, denominator, seconds=2.0, seed=11 + n
    )
    certificate = certify(points, denominator)
    assert certificate.n == n
    assert certificate.min_double_area == minimum
    assert certificate.area > 0
    assert certificate.digest() == canonical_sha256(certificate.to_payload())
    # The search is a heuristic and is never allowed to assert novelty on its own.
    assert certificate.to_payload()["absence_establishes_novelty"] is False
