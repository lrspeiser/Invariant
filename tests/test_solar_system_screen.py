"""Gates for the solar-system screen.

The screen's whole value is that it is one-sided, so the tests are organised around the ways
one-sidedness can be lost: a bracket that does not contain the truth, a bound that decides a
cell it has not proved, a fractional channel that mistakes an unobservable ``GM`` rescaling for
a deviation, a receipt whose framing has been quietly upgraded from screen to confirmation.

Every positive control is paired with a negative that must fail, and the sharpest pair differs
by a single coefficient.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

from sigma_theory_compiler import solar_system_screen as s
from sigma_theory_compiler.sigma_core import canonical_sha256


def _law(name, numerator, denominator, beta):
    return s.ResponseLaw(
        name=name,
        numerator=tuple(Fraction(value) for value in numerator),
        denominator=tuple(Fraction(value) for value in denominator),
        beta=Fraction(beta),
    )


def _mpf(value: Fraction) -> mp.mpf:
    """An mpmath value from an exact rational, without ever going through a float."""

    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _cells(result, channel):
    return {cell["anchor"]: cell for cell in result["cells"] if cell["channel"] == channel}


# ---------------------------------------------------------------------------
# The regime, before any law is screened
# ---------------------------------------------------------------------------


def test_solar_accelerations_land_where_the_screen_claims_they_do() -> None:
    """Roughly 5.9e-3 at Earth down to 6.5e-6 at Neptune, all far above a0."""

    anchors = {anchor.name: anchor for anchor in s.build_anchors()}
    earth = float(anchors["earth"].newtonian_acceleration)
    neptune = float(anchors["neptune"].newtonian_acceleration)
    assert earth == pytest.approx(5.93e-3, rel=1e-2)
    assert neptune == pytest.approx(6.56e-6, rel=1e-2)
    assert float(anchors["mercury"].newtonian_acceleration) == pytest.approx(3.96e-2, rel=1e-2)


def test_every_anchor_sits_at_least_four_decades_above_a0() -> None:
    """The premise of the whole module: no anchor is anywhere near the MOND scale."""

    for anchor in s.build_anchors():
        assert anchor.y > 10**4
    weakest = min(anchor.y for anchor in s.build_anchors())
    strongest = max(anchor.y for anchor in s.build_anchors())
    assert weakest > 5 * 10**4
    assert strongest < 10**9


def test_the_declared_anchor_table_is_internally_consistent() -> None:
    """Kepler's third law checks the axis column against the period column."""

    rows = s.kepler_consistency(s.build_anchors())
    assert len(rows) == len(s.ANCHORS)
    assert all(row["within_tolerance"] for row in rows)


def test_a_mistyped_anchor_period_breaks_the_kepler_check() -> None:
    """The self-check must actually be able to fail, or it is decoration."""

    good = s.build_anchors()[0]
    broken = s.Anchor(
        name=good.name,
        semi_major_axis_au=good.semi_major_axis_au,
        period_years=good.period_years * 10,
        newtonian_acceleration=good.newtonian_acceleration,
        y=good.y,
    )
    assert not s.kepler_consistency([broken])[0]["within_tolerance"]


# ---------------------------------------------------------------------------
# Exact arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3, 5, 7])
def test_integer_nth_root_is_exact(degree: int) -> None:
    for value in [0, 1, 2, 3, 10, 12345, 2**61 - 1, 3**40, 10**30 + 7]:
        root = s.integer_nth_root(value, degree)
        assert root**degree <= value
        assert (root + 1) ** degree > value


def test_a_bracket_encloses_the_value_an_independent_library_computes() -> None:
    """The exact interval must contain what 50-digit mpmath says, at every anchor.

    This is the control on the arithmetic itself: the screen's endpoints are produced by
    integer operations and rounded outward, and mpmath knows nothing about any of that.
    """

    mp.mp.dps = 50
    law = _law("probe", (1, 0, 1), (1, 3), Fraction(1, 3))
    for anchor in s.build_anchors():
        evaluation = s.evaluate_at_anchor(law, anchor, 128)
        u = mp.sqrt(1 / _mpf(anchor.y))
        top = 1 + u**2
        bottom = 1 + 3 * u
        nu = (top / bottom) ** (mp.mpf(1) / 3)
        assert _mpf(evaluation.nu.lo) <= nu <= _mpf(evaluation.nu.hi)

        slope = -(mp.mpf(1) / 3) / 2 * (u * (2 * u) / top - u * 3 / bottom)
        assert _mpf(evaluation.log_slope.lo) <= slope <= _mpf(evaluation.log_slope.hi)


def test_the_precession_channel_agrees_with_an_independent_evaluation() -> None:
    """The published formula, re-evaluated in mpmath, must land inside the bracket."""

    mp.mp.dps = 50
    law = _law("simple_like", (1, 0, 1), (1,), 1)
    for anchor in s.build_anchors():
        evaluation = s.evaluate_at_anchor(law, anchor, 128)
        y = _mpf(anchor.y)
        nu = 1 + 1 / y
        slope = (-1 / y) / nu
        advance = (1 / mp.sqrt(1 - 2 * slope) - 1) * 1296000000
        expected = advance * 100 / _mpf(anchor.period_years)
        bracket = evaluation.precession_mas_per_century
        assert _mpf(bracket.lo) <= expected <= _mpf(bracket.hi)


def test_no_float_reaches_the_certificate() -> None:
    """`canonical_sha256` rejects floats outright, so sealing at all is the proof."""

    receipt = s.build_receipt()
    assert canonical_sha256(receipt)

    def walk(node) -> None:
        assert not isinstance(node, float), "a float reached the certificate"
        if isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(receipt)


# ---------------------------------------------------------------------------
# The required controls: Newton passes, a non-reducing law fails
# ---------------------------------------------------------------------------


def test_newton_passes() -> None:
    result = s.screen_law(_law("newton", (1,), (1,), 1))
    assert result["verdict"] == s.PASS
    assert not result["decisive_cells"]


def test_a_deliberately_non_reducing_law_fails() -> None:
    """nu = y^-1/2 everywhere: deep MOND at Mercury, which the solar system rules out."""

    result = s.screen_law(_law("deep_mond_everywhere", (0, 1), (1,), 1))
    assert result["verdict"] == s.FAIL
    failing = {(cell["anchor"], cell["channel"]) for cell in result["decisive_cells"]}
    assert ("mercury", s.CHANNEL_PRECESSION) in failing
    assert Fraction(result["worst_margin_ratio"]) > 10**6


def test_the_whole_declared_battery_lands_where_it_must() -> None:
    controls = s.run_controls()
    assert controls["all_as_expected"], [row for row in controls["rows"] if not row["as_expected"]]
    assert controls["positives"] >= 4
    assert controls["negatives"] >= 4


def test_one_coefficient_separates_a_pass_from_a_fail() -> None:
    """The threshold is where the screen says it is, not somewhere vaguely near it."""

    inside = s.screen_law(_law("inside", (1, 0, Fraction(1, 400)), (1,), 1))
    outside = s.screen_law(_law("outside", (1, 0, Fraction(1, 300)), (1,), 1))
    assert inside["verdict"] == s.PASS
    assert outside["verdict"] == s.FAIL
    assert Fraction(inside["worst_margin_ratio"]) < 1
    assert Fraction(outside["worst_margin_ratio"]) > 1


# ---------------------------------------------------------------------------
# The measured result: the repository's own surviving local factor
# ---------------------------------------------------------------------------


def test_the_surviving_family_local_factor_fails_the_screen() -> None:
    """sqrt(1 + a0/g_N) is the local factor of all twelve nonlocal-localization survivors."""

    result = s.screen_law(_law("local_factor", (1, 0, 1), (1,), Fraction(1, 2)))
    assert result["verdict"] == s.FAIL
    precession = _cells(result, s.CHANNEL_PRECESSION)
    assert precession["mercury"]["status"] == s.EXCEEDS
    assert Fraction(precession["mercury"]["magnitude"]["lo"]) > 100


def test_the_receipt_states_the_scope_restriction_on_that_refutation() -> None:
    """The families carry a screening factor this module does not evaluate; say so."""

    receipt = s.build_receipt()
    scope = receipt["scope_restriction"]
    assert "curvature-screening factor" in scope
    assert "NOT evaluated here" in scope


# ---------------------------------------------------------------------------
# One-sidedness
# ---------------------------------------------------------------------------


def _scaled_bounds(factor: Fraction) -> dict[str, dict[str, str]]:
    return {
        anchor: {
            channel: str(s.exact_rational(value) * factor) for channel, value in channels.items()
        }
        for anchor, channels in s.BOUND_SCHEDULE.items()
    }


def test_tightening_the_bounds_can_never_reverse_a_fail() -> None:
    tighter = _scaled_bounds(Fraction(1, 100))
    for law in s.control_laws()["must_fail"]:
        assert s.screen_law(law, bounds=tighter)["verdict"] == s.FAIL


def test_loosening_the_bounds_can_never_reverse_a_pass() -> None:
    looser = _scaled_bounds(Fraction(100))
    for law in s.control_laws()["must_pass"]:
        assert s.screen_law(law, bounds=looser)["verdict"] == s.PASS


def test_a_straddling_bracket_is_unresolved_and_is_never_rounded_into_a_pass() -> None:
    """A zero bound cannot be met by a bracket that merely contains zero."""

    zero = {anchor: {channel: "0" for channel in s.CHANNELS} for anchor in s.BOUND_SCHEDULE}
    result = s.screen_law(_law("newton", (1,), (1,), 1), bounds=zero)
    assert result["verdict"] == s.UNRESOLVED
    assert any(cell["status"] == s.UNRESOLVED for cell in result["cells"])
    assert not any(cell["status"] == s.EXCEEDS for cell in result["cells"])


def test_decide_cell_only_ever_decides_what_it_has_proved() -> None:
    bound = Fraction(1)
    assert s.decide_cell(s.Bracket(Fraction(2), Fraction(3)), bound) == s.EXCEEDS
    assert s.decide_cell(s.Bracket(Fraction(0), Fraction(1)), bound) == s.WITHIN
    assert s.decide_cell(s.Bracket(Fraction(1, 2), Fraction(3, 2)), bound) == s.UNRESOLVED


def test_an_empty_precision_ladder_is_refused_rather_than_defaulting_to_pass() -> None:
    with pytest.raises(s.SolarScreenError):
        s.screen_law(_law("newton", (1,), (1,), 1), ladder=())


# ---------------------------------------------------------------------------
# The GM degeneracy, which the fractional channel exists to respect
# ---------------------------------------------------------------------------


def test_a_constant_response_is_recognised_as_a_gm_rescaling() -> None:
    assert s.is_gm_degenerate(_law("offset", (1001,), (1000,), 1))
    assert s.is_gm_degenerate(_law("proportional", (2, 4, 6), (1, 2, 3), 1))
    assert not s.is_gm_degenerate(_law("varying", (1, 0, 1), (1,), 1))


def test_a_constant_response_passes_and_the_receipt_calls_that_pass_worthless() -> None:
    result = s.screen_law(_law("offset", (1001,), (1000,), 1))
    assert result["verdict"] == s.PASS
    assert result["gm_degenerate"]
    assert "no information whatsoever" in result["gm_degeneracy_note"]


def test_rescaling_a_law_by_a_constant_moves_no_cell() -> None:
    """The sharp test that the channel measures the shape of nu, not its absolute level."""

    plain = s.screen_law(_law("plain", (1, 0, 1), (1,), 1))
    scaled = s.screen_law(_law("scaled", (1000, 0, 1000), (1,), 1))
    assert plain["verdict"] == scaled["verdict"]
    for left, right in zip(plain["cells"], scaled["cells"], strict=True):
        assert left["anchor"] == right["anchor"] and left["channel"] == right["channel"]
        assert left["status"] == right["status"]
        # The two enclose the same real number, so the intervals must intersect.  They need
        # not be byte-identical: outward rounding depends on the size of the intermediates.
        low = Fraction(left["magnitude"]["lo"]), Fraction(left["magnitude"]["hi"])
        high = Fraction(right["magnitude"]["lo"]), Fraction(right["magnitude"]["hi"])
        assert low[0] <= high[1] and high[0] <= low[1]


# ---------------------------------------------------------------------------
# The admissible region
# ---------------------------------------------------------------------------


def test_the_frontier_brackets_the_largest_surviving_coefficient() -> None:
    for row in s.default_frontier():
        if not row["bracketed"]:
            continue
        low = Fraction(row["largest_passing_coefficient"])
        high = Fraction(row["smallest_failing_coefficient"])
        assert low < high
        assert s.screen_law(s.recovery_law(row["exponent"], low))["verdict"] == s.PASS
        assert s.screen_law(s.recovery_law(row["exponent"], high))["verdict"] == s.FAIL


def test_the_frontier_places_the_classical_interpolating_functions_where_physics_does() -> None:
    """Linear recovery is excluded at coefficient one; quadratic recovery has room to spare."""

    frontier = {row["exponent"]: row for row in s.default_frontier()}
    linear = Fraction(frontier[1]["largest_passing_coefficient"])
    quadratic = Fraction(frontier[2]["largest_passing_coefficient"])
    assert linear < Fraction(1, 100)
    assert quadratic > 100


def test_the_frontier_family_is_monotone_in_its_coefficient() -> None:
    """Bisection is only a proof if the verdict is monotone; check it on a ladder."""

    previous = Fraction(0)
    seen_fail = False
    for step in range(14):
        coefficient = Fraction(1, 10000) * 2**step
        verdict = s.screen_law(s.recovery_law(1, coefficient))["verdict"]
        if verdict == s.FAIL:
            seen_fail = True
        else:
            assert not seen_fail, "a larger coefficient passed after a smaller one failed"
        assert coefficient > previous
        previous = coefficient
    assert seen_fail


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------


def test_the_receipt_builds_verifies_and_is_deterministic() -> None:
    first = s.build_receipt()
    second = s.build_receipt()
    assert first["certificate_sha256"] == second["certificate_sha256"]
    assert first["decision"] == "SCREEN_OPERATIONAL"
    assert s.verify_receipt(first) == {"accepted": True, "problems": []}


def test_every_forgery_is_rejected() -> None:
    probes = s.tamper_probes(s.build_receipt())
    assert probes["honest_receipt_accepted"]
    assert probes["all_probes_rejected"], [row for row in probes["probes"] if not row["rejected"]]
    assert len(probes["probes"]) >= 8


def test_the_framing_cannot_be_upgraded_into_a_confirmation() -> None:
    """The honesty of the receipt is enforced mechanically, not by convention."""

    receipt = s.build_receipt()
    framing = receipt["evidential_framing"]
    assert framing["artifact_kind"] == "screen"
    assert framing["is_a_discovery"] is False
    assert framing["pass_is_necessary"] is True
    assert framing["pass_is_sufficient"] is False
    assert framing["may_be_cited_as_confirmation"] is False
    assert framing["pass_evidential_weight"] == "almost none"

    forged = dict(receipt)
    forged["evidential_framing"] = dict(framing) | {"pass_is_sufficient": True}
    forged.pop("certificate_sha256")
    forged["certificate_sha256"] = canonical_sha256(forged)
    assert not s.verify_receipt(forged)["accepted"]


def test_the_bounds_declare_that_they_are_deliberately_loose() -> None:
    provenance = s.build_receipt()["declarations"]["bounds"]["provenance"]
    assert provenance["provenance_class"] == "conservative_round_number"
    assert provenance["transcribed_from_a_published_table"] is False
    assert provenance["declared_looser_than_published_constraints"] is True
    assert provenance["tightening_cannot_reverse_a_fail"] is True


def test_a_law_declaration_round_trips_through_the_receipt() -> None:
    law = _law("round_trip", (1, 0, Fraction(3, 7)), (1, 2), Fraction(-2, 3))
    assert s.law_from_declaration(law.declaration()) == law


# ---------------------------------------------------------------------------
# The grammar refuses what it cannot screen
# ---------------------------------------------------------------------------


def test_malformed_laws_are_refused() -> None:
    with pytest.raises(s.SolarScreenError):
        s.ResponseLaw(name="", numerator=(Fraction(1),), denominator=(Fraction(1),), beta=Fraction(1))
    with pytest.raises(s.SolarScreenError):
        s.ResponseLaw(name="n", numerator=(), denominator=(Fraction(1),), beta=Fraction(1))
    with pytest.raises(s.SolarScreenError):
        s.ResponseLaw(name="n", numerator=(1,), denominator=(Fraction(1),), beta=Fraction(1))
    with pytest.raises(s.SolarScreenError):
        s.ResponseLaw(
            name="n", numerator=(Fraction(1),), denominator=(Fraction(1),), beta=Fraction(0)
        )


def test_a_law_that_cannot_be_bounded_away_from_zero_is_refused_not_guessed() -> None:
    """nu must be provably positive; a response that may vanish is not screened silently."""

    anchor = s.build_anchors()[0]
    with pytest.raises(s.SolarScreenError):
        s.evaluate_at_anchor(_law("vanishing", (0,), (1,), 1), anchor, 64)


def test_an_undeclared_anchor_bound_is_refused() -> None:
    with pytest.raises(s.SolarScreenError):
        s.screen_law(_law("newton", (1,), (1,), 1), bounds={"mercury": s.BOUND_SCHEDULE["mercury"]})


def test_exact_rational_rejects_what_it_cannot_parse_exactly() -> None:
    assert s.exact_rational("1.2e-10") == Fraction(12, 10**11)
    assert s.exact_rational("1/300") == Fraction(1, 300)
    for bad in ["", "   ", "not a number", "0x10", "1/0"]:
        with pytest.raises(s.SolarScreenError):
            s.exact_rational(bad)
