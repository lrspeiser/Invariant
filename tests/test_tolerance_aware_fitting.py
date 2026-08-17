"""B8 tolerance-aware fitting gates.

With tolerance every model fits, so the load-bearing tests here are the ones that prove a
verdict could have gone the other way: that a sigma the declared rule does not permit
aborts the run, that an entry which reaches every interval is still refused when it cannot
pay its parsimony budget, that an infeasibility certificate re-verifies from the receipt
alone, and that no scalar score can reach the receipt.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.basis_synthesis import synthesize_basis
from sigma_theory_compiler.tolerance_aware_fitting import (
    CLAIMS,
    FORBIDDEN_RECEIPT_KEY_TOKENS,
    PARSIMONY_RULE,
    SIGMA_RULES,
    SYSTEM_CAPS,
    LadderEntry,
    ToleranceFittingError,
    build_controls,
    build_ladder,
    certified_exponent_bracket,
    certify_power_law_outward,
    decide_power_law,
    decimal_string,
    fit_measured,
    forbidden_receipt_keys,
    invariant_interval,
    parse_rows,
    validate_controls,
    validate_result,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LINEAR = build_ladder(include_power_law=False)
POWER = build_ladder(include_linear=False)
SOURCE = "test fixture"
CITATION = "test fixture: declared instrument precision"


def _rows(function, points, sigma="0.0000001", offsets=None):
    rows = []
    for point in points:
        value = Fraction(function(point)) + Fraction((offsets or {}).get(point, "0"))
        rows.append(
            {
                "label": f"n{point}",
                "point": point,
                "point_sigma_rule": "exact",
                "source": SOURCE,
                "value": decimal_string(value, 6),
                "value_sigma": sigma,
                "value_sigma_citation": CITATION,
                "value_sigma_rule": "cited_absolute_on_exact_value",
            }
        )
    return rows


def _published(pairs):
    return [
        {
            "label": label,
            "point": point,
            "point_sigma_rule": "half_ulp_of_last_published_digit",
            "source": SOURCE,
            "value": value,
            "value_sigma_rule": "half_ulp_of_last_published_digit",
        }
        for label, point, value in pairs
    ]


def _interval(entry):
    return (
        Fraction(entry["lower"]["numerator"], entry["lower"]["denominator"]),
        Fraction(entry["upper"]["numerator"], entry["upper"]["denominator"]),
    )


@pytest.fixture(scope="module")
def controls_receipt():
    return build_controls()


# ---------------------------------------------------------------------------
# G1 -- uncertainties are declared, never fitted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("supplied", ["0.5", "0.05", "0.0000001", "0.006"])
def test_a_sigma_the_published_digit_rule_forbids_aborts_the_run(supplied):
    """The whole guard: under a published-digit rule the sigma is *derived*, not accepted."""

    row = {
        "label": "row",
        "point": 1,
        "point_sigma_rule": "exact",
        "source": SOURCE,
        "value": "1.00",
        "value_sigma": supplied,
        "value_sigma_rule": "half_ulp_of_last_published_digit",
    }
    with pytest.raises(ToleranceFittingError, match="sigma_not_derivable_from_declared_rule"):
        parse_rows([row])


def test_published_digit_rules_derive_the_sigma_they_declare():
    parsed = parse_rows(_published([("a", "0.38709893", "0.2408467")]))
    assert parsed[0].point_sigma == Fraction(5, 10**9)
    assert parsed[0].value_sigma == Fraction(5, 10**8)


def test_the_ulp_rule_and_the_half_ulp_rule_differ_by_exactly_two():
    half = parse_rows(_published([("a", "1.00", "2.000")]))[0]
    rows = _published([("a", "1.00", "2.000")])
    rows[0]["value_sigma_rule"] = "ulp_of_last_published_digit"
    full = parse_rows(rows)[0]
    assert full.value_sigma == 2 * half.value_sigma


def test_inflation_that_would_rescue_an_infeasible_run_is_refused():
    """The attack in its natural habitat: widen sigma until INFEASIBLE becomes FEASIBLE."""

    pairs = [
        ("a", "1.0000", "1.0000"),
        ("b", "2.0000", "2.9000"),
        ("c", "3.0000", "5.3000"),
        ("d", "4.0000", "8.1000"),
    ]
    honest = fit_measured(_published(pairs), coverage_k=1, ladder=POWER)
    assert honest["decision"] == "INFEASIBLE_ALL_LADDER"
    inflated = _published(pairs)
    for row in inflated:
        row["value_sigma"] = "0.5"
    with pytest.raises(ToleranceFittingError, match="sigma_not_derivable_from_declared_rule"):
        fit_measured(inflated, coverage_k=1, ladder=POWER)


def test_a_cited_absolute_sigma_needs_its_citation():
    rows = _rows(lambda n: n, range(6))
    del rows[0]["value_sigma_citation"]
    with pytest.raises(ToleranceFittingError, match="requires a nonempty"):
        parse_rows(rows)


def test_a_cited_sigma_may_not_claim_to_be_finer_than_the_printed_value():
    row = {
        "label": "row",
        "point": 1,
        "point_sigma_rule": "exact",
        "source": SOURCE,
        "value": "1.23",
        "value_sigma": "0.0001",
        "value_sigma_citation": CITATION,
        "value_sigma_rule": "cited_absolute",
    }
    with pytest.raises(ToleranceFittingError, match="finer than the published last digit"):
        parse_rows([row])


def test_every_row_must_cite_a_source():
    rows = _rows(lambda n: n, range(6))
    rows[2]["source"] = "   "
    with pytest.raises(ToleranceFittingError, match="cite a nonempty source"):
        parse_rows(rows)


def test_an_undeclared_sigma_rule_is_refused():
    rows = _rows(lambda n: n, range(6))
    rows[0]["value_sigma_rule"] = "estimated_from_the_residuals"
    with pytest.raises(ToleranceFittingError, match="undeclared sigma rule"):
        parse_rows(rows)


def test_widening_the_coverage_factor_past_the_cap_is_refused():
    with pytest.raises(ToleranceFittingError, match="coverage_k exceeds"):
        fit_measured(_rows(lambda n: n, range(6)), coverage_k=SYSTEM_CAPS["max_coverage_k"] + 1)


def test_the_api_exposes_no_way_to_scale_a_declared_sigma():
    """A guard that can be bypassed by a keyword is not a guard."""

    import inspect

    from sigma_theory_compiler import tolerance_aware_fitting as module

    parameters = set(inspect.signature(module.fit_measured).parameters)
    assert not {name for name in parameters if "scale" in name or "inflat" in name}
    assert parameters == {"rows", "coverage_k", "ladder", "exponent_probes"}
    text = Path(module.__file__).read_text(encoding="utf-8")
    assert "residual" not in text.split("def _resolve_sigma")[1].split("def parse_rows")[0]


def test_the_sigma_binding_changes_when_a_declared_uncertainty_changes():
    base = fit_measured(_rows(lambda n: n, range(6)), coverage_k=1, ladder=LINEAR)
    other = fit_measured(_rows(lambda n: n, range(6), sigma="0.001"), coverage_k=1, ladder=LINEAR)
    assert base["sigma_binding_sha256"] != other["sigma_binding_sha256"]


# ---------------------------------------------------------------------------
# G2 -- holdout stays sovereign
# ---------------------------------------------------------------------------


def test_a_holdout_row_its_own_interval_cannot_reach_rejects_the_entry():
    rows = _rows(lambda n: 3 * n + 4, range(8), sigma="0.001")
    rows[6]["value"] = decimal_string(Fraction(3 * 6 + 4) + Fraction(1, 2), 6)
    result = fit_measured(rows, coverage_k=1, ladder=LINEAR)
    rejected = {
        item["entry_id"]: item
        for item in result["minimality_certificate"]["strictly_simpler_entries_rejected"]
    }
    line = rejected["linear:polynomial_1"]
    assert line["verdict"] in {"REJECT", "INFEASIBLE"}
    assert "n6" in (line["witness"]["unreachable_rows"])


def test_confirmations_are_the_rows_left_untouched_by_the_fit():
    result = fit_measured(_rows(lambda n: 3 * n + 4, range(9)), coverage_k=1, ladder=LINEAR)
    accepted = result["result"]
    assert accepted["entry_id"] == "linear:polynomial_1"
    assert accepted["confirmations"] == 9 - accepted["parameters"]
    assert len(accepted["fit_rows"]) == accepted["linear_parameters"]
    assert len(accepted["holdout"]) == 9 - accepted["linear_parameters"]
    assert all(item["reachable"] for item in accepted["holdout"])


def test_holdout_predictions_come_from_the_fit_rows_alone():
    result = fit_measured(_rows(lambda n: 3 * n + 4, range(9)), coverage_k=1, ladder=LINEAR)
    accepted = result["result"]
    for item in accepted["holdout"]:
        predicted = item["predicted_from_fit_rows"]
        declared = _interval(item["declared_interval"])
        assert not predicted["unbounded"]
        low, high = _interval(predicted)
        assert low <= declared[1] and high >= declared[0]


# ---------------------------------------------------------------------------
# G3 -- parsimony is a budget and a rule, never a score
# ---------------------------------------------------------------------------


def test_an_entry_that_cannot_pay_its_budget_is_blocked_with_the_rule_cited():
    result = fit_measured(_rows(lambda n: n**2, range(4)), coverage_k=1, ladder=LINEAR)
    blocked = [
        item
        for item in result["minimality_certificate"]["strictly_simpler_entries_rejected"]
        if item["verdict"] == "BLOCK"
    ]
    assert blocked
    for item in blocked:
        assert item["reason"] == "parsimony_budget_violated"
        assert item["parsimony_rule"] == PARSIMONY_RULE
        assert item["rows"] - item["parameters"] < SYSTEM_CAPS["min_confirmations"]


def test_the_simplest_feasible_entry_wins_and_larger_entries_are_never_examined():
    result = fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR)
    accepted = result["result"]
    assert accepted["entry_id"] == "linear:polynomial_1"
    examined = result["parsimony_comparison"]
    assert all(item["parameters"] <= accepted["parameters"] for item in examined)


def test_the_parsimony_rule_is_stated_in_the_receipt():
    result = fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR)
    assert result["minimality_certificate"]["parsimony_rule"] == PARSIMONY_RULE
    assert "never compared by residual magnitude" in PARSIMONY_RULE


def test_every_strictly_simpler_entry_carries_its_own_rejection():
    result = fit_measured(_rows(lambda n: n * (n + 1) // 2, range(10)), coverage_k=1, ladder=LINEAR)
    rejected = result["minimality_certificate"]["strictly_simpler_entries_rejected"]
    accepted = result["result"]
    assert rejected
    for item in rejected:
        assert item["verdict"] in {"INFEASIBLE", "REJECT", "BLOCK", "SKIP"}
        assert item["reason"]
        assert item["parameters"] <= accepted["parameters"]


# ---------------------------------------------------------------------------
# G4 -- exactness, and certificates that re-verify without trusting the solver
# ---------------------------------------------------------------------------


def test_the_farkas_witness_re_verifies_from_the_receipt_alone():
    """lambda >= 0, lambda^T A = 0, lambda^T b < 0 -- recomputed here from the declared rows."""

    rows = _rows(lambda n: n**3, range(8), sigma="0.000001")
    result = fit_measured(rows, coverage_k=1, ladder=LINEAR)
    entry = next(
        item
        for item in result["minimality_certificate"]["strictly_simpler_entries_rejected"]
        if item["entry_id"] == "linear:polynomial_1" and item["verdict"] == "INFEASIBLE"
    )
    witness = entry["witness"]
    assert witness["kind"] == "farkas_nonnegative_combination"
    declared = {item["label"]: item for item in result["declared_rows"]}
    combination = [Fraction(0), Fraction(0)]
    total = Fraction(0)
    for term in witness["terms"]:
        multiplier = Fraction(term["multiplier"]["numerator"], term["multiplier"]["denominator"])
        assert multiplier >= 0
        row = declared[term["row"]]
        point = Fraction(row["point_exact"]["numerator"], row["point_exact"]["denominator"])
        low, high = _interval(row["declared_value_interval"])
        sign = 1 if term["bound"] == "upper" else -1
        combination[0] += multiplier * sign
        combination[1] += multiplier * sign * point
        total += multiplier * (high if sign == 1 else -low)
    assert combination == [Fraction(0), Fraction(0)]
    assert total < 0
    assert total == Fraction(
        witness["combined_right_hand_side"]["numerator"],
        witness["combined_right_hand_side"]["denominator"],
    )


def test_the_feasible_witness_point_re_verifies_from_the_receipt_alone():
    """A FEASIBLE verdict must be checkable too: the reported point really is inside."""

    result = fit_measured(
        _rows(lambda n: 2 * n**3 + 2 * n**2 + n + 7, range(10), "0.01"), coverage_k=1, ladder=LINEAR
    )
    accepted = result["result"]
    coefficients = [
        Fraction(item["numerator"], item["denominator"]) for item in accepted["witness_point"]
    ]
    assert len(coefficients) == accepted["parameters"]
    for row in result["declared_rows"]:
        point = Fraction(row["point_exact"]["numerator"], row["point_exact"]["denominator"])
        predicted = sum(
            (coefficient * point**degree for degree, coefficient in enumerate(coefficients)),
            Fraction(0),
        )
        low, high = _interval(row["declared_value_interval"])
        assert low <= predicted <= high, row["label"]


def test_the_reported_coefficient_region_really_is_a_region():
    """Every corner-adjacent claim: the box bounds are attained, not decorative."""

    result = fit_measured(
        _rows(lambda n: 3 * n + 4, range(10), "0.01"), coverage_k=1, ladder=LINEAR
    )
    accepted = result["result"]
    intervals = {item["term"]: item for item in accepted["coefficient_intervals"]}
    assert set(intervals) == {"1", "n^1"}
    for item in intervals.values():
        low, high = _interval(item)
        assert item["unbounded"] is False
        assert low < high, "a declared tolerance must leave the coefficient genuinely free"
    slope_low, slope_high = _interval(intervals["n^1"])
    assert slope_low <= 3 <= slope_high


def test_the_power_law_witness_re_verifies_from_the_declared_rows():
    rows = _published(
        [("a", "1.0000", "1.0000"), ("b", "2.0000", "2.8000"), ("c", "4.0000", "8.1000")]
    )
    result = decide_power_law(parse_rows(rows), Fraction(3, 2), Fraction(1))
    assert result["verdict"] == "INFEASIBLE"
    witness = result["witness"]
    parsed = {row.label: row for row in parse_rows(rows)}
    low_row = parsed[witness["requires_at_least"]["row"]]
    high_row = parsed[witness["requires_at_most"]["row"]]
    low = invariant_interval(low_row, Fraction(3, 2), Fraction(1))[0]
    high = invariant_interval(high_row, Fraction(3, 2), Fraction(1))[1]
    assert low > high
    assert low == Fraction(
        witness["requires_at_least"]["bound"]["numerator"],
        witness["requires_at_least"]["bound"]["denominator"],
    )


def test_the_invariant_interval_is_the_exact_range_over_the_declared_box():
    row = parse_rows(_published([("a", "2.00", "3.00")]))[0]
    low, high = invariant_interval(row, Fraction(3, 2), Fraction(1))
    corners = [
        value**2 / point**3
        for value in (Fraction("2.995"), Fraction("3.005"))
        for point in (Fraction("1.995"), Fraction("2.005"))
    ]
    assert low == min(corners)
    assert high == max(corners)


def test_point_uncertainty_is_refused_by_the_linear_track_rather_than_ignored():
    rows = _published([("a", "1.0", "1.0"), ("b", "2.0", "2.0"), ("c", "3.0", "3.0"), ("d", "4.0", "4.0")])
    result = fit_measured(rows, coverage_k=1, ladder=LINEAR)
    skipped = result["minimality_certificate"]["strictly_simpler_entries_rejected"]
    assert skipped
    assert all(item["reason"] == "linear_basis_track_requires_exact_points" for item in skipped)
    assert result["decision"] == "BLOCKED"


def test_the_outward_certification_agrees_with_the_exact_test():
    rows = parse_rows(
        _published([("a", "1.0000", "1.0000"), ("b", "4.0000", "8.0000"), ("c", "9.0000", "27.0000")])
    )
    for exponent in (Fraction(3, 2), Fraction(1), Fraction(2), Fraction(-1)):
        exact = decide_power_law(rows, exponent, Fraction(1))["verdict"]
        outward = certify_power_law_outward(rows, exponent, Fraction(1))
        assert outward != "unresolved_straddle"
        assert (outward == "certified_feasible") == (exact == "FEASIBLE")


def test_a_non_positive_declared_interval_is_refused_by_the_power_law_track():
    rows = parse_rows(
        _published([("a", "0.001", "1.0"), ("b", "2.0", "2.0"), ("c", "3.0", "3.0"), ("d", "4.0", "4.0")])
    )
    with pytest.raises(ToleranceFittingError, match="strictly positive"):
        invariant_interval(rows[0], Fraction(3, 2), Fraction(1000))


# ---------------------------------------------------------------------------
# G5 -- no scalar goodness score anywhere in a receipt
# ---------------------------------------------------------------------------


def test_no_receipt_carries_a_scalar_goodness_key(controls_receipt):
    for receipt in (
        fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR),
        fit_measured(_rows(lambda n: n**3, range(8), sigma="0.000001"), coverage_k=1, ladder=LINEAR),
        controls_receipt,
    ):
        assert forbidden_receipt_keys(receipt) == []


def test_a_smuggled_score_is_caught_by_validation():
    result = fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR)
    for key in ("chi_squared", "r_squared", "aic", "fit_quality", "p_value"):
        tampered = dict(result)
        tampered[key] = 1
        with pytest.raises(ToleranceFittingError, match="scalar goodness key"):
            validate_result(tampered, ladder=LINEAR)


def test_the_forbidden_token_list_covers_the_usual_suspects():
    for token in ("aic", "bic", "chi_squared", "p_value", "r_squared", "likelihood", "score"):
        assert token in FORBIDDEN_RECEIPT_KEY_TOKENS


# ---------------------------------------------------------------------------
# The five mandatory controls
# ---------------------------------------------------------------------------


def test_every_mandatory_control_is_satisfied(controls_receipt):
    assert controls_receipt["decision"] == "PASS"
    assert controls_receipt["counts"]["controls"] == 5
    for control in controls_receipt["controls"]:
        assert control["satisfied"], control["control_id"]


def test_control_a_matches_b1_family_for_family(controls_receipt):
    control = next(
        item for item in controls_receipt["controls"] if item["control_id"].startswith("a_")
    )
    assert len(control["detail"]) >= 5
    for item in control["detail"]:
        assert item["b1_decision"] == "PASS"
        assert item["b8_decision"] == "FEASIBLE_MINIMAL"
        assert item["b8_entry_id"] == f"linear:{item['b1_family_id']}"


@pytest.mark.parametrize(
    ("name", "function", "points"),
    [
        ("triangular", lambda n: n * (n + 1) // 2, tuple(range(10))),
        ("cubic", lambda n: 2 * n**3 + 2 * n**2 + n + 7, tuple(range(10))),
        ("geometric", lambda n: 3 * 2**n, tuple(range(10))),
        ("shifted_reciprocal", lambda n: Fraction(1, n + 1), tuple(range(10))),
    ],
)
def test_control_a_agreement_extends_beyond_the_sealed_list(name, function, points):
    # sigma must clear the six-decimal rendering of the row values, which is itself a
    # rounding of the exact sequence; 1e-6 is above the 5e-7 that rendering can introduce
    fitted = fit_measured(_rows(function, points, "0.000001"), coverage_k=1, ladder=LINEAR)
    exact = synthesize_basis(
        [{"point": point, "value": _fraction_row(function(point))} for point in points]
    )
    assert exact["decision"] == "PASS", name
    assert fitted["result"]["entry_id"] == f"linear:{exact['result']['family_id']}", name


def _fraction_row(value):
    fraction = Fraction(value)
    return {"numerator": fraction.numerator, "denominator": fraction.denominator}


def test_control_b_reports_the_true_coefficients_inside_the_region(controls_receipt):
    control = next(
        item for item in controls_receipt["controls"] if item["control_id"].startswith("b_")
    )
    coverage = control["detail"]["coefficient_coverage"]
    assert len(coverage) == 4
    assert all(item["contains_true_value"] for item in coverage)


def test_control_c_is_infeasible_not_merely_worse(controls_receipt):
    control = next(
        item for item in controls_receipt["controls"] if item["control_id"].startswith("c_")
    )
    assert control["detail"]["decision"] == "INFEASIBLE_ALL_LADDER"


def test_control_d_shows_the_blocked_entry_would_otherwise_have_fitted(controls_receipt):
    control = next(
        item for item in controls_receipt["controls"] if item["control_id"].startswith("d_")
    )
    starved = control["detail"]["budget_starved"]
    assert starved["interpolating_entry_reaches_every_interval_when_the_budget_is_ignored"]
    assert starved["decision"] == "INFEASIBLE_ALL_LADDER"
    assert starved["selected_entry_id"] is None
    assert control["detail"]["simpler_entry_wins"]["selected_entry_id"] == "linear:polynomial_1"
    assert control["detail"]["simpler_entry_wins"]["larger_entries_examined_after_acceptance"] == []


def test_control_e_refuses_inflation_deflation_and_coverage_widening(controls_receipt):
    control = next(
        item for item in controls_receipt["controls"] if item["control_id"].startswith("e_")
    )
    attacks = {item["attack_id"]: item for item in control["detail"]}
    assert set(attacks) == {
        "deflate_declared_half_ulp_sigma",
        "inflate_declared_half_ulp_sigma",
        "widen_the_coverage_factor",
    }
    assert all(item["refused"] for item in attacks.values())


# ---------------------------------------------------------------------------
# Declared ladder, caps, and claims
# ---------------------------------------------------------------------------


def test_the_ladder_order_is_frozen():
    assert [entry.entry_id for entry in POWER[:8]] == [
        "power_law:0",
        "power_law:-1",
        "power_law:1",
        "power_law:-2",
        "power_law:2",
        "power_law:-1/2",
        "power_law:1/2",
        "power_law:-3",
    ]
    assert LINEAR[0].entry_id == "linear:constant"
    assert all(
        LINEAR[index].sort_key <= LINEAR[index + 1].sort_key for index in range(len(LINEAR) - 1)
    )


def test_the_exponent_grid_bounds_are_capped():
    with pytest.raises(ToleranceFittingError, match="denominator bound"):
        build_ladder(
            include_linear=False,
            exponent_denominator_bound=SYSTEM_CAPS["max_exponent_denominator_bound"] + 1,
        )
    with pytest.raises(ToleranceFittingError, match="absolute bound"):
        build_ladder(
            include_linear=False,
            exponent_absolute_bound=SYSTEM_CAPS["max_exponent_absolute_bound"] + 1,
        )


def test_the_declared_caps_and_claims_are_pinned():
    assert SYSTEM_CAPS["min_confirmations"] == 2
    assert SYSTEM_CAPS["max_coverage_k"] == 6
    assert CLAIMS["uncertainties_are_declared_never_fitted"] is True
    assert CLAIMS["models_ranked_by_residual_magnitude"] is False
    assert CLAIMS["coefficient_output_is_a_region_not_a_point"] is True
    assert set(SIGMA_RULES) == {
        "cited_absolute",
        "cited_absolute_on_exact_value",
        "exact",
        "half_ulp_of_last_published_digit",
        "propagated_outward",
        "ulp_of_last_published_digit",
    }


def test_the_row_budget_is_capped():
    with pytest.raises(ToleranceFittingError, match="row count exceeds cap"):
        parse_rows(_rows(lambda n: n, range(SYSTEM_CAPS["max_rows"] + 1)))


def test_a_duplicate_row_label_is_refused():
    rows = _rows(lambda n: n, range(6))
    rows[3]["label"] = rows[0]["label"]
    with pytest.raises(ToleranceFittingError, match="unique nonempty string"):
        parse_rows(rows)


# ---------------------------------------------------------------------------
# Determinism, replay, and tamper
# ---------------------------------------------------------------------------


def test_the_receipt_is_byte_stable_across_runs():
    from sigma_theory_compiler.sigma_core import canonical_json_bytes

    first = fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR)
    second = fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_validation_replays_the_receipt_exactly():
    result = fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR)
    validate_result(result, ladder=LINEAR)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"decision": "BLOCKED"}),
        lambda value: value.update({"content_sha256": "0" * 64}),
        lambda value: value["declared_rows"][0].update({"value": "999.0"}),
        lambda value: value["coverage_factor"].update({"k": "2"}),
        lambda value: value.update({"sigma_binding_sha256": "0" * 64}),
    ],
)
def test_tamper_is_rejected(mutate):
    result = json.loads(
        json.dumps(fit_measured(_rows(lambda n: 3 * n + 4, range(10)), coverage_k=1, ladder=LINEAR))
    )
    mutate(result)
    with pytest.raises(ToleranceFittingError):
        validate_result(result, ladder=LINEAR)


def test_the_sealed_control_receipt_on_disk_replays(controls_receipt):
    path = REPOSITORY_ROOT / "runs" / "math" / "tolerance-fitting" / "controls-v1.json"
    sealed = json.loads(path.read_text(encoding="utf-8"))
    validate_controls(sealed)
    assert sealed == controls_receipt


# ---------------------------------------------------------------------------
# Certified exponent bracket
# ---------------------------------------------------------------------------


def test_the_bracket_pins_a_clean_power_law_and_excludes_the_competitors():
    rows = _published(
        [
            ("a", "1.00000000", "1.00000000"),
            ("b", "4.00000000", "8.00000000"),
            ("c", "9.00000000", "27.00000000"),
            ("d", "16.00000000", "64.00000000"),
        ]
    )
    bracket = certified_exponent_bracket(
        rows, coverage_k=1, centre="3/2", outer_offset="1/100", places=20
    )
    assert bracket["centre_verdict"] == "certified_feasible"
    for side in bracket["sides"].values():
        assert side["outer_probe_verdict"] == "certified_infeasible"
        assert side["bracketed"] is True
        assert abs(Fraction(side["certified_feasible_at"]) - Fraction(3, 2)) < Fraction(1, 100)


def test_the_bracket_declares_that_connectedness_is_assumed_not_proved():
    rows = _published(
        [
            ("a", "1.00000000", "1.00000000"),
            ("b", "4.00000000", "8.00000000"),
            ("c", "9.00000000", "27.00000000"),
            ("d", "16.00000000", "64.00000000"),
        ]
    )
    bracket = certified_exponent_bracket(
        rows, coverage_k=1, centre="3/2", outer_offset="1/100", places=12
    )
    assert "not proved" in bracket["connectedness_of_the_feasible_set"]
    assert bracket["rounding"] == "outward on every interval operation"


# ---------------------------------------------------------------------------
# Probes and the ladder-free surface
# ---------------------------------------------------------------------------


def test_exponent_probes_are_decided_and_recorded_but_never_selected():
    rows = _published(
        [
            ("a", "1.00000000", "1.00000000"),
            ("b", "4.00000000", "8.00000000"),
            ("c", "9.00000000", "27.00000000"),
            ("d", "16.00000000", "64.00000000"),
        ]
    )
    result = fit_measured(
        rows, coverage_k=1, ladder=POWER, exponent_probes=("149/100", "151/100")
    )
    assert result["result"]["entry_id"] == "power_law:3/2"
    assert {probe["exponent"]: probe["verdict"] for probe in result["exponent_probes"]} == {
        "149/100": "INFEASIBLE",
        "151/100": "INFEASIBLE",
    }


def test_a_custom_ladder_of_one_entry_still_certifies_minimality():
    entry = LadderEntry(
        entry_id="power_law:3/2",
        track="power_law",
        track_rank=1,
        entry_rank=0,
        parameters=2,
        linear_parameters=1,
        exponent=Fraction(3, 2),
    )
    rows = _published(
        [
            ("a", "1.00000000", "1.00000000"),
            ("b", "4.00000000", "8.00000000"),
            ("c", "9.00000000", "27.00000000"),
            ("d", "16.00000000", "64.00000000"),
        ]
    )
    result = fit_measured(rows, coverage_k=1, ladder=[entry])
    assert result["decision"] == "FEASIBLE_MINIMAL"
    assert result["minimality_certificate"]["strictly_simpler_entries_rejected"] == []
