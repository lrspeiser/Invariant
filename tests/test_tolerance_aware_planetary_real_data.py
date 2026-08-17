"""B8 on real published planetary elements.

The point of this file is that the published cohorts are allowed to come back INFEASIBLE
and the run still passes.  What is pinned is the *instrument*: that the matched two-body
control recovers 3/2 and -2 with the true constants inside the reported regions, that the
recovered exponent is pinned well inside a hundredth, that every cohort reaches a decided
verdict rather than failing to decide, and that the infeasibility on the published rows is
carried by a named witness a reader can recheck by hand.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.tolerance_aware_fitting import (
    decide_power_law,
    forbidden_receipt_keys,
    invariant_interval,
    parse_rows,
)
from sigma_theory_compiler.tolerance_aware_planetary_real_data import (
    CLAIMS,
    OUTPUT_PATH,
    RECEIPT_DIRECTORY,
    PlanetaryRealDataError,
    build_campaign,
    counterfactual_kepler_rows,
    load_anchors,
    published_kepler_rows,
    validate_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FOUR_PI_SQUARED = Fraction("39.47841760435743447533796399950460454125479762896316")


@pytest.fixture(scope="module")
def artifacts():
    return build_campaign(REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def sealed():
    campaign = json.loads((REPOSITORY_ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    cohorts = {
        summary["cohort_id"]: json.loads(
            (
                REPOSITORY_ROOT / f"{RECEIPT_DIRECTORY}/cohort-{summary['cohort_id']}.json"
            ).read_text(encoding="utf-8")
        )
        for summary in campaign["cohort_summaries"]
    }
    return campaign, cohorts


def _summary(campaign, cohort_id):
    return next(item for item in campaign["cohort_summaries"] if item["cohort_id"] == cohort_id)


# ---------------------------------------------------------------------------
# The published values really are the published values
# ---------------------------------------------------------------------------


def test_the_published_rows_are_the_printed_values_not_the_re_anchored_ones(artifacts):
    """The whole point: nothing here is the campaign's exactly-Keplerian reconstruction."""

    anchors, _, _ = load_anchors(REPOSITORY_ROOT)
    published = {row["label"]: row for row in published_kepler_rows(anchors)}
    counterfactual = {row["label"]: row for row in counterfactual_kepler_rows(anchors)}
    by_body = {anchor["body"].lower().replace(" ", "_"): anchor for anchor in anchors}
    differing = 0
    for label, row in published.items():
        assert row["point"] == by_body[label]["semi_major_axis_au"]
        assert row["value"] == by_body[label]["sidereal_orbit_period_yr"]
        if counterfactual[label]["value"] != row["value"]:
            differing += 1
    assert differing >= 8, "the counterfactual must actually differ from the published data"
    assert artifacts["campaign"]["claims"]["re_anchored_values_used_in_the_published_cohorts"] is False


def test_the_counterfactual_is_printed_to_the_same_precision_as_the_published_data():
    anchors, _, _ = load_anchors(REPOSITORY_ROOT)
    published = {row["label"]: row for row in published_kepler_rows(anchors)}
    for row in counterfactual_kepler_rows(anchors):
        real = published[row["label"]]
        for field in ("point", "value"):
            assert len(row[field].partition(".")[2]) == len(real[field].partition(".")[2])


def test_every_row_declares_the_published_digit_rule(artifacts):
    for cohort in artifacts["cohorts"].values():
        for row in cohort["fit_receipt"]["declared_rows"]:
            assert row["point_sigma_rule"] == "half_ulp_of_last_published_digit"
            assert row["value_sigma_rule"] in {
                "half_ulp_of_last_published_digit",
                "propagated_outward",
            }
            assert row["source"].strip()


def test_the_anchor_fixture_is_bound_by_content_hash(artifacts):
    import hashlib

    binding = artifacts["campaign"]["source_bindings"]["anchor_fixture"]
    raw = (REPOSITORY_ROOT / binding["path"]).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert binding["file_sha256"] == hashlib.sha256(normalized).hexdigest()


# ---------------------------------------------------------------------------
# The real-data result
# ---------------------------------------------------------------------------


def test_the_published_kepler_cohort_is_decided_and_infeasible(artifacts):
    summary = _summary(artifacts["campaign"], "published_kepler")
    assert summary["decision"] == "INFEASIBLE_ALL_LADDER"
    assert summary["recovered_exponent"] is None
    assert summary["target_exponent_verdict_at_centre"] == "certified_infeasible"
    assert summary["exclusion_probe_verdicts"] == {"149/100": "INFEASIBLE", "151/100": "INFEASIBLE"}


def test_the_published_infeasibility_names_two_rows_and_rechecks_by_hand(artifacts):
    """An INFEASIBLE verdict on real data is only worth anything if it is checkable."""

    summary = _summary(artifacts["campaign"], "published_kepler")
    witness = summary["witness"]
    assert witness["kind"] == "disjoint_invariant_intervals"
    anchors, _, _ = load_anchors(REPOSITORY_ROOT)
    rows = {row.label: row for row in parse_rows(published_kepler_rows(anchors))}
    low = invariant_interval(
        rows[witness["requires_at_least"]["row"]], Fraction(3, 2), Fraction(1)
    )[0]
    high = invariant_interval(
        rows[witness["requires_at_most"]["row"]], Fraction(3, 2), Fraction(1)
    )[1]
    assert low > high, "the two named rows must actually be irreconcilable"
    assert float(low - high) > 1e-4


def test_the_counterfactual_recovers_three_halves_with_the_constant_containing_one(artifacts):
    summary = _summary(artifacts["campaign"], "two_body_counterfactual_kepler")
    assert summary["decision"] == "FEASIBLE_MINIMAL"
    assert summary["recovered_exponent"] == "3/2"
    assert summary["simpler_exponents_rejected"] == 12
    lower, upper = (Fraction(text) for text in summary["constant_interval_decimal"])
    assert lower <= 1 <= upper


def test_the_counterfactual_recovers_minus_two_with_the_constant_containing_four_pi_squared(
    artifacts,
):
    summary = _summary(artifacts["campaign"], "two_body_counterfactual_inverse_square")
    assert summary["decision"] == "FEASIBLE_MINIMAL"
    assert summary["recovered_exponent"] == "-2"
    lower, upper = (Fraction(text) for text in summary["constant_interval_decimal"])
    assert lower <= FOUR_PI_SQUARED <= upper


def test_the_published_inverse_square_cohort_is_decided_and_infeasible(artifacts):
    summary = _summary(artifacts["campaign"], "published_inverse_square")
    assert summary["decision"] == "INFEASIBLE_ALL_LADDER"
    assert summary["exclusion_probe_verdicts"] == {
        "-199/100": "INFEASIBLE",
        "-201/100": "INFEASIBLE",
    }


@pytest.mark.parametrize(
    ("cohort_id", "target"),
    [("two_body_counterfactual_kepler", "3/2"), ("two_body_counterfactual_inverse_square", "-2")],
)
def test_the_recovered_exponent_is_pinned_far_inside_a_hundredth(artifacts, cohort_id, target):
    summary = _summary(artifacts["campaign"], cohort_id)
    bracket = summary["exponent_bracket"]
    centre = Fraction(target)
    lower = Fraction(bracket["lower"]["certified_feasible_at"])
    upper = Fraction(bracket["upper"]["certified_feasible_at"])
    assert lower <= centre <= upper
    assert upper - lower < Fraction(1, 10**7), "the bracket must be far tighter than +/-0.01"
    assert abs(upper - centre) < Fraction(1, 100)
    assert abs(centre - lower) < Fraction(1, 100)


def test_the_infeasible_cohorts_certify_the_target_exponent_infeasible_outward_too(artifacts):
    for cohort_id in ("published_kepler", "published_inverse_square"):
        cohort = artifacts["cohorts"][cohort_id]
        assert cohort["certified_exponent_bracket"]["centre_verdict"] == "certified_infeasible"
        for side in cohort["certified_exponent_bracket"]["sides"].values():
            assert side["bracketed"] is False


# ---------------------------------------------------------------------------
# The residual the tolerance refuses to absorb
# ---------------------------------------------------------------------------


def test_the_run_reproduces_the_sealed_campaign_fidelity_figure(artifacts):
    """An independent recomputation of the number the blind campaign already declared."""

    residual = artifacts["campaign"]["multi_body_residual"]
    assert residual["reproduces_the_sealed_campaign_fidelity_figure"] is True
    assert residual["worst_body"] == "uranus"
    assert (
        residual["this_run_recomputed_max_relative_deviation"]
        == residual["sealed_campaign_declared_max_relative_deviation"]
    )


def test_the_deviation_is_thousands_of_declared_half_widths_wide(artifacts):
    residual = artifacts["campaign"]["multi_body_residual"]
    by_row = {item["row"]: item for item in residual["bodies"]}
    for body in ("jupiter", "saturn", "uranus", "neptune"):
        widths = Fraction(by_row[body]["deviation_in_units_of_the_declared_half_width"])
        assert widths > 1000, body


def test_the_coverage_diagnostic_is_labelled_as_never_deciding_anything(artifacts):
    diagnostic = artifacts["campaign"]["coverage_factor_diagnostic"]
    assert diagnostic["all_published_rows"]["is_a_decision"] is False
    assert diagnostic["witness_pair"]["is_a_decision"] is False
    assert diagnostic["witness_pair"]["bracketed"] is True
    assert Fraction(diagnostic["witness_pair"]["certified_infeasible_at_k"]) > 1000
    assert artifacts["campaign"]["coverage_factor"] == "1"


def test_widening_to_the_diagnostic_factor_really_would_have_flipped_the_verdict(artifacts):
    """The diagnostic is only meaningful if the number it reports is the real threshold."""

    diagnostic = artifacts["campaign"]["coverage_factor_diagnostic"]["witness_pair"]
    anchors, _, _ = load_anchors(REPOSITORY_ROOT)
    rows = {row.label: row for row in parse_rows(published_kepler_rows(anchors))}
    pair = [rows[label] for label in diagnostic["rows"]]
    below = Fraction(diagnostic["certified_infeasible_at_k"])
    above = Fraction(diagnostic["certified_feasible_at_k"]) + Fraction(1, 1000)
    assert decide_power_law(pair, Fraction(3, 2), below)["verdict"] == "INFEASIBLE"
    assert decide_power_law(pair, Fraction(3, 2), above)["verdict"] == "FEASIBLE"


def test_the_derived_inverse_square_column_encloses_its_exact_range(artifacts):
    audit = artifacts["campaign"]["derived_column_audit"]
    assert set(audit) == {"published_inverse_square", "two_body_counterfactual_inverse_square"}
    for rows in audit.values():
        assert rows
        assert all(item["exact_range_contained"] for item in rows)


def test_the_correlated_derivation_is_declared_rather_than_hidden(artifacts):
    cohort = artifacts["cohorts"]["published_inverse_square"]
    assert "correlated rounding as independent" in cohort["boundary"]
    assert "not independent evidence" in cohort["boundary"]


# ---------------------------------------------------------------------------
# Policy, determinism, and tamper
# ---------------------------------------------------------------------------


def test_the_pass_policy_is_about_the_instrument_not_about_nature(artifacts):
    campaign = artifacts["campaign"]
    assert campaign["decision"] == "PASS"
    assert all(campaign["checks"].values())
    assert "not about nature" in campaign["instrument_pass_policy"]
    assert "is a finding about the published numbers" in campaign["instrument_pass_policy"]
    # a PASS coexists with two INFEASIBLE published cohorts, by design
    infeasible = [
        item["cohort_id"]
        for item in campaign["cohort_summaries"]
        if item["decision"] == "INFEASIBLE_ALL_LADDER"
    ]
    assert sorted(infeasible) == ["published_inverse_square", "published_kepler"]


def test_the_claims_and_sigma_convention_are_pinned(artifacts):
    campaign = artifacts["campaign"]
    assert campaign["claims"] == CLAIMS
    assert CLAIMS["novelty_claimed"] is False
    assert CLAIMS["uncertainties_adjusted_to_obtain_a_verdict"] is False
    assert CLAIMS["observational_archive_opened"] is False
    assert "rounding bound, not a physical error bar" in campaign["sigma_convention"]


def test_no_campaign_receipt_carries_a_scalar_goodness_key(artifacts):
    assert forbidden_receipt_keys(artifacts["campaign"]) == []
    for cohort in artifacts["cohorts"].values():
        assert forbidden_receipt_keys(cohort) == []


def test_the_campaign_is_byte_stable_and_replays(artifacts, sealed):
    from sigma_theory_compiler.sigma_core import canonical_json_bytes

    campaign, cohorts = sealed
    assert canonical_json_bytes(campaign) == canonical_json_bytes(artifacts["campaign"])
    assert cohorts == artifacts["cohorts"]
    validate_artifacts(campaign, cohorts, root=REPOSITORY_ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"decision": "BLOCK"}),
        lambda value: value.update({"content_sha256": "0" * 64}),
        lambda value: value["cohort_summaries"][0].update({"decision": "FEASIBLE_MINIMAL"}),
        lambda value: value["multi_body_residual"].update({"worst_body": "mars"}),
    ],
)
def test_campaign_tamper_is_rejected(sealed, mutate):
    campaign, cohorts = sealed
    tampered = json.loads(json.dumps(campaign))
    mutate(tampered)
    with pytest.raises(PlanetaryRealDataError):
        validate_artifacts(tampered, cohorts, root=REPOSITORY_ROOT)


def test_a_tampered_cohort_receipt_is_rejected(sealed):
    campaign, cohorts = sealed
    tampered = json.loads(json.dumps(cohorts))
    tampered["published_kepler"]["decision"] = "FEASIBLE_MINIMAL"
    with pytest.raises(PlanetaryRealDataError):
        validate_artifacts(campaign, tampered, root=REPOSITORY_ROOT)
