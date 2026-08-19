"""Tests for Tier 7 R1 (per-object decomposition) and R2 (parameter-variation diagnosis).

Each test names the thing that would embarrass the claim if it stopped holding, and every
positive result here is paired with a control that must fail.  The three that matter most:

* a confirmation-set object handed to any fitting entry point must raise, not be declined
  politely -- that is R1's falsifier turned into a call-graph guard;
* a constancy verdict must be re-derivable from the exhibited intervals, and claiming
  CONSTANT over disjoint intervals (or VARIES when a shared value exists) must raise -- that
  is R2's falsifier turned into a guard;
* the deliberately wrong law must need a strictly larger coverage than every real family,
  and must *still* price universality at 1, which is what stops the price from being read
  as a score.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.per_object_law_decomposition import (
    CLAIMS as R1_CLAIMS,
)
from sigma_theory_compiler.per_object_law_decomposition import (
    CONFIRMATION_COUNT,
    CONSTANT,
    COVERAGE_GRID,
    NO_POPULATION,
    RECEIPT_PATH,
    RESULT_SCHEMA,
    SPLIT_RULE,
    SPLIT_SALT,
    STRUCTURE_RUN_DIVISOR,
    TRIAL_TYPE,
    VARIES,
    Axis,
    ConfirmationSetTouched,
    PerObjectError,
    adjudicate,
    build_axis,
    build_law_spaces,
    build_receipt,
    critical_coverage,
    declare_split,
    decompose,
    diagnose,
    interval_at,
    residual_structure,
    two_parameter_diagnosis,
    validate_receipt,
    verify_adjudication,
)
from sigma_theory_compiler.real_data_gravity_confrontation import (
    QUADRATURE,
    REFERENCE_GRID_POINT,
    ColumnCache,
    _family_columns,
    load_families,
    load_galaxies,
    measured_rows,
    prepare_galaxy,
    select_best_family,
)
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.tolerance_aware_fitting import (
    FEASIBLE,
    INFEASIBLE,
    forbidden_receipt_keys,
    parse_rows,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures.  The receipt is built once; everything else is cheap.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(ROOT)


@pytest.fixture(scope="module")
def sealed() -> dict:
    return json.loads((ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalogue():
    galaxies, provenance = load_galaxies(ROOT)
    families = load_families(ROOT)
    split = declare_split([galaxy.name for galaxy in galaxies])
    exploration = [galaxy for galaxy in galaxies if galaxy.name in set(split.exploration)]
    withheld = [galaxy for galaxy in galaxies if galaxy.name in set(split.confirmation)]
    convention = provenance["mass_to_light_convention"]
    prepared = {
        galaxy.name: prepare_galaxy(
            galaxy,
            Fraction(convention["disk_3_6um"]),
            Fraction(convention["bulge_3_6um"]),
            QUADRATURE,
        )
        for galaxy in galaxies
    }
    rows = {galaxy.name: measured_rows(galaxy, "test source citation") for galaxy in galaxies}
    cache = ColumnCache(prepared)
    spaces = build_law_spaces(
        families,
        prepared,
        cache,
        REFERENCE_GRID_POINT["a0"],
        REFERENCE_GRID_POINT["length_unit"],
    )
    return {
        "cache": cache,
        "exploration": exploration,
        "families": families,
        "prepared": prepared,
        "rows": rows,
        "spaces": {space.name: space for space in spaces},
        "split": split,
        "withheld": withheld,
    }


def _row(label: str, value: str, sigma: str):
    return parse_rows(
        [
            {
                "label": label,
                "point": "0",
                "point_sigma_rule": "exact",
                "source": "synthetic control",
                "value": value,
                "value_sigma": sigma,
                "value_sigma_citation": "synthetic control row; no measurement is claimed",
                "value_sigma_rule": "cited_absolute_on_exact_value",
            }
        ]
    )[0]


def _interval(lower: str, upper: str) -> dict:
    return {
        "empty": False,
        "lower": {"decimal": lower, "exact": _exact(lower)},
        "upper": {"decimal": upper, "exact": _exact(upper)},
    }


def _exact(text: str) -> dict:
    value = Fraction(text)
    return {"denominator": value.denominator, "numerator": value.numerator}


def _frac(block) -> Fraction:
    return Fraction(int(block["exact"]["numerator"]), int(block["exact"]["denominator"]))


# ---------------------------------------------------------------------------
# The split: declared from names alone, sealed, and structurally enforced
# ---------------------------------------------------------------------------


def test_the_split_is_a_partition_computed_from_names_alone(catalogue) -> None:
    split = catalogue["split"]
    assert len(split.confirmation) == CONFIRMATION_COUNT
    assert not set(split.confirmation) & set(split.exploration)
    names = [galaxy.name for galaxy in catalogue["exploration"] + catalogue["withheld"]]
    assert sorted(split.exploration + split.confirmation) == sorted(names)
    # Recomputing from a shuffled name list must give the same partition: the rule reads
    # names, never data, and never the order they arrived in.
    again = declare_split(sorted(names, reverse=True))
    assert again.confirmation == split.confirmation
    assert again.exploration == split.exploration


def test_the_split_is_driven_by_the_declared_salt_and_not_hard_coded(catalogue) -> None:
    """Control: a different salt must move the partition, or the rule is decoration."""

    import sigma_theory_compiler.per_object_law_decomposition as module

    names = [galaxy.name for galaxy in catalogue["exploration"] + catalogue["withheld"]]
    baseline = declare_split(names)
    moved = []
    for probe in ("probe-a", "probe-b", "probe-c", "probe-d"):
        original = module.SPLIT_SALT
        try:
            module.SPLIT_SALT = probe
            moved.append(declare_split(names).confirmation)
        finally:
            module.SPLIT_SALT = original
    assert module.SPLIT_SALT == SPLIT_SALT
    assert any(item != baseline.confirmation for item in moved)


def test_declaring_a_split_over_a_degenerate_population_is_refused() -> None:
    with pytest.raises(PerObjectError):
        declare_split(["A", "A", "B", "C"])
    with pytest.raises(PerObjectError):
        declare_split(["A", "B"])


def test_fitting_a_confirmation_set_object_raises(catalogue) -> None:
    """The R1 falsifier as a guard: a withheld object must not be fittable at all."""

    split = catalogue["split"]
    law = catalogue["spaces"]["newtonian_baryons_only"]
    assert catalogue["withheld"], "the split withheld nothing, so this control proves nothing"
    with pytest.raises(ConfirmationSetTouched):
        decompose(law, catalogue["withheld"], catalogue["rows"], split)
    with pytest.raises(ConfirmationSetTouched):
        decompose(law, catalogue["exploration"] + catalogue["withheld"], catalogue["rows"], split)


def test_the_two_parameter_entry_point_also_refuses_the_confirmation_set(catalogue) -> None:
    split = catalogue["split"]
    with pytest.raises(ConfirmationSetTouched):
        two_parameter_diagnosis(
            select_best_family(catalogue["families"]),
            catalogue["withheld"],
            catalogue["rows"],
            catalogue["prepared"],
            catalogue["cache"],
            split,
            REFERENCE_GRID_POINT["a0"],
            REFERENCE_GRID_POINT["length_unit"],
            [("1", Fraction(1))],
        )


def test_the_receipt_records_the_split_before_any_population(receipt) -> None:
    split = receipt["exploration_confirmation_split"]
    assert split["sealed_before_any_fit"] is True
    assert split["salt"] == SPLIT_SALT
    assert split["rule"] == SPLIT_RULE
    body = {key: value for key, value in split.items() if key != "split_sha256"}
    assert split["split_sha256"] == canonical_sha256(body)
    withheld = set(split["confirmation"])
    assert withheld
    for decomposition in receipt["per_object_decomposition_r1"].values():
        assert not withheld & set(decomposition["objects_fitted"])
        assert not withheld & {entry["object"] for entry in decomposition["population"]}


def test_a_population_reported_without_its_split_is_refused(receipt) -> None:
    """The other half of R1's falsifier: no split block, no receipt."""

    tampered = copy.deepcopy(receipt)
    del tampered["exploration_confirmation_split"]
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(PerObjectError):
        validate_receipt(tampered, root=ROOT)


def test_a_receipt_that_fitted_a_confirmation_object_is_refused(receipt) -> None:
    tampered = copy.deepcopy(receipt)
    withheld = tampered["exploration_confirmation_split"]["confirmation"][0]
    tampered["per_object_decomposition_r1"]["newtonian_baryons_only"]["objects_fitted"].append(
        withheld
    )
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(PerObjectError):
        validate_receipt(tampered, root=ROOT)


def test_a_resealed_split_with_changed_membership_is_refused(receipt) -> None:
    tampered = copy.deepcopy(receipt)
    split = tampered["exploration_confirmation_split"]
    split["confirmation"] = []
    split["exploration"] = sorted(split["exploration"] + receipt["exploration_confirmation_split"]["confirmation"])
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(PerObjectError):
        validate_receipt(tampered, root=ROOT)


# ---------------------------------------------------------------------------
# The exact per-object arithmetic
# ---------------------------------------------------------------------------


def test_the_critical_coverage_is_the_exact_point_where_the_interval_closes() -> None:
    """Two rows, worked by hand: alpha = 1 and 2, beta = 1/2 each, so k* = 1."""

    rows = [_row("low", "1", "0.5"), _row("high", "2", "0.5")]
    axis = build_axis([Fraction(0), Fraction(0)], [Fraction(1), Fraction(1)], rows)
    assert axis.alphas == (Fraction(1), Fraction(2))
    assert axis.betas == (Fraction(1, 2), Fraction(1, 2))
    critical, pair = critical_coverage(axis)
    assert critical == Fraction(1)
    assert pair == ("high", "low")
    at = interval_at(axis, critical)
    assert at["empty"] is False
    assert at["pinned_to_a_point"] is True
    assert _frac(at["lower"]) == Fraction(3, 2)
    below = interval_at(axis, Fraction(999, 1000))
    assert below["empty"] is True
    assert below["certificate"]["kind"] == "two_row_contradiction"
    above = interval_at(axis, Fraction(2))
    assert above["empty"] is False
    assert _frac(above["lower"]) == Fraction(1)
    assert _frac(above["upper"]) == Fraction(2)


def test_a_row_the_parameter_cannot_move_still_bounds_the_coverage() -> None:
    rows = [_row("free", "1", "1"), _row("blind", "10", "2")]
    axis = build_axis([Fraction(0), Fraction(0)], [Fraction(1), Fraction(0)], rows)
    assert axis.blind_floor == Fraction(5)
    assert axis.blind_label == "blind"
    critical, _pair = critical_coverage(axis)
    assert critical == Fraction(5)
    assert interval_at(axis, Fraction(4))["empty"] is True
    assert interval_at(axis, Fraction(5))["empty"] is False


def test_an_axis_with_no_movable_row_is_refused() -> None:
    with pytest.raises(PerObjectError):
        build_axis([Fraction(0)], [Fraction(0)], [_row("blind", "1", "1")])


def test_a_non_positive_declared_sigma_is_refused() -> None:
    row = _row("free", "1", "1")
    broken = type(row)(
        label=row.label,
        point=row.point,
        point_sigma=row.point_sigma,
        point_sigma_rule=row.point_sigma_rule,
        value=row.value,
        value_sigma=Fraction(0),
        value_sigma_rule=row.value_sigma_rule,
        source=row.source,
        point_citation=row.point_citation,
        value_citation=row.value_citation,
        point_declared=row.point_declared,
        value_declared=row.value_declared,
    )
    with pytest.raises(PerObjectError):
        build_axis([Fraction(0)], [Fraction(1)], [broken])


def test_every_published_object_closes_exactly_at_its_reported_critical_coverage(
    receipt, catalogue
) -> None:
    """Recompute each object's axis and confirm the receipt's k* is the exact minimum."""

    checked = 0
    for name in ("newtonian_baryons_only", "deliberately_wrong_law"):
        law = catalogue["spaces"][name]
        for entry in receipt["per_object_decomposition_r1"][name]["population"]:
            galaxy = next(
                item for item in catalogue["exploration"] if item.name == entry["object"]
            )
            offsets, slopes = law.columns(galaxy)
            axis = build_axis(offsets, slopes, catalogue["rows"][galaxy.name])
            critical, _pair = critical_coverage(axis)
            assert critical == _frac(entry["critical_coverage"])
            assert interval_at(axis, critical)["empty"] is False
            assert interval_at(axis, critical * Fraction(999999, 1000000))["empty"] is True
            checked += 1
    assert checked == 2 * len(catalogue["exploration"])


def test_the_closed_form_and_the_audited_simplex_never_disagree(receipt) -> None:
    crosschecks = receipt["instrument_crosscheck"]
    assert set(crosschecks) >= {"newtonian_baryons_only", "deliberately_wrong_law"}
    for name, block in crosschecks.items():
        assert block["disagreements"] == 0, name
        assert block["agreements"] == receipt["counts"]["exploration_objects"], name
        assert len(block["checks"]) == block["agreements"]
        for check in block["checks"]:
            assert check["at_critical_coverage"] == FEASIBLE
            assert check["below_critical_coverage"] == INFEASIBLE
            assert check["farkas_term_count_below"] >= 2


def test_no_float_reaches_the_receipt(receipt) -> None:
    """I3, locally: every number on a certificate path is an int or a decimal string."""

    def walk(value, path="$"):
        assert not isinstance(value, float), f"float at {path}"
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(receipt)


def test_every_exact_block_is_a_pair_of_integers(receipt) -> None:
    seen = 0

    def walk(value):
        nonlocal seen
        if isinstance(value, dict):
            if set(value) == {"denominator", "numerator"}:
                assert isinstance(value["numerator"], int)
                assert isinstance(value["denominator"], int)
                assert value["denominator"] > 0
                seen += 1
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(receipt)
    assert seen > 100


# ---------------------------------------------------------------------------
# R2: the adjudicator, and both directions of its falsifier
# ---------------------------------------------------------------------------


def test_a_shared_value_is_adjudicated_constant_and_the_witness_is_verified() -> None:
    intervals = {
        "A": _interval("0", "2"),
        "B": _interval("1", "3"),
        "C": _interval("1/2", "5/2"),
    }
    record = adjudicate(intervals)
    verify_adjudication(record, intervals)
    assert record["verdict"] == CONSTANT
    witness = _frac(record["witness"])
    assert witness == Fraction(1)
    for entry in intervals.values():
        assert _frac(entry["lower"]) <= witness <= _frac(entry["upper"])


def test_disjoint_intervals_are_adjudicated_varies_with_an_exact_gap() -> None:
    intervals = {"A": _interval("0", "1"), "B": _interval("3", "4")}
    record = adjudicate(intervals)
    verify_adjudication(record, intervals)
    assert record["verdict"] == VARIES
    assert record["certificate"]["highest_lower_bound_from"] == "B"
    assert record["certificate"]["lowest_upper_bound_from"] == "A"
    assert _frac(record["certificate"]["gap"]) == Fraction(2)


def test_an_object_with_no_solution_blocks_the_question_rather_than_answering_it() -> None:
    intervals = {"A": _interval("0", "1"), "B": {"empty": True, "reason": "control"}}
    record = adjudicate(intervals)
    verify_adjudication(record, intervals)
    assert record["verdict"] == NO_POPULATION
    assert record["objects_without_any_solution"] == ["B"]


def test_claiming_constant_over_disjoint_intervals_raises() -> None:
    """R2 falsifier, direction one.  This is the control that must fail."""

    intervals = {"A": _interval("0", "1"), "B": _interval("3", "4")}
    forged = {
        "verdict": CONSTANT,
        "witness": {"decimal": "5.0e-01", "exact": _exact("1/2")},
        "witness_verified_in_every_interval": True,
    }
    with pytest.raises(PerObjectError):
        verify_adjudication(forged, intervals)


def test_claiming_varies_when_a_shared_value_exists_raises() -> None:
    """R2 falsifier, direction two.  This is the control that must fail."""

    intervals = {"A": _interval("0", "2"), "B": _interval("1", "3")}
    forged = {
        "verdict": VARIES,
        "certificate": {
            "gap": {"decimal": "1.0e+00", "exact": _exact("1")},
            "highest_lower_bound_from": "B",
            "kind": "disjoint_object_intervals",
            "lowest_upper_bound_from": "A",
        },
    }
    with pytest.raises(PerObjectError):
        verify_adjudication(forged, intervals)


def test_a_varies_certificate_that_names_the_wrong_objects_raises() -> None:
    intervals = {
        "A": _interval("0", "1"),
        "B": _interval("3", "4"),
        "C": _interval("10", "11"),
    }
    honest = adjudicate(intervals)
    verify_adjudication(honest, intervals)
    assert honest["certificate"]["highest_lower_bound_from"] == "C"
    forged = copy.deepcopy(honest)
    forged["certificate"]["highest_lower_bound_from"] = "B"
    forged["certificate"]["gap"] = {"decimal": "2.0e+00", "exact": _exact("2")}
    with pytest.raises(PerObjectError):
        verify_adjudication(forged, intervals)


def test_a_gap_that_does_not_match_its_own_endpoints_raises() -> None:
    intervals = {"A": _interval("0", "1"), "B": _interval("3", "4")}
    forged = adjudicate(intervals)
    forged["certificate"]["gap"] = {"decimal": "9.9e+02", "exact": _exact("990")}
    with pytest.raises(PerObjectError):
        verify_adjudication(forged, intervals)


def test_claiming_no_population_when_every_object_has_one_raises() -> None:
    intervals = {"A": _interval("0", "1"), "B": _interval("0", "1")}
    with pytest.raises(PerObjectError):
        verify_adjudication({"verdict": NO_POPULATION}, intervals)


def test_an_unknown_verdict_is_refused() -> None:
    intervals = {"A": _interval("0", "1")}
    with pytest.raises(PerObjectError):
        verify_adjudication({"verdict": "PROBABLY"}, intervals)


def test_the_price_of_universality_must_agree_with_the_adjudicated_verdict(
    catalogue,
) -> None:
    """Control: two independent derivations of the same fact, forced to agree."""

    law = catalogue["spaces"]["newtonian_baryons_only"]
    decomposition = decompose(law, catalogue["exploration"], catalogue["rows"], catalogue["split"])
    honest = diagnose(decomposition)
    assert honest["verdict_at_population_coverage"] == VARIES
    assert honest["price_of_universality"]["is_one"] is False
    forged = copy.deepcopy(decomposition)
    forged["shared_parameter_coverage"] = forged["smallest_coverage_with_a_population"]
    with pytest.raises(PerObjectError):
        diagnose(forged)


# ---------------------------------------------------------------------------
# R1: the whole population is kept
# ---------------------------------------------------------------------------


def test_every_law_space_keeps_one_solution_per_exploration_object(receipt) -> None:
    expected = sorted(receipt["exploration_confirmation_split"]["exploration"])
    assert len(receipt["per_object_decomposition_r1"]) == receipt["counts"]["law_spaces"]
    for name, decomposition in receipt["per_object_decomposition_r1"].items():
        assert decomposition["population_kept_not_ranked"] is True, name
        assert sorted(entry["object"] for entry in decomposition["population"]) == expected
        assert decomposition["objects_fitted"] == expected
        for entry in decomposition["population"]:
            assert entry["binding_pair"] is not None
            assert entry["at_critical_coverage"]["empty"] is False
            assert entry["interval_at_population_coverage"]["empty"] is False
            assert set(entry["declared_coverage_intervals"]) == set(COVERAGE_GRID)
    assert receipt["counts"]["one_parameter_fits"] == (
        receipt["counts"]["law_spaces"] * receipt["counts"]["exploration_objects"]
    )


def test_the_population_coverage_is_the_worst_object_and_never_below_it(receipt) -> None:
    for name, decomposition in receipt["per_object_decomposition_r1"].items():
        k_pop = _frac(decomposition["smallest_coverage_with_a_population"])
        per_object = {
            entry["object"]: _frac(entry["critical_coverage"])
            for entry in decomposition["population"]
        }
        assert k_pop == max(per_object.values()), name
        assert decomposition["smallest_coverage_with_a_population_set_by"] in per_object
        assert _frac(decomposition["shared_parameter_coverage"]) >= k_pop, name


def test_the_interval_at_the_population_coverage_contains_the_point_estimate(receipt) -> None:
    for name, decomposition in receipt["per_object_decomposition_r1"].items():
        for entry in decomposition["population"]:
            interval = entry["interval_at_population_coverage"]
            theta = _frac(entry["theta_at_critical_coverage"])
            assert _frac(interval["lower"]) <= theta <= _frac(interval["upper"]), (
                name,
                entry["object"],
            )


# ---------------------------------------------------------------------------
# Residual structure
# ---------------------------------------------------------------------------


def test_residual_structure_separates_a_single_bow_from_alternation() -> None:
    bowed = [_row(f"r{i}", "1", "1") for i in range(8)]
    offsets = [Fraction(0)] * 8
    slopes = [Fraction(1)] * 8
    # theta = 0 leaves every residual at +1: one run, no sign changes.
    structured = residual_structure(offsets, slopes, bowed, Fraction(0))
    assert structured["sign_changes"] == 0
    assert structured["longest_same_sign_run"] == 8
    assert structured["structured"] is True
    # Control: alternating values must NOT be called structured.
    alternating = [_row(f"a{i}", "1" if i % 2 else "-1", "1") for i in range(8)]
    noisy = residual_structure(offsets, slopes, alternating, Fraction(0))
    assert noisy["sign_changes"] == 7
    assert noisy["structured"] is False
    assert noisy["sign_changes"] * STRUCTURE_RUN_DIVISOR > len(alternating) - 1


def test_every_published_residual_pattern_is_reported_with_integer_counts(receipt) -> None:
    for name, block in receipt["parameter_variation_diagnosis_r2"].items():
        for galaxy, structure in block["residual_structure"].items():
            assert isinstance(structure["sign_changes"], int), (name, galaxy)
            assert isinstance(structure["longest_same_sign_run"], int)
            assert 1 <= structure["longest_same_sign_run"] <= structure["points"]
            assert structure["structured"] is (
                structure["sign_changes"] * STRUCTURE_RUN_DIVISOR <= structure["points"] - 1
            )


# ---------------------------------------------------------------------------
# The deliverable: the contrast the pooled INFEASIBLE could not make
# ---------------------------------------------------------------------------


def test_the_newtonian_per_object_rescale_is_not_constant(receipt) -> None:
    block = receipt["parameter_variation_diagnosis_r2"]["newtonian_baryons_only"]
    assert block["verdict_at_population_coverage"] == VARIES
    at_population = block["at_population_coverage"]
    assert at_population["verdict"] == VARIES
    # Re-derive the certificate by hand from the receipt's own intervals.
    intervals = at_population["intervals"]
    certificate = at_population["certificate"]
    low = _frac(intervals[certificate["highest_lower_bound_from"]]["lower"])
    high = _frac(intervals[certificate["lowest_upper_bound_from"]]["upper"])
    assert low > high
    assert _frac(certificate["gap"]) == low - high
    assert _frac(block["price_of_universality"]) > 1
    # And every object individually does have a solution there, so this is a real
    # disagreement between objects and not one object having no answer at all.
    assert all(entry["empty"] is False for entry in intervals.values())


def test_the_screened_family_amplitude_is_constant_with_an_exhibited_witness(receipt) -> None:
    name = receipt["contrast"]["screened_family"]["law"]
    block = receipt["parameter_variation_diagnosis_r2"][name]
    assert block["verdict_at_population_coverage"] == CONSTANT
    at_population = block["at_population_coverage"]
    witness = _frac(at_population["witness"])
    for galaxy, interval in at_population["intervals"].items():
        assert interval["empty"] is False, galaxy
        assert _frac(interval["lower"]) <= witness <= _frac(interval["upper"]), galaxy
    assert _frac(block["price_of_universality"]) == 1


def test_every_screened_family_prices_universality_at_one_and_newtonian_does_not(
    receipt,
) -> None:
    diagnoses = receipt["parameter_variation_diagnosis_r2"]
    families = [name for name in diagnoses if name.startswith("family_")]
    assert len(families) == 12
    for name in families:
        assert _frac(diagnoses[name]["price_of_universality"]) == 1, name
        assert diagnoses[name]["verdict_at_population_coverage"] == CONSTANT, name
    assert _frac(diagnoses["newtonian_baryons_only"]["price_of_universality"]) > 1
    assert diagnoses["newtonian_baryons_only"]["verdict_at_population_coverage"] == VARIES


def test_the_wrong_law_control_needs_a_larger_coverage_than_every_real_family(
    receipt,
) -> None:
    """The control that must fail: if it ever fits better, nothing here measures anything."""

    decompositions = receipt["per_object_decomposition_r1"]
    control = _frac(decompositions["deliberately_wrong_law"]["smallest_coverage_with_a_population"])
    for name, decomposition in decompositions.items():
        if name == "deliberately_wrong_law":
            continue
        assert (
            _frac(decomposition["smallest_coverage_with_a_population"]) < control
        ), name


def test_a_constant_parameter_is_not_a_measure_of_a_good_law(receipt) -> None:
    """The anti-scoreboard control: the wrong law is CONSTANT too, and is still wrong."""

    control = receipt["parameter_variation_diagnosis_r2"]["deliberately_wrong_law"]
    assert control["verdict_at_population_coverage"] == CONSTANT
    assert _frac(control["price_of_universality"]) == 1
    ladder = receipt["contrast"]["control_law"]["objects_with_a_solution_by_declared_coverage"]
    assert set(ladder.values()) == {0}


def test_the_declared_coverage_ladder_separates_the_two_law_spaces(receipt) -> None:
    contrast = receipt["contrast"]
    newtonian = contrast["newtonian_baryons"]["objects_with_a_solution_by_declared_coverage"]
    screened = contrast["screened_family"]["objects_with_a_solution_by_declared_coverage"]
    assert set(newtonian) == set(COVERAGE_GRID)
    assert all(newtonian[key] <= screened[key] for key in COVERAGE_GRID)
    assert any(newtonian[key] < screened[key] for key in COVERAGE_GRID)


def test_the_decision_names_both_verdicts_and_refuses_to_confirm(receipt) -> None:
    decision = receipt["decision"]
    assert decision.startswith("EXPLORATORY")
    assert VARIES in decision and CONSTANT in decision
    assert "Nothing here may be cited as a confirmation." in decision
    assert receipt["exploratory_caveat"]["may_be_cited_as_confirmation"] is False
    assert receipt["exploratory_caveat"]["sealed_no_refit_trial"] is False
    assert receipt["exploratory_caveat"]["confirmation_set_is_not_virgin_data"]
    assert receipt["trial_type"] == TRIAL_TYPE == "exploratory"


# ---------------------------------------------------------------------------
# The two-parameter law space
# ---------------------------------------------------------------------------


def test_the_two_parameter_projections_are_exact_and_answer_per_parameter(receipt) -> None:
    block = receipt["two_parameter_diagnosis_r2"]
    assert block["parameters"] == ["w_yukawa", "w_power"]
    widened = block["at_coverage_factors"]["one_parameter_population_coverage"]
    for galaxy, box in widened["coordinate_projections"].items():
        assert box["empty"] is False, galaxy
        for name in block["parameters"]:
            bounds = box["intervals"][name]
            assert bounds["lower_status"] == "OPTIMAL"
            assert bounds["upper_status"] == "OPTIMAL"
            assert _frac(bounds["lower"]) <= _frac(bounds["upper"])
    for name, record in widened["per_parameter"].items():
        assert record["verdict"] == CONSTANT, name
        assert record["objects_with_a_solution"] == record["objects_offered"]


def test_the_one_parameter_solution_lies_inside_every_two_parameter_box(receipt) -> None:
    """The containment that justifies reusing the one-parameter coverage as the widening."""

    block = receipt["two_parameter_diagnosis_r2"]
    widened = block["at_coverage_factors"]["one_parameter_population_coverage"]
    name = receipt["contrast"]["screened_family"]["law"]
    shared = _frac(
        receipt["parameter_variation_diagnosis_r2"][name]["at_population_coverage"]["witness"]
    )
    ordinal = block["family_ordinal"]
    family = next(item for item in load_families(ROOT) if item.ordinal == ordinal)
    declared = {
        "w_power": Fraction(family.parameters["w_power"]),
        "w_yukawa": Fraction(family.parameters["w_yukawa"]),
    }
    for galaxy, box in widened["coordinate_projections"].items():
        for parameter, weight in declared.items():
            bounds = box["intervals"][parameter]
            assert _frac(bounds["lower"]) <= shared * weight <= _frac(bounds["upper"]), (
                galaxy,
                parameter,
            )


def test_per_parameter_constancy_is_reported_separately_from_the_joint_question(
    receipt,
) -> None:
    block = receipt["two_parameter_diagnosis_r2"]
    assert "projections of a set do not determine the set" in block["caveat"]
    for label, entry in block["at_coverage_factors"].items():
        assert entry["joint_shared_pair"]["verdict"] in {FEASIBLE, INFEASIBLE}, label
        assert isinstance(entry["per_parameter_constant_but_no_joint_pair"], bool)
        if entry["per_parameter_constant_but_no_joint_pair"]:
            assert entry["joint_shared_pair"]["verdict"] == INFEASIBLE


def test_the_families_own_arm_weights_are_checked_against_every_object(receipt) -> None:
    block = receipt["two_parameter_diagnosis_r2"]
    widened = block["at_coverage_factors"]["one_parameter_population_coverage"]
    declared = {
        name: record["declared_family_value"] for name, record in widened["per_parameter"].items()
    }
    assert set(declared) == {"w_yukawa", "w_power"}
    for name, entry in declared.items():
        inside = entry["inside_every_object_interval"]
        value = Fraction(int(entry["exact"]["numerator"]), int(entry["exact"]["denominator"]))
        recomputed = all(
            Fraction(
                int(box["intervals"][name]["lower"]["exact"]["numerator"]),
                int(box["intervals"][name]["lower"]["exact"]["denominator"]),
            )
            <= value
            <= Fraction(
                int(box["intervals"][name]["upper"]["exact"]["numerator"]),
                int(box["intervals"][name]["upper"]["exact"]["denominator"]),
            )
            for box in widened["coordinate_projections"].values()
        )
        assert inside is recomputed, name


# ---------------------------------------------------------------------------
# Anti-divergence: the law spaces must be the predecessor module's own columns
# ---------------------------------------------------------------------------


def test_the_family_law_space_is_the_predecessor_modules_design(catalogue) -> None:
    family = select_best_family(catalogue["families"])
    law = catalogue["spaces"][f"family_{family.ordinal}_amplitude"]
    weight_yukawa = Fraction(family.parameters["w_yukawa"])
    weight_power = Fraction(family.parameters["w_power"])
    tolerance = Fraction(1, 10**12)
    for galaxy in catalogue["exploration"]:
        entry = catalogue["prepared"][galaxy.name]
        free = _family_columns(
            galaxy,
            family,
            REFERENCE_GRID_POINT["a0"],
            REFERENCE_GRID_POINT["length_unit"],
            catalogue["cache"],
            entry["v_bar_squared"],
            entry["radius"],
            free_arms=True,
        )
        pinned = _family_columns(
            galaxy,
            family,
            REFERENCE_GRID_POINT["a0"],
            REFERENCE_GRID_POINT["length_unit"],
            catalogue["cache"],
            entry["v_bar_squared"],
            entry["radius"],
            free_arms=False,
        )
        offsets, slopes = law.columns(galaxy)
        for index, row in enumerate(free):
            assert offsets[index] == row[0]
            assert slopes[index] == weight_yukawa * row[1] + weight_power * row[2]
            # theta = 1 must reproduce the predecessor's zero-freedom column to within the
            # declared 15-digit freeze; a larger gap means the two paths have diverged.
            mine = offsets[index] + slopes[index]
            theirs = pinned[index][0]
            assert abs(mine - theirs) <= tolerance * abs(theirs)


def test_the_newtonian_law_space_is_the_published_baryonic_column(catalogue) -> None:
    law = catalogue["spaces"]["newtonian_baryons_only"]
    for galaxy in catalogue["exploration"]:
        offsets, slopes = law.columns(galaxy)
        assert all(value == 0 for value in offsets)
        assert tuple(slopes) == catalogue["prepared"][galaxy.name]["v_bar_squared_exact"]


# ---------------------------------------------------------------------------
# Receipt determinism, seal, and tamper
# ---------------------------------------------------------------------------


def test_the_receipt_is_deterministic_and_sealed(receipt) -> None:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)
    assert receipt["schema_version"] == RESULT_SCHEMA
    assert build_receipt(ROOT) == receipt


def test_the_written_receipt_matches_the_build(receipt, sealed) -> None:
    assert sealed == receipt
    validate_receipt(sealed, root=ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        # All but the first are caught by the cheap guards before the exact replay runs.
        lambda value: value.update({"decision": "PASS"}),
        lambda value: value.update({"trial_type": "confirmatory"}),
        lambda value: value["claims"].update({"confirmation_set_fitted": True}),
        lambda value: value["claims"].update({"split_sealed_before_fitting": False}),
        lambda value: value.update({"schema_version": "invariant-something-else-1.0"}),
        lambda value: value["exploration_confirmation_split"].update({"salt": "other"}),
    ],
)
def test_receipt_tamper_fails_closed(receipt, mutate) -> None:
    tampered = copy.deepcopy(receipt)
    mutate(tampered)
    with pytest.raises(PerObjectError):
        validate_receipt(tampered, root=ROOT)
    resealed = {key: value for key, value in tampered.items() if key != "content_sha256"}
    resealed["content_sha256"] = canonical_sha256(resealed)
    with pytest.raises(PerObjectError):
        validate_receipt(resealed, root=ROOT)


def test_the_receipt_carries_no_scalar_goodness_key(receipt) -> None:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert forbidden_receipt_keys(body) == []


def test_the_claims_block_is_pinned(receipt) -> None:
    assert receipt["claims"] == R1_CLAIMS
    assert receipt["claims"]["confirmation_set_fitted"] is False
    assert receipt["claims"]["split_sealed_before_fitting"] is True
    assert receipt["claims"]["population_kept_not_ranked"] is True


def test_the_scope_declares_r3_out_of_bounds_and_names_no_channel(receipt) -> None:
    """R3's falsifier is naming a channel before it is measured.  Nothing here names one."""

    assert "R3" in receipt["scope"]
    assert "out of scope" in receipt["scope"]
    text = json.dumps(receipt).lower()
    for leaked in ("surface_brightness", "gas_fraction", "stellar_mass_channel", "channel_value"):
        assert leaked not in text


def test_the_axis_dataclass_is_frozen() -> None:
    axis = Axis(
        alphas=(Fraction(1),),
        betas=(Fraction(1),),
        labels=("a",),
        blind_floor=Fraction(0),
        blind_label=None,
    )
    with pytest.raises((AttributeError, TypeError, ValueError)):
        axis.alphas = ()  # type: ignore[misc]
