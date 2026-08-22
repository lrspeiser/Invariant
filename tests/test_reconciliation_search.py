"""Gates for the Tier 7 reconciliation search (R4).

R4's claim is narrow and its falsifier is sharp: **one law, drawn from a declared space,
reproduces the whole per-object population, and no per-object parameter survives**.  The
falsifier is a reconciliation that quietly keeps a per-object knob, or a claimed
reconciliation whose residuals were never checked against R2's intervals.  These tests pin
both directions.

*A population built from a universal law is reconciled back to that law.*  Not "something
fits" -- the search returns the exact generating form ``alpha=one+x|beta=one+y``, the
generating coefficient vector is shown to violate no declared row, and it lies inside the
exact coefficient box the search reports.  The test recomputes every per-object prediction
itself, in ``Fraction`` arithmetic, and requires each one inside its own R2 interval.

*A genuinely per-object population returns the obstruction, not a fake reconciliation.*  Two
controls, and the first does not depend on the search at all: twin objects with identical
covariates and disjoint intervals cannot be reconciled by *any* function of covariates, and
that argument is checked directly before the linear program is consulted.  The second is a
scrambled population whose offsets are tracked by nothing in the catalogue.

*The Farkas certificates are re-derived here, not trusted.*  For every transcribed
certificate the test rebuilds ``lambda^T A`` and ``lambda^T b`` from the receipt's own
multipliers using nothing but ``Fraction``, and requires ``lambda >= 0``,
``lambda^T A = 0`` and ``lambda^T b < 0``.  Six mutations of an honest certificate are
required to fail.

*The break-even coverage is a prediction, and it is tested as one.*  The obstruction reports
the exact factor below which no widening of R2's intervals can help.  The test widens every
interval by ``999/1000`` of that factor and requires the obstruction to survive, then widens
by twenty times it and requires a reconciliation -- so the number is neither decoration nor
a search that always says no.  For the twin population the factor comes out at exactly ``2``,
which is what the geometry says it must be: intervals of halfwidth ``h`` separated by ``2h``
touch when the halfwidths double.

*The per-object-freedom gate is load-bearing.*  The smuggled-knob population hands the search
a covariate that is an indicator of one object.  The linear program calls three such forms
FEASIBLE; the gate refuses all three, and the test asserts both halves -- the refusal, and
the fact that there was something real to refuse.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.real_data_gravity_confrontation import universal_parameter_width
from sigma_theory_compiler.reconciliation_search import (
    CONSTANT_CHANNEL,
    DECLARED_COVERAGE,
    OBSTRUCTED,
    RECEIPT_PATH,
    RECONCILED,
    REFUSAL_PER_OBJECT_FREEDOM,
    RESULT_SCHEMA,
    SEARCH_CAPS,
    TRUE_FORM,
    Channel,
    Form,
    LawSpace,
    Population,
    PopulationObject,
    ReconciliationError,
    build_design,
    build_receipt,
    certificate_break_even,
    channel_catalogue,
    decide_form,
    default_space,
    freedom_report,
    identical_covariate_obstruction,
    main,
    recheck_farkas,
    reconcile,
    reconciliation_controls,
    regenerate_population,
    satisfies_every_row,
    scale_population_intervals,
    synthetic_population,
    tagged_space,
    validate_population,
    verify_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.tolerance_aware_fitting import FEASIBLE, INFEASIBLE

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260819


def _q(block: Any) -> Fraction:
    return Fraction(int(block["numerator"]), int(block["denominator"]))


# ---------------------------------------------------------------------------
# Fixtures: the four declared scenarios, computed once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def universal() -> Population:
    return synthetic_population(kind="universal", object_count=8, seed=SEED)


@pytest.fixture(scope="module")
def twin() -> Population:
    return synthetic_population(kind="twin", object_count=6, seed=SEED)


@pytest.fixture(scope="module")
def scrambled() -> Population:
    return synthetic_population(kind="scrambled", object_count=8, seed=SEED)


@pytest.fixture(scope="module")
def smuggled() -> Population:
    return synthetic_population(kind="twin", object_count=6, seed=SEED, tagged=True)


@pytest.fixture(scope="module")
def universal_result(universal: Population) -> dict:
    return reconcile(universal, default_space(universal.covariate_names))


@pytest.fixture(scope="module")
def twin_result(twin: Population) -> dict:
    return reconcile(twin, default_space(twin.covariate_names))


@pytest.fixture(scope="module")
def scrambled_result(scrambled: Population) -> dict:
    return reconcile(scrambled, default_space(scrambled.covariate_names))


@pytest.fixture(scope="module")
def smuggled_result(smuggled: Population) -> dict:
    return reconcile(smuggled, tagged_space())


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt()


# ---------------------------------------------------------------------------
# The declared law space is counted, not sampled
# ---------------------------------------------------------------------------


def test_law_space_enumeration_matches_its_closed_form_count() -> None:
    space = default_space(("x", "y"))
    coverage = space.coverage_certificate()
    # constant plus {x, x^2, y, y^2}: per parameter 1 + 4 + C(4,2) = 11, two parameters
    assert coverage["options_per_parameter"] == 11
    assert coverage["declared_cardinality"] == 121
    assert coverage["traversed_cardinality"] == 121
    assert coverage["distinct_traversed"] == 121
    assert coverage["enumeration_equals_declared"] is True


def test_forms_are_occam_ordered_and_every_one_carries_the_constant() -> None:
    forms = default_space(("x", "y")).forms()
    extras = [form.extra_count for form in forms]
    assert extras == sorted(extras)
    assert forms[0].extra_count == 0
    for form in forms:
        for _, names in form.assignment:
            assert names[0] == CONSTANT_CHANNEL
        assert form.width == sum(len(names) for _, names in form.assignment)


def test_law_space_declaration_refuses_malformed_catalogues() -> None:
    with pytest.raises(ReconciliationError):
        channel_catalogue(("x", "x"), (1,))
    with pytest.raises(ReconciliationError):
        channel_catalogue(("x",), (0,))
    with pytest.raises(ReconciliationError):
        channel_catalogue((), (1,))
    with pytest.raises(ReconciliationError):
        LawSpace(
            name="budget-too-big",
            covariate_names=("x",),
            parameter_names=("alpha",),
            channels=channel_catalogue(("x",), (1,)),
            max_extra_channels=SEARCH_CAPS["max_extra_channels_per_parameter"] + 1,
        )


def test_channels_are_exact_and_refuse_undefined_points() -> None:
    channel = Channel("x^m1", (("x", -1),))
    assert channel.evaluate({"x": Fraction(4)}) == Fraction(1, 4)
    with pytest.raises(ReconciliationError):
        channel.evaluate({"x": Fraction(0)})
    with pytest.raises(ReconciliationError):
        channel.evaluate({"y": Fraction(1)})
    assert Channel(CONSTANT_CHANNEL, ()).evaluate({}) == Fraction(1)


# ---------------------------------------------------------------------------
# The Tier 7 split is not optional
# ---------------------------------------------------------------------------


def _mutated(population: Population, **fields: Any) -> Population:
    base = {
        "name": population.name,
        "covariate_names": population.covariate_names,
        "parameter_names": population.parameter_names,
        "objects": population.objects,
        "confirmation": population.confirmation,
        "provenance": population.provenance,
        "generator": population.generator,
    }
    return Population(**{**base, **fields})


def test_population_without_a_confirmation_set_is_refused(universal: Population) -> None:
    with pytest.raises(ReconciliationError, match="Tier 7 split is missing"):
        reconcile(_mutated(universal, confirmation=()), default_space(universal.covariate_names))


def test_population_intruding_on_the_confirmation_set_is_refused(universal: Population) -> None:
    broken = _mutated(universal, confirmation=(universal.labels()[2], "HOLD-000"))
    with pytest.raises(ReconciliationError, match="split violated"):
        reconcile(broken, default_space(universal.covariate_names))


def test_population_validation_fails_closed(universal: Population) -> None:
    with pytest.raises(ReconciliationError):
        validate_population(_mutated(universal, objects=()))
    with pytest.raises(ReconciliationError):
        validate_population(_mutated(universal, objects=universal.objects[:1] * 2))
    with pytest.raises(ReconciliationError):
        validate_population(_mutated(universal, provenance="   "))
    first = universal.objects[0]
    inverted = PopulationObject(
        label=first.label,
        covariates=first.covariates,
        intervals=tuple(
            (name, upper, lower) for name, lower, upper in first.intervals
        ),
    )
    with pytest.raises(ReconciliationError, match="inverted"):
        validate_population(_mutated(universal, objects=(inverted, *universal.objects[1:])))
    short = PopulationObject(
        label=first.label, covariates=first.covariates[:1], intervals=first.intervals
    )
    with pytest.raises(ReconciliationError, match="covariate set"):
        validate_population(_mutated(universal, objects=(short, *universal.objects[1:])))


# ---------------------------------------------------------------------------
# Zero per-object freedom, structurally
# ---------------------------------------------------------------------------


def test_universal_width_does_not_grow_with_the_population(universal: Population) -> None:
    space = default_space(universal.covariate_names)
    form = TRUE_FORM
    widths = set()
    for count in (2, 4, 6, 8):
        trimmed = _mutated(universal, objects=universal.objects[:count])
        design = build_design(trimmed, space, form)
        widths.add(universal_parameter_width(design))
        assert len(design.rows) == count * len(universal.parameter_names)
        assert len(design.columns) == len(design.rows)
    assert widths == {form.width} == {4}


def test_design_rows_reproduce_the_declared_r2_intervals(universal: Population) -> None:
    space = default_space(universal.covariate_names)
    design = build_design(universal, space, TRUE_FORM)
    declared = {
        f"{item.label}::{name}": (lower, upper)
        for item in universal.objects
        for name, lower, upper in item.intervals
    }
    assert len(declared) == len(design.rows)
    for row in design.rows:
        assert row.value_interval(DECLARED_COVERAGE) == declared[row.label]


def test_freedom_report_is_clean_on_a_universal_form(universal: Population) -> None:
    report = freedom_report(universal, default_space(universal.covariate_names), TRUE_FORM)
    assert report["refusals"] == []
    assert report["free_parameters_per_object"] == 0
    assert report["width_is_population_independent"] is True
    assert report["universal_width"] == 4
    assert {entry["support_size"] for entry in report["column_support"]} == {8}


def test_the_freedom_gate_refuses_a_smuggled_per_object_knob(smuggled: Population) -> None:
    space = tagged_space()
    knob = Form((("alpha", (CONSTANT_CHANNEL, "tag", "x")), ("beta", (CONSTANT_CHANNEL, "y"))))
    report = freedom_report(smuggled, space, knob)
    assert REFUSAL_PER_OBJECT_FREEDOM in report["refusals"]
    assert report["free_parameters_per_object"] is None
    tag_column = next(
        entry for entry in report["column_support"] if entry["coefficient"] == "alpha:tag"
    )
    assert tag_column["support_size"] == 1
    # The gate is load-bearing only if the program would otherwise have accepted this form.
    decision = decide_form(smuggled, space, knob)
    assert decision["status"] == "REFUSED"
    assert decision["linear_program_verdict"] == FEASIBLE


# ---------------------------------------------------------------------------
# Positive control: a universal population is reconciled back to its own law
# ---------------------------------------------------------------------------


def test_universal_population_recovers_the_generating_form(universal_result: dict) -> None:
    assert universal_result["verdict"] == RECONCILED
    assert universal_result["reconciliation"]["form"]["id"] == TRUE_FORM.identifier()
    assert universal_result["reconciliation"]["free_parameters_per_object"] == 0
    assert universal_result["reconciliation"]["universal_width"] == 4
    assert universal_result["obstruction"] is None


def test_the_generating_law_itself_violates_no_declared_row(universal: Population) -> None:
    space = default_space(universal.covariate_names)
    design = build_design(universal, space, TRUE_FORM)
    truth = [Fraction(3), Fraction(2), Fraction(5), Fraction(-4)]
    assert list(design.parameter_names) == ["alpha:one", "alpha:x", "beta:one", "beta:y"]
    assert satisfies_every_row(design, truth) == []


def test_the_generating_law_lies_inside_the_reported_coefficient_box(
    universal_result: dict,
) -> None:
    box = universal_result["reconciliation"]["coefficient_box"]
    truth = {"alpha:one": Fraction(3), "alpha:x": Fraction(2), "beta:one": Fraction(5),
             "beta:y": Fraction(-4)}
    assert [entry["coefficient"] for entry in box] == list(truth)
    for entry in box:
        assert entry["unbounded"] is False
        assert _q(entry["lower"]) <= truth[entry["coefficient"]] <= _q(entry["upper"])


def test_every_reported_prediction_is_recomputed_and_lands_inside_its_interval(
    universal: Population, universal_result: dict
) -> None:
    space = default_space(universal.covariate_names)
    design = build_design(universal, space, TRUE_FORM)
    point = [_q(entry["value"]) for entry in universal_result["reconciliation"]["witness"]]
    reported = universal_result["reconciliation"]["per_object_predictions"]
    assert len(reported) == len(design.rows) == 16
    for index, row in enumerate(reported):
        predicted = sum(
            (cell * value for cell, value in zip(design.columns[index], point, strict=True)),
            Fraction(0),
        )
        lower, upper = design.rows[index].value_interval(DECLARED_COVERAGE)
        assert predicted == _q(row["predicted"])
        assert lower <= predicted <= upper
        assert (_q(row["interval"]["lower"]), _q(row["interval"]["upper"])) == (lower, upper)
        assert row["inside"] is True


def test_no_smaller_form_than_the_generating_one_is_feasible(universal: Population) -> None:
    space = default_space(universal.covariate_names)
    feasible_below = [
        form
        for form in space.forms()
        if form.extra_count < TRUE_FORM.extra_count
        and decide_form(universal, space, form)["status"] == FEASIBLE
    ]
    assert feasible_below == []


# ---------------------------------------------------------------------------
# Negative control: per-object populations return an exact obstruction
# ---------------------------------------------------------------------------


def test_twin_objects_are_irreconcilable_without_consulting_the_search(twin: Population) -> None:
    argument = identical_covariate_obstruction(twin)
    assert argument is not None
    assert argument["objects"] == ["OBJ-000", "OBJ-001"]
    assert argument["parameter"] == "alpha"
    assert argument["holds_for_every_law_space_on_these_covariates"] is True
    left, right = twin.objects[0], twin.objects[1]
    assert left.covariate_map() == right.covariate_map()
    high_left = left.interval("alpha")[1]
    low_right = right.interval("alpha")[0]
    assert high_left < low_right
    assert _q(argument["gap"]["exact"]) == low_right - high_left > 0


def test_twin_population_is_obstructed_on_every_declared_form(twin_result: dict) -> None:
    assert twin_result["verdict"] == OBSTRUCTED
    assert twin_result["reconciliation"] is None
    assert twin_result["reconciled_form_count"] == 0
    obstruction = twin_result["obstruction"]
    assert obstruction["infeasible_form_count"] == obstruction["declared_form_count"] == 121
    assert obstruction["universally_irreconcilable_objects"] == ["OBJ-000", "OBJ-001"]
    assert obstruction["identical_covariate_argument"] is not None


def test_scrambled_population_is_obstructed_and_tracks_nothing(scrambled_result: dict) -> None:
    assert scrambled_result["verdict"] == OBSTRUCTED
    assert scrambled_result["reconciled_form_count"] == 0
    obstruction = scrambled_result["obstruction"]
    assert obstruction["infeasible_form_count"] == 121
    assert obstruction["identical_covariate_argument"] is None
    assert len(obstruction["universally_irreconcilable_objects"]) == 8


def test_smuggled_population_is_obstructed_because_the_gate_refused_the_knob(
    smuggled_result: dict,
) -> None:
    assert smuggled_result["verdict"] == OBSTRUCTED
    refused = [row for row in smuggled_result["forms"] if row["status"] == "REFUSED"]
    assert len(refused) == 33
    would_have_fitted = [
        row for row in refused if row.get("linear_program_verdict") == FEASIBLE
    ]
    assert len(would_have_fitted) == 3, "the gate must have had something real to refuse"
    for row in would_have_fitted:
        assert REFUSAL_PER_OBJECT_FREEDOM in row["refusals"]
    # Every one of them hands alpha the indicator, which is exactly where the twin
    # difference sits -- and hands it `x` too, because the other four objects still need the
    # real dependence.  The knob is not fitting the population; it is fitting one object.
    assert {row["form"]["id"] for row in would_have_fitted} == {
        "alpha=one+tag+x|beta=one+tag+y",
        "alpha=one+tag+x|beta=one+x+y",
        "alpha=one+tag+x|beta=one+y",
    }
    for row in would_have_fitted:
        channels = {parameter: tuple(names) for parameter, names in row["form"]["assignment"]}
        assert "tag" in channels["alpha"]


# ---------------------------------------------------------------------------
# The Farkas certificates, re-derived here from nothing but Fraction
# ---------------------------------------------------------------------------


def _hand_check(design: Any, terms: list[dict]) -> tuple[bool, list[Fraction], Fraction]:
    """Rebuild lambda^T A and lambda^T b without using the module's own recheck."""

    width = len(design.parameter_names)
    combination = [Fraction(0)] * width
    total = Fraction(0)
    nonnegative = True
    index_of = {row.label: index for index, row in enumerate(design.rows)}
    for term in terms:
        multiplier = _q(term["multiplier"])
        if multiplier < 0:
            nonnegative = False
        index = index_of[term["row"]]
        sign = Fraction(1) if term["bound"] == "upper" else Fraction(-1)
        lower, upper = design.rows[index].value_interval(DECLARED_COVERAGE)
        bound = upper if term["bound"] == "upper" else -lower
        for column in range(width):
            combination[column] += multiplier * sign * design.columns[index][column]
        total += multiplier * bound
    return nonnegative, combination, total


@pytest.mark.parametrize("scenario", ["twin", "scrambled"])
def test_every_transcribed_certificate_verifies_by_hand(scenario: str, request) -> None:
    population = request.getfixturevalue(scenario)
    result = request.getfixturevalue(f"{scenario}_result")
    space = default_space(population.covariate_names)
    checked = 0
    for record in result["forms"]:
        assert record["status"] == INFEASIBLE
        assert len(record["certificate"]["terms"]) >= 2
    for index, form in enumerate(space.forms()):
        if index >= 8:
            break
        record = result["forms"][index]
        assert record["form"]["id"] == form.identifier()
        design = build_design(population, space, form)
        nonnegative, combination, total = _hand_check(design, record["certificate"]["terms"])
        assert nonnegative, form.identifier()
        assert all(value == 0 for value in combination), form.identifier()
        assert total < 0, form.identifier()
        checked += 1
    assert checked == 8


def test_a_tampered_certificate_does_not_verify(twin: Population, twin_result: dict) -> None:
    space = default_space(twin.covariate_names)
    record = twin_result["forms"][0]
    form = space.forms()[0]
    assert record["form"]["id"] == form.identifier()
    design = build_design(twin, space, form)
    terms = record["certificate"]["terms"]
    assert recheck_farkas(design, terms)["valid"] is True

    def mutate(index: int, factor: Fraction) -> list[dict]:
        return [
            {**term, "multiplier": {"numerator": (_q(term["multiplier"]) * factor).numerator,
                                    "denominator": (_q(term["multiplier"]) * factor).denominator}}
            if position == index
            else term
            for position, term in enumerate(terms)
        ]

    negated = mutate(0, Fraction(-1))
    assert recheck_farkas(design, negated)["multipliers_nonnegative"] is False
    assert recheck_farkas(design, negated)["valid"] is False

    doubled = mutate(0, Fraction(2))
    assert recheck_farkas(design, doubled)["annihilates_the_model"] is False
    assert recheck_farkas(design, doubled)["valid"] is False

    zeroed = mutate(0, Fraction(0))
    assert recheck_farkas(design, zeroed)["valid"] is False

    flipped = [{**term, "bound": "lower" if term["bound"] == "upper" else "upper"}
               for term in terms]
    assert recheck_farkas(design, flipped)["valid"] is False

    # A positive rescaling of the whole witness is still a certificate: the verifier must not
    # pass by rejecting everything handed to it.
    halved = [
        {**term, "multiplier": {"numerator": (_q(term["multiplier"]) / 2).numerator,
                                "denominator": (_q(term["multiplier"]) / 2).denominator}}
        for term in terms
    ]
    assert recheck_farkas(design, halved)["valid"] is True

    with pytest.raises(ReconciliationError):
        recheck_farkas(design, [{**terms[0], "row": "NO-SUCH-OBJECT::alpha"}])


# ---------------------------------------------------------------------------
# The break-even coverage is a prediction, and it is tested as one
# ---------------------------------------------------------------------------


def test_twin_break_even_is_exactly_two(twin_result: dict) -> None:
    space_break = twin_result["obstruction"]["space_break_even_coverage"]
    assert space_break["coverage_independent"] is False
    # Two intervals of halfwidth h whose centres sit 4h apart touch exactly when the
    # halfwidths double.  The certificate has to say 2, and it does, exactly.
    assert _q(space_break["value"]["exact"]) == Fraction(2)


@pytest.mark.parametrize("scenario", ["twin", "scrambled"])
def test_the_obstruction_survives_every_widening_below_its_break_even(
    scenario: str, request
) -> None:
    population = request.getfixturevalue(scenario)
    result = request.getfixturevalue(f"{scenario}_result")
    space = default_space(population.covariate_names)
    floor = _q(result["obstruction"]["space_break_even_coverage"]["value"]["exact"])
    assert floor > 1
    below = reconcile(scale_population_intervals(population, floor * Fraction(999, 1000)), space)
    assert below["verdict"] == OBSTRUCTED
    assert below["reconciled_form_count"] == 0
    # ...and the search is not a constant "no": widen far enough and it reconciles.
    above = reconcile(scale_population_intervals(population, floor * 20), space)
    assert above["verdict"] == RECONCILED


def test_break_even_is_recomputed_from_the_multipliers_it_quotes(
    twin: Population, twin_result: dict
) -> None:
    space = default_space(twin.covariate_names)
    record = twin_result["forms"][0]
    design = build_design(twin, space, space.forms()[0])
    assert certificate_break_even(design, record["certificate"]["terms"]) == record["break_even"]
    with pytest.raises(ReconciliationError, match="negative"):
        certificate_break_even(
            design,
            [{**record["certificate"]["terms"][0],
              "multiplier": {"numerator": -1, "denominator": 1}}],
        )


def test_zero_width_intervals_give_a_coverage_independent_obstruction() -> None:
    space = LawSpace(
        name="constant-only",
        covariate_names=("x",),
        parameter_names=("alpha",),
        channels=channel_catalogue(("x",), (1,)),
        max_extra_channels=0,
    )
    objects = tuple(
        PopulationObject(
            label=f"OBJ-{index}",
            covariates=(("x", Fraction(index + 1)),),
            intervals=(("alpha", Fraction(index), Fraction(index)),),
        )
        for index in range(2)
    )
    population = Population(
        name="pinned",
        covariate_names=("x",),
        parameter_names=("alpha",),
        objects=objects,
        confirmation=("HOLD-0",),
        provenance="hand-declared control",
    )
    result = reconcile(population, space)
    assert result["verdict"] == OBSTRUCTED
    assert result["obstruction"]["space_break_even_coverage"]["coverage_independent"] is True


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------


def test_controls_all_pass() -> None:
    controls = reconciliation_controls()
    assert controls["all_honest_accepted"] is True, [
        row["name"] for row in controls["honest"] if not row["accepted"]
    ]
    assert controls["all_probes_rejected"] is True, [
        row["name"] for row in controls["probes"] if not row["rejected"]
    ]
    assert len(controls["honest"]) == 5
    assert len(controls["probes"]) == 7


def test_receipt_seals_and_decides(receipt: dict) -> None:
    assert receipt["schema_version"] == RESULT_SCHEMA
    assert receipt["decision"] == "RECONCILIATION_SEARCH_CERTIFIED"
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert canonical_sha256(body) == receipt["content_sha256"]
    assert receipt["claims"]["sealed_no_refit_trial"] is False
    assert receipt["claims"]["may_be_cited_as_confirmation"] is False
    assert receipt["claims"]["synthetic_populations_only"] is True
    names = [scenario["population_name"] for scenario in receipt["scenarios"]]
    assert names == ["universal-8", "twin-6", "scrambled-8", "twin-6-tagged"]
    verdicts = [scenario["result"]["verdict"] for scenario in receipt["scenarios"]]
    assert verdicts == [RECONCILED, OBSTRUCTED, OBSTRUCTED, OBSTRUCTED]


def test_receipt_carries_no_floating_point(receipt: dict) -> None:
    def walk(node: Any, path: str) -> None:
        assert not isinstance(node, float), f"float on a certificate path at {path}"
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(receipt, "$")


def test_receipt_is_deterministic(receipt: dict) -> None:
    again = build_receipt()
    assert again == receipt


def test_every_compact_row_addresses_its_full_record(receipt: dict) -> None:
    for scenario in receipt["scenarios"]:
        result = scenario["result"]
        compact = {row["form_id"]: row for row in result["forms"]}
        assert len(compact) == result["coverage"]["declared_cardinality"]
        for record in result["transcribed_forms"]:
            row = compact[record["form"]["id"]]
            assert row["record_sha256"] == canonical_sha256(record)
            assert row["status"] == record["status"]


def test_receipt_verifies(receipt: dict) -> None:
    report = verify_receipt(receipt, regenerated=receipt)
    assert report["verified"] is True, report["findings"]
    assert all(report["regeneration"].values())


def test_population_is_regenerable_from_its_declared_generator(receipt: dict) -> None:
    for scenario in receipt["scenarios"]:
        declaration = scenario["result"]["population"]
        rebuilt = regenerate_population(declaration["generator"])
        assert rebuilt.declaration() == declaration
        assert rebuilt.binding() == scenario["result"]["population_binding_sha256"]


# ---------------------------------------------------------------------------
# Receipt tamper probes: every one of them must be caught
# ---------------------------------------------------------------------------


def _resealed(receipt: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    return {**body, "content_sha256": canonical_sha256(body)}


def test_widened_interval_in_the_receipt_is_caught(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    interval = tampered["scenarios"][1]["result"]["population"]["objects"][0]["intervals"][0]
    interval["upper"] = {"numerator": interval["upper"]["numerator"] * 4,
                         "denominator": interval["upper"]["denominator"]}
    report = verify_receipt(_resealed(tampered), regenerated=receipt)
    assert report["verified"] is False
    assert any("does not match its generator" in finding for finding in report["findings"])


def test_verdict_flipped_without_resealing_is_caught(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    tampered["scenarios"][1]["result"]["verdict"] = RECONCILED
    report = verify_receipt(tampered, regenerated=receipt)
    assert report["verified"] is False
    assert "seal does not match the body" in report["findings"]


def test_verdict_flipped_and_resealed_is_still_caught(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    tampered["scenarios"][1]["result"]["verdict"] = "SOMETHING_ELSE"
    report = verify_receipt(_resealed(tampered), regenerated=receipt)
    assert report["verified"] is False
    assert any("undeclared verdict" in finding for finding in report["findings"])
    assert report["regeneration"]["scenarios"] is False


def test_tampered_transcribed_multiplier_is_caught(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    record = tampered["scenarios"][1]["result"]["transcribed_forms"][0]
    record["certificate"]["terms"][0]["multiplier"] = {"numerator": 7, "denominator": 1}
    report = verify_receipt(_resealed(tampered), regenerated=receipt)
    assert report["verified"] is False
    assert any("does not verify" in finding for finding in report["findings"])


def test_tampered_break_even_is_caught(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    record = tampered["scenarios"][1]["result"]["transcribed_forms"][0]
    record["break_even"]["value"]["exact"] = {"numerator": 1, "denominator": 1000}
    report = verify_receipt(_resealed(tampered), regenerated=receipt)
    assert report["verified"] is False
    assert any("break-even was not recomputed" in finding for finding in report["findings"])


def test_tampered_compact_digest_is_caught(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    tampered["scenarios"][1]["result"]["forms"][0]["record_sha256"] = "0" * 64
    report = verify_receipt(_resealed(tampered), regenerated=receipt)
    assert report["verified"] is False
    assert any("does not address this record" in finding for finding in report["findings"])


def test_law_space_declaration_drift_is_caught(receipt: dict) -> None:
    tampered = deepcopy(receipt)
    tampered["scenarios"][1]["result"]["law_space"]["max_extra_channels_per_parameter"] = 3
    report = verify_receipt(_resealed(tampered), regenerated=receipt)
    assert report["verified"] is False
    assert any("law-space declaration drift" in finding for finding in report["findings"])


# ---------------------------------------------------------------------------
# CLI and the committed receipt
# ---------------------------------------------------------------------------


def test_cli_writes_and_verifies_a_sealed_receipt(tmp_path: Path) -> None:
    output = tmp_path / "search.json"
    assert main(["--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["decision"] == "RECONCILIATION_SEARCH_CERTIFIED"
    assert main(["--verify", str(output)]) == 0
    written["decision"] = "TAMPERED"
    output.write_text(json.dumps(written, indent=2, sort_keys=True), encoding="utf-8")
    assert main(["--verify", str(output)]) == 1


def test_module_runs_as_a_script(tmp_path: Path) -> None:
    output = tmp_path / "script.json"
    completed = subprocess.run(
        [sys.executable, "-m", "sigma_theory_compiler.reconciliation_search",
         "--output", str(output)],
        capture_output=True,
        check=False,
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[-2000:]
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == RESULT_SCHEMA


def test_committed_receipt_matches_a_fresh_regeneration(receipt: dict) -> None:
    path = ROOT / RECEIPT_PATH
    if not path.exists():
        pytest.skip("receipt not present")
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == receipt, "the committed receipt was edited rather than regenerated"
    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert path.read_bytes().replace(b"\r\n", b"\n") == text.encode("utf-8")
