"""Controls for the widened SPARC sample and the per-object run across it.

Every positive assertion here is paired with a negative one that must fail: a dataset that
validates is matched by a dataset with one digit changed that does not, a cross-retrieval
control that agrees is matched by one edited to disagree, a fast solver that reproduces the
reference is matched by a deliberately wrong solver the guard has to reject, and a ladder
that is monotone is matched by a hand-built non-monotone one the builder has to refuse.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.per_object_law_decomposition import (
    CONSTANT,
    NO_POPULATION,
    VARIES,
    Axis,
    ConfirmationSetTouched,
    LawSpace,
    build_axis,
    critical_coverage,
    decompose,
    interval_at,
    newtonian_columns,
)
from sigma_theory_compiler.sigma_core import (
    SchemaViolation,
    canonical_json_bytes,
    canonical_sha256,
)
from sigma_theory_compiler.sparc_full_sample import (
    ADMISSION_RULE,
    CONFIRMATION_FRACTION,
    DATASET_PATH,
    DATASET_SCHEMA,
    DECLARED_SUBSET_PATH,
    FULL_SPLIT_RULE,
    FULL_SPLIT_SALT,
    LADDER_SIZES,
    PREDECESSOR_POPULATION_SIZE,
    PUBLISHED_GALAXY_COUNT,
    RECEIPT_PATH,
    RESULT_SCHEMA,
    TRIAL_TYPE,
    FullSampleError,
    admit,
    assemble,
    assert_monotone_ladder,
    constancy_ladder,
    critical_coverage_fast,
    crosscheck_declared_subset,
    dataset_digest,
    load_full_sample,
    solver_equivalence,
    validate_dataset,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]

PUBLISHED_POINT_COUNT = 3391
DECLARED_SUBSET_GALAXIES = 6
DECLARED_SUBSET_POINTS = 214
DECLARED_SUBSET_FIELDS = 1284
EXCLUDED_BY_ADMISSION = ("UGC01281",)

_CACHE: dict[str, Any] = {}


def _dataset() -> dict[str, Any]:
    if "dataset" not in _CACHE:
        _CACHE["dataset"] = json.loads((ROOT / DATASET_PATH).read_text(encoding="utf-8"))
    return copy.deepcopy(_CACHE["dataset"])


def _subset() -> dict[str, Any]:
    if "subset" not in _CACHE:
        _CACHE["subset"] = json.loads((ROOT / DECLARED_SUBSET_PATH).read_text(encoding="utf-8"))
    return copy.deepcopy(_CACHE["subset"])


def _receipt() -> dict[str, Any]:
    if "receipt" not in _CACHE:
        _CACHE["receipt"] = json.loads((ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))
    return _CACHE["receipt"]


def _population() -> Any:
    if "population" not in _CACHE:
        _CACHE["population"] = assemble(ROOT)
    return _CACHE["population"]


def _newtonian_space(population: Any) -> LawSpace:
    """The cheapest declared law space: exact rationals from the published columns alone."""

    return LawSpace(
        columns=newtonian_columns(population.prepared),
        meaning="the published baryons with one per-object rescale",
        name="newtonian_baryons_only",
        parameter="baryonic_rescale",
    )


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-seal a mutated dataset so a control tests the field it means to test."""

    for entry in payload["galaxies"]:
        entry["provenance"]["rows_sha256"] = canonical_sha256(
            {
                "name": entry["name"],
                "point_count": entry["point_count"],
                "rows": entry["rows"],
            }
        )
    payload["galaxy_digest_sha256"] = dataset_digest(payload["galaxies"])
    payload["selection"]["galaxy_count"] = len(payload["galaxies"])
    payload["selection"]["point_count"] = sum(item["point_count"] for item in payload["galaxies"])
    return payload


def _floats(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, float):
        found.append(path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_floats(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_floats(item, f"{path}[{index}]"))
    return found


# ---------------------------------------------------------------------------
# The dataset, and the schema control that must reject a tampered one
# ---------------------------------------------------------------------------


def test_widened_dataset_carries_the_whole_published_sample() -> None:
    payload = _dataset()
    validate_dataset(payload)
    assert payload["schema_version"] == DATASET_SCHEMA
    assert payload["selection"]["galaxy_count"] == PUBLISHED_GALAXY_COUNT
    assert payload["selection"]["point_count"] == PUBLISHED_POINT_COUNT
    assert payload["selection"]["widens"]["from_galaxy_count"] == DECLARED_SUBSET_GALAXIES
    assert payload["selection"]["widens"]["from_point_count"] == DECLARED_SUBSET_POINTS
    galaxies, provenance = load_full_sample(ROOT / DATASET_PATH)
    assert len(galaxies) == PUBLISHED_GALAXY_COUNT
    assert provenance["point_count"] == PUBLISHED_POINT_COUNT
    assert sum(galaxy.count for galaxy in galaxies) == PUBLISHED_POINT_COUNT


def test_every_galaxy_carries_its_own_provenance_and_its_own_seal() -> None:
    payload = _dataset()
    for entry in payload["galaxies"]:
        provenance = entry["provenance"]
        assert provenance["source_file"] == f"{entry['name']}_rotmod.dat"
        assert len(provenance["source_file_sha256"]) == 64
        assert len(provenance["rows_sha256"]) == 64
        assert len(provenance["published_header"]) == 3
        assert provenance["published_header"][0].startswith("# Distance")
        assert provenance["rows_sha256"] == canonical_sha256(
            {
                "name": entry["name"],
                "point_count": entry["point_count"],
                "rows": entry["rows"],
            }
        )
    digests = [entry["provenance"]["source_file_sha256"] for entry in payload["galaxies"]]
    assert len(set(digests)) == len(digests)


def test_every_published_value_is_a_decimal_string_and_no_float_survives() -> None:
    payload = _dataset()
    assert _floats(payload) == []
    for entry in payload["galaxies"]:
        for row in entry["rows"]:
            assert all(isinstance(field, str) for field in row)
            # An exact rational from the published literal, never a float.
            assert all(Fraction(field) == Fraction(field) for field in row)
    canonical_json_bytes(payload)


def test_dataset_control_rejects_one_altered_digit() -> None:
    payload = _dataset()
    entry = next(item for item in payload["galaxies"] if item["name"] == "NGC2403")
    entry["rows"][0][1] = "99.99"
    with pytest.raises(FullSampleError, match="do not match their own seal"):
        validate_dataset(payload)


def test_dataset_control_rejects_a_float_in_place_of_a_published_decimal() -> None:
    payload = _dataset()
    payload["galaxies"][0]["rows"][0][1] = 13.8
    with pytest.raises(FullSampleError, match="not a published fixed-point decimal string"):
        validate_dataset(payload)
    # And a second, independent refusal: the canonical hasher will not seal a float at all,
    # so a float cannot reach a certificate path even if a validator were bypassed.
    with pytest.raises(SchemaViolation, match="floating value forbidden"):
        canonical_sha256(payload["galaxies"][0])


def test_dataset_control_rejects_an_exponent_or_a_bare_integer() -> None:
    for smuggled in ("1.38e1", "13", "NaN", "+13.8"):
        payload = _dataset()
        payload["galaxies"][0]["rows"][0][1] = smuggled
        payload = _reseal(payload)
        with pytest.raises(FullSampleError, match="not a published fixed-point decimal string"):
            validate_dataset(payload)


def test_dataset_control_rejects_a_point_count_that_disagrees_with_the_rows() -> None:
    payload = _dataset()
    payload["galaxies"][0]["point_count"] += 1
    with pytest.raises(FullSampleError, match="point count disagrees"):
        validate_dataset(payload)


def test_dataset_control_rejects_non_increasing_radii() -> None:
    payload = _dataset()
    entry = next(item for item in payload["galaxies"] if item["name"] == "NGC2403")
    entry["rows"][0], entry["rows"][1] = entry["rows"][1], entry["rows"][0]
    payload = _reseal(payload)
    with pytest.raises(FullSampleError, match="strictly increasing"):
        validate_dataset(payload)


def test_dataset_control_rejects_a_non_positive_uncertainty() -> None:
    payload = _dataset()
    payload["galaxies"][0]["rows"][0][2] = "0.00"
    payload = _reseal(payload)
    with pytest.raises(FullSampleError, match="uncertainty is not positive"):
        validate_dataset(payload)


def test_dataset_control_rejects_a_reordered_or_duplicated_galaxy_list() -> None:
    payload = _dataset()
    payload["galaxies"] = list(reversed(payload["galaxies"]))
    payload = _reseal(payload)
    with pytest.raises(FullSampleError, match="canonical name order"):
        validate_dataset(payload)

    payload = _dataset()
    payload["galaxies"].insert(1, copy.deepcopy(payload["galaxies"][0]))
    payload = _reseal(payload)
    with pytest.raises(FullSampleError):
        validate_dataset(payload)


def test_dataset_control_rejects_a_tampered_dataset_seal() -> None:
    payload = _dataset()
    payload["galaxy_digest_sha256"] = "0" * 64
    with pytest.raises(FullSampleError, match="dataset seal"):
        validate_dataset(payload)


# ---------------------------------------------------------------------------
# The cross-retrieval control: the six declared galaxies must survive verbatim
# ---------------------------------------------------------------------------


def test_cross_retrieval_control_agrees_on_every_declared_galaxy() -> None:
    report = crosscheck_declared_subset(_dataset(), _subset())
    assert report["objects"] == DECLARED_SUBSET_GALAXIES
    assert report["disagreements"] == 0
    assert report["published_fields_compared"] == DECLARED_SUBSET_FIELDS
    assert all(check["rows_identical"] for check in report["checks"])
    assert all(check["distance_mpc_equal_as_numbers"] for check in report["checks"])
    assert sum(check["points"] for check in report["checks"]) == DECLARED_SUBSET_POINTS


def test_cross_retrieval_control_fails_on_one_edited_digit() -> None:
    payload = _reseal(_dataset())
    entry = next(item for item in payload["galaxies"] if item["name"] == "NGC3198")
    entry["rows"][5][1] = str(Fraction(entry["rows"][5][1]) + 1) + ".00"
    with pytest.raises(FullSampleError, match="disagree at published row"):
        crosscheck_declared_subset(payload, _subset())


def test_cross_retrieval_control_fails_when_a_declared_galaxy_is_missing() -> None:
    payload = _dataset()
    payload["galaxies"] = [item for item in payload["galaxies"] if item["name"] != "NGC6503"]
    with pytest.raises(FullSampleError, match="is declared by the repository and is missing"):
        crosscheck_declared_subset(payload, _subset())


def test_cross_retrieval_control_fails_on_a_dropped_row() -> None:
    payload = _dataset()
    entry = next(item for item in payload["galaxies"] if item["name"] == "DDO154")
    entry["rows"] = entry["rows"][:-1]
    entry["point_count"] -= 1
    with pytest.raises(FullSampleError, match="disagree on the point count"):
        crosscheck_declared_subset(payload, _subset())


# ---------------------------------------------------------------------------
# The split reads names only; the admission rule reads columns only, and later
# ---------------------------------------------------------------------------


def test_split_is_recomputable_from_the_declared_salt_and_names_alone() -> None:
    payload = _dataset()
    names = [entry["name"] for entry in payload["galaxies"]]
    count = int(CONFIRMATION_FRACTION * len(names))
    ordered = sorted(
        (hashlib.sha256(f"{FULL_SPLIT_SALT}|{name}".encode()).hexdigest(), name) for name in names
    )
    expected = sorted(name for _, name in ordered[:count])

    block = _receipt()["exploration_confirmation_split"]
    assert block["salt"] == FULL_SPLIT_SALT
    assert block["rule"] == FULL_SPLIT_RULE
    assert block["confirmation_count"] == count
    assert block["confirmation"] == expected
    assert set(block["confirmation"]) & set(block["exploration"]) == set()
    assert sorted(block["confirmation"] + block["exploration"]) == sorted(names)
    sealed = {key: value for key, value in block.items() if key != "split_sha256"}
    assert block["split_sha256"] == canonical_sha256(sealed)


def test_admission_excludes_exactly_the_named_galaxies_and_exhibits_their_rows() -> None:
    galaxies, provenance = load_full_sample(ROOT / DATASET_PATH)
    convention = provenance["mass_to_light_convention"]
    admitted, report = admit(
        galaxies, Fraction(convention["disk_3_6um"]), Fraction(convention["bulge_3_6um"])
    )
    assert report["rule"] == ADMISSION_RULE
    assert tuple(entry["object"] for entry in report["excluded"]) == EXCLUDED_BY_ADMISSION
    assert len(admitted) == PUBLISHED_GALAXY_COUNT - len(EXCLUDED_BY_ADMISSION)
    for entry in report["excluded"]:
        assert entry["offending_rows"]
        for row in entry["offending_rows"]:
            numerator = row["baryonic_v_bar_sq"]["exact"]["numerator"]
            assert numerator <= 0
            assert len(row["published_row"]) == 6


def test_admission_runs_after_the_split_and_cannot_move_an_object_across_it() -> None:
    population = _population()
    excluded = {entry["object"] for entry in population.admission["excluded"]}
    declared = set(population.split.exploration) | set(population.split.confirmation)
    # The split partitions every *published* name, admitted or not.
    assert excluded <= declared
    assert set(population.admission["excluded_from_exploration_set"]) == (
        excluded & set(population.split.exploration)
    )
    assert set(population.admission["excluded_from_confirmation_set"]) == (
        excluded & set(population.split.confirmation)
    )
    fitted = {galaxy.name for galaxy in population.exploration}
    assert fitted == set(population.split.exploration) - excluded
    assert fitted & set(population.split.confirmation) == set()


def test_a_confirmation_galaxy_handed_to_the_fitter_raises_rather_than_declines() -> None:
    population = _population()
    withheld = population.split.confirmation[0]
    intruder = next(galaxy for galaxy in population.admitted if galaxy.name == withheld)
    space = _newtonian_space(population)
    with pytest.raises(ConfirmationSetTouched):
        decompose(
            space,
            [*population.exploration[:1], intruder],
            population.rows,
            population.split,
            solver=critical_coverage_fast,
        )


# ---------------------------------------------------------------------------
# The fast solver is the reference, exactly -- and the guard can reject one that is not
# ---------------------------------------------------------------------------


def test_fast_solver_reproduces_the_reference_on_published_axes() -> None:
    population = _population()
    space = _newtonian_space(population)
    axes: list[tuple[str, Axis]] = []
    for galaxy in population.exploration[:12]:
        offsets, slopes = space.columns(galaxy)
        axes.append((galaxy.name, build_axis(offsets, slopes, population.rows[galaxy.name])))
    report = solver_equivalence(axes)
    assert report["agreements"] == len(axes)
    assert report["disagreements"] == 0
    assert all(check["same_binding_pair"] for check in report["checks"])


def test_fast_solver_guard_rejects_a_solver_that_is_not_the_reference() -> None:
    population = _population()
    space = _newtonian_space(population)
    galaxy = population.exploration[0]
    offsets, slopes = space.columns(galaxy)
    axis = build_axis(offsets, slopes, population.rows[galaxy.name])

    def halved(item: Axis) -> tuple[Fraction, tuple[str, str] | None]:
        value, pair = critical_coverage(item)
        return value / 2, pair

    with pytest.raises(FullSampleError, match="derivations disagree"):
        solver_equivalence([(galaxy.name, axis)], fast=halved)


def test_the_critical_coverage_is_the_minimum_it_claims_to_be() -> None:
    population = _population()
    space = _newtonian_space(population)
    for galaxy in population.exploration[:12]:
        offsets, slopes = space.columns(galaxy)
        axis = build_axis(offsets, slopes, population.rows[galaxy.name])
        critical, _ = critical_coverage_fast(axis)
        assert not interval_at(axis, critical)["empty"]
        assert interval_at(axis, critical * Fraction(999, 1000))["empty"]


def test_fast_solver_matches_the_reference_on_adversarial_axes() -> None:
    # Ties, a single row, a blind floor, and lopsided widths: the cases a walk could miss.
    cases = {
        "one_row": Axis((Fraction(3),), (Fraction(1),), ("a",), Fraction(0), None),
        "tied_alphas": Axis(
            (Fraction(1), Fraction(1), Fraction(1)),
            (Fraction(1), Fraction(2), Fraction(3)),
            ("a", "b", "c"),
            Fraction(0),
            None,
        ),
        "blind_floor_dominates": Axis(
            (Fraction(1), Fraction(2)),
            (Fraction(1), Fraction(1)),
            ("a", "b"),
            Fraction(500),
            "blind",
        ),
        "lopsided": Axis(
            (Fraction(0), Fraction(10**6)),
            (Fraction(1, 10**6), Fraction(10**6)),
            ("a", "b"),
            Fraction(0),
            None,
        ),
        "long_chain": Axis(
            tuple(Fraction(index * index, 7) for index in range(40)),
            tuple(Fraction(1, index + 1) for index in range(40)),
            tuple(f"r{index}" for index in range(40)),
            Fraction(0),
            None,
        ),
    }
    report = solver_equivalence(sorted(cases.items()))
    assert report["disagreements"] == 0
    assert report["agreements"] == len(cases)


# ---------------------------------------------------------------------------
# R2: which per-object parameters are constant, and which are not
# ---------------------------------------------------------------------------


def test_receipt_reports_a_verdict_for_every_declared_law_space() -> None:
    receipt = _receipt()
    summaries = receipt["law_space_summaries"]
    assert len(summaries) == receipt["counts"]["law_spaces"] == 14
    constancy = receipt["parameter_constancy_r2"]
    named = set(constancy["constant_law_spaces"]) | set(constancy["varies_law_spaces"]) | set(
        constancy["unresolved_law_spaces"]
    )
    assert named == set(summaries)
    for summary in summaries.values():
        assert summary["verdict_at_population_coverage"] in {CONSTANT, VARIES, NO_POPULATION}


def test_every_varies_verdict_exhibits_two_named_galaxies_and_an_exact_gap() -> None:
    receipt = _receipt()
    populations = receipt["per_object_population_r1"]
    checked = 0
    for name in receipt["parameter_constancy_r2"]["varies_law_spaces"]:
        certificate = receipt["law_space_summaries"][name]["variation_certificate"]
        assert certificate is not None
        assert certificate["kind"] == "disjoint_object_intervals"
        intervals = {
            entry["object"]: entry["interval_at_population_coverage"]
            for entry in populations[name]
        }
        low_object = certificate["highest_lower_bound_from"]
        high_object = certificate["lowest_upper_bound_from"]
        assert low_object != high_object
        low = Fraction(
            intervals[low_object]["lower"]["exact"]["numerator"],
            intervals[low_object]["lower"]["exact"]["denominator"],
        )
        high = Fraction(
            intervals[high_object]["upper"]["exact"]["numerator"],
            intervals[high_object]["upper"]["exact"]["denominator"],
        )
        gap = Fraction(
            certificate["gap"]["exact"]["numerator"],
            certificate["gap"]["exact"]["denominator"],
        )
        assert gap > 0
        assert low - high == gap
        # No third galaxy already separates the population more sharply than the pair named.
        for entry in intervals.values():
            entry_low = Fraction(
                entry["lower"]["exact"]["numerator"], entry["lower"]["exact"]["denominator"]
            )
            entry_high = Fraction(
                entry["upper"]["exact"]["numerator"], entry["upper"]["exact"]["denominator"]
            )
            assert entry_low <= low
            assert entry_high >= high
        checked += 1
    assert checked == len(receipt["parameter_constancy_r2"]["varies_law_spaces"])


def test_every_constant_verdict_exhibits_a_witness_inside_every_interval() -> None:
    receipt = _receipt()
    populations = receipt["per_object_population_r1"]
    for name in receipt["parameter_constancy_r2"]["constant_law_spaces"]:
        witness_block = receipt["law_space_summaries"][name]["witness"]
        assert witness_block is not None
        witness = Fraction(
            witness_block["exact"]["numerator"], witness_block["exact"]["denominator"]
        )
        for entry in populations[name]:
            interval = entry["interval_at_population_coverage"]
            low = Fraction(
                interval["lower"]["exact"]["numerator"],
                interval["lower"]["exact"]["denominator"],
            )
            high = Fraction(
                interval["upper"]["exact"]["numerator"],
                interval["upper"]["exact"]["denominator"],
            )
            assert low <= witness <= high


def test_the_whole_population_is_kept_and_nothing_is_ranked() -> None:
    receipt = _receipt()
    fitted = receipt["counts"]["exploration_galaxies_fitted"]
    for name, population in receipt["per_object_population_r1"].items():
        assert len(population) == fitted, name
        assert [entry["object"] for entry in population] == sorted(
            entry["object"] for entry in population
        )
        for entry in population:
            assert entry["points"] >= 1
            assert entry["critical_coverage"]["exact"]["denominator"] > 0


# ---------------------------------------------------------------------------
# The contrast the widening exists to publish
# ---------------------------------------------------------------------------


def test_four_galaxy_blocks_call_it_constant_where_the_whole_population_does_not() -> None:
    receipt = _receipt()
    survey = receipt["parameter_constancy_r2"]["four_object_block_survey"]
    assert set(survey) == set(receipt["law_space_summaries"])
    constant_blocks = sum(entry["blocks_constant"] for entry in survey.values())
    total_blocks = sum(entry["blocks_total"] for entry in survey.values())
    assert total_blocks > 0
    # The measurement: a per-object parameter that is CONSTANT on four galaxies at a time
    # is VARIES once every galaxy has to agree.
    assert constant_blocks > 0
    assert receipt["parameter_constancy_r2"]["constant_law_spaces"] == []
    for name, entry in survey.items():
        assert entry["block_size"] == PREDECESSOR_POPULATION_SIZE
        assert entry["blocks_constant"] + entry["blocks_varies"] == entry["blocks_total"]
        assert len(entry["remainder_objects"]) < PREDECESSOR_POPULATION_SIZE
        objects = [name for block in entry["blocks"] for name in block["objects"]]
        assert len(set(objects)) == len(objects)
        assert (
            len(objects) + len(entry["remainder_objects"])
            == receipt["counts"]["exploration_galaxies_fitted"]
        )
        assert receipt["law_space_summaries"][name]["four_object_blocks_calling_it_constant"] == (
            entry["blocks_constant"]
        )


def test_the_deliberately_wrong_law_is_among_the_laws_four_galaxies_call_constant() -> None:
    # The point of the control: constancy on a small sample is a statement about the
    # sample, not about the law. If the wrong law could not also earn it, the measurement
    # would not be showing what it claims to show.
    survey = _receipt()["parameter_constancy_r2"]["four_object_block_survey"]
    assert survey["deliberately_wrong_law"]["blocks_constant"] > 0


def test_the_wrong_law_control_still_separates_on_the_widened_sample() -> None:
    receipt = _receipt()
    control = receipt["wrong_law_control"]
    assert control["held"] is True
    assert control["families_it_did_not_beat"] == []
    summaries = receipt["law_space_summaries"]

    def coverage(name: str) -> Fraction:
        block = summaries[name]["smallest_coverage_with_a_population"]["exact"]
        return Fraction(block["numerator"], block["denominator"])

    wrong = coverage("deliberately_wrong_law")
    families = [name for name in summaries if name.startswith("family_")]
    assert families
    for name in families:
        assert coverage(name) < wrong


def test_the_fixed_coverage_ladder_is_monotone_in_every_law_space() -> None:
    receipt = _receipt()
    for name, ladder in receipt["parameter_constancy_r2"]["constancy_ladders"].items():
        sizes = [step["objects"] for step in ladder["steps"]]
        assert sizes == sorted(sizes)
        assert sizes[-1] == receipt["counts"]["exploration_galaxies_fitted"]
        assert sizes[0] == LADDER_SIZES[0]
        seen_varies = False
        for step in ladder["steps"]:
            if step["verdict"] == VARIES:
                seen_varies = True
            elif seen_varies:
                raise AssertionError(f"{name}: the ladder recovered CONSTANT after VARIES")
        assert ladder["monotone"] is True


def test_the_ladder_guard_refuses_a_sequence_that_recovers_constancy() -> None:
    # The guard is fail-closed on an invariant nested prefixes cannot violate, so it is
    # exercised by calling it directly. A guard reachable only through data that cannot
    # exist is a guard nobody has tested.
    assert_monotone_ladder([CONSTANT, CONSTANT, VARIES, VARIES])
    assert_monotone_ladder([VARIES, VARIES])
    assert_monotone_ladder([NO_POPULATION, CONSTANT, VARIES])
    with pytest.raises(FullSampleError, match="recovered a CONSTANT verdict"):
        assert_monotone_ladder([CONSTANT, VARIES, CONSTANT])
    with pytest.raises(FullSampleError, match="recovered a CONSTANT verdict"):
        assert_monotone_ladder([VARIES, CONSTANT])


def test_the_ladder_built_from_real_intervals_is_the_one_the_guard_accepted() -> None:
    # The positive side: a real law space's ladder, rebuilt here from the receipt's own
    # per-object intervals, reproduces the verdict sequence the receipt published.
    receipt = _receipt()
    name = "newtonian_baryons_only"
    ladder = receipt["parameter_constancy_r2"]["constancy_ladders"][name]
    population = receipt["per_object_population_r1"][name]
    order = [
        item
        for _, item in sorted(
            (receipt["exploration_confirmation_split"]["name_digests"][entry["object"]],
             entry["object"])
            for entry in population
        )
    ]
    decomposition = {
        "population": [
            {
                "object": entry["object"],
                "interval_at_population_coverage": entry["interval_at_population_coverage"],
            }
            for entry in population
        ],
        "smallest_coverage_with_a_population": ladder["coverage_factor"],
    }
    rebuilt = constancy_ladder(decomposition, order)
    assert [step["verdict"] for step in rebuilt["steps"]] == [
        step["verdict"] for step in ladder["steps"]
    ]
    assert rebuilt["breaks_at_objects"] == ladder["breaks_at_objects"]


def test_widening_can_only_remove_constancy_never_create_it() -> None:
    # The monotonicity claim, checked directly on the receipt's own intervals: at one fixed
    # coverage, any value shared by every object of a superset is shared by every object of
    # a subset, so a CONSTANT full population forces CONSTANT on every prefix.
    receipt = _receipt()
    for name, ladder in receipt["parameter_constancy_r2"]["constancy_ladders"].items():
        final = ladder["steps"][-1]["verdict"]
        if final == CONSTANT:
            assert all(step["verdict"] == CONSTANT for step in ladder["steps"]), name


# ---------------------------------------------------------------------------
# The receipt itself
# ---------------------------------------------------------------------------


def test_receipt_is_sealed_exact_and_free_of_floats() -> None:
    receipt = _receipt()
    assert receipt["schema_version"] == RESULT_SCHEMA
    assert receipt["trial_type"] == TRIAL_TYPE
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)
    assert _floats(receipt) == []
    canonical_json_bytes(receipt)


def test_receipt_refuses_a_confirmation_galaxy_in_a_fitted_population() -> None:
    receipt = copy.deepcopy(_receipt())
    withheld = receipt["exploration_confirmation_split"]["confirmation"][0]
    law = next(iter(receipt["per_object_population_r1"]))
    receipt["per_object_population_r1"][law][0]["object"] = withheld
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = canonical_sha256(body)
    with pytest.raises(FullSampleError, match="confirmation-set object"):
        validate_receipt(receipt, root=ROOT)


def test_receipt_refuses_a_broken_seal_and_a_rewritten_rule() -> None:
    receipt = copy.deepcopy(_receipt())
    receipt["content_sha256"] = "0" * 64
    with pytest.raises(FullSampleError, match="receipt seal changed"):
        validate_receipt(receipt, root=ROOT)

    receipt = copy.deepcopy(_receipt())
    receipt["exploration_confirmation_split"]["salt"] = "convenient"
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = canonical_sha256(body)
    with pytest.raises(FullSampleError, match="split seal changed"):
        validate_receipt(receipt, root=ROOT)

    receipt = copy.deepcopy(_receipt())
    receipt["claims"]["confirmation_set_fitted"] = True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = canonical_sha256(body)
    with pytest.raises(FullSampleError, match="claims changed"):
        validate_receipt(receipt, root=ROOT)


def test_receipt_records_the_provenance_of_every_galaxy_it_could_have_fitted() -> None:
    receipt = _receipt()
    provenance = receipt["data_provenance"]["per_galaxy_provenance"]
    assert len(provenance) == PUBLISHED_GALAXY_COUNT
    control = receipt["data_provenance"]["cross_retrieval_control"]
    assert control["disagreements"] == 0
    assert control["published_fields_compared"] == DECLARED_SUBSET_FIELDS
    assert receipt["exploratory_caveat"]["may_be_cited_as_confirmation"] is False
    assert receipt["exploratory_caveat"]["sealed_no_refit_trial"] is False
    # The withheld set is mostly data nothing in this repository has ever read.
    overlap = receipt["exploratory_caveat"][
        "confirmation_set_overlap_with_the_six_already_scanned"
    ]
    withheld = receipt["exploration_confirmation_split"]["confirmation"]
    assert set(overlap) <= set(withheld)
    assert len(overlap) < len(withheld)


def test_the_solver_equivalence_covers_the_pooled_axis_it_exists_for() -> None:
    receipt = _receipt()
    equivalence = receipt["solver_equivalence"]
    assert equivalence["disagreements"] == 0
    pooled = [check for check in equivalence["checks"] if check["axis"].endswith("|pooled")]
    assert pooled
    assert max(check["rows"] for check in pooled) == receipt["counts"][
        "exploration_points_fitted"
    ]
    assert all(check["same_binding_pair"] for check in equivalence["checks"])


def test_the_simplex_agrees_with_the_closed_form_on_every_fitted_galaxy() -> None:
    receipt = _receipt()
    for name, report in receipt["instrument_crosscheck"].items():
        assert report["disagreements"] == 0, name
        assert report["agreements"] == receipt["counts"]["exploration_galaxies_fitted"]
        for check in report["checks"]:
            assert check["at_critical_coverage"] == "FEASIBLE"
            assert check["below_critical_coverage"] == "INFEASIBLE"
            assert check["farkas_term_count_below"] > 0


def test_receipt_replays_exactly() -> None:
    validate_receipt(_receipt(), root=ROOT)
