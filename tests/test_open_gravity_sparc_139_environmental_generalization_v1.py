from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_sparc_139_environmental_generalization_v1 as generalization,
)


@pytest.fixture(scope="module")
def config() -> dict:
    value = json.loads(Path(generalization.CONFIG_PATH).read_text(encoding="utf-8"))
    generalization.validate_config(value)
    return value


@pytest.fixture(scope="module")
def ledger(config: dict) -> dict:
    return generalization.build_score_ledger(config)


def test_chronology_and_disjoint_development_scope_are_explicit(config: dict) -> None:
    chronology = config["chronology_and_scope"]
    assert chronology["candidate_ids_selected_after_PHANGS_XCOP_and_NGC2903_response_access"]
    assert chronology["target_blind_claim"] is False
    assert chronology["historical_139_exploration_ledger_is_disjoint_from_NGC2903"]
    assert chronology["development_generalization_not_confirmation"]
    assert chronology["formula_or_grid_repair_after_this_freeze"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "PUBLICATION_READY"),
        (("chronology_and_scope", "target_blind_claim"), True),
        (("chronology_and_scope", "formula_or_grid_repair_after_this_freeze"), True),
        (("sparc_binding", "scored_exploration_galaxies"), 3),
        (("sparc_binding", "historical_confirmation_galaxies_scored"), 1),
        (("source_reconstruction", "density_driver_available"), True),
        (("source_reconstruction", "source_values_may_not_use_Vobs_or_eVobs"), False),
        (("candidate_program", "per_galaxy_parameters"), 1),
        (("scoring_and_adjudication", "response_based_source_cell_selection"), True),
        (("access_scope", "independent_rows_opened"), 1),
        (("access_scope", "network_calls"), 1),
    ],
)
def test_material_config_mutations_fail_closed(
    config: dict, path: tuple[str, ...], replacement: object
) -> None:
    mutated = copy.deepcopy(config)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(generalization.SparcGeneralizationError):
        generalization.validate_config(mutated)


def test_every_source_operator_has_data_paper_and_analytic_limits(config: dict) -> None:
    gate = config["published_and_analytic_admission_gate"]
    assert [row["id"] for row in gate["primary_sources"]] == [
        "SPARC_2016",
        "RAR_2016",
        "RAR_INDIVIDUAL_FITS_2018",
    ]
    assert all(row["url"].startswith("https://arxiv.org/abs/") for row in gate["primary_sources"])
    assert gate["mandatory_operator_limits"]["point_mass_outer_potential_tail"]
    assert gate["mandatory_operator_limits"]["point_mass_radial_hessian_norm"]
    assert gate["missing_source_disposition"] == "SOURCE_BLOCKED_RETAINED_NOT_SCORED"


def test_point_mass_potential_and_tidal_limits_pass(config: dict) -> None:
    benchmark = generalization._operator_benchmarks(config)
    assert benchmark["all_operator_gates_pass"] is True
    assert benchmark["point_mass_potential_max_fractional_error"] < 1.0e-6
    assert benchmark["point_mass_tidal_interior_max_fractional_error"] < 1.0e-5
    assert benchmark["RAR_at_F_equal_1_exact"] is True
    assert benchmark["maximum_solar_fractional_deviation"] < 1.0e-6


def test_point_mass_source_equations_have_the_expected_scaling() -> None:
    radius = np.geomspace(1.0e19, 1.0e21, 4097)
    gravity_mass = 6.67430e30
    g_b = gravity_mass / radius**2
    potential, tidal = generalization._radial_sources(radius, g_b, tail_multiplier=1.0)
    assert np.all(np.diff(potential) < 0.0)
    expected = math.sqrt(6.0) * gravity_mass / radius**3
    assert np.max(np.abs(tidal[1:-1] / expected[1:-1] - 1.0)) < 1.0e-5


def test_access_is_exactly_139_exploration_galaxies_and_excludes_ngc2903(
    ledger: dict,
) -> None:
    access = ledger["access"]
    assert access["container_galaxies_parsed"] == 175
    assert access["container_rows_parsed"] == 3391
    assert access["scored_exploration_galaxies"] == 139
    assert access["scored_exploration_rows"] == 2720
    assert access["historical_confirmation_galaxies_scored"] == 0
    assert access["NGC2903_scored"] is False
    for key in (
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        assert access[key] == 0


def test_all_frozen_leads_are_scored_or_source_blocked_and_retained(ledger: dict) -> None:
    scores = ledger["scores"]
    assert scores["candidate_count"] == 18
    assert scores["scored_candidate_count"] == 15
    assert scores["blocked_candidate_count"] == 3
    assert scores["source_cell_count"] == 3
    assert scores["score_row_count"] == 45
    assert scores["formula_row_evaluations"] == 15 * 3 * 2720
    assert scores["all_candidates_retained"] == 18
    assert all(row["retained"] is True for row in scores["candidate_summaries"])


def test_density_variants_are_not_replaced_by_a_fake_spherical_density(ledger: dict) -> None:
    blocked = ledger["scores"]["blocked_rows"]
    assert {row["candidate_id"] for row in blocked} == {
        "RAR_LOW_FIELD_TOTAL_GAIN__INTERACTION_01_POTENTIAL_DENSITY",
        "RAR_EXCESS_AMPLITUDE__INTERACTION_01_POTENTIAL_DENSITY",
        "RAR_TRANSITION_SCALE__INTERACTION_01_POTENTIAL_DENSITY",
    }
    assert all(row["unavailable_drivers"] == ["DENSITY"] for row in blocked)
    assert all(row["retained"] is True for row in blocked)


def test_none_of_the_fourteen_leads_generalizes_to_the_139_galaxies(ledger: dict) -> None:
    assert ledger["scores"]["generalization_signal_ids"] == []
    summaries = {row["candidate_id"]: row for row in ledger["scores"]["candidate_summaries"]}
    assert summaries["RAR_2016_EMPIRICAL"]["primary_fractional_improvement"] == 0.0
    assert (
        summaries["RAR_TRANSITION_SCALE__POTENTIAL_POWER_P0P25"]["primary_fractional_improvement"]
        < -0.12
    )


def test_previous_all_tracer_lead_retains_its_population_counterexample(ledger: dict) -> None:
    summaries = {row["candidate_id"]: row for row in ledger["scores"]["candidate_summaries"]}
    row = summaries["RAR_TRANSITION_SCALE__INTERACTION_04_POTENTIAL_TIDAL"]
    assert row["disposition"] == "SCORED"
    assert row["primary_fractional_improvement"] < -0.20
    assert row["primary_object_support"] == 46
    assert row["median_paired_fractional_improvement"] < -0.26
    assert row["worst_paired_fractional_improvement"] < -0.39
    assert len(row["primary_objects"]) == 139
    assert (
        sum(object_row["improves_strongest_control"] for object_row in row["primary_objects"]) == 46
    )


def test_no_theory_health_or_publication_claim_is_inferred(ledger: dict) -> None:
    boundary = ledger["claim_boundary"]
    assert boundary["post_response_disjoint_development_generalization_signal_only"] is True
    assert boundary["target_blind"] is False
    assert not any(
        value
        for key, value in boundary.items()
        if key == "independent_confirmation"
        or key.endswith("established")
        or key in {"publication_ready", "dark_matter_eliminated", "full_3D_source_validation"}
    )


def test_packet_rebuild_is_deterministic_and_public_receipt_omits_response_rows(
    config: dict, ledger: dict
) -> None:
    rebuilt, receipt = generalization.build_packet(config)
    assert rebuilt == ledger
    assert receipt["generalization_signal_ids"] == []
    assert receipt["decision"] == "NO_DISJOINT_SPARC_GENERALIZATION_SIGNAL_ALL_CANDIDATES_RETAINED"
    assert receipt["candidate_count"] == 18
    assert receipt["scored_candidate_count"] == 15
    assert receipt["blocked_candidate_count"] == 3
    rendered = json.dumps(receipt, sort_keys=True)
    for forbidden in ('"primary_objects"', '"v_obs"', '"e_v_obs"', '"worst_standardized_residual"'):
        assert forbidden not in rendered


def test_package_hash_pins_match_after_final_seal() -> None:
    if (
        generalization._MODULE_SEMANTIC_SHA256 == "0" * 64
        or generalization._TEST_RAW_SHA256 == "0" * 64
    ):
        pytest.skip("package self-pins are installed only at the final mutation seal")
    assert (
        generalization.module_semantic_sha256(generalization._repo_path(generalization.MODULE_PATH))
        == generalization._MODULE_SEMANTIC_SHA256
    )
    assert (
        generalization.file_sha256(generalization._repo_path(generalization.TEST_PATH))
        == generalization._TEST_RAW_SHA256
    )
