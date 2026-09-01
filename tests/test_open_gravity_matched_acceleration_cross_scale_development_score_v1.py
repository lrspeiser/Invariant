from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_development_score_v1 as scorer,
)


@pytest.fixture(scope="module")
def config() -> dict:
    value = json.loads(Path(scorer.CONFIG_PATH).read_text(encoding="utf-8"))
    scorer.validate_config(value)
    return value


@pytest.fixture(scope="module")
def ledger(config: dict) -> dict:
    return scorer.build_score_ledger(config)


def test_scoring_contract_separates_data_fit_from_theory_health(config: dict) -> None:
    adjudication = config["adjudication"]
    assert adjudication["theory_health_can_veto_data_fit_signal"] is False
    assert adjudication["theory_health_is_separate_followup"] is True
    assert adjudication["candidate_repair_or_tuning_after_response_access"] is False
    assert adjudication["global_discovery_p_value_claimed"] is False
    assert config["galaxy_scoring"]["response_based_source_cell_selection"] is False
    assert config["galaxy_scoring"]["best_source_cell_is_diagnostic_only"] is True


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "PUBLICATION_READY"),
        (("galaxy_scoring", "response_based_source_cell_selection"), True),
        (("xcop_scoring", "minimum_fractional_error"), 0.0),
        (("xcop_scoring", "outer_pressure_boundary_scored"), True),
        (("xcop_scoring", "inferred_total_mass_used"), True),
        (("adjudication", "theory_health_can_veto_data_fit_signal"), True),
        (("adjudication", "candidate_repair_or_tuning_after_response_access"), True),
        (("access_scope", "confirmation_rows_opened"), 1),
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
    with pytest.raises(scorer.DevelopmentScoreError):
        scorer.validate_config(mutated)


def test_every_response_has_real_data_and_primary_method_anchor(config: dict) -> None:
    assert [row["id"] for row in config["primary_data_and_method_anchors"]] == [
        "PHANGS_CO_KINEMATICS_2020",
        "SPARC_2016",
        "XCOP_THERMODYNAMIC_PROFILES_2019",
        "HYDROSTATIC_FORWARD_EQUATION",
    ]
    assert all(
        row["url"].startswith("https://") and row["role"]
        for row in config["primary_data_and_method_anchors"]
    )


def test_access_scope_is_exact_and_development_only(ledger: dict) -> None:
    access = ledger["access"]
    assert access["PHANGS_CO"] == {
        "container_objects_opened": 33,
        "container_response_rows_opened": 1321,
        "scored_objects": ["NGC2903", "NGC3351", "NGC3627"],
        "selected_rows_available": 110,
    }
    assert access["SPARC"]["container_objects_opened"] == 175
    assert access["SPARC"]["container_response_rows_opened"] == 3391
    assert access["SPARC"]["scored_objects"] == ["NGC2903"]
    assert access["XCOP"]["input_files_opened"] == 29
    assert access["XCOP"]["response_rows_opened"] == 184
    assert access["XCOP"]["response_rows_scored"] == 156
    for key in (
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
        "candidate_generation_after_response_access",
    ):
        assert access[key] == 0


def test_all_fixed_candidates_and_source_cells_are_scored_or_explicitly_blocked(
    ledger: dict,
) -> None:
    galaxy = ledger["galaxy_scores"]
    xcop = ledger["xcop_scores"]
    assert galaxy["source_cell_count"] == 75
    assert galaxy["score_row_count"] == 32 * 75
    assert len(galaxy["candidate_summaries"]) == 32
    assert xcop["score_row_count"] == 24 * 3 * 8
    assert xcop["blocked_row_count"] == 8 * 3 * 8
    assert len(xcop["candidate_summaries"]) == 32
    assert len(ledger["adjudication"]["candidate_rows"]) == 32
    assert all(row["retained"] is True for row in ledger["adjudication"]["candidate_rows"])


def test_galaxy_primary_scores_retain_every_object_counterexample(ledger: dict) -> None:
    by_id = {row["candidate_id"]: row for row in ledger["galaxy_scores"]["candidate_summaries"]}
    potential = by_id["POTENTIAL_POWER_P0P75"]["PHANGS_CO"]
    assert potential["primary_object_support"] == 0
    assert [row["object_id"] for row in potential["primary_objects"]] == [
        "NGC2903",
        "NGC3351",
        "NGC3627",
    ]
    assert [row["rows_scored"] for row in potential["primary_objects"]] == [35, 16, 50]
    assert all(row["loss"] > 0.0 for row in potential["primary_objects"])
    assert potential["primary_fractional_improvement"] < -1.0
    assert potential["best_source_cell_diagnostic_only"]["source_cell_id"]


def test_potential_depth_has_a_large_cluster_signal_but_not_cross_scale(ledger: dict) -> None:
    cluster = {row["candidate_id"]: row for row in ledger["xcop_scores"]["candidate_summaries"]}
    p075 = cluster["POTENTIAL_POWER_P0P75"]
    assert p075["disposition"] == "SCORED"
    assert p075["nominal_fractional_improvement"] > 0.79
    assert p075["nominal_object_support"] == 8
    nominal = next(
        row for row in p075["scenario_results"] if row["scenario_id"] == "XCOP-SOURCE-NOMINAL"
    )
    assert len(nominal["objects"]) == 8
    assert {row["object_id"] for row in nominal["objects"]} == {
        "A1644",
        "A1795",
        "A2142",
        "A2255",
        "A2319",
        "A3266",
        "A85",
        "ZW1215",
    }


def test_no_cross_scale_signal_but_seven_domain_specific_leads_are_retained(
    ledger: dict,
) -> None:
    adjudication = ledger["adjudication"]
    assert adjudication["data_fit_signal_ids"] == []
    assert adjudication["domain_specific_signal_ids"] == [
        "POTENTIAL_POWER_P0P75",
        "POTENTIAL_POWER_P1",
        "POTENTIAL_POWER_P0P5",
        "INTERACTION_04_POTENTIAL_TIDAL",
        "INTERACTION_02_POTENTIAL_DENSITY",
        "TIDAL_POWER_M0P5",
        "DENSITY_POWER_M0P5",
    ]
    assert adjudication["theory_health_separate"] is True
    assert adjudication["all_candidates_retained"] == 32


def test_geometry_candidates_are_not_faked_on_spherical_clusters(ledger: dict) -> None:
    blocked_ids = {row["candidate_id"] for row in ledger["xcop_scores"]["blocked_rows"]}
    assert len(blocked_ids) == 8
    assert all("GEOMETRY" in candidate_id for candidate_id in blocked_ids)
    summaries = {row["candidate_id"]: row for row in ledger["xcop_scores"]["candidate_summaries"]}
    assert all(
        summaries[candidate_id]["disposition"] == "SOURCE_BLOCKED_NONSPHERICAL_GEOMETRY"
        for candidate_id in blocked_ids
    )


def test_claim_ceiling_does_not_overstate_the_development_result(ledger: dict) -> None:
    boundary = ledger["claim_boundary"]
    assert boundary["development_data_fit_signal_only"] is True
    assert not any(
        value for key, value in boundary.items() if key != "development_data_fit_signal_only"
    )


def test_packet_summary_is_deterministic_and_contains_no_raw_response_arrays(
    config: dict, ledger: dict
) -> None:
    rebuilt, receipt = scorer.build_packet(config)
    assert rebuilt == ledger
    assert receipt["data_fit_signal_ids"] == []
    assert len(receipt["domain_specific_signal_ids"]) == 7
    assert receipt["galaxy_score_rows"] == 2400
    assert receipt["xcop_score_rows"] == 576
    assert receipt["xcop_blocked_rows"] == 192
    rendered = json.dumps(receipt, sort_keys=True)
    for forbidden in ('"observed"', '"pressure_kev_cm3"', '"temperature_kev"', '"v_obs"'):
        assert forbidden not in rendered


def test_package_hash_pins_match_after_final_seal() -> None:
    if scorer._MODULE_SEMANTIC_SHA256 == "0" * 64 or scorer._TEST_RAW_SHA256 == "0" * 64:
        pytest.skip("package self-pins are installed only at the final mutation seal")
    assert (
        scorer.module_semantic_sha256(scorer._repo_path(scorer.MODULE_PATH))
        == scorer._MODULE_SEMANTIC_SHA256
    )
    assert scorer.file_sha256(scorer._repo_path(scorer.TEST_PATH)) == scorer._TEST_RAW_SHA256
