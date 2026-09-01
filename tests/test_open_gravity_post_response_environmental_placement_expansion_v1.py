from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_post_response_environmental_placement_expansion_v1 as expansion,
)


@pytest.fixture(scope="module")
def config() -> dict:
    value = json.loads(Path(expansion.CONFIG_PATH).read_text(encoding="utf-8"))
    expansion.validate_config(value)
    return value


@pytest.fixture(scope="module")
def ledger(config: dict) -> dict:
    return expansion.build_score_ledger(config)


def test_chronology_is_explicit_and_never_claims_target_blindness(config: dict) -> None:
    chronology = config["chronology_and_honesty"]
    assert chronology["designed_after_development_response_access"] is True
    assert chronology["target_blind_claim"] is False
    assert chronology["same_development_data_reuse_disclosed"] is True
    assert chronology["independent_confirmation_required"] is True
    assert chronology["post_score_formula_repair_allowed"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "PUBLICATION_READY"),
        (("chronology_and_honesty", "target_blind_claim"), True),
        (("chronology_and_honesty", "post_score_formula_repair_allowed"), True),
        (("candidate_program", "new_candidate_count"), 1),
        (("candidate_program", "per_object_parameters"), 1),
        (("benchmarks_and_admission", "real_source_data_required"), False),
        (("benchmarks_and_admission", "primary_paper_or_exact_analytic_limit_required"), False),
        (("scoring_and_adjudication", "theory_health_separate"), False),
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
    with pytest.raises(expansion.PlacementExpansionError):
        expansion.validate_config(mutated)


def test_every_new_mechanism_has_real_sources_and_a_published_or_analytic_anchor(
    config: dict,
) -> None:
    gate = config["benchmarks_and_admission"]
    assert gate["real_source_data_required"] is True
    assert gate["primary_paper_or_exact_analytic_limit_required"] is True
    assert gate["failed_benchmark_disposition"] == "RETAINED_NOT_SCORED"
    assert config["published_method_anchor"]["id"] == "RAR_2016"
    assert config["published_method_anchor"]["url"] == "https://arxiv.org/abs/1610.06183"


def test_registry_is_exactly_four_controls_plus_84_charged_hypotheses(
    config: dict,
) -> None:
    source_config, _scorer_config, source_ledger = expansion._load_predecessors(config)
    rows = expansion.candidate_registry(config, source_ledger)
    assert len(rows) == 88
    assert len({row["candidate_id"] for row in rows}) == 88
    assert sum(row["kind"].startswith("PUBLISHED") for row in rows) == 4
    assert sum(row["kind"] == "NEW_POST_RESPONSE_REPAIR_HYPOTHESIS" for row in rows) == 84
    assert {row["placement_id"] for row in rows if row["kind"].startswith("NEW_")} == {
        "RAR_TRANSITION_SCALE",
        "RAR_EXCESS_AMPLITUDE",
        "RAR_LOW_FIELD_TOTAL_GAIN",
    }
    assert source_config["constants"]["a0_m_s2"] > 0.0


def test_all_placements_recover_published_rar_at_unit_factor(config: dict) -> None:
    source_config, _scorer_config, _source_ledger = expansion._load_predecessors(config)
    benchmark = expansion._benchmark_placements(config, source_config)
    assert benchmark["RAR_at_F_equal_1_exact"] is True
    assert benchmark["maximum_solar_fractional_deviation"] < 1.0e-6


def test_access_is_exact_and_no_new_domain_is_opened(ledger: dict) -> None:
    access = ledger["access"]
    assert access["PHANGS_CO"]["container_objects_opened"] == 33
    assert access["PHANGS_CO"]["container_response_rows_opened"] == 1321
    assert access["SPARC"]["container_objects_opened"] == 175
    assert access["SPARC"]["container_response_rows_opened"] == 3391
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
    ):
        assert access[key] == 0


def test_all_candidates_are_scored_or_explicitly_source_blocked(ledger: dict) -> None:
    assert ledger["galaxy_scores"]["source_cell_count"] == 75
    assert ledger["galaxy_scores"]["score_row_count"] == 88 * 75
    assert len(ledger["galaxy_scores"]["candidate_summaries"]) == 88
    assert ledger["xcop_scores"]["score_row_count"] == 64 * 3 * 8
    assert ledger["xcop_scores"]["blocked_row_count"] == 24 * 3 * 8
    assert len(ledger["xcop_scores"]["candidate_summaries"]) == 88
    assert ledger["adjudication"]["all_candidates_retained"] == 88
    assert ledger["adjudication"]["charged_post_response_candidates"] == 84
    assert all(row["retained"] is True for row in ledger["adjudication"]["candidate_rows"])


def test_geometry_is_not_faked_for_spherical_clusters(ledger: dict) -> None:
    blocked = ledger["xcop_scores"]["blocked_rows"]
    assert len(blocked) == 576
    assert len({row["candidate_id"] for row in blocked}) == 24
    assert all("GEOMETRY" in row["candidate_id"] for row in blocked)


def test_fourteen_post_response_development_signals_are_retained(ledger: dict) -> None:
    assert ledger["adjudication"]["signal_ids"] == [
        "RAR_LOW_FIELD_TOTAL_GAIN__INTERACTION_01_POTENTIAL_DENSITY",
        "RAR_LOW_FIELD_TOTAL_GAIN__INTERACTION_03_POTENTIAL_TIDAL",
        "RAR_TRANSITION_SCALE__POTENTIAL_POWER_P0P75",
        "RAR_LOW_FIELD_TOTAL_GAIN__POTENTIAL_POWER_P0P25",
        "RAR_TRANSITION_SCALE__POTENTIAL_POWER_P0P5",
        "RAR_EXCESS_AMPLITUDE__INTERACTION_01_POTENTIAL_DENSITY",
        "RAR_TRANSITION_SCALE__INTERACTION_04_POTENTIAL_TIDAL",
        "RAR_EXCESS_AMPLITUDE__INTERACTION_03_POTENTIAL_TIDAL",
        "RAR_EXCESS_AMPLITUDE__POTENTIAL_POWER_P0P5",
        "RAR_EXCESS_AMPLITUDE__POTENTIAL_POWER_P0P25",
        "RAR_TRANSITION_SCALE__POTENTIAL_POWER_P1",
        "RAR_TRANSITION_SCALE__INTERACTION_01_POTENTIAL_DENSITY",
        "RAR_TRANSITION_SCALE__INTERACTION_03_POTENTIAL_TIDAL",
        "RAR_TRANSITION_SCALE__POTENTIAL_POWER_P0P25",
    ]


def test_potential_tidal_transition_is_the_only_lead_improving_all_three_tracers(
    ledger: dict,
) -> None:
    signals = {
        row["candidate_id"]: row
        for row in ledger["adjudication"]["candidate_rows"]
        if row["post_response_development_repair_signal"]
    }
    all_three = [
        candidate_id
        for candidate_id, row in signals.items()
        if row["PHANGS_primary_fractional_improvement"] > 0.02
        and row["SPARC_primary_fractional_improvement"] > 0.02
        and row["XCOP_nominal_fractional_improvement"] > 0.02
    ]
    assert all_three == ["RAR_TRANSITION_SCALE__INTERACTION_04_POTENTIAL_TIDAL"]
    row = signals[all_three[0]]
    assert row["PHANGS_primary_object_support"] == 2
    assert row["XCOP_nominal_object_support"] == 8
    assert row["PHANGS_primary_fractional_improvement"] > 0.039
    assert row["SPARC_primary_fractional_improvement"] > 0.053
    assert row["XCOP_nominal_fractional_improvement"] > 0.774


def test_numerical_top_rank_retains_its_sparc_counterexample(ledger: dict) -> None:
    top = ledger["adjudication"]["candidate_rows"][0]
    assert top["candidate_id"] == "RAR_LOW_FIELD_TOTAL_GAIN__INTERACTION_01_POTENTIAL_DENSITY"
    assert top["PHANGS_primary_fractional_improvement"] > 0.44
    assert top["XCOP_nominal_fractional_improvement"] > 0.75
    assert top["SPARC_primary_fractional_improvement"] < -5.9


def test_potential_depth_transition_lead_is_strong_but_sparc_negative(ledger: dict) -> None:
    rows = {row["candidate_id"]: row for row in ledger["adjudication"]["candidate_rows"]}
    row = rows["RAR_TRANSITION_SCALE__POTENTIAL_POWER_P0P75"]
    assert row["PHANGS_primary_fractional_improvement"] > 0.22
    assert row["XCOP_nominal_fractional_improvement"] > 0.78
    assert row["SPARC_primary_fractional_improvement"] < -0.36


def test_claim_ceiling_remains_post_response_and_unconfirmed(ledger: dict) -> None:
    boundary = ledger["claim_boundary"]
    assert boundary["post_response_development_repair_signal_only"] is True
    assert boundary["target_blind"] is False
    assert not any(
        value
        for key, value in boundary.items()
        if key not in {"post_response_development_repair_signal_only"}
    )


def test_packet_is_deterministic_and_public_receipt_omits_response_arrays(
    config: dict, ledger: dict
) -> None:
    rebuilt, receipt = expansion.build_packet(config)
    assert rebuilt == ledger
    assert receipt["candidate_count"] == 88
    assert receipt["new_candidate_count"] == 84
    assert len(receipt["signal_ids"]) == 14
    assert receipt["galaxy_score_rows"] == 6600
    assert receipt["xcop_score_rows"] == 1536
    assert receipt["xcop_blocked_rows"] == 576
    rendered = json.dumps(receipt, sort_keys=True)
    for forbidden in ('"observed"', '"pressure_kev_cm3"', '"temperature_kev"', '"v_obs"'):
        assert forbidden not in rendered


def test_package_hash_pins_match_after_final_seal() -> None:
    if expansion._MODULE_SEMANTIC_SHA256 == "0" * 64 or expansion._TEST_RAW_SHA256 == "0" * 64:
        pytest.skip("package self-pins are installed only at the final mutation seal")
    assert (
        expansion.module_semantic_sha256(expansion._repo_path(expansion.MODULE_PATH))
        == expansion._MODULE_SEMANTIC_SHA256
    )
    assert (
        expansion.file_sha256(expansion._repo_path(expansion.TEST_PATH))
        == expansion._TEST_RAW_SHA256
    )
