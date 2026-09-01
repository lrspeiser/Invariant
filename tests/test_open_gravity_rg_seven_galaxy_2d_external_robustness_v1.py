from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_seven_galaxy_2d_external_robustness_v1 as packet,
)


def test_config_and_all_three_input_receipts_are_valid() -> None:
    config = packet.load_config(verify_package=False)
    packet.validate_config(config)
    inputs = packet._load_inputs(config)
    assert set(inputs) == {
        "SIX_EXTERNAL_FIXED_2D_SCORE",
        "HOLMBERG_II_FIXED_2D_SCORE",
        "BOUNDED_PRIMARY_LITERATURE_AND_PUBLICATION_SYNTHESIS",
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("robustness_contract", "external_primary_objects", 5),
        ("robustness_contract", "minimum_material_fractional_aggregate_improvement", 0.0),
        ("robustness_contract", "no_postscore_parameter_tuning", False),
        ("publication_contract", "bounded_primary_rg_corpus_size", 0),
        ("claim_boundary", "universal_rg_replication", True),
        ("claim_boundary", "publication_ready", True),
    ],
)
def test_material_config_mutations_fail(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(packet.load_config(verify_package=False))
    config[section][key] = value
    with pytest.raises(packet.SevenGalaxyRobustnessError):
        packet.validate_config(config)


def test_deterministic_analysis_retains_signal_and_counterexamples() -> None:
    config = packet.load_config(verify_package=False)
    receipt = packet.build_receipt(config)
    assert receipt["external_primary_result"]["rg_primary_object_wins"] == 0
    assert receipt["external_primary_result"]["rg_wins_all_three_comparators_in_full_cells"] == 3
    assert (
        receipt["external_primary_result"]["material_two_percent_improvement_over_newton"] is False
    )
    assert len(receipt["counterexamples"]) == 4


def test_ic2574_signal_is_retained_but_not_promoted() -> None:
    receipt = packet.build_receipt(packet.load_config(verify_package=False))
    row = receipt["ic2574_follow_up"]
    assert row["rg_winning_sensitivity_cells"] == 3
    assert row["sensitivity_cell_count"] == 4
    assert row["primary_cell_winner"] == "NEWTON_3D_DST"
    assert row["robust_object_replication"] is False


def test_simple_low_inclination_pattern_fails_external_crosscheck() -> None:
    receipt = packet.build_receipt(packet.load_config(verify_package=False))
    row = receipt["inclination_crosscheck"]
    assert row["holmberg_ii_i27_rg_wins"] > 0
    assert row["ngc6946_i32p6_rg_wins"] == 0
    assert row["simple_low_inclination_crossover_generalized"] is False


def test_leave_one_out_exposes_aggregate_instability() -> None:
    receipt = packet.build_receipt(packet.load_config(verify_package=False))
    summary = receipt["leave_one_out_summary"]
    assert summary == {"rg_rank_one_count": 1, "rg_beats_newton_count": 3, "reaggregations": 6}
    assert {row["omitted_object_id"] for row in receipt["leave_one_object_out"]} == {
        "NGC2841",
        "IC2574",
        "DDO154",
        "NGC5055",
        "NGC6946",
        "NGC7331",
    }


def test_publication_candidate_is_a_constraint_result_not_theory_confirmation() -> None:
    receipt = packet.build_receipt(packet.load_config(verify_package=False))
    claims = receipt["claim_boundary"]
    assert claims["negative_or_constraint_result_publication_candidate"] is True
    assert claims["universal_rg_replication"] is False
    assert claims["unique_theory_established"] is False
    assert claims["publication_ready"] is False


def test_analysis_does_not_open_raw_fits_or_prediction_arrays() -> None:
    source = packet._repo_path(packet.MODULE_PATH).read_text(encoding="utf-8")
    assert "astropy" not in source
    assert "fits." not in source
    assert "np.load" not in source


def test_atomic_no_clobber_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "sealed.json"
    assert packet._atomic_no_clobber(path, b"same") == "CREATED"
    assert packet._atomic_no_clobber(path, b"same") == "EXISTING_IDENTICAL"
    with pytest.raises(packet.SevenGalaxyRobustnessError):
        packet._atomic_no_clobber(path, b"different")


def test_config_is_valid_json_and_keeps_claim_narrow() -> None:
    config = json.loads(packet._repo_path(packet.CONFIG_PATH).read_text(encoding="utf-8"))
    assert (
        config["publication_contract"]["prior_2d_velocity_map_rg_test_found_in_bounded_corpus"]
        is False
    )
    assert "RG confirmed" in config["publication_contract"]["forbidden_claims"]


def test_package_seals_after_finalization() -> None:
    assert packet.load_config()["robustness_contract"]["leave_one_object_out_reaggregations"] == 6
