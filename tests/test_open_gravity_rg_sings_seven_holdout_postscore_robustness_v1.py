from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_sings_seven_holdout_postscore_robustness_v1 as package,
)


def test_config_freezes_postscore_no_retuning_boundary() -> None:
    config = package.load_config()
    analysis = config["analysis_contract"]
    assert analysis["parameters_fitted"] == 0
    assert analysis["formulas_changed"] == 0
    assert analysis["row_gates_changed"] == 0
    assert analysis["uncertainty_floors_added"] == 0
    assert analysis["best_cell_selection_events"] == 0
    assert config["access_ceiling"]["raw_response_files_opened"] == 0
    assert config["interpretation_contract"]["published_error_metric_remains_primary"] is True


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("analysis_contract", "parameters_fitted", 1),
        ("analysis_contract", "best_cell_selection_events", 1),
        ("interpretation_contract", "mean_fractional_win_is_not_a_general_preference", False),
        ("access_ceiling", "raw_response_files_opened", 1),
        ("claim_boundary", "publication_ready", True),
    ],
)
def test_material_config_mutations_fail_closed(section: str, key: str, value: object) -> None:
    changed = copy.deepcopy(package.load_config())
    changed[section][key] = value
    with pytest.raises(package.PostscoreRobustnessError):
        package.validate_config(changed)


def test_predecessor_is_exact_and_analysis_does_not_recompute_raw_scores(monkeypatch) -> None:
    config = package.load_config()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("raw score recomputation must not run")

    monkeypatch.setattr(package.score, "build_receipt", forbidden)
    receipt = package._load_score_receipt(config)
    assert receipt["content_sha256"] == config["predecessor_score"]["receipt_content_sha256"]
    assert receipt["response_rows_parsed"] == 303


def test_aggregate_disagreement_is_reproduced_exactly() -> None:
    receipt = package.build_receipt(package.load_config())
    aggregates = receipt["primary_aggregates"]
    assert aggregates["mean_object_standardized_mse"]["winner"] == package.MOND
    assert aggregates["mean_object_fractional_mse"]["winner"] == package.RG
    assert aggregates["median_object_fractional_mse"]["winner"] == package.NEWTON
    assert aggregates["mean_object_standardized_mse"]["values"][package.RG] == pytest.approx(
        1604.710897149491
    )
    assert aggregates["mean_object_fractional_mse"]["values"][package.RG] == pytest.approx(
        0.07933096689831945
    )


def test_leave_one_out_exposes_holmberg_dependence_without_hiding_other_stability() -> None:
    receipt = package.build_receipt(package.load_config())
    rows = receipt["leave_one_object_out"]["mean_object_fractional_mse"]
    assert sum(row["winner"] == package.RG for row in rows) == 6
    without_holmberg = next(row for row in rows if row["dropped_object_id"] == "UGC04305")
    assert without_holmberg["winner"] == package.MOND
    assert without_holmberg["rg_rank"] == 3
    primary_rows = receipt["leave_one_object_out"]["mean_object_standardized_mse"]
    assert all(row["winner"] != package.RG for row in primary_rows)


def test_every_all_three_rg_win_is_holmberg_ii_and_sensitivity_is_retained() -> None:
    receipt = package.build_receipt(package.load_config())
    wins = receipt["rg_all_three_comparator_wins"]
    assert wins["cell_count"] == 7
    assert wins["holmberg_ii_cells"] == 7
    assert wins["holmberg_ii_total_cells"] == 9
    assert wins["non_holmberg_cells"] == 0
    assert wins["non_holmberg_total_cells"] == 15
    assert wins["by_object"] == {"UGC04305": 7}
    assert receipt["robustness_findings"]["all_rg_all_three_wins_are_holmberg_ii"] is True


def test_all_24_cells_and_directional_failures_are_retained() -> None:
    receipt = package.build_receipt(package.load_config())
    expected = {
        package.NEWTON: 6,
        package.RAR: 5,
        package.MOND: 6,
        package.RG: 7,
    }
    assert receipt["all_cell_winner_counts"]["standardized_mse"] == expected
    assert receipt["all_cell_winner_counts"]["fractional_mse"] == expected
    bias = {row["object_id"]: row for row in receipt["primary_rg_signed_bias"]}
    assert bias["DDO154"]["mean_signed_fractional_residual"] < -0.4
    assert bias["IC2574"]["mean_signed_fractional_residual"] < -0.2
    assert bias["NGC5055"]["mean_signed_fractional_residual"] > 0.3
    assert abs(bias["NGC6946"]["mean_signed_fractional_residual"]) < 0.01


def test_claims_remain_narrow_and_next_tests_are_independent() -> None:
    receipt = package.build_receipt(package.load_config())
    claims = receipt["claim_boundary"]
    assert claims["postscore_robustness_completed"] is True
    assert claims["holmberg_ii_specific_signal_observed"] is True
    assert claims["refracted_gravity_generalizes"] is False
    assert claims["unique_theory_established"] is False
    assert claims["publication_ready"] is False
    assert len(receipt["required_next_tests"]) == 4


def test_coherently_rehashed_receipt_forgery_fails_deterministic_rebuild() -> None:
    config = package.load_config()
    changed = copy.deepcopy(package.build_receipt(config))
    changed["robustness_findings"]["signal_classification"] = "PUBLICATION_READY"
    changed["content_sha256"] = package.content_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(package.PostscoreRobustnessError, match="rebuild differs"):
        package.validate_receipt(config, changed)


def test_atomic_no_clobber_is_exact(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert package._atomic_no_clobber(output, b"one") == "CREATED"
    assert package._atomic_no_clobber(output, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(package.PostscoreRobustnessError, match="differs"):
        package._atomic_no_clobber(output, b"two")
    assert output.read_bytes() == b"one"


def test_package_seals_match_current_files() -> None:
    if package._CONFIG_RAW_SHA256 == "0" * 64:
        pytest.skip("package pins not yet sealed")
    assert (
        package.file_sha256(package._repo_path(package.CONFIG_PATH)) == package._CONFIG_RAW_SHA256
    )
    assert package.content_sha256(package.load_config()) == package._CONFIG_CONTENT_SHA256
    assert (
        package.module_semantic_sha256(package._repo_path(package.MODULE_PATH))
        == package._MODULE_SEMANTIC_SHA256
    )
    assert package.file_sha256(package._repo_path(package.TEST_PATH)) == package._TEST_RAW_SHA256
