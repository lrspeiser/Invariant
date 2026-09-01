from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_holmberg_ii_things_2d_postscore_robustness_v1 as robustness,
)


def test_config_and_score_binding_are_valid() -> None:
    config = robustness.load_config(verify_package=False)
    robustness.validate_config(config)
    score = robustness._load_score(config)
    assert score["decision"] == config["score_binding"]["required_decision"]
    assert len(score["scores"]) == 18


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("score_binding", "required_decision", "RG_WON"),
        ("analysis_contract", "no_response_reopening", False),
        ("analysis_contract", "no_parameter_tuning", False),
        ("interpretation_contract", "primary_failure_must_remain_primary", False),
        ("claim_boundary", "primary_rg_replication_passed", True),
        ("claim_boundary", "unique_theory_established", True),
    ],
)
def test_material_mutations_fail(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(robustness.load_config(verify_package=False))
    config[section][key] = value
    with pytest.raises(robustness.HolmbergRobustnessError):
        robustness.validate_config(config)


def test_all_five_metrics_retain_exact_winner_partition() -> None:
    receipt = robustness.build_receipt(robustness.load_config(verify_package=False))
    for counts in receipt["winner_counts_by_metric_all_18_cells"].values():
        assert counts[robustness._RG] == 6
        assert counts[robustness._NEWTON] == 12
        assert counts["RAR_2016_ON_NEWTON_3D"] == 0
        assert counts["MOND_STANDARD_MU_ON_NEWTON_3D"] == 0


def test_inclination_partition_is_complete_and_directionally_exact() -> None:
    receipt = robustness.build_receipt(robustness.load_config(verify_package=False))
    groups = {row["inclination_deg"]: row for row in receipt["inclination_groups"]}
    assert groups[27.0]["rg_rmse_wins"] == 6
    assert groups[27.0]["newton_rmse_wins"] == 0
    assert groups[38.0]["rg_rmse_wins"] == 0
    assert groups[38.0]["newton_rmse_wins"] == 6
    assert groups[49.0]["rg_rmse_wins"] == 0
    assert groups[49.0]["newton_rmse_wins"] == 6


def test_descriptive_crossings_are_bounded_and_not_fits() -> None:
    receipt = robustness.build_receipt(robustness.load_config(verify_package=False))
    assert len(receipt["linear_descriptive_crossings"]) == 6
    assert receipt["crossing_range_deg"]["minimum"] == pytest.approx(29.221011911544043)
    assert receipt["crossing_range_deg"]["maximum"] == pytest.approx(34.76553056863165)
    assert receipt["crossing_range_deg"]["not_an_inclination_fit"] is True
    assert all(not row["is_fitted_inclination"] for row in receipt["linear_descriptive_crossings"])


def test_primary_failure_is_never_relabelled() -> None:
    receipt = robustness.build_receipt(robustness.load_config(verify_package=False))
    assert receipt["primary_result_retained"] == {
        "cell_score_id": "UGC04305__IRAC1_FIXED_ML0P6__I38P0__NATURAL",
        "winner": "NEWTON_3D_DST",
        "rg_replication_passed": False,
        "decision": "HOLMBERG_II_RG_SIGNAL_DOES_NOT_REPLICATE_ON_FIXED_PRIMARY_2D_CELL",
    }


def test_rg_beats_mond_and_rar_in_every_cell() -> None:
    receipt = robustness.build_receipt(robustness.load_config(verify_package=False))
    assert receipt["rg_rmse_beats_mond_and_rar_counts"] == {
        "RAR_2016_ON_NEWTON_3D": 18,
        "MOND_STANDARD_MU_ON_NEWTON_3D": 18,
    }


def test_no_response_or_prediction_loader_exists_in_module() -> None:
    source = robustness._repo_path(robustness.MODULE_PATH).read_text(encoding="utf-8")
    assert "astropy" not in source
    assert "numpy" not in source
    assert "getdata" not in source
    assert "np.load" not in source


def test_atomic_output_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert robustness._atomic_no_clobber(path, b"same") == "CREATED"
    assert robustness._atomic_no_clobber(path, b"same") == "EXISTING_IDENTICAL"
    with pytest.raises(robustness.HolmbergRobustnessError):
        robustness._atomic_no_clobber(path, b"different")


def test_package_seals_after_finalization() -> None:
    config = robustness.load_config()
    assert config["scientific_boundary"]["response_pixels_decoded"] == 0
