from __future__ import annotations

import copy
import math

import pytest

from sigma_theory_compiler import open_gravity_rg_sings_stellar_conversion_contract_v1 as contract


def test_source_paper_and_known_answer_admission_is_frozen() -> None:
    config = contract.load_config(verify_package=False)
    assert len(config["admission"]["primary_papers"]) == 4
    assert len(config["admission"]["independent_benchmarks"]) == 3
    assert config["admission"]["response_data_used"] is False


def test_exact_overlap_and_holdout_objects() -> None:
    receipt = contract.build_receipt()
    assert [row["object_id"] for row in receipt["overlap_records"]] == [
        "NGC2976",
        "NGC3198",
        "NGC3521",
    ]
    assert receipt["holdout_admission"]["object_order"] == [
        "UGC04305",
        "NGC2841",
        "IC2574",
        "DDO154",
        "NGC5055",
        "NGC6946",
        "NGC7331",
    ]
    assert receipt["holdout_admission"]["object_count"] == 7
    assert receipt["holdout_admission"]["irac1_irac2_file_count"] == 28


def test_three_conversion_cells_are_retained_without_primary_selection() -> None:
    config = contract.load_config(verify_package=False)
    assert [row["cell_id"] for row in config["conversion_cells"]] == [
        "IRAC1_FIXED_ML0P6",
        "IRAC1_GLOBAL_COLOR_ML",
        "IRAC1_IRAC2_FASTICA36",
    ]
    score = config["response_score_contract"]
    assert score["primary_cell"] is None
    assert score["all_three_cells_scored"] is True
    assert score["cell_selected_by_response"] is False
    assert score["minority_cells_retained"] is True


def test_raw_sings_tracks_stellar_plus_nonstellar_not_clean_stellar() -> None:
    summary = contract.build_receipt()["benchmark_summary"]
    assert summary["raw_irac1_tracks_stellar_plus_nonstellar_better_than_stellar_only"] is True
    assert summary["raw_irac1_max_abs_integrated_error_vs_stellar_plus_nonstellar"] < 0.03
    assert summary["single_conversion_is_s4g_equivalent"] is False


def test_source_uncertainty_is_material_and_retained() -> None:
    summary = contract.build_receipt()["benchmark_summary"]
    assert summary["fixed_ml0p6_max_abs_mass_error_vs_clean_fixed_ml"] > 0.7
    assert summary["global_color_ml_max_abs_mass_error_vs_clean_color_ml"] > 0.6
    assert summary["fastica36_max_abs_integrated_stellar_error_vs_s4g"] > 0.4
    assert summary["fastica36_min_stellar_pearson_r"] > 0.7


def test_published_color_and_ml_formulae() -> None:
    color = contract.irac_color(2.0, 1.0)
    assert math.isfinite(color)
    ml = contract.effective_ml(color)
    assert ml > 0.0
    assert math.isclose(contract.effective_ml(0.0), 10.0**-0.336)
    with pytest.raises(contract.StellarConversionContractError):
        contract.irac_color(0.0, 1.0)


def test_two_component_known_answer_reconstruction() -> None:
    result = contract.reconstruct_two_component(3.0, 2.0, -0.1, 0.6)
    assert result["f36"] == 5.0
    assert result["star_f36"] == 3.0
    assert result["dust_f36"] == 2.0
    assert result["f45"] > 0.0
    with pytest.raises(contract.StellarConversionContractError):
        contract.reconstruct_two_component(-1.0, 2.0, -0.1, 0.6)


def test_zero_response_and_no_gravity_tuning() -> None:
    config = contract.load_config(verify_package=False)
    assert set(config["access_scope"].values()) == {0}
    assert config["response_score_contract"]["new_rg_parameters_fitted"] is False
    claims = config["claim_boundary"]
    assert claims["gravity_result_established"] is False
    assert claims["publication_ready"] is False


def test_config_mutation_fails_closed() -> None:
    config = contract.load_config(verify_package=False)
    forged = copy.deepcopy(config)
    forged["response_score_contract"]["primary_cell"] = "IRAC1_IRAC2_FASTICA36"
    with pytest.raises(contract.StellarConversionContractError):
        contract.validate_config(forged)


def test_receipt_mutation_fails_closed() -> None:
    receipt = contract.build_receipt()
    forged = copy.deepcopy(receipt)
    forged["claim_boundary"]["publication_ready"] = True
    forged["content_sha256"] = contract.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(contract.StellarConversionContractError):
        contract.validate_receipt(forged)


def test_deterministic_rebuild_and_self_hash() -> None:
    first = contract.build_receipt()
    second = contract.build_receipt()
    assert first == second
    assert first["content_sha256"] == contract.content_sha256(
        {key: value for key, value in first.items() if key != "content_sha256"}
    )
    contract.validate_receipt(first)
