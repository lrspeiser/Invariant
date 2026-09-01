from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_holmberg_ii_things_2d_replication_preflight_v1 as package,
)


def test_config_freezes_full_two_resolution_three_geometry_four_law_replication() -> None:
    config = package.load_config()
    prediction = config["prediction_contract"]
    assert prediction["candidate_resolution_cells"] == 72
    assert prediction["source_cells"] == 9
    assert prediction["inclination_cells_deg"] == [27.0, 38.0, 49.0]
    assert prediction["response_resolutions"] == ["NATURAL", "ROBUST"]
    assert prediction["best_geometry_or_conversion_selection"] is False
    assert config["future_score_contract"]["all_18_source_resolution_cells_reported"] is True


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("prediction_contract", "source_cells", 1),
        ("prediction_contract", "best_geometry_or_conversion_selection", True),
        ("prediction_contract", "parameters_tuned", 1),
        ("future_score_contract", "model_specific_systemic_offset_primary", True),
        ("future_score_contract", "source_geometry_reselection", True),
        ("preparation_access_disclosure", "velocity_or_dispersion_pixels_decoded", 1),
        ("claim_boundary", "publication_ready", True),
    ],
)
def test_material_mutations_fail_closed(section: str, key: str, value: object) -> None:
    changed = copy.deepcopy(package.load_config())
    changed[section][key] = value
    with pytest.raises(package.ReplicationPreflightError):
        package.validate_config(changed)


def test_response_assets_and_headers_are_exact_without_data_array_access(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("scientific data array access is forbidden in preflight")

    monkeypatch.setattr(package.fits, "getdata", forbidden)
    receipt = package.build_receipt(package.load_config())
    assert receipt["response_asset_count"] == 4
    assert receipt["response_network_bytes"] == 16_951_680
    assert all(row["header"]["data_array_opened"] is False for row in receipt["response_assets"])
    assert {tuple(row["header"]["beam_deg"]) for row in receipt["response_assets"]} == {
        (0.0038158, 0.0034928, -40.24),
        (0.00193, 0.0016793, -32.8),
    }


def test_trigger_and_nine_source_cells_are_exact() -> None:
    receipt = package.build_receipt(package.load_config())
    assert receipt["trigger_receipt_content_sha256"] == (
        "c6715addbbc77ce24a0606f54af3be0d60bff47648242ee5cf5dc5ff59a89159"
    )
    assert len(receipt["holmberg_ii_source_cell_ids"]) == 9
    assert receipt["holmberg_ii_source_cell_ids"][0].startswith("IRAC1_FIXED_ML0P6")


def test_incidental_header_extrema_disclosure_is_retained_and_unused() -> None:
    disclosure = package.build_receipt(package.load_config())["preparation_access_disclosure"]
    assert disclosure["data_derived_header_extrema_values_incidentally_observed"] == 8
    assert disclosure["velocity_or_dispersion_pixels_decoded"] == 0
    assert disclosure["candidate_or_parameter_changes_after_header_access"] == 0
    assert disclosure["header_extrema_used_for_candidate_geometry_mask_or_score_design"] is False


def test_claims_and_source_matched_limitation_remain_narrow() -> None:
    receipt = package.build_receipt(package.load_config())
    claims = receipt["claim_boundary"]
    assert claims["response_assets_sealed"] is True
    assert claims["response_pixels_opened"] is False
    assert claims["holmberg_ii_signal_replicated"] is False
    assert claims["unique_theory_established"] is False
    assert claims["publication_ready"] is False
    assert "source-matched internal replication" in receipt["scientific_limitations"][0]


def test_coherently_rehashed_receipt_forgery_fails_rebuild() -> None:
    config = package.load_config()
    changed = copy.deepcopy(package.build_receipt(config))
    changed["claim_boundary"]["publication_ready"] = True
    changed["content_sha256"] = package.content_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(package.ReplicationPreflightError):
        package.validate_receipt(config, changed)


def test_atomic_no_clobber_is_exact(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert package._atomic_no_clobber(output, b"one") == "CREATED"
    assert package._atomic_no_clobber(output, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(package.ReplicationPreflightError, match="differs"):
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
