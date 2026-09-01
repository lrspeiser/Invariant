from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_rg_things_six_external_2d_response_blind_predictions_v1 as packet,
)


def test_config_and_predecessor_inventory_are_valid() -> None:
    config = packet.load_config(verify_package=False)
    packet.validate_config(config)
    predecessors = packet._load_predecessors(config)
    assert set(predecessors) == {
        "SIX_EXTERNAL_2D_REPLICATION_PREFLIGHT",
        "SEVEN_HOLDOUT_SOURCE_BUILDER",
        "HOLMBERG_RESPONSE_BLIND_2D_METHOD",
        "AUDITED_3D_DST_PCG_MECHANICS",
        "PUBLISHED_CONTROL_FORMULAS",
        "AUDITED_2D_WCS_BEAM_PROJECTION",
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("candidate_contract", "a0_m_s2", 9e-11),
        ("candidate_contract", "response_parameter_fitting", True),
        ("source_contract", "model_lift_label", "FULL_3D"),
        ("projection_contract", "response_values_used", True),
        ("operator_contract", "maximum_local_relative_difference", 1.0),
        ("claim_boundary", "publication_ready", True),
    ],
)
def test_material_config_mutations_fail(section: str, key: str, value: object) -> None:
    config = copy.deepcopy(packet.load_config(verify_package=False))
    config[section][key] = value
    with pytest.raises(packet.SixGalaxyPredictionError):
        packet.validate_config(config)


def test_exact_source_inventory_retains_all_three_failures() -> None:
    config = packet.load_config(verify_package=False)
    predecessors = packet._load_predecessors(config)
    built, failed = packet._source_inventory(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
    assert len(built) == 15
    assert len(failed) == 3
    assert [row["object_id"] for row in failed] == ["IC2574", "DDO154", "NGC6946"]
    assert all("failure_evidence" in row for row in failed)


def test_preflight_metadata_is_exact_and_pixel_free() -> None:
    config = packet.load_config(verify_package=False)
    predecessors = packet._load_predecessors(config)
    preflight = predecessors["SIX_EXTERNAL_2D_REPLICATION_PREFLIGHT"]
    packet._validate_preflight_metadata(config, preflight)
    assert preflight["inventory_counts"]["response_pixels_decoded"] == 0
    assert len(preflight["response_assets"]) == 24


def test_prediction_build_constructs_wcs_without_opening_response_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = packet.load_config(verify_package=False)
    predecessors = packet._load_predecessors(config)
    source_receipt = predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    source_cell = packet._source_inventory(config, source_receipt)[0][0]
    observed: dict[str, object] = {}

    def fake_build(
        cell_config: dict[str, object],
        passed_cell: dict[str, object],
        *_args: object,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        observed["contract"] = cell_config["response_header_contract"]
        observed["cell"] = passed_cell
        return {}, {}

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("response FITS loader was called")

    monkeypatch.setattr(packet.method, "_build_cell_arrays", fake_build)
    monkeypatch.setattr(packet.fits, "getheader", forbidden)
    monkeypatch.setattr(packet.fits, "getdata", forbidden)
    packet._build_cell_arrays(
        config,
        source_cell,
        source_receipt,
        source_receipt["_config"],
    )
    assert observed["cell"] is source_cell
    assert observed["contract"] == config["response_header_contracts"]["NGC2841"]


def test_each_frozen_header_constructs_a_valid_1024_wcs() -> None:
    config = packet.load_config(verify_package=False)
    for contract in config["response_header_contracts"].values():
        header = packet._header_from_contract(contract)
        packet.method._validate_header(header, contract)
        assert header["NAXIS1"] == 1024
        assert header["NAXIS2"] == 1024


def test_atomic_no_clobber_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "sealed.bin"
    assert packet._atomic_no_clobber(path, b"same") == "CREATED"
    assert packet._atomic_no_clobber(path, b"same") == "EXISTING_IDENTICAL"
    with pytest.raises(packet.SixGalaxyPredictionError):
        packet._atomic_no_clobber(path, b"different")


def test_module_contains_no_response_pixel_decoder() -> None:
    source = packet._repo_path(packet.MODULE_PATH).read_text(encoding="utf-8")
    assert "fits.getdata" not in source
    assert ".data[" not in source
    assert "fits.getheader" not in source


def test_public_build_api_has_no_fit_or_selection_argument() -> None:
    assert list(inspect.signature(packet.write_cell).parameters) == ["cell_id"]
    assert "response" not in inspect.signature(packet._build_cell_arrays).parameters


def test_config_is_valid_json_and_counts_exactly_120_predictions() -> None:
    config = json.loads(packet._repo_path(packet.CONFIG_PATH).read_text(encoding="utf-8"))
    assert config["execution_contract"]["candidate_resolution_predictions"] == 120
    assert config["execution_contract"]["private_array_files"] == 195
    assert len(config["private_output"]["array_roles"]) == 13


def test_package_seals_after_finalization() -> None:
    config = packet.load_config()
    assert config["source_contract"]["source_cells"] == 15
