from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_things_six_external_2d_replication_preflight_v1 as preflight,
)


def test_exact_inclination_stratified_sample_and_assets() -> None:
    config = preflight.load_config(verify_package=False)
    assert [row["object_id"] for row in config["objects"]] == list(preflight._OBJECTS)
    assert len(config["response_assets"]) == 24
    assert {
        (row["object_id"], row["resolution"], row["observable"])
        for row in config["response_assets"]
    } == {
        (object_id, resolution, observable)
        for object_id in preflight._OBJECTS
        for resolution in preflight._RESOLUTIONS
        for observable in preflight._OBSERVABLES
    }


def test_all_assets_hash_and_headers_validate_without_pixels() -> None:
    receipt = preflight.build_receipt(preflight.load_config(verify_package=False))
    assert receipt["status"] == "PASS_OPAQUE_ACQUISITION_AND_HEADER_PREFLIGHT"
    assert receipt["decision"] == "READY_TO_BUILD_ALL_SIX_GALAXY_RESPONSE_BLIND_PREDICTIONS"
    assert receipt["inventory_counts"]["assets"] == 24
    assert receipt["inventory_counts"]["opaque_response_bytes"] == 102101760
    assert receipt["inventory_counts"]["response_pixels_decoded"] == 0
    assert all(row["pixel_values_decoded"] == 0 for row in receipt["response_assets"])


def test_every_header_has_two_spatial_axes_and_beam_bound() -> None:
    receipt = preflight.build_receipt(preflight.load_config(verify_package=False))
    for row in receipt["response_assets"]:
        assert len(row["shape"]) == 2
        assert all(value > 0 for value in row["shape"])
        assert row["wcs_ctype"][0].startswith("RA---")
        assert row["wcs_ctype"][1].startswith("DEC--")
        assert row["beam_deg"][0] > 0.0
        assert row["beam_deg"][1] > 0.0


def test_predecessors_are_exactly_bound() -> None:
    config = preflight.load_config(verify_package=False)
    receipts = preflight.validate_predecessors(config)
    assert set(receipts) == {
        "SEVEN_OBJECT_MODEL_LIFTED_SOURCE_BUILDER",
        "NEWTON_AQUAL_QUMOND_BASELINES",
        "THINGS_2D_PROJECTION_BENCHMARK",
        "HOLMBERG_II_RESPONSE_BLIND_2D_METHOD",
    }


def test_claims_and_access_remain_response_blind() -> None:
    config = preflight.load_config(verify_package=False)
    assert config["claim_boundary"]["public_response_assets_sealed"] is True
    assert config["claim_boundary"]["response_blind_predictions_complete"] is False
    assert config["claim_boundary"]["response_scoring_complete"] is False
    assert config["acquisition_accounting"]["response_pixels_decoded"] == 0
    assert (
        config["prediction_contract"]["all_predictions_must_be_sealed_before_response_pixels_open"]
        is True
    )


def test_material_mutations_fail_closed() -> None:
    config = preflight.load_config(verify_package=False)
    mutations = []
    changed = copy.deepcopy(config)
    changed["claim_boundary"]["publication_ready"] = True
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["acquisition_accounting"]["response_pixels_decoded"] = 1
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["prediction_contract"]["response_parameter_fitting"] = True
    mutations.append(changed)
    changed = copy.deepcopy(config)
    changed["objects"][0]["inclination_deg"] = 20.0
    mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(preflight.ReplicationPreflightError):
            preflight.validate_config(mutation)


def test_asset_hash_mutation_fails() -> None:
    config = preflight.load_config(verify_package=False)
    changed = copy.deepcopy(config)
    changed["response_assets"][0]["sha256"] = "0" * 64
    with pytest.raises(preflight.ReplicationPreflightError):
        preflight.build_receipt(changed)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert preflight._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert preflight._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(preflight.ReplicationPreflightError):
        preflight._atomic_no_clobber(path, b"two\n")


def test_stored_receipt_when_present() -> None:
    path = preflight._repo_path(preflight.OUTPUT_PATH)
    if path.exists():
        assert preflight.check_receipt() == "VALID"
