from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_things_matched_pair_2d_kinematics_source_v1 as source,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return source.load_config(verify_package=False)


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return source.build_receipt(config)


def test_pair_was_selected_before_velocity_pixels(config: dict) -> None:
    selection = config["selection_binding"]
    assert selection["selected_pair"] == ["NGC2976", "NGC4214"]
    assert selection["selection_used_things_velocity_pixels"] is False
    diagnostic = source._validate_selection(config)
    assert diagnostic["matched_pair"]["opposite_support_direction"] is True


def test_real_data_and_primary_papers_are_bound(config: dict) -> None:
    assert len(config["primary_sources"]) == 2
    assert {row["id"] for row in config["primary_sources"]} == {
        "THINGS_SURVEY_DATA_RELEASE",
        "THINGS_HIGH_RESOLUTION_KINEMATICS",
    }
    assert len(config["files"]) == 4
    assert sum(row["bytes"] for row in config["files"]) == 16974720


def test_source_files_and_header_schemas_match(config: dict, receipt: dict) -> None:
    assert receipt["file_count"] == 4
    assert receipt["byte_count"] == 16974720
    for row in receipt["files"]:
        assert row["shape"] == [1, 1, 1024, 1024]
        assert row["bitpix"] == -32
        assert row["bunit"] == "METR/SEC"
        assert row["wcs"]["ctype1"] == "RA---SIN"
        assert row["wcs"]["ctype2"] == "DEC--SIN"
        assert row["beam"]["major_deg"] > row["beam"]["minor_deg"] > 0.0
        assert row["data_values_read"] == 0


def test_build_never_calls_pixel_data_loader(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("pixel data loader called")

    monkeypatch.setattr(source.fits, "getdata", forbidden)
    built = source.build_receipt(config)
    assert built["access_accounting"]["velocity_pixel_values_decoded"] == 0
    assert built["access_accounting"]["dispersion_pixel_values_decoded"] == 0


def test_solver_stays_blocked_until_benchmarks(config: dict, receipt: dict) -> None:
    gate = receipt["future_builder_gate"]
    assert gate["real_public_source_data_present"] is True
    assert gate["primary_measurement_and_method_papers_present"] is True
    assert gate["independent_solver_benchmarks_passed"] is False
    assert gate["response_scoring_allowed"] is False
    assert gate["general_3d_claim_allowed"] is False
    assert len(gate["required_before_pixel_decode"]) == 6


def test_access_accounting_is_exact(config: dict, receipt: dict) -> None:
    access = receipt["access_accounting"]
    assert access == config["access_accounting"]
    assert access["head_calls"] == 4
    assert access["get_calls"] == 4
    assert access["network_bytes"] == 16974720
    assert access["scores_computed"] == 0
    assert access["model_calls"] == 0
    assert access["paid_calls"] == 0


def test_receipt_is_deterministic(config: dict, receipt: dict) -> None:
    assert source.build_receipt(config) == receipt
    assert receipt["content_sha256"] == source.content_sha256({**receipt, "content_sha256": ""})


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "READY_TO_SCORE"),
        (("selection_binding", "selected_pair"), ["NGC2903", "NGC3198"]),
        (("future_builder_gate", "independent_solver_benchmarks_passed"), True),
        (("future_builder_gate", "response_scoring_allowed"), True),
        (("access_accounting", "velocity_pixel_values_decoded"), 1),
        (("claim_boundary", "publication_ready"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(source.KinematicsSourceError):
            source.validate_config(mutated)


def test_receipt_mutation_fails(config: dict, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["scientific_fit_tested"] = True
    mutated["content_sha256"] = source.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(source.KinematicsSourceError):
        source.validate_receipt_payload(config, mutated)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert source._atomic_no_clobber(output, b"one\n") == "CREATED"
    assert source._atomic_no_clobber(output, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(source.KinematicsSourceError):
        source._atomic_no_clobber(output, b"two\n")
