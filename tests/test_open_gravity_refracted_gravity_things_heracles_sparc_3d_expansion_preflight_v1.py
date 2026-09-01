from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1 as preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict[str, object]:
    return json.loads((ROOT / preflight.CONFIG_PATH).read_text(encoding="utf-8"))


def test_exact_object_source_and_byte_inventory() -> None:
    config = preflight.load_config()
    rows = preflight.flatten_source_files(config)
    assert [row["object_id"] for row in config["object_source_contracts"]] == list(
        preflight._OBJECTS
    )
    assert len(rows) == 77
    assert len({row["url"] for row in rows}) == 77
    assert sum(row["bytes"] for row in rows) == 351_461_418


def test_five_s4g_and_seven_sings_branches_are_honestly_distinct() -> None:
    config = preflight.load_config()
    branches = [row["stellar_branch"] for row in config["object_source_contracts"]]
    assert sum(row["survey"] == "S4G_P5" for row in branches) == 5
    assert sum(row["survey"] == "SINGS_IRAC1" for row in branches) == 7
    assert all(
        row["disposition"] == "DATA_AND_PAPER_ADMITTED_BUILDER_READY"
        for row in branches
        if row["survey"] == "S4G_P5"
    )
    assert all(
        row["disposition"] == "DATA_AND_PAPER_ADMITTED_CONVERSION_REQUIRED"
        for row in branches
        if row["survey"] == "SINGS_IRAC1"
    )


def test_every_object_has_stellar_hi_and_molecular_sources() -> None:
    config = preflight.load_config()
    for row in config["object_source_contracts"]:
        assert len(row["stellar_branch"]["files"]) in (2, 3)
        assert {item["role"] for item in row["hi_files"]} == {
            "HI_MOM0_NATURAL",
            "HI_MOM0_ROBUST",
        }
        assert {item["role"] for item in row["molecular_files"]} == {
            "CO21_MOM0",
            "CO21_EMOM0",
        }


def test_primary_measurement_and_theory_papers_are_bound() -> None:
    config = preflight.load_config()
    assert {row["paper_id"] for row in config["primary_papers"]} == preflight._PAPER_IDS
    assert len(config["benchmark_contract"]["required_before_real_scoring"]) == 7
    assert config["benchmark_contract"]["paper_only_is_not_data_validation"] is True
    assert (
        config["benchmark_contract"][
            "one_dimensional_response_cannot_validate_three_dimensional_solver"
        ]
        is True
    )


def test_no_response_or_cube_endpoint_is_admitted() -> None:
    rows = preflight.flatten_source_files(preflight.load_config())
    forbidden = ("MOM1", "MOM2", "VROT", "VELOCITY", ".HANS.")
    for row in rows:
        assert not any(token in row["url"].upper() for token in forbidden)


def test_selection_is_name_only_and_response_free() -> None:
    config = preflight.load_config()
    selection = config["selection_contract"]
    response = config["response_boundary"]
    assert selection["candidate_count"] == 18
    assert selection["admitted_count"] == 12
    assert len(selection["excluded_not_in_local_sparc"]) == 6
    assert selection["selection_used_response_values"] is False
    assert response["sparc_membership_names_checked"] == 175
    assert response["sparc_velocity_values_opened"] == 0
    assert response["scientific_image_pixels_read"] == 0
    assert response["rotation_response_rows_read"] == 0
    assert response["scores_computed"] == 0


def test_receipt_exactly_rebuilds_and_retains_claim_ceiling() -> None:
    config = preflight.load_config()
    receipt = preflight.build_receipt(config)
    preflight.validate_receipt(receipt, config)
    assert receipt["admission"]["object_count"] == 12
    assert receipt["admission"]["s4g_builder_ready_after_download"] == 5
    assert receipt["admission"]["sings_conversion_builder_required"] == 7
    assert receipt["admission"]["source_blocked"] == 0
    assert receipt["sources"]["selected_endpoint_count"] == 77
    assert receipt["sources"]["network_byte_ceiling"] == 351_461_418
    assert receipt["claims"]["source_payloads_downloaded"] is False
    assert receipt["claims"]["three_dimensional_fields_built"] is False
    assert receipt["claims"]["rotation_curves_scored"] is False
    assert receipt["claims"]["refracted_gravity_supported"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "PUBLICATION_READY"),
        (("selection_contract", "admitted_count"), 13),
        (("selection_contract", "selection_used_response_values"), True),
        (("future_acquisition", "payload_downloads_authorized_by_this_packet"), True),
        (("benchmark_contract", "paper_only_is_not_data_validation"), False),
        (("claims", "three_dimensional_fields_built"), True),
        (("claims", "discovery_or_publication_claim"), True),
        (("response_boundary", "scores_computed"), 1),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    mutated = copy.deepcopy(_raw_config())
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(preflight.ExpansionPreflightError):
        preflight.validate_config(mutated)


def test_source_url_and_byte_mutations_fail_closed() -> None:
    for field, value in (("bytes", 1), ("url", "https://example.invalid/source.fits")):
        mutated = copy.deepcopy(_raw_config())
        mutated["object_source_contracts"][0]["hi_files"][0][field] = value
        with pytest.raises(preflight.ExpansionPreflightError):
            preflight.validate_config(mutated)


def test_receipt_mutations_fail_closed() -> None:
    config = preflight.load_config()
    receipt = preflight.build_receipt(config)
    receipt["checks"]["no_scientific_payload_or_response_opened"] = False
    receipt["content_sha256"] = preflight.content_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    with pytest.raises(preflight.ExpansionPreflightError):
        preflight.validate_receipt(receipt, config)


def test_zero_access_disclosure_does_not_claim_exact_discovery_request_count() -> None:
    access = preflight.load_config()["access_state"]
    assert access["selected_endpoint_head_observations"] == 77
    assert access["discovery_metadata_request_count"] is None
    assert access["discovery_metadata_request_count_not_claimed_exact"] is True
    assert access["scientific_payload_bytes_observed"] == 0
    assert access["source_payload_files_downloaded"] == 0
    assert access["response_rows_read"] == 0


def test_cli_has_no_caller_selected_input_or_output_path() -> None:
    parser = preflight._parser()
    args = parser.parse_args(["status"])
    assert vars(args) == {"command": "status"}
    with pytest.raises(SystemExit):
        parser.parse_args(["status", "--output", "C:/forged.json"])
