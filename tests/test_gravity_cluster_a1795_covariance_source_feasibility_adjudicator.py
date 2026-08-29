from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_cluster_a1795_covariance_source_feasibility_adjudicator as adjudicator,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / adjudicator.CONFIG_PATH
RECEIPT = ROOT / adjudicator.RECEIPT_PATH


def load_contract() -> dict[str, object]:
    return adjudicator.load_config(CONFIG, adjudicator.file_sha256(CONFIG))


def original_config() -> dict[str, object]:
    path = ROOT / adjudicator.EXPECTED_ORIGINAL_ARTIFACTS["config"]["path"]
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_binds_all_original_and_item59_artifacts_exactly() -> None:
    config = load_contract()
    assert config["original_artifacts"] == adjudicator.EXPECTED_ORIGINAL_ARTIFACTS
    assert config["item59_bindings"] == adjudicator.EXPECTED_ITEM59_BINDINGS
    for group in ("original_artifacts", "item59_bindings"):
        for binding in config[group].values():
            assert adjudicator.file_sha256(ROOT / binding["path"]) == binding["file_sha256"]


def test_original_config_and_receipt_reconstruct_strict_block() -> None:
    config = load_contract()
    result = adjudicator.adjudicate(config)
    assert result["strict_verifier_passed"] is True
    assert result["nested_factual_sections_exact"] is True
    assert result["item59_archive_provenance_exact"] is True
    assert result["observation_ids"] == list(adjudicator.feasibility.OBSERVATION_IDS)
    assert result["planck_public_bytes_manifested"] == 13_314_915_231
    assert result["complete_public_covariance_source_packet"] is False
    assert result["CP5_2_through_CP5_6_complete"] is False
    assert result["decision"] == adjudicator.DECISION


@pytest.mark.parametrize(
    "mutation",
    [
        "source_url",
        "xmm_license",
        "planck_license",
        "ra",
        "duration",
        "role",
        "exposures",
        "component_dispositions",
        "archive_sha",
        "archive_bytes",
        "planck_product_url",
        "planck_product_bytes",
        "access_count",
        "decision",
    ],
)
def test_every_previously_unsealed_fact_class_fails_closed(mutation: str) -> None:
    changed = copy.deepcopy(original_config())
    if mutation == "source_url":
        changed["source_references"][0]["url"] = "https://invalid.example/forged"
    elif mutation == "xmm_license":
        changed["xmm_source_packet"]["license"]["status"] = "FORGED"
    elif mutation == "planck_license":
        changed["planck_source_packet"]["license"]["public_access_verified"] = False
    elif mutation == "ra":
        changed["xmm_observations"][0]["ra_deg"] = 0.0
    elif mutation == "duration":
        changed["xmm_observations"][0]["duration_s"] = 1
    elif mutation == "role":
        changed["xmm_observations"][5]["mosaic_role"] = "CENTER"
    elif mutation == "exposures":
        changed["xmm_observations"][0]["science_exposures"] = []
    elif mutation == "component_dispositions":
        changed["xmm_source_packet"]["component_dispositions"] = []
    elif mutation == "archive_sha":
        changed["planck_source_packet"]["xcop_basic_archive"]["observed_archive_sha256"] = "0" * 64
    elif mutation == "archive_bytes":
        changed["planck_source_packet"]["xcop_basic_archive"]["observed_archive_bytes"] = 1
    elif mutation == "planck_product_url":
        changed["planck_source_packet"]["public_products"][0]["url"] = (
            "https://invalid.example/forged"
        )
    elif mutation == "planck_product_bytes":
        changed["planck_source_packet"]["public_products"][0]["head_content_length"] += 1
    elif mutation == "access_count":
        changed["scope"]["scientific_payload_rows_opened"] = 1
    else:
        changed["cp5_adjudication"]["decision"] = "FORGED_PASS"
    with pytest.raises(RuntimeError, match="changed|count"):
        adjudicator.validate_frozen_feasibility_config(changed)


def test_nested_key_addition_and_removal_fail_closed() -> None:
    added = copy.deepcopy(original_config())
    added["source_references"][0]["forged"] = True
    with pytest.raises(RuntimeError, match="keys changed"):
        adjudicator.validate_frozen_feasibility_config(added)
    removed = copy.deepcopy(original_config())
    del removed["xmm_observations"][0]["target"]
    with pytest.raises(RuntimeError, match="keys changed"):
        adjudicator.validate_frozen_feasibility_config(removed)


def test_original_config_and_receipt_cannot_be_cross_swapped() -> None:
    config_binding = adjudicator.EXPECTED_ORIGINAL_ARTIFACTS["config"]
    receipt_path = ROOT / adjudicator.EXPECTED_ORIGINAL_ARTIFACTS["receipt"]["path"]
    with pytest.raises(RuntimeError, match="path changed or swapped"):
        adjudicator.read_bound_json(receipt_path, config_binding, "cross-swapped config")


def test_item59_source_and_preflight_cannot_be_cross_swapped() -> None:
    source_binding = adjudicator.EXPECTED_ITEM59_BINDINGS["source_receipt"]
    preflight_path = ROOT / adjudicator.EXPECTED_ITEM59_BINDINGS["preflight_manifest"]["path"]
    with pytest.raises(RuntimeError, match="path changed or swapped"):
        adjudicator.read_bound_json(preflight_path, source_binding, "cross-swapped Item59")


def test_factual_qualifications_are_explicit_and_restrained() -> None:
    qualifications = adjudicator.FACTUAL_QUALIFICATIONS
    assert "both basic and high-level" in qualifications["xcop_release_page"]["corrected_fact"]
    assert qualifications["xmm_license"]["verified_scope"] == (
        "proposal_074441_only_from_the_bound_primary_rights_page"
    )
    assert qualifications["xcop_archive_provenance"]["archive_bytes"] == 315_080_566
    assert (
        qualifications["xcop_archive_provenance"]["current_adjudicator_downloaded_archive"] is False
    )
    assert qualifications["planck_release_label"]["product_label"] == "R2.02"
    assert qualifications["planck_release_label"]["storage_path_contains"] == ("/Planck/release_3/")


def test_receipt_exactly_reconstructs_and_preserves_block() -> None:
    checked = adjudicator.check_receipt(CONFIG, adjudicator.file_sha256(CONFIG), RECEIPT)
    assert checked["valid"] is True
    assert checked["strict_verifier_passed"] is True
    assert checked["decision"] == adjudicator.DECISION
    assert checked["complete_public_covariance_source_packet"] is False
    assert checked["CP5_2_through_CP5_6_complete"] is False
    assert checked["downloads_authorized"] is False
    assert checked["payload_access_authorized"] is False
    assert checked["scientific_claim_allowed"] is False


def test_forged_receipt_and_config_swap_are_rejected() -> None:
    config = load_contract()
    expected = adjudicator.expected_receipt(config)
    forged = copy.deepcopy(expected)
    forged["decision"] = "FORGED_PASS"
    forged["content_sha256"] = adjudicator.canonical_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(RuntimeError, match="does not exactly reconstruct"):
        adjudicator.require_exact_receipt(forged, expected, "forged receipt")
    with pytest.raises(RuntimeError, match="config path changed"):
        adjudicator.load_config(RECEIPT, adjudicator.file_sha256(RECEIPT))


def test_zero_access_boundary_and_no_direct_work_artifact_binding() -> None:
    assert all(value == 0 for value in adjudicator.DATA_BOUNDARY.values())
    for path in (CONFIG, RECEIPT):
        text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
        assert "work/" not in text
