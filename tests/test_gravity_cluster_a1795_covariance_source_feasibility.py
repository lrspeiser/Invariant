from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_cluster_a1795_covariance_source_feasibility as feasibility,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return feasibility.load_config(ROOT)


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return feasibility.build_receipt(ROOT)


def test_audit_is_metadata_only_and_unauthorized(
    config: dict[str, object], receipt: dict[str, object]
) -> None:
    assert receipt["decision"] == feasibility.DECISION
    assert receipt["counts"] == {
        "metadata_source_references": 18,
        "metadata_network_calls_during_receipt_build": 0,
        "scientific_payload_rows_read": 0,
        "confirmation_rows_read": 0,
        "independent_rows_read": 0,
        "hidden_answers_read": 0,
        "large_files_downloaded": 0,
        "downloaded_bytes": 0,
        "scientific_scores_computed": 0,
        "paid_or_model_calls": 0,
    }
    assert config["authorization"] == receipt["authorization"]
    assert not receipt["authorization"]["authorized"]


def test_all_six_primary_a1795_observations_are_frozen(
    config: dict[str, object], receipt: dict[str, object]
) -> None:
    assert tuple(receipt["observation_ids"]) == feasibility.OBSERVATION_IDS
    assert receipt["xsa_release"] == {
        "pps_version": feasibility.PPS_VERSION,
        "sas_version": feasibility.SAS_VERSION,
        "public_observations": 6,
        "archive_size_fields_reported": 0,
    }
    observations = config["xmm_observations"]
    assert len(observations) == 6
    assert sum(len(row["science_exposures"]) for row in observations) == 20


def test_xmm_archive_endpoints_are_manifested_but_not_authorized(
    config: dict[str, object], receipt: dict[str, object]
) -> None:
    archives = config["xmm_source_packet"]["observation_archives"]
    assert len(archives) == 6
    assert all(row["odf_head_status"] == 200 for row in archives)
    assert all(row["pps_event_head_status"] == 200 for row in archives)
    assert all(row["odf_expected_bytes"] is None for row in archives)
    assert all(row["pps_event_expected_bytes"] is None for row in archives)
    assert all(not row["download_authorized"] for row in archives)
    assert receipt["public_packet"]["xmm_observation_archives_located"] == 6


def test_planck_public_products_have_observed_sizes_but_are_insufficient(
    config: dict[str, object], receipt: dict[str, object]
) -> None:
    products = config["planck_source_packet"]["public_products"]
    assert {row["file"]: row["head_content_length"] for row in products} == (
        feasibility.PLANCK_PRODUCTS
    )
    assert receipt["public_packet"]["planck_public_bytes_manifested"] == sum(
        feasibility.PLANCK_PRODUCTS.values()
    )
    assert not receipt["public_packet"][
        "exact_xcop_covariance_reconstruction_feasible"
    ]


def test_cp5_2_through_cp5_6_remain_blocked(receipt: dict[str, object]) -> None:
    assert receipt["cp5_statuses"] == feasibility.CP_STATUS
    assert set(receipt["missing_asset_ids"]) == feasibility.REQUIRED_MISSING_IDS
    assert not receipt["claims"]["CP5_2_through_CP5_6_complete"]
    assert receipt["claims"]["public_inputs_exist_for_a_new_a1795_reduction"]
    assert not receipt["claims"]["complete_bounded_source_packet_frozen"]
    assert not receipt["claims"]["publication_claim_supported"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["authorization"].__setitem__("authorized", True),
        lambda value: value["scope"].__setitem__("scientific_payload_rows_opened", 1),
        lambda value: value["xmm_observations"][0].__setitem__(
            "pps_version", "changed"
        ),
        lambda value: value["xmm_source_packet"]["observation_archives"][0].__setitem__(
            "download_authorized", True
        ),
        lambda value: value["planck_source_packet"]["public_products"][0].__setitem__(
            "head_content_length", 1
        ),
        lambda value: value["required_missing_assets"].pop(),
        lambda value: value["claim_boundary"].__setitem__(
            "joint_xray_sz_covariance_reconstructible", True
        ),
    ],
)
def test_manifest_tampering_fails_closed(
    config: dict[str, object], mutation: object
) -> None:
    changed = copy.deepcopy(config)
    mutation(changed)  # type: ignore[operator]
    with pytest.raises(
        feasibility.GravityClusterA1795CovarianceSourceFeasibilityError
    ):
        feasibility.validate_config(changed)


def test_implementation_tampering_is_rejected(
    config: dict[str, object], tmp_path: Path
) -> None:
    fake_root = tmp_path / "repo"
    config_path = fake_root / feasibility.CONFIG_PATH
    implementation_path = fake_root / feasibility.IMPLEMENTATION_PATH
    config_path.parent.mkdir(parents=True)
    implementation_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    implementation_path.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(
        feasibility.GravityClusterA1795CovarianceSourceFeasibilityError,
        match="implementation hash changed",
    ):
        feasibility.load_config(fake_root)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads(
        (ROOT / feasibility.OUTPUT_PATH).read_text(encoding="utf-8")
    )
    feasibility.validate_receipt(stored, ROOT)
    assert stored == feasibility.build_receipt(ROOT)
