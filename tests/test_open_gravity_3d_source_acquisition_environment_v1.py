from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_3d_source_acquisition_environment_v1 as acquisition


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = acquisition.load_config()
    return config, acquisition.run_suite(config)


def test_config_and_exact_predecessor_chain(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    acquisition.validate_config(config)
    assert [row["role"] for row in config["bindings"]] == [
        "SOURCE_AVAILABILITY_V2",
        "FULL3D_SOURCE_GEOMETRY",
    ]
    assert all(len(row["commit"]) == 40 for row in config["bindings"])


def test_exact_object_ledger_and_hash_root(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    ledger = suite["object_ledger"]
    assert len(ledger) == 147
    assert sum(row["domain"] == "SPARC" for row in ledger) == 139
    assert sum(row["domain"] == "XCOP" for row in ledger) == 8
    assert len({row["object_id"] for row in ledger}) == 147
    assert len({row["row_sha256"] for row in ledger}) == 147
    assert suite["object_ledger_stream_sha256"] == (
        "16f899907919ae08ac03c4fe0cf1261728533aa057de0d7866fbf5f22b957810"
    )


def test_all_twelve_metadata_gates_pass(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert list(suite["gates"]) == config["required_gates"]
    assert suite["passed"] == 12
    assert suite["failed"] == 0
    assert all(row["passed"] is True for row in suite["gates"].values())


def test_no_current_object_is_full_3d_ready(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    assert suite["full_3d_source_ready_objects"] == 0
    assert all(
        row["full_3d_status"] == "SOURCE_BLOCKED_MISSING_DEPTH"
        for row in suite["object_ledger"]
        if row["domain"] == "SPARC"
    )
    assert all(
        row["full_3d_status"] == "SPHERICAL_ONLY"
        for row in suite["object_ledger"]
        if row["domain"] == "XCOP"
    )


def test_xcop_stellar_profile_split_is_retained(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    clusters = [row for row in suite["object_ledger"] if row["domain"] == "XCOP"]
    assert sum(row["stellar_profile_1d_available"] for row in clusters) == 5
    assert sum(not row["stellar_profile_1d_available"] for row in clusters) == 3
    assert {row["object_id"] for row in clusters if not row["stellar_profile_1d_available"]} == {
        "A1644",
        "A2255",
        "A3266",
    }


def test_galaxy_and_cluster_source_product_roles_are_complete(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert len(config["galaxy_products"]) == 6
    assert len(config["cluster_products"]) == 5
    assert suite["gates"]["GALAXY_PRODUCT_ROLES_COMPLETE"]["passed"] is True
    assert suite["gates"]["CLUSTER_PRODUCT_ROLES_COMPLETE"]["passed"] is True


def test_forbidden_substitutes_prevent_response_leakage(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    forbidden = config["forbidden_source_substitutes"]
    assert any("rotation residual" in value for value in forbidden)
    assert any("hydrostatic total mass" in value for value in forbidden)
    assert any("lensing mass" in value for value in forbidden)
    assert suite["gates"]["SOURCE_RESPONSE_SEPARATION"]["passed"] is True


def test_environment_history_and_matched_design_are_pre_response(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    contract = config["environment_history_contract"]
    assert contract["response_derived_environment_forbidden"] is True
    assert len(contract["labels_required_before_response"]) == 6
    assert len(contract["matched_covariates"]) == 7
    assert len(contract["negative_controls"]) == 7
    assert (
        suite["gates"]["HISTORY_REQUIRES_REAL_HISTORY_OR_DECLARED_SIMULATION"]["metrics"][
            "currently_history_ready"
        ]
        == 0
    )


def test_acquisition_and_campaign_authority_are_withheld(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert not any(config["authority"].values())
    assert suite["campaign_ready"] is False
    assert suite["gates"]["ACQUISITION_AUTHORITY_WITHHELD"]["passed"] is True
    assert suite["gates"]["CAMPAIGN_FREEZE_WITHHELD"]["passed"] is True


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "bindings",
        "galaxy_products",
        "cluster_products",
        "forbidden_source_substitutes",
        "environment_history_contract",
        "upgrade_rules",
        "required_gates",
        "authority",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple[dict, dict], section: str) -> None:
    config, _suite = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(acquisition.AcquisitionLedgerError, match="config semantics changed"):
        acquisition.validate_config(changed)


def test_noncanonical_receipt_path_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(acquisition, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(acquisition, "_read_json", forbidden)
    with pytest.raises(acquisition.AcquisitionLedgerError, match="output path changed"):
        acquisition.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple[dict, dict]) -> None:
    _config, _suite = packet
    receipt = acquisition.build_receipt()
    acquisition.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["suite"]["campaign_ready"] = True
    forged["content_sha256"] = acquisition.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(acquisition.AcquisitionLedgerError, match="not reproducible"):
        acquisition.validate_receipt_payload(forged)


def test_zero_access_and_honest_claim_boundary(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    receipt = acquisition.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert (
        "a measured 3-D source for any current object"
        in config["claim_boundary"]["does_not_establish"]
    )
    assert "campaign readiness" in config["claim_boundary"]["does_not_establish"]
