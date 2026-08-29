from __future__ import annotations

import copy
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_group_scale_source_audit as audit

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_audit_advances_cp10_without_completing_or_opening_it() -> None:
    receipt = audit.build_receipt(ROOT)
    assert (
        receipt["decision"]
        == "METADATA_AUDIT_SEALED_DISCLOSED_EXCERPT_ZERO_READY_LANES_CP10_1_CP10_2_REMAIN_OPEN"
    )
    assert receipt["CP10_status"] == {
        "CP10.1": "ADVANCED_METADATA_SOURCE_AUDIT_ONLY_NOT_COMPLETE",
        "CP10.2": "ADVANCED_ENDPOINT_AND_SPLIT_CONTRACT_ONLY_NOT_COMPLETE",
    }
    assert receipt["counts"] == {
        "scope": "RECEIPT_BUILDER_ONLY",
        "authoritative_metadata_sources": 12,
        "candidate_lanes": 3,
        "ready_lanes": 0,
        "file_manifest_records": 4,
        "direct_endpoint_roles": 4,
        "covariance_roles": 7,
        "calibration_roles": 7,
        "payload_rows_opened": 0,
        "thermodynamic_rows_opened": 0,
        "stellar_baryon_rows_opened": 0,
        "inferred_mass_rows_opened": 0,
        "lensing_rows_opened": 0,
        "scientific_scores_computed": 0,
        "downloads": 0,
        "downloaded_bytes": 0,
        "network_calls_by_receipt_builder": 0,
        "paid_or_model_calls": 0,
    }
    assert receipt["claims"]["CP10_1_complete"] is False
    assert receipt["claims"]["CP10_2_complete"] is False
    assert receipt["claims"]["source_metadata_audit_complete"] is True
    assert receipt["claims"]["interactive_audit_zero_row_purity"] is False
    assert receipt["claims"]["receipt_builder_zero_row_purity"] is True
    assert receipt["claims"]["public_lane_ready"] is False
    assert receipt["claims"]["observational_authorization"] is False
    assert receipt["claims"]["central_readiness_changed"] is False
    disclosure = receipt["interactive_audit_disclosure"]
    assert disclosure["public_appendix_measurement_rows_rendered"] == 1
    assert disclosure["persisted_in_package"] is False
    assert disclosure["receipt_builder_involved"] is False


def test_xgap_is_mass_matched_but_blocked_at_current_file_level() -> None:
    config = audit.load_config(ROOT)
    lanes = {lane["lane_id"]: lane for lane in config["candidate_lanes"]}
    xgap = lanes["XGAP_49_XMM"]
    assert "49" in xgap["sample_scope"]
    assert "1e13 < M500 < 1e14" in xgap["sample_scope"]
    assert "legacy page says z < 0.05" in xgap["sample_scope"]
    assert "2024 primary says z < 0.06" in xgap["sample_scope"]
    redshift = config["xgap_redshift_reconciliation"]
    assert redshift["discrepancy_resolved"] is False
    assert redshift["payload_selection_rule_chosen"] is None
    assert redshift["failure_action"] == "BLOCK_SAMPLE_ASSEMBLY_AND_SPLIT"
    assert xgap["readiness"]["public_sample_scope_documented"] is True
    assert xgap["readiness"]["machine_readable_sample_manifest_documented"] is True
    assert xgap["readiness"]["machine_readable_sample_manifest_currently_retrievable"] is False
    assert xgap["readiness"]["density_endpoint_files_verified"] is False
    assert xgap["readiness"]["stellar_baryon_endpoint_files_verified"] is False
    assert xgap["readiness"]["cross_endpoint_covariance_verified"] is False
    manifest = config["file_manifest_audit"][0]
    assert manifest["reported_filename"] == "xgap_master_v1.1.fits"
    assert manifest["reported_size"] == "25.31 KB"
    assert manifest["file_sha256"] is None
    assert manifest["payload_opened"] is False
    assert manifest["current_retrieval_state"].startswith("BLOCKED_")


def test_current_xgap_publications_do_not_substitute_integrated_observables() -> None:
    config = audit.load_config(ROOT)
    sources = {row["source_id"]: row for row in config["authoritative_sources"]}
    assert sources["XGAP_2026_SCALING_RELATIONS"]["observed_release"] == (
        "Astronomy & Astrophysics 709, L4 (2026); arXiv v2 2026-05-19"
    )
    assert sources["XGAP_2026_FORWARD_MODEL"]["doi"] == ("10.1051/0004-6361/202660011")
    xgap = next(lane for lane in config["candidate_lanes"] if lane["lane_id"] == "XGAP_49_XMM")
    assert "integrated observables" in xgap["audit_details"]["endpoint_blocker"]
    assert "Population-summary covariance" in xgap["audit_details"]["covariance_blocker"]
    assert not xgap["readiness"]["density_endpoint_files_verified"]
    assert not xgap["readiness"]["cross_endpoint_covariance_verified"]


def test_every_authoritative_fact_has_exact_local_provenance_binding() -> None:
    config = audit.load_config(ROOT)
    assert tuple(audit.SOURCE_FACT_BINDINGS) == audit.SOURCE_IDS
    for source in config["authoritative_sources"]:
        assert source["retrieved_at"] == "2026-08-29"
        assert source["observation_method"]
        assert source["evidence_locator"]
        assert source["fact_binding_sha256"] == audit._sha(audit._source_fact_material(source))
        assert source["fact_binding_sha256"] == audit.SOURCE_FACT_BINDINGS[source["source_id"]]
    assert config["fact_provenance_contract"]["remote_content_archived"] is False
    assert config["fact_provenance_contract"]["remote_content_sha256_available"] is False


def test_comparison_lanes_do_not_substitute_stacks_or_published_claims_for_rows() -> None:
    config = audit.load_config(ROOT)
    lanes = {lane["lane_id"]: lane for lane in config["candidate_lanes"]}
    sun09 = lanes["SUN09_CHANDRA_43"]
    assert "43" in sun09["sample_scope"]
    assert "23" in sun09["sample_scope"]
    assert sun09["readiness"]["temperature_endpoint_files_verified"] is False
    erass = lanes["ERASS1_STACKED_GROUP_THERMODYNAMICS"]
    assert "1178" in erass["sample_scope"]
    assert "271" in erass["sample_scope"]
    assert not erass["readiness"]["machine_readable_sample_manifest_documented"]
    assert not erass["readiness"]["machine_readable_sample_manifest_currently_retrievable"]
    assert not erass["readiness"]["mass_scope_1e13_1e14_documented"]
    assert not erass["readiness"]["no_inferred_mass_or_lensing_target_dependency"]
    assert "stack-level averages" in erass["audit_details"]["endpoint_blocker"]


def test_direct_endpoints_pressure_covariance_and_stellar_baryons_are_explicit() -> None:
    config = audit.load_config(ROOT)
    endpoints = {row["endpoint_id"]: row for row in config["direct_endpoint_contract"]}
    assert tuple(endpoints) == audit.ENDPOINT_IDS
    assert (
        "density-temperature cross-covariance"
        in endpoints["ELECTRON_PRESSURE_RADIAL"]["allowed_derivation"]
    )
    assert (
        "stack_average_as_object_endpoint"
        in endpoints["ELECTRON_PRESSURE_RADIAL"]["forbidden_substitutes"]
    )
    assert (
        "luminosity_temperature_scaling_estimate"
        in endpoints["SPECTROSCOPIC_TEMPERATURE_RADIAL"]["forbidden_substitutes"]
    )
    stellar = endpoints["STELLAR_BARYON_CUMULATIVE"]
    assert "BGG, satellites, and diffuse intragroup light" in stellar["required_source"]
    assert tuple(config["required_covariance_roles"]) == audit.COVARIANCE_ROLES
    assert tuple(config["required_calibration_roles"]) == audit.CALIBRATION_ROLES


def test_whole_group_split_is_deterministic_and_target_blind_but_not_executed() -> None:
    config = audit.load_config(ROOT)
    split = config["whole_group_split_contract"]
    assert split["response_or_target_fields_used"] == []
    assert split["post_response_movement_allowed"] is False
    assert split["split_execution_authorized"] is False
    key = "SYNTHETIC_LANE|CONTROL_001|150.000000|2.000000|0.030000"
    first = audit.split_bucket(key)
    second = audit.split_bucket(key)
    assert first == second
    assert 0 <= first[0] <= 9
    assert first[1] in split["assignments"]
    with pytest.raises(audit.GravityGroupScaleSourceAuditError, match="incomplete"):
        audit.split_bucket("CONTROL_001")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["selection_state"].__setitem__("selected_lane", "XGAP_49_XMM"),
        lambda value: value["selection_state"].__setitem__("observational_authorization", True),
        lambda value: value["selection_state"].__setitem__("payload_rows_opened", 1),
        lambda value: value["selection_state"].__setitem__("lensing_rows_opened", 1),
        lambda value: value["claim_boundary"].__setitem__("CP10_1_complete", True),
        lambda value: value["alias_overlap_contract"].__setitem__("xcop_overlap_audited", True),
        lambda value: value["whole_group_split_contract"]["response_or_target_fields_used"].append(
            "temperature"
        ),
        lambda value: value["file_manifest_audit"][0].__setitem__("payload_opened", True),
        lambda value: value["candidate_lanes"][0]["audit_details"].__setitem__(
            "payload_commitment", "unsealed-packet"
        ),
        lambda value: value["candidate_lanes"][2]["readiness"].__setitem__(
            "no_inferred_mass_or_lensing_target_dependency", True
        ),
        lambda value: value["interactive_audit_disclosure"].__setitem__(
            "public_appendix_measurement_rows_rendered", 0
        ),
        lambda value: value["xgap_redshift_reconciliation"].__setitem__(
            "discrepancy_resolved", True
        ),
        lambda value: value["implementation_evidence"].__setitem__("test_file_sha256", "0" * 64),
    ],
)
def test_authorization_claim_alias_payload_and_leakage_mutations_fail_closed(
    mutation: object,
) -> None:
    config = copy.deepcopy(audit.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(audit.GravityGroupScaleSourceAuditError):
        audit.validate_config(config)


def test_config_and_stored_receipt_are_content_bound() -> None:
    assert audit._file_sha(ROOT / audit.CONFIG_PATH) == audit.CONFIG_FILE_SHA256
    stored = json.loads((ROOT / audit.OUTPUT_PATH).read_text(encoding="utf-8"))
    audit.validate_receipt(stored, ROOT)
    assert stored == audit.build_receipt(ROOT)
    assert stored["config_binding"]["file_sha256"] == audit.CONFIG_FILE_SHA256
    assert stored["implementation_binding"]["test_path"] == audit.TEST_PATH.as_posix()
    assert stored["implementation_binding"]["test_file_sha256"] == (audit.TEST_FILE_SHA256)
    assert stored["implementation_binding"]["test_config_binding_matches"] is True


def test_receipt_writer_is_append_only_and_refuses_clobber(tmp_path: Path) -> None:
    config_path = tmp_path / audit.CONFIG_PATH
    source_path = tmp_path / "src/sigma_theory_compiler/gravity_group_scale_source_audit.py"
    test_path = tmp_path / audit.TEST_PATH
    config_path.parent.mkdir(parents=True)
    source_path.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / audit.CONFIG_PATH, config_path)
    shutil.copy2(
        ROOT / "src/sigma_theory_compiler/gravity_group_scale_source_audit.py",
        source_path,
    )
    shutil.copy2(ROOT / audit.TEST_PATH, test_path)
    output = audit.write_receipt(tmp_path)
    first = output.read_bytes()
    assert audit.write_receipt(tmp_path).read_bytes() == first
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(audit.GravityGroupScaleSourceAuditError, match="overwrite"):
        audit.write_receipt(tmp_path)


def test_same_directory_atomic_no_replace_publication_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "receipt.json"
    payloads = (b'{"writer":1}\n', b'{"writer":2}\n')
    barrier = threading.Barrier(2)
    original_link = audit.os.link

    def synchronized_link(source: Path, destination: Path) -> None:
        barrier.wait(timeout=5)
        original_link(source, destination)

    monkeypatch.setattr(audit.os, "link", synchronized_link)

    def publish(payload: bytes) -> str:
        try:
            audit._atomic_publish_no_replace(target, payload)
        except FileExistsError:
            return "lost"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, payloads))
    assert sorted(results) == ["lost", "published"]
    assert target.read_bytes() in payloads
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("url", "https://example.invalid/tampered"),
        ("doi", "10.0000/tampered"),
        ("observed_release", "tampered release"),
        ("audited_fact", "tampered fact"),
        ("observation_method", "tampered method"),
        ("retrieved_at", "2026-08-28"),
        ("evidence_locator", "tampered locator"),
        ("fact_binding_sha256", "0" * 64),
    ],
)
def test_nested_authoritative_source_fact_mutations_fail_closed(
    field: str, replacement: object
) -> None:
    config = copy.deepcopy(audit.load_config(ROOT))
    config["authoritative_sources"][0][field] = replacement
    with pytest.raises(audit.GravityGroupScaleSourceAuditError):
        audit.validate_config(config)


def test_recomputed_mutated_source_binding_still_fails_expected_fact_seal() -> None:
    config = copy.deepcopy(audit.load_config(ROOT))
    source = config["authoritative_sources"][0]
    source["url"] = "https://example.invalid/rebound"
    source["fact_binding_sha256"] = audit._sha(audit._source_fact_material(source))
    with pytest.raises(audit.GravityGroupScaleSourceAuditError):
        audit.validate_config(config)


def test_test_file_tampering_fails_before_receipt_build(tmp_path: Path) -> None:
    config_path = tmp_path / audit.CONFIG_PATH
    test_path = tmp_path / audit.TEST_PATH
    config_path.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / audit.CONFIG_PATH, config_path)
    test_path.write_bytes((ROOT / audit.TEST_PATH).read_bytes() + b"# tampered\n")
    with pytest.raises(audit.GravityGroupScaleSourceAuditError, match="test binding changed"):
        audit.load_config(tmp_path)


def test_implementation_has_no_network_or_scientific_payload_loader() -> None:
    source = (ROOT / "src/sigma_theory_compiler/gravity_group_scale_source_audit.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "import requests",
        "import urllib",
        "import httpx",
        "import pandas",
        "import astropy",
        "fits.open",
        "urlopen(",
    ):
        assert forbidden not in source
