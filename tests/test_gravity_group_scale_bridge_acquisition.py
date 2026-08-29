from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_group_scale_bridge_acquisition as bridge


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_package(tmp_path: Path) -> Path:
    root = _repo_root()
    paths = (
        bridge.CONFIG_PATH,
        bridge.MODULE_PATH,
        bridge.TEST_PATH,
        bridge.SOURCE_PATH,
        Path("configs/gravity_group_scale_source_audit_v1.json"),
        Path("src/sigma_theory_compiler/gravity_group_scale_source_audit.py"),
        Path("tests/test_gravity_group_scale_source_audit.py"),
        Path("runs/gravity/publication-readiness/group-scale-source-audit-v1.json"),
    )
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return tmp_path


def _asset(config: dict, asset_id: str) -> dict:
    return next(item for item in config["remote_assets"] if item["asset_id"] == asset_id)


def _coordinated_source_rebind(config: dict) -> None:
    source = config["authoritative_sources"][0]
    source["evidence_locator"] = "coordinated tamper"
    source["fact_binding_sha256"] = bridge._sha(bridge._fact_material(source))


def test_current_package_and_stored_receipt_match() -> None:
    root = _repo_root()
    config = bridge.load_config(root)
    receipt = bridge.build_receipt(root)
    stored = json.loads((root / bridge.OUTPUT_PATH).read_text(encoding="utf-8"))
    bridge.validate_receipt(stored, root)
    assert stored == receipt
    assert config["claim_boundary"]["group_scale_bridge_ready"] is False
    assert receipt["counts"]["ready_science_lanes"] == 0


def test_access_accounting_separates_manifest_from_scientific_rows() -> None:
    config = bridge.load_config(_repo_root())
    access = config["access_chronology"]
    assert access["clean_metadata_manifest_downloads"] == 1
    assert access["clean_metadata_manifest_bytes"] == 6278
    assert access["scientific_payload_downloads"] == 0
    assert access["scientific_payload_bytes"] == 0
    assert access["sample_alias_rows_opened"] == 0
    assert access["thermodynamic_rows_opened"] == 0
    assert access["stellar_baryon_rows_opened"] == 0
    assert access["inferred_mass_rows_opened"] == 0
    assert access["lensing_rows_opened"] == 0
    assert access["target_rows_opened"] == 0
    assert access["scores_computed"] == 0
    assert access["authorization"] is False


def test_frozen_source_is_only_the_cds_readme_schema_manifest() -> None:
    root = _repo_root()
    path = root / bridge.SOURCE_PATH
    payload = path.read_bytes()
    assert len(payload) == bridge.SOURCE_FILE_BYTES
    assert bridge._file_sha(path) == bridge.SOURCE_FILE_SHA256
    text = payload.decode("ascii")
    assert "File Summary:" in text
    assert "axes2mrs.dat     147      558" in text
    assert "Byte-by-byte Description of file: axes2mrs.dat" in text
    config = bridge.load_config(root)
    data = _asset(config, "AXES2MRS_DATA")
    assert data["downloaded"] is False
    assert data["local_path"] is None
    assert data["file_sha256"] is None


def test_xgap_redirect_redshift_discrepancy_and_overlap_stay_blocked() -> None:
    config = bridge.load_config(_repo_root())
    historical_ids = {item["source_id"] for item in config["historical_non_authoritative_evidence"]}
    authoritative_ids = {item["source_id"] for item in config["authoritative_sources"]}
    assert historical_ids == {"XGAP_SAMPLE_SELECTION_LEGACY_CACHE"}
    assert "XGAP_SAMPLE_SELECTION_LEGACY_CACHE" not in authoritative_ids
    assert config["fact_provenance_contract"]["historical_legacy_excerpt_archived"] is False
    sample_page = _asset(config, "XGAP_SAMPLE_PAGE")
    master = _asset(config, "XGAP_MASTER_V1_1")
    assert sample_page["observed_http_status"] == 301
    assert sample_page["observed_redirect"] == "https://www.unige.ch/sciences/astro/en"
    assert master["observed_http_status"] == 301
    assert master["downloaded"] is False
    xgap = config["sample_and_alias_state"]["xgap"]
    assert xgap["documented_final_object_count"] == 49
    assert xgap["legacy_selection_redshift_rule"] == "z < 0.05"
    assert xgap["primary_selection_redshift_rule"] == "z < 0.06"
    assert xgap["redshift_discrepancy_resolved"] is False
    overlap = config["xcop_overlap_contract"]
    assert overlap["executed"] is False
    assert overlap["overlap_count"] is None
    assert overlap["input_alias_rows"] == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["authoritative_sources"][0].__setitem__(
                "url", "https://example.invalid/tampered"
            ),
            "source fact binding changed",
        ),
        (
            lambda value: value["authoritative_sources"][0].__setitem__("audited_fact", "tampered"),
            "source fact binding changed",
        ),
        (
            lambda value: value["authoritative_sources"][0].__setitem__(
                "observation_method", "tampered"
            ),
            "source fact binding changed",
        ),
        (
            lambda value: value["authoritative_sources"][0].__setitem__(
                "evidence_locator", "tampered"
            ),
            "source fact binding changed",
        ),
        (
            _coordinated_source_rebind,
            "canonical nested section changed: authoritative_sources",
        ),
        (
            lambda value: _asset(value, "AXES2MRS_DATA").__setitem__("downloaded", True),
            "payload boundary changed",
        ),
        (
            lambda value: _asset(value, "AXES2MRS_README").__setitem__(
                "observed_etag", '"tampered"'
            ),
            "canonical nested section changed: remote_assets",
        ),
        (
            lambda value: _asset(value, "AXES2MRS_README").__setitem__(
                "observed_last_modified", "tampered"
            ),
            "canonical nested section changed: remote_assets",
        ),
        (
            lambda value: value["source_discovery_scope"].__setitem__(
                "observed_result", "tampered"
            ),
            "canonical nested section changed: source_discovery_scope",
        ),
        (
            lambda value: value["license_and_redistribution_state"].__setitem__("rule", "tampered"),
            "canonical nested section changed: license_and_redistribution_state",
        ),
        (
            lambda value: value["access_chronology"].__setitem__("thermodynamic_rows_opened", 1),
            "access chronology changed",
        ),
        (
            lambda value: value["claim_boundary"].__setitem__("group_scale_bridge_ready", True),
            "claim boundary overstated",
        ),
        (
            lambda value: value["license_and_redistribution_state"].__setitem__(
                "redistribution_claim", True
            ),
            "license boundary overstated",
        ),
        (
            lambda value: value["xcop_overlap_contract"].__setitem__("executed", True),
            "overlap boundary changed",
        ),
        (
            lambda value: value["parent_source_audit_binding"].__setitem__(
                "receipt_content_sha256", "0" * 64
            ),
            "parent source-audit binding changed",
        ),
    ],
)
def test_semantic_mutations_fail_closed(mutation, match: str) -> None:
    config = bridge.load_config(_repo_root())
    changed = copy.deepcopy(config)
    mutation(changed)
    with pytest.raises(bridge.GravityGroupScaleBridgeAcquisitionError, match=match):
        bridge.validate_config(changed)


def test_source_and_parent_tampering_fail_before_receipt(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    source = copied / bridge.SOURCE_PATH
    source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(bridge.GravityGroupScaleBridgeAcquisitionError, match="ReadMe changed"):
        bridge.load_config(copied)

    copied = _copy_package(tmp_path / "parent")
    parent_receipt = copied / "runs/gravity/publication-readiness/group-scale-source-audit-v1.json"
    parent_receipt.write_bytes(parent_receipt.read_bytes() + b" ")
    with pytest.raises(
        bridge.GravityGroupScaleBridgeAcquisitionError, match="parent binding changed"
    ):
        bridge.load_config(copied)


def test_atomic_no_replace_and_race(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    expected = bridge.build_receipt(copied)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: bridge.write_receipt(copied), range(8)))
    assert paths == [copied / bridge.OUTPUT_PATH] * 8
    stored = json.loads((copied / bridge.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert stored == expected

    (copied / bridge.OUTPUT_PATH).write_text('{"different":true}\n', encoding="utf-8")
    with pytest.raises(
        bridge.GravityGroupScaleBridgeAcquisitionError,
        match="refusing to replace a different acquisition receipt",
    ):
        bridge.write_receipt(copied)


def test_module_has_no_network_or_scientific_payload_loader() -> None:
    source = (_repo_root() / bridge.MODULE_PATH).read_text(encoding="utf-8")
    for forbidden in ("requests", "urllib", "astropy", "fits.open", "pandas", "numpy"):
        assert forbidden not in source
