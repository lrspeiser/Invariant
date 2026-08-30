from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_group_scale_source_audit_v3 as audit


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_package(tmp_path: Path) -> Path:
    root = _repo_root()
    config = audit.load_config(root)
    paths = [audit.CONFIG_PATH, audit.MODULE_PATH, audit.TEST_PATH]
    for predecessor in config["predecessor_bindings"]:
        paths.extend(Path(predecessor[key]) for key in audit.PREDECESSOR_PATH_KEYS)
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return tmp_path


def _lane(config: dict, lane_id: str) -> dict:
    return next(row for row in config["lane_readiness"] if row["lane_id"] == lane_id)


def test_current_package_builds_and_receipt_matches() -> None:
    root = _repo_root()
    expected = audit.build_receipt(root)
    stored = json.loads((root / audit.OUTPUT_PATH).read_text(encoding="utf-8"))
    audit.validate_receipt(stored, root)
    assert stored == expected
    assert stored["counts"]["ready_science_lanes"] == 0
    assert stored["claims"]["CP10_1_complete"] is False
    assert stored["claims"]["CP10_2_complete"] is False


def test_lane_priority_and_readiness_are_honest() -> None:
    config = audit.load_config(_repo_root())
    assert _lane(config, "XCLASS_LOWZ_155")["role"] == "PREFERRED_RAW_REDUCTION_COHORT"
    assert _lane(config, "XCLASS_LOWZ_155")["decision"] == "PARTIAL"
    assert _lane(config, "EFEDS_542_RAW_REDUCTION")["role"] == ("BACKUP_COMMON_INSTRUMENT_COHORT")
    assert _lane(config, "XGAP_49_XMM")["decision"] == "BLOCKED"
    assert _lane(config, "ERASS1_2MRS_619")["license_state"] == (
        "CATALOG_IDENTIFIER_AND_TERMS_UNRESOLVED"
    )
    assert sum(row["decision"] == "READY" for row in config["lane_readiness"]) == 0


def test_accept_239_author_sample_vs_240_heasarc_rows_is_explicitly_unresolved() -> None:
    config = audit.load_config(_repo_root())
    lane = _lane(config, "ACCEPT_239")
    source = next(
        row for row in config["authoritative_source_facts"] if row["source_id"] == "ACCEPT_OFFICIAL"
    )
    assert lane["documented_objects"] is None
    assert lane["reported_counts"] == {
        "author_project_overview_sample": 239,
        "current_heasarc_one_row_per_cluster_table": 240,
    }
    assert lane["population_count_state"] == (
        "UNRESOLVED_239_AUTHOR_SAMPLE_VS_240_CURRENT_HEASARC_ROWS"
    )
    assert lane["decision"] == "BLOCKED"
    assert "original sample of 239" in source["audited_fact"]
    assert "reports 240 rows" in source["audited_fact"]


def test_builder_accounting_and_accept_disclosure_are_separate() -> None:
    config = audit.load_config(_repo_root())
    assert set(config["access_chronology"].values()) - {"ARTIFACT_BUILDER_ONLY", 0} == set()
    incident = config["interactive_audit_disclosure"]
    assert incident["incident_scope"] == "INTERACTIVE_RESEARCH_SESSION_OUTSIDE_ARTIFACT_BUILDER"
    assert incident["webpage_rendered_embedded_scientific_table_rows"] is True
    assert incident["rendered_row_count"] is None
    assert incident["query_executed"] is False
    assert incident["rows_persisted"] is False
    assert incident["rows_used_for_selection_overlap_scoring_or_facts"] is False
    assert incident["receipt_builder_involved"] is False


def test_xcop_overlap_remains_unknown() -> None:
    overlap = audit.load_config(_repo_root())["xcop_overlap_contract"]
    assert overlap["executed"] is False
    assert overlap["overlap_count"] is None
    assert len(overlap["frozen_xcop_names"]) == 12
    assert overlap["development_count"] == 8
    assert overlap["formerly_exposed_same_release_holdout_count"] == 4


def test_future_protocols_are_defined_but_locked() -> None:
    config = audit.load_config(_repo_root())
    acquisition = config["future_identity_obsid_acquisition"]
    pilot = config["future_xclass_five_object_pilot"]
    assert acquisition["preferred_lane"] == "XCLASS_LOWZ_155"
    assert acquisition["backup_lane"] == "EFEDS_542_RAW_REDUCTION"
    assert acquisition["executed"] is False and acquisition["authorized"] is False
    assert "temperature" in acquisition["forbidden_fields"]
    assert "mission_observation_id" in acquisition["allowed_fields"]
    assert pilot["object_count"] == 5
    assert pilot["object_identities_selected"] is False
    assert pilot["executed"] is False and pilot["authorized"] is False
    assert pilot["selection_must_be_target_blind"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["audit_method"].__setitem__("remote_content_archived", True),
            "audit method",
        ),
        (
            lambda value: value["authoritative_source_facts"][0].__setitem__(
                "url", "https://example.invalid"
            ),
            "section",
        ),
        (
            lambda value: value["authoritative_source_facts"][0]["assets"][1].__setitem__(
                "payload_opened", True
            ),
            "source payload boundary",
        ),
        (
            lambda value: _lane(value, "XCLASS_LOWZ_155").__setitem__("decision", "READY"),
            "ready lane",
        ),
        (
            lambda value: _lane(value, "ACCEPT_239").__setitem__(
                "population_count_state", "RESOLVED"
            ),
            "ACCEPT",
        ),
        (
            lambda value: value["xcop_overlap_contract"].__setitem__("overlap_count", 0),
            "X-COP overlap",
        ),
        (
            lambda value: value["interactive_audit_disclosure"].__setitem__(
                "webpage_rendered_embedded_scientific_table_rows", False
            ),
            "ACCEPT disclosure",
        ),
        (
            lambda value: value["future_identity_obsid_acquisition"].__setitem__(
                "authorized", True
            ),
            "future acquisition",
        ),
        (
            lambda value: value["future_xclass_five_object_pilot"].__setitem__("executed", True),
            "future pilot",
        ),
        (
            lambda value: value["access_chronology"].__setitem__("scientific_rows_opened", 1),
            "access chronology",
        ),
        (
            lambda value: value["claim_boundary"].__setitem__("group_bridge_ready", True),
            "claim boundary",
        ),
        (
            lambda value: value["predecessor_bindings"][0].__setitem__(
                "receipt_file_sha256", "0" * 64
            ),
            "predecessor",
        ),
    ],
)
def test_semantic_and_nested_mutations_fail_closed(mutation, match: str) -> None:
    changed = copy.deepcopy(audit.load_config(_repo_root()))
    mutation(changed)
    with pytest.raises(audit.GravityGroupScaleSourceAuditV3Error, match=match):
        audit.validate_config(changed)


def test_predecessor_tampering_fails_before_receipt(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    predecessor = copied / "runs/gravity/publication-readiness/group-scale-source-audit-v1.json"
    predecessor.write_bytes(predecessor.read_bytes() + b" ")
    with pytest.raises(audit.GravityGroupScaleSourceAuditV3Error, match="predecessor binding"):
        audit.load_config(copied)


def test_atomic_no_replace_and_concurrent_creators(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    expected = audit.build_receipt(copied)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: audit.write_receipt(copied), range(8)))
    assert paths == [copied / audit.OUTPUT_PATH] * 8
    assert json.loads((copied / audit.OUTPUT_PATH).read_text(encoding="utf-8")) == expected

    (copied / audit.OUTPUT_PATH).write_text('{"different":true}\n', encoding="utf-8")
    with pytest.raises(audit.GravityGroupScaleSourceAuditV3Error, match="refusing to replace"):
        audit.write_receipt(copied)


def test_module_has_no_network_science_or_model_loader() -> None:
    source = (_repo_root() / audit.MODULE_PATH).read_text(encoding="utf-8")
    for forbidden in (
        "requests",
        "urllib",
        "httpx",
        "astropy",
        "fits.open",
        "pandas",
        "numpy",
        "anthropic",
        "openai",
    ):
        assert forbidden not in source
