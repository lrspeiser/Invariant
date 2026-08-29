from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_act_dr6_erass1_overlap_preflight as preflight


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_package(tmp_path: Path) -> Path:
    root = _repo_root()
    config = json.loads((root / preflight.CONFIG_PATH).read_text(encoding="utf-8"))
    paths = [preflight.CONFIG_PATH, preflight.MODULE_PATH, preflight.TEST_PATH, preflight.AUTH_PATH]
    for parent in config["parent_bindings"]:
        paths.extend(
            Path(parent[key]) for key in ("config_path", "module_path", "test_path", "receipt_path")
        )
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return tmp_path


def _asset(config: dict, asset_id: str) -> dict:
    return next(item for item in config["file_manifest"] if item["asset_id"] == asset_id)


def _coordinated_source_rebind(config: dict) -> None:
    source = config["authoritative_sources"][0]
    source["observation_method"] = "tampered"
    source["fact_binding_sha256"] = preflight._sha(preflight._fact_material(source))


def test_current_package_and_stored_receipt_match() -> None:
    root = _repo_root()
    config = preflight.load_config(root)
    receipt = preflight.build_receipt(root)
    stored = json.loads((root / preflight.OUTPUT_PATH).read_text(encoding="utf-8"))
    preflight.validate_receipt(stored, root)
    assert stored == receipt
    assert config["access_and_decision"]["catalog_rows_opened"] == 0
    assert receipt["counts"]["ready_lanes"] == 0


def test_catalog_only_boundary_and_future_approval_are_exact() -> None:
    root = _repo_root()
    config = preflight.load_config(root)
    auth = json.loads((root / preflight.AUTH_PATH).read_text(encoding="utf-8"))
    assert auth["access_state"]["authorization"] is False
    assert all(item["authorized"] is False for item in auth["catalogs"])
    assert auth["future_exact_approval_schema"]["maximum_rows"] == {
        "ACT_DR6_LEGACY_V1_0": 3747,
        "ERASS1_PRIMARY_V3_2": 12247,
    }
    assert config["access_and_decision"]["catalog_row_access_required_for_overlap_count"] is True
    assert config["population_gate"]["catalog_overlap_count"] is None
    assert config["population_gate"]["rule_evaluated"] is False


def test_versions_discrepancy_projection_and_no_mass_leakage() -> None:
    config = preflight.load_config(_repo_root())
    assert _asset(config, "ACT_DR6_FULL_V1_0")["rows"] == 10040
    assert _asset(config, "ACT_DR6_LEGACY_V1_0")["rows"] == 3747
    assert _asset(config, "ERASS1_PRIMARY_V3_2")["rows"] == 12247
    header_fact = next(
        item
        for item in config["authoritative_sources"]
        if item["source_id"] == "ACT_DR6_FITS_HEADERS"
    )
    assert "3,747" in header_fact["audited_fact"]
    assert "3,758" in header_fact["audited_fact"]
    projection = config["catalog_projection_contract"]
    allowed = [
        column
        for catalog in ("ACT_DR6_LEGACY_V1_0", "ERASS1_PRIMARY_V3_2")
        for column in projection[catalog]["allowed_columns"]
    ]
    assert all("M500" not in column.upper() for column in allowed)
    assert "*M500*" in projection["globally_forbidden_column_patterns"]
    assert "*MGAS*" in projection["globally_forbidden_column_patterns"]


def test_primary_paper_fact_is_version_pinned_to_current_v3() -> None:
    config = preflight.load_config(_repo_root())
    source = next(
        item
        for item in config["authoritative_sources"]
        if item["source_id"] == "ACT_DR6_PRIMARY_PAPER"
    )
    assert source["url"] == "https://arxiv.org/abs/2507.21459v3"
    assert "submitted 2026-01-26" in source["observed_release"]
    assert "7,043" in source["audited_fact"]
    assert "1,690" in source["audited_fact"]
    assert "7,040" not in source["audited_fact"]
    assert "Section 5.2" in source["evidence_locator"]


def test_matching_xcop_and_population_gates_are_frozen() -> None:
    config = preflight.load_config(_repo_root())
    match = config["act_erass_match_contract"]
    assert "max(1.22 arcmin, angular size of 0.5 Mpc" in match["coordinate_radius"]
    assert match["ambiguous_or_missing_action"] == "QUARANTINE_AND_DO_NOT_COUNT"
    xcop = config["xcop_exclusion_ledger_contract"]
    assert tuple(xcop["canonical_xcop_objects"]) == preflight.XCOP_OBJECTS
    assert xcop["executed"] is False
    gate = config["population_gate"]
    assert gate["confirmatory_target_clusters"] == 192
    assert gate["underpowered_execution_floor_clusters"] == 120


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["authoritative_sources"][0].__setitem__("url", "tampered"),
            "source fact binding changed",
        ),
        (_coordinated_source_rebind, "canonical nested section changed"),
        (
            lambda value: _asset(value, "ACT_DR6_LEGACY_V1_0").__setitem__("rows", 3758),
            "asset state changed",
        ),
        (
            lambda value: _asset(value, "ERASS1_PRIMARY_V3_2").__setitem__(
                "etag_noncryptographic", "tampered"
            ),
            "asset state changed",
        ),
        (
            lambda value: value["catalog_projection_contract"]["ACT_DR6_LEGACY_V1_0"][
                "allowed_columns"
            ].append("M500"),
            "mass field entered projection",
        ),
        (
            lambda value: value["act_erass_match_contract"].__setitem__(
                "coordinate_radius", "tampered"
            ),
            "match contract changed",
        ),
        (
            lambda value: value["xcop_exclusion_ledger_contract"].__setitem__("executed", True),
            "X-COP exclusion boundary changed",
        ),
        (
            lambda value: value["population_gate"].__setitem__("catalog_overlap_count", 500),
            "population gate changed",
        ),
        (
            lambda value: value["license_and_terms"]["ACT_LAMBDA"].__setitem__(
                "redistribution_authorized", True
            ),
            "license boundary overstated",
        ),
        (
            lambda value: value["access_and_decision"].__setitem__("catalog_rows_opened", 1),
            "access boundary changed",
        ),
        (
            lambda value: value["claim_boundary"].__setitem__(
                "independent_replication_ready", True
            ),
            "claim boundary overstated",
        ),
        (
            lambda value: value["parent_bindings"][0].__setitem__("receipt_file_sha256", "0" * 64),
            "canonical nested section changed|parent binding changed",
        ),
    ],
)
def test_semantic_mutations_fail_closed(mutation, match: str) -> None:
    config = preflight.load_config(_repo_root())
    changed = copy.deepcopy(config)
    mutation(changed)
    with pytest.raises(preflight.GravityClusterActErassOverlapPreflightError, match=match):
        preflight.validate_config(changed)


def test_authorization_and_parent_tampering_fail_before_receipt(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    auth = copied / preflight.AUTH_PATH
    auth.write_bytes(auth.read_bytes() + b" ")
    with pytest.raises(
        preflight.GravityClusterActErassOverlapPreflightError, match="authorization file changed"
    ):
        preflight.load_config(copied)

    copied = _copy_package(tmp_path / "parent")
    config = json.loads((copied / preflight.CONFIG_PATH).read_text(encoding="utf-8"))
    parent_receipt = copied / config["parent_bindings"][0]["receipt_path"]
    parent_receipt.write_bytes(parent_receipt.read_bytes() + b" ")
    with pytest.raises(
        preflight.GravityClusterActErassOverlapPreflightError, match="parent binding changed"
    ):
        preflight.load_config(copied)


def test_atomic_no_replace_and_race(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    expected = preflight.build_receipt(copied)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: preflight.write_receipt(copied), range(8)))
    assert paths == [copied / preflight.OUTPUT_PATH] * 8
    stored = json.loads((copied / preflight.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert stored == expected
    (copied / preflight.OUTPUT_PATH).write_text('{"different":true}\n', encoding="utf-8")
    with pytest.raises(
        preflight.GravityClusterActErassOverlapPreflightError,
        match="refusing to replace a different overlap receipt",
    ):
        preflight.write_receipt(copied)


def test_builder_has_no_network_or_scientific_loader() -> None:
    source = (_repo_root() / preflight.MODULE_PATH).read_text(encoding="utf-8")
    for forbidden in ("requests", "urllib", "astropy", "fits.open", "pandas", "numpy"):
        assert forbidden not in source
