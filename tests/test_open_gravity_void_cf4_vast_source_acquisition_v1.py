from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_cf4_vast_source_acquisition_v1 as source

ROOT = Path(__file__).resolve().parents[1]


def test_exact_inventory_and_opaque_bytes() -> None:
    config = source.load_config(ROOT)
    rows = source.validate_opaque_sources(config, ROOT)
    assert tuple(row["id"] for row in rows) == source.EXPECTED_IDS
    assert sum(row["bytes"] for row in rows) == 8_340_564
    assert all(row["verification"] == "OPAQUE_BYTES_ONLY_NO_DECODE" for row in rows)


def test_response_and_void_roles_are_separate() -> None:
    config = source.load_config(ROOT)
    response = [row["id"] for row in config["files"] if row["scientific_response"]]
    assert response == ["CF4_TABLE3_GROUP_METHODS", "CF4_TABLE4_GROUP_DISTANCE_VELOCITY"]
    assert all(not row["scientific_response"] for row in config["files"][3:])


def test_claims_and_access_are_zero_row() -> None:
    config = source.load_config(ROOT)
    access = config["access_accounting"]
    assert access["scientific_rows_decoded"] == 0
    assert access["response_values_inspected"] == 0
    assert access["scores_computed"] == 0
    assert config["claim_boundary"]["real_data_fit"] is False
    assert config["claim_boundary"]["law_repaired"] is False


def test_future_contract_forbids_response_and_distance_shortcuts() -> None:
    contract = source.load_config(ROOT)["future_decode_contract"]
    assert "never use the published Vpec" in contract["primary_response"]
    assert "never cz/H0" in contract["target_distance"]
    assert "separate successor before" in contract["no_retuning"]
    assert any("random sky rotations" in row for row in contract["anti_circularity"])


def test_source_byte_tamper_fails(tmp_path: Path) -> None:
    config = source.load_config(ROOT)
    altered = copy.deepcopy(config)
    original = ROOT / altered["files"][0]["local_path"]
    forged = tmp_path / "forged"
    forged.write_bytes(original.read_bytes() + b"x")
    altered["files"][0]["local_path"] = forged.as_posix()
    with pytest.raises(source.VoidSourceAcquisitionError, match="source path escaped"):
        source.validate_config(altered)


def test_coherent_hash_claim_mutation_rejects() -> None:
    config = source.load_config(ROOT)
    widened = copy.deepcopy(config)
    widened["claim_boundary"]["real_data_fit"] = True
    with pytest.raises(source.VoidSourceAcquisitionError, match="claim widened"):
        source.validate_config(widened)
    accessed = copy.deepcopy(config)
    accessed["access_accounting"]["scientific_rows_decoded"] = 1
    with pytest.raises(source.VoidSourceAcquisitionError, match="access accounting drift"):
        source.validate_config(accessed)


def test_predecessor_and_origin_are_exact() -> None:
    config = source.load_config(ROOT)
    source._validate_predecessor_and_origin(config, ROOT)


def test_receipt_build_is_deterministic_and_self_hashed() -> None:
    first = source.build_receipt(ROOT)
    second = source.build_receipt(ROOT)
    assert first == second
    assert first["content_sha256"] == source._self_hash(first)
    assert first["counts"]["scientific_rows_decoded"] == 0
    assert len(first["source_bundle_root_sha256"]) == 64


def test_stored_receipt_and_replay_are_exact() -> None:
    source.check_receipt(ROOT)
    path = ROOT / source.OUTPUT_PATH
    before = path.read_bytes()
    assert source.write_receipt(ROOT) == "EXISTING_IDENTICAL"
    assert path.read_bytes() == before
    observed = json.loads(path.read_text(encoding="utf-8"))
    assert observed == source.build_receipt(ROOT)


def test_module_has_no_scientific_decoder() -> None:
    text = (ROOT / source.MODULE_PATH).read_text(encoding="utf-8")
    for forbidden in ("import gzip", "import csv", "numpy", "pandas", "readline("):
        assert forbidden not in text
