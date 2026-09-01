from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_twell_400_source_shaped_rebind_replay_v2 as subject

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> dict[str, object]:
    return subject.load_config(ROOT)


def test_config_binds_all_v1_files_and_blocked_audit(config: dict[str, object]) -> None:
    assert config["schema"] == subject.CONFIG_SCHEMA
    assert len(config["v1_bindings"]) == 14
    assert sum(row["id"] in subject._SCIENCE_IDS for row in config["v1_bindings"]) == 10
    assert config["blocked_audit"]["raw_sha256"] == (
        "41f5ae2d01aea055ac10373b7789fe7cba0da543f9d73e253dde0273dac502e1"
    )
    assert config["blocked_audit"]["content_sha256"] == (
        "786ac589b8e2221d7679bb58d4c2b1e2338dddda74893c82d1d6bd8fb912da24"
    )


def test_config_mutation_is_rejected(config: dict[str, object]) -> None:
    mutated = copy.deepcopy(config)
    mutated["status"] = "MUTATED"
    with pytest.raises(subject.TwellSourceRebindV2Error, match="status changed"):
        subject.validate_config(ROOT, mutated, verify_files=False)


def test_every_v1_raw_and_content_hash_is_verified(config: dict[str, object]) -> None:
    for row in config["v1_bindings"]:
        path = ROOT / row["path"]
        assert subject.file_sha256(path) == row["raw_sha256"]
        semantic, count = subject._binding_semantics(path, row["kind"])
        assert semantic == row["content_sha256"]
        assert count == row["row_count"]


def test_v1_scientific_payload_counts_are_preserved(config: dict[str, object]) -> None:
    result = subject.validate_v1_scientific_payload(ROOT, config)
    assert result["compatibility_counts"] == {
        "EXECUTABLE": 110,
        "INCOMPATIBLE_FEATURE_SET": 1600,
        "SOURCE_BLOCKED": 290,
    }
    assert result["unique_execution_counts"] == {
        "COMPLETED": 2554,
        "NUMERICAL_INVALID": 38,
    }
    assert result["replay_counts"] == {
        "COMPLETED": 61296,
        "NUMERICAL_INVALID": 912,
    }
    assert result["invalid_formula_counts"] == {
        "TW2-A11-D03": 16,
        "TW2-A11-D04": 5,
        "TW2-A11-D06": 16,
        "TW2-A11-D07": 1,
    }
    assert result["finite_source_prediction_tie_group_count"] == 122
    assert result["independent_array_comparisons"] == 5184
    assert result["independent_recomputation_mismatches"] == 0


def test_v2_has_no_science_mirror_and_does_not_modify_v1(config: dict[str, object]) -> None:
    preservation = config["preservation_contract"]
    assert preservation["v1_science_artifact_count"] == 10
    assert preservation["v2_science_artifact_mirror_emitted"] is False
    assert preservation["v1_files_modified"] == 0
    assert preservation["scientific_payload_recomputed"] is False
    assert preservation["scientific_payload_reused_byte_exact"] is True
    output = ROOT / subject.OUTPUT_DIR
    assert not output.exists() or {path.name for path in output.iterdir()} <= {"receipt.json"}


def test_build_receipt_corrects_only_mechanical_release_claim() -> None:
    receipt = subject.build_receipt(ROOT)
    assert receipt["repair"]["kind"] == ("MECHANICAL_FORMAT_GATE_AND_RELEASE_CLAIM_CORRECTION_ONLY")
    assert receipt["repair"]["v1_mechanical_disposition"] == ("BLOCKED_RUFF_FORMAT_CHECK_FAILURE")
    assert receipt["repair"]["v1_numerical_disposition"] == (
        "INDEPENDENTLY_RECOMPUTED_SOURCE_ONLY_SYNTHETIC_MATCH"
    )
    assert receipt["repair"]["mechanical_pass_claimed_before_distinct_audit"] is False
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True


def test_zero_response_access_and_claim_ceiling() -> None:
    receipt = subject.build_receipt(ROOT)
    access = receipt["access_accounting"]
    assert access["response_npz_members_opened"] == 0
    assert access["response_values_opened"] == 0
    assert access["candidate_npz_members_opened"] == 0
    assert access["variance_npz_members_opened"] == 0
    assert access["truth_npz_members_opened"] == 0
    assert access["scientific_scores_computed"] == 0
    assert receipt["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL"
    assert all(
        value is False for key, value in receipt["claim_boundary"].items() if key != "claim_class"
    )


def test_frozen_receipt_matches_deterministic_rebuild() -> None:
    receipt = subject.validate_receipt(ROOT)
    assert receipt["content_sha256"] == subject._receipt_content_sha256(receipt)
    assert (
        subject.file_sha256(ROOT / subject.CONFIG_PATH)
        == receipt["package_hashes"]["config_raw_sha256"]
    )
    assert (
        subject.file_sha256(ROOT / subject.MODULE_PATH)
        == receipt["package_hashes"]["module_raw_sha256"]
    )
    assert (
        subject.file_sha256(ROOT / subject.TEST_PATH)
        == receipt["package_hashes"]["test_raw_sha256"]
    )


def test_alternate_working_directory_check_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    receipt = subject.validate_receipt(ROOT)
    assert receipt["package_id"] == subject.OUTPUT_DIR.name
    assert subject.write_receipt(ROOT) == "EXISTING_IDENTICAL"


def test_blocked_audit_self_seal_and_mechanical_status(config: dict[str, object]) -> None:
    binding = config["blocked_audit"]
    audit = json.loads((ROOT / binding["path"]).read_text(encoding="utf-8"))
    assert subject.file_sha256(ROOT / binding["path"]) == binding["raw_sha256"]
    assert subject._receipt_content_sha256(audit) == binding["content_sha256"]
    assert audit["status"] == "BLOCK_MECHANICAL_FORMAT_GATE_ONLY"
    assert audit["mismatches"] == 0
    assert audit["scientific_replay"] == "PASS_SOURCE_ONLY_SYNTHETIC"
