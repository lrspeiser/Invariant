from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_correlation_ids_partition_v1 as partition


def synthetic_line(token: bytes, tail_byte: int = 0xFF, ending: bytes = b"\n") -> bytes:
    assert len(token) == 7
    return token + bytes([tail_byte]) * 150 + ending


def test_release_gate_binds_exact_v3_and_independent_audit() -> None:
    config = partition.load_config()
    release = partition.validate_release_gate(config)
    assert release == {
        "executor_receipt_content_sha256": "a0dad57f9815c501e640155c87359f2ba8e6b96b33d573afb8d8570e39d291a9",
        "audit_receipt_content_sha256": "1025428193a0c7b9984f546da5796d46b7c9961f8410da027c3a97c573bed6e1",
    }
    assert config["independent_audit_release"]["status"] == "PASS_IDS_PARTITIONING_ONLY"


def test_i7_parser_accepts_only_authorized_positive_integer_grammar() -> None:
    assert partition.parse_i7_identifier(b"  00123") == (123, "123")
    assert partition.parse_i7_identifier(b"   +123") == (123, "123")
    for token in (b"       ", b"     -1", b"  1.000", b"    1e2", b"  1\t  ", b"12 3   ", b"\xff123456"):
        with pytest.raises(partition.VoidIdsPartitionError):
            partition.parse_i7_identifier(token)


def test_split_matches_exact_sha256_big_endian_rule_and_v3() -> None:
    for identifier in (1, 7, 123, 9999999):
        bucket, role = partition.split_bucket_role(identifier)
        expected = int.from_bytes(hashlib.sha256(str(identifier).encode("ascii")).digest()[:8], "big") % 10
        assert bucket == expected
        assert role == partition._ROLE_BY_BUCKET[expected]


def test_synthetic_scan_freezes_offsets_and_hashes_but_never_decodes_tail() -> None:
    lines = [
        synthetic_line(b"      1", 0xFF, b"\n"),
        synthetic_line(b"     22", 0x00, b"\r\n"),
        synthetic_line(b"    333", 0x80, b"\n"),
    ]
    scan = partition.scan_identifier_lines(io.BytesIO(b"".join(lines)), expected_records=3)
    assert scan["failures"] == []
    assert [entry["identifier"] for entry in scan["entries"]] == [1, 22, 333]
    assert [entry["framed_start"] for entry in scan["entries"]] == [0, len(lines[0]), len(lines[0]) + len(lines[1])]
    assert [entry["line_ending_bytes"] for entry in scan["entries"]] == [1, 2, 1]
    assert scan["decompressed_bytes"] == sum(map(len, lines))
    assert scan["entries"][0]["opaque_tail_raw_sha256"] == hashlib.sha256(bytes([0xFF]) * 150).hexdigest()
    assert scan["entries"][1]["opaque_tail_raw_sha256"] == hashlib.sha256(bytes([0x00]) * 150).hexdigest()
    assert scan["entries"][0]["identifier_field_raw_sha256"] == hashlib.sha256(b"      1").hexdigest()


def test_tail_mutation_changes_only_opaque_and_row_hashes_not_identifier_or_role() -> None:
    first = partition.scan_identifier_lines([synthetic_line(b"    123", 0x00)], expected_records=1)["entries"][0]
    second = partition.scan_identifier_lines([synthetic_line(b"    123", 0xFE)], expected_records=1)["entries"][0]
    for key in ("identifier", "canonical_identifier", "bucket", "role", "identifier_field_raw_sha256"):
        assert first[key] == second[key]
    for key in ("framed_raw_sha256", "payload_raw_sha256", "opaque_tail_raw_sha256", "leaf_sha256"):
        assert first[key] != second[key]


def test_framing_and_identifier_failures_are_retained_without_raw_values() -> None:
    lines = [
        synthetic_line(b"      1"),
        synthetic_line(b"  1.000"),
        synthetic_line(b"      2", ending=b""),
    ]
    scan = partition.scan_identifier_lines(lines, expected_records=4)
    assert len(scan["entries"]) == 1
    assert len(scan["failures"]) == 3
    assert scan["failures"][0]["failure"] == "invalid I7 identifier grammar"
    assert scan["failures"][1]["failure"] == "missing terminal LF"
    assert scan["failures"][2] == {"failure": "record count mismatch", "expected": 4, "observed": 3}
    assert all("raw_value" not in failure and "payload" not in failure for failure in scan["failures"])


def test_duplicate_and_minimum_failures_are_explicit_and_blocking() -> None:
    scan = partition.scan_identifier_lines(
        [synthetic_line(b"      1"), synthetic_line(b"      1")], expected_records=2
    )
    config = copy.deepcopy(partition.load_config())
    config["source"]["expected_records"] = 2
    config["staged_minima"].update({"development": 1, "validation": 1, "confirmation": 1})
    summary, failures = partition._partition_summary(scan, config)
    assert summary["status"] == config["block_status"]
    assert failures["duplicate_identifiers"] == [{"identifier": 1, "source_indexes": [0, 1]}]
    assert len(failures["minimum_failures"]) >= 1


def test_roots_are_exact_and_order_contract_is_distinct() -> None:
    scan = partition.scan_identifier_lines(
        [synthetic_line(b"      3"), synthetic_line(b"      1"), synthetic_line(b"      2")],
        expected_records=3,
    )
    entries = scan["entries"]
    source_root = partition._digest_root(entries)
    canonical_root = partition._digest_root(sorted(entries, key=lambda entry: entry["identifier"]))
    assert source_root != canonical_root
    assert partition._digest_root([]) == hashlib.sha256(b"").hexdigest()


def test_module_has_no_named_scientific_field_parser_surface() -> None:
    source = partition.MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("DMzp", "e_DMzp", "V3k", "RAdeg", "DEdeg", "Dist"):
        assert forbidden not in source
    assert "payload[ID_START:ID_STOP]" in source
    assert "payload[ID_STOP:]" in source
    assert "choices=(\"build\", \"check\", \"status\")" in source


def test_real_frozen_package_passes_counts_roots_and_zero_scientific_access() -> None:
    receipt = partition.check_package()
    assert receipt["status"] == "PASS_IDS_PARTITIONED_ONLY_AWAIT_INDEPENDENT_AUDIT_BEFORE_DEVELOPMENT_RELEASE"
    assert sum(receipt["bucket_counts"].values()) == 38053
    assert sum(receipt["role_counts"].values()) == 38053
    assert receipt["failures"] == {"status": "PASS_NO_FAILURES", "parse_or_framing": 0, "duplicates": 0, "minimums": 0}
    assert all(receipt["role_counts"][role] >= minimum for role, minimum in {"development": 500, "validation": 150, "confirmation": 150}.items())
    assert receipt["access_accounting"]["identifier_rows_decoded"] == 38053
    assert receipt["access_accounting"]["identifier_bytes_decoded"] == 38053 * 7
    assert receipt["access_accounting"]["nonidentifier_cf4_bytes_semantically_decoded"] == 0
    assert receipt["access_accounting"]["scientific_rows_decoded"] == 0
    assert receipt["access_accounting"]["VAST_files_opened"] == 0
    assert receipt["access_accounting"]["Pantheon_files_opened"] == 0
    assert receipt["access_accounting"]["response_values_inspected"] == 0
    assert receipt["access_accounting"]["real_scores"] == 0


def test_frozen_ledger_recomputes_every_split_leaf_count_and_root() -> None:
    _raw, entries = partition._read_ledger_strict(partition.LEDGER_PATH)
    assert len(entries) == 38053
    assert len({entry["identifier"] for entry in entries}) == len(entries)
    for entry in entries:
        assert partition.split_bucket_role(entry["identifier"]) == (entry["bucket"], entry["role"])
        body = dict(entry)
        leaf = body.pop("leaf_sha256")
        assert leaf == partition._leaf_hash(body)
    receipt = partition.check_package()
    assert receipt["roots"]["source_order_root"] == partition._digest_root(entries)
    ordered = sorted(entries, key=lambda entry: (entry["identifier"], entry["source_index"]))
    assert receipt["roots"]["canonical_id_root"] == partition._digest_root(ordered)


def test_check_and_existing_build_never_decompress_source_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_gzip(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("check/existing build attempted a second source decompression")

    monkeypatch.setattr(gzip, "open", forbidden_gzip)
    assert partition.check_package()["status"].startswith("PASS_IDS_PARTITIONED_ONLY")
    assert partition.write_package() == "EXISTING_VALID_NO_SOURCE_DECOMPRESSION"


def test_vast_and_pantheon_paths_are_never_opened_by_check(monkeypatch: pytest.MonkeyPatch) -> None:
    original_open = Path.open
    opened: list[str] = []

    def traced_open(path: Path, *args: object, **kwargs: object) -> object:
        opened.append(path.as_posix())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced_open)
    partition.check_package()
    assert not any("vast-table" in path.lower() or "pantheon" in path.lower() for path in opened)


def test_receipt_and_artifacts_are_canonical_and_self_hashed() -> None:
    receipt = json.loads(partition.RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["content_sha256"] == partition._self_hash(receipt)
    assert receipt["mutation_freeze"]["config_raw_sha256"] == partition.file_sha256(partition.CONFIG_PATH)
    assert receipt["mutation_freeze"]["module_semantic_sha256"] == partition.module_semantic_sha256()
    for artifact in receipt["artifacts"].values():
        path = partition.canonical_bound_path(artifact["path"])
        assert partition.file_sha256(path) == artifact["raw_sha256"]
