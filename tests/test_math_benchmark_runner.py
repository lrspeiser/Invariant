from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.math_benchmark_runner import (
    OUTPUT_PATH,
    _inspect_implementation,
    _load_config,
    _write_immutable,
    expand_slots,
    run,
    validate_readiness,
)

ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.fixture(scope="module")
def checked() -> dict[str, object]:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_readiness(value, ROOT)
    assert value == run(ROOT)
    return value


def test_closed_registry_has_exactly_one_hundred_slots_per_cohort() -> None:
    config = _load_config(ROOT)
    slots = expand_slots(config)
    assert len(slots) == len({slot["slot_id"] for slot in slots}) == 200
    assert Counter(slot["cohort"] for slot in slots) == {"historical": 100, "synthetic": 100}
    partitions = Counter(
        (slot["cohort"], slot["domain"], slot["artifact_type"], slot["level"]) for slot in slots
    )
    assert len(partitions) == 100
    assert set(partitions.values()) == {2}
    assert {slot["slot_kind"] for slot in slots} == {"blind_holdout"}
    assert all(slot["partition_ordinal"] in {1, 2} for slot in slots)


def test_only_three_validated_controls_are_ready_and_197_are_explicitly_missing(
    checked: dict[str, object],
) -> None:
    metrics = checked["aggregate_metrics"]
    assert metrics["counts"] == {
        "registered_slots": 200,
        "implemented_controls": 3,
        "ready_slots": 3,
        "missing_slots": 197,
        "invalid_slots": 0,
    }
    assert metrics["coverage"] == {"coverage_numerator": 3, "coverage_denominator": 200}
    assert metrics["outcomes_from_validated_controls_only"] == {
        "pass": 3,
        "reject": 0,
        "blocked": 0,
    }
    status_counts = Counter(slot["status"] for slot in checked["slots"])
    assert status_counts == {"missing": 197, "ready": 3}
    assert all(
        slot["reason"] == "no_immutable_result_registered"
        and slot["immutable_manifest_expectation"]
        == {
            "required": True,
            "registration_status": "missing",
            "artifact_path": None,
            "file_sha256": None,
            "content_sha256": None,
        }
        and slot["implemented"] is False
        for slot in checked["slots"]
        if slot["status"] == "missing"
    )
    assert {slot["slot_id"] for slot in checked["slots"] if slot["status"] == "ready"} == {
        "historical.arithmetic.theorem_rediscovery.l1.001",
        "synthetic.algebra.theorem_rediscovery.l1.001",
        "synthetic.combinatorics.theorem_rediscovery.l2.001",
    }


def test_aggregate_schema_partitions_and_no_false_success(checked: dict[str, object]) -> None:
    assert checked["decision"] == "not_ready_missing_or_invalid_benchmarks"
    assert checked["curriculum_success"] is False
    assert checked["claims"] == {
        "closed_unique_registry_validated": True,
        "only_implemented_validated_benchmarks_marked_ready": True,
        "missing_benchmarks_count_as_success": False,
        "unregistered_results_admitted": False,
        "all_registered_benchmarks_implemented": False,
        "curriculum_complete": False,
    }
    partitions = checked["aggregate_metrics"]["partitions"]
    assert {key: len(rows) for key, rows in partitions.items()} == {
        "cohort": 2,
        "domain": 5,
        "artifact_type": 2,
        "level": 5,
    }
    for rows in partitions.values():
        for row in rows:
            assert row["registered"] == row["ready"] + row["missing"] + row["invalid"]


def test_artifact_tamper_and_aggregate_tamper_fail_closed(
    tmp_path: Path, checked: dict[str, object]
) -> None:
    config = _load_config(ROOT)
    control = copy.deepcopy(config["implemented_controls"][0])
    original = ROOT / control["artifact_path"]
    local = tmp_path / "artifact.json"
    local.write_bytes(original.read_bytes())
    control["artifact_path"] = "artifact.json"
    slot = next(row for row in expand_slots(config) if row["slot_id"] == control["slot_id"])
    assert _inspect_implementation(tmp_path, slot, control)["status"] == "ready"
    payload = json.loads(local.read_text(encoding="utf-8"))
    payload["decision"] = "forged_success"
    payload["content_sha256"] = _canonical_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    local.write_text(json.dumps(payload), encoding="utf-8")
    inspected = _inspect_implementation(tmp_path, slot, control)
    assert inspected["status"] == "invalid"
    assert "file_sha256_mismatch" in inspected["validation_errors"]
    assert "decision_mismatch" in inspected["validation_errors"]

    forged = copy.deepcopy(checked)
    forged["curriculum_success"] = True
    forged["content_sha256"] = _canonical_sha(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="readiness contract changed"):
        validate_readiness(forged, ROOT)


def test_immutable_writer_is_idempotent_and_refuses_replacement(tmp_path: Path) -> None:
    path = tmp_path / "readiness.json"
    _write_immutable(path, {"state": "not_ready"})
    before = path.read_bytes()
    _write_immutable(path, {"state": "not_ready"})
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="differs"):
        _write_immutable(path, {"state": "ready"})
