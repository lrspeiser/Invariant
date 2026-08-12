from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.continuous_scientific_pipeline_candidate_followup import (
    CONFIG_REL,
    RESULT_REL,
    _sealed,
    build_result,
    load_config,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_REL
RESULT = ROOT / RESULT_REL


def test_checked_followup_is_exact_fail_closed_candidate_evidence() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    validate_result(value, ROOT, CONFIG)
    assert value["decision"] == (
        "candidate_specific_followup_blocked_no_comparable_evidence_no_promotion"
    )
    assert value["source_pass_batch_indices"] == [1, 2, 3, 5, 6, 7]
    assert value["counts"]["source_pass_batches"] == 6
    assert value["counts"]["durably_recorded_candidates"] == len(
        value["candidate_decision_records"]
    )
    assert (
        value["counts"]["source_survivor_candidates"]
        >= value["counts"]["durably_recorded_candidates"]
    )
    assert value["counts"]["candidate_blocks"] + value["counts"]["candidate_rejects"] == len(
        value["candidate_decision_records"]
    )
    assert value["counts"]["candidate_passes"] == 0
    assert value["counts"]["formal_passes"] == 0
    assert value["complete_comparable_evidence"] is False
    assert not any(value["promotion_contract"].values())
    assert not any(value["seals"].values())


def test_candidate_ordinals_and_evidence_roots_are_durable_and_unique() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = value["candidate_decision_records"]
    identities = [(row["candidate_id"], row["ordinal"]) for row in rows]
    assert len(identities) == len(set(identities))
    assert all(len(row["covariant_mapping_payload_sha256"]) == 64 for row in rows)
    assert all(row["decision"] in {"block", "reject"} for row in rows)
    assert len(value["candidate_decision_records_root_sha256"]) == 64
    assert len(value["batch_artifact_bindings"]) == 6


def test_completed_batches_are_reused_idempotently() -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))
    assert build_result(ROOT, CONFIG) == checked


def test_validation_never_opens_mutable_runtime_gpu_or_live_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def forbidden(path: Path) -> bool:
        normalized = path.as_posix()
        return (
            "continuous-scientific-pipeline-service-runtime-v2/epoch-003" in normalized
            or normalized.endswith("runs/campaigns/campaign-v1-live.sqlite")
            or "gpu-scheduler-runtime" in normalized
        )

    def guarded_text(path: Path, *args: object, **kwargs: object) -> str:
        if forbidden(path):
            raise AssertionError("follow-up validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if forbidden(path):
            raise AssertionError("follow-up validator opened excluded mutable state")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    validate_result(value, ROOT, CONFIG)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("candidate_passes", 1),
        lambda value: value["candidate_decision_records"][0].__setitem__("decision", "pass"),
        lambda value: value["promotion_contract"].__setitem__(
            "leaderboard_rebuild_requested", True
        ),
        lambda value: value["batch_artifact_bindings"][0].__setitem__("content_sha256", "0" * 64),
    ],
)
def test_result_tampering_fails_closed(mutation: object) -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    body = copy.deepcopy(value)
    body.pop("content_sha256")
    mutation(body)  # type: ignore[operator]
    with pytest.raises(ValueError):
        validate_result(_sealed(body), ROOT, CONFIG)


def test_config_fixes_owned_child_deadlines_and_zero_gpu() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["maximum_generation_seconds_per_batch"] == 120
    assert config["maximum_formal_seconds_per_batch"] == 120
    assert config["maximum_total_seconds"] == 1440
    assert config["resource_contract"] == {
        "cpu_workers": 15,
        "gpu_workers": 0,
        "live_campaign_SQLite_access": False,
        "observations_opened": False,
        "external_process_signals": False,
        "leaderboard_or_rank_writes": False,
    }
