from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_worker as worker
from sigma_theory_compiler.continuous_formula_formal_backend import validate_formal_evidence
from sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_worker import (
    CONFIG_REL,
    RESULT_REL,
    _candidate_receipts,
    _sealed,
    _selected_leaf_hierarchy_path,
    build_result,
    load_config,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_REL
RESULT = ROOT / RESULT_REL


def test_checked_partition_has_exact_fail_closed_receipt_accounting() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    validate_result(value, ROOT, CONFIG)
    assert value["decision"] == (
        "bounded_partition_formal_receipts_complete_global_queue_incomplete_no_promotion"
    )
    assert value["counts"] == {
        "global_survivor_candidates": 11_439,
        "global_pending_formal_receipts_before_partition": 11_247,
        "partition_candidates": 24,
        "newly_processed_candidates": 22,
        "reconciled_preserved_candidates": 2,
        "candidate_rejects": 24,
        "candidate_blocks": 0,
        "candidate_passes": 0,
        "formal_passes": 0,
        "global_pending_formal_receipts_after_partition": 11_225,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    assert value["complete_partition_formal_receipts"] is True
    assert value["complete_global_formal_receipts"] is False
    assert value["complete_comparable_evidence"] is False
    assert value["pending_count_semantics"] == (
        "partition_overlay_no_updated_global_queue_root"
    )
    assert len(value["selected_leaf_hierarchy_path"]) >= 3
    assert value["selected_leaf_hierarchy_path"][-1] == value["selected_leaf_page_binding"]
    assert not any(value["promotion_contract"].values())
    assert not any(value["seals"].values())


def test_partition_receipts_bind_every_leaf_candidate_and_formal_record() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    partition = json.loads(
        (ROOT / result["partition_result_binding"]["path"]).read_text(encoding="utf-8")
    )
    evidence = partition["formal_evidence"]
    validate_formal_evidence(evidence)
    receipts = partition["candidate_formal_receipts"]
    assert len(receipts) == len(evidence["candidate_records"]) == 24
    assert [row["ordinal"] for row in receipts] == sorted({row["ordinal"] for row in receipts})
    assert sum(row["newly_processed"] for row in receipts) == 22
    assert all(row["decision"] == "reject" for row in receipts)
    assert all(len(row["formal_evidence_record_sha256"]) == 64 for row in receipts)
    assert sum(row["prior_formal_receipt_binding_sha256"] is not None for row in receipts) == 2
    assert partition["candidate_manifest"]["sample_complete"] is True
    assert partition["source_leaf_formal_receipt_queue_root_sha256"] == (
        "fecc6dd169e0cf8f09bc0be602ca2d7361c9acca9671bc53eb3590685bd86640"
    )


def test_selected_leaf_requires_registered_hierarchy_path() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    pagination = json.loads(
        (ROOT / result["pagination_result_binding"]["path"]).read_text(encoding="utf-8")
    )
    selected = json.loads(
        (ROOT / result["selected_leaf_page_binding"]["path"]).read_text(encoding="utf-8")
    )
    transplanted = dict(result["selected_leaf_page_binding"])
    transplanted["path"] = "runs/engine/transplanted-formal-receipt-leaf.json"
    with pytest.raises(ValueError, match="not reachable"):
        _selected_leaf_hierarchy_path(ROOT, pagination, transplanted, selected)


def test_preserved_receipts_must_reconcile_decision_and_blocker() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    leaf = json.loads(
        (ROOT / result["selected_leaf_page_binding"]["path"]).read_text(encoding="utf-8")
    )
    partition = json.loads(
        (ROOT / result["partition_result_binding"]["path"]).read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(leaf)
    mutated["formal_receipt_queue"][0]["formal_receipt_binding"]["decision"] = "block"
    with pytest.raises(ValueError, match="reconciliation mismatch"):
        _candidate_receipts(mutated, partition["formal_evidence"], partition["formal_receipt"])


def test_completed_partition_is_reused_without_new_formal_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed partition attempted formal execution")

    monkeypatch.setattr(worker, "_run_owned_child", forbidden)
    assert build_result(ROOT, CONFIG) == checked


def test_validation_never_opens_runtime_gpu_live_sqlite_or_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def forbidden(path: Path) -> bool:
        normalized = path.as_posix().lower()
        return any(
            token in normalized
            for token in (
                "continuous-scientific-pipeline-service-runtime-v2/epoch-003",
                "runs/campaigns/campaign-v1-live.sqlite",
                "gpu-scheduler-runtime",
                "parallel-supervisor",
            )
        )

    def guarded_text(path: Path, *args: object, **kwargs: object) -> str:
        if forbidden(path):
            raise AssertionError("formal receipt validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if forbidden(path):
            raise AssertionError("formal receipt validator opened excluded mutable state")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    validate_result(value, ROOT, CONFIG)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__(
            "global_pending_formal_receipts_after_partition", 0
        ),
        lambda value: value.__setitem__("complete_global_formal_receipts", True),
        lambda value: value["promotion_contract"].__setitem__(
            "leaderboard_rebuild_requested", True
        ),
        lambda value: value["processed_partition_roots"].__setitem__(
            "candidate_formal_receipts_root_sha256", "0" * 64
        ),
    ],
)
def test_result_tampering_fails_closed(mutation: object) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))
    body = copy.deepcopy(checked)
    body.pop("content_sha256")
    mutation(body)  # type: ignore[operator]
    with pytest.raises(ValueError):
        validate_result(_sealed(body), ROOT, CONFIG)


def test_config_preregisters_one_cpu_partition_with_hard_bounds() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["maximum_partition_candidates"] == 32
    assert config["maximum_formal_seconds"] == 120
    assert config["maximum_total_seconds"] == 180
    assert config["resource_gate"] == {
        "cpu_utilization_below_percent": 92,
        "minimum_available_ram_mib": 32768,
        "cpu_workers": 1,
        "gpu_workers": 0,
    }
    assert not any(config["seals"].values())
