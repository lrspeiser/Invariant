from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_cumulative_formal_receipt_worker as worker
from sigma_theory_compiler.continuous_scientific_pipeline_cumulative_formal_receipt_worker import (
    CONFIG_REL,
    RESULT_REL,
    _catalog_entry_for_binding,
    _leaf_catalog,
    _load_bound_json,
    _sealed,
    _validate_preserved_receipts,
    _validate_processed_prefix,
    build_result,
    load_config,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_REL
RESULT = ROOT / RESULT_REL


def _checked() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    ledger = json.loads(
        (ROOT / result["cumulative_ledger_binding"]["path"]).read_text(encoding="utf-8")
    )
    partition = json.loads(
        (ROOT / result["executed_partition_binding"]["path"]).read_text(encoding="utf-8")
    )
    return result, ledger, partition


def test_checked_cumulative_ledger_advances_exact_prefix_without_promotion() -> None:
    result, ledger, partition = _checked()
    validate_result(result, ROOT, CONFIG)
    assert result["decision"] == (
        "cumulative_formal_receipt_prefix_advanced_global_queue_incomplete_no_promotion"
    )
    assert result["counts"] == {
        "processed_partition_prefix_length": 2,
        "processed_leaf_pages": 2,
        "cumulative_formally_checked_candidates": 36,
        "cumulative_newly_processed_candidates": 34,
        "cumulative_reconciled_preserved_candidates": 2,
        "cumulative_candidate_rejects": 36,
        "cumulative_candidate_blocks": 0,
        "cumulative_candidate_passes": 0,
        "cumulative_formal_passes": 0,
        "initial_pending_formal_receipts": 11_247,
        "remaining_pending_formal_receipts": 11_213,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    assert partition["counts"]["partition_candidates"] == 12
    assert partition["counts"]["newly_processed_candidates"] == 12
    assert partition["counts"]["candidate_rejects"] == 12
    assert ledger["complete_processed_partition_prefix"] is True
    assert result["complete_global_formal_receipts"] is False
    assert result["complete_comparable_evidence"] is False
    assert not any(result["promotion_contract"].values())
    assert not any(result["seals"].values())


def test_next_leaf_is_deterministic_and_merkle_reachable() -> None:
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    assert catalog[0]["leaf_binding"]["path"].endswith("node-1004881408-1004881664.json")
    assert catalog[1]["leaf_binding"]["path"].endswith("node-1004881664-1004881920.json")
    assert catalog[1]["pending_candidate_count"] == 12
    assert len(catalog[1]["hierarchy_path"]) == 9
    leaf = _load_bound_json(ROOT, catalog[1]["leaf_binding"])
    assert leaf["ordinal_interval"] == {
        "candidate_count": 256,
        "start_ordinal": 1004881664,
        "end_ordinal_exclusive": 1004881920,
    }


def test_transplanted_leaf_binding_is_rejected() -> None:
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    transplanted = dict(catalog[1]["leaf_binding"])
    transplanted["path"] = "runs/engine/transplanted-cumulative-leaf.json"
    with pytest.raises(ValueError, match="transplanted or absent"):
        _catalog_entry_for_binding(catalog, transplanted)


@pytest.mark.parametrize("mutation", ["overlap", "gap"])
def test_processed_partition_prefix_rejects_leaf_overlap_or_gap(mutation: str) -> None:
    result, ledger, _ = _checked()
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    summaries = copy.deepcopy(ledger["processed_partition_summaries"])
    if mutation == "overlap":
        summaries[1]["leaf_catalog_index"] = 0
        summaries[1]["leaf_binding"] = summaries[0]["leaf_binding"]
    else:
        summaries[1]["leaf_catalog_index"] = 2
        summaries[1]["leaf_binding"] = catalog[2]["leaf_binding"]
    partitions = [
        _load_bound_json(ROOT, row["partition_binding"])
        for row in ledger["processed_partition_summaries"]
    ]
    with pytest.raises(ValueError, match="gap or overlap"):
        _validate_processed_prefix(ROOT, catalog, summaries, partitions)
    assert result["counts"]["processed_partition_prefix_length"] == 2


def test_preserved_receipts_are_reconciled_field_for_field() -> None:
    config = load_config(ROOT, CONFIG)
    predecessor = _load_bound_json(ROOT, config["predecessor_worker_result"])
    partition = _load_bound_json(ROOT, predecessor["partition_result_binding"])
    leaf = _load_bound_json(ROOT, predecessor["selected_leaf_page_binding"])
    mutated = copy.deepcopy(leaf)
    preserved = next(
        row
        for row in mutated["formal_receipt_queue"]
        if row["state"] == "completed_preserved_formal_reject"
    )
    preserved["formal_receipt_binding"]["formal_evidence_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="field reconciliation"):
        _validate_preserved_receipts(ROOT, mutated, partition)


def test_completed_successor_resumes_without_child_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("resume attempted a new formal child")

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
            raise AssertionError("cumulative validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if forbidden(path):
            raise AssertionError("cumulative validator opened excluded mutable state")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    validate_result(value, ROOT, CONFIG)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("remaining_pending_formal_receipts", 0),
        lambda value: value.__setitem__("complete_global_formal_receipts", True),
        lambda value: value.__setitem__("complete_comparable_evidence", True),
        lambda value: value["promotion_contract"].__setitem__(
            "leaderboard_rebuild_requested", True
        ),
        lambda value: value.__setitem__("cumulative_formal_receipt_ledger_root_sha256", "0" * 64),
    ],
)
def test_result_tamper_or_forged_promotion_fails_closed(mutation: object) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))
    body = copy.deepcopy(checked)
    body.pop("content_sha256")
    mutation(body)  # type: ignore[operator]
    with pytest.raises(ValueError):
        validate_result(_sealed(body), ROOT, CONFIG)


def test_config_fixes_one_page_cpu_only_hard_bounds() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["maximum_partitions_per_invocation"] == 1
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
