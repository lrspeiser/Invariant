from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_cumulative_formal_partition_0007 as worker
from sigma_theory_compiler.continuous_scientific_pipeline_cumulative_formal_partition_0007 import (
    CONFIG_REL,
    RESULT_REL,
    _leaf_catalog,
    _load_bound_json,
    _predecessor_state,
    _sealed,
    _selected_entry,
    _validate_processed_prefix,
    build_result,
    load_config,
    validate_result,
)
from sigma_theory_compiler.continuous_scientific_pipeline_cumulative_formal_receipt_worker import (
    _validate_preserved_receipts,
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


def test_checked_partition_0007_extends_exact_prefix_without_promotion() -> None:
    result, ledger, partition = _checked()
    validate_result(result, ROOT, CONFIG)
    assert result["decision"] == (
        "cumulative_formal_receipt_prefix_advanced_to_partition_0007_no_promotion"
    )
    assert result["counts"] == {
        "processed_partition_prefix_length": 7,
        "processed_leaf_pages": 7,
        "cumulative_formally_checked_candidates": 104,
        "cumulative_newly_processed_candidates": 102,
        "cumulative_reconciled_preserved_candidates": 2,
        "cumulative_candidate_rejects": 104,
        "cumulative_candidate_blocks": 0,
        "cumulative_candidate_passes": 0,
        "cumulative_formal_passes": 0,
        "initial_pending_formal_receipts": 11_247,
        "remaining_pending_formal_receipts": 11_145,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    assert partition["counts"]["partition_candidates"] == 20
    assert partition["counts"]["newly_processed_candidates"] == 20
    assert partition["counts"]["candidate_rejects"] == 20
    assert ledger["complete_processed_partition_prefix"] is True
    assert result["complete_global_formal_receipts"] is False
    assert result["complete_comparable_evidence"] is False
    assert not any(result["promotion_contract"].values())
    assert not any(result["seals"].values())


def test_partition_0007_leaf_is_deterministic_and_merkle_reachable() -> None:
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    predecessor = _load_bound_json(ROOT, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    _, summaries, _ = _predecessor_state(ROOT, predecessor, catalog)
    entry = _selected_entry(catalog, summaries)
    assert entry["catalog_index"] == 6
    assert entry["leaf_binding"]["path"].endswith("node-1004908032-1004909056.json")
    assert entry["pending_candidate_count"] == 20
    assert entry["leaf_queue_root_sha256"] == (
        "bfa4d8bc2d105181043ac70b05047f5c9fb7a43c1e953050348e86e3f7899522"
    )
    assert len(entry["hierarchy_path"]) == 7
    leaf = _load_bound_json(ROOT, entry["leaf_binding"])
    assert leaf["ordinal_interval"] == {
        "candidate_count": 1024,
        "start_ordinal": 1_004_908_032,
        "end_ordinal_exclusive": 1_004_909_056,
    }


def test_predecessor_ledger_is_preserved_verbatim() -> None:
    _, ledger, _ = _checked()
    config = load_config(ROOT, CONFIG)
    predecessor = _load_bound_json(ROOT, config["predecessor_cumulative_result"])
    previous = _load_bound_json(ROOT, predecessor["cumulative_ledger_binding"])
    assert ledger["processed_partition_summaries"][:6] == previous["processed_partition_summaries"]
    assert (
        ledger["cumulative_formal_receipt_records"][:84]
        == previous["cumulative_formal_receipt_records"]
    )
    assert (
        ledger["predecessor_cumulative_ledger_root_sha256"]
        == previous["cumulative_formal_receipt_ledger_root_sha256"]
    )


def test_selected_leaf_overlap_is_rejected() -> None:
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    predecessor = _load_bound_json(ROOT, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    _, summaries, _ = _predecessor_state(ROOT, predecessor, catalog)
    transplanted = copy.deepcopy(catalog)
    transplanted[6]["leaf_binding"] = summaries[0]["leaf_binding"]
    with pytest.raises(ValueError, match="overlaps predecessor"):
        _selected_entry(transplanted, summaries)


@pytest.mark.parametrize("mutation", ["overlap", "gap"])
def test_three_partition_prefix_rejects_overlap_or_gap(mutation: str) -> None:
    _, ledger, _ = _checked()
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    summaries = copy.deepcopy(ledger["processed_partition_summaries"])
    if mutation == "overlap":
        summaries[6]["leaf_catalog_index"] = 5
        summaries[6]["leaf_binding"] = summaries[5]["leaf_binding"]
    else:
        summaries[6]["leaf_catalog_index"] = 7
        summaries[6]["leaf_binding"] = catalog[7]["leaf_binding"]
    partitions = [
        _load_bound_json(ROOT, row["partition_binding"])
        for row in ledger["processed_partition_summaries"]
    ]
    with pytest.raises(ValueError, match="gap or overlap"):
        _validate_processed_prefix(ROOT, catalog, summaries, partitions)


def test_preserved_receipts_remain_reconciled_field_for_field() -> None:
    config = load_config(ROOT, CONFIG)
    predecessor = _load_bound_json(ROOT, config["predecessor_cumulative_result"])
    previous_ledger = _load_bound_json(ROOT, predecessor["cumulative_ledger_binding"])
    first_summary = previous_ledger["processed_partition_summaries"][0]
    first_partition = _load_bound_json(ROOT, first_summary["partition_binding"])
    first_leaf = _load_bound_json(ROOT, first_summary["leaf_binding"])
    mutated = copy.deepcopy(first_leaf)
    preserved = next(
        row
        for row in mutated["formal_receipt_queue"]
        if row["state"] == "completed_preserved_formal_reject"
    )
    preserved["formal_receipt_binding"]["formal_evidence_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="field reconciliation"):
        _validate_preserved_receipts(ROOT, mutated, first_partition)


def test_completed_partition_0007_resumes_without_child_execution(
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
            raise AssertionError("partition 0007 validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if forbidden(path):
            raise AssertionError("partition 0007 validator opened excluded mutable state")
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
            "candidate_promotion_performed", True
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


def test_config_fixes_one_leaf_cpu_only_cleanup_inclusive_bounds() -> None:
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
