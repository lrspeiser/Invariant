from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_batch_worker as worker
from sigma_theory_compiler.continuous_scientific_pipeline_cumulative_formal_receipt_worker import (
    _leaf_catalog,
    _validate_processed_prefix,
)
from sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_batch_worker import (
    CONFIG_REL,
    RESULT_REL,
    _load_bound_json,
    _predecessor_state,
    _sealed,
    _select_entries,
    build_result,
    load_config,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_REL
RESULT = ROOT / RESULT_REL


def _checked() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    cursor = _load_bound_json(ROOT, result["cumulative_ledger_binding"])
    leaves = [_load_bound_json(ROOT, binding) for binding in result["executed_leaf_bindings"]]
    return result, cursor, leaves


def test_checked_batch_advances_two_leaves_without_promotion() -> None:
    result, cursor, leaves = _checked()
    validate_result(result, ROOT, CONFIG)
    assert result["decision"] == (
        "formal_receipt_cursor_advanced_by_bounded_multi_leaf_batch_no_promotion"
    )
    assert result["batch_leaf_catalog_indices"] == [7, 8]
    assert [leaf["counts"]["partition_candidates"] for leaf in leaves] == [24, 7]
    assert sum(leaf["counts"]["newly_processed_candidates"] for leaf in leaves) == 31
    assert result["counts"] == {
        "processed_partition_prefix_length": 9,
        "processed_leaf_pages": 9,
        "cumulative_formally_checked_candidates": 135,
        "cumulative_newly_processed_candidates": 133,
        "cumulative_reconciled_preserved_candidates": 2,
        "cumulative_candidate_rejects": 135,
        "cumulative_candidate_blocks": 0,
        "cumulative_candidate_passes": 0,
        "cumulative_formal_passes": 0,
        "initial_pending_formal_receipts": 11_247,
        "remaining_pending_formal_receipts": 11_114,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    assert cursor["complete_processed_partition_prefix"] is True
    assert result["complete_global_formal_receipts"] is False
    assert result["complete_comparable_evidence"] is False
    assert not any(result["promotion_contract"].values())
    assert not any(result["seals"].values())


def test_batch_selection_is_deterministic_bounded_and_merkle_reachable() -> None:
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    predecessor = _load_bound_json(ROOT, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    _, summaries, _ = _predecessor_state(ROOT, predecessor, catalog)
    selected = _select_entries(catalog, summaries, config)
    assert [row["catalog_index"] for row in selected] == [7, 8]
    assert [row["pending_candidate_count"] for row in selected] == [24, 7]
    assert sum(row["pending_candidate_count"] for row in selected) == 31
    assert [row["leaf_queue_root_sha256"] for row in selected] == [
        "786ba7da8db3763c410137d7aed9d14f7a5b2eaf792dc0f145359b4a7f46f182",
        "e24e5f4ef3e13af3ac9d06a8361074b93f613f0d2463765676725e9aa307c21f",
    ]
    assert [len(row["hierarchy_path"]) for row in selected] == [7, 5]


def test_predecessor_cursor_and_receipts_are_preserved_verbatim() -> None:
    result, cursor, _ = _checked()
    predecessor = _load_bound_json(ROOT, result["predecessor_cumulative_result_binding"])
    previous = _load_bound_json(ROOT, predecessor["cumulative_ledger_binding"])
    assert cursor["processed_partition_summaries"][:7] == previous["processed_partition_summaries"]
    assert (
        cursor["cumulative_formal_receipt_records"][:104]
        == previous["cumulative_formal_receipt_records"]
    )
    assert (
        cursor["predecessor_cumulative_ledger_root_sha256"]
        == previous["cumulative_formal_receipt_ledger_root_sha256"]
    )


def test_new_leaf_receipts_are_disjoint_and_source_bound() -> None:
    _, _, leaves = _checked()
    ordinals: list[int] = []
    for leaf in leaves:
        receipts = leaf["candidate_formal_receipts"]
        assert leaf["candidate_formal_receipts_root_sha256"] == worker._sha(receipts)
        assert all(row["newly_processed"] is True for row in receipts)
        assert all(
            row["source_queue_state"] == "pending_candidate_specific_formal_receipt"
            for row in receipts
        )
        ordinals.extend(row["ordinal"] for row in receipts)
    assert len(ordinals) == len(set(ordinals)) == 31


def test_selection_rejects_transplanted_overlap() -> None:
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    predecessor = _load_bound_json(ROOT, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    _, summaries, _ = _predecessor_state(ROOT, predecessor, catalog)
    transplanted = copy.deepcopy(catalog)
    transplanted[7]["leaf_binding"] = summaries[0]["leaf_binding"]
    with pytest.raises(ValueError, match="overlaps predecessor"):
        _select_entries(transplanted, summaries, config)


@pytest.mark.parametrize("mutation", ["overlap", "gap"])
def test_cursor_prefix_rejects_overlap_or_gap(mutation: str) -> None:
    _, cursor, _ = _checked()
    config = load_config(ROOT, CONFIG)
    pagination = _load_bound_json(ROOT, config["pagination_result"])
    catalog = _leaf_catalog(ROOT, pagination)
    summaries = copy.deepcopy(cursor["processed_partition_summaries"])
    if mutation == "overlap":
        summaries[8]["leaf_catalog_index"] = 7
        summaries[8]["leaf_binding"] = summaries[7]["leaf_binding"]
    else:
        summaries[8]["leaf_catalog_index"] = 9
        summaries[8]["leaf_binding"] = catalog[9]["leaf_binding"]
    partitions = [
        _load_bound_json(ROOT, row["partition_binding"])
        for row in cursor["processed_partition_summaries"]
    ]
    with pytest.raises(ValueError, match="gap or overlap"):
        _validate_processed_prefix(ROOT, catalog, summaries, partitions)


def test_completed_batch_resumes_without_child_execution(
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
            raise AssertionError("batch validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if forbidden(path):
            raise AssertionError("batch validator opened excluded mutable state")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    validate_result(value, ROOT, CONFIG)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("remaining_pending_formal_receipts", 0),
        lambda value: value.__setitem__("batch_leaf_catalog_indices", [7, 9]),
        lambda value: value.__setitem__("complete_global_formal_receipts", True),
        lambda value: value.__setitem__("complete_comparable_evidence", True),
        lambda value: value["promotion_contract"].__setitem__(
            "candidate_promotion_performed", True
        ),
        lambda value: value.__setitem__("cumulative_formal_receipt_ledger_root_sha256", "0" * 64),
    ],
)
def test_tamper_or_forged_promotion_fails_closed(mutation: object) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))
    body = copy.deepcopy(checked)
    body.pop("content_sha256")
    mutation(body)  # type: ignore[operator]
    with pytest.raises(ValueError):
        validate_result(_sealed(body), ROOT, CONFIG)


def test_config_fixes_multi_leaf_single_child_cpu_only_bounds() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["maximum_leaves_per_invocation"] == 2
    assert config["maximum_candidates_per_leaf"] == 32
    assert config["maximum_candidates_per_invocation"] == 64
    assert config["maximum_formal_seconds"] == 120
    assert config["maximum_total_seconds"] == 180
    assert config["resource_gate"] == {
        "cpu_utilization_below_percent": 92,
        "minimum_available_ram_mib": 32768,
        "cpu_workers": 1,
        "gpu_workers": 0,
    }
    assert not any(config["seals"].values())
