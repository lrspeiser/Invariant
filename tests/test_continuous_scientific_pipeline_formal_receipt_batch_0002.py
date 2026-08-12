from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_batch_0002 as worker
import sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_batch_worker as engine
from sigma_theory_compiler.continuous_scientific_pipeline_cumulative_formal_receipt_worker import (
    _validate_processed_prefix,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / worker.CONFIG_REL
RESULT = ROOT / worker.RESULT_REL


def _checked() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    cursor = worker._load_bound_json(ROOT, result["cumulative_ledger_binding"])
    leaves = [
        worker._load_bound_json(ROOT, binding) for binding in result["executed_leaf_bindings"]
    ]
    return result, cursor, leaves


def _selection() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    config = worker.load_config(ROOT, CONFIG)
    pagination = worker._load_bound_json(ROOT, config["pagination_result"])
    predecessor = worker._load_bound_json(ROOT, config["predecessor_cumulative_result"])
    catalog = worker._leaf_catalog(ROOT, pagination)
    _, summaries, _ = worker._predecessor_state(ROOT, predecessor, catalog)
    return config, catalog, worker._select_entries(catalog, summaries, config)


def test_checked_successor_advances_prefix_by_two_leaves_without_promotion() -> None:
    result, cursor, leaves = _checked()
    worker.validate_result(result, ROOT, CONFIG)
    assert result["batch_leaf_catalog_indices"] == [9, 10]
    assert [leaf["counts"]["partition_candidates"] for leaf in leaves] == [22, 23]
    assert sum(leaf["counts"]["newly_processed_candidates"] for leaf in leaves) == 45
    assert sum(leaf["counts"]["candidate_rejects"] for leaf in leaves) == 45
    assert result["counts"] == {
        "processed_partition_prefix_length": 11,
        "processed_leaf_pages": 11,
        "cumulative_formally_checked_candidates": 180,
        "cumulative_newly_processed_candidates": 178,
        "cumulative_reconciled_preserved_candidates": 2,
        "cumulative_candidate_rejects": 180,
        "cumulative_candidate_blocks": 0,
        "cumulative_candidate_passes": 0,
        "cumulative_formal_passes": 0,
        "initial_pending_formal_receipts": 11_247,
        "remaining_pending_formal_receipts": 11_069,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    assert cursor["complete_processed_partition_prefix"] is True
    assert result["complete_global_formal_receipts"] is False
    assert result["complete_comparable_evidence"] is False
    assert result["first_remaining_blocker"] == ("11069_candidate_specific_formal_receipts_pending")
    assert not any(result["promotion_contract"].values())
    assert not any(result["seals"].values())


def test_selection_is_deterministic_bounded_and_merkle_reachable() -> None:
    config, _, selected = _selection()
    assert [row["catalog_index"] for row in selected] == [9, 10]
    assert [row["pending_candidate_count"] for row in selected] == [22, 23]
    assert sum(row["pending_candidate_count"] for row in selected) == 45
    assert [row["leaf_queue_root_sha256"] for row in selected] == [
        "f9027925833336c1fba2925c0c2c1069959383de47503c747d4fde62ee516622",
        "efe989d9489e73abd09939421e0a4094cce12afa6c64c697170c2f101832c7e7",
    ]
    assert [len(row["hierarchy_path"]) for row in selected] == [7, 7]
    assert len(selected) <= config["maximum_leaves_per_invocation"]


def test_predecessor_lineage_and_receipts_are_preserved_verbatim() -> None:
    result, cursor, _ = _checked()
    predecessor = worker._load_bound_json(ROOT, result["predecessor_cumulative_result_binding"])
    previous = worker._load_bound_json(ROOT, predecessor["cumulative_ledger_binding"])
    assert cursor["processed_partition_summaries"][:9] == previous["processed_partition_summaries"]
    assert (
        cursor["cumulative_formal_receipt_records"][:135]
        == previous["cumulative_formal_receipt_records"]
    )
    assert (
        cursor["predecessor_cumulative_ledger_root_sha256"]
        == previous["cumulative_formal_receipt_ledger_root_sha256"]
    )


def test_new_receipts_are_disjoint_and_source_bound() -> None:
    _, _, leaves = _checked()
    ordinals: list[int] = []
    for leaf in leaves:
        receipts = leaf["candidate_formal_receipts"]
        assert leaf["candidate_formal_receipts_root_sha256"] == engine._sha(receipts)
        assert all(row["newly_processed"] is True for row in receipts)
        assert all(
            row["source_queue_state"] == "pending_candidate_specific_formal_receipt"
            for row in receipts
        )
        ordinals.extend(row["ordinal"] for row in receipts)
    assert len(ordinals) == len(set(ordinals)) == 45


@pytest.mark.parametrize("mutation", ["overlap", "gap"])
def test_prefix_rejects_overlap_and_gap(mutation: str) -> None:
    _, cursor, _ = _checked()
    _, catalog, _ = _selection()
    summaries = copy.deepcopy(cursor["processed_partition_summaries"])
    if mutation == "overlap":
        summaries[10]["leaf_catalog_index"] = 9
        summaries[10]["leaf_binding"] = summaries[9]["leaf_binding"]
    else:
        summaries[10]["leaf_catalog_index"] = 11
        summaries[10]["leaf_binding"] = catalog[11]["leaf_binding"]
    partitions = [
        worker._load_bound_json(ROOT, row["partition_binding"])
        for row in cursor["processed_partition_summaries"]
    ]
    with pytest.raises(ValueError, match="gap or overlap"):
        _validate_processed_prefix(ROOT, catalog, summaries, partitions)


def test_completed_successor_resumes_without_child_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("immutable resume attempted a new formal child")

    monkeypatch.setattr(engine, "_run_owned_child", forbidden)
    assert worker.build_result(ROOT, CONFIG) == checked


def test_total_deadline_fails_closed_before_child_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, catalog, selected = _selection()
    leaves = [worker._load_bound_json(ROOT, row["leaf_binding"]) for row in selected]
    monkeypatch.setattr(engine, "_artifact_path", lambda *_args: tmp_path / "missing.json")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("expired total deadline started a formal child")

    monkeypatch.setattr(engine, "_run_owned_child", forbidden)
    with pytest.raises(TimeoutError, match="total wall-clock bound"):
        engine._build_leaf_partitions(
            ROOT,
            config,
            selected,
            leaves,
            {},
            time.monotonic() - config["maximum_total_seconds"],
        )
    assert len(catalog) > len(selected)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("remaining_pending_formal_receipts", 0),
        lambda value: value.__setitem__("batch_leaf_catalog_indices", [9, 11]),
        lambda value: value.__setitem__("complete_global_formal_receipts", True),
        lambda value: value.__setitem__("complete_comparable_evidence", True),
        lambda value: value["promotion_contract"].__setitem__(
            "candidate_promotion_performed", True
        ),
        lambda value: value.__setitem__("cumulative_formal_receipt_ledger_root_sha256", "0" * 64),
    ],
)
def test_resealed_tamper_or_forged_promotion_fails_closed(mutation: object) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))
    body = copy.deepcopy(checked)
    body.pop("content_sha256")
    mutation(body)  # type: ignore[operator]
    with pytest.raises(ValueError):
        worker.validate_result(worker._sealed(body), ROOT, CONFIG)


def test_validation_avoids_runtime_gpu_sqlite_and_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def excluded(path: Path) -> bool:
        normalized = path.as_posix().lower()
        return any(
            token in normalized
            for token in (
                "continuous-scientific-pipeline-service-runtime-v2/epoch-003",
                "campaign-v1-live.sqlite",
                "gpu-scheduler-runtime",
                "parallel-supervisor",
            )
        )

    def guarded_text(path: Path, *args: object, **kwargs: object) -> str:
        if excluded(path):
            raise AssertionError("validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if excluded(path):
            raise AssertionError("validator opened excluded mutable state")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    worker.validate_result(value, ROOT, CONFIG)


def test_config_preserves_owned_child_cpu_only_caps() -> None:
    config = worker.load_config(ROOT, CONFIG)
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
