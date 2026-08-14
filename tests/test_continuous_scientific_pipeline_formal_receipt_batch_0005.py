from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_batch_0005 as worker
import sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_batch_worker as engine

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


def _selection() -> tuple[dict[str, object], list[dict[str, object]]]:
    config = worker.load_config(ROOT, CONFIG)
    _, selected, _, _, _ = worker._selection(ROOT, config)
    return config, selected


def test_checked_successor_advances_two_more_leaves_without_promotion() -> None:
    result, cursor, leaves = _checked()
    worker.validate_result(result, ROOT, CONFIG)
    assert result["batch_leaf_catalog_indices"] == [15, 16]
    assert result["counts"]["processed_partition_prefix_length"] == 17
    assert result["counts"]["processed_leaf_pages"] == 17
    assert result["counts"]["cumulative_formally_checked_candidates"] == 314
    assert result["counts"]["cumulative_newly_processed_candidates"] == 312
    assert result["counts"]["cumulative_reconciled_preserved_candidates"] == 2
    assert result["counts"]["cumulative_candidate_rejects"] == 314
    assert result["counts"]["remaining_pending_formal_receipts"] == 10_935
    assert [leaf["counts"]["partition_candidates"] for leaf in leaves] == [28, 18]
    assert cursor["complete_processed_partition_prefix"] is True
    assert result["complete_global_formal_receipts"] is False
    assert result["complete_comparable_evidence"] is False
    assert result["first_remaining_blocker"] == "10935_candidate_specific_formal_receipts_pending"
    assert not any(result["promotion_contract"].values())
    assert not any(result["seals"].values())


def test_selection_is_deterministic_disjoint_bounded_and_merkle_reachable() -> None:
    config, selected = _selection()
    assert [row["catalog_index"] for row in selected] == [15, 16]
    assert [row["pending_candidate_count"] for row in selected] == [28, 18]
    assert sum(row["pending_candidate_count"] for row in selected) == 46
    assert len({row["leaf_binding"]["content_sha256"] for row in selected}) == 2
    assert all(row["hierarchy_path"] for row in selected)
    assert len(selected) <= config["maximum_leaves_per_invocation"]


def test_predecessor_lineage_and_receipts_are_preserved_verbatim() -> None:
    result, cursor, _ = _checked()
    predecessor = worker._load_bound_json(ROOT, result["predecessor_cumulative_result_binding"])
    previous = worker._load_bound_json(ROOT, predecessor["cumulative_ledger_binding"])
    assert cursor["processed_partition_summaries"][:15] == previous["processed_partition_summaries"]
    assert (
        cursor["cumulative_formal_receipt_records"][:268]
        == previous["cumulative_formal_receipt_records"]
    )
    assert (
        cursor["predecessor_cumulative_ledger_root_sha256"]
        == previous["cumulative_formal_receipt_ledger_root_sha256"]
    )


def test_new_receipts_are_disjoint_source_bound_and_all_rejected() -> None:
    _, _, leaves = _checked()
    ordinals: list[int] = []
    for leaf in leaves:
        receipts = leaf["candidate_formal_receipts"]
        assert leaf["candidate_formal_receipts_root_sha256"] == engine._sha(receipts)
        assert all(row["newly_processed"] is True for row in receipts)
        assert all(row["decision"] == "reject" for row in receipts)
        assert all(
            row["source_queue_state"] == "pending_candidate_specific_formal_receipt"
            for row in receipts
        )
        ordinals.extend(row["ordinal"] for row in receipts)
    assert len(ordinals) == len(set(ordinals)) == 46


def test_failed_current_admission_returns_typed_block_without_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, selected = _selection()
    sample = worker._sealed(
        {
            "schema_version": engine.PREFLIGHT_SCHEMA,
            "sampled_at": "2026-08-13T00:00:00+00:00",
            "cpu_utilization_percent": 92.0,
            "available_ram_mib": 74_107,
            "resource_gate": config["resource_gate"],
            "admitted": False,
            "gpu_or_cuda_probed": False,
            "processes_signaled": False,
        }
    )
    monkeypatch.setattr(worker, "RESULT_REL", "runs/engine/test-only-missing-result.json")
    # The config and its immutable predecessor chain were validated above. Keep
    # the test-only artifact-path hook scoped to this successor instead of
    # accidentally redirecting predecessor cursor bindings into tmp_path.
    monkeypatch.setattr(worker, "load_config", lambda *_args: config)
    monkeypatch.setattr(engine, "_artifact_path", lambda *_args: tmp_path / "preflight.json")
    monkeypatch.setattr(worker, "_admission_sample", lambda _config: sample)
    monkeypatch.setattr(
        engine,
        "_build_leaf_partitions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked admission started formal execution")
        ),
    )
    result = worker.build_result(ROOT, CONFIG)
    assert result["decision"] == worker.BLOCK_DECISION
    assert result["selected_leaf_catalog_indices"] == [row["catalog_index"] for row in selected]
    assert result["formal_execution_started"] is False
    assert not tmp_path.joinpath("preflight.json").exists()


def test_completed_successor_resumes_without_sampling_or_child_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("immutable resume attempted execution")

    monkeypatch.setattr(worker, "_admission_sample", forbidden)
    monkeypatch.setattr(engine, "_run_owned_child", forbidden)
    assert worker.build_result(ROOT, CONFIG) == checked


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("remaining_pending_formal_receipts", 0),
        lambda value: value.__setitem__("batch_leaf_catalog_indices", [15, 17]),
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


def test_validation_avoids_runtime_gpu_sqlite_observations_and_supervisor(
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
                "/observations/",
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


def test_config_preserves_strict_owned_child_cpu_only_caps() -> None:
    config = worker.load_config(ROOT, CONFIG)
    assert config["maximum_leaves_per_invocation"] == 2
    assert config["maximum_candidates_per_leaf"] == 32
    assert config["maximum_candidates_per_invocation"] == 64
    assert config["maximum_formal_seconds"] == 120
    assert config["maximum_total_seconds"] == 180
    assert config["resource_gate"] == {
        "cpu_utilization_below_percent": 92,
        "minimum_available_ram_mib": 32_768,
        "cpu_workers": 1,
        "gpu_workers": 0,
    }
    assert not any(config["seals"].values())
