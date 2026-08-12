from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_survivor_pagination as pagination
from sigma_theory_compiler.continuous_scientific_pipeline_survivor_pagination import (
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


def test_checked_pagination_is_complete_and_formal_queue_is_fail_closed() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    validate_result(value, ROOT, CONFIG)
    assert value["decision"] == "complete_survivor_pagination_formal_queue_incomplete_no_promotion"
    assert value["counts"]["source_pass_batches"] == 6
    assert value["counts"]["source_survivors"] == 11_439
    assert value["counts"]["durable_formal_receipt_queue_entries"] == 11_439
    assert value["counts"]["preserved_completed_formal_receipts"] == 192
    assert value["counts"]["pending_formal_receipts"] == 11_247
    assert value["counts"]["worker_roots"] == 90
    assert value["complete_survivor_pagination"] is True
    assert value["complete_formal_receipts"] is False
    assert value["complete_comparable_evidence"] is False
    assert not any(value["promotion_contract"].values())
    assert not any(value["seals"].values())


def test_every_leaf_is_complete_bounded_and_queue_bound() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    ordinals: list[int] = []
    completed = pending = leaves = 0
    for batch_binding in result["ordered_batch_index_bindings"]:
        batch = json.loads((ROOT / batch_binding["path"]).read_text(encoding="utf-8"))
        stack = list(reversed(batch["ordered_worker_root_bindings"]))
        while stack:
            binding = stack.pop()
            node = json.loads((ROOT / binding["path"]).read_text(encoding="utf-8"))
            if not node["leaf_page"]:
                stack.extend(reversed(node["child_bindings"]))
                continue
            leaves += 1
            manifest = node["candidate_manifest"]
            queue = node["formal_receipt_queue"]
            assert manifest["sample_complete"] is True
            assert len(manifest["survivor_records"]) <= 32
            assert len(queue) == len(manifest["survivor_records"])
            ordinals.extend(row["ordinal"] for row in queue)
            completed += sum(row["state"] == "completed_preserved_formal_reject" for row in queue)
            pending += sum(
                row["state"] == "pending_candidate_specific_formal_receipt" for row in queue
            )
    assert leaves == result["counts"]["leaf_pages"]
    assert ordinals == sorted(set(ordinals))
    assert len(ordinals) == 11_439
    assert (completed, pending) == (192, 11_247)


def test_completed_nodes_are_reused_without_new_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    checked = json.loads(RESULT.read_text(encoding="utf-8"))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed pagination attempted to execute a new page")

    monkeypatch.setattr(pagination, "_evaluate_manifest", forbidden)
    assert build_result(ROOT, CONFIG) == checked


def test_atomic_immutable_writer_reuses_exact_bytes_and_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "page.json"
    pagination._write_atomic_immutable(path, {"value": 1}, 1024)
    pagination._write_atomic_immutable(path, {"value": 1}, 1024)
    with pytest.raises(ValueError, match="immutable pagination artifact differs"):
        pagination._write_atomic_immutable(path, {"value": 2}, 1024)
    assert not list(tmp_path.glob(".page.json.*"))


def test_generation_payload_resolves_generator_against_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genesis = json.loads(
        (ROOT / "runs/engine/continuous-scientific-pipeline-epoch-003-genesis.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.chdir(tmp_path)
    payload = pagination._payload(ROOT, genesis, 1004504576, 1004508672)
    assert Path(payload["generator_config_path"]).is_absolute()
    assert Path(payload["generator_config_path"]) == ROOT / "configs/generator_v2_billion.json"


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
            raise AssertionError("pagination validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if forbidden(path):
            raise AssertionError("pagination validator opened excluded mutable state")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    validate_result(value, ROOT, CONFIG)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("pending_formal_receipts", 0),
        lambda value: value.__setitem__("complete_formal_receipts", True),
        lambda value: value["promotion_contract"].__setitem__(
            "leaderboard_rebuild_requested", True
        ),
        lambda value: value["ordered_batch_index_bindings"][0].__setitem__(
            "content_sha256", "0" * 64
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


def test_config_fixes_caps_deadlines_atomic_resume_and_zero_gpu() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["worker_ordinal_span"] == 32_768
    assert config["maximum_survivors_per_leaf_page"] == 32
    assert config["maximum_node_seconds"] == 60
    assert config["maximum_total_seconds"] == 900
    assert config["resource_contract"] == {
        "cpu_workers": 1,
        "gpu_workers": 0,
        "campaign_owned_child_isolation": True,
        "live_campaign_SQLite_access": False,
        "observations_opened": False,
        "external_process_signals": False,
        "leaderboard_or_rank_writes": False,
    }
