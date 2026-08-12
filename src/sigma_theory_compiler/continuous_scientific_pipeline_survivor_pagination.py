"""Complete, immutable survivor pagination for Epoch 003 formal follow-up."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .continuous_formula_formal_backend import (
    extract_candidate_manifest,
    validate_candidate_manifest,
)
from .continuous_scientific_pipeline_candidate_followup import validate_result as validate_followup
from .continuous_scientific_pipeline_epoch import validate_epoch_genesis
from .continuous_scientific_pipeline_epoch_result import validate_epoch_result
from .continuous_scientific_pipeline_service import _run_owned_child
from .persistent_parallel_search import WorkLease
from .real_formula_execution import cpu_formula_batch_evaluator

CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-survivor-pagination-config-1.0"
NODE_SCHEMA = "sigma-continuous-scientific-pipeline-survivor-page-node-1.0"
BATCH_SCHEMA = "sigma-continuous-scientific-pipeline-survivor-page-index-1.0"
RESULT_SCHEMA = "sigma-continuous-scientific-pipeline-survivor-pagination-result-1.0"
DECISION = "complete_survivor_pagination_formal_queue_incomplete_no_promotion"
CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_survivor_pagination.json"
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_survivor_pagination.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_survivor_pagination.py"
RESULT_REL = "runs/engine/continuous-scientific-pipeline-epoch-003-survivor-pagination/result.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def _validate_sealed(value: Mapping[str, Any], label: str) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256", None)
    if claimed != _sha(body):
        raise ValueError(f"{label} content hash mismatch")


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} is not a JSON object")
    return value


def _load_bound_json(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("pagination source binding contract mismatch")
    path = _resolve(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("pagination source binding file hash mismatch")
    value = _load_json(path)
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("pagination source binding content hash mismatch")
    return value


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = _load_json(path)
    expected = {
        "schema_version",
        "campaign_id",
        "epoch_genesis",
        "epoch_terminal_result",
        "bounded_sample_followup",
        "artifact_directory",
        "pass_batch_indices",
        "worker_ordinal_span",
        "maximum_survivors_per_leaf_page",
        "maximum_node_seconds",
        "maximum_total_seconds",
        "maximum_artifact_bytes",
        "resource_contract",
    }
    resource = config.get("resource_contract")
    if (
        set(config) != expected
        or config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"] != "continuous-scientific-pipeline-epoch-003-survivor-pagination"
        or config["pass_batch_indices"] != [1, 2, 3, 5, 6, 7]
        or config["worker_ordinal_span"] != 32768
        or config["maximum_survivors_per_leaf_page"] != 32
        or config["maximum_node_seconds"] != 60
        or config["maximum_total_seconds"] != 900
        or config["maximum_artifact_bytes"] != 4_194_304
        or resource
        != {
            "cpu_workers": 1,
            "gpu_workers": 0,
            "campaign_owned_child_isolation": True,
            "live_campaign_SQLite_access": False,
            "observations_opened": False,
            "external_process_signals": False,
            "leaderboard_or_rank_writes": False,
        }
    ):
        raise ValueError("pagination config contract mismatch")
    _resolve(root, config["artifact_directory"])
    _load_bound_json(root, config["epoch_genesis"])
    _load_bound_json(root, config["epoch_terminal_result"])
    _load_bound_json(root, config["bounded_sample_followup"])
    return config


def _write_atomic_immutable(path: Path, value: Mapping[str, Any], maximum_bytes: int) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(raw) > maximum_bytes:
        raise RuntimeError("pagination artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"immutable pagination artifact differs: {path.name}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError(f"immutable pagination artifact differs: {path.name}") from None
    finally:
        temporary_path.unlink(missing_ok=True)


def _binding(root: Path, path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": _file_sha(path),
        "content_sha256": str(value["content_sha256"]),
    }


def _payload(root: Path, genesis: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    service = genesis["derived_service_config"]
    generator_path = _resolve(root, str(service["generator_config_path"]))
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    return {
        "generator_config_path": str(generator_path),
        "generator_config_sha256": _file_sha(generator_path),
        "protocol_version": generator["protocol_version"],
        "basis_count": generator["basis_count"],
        "max_action_terms": generator["max_action_terms"],
        "start_ordinal": start,
        "end_ordinal_exclusive": stop,
        "candidate_count": stop - start,
        "ambiguity_guard": service["ambiguity_guard"],
        "data_eligibility": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
        },
    }


def _manifest_worker(payload: Mapping[str, Any], output: Any) -> None:
    try:
        lease = WorkLease(
            f"epoch-003-page-{payload['start_ordinal']}",
            int(payload["start_ordinal"]),
            "cpu",
            int(payload["start_ordinal"]),
            1,
            1,
            dict(payload),
        )
        result = cpu_formula_batch_evaluator(lease)
        manifest = extract_candidate_manifest(payload, result)
        output.put({"ok": True, "result": {"manifest": manifest}})
    except Exception as error:  # noqa: BLE001 - owned child returns a bounded failure receipt
        output.put({"ok": False, "error": f"{type(error).__name__}: {error}"})


def _evaluate_manifest(
    root: Path,
    genesis: Mapping[str, Any],
    start: int,
    stop: int,
    maximum_seconds: float,
) -> dict[str, Any]:
    result = _run_owned_child(
        _manifest_worker,
        (_payload(root, genesis, start, stop),),
        maximum_seconds=maximum_seconds,
        action_name=f"survivor page {start}:{stop}",
    )
    manifest = dict(result["manifest"])
    validate_candidate_manifest(manifest)
    return manifest


def _prior_evidence(
    root: Path, followup: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[int, Mapping[str, Any]]]:
    records: dict[int, dict[str, Any]] = {}
    batches: dict[int, Mapping[str, Any]] = {}
    for binding in followup["batch_artifact_bindings"]:
        batch = _load_bound_json(root, binding)
        batches[int(batch["batch_index"])] = batch
        for record in batch["formal_evidence"]["candidate_records"]:
            ordinal = int(record["ordinal"])
            if ordinal in records:
                raise ValueError("bounded sample repeats a survivor ordinal")
            records[ordinal] = {
                "candidate_id": record["candidate_id"],
                "batch_index": batch["batch_index"],
                "batch_binding": dict(binding),
                "formal_receipt_content_sha256": batch["formal_receipt"]["content_sha256"],
                "formal_evidence_content_sha256": batch["formal_evidence"]["content_sha256"],
                "formal_evidence_record_sha256": _sha(record),
                "decision": record["decision"],
                "first_blocker": record["first_blocker"],
            }
    if len(records) != 192:
        raise ValueError("bounded sample receipt count changed")
    return records, batches


def _queue_rows(
    manifest: Mapping[str, Any], batch_index: int, prior: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for candidate in manifest["survivor_records"]:
        ordinal = int(candidate["ordinal"])
        earlier = prior.get(ordinal)
        if earlier is not None:
            if earlier["candidate_id"] != candidate["candidate_id"]:
                raise ValueError("bounded sample candidate identity changed")
            receipt = {
                key: earlier[key]
                for key in (
                    "batch_binding",
                    "formal_receipt_content_sha256",
                    "formal_evidence_content_sha256",
                    "formal_evidence_record_sha256",
                    "decision",
                    "first_blocker",
                )
            }
            state = "completed_preserved_formal_reject"
        else:
            receipt = None
            state = "pending_candidate_specific_formal_receipt"
        rows.append(
            {
                "batch_index": batch_index,
                "candidate_id": candidate["candidate_id"],
                "ordinal": ordinal,
                "candidate_record_sha256": _sha(candidate),
                "state": state,
                "formal_receipt_binding": receipt,
            }
        )
    return rows


def _node_path(directory: Path, batch_index: int, start: int, stop: int) -> Path:
    return directory / f"batch-{batch_index:02d}" / f"node-{start}-{stop}.json"


def _validate_queue_rows(
    rows: Any,
    manifest: Mapping[str, Any],
    batch_index: int,
    prior: Mapping[int, Mapping[str, Any]],
) -> None:
    if not isinstance(rows, list) or rows != _queue_rows(manifest, batch_index, prior):
        raise ValueError("formal receipt queue contract mismatch")


def _validate_node(
    value: Mapping[str, Any],
    batch_index: int,
    prior: Mapping[int, Mapping[str, Any]],
) -> None:
    _validate_sealed(value, "survivor page node")
    expected = {
        "schema_version",
        "batch_index",
        "ordinal_interval",
        "candidate_manifest",
        "leaf_page",
        "child_bindings",
        "formal_receipt_queue",
        "formal_receipt_queue_root_sha256",
        "complete_survivor_records",
        "rank_or_promotion_requested",
        "content_sha256",
    }
    manifest = value.get("candidate_manifest", {})
    validate_candidate_manifest(manifest)
    leaf = value.get("leaf_page")
    queue = value.get("formal_receipt_queue")
    children = value.get("child_bindings")
    if (
        set(value) != expected
        or value["schema_version"] != NODE_SCHEMA
        or value["batch_index"] != batch_index
        or value["ordinal_interval"] != manifest["batch"]
        or not isinstance(leaf, bool)
        or not isinstance(children, list)
        or not isinstance(queue, list)
        or value["formal_receipt_queue_root_sha256"] != _sha(queue)
        or value["complete_survivor_records"] is not leaf
        or value["rank_or_promotion_requested"] is not False
    ):
        raise ValueError("survivor page node contract mismatch")
    if leaf:
        if children or manifest["sample_complete"] is not True:
            raise ValueError("leaf page is not complete")
        _validate_queue_rows(queue, manifest, batch_index, prior)
    elif (
        len(children) != 2
        or queue
        or manifest["sample_complete"] is not False
        or manifest["survivor_record_count"] <= 32
    ):
        raise ValueError("internal survivor page node contract mismatch")


def _load_node_binding(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    return _load_bound_json(root, binding)


def _build_node(
    *,
    root: Path,
    directory: Path,
    config: Mapping[str, Any],
    genesis: Mapping[str, Any],
    batch_index: int,
    start: int,
    stop: int,
    prior: Mapping[int, Mapping[str, Any]],
    started: float,
) -> tuple[dict[str, Any], list[int]]:
    path = _node_path(directory, batch_index, start, stop)
    if path.exists():
        value = _load_json(path)
        _validate_node(value, batch_index, prior)
        if value["ordinal_interval"] != {
            "start_ordinal": start,
            "end_ordinal_exclusive": stop,
            "candidate_count": stop - start,
        }:
            raise ValueError("resumed survivor node interval changed")
        ordinals: list[int] = []
        if value["leaf_page"]:
            ordinals = [
                int(row["ordinal"]) for row in value["candidate_manifest"]["survivor_records"]
            ]
        else:
            for child_binding in value["child_bindings"]:
                child = _load_node_binding(root, child_binding)
                _, child_ordinals = _validate_tree(root, child, batch_index, prior)
                ordinals.extend(child_ordinals)
        return value, ordinals
    remaining = float(config["maximum_total_seconds"]) - (time.monotonic() - started)
    if remaining <= 2:
        raise TimeoutError("survivor pagination exceeded total cleanup-inclusive deadline")
    manifest = _evaluate_manifest(
        root,
        genesis,
        start,
        stop,
        min(float(config["maximum_node_seconds"]), remaining),
    )
    leaf = manifest["survivor_record_count"] <= config["maximum_survivors_per_leaf_page"]
    children: list[dict[str, str]] = []
    ordinals: list[int] = []
    if leaf:
        if manifest["sample_complete"] is not True:
            raise ValueError("bounded backend did not make leaf page complete")
        queue = _queue_rows(manifest, batch_index, prior)
        ordinals = [int(row["ordinal"]) for row in manifest["survivor_records"]]
    else:
        midpoint = start + (stop - start) // 2
        if midpoint in {start, stop}:
            raise ValueError("survivor page cannot be split further")
        queue = []
        for child_start, child_stop in ((start, midpoint), (midpoint, stop)):
            child, child_ordinals = _build_node(
                root=root,
                directory=directory,
                config=config,
                genesis=genesis,
                batch_index=batch_index,
                start=child_start,
                stop=child_stop,
                prior=prior,
                started=started,
            )
            child_path = _node_path(directory, batch_index, child_start, child_stop)
            children.append(_binding(root, child_path, child))
            ordinals.extend(child_ordinals)
        if manifest["all_survivor_ordinals_root_sha256"] != _sha(ordinals):
            raise ValueError("child survivor ordinal hierarchy differs from parent")
    body = {
        "schema_version": NODE_SCHEMA,
        "batch_index": batch_index,
        "ordinal_interval": manifest["batch"],
        "candidate_manifest": manifest,
        "leaf_page": leaf,
        "child_bindings": children,
        "formal_receipt_queue": queue,
        "formal_receipt_queue_root_sha256": _sha(queue),
        "complete_survivor_records": leaf,
        "rank_or_promotion_requested": False,
    }
    value = _sealed(body)
    _validate_node(value, batch_index, prior)
    _write_atomic_immutable(path, value, int(config["maximum_artifact_bytes"]))
    return value, ordinals


def _validate_tree(
    root: Path,
    value: Mapping[str, Any],
    batch_index: int,
    prior: Mapping[int, Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[int]]:
    _validate_node(value, batch_index, prior)
    if value["leaf_page"]:
        return [value], [
            int(row["ordinal"]) for row in value["candidate_manifest"]["survivor_records"]
        ]
    leaves: list[Mapping[str, Any]] = []
    ordinals: list[int] = []
    children = [_load_node_binding(root, binding) for binding in value["child_bindings"]]
    intervals = [child["ordinal_interval"] for child in children]
    parent = value["ordinal_interval"]
    if (
        intervals[0]["start_ordinal"] != parent["start_ordinal"]
        or intervals[0]["end_ordinal_exclusive"] != intervals[1]["start_ordinal"]
        or intervals[1]["end_ordinal_exclusive"] != parent["end_ordinal_exclusive"]
    ):
        raise ValueError("child page coverage is not ordered, contiguous, and disjoint")
    for child in children:
        child_leaves, child_ordinals = _validate_tree(root, child, batch_index, prior)
        leaves.extend(child_leaves)
        ordinals.extend(child_ordinals)
    manifest = value["candidate_manifest"]
    if (
        len(ordinals) != manifest["survivor_record_count"]
        or ordinals != sorted(set(ordinals))
        or _sha(ordinals) != manifest["all_survivor_ordinals_root_sha256"]
    ):
        raise ValueError("survivor page hierarchy root mismatch")
    return leaves, ordinals


def _source_batch(
    root: Path, followup: Mapping[str, Any], batch_index: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    for binding in followup["batch_artifact_bindings"]:
        batch = _load_bound_json(root, binding)
        if batch["batch_index"] == batch_index:
            return binding, batch
    raise ValueError("bounded follow-up batch binding missing")


def _batch_index_path(directory: Path, batch_index: int) -> Path:
    return directory / f"batch-{batch_index:02d}" / "index.json"


def _derive_batch_index(
    *,
    root: Path,
    batch_index: int,
    source_binding: Mapping[str, Any],
    source_batch: Mapping[str, Any],
    worker_bindings: list[Mapping[str, Any]],
    worker_nodes: list[Mapping[str, Any]],
    leaves: list[Mapping[str, Any]],
    ordinals: list[int],
) -> dict[str, Any]:
    queue = [row for leaf in leaves for row in leaf["formal_receipt_queue"]]
    source_manifest = source_batch["candidate_manifest"]
    worker_manifests = [node["candidate_manifest"] for node in worker_nodes]
    return {
        "schema_version": BATCH_SCHEMA,
        "batch_index": batch_index,
        "source_bounded_followup_batch_binding": dict(source_binding),
        "source_candidate_manifest_binding": {
            "content_sha256": source_manifest["content_sha256"],
            "candidate_root_sha256": source_manifest["candidate_root_sha256"],
            "all_survivor_ordinals_root_sha256": source_manifest[
                "all_survivor_ordinals_root_sha256"
            ],
            "survivor_record_count": source_manifest["survivor_record_count"],
        },
        "ordered_worker_root_bindings": worker_bindings,
        "ordered_worker_candidate_root_sha256": hashlib.sha256(
            "".join(item["candidate_root_sha256"] for item in worker_manifests).encode()
        ).hexdigest(),
        "ordered_worker_survivor_root_sha256": _sha(
            [item["all_survivor_ordinals_root_sha256"] for item in worker_manifests]
        ),
        "ordered_leaf_page_content_root_sha256": _sha([leaf["content_sha256"] for leaf in leaves]),
        "complete_survivor_ordinals_root_sha256": _sha(ordinals),
        "formal_receipt_queue_root_sha256": _sha(queue),
        "counts": {
            "worker_roots": len(worker_nodes),
            "leaf_pages": len(leaves),
            "survivors": len(ordinals),
            "preserved_completed_formal_receipts": sum(
                row["state"] == "completed_preserved_formal_reject" for row in queue
            ),
            "pending_formal_receipts": sum(
                row["state"] == "pending_candidate_specific_formal_receipt" for row in queue
            ),
        },
        "complete_survivor_pagination": True,
        "complete_formal_receipts": False,
        "rank_or_promotion_requested": False,
    }


def _validate_batch_index(
    value: Mapping[str, Any],
    root: Path,
    followup: Mapping[str, Any],
    prior: Mapping[int, Mapping[str, Any]],
) -> None:
    _validate_sealed(value, "survivor batch index")
    expected_keys = {
        "schema_version",
        "batch_index",
        "source_bounded_followup_batch_binding",
        "source_candidate_manifest_binding",
        "ordered_worker_root_bindings",
        "ordered_worker_candidate_root_sha256",
        "ordered_worker_survivor_root_sha256",
        "ordered_leaf_page_content_root_sha256",
        "complete_survivor_ordinals_root_sha256",
        "formal_receipt_queue_root_sha256",
        "counts",
        "complete_survivor_pagination",
        "complete_formal_receipts",
        "rank_or_promotion_requested",
        "content_sha256",
    }
    if set(value) != expected_keys or value["schema_version"] != BATCH_SCHEMA:
        raise ValueError("survivor batch index contract mismatch")
    batch_index = value["batch_index"]
    source_binding, source_batch = _source_batch(root, followup, batch_index)
    if value["source_bounded_followup_batch_binding"] != source_binding:
        raise ValueError("survivor batch source binding mismatch")
    worker_nodes = [
        _load_node_binding(root, item) for item in value["ordered_worker_root_bindings"]
    ]
    leaves: list[Mapping[str, Any]] = []
    ordinals: list[int] = []
    for worker in worker_nodes:
        worker_leaves, worker_ordinals = _validate_tree(root, worker, batch_index, prior)
        leaves.extend(worker_leaves)
        ordinals.extend(worker_ordinals)
    intervals = [node["ordinal_interval"] for node in worker_nodes]
    source_interval = source_batch["ordinal_interval"]
    if (
        len(worker_nodes) != 15
        or intervals[0]["start_ordinal"] != source_interval["start_ordinal"]
        or intervals[-1]["end_ordinal_exclusive"] != source_interval["stop_ordinal_exclusive"]
        or any(
            left["end_ordinal_exclusive"] != right["start_ordinal"]
            for left, right in itertools.pairwise(intervals)
        )
    ):
        raise ValueError("worker roots do not cover the source batch")
    expected = _sealed(
        _derive_batch_index(
            root=root,
            batch_index=batch_index,
            source_binding=source_binding,
            source_batch=source_batch,
            worker_bindings=value["ordered_worker_root_bindings"],
            worker_nodes=worker_nodes,
            leaves=leaves,
            ordinals=ordinals,
        )
    )
    if dict(value) != expected:
        raise ValueError("survivor batch index derivation mismatch")
    source_manifest = source_batch["candidate_manifest"]
    if (
        value["ordered_worker_candidate_root_sha256"] != source_manifest["candidate_root_sha256"]
        or value["ordered_worker_survivor_root_sha256"]
        != source_manifest["all_survivor_ordinals_root_sha256"]
        or value["counts"]["survivors"] != source_manifest["survivor_record_count"]
    ):
        raise ValueError("complete pagination does not replay the source batch roots")


def _build_batch(
    root: Path,
    directory: Path,
    config: Mapping[str, Any],
    genesis: Mapping[str, Any],
    followup: Mapping[str, Any],
    prior: Mapping[int, Mapping[str, Any]],
    batch_index: int,
    started: float,
) -> dict[str, Any]:
    index_path = _batch_index_path(directory, batch_index)
    if index_path.exists():
        value = _load_json(index_path)
        _validate_batch_index(value, root, followup, prior)
        return value
    source_binding, source_batch = _source_batch(root, followup, batch_index)
    interval = source_batch["ordinal_interval"]
    worker_bindings: list[Mapping[str, Any]] = []
    worker_nodes: list[Mapping[str, Any]] = []
    leaves: list[Mapping[str, Any]] = []
    ordinals: list[int] = []
    span = int(config["worker_ordinal_span"])
    for worker_start in range(interval["start_ordinal"], interval["stop_ordinal_exclusive"], span):
        worker_stop = min(worker_start + span, interval["stop_ordinal_exclusive"])
        node, _ = _build_node(
            root=root,
            directory=directory,
            config=config,
            genesis=genesis,
            batch_index=batch_index,
            start=worker_start,
            stop=worker_stop,
            prior=prior,
            started=started,
        )
        path = _node_path(directory, batch_index, worker_start, worker_stop)
        worker_bindings.append(_binding(root, path, node))
        worker_nodes.append(node)
        worker_leaves, worker_ordinals = _validate_tree(root, node, batch_index, prior)
        leaves.extend(worker_leaves)
        ordinals.extend(worker_ordinals)
    body = _derive_batch_index(
        root=root,
        batch_index=batch_index,
        source_binding=source_binding,
        source_batch=source_batch,
        worker_bindings=worker_bindings,
        worker_nodes=worker_nodes,
        leaves=leaves,
        ordinals=ordinals,
    )
    value = _sealed(body)
    _validate_batch_index(value, root, followup, prior)
    _write_atomic_immutable(index_path, value, int(config["maximum_artifact_bytes"]))
    return value


def _derive_result(
    root: Path,
    config: Mapping[str, Any],
    batch_bindings: list[Mapping[str, Any]],
    batches: list[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {
        "source_pass_batches": len(batches),
        "source_survivors": sum(item["counts"]["survivors"] for item in batches),
        "worker_roots": sum(item["counts"]["worker_roots"] for item in batches),
        "leaf_pages": sum(item["counts"]["leaf_pages"] for item in batches),
        "durable_formal_receipt_queue_entries": sum(
            item["counts"]["survivors"] for item in batches
        ),
        "preserved_completed_formal_receipts": sum(
            item["counts"]["preserved_completed_formal_receipts"] for item in batches
        ),
        "pending_formal_receipts": sum(
            item["counts"]["pending_formal_receipts"] for item in batches
        ),
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "epoch_terminal_result_binding": config["epoch_terminal_result"],
        "bounded_sample_followup_binding": config["bounded_sample_followup"],
        "ordered_batch_index_bindings": batch_bindings,
        "ordered_batch_content_root_sha256": _sha([item["content_sha256"] for item in batches]),
        "ordered_complete_survivor_roots_sha256": _sha(
            [item["complete_survivor_ordinals_root_sha256"] for item in batches]
        ),
        "formal_receipt_queue_hierarchy_root_sha256": _sha(
            [item["formal_receipt_queue_root_sha256"] for item in batches]
        ),
        "counts": counts,
        "complete_survivor_pagination": True,
        "complete_formal_receipts": False,
        "complete_comparable_evidence": False,
        "first_remaining_blocker": "candidate_specific_formal_receipts_pending",
        "execution_contract": {
            "cpu_workers_per_node": 1,
            "gpu_workers": 0,
            "maximum_node_seconds": config["maximum_node_seconds"],
            "maximum_total_seconds": config["maximum_total_seconds"],
            "page_cap": config["maximum_survivors_per_leaf_page"],
            "atomic_resume": "validate_and_reuse_each_sealed_node_then_atomic_replace_new_nodes",
            "deadline": "campaign_owned_child_cleanup_inclusive_hard_wall_clock_bound",
        },
        "promotion_contract": {
            "formal_pass_claimed": False,
            "leaderboard_rebuild_requested": False,
            "rank_assignment_performed": False,
            "candidate_promotion_performed": False,
        },
        "bindings": {
            label: {"path": relative, "file_sha256": _file_sha(root / relative)}
            for label, relative in (
                ("config", CONFIG_REL),
                ("source", SOURCE_REL),
                ("test", TEST_REL),
            )
        },
        "seals": {
            "gpu_or_cuda_access": False,
            "campaign_live_SQLite_access": False,
            "observations_opened": False,
            "external_process_signals": False,
            "direct_rank_assignment": False,
        },
    }


def validate_result(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value, "survivor pagination result")
    config = load_config(root, config_path)
    terminal = _load_bound_json(root, config["epoch_terminal_result"])
    validate_epoch_result(terminal, root)
    followup = _load_bound_json(root, config["bounded_sample_followup"])
    validate_followup(
        followup,
        root,
        root / "configs/continuous_scientific_pipeline_epoch_003_candidate_followup.json",
    )
    prior, _ = _prior_evidence(root, followup)
    bindings = value.get("ordered_batch_index_bindings")
    if not isinstance(bindings, list) or len(bindings) != 6:
        raise ValueError("pagination batch binding count mismatch")
    batches = []
    for expected_index, binding in zip(config["pass_batch_indices"], bindings, strict=True):
        batch = _load_bound_json(root, binding)
        _validate_batch_index(batch, root, followup, prior)
        if batch["batch_index"] != expected_index:
            raise ValueError("pagination batch ordering mismatch")
        batches.append(batch)
    expected = _sealed(_derive_result(root, config, bindings, batches))
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["counts"]["source_survivors"] != 11_439
        or value["counts"]["preserved_completed_formal_receipts"] != 192
        or value["counts"]["pending_formal_receipts"] != 11_247
        or value["complete_survivor_pagination"] is not True
        or value["complete_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("survivor pagination result contract mismatch")


def build_result(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(root, config_path)
    genesis = _load_bound_json(root, config["epoch_genesis"])
    validate_epoch_genesis(
        genesis, root, root / "configs/continuous_scientific_pipeline_epoch_003.json"
    )
    terminal = _load_bound_json(root, config["epoch_terminal_result"])
    validate_epoch_result(terminal, root)
    followup = _load_bound_json(root, config["bounded_sample_followup"])
    validate_followup(
        followup,
        root,
        root / "configs/continuous_scientific_pipeline_epoch_003_candidate_followup.json",
    )
    prior, _ = _prior_evidence(root, followup)
    directory = _resolve(root, config["artifact_directory"])
    started = time.monotonic()
    batches = []
    bindings = []
    for batch_index in config["pass_batch_indices"]:
        batch = _build_batch(
            root,
            directory,
            config,
            genesis,
            followup,
            prior,
            batch_index,
            started,
        )
        path = _batch_index_path(directory, batch_index)
        batches.append(batch)
        bindings.append(_binding(root, path, batch))
    result = _sealed(_derive_result(root, config, bindings, batches))
    validate_result(result, root, config_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_REL)
    parser.add_argument("--output", default=RESULT_REL)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    config_path = _resolve(root, arguments.config)
    result = build_result(root, config_path)
    output = _resolve(root, arguments.output)
    _write_atomic_immutable(
        output, result, int(load_config(root, config_path)["maximum_artifact_bytes"])
    )
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
