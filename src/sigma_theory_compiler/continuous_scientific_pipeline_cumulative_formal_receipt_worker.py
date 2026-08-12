"""Cumulative, prefix-ordered formal receipt worker for Epoch 003 survivor pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .continuous_formula_formal_backend import (
    _validate_semantic_health,
    load_backend_config,
    validate_candidate_manifest,
    validate_formal_evidence,
)
from .continuous_scientific_pipeline_admission import (
    _validate_formal_receipt,
    _validate_generated_receipt,
)
from .continuous_scientific_pipeline_formal_receipt_worker import (
    validate_result as validate_predecessor_result,
)
from .continuous_scientific_pipeline_service import _real_formal_worker, _run_owned_child
from .continuous_scientific_pipeline_survivor_pagination import (
    validate_result as validate_pagination_result,
)

CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-worker-config-1.0"
PREFLIGHT_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-preflight-1.0"
PARTITION_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-partition-1.0"
LEDGER_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-ledger-1.0"
RESULT_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-result-1.0"
DECISION = "cumulative_formal_receipt_prefix_advanced_global_queue_incomplete_no_promotion"
CONFIG_REL = (
    "configs/continuous_scientific_pipeline_epoch_003_cumulative_formal_receipt_worker.json"
)
SOURCE_REL = (
    "src/sigma_theory_compiler/continuous_scientific_pipeline_cumulative_formal_receipt_worker.py"
)
TEST_REL = "tests/test_continuous_scientific_pipeline_cumulative_formal_receipt_worker.py"
RESULT_REL = (
    "runs/engine/continuous-scientific-pipeline-epoch-003-cumulative-formal-receipt-worker/"
    "result.json"
)
PAGINATION_CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_survivor_pagination.json"
PREDECESSOR_CONFIG_REL = (
    "configs/continuous_scientific_pipeline_epoch_003_formal_receipt_worker_partition_0001.json"
)


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
        raise ValueError("cumulative formal JSON binding contract mismatch")
    path = _resolve(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("cumulative formal JSON binding file hash mismatch")
    value = _load_json(path)
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("cumulative formal JSON binding content hash mismatch")
    return value


def _binding(root: Path, path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": _file_sha(path),
        "content_sha256": str(value["content_sha256"]),
    }


def _write_atomic_immutable(path: Path, value: Mapping[str, Any], maximum_bytes: int) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(raw) > maximum_bytes:
        raise RuntimeError("cumulative formal artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"immutable cumulative formal artifact differs: {path.name}")
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
                raise ValueError(
                    f"immutable cumulative formal artifact differs: {path.name}"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = _load_json(path)
    expected = {
        "schema_version",
        "campaign_id",
        "pagination_result",
        "predecessor_worker_result",
        "formal_backend_config",
        "artifact_directory",
        "maximum_partitions_per_invocation",
        "maximum_partition_candidates",
        "maximum_formal_seconds",
        "maximum_total_seconds",
        "maximum_artifact_bytes",
        "resource_gate",
        "seals",
    }
    if (
        set(config) != expected
        or config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"]
        != "continuous-scientific-pipeline-epoch-003-cumulative-formal-receipt-worker"
        or config["maximum_partitions_per_invocation"] != 1
        or config["maximum_partition_candidates"] != 32
        or config["maximum_formal_seconds"] != 120
        or config["maximum_total_seconds"] != 180
        or config["maximum_artifact_bytes"] != 4_194_304
        or config["resource_gate"]
        != {
            "cpu_utilization_below_percent": 92,
            "minimum_available_ram_mib": 32768,
            "cpu_workers": 1,
            "gpu_workers": 0,
        }
        or config["seals"]
        != {
            "live_campaign_SQLite_access": False,
            "observations_opened": False,
            "gpu_or_cuda_access": False,
            "external_process_signals": False,
            "leaderboard_or_rank_writes": False,
        }
    ):
        raise ValueError("cumulative formal config contract mismatch")
    _resolve(root, str(config["artifact_directory"]))
    pagination = _load_bound_json(root, config["pagination_result"])
    validate_pagination_result(pagination, root, root / PAGINATION_CONFIG_REL)
    predecessor = _load_bound_json(root, config["predecessor_worker_result"])
    validate_predecessor_result(predecessor, root, root / PREDECESSOR_CONFIG_REL)
    backend = config["formal_backend_config"]
    if set(backend) != {"path", "file_sha256"}:
        raise ValueError("cumulative formal backend binding contract mismatch")
    backend_path = _resolve(root, str(backend["path"]))
    if _file_sha(backend_path) != backend["file_sha256"]:
        raise ValueError("cumulative formal backend binding mismatch")
    load_backend_config(root, backend_path)
    return config


def _leaf_catalog(root: Path, pagination: Mapping[str, Any]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for batch_position, index_binding in enumerate(pagination["ordered_batch_index_bindings"]):
        index = _load_bound_json(root, index_binding)
        for worker_position, worker_binding in enumerate(index["ordered_worker_root_bindings"]):
            stack: list[tuple[Mapping[str, Any], list[dict[str, str]]]] = [
                (worker_binding, [dict(index_binding)])
            ]
            while stack:
                binding, ancestors = stack.pop()
                node = _load_bound_json(root, binding)
                path = [*ancestors, dict(binding)]
                if node["leaf_page"] is not True:
                    for child in reversed(node["child_bindings"]):
                        stack.append((child, path))
                    continue
                pending = [
                    row
                    for row in node["formal_receipt_queue"]
                    if row["state"] == "pending_candidate_specific_formal_receipt"
                ]
                if not pending:
                    continue
                catalog.append(
                    {
                        "catalog_index": len(catalog),
                        "batch_position": batch_position,
                        "worker_position": worker_position,
                        "batch_index": node["batch_index"],
                        "leaf_binding": dict(binding),
                        "hierarchy_path": path,
                        "ordinal_interval": node["ordinal_interval"],
                        "leaf_queue_root_sha256": node["formal_receipt_queue_root_sha256"],
                        "pending_candidate_count": len(pending),
                        "pending_candidate_ordinals_root_sha256": _sha(
                            [row["ordinal"] for row in pending]
                        ),
                    }
                )
    if [row["catalog_index"] for row in catalog] != list(range(len(catalog))):
        raise ValueError("pending leaf catalog ordering mismatch")
    leaves = [row["leaf_binding"]["content_sha256"] for row in catalog]
    if len(leaves) != len(set(leaves)):
        raise ValueError("pending leaf catalog contains overlapping leaf roots")
    return catalog


def _catalog_entry_for_binding(
    catalog: list[Mapping[str, Any]], binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = [row for row in catalog if row["leaf_binding"] == dict(binding)]
    if len(matches) != 1:
        raise ValueError("processed leaf is transplanted or absent from pagination hierarchy")
    return matches[0]


def _validate_preserved_receipts(
    root: Path, leaf: Mapping[str, Any], partition: Mapping[str, Any]
) -> None:
    receipts = {int(row["ordinal"]): row for row in partition["candidate_formal_receipts"]}
    for queue_row in leaf["formal_receipt_queue"]:
        if queue_row["state"] != "completed_preserved_formal_reject":
            continue
        prior = queue_row["formal_receipt_binding"]
        if not isinstance(prior, Mapping):
            raise TypeError("preserved receipt lacks prior binding")
        batch = _load_bound_json(root, prior["batch_binding"])
        prior_records = [
            row
            for row in batch["formal_evidence"]["candidate_records"]
            if row["ordinal"] == queue_row["ordinal"]
        ]
        current = receipts.get(int(queue_row["ordinal"]))
        if len(prior_records) != 1 or current is None:
            raise ValueError("preserved receipt candidate lineage mismatch")
        old = prior_records[0]
        expected_prior = {
            "batch_binding": prior["batch_binding"],
            "formal_receipt_content_sha256": batch["formal_receipt"]["content_sha256"],
            "formal_evidence_content_sha256": batch["formal_evidence"]["content_sha256"],
            "formal_evidence_record_sha256": _sha(old),
            "decision": old["decision"],
            "first_blocker": old["first_blocker"],
        }
        if (
            dict(prior) != expected_prior
            or current["candidate_id"] != queue_row["candidate_id"]
            or current["candidate_id"] != old["candidate_id"]
            or current["source_queue_entry_sha256"] != _sha(queue_row)
            or current["source_queue_state"] != "completed_preserved_formal_reject"
            or current["prior_formal_receipt_binding_sha256"] != _sha(prior)
            or current["newly_processed"] is not False
            or current["decision"] != old["decision"]
            or current["first_blocker"] != old["first_blocker"]
        ):
            raise ValueError("preserved receipt field reconciliation mismatch")


def _predecessor_partition(
    root: Path, predecessor: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    partition = _load_bound_json(root, predecessor["partition_result_binding"])
    leaf = _load_bound_json(root, predecessor["selected_leaf_page_binding"])
    _validate_preserved_receipts(root, leaf, partition)
    return partition, leaf


def _partition_summary(
    *,
    sequence: int,
    catalog_entry: Mapping[str, Any],
    binding: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = partition["candidate_formal_receipts"]
    return {
        "partition_sequence": sequence,
        "leaf_catalog_index": catalog_entry["catalog_index"],
        "leaf_binding": catalog_entry["leaf_binding"],
        "hierarchy_path_root_sha256": _sha(catalog_entry["hierarchy_path"]),
        "partition_binding": dict(binding),
        "partition_candidate_count": len(receipts),
        "newly_processed_candidate_count": sum(row["newly_processed"] for row in receipts),
        "reconciled_preserved_candidate_count": sum(not row["newly_processed"] for row in receipts),
        "all_candidate_ordinals_root_sha256": _sha([row["ordinal"] for row in receipts]),
        "newly_processed_ordinals_root_sha256": _sha(
            [row["ordinal"] for row in receipts if row["newly_processed"]]
        ),
        "candidate_formal_receipts_root_sha256": partition["candidate_formal_receipts_root_sha256"],
        "decision_counts": {
            "reject": sum(row["decision"] == "reject" for row in receipts),
            "block": sum(row["decision"] == "block" for row in receipts),
            "pass": 0,
        },
    }


def _validate_processed_prefix(
    root: Path,
    catalog: list[Mapping[str, Any]],
    summaries: list[Mapping[str, Any]],
    partitions: list[Mapping[str, Any]],
) -> None:
    if len(summaries) != len(partitions) or not summaries:
        raise ValueError("cumulative partition ledger is empty or misaligned")
    if [row["partition_sequence"] for row in summaries] != list(range(1, len(summaries) + 1)):
        raise ValueError("cumulative partition sequence gap")
    if [row["leaf_catalog_index"] for row in summaries] != list(range(len(summaries))):
        raise ValueError("cumulative leaf prefix has a gap or overlap")
    leaf_roots = [row["leaf_binding"]["content_sha256"] for row in summaries]
    if len(leaf_roots) != len(set(leaf_roots)):
        raise ValueError("cumulative partition leaf overlap")
    ordinals: list[int] = []
    for position, (summary, partition) in enumerate(zip(summaries, partitions, strict=True)):
        entry = catalog[position]
        if summary["leaf_binding"] != entry["leaf_binding"]:
            raise ValueError("cumulative processed leaf is transplanted")
        leaf = _load_bound_json(root, entry["leaf_binding"])
        _validate_preserved_receipts(root, leaf, partition)
        rows = partition["candidate_formal_receipts"]
        if summary != _partition_summary(
            sequence=position + 1,
            catalog_entry=entry,
            binding=summary["partition_binding"],
            partition=partition,
        ):
            raise ValueError("cumulative partition summary mismatch")
        current = [int(row["ordinal"]) for row in rows]
        if set(ordinals).intersection(current):
            raise ValueError("cumulative partition candidate overlap")
        ordinals.extend(current)


def _selected_next_leaf(
    catalog: list[Mapping[str, Any]], predecessor: Mapping[str, Any]
) -> Mapping[str, Any]:
    previous = _catalog_entry_for_binding(catalog, predecessor["selected_leaf_page_binding"])
    if previous["catalog_index"] != 0:
        raise ValueError("predecessor is not the first deterministic pending leaf")
    if len(catalog) < 2:
        raise ValueError("no next pending leaf remains")
    return catalog[1]


def _artifact_path(root: Path, config: Mapping[str, Any], name: str) -> Path:
    return _resolve(root, str(config["artifact_directory"])) / name


def _build_preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = _artifact_path(root, config, "preflight.json")
    if path.exists():
        value = _load_json(path)
        _validate_preflight(value, config)
        return value
    import psutil

    cpu = float(psutil.cpu_percent(interval=1.0))
    ram = int(psutil.virtual_memory().available // (1024 * 1024))
    gate = config["resource_gate"]
    admitted = (
        cpu < gate["cpu_utilization_below_percent"] and ram >= gate["minimum_available_ram_mib"]
    )
    value = _sealed(
        {
            "schema_version": PREFLIGHT_SCHEMA,
            "sampled_at": datetime.now(UTC).isoformat(),
            "cpu_utilization_percent": cpu,
            "available_ram_mib": ram,
            "resource_gate": gate,
            "admitted": admitted,
            "gpu_or_cuda_probed": False,
            "processes_signaled": False,
        }
    )
    if not admitted:
        raise RuntimeError("cumulative formal worker failed resource admission")
    _validate_preflight(value, config)
    _write_atomic_immutable(path, value, int(config["maximum_artifact_bytes"]))
    return value


def _validate_preflight(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _validate_sealed(value, "cumulative formal preflight")
    gate = config["resource_gate"]
    if (
        set(value)
        != {
            "schema_version",
            "sampled_at",
            "cpu_utilization_percent",
            "available_ram_mib",
            "resource_gate",
            "admitted",
            "gpu_or_cuda_probed",
            "processes_signaled",
            "content_sha256",
        }
        or value["schema_version"] != PREFLIGHT_SCHEMA
        or not isinstance(value["sampled_at"], str)
        or not isinstance(value["cpu_utilization_percent"], (int, float))
        or isinstance(value["cpu_utilization_percent"], bool)
        or not isinstance(value["available_ram_mib"], int)
        or isinstance(value["available_ram_mib"], bool)
        or value["resource_gate"] != gate
        or value["admitted"] is not True
        or value["cpu_utilization_percent"] >= gate["cpu_utilization_below_percent"]
        or value["available_ram_mib"] < gate["minimum_available_ram_mib"]
        or value["gpu_or_cuda_probed"] is not False
        or value["processes_signaled"] is not False
    ):
        raise ValueError("cumulative formal preflight contract mismatch")


def _generated_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _sealed(
        {
            "candidate_root_sha256": manifest["candidate_root_sha256"],
            "screen_decision": "pass",
            "unique_formula_count": manifest["batch"]["candidate_count"],
            "theory_pass_claimed": False,
            "observations_opened": False,
            "rank_eligible": False,
        }
    )
    _validate_generated_receipt(value)
    return value


def _artifact_bindings(root: Path, directory: Path) -> list[dict[str, str]]:
    if not directory.exists():
        return []
    return [
        {"path": path.relative_to(root).as_posix(), "file_sha256": _file_sha(path)}
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def _validate_artifact_bindings(root: Path, bindings: Any) -> None:
    if not isinstance(bindings, list):
        raise TypeError("cumulative formal artifact bindings are not a list")
    paths = []
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or _file_sha(_resolve(root, str(binding["path"]))) != binding["file_sha256"]
        ):
            raise ValueError("cumulative formal artifact binding mismatch")
        paths.append(binding["path"])
    if paths != sorted(set(paths)):
        raise ValueError("cumulative formal artifact binding ordering mismatch")


def _candidate_receipts(
    leaf: Mapping[str, Any], evidence: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    queue = {int(row["ordinal"]): row for row in leaf["formal_receipt_queue"]}
    rows = []
    for record in evidence["candidate_records"]:
        source = queue[int(record["ordinal"])]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "ordinal": record["ordinal"],
                "source_queue_entry_sha256": _sha(source),
                "source_queue_state": source["state"],
                "prior_formal_receipt_binding_sha256": (
                    _sha(source["formal_receipt_binding"])
                    if source["formal_receipt_binding"] is not None
                    else None
                ),
                "newly_processed": source["state"] == "pending_candidate_specific_formal_receipt",
                "formal_receipt_content_sha256": receipt["content_sha256"],
                "formal_evidence_content_sha256": evidence["content_sha256"],
                "formal_evidence_record_sha256": _sha(record),
                "covariant_mapping_payload_sha256": record["covariant_mapping_payload_sha256"],
                "semantic_action_health_sha256": record["semantic_action_health_sha256"],
                "decision": record["decision"],
                "first_blocker": record["first_blocker"],
            }
        )
    return rows


def _derive_partition(
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    leaf: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
    generated: Mapping[str, Any],
    formal: Mapping[str, Any],
    evidence: Mapping[str, Any],
    artifact_bindings: list[Mapping[str, Any]],
) -> dict[str, Any]:
    receipts = _candidate_receipts(leaf, evidence, formal)
    return {
        "schema_version": PARTITION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "partition_sequence": 2,
        "leaf_catalog_index": entry["catalog_index"],
        "selected_leaf_page_binding": entry["leaf_binding"],
        "selected_leaf_hierarchy_path": entry["hierarchy_path"],
        "source_leaf_formal_receipt_queue_root_sha256": leaf["formal_receipt_queue_root_sha256"],
        "resource_preflight_binding": dict(preflight_binding),
        "generated_receipt": generated,
        "candidate_manifest": leaf["candidate_manifest"],
        "formal_receipt": formal,
        "formal_evidence": evidence,
        "candidate_artifact_bindings": artifact_bindings,
        "candidate_formal_receipts": receipts,
        "candidate_formal_receipts_root_sha256": _sha(receipts),
        "counts": {
            "partition_candidates": len(receipts),
            "newly_processed_candidates": sum(row["newly_processed"] for row in receipts),
            "reconciled_preserved_candidates": sum(not row["newly_processed"] for row in receipts),
            "candidate_rejects": sum(row["decision"] == "reject" for row in receipts),
            "candidate_blocks": sum(row["decision"] == "block" for row in receipts),
            "candidate_passes": 0,
            "formal_passes": 0,
            "rank_assignments": 0,
            "candidate_promotions": 0,
        },
        "decision": formal["decision"],
        "first_blocker": evidence["first_blocker"],
        "complete_partition_formal_receipts": True,
        "complete_comparable_evidence": False,
        "rank_or_promotion_requested": False,
        "seals": config["seals"],
    }


def _validate_partition(
    value: Mapping[str, Any],
    root: Path,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    leaf: Mapping[str, Any],
) -> None:
    _validate_sealed(value, "cumulative formal partition")
    preflight = _load_bound_json(root, value.get("resource_preflight_binding", {}))
    _validate_preflight(preflight, config)
    generated = value.get("generated_receipt", {})
    manifest = value.get("candidate_manifest", {})
    formal = value.get("formal_receipt", {})
    evidence = value.get("formal_evidence", {})
    _validate_generated_receipt(generated)
    validate_candidate_manifest(manifest)
    _validate_formal_receipt(formal)
    validate_formal_evidence(evidence)
    _validate_artifact_bindings(root, value.get("candidate_artifact_bindings"))
    artifact_root = _resolve(root, str(config["artifact_directory"])) / "candidate-artifacts"
    for record in evidence["candidate_records"]:
        if record["semantic_action_health"] is not None:
            _validate_semantic_health(
                record["semantic_action_health"],
                candidate_dir=artifact_root / record["candidate_id"],
            )
    expected = _sealed(
        _derive_partition(
            config,
            entry,
            leaf,
            value["resource_preflight_binding"],
            generated,
            formal,
            evidence,
            value["candidate_artifact_bindings"],
        )
    )
    if (
        dict(value) != expected
        or entry["catalog_index"] != 1
        or value["selected_leaf_page_binding"] != entry["leaf_binding"]
        or value["selected_leaf_hierarchy_path"] != entry["hierarchy_path"]
        or manifest != leaf["candidate_manifest"]
        or generated["candidate_root_sha256"] != manifest["candidate_root_sha256"]
        or formal["candidate_root_sha256"] != generated["candidate_root_sha256"]
        or evidence["candidate_manifest_sha256"] != manifest["content_sha256"]
        or evidence["generated_receipt_sha256"] != generated["content_sha256"]
        or value["counts"]["newly_processed_candidates"] != entry["pending_candidate_count"]
        or value["counts"]["reconciled_preserved_candidates"] != 0
        or value["complete_partition_formal_receipts"] is not True
        or value["complete_comparable_evidence"] is not False
        or value["rank_or_promotion_requested"] is not False
        or any(value["seals"].values())
    ):
        raise ValueError("cumulative formal partition contract mismatch")


def _publish_candidate_artifacts(attempt: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(item for item in attempt.iterdir() if item.is_dir()):
        target = destination / source.name
        if target.exists():
            existing = [
                (path.relative_to(target).as_posix(), _file_sha(path))
                for path in sorted(item for item in target.rglob("*") if item.is_file())
            ]
            proposed = [
                (path.relative_to(source).as_posix(), _file_sha(path))
                for path in sorted(item for item in source.rglob("*") if item.is_file())
            ]
            if existing != proposed:
                raise ValueError("immutable cumulative candidate artifacts differ")
            continue
        os.replace(source, target)


def _build_partition(
    root: Path,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    leaf: Mapping[str, Any],
    preflight: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    path = _artifact_path(root, config, "partition-0002.json")
    if path.exists():
        value = _load_json(path)
        _validate_partition(value, root, config, entry, leaf)
        return value
    remaining = config["maximum_total_seconds"] - (time.monotonic() - started)
    if remaining <= 2:
        raise TimeoutError("cumulative formal worker exceeded total wall-clock bound")
    generated = _generated_receipt(leaf["candidate_manifest"])
    directory = _resolve(root, str(config["artifact_directory"]))
    directory.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=".formal-attempt-", dir=directory))
    try:
        formal_result = _run_owned_child(
            _real_formal_worker,
            (
                str(root),
                str(attempt),
                generated,
                leaf["candidate_manifest"],
                {"formal_backend_config_path": config["formal_backend_config"]["path"]},
            ),
            maximum_seconds=min(float(config["maximum_formal_seconds"]), remaining),
            action_name="Epoch 003 cumulative formal receipt partition 0002",
        )
        formal = formal_result["receipt"]
        evidence = formal_result["evidence"]
        _validate_formal_receipt(formal)
        validate_formal_evidence(evidence)
        artifact_root = directory / "candidate-artifacts"
        _publish_candidate_artifacts(attempt, artifact_root)
        artifact_bindings = _artifact_bindings(root, artifact_root)
        value = _sealed(
            _derive_partition(
                config,
                entry,
                leaf,
                _binding(root, _artifact_path(root, config, "preflight.json"), preflight),
                generated,
                formal,
                evidence,
                artifact_bindings,
            )
        )
        _validate_partition(value, root, config, entry, leaf)
        _write_atomic_immutable(path, value, int(config["maximum_artifact_bytes"]))
        return value
    finally:
        shutil.rmtree(attempt, ignore_errors=True)


def _derive_ledger(
    root: Path,
    config: Mapping[str, Any],
    pagination: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    predecessor_partition: Mapping[str, Any],
    partition: Mapping[str, Any],
    catalog: list[Mapping[str, Any]],
) -> dict[str, Any]:
    predecessor_summary = _partition_summary(
        sequence=1,
        catalog_entry=catalog[0],
        binding=predecessor["partition_result_binding"],
        partition=predecessor_partition,
    )
    current_binding = _binding(root, _artifact_path(root, config, "partition-0002.json"), partition)
    current_summary = _partition_summary(
        sequence=2,
        catalog_entry=catalog[1],
        binding=current_binding,
        partition=partition,
    )
    summaries = [predecessor_summary, current_summary]
    partitions = [predecessor_partition, partition]
    _validate_processed_prefix(root, catalog, summaries, partitions)
    receipt_rows = [
        {
            "partition_sequence": summary["partition_sequence"],
            **row,
        }
        for summary, item in zip(summaries, partitions, strict=True)
        for row in item["candidate_formal_receipts"]
    ]
    newly_processed = [row for row in receipt_rows if row["newly_processed"]]
    initial_pending = pagination["counts"]["pending_formal_receipts"]
    counts = {
        "processed_partition_prefix_length": len(summaries),
        "processed_leaf_pages": len(summaries),
        "cumulative_formally_checked_candidates": len(receipt_rows),
        "cumulative_newly_processed_candidates": len(newly_processed),
        "cumulative_reconciled_preserved_candidates": len(receipt_rows) - len(newly_processed),
        "cumulative_candidate_rejects": sum(row["decision"] == "reject" for row in receipt_rows),
        "cumulative_candidate_blocks": sum(row["decision"] == "block" for row in receipt_rows),
        "cumulative_candidate_passes": 0,
        "cumulative_formal_passes": 0,
        "initial_pending_formal_receipts": initial_pending,
        "remaining_pending_formal_receipts": initial_pending - len(newly_processed),
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    return {
        "schema_version": LEDGER_SCHEMA,
        "campaign_id": config["campaign_id"],
        "pagination_result_binding": config["pagination_result"],
        "predecessor_worker_result_binding": config["predecessor_worker_result"],
        "pending_leaf_catalog_count": len(catalog),
        "pending_leaf_catalog_root_sha256": _sha(catalog),
        "processed_partition_summaries": summaries,
        "processed_partition_summaries_root_sha256": _sha(summaries),
        "cumulative_formal_receipt_records": receipt_rows,
        "cumulative_formal_receipt_ledger_root_sha256": _sha(receipt_rows),
        "cumulative_newly_processed_ordinals_root_sha256": _sha(
            [row["ordinal"] for row in newly_processed]
        ),
        "counts": counts,
        "complete_processed_partition_prefix": True,
        "complete_global_formal_receipts": False,
        "complete_comparable_evidence": False,
        "first_remaining_blocker": (
            f"{counts['remaining_pending_formal_receipts']}_candidate_specific_formal_receipts_pending"
        ),
        "promotion_contract": {
            "formal_pass_claimed": False,
            "leaderboard_rebuild_requested": False,
            "rank_assignment_performed": False,
            "candidate_promotion_performed": False,
        },
        "seals": config["seals"],
    }


def _validate_ledger(
    value: Mapping[str, Any],
    root: Path,
    config: Mapping[str, Any],
    pagination: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    predecessor_partition: Mapping[str, Any],
    partition: Mapping[str, Any],
    catalog: list[Mapping[str, Any]],
) -> None:
    _validate_sealed(value, "cumulative formal ledger")
    summaries = value.get("processed_partition_summaries")
    records = value.get("cumulative_formal_receipt_records")
    if not isinstance(summaries, list) or not isinstance(records, list):
        raise TypeError("cumulative formal ledger rows are not lists")
    partitions = []
    for summary in summaries:
        partitions.append(_load_bound_json(root, summary["partition_binding"]))
    _validate_processed_prefix(root, catalog, summaries, partitions)
    expected = _sealed(
        _derive_ledger(
            root,
            config,
            pagination,
            predecessor,
            predecessor_partition,
            partition,
            catalog,
        )
    )
    if (
        dict(value) != expected
        or value["processed_partition_summaries_root_sha256"] != _sha(summaries)
        or value["cumulative_formal_receipt_ledger_root_sha256"] != _sha(records)
        or value["counts"]["remaining_pending_formal_receipts"] != 11_213
        or value["complete_processed_partition_prefix"] is not True
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("cumulative formal ledger contract mismatch")


def _derive_result(
    root: Path,
    config: Mapping[str, Any],
    ledger: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "pagination_result_binding": config["pagination_result"],
        "predecessor_worker_result_binding": config["predecessor_worker_result"],
        "executed_partition_binding": _binding(
            root, _artifact_path(root, config, "partition-0002.json"), partition
        ),
        "cumulative_ledger_binding": _binding(
            root, _artifact_path(root, config, "cumulative-ledger.json"), ledger
        ),
        "pending_leaf_catalog_root_sha256": ledger["pending_leaf_catalog_root_sha256"],
        "processed_partition_summaries_root_sha256": ledger[
            "processed_partition_summaries_root_sha256"
        ],
        "cumulative_formal_receipt_ledger_root_sha256": ledger[
            "cumulative_formal_receipt_ledger_root_sha256"
        ],
        "cumulative_newly_processed_ordinals_root_sha256": ledger[
            "cumulative_newly_processed_ordinals_root_sha256"
        ],
        "counts": ledger["counts"],
        "complete_processed_partition_prefix": True,
        "complete_global_formal_receipts": False,
        "complete_comparable_evidence": False,
        "first_remaining_blocker": ledger["first_remaining_blocker"],
        "execution_contract": {
            "maximum_partitions_per_invocation": 1,
            "maximum_partition_candidates": 32,
            "cpu_workers": 1,
            "gpu_workers": 0,
            "maximum_formal_seconds": 120,
            "maximum_total_seconds": 180,
            "resume": "validate_immutable_partition_prefix_and_reuse_completed_successor",
            "deadline": "campaign_owned_child_cleanup_inclusive_hard_wall_clock_bound",
        },
        "promotion_contract": ledger["promotion_contract"],
        "bindings": {
            label: {"path": relative, "file_sha256": _file_sha(root / relative)}
            for label, relative in (
                ("config", CONFIG_REL),
                ("source", SOURCE_REL),
                ("test", TEST_REL),
            )
        },
        "seals": config["seals"],
    }


def validate_result(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value, "cumulative formal result")
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_worker_result"])
    predecessor_partition, _ = _predecessor_partition(root, predecessor)
    catalog = _leaf_catalog(root, pagination)
    entry = _selected_next_leaf(catalog, predecessor)
    leaf = _load_bound_json(root, entry["leaf_binding"])
    partition = _load_bound_json(root, value.get("executed_partition_binding", {}))
    _validate_partition(partition, root, config, entry, leaf)
    ledger = _load_bound_json(root, value.get("cumulative_ledger_binding", {}))
    _validate_ledger(
        ledger,
        root,
        config,
        pagination,
        predecessor,
        predecessor_partition,
        partition,
        catalog,
    )
    expected = _sealed(_derive_result(root, config, ledger, partition))
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["counts"]["processed_partition_prefix_length"] != 2
        or value["counts"]["cumulative_newly_processed_candidates"] != 34
        or value["counts"]["remaining_pending_formal_receipts"] != 11_213
        or value["complete_processed_partition_prefix"] is not True
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("cumulative formal result contract mismatch")


def build_result(root: Path, config_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_worker_result"])
    predecessor_partition, _ = _predecessor_partition(root, predecessor)
    catalog = _leaf_catalog(root, pagination)
    entry = _selected_next_leaf(catalog, predecessor)
    leaf = _load_bound_json(root, entry["leaf_binding"])
    preflight = _build_preflight(root, config)
    partition = _build_partition(root, config, entry, leaf, preflight, started)
    ledger_path = _artifact_path(root, config, "cumulative-ledger.json")
    if ledger_path.exists():
        ledger = _load_json(ledger_path)
    else:
        ledger = _sealed(
            _derive_ledger(
                root,
                config,
                pagination,
                predecessor,
                predecessor_partition,
                partition,
                catalog,
            )
        )
        _write_atomic_immutable(ledger_path, ledger, int(config["maximum_artifact_bytes"]))
    _validate_ledger(
        ledger,
        root,
        config,
        pagination,
        predecessor,
        predecessor_partition,
        partition,
        catalog,
    )
    result = _sealed(_derive_result(root, config, ledger, partition))
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
