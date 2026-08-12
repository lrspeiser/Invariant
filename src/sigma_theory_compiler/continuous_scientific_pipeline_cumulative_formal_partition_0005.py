"""Append exactly partition 0005 to the immutable Epoch 003 formal receipt ledger."""

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
from .continuous_scientific_pipeline_cumulative_formal_partition_0004 import (
    validate_result as validate_predecessor_result,
)
from .continuous_scientific_pipeline_cumulative_formal_receipt_worker import (
    _leaf_catalog,
    _partition_summary,
    _validate_processed_prefix,
)
from .continuous_scientific_pipeline_cumulative_formal_receipt_worker import (
    _load_bound_json as _load_predecessor_bound_json,
)
from .continuous_scientific_pipeline_service import _real_formal_worker, _run_owned_child
from .continuous_scientific_pipeline_survivor_pagination import (
    validate_result as validate_pagination_result,
)

CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-successor-config-1.0"
PREFLIGHT_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-preflight-1.0"
PARTITION_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-partition-1.0"
LEDGER_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-ledger-1.0"
RESULT_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-successor-result-1.0"
DECISION = "cumulative_formal_receipt_prefix_advanced_to_partition_0005_no_promotion"
CONFIG_REL = (
    "configs/continuous_scientific_pipeline_epoch_003_cumulative_formal_receipt_partition_0005.json"
)
SOURCE_REL = (
    "src/sigma_theory_compiler/continuous_scientific_pipeline_cumulative_formal_partition_0005.py"
)
TEST_REL = "tests/test_continuous_scientific_pipeline_cumulative_formal_partition_0005.py"
RESULT_REL = (
    "runs/engine/continuous-scientific-pipeline-epoch-003-cumulative-formal-partition-0005/"
    "result.json"
)
PAGINATION_CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_survivor_pagination.json"
PREDECESSOR_CONFIG_REL = (
    "configs/continuous_scientific_pipeline_epoch_003_cumulative_formal_receipt_partition_0004.json"
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
        raise ValueError("partition 0005 JSON binding contract mismatch")
    path = _resolve(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("partition 0005 JSON binding file hash mismatch")
    value = _load_json(path)
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("partition 0005 JSON binding content hash mismatch")
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
        raise RuntimeError("partition 0005 artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"immutable partition 0005 artifact differs: {path.name}")
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
                    f"immutable partition 0005 artifact differs: {path.name}"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "pagination_result",
            "predecessor_cumulative_result",
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
        or config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"]
        != "continuous-scientific-pipeline-epoch-003-cumulative-formal-partition-0005"
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
        raise ValueError("partition 0005 config contract mismatch")
    _resolve(root, str(config["artifact_directory"]))
    pagination = _load_bound_json(root, config["pagination_result"])
    validate_pagination_result(pagination, root, root / PAGINATION_CONFIG_REL)
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    validate_predecessor_result(predecessor, root, root / PREDECESSOR_CONFIG_REL)
    backend = config["formal_backend_config"]
    if set(backend) != {"path", "file_sha256"}:
        raise ValueError("partition 0005 backend binding contract mismatch")
    backend_path = _resolve(root, str(backend["path"]))
    if _file_sha(backend_path) != backend["file_sha256"]:
        raise ValueError("partition 0005 backend binding mismatch")
    load_backend_config(root, backend_path)
    return config


def _artifact_path(root: Path, config: Mapping[str, Any], name: str) -> Path:
    return _resolve(root, str(config["artifact_directory"])) / name


def _predecessor_state(
    root: Path,
    predecessor: Mapping[str, Any],
    catalog: list[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    ledger = _load_bound_json(root, predecessor["cumulative_ledger_binding"])
    summaries = ledger["processed_partition_summaries"]
    partitions = [
        _load_predecessor_bound_json(root, summary["partition_binding"]) for summary in summaries
    ]
    _validate_processed_prefix(root, catalog, summaries, partitions)
    if (
        len(summaries) != 4
        or [row["leaf_catalog_index"] for row in summaries] != [0, 1, 2, 3]
        or ledger["cumulative_formal_receipt_ledger_root_sha256"]
        != predecessor["cumulative_formal_receipt_ledger_root_sha256"]
    ):
        raise ValueError("partition 0005 predecessor prefix mismatch")
    return ledger, summaries, partitions


def _selected_entry(
    catalog: list[Mapping[str, Any]], summaries: list[Mapping[str, Any]]
) -> Mapping[str, Any]:
    index = len(summaries)
    if index != 4 or len(catalog) <= index:
        raise ValueError("partition 0005 deterministic next leaf unavailable")
    entry = catalog[index]
    prior_roots = {row["leaf_binding"]["content_sha256"] for row in summaries}
    if entry["leaf_binding"]["content_sha256"] in prior_roots:
        raise ValueError("partition 0005 selected leaf overlaps predecessor prefix")
    return entry


def _validate_preflight(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _validate_sealed(value, "partition 0005 preflight")
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
        raise ValueError("partition 0005 preflight contract mismatch")


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
        raise RuntimeError("partition 0005 failed resource admission")
    _validate_preflight(value, config)
    _write_atomic_immutable(path, value, int(config["maximum_artifact_bytes"]))
    return value


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
        raise TypeError("partition 0005 artifact bindings are not a list")
    paths = []
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or _file_sha(_resolve(root, str(binding["path"]))) != binding["file_sha256"]
        ):
            raise ValueError("partition 0005 artifact binding mismatch")
        paths.append(binding["path"])
    if paths != sorted(set(paths)):
        raise ValueError("partition 0005 artifact binding ordering mismatch")


def _candidate_receipts(
    leaf: Mapping[str, Any], evidence: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    queue = {int(row["ordinal"]): row for row in leaf["formal_receipt_queue"]}
    return [
        {
            "candidate_id": record["candidate_id"],
            "ordinal": record["ordinal"],
            "source_queue_entry_sha256": _sha(queue[int(record["ordinal"])]),
            "source_queue_state": queue[int(record["ordinal"])]["state"],
            "prior_formal_receipt_binding_sha256": None,
            "newly_processed": True,
            "formal_receipt_content_sha256": receipt["content_sha256"],
            "formal_evidence_content_sha256": evidence["content_sha256"],
            "formal_evidence_record_sha256": _sha(record),
            "covariant_mapping_payload_sha256": record["covariant_mapping_payload_sha256"],
            "semantic_action_health_sha256": record["semantic_action_health_sha256"],
            "decision": record["decision"],
            "first_blocker": record["first_blocker"],
        }
        for record in evidence["candidate_records"]
    ]


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
        "partition_sequence": 5,
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
            "newly_processed_candidates": len(receipts),
            "reconciled_preserved_candidates": 0,
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
    _validate_sealed(value, "partition 0005")
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
        or entry["catalog_index"] != 4
        or value["selected_leaf_page_binding"] != entry["leaf_binding"]
        or value["selected_leaf_hierarchy_path"] != entry["hierarchy_path"]
        or manifest != leaf["candidate_manifest"]
        or any(
            row["state"] != "pending_candidate_specific_formal_receipt"
            for row in leaf["formal_receipt_queue"]
        )
        or generated["candidate_root_sha256"] != manifest["candidate_root_sha256"]
        or formal["candidate_root_sha256"] != generated["candidate_root_sha256"]
        or evidence["candidate_manifest_sha256"] != manifest["content_sha256"]
        or evidence["generated_receipt_sha256"] != generated["content_sha256"]
        or value["counts"]["partition_candidates"] != entry["pending_candidate_count"]
        or value["counts"]["newly_processed_candidates"] != entry["pending_candidate_count"]
        or value["counts"]["reconciled_preserved_candidates"] != 0
        or value["complete_partition_formal_receipts"] is not True
        or value["complete_comparable_evidence"] is not False
        or value["rank_or_promotion_requested"] is not False
        or any(value["seals"].values())
    ):
        raise ValueError("partition 0005 contract mismatch")


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
                raise ValueError("immutable partition 0005 candidate artifacts differ")
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
    path = _artifact_path(root, config, "partition-0005.json")
    if path.exists():
        value = _load_json(path)
        _validate_partition(value, root, config, entry, leaf)
        return value
    remaining = config["maximum_total_seconds"] - (time.monotonic() - started)
    if remaining <= 2:
        raise TimeoutError("partition 0005 exceeded total wall-clock bound")
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
            action_name="Epoch 003 cumulative formal receipt partition 0005",
        )
        formal = formal_result["receipt"]
        evidence = formal_result["evidence"]
        _validate_formal_receipt(formal)
        validate_formal_evidence(evidence)
        artifacts = directory / "candidate-artifacts"
        _publish_candidate_artifacts(attempt, artifacts)
        bindings = _artifact_bindings(root, artifacts)
        value = _sealed(
            _derive_partition(
                config,
                entry,
                leaf,
                _binding(root, _artifact_path(root, config, "preflight.json"), preflight),
                generated,
                formal,
                evidence,
                bindings,
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
    predecessor_ledger: Mapping[str, Any],
    predecessor_summaries: list[Mapping[str, Any]],
    predecessor_partitions: list[Mapping[str, Any]],
    partition: Mapping[str, Any],
    catalog: list[Mapping[str, Any]],
) -> dict[str, Any]:
    partition_binding = _binding(
        root, _artifact_path(root, config, "partition-0005.json"), partition
    )
    summary = _partition_summary(
        sequence=5,
        catalog_entry=catalog[4],
        binding=partition_binding,
        partition=partition,
    )
    summaries = [*predecessor_summaries, summary]
    partitions = [*predecessor_partitions, partition]
    _validate_processed_prefix(root, catalog, summaries, partitions)
    records = [
        *predecessor_ledger["cumulative_formal_receipt_records"],
        *({"partition_sequence": 5, **row} for row in partition["candidate_formal_receipts"]),
    ]
    ordinals = [int(row["ordinal"]) for row in records]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("partition 0005 cumulative candidate overlap")
    newly = [row for row in records if row["newly_processed"]]
    initial = predecessor_ledger["counts"]["initial_pending_formal_receipts"]
    counts = {
        "processed_partition_prefix_length": 5,
        "processed_leaf_pages": 5,
        "cumulative_formally_checked_candidates": len(records),
        "cumulative_newly_processed_candidates": len(newly),
        "cumulative_reconciled_preserved_candidates": len(records) - len(newly),
        "cumulative_candidate_rejects": sum(row["decision"] == "reject" for row in records),
        "cumulative_candidate_blocks": sum(row["decision"] == "block" for row in records),
        "cumulative_candidate_passes": 0,
        "cumulative_formal_passes": 0,
        "initial_pending_formal_receipts": initial,
        "remaining_pending_formal_receipts": initial - len(newly),
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    return {
        "schema_version": LEDGER_SCHEMA,
        "campaign_id": config["campaign_id"],
        "pagination_result_binding": config["pagination_result"],
        "predecessor_cumulative_result_binding": config["predecessor_cumulative_result"],
        "predecessor_cumulative_ledger_root_sha256": predecessor_ledger[
            "cumulative_formal_receipt_ledger_root_sha256"
        ],
        "pending_leaf_catalog_count": len(catalog),
        "pending_leaf_catalog_root_sha256": _sha(catalog),
        "processed_partition_summaries": summaries,
        "processed_partition_summaries_root_sha256": _sha(summaries),
        "cumulative_formal_receipt_records": records,
        "cumulative_formal_receipt_ledger_root_sha256": _sha(records),
        "cumulative_newly_processed_ordinals_root_sha256": _sha([row["ordinal"] for row in newly]),
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
    predecessor_ledger: Mapping[str, Any],
    predecessor_summaries: list[Mapping[str, Any]],
    predecessor_partitions: list[Mapping[str, Any]],
    partition: Mapping[str, Any],
    catalog: list[Mapping[str, Any]],
) -> None:
    _validate_sealed(value, "partition 0005 cumulative ledger")
    summaries = value.get("processed_partition_summaries")
    records = value.get("cumulative_formal_receipt_records")
    if not isinstance(summaries, list) or not isinstance(records, list):
        raise TypeError("partition 0005 ledger rows are not lists")
    loaded = [_load_bound_json(root, row["partition_binding"]) for row in summaries]
    _validate_processed_prefix(root, catalog, summaries, loaded)
    expected = _sealed(
        _derive_ledger(
            root,
            config,
            predecessor_ledger,
            predecessor_summaries,
            predecessor_partitions,
            partition,
            catalog,
        )
    )
    if (
        dict(value) != expected
        or summaries[:4] != predecessor_summaries
        or records[: len(predecessor_ledger["cumulative_formal_receipt_records"])]
        != predecessor_ledger["cumulative_formal_receipt_records"]
        or value["processed_partition_summaries_root_sha256"] != _sha(summaries)
        or value["cumulative_formal_receipt_ledger_root_sha256"] != _sha(records)
        or value["counts"]["remaining_pending_formal_receipts"] != 11_167
        or value["complete_processed_partition_prefix"] is not True
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("partition 0005 cumulative ledger contract mismatch")


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
        "predecessor_cumulative_result_binding": config["predecessor_cumulative_result"],
        "executed_partition_binding": _binding(
            root, _artifact_path(root, config, "partition-0005.json"), partition
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
            "resume": "validate_predecessor_prefix_and_reuse_immutable_partition_0005",
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
    _validate_sealed(value, "partition 0005 result")
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(root, pagination)
    predecessor_ledger, summaries, partitions = _predecessor_state(root, predecessor, catalog)
    entry = _selected_entry(catalog, summaries)
    leaf = _load_bound_json(root, entry["leaf_binding"])
    partition = _load_bound_json(root, value.get("executed_partition_binding", {}))
    _validate_partition(partition, root, config, entry, leaf)
    ledger = _load_bound_json(root, value.get("cumulative_ledger_binding", {}))
    _validate_ledger(
        ledger,
        root,
        config,
        predecessor_ledger,
        summaries,
        partitions,
        partition,
        catalog,
    )
    expected = _sealed(_derive_result(root, config, ledger, partition))
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["counts"]["processed_partition_prefix_length"] != 5
        or value["counts"]["cumulative_newly_processed_candidates"] != 80
        or value["counts"]["remaining_pending_formal_receipts"] != 11_167
        or value["complete_processed_partition_prefix"] is not True
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("partition 0005 result contract mismatch")


def build_result(root: Path, config_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(root, pagination)
    predecessor_ledger, summaries, partitions = _predecessor_state(root, predecessor, catalog)
    entry = _selected_entry(catalog, summaries)
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
                predecessor_ledger,
                summaries,
                partitions,
                partition,
                catalog,
            )
        )
        _write_atomic_immutable(ledger_path, ledger, int(config["maximum_artifact_bytes"]))
    _validate_ledger(
        ledger,
        root,
        config,
        predecessor_ledger,
        summaries,
        partitions,
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
