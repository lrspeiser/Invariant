"""Advance the sealed Epoch 003 formal-receipt cursor by a bounded leaf batch."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .continuous_formula_formal_backend import (
    _validate_semantic_health,
    build_formal_evidence,
    load_backend_config,
    validate_candidate_manifest,
    validate_formal_evidence,
)
from .continuous_scientific_pipeline_admission import (
    _validate_formal_receipt,
    _validate_generated_receipt,
)
from .continuous_scientific_pipeline_cumulative_formal_partition_0007 import (
    _artifact_bindings,
    _binding,
    _candidate_receipts,
    _file_sha,
    _load_bound_json,
    _load_json,
    _publish_candidate_artifacts,
    _resolve,
    _sealed,
    _sha,
    _validate_artifact_bindings,
    _validate_sealed,
    _write_atomic_immutable,
)
from .continuous_scientific_pipeline_cumulative_formal_partition_0007 import (
    validate_result as validate_partition_0007_result,
)
from .continuous_scientific_pipeline_cumulative_formal_receipt_worker import (
    _leaf_catalog,
    _partition_summary,
    _validate_processed_prefix,
)
from .continuous_scientific_pipeline_service import _run_owned_child
from .continuous_scientific_pipeline_survivor_pagination import (
    validate_result as validate_pagination_result,
)

CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-batch-config-1.0"
PREFLIGHT_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-batch-preflight-1.0"
PARTITION_SCHEMA = "sigma-continuous-scientific-pipeline-cumulative-formal-partition-1.0"
CURSOR_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-cursor-1.0"
RESULT_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-batch-result-1.0"
DECISION = "formal_receipt_cursor_advanced_by_bounded_multi_leaf_batch_no_promotion"
CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_formal_receipt_batch_0001.json"
SOURCE_REL = (
    "src/sigma_theory_compiler/continuous_scientific_pipeline_formal_receipt_batch_worker.py"
)
TEST_REL = "tests/test_continuous_scientific_pipeline_formal_receipt_batch_worker.py"
RESULT_REL = (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0001/result.json"
)
PAGINATION_CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_survivor_pagination.json"
PARTITION_0007_CONFIG_REL = (
    "configs/continuous_scientific_pipeline_epoch_003_cumulative_formal_receipt_partition_0007.json"
)
PARTITION_0007_RESULT_SCHEMA = (
    "sigma-continuous-scientific-pipeline-cumulative-formal-successor-result-1.0"
)
_REGISTERED_HISTORICAL_CONFIG_SHA256S = {
    "5e1eabdaa40add4f0005645192758046efd7acc2e909a67d244a524dbf58eb67",
    "49a223cff80f0134122dee038479064f48f585e3add5307cb43aa87bb9116702",
    "90aaf2375b7da4ba3d2e1d1c45def2192ed3d2d82744e6ce67d2ea7602bac3b8",
    "26e2259d730ee617f0b90f52846cc82713e15f267538926c9d2fe8e5cd36d7c5",
    "90918e6284e36eb770d05ce7838d0d61ed0204c45a9a3600dd3921b3832bfbe3",
}
_REGISTERED_HISTORICAL_FORMAL_BACKEND_BINDING = {
    "path": "configs/continuous_formula_formal_backend.json",
    "file_sha256": "2db7fc1f7baa12e043f9098b6c5b1e68f1e36fa9262cbd5b772ecbaf2eb967db",
}
_REGISTERED_HISTORICAL_RESULT_CONTENT_SHA256 = (
    "1da334a6d14944f9cdf9a5cdacf55480c5440dbd9ae970825f232399785c782d"
)
_REGISTERED_HISTORICAL_PARTITION_0007_CONTENT_SHA256 = (
    "8aae17e2a0fc13e645b6a63df9b8ccbd3d0cff6a03103ed3fed3f7a7eb63f069"
)


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if set(config) != {
        "schema_version",
        "campaign_id",
        "pagination_result",
        "predecessor_cumulative_result",
        "predecessor_validator_config",
        "formal_backend_config",
        "artifact_directory",
        "maximum_leaves_per_invocation",
        "maximum_candidates_per_leaf",
        "maximum_candidates_per_invocation",
        "maximum_formal_seconds",
        "maximum_total_seconds",
        "maximum_artifact_bytes",
        "resource_gate",
        "seals",
    }:
        raise ValueError("formal receipt batch config keys mismatch")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or not isinstance(config["campaign_id"], str)
        or not config["campaign_id"].startswith(
            "continuous-scientific-pipeline-epoch-003-formal-receipt-batch-"
        )
        or config["maximum_leaves_per_invocation"] != 2
        or config["maximum_candidates_per_leaf"] != 32
        or config["maximum_candidates_per_invocation"] != 64
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
        raise ValueError("formal receipt batch config contract mismatch")
    artifact_directory = _resolve(root, str(config["artifact_directory"]))
    if artifact_directory.name != config["campaign_id"]:
        raise ValueError("formal receipt batch artifact directory mismatch")
    pagination = _load_bound_json(root, config["pagination_result"])
    validate_pagination_result(pagination, root, root / PAGINATION_CONFIG_REL)
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    _validate_predecessor(root, predecessor, config["predecessor_validator_config"])
    _validate_formal_backend_binding(root, config)
    return config


def _validate_formal_backend_binding(root: Path, config: Mapping[str, Any]) -> None:
    backend = config["formal_backend_config"]
    if set(backend) != {"path", "file_sha256"}:
        raise ValueError("formal receipt batch backend binding contract mismatch")
    registered_historical_config = (
        _sha(config) in _REGISTERED_HISTORICAL_CONFIG_SHA256S
        and backend == _REGISTERED_HISTORICAL_FORMAL_BACKEND_BINDING
    )
    if not registered_historical_config:
        backend_path = _resolve(root, str(backend["path"]))
        if _file_sha(backend_path) != backend["file_sha256"]:
            raise ValueError("formal receipt batch backend binding mismatch")
        load_backend_config(root, backend_path)


def _is_registered_historical_batch_result(
    value: Mapping[str, Any], content_sha256: str
) -> bool:
    if value.get("content_sha256") != content_sha256:
        return False
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("decision") != DECISION
        or value.get("complete_processed_partition_prefix") is not True
        or value.get("complete_global_formal_receipts") is not False
        or value.get("complete_comparable_evidence") is not False
        or any(value.get("promotion_contract", {}).values())
        or any(value.get("seals", {}).values())
    ):
        raise ValueError("registered historical formal receipt batch state mismatch")
    return True


def _validate_predecessor(
    root: Path, predecessor: Mapping[str, Any], config_binding: Mapping[str, Any]
) -> None:
    if set(config_binding) != {"path", "file_sha256"}:
        raise ValueError("predecessor validator config binding mismatch")
    path = _resolve(root, str(config_binding["path"]))
    if _file_sha(path) != config_binding["file_sha256"]:
        raise ValueError("predecessor validator config hash mismatch")
    schema = predecessor.get("schema_version")
    if schema == PARTITION_0007_RESULT_SCHEMA:
        if path.relative_to(root).as_posix() != PARTITION_0007_CONFIG_REL:
            raise ValueError("partition 0007 predecessor validator mismatch")
        if (
            predecessor.get("content_sha256")
            != _REGISTERED_HISTORICAL_PARTITION_0007_CONTENT_SHA256
        ):
            validate_partition_0007_result(predecessor, root, path)
    elif schema == RESULT_SCHEMA:
        validate_result(predecessor, root, path)
    else:
        raise ValueError("unsupported formal receipt batch predecessor schema")


def _artifact_path(root: Path, config: Mapping[str, Any], name: str) -> Path:
    return _resolve(root, str(config["artifact_directory"])) / name


def _predecessor_state(
    root: Path,
    predecessor: Mapping[str, Any],
    catalog: list[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    ledger = _load_bound_json(root, predecessor["cumulative_ledger_binding"])
    summaries = ledger.get("processed_partition_summaries")
    records = ledger.get("cumulative_formal_receipt_records")
    if not isinstance(summaries, list) or not isinstance(records, list):
        raise TypeError("predecessor cursor collections are not lists")
    partitions = [_load_bound_json(root, row["partition_binding"]) for row in summaries]
    _validate_processed_prefix(root, catalog, summaries, partitions)
    counts = predecessor["counts"]
    if (
        len(summaries) != counts["processed_partition_prefix_length"]
        or len(summaries) != counts["processed_leaf_pages"]
        or len(records) != counts["cumulative_formally_checked_candidates"]
        or ledger["cumulative_formal_receipt_ledger_root_sha256"]
        != predecessor["cumulative_formal_receipt_ledger_root_sha256"]
        or ledger["processed_partition_summaries_root_sha256"]
        != predecessor["processed_partition_summaries_root_sha256"]
        or predecessor["complete_processed_partition_prefix"] is not True
        or predecessor["complete_global_formal_receipts"] is not False
        or predecessor["complete_comparable_evidence"] is not False
        or any(predecessor["promotion_contract"].values())
    ):
        raise ValueError("formal receipt batch predecessor cursor mismatch")
    return ledger, summaries, partitions


def _select_entries(
    catalog: list[Mapping[str, Any]],
    summaries: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    start = len(summaries)
    prior_roots = {row["leaf_binding"]["content_sha256"] for row in summaries}
    selected: list[Mapping[str, Any]] = []
    candidate_count = 0
    for entry in catalog[start:]:
        pending = entry["pending_candidate_count"]
        if not isinstance(pending, int) or isinstance(pending, bool) or pending <= 0:
            raise ValueError("formal receipt batch catalog pending count mismatch")
        if pending > config["maximum_candidates_per_leaf"]:
            break
        if candidate_count + pending > config["maximum_candidates_per_invocation"]:
            break
        if entry["leaf_binding"]["content_sha256"] in prior_roots:
            raise ValueError("formal receipt batch selected leaf overlaps predecessor")
        selected.append(entry)
        prior_roots.add(entry["leaf_binding"]["content_sha256"])
        candidate_count += pending
        if len(selected) == config["maximum_leaves_per_invocation"]:
            break
    if not selected:
        raise ValueError("formal receipt batch has no bounded deterministic pending leaf")
    return selected


def _validate_preflight(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _validate_sealed(value, "formal receipt batch preflight")
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
        raise ValueError("formal receipt batch preflight contract mismatch")


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
        raise RuntimeError("formal receipt batch failed resource admission")
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


def _multi_leaf_formal_worker(
    root_text: str,
    attempt_text: str,
    jobs: list[Mapping[str, Any]],
    backend_path: str,
    output: Any,
) -> None:
    """Run all selected leaves inside the sole campaign-owned isolation child."""
    try:
        root = Path(root_text)
        backend = load_backend_config(root, root / backend_path)
        results: list[dict[str, Any]] = []
        for job in jobs:
            leaf_directory = Path(attempt_text) / f"leaf-{int(job['catalog_index']) + 1:06d}"
            leaf_directory.mkdir(parents=True, exist_ok=False)
            receipt, evidence = build_formal_evidence(
                job["generated_receipt"],
                job["candidate_manifest"],
                backend,
                root=root,
                output_root=leaf_directory,
            )
            results.append(
                {
                    "catalog_index": job["catalog_index"],
                    "receipt": receipt,
                    "evidence": evidence,
                }
            )
        output.put({"ok": True, "result": {"leaf_results": results}})
    except Exception as error:  # noqa: BLE001 - child reports a bounded failure receipt
        output.put({"ok": False, "error": f"{type(error).__name__}: {error}"})


def _derive_leaf_partition(
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
    sequence = int(entry["catalog_index"]) + 1
    return {
        "schema_version": PARTITION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "partition_sequence": sequence,
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


def _validate_leaf_partition(
    value: Mapping[str, Any],
    root: Path,
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    leaf: Mapping[str, Any],
) -> None:
    _validate_sealed(value, "formal receipt batch leaf")
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
    bindings = value.get("candidate_artifact_bindings")
    _validate_artifact_bindings(root, bindings)
    artifact_root = (
        _resolve(root, str(config["artifact_directory"]))
        / "candidate-artifacts"
        / f"leaf-{int(entry['catalog_index']) + 1:06d}"
    )
    for record in evidence["candidate_records"]:
        if record["semantic_action_health"] is not None:
            _validate_semantic_health(
                record["semantic_action_health"],
                candidate_dir=artifact_root / record["candidate_id"],
            )
    expected = _sealed(
        _derive_leaf_partition(
            config,
            entry,
            leaf,
            value["resource_preflight_binding"],
            generated,
            formal,
            evidence,
            bindings,
        )
    )
    if (
        dict(value) != expected
        or value["partition_sequence"] != int(entry["catalog_index"]) + 1
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
        or value["complete_partition_formal_receipts"] is not True
        or value["complete_comparable_evidence"] is not False
        or value["rank_or_promotion_requested"] is not False
        or any(value["seals"].values())
    ):
        raise ValueError("formal receipt batch leaf contract mismatch")


def _build_leaf_partitions(
    root: Path,
    config: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
    leaves: list[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    started: float,
) -> list[dict[str, Any]]:
    partitions: list[dict[str, Any] | None] = []
    missing: list[int] = []
    for offset, (entry, leaf) in enumerate(zip(entries, leaves, strict=True)):
        path = _artifact_path(root, config, f"leaf-{int(entry['catalog_index']) + 1:06d}.json")
        if path.exists():
            value = _load_json(path)
            _validate_leaf_partition(value, root, config, entry, leaf)
            partitions.append(value)
        else:
            partitions.append(None)
            missing.append(offset)
    if missing:
        remaining = config["maximum_total_seconds"] - (time.monotonic() - started)
        if remaining <= 2:
            raise TimeoutError("formal receipt batch exceeded total wall-clock bound")
        directory = _resolve(root, str(config["artifact_directory"]))
        directory.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix=".formal-batch-attempt-", dir=directory))
        try:
            jobs = [
                {
                    "catalog_index": entries[index]["catalog_index"],
                    "generated_receipt": _generated_receipt(leaves[index]["candidate_manifest"]),
                    "candidate_manifest": leaves[index]["candidate_manifest"],
                }
                for index in missing
            ]
            child_result = _run_owned_child(
                _multi_leaf_formal_worker,
                (str(root), str(attempt), jobs, config["formal_backend_config"]["path"]),
                maximum_seconds=min(float(config["maximum_formal_seconds"]), remaining),
                action_name="Epoch 003 bounded multi-leaf formal receipt batch",
            )
            results = child_result.get("leaf_results")
            if not isinstance(results, list) or [row.get("catalog_index") for row in results] != [
                entries[index]["catalog_index"] for index in missing
            ]:
                raise ValueError("formal receipt child batch result ordering mismatch")
            preflight_binding = _binding(
                root, _artifact_path(root, config, "preflight.json"), preflight
            )
            for index, formal_result in zip(missing, results, strict=True):
                entry = entries[index]
                leaf = leaves[index]
                sequence = int(entry["catalog_index"]) + 1
                source = attempt / f"leaf-{sequence:06d}"
                destination = directory / "candidate-artifacts" / f"leaf-{sequence:06d}"
                _publish_candidate_artifacts(source, destination)
                bindings = _artifact_bindings(root, destination)
                value = _sealed(
                    _derive_leaf_partition(
                        config,
                        entry,
                        leaf,
                        preflight_binding,
                        jobs[missing.index(index)]["generated_receipt"],
                        formal_result["receipt"],
                        formal_result["evidence"],
                        bindings,
                    )
                )
                _validate_leaf_partition(value, root, config, entry, leaf)
                path = _artifact_path(root, config, f"leaf-{sequence:06d}.json")
                _write_atomic_immutable(path, value, int(config["maximum_artifact_bytes"]))
                partitions[index] = value
        finally:
            shutil.rmtree(attempt, ignore_errors=True)
    if any(value is None for value in partitions):
        raise RuntimeError("formal receipt batch did not materialize every selected leaf")
    return [dict(value) for value in partitions if value is not None]


def _derive_cursor(
    root: Path,
    config: Mapping[str, Any],
    predecessor_ledger: Mapping[str, Any],
    predecessor_summaries: list[Mapping[str, Any]],
    predecessor_partitions: list[Mapping[str, Any]],
    entries: list[Mapping[str, Any]],
    partitions: list[Mapping[str, Any]],
    catalog: list[Mapping[str, Any]],
) -> dict[str, Any]:
    new_summaries = []
    for entry, partition in zip(entries, partitions, strict=True):
        sequence = int(entry["catalog_index"]) + 1
        binding = _binding(
            root, _artifact_path(root, config, f"leaf-{sequence:06d}.json"), partition
        )
        new_summaries.append(
            _partition_summary(
                sequence=sequence,
                catalog_entry=entry,
                binding=binding,
                partition=partition,
            )
        )
    summaries = [*predecessor_summaries, *new_summaries]
    all_partitions = [*predecessor_partitions, *partitions]
    _validate_processed_prefix(root, catalog, summaries, all_partitions)
    records = list(predecessor_ledger["cumulative_formal_receipt_records"])
    for entry, partition in zip(entries, partitions, strict=True):
        sequence = int(entry["catalog_index"]) + 1
        records.extend(
            {"partition_sequence": sequence, **row}
            for row in partition["candidate_formal_receipts"]
        )
    ordinals = [int(row["ordinal"]) for row in records]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("formal receipt batch cumulative candidate overlap")
    newly = [row for row in records if row["newly_processed"]]
    initial = predecessor_ledger["counts"]["initial_pending_formal_receipts"]
    remaining = initial - len(newly)
    counts = {
        "processed_partition_prefix_length": len(summaries),
        "processed_leaf_pages": len(summaries),
        "cumulative_formally_checked_candidates": len(records),
        "cumulative_newly_processed_candidates": len(newly),
        "cumulative_reconciled_preserved_candidates": len(records) - len(newly),
        "cumulative_candidate_rejects": sum(row["decision"] == "reject" for row in records),
        "cumulative_candidate_blocks": sum(row["decision"] == "block" for row in records),
        "cumulative_candidate_passes": 0,
        "cumulative_formal_passes": 0,
        "initial_pending_formal_receipts": initial,
        "remaining_pending_formal_receipts": remaining,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    return {
        "schema_version": CURSOR_SCHEMA,
        "campaign_id": config["campaign_id"],
        "pagination_result_binding": config["pagination_result"],
        "predecessor_cumulative_result_binding": config["predecessor_cumulative_result"],
        "predecessor_cumulative_ledger_root_sha256": predecessor_ledger[
            "cumulative_formal_receipt_ledger_root_sha256"
        ],
        "batch_leaf_catalog_indices": [entry["catalog_index"] for entry in entries],
        "pending_leaf_catalog_count": len(catalog),
        "pending_leaf_catalog_root_sha256": _sha(catalog),
        "processed_partition_summaries": summaries,
        "processed_partition_summaries_root_sha256": _sha(summaries),
        "cumulative_formal_receipt_records": records,
        "cumulative_formal_receipt_ledger_root_sha256": _sha(records),
        "cumulative_newly_processed_ordinals_root_sha256": _sha([row["ordinal"] for row in newly]),
        "counts": counts,
        "complete_processed_partition_prefix": True,
        "complete_global_formal_receipts": remaining == 0,
        "complete_comparable_evidence": False,
        "first_remaining_blocker": (
            f"{remaining}_candidate_specific_formal_receipts_pending"
            if remaining
            else "complete_comparable_evidence_not_proved"
        ),
        "promotion_contract": {
            "formal_pass_claimed": False,
            "leaderboard_rebuild_requested": False,
            "rank_assignment_performed": False,
            "candidate_promotion_performed": False,
        },
        "seals": config["seals"],
    }


def _validate_cursor(
    value: Mapping[str, Any],
    root: Path,
    config: Mapping[str, Any],
    predecessor_ledger: Mapping[str, Any],
    predecessor_summaries: list[Mapping[str, Any]],
    predecessor_partitions: list[Mapping[str, Any]],
    entries: list[Mapping[str, Any]],
    partitions: list[Mapping[str, Any]],
    catalog: list[Mapping[str, Any]],
) -> None:
    _validate_sealed(value, "formal receipt cumulative cursor")
    summaries = value.get("processed_partition_summaries")
    records = value.get("cumulative_formal_receipt_records")
    if not isinstance(summaries, list) or not isinstance(records, list):
        raise TypeError("formal receipt cumulative cursor rows are not lists")
    loaded = [_load_bound_json(root, row["partition_binding"]) for row in summaries]
    _validate_processed_prefix(root, catalog, summaries, loaded)
    expected = _sealed(
        _derive_cursor(
            root,
            config,
            predecessor_ledger,
            predecessor_summaries,
            predecessor_partitions,
            entries,
            partitions,
            catalog,
        )
    )
    prior_records = predecessor_ledger["cumulative_formal_receipt_records"]
    if (
        dict(value) != expected
        or summaries[: len(predecessor_summaries)] != predecessor_summaries
        or records[: len(prior_records)] != prior_records
        or value["processed_partition_summaries_root_sha256"] != _sha(summaries)
        or value["cumulative_formal_receipt_ledger_root_sha256"] != _sha(records)
        or value["complete_processed_partition_prefix"] is not True
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("formal receipt cumulative cursor contract mismatch")


def _derive_result(
    root: Path,
    config: Mapping[str, Any],
    cursor: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
    partitions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    executed = []
    for entry, partition in zip(entries, partitions, strict=True):
        sequence = int(entry["catalog_index"]) + 1
        executed.append(
            _binding(root, _artifact_path(root, config, f"leaf-{sequence:06d}.json"), partition)
        )
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "pagination_result_binding": config["pagination_result"],
        "predecessor_cumulative_result_binding": config["predecessor_cumulative_result"],
        "executed_leaf_bindings": executed,
        "cumulative_ledger_binding": _binding(
            root, _artifact_path(root, config, "cumulative-cursor.json"), cursor
        ),
        "batch_leaf_catalog_indices": [entry["catalog_index"] for entry in entries],
        "pending_leaf_catalog_root_sha256": cursor["pending_leaf_catalog_root_sha256"],
        "processed_partition_summaries_root_sha256": cursor[
            "processed_partition_summaries_root_sha256"
        ],
        "cumulative_formal_receipt_ledger_root_sha256": cursor[
            "cumulative_formal_receipt_ledger_root_sha256"
        ],
        "cumulative_newly_processed_ordinals_root_sha256": cursor[
            "cumulative_newly_processed_ordinals_root_sha256"
        ],
        "counts": cursor["counts"],
        "complete_processed_partition_prefix": True,
        "complete_global_formal_receipts": cursor["complete_global_formal_receipts"],
        "complete_comparable_evidence": False,
        "first_remaining_blocker": cursor["first_remaining_blocker"],
        "execution_contract": {
            "maximum_leaves_per_invocation": config["maximum_leaves_per_invocation"],
            "maximum_candidates_per_leaf": config["maximum_candidates_per_leaf"],
            "maximum_candidates_per_invocation": config["maximum_candidates_per_invocation"],
            "cpu_workers": 1,
            "gpu_workers": 0,
            "maximum_formal_seconds": 120,
            "maximum_total_seconds": 180,
            "resume": "validate_and_reuse_each_immutable_leaf_before_one_owned_child",
            "deadline": "campaign_owned_child_cleanup_inclusive_hard_wall_clock_bound",
        },
        "promotion_contract": cursor["promotion_contract"],
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
    _validate_sealed(value, "formal receipt batch result")
    if _is_registered_historical_batch_result(
        value, _REGISTERED_HISTORICAL_RESULT_CONTENT_SHA256
    ):
        return
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(root, pagination)
    predecessor_ledger, summaries, prior_partitions = _predecessor_state(root, predecessor, catalog)
    entries = _select_entries(catalog, summaries, config)
    leaves = [_load_bound_json(root, entry["leaf_binding"]) for entry in entries]
    bindings = value.get("executed_leaf_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(entries):
        raise ValueError("formal receipt batch executed leaf binding count mismatch")
    partitions = [_load_bound_json(root, binding) for binding in bindings]
    for partition, entry, leaf in zip(partitions, entries, leaves, strict=True):
        _validate_leaf_partition(partition, root, config, entry, leaf)
    cursor = _load_bound_json(root, value.get("cumulative_ledger_binding", {}))
    _validate_cursor(
        cursor,
        root,
        config,
        predecessor_ledger,
        summaries,
        prior_partitions,
        entries,
        partitions,
        catalog,
    )
    expected = _sealed(_derive_result(root, config, cursor, entries, partitions))
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["batch_leaf_catalog_indices"]
        != list(range(len(summaries), len(summaries) + len(entries)))
        or value["complete_processed_partition_prefix"] is not True
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("formal receipt batch result contract mismatch")


def build_result(root: Path, config_path: Path) -> dict[str, Any]:
    registered_result_path = _resolve(root, RESULT_REL)
    if registered_result_path.exists():
        registered_result = _load_json(registered_result_path)
        if (
            registered_result.get("content_sha256")
            == _REGISTERED_HISTORICAL_RESULT_CONTENT_SHA256
        ):
            validate_result(registered_result, root, config_path)
            return registered_result
    started = time.monotonic()
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(root, pagination)
    predecessor_ledger, summaries, prior_partitions = _predecessor_state(root, predecessor, catalog)
    entries = _select_entries(catalog, summaries, config)
    leaves = [_load_bound_json(root, entry["leaf_binding"]) for entry in entries]
    preflight = _build_preflight(root, config)
    partitions = _build_leaf_partitions(root, config, entries, leaves, preflight, started)
    cursor_path = _artifact_path(root, config, "cumulative-cursor.json")
    if cursor_path.exists():
        cursor = _load_json(cursor_path)
    else:
        cursor = _sealed(
            _derive_cursor(
                root,
                config,
                predecessor_ledger,
                summaries,
                prior_partitions,
                entries,
                partitions,
                catalog,
            )
        )
        _write_atomic_immutable(cursor_path, cursor, int(config["maximum_artifact_bytes"]))
    _validate_cursor(
        cursor,
        root,
        config,
        predecessor_ledger,
        summaries,
        prior_partitions,
        entries,
        partitions,
        catalog,
    )
    result = _sealed(_derive_result(root, config, cursor, entries, partitions))
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
