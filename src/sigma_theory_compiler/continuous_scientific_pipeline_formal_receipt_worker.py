"""Bounded formal receipt worker for one immutable Epoch 003 survivor page."""

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
from .continuous_scientific_pipeline_service import _real_formal_worker, _run_owned_child
from .continuous_scientific_pipeline_survivor_pagination import (
    validate_result as validate_pagination_result,
)

CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-worker-config-1.0"
PREFLIGHT_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-preflight-1.0"
PARTITION_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-partition-1.0"
RESULT_SCHEMA = "sigma-continuous-scientific-pipeline-formal-receipt-worker-result-1.0"
DECISION = "bounded_partition_formal_receipts_complete_global_queue_incomplete_no_promotion"
CONFIG_REL = (
    "configs/continuous_scientific_pipeline_epoch_003_formal_receipt_worker_partition_0001.json"
)
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_formal_receipt_worker.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_formal_receipt_worker.py"
RESULT_REL = (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-worker-partition-0001/"
    "result.json"
)
PAGINATION_CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_survivor_pagination.json"
PARTITION_NAME = "partition-0001.json"
PREFLIGHT_NAME = "preflight.json"


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
        raise ValueError("formal worker JSON binding contract mismatch")
    path = _resolve(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("formal worker JSON binding file hash mismatch")
    value = _load_json(path)
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("formal worker JSON binding content hash mismatch")
    return value


def _write_atomic_immutable(path: Path, value: Mapping[str, Any], maximum_bytes: int) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if len(raw) > maximum_bytes:
        raise RuntimeError("formal receipt artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"immutable formal receipt artifact differs: {path.name}")
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
                    f"immutable formal receipt artifact differs: {path.name}"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


def _binding(root: Path, path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": _file_sha(path),
        "content_sha256": str(value["content_sha256"]),
    }


def _selected_leaf_hierarchy_path(
    root: Path,
    pagination: Mapping[str, Any],
    selected_binding: Mapping[str, Any],
    selected_leaf: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Prove that the selected leaf is reachable from a registered batch index."""

    target_interval = selected_leaf["ordinal_interval"]
    target_batch = selected_leaf["batch_index"]
    for index_binding in pagination["ordered_batch_index_bindings"]:
        index = _load_bound_json(root, index_binding)
        if index["batch_index"] != target_batch:
            continue
        for worker_binding in index["ordered_worker_root_bindings"]:
            worker = _load_bound_json(root, worker_binding)
            interval = worker["ordinal_interval"]
            if not (
                interval["start_ordinal"] <= target_interval["start_ordinal"]
                and interval["end_ordinal_exclusive"]
                >= target_interval["end_ordinal_exclusive"]
            ):
                continue
            path = [dict(index_binding), dict(worker_binding)]
            binding = worker_binding
            node = worker
            while node["leaf_page"] is not True:
                matches = []
                for child_binding in node["child_bindings"]:
                    child = _load_bound_json(root, child_binding)
                    child_interval = child["ordinal_interval"]
                    if (
                        child_interval["start_ordinal"]
                        <= target_interval["start_ordinal"]
                        and child_interval["end_ordinal_exclusive"]
                        >= target_interval["end_ordinal_exclusive"]
                    ):
                        matches.append((child_binding, child))
                if len(matches) != 1:
                    raise ValueError("selected formal receipt leaf hierarchy path is ambiguous")
                binding, node = matches[0]
                path.append(dict(binding))
            if dict(binding) == dict(selected_binding) and dict(node) == dict(selected_leaf):
                return path
    raise ValueError("selected formal receipt leaf is not reachable from pagination hierarchy")


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = _load_json(path)
    expected = {
        "schema_version",
        "campaign_id",
        "pagination_result",
        "selected_leaf_page",
        "formal_backend_config",
        "artifact_directory",
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
        != "continuous-scientific-pipeline-epoch-003-formal-receipt-worker-partition-0001"
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
        raise ValueError("formal receipt worker config contract mismatch")
    _resolve(root, str(config["artifact_directory"]))
    pagination = _load_bound_json(root, config["pagination_result"])
    validate_pagination_result(pagination, root, root / PAGINATION_CONFIG_REL)
    leaf = _load_bound_json(root, config["selected_leaf_page"])
    _selected_leaf_hierarchy_path(
        root, pagination, config["selected_leaf_page"], leaf
    )
    manifest = leaf.get("candidate_manifest", {})
    validate_candidate_manifest(manifest)
    queue = leaf.get("formal_receipt_queue")
    if (
        leaf.get("leaf_page") is not True
        or manifest["sample_complete"] is not True
        or not isinstance(queue, list)
        or len(queue) != len(manifest["survivor_records"])
        or not 0 < len(queue) <= config["maximum_partition_candidates"]
        or leaf.get("formal_receipt_queue_root_sha256") != _sha(queue)
        or not any(row["state"] == "pending_candidate_specific_formal_receipt" for row in queue)
    ):
        raise ValueError("selected formal receipt leaf is not an eligible complete partition")
    backend_binding = config["formal_backend_config"]
    if set(backend_binding) != {"path", "file_sha256"}:
        raise ValueError("formal backend config binding contract mismatch")
    backend_path = _resolve(root, str(backend_binding["path"]))
    if _file_sha(backend_path) != backend_binding["file_sha256"]:
        raise ValueError("formal backend config binding mismatch")
    load_backend_config(root, backend_path)
    return config


def _preflight_path(root: Path, config: Mapping[str, Any]) -> Path:
    return _resolve(root, str(config["artifact_directory"])) / PREFLIGHT_NAME


def _partition_path(root: Path, config: Mapping[str, Any]) -> Path:
    return _resolve(root, str(config["artifact_directory"])) / PARTITION_NAME


def _validate_preflight(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _validate_sealed(value, "formal receipt preflight")
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
        or not 0 <= value["cpu_utilization_percent"] <= 100
        or not isinstance(value["available_ram_mib"], int)
        or isinstance(value["available_ram_mib"], bool)
        or value["resource_gate"] != gate
        or value["admitted"] is not True
        or value["cpu_utilization_percent"] >= gate["cpu_utilization_below_percent"]
        or value["available_ram_mib"] < gate["minimum_available_ram_mib"]
        or value["gpu_or_cuda_probed"] is not False
        or value["processes_signaled"] is not False
    ):
        raise ValueError("formal receipt resource preflight contract mismatch")


def _build_preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = _preflight_path(root, config)
    if path.exists():
        value = _load_json(path)
        _validate_preflight(value, config)
        return value
    import psutil

    cpu = float(psutil.cpu_percent(interval=1.0))
    available_ram = int(psutil.virtual_memory().available // (1024 * 1024))
    gate = config["resource_gate"]
    admitted = (
        cpu < gate["cpu_utilization_below_percent"]
        and available_ram >= gate["minimum_available_ram_mib"]
    )
    body = {
        "schema_version": PREFLIGHT_SCHEMA,
        "sampled_at": datetime.now(UTC).isoformat(),
        "cpu_utilization_percent": cpu,
        "available_ram_mib": available_ram,
        "resource_gate": gate,
        "admitted": admitted,
        "gpu_or_cuda_probed": False,
        "processes_signaled": False,
    }
    value = _sealed(body)
    if not admitted:
        raise RuntimeError("formal receipt partition failed closed at resource admission")
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
    bindings = []
    if not directory.exists():
        return bindings
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        bindings.append({"path": path.relative_to(root).as_posix(), "file_sha256": _file_sha(path)})
    return bindings


def _validate_artifact_bindings(root: Path, bindings: Any) -> None:
    if not isinstance(bindings, list):
        raise TypeError("formal candidate artifact bindings are not a list")
    paths = []
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or _file_sha(_resolve(root, str(binding["path"]))) != binding["file_sha256"]
        ):
            raise ValueError("formal candidate artifact binding mismatch")
        paths.append(binding["path"])
    if paths != sorted(set(paths)):
        raise ValueError("formal candidate artifact binding ordering mismatch")


def _candidate_receipts(
    leaf: Mapping[str, Any], evidence: Mapping[str, Any], receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    queue_by_ordinal = {int(row["ordinal"]): row for row in leaf["formal_receipt_queue"]}
    rows = []
    for record in evidence["candidate_records"]:
        source = queue_by_ordinal[int(record["ordinal"])]
        prior_binding = source["formal_receipt_binding"]
        if source["state"] == "completed_preserved_formal_reject" and (
            not isinstance(prior_binding, Mapping)
            or prior_binding["decision"] != record["decision"]
            or prior_binding["first_blocker"] != record["first_blocker"]
        ):
            raise ValueError("preserved formal receipt reconciliation mismatch")
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "ordinal": record["ordinal"],
                "source_queue_entry_sha256": _sha(source),
                "source_queue_state": source["state"],
                "prior_formal_receipt_binding_sha256": (
                    _sha(prior_binding) if prior_binding is not None else None
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
    root: Path,
    config: Mapping[str, Any],
    leaf: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
    generated: Mapping[str, Any],
    formal: Mapping[str, Any],
    evidence: Mapping[str, Any],
    candidate_artifact_bindings: list[Mapping[str, Any]],
) -> dict[str, Any]:
    receipts = _candidate_receipts(leaf, evidence, formal)
    return {
        "schema_version": PARTITION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "pagination_result_binding": config["pagination_result"],
        "selected_leaf_page_binding": config["selected_leaf_page"],
        "source_leaf_content_sha256": leaf["content_sha256"],
        "source_leaf_formal_receipt_queue_root_sha256": leaf["formal_receipt_queue_root_sha256"],
        "source_leaf_all_survivor_ordinals_root_sha256": leaf["candidate_manifest"][
            "all_survivor_ordinals_root_sha256"
        ],
        "resource_preflight_binding": dict(preflight_binding),
        "generated_receipt": generated,
        "candidate_manifest": leaf["candidate_manifest"],
        "formal_receipt": formal,
        "formal_evidence": evidence,
        "candidate_artifact_bindings": candidate_artifact_bindings,
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
            "partition_pending_after_execution": 0,
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
    value: Mapping[str, Any], root: Path, config: Mapping[str, Any], leaf: Mapping[str, Any]
) -> None:
    _validate_sealed(value, "formal receipt partition")
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
            root,
            config,
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
        or manifest != leaf["candidate_manifest"]
        or generated["candidate_root_sha256"] != manifest["candidate_root_sha256"]
        or formal["candidate_root_sha256"] != generated["candidate_root_sha256"]
        or formal["generated_receipt_sha256"] != generated["content_sha256"]
        or evidence["candidate_manifest_sha256"] != manifest["content_sha256"]
        or evidence["generated_receipt_sha256"] != generated["content_sha256"]
        or evidence["candidate_records_root_sha256"] != _sha(evidence["candidate_records"])
        or value["counts"]["partition_candidates"] != len(leaf["formal_receipt_queue"])
        or value["counts"]["partition_pending_after_execution"] != 0
        or value["complete_partition_formal_receipts"] is not True
        or value["complete_comparable_evidence"] is not False
        or value["rank_or_promotion_requested"] is not False
        or any(value["seals"].values())
    ):
        raise ValueError("formal receipt partition contract mismatch")


def _publish_candidate_artifacts(attempt: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for candidate_dir in sorted(item for item in attempt.iterdir() if item.is_dir()):
        target = destination / candidate_dir.name
        if target.exists():
            existing = [
                (path.relative_to(target).as_posix(), _file_sha(path))
                for path in sorted(item for item in target.rglob("*") if item.is_file())
            ]
            proposed = [
                (path.relative_to(candidate_dir).as_posix(), _file_sha(path))
                for path in sorted(item for item in candidate_dir.rglob("*") if item.is_file())
            ]
            if existing != proposed:
                raise ValueError("immutable formal candidate artifact differs")
            continue
        os.replace(candidate_dir, target)


def _build_partition(
    root: Path,
    config: Mapping[str, Any],
    leaf: Mapping[str, Any],
    preflight: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    path = _partition_path(root, config)
    if path.exists():
        value = _load_json(path)
        _validate_partition(value, root, config, leaf)
        return value
    remaining = config["maximum_total_seconds"] - (time.monotonic() - started)
    if remaining <= 2:
        raise TimeoutError("formal receipt worker exceeded total wall-clock bound")
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
            action_name="Epoch 003 formal receipt partition 0001",
        )
        formal = formal_result["receipt"]
        evidence = formal_result["evidence"]
        _validate_formal_receipt(formal)
        validate_formal_evidence(evidence)
        artifact_root = directory / "candidate-artifacts"
        _publish_candidate_artifacts(attempt, artifact_root)
        bindings = _artifact_bindings(root, artifact_root)
        body = _derive_partition(
            root,
            config,
            leaf,
            _binding(root, _preflight_path(root, config), preflight),
            generated,
            formal,
            evidence,
            bindings,
        )
        value = _sealed(body)
        _validate_partition(value, root, config, leaf)
        _write_atomic_immutable(path, value, int(config["maximum_artifact_bytes"]))
        return value
    finally:
        shutil.rmtree(attempt, ignore_errors=True)


def _derive_result(
    root: Path,
    config: Mapping[str, Any],
    pagination: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    processed = partition["counts"]["newly_processed_candidates"]
    pending_before = pagination["counts"]["pending_formal_receipts"]
    leaf = _load_bound_json(root, config["selected_leaf_page"])
    hierarchy_path = _selected_leaf_hierarchy_path(
        root, pagination, config["selected_leaf_page"], leaf
    )
    counts = {
        "global_survivor_candidates": pagination["counts"]["source_survivors"],
        "global_pending_formal_receipts_before_partition": pending_before,
        "partition_candidates": partition["counts"]["partition_candidates"],
        "newly_processed_candidates": processed,
        "reconciled_preserved_candidates": partition["counts"]["reconciled_preserved_candidates"],
        "candidate_rejects": partition["counts"]["candidate_rejects"],
        "candidate_blocks": partition["counts"]["candidate_blocks"],
        "candidate_passes": 0,
        "formal_passes": 0,
        "global_pending_formal_receipts_after_partition": pending_before - processed,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "pagination_result_binding": config["pagination_result"],
        "selected_leaf_page_binding": config["selected_leaf_page"],
        "selected_leaf_hierarchy_path": hierarchy_path,
        "partition_result_binding": _binding(root, _partition_path(root, config), partition),
        "source_pagination_hierarchy_roots": {
            "ordered_batch_content_root_sha256": pagination["ordered_batch_content_root_sha256"],
            "ordered_complete_survivor_roots_sha256": pagination[
                "ordered_complete_survivor_roots_sha256"
            ],
            "formal_receipt_queue_hierarchy_root_sha256": pagination[
                "formal_receipt_queue_hierarchy_root_sha256"
            ],
        },
        "processed_partition_roots": {
            "source_leaf_content_sha256": partition["source_leaf_content_sha256"],
            "source_leaf_formal_receipt_queue_root_sha256": partition[
                "source_leaf_formal_receipt_queue_root_sha256"
            ],
            "candidate_formal_receipts_root_sha256": partition[
                "candidate_formal_receipts_root_sha256"
            ],
            "formal_evidence_content_sha256": partition["formal_evidence"]["content_sha256"],
        },
        "counts": counts,
        "complete_partition_formal_receipts": True,
        "complete_global_formal_receipts": False,
        "complete_comparable_evidence": False,
        "first_remaining_blocker": "11225_candidate_specific_formal_receipts_pending",
        "pending_count_semantics": "partition_overlay_no_updated_global_queue_root",
        "execution_contract": {
            "cpu_workers": 1,
            "gpu_workers": 0,
            "maximum_formal_seconds": config["maximum_formal_seconds"],
            "maximum_total_seconds": config["maximum_total_seconds"],
            "resume": "validate_and_reuse_immutable_preflight_and_partition_receipt",
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
        "seals": config["seals"],
    }


def validate_result(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value, "formal receipt worker result")
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    leaf = _load_bound_json(root, config["selected_leaf_page"])
    partition = _load_bound_json(root, value.get("partition_result_binding", {}))
    _validate_partition(partition, root, config, leaf)
    expected = _sealed(_derive_result(root, config, pagination, partition))
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["counts"]["global_survivor_candidates"] != 11_439
        or value["counts"]["global_pending_formal_receipts_before_partition"] != 11_247
        or value["counts"]["global_pending_formal_receipts_after_partition"] != 11_225
        or value["complete_partition_formal_receipts"] is not True
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("formal receipt worker result contract mismatch")


def build_result(root: Path, config_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    leaf = _load_bound_json(root, config["selected_leaf_page"])
    preflight = _build_preflight(root, config)
    partition = _build_partition(root, config, leaf, preflight, started)
    result = _sealed(_derive_result(root, config, pagination, partition))
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
