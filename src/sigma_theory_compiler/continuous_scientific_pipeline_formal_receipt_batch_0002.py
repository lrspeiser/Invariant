"""Advance the sealed Epoch 003 formal-receipt cursor beyond prefix nine."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import continuous_scientific_pipeline_formal_receipt_batch_worker as engine

CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_formal_receipt_batch_0002.json"
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_formal_receipt_batch_0002.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_formal_receipt_batch_0002.py"
RESULT_REL = (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0002/result.json"
)

# Keep the reusable cursor engine public to focused replay and adversarial tests.
load_config = engine.load_config
_leaf_catalog = engine._leaf_catalog
_load_bound_json = engine._load_bound_json
_predecessor_state = engine._predecessor_state
_sealed = engine._sealed
_select_entries = engine._select_entries


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
            engine._binding(
                root,
                engine._artifact_path(root, config, f"leaf-{sequence:06d}.json"),
                partition,
            )
        )
    return {
        "schema_version": engine.RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": engine.DECISION,
        "pagination_result_binding": config["pagination_result"],
        "predecessor_cumulative_result_binding": config["predecessor_cumulative_result"],
        "executed_leaf_bindings": executed,
        "cumulative_ledger_binding": engine._binding(
            root, engine._artifact_path(root, config, "cumulative-cursor.json"), cursor
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
            label: {"path": relative, "file_sha256": engine._file_sha(root / relative)}
            for label, relative in (
                ("config", CONFIG_REL),
                ("source", SOURCE_REL),
                ("test", TEST_REL),
            )
        },
        "seals": config["seals"],
    }


def validate_result(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    """Replay the successor exclusively from sealed predecessor/catalog artifacts."""
    engine._validate_sealed(value, "formal receipt batch 0002 result")
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(root, pagination)
    predecessor_ledger, summaries, prior_partitions = _predecessor_state(root, predecessor, catalog)
    entries = _select_entries(catalog, summaries, config)
    leaves = [_load_bound_json(root, entry["leaf_binding"]) for entry in entries]
    bindings = value.get("executed_leaf_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(entries):
        raise ValueError("formal receipt batch 0002 executed leaf binding count mismatch")
    partitions = [_load_bound_json(root, binding) for binding in bindings]
    for partition, entry, leaf in zip(partitions, entries, leaves, strict=True):
        engine._validate_leaf_partition(partition, root, config, entry, leaf)
    cursor = _load_bound_json(root, value.get("cumulative_ledger_binding", {}))
    engine._validate_cursor(
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
        or value["decision"] != engine.DECISION
        or value["batch_leaf_catalog_indices"]
        != list(range(len(summaries), len(summaries) + len(entries)))
        or value["complete_processed_partition_prefix"] is not True
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("formal receipt batch 0002 result contract mismatch")


def build_result(root: Path, config_path: Path) -> dict[str, Any]:
    """Resume atomically, executing at most one bounded campaign-owned child."""
    started = time.monotonic()
    config = load_config(root, config_path)
    pagination = _load_bound_json(root, config["pagination_result"])
    predecessor = _load_bound_json(root, config["predecessor_cumulative_result"])
    catalog = _leaf_catalog(root, pagination)
    predecessor_ledger, summaries, prior_partitions = _predecessor_state(root, predecessor, catalog)
    entries = _select_entries(catalog, summaries, config)
    leaves = [_load_bound_json(root, entry["leaf_binding"]) for entry in entries]
    preflight = engine._build_preflight(root, config)
    partitions = engine._build_leaf_partitions(root, config, entries, leaves, preflight, started)
    cursor_path = engine._artifact_path(root, config, "cumulative-cursor.json")
    if cursor_path.exists():
        cursor = engine._load_json(cursor_path)
    else:
        cursor = _sealed(
            engine._derive_cursor(
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
        engine._write_atomic_immutable(cursor_path, cursor, int(config["maximum_artifact_bytes"]))
    engine._validate_cursor(
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
    config_path = engine._resolve(root, arguments.config)
    result = build_result(root, config_path)
    output = engine._resolve(root, arguments.output)
    engine._write_atomic_immutable(
        output, result, int(load_config(root, config_path)["maximum_artifact_bytes"])
    )
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
