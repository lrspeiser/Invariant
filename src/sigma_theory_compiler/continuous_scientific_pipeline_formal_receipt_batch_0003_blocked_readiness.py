"""Immutable no-execution readiness evidence for non-admitted formal batch 0003."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from . import continuous_scientific_pipeline_formal_receipt_batch_0003 as target_worker
from . import continuous_scientific_pipeline_formal_receipt_batch_worker as engine

CONFIG_SCHEMA = "sigma-formal-receipt-batch-blocked-readiness-config-1.0"
READINESS_SCHEMA = "sigma-formal-receipt-batch-blocked-readiness-1.0"
DECISION = "blocked_readiness_resource_not_admitted_execution_not_started"
CONFIG_REL = (
    "configs/continuous_scientific_pipeline_epoch_003_formal_receipt_batch_0003_"
    "blocked_readiness.json"
)
SOURCE_REL = (
    "src/sigma_theory_compiler/continuous_scientific_pipeline_formal_receipt_batch_0003_"
    "blocked_readiness.py"
)
TEST_REL = (
    "tests/test_continuous_scientific_pipeline_formal_receipt_batch_0003_blocked_readiness.py"
)
READINESS_REL = (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003-"
    "blocked-readiness.json"
)


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = engine._sha(body)
    return body


def _validate_sealed(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping) or value.get("content_sha256") != engine._sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    ):
        raise ValueError(f"{label} content hash mismatch")


def _validate_file_binding(root: Path, value: Mapping[str, Any], expected_path: str) -> None:
    if set(value) != {"path", "file_sha256"} or value["path"] != expected_path:
        raise ValueError(f"file binding path mismatch: {expected_path}")
    if engine._file_sha(engine._resolve(root, expected_path)) != value["file_sha256"]:
        raise ValueError(f"file binding hash mismatch: {expected_path}")


def load_config(root: Path, path: Path) -> dict[str, Any]:
    """Load only immutable implementation/predecessor bindings and sealed thresholds."""
    config = engine._load_json(path)
    if set(config) != {
        "schema_version",
        "campaign_id",
        "target_batch_config",
        "target_batch_source",
        "target_batch_test",
        "predecessor_result",
        "resource_sampling_contract",
        "readiness_artifact",
        "seals",
    }:
        raise ValueError("batch 0003 blocked-readiness config keys mismatch")
    expected_sampling = {
        "source": "psutil_device_wide",
        "sample_count": 3,
        "sample_interval_seconds": 1,
        "cpu_utilization_strictly_below_percent": 92,
        "minimum_available_ram_mib": 32_768,
        "require_every_sample_admissible_to_start": True,
    }
    expected_seals = {
        "execution_started": False,
        "result_artifact_created": False,
        "leaf_artifacts_created": False,
        "cursor_artifact_created": False,
        "live_campaign_SQLite_access": False,
        "gpu_or_cuda_access": False,
        "runtime_access": False,
        "supervisor_access": False,
        "external_process_signals": False,
        "leaderboard_or_rank_writes": False,
    }
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"]
        != "continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003-blocked-readiness"
        or config["resource_sampling_contract"] != expected_sampling
        or config["readiness_artifact"] != READINESS_REL
        or config["seals"] != expected_seals
    ):
        raise ValueError("batch 0003 blocked-readiness config contract mismatch")
    for key, expected_path in (
        ("target_batch_config", target_worker.CONFIG_REL),
        ("target_batch_source", target_worker.SOURCE_REL),
        ("target_batch_test", target_worker.TEST_REL),
    ):
        _validate_file_binding(root, config[key], expected_path)
    target_config_path = engine._resolve(root, target_worker.CONFIG_REL)
    target_config = target_worker.load_config(root, target_config_path)
    if target_config["predecessor_cumulative_result"] != config["predecessor_result"]:
        raise ValueError("blocked-readiness predecessor differs from target batch")
    return config


def _parse_sampled_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("resource sample timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("resource sample timestamp is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("resource sample timestamp must be explicit UTC")
    return parsed


def _normalize_samples(
    samples: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes, bytearray)):
        raise TypeError("resource samples must be a sequence")
    if len(samples) != contract["sample_count"]:
        raise ValueError("resource sample count mismatch")
    timestamps = []
    for sequence, sample in enumerate(samples):
        if set(sample) != {
            "sequence",
            "sampled_at",
            "cpu_utilization_percent",
            "available_ram_mib",
        }:
            raise ValueError("resource sample keys mismatch")
        cpu = sample["cpu_utilization_percent"]
        ram = sample["available_ram_mib"]
        if (
            sample["sequence"] != sequence
            or not isinstance(cpu, (int, float))
            or isinstance(cpu, bool)
            or not 0 <= cpu <= 100
            or not isinstance(ram, int)
            or isinstance(ram, bool)
            or ram <= 0
        ):
            raise ValueError("resource sample value contract mismatch")
        timestamp = _parse_sampled_at(sample["sampled_at"])
        timestamps.append(timestamp)
        admitted = (
            cpu < contract["cpu_utilization_strictly_below_percent"]
            and ram >= contract["minimum_available_ram_mib"]
        )
        rows.append(
            {
                **dict(sample),
                "cpu_utilization_percent": float(cpu),
                "sample_admitted": admitted,
            }
        )
    if timestamps != sorted(set(timestamps)):
        raise ValueError("resource sample timestamps must be strictly increasing")
    if any(
        (right - left).total_seconds() < contract["sample_interval_seconds"]
        for left, right in pairwise(timestamps)
    ):
        raise ValueError("resource sample timestamps violate the minimum sample interval")
    if any(row["sample_admitted"] for row in rows):
        raise ValueError("blocked readiness requires every recorded sample to be non-admitted")
    return rows


def _artifact_absence_observations(target_config: Mapping[str, Any]) -> dict[str, Any]:
    directory = target_config["artifact_directory"]
    return {
        "artifact_directory": directory,
        "artifact_directory_existed": False,
        "result_path": target_worker.RESULT_REL,
        "result_existed": False,
        "cursor_path": f"{directory}/cumulative-cursor.json",
        "cursor_existed": False,
        "selected_leaf_paths": [f"{directory}/leaf-000012.json", f"{directory}/leaf-000013.json"],
        "selected_leaf_artifacts_existed": False,
        "candidate_artifacts_path": f"{directory}/candidate-artifacts",
        "candidate_artifacts_existed": False,
    }


def _assert_execution_namespace_absent(root: Path, target_config: Mapping[str, Any]) -> None:
    observation = _artifact_absence_observations(target_config)
    paths = [
        observation["artifact_directory"],
        observation["result_path"],
        observation["cursor_path"],
        *observation["selected_leaf_paths"],
        observation["candidate_artifacts_path"],
    ]
    if any(engine._resolve(root, item).exists() for item in paths):
        raise FileExistsError("batch 0003 execution artifact namespace is not empty")


def _derive_readiness(
    root: Path,
    config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = config["resource_sampling_contract"]
    rows = _normalize_samples(samples, contract)
    target_config = engine._load_json(engine._resolve(root, config["target_batch_config"]["path"]))
    cpu_blocked = any(
        row["cpu_utilization_percent"] >= contract["cpu_utilization_strictly_below_percent"]
        for row in rows
    )
    summary = {
        "maximum_cpu_utilization_percent": max(row["cpu_utilization_percent"] for row in rows),
        "minimum_available_ram_mib": min(row["available_ram_mib"] for row in rows),
        "admitted_sample_count": sum(row["sample_admitted"] for row in rows),
        "non_admitted_sample_count": sum(not row["sample_admitted"] for row in rows),
        "all_resource_samples_admissible": all(row["sample_admitted"] for row in rows),
        "first_blocker": (
            "cpu_utilization_not_strictly_below_92_percent"
            if cpu_blocked
            else "available_ram_below_32768_mib"
        ),
    }
    return {
        "schema_version": READINESS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "target_batch_bindings": {
            "config": config["target_batch_config"],
            "source": config["target_batch_source"],
            "test": config["target_batch_test"],
        },
        "predecessor_result_binding": config["predecessor_result"],
        "resource_sampling_contract": contract,
        "resource_samples": rows,
        "resource_summary": summary,
        "admitted": False,
        "execution_started": False,
        "execution_artifact_absence": _artifact_absence_observations(target_config),
        "result_artifact_created": False,
        "leaf_artifacts_created": False,
        "cursor_artifact_created": False,
        "complete_global_formal_receipts": False,
        "complete_comparable_evidence": False,
        "promotion_contract": {
            "formal_pass_claimed": False,
            "leaderboard_rebuild_requested": False,
            "rank_assignment_performed": False,
            "candidate_promotion_performed": False,
        },
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


def build_readiness(
    root: Path,
    config_path: Path,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a readiness-only record after proving the execution namespace is absent."""
    config = load_config(root, config_path)
    target_config = engine._load_json(engine._resolve(root, config["target_batch_config"]["path"]))
    _assert_execution_namespace_absent(root, target_config)
    value = _sealed(_derive_readiness(root, config, samples))
    validate_readiness(value, root, config_path)
    return value


def validate_readiness(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    """Replay the historical no-execution record without reading mutable runtime state."""
    _validate_sealed(value, "batch 0003 blocked readiness")
    if (
        value.get("decision") != DECISION
        or value.get("admitted") is not False
        or value.get("execution_started") is not False
        or value.get("result_artifact_created") is not False
        or value.get("leaf_artifacts_created") is not False
        or value.get("cursor_artifact_created") is not False
        or value.get("complete_global_formal_receipts") is not False
        or value.get("complete_comparable_evidence") is not False
        or not isinstance(value.get("promotion_contract"), Mapping)
        or any(value["promotion_contract"].values())
        or not isinstance(value.get("seals"), Mapping)
        or any(value["seals"].values())
    ):
        raise ValueError("batch 0003 blocked-readiness execution boundary changed")
    config = load_config(root, config_path)
    samples = value.get("resource_samples")
    if not isinstance(samples, list):
        raise TypeError("blocked-readiness resource_samples must be a list")
    original_samples = [
        {key: item for key, item in row.items() if key != "sample_admitted"} for row in samples
    ]
    expected = _sealed(_derive_readiness(root, config, original_samples))
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["admitted"] is not False
        or value["execution_started"] is not False
        or value["result_artifact_created"] is not False
        or value["leaf_artifacts_created"] is not False
        or value["cursor_artifact_created"] is not False
        or value["complete_global_formal_receipts"] is not False
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
    ):
        raise ValueError("batch 0003 blocked-readiness contract mismatch")


def replay_readiness(value: Mapping[str, Any], root: Path, config_path: Path) -> dict[str, Any]:
    """Return the independently replayed readiness record after exact validation."""
    validate_readiness(value, root, config_path)
    return dict(value)


def capture_resource_samples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Capture the configured device-wide samples without controlling any process."""
    import psutil

    contract = config["resource_sampling_contract"]
    rows = []
    for sequence in range(contract["sample_count"]):
        cpu = float(psutil.cpu_percent(interval=1.0))
        rows.append(
            {
                "sequence": sequence,
                "sampled_at": datetime.now(UTC).isoformat(),
                "cpu_utilization_percent": cpu,
                "available_ram_mib": int(psutil.virtual_memory().available // (1024 * 1024)),
            }
        )
        if sequence + 1 < contract["sample_count"]:
            time.sleep(max(0, contract["sample_interval_seconds"] - 1))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_REL)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    config_path = engine._resolve(root, arguments.config)
    config = load_config(root, config_path)
    samples = capture_resource_samples(config)
    readiness = build_readiness(root, config_path, samples)
    print(json.dumps(readiness, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
