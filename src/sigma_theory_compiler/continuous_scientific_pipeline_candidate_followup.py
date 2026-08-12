"""Bounded immutable candidate follow-up for Epoch 003 survivor batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .continuous_formula_formal_backend import (
    _validate_semantic_health,
    load_backend_config,
    validate_candidate_manifest,
    validate_formal_evidence,
)
from .continuous_scientific_pipeline_epoch import validate_epoch_genesis
from .continuous_scientific_pipeline_epoch_result import validate_epoch_result
from .continuous_scientific_pipeline_service import (
    _real_formal_worker,
    _run_owned_child,
    execute_real_generation,
)

CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-candidate-followup-config-1.0"
BATCH_SCHEMA = "sigma-continuous-scientific-pipeline-candidate-followup-batch-1.0"
RESULT_SCHEMA = "sigma-continuous-scientific-pipeline-candidate-followup-result-1.0"
DECISION = "candidate_specific_followup_blocked_no_comparable_evidence_no_promotion"
CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003_candidate_followup.json"
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_candidate_followup.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_candidate_followup.py"
RESULT_REL = "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/result.json"
MAXIMUM_ARTIFACT_BYTES = 4_194_304


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


def _load_bound_json(root: Path, binding: Mapping[str, Any], *, content: bool) -> dict[str, Any]:
    expected = {"path", "file_sha256", "content_sha256"} if content else {"path", "file_sha256"}
    if set(binding) != expected:
        raise ValueError("follow-up source binding contract mismatch")
    path = _resolve(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("follow-up source binding file hash mismatch")
    value = _load_json(path)
    if content and value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("follow-up source binding content hash mismatch")
    return value


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = _load_json(path)
    expected = {
        "schema_version",
        "campaign_id",
        "epoch_genesis",
        "epoch_terminal_result",
        "formal_backend_config",
        "artifact_directory",
        "pass_batch_indices",
        "maximum_candidate_records_per_batch",
        "maximum_generation_seconds_per_batch",
        "maximum_formal_seconds_per_batch",
        "maximum_total_seconds",
        "resource_contract",
    }
    resource = config.get("resource_contract", {})
    if (
        set(config) != expected
        or config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"] != "continuous-scientific-pipeline-epoch-003-candidate-followup"
        or config["pass_batch_indices"] != [1, 2, 3, 5, 6, 7]
        or config["maximum_candidate_records_per_batch"] != 32
        or config["maximum_generation_seconds_per_batch"] != 120
        or config["maximum_formal_seconds_per_batch"] != 120
        or config["maximum_total_seconds"] != 1440
        or resource
        != {
            "cpu_workers": 15,
            "gpu_workers": 0,
            "live_campaign_SQLite_access": False,
            "observations_opened": False,
            "external_process_signals": False,
            "leaderboard_or_rank_writes": False,
        }
    ):
        raise ValueError("follow-up config contract mismatch")
    _resolve(root, str(config["artifact_directory"]))
    _load_bound_json(root, config["epoch_genesis"], content=True)
    _load_bound_json(root, config["epoch_terminal_result"], content=True)
    _load_bound_json(root, config["formal_backend_config"], content=False)
    return config


def _artifact_bindings(root: Path, directory: Path) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(root).as_posix(), "file_sha256": _file_sha(path)}
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def _validate_artifact_bindings(root: Path, bindings: Any) -> None:
    if not isinstance(bindings, list):
        raise TypeError("candidate artifact bindings are not a list")
    paths = []
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or not isinstance(binding["path"], str)
            or not isinstance(binding["file_sha256"], str)
            or len(binding["file_sha256"]) != 64
            or _file_sha(_resolve(root, binding["path"])) != binding["file_sha256"]
        ):
            raise ValueError("candidate artifact binding mismatch")
        paths.append(binding["path"])
    if paths != sorted(set(paths)):
        raise ValueError("candidate artifact binding ordering mismatch")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if len(raw.encode()) > MAXIMUM_ARTIFACT_BYTES:
        raise RuntimeError("follow-up artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != raw:
            raise ValueError("immutable follow-up artifact differs")
        return
    path.write_text(raw, encoding="utf-8")


def _batch_result_path(root: Path, config: Mapping[str, Any], batch_index: int) -> Path:
    return _resolve(root, str(config["artifact_directory"])) / f"batch-{batch_index:02d}.json"


def _candidate_directory(root: Path, config: Mapping[str, Any], batch_index: int) -> Path:
    return (
        _resolve(root, str(config["artifact_directory"]))
        / f"batch-{batch_index:02d}-candidate-artifacts"
    )


def _validate_batch_result(
    value: Mapping[str, Any],
    root: Path,
    config: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> None:
    _validate_sealed(value, "follow-up batch")
    expected = {
        "schema_version",
        "batch_index",
        "ordinal_interval",
        "terminal_receipt_binding",
        "generated_receipt",
        "candidate_manifest",
        "formal_receipt",
        "formal_evidence",
        "candidate_artifact_bindings",
        "decision",
        "first_blocker",
        "complete_comparable_evidence",
        "rank_or_promotion_requested",
        "content_sha256",
    }
    index = value.get("batch_index")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index not in config["pass_batch_indices"]
    ):
        raise ValueError("follow-up batch index mismatch")
    source = terminal["completed_receipt_bindings"][index]
    manifest = value.get("candidate_manifest", {})
    evidence = value.get("formal_evidence", {})
    receipt = value.get("formal_receipt", {})
    validate_candidate_manifest(manifest)
    validate_formal_evidence(evidence)
    _validate_artifact_bindings(root, value.get("candidate_artifact_bindings"))
    for record in evidence["candidate_records"]:
        health = record["semantic_action_health"]
        if health is not None:
            candidate_dir = _candidate_directory(root, config, index) / record["candidate_id"]
            _validate_semantic_health(health, candidate_dir=candidate_dir)
    if (
        set(value) != expected
        or value["schema_version"] != BATCH_SCHEMA
        or value["ordinal_interval"] != source["ordinal_interval"]
        or value["terminal_receipt_binding"] != source
        or value["generated_receipt"] != source["generated_receipt"]
        or manifest["content_sha256"] != source["candidate_manifest_binding"]["content_sha256"]
        or manifest["candidate_root_sha256"]
        != source["candidate_manifest_binding"]["candidate_root_sha256"]
        or receipt != source["formal_receipt"]
        or evidence["content_sha256"] != source["formal_evidence_binding"]["content_sha256"]
        or evidence["candidate_manifest_sha256"] != manifest["content_sha256"]
        or evidence["generated_receipt_sha256"] != value["generated_receipt"]["content_sha256"]
        or value["decision"] != evidence["decision"]
        or value["first_blocker"] != evidence["first_blocker"]
        or value["complete_comparable_evidence"] is not False
        or value["rank_or_promotion_requested"] is not False
    ):
        raise ValueError("follow-up batch contract mismatch")


def _build_batch(
    root: Path,
    config: Mapping[str, Any],
    genesis: Mapping[str, Any],
    terminal: Mapping[str, Any],
    batch_index: int,
) -> dict[str, Any]:
    existing_path = _batch_result_path(root, config, batch_index)
    if existing_path.exists():
        existing = _load_json(existing_path)
        _validate_batch_result(existing, root, config, terminal)
        return existing
    service_config = dict(genesis["derived_service_config"])
    service_config["maximum_action_seconds"] = config["maximum_generation_seconds_per_batch"]
    interval = terminal["completed_receipt_bindings"][batch_index]["ordinal_interval"]
    generated = execute_real_generation({"next_ordinal": interval["start_ordinal"]}, service_config)
    source = terminal["completed_receipt_bindings"][batch_index]
    if (
        generated["receipt"] != source["generated_receipt"]
        or generated["manifest"]["content_sha256"]
        != source["candidate_manifest_binding"]["content_sha256"]
    ):
        raise ValueError("follow-up generation replay differs from terminal binding")
    candidate_dir = _candidate_directory(root, config, batch_index)
    formal = _run_owned_child(
        _real_formal_worker,
        (
            str(root),
            str(candidate_dir),
            generated["receipt"],
            generated["manifest"],
            service_config,
        ),
        maximum_seconds=float(config["maximum_formal_seconds_per_batch"]),
        action_name=f"epoch 003 candidate follow-up batch {batch_index}",
    )
    body = {
        "schema_version": BATCH_SCHEMA,
        "batch_index": batch_index,
        "ordinal_interval": interval,
        "terminal_receipt_binding": source,
        "generated_receipt": generated["receipt"],
        "candidate_manifest": generated["manifest"],
        "formal_receipt": formal["receipt"],
        "formal_evidence": formal["evidence"],
        "candidate_artifact_bindings": _artifact_bindings(root, candidate_dir),
        "decision": formal["evidence"]["decision"],
        "first_blocker": formal["evidence"]["first_blocker"],
        "complete_comparable_evidence": False,
        "rank_or_promotion_requested": False,
    }
    result = _sealed(body)
    _validate_batch_result(result, root, config, terminal)
    _write_immutable(existing_path, result)
    return result


def _batch_binding(root: Path, path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": _file_sha(path),
        "content_sha256": value["content_sha256"],
    }


def _summary_records(batches: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "batch_index": batch["batch_index"],
            "candidate_id": record["candidate_id"],
            "ordinal": record["ordinal"],
            "covariant_mapping_decision": record["covariant_mapping_decision"],
            "covariant_mapping_payload_sha256": record["covariant_mapping_payload_sha256"],
            "semantic_action_health_sha256": record["semantic_action_health_sha256"],
            "decision": record["decision"],
            "first_blocker": record["first_blocker"],
        }
        for batch in batches
        for record in batch["formal_evidence"]["candidate_records"]
    ]


def _derive_result(
    root: Path,
    config: Mapping[str, Any],
    terminal: Mapping[str, Any],
    batches: list[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries = _summary_records(batches)
    manifests = [batch["candidate_manifest"] for batch in batches]
    evidence = [batch["formal_evidence"] for batch in batches]
    counts = {
        "source_pass_batches": len(batches),
        "source_survivor_candidates": sum(item["survivor_record_count"] for item in manifests),
        "durably_recorded_candidates": len(summaries),
        "sample_complete_batches": sum(item["sample_complete"] for item in manifests),
        "symbolic_local_preflight_passes": sum(
            item["symbolic_local_preflight_pass_count"] for item in evidence
        ),
        "covariant_action_mapped_candidates": sum(
            item["covariant_action_mapped_count"] for item in evidence
        ),
        "action_health_executions": sum(item["action_health_execution_count"] for item in evidence),
        "candidate_blocks": sum(row["decision"] == "block" for row in summaries),
        "candidate_rejects": sum(row["decision"] == "reject" for row in summaries),
        "candidate_passes": 0,
        "formal_passes": 0,
        "leaderboard_rebuild_requests": 0,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    first_blocker = (
        "candidate_manifest_is_bounded_not_complete"
        if counts["sample_complete_batches"] != counts["source_pass_batches"]
        else "complete_comparable_candidate_evidence_not_registered"
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "decision": DECISION,
        "campaign_id": config["campaign_id"],
        "epoch_terminal_result_binding": config["epoch_terminal_result"],
        "reviewed_backend_binding": config["formal_backend_config"],
        "source_pass_batch_indices": config["pass_batch_indices"],
        "batch_artifact_bindings": [
            _batch_binding(root, _batch_result_path(root, config, batch["batch_index"]), batch)
            for batch in batches
        ],
        "candidate_decision_records": summaries,
        "candidate_decision_records_root_sha256": _sha(summaries),
        "counts": counts,
        "first_remaining_blocker": first_blocker,
        "complete_comparable_evidence": False,
        "execution_contract": {
            "cpu_workers_per_generation_batch": 15,
            "gpu_workers": 0,
            "maximum_generation_seconds_per_batch": config["maximum_generation_seconds_per_batch"],
            "maximum_formal_seconds_per_batch": config["maximum_formal_seconds_per_batch"],
            "maximum_total_seconds": config["maximum_total_seconds"],
            "generation_deadline": "reviewed_15_owned_children_hard_wall_clock_bound",
            "formal_deadline": "reviewed_single_owned_child_cleanup_inclusive_hard_bound",
            "resume": "validate_and_reuse_each_immutable_completed_batch_artifact",
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
            "dark_matter_or_halo_inputs": False,
            "redshift_or_cosmology_inputs": False,
            "paid_llm_calls": False,
            "external_process_signals": False,
            "direct_rank_assignment": False,
        },
    }


def validate_result(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value, "candidate follow-up result")
    config = load_config(root, config_path)
    terminal = _load_bound_json(root, config["epoch_terminal_result"], content=True)
    validate_epoch_result(terminal, root)
    bindings = value.get("batch_artifact_bindings", [])
    if not isinstance(bindings, list) or len(bindings) != len(config["pass_batch_indices"]):
        raise ValueError("follow-up batch artifact binding count mismatch")
    batches = []
    for index, binding in zip(config["pass_batch_indices"], bindings, strict=True):
        batch = _load_bound_json(root, binding, content=True)
        _validate_batch_result(batch, root, config, terminal)
        if batch["batch_index"] != index:
            raise ValueError("follow-up batch artifact order mismatch")
        batches.append(batch)
    expected = _sealed(_derive_result(root, config, terminal, batches))
    if (
        dict(value) != expected
        or value["decision"] != DECISION
        or value["complete_comparable_evidence"] is not False
        or any(value["promotion_contract"].values())
        or any(value["seals"].values())
        or value["counts"]["candidate_passes"] != 0
        or value["counts"]["formal_passes"] != 0
    ):
        raise ValueError("candidate follow-up result contract mismatch")


def build_result(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(root, config_path)
    genesis = _load_bound_json(root, config["epoch_genesis"], content=True)
    validate_epoch_genesis(
        genesis, root, root / "configs/continuous_scientific_pipeline_epoch_003.json"
    )
    terminal = _load_bound_json(root, config["epoch_terminal_result"], content=True)
    validate_epoch_result(terminal, root)
    load_backend_config(root, root / str(config["formal_backend_config"]["path"]))
    started = time.monotonic()
    batches = []
    for index in config["pass_batch_indices"]:
        if time.monotonic() - started >= config["maximum_total_seconds"]:
            raise TimeoutError("candidate follow-up exceeded total wall-clock bound")
        batches.append(_build_batch(root, config, genesis, terminal, index))
    result = _sealed(_derive_result(root, config, terminal, batches))
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
    output = _resolve(root, arguments.output)
    result = build_result(root, config_path)
    _write_immutable(output, result)
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
