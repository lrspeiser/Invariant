"""Immutable terminal archive for a completed continuous-pipeline epoch.

Construction reads a mutable epoch runtime exactly once. Validation deliberately
reconstructs every deterministic binding from the embedded archive and immutable
genesis, and never opens the mutable runtime directory.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .continuous_scientific_pipeline_epoch import (
    ARTIFACT_REL as GENESIS_REL,
)
from .continuous_scientific_pipeline_epoch import (
    CONFIG_REL as EPOCH_CONFIG_REL,
)
from .continuous_scientific_pipeline_epoch import validate_epoch_genesis
from .continuous_scientific_pipeline_service import (
    _atomic_json,
    _completed_replay_records,
    _validate_checkpoint,
    _validate_queue,
)
from .continuous_scientific_pipeline_service import (
    _sha as service_sha,
)

SCHEMA_VERSION = "sigma-continuous-scientific-pipeline-epoch-result-1.0"
DECISION = "bounded_epoch_complete_fail_closed_no_promotion"
PREFLIGHT_SCHEMA = "sigma-continuous-scientific-pipeline-epoch-preflight-1.0"
PREFLIGHT_REL = "runs/engine/continuous-scientific-pipeline-epoch-003-preflight.json"
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_epoch_result.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_epoch_result.py"
RESULT_REL = "runs/engine/continuous-scientific-pipeline-epoch-003-result.json"
MAXIMUM_RESULT_BYTES = 4_194_304


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


def _load_json(root: Path, relative: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{relative} is not a JSON object")
    return value


def _content_binding(root: Path, relative: str) -> dict[str, str]:
    value = _load_json(root, relative)
    _validate_sealed(value, relative)
    return {
        "path": relative,
        "file_sha256": _file_sha(root / relative),
        "content_sha256": value["content_sha256"],
    }


def _validate_preflight(value: Mapping[str, Any]) -> None:
    _validate_sealed(value, "epoch preflight")
    expected_keys = {
        "schema_version",
        "decision",
        "sampling_contract",
        "resource_contract",
        "samples",
        "summary",
        "historical_path_state",
        "ownership_observations",
        "validator_preflight",
        "content_sha256",
    }
    samples = value.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("epoch preflight samples are empty")
    maximum_cpu = max(row["cpu_percent"] for row in samples)
    minimum_ram = min(row["available_ram_mib"] for row in samples)
    resources = value.get("resource_contract", {})
    sampling = value.get("sampling_contract", {})
    if (
        set(value) != expected_keys
        or value.get("schema_version") != PREFLIGHT_SCHEMA
        or value.get("decision") != "exact_epoch_preflight_passed_before_start"
        or sampling.get("sample_count") != len(samples)
        or sampling.get("captured_immediately_before_service_start") is not True
        or sampling.get("historical_measurement_not_current_runtime_claim") is not True
        or resources.get("cpu_workers") != 15
        or resources.get("gpu_workers") != 0
        or maximum_cpu >= resources.get("cpu_backoff_at_or_above_percent", 0)
        or minimum_ram < resources.get("minimum_available_ram_mib", 0)
        or value.get("summary")
        != {
            "maximum_cpu_percent": maximum_cpu,
            "minimum_available_ram_mib": minimum_ram,
            "all_resource_samples_admissible": True,
        }
        or any(value.get("historical_path_state", {}).values())
        or value.get("ownership_observations", {}).get("epoch_service_owner_active") is not False
        or not all(value.get("validator_preflight", {}).values())
    ):
        raise ValueError("epoch preflight contract mismatch")


def _replay_dependencies(
    root: Path, genesis: Mapping[str, Any], service_config: Mapping[str, Any]
) -> dict[str, Any]:
    backend = _load_json(root, str(service_config["formal_backend_config_path"]))
    relatives = {
        "epoch_genesis": GENESIS_REL,
        "epoch_config": EPOCH_CONFIG_REL,
        "epoch_source": "src/sigma_theory_compiler/continuous_scientific_pipeline_epoch.py",
        "epoch_test": "tests/test_continuous_scientific_pipeline_epoch.py",
        "base_service_config": genesis["bindings"]["base_service_config"]["path"],
        "admission_config": str(service_config["admission_config_path"]),
        "service_source": "src/sigma_theory_compiler/continuous_scientific_pipeline_service.py",
        "service_test": "tests/test_continuous_scientific_pipeline_service.py",
        "generator_config": str(service_config["generator_config_path"]),
        "formal_backend_config": str(service_config["formal_backend_config_path"]),
        "formal_backend_source": ("src/sigma_theory_compiler/continuous_formula_formal_backend.py"),
        "formal_backend_test": "tests/test_continuous_formula_formal_backend.py",
        "grammar": str(backend["grammar_path"]),
        "field_contract": str(backend["field_contract_path"]),
        "formal_controls": str(backend["formal_controls_path"]),
        "candidate_mapper_source": str(backend["candidate_mapper_source_path"]),
        "action_health_source": str(backend["action_health_source_path"]),
        "real_formula_execution_source": ("src/sigma_theory_compiler/real_formula_execution.py"),
        "high_throughput_source": "src/sigma_theory_compiler/high_throughput.py",
        "gpu_screen_source": "src/sigma_theory_compiler/gpu_screen.py",
    }
    files = {
        label: {"path": relative, "file_sha256": _file_sha(root / relative)}
        for label, relative in relatives.items()
    }
    body = {"files": files}
    return {
        "replay_method": ("deterministic_ordinal_generation_then_candidate_bound_formal_backend"),
        "files": files,
        "replay_dependency_root_sha256": _sha(body),
    }


def _outcomes(completed: list[Mapping[str, Any]], queue: Mapping[str, Any]) -> dict[str, int]:
    return {
        "sampled_static_reject_batches": sum(
            row["screen_decision"] == "reject" for row in completed
        ),
        "sampled_static_pass_batches": sum(row["screen_decision"] == "pass" for row in completed),
        "formal_receipts": sum(row["formal_receipt_sha256"] is not None for row in completed),
        "formal_blocks": sum(row["formal_decision"] == "block" for row in completed),
        "formal_passes": sum(row["formal_decision"] == "pass" for row in completed),
        "leaderboard_rebuild_requests": len(queue["leaderboard_rebuild_requests"]),
        "rank_assignments": 0,
    }


def _interpretation(outcomes: Mapping[str, int]) -> str:
    return (
        f"{outcomes['sampled_static_reject_batches']} batches failed the sampled-static screen; "
        f"{outcomes['sampled_static_pass_batches']} batches produced bounded survivor manifests, "
        f"and {outcomes['formal_blocks']} were formally blocked. Epoch execution is complete, "
        "but no formal pass, theory verdict, leaderboard request, rank assignment, candidate "
        "promotion, GPU use, live campaign SQLite access, or observational claim follows."
    )


def _bindings(root: Path) -> dict[str, Any]:
    return {
        "result_source": {"path": SOURCE_REL, "file_sha256": _file_sha(root / SOURCE_REL)},
        "result_test": {"path": TEST_REL, "file_sha256": _file_sha(root / TEST_REL)},
        "genesis": _content_binding(root, GENESIS_REL),
        "preflight": _content_binding(root, PREFLIGHT_REL),
    }


def _derive(
    root: Path,
    genesis: Mapping[str, Any],
    preflight: Mapping[str, Any],
    queue: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    service_config = genesis["derived_service_config"]
    completed = queue["completed_action_receipts"]
    replay_dependencies = _replay_dependencies(root, genesis, service_config)
    replay_records = _completed_replay_records(completed, service_config, replay_dependencies)
    intervals = [row["ordinal_interval"] for row in replay_records]
    outcomes = _outcomes(completed, queue)
    coverage = {
        "start_ordinal": service_config["start_ordinal"],
        "stop_ordinal_exclusive": service_config["stop_ordinal_exclusive"],
        "unique_formula_count": (
            service_config["stop_ordinal_exclusive"] - service_config["start_ordinal"]
        ),
        "real_CPU_batches": len(completed),
        "workers_per_batch": service_config["cpu_workers"],
        "formulas_per_worker": service_config["batch_candidates_per_worker"],
        "intervals": intervals,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "epoch_id": genesis["epoch_id"],
        "genesis_binding": _content_binding(root, GENESIS_REL),
        "preflight_resource_evidence": dict(preflight),
        "coverage": coverage,
        "outcomes": outcomes,
        "completed_receipt_bindings": replay_records,
        "replay_dependencies": replay_dependencies,
        "terminal_runtime_archive": {"queue": dict(queue), "checkpoint": dict(checkpoint)},
        "runtime_binding": {
            "queue_content_sha256": queue["content_sha256"],
            "checkpoint_content_sha256": checkpoint["content_sha256"],
            "completed_receipts_sha256": _sha(completed),
            "completed_replay_records_sha256": _sha(replay_records),
            "terminal_state": checkpoint["state"],
            "cycles": checkpoint["cycles"],
        },
        "promotion_contract": {
            "formal_pass_claimed": outcomes["formal_passes"] > 0,
            "leaderboard_rebuild_requested": outcomes["leaderboard_rebuild_requests"] > 0,
            "rank_assignment_performed": False,
            "candidate_promotion_performed": False,
        },
        "interpretation": _interpretation(outcomes),
        "bindings": _bindings(root),
        "seals": genesis["seals"],
    }


def build_epoch_result(root: Path, runtime: Path | None = None) -> dict[str, Any]:
    """Archive one terminal runtime; this is the only mutable-runtime read path."""
    genesis = _load_json(root, GENESIS_REL)
    validate_epoch_genesis(genesis, root, root / EPOCH_CONFIG_REL)
    preflight = _load_json(root, PREFLIGHT_REL)
    _validate_preflight(preflight)
    service_config = genesis["derived_service_config"]
    runtime_path = runtime or (root / str(service_config["runtime_directory"]))
    runtime_path = runtime_path.resolve()
    runtime_path.relative_to(root.resolve())
    queue = json.loads((runtime_path / str(service_config["queue_name"])).read_text())
    checkpoint = json.loads((runtime_path / str(service_config["checkpoint_name"])).read_text())
    _validate_queue(queue)
    _validate_checkpoint(checkpoint, queue)
    body = _derive(root, genesis, preflight, queue, checkpoint)
    validate_epoch_result(_sealed(body), root)
    return _sealed(body)


def validate_epoch_result(value: Mapping[str, Any], root: Path) -> None:
    """Validate exclusively from immutable sources and the embedded terminal archive."""
    _validate_sealed(value, "epoch result")
    genesis = _load_json(root, GENESIS_REL)
    validate_epoch_genesis(genesis, root, root / EPOCH_CONFIG_REL)
    preflight = _load_json(root, PREFLIGHT_REL)
    _validate_preflight(preflight)
    terminal = value.get("terminal_runtime_archive", {})
    queue = terminal.get("queue", {})
    checkpoint = terminal.get("checkpoint", {})
    try:
        _validate_queue(queue)
        _validate_checkpoint(checkpoint, queue)
        expected = _derive(root, genesis, preflight, queue, checkpoint)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("epoch result terminal archive mismatch") from error
    service_config = genesis["derived_service_config"]
    completed = queue["completed_action_receipts"]
    intervals = expected["coverage"]["intervals"]
    interval_contract = (
        len(intervals) == genesis["coverage"]["batch_count"]
        and intervals[0]["start_ordinal"] == service_config["start_ordinal"]
        and intervals[-1]["stop_ordinal_exclusive"] == service_config["stop_ordinal_exclusive"]
        and all(
            left["stop_ordinal_exclusive"] == right["start_ordinal"]
            for left, right in itertools.pairwise(intervals)
        )
        and sum(row["unique_formula_count"] for row in intervals)
        == expected["coverage"]["unique_formula_count"]
    )
    if (
        dict(value) != _sealed(expected)
        or checkpoint["state"] != "bounded_complete"
        or queue["next_ordinal"] != service_config["stop_ordinal_exclusive"]
        or queue["stop_ordinal_exclusive"] != service_config["stop_ordinal_exclusive"]
        or queue["service_config_sha256"] != service_sha(service_config)
        or len(completed) != genesis["coverage"]["batch_count"]
        or queue["generated_receipt"] is not None
        or queue["generation_manifest"] is not None
        or queue["formal_receipt"] is not None
        or queue["formal_evidence"] is not None
        or queue["leaderboard_rebuild_requests"] != []
        or queue["last_ranked_candidate_root"] is not None
        or expected["outcomes"]["formal_passes"] != 0
        or expected["outcomes"]["rank_assignments"] != 0
        or any(expected["promotion_contract"].values())
        or any(expected["seals"].values())
        or not interval_contract
    ):
        raise ValueError("epoch result contract mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default=RESULT_REL)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    output = (root / arguments.output).resolve()
    output.relative_to(root)
    result = build_epoch_result(root)
    _atomic_json(output, result, MAXIMUM_RESULT_BYTES)
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
