from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .continuous_scientific_pipeline_service import (
    CHECKPOINT_SCHEMA,
    _atomic_json,
    _validate_checkpoint,
    _validate_queue,
    execute_real_formal,
    execute_real_generation,
    initial_queue,
    load_service_config,
    run_bounded_service,
    validate_execution_result,
)
from .continuous_scientific_pipeline_service import (
    _sealed as service_sealed,
)
from .continuous_scientific_pipeline_service import (
    _sha as service_sha,
)

CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-epoch-config-1.0"
ARTIFACT_SCHEMA = "sigma-continuous-scientific-pipeline-epoch-genesis-1.0"
SERVICE_CONFIG_SCHEMA = "sigma-continuous-scientific-pipeline-epoch-service-config-1.0"
CONFIG_REL = "configs/continuous_scientific_pipeline_epoch_003.json"
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_epoch.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_epoch.py"
ARTIFACT_REL = "runs/engine/continuous-scientific-pipeline-epoch-003-genesis.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def _validate_sealed(value: Mapping[str, Any]) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256", None)
    if claimed != _sha(body):
        raise ValueError("epoch artifact content hash mismatch")


def _load_bound_json(root: Path, binding: Mapping[str, Any], *, content: bool) -> dict[str, Any]:
    if set(binding) != (
        {"path", "file_sha256", "content_sha256"} if content else {"path", "file_sha256"}
    ):
        raise ValueError("epoch binding contract mismatch")
    path = (root / str(binding["path"])).resolve()
    path.relative_to(root.resolve())
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["file_sha256"]:
        raise ValueError("epoch binding file hash mismatch")
    value = json.loads(raw)
    if content and value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("epoch binding content hash mismatch")
    return value


def load_epoch_config(root: Path, path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "epoch_id",
        "predecessor_result",
        "base_service_config",
        "runtime_directory",
        "batch_count",
        "maximum_cycles_per_invocation",
        "maximum_service_seconds_per_invocation",
        "maximum_action_seconds",
        "seals",
    }
    if set(config) != expected or config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("epoch config contract mismatch")
    if (
        config["epoch_id"] != "continuous-scientific-pipeline-epoch-003"
        or config["batch_count"] != 8
        or not 0 < config["maximum_cycles_per_invocation"] <= 64
        or not 0 < config["maximum_service_seconds_per_invocation"] <= 600
        or not 0 < config["maximum_action_seconds"] <= 120
        or any(config["seals"].values())
    ):
        raise ValueError("epoch bound or seal mismatch")
    runtime = (root / config["runtime_directory"]).resolve()
    runtime.relative_to(root.resolve())
    _load_bound_json(root, config["base_service_config"], content=False)
    _load_bound_json(root, config["predecessor_result"], content=True)
    return config


def derive_service_config(root: Path, epoch_config: Mapping[str, Any]) -> dict[str, Any]:
    base_path = root / str(epoch_config["base_service_config"]["path"])
    base = load_service_config(root, base_path)
    predecessor = _load_bound_json(root, epoch_config["predecessor_result"], content=True)
    validate_execution_result(predecessor, root, base_path)
    coverage = predecessor["coverage"]
    outcomes = predecessor["outcomes"]
    if (
        predecessor["runtime_binding"]["terminal_state"] != "bounded_complete"
        or outcomes["formal_passes"] != 0
        or outcomes["leaderboard_rebuild_requests"] != 0
        or outcomes["rank_assignments"] != 0
    ):
        raise ValueError("epoch predecessor is not a completed fail-closed interval")
    batch_size = int(base["cpu_workers"]) * int(base["batch_candidates_per_worker"])
    start = int(coverage["stop_ordinal_exclusive"])
    stop = start + int(epoch_config["batch_count"]) * batch_size
    derived = {
        **base,
        "service_id": str(epoch_config["epoch_id"]),
        "runtime_directory": str(epoch_config["runtime_directory"]),
        "maximum_cycles": int(epoch_config["maximum_cycles_per_invocation"]),
        "maximum_service_seconds": int(epoch_config["maximum_service_seconds_per_invocation"]),
        "maximum_action_seconds": int(epoch_config["maximum_action_seconds"]),
        "start_ordinal": start,
        "stop_ordinal_exclusive": stop,
    }
    if start != int(coverage["start_ordinal"]) + int(coverage["unique_formula_count"]):
        raise ValueError("epoch predecessor coverage is not contiguous")
    return derived


def _intervals(service_config: Mapping[str, Any]) -> list[dict[str, int]]:
    size = int(service_config["cpu_workers"]) * int(service_config["batch_candidates_per_worker"])
    start = int(service_config["start_ordinal"])
    stop = int(service_config["stop_ordinal_exclusive"])
    if (stop - start) % size:
        raise ValueError("epoch interval is not an exact whole-batch multiple")
    return [
        {
            "batch_index": index,
            "start_ordinal": ordinal,
            "stop_ordinal_exclusive": ordinal + size,
            "unique_formula_count": size,
        }
        for index, ordinal in enumerate(range(start, stop, size))
    ]


def _service_config_document(service_config: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": SERVICE_CONFIG_SCHEMA,
            "service_config": dict(service_config),
            "service_config_sha256": service_sha(service_config),
        }
    )


def build_epoch_genesis(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_epoch_config(root, config_path)
    service_config = derive_service_config(root, config)
    queue = initial_queue(service_config)
    intervals = _intervals(service_config)
    predecessor = config["predecessor_result"]
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "decision": "disjoint_epoch_genesis_ready_for_persistent_resume_not_executed",
        "epoch_id": config["epoch_id"],
        "predecessor": {
            **predecessor,
            "stop_ordinal_exclusive": service_config["start_ordinal"],
        },
        "coverage": {
            "start_ordinal": service_config["start_ordinal"],
            "stop_ordinal_exclusive": service_config["stop_ordinal_exclusive"],
            "unique_formula_count": (
                service_config["stop_ordinal_exclusive"] - service_config["start_ordinal"]
            ),
            "batch_count": len(intervals),
            "workers_per_batch": service_config["cpu_workers"],
            "formulas_per_worker": service_config["batch_candidates_per_worker"],
            "intervals": intervals,
        },
        "derived_service_config": service_config,
        "initial_queue": queue,
        "resume_contract": {
            "runtime_directory": service_config["runtime_directory"],
            "immutable_service_config_materialized_once": True,
            "atomic_queue_and_checkpoint_files": True,
            "incomplete_budget_exit_normalized_to_bounded_pause": True,
            "resume_requires_exact_queue_checkpoint_root": True,
            "single_owner_lease_required": True,
            "hard_generation_and_formal_child_deadlines": True,
        },
        "execution_state": {
            "runtime_materialized": False,
            "formulas_evaluated": 0,
            "formal_receipts": 0,
            "epoch_complete": False,
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
                ("epoch_config", CONFIG_REL),
                ("epoch_source", SOURCE_REL),
                ("epoch_test", TEST_REL),
                ("base_service_config", str(config["base_service_config"]["path"])),
                ("predecessor_result", str(predecessor["path"])),
            )
        },
        "seals": config["seals"],
    }
    return _sealed(body)


def validate_epoch_genesis(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value)
    if value != build_epoch_genesis(root, config_path):
        raise ValueError("epoch genesis differs from exact reconstruction")


def _genesis_checkpoint(
    service_config: Mapping[str, Any], queue: Mapping[str, Any]
) -> dict[str, Any]:
    return service_sealed(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "service_id": service_config["service_id"],
            "pid": 0,
            "argv_sha256": service_sha(["epoch-genesis"]),
            "cycles": 0,
            "last_action": None,
            "queue_content_sha256": queue["content_sha256"],
            "state": "bounded_pause",
        }
    )


def materialize_epoch_runtime(root: Path, genesis: Mapping[str, Any]) -> Path:
    _validate_sealed(genesis)
    service_config = genesis["derived_service_config"]
    queue = genesis["initial_queue"]
    _validate_queue(queue)
    if queue["service_config_sha256"] != service_sha(service_config):
        raise ValueError("epoch queue/service-config binding mismatch")
    runtime = (root / str(service_config["runtime_directory"])).resolve()
    runtime.relative_to(root.resolve())
    runtime.mkdir(parents=True, exist_ok=True)
    checkpoint = _genesis_checkpoint(service_config, queue)
    files = {
        runtime / "service-config.json": _service_config_document(service_config),
        runtime / str(service_config["queue_name"]): queue,
        runtime / str(service_config["checkpoint_name"]): checkpoint,
    }
    for path, value in files.items():
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise RuntimeError("epoch runtime contains non-matching persistent state")
        else:
            _atomic_json(path, value, int(service_config["maximum_state_bytes"]))
    _validate_checkpoint(checkpoint, queue)
    return runtime


def normalize_bounded_pause(runtime: Path, service_config: Mapping[str, Any]) -> dict[str, Any]:
    queue_path = runtime / str(service_config["queue_name"])
    checkpoint_path = runtime / str(service_config["checkpoint_name"])
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    _validate_queue(queue)
    _validate_checkpoint(checkpoint, queue)
    complete = (
        queue["next_ordinal"] >= queue["stop_ordinal_exclusive"]
        and queue["generated_receipt"] is None
        and queue["formal_receipt"] is None
    )
    desired = "bounded_complete" if complete else "bounded_pause"
    if checkpoint["state"] == "stop_requested":
        desired = "stop_requested"
    body = {key: item for key, item in checkpoint.items() if key != "content_sha256"}
    body["state"] = desired
    normalized = service_sealed(body)
    _atomic_json(checkpoint_path, normalized, int(service_config["maximum_state_bytes"]))
    return normalized


def run_epoch_once(
    root: Path,
    config_path: Path,
    *,
    resource_probe: Callable[[], tuple[float, int]],
    generation_executor: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] = (
        execute_real_generation
    ),
    formal_executor: Callable[[Path, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] = (
        execute_real_formal
    ),
) -> dict[str, Any]:
    genesis = build_epoch_genesis(root, config_path)
    runtime = materialize_epoch_runtime(root, genesis)
    runtime_config_path = runtime / "service-config.json"
    document = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    _validate_sealed(document)
    service_config = document["service_config"]
    if (
        document.get("schema_version") != SERVICE_CONFIG_SCHEMA
        or document.get("service_config_sha256") != service_sha(service_config)
        or service_config != genesis["derived_service_config"]
    ):
        raise ValueError("persistent epoch service config mismatch")
    transient_path = runtime / ".active-service-config.json"
    _atomic_json(transient_path, service_config, int(service_config["maximum_state_bytes"]))
    try:
        run_bounded_service(
            root,
            transient_path,
            resource_probe=resource_probe,
            generation_executor=generation_executor,
            formal_executor=formal_executor,
            argv=["continuous-scientific-pipeline-epoch", genesis["epoch_id"]],
        )
    finally:
        if transient_path.exists():
            transient_path.unlink()
    return normalize_bounded_pause(runtime, genesis["derived_service_config"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_REL)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    root = Path.cwd()
    if not arguments.run:
        raise SystemExit("--run is required; genesis reconstruction never starts the epoch")
    try:
        import psutil
    except ImportError as error:
        raise SystemExit("psutil is required for fail-closed resource gates") from error
    checkpoint = run_epoch_once(
        root,
        root / arguments.config,
        resource_probe=lambda: (
            float(psutil.cpu_percent(interval=0.2)),
            int(psutil.virtual_memory().available // (1024 * 1024)),
        ),
    )
    print(json.dumps(checkpoint, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
