from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import queue as queue_module
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .continuous_scientific_pipeline_admission import admit_cycle

SCHEMA = "sigma-continuous-scientific-pipeline-service-config-1.0"
CHECKPOINT_SCHEMA = "sigma-continuous-scientific-pipeline-service-checkpoint-1.0"
QUEUE_SCHEMA = "sigma-continuous-scientific-pipeline-service-queue-1.0"
LEASE_SCHEMA = "sigma-continuous-scientific-pipeline-service-lease-1.0"
ARTIFACT_SCHEMA = "sigma-continuous-scientific-pipeline-service-readiness-1.0"
EXPECTED_EVALUATOR = "sigma_theory_compiler.real_formula_execution:cpu_formula_batch_evaluator"
CONFIG_REL = "configs/continuous_scientific_pipeline_service.json"
ADMISSION_CONFIG_REL = "configs/continuous_scientific_pipeline_admission.json"
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_service.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_service.py"


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
        raise ValueError("state content hash mismatch")


def _validate_queue(value: Mapping[str, Any]) -> None:
    _validate_sealed(value)
    if (
        set(value)
        != {
            "schema_version",
            "next_ordinal",
            "stop_ordinal_exclusive",
            "generated_receipt",
            "formal_receipt",
            "last_ranked_candidate_root",
            "leaderboard_rebuild_requests",
            "content_sha256",
        }
        or value.get("schema_version") != QUEUE_SCHEMA
    ):
        raise ValueError("queue key contract mismatch")
    if not isinstance(value["next_ordinal"], int) or not isinstance(
        value["stop_ordinal_exclusive"], int
    ):
        raise TypeError("queue ordinal contract mismatch")
    if not isinstance(value["leaderboard_rebuild_requests"], list):
        raise TypeError("queue rebuild request contract mismatch")


def _validate_checkpoint(value: Mapping[str, Any], queue: Mapping[str, Any]) -> None:
    _validate_sealed(value)
    if (
        set(value)
        != {
            "schema_version",
            "service_id",
            "pid",
            "argv_sha256",
            "cycles",
            "last_action",
            "queue_content_sha256",
            "state",
            "content_sha256",
        }
        or value.get("schema_version") != CHECKPOINT_SCHEMA
    ):
        raise ValueError("checkpoint key contract mismatch")
    if value["queue_content_sha256"] != queue["content_sha256"]:
        raise ValueError("checkpoint/queue root mismatch")


def load_service_config(root: Path, path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "service_id",
        "runtime_directory",
        "lease_name",
        "checkpoint_name",
        "queue_name",
        "stop_name",
        "maximum_cycles",
        "maximum_service_seconds",
        "maximum_action_seconds",
        "poll_seconds",
        "cpu_workers",
        "gpu_workers",
        "cpu_backoff_at_or_above_percent",
        "minimum_available_ram_mib",
        "batch_candidates_per_worker",
        "start_ordinal",
        "stop_ordinal_exclusive",
        "generator_config_path",
        "ambiguity_guard",
        "maximum_state_bytes",
        "admission_config_path",
        "allowlisted_evaluator",
        "formal_backend",
        "ranking_action",
        "seals",
    }
    if set(config) != expected or config["schema_version"] != SCHEMA:
        raise ValueError("service config contract mismatch")
    if config["cpu_workers"] != 15 or config["gpu_workers"] != 0:
        raise ValueError("worker contract mismatch")
    if (
        config["cpu_backoff_at_or_above_percent"] != 92
        or config["minimum_available_ram_mib"] != 32768
    ):
        raise ValueError("resource gate mismatch")
    if config["maximum_cycles"] <= 0 or not 0 < config["maximum_service_seconds"] <= 600:
        raise ValueError("service bound mismatch")
    if not 0 < config["maximum_action_seconds"] <= 120:
        raise ValueError("action bound mismatch")
    if config["allowlisted_evaluator"] != EXPECTED_EVALUATOR:
        raise ValueError("evaluator allowlist mismatch")
    if (
        config["formal_backend"] != "none_fail_closed"
        or config["ranking_action"] != "leaderboard_rebuild_request_only"
    ):
        raise ValueError("formal/ranking contract mismatch")
    if any(config["seals"].values()) or "campaign-v1-live.sqlite" in json.dumps(config):
        raise ValueError("service seal mismatch")
    runtime = (root / config["runtime_directory"]).resolve()
    runtime.relative_to(root.resolve())
    return config


def initial_queue(config: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": QUEUE_SCHEMA,
            "next_ordinal": config["start_ordinal"],
            "stop_ordinal_exclusive": config["stop_ordinal_exclusive"],
            "generated_receipt": None,
            "formal_receipt": None,
            "last_ranked_candidate_root": None,
            "leaderboard_rebuild_requests": [],
        }
    )


def _atomic_json(path: Path, value: Mapping[str, Any], maximum: int) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > maximum:
        raise ValueError("state exceeds byte bound")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_lease(
    runtime: Path, config: Mapping[str, Any], argv: list[str]
) -> tuple[Path, dict[str, Any]]:
    runtime.mkdir(parents=True, exist_ok=True)
    lease_path = runtime / config["lease_name"]
    body = {
        "schema_version": LEASE_SCHEMA,
        "service_id": config["service_id"],
        "pid": os.getpid(),
        "argv_sha256": _sha(argv),
    }
    lease = _sealed(body)
    raw = json.dumps(lease, sort_keys=True, separators=(",", ":")).encode()
    try:
        descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("service lease already exists; exact recovery is required") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return lease_path, lease


def release_lease(path: Path, owned: Mapping[str, Any]) -> None:
    current = json.loads(path.read_text(encoding="utf-8"))
    if current != owned:
        raise RuntimeError("lease ownership changed")
    path.unlink()


def cycle_state(queue: Mapping[str, Any], cpu_percent: float, ram_mib: int) -> dict[str, Any]:
    _validate_queue(queue)
    return {
        "cpu_utilization_percent": cpu_percent,
        "available_ram_mib": ram_mib,
        "cpu_generation_owner_active": False,
        "gpu_handoff_state": "waiting",
        "generated_receipt": queue["generated_receipt"],
        "formal_receipt": queue["formal_receipt"],
        "last_ranked_candidate_root": queue["last_ranked_candidate_root"],
        "dashboard_service_healthy": False,
        "dashboard_core_parity": False,
    }


def apply_action(
    queue: Mapping[str, Any], action: str, result: Mapping[str, Any] | None
) -> dict[str, Any]:
    _validate_queue(queue)
    body = {key: value for key, value in queue.items() if key != "content_sha256"}
    if action == "generate_and_screen":
        if result is None:
            raise ValueError("generation result missing")
        receipt = dict(result)
        receipt_body = dict(receipt)
        claimed = receipt_body.pop("content_sha256", None)
        if claimed != _sha(receipt_body):
            raise ValueError("generation receipt hash mismatch")
        body["generated_receipt"] = receipt
        body["next_ordinal"] += int(receipt["unique_formula_count"])
    elif action == "formal_validate":
        generated = body["generated_receipt"]
        formal_body = {
            "candidate_root_sha256": generated["candidate_root_sha256"],
            "generated_receipt_sha256": generated["content_sha256"],
            "decision": "block",
            "complete_comparable_evidence": False,
            "observations_opened": False,
            "forbidden_target_inputs_opened": False,
        }
        body["formal_receipt"] = _sealed(formal_body)
    elif action == "rank_project":
        formal = body["formal_receipt"]
        if formal["decision"] != "pass" or not formal["complete_comparable_evidence"]:
            raise ValueError("ranking request lacks formal evidence")
        body["leaderboard_rebuild_requests"] = [formal["candidate_root_sha256"]]
        body["last_ranked_candidate_root"] = formal["candidate_root_sha256"]
    elif action != "wait":
        raise ValueError("unknown service action")
    return _sealed(body)


def run_bounded_service(
    root: Path,
    config_path: Path,
    *,
    resource_probe: Callable[[], tuple[float, int]],
    generation_executor: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    argv: list[str] | None = None,
) -> dict[str, Any]:
    config = load_service_config(root, config_path)
    admission = json.loads((root / config["admission_config_path"]).read_text(encoding="utf-8"))
    runtime = (root / config["runtime_directory"]).resolve()
    queue_path = runtime / config["queue_name"]
    checkpoint_path = runtime / config["checkpoint_name"]
    stop_path = runtime / config["stop_name"]
    started = time.monotonic()
    if queue_path.exists() != checkpoint_path.exists():
        raise RuntimeError("queue/checkpoint resume pair is incomplete")
    queue = (
        json.loads(queue_path.read_text(encoding="utf-8"))
        if queue_path.exists()
        else initial_queue(config)
    )
    _validate_queue(queue)
    if checkpoint_path.exists():
        _validate_checkpoint(json.loads(checkpoint_path.read_text(encoding="utf-8")), queue)
    lease_path, owned = acquire_lease(runtime, config, argv or sys.argv)
    cycles = 0
    state = "bounded_complete"
    try:
        while (
            cycles < config["maximum_cycles"]
            and time.monotonic() - started < config["maximum_service_seconds"]
        ):
            if stop_path.exists():
                state = "stop_requested"
                break
            cpu, ram = resource_probe()
            decision = admit_cycle(cycle_state(queue, cpu, ram), admission)
            action = decision["action"]
            result = None
            if action == "generate_and_screen":
                result = generation_executor(queue, config)
            queue = apply_action(queue, action, result)
            cycles += 1
            checkpoint = _sealed(
                {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "service_id": config["service_id"],
                    "pid": os.getpid(),
                    "argv_sha256": owned["argv_sha256"],
                    "cycles": cycles,
                    "last_action": action,
                    "queue_content_sha256": queue["content_sha256"],
                    "state": "running",
                }
            )
            _atomic_json(queue_path, queue, config["maximum_state_bytes"])
            _atomic_json(checkpoint_path, checkpoint, config["maximum_state_bytes"])
            if action == "wait":
                time.sleep(
                    min(
                        config["poll_seconds"],
                        max(0, config["maximum_service_seconds"] - (time.monotonic() - started)),
                    )
                )
        final = _sealed(
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "service_id": config["service_id"],
                "pid": os.getpid(),
                "argv_sha256": owned["argv_sha256"],
                "cycles": cycles,
                "last_action": None,
                "queue_content_sha256": queue["content_sha256"],
                "state": state,
            }
        )
        _atomic_json(checkpoint_path, final, config["maximum_state_bytes"])
        return final
    finally:
        release_lease(lease_path, owned)


def _real_generation_worker(start: int, count: int, config: dict[str, Any], output: Any) -> None:
    from .persistent_parallel_search import WorkLease
    from .real_formula_execution import cpu_formula_batch_evaluator

    try:
        generator_path = Path(config["generator_config_path"])
        generator = json.loads(generator_path.read_text(encoding="utf-8"))
        payload = {
            "generator_config_path": str(generator_path),
            "generator_config_sha256": _file_sha(generator_path),
            "protocol_version": generator["protocol_version"],
            "basis_count": generator["basis_count"],
            "max_action_terms": generator["max_action_terms"],
            "start_ordinal": start,
            "end_ordinal_exclusive": start + count,
            "candidate_count": count,
            "ambiguity_guard": config["ambiguity_guard"],
            "data_eligibility": {
                "observational_data_opened": False,
                "dark_matter_or_halo_inputs": False,
                "redshift_distance_inputs": False,
            },
        }
        result = cpu_formula_batch_evaluator(
            WorkLease(f"continuous-{start}", start, "cpu", start, 1, 1, payload)
        )
        output.put({"ok": True, "start": start, "result": result})
    except Exception as error:  # noqa: BLE001 - child returns bounded failure receipt
        output.put({"ok": False, "start": start, "error": f"{type(error).__name__}: {error}"})


def execute_real_generation(
    queue: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Run exactly 15 real CPU shards with a hard owned-child deadline."""
    start = int(queue["next_ordinal"])
    per_worker = int(config["batch_candidates_per_worker"])
    total = min(
        int(config["cpu_workers"]) * per_worker,
        int(config["stop_ordinal_exclusive"]) - start,
    )
    if total != int(config["cpu_workers"]) * per_worker:
        raise RuntimeError("full 15-worker batch unavailable; ordinal interval exhausted")
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    children = []
    config_copy = dict(config)
    for worker in range(int(config["cpu_workers"])):
        child = context.Process(
            target=_real_generation_worker,
            args=(start + worker * per_worker, per_worker, config_copy, output),
        )
        child.start()
        children.append(child)
    deadline = time.monotonic() + float(config["maximum_action_seconds"])
    receipts = []
    while len(receipts) < len(children):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            receipts.append(output.get(timeout=min(1.0, remaining)))
        except queue_module.Empty:
            if all(not child.is_alive() for child in children):
                break
    for child in children:
        child.join(max(0.0, deadline - time.monotonic()))
    unfinished = [child for child in children if child.is_alive()]
    if unfinished:
        for child in unfinished:
            child.terminate()
        for child in unfinished:
            child.join(5)
    if unfinished or len(receipts) != len(children):
        raise TimeoutError("real formula generation exceeded hard action deadline")
    if any(child.exitcode != 0 for child in children):
        raise RuntimeError("real formula child exited unsuccessfully")
    if any(not receipt["ok"] for receipt in receipts):
        raise RuntimeError("real formula child failed closed")
    ordered = [item["result"] for item in sorted(receipts, key=lambda item: item["start"])]
    counts = {
        key: sum(int(result["counts"][key]) for result in ordered)
        for key in ("pass", "reject", "ambiguous")
    }
    screen = "pass" if counts["pass"] else ("ambiguous" if counts["ambiguous"] else "reject")
    body = {
        "candidate_root_sha256": hashlib.sha256(
            "".join(result["status_root_sha256"] for result in ordered).encode()
        ).hexdigest(),
        "screen_decision": screen,
        "unique_formula_count": total,
        "theory_pass_claimed": False,
        "observations_opened": False,
        "rank_eligible": False,
    }
    return _sealed(body)


def build_readiness(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_service_config(root, config_path)
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "decision": "bounded_single_owner_CPU_pipeline_service_implemented_not_started",
        "service_contract": {
            "single_owner_O_EXCL_PID_argv_lease": True,
            "isolated_atomic_JSON_queue": True,
            "checkpoint_resume": True,
            "external_stop_request": True,
            "maximum_actions_per_cycle": 1,
            "hard_owned_child_action_timeout": True,
            "maximum_cycles": config["maximum_cycles"],
            "maximum_service_seconds": config["maximum_service_seconds"],
            "maximum_action_seconds": config["maximum_action_seconds"],
        },
        "resource_contract": {
            "cpu_workers": 15,
            "gpu_workers": 0,
            "CPU_backoff_percent": 92,
            "minimum_RAM_MiB": 32768,
        },
        "scientific_contract": {
            "real_formula_evaluator_allowlisted": True,
            "formal_backend_available": False,
            "formal_stage_fails_closed": True,
            "ranking_is_request_only": True,
            "direct_rank_assignment": False,
        },
        "execution_state": {
            "service_started": False,
            "cycles_executed": 0,
            "queue_created": False,
            "live_SQLite_accessed": False,
        },
        "safe_start_criteria": [
            "runtime path remains Git-ignored and contains no foreign lease",
            "available RAM is at least 32768 MiB",
            "device-wide CPU is below 92 percent",
            "no other CPU generation owner is active",
            "ordinal interval is unconsumed",
            "formal blocks are expected until a reviewed candidate-specific backend is registered",
        ],
        "first_remaining_blocker": "register_a_candidate_specific_formal_backend_before_any_rank_rebuild_can_be_admitted",
        "bindings": {
            label: {"path": rel, "file_sha256": _file_sha(root / rel)}
            for label, rel in (
                ("config", CONFIG_REL),
                ("admission_config", ADMISSION_CONFIG_REL),
                ("source", SOURCE_REL),
                ("test", TEST_REL),
            )
        },
        "seals": config["seals"],
    }
    return _sealed(body)


def validate_readiness(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value)
    if value != build_readiness(root, config_path):
        raise ValueError("readiness differs from reconstruction")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_REL)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("--run is required; readiness never starts the service")
    root = Path.cwd()
    try:
        import psutil

        probe = lambda: (
            float(psutil.cpu_percent(interval=0.2)),
            int(psutil.virtual_memory().available // (1024 * 1024)),
        )
    except ImportError as error:
        raise SystemExit("psutil is required for fail-closed resource gates") from error
    run_bounded_service(
        root, root / args.config, resource_probe=probe, generation_executor=execute_real_generation
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
