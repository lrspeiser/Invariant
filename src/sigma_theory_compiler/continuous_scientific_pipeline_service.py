from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import queue as queue_module
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .continuous_formula_formal_backend import (
    BACKEND_ID,
    build_formal_evidence,
    combine_candidate_manifests,
    extract_candidate_manifest,
    load_backend_config,
    validate_candidate_manifest,
    validate_formal_evidence,
)
from .continuous_scientific_pipeline_admission import (
    _validate_formal_receipt,
    _validate_generated_receipt,
    admit_cycle,
)

SCHEMA = "sigma-continuous-scientific-pipeline-service-config-2.0"
CHECKPOINT_SCHEMA = "sigma-continuous-scientific-pipeline-service-checkpoint-2.0"
QUEUE_SCHEMA = "sigma-continuous-scientific-pipeline-service-queue-2.0"
LEASE_SCHEMA = "sigma-continuous-scientific-pipeline-service-lease-1.0"
ARTIFACT_SCHEMA = "sigma-continuous-scientific-pipeline-service-readiness-2.0"
RESULT_SCHEMA = "sigma-continuous-scientific-pipeline-service-result-2.0"
EXPECTED_EVALUATOR = "sigma_theory_compiler.real_formula_execution:cpu_formula_batch_evaluator"
CONFIG_REL = "configs/continuous_scientific_pipeline_service.json"
ADMISSION_CONFIG_REL = "configs/continuous_scientific_pipeline_admission.json"
SOURCE_REL = "src/sigma_theory_compiler/continuous_scientific_pipeline_service.py"
TEST_REL = "tests/test_continuous_scientific_pipeline_service.py"
RESULT_REL = "runs/engine/continuous-scientific-pipeline-service-result.json"
EXPECTED_COMPLETED_RECEIPTS_SHA256 = (
    "223ede8c98e414d5bb71905b8ddc72ac1818adb742e544bccc456e18f09ae960"
)
RESULT_DECISION = "bounded_interval_complete_survivor_batches_formally_blocked_no_rank"
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _validate_completed_receipt_binding(value: Mapping[str, Any]) -> None:
    expected = {
        "candidate_root_sha256",
        "generated_receipt_sha256",
        "candidate_manifest_sha256",
        "screen_decision",
        "formal_receipt_sha256",
        "formal_evidence_sha256",
        "formal_decision",
    }
    formal_values = (
        value.get("formal_receipt_sha256"),
        value.get("formal_evidence_sha256"),
        value.get("formal_decision"),
    )
    if (
        set(value) != expected
        or value.get("screen_decision") not in {"reject", "pass", "ambiguous"}
        or any(
            not _SHA256.fullmatch(str(value.get(key, "")))
            for key in (
                "candidate_root_sha256",
                "generated_receipt_sha256",
                "candidate_manifest_sha256",
            )
        )
        or (all(item is None for item in formal_values) != (value["screen_decision"] != "pass"))
        or (
            value["screen_decision"] == "pass"
            and (
                value["formal_decision"] not in {"block", "reject"}
                or not _SHA256.fullmatch(str(value["formal_receipt_sha256"]))
                or not _SHA256.fullmatch(str(value["formal_evidence_sha256"]))
            )
        )
    ):
        raise ValueError("completed receipt binding contract mismatch")


def _validate_queue(value: Mapping[str, Any]) -> None:
    _validate_sealed(value)
    if (
        set(value)
        != {
            "schema_version",
            "next_ordinal",
            "stop_ordinal_exclusive",
            "generated_receipt",
            "generation_manifest",
            "formal_receipt",
            "formal_evidence",
            "completed_action_receipts",
            "service_config_sha256",
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
    if not isinstance(value["completed_action_receipts"], list):
        raise TypeError("queue completed receipt contract mismatch")
    for completed_receipt in value["completed_action_receipts"]:
        _validate_completed_receipt_binding(completed_receipt)
    generated = value["generated_receipt"]
    manifest = value["generation_manifest"]
    formal = value["formal_receipt"]
    evidence = value["formal_evidence"]
    if (generated is None) != (manifest is None):
        raise ValueError("queue generation receipt/manifest state mismatch")
    if generated is not None:
        _validate_generated_receipt(generated)
        validate_candidate_manifest(manifest)
        if manifest["candidate_root_sha256"] != generated["candidate_root_sha256"]:
            raise ValueError("queue generated receipt/manifest root mismatch")
    if (formal is None) != (evidence is None):
        raise ValueError("queue formal receipt/evidence state mismatch")
    if formal is not None:
        _validate_formal_receipt(formal)
        validate_formal_evidence(evidence)
        if (
            generated is None
            or formal["decision"] == "pass"
            or formal["complete_comparable_evidence"] is not False
            or formal["candidate_root_sha256"] != generated["candidate_root_sha256"]
            or formal["generated_receipt_sha256"] != generated["content_sha256"]
            or evidence["candidate_root_sha256"] != formal["candidate_root_sha256"]
            or evidence["generated_receipt_sha256"] != formal["generated_receipt_sha256"]
            or evidence["candidate_manifest_sha256"] != manifest["content_sha256"]
            or evidence["decision"] != formal["decision"]
        ):
            raise ValueError("queue formal evidence lineage mismatch")
    if value["leaderboard_rebuild_requests"] or value["last_ranked_candidate_root"] is not None:
        raise ValueError("current fail-closed backend cannot create ranking requests")


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


def _close_process_queue(output: Any) -> None:
    output.close()
    output.cancel_join_thread()


def _run_owned_child(
    target: Callable[..., None],
    args: tuple[Any, ...],
    *,
    maximum_seconds: float,
    action_name: str,
) -> Mapping[str, Any]:
    """Run one campaign-owned spawn child inside a cleanup-inclusive wall-clock bound."""
    if not 0 < maximum_seconds <= 120:
        raise ValueError("owned child deadline contract mismatch")
    context = multiprocessing.get_context("spawn")
    output = context.Queue(maxsize=1)
    child = context.Process(target=target, args=(*args, output))
    started = time.monotonic()
    final_deadline = started + maximum_seconds
    cleanup_reserve = min(2.0, max(0.1, maximum_seconds / 4.0))
    work_deadline = final_deadline - cleanup_reserve
    payload: Mapping[str, Any] | None = None
    try:
        child.start()
        while time.monotonic() < work_deadline:
            remaining = work_deadline - time.monotonic()
            try:
                payload = output.get(timeout=min(0.1, max(0.0, remaining)))
                break
            except queue_module.Empty:
                if not child.is_alive():
                    break
        if payload is not None:
            child.join(max(0.0, final_deadline - time.monotonic()))
        if child.is_alive():
            child.terminate()
            child.join(max(0.0, final_deadline - time.monotonic()))
        if child.is_alive():
            child.kill()
            child.join(max(0.0, final_deadline - time.monotonic()))
        if child.is_alive():
            raise TimeoutError(f"{action_name} child could not be terminated by hard deadline")
        if payload is None:
            if time.monotonic() >= work_deadline:
                raise TimeoutError(f"{action_name} exceeded hard action deadline")
            raise RuntimeError(f"{action_name} child exited without a result")
        if child.exitcode != 0:
            raise RuntimeError(f"{action_name} child exited unsuccessfully")
        if set(payload) != {"ok", "result"} and set(payload) != {"ok", "error"}:
            raise RuntimeError(f"{action_name} child result contract mismatch")
        if payload["ok"] is not True:
            raise RuntimeError(f"{action_name} child failed closed: {payload['error']}")
        result = payload["result"]
        if not isinstance(result, Mapping):
            raise TypeError(f"{action_name} child returned a non-mapping result")
        return result
    finally:
        if child.is_alive():
            child.kill()
            child.join(max(0.0, final_deadline - time.monotonic()))
        _close_process_queue(output)


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
        "formal_backend_config_path",
        "maximum_candidate_records",
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
        config["formal_backend"] != BACKEND_ID
        or config["ranking_action"] != "leaderboard_rebuild_request_only"
    ):
        raise ValueError("formal/ranking contract mismatch")
    if config["maximum_candidate_records"] != 32:
        raise ValueError("candidate manifest bound mismatch")
    load_backend_config(root, root / config["formal_backend_config_path"])
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
            "generation_manifest": None,
            "formal_receipt": None,
            "formal_evidence": None,
            "completed_action_receipts": [],
            "service_config_sha256": _sha(config),
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
        if set(result) != {"receipt", "manifest"}:
            raise ValueError("generation result contract mismatch")
        receipt = dict(result["receipt"])
        receipt_body = dict(receipt)
        claimed = receipt_body.pop("content_sha256", None)
        if claimed != _sha(receipt_body):
            raise ValueError("generation receipt hash mismatch")
        manifest = dict(result["manifest"])
        validate_candidate_manifest(manifest)
        if manifest["candidate_root_sha256"] != receipt["candidate_root_sha256"]:
            raise ValueError("generation manifest root mismatch")
        body["generated_receipt"] = receipt
        body["generation_manifest"] = manifest
        body["next_ordinal"] += int(receipt["unique_formula_count"])
    elif action == "formal_validate":
        if result is None or set(result) != {"receipt", "evidence"}:
            raise ValueError("formal result contract mismatch")
        generated = body["generated_receipt"]
        formal = dict(result["receipt"])
        evidence = dict(result["evidence"])
        formal_body = dict(formal)
        formal_claimed = formal_body.pop("content_sha256", None)
        if formal_claimed != _sha(formal_body):
            raise ValueError("formal receipt hash mismatch")
        validate_formal_evidence(evidence)
        if (
            formal["candidate_root_sha256"] != generated["candidate_root_sha256"]
            or formal["generated_receipt_sha256"] != generated["content_sha256"]
            or evidence["candidate_root_sha256"] != formal["candidate_root_sha256"]
            or evidence["generated_receipt_sha256"] != formal["generated_receipt_sha256"]
            or evidence["candidate_manifest_sha256"]
            != body["generation_manifest"]["content_sha256"]
            or evidence["decision"] != formal["decision"]
        ):
            raise ValueError("formal result lineage mismatch")
        body["formal_receipt"] = formal
        body["formal_evidence"] = evidence
    elif action == "rank_project":
        raise ValueError("current formal backend cannot admit ranking requests")
    elif action == "wait":
        generated = body["generated_receipt"]
        formal = body["formal_receipt"]
        should_archive = generated is not None and (
            generated["screen_decision"] != "pass"
            or (formal is not None and formal["decision"] != "pass")
        )
        if should_archive:
            body["completed_action_receipts"].append(
                {
                    "candidate_root_sha256": generated["candidate_root_sha256"],
                    "generated_receipt_sha256": generated["content_sha256"],
                    "candidate_manifest_sha256": body["generation_manifest"]["content_sha256"],
                    "screen_decision": generated["screen_decision"],
                    "formal_receipt_sha256": formal["content_sha256"] if formal else None,
                    "formal_evidence_sha256": (
                        body["formal_evidence"]["content_sha256"]
                        if body["formal_evidence"]
                        else None
                    ),
                    "formal_decision": formal["decision"] if formal else None,
                }
            )
            body["generated_receipt"] = None
            body["generation_manifest"] = None
            body["formal_receipt"] = None
            body["formal_evidence"] = None
    else:
        raise ValueError("unknown service action")
    return _sealed(body)


def run_bounded_service(
    root: Path,
    config_path: Path,
    *,
    resource_probe: Callable[[], tuple[float, int]],
    generation_executor: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    formal_executor: Callable[[Path, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
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
    if queue["service_config_sha256"] != _sha(config):
        raise RuntimeError("queue service-config root mismatch")
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
            if (
                queue["next_ordinal"] >= queue["stop_ordinal_exclusive"]
                and queue["generated_receipt"] is None
                and queue["formal_receipt"] is None
            ):
                state = "bounded_complete"
                break
            cpu, ram = resource_probe()
            decision = admit_cycle(cycle_state(queue, cpu, ram), admission)
            action = decision["action"]
            result = None
            if action == "generate_and_screen":
                result = generation_executor(queue, config)
            elif action == "formal_validate":
                result = formal_executor(root, queue, config)
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
        manifest = extract_candidate_manifest(payload, result)
        output.put({"ok": True, "start": start, "result": result, "manifest": manifest})
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
    receipt_rows = sorted(receipts, key=lambda item: item["start"])
    ordered = [item["result"] for item in receipt_rows]
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
    receipt = _sealed(body)
    manifest = combine_candidate_manifests(
        [item["manifest"] for item in receipt_rows], int(config["maximum_candidate_records"])
    )
    if manifest["candidate_root_sha256"] != receipt["candidate_root_sha256"]:
        raise ValueError("combined candidate manifest root mismatch")
    return {"receipt": receipt, "manifest": manifest}


def _real_formal_worker(
    root_text: str,
    output_root_text: str,
    generated: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Any,
) -> None:
    """Execute the formal backend only inside the service-owned isolation child."""
    try:
        root = Path(root_text)
        backend = load_backend_config(root, root / str(config["formal_backend_config_path"]))
        receipt, evidence = build_formal_evidence(
            generated,
            manifest,
            backend,
            root=root,
            output_root=Path(output_root_text),
        )
        output.put({"ok": True, "result": {"receipt": receipt, "evidence": evidence}})
    except Exception as error:  # noqa: BLE001 - child returns bounded failure receipt
        output.put({"ok": False, "error": f"{type(error).__name__}: {error}"})


def execute_real_formal(
    root: Path, queue: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Run formal validation in one owned spawn child with a hard wall-clock bound."""
    generated = queue["generated_receipt"]
    manifest = queue["generation_manifest"]
    if generated is None or manifest is None:
        raise ValueError("formal execution requires generated receipt and manifest")
    runtime = (root / str(config["runtime_directory"])).resolve()
    runtime.relative_to(root.resolve())
    attempts = runtime / "formal-attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="owned-", dir=attempts))
    try:
        return _run_owned_child(
            _real_formal_worker,
            (
                str(root.resolve()),
                str(attempt),
                dict(generated),
                dict(manifest),
                dict(config),
            ),
            maximum_seconds=float(config["maximum_action_seconds"]),
            action_name="formal validation",
        )
    finally:
        attempt.resolve().relative_to(attempts.resolve())
        shutil.rmtree(attempt)


def build_readiness(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_service_config(root, config_path)
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "decision": "preexecution_preregistration_snapshot_no_current_runtime_claim",
        "snapshot_scope": {
            "artifact_role": "preexecution_preregistration",
            "runtime_status_asserted": False,
            "completed_execution_reported_separately": True,
            "completed_execution_result_path": RESULT_REL,
        },
        "service_contract": {
            "single_owner_O_EXCL_PID_argv_lease": True,
            "isolated_atomic_JSON_queue": True,
            "checkpoint_resume": True,
            "external_stop_request": True,
            "maximum_actions_per_cycle": 1,
            "hard_owned_child_generation_timeout": True,
            "hard_owned_child_formal_timeout": True,
            "timeout_cleanup_is_campaign_owned_child_only": True,
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
            "candidate_manifest_reconstructed_from_ordinals": True,
            "formal_backend_available": True,
            "formal_backend": config["formal_backend"],
            "covariant_action_health_executed_only_after_exact_mapping": True,
            "formal_stage_fails_closed": True,
            "ranking_is_request_only": True,
            "direct_rank_assignment": False,
        },
        "execution_state": {
            "service_started_at_preregistration": False,
            "cycles_executed_at_preregistration": 0,
            "queue_created_at_preregistration": False,
            "current_runtime_status_claimed": False,
            "live_SQLite_accessed": False,
        },
        "safe_start_criteria": [
            "runtime path remains Git-ignored and contains no foreign lease",
            "available RAM is at least 32768 MiB",
            "device-wide CPU is below 92 percent",
            "no other CPU generation owner is active",
            "ordinal interval is unconsumed",
            "survivor count is within the fixed candidate-manifest bound",
            "formal blocks or exact hard rejections are expected until complete comparable evidence exists",
        ],
        "first_remaining_blocker": "complete_candidate_specific_comparable_evidence_after_covariant_action_health_before_any_rank_rebuild_can_be_admitted",
        "bindings": {
            label: {"path": rel, "file_sha256": _file_sha(root / rel)}
            for label, rel in (
                ("config", CONFIG_REL),
                ("admission_config", ADMISSION_CONFIG_REL),
                ("source", SOURCE_REL),
                ("test", TEST_REL),
                (
                    "formal_backend_config",
                    config["formal_backend_config_path"],
                ),
                (
                    "formal_backend_source",
                    "src/sigma_theory_compiler/continuous_formula_formal_backend.py",
                ),
                (
                    "formal_backend_test",
                    "tests/test_continuous_formula_formal_backend.py",
                ),
            )
        },
        "seals": config["seals"],
    }
    return _sealed(body)


def validate_readiness(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value)
    if value != build_readiness(root, config_path):
        raise ValueError("readiness differs from reconstruction")


def _result_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        label: {"path": rel, "file_sha256": _file_sha(root / rel)}
        for label, rel in (
            ("config", CONFIG_REL),
            ("source", SOURCE_REL),
            ("test", TEST_REL),
            ("formal_backend_config", str(config["formal_backend_config_path"])),
        )
    }


def _replay_dependencies(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    backend = json.loads(
        (root / str(config["formal_backend_config_path"])).read_text(encoding="utf-8")
    )
    paths = (
        ("service_config", CONFIG_REL),
        ("admission_config", ADMISSION_CONFIG_REL),
        ("service_source", SOURCE_REL),
        ("service_test", TEST_REL),
        ("generator_config", str(config["generator_config_path"])),
        ("formal_backend_config", str(config["formal_backend_config_path"])),
        ("grammar", str(backend["grammar_path"])),
        ("field_contract", str(backend["field_contract_path"])),
        ("formal_controls", str(backend["formal_controls_path"])),
        ("candidate_mapper_source", str(backend["candidate_mapper_source_path"])),
        ("action_health_source", str(backend["action_health_source_path"])),
    )
    source_file_hashes = {
        "formal_backend_source": _file_sha(
            root / "src/sigma_theory_compiler/continuous_formula_formal_backend.py"
        ),
        "real_formula_execution_source": _file_sha(
            root / "src/sigma_theory_compiler/real_formula_execution.py"
        ),
        "high_throughput_source": _file_sha(root / "src/sigma_theory_compiler/high_throughput.py"),
        "gpu_screen_source": _file_sha(root / "src/sigma_theory_compiler/gpu_screen.py"),
    }
    files = {
        label: {"path": relative, "file_sha256": _file_sha(root / relative)}
        for label, relative in paths
    }
    dependency_body = {"files": files, "source_file_hashes": source_file_hashes}
    return {
        "replay_method": "deterministic_ordinal_generation_then_candidate_bound_formal_backend",
        "files": files,
        "source_file_hashes": source_file_hashes,
        "replay_dependency_root_sha256": _sha(dependency_body),
    }


def _completed_replay_records(
    completed: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    replay_dependencies: Mapping[str, Any],
) -> list[dict[str, Any]]:
    batch_size = int(config["cpu_workers"]) * int(config["batch_candidates_per_worker"])
    records = []
    for index, archived in enumerate(completed):
        _validate_completed_receipt_binding(archived)
        start = int(config["start_ordinal"]) + index * batch_size
        stop = start + batch_size
        generated_receipt = _sealed(
            {
                "candidate_root_sha256": archived["candidate_root_sha256"],
                "screen_decision": archived["screen_decision"],
                "unique_formula_count": batch_size,
                "theory_pass_claimed": False,
                "observations_opened": False,
                "rank_eligible": False,
            }
        )
        if generated_receipt["content_sha256"] != archived["generated_receipt_sha256"]:
            raise ValueError("completed generated receipt is not independently reconstructable")
        formal_receipt = None
        formal_evidence_binding = None
        if archived["formal_receipt_sha256"] is not None:
            formal_receipt = _sealed(
                {
                    "candidate_root_sha256": archived["candidate_root_sha256"],
                    "generated_receipt_sha256": generated_receipt["content_sha256"],
                    "decision": archived["formal_decision"],
                    "complete_comparable_evidence": False,
                    "observations_opened": False,
                    "forbidden_target_inputs_opened": False,
                }
            )
            if formal_receipt["content_sha256"] != archived["formal_receipt_sha256"]:
                raise ValueError("completed formal receipt is not independently reconstructable")
            formal_evidence_binding = {
                "content_sha256": archived["formal_evidence_sha256"],
                "candidate_manifest_sha256": archived["candidate_manifest_sha256"],
                "formal_receipt_sha256": formal_receipt["content_sha256"],
                "decision": archived["formal_decision"],
            }
        records.append(
            {
                "batch_index": index,
                "ordinal_interval": {
                    "start_ordinal": start,
                    "stop_ordinal_exclusive": stop,
                    "unique_formula_count": batch_size,
                },
                "generated_receipt": generated_receipt,
                "candidate_manifest_binding": {
                    "content_sha256": archived["candidate_manifest_sha256"],
                    "candidate_root_sha256": archived["candidate_root_sha256"],
                    "replay_dependency_root_sha256": replay_dependencies[
                        "replay_dependency_root_sha256"
                    ],
                },
                "formal_receipt": formal_receipt,
                "formal_evidence_binding": formal_evidence_binding,
            }
        )
    return records


def _execution_outcomes(completed: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "sampled_static_reject_batches": sum(
            row["screen_decision"] == "reject" for row in completed
        ),
        "sampled_static_pass_batches": sum(row["screen_decision"] == "pass" for row in completed),
        "formal_receipts": sum(row["formal_receipt_sha256"] is not None for row in completed),
        "formal_blocks": sum(row["formal_decision"] == "block" for row in completed),
        "formal_passes": 0,
        "leaderboard_rebuild_requests": 0,
        "rank_assignments": 0,
    }


def _execution_interpretation(outcomes: Mapping[str, int]) -> str:
    return (
        f"{outcomes['sampled_static_reject_batches']} batches failed the sampled-static screen; "
        f"{outcomes['sampled_static_pass_batches']} batches produced bounded survivor manifests, "
        f"and {outcomes['formal_blocks']} were formally blocked. No formal pass, theory verdict, "
        "ranking change, or observational claim follows."
    )


def build_execution_result(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_service_config(root, config_path)
    runtime = root / config["runtime_directory"]
    queue = json.loads((runtime / config["queue_name"]).read_text(encoding="utf-8"))
    checkpoint = json.loads((runtime / config["checkpoint_name"]).read_text(encoding="utf-8"))
    _validate_queue(queue)
    _validate_checkpoint(checkpoint, queue)
    completed = queue["completed_action_receipts"]
    if (
        checkpoint["state"] != "bounded_complete"
        or queue["next_ordinal"] != queue["stop_ordinal_exclusive"]
        or queue["generated_receipt"] is not None
        or queue["formal_receipt"] is not None
        or queue["generation_manifest"] is not None
        or queue["formal_evidence"] is not None
        or queue["leaderboard_rebuild_requests"] != []
        or any(row["formal_decision"] == "pass" for row in completed)
        or _sha(completed) != EXPECTED_COMPLETED_RECEIPTS_SHA256
    ):
        raise ValueError("service result is not the registered completed fail-closed run")
    replay_dependencies = _replay_dependencies(root, config)
    replay_records = _completed_replay_records(completed, config, replay_dependencies)
    outcomes = _execution_outcomes(completed)
    body = {
        "schema_version": RESULT_SCHEMA,
        "decision": RESULT_DECISION,
        "coverage": {
            "start_ordinal": config["start_ordinal"],
            "stop_ordinal_exclusive": config["stop_ordinal_exclusive"],
            "unique_formula_count": config["stop_ordinal_exclusive"] - config["start_ordinal"],
            "real_CPU_batches": len(completed),
            "workers_per_batch": config["cpu_workers"],
            "formulas_per_worker": config["batch_candidates_per_worker"],
        },
        "outcomes": outcomes,
        "completed_receipt_bindings": replay_records,
        "replay_dependencies": replay_dependencies,
        "terminal_runtime_archive": {"queue": queue, "checkpoint": checkpoint},
        "runtime_binding": {
            "queue_content_sha256": queue["content_sha256"],
            "checkpoint_content_sha256": checkpoint["content_sha256"],
            "completed_receipts_sha256": _sha(completed),
            "completed_replay_records_sha256": _sha(replay_records),
            "terminal_state": checkpoint["state"],
        },
        "interpretation": _execution_interpretation(outcomes),
        "bindings": _result_bindings(root, config),
        "seals": config["seals"],
    }
    return _sealed(body)


def validate_execution_result(value: Mapping[str, Any], root: Path, config_path: Path) -> None:
    _validate_sealed(value)
    config = load_service_config(root, config_path)
    expected_keys = {
        "schema_version",
        "decision",
        "coverage",
        "outcomes",
        "completed_receipt_bindings",
        "replay_dependencies",
        "terminal_runtime_archive",
        "runtime_binding",
        "interpretation",
        "bindings",
        "seals",
        "content_sha256",
    }
    terminal = value.get("terminal_runtime_archive", {})
    queue = terminal.get("queue", {})
    checkpoint = terminal.get("checkpoint", {})
    try:
        _validate_queue(queue)
        _validate_checkpoint(checkpoint, queue)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("service execution result terminal archive mismatch") from error
    completed = queue["completed_action_receipts"]
    replay_dependencies = _replay_dependencies(root, config)
    try:
        replay_records = _completed_replay_records(completed, config, replay_dependencies)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("service execution result replay archive mismatch") from error
    outcomes = _execution_outcomes(completed)
    expected_coverage = {
        "start_ordinal": config["start_ordinal"],
        "stop_ordinal_exclusive": config["stop_ordinal_exclusive"],
        "unique_formula_count": config["stop_ordinal_exclusive"] - config["start_ordinal"],
        "real_CPU_batches": len(completed),
        "workers_per_batch": config["cpu_workers"],
        "formulas_per_worker": config["batch_candidates_per_worker"],
    }
    expected_runtime_binding = {
        "queue_content_sha256": queue["content_sha256"],
        "checkpoint_content_sha256": checkpoint["content_sha256"],
        "completed_receipts_sha256": _sha(completed),
        "completed_replay_records_sha256": _sha(replay_records),
        "terminal_state": checkpoint["state"],
    }
    intervals = [row["ordinal_interval"] for row in replay_records]
    interval_contract = (
        bool(intervals)
        and intervals[0]["start_ordinal"] == config["start_ordinal"]
        and intervals[-1]["stop_ordinal_exclusive"] == config["stop_ordinal_exclusive"]
        and all(
            left["stop_ordinal_exclusive"] == right["start_ordinal"]
            for left, right in itertools.pairwise(intervals)
        )
        and sum(row["unique_formula_count"] for row in intervals)
        == expected_coverage["unique_formula_count"]
    )
    if (
        set(value) != expected_keys
        or value.get("schema_version") != RESULT_SCHEMA
        or value.get("decision") != RESULT_DECISION
        or value.get("coverage") != expected_coverage
        or value.get("outcomes") != outcomes
        or value.get("completed_receipt_bindings") != replay_records
        or value.get("replay_dependencies") != replay_dependencies
        or value.get("runtime_binding") != expected_runtime_binding
        or value.get("interpretation") != _execution_interpretation(outcomes)
        or value.get("bindings") != _result_bindings(root, config)
        or value.get("seals") != config["seals"]
        or any(value["seals"].values())
        or not interval_contract
        or len(completed) != 8
        or _sha(completed) != EXPECTED_COMPLETED_RECEIPTS_SHA256
        or checkpoint["state"] != "bounded_complete"
        or queue["next_ordinal"] != queue["stop_ordinal_exclusive"]
        or queue["next_ordinal"] != config["stop_ordinal_exclusive"]
        or queue["service_config_sha256"] != _sha(config)
        or queue["generated_receipt"] is not None
        or queue["generation_manifest"] is not None
        or queue["formal_receipt"] is not None
        or queue["formal_evidence"] is not None
        or queue["leaderboard_rebuild_requests"] != []
        or queue["last_ranked_candidate_root"] is not None
        or any(row["formal_decision"] == "pass" for row in completed)
    ):
        raise ValueError("service execution result contract mismatch")


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
        root,
        root / args.config,
        resource_probe=probe,
        generation_executor=execute_real_generation,
        formal_executor=execute_real_formal,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
