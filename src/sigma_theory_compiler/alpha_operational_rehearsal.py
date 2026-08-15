"""Bounded alpha rehearsal for recovery, budgets, and disabled accelerators."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from . import kastner_schlatter_set_indexed_gpu_scheduler_adapter as gpu_adapter
from .llm_formula_proposal_adapter import (
    AdapterConfig,
    FormulaProposalAdapter,
    ProposalRequest,
    SpendLedger,
)
from .persistent_parallel_search import PersistentParallelSearch

SCHEMA = "sigma-alpha-operational-rehearsal-1.0"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: dict[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _prepare_scratch(repo_root: Path, scratch_root: Path) -> Path:
    repo = repo_root.resolve()
    scratch = scratch_root.resolve()
    try:
        scratch.relative_to(repo / "runs")
    except ValueError:
        pass
    else:
        raise ValueError("operational rehearsal scratch may not use repository runtime")
    if scratch.exists() and any(scratch.iterdir()):
        raise ValueError("operational rehearsal scratch must be empty")
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def _scheduler_rehearsal(repo: Path, scratch: Path) -> dict[str, Any]:
    config = json.loads(
        (repo / "configs/persistent_parallel_search_5090.json").read_text(encoding="utf-8")
    )
    profile = json.loads((repo / "configs/resource_profile_5090.json").read_text(encoding="utf-8"))
    config["external_paid_llm_calls"] = False
    config["queue"] = {
        **config["queue"],
        "checkpoint_every_completions": 1,
        "lease_seconds": 1,
        "maximum_attempts": 2,
        "maximum_pending_work": 1,
    }
    config["budget"] = {"maximum_tasks": 1, "maximum_wall_seconds": 60}
    config["cpu"] = {"maximum_workers": 1}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 1,
        "gpu_workers": 0,
    }
    profile["hardware"] = {**profile["hardware"], "gpu_memory_mib": 0}

    database = scratch / "scheduler" / "rehearsal-queue.sqlite"
    coordinator = PersistentParallelSearch(database, config, profile)
    payload = {
        "ordinal": 0,
        "control_id": "alpha-owned-interruption-recovery",
        "synthetic_control_only": True,
    }
    admitted = coordinator.enqueue([payload], lane="cpu", max_attempts=2)
    first = coordinator.claim("cpu", "alpha-interrupted-owner", lease_seconds=-1)
    if admitted["accepted"] != 1 or first is None or first.attempt != 1:
        raise RuntimeError("alpha recovery control admission failed")

    resumed = PersistentParallelSearch(database, config, profile)
    recovery = resumed.recover_expired()
    second = resumed.claim("cpu", "alpha-replacement-owner")
    if recovery != {"recovered": 1, "failed": 0} or second is None:
        raise RuntimeError("alpha expired lease did not recover exactly once")
    result = {
        "control_id": payload["control_id"],
        "scientific_pass": False,
        "synthetic_control_only": True,
    }
    if second.work_id != first.work_id or second.attempt != 2:
        raise RuntimeError("alpha recovery lineage changed")
    if not resumed.finish(second, "alpha-replacement-owner", result):
        raise RuntimeError("alpha recovered lease could not finish")
    checkpoint = resumed.checkpoint()
    telemetry = resumed.telemetry()
    return {
        "checkpoint": {
            "config_sha256": checkpoint["config_sha256"],
            "sequence": checkpoint["sequence"],
            "work_state_root": checkpoint["work_state_root"],
        },
        "counts": telemetry["counts"],
        "database_name": database.name,
        "first_attempt": first.attempt,
        "recovered_attempt": second.attempt,
        "recovery": recovery,
        "synthetic_control_only": True,
        "worker_counts": {"cpu": 1, "gpu": 0},
    }


def _llm_rehearsal(repo: Path, scratch: Path) -> dict[str, Any]:
    config_path = repo / "configs/llm_formula_proposal_adapter.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = AdapterConfig.from_mapping(raw["adapter"])
    provider_calls = 0

    def forbidden_provider(_request: dict[str, Any], _secret: str) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("disabled rehearsal attempted a provider call")

    ledger = SpendLedger(scratch / "llm" / "spend-ledger.sqlite", config)
    request = ProposalRequest(
        request_id="alpha:disabled:proposal:0001",
        prompt="Bounded synthetic formula proposal rehearsal",
        prompt_template_sha256="1" * 64,
        context_packets=({"content_sha256": "2" * 64, "data_class": "formal_artifact"},),
        dsl_version="sigma-gravity-dsl-3",
        deterministic_seed=1,
        maximum_call_usd="1.000000",
    )
    decision = FormulaProposalAdapter(config, ledger, forbidden_provider).propose(request)
    telemetry = ledger.telemetry()
    if (
        decision.get("reason") != "paid_calls_disabled_by_default"
        or ledger.status(request.request_id) is not None
        or provider_calls != 0
        or telemetry["settled_usd"] != "0.000000"
    ):
        raise RuntimeError("disabled LLM rehearsal crossed its budget boundary")
    receipt = {
        "adapter_config_file_sha256": _file_sha(config_path),
        "budget": {
            "maximum_call_usd": "5.000000",
            "maximum_total_usd": telemetry["maximum_total_usd"],
            "reserved_usd": "0.000000",
            "settled_usd": telemetry["settled_usd"],
        },
        "decision": "blocked",
        "network_calls": 0,
        "provider_calls": provider_calls,
        "reason": decision["reason"],
        "request_rows": 0,
        "secrets_persisted": False,
    }
    del ledger
    gc.collect()
    return receipt


def _gpu_control(repo: Path) -> dict[str, Any]:
    config_path = repo / "configs/kastner_schlatter_set_indexed_gpu_scheduler_adapter.json"
    readiness_path = (
        repo / "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-adapter-readiness.json"
    )
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if readiness.get("content_sha256") != gpu_adapter._content_sha(readiness):
        raise ValueError("GPU readiness content hash mismatch")
    validation = "passed"
    try:
        gpu_adapter.validate_readiness(readiness, config_path)
    except (KeyError, TypeError, ValueError):
        validation = "blocked_binding_drift"
    return {
        "adapter_config_file_sha256": _file_sha(config_path),
        "configured_cpu_workers": readiness["scheduler_contract"]["cpu_worker_count"],
        "configured_gpu_owners": readiness["scheduler_contract"]["gpu_owner_count"],
        "execution_started": False,
        "gpu_reserved": False,
        "nvml_sampled": False,
        "readiness_file_sha256": _file_sha(readiness_path),
        "runtime_created": False,
        "validation": validation,
    }


def run_operational_rehearsal(repo_root: str | Path, scratch_root: str | Path) -> dict[str, Any]:
    """Run only owned synthetic recovery and zero-spend control paths."""

    repo = Path(repo_root).resolve()
    scratch = _prepare_scratch(repo, Path(scratch_root))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "gpu_control": _gpu_control(repo),
        "llm_control": _llm_rehearsal(repo, scratch),
        "scheduler_control": _scheduler_rehearsal(repo, scratch),
        "claims": {
            "network_calls": 0,
            "observations_opened": False,
            "promotion": False,
            "scientific_pass": False,
            "synthetic_control_only": True,
        },
    }
    receipt["content_sha256"] = _content_sha(receipt)
    validate_operational_receipt(receipt)
    return receipt


def validate_operational_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != SCHEMA or receipt.get("content_sha256") != _content_sha(
        receipt
    ):
        raise ValueError("operational rehearsal receipt hash or schema mismatch")
    if receipt.get("claims") != {
        "network_calls": 0,
        "observations_opened": False,
        "promotion": False,
        "scientific_pass": False,
        "synthetic_control_only": True,
    }:
        raise ValueError("operational rehearsal claims changed")
    scheduler = receipt.get("scheduler_control", {})
    if (
        scheduler.get("recovery") != {"recovered": 1, "failed": 0}
        or scheduler.get("first_attempt") != 1
        or scheduler.get("recovered_attempt") != 2
        or scheduler.get("counts") != {"succeeded": 1}
        or scheduler.get("worker_counts") != {"cpu": 1, "gpu": 0}
    ):
        raise ValueError("operational scheduler recovery receipt changed")
    llm = receipt.get("llm_control", {})
    if (
        llm.get("network_calls") != 0
        or llm.get("provider_calls") != 0
        or llm.get("request_rows") != 0
        or llm.get("budget", {}).get("settled_usd") != "0.000000"
    ):
        raise ValueError("operational LLM budget receipt changed")
    gpu = receipt.get("gpu_control", {})
    if any(
        gpu.get(key) is not False
        for key in ("execution_started", "gpu_reserved", "nvml_sampled", "runtime_created")
    ):
        raise ValueError("operational GPU control crossed execution boundary")
