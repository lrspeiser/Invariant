from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .grammar_v3_parameter_cell_compilation_campaign import (
    _action_density_key,
    structural_policy_gates,
)
from .grammar_v3_parameter_cell_manifest_campaign import _cell
from .persistent_parallel_search import PersistentParallelSearch
from .promotion_orchestrator import ELIGIBILITY
from .scalable_campaign_epoch_service import FUTURE_SCHEMA
from .scalable_formal_candidate_evidence_export import (
    iter_scalable_formal_candidate_evidence_records,
    validate_scalable_formal_candidate_evidence_export,
)

CONFIG_SCHEMA = "sigma-scalable-future-parameter-chunk-config-1.0"
RESULT_SCHEMA = "sigma-scalable-future-parameter-compilation-result-1.0"
STATUS_SCHEMA = "sigma-scalable-future-parameter-compilation-status-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _bound(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = _load(path)
    if "content_sha256" in binding and value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError(f"{label} content hash mismatch")
    return value


def _bound_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _validate(config: dict[str, Any]) -> None:
    required = {
        "schema_version", "execution_enabled", "campaign_id", "parent_evidence_export",
        "source_seed_manifest", "base_compilation_config", "compiler_implementation",
        "admission_adapter_descriptor", "coordinator_config",
        "resource_profile", "chunk", "budget", "data_eligibility", "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future parameter chunk config is invalid")
    if not isinstance(config["execution_enabled"], bool):
        raise TypeError("execution_enabled must be boolean")
    if config["data_eligibility"] != ELIGIBILITY or config["external_paid_llm_calls"] is not False:
        raise ValueError("future parameter chunk seals are open")
    chunk = config["chunk"]
    if set(chunk) != {"chunk_id", "start_ordinal", "cells_per_work_item"}:
        raise ValueError("future parameter chunk specification is invalid")
    budget = config["budget"]
    if set(budget) != {
        "maximum_cells", "maximum_tasks", "maximum_attempts_per_task", "maximum_wall_seconds",
        "maximum_disk_bytes", "maximum_paid_llm_spend_usd",
    }:
        raise ValueError("future parameter chunk budget is invalid")
    if (
        int(budget["maximum_cells"]) != 32
        or int(budget["maximum_tasks"]) != 4
        or int(chunk["cells_per_work_item"]) != 8
        or int(chunk["start_ordinal"]) != 256
        or not 1 <= int(budget["maximum_attempts_per_task"]) <= 3
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 1024 * 1024 <= int(budget["maximum_disk_bytes"]) <= 128 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("future parameter chunk budget is inconsistent")


def _points() -> dict[str, list[dict[str, Any]]]:
    return {
        "AETHER_K1234_PARAMETER_CELL": [
            {
                "parameters": {"c1": "1/32", "c2": c2, "c3": c3, "c4": "1/32"},
                "rational_coordinates": {"c1": "1/32", "c2": c2, "c3": c3, "c4": "1/32"},
                "domain_contract": "bounded_coefficients_only; formal stability remains unresolved",
            }
            for c2 in ("0", "1/32", "1/16", "1/8")
            for c3 in ("-1/16", "0", "1/16", "1/8")
        ],
        "KESSENCE_G2_CONVEX": [
            {
                "parameters": {"G2": f"X_phi+({alpha})*X_phi^2", "X_domain": f"0<=X_phi<={xmax}"},
                "rational_coordinates": {"alpha": alpha, "X_max": xmax},
                "domain_contract": "G2_X>=1 and G2_X+2XG2_XX>=1 on the declared cell",
            }
            for alpha in ("1/8", "1/4")
            for xmax in ("1/64", "3/64", "5/64", "7/64")
        ],
        "CUBIC_HORNDESKI_G3_WEAK_CELL": [
            {
                "parameters": {"G2": "X_phi", "G3": f"({beta})*X_phi", "jet_domain": f"dimensionless derivative ratios<={beta}"},
                "rational_coordinates": {"beta": beta, "jet_max": beta},
                "domain_contract": "weak derivative cell only; common-cone proof remains unresolved",
            }
            for beta in ("33/4000", "17/2000", "7/800", "9/1000")
        ],
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": [
            {
                "parameters": {"G2": "X_phi", "G4": "1/2+(1/100)*phi^2", "phi_domain": f"abs(phi)<={phi}"},
                "rational_coordinates": {"xi": "1/100", "phi_max": phi},
                "domain_contract": "G4>=1/2 locally; global lapse and energy are not inferred",
            }
            for phi in ("1/64", "3/64", "5/64", "7/64")
        ],
    }


def build_future_parameter_manifest_chunk(config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    _validate(config)
    root = Path(root).resolve()
    parent = _bound(root, config["parent_evidence_export"], "parent evidence export")
    _bound_file(root, config["compiler_implementation"], "future compiler implementation")
    adapter = _bound(root, config["admission_adapter_descriptor"], "admission adapter")
    adapter_body = {key: value for key, value in adapter.items() if key != "content_sha256"}
    if (
        adapter.get("content_sha256") != _sha(adapter_body)
        or adapter.get("parent_epoch_content_sha256") != parent["content_sha256"]
        or adapter.get("task_type") != "reviewed_future_manifest_chunk_admission"
        or adapter.get("next_task_type") != "reviewed_future_candidate_compilation"
        or adapter.get("data_eligibility") != ELIGIBILITY
        or adapter.get("external_paid_llm_calls") is not False
    ):
        raise ValueError("reviewed future admission adapter binding is invalid")
    _bound_file(
        root,
        {"path": adapter["callback_source_path"], "file_sha256": adapter["callback_source_file_sha256"]},
        "reviewed future admission callback",
    )
    validate_scalable_formal_candidate_evidence_export(parent)
    source = _bound(root, config["source_seed_manifest"], "source seed manifest")
    families = {
        family["family_id"]: family
        for family in source["typed_family_seeds"]
        if family["enabled_for_generation"]
    }
    cells = []
    ordinal = int(config["chunk"]["start_ordinal"])
    for family_id in sorted(_points()):
        family = families[family_id]
        for family_index, point in enumerate(_points()[family_id]):
            for value in point["rational_coordinates"].values():
                Fraction(value)
            cells.append(_cell(ordinal, family_index, family, point, {"content_sha256": source["content_sha256"]}))
            ordinal += 1
    if len(cells) != 32 or len({cell["parameter_cell_id"] for cell in cells}) != 32:
        raise ValueError("future parameter cells are not an exact disjoint 32-cell chunk")
    body = {
        "schema_version": FUTURE_SCHEMA,
        "chunk_id": config["chunk"]["chunk_id"],
        "parent_epoch_content_sha256": parent["content_sha256"],
        "range": {"start": 256, "stop": 288},
        "parameter_cells": cells,
        "parameter_cell_registry_root_sha256": _sha([[c["parameter_cell_id"], c["parameter_cell_lineage_sha256"]] for c in cells]),
        "family_cell_counts": dict(sorted(Counter(c["family_id"] for c in cells).items())),
        "formal_evaluation_performed": False,
        "data_eligibility": dict(ELIGIBILITY),
        "external_paid_llm_calls": False,
    }
    return {**body, "content_sha256": _sha(body)}


def publish_future_parameter_manifest_chunk(
    config: dict[str, Any], root: str | Path, target: str | Path
) -> dict[str, Any]:
    """Atomically publish once; exact replay is idempotent and divergence is refused."""
    chunk = build_future_parameter_manifest_chunk(config, root)
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _load(target) != chunk:
            raise ValueError("refusing to replace a divergent future manifest chunk")
        return chunk
    encoded = (_canonical(chunk) + "\n").encode()
    if len(encoded) > int(config["budget"]["maximum_disk_bytes"]):
        raise RuntimeError("future manifest publication exceeds disk budget")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return chunk


def compile_future_parameter_chunk(config: dict[str, Any], root: str | Path, chunk: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    parent = _bound(root, config["parent_evidence_export"], "parent evidence export")
    chunk_body = {key: value for key, value in chunk.items() if key != "content_sha256"}
    if (
        chunk.get("schema_version") != FUTURE_SCHEMA
        or chunk.get("content_sha256") != _sha(chunk_body)
        or chunk.get("parent_epoch_content_sha256") != parent["content_sha256"]
        or chunk.get("data_eligibility") != ELIGIBILITY
        or chunk.get("external_paid_llm_calls") is not False
        or len(chunk.get("parameter_cells", [])) != 32
    ):
        raise ValueError("future parameter manifest chunk validation failed")
    source = _bound(root, config["source_seed_manifest"], "source seed manifest")
    base = _bound(root, config["base_compilation_config"], "base compilation config")
    field_contract = _bound(root, base["field_contract"], "field contract")
    action_policy = _bound(root, base["action_policy"], "action policy")
    existing = {record["candidate_id"] for record in iter_scalable_formal_candidate_evidence_records(parent)}
    families = {f["family_id"]: f for f in source["typed_family_seeds"] if f["enabled_for_generation"]}
    receipts = []
    local_representatives: dict[str, str] = {}
    manifest_binding = {
        "future_manifest_chunk_content_sha256": chunk["content_sha256"],
        "parameter_cell_registry_root_sha256": chunk["parameter_cell_registry_root_sha256"],
    }
    for cell in chunk["parameter_cells"]:
        family = families[cell["family_id"]]
        pseudo = {
            "seed_id": cell["parameter_cell_id"], "seed_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"], "family_lineage_sha256": cell["family_lineage_sha256"],
            "theory_contract": cell["theory_contract"], "operator_atoms": cell["operator_atoms"],
            "parameters": cell["parameters"],
        }
        action = _compile_action_ir(pseudo, family, manifest_binding)
        gates = structural_policy_gates(action, cell, family, field_contract, base["action_policy"], action_policy, int(base["finite_budget"]["maximum_action_terms"]))
        if not all(gates.values()):
            raise ValueError("future reviewed cell failed structural policy")
        equivalence = _sha(_action_density_key(cell))
        candidate_id = "G3A-" + equivalence[:24]
        if candidate_id in existing:
            disposition = "deduplicated_existing_candidate"
        elif equivalence in local_representatives:
            disposition = "deduplicated_future_chunk"
        else:
            disposition = "admitted_new_candidate"
            local_representatives[equivalence] = candidate_id
        receipt = {
            "parameter_cell_id": cell["parameter_cell_id"], "parameter_cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            "family_id": cell["family_id"], "typed_action_ir_sha256": action["content_sha256"],
            "action_density_equivalence_sha256": equivalence, "candidate_id": candidate_id,
            "disposition": disposition, "structural_gate_root_sha256": _sha(gates),
            "decision": "pass" if disposition == "admitted_new_candidate" else "deduplicated",
            "data_eligibility": dict(ELIGIBILITY),
        }
        receipts.append({**receipt, "content_sha256": _sha(receipt)})
    body = {
        "schema_version": RESULT_SCHEMA,
        "future_chunk_content_sha256": chunk["content_sha256"],
        "parent_evidence_export_content_sha256": parent["content_sha256"],
        "input_cell_count": len(receipts),
        "disposition_counts": dict(sorted(Counter(r["disposition"] for r in receipts).items())),
        "family_counts": dict(sorted(Counter(r["family_id"] for r in receipts).items())),
        "receipt_registry_root_sha256": _sha([[r["parameter_cell_id"], r["content_sha256"]] for r in receipts]),
        "receipts": receipts,
        "expensive_formal_evaluation_performed": False,
        "next_stage": "reviewed_formal_preflight_required_for_new_candidates",
        "data_eligibility": dict(ELIGIBILITY),
        "external_paid_llm_calls": False,
    }
    return {**body, "content_sha256": _sha(body)}


class ScalableFutureParameterCompilationService:
    def __init__(self, directory: str | Path, config: dict[str, Any], root: str | Path) -> None:
        _validate(config)
        self.root, self.directory, self.config = Path(root).resolve(), Path(directory).resolve(), config
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk = build_future_parameter_manifest_chunk(config, self.root)
        self.result = compile_future_parameter_chunk(config, self.root, self.chunk)
        base = _bound(self.root, config["coordinator_config"], "coordinator config")
        profile = _bound(self.root, config["resource_profile"], "resource profile")
        coordinator = json.loads(_canonical(base))
        budget = config["budget"]
        coordinator["queue"].update(maximum_pending_work=4, maximum_attempts=int(budget["maximum_attempts_per_task"]), lease_seconds=int(budget["maximum_wall_seconds"]), checkpoint_every_completions=1)
        coordinator["budget"] = {"maximum_tasks": 4, "maximum_wall_seconds": float(budget["maximum_wall_seconds"])}
        coordinator["cpu"]["maximum_workers"] = 1
        coordinator["external_paid_llm_calls"] = False
        self.coordinator = PersistentParallelSearch(self.directory / "future.sqlite", coordinator, profile)
        self.recovered_on_start = self.coordinator.recover_expired()

    def items(self) -> list[dict[str, Any]]:
        receipts = self.result["receipts"]
        return [{"ordinal": i, "chunk_content_sha256": self.chunk["content_sha256"], "receipt_root_sha256": _sha([[r["parameter_cell_id"], r["content_sha256"]] for r in receipts[i * 8:(i + 1) * 8]])} for i in range(4)]

    def enqueue(self) -> dict[str, int]:
        if not self.config["execution_enabled"]:
            raise PermissionError("future parameter compilation is disabled")
        disk_bytes = sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())
        if disk_bytes > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("future parameter compilation disk budget exhausted")
        return self.coordinator.enqueue(self.items(), lane="cpu", max_attempts=int(self.config["budget"]["maximum_attempts_per_task"]))

    def run_ready(self, worker_id: str = "future-parameter-compiler") -> int:
        if not self.config["execution_enabled"]:
            raise PermissionError("future parameter compilation is disabled")
        completed = 0
        while (lease := self.coordinator.claim("cpu", worker_id)) is not None:
            expected = self.items()[lease.ordinal]
            if lease.payload != expected:
                raise ValueError("future compilation lease payload mismatch")
            result = {**expected, "decision": "attested", "result_sha256": _sha(expected)}
            if not self.coordinator.finish(lease, worker_id, result):
                raise RuntimeError("future compilation lease lost")
            completed += 1
        self.coordinator.checkpoint()
        return completed

    def status(self) -> dict[str, Any]:
        telemetry = self.coordinator.telemetry()
        body = {
            "schema_version": STATUS_SCHEMA, "campaign_id": self.config["campaign_id"],
            "immutable_config_sha256": _sha(self.config), "future_chunk_content_sha256": self.chunk["content_sha256"],
            "compilation_result_content_sha256": self.result["content_sha256"],
            "receipt_registry_root_sha256": self.result["receipt_registry_root_sha256"],
            "input_cell_count": 32, "disposition_counts": self.result["disposition_counts"],
            "queue_counts": telemetry["counts"], "recovered_on_start": self.recovered_on_start,
            "next_blocker": "reviewed_formal_preflight_not_run_for_new_candidates",
            "data_eligibility": dict(ELIGIBILITY), "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}
