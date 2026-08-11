from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .grammar_v3_parameter_cell_execution import (
    GrammarV3ParameterCellExecution,
    build_portable_parameter_cell_status,
    iter_parameter_cell_range,
    load_bound_callback_descriptor,
)
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-parameter-cell-expansion-service-config-1.0"
CHUNK_SCHEMA = "sigma-grammar-v3-parameter-cell-chunk-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-parameter-cell-chunk-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-parameter-cell-expansion-service-status-1.0"

STATE_SQL = """
CREATE TABLE IF NOT EXISTS grammar_v3_parameter_cell_expansion_service (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  immutable_config_sha256 TEXT NOT NULL,
  source_manifest_content_sha256 TEXT NOT NULL,
  chunk_registry_root_sha256 TEXT NOT NULL,
  callback_registry_root_sha256 TEXT NOT NULL
);
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _validate(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "execution_enabled",
        "parameter_cell_execution_config",
        "coordinator_config",
        "resource_profile",
        "chunking",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 expansion service config is invalid")
    if not isinstance(config.get("execution_enabled"), bool):
        raise TypeError("grammar-v3 expansion execution_enabled must be boolean")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 expansion data eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("grammar-v3 expansion enabled paid LLM calls")
    chunking = config.get("chunking", {})
    if set(chunking) != {"range_start", "range_stop", "cells_per_chunk"}:
        raise ValueError("grammar-v3 expansion chunking fields are invalid")
    start, stop, width = (
        int(chunking["range_start"]),
        int(chunking["range_stop"]),
        int(chunking["cells_per_chunk"]),
    )
    if not 0 <= start < stop <= 6 or not 1 <= width <= 6:
        raise ValueError("grammar-v3 expansion range exceeds the reviewed six-cell manifest")
    chunks = (stop - start + width - 1) // width
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_cells",
        "maximum_chunks",
        "maximum_attempts_per_chunk",
        "maximum_wall_seconds",
        "maximum_disk_bytes",
        "maximum_paid_llm_spend_usd",
    }:
        raise ValueError("grammar-v3 expansion budget fields are invalid")
    if (
        int(budget["maximum_cells"]) != stop - start
        or int(budget["maximum_chunks"]) != chunks
        or not 1 <= int(budget["maximum_attempts_per_chunk"]) <= 3
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 1024 * 1024 <= int(budget["maximum_disk_bytes"]) <= 128 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("grammar-v3 expansion budget is inconsistent or unbounded")
    for key in (
        "parameter_cell_execution_config",
        "coordinator_config",
        "resource_profile",
    ):
        binding = config.get(key, {})
        if set(binding) != {"path", "file_sha256"}:
            raise ValueError(f"grammar-v3 expansion {key} binding is invalid")


class GrammarV3ParameterCellExpansionService:
    """Durable bounded chunk scheduler over the reviewed finite grammar-v3 iterator."""

    def __init__(
        self,
        service_directory: str | Path,
        config: dict[str, Any],
        repo_root: str | Path,
        *,
        register_reviewed_callback: bool = True,
    ) -> None:
        _validate(config)
        self.directory = Path(service_directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.execution_config_path = self._bound_path("parameter_cell_execution_config")
        self.coordinator_config_path = self._bound_path("coordinator_config")
        self.resource_profile_path = self._bound_path("resource_profile")
        self.base_execution = _load(self.execution_config_path)
        self.base_coordinator = _load(self.coordinator_config_path)
        self.resource_profile = _load(self.resource_profile_path)
        manifest_binding = self.base_execution["source_manifest"]
        self.manifest_path = (self.repo_root / manifest_binding["path"]).resolve()
        if _file_sha(self.manifest_path) != manifest_binding["file_sha256"]:
            raise ValueError("grammar-v3 expansion source manifest file hash mismatch")
        self.manifest = _load(self.manifest_path)
        if self.manifest.get("content_sha256") != manifest_binding["content_sha256"]:
            raise ValueError("grammar-v3 expansion source manifest content binding mismatch")
        self.callback_descriptor = (
            self._load_callback_if_available() if register_reviewed_callback else None
        )
        self.callback_registry_root_sha256 = _sha(
            {
                "state": (
                    "reviewed_callback_bound"
                    if self.callback_descriptor is not None
                    else "reviewed_candidate_compiler_formal_callback_missing"
                ),
                "callback_binding_sha256": self.base_execution["reviewed_callback"][
                    "callback_binding_sha256"
                ]
                if self.callback_descriptor is not None
                else None,
            }
        )
        self.chunks = self._chunks()
        self.chunk_registry_root_sha256 = _sha(self.chunks)
        self.coordinator = PersistentParallelSearch(
            self.directory / "expansion.sqlite",
            self._outer_coordinator_config(),
            self.resource_profile,
        )
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _bound_path(self, key: str) -> Path:
        binding = self.config[key]
        path = (self.repo_root / binding["path"]).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError("grammar-v3 expansion binding escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"grammar-v3 expansion {key} file hash mismatch")
        return path

    def _load_callback_if_available(self) -> dict[str, Any] | None:
        reviewed = self.base_execution["reviewed_callback"]
        descriptor_path = (self.repo_root / reviewed["descriptor_path"]).resolve()
        if not descriptor_path.exists():
            return None
        if _file_sha(descriptor_path) != reviewed["descriptor_file_sha256"]:
            raise ValueError("grammar-v3 expansion reviewed callback descriptor was tampered")
        return load_bound_callback_descriptor(self.base_execution, self.execution_config_path)

    def _all_cells(self) -> list[dict[str, Any]]:
        execution = json.loads(_canonical(self.base_execution))
        execution["range"] = {
            "start": int(self.config["chunking"]["range_start"]),
            "stop": int(self.config["chunking"]["range_stop"]),
        }
        execution["budget"]["maximum_tasks"] = (
            execution["range"]["stop"] - execution["range"]["start"]
        )
        return list(iter_parameter_cell_range(self.manifest, execution))

    def _chunks(self) -> list[dict[str, Any]]:
        cells = self._all_cells()
        width = int(self.config["chunking"]["cells_per_chunk"])
        range_start = int(self.config["chunking"]["range_start"])
        chunks = []
        for offset in range(0, len(cells), width):
            selected = cells[offset : offset + width]
            start = range_start + offset
            stop = start + len(selected)
            identities = [
                {
                    "parameter_cell_id": cell["parameter_cell_id"],
                    "parameter_cell_lineage_sha256": cell[
                        "parameter_cell_lineage_sha256"
                    ],
                    "seed_id": cell["seed_id"],
                    "seed_lineage_sha256": cell["seed_lineage_sha256"],
                }
                for cell in selected
            ]
            body = {
                "schema_version": CHUNK_SCHEMA,
                "ordinal": len(chunks),
                "range": {"start": start, "stop": stop},
                "source_manifest_content_sha256": self.manifest["content_sha256"],
                "parameter_cells": identities,
                "data_eligibility": ELIGIBILITY,
                "external_paid_llm_calls": False,
            }
            lineage = _sha(body)
            chunks.append(
                {
                    **body,
                    "chunk_id": "G3PCX-" + lineage[:24],
                    "chunk_lineage_sha256": lineage,
                }
            )
        flat = [cell["parameter_cell_id"] for chunk in chunks for cell in chunk["parameter_cells"]]
        if len(flat) != len(set(flat)) or len(flat) != int(self.config["budget"]["maximum_cells"]):
            raise ValueError("grammar-v3 expansion chunks overlap or do not cover the bounded range")
        return chunks

    def _outer_coordinator_config(self) -> dict[str, Any]:
        config = json.loads(_canonical(self.base_coordinator))
        chunks = int(self.config["budget"]["maximum_chunks"])
        config["queue"].update(
            maximum_pending_work=chunks,
            maximum_attempts=int(self.config["budget"]["maximum_attempts_per_chunk"]),
            lease_seconds=int(self.config["budget"]["maximum_wall_seconds"]),
            checkpoint_every_completions=1,
        )
        config["budget"] = {
            "maximum_tasks": chunks,
            "maximum_wall_seconds": float(self.config["budget"]["maximum_wall_seconds"]),
        }
        config["cpu"]["maximum_workers"] = min(chunks, 2)
        config["external_paid_llm_calls"] = False
        return config

    def _initialize_state(self) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "immutable_config_sha256": _sha(self.config),
            "source_manifest_content_sha256": self.manifest["content_sha256"],
            "chunk_registry_root_sha256": self.chunk_registry_root_sha256,
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
        }
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SQL)
            row = connection.execute(
                "SELECT * FROM grammar_v3_parameter_cell_expansion_service WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_parameter_cell_expansion_service VALUES (1,?,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to resume changed grammar-v3 expansion service")
            for row in connection.execute("SELECT payload_json FROM work"):
                if json.loads(row[0]).get("schema_version") != CHUNK_SCHEMA:
                    raise ValueError("grammar-v3 expansion requires a dedicated coordinator DB")

    def _disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def _enforce_budget(self, started: float | None = None) -> None:
        if self._disk_bytes() > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("grammar-v3 expansion disk budget exhausted")
        if started is not None and time.monotonic() - started > float(
            self.config["budget"]["maximum_wall_seconds"]
        ):
            raise TimeoutError("grammar-v3 expansion wall budget exhausted")

    def enqueue(self) -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 parameter-cell expansion is disabled by config")
        self._enforce_budget()
        admission = self.coordinator.enqueue(
            self.chunks,
            lane="cpu",
            max_attempts=int(self.config["budget"]["maximum_attempts_per_chunk"]),
        )
        checkpoint = self.coordinator.checkpoint()
        return {**admission, "requested": len(self.chunks), "checkpoint_sha256": checkpoint["content_sha256"]}

    def _child_config(self, chunk: dict[str, Any]) -> dict[str, Any]:
        config = json.loads(_canonical(self.base_execution))
        config["range"] = dict(chunk["range"])
        count = chunk["range"]["stop"] - chunk["range"]["start"]
        config["budget"] = {
            "maximum_tasks": count,
            "maximum_wall_seconds": min(
                float(config["budget"]["maximum_wall_seconds"]),
                float(self.config["budget"]["maximum_wall_seconds"]),
            ),
            "maximum_database_bytes": min(
                int(config["budget"]["maximum_database_bytes"]),
                int(self.config["budget"]["maximum_disk_bytes"]),
            ),
        }
        return config

    def _child_coordinator_config(self, child: dict[str, Any]) -> dict[str, Any]:
        config = json.loads(_canonical(self.base_coordinator))
        count = int(child["budget"]["maximum_tasks"])
        config["queue"].update(
            maximum_pending_work=count,
            maximum_attempts=3,
            lease_seconds=30,
            checkpoint_every_completions=1,
        )
        config["budget"] = {
            "maximum_tasks": count,
            "maximum_wall_seconds": child["budget"]["maximum_wall_seconds"],
        }
        config["cpu"]["maximum_workers"] = min(count, 2)
        config["external_paid_llm_calls"] = False
        return config

    def _execute_chunk(self, lease: WorkLease) -> dict[str, Any]:
        chunk = lease.payload
        if lease.ordinal >= len(self.chunks) or chunk != self.chunks[lease.ordinal]:
            raise ValueError("leased grammar-v3 expansion chunk identity changed")
        child_config = self._child_config(chunk)
        child_db = self.directory / "chunks" / f"{chunk['chunk_id']}.sqlite"
        child_db.parent.mkdir(parents=True, exist_ok=True)
        child_coordinator = PersistentParallelSearch(
            child_db,
            self._child_coordinator_config(child_config),
            self.resource_profile,
        )
        adapter = GrammarV3ParameterCellExecution(
            child_coordinator,
            child_config,
            self.manifest_path,
            callback_descriptor=self.callback_descriptor,
        )
        admission = adapter.enqueue()
        run = adapter.run_bounded(worker_id=f"{chunk['chunk_id']}-reviewed")
        portable = build_portable_parameter_cell_status(run["status"], child_config)
        if portable["parameter_cell_count"] != len(chunk["parameter_cells"]):
            raise ValueError("reviewed queue returned incomplete grammar-v3 chunk")
        body = {
            "schema_version": RESULT_SCHEMA,
            "chunk_id": chunk["chunk_id"],
            "chunk_lineage_sha256": chunk["chunk_lineage_sha256"],
            "admission": {
                key: admission[key]
                for key in (
                    "accepted",
                    "duplicate",
                    "backpressured",
                    "budget_rejected",
                    "requested",
                    "parameter_cell_manifest_root_sha256",
                )
            },
            "reviewed_queue_status": portable,
            "reviewed_queue_status_sha256": portable["content_sha256"],
            "decision_counts": portable["decision_counts"],
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "observational_data_opened": False,
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self, *, worker_id: str = "grammar-v3-expansion") -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 parameter-cell expansion is disabled by config")
        started = time.monotonic()
        current = self.coordinator.recover_expired()
        recovered = {
            key: int(self.recovered_on_start[key]) + int(current[key])
            for key in ("recovered", "failed")
        }
        self.recovered_on_start = {"recovered": 0, "failed": 0}
        executed = 0
        for _ in range(int(self.config["budget"]["maximum_chunks"])):
            self._enforce_budget(started)
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self._execute_chunk(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("grammar-v3 expansion chunk lease was lost")
            except Exception as error:
                self.coordinator.fail(lease, worker_id, f"{type(error).__name__}: {error}")
                raise
            executed += 1
        checkpoint = self.coordinator.checkpoint()
        return {
            "executed_chunks": executed,
            "recovered": recovered,
            "checkpoint_sha256": checkpoint["content_sha256"],
            "status": self.status(),
        }

    def status(self) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        records = []
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT ordinal,payload_json,state,attempt,result_json,error_text FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            if int(row["ordinal"]) >= len(self.chunks) or payload != self.chunks[int(row["ordinal"])]:
                raise ValueError("stored grammar-v3 expansion chunk was tampered")
            result_sha = None
            if row["result_json"]:
                result = json.loads(row["result_json"])
                body = {key: value for key, value in result.items() if key != "content_sha256"}
                if result.get("content_sha256") != _sha(body):
                    raise ValueError("stored grammar-v3 expansion result hash mismatch")
                decisions.update(result["decision_counts"])
                result_sha = result["content_sha256"]
            counts[str(row["state"])] += 1
            records.append(
                {
                    "chunk_id": payload["chunk_id"],
                    "chunk_lineage_sha256": payload["chunk_lineage_sha256"],
                    "range": payload["range"],
                    "state": row["state"],
                    "attempt": int(row["attempt"]),
                    "result_sha256": result_sha,
                    "error_text": row["error_text"],
                }
            )
        body = {
            "schema_version": STATUS_SCHEMA,
            "execution_enabled": self.config["execution_enabled"],
            "source_manifest_content_sha256": self.manifest["content_sha256"],
            "chunk_count": len(self.chunks),
            "parameter_cell_count": sum(len(c["parameter_cells"]) for c in self.chunks),
            "chunk_registry_root_sha256": self.chunk_registry_root_sha256,
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
            "work_state_counts": dict(sorted(counts.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "chunk_records": records,
            "chunk_records_root_sha256": _sha(records),
            "checkpoint_sequence": self.coordinator.telemetry()["checkpoint_sequence"],
            "disk_bytes": self._disk_bytes(),
            "budget": self.config["budget"],
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "scientific_scope": (
                "execution scaling only; no cells beyond the reviewed manifest are inferred"
            ),
        }
        return {**body, "content_sha256": _sha(body)}


def portable_status(status: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in status.items()
        if key not in {"content_sha256", "disk_bytes", "checkpoint_sequence"}
    }
    return {**body, "content_sha256": _sha(body)}
