from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from .grammar_v3_seed_compilation_callback import (
    CONFIG_FILE_SHA256,
    RESULT_CONTENT_SHA256,
    RESULT_FILE_SHA256,
    _reviewed_campaign,
)
from .grammar_v3_seed_execution import GrammarV3SeedExecution
from .persistent_parallel_search import PersistentParallelSearch
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-parameter-cell-execution-config-1.0"
CELL_SCHEMA = "sigma-grammar-v3-parameter-cell-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-parameter-cell-execution-status-1.0"

STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS grammar_v3_parameter_cell_adapter (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  execution_config_sha256 TEXT NOT NULL,
  parameter_cell_manifest_root_sha256 TEXT NOT NULL,
  callback_attestation_root_sha256 TEXT NOT NULL
);
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _validate_execution_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "source_manifest",
        "reviewed_callback",
        "range",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 parameter-cell execution config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 parameter-cell config eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("grammar-v3 parameter-cell config enabled paid LLM calls")
    budget = config.get("budget", {})
    if (
        set(budget) != {
            "maximum_tasks",
            "maximum_wall_seconds",
            "maximum_database_bytes",
        }
        or not 1 <= int(budget["maximum_tasks"]) <= 6
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 4096 <= int(budget["maximum_database_bytes"]) <= 64 * 1024 * 1024
    ):
        raise ValueError("grammar-v3 parameter-cell budget is invalid or unbounded")
    selected = config.get("range", {})
    if set(selected) != {"start", "stop"}:
        raise ValueError("grammar-v3 parameter-cell range fields are invalid")
    start, stop = int(selected["start"]), int(selected["stop"])
    if not 0 <= start < stop <= 6 or stop - start != int(budget["maximum_tasks"]):
        raise ValueError("grammar-v3 parameter-cell range exceeds its finite six-cell manifest")


def iter_parameter_cell_range(
    manifest: dict[str, Any], config: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    _validate_execution_config(config)
    hook = manifest.get("scalable_generator_hook", {})
    seeds = hook.get("concrete_seeds")
    if not isinstance(seeds, list) or len(seeds) != 6 or hook.get("concrete_seed_count") != 6:
        raise ValueError("grammar-v3 source does not contain the exact finite six-cell manifest")
    if manifest.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 source manifest eligibility is not fail-closed")
    start, stop = int(config["range"]["start"]), int(config["range"]["stop"])
    manifest_content = str(manifest.get("content_sha256"))
    for cell_index in range(start, stop):
        seed = seeds[cell_index]
        cell_body = {
            "schema_version": CELL_SCHEMA,
            "cell_index": cell_index,
            "source_manifest_content_sha256": manifest_content,
            "seed_id": seed["seed_id"],
            "seed_lineage_sha256": seed["seed_lineage_sha256"],
            "family_id": seed["family_id"],
            "family_lineage_sha256": seed["family_lineage_sha256"],
            "parameter_index": seed["parameter_index"],
            "parameters": seed["parameters"],
            "data_eligibility": dict(ELIGIBILITY),
        }
        lineage = _sha(cell_body)
        yield {
            **cell_body,
            "parameter_cell_id": "G3C-" + lineage[:24],
            "parameter_cell_lineage_sha256": lineage,
        }


@lru_cache(maxsize=1)
def _attest_reviewed_campaign_once_per_process() -> dict[str, Any]:
    records = _reviewed_campaign()
    record_lineage = [
        {
            "seed_id": seed_id,
            "decision": record["decision"],
            "binding_sha256": record["provenance"]["binding_sha256"],
        }
        for seed_id, record in sorted(records.items())
    ]
    if len(record_lineage) != 6 or Counter(
        record["decision"] for record in records.values()
    ) != {"blocked": 6}:
        raise ValueError("reviewed grammar-v3 worker attestation outcome changed")
    body = {
        "scope": "one immutable exact campaign rebuild per Python worker process",
        "campaign_config_file_sha256": CONFIG_FILE_SHA256,
        "campaign_result_file_sha256": RESULT_FILE_SHA256,
        "campaign_result_content_sha256": RESULT_CONTENT_SHA256,
        "candidate_record_root_sha256": _sha(record_lineage),
        "candidate_count": 6,
        "decision_counts": {"blocked": 6},
        "data_eligibility": dict(ELIGIBILITY),
    }
    return {**body, "content_sha256": _sha(body)}


class GrammarV3ParameterCellExecution:
    """Finite range producer over the six reviewed grammar-v3 parameter cells."""

    def __init__(
        self,
        coordinator: PersistentParallelSearch,
        execution_config: dict[str, Any],
        manifest_path: str | Path,
        *,
        callback_descriptor: dict[str, Any] | None,
    ) -> None:
        _validate_execution_config(execution_config)
        self.coordinator = coordinator
        self.config = execution_config
        source = execution_config["source_manifest"]
        if (
            coordinator.config.get("external_paid_llm_calls") is not False
            or int(coordinator.config["budget"]["maximum_tasks"])
            != int(execution_config["budget"]["maximum_tasks"])
            or float(coordinator.config["budget"]["maximum_wall_seconds"])
            > float(execution_config["budget"]["maximum_wall_seconds"])
            or int(coordinator.config["queue"]["maximum_pending_work"])
            < int(execution_config["budget"]["maximum_tasks"])
        ):
            raise ValueError("coordinator does not preserve parameter-cell execution bounds")
        self.seed_adapter = GrammarV3SeedExecution(
            coordinator,
            manifest_path,
            expected_manifest_file_sha256=source["file_sha256"],
            expected_manifest_content_sha256=source["content_sha256"],
            callback_descriptor=callback_descriptor,
        )
        self.recovered_on_start = dict(self.seed_adapter.recovered_on_start)
        self.cells = list(iter_parameter_cell_range(self.seed_adapter.manifest, execution_config))
        self.cell_seed_ids = [cell["seed_id"] for cell in self.cells]
        self.parameter_cell_manifest_root_sha256 = _sha(self.cells)
        if callback_descriptor is None:
            attestation = {
                "state": "reviewed_candidate_compiler_formal_callback_missing",
                "data_eligibility": dict(ELIGIBILITY),
            }
            self.callback_attestation = {**attestation, "content_sha256": _sha(attestation)}
        else:
            reviewed = execution_config["reviewed_callback"]
            if (
                callback_descriptor.get("artifact_sha256") != reviewed["artifact_sha256"]
                or self.seed_adapter.callback_binding_sha256
                != reviewed["callback_binding_sha256"]
            ):
                raise ValueError("reviewed grammar-v3 callback binding changed")
            self.callback_attestation = _attest_reviewed_campaign_once_per_process()
        self._initialize_state()

    def _initialize_state(self) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "execution_config_sha256": _sha(self.config),
            "parameter_cell_manifest_root_sha256": self.parameter_cell_manifest_root_sha256,
            "callback_attestation_root_sha256": self.callback_attestation["content_sha256"],
        }
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SCHEMA)
            row = connection.execute(
                "SELECT * FROM grammar_v3_parameter_cell_adapter WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_parameter_cell_adapter VALUES (1,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to resume a changed grammar-v3 parameter-cell range")

    def _selected_work_items(self) -> list[dict[str, Any]]:
        selected = set(self.cell_seed_ids)
        items = [
            item for item in self.seed_adapter._work_items() if item["seed_id"] in selected
        ]
        if [item["seed_id"] for item in items] != self.cell_seed_ids:
            raise ValueError("parameter-cell range ordering differs from reviewed seed ordering")
        return items

    def _database_bytes(self) -> int:
        path = self.coordinator.database
        return sum(
            candidate.stat().st_size
            for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"))
            if candidate.is_file()
        )

    def _enforce_disk_budget(self) -> int:
        consumed = self._database_bytes()
        if consumed > int(self.config["budget"]["maximum_database_bytes"]):
            raise RuntimeError("grammar-v3 parameter-cell database disk budget exhausted")
        return consumed

    def enqueue(self) -> dict[str, Any]:
        self._enforce_disk_budget()
        admitted = self.coordinator.enqueue(
            self._selected_work_items(), lane="cpu", max_attempts=3
        )
        checkpoint = self.coordinator.checkpoint()
        return {
            **admitted,
            "requested": len(self.cells),
            "parameter_cell_manifest_root_sha256": self.parameter_cell_manifest_root_sha256,
            "checkpoint_sha256": checkpoint["content_sha256"],
        }

    def run_bounded(self, *, worker_id: str = "grammar-v3-parameter-cell") -> dict[str, Any]:
        started = time.monotonic()
        recovered_now = self.coordinator.recover_expired()
        recovered = {
            key: int(self.recovered_on_start[key]) + int(recovered_now[key])
            for key in ("recovered", "failed")
        }
        self.recovered_on_start = {"recovered": 0, "failed": 0}
        executed = 0
        maximum_tasks = int(self.config["budget"]["maximum_tasks"])
        maximum_wall = float(self.config["budget"]["maximum_wall_seconds"])
        for _ in range(maximum_tasks):
            if time.monotonic() - started > maximum_wall:
                raise TimeoutError("grammar-v3 parameter-cell worker wall budget exhausted")
            self._enforce_disk_budget()
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self.seed_adapter.execute_lease(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("grammar-v3 parameter-cell lease was lost")
            except Exception as error:
                self.coordinator.fail(lease, worker_id, f"{type(error).__name__}: {error}")
                raise
            executed += 1
        checkpoint = self.coordinator.checkpoint()
        return {
            "executed": executed,
            "recovered": recovered,
            "checkpoint_sha256": checkpoint["content_sha256"],
            "status": self.status(),
        }

    def status(self) -> dict[str, Any]:
        status = self.seed_adapter.status()
        records = status["work_records"]
        if any(record["seed_id"] not in self.cell_seed_ids for record in records):
            raise ValueError("coordinator contains work outside the selected parameter-cell range")
        body = {
            "schema_version": STATUS_SCHEMA,
            "range": dict(self.config["range"]),
            "parameter_cell_count": len(self.cells),
            "parameter_cell_manifest_root_sha256": self.parameter_cell_manifest_root_sha256,
            "callback_attestation": self.callback_attestation,
            "callback_registry_root_sha256": status["callback_registry_root_sha256"],
            "work_state_counts": status["work_state_counts"],
            "decision_counts": status["decision_counts"],
            "work_records": records,
            "work_records_root_sha256": status["work_records_root_sha256"],
            "checkpoint_sequence": status["checkpoint_sequence"],
            "recovered_leases": status["recovered_leases"],
            "database_bytes": self._enforce_disk_budget(),
            "budget": dict(self.config["budget"]),
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "next_scaling_hook": (
                "a new hash-reviewed campaign result must register additional parameter "
                "cells before this finite range may expand beyond six"
            ),
        }
        return {**body, "content_sha256": _sha(body)}


def load_bound_callback_descriptor(
    config: dict[str, Any], config_path: str | Path
) -> dict[str, Any]:
    reviewed = config["reviewed_callback"]
    path = Path(config_path).resolve().parent.parent / reviewed["descriptor_path"]
    if not path.is_file() or _file_sha(path) != reviewed["descriptor_file_sha256"]:
        raise ValueError("grammar-v3 parameter-cell callback descriptor hash mismatch")
    descriptor = _load(path)
    artifact = Path(str(descriptor["artifact_path"]))
    if not artifact.is_absolute():
        artifact = (path.parent.parent / artifact).resolve()
    descriptor["artifact_path"] = str(artifact)
    return descriptor


def build_portable_parameter_cell_status(
    status: dict[str, Any], execution_config: dict[str, Any]
) -> dict[str, Any]:
    records = [
        {
            key: record[key]
            for key in (
                "work_id",
                "adapter_work_id",
                "seed_id",
                "seed_lineage_sha256",
                "ordinal",
                "coordinator_seed",
                "state",
                "attempt",
                "result_sha256",
                "output_lineage_sha256",
            )
        }
        for record in status["work_records"]
    ]
    body = {
        key: status[key]
        for key in (
            "schema_version",
            "range",
            "parameter_cell_count",
            "parameter_cell_manifest_root_sha256",
            "callback_attestation",
            "callback_registry_root_sha256",
            "work_state_counts",
            "decision_counts",
            "checkpoint_sequence",
            "recovered_leases",
            "budget",
            "observational_data_opened",
            "data_eligibility",
            "paid_llm_spend_usd",
            "next_scaling_hook",
        )
    }
    body.update(
        {
            "execution_config_sha256": _sha(execution_config),
            "work_records": records,
            "portable_work_records_root_sha256": _sha(records),
        }
    )
    return {**body, "content_sha256": _sha(body)}
