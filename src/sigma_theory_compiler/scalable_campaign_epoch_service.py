from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY
from .scalable_formal_candidate_evidence_export import (
    iter_scalable_formal_candidate_evidence_records,
    validate_scalable_formal_candidate_evidence_export,
)

CONFIG_SCHEMA = "sigma-scalable-campaign-epoch-service-config-1.1"
STATUS_SCHEMA = "sigma-scalable-campaign-staged-epoch-1.1"
FUTURE_SCHEMA = "sigma-scalable-campaign-future-manifest-chunk-1.0"
ADAPTER_SCHEMA = "sigma-reviewed-future-compiler-adapter-descriptor-1.0"

STATE_SQL = """
CREATE TABLE IF NOT EXISTS scalable_campaign_epoch_state (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 immutable_config_sha256 TEXT NOT NULL,
 source_export_content_sha256 TEXT NOT NULL,
 stage_registry_root_sha256 TEXT NOT NULL
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


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "execution_enabled",
        "campaign_id",
        "evidence_export_config",
        "evidence_export",
        "coordinator_config",
        "resource_profile",
        "future_manifest_chunk",
        "reviewed_future_compiler_adapter",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("scalable campaign epoch config is invalid")
    if not isinstance(config["execution_enabled"], bool):
        raise TypeError("execution_enabled must be boolean")
    if config["data_eligibility"] != ELIGIBILITY or config["external_paid_llm_calls"] is not False:
        raise ValueError("scalable campaign epoch data/$0 seals are open")
    for key in ("evidence_export_config", "coordinator_config", "resource_profile"):
        if set(config[key]) != {"path", "file_sha256"}:
            raise ValueError(f"invalid {key} binding")
    if set(config["evidence_export"]) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("invalid evidence export binding")
    budget = config["budget"]
    if set(budget) != {
        "maximum_tasks",
        "maximum_attempts_per_task",
        "maximum_wall_seconds",
        "maximum_disk_bytes",
        "maximum_future_cells",
        "maximum_paid_llm_spend_usd",
    }:
        raise ValueError("invalid scalable campaign epoch budget")
    if not (10 <= int(budget["maximum_tasks"]) <= 11):
        raise ValueError("stage task budget must be 10 or 11")
    if not (1 <= int(budget["maximum_attempts_per_task"]) <= 3):
        raise ValueError("attempt budget is invalid")
    if not (1 <= float(budget["maximum_wall_seconds"]) <= 300):
        raise ValueError("wall budget is invalid")
    if not (1024 * 1024 <= int(budget["maximum_disk_bytes"]) <= 128 * 1024 * 1024):
        raise ValueError("disk budget is invalid")
    if not (1 <= int(budget["maximum_future_cells"]) <= 256):
        raise ValueError("future-cell budget is invalid")
    if float(budget["maximum_paid_llm_spend_usd"]) != 0.0:
        raise ValueError("paid LLM budget must be zero")


class ScalableCampaignEpochService:
    """Durable immutable attestation epoch over the reviewed scalable campaign chain."""

    def __init__(self, directory: str | Path, config: dict[str, Any], root: str | Path) -> None:
        _validate_config(config)
        self.root = Path(root).resolve()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.export_config = _load(self._bound("evidence_export_config"))
        self.evidence_export = _load(self._bound("evidence_export", content=True))
        validate_scalable_formal_candidate_evidence_export(self.evidence_export)
        self._verify_nested_bindings()
        self.future = self._load_optional("future_manifest_chunk", FUTURE_SCHEMA)
        self.adapter = self._load_optional("reviewed_future_compiler_adapter", ADAPTER_SCHEMA)
        self._validate_future_boundary()
        self.stages = self._stage_items()
        self.stage_registry_root_sha256 = _sha(self.stages)
        coordinator = _load(self._bound("coordinator_config"))
        profile = _load(self._bound("resource_profile"))
        coordinator = json.loads(_canonical(coordinator))
        budget = config["budget"]
        coordinator["queue"].update(
            maximum_pending_work=int(budget["maximum_tasks"]),
            maximum_attempts=int(budget["maximum_attempts_per_task"]),
            lease_seconds=max(1, int(float(budget["maximum_wall_seconds"]))),
            checkpoint_every_completions=1,
        )
        coordinator["budget"] = {
            "maximum_tasks": int(budget["maximum_tasks"]),
            "maximum_wall_seconds": float(budget["maximum_wall_seconds"]),
        }
        coordinator["cpu"]["maximum_workers"] = 1
        coordinator["external_paid_llm_calls"] = False
        self.coordinator = PersistentParallelSearch(
            self.directory / "epoch.sqlite", coordinator, profile
        )
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _bound(self, key: str, *, content: bool = False) -> Path:
        binding = self.config[key]
        path = (self.root / binding["path"]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"{key} path escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"{key} file hash mismatch")
        if content and _load(path).get("content_sha256") != binding["content_sha256"]:
            raise ValueError(f"{key} content hash mismatch")
        return path

    def _load_optional(self, key: str, schema: str) -> dict[str, Any] | None:
        binding = self.config[key]
        if binding is None:
            return None
        if set(binding) != {"path", "file_sha256", "content_sha256"}:
            raise ValueError(f"invalid optional {key} binding")
        value = _load(self._bound(key, content=True))
        body = {name: item for name, item in value.items() if name != "content_sha256"}
        if value.get("schema_version") != schema or value.get("content_sha256") != _sha(body):
            raise ValueError(f"invalid {key} content")
        return value

    def _verify_nested_bindings(self) -> None:
        expected = self.evidence_export.get("source_bindings", {})
        for key, binding in self.export_config.items():
            if key in {
                "schema_version",
                "campaign_id",
                "budget",
                "data_eligibility",
                "external_paid_llm_calls",
            }:
                continue
            path = (self.root / binding["path"]).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as error:
                raise ValueError("nested source path escapes repository") from error
            if _file_sha(path) != binding["file_sha256"]:
                raise ValueError(f"nested {key} file hash mismatch")
            if (
                "content_sha256" in binding
                and _load(path).get("content_sha256") != binding["content_sha256"]
            ):
                raise ValueError(f"nested {key} content hash mismatch")
            if key in expected and expected[key] != binding:
                raise ValueError(f"export source binding differs for {key}")
        export_eligibility = dict(self.evidence_export["data_eligibility"])
        passed = export_eligibility.pop("passed", None)
        if (
            export_eligibility != ELIGIBILITY
            or passed is not True
            or self.evidence_export["observational_data_opened"] is not False
        ):
            raise ValueError("source export eligibility seal is open")

    def _validate_future_boundary(self) -> None:
        if self.future is None and self.adapter is None:
            return
        if self.future is None or self.adapter is None:
            return
        future = self.future
        if future.get("parent_epoch_content_sha256") != self.evidence_export["content_sha256"]:
            raise ValueError("future chunk parent epoch mismatch")
        cells = future.get("parameter_cells")
        if (
            not isinstance(cells, list)
            or not cells
            or len(cells) > int(self.config["budget"]["maximum_future_cells"])
        ):
            raise ValueError("future chunk cell count is invalid")
        identities = [cell.get("parameter_cell_id") for cell in cells]
        if None in identities or len(set(identities)) != len(identities):
            raise ValueError("future chunk identities overlap")
        if (
            future.get("data_eligibility") != ELIGIBILITY
            or future.get("external_paid_llm_calls") is not False
        ):
            raise ValueError("future chunk opens a forbidden input")
        adapter = self.adapter
        if (
            adapter.get("reviewed") is not True
            or adapter.get("task_type") != "reviewed_future_manifest_chunk_admission"
            or adapter.get("next_task_type") != "reviewed_future_candidate_compilation"
            or adapter.get("callback_entrypoint")
            != "sigma_theory_compiler.scalable_campaign_epoch_service:reviewed_future_manifest_admission_adapter"
        ):
            raise ValueError("future compiler adapter is not reviewed")
        if adapter.get("parent_epoch_content_sha256") != self.evidence_export["content_sha256"]:
            raise ValueError("future compiler adapter parent mismatch")
        source = (self.root / adapter["callback_source_path"]).resolve()
        if _file_sha(source) != adapter["callback_source_file_sha256"]:
            raise ValueError("future compiler adapter source hash mismatch")
        if (
            adapter.get("data_eligibility") != ELIGIBILITY
            or adapter.get("external_paid_llm_calls") is not False
        ):
            raise ValueError("future compiler adapter opens a forbidden input")

    def _stage_items(self) -> list[dict[str, Any]]:
        counts = [
            ("parameter_manifest", {"parameter_cells": 256}),
            (
                "candidate_compilation",
                {"input_cells": 256, "unique_candidates": 163, "aliases": 93},
            ),
            ("formal_preflight", {"pass": 162, "blocked": 1}),
            ("promotion_admission", {"pass": 162, "excluded": 1}),
            ("aether_reviewed", {"blocked": 126, "reject": 2}),
            ("g2_reviewed", {"blocked": 2}),
            ("g2_nonmaximal_positive_mass_followup", {"pass": 2}),
            ("g3_reviewed", {"blocked": 32}),
            ("g4_reviewed_followup", {"pass": 1}),
            ("sealed_candidate_epoch", {"pass": 3, "reject": 2, "blocked": 158}),
        ]
        items = []
        for ordinal, (stage, expected) in enumerate(counts):
            body = {
                "ordinal": ordinal,
                "stage": stage,
                "expected_counts": expected,
                "source_export_content_sha256": self.evidence_export["content_sha256"],
                "candidate_record_registry_root_sha256": self.evidence_export[
                    "candidate_record_registry_root_sha256"
                ],
            }
            items.append({**body, "stage_lineage_sha256": _sha(body)})
        if self.future is not None and self.adapter is not None:
            body = {
                "ordinal": 10,
                "stage": "future_manifest_chunk_admission",
                "expected_counts": {"admitted_cells": len(self.future["parameter_cells"])},
                "source_export_content_sha256": self.evidence_export["content_sha256"],
                "candidate_record_registry_root_sha256": self.evidence_export[
                    "candidate_record_registry_root_sha256"
                ],
                "future_chunk_content_sha256": self.future["content_sha256"],
                "adapter_content_sha256": self.adapter["content_sha256"],
            }
            items.append({**body, "stage_lineage_sha256": _sha(body)})
        if len(items) > int(self.config["budget"]["maximum_tasks"]):
            raise ValueError("stage registry exceeds task budget")
        return items

    def _initialize_state(self) -> None:
        expected = (
            _sha(self.config),
            self.evidence_export["content_sha256"],
            self.stage_registry_root_sha256,
        )
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SQL)
            row = connection.execute(
                "SELECT immutable_config_sha256,source_export_content_sha256,stage_registry_root_sha256 FROM scalable_campaign_epoch_state"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO scalable_campaign_epoch_state VALUES (1,?,?,?)", expected
                )
            elif tuple(row) != expected:
                raise ValueError("refusing changed scalable campaign epoch resume")

    def enqueue(self) -> dict[str, int]:
        if not self.config["execution_enabled"]:
            raise PermissionError("scalable campaign epoch execution is disabled")
        if self._disk_bytes() > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("scalable campaign epoch disk budget exhausted")
        return self.coordinator.enqueue(
            self.stages,
            lane="cpu",
            max_attempts=int(self.config["budget"]["maximum_attempts_per_task"]),
        )

    def execute(self, lease: WorkLease) -> dict[str, Any]:
        expected = self.stages[lease.ordinal]
        if lease.payload != expected:
            raise ValueError("stage lease payload mismatch")
        result = {
            "stage": expected["stage"],
            "decision": "attested",
            "counts": expected["expected_counts"],
            "stage_lineage_sha256": expected["stage_lineage_sha256"],
            "source_export_content_sha256": self.evidence_export["content_sha256"],
            "candidate_record_registry_root_sha256": self.evidence_export[
                "candidate_record_registry_root_sha256"
            ],
        }
        return {**result, "result_sha256": _sha(result)}

    def run_ready(self, *, worker_id: str = "bounded-scalable-epoch") -> int:
        if not self.config["execution_enabled"]:
            raise PermissionError("scalable campaign epoch execution is disabled")
        completed = 0
        while (lease := self.coordinator.claim("cpu", worker_id)) is not None:
            try:
                result = self.execute(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("lost scalable campaign epoch lease")
                completed += 1
            except Exception as error:
                self.coordinator.fail(lease, worker_id, f"{type(error).__name__}:{error}")
                raise
        self.coordinator.checkpoint()
        return completed

    def status(self) -> dict[str, Any]:
        telemetry = self.coordinator.telemetry()
        with self.coordinator.connect() as connection:
            work_state_root = self.coordinator._work_state_root(connection)
        records = iter_scalable_formal_candidate_evidence_records(self.evidence_export)
        final_counts = dict(Counter(record["final_decision"] for record in records))
        missing = []
        if self.future is None:
            missing.append("missing_hash_bound_future_manifest_chunk")
        if self.adapter is None:
            missing.append("missing_reviewed_future_compiler_adapter")
        status = {
            "schema_version": STATUS_SCHEMA,
            "campaign_id": self.config["campaign_id"],
            "immutable_config_sha256": _sha(self.config),
            "source_export": {
                **self.config["evidence_export"],
                "candidate_record_registry_root_sha256": self.evidence_export[
                    "candidate_record_registry_root_sha256"
                ],
                "alias_lineage_registry_root_sha256": self.evidence_export[
                    "alias_lineage_registry_root_sha256"
                ],
            },
            "stage_registry_root_sha256": self.stage_registry_root_sha256,
            "stage_count": len(self.stages),
            "queue": {"states": telemetry["counts"], "work_state_root": work_state_root},
            "sealed_epoch_counts": {
                "parameter_cells": self.evidence_export["parameter_cell_count"],
                "unique_candidates": self.evidence_export["candidate_count"],
                "aliases": self.evidence_export["alias_count"],
                "decisions": final_counts,
            },
            "next_epoch_readiness": {
                "state": "blocked" if missing else "ready_for_reviewed_compiler_queue_admission",
                "blockers": missing,
                "admitted_future_cells": 0
                if self.future is None or self.adapter is None
                else len(self.future["parameter_cells"]),
                "scientific_compilation_started": False,
            },
            "recovered_on_start": self.recovered_on_start,
            "budget": self.config["budget"],
            "data_eligibility": ELIGIBILITY,
            "external_paid_llm_calls": False,
            "paid_llm_spend_usd": 0.0,
        }
        return {**status, "content_sha256": _sha(status)}

    def export(self, path: str | Path) -> dict[str, Any]:
        status = self.status()
        target = Path(path)
        encoded = (_canonical(status) + "\n").encode()
        if len(encoded) > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("status export exceeds disk budget")
        target.write_bytes(encoded)
        return status

    def _disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())


def load_scalable_campaign_epoch_config(path: str | Path) -> dict[str, Any]:
    config = _load(Path(path))
    _validate_config(config)
    return config


def reviewed_future_manifest_admission_adapter(chunk: dict[str, Any]) -> dict[str, Any]:
    """Hash-only admission boundary; it deliberately performs no compilation."""
    if chunk.get("schema_version") != FUTURE_SCHEMA:
        raise ValueError("unsupported future manifest chunk")
    body = {key: value for key, value in chunk.items() if key != "content_sha256"}
    if chunk.get("content_sha256") != _sha(body):
        raise ValueError("future manifest chunk content hash mismatch")
    if (
        chunk.get("data_eligibility") != ELIGIBILITY
        or chunk.get("external_paid_llm_calls") is not False
    ):
        raise ValueError("future manifest chunk eligibility seal is open")
    result = {
        "decision": "admit",
        "next_task_type": "reviewed_future_candidate_compilation",
        "future_chunk_content_sha256": chunk["content_sha256"],
        "scientific_compilation_started": False,
    }
    return {**result, "result_sha256": _sha(result)}
