from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .promotion_orchestrator import (
    ELIGIBILITY,
    PromotionOrchestrator,
    evaluator_binding,
    validate_pipeline,
)
from .rust_promotion_bridge import RustPromotionBridge

SCHEMA_VERSION = "sigma-rust-promotion-service-stage-1.0"
STATUS_SCHEMA = "sigma-rust-promotion-service-status-1.0"
SERVICE_ELIGIBILITY = {**ELIGIBILITY, "passed": True}

SQL = """
CREATE TABLE IF NOT EXISTS downstream_stage_state (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  config_sha256 TEXT NOT NULL,
  source_service_id TEXT,
  deadline_utc TEXT NOT NULL,
  run_count INTEGER NOT NULL,
  evaluator_registry_root_sha256 TEXT,
  last_status_json TEXT,
  last_status_sha256 TEXT
);
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_stage_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "enabled",
        "external_paid_llm_calls",
        "pipeline_config_path",
        "generator_config_path",
        "reviewed_evaluator_descriptors",
        "maximum_records_per_run",
        "maximum_blocks_per_run",
        "maximum_orchestrator_tasks_per_run",
        "maximum_total_candidates",
        "maximum_disk_bytes",
        "maximum_wall_seconds",
        "data_eligibility",
    }
    if set(config) != required or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("promotion service stage fields or schema are invalid")
    if config.get("enabled") is not True:
        raise ValueError("configured promotion service stage must be explicitly enabled")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("promotion service paid LLM calls must remain disabled")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("promotion service eligibility is not fail-closed")
    bounded = {
        "maximum_records_per_run": 1_000_000,
        "maximum_blocks_per_run": 10_000,
        "maximum_orchestrator_tasks_per_run": 1_000_000,
        "maximum_total_candidates": 1_000_000,
    }
    for key, maximum in bounded.items():
        if not 1 <= int(config.get(key, 0)) <= maximum:
            raise ValueError(f"{key} is outside its fail-closed bound")
    if int(config.get("maximum_disk_bytes", 0)) <= 0:
        raise ValueError("promotion service disk budget must be positive")
    if not 0 < float(config.get("maximum_wall_seconds", 0)) <= 1_209_600:
        raise ValueError("promotion service wall budget is invalid")
    descriptors = config.get("reviewed_evaluator_descriptors")
    if not isinstance(descriptors, list) or len(descriptors) > 16:
        raise ValueError("reviewed evaluator descriptor allowlist is invalid")
    evaluator_ids: set[str] = set()
    for item in descriptors:
        if not isinstance(item, dict) or set(item) != {
            "evaluator_id",
            "descriptor_path",
            "descriptor_sha256",
            "required_binding_sha256",
        }:
            raise ValueError("reviewed evaluator allowlist entry fields are invalid")
        evaluator_id = str(item["evaluator_id"])
        if not evaluator_id or evaluator_id in evaluator_ids:
            raise ValueError("reviewed evaluator ids must be unique")
        evaluator_ids.add(evaluator_id)
        for key in ("descriptor_sha256", "required_binding_sha256"):
            value = str(item[key])
            if len(value) != 64:
                raise ValueError("reviewed evaluator allowlist hash is invalid")
            try:
                bytes.fromhex(value)
            except ValueError as error:
                raise ValueError("reviewed evaluator allowlist hash is invalid") from error


class RustPromotionServiceStage:
    """Bounded one-way service stage from verified promotion blocks to durable gates."""

    def __init__(self, directory: str | Path, config: dict[str, Any]) -> None:
        validate_stage_config(config)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.pipeline_path = Path(config["pipeline_config_path"]).resolve()
        self.generator_path = Path(config["generator_config_path"]).resolve()
        if not self.pipeline_path.is_file() or not self.generator_path.is_file():
            raise FileNotFoundError("promotion pipeline or generator config is unavailable")
        self.pipeline = _load(self.pipeline_path)
        validate_pipeline(self.pipeline)
        generator = _load(self.generator_path)
        if generator.get("observational_data_opened") is not False:
            raise ValueError("promotion service generator opens observations")
        self.config_sha = _sha(config)
        self.pipeline_sha = _file_sha(self.pipeline_path)
        self.generator_sha = _file_sha(self.generator_path)
        self.reviewed_descriptors = self._load_reviewed_descriptors()
        self.evaluator_registry_root = _sha(
            [
                {
                    "evaluator_id": item["evaluator_id"],
                    "descriptor_sha256": item["descriptor_sha256"],
                    "binding_sha256": item["binding_sha256"],
                }
                for item in self.reviewed_descriptors
            ]
        )
        self.state_database = self.directory / "stage.sqlite"
        self.bridge_database = self.directory / "bridge.sqlite"
        self.orchestrator_database = self.directory / "promotion.sqlite"
        connection = sqlite3.connect(self.state_database)
        connection.executescript(SQL)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(downstream_stage_state)")
        }
        if "evaluator_registry_root_sha256" not in columns:
            connection.execute(
                "ALTER TABLE downstream_stage_state ADD COLUMN evaluator_registry_root_sha256 TEXT"
            )
        row = connection.execute(
            "SELECT config_sha256 FROM downstream_stage_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            deadline = datetime.now(UTC) + timedelta(
                seconds=float(config["maximum_wall_seconds"])
            )
            connection.execute(
                "INSERT INTO downstream_stage_state "
                "(singleton,config_sha256,source_service_id,deadline_utc,run_count,"
                "evaluator_registry_root_sha256,last_status_json,last_status_sha256) "
                "VALUES (1,?,NULL,?,0,?,NULL,NULL)",
                (self.config_sha, deadline.isoformat(), self.evaluator_registry_root),
            )
        elif row[0] != self.config_sha:
            connection.close()
            raise ValueError("refusing to resume a changed promotion service stage")
        else:
            stored_root = connection.execute(
                "SELECT evaluator_registry_root_sha256 FROM downstream_stage_state "
                "WHERE singleton=1"
            ).fetchone()[0]
            if stored_root not in (None, self.evaluator_registry_root):
                connection.close()
                raise ValueError("reviewed evaluator registry changed across restart")
            connection.execute(
                "UPDATE downstream_stage_state SET evaluator_registry_root_sha256=? "
                "WHERE singleton=1",
                (self.evaluator_registry_root,),
            )
        connection.commit()
        connection.close()

    def _load_reviewed_descriptors(self) -> list[dict[str, Any]]:
        expected = {
            str(stage["evaluator_id"]): stage["required_evaluator_binding_sha256"]
            for stage in self.pipeline["stages"]
            if stage["evaluator_id"] is not None
        }
        loaded: list[dict[str, Any]] = []
        for allow in self.config["reviewed_evaluator_descriptors"]:
            path = Path(allow["descriptor_path"]).resolve()
            if not path.is_file() or _file_sha(path) != allow["descriptor_sha256"]:
                raise ValueError("reviewed evaluator descriptor file hash mismatch")
            descriptor = _load(path)
            if descriptor.get("evaluator_id") != allow["evaluator_id"]:
                raise ValueError("reviewed evaluator descriptor id mismatch")
            artifact = Path(descriptor["artifact_path"])
            if not artifact.is_absolute():
                candidates = [
                    (Path.cwd() / artifact).resolve(),
                    (path.parent / artifact).resolve(),
                    (path.parent.parent / artifact).resolve(),
                ]
                artifact = next(
                    (candidate for candidate in candidates if candidate.is_file()), candidates[0]
                )
            descriptor["artifact_path"] = str(artifact)
            binding = evaluator_binding(descriptor)
            if (
                binding != allow["required_binding_sha256"]
                or expected.get(str(allow["evaluator_id"])) != binding
            ):
                raise ValueError("reviewed evaluator binding does not match the pipeline")
            loaded.append(
                {
                    "evaluator_id": str(allow["evaluator_id"]),
                    "descriptor_sha256": str(allow["descriptor_sha256"]),
                    "binding_sha256": binding,
                    "descriptor": descriptor,
                }
            )
        return sorted(loaded, key=lambda item: item["evaluator_id"])

    def _register_reviewed_evaluators(
        self, orchestrator: PromotionOrchestrator
    ) -> list[dict[str, str]]:
        registered = []
        for item in self.reviewed_descriptors:
            binding = orchestrator.register_evaluator(item["descriptor"])
            registered.append(
                {"evaluator_id": item["evaluator_id"], "binding_sha256": binding}
            )
        actual = orchestrator.status()["registered_evaluators"]
        if actual != registered:
            raise ValueError("promotion registry contains an unlisted evaluator")
        return registered

    def _state(self) -> sqlite3.Row:
        connection = sqlite3.connect(self.state_database)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM downstream_stage_state WHERE singleton=1"
        ).fetchone()
        connection.close()
        return row

    def _provenance(self, export_sha: str) -> dict[str, Any]:
        bridge = RustPromotionBridge(self.bridge_database)
        orchestrator = PromotionOrchestrator(self.orchestrator_database, self.pipeline)
        with bridge.connect() as connection:
            verified = [
                dict(row)
                for row in connection.execute(
                    "SELECT block_index,start_ordinal,end_ordinal_exclusive,block_sha256,"
                    "block_lineage_sha256,record_count FROM verified_blocks ORDER BY block_index"
                )
            ]
            consumed = [
                dict(row)
                for row in connection.execute(
                    "SELECT block_index,record_index,ordinal,sampled_status,candidate_id,"
                    "initial_lineage_sha256 FROM consumed_records ORDER BY block_index,record_index"
                )
            ]
        with orchestrator.connect() as connection:
            candidates = [
                dict(row)
                for row in connection.execute(
                    "SELECT candidate_id,ordinal,source_sha256,initial_lineage_sha256 "
                    "FROM candidates ORDER BY candidate_id"
                )
            ]
        return {
            "source_export_sha256": export_sha,
            "generator_config_sha256": self.generator_sha,
            "pipeline_config_sha256": self.pipeline_sha,
            "verified_blocks_root_sha256": _sha(verified),
            "consumed_records_root_sha256": _sha(consumed),
            "candidate_lineage_root_sha256": _sha(candidates),
            "evaluator_registry_root_sha256": self.evaluator_registry_root,
            "combined_root_sha256": _sha(
                {
                    "export": export_sha,
                    "generator": self.generator_sha,
                    "pipeline": self.pipeline_sha,
                    "evaluators": self.evaluator_registry_root,
                    "verified": verified,
                    "consumed": consumed,
                    "candidates": candidates,
                }
            ),
        }

    def run(self, export_path: str | Path) -> dict[str, Any]:
        export_path = Path(export_path).resolve()
        export = _load(export_path)
        state = self._state()
        now = datetime.now(UTC)
        deadline = datetime.fromisoformat(state["deadline_utc"])
        if now >= deadline:
            raise TimeoutError("promotion service original deadline is exhausted")
        service_id = str(export.get("service_id", ""))
        if not service_id:
            raise ValueError("promotion export lacks a service identity")
        if state["source_service_id"] not in (None, service_id):
            raise ValueError("promotion service source identity changed")
        if _directory_bytes(self.directory) >= int(self.config["maximum_disk_bytes"]):
            raise ValueError("promotion service disk budget is exhausted")
        orchestrator = PromotionOrchestrator(self.orchestrator_database, self.pipeline)
        registered_evaluators = self._register_reviewed_evaluators(orchestrator)
        before = orchestrator.status()
        capacity = int(self.config["maximum_total_candidates"]) - int(
            before["candidate_count"]
        )
        bridge_report: dict[str, Any] | None = None
        run_report: dict[str, Any] = {
            "processed": 0,
            "passed": 0,
            "rejected": 0,
            "blocked": 0,
            "failed": 0,
        }
        outcome = "backpressured" if capacity <= 0 else "completed"
        started = time.monotonic()
        if capacity > 0:
            maximum_records = min(int(self.config["maximum_records_per_run"]), capacity)
            bridge_report = RustPromotionBridge(self.bridge_database).import_incremental(
                export_path,
                self.generator_path,
                orchestrator,
                maximum_records=maximum_records,
                maximum_blocks=int(self.config["maximum_blocks_per_run"]),
            )
            if datetime.now(UTC) >= deadline:
                raise TimeoutError("promotion service deadline exhausted after bridge import")
            run_report = orchestrator.run_ready(
                maximum_tasks=int(self.config["maximum_orchestrator_tasks_per_run"])
            )
        after = orchestrator.status()
        disk_bytes = _directory_bytes(self.directory)
        if disk_bytes > int(self.config["maximum_disk_bytes"]):
            raise ValueError("promotion service exceeded its disk budget")
        export_sha = _file_sha(export_path)
        provenance = self._provenance(export_sha)
        status = {
            "schema_version": STATUS_SCHEMA,
            "outcome": outcome,
            "source_service_id": service_id,
            "deadline_utc": state["deadline_utc"],
            "elapsed_seconds": time.monotonic() - started,
            "bridge_run": bridge_report,
            "orchestrator_run": run_report,
            "bridge": RustPromotionBridge(self.bridge_database).status(),
            "orchestrator": after,
            "candidate_count_before": before["candidate_count"],
            "candidate_count_after": after["candidate_count"],
            "maximum_total_candidates": int(self.config["maximum_total_candidates"]),
            "disk_bytes": disk_bytes,
            "maximum_disk_bytes": int(self.config["maximum_disk_bytes"]),
            "provenance": provenance,
            "reviewed_evaluator_registry": registered_evaluators,
            "evaluator_registry_root_sha256": self.evaluator_registry_root,
            "upstream_mutation_contract": (
                "one-way artifact import only; downstream states have no write path to upstream"
            ),
            "data_eligibility": SERVICE_ELIGIBILITY,
            "paid_llm_spend_usd": 0.0,
        }
        status["content_sha256"] = _sha(status)
        connection = sqlite3.connect(self.state_database)
        connection.execute(
            "UPDATE downstream_stage_state SET source_service_id=?,run_count=run_count+1,"
            "last_status_json=?,last_status_sha256=? WHERE singleton=1",
            (service_id, _canonical(status), status["content_sha256"]),
        )
        connection.commit()
        connection.close()
        return status

    def status(self) -> dict[str, Any]:
        state = self._state()
        if state["last_status_json"]:
            status = json.loads(state["last_status_json"])
            if _sha({key: value for key, value in status.items() if key != "content_sha256"}) != (
                state["last_status_sha256"]
            ):
                raise ValueError("stored promotion service status hash mismatch")
            return {**status, "run_count": int(state["run_count"])}
        return {
            "schema_version": STATUS_SCHEMA,
            "outcome": "not_run",
            "deadline_utc": state["deadline_utc"],
            "run_count": 0,
            "reviewed_evaluator_count": len(self.reviewed_descriptors),
            "evaluator_registry_root_sha256": self.evaluator_registry_root,
            "data_eligibility": SERVICE_ELIGIBILITY,
            "paid_llm_spend_usd": 0.0,
        }
