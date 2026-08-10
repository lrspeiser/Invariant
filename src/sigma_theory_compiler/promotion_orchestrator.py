from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .high_throughput import (
    build_basis,
    correction_expression,
    decode_ordinal,
)
from .high_throughput import candidate_id as generator_candidate_id
from .survivors import iter_survivors

PIPELINE_SCHEMA = "sigma-promotion-pipeline-1.0"
STATUS_SCHEMA = "sigma-promotion-status-1.0"
EVIDENCE_SCHEMA = "sigma-sampled-static-candidate-evidence-1.0"
ELIGIBILITY = {
    "observational_data_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "paid_llm_calls": False,
}
CATEGORY_ORDER = {"cheap": 0, "symbolic": 1, "formal": 2, "observational": 3}
Evaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def evaluator_binding(descriptor: dict[str, Any]) -> str:
    required = {
        "evaluator_id",
        "version",
        "callback",
        "artifact_path",
        "artifact_sha256",
        "data_eligibility",
    }
    if set(descriptor) != required:
        raise ValueError("evaluator descriptor fields are not exact")
    if descriptor["data_eligibility"] != ELIGIBILITY:
        raise ValueError("evaluator eligibility is not fail-closed")
    artifact = Path(descriptor["artifact_path"]).resolve()
    if not artifact.is_file() or _file_sha(artifact) != descriptor["artifact_sha256"]:
        raise ValueError("evaluator artifact hash mismatch")
    return _sha({key: value for key, value in descriptor.items() if key != "artifact_path"})


def _resolve_evaluator(descriptor: dict[str, Any]) -> Evaluator:
    module_name, separator, attribute = str(descriptor["callback"]).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("evaluator callback must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError("promotion evaluator is not callable")
    source = inspect.getsourcefile(callback)
    if source is None or Path(source).resolve() != Path(descriptor["artifact_path"]).resolve():
        raise ValueError("evaluator callback is not defined by its bound artifact")
    return callback


def validate_pipeline(config: dict[str, Any]) -> None:
    if config.get("schema_version") != PIPELINE_SCHEMA:
        raise ValueError("unsupported promotion pipeline schema")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("paid LLM calls must remain disabled")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("pipeline eligibility is not fail-closed")
    if int(config.get("maximum_evaluator_attempts", 0)) <= 0:
        raise ValueError("maximum evaluator attempts must be positive")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("promotion pipeline requires stages")
    names: set[str] = set()
    previous_category = -1
    for index, stage in enumerate(stages):
        if set(stage) != {
            "name",
            "category",
            "evaluator_id",
            "required_evaluator_binding_sha256",
        }:
            raise ValueError("promotion stage fields are not exact")
        name = str(stage["name"])
        category = str(stage["category"])
        if not name or name in names or category not in CATEGORY_ORDER:
            raise ValueError("invalid or duplicate promotion stage")
        names.add(name)
        order = CATEGORY_ORDER[category]
        if order < previous_category:
            raise ValueError("promotion categories cannot move backward")
        previous_category = order
        binding = stage["required_evaluator_binding_sha256"]
        if binding is not None and not _is_sha256(binding):
            raise ValueError("promotion evaluator binding must be SHA-256 or null")
        if index == 0:
            if name != "sampled_static" or category != "cheap" or binding is not None:
                raise ValueError("first stage must be imported sampled_static evidence")
        elif binding is None and stage["evaluator_id"] is not None:
            raise ValueError("unimplemented stages cannot name an evaluator")
        elif binding is not None and not stage["evaluator_id"]:
            raise ValueError("implemented stages require an evaluator id")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS pipeline (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  config_json TEXT NOT NULL,
  config_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluators (
  evaluator_id TEXT PRIMARY KEY,
  descriptor_json TEXT NOT NULL,
  binding_sha256 TEXT NOT NULL UNIQUE,
  registered_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY,
  ordinal INTEGER NOT NULL,
  source_sha256 TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  initial_lineage_sha256 TEXT NOT NULL,
  registered_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_stages (
  candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
  stage_index INTEGER NOT NULL,
  stage_name TEXT NOT NULL,
  category TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending','running','passed','rejected','blocked')),
  blocker TEXT,
  attempt INTEGER NOT NULL DEFAULT 0,
  evaluator_binding_sha256 TEXT,
  input_lineage_sha256 TEXT,
  result_json TEXT,
  result_sha256 TEXT,
  output_lineage_sha256 TEXT,
  updated_utc TEXT NOT NULL,
  PRIMARY KEY(candidate_id,stage_index)
);
CREATE TABLE IF NOT EXISTS promotion_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc TEXT NOT NULL,
  candidate_id TEXT,
  stage_name TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_promotion_ready ON candidate_stages(state,stage_index,candidate_id);
"""


class PromotionOrchestrator:
    """Durable fail-closed promotion ledger for sampled-static survivors."""

    def __init__(self, database: str | Path, pipeline: dict[str, Any]) -> None:
        validate_pipeline(pipeline)
        self.database = Path(database).resolve()
        self.pipeline = pipeline
        self.stages = list(pipeline["stages"])
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            row = connection.execute("SELECT config_sha256 FROM pipeline WHERE singleton=1").fetchone()
            pipeline_sha = _sha(pipeline)
            if row is None:
                connection.execute(
                    "INSERT INTO pipeline VALUES (1,?,?)", (_canonical(pipeline), pipeline_sha)
                )
            elif row[0] != pipeline_sha:
                raise ValueError("refusing to resume with a changed promotion pipeline")
        self.recover_interrupted()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        event_type: str,
        candidate_id: str | None,
        stage_name: str | None,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO promotion_events(created_utc,candidate_id,stage_name,event_type,payload_json) "
            "VALUES (?,?,?,?,?)",
            (_now(), candidate_id, stage_name, event_type, _canonical(payload)),
        )

    def register_evaluator(self, descriptor: dict[str, Any]) -> str:
        binding = evaluator_binding(descriptor)
        callback = _resolve_evaluator(descriptor)
        del callback
        evaluator_id = str(descriptor["evaluator_id"])
        expected = {
            str(stage["evaluator_id"]): stage["required_evaluator_binding_sha256"]
            for stage in self.stages
            if stage["evaluator_id"] is not None
        }
        if evaluator_id not in expected or expected[evaluator_id] != binding:
            raise ValueError("evaluator is not the hash-bound implementation expected by pipeline")
        portable = {**descriptor, "artifact_path": str(Path(descriptor["artifact_path"]).resolve())}
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT binding_sha256 FROM evaluators WHERE evaluator_id=?", (evaluator_id,)
            ).fetchone()
            if existing is not None and existing[0] != binding:
                raise ValueError("refusing to replace a registered evaluator")
            if existing is None:
                connection.execute(
                    "INSERT INTO evaluators VALUES (?,?,?,?)",
                    (evaluator_id, _canonical(portable), binding, _now()),
                )
                self._event(connection, "evaluator_registered", None, None, {"binding": binding})
        self.reopen_available_gates()
        return binding

    def register_candidate(
        self,
        candidate: dict[str, Any],
        sampled_static_evidence: dict[str, Any],
    ) -> str:
        if candidate.get("data_eligibility") != ELIGIBILITY:
            raise ValueError("candidate eligibility is not fail-closed")
        candidate_id = str(candidate["candidate_id"])
        if sampled_static_evidence != {
            **sampled_static_evidence,
            "schema_version": EVIDENCE_SCHEMA,
            "candidate_id": candidate_id,
            "status": "pass",
            "data_eligibility": ELIGIBILITY,
        }:
            raise ValueError("sampled-static candidate evidence is invalid")
        source_sha = str(sampled_static_evidence.get("source_result_sha256", ""))
        if not _is_sha256(source_sha):
            raise ValueError("sampled-static evidence lacks a source result hash")
        initial_lineage = _sha(
            {"candidate": candidate, "sampled_static_evidence": sampled_static_evidence}
        )
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT initial_lineage_sha256 FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if existing is not None:
                if existing[0] != initial_lineage:
                    raise ValueError("candidate id already has different lineage")
                return initial_lineage
            connection.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?)",
                (
                    candidate_id,
                    int(candidate["ordinal"]),
                    source_sha,
                    _canonical(candidate),
                    initial_lineage,
                    _now(),
                ),
            )
            sampled_hash = _sha(sampled_static_evidence)
            for index, stage in enumerate(self.stages):
                if index == 0:
                    state, blocker = "passed", None
                    result_json, result_sha, output_lineage = (
                        _canonical(sampled_static_evidence),
                        sampled_hash,
                        initial_lineage,
                    )
                elif index == 1:
                    available = self._registered_binding(connection, stage)
                    state, blocker = (
                        ("pending", None)
                        if available
                        else ("blocked", self._missing_reason(stage))
                    )
                    result_json = result_sha = output_lineage = None
                else:
                    state, blocker = "blocked", "awaiting_prior_stage"
                    result_json = result_sha = output_lineage = None
                connection.execute(
                    "INSERT INTO candidate_stages VALUES (?,?,?,?,?,?,0,NULL,NULL,?,?,?,?)",
                    (
                        candidate_id,
                        index,
                        stage["name"],
                        stage["category"],
                        state,
                        blocker,
                        result_json,
                        result_sha,
                        output_lineage,
                        _now(),
                    ),
                )
            self._event(
                connection,
                "candidate_registered",
                candidate_id,
                "sampled_static",
                {"initial_lineage_sha256": initial_lineage},
            )
        return initial_lineage

    def import_rust_survivors(
        self,
        manifest_path: str | Path,
        generator_config_path: str | Path,
        survivor_directory: str | Path,
        *,
        maximum_candidates: int,
    ) -> dict[str, int]:
        """Import a bounded prefix of independently hash-checked SGSURV2 survivors."""
        if not 1 <= maximum_candidates <= 1_000_000:
            raise ValueError("survivor import must be bounded between one and one million")
        manifest_path = Path(manifest_path).resolve()
        generator_path = Path(generator_config_path).resolve()
        directory = Path(survivor_directory).resolve()
        manifest_raw = manifest_path.read_bytes()
        generator_raw = generator_path.read_bytes()
        manifest = json.loads(manifest_raw)
        generator = json.loads(generator_raw)
        if (
            manifest.get("observational_data_opened") is not False
            or generator.get("observational_data_opened") is not False
            or manifest.get("config_sha256") != hashlib.sha256(generator_raw).hexdigest()
            or manifest.get("protocol_version") != generator.get("protocol_version")
        ):
            raise ValueError("survivor manifest/config provenance is invalid")
        for block in manifest.get("blocks", []):
            export = block.get("survivor_export")
            if not export:
                continue
            path = directory / export["file"]
            if (
                not path.is_file()
                or path.stat().st_size != int(export["file_size_bytes"])
                or _file_sha(path) != export["file_sha256"]
            ):
                raise ValueError("survivor binary hash or size mismatch")
        basis = build_basis(int(generator["basis_count"]))
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        status_root = str(manifest.get("blocks_root_sha256", ""))
        if not _is_sha256(status_root):
            raise ValueError("survivor manifest lacks a blocks root")
        accepted = duplicates = 0
        for index, survivor in enumerate(iter_survivors(manifest_path, directory)):
            if index >= maximum_candidates:
                break
            decoded = decode_ordinal(
                int(generator["basis_count"]),
                int(generator["max_action_terms"]),
                int(survivor["ordinal"]),
            )
            decoded_mask = sum(
                1 << position
                for position, sign in enumerate(decoded["signs"])
                if sign > 0
            )
            if decoded["term_ids"] != list(survivor["term_ids"]) or decoded_mask != int(
                survivor["sign_mask"]
            ):
                raise ValueError("survivor record differs from ordinal decoder")
            identifier = generator_candidate_id(str(generator["protocol_version"]), decoded)
            candidate = {
                "candidate_id": identifier,
                "ordinal": int(survivor["ordinal"]),
                "term_ids": list(decoded["term_ids"]),
                "signs": list(decoded["signs"]),
                "correction_expression": correction_expression(decoded, basis),
                "source_manifest_sha256": manifest_sha,
                "data_eligibility": ELIGIBILITY,
            }
            evidence = {
                "schema_version": EVIDENCE_SCHEMA,
                "candidate_id": identifier,
                "ordinal": int(survivor["ordinal"]),
                "status": "pass",
                "source_result_sha256": manifest_sha,
                "status_root_sha256": status_root,
                "data_eligibility": ELIGIBILITY,
            }
            with self.connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM candidates WHERE candidate_id=?", (identifier,)
                ).fetchone()
            self.register_candidate(candidate, evidence)
            if exists:
                duplicates += 1
            else:
                accepted += 1
        return {"accepted": accepted, "duplicates": duplicates, "limit": maximum_candidates}

    @staticmethod
    def _missing_reason(stage: dict[str, Any]) -> str:
        return (
            "unimplemented_gate_fail_closed"
            if stage["required_evaluator_binding_sha256"] is None
            else "hash_bound_evaluator_not_registered"
        )

    @staticmethod
    def _registered_binding(
        connection: sqlite3.Connection, stage: dict[str, Any]
    ) -> str | None:
        expected = stage["required_evaluator_binding_sha256"]
        if expected is None:
            return None
        row = connection.execute(
            "SELECT binding_sha256 FROM evaluators WHERE evaluator_id=?",
            (stage["evaluator_id"],),
        ).fetchone()
        return expected if row is not None and row[0] == expected else None

    def reopen_available_gates(self) -> int:
        reopened = 0
        with self.connect() as connection:
            for index, stage in enumerate(self.stages[1:], start=1):
                binding = self._registered_binding(connection, stage)
                if not binding:
                    continue
                rows = connection.execute(
                    "SELECT candidate_id FROM candidate_stages WHERE stage_index=? AND state='blocked' "
                    "AND blocker IN ('hash_bound_evaluator_not_registered','awaiting_prior_stage')",
                    (index,),
                ).fetchall()
                for row in rows:
                    prior = connection.execute(
                        "SELECT state,output_lineage_sha256 FROM candidate_stages "
                        "WHERE candidate_id=? AND stage_index=?",
                        (row["candidate_id"], index - 1),
                    ).fetchone()
                    if prior["state"] == "passed":
                        connection.execute(
                            "UPDATE candidate_stages SET state='pending',blocker=NULL,"
                            "input_lineage_sha256=?,updated_utc=? WHERE candidate_id=? AND stage_index=?",
                            (prior["output_lineage_sha256"], _now(), row["candidate_id"], index),
                        )
                        reopened += 1
        return reopened

    def recover_interrupted(self) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT candidate_id,stage_name FROM candidate_stages WHERE state='running'"
            ).fetchall()
            connection.execute(
                "UPDATE candidate_stages SET state='pending',blocker='recovered_interrupted_run',"
                "updated_utc=? WHERE state='running'",
                (_now(),),
            )
            for row in rows:
                self._event(
                    connection,
                    "interrupted_gate_recovered",
                    row["candidate_id"],
                    row["stage_name"],
                    {},
                )
        return len(rows)

    def _descriptor(self, evaluator_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT descriptor_json FROM evaluators WHERE evaluator_id=?", (evaluator_id,)
            ).fetchone()
        if row is None:
            raise ValueError("promotion evaluator is not registered")
        descriptor = json.loads(row[0])
        evaluator_binding(descriptor)
        return descriptor

    def run_ready(self, maximum_tasks: int = 100) -> dict[str, int]:
        if maximum_tasks <= 0:
            raise ValueError("promotion maximum tasks must be positive")
        passed = rejected = blocked = evaluated = 0
        for _ in range(maximum_tasks):
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT s.*,c.payload_json,c.initial_lineage_sha256 FROM candidate_stages s "
                    "JOIN candidates c USING(candidate_id) WHERE s.state='pending' "
                    "ORDER BY s.stage_index,s.candidate_id LIMIT 1"
                ).fetchone()
                if row is None:
                    break
                stage = self.stages[int(row["stage_index"])]
                binding = self._registered_binding(connection, stage)
                if binding is None:
                    connection.execute(
                        "UPDATE candidate_stages SET state='blocked',blocker=?,updated_utc=? "
                        "WHERE candidate_id=? AND stage_index=?",
                        (self._missing_reason(stage), _now(), row["candidate_id"], row["stage_index"]),
                    )
                    blocked += 1
                    continue
                prior_lineage = row["input_lineage_sha256"]
                if prior_lineage is None:
                    prior = connection.execute(
                        "SELECT output_lineage_sha256 FROM candidate_stages WHERE candidate_id=? "
                        "AND stage_index=?",
                        (row["candidate_id"], int(row["stage_index"]) - 1),
                    ).fetchone()
                    prior_lineage = prior[0]
                input_hash = _sha(
                    {
                        "candidate_id": row["candidate_id"],
                        "stage": stage,
                        "prior_lineage_sha256": prior_lineage,
                        "evaluator_binding_sha256": binding,
                    }
                )
                connection.execute(
                    "UPDATE candidate_stages SET state='running',attempt=attempt+1,blocker=NULL,"
                    "evaluator_binding_sha256=?,input_lineage_sha256=?,updated_utc=? "
                    "WHERE candidate_id=? AND stage_index=?",
                    (binding, input_hash, _now(), row["candidate_id"], row["stage_index"]),
                )
                candidate = json.loads(row["payload_json"])
                attempt = int(row["attempt"]) + 1
            descriptor = self._descriptor(str(stage["evaluator_id"]))
            callback = _resolve_evaluator(descriptor)
            try:
                output = callback(
                    candidate,
                    {
                        "stage_name": stage["name"],
                        "category": stage["category"],
                        "attempt": attempt,
                        "input_lineage_sha256": input_hash,
                        "data_eligibility": ELIGIBILITY,
                    },
                )
                if not isinstance(output, dict) or output.get("decision") not in {
                    "pass",
                    "reject",
                    "blocked",
                }:
                    raise ValueError("evaluator must return pass/reject/blocked decision")
                if output.get("data_eligibility") != ELIGIBILITY:
                    raise ValueError("evaluator output eligibility is not fail-closed")
                if output.get("decision") == "blocked" and not str(output.get("blocker", "")):
                    raise ValueError("blocked evaluator decisions require an explicit blocker")
            except Exception as error:  # noqa: BLE001 - gate failures must be persisted
                with self.connect() as connection:
                    retry = attempt < int(self.pipeline["maximum_evaluator_attempts"])
                    connection.execute(
                        "UPDATE candidate_stages SET state=?,blocker=?,updated_utc=? "
                        "WHERE candidate_id=? AND stage_index=? AND state='running'",
                        (
                            "pending" if retry else "blocked",
                            f"evaluator_error:{type(error).__name__}:{error}",
                            _now(),
                            row["candidate_id"],
                            row["stage_index"],
                        ),
                    )
                if not retry:
                    blocked += 1
                evaluated += 1
                continue
            wrapper = {
                "schema_version": "sigma-promotion-gate-result-1.0",
                "candidate_id": row["candidate_id"],
                "stage_name": stage["name"],
                "category": stage["category"],
                "evaluator_binding_sha256": binding,
                "input_lineage_sha256": input_hash,
                "output": output,
                "data_eligibility": ELIGIBILITY,
            }
            result_hash = _sha(wrapper)
            output_lineage = _sha(
                {"input_lineage_sha256": input_hash, "result_sha256": result_hash}
            )
            decision = str(output["decision"])
            stage_state = {
                "pass": "passed",
                "reject": "rejected",
                "blocked": "blocked",
            }[decision]
            stage_blocker = str(output["blocker"]) if decision == "blocked" else None
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE candidate_stages SET state=?,blocker=?,result_json=?,result_sha256=?,"
                    "output_lineage_sha256=?,updated_utc=? WHERE candidate_id=? AND stage_index=? "
                    "AND state='running'",
                    (
                        stage_state,
                        stage_blocker,
                        _canonical(wrapper),
                        result_hash,
                        output_lineage,
                        _now(),
                        row["candidate_id"],
                        row["stage_index"],
                    ),
                )
                self._event(
                    connection,
                    f"gate_{decision}",
                    row["candidate_id"],
                    stage["name"],
                    {"result_sha256": result_hash, "output_lineage_sha256": output_lineage},
                )
                next_index = int(row["stage_index"]) + 1
                if decision == "reject":
                    connection.execute(
                        "UPDATE candidate_stages SET state='blocked',blocker='upstream_rejected',"
                        "updated_utc=? WHERE candidate_id=? AND stage_index>?",
                        (_now(), row["candidate_id"], row["stage_index"]),
                    )
                    rejected += 1
                elif decision == "blocked":
                    blocked += 1
                elif next_index < len(self.stages):
                    next_stage = self.stages[next_index]
                    next_binding = self._registered_binding(connection, next_stage)
                    connection.execute(
                        "UPDATE candidate_stages SET state=?,blocker=?,input_lineage_sha256=?,"
                        "updated_utc=? WHERE candidate_id=? AND stage_index=?",
                        (
                            "pending" if next_binding else "blocked",
                            None if next_binding else self._missing_reason(next_stage),
                            output_lineage,
                            _now(),
                            row["candidate_id"],
                            next_index,
                        ),
                    )
                    passed += 1
                else:
                    passed += 1
            evaluated += 1
        return {"evaluated": evaluated, "passed": passed, "rejected": rejected, "blocked": blocked}

    def status(self) -> dict[str, Any]:
        with self.connect() as connection:
            stage_rows = connection.execute(
                "SELECT stage_index,stage_name,category,state,COUNT(*) AS count "
                "FROM candidate_stages GROUP BY stage_index,stage_name,category,state "
                "ORDER BY stage_index,state"
            ).fetchall()
            candidate_rows = connection.execute(
                "SELECT c.candidate_id,c.ordinal,s.stage_name,s.category,s.state,s.blocker,"
                "s.output_lineage_sha256 FROM candidates c JOIN candidate_stages s "
                "ON s.candidate_id=c.candidate_id WHERE s.stage_index=COALESCE((SELECT MIN(x.stage_index) "
                "FROM candidate_stages x WHERE x.candidate_id=c.candidate_id AND x.state!='passed'),"
                "(SELECT MAX(y.stage_index) FROM candidate_stages y WHERE "
                "y.candidate_id=c.candidate_id)) ORDER BY c.candidate_id"
            ).fetchall()
            evaluators = connection.execute(
                "SELECT evaluator_id,binding_sha256 FROM evaluators ORDER BY evaluator_id"
            ).fetchall()
            event_root = _sha(
                [
                    dict(row)
                    for row in connection.execute(
                        "SELECT candidate_id,stage_name,event_type,payload_json "
                        "FROM promotion_events ORDER BY sequence"
                    )
                ]
            )
        by_stage: dict[str, Any] = {
            stage["name"]: {
                "index": index,
                "category": stage["category"],
                "implemented": index == 0
                or stage["required_evaluator_binding_sha256"] is not None,
                "counts": {},
            }
            for index, stage in enumerate(self.stages)
        }
        for row in stage_rows:
            by_stage[row["stage_name"]]["counts"][row["state"]] = int(row["count"])
        report = {
            "schema_version": STATUS_SCHEMA,
            "pipeline_sha256": _sha(self.pipeline),
            "candidate_count": len(candidate_rows),
            "stages": by_stage,
            "registered_evaluators": [dict(row) for row in evaluators],
            "candidates": [dict(row) for row in candidate_rows],
            "unimplemented_gates_fail_closed": [
                stage["name"]
                for stage in self.stages
                if stage["required_evaluator_binding_sha256"] is None and stage["name"] != "sampled_static"
            ],
            "event_root_sha256": event_root,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "interpretation": (
                "Promotion records gate evidence and lineage only. A blocked or missing gate "
                "cannot be treated as passed, and promotion is not proof of a gravity theory."
            ),
        }
        report["content_sha256"] = _sha(report)
        return report

    def write_status(self, output: str | Path) -> Path:
        path = Path(output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.status()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path
