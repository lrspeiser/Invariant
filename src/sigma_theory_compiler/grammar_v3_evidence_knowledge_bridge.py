from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .knowledge import pareto_fronts
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-evidence-knowledge-config-1.0"
PACKET_SCHEMA = "sigma-formula-knowledge-evidence-packet-1.0"
REPORT_SCHEMA = "sigma-grammar-v3-evidence-pareto-report-1.0"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS bridge_state (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  source_registry_root_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_sources (
  source_id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  path TEXT NOT NULL,
  file_sha256 TEXT NOT NULL,
  content_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
  seed_id TEXT PRIMARY KEY,
  seed_lineage_sha256 TEXT NOT NULL,
  family_id TEXT NOT NULL,
  family_lineage_sha256 TEXT NOT NULL,
  action_sha256 TEXT NOT NULL,
  compilation_provenance_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_packets (
  packet_id TEXT PRIMARY KEY,
  candidate_id TEXT,
  outcome_class TEXT NOT NULL CHECK(outcome_class IN ('pass','reject','blocked')),
  calibration_only INTEGER NOT NULL CHECK(calibration_only IN (0,1)),
  packet_json TEXT NOT NULL,
  packet_sha256 TEXT NOT NULL,
  FOREIGN KEY(candidate_id) REFERENCES candidates(seed_id)
);
CREATE TABLE IF NOT EXISTS source_packet_links (
  source_id TEXT NOT NULL,
  packet_id TEXT NOT NULL,
  PRIMARY KEY(source_id,packet_id),
  FOREIGN KEY(source_id) REFERENCES evidence_sources(source_id),
  FOREIGN KEY(packet_id) REFERENCES evidence_packets(packet_id)
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


def _content_hash(value: dict[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _outcome_class(status: Any) -> str:
    normalized = str(status).lower()
    if normalized == "pass":
        return "pass"
    if normalized in {"reject", "rejected", "fail", "failed"}:
        return "reject"
    return "blocked"


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "sources",
        "budget",
        "priority_axes",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 evidence knowledge config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 evidence knowledge eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("grammar-v3 evidence knowledge config enabled paid LLM calls")
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("grammar-v3 evidence knowledge requires exactly five bound sources")
    if len({item.get("source_id") for item in sources}) != 5:
        raise ValueError("grammar-v3 evidence source ids are not unique")
    budget = config.get("budget", {})
    if (
        set(budget) != {"maximum_candidates", "maximum_packets", "maximum_database_bytes"}
        or int(budget["maximum_candidates"]) != 6
        or not 23 <= int(budget["maximum_packets"]) <= 64
        or not 4096 <= int(budget["maximum_database_bytes"]) <= 64 * 1024 * 1024
    ):
        raise ValueError("grammar-v3 evidence knowledge budget is invalid")
    expected_axes = [
        "formal_pass_count",
        "candidate_evidence_packet_count",
        "source_lineage_depth",
        "blocker_reduction_margin",
    ]
    if config.get("priority_axes") != expected_axes:
        raise ValueError("grammar-v3 Pareto axes changed or collapsed")


def _gate_ledger(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("gate_ledger", "premise_ledger"):
        if isinstance(record.get(key), dict):
            return record[key]
    return {}


def _blocker_taxonomy(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes = []
    blockers = []
    for gate_id, gate in sorted(_gate_ledger(record).items()):
        source_status = str(gate.get("status", "blocked"))
        outcome = _outcome_class(source_status)
        item = {
            "gate_id": gate_id,
            "source_status": source_status,
            "outcome_class": outcome,
            "reason": gate.get("reason"),
            "evidence_root_sha256": gate.get("evidence_root_sha256"),
        }
        outcomes.append(item)
        if outcome != "pass":
            blockers.append(item)
    return outcomes, blockers


def _missing_premises(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "first_missing_premise",
            "first_missing_adm_dirac_premise",
            "first_missing_uniform_principal_premise",
            "global_energy_blocker",
        )
        if key in record
    }


def _candidate_packet(source_id: str, record: dict[str, Any]) -> dict[str, Any]:
    gates, blockers = _blocker_taxonomy(record)
    body = {
        "schema_version": PACKET_SCHEMA,
        "packet_role": "candidate_specific_evidence",
        "source_id": source_id,
        "candidate_id": record["seed_id"],
        "family_id": record.get("family_id"),
        "outcome_class": _outcome_class(record["decision"]),
        "source_decision": record["decision"],
        "calibration_only": False,
        "eligible_for_candidate_priority": True,
        "gate_outcomes": gates,
        "blocker_taxonomy": blockers,
        "first_missing_prerequisites": _missing_premises(record),
        "provenance": record["provenance"],
        "data_eligibility": dict(ELIGIBILITY),
    }
    packet_hash = _sha(body)
    return {**body, "packet_id": "EVP-" + packet_hash[:24], "content_sha256": packet_hash}


def _calibration_packet(control_id: str, control: dict[str, Any]) -> dict[str, Any]:
    evidence_sha = control.get("evidence_sha256", control.get("result_sha256"))
    body = {
        "schema_version": PACKET_SCHEMA,
        "packet_role": "calibration_only_control",
        "control_id": control_id,
        "candidate_id": None,
        "outcome_class": _outcome_class(control.get("status", "pass")),
        "source_status": control.get("status", "pass"),
        "calibration_only": True,
        "eligible_for_candidate_priority": False,
        "entrypoint": control.get("entrypoint"),
        "scope": control.get("scope"),
        "evidence_sha256": evidence_sha,
        "control_payload": control,
        "data_eligibility": dict(ELIGIBILITY),
    }
    packet_hash = _sha(body)
    return {**body, "packet_id": "EVP-" + packet_hash[:24], "content_sha256": packet_hash}


def _special_control_packets(source: dict[str, Any]) -> list[dict[str, Any]]:
    packets = []
    known = source.get("known_answer_control")
    if isinstance(known, dict):
        packets.append(
            _calibration_packet(
                "canonical_scalar_gr_known_answer",
                {
                    **known,
                    "status": "pass",
                    "scope": "calibration-only known answer; never candidate evidence",
                    "evidence_sha256": known["binding_sha256"],
                },
            )
        )
    witness = source.get("twisting_unit_aether_witness")
    if isinstance(witness, dict):
        packets.append(
            _calibration_packet(
                "twisting_unit_aether_hypersurface_orthogonality_negative_control",
                {
                    **witness,
                    "status": "reject",
                    "scope": witness["scope"],
                    "evidence_sha256": witness["content_sha256"],
                },
            )
        )
    return packets


class GrammarV3EvidenceKnowledgeBridge:
    """Immutable grammar-v3 evidence registry and multi-axis Pareto follow-up queue."""

    def __init__(
        self, database: str | Path, config: dict[str, Any], project_root: str | Path
    ) -> None:
        _validate_config(config)
        self.database = Path(database).resolve()
        if self.database.name.lower() == "campaign-v1-live.sqlite":
            raise ValueError("refusing to use the live campaign watchdog database")
        self.config = config
        self.root = Path(project_root).resolve()
        self.sources = self._load_sources()
        self.source_registry_root_sha256 = _sha(
            [
                {
                    key: descriptor[key]
                    for key in ("source_id", "role", "file_sha256", "content_sha256")
                }
                for descriptor, _ in self.sources
            ]
        )
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            row = connection.execute("SELECT * FROM bridge_state WHERE singleton=1").fetchone()
            expected = {
                "singleton": 1,
                "schema_version": REPORT_SCHEMA,
                "config_sha256": _sha(config),
                "source_registry_root_sha256": self.source_registry_root_sha256,
            }
            if row is None:
                connection.execute(
                    "INSERT INTO bridge_state VALUES (1,?,?,?)",
                    (REPORT_SCHEMA, _sha(config), self.source_registry_root_sha256),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to replay a changed grammar-v3 evidence registry")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _load_sources(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        loaded = []
        for descriptor in self.config["sources"]:
            path = self.root / descriptor["path"]
            if not path.is_file() or _file_sha(path) != descriptor["file_sha256"]:
                raise ValueError(f"bound evidence source file mismatch: {descriptor['source_id']}")
            value = _load(path)
            if (
                value.get("content_sha256") != descriptor["content_sha256"]
                or _content_hash(value) != descriptor["content_sha256"]
                or value.get("data_eligibility") != ELIGIBILITY
                or value.get("observational_data_opened") is not False
            ):
                raise ValueError(f"bound evidence source content mismatch: {descriptor['source_id']}")
            loaded.append((descriptor, value))
        return loaded

    def _database_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in (
                self.database,
                Path(str(self.database) + "-wal"),
                Path(str(self.database) + "-shm"),
            )
            if path.is_file()
        )

    def _enforce_disk_budget(self) -> None:
        if self._database_bytes() > int(self.config["budget"]["maximum_database_bytes"]):
            raise RuntimeError("grammar-v3 evidence registry disk budget exhausted")

    @staticmethod
    def _insert_immutable(
        connection: sqlite3.Connection,
        table: str,
        identity_column: str,
        identity: str,
        values: dict[str, Any],
    ) -> bool:
        row = connection.execute(
            f"SELECT * FROM {table} WHERE {identity_column}=?", (identity,)
        ).fetchone()
        if row is not None:
            if dict(row) != values:
                raise ValueError(f"immutable {table} replay mismatch: {identity}")
            return False
        columns = list(values)
        connection.execute(
            f"INSERT INTO {table}({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(values[column] for column in columns),
        )
        return True

    def ingest(self) -> dict[str, int]:
        compilation_descriptor, compilation = self.sources[0]
        if (
            compilation_descriptor["role"] != "candidate_compilation"
            or compilation.get("seed_count") != 6
            or compilation.get("decision_counts") != {"blocked": 6}
        ):
            raise ValueError("first evidence source is not the exact six-candidate compilation")
        compilation_records = {
            record["seed_id"]: record for record in compilation["candidate_records"]
        }
        if len(compilation_records) != 6:
            raise ValueError("compilation candidate identities are not unique")
        accepted_candidates = accepted_packets = accepted_links = 0
        duplicate_candidates = duplicate_packets = duplicate_links = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for descriptor, _ in self.sources:
                values = {
                    "source_id": descriptor["source_id"],
                    "role": descriptor["role"],
                    "path": descriptor["path"],
                    "file_sha256": descriptor["file_sha256"],
                    "content_sha256": descriptor["content_sha256"],
                }
                self._insert_immutable(
                    connection, "evidence_sources", "source_id", descriptor["source_id"], values
                )
            for seed_id, record in sorted(compilation_records.items()):
                provenance = record["provenance"]
                values = {
                    "seed_id": seed_id,
                    "seed_lineage_sha256": provenance["seed_lineage_sha256"],
                    "family_id": record["family_id"],
                    "family_lineage_sha256": provenance["family_lineage_sha256"],
                    "action_sha256": provenance["action_ir_sha256"],
                    "compilation_provenance_sha256": provenance["binding_sha256"],
                }
                if self._insert_immutable(
                    connection, "candidates", "seed_id", seed_id, values
                ):
                    accepted_candidates += 1
                else:
                    duplicate_candidates += 1

            packets_with_sources: list[tuple[str, dict[str, Any]]] = []
            for descriptor, source in self.sources:
                for record in source["candidate_records"]:
                    seed_id = record["seed_id"]
                    if seed_id not in compilation_records:
                        raise ValueError("prerequisite packet references an unknown candidate")
                    compilation_provenance = compilation_records[seed_id]["provenance"]
                    provenance = record["provenance"]
                    if descriptor["role"] == "formal_prerequisite" and (
                        provenance.get("predecessor_content_sha256")
                        != compilation["content_sha256"]
                        or provenance.get("predecessor_provenance_sha256")
                        != compilation_provenance["binding_sha256"]
                        or provenance.get("action_sha256")
                        != compilation_provenance["action_ir_sha256"]
                    ):
                        raise ValueError("formal prerequisite packet lineage mismatch")
                    packets_with_sources.append(
                        (descriptor["source_id"], _candidate_packet(descriptor["source_id"], record))
                    )
                for control_id, control in sorted(source.get("adapter_results", {}).items()):
                    packets_with_sources.append(
                        (descriptor["source_id"], _calibration_packet(control_id, control))
                    )
                packets_with_sources.extend(
                    (descriptor["source_id"], packet)
                    for packet in _special_control_packets(source)
                )
            if len({packet["packet_id"] for _, packet in packets_with_sources}) > int(
                self.config["budget"]["maximum_packets"]
            ):
                raise ValueError("evidence packet budget exceeded")
            for source_id, packet in packets_with_sources:
                values = {
                    "packet_id": packet["packet_id"],
                    "candidate_id": packet["candidate_id"],
                    "outcome_class": packet["outcome_class"],
                    "calibration_only": int(packet["calibration_only"]),
                    "packet_json": _canonical(packet),
                    "packet_sha256": packet["content_sha256"],
                }
                if self._insert_immutable(
                    connection, "evidence_packets", "packet_id", packet["packet_id"], values
                ):
                    accepted_packets += 1
                else:
                    duplicate_packets += 1
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO source_packet_links VALUES (?,?)",
                    (source_id, packet["packet_id"]),
                )
                if cursor.rowcount:
                    accepted_links += 1
                else:
                    duplicate_links += 1
            connection.commit()
        self._enforce_disk_budget()
        return {
            "accepted_candidates": accepted_candidates,
            "duplicate_candidates": duplicate_candidates,
            "accepted_packets": accepted_packets,
            "duplicate_packets": duplicate_packets,
            "accepted_source_links": accepted_links,
            "duplicate_source_links": duplicate_links,
        }

    def priority_report(self) -> dict[str, Any]:
        with self._connect() as connection:
            candidate_rows = connection.execute(
                "SELECT * FROM candidates ORDER BY seed_id"
            ).fetchall()
            packets = [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT packet_json FROM evidence_packets ORDER BY packet_id"
                )
            ]
            source_link_count = connection.execute(
                "SELECT COUNT(*) FROM source_packet_links"
            ).fetchone()[0]
        candidate_packets: dict[str, list[dict[str, Any]]] = {
            row["seed_id"]: [] for row in candidate_rows
        }
        calibration = []
        for packet in packets:
            if packet["calibration_only"]:
                calibration.append(packet)
            else:
                candidate_packets[packet["candidate_id"]].append(packet)
        rows = []
        excluded = []
        for candidate in candidate_rows:
            seed_id = candidate["seed_id"]
            evidence = candidate_packets[seed_id]
            outcomes = Counter(packet["outcome_class"] for packet in evidence)
            blockers = [
                {**blocker, "source_id": packet["source_id"]}
                for packet in evidence
                for blocker in packet["blocker_taxonomy"]
            ]
            formal_passes = sum(
                gate["outcome_class"] == "pass"
                for packet in evidence
                for gate in packet["gate_outcomes"]
            )
            row = {
                "formula_id": seed_id,
                "family_id": candidate["family_id"],
                "seed_lineage_sha256": candidate["seed_lineage_sha256"],
                "action_sha256": candidate["action_sha256"],
                "candidate_decision": (
                    "reject" if outcomes["reject"] else "blocked" if outcomes["blocked"] else "pass"
                ),
                "formal_pass_count": formal_passes,
                "candidate_evidence_packet_count": len(evidence),
                "source_lineage_depth": len({packet["source_id"] for packet in evidence}),
                "blocker_count": len(blockers),
                "blocker_taxonomy": blockers,
                "evidence_packet_root_sha256": _sha(
                    sorted(packet["content_sha256"] for packet in evidence)
                ),
                "priority_semantics": "multi-axis work ordering only; not probability or truth",
            }
            if row["candidate_decision"] == "reject":
                excluded.append({**row, "exclusion_reason": "terminal candidate-specific reject"})
            else:
                rows.append(row)
        maximum_blockers = max((row["blocker_count"] for row in rows), default=0)
        for row in rows:
            row["blocker_reduction_margin"] = maximum_blockers - row["blocker_count"]
        fronts = pareto_fronts(rows, self.config["priority_axes"])
        queue = []
        for front_index, front in enumerate(fronts, start=1):
            for row in front:
                queue.append({**row, "pareto_front": front_index})
        queue.sort(key=lambda row: (row["pareto_front"], row["formula_id"]))
        calibration_counts = Counter(packet["outcome_class"] for packet in calibration)
        body = {
            "schema_version": REPORT_SCHEMA,
            "source_registry_root_sha256": self.source_registry_root_sha256,
            "candidate_count": len(candidate_rows),
            "candidate_packet_count": sum(len(value) for value in candidate_packets.values()),
            "calibration_packet_count": len(calibration),
            "source_packet_link_count": source_link_count,
            "candidate_decision_counts": dict(
                sorted(Counter(row["candidate_decision"] for row in [*queue, *excluded]).items())
            ),
            "evidence_packet_outcome_counts": dict(
                sorted(Counter(packet["outcome_class"] for packet in packets).items())
            ),
            "calibration_outcome_counts": dict(sorted(calibration_counts.items())),
            "priority_axes": list(self.config["priority_axes"]),
            "pareto_front_count": len(fronts),
            "pareto_follow_up_queue": queue,
            "terminally_excluded": excluded,
            "calibration_control_registry_root_sha256": _sha(
                sorted(packet["content_sha256"] for packet in calibration)
            ),
            "candidate_registry_root_sha256": _sha(
                [dict(row) for row in candidate_rows]
            ),
            "evidence_packet_registry_root_sha256": _sha(
                sorted(packet["content_sha256"] for packet in packets)
            ),
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "interpretation": (
                "Pareto fronts order prerequisite follow-up only. Candidate pass, reject, and block "
                "remain distinct; calibration-only controls cannot enter or promote the queue."
            ),
        }
        return {**body, "content_sha256": _sha(body)}
