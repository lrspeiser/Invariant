from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TASK_STATES = {"queued", "running", "succeeded", "failed", "deferred", "blocked", "cancelled"}
CAMPAIGN_STATES = {"active", "paused", "complete", "budget_exhausted", "failed"}
TERMINAL_TASK_STATES = {"succeeded", "failed", "deferred", "blocked", "cancelled"}


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS campaigns (
  campaign_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  state TEXT NOT NULL,
  config_json TEXT NOT NULL,
  scientific_contract_json TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  updated_utc TEXT NOT NULL,
  started_utc TEXT,
  deadline_utc TEXT,
  max_tasks INTEGER NOT NULL,
  max_failures INTEGER NOT NULL,
  max_cycles INTEGER NOT NULL,
  tasks_started INTEGER NOT NULL DEFAULT 0,
  tasks_succeeded INTEGER NOT NULL DEFAULT 0,
  tasks_failed INTEGER NOT NULL DEFAULT 0,
  cycles_completed INTEGER NOT NULL DEFAULT 0,
  stop_reason TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  parent_candidate_id TEXT REFERENCES candidates(candidate_id),
  family_id TEXT,
  kind TEXT NOT NULL,
  expression TEXT NOT NULL,
  canonical_json TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  generation INTEGER NOT NULL,
  pareto_front INTEGER,
  mechanism_tags_json TEXT NOT NULL,
  status TEXT NOT NULL,
  hard_gate_status TEXT NOT NULL,
  exclusion_reason TEXT,
  created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  candidate_id TEXT REFERENCES candidates(candidate_id),
  task_type TEXT NOT NULL,
  stage INTEGER NOT NULL,
  status TEXT NOT NULL,
  priority REAL NOT NULL,
  diversity_bucket TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL,
  leased_by TEXT,
  lease_expires_utc TEXT,
  heartbeat_utc TEXT,
  not_before_utc TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  payload_json TEXT NOT NULL,
  result_json TEXT,
  error_text TEXT,
  created_utc TEXT NOT NULL,
  started_utc TEXT,
  completed_utc TEXT
);
CREATE TABLE IF NOT EXISTS task_dependencies (
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  depends_on_task_id TEXT NOT NULL REFERENCES tasks(task_id),
  PRIMARY KEY(task_id, depends_on_task_id)
);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
  task_id TEXT REFERENCES tasks(task_id),
  gate_id TEXT NOT NULL,
  gate_version TEXT NOT NULL,
  stage INTEGER NOT NULL,
  is_hard INTEGER NOT NULL,
  outcome TEXT NOT NULL,
  margin REAL,
  units TEXT,
  evidence_class TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  UNIQUE(candidate_id, gate_id, gate_version, task_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  candidate_id TEXT REFERENCES candidates(candidate_id),
  task_id TEXT REFERENCES tasks(task_id),
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  metadata_json TEXT NOT NULL,
  created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
  proposal_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  source_task_id TEXT REFERENCES tasks(task_id),
  parent_candidate_id TEXT REFERENCES candidates(candidate_id),
  status TEXT NOT NULL,
  proposal_json TEXT NOT NULL,
  validation_json TEXT NOT NULL,
  created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failure_clusters (
  cluster_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  gate_id TEXT NOT NULL,
  mechanism_tag TEXT NOT NULL,
  rejection_count INTEGER NOT NULL,
  candidate_ids_json TEXT NOT NULL,
  summary TEXT NOT NULL,
  created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  event_type TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_budgets (
  campaign_id TEXT PRIMARY KEY REFERENCES campaigns(campaign_id),
  limit_microusd INTEGER NOT NULL,
  reserved_microusd INTEGER NOT NULL DEFAULT 0,
  spent_microusd INTEGER NOT NULL DEFAULT 0,
  max_calls INTEGER NOT NULL,
  calls_started INTEGER NOT NULL DEFAULT 0,
  calls_completed INTEGER NOT NULL DEFAULT 0,
  updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
  call_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
  task_id TEXT REFERENCES tasks(task_id),
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  reserved_microusd INTEGER NOT NULL,
  actual_microusd INTEGER,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  completed_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(campaign_id, status, not_before_utc, priority);
CREATE INDEX IF NOT EXISTS idx_tasks_candidate ON tasks(candidate_id, stage, status);
CREATE INDEX IF NOT EXISTS idx_evidence_candidate ON evidence(candidate_id, stage, outcome);
CREATE INDEX IF NOT EXISTS idx_candidates_family ON candidates(campaign_id, family_id, status);
CREATE INDEX IF NOT EXISTS idx_llm_calls_campaign ON llm_calls(campaign_id, created_utc);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:length]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ClaimedTask:
    task_id: str
    campaign_id: str
    candidate_id: str | None
    task_type: str
    stage: int
    attempt: int
    max_attempts: int
    payload: dict[str, Any]


class CampaignStore:
    def __init__(self, database: str | Path):
        self.database = Path(database).resolve()

    def initialize(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def event(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO events(campaign_id,event_type,entity_type,entity_id,payload_json,created_utc) "
            "VALUES (?,?,?,?,?,?)",
            (
                campaign_id,
                event_type,
                entity_type,
                entity_id,
                canonical_json(payload or {}),
                utc_now(),
            ),
        )

    def create_campaign(self, config: dict[str, Any]) -> str:
        now = utc_now()
        identity = canonical_json({"name": config["name"], "created_utc": now, "config": config})
        campaign_id = stable_id("CMP", identity)
        budget = config["budget"]
        duration_days = float(budget.get("duration_days", 14))
        deadline = (datetime.now(UTC) + timedelta(days=duration_days)).isoformat()
        contract = config["scientific_contract"]
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO campaigns VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    campaign_id,
                    config["name"],
                    "active",
                    canonical_json(config),
                    canonical_json(contract),
                    now,
                    now,
                    now,
                    deadline,
                    int(budget.get("max_tasks", 100_000)),
                    int(budget.get("max_failures", 10_000)),
                    int(budget.get("max_cycles", 336)),
                    0,
                    0,
                    0,
                    0,
                    None,
                ),
            )
            self.event(connection, campaign_id, "campaign_created", "campaign", campaign_id, config)
            llm = config.get("llm", {})
            if "total_budget_usd" in llm:
                limit = self._usd_to_microusd(llm["total_budget_usd"])
                max_calls = int(llm.get("max_calls", 0))
                if limit <= 0 or max_calls <= 0:
                    raise ValueError("LLM total_budget_usd and max_calls must be positive")
                connection.execute(
                    "INSERT INTO llm_budgets VALUES (?,?,?,?,?,?,?,?)",
                    (campaign_id, limit, 0, 0, max_calls, 0, 0, now),
                )
        return campaign_id

    @staticmethod
    def _usd_to_microusd(value: float | str) -> int:
        amount = float(value)
        if amount < 0:
            raise ValueError("USD amount cannot be negative")
        return round(amount * 1_000_000)

    @staticmethod
    def _microusd_to_usd(value: int) -> float:
        return round(value / 1_000_000, 6)

    def configure_llm_budget(
        self, campaign_id: str, *, total_budget_usd: float, max_calls: int
    ) -> dict[str, Any]:
        """Create a budget, or tighten an unused budget. Never silently raises a live cap."""
        limit = self._usd_to_microusd(total_budget_usd)
        if limit <= 0 or max_calls <= 0:
            raise ValueError("LLM total budget and max_calls must be positive")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM llm_budgets WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO llm_budgets VALUES (?,?,?,?,?,?,?,?)",
                    (campaign_id, limit, 0, 0, max_calls, 0, 0, now),
                )
            else:
                active = row["spent_microusd"] or row["reserved_microusd"] or row["calls_started"]
                if active and (limit > row["limit_microusd"] or max_calls > row["max_calls"]):
                    raise ValueError("Refusing to raise an active LLM budget without a new campaign")
                if limit < row["spent_microusd"] + row["reserved_microusd"]:
                    raise ValueError("New LLM budget is below already spent plus reserved usage")
                if max_calls < row["calls_started"]:
                    raise ValueError("New LLM call limit is below calls already started")
                connection.execute(
                    "UPDATE llm_budgets SET limit_microusd=?,max_calls=?,updated_utc=? "
                    "WHERE campaign_id=?",
                    (limit, max_calls, now, campaign_id),
                )
            self.event(
                connection,
                campaign_id,
                "llm_budget_configured",
                "campaign",
                campaign_id,
                {"total_budget_usd": total_budget_usd, "max_calls": max_calls},
            )
        return self.llm_budget_status(campaign_id)

    def reserve_llm_call(
        self,
        campaign_id: str,
        *,
        task_id: str | None,
        provider: str,
        model: str,
        max_cost_usd: float,
    ) -> str:
        reservation = self._usd_to_microusd(max_cost_usd)
        if reservation <= 0:
            raise ValueError("LLM per-call reservation must be positive")
        now = utc_now()
        call_id = stable_id("LLMCALL", campaign_id, task_id or "manual", provider, model, now)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            budget = connection.execute(
                "SELECT * FROM llm_budgets WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
            if budget is None:
                raise ValueError("LLM budget is not configured")
            if budget["calls_started"] >= budget["max_calls"]:
                raise ValueError("LLM call-count budget exhausted")
            available = (
                budget["limit_microusd"]
                - budget["spent_microusd"]
                - budget["reserved_microusd"]
            )
            if reservation > available:
                raise ValueError("LLM dollar budget cannot cover the requested call reservation")
            connection.execute(
                "UPDATE llm_budgets SET reserved_microusd=reserved_microusd+?,"
                "calls_started=calls_started+1,updated_utc=? WHERE campaign_id=?",
                (reservation, now, campaign_id),
            )
            connection.execute(
                "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    campaign_id,
                    task_id,
                    provider,
                    model,
                    reservation,
                    None,
                    "reserved",
                    canonical_json({}),
                    now,
                    None,
                ),
            )
            self.event(
                connection,
                campaign_id,
                "llm_call_reserved",
                "llm_call",
                call_id,
                {"max_cost_usd": max_cost_usd, "provider": provider, "model": model},
            )
        return call_id

    def settle_llm_call(
        self,
        call_id: str,
        *,
        actual_cost_usd: float | None,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Settle once. Unknown cost is charged at the full reservation (fail closed)."""
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            call = connection.execute(
                "SELECT * FROM llm_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            if call is None:
                raise ValueError(f"Unknown LLM call: {call_id}")
            if call["status"] == "reserved":
                actual = (
                    call["reserved_microusd"]
                    if actual_cost_usd is None
                    else self._usd_to_microusd(actual_cost_usd)
                )
                actual = max(actual, 0)
                connection.execute(
                    "UPDATE llm_budgets SET reserved_microusd=reserved_microusd-?,"
                    "spent_microusd=spent_microusd+?,calls_completed=calls_completed+1,"
                    "updated_utc=? WHERE campaign_id=?",
                    (call["reserved_microusd"], actual, now, call["campaign_id"]),
                )
                connection.execute(
                    "UPDATE llm_calls SET actual_microusd=?,status=?,metadata_json=?,completed_utc=? "
                    "WHERE call_id=?",
                    (actual, status, canonical_json(metadata or {}), now, call_id),
                )
                self.event(
                    connection,
                    call["campaign_id"],
                    "llm_call_settled",
                    "llm_call",
                    call_id,
                    {"actual_cost_usd": self._microusd_to_usd(actual), "status": status},
                )
            campaign_id = call["campaign_id"]
        return self.llm_budget_status(campaign_id)

    def llm_budget_status(self, campaign_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM llm_budgets WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "limit_usd": self._microusd_to_usd(row["limit_microusd"]),
            "spent_usd": self._microusd_to_usd(row["spent_microusd"]),
            "reserved_usd": self._microusd_to_usd(row["reserved_microusd"]),
            "remaining_usd": self._microusd_to_usd(
                row["limit_microusd"] - row["spent_microusd"] - row["reserved_microusd"]
            ),
            "max_calls": row["max_calls"],
            "calls_started": row["calls_started"],
            "calls_completed": row["calls_completed"],
        }

    def campaign(self, campaign_id: str | None = None) -> sqlite3.Row:
        with self.connect() as connection:
            if campaign_id:
                row = connection.execute(
                    "SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM campaigns ORDER BY created_utc DESC LIMIT 1"
                ).fetchone()
        if row is None:
            raise ValueError("No campaign found")
        return row

    def configure_llm_runtime(
        self, campaign_id: str, llm_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist an explicit, auditable LLM adapter configuration for one campaign."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT config_json FROM campaigns WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown campaign: {campaign_id}")
            config = json.loads(row["config_json"])
            config["llm"] = llm_config
            connection.execute(
                "UPDATE campaigns SET config_json=?,updated_utc=? WHERE campaign_id=?",
                (canonical_json(config), utc_now(), campaign_id),
            )
            self.event(
                connection,
                campaign_id,
                "llm_runtime_configured",
                "campaign",
                campaign_id,
                {
                    "provider": llm_config.get("provider"),
                    "model": llm_config.get("model"),
                    "per_call_budget_usd": llm_config.get("per_call_budget_usd"),
                    "total_budget_usd": llm_config.get("total_budget_usd"),
                    "max_calls": llm_config.get("max_calls"),
                    "enabled": bool(llm_config.get("command")),
                },
            )
        return llm_config

    def add_candidate(
        self,
        campaign_id: str,
        *,
        kind: str,
        expression: str,
        canonical: dict[str, Any],
        family_id: str | None = None,
        parent_candidate_id: str | None = None,
        generation: int = 0,
        pareto_front: int | None = None,
        mechanism_tags: list[str] | None = None,
    ) -> str:
        canonical_payload = canonical_json(canonical)
        content_sha = hashlib.sha256(canonical_payload.encode()).hexdigest()
        candidate_id = stable_id("CAND", campaign_id, kind, content_sha)
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    candidate_id,
                    campaign_id,
                    parent_candidate_id,
                    family_id,
                    kind,
                    expression,
                    canonical_payload,
                    content_sha,
                    generation,
                    pareto_front,
                    canonical_json(sorted(mechanism_tags or [])),
                    "active",
                    "unresolved",
                    None,
                    now,
                ),
            )
            self.event(
                connection,
                campaign_id,
                "candidate_registered",
                "candidate",
                candidate_id,
                {"kind": kind, "family_id": family_id, "parent": parent_candidate_id},
            )
        return candidate_id

    def add_task(
        self,
        campaign_id: str,
        task_type: str,
        *,
        stage: int,
        payload: dict[str, Any],
        candidate_id: str | None = None,
        priority: float = 0.0,
        diversity_bucket: str = "global",
        max_attempts: int = 3,
        depends_on: list[str] | None = None,
        not_before_utc: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        key = idempotency_key or canonical_json(
            {
                "campaign": campaign_id,
                "candidate": candidate_id,
                "type": task_type,
                "payload": payload,
            }
        )
        task_id = stable_id("TASK", key)
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    campaign_id,
                    candidate_id,
                    task_type,
                    stage,
                    "queued",
                    priority,
                    diversity_bucket,
                    0,
                    max_attempts,
                    None,
                    None,
                    None,
                    not_before_utc or now,
                    key,
                    canonical_json(payload),
                    None,
                    None,
                    now,
                    None,
                    None,
                ),
            )
            for dependency in depends_on or []:
                connection.execute(
                    "INSERT OR IGNORE INTO task_dependencies VALUES (?,?)", (task_id, dependency)
                )
            self.event(
                connection,
                campaign_id,
                "task_queued",
                "task",
                task_id,
                {"type": task_type, "candidate_id": candidate_id},
            )
        return task_id

    def record_evidence(
        self,
        campaign_id: str,
        candidate_id: str,
        task_id: str | None,
        evidence: dict[str, Any],
    ) -> str:
        version = str(evidence.get("gate_version", "1.0"))
        gate_id = evidence["gate_id"]
        evidence_id = stable_id("EVD", candidate_id, gate_id, version, task_id)
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    evidence_id,
                    campaign_id,
                    candidate_id,
                    task_id,
                    gate_id,
                    version,
                    int(evidence["stage"]),
                    int(bool(evidence.get("is_hard", True))),
                    evidence["outcome"],
                    evidence.get("margin"),
                    evidence.get("units"),
                    evidence.get("evidence_class", "theory"),
                    canonical_json(evidence.get("payload", {})),
                    utc_now(),
                ),
            )
            if evidence.get("is_hard", True) and evidence["outcome"] == "reject":
                reason = f"{gate_id}@{version}"
                connection.execute(
                    "UPDATE candidates SET status='rejected',hard_gate_status='rejected',"
                    "exclusion_reason=? WHERE candidate_id=?",
                    (reason, candidate_id),
                )
                connection.execute(
                    "UPDATE tasks SET status='cancelled',completed_utc=?,error_text=? "
                    "WHERE candidate_id=? AND status='queued' AND task_type!='candidate_dossier'",
                    (utc_now(), f"terminal hard-gate rejection: {reason}", candidate_id),
                )
            elif evidence.get("is_hard", True) and evidence["outcome"] == "pass":
                connection.execute(
                    "UPDATE candidates SET hard_gate_status='passing_completed_gates' "
                    "WHERE candidate_id=? AND hard_gate_status='unresolved'",
                    (candidate_id,),
                )
            self.event(
                connection,
                campaign_id,
                "evidence_recorded",
                "candidate",
                candidate_id,
                {"gate_id": gate_id, "outcome": evidence["outcome"]},
            )
        return evidence_id

    def register_artifact(
        self,
        campaign_id: str,
        path: str | Path,
        *,
        kind: str,
        task_id: str | None = None,
        candidate_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        known_sha256: str | None = None,
    ) -> str:
        path = Path(path).resolve()
        digest = known_sha256 or sha256_file(path)
        artifact_id = stable_id("ART", campaign_id, str(path), digest)
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    campaign_id,
                    candidate_id,
                    task_id,
                    kind,
                    str(path),
                    digest,
                    path.stat().st_size,
                    canonical_json(metadata or {}),
                    utc_now(),
                ),
            )
            self.event(
                connection,
                campaign_id,
                "artifact_registered",
                "artifact",
                artifact_id,
                {"path": str(path), "kind": kind},
            )
        return artifact_id

    def recover_expired_leases(self, campaign_id: str) -> dict[str, int]:
        now = utc_now()
        recovered = 0
        failed = 0
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE campaign_id=? AND status='running' "
                "AND lease_expires_utc < ?",
                (campaign_id, now),
            ).fetchall()
            for row in rows:
                if row["attempt"] >= row["max_attempts"]:
                    connection.execute(
                        "UPDATE tasks SET status='failed',completed_utc=?,error_text=?,leased_by=NULL,"
                        "lease_expires_utc=NULL WHERE task_id=?",
                        (now, "lease expired after maximum attempts", row["task_id"]),
                    )
                    failed += 1
                else:
                    connection.execute(
                        "UPDATE tasks SET status='queued',leased_by=NULL,lease_expires_utc=NULL,"
                        "heartbeat_utc=NULL,error_text=? WHERE task_id=?",
                        ("recovered expired worker lease", row["task_id"]),
                    )
                    recovered += 1
                self.event(
                    connection,
                    campaign_id,
                    "lease_recovered",
                    "task",
                    row["task_id"],
                    {"new_state": "failed" if row["attempt"] >= row["max_attempts"] else "queued"},
                )
        return {"recovered": recovered, "failed": failed}

    def requeue_cancelled_dossiers(self, campaign_id: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status='queued',completed_utc=NULL,error_text=NULL,not_before_utc=? "
                "WHERE campaign_id=? AND task_type='candidate_dossier' AND status='cancelled'",
                (utc_now(), campaign_id),
            )
        return cursor.rowcount

    def reconcile_candidate_states(self, campaign_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE candidates SET status='deferred' WHERE campaign_id=? AND status!='rejected' "
                "AND EXISTS (SELECT 1 FROM tasks t WHERE t.candidate_id=candidates.candidate_id "
                "AND t.status='deferred')",
                (campaign_id,),
            )

    def _enforce_budget(self, connection: sqlite3.Connection, campaign: sqlite3.Row) -> bool:
        now = datetime.now(UTC)
        reason = None
        if campaign["tasks_started"] >= campaign["max_tasks"]:
            reason = "maximum task budget reached"
        elif campaign["tasks_failed"] >= campaign["max_failures"]:
            reason = "maximum failure budget reached"
        elif campaign["deadline_utc"] and now >= datetime.fromisoformat(campaign["deadline_utc"]):
            reason = "campaign deadline reached"
        if reason:
            connection.execute(
                "UPDATE campaigns SET state='budget_exhausted',stop_reason=?,updated_utc=? "
                "WHERE campaign_id=?",
                (reason, utc_now(), campaign["campaign_id"]),
            )
            return False
        return campaign["state"] == "active"

    def claim_task(
        self,
        campaign_id: str,
        worker_id: str,
        lease_seconds: int,
        allowed_task_types: set[str] | None = None,
    ) -> ClaimedTask | None:
        if allowed_task_types is not None and not allowed_task_types:
            return None
        connection = sqlite3.connect(self.database, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            campaign = connection.execute(
                "SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
            if campaign is None or not self._enforce_budget(connection, campaign):
                connection.commit()
                return None
            connection.execute(
                "UPDATE tasks SET status='blocked',completed_utc=?,error_text='dependency did not succeed' "
                "WHERE campaign_id=? AND status='queued' AND EXISTS ("
                "SELECT 1 FROM task_dependencies d JOIN tasks p ON p.task_id=d.depends_on_task_id "
                "WHERE d.task_id=tasks.task_id AND p.status IN ('failed','blocked','cancelled','deferred'))",
                (utc_now(), campaign_id),
            )
            type_filter = ""
            parameters: list[Any] = [campaign_id, utc_now()]
            if allowed_task_types is not None:
                ordered_types = sorted(allowed_task_types)
                placeholders = ",".join("?" for _ in ordered_types)
                type_filter = f"AND t.task_type IN ({placeholders}) "
                parameters.extend(ordered_types)
            row = connection.execute(
                "SELECT t.* FROM tasks t WHERE t.campaign_id=? AND t.status='queued' "
                "AND t.not_before_utc<=? "
                + type_filter
                + "AND NOT EXISTS ("
                "SELECT 1 FROM task_dependencies d JOIN tasks p ON p.task_id=d.depends_on_task_id "
                "WHERE d.task_id=t.task_id AND p.status!='succeeded') "
                "ORDER BY t.priority - 0.01*(SELECT COUNT(*) FROM tasks f WHERE "
                "f.campaign_id=t.campaign_id AND f.diversity_bucket=t.diversity_bucket "
                "AND f.status IN ('running','succeeded')) DESC, t.stage, t.created_utc LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            now = datetime.now(UTC)
            lease_expires = (now + timedelta(seconds=lease_seconds)).isoformat()
            connection.execute(
                "UPDATE tasks SET status='running',attempt=attempt+1,leased_by=?,lease_expires_utc=?,"
                "heartbeat_utc=?,started_utc=COALESCE(started_utc,?) WHERE task_id=?",
                (worker_id, lease_expires, now.isoformat(), now.isoformat(), row["task_id"]),
            )
            connection.execute(
                "UPDATE campaigns SET tasks_started=tasks_started+1,updated_utc=? WHERE campaign_id=?",
                (now.isoformat(), campaign_id),
            )
            connection.execute(
                "INSERT INTO events(campaign_id,event_type,entity_type,entity_id,payload_json,created_utc) "
                "VALUES (?,?,?,?,?,?)",
                (
                    campaign_id,
                    "task_leased",
                    "task",
                    row["task_id"],
                    canonical_json({"worker_id": worker_id, "lease_expires": lease_expires}),
                    now.isoformat(),
                ),
            )
            connection.commit()
            return ClaimedTask(
                task_id=row["task_id"],
                campaign_id=campaign_id,
                candidate_id=row["candidate_id"],
                task_type=row["task_type"],
                stage=row["stage"],
                attempt=row["attempt"] + 1,
                max_attempts=row["max_attempts"],
                payload=json.loads(row["payload_json"]),
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat(self, task_id: str, worker_id: str, lease_seconds: int) -> bool:
        now = datetime.now(UTC)
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET heartbeat_utc=?,lease_expires_utc=? WHERE task_id=? "
                "AND status='running' AND leased_by=?",
                (
                    now.isoformat(),
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    task_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def finish_task(
        self,
        task: ClaimedTask,
        worker_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if status not in TERMINAL_TASK_STATES:
            raise ValueError(f"Invalid terminal task status: {status}")
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status=?,result_json=?,error_text=?,completed_utc=?,leased_by=NULL,"
                "lease_expires_utc=NULL,heartbeat_utc=NULL WHERE task_id=? AND status='running' "
                "AND leased_by=?",
                (status, canonical_json(result or {}), error, now, task.task_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Task lease lost before completion: {task.task_id}")
            if status == "deferred" and task.candidate_id:
                connection.execute(
                    "UPDATE candidates SET status='deferred' WHERE candidate_id=? "
                    "AND status!='rejected'",
                    (task.candidate_id,),
                )
            success_increment = 1 if status == "succeeded" else 0
            failure_increment = 1 if status == "failed" else 0
            connection.execute(
                "UPDATE campaigns SET tasks_succeeded=tasks_succeeded+?,tasks_failed=tasks_failed+?,"
                "updated_utc=? WHERE campaign_id=?",
                (success_increment, failure_increment, now, task.campaign_id),
            )
            self.event(
                connection,
                task.campaign_id,
                "task_completed",
                "task",
                task.task_id,
                {"status": status, "error": error},
            )

    def retry_or_fail_task(
        self,
        task: ClaimedTask,
        worker_id: str,
        error: str,
        backoff_seconds: int = 5,
    ) -> str:
        now = datetime.now(UTC)
        retry = task.attempt < task.max_attempts
        new_status = "queued" if retry else "failed"
        not_before = (now + timedelta(seconds=backoff_seconds * task.attempt)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status=?,error_text=?,completed_utc=?,not_before_utc=?,"
                "leased_by=NULL,lease_expires_utc=NULL,heartbeat_utc=NULL WHERE task_id=? "
                "AND status='running' AND leased_by=?",
                (
                    new_status,
                    error,
                    None if retry else now.isoformat(),
                    not_before,
                    task.task_id,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Task lease lost during failure handling: {task.task_id}")
            if not retry:
                connection.execute(
                    "UPDATE campaigns SET tasks_failed=tasks_failed+1,updated_utc=? "
                    "WHERE campaign_id=?",
                    (now.isoformat(), task.campaign_id),
                )
            self.event(
                connection,
                task.campaign_id,
                "task_retry_scheduled" if retry else "task_failed",
                "task",
                task.task_id,
                {"attempt": task.attempt, "error": error, "not_before": not_before},
            )
        return new_status

    def set_campaign_state(self, campaign_id: str, state: str, reason: str | None = None) -> None:
        if state not in CAMPAIGN_STATES:
            raise ValueError(f"Invalid campaign state: {state}")
        with self.connect() as connection:
            connection.execute(
                "UPDATE campaigns SET state=?,stop_reason=?,updated_utc=? WHERE campaign_id=?",
                (state, reason, utc_now(), campaign_id),
            )
            self.event(
                connection,
                campaign_id,
                f"campaign_{state}",
                "campaign",
                campaign_id,
                {"reason": reason},
            )

    def candidate(self, candidate_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown candidate: {candidate_id}")
        return row

    def task_counts(self, campaign_id: str) -> dict[str, int]:
        with self.connect() as connection:
            return dict(
                connection.execute(
                    "SELECT status,COUNT(*) FROM tasks WHERE campaign_id=? GROUP BY status",
                    (campaign_id,),
                ).fetchall()
            )

    def status(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.campaign(campaign_id)
        with self.connect() as connection:
            candidate_counts = dict(
                connection.execute(
                    "SELECT status,COUNT(*) FROM candidates WHERE campaign_id=? GROUP BY status",
                    (campaign_id,),
                ).fetchall()
            )
            hard_gates = dict(
                connection.execute(
                    "SELECT outcome,COUNT(*) FROM evidence WHERE campaign_id=? AND is_hard=1 "
                    "GROUP BY outcome",
                    (campaign_id,),
                ).fetchall()
            )
            latest = [
                dict(row)
                for row in connection.execute(
                    "SELECT event_type,entity_type,entity_id,created_utc FROM events "
                    "WHERE campaign_id=? ORDER BY event_id DESC LIMIT 10",
                    (campaign_id,),
                ).fetchall()
            ]
        return {
            "campaign_id": campaign_id,
            "name": campaign["name"],
            "state": campaign["state"],
            "deadline_utc": campaign["deadline_utc"],
            "stop_reason": campaign["stop_reason"],
            "task_counts": self.task_counts(campaign_id),
            "candidate_counts": candidate_counts,
            "hard_gate_evidence": hard_gates,
            "budget": {
                "tasks_started": campaign["tasks_started"],
                "max_tasks": campaign["max_tasks"],
                "tasks_failed": campaign["tasks_failed"],
                "max_failures": campaign["max_failures"],
                "cycles_completed": campaign["cycles_completed"],
                "max_cycles": campaign["max_cycles"],
                "llm": self.llm_budget_status(campaign_id),
            },
            "latest_events": latest,
        }

    def gate_summary(self, candidate_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                {**dict(row), "payload": json.loads(row["payload_json"])}
                for row in connection.execute(
                    "SELECT gate_id,gate_version,stage,is_hard,outcome,margin,units,evidence_class,"
                    "payload_json,created_utc FROM evidence WHERE candidate_id=? ORDER BY stage,created_utc",
                    (candidate_id,),
                ).fetchall()
            ]

    def candidate_tasks(self, candidate_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT task_id,task_type,stage,status,attempt,error_text,created_utc,completed_utc "
                    "FROM tasks WHERE candidate_id=? ORDER BY stage,created_utc",
                    (candidate_id,),
                ).fetchall()
            ]

    def build_failure_clusters(self, campaign_id: str) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT e.gate_id,c.mechanism_tags_json,c.candidate_id FROM evidence e "
                "JOIN candidates c ON c.candidate_id=e.candidate_id "
                "WHERE e.campaign_id=? AND e.is_hard=1 AND e.outcome='reject'",
                (campaign_id,),
            ).fetchall()
            groups: dict[tuple[str, str], list[str]] = {}
            for row in rows:
                tags = json.loads(row["mechanism_tags_json"]) or ["untagged"]
                for tag in tags:
                    groups.setdefault((row["gate_id"], tag), []).append(row["candidate_id"])
            connection.execute("DELETE FROM failure_clusters WHERE campaign_id=?", (campaign_id,))
            for (gate, tag), candidate_ids in sorted(groups.items()):
                cluster_id = stable_id("FCL", campaign_id, gate, tag)
                connection.execute(
                    "INSERT INTO failure_clusters VALUES (?,?,?,?,?,?,?,?)",
                    (
                        cluster_id,
                        campaign_id,
                        gate,
                        tag,
                        len(candidate_ids),
                        canonical_json(sorted(candidate_ids)),
                        f"{len(candidate_ids)} exact candidates rejected at {gate} with mechanism {tag}.",
                        utc_now(),
                    ),
                )
        return len(groups)

    def unresolved_claims(self, candidate_id: str) -> list[str]:
        completed = {
            row["gate_id"] for row in self.gate_summary(candidate_id) if row["outcome"] == "pass"
        }
        ladder = [
            "covariant_action_complete",
            "covariant_variation",
            "kinetic_rank",
            "hamiltonian_boundedness",
            "constraint_algebra",
            "physical_degree_count",
            "characteristic_cones",
            "covariance_identity",
            "gr_limit",
            "solar_system_controls",
            "audited_measurement_holdout",
        ]
        return [gate for gate in ladder if gate not in completed]

    def integrity_check(self) -> str:
        with self.connect() as connection:
            return connection.execute("PRAGMA integrity_check").fetchone()[0]
