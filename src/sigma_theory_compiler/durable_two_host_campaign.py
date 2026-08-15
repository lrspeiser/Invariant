"""Fenced two-logical-host durable campaign with bounded SQLite-family storage.

This is an operational durability layer, not a scientific evaluator.  Two independent host
processes may share one local SQLite/WAL store.  Host sessions and work leases are heartbeat-bound,
dead sessions are fenced before their work is recovered, and every mutation is chained into a
tamper-evident event ledger.  A six-hour receipt is available only after real monotonic runtime has
been accumulated by cleanly closed sessions; this module ships no precomputed duration receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-durable-two-host-campaign-config-1.0"
DATABASE_SCHEMA = "sigma-durable-two-host-campaign-database-1.0"
STATUS_SCHEMA = "sigma-durable-two-host-campaign-status-1.0"
RECEIPT_SCHEMA = "sigma-durable-two-host-campaign-duration-receipt-1.0"
EVENT_GENESIS = "0" * 64
DATABASE_NAME = "campaign.sqlite"
MAX_CONFIG_BYTES = 256 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024

_HEX = __import__("re").compile(r"[0-9a-f]{64}\Z")
_CONFIG_KEYS = {
    "campaign_id",
    "duration",
    "evaluator",
    "logical_hosts",
    "queue",
    "schema_version",
    "storage",
}
_QUEUE_KEYS = {
    "dead_host_seconds",
    "lease_seconds",
    "maximum_attempts",
    "maximum_payload_bytes",
    "maximum_pending_work",
    "maximum_result_bytes",
}
_STORAGE_KEYS = {
    "busy_timeout_ms",
    "maximum_sqlite_family_bytes",
    "maximum_transaction_reserve_bytes",
    "page_size_bytes",
    "wal_autocheckpoint_pages",
}
_DURATION_KEYS = {
    "default_run_slice_seconds",
    "heartbeat_interval_seconds",
    "maximum_run_slice_seconds",
    "minimum_each_host_credited_seconds",
    "poll_interval_seconds",
    "required_credited_seconds",
}


class DurableCampaignError(ValueError):
    """Base error for invalid campaign state or operations."""


class StorageCeilingError(DurableCampaignError):
    """Raised before a write that cannot preserve the configured SQLite-family ceiling."""


class DurationNotReachedError(DurableCampaignError):
    """Raised when callers attempt to promote a duration receipt prematurely."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise DurableCampaignError("campaign timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the one public production contract without a fast-duration escape hatch."""

    if set(config) != _CONFIG_KEYS or config.get("schema_version") != CONFIG_SCHEMA:
        raise DurableCampaignError("durable campaign config schema changed")
    campaign_id = config.get("campaign_id")
    hosts = config.get("logical_hosts")
    queue = config.get("queue")
    storage = config.get("storage")
    duration = config.get("duration")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or not isinstance(hosts, list)
        or len(hosts) != 2
        or len(set(hosts)) != 2
        or any(not isinstance(host, str) or not host or len(host) > 64 for host in hosts)
        or config.get("evaluator") != "sha256_payload_v1"
        or not isinstance(queue, Mapping)
        or set(queue) != _QUEUE_KEYS
        or not isinstance(storage, Mapping)
        or set(storage) != _STORAGE_KEYS
        or not isinstance(duration, Mapping)
        or set(duration) != _DURATION_KEYS
    ):
        raise DurableCampaignError("durable campaign config semantics changed")
    if (
        not 1 <= int(queue["maximum_attempts"]) <= 100
        or not 1 <= int(queue["maximum_pending_work"]) <= 10_000_000
        or not 1 <= float(queue["lease_seconds"]) <= 3600
        or not float(queue["lease_seconds"]) < float(queue["dead_host_seconds"]) <= 7200
        or not 256 <= int(queue["maximum_payload_bytes"]) <= 16 * 1024 * 1024
        or not 256 <= int(queue["maximum_result_bytes"]) <= 16 * 1024 * 1024
    ):
        raise DurableCampaignError("durable campaign queue bounds changed")
    maximum = int(storage["maximum_sqlite_family_bytes"])
    reserve = int(storage["maximum_transaction_reserve_bytes"])
    page_size = int(storage["page_size_bytes"])
    if (
        not 256 * 1024 <= maximum <= 64 * 1024**3
        or not 16 * 1024 <= reserve < maximum // 2
        or reserve
        < max(int(queue["maximum_payload_bytes"]), int(queue["maximum_result_bytes"])) + 64 * 1024
        or page_size not in {4096, 8192, 16384, 32768, 65536}
        or not 1 <= int(storage["wal_autocheckpoint_pages"]) <= 65536
        or not 100 <= int(storage["busy_timeout_ms"]) <= 300_000
        or maximum - reserve - 64 * 1024 < 16 * page_size
    ):
        raise DurableCampaignError("durable campaign SQLite-family bounds changed")
    required = float(duration["required_credited_seconds"])
    per_host = float(duration["minimum_each_host_credited_seconds"])
    maximum_slice = float(duration["maximum_run_slice_seconds"])
    if (
        required < 6 * 60 * 60
        or not 1 <= per_host <= required
        or not 0
        < float(duration["heartbeat_interval_seconds"])
        < min(float(queue["lease_seconds"]), float(queue["dead_host_seconds"])) / 2
        or not 0 < float(duration["poll_interval_seconds"]) <= 60
        or not 0 < float(duration["default_run_slice_seconds"]) <= maximum_slice
        or maximum_slice < required
        or maximum_slice > 14 * 24 * 60 * 60
    ):
        raise DurableCampaignError("durable campaign real-duration contract changed")


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > MAX_CONFIG_BYTES:
            raise DurableCampaignError("durable campaign config byte budget violated")
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DurableCampaignError("durable campaign config is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise DurableCampaignError("durable campaign config must be one JSON object")
    validate_config(value)
    return value


SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign_meta (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  campaign_id TEXT NOT NULL,
  config_json TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  storage_breaches INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS hosts (
  host_id TEXT PRIMARY KEY,
  generation INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('active','stopped','dead')),
  registered_utc TEXT NOT NULL,
  last_heartbeat_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  host_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('active','stopped','dead')),
  started_utc TEXT NOT NULL,
  last_heartbeat_utc TEXT NOT NULL,
  ended_utc TEXT,
  monotonic_elapsed_seconds REAL,
  credited_seconds REAL,
  stop_reason TEXT,
  FOREIGN KEY(host_id) REFERENCES hosts(host_id)
);
CREATE TABLE IF NOT EXISTS work (
  work_id TEXT PRIMARY KEY,
  ordinal INTEGER NOT NULL UNIQUE,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed')),
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL,
  leased_host_id TEXT,
  leased_session_id TEXT,
  lease_token TEXT,
  lease_expires_utc TEXT,
  result_json TEXT,
  error_text TEXT,
  created_utc TEXT NOT NULL,
  completed_utc TEXT
);
CREATE INDEX IF NOT EXISTS work_claim_order ON work(state,ordinal);
CREATE TABLE IF NOT EXISTS events (
  sequence INTEGER PRIMARY KEY,
  created_utc TEXT NOT NULL,
  event_type TEXT NOT NULL,
  host_id TEXT,
  work_id TEXT,
  payload_json TEXT NOT NULL,
  previous_sha256 TEXT NOT NULL,
  event_sha256 TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class DurableLease:
    work_id: str
    ordinal: int
    attempt: int
    max_attempts: int
    host_id: str
    session_id: str
    lease_token: str
    payload: dict[str, Any]


class DurableTwoHostCampaign:
    """Two-host SQLite coordinator with session fencing and conservative byte reservations."""

    def __init__(
        self,
        state_directory: str | Path,
        config: Mapping[str, Any],
        *,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        validate_config(config)
        self.config = dict(config)
        self.state_directory = Path(state_directory).resolve()
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.database = self.state_directory / DATABASE_NAME
        self.clock = clock
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            existing = connection.execute("SELECT * FROM campaign_meta").fetchone()
            config_sha = _sha(config)
            if existing is None:
                connection.execute(
                    "INSERT INTO campaign_meta VALUES (1,?,?,?,?,?,0)",
                    (
                        DATABASE_SCHEMA,
                        config["campaign_id"],
                        _canonical(config),
                        config_sha,
                        self._utc(),
                    ),
                )
                self._event(connection, "campaign_initialized", None, None, {})
            elif (
                existing["schema_version"] != DATABASE_SCHEMA
                or existing["campaign_id"] != config["campaign_id"]
                or existing["config_sha256"] != config_sha
                or existing["config_json"] != _canonical(config)
            ):
                raise DurableCampaignError("refusing to open campaign with a different config")
        self._post_write_storage_check()

    def _utc(self) -> str:
        value = self.clock()
        if value.tzinfo is None:
            raise DurableCampaignError("campaign clock returned a naive datetime")
        return value.astimezone(UTC).isoformat()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        timeout = int(self.config["storage"]["busy_timeout_ms"]) / 1000
        connection = sqlite3.connect(self.database, timeout=timeout)
        connection.row_factory = sqlite3.Row
        storage = self.config["storage"]
        page_size = int(storage["page_size_bytes"])
        main_budget = (
            int(storage["maximum_sqlite_family_bytes"])
            - int(storage["maximum_transaction_reserve_bytes"])
            - 64 * 1024
        )
        try:
            connection.execute(f"PRAGMA busy_timeout={int(storage['busy_timeout_ms'])}")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA page_size={page_size}")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                f"PRAGMA wal_autocheckpoint={int(storage['wal_autocheckpoint_pages'])}"
            )
            connection.execute(
                f"PRAGMA journal_size_limit={int(storage['maximum_transaction_reserve_bytes'])}"
            )
            connection.execute(f"PRAGMA max_page_count={max(16, main_budget // page_size)}")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def storage_snapshot(self) -> dict[str, Any]:
        sizes = {}
        for key, path in (
            ("database_bytes", self.database),
            ("wal_bytes", Path(str(self.database) + "-wal")),
            ("shm_bytes", Path(str(self.database) + "-shm")),
        ):
            sizes[key] = path.stat().st_size if path.is_file() else 0
        total = sum(sizes.values())
        maximum = int(self.config["storage"]["maximum_sqlite_family_bytes"])
        return {
            **sizes,
            "total_sqlite_family_bytes": total,
            "maximum_sqlite_family_bytes": maximum,
            "within_ceiling": total <= maximum,
            "paths_persisted": False,
            "measurement_scope": "campaign.sqlite + campaign.sqlite-wal + campaign.sqlite-shm",
        }

    def _preflight_write(self, connection: sqlite3.Connection) -> None:
        # Serialize the measurement with the write.  Measuring before BEGIN IMMEDIATE would let
        # two processes reserve the same remaining bytes concurrently.
        connection.execute("BEGIN IMMEDIATE")
        snapshot = self.storage_snapshot()
        reserve = int(self.config["storage"]["maximum_transaction_reserve_bytes"])
        if (
            not snapshot["within_ceiling"]
            or snapshot["total_sqlite_family_bytes"] + reserve
            > snapshot["maximum_sqlite_family_bytes"]
        ):
            raise StorageCeilingError("SQLite/WAL/SHM write reserve would exceed byte ceiling")

    def _post_write_storage_check(self) -> None:
        snapshot = self.storage_snapshot()
        if not snapshot["within_ceiling"]:
            raise StorageCeilingError("SQLite/WAL/SHM byte ceiling was exceeded")

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._connect() as connection:
                self._preflight_write(connection)
                yield connection
        except sqlite3.DatabaseError as error:
            if "full" in str(error).lower():
                raise StorageCeilingError(
                    "SQLite main-file page ceiling blocked the write"
                ) from error
            raise
        self._post_write_storage_check()

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        host_id: str | None,
        work_id: str | None,
        payload: Mapping[str, Any],
    ) -> str:
        previous = connection.execute(
            "SELECT sequence,event_sha256 FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if previous is None else int(previous["sequence"]) + 1
        previous_sha = EVENT_GENESIS if previous is None else previous["event_sha256"]
        created = self._utc()
        body = {
            "sequence": sequence,
            "created_utc": created,
            "event_type": event_type,
            "host_id": host_id,
            "work_id": work_id,
            "payload": dict(payload),
            "previous_sha256": previous_sha,
        }
        event_sha = _sha(body)
        connection.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
            (
                sequence,
                created,
                event_type,
                host_id,
                work_id,
                _canonical(payload),
                previous_sha,
                event_sha,
            ),
        )
        return event_sha

    def register_host(self, host_id: str, session_id: str) -> dict[str, Any]:
        if host_id not in self.config["logical_hosts"] or not session_id or len(session_id) > 128:
            raise DurableCampaignError(
                "host or session is outside the registered two-host contract"
            )
        self.recover_dead_hosts()
        with self._write() as connection:
            row = connection.execute("SELECT * FROM hosts WHERE host_id=?", (host_id,)).fetchone()
            if row is not None and row["state"] == "active":
                if row["session_id"] == session_id:
                    return dict(row)
                raise DurableCampaignError("logical host already has a fresh active session")
            generation = 1 if row is None else int(row["generation"]) + 1
            now = self._utc()
            connection.execute(
                "INSERT INTO hosts VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(host_id) DO UPDATE SET generation=excluded.generation,"
                "session_id=excluded.session_id,state='active',registered_utc=excluded.registered_utc,"
                "last_heartbeat_utc=excluded.last_heartbeat_utc",
                (host_id, generation, session_id, "active", now, now),
            )
            connection.execute(
                "INSERT INTO sessions VALUES (?,?,?,'active',?,?,NULL,NULL,NULL,NULL)",
                (session_id, host_id, generation, now, now),
            )
            self._event(
                connection,
                "host_registered",
                host_id,
                None,
                {"session_id": session_id, "generation": generation},
            )
        return {"host_id": host_id, "session_id": session_id, "generation": generation}

    def heartbeat_host(self, host_id: str, session_id: str) -> bool:
        with self._write() as connection:
            now = self._utc()
            cursor = connection.execute(
                "UPDATE hosts SET last_heartbeat_utc=? WHERE host_id=? AND session_id=? "
                "AND state='active'",
                (now, host_id, session_id),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE sessions SET last_heartbeat_utc=? WHERE session_id=? AND state='active'",
                    (now, session_id),
                )
            return bool(cursor.rowcount)

    def enqueue(self, items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        accepted = duplicate = backpressured = 0
        maximum_pending = int(self.config["queue"]["maximum_pending_work"])
        maximum_payload = int(self.config["queue"]["maximum_payload_bytes"])
        max_attempts = int(self.config["queue"]["maximum_attempts"])
        for item in items:
            if not isinstance(item, Mapping) or isinstance(item.get("ordinal"), bool):
                raise DurableCampaignError("work packet is not a closed ordinal JSON object")
            payload = dict(item)
            raw = _canonical(payload).encode("utf-8")
            if len(raw) > maximum_payload:
                raise DurableCampaignError("work payload exceeds byte ceiling")
            ordinal = int(payload["ordinal"])
            payload_sha = hashlib.sha256(raw).hexdigest()
            work_id = f"DHC-{hashlib.sha256(f'{ordinal}:{payload_sha}'.encode()).hexdigest()[:24]}"
            with self._write() as connection:
                if connection.execute("SELECT 1 FROM work WHERE ordinal=?", (ordinal,)).fetchone():
                    duplicate += 1
                    continue
                pending = connection.execute(
                    "SELECT COUNT(*) FROM work WHERE state IN ('queued','running')"
                ).fetchone()[0]
                if pending >= maximum_pending:
                    backpressured += 1
                    continue
                now = self._utc()
                connection.execute(
                    "INSERT INTO work VALUES (?,?,?,?, 'queued',0,?,NULL,NULL,NULL,NULL,NULL,NULL,?,NULL)",
                    (work_id, ordinal, raw.decode("utf-8"), payload_sha, max_attempts, now),
                )
                self._event(
                    connection,
                    "work_enqueued",
                    None,
                    work_id,
                    {"ordinal": ordinal, "payload_sha256": payload_sha},
                )
                accepted += 1
        return {"accepted": accepted, "duplicate": duplicate, "backpressured": backpressured}

    def _require_active(
        self, connection: sqlite3.Connection, host_id: str, session_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM hosts WHERE host_id=? AND session_id=? AND state='active'",
            (host_id, session_id),
        ).fetchone()
        if row is None:
            raise DurableCampaignError("host session is stale or not active")
        return row

    def claim(self, host_id: str, session_id: str) -> DurableLease | None:
        self.recover_dead_hosts()
        with self._write() as connection:
            self._require_active(connection, host_id, session_id)
            row = connection.execute(
                "SELECT * FROM work WHERE state='queued' ORDER BY ordinal LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            attempt = int(row["attempt"]) + 1
            expiry = self.clock().astimezone(UTC) + timedelta(
                seconds=float(self.config["queue"]["lease_seconds"])
            )
            token = _sha(
                {
                    "campaign_id": self.config["campaign_id"],
                    "work_id": row["work_id"],
                    "attempt": attempt,
                    "host_id": host_id,
                    "session_id": session_id,
                }
            )
            connection.execute(
                "UPDATE work SET state='running',attempt=?,leased_host_id=?,leased_session_id=?,"
                "lease_token=?,lease_expires_utc=?,error_text=NULL WHERE work_id=?",
                (attempt, host_id, session_id, token, expiry.isoformat(), row["work_id"]),
            )
            self._event(
                connection,
                "work_claimed",
                host_id,
                row["work_id"],
                {"session_id": session_id, "attempt": attempt, "lease_token": token},
            )
            return DurableLease(
                row["work_id"],
                int(row["ordinal"]),
                attempt,
                int(row["max_attempts"]),
                host_id,
                session_id,
                token,
                json.loads(row["payload_json"]),
            )

    def heartbeat_lease(self, lease: DurableLease) -> bool:
        with self._write() as connection:
            now = self._utc()
            expiry = self.clock().astimezone(UTC) + timedelta(
                seconds=float(self.config["queue"]["lease_seconds"])
            )
            cursor = connection.execute(
                "UPDATE work SET lease_expires_utc=? WHERE work_id=? AND state='running' "
                "AND leased_host_id=? AND leased_session_id=? AND lease_token=? "
                "AND EXISTS (SELECT 1 FROM hosts WHERE host_id=? AND session_id=? AND state='active')",
                (
                    expiry.isoformat(),
                    lease.work_id,
                    lease.host_id,
                    lease.session_id,
                    lease.lease_token,
                    lease.host_id,
                    lease.session_id,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE hosts SET last_heartbeat_utc=? WHERE host_id=? AND session_id=?",
                    (now, lease.host_id, lease.session_id),
                )
                connection.execute(
                    "UPDATE sessions SET last_heartbeat_utc=? WHERE session_id=? AND state='active'",
                    (now, lease.session_id),
                )
            return bool(cursor.rowcount)

    def finish(self, lease: DurableLease, result: Mapping[str, Any]) -> bool:
        result_raw = _canonical(result).encode("utf-8")
        if len(result_raw) > int(self.config["queue"]["maximum_result_bytes"]):
            raise DurableCampaignError("work result exceeds byte ceiling")
        with self._write() as connection:
            cursor = connection.execute(
                "UPDATE work SET state='succeeded',result_json=?,completed_utc=?,"
                "leased_host_id=NULL,leased_session_id=NULL,lease_token=NULL,lease_expires_utc=NULL "
                "WHERE work_id=? AND state='running' AND leased_host_id=? AND leased_session_id=? "
                "AND lease_token=? AND EXISTS (SELECT 1 FROM hosts WHERE host_id=? AND session_id=? "
                "AND state='active')",
                (
                    result_raw.decode("utf-8"),
                    self._utc(),
                    lease.work_id,
                    lease.host_id,
                    lease.session_id,
                    lease.lease_token,
                    lease.host_id,
                    lease.session_id,
                ),
            )
            if cursor.rowcount:
                self._event(
                    connection,
                    "work_succeeded",
                    lease.host_id,
                    lease.work_id,
                    {
                        "attempt": lease.attempt,
                        "result_sha256": hashlib.sha256(result_raw).hexdigest(),
                    },
                )
            return bool(cursor.rowcount)

    def fail(self, lease: DurableLease, error: str) -> str:
        if not error or len(error.encode("utf-8")) > 4096:
            raise DurableCampaignError("work error text is invalid")
        with self._write() as connection:
            row = connection.execute(
                "SELECT * FROM work WHERE work_id=? AND state='running' AND leased_host_id=? "
                "AND leased_session_id=? AND lease_token=?",
                (lease.work_id, lease.host_id, lease.session_id, lease.lease_token),
            ).fetchone()
            if row is None:
                return "stale"
            retry = int(row["attempt"]) < int(row["max_attempts"])
            state = "queued" if retry else "failed"
            connection.execute(
                "UPDATE work SET state=?,error_text=?,completed_utc=?,leased_host_id=NULL,"
                "leased_session_id=NULL,lease_token=NULL,lease_expires_utc=NULL WHERE work_id=?",
                (state, error, None if retry else self._utc(), lease.work_id),
            )
            self._event(
                connection,
                f"work_{state}",
                lease.host_id,
                lease.work_id,
                {"attempt": lease.attempt, "error": error},
            )
            return state

    def recover_dead_hosts(self) -> dict[str, int]:
        recovered = failed = dead_hosts = expired_leases = 0
        now = self.clock().astimezone(UTC)
        cutoff = now - timedelta(seconds=float(self.config["queue"]["dead_host_seconds"]))
        with self._write() as connection:
            stale_hosts = connection.execute(
                "SELECT * FROM hosts WHERE state='active' AND last_heartbeat_utc<?",
                (cutoff.isoformat(),),
            ).fetchall()
            for host in stale_hosts:
                dead_hosts += 1
                connection.execute(
                    "UPDATE hosts SET state='dead' WHERE host_id=? AND session_id=? AND state='active'",
                    (host["host_id"], host["session_id"]),
                )
                connection.execute(
                    "UPDATE sessions SET state='dead',ended_utc=?,monotonic_elapsed_seconds=NULL,"
                    "credited_seconds=0,stop_reason='dead_host_recovered' WHERE session_id=? "
                    "AND state='active'",
                    (now.isoformat(), host["session_id"]),
                )
                rows = connection.execute(
                    "SELECT * FROM work WHERE state='running' AND leased_host_id=? "
                    "AND leased_session_id=?",
                    (host["host_id"], host["session_id"]),
                ).fetchall()
                for row in rows:
                    was_recovered = self._recover_row(connection, row, "dead host")
                    recovered += int(was_recovered)
                    failed += int(not was_recovered)
                self._event(
                    connection,
                    "dead_host_recovered",
                    host["host_id"],
                    None,
                    {"session_id": host["session_id"], "work_items": len(rows)},
                )
            expired = connection.execute(
                "SELECT * FROM work WHERE state='running' AND lease_expires_utc<?",
                (now.isoformat(),),
            ).fetchall()
            for row in expired:
                expired_leases += 1
                was_recovered = self._recover_row(connection, row, "expired lease")
                recovered += int(was_recovered)
                failed += int(not was_recovered)
            if expired:
                self._event(
                    connection,
                    "expired_leases_recovered",
                    None,
                    None,
                    {"work_items": len(expired)},
                )
        return {
            "dead_hosts": dead_hosts,
            "expired_leases": expired_leases,
            "recovered": recovered,
            "failed": failed,
        }

    def _recover_row(self, connection: sqlite3.Connection, row: sqlite3.Row, reason: str) -> bool:
        retry = int(row["attempt"]) < int(row["max_attempts"])
        state = "queued" if retry else "failed"
        connection.execute(
            "UPDATE work SET state=?,error_text=?,completed_utc=?,leased_host_id=NULL,"
            "leased_session_id=NULL,lease_token=NULL,lease_expires_utc=NULL WHERE work_id=?",
            (state, reason, None if retry else self._utc(), row["work_id"]),
        )
        self._event(
            connection,
            f"work_recovered_{state}",
            row["leased_host_id"],
            row["work_id"],
            {"reason": reason, "attempt": int(row["attempt"])},
        )
        return retry

    def close_host(
        self,
        host_id: str,
        session_id: str,
        *,
        monotonic_elapsed_seconds: float,
        stop_reason: str,
    ) -> dict[str, Any]:
        if monotonic_elapsed_seconds < 0 or not stop_reason or len(stop_reason) > 128:
            raise DurableCampaignError("host close accounting is invalid")
        with self._write() as connection:
            host = self._require_active(connection, host_id, session_id)
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id=? AND state='active'", (session_id,)
            ).fetchone()
            if session is None:
                raise DurableCampaignError("active host session row is missing")
            now = self.clock().astimezone(UTC)
            wall_elapsed = max(0.0, (now - _parse_time(session["started_utc"])).total_seconds())
            credited = min(float(monotonic_elapsed_seconds), wall_elapsed)
            connection.execute(
                "UPDATE hosts SET state='stopped',last_heartbeat_utc=? WHERE host_id=? "
                "AND session_id=? AND state='active'",
                (now.isoformat(), host_id, session_id),
            )
            connection.execute(
                "UPDATE sessions SET state='stopped',last_heartbeat_utc=?,ended_utc=?,"
                "monotonic_elapsed_seconds=?,credited_seconds=?,stop_reason=? WHERE session_id=?",
                (
                    now.isoformat(),
                    now.isoformat(),
                    float(monotonic_elapsed_seconds),
                    credited,
                    stop_reason,
                    session_id,
                ),
            )
            self._event(
                connection,
                "host_stopped",
                host_id,
                None,
                {
                    "session_id": session_id,
                    "generation": int(host["generation"]),
                    "credited_seconds": credited,
                    "stop_reason": stop_reason,
                },
            )
        return {
            "host_id": host_id,
            "session_id": session_id,
            "monotonic_elapsed_seconds": float(monotonic_elapsed_seconds),
            "wall_elapsed_seconds": wall_elapsed,
            "credited_seconds": credited,
            "stop_reason": stop_reason,
        }

    @staticmethod
    def _coverage(intervals: Sequence[tuple[datetime, datetime]]) -> float:
        if not intervals:
            return 0.0
        ordered = sorted(intervals)
        start, end = ordered[0]
        total = 0.0
        for next_start, next_end in ordered[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += (end - start).total_seconds()
                start, end = next_start, next_end
        return total + (end - start).total_seconds()

    def _duration_summary(self, connection: sqlite3.Connection) -> dict[str, Any]:
        by_host: dict[str, list[tuple[datetime, datetime]]] = {
            host: [] for host in self.config["logical_hosts"]
        }
        sessions = []
        for row in connection.execute(
            "SELECT * FROM sessions WHERE state='stopped' ORDER BY ended_utc,session_id"
        ):
            credit = max(0.0, float(row["credited_seconds"] or 0.0))
            end = _parse_time(row["ended_utc"])
            start = end - timedelta(seconds=credit)
            by_host[row["host_id"]].append((start, end))
            sessions.append(
                {
                    "session_id_sha256": hashlib.sha256(row["session_id"].encode()).hexdigest(),
                    "host_id": row["host_id"],
                    "generation": int(row["generation"]),
                    "started_utc": row["started_utc"],
                    "ended_utc": row["ended_utc"],
                    "credited_seconds": credit,
                    "stop_reason": row["stop_reason"],
                }
            )
        combined = [interval for intervals in by_host.values() for interval in intervals]
        return {
            "credited_wall_seconds": self._coverage(combined),
            "credited_seconds_by_host": {
                host: self._coverage(intervals) for host, intervals in by_host.items()
            },
            "cleanly_closed_sessions": sessions,
            "dead_session_credit_policy": "zero_seconds",
            "overlap_policy": "union_not_sum",
            "credit_policy": "min_monotonic_elapsed_and_wall_elapsed",
        }

    def _validate_event_chain(self, connection: sqlite3.Connection) -> tuple[int, str]:
        previous = EVENT_GENESIS
        count = 0
        for row in connection.execute("SELECT * FROM events ORDER BY sequence"):
            count += 1
            payload = json.loads(row["payload_json"])
            body = {
                "sequence": count,
                "created_utc": row["created_utc"],
                "event_type": row["event_type"],
                "host_id": row["host_id"],
                "work_id": row["work_id"],
                "payload": payload,
                "previous_sha256": previous,
            }
            if (
                int(row["sequence"]) != count
                or row["previous_sha256"] != previous
                or row["event_sha256"] != _sha(body)
            ):
                raise DurableCampaignError("durable campaign event chain changed")
            previous = row["event_sha256"]
        return count, previous

    def _validate_work_integrity(self, connection: sqlite3.Connection) -> None:
        for row in connection.execute("SELECT * FROM work ORDER BY ordinal"):
            try:
                payload = json.loads(row["payload_json"])
                raw = _canonical(payload).encode("utf-8")
                ordinal = int(payload["ordinal"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise DurableCampaignError("durable work payload changed") from error
            payload_sha = hashlib.sha256(raw).hexdigest()
            expected_id = (
                "DHC-" + hashlib.sha256(f"{ordinal}:{payload_sha}".encode()).hexdigest()[:24]
            )
            if (
                ordinal != int(row["ordinal"])
                or payload_sha != row["payload_sha256"]
                or expected_id != row["work_id"]
            ):
                raise DurableCampaignError("durable work source seal changed")
            if row["state"] == "succeeded":
                result = connection.execute(
                    "SELECT payload_json FROM events WHERE work_id=? AND event_type='work_succeeded' "
                    "ORDER BY sequence DESC LIMIT 1",
                    (row["work_id"],),
                ).fetchone()
                if result is None or row["result_json"] is None:
                    raise DurableCampaignError("durable work result seal is missing")
                event_payload = json.loads(result["payload_json"])
                result_sha = hashlib.sha256(row["result_json"].encode("utf-8")).hexdigest()
                if event_payload.get("result_sha256") != result_sha:
                    raise DurableCampaignError("durable work result seal changed")

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            meta = connection.execute("SELECT * FROM campaign_meta").fetchone()
            event_count, event_root = self._validate_event_chain(connection)
            self._validate_work_integrity(connection)
            work_counts = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM work GROUP BY state"
                )
            }
            host_counts = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM hosts GROUP BY state"
                )
            }
            invalid_running = connection.execute(
                "SELECT COUNT(*) FROM work w LEFT JOIN hosts h ON h.host_id=w.leased_host_id "
                "WHERE w.state='running' AND (h.state!='active' OR h.session_id!=w.leased_session_id)"
            ).fetchone()[0]
            if invalid_running:
                raise DurableCampaignError("running work is attached to a fenced host session")
            duration = self._duration_summary(connection)
            active_sessions = connection.execute(
                "SELECT COUNT(*) FROM sessions WHERE state='active'"
            ).fetchone()[0]
        storage = self.storage_snapshot()
        if not storage["within_ceiling"]:
            raise StorageCeilingError("SQLite/WAL/SHM byte ceiling changed")
        required = float(self.config["duration"]["required_credited_seconds"])
        per_host = float(self.config["duration"]["minimum_each_host_credited_seconds"])
        eligible = (
            duration["credited_wall_seconds"] >= required
            and all(value >= per_host for value in duration["credited_seconds_by_host"].values())
            and active_sessions == 0
            and work_counts.get("running", 0) == 0
            and work_counts.get("queued", 0) == 0
            and work_counts.get("failed", 0) == 0
            and work_counts.get("succeeded", 0) > 0
            and int(meta["storage_breaches"]) == 0
        )
        body = {
            "schema_version": STATUS_SCHEMA,
            "campaign_id": self.config["campaign_id"],
            "config_sha256": _sha(self.config),
            "logical_hosts": list(self.config["logical_hosts"]),
            "host_counts": host_counts,
            "active_sessions": int(active_sessions),
            "work_counts": work_counts,
            "event_count": event_count,
            "event_chain_root_sha256": event_root,
            "duration": duration,
            "duration_receipt_eligible": eligible,
            "storage": storage,
            "claims": {
                "two_physical_machines_established": False,
                "two_logical_host_contract_enforced": True,
                "six_hour_campaign_completed": eligible,
                "scientific_result_inferred": False,
            },
        }
        return {**body, "content_sha256": _sha(body)}

    def build_duration_receipt(self) -> dict[str, Any]:
        status = self.status()
        if not status["duration_receipt_eligible"]:
            raise DurationNotReachedError(
                "real >=6h two-host duration, clean shutdown, and zero-failure gate not reached"
            )
        body = {
            "schema_version": RECEIPT_SCHEMA,
            "campaign_id": status["campaign_id"],
            "config_sha256": status["config_sha256"],
            "decision": "PASS",
            "event_count": status["event_count"],
            "event_chain_root_sha256": status["event_chain_root_sha256"],
            "duration": status["duration"],
            "work_counts": status["work_counts"],
            "storage": status["storage"],
            "claims": {
                "real_credited_wall_seconds_at_least_six_hours": True,
                "two_logical_hosts_observed": True,
                "two_physical_machines_established": False,
                "overlapping_host_runtime_double_counted": False,
                "dead_session_runtime_credited": False,
                "scientific_result_inferred": False,
            },
            "scope": (
                "operational two-logical-host durability and storage-ceiling campaign only; "
                "not evidence of two physical machines or scientific validity"
            ),
        }
        return {**body, "content_sha256": _sha(body)}

    def validate_duration_receipt(self, receipt: Mapping[str, Any]) -> None:
        if dict(receipt) != self.build_duration_receipt():
            raise DurableCampaignError("duration receipt exact database replay changed")


def evaluate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded operational evaluator used to exercise durable mechanics, not scientific truth."""

    sleep_seconds = float(payload.get("sleep_seconds", 0.0))
    if not 0 <= sleep_seconds <= 300:
        raise DurableCampaignError("operational payload sleep is outside [0,300]")
    if sleep_seconds:
        time.sleep(sleep_seconds)
    canonical = _canonical(payload)
    return {
        "schema_version": "sigma-durable-two-host-operational-result-1.0",
        "ordinal": int(payload["ordinal"]),
        "payload_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "decision": "processed_operational_control",
        "scientific_result_inferred": False,
    }


def run_host_slice(
    campaign: DurableTwoHostCampaign,
    host_id: str,
    *,
    maximum_slice_seconds: float | None = None,
    session_id: str | None = None,
    stop_path: Path | None = None,
) -> dict[str, Any]:
    """Run one resumable host slice with independent host and lease heartbeats."""

    duration = campaign.config["duration"]
    maximum = float(duration["maximum_run_slice_seconds"])
    requested = float(
        duration["default_run_slice_seconds"]
        if maximum_slice_seconds is None
        else maximum_slice_seconds
    )
    if not 0 < requested <= maximum:
        raise DurableCampaignError("host run slice is outside the configured real-time bound")
    session = session_id or uuid.uuid4().hex
    campaign.register_host(host_id, session)
    start = time.monotonic()
    stop = threading.Event()
    current: list[DurableLease | None] = [None]
    heartbeat_failures: list[str] = []
    interval = float(duration["heartbeat_interval_seconds"])

    def heartbeat_loop() -> None:
        while not stop.wait(interval):
            try:
                if not campaign.heartbeat_host(host_id, session):
                    heartbeat_failures.append("host_session_fenced")
                    return
                lease = current[0]
                if lease is not None and not campaign.heartbeat_lease(lease):
                    heartbeat_failures.append("work_lease_fenced")
                    return
            except Exception as error:  # noqa: BLE001 - fail closed in main loop
                heartbeat_failures.append(f"{type(error).__name__}:{error}")
                return

    heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat.start()
    succeeded = failed = idle_polls = 0
    reason = "slice_duration_reached"
    try:
        while time.monotonic() - start < requested:
            if heartbeat_failures:
                reason = "heartbeat_failed"
                break
            if stop_path is not None and stop_path.exists():
                reason = "external_stop_requested"
                break
            lease = campaign.claim(host_id, session)
            if lease is None:
                idle_polls += 1
                time.sleep(
                    min(
                        float(duration["poll_interval_seconds"]),
                        max(0.0, requested - (time.monotonic() - start)),
                    )
                )
                continue
            current[0] = lease
            try:
                result = evaluate_payload(lease.payload)
                if campaign.finish(lease, result):
                    succeeded += 1
                else:
                    reason = "lease_fenced_before_finish"
                    break
            except Exception as error:  # noqa: BLE001 - task failure is durable
                state = campaign.fail(lease, f"{type(error).__name__}: {error}")
                failed += int(state == "failed")
            finally:
                current[0] = None
    finally:
        stop.set()
        heartbeat.join(timeout=max(1.0, interval * 2))
        elapsed = time.monotonic() - start
        closed = campaign.close_host(
            host_id,
            session,
            monotonic_elapsed_seconds=elapsed,
            stop_reason=reason,
        )
    return {
        "schema_version": "sigma-durable-two-host-run-slice-1.0",
        "host_id": host_id,
        "session_id_sha256": hashlib.sha256(session.encode()).hexdigest(),
        "stop_reason": reason,
        "elapsed_seconds": elapsed,
        "credited_seconds": closed["credited_seconds"],
        "succeeded": succeeded,
        "failed": failed,
        "idle_polls": idle_polls,
        "heartbeat_failures": heartbeat_failures,
        "duration_receipt_eligible": campaign.status()["duration_receipt_eligible"],
    }


def _load_packets(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise DurableCampaignError("work input byte budget violated")
    packets = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise DurableCampaignError(f"work input line {line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise DurableCampaignError("each work input line must be one JSON object")
        packets.append(value)
    return packets


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except OSError as error:
        raise DurableCampaignError("refusing to overwrite or create duration receipt") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-durable-campaign")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("--input", type=Path, required=True)
    run = commands.add_parser("run-host")
    run.add_argument("--host-id", required=True)
    run.add_argument("--maximum-slice-seconds", type=float)
    run.add_argument("--stop-path", type=Path)
    commands.add_parser("recover")
    commands.add_parser("status")
    receipt = commands.add_parser("receipt")
    receipt.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        campaign = DurableTwoHostCampaign(arguments.state_directory, load_config(arguments.config))
        if arguments.command == "init":
            result = campaign.status()
        elif arguments.command == "enqueue":
            result = campaign.enqueue(_load_packets(arguments.input))
        elif arguments.command == "run-host":
            result = run_host_slice(
                campaign,
                arguments.host_id,
                maximum_slice_seconds=arguments.maximum_slice_seconds,
                stop_path=arguments.stop_path,
            )
        elif arguments.command == "recover":
            result = campaign.recover_dead_hosts()
        elif arguments.command == "status":
            result = campaign.status()
        else:
            result = campaign.build_duration_receipt()
            _write_new_json(arguments.output, result)
    except DurationNotReachedError as error:
        print(_canonical({"decision": "BLOCK", "error": str(error)}))
        return 20
    except DurableCampaignError as error:
        print(_canonical({"decision": "ERROR", "error": str(error)}))
        return 40
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "DATABASE_NAME",
    "DurableCampaignError",
    "DurableLease",
    "DurableTwoHostCampaign",
    "DurationNotReachedError",
    "StorageCeilingError",
    "evaluate_payload",
    "load_config",
    "main",
    "run_host_slice",
    "validate_config",
]
