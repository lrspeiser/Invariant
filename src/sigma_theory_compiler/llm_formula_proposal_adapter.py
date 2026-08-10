"""Secret-safe, budget-capped LLM proposal adapter with quarantined outputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProposalAdapterError(ValueError):
    """Raised when policy, schema, lineage, or budget validation fails."""


MICRO_USD = 1_000_000
_REQUEST_ID_RE = re.compile(r"^[a-zA-Z0-9_.:-]{8,128}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN = re.compile(r"dark\s*matter|\bhalo\b|\bredshift\b|supernova", re.IGNORECASE)
_DSL_RE = re.compile(r"^[A-Za-z0-9_+*/^()., \-]{1,2048}$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _money_to_micro_usd(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ProposalAdapterError(f"{name} must be decimal money")
    text = str(value)
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?", text):
        raise ProposalAdapterError(f"{name} must have at most six decimal places")
    whole, dot, fraction = text.partition(".")
    return int(whole) * MICRO_USD + int((fraction if dot else "").ljust(6, "0"))


def _micro_usd_text(value: int) -> str:
    return f"{value // MICRO_USD}.{value % MICRO_USD:06d}"


@dataclass(frozen=True)
class AdapterConfig:
    paid_calls_enabled: bool
    provider_id: str
    api_key_env_var: str
    maximum_total_micro_usd: int
    maximum_call_micro_usd: int
    maximum_attempts: int
    allowed_data_classes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AdapterConfig:
        required = {
            "allowed_data_classes",
            "api_key_env_var",
            "maximum_attempts",
            "maximum_call_usd",
            "maximum_total_usd",
            "paid_calls_enabled",
            "provider_id",
        }
        if set(raw) != required:
            raise ProposalAdapterError("adapter config shape mismatch")
        env_var = str(raw["api_key_env_var"])
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", env_var):
            raise ProposalAdapterError("API key reference must be an environment variable name")
        maximum_total = _money_to_micro_usd(raw["maximum_total_usd"], name="maximum_total_usd")
        maximum_call = _money_to_micro_usd(raw["maximum_call_usd"], name="maximum_call_usd")
        if maximum_total != 500 * MICRO_USD:
            raise ProposalAdapterError("initial total budget must be exactly $500")
        if not 0 < maximum_call <= maximum_total:
            raise ProposalAdapterError("per-call cap must be positive and within total cap")
        attempts = int(raw["maximum_attempts"])
        if not 1 <= attempts <= 8:
            raise ProposalAdapterError("maximum_attempts must be in [1, 8]")
        allowed = tuple(str(value) for value in raw["allowed_data_classes"])
        if not allowed or any(
            value not in {"formal_artifact", "formula_dsl", "synthetic_known_answer"}
            for value in allowed
        ):
            raise ProposalAdapterError("data-class allowlist contains an ineligible class")
        return cls(
            bool(raw["paid_calls_enabled"]),
            str(raw["provider_id"]),
            env_var,
            maximum_total,
            maximum_call,
            attempts,
            allowed,
        )


@dataclass(frozen=True)
class ProposalRequest:
    request_id: str
    prompt: str
    prompt_template_sha256: str
    context_packets: tuple[Mapping[str, str], ...]
    dsl_version: str
    deterministic_seed: int
    maximum_call_usd: str

    def validate(self, config: AdapterConfig) -> dict[str, Any]:
        if not _REQUEST_ID_RE.fullmatch(self.request_id):
            raise ProposalAdapterError("invalid idempotent request ID")
        if not self.prompt or len(self.prompt.encode("utf-8")) > 65_536:
            raise ProposalAdapterError("prompt is empty or exceeds bounded size")
        if _FORBIDDEN.search(self.prompt):
            raise ProposalAdapterError("prompt contains forbidden observational inference material")
        if not _SHA_RE.fullmatch(self.prompt_template_sha256):
            raise ProposalAdapterError("prompt template hash is invalid")
        if not re.fullmatch(r"sigma-gravity-dsl-[0-9]+", self.dsl_version):
            raise ProposalAdapterError("DSL version is not reviewed")
        if not 0 <= self.deterministic_seed < 2**63:
            raise ProposalAdapterError("deterministic seed is outside range")
        if not self.context_packets or len(self.context_packets) > 64:
            raise ProposalAdapterError("context packet count is outside bounded range")
        normalized_packets = []
        for packet in self.context_packets:
            if set(packet) != {"content_sha256", "data_class"}:
                raise ProposalAdapterError("context packet shape mismatch")
            digest = str(packet["content_sha256"])
            data_class = str(packet["data_class"])
            if not _SHA_RE.fullmatch(digest) or data_class not in config.allowed_data_classes:
                raise ProposalAdapterError("context packet is unbound or ineligible")
            normalized_packets.append({"content_sha256": digest, "data_class": data_class})
        call_cap = _money_to_micro_usd(self.maximum_call_usd, name="request maximum_call_usd")
        if call_cap > config.maximum_call_micro_usd:
            raise ProposalAdapterError("request exceeds configured per-call cap")
        lineage = {
            "context_packets": normalized_packets,
            "deterministic_seed": self.deterministic_seed,
            "dsl_version": self.dsl_version,
            "prompt_sha256": sha256_bytes(self.prompt.encode("utf-8")),
            "prompt_template_sha256": self.prompt_template_sha256,
        }
        return {
            "call_cap_micro_usd": call_cap,
            "lineage": lineage,
            "lineage_sha256": canonical_sha256(lineage),
            "prompt_sha256": lineage["prompt_sha256"],
        }


def validate_proposal_output(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) != {"proposals", "schema_version"}:
        raise ProposalAdapterError("provider output shape mismatch")
    if raw["schema_version"] != "sigma-formula-proposals-1.0":
        raise ProposalAdapterError("provider output schema version mismatch")
    proposals = raw["proposals"]
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= 32:
        raise ProposalAdapterError("proposal count is outside bounded range")
    normalized = []
    seen = set()
    for item in proposals:
        if not isinstance(item, Mapping) or set(item) != {
            "concept_tags",
            "expression",
            "parameters",
            "proposal_id",
        }:
            raise ProposalAdapterError("proposal item shape mismatch")
        proposal_id = str(item["proposal_id"])
        expression = str(item["expression"])
        if not _REQUEST_ID_RE.fullmatch(proposal_id) or proposal_id in seen:
            raise ProposalAdapterError("proposal ID is invalid or duplicated")
        if not _DSL_RE.fullmatch(expression) or _FORBIDDEN.search(expression):
            raise ProposalAdapterError("proposal expression is outside quarantined DSL syntax")
        parameters = item["parameters"]
        tags = item["concept_tags"]
        if not isinstance(parameters, list) or not isinstance(tags, list):
            raise ProposalAdapterError("proposal parameters and tags must be lists")
        if len(parameters) > 16 or len(tags) > 16:
            raise ProposalAdapterError("proposal metadata exceeds bounded size")
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", str(x)) for x in parameters):
            raise ProposalAdapterError("proposal parameter name is invalid")
        if any(not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", str(x)) for x in tags):
            raise ProposalAdapterError("proposal concept tag is invalid")
        seen.add(proposal_id)
        normalized.append(
            {
                "concept_tags": [str(x) for x in tags],
                "expression": expression,
                "parameters": [str(x) for x in parameters],
                "proposal_id": proposal_id,
            }
        )
    return {"proposals": normalized, "schema_version": raw["schema_version"]}


class SpendLedger:
    """SQLite ledger containing hashes and integer money only, never secrets or bodies."""

    def __init__(self, path: Path, config: AdapterConfig) -> None:
        self.path = path
        self.config = config
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS proposal_requests (
                    request_id TEXT PRIMARY KEY,
                    prompt_sha256 TEXT NOT NULL,
                    lineage_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_micro_usd INTEGER NOT NULL,
                    settled_micro_usd INTEGER,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    output_sha256 TEXT,
                    error_code TEXT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def reserve(self, request: ProposalRequest) -> dict[str, Any]:
        validated = request.validate(self.config)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM proposal_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["prompt_sha256"] != validated["prompt_sha256"]
                    or existing["lineage_sha256"] != validated["lineage_sha256"]
                    or existing["reserved_micro_usd"] != validated["call_cap_micro_usd"]
                ):
                    connection.rollback()
                    raise ProposalAdapterError("request ID replay has different lineage or cap")
                connection.commit()
                return self._row(existing)
            exposure = connection.execute(
                """SELECT COALESCE(SUM(CASE WHEN status LIKE 'settled%' THEN settled_micro_usd
                    ELSE reserved_micro_usd END),0) FROM proposal_requests
                    WHERE status IN ('reserved','retryable','settled','settled_invalid')"""
            ).fetchone()[0]
            if exposure + validated["call_cap_micro_usd"] > self.config.maximum_total_micro_usd:
                connection.rollback()
                raise ProposalAdapterError("atomic total spend reservation would exceed $500 cap")
            connection.execute(
                """INSERT INTO proposal_requests
                (request_id,prompt_sha256,lineage_sha256,status,reserved_micro_usd)
                VALUES (?,?,?,?,?)""",
                (
                    request.request_id,
                    validated["prompt_sha256"],
                    validated["lineage_sha256"],
                    "reserved",
                    validated["call_cap_micro_usd"],
                ),
            )
            row = connection.execute(
                "SELECT * FROM proposal_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            connection.commit()
            return self._row(row)

    def begin_attempt(self, request_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM proposal_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None or row["status"] not in {"reserved", "retryable"}:
                connection.rollback()
                raise ProposalAdapterError("request is not reserved for an attempt")
            if row["attempts"] >= self.config.maximum_attempts:
                connection.rollback()
                raise ProposalAdapterError("maximum provider attempts exhausted")
            connection.execute(
                "UPDATE proposal_requests SET attempts=attempts+1,status='reserved',error_code=NULL WHERE request_id=?",
                (request_id,),
            )
            updated = connection.execute(
                "SELECT * FROM proposal_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            connection.commit()
            return self._row(updated)

    def mark_retryable(self, request_id: str, error_code: str) -> None:
        if not re.fullmatch(r"[a-z0-9_]{1,64}", error_code):
            raise ProposalAdapterError("unsafe error code")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE proposal_requests SET status='retryable',error_code=? WHERE request_id=? AND status='reserved'",
                (error_code, request_id),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise ProposalAdapterError("retry transition failed")
            connection.commit()

    def settle(
        self,
        request_id: str,
        *,
        billed_usd: Any,
        output_sha256: str,
        output_valid: bool = True,
    ) -> dict[str, Any]:
        billed = _money_to_micro_usd(billed_usd, name="provider billed_usd")
        if not _SHA_RE.fullmatch(output_sha256):
            raise ProposalAdapterError("output hash is invalid")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM proposal_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ProposalAdapterError("unknown request settlement")
            final_status = "settled" if output_valid else "settled_invalid"
            if row["status"] in {"settled", "settled_invalid"}:
                if row["settled_micro_usd"] != billed or row["output_sha256"] != output_sha256:
                    connection.rollback()
                    raise ProposalAdapterError("settled request replay differs")
                connection.commit()
                return self._row(row)
            if row["status"] != "reserved" or billed > row["reserved_micro_usd"]:
                connection.rollback()
                raise ProposalAdapterError("settlement exceeds reservation or invalid state")
            connection.execute(
                """UPDATE proposal_requests SET status=?,settled_micro_usd=?,
                output_sha256=?,error_code=NULL WHERE request_id=?""",
                (final_status, billed, output_sha256, request_id),
            )
            updated = connection.execute(
                "SELECT * FROM proposal_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            connection.commit()
            return self._row(updated)

    def status(self, request_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM proposal_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        return None if row is None else self._row(row)

    def telemetry(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status,COUNT(*) AS count,COALESCE(SUM(settled_micro_usd),0) AS settled FROM proposal_requests GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        settled = sum(row["settled"] for row in rows)
        return {
            "maximum_total_usd": _micro_usd_text(self.config.maximum_total_micro_usd),
            "request_status_counts": counts,
            "settled_usd": _micro_usd_text(settled),
        }


Provider = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


class FormulaProposalAdapter:
    def __init__(self, config: AdapterConfig, ledger: SpendLedger, provider: Provider) -> None:
        self.config = config
        self.ledger = ledger
        self.provider = provider

    def propose(self, request: ProposalRequest) -> dict[str, Any]:
        if not self.config.paid_calls_enabled:
            return {
                "decision": "blocked",
                "reason": "paid_calls_disabled_by_default",
                "request_id": request.request_id,
            }
        secret = os.environ.get(self.config.api_key_env_var)
        if secret is None or not secret.strip():
            return {
                "decision": "blocked",
                "reason": "referenced_api_secret_absent",
                "request_id": request.request_id,
            }
        row = self.ledger.reserve(request)
        if row["status"] in {"settled", "settled_invalid"}:
            return {
                "decision": "quarantined" if row["status"] == "settled" else "rejected_quarantine",
                "output_sha256": row["output_sha256"],
                "request_id": request.request_id,
                "replayed": True,
                "settled_usd": _micro_usd_text(row["settled_micro_usd"]),
            }
        while True:
            attempt = self.ledger.begin_attempt(request.request_id)
            provider_request = {
                "context_packets": [dict(packet) for packet in request.context_packets],
                "deterministic_seed": request.deterministic_seed,
                "dsl_version": request.dsl_version,
                "idempotency_key": request.request_id,
                "maximum_billed_usd": _micro_usd_text(attempt["reserved_micro_usd"]),
                "prompt": request.prompt,
                "prompt_sha256": attempt["prompt_sha256"],
            }
            try:
                response = self.provider(provider_request, secret)
            except Exception:  # noqa: BLE001 - provider boundary must classify arbitrary SDK failures
                self.ledger.mark_retryable(request.request_id, "provider_exception")
                if attempt["attempts"] >= self.config.maximum_attempts:
                    raise ProposalAdapterError("provider attempts exhausted with reservation retained") from None
                continue
            if set(response) != {"billed_usd", "output"}:
                self.ledger.mark_retryable(request.request_id, "provider_schema")
                raise ProposalAdapterError("provider envelope schema mismatch")
            try:
                output = validate_proposal_output(response["output"])
            except ProposalAdapterError:
                invalid_hash = canonical_sha256(response["output"])
                settled = self.ledger.settle(
                    request.request_id,
                    billed_usd=response["billed_usd"],
                    output_sha256=invalid_hash,
                    output_valid=False,
                )
                return {
                    "decision": "rejected_quarantine",
                    "output_sha256": invalid_hash,
                    "reason": "provider_output_schema_invalid",
                    "request_id": request.request_id,
                    "settled_usd": _micro_usd_text(settled["settled_micro_usd"]),
                }
            output_sha256 = canonical_sha256(output)
            settled = self.ledger.settle(
                request.request_id,
                billed_usd=response["billed_usd"],
                output_sha256=output_sha256,
            )
            return {
                "decision": "quarantined",
                "downstream_validation_required": True,
                "lineage_sha256": settled["lineage_sha256"],
                "output": output,
                "output_sha256": output_sha256,
                "request_id": request.request_id,
                "replayed": False,
                "settled_usd": _micro_usd_text(settled["settled_micro_usd"]),
            }


def build_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "sigma-llm-formula-proposal-adapter-config-1.0":
        raise ProposalAdapterError("unexpected campaign config schema")
    config = AdapterConfig.from_mapping(raw["adapter"])
    if config.paid_calls_enabled:
        raise ProposalAdapterError("checked-in adapter must default paid calls to disabled")
    source_sha256 = sha256_file(repo_root / "src/sigma_theory_compiler/llm_formula_proposal_adapter.py")
    artifact: dict[str, Any] = {
        "campaign_id": raw["campaign_id"],
        "credential_persistence": False,
        "credential_reference": config.api_key_env_var,
        "data_eligibility": raw["data_eligibility"],
        "default_paid_calls_enabled": False,
        "maximum_call_usd": _micro_usd_text(config.maximum_call_micro_usd),
        "maximum_total_usd": _micro_usd_text(config.maximum_total_micro_usd),
        "network_calls_made": 0,
        "output_status": "quarantine_until_downstream_validation",
        "paid_spend_usd": "0.000000",
        "schema_version": "sigma-llm-formula-proposal-adapter-readiness-1.0",
        "source_sha256": source_sha256,
        "status": "ready_disabled_no_network_no_spend",
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact
