"""Hash-chained, fsync-backed private journal for bounded LLM attempts.

The journal is deliberately private because it may contain raw prompts and model responses.  It
never stores header values or credential material.  A message dispatch is sealed before the
transport is invoked, and its response or transport error is sealed before downstream parsing or
contract validation can run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .claude_creativity_api import Transport
from .sigma_core import canonical_sha256

JOURNAL_SCHEMA = "invariant-private-llm-attempt-journal-1.0"
EVENT_SCHEMA = "invariant-private-llm-attempt-event-1.0"
_ZERO_SHA256 = "0" * 64
_MAXIMUM_JOURNAL_BYTES = 128 * 1024 * 1024
_MAXIMUM_EVENTS = 4096


class AttemptJournalError(ValueError):
    """The private attempt journal, hash chain, or transport boundary failed closed."""


def _bounded_text(value: Any, label: str, *, maximum_bytes: int = 2048) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttemptJournalError(f"{label} is empty")
    normalized = value.strip()
    if len(normalized.encode("utf-8")) > maximum_bytes:
        normalized = normalized.encode("utf-8")[:maximum_bytes].decode(
            "utf-8", errors="ignore"
        ).rstrip()
    if not normalized:
        raise AttemptJournalError(f"{label} is empty after bounding")
    return normalized


def _event(
    sequence: int,
    previous_event_sha256: str,
    event_kind: str,
    scheduled_call_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA,
        "sequence": sequence,
        "previous_event_sha256": previous_event_sha256,
        "event_kind": _bounded_text(event_kind, "attempt event kind", maximum_bytes=128),
        "scheduled_call_id": _bounded_text(
            scheduled_call_id, "scheduled call ID", maximum_bytes=256
        ),
        "payload": dict(payload),
    }
    body["event_sha256"] = canonical_sha256(body)
    return body


def _validate_event(
    value: Mapping[str, Any], *, expected_sequence: int, expected_previous: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "event_kind",
        "event_sha256",
        "payload",
        "previous_event_sha256",
        "scheduled_call_id",
        "schema_version",
        "sequence",
    }:
        raise AttemptJournalError("attempt event keys changed")
    body = {key: item for key, item in value.items() if key != "event_sha256"}
    if (
        value["schema_version"] != EVENT_SCHEMA
        or value["sequence"] != expected_sequence
        or value["previous_event_sha256"] != expected_previous
        or value["event_sha256"] != canonical_sha256(body)
        or not isinstance(value["payload"], Mapping)
    ):
        raise AttemptJournalError("attempt event identity or hash chain changed")
    _bounded_text(value["event_kind"], "attempt event kind", maximum_bytes=128)
    _bounded_text(value["scheduled_call_id"], "scheduled call ID", maximum_bytes=256)


class DurableAttemptJournal:
    """Append-only event journal with a private experiment header."""

    def __init__(self, path: Path, events: Sequence[Mapping[str, Any]]) -> None:
        self.path = path.resolve()
        self._events = [dict(item) for item in events]

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        experiment_id: str,
        source_bindings: Mapping[str, Any],
        unblinding_key: bytes,
    ) -> DurableAttemptJournal:
        path = path.resolve()
        if path.exists():
            raise AttemptJournalError("attempt journal already exists")
        if len(unblinding_key) < 32:
            raise AttemptJournalError("attempt journal unblinding key is shorter than 256 bits")
        path.parent.mkdir(parents=True, exist_ok=True)
        journal = cls(path, [])
        journal.append(
            "journal_header",
            "journal.header",
            {
                "journal_schema": JOURNAL_SCHEMA,
                "experiment_id": _bounded_text(
                    experiment_id, "attempt experiment ID", maximum_bytes=256
                ),
                "source_bindings": dict(source_bindings),
                "unblinding_key_hex": unblinding_key.hex(),
            },
        )
        return journal

    @classmethod
    def load(cls, path: Path) -> DurableAttemptJournal:
        path = path.resolve()
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise AttemptJournalError("attempt journal could not be read") from error
        if not raw or len(raw) > _MAXIMUM_JOURNAL_BYTES:
            raise AttemptJournalError("attempt journal byte budget changed")
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise AttemptJournalError("attempt journal is not UTF-8") from error
        if not lines or len(lines) > _MAXIMUM_EVENTS:
            raise AttemptJournalError("attempt journal event budget changed")
        events = []
        previous = _ZERO_SHA256
        for sequence, line in enumerate(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AttemptJournalError("attempt journal line is not JSON") from error
            _validate_event(value, expected_sequence=sequence, expected_previous=previous)
            events.append(value)
            previous = value["event_sha256"]
        journal = cls(path, events)
        header = journal.header
        if header.get("journal_schema") != JOURNAL_SCHEMA:
            raise AttemptJournalError("attempt journal header schema changed")
        return journal

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._events)

    @property
    def header(self) -> dict[str, Any]:
        if (
            not self._events
            or self._events[0]["event_kind"] != "journal_header"
            or self._events[0]["scheduled_call_id"] != "journal.header"
        ):
            raise AttemptJournalError("attempt journal header is missing")
        return dict(self._events[0]["payload"])

    @property
    def unblinding_key(self) -> bytes:
        try:
            key = bytes.fromhex(self.header["unblinding_key_hex"])
        except (KeyError, TypeError, ValueError) as error:
            raise AttemptJournalError("attempt journal unblinding key is invalid") from error
        if len(key) < 32:
            raise AttemptJournalError("attempt journal unblinding key is too short")
        return key

    @property
    def content_sha256(self) -> str:
        return canonical_sha256([item["event_sha256"] for item in self._events])

    def events_for(self, scheduled_call_id: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(item)
            for item in self._events
            if item["scheduled_call_id"] == scheduled_call_id
        )

    def append(
        self, event_kind: str, scheduled_call_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if len(self._events) >= _MAXIMUM_EVENTS:
            raise AttemptJournalError("attempt journal event cap exhausted")
        event = _event(
            len(self._events),
            self._events[-1]["event_sha256"] if self._events else _ZERO_SHA256,
            event_kind,
            scheduled_call_id,
            payload,
        )
        serialized = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise AttemptJournalError("attempt journal event could not be persisted") from error
        self._events.append(event)
        return dict(event)

    def event_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self._events:
            kind = event["event_kind"]
            counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items()))


class JournaledScheduledTransport:
    """Persist one scheduled call's transport evidence before returning it to the parser."""

    def __init__(
        self,
        journal: DurableAttemptJournal,
        *,
        scheduled_call_id: str,
        arm: str,
        task_id: str,
        role: str,
        base_transport: Transport,
    ) -> None:
        self.journal = journal
        self.scheduled_call_id = scheduled_call_id
        self.arm = arm
        self.task_id = task_id
        self.role = role
        self.base_transport = base_transport
        self._message_dispatched = False

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        if method == "POST":
            if self._message_dispatched:
                raise AttemptJournalError("scheduled call attempted more than one message dispatch")
            self._message_dispatched = True
            kind = "message"
        elif method == "GET":
            kind = "model_probe"
        else:
            raise AttemptJournalError("journaled LLM transport method is unsupported")
        body_value: Any = None
        if body is not None:
            try:
                body_value = json.loads(body)
            except json.JSONDecodeError as error:
                raise AttemptJournalError("journaled LLM request body is not JSON") from error
        self.journal.append(
            f"{kind}_dispatch",
            self.scheduled_call_id,
            {
                "arm": self.arm,
                "body": body_value,
                "body_sha256": None if body is None else hashlib.sha256(body).hexdigest(),
                "header_names": sorted(str(name).lower() for name in headers),
                "method": method,
                "role": self.role,
                "task_id": self.task_id,
                "timeout_seconds_decimal": format(timeout, ".17g"),
                "url": url,
            },
        )
        try:
            status, response = self.base_transport(method, url, headers, body, timeout)
        except Exception as error:
            self.journal.append(
                f"{kind}_transport_error",
                self.scheduled_call_id,
                {
                    "arm": self.arm,
                    "error_message": _bounded_text(
                        str(error) or type(error).__name__,
                        "transport error",
                        maximum_bytes=1024,
                    ),
                    "error_type": type(error).__name__,
                    "role": self.role,
                    "task_id": self.task_id,
                },
            )
            raise
        response_value = dict(response)
        self.journal.append(
            f"{kind}_response",
            self.scheduled_call_id,
            {
                "arm": self.arm,
                "response": response_value,
                "response_sha256": canonical_sha256(response_value),
                "role": self.role,
                "status": status,
                "task_id": self.task_id,
            },
        )
        return status, response
