from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.durable_llm_attempt_journal import (
    AttemptJournalError,
    DurableAttemptJournal,
    JournaledScheduledTransport,
)


def _journal(tmp_path: Path) -> DurableAttemptJournal:
    return DurableAttemptJournal.create(
        tmp_path / "work" / "attempts.jsonl",
        experiment_id="confirmatory.test",
        source_bindings={"runner_sha256": "a" * 64},
        unblinding_key=b"k" * 32,
    )


def test_dispatch_and_response_are_durable_before_parser_receives_them(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    observed_event_counts = []

    def transport(method, _url, headers, body, _timeout):
        observed_event_counts.append(len(DurableAttemptJournal.load(journal.path).events))
        assert method == "POST"
        assert headers["x-api-key"] == "secret-never-persisted"
        assert json.loads(body)["model"] == "claude-opus-4-6"
        return 200, {"id": "msg.test", "usage": {"input_tokens": 2, "output_tokens": 1}}

    wrapped = JournaledScheduledTransport(
        journal,
        scheduled_call_id="blind.task:baseline:proposer",
        arm="baseline",
        task_id="blind.task",
        role="proposer",
        base_transport=transport,
    )
    status, response = wrapped(
        "POST",
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": "secret-never-persisted", "content-type": "application/json"},
        json.dumps({"model": "claude-opus-4-6"}).encode(),
        120,
    )
    assert status == 200 and response["id"] == "msg.test"
    assert observed_event_counts == [2]
    loaded = DurableAttemptJournal.load(journal.path)
    assert loaded.event_counts() == {
        "journal_header": 1,
        "message_dispatch": 1,
        "message_response": 1,
    }
    serialized = journal.path.read_text(encoding="utf-8")
    assert "secret-never-persisted" not in serialized
    assert loaded.unblinding_key == b"k" * 32


def test_transport_error_is_durable_and_never_retried_by_wrapper(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    calls = 0

    def broken(*_args):
        nonlocal calls
        calls += 1
        raise TimeoutError("bounded timeout")

    wrapped = JournaledScheduledTransport(
        journal,
        scheduled_call_id="blind.task:treatment:critic",
        arm="treatment",
        task_id="blind.task",
        role="critic",
        base_transport=broken,
    )
    with pytest.raises(TimeoutError, match="bounded timeout"):
        wrapped("POST", "https://example.test", {}, b"{}", 1)
    assert calls == 1
    assert DurableAttemptJournal.load(journal.path).event_counts()[
        "message_transport_error"
    ] == 1


def test_hash_chain_tamper_and_second_dispatch_fail_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)

    def transport(*_args):
        return 200, {"ok": True}

    wrapped = JournaledScheduledTransport(
        journal,
        scheduled_call_id="blind.task:baseline:proposer",
        arm="baseline",
        task_id="blind.task",
        role="proposer",
        base_transport=transport,
    )
    wrapped("POST", "https://example.test", {}, b"{}", 1)
    with pytest.raises(AttemptJournalError, match="more than one"):
        wrapped("POST", "https://example.test", {}, b"{}", 1)

    lines = journal.path.read_text(encoding="utf-8").splitlines()
    value = json.loads(lines[1])
    value["payload"]["task_id"] = "tampered"
    lines[1] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    journal.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AttemptJournalError, match="hash chain"):
        DurableAttemptJournal.load(journal.path)
