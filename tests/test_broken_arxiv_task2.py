from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sigma_theory_compiler import broken_arxiv_task2 as task2

ROOT = Path(__file__).resolve().parents[1]


def _authorization() -> dict:
    return task2.build_authorization(
        ROOT, now=datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    )


def _catalog(*, future: bool) -> list[dict]:
    rows = [
        {
            "id": "MathArena/brokenarxiv-0626",
            "lastModified": "2026-06-11T00:00:00Z",
            "sha": "6" * 40,
        },
        {
            "id": "MathArena/brokenarxiv-0626_outputs",
            "lastModified": "2026-06-12T00:00:00Z",
            "sha": "a" * 40,
        },
    ]
    if future:
        rows.extend(
            [
                {
                    "id": "MathArena/brokenarxiv-0826",
                    "lastModified": "2026-08-30T00:00:00Z",
                    "sha": "8" * 40,
                },
                {
                    "id": "MathArena/brokenarxiv-0726",
                    "lastModified": "2026-08-29T00:00:00Z",
                    "sha": "7" * 40,
                },
            ]
        )
    return rows


def test_authorization_binds_implementation_selector_and_zero_target_reads() -> None:
    authorization = _authorization()
    task2.validate_authorization(authorization, ROOT)
    assert authorization["source_cutoff"]["problem_rows_read"] == 0
    assert authorization["source_cutoff"]["reference_answers_read"] == 0
    assert len(authorization["implementation_bindings"]) == 4
    assert len(authorization["selector_commitment"]) == 64


def test_catalog_check_blocks_without_reading_current_problem_rows() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=False))
    task2.validate_source_check(checked, authorization, config)
    assert checked["status"] == "BLOCKED_FUTURE_RELEASE_NOT_PUBLISHED"
    assert checked["selected_release"] is None
    assert checked["catalog_query"]["problem_rows_read"] == 0


def test_first_future_release_is_forced_even_if_later_release_is_listed_first() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=True))
    task2.validate_source_check(checked, authorization, config)
    assert checked["status"] == "READY_FIRST_ELIGIBLE_RELEASE_METADATA_ONLY"
    assert checked["selected_release"]["dataset_id"] == "MathArena/brokenarxiv-0726"


def test_problem_selection_is_deterministic_and_reference_blind() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=True))
    packet = {
        "dataset_id": "MathArena/brokenarxiv-0726",
        "revision": "7" * 40,
        "items": [
            {"problem_id": "p3", "problem": "False statement three."},
            {"problem_id": "p1", "problem": "False statement one."},
            {"problem_id": "p2", "problem": "False statement two."},
        ],
    }
    first = task2.stage_problem(authorization, checked, config, packet)
    second = task2.stage_problem(
        authorization, checked, config, {**packet, "items": list(reversed(packet["items"]))}
    )
    assert first == second
    assert first["blindness"]["reference_answers_read"] == 0
    selected = first["selection"]
    assert selected["statement_sha256"] == hashlib.sha256(
        selected["statement"].encode()
    ).hexdigest()


def test_reference_material_and_manual_resealing_fail_closed() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=True))
    packet = {
        "dataset_id": "MathArena/brokenarxiv-0726",
        "revision": "7" * 40,
        "items": [{"id": "p1", "statement": "False.", "solution": "Known witness."}],
    }
    with pytest.raises(task2.BrokenArxivTask2Error, match="reference or judge"):
        task2.stage_problem(authorization, checked, config, packet)
    changed = deepcopy(checked)
    changed["catalog_query"]["problem_rows_read"] = 1
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = task2.canonical_sha256(body)
    with pytest.raises(task2.BrokenArxivTask2Error, match="metadata-only chronology"):
        task2.validate_source_check(changed, authorization, config)
