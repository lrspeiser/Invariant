from __future__ import annotations

from pathlib import Path

import pytest

from sigma_theory_compiler import mathoverflow_task2 as task2
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return task2.load_config(ROOT)


def _authorization(config):
    return task2.build_authorization(ROOT, config)


def _metadata_items():
    return [
        {
            "accepted_answer_id": 9002,
            "answer_count": 1,
            "creation_date": 1784624170,
            "is_answered": True,
            "last_activity_date": 1784707249,
            "question_id": 513421,
            "tags": ["reference-request", "co.combinatorics", "counterexamples"],
        },
        {
            "answer_count": 0,
            "creation_date": 1784628689,
            "is_answered": False,
            "last_activity_date": 1784634856,
            "question_id": 513430,
            "tags": ["ct.category-theory", "counterexamples"],
        },
        {
            "accepted_answer_id": 1,
            "answer_count": 1,
            "creation_date": 1700000000,
            "is_answered": True,
            "last_activity_date": 1700000010,
            "question_id": 1,
            "tags": ["co.combinatorics", "counterexamples"],
        },
    ]


def _ready_chain():
    config = _config()
    authorization = _authorization(config)
    source = task2.check_source(
        authorization,
        config,
        fetch_json=lambda _url: {"items": _metadata_items()},
    )
    return config, authorization, source


def test_config_reuses_exact_36_call_matched_generation_contract():
    config = _config()
    effective = task2.effective_generation_config(ROOT, config)
    staged = {"content_sha256": "a" * 64}
    specs = task2.broken.build_arm_specs(staged, effective)

    assert set(specs) == {
        "old_failure_first_llm",
        "creativity_first_llm",
        "matched_random_falsifier",
    }
    assert {len(rows) for rows in specs.values()} == {12}
    assert sum(map(len, specs.values())) == 36
    assert effective["task_id"] == config["task_id"]


def test_authorization_binds_zero_body_reads_and_every_implementation_dependency():
    config = _config()
    authorization = _authorization(config)
    task2.validate_authorization(authorization, ROOT, config)

    assert authorization["question_titles_read"] == 0
    assert authorization["question_bodies_read"] == 0
    assert authorization["answer_bodies_read"] == 0
    assert set(authorization["file_sha256"]) == set(config["implementation_paths"]) | set(
        config["dependency_paths"]
    )


def test_metadata_only_selector_requires_recent_open_accepted_counterexample():
    _config_value, authorization, source = _ready_chain()
    task2.validate_source(source, authorization)

    assert source["metadata_query"]["question_titles_read"] == 0
    assert source["metadata_query"]["question_bodies_read"] == 0
    assert source["metadata_query"]["answer_bodies_read"] == 0
    assert [row["question_id"] for row in source["eligible_metadata"]] == [513421]
    assert source["selected_question"]["question_id"] == 513421
    assert source["status"] == "READY_SELECTED_BODY_UNREAD"


def test_staging_fetches_only_selected_question_and_keeps_answer_closed():
    config, authorization, source = _ready_chain()
    calls = []

    def fetch(url):
        calls.append(url)
        return {
            "items": [
                {
                    "accepted_answer_id": 9002,
                    "body": "<p>Is the proposed bound false for all <i>n</i>?</p>",
                    "last_activity_date": 1784707249,
                    "question_id": 513421,
                    "tags": ["co.combinatorics", "counterexamples"],
                    "title": "A counterexample problem",
                }
            ]
        }

    staged = task2.stage_question(source, authorization, config, fetch_json=fetch)

    assert len(calls) == 1
    assert "/questions/513421?" in calls[0]
    assert "/answers/" not in calls[0]
    assert staged["selection"]["accepted_answer_id"] == 9002
    assert staged["blindness"]["accepted_answer_body_read"] == 0
    assert staged["status"] == "STAGED_ANSWER_BLIND_READY_FOR_GENERATION"


def test_reference_cannot_open_until_all_36_submissions_are_frozen():
    config, authorization, source = _ready_chain()
    staged = task2.stage_question(
        source,
        authorization,
        config,
        fetch_json=lambda _url: {
            "items": [
                {
                    "accepted_answer_id": 9002,
                    "body": "<p>Question body.</p>",
                    "last_activity_date": 1784707249,
                    "question_id": 513421,
                    "title": "Question title",
                }
            ]
        },
    )
    public = {
        "blindness": {"submissions_frozen": True},
        "content_sha256": "b" * 64,
        "staged_problem_content_sha256": staged["content_sha256"],
        "submissions": [{}] * 35,
    }

    with pytest.raises(task2.MathOverflowTask2Error, match="36 submissions"):
        task2.open_reference(public, staged, config, fetch_json=lambda _url: {"items": []})


def test_reference_opens_after_seal_but_does_not_auto_validate_candidates():
    config, authorization, source = _ready_chain()
    staged = task2.stage_question(
        source,
        authorization,
        config,
        fetch_json=lambda _url: {
            "items": [
                {
                    "accepted_answer_id": 9002,
                    "body": "<p>Question body.</p>",
                    "last_activity_date": 1784707249,
                    "question_id": 513421,
                    "title": "Question title",
                }
            ]
        },
    )
    public_body = {
        "blindness": {"submissions_frozen": True},
        "staged_problem_content_sha256": staged["content_sha256"],
        "submissions": [{}] * 36,
    }
    public = dict(public_body, content_sha256=canonical_sha256(public_body))
    reference = task2.open_reference(
        public,
        staged,
        config,
        fetch_json=lambda _url: {
            "items": [
                {
                    "answer_id": 9002,
                    "body": "<p>Here is an independently authored counterexample.</p>",
                    "is_accepted": True,
                    "link": "https://mathoverflow.net/a/9002",
                    "question_id": 513421,
                }
            ]
        },
    )

    assert reference["reference_opened_after_submissions_frozen"] is True
    assert "independently authored" in reference["accepted_answer"]
    assert reference["claims"]["candidate_correctness_automatically_established"] is False


def test_authorization_tamper_fails_closed():
    config = _config()
    authorization = dict(_authorization(config))
    authorization["question_bodies_read"] = 1

    with pytest.raises(task2.MathOverflowTask2Error, match="seal"):
        task2.validate_authorization(authorization, ROOT, config)
