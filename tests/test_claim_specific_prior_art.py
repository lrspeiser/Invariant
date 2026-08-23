from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from sigma_theory_compiler.claim_specific_prior_art import (
    HTTPResponse,
    PriorArtError,
    adjudicate_screen,
    load_config,
    run_screen,
    validate_screen,
)

ROOT = Path(__file__).resolve().parents[1]
CLAIM = ROOT / "configs/claims/live_uncertain_recaman_behavior.json"
NOW = "2026-08-23T06:00:00Z"


def _transport(uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
    parsed = urlparse(uri)
    query = parse_qs(parsed.query)
    if parsed.hostname == "oeis.org":
        body = json.dumps(
            [
                {
                    "number": 5132,
                    "name": "Recaman's sequence",
                    "data": "0,1,3,6,2,7,13,20,12,21,11,22,10,23,9,24,8,25,43",
                }
            ]
        ).encode()
        return HTTPResponse(200, {"content-type": "application/json"}, body)
    if parsed.hostname == "api.crossref.org":
        assert "query.bibliographic" in query
        body = json.dumps(
            {
                "message": {
                    "items": [
                        {"DOI": "10.1000/recaman-control", "title": ["A sequence control"]}
                    ]
                }
            }
        ).encode()
        return HTTPResponse(200, {"content-type": "application/json"}, body)
    if parsed.hostname == "export.arxiv.org":
        body = b"""<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'>
        <entry><id>https://arxiv.org/abs/1234.5678</id><title>Parity recurrence control</title></entry>
        </feed>"""
        return HTTPResponse(200, {"content-type": "application/atom+xml"}, body)
    return HTTPResponse(429, {"content-type": "application/json"}, b'{"error":"rate limited"}')


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return run_screen(ROOT, CLAIM, transport=_transport, retrieved_utc=NOW)


def test_screen_is_claim_bound_multichannel_and_non_authoritative(receipt: dict[str, object]) -> None:
    validate_screen(receipt, ROOT)
    assert receipt["automated_screen"] == {
        "status": "COMPLETED_NOT_NOVELTY_CLEARED",
        "successful_external_providers": 3,
        "total_external_providers": 4,
    }
    assert len(receipt["queries"]) == 5
    assert receipt["channel_assessment"]["behavior"] == "KNOWN_EXTERNAL_BEHAVIOR_MATCH"
    assert receipt["llm_origin_calibration"]["model_assessment"] == "uncertain"
    assert receipt["human_review"]["status"] == "PENDING_NAMED_HUMAN_REVIEW"
    assert receipt["release_gate"]["novelty_language_authorized"] is False
    assert not any(receipt["claims"].values())


def test_rate_limit_and_response_hashes_are_evidence_not_silent_success(
    receipt: dict[str, object],
) -> None:
    providers = {item["provider_id"]: item for item in receipt["provider_evidence"]}
    assert providers["semantic_scholar"]["status"] == "UNAVAILABLE_RECORDED"
    assert providers["semantic_scholar"]["error"] == "HTTP_429"
    for provider_id in ("oeis", "crossref", "arxiv"):
        assert providers[provider_id]["status"] == "COMPLETED"
        assert len(providers[provider_id]["response_sha256"]) == 64


def test_source_or_receipt_tamper_fails_closed(receipt: dict[str, object]) -> None:
    tampered = json.loads(json.dumps(receipt))
    tampered["release_gate"]["novelty_language_authorized"] = True
    with pytest.raises(PriorArtError):
        validate_screen(tampered)
    tampered = json.loads(json.dumps(receipt))
    tampered["source_bindings"]["claim"]["sha256"] = "0" * 64
    tampered["content_sha256"] = __import__(
        "sigma_theory_compiler.sigma_core", fromlist=["canonical_sha256"]
    ).canonical_sha256({key: value for key, value in tampered.items() if key != "content_sha256"})
    with pytest.raises(PriorArtError):
        validate_screen(tampered, ROOT)


def test_named_human_review_is_required_and_still_does_not_establish_novelty(
    receipt: dict[str, object],
) -> None:
    review = {
        "schema_version": "invariant-claim-prior-art-human-review-1.0",
        "claim_id": receipt["claim_id"],
        "screen_content_sha256": receipt["content_sha256"],
        "reviewer": {"name": "Ada Lovelace", "affiliation_or_independent": "independent"},
        "reviewed_utc": NOW,
        "nearest_match_decisions": [
            {
                "identifier": match["identifier"],
                "classification": (
                    "behavior_only" if match["provider_id"] == "oeis" else "unrelated"
                ),
                "notes": "Inspected against the behavior, construction, and proof channels.",
            }
            for match in receipt["nearest_matches"]
        ],
        "overall_classification": "unresolved",
        "notes": "A known behavior does not settle construction or proof provenance.",
    }
    adjudicated = adjudicate_screen(receipt, review, load_config(ROOT))
    assert adjudicated["release_gate"]["claim_prior_art_complete"] is True
    assert adjudicated["release_gate"]["novelty_language_authorized"] is False
    assert not any(adjudicated["claims"].values())
    incomplete = json.loads(json.dumps(review))
    incomplete["nearest_match_decisions"].pop()
    with pytest.raises(PriorArtError):
        adjudicate_screen(receipt, incomplete, load_config(ROOT))
    review["reviewer"]["name"] = "pending"
    with pytest.raises(PriorArtError):
        adjudicate_screen(receipt, review, load_config(ROOT))
