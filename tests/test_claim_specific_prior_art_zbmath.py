from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.claim_specific_prior_art import HTTPResponse
from sigma_theory_compiler.claim_specific_prior_art_portfolio import (
    validate_pending_review_form,
)
from sigma_theory_compiler.claim_specific_prior_art_zbmath import (
    OUTPUT_PATH,
    ZbMathSupplementError,
    build_supplement,
    validate_supplement,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-24T12:00:00Z"


def _success_body() -> bytes:
    return json.dumps(
        {
            "result": [
                {
                    "contributors": {
                        "authors": [
                            {"name": "Noether, Emmy"},
                            {"name": "Noether, Emmy"},
                        ]
                    },
                    "editorial_contributions": [
                        {"text": "This copyrighted review must never enter the receipt."}
                    ],
                    "id": 1234567,
                    "identifier": "0999.00001",
                    "links": [
                        {
                            "identifier": "2401.00001",
                            "type": "arxiv",
                            "url": "https://arxiv.org/abs/2401.00001",
                        },
                        {
                            "identifier": "insecure",
                            "type": "other",
                            "url": "http://example.test/not-retained",
                        },
                    ],
                    "msc": [{"code": "11B37", "text": "Recurrences"}],
                    "title": {"title": "A math-specific prior-art control"},
                    "year": "2024",
                    "zbmath_url": "https://zbmath.org/1234567",
                },
                {
                    "contributors": {"authors": []},
                    "id": 1234567,
                    "links": [],
                    "msc": [],
                    "title": {"title": "Duplicate identifier"},
                    "year": "2024",
                    "zbmath_url": "https://zbmath.org/1234567",
                },
            ],
            "status": {
                "execution_bool": True,
                "nr_request_results": 2,
                "status_code": 200,
            },
        }
    ).encode()


def _success_transport(uri: str, headers: object, timeout: int, maximum: int) -> HTTPResponse:
    assert uri.startswith("https://api.zbmath.org/v1/document/_search?")
    assert "search_string=" in uri and "results_per_page=5" in uri
    assert headers == {
        "Accept": "application/json",
        "User-Agent": "InvariantPriorArtSupplement/0.1 (https://github.com/lrspeiser/Invariant)",
    }
    assert timeout == 30 and maximum == 1_000_000
    return HTTPResponse(200, {"content-type": "application/json"}, _success_body())


def _no_results_transport(
    _uri: str, _headers: object, _timeout: int, _maximum: int
) -> HTTPResponse:
    return HTTPResponse(
        404,
        {"content-type": "application/json"},
        json.dumps(
            {
                "result": None,
                "status": {
                    "execution_bool": False,
                    "internal_code": "successful access. No results found.",
                    "status_code": 404,
                },
            }
        ).encode(),
    )


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_fake_math_specific_screen_covers_all_25_claims_without_pruning() -> None:
    receipt = build_supplement(ROOT, transport=_success_transport, retrieved_utc=NOW)
    validate_supplement(receipt, ROOT)
    assert receipt["summary"]["claim_count"] == 25
    assert receipt["summary"]["origin_counts"] == {
        "cross_domain_synthesis": 4,
        "known_rewrite": 6,
        "proposed_new_construction": 8,
        "uncertain": 7,
    }
    assert receipt["summary"]["unique_zbmath_results"] == 1
    assert receipt["summary"]["zbmath_status_counts"] == {
        "COMPLETED": 25,
        "COMPLETED_NO_RESULTS": 0,
        "UNAVAILABLE_RECORDED": 0,
    }
    assert receipt["claims_review_dossier"][-1]["source_id"] == "earlier-a005132"
    encoded = json.dumps(receipt)
    assert "editorial_contributions" not in encoded
    assert "copyrighted review" not in encoded
    assert "http://example.test" not in encoded
    assert receipt["release_gate"]["novelty_language_authorized"] is False


def test_no_results_is_completed_absence_but_never_novelty() -> None:
    receipt = build_supplement(ROOT, transport=_no_results_transport, retrieved_utc=NOW)
    assert receipt["summary"]["zbmath_status_counts"]["COMPLETED_NO_RESULTS"] == 25
    assert receipt["claims"]["automated_absence_establishes_novelty"] is False
    assert receipt["release_gate"]["status"] == "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED"


def test_malformed_success_fails_closed() -> None:
    def malformed(_uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
        return HTTPResponse(200, {}, b"not-json")

    with pytest.raises(ZbMathSupplementError, match="malformed success"):
        build_supplement(ROOT, transport=malformed, retrieved_utc=NOW)


def test_resealed_novelty_or_request_substitution_is_rejected() -> None:
    receipt = build_supplement(ROOT, transport=_success_transport, retrieved_utc=NOW)
    novelty = deepcopy(receipt)
    novelty["release_gate"]["novelty_language_authorized"] = True
    _reseal(novelty)
    with pytest.raises(ZbMathSupplementError, match="opened a novelty gate"):
        validate_supplement(novelty, ROOT)

    request = deepcopy(receipt)
    request["claims_review_dossier"][0]["zbmath_evidence"]["request_uri"] = (
        "https://example.test/rewritten"
    )
    _reseal(request)
    with pytest.raises(ZbMathSupplementError, match="escaped authority"):
        validate_supplement(request, ROOT)


def test_earlier_a005132_pending_form_is_machine_valid() -> None:
    screen = json.loads(
        (ROOT / "runs/math/claim-specific-prior-art/live-uncertain-recaman.json").read_text(
            encoding="utf-8"
        )
    )
    form = json.loads(
        (
            ROOT
            / "runs/math/claim-specific-prior-art/live-uncertain-recaman-human-review-form.json"
        ).read_text(encoding="utf-8")
    )
    validate_pending_review_form(form, screen)
    assert form["reviewer"]["name"] == "pending"
    assert len(form["nearest_match_decisions"]) == 12


def test_committed_live_supplement_is_sealed_and_still_human_blocked() -> None:
    receipt = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_supplement(receipt, ROOT)
    assert receipt["summary"]["claim_count"] == 25
    assert receipt["release_gate"] == {
        "all_named_human_reviews_complete": False,
        "novelty_language_authorized": False,
        "status": "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED",
    }
