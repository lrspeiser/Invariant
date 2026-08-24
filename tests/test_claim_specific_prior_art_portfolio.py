from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.claim_specific_prior_art import (
    HTTPResponse,
    PriorArtError,
    adjudicate_screen,
    load_claim,
    load_config,
    rebind_screen,
    run_screen,
)
from sigma_theory_compiler.claim_specific_prior_art_portfolio import (
    CLAIM_DIRECTORY,
    COMPLETE_PATH,
    CONFIG_PATH,
    PREFLIGHT_PATH,
    REVIEW_DIRECTORY,
    SCREEN_DIRECTORY,
    PriorArtPortfolioError,
    build_pending_review_form,
    build_preflight,
    screen_portfolio,
    validate_batch_receipt,
    validate_pending_review_form,
    validate_preflight,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-24T08:00:00Z"


def _transport(uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
    if "oeis.org" in uri:
        return HTTPResponse(200, {"content-type": "application/json"}, b"[]")
    if "crossref.org" in uri:
        body = json.dumps(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/portfolio-control",
                            "title": ["Portfolio prior-art control"],
                        }
                    ]
                }
            }
        ).encode()
        return HTTPResponse(200, {"content-type": "application/json"}, body)
    if "arxiv.org" in uri:
        body = b"""<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'>
        <entry><id>https://arxiv.org/abs/1234.5678</id><title>Portfolio control</title></entry>
        </feed>"""
        return HTTPResponse(200, {"content-type": "application/atom+xml"}, body)
    body = json.dumps(
        {
            "data": [
                {
                    "paperId": "portfolio-control-paper",
                    "title": "Portfolio control",
                    "url": "https://www.semanticscholar.org/paper/portfolio-control-paper",
                }
            ]
        }
    ).encode()
    return HTTPResponse(200, {"content-type": "application/json"}, body)


def _mini_root(tmp_path: Path) -> Path:
    for relative in (
        "configs/claim_specific_prior_art.json",
        CONFIG_PATH,
        "src/sigma_theory_compiler/claim_specific_prior_art.py",
        "src/sigma_theory_compiler/claim_specific_prior_art_portfolio.py",
        "runs/math/retained-piecewise-descendant-campaign/live-runtime.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def test_committed_portfolio_binds_every_retained_descendant_without_pruning() -> None:
    preflight = json.loads((ROOT / PREFLIGHT_PATH).read_text(encoding="utf-8"))
    validate_preflight(preflight, ROOT)
    assert preflight["summary"] == {
        "executable_claims": 16,
        "maximum_external_requests": 88,
        "nonexecutable_claims_retained": 8,
        "retained_claims": 24,
        "status": "READY_FOR_RESUMABLE_CLAIM_SPECIFIC_SCREENING",
    }
    assert {item["llm_origin_assessment"] for item in preflight["claims"]} == {
        "known_rewrite",
        "cross_domain_synthesis",
        "proposed_new_construction",
        "uncertain",
    }
    for row in preflight["claims"]:
        claim = load_claim(ROOT, ROOT / row["claim_path"])
        assert claim["source_binding"]["source_id"] == row["source_id"]


def test_complete_live_portfolio_binds_every_screen_and_pending_human_form() -> None:
    preflight = json.loads((ROOT / PREFLIGHT_PATH).read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / COMPLETE_PATH).read_text(encoding="utf-8"))
    validate_batch_receipt(receipt, preflight, ROOT)
    assert receipt["batch"]["remaining_claims"] == 0
    assert receipt["batch"]["cumulative_external_request_budget"] == 88
    assert len(receipt["batch"]["completed_claims"]) == 24
    assert receipt["release_gate"] == {
        "all_automated_screens_complete": True,
        "all_named_human_reviews_complete": False,
        "novelty_language_authorized": False,
        "status": "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED",
    }
    assert (
        sum(
            row["behavior_assessment"] == "NOT_APPLICABLE_NO_EXECUTABLE_BEHAVIOR"
            for row in receipt["batch"]["completed_claims"]
        )
        == 8
    )


def test_preflight_rejects_resealed_descendant_substitution() -> None:
    preflight = json.loads((ROOT / PREFLIGHT_PATH).read_text(encoding="utf-8"))
    changed = deepcopy(preflight)
    changed["claims"][0]["source_content_sha256"] = "0" * 64
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(PriorArtPortfolioError, match="claim binding"):
        validate_preflight(changed, ROOT)


def test_resumable_batch_skips_behavior_request_for_nonexecutable_branches(
    tmp_path: Path,
) -> None:
    root = _mini_root(tmp_path)
    preflight = build_preflight(root, Path(CLAIM_DIRECTORY))
    preflight_path = root / PREFLIGHT_PATH
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    first = screen_portfolio(
        root,
        Path(PREFLIGHT_PATH),
        Path(SCREEN_DIRECTORY),
        Path(REVIEW_DIRECTORY),
        maximum_claims=4,
        transport=_transport,
        retrieved_utc=NOW,
    )
    validate_batch_receipt(first, preflight)
    assert first["batch"]["external_request_budget_consumed"] == sum(
        row["external_request_budget"] for row in preflight["claims"][:4]
    )
    assert len(first["batch"]["newly_screened_source_ids"]) == 4
    assert all(
        row["llm_origin_assessment"] == "proposed_new_construction"
        for row in preflight["claims"][:4]
    )
    nonexecutable = next(
        row for row in preflight["claims"] if row["behavior_status"] == "NO_EXECUTABLE_BEHAVIOR"
    )
    screen = run_screen(
        root,
        root / nonexecutable["claim_path"],
        transport=_transport,
        retrieved_utc=NOW,
    )
    providers = {item["provider_id"]: item for item in screen["provider_evidence"]}
    assert providers["oeis"]["status"] == "NOT_APPLICABLE_RECORDED"
    assert screen["channel_assessment"]["behavior"] == ("NOT_APPLICABLE_NO_EXECUTABLE_BEHAVIOR")
    form = build_pending_review_form(screen)
    validate_pending_review_form(form, screen)
    with pytest.raises(PriorArtError):
        adjudicate_screen(screen, form, load_config(root))

    second = screen_portfolio(
        root,
        Path(PREFLIGHT_PATH),
        Path(SCREEN_DIRECTORY),
        Path(REVIEW_DIRECTORY),
        maximum_claims=4,
        transport=_transport,
        retrieved_utc=NOW,
    )
    validate_batch_receipt(second, preflight)
    assert len(second["batch"]["reused_valid_source_ids"]) == 4
    assert len(second["batch"]["newly_screened_source_ids"]) == 4
    assert second["batch"]["remaining_claims"] == 16
    assert second["release_gate"]["novelty_language_authorized"] is False


def test_screen_rebind_reuses_provider_evidence_only_for_identical_queries(
    tmp_path: Path,
) -> None:
    root = _mini_root(tmp_path)
    preflight = build_preflight(root, Path(CLAIM_DIRECTORY))
    selected = preflight["claims"][0]
    claim_path = root / selected["claim_path"]
    original = run_screen(root, claim_path, transport=_transport, retrieved_utc=NOW)

    descendant_path = root / "runs/math/retained-piecewise-descendant-campaign/live-runtime.json"
    descendant = json.loads(descendant_path.read_text(encoding="utf-8"))
    descendant["rebind_test_marker"] = True
    descendant["content_sha256"] = canonical_sha256(
        {key: value for key, value in descendant.items() if key != "content_sha256"}
    )
    descendant_path.write_text(json.dumps(descendant), encoding="utf-8")
    config_path = root / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source"]["expected_content_sha256"] = descendant["content_sha256"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rebound_preflight = build_preflight(root, Path(CLAIM_DIRECTORY))
    rebound_row = next(
        row for row in rebound_preflight["claims"] if row["source_id"] == selected["source_id"]
    )
    rebound_claim_path = root / rebound_row["claim_path"]
    rebound = rebind_screen(root, rebound_claim_path, original)
    assert rebound["provider_evidence"] == original["provider_evidence"]
    assert rebound["retrieved_utc"] == original["retrieved_utc"]
    assert (
        rebound["source_bindings"]["idea"]["receipt_content_sha256"] == descendant["content_sha256"]
    )

    changed_claim = json.loads(rebound_claim_path.read_text(encoding="utf-8"))
    changed_claim["claim"]["formula_or_construction"] += " + semantic_mutation"
    rebound_claim_path.write_text(json.dumps(changed_claim), encoding="utf-8")
    with pytest.raises(PriorArtError, match="semantic claim or query set"):
        rebind_screen(root, rebound_claim_path, original)


def test_batch_receipt_cannot_reseal_novelty_authority(tmp_path: Path) -> None:
    root = _mini_root(tmp_path)
    preflight = build_preflight(root, Path(CLAIM_DIRECTORY))
    preflight_path = root / PREFLIGHT_PATH
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    receipt = screen_portfolio(
        root,
        Path(PREFLIGHT_PATH),
        Path(SCREEN_DIRECTORY),
        Path(REVIEW_DIRECTORY),
        maximum_claims=1,
        transport=_transport,
        retrieved_utc=NOW,
    )
    changed = deepcopy(receipt)
    changed["release_gate"]["novelty_language_authorized"] = True
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(PriorArtPortfolioError, match="release boundary"):
        validate_batch_receipt(changed, preflight)
