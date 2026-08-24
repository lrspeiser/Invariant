"""Prepare and resumably screen every retained descendant for claim-specific prior art.

The portfolio keeps non-executable branches instead of deleting them. Executable descendants get
behavior and literature searches; non-executable descendants explicitly skip only the meaningless
OEIS behavior request and still receive construction, representation, proof, and repository
searches. Automated results always emit an incomplete named-human review form and cannot authorize
novelty language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .claim_specific_prior_art import (
    DESCENDANT_CLAIM_SCHEMA,
    REVIEW_SCHEMA,
    Transport,
    build_queries,
    load_claim,
    rebind_screen,
    run_screen,
    urllib_transport,
    validate_screen,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/claim_specific_prior_art_portfolio.json"
CLAIM_CONFIG_PATH = "configs/claim_specific_prior_art.json"
CLAIM_MODULE_PATH = "src/sigma_theory_compiler/claim_specific_prior_art.py"
PORTFOLIO_MODULE_PATH = "src/sigma_theory_compiler/claim_specific_prior_art_portfolio.py"
CLAIM_DIRECTORY = "runs/math/claim-specific-prior-art/descendant-claims"
PREFLIGHT_PATH = "runs/math/claim-specific-prior-art/descendant-portfolio-preflight.json"
SCREEN_DIRECTORY = "runs/math/claim-specific-prior-art/descendant-screens"
REVIEW_DIRECTORY = "runs/math/claim-specific-prior-art/descendant-human-review-forms"
COMPLETE_PATH = "runs/math/claim-specific-prior-art/descendant-portfolio-complete.json"
CONFIG_SCHEMA = "invariant-claim-specific-prior-art-portfolio-config-1.0"
PREFLIGHT_SCHEMA = "invariant-claim-specific-prior-art-portfolio-preflight-1.0"
BATCH_SCHEMA = "invariant-claim-specific-prior-art-portfolio-batch-1.0"

_HEX = frozenset("0123456789abcdef")
_ORIGINS = {
    "known_rewrite",
    "cross_domain_synthesis",
    "proposed_new_construction",
    "uncertain",
}
_SCREENING_PRIORITY = [
    "proposed_new_construction",
    "uncertain",
    "cross_domain_synthesis",
    "known_rewrite",
]


class PriorArtPortfolioError(ValueError):
    """The retained-descendant prior-art portfolio failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PriorArtPortfolioError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PriorArtPortfolioError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _under(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise PriorArtPortfolioError(f"{label} escapes the repository") from error
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PriorArtPortfolioError(f"{label} is not an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_portfolio_config(root: Path) -> dict[str, Any]:
    value = _read_json(root / CONFIG_PATH, "prior-art portfolio config")
    _strict(
        value,
        {
            "portfolio_id",
            "release_policy",
            "request_policy",
            "schema_version",
            "selection",
            "source",
        },
        "prior-art portfolio config",
    )
    if value["schema_version"] != CONFIG_SCHEMA:
        raise PriorArtPortfolioError("prior-art portfolio config schema changed")
    _strict(value["source"], {"expected_content_sha256", "path"}, "portfolio source")
    _sha(value["source"]["expected_content_sha256"], "portfolio source content hash")
    _strict(
        value["selection"],
        {
            "include_executable",
            "include_nonexecutable",
            "maximum_behavior_terms",
            "required_retained_descendants",
            "screening_priority",
        },
        "portfolio selection",
    )
    selection = value["selection"]
    if (
        selection["include_executable"] is not True
        or selection["include_nonexecutable"] is not True
        or selection["maximum_behavior_terms"] != 18
        or selection["required_retained_descendants"] != 24
        or selection["screening_priority"] != _SCREENING_PRIORITY
    ):
        raise PriorArtPortfolioError("portfolio selection weakened")
    _strict(
        value["request_policy"],
        {
            "executable_external_requests_per_claim",
            "maximum_claims_per_batch",
            "maximum_external_requests",
            "nonexecutable_external_requests_per_claim",
            "resume_from_valid_receipts",
        },
        "portfolio request policy",
    )
    request = value["request_policy"]
    if request != {
        "executable_external_requests_per_claim": 4,
        "maximum_claims_per_batch": 4,
        "maximum_external_requests": 90,
        "nonexecutable_external_requests_per_claim": 3,
        "resume_from_valid_receipts": True,
    }:
        raise PriorArtPortfolioError("portfolio request policy changed")
    _strict(
        value["release_policy"],
        {
            "automated_absence_establishes_novelty",
            "named_human_review_required",
            "retain_unscreened_claims",
        },
        "portfolio release policy",
    )
    if value["release_policy"] != {
        "automated_absence_establishes_novelty": False,
        "named_human_review_required": True,
        "retain_unscreened_claims": True,
    }:
        raise PriorArtPortfolioError("portfolio release boundary weakened")
    return value


def _load_source(root: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_path = _under(root, Path(config["source"]["path"]), "portfolio source")
    receipt = _read_json(source_path, "retained-descendant receipt")
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    if (
        receipt.get("content_sha256") != canonical_sha256(body)
        or receipt.get("content_sha256") != config["source"]["expected_content_sha256"]
        or receipt.get("summary", {}).get("descendant_ideas_retained") != 24
    ):
        raise PriorArtPortfolioError("retained-descendant source binding changed")
    descendants = receipt.get("descendants", [])
    if (
        not isinstance(descendants, list)
        or len(descendants) != config["selection"]["required_retained_descendants"]
        or any(item.get("retention_status") != "RETAINED_ACTIVE" for item in descendants)
    ):
        raise PriorArtPortfolioError("retained-descendant portfolio changed")
    return source_path, receipt


def build_claim_document(
    descendant: Mapping[str, Any], receipt: Mapping[str, Any], source_path: str
) -> dict[str, Any]:
    hypothesis = descendant.get("hypothesis", {})
    execution = descendant.get("execution", {})
    predictions = execution.get("independent_predictions", [])
    executable = descendant.get("admission", {}).get("status") == "ADMITTED_EXECUTABLE"
    if executable and (not isinstance(predictions, list) or len(predictions) < 6):
        raise PriorArtPortfolioError("executable descendant lacks behavior predictions")
    if not executable and predictions:
        raise PriorArtPortfolioError("non-executable descendant unexpectedly has predictions")
    selected_routes = execution.get("proof_plan_search", {}).get("selected_route", [])
    declared_plan = hypothesis.get("proof_plan", [])
    proof_mechanisms = [str(item)[:500] for item in [*selected_routes, *declared_plan] if str(item)]
    known_analogues = [
        str(item)[:500] for item in hypothesis.get("known_analogues", []) if str(item)
    ]
    source_domains = [
        str(item)[:500] for item in hypothesis.get("source_idea_domains", []) if str(item)
    ]
    if not proof_mechanisms:
        proof_mechanisms = ["model supplied no executable proof route"]
    if not known_analogues:
        known_analogues = ["model supplied no named analogue"]
    if not source_domains:
        source_domains = ["model supplied no source domain"]
    source_id = descendant.get("descendant_id")
    if not isinstance(source_id, str) or not source_id.startswith("descendant."):
        raise PriorArtPortfolioError("descendant ID changed")
    origin = descendant.get("llm_self_assessed_origin")
    if origin not in _ORIGINS:
        raise PriorArtPortfolioError("descendant origin label changed")
    statement = str(hypothesis.get("rationale") or hypothesis.get("synthesis_note") or source_id)
    document = {
        "schema_version": DESCENDANT_CLAIM_SCHEMA,
        "claim_id": f"retained-{source_id}.prior-art",
        "source_binding": {
            "path": source_path,
            "receipt_content_sha256": receipt["content_sha256"],
            "source_content_sha256": canonical_sha256(descendant),
            "source_id": source_id,
            "source_kind": "retained_descendant",
        },
        "claim": {
            "behavior_status": (
                "EXECUTABLE_PREDICTIONS" if executable else "NO_EXECUTABLE_BEHAVIOR"
            ),
            "behavior_terms": [str(item) for item in predictions[:18]] if executable else [],
            "formula_or_construction": str(hypothesis.get("expression", "")),
            "known_analogues": known_analogues,
            "llm_origin_assessment": origin,
            "proof_mechanisms": proof_mechanisms,
            "representation": str(hypothesis.get("representation", "")),
            "source_domains": source_domains,
            "statement": statement[:2000],
        },
    }
    return document


def build_preflight(root: Path, claim_directory: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_portfolio_config(root)
    source_path, receipt = _load_source(root, config)
    claim_directory = _under(root, claim_directory, "portfolio claim directory")
    rows = []
    executable_count = 0
    nonexecutable_count = 0
    external_requests = 0
    for source_order, descendant in enumerate(receipt["descendants"]):
        document = build_claim_document(
            descendant, receipt, source_path.relative_to(root).as_posix()
        )
        source_id = document["source_binding"]["source_id"]
        claim_path = claim_directory / f"{source_id}.json"
        _write_json_atomic(claim_path, document)
        load_claim(root, claim_path)
        executable = document["claim"]["behavior_status"] == "EXECUTABLE_PREDICTIONS"
        request_count = 4 if executable else 3
        executable_count += int(executable)
        nonexecutable_count += int(not executable)
        external_requests += request_count
        rows.append(
            {
                "behavior_status": document["claim"]["behavior_status"],
                "claim_id": document["claim_id"],
                "claim_path": claim_path.relative_to(root).as_posix(),
                "claim_sha256": _normalized_file_sha256(claim_path),
                "external_request_budget": request_count,
                "llm_origin_assessment": document["claim"]["llm_origin_assessment"],
                "query_set_sha256": canonical_sha256(list(build_queries(document))),
                "representation": document["claim"]["representation"],
                "source_order": source_order,
                "source_content_sha256": document["source_binding"]["source_content_sha256"],
                "source_id": source_id,
            }
        )
    priority = {origin: rank for rank, origin in enumerate(_SCREENING_PRIORITY)}
    rows.sort(key=lambda item: (priority[item["llm_origin_assessment"]], item["source_order"]))
    for queue_rank, row in enumerate(rows):
        row["queue_rank"] = queue_rank
    body: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "portfolio_id": config["portfolio_id"],
        "source_bindings": {
            "claim_specific_module": {
                "path": CLAIM_MODULE_PATH,
                "sha256": _normalized_file_sha256(root / CLAIM_MODULE_PATH),
            },
            "claim_specific_config": {
                "path": CLAIM_CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / CLAIM_CONFIG_PATH),
            },
            "portfolio_config": {
                "path": CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / CONFIG_PATH),
            },
            "portfolio_module": {
                "path": PORTFOLIO_MODULE_PATH,
                "sha256": _normalized_file_sha256(root / PORTFOLIO_MODULE_PATH),
            },
            "retained_descendant_receipt": {
                "content_sha256": receipt["content_sha256"],
                "path": source_path.relative_to(root).as_posix(),
            },
        },
        "claims": rows,
        "summary": {
            "executable_claims": executable_count,
            "maximum_external_requests": external_requests,
            "nonexecutable_claims_retained": nonexecutable_count,
            "retained_claims": len(rows),
            "status": "READY_FOR_RESUMABLE_CLAIM_SPECIFIC_SCREENING",
        },
        "release_gate": {
            "all_automated_screens_complete": False,
            "all_named_human_reviews_complete": False,
            "novelty_language_authorized": False,
            "status": "BLOCKED_SCREENING_AND_NAMED_HUMAN_REVIEW_INCOMPLETE",
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_preflight(body, root)
    return body


def validate_preflight(value: Mapping[str, Any], root: Path | None = None) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise PriorArtPortfolioError("portfolio preflight content seal changed")
    if value.get("schema_version") != PREFLIGHT_SCHEMA:
        raise PriorArtPortfolioError("portfolio preflight schema changed")
    claims = value.get("claims", [])
    summary = value.get("summary", {})
    origin_labels = [item.get("llm_origin_assessment") for item in claims]
    if any(origin not in _ORIGINS for origin in origin_labels):
        raise PriorArtPortfolioError("portfolio preflight origin labels changed")
    origin_ranks = [_SCREENING_PRIORITY.index(origin) for origin in origin_labels]
    if (
        len(claims) != 24
        or len({item.get("source_id") for item in claims}) != 24
        or [item.get("queue_rank") for item in claims] != list(range(24))
        or origin_ranks != sorted(origin_ranks)
        or sum(item.get("behavior_status") == "EXECUTABLE_PREDICTIONS" for item in claims) != 16
        or sum(item.get("behavior_status") == "NO_EXECUTABLE_BEHAVIOR" for item in claims) != 8
        or sum(item.get("external_request_budget", 0) for item in claims) != 88
        or summary
        != {
            "executable_claims": 16,
            "maximum_external_requests": 88,
            "nonexecutable_claims_retained": 8,
            "retained_claims": 24,
            "status": "READY_FOR_RESUMABLE_CLAIM_SPECIFIC_SCREENING",
        }
    ):
        raise PriorArtPortfolioError("portfolio preflight coverage changed")
    if value.get("release_gate") != {
        "all_automated_screens_complete": False,
        "all_named_human_reviews_complete": False,
        "novelty_language_authorized": False,
        "status": "BLOCKED_SCREENING_AND_NAMED_HUMAN_REVIEW_INCOMPLETE",
    }:
        raise PriorArtPortfolioError("portfolio preflight release boundary changed")
    if root is None:
        return
    root = root.resolve()
    config = load_portfolio_config(root)
    source_path, receipt = _load_source(root, config)
    bindings = value.get("source_bindings", {})
    if bindings != {
        "claim_specific_module": {
            "path": CLAIM_MODULE_PATH,
            "sha256": _normalized_file_sha256(root / CLAIM_MODULE_PATH),
        },
        "claim_specific_config": {
            "path": CLAIM_CONFIG_PATH,
            "sha256": _normalized_file_sha256(root / CLAIM_CONFIG_PATH),
        },
        "portfolio_config": {
            "path": CONFIG_PATH,
            "sha256": _normalized_file_sha256(root / CONFIG_PATH),
        },
        "portfolio_module": {
            "path": PORTFOLIO_MODULE_PATH,
            "sha256": _normalized_file_sha256(root / PORTFOLIO_MODULE_PATH),
        },
        "retained_descendant_receipt": {
            "content_sha256": receipt["content_sha256"],
            "path": source_path.relative_to(root).as_posix(),
        },
    }:
        raise PriorArtPortfolioError("portfolio preflight source binding changed")
    for row in claims:
        claim_path = _under(root, Path(row["claim_path"]), "portfolio claim")
        document = load_claim(root, claim_path)
        if (
            row["claim_sha256"] != _normalized_file_sha256(claim_path)
            or row["claim_id"] != document["claim_id"]
            or row["source_id"] != document["source_binding"]["source_id"]
            or row["source_content_sha256"] != document["source_binding"]["source_content_sha256"]
            or row["query_set_sha256"] != canonical_sha256(list(build_queries(document)))
        ):
            raise PriorArtPortfolioError("portfolio claim binding changed")


def build_pending_review_form(screen: Mapping[str, Any]) -> dict[str, Any]:
    validate_screen(screen)
    return {
        "schema_version": REVIEW_SCHEMA,
        "claim_id": screen["claim_id"],
        "screen_content_sha256": screen["content_sha256"],
        "reviewer": {"name": "pending", "affiliation_or_independent": "pending"},
        "reviewed_utc": "1970-01-01T00:00:00Z",
        "nearest_match_decisions": [
            {"identifier": item["identifier"], "classification": "pending", "notes": "pending"}
            for item in screen["nearest_matches"]
        ],
        "overall_classification": "unresolved",
        "notes": (
            "Replace every pending value after a specifically named human inspects behavior, "
            "formula/construction, representation, and proof-mechanism proximity."
        ),
    }


def validate_pending_review_form(value: Mapping[str, Any], screen: Mapping[str, Any]) -> None:
    validate_screen(screen)
    if (
        value != build_pending_review_form(screen)
        or value.get("reviewer", {}).get("name") != "pending"
        or any(
            item.get("classification") != "pending"
            for item in value.get("nearest_match_decisions", [])
        )
    ):
        raise PriorArtPortfolioError("pending human review form changed")


def screen_portfolio(
    root: Path,
    preflight_path: Path,
    screen_directory: Path,
    review_directory: Path,
    *,
    maximum_claims: int,
    transport: Transport = urllib_transport,
    retrieved_utc: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_portfolio_config(root)
    if not 1 <= maximum_claims <= config["request_policy"]["maximum_claims_per_batch"]:
        raise PriorArtPortfolioError("portfolio batch claim limit changed")
    preflight_path = _under(root, preflight_path, "portfolio preflight")
    preflight = _read_json(preflight_path, "portfolio preflight")
    validate_preflight(preflight, root)
    screen_directory = _under(root, screen_directory, "portfolio screen directory")
    review_directory = _under(root, review_directory, "portfolio review directory")
    newly_screened = []
    reused = []
    completed_ids = set()
    completed_claims = []
    for row in preflight["claims"]:
        screen_path = screen_directory / f"{row['source_id']}.json"
        review_path = review_directory / f"{row['source_id']}.json"
        if screen_path.is_file():
            screen = _read_json(screen_path, "portfolio screen")
            validate_screen(screen, root)
            if screen["claim_id"] != row["claim_id"]:
                raise PriorArtPortfolioError("resumed screen is bound to another claim")
            form = _read_json(review_path, "pending human review form")
            validate_pending_review_form(form, screen)
            reused.append(row["source_id"])
            completed_ids.add(row["source_id"])
            completed_claims.append(
                _completed_claim_row(root, row, screen_path, review_path, screen)
            )
            continue
        if len(newly_screened) >= maximum_claims:
            continue
        screen = run_screen(
            root,
            root / row["claim_path"],
            transport=transport,
            retrieved_utc=retrieved_utc,
        )
        _write_json_atomic(screen_path, screen)
        form = build_pending_review_form(screen)
        _write_json_atomic(review_path, form)
        validate_screen(_read_json(screen_path, "portfolio screen"), root)
        validate_pending_review_form(_read_json(review_path, "pending human review form"), screen)
        newly_screened.append(row["source_id"])
        completed_ids.add(row["source_id"])
        completed_claims.append(_completed_claim_row(root, row, screen_path, review_path, screen))
    external_requests = sum(
        row["external_request_budget"]
        for row in preflight["claims"]
        if row["source_id"] in newly_screened
    )
    cumulative_requests = sum(
        row["external_request_budget"]
        for row in preflight["claims"]
        if row["source_id"] in completed_ids
    )
    remaining_claims = len(preflight["claims"]) - len(completed_ids)
    screening_complete = remaining_claims == 0
    body: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA,
        "portfolio_id": preflight["portfolio_id"],
        "source_bindings": {
            "preflight": {
                "content_sha256": preflight["content_sha256"],
                "path": preflight_path.relative_to(root).as_posix(),
            }
        },
        "batch": {
            "completed_claims": completed_claims,
            "cumulative_external_request_budget": cumulative_requests,
            "external_request_budget_consumed": external_requests,
            "newly_screened_source_ids": newly_screened,
            "remaining_claims": remaining_claims,
            "reused_valid_source_ids": reused,
        },
        "release_gate": {
            "all_automated_screens_complete": screening_complete,
            "all_named_human_reviews_complete": False,
            "novelty_language_authorized": False,
            "status": (
                "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED"
                if screening_complete
                else "BLOCKED_SCREENING_AND_NAMED_HUMAN_REVIEW_INCOMPLETE"
            ),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_batch_receipt(body, preflight, root)
    return body


def _completed_claim_row(
    root: Path,
    preflight_row: Mapping[str, Any],
    screen_path: Path,
    review_path: Path,
    screen: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "automated_screen_status": screen["automated_screen"]["status"],
        "behavior_assessment": screen["channel_assessment"]["behavior"],
        "human_review_status": screen["human_review"]["status"],
        "llm_origin_assessment": preflight_row["llm_origin_assessment"],
        "provider_statuses": {
            item["provider_id"]: item["status"] for item in screen["provider_evidence"]
        },
        "review_form_path": review_path.relative_to(root).as_posix(),
        "review_form_sha256": _normalized_file_sha256(review_path),
        "screen_content_sha256": screen["content_sha256"],
        "screen_file_sha256": _normalized_file_sha256(screen_path),
        "screen_path": screen_path.relative_to(root).as_posix(),
        "source_id": preflight_row["source_id"],
    }


def validate_batch_receipt(
    value: Mapping[str, Any], preflight: Mapping[str, Any], root: Path | None = None
) -> None:
    validate_preflight(preflight)
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("content_sha256") != canonical_sha256(body)
        or value.get("schema_version") != BATCH_SCHEMA
        or value.get("portfolio_id") != preflight["portfolio_id"]
        or value.get("source_bindings", {}).get("preflight", {}).get("content_sha256")
        != preflight["content_sha256"]
    ):
        raise PriorArtPortfolioError("portfolio batch seal changed")
    batch = value.get("batch", {})
    known = {item["source_id"] for item in preflight["claims"]}
    new = batch.get("newly_screened_source_ids", [])
    reused = batch.get("reused_valid_source_ids", [])
    completed = batch.get("completed_claims", [])
    completed_ids = [item.get("source_id") for item in completed]
    selected = set(new) | set(reused)
    if (
        not isinstance(new, list)
        or not isinstance(reused, list)
        or not isinstance(completed, list)
        or len(new) > 4
        or len(set(new) | set(reused)) != len(new) + len(reused)
        or not selected.issubset(known)
        or len(completed_ids) != len(set(completed_ids))
        or set(completed_ids) != selected
        or batch.get("remaining_claims") != 24 - len(new) - len(reused)
        or batch.get("cumulative_external_request_budget")
        != sum(
            item["external_request_budget"]
            for item in preflight["claims"]
            if item["source_id"] in selected
        )
        or batch.get("external_request_budget_consumed")
        != sum(
            item["external_request_budget"]
            for item in preflight["claims"]
            if item["source_id"] in new
        )
    ):
        raise PriorArtPortfolioError("portfolio batch accounting changed")
    screening_complete = batch["remaining_claims"] == 0
    expected_release = {
        "all_automated_screens_complete": screening_complete,
        "all_named_human_reviews_complete": False,
        "novelty_language_authorized": False,
        "status": (
            "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED"
            if screening_complete
            else "BLOCKED_SCREENING_AND_NAMED_HUMAN_REVIEW_INCOMPLETE"
        ),
    }
    if value.get("release_gate") != expected_release:
        raise PriorArtPortfolioError("portfolio batch release boundary changed")
    expected_by_id = {item["source_id"]: item for item in preflight["claims"]}
    for row in completed:
        _strict(
            row,
            {
                "automated_screen_status",
                "behavior_assessment",
                "human_review_status",
                "llm_origin_assessment",
                "provider_statuses",
                "review_form_path",
                "review_form_sha256",
                "screen_content_sha256",
                "screen_file_sha256",
                "screen_path",
                "source_id",
            },
            "completed portfolio claim",
        )
        expected = expected_by_id[row["source_id"]]
        if (
            row["llm_origin_assessment"] != expected["llm_origin_assessment"]
            or row["human_review_status"] != "PENDING_NAMED_HUMAN_REVIEW"
            or row["automated_screen_status"]
            not in {"COMPLETED_NOT_NOVELTY_CLEARED", "INCOMPLETE_PROVIDER_COVERAGE"}
            or set(row["provider_statuses"]) != {"arxiv", "crossref", "oeis", "semantic_scholar"}
            or any(
                status not in {"COMPLETED", "NOT_APPLICABLE_RECORDED", "UNAVAILABLE_RECORDED"}
                for status in row["provider_statuses"].values()
            )
            or any(
                _sha(row[key], f"completed claim {key}") != row[key]
                for key in (
                    "review_form_sha256",
                    "screen_content_sha256",
                    "screen_file_sha256",
                )
            )
        ):
            raise PriorArtPortfolioError("completed portfolio claim evidence changed")
    if root is None:
        return
    root = root.resolve()
    preflight_path = _under(
        root,
        Path(value["source_bindings"]["preflight"]["path"]),
        "portfolio batch preflight",
    )
    if (
        _read_json(preflight_path, "portfolio batch preflight")["content_sha256"]
        != preflight["content_sha256"]
    ):
        raise PriorArtPortfolioError("portfolio batch preflight file changed")
    for row in completed:
        screen_path = _under(root, Path(row["screen_path"]), "completed portfolio screen")
        review_path = _under(root, Path(row["review_form_path"]), "portfolio review form")
        screen = _read_json(screen_path, "completed portfolio screen")
        form = _read_json(review_path, "portfolio review form")
        validate_screen(screen, root)
        validate_pending_review_form(form, screen)
        if (
            row["screen_content_sha256"] != screen["content_sha256"]
            or row["screen_file_sha256"] != _normalized_file_sha256(screen_path)
            or row["review_form_sha256"] != _normalized_file_sha256(review_path)
            or row["behavior_assessment"] != screen["channel_assessment"]["behavior"]
            or row["provider_statuses"]
            != {item["provider_id"]: item["status"] for item in screen["provider_evidence"]}
        ):
            raise PriorArtPortfolioError("completed portfolio file binding changed")


def _batch_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    batch = receipt["batch"]
    return {
        "completed_claims": len(batch["completed_claims"]),
        "cumulative_external_request_budget": batch["cumulative_external_request_budget"],
        "external_request_budget_consumed": batch["external_request_budget_consumed"],
        "newly_screened_claims": len(batch["newly_screened_source_ids"]),
        "remaining_claims": batch["remaining_claims"],
        "reused_valid_claims": len(batch["reused_valid_source_ids"]),
        "status": receipt["release_gate"]["status"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=Path.cwd())
    prepare.add_argument("--claim-directory", type=Path, default=Path(CLAIM_DIRECTORY))
    prepare.add_argument("--output", type=Path, default=Path(PREFLIGHT_PATH))
    validate = subparsers.add_parser("validate-preflight")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(PREFLIGHT_PATH))
    batch = subparsers.add_parser("screen-batch")
    batch.add_argument("--root", type=Path, default=Path.cwd())
    batch.add_argument("--preflight", type=Path, default=Path(PREFLIGHT_PATH))
    batch.add_argument("--screen-directory", type=Path, default=Path(SCREEN_DIRECTORY))
    batch.add_argument("--review-directory", type=Path, default=Path(REVIEW_DIRECTORY))
    batch.add_argument("--maximum-claims", type=int, default=4)
    batch.add_argument("--output", type=Path, required=True)
    rebind = subparsers.add_parser("rebind-screens")
    rebind.add_argument("--root", type=Path, default=Path.cwd())
    rebind.add_argument("--preflight", type=Path, default=Path(PREFLIGHT_PATH))
    rebind.add_argument("--screen-directory", type=Path, default=Path(SCREEN_DIRECTORY))
    rebind.add_argument("--review-directory", type=Path, default=Path(REVIEW_DIRECTORY))
    validate_batch = subparsers.add_parser("validate-batch")
    validate_batch.add_argument("--root", type=Path, default=Path.cwd())
    validate_batch.add_argument("--preflight", type=Path, default=Path(PREFLIGHT_PATH))
    validate_batch.add_argument("--receipt", type=Path, default=Path(COMPLETE_PATH))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        receipt = build_preflight(args.root, args.claim_directory)
        output = _under(args.root.resolve(), args.output, "portfolio preflight output")
        _write_json_atomic(output, receipt)
        summary = receipt["summary"]
    elif args.command == "validate-preflight":
        receipt_path = _under(args.root.resolve(), args.receipt, "portfolio preflight")
        receipt = _read_json(receipt_path, "portfolio preflight")
        validate_preflight(receipt, args.root)
        summary = receipt["summary"]
    elif args.command == "rebind-screens":
        root = args.root.resolve()
        preflight_path = _under(root, args.preflight, "portfolio preflight")
        preflight = _read_json(preflight_path, "portfolio preflight")
        validate_preflight(preflight, root)
        screen_directory = _under(root, args.screen_directory, "portfolio screen directory")
        review_directory = _under(root, args.review_directory, "portfolio review directory")
        rebound = []
        for row in preflight["claims"]:
            screen_path = screen_directory / f"{row['source_id']}.json"
            review_path = review_directory / f"{row['source_id']}.json"
            previous = _read_json(screen_path, "portfolio screen")
            screen = rebind_screen(root, root / row["claim_path"], previous)
            _write_json_atomic(screen_path, screen)
            _write_json_atomic(review_path, build_pending_review_form(screen))
            rebound.append(row["source_id"])
        summary = {"new_provider_calls": 0, "rebound_screens": rebound}
    elif args.command == "screen-batch":
        receipt = screen_portfolio(
            args.root,
            args.preflight,
            args.screen_directory,
            args.review_directory,
            maximum_claims=args.maximum_claims,
        )
        output = _under(args.root.resolve(), args.output, "portfolio batch output")
        _write_json_atomic(output, receipt)
        summary = _batch_summary(receipt)
    else:
        preflight_path = _under(args.root.resolve(), args.preflight, "portfolio preflight")
        receipt_path = _under(args.root.resolve(), args.receipt, "portfolio batch receipt")
        preflight = _read_json(preflight_path, "portfolio preflight")
        receipt = _read_json(receipt_path, "portfolio batch receipt")
        validate_preflight(preflight, args.root)
        validate_batch_receipt(receipt, preflight, args.root)
        summary = _batch_summary(receipt)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
