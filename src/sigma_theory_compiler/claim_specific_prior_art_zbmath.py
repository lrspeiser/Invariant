"""Add a sealed, math-specific zbMATH Open screen to the pending claim-review dossier.

This supplement never rewrites an original provider receipt and never clears novelty. It stores
only compact bibliographic metadata, not zbMATH editorial reviews or abstracts, and leaves every
claim open until a specifically named human compares behavior, construction, representation, and
proof mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .claim_specific_prior_art import (
    HTTPResponse,
    Transport,
    load_claim,
    urllib_transport,
    validate_screen,
)
from .claim_specific_prior_art_portfolio import (
    validate_batch_receipt,
    validate_pending_review_form,
    validate_preflight,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/claim_specific_prior_art_zbmath.json"
MODULE_PATH = "src/sigma_theory_compiler/claim_specific_prior_art_zbmath.py"
OUTPUT_PATH = "runs/math/claim-specific-prior-art/zbmath-25-claim-supplement.json"
CONFIG_SCHEMA = "invariant-claim-prior-art-zbmath-config-1.0"
RECEIPT_SCHEMA = "invariant-claim-prior-art-zbmath-supplement-1.0"
PROVIDER_ID = "zbmath_open"
PROVIDER_HOST = "api.zbmath.org"

_HEX = frozenset("0123456789abcdef")
_ORIGINS = {
    "known_rewrite",
    "cross_domain_synthesis",
    "proposed_new_construction",
    "uncertain",
}
_STATUS = {"COMPLETED", "COMPLETED_NO_RESULTS", "UNAVAILABLE_RECORDED"}


class ZbMathSupplementError(ValueError):
    """The additive zbMATH evidence or its fail-closed boundary is invalid."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ZbMathSupplementError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ZbMathSupplementError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _text(value: Any, label: str, *, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ZbMathSupplementError(f"{label} is empty or too long")
    return value.strip()


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / _text(relative, label, maximum=500)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ZbMathSupplementError(f"{label} escapes the repository") from error
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ZbMathSupplementError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise ZbMathSupplementError(f"{label} is not a JSON object")
    return value


def _utc(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ZbMathSupplementError("retrieval time is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ZbMathSupplementError("retrieval time lacks an offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def load_config(root: Path) -> dict[str, Any]:
    value = _read_json(root / CONFIG_PATH, "zbMATH supplement config")
    _strict(
        value,
        {"network", "provider", "release_policy", "schema_version", "source_set", "supplement_id"},
        "zbMATH supplement config",
    )
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ZbMathSupplementError("zbMATH config schema changed")
    _text(value["supplement_id"], "supplement ID", maximum=160)
    provider = value["provider"]
    _strict(provider, {"base_uri", "maximum_results", "provider_id"}, "zbMATH provider")
    parsed = urllib.parse.urlparse(provider["base_uri"])
    if (
        provider["provider_id"] != PROVIDER_ID
        or parsed.scheme != "https"
        or parsed.hostname != PROVIDER_HOST
        or parsed.path != "/v1/document/_search"
        or not isinstance(provider["maximum_results"], int)
        or not 1 <= provider["maximum_results"] <= 10
    ):
        raise ZbMathSupplementError("zbMATH provider authority or bounds changed")
    network = value["network"]
    _strict(
        network,
        {"maximum_response_bytes", "request_timeout_seconds", "user_agent"},
        "zbMATH network policy",
    )
    if (
        not isinstance(network["maximum_response_bytes"], int)
        or not 10_000 <= network["maximum_response_bytes"] <= 2_000_000
        or not isinstance(network["request_timeout_seconds"], int)
        or not 1 <= network["request_timeout_seconds"] <= 60
        or "InvariantPriorArtSupplement/" not in network["user_agent"]
    ):
        raise ZbMathSupplementError("zbMATH network bounds changed")
    sources = value["source_set"]
    _strict(
        sources,
        {
            "additional_claims",
            "portfolio_preflight_path",
            "portfolio_receipt_path",
            "required_claims",
        },
        "zbMATH source set",
    )
    if sources["required_claims"] != 25 or len(sources["additional_claims"]) != 1:
        raise ZbMathSupplementError("zbMATH claim coverage changed")
    for item in sources["additional_claims"]:
        _strict(
            item,
            {"claim_path", "review_form_path", "screen_path", "source_id"},
            "additional claim",
        )
        for key in ("claim_path", "review_form_path", "screen_path", "source_id"):
            _text(item[key], f"additional claim {key}", maximum=500)
    release = value["release_policy"]
    expected_release = {
        "automated_absence_establishes_novelty": False,
        "named_human_review_required": True,
        "supplement_establishes_literature_novelty": False,
    }
    if release != expected_release:
        raise ZbMathSupplementError("zbMATH release boundary weakened")
    return value


def _clean_phrase(value: str) -> str:
    phrase = re.sub(r"[\x00-\x1f\x7f\"\\|]", " ", value)
    phrase = " ".join(phrase.replace("_", " ").split())
    words = phrase.split()[:16]
    return " ".join(words)[:160]


def build_query(claim_document: Mapping[str, Any]) -> str:
    claim = claim_document["claim"]
    phrases = []
    for analogue in claim["known_analogues"][:3]:
        clean = _clean_phrase(analogue)
        if clean and "no named analogue" not in clean.lower():
            phrases.append(clean)
    if len(phrases) < 2:
        for fallback in (
            claim["representation"],
            *claim["proof_mechanisms"][:1],
            *claim["source_domains"][:1],
        ):
            clean = _clean_phrase(fallback)
            if clean and clean not in phrases:
                phrases.append(clean)
            if len(phrases) == 3:
                break
    query = " | ".join(f'"{phrase}"' for phrase in phrases[:3])
    if not query or len(query) > 500:
        raise ZbMathSupplementError("zbMATH query is empty or too long")
    return query


def _request_uri(config: Mapping[str, Any], query: str) -> str:
    parameters = urllib.parse.urlencode(
        {
            "results_per_page": str(config["provider"]["maximum_results"]),
            "search_string": query,
        }
    )
    uri = f"{config['provider']['base_uri']}?{parameters}"
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "https" or parsed.hostname != PROVIDER_HOST:
        raise ZbMathSupplementError("zbMATH request escaped provider authority")
    return uri


def _string_or_none(value: Any, *, maximum: int) -> str | None:
    return value.strip()[:maximum] if isinstance(value, str) and value.strip() else None


def _normalize_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ZbMathSupplementError("zbMATH result is not an object")
    raw_id = value.get("id")
    if not isinstance(raw_id, (int, str)) or not str(raw_id).strip():
        raise ZbMathSupplementError("zbMATH result lacks an ID")
    title_value = value.get("title", {})
    if not isinstance(title_value, Mapping):
        raise ZbMathSupplementError("zbMATH title structure changed")
    title = _text(title_value.get("title"), "zbMATH title", maximum=1000)
    uri = _text(value.get("zbmath_url"), "zbMATH result URI", maximum=500)
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "https" or parsed.hostname not in {"zbmath.org", "www.zbmath.org"}:
        raise ZbMathSupplementError("zbMATH result URI escaped its authority")
    contributor_value = value.get("contributors", {})
    authors = []
    if isinstance(contributor_value, Mapping):
        for author in contributor_value.get("authors", [])[:12]:
            if isinstance(author, Mapping):
                name = _string_or_none(author.get("name"), maximum=240)
                if name and name not in authors:
                    authors.append(name)
    classifications = []
    for item in value.get("msc", [])[:12]:
        if not isinstance(item, Mapping):
            continue
        code = _string_or_none(item.get("code"), maximum=20)
        text = _string_or_none(item.get("text"), maximum=240)
        if code:
            classifications.append({"code": code, "text": text})
    links = []
    for item in value.get("links", [])[:24]:
        if not isinstance(item, Mapping):
            continue
        url = _string_or_none(item.get("url"), maximum=500)
        if not url or urllib.parse.urlparse(url).scheme != "https":
            continue
        normalized = {
            "identifier": _string_or_none(item.get("identifier"), maximum=240),
            "type": _string_or_none(item.get("type"), maximum=80),
            "url": url,
        }
        if normalized not in links:
            links.append(normalized)
        if len(links) == 8:
            break
    return {
        "authors": authors,
        "identifier": f"ZBMATH:{str(raw_id).strip()}",
        "links": links,
        "msc": classifications,
        "title": title,
        "uri": uri,
        "year": _string_or_none(value.get("year"), maximum=20),
        "zbmath_number": _string_or_none(value.get("identifier"), maximum=80),
    }


def _parse_response(
    response: HTTPResponse, maximum_results: int
) -> tuple[str, list[dict[str, Any]], str | None]:
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        if response.status == 200:
            raise ZbMathSupplementError("zbMATH returned malformed success JSON") from error
        return "UNAVAILABLE_RECORDED", [], f"HTTP_{response.status}"
    if response.status == 404 and isinstance(payload, Mapping):
        status = payload.get("status", {})
        if isinstance(status, Mapping) and status.get("internal_code") == (
            "successful access. No results found."
        ):
            return "COMPLETED_NO_RESULTS", [], None
    if response.status != 200:
        return "UNAVAILABLE_RECORDED", [], f"HTTP_{response.status}"
    if not isinstance(payload, Mapping) or set(payload) != {"result", "status"}:
        raise ZbMathSupplementError("zbMATH success envelope changed")
    status = payload["status"]
    results = payload["result"]
    if (
        not isinstance(status, Mapping)
        or status.get("execution_bool") is not True
        or status.get("status_code") != 200
        or not isinstance(results, list)
    ):
        raise ZbMathSupplementError("zbMATH success status changed")
    normalized = []
    seen = set()
    for item in results[:maximum_results]:
        result = _normalize_result(item)
        if result["identifier"] not in seen:
            seen.add(result["identifier"])
            normalized.append(result)
    return "COMPLETED", normalized, None


def _compact_match(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "channel": value.get("channel"),
        "identifier": value.get("identifier"),
        "provider_id": value.get("provider_id"),
        "title": value.get("title"),
        "uri": value.get("uri"),
    }


def _collect_sources(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_set = config["source_set"]
    preflight_path = _under(root, source_set["portfolio_preflight_path"], "portfolio preflight")
    receipt_path = _under(root, source_set["portfolio_receipt_path"], "portfolio receipt")
    preflight = _read_json(preflight_path, "portfolio preflight")
    portfolio = _read_json(receipt_path, "portfolio receipt")
    validate_preflight(preflight, root)
    validate_batch_receipt(portfolio, preflight, root)
    if portfolio["batch"]["remaining_claims"] != 0:
        raise ZbMathSupplementError("portfolio is not fully screened")
    rows = []
    for original in portfolio["batch"]["completed_claims"]:
        screen_path = _under(root, original["screen_path"], "portfolio screen")
        review_path = _under(root, original["review_form_path"], "portfolio review form")
        screen = _read_json(screen_path, "portfolio screen")
        validate_screen(screen, root)
        form = _read_json(review_path, "portfolio review form")
        validate_pending_review_form(form, screen)
        claim_path = _under(root, screen["source_bindings"]["claim"]["path"], "portfolio claim")
        claim = load_claim(root, claim_path)
        rows.append(
            {
                "claim": claim,
                "claim_path": claim_path,
                "review_form": form,
                "review_path": review_path,
                "screen": screen,
                "screen_path": screen_path,
                "source_id": original["source_id"],
            }
        )
    for additional in source_set["additional_claims"]:
        claim_path = _under(root, additional["claim_path"], "additional claim")
        screen_path = _under(root, additional["screen_path"], "additional screen")
        review_path = _under(root, additional["review_form_path"], "additional review form")
        claim = load_claim(root, claim_path)
        screen = _read_json(screen_path, "additional screen")
        validate_screen(screen, root)
        form = _read_json(review_path, "additional review form")
        validate_pending_review_form(form, screen)
        rows.append(
            {
                "claim": claim,
                "claim_path": claim_path,
                "review_form": form,
                "review_path": review_path,
                "screen": screen,
                "screen_path": screen_path,
                "source_id": additional["source_id"],
            }
        )
    if len(rows) != source_set["required_claims"] or len({row["source_id"] for row in rows}) != len(
        rows
    ):
        raise ZbMathSupplementError("zbMATH source coverage changed")
    portfolio_binding = {
        "content_sha256": portfolio["content_sha256"],
        "file_sha256": _normalized_file_sha256(receipt_path),
        "path": receipt_path.relative_to(root).as_posix(),
        "preflight_content_sha256": preflight["content_sha256"],
        "preflight_path": preflight_path.relative_to(root).as_posix(),
    }
    return rows, portfolio_binding


def _claim_record(
    row: Mapping[str, Any], evidence: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    claim = row["claim"]
    screen = row["screen"]
    return {
        "claim_id": claim["claim_id"],
        "claim_path": row["claim_path"].relative_to(root).as_posix(),
        "claim_file_sha256": _normalized_file_sha256(row["claim_path"]),
        "construction": claim["claim"]["formula_or_construction"],
        "known_analogues": claim["claim"]["known_analogues"],
        "llm_origin_assessment": claim["claim"]["llm_origin_assessment"],
        "original_screen": {
            "automated_status": screen["automated_screen"]["status"],
            "content_sha256": screen["content_sha256"],
            "file_sha256": _normalized_file_sha256(row["screen_path"]),
            "nearest_matches": [_compact_match(item) for item in screen["nearest_matches"]],
            "path": row["screen_path"].relative_to(root).as_posix(),
            "provider_statuses": {
                item["provider_id"]: item["status"] for item in screen["provider_evidence"]
            },
        },
        "proof_mechanisms": claim["claim"]["proof_mechanisms"],
        "representation": claim["claim"]["representation"],
        "review_form": {
            "file_sha256": _normalized_file_sha256(row["review_path"]),
            "path": row["review_path"].relative_to(root).as_posix(),
            "status": "PENDING_NAMED_HUMAN_REVIEW",
        },
        "source_domains": claim["claim"]["source_domains"],
        "source_id": row["source_id"],
        "statement": claim["claim"]["statement"],
        "zbmath_evidence": dict(evidence),
    }


def build_supplement(
    root: Path,
    *,
    transport: Transport = urllib_transport,
    retrieved_utc: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    rows, portfolio_binding = _collect_sources(root, config)
    timestamp = _utc(retrieved_utc)
    claims = []
    for row in rows:
        query = build_query(row["claim"])
        uri = _request_uri(config, query)
        try:
            response = transport(
                uri,
                {"Accept": "application/json", "User-Agent": config["network"]["user_agent"]},
                config["network"]["request_timeout_seconds"],
                config["network"]["maximum_response_bytes"],
            )
            status, results, error = _parse_response(
                response, config["provider"]["maximum_results"]
            )
            evidence = {
                "error": error,
                "http_status": response.status,
                "query": query,
                "request_uri": uri,
                "response_sha256": hashlib.sha256(response.body).hexdigest(),
                "results": results,
                "retrieved_utc": timestamp,
                "status": status,
            }
        except ZbMathSupplementError:
            raise
        except OSError as error:  # network failure is evidence, not idea rejection
            evidence = {
                "error": f"TRANSPORT_{type(error).__name__}",
                "http_status": None,
                "query": query,
                "request_uri": uri,
                "response_sha256": None,
                "results": [],
                "retrieved_utc": timestamp,
                "status": "UNAVAILABLE_RECORDED",
            }
        claims.append(_claim_record(row, evidence, root))
    statuses = Counter(item["zbmath_evidence"]["status"] for item in claims)
    provider_statuses = Counter(
        status
        for item in claims
        for status in item["original_screen"]["provider_statuses"].values()
    )
    identifiers = {
        result["identifier"] for item in claims for result in item["zbmath_evidence"]["results"]
    }
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "supplement_id": config["supplement_id"],
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "sha256": _normalized_file_sha256(root / CONFIG_PATH)},
            "module": {"path": MODULE_PATH, "sha256": _normalized_file_sha256(root / MODULE_PATH)},
            "portfolio": portfolio_binding,
        },
        "claims_review_dossier": claims,
        "summary": {
            "claim_count": len(claims),
            "origin_counts": dict(
                sorted(Counter(item["llm_origin_assessment"] for item in claims).items())
            ),
            "original_provider_status_counts": dict(sorted(provider_statuses.items())),
            "unique_zbmath_results": len(identifiers),
            "zbmath_status_counts": {status: statuses.get(status, 0) for status in sorted(_STATUS)},
        },
        "human_review_protocol": {
            "questions": [
                "Does a result match only the observed behavior, or the actual construction?",
                "Is the proposal an equivalent notation or parameterization of a known formula?",
                "Are the components known but combined across domains in a materially different way?",
                "Does prior work use the same proof mechanism or representation dual?",
                "What exact citation and reasoning support the named reviewer's classification?",
            ],
            "reviewer": None,
            "status": "PENDING_NAMED_HUMAN_REVIEW",
        },
        "release_gate": {
            "all_named_human_reviews_complete": False,
            "novelty_language_authorized": False,
            "status": "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED",
        },
        "claims": {
            "automated_absence_establishes_novelty": False,
            "original_provider_history_was_overwritten": False,
            "supplement_establishes_literature_novelty": False,
            "zbmath_match_proves_known_rewrite_without_human_review": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_supplement(body, root)
    return body


def _validate_result(result: Mapping[str, Any]) -> None:
    _strict(
        result,
        {"authors", "identifier", "links", "msc", "title", "uri", "year", "zbmath_number"},
        "normalized zbMATH result",
    )
    _text(result["identifier"], "zbMATH identifier", maximum=180)
    _text(result["title"], "zbMATH title", maximum=1000)
    uri = urllib.parse.urlparse(_text(result["uri"], "zbMATH URI", maximum=500))
    if uri.scheme != "https" or uri.hostname not in {"zbmath.org", "www.zbmath.org"}:
        raise ZbMathSupplementError("normalized zbMATH URI escaped authority")
    if (
        not isinstance(result["authors"], list)
        or not isinstance(result["links"], list)
        or not isinstance(result["msc"], list)
    ):
        raise ZbMathSupplementError("normalized zbMATH metadata shape changed")


def validate_supplement(value: Mapping[str, Any], root: Path | None = None) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("content_sha256") != canonical_sha256(body)
        or value.get("schema_version") != RECEIPT_SCHEMA
    ):
        raise ZbMathSupplementError("zbMATH supplement content seal changed")
    if value.get("release_gate") != {
        "all_named_human_reviews_complete": False,
        "novelty_language_authorized": False,
        "status": "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED",
    } or value.get("claims") != {
        "automated_absence_establishes_novelty": False,
        "original_provider_history_was_overwritten": False,
        "supplement_establishes_literature_novelty": False,
        "zbmath_match_proves_known_rewrite_without_human_review": False,
    }:
        raise ZbMathSupplementError("zbMATH supplement opened a novelty gate")
    dossier = value.get("claims_review_dossier")
    if (
        not isinstance(dossier, list)
        or len(dossier) != 25
        or len({item.get("source_id") for item in dossier}) != 25
    ):
        raise ZbMathSupplementError("zbMATH dossier coverage changed")
    if (
        value.get("human_review_protocol", {}).get("status") != "PENDING_NAMED_HUMAN_REVIEW"
        or value["human_review_protocol"].get("reviewer") is not None
    ):
        raise ZbMathSupplementError("zbMATH dossier pretends human review is complete")
    status_counts = Counter()
    identifiers = set()
    origin_counts = Counter()
    original_provider_status_counts = Counter()
    for item in dossier:
        _strict(
            item,
            {
                "claim_file_sha256",
                "claim_id",
                "claim_path",
                "construction",
                "known_analogues",
                "llm_origin_assessment",
                "original_screen",
                "proof_mechanisms",
                "representation",
                "review_form",
                "source_domains",
                "source_id",
                "statement",
                "zbmath_evidence",
            },
            "zbMATH dossier claim",
        )
        if (
            item["llm_origin_assessment"] not in _ORIGINS
            or item.get("review_form", {}).get("status") != "PENDING_NAMED_HUMAN_REVIEW"
        ):
            raise ZbMathSupplementError("zbMATH claim origin or review status changed")
        evidence = item.get("zbmath_evidence", {})
        _strict(
            evidence,
            {
                "error",
                "http_status",
                "query",
                "request_uri",
                "response_sha256",
                "results",
                "retrieved_utc",
                "status",
            },
            "zbMATH evidence",
        )
        if evidence["status"] not in _STATUS:
            raise ZbMathSupplementError("zbMATH evidence status changed")
        parsed = urllib.parse.urlparse(evidence["request_uri"])
        if parsed.scheme != "https" or parsed.hostname != PROVIDER_HOST:
            raise ZbMathSupplementError("sealed zbMATH request escaped authority")
        if evidence["response_sha256"] is not None:
            _sha(evidence["response_sha256"], "zbMATH response hash")
        if _utc(evidence["retrieved_utc"]) != evidence["retrieved_utc"]:
            raise ZbMathSupplementError("zbMATH retrieval time is not normalized UTC")
        if evidence["status"] != "COMPLETED" and evidence["results"]:
            raise ZbMathSupplementError("non-completed zbMATH request has results")
        if evidence["status"] == "COMPLETED" and (
            evidence["http_status"] != 200 or evidence["error"] is not None
        ):
            raise ZbMathSupplementError("completed zbMATH evidence status is inconsistent")
        if evidence["status"] == "COMPLETED_NO_RESULTS" and (
            evidence["http_status"] != 404
            or evidence["error"] is not None
            or evidence["response_sha256"] is None
        ):
            raise ZbMathSupplementError("empty zbMATH evidence status is inconsistent")
        if evidence["status"] == "UNAVAILABLE_RECORDED" and not isinstance(evidence["error"], str):
            raise ZbMathSupplementError("unavailable zbMATH evidence lacks an error class")
        if len(evidence["results"]) > 5:
            raise ZbMathSupplementError("zbMATH result bound changed")
        for result in evidence["results"]:
            _validate_result(result)
            identifiers.add(result["identifier"])
        status_counts[evidence["status"]] += 1
        origin_counts[item["llm_origin_assessment"]] += 1
        original_provider_status_counts.update(
            item.get("original_screen", {}).get("provider_statuses", {}).values()
        )
    summary = value.get("summary", {})
    expected_summary = {
        "claim_count": 25,
        "origin_counts": dict(sorted(origin_counts.items())),
        "original_provider_status_counts": dict(sorted(original_provider_status_counts.items())),
        "unique_zbmath_results": len(identifiers),
        "zbmath_status_counts": {
            status: status_counts.get(status, 0) for status in sorted(_STATUS)
        },
    }
    if summary != expected_summary:
        raise ZbMathSupplementError("zbMATH summary accounting changed")
    if root is None:
        return
    root = root.resolve()
    config = load_config(root)
    rows, portfolio_binding = _collect_sources(root, config)
    bindings = value.get("source_bindings", {})
    if bindings != {
        "config": {"path": CONFIG_PATH, "sha256": _normalized_file_sha256(root / CONFIG_PATH)},
        "module": {"path": MODULE_PATH, "sha256": _normalized_file_sha256(root / MODULE_PATH)},
        "portfolio": portfolio_binding,
    }:
        raise ZbMathSupplementError("zbMATH source bindings changed")
    if [item["source_id"] for item in dossier] != [row["source_id"] for row in rows]:
        raise ZbMathSupplementError("zbMATH dossier source order changed")
    for item, row in zip(dossier, rows, strict=True):
        if item["claim_id"] != row["claim"]["claim_id"] or item["zbmath_evidence"][
            "query"
        ] != build_query(row["claim"]):
            raise ZbMathSupplementError("zbMATH claim/query binding changed")
        expected_uri = _request_uri(config, item["zbmath_evidence"]["query"])
        if item["zbmath_evidence"]["request_uri"] != expected_uri:
            raise ZbMathSupplementError("zbMATH request URI binding changed")
        expected = _claim_record(row, item["zbmath_evidence"], root)
        if item != expected:
            raise ZbMathSupplementError("zbMATH dossier file or claim binding changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    if args.command == "build":
        receipt = build_supplement(args.root)
        output = _under(args.root.resolve(), args.output.as_posix(), "zbMATH output")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {"content_sha256": receipt["content_sha256"], **receipt["summary"]}
    else:
        receipt_path = _under(args.root.resolve(), args.receipt.as_posix(), "zbMATH receipt")
        receipt = _read_json(receipt_path, "zbMATH receipt")
        validate_supplement(receipt, args.root)
        summary = {"content_sha256": receipt["content_sha256"], "status": "VALID"}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
