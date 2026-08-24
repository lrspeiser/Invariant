"""Claim-specific, multi-provider prior-art search with a mandatory named-human gate.

The automated screen separates behavioral matches from formula/representation and proof-mechanism
matches.  It seals exact queries, provider response hashes, normalized identifiers, and failures.
Automated absence is never a novelty decision.  A separate named human attestation is required even
when every configured provider responds successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/claim_specific_prior_art.json"
CONFIG_SCHEMA = "invariant-claim-specific-prior-art-config-1.0"
CLAIM_SCHEMA = "invariant-claim-prior-art-input-1.0"
DESCENDANT_CLAIM_SCHEMA = "invariant-claim-prior-art-input-1.1"
SCREEN_SCHEMA = "invariant-claim-prior-art-screen-1.0"
REVIEW_SCHEMA = "invariant-claim-prior-art-human-review-1.0"
ADJUDICATED_SCHEMA = "invariant-claim-prior-art-adjudicated-1.0"

_HEX = frozenset("0123456789abcdef")
_PROVIDER_HOSTS = {
    "arxiv": "export.arxiv.org",
    "crossref": "api.crossref.org",
    "oeis": "oeis.org",
    "semantic_scholar": "api.semanticscholar.org",
}
_TEXT_SUFFIXES = {".json", ".lean", ".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
_REPOSITORY_ROOTS = ("README.md", "configs", "docs", "formal", "src")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


class PriorArtError(ValueError):
    """A query, response, binding, or human-review gate failed closed."""


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[str, Mapping[str, str], int, int], HTTPResponse]


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PriorArtError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PriorArtError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,159}", value) is None
    ):
        raise PriorArtError(f"{label} is not a portable identifier")
    return value


def _text(value: Any, label: str, *, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise PriorArtError(f"{label} is empty or too long")
    return value.strip()


def _text_array(value: Any, label: str, *, minimum: int = 1) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise PriorArtError(f"{label} must be a nonempty array")
    result = [_text(item, label, maximum=500) for item in value]
    if len(set(result)) != len(result):
        raise PriorArtError(f"{label} contains duplicates")
    return result


def _utc(value: Any, label: str) -> str:
    value = _text(value, label, maximum=40)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PriorArtError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PriorArtError(f"{label} lacks a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def urllib_transport(
    uri: str, headers: Mapping[str, str], timeout_seconds: int, maximum_bytes: int
) -> HTTPResponse:
    request = urllib.request.Request(uri, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(maximum_bytes + 1)
            if len(body) > maximum_bytes:
                raise PriorArtError("prior-art provider response exceeds byte limit")
            return HTTPResponse(
                status=int(response.status),
                headers={key.lower(): value for key, value in response.headers.items()},
                body=body,
            )
    except urllib.error.HTTPError as error:
        body = error.read(maximum_bytes + 1)
        if len(body) > maximum_bytes:
            body = body[:maximum_bytes]
        return HTTPResponse(
            status=int(error.code),
            headers={key.lower(): value for key, value in error.headers.items()},
            body=body,
        )


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    _strict(
        value,
        {"network", "providers", "query_policy", "release_policy", "schema_version", "screen_id"},
        "prior-art config",
    )
    if value["schema_version"] != CONFIG_SCHEMA or value["screen_id"] != (
        "invariant.claim-specific-prior-art"
    ):
        raise PriorArtError("prior-art config identity changed")
    network = value["network"]
    _strict(
        network,
        {"maximum_response_bytes", "request_timeout_seconds", "user_agent"},
        "prior-art network policy",
    )
    if (
        not isinstance(network["maximum_response_bytes"], int)
        or not 10_000 <= network["maximum_response_bytes"] <= 5_000_000
        or not isinstance(network["request_timeout_seconds"], int)
        or not 1 <= network["request_timeout_seconds"] <= 60
        or "InvariantPriorArt/" not in network["user_agent"]
    ):
        raise PriorArtError("prior-art network bounds changed")
    providers = value["providers"]
    if not isinstance(providers, list) or len(providers) < 4:
        raise PriorArtError("prior-art provider coverage is too small")
    seen = set()
    for provider in providers:
        _strict(
            provider,
            {"authority", "base_uri", "maximum_results", "provider_id", "query_channel"},
            "prior-art provider",
        )
        provider_id = _identifier(provider["provider_id"], "provider ID")
        parsed = urllib.parse.urlparse(provider["base_uri"])
        if (
            provider_id in seen
            or provider_id not in _PROVIDER_HOSTS
            or parsed.scheme != "https"
            or parsed.hostname != _PROVIDER_HOSTS[provider_id]
            or not isinstance(provider["maximum_results"], int)
            or not 1 <= provider["maximum_results"] <= 20
        ):
            raise PriorArtError("prior-art provider identity or bounds changed")
        seen.add(provider_id)
    policy = value["query_policy"]
    _strict(policy, {"minimum_distinct_channels", "required_channels"}, "query policy")
    required = _text_array(policy["required_channels"], "required query channels", minimum=4)
    if policy["minimum_distinct_channels"] < 4 or set(required) != {
        "behavior",
        "equivalent_notation",
        "literal_construction",
        "proof_mechanism",
        "representation_dual",
    }:
        raise PriorArtError("claim-specific query channels changed")
    release = value["release_policy"]
    _strict(
        release,
        {
            "human_review_required",
            "minimum_successful_external_providers",
            "no_match_means_novel",
            "reviewer_name_placeholders",
        },
        "prior-art release policy",
    )
    if (
        release["human_review_required"] is not True
        or release["no_match_means_novel"] is not False
        or release["minimum_successful_external_providers"] < 3
    ):
        raise PriorArtError("prior-art release boundary weakened")
    return value


def load_claim(root: Path, claim_path: Path) -> dict[str, Any]:
    value = json.loads(claim_path.read_text(encoding="utf-8"))
    _strict(value, {"claim", "claim_id", "schema_version", "source_binding"}, "prior-art claim")
    schema_version = value["schema_version"]
    if schema_version not in {CLAIM_SCHEMA, DESCENDANT_CLAIM_SCHEMA}:
        raise PriorArtError("prior-art claim schema changed")
    _identifier(value["claim_id"], "claim ID")
    binding = value["source_binding"]
    if schema_version == CLAIM_SCHEMA:
        _strict(
            binding,
            {"idea_content_sha256", "lineage_id", "path", "receipt_content_sha256"},
            "claim source binding",
        )
    else:
        _strict(
            binding,
            {
                "path",
                "receipt_content_sha256",
                "source_content_sha256",
                "source_id",
                "source_kind",
            },
            "claim source binding",
        )
        if binding["source_kind"] != "retained_descendant":
            raise PriorArtError("claim source kind changed")
    _sha(binding["receipt_content_sha256"], "bound receipt content hash")
    if schema_version == CLAIM_SCHEMA:
        _sha(binding["idea_content_sha256"], "bound idea content hash")
        _identifier(binding["lineage_id"], "lineage ID")
    else:
        _sha(binding["source_content_sha256"], "bound descendant content hash")
        _identifier(binding["source_id"], "descendant ID")
    bound_path = (root / _text(binding["path"], "bound receipt path", maximum=500)).resolve()
    try:
        bound_path.relative_to(root.resolve())
    except ValueError as error:
        raise PriorArtError("claim source binding escapes the repository") from error
    receipt = json.loads(bound_path.read_text(encoding="utf-8"))
    receipt_body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != binding["receipt_content_sha256"] or receipt.get(
        "content_sha256"
    ) != canonical_sha256(receipt_body):
        raise PriorArtError("claim source receipt hash changed")
    if schema_version == CLAIM_SCHEMA:
        ideas = receipt.get("idea_lineage_archive", {}).get("ideas", [])
        matching = [idea for idea in ideas if idea.get("lineage_id") == binding["lineage_id"]]
        proposal_keys = (
            "benchmark_id",
            "expression",
            "family",
            "falsifiers",
            "hypothesis_id",
            "invariants",
            "known_analogues",
            "llm_origin_assessment",
            "proof_plan",
            "rationale",
            "representation",
            "source_idea_domains",
            "synthesis_note",
        )
        if (
            len(matching) != 1
            or matching[0].get("idea_content_sha256") != binding["idea_content_sha256"]
            or canonical_sha256({key: matching[0].get(key) for key in proposal_keys})
            != binding["idea_content_sha256"]
        ):
            raise PriorArtError("claim does not bind exactly one retained LLM idea")
    else:
        descendants = receipt.get("descendants", [])
        matching = [
            descendant
            for descendant in descendants
            if descendant.get("descendant_id") == binding["source_id"]
        ]
        if (
            len(matching) != 1
            or matching[0].get("retention_status") != "RETAINED_ACTIVE"
            or canonical_sha256(matching[0]) != binding["source_content_sha256"]
        ):
            raise PriorArtError("claim does not bind exactly one retained descendant")
    claim = value["claim"]
    claim_keys = {
        "behavior_terms",
        "formula_or_construction",
        "known_analogues",
        "llm_origin_assessment",
        "proof_mechanisms",
        "representation",
        "source_domains",
        "statement",
    }
    if schema_version == DESCENDANT_CLAIM_SCHEMA:
        claim_keys.add("behavior_status")
    _strict(claim, claim_keys, "claim content")
    _text(claim["statement"], "claim statement")
    _text(claim["formula_or_construction"], "claim construction")
    _identifier(claim["representation"], "claim representation")
    if schema_version == CLAIM_SCHEMA:
        _text_array(claim["behavior_terms"], "claim behavior terms", minimum=6)
    elif claim["behavior_status"] == "EXECUTABLE_PREDICTIONS":
        if not isinstance(claim["behavior_terms"], list) or len(claim["behavior_terms"]) < 6:
            raise PriorArtError("claim behavior terms must contain at least six predictions")
        for item in claim["behavior_terms"]:
            _text(item, "claim behavior term", maximum=500)
    elif claim["behavior_status"] == "NO_EXECUTABLE_BEHAVIOR":
        if claim["behavior_terms"] != []:
            raise PriorArtError("non-executable claim invents behavior terms")
    else:
        raise PriorArtError("claim behavior status changed")
    _text_array(claim["known_analogues"], "claim known analogues")
    _text_array(claim["proof_mechanisms"], "claim proof mechanisms")
    _text_array(claim["source_domains"], "claim source domains")
    if claim["llm_origin_assessment"] not in {
        "known_rewrite",
        "cross_domain_synthesis",
        "proposed_new_construction",
        "uncertain",
    }:
        raise PriorArtError("claim LLM origin assessment changed")
    return value


def build_queries(claim_document: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    claim = claim_document["claim"]
    construction = claim["formula_or_construction"]
    representation = claim["representation"].replace("_", " ")
    domains = " ".join(claim["source_domains"])
    analogues = " ".join(claim["known_analogues"])
    proofs = " ".join(claim["proof_mechanisms"])
    duals = {
        "finite sum": "finite product recurrence generating function coefficient extraction",
        "finite product": "finite sum recurrence logarithmic transform",
        "generating function": "coefficient recurrence closed form transform",
        "piecewise recurrence": "closed form generating function parity recurrence",
        "linear recurrence": "closed form generating function characteristic polynomial",
        "tensor identity": "index symmetry invariant contraction differential identity",
        "variational functional": "Euler Lagrange functional stationary action duality",
    }
    behavior_query = (
        ",".join(claim["behavior_terms"][:18])
        if claim["behavior_terms"]
        else f"no executable behavior {construction} {domains}"[:500]
    )
    queries = (
        {
            "channel": "behavior",
            "query": behavior_query,
        },
        {
            "channel": "literal_construction",
            "query": f"{construction} {domains}"[:500],
        },
        {
            "channel": "equivalent_notation",
            "query": f"{analogues} {representation} {domains}"[:500],
        },
        {
            "channel": "proof_mechanism",
            "query": f"{proofs} {representation} {domains}"[:500],
        },
        {
            "channel": "representation_dual",
            "query": f"{duals.get(representation, 'equivalent representation transform dual')} {construction} {domains}"[
                :500
            ],
        },
    )
    if len({item["query"] for item in queries}) != len(queries):
        raise PriorArtError("claim-specific queries collapsed to duplicates")
    return queries


def _provider_uri(provider: Mapping[str, Any], query: str) -> str:
    provider_id = provider["provider_id"]
    limit = provider["maximum_results"]
    if provider_id == "oeis":
        params = {"fmt": "json", "q": query}
    elif provider_id == "crossref":
        params = {
            "query.bibliographic": query,
            "rows": str(limit),
            "select": "DOI,title,author,published",
        }
    elif provider_id == "arxiv":
        safe = " ".join(_WORD.findall(query)[:24])
        params = {"max_results": str(limit), "search_query": f'all:"{safe}"', "start": "0"}
    elif provider_id == "semantic_scholar":
        params = {
            "fields": "title,year,authors,externalIds,url",
            "limit": str(limit),
            "query": " ".join(_WORD.findall(query)[:40]),
        }
    else:  # guarded by config validation
        raise PriorArtError("unsupported prior-art provider")
    return f"{provider['base_uri']}?{urllib.parse.urlencode(params)}"


def _clean_title(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:500]


def _parse_oeis(body: bytes, limit: int, query: str) -> list[dict[str, Any]]:
    value = json.loads(body.decode("utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = value.get("results", [])
    else:
        rows = []
    expected_prefix = query.replace(" ", "").split(",")
    matches = []
    for rank, item in enumerate(rows[:limit]):
        number = int(item["number"])
        terms = str(item.get("data", "")).replace(" ", "").split(",")
        prefix_length = 0
        for left, right in zip(expected_prefix, terms, strict=False):
            if left != right:
                break
            prefix_length += 1
        matches.append(
            {
                "channel": "behavior",
                "identifier": f"OEIS:A{number:06d}",
                "matched_prefix_terms": prefix_length,
                "rank": rank,
                "title": _clean_title(item.get("name")),
                "uri": f"https://oeis.org/A{number:06d}",
            }
        )
    return matches


def _parse_crossref(body: bytes, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(body.decode("utf-8")).get("message", {}).get("items", [])
    matches = []
    for rank, item in enumerate(rows[:limit]):
        doi = str(item.get("DOI", "")).strip()
        if not doi:
            continue
        matches.append(
            {
                "channel": "formula_and_proof",
                "identifier": f"DOI:{doi}",
                "matched_prefix_terms": 0,
                "rank": rank,
                "title": _clean_title(item.get("title")),
                "uri": f"https://doi.org/{urllib.parse.quote(doi, safe='/()')}",
            }
        )
    return matches


def _parse_arxiv(body: bytes, limit: int) -> list[dict[str, Any]]:
    root = ET.fromstring(body)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    matches = []
    for rank, entry in enumerate(root.findall("atom:entry", namespace)[:limit]):
        uri = _clean_title(entry.findtext("atom:id", default="", namespaces=namespace))
        identifier = uri.rsplit("/", 1)[-1]
        matches.append(
            {
                "channel": "formula_and_proof",
                "identifier": f"ARXIV:{identifier}",
                "matched_prefix_terms": 0,
                "rank": rank,
                "title": _clean_title(
                    entry.findtext("atom:title", default="", namespaces=namespace)
                ),
                "uri": uri,
            }
        )
    return matches


def _parse_semantic_scholar(body: bytes, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(body.decode("utf-8")).get("data", [])
    matches = []
    for rank, item in enumerate(rows[:limit]):
        paper_id = str(item.get("paperId", "")).strip()
        if not paper_id:
            continue
        matches.append(
            {
                "channel": "formula_and_proof",
                "identifier": f"S2:{paper_id}",
                "matched_prefix_terms": 0,
                "rank": rank,
                "title": _clean_title(item.get("title")),
                "uri": _clean_title(item.get("url"))
                or f"https://www.semanticscholar.org/paper/{paper_id}",
            }
        )
    return matches


def _parse_provider(provider_id: str, body: bytes, limit: int, query: str) -> list[dict[str, Any]]:
    if provider_id == "oeis":
        return _parse_oeis(body, limit, query)
    if provider_id == "crossref":
        return _parse_crossref(body, limit)
    if provider_id == "arxiv":
        return _parse_arxiv(body, limit)
    if provider_id == "semantic_scholar":
        return _parse_semantic_scholar(body, limit)
    raise PriorArtError("unsupported provider parser")


def _repository_matches(root: Path, queries: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    tokens = {
        word.lower()
        for item in queries
        for word in _WORD.findall(item["query"])
        if len(word) >= 5 and word.lower() not in {"formula", "sequence", "where"}
    }
    candidates: list[tuple[int, str, str, str]] = []
    for relative in _REPOSITORY_ROOTS:
        base = root / relative
        paths = [base] if base.is_file() else base.rglob("*") if base.exists() else []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8").lower()
            except (OSError, UnicodeDecodeError):
                continue
            hits = sorted(token for token in tokens if token in text)
            if len(hits) < 2:
                continue
            rel = path.relative_to(root).as_posix()
            candidates.append((len(hits), rel, _normalized_file_sha256(path), ", ".join(hits[:12])))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "channel": "repository",
            "identifier": f"REPO:{path}",
            "matched_prefix_terms": 0,
            "rank": rank,
            "title": f"repository token overlap: {hits}",
            "uri": path,
            "file_sha256": file_sha,
        }
        for rank, (_, path, file_sha, hits) in enumerate(candidates[:5])
    ]


def run_screen(
    root: Path,
    claim_path: Path,
    *,
    transport: Transport = urllib_transport,
    retrieved_utc: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    claim_document = load_claim(root, claim_path)
    queries = build_queries(claim_document)
    retrieved_utc = _utc(
        retrieved_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "screen retrieval time",
    )
    network = config["network"]
    headers = {
        "Accept": "application/json, application/atom+xml, text/plain",
        "User-Agent": network["user_agent"],
    }
    evidence = []
    all_matches = []
    query_by_channel = {item["channel"]: item["query"] for item in queries}
    textual_query = " ".join(
        query_by_channel[channel]
        for channel in (
            "equivalent_notation",
            "proof_mechanism",
            "representation_dual",
        )
    )[:500]
    for provider in config["providers"]:
        provider_id = provider["provider_id"]
        query = query_by_channel["behavior"] if provider_id == "oeis" else textual_query
        uri = _provider_uri(provider, query)
        if provider_id == "oeis" and not claim_document["claim"]["behavior_terms"]:
            evidence.append(
                {
                    "content_type": "",
                    "error": "NO_EXECUTABLE_BEHAVIOR",
                    "matches": [],
                    "provider_id": provider_id,
                    "query": query,
                    "request_uri": uri,
                    "response_sha256": None,
                    "status": "NOT_APPLICABLE_RECORDED",
                }
            )
            continue
        try:
            response = transport(
                uri,
                headers,
                network["request_timeout_seconds"],
                network["maximum_response_bytes"],
            )
            response_sha = hashlib.sha256(response.body).hexdigest()
            content_type = response.headers.get("content-type", "")[:200]
            if response.status != 200:
                evidence.append(
                    {
                        "content_type": content_type,
                        "error": f"HTTP_{response.status}",
                        "matches": [],
                        "provider_id": provider_id,
                        "query": query,
                        "request_uri": uri,
                        "response_sha256": response_sha,
                        "status": "UNAVAILABLE_RECORDED",
                    }
                )
                continue
            matches = _parse_provider(
                provider_id, response.body, provider["maximum_results"], query
            )
            evidence.append(
                {
                    "content_type": content_type,
                    "error": None,
                    "matches": matches,
                    "provider_id": provider_id,
                    "query": query,
                    "request_uri": uri,
                    "response_sha256": response_sha,
                    "status": "COMPLETED",
                }
            )
            all_matches.extend({**item, "provider_id": provider_id} for item in matches)
        except (ET.ParseError, OSError, PriorArtError, UnicodeDecodeError, ValueError) as error:
            evidence.append(
                {
                    "content_type": "",
                    "error": f"{type(error).__name__}:{str(error)[:240]}",
                    "matches": [],
                    "provider_id": provider_id,
                    "query": query,
                    "request_uri": uri,
                    "response_sha256": None,
                    "status": "UNAVAILABLE_RECORDED",
                }
            )
    repository = _repository_matches(root, queries)
    all_matches.extend({**item, "provider_id": "repository"} for item in repository)
    exact_behavior = [
        item
        for item in all_matches
        if claim_document["claim"]["behavior_terms"]
        and item["provider_id"] == "oeis"
        and item.get("matched_prefix_terms", 0) >= len(claim_document["claim"]["behavior_terms"])
    ]
    successful = sum(item["status"] == "COMPLETED" for item in evidence)
    automated_status = (
        "COMPLETED_NOT_NOVELTY_CLEARED"
        if successful >= config["release_policy"]["minimum_successful_external_providers"]
        else "INCOMPLETE_PROVIDER_COVERAGE"
    )
    body: dict[str, Any] = {
        "schema_version": SCREEN_SCHEMA,
        "screen_id": config["screen_id"],
        "claim_id": claim_document["claim_id"],
        "source_bindings": {
            "claim": {
                "path": claim_path.resolve().relative_to(root).as_posix(),
                "sha256": _normalized_file_sha256(claim_path),
            },
            "config": {"path": CONFIG_PATH, "sha256": _normalized_file_sha256(root / CONFIG_PATH)},
            "idea": dict(claim_document["source_binding"]),
        },
        "retrieved_utc": retrieved_utc,
        "queries": list(queries),
        "provider_evidence": evidence,
        "repository_matches": repository,
        "nearest_matches": sorted(
            all_matches,
            key=lambda item: (
                -int(item.get("matched_prefix_terms", 0)),
                item["rank"],
                item["provider_id"],
                item["identifier"],
            ),
        )[:12],
        "channel_assessment": {
            "behavior": (
                "NOT_APPLICABLE_NO_EXECUTABLE_BEHAVIOR"
                if not claim_document["claim"]["behavior_terms"]
                else (
                    "KNOWN_EXTERNAL_BEHAVIOR_MATCH"
                    if exact_behavior
                    else "NO_EXACT_MATCH_IN_QUERIED_RESULTS"
                )
            ),
            "formula_or_construction": "REQUIRES_NAMED_HUMAN_REVIEW",
            "proof_mechanism": "REQUIRES_NAMED_HUMAN_REVIEW",
        },
        "llm_origin_calibration": {
            "model_assessment": claim_document["claim"]["llm_origin_assessment"],
            "behavior_match_found": bool(exact_behavior),
            "calibration_note": (
                "model uncertainty contained an externally indexed known behavior"
                if exact_behavior
                else (
                    "behavior search was inapplicable because the retained branch is not executable"
                    if not claim_document["claim"]["behavior_terms"]
                    else "automated screen did not resolve the model origin label"
                )
            ),
        },
        "automated_screen": {
            "successful_external_providers": successful,
            "total_external_providers": len(evidence),
            "status": automated_status,
        },
        "human_review": {
            "nearest_matches_must_be_reviewed": True,
            "reviewer": None,
            "status": "PENDING_NAMED_HUMAN_REVIEW",
        },
        "release_gate": {
            "claim_prior_art_complete": False,
            "novelty_language_authorized": False,
            "status": "BLOCKED_NAMED_HUMAN_REVIEW_REQUIRED",
        },
        "claims": {
            "automated_absence_establishes_novelty": False,
            "behavioral_match_resolves_formula_novelty": False,
            "llm_origin_label_is_prior_art_authority": False,
            "literature_novelty_established": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_screen(body, root)
    return body


def validate_screen(value: Mapping[str, Any], root: Path | None = None) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise PriorArtError("prior-art screen content seal changed")
    if value.get("schema_version") != SCREEN_SCHEMA:
        raise PriorArtError("prior-art screen schema changed")
    if len(value.get("queries", [])) < 5 or len(value.get("provider_evidence", [])) < 4:
        raise PriorArtError("prior-art query/provider coverage changed")
    if value.get("human_review", {}).get("status") != "PENDING_NAMED_HUMAN_REVIEW":
        raise PriorArtError("unadjudicated screen pretends human review is complete")
    release = value.get("release_gate", {})
    claims = value.get("claims", {})
    if (
        release.get("claim_prior_art_complete") is not False
        or release.get("novelty_language_authorized") is not False
        or any(claims.get(key) is not False for key in claims)
    ):
        raise PriorArtError("automated prior-art screen opened a novelty gate")
    for provider in value["provider_evidence"]:
        if provider.get("status") not in {
            "COMPLETED",
            "NOT_APPLICABLE_RECORDED",
            "UNAVAILABLE_RECORDED",
        }:
            raise PriorArtError("provider evidence status changed")
        parsed = urllib.parse.urlparse(provider["request_uri"])
        if parsed.scheme != "https" or parsed.hostname != _PROVIDER_HOSTS[provider["provider_id"]]:
            raise PriorArtError("provider evidence URI escaped its authority")
        if provider["status"] == "COMPLETED":
            _sha(provider["response_sha256"], "provider response hash")
        if provider["status"] == "NOT_APPLICABLE_RECORDED" and (
            provider["provider_id"] != "oeis"
            or provider.get("error") != "NO_EXECUTABLE_BEHAVIOR"
            or provider.get("response_sha256") is not None
        ):
            raise PriorArtError("inapplicable behavior evidence changed")
    if root is not None:
        root = root.resolve()
        for key in ("claim", "config"):
            binding = value["source_bindings"][key]
            path = (root / binding["path"]).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise PriorArtError("prior-art source binding escapes root") from error
            if binding["sha256"] != _normalized_file_sha256(path):
                raise PriorArtError("prior-art source binding changed")
        claim_document = load_claim(root, root / value["source_bindings"]["claim"]["path"])
        if (
            claim_document["claim_id"] != value["claim_id"]
            or dict(claim_document["source_binding"]) != value["source_bindings"]["idea"]
        ):
            raise PriorArtError("prior-art claim identity changed")


def rebind_screen(
    root: Path,
    claim_path: Path,
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebind unchanged queries to a new upstream receipt without new provider calls."""

    root = root.resolve()
    validate_screen(previous)
    claim_path = claim_path.resolve() if claim_path.is_absolute() else (root / claim_path).resolve()
    try:
        claim_relative = claim_path.relative_to(root).as_posix()
    except ValueError as error:
        raise PriorArtError("rebound claim path escapes root") from error
    claim_document = load_claim(root, claim_path)
    queries = list(build_queries(claim_document))
    previous_idea = dict(previous.get("source_bindings", {}).get("idea", {}))
    current_idea = dict(claim_document["source_binding"])
    previous_semantic_binding = {
        key: value for key, value in previous_idea.items() if key != "receipt_content_sha256"
    }
    current_semantic_binding = {
        key: value for key, value in current_idea.items() if key != "receipt_content_sha256"
    }
    if (
        claim_document["claim_id"] != previous.get("claim_id")
        or queries != previous.get("queries")
        or previous_semantic_binding != current_semantic_binding
        or claim_document["claim"]["llm_origin_assessment"]
        != previous.get("llm_origin_calibration", {}).get("model_assessment")
    ):
        raise PriorArtError("prior-art rebind changed the semantic claim or query set")

    provider_evidence = list(previous["provider_evidence"])
    external_matches = [
        {**item, "provider_id": evidence["provider_id"]}
        for evidence in provider_evidence
        for item in evidence["matches"]
    ]
    repository = _repository_matches(root, queries)
    all_matches = [
        *external_matches,
        *({**item, "provider_id": "repository"} for item in repository),
    ]
    behavior_terms = claim_document["claim"]["behavior_terms"]
    exact_behavior = [
        item
        for item in all_matches
        if behavior_terms
        and item["provider_id"] == "oeis"
        and item.get("matched_prefix_terms", 0) >= len(behavior_terms)
    ]
    successful = sum(item["status"] == "COMPLETED" for item in provider_evidence)
    body = {key: value for key, value in previous.items() if key != "content_sha256"}
    body["source_bindings"] = {
        "claim": {"path": claim_relative, "sha256": _normalized_file_sha256(claim_path)},
        "config": {
            "path": CONFIG_PATH,
            "sha256": _normalized_file_sha256(root / CONFIG_PATH),
        },
        "idea": current_idea,
    }
    body["repository_matches"] = repository
    body["nearest_matches"] = sorted(
        all_matches,
        key=lambda item: (
            -int(item.get("matched_prefix_terms", 0)),
            item["rank"],
            item["provider_id"],
            item["identifier"],
        ),
    )[:12]
    body["channel_assessment"] = {
        "behavior": (
            "NOT_APPLICABLE_NO_EXECUTABLE_BEHAVIOR"
            if not behavior_terms
            else (
                "KNOWN_EXTERNAL_BEHAVIOR_MATCH"
                if exact_behavior
                else "NO_EXACT_MATCH_IN_QUERIED_RESULTS"
            )
        ),
        "formula_or_construction": "REQUIRES_NAMED_HUMAN_REVIEW",
        "proof_mechanism": "REQUIRES_NAMED_HUMAN_REVIEW",
    }
    body["llm_origin_calibration"] = {
        "model_assessment": claim_document["claim"]["llm_origin_assessment"],
        "behavior_match_found": bool(exact_behavior),
        "calibration_note": (
            "model uncertainty contained an externally indexed known behavior"
            if exact_behavior
            else (
                "behavior search was inapplicable because the retained branch is not executable"
                if not behavior_terms
                else "automated screen did not resolve the model origin label"
            )
        ),
    }
    body["automated_screen"] = {
        "successful_external_providers": successful,
        "total_external_providers": len(provider_evidence),
        "status": previous["automated_screen"]["status"],
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_screen(body, root)
    return body


def adjudicate_screen(
    screen: Mapping[str, Any], review: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    validate_screen(screen)
    _strict(
        review,
        {
            "claim_id",
            "nearest_match_decisions",
            "notes",
            "overall_classification",
            "reviewed_utc",
            "reviewer",
            "schema_version",
            "screen_content_sha256",
        },
        "human prior-art review",
    )
    if review["schema_version"] != REVIEW_SCHEMA:
        raise PriorArtError("human review schema changed")
    if (
        review["claim_id"] != screen["claim_id"]
        or review["screen_content_sha256"] != screen["content_sha256"]
    ):
        raise PriorArtError("human review is not bound to this screen")
    reviewer = review["reviewer"]
    _strict(reviewer, {"affiliation_or_independent", "name"}, "human reviewer")
    name = _text(reviewer["name"], "human reviewer name", maximum=160)
    placeholders = {item.lower() for item in config["release_policy"]["reviewer_name_placeholders"]}
    if name.lower() in placeholders or len(name.split()) < 2:
        raise PriorArtError("human prior-art reviewer is not specifically named")
    _text(reviewer["affiliation_or_independent"], "reviewer affiliation", maximum=240)
    _utc(review["reviewed_utc"], "human review time")
    decisions = review["nearest_match_decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise PriorArtError("human review contains no nearest-match decisions")
    available = {item["identifier"] for item in screen["nearest_matches"]}
    decided = set()
    for decision in decisions:
        _strict(decision, {"classification", "identifier", "notes"}, "nearest-match decision")
        identifier = _text(decision["identifier"], "reviewed match ID", maximum=500)
        if identifier not in available or identifier in decided:
            raise PriorArtError("human review references an absent or duplicate match")
        decided.add(identifier)
        if decision["classification"] not in {
            "behavior_only",
            "formula_or_construction_match",
            "proof_mechanism_match",
            "unrelated",
        }:
            raise PriorArtError("human nearest-match classification changed")
        _text(decision["notes"], "human match notes")
    if decided != available:
        raise PriorArtError("human review did not inspect every sealed nearest match")
    if review["overall_classification"] not in {
        "known_rewrite",
        "cross_domain_synthesis_with_known_components",
        "potentially_distinct_not_novelty_cleared",
        "unresolved",
    }:
        raise PriorArtError("human overall classification changed")
    _text(review["notes"], "human review notes")
    body: dict[str, Any] = {
        "schema_version": ADJUDICATED_SCHEMA,
        "claim_id": screen["claim_id"],
        "screen_content_sha256": screen["content_sha256"],
        "review": dict(review),
        "release_gate": {
            "claim_prior_art_complete": True,
            "novelty_language_authorized": False,
            "status": "HUMAN_REVIEW_COMPLETE_NOVELTY_NOT_ESTABLISHED",
        },
        "claims": {
            "automated_absence_establishes_novelty": False,
            "human_review_alone_establishes_novelty": False,
            "literature_novelty_established": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--root", type=Path, default=Path.cwd())
    screen.add_argument("--claim", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, required=True)
    review = subparsers.add_parser("review")
    review.add_argument("--root", type=Path, default=Path.cwd())
    review.add_argument("--screen", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "screen":
        receipt = run_screen(args.root, args.claim)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "behavior": receipt["channel_assessment"]["behavior"],
            "content_sha256": receipt["content_sha256"],
            "human_review": receipt["human_review"]["status"],
            "status": receipt["automated_screen"]["status"],
        }
    elif args.command == "validate":
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_screen(receipt, args.root)
        summary = {"content_sha256": receipt["content_sha256"], "status": "VALID"}
    else:
        screen_receipt = json.loads(args.screen.read_text(encoding="utf-8"))
        review_document = json.loads(args.review.read_text(encoding="utf-8"))
        receipt = adjudicate_screen(screen_receipt, review_document, load_config(args.root))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "content_sha256": receipt["content_sha256"],
            "status": receipt["release_gate"]["status"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
