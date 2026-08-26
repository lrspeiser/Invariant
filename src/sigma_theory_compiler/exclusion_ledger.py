"""Sound negative-knowledge ledger for counterexample-driven formula and proof search."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .sigma_core import canonical_sha256

CERTIFICATE_SCHEMA = "invariant-exclusion-certificate-1.0"
LEDGER_SCHEMA = "invariant-exclusion-ledger-1.0"
_OPS = {"eq", "ne", "in", "contains", "lt", "le", "gt", "ge"}
_GENERALIZATION_LEVELS = {"exact_instance", "finite_enumerated_region", "proved_parametric_family"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExclusionLedgerError(ValueError):
    """An exclusion tried to exceed its verified scope or a ledger failed replay."""


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["content_sha256"] = canonical_sha256(body)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    if value.get("schema_version") != schema:
        raise ExclusionLedgerError(f"{label} schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ExclusionLedgerError(f"{label} seal changed")


def make_certificate(
    *,
    certificate_id: str,
    domain: str,
    failure_mode: str,
    scope_predicate: Sequence[Mapping[str, Any]],
    witness: Mapping[str, Any],
    verifier: Mapping[str, Any],
    generalization_level: str,
    status: str = "VERIFIED_EXCLUSION",
    explanation: str,
) -> dict[str, Any]:
    body = {
        "schema_version": CERTIFICATE_SCHEMA,
        "certificate_id": certificate_id,
        "domain": domain,
        "failure_mode": failure_mode,
        "scope_predicate": [dict(clause) for clause in scope_predicate],
        "witness": dict(witness),
        "verifier": dict(verifier),
        "generalization_level": generalization_level,
        "status": status,
        "explanation": explanation,
        "claims": {
            "may_prune_matching_candidates": status == "VERIFIED_EXCLUSION",
            "may_prune_outside_scope": False,
            "historical_novelty_established": False,
        },
    }
    result = _sealed(body)
    validate_certificate(result)
    return result


def validate_certificate(certificate: Mapping[str, Any]) -> None:
    _validate_seal(certificate, CERTIFICATE_SCHEMA, "exclusion certificate")
    status = certificate.get("status")
    level = certificate.get("generalization_level")
    predicate = certificate.get("scope_predicate")
    verifier = certificate.get("verifier")
    if not isinstance(certificate.get("certificate_id"), str) or not certificate[
        "certificate_id"
    ]:
        raise ExclusionLedgerError("exclusion certificate ID is invalid")
    if not isinstance(certificate.get("domain"), str) or not certificate["domain"]:
        raise ExclusionLedgerError("exclusion certificate domain is invalid")
    if not isinstance(certificate.get("failure_mode"), str) or not certificate[
        "failure_mode"
    ]:
        raise ExclusionLedgerError("exclusion failure mode is invalid")
    if not isinstance(certificate.get("witness"), Mapping):
        raise ExclusionLedgerError("exclusion witness is invalid")
    if not isinstance(certificate.get("explanation"), str) or not certificate[
        "explanation"
    ]:
        raise ExclusionLedgerError("exclusion explanation is invalid")
    if status not in {"VERIFIED_EXCLUSION", "HEURISTIC_FAILURE"}:
        raise ExclusionLedgerError("exclusion certificate status changed")
    if level not in _GENERALIZATION_LEVELS:
        raise ExclusionLedgerError("exclusion generalization level changed")
    if not isinstance(predicate, list) or not predicate:
        raise ExclusionLedgerError("exclusion scope is empty")
    for clause in predicate:
        if (
            not isinstance(clause, Mapping)
            or set(clause) != {"field", "op", "value"}
            or not isinstance(clause["field"], str)
            or clause["op"] not in _OPS
        ):
            raise ExclusionLedgerError("exclusion predicate is invalid")
    if level == "exact_instance" and not any(
        clause["field"] == "canonical_sha256"
        and clause["op"] == "eq"
        and isinstance(clause["value"], str)
        for clause in predicate
    ):
        raise ExclusionLedgerError("exact-instance exclusion lacks a canonical hash")
    if (
        not isinstance(verifier, Mapping)
        or not isinstance(verifier.get("artifact_sha256"), str)
        or _SHA256.fullmatch(verifier["artifact_sha256"]) is None
        or status == "VERIFIED_EXCLUSION"
        and verifier.get("decision") != "REJECT"
    ):
        raise ExclusionLedgerError("verified exclusion lacks rejecting verifier evidence")
    claims = certificate.get("claims", {})
    if (
        claims.get("may_prune_matching_candidates")
        is not (status == "VERIFIED_EXCLUSION")
        or claims.get("may_prune_outside_scope") is not False
        or claims.get("historical_novelty_established") is not False
    ):
        raise ExclusionLedgerError("exclusion claim boundary changed")


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return isinstance(expected, list) and actual in expected
    if op == "contains":
        return isinstance(actual, (list, str)) and expected in actual
    if actual is None or isinstance(actual, bool) or isinstance(expected, bool):
        return False
    try:
        if op == "lt":
            return bool(actual < expected)
        if op == "le":
            return bool(actual <= expected)
        if op == "gt":
            return bool(actual > expected)
        if op == "ge":
            return bool(actual >= expected)
    except TypeError:
        return False
    raise ExclusionLedgerError(f"unknown predicate operation: {op}")


def certificate_matches(
    certificate: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    validate_certificate(certificate)
    features = candidate.get("features", {})
    if not isinstance(features, Mapping):
        raise ExclusionLedgerError("candidate features are invalid")
    view = {**features, "canonical_sha256": candidate.get("canonical_sha256")}
    return all(
        clause["field"] in view
        and _compare(view[clause["field"]], clause["op"], clause["value"])
        for clause in certificate["scope_predicate"]
    )


def evaluate_candidate(
    candidate: Mapping[str, Any], certificates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if (
        not isinstance(candidate.get("candidate_id"), str)
        or not isinstance(candidate.get("canonical_sha256"), str)
        or _SHA256.fullmatch(candidate["canonical_sha256"]) is None
    ):
        raise ExclusionLedgerError("candidate identity is invalid")
    verified_matches = []
    heuristic_matches = []
    for certificate in certificates:
        if certificate_matches(certificate, candidate):
            target = (
                verified_matches
                if certificate["status"] == "VERIFIED_EXCLUSION"
                else heuristic_matches
            )
            target.append(certificate["certificate_id"])
    return {
        "candidate_id": candidate["candidate_id"],
        "canonical_sha256": candidate["canonical_sha256"],
        "decision": "EXCLUDE" if verified_matches else "SURVIVE",
        "verified_exclusion_certificate_ids": sorted(verified_matches),
        "heuristic_warning_certificate_ids": sorted(heuristic_matches),
    }


def build_ledger(
    *,
    ledger_id: str,
    domain: str,
    candidates: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    frontier_constraints: Sequence[Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    certificate_ids = [row.get("certificate_id") for row in certificates]
    if len(set(certificate_ids)) != len(certificate_ids):
        raise ExclusionLedgerError("duplicate exclusion certificate ID")
    outcomes = [evaluate_candidate(candidate, certificates) for candidate in candidates]
    outcomes.sort(key=lambda row: row["candidate_id"])
    certificate_rows = sorted(
        [dict(row) for row in certificates], key=lambda row: row["certificate_id"]
    )
    excluded = [row for row in outcomes if row["decision"] == "EXCLUDE"]
    survivors = [row for row in outcomes if row["decision"] == "SURVIVE"]
    body = {
        "schema_version": LEDGER_SCHEMA,
        "ledger_id": ledger_id,
        "domain": domain,
        "source_bindings": dict(source_bindings),
        "certificates": certificate_rows,
        "candidate_outcomes": outcomes,
        "survivor_frontier": {
            "candidate_ids": [row["candidate_id"] for row in survivors],
            "constraints": [dict(row) for row in frontier_constraints],
        },
        "counts": {
            "candidates": len(outcomes),
            "certificates": len(certificates),
            "verified_certificates": sum(
                row["status"] == "VERIFIED_EXCLUSION" for row in certificates
            ),
            "heuristic_certificates": sum(
                row["status"] == "HEURISTIC_FAILURE" for row in certificates
            ),
            "excluded_candidates": len(excluded),
            "surviving_candidates": len(survivors),
        },
        "claims": {
            "only_verified_scope_matches_were_pruned": True,
            "heuristic_failures_pruned_candidates": False,
            "evaluated_survivors_present": bool(survivors),
            "unexplored_remainder_characterized_not_certified_novel": True,
            "historical_novelty_established": False,
        },
    }
    result = _sealed(body)
    validate_ledger(result, candidates)
    return result


def validate_ledger(
    ledger: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> None:
    _validate_seal(ledger, LEDGER_SCHEMA, "exclusion ledger")
    for certificate in ledger.get("certificates", []):
        validate_certificate(certificate)
    outcomes = ledger.get("candidate_outcomes", [])
    replay = [evaluate_candidate(row, ledger["certificates"]) for row in candidates]
    replay.sort(key=lambda row: row["candidate_id"])
    if replay != outcomes:
        raise ExclusionLedgerError("exclusion ledger candidate replay changed")
    counts = ledger.get("counts", {})
    if (
        counts.get("candidates") != len(outcomes)
        or counts.get("excluded_candidates")
        != sum(row["decision"] == "EXCLUDE" for row in outcomes)
        or counts.get("surviving_candidates")
        != sum(row["decision"] == "SURVIVE" for row in outcomes)
        or ledger.get("claims", {}).get("only_verified_scope_matches_were_pruned")
        is not True
        or ledger.get("claims", {}).get("heuristic_failures_pruned_candidates") is not False
        or ledger.get("claims", {}).get(
            "unexplored_remainder_characterized_not_certified_novel"
        )
        is not True
        or ledger.get("claims", {}).get("historical_novelty_established") is not False
    ):
        raise ExclusionLedgerError("exclusion ledger counts or claims changed")
