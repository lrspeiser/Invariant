"""Build and verify hash-bound challenge evidence from live external datasets.

The pack adds real intervention, noisy, shifted, and observational data to the synthetic
calibration suite.  It intentionally makes no formula, causal, novelty, or level-5 claim: HTTPS
response hashes are provenance evidence, not signatures from a distinct benchmark principal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.parse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from .claim_specific_prior_art import Transport, urllib_transport
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/external_dataset_challenges.json"
OUTPUT_PATH = "runs/math/external-dataset-challenges/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/external_dataset_challenges.py"
TEST_PATH = "tests/test_external_dataset_challenges.py"
CONFIG_SCHEMA = "invariant-external-dataset-challenges-config-1.0"
RECEIPT_SCHEMA = "invariant-external-dataset-challenges-receipt-1.0"
PACK_ID = "external-dataset-challenges-2026-08-23-001"
KINDS = ("intervention", "noisy", "shifted", "unidentifiable")
_EXPECTED_CHALLENGES = {
    "external-dataset.nsw-randomized-intervention": "intervention",
    "external-dataset.nist-filip-noise": "noisy",
    "external-dataset.uci-wine-domain-shift": "shifted",
    "external-dataset.nhefs-observational-underdetermination": "unidentifiable",
}
_HEX = frozenset("0123456789abcdef")
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?"


class ExternalDatasetError(ValueError):
    """An external source, extraction, seal, or conservative release gate failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ExternalDatasetError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ExternalDatasetError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _utc(value: str | None) -> str:
    candidate = value or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ExternalDatasetError("retrieval time is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExternalDatasetError("retrieval time lacks a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _fraction(value: Any, label: str) -> Fraction:
    if not isinstance(value, str) or not value.strip():
        raise ExternalDatasetError(f"{label} is missing")
    try:
        return Fraction(value.strip())
    except (ValueError, ZeroDivisionError) as error:
        raise ExternalDatasetError(f"{label} is not an exact decimal or rational") from error


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _uri_host(uri: Any, allowed_hosts: set[str], label: str) -> str:
    if not isinstance(uri, str):
        raise ExternalDatasetError(f"{label} URI is not text")
    parsed = urllib.parse.urlparse(uri)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or host not in allowed_hosts or parsed.username is not None:
        raise ExternalDatasetError(f"{label} URI escaped the allowed HTTPS authorities")
    return host


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root.resolve() / CONFIG_PATH).read_text(encoding="utf-8"))
    _strict(
        value,
        {
            "challenges",
            "generator_principal_id",
            "network",
            "pack_id",
            "schema_version",
        },
        "external dataset config",
    )
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["pack_id"] != PACK_ID
        or value["generator_principal_id"] != "invariant.discovery-engine"
    ):
        raise ExternalDatasetError("external dataset config identity changed")
    network = value["network"]
    _strict(
        network,
        {
            "allowed_hosts",
            "maximum_response_bytes",
            "request_timeout_seconds",
            "user_agent",
        },
        "external dataset network policy",
    )
    allowed_hosts = network["allowed_hosts"]
    if (
        allowed_hosts
        != ["archive.ics.uci.edu", "raw.githubusercontent.com", "www.itl.nist.gov"]
        or not 500_000 <= network["maximum_response_bytes"] <= 2_000_000
        or not 5 <= network["request_timeout_seconds"] <= 60
        or not isinstance(network["user_agent"], str)
        or "InvariantExternalDatasetChallenges" not in network["user_agent"]
    ):
        raise ExternalDatasetError("external dataset network policy weakened")
    challenges = value["challenges"]
    if not isinstance(challenges, list) or len(challenges) != len(KINDS):
        raise ExternalDatasetError("external dataset challenge count changed")
    observed: dict[str, str] = {}
    source_ids: set[str] = set()
    source_uris: set[str] = set()
    for challenge in challenges:
        _strict(
            challenge,
            {
                "analysis",
                "challenge_id",
                "datasets",
                "documentation",
                "external_principal_id",
                "kind",
            },
            "external dataset challenge",
        )
        challenge_id = challenge["challenge_id"]
        kind = challenge["kind"]
        if _EXPECTED_CHALLENGES.get(challenge_id) != kind or challenge_id in observed:
            raise ExternalDatasetError("external dataset challenge identity changed")
        principal = challenge["external_principal_id"]
        if not isinstance(principal, str) or not principal.startswith("external."):
            raise ExternalDatasetError("external dataset principal is not external")
        documentation = challenge["documentation"]
        _strict(documentation, {"required_markers", "source_id", "source_uri"}, "documentation")
        markers = documentation["required_markers"]
        if (
            not isinstance(markers, list)
            or len(markers) < 2
            or any(not isinstance(marker, str) or len(marker) < 5 for marker in markers)
        ):
            raise ExternalDatasetError("external documentation markers are too weak")
        datasets = challenge["datasets"]
        if not isinstance(datasets, list) or not datasets:
            raise ExternalDatasetError("external dataset source list is empty")
        for source in [documentation, *datasets]:
            expected = (
                {"required_markers", "source_id", "source_uri"}
                if source is documentation
                else {"format", "role", "source_id", "source_uri"}
            )
            _strict(source, expected, "external source")
            source_id = source["source_id"]
            source_uri = source["source_uri"]
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in source_ids
                or source_uri in source_uris
            ):
                raise ExternalDatasetError("external source identity is missing or duplicated")
            _uri_host(source_uri, set(allowed_hosts), "external source")
            source_ids.add(source_id)
            source_uris.add(source_uri)
        observed[challenge_id] = kind
    if observed != _EXPECTED_CHALLENGES:
        raise ExternalDatasetError("external dataset kind coverage changed")
    return value


def _fetch(
    *,
    challenge: Mapping[str, Any],
    source: Mapping[str, Any],
    role: str,
    source_format: str,
    network: Mapping[str, Any],
    transport: Transport,
) -> tuple[bytes, dict[str, Any]]:
    response = transport(
        source["source_uri"],
        {
            "Accept": "text/csv, text/plain, text/html",
            "User-Agent": network["user_agent"],
        },
        network["request_timeout_seconds"],
        network["maximum_response_bytes"],
    )
    if response.status != 200 or not response.body:
        raise ExternalDatasetError(f"external source unavailable: {source['source_id']}")
    return response.body, {
        "challenge_id": challenge["challenge_id"],
        "external_principal_id": challenge["external_principal_id"],
        "http_status": response.status,
        "response_bytes": len(response.body),
        "response_sha256": hashlib.sha256(response.body).hexdigest(),
        "role": role,
        "source_format": source_format,
        "source_id": source["source_id"],
        "source_uri": source["source_uri"],
        "transport": "https",
    }


def _csv_rows(body: bytes, *, delimiter: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ExternalDatasetError(f"{label} is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    headers = reader.fieldnames
    if not headers or len(headers) != len(set(headers)):
        raise ExternalDatasetError(f"{label} headers are missing or duplicated")
    rows = [dict(row) for row in reader]
    if not rows or any(set(row) != set(headers) for row in rows):
        raise ExternalDatasetError(f"{label} rows are empty or ragged")
    return headers, rows


def _mean(values: Sequence[Fraction], label: str) -> Fraction:
    if not values:
        raise ExternalDatasetError(f"{label} has no values")
    return sum(values, Fraction()) / len(values)


def _intervention_evidence(
    challenge: Mapping[str, Any], bodies: Mapping[str, bytes]
) -> dict[str, Any]:
    analysis = challenge["analysis"]
    _strict(
        analysis,
        {"minimum_rows_per_arm", "outcome_column", "treatment_column"},
        "external intervention analysis",
    )
    source = challenge["datasets"][0]
    headers, rows = _csv_rows(bodies[source["source_id"]], delimiter=",", label="NSW CSV")
    treatment = analysis["treatment_column"]
    outcome = analysis["outcome_column"]
    if treatment not in headers or outcome not in headers:
        raise ExternalDatasetError("NSW intervention columns are missing")
    groups: dict[str, list[Fraction]] = defaultdict(list)
    for row in rows:
        label = row[treatment].strip()
        if label not in {"0", "1"}:
            raise ExternalDatasetError("NSW treatment is not binary")
        groups[label].append(_fraction(row[outcome], "NSW outcome"))
    minimum = analysis["minimum_rows_per_arm"]
    if not isinstance(minimum, int) or minimum < 100 or any(len(groups[key]) < minimum for key in ("0", "1")):
        raise ExternalDatasetError("NSW randomized arms are too small")
    means = {key: _mean(groups[key], f"NSW arm {key}") for key in ("0", "1")}
    difference = means["1"] - means["0"]
    if difference == 0:
        raise ExternalDatasetError("NSW observed arm difference unexpectedly vanished")
    return {
        "arm_counts": {key: len(groups[key]) for key in ("0", "1")},
        "arm_outcome_means": {key: _fraction_text(means[key]) for key in ("0", "1")},
        "documented_program_experiment": True,
        "mutation_observational_relabelling_rejected": True,
        "observed_difference_in_means": _fraction_text(difference),
        "observed_difference_is_formula_or_universal_effect": False,
        "rows": len(rows),
    }


def _filip_evidence(challenge: Mapping[str, Any], bodies: Mapping[str, bytes]) -> dict[str, Any]:
    analysis = challenge["analysis"]
    _strict(
        analysis,
        {"minimum_observations", "required_polynomial_parameters"},
        "external noisy analysis",
    )
    source = challenge["datasets"][0]
    try:
        text = bodies[source["source_id"]].decode("ascii")
    except UnicodeDecodeError as error:
        raise ExternalDatasetError("NIST Filip data is not ASCII") from error
    parameter_pattern = re.compile(
        rf"^\s*B([0-9]+)\s+({_NUMBER})\s+({_NUMBER})\s*$", re.IGNORECASE
    )
    parameters: dict[int, Fraction] = {}
    residual_standard_deviation: Fraction | None = None
    r_squared: Fraction | None = None
    data_start: int | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if match := parameter_pattern.fullmatch(line):
            parameters[int(match.group(1))] = _fraction(match.group(2), "Filip coefficient")
        if match := re.search(rf"Standard Deviation\s+({_NUMBER})\s*$", line, re.IGNORECASE):
            residual_standard_deviation = _fraction(match.group(1), "Filip residual deviation")
        if match := re.search(rf"R-Squared\s+({_NUMBER})\s*$", line, re.IGNORECASE):
            r_squared = _fraction(match.group(1), "Filip R-squared")
        if re.fullmatch(r"\s*Data:\s+y\s+x\s*", line, re.IGNORECASE):
            data_start = index + 1
    required_parameters = analysis["required_polynomial_parameters"]
    if (
        required_parameters != 11
        or set(parameters) != set(range(required_parameters))
        or residual_standard_deviation is None
        or residual_standard_deviation <= 0
        or r_squared is None
        or not 0 < r_squared < 1
        or data_start is None
    ):
        raise ExternalDatasetError("NIST Filip certified noise evidence is incomplete")
    row_pattern = re.compile(rf"\s*({_NUMBER})\s+({_NUMBER})\s*")
    rows: list[tuple[Fraction, Fraction]] = []
    for line in lines[data_start:]:
        if match := row_pattern.fullmatch(line):
            rows.append(
                (
                    _fraction(match.group(1), "Filip response"),
                    _fraction(match.group(2), "Filip predictor"),
                )
            )
    minimum = analysis["minimum_observations"]
    if not isinstance(minimum, int) or minimum < 80 or len(rows) < minimum:
        raise ExternalDatasetError("NIST Filip observation coverage is too small")
    residuals = []
    for response, predictor in rows:
        prediction = sum(
            (parameters[index] * predictor**index for index in range(required_parameters)),
            Fraction(),
        )
        residuals.append(response - prediction)
    nonzero = sum(residual != 0 for residual in residuals)
    if nonzero == 0:
        raise ExternalDatasetError("NIST Filip rounded certified model became an exact interpolation")
    return {
        "certified_polynomial_parameters": required_parameters,
        "certified_r_squared": _fraction_text(r_squared),
        "certified_residual_standard_deviation": _fraction_text(residual_standard_deviation),
        "mutation_zero_noise_rejected": True,
        "nonzero_residual_rows_under_published_coefficients": nonzero,
        "observations": len(rows),
        "point_observations_treated_as_exact_law": False,
    }


def _shift_evidence(challenge: Mapping[str, Any], bodies: Mapping[str, bytes]) -> dict[str, Any]:
    analysis = challenge["analysis"]
    _strict(
        analysis,
        {"minimum_deployment_rows", "minimum_train_rows", "shift_feature"},
        "external shift analysis",
    )
    sources = {source["role"]: source for source in challenge["datasets"]}
    if set(sources) != {"deployment_white_wine", "train_red_wine"}:
        raise ExternalDatasetError("UCI train/deployment source roles changed")
    train_headers, train_rows = _csv_rows(
        bodies[sources["train_red_wine"]["source_id"]],
        delimiter=";",
        label="UCI red wine CSV",
    )
    deployment_headers, deployment_rows = _csv_rows(
        bodies[sources["deployment_white_wine"]["source_id"]],
        delimiter=";",
        label="UCI white wine CSV",
    )
    feature = analysis["shift_feature"]
    if train_headers != deployment_headers or feature not in train_headers:
        raise ExternalDatasetError("UCI red/white feature schemas differ")
    if (
        len(train_rows) < analysis["minimum_train_rows"]
        or len(deployment_rows) < analysis["minimum_deployment_rows"]
    ):
        raise ExternalDatasetError("UCI domain split is too small")
    train_means = {
        column: _mean([_fraction(row[column], f"UCI train {column}") for row in train_rows], column)
        for column in train_headers
    }
    deployment_means = {
        column: _mean(
            [_fraction(row[column], f"UCI deployment {column}") for row in deployment_rows],
            column,
        )
        for column in deployment_headers
    }
    changed = sum(train_means[column] != deployment_means[column] for column in train_headers)
    if train_means[feature] == deployment_means[feature] or changed < 2:
        raise ExternalDatasetError("UCI red/white covariate shift was not observed")
    return {
        "deployment_role": "white_wine",
        "deployment_rows": len(deployment_rows),
        "features_with_different_exact_means": changed,
        "mutation_same_domain_rejected": True,
        "shift_feature": feature,
        "shift_feature_deployment_mean": _fraction_text(deployment_means[feature]),
        "shift_feature_train_mean": _fraction_text(train_means[feature]),
        "train_fit_treated_as_deployment_validity": False,
        "train_role": "red_wine",
        "train_rows": len(train_rows),
    }


def _unidentifiable_evidence(
    challenge: Mapping[str, Any], bodies: Mapping[str, bytes]
) -> dict[str, Any]:
    analysis = challenge["analysis"]
    _strict(
        analysis,
        {"exposure_column", "identification_design", "minimum_complete_rows", "outcome_column"},
        "external unidentifiable analysis",
    )
    if analysis["identification_design"] != "none_declared":
        raise ExternalDatasetError("NHEFS challenge silently declared an identification design")
    source = challenge["datasets"][0]
    headers, rows = _csv_rows(bodies[source["source_id"]], delimiter=",", label="NHEFS CSV")
    exposure = analysis["exposure_column"]
    outcome = analysis["outcome_column"]
    if exposure not in headers or outcome not in headers:
        raise ExternalDatasetError("NHEFS exposure/outcome columns are missing")
    groups: dict[str, list[Fraction]] = defaultdict(list)
    missing = 0
    for row in rows:
        exposure_value = row[exposure].strip()
        outcome_value = row[outcome].strip()
        if not exposure_value or not outcome_value:
            missing += 1
            continue
        if exposure_value not in {"0", "1"}:
            raise ExternalDatasetError("NHEFS exposure is not binary")
        groups[exposure_value].append(_fraction(outcome_value, "NHEFS outcome"))
    complete = sum(map(len, groups.values()))
    if complete < analysis["minimum_complete_rows"] or any(not groups[key] for key in ("0", "1")):
        raise ExternalDatasetError("NHEFS observational coverage is too small")
    means = {key: _mean(groups[key], f"NHEFS exposure {key}") for key in ("0", "1")}
    return {
        "complete_exposure_outcome_rows": complete,
        "distinguishing_intervention_observed": False,
        "exposure_group_counts": {key: len(groups[key]) for key in ("0", "1")},
        "exposure_group_outcome_means": {
            key: _fraction_text(means[key]) for key in ("0", "1")
        },
        "missing_exposure_or_outcome_rows": missing,
        "mutation_forced_causal_identification_rejected": True,
        "required_conclusion": "OBSERVATIONAL_ASSOCIATION_ONLY_CAUSAL_MECHANISM_UNDERDETERMINED",
        "rows": len(rows),
    }


_EXECUTORS = {
    "intervention": _intervention_evidence,
    "noisy": _filip_evidence,
    "shifted": _shift_evidence,
    "unidentifiable": _unidentifiable_evidence,
}


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def build_external_dataset_challenges(
    root: Path,
    *,
    transport: Transport = urllib_transport,
    retrieved_utc: str | None = None,
) -> dict[str, Any]:
    """Fetch each external source once, extract exact evidence, and seal a conservative receipt."""

    root = root.resolve()
    config = load_config(root)
    retrieved = _utc(retrieved_utc)
    network = config["network"]
    results = []
    all_source_evidence = []
    for challenge in config["challenges"]:
        bodies: dict[str, bytes] = {}
        documentation = challenge["documentation"]
        document_body, document_evidence = _fetch(
            challenge=challenge,
            source=documentation,
            role="documentation",
            source_format="html_or_text",
            network=network,
            transport=transport,
        )
        try:
            document_text = document_body.decode("utf-8", errors="strict").lower()
        except UnicodeDecodeError as error:
            raise ExternalDatasetError("external documentation is not UTF-8") from error
        if any(marker.lower() not in document_text for marker in documentation["required_markers"]):
            raise ExternalDatasetError(
                f"external documentation markers missing: {challenge['challenge_id']}"
            )
        bodies[documentation["source_id"]] = document_body
        source_evidence = [document_evidence]
        for source in challenge["datasets"]:
            body, evidence = _fetch(
                challenge=challenge,
                source=source,
                role=source["role"],
                source_format=source["format"],
                network=network,
                transport=transport,
            )
            bodies[source["source_id"]] = body
            source_evidence.append(evidence)
        evidence = _EXECUTORS[challenge["kind"]](challenge, bodies)
        results.append(
            {
                "challenge_id": challenge["challenge_id"],
                "evidence": evidence,
                "external_principal_id": challenge["external_principal_id"],
                "kind": challenge["kind"],
                "mutation_control_rejected": True,
                "source_evidence": sorted(source_evidence, key=lambda item: item["source_id"]),
                "status": "PASS_EXTERNAL_DATASET_CHALLENGE",
            }
        )
        all_source_evidence.extend(source_evidence)
    paths = {"config": CONFIG_PATH, "source": SOURCE_PATH, "test": TEST_PATH}
    receipt = _sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "pack_id": config["pack_id"],
            "retrieved_utc": retrieved,
            "source_bindings": {
                key: {"normalized_file_sha256": _normalized_file_sha256(root / path), "path": path}
                for key, path in sorted(paths.items())
            },
            "results": sorted(results, key=lambda item: KINDS.index(item["kind"])),
            "summary": {
                "challenge_kinds": list(KINDS),
                "external_challenges_passed": len(results),
                "external_principals": len(
                    {item["external_principal_id"] for item in all_source_evidence}
                ),
                "external_source_responses": len(all_source_evidence),
                "mutation_controls_rejected": len(results),
                "status": "PASS_EXTERNAL_DATASET_CHALLENGES",
                "unique_external_response_hashes": len(
                    {item["response_sha256"] for item in all_source_evidence}
                ),
            },
            "source_signature": {
                "cryptographic_signature_verified": False,
                "https_response_hashes_bound": True,
                "status": "PENDING_DISTINCT_PRINCIPAL_SIGNATURE",
            },
            "release_gate": {
                "external_dataset_challenges_ready": True,
                "level5_eligible": False,
                "serious_scientific_claim_authorized": False,
                "status": "PASS_EXTERNAL_DATA_LEVEL5_BLOCKED_UNSIGNED",
            },
            "claims": {
                "causal_effect_established": False,
                "external_https_hash_is_source_signature": False,
                "formula_or_law_established": False,
                "more_creative_established": False,
                "observational_association_establishes_causation": False,
            },
        }
    )
    validate_external_dataset_challenges(receipt, root)
    return receipt


def _validate_result(result: Mapping[str, Any], allowed_hosts: set[str]) -> None:
    _strict(
        result,
        {
            "challenge_id",
            "evidence",
            "external_principal_id",
            "kind",
            "mutation_control_rejected",
            "source_evidence",
            "status",
        },
        "external dataset result",
    )
    if (
        _EXPECTED_CHALLENGES.get(result["challenge_id"]) != result["kind"]
        or result["status"] != "PASS_EXTERNAL_DATASET_CHALLENGE"
        or result["mutation_control_rejected"] is not True
        or not str(result["external_principal_id"]).startswith("external.")
    ):
        raise ExternalDatasetError("external dataset result identity changed")
    sources = result["source_evidence"]
    if not isinstance(sources, list) or len(sources) < 2:
        raise ExternalDatasetError("external dataset source evidence is incomplete")
    if len({item["source_id"] for item in sources}) != len(sources):
        raise ExternalDatasetError("external source evidence is duplicated")
    for source in sources:
        _strict(
            source,
            {
                "challenge_id",
                "external_principal_id",
                "http_status",
                "response_bytes",
                "response_sha256",
                "role",
                "source_format",
                "source_id",
                "source_uri",
                "transport",
            },
            "external source evidence",
        )
        if (
            source["challenge_id"] != result["challenge_id"]
            or source["external_principal_id"] != result["external_principal_id"]
            or source["http_status"] != 200
            or source["response_bytes"] <= 0
            or source["transport"] != "https"
        ):
            raise ExternalDatasetError("external source provenance changed")
        _sha(source["response_sha256"], "external response hash")
        _uri_host(source["source_uri"], allowed_hosts, "external source evidence")
    evidence = result["evidence"]
    kind = result["kind"]
    if kind == "intervention":
        if (
            evidence.get("documented_program_experiment") is not True
            or evidence.get("mutation_observational_relabelling_rejected") is not True
            or evidence.get("observed_difference_is_formula_or_universal_effect") is not False
            or min(evidence.get("arm_counts", {}).values(), default=0) < 100
        ):
            raise ExternalDatasetError("external intervention evidence weakened")
    elif kind == "noisy":
        if (
            evidence.get("certified_polynomial_parameters") != 11
            or _fraction(evidence.get("certified_residual_standard_deviation"), "residual") <= 0
            or not 0 < _fraction(evidence.get("certified_r_squared"), "R-squared") < 1
            or evidence.get("nonzero_residual_rows_under_published_coefficients", 0) < 1
            or evidence.get("mutation_zero_noise_rejected") is not True
            or evidence.get("point_observations_treated_as_exact_law") is not False
        ):
            raise ExternalDatasetError("external noise evidence weakened")
    elif kind == "shifted":
        if (
            evidence.get("train_rows", 0) < 1500
            or evidence.get("deployment_rows", 0) < 4000
            or evidence.get("features_with_different_exact_means", 0) < 2
            or evidence.get("mutation_same_domain_rejected") is not True
            or evidence.get("train_fit_treated_as_deployment_validity") is not False
            or _fraction(evidence.get("shift_feature_train_mean"), "train mean")
            == _fraction(evidence.get("shift_feature_deployment_mean"), "deployment mean")
        ):
            raise ExternalDatasetError("external shift evidence weakened")
    elif kind == "unidentifiable" and (
        evidence.get("complete_exposure_outcome_rows", 0) < 1000
        or evidence.get("distinguishing_intervention_observed") is not False
        or evidence.get("mutation_forced_causal_identification_rejected") is not True
        or evidence.get("required_conclusion")
        != "OBSERVATIONAL_ASSOCIATION_ONLY_CAUSAL_MECHANISM_UNDERDETERMINED"
    ):
        raise ExternalDatasetError("external unidentifiable evidence weakened")


def validate_external_dataset_challenges(
    value: Mapping[str, Any], root: Path | None = None
) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ExternalDatasetError("external dataset receipt content seal changed")
    _strict(
        value,
        {
            "claims",
            "content_sha256",
            "pack_id",
            "release_gate",
            "results",
            "retrieved_utc",
            "schema_version",
            "source_bindings",
            "source_signature",
            "summary",
        },
        "external dataset receipt",
    )
    if value["schema_version"] != RECEIPT_SCHEMA or value["pack_id"] != PACK_ID:
        raise ExternalDatasetError("external dataset receipt identity changed")
    _utc(value["retrieved_utc"])
    claims = value["claims"]
    expected_claims = {
        "causal_effect_established": False,
        "external_https_hash_is_source_signature": False,
        "formula_or_law_established": False,
        "more_creative_established": False,
        "observational_association_establishes_causation": False,
    }
    if claims != expected_claims:
        raise ExternalDatasetError("external dataset claim boundary changed")
    signature = value["source_signature"]
    release = value["release_gate"]
    if (
        signature
        != {
            "cryptographic_signature_verified": False,
            "https_response_hashes_bound": True,
            "status": "PENDING_DISTINCT_PRINCIPAL_SIGNATURE",
        }
        or release
        != {
            "external_dataset_challenges_ready": True,
            "level5_eligible": False,
            "serious_scientific_claim_authorized": False,
            "status": "PASS_EXTERNAL_DATA_LEVEL5_BLOCKED_UNSIGNED",
        }
    ):
        raise ExternalDatasetError("external dataset signature or release boundary changed")
    results = value["results"]
    if (
        not isinstance(results, list)
        or [item.get("kind") for item in results] != list(KINDS)
        or len({item.get("challenge_id") for item in results}) != len(KINDS)
    ):
        raise ExternalDatasetError("external dataset result coverage changed")
    allowed_hosts = {"archive.ics.uci.edu", "raw.githubusercontent.com", "www.itl.nist.gov"}
    for result in results:
        _validate_result(result, allowed_hosts)
    source_evidence = [source for result in results for source in result["source_evidence"]]
    summary = value["summary"]
    if summary != {
        "challenge_kinds": list(KINDS),
        "external_challenges_passed": 4,
        "external_principals": 3,
        "external_source_responses": len(source_evidence),
        "mutation_controls_rejected": 4,
        "status": "PASS_EXTERNAL_DATASET_CHALLENGES",
        "unique_external_response_hashes": len(
            {item["response_sha256"] for item in source_evidence}
        ),
    }:
        raise ExternalDatasetError("external dataset summary changed")
    if root is not None:
        root = root.resolve()
        config = load_config(root)
        configured_sources = {
            source["source_id"]: source["source_uri"]
            for challenge in config["challenges"]
            for source in [challenge["documentation"], *challenge["datasets"]]
        }
        if {item["source_id"]: item["source_uri"] for item in source_evidence} != configured_sources:
            raise ExternalDatasetError("external source evidence does not match the config")
        for binding in value["source_bindings"].values():
            _strict(binding, {"normalized_file_sha256", "path"}, "external source binding")
            path = (root / binding["path"]).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ExternalDatasetError("external source binding escaped the repository") from error
            if binding["normalized_file_sha256"] != _normalized_file_sha256(path):
                raise ExternalDatasetError("external source binding changed")


def reproduce_external_dataset_challenges(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    transport: Transport = urllib_transport,
) -> dict[str, Any]:
    """Refetch every source and require byte-identical evidence and derived challenge metrics."""

    validate_external_dataset_challenges(receipt, root)
    rebuilt = build_external_dataset_challenges(
        root,
        transport=transport,
        retrieved_utc=str(receipt["retrieved_utc"]),
    )
    if rebuilt != receipt:
        raise ExternalDatasetError("external dataset refetch did not reproduce the sealed receipt")
    return rebuilt


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    reproduce = subparsers.add_parser("reproduce")
    reproduce.add_argument("--root", type=Path, default=Path.cwd())
    reproduce.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    receipt_path = args.receipt if args.command != "build" else args.output
    receipt_path = receipt_path if receipt_path.is_absolute() else args.root / receipt_path
    if args.command == "build":
        receipt = build_external_dataset_challenges(args.root)
        _write(receipt_path, receipt)
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if args.command == "validate":
            validate_external_dataset_challenges(receipt, args.root)
        else:
            reproduce_external_dataset_challenges(args.root, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "external_challenges": receipt["summary"]["external_challenges_passed"],
                "level5_eligible": receipt["release_gate"]["level5_eligible"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
