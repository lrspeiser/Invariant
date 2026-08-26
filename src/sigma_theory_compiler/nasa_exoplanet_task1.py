"""Task-1 measured-catalog calibration with a frozen anonymous discovery phase.

The NASA Exoplanet Archive supplies real published catalog parameters.  The discovery lane sees
only positive columns named ``x0``, ``x1``, and ``x2`` plus their reported uncertainties.  Host
identities, physical column names, the target interpretation, and the newest host-group holdout are
not passed to candidate generation.

This first receipt is intentionally an exploratory pilot.  Aggregate holdout performance was
inspected while the implementation and thresholds were designed, so it cannot unlock the gated
roadmap.  Its purpose is to build the machinery, measure the current system, and freeze a protocol
for a later externally selected holdout.  Catalog parameters may also be inferred or mutually
dependent; recovering a classical relation from them is not independent physical confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .anonymous_monomial_discovery import (
    AnonymousMonomialError,
    discover,
    score_frozen_candidate,
)
from .sigma_core import canonical_sha256
from .tolerance_aware_fitting import write_immutable

CONFIG_SCHEMA = "invariant-nasa-exoplanet-task1-config-1.0"
TARGET_SCHEMA = "invariant-nasa-exoplanet-task1-target-1.0"
RECEIPT_SCHEMA = "invariant-nasa-exoplanet-task1-receipt-1.0"
TRAINING_SCHEMA = "invariant-anonymous-positive-column-training-1.0"
AUTHORIZATION_SCHEMA = "invariant-nasa-exoplanet-task1-authorization-1.0"
CONFIG_PATH = "configs/nasa_exoplanet_task1.json"
SOURCE_PATH = "src/sigma_theory_compiler/nasa_exoplanet_task1.py"
GENERATOR_PATH = "src/sigma_theory_compiler/anonymous_monomial_discovery.py"
TEST_PATH = "tests/test_nasa_exoplanet_task1.py"

EXPECTED_COLUMNS = (
    "hostname",
    "pl_name",
    "disc_year",
    "pl_orbper",
    "pl_orbpererr1",
    "pl_orbpererr2",
    "pl_orbsmax",
    "pl_orbsmaxerr1",
    "pl_orbsmaxerr2",
    "st_mass",
    "st_masserr1",
    "st_masserr2",
    "pl_refname",
)
VALUE_COLUMNS = ("pl_orbper", "pl_orbsmax", "st_mass")
FORBIDDEN_GENERATOR_WORDS = frozenset(
    {
        "astronomy",
        "exoplanet",
        "kepler",
        "mass",
        "nasa",
        "orbit",
        "orbital",
        "period",
        "planet",
        "semimajor",
        "stellar",
    }
)


class NASAExoplanetTask1Error(ValueError):
    """Raised on source drift, leakage, chronology failure, or receipt tamper."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NASAExoplanetTask1Error(f"{label} keys changed")


def _normalized_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _normalized_sha256(raw: bytes) -> str:
    return hashlib.sha256(_normalized_bytes(raw)).hexdigest()


def _file_sha256(path: Path) -> str:
    return _normalized_sha256(path.read_bytes())


def _utc(value: str | None) -> str:
    candidate = value or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise NASAExoplanetTask1Error("retrieval time is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NASAExoplanetTask1Error("retrieval time lacks a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise NASAExoplanetTask1Error("authorization time is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NASAExoplanetTask1Error("authorization time lacks a UTC offset")
    return parsed.astimezone(UTC)


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise NASAExoplanetTask1Error("non-finite metric")
    return format(value, ".17g")


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise NASAExoplanetTask1Error("path is not portable")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise NASAExoplanetTask1Error("path escapes repository root") from error
    return path


def load_config(root: Path, config_path: str = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(_resolve(root, config_path).read_text(encoding="utf-8"))
    _strict(
        config,
        {
            "campaign_id",
            "classification",
            "discovery",
            "eligibility",
            "evaluation",
            "network",
            "outputs",
            "schema_version",
            "source",
            "split",
            "target_fixture",
        },
        "task-1 config",
    )
    if config["schema_version"] != CONFIG_SCHEMA:
        raise NASAExoplanetTask1Error("task-1 config schema changed")
    if not isinstance(config["campaign_id"], str) or not config["campaign_id"]:
        raise NASAExoplanetTask1Error("campaign identity is missing")

    source = config["source"]
    _strict(
        source,
        {
            "dependency_warning",
            "documentation_uri",
            "external_principal_id",
            "query",
            "table",
            "tap_base_uri",
        },
        "source config",
    )
    if source["external_principal_id"] != "external.nasa-exoplanet-archive":
        raise NASAExoplanetTask1Error("external source principal changed")
    if source["table"] != "ps" or not any(
        marker in source["query"] for marker in ("default_flag=1", "default_flag=0")
    ):
        raise NASAExoplanetTask1Error("source query no longer pins a parameter-set lane")
    for column in EXPECTED_COLUMNS:
        if column not in source["query"]:
            raise NASAExoplanetTask1Error(f"source query omitted {column}")

    network = config["network"]
    _strict(
        network,
        {"allowed_host", "maximum_response_bytes", "request_timeout_seconds", "user_agent"},
        "network config",
    )
    if (
        network["allowed_host"] != "exoplanetarchive.ipac.caltech.edu"
        or not 500_000 <= network["maximum_response_bytes"] <= 5_000_000
        or not 5 <= network["request_timeout_seconds"] <= 120
        or "InvariantNASAExoplanetTask1" not in network["user_agent"]
    ):
        raise NASAExoplanetTask1Error("network policy weakened")

    eligibility = config["eligibility"]
    _strict(
        eligibility,
        {
            "maximum_relative_uncertainty",
            "minimum_eligible_hosts",
            "minimum_eligible_rows",
            "require_all_six_uncertainty_bounds",
            "require_positive_values",
        },
        "eligibility config",
    )
    if eligibility["require_all_six_uncertainty_bounds"] is not True:
        raise NASAExoplanetTask1Error("uncertainty requirement weakened")
    if eligibility["require_positive_values"] is not True:
        raise NASAExoplanetTask1Error("positive-value requirement weakened")
    maximum_relative = float(eligibility["maximum_relative_uncertainty"])
    if not 0 < maximum_relative <= 0.5:
        raise NASAExoplanetTask1Error("relative-uncertainty limit weakened")

    split = config["split"]
    _strict(
        split,
        {
            "group_key",
            "holdout_fraction_denominator",
            "holdout_fraction_numerator",
            "order",
            "selection",
        },
        "split config",
    )
    if split != {
        "group_key": "hostname",
        "holdout_fraction_denominator": 5,
        "holdout_fraction_numerator": 1,
        "order": "ascending_by_host_maximum_discovery_year_then_hostname",
        "selection": "last_host_fraction_is_holdout",
    }:
        raise NASAExoplanetTask1Error("host-disjoint chronological split changed")

    discovery_config = config["discovery"]
    _strict(
        discovery_config,
        {
            "anonymous_columns",
            "candidate_budget_per_run",
            "exponent_bound",
            "new_strategy",
            "old_strategy",
            "random_seeds",
            "unit_rescaling_factors",
        },
        "discovery config",
    )
    if (
        discovery_config["anonymous_columns"] != ["x0", "x1", "x2"]
        or discovery_config["new_strategy"] != "new_occam"
        or discovery_config["old_strategy"] != "old_pairwise"
        or not 32 <= discovery_config["candidate_budget_per_run"] <= 10_000
        or not 2 <= discovery_config["exponent_bound"] <= 32
        or len(discovery_config["random_seeds"]) < 16
        or len(set(discovery_config["random_seeds"])) != len(discovery_config["random_seeds"])
    ):
        raise NASAExoplanetTask1Error("discovery comparison policy changed")

    classification = config["classification"]
    _strict(
        classification,
        {"gate_eligible", "reason", "status_if_all_performance_checks_pass"},
        "classification config",
    )
    if classification["gate_eligible"] is False:
        if "inspected" not in str(classification["reason"]).lower():
            raise NASAExoplanetTask1Error("exploratory classification omitted its contamination")
    elif classification["gate_eligible"] is True:
        query = source["query"]
        if (
            "default_flag=0" not in query
            or "pl_pubdate>='2020-01-01'" not in query
            or "order by hostname,pl_name,pl_refname" not in query
            or "untouched" not in str(classification["reason"]).lower()
        ):
            raise NASAExoplanetTask1Error("gate-eligible source lane is not the frozen fresh query")
    else:
        raise NASAExoplanetTask1Error("gate eligibility is not Boolean")

    evaluation = config["evaluation"]
    _strict(
        evaluation,
        {
            "minimum_holdout_within_1sigma_fraction",
            "minimum_holdout_within_2sigma_fraction",
            "require_exact_target_structure",
            "require_new_better_than_every_baseline",
            "require_new_better_than_old",
            "require_unit_rescaling_stability",
        },
        "evaluation config",
    )
    if (
        float(evaluation["minimum_holdout_within_1sigma_fraction"]) < 0.9
        or float(evaluation["minimum_holdout_within_2sigma_fraction"]) < 0.95
        or any(
            evaluation[key] is not True
            for key in (
                "require_exact_target_structure",
                "require_new_better_than_every_baseline",
                "require_new_better_than_old",
                "require_unit_rescaling_stability",
            )
        )
    ):
        raise NASAExoplanetTask1Error("evaluation gate weakened")

    outputs = config["outputs"]
    _strict(outputs, {"receipt", "sanitized_training_rows", "source_snapshot"}, "outputs")
    for relative in outputs.values():
        _resolve(root, relative)

    target = config["target_fixture"]
    _strict(target, {"normalized_sha256", "path"}, "target fixture config")
    target_path = _resolve(root, target["path"])
    if _normalized_sha256(target_path.read_bytes()) != target["normalized_sha256"]:
        raise NASAExoplanetTask1Error("target fixture content changed")
    return config


def build_source_uri(config: Mapping[str, Any]) -> str:
    source = config["source"]
    query = urllib.parse.urlencode({"query": source["query"], "format": "csv"})
    uri = f"{source['tap_base_uri']}?{query}"
    parsed = urllib.parse.urlparse(uri)
    if (
        parsed.scheme != "https"
        or parsed.hostname != config["network"]["allowed_host"]
        or parsed.username is not None
    ):
        raise NASAExoplanetTask1Error("source URI escaped the allowed HTTPS authority")
    return uri


def fetch_snapshot(config: Mapping[str, Any]) -> bytes:
    uri = build_source_uri(config)
    request = urllib.request.Request(
        uri,
        headers={"User-Agent": config["network"]["user_agent"]},
        method="GET",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config["network"]["request_timeout_seconds"]
        ) as response:
            raw = response.read(config["network"]["maximum_response_bytes"] + 1)
    except OSError as error:
        raise NASAExoplanetTask1Error("NASA archive retrieval failed") from error
    if len(raw) > config["network"]["maximum_response_bytes"]:
        raise NASAExoplanetTask1Error("NASA archive response exceeded the byte limit")
    return raw


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise NASAExoplanetTask1Error(f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise NASAExoplanetTask1Error(f"git {' '.join(arguments)} failed")
    return completed.stdout


def build_authorization(
    root: Path,
    config_path: str,
    *,
    frozen_at: str | None = None,
) -> dict[str, Any]:
    """Freeze a gate-eligible config after implementation commit and before row retrieval."""

    root = root.resolve()
    config = load_config(root, config_path)
    if config["classification"]["gate_eligible"] is not True:
        raise NASAExoplanetTask1Error("authorization is only valid for a gate-eligible config")
    bound_paths = [
        ".gitattributes",
        config_path,
        GENERATOR_PATH,
        SOURCE_PATH,
        TEST_PATH,
        config["target_fixture"]["path"],
        "pyproject.toml",
    ]
    tracked_changes = _git(root, "status", "--porcelain", "--", *bound_paths)
    if tracked_changes:
        raise NASAExoplanetTask1Error(
            "authorization requires all bound implementation files to be committed"
        )
    commit = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise NASAExoplanetTask1Error("git commit identity is invalid")
    return {
        "authorization_id": "nasa-exoplanet-task1-confirmation-authorization-001",
        "bound_files": {
            relative: _file_sha256(_resolve(root, relative)) for relative in bound_paths
        },
        "config_path": config_path,
        "config_sha256": canonical_sha256(config),
        "frozen_at": _utc(frozen_at),
        "frozen_git_commit": commit,
        "prior_access_declaration": {
            "aggregate_availability_counts_requested": True,
            "candidate_scores_observed": False,
            "holdout_performance_observed": False,
            "row_values_retrieved": False,
        },
        "schema_version": AUTHORIZATION_SCHEMA,
        "source_query_sha256": canonical_sha256(config["source"]["query"]),
    }


def validate_authorization(
    root: Path,
    authorization: Mapping[str, Any],
    config_path: str,
) -> dict[str, Any]:
    _strict(
        authorization,
        {
            "authorization_id",
            "bound_files",
            "config_path",
            "config_sha256",
            "frozen_at",
            "frozen_git_commit",
            "prior_access_declaration",
            "schema_version",
            "source_query_sha256",
        },
        "authorization",
    )
    if authorization["schema_version"] != AUTHORIZATION_SCHEMA:
        raise NASAExoplanetTask1Error("authorization schema changed")
    if authorization["config_path"] != config_path:
        raise NASAExoplanetTask1Error("authorization config path changed")
    config = load_config(root, config_path)
    if authorization["config_sha256"] != canonical_sha256(config):
        raise NASAExoplanetTask1Error("authorization config commitment changed")
    if authorization["source_query_sha256"] != canonical_sha256(config["source"]["query"]):
        raise NASAExoplanetTask1Error("authorization source query commitment changed")
    expected_prior_access = {
        "aggregate_availability_counts_requested": True,
        "candidate_scores_observed": False,
        "holdout_performance_observed": False,
        "row_values_retrieved": False,
    }
    if authorization["prior_access_declaration"] != expected_prior_access:
        raise NASAExoplanetTask1Error("authorization prior-access declaration changed")
    _utc_datetime(authorization["frozen_at"])
    commit = authorization["frozen_git_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise NASAExoplanetTask1Error("authorization git commit is invalid")
    if _git(root, "rev-parse", commit) != commit:
        raise NASAExoplanetTask1Error("authorization git commit is unavailable")
    bound_files = authorization["bound_files"]
    if not isinstance(bound_files, Mapping) or not bound_files:
        raise NASAExoplanetTask1Error("authorization bound-file inventory is missing")
    for relative, digest in bound_files.items():
        if not isinstance(relative, str) or digest != _file_sha256(_resolve(root, relative)):
            raise NASAExoplanetTask1Error("authorization bound file changed")
        if digest != _normalized_sha256(_git_bytes(root, "show", f"{commit}:{relative}")):
            raise NASAExoplanetTask1Error("authorization file was not frozen in its git commit")
    return dict(authorization)


def _number(value: str, label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise NASAExoplanetTask1Error(f"missing {label}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise NASAExoplanetTask1Error(f"invalid {label}") from error
    if not math.isfinite(parsed):
        raise NASAExoplanetTask1Error(f"non-finite {label}")
    return parsed


def parse_snapshot(raw: bytes, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        text = _normalized_bytes(raw).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise NASAExoplanetTask1Error("NASA archive response is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
        raise NASAExoplanetTask1Error("NASA archive CSV columns changed")
    maximum_relative = float(config["eligibility"]["maximum_relative_uncertainty"])
    eligible: list[dict[str, Any]] = []
    reasons: defaultdict[str, int] = defaultdict(int)
    seen_sources: set[tuple[str, str]] = set()
    for raw_row in reader:
        try:
            host = raw_row["hostname"].strip()
            planet = raw_row["pl_name"].strip()
            year = int(raw_row["disc_year"])
            values = [_number(raw_row[column], column) for column in VALUE_COLUMNS]
            uncertainties = [
                max(
                    abs(_number(raw_row[f"{column}err1"], f"{column}err1")),
                    abs(_number(raw_row[f"{column}err2"], f"{column}err2")),
                )
                for column in VALUE_COLUMNS
            ]
        except (NASAExoplanetTask1Error, ValueError):
            reasons["missing_or_invalid_required_value"] += 1
            continue
        source_identity = (planet, raw_row["pl_refname"].strip())
        if not host or not planet or source_identity in seen_sources:
            reasons["missing_or_duplicate_identity"] += 1
            continue
        if year < 1980 or year > 2100:
            reasons["invalid_discovery_year"] += 1
            continue
        if any(value <= 0 for value in values):
            reasons["non_positive_value"] += 1
            continue
        relative = [
            uncertainty / value
            for value, uncertainty in zip(values, uncertainties, strict=True)
        ]
        if any(value > maximum_relative for value in relative):
            reasons["relative_uncertainty_above_limit"] += 1
            continue
        seen_sources.add(source_identity)
        eligible.append(
            {
                "host": host,
                "planet": planet,
                "year": year,
                "values": values,
                "uncertainties": uncertainties,
            }
        )
    eligible.sort(key=lambda row: (row["host"], row["planet"]))
    if len(eligible) < config["eligibility"]["minimum_eligible_rows"]:
        raise NASAExoplanetTask1Error("eligible NASA row count fell below the declared minimum")
    hosts = {row["host"] for row in eligible}
    if len(hosts) < config["eligibility"]["minimum_eligible_hosts"]:
        raise NASAExoplanetTask1Error("eligible NASA host count fell below the declared minimum")
    return eligible, dict(sorted(reasons.items()))


def split_and_sanitize(
    eligible: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_host: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_host[row["host"]].append(row)
    ordered_hosts = sorted(
        by_host,
        key=lambda host: (max(int(row["year"]) for row in by_host[host]), host),
    )
    numerator = config["split"]["holdout_fraction_numerator"]
    denominator = config["split"]["holdout_fraction_denominator"]
    training_host_count = len(ordered_hosts) - math.ceil(len(ordered_hosts) * numerator / denominator)
    if training_host_count <= 0:
        raise NASAExoplanetTask1Error("split leaves no training hosts")
    training_hosts = set(ordered_hosts[:training_host_count])
    holdout_hosts = set(ordered_hosts[training_host_count:])
    if training_hosts & holdout_hosts:
        raise NASAExoplanetTask1Error("host leakage across the split")

    training: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for index, row in enumerate(eligible):
        sanitized = {
            "label": f"r{index:06d}",
            "uncertainties": [_float_text(value) for value in row["uncertainties"]],
            "values": [_float_text(value) for value in row["values"]],
        }
        (training if row["host"] in training_hosts else holdout).append(sanitized)
    split_summary = {
        "holdout_host_count": len(holdout_hosts),
        "holdout_maximum_discovery_year": max(
            int(row["year"]) for row in eligible if row["host"] in holdout_hosts
        ),
        "holdout_minimum_discovery_year": min(
            int(row["year"]) for row in eligible if row["host"] in holdout_hosts
        ),
        "holdout_row_count": len(holdout),
        "host_intersection_count": len(training_hosts & holdout_hosts),
        "training_host_count": len(training_hosts),
        "training_maximum_discovery_year": max(
            int(row["year"]) for row in eligible if row["host"] in training_hosts
        ),
        "training_minimum_discovery_year": min(
            int(row["year"]) for row in eligible if row["host"] in training_hosts
        ),
        "training_row_count": len(training),
    }
    return training, holdout, split_summary


def generator_leakage_audit(root: Path, training: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = _resolve(root, GENERATOR_PATH).read_text(encoding="utf-8").lower()
    words = set(re.findall(r"[a-z][a-z0-9_-]*", source))
    vocabulary_hits = sorted(words & FORBIDDEN_GENERATOR_WORDS)
    compact = re.sub(r"\s+", "", source)
    fingerprint_forms = ("[2,-3,1]", "(2,-3,1)", "2,-3,1")
    fingerprint_hits = sorted(form for form in fingerprint_forms if form in compact)
    training_text = json.dumps(training, sort_keys=True).lower()
    training_vocabulary_hits = sorted(
        token for token in FORBIDDEN_GENERATOR_WORDS if re.search(rf"\b{re.escape(token)}\b", training_text)
    )
    passed = not vocabulary_hits and not fingerprint_hits and not training_vocabulary_hits
    return {
        "forbidden_numeric_fingerprint_hits": fingerprint_hits,
        "forbidden_vocabulary_hits_in_generator": vocabulary_hits,
        "forbidden_vocabulary_hits_in_training_input": training_vocabulary_hits,
        "generator_path": GENERATOR_PATH,
        "passed": passed,
        "scope": (
            "The executable generic generator and serialized training input only. The broader "
            "repository contains prior classical-law fixtures and is not claimed exposure-free."
        ),
    }


def _scaled_rows(
    rows: Sequence[Mapping[str, Any]], factors: Sequence[float]
) -> list[dict[str, Any]]:
    return [
        {
            "label": row["label"],
            "values": [
                _float_text(float(value) * factor)
                for value, factor in zip(row["values"], factors, strict=True)
            ],
            "uncertainties": [
                _float_text(float(value) * factor)
                for value, factor in zip(row["uncertainties"], factors, strict=True)
            ],
        }
        for row in rows
    ]


def _baseline_metrics(
    training: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    train = np.log(np.array([[float(value) for value in row["values"]] for row in training]))
    test = np.log(np.array([[float(value) for value in row["values"]] for row in holdout]))
    train_y = train[:, 0]
    test_y = test[:, 0]

    def summarize(identifier: str, coefficients: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
        errors = np.abs(test_y - predictions)
        return {
            "baseline_id": identifier,
            "coefficients": [_float_text(float(value)) for value in coefficients],
            "holdout_median_absolute_response_log_error": _float_text(float(np.median(errors))),
            "holdout_p90_absolute_response_log_error": _float_text(
                float(np.quantile(errors, 0.9, method="higher"))
            ),
        }

    baselines: list[dict[str, Any]] = []
    constant = np.array([float(np.median(train_y))])
    baselines.append(
        summarize("constant_log_response", constant, np.repeat(constant[0], len(test_y)))
    )

    univariate: list[tuple[float, int, np.ndarray]] = []
    for column in (1, 2):
        design = np.column_stack((np.ones(len(train)), train[:, column]))
        coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
        train_error = float(np.median(np.abs(train_y - design @ coefficients)))
        univariate.append((train_error, column, coefficients))
    _, selected_column, selected_coefficients = min(univariate, key=lambda item: (item[0], item[1]))
    test_design = np.column_stack((np.ones(len(test)), test[:, selected_column]))
    baselines.append(
        summarize(
            f"best_univariate_log_linear_x{selected_column}",
            selected_coefficients,
            test_design @ selected_coefficients,
        )
    )

    multivariate_design = np.column_stack((np.ones(len(train)), train[:, 1:]))
    multivariate_coefficients, *_ = np.linalg.lstsq(multivariate_design, train_y, rcond=None)
    multivariate_test = np.column_stack((np.ones(len(test)), test[:, 1:]))
    baselines.append(
        summarize(
            "unconstrained_multivariate_log_linear",
            multivariate_coefficients,
            multivariate_test @ multivariate_coefficients,
        )
    )

    mean = train[:, 1:].mean(axis=0)
    scale = train[:, 1:].std(axis=0)
    if np.any(scale <= 0):
        raise NASAExoplanetTask1Error("quadratic baseline has a constant predictor")
    train_z = (train[:, 1:] - mean) / scale
    test_z = (test[:, 1:] - mean) / scale
    quadratic_design = np.column_stack(
        (
            np.ones(len(train_z)),
            train_z,
            train_z[:, 0] ** 2,
            train_z[:, 0] * train_z[:, 1],
            train_z[:, 1] ** 2,
        )
    )
    quadratic_coefficients, *_ = np.linalg.lstsq(quadratic_design, train_y, rcond=None)
    quadratic_test = np.column_stack(
        (
            np.ones(len(test_z)),
            test_z,
            test_z[:, 0] ** 2,
            test_z[:, 0] * test_z[:, 1],
            test_z[:, 1] ** 2,
        )
    )
    baselines.append(
        summarize(
            "unconstrained_quadratic_log_predictors",
            quadratic_coefficients,
            quadratic_test @ quadratic_coefficients,
        )
    )
    return baselines


def _load_target(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(root, config["target_fixture"]["path"])
    target = json.loads(path.read_text(encoding="utf-8"))
    _strict(
        target,
        {
            "canonical_primitive_exponents",
            "claims",
            "column_mapping",
            "data_dependency_warning",
            "equivalence_rule",
            "historical_interpretation",
            "schema_version",
            "target_id",
        },
        "target fixture",
    )
    if target["schema_version"] != TARGET_SCHEMA:
        raise NASAExoplanetTask1Error("target fixture schema changed")
    if target["claims"] != {
        "historically_novel": False,
        "independent_physical_confirmation": False,
        "known_result_calibration": True,
        "level5_eligible": False,
    }:
        raise NASAExoplanetTask1Error("target claim boundary changed")
    return target


def build_campaign(
    root: Path,
    raw: bytes,
    *,
    retrieved_at: str | None = None,
    config_path: str = CONFIG_PATH,
    authorization: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the pilot receipt and public training artifact from one immutable snapshot."""

    root = root.resolve()
    config = load_config(root, config_path)
    retrieval_time = _utc(retrieved_at)
    gate_eligible = config["classification"]["gate_eligible"]
    if gate_eligible:
        if authorization is None:
            raise NASAExoplanetTask1Error("gate-eligible run lacks a frozen authorization")
        checked_authorization = validate_authorization(root, authorization, config_path)
        if _utc_datetime(retrieval_time) <= _utc_datetime(checked_authorization["frozen_at"]):
            raise NASAExoplanetTask1Error("snapshot retrieval did not occur after authorization")
    elif authorization is not None:
        raise NASAExoplanetTask1Error("exploratory pilot must not carry gate authorization")
    else:
        checked_authorization = None
    eligible, exclusions = parse_snapshot(raw, config)
    training, holdout, split_summary = split_and_sanitize(eligible, config)
    leakage = generator_leakage_audit(root, training)
    if not leakage["passed"]:
        raise NASAExoplanetTask1Error("anonymous discovery input leaked target vocabulary or vector")

    search = config["discovery"]
    common = {
        "arity": len(search["anonymous_columns"]),
        "candidate_budget": search["candidate_budget_per_run"],
        "exponent_bound": search["exponent_bound"],
    }
    chronology: list[dict[str, Any]] = [
        {"event": "source_snapshot_hashed", "holdout_rows_exposed_to_generator": 0},
        {"event": "host_disjoint_split_committed", "holdout_rows_exposed_to_generator": 0},
        {"event": "anonymous_training_input_audited", "holdout_rows_exposed_to_generator": 0},
    ]
    try:
        new_search = discover(training, strategy=search["new_strategy"], **common)
        old_search = discover(training, strategy=search["old_strategy"], **common)
        random_searches = [
            discover(training, strategy="uniform_random", random_seed=seed, **common)
            for seed in search["random_seeds"]
        ]
        factors = [float(value) for value in search["unit_rescaling_factors"]]
        scaled_search = discover(
            _scaled_rows(training, factors), strategy=search["new_strategy"], **common
        )
    except AnonymousMonomialError as error:
        raise NASAExoplanetTask1Error("anonymous discovery failed") from error
    candidate_phase = {
        "new_search": new_search,
        "old_search": old_search,
        "random_searches": random_searches,
        "unit_rescaling_search": scaled_search,
    }
    candidate_phase_sha256 = canonical_sha256(candidate_phase)
    chronology.append(
        {
            "candidate_phase_sha256": candidate_phase_sha256,
            "event": "all_candidates_frozen",
            "holdout_rows_exposed_to_generator": 0,
        }
    )

    # Target meaning and holdout scoring begin only after every compared candidate is frozen.
    target = _load_target(root, config)
    chronology.append(
        {
            "event": "target_fixture_and_holdout_opened_for_scoring",
            "holdout_rows_exposed_to_generator": 0,
        }
    )
    new_holdout = score_frozen_candidate(holdout, new_search["best_candidate"])
    old_holdout = score_frozen_candidate(holdout, old_search["best_candidate"])
    random_holdouts = [
        {
            "best_exponents": search_receipt["best_candidate"]["exponents"],
            "holdout": score_frozen_candidate(holdout, search_receipt["best_candidate"]),
            "seed": search_receipt["random_seed"],
        }
        for search_receipt in random_searches
    ]
    baselines = _baseline_metrics(training, holdout)
    target_exponents = target["canonical_primitive_exponents"]
    expected_structure = new_search["best_candidate"]["exponents"] == target_exponents
    unit_stable = scaled_search["best_candidate"]["exponents"] == new_search["best_candidate"]["exponents"]
    new_error = float(new_holdout["median_absolute_response_log_error"])
    old_error = float(old_holdout["median_absolute_response_log_error"])
    baseline_errors = [
        float(row["holdout_median_absolute_response_log_error"]) for row in baselines
    ]
    random_better_or_equal = sum(
        float(row["holdout"]["median_absolute_response_log_error"]) <= new_error
        for row in random_holdouts
    )
    random_exact = sum(row["best_exponents"] == target_exponents for row in random_holdouts)

    evaluation = config["evaluation"]
    checks = {
        "candidate_frozen_before_holdout_scoring": chronology[3]["event"] == "all_candidates_frozen",
        "exact_target_structure": expected_structure,
        "host_disjoint_split": split_summary["host_intersection_count"] == 0,
        "new_better_than_every_baseline": new_error < min(baseline_errors),
        "new_better_than_old": new_error < old_error,
        "unit_rescaling_stability": unit_stable,
        "within_1sigma": float(new_holdout["within_1sigma_fraction"])
        >= float(evaluation["minimum_holdout_within_1sigma_fraction"]),
        "within_2sigma": float(new_holdout["within_2sigma_fraction"])
        >= float(evaluation["minimum_holdout_within_2sigma_fraction"]),
    }
    performance_passed = all(checks.values())
    if performance_passed and gate_eligible:
        decision = "PASS"
        observed_status = "GATE_PASS"
    elif performance_passed:
        decision = "BLOCKED"
        observed_status = config["classification"]["status_if_all_performance_checks_pass"]
    else:
        decision = "REJECT"
        observed_status = "PERFORMANCE_GATE_FAILED"

    training_artifact = {
        "anonymous_columns": search["anonymous_columns"],
        "campaign_id": config["campaign_id"],
        "claims": {
            "column_meanings_present": False,
            "holdout_rows_present": False,
            "target_formula_present": False,
        },
        "rows": training,
        "schema_version": TRAINING_SCHEMA,
    }
    receipt: dict[str, Any] = {
        "baselines": baselines,
        "authorization": checked_authorization,
        "campaign_id": config["campaign_id"],
        "candidate_phase": candidate_phase,
        "candidate_phase_sha256": candidate_phase_sha256,
        "checks": checks,
        "chronology": chronology,
        "claims": {
            "creative_method_established": False,
            "data_columns_are_independent_direct_measurements": False,
            "gate_eligible": gate_eligible,
            "historically_novel": False,
            "independent_physical_confirmation": False,
            "known_result_recovered": expected_structure,
            "level5_eligible": False,
            "llm_calls_made": 0,
            "real_external_catalog_snapshot_used": True,
            "task_1_completed": decision == "PASS",
        },
        "config_sha256": canonical_sha256(config),
        "config_path": config_path,
        "decision": decision,
        "evaluation": {
            "new_holdout": new_holdout,
            "old_holdout": old_holdout,
            "performance_checks_passed": performance_passed,
            "random_best_exact_target_count": random_exact,
            "random_better_or_equal_to_new_count": random_better_or_equal,
            "random_holdouts": random_holdouts,
            "random_replicates": len(random_holdouts),
        },
        "exclusions": exclusions,
        "leakage_audit": leakage,
        "implementation": {
            "generator_normalized_sha256": _file_sha256(_resolve(root, GENERATOR_PATH)),
            "generator_path": GENERATOR_PATH,
            "orchestrator_normalized_sha256": _file_sha256(_resolve(root, SOURCE_PATH)),
            "orchestrator_path": SOURCE_PATH,
            "test_normalized_sha256": _file_sha256(_resolve(root, TEST_PATH)),
            "test_path": TEST_PATH,
        },
        "observed_status": observed_status,
        "retrieved_at": retrieval_time,
        "schema_version": RECEIPT_SCHEMA,
        "scope": (
            "Exploratory recovery of a known three-column scaling from a real NASA catalog "
            "snapshot. The candidate generator sees sanitized training rows only. Because "
            "aggregate holdout performance was inspected during implementation and catalog "
            "parameters may be inferred or dependent, this receipt cannot unlock Task 2, count "
            "as Level 5, establish creative superiority, or confirm the physical law independently."
        ),
        "source": {
            "dependency_warning": config["source"]["dependency_warning"],
            "documentation_uri": config["source"]["documentation_uri"],
            "external_principal_id": config["source"]["external_principal_id"],
            "normalized_snapshot_sha256": _normalized_sha256(raw),
            "response_bytes": len(raw),
            "source_uri": build_source_uri(config),
        },
        "split": {
            **split_summary,
            "holdout_commitment_sha256": canonical_sha256(holdout),
            "split_policy": config["split"],
            "training_commitment_sha256": canonical_sha256(training),
        },
        "target_opening": {
            "canonical_primitive_exponents": target_exponents,
            "claims": target["claims"],
            "data_dependency_warning": target["data_dependency_warning"],
            "equivalence_rule": target["equivalence_rule"],
            "historical_interpretation": target["historical_interpretation"],
            "target_fixture_normalized_sha256": config["target_fixture"]["normalized_sha256"],
            "target_id": target["target_id"],
        },
        "training_artifact_sha256": canonical_sha256(training_artifact),
    }
    return receipt, training_artifact


def validate_campaign(
    root: Path,
    receipt: Mapping[str, Any],
    training_artifact: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    """Rebuild the complete receipt from its archived source snapshot."""

    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise NASAExoplanetTask1Error("receipt schema changed")
    if training_artifact.get("schema_version") != TRAINING_SCHEMA:
        raise NASAExoplanetTask1Error("training artifact schema changed")
    config_path = receipt.get("config_path")
    if not isinstance(config_path, str):
        raise NASAExoplanetTask1Error("receipt config path is missing")
    rebuilt_receipt, rebuilt_training = build_campaign(
        root,
        raw,
        retrieved_at=str(receipt.get("retrieved_at")),
        config_path=config_path,
        authorization=receipt.get("authorization"),
    )
    if rebuilt_training != training_artifact:
        raise NASAExoplanetTask1Error("training artifact does not replay")
    if rebuilt_receipt != receipt:
        raise NASAExoplanetTask1Error("campaign receipt does not replay")
    return dict(receipt)


def _write_bytes_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as error:
        raise NASAExoplanetTask1Error(f"refusing to overwrite immutable file: {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser(
        "authorization-template", help="freeze the gate-eligible implementation before retrieval"
    )
    authorize.add_argument("--root", type=Path, default=Path.cwd())
    authorize.add_argument("--config", required=True)
    authorize.add_argument("--frozen-at")
    authorize.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run", help="retrieve or ingest a snapshot and write the pilot")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--raw-csv", type=Path)
    run.add_argument("--retrieved-at")
    run.add_argument("--config", default=CONFIG_PATH)
    run.add_argument("--authorization", type=Path)
    validate = subparsers.add_parser("validate", help="replay the committed pilot")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--config", default=CONFIG_PATH)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if arguments.command == "authorization-template":
        authorization = build_authorization(
            root, arguments.config, frozen_at=arguments.frozen_at
        )
        write_immutable(arguments.output.resolve(), authorization)
        print(json.dumps(authorization, sort_keys=True, indent=2))
        return 0
    config = load_config(root, arguments.config)
    outputs = config["outputs"]
    source_snapshot = _resolve(root, outputs["source_snapshot"])
    receipt_path = _resolve(root, outputs["receipt"])
    training_path = _resolve(root, outputs["sanitized_training_rows"])

    if arguments.command == "run":
        gate_eligible = config["classification"]["gate_eligible"]
        if gate_eligible and arguments.raw_csv is not None:
            raise NASAExoplanetTask1Error(
                "gate-eligible run must retrieve its snapshot after authorization"
            )
        if gate_eligible and arguments.authorization is None:
            raise NASAExoplanetTask1Error("gate-eligible run requires --authorization")
        authorization = (
            json.loads(arguments.authorization.resolve().read_text(encoding="utf-8"))
            if arguments.authorization is not None
            else None
        )
        if arguments.raw_csv is None:
            raw = fetch_snapshot(config)
            retrieved_at = _utc(None)
        else:
            raw = arguments.raw_csv.resolve().read_bytes()
            retrieved_at = arguments.retrieved_at
        receipt, training_artifact = build_campaign(
            root,
            raw,
            retrieved_at=retrieved_at,
            config_path=arguments.config,
            authorization=authorization,
        )
        _write_bytes_immutable(source_snapshot, _normalized_bytes(raw))
        write_immutable(training_path, training_artifact)
        write_immutable(receipt_path, receipt)
        print(json.dumps(receipt, sort_keys=True, indent=2))
        return 0

    raw = source_snapshot.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    training_artifact = json.loads(training_path.read_text(encoding="utf-8"))
    validate_campaign(root, receipt, training_artifact, raw)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
