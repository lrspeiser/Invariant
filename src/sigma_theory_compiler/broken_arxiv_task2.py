"""Seal and stage the first fresh BrokenArXiv creative-falsification trial.

This module deliberately separates metadata-only source discovery from problem ingestion.
The authorization must be frozen before an eligible release exists.  Source checking never
downloads problem rows, and staging rejects reference answers and manual problem selection.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .claude_creativity_api import (
    ClaudeAPIConfig,
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeRole,
    Transport,
    urllib_transport,
)
from .core_credential import CredentialActivationError, activated_credential
from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "invariant-broken-arxiv-task2-config-4.0"
AUTHORIZATION_SCHEMA = "invariant-broken-arxiv-task2-authorization-4.0"
SOURCE_CHECK_SCHEMA = "invariant-broken-arxiv-task2-source-check-4.0"
STAGED_PROBLEM_SCHEMA = "invariant-broken-arxiv-task2-staged-problem-4.0"
RELEASE_PACKET_SCHEMA = "invariant-broken-arxiv-task2-projected-release-packet-1.0"
PUBLIC_SUBMISSIONS_SCHEMA = "invariant-broken-arxiv-task2-public-submissions-1.0"
GENERATION_RECEIPT_SCHEMA = "invariant-broken-arxiv-task2-generation-receipt-1.0"
PRIVATE_COORDINATOR_SCHEMA = "invariant-broken-arxiv-task2-private-coordinator-1.0"
EVALUATION_PACKET_SCHEMA = "invariant-broken-arxiv-task2-independent-evaluation-1.0"
ADJUDICATION_SCHEMA = "invariant-broken-arxiv-task2-adjudication-1.0"
REQUIRED_PYARROW_VERSION = "21.0.0"
REQUIRED_FSSPEC_VERSION = "2025.7.0"
DEFAULT_CONFIG_PATH = Path("configs/broken_arxiv_task2.json")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_MONTH = re.compile(r"(20[0-9]{2})-(0[1-9]|1[0-2])\Z")


class BrokenArxivTask2Error(ValueError):
    """The sealed Task 2 chronology, source, or content contract failed closed."""


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise BrokenArxivTask2Error(f"{label} keys changed")


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["content_sha256"] = canonical_sha256(body)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    if value.get("schema_version") != schema:
        raise BrokenArxivTask2Error(f"{label} schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise BrokenArxivTask2Error(f"{label} content seal changed")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BrokenArxivTask2Error(f"could not read JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise BrokenArxivTask2Error(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    # Every bound implementation file is text. Normalize checkout line endings so the
    # authorization replays identically on Windows and Linux CI runners.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _month_number(value: str) -> int:
    match = _MONTH.fullmatch(value)
    if match is None:
        raise BrokenArxivTask2Error(f"invalid release month: {value}")
    return int(match.group(1)) * 12 + int(match.group(2))


def _iso_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise BrokenArxivTask2Error(f"{label} is not an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise BrokenArxivTask2Error(f"{label} is not an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise BrokenArxivTask2Error(f"{label} has no timezone")
    return parsed.astimezone(UTC)


def load_config(root: Path, path: Path = DEFAULT_CONFIG_PATH) -> Mapping[str, Any]:
    config_path = path if path.is_absolute() else root / path
    config = _read_json(config_path)
    _strict_keys(
        config,
        {
            "adjudication",
            "implementation_paths",
            "ingestion",
            "pass_gate",
            "preauthorization_format_probe",
            "schema_version",
            "selection",
            "source",
            "supersessions",
            "task_id",
            "trial",
        },
        "Task 2 config",
    )
    if config["schema_version"] != CONFIG_SCHEMA:
        raise BrokenArxivTask2Error("Task 2 config schema changed")
    source = config["source"]
    selection = config["selection"]
    trial = config["trial"]
    adjudication = config["adjudication"]
    ingestion = config["ingestion"]
    supersessions = config["supersessions"]
    format_probe = config["preauthorization_format_probe"]
    gate = config["pass_gate"]
    if not isinstance(source, Mapping) or not isinstance(selection, Mapping):
        raise BrokenArxivTask2Error("Task 2 source or selection config is invalid")
    if not isinstance(supersessions, list) or len(supersessions) != 3:
        raise BrokenArxivTask2Error("Task 2 supersession evidence changed")
    for supersession in supersessions:
        if (
            not isinstance(supersession, Mapping)
            or set(supersession)
            != {
                "authorization_content_sha256",
                "eligible_problem_rows_opened",
                "ineligible_problem_rows_materialized",
                "reason",
                "reference_answers_opened",
                "source_check_content_sha256",
            }
            or _HASH.fullmatch(str(supersession["authorization_content_sha256"])) is None
            or _HASH.fullmatch(str(supersession["source_check_content_sha256"])) is None
            or supersession["eligible_problem_rows_opened"] != 0
            or isinstance(supersession["ineligible_problem_rows_materialized"], bool)
            or not isinstance(supersession["ineligible_problem_rows_materialized"], int)
            or supersession["ineligible_problem_rows_materialized"] < 0
            or supersession["reference_answers_opened"] != 0
        ):
            raise BrokenArxivTask2Error("Task 2 supersession evidence changed")
    if (
        not isinstance(format_probe, Mapping)
        or set(format_probe)
        != {
            "dataset_id",
            "materialized_columns",
            "packet_persisted",
            "problem_rows_materialized",
            "purpose",
            "reference_columns_materialized",
            "revision",
        }
        or format_probe["dataset_id"] != "MathArena/brokenarxiv-0626"
        or format_probe["materialized_columns"] != ["problem_idx", "problem"]
        or format_probe["problem_rows_materialized"] != 54
        or format_probe["reference_columns_materialized"] != []
        or format_probe["packet_persisted"] is not False
        or _COMMIT.fullmatch(str(format_probe["revision"])) is None
    ):
        raise BrokenArxivTask2Error("Task 2 preauthorization format-probe disclosure changed")
    if (
        not isinstance(ingestion, Mapping)
        or ingestion.get("required_projected_columns") != ["problem_idx", "problem"]
        or ingestion.get("projection_engine")
        != "pyarrow_parquet_column_projection_v1"
        or ingestion.get("required_pyarrow_version") != REQUIRED_PYARROW_VERSION
        or ingestion.get("required_fsspec_version") != REQUIRED_FSSPEC_VERSION
        or ingestion.get("manual_release_packet_allowed") is not False
        or ingestion.get("release_and_staged_packets_private_until_submissions_frozen") is not True
        or ingestion.get("maximum_problem_rows") != 1000
        or "original_problem" not in ingestion.get("forbidden_materialized_columns", [])
    ):
        raise BrokenArxivTask2Error("Task 2 projected ingestion contract changed")
    if _month_number(source["first_eligible_release_month"]) <= _month_number(
        source["last_release_visible_before_freeze"]
    ):
        raise BrokenArxivTask2Error("eligible release is not strictly after the frozen cutoff")
    if selection.get("algorithm") != "sha256_min_rank_v1":
        raise BrokenArxivTask2Error("Task 2 selector algorithm changed")
    if selection.get("select_count") != 1 or selection.get("manual_substitution_allowed") is not False:
        raise BrokenArxivTask2Error("Task 2 must select exactly one problem without substitution")
    if source.get("problem_content_may_be_read_during_source_check") is not False:
        raise BrokenArxivTask2Error("metadata-only source check was weakened")
    if source.get("reference_answers_may_be_read_before_submissions_freeze") is not False:
        raise BrokenArxivTask2Error("reference-answer blindness was weakened")
    if source.get("eligible_release_last_modified_must_follow_authorization") is not True:
        raise BrokenArxivTask2Error("post-authorization publication requirement was weakened")
    if (
        not isinstance(trial, Mapping)
        or trial.get("arms")
        != [
            "old_failure_first_llm",
            "creativity_first_llm",
            "matched_random_falsifier",
        ]
        or trial.get("candidate_slots_per_arm") != 12
        or len(trial.get("random_falsifier_families", [])) != 12
        or len(trial.get("creativity_role_schedule", [])) != 12
    ):
        raise BrokenArxivTask2Error("matched Task 2 arm allocation changed")
    if not isinstance(trial.get("claude"), Mapping) or set(trial["claude"]) != {
        "credential_env_var",
        "effort",
        "maximum_output_tokens_per_call",
        "maximum_total_tokens_per_arm",
        "model",
        "timeout_seconds",
    }:
        raise BrokenArxivTask2Error("Task 2 Claude resource contract changed")
    if not isinstance(trial.get("prompt_policies"), Mapping) or set(
        trial["prompt_policies"]
    ) != set(trial["arms"]):
        raise BrokenArxivTask2Error("Task 2 prompt policies changed")
    baseline = trial.get("old_system_baseline_commit")
    if not isinstance(baseline, str) or _COMMIT.fullmatch(baseline) is None:
        raise BrokenArxivTask2Error("Task 2 old-system baseline commit is invalid")
    if (
        not isinstance(adjudication, Mapping)
        or adjudication.get("all_submissions_must_be_scored") is not True
        or adjudication.get("arm_identity_blinded_until_scoring_complete") is not True
        or adjudication.get("reference_material_may_open_only_after_submission_seal") is not True
        or adjudication.get("minimum_named_human_reviewers_for_human_only_acceptance") != 2
        or set(adjudication.get("allowed_verifier_kinds", []))
        != {
            "exact_executable_verifier",
            "formal_kernel",
            "matharena_official_judge",
            "named_independent_human_review",
        }
    ):
        raise BrokenArxivTask2Error("Task 2 independent adjudication contract changed")
    required_true = {
        "false_as_written_decision_required",
        "exact_counterexample_or_independent_external_rejection_required",
        "smallest_failed_assumption_required",
        "nonvacuous_repaired_statement_required",
        "repaired_statement_bounded_proof_or_independent_external_acceptance_required",
        "old_new_random_equal_budget_required",
        "creative_arm_must_beat_random_on_cost_or_supply_distinct_valid_repair",
        "self_reported_origin_is_not_novelty_evidence",
        "finite_reference_nonoverlap_is_not_historical_novelty",
    }
    if not isinstance(gate, Mapping) or set(gate) != required_true or not all(gate.values()):
        raise BrokenArxivTask2Error("Task 2 pass gate changed")
    paths = config["implementation_paths"]
    if not isinstance(paths, list) or len(paths) != 4 or len(set(paths)) != 4:
        raise BrokenArxivTask2Error("Task 2 implementation path coverage changed")
    return config


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if _COMMIT.fullmatch(commit) is None:
        raise BrokenArxivTask2Error("could not bind Task 2 to a Git commit")
    return commit


def _git_resolve(root: Path, revision: str, label: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if _COMMIT.fullmatch(commit) is None:
        raise BrokenArxivTask2Error(f"could not resolve {label}")
    return commit


def build_authorization(
    root: Path,
    *,
    now: datetime | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root, config_path)
    paths = tuple(str(item) for item in config["implementation_paths"])
    bindings = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise BrokenArxivTask2Error(f"Task 2 implementation path is missing: {relative}")
        bindings.append({"path": relative, "sha256": _sha256_file(path)})
    frozen_at = now or datetime.now(UTC)
    if frozen_at.tzinfo is None:
        raise BrokenArxivTask2Error("authorization time must be timezone-aware")
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "task_id": config["task_id"],
        "frozen_at_utc": frozen_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(root),
        "old_system_baseline_commit": _git_resolve(
            root, config["trial"]["old_system_baseline_commit"], "old-system baseline commit"
        ),
        "supersessions": [dict(item) for item in config["supersessions"]],
        "config_sha256": canonical_sha256(config),
        "implementation_bindings": bindings,
        "source_cutoff": {
            "last_visible_release_month": config["source"]["last_release_visible_before_freeze"],
            "first_eligible_release_month": config["source"]["first_eligible_release_month"],
            "eligible_problem_rows_read": 0,
            "future_problem_rows_read": 0,
            "reference_answers_read": 0,
            "preauthorization_format_probe": dict(config["preauthorization_format_probe"]),
        },
        "selector_commitment": canonical_sha256(
            {
                "algorithm": config["selection"]["algorithm"],
                "seed": config["selection"]["seed"],
                "stable_problem_id_fields": config["selection"]["stable_problem_id_fields"],
                "statement_fields": config["selection"]["statement_fields"],
            }
        ),
        "trial_contract_sha256": canonical_sha256(
            {
                "adjudication": config["adjudication"],
                "pass_gate": config["pass_gate"],
                "trial": config["trial"],
            }
        ),
        "status": "FULL_HARNESS_FROZEN_WAITING_FOR_FIRST_ELIGIBLE_RELEASE",
    }
    authorization = _sealed(body)
    validate_authorization(authorization, root, config_path=config_path)
    return authorization


def validate_authorization(
    authorization: Mapping[str, Any],
    root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    _validate_seal(authorization, AUTHORIZATION_SCHEMA, "Task 2 authorization")
    config = load_config(root.resolve(), config_path)
    if (
        authorization.get("task_id") != config["task_id"]
        or authorization.get("config_sha256") != canonical_sha256(config)
        or authorization.get("status")
        != "FULL_HARNESS_FROZEN_WAITING_FOR_FIRST_ELIGIBLE_RELEASE"
        or authorization.get("source_cutoff", {}).get("eligible_problem_rows_read") != 0
        or authorization.get("source_cutoff", {}).get("future_problem_rows_read") != 0
        or authorization.get("source_cutoff", {}).get("reference_answers_read") != 0
        or authorization.get("source_cutoff", {}).get("preauthorization_format_probe")
        != config["preauthorization_format_probe"]
        or authorization.get("old_system_baseline_commit")
        != config["trial"]["old_system_baseline_commit"]
        or authorization.get("supersessions") != config["supersessions"]
    ):
        raise BrokenArxivTask2Error("Task 2 authorization contract changed")
    expected_bindings = [
        {"path": relative, "sha256": _sha256_file(root.resolve() / relative)}
        for relative in config["implementation_paths"]
    ]
    if authorization.get("implementation_bindings") != expected_bindings:
        raise BrokenArxivTask2Error("Task 2 implementation changed after authorization")
    if authorization.get("selector_commitment") != canonical_sha256(
        {
            "algorithm": config["selection"]["algorithm"],
            "seed": config["selection"]["seed"],
            "stable_problem_id_fields": config["selection"]["stable_problem_id_fields"],
            "statement_fields": config["selection"]["statement_fields"],
        }
    ):
        raise BrokenArxivTask2Error("Task 2 selector commitment changed")
    if authorization.get("trial_contract_sha256") != canonical_sha256(
        {
            "adjudication": config["adjudication"],
            "pass_gate": config["pass_gate"],
            "trial": config["trial"],
        }
    ):
        raise BrokenArxivTask2Error("Task 2 trial contract changed")
    if _COMMIT.fullmatch(str(authorization.get("git_commit"))) is None:
        raise BrokenArxivTask2Error("Task 2 authorization commit is invalid")


def fetch_catalog(config: Mapping[str, Any], *, timeout_seconds: int = 30) -> list[Mapping[str, Any]]:
    source = config["source"]
    query = urllib.parse.urlencode(
        {
            "author": source["catalog_author"],
            "search": source["catalog_search"],
            "limit": 100,
            "full": "true",
        }
    )
    request = urllib.request.Request(
        f"{source['catalog_endpoint']}?{query}",
        headers={"User-Agent": "Invariant-Task2-Metadata-Only/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read(5_000_001))
    except (OSError, json.JSONDecodeError) as error:
        raise BrokenArxivTask2Error("BrokenArXiv metadata catalog request failed") from error
    if not isinstance(payload, list):
        raise BrokenArxivTask2Error("BrokenArXiv metadata catalog response changed")
    return [item for item in payload if isinstance(item, Mapping)]


def _release_month(dataset_id: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.fullmatch(dataset_id)
    if match is None:
        return None
    return f"20{match.group(2)}-{match.group(1)}"


def evaluate_catalog(
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pattern = re.compile(config["source"]["dataset_id_pattern"])
    first_eligible = _month_number(config["source"]["first_eligible_release_month"])
    frozen_at = _iso_datetime(authorization["frozen_at_utc"], "authorization frozen_at_utc")
    releases = []
    for item in catalog:
        dataset_id = item.get("id")
        if not isinstance(dataset_id, str):
            continue
        month = _release_month(dataset_id, pattern)
        if month is None:
            continue
        releases.append(
            {
                "dataset_id": dataset_id,
                "last_modified": item.get("lastModified"),
                "release_month": month,
                "revision": item.get("sha"),
            }
        )
    releases.sort(key=lambda item: (item["release_month"], item["dataset_id"]))
    eligible = [
        item
        for item in releases
        if _month_number(item["release_month"]) >= first_eligible
        and _iso_datetime(item["last_modified"], "release last_modified") > frozen_at
    ]
    selected = eligible[0] if eligible else None
    body = {
        "schema_version": SOURCE_CHECK_SCHEMA,
        "task_id": config["task_id"],
        "authorization_content_sha256": authorization["content_sha256"],
        "catalog_query": {
            "author": config["source"]["catalog_author"],
            "problem_rows_read": 0,
            "reference_answers_read": 0,
            "search": config["source"]["catalog_search"],
        },
        "observed_release_metadata": releases,
        "selected_release": selected,
        "status": (
            "READY_FIRST_ELIGIBLE_RELEASE_METADATA_ONLY"
            if selected is not None
            else "BLOCKED_FUTURE_RELEASE_NOT_PUBLISHED"
        ),
    }
    return _sealed(body)


def validate_source_check(
    source_check: Mapping[str, Any],
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _validate_seal(source_check, SOURCE_CHECK_SCHEMA, "Task 2 source check")
    query = source_check.get("catalog_query", {})
    if (
        source_check.get("task_id") != config["task_id"]
        or source_check.get("authorization_content_sha256")
        != authorization["content_sha256"]
        or query.get("problem_rows_read") != 0
        or query.get("reference_answers_read") != 0
    ):
        raise BrokenArxivTask2Error("Task 2 metadata-only chronology changed")
    releases = source_check.get("observed_release_metadata")
    if not isinstance(releases, list) or releases != sorted(
        releases, key=lambda item: (item["release_month"], item["dataset_id"])
    ):
        raise BrokenArxivTask2Error("Task 2 release metadata is not canonical")
    eligible = [
        item
        for item in releases
        if _month_number(item["release_month"])
        >= _month_number(config["source"]["first_eligible_release_month"])
        and _iso_datetime(item["last_modified"], "release last_modified")
        > _iso_datetime(authorization["frozen_at_utc"], "authorization frozen_at_utc")
    ]
    expected = eligible[0] if eligible else None
    if source_check.get("selected_release") != expected:
        raise BrokenArxivTask2Error("Task 2 did not select the first eligible release")
    expected_status = (
        "READY_FIRST_ELIGIBLE_RELEASE_METADATA_ONLY"
        if expected is not None
        else "BLOCKED_FUTURE_RELEASE_NOT_PUBLISHED"
    )
    if source_check.get("status") != expected_status:
        raise BrokenArxivTask2Error("Task 2 source readiness status changed")


def _field(row: Mapping[str, Any], names: Sequence[str], label: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise BrokenArxivTask2Error(f"Task 2 row has no stable {label}")


MetadataFetcher = Callable[[str], Mapping[str, Any]]
TableProjector = Callable[[str, Sequence[str]], Sequence[Mapping[str, Any]]]


def _fetch_json_url(url: str) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Invariant-Task2-Projected-Ingestion/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read(5_000_001))
    except (OSError, json.JSONDecodeError) as error:
        raise BrokenArxivTask2Error("Task 2 dataset metadata request failed") from error
    if not isinstance(payload, Mapping):
        raise BrokenArxivTask2Error("Task 2 dataset metadata response changed")
    return payload


def _project_parquet_columns(
    url: str, columns: Sequence[str]
) -> Sequence[Mapping[str, Any]]:
    try:
        import fsspec
        import pyarrow
        from pyarrow import parquet
    except ImportError as error:
        raise BrokenArxivTask2Error(
            "projected ingestion requires the installed pyarrow and fsspec runtimes"
        ) from error
    if (
        pyarrow.__version__ != REQUIRED_PYARROW_VERSION
        or fsspec.__version__ != REQUIRED_FSSPEC_VERSION
    ):
        raise BrokenArxivTask2Error("Task 2 projected-ingestion runtime version changed")
    filesystem = fsspec.filesystem(
        "http", headers={"User-Agent": "Invariant-Task2-Projected-Ingestion/1.0"}
    )
    try:
        table = parquet.read_table(url, columns=list(columns), filesystem=filesystem)
    except Exception as error:
        raise BrokenArxivTask2Error("Task 2 projected Parquet read failed") from error
    if list(table.column_names) != list(columns):
        raise BrokenArxivTask2Error("Task 2 Parquet projection returned unexpected columns")
    return table.to_pylist()


def fetch_projected_release_packet(
    source_check: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    metadata_fetcher: MetadataFetcher = _fetch_json_url,
    table_projector: TableProjector = _project_parquet_columns,
) -> dict[str, Any]:
    """Fetch only problem IDs and false statements from the pinned official revision."""

    selected = source_check.get("selected_release")
    if not isinstance(selected, Mapping):
        raise BrokenArxivTask2Error("no eligible release exists for projected ingestion")
    ingestion = config["ingestion"]
    dataset_id = selected["dataset_id"]
    revision = selected["revision"]
    metadata_url = ingestion["dataset_metadata_endpoint_template"].format(
        dataset_id=urllib.parse.quote(dataset_id, safe="/")
    )
    metadata = metadata_fetcher(metadata_url)
    siblings = metadata.get("siblings")
    if (
        metadata.get("id") != dataset_id
        or metadata.get("sha") != revision
        or not isinstance(siblings, list)
    ):
        raise BrokenArxivTask2Error("official dataset revision no longer matches source check")
    paths = sorted(
        item["rfilename"]
        for item in siblings
        if isinstance(item, Mapping)
        and isinstance(item.get("rfilename"), str)
        and re.fullmatch(r"data/train-[0-9]{5}-of-[0-9]{5}\.parquet", item["rfilename"])
    )
    if not paths:
        raise BrokenArxivTask2Error("official dataset has no canonical train Parquet shards")
    columns = ingestion["required_projected_columns"]
    rows: list[dict[str, Any]] = []
    source_files = []
    for path in paths:
        file_url = ingestion["dataset_file_endpoint_template"].format(
            dataset_id=urllib.parse.quote(dataset_id, safe="/"),
            revision=urllib.parse.quote(revision, safe=""),
            path=urllib.parse.quote(path, safe="/"),
        )
        projected = table_projector(file_url, columns)
        if not isinstance(projected, Sequence):
            raise BrokenArxivTask2Error("Task 2 projection did not return rows")
        before = len(rows)
        for row in projected:
            if not isinstance(row, Mapping) or set(row) != set(columns):
                raise BrokenArxivTask2Error(
                    "Task 2 projection materialized missing or forbidden columns"
                )
            problem_idx = row["problem_idx"]
            problem = row["problem"]
            if (
                isinstance(problem_idx, bool)
                or not isinstance(problem_idx, (int, str))
                or not isinstance(problem, str)
                or not problem.strip()
            ):
                raise BrokenArxivTask2Error("Task 2 projected problem row is invalid")
            rows.append({"problem_id": str(problem_idx), "problem": problem.strip()})
        source_files.append(
            {
                "path": path,
                "projected_rows": len(rows) - before,
                "revision_pinned_url_sha256": hashlib.sha256(file_url.encode()).hexdigest(),
            }
        )
    if not rows or len(rows) > ingestion["maximum_problem_rows"]:
        raise BrokenArxivTask2Error("Task 2 projected row count is outside the frozen bound")
    if len({row["problem_id"] for row in rows}) != len(rows):
        raise BrokenArxivTask2Error("Task 2 projected problem IDs are not unique")
    body = {
        "schema_version": RELEASE_PACKET_SCHEMA,
        "dataset_id": dataset_id,
        "revision": revision,
        "items": sorted(rows, key=lambda row: row["problem_id"]),
        "projection": {
            "engine": ingestion["projection_engine"],
            "forbidden_columns_materialized": False,
            "materialized_columns": list(columns),
            "metadata_url_sha256": hashlib.sha256(metadata_url.encode()).hexdigest(),
            "runtime_versions": {
                "fsspec": ingestion["required_fsspec_version"],
                "pyarrow": ingestion["required_pyarrow_version"],
            },
            "source_files": source_files,
        },
        "status": "PASS_REVISION_PINNED_FALSE_STATEMENT_COLUMNS_ONLY",
    }
    return _sealed(body)


def validate_release_packet(
    release_packet: Mapping[str, Any],
    source_check: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _validate_seal(release_packet, RELEASE_PACKET_SCHEMA, "Task 2 projected release packet")
    selected = source_check.get("selected_release")
    projection = release_packet.get("projection", {})
    rows = release_packet.get("items")
    dataset_id = release_packet.get("dataset_id")
    revision = release_packet.get("revision")
    metadata_url = config["ingestion"]["dataset_metadata_endpoint_template"].format(
        dataset_id=urllib.parse.quote(str(dataset_id), safe="/")
    )
    if (
        not isinstance(selected, Mapping)
        or dataset_id != selected["dataset_id"]
        or revision != selected["revision"]
        or release_packet.get("status")
        != "PASS_REVISION_PINNED_FALSE_STATEMENT_COLUMNS_ONLY"
        or projection.get("engine") != config["ingestion"]["projection_engine"]
        or projection.get("materialized_columns")
        != config["ingestion"]["required_projected_columns"]
        or projection.get("forbidden_columns_materialized") is not False
        or set(projection)
        != {
            "engine",
            "forbidden_columns_materialized",
            "materialized_columns",
            "metadata_url_sha256",
            "runtime_versions",
            "source_files",
        }
        or projection.get("metadata_url_sha256")
        != hashlib.sha256(metadata_url.encode()).hexdigest()
        or projection.get("runtime_versions")
        != {
            "fsspec": config["ingestion"]["required_fsspec_version"],
            "pyarrow": config["ingestion"]["required_pyarrow_version"],
        }
        or not isinstance(projection.get("source_files"), list)
        or not projection["source_files"]
        or not isinstance(rows, list)
        or not rows
        or len(rows) > config["ingestion"]["maximum_problem_rows"]
        or rows != sorted(rows, key=lambda row: row["problem_id"])
        or len({row.get("problem_id") for row in rows if isinstance(row, Mapping)})
        != len(rows)
    ):
        raise BrokenArxivTask2Error("Task 2 projected release packet contract changed")
    projected_row_sum = 0
    paths = []
    for source_file in projection["source_files"]:
        if not isinstance(source_file, Mapping) or set(source_file) != {
            "path",
            "projected_rows",
            "revision_pinned_url_sha256",
        }:
            raise BrokenArxivTask2Error("Task 2 projected source-file evidence changed")
        path = source_file["path"]
        if not isinstance(path, str) or re.fullmatch(
            r"data/train-[0-9]{5}-of-[0-9]{5}\.parquet", path
        ) is None:
            raise BrokenArxivTask2Error("Task 2 projected source path changed")
        file_url = config["ingestion"]["dataset_file_endpoint_template"].format(
            dataset_id=urllib.parse.quote(str(dataset_id), safe="/"),
            revision=urllib.parse.quote(str(revision), safe=""),
            path=urllib.parse.quote(path, safe="/"),
        )
        projected_rows = source_file["projected_rows"]
        if (
            isinstance(projected_rows, bool)
            or not isinstance(projected_rows, int)
            or projected_rows < 1
            or source_file["revision_pinned_url_sha256"]
            != hashlib.sha256(file_url.encode()).hexdigest()
        ):
            raise BrokenArxivTask2Error("Task 2 projected source-file binding changed")
        projected_row_sum += projected_rows
        paths.append(path)
    if paths != sorted(set(paths)) or projected_row_sum != len(rows):
        raise BrokenArxivTask2Error("Task 2 projected shard coverage changed")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"problem", "problem_id"}
            or not isinstance(row["problem_id"], str)
            or not row["problem_id"]
            or not isinstance(row["problem"], str)
            or not row["problem"]
        ):
            raise BrokenArxivTask2Error("Task 2 projected release row changed")


def stage_problem(
    authorization: Mapping[str, Any],
    source_check: Mapping[str, Any],
    config: Mapping[str, Any],
    release_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one statement from a content packet without accepting reference material."""

    validate_source_check(source_check, authorization, config)
    selected_release = source_check.get("selected_release")
    if selected_release is None:
        raise BrokenArxivTask2Error("no future BrokenArXiv release is eligible yet")
    validate_release_packet(release_packet, source_check, config)
    if (
        release_packet["dataset_id"] != selected_release["dataset_id"]
        or release_packet["revision"] != selected_release["revision"]
    ):
        raise BrokenArxivTask2Error("release packet does not open the selected metadata")
    rows = release_packet["items"]
    if not isinstance(rows, list) or not rows:
        raise BrokenArxivTask2Error("release packet has no problems")
    forbidden = {"answer", "reference", "reference_answer", "solution", "judge"}
    normalized = []
    selection = config["selection"]
    for row in rows:
        if not isinstance(row, Mapping):
            raise BrokenArxivTask2Error("release packet contains a non-object problem")
        if forbidden.intersection(str(key).lower() for key in row):
            raise BrokenArxivTask2Error("reference or judge material was opened before submission")
        problem_id = _field(row, selection["stable_problem_id_fields"], "problem ID")
        statement = _field(row, selection["statement_fields"], "statement")
        normalized.append({"problem_id": problem_id, "statement": statement})
    if len({row["problem_id"] for row in normalized}) != len(normalized):
        raise BrokenArxivTask2Error("release packet contains duplicate problem IDs")
    normalized.sort(key=lambda row: row["problem_id"])
    seed = selection["seed"]
    ranked = sorted(
        normalized,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}\0{release_packet['dataset_id']}\0{release_packet['revision']}\0{row['problem_id']}".encode()
            ).hexdigest(),
            row["problem_id"],
        ),
    )
    selected = ranked[0]
    body = {
        "schema_version": STAGED_PROBLEM_SCHEMA,
        "task_id": config["task_id"],
        "authorization_content_sha256": authorization["content_sha256"],
        "source_check_content_sha256": source_check["content_sha256"],
        "release_binding": {
            "dataset_id": release_packet["dataset_id"],
            "problem_count": len(normalized),
            "problems_sha256": canonical_sha256(normalized),
            "release_packet_content_sha256": release_packet["content_sha256"],
            "revision": release_packet["revision"],
        },
        "selection": {
            "algorithm": selection["algorithm"],
            "manual_substitution": False,
            "problem_id": selected["problem_id"],
            "statement": selected["statement"],
            "statement_sha256": hashlib.sha256(selected["statement"].encode()).hexdigest(),
        },
        "blindness": {
            "problem_rows_processed": len(normalized),
            "reference_answers_read": 0,
            "submissions_frozen": False,
        },
        "status": "STAGED_ONE_FRESH_PROBLEM_SUBMISSIONS_NOT_YET_RUN",
    }
    return _sealed(body)


def validate_staged_problem(
    staged: Mapping[str, Any],
    authorization: Mapping[str, Any],
    source_check: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _validate_seal(staged, STAGED_PROBLEM_SCHEMA, "Task 2 staged problem")
    validate_source_check(source_check, authorization, config)
    selected_release = source_check.get("selected_release")
    selection = staged.get("selection", {})
    blindness = staged.get("blindness", {})
    release = staged.get("release_binding", {})
    statement = selection.get("statement")
    if (
        selected_release is None
        or staged.get("task_id") != config["task_id"]
        or staged.get("authorization_content_sha256") != authorization["content_sha256"]
        or staged.get("source_check_content_sha256") != source_check["content_sha256"]
        or release.get("dataset_id") != selected_release["dataset_id"]
        or release.get("revision") != selected_release["revision"]
        or _HASH.fullmatch(str(release.get("release_packet_content_sha256"))) is None
        or selection.get("algorithm") != config["selection"]["algorithm"]
        or selection.get("manual_substitution") is not False
        or not isinstance(statement, str)
        or selection.get("statement_sha256") != hashlib.sha256(statement.encode()).hexdigest()
        or blindness.get("reference_answers_read") != 0
        or blindness.get("submissions_frozen") is not False
        or staged.get("status") != "STAGED_ONE_FRESH_PROBLEM_SUBMISSIONS_NOT_YET_RUN"
    ):
        raise BrokenArxivTask2Error("Task 2 staged problem binding or blindness changed")


def _resource_budget(config: Mapping[str, Any]) -> dict[str, Any]:
    trial = config["trial"]
    claude = trial["claude"]
    return {
        "candidate_slots": trial["candidate_slots_per_arm"],
        "llm_calls": trial["candidate_slots_per_arm"],
        "llm_model": claude["model"],
        "maximum_output_tokens_per_call": claude["maximum_output_tokens_per_call"],
        "maximum_total_tokens_per_arm": claude["maximum_total_tokens_per_arm"],
        "statement_access": "identical_full_statement",
        "verifier_invocations_per_candidate": trial["verifier_invocations_per_candidate"],
        "wall_clock_milliseconds_per_candidate": trial[
            "wall_clock_milliseconds_per_candidate"
        ],
    }


def build_arm_specs(
    staged: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Build the three equal-budget prompt schedules before any model call occurs."""

    trial = config["trial"]
    slots = trial["candidate_slots_per_arm"]
    policies = trial["prompt_policies"]
    specs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in trial["arms"]}
    for slot in range(slots):
        specs["old_failure_first_llm"].append(
            {
                "arm": "old_failure_first_llm",
                "falsifier_family": None,
                "instruction": policies["old_failure_first_llm"]["instruction"],
                "role": ClaudeRole.PROPOSER.value,
                "slot_index": slot,
                "system": policies["old_failure_first_llm"]["system"],
            }
        )
    schedule = trial["creativity_role_schedule"]
    for slot, role_name in enumerate(schedule):
        try:
            role = ClaudeRole(role_name)
        except ValueError as error:
            raise BrokenArxivTask2Error("Task 2 creativity role schedule is invalid") from error
        if role is ClaudeRole.CRITIC:
            raise BrokenArxivTask2Error("Task 2 one-slot generation cannot schedule a critic")
        specs["creativity_first_llm"].append(
            {
                "arm": "creativity_first_llm",
                "falsifier_family": None,
                "instruction": policies["creativity_first_llm"]["instruction"],
                "role": role.value,
                "slot_index": slot,
                "system": policies["creativity_first_llm"]["system"],
            }
        )
    random_families = sorted(
        trial["random_falsifier_families"],
        key=lambda family: hashlib.sha256(
            (
                config["selection"]["seed"]
                + "\0"
                + staged["content_sha256"]
                + "\0"
                + family
            ).encode()
        ).hexdigest(),
    )
    template = policies["matched_random_falsifier"]["instruction_template"]
    for slot, family in enumerate(random_families):
        specs["matched_random_falsifier"].append(
            {
                "arm": "matched_random_falsifier",
                "falsifier_family": family,
                "instruction": template.format(family=family),
                "role": ClaudeRole.PROPOSER.value,
                "slot_index": slot,
                "system": policies["matched_random_falsifier"]["system"],
            }
        )
    if set(specs) != set(trial["arms"]) or {len(items) for items in specs.values()} != {
        slots
    }:
        raise BrokenArxivTask2Error("Task 2 arm schedules are not resource matched")
    return specs


def _client_config(config: Mapping[str, Any]) -> ClaudeAPIConfig:
    trial = config["trial"]
    claude = trial["claude"]
    return ClaudeAPIConfig(
        model=claude["model"],
        credential_env_var=claude["credential_env_var"],
        execution_enabled=True,
        maximum_calls=trial["candidate_slots_per_arm"],
        maximum_total_tokens=claude["maximum_total_tokens_per_arm"],
        maximum_output_tokens=claude["maximum_output_tokens_per_call"],
        timeout_seconds=claude["timeout_seconds"],
        effort=claude["effort"],
    )


def _blinded_submission_id(key: bytes, staged_sha256: str, arm: str, slot: int) -> str:
    message = f"{staged_sha256}:{arm}:{slot}".encode()
    return "submission." + hmac.new(key, message, hashlib.sha256).hexdigest()[:32]


def _validate_hypothesis(hypothesis: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    required = {
        "expression",
        "falsifiers",
        "family",
        "hypothesis_id",
        "invariants",
        "known_analogues",
        "llm_origin_assessment",
        "proof_plan",
        "rationale",
        "representation",
        "source_idea_domains",
        "synthesis_note",
    }
    if (
        not isinstance(hypothesis, Mapping)
        or set(hypothesis) != required
        or hypothesis.get("llm_origin_assessment")
        not in config["trial"]["required_origin_labels"]
        or not isinstance(hypothesis.get("expression"), str)
        or not hypothesis["expression"].strip()
        or not isinstance(hypothesis.get("proof_plan"), list)
    ):
        raise BrokenArxivTask2Error("Task 2 candidate hypothesis is invalid")


def compile_generation(
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    unblinding_key: bytes,
    credential_activation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile live call results into blinded submissions and a private arm map."""

    if len(unblinding_key) < 32:
        raise BrokenArxivTask2Error("Task 2 unblinding key is shorter than 256 bits")
    if (
        credential_activation.get("credential_persisted") is not False
        or credential_activation.get("credential_value_recorded") is not False
    ):
        raise BrokenArxivTask2Error("Task 2 credential evidence is unsafe")
    specs = build_arm_specs(staged, config)
    expected = {
        (spec["arm"], spec["slot_index"]): spec for items in specs.values() for spec in items
    }
    if len(candidates) != len(expected):
        raise BrokenArxivTask2Error("Task 2 did not fill every candidate slot")
    seen: set[tuple[str, int]] = set()
    public_rows = []
    mapping = []
    calls = []
    budget = _resource_budget(config)
    for candidate in candidates:
        _strict_keys(
            candidate,
            {"arm", "call", "falsifier_family", "hypothesis", "role", "slot_index"},
            "Task 2 generated candidate",
        )
        key = (candidate["arm"], candidate["slot_index"])
        spec = expected.get(key)
        if key in seen or spec is None:
            raise BrokenArxivTask2Error("Task 2 generated an unknown or duplicate slot")
        seen.add(key)
        if (
            candidate["role"] != spec["role"]
            or candidate["falsifier_family"] != spec["falsifier_family"]
        ):
            raise BrokenArxivTask2Error("Task 2 generated candidate crossed its frozen prompt slot")
        _validate_hypothesis(candidate["hypothesis"], config)
        call = candidate["call"]
        if (
            not isinstance(call, Mapping)
            or call.get("status") != ClaudeCallStatus.COMPLETED.value
            or call.get("evidence", {}).get("credential_persisted") is not False
        ):
            raise BrokenArxivTask2Error("Task 2 Claude call did not complete safely")
        submission_id = _blinded_submission_id(
            unblinding_key, staged["content_sha256"], candidate["arm"], candidate["slot_index"]
        )
        public_rows.append(
            {
                "hypothesis": dict(candidate["hypothesis"]),
                "resource_budget": budget,
                "submission_id": submission_id,
            }
        )
        mapping.append(
            {
                "arm": candidate["arm"],
                "call_content_sha256": canonical_sha256(call),
                "falsifier_family": candidate["falsifier_family"],
                "hypothesis_sha256": canonical_sha256(candidate["hypothesis"]),
                "role": candidate["role"],
                "slot_index": candidate["slot_index"],
                "submission_id": submission_id,
            }
        )
        calls.append(dict(call))
    if seen != set(expected):
        raise BrokenArxivTask2Error("Task 2 generated candidate coverage is incomplete")
    public_rows.sort(key=lambda row: row["submission_id"])
    mapping.sort(key=lambda row: row["submission_id"])
    statement = staged["selection"]["statement"]
    public = _sealed(
        {
            "schema_version": PUBLIC_SUBMISSIONS_SCHEMA,
            "task_id": config["task_id"],
            "staged_problem_content_sha256": staged["content_sha256"],
            "problem": {
                "problem_id": staged["selection"]["problem_id"],
                "statement": statement,
                "statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
            },
            "submissions": public_rows,
            "blindness": {
                "arm_identity_disclosed": False,
                "reference_answers_read": 0,
                "submissions_frozen": True,
            },
            "status": "SUBMISSIONS_FROZEN_AWAITING_INDEPENDENT_ADJUDICATION",
        }
    )
    call_root = canonical_sha256(sorted(calls, key=lambda call: canonical_sha256(call)))
    receipt = _sealed(
        {
            "schema_version": GENERATION_RECEIPT_SCHEMA,
            "task_id": config["task_id"],
            "config_sha256": canonical_sha256(config),
            "staged_problem_content_sha256": staged["content_sha256"],
            "public_submissions_content_sha256": public["content_sha256"],
            "credential_activation": dict(credential_activation),
            "generation": {
                "arm_count": len(config["trial"]["arms"]),
                "call_evidence_root_sha256": call_root,
                "calls": len(calls),
                "candidates": len(public_rows),
                "resource_budget_per_arm": budget,
                "unblinding_key_sha256": hashlib.sha256(unblinding_key).hexdigest(),
            },
            "claims": {
                "candidate_correctness_established": False,
                "creative_advantage_established": False,
                "historical_novelty_established": False,
                "task_2_completed": False,
            },
            "status": "PASS_GENERATION_ONLY_CORRECTNESS_GATE_CLOSED",
        }
    )
    coordinator = _sealed(
        {
            "schema_version": PRIVATE_COORDINATOR_SCHEMA,
            "task_id": config["task_id"],
            "public_submissions_content_sha256": public["content_sha256"],
            "generation_receipt_content_sha256": receipt["content_sha256"],
            "unblinding_key_hex": unblinding_key.hex(),
            "mapping": mapping,
            "calls": sorted(calls, key=lambda call: canonical_sha256(call)),
            "claims": {"safe_to_publish_before_independent_scoring": False},
        }
    )
    validate_generation(public, receipt, coordinator, staged, config)
    return public, receipt, coordinator


def validate_generation(
    public: Mapping[str, Any],
    receipt: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _validate_seal(public, PUBLIC_SUBMISSIONS_SCHEMA, "Task 2 public submissions")
    _validate_seal(receipt, GENERATION_RECEIPT_SCHEMA, "Task 2 generation receipt")
    _validate_seal(coordinator, PRIVATE_COORDINATOR_SCHEMA, "Task 2 private coordinator")
    submissions = public.get("submissions", [])
    mapping = coordinator.get("mapping", [])
    calls = coordinator.get("calls", [])
    slots = config["trial"]["candidate_slots_per_arm"]
    expected_count = slots * len(config["trial"]["arms"])
    ids = [row.get("submission_id") for row in submissions]
    if (
        len(submissions) != expected_count
        or len(set(ids)) != expected_count
        or ids != sorted(ids)
        or public.get("staged_problem_content_sha256") != staged["content_sha256"]
        or public.get("blindness")
        != {
            "arm_identity_disclosed": False,
            "reference_answers_read": 0,
            "submissions_frozen": True,
        }
        or public.get("status") != "SUBMISSIONS_FROZEN_AWAITING_INDEPENDENT_ADJUDICATION"
    ):
        raise BrokenArxivTask2Error("Task 2 public submission seal changed")
    if any(row.get("resource_budget") != _resource_budget(config) for row in submissions):
        raise BrokenArxivTask2Error("Task 2 public submissions are not resource matched")
    if (
        receipt.get("config_sha256") != canonical_sha256(config)
        or receipt.get("public_submissions_content_sha256") != public["content_sha256"]
        or receipt.get("claims")
        != {
            "candidate_correctness_established": False,
            "creative_advantage_established": False,
            "historical_novelty_established": False,
            "task_2_completed": False,
        }
        or receipt.get("generation", {}).get("calls") != expected_count
        or receipt.get("generation", {}).get("candidates") != expected_count
        or receipt.get("credential_activation", {}).get("credential_persisted") is not False
        or receipt.get("credential_activation", {}).get("credential_value_recorded") is not False
    ):
        raise BrokenArxivTask2Error("Task 2 generation receipt changed")
    try:
        key = bytes.fromhex(coordinator["unblinding_key_hex"])
    except (KeyError, ValueError) as error:
        raise BrokenArxivTask2Error("Task 2 private unblinding key is invalid") from error
    if (
        len(key) < 32
        or hashlib.sha256(key).hexdigest()
        != receipt["generation"]["unblinding_key_sha256"]
        or coordinator.get("public_submissions_content_sha256") != public["content_sha256"]
        or coordinator.get("generation_receipt_content_sha256") != receipt["content_sha256"]
        or coordinator.get("claims", {}).get("safe_to_publish_before_independent_scoring")
        is not False
        or len(mapping) != expected_count
        or not isinstance(calls, list)
        or len(calls) != expected_count
        or {row.get("submission_id") for row in mapping} != set(ids)
        or canonical_sha256(sorted(calls, key=lambda call: canonical_sha256(call)))
        != receipt["generation"]["call_evidence_root_sha256"]
    ):
        raise BrokenArxivTask2Error("Task 2 private coordinator binding changed")
    public_by_id = {row["submission_id"]: row for row in submissions}
    call_by_hash = {canonical_sha256(call): call for call in calls}
    if (
        len(call_by_hash) != expected_count
        or {row.get("call_content_sha256") for row in mapping} != set(call_by_hash)
    ):
        raise BrokenArxivTask2Error("Task 2 call evidence coverage changed")
    arm_counts = {arm: 0 for arm in config["trial"]["arms"]}
    for row in mapping:
        arm = row.get("arm")
        slot = row.get("slot_index")
        if arm not in arm_counts or not isinstance(slot, int) or not 0 <= slot < slots:
            raise BrokenArxivTask2Error("Task 2 private arm map is invalid")
        arm_counts[arm] += 1
        if (
            row["submission_id"]
            != _blinded_submission_id(key, staged["content_sha256"], arm, slot)
            or row.get("call_content_sha256") not in call_by_hash
            or row.get("hypothesis_sha256")
            != canonical_sha256(public_by_id[row["submission_id"]]["hypothesis"])
            or public_by_id[row["submission_id"]]["hypothesis"]
            not in call_by_hash[row["call_content_sha256"]].get("output", {}).get(
                "hypotheses", []
            )
        ):
            raise BrokenArxivTask2Error("Task 2 blinded submission ID changed")
    if set(arm_counts.values()) != {slots}:
        raise BrokenArxivTask2Error("Task 2 private arm allocation is not matched")


def run_generation(
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
    unblinding_key: bytes,
    credential_file: Path | None = None,
    transport: Transport = urllib_transport,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    specs = build_arm_specs(staged, config)
    clients = {
        arm: ClaudeCreativityClient(_client_config(config), transport)
        for arm in config["trial"]["arms"]
    }
    public_payload = {
        "benchmark_kind": "fresh_broken_arxiv_false_statement",
        "problem_id": staged["selection"]["problem_id"],
        "source_dataset": staged["release_binding"]["dataset_id"],
        "statement": staged["selection"]["statement"],
    }
    benchmark_id = "task2." + staged["selection"]["statement_sha256"][:24]
    ordered_specs = sorted(
        [spec for items in specs.values() for spec in items],
        key=lambda spec: (
            spec["slot_index"],
            hashlib.sha256(
                f"{staged['content_sha256']}:{spec['slot_index']}:{spec['arm']}".encode()
            ).hexdigest(),
        ),
    )
    environment = None
    if credential_file is not None:
        environment = dict(os.environ)
        environment[config["trial"]["claude"]["credential_env_var"]] = ""
        environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())
    candidates = []
    try:
        with activated_credential(
            project_root=root.resolve(),
            env_var=config["trial"]["claude"]["credential_env_var"],
            environment=environment,
        ) as activation:
            for spec in ordered_specs:
                result = clients[spec["arm"]].run(
                    ClaudeRole(spec["role"]),
                    benchmark_id,
                    public_payload,
                    instruction_override=spec["instruction"],
                    system_override=spec["system"],
                    hypothesis_slots=1,
                )
                if (
                    result.status is not ClaudeCallStatus.COMPLETED
                    or result.output is None
                    or len(result.output.hypotheses) != 1
                ):
                    raise BrokenArxivTask2Error("Task 2 Claude candidate slot did not complete")
                candidates.append(
                    {
                        "arm": spec["arm"],
                        "call": result.to_dict(),
                        "falsifier_family": spec["falsifier_family"],
                        "hypothesis": result.output.hypotheses[0].to_dict(),
                        "role": spec["role"],
                        "slot_index": spec["slot_index"],
                    }
                )
            activation_evidence = activation.to_evidence()
    except CredentialActivationError as error:
        raise BrokenArxivTask2Error(str(error)) from error
    return compile_generation(
        staged,
        config,
        candidates,
        unblinding_key=unblinding_key,
        credential_activation=activation_evidence,
    )


def seal_evaluation_packet(body: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed({"schema_version": EVALUATION_PACKET_SCHEMA, **dict(body)})


def _valid_evaluation(evaluation: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    exact_or_external = (
        evaluation["exact_counterexample_valid"]
        or evaluation["independent_external_rejection_valid"]
    )
    return bool(
        evaluation["false_as_written"]
        and exact_or_external
        and evaluation["smallest_failed_assumption_valid"]
        and evaluation["repair_nonvacuous_valid"]
        and evaluation["repair_proof_or_external_acceptance_valid"]
        and 1
        <= evaluation["verifier_invocations"]
        <= config["trial"]["verifier_invocations_per_candidate"]
    )


def validate_evaluation_packet(
    packet: Mapping[str, Any], public: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _validate_seal(packet, EVALUATION_PACKET_SCHEMA, "Task 2 independent evaluation")
    _strict_keys(
        packet,
        {
            "content_sha256",
            "evaluations",
            "evaluator",
            "public_submissions_content_sha256",
            "reference_material_opened_after_submissions_sealed",
            "schema_version",
            "task_id",
        },
        "Task 2 independent evaluation packet",
    )
    evaluator = packet["evaluator"]
    _strict_keys(
        evaluator,
        {
            "evidence_uri",
            "counterexample_canonicalizer",
            "independent_from_generator",
            "name",
            "named_human_reviewers",
            "organization",
            "proof_graph_canonicalizer",
            "signed_artifact_sha256",
            "verifier_kind",
        },
        "Task 2 evaluator",
    )
    verifier_kind = evaluator["verifier_kind"]
    reviewers = evaluator["named_human_reviewers"]
    if (
        packet["task_id"] != config["task_id"]
        or packet["public_submissions_content_sha256"] != public["content_sha256"]
        or packet["reference_material_opened_after_submissions_sealed"] is not True
        or evaluator["independent_from_generator"] is not True
        or verifier_kind not in config["adjudication"]["allowed_verifier_kinds"]
        or not isinstance(evaluator["name"], str)
        or not evaluator["name"].strip()
        or not isinstance(evaluator["counterexample_canonicalizer"], str)
        or not evaluator["counterexample_canonicalizer"].strip()
        or not isinstance(evaluator["proof_graph_canonicalizer"], str)
        or not evaluator["proof_graph_canonicalizer"].strip()
        or _HASH.fullmatch(str(evaluator["signed_artifact_sha256"])) is None
        or not isinstance(reviewers, list)
        or len(set(reviewers)) != len(reviewers)
        or (
            verifier_kind == "named_independent_human_review"
            and len(reviewers)
            < config["adjudication"]["minimum_named_human_reviewers_for_human_only_acceptance"]
        )
    ):
        raise BrokenArxivTask2Error("Task 2 evaluator is not independently admissible")
    evaluations = packet["evaluations"]
    expected_ids = {row["submission_id"] for row in public["submissions"]}
    if not isinstance(evaluations, list) or {
        row.get("submission_id") for row in evaluations if isinstance(row, Mapping)
    } != expected_ids or len(evaluations) != len(expected_ids):
        raise BrokenArxivTask2Error("Task 2 did not independently score every submission")
    required = {
        "canonical_counterexample_sha256",
        "canonical_repair_graph_sha256",
        "counterexample_or_rejection",
        "exact_counterexample_valid",
        "failed_assumption",
        "false_as_written",
        "independent_external_rejection_valid",
        "notes",
        "repair_nonvacuous_valid",
        "repair_proof_or_external_acceptance_valid",
        "repaired_statement",
        "smallest_failed_assumption_valid",
        "submission_id",
        "verifier_invocations",
    }
    for evaluation in evaluations:
        _strict_keys(evaluation, required, "Task 2 candidate evaluation")
        booleans = (
            "exact_counterexample_valid",
            "false_as_written",
            "independent_external_rejection_valid",
            "repair_nonvacuous_valid",
            "repair_proof_or_external_acceptance_valid",
            "smallest_failed_assumption_valid",
        )
        if any(not isinstance(evaluation[key], bool) for key in booleans):
            raise BrokenArxivTask2Error("Task 2 evaluation verdict is not boolean")
        invocations = evaluation["verifier_invocations"]
        if isinstance(invocations, bool) or not isinstance(invocations, int) or invocations < 1:
            raise BrokenArxivTask2Error("Task 2 verifier invocation count is invalid")
        if _valid_evaluation(evaluation, config) and (
            not str(evaluation["counterexample_or_rejection"]).strip()
            or not str(evaluation["failed_assumption"]).strip()
            or not str(evaluation["repaired_statement"]).strip()
            or _HASH.fullmatch(str(evaluation["canonical_repair_graph_sha256"])) is None
            or (
                evaluation["exact_counterexample_valid"]
                and _HASH.fullmatch(str(evaluation["canonical_counterexample_sha256"])) is None
            )
        ):
            raise BrokenArxivTask2Error("Task 2 valid evaluation lacks exact evidence")


def _adjudication_body(
    public: Mapping[str, Any],
    receipt: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
    evaluation_packet: Mapping[str, Any],
) -> dict[str, Any]:
    validate_generation(public, receipt, coordinator, staged, config)
    validate_evaluation_packet(evaluation_packet, public, config)
    mapping = {row["submission_id"]: row for row in coordinator["mapping"]}
    evaluations = {
        row["submission_id"]: row for row in evaluation_packet["evaluations"]
    }
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in config["trial"]["arms"]:
        rows = []
        for submission_id, private in mapping.items():
            if private["arm"] != arm:
                continue
            evaluation = evaluations[submission_id]
            valid = _valid_evaluation(evaluation, config)
            rows.append(
                {
                    "canonical_repair_graph_sha256": evaluation[
                        "canonical_repair_graph_sha256"
                    ],
                    "decisive_valid_result": valid,
                    "search_cost_candidate_slots": private["slot_index"] + 1,
                    "submission_id": submission_id,
                    "verifier_invocations": evaluation["verifier_invocations"],
                }
            )
        rows.sort(key=lambda row: row["search_cost_candidate_slots"])
        valid_rows = [row for row in rows if row["decisive_valid_result"]]
        per_arm[arm] = {
            "decisive_valid_results": len(valid_rows),
            "minimum_search_cost_candidate_slots": (
                min(row["search_cost_candidate_slots"] for row in valid_rows)
                if valid_rows
                else None
            ),
            "valid_repair_graph_sha256": sorted(
                {row["canonical_repair_graph_sha256"] for row in valid_rows}
            ),
            "results": rows,
        }
    creative = per_arm["creativity_first_llm"]
    old = per_arm["old_failure_first_llm"]
    random_arm = per_arm["matched_random_falsifier"]
    creative_cost = creative["minimum_search_cost_candidate_slots"]
    old_cost = old["minimum_search_cost_candidate_slots"] or 13
    random_cost = random_arm["minimum_search_cost_candidate_slots"] or 13
    lower_cost = creative_cost is not None and creative_cost < old_cost and creative_cost < random_cost
    control_repairs = set(old["valid_repair_graph_sha256"]) | set(
        random_arm["valid_repair_graph_sha256"]
    )
    distinct_valid_repair = bool(
        set(creative["valid_repair_graph_sha256"]) - control_repairs
    )
    creative_has_valid = creative["decisive_valid_results"] > 0
    advantage = creative_has_valid and (lower_cost or distinct_valid_repair)
    passed = creative_has_valid and advantage
    body = {
        "schema_version": ADJUDICATION_SCHEMA,
        "task_id": config["task_id"],
        "source_bindings": {
            "evaluation_packet_content_sha256": evaluation_packet["content_sha256"],
            "generation_receipt_content_sha256": receipt["content_sha256"],
            "public_submissions_content_sha256": public["content_sha256"],
            "staged_problem_content_sha256": staged["content_sha256"],
        },
        "evaluator": dict(evaluation_packet["evaluator"]),
        "arm_results": per_arm,
        "comparison": {
            "creative_has_decisive_valid_result": creative_has_valid,
            "creative_lower_search_cost_than_both_controls": lower_cost,
            "creative_distinct_valid_repair_missed_by_both_controls": distinct_valid_repair,
            "equal_resource_budgets": True,
            "creative_advantage_gate": advantage,
        },
        "decision": "PASS" if passed else "REJECT",
        "claims": {
            "candidate_correctness_established_by_independent_evaluator": creative_has_valid,
            "creative_method_advantage_established_on_this_problem": advantage,
            "historical_novelty_established": False,
            "self_reported_origin_is_novelty_evidence": False,
            "task_2_completed": passed,
        },
        "status": "GATE_PASS" if passed else "PERFORMANCE_OR_CORRECTNESS_GATE_FAILED",
    }
    return body


def build_adjudication(
    public: Mapping[str, Any],
    receipt: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
    evaluation_packet: Mapping[str, Any],
) -> dict[str, Any]:
    result = _sealed(
        _adjudication_body(public, receipt, coordinator, staged, config, evaluation_packet)
    )
    validate_adjudication(
        result, public, receipt, coordinator, staged, config, evaluation_packet
    )
    return result


def validate_adjudication(
    adjudication: Mapping[str, Any],
    public: Mapping[str, Any],
    receipt: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    staged: Mapping[str, Any],
    config: Mapping[str, Any],
    evaluation_packet: Mapping[str, Any],
) -> None:
    _validate_seal(adjudication, ADJUDICATION_SCHEMA, "Task 2 adjudication")
    expected = _sealed(
        _adjudication_body(public, receipt, coordinator, staged, config, evaluation_packet)
    )
    if adjudication != expected:
        raise BrokenArxivTask2Error("Task 2 adjudication replay changed")
    bindings = adjudication.get("source_bindings", {})
    if (
        bindings.get("evaluation_packet_content_sha256")
        != evaluation_packet["content_sha256"]
        or bindings.get("generation_receipt_content_sha256") != receipt["content_sha256"]
        or bindings.get("public_submissions_content_sha256") != public["content_sha256"]
        or bindings.get("staged_problem_content_sha256") != staged["content_sha256"]
        or adjudication.get("claims", {}).get("historical_novelty_established") is not False
        or adjudication.get("claims", {}).get("self_reported_origin_is_novelty_evidence")
        is not False
        or adjudication.get("comparison", {}).get("equal_resource_budgets") is not True
        or (adjudication.get("decision") == "PASS")
        is not adjudication.get("claims", {}).get("task_2_completed")
    ):
        raise BrokenArxivTask2Error("Task 2 adjudication binding or claim gate changed")


def _private_output_path(root: Path, path: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to((root / "work").resolve())
    except ValueError as error:
        raise BrokenArxivTask2Error(
            "Task 2 private coordinator must remain under the ignored work directory"
        ) from error
    return resolved


def _add_chain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-check", type=Path, required=True)
    parser.add_argument("--staged-problem", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--root", type=Path, default=Path.cwd())
    authorize.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-authorization")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--authorization", type=Path, required=True)
    check = subparsers.add_parser("check-source")
    check.add_argument("--root", type=Path, default=Path.cwd())
    check.add_argument("--authorization", type=Path, required=True)
    check.add_argument("--catalog", type=Path)
    check.add_argument("--output", type=Path, required=True)
    fetch_release = subparsers.add_parser("fetch-release")
    fetch_release.add_argument("--root", type=Path, default=Path.cwd())
    fetch_release.add_argument("--authorization", type=Path, required=True)
    fetch_release.add_argument("--source-check", type=Path, required=True)
    fetch_release.add_argument("--output", type=Path, required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--root", type=Path, default=Path.cwd())
    stage.add_argument("--authorization", type=Path, required=True)
    stage.add_argument("--source-check", type=Path, required=True)
    stage.add_argument("--release-packet", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    validate_staged = subparsers.add_parser("validate-staged")
    _add_chain_arguments(validate_staged)
    generate = subparsers.add_parser("run-generation")
    _add_chain_arguments(generate)
    generate.add_argument("--credential-file", type=Path)
    generate.add_argument("--public-output", type=Path, required=True)
    generate.add_argument("--receipt-output", type=Path, required=True)
    generate.add_argument("--coordinator-output", type=Path, required=True)
    validate_generated = subparsers.add_parser("validate-generation")
    _add_chain_arguments(validate_generated)
    validate_generated.add_argument("--public-submissions", type=Path, required=True)
    validate_generated.add_argument("--generation-receipt", type=Path, required=True)
    validate_generated.add_argument("--coordinator", type=Path, required=True)
    adjudicate = subparsers.add_parser("adjudicate")
    _add_chain_arguments(adjudicate)
    adjudicate.add_argument("--public-submissions", type=Path, required=True)
    adjudicate.add_argument("--generation-receipt", type=Path, required=True)
    adjudicate.add_argument("--coordinator", type=Path, required=True)
    adjudicate.add_argument("--evaluation-packet", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = load_config(root)
    if args.command == "authorize":
        result = build_authorization(root)
        _write_json(args.output, result)
    else:
        authorization = _read_json(args.authorization)
        validate_authorization(authorization, root)
        if args.command == "validate-authorization":
            result = authorization
        elif args.command == "check-source":
            catalog = (
                list(_read_json(args.catalog)["datasets"])
                if args.catalog is not None
                else fetch_catalog(config)
            )
            result = evaluate_catalog(authorization, config, catalog)
            validate_source_check(result, authorization, config)
            _write_json(args.output, result)
        elif args.command == "fetch-release":
            source_check = _read_json(args.source_check)
            validate_source_check(source_check, authorization, config)
            result = fetch_projected_release_packet(source_check, config)
            validate_release_packet(result, source_check, config)
            _write_json(_private_output_path(root, args.output), result)
        elif args.command == "stage":
            source_check = _read_json(args.source_check)
            packet = _read_json(args.release_packet)
            result = stage_problem(authorization, source_check, config, packet)
            validate_staged_problem(result, authorization, source_check, config)
            _write_json(_private_output_path(root, args.output), result)
        else:
            source_check = _read_json(args.source_check)
            staged_problem = _read_json(args.staged_problem)
            validate_staged_problem(staged_problem, authorization, source_check, config)
            if args.command == "validate-staged":
                result = staged_problem
            elif args.command == "run-generation":
                coordinator_path = _private_output_path(root, args.coordinator_output)
                if coordinator_path.exists():
                    raise BrokenArxivTask2Error(
                        "Task 2 private coordinator already exists; refusing to change the blind map"
                    )
                public, receipt, coordinator_record = run_generation(
                    staged_problem,
                    config,
                    root=root,
                    unblinding_key=secrets.token_bytes(32),
                    credential_file=args.credential_file,
                )
                _write_json(args.public_output, public)
                _write_json(args.receipt_output, receipt)
                _write_json(coordinator_path, coordinator_record)
                result = receipt
            else:
                public = _read_json(args.public_submissions)
                receipt = _read_json(args.generation_receipt)
                coordinator_record = _read_json(args.coordinator)
                validate_generation(
                    public, receipt, coordinator_record, staged_problem, config
                )
                if args.command == "validate-generation":
                    result = receipt
                else:
                    evaluation = _read_json(args.evaluation_packet)
                    result = build_adjudication(
                        public,
                        receipt,
                        coordinator_record,
                        staged_problem,
                        config,
                        evaluation,
                    )
                    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "content_sha256": result["content_sha256"],
                "schema_version": result["schema_version"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
