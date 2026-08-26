"""Seal and stage the first fresh BrokenArXiv creative-falsification trial.

This module deliberately separates metadata-only source discovery from problem ingestion.
The authorization must be frozen before an eligible release exists.  Source checking never
downloads problem rows, and staging rejects reference answers and manual problem selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "invariant-broken-arxiv-task2-config-1.0"
AUTHORIZATION_SCHEMA = "invariant-broken-arxiv-task2-authorization-1.0"
SOURCE_CHECK_SCHEMA = "invariant-broken-arxiv-task2-source-check-1.0"
STAGED_PROBLEM_SCHEMA = "invariant-broken-arxiv-task2-staged-problem-1.0"
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


def load_config(root: Path, path: Path = DEFAULT_CONFIG_PATH) -> Mapping[str, Any]:
    config_path = path if path.is_absolute() else root / path
    config = _read_json(config_path)
    _strict_keys(
        config,
        {
            "implementation_paths",
            "pass_gate",
            "schema_version",
            "selection",
            "source",
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
    gate = config["pass_gate"]
    if not isinstance(source, Mapping) or not isinstance(selection, Mapping):
        raise BrokenArxivTask2Error("Task 2 source or selection config is invalid")
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
    if (
        not isinstance(trial, Mapping)
        or trial.get("arms") != ["creativity_first_llm", "matched_random_falsifier"]
        or trial.get("candidate_slots_per_arm") != 12
        or len(trial.get("random_falsifier_families", [])) != 12
    ):
        raise BrokenArxivTask2Error("matched Task 2 arm allocation changed")
    required_true = {
        "false_as_written_decision_required",
        "exact_counterexample_or_independent_external_rejection_required",
        "smallest_failed_assumption_required",
        "nonvacuous_repaired_statement_required",
        "repaired_statement_bounded_proof_or_independent_external_acceptance_required",
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
        "config_sha256": canonical_sha256(config),
        "implementation_bindings": bindings,
        "source_cutoff": {
            "last_visible_release_month": config["source"]["last_release_visible_before_freeze"],
            "first_eligible_release_month": config["source"]["first_eligible_release_month"],
            "problem_rows_read": 0,
            "reference_answers_read": 0,
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
            {"pass_gate": config["pass_gate"], "trial": config["trial"]}
        ),
        "status": "FROZEN_WAITING_FOR_FIRST_ELIGIBLE_RELEASE",
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
        or authorization.get("status") != "FROZEN_WAITING_FOR_FIRST_ELIGIBLE_RELEASE"
        or authorization.get("source_cutoff", {}).get("problem_rows_read") != 0
        or authorization.get("source_cutoff", {}).get("reference_answers_read") != 0
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
        {"pass_gate": config["pass_gate"], "trial": config["trial"]}
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
    eligible = [item for item in releases if _month_number(item["release_month"]) >= first_eligible]
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
    _strict_keys(release_packet, {"dataset_id", "items", "revision"}, "release packet")
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
    stage = subparsers.add_parser("stage")
    stage.add_argument("--root", type=Path, default=Path.cwd())
    stage.add_argument("--authorization", type=Path, required=True)
    stage.add_argument("--source-check", type=Path, required=True)
    stage.add_argument("--release-packet", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = load_config(root)
    if args.command == "authorize":
        result = build_authorization(root)
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
        else:
            source_check = _read_json(args.source_check)
            packet = _read_json(args.release_packet)
            result = stage_problem(authorization, source_check, config, packet)
    if args.command != "validate-authorization":
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
