"""Versioned Task-1 confirmation with statistically calibrated coverage gates.

Version 1 remains immutable and REJECTED.  This module reuses its frozen ingestion, anonymous
candidate generation, comparators, and baseline implementations while changing two things before
opening a different source lane:

* one-/two-sigma coverage floors are calibrated around the Gaussian reference coverages rather
  than requiring 90% inside a one-sigma interval;
* a p90 standardized-residual cap prevents the relaxed one-sigma floor from hiding a broad tail.

The 2015--2019 alternate-reference values may be retrieved only after a Git- and hash-bound
authorization.  A PASS is a real-catalog known-law calibration, never novelty or independent
physical confirmation, because archive parameters may be inferred or mutually dependent.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import nasa_exoplanet_task1 as v1
from .anonymous_monomial_discovery import AnonymousMonomialError, discover, score_frozen_candidate
from .sigma_core import canonical_sha256
from .tolerance_aware_fitting import write_immutable

CONFIG_SCHEMA = "invariant-nasa-exoplanet-task1-config-2.0"
AUTHORIZATION_SCHEMA = "invariant-nasa-exoplanet-task1-authorization-2.0"
RECEIPT_SCHEMA = "invariant-nasa-exoplanet-task1-receipt-2.0"
CONFIG_PATH = "configs/nasa_exoplanet_task1_confirmation_v2.json"
SOURCE_PATH = "src/sigma_theory_compiler/nasa_exoplanet_task1_v2.py"
TEST_PATH = "tests/test_nasa_exoplanet_task1_v2.py"


class NASAExoplanetTask1V2Error(ValueError):
    """Raised when the fresh-source or revised-coverage protocol changes."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NASAExoplanetTask1V2Error(f"{label} keys changed")


def _resolve(root: Path, relative: str) -> Path:
    try:
        return v1._resolve(root, relative)
    except v1.NASAExoplanetTask1Error as error:
        raise NASAExoplanetTask1V2Error(str(error)) from error


def _file_sha256(path: Path) -> str:
    return v1._file_sha256(path)


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
        "v2 config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"] != "nasa-exoplanet-task1-confirmation-002"
    ):
        raise NASAExoplanetTask1V2Error("v2 config identity changed")

    classification = config["classification"]
    _strict(
        classification,
        {"gate_eligible", "reason", "status_if_all_performance_checks_pass"},
        "classification",
    )
    if (
        classification["gate_eligible"] is not True
        or classification["status_if_all_performance_checks_pass"] != "GATE_PASS"
        or "untouched" not in classification["reason"].lower()
    ):
        raise NASAExoplanetTask1V2Error("v2 gate classification changed")

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
        "source",
    )
    query = source["query"]
    required_query_markers = (
        "default_flag=0",
        "pl_pubdate>='2015-01-01'",
        "pl_pubdate<'2020-01-01'",
        "pl_orbpererr1 is not null",
        "pl_orbpererr2 is not null",
        "pl_orbsmaxerr1 is not null",
        "pl_orbsmaxerr2 is not null",
        "st_masserr1 is not null",
        "st_masserr2 is not null",
        "order by hostname,pl_name,pl_refname",
    )
    if (
        source["external_principal_id"] != "external.nasa-exoplanet-archive"
        or source["table"] != "ps"
        or any(marker not in query for marker in required_query_markers)
        or any(column not in query for column in v1.EXPECTED_COLUMNS)
    ):
        raise NASAExoplanetTask1V2Error("fresh v2 source query changed")

    network = config["network"]
    _strict(
        network,
        {"allowed_host", "maximum_response_bytes", "request_timeout_seconds", "user_agent"},
        "network",
    )
    if (
        network["allowed_host"] != "exoplanetarchive.ipac.caltech.edu"
        or not 500_000 <= network["maximum_response_bytes"] <= 5_000_000
        or not 5 <= network["request_timeout_seconds"] <= 120
        or network["user_agent"] != "InvariantNASAExoplanetTask1V2/2.0"
    ):
        raise NASAExoplanetTask1V2Error("v2 network policy changed")

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
        "eligibility",
    )
    if (
        float(eligibility["maximum_relative_uncertainty"]) != 0.5
        or eligibility["minimum_eligible_hosts"] != 700
        or eligibility["minimum_eligible_rows"] != 1000
        or eligibility["require_all_six_uncertainty_bounds"] is not True
        or eligibility["require_positive_values"] is not True
    ):
        raise NASAExoplanetTask1V2Error("v2 eligibility changed")

    if config["split"] != {
        "group_key": "hostname",
        "holdout_fraction_denominator": 5,
        "holdout_fraction_numerator": 1,
        "order": "ascending_by_host_maximum_discovery_year_then_hostname",
        "selection": "last_host_fraction_is_holdout",
    }:
        raise NASAExoplanetTask1V2Error("v2 split changed")

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
        "discovery",
    )
    if (
        discovery_config["anonymous_columns"] != ["x0", "x1", "x2"]
        or discovery_config["candidate_budget_per_run"] != 256
        or discovery_config["exponent_bound"] != 12
        or discovery_config["new_strategy"] != "new_occam"
        or discovery_config["old_strategy"] != "old_pairwise"
        or len(discovery_config["random_seeds"]) != 32
        or len(set(discovery_config["random_seeds"])) != 32
        or discovery_config["unit_rescaling_factors"] != ["7", "11", "13"]
    ):
        raise NASAExoplanetTask1V2Error("v2 discovery budget changed")

    evaluation = config["evaluation"]
    _strict(
        evaluation,
        {
            "maximum_p90_standardized_residual",
            "minimum_empirical_1sigma_coverage",
            "minimum_empirical_2sigma_coverage",
            "reference_gaussian_1sigma_coverage",
            "reference_gaussian_2sigma_coverage",
            "require_exact_target_structure",
            "require_new_better_than_every_baseline",
            "require_new_better_than_old",
            "require_unit_rescaling_stability",
            "revision_reason",
        },
        "evaluation",
    )
    if (
        evaluation["minimum_empirical_1sigma_coverage"] != "0.6"
        or evaluation["minimum_empirical_2sigma_coverage"] != "0.9"
        or evaluation["maximum_p90_standardized_residual"] != "2.0"
        or evaluation["reference_gaussian_1sigma_coverage"] != "0.682689492137086"
        or evaluation["reference_gaussian_2sigma_coverage"] != "0.954499736103642"
        or "Version 1 remains REJECT" not in evaluation["revision_reason"]
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
        raise NASAExoplanetTask1V2Error("v2 coverage policy changed")

    target = config["target_fixture"]
    _strict(target, {"normalized_sha256", "path"}, "target fixture")
    if _file_sha256(_resolve(root, target["path"])) != target["normalized_sha256"]:
        raise NASAExoplanetTask1V2Error("target fixture changed")
    outputs = config["outputs"]
    _strict(outputs, {"receipt", "sanitized_training_rows", "source_snapshot"}, "outputs")
    for relative in outputs.values():
        _resolve(root, relative)
    return config


def build_source_uri(config: Mapping[str, Any]) -> str:
    try:
        return v1.build_source_uri(config)
    except v1.NASAExoplanetTask1Error as error:
        raise NASAExoplanetTask1V2Error(str(error)) from error


def fetch_snapshot(config: Mapping[str, Any]) -> bytes:
    try:
        return v1.fetch_snapshot(config)
    except v1.NASAExoplanetTask1Error as error:
        raise NASAExoplanetTask1V2Error(str(error)) from error


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise NASAExoplanetTask1V2Error(f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise NASAExoplanetTask1V2Error(f"git {' '.join(arguments)} failed")
    return completed.stdout


def _bound_paths(config: Mapping[str, Any], config_path: str) -> list[str]:
    return [
        ".gitattributes",
        config_path,
        v1.GENERATOR_PATH,
        v1.SOURCE_PATH,
        SOURCE_PATH,
        TEST_PATH,
        config["target_fixture"]["path"],
        "pyproject.toml",
    ]


def build_authorization(
    root: Path, config_path: str = CONFIG_PATH, *, frozen_at: str | None = None
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root, config_path)
    bound_paths = _bound_paths(config, config_path)
    if _git(root, "status", "--porcelain", "--", *bound_paths):
        raise NASAExoplanetTask1V2Error("authorization requires committed bound files")
    commit = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise NASAExoplanetTask1V2Error("authorization commit is invalid")
    return {
        "authorization_id": "nasa-exoplanet-task1-confirmation-authorization-002",
        "bound_files": {
            relative: _file_sha256(_resolve(root, relative)) for relative in bound_paths
        },
        "config_path": config_path,
        "config_sha256": canonical_sha256(config),
        "frozen_at": v1._utc(frozen_at),
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
    root: Path, authorization: Mapping[str, Any], config_path: str = CONFIG_PATH
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
    config = load_config(root, config_path)
    if (
        authorization["schema_version"] != AUTHORIZATION_SCHEMA
        or authorization["config_path"] != config_path
        or authorization["config_sha256"] != canonical_sha256(config)
        or authorization["source_query_sha256"] != canonical_sha256(config["source"]["query"])
    ):
        raise NASAExoplanetTask1V2Error("authorization commitment changed")
    if authorization["prior_access_declaration"] != {
        "aggregate_availability_counts_requested": True,
        "candidate_scores_observed": False,
        "holdout_performance_observed": False,
        "row_values_retrieved": False,
    }:
        raise NASAExoplanetTask1V2Error("authorization access declaration changed")
    try:
        v1._utc_datetime(authorization["frozen_at"])
    except v1.NASAExoplanetTask1Error as error:
        raise NASAExoplanetTask1V2Error(str(error)) from error
    commit = authorization["frozen_git_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise NASAExoplanetTask1V2Error("authorization commit is invalid")
    if _git(root, "rev-parse", commit) != commit:
        raise NASAExoplanetTask1V2Error("authorization commit is unavailable")
    bound_files = authorization["bound_files"]
    if set(bound_files) != set(_bound_paths(config, config_path)):
        raise NASAExoplanetTask1V2Error("authorization file inventory changed")
    for relative, digest in bound_files.items():
        if digest != _file_sha256(_resolve(root, relative)):
            raise NASAExoplanetTask1V2Error("authorization bound file changed")
        if digest != v1._normalized_sha256(_git_bytes(root, "show", f"{commit}:{relative}")):
            raise NASAExoplanetTask1V2Error("authorization file was not committed at freeze")
    return dict(authorization)


def build_campaign(
    root: Path,
    raw: bytes,
    *,
    authorization: Mapping[str, Any],
    retrieved_at: str,
    config_path: str = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config = load_config(root, config_path)
    checked_authorization = validate_authorization(root, authorization, config_path)
    retrieval_time = v1._utc(retrieved_at)
    if v1._utc_datetime(retrieval_time) <= v1._utc_datetime(checked_authorization["frozen_at"]):
        raise NASAExoplanetTask1V2Error("snapshot predates the frozen authorization")
    try:
        eligible, exclusions = v1.parse_snapshot(raw, config)
        training, holdout, split_summary = v1.split_and_sanitize(eligible, config)
        leakage = v1.generator_leakage_audit(root, training)
    except v1.NASAExoplanetTask1Error as error:
        raise NASAExoplanetTask1V2Error(str(error)) from error
    if not leakage["passed"]:
        raise NASAExoplanetTask1V2Error("anonymous discovery input leaked target information")

    search = config["discovery"]
    common = {
        "arity": len(search["anonymous_columns"]),
        "candidate_budget": search["candidate_budget_per_run"],
        "exponent_bound": search["exponent_bound"],
    }
    chronology: list[dict[str, Any]] = [
        {"event": "authorization_verified", "holdout_rows_exposed_to_generator": 0},
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
            v1._scaled_rows(training, factors), strategy=search["new_strategy"], **common
        )
    except AnonymousMonomialError as error:
        raise NASAExoplanetTask1V2Error("anonymous discovery failed") from error
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

    try:
        target = v1._load_target(root, config)
    except v1.NASAExoplanetTask1Error as error:
        raise NASAExoplanetTask1V2Error(str(error)) from error
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
            "best_exponents": row["best_candidate"]["exponents"],
            "holdout": score_frozen_candidate(holdout, row["best_candidate"]),
            "seed": row["random_seed"],
        }
        for row in random_searches
    ]
    try:
        baselines = v1._baseline_metrics(training, holdout)
    except v1.NASAExoplanetTask1Error as error:
        raise NASAExoplanetTask1V2Error(str(error)) from error
    target_exponents = target["canonical_primitive_exponents"]
    new_error = float(new_holdout["median_absolute_response_log_error"])
    old_error = float(old_holdout["median_absolute_response_log_error"])
    baseline_errors = [
        float(row["holdout_median_absolute_response_log_error"]) for row in baselines
    ]
    evaluation = config["evaluation"]
    checks = {
        "candidate_frozen_before_holdout_scoring": chronology[-2]["event"]
        == "all_candidates_frozen",
        "empirical_1sigma_coverage": float(new_holdout["within_1sigma_fraction"])
        >= float(evaluation["minimum_empirical_1sigma_coverage"]),
        "empirical_2sigma_coverage": float(new_holdout["within_2sigma_fraction"])
        >= float(evaluation["minimum_empirical_2sigma_coverage"]),
        "exact_target_structure": new_search["best_candidate"]["exponents"]
        == target_exponents,
        "host_disjoint_split": split_summary["host_intersection_count"] == 0,
        "new_better_than_every_baseline": new_error < min(baseline_errors),
        "new_better_than_old": new_error < old_error,
        "p90_standardized_residual": float(new_holdout["p90_standardized_residual"])
        <= float(evaluation["maximum_p90_standardized_residual"]),
        "unit_rescaling_stability": scaled_search["best_candidate"]["exponents"]
        == new_search["best_candidate"]["exponents"],
    }
    passed = all(checks.values())
    decision = "PASS" if passed else "REJECT"
    observed_status = "GATE_PASS" if passed else "PERFORMANCE_GATE_FAILED"
    random_better_or_equal = sum(
        float(row["holdout"]["median_absolute_response_log_error"]) <= new_error
        for row in random_holdouts
    )
    random_exact = sum(row["best_exponents"] == target_exponents for row in random_holdouts)

    training_artifact = {
        "anonymous_columns": search["anonymous_columns"],
        "campaign_id": config["campaign_id"],
        "claims": {
            "column_meanings_present": False,
            "holdout_rows_present": False,
            "target_formula_present": False,
        },
        "rows": training,
        "schema_version": v1.TRAINING_SCHEMA,
    }
    receipt: dict[str, Any] = {
        "authorization": checked_authorization,
        "baselines": baselines,
        "campaign_id": config["campaign_id"],
        "candidate_phase": candidate_phase,
        "candidate_phase_sha256": candidate_phase_sha256,
        "checks": checks,
        "chronology": chronology,
        "claims": {
            "creative_method_established": False,
            "data_columns_are_independent_direct_measurements": False,
            "gate_eligible": True,
            "historically_novel": False,
            "independent_physical_confirmation": False,
            "known_result_recovered": checks["exact_target_structure"],
            "level5_eligible": False,
            "llm_calls_made": 0,
            "real_external_catalog_snapshot_used": True,
            "task_1_completed": passed,
        },
        "config_path": config_path,
        "config_sha256": canonical_sha256(config),
        "decision": decision,
        "evaluation": {
            "new_holdout": new_holdout,
            "old_holdout": old_holdout,
            "performance_checks_passed": passed,
            "random_best_exact_target_count": random_exact,
            "random_better_or_equal_to_new_count": random_better_or_equal,
            "random_holdouts": random_holdouts,
            "random_replicates": len(random_holdouts),
        },
        "exclusions": exclusions,
        "implementation": {
            "generator_normalized_sha256": _file_sha256(_resolve(root, v1.GENERATOR_PATH)),
            "generator_path": v1.GENERATOR_PATH,
            "shared_v1_normalized_sha256": _file_sha256(_resolve(root, v1.SOURCE_PATH)),
            "shared_v1_path": v1.SOURCE_PATH,
            "v2_normalized_sha256": _file_sha256(_resolve(root, SOURCE_PATH)),
            "v2_path": SOURCE_PATH,
            "v2_test_normalized_sha256": _file_sha256(_resolve(root, TEST_PATH)),
            "v2_test_path": TEST_PATH,
        },
        "leakage_audit": leakage,
        "observed_status": observed_status,
        "retrieved_at": retrieval_time,
        "schema_version": RECEIPT_SCHEMA,
        "scope": (
            "Gate-eligible recovery of a known three-column scaling from a fresh NASA catalog "
            "lane under revised, predeclared empirical coverage gates. PASS completes Task 1 as "
            "an engineering calibration only. It does not establish creative superiority, "
            "historical novelty, or independent physical confirmation because catalog parameters "
            "may be inferred, dependent, or repeated across literature references."
        ),
        "source": {
            "dependency_warning": config["source"]["dependency_warning"],
            "documentation_uri": config["source"]["documentation_uri"],
            "external_principal_id": config["source"]["external_principal_id"],
            "normalized_snapshot_sha256": v1._normalized_sha256(raw),
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
        "uncertainty_policy": evaluation,
    }
    return receipt, training_artifact


def validate_campaign(
    root: Path,
    receipt: Mapping[str, Any],
    training_artifact: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise NASAExoplanetTask1V2Error("v2 receipt schema changed")
    if training_artifact.get("schema_version") != v1.TRAINING_SCHEMA:
        raise NASAExoplanetTask1V2Error("v2 training schema changed")
    rebuilt_receipt, rebuilt_training = build_campaign(
        root,
        raw,
        authorization=receipt["authorization"],
        retrieved_at=receipt["retrieved_at"],
        config_path=receipt["config_path"],
    )
    if rebuilt_training != training_artifact:
        raise NASAExoplanetTask1V2Error("v2 training artifact does not replay")
    if rebuilt_receipt != receipt:
        raise NASAExoplanetTask1V2Error("v2 receipt does not replay")
    return dict(receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorization-template")
    authorize.add_argument("--root", type=Path, default=Path.cwd())
    authorize.add_argument("--config", default=CONFIG_PATH)
    authorize.add_argument("--frozen-at")
    authorize.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--config", default=CONFIG_PATH)
    run.add_argument("--authorization", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--config", default=CONFIG_PATH)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if arguments.command == "authorization-template":
        value = build_authorization(root, arguments.config, frozen_at=arguments.frozen_at)
        write_immutable(arguments.output.resolve(), value)
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0

    config = load_config(root, arguments.config)
    outputs = config["outputs"]
    receipt_path = _resolve(root, outputs["receipt"])
    training_path = _resolve(root, outputs["sanitized_training_rows"])
    snapshot_path = _resolve(root, outputs["source_snapshot"])
    if arguments.command == "run":
        authorization = json.loads(
            arguments.authorization.resolve().read_text(encoding="utf-8")
        )
        raw = fetch_snapshot(config)
        retrieved_at = v1._utc(None)
        receipt, training_artifact = build_campaign(
            root,
            raw,
            authorization=authorization,
            retrieved_at=retrieved_at,
            config_path=arguments.config,
        )
        try:
            v1._write_bytes_immutable(snapshot_path, v1._normalized_bytes(raw))
        except v1.NASAExoplanetTask1Error as error:
            raise NASAExoplanetTask1V2Error(str(error)) from error
        write_immutable(training_path, training_artifact)
        write_immutable(receipt_path, receipt)
        print(json.dumps(receipt, sort_keys=True, indent=2))
        return 0


    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    training_artifact = json.loads(training_path.read_text(encoding="utf-8"))
    validate_campaign(root, receipt, training_artifact, snapshot_path.read_bytes())
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
