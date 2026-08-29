"""Build the machine-readable development-evidence package for a bounded cluster paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_manuscript_evidence_package_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/manuscript-evidence-package-v1.json")
READINESS_PATH = Path("runs/engine/gravity-cluster-publication-readiness-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-manuscript-evidence-package-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-manuscript-evidence-package-receipt-1.0"
SOURCE_IDS = (
    "item59",
    "comparators",
    "uncertainty",
    "numerical_controls",
    "data_contract",
    "replication_protocol",
    "prior_art",
    "nuisance_quotient_sampler_implementation",
)


class GravityClusterManuscriptPackageError(RuntimeError):
    """Raised when evidence, environment, or claim boundaries change."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterManuscriptPackageError(f"expected JSON object: {path}")
    return value


def _content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    if expected != actual:
        raise GravityClusterManuscriptPackageError("bound source content hash changed")
    return actual


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterManuscriptPackageError(f"{label} keys changed")


def load_config(root: Path) -> dict[str, Any]:
    value = _read_json(root.resolve() / CONFIG_PATH)
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "package_id",
            "purpose",
            "environment_freeze",
            "source_bindings",
            "included_sections",
            "claim_boundary",
            "output_path",
        },
        "manuscript package config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_development_evidence_only"
        or config["package_id"] != "gravity-cluster-manuscript-evidence-package-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterManuscriptPackageError("manuscript package identity changed")
    environment = config["environment_freeze"]
    if (
        environment["python"] != "3.13.5"
        or environment["numpy"] != "2.2.6"
        or environment["scipy"] != "1.16.1"
        or environment["pytest"] != "8.4.2"
        or len(environment["random_seed_sources"]) != 3
        or len(environment["scientific_freeze_commit"]) != 40
        or len(environment["xcop_archive_sha256"]) != 64
    ):
        raise GravityClusterManuscriptPackageError("environment or source revision changed")
    bindings = config["source_bindings"]
    if tuple(row["source_id"] for row in bindings) != SOURCE_IDS:
        raise GravityClusterManuscriptPackageError("evidence binding inventory changed")
    for row in bindings:
        if set(row) != {"source_id", "path", "file_sha256", "content_sha256"}:
            raise GravityClusterManuscriptPackageError("evidence binding keys changed")
        if len(row["file_sha256"]) != 64 or len(row["content_sha256"]) != 64:
            raise GravityClusterManuscriptPackageError("evidence binding hash changed")
    if len(config["included_sections"]) != 10:
        raise GravityClusterManuscriptPackageError("included evidence sections changed")
    if config["claim_boundary"] != {
        "development_evidence": True,
        "same_release_confirmation": True,
        "independent_replication": False,
        "full_source_covariance": False,
        "bounded_paper_ready": False,
        "physical_mechanism_ready": False,
        "universal_theory_ready": False,
        "historical_novelty_established": False,
        "alternative_to_gr_established": False,
        "dark_matter_eliminated": False,
    }:
        raise GravityClusterManuscriptPackageError("manuscript claim boundary weakened")


def _load_sources(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for binding in config["source_bindings"]:
        path = (root / str(binding["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GravityClusterManuscriptPackageError("evidence path escaped root") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterManuscriptPackageError(f"evidence file changed: {binding['path']}")
        value = _read_json(path)
        if _content_sha(value) != binding["content_sha256"]:
            raise GravityClusterManuscriptPackageError(
                f"evidence content changed: {binding['path']}"
            )
        result[str(binding["source_id"])] = value
    return result


def _score_without_rows(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "per_row"}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sources = _load_sources(root, config)
    item59 = sources["item59"]
    comparators = sources["comparators"]
    uncertainty = sources["uncertainty"]
    numerical = sources["numerical_controls"]
    data_contract = sources["data_contract"]
    protocol = sources["replication_protocol"]
    prior_art = sources["prior_art"]
    nuisance_sampler = sources["nuisance_quotient_sampler_implementation"]
    readiness = _read_json(root / READINESS_PATH)
    _content_sha(readiness)

    per_row = []
    split_summaries = {}
    for split in ("development_train", "development_holdout", "confirmation"):
        candidate = item59["splits"][split]["candidate"]
        split_summaries[split] = {
            "candidate": _score_without_rows(candidate),
            "baselines": item59["splits"][split]["baselines"],
            "improvements": item59["splits"][split]["improvements"],
        }
        for row in candidate["per_row"]:
            per_row.append({"split": split, **row})
    if len(per_row) != 233 or len({row["row_id"] for row in per_row}) != 233:
        raise GravityClusterManuscriptPackageError("per-row Item 59 inventory changed")

    access = {
        "scientific_freeze_commit": item59["scientific_freeze_commit"],
        "confirmation_response_files_opened_after_freeze": item59["counts"][
            "confirmation_response_files_opened_after_freeze"
        ],
        "development_holdout_rows": item59["counts"]["development_holdout_rows"],
        "same_release_confirmation_rows": item59["counts"]["confirmation_rows"],
        "direct_lensing_likelihood_evaluations": item59["counts"][
            "direct_lensing_likelihood_evaluations"
        ],
        "inferred_total_mass_rows": item59["counts"]["inferred_total_mass_rows"],
        "independent_target_rows_opened": readiness["counts"][
            "independent_target_rows_opened"
        ],
        "independent_observational_authorization": readiness["readiness"][
            "observational_authorization"
        ],
    }
    if access["independent_target_rows_opened"] != 0 or access[
        "independent_observational_authorization"
    ] is not False:
        raise GravityClusterManuscriptPackageError("independent target seal changed")

    body = {
        "schema_version": RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "decision": "DEVELOPMENT_MANUSCRIPT_EVIDENCE_PACKAGED_NOT_PAPER_READY",
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "completed_goal_evidence": {
            "CP12.2": "environment_dependencies_seeds_hardware_tolerance_scientific_freeze_and_source_archive_revision_frozen",
            "CP12.4": "all_233_candidate_rows_predictions_residuals_object_summaries_and_counterexample_ledgers_packaged",
            "CP12.5": "preflight_same_release_confirmation_and_independent_access_counts_packaged",
            "CP12.7": "ablations_negative_results_nuisance_failures_and_sensitivity_envelopes_packaged",
            "CP12.8": "absolute_and_comparator_relative_performance_packaged_together",
            "CP12.9": "bounded_mechanism_and_universal_claim_tracks_packaged",
        },
        "blocked_goal_evidence": {
            "CP12.1": "no_one_command_primary_figure_and_table_renderer",
            "CP12.3": "independent_source_calibration_covariance_split_and_exclusion_manifests_absent",
            "CP12.6": "no_external_analyst_or_separately_maintained_full_replay",
            "CP12.10": "statistical_cluster_astrophysics_and_modified_gravity_reviews_absent",
            "CP12.11": "eligible_data_licensing_and_public_release_package_incomplete",
            "CP12.12": "bounded_paper_gates_not_passed_and_no_submission_authorized",
        },
        "environment_and_revisions": config["environment_freeze"],
        "nuisance_quotient_sampler_implementation": {
            "decision": nuisance_sampler["decision"],
            "status": nuisance_sampler["status"],
            "implementation_evidence_only": nuisance_sampler["publication_readiness"][
                "implementation_evidence_only"
            ],
            "scientific_claims_added": nuisance_sampler["publication_readiness"][
                "scientific_claims_added"
            ],
            "production_authorized": nuisance_sampler["authorization_and_execution"][
                "production_authorized"
            ],
            "production_launches": nuisance_sampler["authorization_and_execution"][
                "production_launches"
            ],
            "bounded_smoke_forward_evaluations": nuisance_sampler["frozen_mechanics"][
                "bounded_smoke_forward_evaluations"
            ],
            "CP5_status": nuisance_sampler["publication_readiness"]["CP5_status"],
            "CP5_7_through_CP5_10": nuisance_sampler["publication_readiness"][
                "CP5_7_through_CP5_10"
            ],
        },
        "access_ledger": access,
        "split_summaries": split_summaries,
        "per_row_candidate_predictions": per_row,
        "object_and_counterexample_ledger": {
            "confirmation_cluster_wins": item59["confirmation_cluster_wins"],
            "confirmation_counterexamples": item59["confirmation_counterexamples"],
            "confirmation_counterexamples_by_cluster_observable": item59[
                "confirmation_counterexamples_by_cluster_observable"
            ],
            "counterexample_policy_assessment": item59[
                "counterexample_policy_assessment"
            ],
        },
        "comparators_and_ablations": {
            "ranking": comparators["ranking"],
            "candidate": comparators["candidate"],
            "comparators": comparators["comparators"],
            "ablations": comparators["ablations"],
            "limitations": comparators["limitations"],
        },
        "negative_and_numerical_controls": {
            "synthetic_recovery": numerical["synthetic_recovery"],
            "false_selection": numerical["false_selection"],
            "implementation_agreement": numerical["implementation_agreement"],
            "leakage_mutations": numerical["leakage_mutations"],
            "fold_controls": numerical["fold_controls"],
            "prospective_power_and_stopping": numerical["prospective_power_and_stopping"],
            "limitations": numerical["limitations"],
        },
        "uncertainty_and_alternative_cause_boundary": {
            "decision": uncertainty["decision"],
            "marginalization": uncertainty["marginalization"],
            "covariance_sensitivity": uncertainty["covariance_sensitivity"],
            "missingness_sensitivity": uncertainty["missingness_sensitivity"],
            "observationally_indistinguishable_causes": uncertainty[
                "observational_indistinguishability"
            ][
                "causes_remaining_indistinguishable_with_current_single_source_diagonal_errors"
            ],
            "source_covariance_blockers": {
                task: uncertainty["blocked_goal_evidence"][task]
                for task in ("CP5.1", "CP5.2", "CP5.3", "CP5.4", "CP5.5", "CP5.6")
            },
            "limitations": uncertainty["limitations"],
        },
        "prior_art_boundary": {
            "decision": prior_art["decision"],
            "candidate_adjudication": prior_art["candidate_adjudication"],
            "closest_behavioral_neighbor": prior_art["closest_behavioral_neighbor"],
        },
        "independent_replication_boundary": {
            "data_contract_decision": data_contract["decision"],
            "protocol_decision": protocol["decision"],
            "data_contract_claims": data_contract["claims"],
            "protocol_claims": protocol["claims"],
            "frozen_decision_summary": protocol["frozen_decision_summary"],
        },
        "claim_tracks": readiness["claim_tracks"],
        "claims": config["claim_boundary"],
        "limitations": sorted(
            {
                *map(str, item59["limitations"]),
                *map(str, comparators["limitations"]),
                *map(str, uncertainty["limitations"]),
                *map(str, numerical["limitations"]),
            }
        ),
        "counts": {
            "per_row_candidate_predictions": len(per_row),
            "clusters": item59["counts"]["clusters"],
            "development_clusters": item59["counts"]["development_clusters"],
            "same_release_confirmation_clusters": item59["counts"]["confirmation_clusters"],
            "comparators": comparators["counts"]["comparators"],
            "ablations": len(comparators["ablations"]),
            "null_trials": numerical["counts"]["null_trials"],
            "covariance_stress_scenarios": uncertainty["counts"][
                "covariance_sensitivity_scenarios"
            ],
            "missingness_stress_scenarios": uncertainty["counts"][
                "missingness_sensitivity_scenarios"
            ],
            "independent_target_rows_opened": 0,
        },
        "reproduction": {
            "package_command": "python -m sigma_theory_compiler.gravity_cluster_manuscript_evidence_package write",
            "package_check_command": "python -m sigma_theory_compiler.gravity_cluster_manuscript_evidence_package check",
            "item59_replay_command": "python -m sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate replay",
            "scope": "Recreates this machine-readable evidence package and the bound Item 59 result; it does not yet render every manuscript figure or table.",
        },
        "next_action": "Complete the independent source packet and covariance gates, then add a one-command figure/table renderer and external replay before any bounded-paper submission.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterManuscriptPackageError("manuscript evidence package changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "counts": receipt["counts"],
            "next_action": receipt["next_action"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
