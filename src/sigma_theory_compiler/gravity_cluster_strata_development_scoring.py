from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_strata_development_scoring_v1.json")
TEST_PATH = Path("tests/test_gravity_cluster_strata_development_scoring.py")
RECEIPT_PATH = Path("runs/gravity/publication-readiness/cluster-strata-development-scoring-v1.json")

CONFIG_SCHEMA = "invariant-gravity-cluster-strata-development-scoring-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-strata-development-scoring-receipt-1.0"
CLUSTERS = ["A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"]
SPLITS = ["development_holdout", "development_train"]
CANDIDATE = "ITEM59_CROSS_SCALE_BOUNDARY"
COMPARATOR = "GR_PLUS_NFW"

EXPECTED_INPUTS = {
    "predictor_preflight_receipt": {
        "path": "runs/gravity/publication-readiness/cluster-predictor-strata-preflight-v1.json",
        "file_sha256": "8cafbd3fb5a042d125a7667420967a7343b3934e0862121b4359fca0de6d9cc1",
        "content_sha256": "c64a0a0dc86c4bb7518c7634821b8f87e8265d79d405a16f8f72bc3ac4fc5acb",
    },
    "predictor_strata": {
        "path": (
            "runs/gravity/publication-readiness/cluster-predictor-strata-preflight-v1/"
            "predictor-only-strata.json"
        ),
        "file_sha256": "5e657a3fc5f9ab1179fcc18b1577b18cb3457b8acc364ec0151952740205f1cd",
        "content_sha256": "f30957f7a3a34c2309815ba95d60fde213abc190e09c9d9ea9d587e8317f045f",
    },
    "item59_receipt": {
        "path": "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json",
        "file_sha256": "2fd76f58bbd1754ea33b15fa0545a1397e8e12e441b6a25f11ddf8db735a5051",
        "content_sha256": "b8d824c89779c741d4d8bccebe61fe0b378554219424d4637835ef0519696df7",
    },
    "covariance_pilot": {
        "path": "runs/gravity/publication-readiness/pressure-covariance-scoring-pilot-v1.json",
        "file_sha256": "da5a61f29ff9366c431ba07503a998603e592b9d465a6eaf37c2faac3c8bd748",
        "content_sha256": "a84730f92449a7b78ce9b4bd522a602a88db83662bc714a8fbe240984f401193",
    },
}

EXPECTED_MODEL_FREEZE = {
    "candidate": {
        "family_id": "cross_scale_boundary",
        "model_id": CANDIDATE,
        "nuisances": {
            "missing_stellar_to_gas_mass_ratio": 0.2,
            "outer_nonthermal_fraction": 0.3,
            "published_stellar_mass_scale": 1.3,
            "xray_temperature_cross_calibration": 1,
        },
        "parameters": {"beta": 1.5},
        "refit": False,
        "variant_id": "cross_scale_boundary:5e945be899287b75",
    },
    "strongest_frozen_comparator": {
        "model_id": COMPARATOR,
        "nuisances": {
            "missing_stellar_to_gas_mass_ratio": 0.05,
            "outer_nonthermal_fraction": 0,
            "published_stellar_mass_scale": 1.3,
            "xray_temperature_cross_calibration": 1,
        },
        "parameters": {"c500": 3.5, "log10_halo_m500_solar_mass": 14.6},
        "refit": False,
        "selection_source": "frozen_development_comparator_suite_strongest_conventional",
    },
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    ).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return target


def write_json_no_clobber(path: Path, value: dict[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing artifact: {target}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if file_sha256(target) != expected_sha256:
        raise RuntimeError("config SHA256 mismatch")
    config = json.loads(target.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("config schema changed")
    if config.get("status") != "frozen_before_append_only_development_scoring":
        raise RuntimeError("config status changed")
    if config.get("implementation_source") != str(Path(__file__).relative_to(ROOT)).replace(
        "\\", "/"
    ):
        raise RuntimeError("implementation path changed")
    if config.get("implementation_source_normalized_sha256") != normalized_sha256(Path(__file__)):
        raise RuntimeError("implementation source seal mismatch")
    if config.get("inputs") != EXPECTED_INPUTS:
        raise RuntimeError("input bindings changed")
    if config.get("population", {}).get("cluster_ids") != CLUSTERS:
        raise RuntimeError("population changed")
    if config.get("model_freeze") != EXPECTED_MODEL_FREEZE:
        raise RuntimeError("model freeze changed")
    scoring = config.get("scoring_freeze", {})
    if scoring.get("primary_split") != "development_holdout":
        raise RuntimeError("primary split changed")
    if scoring.get("aggregation") != "equal_cluster_mean_never_equal_row":
        raise RuntimeError("aggregation changed")
    if scoring.get("partitions") != [
        "relaxation_proxy",
        "cool_core",
        "stellar_profile_availability",
        "positive_assembly_vs_unclassified",
        "boundary_method",
    ]:
        raise RuntimeError("partitions changed")
    if scoring.get("exact_permutation", {}).get("two_sided") is not True:
        raise RuntimeError("permutation rule changed")
    if scoring.get("multiplicity", {}).get("method") != "holm":
        raise RuntimeError("multiplicity rule changed")
    gates = config.get("gates", {})
    if gates != {
        "candidate_absolute_primary_max_score": 1.0,
        "candidate_advantage_primary_must_be_positive": True,
        "candidate_advantage_primary_minimum_cluster_wins": 5,
        "association_familywise_alpha": 0.05,
        "flip_explanation_requires_holm_advantage_and_flip_p": True,
        "flip_explanation_requires_train_holdout_advantage_direction_concordance": True,
        "single_counterexample_is_universal_veto": False,
    }:
        raise RuntimeError("gates changed")
    boundary = config.get("data_boundary", {})
    expected_zero = [
        "new_raw_target_rows_opened",
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "formula_refits",
        "nuisance_refits",
        "threshold_changes_after_scoring",
        "network_calls",
        "paid_or_model_calls",
    ]
    if any(boundary.get(key) != 0 for key in expected_zero):
        raise RuntimeError("data boundary changed")
    return config


def load_bound_json(binding: dict[str, Any], label: str) -> dict[str, Any]:
    target = confined(ROOT / binding["path"])
    if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
        raise RuntimeError(f"{label} file seal mismatch")
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise RuntimeError(f"{label} content seal mismatch")
    body = dict(value)
    stated = body.pop("content_sha256")
    if content_sha256(body) != stated:
        raise RuntimeError(f"{label} self-hash mismatch")
    return value


def validate_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight = load_bound_json(config["inputs"]["predictor_preflight_receipt"], "preflight")
    strata = load_bound_json(config["inputs"]["predictor_strata"], "strata")
    item59 = load_bound_json(config["inputs"]["item59_receipt"], "Item59")
    covariance = load_bound_json(config["inputs"]["covariance_pilot"], "covariance pilot")
    if preflight.get("decision") != "CP5_11_STRATA_FROZEN_CP5_13_REMAINS_OPEN_PLANNING_MATRIX_ONLY":
        raise RuntimeError("predictor preflight decision changed")
    if preflight.get("readiness", {}).get("CP5_13_task_complete") is not False:
        raise RuntimeError("CP5.13 predecessor status changed")
    if strata.get("status") != "predictor_strata_ready_scientific_scoring_not_run":
        raise RuntimeError("strata predecessor status changed")
    if item59.get("decision") != "ITEM59_XCOP_FORWARD_OBSERVABLE_GATE_PASSED_DEVELOPMENT_EVIDENCE":
        raise RuntimeError("Item59 decision changed")
    selected = item59.get("selection", {}).get("selected_qualifying", {}).get("variant")
    if selected != {
        "family_id": "cross_scale_boundary",
        "nuisances": EXPECTED_MODEL_FREEZE["candidate"]["nuisances"],
        "origin_label": "new_combination_of_known_permittivity_and_auxiliary_field_ideas",
        "parameters": {"beta": 1.5},
        "qualifying": True,
        "variant_id": "cross_scale_boundary:5e945be899287b75",
    }:
        raise RuntimeError("Item59 candidate lineage changed")
    if covariance.get("decision") != "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS":
        raise RuntimeError("covariance pilot decision changed")
    if covariance.get("model_freeze") != EXPECTED_MODEL_FREEZE:
        raise RuntimeError("covariance model freeze changed")
    access = covariance.get("access_boundary", {})
    if access.get("development_clusters") != CLUSTERS or access.get("allowed_splits") != [
        "development_train",
        "development_holdout",
    ]:
        raise RuntimeError("covariance development boundary changed")
    for key in (
        "formula_refits",
        "independent_target_rows_opened",
        "lensing_rows_opened",
        "model_selection_operations",
        "network_payload_reads",
        "nuisance_refits",
        "paid_model_calls",
        "same_release_confirmation_clusters_opened",
    ):
        if access.get(key) != 0:
            raise RuntimeError(f"covariance access boundary changed: {key}")
    return strata, covariance


def mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite or empty score vector")
    return math.fsum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def exact_permutation(values: list[float], group_a_indices: list[int]) -> dict[str, Any]:
    n = len(values)
    k = len(group_a_indices)
    if n != 8 or k < 2 or n - k < 2:
        raise RuntimeError("inferential permutation requires two groups with at least two clusters")
    observed_set = set(group_a_indices)

    def difference(indices: set[int]) -> float:
        return mean([values[index] for index in indices]) - mean(
            [values[index] for index in range(n) if index not in indices]
        )

    observed = difference(observed_set)
    distribution = [difference(set(indices)) for indices in itertools.combinations(range(n), k)]
    extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in distribution)
    return {
        "observed_mean_difference_a_minus_b": observed,
        "enumerations": len(distribution),
        "extreme_or_equal": extreme,
        "two_sided_p": extreme / len(distribution),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    models = record.get("models", {})
    if set(models) != {CANDIDATE, COMPARATOR}:
        raise RuntimeError("model set changed")
    output: dict[str, Any] = {"rows": models[CANDIDATE]["rows"]}
    if models[CANDIDATE]["rows"] != models[COMPARATOR]["rows"]:
        raise RuntimeError("candidate/comparator row counts differ")
    for covariance_name, source_key in (
        ("diagonal", "diagonal_score"),
        ("full_covariance", "full_covariance_score"),
    ):
        candidate_score = float(models[CANDIDATE][source_key])
        comparator_score = float(models[COMPARATOR][source_key])
        advantage = comparator_score - candidate_score
        if not math.isclose(
            advantage,
            float(record["candidate_advantage"][covariance_name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("retained advantage is inconsistent with retained scores")
        output[covariance_name] = {
            "candidate_score": candidate_score,
            "nfw_score": comparator_score,
            "candidate_advantage": advantage,
        }
    diagonal = output["diagonal"]["candidate_advantage"]
    full = output["full_covariance"]["candidate_advantage"]
    output["covariance_flip"] = (
        "positive_diagonal_to_negative_full"
        if diagonal > 0 and full < 0
        else "negative_diagonal_to_positive_full"
        if diagonal < 0 and full > 0
        else "no_sign_flip"
    )
    return output


def group_summary(rows: list[dict[str, Any]], members: list[str], split: str) -> dict[str, Any]:
    selected = [row["scores"][split] for row in rows if row["cluster_id"] in members]
    candidate = [row["full_covariance"]["candidate_score"] for row in selected]
    comparator = [row["full_covariance"]["nfw_score"] for row in selected]
    advantage = [row["full_covariance"]["candidate_advantage"] for row in selected]
    loo_means = (
        [mean(advantage[:index] + advantage[index + 1 :]) for index in range(len(advantage))]
        if len(advantage) > 1
        else []
    )
    return {
        "n": len(members),
        "members": members,
        "full_covariance_candidate_score_equal_cluster_mean": mean(candidate),
        "full_covariance_nfw_score_equal_cluster_mean": mean(comparator),
        "full_covariance_candidate_advantage_equal_cluster_mean": mean(advantage),
        "full_covariance_candidate_advantage_median": median(advantage),
        "candidate_wins": sum(value > 0 for value in advantage),
        "candidate_losses": sum(value < 0 for value in advantage),
        "ties": sum(value == 0 for value in advantage),
        "leave_one_cluster_out_advantage_mean_range": (
            [min(loo_means), max(loo_means)] if loo_means else None
        ),
    }


def build_partitions(strata_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def ids(predicate: Any) -> list[str]:
        return [cluster for cluster in CLUSTERS if predicate(strata_rows[cluster])]

    return {
        "relaxation_proxy": {
            "group_a_label": "relaxed_proxy",
            "group_a": ids(lambda row: row["relaxation_proxy_stratum"] == "relaxed_proxy"),
            "group_b_label": "disturbed_proxy",
            "group_b": ids(lambda row: row["relaxation_proxy_stratum"] == "disturbed_proxy"),
            "inferential": True,
        },
        "cool_core": {
            "group_a_label": "CC",
            "group_a": ids(lambda row: row["cool_core_stratum"] == "CC"),
            "group_b_label": "NCC",
            "group_b": ids(lambda row: row["cool_core_stratum"] == "NCC"),
            "inferential": True,
        },
        "stellar_profile_availability": {
            "group_a_label": "available",
            "group_a": ids(lambda row: row["published_stellar_profile_available"] is True),
            "group_b_label": "unavailable",
            "group_b": ids(lambda row: row["published_stellar_profile_available"] is False),
            "inferential": True,
        },
        "positive_assembly_vs_unclassified": {
            "group_a_label": "positive_merger_or_sloshing_report",
            "group_a": ids(
                lambda row: (
                    row["assembly_literature_stratum"]
                    in {"sloshing_reported", "sub_or_post_merger_reported"}
                )
            ),
            "group_b_label": "explicitly_unclassified_not_negative",
            "group_b": ids(
                lambda row: (
                    row["assembly_literature_stratum"] == "no_class_assigned_in_frozen_source"
                )
            ),
            "inferential": True,
        },
        "boundary_method": {
            "group_a_label": "standard_xmm_outer_background",
            "group_a": ids(
                lambda row: (
                    row["boundary_background_method"] == "xmm_mosaic_r_gt_2r500_5pct_systematic"
                )
            ),
            "group_b_label": "A3266_rosat_exception_singleton",
            "group_b": ids(
                lambda row: row["boundary_background_method"] == "rosat_background_30pct_systematic"
            ),
            "inferential": False,
            "reason": "A3266_is_a_singleton_descriptive_only_no_permutation_or_generalization",
        },
    }


def build_analysis(config: dict[str, Any]) -> dict[str, Any]:
    strata, covariance = validate_inputs(config)
    strata_rows = {row["cluster_id"]: row for row in strata["cluster_rows"]}
    if list(strata_rows) != CLUSTERS:
        raise RuntimeError("predictor strata population/order changed")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in covariance.get("per_cluster", []):
        key = (record.get("cluster"), record.get("split"))
        if key in records:
            raise RuntimeError("duplicate covariance score record")
        records[key] = score_record(record)
    if set(records) != {(cluster, split) for cluster in CLUSTERS for split in SPLITS}:
        raise RuntimeError("covariance score record population changed")

    per_object = []
    for cluster in CLUSTERS:
        predictor = strata_rows[cluster]
        per_object.append(
            {
                "cluster_id": cluster,
                "strata": {
                    "relaxation_proxy": predictor["relaxation_proxy_stratum"],
                    "cool_core": predictor["cool_core_stratum"],
                    "stellar_profile_available": predictor["published_stellar_profile_available"],
                    "assembly_literature": predictor["assembly_literature_stratum"],
                    "boundary_background_method": predictor["boundary_background_method"],
                },
                "scores": {split: records[(cluster, split)] for split in SPLITS},
            }
        )

    whole_population = {}
    for split in SPLITS:
        split_rows = [row["scores"][split] for row in per_object]
        whole_population[split] = {}
        for covariance_name in ("diagonal", "full_covariance"):
            candidate = [row[covariance_name]["candidate_score"] for row in split_rows]
            comparator = [row[covariance_name]["nfw_score"] for row in split_rows]
            advantage = [row[covariance_name]["candidate_advantage"] for row in split_rows]
            whole_population[split][covariance_name] = {
                "candidate_score_equal_cluster_mean": mean(candidate),
                "nfw_score_equal_cluster_mean": mean(comparator),
                "candidate_advantage_equal_cluster_mean": mean(advantage),
                "candidate_wins": sum(value > 0 for value in advantage),
                "candidate_losses": sum(value < 0 for value in advantage),
                "ties": sum(value == 0 for value in advantage),
            }
            retained = covariance["aggregates"][split]
            if not math.isclose(
                mean(candidate),
                retained["models"][CANDIDATE][f"{covariance_name}_score"],
                abs_tol=1e-12,
            ) or not math.isclose(
                mean(comparator),
                retained["models"][COMPARATOR][f"{covariance_name}_score"],
                abs_tol=1e-12,
            ):
                raise RuntimeError("equal-cluster aggregate does not reproduce covariance receipt")
        whole_population[split]["positive_diagonal_to_negative_full_clusters"] = [
            row["cluster_id"]
            for row in per_object
            if row["scores"][split]["covariance_flip"] == "positive_diagonal_to_negative_full"
        ]
        whole_population[split]["negative_diagonal_to_positive_full_clusters"] = [
            row["cluster_id"]
            for row in per_object
            if row["scores"][split]["covariance_flip"] == "negative_diagonal_to_positive_full"
        ]

    partitions = build_partitions(strata_rows)
    partition_results: dict[str, Any] = {}
    for name, partition in partitions.items():
        if sorted(partition["group_a"] + partition["group_b"]) != sorted(CLUSTERS):
            raise RuntimeError(f"partition does not cover population: {name}")
        result: dict[str, Any] = {**partition, "splits": {}}
        for split in SPLITS:
            result["splits"][split] = {
                "group_a": group_summary(per_object, partition["group_a"], split),
                "group_b": group_summary(per_object, partition["group_b"], split),
            }
            if partition["inferential"]:
                indices = [CLUSTERS.index(cluster) for cluster in partition["group_a"]]
                advantage = [
                    row["scores"][split]["full_covariance"]["candidate_advantage"]
                    for row in per_object
                ]
                absolute = [
                    row["scores"][split]["full_covariance"]["candidate_score"] for row in per_object
                ]
                result["splits"][split]["exact_permutation"] = {
                    "candidate_advantage": exact_permutation(advantage, indices),
                    "candidate_absolute_score": exact_permutation(absolute, indices),
                }
        if partition["inferential"]:
            indices = [CLUSTERS.index(cluster) for cluster in partition["group_a"]]
            flips = [
                float(
                    row["scores"]["development_holdout"]["covariance_flip"]
                    == "positive_diagonal_to_negative_full"
                )
                for row in per_object
            ]
            result["primary_flip_enrichment_exact_permutation"] = exact_permutation(flips, indices)
        partition_results[name] = result

    inferential_names = [name for name, value in partitions.items() if value["inferential"]]
    primary_advantage_p = {
        name: partition_results[name]["splits"]["development_holdout"]["exact_permutation"][
            "candidate_advantage"
        ]["two_sided_p"]
        for name in inferential_names
    }
    primary_absolute_p = {
        name: partition_results[name]["splits"]["development_holdout"]["exact_permutation"][
            "candidate_absolute_score"
        ]["two_sided_p"]
        for name in inferential_names
    }
    primary_flip_p = {
        name: partition_results[name]["primary_flip_enrichment_exact_permutation"]["two_sided_p"]
        for name in inferential_names
    }
    multiplicity = {
        "candidate_advantage_holm_p": holm_adjust(primary_advantage_p),
        "candidate_absolute_score_holm_p": holm_adjust(primary_absolute_p),
        "flip_enrichment_holm_p": holm_adjust(primary_flip_p),
    }
    alpha = config["gates"]["association_familywise_alpha"]
    explanation_by_partition = {}
    for name in inferential_names:
        holdout_difference = partition_results[name]["splits"]["development_holdout"][
            "exact_permutation"
        ]["candidate_advantage"]["observed_mean_difference_a_minus_b"]
        train_difference = partition_results[name]["splits"]["development_train"][
            "exact_permutation"
        ]["candidate_advantage"]["observed_mean_difference_a_minus_b"]
        direction_concordant = holdout_difference * train_difference > 0
        explanation_by_partition[name] = {
            "advantage_holm_significant": multiplicity["candidate_advantage_holm_p"][name] <= alpha,
            "flip_enrichment_holm_significant": multiplicity["flip_enrichment_holm_p"][name]
            <= alpha,
            "train_holdout_advantage_direction_concordant": direction_concordant,
            "passes_frozen_explanation_gate": (
                multiplicity["candidate_advantage_holm_p"][name] <= alpha
                and multiplicity["flip_enrichment_holm_p"][name] <= alpha
                and direction_concordant
            ),
        }

    primary = whole_population["development_holdout"]["full_covariance"]
    gates = {
        "candidate_absolute_primary": {
            "threshold_max": config["gates"]["candidate_absolute_primary_max_score"],
            "observed": primary["candidate_score_equal_cluster_mean"],
            "passed": primary["candidate_score_equal_cluster_mean"]
            <= config["gates"]["candidate_absolute_primary_max_score"],
        },
        "candidate_vs_nfw_primary": {
            "requires_positive_mean_advantage": True,
            "minimum_cluster_wins": config["gates"][
                "candidate_advantage_primary_minimum_cluster_wins"
            ],
            "observed_mean_advantage": primary["candidate_advantage_equal_cluster_mean"],
            "observed_cluster_wins": primary["candidate_wins"],
            "passed": primary["candidate_advantage_equal_cluster_mean"] > 0
            and primary["candidate_wins"]
            >= config["gates"]["candidate_advantage_primary_minimum_cluster_wins"],
        },
        "covariance_flip_explained_by_any_frozen_stratum": {
            "by_partition": explanation_by_partition,
            "passed": any(
                value["passes_frozen_explanation_gate"]
                for value in explanation_by_partition.values()
            ),
        },
    }
    return {
        "per_object": per_object,
        "whole_population": whole_population,
        "partition_results": partition_results,
        "multiplicity": multiplicity,
        "gates": gates,
    }


def artifact_binding(path: Path) -> dict[str, Any]:
    target = confined(path)
    return {
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "file_sha256": file_sha256(target),
    }


def expected_receipt(config: dict[str, Any]) -> dict[str, Any]:
    analysis = build_analysis(config)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "eight_object_exploratory_development_strata_scored",
        "decision": "NO_FROZEN_STRATUM_EXPLAINS_FULL_COVARIANCE_PRESSURE_FLIPS",
        "chronology": config["chronology"],
        "evidence": {
            "source": artifact_binding(Path(__file__)),
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "tests": artifact_binding(ROOT / TEST_PATH),
            **config["inputs"],
        },
        "model_freeze": config["model_freeze"],
        "scoring_freeze": config["scoring_freeze"],
        "results": analysis,
        "compute_and_access_accounting": {
            "sealed_json_artifacts_read": 4,
            "retained_per_object_split_score_records_read": 16,
            "retained_pressure_rows_summarized_by_predecessor": 54,
            "new_raw_target_rows_opened": 0,
            "new_target_scoring_calls": 0,
            "formula_refits": 0,
            "nuisance_refits": 0,
            "model_selection_operations": 0,
            "threshold_changes_after_scoring": 0,
            "confirmation_rows_opened": 0,
            "independent_rows_opened": 0,
            "group_rows_opened": 0,
            "lensing_rows_opened": 0,
            "network_calls": 0,
            "paid_or_model_calls": 0,
            "cpu_only_exact_enumerations": True,
            "gpu_calls": 0,
        },
        "readiness": {
            "CP5_11_predictor_strata_were_frozen_before_this_score": True,
            "CP5_11_exploratory_development_scoring_executed": True,
            "CP5_13_task_complete": False,
            "all_seven_cause_families_scientifically_compared": False,
            "boundary_method_inference_complete": False,
            "confirmation_or_independent_replication_complete": False,
        },
        "claim_boundary": {
            "eight_object_results_are_exploratory": True,
            "positive_assembly_group_is_positive_evidence_only": True,
            "unclassified_is_not_a_negative_nonmerger_or_nonsloshing_class": True,
            "A3266_boundary_result_is_singleton_descriptive_only": True,
            "strata_explain_covariance_flips": False,
            "causal_variable_identified": False,
            "candidate_supported_or_refuted_scientifically": False,
            "alternative_to_gr_established": False,
            "dark_matter_eliminated": False,
            "publication_readiness_changed": False,
            "scientific_claim_allowed": False,
            "single_counterexample_is_universal_veto": False,
        },
        "limitations": [
            "Only eight historically exposed development clusters are described; no confirmation or independent targets were accessed.",
            "The inputs are retained per-cluster pressure score summaries, not newly opened raw observations.",
            "Exact permutation p-values have coarse finite-sample resolution and cannot establish causality.",
            "Morphology, cool-core state, stellar-profile availability, and positive assembly reports are proxies or coverage labels, not direct measurements of all seven alternative causes.",
            "The A3266 ROSAT boundary-method cell is a singleton and is descriptive only.",
            "CP5.13 remains false because all seven alternative-cause families were not scientifically compared.",
        ],
    }
    if analysis["gates"]["covariance_flip_explained_by_any_frozen_stratum"]["passed"]:
        raise RuntimeError("frozen receipt decision is inconsistent with computed explanation gate")
    body["content_sha256"] = content_sha256(body)
    return body


def build(config_path: Path, expected_config_sha256: str) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    receipt = expected_receipt(config)
    write_json_no_clobber(ROOT / RECEIPT_PATH, receipt)
    return receipt


def check(config_path: Path, expected_config_sha256: str, receipt_path: Path) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    expected = expected_receipt(config)
    target = confined(receipt_path)
    if not target.is_file() or json.loads(target.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("development strata scoring receipt changed")
    return {
        "valid": True,
        "decision": expected["decision"],
        "candidate_absolute_primary_passed": expected["results"]["gates"][
            "candidate_absolute_primary"
        ]["passed"],
        "candidate_vs_nfw_primary_passed": expected["results"]["gates"]["candidate_vs_nfw_primary"][
            "passed"
        ],
        "covariance_flip_explained": False,
        "CP5_13_task_complete": False,
        "new_raw_target_rows_opened": 0,
        "scientific_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "check"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--expected-config-sha256", required=True)
        if name == "check":
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = (
        build(args.config, args.expected_config_sha256)
        if args.command == "build"
        else check(args.config, args.expected_config_sha256, args.receipt)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
