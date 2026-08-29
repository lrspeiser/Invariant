"""Score the frozen Item 59 development pressure predictions with X-COP covariance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparator
from sigma_theory_compiler import (
    gravity_cluster_development_covariance_reconstruction as reconstruction,
)
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

CONFIG_PATH = Path("configs/gravity_cluster_pressure_covariance_scoring_pilot_v1.json")
OUTPUT_PATH = Path(
    "runs/gravity/publication-readiness/pressure-covariance-scoring-pilot-v1.json"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-pressure-covariance-scoring-pilot-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-cluster-pressure-covariance-scoring-pilot-receipt-1.0"
)
CLUSTERS = reconstruction.DEVELOPMENT_CLUSTERS
SPLITS = ("development_train", "development_holdout")
MODELS = ("ITEM59_CROSS_SCALE_BOUNDARY", "GR_PLUS_NFW")


class GravityClusterPressureCovariancePilotError(RuntimeError):
    """Raised when the frozen pilot boundary or evidence changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterPressureCovariancePilotError(f"expected object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterPressureCovariancePilotError(f"{label} keys changed")


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityClusterPressureCovariancePilotError(
            f"{label} escaped repository root"
        ) from error
    return path


def _validate_content_hash(value: Mapping[str, Any], expected: str) -> None:
    body = dict(value)
    stored = body.pop("content_sha256", None)
    if stored != expected or _sha(body) != expected:
        raise GravityClusterPressureCovariancePilotError("bound receipt content changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "pilot_id",
            "purpose",
            "implementation_binding",
            "source_bindings",
            "access_boundary",
            "sample_freeze",
            "model_freeze",
            "scoring_freeze",
            "adjudication_thresholds",
            "claim_boundary",
            "output_path",
        },
        "pressure covariance pilot",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_before_covariance_scoring"
        or config["pilot_id"] != "gravity-cluster-pressure-covariance-scoring-pilot-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterPressureCovariancePilotError("pilot identity changed")

    implementation = config["implementation_binding"]
    if set(implementation) != {"path", "file_sha256"} or (
        implementation["path"]
        != "src/sigma_theory_compiler/gravity_cluster_pressure_covariance_scoring_pilot.py"
        or len(str(implementation["file_sha256"])) != 64
    ):
        raise GravityClusterPressureCovariancePilotError("implementation binding changed")

    expected_sources = (
        "RECONSTRUCTION_CONFIG",
        "RECONSTRUCTION_IMPLEMENTATION",
        "RECONSTRUCTION_RECEIPT",
        "ITEM59_CONFIG",
        "ITEM59_IMPLEMENTATION",
        "COMPARATOR_CONFIG",
        "COMPARATOR_IMPLEMENTATION",
        "COMPARATOR_RECEIPT",
    )
    sources = config["source_bindings"]
    if tuple(row.get("source_id") for row in sources) != expected_sources:
        raise GravityClusterPressureCovariancePilotError("source inventory changed")
    for row in sources:
        _strict(
            row,
            {"source_id", "path", "file_sha256", "content_sha256"},
            "source binding",
        )
        if len(str(row["file_sha256"])) != 64 or (
            row["content_sha256"] is not None
            and len(str(row["content_sha256"])) != 64
        ):
            raise GravityClusterPressureCovariancePilotError("source hash changed")

    if config["access_boundary"] != {
        "development_clusters": list(CLUSTERS),
        "allowed_splits": list(SPLITS),
        "observable": "pressure",
        "same_release_confirmation_clusters_opened": 0,
        "independent_target_rows_opened": 0,
        "lensing_rows_opened": 0,
        "formula_refits": 0,
        "nuisance_refits": 0,
        "model_selection_operations": 0,
        "paid_model_calls": 0,
        "network_payload_reads": 0,
    }:
        raise GravityClusterPressureCovariancePilotError("access boundary changed")

    sample = config["sample_freeze"]
    if tuple(row.get("cluster") for row in sample) != CLUSTERS:
        raise GravityClusterPressureCovariancePilotError("sample population changed")
    total_rows = 0
    for row in sample:
        _strict(
            row,
            {"cluster", "anchor_index", "pressure_rows", "scored_r_over_r500"},
            "cluster row freeze",
        )
        cluster = str(row["cluster"])
        pressure_rows = row["pressure_rows"]
        if not pressure_rows or len({item["row_id"] for item in pressure_rows}) != len(
            pressure_rows
        ):
            raise GravityClusterPressureCovariancePilotError("pressure row freeze changed")
        for item in pressure_rows:
            _strict(item, {"row_id", "index", "split", "r_over_r500"}, "pressure row")
            if (
                item["row_id"] != f"{cluster}:pressure:{item['index']}"
                or item["split"] not in SPLITS
                or not math.isfinite(float(item["r_over_r500"]))
                or float(item["r_over_r500"]) <= 0.0
            ):
                raise GravityClusterPressureCovariancePilotError(
                    "pressure row identity changed"
                )
        radii = [float(item["r_over_r500"]) for item in pressure_rows]
        if row["scored_r_over_r500"] != [min(radii), max(radii)]:
            raise GravityClusterPressureCovariancePilotError("radial range changed")
        if int(row["anchor_index"]) in {int(item["index"]) for item in pressure_rows}:
            raise GravityClusterPressureCovariancePilotError("anchor became scored")
        total_rows += len(pressure_rows)
    if total_rows != 54:
        raise GravityClusterPressureCovariancePilotError("pressure row count changed")

    models = config["model_freeze"]
    if set(models) != {"candidate", "strongest_frozen_comparator"}:
        raise GravityClusterPressureCovariancePilotError("model inventory changed")
    candidate = models["candidate"]
    if candidate != {
        "model_id": "ITEM59_CROSS_SCALE_BOUNDARY",
        "family_id": "cross_scale_boundary",
        "variant_id": "cross_scale_boundary:5e945be899287b75",
        "parameters": {"beta": 1.5},
        "nuisances": {
            "missing_stellar_to_gas_mass_ratio": 0.2,
            "outer_nonthermal_fraction": 0.3,
            "published_stellar_mass_scale": 1.3,
            "xray_temperature_cross_calibration": 1.0,
        },
        "refit": False,
    }:
        raise GravityClusterPressureCovariancePilotError("candidate changed")
    strongest = models["strongest_frozen_comparator"]
    if strongest != {
        "model_id": "GR_PLUS_NFW",
        "selection_source": "frozen_development_comparator_suite_strongest_conventional",
        "parameters": {"c500": 3.5, "log10_halo_m500_solar_mass": 14.6},
        "nuisances": {
            "missing_stellar_to_gas_mass_ratio": 0.05,
            "outer_nonthermal_fraction": 0.0,
            "published_stellar_mass_scale": 1.3,
            "xray_temperature_cross_calibration": 1.0,
        },
        "refit": False,
    }:
        raise GravityClusterPressureCovariancePilotError("comparator changed")

    scoring = config["scoring_freeze"]
    if scoring != {
        "space": "log_pressure",
        "fractional_error_floor": 0.05,
        "covariance_floor_rule": "preserve_released_correlation_and_replace_each_marginal_fractional_sigma_with_max_released_fractional_sigma_0p05",
        "diagonal_metric": "mean_squared_standardized_log_residual",
        "full_metric": "log_residual_transpose_covariance_inverse_log_residual_divided_by_rows",
        "aggregation": "equal_cluster_mean_separately_per_split",
        "primary_split": "development_holdout",
        "expected_diagonal_scores": {
            "development_train": {
                "ITEM59_CROSS_SCALE_BOUNDARY": 2.799233887535375,
                "GR_PLUS_NFW": 9.345947485465649,
            },
            "development_holdout": {
                "ITEM59_CROSS_SCALE_BOUNDARY": 4.202682093406597,
                "GR_PLUS_NFW": 11.038980283164767,
            },
        },
        "expected_diagonal_score_absolute_tolerance": 1e-10,
        "selection_after_scoring": False,
    }:
        raise GravityClusterPressureCovariancePilotError("scoring freeze changed")

    thresholds = config["adjudication_thresholds"]
    if thresholds != {
        "maximum_condition_number": 1e8,
        "maximum_inverse_identity_residual": 1e-10,
        "maximum_diagonal_reconstruction_relative_error": 1e-12,
        "minimum_primary_full_covariance_cluster_wins": 6,
        "require_primary_full_covariance_candidate_advantage_positive": True,
        "require_primary_diagonal_and_full_ranking_concordance": True,
        "magnitude_sensitivity_relative_change": 0.25,
        "single_cluster_failure_terminal": False,
    }:
        raise GravityClusterPressureCovariancePilotError("thresholds changed")

    claims = config["claim_boundary"]
    if claims != {
        "CP5_1_advances_to_development_scored": True,
        "CP5_1_complete": False,
        "CP5_2_through_CP5_6_complete": False,
        "component_covariance_separated": False,
        "temperature_or_density_covariance_used": False,
        "independent_replication": False,
        "physical_mechanism_established": False,
        "alternative_to_gr_established": False,
        "dark_matter_eliminated": False,
        "scientific_promotion_authorized": False,
    }:
        raise GravityClusterPressureCovariancePilotError("claim boundary changed")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root.resolve() / CONFIG_PATH)
    validate_config(config)
    return config


def _load_sources(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for binding in config["source_bindings"]:
        path = _under(root, str(binding["path"]), "source")
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterPressureCovariancePilotError(
                f"bound source changed: {binding['source_id']}"
            )
        content_sha = binding["content_sha256"]
        if content_sha is None:
            result[str(binding["source_id"])] = path
        else:
            value = _read_json(path)
            _validate_content_hash(value, str(content_sha))
            result[str(binding["source_id"])] = value
    implementation = config["implementation_binding"]
    path = _under(root, str(implementation["path"]), "implementation")
    if not path.is_file() or _file_sha(path) != implementation["file_sha256"]:
        raise GravityClusterPressureCovariancePilotError("pilot implementation changed")
    receipt = result["COMPARATOR_RECEIPT"]
    if (
        receipt["sample"]["clusters"] != list(CLUSTERS)
        or receipt["sample"]["xcop_confirmation_rows_used"] is not False
        or receipt["sample"]["independent_source_rows_used"] is not False
        or receipt["ranking"]["strongest_conventional"] != "GR_PLUS_NFW"
        or receipt["candidate"]["selection"]["refit"] is not False
    ):
        raise GravityClusterPressureCovariancePilotError(
            "development comparator evidence changed"
        )
    return result


def _packets_and_predictions(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    config59 = item59.load_config(root)
    packets = comparator._development_packets(root, config59)
    if tuple(str(packet["cluster"]) for packet in packets) != CLUSTERS:
        raise GravityClusterPressureCovariancePilotError("development packets changed")

    frozen_rows = {str(row["cluster"]): row for row in config["sample_freeze"]}
    for packet in packets:
        cluster = str(packet["cluster"])
        actual = [
            {
                "row_id": str(row["row_id"]),
                "index": int(str(row["row_id"]).rsplit(":", 1)[1]),
                "split": str(row["split"]),
                "r_over_r500": float(row["radius_kpc"]) / float(packet["r500_kpc"]),
            }
            for row in packet["rows"]
            if row["observable"] == "pressure"
        ]
        expected = frozen_rows[cluster]
        if (
            actual != expected["pressure_rows"]
            or int(packet["anchor"]["index"]) != expected["anchor_index"]
        ):
            raise GravityClusterPressureCovariancePilotError(
                f"frozen pressure rows changed: {cluster}"
            )

    candidate = config["model_freeze"]["candidate"]
    candidate_predictions = item59._variant_predictions(
        packets,
        {
            "family_id": candidate["family_id"],
            "parameters": candidate["parameters"],
            "nuisances": candidate["nuisances"],
        },
        config59,
    )
    nfw = config["model_freeze"]["strongest_frozen_comparator"]
    nfw_predictions = comparator._gravity_model_predictions(
        packets,
        "GR_PLUS_NFW",
        nfw["parameters"],
        nfw["nuisances"],
        config59,
    )
    return packets, {
        "ITEM59_CROSS_SCALE_BOUNDARY": candidate_predictions,
        "GR_PLUS_NFW": nfw_predictions,
    }


def _score_vector(
    residual: np.ndarray, correlation: np.ndarray, sigma: np.ndarray
) -> dict[str, float | bool]:
    covariance = correlation * np.outer(sigma, sigma)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    if float(eigenvalues[0]) <= 0.0:
        raise GravityClusterPressureCovariancePilotError(
            "scored covariance is not positive definite"
        )
    inverse = np.linalg.inv(covariance)
    identity_residual = float(
        np.max(np.abs(covariance @ inverse - np.eye(len(covariance))))
    )
    diagonal_relative_error = float(
        np.max(np.abs(np.diag(covariance) / (sigma**2) - 1.0))
    )
    return {
        "rows": len(residual),
        "diagonal_score": float(np.mean((residual / sigma) ** 2)),
        "full_covariance_score": float(residual @ inverse @ residual / len(residual)),
        "full_minus_diagonal": float(
            residual @ inverse @ residual / len(residual)
            - np.mean((residual / sigma) ** 2)
        ),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "inverse_identity_residual": identity_residual,
        "diagonal_reconstruction_relative_error": diagonal_relative_error,
        "maximum_absolute_offdiagonal_correlation": float(
            np.max(np.abs(correlation - np.eye(len(correlation))))
        ),
        "cholesky_succeeded": bool(np.linalg.cholesky(covariance).size),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sources = _load_sources(root, config)
    covariances = reconstruction.reconstruct_pressure_covariances(root)
    packets, predictions = _packets_and_predictions(root, config)
    packet_by_cluster = {str(packet["cluster"]): packet for packet in packets}
    frozen_by_cluster = {str(row["cluster"]): row for row in config["sample_freeze"]}
    floor = float(config["scoring_freeze"]["fractional_error_floor"])

    per_cluster: list[dict[str, Any]] = []
    for cluster in CLUSTERS:
        packet = packet_by_cluster[cluster]
        row_lookup = {str(row["row_id"]): row for row in packet["rows"]}
        full_correlation = np.asarray(covariances[cluster]["correlation"], dtype=float)
        for split in SPLITS:
            frozen_rows = [
                row
                for row in frozen_by_cluster[cluster]["pressure_rows"]
                if row["split"] == split
            ]
            indices = np.asarray([int(row["index"]) for row in frozen_rows], dtype=int)
            correlation = full_correlation[np.ix_(indices, indices)]
            observed = np.asarray(
                [float(row_lookup[row["row_id"]]["observed"]) for row in frozen_rows],
                dtype=float,
            )
            released_error = np.asarray(
                [float(row_lookup[row["row_id"]]["error"]) for row in frozen_rows],
                dtype=float,
            )
            sigma = np.maximum(released_error / observed, floor)
            model_scores = {}
            for model in MODELS:
                predicted = np.asarray(
                    [float(predictions[model][row["row_id"]]) for row in frozen_rows],
                    dtype=float,
                )
                if np.any(predicted <= 0.0) or not np.all(np.isfinite(predicted)):
                    raise GravityClusterPressureCovariancePilotError(
                        f"invalid frozen prediction: {model}:{cluster}:{split}"
                    )
                residual = np.log(predicted / observed)
                model_scores[model] = _score_vector(residual, correlation, sigma)
            diagonal_advantage = float(
                model_scores["GR_PLUS_NFW"]["diagonal_score"]
                - model_scores["ITEM59_CROSS_SCALE_BOUNDARY"]["diagonal_score"]
            )
            full_advantage = float(
                model_scores["GR_PLUS_NFW"]["full_covariance_score"]
                - model_scores["ITEM59_CROSS_SCALE_BOUNDARY"]["full_covariance_score"]
            )
            per_cluster.append(
                {
                    "cluster": cluster,
                    "split": split,
                    "row_ids": [row["row_id"] for row in frozen_rows],
                    "r_over_r500": [row["r_over_r500"] for row in frozen_rows],
                    "models": model_scores,
                    "candidate_advantage": {
                        "diagonal": diagonal_advantage,
                        "full_covariance": full_advantage,
                        "full_minus_diagonal": full_advantage - diagonal_advantage,
                    },
                }
            )

    aggregates: dict[str, dict[str, Any]] = {}
    expected = config["scoring_freeze"]["expected_diagonal_scores"]
    tolerance = float(
        config["scoring_freeze"]["expected_diagonal_score_absolute_tolerance"]
    )
    for split in SPLITS:
        members = [row for row in per_cluster if row["split"] == split]
        model_aggregates = {}
        for model in MODELS:
            diagonal = float(
                np.mean([row["models"][model]["diagonal_score"] for row in members])
            )
            full = float(
                np.mean(
                    [row["models"][model]["full_covariance_score"] for row in members]
                )
            )
            if abs(diagonal - float(expected[split][model])) > tolerance:
                raise GravityClusterPressureCovariancePilotError(
                    f"frozen diagonal score changed: {split}:{model}"
                )
            model_aggregates[model] = {
                "diagonal_score": diagonal,
                "full_covariance_score": full,
                "full_minus_diagonal": full - diagonal,
                "relative_change": (full - diagonal) / diagonal,
            }
        diagonal_advantage = (
            model_aggregates["GR_PLUS_NFW"]["diagonal_score"]
            - model_aggregates["ITEM59_CROSS_SCALE_BOUNDARY"]["diagonal_score"]
        )
        full_advantage = (
            model_aggregates["GR_PLUS_NFW"]["full_covariance_score"]
            - model_aggregates["ITEM59_CROSS_SCALE_BOUNDARY"]["full_covariance_score"]
        )
        aggregates[split] = {
            "models": model_aggregates,
            "candidate_advantage": {
                "diagonal": diagonal_advantage,
                "full_covariance": full_advantage,
                "full_minus_diagonal": full_advantage - diagonal_advantage,
            },
            "candidate_cluster_wins": {
                "diagonal": sum(
                    row["candidate_advantage"]["diagonal"] > 0.0 for row in members
                ),
                "full_covariance": sum(
                    row["candidate_advantage"]["full_covariance"] > 0.0
                    for row in members
                ),
            },
            "ranking_concordant": (diagonal_advantage > 0.0) == (full_advantage > 0.0),
        }

    thresholds = config["adjudication_thresholds"]
    all_scores = [
        metrics
        for row in per_cluster
        for metrics in row["models"].values()
    ]
    numerical_gates = {
        "all_cholesky_succeeded": all(row["cholesky_succeeded"] for row in all_scores),
        "condition_numbers_within_threshold": all(
            row["condition_number"] <= thresholds["maximum_condition_number"]
            for row in all_scores
        ),
        "inverse_residuals_within_threshold": all(
            row["inverse_identity_residual"]
            <= thresholds["maximum_inverse_identity_residual"]
            for row in all_scores
        ),
        "diagonal_reconstruction_within_threshold": all(
            row["diagonal_reconstruction_relative_error"]
            <= thresholds["maximum_diagonal_reconstruction_relative_error"]
            for row in all_scores
        ),
        "diagonal_scores_reproduce_frozen_evidence": True,
    }
    primary = aggregates[config["scoring_freeze"]["primary_split"]]
    robustness_gates = {
        "primary_full_covariance_candidate_advantage_positive": (
            primary["candidate_advantage"]["full_covariance"] > 0.0
        ),
        "primary_full_covariance_cluster_wins_meet_threshold": (
            primary["candidate_cluster_wins"]["full_covariance"]
            >= thresholds["minimum_primary_full_covariance_cluster_wins"]
        ),
        "primary_diagonal_and_full_ranking_concordant": primary[
            "ranking_concordant"
        ],
    }
    numerical_pass = all(numerical_gates.values())
    robustness_pass = all(robustness_gates.values())
    magnitude_threshold = float(thresholds["magnitude_sensitivity_relative_change"])
    primary_magnitude_sensitive = any(
        abs(row["relative_change"]) > magnitude_threshold
        for row in primary["models"].values()
    )
    if not numerical_pass:
        adjudication = "FAIL_NUMERICAL_COVARIANCE_VALIDITY"
    elif not robustness_pass:
        adjudication = "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS"
    elif primary_magnitude_sensitive:
        adjudication = "PASS_RANKING_STABLE_MAGNITUDE_SENSITIVE"
    else:
        adjudication = "PASS_RANKING_AND_MAGNITUDE_STABLE"

    body = {
        "schema_version": RECEIPT_SCHEMA,
        "pilot_id": config["pilot_id"],
        "decision": adjudication,
        "CP5_1_status": "DEVELOPMENT_PRESSURE_COVARIANCE_SCORED_NOT_COMPONENT_COMPLETE",
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "implementation_binding": config["implementation_binding"],
        "source_bindings": config["source_bindings"],
        "access_boundary": config["access_boundary"],
        "model_freeze": config["model_freeze"],
        "sample_summary": {
            "clusters": list(CLUSTERS),
            "pressure_rows": 54,
            "development_train_pressure_rows": sum(
                row["split"] == "development_train"
                for cluster in config["sample_freeze"]
                for row in cluster["pressure_rows"]
            ),
            "development_holdout_pressure_rows": sum(
                row["split"] == "development_holdout"
                for cluster in config["sample_freeze"]
                for row in cluster["pressure_rows"]
            ),
            "unscored_outer_boundary_rows": 8,
            "same_release_confirmation_rows": 0,
            "independent_rows": 0,
            "lensing_rows": 0,
        },
        "scoring_freeze": config["scoring_freeze"],
        "aggregates": aggregates,
        "per_cluster": per_cluster,
        "conditioning_summary": {
            "maximum_condition_number": max(
                row["condition_number"] for row in all_scores
            ),
            "maximum_inverse_identity_residual": max(
                row["inverse_identity_residual"] for row in all_scores
            ),
            "minimum_eigenvalue": min(row["minimum_eigenvalue"] for row in all_scores),
            "maximum_absolute_offdiagonal_correlation": max(
                row["maximum_absolute_offdiagonal_correlation"] for row in all_scores
            ),
        },
        "adjudication_thresholds": thresholds,
        "numerical_gates": numerical_gates,
        "robustness_gates": robustness_gates,
        "advanced_goal_evidence": {
            "CP5.1": "frozen_development_pressure_predictions_scored_with_reconstructed_released_correlation"
        },
        "completed_goal_evidence": {},
        "claims": config["claim_boundary"],
        "limitations": [
            "This is a pressure-only development audit; temperature, density, shared-calibration, background, beam/PSF-component, and cross-instrument covariance remain absent.",
            "The released pressure correlation is diagonal-calibrated to the Item 59 pressure errors and does not identify separate covariance-generating components.",
            "No formula, nuisance, comparator, row, radial cut, or threshold was selected after covariance scoring.",
            "A pass cannot establish an independent replication, a physical mechanism, an alternative to GR, or removal of dark matter.",
        ],
        "next_action": "Keep CP5.1 incomplete and acquire one complete already-exposed development response/background/deprojection packet before extending covariance claims.",
        "source_receipt_content_sha256": sources["RECONSTRUCTION_RECEIPT"][
            "content_sha256"
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterPressureCovariancePilotError("pilot receipt changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output: Any = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "CP5_1_status": receipt["CP5_1_status"],
            "aggregates": receipt["aggregates"],
            "robustness_gates": receipt["robustness_gates"],
            "claims": receipt["claims"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
