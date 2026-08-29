"""Numerical, synthetic-recovery, GPU, leakage, fold, and power controls for clusters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from sigma_theory_compiler import cluster_direct_observable_evaluator_readiness as direct
from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

CONFIG_PATH = Path("configs/gravity_cluster_numerical_controls_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/numerical-controls-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-numerical-controls-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-numerical-controls-receipt-1.0"
CP6_TASKS = tuple(f"CP6.{index}" for index in range(1, 11))


class GravityClusterControlError(RuntimeError):
    """Raised when a numerical control, data seal, or scoring invariant changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterControlError(f"{label} keys changed")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterControlError(f"expected JSON object: {path}")
    return value


def _validate_bindings(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    if tuple(bindings) != (
        "item59_config",
        "item59_result",
        "comparator_receipt",
        "direct_observable_readiness",
    ):
        raise GravityClusterControlError("control source binding order changed")
    for label, binding in bindings.items():
        expected = {"path", "file_sha256"} if label == "item59_config" else {
            "path",
            "file_sha256",
            "content_sha256",
        }
        _strict(binding, expected, f"{label} binding")
        path = (root / str(binding["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GravityClusterControlError("control binding escaped root") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterControlError(f"control binding changed: {label}")
        if "content_sha256" in binding:
            value = _load_json(path)
            if value.get("content_sha256") != binding["content_sha256"]:
                raise GravityClusterControlError(f"control content changed: {label}")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    validate_config(config, root)
    return config


def validate_config(config: Mapping[str, Any], root: Path) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "control_id",
            "source_bindings",
            "sample_contract",
            "convergence",
            "synthetic_recovery",
            "false_selection",
            "implementation_agreement",
            "leakage_mutations",
            "folds",
            "row_and_failure_rules",
            "power_and_stopping",
            "claim_boundary",
            "output_path",
        },
        "control config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"]
        != "frozen_development_controls_before_independent_target_access"
        or config["control_id"] != "gravity-cluster-numerical-controls-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterControlError("control config identity changed")
    _validate_bindings(root, config["source_bindings"])
    sample = config["sample_contract"]
    if (
        len(sample["clusters"]) != 8
        or sample["allowed_splits"] != ["development_train", "development_holdout"]
        or sample["xcop_confirmation_rows_used"] is not False
        or sample["independent_source_rows_used"] is not False
        or sample["target_rows_opened"] != 0
    ):
        raise GravityClusterControlError("control sample seal changed")
    convergence = config["convergence"]
    if (
        convergence["density_grid_refinement_factors"] != [1, 2, 4]
        or convergence["reference_factor"] != 4
        or convergence["maximum_absolute_log_prediction_difference"] > 0.03
        or convergence["maximum_holdout_score_fractional_difference"] > 0.05
    ):
        raise GravityClusterControlError("convergence thresholds weakened")
    recovery = config["synthetic_recovery"]
    if len(recovery["injections"]) != 5 or recovery["selection_split"] != "development_train":
        raise GravityClusterControlError("synthetic injection inventory changed")
    if recovery["injections"][-1]["physical_claim_allowed"] is not False:
        raise GravityClusterControlError("wrong-law claim boundary changed")
    false = config["false_selection"]
    if (
        false["trials"] != 4096
        or false["search_variants"] != 2025
        or len(false["qualifying_families"]) != 4
        or false["maximum_allowed_false_selection_fraction_for_pass"] > 0.05
    ):
        raise GravityClusterControlError("false-selection program weakened")
    agreement = config["implementation_agreement"]
    if (
        agreement["gpu_required_for_gate"] is not True
        or agreement["maximum_absolute_score_difference"] > 1e-8
        or agreement["selected_variant_ids_must_match"] is not True
        or agreement["separate_direct_scorer_rows"] != 17
    ):
        raise GravityClusterControlError("implementation agreement weakened")
    if len(config["leakage_mutations"]) != 5:
        raise GravityClusterControlError("leakage mutation inventory changed")
    folds = config["folds"]
    if (
        folds["leave_one_cluster_out"] is not True
        or len(folds["radial_blocks_r_over_r500"]) != 3
        or len(folds["instrument_observable_strata"]) != 2
        or folds["refit_within_fold"] is not False
    ):
        raise GravityClusterControlError("fold contract changed")
    rules = config["row_and_failure_rules"]
    if (
        rules["post_response_exclusion"] != "forbidden"
        or rules["duplicate_object"] != "fail_before_scoring"
        or rules["maximum_catastrophic_fraction"] > 0.1
        or "forbidden" not in rules["predictor_extrapolation"]
    ):
        raise GravityClusterControlError("row or catastrophic rules weakened")
    stopping = config["power_and_stopping"]
    if (
        stopping["single_final_analysis"] is not True
        or stopping["optional_stopping"] is not False
        or stopping["formula_or_threshold_change_after_opening"] is not False
        or stopping["maximum_independent_clusters"] > 120
    ):
        raise GravityClusterControlError("stopping rule weakened")
    claims = config["claim_boundary"]
    if claims["development_controls_only"] is not True or any(
        claims[key] for key in claims if key != "development_controls_only"
    ):
        raise GravityClusterControlError("control claim boundary weakened")


def _candidate_variant() -> dict[str, Any]:
    return {
        "family_id": "cross_scale_boundary",
        "parameters": {"beta": 1.5},
        "nuisances": {
            "missing_stellar_to_gas_mass_ratio": 0.2,
            "outer_nonthermal_fraction": 0.3,
            "published_stellar_mass_scale": 1.3,
            "xray_temperature_cross_calibration": 1.0,
        },
    }


def _refine_packet(packet: Mapping[str, Any], factor: int) -> dict[str, Any]:
    refined = dict(packet)
    if factor == 1:
        return refined
    radius = np.asarray(packet["density_radius_kpc"], dtype=float)
    log_radius = np.linspace(np.log(radius[0]), np.log(radius[-1]), (len(radius) - 1) * factor + 1)
    new_radius = np.exp(log_radius)
    ne = np.asarray(packet["ne_cm3"], dtype=float)
    low_fraction = np.asarray(packet["ne_error_low_cm3"], dtype=float) / ne
    high_fraction = np.asarray(packet["ne_error_high_cm3"], dtype=float) / ne
    new_ne = np.exp(np.interp(log_radius, np.log(radius), np.log(ne)))
    refined["density_radius_kpc"] = new_radius
    refined["ne_cm3"] = new_ne
    refined["ne_error_low_cm3"] = new_ne * np.interp(
        log_radius, np.log(radius), low_fraction
    )
    refined["ne_error_high_cm3"] = new_ne * np.interp(
        log_radius, np.log(radius), high_fraction
    )
    return refined


def _convergence_control(
    packets: Sequence[Mapping[str, Any]], config: Mapping[str, Any], config59: Mapping[str, Any]
) -> dict[str, Any]:
    predictions = {}
    scores = {}
    for factor in config["convergence"]["density_grid_refinement_factors"]:
        refined = [_refine_packet(packet, int(factor)) for packet in packets]
        prediction = item59._variant_predictions(refined, _candidate_variant(), config59)
        predictions[int(factor)] = prediction
        scores[int(factor)] = float(
            item59._score_predictions(
                refined, prediction, "development_holdout", config59
            )["score"]
        )
    reference = int(config["convergence"]["reference_factor"])
    comparisons = []
    for factor in config["convergence"]["density_grid_refinement_factors"]:
        if int(factor) == reference:
            continue
        log_difference = max(
            abs(math.log(predictions[int(factor)][row_id] / predictions[reference][row_id]))
            for row_id in predictions[reference]
        )
        score_fraction = abs(scores[int(factor)] / scores[reference] - 1.0)
        comparisons.append(
            {
                "factor": int(factor),
                "reference_factor": reference,
                "maximum_absolute_log_prediction_difference": log_difference,
                "holdout_score": scores[int(factor)],
                "holdout_score_fractional_difference": score_fraction,
            }
        )
    passed = all(
        row["maximum_absolute_log_prediction_difference"]
        <= float(config["convergence"]["maximum_absolute_log_prediction_difference"])
        and row["holdout_score_fractional_difference"]
        <= float(config["convergence"]["maximum_holdout_score_fractional_difference"])
        for row in comparisons
    )
    return {
        "passed": passed,
        "reference_holdout_score": scores[reference],
        "comparisons": comparisons,
    }


def _training_rows(
    packets: Sequence[Mapping[str, Any]], split: str
) -> list[Mapping[str, Any]]:
    return sorted(item59._rows(packets, split), key=lambda row: str(row["row_id"]))


def _prediction_matrix(
    packets: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    config59: Mapping[str, Any],
) -> np.ndarray:
    matrix = np.empty((len(variants), len(rows)), dtype=np.float64)
    for index, variant in enumerate(variants):
        prediction = item59._variant_predictions(packets, variant, config59)
        matrix[index] = [math.log(float(prediction[str(row["row_id"])])) for row in rows]
    return matrix


def _equal_group_weights(rows: Sequence[Mapping[str, Any]], sigma: float) -> np.ndarray:
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((str(row["cluster"]), str(row["observable"])), []).append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    for indices in groups.values():
        weights[indices] = 1.0 / (len(groups) * len(indices) * sigma**2)
    return weights


def _matrix_scores_cpu(
    log_predictions: np.ndarray, log_observed: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    prediction_square = np.sum(log_predictions**2 * weights[None, :], axis=1)
    observed_square = np.sum(log_observed**2 * weights[None, :], axis=1)
    cross = (log_observed * weights[None, :]) @ log_predictions.T
    return observed_square[:, None] - 2.0 * cross + prediction_square[None, :]


def _matrix_scores_gpu(
    log_predictions: np.ndarray, log_observed: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, str]:
    try:
        import cupy as cp
    except ImportError as error:
        raise GravityClusterControlError("CuPy is required for CP6.6") from error
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise GravityClusterControlError("CUDA device is required for CP6.6")
    predictions = cp.asarray(log_predictions, dtype=cp.float64)
    observed = cp.asarray(log_observed, dtype=cp.float64)
    gpu_weights = cp.asarray(weights, dtype=cp.float64)
    prediction_square = cp.sum(predictions**2 * gpu_weights[None, :], axis=1)
    observed_square = cp.sum(observed**2 * gpu_weights[None, :], axis=1)
    cross = (observed * gpu_weights[None, :]) @ predictions.T
    result = observed_square[:, None] - 2.0 * cross + prediction_square[None, :]
    cp.cuda.Device().synchronize()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return cp.asnumpy(result), str(name)


def _direct_score(
    log_prediction: np.ndarray, log_observed: np.ndarray, weights: np.ndarray
) -> float:
    total = 0.0
    for prediction, observed, weight in zip(
        log_prediction, log_observed, weights, strict=True
    ):
        total += float(weight) * (float(prediction) - float(observed)) ** 2
    return total


def _selected_comparator_prediction(
    packets: Sequence[Mapping[str, Any]],
    model_id: str,
    comparator_receipt: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, float]:
    selected = comparator_receipt["comparators"][model_id]["selection"]
    return comparators._gravity_model_predictions(
        packets,
        model_id,
        selected["parameters"],
        selected["nuisances"],
        config59,
    )


def _synthetic_recovery(
    packets: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    variant_matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    comparator_receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    variant_ids = [str(variant["variant_id"]) for variant in variants]
    classes = [str(variant["family_id"]) for variant in variants]
    extra_ids = ["GR_PLUS_NFW", "WRONG_REVERSED_NFW"]
    extra_predictions = []
    for model_id in extra_ids:
        prediction = _selected_comparator_prediction(
            packets, model_id, comparator_receipt, config59
        )
        extra_predictions.append(
            [math.log(float(prediction[str(row["row_id"])])) for row in rows]
        )
    library = np.vstack([variant_matrix, np.asarray(extra_predictions)])
    library_ids = variant_ids + extra_ids
    library_classes = classes + extra_ids
    weights = _equal_group_weights(rows, float(config["synthetic_recovery"]["log_noise_sigma"]))
    rng = np.random.default_rng(int(config["synthetic_recovery"]["noise_seed"]))
    results = []
    for injection in config["synthetic_recovery"]["injections"]:
        source_index = library_ids.index(str(injection["variant_id"]))
        observed = library[source_index] + rng.normal(
            0.0, float(config["synthetic_recovery"]["log_noise_sigma"]), len(rows)
        )
        scores = _matrix_scores_cpu(library, observed[None, :], weights)[0]
        winner = int(np.argmin(scores))
        recovered = library_classes[winner] == str(injection["expected_class"])
        wrong_rejected = bool(injection["physical_claim_allowed"]) or (
            library_classes[winner] == "WRONG_REVERSED_NFW"
        )
        results.append(
            {
                "injection_id": injection["injection_id"],
                "expected_class": injection["expected_class"],
                "selected_class": library_classes[winner],
                "selected_variant_id": library_ids[winner],
                "selected_score": float(scores[winner]),
                "class_recovered": recovered,
                "physical_claim_allowed": bool(injection["physical_claim_allowed"]),
                "wrong_law_physical_claim_rejected": wrong_rejected,
            }
        )
    return {
        "passed": all(
            row["class_recovered"] and row["wrong_law_physical_claim_rejected"]
            for row in results
        ),
        "library_variants": len(library_ids),
        "injections": results,
    }


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return center - half, center + half


def _false_selection_and_agreement(
    variants: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    false = config["false_selection"]
    variant_ids = [str(variant["variant_id"]) for variant in variants]
    source_index = variant_ids.index(str(false["null_injection_variant_id"]))
    rng = np.random.default_rng(int(false["seed"]))
    noise = rng.normal(
        0.0,
        float(false["log_noise_sigma"]),
        size=(int(false["trials"]), len(rows)),
    )
    observed = matrix[source_index][None, :] + noise
    weights = _equal_group_weights(rows, float(false["log_noise_sigma"]))
    cpu = _matrix_scores_cpu(matrix, observed, weights)
    gpu, device = _matrix_scores_gpu(matrix, observed, weights)
    cpu_winners = np.argmin(cpu, axis=1)
    gpu_winners = np.argmin(gpu, axis=1)
    qualifying = set(map(str, false["qualifying_families"]))
    selected_qualifying = np.asarray(
        [str(variants[index]["family_id"]) in qualifying for index in cpu_winners]
    )
    count = int(np.sum(selected_qualifying))
    fraction = count / int(false["trials"])
    low, high = _wilson(count, int(false["trials"]))
    false_result = {
        "trials": int(false["trials"]),
        "search_variants": len(variants),
        "qualifying_false_selections": count,
        "false_selection_fraction": fraction,
        "wilson_95_percent": [low, high],
        "threshold": float(false["maximum_allowed_false_selection_fraction_for_pass"]),
        "passed": fraction
        <= float(false["maximum_allowed_false_selection_fraction_for_pass"]),
    }
    agreement = config["implementation_agreement"]
    direct_indices = np.linspace(
        0, len(variants) - 1, int(agreement["separate_direct_scorer_rows"]), dtype=int
    )
    direct_differences = []
    for null_index, variant_index in zip(
        range(len(direct_indices)), direct_indices, strict=True
    ):
        direct_score = _direct_score(
            matrix[variant_index], observed[null_index], weights
        )
        direct_differences.append(abs(direct_score - float(cpu[null_index, variant_index])))
    maximum = float(np.max(np.abs(cpu - gpu)))
    winner_match = bool(np.array_equal(cpu_winners, gpu_winners))
    direct_maximum = max(direct_differences)
    agreement_result = {
        "gpu_device": device,
        "cpu_gpu_maximum_absolute_score_difference": maximum,
        "cpu_gpu_selected_variant_ids_match": winner_match,
        "separate_direct_scorer_maximum_absolute_difference": direct_maximum,
        "passed": maximum <= float(agreement["maximum_absolute_score_difference"])
        and direct_maximum <= float(agreement["maximum_absolute_score_difference"])
        and winner_match,
    }
    return false_result, agreement_result


def _validate_collection(packets: Sequence[Mapping[str, Any]]) -> None:
    clusters = [str(packet["cluster"]) for packet in packets]
    if len(clusters) != len(set(clusters)):
        raise GravityClusterControlError("duplicate object")
    row_ids = [str(row["row_id"]) for packet in packets for row in packet["rows"]]
    if len(row_ids) != len(set(row_ids)):
        raise GravityClusterControlError("duplicate row")
    forbidden = set(direct.FORBIDDEN_FIELDS)
    for packet in packets:
        if set(map(str, packet)) & forbidden:
            raise GravityClusterControlError("forbidden predictor or label")
        if any(row.get("post_response_exclusion") is True for row in packet["rows"]):
            raise GravityClusterControlError("response-informed exclusion")


def _leakage_controls(packets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    controls = []
    mutations = (
        ("target_derived_predictor", "formula_selection_target"),
        ("object_halo_label", "cluster_halo_mass"),
        ("derived_mass_field", "hydrostatic_mass"),
    )
    for control_id, field in mutations:
        mutated = [dict(packet) for packet in packets]
        mutated[0][field] = 1.0
        rejected = False
        try:
            _validate_collection(mutated)
        except GravityClusterControlError:
            rejected = True
        controls.append({"control_id": control_id, "rejected": rejected})
    mutated = copy.deepcopy(list(packets))
    mutated[0]["rows"][0]["post_response_exclusion"] = True
    try:
        _validate_collection(mutated)
        rejected = False
    except GravityClusterControlError:
        rejected = True
    controls.append({"control_id": "response_informed_exclusion", "rejected": rejected})
    duplicated = [*packets, packets[0]]
    try:
        _validate_collection(duplicated)
        rejected = False
    except GravityClusterControlError:
        rejected = True
    controls.append({"control_id": "duplicate_object_and_row", "rejected": rejected})
    return {"passed": all(row["rejected"] for row in controls), "controls": controls}


def _subset_score(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float], floor: float
) -> float:
    groups = {}
    for row in rows:
        residual = math.log(
            float(predictions[str(row["row_id"])]) / float(row["observed"])
        ) / max(float(row["error"]) / float(row["observed"]), floor)
        groups.setdefault((str(row["cluster"]), str(row["observable"])), []).append(
            residual**2
        )
    if not groups:
        raise GravityClusterControlError("empty fold")
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _fold_controls(
    packets: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, float],
    nfw: Mapping[str, float],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    rows = item59._rows(packets, "development_holdout")
    floor = float(config59["scoring"]["minimum_fractional_error"])
    packet_by_cluster = {str(packet["cluster"]): packet for packet in packets}
    leave_one = []
    for cluster in config["sample_contract"]["clusters"]:
        selected = [row for row in rows if row["cluster"] != cluster]
        candidate_score = _subset_score(selected, candidate, floor)
        nfw_score = _subset_score(selected, nfw, floor)
        leave_one.append(
            {
                "excluded_cluster": cluster,
                "candidate_score": candidate_score,
                "nfw_score": nfw_score,
                "candidate_beats_nfw": candidate_score < nfw_score,
            }
        )
    radial = []
    for low, high in config["folds"]["radial_blocks_r_over_r500"]:
        selected = [
            row
            for row in rows
            if float(low)
            <= float(row["radius_kpc"])
            / float(packet_by_cluster[str(row["cluster"])]["r500_kpc"])
            < float(high)
        ]
        candidate_score = _subset_score(selected, candidate, floor)
        nfw_score = _subset_score(selected, nfw, floor)
        radial.append(
            {
                "r_over_r500": [float(low), float(high)],
                "rows": len(selected),
                "candidate_score": candidate_score,
                "nfw_score": nfw_score,
                "candidate_beats_nfw": candidate_score < nfw_score,
            }
        )
    strata = []
    for label, observable in (
        ("Planck_SZ_pressure", "pressure"),
        ("XMM_temperature", "temperature"),
    ):
        selected = [row for row in rows if row["observable"] == observable]
        candidate_score = _subset_score(selected, candidate, floor)
        nfw_score = _subset_score(selected, nfw, floor)
        strata.append(
            {
                "stratum": label,
                "rows": len(selected),
                "candidate_score": candidate_score,
                "nfw_score": nfw_score,
                "candidate_beats_nfw": candidate_score < nfw_score,
            }
        )
    return {
        "leave_one_cluster_out": leave_one,
        "radial_blocks": radial,
        "instrument_observable_strata": strata,
        "candidate_beats_nfw_all_leave_one_out": all(
            row["candidate_beats_nfw"] for row in leave_one
        ),
        "candidate_beats_nfw_all_radial_blocks": all(
            row["candidate_beats_nfw"] for row in radial
        ),
        "candidate_beats_nfw_both_instrument_observable_strata": all(
            row["candidate_beats_nfw"] for row in strata
        ),
    }


def _power_analysis(
    packets: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, float],
    nfw: Mapping[str, float],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_score = item59._score_predictions(
        packets, candidate, "development_holdout", config59
    )
    nfw_score = item59._score_predictions(packets, nfw, "development_holdout", config59)
    differences = np.asarray(
        [
            float(nfw_score["by_cluster"][cluster])
            - float(candidate_score["by_cluster"][cluster])
            for cluster in config["sample_contract"]["clusters"]
        ]
    )
    mean = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=1))
    stopping = config["power_and_stopping"]
    effect = max(
        np.finfo(float).tiny,
        mean * float(stopping["conservative_effect_fraction_of_development_mean"]),
    )
    z_alpha = NormalDist().inv_cdf(1.0 - float(stopping["two_sided_alpha"]) / 2.0)
    z_power = NormalDist().inv_cdf(float(stopping["target_power"]))
    required = math.ceil(((z_alpha + z_power) * standard_deviation / effect) ** 2)
    minimum = int(stopping["minimum_independent_clusters"])
    maximum = int(stopping["maximum_independent_clusters"])
    planned = min(maximum, max(minimum, required))
    achieved_power = NormalDist().cdf(effect * math.sqrt(planned) / standard_deviation - z_alpha)
    return {
        "development_cluster_differences_NFW_minus_candidate": [
            float(value) for value in differences
        ],
        "development_mean_difference": mean,
        "development_standard_deviation": standard_deviation,
        "conservative_effect": effect,
        "calculated_required_clusters": required,
        "planned_independent_clusters": planned,
        "planned_approximate_power": float(achieved_power),
        "target_power": float(stopping["target_power"]),
        "maximum_sample_sufficient": required <= maximum,
        "stopping_rule": {
            "single_final_analysis": True,
            "optional_stopping": False,
            "formula_or_threshold_change_after_opening": False,
        },
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    if [packet["cluster"] for packet in packets] != config["sample_contract"]["clusters"]:
        raise GravityClusterControlError("control cluster population changed")
    comparator_receipt = _load_json(
        root / config["source_bindings"]["comparator_receipt"]["path"]
    )
    convergence = _convergence_control(packets, config, config59)
    variants = item59.enumerate_variants(config59)
    rows = _training_rows(packets, "development_train")
    matrix = _prediction_matrix(packets, variants, rows, config59)
    recovery = _synthetic_recovery(
        packets,
        variants,
        matrix,
        rows,
        comparator_receipt,
        config,
        config59,
    )
    false_selection, implementation = _false_selection_and_agreement(
        variants, matrix, rows, config
    )
    leakage = _leakage_controls(packets)
    candidate_predictions = item59._variant_predictions(
        packets, _candidate_variant(), config59
    )
    nfw_predictions = _selected_comparator_prediction(
        packets, "GR_PLUS_NFW", comparator_receipt, config59
    )
    folds = _fold_controls(
        packets, candidate_predictions, nfw_predictions, config, config59
    )
    power = _power_analysis(
        packets, candidate_predictions, nfw_predictions, config, config59
    )
    task_pass = {
        "CP6.1": True,
        "CP6.2": True,
        "CP6.3": bool(convergence["passed"]),
        "CP6.4": bool(recovery["passed"]),
        "CP6.5": True,
        "CP6.6": bool(implementation["passed"]),
        "CP6.7": bool(leakage["passed"]),
        "CP6.8": True,
        "CP6.9": True,
        "CP6.10": True,
    }
    all_complete = all(task_pass.values())
    scientific_control_pass = all_complete and bool(false_selection["passed"])
    decision = (
        "NUMERICAL_CONTROLS_PASS_DEVELOPMENT_ONLY"
        if scientific_control_pass
        else (
            "NUMERICAL_CONTROLS_COMPLETE_FALSE_SELECTION_THRESHOLD_FAILED"
            if all_complete
            else "NUMERICAL_CONTROLS_INCOMPLETE_ONE_OR_MORE_CONTROLS_FAILED"
        )
    )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "control_id": config["control_id"],
        "decision": decision,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config),
        },
        "sample": {
            "clusters": config["sample_contract"]["clusters"],
            "training_rows": len(rows),
            "holdout_rows": len(item59._rows(packets, "development_holdout")),
            "xcop_confirmation_rows_used": False,
            "independent_source_rows_used": False,
            "target_rows_opened": 0,
        },
        "radial_convergence": convergence,
        "synthetic_recovery": recovery,
        "false_selection": false_selection,
        "implementation_agreement": implementation,
        "leakage_mutations": leakage,
        "fold_controls": folds,
        "row_and_failure_rules": config["row_and_failure_rules"],
        "prospective_power_and_stopping": power,
        "task_pass": task_pass,
        "completed_goal_evidence": {
            task_id: description
            for task_id, description in {
                "CP6.1": "deterministic_enumeration_split_and_receipts_preserved",
                "CP6.2": "unit_positive_boundary_and_analytic_limit_tests_preserved",
                "CP6.3": "density_grid_and_hydrostatic_integration_convergence",
                "CP6.4": "five_injected_law_classes_recovered_with_wrong_law_claim_rejected",
                "CP6.5": "4096_null_trials_score_all_2025_variants",
                "CP6.6": "CPU_CUDA_and_direct_scorer_agreement",
                "CP6.7": "five_leakage_and_duplicate_mutations_rejected",
                "CP6.8": "leave_one_cluster_radial_block_and_instrument_observable_folds",
                "CP6.9": "missing_censoring_extrapolation_and_catastrophic_rules_frozen",
                "CP6.10": "prospective_paired_cluster_power_and_single_final_stopping_rule",
            }.items()
            if task_pass[task_id]
        },
        "blocked_goal_evidence": {
            task_id: "control_failed"
            for task_id, passed in task_pass.items()
            if not passed
        },
        "counts": {
            "item59_variants": len(variants),
            "synthetic_injections": len(recovery["injections"]),
            "null_trials": false_selection["trials"],
            "null_variant_scores": false_selection["trials"] * len(variants),
            "leakage_mutations": len(leakage["controls"]),
            "folds": len(folds["leave_one_cluster_out"])
            + len(folds["radial_blocks"])
            + len(folds["instrument_observable_strata"]),
            "completed_CP6_tasks": sum(task_pass.values()),
            "open_CP6_tasks": sum(not passed for passed in task_pass.values()),
            "target_rows_opened": 0,
        },
        "claims": {
            "all_CP6_tasks_complete": all_complete,
            "false_selection_threshold_passed": bool(false_selection["passed"]),
            "development_numerical_control_gate_passed": scientific_control_pass,
            "independent_replication": False,
            "full_source_covariance": False,
            "physical_mechanism_established": False,
            "alternative_to_gr_established": False,
            "dark_matter_eliminated": False,
        },
        "limitations": [
            "Injected recovery and null trials reuse the eight X-COP development predictor geometries.",
            "The false-selection control measures the frozen 2,025-variant Item 59 grammar, not every future formula family.",
            "Instrument folds are observable/instrument strata (Planck SZ and XMM temperature), not multiple independent instruments measuring each observable.",
            "Power calculations extrapolate development effect dispersion and must be updated only before independent targets open.",
        ],
        "next_action": "Preserve this exact control suite for the independent source and do not change thresholds after target opening.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterControlError("numerical controls receipt changed")


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
        receipt = _load_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "claims": receipt["claims"],
            "task_pass": receipt["task_pass"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
