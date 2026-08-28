"""Nested whole-galaxy experiment for gravity roadmap Item 5 attempt 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import gravity_item5_pressure_support as source
from .sigma_core import canonical_json_bytes, canonical_sha256


class GravityItem5PressureSupportExperimentError(RuntimeError):
    """Raised when the frozen Item 5 experiment or receipt drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem5PressureSupportExperimentError("non-finite metric")
    return f"{float(value):.12e}"


def _load_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    with (root / config["feature_output"]).open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    integer = {"radius_index"}
    rows: list[dict[str, Any]] = []
    for input_row in raw:
        rows.append(
            {
                key: (
                    str(value)
                    if key in {"galaxy", "pressure_stratum"}
                    else int(value)
                    if key in integer
                    else float(value)
                )
                for key, value in input_row.items()
            }
        )
    keys = {(str(row["galaxy"]), int(row["radius_index"])) for row in rows}
    if len(keys) != len(rows):
        raise GravityItem5PressureSupportExperimentError("duplicate radial row")
    return sorted(rows, key=lambda row: (str(row["galaxy"]), int(row["radius_index"])))


def fold_assignments(rows: Sequence[Mapping[str, Any]], *, salt: str, folds: int) -> dict[str, int]:
    galaxies = sorted({str(row["galaxy"]) for row in rows})
    ordered = sorted(
        galaxies,
        key=lambda galaxy: hashlib.sha256(f"{salt}|{galaxy}".encode()).hexdigest(),
    )
    return {galaxy: ordinal % folds for ordinal, galaxy in enumerate(ordered)}


def _matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(rows), 0), dtype=np.float64)
    result = np.asarray(
        [[float(row[feature]) for feature in features] for row in rows], dtype=np.float64
    )
    if np.any(~np.isfinite(result)):
        raise GravityItem5PressureSupportExperimentError("non-finite model matrix")
    return result


def _row_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["galaxy"]) for row in rows)
    galaxies = len(counts)
    return np.asarray(
        [1.0 / (galaxies * counts[str(row["galaxy"])]) for row in rows],
        dtype=np.float64,
    )


def _fit_ridge(
    rows: Sequence[Mapping[str, Any]],
    *,
    features: Sequence[str],
    response: str,
    alpha: float,
) -> dict[str, Any]:
    y = np.asarray([float(row[response]) for row in rows])
    raw = _matrix(rows, features)
    weights = _row_weights(rows)
    if raw.shape[1]:
        means = np.average(raw, axis=0, weights=weights)
        scales = np.sqrt(np.average((raw - means) ** 2, axis=0, weights=weights))
        scales = np.where(scales > 1.0e-12, scales, 1.0)
        standardized = (raw - means) / scales
    else:
        means = np.empty(0)
        scales = np.empty(0)
        standardized = raw
    design = np.column_stack((np.ones(len(rows)), standardized))
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_y = y * root_weight
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_y,
    )
    return {
        "coefficients": coefficients,
        "features": tuple(features),
        "means": means,
        "scales": scales,
    }


def _predict(fit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    raw = _matrix(rows, fit["features"])
    standardized = (
        (raw - np.asarray(fit["means"])) / np.asarray(fit["scales"]) if raw.shape[1] else raw
    )
    return np.column_stack((np.ones(len(rows)), standardized)) @ np.asarray(fit["coefficients"])


def _equal_galaxy_mse(
    rows: Sequence[Mapping[str, Any]], predicted: Sequence[float], response: str
) -> float:
    by_galaxy: defaultdict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, predicted, strict=True):
        by_galaxy[str(row["galaxy"])].append((float(row[response]) - float(value)) ** 2)
    return float(np.mean([np.mean(values) for values in by_galaxy.values()]))


def _inner_score(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    *,
    excluded_outer_fold: int,
    features: Sequence[str],
    response: str,
    alpha: float,
) -> tuple[float, int]:
    scores: list[float] = []
    fits = 0
    for validation_fold in sorted(set(assignments.values()) - {excluded_outer_fold}):
        training = [
            row
            for row in rows
            if assignments[str(row["galaxy"])] not in {excluded_outer_fold, validation_fold}
        ]
        validation = [row for row in rows if assignments[str(row["galaxy"])] == validation_fold]
        if not training or not validation:
            continue
        fit = _fit_ridge(training, features=features, response=response, alpha=alpha)
        scores.append(_equal_galaxy_mse(validation, _predict(fit, validation), response))
        fits += 1
    if not scores:
        raise GravityItem5PressureSupportExperimentError("empty inner evaluation")
    return float(np.mean(scores)), fits


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[tuple[str, int], float], response: str
) -> dict[str, Any]:
    def calculate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        predicted = np.asarray(
            [predictions[(str(row["galaxy"]), int(row["radius_index"]))] for row in subset]
        )
        observed = np.asarray([float(row[response]) for row in subset])
        residual = observed - predicted
        mse = _equal_galaxy_mse(subset, predicted, response)
        galaxy_means = {
            galaxy: float(
                np.mean([float(row[response]) for row in subset if row["galaxy"] == galaxy])
            )
            for galaxy in {str(row["galaxy"]) for row in subset}
        }
        denominator = _equal_galaxy_mse(
            subset,
            [galaxy_means[str(row["galaxy"])] for row in subset],
            response,
        )
        global_mean = float(np.mean(observed))
        total = float(np.sum((observed - global_mean) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 0.0
        return {
            "equal_galaxy_mean_squared_error": _metric(mse),
            "mean_absolute_error": _metric(float(np.mean(np.abs(residual)))),
            "objects": len({str(row["galaxy"]) for row in subset}),
            "radial_rows": len(subset),
            "r2": _metric(r2),
            "within_galaxy_variance_reference_mse": _metric(denominator),
        }

    strata = sorted({str(row["pressure_stratum"]) for row in rows})
    return {
        "overall": calculate(rows),
        "by_pressure_stratum": {
            stratum: calculate([row for row in rows if str(row["pressure_stratum"]) == stratum])
            for stratum in strata
        },
    }


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    response: str = "log10_target_velocity",
    selection_pool: str = "all",
) -> dict[str, Any]:
    cv = config["cross_validation"]
    folds = int(cv["outer_folds"])
    assignments = fold_assignments(rows, salt=str(cv["fold_salt"]), folds=folds)
    models = list(config["model_families"])
    penalties = [float(value) for value in cv["ridge_penalties"]]
    by_model: dict[str, dict[tuple[str, int], float]] = {str(model["id"]): {} for model in models}
    selected_predictions: dict[tuple[str, int], float] = {}
    ledger: list[dict[str, Any]] = []
    inner_fits = 0
    final_fits = 0
    for fold in range(folds):
        training = [row for row in rows if assignments[str(row["galaxy"])] != fold]
        heldout = [row for row in rows if assignments[str(row["galaxy"])] == fold]
        candidates: list[dict[str, Any]] = []
        for order, model in enumerate(models):
            features = tuple(str(value) for value in model["features"])
            model_candidates: list[dict[str, Any]] = []
            for alpha in penalties if features else penalties[:1]:
                score, fits = _inner_score(
                    rows,
                    assignments,
                    excluded_outer_fold=fold,
                    features=features,
                    response=response,
                    alpha=alpha,
                )
                inner_fits += fits
                candidate = {
                    "alpha": alpha,
                    "features": features,
                    "inner_mse": score,
                    "model": model,
                    "order": order,
                }
                candidates.append(candidate)
                model_candidates.append(candidate)
            best = min(
                model_candidates,
                key=lambda value: (value["inner_mse"], len(value["features"]), value["alpha"]),
            )
            fit = _fit_ridge(
                training,
                features=features,
                response=response,
                alpha=float(best["alpha"]),
            )
            final_fits += 1
            for row, value in zip(heldout, _predict(fit, heldout), strict=True):
                by_model[str(model["id"])][(str(row["galaxy"]), int(row["radius_index"]))] = float(
                    value
                )
        selectable = (
            [value for value in candidates if bool(value["model"]["qualifying"])]
            if selection_pool == "qualifying"
            else candidates
        )
        selected = min(
            selectable,
            key=lambda value: (
                value["inner_mse"],
                len(value["features"]),
                value["order"],
                value["alpha"],
            ),
        )
        fit = _fit_ridge(
            training,
            features=selected["features"],
            response=response,
            alpha=float(selected["alpha"]),
        )
        final_fits += 1
        for row, value in zip(heldout, _predict(fit, heldout), strict=True):
            selected_predictions[(str(row["galaxy"]), int(row["radius_index"]))] = float(value)
        ledger.append(
            {
                "alpha": _metric(float(selected["alpha"])),
                "features": list(selected["features"]),
                "fold": fold,
                "heldout_galaxies": sorted({str(row["galaxy"]) for row in heldout}),
                "inner_mse": _metric(float(selected["inner_mse"])),
                "model_id": str(selected["model"]["id"]),
                "qualifying": bool(selected["model"]["qualifying"]),
            }
        )
    return {
        "compute_counts": {"inner_ridge_fits": inner_fits, "final_ridge_fits": final_fits},
        "fold_ledger": ledger,
        "model_metrics": {
            model_id: _metrics(rows, predictions, response)
            for model_id, predictions in by_model.items()
        },
        "selected_metrics": _metrics(rows, selected_predictions, response),
        "selection_pool": selection_pool,
    }


def _internal_positive_control(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mse(field: str) -> float:
        by_galaxy: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            observed = math.log10(float(row["internal_vc_km_s"]))
            predicted = (
                math.log10(float(row["vrot_km_s"]))
                if field == "vrot_km_s"
                else float(row["log10_v_classical"])
            )
            by_galaxy[str(row["galaxy"])].append((observed - predicted) ** 2)
        return float(np.mean([np.mean(values) for values in by_galaxy.values()]))

    rotation = mse("vrot_km_s")
    classical = mse("log10_v_classical")
    return {
        "classical_mse": _metric(classical),
        "classical_minus_rotation_mse": _metric(classical - rotation),
        "improves": classical < rotation,
        "interpretation": "retrospective pipeline self-consistency only; not independent evidence",
        "rotation_only_mse": _metric(rotation),
    }


def _permute_galaxy_blocks(
    rows: Sequence[Mapping[str, Any]], *, salt: str, ordinal: int
) -> list[dict[str, Any]]:
    galaxies = sorted({str(row["galaxy"]) for row in rows})
    seed = int.from_bytes(hashlib.sha256(f"{salt}|{ordinal}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    sources = list(rng.permutation(galaxies))
    by_galaxy = {
        galaxy: sorted(
            [dict(row) for row in rows if str(row["galaxy"]) == galaxy],
            key=lambda row: float(row["log10_radius"]),
        )
        for galaxy in galaxies
    }
    result: list[dict[str, Any]] = []
    for destination, source_galaxy in zip(galaxies, sources, strict=True):
        destination_rows = by_galaxy[destination]
        source_values = np.asarray(
            [float(row["log10_target_velocity"]) for row in by_galaxy[str(source_galaxy)]]
        )
        source_axis = np.linspace(0.0, 1.0, len(source_values))
        destination_axis = np.linspace(0.0, 1.0, len(destination_rows))
        for row, value in zip(
            destination_rows,
            np.interp(destination_axis, source_axis, source_values),
            strict=True,
        ):
            row["log10_target_velocity"] = float(value)
            result.append(row)
    return sorted(result, key=lambda row: (str(row["galaxy"]), int(row["radius_index"])))


def _permutation_test(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    observed = _evaluate(rows, config, selection_pool="qualifying")
    controls = (
        observed["model_metrics"]["rotation_pipeline_calibration"],
        observed["model_metrics"]["classical_local_pressure"],
    )
    control_mse = min(
        float(value["overall"]["equal_galaxy_mean_squared_error"]) for value in controls
    )
    candidate_mse = float(
        observed["selected_metrics"]["overall"]["equal_galaxy_mean_squared_error"]
    )
    observed_improvement = control_mse - candidate_mse
    count = int(config["cross_validation"]["permutation_count"])
    salt = str(config["cross_validation"]["permutation_salt"])
    null_values: list[float] = []
    for ordinal in range(count):
        permuted = _permute_galaxy_blocks(rows, salt=salt, ordinal=ordinal)
        evaluated = _evaluate(permuted, config, selection_pool="qualifying")
        null_control = min(
            float(evaluated["model_metrics"][model]["overall"]["equal_galaxy_mean_squared_error"])
            for model in ("rotation_pipeline_calibration", "classical_local_pressure")
        )
        null_candidate = float(
            evaluated["selected_metrics"]["overall"]["equal_galaxy_mean_squared_error"]
        )
        null_values.append(null_control - null_candidate)
    exceedances = sum(value >= observed_improvement for value in null_values)
    return {
        "exceedances": exceedances,
        "null_improvement_max": _metric(max(null_values)),
        "null_improvement_median": _metric(float(np.median(null_values))),
        "observed_improvement": _metric(observed_improvement),
        "permutations": count,
        "p_value": _metric((1 + exceedances) / (1 + count)),
        "statistic": "best rotation/classical control MSE minus qualifying-selector MSE",
    }


def _gate_checks(
    *,
    primary: Mapping[str, Any],
    qualifying: Mapping[str, Any],
    positive: Mapping[str, Any],
    permutation: Mapping[str, Any],
    extraction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, bool]:
    selected = primary["selected_metrics"]
    candidate = qualifying["selected_metrics"]
    rotation = qualifying["model_metrics"]["rotation_pipeline_calibration"]
    classical = qualifying["model_metrics"]["classical_local_pressure"]
    strata = ("low", "high")

    def candidate_beats(
        control: Mapping[str, Any], location: str, stratum: str | None = None
    ) -> bool:
        left = candidate[location] if stratum is None else candidate[location][stratum]
        right = control[location] if stratum is None else control[location][stratum]
        return float(left["equal_galaxy_mean_squared_error"]) < float(
            right["equal_galaxy_mean_squared_error"]
        )

    return {
        "all_11_exploration_galaxies_pass_frozen_quality": (
            extraction["decision"] == "PASS_ITEM5_EXPLORATION_REPRESENTATION_QUALITY"
            and int(extraction["counts"]["quality_passing_galaxies"]) == 11
        ),
        "classical_formula_improves_Iorio_internal_positive_control_over_rotation_only": bool(
            positive["improves"]
        ),
        "selected_model_qualifying_in_every_fold": all(
            bool(row["qualifying"]) for row in primary["fold_ledger"]
        ),
        "primary_r2_positive_overall_and_in_each_frozen_pressure_stratum": (
            float(selected["overall"]["r2"]) > 0
            and all(float(selected["by_pressure_stratum"][value]["r2"]) > 0 for value in strata)
        ),
        "qualifying_model_beats_rotation_and_classical_controls_overall_and_in_each_stratum": (
            all(candidate_beats(control, "overall") for control in (rotation, classical))
            and all(
                candidate_beats(control, "by_pressure_stratum", stratum)
                for control in (rotation, classical)
                for stratum in strata
            )
        ),
        "galaxy_level_permutation_p_at_most_frozen_threshold": (
            float(permutation["p_value"])
            <= float(config["exploration_admission"]["galaxy_level_permutation_p_at_most"])
        ),
        "reserved_confirmation_untouched": (
            int(extraction["counts"]["reserved_confirmation_target_accesses"]) == 0
            and int(extraction["counts"]["reserved_confirmation_predictor_member_accesses"]) == 0
        ),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = source.load_config(root)
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    source.validate_sample_manifest(sample, config)
    source_path = root / config["source_manifest_output"]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source.validate_source_manifest(source_manifest, sample=sample)
    extraction_path = root / config["extraction_summary_output"]
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    copy = dict(extraction)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem5PressureSupportExperimentError("extraction hash changed")
    rows = _load_rows(root, config)
    primary = _evaluate(rows, config, selection_pool="all")
    qualifying = _evaluate(rows, config, selection_pool="qualifying")
    positive = _internal_positive_control(rows)
    permutation = _permutation_test(rows, config)
    gates = _gate_checks(
        primary=primary,
        qualifying=qualifying,
        positive=positive,
        permutation=permutation,
        extraction=extraction,
        config=config,
    )
    decision = (
        "PASS_ITEM5_PRESSURE_SUPPORT_EXPLORATION_REQUIRES_AUTHORIZATION"
        if all(gates.values())
        else (
            "INCONCLUSIVE_ITEM5_PRESSURE_SUPPORT_QUALITY_GATE"
            if not gates["all_11_exploration_galaxies_pass_frozen_quality"]
            else "REJECT_ITEM5_PRESSURE_SUPPORT_EXPLORATION"
        )
    )
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-roadmap-item5-pressure-support-receipt-1.0",
        "goal": config["goal"],
        "decision": decision,
        "hypothesis": config["scientific_contract"]["hypothesis"],
        "creativity": {
            "label": config["scientific_contract"]["creativity_label"],
            "known_components": config["scientific_contract"]["known_components"],
            "nonqualifying_rewrites": config["scientific_contract"]["nonqualifying_rewrites"],
            "historical_novelty_established": False,
        },
        "preregistration": {
            "git_commit": source.FREEZE_COMMIT,
            "archive_member_contents_read_before_commit": 0,
            "target_rows_read_before_commit": 0,
        },
        "data_lineage": {
            "predictors": "Iorio et al. 2017 3D-BAROLO raw rotation, H I dispersion, and H I surface density",
            "primary_target": "Oh et al. 2015 independently reduced corrected circular speed",
            "same_telescope_cubes": True,
            "different_reduction_pipelines": True,
            "published_Iorio_Vc_used_as_predictor": False,
            "dark_halo_target_used": False,
            "lensing_target_used": False,
        },
        "response": {
            "primary_all_family_selector": {
                "fold_ledger": primary["fold_ledger"],
                "selected_metrics": primary["selected_metrics"],
                "model_metrics": primary["model_metrics"],
            },
            "qualifying_only_selector": {
                "fold_ledger": qualifying["fold_ledger"],
                "selected_metrics": qualifying["selected_metrics"],
                "model_metrics": qualifying["model_metrics"],
            },
            "internal_positive_control": positive,
            "permutation_test": permutation,
        },
        "gate_checks": gates,
        "counterexamples": {
            "quality_failures": extraction["failures"],
            "failed_gate_names": [name for name, passed in gates.items() if not passed],
            "selected_nonqualifying_folds": [
                int(row["fold"]) for row in primary["fold_ledger"] if not row["qualifying"]
            ],
        },
        "counts": {
            "candidate_model_families": len(config["model_families"]),
            "candidate_model_ridge_cells": sum(
                len(config["cross_validation"]["ridge_penalties"]) if model["features"] else 1
                for model in config["model_families"]
            ),
            "exploration_selected": 11,
            "exploration_quality_passing": int(extraction["counts"]["quality_passing_galaxies"]),
            "exploration_quality_failures": int(extraction["counts"]["quality_failures"]),
            "radial_rows": len(rows),
            "reserved_confirmation_galaxies": 5,
            "reserved_confirmation_target_accesses": 0,
            "outer_folds": int(config["cross_validation"]["outer_folds"]),
            "inner_ridge_fits_primary": int(primary["compute_counts"]["inner_ridge_fits"]),
            "inner_ridge_fits_qualifying": int(qualifying["compute_counts"]["inner_ridge_fits"]),
            "permutations": int(permutation["permutations"]),
            "paid_model_calls": 0,
            "direct_lensing_likelihood_evaluations": 0,
        },
        "limitations": config["scientific_contract"]["not_claimed"],
        "claims": {
            "alternative_to_gr_established": False,
            "historical_novelty_established": False,
            "independent_telescope_confirmation_completed": False,
            "reserved_confirmation_opened": False,
            "roadmap_item_5_complete": False,
            "unified_all_support_types_established": False,
        },
        "next_action": (
            "Retain all six representation failures and do not open the five reserved "
            "galaxies. A second Item 5 attempt must preregister a physically smooth "
            "pressure representation on a materially different real source before "
            "access, rather than retuning finite differences on these opened curves."
        ),
        "source_bindings": {
            "config": {
                "path": source.CONFIG_PATH,
                "sha256": _sha256_file(root / source.CONFIG_PATH),
            },
            "sample_manifest": {
                "path": config["sample_manifest_output"],
                "sha256": _sha256_file(sample_path),
            },
            "source_manifest": {
                "path": config["source_manifest_output"],
                "sha256": _sha256_file(source_path),
            },
            "extraction_summary": {
                "path": config["extraction_summary_output"],
                "sha256": _sha256_file(extraction_path),
            },
            "feature_table": {
                "path": config["feature_output"],
                "sha256": _sha256_file(root / config["feature_output"]),
            },
            "source_code": {
                "path": str(Path(source.__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(source.__file__)),
            },
            "experiment_code": {
                "path": str(Path(__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(__file__)),
            },
            "test": {
                "path": "tests/test_gravity_item5_pressure_support.py",
                "sha256": _sha256_file(root / "tests/test_gravity_item5_pressure_support.py"),
            },
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem5PressureSupportExperimentError("receipt hash changed")
    if value.get("decision") not in {
        "PASS_ITEM5_PRESSURE_SUPPORT_EXPLORATION_REQUIRES_AUTHORIZATION",
        "INCONCLUSIVE_ITEM5_PRESSURE_SUPPORT_QUALITY_GATE",
        "REJECT_ITEM5_PRESSURE_SUPPORT_EXPLORATION",
    }:
        raise GravityItem5PressureSupportExperimentError("unknown decision")
    if int(value["counts"]["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem5PressureSupportExperimentError("confirmation was accessed")
    if any(bool(claim) for claim in value["claims"].values()):
        raise GravityItem5PressureSupportExperimentError("receipt contains overclaim")
    if str(value["decision"]).startswith("PASS_") and not all(
        bool(value) for value in value["gate_checks"].values()
    ):
        raise GravityItem5PressureSupportExperimentError("false PASS")
    for binding in value["source_bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = root / path
        if _sha256_file(path) != binding["sha256"]:
            raise GravityItem5PressureSupportExperimentError(f"binding changed: {path}")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = source.load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path = root / config["output"]
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = source.load_config(root)
    if args.check:
        path = root / config["output"]
        stored = json.loads(path.read_text(encoding="utf-8"))
        validate_receipt(stored, root=root)
        if build_receipt(root) != stored:
            raise GravityItem5PressureSupportExperimentError("receipt drifted")
        return 0
    print(write_receipt(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
