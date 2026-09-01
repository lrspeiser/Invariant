from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_rg_things_matched_pair_2d_pixel_score_v1 as score
from sigma_theory_compiler import (
    open_gravity_rg_things_matched_pair_2d_source_predictions_v1 as predictions,
)

CONFIG_PATH = Path("configs/open_gravity_rg_things_matched_pair_2d_comparator_diagnostics_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_matched_pair_2d_comparator_diagnostics_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_matched_pair_2d_comparator_diagnostics_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-matched-pair-2d-comparator-diagnostics-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-comparator-diagnostics-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-things-matched-pair-2d-comparator-diagnostics-receipt-1.0"
)
_OBJECTS = ("NGC2976", "NGC4214")
_NEWTON = "NEWTON_3D_DST"
_RG = "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
_RAR = "EMPIRICAL_RAR_MCGaugh_2016_FIXED"
_MODELS = (_NEWTON, _RG, _RAR)
_CONFIG_RAW_SHA256 = "6fb9a5af369e7dcf45c3adacf00831e02a8d7f05bedca0da7fb5beb4f1052e27"
_CONFIG_CONTENT_SHA256 = "01075bbf7142b6ac35ef904a62e04281ea60df2780ec1d5548c2a19baff0ff17"
_MODULE_SEMANTIC_SHA256 = "b8ab5a2b4f0d406d1dae8ee6070a0f5f126ab161e9f833ac4b708c4842fbecaa"
_TEST_RAW_SHA256 = "ab161b0ddfb662e69385ada58ae4b435ed64612a9384dbaa1f1e839b8fcfba76"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class ComparatorDiagnosticError(RuntimeError):
    """Raised when comparator or sensitivity evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparatorDiagnosticError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        b'_MODULE_SEMANTIC_SHA256 = "' + b"0" * 64 + b'"', path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path == _ROOT or _ROOT in path.parents, "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparatorDiagnosticError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_POST_PRIMARY_SCORE_EXPLORATORY_COMPARATOR_DIAGNOSTICS",
        "status changed",
    )
    audit = config["audit_disclosure"]
    for key in (
        "primary_newton_vs_rg_pixel_score_was_opened_before_this_packet",
        "rar_formula_and_gdagger_are_published_fixed_values_not_fit_here",
        "all_sensitivity_cells_are_exploratory_not_preregistered_confirmation",
        "no_cell_may_replace_or erase_the_sealed_primary_result",
        "one_failure_never_prunes_family",
    ):
        _require(audit[key] is True, f"audit disclosure changed: {key}")
    admission = config["source_admission"]
    for key in (
        "real_public_sources_and_responses_bound_by_predecessor",
        "primary_measurement_and_method_papers_bound_by_predecessor",
        "rar_primary_paper_bound",
        "target_free_operator_and_projection_benchmarks_bound_by_predecessor",
        "newtonian_control_required",
    ):
        _require(admission[key] is True, f"source admission changed: {key}")
    _require(admission["missing_source_disposition"] == "SOURCE_BLOCKED", "source rule changed")
    _require(admission["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY", "paper rule changed")
    _require(
        admission["model_lifted_vertical_structure_disposition"] == "MODEL_LIFTED_2P5D",
        "2.5D label changed",
    )
    _require(admission["general_3d_validation_allowed"] is False, "3D overclaim")
    binding = config["score_binding"]
    _require(
        binding["sealed_primary_ngc2976_rg_fractional_improvement"] == 0.1533607170491626,
        "sealed primary changed",
    )
    _require(
        binding["sealed_stress_ngc4214_rg_fractional_improvement"] == -0.010917240981521815,
        "sealed stress result changed",
    )
    rar = config["rar_comparator"]
    _require(rar["id"] == _RAR, "RAR ID changed")
    _require(rar["g_dagger_m_s2"] == 1.2e-10, "g_dagger changed")
    _require(rar["field_equation_claim"] is False, "RAR field-equation overclaim")
    _require(rar["new_theory_claim"] is False, "RAR novelty overclaim")
    _require(rar["parameter_fit_to_things"] is False, "RAR tuning enabled")
    contract = config["common_score_contract"]
    _require(contract["models"] == list(_MODELS), "models changed")
    _require(contract["minimum_dispersion_scale_m_s"] == 3000.0, "dispersion floor changed")
    _require(contract["response_tuning_calls"] == 0, "response tuning enabled")
    _require(contract["p_values_computed"] is False, "p-value overclaim")
    grid = config["exploratory_sensitivity_grid"]
    _require(
        grid["source_tangential_ratio_maximum"] == [None, 1.0, 0.5, 0.25, 0.1],
        "tangential grid changed",
    )
    _require(
        grid["radial_bins_kpc"] == [[0.5, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, 15.0]],
        "radial grid changed",
    )
    _require(grid["subcell_systemic_offsets_refit"] is False, "subcell refit enabled")
    boundary = config["scientific_boundary"]
    for key in ("network_calls", "model_calls", "paid_calls", "tuning_calls"):
        _require(boundary[key] == 0, f"forbidden call enabled: {key}")
    _require(boundary["post_score_exploratory"] is True, "post-score disclosure removed")
    _require(boundary["general_3d_validated"] is False, "3D overclaim")
    claims = config["claim_boundary"]
    for key in (
        "preregistered_confirmation",
        "ngc4214_standard_rotation_curve_valid",
        "refracted_gravity_beats_known_family",
        "unique_theory_established",
        "publication_candidate",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overreach: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def _validate_package() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    if _CONFIG_RAW_SHA256 != "0" * 64:
        _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _load_score_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["score_binding"]
    for row in binding["artifacts"]:
        path = _repo_path(row["path"])
        _require(path.is_file(), "score artifact missing")
        _require(file_sha256(path) == row["sha256"], "score artifact changed")
    receipt = _read_json(_repo_path(score.OUTPUT_PATH), "score receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "score receipt changed"
    )
    primary = next(row for row in receipt["objects"] if row["object_id"] == "NGC2976")
    stress = next(row for row in receipt["objects"] if row["object_id"] == "NGC4214")
    _require(
        primary["rg_fractional_rmse_improvement_over_newton"]
        == binding["sealed_primary_ngc2976_rg_fractional_improvement"],
        "primary score changed",
    )
    _require(
        stress["rg_fractional_rmse_improvement_over_newton"]
        == binding["sealed_stress_ngc4214_rg_fractional_improvement"],
        "stress score changed",
    )
    return receipt


def rar_vector_field(
    field: tuple[np.ndarray, np.ndarray], *, a0_m_s2: float, g_dagger_m_s2: float
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.asarray(field[0], dtype=np.float64)
    gy = np.asarray(field[1], dtype=np.float64)
    gbar = np.hypot(gx, gy) * a0_m_s2
    x = np.sqrt(np.maximum(gbar, 0.0) / g_dagger_m_s2)
    denominator = -np.expm1(-x)
    multiplier = np.divide(
        1.0,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return gx * multiplier, gy * multiplier


def rar_target_free_benchmarks() -> dict[str, Any]:
    g_dagger = 1.2e-10
    low = np.asarray([1.0e-18, 1.0e-16, 1.0e-14])
    high = np.asarray([1.0e-6, 1.0e-4, 1.0e-2])

    def magnitude(gbar: np.ndarray) -> np.ndarray:
        x = np.sqrt(gbar / g_dagger)
        return gbar / (-np.expm1(-x))

    low_ratio = magnitude(low) / np.sqrt(low * g_dagger)
    high_ratio = magnitude(high) / high
    zero = rar_vector_field(
        (np.zeros((2, 2)), np.zeros((2, 2))),
        a0_m_s2=1.2e-10,
        g_dagger_m_s2=g_dagger,
    )
    direction = rar_vector_field(
        (np.asarray([[3.0]]), np.asarray([[4.0]])),
        a0_m_s2=1.2e-10,
        g_dagger_m_s2=g_dagger,
    )
    cross = float(direction[0][0, 0] * 4.0 - direction[1][0, 0] * 3.0)
    checks = {
        "high_acceleration_newtonian": bool(abs(float(high_ratio[-1]) - 1.0) < 1.0e-12),
        "deep_rar_sqrt_gbar_gdagger": bool(abs(float(low_ratio[0]) - 1.0) < 5.0e-5),
        "direction_preserved": bool(abs(cross) < 1.0e-12),
        "zero_field_zero_response": bool(
            np.count_nonzero(zero[0]) == 0 and np.count_nonzero(zero[1]) == 0
        ),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "low_acceleration_ratio": [float(value) for value in low_ratio],
        "high_acceleration_ratio": [float(value) for value in high_ratio],
        "direction_cross_product": cross,
    }


def _rar_prediction(
    config: Mapping[str, Any],
    prediction_config: Mapping[str, Any],
    object_id: str,
    source_config: Mapping[str, Any],
    geometry: Mapping[str, Any],
    paths: Mapping[tuple[str, str], Path],
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    bridge_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    metadata = predictions.sources.geometry_variants(source_config, dict(geometry[object_id]))[0]
    images = predictions.sources._load_images(object_id, dict(paths))
    maps = predictions.sources._maps(
        source_config,
        metadata,
        images,
        n=int(prediction_config["field_grid"]["source_map_pixels"]),
        box_kpc=float(prediction_config["field_grid"]["source_map_box_kpc"]),
    )
    _rhalf_pc, scale_pc = predictions.sources._scale_length(maps)
    source_id = prediction_config["source_cell"]["id"]
    fine_metrics, fine_fields = predictions._solve_grid(
        prediction_config,
        bridge_config,
        maps,
        expected[(object_id, source_id)],
        exponential_scale_pc=scale_pc,
        nodes=int(prediction_config["field_grid"]["fine_nodes_per_axis"]),
    )
    coarse_metrics, coarse_fields = predictions._solve_grid(
        prediction_config,
        bridge_config,
        maps,
        expected[(object_id, source_id)],
        exponential_scale_pc=scale_pc,
        nodes=int(prediction_config["field_grid"]["convergence_nodes_per_axis"]),
    )
    a0 = float(bridge_config["normalization_contract"]["a0_m_s2"])
    g_dagger = float(config["rar_comparator"]["g_dagger_m_s2"])
    fine_rar = rar_vector_field(fine_fields[_NEWTON], a0_m_s2=a0, g_dagger_m_s2=g_dagger)
    coarse_rar = rar_vector_field(coarse_fields[_NEWTON], a0_m_s2=a0, g_dagger_m_s2=g_dagger)
    response_path, response_row = predictions._response_header_path(prediction_config, object_id)
    header = predictions._response_header_only(response_path)
    ra, dec = predictions._world_grid(header)
    major, disk_y, radius, cosine = predictions._disk_sky_coordinates(metadata, ra, dec)
    intensity, robust_header = predictions._native_robust_intensity(images, ra, dec)
    robust_beam = predictions.sources.base._things_beam(robust_header)
    target_beam = (
        float(response_row["beam_major_deg"]),
        float(response_row["beam_minor_deg"]),
        float(response_row["beam_position_angle_deg"]),
    )
    beam = predictions.additional_beam(robust_beam, target_beam, abs(float(header["CDELT2"])))
    half_box = float(prediction_config["field_grid"]["half_box_kpc"])
    fine_radial, fine_tangential = predictions._sample_force(
        fine_rar,
        major,
        disk_y,
        radius,
        half_box_kpc=half_box,
        a0_m_s2=a0,
    )
    coarse_radial, _ = predictions._sample_force(
        coarse_rar,
        major,
        disk_y,
        radius,
        half_box_kpc=half_box,
        a0_m_s2=a0,
    )
    relative = np.abs(fine_radial - coarse_radial) / np.maximum(
        np.maximum(np.abs(fine_radial), np.abs(coarse_radial)), a0 * 1.0e-4
    )
    radius_m = radius * 1000.0 * float(bridge_config["normalization_contract"]["pc_m"])
    speed = np.sqrt(np.maximum(radius_m * fine_radial, 0.0))
    raw = speed * math.sin(math.radians(float(metadata["inclination_deg"]))) * cosine
    convolved, denominator = predictions.projection.intensity_weighted_beam(
        np.where(np.isfinite(raw), raw, 0.0), intensity, np.asarray(beam["kernel"])
    )
    tangential = np.abs(fine_tangential) / np.maximum(np.abs(fine_radial), a0 * 1.0e-4)
    result = {
        "vlos_plus_m_s": convolved,
        "convergence_relative": relative,
        "tangential_ratio": tangential,
        "major_kpc": major,
        "radius_kpc": radius,
        "intensity": denominator,
        "fine_solver_residual": np.asarray(
            [
                fine_metrics["newton_relative_residual"],
                fine_metrics["refracted_gravity_solver"]["relative_residual"],
                coarse_metrics["newton_relative_residual"],
                coarse_metrics["refracted_gravity_solver"]["relative_residual"],
            ]
        ),
    }
    del maps, images, fine_fields, coarse_fields, fine_rar, coarse_rar, ra, dec
    gc.collect()
    return result


def _fixed_offset_metrics(
    observed: np.ndarray,
    dispersion: np.ndarray,
    predicted_plus: np.ndarray,
    mask: np.ndarray,
    *,
    sign: float,
    offset: float,
    minimum_dispersion: float,
) -> dict[str, Any] | None:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return None
    residual = observed[mask] - (sign * predicted_plus[mask] + offset)
    scale = np.maximum(dispersion[mask], minimum_dispersion)
    return {
        "pixel_count": count,
        "rmse_m_s": float(np.sqrt(np.mean(residual**2))),
        "mae_m_s": float(np.mean(np.abs(residual))),
        "mom2_scaled_mse": float(np.mean((residual / scale) ** 2)),
        "residual_mean_m_s": float(np.mean(residual)),
    }


def _score_diagnostics(
    config: Mapping[str, Any],
    score_receipt: Mapping[str, Any],
    prediction_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    prediction_config: Mapping[str, Any],
    object_id: str,
    rar: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    response_rows = score._response_rows(prediction_config, object_id)
    observed, _header = score._load_response(response_rows["HI_MOM1_NATURAL_VELOCITY_FIELD"])
    dispersion, _ = score._load_response(response_rows["HI_MOM2_NATURAL_VELOCITY_DISPERSION"])
    eligibility = score._load_prediction_array(manifest, object_id, "source_eligibility").astype(
        bool
    )
    arrays = {
        _NEWTON: score._load_prediction_array(manifest, object_id, "newton_vlos_plus_m_s"),
        _RG: score._load_prediction_array(manifest, object_id, "rg_vlos_plus_m_s"),
        _RAR: np.asarray(rar["vlos_plus_m_s"]),
    }
    tangential = {
        _NEWTON: score._load_prediction_array(manifest, object_id, "newton_tangential_ratio"),
        _RG: score._load_prediction_array(manifest, object_id, "rg_tangential_ratio"),
        _RAR: np.asarray(rar["tangential_ratio"]),
    }
    convergence_limit = float(
        prediction_config["field_grid"]["maximum_local_radial_acceleration_relative_difference"]
    )
    common = (
        eligibility
        & np.isfinite(observed)
        & np.isfinite(dispersion)
        & (dispersion > 0.0)
        & np.isfinite(rar["intensity"])
        & (rar["intensity"] > 0.0)
        & np.isfinite(rar["convergence_relative"])
        & (rar["convergence_relative"] <= convergence_limit)
    )
    for value in arrays.values():
        common &= np.isfinite(value)
    count = int(np.count_nonzero(common))
    _require(count > 0, f"no comparator pixels for {object_id}")
    sealed = next(row for row in score_receipt["objects"] if row["object_id"] == object_id)
    sign = float(sealed["rotation_sign"])
    offsets = {
        model_id: float(np.mean(observed[common] - sign * value[common]))
        for model_id, value in arrays.items()
    }
    minimum_dispersion = float(config["common_score_contract"]["minimum_dispersion_scale_m_s"])
    all_metrics = {
        model_id: _fixed_offset_metrics(
            observed,
            dispersion,
            arrays[model_id],
            common,
            sign=sign,
            offset=offsets[model_id],
            minimum_dispersion=minimum_dispersion,
        )
        for model_id in _MODELS
    }
    _require(all(value is not None for value in all_metrics.values()), "all-pixel metric missing")
    cells: list[dict[str, Any]] = []

    def add_cell(cell_id: str, mask: np.ndarray) -> None:
        cell_mask = common & mask
        cells.append(
            {
                "cell_id": cell_id,
                "pixel_count": int(np.count_nonzero(cell_mask)),
                "models": {
                    model_id: _fixed_offset_metrics(
                        observed,
                        dispersion,
                        arrays[model_id],
                        cell_mask,
                        sign=sign,
                        offset=offsets[model_id],
                        minimum_dispersion=minimum_dispersion,
                    )
                    for model_id in _MODELS
                },
            }
        )

    add_cell("ALL_COMMON", np.ones_like(common, dtype=bool))
    max_tangential = np.maximum.reduce([np.asarray(tangential[model]) for model in _MODELS])
    for threshold in config["exploratory_sensitivity_grid"]["source_tangential_ratio_maximum"][1:]:
        add_cell(f"MAX_TANGENTIAL_LE_{threshold}", max_tangential <= float(threshold))
    radius = np.asarray(rar["radius_kpc"])
    for lower, upper in config["exploratory_sensitivity_grid"]["radial_bins_kpc"]:
        add_cell(f"RADIUS_{lower}_{upper}_KPC", (radius >= lower) & (radius < upper))
    major = np.asarray(rar["major_kpc"])
    add_cell("MAJOR_NEGATIVE", major < 0.0)
    add_cell("MAJOR_POSITIVE", major >= 0.0)
    intensity = np.asarray(rar["intensity"])
    quantiles = np.quantile(intensity[common], [0.0, 0.25, 0.5, 0.75, 1.0])
    for index, (lower_q, upper_q) in enumerate(
        config["exploratory_sensitivity_grid"]["source_intensity_quantiles"]
    ):
        lower_value = float(quantiles[index])
        upper_value = float(quantiles[index + 1])
        upper_condition = intensity <= upper_value if index == 3 else intensity < upper_value
        add_cell(
            f"SOURCE_INTENSITY_Q{lower_q}_{upper_q}",
            (intensity >= lower_value) & upper_condition,
        )
    rg_rmse = float(all_metrics[_RG]["rmse_m_s"])
    rar_rmse = float(all_metrics[_RAR]["rmse_m_s"])
    newton_rmse = float(all_metrics[_NEWTON]["rmse_m_s"])
    object_prediction = next(
        row for row in prediction_receipt["objects"] if row["object_id"] == object_id
    )
    return {
        "object_id": object_id,
        "inference_role": sealed["inference_role"],
        "common_pixel_count": count,
        "rotation_sign_reused": sign,
        "all_pixel_systemic_offsets_m_s": offsets,
        "all_pixel_metrics": all_metrics,
        "rg_fractional_improvement_over_newton": float((newton_rmse - rg_rmse) / newton_rmse),
        "rg_fractional_improvement_over_rar": float((rar_rmse - rg_rmse) / rar_rmse),
        "rg_beats_rar": bool(rg_rmse < rar_rmse),
        "sensitivity_cells": cells,
        "sensitivity_cell_count": len(cells),
        "rar_solver": {
            "maximum_parent_solver_residual": float(np.max(rar["fine_solver_residual"])),
            "converged_pixels": int(
                np.count_nonzero(
                    np.isfinite(rar["convergence_relative"])
                    & (rar["convergence_relative"] <= convergence_limit)
                )
            ),
            "maximum_common_tangential_ratio": float(np.max(max_tangential[common])),
        },
        "sealed_primary_predecessor_metrics": object_prediction["models"],
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    score_receipt = _load_score_evidence(config)
    prediction_config = predictions.load_config()
    prediction_receipt, manifest = score._load_prediction_evidence(score.load_config())
    source_config, _acquisition, geometry, paths, expected = (
        predictions.source_resolution._source_evidence(predictions.source_resolution.load_config())
    )
    bridge_config = predictions.bridge.load_config()
    benchmark = rar_target_free_benchmarks()
    _require(benchmark["all_pass"], "RAR target-free benchmark failed")
    objects: list[dict[str, Any]] = []
    for object_id in _OBJECTS:
        rar = _rar_prediction(
            config,
            prediction_config,
            object_id,
            source_config,
            geometry,
            paths,
            expected,
            bridge_config,
        )
        objects.append(
            _score_diagnostics(
                config,
                score_receipt,
                prediction_receipt,
                manifest,
                prediction_config,
                object_id,
                rar,
            )
        )
        del rar
        gc.collect()
    primary = next(row for row in objects if row["object_id"] == "NGC2976")
    stress = next(row for row in objects if row["object_id"] == "NGC4214")
    threshold = 0.05
    if primary["rg_fractional_improvement_over_rar"] >= threshold:
        decision = "PRIMARY_RG_BEATS_FIXED_RAR_EXPLORATORY_EXPANSION_WARRANTED"
    else:
        decision = "PRIMARY_RG_SIGNAL_EXPLAINED_OR_EXCEEDED_BY_FIXED_RAR"
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_POST_PRIMARY_SCORE_RAR_AND_SOURCE_SENSITIVITY_DIAGNOSTICS",
        "decision": decision,
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "score_receipt_content_sha256": score_receipt["content_sha256"],
        "rar_primary_source": config["primary_sources"][0],
        "rar_target_free_benchmarks": benchmark,
        "objects": objects,
        "diagnostic_summary": {
            "primary_rg_fractional_improvement_over_rar": primary[
                "rg_fractional_improvement_over_rar"
            ],
            "primary_rg_beats_rar": primary["rg_beats_rar"],
            "stress_rg_fractional_improvement_over_rar": stress[
                "rg_fractional_improvement_over_rar"
            ],
            "stress_rg_beats_rar": stress["rg_beats_rar"],
            "exploratory_expansion_threshold": threshold,
            "preregistered_confirmation": False,
        },
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(
    config: Mapping[str, Any], payload: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> None:
    validate_config(config)
    _require(dict(payload) == dict(rebuilt), "receipt differs from rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing nonidentical overwrite")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent nonidentical output")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n")


def check_receipt() -> None:
    config = load_config()
    rebuilt = build_receipt(config)
    _require(
        _repo_path(OUTPUT_PATH).read_bytes() == canonical_bytes(rebuilt) + b"\n",
        "receipt differs",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"), nargs="?", default="check")
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID")
    else:
        path = _repo_path(OUTPUT_PATH)
        if path.is_file():
            receipt = _read_json(path, "receipt")
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "decision": receipt["decision"],
                        "primary_rg_improvement_over_rar": receipt["diagnostic_summary"][
                            "primary_rg_fractional_improvement_over_rar"
                        ],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps({"status": "UNBUILT_EXPLORATORY", "output_exists": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
