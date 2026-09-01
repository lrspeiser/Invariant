from __future__ import annotations

import argparse
import copy
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
from astropy.io import fits

from sigma_theory_compiler import (
    open_gravity_rg_things_matched_pair_2d_comparator_diagnostics_v1 as diagnostics,
)
from sigma_theory_compiler import open_gravity_rg_things_matched_pair_2d_pixel_score_v1 as score
from sigma_theory_compiler import (
    open_gravity_rg_things_matched_pair_2d_source_predictions_v1 as predictions,
)

CONFIG_PATH = Path("configs/open_gravity_rg_things_five_object_2d_expansion_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_five_object_2d_expansion_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_five_object_2d_expansion_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-things-five-object-2d-expansion-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-five-object-2d-expansion-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-five-object-2d-expansion-receipt-1.0"
_OBJECTS = ("NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214")
_NEW_OBJECTS = ("NGC2903", "NGC3198", "NGC3521")
_MODELS = diagnostics._MODELS
_CONFIG_RAW_SHA256 = "d0cf1e77723ba7c2643e5249d6854c110018e73e07675fa6ac57db74ccdc9363"
_CONFIG_CONTENT_SHA256 = "d0458d089116d0616a6e3e4e37139befb828708df14dcdd7b606042c27d21dc9"
_MODULE_SEMANTIC_SHA256 = "431e33e6dfef5de5f79e29d95b0cac4d68f03dc837edc618b09a0759d3f95df2"
_TEST_RAW_SHA256 = "28c1cc8e3aacd00fe0cb228ca9c655f0726f16d0e4387321b8e537c4ab6f1121"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class FiveObjectExpansionError(RuntimeError):
    """Raised when source, response, solver, or score expansion fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FiveObjectExpansionError(message)


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
        raise FiveObjectExpansionError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_POST_FAILED_NUMERICAL_GATE_REPAIR_NEW_RESPONSES_EXPOSED",
        "status changed",
    )
    selection = config["selection_contract"]
    _require(selection["full_source_ready_set"] == list(_OBJECTS), "full set changed")
    _require(
        selection["new_response_blind_objects_at_original_freeze"] == list(_NEW_OBJECTS),
        "new set changed",
    )
    _require(selection["selection_used_new_response_pixels"] is False, "selection leakage")
    _require(selection["one_failure_never_prunes_family"] is True, "failure retention removed")
    repair = config["failed_run_and_numerical_repair"]
    _require(repair["failed_run_occurred"] is True, "failed run erased")
    _require(repair["failed_run_response_pixels_were_decoded"] is True, "exposure erased")
    _require(repair["failed_run_receipt_was_not_written"] is True, "failed receipt claim changed")
    _require(
        repair["failed_fine_rg_relative_residual"] == 5.8304817824339535e-8,
        "failed residual changed",
    )
    _require(repair["required_maximum_relative_residual"] == 1e-8, "residual gate changed")
    _require(repair["repair_pcg_relative_tolerance"] == 1e-10, "repair tolerance changed")
    _require(repair["repair_pcg_max_iterations"] == 400, "repair iterations changed")
    for key in (
        "loss_values_were_not_inspected_or_used_to_choose_repair",
        "final_new_object_results_are_not_strictly_response_blind",
    ):
        _require(repair[key] is True, f"repair disclosure changed: {key}")
    for key in (
        "physical_parameters_changed",
        "source_geometry_or_mask_changed",
        "decision_rule_changed",
    ):
        _require(repair[key] is False, f"repair scope changed: {key}")
    admission = config["source_admission"]
    for key in (
        "real_public_source_and_response_data_required",
        "primary_measurement_and_method_papers_required",
        "target_free_operator_projection_and_rar_benchmarks_required",
        "newtonian_and_rar_controls_required",
    ):
        _require(admission[key] is True, f"admission changed: {key}")
    _require(admission["missing_source_disposition"] == "SOURCE_BLOCKED", "source rule changed")
    _require(admission["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY", "paper rule changed")
    _require(
        admission["model_lifted_vertical_structure_disposition"] == "MODEL_LIFTED_2P5D",
        "2.5D label changed",
    )
    _require(admission["general_3d_validation_allowed"] is False, "3D overclaim")
    files = config["new_response_files"]
    _require(len(files) == 6, "response inventory changed")
    _require(sum(int(row["bytes"]) for row in files) == 25_655_040, "response bytes changed")
    _require(
        {(row["object_id"], row["role"]) for row in files}
        == {
            (object_id, role)
            for object_id in _NEW_OBJECTS
            for role in (
                "HI_MOM1_NATURAL_VELOCITY_FIELD",
                "HI_MOM2_NATURAL_VELOCITY_DISPERSION",
            )
        },
        "response roles changed",
    )
    incident = config["incidental_archive_access"]
    _require(incident["object_id"] == "NGC5055", "incident object changed")
    _require(incident["scientific_pixels_decoded"] == 0, "incident pixel access changed")
    _require(incident["used_in_campaign"] is False, "incidental object admitted")
    model = config["fixed_model_contract"]
    _require(model["models"] == list(_MODELS), "models changed")
    _require(
        model["rg_parameters"] == {"epsilon_0": 0.661, "Q": 1.79, "log10_rho_c_g_cm3": -24.54},
        "RG parameters changed",
    )
    _require(model["rar_g_dagger_m_s2"] == 1.2e-10, "RAR scale changed")
    _require(model["pcg_relative_tolerance"] == 1e-10, "PCG tolerance changed")
    _require(model["pcg_max_iterations"] == 400, "PCG iterations changed")
    _require(model["response_parameter_fits"] == 0, "response tuning enabled")
    score_contract = config["score_contract"]
    _require(score_contract["same_as_sealed_pair"] is True, "score changed")
    _require(score_contract["minimum_dispersion_scale_m_s"] == 3000.0, "dispersion floor changed")
    _require(score_contract["p_values_computed"] is False, "p-value overclaim")
    decision = config["expansion_decision_rule"]
    _require(decision["minimum_objects_for_broader_signal"] == 3, "decision count changed")
    _require(decision["minimum_new_blind_objects_for_broader_signal"] == 2, "blind count changed")
    _require(decision["decision_is_not_confirmation_or_publication"] is True, "decision overclaim")
    access = config["access_accounting"]
    _require(access["head_calls"] == 8 and access["get_calls"] == 8, "network accounting changed")
    _require(access["network_bytes"] == 34_145_280, "network bytes changed")
    _require(access["campaign_response_array_slots_decoded"] == 6_291_456, "pixel slots changed")
    _require(access["incidental_pixels_decoded"] == 0, "incidental pixels changed")
    claims = config["claim_boundary"]
    for key in (
        "broader_development_signal",
        "preregistered_confirmation",
        "general_3d_validated",
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


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["bindings"]:
        receipt_path: Path | None = None
        for row in binding["artifacts"]:
            path = _repo_path(row["path"])
            _require(path.is_file(), "bound artifact missing")
            _require(file_sha256(path) == row["sha256"], "bound artifact changed")
            if row["path"].endswith("/receipt.json"):
                receipt_path = path
        _require(receipt_path is not None, "bound receipt missing")
        receipt = _read_json(receipt_path, "bound receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"], "bound content changed"
        )
        receipts[binding["role"]] = receipt
    source = receipts["FIVE_OBJECT_REAL_SOURCE_BUILDER"]
    _require(
        sorted(row["object_id"] for row in source["object_summaries"]) == list(_OBJECTS),
        "source object inventory changed",
    )
    diagnostic = receipts["SEALED_MATCHED_PAIR_COMPARATOR_DIAGNOSTICS"]
    _require(len(diagnostic["objects"]) == 2, "pair result count changed")
    return receipts


def _response_rows(config: Mapping[str, Any], object_id: str) -> dict[str, dict[str, Any]]:
    rows = {
        row["role"]: row for row in config["new_response_files"] if row["object_id"] == object_id
    }
    _require(
        set(rows)
        == {
            "HI_MOM1_NATURAL_VELOCITY_FIELD",
            "HI_MOM2_NATURAL_VELOCITY_DISPERSION",
        },
        "response roles missing",
    )
    return rows


def _validate_response_header(row: Mapping[str, Any]) -> fits.Header:
    path = _repo_path(row["relative_path"])
    _require(path.is_file(), "response file missing")
    _require(path.stat().st_size == row["bytes"], "response bytes changed")
    _require(file_sha256(path) == row["sha256"], "response hash changed")
    header = fits.getheader(path)
    _require(
        int(header["NAXIS1"]) == 1024 and int(header["NAXIS2"]) == 1024, "response shape changed"
    )
    _require(header["BUNIT"] == "METR/SEC", "response unit changed")
    return header


def _predict_models(
    config: Mapping[str, Any],
    prediction_config: Mapping[str, Any],
    object_id: str,
    source_config: Mapping[str, Any],
    geometry: Mapping[str, Any],
    paths: Mapping[tuple[str, str], Path],
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    bridge_config: Mapping[str, Any],
) -> dict[str, Any]:
    numerical_config = copy.deepcopy(dict(prediction_config))
    numerical_config["operator"]["pcg_relative_tolerance"] = float(
        config["fixed_model_contract"]["pcg_relative_tolerance"]
    )
    numerical_config["operator"]["pcg_max_iterations"] = int(
        config["fixed_model_contract"]["pcg_max_iterations"]
    )
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
        numerical_config,
        bridge_config,
        maps,
        expected[(object_id, source_id)],
        exponential_scale_pc=scale_pc,
        nodes=int(prediction_config["field_grid"]["fine_nodes_per_axis"]),
    )
    coarse_metrics, coarse_fields = predictions._solve_grid(
        numerical_config,
        bridge_config,
        maps,
        expected[(object_id, source_id)],
        exponential_scale_pc=scale_pc,
        nodes=int(prediction_config["field_grid"]["convergence_nodes_per_axis"]),
    )
    a0 = float(bridge_config["normalization_contract"]["a0_m_s2"])
    g_dagger = float(config["fixed_model_contract"]["rar_g_dagger_m_s2"])
    fine_fields[diagnostics._RAR] = diagnostics.rar_vector_field(
        fine_fields[diagnostics._NEWTON], a0_m_s2=a0, g_dagger_m_s2=g_dagger
    )
    coarse_fields[diagnostics._RAR] = diagnostics.rar_vector_field(
        coarse_fields[diagnostics._NEWTON], a0_m_s2=a0, g_dagger_m_s2=g_dagger
    )
    rows = _response_rows(config, object_id)
    header = _validate_response_header(rows["HI_MOM1_NATURAL_VELOCITY_FIELD"])
    ra, dec = predictions._world_grid(header)
    major, disk_y, radius, cosine = predictions._disk_sky_coordinates(metadata, ra, dec)
    intensity, robust_header = predictions._native_robust_intensity(images, ra, dec)
    robust_beam = predictions.sources.base._things_beam(robust_header)
    response_row = rows["HI_MOM1_NATURAL_VELOCITY_FIELD"]
    target_beam = (
        float(response_row["beam_major_deg"]),
        float(response_row["beam_minor_deg"]),
        float(response_row["beam_position_angle_deg"]),
    )
    beam = predictions.additional_beam(robust_beam, target_beam, abs(float(header["CDELT2"])))
    half_box = float(prediction_config["field_grid"]["half_box_kpc"])
    pc_m = float(bridge_config["normalization_contract"]["pc_m"])
    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    for model_id in _MODELS:
        fine_radial, fine_tangential = predictions._sample_force(
            fine_fields[model_id],
            major,
            disk_y,
            radius,
            half_box_kpc=half_box,
            a0_m_s2=a0,
        )
        coarse_radial, _ = predictions._sample_force(
            coarse_fields[model_id],
            major,
            disk_y,
            radius,
            half_box_kpc=half_box,
            a0_m_s2=a0,
        )
        relative = np.abs(fine_radial - coarse_radial) / np.maximum(
            np.maximum(np.abs(fine_radial), np.abs(coarse_radial)), a0 * 1.0e-4
        )
        speed = np.sqrt(np.maximum(radius * 1000.0 * pc_m * fine_radial, 0.0))
        raw = speed * math.sin(math.radians(float(metadata["inclination_deg"]))) * cosine
        convolved, denominator = predictions.projection.intensity_weighted_beam(
            np.where(np.isfinite(raw), raw, 0.0), intensity, np.asarray(beam["kernel"])
        )
        model_arrays[model_id] = {
            "vlos": convolved,
            "relative": relative,
            "positive": fine_radial > 0.0,
            "tangential": np.abs(fine_tangential) / np.maximum(np.abs(fine_radial), a0 * 1.0e-4),
        }
    result = {
        "metadata": metadata,
        "header": header,
        "major": major,
        "radius": radius,
        "intensity": denominator,
        "models": model_arrays,
        "solver": {
            "fine_newton_residual": fine_metrics["newton_relative_residual"],
            "fine_rg_residual": fine_metrics["refracted_gravity_solver"]["relative_residual"],
            "coarse_newton_residual": coarse_metrics["newton_relative_residual"],
            "coarse_rg_residual": coarse_metrics["refracted_gravity_solver"]["relative_residual"],
            "fine_source_mass_relative_error": fine_metrics["source_mass_relative_error"],
            "coarse_source_mass_relative_error": coarse_metrics["source_mass_relative_error"],
        },
    }
    del maps, images, fine_fields, coarse_fields, ra, dec
    gc.collect()
    return result


def _score_new_object(
    config: Mapping[str, Any], object_id: str, predicted: Mapping[str, Any]
) -> dict[str, Any]:
    solver = predicted["solver"]
    _require(
        max(
            float(solver["fine_newton_residual"]),
            float(solver["fine_rg_residual"]),
            float(solver["coarse_newton_residual"]),
            float(solver["coarse_rg_residual"]),
        )
        <= float(config["failed_run_and_numerical_repair"]["required_maximum_relative_residual"]),
        f"solver residual gate failed for {object_id}",
    )
    _require(
        max(
            float(solver["fine_source_mass_relative_error"]),
            float(solver["coarse_source_mass_relative_error"]),
        )
        <= 2.0e-9,
        f"source mass gate failed for {object_id}",
    )
    rows = _response_rows(config, object_id)
    observed, header = score._load_response(rows["HI_MOM1_NATURAL_VELOCITY_FIELD"])
    dispersion, _ = score._load_response(rows["HI_MOM2_NATURAL_VELOCITY_DISPERSION"])
    grid = predicted["models"]
    convergence_limit = 0.1
    common = (
        np.isfinite(observed)
        & np.isfinite(dispersion)
        & (dispersion > 0.0)
        & np.isfinite(predicted["intensity"])
        & (predicted["intensity"] > 0.0)
        & (predicted["radius"] >= 0.5)
        & (predicted["radius"] <= 15.0)
    )
    for model in _MODELS:
        values = grid[model]
        common &= (
            np.isfinite(values["vlos"])
            & np.isfinite(values["relative"])
            & (values["relative"] <= convergence_limit)
            & values["positive"]
        )
    count = int(np.count_nonzero(common))
    _require(count > 0, f"no score pixels for {object_id}")
    sign, covariance = score._rotation_sign(predicted["major"], observed, common)
    metrics = {
        model_id: score._model_metrics(
            observed,
            dispersion,
            grid[model_id]["vlos"],
            common,
            sign,
            3000.0,
        )
        for model_id in _MODELS
    }
    newton = float(metrics[diagnostics._NEWTON]["rmse_m_s"])
    rg = float(metrics[diagnostics._RG]["rmse_m_s"])
    rar = float(metrics[diagnostics._RAR]["rmse_m_s"])
    threshold = 0.05
    object_signal = rg <= (1.0 - threshold) * min(newton, rar)
    pixel_area = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    response = rows["HI_MOM1_NATURAL_VELOCITY_FIELD"]
    beam_area = (
        math.pi
        * float(response["beam_major_deg"])
        * float(response["beam_minor_deg"])
        / (4.0 * math.log(2.0))
    )
    return {
        "object_id": object_id,
        "result_source": "NEW_RESPONSE_BLIND_AT_FREEZE",
        "geometry": predicted["metadata"],
        "common_pixel_count": count,
        "beam_equivalent_count": float(count * pixel_area / beam_area),
        "rotation_sign": sign,
        "major_axis_velocity_covariance_kpc_m_s": covariance,
        "models": metrics,
        "rg_fractional_improvement_over_newton": float((newton - rg) / newton),
        "rg_fractional_improvement_over_rar": float((rar - rg) / rar),
        "rg_broader_signal_object_gate": bool(object_signal),
        "maximum_common_tangential_ratio": float(
            np.max(np.maximum.reduce([grid[model]["tangential"] for model in _MODELS])[common])
        ),
        "solver": predicted["solver"],
    }


def _pair_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    newton = float(row["all_pixel_metrics"][diagnostics._NEWTON]["rmse_m_s"])
    rg = float(row["all_pixel_metrics"][diagnostics._RG]["rmse_m_s"])
    rar = float(row["all_pixel_metrics"][diagnostics._RAR]["rmse_m_s"])
    return {
        "object_id": row["object_id"],
        "result_source": "SEALED_PAIR_RESULT_REUSED",
        "common_pixel_count": row["common_pixel_count"],
        "models": row["all_pixel_metrics"],
        "rg_fractional_improvement_over_newton": row["rg_fractional_improvement_over_newton"],
        "rg_fractional_improvement_over_rar": row["rg_fractional_improvement_over_rar"],
        "rg_broader_signal_object_gate": bool(rg <= 0.95 * min(newton, rar)),
        "inference_role": row["inference_role"],
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    receipts = _validate_bindings(config)
    benchmark = diagnostics.rar_target_free_benchmarks()
    _require(benchmark["all_pass"], "RAR benchmark failed")
    prediction_config = predictions.load_config()
    source_config, _acquisition, geometry, paths, expected = (
        predictions.source_resolution._source_evidence(predictions.source_resolution.load_config())
    )
    bridge_config = predictions.bridge.load_config()
    new_details: list[dict[str, Any]] = []
    for object_id in _NEW_OBJECTS:
        predicted = _predict_models(
            config,
            prediction_config,
            object_id,
            source_config,
            geometry,
            paths,
            expected,
            bridge_config,
        )
        new_details.append(_score_new_object(config, object_id, predicted))
        del predicted
        gc.collect()
    pair = receipts["SEALED_MATCHED_PAIR_COMPARATOR_DIAGNOSTICS"]
    summaries = [_pair_summary(row) for row in pair["objects"]] + [
        {
            "object_id": row["object_id"],
            "result_source": row["result_source"],
            "common_pixel_count": row["common_pixel_count"],
            "models": row["models"],
            "rg_fractional_improvement_over_newton": row["rg_fractional_improvement_over_newton"],
            "rg_fractional_improvement_over_rar": row["rg_fractional_improvement_over_rar"],
            "rg_broader_signal_object_gate": row["rg_broader_signal_object_gate"],
        }
        for row in new_details
    ]
    summaries.sort(key=lambda row: row["object_id"])
    _require([row["object_id"] for row in summaries] == list(_OBJECTS), "summary order changed")
    signal_objects = [row["object_id"] for row in summaries if row["rg_broader_signal_object_gate"]]
    new_signal_objects = [object_id for object_id in signal_objects if object_id in _NEW_OBJECTS]
    broader = len(signal_objects) >= 3 and len(new_signal_objects) >= 2
    decision = (
        "BROADER_FIVE_OBJECT_RG_DEVELOPMENT_SIGNAL_WARRANTS_PUBLICATION_CAMPAIGN"
        if broader
        else "NGC2976_RG_SIGNAL_DID_NOT_GENERALIZE_TO_PREREGISTERED_FIVE_OBJECT_GATE"
    )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_COMPLETE_FIVE_OBJECT_REAL_THINGS_2D_EXPANSION",
        "decision": decision,
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "binding_content_sha256": {
            key: value["content_sha256"] for key, value in sorted(receipts.items())
        },
        "primary_sources": config["primary_sources"],
        "rar_target_free_benchmarks": benchmark,
        "objects": summaries,
        "new_object_details": new_details,
        "expansion_gate": {
            "signal_object_count": len(signal_objects),
            "signal_objects": signal_objects,
            "new_blind_signal_object_count": len(new_signal_objects),
            "new_blind_signal_objects": new_signal_objects,
            "minimum_signal_objects": 3,
            "minimum_new_blind_signal_objects": 2,
            "broader_development_signal": bool(broader),
            "confirmation": False,
        },
        "incidental_archive_access": config["incidental_archive_access"],
        "access_accounting": config["access_accounting"],
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
                        "signal_objects": receipt["expansion_gate"]["signal_objects"],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps({"status": "UNRUN_NEW_OBJECTS_RESPONSE_BLIND", "output_exists": False})
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
