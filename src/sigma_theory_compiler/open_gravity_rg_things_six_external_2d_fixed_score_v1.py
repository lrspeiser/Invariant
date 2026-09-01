"""Score six-galaxy response-blind predictions against sealed THINGS maps."""

from __future__ import annotations

import argparse
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
    open_gravity_rg_things_six_external_2d_response_blind_predictions_v1 as predictions,
)

CONFIG_PATH = Path("configs/open_gravity_rg_things_six_external_2d_fixed_score_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_six_external_2d_fixed_score_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_six_external_2d_fixed_score_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-six-external-2d-fixed-score-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-six-external-2d-fixed-score-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-six-external-2d-fixed-score-receipt-1.0"
_CANDIDATES = predictions._CANDIDATES
_RESOLUTIONS = ("NATURAL", "ROBUST")
_RG = "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
_CONFIG_RAW_SHA256 = "12ab1b88bb74a18fc4fcbd0846f9cb3eb4d1221d0d7ea360d7f5ac080d7768f0"
_CONFIG_CONTENT_SHA256 = "11f7ac251ebfc7c41ec362d435a00b62419bb9578c8e19980f63b5c9abc0f640"
_MODULE_SEMANTIC_SHA256 = "fe8bbae3af82f8698cec0cd2eee04c646797daa0ab085ec4051ee47895529844"
_TEST_RAW_SHA256 = "1d3615977a21e9a218e678606e9299656ca1762bc9fa09cf73efec7bb9f0a193"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class SixGalaxyScoreError(RuntimeError):
    """Raised when a frozen prediction, response, score, or seal gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SixGalaxyScoreError(message)


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
        rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes()
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
        raise SixGalaxyScoreError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_PREREGISTERED_SIX_EXTERNAL_THINGS_2D_FIXED_SCORE",
        "status changed",
    )
    prediction = config["prediction_binding"]
    _require(prediction["objects"] == 6, "prediction objects changed")
    _require(prediction["source_cells"] == 15, "prediction cells changed")
    _require(prediction["retained_source_failures"] == 3, "prediction failures changed")
    _require(prediction["array_file_count"] == 195, "prediction arrays changed")
    _require(prediction["candidate_resolution_predictions"] == 120, "predictions changed")
    _require(prediction["all_solver_gates_pass"] is True, "failed predictions admitted")
    _require(prediction["response_pixels_used_to_build_predictions"] == 0, "prediction leak")
    response = config["response_contract"]
    object_order = ["NGC2841", "IC2574", "DDO154", "NGC5055", "NGC6946", "NGC7331"]
    _require(response["object_order"] == object_order, "objects changed")
    _require(response["resolutions"] == list(_RESOLUTIONS), "resolutions changed")
    _require(response["observables"] == ["MOM1", "MOM2"], "observables changed")
    _require(response["files"] == 24, "response file count changed")
    _require(response["bytes"] == 102_101_760, "response bytes changed")
    _require(response["decoded_array_slots"] == 25_165_824, "response slots changed")
    _require(response["minimum_dispersion_scale_m_s"] == 3000.0, "dispersion floor changed")
    _require(
        response["dispersion_metric_is_not_measurement_likelihood"] is True, "likelihood overclaim"
    )
    score = config["score_contract"]
    _require(score["candidate_ids"] == list(_CANDIDATES), "candidate inventory changed")
    _require(score["primary_candidate_id"] == _RG, "primary candidate changed")
    _require(score["source_cells"] == 15, "source score count changed")
    _require(score["source_resolution_cells"] == 30, "cell score count changed")
    _require(score["model_scores"] == 120, "model score count changed")
    _require(
        list(score["nuisance_reference_cell_by_object"]) == object_order,
        "reference objects changed",
    )
    _require(list(score["primary_cell_by_object"]) == object_order, "primary objects changed")
    _require(
        score["primary_metric"]
        == "unweighted common-pixel velocity RMSE using the object-resolution shared sign and systemic velocity",
        "primary metric changed",
    )
    _require(score["minimum_rg_primary_object_wins_for_strong_signal"] == 4, "win gate changed")
    _require(
        score["require_equal_object_aggregate_win_for_strong_signal"] is True,
        "aggregate gate removed",
    )
    _require(
        score["require_all_inclination_strata_for_strong_signal"] is True, "stratum gate removed"
    )
    for key in (
        "per_model_sign_selection",
        "source_or_geometry_reselection",
        "response_parameter_tuning",
        "p_values_computed",
    ):
        _require(score[key] is False, f"forbidden score behavior enabled: {key}")
    _require(score["all_30_cells_reported"] is True, "cell suppression enabled")
    _require(score["retain_every_failure_and_counterexample"] is True, "failure retention removed")
    boundary = config["scientific_boundary"]
    _require(boundary["response_files_opened"] == 24, "response accounting changed")
    _require(boundary["response_array_slots_decoded"] == 25_165_824, "decode accounting changed")
    _require(boundary["source_resolution_cells_scored"] == 30, "cell accounting changed")
    _require(boundary["model_scores"] == 120, "score accounting changed")
    for key in ("network_calls", "model_calls", "paid_calls", "tuning_calls"):
        _require(boundary[key] == 0, f"forbidden access enabled: {key}")
    _require(boundary["general_3d_validated"] is False, "3D overclaim")
    _require(boundary["model_lifted_2p5d_only"] is True, "2.5D caveat removed")
    claims = config["claim_boundary"]
    _require(claims["external_object_replication_test"] is True, "external test removed")
    for key in (
        "real_response_scored",
        "fixed_predictions_scored",
        "dispersion_metric_is_likelihood",
        "gas_dynamics_solved",
        "inclination_crossover_generalized",
        "general_3d_validated",
        "refracted_gravity_confirmed",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overpromoted: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


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


def _verify_artifacts(binding: Mapping[str, Any], label: str) -> None:
    for artifact in binding["artifacts"]:
        path = _repo_path(artifact["path"])
        _require(path.is_file(), f"{label} artifact missing")
        _require(file_sha256(path) == artifact["sha256"], f"{label} artifact changed")


def _load_prediction_evidence(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = config["prediction_binding"]
    _verify_artifacts(binding, "prediction")
    prediction_config = predictions.load_config()
    receipt = _read_json(_repo_path(predictions.OUTPUT_PATH), "prediction receipt")
    manifest = _read_json(_repo_path(predictions.PRIVATE_MANIFEST_PATH), "prediction manifest")
    predictions.validate_manifest(prediction_config, manifest)
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "prediction receipt changed"
    )
    _require(
        manifest["content_sha256"] == binding["manifest_content_sha256"],
        "prediction manifest changed",
    )
    _require(receipt["candidate_resolution_predictions"] == 120, "prediction count changed")
    _require(receipt["all_solver_gates_pass"] is True, "failed prediction packet admitted")
    _require(
        receipt["response_boundary"]["velocity_pixels_decoded"] == 0, "prediction response leak"
    )
    return prediction_config, receipt, manifest


def _load_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["preflight_binding"]
    _verify_artifacts(binding, "preflight")
    receipt = _read_json(_repo_path(binding["artifacts"][-1]["path"]), "preflight receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "preflight receipt changed"
    )
    _require(
        receipt["status"] == "PASS_OPAQUE_ACQUISITION_AND_HEADER_PREFLIGHT",
        "preflight status changed",
    )
    return receipt


def _cell_payloads(
    prediction_receipt: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_hashes = manifest["cell_content_sha256"]
    output: list[dict[str, Any]] = []
    for summary, expected_hash in zip(
        prediction_receipt["cell_summaries"], expected_hashes, strict=True
    ):
        path = _repo_path(predictions.PRIVATE_DIRECTORY / f"{summary['cell_run_id']}__cell.json")
        cell = _read_json(path, "prediction cell")
        _require(cell["content_sha256"] == expected_hash, "prediction cell changed")
        copy = dict(cell)
        observed = copy.pop("content_sha256")
        _require(observed == predictions.content_sha256(copy), "prediction cell hash invalid")
        output.append(cell)
    _require(len(output) == 15, "prediction cell inventory changed")
    return output


def _load_prediction_array(manifest: Mapping[str, Any], cell_id: str, role: str) -> np.ndarray:
    matches = [
        row for row in manifest["arrays"] if row["cell_run_id"] == cell_id and row["role"] == role
    ]
    _require(len(matches) == 1, "prediction array identity changed")
    row = matches[0]
    path = _repo_path(predictions.PRIVATE_DIRECTORY / row["relative_path"])
    _require(path.is_file(), "prediction array missing")
    _require(path.stat().st_size == row["bytes"], "prediction array size changed")
    _require(file_sha256(path) == row["file_sha256"], "prediction array bytes changed")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SixGalaxyScoreError("invalid prediction array") from exc
    _require(list(value.shape) == row["shape"], "prediction array shape changed")
    _require(str(value.dtype) == row["dtype"], "prediction array dtype changed")
    _require(predictions.array_sha256(value) == row["array_sha256"], "prediction values changed")
    return np.asarray(value)


def _response_rows(preflight: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows = {
        (row["object_id"], row["resolution"], row["observable"]): row
        for row in preflight["response_assets"]
    }
    object_ids = [row["object_id"] for row in preflight["objects"]]
    expected = {
        (object_id, resolution, observable)
        for object_id in object_ids
        for resolution in _RESOLUTIONS
        for observable in ("MOM1", "MOM2")
    }
    _require(set(rows) == expected, "response inventory changed")
    return rows


def _load_response(row: Mapping[str, Any]) -> tuple[np.ndarray, fits.Header]:
    path = _repo_path(row["relative_path"])
    _require(path.is_file(), "response file missing")
    _require(path.stat().st_size == row["bytes"], "response size changed")
    _require(file_sha256(path) == row["sha256"], "response bytes changed")
    try:
        raw, header = fits.getdata(path, header=True, memmap=False)
    except (OSError, ValueError) as exc:
        raise SixGalaxyScoreError("invalid response array") from exc
    value = np.asarray(raw, dtype=np.float64).squeeze()
    _require(value.shape == (1024, 1024), "response array shape changed")
    _require(str(header["BUNIT"]).strip() == "METR/SEC", "response unit changed")
    return value, header


def _common_mask(
    manifest: Mapping[str, Any],
    cell_id: str,
    resolution: str,
    observed: np.ndarray,
    dispersion: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    eligibility = _load_prediction_array(
        manifest, cell_id, f"{resolution.lower()}_eligibility"
    ).astype(bool)
    arrays = {
        candidate: _load_prediction_array(manifest, cell_id, f"{candidate}__{resolution}")
        for candidate in _CANDIDATES
    }
    mask = eligibility & np.isfinite(observed) & np.isfinite(dispersion) & (dispersion > 0.0)
    for values in arrays.values():
        mask &= np.isfinite(values)
    _require(int(np.count_nonzero(mask)) > 0, "common score mask is empty")
    return mask, arrays


def _nuisance_values(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    responses: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray, fits.Header]],
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    references = config["score_contract"]["nuisance_reference_cell_by_object"]
    for object_id in config["response_contract"]["object_order"]:
        reference = next(cell for cell in cells if cell["cell_run_id"] == references[object_id])
        output[object_id] = {}
        for resolution in _RESOLUTIONS:
            observed, dispersion, header = responses[(object_id, resolution)]
            mask, _arrays = _common_mask(
                manifest, reference["cell_run_id"], resolution, observed, dispersion
            )
            ra, dec = predictions.method._world_grid(header)
            major, _disk_y, _radius, _cosine = predictions.method._disk_sky_coordinates(
                reference["geometry"], ra, dec
            )
            x = np.asarray(major[mask], dtype=np.float64)
            y = np.asarray(observed[mask], dtype=np.float64)
            covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
            _require(math.isfinite(covariance) and covariance != 0.0, "rotation sign undefined")
            output[object_id][resolution] = {
                "rotation_sign": 1.0 if covariance > 0.0 else -1.0,
                "major_axis_velocity_covariance_kpc_m_s": covariance,
                "shared_systemic_velocity_m_s": float(np.mean(y)),
                "reference_common_pixel_count": int(y.size),
            }
    return output


def _model_metrics(
    observed: np.ndarray,
    dispersion: np.ndarray,
    predicted_plus: np.ndarray,
    mask: np.ndarray,
    *,
    sign: float,
    shared_systemic: float,
    minimum_dispersion: float,
) -> dict[str, Any]:
    obs = np.asarray(observed[mask], dtype=np.float64)
    plus = np.asarray(predicted_plus[mask], dtype=np.float64)
    prediction = sign * plus + shared_systemic
    residual = obs - prediction
    scale = np.maximum(np.asarray(dispersion[mask], dtype=np.float64), minimum_dispersion)
    best_offset = float(np.mean(obs - sign * plus))
    best_residual = obs - (sign * plus + best_offset)
    opposite_residual = obs - (-sign * plus + shared_systemic)
    return {
        "shared_systemic_velocity_m_s": shared_systemic,
        "rmse_m_s": float(np.sqrt(np.mean(residual**2))),
        "mae_m_s": float(np.mean(np.abs(residual))),
        "median_absolute_residual_m_s": float(np.median(np.abs(residual))),
        "mom2_scaled_mean_squared_residual": float(np.mean((residual / scale) ** 2)),
        "residual_mean_m_s": float(np.mean(residual)),
        "residual_count": int(residual.size),
        "model_specific_best_offset_diagnostic": {
            "offset_m_s": best_offset,
            "rmse_m_s": float(np.sqrt(np.mean(best_residual**2))),
        },
        "opposite_sign_fixed_systemic_control": {
            "rmse_m_s": float(np.sqrt(np.mean(opposite_residual**2)))
        },
    }


def _score_cell(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    resolution: str,
    responses: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray, fits.Header]],
    nuisance: Mapping[str, Mapping[str, Mapping[str, float]]],
    response_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    object_id = cell["object_id"]
    observed, dispersion, header = responses[(object_id, resolution)]
    mask, arrays = _common_mask(manifest, cell["cell_run_id"], resolution, observed, dispersion)
    values = nuisance[object_id][resolution]
    metrics = {
        candidate: _model_metrics(
            observed,
            dispersion,
            arrays[candidate],
            mask,
            sign=float(values["rotation_sign"]),
            shared_systemic=float(values["shared_systemic_velocity_m_s"]),
            minimum_dispersion=float(config["response_contract"]["minimum_dispersion_scale_m_s"]),
        )
        for candidate in _CANDIDATES
    }
    ranking = sorted(_CANDIDATES, key=lambda candidate: (metrics[candidate]["rmse_m_s"], candidate))
    observed_values = observed[mask]
    systemic_residual = observed_values - float(values["shared_systemic_velocity_m_s"])
    response_row = response_rows[(object_id, resolution, "MOM1")]
    beam_values = response_row["beam_deg"]
    beam_area_deg2 = math.pi * float(beam_values[0]) * float(beam_values[1]) / (4.0 * math.log(2.0))
    pixel_area_deg2 = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    rg_rmse = metrics[_RG]["rmse_m_s"]
    comparator_fractional = {
        candidate: float(
            (metrics[candidate]["rmse_m_s"] - rg_rmse) / metrics[candidate]["rmse_m_s"]
        )
        for candidate in _CANDIDATES[:3]
    }
    return {
        "cell_score_id": f"{cell['cell_run_id']}__{resolution}",
        "cell_run_id": cell["cell_run_id"],
        "object_id": object_id,
        "inclination_stratum": config["response_contract"]["inclination_strata"][object_id],
        "conversion_cell_id": cell["conversion_cell_id"],
        "geometry_variant_id": cell["geometry"]["geometry_variant_id"],
        "inclination_deg": cell["geometry"]["inclination_deg"],
        "resolution": resolution,
        "response_asset_sha256": response_row["sha256"],
        "common_pixel_count": int(np.count_nonzero(mask)),
        "beam_equivalent_count": float(np.count_nonzero(mask) * pixel_area_deg2 / beam_area_deg2),
        "observed_mean_m_s": float(np.mean(observed_values)),
        "observed_standard_deviation_m_s": float(np.std(observed_values)),
        "median_mom2_m_s": float(np.median(dispersion[mask])),
        "systemic_only_control_rmse_m_s": float(np.sqrt(np.mean(systemic_residual**2))),
        "shared_nuisance": dict(values),
        "models": metrics,
        "rmse_ranking": ranking,
        "winner": ranking[0],
        "rg_fractional_rmse_improvement_over_comparators": comparator_fractional,
        "rg_beats_all_three_comparators": all(
            value > 0.0 for value in comparator_fractional.values()
        ),
    }


def _adjudicate_primary(
    config: Mapping[str, Any], scores: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    primary_ids = config["score_contract"]["primary_cell_by_object"]
    primary = [
        next(row for row in scores if row["cell_score_id"] == cell_id)
        for cell_id in primary_ids.values()
    ]
    aggregates = {
        candidate: float(np.mean([row["models"][candidate]["rmse_m_s"] for row in primary]))
        for candidate in _CANDIDATES
    }
    aggregate_ranking = sorted(
        _CANDIDATES, key=lambda candidate: (aggregates[candidate], candidate)
    )
    rg_wins = [row for row in primary if row["rg_beats_all_three_comparators"]]
    win_strata = sorted({row["inclination_stratum"] for row in rg_wins})
    all_strata = sorted(set(config["response_contract"]["inclination_strata"].values()))
    conditions = {
        "minimum_object_wins": len(rg_wins)
        >= int(config["score_contract"]["minimum_rg_primary_object_wins_for_strong_signal"]),
        "equal_object_aggregate_win": aggregate_ranking[0] == _RG,
        "all_inclination_strata": win_strata == all_strata,
    }
    strong = all(conditions.values())
    mixed = len(rg_wins) >= 2 or aggregates[_RG] < aggregates["NEWTON_3D_DST"]
    if strong:
        decision = "STRONG_EXTERNAL_SIX_GALAXY_RG_2D_REPLICATION_SIGNAL"
    elif mixed:
        decision = "MIXED_EXTERNAL_RG_2D_SIGNAL_RETAIN_FOR_FOLLOW_UP"
    else:
        decision = "EXTERNAL_SIX_GALAXY_RG_2D_SIGNAL_NOT_REPLICATED"
    return {
        "decision": decision,
        "primary_cells": primary,
        "primary_object_winners": {row["object_id"]: row["winner"] for row in primary},
        "rg_primary_object_wins": len(rg_wins),
        "rg_primary_win_objects": [row["object_id"] for row in rg_wins],
        "rg_primary_win_inclination_strata": win_strata,
        "equal_object_mean_primary_rmse_m_s": aggregates,
        "equal_object_aggregate_ranking": aggregate_ranking,
        "strong_signal_conditions": conditions,
        "strong_external_replication": strong,
        "mixed_signal": mixed,
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    _prediction_config, prediction_receipt, manifest = _load_prediction_evidence(config)
    preflight = _load_preflight(config)
    cells = _cell_payloads(prediction_receipt, manifest)
    response_rows = _response_rows(preflight)
    responses: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, fits.Header]] = {}
    for object_id in config["response_contract"]["object_order"]:
        for resolution in _RESOLUTIONS:
            observed, header = _load_response(response_rows[(object_id, resolution, "MOM1")])
            dispersion, dispersion_header = _load_response(
                response_rows[(object_id, resolution, "MOM2")]
            )
            _require(
                [header["NAXIS1"], header["NAXIS2"]]
                == [dispersion_header["NAXIS1"], dispersion_header["NAXIS2"]],
                "response grids differ",
            )
            responses[(object_id, resolution)] = (observed, dispersion, header)
    nuisance = _nuisance_values(config, manifest, cells, responses)
    scores = [
        _score_cell(config, manifest, cell, resolution, responses, nuisance, response_rows)
        for cell in cells
        for resolution in _RESOLUTIONS
    ]
    _require(len(scores) == 30, "score cell count changed")
    adjudication = _adjudicate_primary(config, scores)
    win_counts = {
        candidate: sum(row["winner"] == candidate for row in scores) for candidate in _CANDIDATES
    }
    object_win_counts = {
        object_id: {
            candidate: sum(
                row["object_id"] == object_id and row["winner"] == candidate for row in scores
            )
            for candidate in _CANDIDATES
        }
        for object_id in config["response_contract"]["object_order"]
    }
    claims = dict(config["claim_boundary"])
    claims["real_response_scored"] = True
    claims["fixed_predictions_scored"] = True
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_FIXED_SIX_EXTERNAL_GALAXY_REAL_THINGS_2D_PIXEL_SCORE",
        "decision": adjudication["decision"],
        "package_bindings": _package_bindings(),
        "prediction_receipt_content_sha256": prediction_receipt["content_sha256"],
        "prediction_manifest_content_sha256": manifest["content_sha256"],
        "preflight_receipt_content_sha256": config["preflight_binding"]["receipt_content_sha256"],
        "primary_sources": config["primary_sources"],
        "score_contract": config["score_contract"],
        "retained_source_failures": prediction_receipt["retained_source_failures"],
        "shared_nuisance_by_object_resolution": nuisance,
        "scores": scores,
        "primary_adjudication": adjudication,
        "aggregate": {
            "winner_counts_all_30_cells": win_counts,
            "winner_counts_by_object": object_win_counts,
            "rg_all_comparator_win_cells": sum(
                row["rg_beats_all_three_comparators"] for row in scores
            ),
            "all_cells_reported": len(scores) == 30,
            "independent_object_count": 6,
            "source_conversion_and_resolution_cells_are_not_independent": True,
        },
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": claims,
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(dict(receipt) == build_receipt(config), "receipt differs from deterministic rebuild")


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


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    receipt = _read_json(path, "receipt")
    validate_receipt(config, receipt)
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    if not _repo_path(OUTPUT_PATH).is_file():
        return {"package_id": config["package_id"], "status": "FROZEN_UNRUN"}
    receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
    return {
        "package_id": config["package_id"],
        "status": receipt["status"],
        "decision": receipt["decision"],
        "rg_primary_object_wins": receipt["primary_adjudication"]["rg_primary_object_wins"],
        "primary_object_winners": receipt["primary_adjudication"]["primary_object_winners"],
        "winner_counts": receipt["aggregate"]["winner_counts_all_30_cells"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    sub.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        print(write_receipt())
    elif arguments.command == "check":
        print(check_receipt())
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
