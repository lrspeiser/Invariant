"""Score sealed Holmberg II two-dimensional predictions against sealed THINGS maps."""

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
    open_gravity_rg_holmberg_ii_things_2d_response_blind_predictions_v1 as predictions,
)

CONFIG_PATH = Path("configs/open_gravity_rg_holmberg_ii_things_2d_fixed_score_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_holmberg_ii_things_2d_fixed_score_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_holmberg_ii_things_2d_fixed_score_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-holmberg-ii-things-2d-fixed-score-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-things-2d-fixed-score-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-things-2d-fixed-score-receipt-1.0"
_CANDIDATES = (
    "NEWTON_3D_DST",
    "RAR_2016_ON_NEWTON_3D",
    "MOND_STANDARD_MU_ON_NEWTON_3D",
    "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
)
_RESOLUTIONS = ("NATURAL", "ROBUST")
_REFERENCE_CELL_ID = "UGC04305__IRAC1_FIXED_ML0P6__I38P0"
_PRIMARY_CELL_ID = f"{_REFERENCE_CELL_ID}__NATURAL"
_CONFIG_RAW_SHA256 = "3d2a0303ef88f2055810e0c1d48ec18e2529e142c118d67e1b0d7f49fe070f10"
_CONFIG_CONTENT_SHA256 = "7bbe06ae4bb4e1e97cc40faaad31ae1e804e2e113d8b648b1147842d2d6efc97"
_MODULE_SEMANTIC_SHA256 = "ec93ce425d2c9c91c18ecf2428ff9142170b77c2d24441ff051b25da1fd15d6a"
_TEST_RAW_SHA256 = "75706ecf0afbaebb4e66b385bb715c2a7d40be38cb328c8b57d03fdc9c2ebf1a"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class HolmbergScoreError(RuntimeError):
    """Raised when a frozen prediction, response, score, or seal gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HolmbergScoreError(message)


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
        raise HolmbergScoreError(f"invalid {label}") from exc
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
        config["status"] == "FROZEN_PREREGISTERED_HOLMBERG_II_THINGS_2D_FIXED_SCORE",
        "status changed",
    )
    prediction = config["prediction_binding"]
    _require(prediction["source_cells"] == 9, "prediction cells changed")
    _require(prediction["array_file_count"] == 117, "prediction arrays changed")
    _require(prediction["candidate_resolution_predictions"] == 72, "predictions changed")
    _require(prediction["all_solver_gates_pass"] is True, "failed predictions admitted")
    _require(prediction["response_pixels_used_to_build_predictions"] == 0, "prediction leak")
    response = config["response_contract"]
    _require(response["resolutions"] == list(_RESOLUTIONS), "resolutions changed")
    _require(response["observables"] == ["MOM1", "MOM2"], "observables changed")
    _require(response["files"] == 4, "response file count changed")
    _require(response["bytes"] == 16_951_680, "response bytes changed")
    _require(response["decoded_array_slots"] == 4_194_304, "response slots changed")
    _require(response["minimum_dispersion_scale_m_s"] == 3000.0, "dispersion floor changed")
    _require(
        response["dispersion_metric_is_not_measurement_likelihood"] is True, "likelihood overclaim"
    )
    score = config["score_contract"]
    _require(score["candidate_ids"] == list(_CANDIDATES), "candidate score inventory changed")
    _require(score["source_cells"] == 9, "source score count changed")
    _require(score["source_resolution_cells"] == 18, "cell score count changed")
    _require(score["model_scores"] == 72, "model score count changed")
    _require(score["primary_cell_id"] == _PRIMARY_CELL_ID, "primary cell changed")
    _require(
        score["nuisance_reference_cell_id"] == _REFERENCE_CELL_ID, "nuisance reference changed"
    )
    _require(
        score["primary_metric"]
        == "unweighted common-pixel velocity RMSE using the shared sign and shared systemic velocity",
        "primary metric changed",
    )
    for key in (
        "per_model_sign_selection",
        "source_or_geometry_reselection",
        "response_parameter_tuning",
        "p_values_computed",
    ):
        _require(score[key] is False, f"forbidden score behavior enabled: {key}")
    _require(score["all_18_cells_reported"] is True, "cell suppression enabled")
    _require(score["retain_every_failure_and_counterexample"] is True, "failure retention removed")
    boundary = config["scientific_boundary"]
    _require(boundary["response_files_opened"] == 4, "response accounting changed")
    _require(boundary["response_array_slots_decoded"] == 4_194_304, "decode accounting changed")
    _require(boundary["model_scores"] == 72, "score accounting changed")
    for key in ("network_calls", "model_calls", "paid_calls", "tuning_calls"):
        _require(boundary[key] == 0, f"forbidden access enabled: {key}")
    _require(boundary["general_3d_validated"] is False, "3D overclaim")
    _require(boundary["model_lifted_2p5d_only"] is True, "2.5D caveat removed")
    claims = config["claim_boundary"]
    _require(claims["real_response_scored"] is True, "score claim removed")
    _require(claims["fixed_predictions_scored"] is True, "fixed score removed")
    _require(claims["source_matched_internal_replication_only"] is True, "independence overclaim")
    for key in (
        "dispersion_metric_is_likelihood",
        "gas_dynamics_solved",
        "inclination_resolved",
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
    _require(receipt["candidate_resolution_predictions"] == 72, "prediction count changed")
    _require(receipt["all_solver_gates_pass"] is True, "failed prediction packet admitted")
    _require(
        receipt["response_boundary"]["velocity_pixels_decoded"] == 0, "prediction response leak"
    )
    return prediction_config, receipt, manifest


def _load_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["preflight_binding"]
    _verify_artifacts(binding, "preflight")
    value = _read_json(_repo_path(binding["artifacts"][0]["path"]), "preflight config")
    receipt = _read_json(_repo_path(binding["artifacts"][-1]["path"]), "preflight receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "preflight receipt changed"
    )
    return value


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
    _require(len(output) == 9, "prediction cell inventory changed")
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
        raise HolmbergScoreError("invalid prediction array") from exc
    _require(list(value.shape) == row["shape"], "prediction array shape changed")
    _require(str(value.dtype) == row["dtype"], "prediction array dtype changed")
    _require(predictions.array_sha256(value) == row["array_sha256"], "prediction values changed")
    return np.asarray(value)


def _response_rows(preflight: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = {(row["resolution"], row["observable"]): row for row in preflight["response_assets"]}
    _require(
        set(rows)
        == {("NATURAL", "MOM1"), ("NATURAL", "MOM2"), ("ROBUST", "MOM1"), ("ROBUST", "MOM2")},
        "response inventory changed",
    )
    return rows


def _load_response(row: Mapping[str, Any]) -> tuple[np.ndarray, fits.Header]:
    path = _repo_path(row["relative_path"])
    _require(path.is_file(), "response file missing")
    _require(path.stat().st_size == row["bytes"], "response size changed")
    _require(file_sha256(path) == row["sha256"], "response bytes changed")
    try:
        raw, header = fits.getdata(path, header=True, memmap=False)
    except (OSError, ValueError) as exc:
        raise HolmbergScoreError("invalid response array") from exc
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
    manifest: Mapping[str, Any],
    reference_cell: Mapping[str, Any],
    responses: Mapping[str, tuple[np.ndarray, np.ndarray, fits.Header]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for resolution in _RESOLUTIONS:
        observed, dispersion, header = responses[resolution]
        mask, _arrays = _common_mask(
            manifest, reference_cell["cell_run_id"], resolution, observed, dispersion
        )
        ra, dec = predictions._world_grid(header)
        major, _disk_y, _radius, _cosine = predictions._disk_sky_coordinates(
            reference_cell["geometry"], ra, dec
        )
        x = np.asarray(major[mask], dtype=np.float64)
        y = np.asarray(observed[mask], dtype=np.float64)
        covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
        _require(math.isfinite(covariance) and covariance != 0.0, "rotation sign undefined")
        output[resolution] = {
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
    responses: Mapping[str, tuple[np.ndarray, np.ndarray, fits.Header]],
    nuisance: Mapping[str, Mapping[str, float]],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    observed, dispersion, header = responses[resolution]
    mask, arrays = _common_mask(manifest, cell["cell_run_id"], resolution, observed, dispersion)
    values = nuisance[resolution]
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
    response_row = next(
        row
        for row in preflight["response_assets"]
        if row["resolution"] == resolution and row["observable"] == "MOM1"
    )
    beam_values = preflight["exact_header_contract"][f"{resolution.lower()}_beam_deg"]
    beam_area_deg2 = math.pi * float(beam_values[0]) * float(beam_values[1]) / (4.0 * math.log(2.0))
    pixel_area_deg2 = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    rg_rmse = metrics[_CANDIDATES[3]]["rmse_m_s"]
    comparator_fractional = {
        candidate: float(
            (metrics[candidate]["rmse_m_s"] - rg_rmse) / metrics[candidate]["rmse_m_s"]
        )
        for candidate in _CANDIDATES[:3]
    }
    return {
        "cell_score_id": f"{cell['cell_run_id']}__{resolution}",
        "cell_run_id": cell["cell_run_id"],
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


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    _prediction_config, prediction_receipt, manifest = _load_prediction_evidence(config)
    preflight = _load_preflight(config)
    cells = _cell_payloads(prediction_receipt, manifest)
    response_rows = _response_rows(preflight)
    responses: dict[str, tuple[np.ndarray, np.ndarray, fits.Header]] = {}
    for resolution in _RESOLUTIONS:
        observed, header = _load_response(response_rows[(resolution, "MOM1")])
        dispersion, dispersion_header = _load_response(response_rows[(resolution, "MOM2")])
        _require(
            [header["NAXIS1"], header["NAXIS2"]]
            == [dispersion_header["NAXIS1"], dispersion_header["NAXIS2"]],
            "response grids differ",
        )
        responses[resolution] = (observed, dispersion, header)
    reference = next(cell for cell in cells if cell["cell_run_id"] == _REFERENCE_CELL_ID)
    nuisance = _nuisance_values(manifest, reference, responses)
    scores = [
        _score_cell(config, manifest, cell, resolution, responses, nuisance, preflight)
        for cell in cells
        for resolution in _RESOLUTIONS
    ]
    _require(len(scores) == 18, "score cell count changed")
    primary = next(row for row in scores if row["cell_score_id"] == _PRIMARY_CELL_ID)
    win_counts = {
        candidate: sum(row["winner"] == candidate for row in scores) for candidate in _CANDIDATES
    }
    resolution_win_counts = {
        resolution: {
            candidate: sum(
                row["resolution"] == resolution and row["winner"] == candidate for row in scores
            )
            for candidate in _CANDIDATES
        }
        for resolution in _RESOLUTIONS
    }
    rg_primary = primary["rg_beats_all_three_comparators"]
    decision = (
        "INTERESTING_HOLMBERG_II_RG_SIGNAL_REPLICATES_ON_FIXED_PRIMARY_2D_CELL"
        if rg_primary
        else "HOLMBERG_II_RG_SIGNAL_DOES_NOT_REPLICATE_ON_FIXED_PRIMARY_2D_CELL"
    )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_FIXED_HOLMBERG_II_REAL_THINGS_2D_PIXEL_SCORE",
        "decision": decision,
        "package_bindings": _package_bindings(),
        "prediction_receipt_content_sha256": prediction_receipt["content_sha256"],
        "prediction_manifest_content_sha256": manifest["content_sha256"],
        "preflight_receipt_content_sha256": config["preflight_binding"]["receipt_content_sha256"],
        "primary_sources": config["primary_sources"],
        "score_contract": config["score_contract"],
        "shared_nuisance_by_resolution": nuisance,
        "scores": scores,
        "primary_cell": primary,
        "aggregate": {
            "winner_counts_all_18_cells": win_counts,
            "winner_counts_by_resolution": resolution_win_counts,
            "rg_all_comparator_win_cells": sum(
                row["rg_beats_all_three_comparators"] for row in scores
            ),
            "all_cells_reported": len(scores) == 18,
            "independent_object_count": 1,
            "geometry_and_conversion_sensitivity_cells_are_not_independent": True,
        },
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    rebuilt = build_receipt(config)
    _require(dict(receipt) == rebuilt, "receipt differs from deterministic rebuild")


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
        "primary_winner": receipt["primary_cell"]["winner"],
        "rg_primary_replication": receipt["primary_cell"]["rg_beats_all_three_comparators"],
        "winner_counts": receipt["aggregate"]["winner_counts_all_18_cells"],
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
