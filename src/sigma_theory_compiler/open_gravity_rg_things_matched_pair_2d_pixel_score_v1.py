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
    open_gravity_rg_things_matched_pair_2d_source_predictions_v1 as predictions,
)

CONFIG_PATH = Path("configs/open_gravity_rg_things_matched_pair_2d_pixel_score_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_matched_pair_2d_pixel_score_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_matched_pair_2d_pixel_score_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-matched-pair-2d-pixel-score-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-pixel-score-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-pixel-score-receipt-1.0"
_OBJECTS = ("NGC2976", "NGC4214")
_MODELS = ("NEWTON_3D_DST", "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG")
_MODEL_ARRAYS = {
    _MODELS[0]: "newton_vlos_plus_m_s",
    _MODELS[1]: "rg_vlos_plus_m_s",
}
_CONFIG_RAW_SHA256 = "f4427d3ec0a59cae1415b224ff7a02c1ace35fc23738b981eeb647aecde1ccb6"
_CONFIG_CONTENT_SHA256 = "3787c3397fc07fe1c4117d99bd9e4ba00b470ddbab93f1e827889ddf76208c97"
_MODULE_SEMANTIC_SHA256 = "4c646aa0c547425ae3b35a1d5f0202a61e74046db1aea3489ca15a6e87348a5e"
_TEST_RAW_SHA256 = "7c3dcbc2aafd9ab402139af49060720f645e65dfb41c4bfd1237a3384026a4f0"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class MatchedPairPixelScoreError(RuntimeError):
    """Raised when a response, score, or evidence invariant fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedPairPixelScoreError(message)


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
        raise MatchedPairPixelScoreError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_PREREGISTERED_MATCHED_PAIR_THINGS_PIXEL_SCORE",
        "status changed",
    )
    admission = config["admission_rule"]
    for key in (
        "real_public_response_data_required",
        "primary_measurement_and_method_papers_required",
        "target_free_operator_and_projection_benchmarks_required",
        "newtonian_and_systemic_only_controls_required",
        "spherical_or_1d_data_cannot_validate_general_3d",
        "one_failure_never_prunes_family",
    ):
        _require(admission[key] is True, f"admission gate changed: {key}")
    _require(admission["missing_data_disposition"] == "SOURCE_BLOCKED", "source rule changed")
    _require(
        admission["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY",
        "paper-only rule changed",
    )
    _require(
        admission["model_lifted_vertical_structure_disposition"] == "MODEL_LIFTED_2P5D",
        "2.5D label changed",
    )
    binding = config["prediction_binding"]
    _require(binding["array_count"] == 18, "prediction array count changed")
    _require(
        binding["array_root_sha256"]
        == "5216014b614d0b6a5786b4b6bd13d6c14b24d6a3214d6464b9b6fe4642f0c649",
        "prediction root changed",
    )
    _require([row["object_id"] for row in config["objects"]] == list(_OBJECTS), "objects changed")
    _require(
        config["objects"][1]["inference_role"] == "LOW_INCLINATION_STRESS_TEST_RETAINED",
        "NGC4214 caveat changed",
    )
    _require(
        config["objects"][1]["standard_rotation_curve_claim_allowed"] is False,
        "NGC4214 overclaim",
    )
    response = config["response_contract"]
    _require(response["expected_files"] == 4, "response file count changed")
    _require(response["expected_bytes"] == 16_974_720, "response bytes changed")
    _require(response["expected_array_slots"] == 4_194_304, "response slots changed")
    _require(response["minimum_dispersion_scale_m_s"] == 3000.0, "dispersion floor changed")
    _require(
        response["dispersion_metric_is_not_measurement_likelihood"] is True,
        "dispersion overclaim",
    )
    score = config["score_contract"]
    _require(score["models"] == list(_MODELS), "models changed")
    _require(score["rotation_sign_shared_by_all_models"] is True, "sign sharing removed")
    _require(score["opposite_sign_control_retained"] is True, "sign control removed")
    _require(score["response_tuning_calls"] == 0, "response tuning enabled")
    _require(score["per_model_sign_selection"] is False, "per-model sign enabled")
    _require(score["source_or_geometry_reselection"] is False, "source reselection enabled")
    _require(score["p_values_computed"] is False, "p-value overclaim")
    rule = config["interesting_signal_rule"]
    _require(rule["primary_object"] == "NGC2976", "primary object changed")
    _require(
        rule["minimum_primary_rmse_fractional_improvement_over_newton"] == 0.05,
        "signal gate changed",
    )
    _require(
        rule["interesting_signal_is_not_theory_confirmation"] is True, "confirmation overclaim"
    )
    boundary = config["scientific_boundary"]
    _require(boundary["response_files_opened"] == 4, "response access changed")
    _require(boundary["response_array_slots_decoded"] == 4_194_304, "decoded slots changed")
    _require(boundary["objects_scored"] == 2, "object accounting changed")
    _require(boundary["models_scored"] == 2, "model accounting changed")
    for key in ("network_calls", "model_calls", "paid_calls", "tuning_calls"):
        _require(boundary[key] == 0, f"forbidden call enabled: {key}")
    _require(boundary["general_3d_validated"] is False, "3D overclaim")
    _require(boundary["model_lifted_2p5d_only"] is True, "2.5D caveat removed")
    claims = config["claim_boundary"]
    for key in (
        "ngc4214_standard_rotation_curve_valid",
        "dispersion_metric_is_likelihood",
        "gas_dynamics_solved",
        "general_3d_validated",
        "refracted_gravity_confirmed",
        "unique_theory_established",
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


def _load_prediction_evidence(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = config["prediction_binding"]
    for row in binding["artifacts"]:
        path = _repo_path(row["path"])
        _require(path.is_file(), "prediction artifact missing")
        _require(file_sha256(path) == row["sha256"], "prediction artifact changed")
    receipt = _read_json(_repo_path(predictions.OUTPUT_PATH), "prediction receipt")
    manifest = _read_json(_repo_path(predictions.PRIVATE_MANIFEST_PATH), "prediction manifest")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "prediction receipt changed"
    )
    _require(
        manifest["content_sha256"] == binding["manifest_content_sha256"],
        "prediction manifest changed",
    )
    _require(manifest["array_count"] == binding["array_count"], "prediction array count changed")
    _require(
        manifest["array_root_sha256"] == binding["array_root_sha256"],
        "prediction array root changed",
    )
    _require(
        receipt["scientific_boundary"]["velocity_pixel_values_decoded"] == 0,
        "prediction response leakage",
    )
    _require(
        receipt["claim_boundary"]["scientific_fit_tested"] is False, "prediction already scored"
    )
    return receipt, manifest


def _load_prediction_array(
    manifest: Mapping[str, Any], object_id: str, array_id: str
) -> np.ndarray:
    row = next(
        item
        for item in manifest["arrays"]
        if item["object_id"] == object_id and item["array_id"] == array_id
    )
    path = _repo_path(row["relative_path"])
    _require(path.is_file(), "prediction array missing")
    _require(file_sha256(path) == row["file_sha256"], "prediction array changed")
    value = np.load(path, allow_pickle=False)
    _require(list(value.shape) == row["shape"], "prediction array shape changed")
    _require(str(value.dtype) == row["dtype"], "prediction array dtype changed")
    _require(
        predictions.array_sha256(value) == row["array_sha256"], "prediction array content changed"
    )
    return np.asarray(value)


def _response_rows(
    prediction_config: Mapping[str, Any], object_id: str
) -> dict[str, dict[str, Any]]:
    source = predictions._response_source_config(prediction_config)
    rows = {row["role"]: row for row in source["files"] if row["object_id"] == object_id}
    _require(
        set(rows) == set(load_config(verify_package=False)["response_contract"]["roles"]),
        "response roles changed",
    )
    return rows


def _load_response(row: Mapping[str, Any]) -> tuple[np.ndarray, fits.Header]:
    path = _repo_path(row["relative_path"])
    _require(path.is_file(), "response file missing")
    _require(path.stat().st_size == row["bytes"], "response bytes changed")
    _require(file_sha256(path) == row["sha256"], "response hash changed")
    raw, header = fits.getdata(path, header=True, memmap=False)
    value = np.asarray(raw, dtype=np.float64).squeeze()
    _require(value.shape == (1024, 1024), "response shape changed")
    _require(header["BUNIT"] == "METR/SEC", "response unit changed")
    return value, header


def _rotation_sign(
    major_kpc: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    x = np.asarray(major_kpc[mask], dtype=np.float64)
    y = np.asarray(observed[mask], dtype=np.float64)
    covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
    _require(math.isfinite(covariance) and covariance != 0.0, "rotation sign is undefined")
    return (1.0 if covariance > 0.0 else -1.0), covariance


def _model_metrics(
    observed: np.ndarray,
    dispersion: np.ndarray,
    predicted_plus: np.ndarray,
    mask: np.ndarray,
    sign: float,
    minimum_dispersion: float,
) -> dict[str, Any]:
    obs = np.asarray(observed[mask], dtype=np.float64)
    prediction = sign * np.asarray(predicted_plus[mask], dtype=np.float64)
    systemic = float(np.mean(obs - prediction))
    residual = obs - (prediction + systemic)
    scale = np.maximum(np.asarray(dispersion[mask], dtype=np.float64), minimum_dispersion)
    opposite = -sign * np.asarray(predicted_plus[mask], dtype=np.float64)
    opposite_offset = float(np.mean(obs - opposite))
    opposite_residual = obs - (opposite + opposite_offset)
    return {
        "systemic_offset_m_s": systemic,
        "rmse_m_s": float(np.sqrt(np.mean(residual**2))),
        "mae_m_s": float(np.mean(np.abs(residual))),
        "median_absolute_residual_m_s": float(np.median(np.abs(residual))),
        "mom2_scaled_mean_squared_residual": float(np.mean((residual / scale) ** 2)),
        "opposite_sign_control": {
            "systemic_offset_m_s": opposite_offset,
            "rmse_m_s": float(np.sqrt(np.mean(opposite_residual**2))),
        },
        "residual_mean_m_s": float(np.mean(residual)),
        "residual_count": int(residual.size),
    }


def _score_object(
    config: Mapping[str, Any],
    prediction_config: Mapping[str, Any],
    prediction_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    object_id: str,
) -> dict[str, Any]:
    rows = _response_rows(prediction_config, object_id)
    observed, header = _load_response(rows["HI_MOM1_NATURAL_VELOCITY_FIELD"])
    dispersion, dispersion_header = _load_response(rows["HI_MOM2_NATURAL_VELOCITY_DISPERSION"])
    _require(
        [header["NAXIS1"], header["NAXIS2"]]
        == [dispersion_header["NAXIS1"], dispersion_header["NAXIS2"]],
        "response grids differ",
    )
    eligibility = _load_prediction_array(manifest, object_id, "source_eligibility").astype(bool)
    intensity = _load_prediction_array(manifest, object_id, "source_intensity")
    model_arrays = {
        model_id: _load_prediction_array(manifest, object_id, array_id)
        for model_id, array_id in _MODEL_ARRAYS.items()
    }
    object_prediction = next(
        row for row in prediction_receipt["objects"] if row["object_id"] == object_id
    )
    ra, dec = predictions._world_grid(header)
    major, _disk_y, _radius, _cosine = predictions._disk_sky_coordinates(
        object_prediction["geometry"], ra, dec
    )
    common = (
        eligibility
        & np.isfinite(intensity)
        & (intensity > 0.0)
        & np.isfinite(observed)
        & np.isfinite(dispersion)
        & (dispersion > 0.0)
    )
    for value in model_arrays.values():
        common &= np.isfinite(value)
    count = int(np.count_nonzero(common))
    _require(count > 0, f"no common score pixels for {object_id}")
    sign, covariance = _rotation_sign(major, observed, common)
    minimum_dispersion = float(config["response_contract"]["minimum_dispersion_scale_m_s"])
    metrics = {
        model_id: _model_metrics(
            observed, dispersion, model_arrays[model_id], common, sign, minimum_dispersion
        )
        for model_id in _MODELS
    }
    observed_values = observed[common]
    systemic_only = observed_values - float(np.mean(observed_values))
    newton_rmse = metrics[_MODELS[0]]["rmse_m_s"]
    rg_rmse = metrics[_MODELS[1]]["rmse_m_s"]
    fractional_improvement = float((newton_rmse - rg_rmse) / newton_rmse)
    pixel_area_deg2 = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    beam = rows["HI_MOM1_NATURAL_VELOCITY_FIELD"]
    beam_area_deg2 = (
        math.pi
        * float(beam["beam_major_deg"])
        * float(beam["beam_minor_deg"])
        / (4.0 * math.log(2.0))
    )
    inference = next(row for row in config["objects"] if row["object_id"] == object_id)
    return {
        "object_id": object_id,
        "inference_role": inference["inference_role"],
        "standard_rotation_curve_claim_allowed": inference["standard_rotation_curve_claim_allowed"],
        "response": {
            "common_pixel_count": count,
            "finite_mom1_count": int(np.count_nonzero(np.isfinite(observed))),
            "finite_positive_mom2_count": int(
                np.count_nonzero(np.isfinite(dispersion) & (dispersion > 0.0))
            ),
            "mean_velocity_m_s": float(np.mean(observed_values)),
            "velocity_standard_deviation_m_s": float(np.std(observed_values)),
            "median_dispersion_m_s": float(np.median(dispersion[common])),
            "beam_area_pixels": float(beam_area_deg2 / pixel_area_deg2),
            "beam_equivalent_count": float(count * pixel_area_deg2 / beam_area_deg2),
        },
        "rotation_sign": sign,
        "major_axis_velocity_covariance_kpc_m_s": covariance,
        "systemic_only_control": {
            "systemic_velocity_m_s": float(np.mean(observed_values)),
            "rmse_m_s": float(np.sqrt(np.mean(systemic_only**2))),
        },
        "models": metrics,
        "rg_fractional_rmse_improvement_over_newton": fractional_improvement,
        "rg_beats_newton": bool(fractional_improvement > 0.0),
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    prediction_receipt, manifest = _load_prediction_evidence(config)
    prediction_config = predictions.load_config()
    objects = [
        _score_object(
            config,
            prediction_config,
            prediction_receipt,
            manifest,
            object_id,
        )
        for object_id in _OBJECTS
    ]
    primary = next(row for row in objects if row["object_id"] == "NGC2976")
    stress = next(row for row in objects if row["object_id"] == "NGC4214")
    threshold = float(
        config["interesting_signal_rule"]["minimum_primary_rmse_fractional_improvement_over_newton"]
    )
    primary_interesting = primary["rg_fractional_rmse_improvement_over_newton"] >= threshold
    if primary_interesting and stress["rg_beats_newton"]:
        decision = "INTERESTING_FIXED_RG_MATCHED_PAIR_SIGNAL_REQUIRES_EXPANSION"
    elif primary_interesting:
        decision = "INTERESTING_PRIMARY_RG_SIGNAL_MIXED_LOW_INCLINATION_STRESS_TEST"
    else:
        decision = "NO_INTERESTING_FIXED_RG_PIXEL_SIGNAL_ON_PRIMARY_MATCHED_PAIR"
    aggregate = {
        model_id: {
            "equal_object_mean_rmse_m_s": float(
                np.mean([row["models"][model_id]["rmse_m_s"] for row in objects])
            ),
            "equal_object_mean_mom2_scaled_mse": float(
                np.mean(
                    [
                        row["models"][model_id]["mom2_scaled_mean_squared_residual"]
                        for row in objects
                    ]
                )
            ),
        }
        for model_id in _MODELS
    }
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_FIXED_MATCHED_PAIR_REAL_THINGS_PIXEL_SCORE",
        "decision": decision,
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "prediction_receipt_content_sha256": prediction_receipt["content_sha256"],
        "prediction_manifest_content_sha256": manifest["content_sha256"],
        "primary_sources": config["primary_sources"],
        "score_contract": config["score_contract"],
        "objects": objects,
        "aggregate": aggregate,
        "interesting_signal": {
            "primary_object": "NGC2976",
            "required_fractional_improvement": threshold,
            "observed_fractional_improvement": primary[
                "rg_fractional_rmse_improvement_over_newton"
            ],
            "primary_gate_pass": bool(primary_interesting),
            "stress_test_rg_beats_newton": stress["rg_beats_newton"],
            "theory_confirmation": false_value(),
        },
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def false_value() -> bool:
    """Spells out a fixed false claim without relying on truthy numerics."""
    return False


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
    path = _repo_path(OUTPUT_PATH)
    _require(path.read_bytes() == canonical_bytes(rebuilt) + b"\n", "receipt differs")


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
                        "primary_rg_fractional_improvement": receipt["interesting_signal"][
                            "observed_fractional_improvement"
                        ],
                        "response_array_slots_decoded": receipt["scientific_boundary"][
                            "response_array_slots_decoded"
                        ],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps({"status": "UNRUN_PREREGISTERED", "output_exists": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
