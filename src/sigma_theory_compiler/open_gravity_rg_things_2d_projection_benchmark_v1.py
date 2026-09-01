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
from astropy.wcs import WCS
from scipy.signal import fftconvolve

CONFIG_PATH = Path("configs/open_gravity_rg_things_2d_projection_benchmark_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_rg_things_2d_projection_benchmark_v1.py")
TEST_PATH = Path("tests/test_open_gravity_rg_things_2d_projection_benchmark_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-things-2d-projection-benchmark-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-2d-projection-benchmark-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-2d-projection-benchmark-receipt-1.0"
_CONFIG_RAW_SHA256 = "0708dfb692a0437f5e62d618c7745b7163d72b0c9210fcade44d51e0216423c2"
_CONFIG_CONTENT_SHA256 = "90132b6dd74fbcc81163084d18fe4775ee026c54049742a6004bedcab7fab4c2"
_MODULE_SEMANTIC_SHA256 = "1d54fba5ec1c62e44c8ea56df005c36ac66e0f473bd8594029d0133fa0f9fe5b"
_TEST_RAW_SHA256 = "da8567ca51e31c29d77ba14b8a0371fa59ce06504aa0e67bde850db8a0fd9ceb"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class ProjectionBenchmarkError(RuntimeError):
    """Raised when the target-free 2D operator fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProjectionBenchmarkError(message)


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
        raise ProjectionBenchmarkError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["status"] == "FROZEN_TARGET_FREE_2D_FORWARD_OPERATOR", "status changed")
    operator = config["operator_contract"]
    _require(operator["rotation_sign_cells"] == [-1.0, 1.0], "sign controls changed")
    _require(operator["response_threshold_tuning"] is False, "threshold tuning enabled")
    _require(operator["per_pixel_or_radial_parameter_fitting"] is False, "fitting enabled")
    _require("not chi-square" in operator["primary_loss"], "loss caveat removed")
    gates = config["benchmark_contract"]
    _require(len(gates) == 7, "benchmark ledger changed")
    admission = config["builder_admission"]
    _require(admission["real_public_data_bound"] is True, "real data gate lost")
    _require(admission["primary_data_and_kinematic_papers_bound"] is True, "paper gate lost")
    _require(
        admission["independent_target_free_benchmarks_required"] is True, "benchmark gate lost"
    )
    _require(
        admission["response_pixel_decode_allowed_only_if_all_benchmarks_pass"] is True,
        "predecode gate lost",
    )
    _require(admission["missing_data_disposition"] == "SOURCE_BLOCKED", "source block changed")
    _require(
        admission["benchmark_failure_disposition"] == "BUILDER_BLOCKED_RETAIN_FAILURE",
        "benchmark failure rule changed",
    )
    science = config["scientific_boundary"]
    _require(science["quasi_circular_projection_not_gas_dynamics"] is True, "gas overclaim")
    _require(science["moment2_is_not_measurement_uncertainty"] is True, "uncertainty overclaim")
    _require(science["primary_loss_is_not_chi_square"] is True, "chi-square overclaim")
    for key in (
        "noncircular_streaming_predicted",
        "general_3d_motion_predicted",
    ):
        _require(science[key] is False, f"motion overclaim: {key}")
    for key in (
        "velocity_pixel_values_decoded",
        "dispersion_pixel_values_decoded",
        "scientific_scores_computed",
        "network_calls",
        "model_calls",
        "paid_calls",
    ):
        _require(science[key] == 0, f"target access enabled: {key}")
    claims = config["claim_boundary"]
    _require(claims["target_free_operator_benchmarked"] is True, "operator claim lost")
    for key in (
        "scientific_fit_tested",
        "gas_dynamics_solved",
        "new_gravity_law_supported",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overreach: {key}")
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


def _load_source_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["source_binding"]
    for role in ("config", "module", "test", "receipt"):
        path = _repo_path(binding[f"{role}_path"])
        _require(path.is_file(), f"source {role} missing")
        _require(file_sha256(path) == binding[f"{role}_raw_sha256"], f"source {role} changed")
    receipt = _read_json(_repo_path(binding["receipt_path"]), "source receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "source content changed"
    )
    _require(
        receipt["file_count"] == 4 and receipt["byte_count"] == 16974720, "source inventory changed"
    )
    _require(
        receipt["access_accounting"]["velocity_pixel_values_decoded"] == 0,
        "source velocity leak",
    )
    _require(
        receipt["future_builder_gate"]["independent_solver_benchmarks_passed"] is False,
        "source benchmark state changed",
    )
    return receipt


def disk_coordinates(
    x_east: np.ndarray,
    y_north: np.ndarray,
    *,
    position_angle_deg: float,
    inclination_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_east = np.asarray(x_east, dtype=np.float64)
    y_north = np.asarray(y_north, dtype=np.float64)
    _require(x_east.shape == y_north.shape, "coordinate shape mismatch")
    inclination = math.radians(float(inclination_deg))
    _require(0.0 <= inclination < math.pi / 2.0, "invalid inclination")
    position_angle = math.radians(float(position_angle_deg))
    major = x_east * math.sin(position_angle) + y_north * math.cos(position_angle)
    minor = -x_east * math.cos(position_angle) + y_north * math.sin(position_angle)
    disk_y = minor / math.cos(inclination)
    radius = np.hypot(major, disk_y)
    cosine = np.divide(major, radius, out=np.zeros_like(radius), where=radius > 0.0)
    return major, disk_y, radius, cosine


def project_quasi_circular(
    x_east_m: np.ndarray,
    y_north_m: np.ndarray,
    radial_acceleration_m_s2: np.ndarray,
    *,
    position_angle_deg: float,
    inclination_deg: float,
    systemic_velocity_m_s: float,
    rotation_sign: float,
) -> np.ndarray:
    _require(rotation_sign in (-1.0, 1.0), "invalid rotation sign")
    _major, _disk_y, radius, cosine = disk_coordinates(
        x_east_m,
        y_north_m,
        position_angle_deg=position_angle_deg,
        inclination_deg=inclination_deg,
    )
    acceleration = np.asarray(radial_acceleration_m_s2, dtype=np.float64)
    _require(acceleration.shape == radius.shape, "acceleration shape mismatch")
    _require(bool(np.all(np.isfinite(acceleration) & (acceleration >= 0.0))), "bad acceleration")
    speed = np.sqrt(radius * acceleration)
    return (
        float(systemic_velocity_m_s)
        + float(rotation_sign) * speed * math.sin(math.radians(inclination_deg)) * cosine
    )


def elliptical_gaussian_kernel(
    size: int,
    *,
    beam_major_pixels: float,
    beam_minor_pixels: float,
    beam_position_angle_deg: float,
) -> np.ndarray:
    _require(size >= 3 and size % 2 == 1, "bad kernel size")
    _require(beam_major_pixels >= beam_minor_pixels > 0.0, "bad beam")
    coordinate = np.arange(size, dtype=np.float64) - size // 2
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    angle = math.radians(float(beam_position_angle_deg))
    major = x * math.sin(angle) + y * math.cos(angle)
    minor = -x * math.cos(angle) + y * math.sin(angle)
    sigma_major = float(beam_major_pixels) / math.sqrt(8.0 * math.log(2.0))
    sigma_minor = float(beam_minor_pixels) / math.sqrt(8.0 * math.log(2.0))
    kernel = np.exp(-0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2))
    kernel /= np.sum(kernel)
    _require(abs(float(np.sum(kernel)) - 1.0) < 1.0e-15, "kernel normalization failed")
    return kernel


def intensity_weighted_beam(
    velocity: np.ndarray, intensity: np.ndarray, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.asarray(velocity, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    _require(velocity.shape == intensity.shape and velocity.ndim == 2, "image shape mismatch")
    _require(bool(np.all(np.isfinite(velocity))), "nonfinite velocity")
    _require(bool(np.all(np.isfinite(intensity) & (intensity >= 0.0))), "bad intensity")
    denominator = fftconvolve(intensity, kernel, mode="same")
    numerator = fftconvolve(intensity * velocity, kernel, mode="same")
    convolved = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1.0e-15 * max(float(np.max(denominator)), 1.0),
    )
    return convolved, denominator


def analytic_systemic_offset(
    predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray
) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    _require(predicted.shape == observed.shape == weights.shape, "nuisance shape mismatch")
    _require(bool(np.all(np.isfinite(predicted) & np.isfinite(observed))), "bad nuisance data")
    _require(bool(np.all(np.isfinite(weights) & (weights > 0.0))), "bad nuisance weights")
    return float(np.sum(weights * (observed - predicted)) / np.sum(weights))


def _synthetic_wcs(size: int, pixel_scale_arcsec: float) -> WCS:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = size
    header["NAXIS2"] = size
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (size + 1.0) / 2.0
    header["CRPIX2"] = (size + 1.0) / 2.0
    header["CRVAL1"] = 146.81375
    header["CRVAL2"] = 67.9166666667
    header["CDELT1"] = -pixel_scale_arcsec / 3600.0
    header["CDELT2"] = pixel_scale_arcsec / 3600.0
    return WCS(header)


def run_benchmarks(config: Mapping[str, Any]) -> dict[str, Any]:
    suite = config["synthetic_suite"]
    gates = config["benchmark_contract"]
    size = int(suite["grid_pixels"])
    coordinate = np.arange(size, dtype=np.float64) - size // 2
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    pixel_m = 1.0e18
    _major, _disk_y, radius, cosine = disk_coordinates(
        x * pixel_m,
        y * pixel_m,
        position_angle_deg=float(suite["position_angle_deg"]),
        inclination_deg=float(suite["inclination_deg"]),
    )
    speed = float(suite["rotation_speed_m_s"])
    acceleration = np.divide(
        speed * speed,
        radius,
        out=np.zeros_like(radius),
        where=radius > 0.0,
    )
    predicted = project_quasi_circular(
        x * pixel_m,
        y * pixel_m,
        acceleration,
        position_angle_deg=float(suite["position_angle_deg"]),
        inclination_deg=float(suite["inclination_deg"]),
        systemic_velocity_m_s=0.0,
        rotation_sign=1.0,
    )
    expected = speed * math.sin(math.radians(float(suite["inclination_deg"]))) * cosine
    tilted_error = float(np.max(np.abs(predicted - expected)))
    observed = expected + float(suite["systemic_velocity_m_s"])
    nuisance = analytic_systemic_offset(predicted, observed, np.ones_like(predicted))
    nuisance_error = abs(nuisance - float(suite["systemic_velocity_m_s"]))
    axis_coordinate = coordinate[coordinate != 0.0] * pixel_m
    position_angle = math.radians(float(suite["position_angle_deg"]))
    axis_x_east = axis_coordinate * math.sin(position_angle)
    axis_y_north = axis_coordinate * math.cos(position_angle)
    axis_acceleration = speed * speed / np.abs(axis_coordinate)
    axis_prediction = project_quasi_circular(
        axis_x_east,
        axis_y_north,
        axis_acceleration,
        position_angle_deg=float(suite["position_angle_deg"]),
        inclination_deg=float(suite["inclination_deg"]),
        systemic_velocity_m_s=0.0,
        rotation_sign=1.0,
    )
    recovered_speed = np.divide(
        np.abs(axis_prediction),
        math.sin(math.radians(float(suite["inclination_deg"]))),
    )
    axis_error = float(np.max(np.abs(recovered_speed - speed)) / speed)
    kernel = elliptical_gaussian_kernel(
        size,
        beam_major_pixels=float(suite["beam_major_arcsec"]) / float(suite["pixel_scale_arcsec"]),
        beam_minor_pixels=float(suite["beam_minor_arcsec"]) / float(suite["pixel_scale_arcsec"]),
        beam_position_angle_deg=float(suite["beam_position_angle_deg"]),
    )
    intensity = np.exp(-radius / (15.0 * pixel_m))
    constant = np.ones_like(intensity)
    beam_constant, denominator = intensity_weighted_beam(constant, intensity, kernel)
    valid = denominator > 1.0e-10 * float(np.max(denominator))
    constant_error = float(np.max(np.abs(beam_constant[valid] - constant[valid])))
    delta = np.zeros_like(intensity)
    delta[size // 2, size // 2] = 1.0
    convolved_delta = fftconvolve(delta, kernel, mode="same")
    flux_error = abs(float(np.sum(convolved_delta)) - 1.0)
    wcs = _synthetic_wcs(size, float(suite["pixel_scale_arcsec"]))
    pixels = np.asarray([[13.25, 22.75], [64.0, 64.0], [110.5, 95.125]])
    world = wcs.all_pix2world(pixels, 0)
    roundtrip = wcs.all_world2pix(world, 0)
    wcs_error = float(np.max(np.abs(roundtrip - pixels)))
    face_on = project_quasi_circular(
        x * pixel_m,
        y * pixel_m,
        acceleration,
        position_angle_deg=float(suite["position_angle_deg"]),
        inclination_deg=0.0,
        systemic_velocity_m_s=0.0,
        rotation_sign=1.0,
    )
    face_error = float(np.max(np.abs(face_on)))
    reverse = project_quasi_circular(
        x * pixel_m,
        y * pixel_m,
        acceleration,
        position_angle_deg=float(suite["position_angle_deg"]),
        inclination_deg=float(suite["inclination_deg"]),
        systemic_velocity_m_s=0.0,
        rotation_sign=-1.0,
    )
    sign_error = float(np.max(np.abs(predicted + reverse)))
    metrics = {
        "synthetic_tilted_disk_max_abs_m_s": max(tilted_error, nuisance_error),
        "axisymmetric_major_axis_relative_error": axis_error,
        "beam_constant_field_max_abs_m_s": constant_error,
        "beam_flux_conservation_relative_error": flux_error,
        "wcs_roundtrip_max_pixels": wcs_error,
        "face_on_projected_amplitude_max_m_s": face_error,
        "opposite_rotation_sign_antisymmetry_max_m_s": sign_error,
    }
    checks = {key: metrics[key] <= float(gates[key]) for key in gates}
    return {"metrics": metrics, "checks": checks, "all_pass": all(checks.values())}


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    source = _load_source_binding(config)
    benchmarks = run_benchmarks(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "PASS_TARGET_FREE_2D_PROJECTION_RESPONSE_PIXEL_DECODE_ALLOWED"
            if benchmarks["all_pass"]
            else "BLOCK_2D_PROJECTION_BENCHMARK_FAILURE_RETAINED"
        ),
        "decision": (
            "READY_TO_BUILD_FIXED_MATCHED_PAIR_2D_SOURCE_PREDICTIONS"
            if benchmarks["all_pass"]
            else "BUILDER_BLOCKED_RETAIN_FAILURE"
        ),
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "source_receipt_content_sha256": source["content_sha256"],
        "operator_contract": config["operator_contract"],
        "benchmarks": benchmarks,
        "builder_admission": config["builder_admission"],
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _require(dict(payload) == build_receipt(config), "receipt differs from rebuild")


def _output_path() -> Path:
    path = _repo_path(OUTPUT_PATH)
    _require(path == (_ROOT / OUTPUT_PATH).resolve(), "output path changed")
    return path


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
            _require(path.read_bytes() == payload, "concurrent nonidentical receipt")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    return _atomic_no_clobber(_output_path(), canonical_bytes(build_receipt(config)) + b"\n")


def validate_receipt() -> None:
    config = load_config()
    path = _output_path()
    _require(path.is_file(), "receipt missing")
    validate_receipt_payload(config, _read_json(path, "receipt"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"), nargs="?", default="check")
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "benchmarks_pass": sum(receipt["benchmarks"]["checks"].values()),
                    "benchmark_count": len(receipt["benchmarks"]["checks"]),
                    "velocity_pixel_values_decoded": 0,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
