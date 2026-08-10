"""Synthetic-only raw-to-calibrated Solar direct-signal transformations.

The implementation consumes already parsed, unit-tagged TDF/RSR values.  It
never opens a primary product and keeps calibration covariance explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


class CalibrationContractError(ValueError):
    """Raised when a transformation or covariance contract fails closed."""


_NUISANCE_LABELS = (
    "clock_offset_s",
    "instrument_phase_offset_rad",
    "ionosphere_delay_s",
    "oscillator_fractional_error",
    "propagation_frequency_shift_hz",
    "solar_plasma_delay_s",
    "station_geometry_delay_s",
    "station_geometry_frequency_shift_hz",
    "station_path_delay_s",
    "troposphere_delay_s",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(encoded.encode("ascii"))


def reconstruct_tdf_three_part(parts: Sequence[int]) -> float:
    """Apply the selected TDF label's H*1e6 + I*1e1 + L*1e-6 rule."""

    if len(parts) != 3 or any(not isinstance(value, int) for value in parts):
        raise CalibrationContractError("TDF three-part value requires three integers")
    if any(not 0 <= value < (1 << 24) for value in parts):
        raise CalibrationContractError("TDF three-part component exceeds unsigned 24-bit layout")
    high, intermediate, low = parts
    return high * 1.0e6 + intermediate * 1.0e1 + low * 1.0e-6


def _utc_seconds(tag: Mapping[str, Any]) -> float:
    required = {"year", "day_of_year", "hour", "minute", "second", "time_system"}
    if set(tag) != required or tag["time_system"] != "UTC":
        raise CalibrationContractError("time tag must be the exact UTC component contract")
    try:
        year = int(tag["year"])
        day = int(tag["day_of_year"])
        hour = int(tag["hour"])
        minute = int(tag["minute"])
        second = float(tag["second"])
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError("invalid UTC component type") from exc
    if not 1900 <= year <= 3000 or not 1 <= day <= 366:
        raise CalibrationContractError("UTC year/day is outside the contract")
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second < 61:
        raise CalibrationContractError("UTC clock is outside the contract")
    whole = math.floor(second)
    fraction = second - whole
    try:
        instant = datetime(year, 1, 1, tzinfo=UTC) + timedelta(
            days=day - 1, hours=hour, minutes=minute, seconds=whole
        )
    except ValueError as exc:
        raise CalibrationContractError("invalid UTC calendar tag") from exc
    if instant.year != year:
        raise CalibrationContractError("UTC day-of-year does not exist in the declared year")
    return instant.timestamp() + fraction


def tdf_direct_signals(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct TDF count and range fields with their declared units."""

    required = {
        "doppler_count_parts",
        "range_parts",
        "range_type",
        "sample_interval_centisecond",
        "sky_frequency_hz",
        "uplink_band",
        "utc_time_tag",
        "units",
    }
    if set(packet) != required:
        raise CalibrationContractError("TDF packet shape differs from frozen contract")
    if packet["units"] != {
        "doppler_count": "count",
        "range_type_1": "nanosecond",
        "sample_interval": "centisecond",
        "sky_frequency": "hertz",
    }:
        raise CalibrationContractError("TDF units differ from selected-label contract")
    interval_s = float(packet["sample_interval_centisecond"]) * 0.01
    sky_frequency_hz = float(packet["sky_frequency_hz"])
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise CalibrationContractError("TDF sample interval must be positive and finite")
    if not math.isfinite(sky_frequency_hz) or sky_frequency_hz <= 0.0:
        raise CalibrationContractError("TDF sky frequency must be positive and finite")
    doppler_count = reconstruct_tdf_three_part(packet["doppler_count_parts"])
    range_value = reconstruct_tdf_three_part(packet["range_parts"])
    range_type = int(packet["range_type"])
    uplink_band = str(packet["uplink_band"])
    if range_type == 1:
        range_unit_seconds = 1.0e-9
        range_unit = "nanosecond"
    else:
        factors = {"S": 0.5, "X_HEF": 11.0 / 75.0, "X_BLOCK_V": 221.0 / 1498.0}
        if uplink_band not in factors:
            raise CalibrationContractError("non-nanosecond range requires a declared uplink band")
        range_unit_seconds = factors[uplink_band] * 1.0e-18 * sky_frequency_hz
        range_unit = "range_unit_label_formula"
    round_trip_light_time_s = range_value * range_unit_seconds
    if not math.isfinite(round_trip_light_time_s) or round_trip_light_time_s <= 0.0:
        raise CalibrationContractError("reconstructed TDF light time is not positive and finite")
    return {
        "doppler_count": doppler_count,
        "doppler_count_rate_hz": doppler_count / interval_s,
        "range_unit": range_unit,
        "range_unit_seconds": range_unit_seconds,
        "raw_round_trip_light_time_s": round_trip_light_time_s,
        "sample_interval_s": interval_s,
        "utc_epoch_seconds": _utc_seconds(packet["utc_time_tag"]),
    }


def _validate_covariance(
    labels: Sequence[str], covariance: Sequence[Sequence[float]], *, name: str
) -> np.ndarray:
    if len(labels) == 0 or len(set(labels)) != len(labels):
        raise CalibrationContractError(f"{name} labels must be nonempty and unique")
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (len(labels), len(labels)):
        raise CalibrationContractError(f"{name} covariance shape mismatch")
    if not np.all(np.isfinite(matrix)):
        raise CalibrationContractError(f"{name} covariance contains nonfinite values")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-18):
        raise CalibrationContractError(f"{name} covariance is not symmetric")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12 * scale:
        raise CalibrationContractError(f"{name} covariance is not positive semidefinite")
    return matrix


def rsr_iq_direct_signals(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Convert synthetic/known-answer I/Q samples to phase and residual frequency."""

    required = {
        "i_samples",
        "iq_covariance",
        "q_samples",
        "sample_rate_hz",
        "utc_time_tag",
        "units",
    }
    if set(packet) != required:
        raise CalibrationContractError("RSR packet shape differs from frozen contract")
    if packet["units"] != {
        "i_q": "dimensionless_odd_integer_ADC_code_after_2k_plus_1_bias",
        "sample_rate": "sample_per_second",
    }:
        raise CalibrationContractError("RSR units differ from parser contract")
    i_values = np.asarray(packet["i_samples"], dtype=np.float64)
    q_values = np.asarray(packet["q_samples"], dtype=np.float64)
    if i_values.ndim != 1 or i_values.shape != q_values.shape or len(i_values) < 3:
        raise CalibrationContractError("RSR I/Q arrays must have equal length of at least three")
    if not np.all(np.isfinite(i_values)) or not np.all(np.isfinite(q_values)):
        raise CalibrationContractError("RSR I/Q samples contain nonfinite values")
    if not np.all(i_values == np.rint(i_values)) or not np.all(q_values == np.rint(q_values)):
        raise CalibrationContractError("RSR calibrated parser codes must be integers")
    if np.any(np.mod(i_values, 2.0) == 0.0) or np.any(np.mod(q_values, 2.0) == 0.0):
        raise CalibrationContractError("RSR parser codes must retain the odd 2*k+1 bias")
    amplitude_squared = i_values * i_values + q_values * q_values
    if np.any(amplitude_squared <= 1.0e-24):
        raise CalibrationContractError("RSR phase is undefined at zero amplitude")
    sample_rate_hz = float(packet["sample_rate_hz"])
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise CalibrationContractError("RSR sample rate must be positive and finite")
    principal_phase = np.arctan2(q_values, i_values)
    principal_steps = np.angle(np.exp(1j * np.diff(principal_phase)))
    if np.any(np.abs(principal_steps) >= 0.95 * math.pi):
        raise CalibrationContractError("RSR phase unwrapping is branch-ambiguous")
    phase = np.unwrap(principal_phase)
    times = np.arange(len(phase), dtype=np.float64) / sample_rate_hz
    centered_times = times - float(np.mean(times))
    denominator = float(centered_times @ centered_times)
    slope_weights = centered_times / denominator
    residual_frequency_hz = float(slope_weights @ phase) / (2.0 * math.pi)

    labels = tuple(
        [f"i_{index}" for index in range(len(i_values))]
        + [f"q_{index}" for index in range(len(q_values))]
    )
    iq_covariance = _validate_covariance(labels, packet["iq_covariance"], name="RSR I/Q")
    dphase_di = -q_values / amplitude_squared
    dphase_dq = i_values / amplitude_squared
    phase_jacobian = np.zeros(2 * len(i_values), dtype=np.float64)
    phase_jacobian[0] = dphase_di[0]
    phase_jacobian[len(i_values)] = dphase_dq[0]
    frequency_jacobian = np.concatenate(
        (slope_weights * dphase_di, slope_weights * dphase_dq)
    ) / (2.0 * math.pi)
    jacobian = np.vstack((phase_jacobian, frequency_jacobian))
    output_covariance = jacobian @ iq_covariance @ jacobian.T
    return {
        "carrier_phase_rad": float(phase[0]),
        "iq_to_phase_frequency_jacobian": jacobian.tolist(),
        "measurement_covariance_labels": ["carrier_phase_rad", "residual_frequency_hz"],
        "measurement_covariance_matrix": output_covariance.tolist(),
        "observation_duration_s": float(times[-1] - times[0]),
        "residual_frequency_hz": residual_frequency_hz,
        "sample_count": len(i_values),
        "sample_rate_hz": sample_rate_hz,
        "utc_epoch_seconds": _utc_seconds(packet["utc_time_tag"]),
    }


def calibrate_direct_signals(
    tdf: Mapping[str, Any],
    rsr: Mapping[str, Any],
    calibration_values: Mapping[str, float],
    nuisance_covariance: Sequence[Sequence[float]],
    *,
    raw_tdf_variance_s2: float,
    reference_frequency_hz: float,
    maximum_time_tag_separation_s: float,
) -> dict[str, Any]:
    """Apply corrections and propagate raw plus correlated calibration covariance."""

    required_tdf = {"raw_round_trip_light_time_s", "utc_epoch_seconds"}
    required_rsr = {
        "carrier_phase_rad",
        "measurement_covariance_matrix",
        "observation_duration_s",
        "residual_frequency_hz",
        "utc_epoch_seconds",
    }
    if not required_tdf <= set(tdf) or not required_rsr <= set(rsr):
        raise CalibrationContractError("parsed direct-signal packet is incomplete")
    if tuple(sorted(calibration_values)) != _NUISANCE_LABELS:
        raise CalibrationContractError("calibration nuisance labels differ from frozen allowlist")
    values = np.asarray([calibration_values[label] for label in _NUISANCE_LABELS], dtype=float)
    if not np.all(np.isfinite(values)):
        raise CalibrationContractError("calibration values contain nonfinite entries")
    covariance = _validate_covariance(
        _NUISANCE_LABELS, nuisance_covariance, name="calibration nuisance"
    )
    raw_tdf_variance_s2 = float(raw_tdf_variance_s2)
    reference_frequency_hz = float(reference_frequency_hz)
    if not math.isfinite(raw_tdf_variance_s2) or raw_tdf_variance_s2 < 0.0:
        raise CalibrationContractError("raw TDF variance must be finite and nonnegative")
    if not math.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0.0:
        raise CalibrationContractError("reference frequency must be positive and finite")
    if not math.isfinite(maximum_time_tag_separation_s) or maximum_time_tag_separation_s < 0.0:
        raise CalibrationContractError("association window must be finite and nonnegative")
    time_separation = abs(float(tdf["utc_epoch_seconds"]) - float(rsr["utc_epoch_seconds"]))
    if time_separation > maximum_time_tag_separation_s:
        raise CalibrationContractError("TDF and RSR time tags exceed the frozen association window")

    nuisance = {label: float(calibration_values[label]) for label in _NUISANCE_LABELS}
    delay_labels = (
        "clock_offset_s",
        "ionosphere_delay_s",
        "solar_plasma_delay_s",
        "station_geometry_delay_s",
        "station_path_delay_s",
        "troposphere_delay_s",
    )
    total_delay_s = sum(nuisance[label] for label in delay_labels)
    tau = float(tdf["raw_round_trip_light_time_s"]) - total_delay_s
    frequency = (
        reference_frequency_hz
        + float(rsr["residual_frequency_hz"])
        - reference_frequency_hz * nuisance["oscillator_fractional_error"]
        - nuisance["propagation_frequency_shift_hz"]
        - nuisance["station_geometry_frequency_shift_hz"]
    )
    phase = (
        float(rsr["carrier_phase_rad"])
        - 2.0 * math.pi * reference_frequency_hz * total_delay_s
        - nuisance["instrument_phase_offset_rad"]
        - 2.0
        * math.pi
        * reference_frequency_hz
        * float(rsr["observation_duration_s"])
        * nuisance["oscillator_fractional_error"]
    )
    if tau <= 0.0 or not all(math.isfinite(value) for value in (tau, frequency, phase)):
        raise CalibrationContractError("calibrated signal is nonphysical or nonfinite")

    jacobian = np.zeros((3, len(_NUISANCE_LABELS)), dtype=np.float64)
    for label in delay_labels:
        index = _NUISANCE_LABELS.index(label)
        jacobian[0, index] = -1.0
        jacobian[2, index] = -2.0 * math.pi * reference_frequency_hz
    jacobian[1, _NUISANCE_LABELS.index("oscillator_fractional_error")] = -reference_frequency_hz
    jacobian[2, _NUISANCE_LABELS.index("oscillator_fractional_error")] = (
        -2.0 * math.pi * reference_frequency_hz * float(rsr["observation_duration_s"])
    )
    jacobian[1, _NUISANCE_LABELS.index("propagation_frequency_shift_hz")] = -1.0
    jacobian[1, _NUISANCE_LABELS.index("station_geometry_frequency_shift_hz")] = -1.0
    jacobian[2, _NUISANCE_LABELS.index("instrument_phase_offset_rad")] = -1.0

    raw_covariance = np.zeros((3, 3), dtype=np.float64)
    raw_covariance[0, 0] = raw_tdf_variance_s2
    rsr_covariance = _validate_covariance(
        ("carrier_phase_rad", "residual_frequency_hz"),
        rsr["measurement_covariance_matrix"],
        name="RSR phase/frequency",
    )
    raw_covariance[2, 2] = rsr_covariance[0, 0]
    raw_covariance[2, 1] = raw_covariance[1, 2] = rsr_covariance[0, 1]
    raw_covariance[1, 1] = rsr_covariance[1, 1]
    output_covariance = raw_covariance + jacobian @ covariance @ jacobian.T
    _validate_covariance(
        ("round_trip_light_time_s", "coherent_carrier_frequency_hz", "carrier_phase_rad"),
        output_covariance,
        name="calibrated output",
    )
    return {
        "calibrated_values": {
            "carrier_phase_rad": phase,
            "coherent_carrier_frequency_hz": frequency,
            "round_trip_light_time_s": tau,
        },
        "covariance_matrix": output_covariance.tolist(),
        "covariance_order": [
            "round_trip_light_time_s",
            "coherent_carrier_frequency_hz",
            "carrier_phase_rad",
        ],
        "nuisance_jacobian": jacobian.tolist(),
        "nuisance_order": list(_NUISANCE_LABELS),
        "provenance": {
            "primary_target_records_opened": False,
            "raw_measurement_covariance_retained": True,
            "shared_calibration_correlations_retained": True,
            "time_tag_separation_s": time_separation,
        },
        "units": {
            "carrier_phase_rad": "radian",
            "coherent_carrier_frequency_hz": "hertz",
            "round_trip_light_time_s": "second",
        },
    }


def build_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sigma-solar-direct-signal-calibration-config-1.0":
        raise CalibrationContractError("unexpected calibration config schema")
    for binding in config["local_bindings"].values():
        path = repo_root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["file_sha256"]:
            raise CalibrationContractError(f"local binding mismatch: {binding['path']}")
    expected_seals = {
        "candidate_use_authorized": False,
        "dark_matter_or_halo_inputs": False,
        "observational_data_opened": False,
        "paid_llm_calls": False,
        "redshift_distance_inputs": False,
        "target_values_accessed": False,
    }
    if config["data_eligibility"] != expected_seals:
        raise CalibrationContractError("calibration data seals differ from frozen contract")
    source_sha256 = sha256_file(
        repo_root / "src/sigma_theory_compiler/solar_direct_signal_calibration_readiness.py"
    )
    implementation_sha256 = canonical_sha256(
        {
            "contract": config["transformation_contract"],
            "known_answer_vectors": config["known_answer_vectors"],
            "parser_bindings": config["parser_bindings"],
            "source_sha256": source_sha256,
        }
    )
    absent = [
        "registered_real_source_interval_instantiation_certificate_sha256",
        "selected_primary_file_root_sha256",
        "selected_PDS_label_and_calibration_file_root_sha256",
        "tracking_session_split_commitment_sha256",
        "training_only_initial_state_checkpoint_sha256",
        "reviewed_candidate_solar_evaluator_descriptor_sha256",
    ]
    artifact: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "content_scope": "synthetic_and_documentation_known_answer_only",
        "data_eligibility": config["data_eligibility"],
        "descriptor_registration_status": "blocked_unregistered",
        "filled_registration_field_count": 3,
        "filled_registration_fields": {
            **config["parser_bindings"]["filled_parser_registration_fields"],
            "raw_to_calibrated_transform_and_covariance_implementation_sha256": implementation_sha256,
        },
        "implementation_source_sha256": source_sha256,
        "interpretation": (
            "The direct-signal transformation and covariance implementation is verified on "
            "synthetic and documentation known-answer inputs. No selected primary measurement "
            "or calibration product was opened; the descriptor remains ineligible."
        ),
        "observational_authorization": False,
        "primary_record_access_count": 0,
        "remaining_registration_field_count": len(absent),
        "remaining_registration_fields": absent,
        "schema_version": "sigma-solar-direct-signal-calibration-readiness-1.0",
        "source_bindings": config["local_bindings"],
        "status": "calibration_implementation_ready_primary_records_sealed",
        "transformation_contract_sha256": canonical_sha256(config["transformation_contract"]),
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact
