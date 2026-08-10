from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.solar_direct_signal_calibration_readiness import (
    CalibrationContractError,
    build_readiness_artifact,
    calibrate_direct_signals,
    reconstruct_tdf_three_part,
    rsr_iq_direct_signals,
    tdf_direct_signals,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/solar_direct_signal_calibration_readiness.json"
ARTIFACT_PATH = REPO_ROOT / "runs/engine/solar-direct-signal-calibration-readiness.json"


def _time_tag(second: float = 25.0) -> dict[str, object]:
    return {
        "day_of_year": 180,
        "hour": 13,
        "minute": 11,
        "second": second,
        "time_system": "UTC",
        "year": 2002,
    }


def _tdf_packet(*, range_type: int = 1, uplink_band: str = "S") -> dict[str, object]:
    return {
        "doppler_count_parts": [0, 12, 0],
        "range_parts": [120, 0, 0] if range_type == 1 else [1, 0, 0],
        "range_type": range_type,
        "sample_interval_centisecond": 6000,
        "sky_frequency_hz": 2.0e9,
        "uplink_band": uplink_band,
        "utc_time_tag": _time_tag(),
        "units": {
            "doppler_count": "count",
            "range_type_1": "nanosecond",
            "sample_interval": "centisecond",
            "sky_frequency": "hertz",
        },
    }


def _odd_code(value: float) -> int:
    return 2 * round((value - 1.0) / 2.0) + 1


def _rsr_packet(*, frequency_hz: float = 5.0) -> dict[str, object]:
    sample_count = 20
    sample_rate_hz = 100.0
    phases = 0.25 + 2.0 * math.pi * frequency_hz * np.arange(sample_count) / sample_rate_hz
    i_values = [_odd_code(1000.0 * math.cos(value)) for value in phases]
    q_values = [_odd_code(1000.0 * math.sin(value)) for value in phases]
    covariance = np.eye(2 * sample_count, dtype=float) * 0.25
    return {
        "i_samples": i_values,
        "iq_covariance": covariance.tolist(),
        "q_samples": q_values,
        "sample_rate_hz": sample_rate_hz,
        "utc_time_tag": _time_tag(25.1),
        "units": {
            "i_q": "dimensionless_odd_integer_ADC_code_after_2k_plus_1_bias",
            "sample_rate": "sample_per_second",
        },
    }


def _nuisance_values() -> dict[str, float]:
    return {
        "clock_offset_s": 1.0e-9,
        "instrument_phase_offset_rad": 2.0e-3,
        "ionosphere_delay_s": 3.0e-9,
        "oscillator_fractional_error": 2.0e-12,
        "propagation_frequency_shift_hz": 0.02,
        "solar_plasma_delay_s": 4.0e-9,
        "station_geometry_delay_s": 5.0e-9,
        "station_geometry_frequency_shift_hz": -0.01,
        "station_path_delay_s": 2.0e-9,
        "troposphere_delay_s": 6.0e-9,
    }


def _nuisance_covariance() -> np.ndarray:
    standard_deviations = np.array(
        [1e-10, 1e-4, 2e-10, 1e-13, 2e-3, 3e-10, 2e-10, 2e-3, 1e-10, 3e-10]
    )
    covariance = np.diag(standard_deviations**2)
    # One shared calibration mode creates explicit cross-channel correlations.
    shared = np.array([5e-11, 0.0, 8e-11, 5e-14, 5e-4, 9e-11, 7e-11, 3e-4, 4e-11, 8e-11])
    return covariance + np.outer(shared, shared)


def test_tdf_reconstruction_preserves_units_and_label_formulas() -> None:
    assert reconstruct_tdf_three_part([2, 3, 4]) == pytest.approx(2_000_030.000004)
    result = tdf_direct_signals(_tdf_packet())
    assert result["raw_round_trip_light_time_s"] == pytest.approx(0.12)
    assert result["doppler_count_rate_hz"] == pytest.approx(2.0)
    assert result["range_unit"] == "nanosecond"
    assert result["sample_interval_s"] == pytest.approx(60.0)

    range_unit = tdf_direct_signals(_tdf_packet(range_type=2, uplink_band="S"))
    assert range_unit["range_unit_seconds"] == pytest.approx(1.0e-9)
    assert range_unit["raw_round_trip_light_time_s"] == pytest.approx(0.001)


def test_rsr_iq_to_phase_frequency_and_covariance_known_answer() -> None:
    result = rsr_iq_direct_signals(_rsr_packet())
    assert result["sample_count"] == 20
    assert result["carrier_phase_rad"] == pytest.approx(0.25, abs=0.002)
    assert result["residual_frequency_hz"] == pytest.approx(5.0, abs=0.01)
    covariance = np.asarray(result["measurement_covariance_matrix"])
    assert covariance.shape == (2, 2)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-18
    assert len(result["iq_to_phase_frequency_jacobian"]) == 2


def test_full_calibration_and_shared_covariance_are_propagated() -> None:
    tdf = tdf_direct_signals(_tdf_packet())
    rsr = rsr_iq_direct_signals(_rsr_packet())
    result = calibrate_direct_signals(
        tdf,
        rsr,
        _nuisance_values(),
        _nuisance_covariance().tolist(),
        raw_tdf_variance_s2=4.0e-20,
        reference_frequency_hz=8.4e9,
        maximum_time_tag_separation_s=1.0,
    )
    total_delay = 21.0e-9
    assert result["calibrated_values"]["round_trip_light_time_s"] == pytest.approx(
        0.12 - total_delay
    )
    expected_frequency = 8.4e9 + rsr["residual_frequency_hz"] - 8.4e9 * 2e-12 - 0.02 + 0.01
    assert result["calibrated_values"]["coherent_carrier_frequency_hz"] == pytest.approx(
        expected_frequency
    )
    covariance = np.asarray(result["covariance_matrix"])
    assert covariance.shape == (3, 3)
    assert covariance[0, 2] != 0.0
    assert covariance[1, 2] != 0.0
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-8
    assert result["provenance"]["shared_calibration_correlations_retained"] is True
    assert result["provenance"]["primary_target_records_opened"] is False


def test_unit_time_iq_and_range_negative_controls_fail_closed() -> None:
    bad_units = _tdf_packet()
    bad_units["units"] = {**bad_units["units"], "sample_interval": "second"}
    with pytest.raises(CalibrationContractError, match="units"):
        tdf_direct_signals(bad_units)
    bad_band = _tdf_packet(range_type=2, uplink_band="UNKNOWN")
    with pytest.raises(CalibrationContractError, match="uplink band"):
        tdf_direct_signals(bad_band)
    with pytest.raises(CalibrationContractError, match="24-bit"):
        reconstruct_tdf_three_part([1 << 24, 0, 0])

    zero = _rsr_packet()
    zero["i_samples"] = [0] * 20
    zero["q_samples"] = [0] * 20
    with pytest.raises(CalibrationContractError, match=r"odd 2\*k\+1"):
        rsr_iq_direct_signals(zero)
    ambiguous = _rsr_packet(frequency_hz=49.0)
    with pytest.raises(CalibrationContractError, match="branch-ambiguous"):
        rsr_iq_direct_signals(ambiguous)
    bad_time = _rsr_packet()
    bad_time["utc_time_tag"] = {**bad_time["utc_time_tag"], "time_system": "TDB"}
    with pytest.raises(CalibrationContractError, match="UTC"):
        rsr_iq_direct_signals(bad_time)


def test_covariance_nuisance_and_association_negative_controls_fail_closed() -> None:
    tdf = tdf_direct_signals(_tdf_packet())
    rsr = rsr_iq_direct_signals(_rsr_packet())
    values = _nuisance_values()
    covariance = _nuisance_covariance()
    non_psd = covariance.copy()
    non_psd[0, 0] = -1.0
    with pytest.raises(CalibrationContractError, match="positive semidefinite"):
        calibrate_direct_signals(
            tdf,
            rsr,
            values,
            non_psd.tolist(),
            raw_tdf_variance_s2=0.0,
            reference_frequency_hz=8.4e9,
            maximum_time_tag_separation_s=1.0,
        )
    nonsymmetric = covariance.copy()
    nonsymmetric[0, 1] = 1.0
    with pytest.raises(CalibrationContractError, match="not symmetric"):
        calibrate_direct_signals(
            tdf,
            rsr,
            values,
            nonsymmetric.tolist(),
            raw_tdf_variance_s2=0.0,
            reference_frequency_hz=8.4e9,
            maximum_time_tag_separation_s=1.0,
        )
    with pytest.raises(CalibrationContractError, match="shape mismatch"):
        calibrate_direct_signals(
            tdf,
            rsr,
            values,
            covariance[:-1, :-1].tolist(),
            raw_tdf_variance_s2=0.0,
            reference_frequency_hz=8.4e9,
            maximum_time_tag_separation_s=1.0,
        )
    unknown = {**values, "post_hoc_rescue": 1.0}
    with pytest.raises(CalibrationContractError, match="allowlist"):
        calibrate_direct_signals(
            tdf,
            rsr,
            unknown,
            covariance.tolist(),
            raw_tdf_variance_s2=0.0,
            reference_frequency_hz=8.4e9,
            maximum_time_tag_separation_s=1.0,
        )
    with pytest.raises(CalibrationContractError, match="association window"):
        calibrate_direct_signals(
            tdf,
            rsr,
            values,
            covariance.tolist(),
            raw_tdf_variance_s2=0.0,
            reference_frequency_hz=8.4e9,
            maximum_time_tag_separation_s=0.01,
        )


def test_readiness_artifact_is_deterministic_and_keeps_six_hashes_absent() -> None:
    first = build_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    assert first == build_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    assert first["filled_registration_field_count"] == 3
    assert first["remaining_registration_field_count"] == 6
    assert "raw_to_calibrated_transform_and_covariance_implementation_sha256" in first[
        "filled_registration_fields"
    ]
    assert "selected_primary_file_root_sha256" in first["remaining_registration_fields"]
    assert "tracking_session_split_commitment_sha256" in first[
        "remaining_registration_fields"
    ]
    assert first["primary_record_access_count"] == 0
    assert first["observational_authorization"] is False


def test_checked_in_artifact_matches_and_seal_tamper_fails(tmp_path: Path) -> None:
    assert json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")) == build_readiness_artifact(
        REPO_ROOT, CONFIG_PATH
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["data_eligibility"]["observational_data_opened"] = True
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CalibrationContractError, match="seals"):
        build_readiness_artifact(REPO_ROOT, tampered)
