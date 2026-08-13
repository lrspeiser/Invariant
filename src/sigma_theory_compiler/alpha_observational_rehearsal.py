"""Synthetic-only observational controls for the comprehensive alpha."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import cluster_direct_observable_evaluator_readiness as cluster
from . import lensing_direct_observable_evaluator_readiness as lensing
from .galaxy_direct_observable_evaluator import galaxy_direct_observable_evaluator
from .promotion_orchestrator import ELIGIBILITY
from .solar_direct_signal_calibration_readiness import (
    build_readiness_artifact,
    calibrate_direct_signals,
    rsr_iq_direct_signals,
    tdf_direct_signals,
)

SCHEMA = "sigma-alpha-observational-rehearsal-1.0"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: dict[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _time_tag(second: float) -> dict[str, object]:
    return {
        "day_of_year": 180,
        "hour": 13,
        "minute": 11,
        "second": second,
        "time_system": "UTC",
        "year": 2002,
    }


def _tdf_packet() -> dict[str, object]:
    return {
        "doppler_count_parts": [0, 12, 0],
        "range_parts": [120, 0, 0],
        "range_type": 1,
        "sample_interval_centisecond": 6000,
        "sky_frequency_hz": 2.0e9,
        "uplink_band": "S",
        "utc_time_tag": _time_tag(25.0),
        "units": {
            "doppler_count": "count",
            "range_type_1": "nanosecond",
            "sample_interval": "centisecond",
            "sky_frequency": "hertz",
        },
    }


def _odd_code(value: float) -> int:
    return 2 * round((value - 1.0) / 2.0) + 1


def _rsr_packet() -> dict[str, object]:
    sample_count = 20
    sample_rate_hz = 100.0
    phases = 0.25 + 2.0 * math.pi * 5.0 * np.arange(sample_count) / sample_rate_hz
    i_values = [_odd_code(1000.0 * math.cos(value)) for value in phases]
    q_values = [_odd_code(1000.0 * math.sin(value)) for value in phases]
    return {
        "i_samples": i_values,
        "iq_covariance": (np.eye(2 * sample_count, dtype=float) * 0.25).tolist(),
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


def _nuisance_covariance() -> list[list[float]]:
    deviations = np.array([1e-10, 1e-4, 2e-10, 1e-13, 2e-3, 3e-10, 2e-10, 2e-3, 1e-10, 3e-10])
    shared = np.array([5e-11, 0.0, 8e-11, 5e-14, 5e-4, 9e-11, 7e-11, 3e-4, 4e-11, 8e-11])
    return (np.diag(deviations**2) + np.outer(shared, shared)).tolist()


def _solar_control(root: Path) -> dict[str, Any]:
    config_path = root / "configs/solar_direct_signal_calibration_readiness.json"
    artifact_path = root / "runs/engine/solar-direct-signal-calibration-readiness.json"
    stored = json.loads(artifact_path.read_text(encoding="utf-8"))
    rebuilt = build_readiness_artifact(root, config_path)
    if rebuilt != stored:
        raise ValueError("Solar calibration readiness artifact replay mismatch")
    tdf = tdf_direct_signals(_tdf_packet())
    rsr = rsr_iq_direct_signals(_rsr_packet())
    calibrated = calibrate_direct_signals(
        tdf,
        rsr,
        _nuisance_values(),
        _nuisance_covariance(),
        raw_tdf_variance_s2=4.0e-20,
        reference_frequency_hz=8.4e9,
        maximum_time_tag_separation_s=1.0,
    )
    if calibrated["provenance"]["primary_target_records_opened"] is not False:
        raise ValueError("Solar synthetic control opened a primary target")
    return {
        "artifact_file_sha256": _file_sha(artifact_path),
        "config_file_sha256": _file_sha(config_path),
        "controls": {"calibration": 1, "rsr_parser": 1, "tdf_parser": 1},
        "primary_target_records_opened": False,
        "readiness_replay_exact": True,
        "scientific_pass": False,
        "synthetic_only": True,
    }


def _galaxy_control() -> dict[str, Any]:
    result = galaxy_direct_observable_evaluator(
        {
            "candidate_id": "alpha-synthetic-galaxy-control",
            "data_eligibility": dict(ELIGIBILITY),
        },
        {
            "data_eligibility": dict(ELIGIBILITY),
            "input_lineage_sha256": "3" * 64,
        },
    )
    if (
        result["decision"] != "blocked"
        or result["blocker"] != "missing_candidate_direct_observable_prediction_bundle"
        or result["registered_prediction_bundle_count"] != 0
        or result["observational_data_opened"] is not False
    ):
        raise ValueError("galaxy sealed control changed")
    return {
        "blocker": result["blocker"],
        "decision": result["decision"],
        "registered_prediction_bundle_count": 0,
        "scientific_pass": False,
        "source_registrations_loaded": result["source_registrations_loaded"],
        "synthetic_only": True,
    }


def _readiness_control(root: Path, module: Any, *, domain: str) -> dict[str, Any]:
    config_path = root / module.CONFIG_REL
    artifact_path = root / module.ARTIFACT_REL
    config = module.load_config(root, config_path)
    stored = json.loads(artifact_path.read_text(encoding="utf-8"))
    rebuilt = module.build_readiness(config, root)
    if rebuilt != stored:
        raise ValueError(f"{domain} readiness artifact replay mismatch")
    module.validate_readiness(stored, config, root)
    if domain == "lensing":
        packets = (
            module._synthetic_packet("raw_direct_observable", "raw_detector_counts"),
            module._synthetic_packet(
                "calibrated_direct_observable", "directly_measured_time_delay"
            ),
        )
    else:
        packets = (
            module._synthetic_packet("raw_direct_observable", "raw_xray_detector_counts"),
            module._synthetic_packet(
                "calibrated_direct_observable",
                "calibrated_xray_surface_brightness_or_spectrum",
            ),
        )
    for packet in packets:
        module.validate_direct_observable_packet(packet)
    counts = stored["counts"]
    if (
        counts["authorized_real_source_packets"] != 0
        or counts["real_source_packets_opened"] != 0
        or counts["scientific_passes"] != 0
        or counts["scientific_rejects"] != 0
        or counts["scientific_blocks"] != 1
        or (domain == "cluster" and counts["rank_writes"] != 0)
    ):
        raise ValueError(f"{domain} readiness outcome boundary changed")
    return {
        "artifact_file_sha256": _file_sha(artifact_path),
        "authorized_real_source_packets": 0,
        "config_file_sha256": _file_sha(config_path),
        "decision": stored["decision"],
        "rank_writes": 0,
        "readiness_replay_exact": True,
        "real_source_packets_opened": 0,
        "schema_control_passes": len(packets),
        "scientific_blocks": 1,
        "scientific_pass": False,
        "scientific_reject": False,
        "synthetic_only": True,
    }


def build_observational_rehearsal(repo_root: str | Path) -> dict[str, Any]:
    """Run in-memory schema/calibration controls without observational authorization."""

    root = Path(repo_root).resolve()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "solar": _solar_control(root),
        "galaxy": _galaxy_control(),
        "lensing": _readiness_control(root, lensing, domain="lensing"),
        "cluster": _readiness_control(root, cluster, domain="cluster"),
        "claims": {
            "gpu_access": False,
            "live_sqlite_access": False,
            "network_access": False,
            "observational_authorization": False,
            "rank_writes": 0,
            "real_data_opened": False,
            "registry_writes": 0,
            "runtime_access": False,
            "scientific_pass": False,
            "scientific_reject": False,
            "secret_access": False,
            "synthetic_only": True,
        },
    }
    receipt["content_sha256"] = _content_sha(receipt)
    validate_observational_receipt(receipt)
    return receipt


def validate_observational_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != SCHEMA or receipt.get("content_sha256") != _content_sha(
        receipt
    ):
        raise ValueError("observational rehearsal receipt hash or schema mismatch")
    claims = receipt.get("claims", {})
    if claims != {
        "gpu_access": False,
        "live_sqlite_access": False,
        "network_access": False,
        "observational_authorization": False,
        "rank_writes": 0,
        "real_data_opened": False,
        "registry_writes": 0,
        "runtime_access": False,
        "scientific_pass": False,
        "scientific_reject": False,
        "secret_access": False,
        "synthetic_only": True,
    }:
        raise ValueError("observational rehearsal claims changed")
    if receipt.get("solar", {}).get("controls") != {
        "calibration": 1,
        "rsr_parser": 1,
        "tdf_parser": 1,
    }:
        raise ValueError("Solar observational controls changed")
    if receipt.get("galaxy", {}).get("decision") != "blocked":
        raise ValueError("galaxy observational control is not blocked")
    for domain in ("lensing", "cluster"):
        control = receipt.get(domain, {})
        if (
            control.get("schema_control_passes") != 2
            or control.get("real_source_packets_opened") != 0
            or control.get("scientific_pass") is not False
            or control.get("scientific_reject") is not False
            or control.get("rank_writes") != 0
        ):
            raise ValueError(f"{domain} observational control changed")
