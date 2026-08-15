from __future__ import annotations

import copy
from pathlib import Path

import pytest

import sigma_theory_compiler.cluster_direct_observable_evaluator_readiness as cluster
import sigma_theory_compiler.lensing_direct_observable_evaluator_readiness as lensing
from sigma_theory_compiler.alpha_observational_rehearsal import (
    _galaxy_control,
    _nuisance_covariance,
    _nuisance_values,
    _rsr_packet,
    _tdf_packet,
    build_observational_rehearsal,
    validate_observational_receipt,
)
from sigma_theory_compiler.galaxy_direct_observable_evaluator import (
    galaxy_direct_observable_evaluator,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.solar_direct_signal_calibration_readiness import (
    CalibrationContractError,
    calibrate_direct_signals,
    rsr_iq_direct_signals,
    tdf_direct_signals,
)

ROOT = Path(__file__).resolve().parents[1]


def _reseal(module: object, packet: dict[str, object]) -> dict[str, object]:
    packet.pop("content_sha256", None)
    packet["content_sha256"] = module._sha(packet)  # type: ignore[attr-defined]
    return packet


def test_four_domain_rehearsal_is_synthetic_blocked_and_unranked() -> None:
    receipt = build_observational_rehearsal(ROOT)
    validate_observational_receipt(receipt)
    assert receipt["solar"]["controls"] == {"calibration": 1, "rsr_parser": 1, "tdf_parser": 1}
    assert receipt["galaxy"]["decision"] == "blocked"
    assert receipt["lensing"]["schema_control_passes"] == 2
    assert receipt["cluster"]["schema_control_passes"] == 2
    assert receipt["claims"]["real_data_opened"] is False
    assert receipt["claims"]["scientific_pass"] is False
    assert receipt["claims"]["rank_writes"] == 0
    assert receipt["claims"]["registry_writes"] == 0


def test_receipt_replay_is_byte_stable() -> None:
    first = build_observational_rehearsal(ROOT)
    assert first == build_observational_rehearsal(ROOT)

    def contains_float(value: object) -> bool:
        if isinstance(value, float):
            return True
        if isinstance(value, dict):
            return any(contains_float(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_float(item) for item in value)
        return False

    assert contains_float(first) is False


def test_path_guard_allows_bindings_but_forbids_data_runtime_sqlite_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def prohibited(path: Path) -> bool:
        normalized = path.as_posix().lower()
        return any(
            token in normalized
            for token in (
                "observation-protocol",
                "campaign-v1-live.sqlite",
                "service-runtime",
                "gpu-scheduler-runtime",
                ".env",
                "secret",
                ".lease",
            )
        )

    def guarded_text(path: Path, *args: object, **kwargs: object) -> str:
        if prohibited(path):
            raise AssertionError(f"prohibited read: {path.name}")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if prohibited(path):
            raise AssertionError(f"prohibited read: {path.name}")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    receipt = build_observational_rehearsal(ROOT)
    assert receipt["claims"]["real_data_opened"] is False


def test_solar_units_and_galaxy_eligibility_fail_closed() -> None:
    bad_tdf = {
        "doppler_count_parts": [0, 12, 0],
        "range_parts": [120, 0, 0],
        "range_type": 1,
        "sample_interval_centisecond": 6000,
        "sky_frequency_hz": 2.0e9,
        "uplink_band": "S",
        "utc_time_tag": {
            "day_of_year": 180,
            "hour": 13,
            "minute": 11,
            "second": 25.0,
            "time_system": "UTC",
            "year": 2002,
        },
        "units": {
            "doppler_count": "count",
            "range_type_1": "nanosecond",
            "sample_interval": "second",
            "sky_frequency": "hertz",
        },
    }
    with pytest.raises(CalibrationContractError, match="units"):
        tdf_direct_signals(bad_tdf)
    with pytest.raises(ValueError, match="eligibility"):
        galaxy_direct_observable_evaluator(
            {"candidate_id": "bad", "data_eligibility": dict(ELIGIBILITY)},
            {"data_eligibility": {}, "input_lineage_sha256": "4" * 64},
        )
    assert _galaxy_control()["registered_prediction_bundle_count"] == 0

    unregistered = galaxy_direct_observable_evaluator(
        {
            "candidate_id": "unregistered",
            "data_eligibility": dict(ELIGIBILITY),
            "direct_observable_prediction_provenance": {
                "bundle_id": "synthetic-unregistered",
                "bundle_binding_sha256": "5" * 64,
                "candidate_action_sha256": "6" * 64,
                "prediction_content_sha256": "7" * 64,
                "observable_contract_sha256": "8" * 64,
            },
        },
        {"data_eligibility": dict(ELIGIBILITY), "input_lineage_sha256": "9" * 64},
    )
    assert unregistered["blocker"] == "unregistered_candidate_direct_observable_prediction_bundle"


def test_solar_non_psd_covariance_fails_closed() -> None:
    covariance = _nuisance_covariance()
    covariance[0][0] = -1.0
    with pytest.raises(CalibrationContractError, match="positive semidefinite"):
        calibrate_direct_signals(
            tdf_direct_signals(_tdf_packet()),
            rsr_iq_direct_signals(_rsr_packet()),
            _nuisance_values(),
            covariance,
            raw_tdf_variance_s2=0.0,
            reference_frequency_hz=8.4e9,
            maximum_time_tag_separation_s=1.0,
        )


def test_lensing_and_cluster_forbidden_fields_fail_closed() -> None:
    lens_packet = lensing._synthetic_packet(
        "calibrated_direct_observable", "relative_arc_positions"
    )
    lens_packet["transformation"]["redshift_derived_distance"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden lensing field"):
        lensing.validate_direct_observable_packet(_reseal(lensing, lens_packet))

    cluster_packet = cluster._synthetic_packet(
        "calibrated_direct_observable", "calibrated_xray_surface_brightness_or_spectrum"
    )
    cluster_packet["transformation"]["latent_nonthermal_pressure"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden cluster field"):
        cluster.validate_direct_observable_packet(_reseal(cluster, cluster_packet))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["claims"].__setitem__("scientific_pass", True),
        lambda value: value["claims"].__setitem__("rank_writes", 1),
        lambda value: value["lensing"].__setitem__("real_source_packets_opened", 1),
    ],
)
def test_forged_outcome_or_opening_receipt_fails_closed(mutation: object) -> None:
    receipt = copy.deepcopy(build_observational_rehearsal(ROOT))
    mutation(receipt)  # type: ignore[operator]
    with pytest.raises(ValueError, match="hash or schema"):
        validate_observational_receipt(receipt)
