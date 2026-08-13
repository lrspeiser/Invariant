from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.lensing_direct_observable_evaluator_readiness as lensing

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / lensing.CONFIG_REL
ARTIFACT = ROOT / lensing.ARTIFACT_REL


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = lensing._sha(value)
    return value


def test_readiness_rebuilds_exactly_with_zero_scientific_pass() -> None:
    config = lensing.load_config(ROOT, CONFIG)
    stored = _load(ARTIFACT)
    rebuilt = lensing.build_readiness(config, ROOT)
    assert rebuilt == stored
    lensing.validate_readiness(stored, config, ROOT)
    assert stored["decision"] == lensing.DECISION
    assert stored["counts"] == {
        "allowed_data_classes": 2,
        "allowed_observable_channels": 6,
        "required_calibration_roles": 4,
        "forbidden_fields": 19,
        "positive_schema_controls": 2,
        "positive_schema_control_passes": 2,
        "authorized_real_source_packets": 0,
        "real_source_packets_opened": 0,
        "scientific_passes": 0,
        "scientific_rejects": 0,
        "scientific_blocks": 1,
    }
    assert stored["first_blocker"] == "no_authorized_real_lensing_source_packet_registered"
    assert stored["observational_data_opened"] is False
    assert stored["scientific_pass_claimed"] is False


def test_positive_raw_and_calibrated_controls_are_schema_only() -> None:
    artifact = _load(ARTIFACT)
    assert [row["decision"] for row in artifact["positive_schema_controls"]] == [
        "schema_pass",
        "schema_pass",
    ]
    assert all(row["synthetic_only"] is True for row in artifact["positive_schema_controls"])
    assert all(row["scientific_pass"] is False for row in artifact["positive_schema_controls"])
    raw = lensing._synthetic_packet("raw_direct_observable", "raw_detector_counts")
    calibrated = lensing._synthetic_packet(
        "calibrated_direct_observable", "directly_measured_time_delay"
    )
    lensing.validate_direct_observable_packet(raw)
    lensing.validate_direct_observable_packet(calibrated)


def test_model_dependent_derived_and_latent_classes_are_not_admitted() -> None:
    for data_class in ("derived_model_output", "model_dependent", "latent"):
        packet = lensing._synthetic_packet("raw_direct_observable", "raw_detector_counts")
        packet["data_class"] = data_class
        with pytest.raises(ValueError, match="packet contract mismatch"):
            lensing.validate_direct_observable_packet(_reseal(packet))


@pytest.mark.parametrize(
    "forbidden",
    [
        "dark_matter_label",
        "halo_mass",
        "nfw_fit",
        "gr_derived_convergence",
        "redshift_derived_distance",
        "distance_modulus",
        "supernova_standardization",
        "latent_gravitating_component",
        "object_specific_gravity_parameter",
    ],
)
def test_halo_redshift_supernova_model_and_latent_fields_are_rejected(forbidden: str) -> None:
    packet = lensing._synthetic_packet(
        "calibrated_direct_observable", "relative_multiple_image_positions"
    )
    packet["transformation"][forbidden] = "forbidden"
    with pytest.raises(ValueError, match="forbidden lensing field present"):
        lensing.validate_direct_observable_packet(_reseal(packet))


def test_group_or_target_leakage_cannot_enter_formula_selection() -> None:
    packet = lensing._synthetic_packet("calibrated_direct_observable", "relative_arc_positions")
    packet["formula_selection_use"] = True
    with pytest.raises(ValueError, match="packet contract mismatch"):
        lensing.validate_direct_observable_packet(_reseal(packet))
    packet = lensing._synthetic_packet("calibrated_direct_observable", "relative_arc_positions")
    packet["target_role"] = "formula_selection_validation"
    with pytest.raises(ValueError, match="packet contract mismatch"):
        lensing.validate_direct_observable_packet(_reseal(packet))


def test_calibration_transformation_and_hash_bindings_fail_closed() -> None:
    packet = lensing._synthetic_packet("calibrated_direct_observable", "calibrated_image_pixels")
    packet["calibration_bindings"].pop()
    with pytest.raises(ValueError, match="calibration roles incomplete"):
        lensing.validate_direct_observable_packet(_reseal(packet))
    packet = lensing._synthetic_packet("calibrated_direct_observable", "calibrated_image_pixels")
    packet["transformation"]["kind"] = "model_inferred_convergence_map"
    with pytest.raises(ValueError, match="audited direct transformation"):
        lensing.validate_direct_observable_packet(_reseal(packet))
    packet = lensing._synthetic_packet("raw_direct_observable", "raw_detector_counts")
    packet["payload_file_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="packet hash invalid"):
        lensing.validate_direct_observable_packet(_reseal(packet))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("scientific_passes", 1),
        lambda value: value.__setitem__("observational_data_opened", True),
        lambda value: value.__setitem__("source_packet_opened", True),
        lambda value: value.__setitem__("scientific_pass_claimed", True),
        lambda value: value["data_eligibility"].__setitem__("dark_matter_or_halo_inputs", True),
        lambda value: value["seals"].__setitem__("network_access", True),
    ],
)
def test_resealed_false_pass_opening_or_forbidden_eligibility_fails_closed(
    mutation: object,
) -> None:
    config = lensing.load_config(ROOT, CONFIG)
    value = copy.deepcopy(_load(ARTIFACT))
    mutation(value)  # type: ignore[operator]
    with pytest.raises(ValueError):
        lensing.validate_readiness(_reseal(value), config, ROOT)


def test_policy_and_protocol_are_hash_bound_and_observations_remain_closed() -> None:
    config = lensing.load_config(ROOT, CONFIG)
    artifact = _load(ARTIFACT)
    assert artifact["source_bindings"] == {
        "evidence_policy": config["evidence_policy"],
        "galaxy_protocol": config["galaxy_protocol"],
    }
    assert artifact["authorized_real_source_packet_bindings"] == []
    assert artifact["observational_authorization"] is False
    assert artifact["candidate_use_authorized"] is False
    assert artifact["complete_comparable_evidence"] is False
    assert not any(artifact["seals"].values())


def test_build_and_validation_never_open_observation_runtime_sqlite_gpu_or_secret_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = lensing.load_config(ROOT, CONFIG)
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def excluded(path: Path) -> bool:
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
        if excluded(path):
            raise AssertionError("lensing readiness opened an excluded path")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if excluded(path):
            raise AssertionError("lensing readiness opened an excluded path")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    rebuilt = lensing.build_readiness(config, ROOT)
    lensing.validate_readiness(rebuilt, config, ROOT)
