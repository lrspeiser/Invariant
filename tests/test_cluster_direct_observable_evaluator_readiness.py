from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.cluster_direct_observable_evaluator_readiness as cluster

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / cluster.CONFIG_REL
ARTIFACT = ROOT / cluster.ARTIFACT_REL


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = cluster._sha(value)
    return value


def test_readiness_rebuilds_exactly_and_blocks_every_scientific_outcome() -> None:
    config = cluster.load_config(ROOT, CONFIG)
    stored = _load(ARTIFACT)
    rebuilt = cluster.build_readiness(config, ROOT)
    assert rebuilt == stored
    cluster.validate_readiness(stored, config, ROOT)
    assert stored["decision"] == cluster.DECISION
    assert stored["counts"] == {
        "allowed_data_classes": 2,
        "allowed_observable_channels": 8,
        "required_calibration_roles": 5,
        "required_covariance_roles": 2,
        "forbidden_fields": 24,
        "positive_schema_controls": 2,
        "positive_schema_control_passes": 2,
        "authorized_real_source_packets": 0,
        "real_source_packets_opened": 0,
        "scientific_passes": 0,
        "scientific_rejects": 0,
        "scientific_blocks": 1,
        "rank_writes": 0,
    }
    assert stored["first_blocker"] == "no_authorized_real_cluster_source_packet_registered"
    assert stored["scientific_pass_claimed"] is False
    assert stored["scientific_reject_claimed"] is False
    assert stored["rank_claimed"] is False


def test_positive_raw_and_calibrated_controls_are_schema_only() -> None:
    artifact = _load(ARTIFACT)
    controls = artifact["positive_schema_controls"]
    assert [row["decision"] for row in controls] == ["schema_pass", "schema_pass"]
    assert all(row["synthetic_only"] is True for row in controls)
    assert all(row["scientific_pass"] is False for row in controls)
    raw = cluster._synthetic_packet("raw_direct_observable", "raw_xray_detector_counts")
    calibrated = cluster._synthetic_packet(
        "calibrated_direct_observable", "calibrated_xray_surface_brightness_or_spectrum"
    )
    cluster.validate_direct_observable_packet(raw)
    cluster.validate_direct_observable_packet(calibrated)


def test_repository_text_hash_is_line_ending_invariant(tmp_path: Path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"gate":"closed"}\n')
    crlf.write_bytes(b'{"gate":"closed"}\r\n')
    assert cluster._file_sha(lf) == cluster._file_sha(crlf)


def test_model_dependent_derived_and_latent_classes_are_not_admitted() -> None:
    for data_class in ("hydrostatic_mass_product", "derived_model_output", "latent"):
        packet = cluster._synthetic_packet(
            "raw_direct_observable", "raw_sunyaev_zeldovich_detector_counts"
        )
        packet["data_class"] = data_class
        with pytest.raises(ValueError, match="packet contract mismatch"):
            cluster.validate_direct_observable_packet(_reseal(packet))


@pytest.mark.parametrize(
    "forbidden",
    [
        "cluster_halo_mass",
        "dark_matter_fraction",
        "nfw_fit",
        "hydrostatic_equilibrium_mass",
        "gr_derived_lensing_mass",
        "total_mass_profile",
        "redshift_derived_distance",
        "distance_modulus",
        "supernova_standardization",
        "latent_nonthermal_pressure",
        "object_specific_gravity_parameter",
        "formula_selection_target",
    ],
)
def test_halo_mass_inference_latent_and_leakage_fields_are_rejected(
    forbidden: str,
) -> None:
    packet = cluster._synthetic_packet(
        "calibrated_direct_observable",
        "calibrated_sunyaev_zeldovich_intensity_or_temperature_decrement",
    )
    packet["transformation"][forbidden] = "forbidden"
    with pytest.raises(ValueError, match="forbidden cluster field present"):
        cluster.validate_direct_observable_packet(_reseal(packet))


def test_formula_selection_and_target_role_leakage_are_rejected() -> None:
    packet = cluster._synthetic_packet(
        "calibrated_direct_observable", "calibrated_member_light_or_spectra"
    )
    packet["formula_selection_use"] = True
    with pytest.raises(ValueError, match="packet contract mismatch"):
        cluster.validate_direct_observable_packet(_reseal(packet))
    packet = cluster._synthetic_packet(
        "calibrated_direct_observable", "calibrated_member_light_or_spectra"
    )
    packet["target_role"] = "formula_selection_validation"
    with pytest.raises(ValueError, match="packet contract mismatch"):
        cluster.validate_direct_observable_packet(_reseal(packet))


def test_calibration_covariance_transformation_and_hashes_fail_closed() -> None:
    packet = cluster._synthetic_packet(
        "calibrated_direct_observable", "calibrated_xray_surface_brightness_or_spectrum"
    )
    packet["calibration_bindings"].pop()
    with pytest.raises(ValueError, match="calibration roles incomplete"):
        cluster.validate_direct_observable_packet(_reseal(packet))
    packet = cluster._synthetic_packet(
        "calibrated_direct_observable", "calibrated_xray_surface_brightness_or_spectrum"
    )
    packet["covariance_bindings"].pop()
    with pytest.raises(ValueError, match="covariance roles incomplete"):
        cluster.validate_direct_observable_packet(_reseal(packet))
    packet = cluster._synthetic_packet(
        "calibrated_direct_observable", "calibrated_xray_surface_brightness_or_spectrum"
    )
    packet["transformation"]["kind"] = "hydrostatic_mass_reconstruction"
    with pytest.raises(ValueError, match="audited direct transformation"):
        cluster.validate_direct_observable_packet(_reseal(packet))
    packet = cluster._synthetic_packet(
        "raw_direct_observable", "raw_optical_or_infrared_detector_counts"
    )
    packet["payload_file_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="packet hash invalid"):
        cluster.validate_direct_observable_packet(_reseal(packet))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["counts"].__setitem__("scientific_passes", 1),
        lambda value: value["counts"].__setitem__("scientific_rejects", 1),
        lambda value: value["counts"].__setitem__("rank_writes", 1),
        lambda value: value.__setitem__("observational_data_opened", True),
        lambda value: value.__setitem__("source_packet_opened", True),
        lambda value: value.__setitem__("rank_claimed", True),
        lambda value: value["data_eligibility"].__setitem__(
            "hydrostatic_or_gr_derived_mass_truth", True
        ),
        lambda value: value["seals"].__setitem__("network_access", True),
    ],
)
def test_resealed_false_outcome_opening_or_forbidden_eligibility_fails_closed(
    mutation: object,
) -> None:
    config = cluster.load_config(ROOT, CONFIG)
    value = copy.deepcopy(_load(ARTIFACT))
    mutation(value)  # type: ignore[operator]
    with pytest.raises(ValueError):
        cluster.validate_readiness(_reseal(value), config, ROOT)


def test_policy_bindings_are_scoped_and_observations_remain_closed() -> None:
    config = cluster.load_config(ROOT, CONFIG)
    artifact = _load(ARTIFACT)
    assert artifact["source_bindings"] == {
        "evidence_policy": config["evidence_policy"],
        "shared_galaxy_policy": config["shared_galaxy_policy"],
    }
    assert "no galaxy split or lensing target" in config["shared_galaxy_policy"]["applicability"]
    assert artifact["authorized_real_source_packet_bindings"] == []
    assert artifact["observational_authorization"] is False
    assert artifact["candidate_use_authorized"] is False
    assert artifact["complete_comparable_evidence"] is False
    assert not any(artifact["seals"].values())


def test_build_never_opens_observation_runtime_sqlite_gpu_network_or_secret_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cluster.load_config(ROOT, CONFIG)
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
            raise AssertionError("cluster readiness opened an excluded path")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if excluded(path):
            raise AssertionError("cluster readiness opened an excluded path")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    rebuilt = cluster.build_readiness(config, ROOT)
    cluster.validate_readiness(rebuilt, config, ROOT)
