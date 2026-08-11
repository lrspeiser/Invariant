from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import mpmath
import numpy as np
import pytest

from sigma_theory_compiler.g4_scalar_free_galaxy_forward_model import (
    C_SI,
    G_SI,
    GEOMETRY_CONTRACT,
    UNIT_CONTRACT,
    _synthetic_profile,
    build_g4_scalar_free_galaxy_forward_model_campaign,
    propagate_synthetic_covariance,
    scalar_free_g4_galaxy_forward_model,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.reviewed_g4_candidate_galaxy_evaluator import (
    DESCRIPTOR_FIELD,
    REQUIRED_REGISTRATION_HASHES,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_scalar_free_galaxy_forward_model.json"
ARTIFACT = ROOT / "runs" / "engine" / "g4-scalar-free-galaxy-forward-model.json"
SOURCE = (
    ROOT
    / "src"
    / "sigma_theory_compiler"
    / "g4_scalar_free_galaxy_forward_model.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exponential_disk_matches_exact_rotation_and_projected_mass_lensing() -> None:
    profile, calibration = _synthetic_profile("exponential_thin_disk")
    result = scalar_free_g4_galaxy_forward_model(profile, calibration)
    radius = 1.0e18
    scale = 1.0e18
    central_sigma = 6.0
    y = 0.5
    kernel = float(
        mpmath.besseli(0, y) * mpmath.besselk(0, y)
        - mpmath.besseli(1, y) * mpmath.besselk(1, y)
    )
    expected_v2 = 4.0 * math.pi * G_SI * central_sigma * scale * y**2 * kernel
    expected_mass = (
        2.0 * math.pi * central_sigma * scale**2 * (1.0 - 2.0 / math.e)
    )
    expected_alpha = 4.0 * G_SI * expected_mass / (C_SI**2 * radius) * 0.5
    assert result["intrinsic_circular_speed_m_s"][0] ** 2 == pytest.approx(
        expected_v2, rel=1.0e-13
    )
    assert result["relative_weak_lensing_deflection_rad"][0] == pytest.approx(
        expected_alpha, rel=1.0e-13
    )
    assert result["object_specific_gravity_parameter_count"] == 0
    assert result["dark_matter_or_halo_inputs"] is False
    assert result["redshift_distance_inputs"] is False


def test_plummer_sphere_and_point_lens_match_analytic_known_answers() -> None:
    sphere_profile, calibration = _synthetic_profile("plummer_sphere")
    sphere = scalar_free_g4_galaxy_forward_model(sphere_profile, calibration)
    scale = 1.0e18
    total_mass = math.pi * 6.0 * scale**2
    expected_sphere_v2 = G_SI * total_mass / (2.0**1.5 * scale)
    expected_sphere_mass_2d = total_mass / 2.0
    expected_sphere_alpha = (
        4.0 * G_SI * expected_sphere_mass_2d / (C_SI**2 * scale) * 0.5
    )
    assert sphere["intrinsic_circular_speed_m_s"][0] ** 2 == pytest.approx(
        expected_sphere_v2, rel=1.0e-13
    )
    assert sphere["relative_weak_lensing_deflection_rad"][0] == pytest.approx(
        expected_sphere_alpha, rel=1.0e-13
    )

    point_profile, _ = _synthetic_profile("unresolved_baryonic_point")
    point = scalar_free_g4_galaxy_forward_model(point_profile, calibration)
    mass = 2.0 * 4.0e-10 * (1.0e20) ** 2
    assert point["intrinsic_circular_speed_m_s"][0] ** 2 == pytest.approx(
        G_SI * mass / scale, rel=1.0e-13
    )
    assert point["relative_weak_lensing_deflection_rad"][0] == pytest.approx(
        4.0 * G_SI * mass / (C_SI**2 * scale) * 0.5,
        rel=1.0e-13,
    )


def test_baryonic_components_add_in_potential_and_doppler_is_dimensionless() -> None:
    profile, calibration = _synthetic_profile("exponential_thin_disk")
    single = scalar_free_g4_galaxy_forward_model(profile, calibration)
    doubled_profile = copy.deepcopy(profile)
    doubled_profile["components"].append(copy.deepcopy(profile["components"][0]))
    doubled = scalar_free_g4_galaxy_forward_model(doubled_profile, calibration)
    assert np.square(doubled["intrinsic_circular_speed_m_s"]) == pytest.approx(
        2.0 * np.square(single["intrinsic_circular_speed_m_s"]), rel=1.0e-13
    )
    assert doubled["relative_weak_lensing_deflection_rad"] == pytest.approx(
        2.0 * np.asarray(single["relative_weak_lensing_deflection_rad"]), rel=1.0e-13
    )
    beta = doubled["line_of_sight_speed_m_s"][0] / C_SI
    assert doubled["spectral_wavelength_ratio_receding"][0] == pytest.approx(
        math.sqrt((1.0 + beta) / (1.0 - beta)), rel=1.0e-15
    )
    assert doubled["unit_contract"] == UNIT_CONTRACT
    assert doubled["geometry_contract"] == GEOMETRY_CONTRACT
    assert doubled["observational_data_opened"] is False


def test_covariance_known_answer_and_invalid_covariance_are_exact() -> None:
    propagated = propagate_synthetic_covariance(
        [[1.0, 2.0], [3.0, -1.0]], [[4.0, 1.0], [1.0, 9.0]]
    )
    assert propagated == [[44.0, -1.0], [-1.0, 39.0]]
    assert float(np.min(np.linalg.eigvalsh(propagated))) > 0.0
    with pytest.raises(ValueError, match="covariance propagation inputs"):
        propagate_synthetic_covariance([[1.0, 0.0]], [[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="covariance propagation inputs"):
        propagate_synthetic_covariance([[1.0, 0.0]], [[1.0, 2.0], [2.0, 1.0]])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("halo", "component family is not admitted"),
        ("redshift", "sealed input contract"),
        ("unit", "sealed input contract"),
        ("object_gravity", "not universal and unit-bound"),
        ("unshared", "not universal and unit-bound"),
        ("distance", "angular geometry"),
    ],
)
def test_forbidden_inputs_units_and_geometry_fail_closed(
    mutation: str, message: str
) -> None:
    profile, calibration = _synthetic_profile("exponential_thin_disk")
    if mutation == "halo":
        profile["components"][0]["kind"] = "NFW_halo"
    elif mutation == "redshift":
        profile["distance_provenance"] = "redshift_derived_distance"
    elif mutation == "unit":
        profile["unit_contract"] = {**UNIT_CONTRACT, "angular_radius": "degree"}
    elif mutation == "object_gravity":
        calibration["object_specific_gravity_parameters"] = {"G": 2.0}
    elif mutation == "unshared":
        calibration["shared_across_all_objects"] = False
    else:
        profile["nonredshift_lens_distance_m"] = -1.0
    with pytest.raises(ValueError, match=message):
        scalar_free_g4_galaxy_forward_model(profile, calibration)


def test_artifact_rebuilds_and_fills_only_rotation_and_lensing_implementations() -> None:
    stored = _load(ARTIFACT)
    rebuilt = build_g4_scalar_free_galaxy_forward_model_campaign(_load(CONFIG), ROOT)
    assert rebuilt == stored
    assert stored["content_sha256"] == (
        "cfc514924fda33ee084c03a1b39e72a0c5e0f2513fb5239ad652f4594a2e2a7b"
    )
    assert _file_sha(ARTIFACT) == (
        "5671eb3380eac4468f17afe9359db5e38f2455df8745d358f190a865abac8107"
    )
    assert set(stored["newly_filled_registration_fields"]) == {
        "rotation_prediction_implementation_sha256",
        "lensing_prediction_implementation_sha256",
    }
    assert stored["preserved_predecessor_registration_fields"].keys() == {
        DESCRIPTOR_FIELD
    }
    expected_missing = sorted(
        set(REQUIRED_REGISTRATION_HASHES)
        - {
            DESCRIPTOR_FIELD,
            "rotation_prediction_implementation_sha256",
            "lensing_prediction_implementation_sha256",
        }
    )
    assert stored["unfilled_registration_fields"] == expected_missing
    assert stored["current_evaluator_decision"]["decision"] == "blocked"
    assert stored["current_evaluator_decision"]["filled_registration_hash_count"] == 3
    assert len(expected_missing) == 15
    assert stored["prediction_bundle_registered"] is False
    assert stored["observational_data_opened"] is False
    assert stored["primary_record_access_count"] == 0
    assert stored["object_specific_gravity_parameter_count"] == 0


def test_source_branch_contract_and_authorization_tampering_are_rejected() -> None:
    assert _load(CONFIG)["source_bindings"]["forward_model_source"][
        "file_sha256"
    ] == _file_sha(SOURCE)
    for key in ("scalar_free_branch", "prediction_bundle_contract", "evaluator_readiness"):
        config = _load(CONFIG)
        config["source_bindings"][key]["content_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="content changed"):
            build_g4_scalar_free_galaxy_forward_model_campaign(config, ROOT)
    config = _load(CONFIG)
    config["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_g4_scalar_free_galaxy_forward_model_campaign(config, ROOT)


def test_all_synthetic_controls_and_dimension_checks_pass() -> None:
    controls = _load(ARTIFACT)["synthetic_controls"]
    assert set(controls["analytic_known_answers"].values()) == {"pass"}
    assert max(controls["relative_error_by_control"].values()) < 1.0e-12
    assert controls["covariance"]["propagated_matrix"] == [[44.0, -1.0], [-1.0, 39.0]]
    assert controls["covariance"]["positive_definite"] is True
    assert controls["unit_dimension_checks"] == {
        "G_times_surface_density_times_length": "L^2 T^-2",
        "G_times_mass_over_radius": "L^2 T^-2",
        "G_times_mass_over_c2_radius": "dimensionless",
        "distance_times_angle": "L",
    }
    assert controls["observational_data_opened"] is False
    assert controls["object_specific_gravity_parameter_count"] == 0
    assert _load(ARTIFACT)["data_eligibility"] == ELIGIBILITY
