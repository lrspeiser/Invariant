"""Scalar-free-branch GR galaxy forward model over sealed synthetic inputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

from .promotion_orchestrator import ELIGIBILITY
from .reviewed_g4_candidate_galaxy_evaluator import (
    ACTION_SHA256,
    CANDIDATE_ID,
    DESCRIPTOR_FIELD,
    FORMAL_PROVENANCE_SHA256,
    REQUIRED_REGISTRATION_HASHES,
    reviewed_g4_candidate_galaxy_evaluator,
)

SCHEMA_VERSION = "sigma-g4-scalar-free-galaxy-forward-model-1.0"
PROFILE_SCHEMA = "sigma-g4-galaxy-baryonic-profile-input-1.0"
CALIBRATION_SCHEMA = "sigma-g4-galaxy-shared-calibration-1.0"
G_SI = 6.67430e-11
C_SI = 299_792_458.0
UNIT_CONTRACT = {
    "angular_radius": "rad",
    "nonredshift_lens_distance": "m",
    "lensing_distance_ratio": "dimensionless_D_ls_over_D_s",
    "surface_brightness_and_gas_intensity": "calibrated_input_intensity_per_sr",
    "surface_density_conversion": "kg_per_m2_per_input_intensity",
    "point_flux": "calibrated_integrated_flux",
    "point_mass_conversion": "kg_per_flux_m2",
    "circular_and_line_of_sight_speed": "m_per_s",
    "wavelength_ratio": "dimensionless",
    "lensing_angles": "rad",
}
GEOMETRY_CONTRACT = {
    "rotation": "axisymmetric_circular_orbit_model",
    "lensing": "circularized_axisymmetric_projected_baryonic_profile",
    "inclination_use": "rotation_line_of_sight_projection_only",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound G4 galaxy forward-model artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or (
            "content_sha256" in value and value["content_sha256"] != expected
        ):
            raise ValueError(
                f"bound G4 galaxy forward-model content changed: {binding['path']}"
            )
    return value


def _positive_array(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty vector")
    if np.any(result <= 0.0) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be positive and strictly increasing")
    return result


def _validate_inputs(
    profile: dict[str, Any], calibrations: dict[str, Any]
) -> tuple[np.ndarray, float, float, float, dict[str, float]]:
    required_profile = {
        "schema_version",
        "angular_radii_rad",
        "inclination_rad",
        "nonredshift_lens_distance_m",
        "lensing_distance_ratio",
        "components",
        "geometry_contract",
        "unit_contract",
        "distance_provenance",
        "data_eligibility",
        "observational_data_opened",
    }
    if (
        set(profile) != required_profile
        or profile.get("schema_version") != PROFILE_SCHEMA
        or profile.get("unit_contract") != UNIT_CONTRACT
        or profile.get("geometry_contract") != GEOMETRY_CONTRACT
        or profile.get("distance_provenance")
        != "separately_registered_nonredshift_distance_required"
        or profile.get("data_eligibility") != ELIGIBILITY
        or profile.get("observational_data_opened") is not False
    ):
        raise ValueError("G4 galaxy profile violates the sealed input contract")
    radii = _positive_array(profile["angular_radii_rad"], "angular radii")
    inclination = float(profile["inclination_rad"])
    distance = float(profile["nonredshift_lens_distance_m"])
    ratio = float(profile["lensing_distance_ratio"])
    if (
        not math.isfinite(inclination)
        or not 0.0 <= inclination <= math.pi / 2.0
        or not math.isfinite(distance)
        or distance <= 0.0
        or not math.isfinite(ratio)
        or not 0.0 <= ratio <= 1.0
    ):
        raise ValueError("G4 galaxy angular geometry is outside the admitted domain")

    required_calibration = {
        "schema_version",
        "shared_across_all_objects",
        "gravitational_constant_m3_kg_s2",
        "speed_of_light_m_s",
        "surface_density_per_intensity",
        "point_mass_per_flux_distance_squared",
        "object_specific_gravity_parameters",
        "unit_contract",
        "data_eligibility",
    }
    if (
        set(calibrations) != required_calibration
        or calibrations.get("schema_version") != CALIBRATION_SCHEMA
        or calibrations.get("shared_across_all_objects") is not True
        or calibrations.get("gravitational_constant_m3_kg_s2") != G_SI
        or calibrations.get("speed_of_light_m_s") != C_SI
        or calibrations.get("object_specific_gravity_parameters") != {}
        or calibrations.get("unit_contract") != UNIT_CONTRACT
        or calibrations.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("G4 galaxy calibrations are not universal and unit-bound")
    conversions = calibrations.get("surface_density_per_intensity")
    if not isinstance(conversions, dict) or set(conversions) != {
        "stellar_light",
        "gas_line",
    }:
        raise ValueError("G4 galaxy surface-density calibration channels changed")
    numeric = {
        key: float(value)
        for key, value in {
            **conversions,
            "point": calibrations["point_mass_per_flux_distance_squared"],
        }.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric.values()):
        raise ValueError("G4 galaxy shared calibrations must be finite and positive")
    return radii, inclination, distance, ratio, numeric


def _disk_terms(
    radius_m: np.ndarray, central_sigma: float, scale_m: float
) -> tuple[np.ndarray, np.ndarray]:
    y = radius_m / (2.0 * scale_m)
    kernel = np.asarray(
        [
            float(
                mpmath.besseli(0, value) * mpmath.besselk(0, value)
                - mpmath.besseli(1, value) * mpmath.besselk(1, value)
            )
            for value in y
        ],
        dtype=float,
    )
    velocity_squared = 4.0 * math.pi * G_SI * central_sigma * scale_m * y**2 * kernel
    x = radius_m / scale_m
    projected_mass = (
        2.0
        * math.pi
        * central_sigma
        * scale_m**2
        * (1.0 - np.exp(-x) * (1.0 + x))
    )
    return velocity_squared, projected_mass


def _plummer_terms(
    radius_m: np.ndarray, central_sigma: float, scale_m: float
) -> tuple[np.ndarray, np.ndarray]:
    total_mass = math.pi * central_sigma * scale_m**2
    velocity_squared = (
        G_SI * total_mass * radius_m**2 / (radius_m**2 + scale_m**2) ** 1.5
    )
    projected_mass = total_mass * radius_m**2 / (radius_m**2 + scale_m**2)
    return velocity_squared, projected_mass


def _point_terms(
    radius_m: np.ndarray, total_flux: float, distance_m: float, conversion: float
) -> tuple[np.ndarray, np.ndarray]:
    mass = total_flux * conversion * distance_m**2
    return G_SI * mass / radius_m, np.full(radius_m.shape, mass, dtype=float)


def scalar_free_g4_galaxy_forward_model(
    profile: dict[str, Any], calibrations: dict[str, Any]
) -> dict[str, Any]:
    """Predict baryons-only scalar-free GR rotation and weak-lensing signals."""

    angular_radii, inclination, distance, lensing_ratio, conversions = _validate_inputs(
        profile, calibrations
    )
    physical_radii = distance * angular_radii
    velocity_squared = np.zeros_like(physical_radii)
    projected_mass = np.zeros_like(physical_radii)
    components = profile["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("G4 galaxy profile requires at least one baryonic component")
    component_summaries = []
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("G4 galaxy baryonic component must be a mapping")
        kind = component.get("kind")
        tracer = component.get("tracer")
        if kind in {"exponential_thin_disk", "plummer_sphere"}:
            if set(component) != {
                "kind",
                "tracer",
                "central_intensity_per_sr",
                "angular_scale_rad",
            } or tracer not in {"stellar_light", "gas_line"}:
                raise ValueError("G4 galaxy resolved baryonic component changed")
            intensity = float(component["central_intensity_per_sr"])
            angular_scale = float(component["angular_scale_rad"])
            if (
                not math.isfinite(intensity)
                or intensity <= 0.0
                or not math.isfinite(angular_scale)
                or angular_scale <= 0.0
            ):
                raise ValueError("G4 galaxy resolved component is outside its domain")
            central_sigma = conversions[tracer] * intensity
            scale = distance * angular_scale
            if kind == "exponential_thin_disk":
                v2, mass = _disk_terms(physical_radii, central_sigma, scale)
            else:
                v2, mass = _plummer_terms(physical_radii, central_sigma, scale)
            component_summaries.append(
                {
                    "kind": kind,
                    "tracer": tracer,
                    "central_surface_density_kg_m2": central_sigma,
                    "physical_scale_m": scale,
                }
            )
        elif kind == "unresolved_baryonic_point":
            if set(component) != {"kind", "tracer", "total_flux"} or tracer not in {
                "stellar_light",
                "gas_line",
            }:
                raise ValueError("G4 galaxy point component changed")
            total_flux = float(component["total_flux"])
            if not math.isfinite(total_flux) or total_flux <= 0.0:
                raise ValueError("G4 galaxy point flux is outside its domain")
            v2, mass = _point_terms(
                physical_radii, total_flux, distance, conversions["point"]
            )
            component_summaries.append(
                {"kind": kind, "tracer": tracer, "calibrated_integrated_flux": total_flux}
            )
        else:
            raise ValueError("G4 galaxy component family is not admitted")
        velocity_squared += v2
        projected_mass += mass

    circular_speed = np.sqrt(velocity_squared)
    line_of_sight_speed = circular_speed * math.sin(inclination)
    beta = line_of_sight_speed / C_SI
    if np.any(beta >= 1.0):
        raise ValueError("G4 galaxy synthetic line-of-sight speed is nonphysical")
    wavelength_ratio = np.sqrt((1.0 + beta) / (1.0 - beta))
    reduced_deflection = (
        4.0 * G_SI * projected_mass / (C_SI**2 * physical_radii) * lensing_ratio
    )
    source_plane_relative_angle = angular_radii - reduced_deflection
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "formal_provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "branch": "exact_scalar_free_phi_zero_GR_branch",
        "angular_radii_rad": angular_radii.tolist(),
        "physical_radii_m": physical_radii.tolist(),
        "intrinsic_circular_speed_m_s": circular_speed.tolist(),
        "line_of_sight_speed_m_s": line_of_sight_speed.tolist(),
        "spectral_wavelength_ratio_receding": wavelength_ratio.tolist(),
        "relative_weak_lensing_deflection_rad": reduced_deflection.tolist(),
        "source_plane_relative_angle_rad": source_plane_relative_angle.tolist(),
        "component_summaries": component_summaries,
        "object_specific_gravity_parameter_count": 0,
        "prediction_bundle_registered": False,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "unit_contract": UNIT_CONTRACT,
        "geometry_contract": GEOMETRY_CONTRACT,
        "data_eligibility": dict(ELIGIBILITY),
    }


def propagate_synthetic_covariance(
    jacobian: list[list[float]], input_covariance: list[list[float]]
) -> list[list[float]]:
    jac = np.asarray(jacobian, dtype=float)
    covariance = np.asarray(input_covariance, dtype=float)
    if (
        jac.ndim != 2
        or covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or jac.shape[1] != covariance.shape[0]
        or not np.all(np.isfinite(jac))
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-15)
        or float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-15
    ):
        raise ValueError("synthetic covariance propagation inputs are invalid")
    return (jac @ covariance @ jac.T).tolist()


def _synthetic_profile(kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    component: dict[str, Any]
    if kind == "unresolved_baryonic_point":
        component = {"kind": kind, "tracer": "stellar_light", "total_flux": 2.0}
    else:
        component = {
            "kind": kind,
            "tracer": "stellar_light",
            "central_intensity_per_sr": 3.0,
            "angular_scale_rad": 0.01,
        }
    profile = {
        "schema_version": PROFILE_SCHEMA,
        "angular_radii_rad": [0.01, 0.02],
        "inclination_rad": math.pi / 6.0,
        "nonredshift_lens_distance_m": 1.0e20,
        "lensing_distance_ratio": 0.5,
        "components": [component],
        "geometry_contract": GEOMETRY_CONTRACT,
        "unit_contract": UNIT_CONTRACT,
        "distance_provenance": "separately_registered_nonredshift_distance_required",
        "data_eligibility": dict(ELIGIBILITY),
        "observational_data_opened": False,
    }
    calibrations = {
        "schema_version": CALIBRATION_SCHEMA,
        "shared_across_all_objects": True,
        "gravitational_constant_m3_kg_s2": G_SI,
        "speed_of_light_m_s": C_SI,
        "surface_density_per_intensity": {"stellar_light": 2.0, "gas_line": 3.0},
        "point_mass_per_flux_distance_squared": 4.0e-10,
        "object_specific_gravity_parameters": {},
        "unit_contract": UNIT_CONTRACT,
        "data_eligibility": dict(ELIGIBILITY),
    }
    return profile, calibrations


def _synthetic_controls() -> dict[str, Any]:
    disk_profile, calibration = _synthetic_profile("exponential_thin_disk")
    disk = scalar_free_g4_galaxy_forward_model(disk_profile, calibration)
    sphere_profile, _ = _synthetic_profile("plummer_sphere")
    sphere = scalar_free_g4_galaxy_forward_model(sphere_profile, calibration)
    point_profile, _ = _synthetic_profile("unresolved_baryonic_point")
    point = scalar_free_g4_galaxy_forward_model(point_profile, calibration)

    radius = 1.0e18
    central_sigma = 6.0
    scale = 1.0e18
    total_mass = math.pi * central_sigma * scale**2
    sphere_expected_v2 = G_SI * total_mass / (2.0 ** 1.5 * scale)
    sphere_actual_v2 = sphere["intrinsic_circular_speed_m_s"][0] ** 2
    point_mass = 2.0 * 4.0e-10 * (1.0e20) ** 2
    point_expected_v2 = G_SI * point_mass / radius
    point_actual_v2 = point["intrinsic_circular_speed_m_s"][0] ** 2
    point_expected_alpha = 4.0 * G_SI * point_mass / (C_SI**2 * radius) * 0.5
    point_actual_alpha = point["relative_weak_lensing_deflection_rad"][0]

    disk_mass_expected = (
        2.0 * math.pi * central_sigma * scale**2 * (1.0 - 2.0 / math.e)
    )
    disk_alpha_expected = 4.0 * G_SI * disk_mass_expected / (C_SI**2 * radius) * 0.5
    disk_alpha_actual = disk["relative_weak_lensing_deflection_rad"][0]
    tolerances = {
        "disk_projected_mass_lensing": abs(disk_alpha_actual - disk_alpha_expected)
        / disk_alpha_expected,
        "plummer_speed_at_scale": abs(sphere_actual_v2 - sphere_expected_v2)
        / sphere_expected_v2,
        "point_circular_speed": abs(point_actual_v2 - point_expected_v2)
        / point_expected_v2,
        "point_lens_deflection": abs(point_actual_alpha - point_expected_alpha)
        / point_expected_alpha,
    }
    if max(tolerances.values()) > 1.0e-12:
        raise ValueError("G4 galaxy analytic synthetic known-answer failed")

    propagated = np.asarray(
        propagate_synthetic_covariance(
            [[1.0, 2.0], [3.0, -1.0]], [[4.0, 1.0], [1.0, 9.0]]
        )
    )
    expected = np.array([[44.0, -1.0], [-1.0, 39.0]])
    if not np.allclose(propagated, expected, rtol=0.0, atol=1.0e-14):
        raise ValueError("G4 galaxy covariance synthetic known-answer failed")

    mass = (1, 0, 0)
    length = (0, 1, 0)
    gravitational_constant = (-1, 3, -2)
    surface_density = (1, -2, 0)
    speed_of_light = (0, 1, -1)

    def combine(*dimensions: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(sum(axis) for axis in zip(*dimensions, strict=True))

    def inverse(dimension: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(-axis for axis in dimension)

    dimension_results = {
        "disk_velocity_squared": combine(
            gravitational_constant, surface_density, length
        ),
        "point_velocity_squared": combine(
            gravitational_constant, mass, inverse(length)
        ),
        "lensing_angle": combine(
            gravitational_constant,
            mass,
            inverse(length),
            inverse(speed_of_light),
            inverse(speed_of_light),
        ),
        "physical_radius": length,
    }
    if dimension_results != {
        "disk_velocity_squared": (0, 2, -2),
        "point_velocity_squared": (0, 2, -2),
        "lensing_angle": (0, 0, 0),
        "physical_radius": (0, 1, 0),
    }:
        raise ValueError("G4 galaxy dimension algebra failed")
    dimension_identities = {
        "G_times_surface_density_times_length": "L^2 T^-2",
        "G_times_mass_over_radius": "L^2 T^-2",
        "G_times_mass_over_c2_radius": "dimensionless",
        "distance_times_angle": "L",
    }
    body = {
        "analytic_known_answers": {
            "exponential_disk": "pass",
            "plummer_sphere": "pass",
            "baryonic_point_lens": "pass",
        },
        "relative_error_by_control": tolerances,
        "covariance": {
            "decision": "pass",
            "propagated_matrix": propagated.tolist(),
            "positive_definite": bool(np.min(np.linalg.eigvalsh(propagated)) > 0.0),
        },
        "unit_dimension_checks": dimension_identities,
        "object_specific_gravity_parameter_count": 0,
        "observational_data_opened": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_scalar_free_galaxy_forward_model_campaign(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G4 galaxy forward-model eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("G4 galaxy forward model opened observations")
    bindings = config["source_bindings"]
    source_binding = bindings["forward_model_source"]
    source_path = root / source_binding["path"]
    if not source_path.is_file() or _file_sha(source_path) != source_binding["file_sha256"]:
        raise ValueError("G4 galaxy forward-model source changed")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key != "forward_model_source"
    }
    prediction = sources["scalar_free_branch"]["candidate_records"][0]
    branch = prediction["exact_scalar_free_branch_certificate"]
    if (
        prediction.get("seed_id") != CANDIDATE_ID
        or prediction.get("action_sha256") != ACTION_SHA256
        or prediction.get("provenance", {}).get("formal_provenance_sha256")
        != FORMAL_PROVENANCE_SHA256
        or branch.get("content_sha256")
        != "2bca9d26343843231a8333bc9ac2396c395c388d24f55ae488c04c05f59256dc"
        or branch.get("branch_selection_warning") is None
    ):
        raise ValueError("G4 scalar-free galaxy branch binding changed")
    contract = sources["prediction_bundle_contract"]
    if (
        contract.get("properties", {}).get("action_sha256", {}).get("const")
        != ACTION_SHA256
        or contract.get("properties", {})
        .get("object_specific_gravity_parameter_count", {})
        .get("const")
        != 0
    ):
        raise ValueError("G4 galaxy prediction contract changed")
    readiness = sources["evaluator_readiness"]
    predecessor = readiness["current_evaluator_decision"]
    if (
        readiness.get("decision") != "blocked"
        or predecessor.get("filled_registration_hash_count") != 1
        or len(predecessor.get("missing_registration_hashes", [])) != 17
        or readiness.get("prediction_bundle_registered") is not False
        or readiness.get("observational_data_opened") is not False
    ):
        raise ValueError("G4 galaxy evaluator readiness predecessor changed")

    rotation_binding = _sha(
        {
            "source_sha256": source_binding["file_sha256"],
            "callable": "scalar_free_g4_galaxy_forward_model:rotation",
            "action_sha256": ACTION_SHA256,
        }
    )
    lensing_binding = _sha(
        {
            "source_sha256": source_binding["file_sha256"],
            "callable": "scalar_free_g4_galaxy_forward_model:lensing",
            "action_sha256": ACTION_SHA256,
        }
    )
    registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    registration[DESCRIPTOR_FIELD] = readiness["implementation_readiness"][
        "descriptor_binding_sha256"
    ]
    registration["rotation_prediction_implementation_sha256"] = rotation_binding
    registration["lensing_prediction_implementation_sha256"] = lensing_binding
    current = reviewed_g4_candidate_galaxy_evaluator(
        {
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "role": "generated_candidate",
            "data_eligibility": dict(ELIGIBILITY),
        },
        {
            "data_eligibility": dict(ELIGIBILITY),
            "observational_opening_authorized": False,
            "registration_hashes": registration,
        },
    )
    expected_missing = sorted(
        set(REQUIRED_REGISTRATION_HASHES)
        - {
            DESCRIPTOR_FIELD,
            "rotation_prediction_implementation_sha256",
            "lensing_prediction_implementation_sha256",
        }
    )
    if (
        current.get("decision") != "blocked"
        or current.get("filled_registration_hash_count") != 3
        or current.get("missing_registration_hashes") != expected_missing
    ):
        raise ValueError("G4 galaxy forward-model registration ledger changed")
    controls = _synthetic_controls()
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "formal_provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "scalar_free_branch_sha256": branch["content_sha256"],
        "prediction_bundle_contract_sha256": bindings["prediction_bundle_contract"][
            "content_sha256"
        ],
        "evaluator_readiness_sha256": bindings["evaluator_readiness"][
            "content_sha256"
        ],
        "forward_model_source_sha256": source_binding["file_sha256"],
        "synthetic_controls_sha256": controls["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "branch": "exact_scalar_free_phi_zero_GR_branch",
        "source_bindings": bindings,
        "forward_model_contract": {
            "admitted_component_families": [
                "exponential_thin_disk",
                "plummer_sphere",
                "unresolved_baryonic_point",
            ],
            "admitted_tracers": ["stellar_light", "gas_line"],
            "distance_requirement": "separately_registered_nonredshift_distance",
            "shared_calibrations_only": True,
            "object_specific_gravity_parameter_count": 0,
            "rotation_outputs": [
                "intrinsic_circular_speed_m_s",
                "line_of_sight_speed_m_s",
                "spectral_wavelength_ratio_receding",
            ],
            "lensing_outputs": [
                "relative_weak_lensing_deflection_rad",
                "source_plane_relative_angle_rad",
            ],
            "unit_contract": UNIT_CONTRACT,
            "geometry_contract": GEOMETRY_CONTRACT,
        },
        "synthetic_controls": controls,
        "newly_filled_registration_fields": {
            "rotation_prediction_implementation_sha256": rotation_binding,
            "lensing_prediction_implementation_sha256": lensing_binding,
        },
        "preserved_predecessor_registration_fields": {
            DESCRIPTOR_FIELD: registration[DESCRIPTOR_FIELD]
        },
        "unfilled_registration_fields": expected_missing,
        "current_evaluator_decision": current,
        "prediction_bundle_registered": False,
        "candidate_use_authorized": False,
        "observational_authorization": False,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "object_specific_gravity_parameter_count": 0,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "decision": "blocked",
        "first_missing_premise": "registered_baryonic_source_and_data_contracts",
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "The adapter implements only the exact scalar-free GR branch over typed synthetic "
            "baryonic profiles and separately registered non-redshift geometry. Rotation and "
            "lensing implementation hashes are filled, but no galaxy source, split, likelihood, "
            "calibration hierarchy, prediction bundle, or observation is registered."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
