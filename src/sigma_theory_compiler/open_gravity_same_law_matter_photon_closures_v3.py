"""Narrow dimensioned linear same-state matter and photon closure audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

CONFIG_PATH = Path("configs/open_gravity_same_law_matter_photon_closures_v3.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_same_law_matter_photon_closures_v3.py")
TEST_PATH = Path("tests/test_open_gravity_same_law_matter_photon_closures_v3.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-same-law-matter-photon-closures-v3/receipt.json")
ARTIFACT_DIRECTORY = OUTPUT_PATH.parent / "artifacts"
_RECEIPT_SCHEMA = "invariant-open-gravity-same-law-matter-photon-receipt-3.0"
_FORBIDDEN = (
    "photon_only_multiplier",
    "lens_only_coupling",
    "opacity",
    "frequency_dispersion",
    "path_memory_term",
    "separate_Fermat_coefficient",
    "separate_lensing_normalization",
)

Dimension = tuple[int, int, int]


class SameLawV3Error(RuntimeError):
    """Raised when the append-only narrow linear packet fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameLawV3Error(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _dimension(value: Sequence[int]) -> Dimension:
    _require(len(value) == 3, "dimension vector must have three entries")
    return (int(value[0]), int(value[1]), int(value[2]))


def _dimension_add(*values: Dimension) -> Dimension:
    return tuple(sum(value[index] for value in values) for index in range(3))  # type: ignore[return-value]


def _dimension_subtract(left: Dimension, *rights: Dimension) -> Dimension:
    return tuple(left[index] - sum(value[index] for value in rights) for index in range(3))  # type: ignore[return-value]


def dimension_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    declared = config["unit_system"]["dimensions"]
    dim = {key: _dimension(value) for key, value in declared.items()}
    length = dim["radius_or_distance"]
    mass = dim["source_mass"]
    inverse_length = dim["mediator_inverse_length_mu"]
    potential = dim["potential_U_Y_Phi_Psi"]
    speed = dim["c"]
    dimensionless = dim["angle_or_redshift"]
    checks: dict[str, tuple[Dimension, Dimension]] = {
        "potential_GM_over_r": (
            _dimension_subtract(_dimension_add(dim["G"], mass), length),
            potential,
        ),
        "yukawa_exponent_mu_r": (_dimension_add(inverse_length, length), dimensionless),
        "metric_ratio_Phi_over_c2": (
            _dimension_subtract(potential, speed, speed),
            dim["metric_component"],
        ),
        "radial_acceleration_Phi_prime": (
            _dimension_subtract(potential, length),
            dim["acceleration"],
        ),
        "born_deflection": (
            _dimension_subtract(
                _dimension_add(_dimension_subtract(potential, length), length), speed, speed
            ),
            dim["deflection"],
        ),
        "shapiro_delay": (
            _dimension_subtract(_dimension_add(potential, length), speed, speed, speed),
            dim["time_delay"],
        ),
        "massive_dispersion_c_times_inverse_length": (
            _dimension_add(speed, inverse_length),
            dim["angular_frequency_omega"],
        ),
        "massless_source_equation": (
            _dimension_subtract(potential, length, length),
            _dimension_add(dim["G"], mass, (-3, 0, 0)),
        ),
    }
    rows = [
        {
            "check": name,
            "observed": list(observed),
            "expected": list(expected),
            "pass": observed == expected,
        }
        for name, (observed, expected) in checks.items()
    ]
    return {
        "dimension_vector_order": ["L", "M", "T"],
        "unit_system": config["unit_system"]["name"],
        "rows": rows,
        "all_pass": all(row["pass"] for row in rows),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config["schema"].endswith("3.0"), "wrong schema")
    _require(
        config["package_id"] == "open-gravity-same-law-matter-photon-closures-v3",
        "wrong package id",
    )
    _require(
        config["linear_model"]["id"] == "GR_PLUS_FP_PPN_YUKAWA_LINEAR_FIXTURE",
        "linear model drift",
    )
    _require(
        config["linear_model"]["not_established"],
        "linear model claim ceiling disappeared",
    )
    _require(
        tuple(config["same_law_gate"]["forbidden"]) == _FORBIDDEN,
        "forbidden gate drift",
    )
    _require(
        config["same_law_gate"]["all_physical_parameters_shared"] is True,
        "physical parameters split",
    )
    physical = config["physical_parameter_set"]
    expected_parameters = {
        "G",
        "c",
        "source_mass",
        "universal_coupling_g",
        "mediator_inverse_length_mu",
        "tensor_wavenumber_k",
    }
    _require(set(physical) == expected_parameters, "physical parameter set drift")
    for key in expected_parameters:
        _require(float(physical[key]) > 0.0, f"nonpositive physical parameter: {key}")
    serialized_parameters = json.dumps(physical, sort_keys=True).lower()
    for forbidden in (
        "photon",
        "lens",
        "opacity",
        "frequency",
        "path",
        "fermat",
        "dispersion",
    ):
        _require(
            forbidden not in serialized_parameters,
            f"channel-specific physical parameter present: {forbidden}",
        )
    model = config["linear_model"]
    _require(model["coefficient_phi_extra"] == 4.0 / 3.0, "Phi coefficient drift")
    _require(model["coefficient_psi_extra"] == 2.0 / 3.0, "Psi coefficient drift")
    _require(model["fixed_extra_sector_slip"] == 0.5, "slip contract drift")
    source = config["source_contract"]
    _require(source["ESO325_G004"].startswith("SOURCE_BLOCKED_"), "ESO source widened")
    _require(source["SLACS"] == "UNCHANGED_CONFIRMATION_SEALED", "SLACS source widened")
    for key in (
        "raw_scientific_payloads_downloaded_by_builder",
        "scientific_response_rows_opened",
        "scientific_response_rows_scored",
    ):
        _require(source[key] == 0, f"source access boundary changed: {key}")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "receipt path drift")
    _require(
        config["outputs"]["artifact_directory"] == ARTIFACT_DIRECTORY.as_posix(),
        "artifact path drift",
    )
    _require(dimension_audit(config)["all_pass"], "dimension audit failed")


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing binding: {row['role']}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"binding drift: {row['role']}")
        observed[row["role"]] = digest
    prior_path = Path(config["supersedes"]["path"])
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    _require(file_sha256(prior_path) == config["supersedes"]["file_sha256"], "v2 file drift")
    _require(prior["content_sha256"] == config["supersedes"]["content_sha256"], "v2 content drift")
    return observed


def _source_manifests(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source_contract"]
    path = Path(source["metadata_source"])
    _require(file_sha256(path) == source["metadata_source_sha256"], "source metadata drift")
    prior = json.loads(path.read_text(encoding="utf-8"))
    _require(
        prior["eso325_source_manifest"]["status"] == source["ESO325_G004"],
        "ESO metadata status drift",
    )
    _require(
        prior["slacs_holdout_manifest"]["status"] == source["SLACS"],
        "SLACS metadata status drift",
    )
    _require(
        prior["slacs_holdout_manifest"]["confirmation_opened"] is False,
        "SLACS confirmation opened",
    )
    return {
        "ESO325_G004": prior["eso325_source_manifest"],
        "SLACS_HOLDOUT": prior["slacs_holdout_manifest"],
    }


@dataclass(frozen=True)
class FieldState:
    """One immutable first-order weak-field state at a positive radius."""

    r: float
    U: float
    Y: float
    Phi: float
    Psi: float
    g_tt: float
    g_space: float
    U_prime: float
    Y_prime: float
    Phi_prime: float
    Psi_prime: float
    U_second: float
    Y_second: float
    g_tt_prime: float


def field_state(r: float, config: Mapping[str, Any]) -> FieldState:
    """Construct the only state consumed by all reported matter and null observables."""
    _require(r > 0.0, "the distributional point-source origin is excluded")
    p = config["physical_parameter_set"]
    model = config["linear_model"]
    gm = p["G"] * p["source_mass"]
    mu = p["mediator_inverse_length_mu"]
    coupling = p["universal_coupling_g"]
    exponential = math.exp(-mu * r)
    u = -gm / r
    y = -gm * exponential / r
    u_prime = gm / r**2
    y_prime = gm * exponential * (1.0 / r**2 + mu / r)
    u_second = -2.0 * gm / r**3
    y_second = -gm * exponential * (2.0 / r**3 + 2.0 * mu / r**2 + mu**2 / r)
    coefficient_phi = model["coefficient_phi_extra"]
    coefficient_psi = model["coefficient_psi_extra"]
    phi = u + coefficient_phi * coupling * y
    psi = u + coefficient_psi * coupling * y
    phi_prime = u_prime + coefficient_phi * coupling * y_prime
    psi_prime = u_prime + coefficient_psi * coupling * y_prime
    c2 = p["c"] ** 2
    return FieldState(
        r=r,
        U=u,
        Y=y,
        Phi=phi,
        Psi=psi,
        g_tt=-(1.0 + 2.0 * phi / c2),
        g_space=1.0 - 2.0 * psi / c2,
        U_prime=u_prime,
        Y_prime=y_prime,
        Phi_prime=phi_prime,
        Psi_prime=psi_prime,
        U_second=u_second,
        Y_second=y_second,
        g_tt_prime=-2.0 * phi_prime / c2,
    )


def field_equation_and_source_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    p = config["physical_parameter_set"]
    geometry = config["fixture_geometry"]
    mu = p["mediator_inverse_length_mu"]
    gm = p["G"] * p["source_mass"]
    rows = []
    for radius in geometry["dynamics_radii"]:
        state = field_state(radius, config)
        massless = state.U_second + 2.0 * state.U_prime / radius
        massive = state.Y_second + 2.0 * state.Y_prime / radius - mu**2 * state.Y
        massless_scale = abs(state.U_second) + abs(2.0 * state.U_prime / radius)
        massive_scale = (
            abs(state.Y_second) + abs(2.0 * state.Y_prime / radius) + abs(mu**2 * state.Y)
        )
        rows.append(
            {
                "r": radius,
                "laplacian_U": massless,
                "laplacian_minus_mu2_Y": massive,
                "relative_massless_residual": massless / max(massless_scale, 1.0e-300),
                "relative_massive_residual": massive / max(massive_scale, 1.0e-300),
            }
        )
    source_radius = geometry["source_mapping_radius"]
    source_state = field_state(source_radius, config)
    x = mu * source_radius
    massless_enclosed = source_radius**2 * source_state.U_prime / gm
    massive_surface = source_radius**2 * source_state.Y_prime / gm
    massive_volume_correction = 1.0 - math.exp(-x) * (1.0 + x)
    massive_enclosed = massive_surface + massive_volume_correction
    return {
        "equations": config["linear_model"]["source_equations"],
        "exterior_rows": rows,
        "distributional_source_mapping": {
            "r": source_radius,
            "massless_flux_over_GM": massless_enclosed,
            "massive_surface_term_over_GM": massive_surface,
            "massive_volume_correction_over_GM": massive_volume_correction,
            "massive_total_over_GM": massive_enclosed,
            "target": 1.0,
        },
        "full_fierz_pauli_tensor_constraints_claimed": False,
        "scope": config["linear_model"]["scope"],
    }


def timelike_acceleration(r: float, config: Mapping[str, Any]) -> float:
    """First-order slow-particle radial geodesic acceleration from the shared metric."""
    state = field_state(r, config)
    c = config["physical_parameter_set"]["c"]
    return 0.5 * c**2 * state.g_tt_prime


def radial_null_coordinate_characteristic(r: float, config: Mapping[str, Any]) -> float:
    """First-order radial coordinate characteristic of the shared metric."""
    state = field_state(r, config)
    c = config["physical_parameter_set"]["c"]
    return c * (1.0 + (state.Phi + state.Psi) / c**2)


def _z_breakpoints(impact: float, limit: float) -> list[float]:
    magnitude = abs(impact)
    points = {-limit, 0.0, limit}
    for multiplier in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        value = min(limit, multiplier * magnitude)
        points.add(value)
        points.add(-value)
    return sorted(points)


def _adaptive_integral(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    config: Mapping[str, Any],
    points: Sequence[float] = (),
) -> tuple[float, float]:
    controls = config["numerical_controls"]
    interior = [point for point in points if lower < point < upper]
    value, error = quad(
        function,
        lower,
        upper,
        epsabs=controls["adaptive_absolute_tolerance"],
        epsrel=controls["adaptive_relative_tolerance"],
        limit=controls["adaptive_subdivision_limit"],
        points=interior or None,
    )
    return float(value), float(error)


@lru_cache(maxsize=8)
def _gauss_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _composite_gauss_legendre(
    function: Callable[[float], float], breakpoints: Sequence[float], order: int
) -> float:
    nodes, weights = _gauss_nodes_weights(order)
    total = 0.0
    for lower, upper in pairwise(breakpoints):
        if upper <= lower:
            continue
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        values = np.fromiter(
            (function(midpoint + half_width * node) for node in nodes),
            dtype=float,
            count=order,
        )
        total += half_width * float(np.dot(weights, values))
    return total


def _deflection_integrand(z: float, impact: float, config: Mapping[str, Any]) -> float:
    radius = math.hypot(impact, z)
    state = field_state(radius, config)
    return (state.Phi_prime + state.Psi_prime) * impact / radius


def _adaptive_deflection_with_error(
    impact: float, config: Mapping[str, Any]
) -> tuple[float, float]:
    _require(impact != 0.0, "point-source impact cannot be zero")
    geometry = config["fixture_geometry"]
    c = config["physical_parameter_set"]["c"]
    sign = math.copysign(1.0, impact)
    magnitude = abs(impact)
    limit = geometry["line_of_sight_limit"]
    points = _z_breakpoints(magnitude, limit)
    value, error = _adaptive_integral(
        lambda z: _deflection_integrand(z, magnitude, config),
        -limit,
        limit,
        config,
        points,
    )
    return sign * value / c**2, error / c**2


def deflection(impact: float, config: Mapping[str, Any]) -> float:
    """Born deflection from Phi'+Psi' of the one shared state; no coupling override exists."""
    return _adaptive_deflection_with_error(impact, config)[0]


def _independent_deflection(impact: float, config: Mapping[str, Any]) -> float:
    geometry = config["fixture_geometry"]
    controls = config["numerical_controls"]
    c = config["physical_parameter_set"]["c"]
    sign = math.copysign(1.0, impact)
    magnitude = abs(impact)
    limit = geometry["line_of_sight_limit"]
    value = _composite_gauss_legendre(
        lambda z: _deflection_integrand(z, magnitude, config),
        _z_breakpoints(magnitude, limit),
        controls["independent_gauss_order"],
    )
    return sign * value / c**2


def _shapiro_integrand(z: float, impact: float, config: Mapping[str, Any]) -> float:
    state = field_state(math.hypot(impact, z), config)
    c = config["physical_parameter_set"]["c"]
    return -(state.Phi + state.Psi) / c**3


def _adaptive_shapiro_with_error(impact: float, config: Mapping[str, Any]) -> tuple[float, float]:
    _require(impact != 0.0, "point-source impact cannot be zero")
    geometry = config["fixture_geometry"]
    magnitude = abs(impact)
    limit = geometry["line_of_sight_limit"]
    return _adaptive_integral(
        lambda z: _shapiro_integrand(z, magnitude, config),
        -limit,
        limit,
        config,
        _z_breakpoints(magnitude, limit),
    )


def shapiro_delay(impact: float, config: Mapping[str, Any]) -> float:
    return _adaptive_shapiro_with_error(impact, config)[0]


def _independent_shapiro(impact: float, config: Mapping[str, Any]) -> float:
    geometry = config["fixture_geometry"]
    controls = config["numerical_controls"]
    magnitude = abs(impact)
    limit = geometry["line_of_sight_limit"]
    return _composite_gauss_legendre(
        lambda z: _shapiro_integrand(z, magnitude, config),
        _z_breakpoints(magnitude, limit),
        controls["independent_gauss_order"],
    )


def endpoint_frequency_ratio(config: Mapping[str, Any]) -> float:
    """First-order static endpoint redshift nu_observer/nu_emitter."""
    geometry = config["fixture_geometry"]
    c = config["physical_parameter_set"]["c"]
    emitter = field_state(geometry["redshift_emitter_radius"], config)
    observer = field_state(geometry["redshift_observer_radius"], config)
    return 1.0 + (emitter.Phi - observer.Phi) / c**2


def _reduced_deflection(
    theta: float, config: Mapping[str, Any], deflection_function: Callable[[float], float]
) -> float:
    geometry = config["fixture_geometry"]
    impact = geometry["lens_distance"] * theta
    ratio = geometry["lens_source_distance"] / geometry["source_distance"]
    return ratio * deflection_function(impact)


def _theta_breakpoints(reference: float, magnitude: float) -> list[float]:
    _require(magnitude > reference > 0.0, "invalid lensing-potential interval")
    points = {reference, magnitude}
    value = reference
    while value < magnitude:
        points.add(min(magnitude, value))
        value *= 2.0
    return sorted(points)


def _image_delay_with_method(config: Mapping[str, Any], method: str) -> dict[str, float | str]:
    geometry = config["fixture_geometry"]
    controls = config["numerical_controls"]
    beta = geometry["source_angle_beta"]
    if method == "adaptive_gauss_kronrod":
        alpha = lambda impact: deflection(impact, config)
    elif method == "independent_composite_gauss_legendre":
        alpha = lambda impact: _independent_deflection(impact, config)
    else:
        raise SameLawV3Error(f"unknown image-delay method: {method}")

    def reduced(theta: float) -> float:
        return _reduced_deflection(theta, config, alpha)

    def lens_equation(theta: float) -> float:
        return theta - reduced(theta) - beta

    negative = brentq(
        lens_equation,
        *geometry["negative_image_bracket"],
        xtol=1.0e-14,
        rtol=1.0e-14,
    )
    positive = brentq(
        lens_equation,
        *geometry["positive_image_bracket"],
        xtol=1.0e-14,
        rtol=1.0e-14,
    )
    reference = geometry["lensing_potential_reference_angle"]

    def lensing_potential(theta: float) -> float:
        magnitude = abs(theta)
        points = _theta_breakpoints(reference, magnitude)
        if method == "adaptive_gauss_kronrod":
            return _adaptive_integral(reduced, reference, magnitude, config, points)[0]
        return _composite_gauss_legendre(
            reduced,
            points,
            controls["independent_theta_gauss_order"],
        )

    def fermat(theta: float) -> float:
        return 0.5 * (theta - beta) ** 2 - lensing_potential(theta)

    negative_fermat = fermat(negative)
    positive_fermat = fermat(positive)
    distance = (
        geometry["lens_distance"] * geometry["source_distance"] / geometry["lens_source_distance"]
    )
    c = config["physical_parameter_set"]["c"]
    signed_delay = (
        (1.0 + geometry["lens_redshift"]) * distance * (positive_fermat - negative_fermat) / c
    )
    return {
        "method": method,
        "negative_image": negative,
        "positive_image": positive,
        "negative_fermat": negative_fermat,
        "positive_fermat": positive_fermat,
        "signed_delay": signed_delay,
        "lens_redshift_factor": 1.0 + geometry["lens_redshift"],
    }


def image_delay_fixture(config: Mapping[str, Any]) -> dict[str, float | str]:
    """Local thin-lens Fermat fixture with no independent delay normalization."""
    result = _image_delay_with_method(config, "adaptive_gauss_kronrod")
    result["scope"] = "local_linear_thin_lens_fixture_not_a_cosmological_time_delay_fit"
    return result


def tensor_dispersion(config: Mapping[str, Any]) -> dict[str, Any]:
    p = config["physical_parameter_set"]
    c = p["c"]
    k = p["tensor_wavenumber_k"]
    mu = p["mediator_inverse_length_mu"]
    massive_omega = c * math.sqrt(k**2 + mu**2)
    return {
        "massless_GR_branch": {
            "omega": c * k,
            "characteristic_speed_over_c": 1.0,
            "group_speed_over_c": 1.0,
        },
        "massive_TT_comparator_branch": {
            "omega": massive_omega,
            "characteristic_speed_over_c": 1.0,
            "group_speed_over_c": k / math.sqrt(k**2 + mu**2),
            "same_mediator_inverse_length_mu": mu,
        },
        "scope": config["linear_model"]["tensor_scope"],
    }


def _finite_difference_metric_acceleration(r: float, config: Mapping[str, Any]) -> float:
    step = config["numerical_controls"]["finite_difference_step"]
    c = config["physical_parameter_set"]["c"]
    derivative = (field_state(r + step, config).g_tt - field_state(r - step, config).g_tt) / (
        2.0 * step
    )
    return 0.5 * c**2 * derivative


def _independent_lensing_component(
    impact: float, config: Mapping[str, Any], component: str
) -> float:
    geometry = config["fixture_geometry"]
    controls = config["numerical_controls"]
    c = config["physical_parameter_set"]["c"]
    magnitude = abs(impact)
    limit = geometry["line_of_sight_limit"]

    def integrand(z: float) -> float:
        radius = math.hypot(magnitude, z)
        state = field_state(radius, config)
        if component == "massless_sum":
            radial_gradient = 2.0 * state.U_prime
        elif component == "unit_yukawa_sum_basis":
            radial_gradient = state.Y_prime
        else:
            raise SameLawV3Error(f"unknown lensing component: {component}")
        return radial_gradient * magnitude / radius / c**2

    return _composite_gauss_legendre(
        integrand,
        _z_breakpoints(magnitude, limit),
        controls["independent_gauss_order"],
    )


def two_route_slip_internal_check(config: Mapping[str, Any]) -> dict[str, Any]:
    """Internal recovery from separate metric and ray computations, not empirical evidence."""
    geometry = config["fixture_geometry"]
    dynamic_numerator = 0.0
    dynamic_denominator = 0.0
    dynamic_rows = []
    for radius in geometry["dynamics_radii"]:
        observed_phi_prime = -_finite_difference_metric_acceleration(radius, config)
        state = field_state(radius, config)
        response = observed_phi_prime - state.U_prime
        basis = state.Y_prime
        dynamic_numerator += basis * response
        dynamic_denominator += basis * basis
        dynamic_rows.append(
            {
                "r": radius,
                "observed_phi_prime_from_metric_finite_difference": observed_phi_prime,
                "massless_U_prime": state.U_prime,
                "unit_Y_prime_basis": basis,
            }
        )
    amplitude_phi = dynamic_numerator / dynamic_denominator

    lens_numerator = 0.0
    lens_denominator = 0.0
    lens_rows = []
    for impact in geometry["lensing_impacts"]:
        observed = deflection(impact, config)
        baseline = _independent_lensing_component(impact, config, "massless_sum")
        basis = _independent_lensing_component(impact, config, "unit_yukawa_sum_basis")
        response = observed - baseline
        lens_numerator += basis * response
        lens_denominator += basis * basis
        lens_rows.append(
            {
                "impact": impact,
                "observed_total_deflection_adaptive": observed,
                "massless_baseline_independent_gauss": baseline,
                "unit_yukawa_sum_basis_independent_gauss": basis,
            }
        )
    amplitude_sum = lens_numerator / lens_denominator
    gamma_extra = (amplitude_sum - amplitude_phi) / amplitude_phi
    return {
        "scope": "internal_state_to_observable_consistency_check_not_independent_data_evidence",
        "dynamics_generator": "finite_difference_of_shared_metric_g_tt",
        "lensing_generator": "adaptive_Gauss_Kronrod_of_shared_Phi_prime_plus_Psi_prime",
        "lensing_inference_basis": "independent_composite_Gauss_Legendre_components",
        "dynamics_rows": dynamic_rows,
        "lensing_rows": lens_rows,
        "fitted_phi_extra_amplitude": amplitude_phi,
        "fitted_phi_plus_psi_extra_amplitude": amplitude_sum,
        "reconstructed_gamma_extra": gamma_extra,
        "expected_gamma_extra": config["linear_model"]["fixed_extra_sector_slip"],
    }


def optical_scope(config: Mapping[str, Any]) -> dict[str, Any]:
    geometry = config["fixture_geometry"]
    parity_rows = []
    for impact in geometry["lensing_impacts"]:
        parity_rows.append(
            {
                "impact": impact,
                "deflection_odd_parity_error": deflection(impact, config)
                + deflection(-impact, config),
                "shapiro_even_parity_error": shapiro_delay(impact, config)
                - shapiro_delay(-impact, config),
            }
        )
    return {
        "derived_within_linear_geometric_optics": [
            "radial coordinate null characteristic of the shared metric",
            "Born bending from Phi_prime+Psi_prime",
            "Shapiro delay from Phi+Psi",
            "local static endpoint redshift",
            "local thin-lens Fermat delay with an explicit lens-redshift factor",
        ],
        "structural_assumption_not_a_frequency_test": "Minimal metric geometric-optics propagation contains no photon frequency argument or coefficient.",
        "spherical_parity_check_not_reciprocity": parity_rows,
        "not_evaluated": config["same_law_gate"]["optical_claims_not_made"],
        "distance_duality_eta": None,
        "photon_number_survival_fraction": None,
        "source_observer_Jacobi_reciprocity": "NOT_EVALUATED",
    }


def derived_linear_observables(config: Mapping[str, Any]) -> dict[str, Any]:
    geometry = config["fixture_geometry"]
    radius = geometry["dynamics_radii"][1]
    impact = geometry["lensing_impacts"][2]
    return {
        "field_state": asdict(field_state(radius, config)),
        "slow_timelike_radial_acceleration": timelike_acceleration(radius, config),
        "radial_null_coordinate_characteristic": radial_null_coordinate_characteristic(
            radius, config
        ),
        "Born_deflection": deflection(impact, config),
        "Shapiro_delay": shapiro_delay(impact, config),
        "static_endpoint_frequency_ratio": endpoint_frequency_ratio(config),
        "local_thin_lens_Fermat_delay": image_delay_fixture(config),
        "tensor_dispersion": tensor_dispersion(config),
        "perturbative_order": "first_order_weak_field",
        "photon_only_physical_parameters": 0,
    }


def numerical_convergence(config: Mapping[str, Any]) -> dict[str, Any]:
    geometry = config["fixture_geometry"]
    rows = []
    for impact in geometry["lensing_impacts"]:
        adaptive_deflection, adaptive_deflection_error = _adaptive_deflection_with_error(
            impact, config
        )
        independent_deflection = _independent_deflection(impact, config)
        adaptive_shapiro, adaptive_shapiro_error = _adaptive_shapiro_with_error(impact, config)
        independent_shapiro = _independent_shapiro(impact, config)
        rows.append(
            {
                "impact": impact,
                "adaptive_deflection": adaptive_deflection,
                "adaptive_deflection_estimated_absolute_error": adaptive_deflection_error,
                "independent_deflection": independent_deflection,
                "relative_deflection_disagreement": abs(
                    adaptive_deflection - independent_deflection
                )
                / max(abs(independent_deflection), 1.0e-300),
                "adaptive_shapiro": adaptive_shapiro,
                "adaptive_shapiro_estimated_absolute_error": adaptive_shapiro_error,
                "independent_shapiro": independent_shapiro,
                "relative_shapiro_disagreement": abs(adaptive_shapiro - independent_shapiro)
                / max(abs(independent_shapiro), 1.0e-300),
            }
        )
    adaptive_image = _image_delay_with_method(config, "adaptive_gauss_kronrod")
    independent_image = _image_delay_with_method(config, "independent_composite_gauss_legendre")
    adaptive_delay = float(adaptive_image["signed_delay"])
    independent_delay = float(independent_image["signed_delay"])
    routes = two_route_slip_internal_check(config)
    return {
        "methods": {
            "production": "adaptive_Gauss_Kronrod_QUADPACK_with_impact_scale_breakpoints",
            "reference": "independent_composite_Gauss_Legendre_on_impact_scale_panels",
        },
        "all_frozen_impacts": rows,
        "image_delay": {
            "adaptive": adaptive_image,
            "independent": independent_image,
            "relative_disagreement": abs(adaptive_delay - independent_delay)
            / max(abs(independent_delay), 1.0e-300),
        },
        "slip_absolute_error": abs(
            routes["reconstructed_gamma_extra"] - routes["expected_gamma_extra"]
        ),
    }


def _blocked_v2_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preserved_path": config["supersedes"]["path"],
        "preserved_file_sha256": config["supersedes"]["file_sha256"],
        "preserved_content_sha256": config["supersedes"]["content_sha256"],
        "audit_status": config["supersedes"]["audit_status"],
        "material_blockers": [
            "The single Fierz-Pauli action did not generate the simultaneous massless U and massive Y state or verify the full tensor constraints.",
            "No physical dimension ledger connected the action, parameters, fields, and observables.",
            "The lens route bypassed field_state and hard-coded the desired Phi+Psi Yukawa coefficient.",
            "The frozen n=800 ray grid erred by 10.043 percent and 1.571 percent at the two smallest registered impacts.",
            "Impact parity, algebraic distance-duality identities, and duplicated deflections were mislabeled as reciprocity, photon transport, and chromaticity tests.",
        ],
        "v3_response": "Narrow scope; dimension ledger; one shared state; no lens coupling override; adaptive rays with independent every-impact cross-check; honest optical non-claims.",
    }


def build_artifacts(config: Mapping[str, Any]) -> dict[str, bytes]:
    equations = field_equation_and_source_audit(config)
    dimensions = dimension_audit(config)
    observables = derived_linear_observables(config)
    routes = two_route_slip_internal_check(config)
    convergence = numerical_convergence(config)
    optics = optical_scope(config)
    source_manifests = _source_manifests(config)
    report = (
        b"# Lane-7 same-state linear closure v3\n\n"
        b"PASS only for a narrow, dimensioned linear Green-function/PPN fixture. V2 remains preserved as blocked evidence. V3 does not call the isotropic U+Y state a complete Fierz-Pauli tensor solution and does not claim a nonlinear, screened, ghost-free, or cosmological theory.\n\n"
        b"All reported slow-timelike and geometric-optics null quantities consume one immutable field state. The lens API has no coupling override. Adaptive Gauss-Kronrod rays are cross-checked at every frozen impact by a separate composite Gauss-Legendre implementation; the slip recovery is explicitly an internal consistency check rather than independent evidence.\n\n"
        b"Impact-sign parity is reported only as spherical parity. Source-observer Jacobi reciprocity, Etherington distance duality, photon-number/flux transport, and wave-optics chromaticity are not evaluated.\n\n"
        b"ESO 325-G004 HST/MUSE metadata and the unchanged 57/45/12 SLACS split remain bound, but no payload or scientific response row is opened. The next real-data blocker is acquisition and SHA256 sealing of every exact HST/MUSE payload, followed by registered masks, PSFs, Voronoi-bin covariance, object intersection, and cosmology.\n"
    )
    return {
        "blocked-v2-audit.json": _json_bytes(_blocked_v2_audit(config)),
        "dimension-audit.json": _json_bytes(dimensions),
        "field-equation-and-source-audit.json": _json_bytes(equations),
        "derived-linear-observables.json": _json_bytes(observables),
        "two-route-slip-internal-check.json": _json_bytes(routes),
        "numerical-convergence.json": _json_bytes(convergence),
        "optical-scope.json": _json_bytes(optics),
        "source-and-likelihood-manifests.json": _json_bytes(source_manifests),
        "report.md": report,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_bindings(config)
    dimensions = dimension_audit(config)
    equations = field_equation_and_source_audit(config)
    observables = derived_linear_observables(config)
    routes = two_route_slip_internal_check(config)
    convergence = numerical_convergence(config)
    optics = optical_scope(config)
    artifacts = build_artifacts(config)
    controls = config["numerical_controls"]
    max_deflection_disagreement = max(
        row["relative_deflection_disagreement"] for row in convergence["all_frozen_impacts"]
    )
    max_shapiro_disagreement = max(
        row["relative_shapiro_disagreement"] for row in convergence["all_frozen_impacts"]
    )
    max_field_residual = max(
        max(abs(row["relative_massless_residual"]), abs(row["relative_massive_residual"]))
        for row in equations["exterior_rows"]
    )
    source_mapping = equations["distributional_source_mapping"]
    internal_pass = (
        dimensions["all_pass"]
        and max_field_residual < 1.0e-14
        and abs(source_mapping["massless_flux_over_GM"] - 1.0) < 1.0e-14
        and abs(source_mapping["massive_total_over_GM"] - 1.0) < 1.0e-14
        and max_deflection_disagreement
        < controls["maximum_relative_deflection_disagreement_each_impact"]
        and max_shapiro_disagreement < controls["maximum_relative_shapiro_disagreement_each_impact"]
        and convergence["image_delay"]["relative_disagreement"]
        < controls["maximum_relative_image_delay_disagreement"]
        and convergence["slip_absolute_error"] < controls["maximum_absolute_slip_error"]
        and observables["photon_only_physical_parameters"] == 0
    )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_NARROW_LINEAR_FIXTURE_BLOCK_REAL_SOURCES",
        "decision": "KEEP_ESO_AND_SLACS_RESPONSES_SEALED_PENDING_REAUDIT_AND_SOURCE_COMPLETION",
        "v2_preservation": _blocked_v2_audit(config),
        "bindings": bindings,
        "linear_model": config["linear_model"],
        "unit_system": config["unit_system"],
        "one_physical_parameter_set_sha256": content_sha256(config["physical_parameter_set"]),
        "internal_same_state_pass": internal_pass,
        "photon_only_physical_parameters": 0,
        "dimension_audit": dimensions,
        "field_equation_and_source_audit": equations,
        "derived_linear_observables": observables,
        "two_route_slip_internal_check": routes,
        "numerical_convergence": convergence,
        "optical_scope": optics,
        "source_status": {
            "ESO325_G004": config["source_contract"]["ESO325_G004"],
            "SLACS": config["source_contract"]["SLACS"],
            "scientific_response_rows_opened": 0,
        },
        "next_real_data_blocker": config["source_contract"]["remaining_before_response_access"][0],
        "remaining_real_data_blockers": config["source_contract"][
            "remaining_before_response_access"
        ],
        "claim_boundary": config["claim_boundary"],
        "access_accounting": {
            "raw_scientific_payloads_downloaded_by_builder": 0,
            "scientific_response_rows_opened": 0,
            "scientific_response_rows_scored": 0,
            "network_calls_by_builder": 0,
            "paid_calls": 0,
            "tuning_calls": 0,
        },
        "artifact_manifest": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in sorted(artifacts.items())
        },
        "artifact_bindings": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": file_sha256(CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(TEST_PATH)},
        },
    }
    _require(internal_pass, "narrow internal closure failed")
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing artifact differs: {path.as_posix()}")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        _require(path.read_bytes() == payload, f"concurrent artifact differs: {path.as_posix()}")
        return "EXISTING_IDENTICAL"
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_packet() -> str:
    config = load_config()
    statuses = [
        _atomic_no_clobber(ARTIFACT_DIRECTORY / name, payload)
        for name, payload in build_artifacts(config).items()
    ]
    statuses.append(_atomic_no_clobber(OUTPUT_PATH, _json_bytes(build_receipt())))
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def validate_receipt() -> None:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(observed == build_receipt(), "receipt differs from deterministic rebuild")
    for name, payload in build_artifacts(load_config()).items():
        _require((ARTIFACT_DIRECTORY / name).read_bytes() == payload, f"artifact drift: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "status"))
    arguments = parser.parse_args(argv)
    if arguments.action == "build":
        print(write_packet())
    elif arguments.action == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(receipt["status"])
        print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
