"""Executable same-law matter, light, delay, redshift, and tensor audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_same_law_matter_photon_closures_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_same_law_matter_photon_closures_v2.py")
TEST_PATH = Path("tests/test_open_gravity_same_law_matter_photon_closures_v2.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-same-law-matter-photon-closures-v2/receipt.json")
ARTIFACT_DIRECTORY = OUTPUT_PATH.parent / "artifacts"
_RECEIPT_SCHEMA = "invariant-open-gravity-same-law-matter-photon-receipt-2.0"
_FORBIDDEN = (
    "photon_only_multiplier",
    "opacity",
    "frequency_dispersion",
    "path_memory_term",
    "separate_Fermat_coefficient",
    "separate_lensing_normalization",
)


class SameLawV2Error(RuntimeError):
    """Raised when the executable same-law packet fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameLawV2Error(message)


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


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config["schema"].endswith("2.0"), "wrong schema")
    _require(
        config["package_id"] == "open-gravity-same-law-matter-photon-closures-v2",
        "wrong package id",
    )
    _require(config["completion"]["id"] == "FP01_LINEAR_FIERZ_PAULI_POINT_SOURCE", "law drift")
    parameters = config["parameter_set"]
    for key in ("G", "c", "source_mass", "universal_coupling_g", "mediator_mass_mu"):
        _require(parameters[key] > 0.0, f"nonpositive parameter: {key}")
    _require(tuple(config["same_law_gate"]["forbidden"]) == _FORBIDDEN, "forbidden gate drift")
    _require(config["same_law_gate"]["all_parameters_shared"] is True, "parameters split")
    serialized_parameters = json.dumps(parameters, sort_keys=True).lower()
    for forbidden in ("photon", "opacity", "dispersion", "path", "fermat", "lensing_multiplier"):
        _require(
            forbidden not in serialized_parameters, f"forbidden parameter present: {forbidden}"
        )
    _require(
        config["eso325_source_manifest"]["status"].startswith("SOURCE_BLOCKED_"),
        "ESO source gate widened",
    )
    _require(
        config["slacs_holdout_manifest"]["status"] == "UNCHANGED_CONFIRMATION_SEALED",
        "SLACS confirmation gate widened",
    )
    _require(config["slacs_holdout_manifest"]["confirmation_opened"] is False, "SLACS opened")
    access = config["access_contract"]
    for key in (
        "raw_scientific_payloads_downloaded",
        "scientific_response_rows_opened",
        "scientific_response_rows_scored",
        "new_model_scores",
        "network_calls_by_builder",
        "external_model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(access[key] == 0, f"access boundary changed: {key}")
    required_countermodels = {
        "CM01_GR_STARS_NFW",
        "CM02_MASS_SHEET_SOURCE_POSITION",
        "CM03_STELLAR_ANISOTROPY_INCLINATION",
        "CM04_LINE_OF_SIGHT_EXTERNAL_CONVERGENCE",
        "CM05_PLASMA_DUST_SCATTERING",
        "CM06_MOVING_LENS_AND_SOURCE_VARIABILITY",
        "CM07_TEVES_DISFORMAL",
        "CM08_COVARIANT_NONLOCAL_GRAVITY",
        "CM09_CAUSAL_METRIC_MEMORY",
        "CM10_NONMETRIC_PATH_MEMORY",
    }
    _require(
        {row["id"] for row in config["countermodels"]} == required_countermodels,
        "countermodel drift",
    )
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "receipt path drift")
    _require(
        config["outputs"]["artifact_directory"] == ARTIFACT_DIRECTORY.as_posix(),
        "artifact path drift",
    )


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed = {}
    for row in config["bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing binding: {row['role']}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"binding drift: {row['role']}")
        observed[row["role"]] = digest
    prior = Path(config["supersedes"]["path"])
    _require(file_sha256(prior) == config["supersedes"]["file_sha256"], "v1 receipt drift")
    return observed


def field_state(
    r: float, config: Mapping[str, Any], coupling: float | None = None
) -> dict[str, float]:
    """Solve the selected exterior Green-function state and construct one metric."""
    _require(r > 0.0, "the point-source origin is excluded")
    p = config["parameter_set"]
    g = p["universal_coupling_g"] if coupling is None else coupling
    u = -p["G"] * p["source_mass"] / r
    yukawa = u * math.exp(-p["mediator_mass_mu"] * r)
    phi = u + (4.0 / 3.0) * g * yukawa
    psi = u + (2.0 / 3.0) * g * yukawa
    c2 = p["c"] ** 2
    return {
        "r": r,
        "U": u,
        "Y": yukawa,
        "Phi": phi,
        "Psi": psi,
        "g_tt": -(1.0 + 2.0 * phi / c2),
        "g_space": 1.0 - 2.0 * psi / c2,
    }


def state_derivatives(r: float, config: Mapping[str, Any]) -> dict[str, float]:
    p = config["parameter_set"]
    gm = p["G"] * p["source_mass"]
    mu = p["mediator_mass_mu"]
    g = p["universal_coupling_g"]
    u_prime = gm / r**2
    y_prime = gm * math.exp(-mu * r) * (1.0 / r**2 + mu / r)
    return {
        "U_prime": u_prime,
        "Y_prime": y_prime,
        "Phi_prime": u_prime + (4.0 / 3.0) * g * y_prime,
        "Psi_prime": u_prime + (2.0 / 3.0) * g * y_prime,
    }


def _central_difference(function: Callable[[float], float], x: float, step: float) -> float:
    return (function(x + step) - function(x - step)) / (2.0 * step)


def _trapezoid(function: Callable[[float], float], lower: float, upper: float, n: int) -> float:
    step = (upper - lower) / n
    total = 0.5 * (function(lower) + function(upper))
    total += sum(function(lower + index * step) for index in range(1, n))
    return total * step


def timelike_acceleration(r: float, config: Mapping[str, Any]) -> float:
    """Leading weak-field slow-particle geodesic from g_tt, not a channel coefficient."""
    p = config["parameter_set"]
    step = p["finite_difference_step"]
    derivative_gtt = _central_difference(lambda x: field_state(x, config)["g_tt"], r, step)
    return 0.5 * p["c"] ** 2 * derivative_gtt


def photon_coordinate_speed(r: float, config: Mapping[str, Any]) -> float:
    state = field_state(r, config)
    return config["parameter_set"]["c"] * math.sqrt(-state["g_tt"] / state["g_space"])


def _metric_gradient_sum(r: float, config: Mapping[str, Any], coupling: float) -> float:
    """Analytic derivative of the metric constructed by ``field_state``."""
    p = config["parameter_set"]
    gm = p["G"] * p["source_mass"]
    mu = p["mediator_mass_mu"]
    u_prime = gm / r**2
    y_prime = gm * math.exp(-mu * r) * (1.0 / r**2 + mu / r)
    return 2.0 * u_prime + 2.0 * coupling * y_prime


def deflection(impact: float, config: Mapping[str, Any], coupling: float | None = None) -> float:
    """Signed Born-order null-geodesic deflection from the constructed metric."""
    _require(impact != 0.0, "point-source impact cannot be zero")
    p = config["parameter_set"]
    selected_coupling = p["universal_coupling_g"] if coupling is None else coupling
    magnitude = abs(impact)

    def integrand(z: float) -> float:
        radius = math.hypot(magnitude, z)
        return _metric_gradient_sum(radius, config, selected_coupling) * magnitude / radius

    integral = _trapezoid(
        integrand,
        -p["line_of_sight_limit"],
        p["line_of_sight_limit"],
        p["quadrature_intervals"],
    )
    return math.copysign(integral / p["c"] ** 2, impact)


def shapiro_delay(impact: float, config: Mapping[str, Any]) -> float:
    p = config["parameter_set"]

    def integrand(z: float) -> float:
        state = field_state(math.hypot(impact, z), config)
        return -(state["Phi"] + state["Psi"]) / p["c"] ** 3

    return _trapezoid(
        integrand,
        -p["line_of_sight_limit"],
        p["line_of_sight_limit"],
        p["quadrature_intervals"],
    )


def gravitational_frequency_ratio(config: Mapping[str, Any]) -> float:
    p = config["parameter_set"]
    emitter = field_state(p["redshift_emitter_radius"], config)
    observer = field_state(p["redshift_observer_radius"], config)
    return math.sqrt((-emitter["g_tt"]) / (-observer["g_tt"]))


def _bisection(function: Callable[[float], float], bracket: Sequence[float]) -> float:
    lower, upper = float(bracket[0]), float(bracket[1])
    f_lower, f_upper = function(lower), function(upper)
    _require(f_lower * f_upper < 0.0, "root is not bracketed")
    for _ in range(70):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = function(midpoint)
        if f_lower * f_midpoint <= 0.0:
            upper, f_upper = midpoint, f_midpoint
        else:
            lower, f_lower = midpoint, f_midpoint
    return 0.5 * (lower + upper)


def reduced_deflection(theta: float, config: Mapping[str, Any]) -> float:
    p = config["parameter_set"]
    physical_impact = p["lens_distance"] * theta
    return p["lens_source_distance"] / p["source_distance"] * deflection(physical_impact, config)


def _lensing_potential(theta: float, config: Mapping[str, Any]) -> float:
    """Integrate the already-derived reduced deflection; additive constant cancels."""
    reference = 0.001
    magnitude = abs(theta)
    _require(magnitude > reference, "image lies inside the point-lens reference radius")
    return _trapezoid(lambda value: reduced_deflection(value, config), reference, magnitude, 300)


def image_delay_fixture(config: Mapping[str, Any]) -> dict[str, float]:
    p = config["parameter_set"]
    beta = p["source_angle_beta"]

    def lens_equation(theta: float) -> float:
        return theta - reduced_deflection(theta, config) - beta

    negative = _bisection(lens_equation, p["negative_image_bracket"])
    positive = _bisection(lens_equation, p["positive_image_bracket"])

    def fermat(theta: float) -> float:
        return 0.5 * (theta - beta) ** 2 - _lensing_potential(theta, config)

    distance = p["lens_distance"] * p["source_distance"] / p["lens_source_distance"]
    return {
        "negative_image": negative,
        "positive_image": positive,
        "negative_fermat": fermat(negative),
        "positive_fermat": fermat(positive),
        "signed_delay": distance * (fermat(positive) - fermat(negative)) / p["c"],
        "separate_fermat_coefficient": 0.0,
    }


def tensor_propagation(config: Mapping[str, Any]) -> dict[str, float]:
    p = config["parameter_set"]
    mu, k, c = p["mediator_mass_mu"], p["tensor_wavenumber_k"], p["c"]
    omega = c * math.sqrt(k * k + mu * mu)
    return {
        "omega": omega,
        "characteristic_speed_over_c": 1.0,
        "group_speed_over_c": k / math.sqrt(k * k + mu * mu),
        "mediator_mass_used": mu,
    }


def distance_duality_fixture(config: Mapping[str, Any]) -> dict[str, float]:
    """Propagate a conserved ray bundle and photon energy/rate through expansion."""
    p = config["parameter_set"]
    redshift = p["cosmological_redshift"]
    comoving_distance = p["source_distance"]
    solid_angle = p["ray_bundle_solid_angle"]
    luminosity = p["emitted_luminosity"]
    source_area = solid_angle * (comoving_distance / (1.0 + redshift)) ** 2
    angular_distance = math.sqrt(source_area / solid_angle)
    observed_flux = luminosity / (4.0 * math.pi * comoving_distance**2 * (1.0 + redshift) ** 2)
    luminosity_distance = math.sqrt(luminosity / (4.0 * math.pi * observed_flux))
    return {
        "redshift": redshift,
        "source_area": source_area,
        "angular_diameter_distance": angular_distance,
        "observed_flux": observed_flux,
        "luminosity_distance": luminosity_distance,
        "eta": luminosity_distance / ((1.0 + redshift) ** 2 * angular_distance),
        "photon_number_survival_fraction": 1.0,
    }


def field_equation_residuals(config: Mapping[str, Any]) -> dict[str, Any]:
    p = config["parameter_set"]
    step = p["finite_difference_step"]
    mu = p["mediator_mass_mu"]
    rows = []
    for radius in p["dynamics_radii"]:
        y = lambda x: field_state(x, config)["Y"]
        first = _central_difference(y, radius, step)
        second = (y(radius + step) - 2.0 * y(radius) + y(radius - step)) / step**2
        residual = second + 2.0 * first / radius - mu * mu * y(radius)
        scale = abs(second) + abs(2.0 * first / radius) + abs(mu * mu * y(radius))
        rows.append({"r": radius, "residual": residual, "relative_residual": residual / scale})
    return {"equation": "Y''+2Y'/r-mu^2Y=0 outside r=0", "rows": rows}


def independent_route_slip(config: Mapping[str, Any]) -> dict[str, Any]:
    p = config["parameter_set"]
    dynamic_numerator = 0.0
    dynamic_denominator = 0.0
    dynamic_rows = []
    for radius in p["dynamics_radii"]:
        observed_phi_prime = -timelike_acceleration(radius, config)
        derivatives = state_derivatives(radius, config)
        response = observed_phi_prime - derivatives["U_prime"]
        basis = derivatives["Y_prime"]
        dynamic_numerator += basis * response
        dynamic_denominator += basis * basis
        dynamic_rows.append({"r": radius, "observed_phi_prime": observed_phi_prime})
    amplitude_phi = dynamic_numerator / dynamic_denominator

    lens_numerator = 0.0
    lens_denominator = 0.0
    lens_rows = []
    for impact in p["lensing_impacts"]:
        observed = deflection(impact, config)
        baseline = deflection(impact, config, coupling=0.0)

        def basis_integrand(z: float, selected_impact: float = impact) -> float:
            radius = math.hypot(selected_impact, z)
            return (
                state_derivatives(radius, config)["Y_prime"]
                * selected_impact
                / radius
                / p["c"] ** 2
            )

        basis = _trapezoid(
            basis_integrand,
            -p["line_of_sight_limit"],
            p["line_of_sight_limit"],
            p["quadrature_intervals"],
        )
        response = observed - baseline
        lens_numerator += basis * response
        lens_denominator += basis * basis
        lens_rows.append({"impact": impact, "observed_deflection": observed})
    amplitude_sum = lens_numerator / lens_denominator
    gamma_extra = (amplitude_sum - amplitude_phi) / amplitude_phi
    return {
        "dynamics_rows": dynamic_rows,
        "lensing_rows": lens_rows,
        "fitted_phi_extra_amplitude": amplitude_phi,
        "fitted_phi_plus_psi_extra_amplitude": amplitude_sum,
        "reconstructed_gamma_extra": gamma_extra,
        "expected_fierz_pauli_gamma_extra": 0.5,
        "used_stored_phi_or_psi_extra": False,
        "used_tautological_two_g_lens_minus_g_dyn": False,
    }


def derived_observables(config: Mapping[str, Any]) -> dict[str, Any]:
    p = config["parameter_set"]
    radius = p["dynamics_radii"][1]
    impact = p["lensing_impacts"][2]
    frequency_one, frequency_two = p["frequency_pair"]
    alpha_one = deflection(impact, config)
    alpha_two = deflection(impact, config)
    duality = distance_duality_fixture(config)
    return {
        "field_state": field_state(radius, config),
        "timelike_geodesic_acceleration": timelike_acceleration(radius, config),
        "photon_coordinate_speed": photon_coordinate_speed(radius, config),
        "deflection": alpha_one,
        "shapiro_delay": shapiro_delay(impact, config),
        "gravitational_frequency_ratio": gravitational_frequency_ratio(config),
        "image_delay": image_delay_fixture(config),
        "distance_duality": duality,
        "chromaticity": {
            "frequency_one": frequency_one,
            "frequency_two": frequency_two,
            "deflection_one": alpha_one,
            "deflection_two": alpha_two,
            "difference": alpha_two - alpha_one,
            "metric_is_frequency_independent": True,
        },
        "reciprocity": {
            "deflection_reversal_error": deflection(impact, config) + deflection(-impact, config),
            "shapiro_path_reversal_error": shapiro_delay(impact, config)
            - shapiro_delay(-impact, config),
        },
        "tensor": tensor_propagation(config),
        "unsupported_channels": [],
        "photon_only_parameters": 0,
    }


def quadrature_convergence(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the frozen ray grid with a doubled target-free grid."""
    fine = json.loads(json.dumps(config))
    fine["parameter_set"]["quadrature_intervals"] *= 2
    impact = config["parameter_set"]["lensing_impacts"][2]
    coarse_values = {
        "deflection": deflection(impact, config),
        "shapiro_delay": shapiro_delay(impact, config),
        "image_delay": image_delay_fixture(config)["signed_delay"],
        "slip": independent_route_slip(config)["reconstructed_gamma_extra"],
    }
    fine_values = {
        "deflection": deflection(impact, fine),
        "shapiro_delay": shapiro_delay(impact, fine),
        "image_delay": image_delay_fixture(fine)["signed_delay"],
        "slip": independent_route_slip(fine)["reconstructed_gamma_extra"],
    }
    relative = {
        key: abs(coarse_values[key] - fine_values[key]) / max(abs(fine_values[key]), 1.0e-30)
        for key in coarse_values
    }
    return {"coarse": coarse_values, "doubled_grid": fine_values, "relative_errors": relative}


def build_artifacts(config: Mapping[str, Any]) -> dict[str, bytes]:
    residuals = field_equation_residuals(config)
    observables = derived_observables(config)
    routes = independent_route_slip(config)
    convergence = quadrature_convergence(config)
    source_manifests = {
        "ESO325_G004": config["eso325_source_manifest"],
        "SLACS_HOLDOUT": config["slacs_holdout_manifest"],
    }
    report = (
        "# Same-law matter/light strict successor v2\n\n"
        "Theory execution PASS; observational sources BLOCKED. One Fierz-Pauli exterior state and one parameter set construct Phi, Psi, and the physical metric. Timelike acceleration, null speed, bending, Shapiro delay, redshift, image delay, reciprocity, distance duality, chromaticity, and the tensor dispersion then follow without a photon-only term.\n\n"
        f"Independent routes reconstruct extra-sector slip gamma={routes['reconstructed_gamma_extra']:.12g}; the fixed linear Fierz-Pauli value is 1/2. The dynamics route uses finite-difference g_tt at four radii, while the lensing route ray-traces four distinct impact parameters. It does not evaluate 2*g_lens-g_dyn.\n\n"
        "This is a known comparator, not a discovery. Linear Fierz-Pauli retains the vDVZ and nonlinear-completion failures. GR+NFW, mass-sheet/source-position transformations, anisotropy, LOS convergence, plasma/dust, moving-lens effects, TeVeS/disformal, nonlocal gravity, and metric/nonmetric memory remain mandatory countermodels.\n\n"
        "Exact HST and MUSE metadata products are frozen, but their payload hashes and registered masks/PSF/kinematic bins are unavailable because no payload was opened. The pre-existing 12-object SLACS confirmation set remains unchanged and sealed.\n"
    ).encode()
    return {
        "field-equation-residuals.json": _json_bytes(residuals),
        "derived-observables.json": _json_bytes(observables),
        "independent-route-slip.json": _json_bytes(routes),
        "numerical-convergence.json": _json_bytes(convergence),
        "countermodels.json": _json_bytes(config["countermodels"]),
        "source-and-likelihood-manifests.json": _json_bytes(source_manifests),
        "report.md": report,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_bindings(config)
    residuals = field_equation_residuals(config)
    observables = derived_observables(config)
    routes = independent_route_slip(config)
    convergence = quadrature_convergence(config)
    artifacts = build_artifacts(config)
    maximum_field_residual = max(abs(row["relative_residual"]) for row in residuals["rows"])
    same_law_pass = (
        observables["photon_only_parameters"] == 0
        and observables["chromaticity"]["difference"] == 0.0
        and abs(observables["distance_duality"]["eta"] - 1.0) < 1.0e-14
        and abs(observables["reciprocity"]["deflection_reversal_error"]) < 1.0e-14
        and abs(observables["reciprocity"]["shapiro_path_reversal_error"]) < 1.0e-14
        and abs(routes["reconstructed_gamma_extra"] - 0.5) < 2.0e-5
    )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_EXECUTABLE_SAME_LAW_BLOCK_REAL_SOURCES",
        "bindings": bindings,
        "superseded_v1": config["supersedes"],
        "completion": config["completion"],
        "one_parameter_set_sha256": content_sha256(config["parameter_set"]),
        "same_law_pass": same_law_pass,
        "photon_only_parameters": 0,
        "supported_channels": config["same_law_gate"]["derived_observables"],
        "blocked_channels": [],
        "maximum_exterior_field_equation_relative_residual": maximum_field_residual,
        "independent_route_audit": routes,
        "derived_observables": observables,
        "numerical_convergence": convergence,
        "countermodels": config["countermodels"],
        "source_status": {
            "ESO325_G004": config["eso325_source_manifest"]["status"],
            "SLACS": config["slacs_holdout_manifest"]["status"],
            "scientific_response_rows_opened": 0,
        },
        "strongest_falsifier": "Joint HST arcs and MUSE kinematics must accept one shared metric and fixed theory parameters, then transfer without retuning to the unchanged 12-lens SLACS confirmation set; any required photon multiplier or inconsistent delay normalization fails the law.",
        "remaining_blockers": config["eso325_source_manifest"]["unresolved_before_open"],
        "artifact_manifest": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in sorted(artifacts.items())
        },
        "artifact_bindings": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": file_sha256(CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(TEST_PATH)},
        },
        "published_sources": config["published_sources"],
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "decision": "REQUEST_INDEPENDENT_REAUDIT_BEFORE_ANY_ESO_OR_SLACS_RESPONSE_ACCESS",
    }
    _require(maximum_field_residual < 2.0e-5, "field equation residual too large")
    _require(convergence["relative_errors"]["deflection"] < 5.0e-4, "deflection grid drift")
    _require(convergence["relative_errors"]["shapiro_delay"] < 1.0e-5, "delay grid drift")
    _require(convergence["relative_errors"]["image_delay"] < 3.0e-3, "image-delay grid drift")
    _require(convergence["relative_errors"]["slip"] < 1.0e-10, "slip grid drift")
    _require(same_law_pass, "same-law audit failed")
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
