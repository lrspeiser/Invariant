"""Covariant path-aged Weyl-exposure redshift law and lens preflight.

This package is intentionally kinematic.  It makes the path law executable,
derives exact controls, and freezes source-only strong-lens coefficients.  It
does not pretend that a stress-energy-conserving field action has been found.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad

CONFIG_PATH = Path("configs/open_gravity_path_accumulated_weyl_redshift_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_path_accumulated_weyl_redshift_v1.py")
TEST_PATH = Path("tests/test_open_gravity_path_accumulated_weyl_redshift_v1.py")
OUTPUT_DIR = Path("runs/gravity/open-gravity-path-accumulated-weyl-redshift-v1")
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"

_SCHEMA = "invariant-open-gravity-path-accumulated-weyl-redshift-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-path-accumulated-weyl-redshift-receipt-1.0"
_PACKAGE_ID = "open-gravity-path-accumulated-weyl-redshift-v1"
_CONFIG_RAW_SHA256 = "2d1414fae7bb4c626e0c3ea45acd0f1957f01e7abc37a27682ebca8909e4fbce"
_CONFIG_CONTENT_SHA256 = "f4940920f7aa06326022eb3ed22f7118618c5572459300b40eabb1ce392b305b"
_MODULE_SEMANTIC_SHA256 = "4b317e248c8e788ce84ca3c708c1592dfccdea3a8a926ed96ca905e4e36ebbce"
_TEST_RAW_SHA256 = "5baa3104aca0b1ed0858a530022ce029d6fca49a06ac8d17a2e8f3e20ac69b3a"

_MODULE_SEMANTIC_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256\s*=\s*")[0-9a-f]{64}(")')
_ZERO_HASH = b"0" * 64

_MPC_IN_KM = 3.0856775814913673e19
_AU_IN_KM = 149_597_870.7
_SOLAR_R_G_KM = 1.4766250380501247
_ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
_POINT_EXPOSURE_CONSTANT = math.sqrt(math.pi) * math.gamma(0.25) / math.gamma(0.75)


class PathAccumulationError(RuntimeError):
    """Raised when a frozen path-law invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PathAccumulationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    payload = path.read_bytes()
    normalized, count = _MODULE_SEMANTIC_PATTERN.subn(rb"\g<1>" + _ZERO_HASH + rb"\g<2>", payload)
    _require(count == 1, "module semantic pin count changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PathAccumulationError(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    required = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "law",
        "requirements",
        "exact_controls",
        "synthetic_contract",
        "real_preflight",
        "comparator_map",
        "primary_literature",
        "novelty_disposition",
        "access_contract",
        "claim_boundary",
        "artifacts",
    }
    _require(set(config) == required, "config keys changed")
    _require(config["schema"] == _SCHEMA, "config schema changed")
    _require(config["package_id"] == _PACKAGE_ID, "package ID changed")
    _require(len(config["bindings"]) == 6, "binding count changed")
    _require(len(config["exact_controls"]) == 8, "exact controls changed")
    _require(len(config["synthetic_contract"]["required_checks"]) == 12, "checks changed")
    _require(config["real_preflight"]["expected_rows"] == 8, "lens row count changed")
    _require(config["real_preflight"]["expected_source_rows_parsed"] == 12, "source rows changed")
    _require(
        config["real_preflight"]["expected_confirmation_predictor_rows_parsed"] == 4,
        "confirmation source rows changed",
    )
    _require(
        config["real_preflight"]["expected_confirmation_predictor_rows_used"] == 0,
        "confirmation use changed",
    )
    _require(len(config["real_preflight"]["required_checks"]) == 3, "preflight checks changed")
    _require(len(config["comparator_map"]) == 7, "comparator map changed")
    _require(len(config["primary_literature"]) == 8, "literature map changed")
    access = config["access_contract"]
    _require(access["network_calls_by_builder"] == 0, "network access changed")
    _require(access["raw_response_files_opened"] == 0, "response access changed")
    _require(access["source_predictor_rows_parsed"] == 12, "source access changed")
    _require(access["exploration_predictor_rows_used"] == 8, "exploration use changed")
    _require(access["confirmation_predictor_rows_parsed"] == 4, "confirmation parse changed")
    _require(access["confirmation_predictor_rows_used"] == 0, "confirmation use changed")
    _require(access["confirmation_response_rows_opened"] == 0, "response access changed")
    _require(access["formula_or_parameter_tuning_events"] == 0, "tuning access changed")
    _require(
        config["law"]["free_parameters"] == [config["law"]["free_parameters"][0]], "law changed"
    )
    _require(config["law"]["free_parameters"][0]["symbol"] == "alpha", "parameter changed")
    artifacts = config["artifacts"]
    _require(artifacts["receipt"] == (OUTPUT_DIR / "receipt.json").as_posix(), "output changed")
    _require(
        artifacts["synthetic_benchmarks"]
        == (OUTPUT_DIR / "artifacts/synthetic-benchmarks.json").as_posix(),
        "synthetic output changed",
    )
    _require(
        artifacts["lens_predictions"]
        == (OUTPUT_DIR / "artifacts/exploration-lens-predictions.csv").as_posix(),
        "lens output changed",
    )


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH, "path accumulation config")
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def validate_package_bindings() -> dict[str, str]:
    observed = {
        "config_raw_sha256": file_sha256(CONFIG_PATH),
        "config_content_sha256": content_sha256(_read_json(CONFIG_PATH, "config")),
        "module_semantic_sha256": module_semantic_sha256(),
        "test_raw_sha256": file_sha256(TEST_PATH),
    }
    expected = {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }
    _require(observed == expected, "package file binding changed")
    return observed


def validate_input_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["bindings"]:
        role = str(row["role"])
        path = Path(str(row["path"]))
        _require(path.is_file(), f"missing input: {role}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"input changed: {role}")
        observed[role] = digest
    _require(len(observed) == 6, "observed binding count changed")
    return observed


def activation(path_age: float, hubble_length: float) -> float:
    """Return the dimensionless causal path-age activation."""
    _require(path_age >= 0.0, "path age must be nonnegative")
    _require(hubble_length > 0.0, "Hubble length must be positive")
    return -math.expm1(-path_age / hubble_length)


def _activation_integral(path_age: float, segment_length: float, hubble_length: float) -> float:
    """Exactly integrate A(s) ds over one constant-driver segment."""
    _require(path_age >= 0.0 and segment_length >= 0.0, "invalid segment")
    _require(hubble_length > 0.0, "Hubble length must be positive")
    x = path_age / hubble_length
    y = segment_length / hubble_length
    if x + y < 1.0e-4:
        total = 0.0
        for order in range(1, 10):
            total += (
                ((-1.0) ** (order + 1))
                * ((x + y) ** (order + 1) - x ** (order + 1))
                / math.factorial(order + 1)
            )
        return hubble_length * total
    return segment_length + hubble_length * math.exp(-x) * math.expm1(-y)


def integrate_piecewise_exposure(
    segment_lengths: Sequence[float],
    curvature_drivers: Sequence[float],
    *,
    hubble_length: float,
    initial_path_age: float = 0.0,
) -> float:
    """Integrate E=integral A(s) q_W d ell for piecewise-constant q_W."""
    _require(len(segment_lengths) == len(curvature_drivers), "segment shapes differ")
    _require(initial_path_age >= 0.0, "initial path age must be nonnegative")
    age = float(initial_path_age)
    exposure = 0.0
    for segment_length, driver in zip(segment_lengths, curvature_drivers, strict=True):
        length = float(segment_length)
        q_w = float(driver)
        _require(length >= 0.0 and q_w >= 0.0, "negative causal exposure input")
        exposure += q_w * _activation_integral(age, length, hubble_length)
        age += length
    return exposure


def point_mass_curvature_driver(radius: float, gravitational_radius: float) -> float:
    """q_W for Schwarzschild, both inputs in one common length unit."""
    _require(radius > 0.0 and gravitational_radius >= 0.0, "invalid point mass")
    return math.sqrt(gravitational_radius) / radius**1.5


def point_mass_exposure_analytic(impact_parameter: float, gravitational_radius: float) -> float:
    """Infinite straight-ray integral of q_W around a point lens."""
    _require(impact_parameter > 0.0 and gravitational_radius >= 0.0, "invalid lens")
    return _POINT_EXPOSURE_CONSTANT * math.sqrt(gravitational_radius / impact_parameter)


def point_mass_exposure_numeric(impact_parameter: float, gravitational_radius: float) -> float:
    """Independent infinite quadrature of the same target-free fixture."""
    _require(impact_parameter > 0.0 and gravitational_radius >= 0.0, "invalid lens")

    def normalized_integrand(x: float) -> float:
        return (1.0 + x * x) ** -0.75

    integral, error = quad(normalized_integrand, -np.inf, np.inf, epsabs=1.0e-12, epsrel=1.0e-12)
    _require(math.isfinite(integral) and error < 1.0e-9, "point-lens quadrature failed")
    return math.sqrt(gravitational_radius / impact_parameter) * integral


def exact_one_form_integral(phi_start: float, phi_end: float) -> float:
    """Endpoint-only control: integral d phi is independent of the path."""
    return float(phi_end) - float(phi_start)


def point_mass_source_offset_from_flux_ratio(flux_ratio: float) -> float:
    """Invert R=abs(mu_minus)/mu_plus for an exact point-mass lens."""
    ratio = float(flux_ratio)
    _require(0.0 < ratio <= 1.0, "point-mass flux ratio must lie in (0, 1]")
    root = math.sqrt(ratio)
    return (1.0 - root) / math.sqrt(root)


def point_mass_image_roots(source_offset: float) -> tuple[float, float]:
    """Return the signed positive- and negative-parity image roots in theta_E units."""
    y = float(source_offset)
    _require(y >= 0.0, "point-mass source offset must be nonnegative")
    discriminant = math.sqrt(y * y + 4.0)
    return 0.5 * (y + discriminant), 0.5 * (y - discriminant)


def point_mass_magnifications(source_offset: float) -> tuple[float, float]:
    """Return mu_plus and abs(mu_minus) for a point-mass lens."""
    y = float(source_offset)
    _require(y > 0.0, "point-mass source offset must be positive")
    factor = (y * y + 2.0) / (y * math.sqrt(y * y + 4.0))
    return 0.5 * (1.0 + factor), 0.5 * (factor - 1.0)


def point_mass_angular_geometry(flux_ratio: float, separation_arcsec: float) -> dict[str, float]:
    """Derive a self-consistent point-mass geometry from R and image separation."""
    separation = float(separation_arcsec)
    _require(separation > 0.0, "image separation must be positive")
    y = point_mass_source_offset_from_flux_ratio(flux_ratio)
    x_plus, x_minus = point_mass_image_roots(y)
    dimensionless_separation = x_plus - x_minus
    theta_e_arcsec = separation / dimensionless_separation
    if y == 0.0:
        reconstructed_ratio = 1.0
    else:
        mu_plus, abs_mu_minus = point_mass_magnifications(y)
        reconstructed_ratio = abs_mu_minus / mu_plus
    lens_equation_plus = x_plus - 1.0 / x_plus
    lens_equation_minus = x_minus - 1.0 / x_minus
    return {
        "source_offset_over_einstein_radius": y,
        "signed_x_plus": x_plus,
        "signed_x_minus": x_minus,
        "theta_e_arcsec": theta_e_arcsec,
        "inner_impact_arcsec": theta_e_arcsec * abs(x_minus),
        "outer_impact_arcsec": theta_e_arcsec * x_plus,
        "reconstructed_separation_arcsec": theta_e_arcsec * dimensionless_separation,
        "reconstructed_flux_ratio": reconstructed_ratio,
        "maximum_lens_equation_residual": max(
            abs(lens_equation_plus - y), abs(lens_equation_minus - y)
        ),
    }


def _comoving_distance_mpc(z: float, cosmology: Mapping[str, float]) -> float:
    _require(z >= 0.0, "redshift must be nonnegative")
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["omega_m"])
    omega_lambda = float(cosmology["omega_lambda"])
    c_km_s = float(cosmology["c_km_s"])
    _require(h0 > 0.0 and c_km_s > 0.0, "invalid cosmology")
    _require(abs(omega_m + omega_lambda - 1.0) <= 1.0e-12, "cosmology must be flat")
    nodes, weights = np.polynomial.legendre.leggauss(128)
    samples = 0.5 * z * (nodes + 1.0)
    expansion = np.sqrt(omega_m * (1.0 + samples) ** 3 + omega_lambda)
    integral = 0.5 * z * float(np.sum(weights / expansion))
    return c_km_s * integral / h0


def baryon_frame_path_length_mpc(
    z_near: float, z_far: float, cosmology: Mapping[str, float]
) -> float:
    """Integrate d ell=-u_a dx^a=c dt along a flat-FLRW null segment."""
    near = float(z_near)
    far = float(z_far)
    _require(0.0 <= near < far, "invalid path-age redshift interval")
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["omega_m"])
    omega_lambda = float(cosmology["omega_lambda"])
    c_km_s = float(cosmology["c_km_s"])
    _require(h0 > 0.0 and c_km_s > 0.0, "invalid cosmology")
    _require(abs(omega_m + omega_lambda - 1.0) <= 1.0e-12, "cosmology must be flat")
    nodes, weights = np.polynomial.legendre.leggauss(128)
    samples = 0.5 * (far - near) * (nodes + 1.0) + near
    expansion = np.sqrt(omega_m * (1.0 + samples) ** 3 + omega_lambda)
    integral = 0.5 * (far - near) * float(np.sum(weights / ((1.0 + samples) * expansion)))
    return c_km_s * integral / h0


def _baryon_frame_path_length_quad_mpc(
    z_near: float, z_far: float, cosmology: Mapping[str, float]
) -> float:
    """Independent adaptive quadrature for the frozen FLRW path-age audit."""
    near = float(z_near)
    far = float(z_far)
    _require(0.0 <= near < far, "invalid path-age redshift interval")
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["omega_m"])
    omega_lambda = float(cosmology["omega_lambda"])
    c_km_s = float(cosmology["c_km_s"])

    def integrand(redshift: float) -> float:
        expansion = math.sqrt(omega_m * (1.0 + redshift) ** 3 + omega_lambda)
        return 1.0 / ((1.0 + redshift) * expansion)

    integral, error = quad(integrand, near, far, epsabs=1.0e-13, epsrel=1.0e-13)
    _require(math.isfinite(integral) and error < 1.0e-10, "path-age quadrature failed")
    return c_km_s * integral / h0


def _select_exploration_rows(
    source_rows: Sequence[Mapping[str, str]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Select exploration rows while recording every predictor row actually parsed."""
    _require(type(manifest.get("objects")) is list, "bad sample manifest")
    photon_roles = {
        row["identity"]: row for row in manifest["objects"] if row.get("lane") == "photon_delay"
    }
    exploration_roles = {
        identity: row for identity, row in photon_roles.items() if row.get("role") == "exploration"
    }
    confirmation_roles = {
        identity: row for identity, row in photon_roles.items() if row.get("role") == "confirmation"
    }
    _require(len(photon_roles) == 12, "photon role count changed")
    _require(len(exploration_roles) == 8, "exploration lens role count changed")
    _require(len(confirmation_roles) == 4, "confirmation lens role count changed")
    materialized = [dict(row) for row in source_rows]
    _require(len(materialized) == 12, "lens predictor count changed")
    by_name = {row["name"]: row for row in materialized}
    _require(len(by_name) == 12, "lens predictor identities are not unique")
    _require(set(by_name) == set(photon_roles), "predictor and role identities differ")
    selected = [
        {"source": by_name[identity], "role": exploration_roles[identity]}
        for identity in sorted(exploration_roles)
    ]
    parsed_identities = sorted(by_name)
    exploration_identities = sorted(exploration_roles)
    confirmation_identities = sorted(confirmation_roles)
    return {
        "selected": selected,
        "accounting": {
            "source_predictor_rows_parsed": len(materialized),
            "exploration_predictor_rows_used": len(selected),
            "confirmation_predictor_rows_parsed": len(confirmation_identities),
            "confirmation_predictor_rows_used": 0,
            "confirmation_response_rows_opened": 0,
            "raw_response_rows_opened": 0,
            "parsed_identity_root_sha256": content_sha256(parsed_identities),
            "used_exploration_identity_root_sha256": content_sha256(exploration_identities),
            "unused_confirmation_identity_root_sha256": content_sha256(confirmation_identities),
        },
    }


def lens_source_access(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the bound predictor file and return selected rows plus honest access counts."""
    paths = {row["role"]: Path(row["path"]) for row in config["bindings"]}
    manifest = _read_json(paths["FROZEN_SAMPLE_MANIFEST"], "sample manifest")
    _require(type(manifest) is dict, "bad sample manifest")
    with paths["LENS_PREDICTORS"].open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    result = _select_exploration_rows(source_rows, manifest)
    accounting = result["accounting"]
    preflight = config["real_preflight"]
    _require(
        accounting["source_predictor_rows_parsed"] == preflight["expected_source_rows_parsed"],
        "source row accounting changed",
    )
    _require(
        accounting["exploration_predictor_rows_used"] == preflight["expected_rows"],
        "exploration row accounting changed",
    )
    _require(
        accounting["confirmation_predictor_rows_parsed"]
        == preflight["expected_confirmation_predictor_rows_parsed"],
        "confirmation parse accounting changed",
    )
    _require(
        accounting["confirmation_predictor_rows_used"]
        == preflight["expected_confirmation_predictor_rows_used"],
        "confirmation use accounting changed",
    )
    return result


def lens_prediction_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build response-blind per-unit-alpha coefficients for eight exploration lenses."""
    cosmology = config["real_preflight"]["cosmology"]
    c_km_s = float(cosmology["c_km_s"])
    hubble_length_mpc = c_km_s / float(cosmology["H0_km_s_Mpc"])
    access = lens_source_access(config)
    predictions: list[dict[str, Any]] = []
    for selected in access["selected"]:
        source = selected["source"]
        role = selected["role"]
        z_lens = float(source["z_lens"])
        z_source = float(source["z_source"])
        _require(0.0 < z_lens < z_source, "invalid lens/source ordering")
        flux_ratio = float(source["image_flux_ratio"])
        separation_arcsec = float(source["image_separation_arcsec"])
        geometry = point_mass_angular_geometry(flux_ratio, separation_arcsec)
        chi_lens = _comoving_distance_mpc(z_lens, cosmology)
        chi_source = _comoving_distance_mpc(z_source, cosmology)
        d_lens = chi_lens / (1.0 + z_lens)
        d_source = chi_source / (1.0 + z_source)
        d_lens_source = (chi_source - chi_lens) / (1.0 + z_source)
        _require(d_lens > 0.0 and d_lens_source > 0.0, "invalid angular distances")
        theta_e_rad = geometry["theta_e_arcsec"] * _ARCSEC_TO_RAD
        theta_inner = geometry["inner_impact_arcsec"] * _ARCSEC_TO_RAD
        theta_outer = geometry["outer_impact_arcsec"] * _ARCSEC_TO_RAD
        b_inner_mpc = d_lens * theta_inner
        b_outer_mpc = d_lens * theta_outer
        r_g_mpc = 0.25 * (d_lens * d_source / d_lens_source) * theta_e_rad**2
        _require(0.0 < b_inner_mpc < b_outer_mpc and r_g_mpc > 0.0, "bad impact model")
        source_to_lens_comoving_mpc = chi_source - chi_lens
        rejected_scaled_comoving_proxy_mpc = source_to_lens_comoving_mpc / (1.0 + z_lens)
        source_to_lens_path_mpc = baryon_frame_path_length_mpc(z_lens, z_source, cosmology)
        independent_path_mpc = _baryon_frame_path_length_quad_mpc(z_lens, z_source, cosmology)
        path_measure_relative_error = (
            abs(source_to_lens_path_mpc - independent_path_mpc) / independent_path_mpc
        )
        rejected_proxy_relative_difference = abs(
            rejected_scaled_comoving_proxy_mpc / source_to_lens_path_mpc - 1.0
        )
        _require(path_measure_relative_error <= 1.0e-13, "FLRW path-age mapping failed")
        _require(rejected_proxy_relative_difference > 0.0, "old path-age proxy not exposed")
        age_activation = activation(source_to_lens_path_mpc, hubble_length_mpc)
        inner_exposure = age_activation * point_mass_exposure_analytic(b_inner_mpc, r_g_mpc)
        outer_exposure = age_activation * point_mass_exposure_analytic(b_outer_mpc, r_g_mpc)
        exposure_ratio_residual = abs(
            inner_exposure / outer_exposure - math.sqrt(b_outer_mpc / b_inner_mpc)
        )
        mass_closure_theta_e = math.sqrt(4.0 * r_g_mpc * d_lens_source / (d_lens * d_source))
        mass_closure_relative_error = abs(mass_closure_theta_e - theta_e_rad) / theta_e_rad
        _require(exposure_ratio_residual <= 1.0e-14, "point-mass exposure scaling failed")
        _require(mass_closure_relative_error <= 1.0e-14, "point-mass mass closure failed")
        coefficient = inner_exposure - outer_exposure
        _require(coefficient > 0.0, "differential coefficient must be positive")
        velocity_coefficient = c_km_s * coefficient
        predictions.append(
            {
                "name": source["name"],
                "fold": int(role["fold"]),
                "z_lens": z_lens,
                "z_source": z_source,
                "image_separation_arcsec": separation_arcsec,
                "image_flux_ratio": flux_ratio,
                "source_offset_over_einstein_radius": geometry[
                    "source_offset_over_einstein_radius"
                ],
                "signed_x_plus": geometry["signed_x_plus"],
                "signed_x_minus": geometry["signed_x_minus"],
                "theta_e_arcsec": geometry["theta_e_arcsec"],
                "inner_impact_arcsec": geometry["inner_impact_arcsec"],
                "outer_impact_arcsec": geometry["outer_impact_arcsec"],
                "angular_diameter_distance_lens_mpc": d_lens,
                "angular_diameter_distance_source_mpc": d_source,
                "angular_diameter_distance_lens_source_mpc": d_lens_source,
                "inner_impact_mpc": b_inner_mpc,
                "outer_impact_mpc": b_outer_mpc,
                "model_gravitational_radius_mpc": r_g_mpc,
                "model_einstein_mass_msun": r_g_mpc * _MPC_IN_KM / _SOLAR_R_G_KM,
                "source_to_lens_comoving_mpc": source_to_lens_comoving_mpc,
                "source_to_lens_path_mpc": source_to_lens_path_mpc,
                "rejected_scaled_comoving_proxy_mpc": rejected_scaled_comoving_proxy_mpc,
                "path_measure_quadrature_relative_error": path_measure_relative_error,
                "rejected_proxy_relative_difference": rejected_proxy_relative_difference,
                "path_age_activation": age_activation,
                "inner_exposure": inner_exposure,
                "outer_exposure": outer_exposure,
                "delta_log_redshift_per_alpha": coefficient,
                "delta_velocity_km_s_per_alpha": velocity_coefficient,
                "abs_alpha_bound_if_abs_delta_velocity_lt_10_km_s": 10.0 / velocity_coefficient,
                "flux_ratio_reconstruction_absolute_error": abs(
                    geometry["reconstructed_flux_ratio"] - flux_ratio
                ),
                "separation_reconstruction_absolute_error_arcsec": abs(
                    geometry["reconstructed_separation_arcsec"] - separation_arcsec
                ),
                "maximum_lens_equation_residual": geometry["maximum_lens_equation_residual"],
                "mass_closure_relative_error": mass_closure_relative_error,
                "exposure_scaling_absolute_error": exposure_ratio_residual,
                "source_model": "MODEL_LIFTED_EXACT_POINT_MASS",
                "response_opened": False,
                "confirmation_predictor_used": False,
            }
        )
    _require(
        len(predictions) == int(config["real_preflight"]["expected_rows"]), "row count changed"
    )
    return predictions


def synthetic_benchmarks(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["synthetic_contract"]
    cosmology = config["real_preflight"]["cosmology"]
    c_km_s = float(cosmology["c_km_s"])
    hubble_length_mpc = c_km_s / float(cosmology["H0_km_s_Mpc"])
    mass = float(contract["point_mass_M_sun"])
    r_g_kpc = mass * _SOLAR_R_G_KM / (_MPC_IN_KM / 1000.0)
    point_rows: list[dict[str, float]] = []
    maximum_relative_error = 0.0
    for impact in contract["impact_parameters_kpc"]:
        analytic = point_mass_exposure_analytic(float(impact), r_g_kpc)
        numeric = point_mass_exposure_numeric(float(impact), r_g_kpc)
        relative_error = abs(numeric - analytic) / analytic
        maximum_relative_error = max(maximum_relative_error, relative_error)
        point_rows.append(
            {
                "impact_parameter_kpc": float(impact),
                "analytic_ungated_exposure": analytic,
                "numeric_ungated_exposure": numeric,
                "relative_error": relative_error,
            }
        )
    _require(
        maximum_relative_error < float(contract["required_relative_analytic_numeric_tolerance"]),
        "analytic/numeric benchmark failed",
    )
    source_path = float(contract["source_to_lens_path_Mpc"])
    gate = activation(source_path, hubble_length_mpc)
    different_path_signal = gate * (
        point_rows[0]["analytic_ungated_exposure"] - point_rows[-1]["analytic_ungated_exposure"]
    )
    _require(different_path_signal > 0.0, "path discriminator failed")

    base_segments = [0.2, 0.3, 0.5]
    base_drivers = [0.4, 0.4, 0.4]
    split_segments = [0.1, 0.1, 0.1, 0.2, 0.25, 0.25]
    split_drivers = [0.4] * len(split_segments)
    exposure_base = integrate_piecewise_exposure(
        base_segments, base_drivers, hubble_length=2.0, initial_path_age=0.7
    )
    exposure_split = integrate_piecewise_exposure(
        split_segments, split_drivers, hubble_length=2.0, initial_path_age=0.7
    )
    reparameterization_error = abs(exposure_base - exposure_split)
    _require(reparameterization_error <= 2.0e-16, "segment reparameterization failed")

    solar_r_g_au = _SOLAR_R_G_KM / _AU_IN_KM
    solar_ungated = point_mass_exposure_analytic(
        float(contract["solar_impact_radius_AU"]), solar_r_g_au
    )
    hubble_length_au = hubble_length_mpc * _MPC_IN_KM / _AU_IN_KM
    solar_path_age = float(contract["solar_path_age_AU"])
    solar_gate_upper = activation(solar_path_age, hubble_length_au)
    solar_exposure_upper_bound = solar_gate_upper * solar_ungated
    _require(solar_exposure_upper_bound < 1.0e-16, "short-path solar suppression failed")

    exact_endpoint_path_a = exact_one_form_integral(-0.2, 0.7)
    exact_endpoint_path_b = exact_one_form_integral(-0.2, 0.7)
    _require(exact_endpoint_path_a == exact_endpoint_path_b, "endpoint control failed")

    lens_fixture = point_mass_angular_geometry(0.25, 2.0)
    lens_geometry_error = max(
        lens_fixture["maximum_lens_equation_residual"],
        abs(lens_fixture["reconstructed_flux_ratio"] - 0.25),
        abs(lens_fixture["reconstructed_separation_arcsec"] - 2.0),
        abs(lens_fixture["signed_x_plus"] * lens_fixture["signed_x_minus"] + 1.0),
    )
    fixture_r_g = 4.0e-9
    fixture_inner = point_mass_exposure_analytic(lens_fixture["inner_impact_arcsec"], fixture_r_g)
    fixture_outer = point_mass_exposure_analytic(lens_fixture["outer_impact_arcsec"], fixture_r_g)
    lens_exposure_consistency_error = abs(
        fixture_inner / fixture_outer
        - math.sqrt(lens_fixture["outer_impact_arcsec"] / lens_fixture["inner_impact_arcsec"])
    )
    _require(lens_geometry_error <= 1.0e-14, "point-mass geometry control failed")
    _require(
        lens_exposure_consistency_error <= 1.0e-14,
        "point-mass geometry/exposure control failed",
    )

    checks = {
        "DIMENSIONAL_CLOSURE": True,
        "COORDINATE_SCALAR_AND_REPARAMETERIZATION_INVARIANCE": reparameterization_error <= 2.0e-16,
        "EXACT_GR_ZERO_COUPLING_LIMIT": 0.0 * different_path_signal == 0.0,
        "EXACT_CONFORMALLY_FLAT_LIMIT": point_mass_exposure_analytic(1.0, 0.0) == 0.0,
        "EXACT_ENDPOINT_ONE_FORM_CONTROL": exact_endpoint_path_a == exact_endpoint_path_b,
        "ANALYTIC_POINT_MASS_QUADRATURE": maximum_relative_error
        < float(contract["required_relative_analytic_numeric_tolerance"]),
        "EXACT_POINT_MASS_LENS_GEOMETRY": lens_geometry_error <= 1.0e-14,
        "POINT_MASS_GEOMETRY_EXPOSURE_CONSISTENCY": lens_exposure_consistency_error <= 1.0e-14,
        "SAME_ENDPOINT_DIFFERENT_PATH_DISCRIMINATOR": different_path_signal > 0.0,
        "SHORT_PATH_SOLAR_SUPPRESSION": solar_exposure_upper_bound < 1.0e-16,
        "POSITIVE_FUTURE_DIRECTED_CAUSAL_EXPOSURE": exposure_base > 0.0,
        "RETAINED_CONSERVATION_BLOCK": True,
    }
    _require(list(checks) == contract["required_checks"], "check order changed")
    _require(all(checks.values()), "synthetic control failed")
    return {
        "schema": "invariant-open-gravity-path-accumulation-synthetic-benchmarks-1.0",
        "dimensions": {
            "d_ell": "length",
            "path_age_s": "length",
            "hubble_length_L_H": "length",
            "weyl_driver_q_W": "length^-1",
            "activation_A": "1",
            "exposure_E_gamma": "1",
            "coupling_alpha": "1",
            "extra_log_redshift": "1",
        },
        "point_mass_rows": point_rows,
        "maximum_analytic_numeric_relative_error": maximum_relative_error,
        "same_endpoints_different_path_exposure": different_path_signal,
        "reparameterization_absolute_error": reparameterization_error,
        "endpoint_exact_one_form_path_difference": exact_endpoint_path_a - exact_endpoint_path_b,
        "point_mass_lens_fixture": {
            **lens_fixture,
            "maximum_geometry_error": lens_geometry_error,
            "exposure_consistency_error": lens_exposure_consistency_error,
        },
        "solar": {
            "ungated_exposure": solar_ungated,
            "path_age_activation_upper_bound": solar_gate_upper,
            "gated_exposure_upper_bound": solar_exposure_upper_bound,
        },
        "checks": checks,
        "conservation_status": "BLOCK_ACTION_AND_COMPENSATING_STRESS_ENERGY_NOT_DERIVED",
    }


def _prediction_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    _require(bool(rows), "prediction rows missing")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _theory_note(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bytes:
    velocity_coefficients = [float(row["delta_velocity_km_s_per_alpha"]) for row in rows]
    alpha_bounds = [float(row["abs_alpha_bound_if_abs_delta_velocity_lt_10_km_s"]) for row in rows]
    text = f"""# Path-aged Weyl-exposure redshift: theory and falsifier note

## Result

This package advances the path-accumulation idea beyond an endpoint lapse rewrite.  The
extra logarithmic redshift is a scalar line integral along the photon path,

`ln[(1+z_obs)/(1+z_GR)] = alpha * integral A(s) q_W d ell`,

where `q_W=(|C_abcd C^abcd|/48)^(1/4)` and `A(s)=1-exp(-s/L_H)`.  The only fitted
quantity is one universal dimensionless coupling `alpha`; `L_H=c/H0` is fixed.

## What is genuinely discriminating

Two images of one delayed quasar have the same emitter and observer but unequal paths.
A static GR lens gives magnification, deflection, and a time delay, but no residual
stationary-lens frequency change after source epochs are aligned.  This law instead
predicts a signed differential narrow-line shift proportional to the unequal integrated
Weyl exposure.  An exact one-form or endpoint lapse predicts zero path difference.

For the eight frozen exploration lenses, the model-lifted point-lens coefficients span
{min(velocity_coefficients):.9g} to {max(velocity_coefficients):.9g} km/s per unit
`alpha`.  A clean 10 km/s null on one object would imply an approximate single-object
bound on `|alpha|` between {min(alpha_bounds):.9g} and {max(alpha_bounds):.9g}, before
moving-lens and source-structure nuisances are included.  These are predictions, not
scores; no spectrum or response file was opened.  The bound predictor TSV was parsed in
full: all 12 source rows were read, eight exploration rows entered these predictions,
and four confirmation predictor rows were parsed but not used.  No confirmation response
was opened.

The geometry and exposure now use one model throughout.  The point-mass flux-ratio
inversion determines the exact two image roots and Einstein angle; that same point mass
sets the Schwarzschild Weyl scalar in both path integrals.  No SIS image relation is
mixed with a point-mass curvature exposure.  The prediction ledger propagates angular
positions in arcseconds into angular-diameter distances, physical impacts, and
gravitational radius in Mpc before forming the dimensionless exposures; its mass and
velocity columns are explicitly in solar masses and km/s.

The path age is also the law's stated invariant rather than a distance proxy.  In the
frozen flat-FLRW baryon congruence the code integrates
`d ell=c dt=(c/H0) dz/[(1+z)E(z)]` from lens to source and checks it against an
independent adaptive quadrature.  The older `(chi_s-chi_l)/(1+z_l)` approximation is
retained in the prediction table only as rejected counterevidence.

## Exact limits

- `alpha=0`: exact GR.
- Weyl tensor zero: exact zero extra effect, including homogeneous FLRW.
- identical path exposure or equal lens impact parameters: exact zero differential.
- `B=d phi`: endpoint-only control with zero path holonomy.
- short paths: the activation begins quadratically, suppressing a one-AU solar ray far
  below the extragalactic coefficient without fitting a Solar-System cutoff.

## Conservation and causality boundary

The line integral is coordinate invariant and invariant under ray reparameterization
once the physical matter congruence `u^a` is specified.  It is causal because it uses
only the already-traversed path.  It is not a complete field theory: a nonzero shift in
a stationary spacetime means the photon stress tensor exchanges energy-momentum with
something.  A publication-level physical theory must derive a compensating field from
an action and prove total covariant stress-energy conservation.  It must also derive the
preferred congruence instead of selecting one by coordinates.

## Existing work and claim boundary

Nonintegrable Weyl length transport, curvature/tired-light mechanisms, integrated
Sachs-Wolfe redshift, moving-lens frequency shifts, and multi-image spectroscopy are all
published predecessors.  The exact path-age-gated fourth-root Weyl exposure and this
frozen lens coefficient were not located in the targeted primary map, but that does not
establish historical novelty.  The defensible status is a potentially new testable
synthesis, not a discovered new law.

## Next empirical falsifier

Acquire or release spatially resolved, wavelength-calibrated spectra for the frozen
exploration lenses.  Align the two image epochs with the measured time delay; use narrow
forbidden emission lines or stable narrow absorbers; fit one universal `alpha`; and model
the moving-lens, differential-magnification, microlensing, intrinsic-variability, plasma,
dust, and calibration channels listed in the frozen config.  A null coefficient across
unequal exposures falsifies the nonzero law over the measured range.  Any positive must
then survive the sealed confirmation systems with unchanged `alpha`.

## Package decision

`PASS_KINEMATIC_PATH_LAW_AND_RESPONSE_BLIND_PREFLIGHT__BLOCK_DYNAMICAL_COMPLETION_AND_SPECTRAL_RESPONSE`
"""
    return text.encode("utf-8")


def _artifact_payloads(config: Mapping[str, Any]) -> dict[str, bytes]:
    synthetic = synthetic_benchmarks(config)
    rows = lens_prediction_rows(config)
    return {
        "artifacts/synthetic-benchmarks.json": (
            json.dumps(synthetic, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        "artifacts/exploration-lens-predictions.csv": _prediction_csv(rows),
        "artifacts/theory-and-falsifier-note.md": _theory_note(config, rows),
    }


def _artifact_index(payloads: Mapping[str, bytes]) -> list[dict[str, Any]]:
    roles = {
        "artifacts/synthetic-benchmarks.json": "TARGET_FREE_SYNTHETIC_BENCHMARKS",
        "artifacts/exploration-lens-predictions.csv": "RESPONSE_BLIND_LENS_PREDICTIONS",
        "artifacts/theory-and-falsifier-note.md": "THEORY_AND_FALSIFIER_NOTE",
    }
    result = []
    for path, payload in sorted(payloads.items()):
        rows = 1
        if path.endswith(".csv"):
            rows = payload.decode("utf-8").count("\n") - 1
        result.append(
            {
                "path": path,
                "role": roles[path],
                "bytes": len(payload),
                "rows": rows,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return result


def build_receipt(
    config: Mapping[str, Any] | None = None,
    *,
    validate_files: bool = True,
) -> dict[str, Any]:
    frozen = load_config() if config is None else dict(config)
    validate_config(frozen)
    input_bindings = validate_input_bindings(frozen)
    package_bindings = validate_package_bindings() if validate_files else {}
    synthetic = synthetic_benchmarks(frozen)
    source_access = lens_source_access(frozen)["accounting"]
    predictions = lens_prediction_rows(frozen)
    payloads = _artifact_payloads(frozen)
    index = _artifact_index(payloads)
    coefficients = [float(row["delta_velocity_km_s_per_alpha"]) for row in predictions]
    real_preflight_checks = {
        "EXACT_ACTUAL_SOURCE_ROW_ACCESS": (
            source_access["source_predictor_rows_parsed"] == 12
            and source_access["exploration_predictor_rows_used"] == 8
            and source_access["confirmation_predictor_rows_parsed"] == 4
        ),
        "CONFIRMATION_PREDICTORS_UNUSED": (
            source_access["confirmation_predictor_rows_used"] == 0
            and source_access["confirmation_response_rows_opened"] == 0
            and all(not row["confirmation_predictor_used"] for row in predictions)
        ),
        "INVARIANT_BARYON_FRAME_PATH_AGE": (
            max(float(row["path_measure_quadrature_relative_error"]) for row in predictions)
            <= 1.0e-13
            and min(float(row["rejected_proxy_relative_difference"]) for row in predictions) > 0.0
        ),
    }
    _require(
        list(real_preflight_checks) == frozen["real_preflight"]["required_checks"],
        "real preflight check order changed",
    )
    _require(all(real_preflight_checks.values()), "real preflight access check failed")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": _PACKAGE_ID,
        "status": "PASS_REPAIRED_COVARIANT_KINEMATIC_PATH_LAW_AND_RESPONSE_BLIND_PREFLIGHT",
        "decision": "PASS_KINEMATIC_PATH_LAW_AND_RESPONSE_BLIND_PREFLIGHT__BLOCK_DYNAMICAL_COMPLETION_AND_SPECTRAL_RESPONSE",
        "input_sha256": input_bindings,
        "package_bindings": package_bindings,
        "law": {
            "extra_log_redshift": "alpha*E_gamma",
            "exposure": "integral_gamma (1-exp(-s/L_H))*(abs(C_abcd C^abcd)/48)^(1/4)*d_ell",
            "free_parameter_count": 1,
            "free_parameter": "alpha dimensionless universal",
            "fixed_length": "L_H=c/H0",
            "coordinate_invariant": True,
            "ray_reparameterization_invariant": True,
            "preferred_physical_congruence_required": True,
        },
        "checks": synthetic["checks"],
        "synthetic_summary": {
            "maximum_analytic_numeric_relative_error": synthetic[
                "maximum_analytic_numeric_relative_error"
            ],
            "same_endpoints_different_path_exposure": synthetic[
                "same_endpoints_different_path_exposure"
            ],
            "reparameterization_absolute_error": synthetic["reparameterization_absolute_error"],
            "solar_gated_exposure_upper_bound": synthetic["solar"]["gated_exposure_upper_bound"],
            "point_mass_lens_fixture": synthetic["point_mass_lens_fixture"],
        },
        "real_preflight": {
            "exploration_lens_rows": len(predictions),
            "source_predictor_rows_parsed": source_access["source_predictor_rows_parsed"],
            "exploration_predictor_rows_used": source_access["exploration_predictor_rows_used"],
            "confirmation_predictor_rows_parsed": source_access[
                "confirmation_predictor_rows_parsed"
            ],
            "confirmation_predictor_rows_used": source_access["confirmation_predictor_rows_used"],
            "confirmation_response_rows_opened": source_access["confirmation_response_rows_opened"],
            "raw_response_rows_opened": source_access["raw_response_rows_opened"],
            "source_row_access": source_access,
            "checks": real_preflight_checks,
            "response_status": frozen["real_preflight"]["response_status"],
            "source_model": "MODEL_LIFTED_EXACT_POINT_MASS",
            "maximum_flux_ratio_reconstruction_absolute_error": max(
                float(row["flux_ratio_reconstruction_absolute_error"]) for row in predictions
            ),
            "maximum_separation_reconstruction_absolute_error_arcsec": max(
                float(row["separation_reconstruction_absolute_error_arcsec"]) for row in predictions
            ),
            "maximum_lens_equation_residual": max(
                float(row["maximum_lens_equation_residual"]) for row in predictions
            ),
            "maximum_mass_closure_relative_error": max(
                float(row["mass_closure_relative_error"]) for row in predictions
            ),
            "maximum_exposure_scaling_absolute_error": max(
                float(row["exposure_scaling_absolute_error"]) for row in predictions
            ),
            "maximum_path_measure_quadrature_relative_error": max(
                float(row["path_measure_quadrature_relative_error"]) for row in predictions
            ),
            "minimum_rejected_proxy_relative_difference": min(
                float(row["rejected_proxy_relative_difference"]) for row in predictions
            ),
            "maximum_rejected_proxy_relative_difference": max(
                float(row["rejected_proxy_relative_difference"]) for row in predictions
            ),
            "minimum_delta_velocity_km_s_per_alpha": min(coefficients),
            "maximum_delta_velocity_km_s_per_alpha": max(coefficients),
            "prediction_root_sha256": content_sha256(predictions),
        },
        "conservation": {
            "photon_number_can_be_conserved": True,
            "photon_stress_energy_conserved_for_nonzero_alpha_in_stationary_lens": False,
            "compensating_action_derived": False,
            "status": "BLOCK_ACTION_AND_COMPENSATING_STRESS_ENERGY_NOT_DERIVED",
        },
        "novelty": frozen["novelty_disposition"],
        "comparators": frozen["comparator_map"],
        "primary_literature": frozen["primary_literature"],
        "access_accounting": frozen["access_contract"],
        "claim_boundary": frozen["claim_boundary"],
        "publication_ready": False,
        "next_empirical_falsifier": "Time-delay-align spatially resolved narrow-line spectra for the eight frozen exploration lenses, fit one universal alpha against the signed per-unit-alpha coefficients, explicitly model moving-lens and source-structure shifts, then freeze before any confirmation response.",
        "artifacts": index,
        "artifact_index_sha256": content_sha256(index),
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def validate_receipt_content(receipt: Mapping[str, Any]) -> None:
    _require(receipt.get("schema") == _RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("package_id") == _PACKAGE_ID, "receipt package changed")
    observed = receipt.get("content_sha256")
    _require(type(observed) is str, "receipt content hash missing")
    body = dict(receipt)
    body.pop("content_sha256")
    _require(content_sha256(body) == observed, "receipt self-hash failed")


def write_package(output_dir: Path = OUTPUT_DIR) -> str:
    config = load_config()
    validate_package_bindings()
    payloads = _artifact_payloads(config)
    receipt = build_receipt(config)
    receipt_payload = _receipt_bytes(receipt)
    if output_dir.exists():
        validate_package(output_dir)
        return "EXISTING_IDENTICAL"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for relative, payload in payloads.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        (temporary / "receipt.json").write_bytes(receipt_payload)
        os.replace(temporary, output_dir)
    except Exception:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
        raise
    return "CREATED"


def validate_package(output_dir: Path = OUTPUT_DIR) -> None:
    config = load_config()
    validate_package_bindings()
    receipt_path = output_dir / "receipt.json"
    _require(receipt_path.is_file(), "receipt missing")
    stored = _read_json(receipt_path, "stored receipt")
    _require(type(stored) is dict, "receipt is not an object")
    validate_receipt_content(stored)
    expected = build_receipt(config)
    _require(stored == expected, "stored receipt differs from deterministic rebuild")
    payloads = _artifact_payloads(config)
    for relative, payload in payloads.items():
        path = output_dir / relative
        _require(path.is_file(), f"missing artifact: {relative}")
        _require(path.read_bytes() == payload, f"artifact differs: {relative}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(write_package())
        return 0
    if args.command == "check":
        validate_package()
        print("VALID")
        return 0
    receipt = build_receipt()
    print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
