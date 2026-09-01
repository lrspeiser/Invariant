"""Physical-unit, mutation-bound Lane 7 target-free repair.

V5 preserves V4 as blocked evidence.  It hashes exact sources and sealed manifests but does
not parse or decode any ESO scientific array or SLACS response value.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import irfftn, rfftn
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/open_gravity_same_law_eso325_extended_source_v5.json"
MODULE_PATH = Path(__file__).resolve()
TEST_PATH = REPOSITORY_ROOT / "tests/test_open_gravity_same_law_eso325_extended_source_v5.py"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "runs/gravity/open-gravity-same-law-eso325-extended-source-v5/receipt.json"
)
ARTIFACT_DIRECTORY = OUTPUT_PATH.parent / "artifacts"

EXPECTED_LAW_BINDING = {
    "field_equations": {
        "newton": "nabla^2 U = 4*pi*G*rho_total",
        "yukawa": "(nabla^2 - mu^2) Y = 4*pi*G*rho_total",
    },
    "metric": (
        "ds^2 = -(1 + 2*Phi/c^2)*c^2*dt^2 + "
        "(1 - 2*Psi/c^2)*(dx^2 + dy^2 + dz^2)"
    ),
    "state_map": {
        "Phi": "Phi = U + (4/3)*g*Y",
        "Psi": "Psi = U + (2/3)*g*Y",
    },
    "matter_observable": "a_i = -partial_i Phi; units (km/s)^2/kpc",
    "photon_observable": (
        "alpha_hat_i = c^-2 * integral partial_i(Phi+Psi) dl; "
        "alpha_reduced_i = (D_ls/D_s)*alpha_hat_i"
    ),
    "lens_mapping": "beta_i = theta_i - alpha_reduced_i(D_l*theta)",
    "coefficients": {
        "phi_yukawa": 4.0 / 3.0,
        "psi_yukawa": 2.0 / 3.0,
        "photon_integral_prefactor": 1.0,
    },
    "constants": {
        "G_kpc_km2_s2_Msun": 4.300917270036279e-6,
        "c_km_s": 299792.458,
        "arcsec_per_radian": 206264.80624709636,
    },
    "unit_ledger": {
        "coordinate": "kpc",
        "density": "Msun/kpc^3",
        "U_Y_Phi_Psi": "(km/s)^2",
        "potential_gradient": "(km/s)^2/kpc",
        "line_element": "kpc",
        "deflection": "radian",
        "angular_diameter_distance": "Mpc",
    },
}
EXPECTED_LAW_SHA256 = "f3573bb2c1a1d07a611c186290a083fed278c08b95bf3447124ed39326eaaf72"
EXPECTED_SOURCE_BINDING = [
    {
        "role": "HST_F814W_LENS_LIGHT",
        "path": (
            "work/private/open-gravity-eso325-source-v1/"
            "hst_10429_09_acs_wfc_f814w_j95t09_drc.fits"
        ),
        "bytes": 369486720,
        "sha256": "f2a711874a38cf6364d7222d17cb210e8800b054fb5e2f01bf3f8aa061ad484a",
    },
    {
        "role": "HST_F475W_ARC_PLUS_LENS_LIGHT",
        "path": (
            "work/private/open-gravity-eso325-source-v1/"
            "hst_10429_10_acs_wfc_f475w_j95t10_drc.fits"
        ),
        "bytes": 367663680,
        "sha256": "7e77aa1ca44a26f491fe0ac8d6bfd8614ff6d036a460967f39c7a4bfdf2d0d17",
    },
    {
        "role": "MUSE_PRIMARY_CUBE",
        "path": (
            "work/private/open-gravity-eso325-source-v1/ADP.2016-09-07T12_23_32.515.fits"
        ),
        "bytes": 7378352640,
        "sha256": "dea67d98c39284e7be30c78f3d34a61a4a834c816e1c2b809f0909298ec87367",
    },
    {
        "role": "MUSE_ARCHIVE_WHITELIGHT_AUXILIARY",
        "path": (
            "work/private/open-gravity-eso325-source-v1/ADP.2016-09-07T12_23_32.516.fits"
        ),
        "bytes": 1108800,
        "sha256": "8cf81fdc7f93e285444f7a83bce57cfcd974f165e6e9aa9b25093eb09a35f6e6",
    },
    {
        "role": "PAPER_AND_SUPPLEMENT",
        "path": (
            "work/private/open-gravity-lane7-eso325-supplement-v1/"
            "collett-2018-science-and-supplement.pdf"
        ),
        "bytes": 2703367,
        "sha256": "0a96efeb3f3fc4312a72e8286a9e8a3b93039fbd2443408f2f8af237408dacfe",
    },
    {
        "role": "PUBLISHED_PYLENS_CODE_REFERENCE_ONLY",
        "path": (
            "work/private/open-gravity-lane7-eso325-supplement-v1/"
            "pylens-1cb65f244b8ecf537efea2f93c5951a30d4dae36.tar.gz"
        ),
        "bytes": 652349,
        "sha256": "4996cb3d58b8032b3b96e2783227d41f5aca6eb388c77b371b1428990801c679",
    },
]
EXPECTED_SOURCE_SHA256 = "e06a0f5604edcceca33c44946ff5bffbeaa407570dcdedfaea339cbdfccf5f25"
EXPECTED_CONTRACT_HASHES = {
    "extended_density_contract": "0dab74b843c7d4fe8441b35b8f3c76086fb8aae1b7fb66521798c951876a9568",
    "shared_state_contract": "64ae973fb010cf0d5439d232d14236d131b2ac7b060e32bfd51c2fb1bd688f17",
    "predictive_likelihood_contract": (
        "a51a01f0b25ac4e6ed82e41d098cac19cdeabf6f6983b2caa98273cbaf0abcad"
    ),
    "reduction_readiness": "f69db12440d274f0791649cecf90a836bf735fddda2c74251fab714af91f869b",
    "target_free_gate": "5dd1812be72968fc1c23001bc4aa9c137f3a50ab9ffa3e3f5434e9c8c868ebe3",
}


class SameLawESO325V5Error(RuntimeError):
    """Raised when a V5 frozen contract or target-free invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameLawESO325V5Error(message)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config["schema"] == "invariant-open-gravity-same-law-eso325-extended-source-5.0",
        "schema widened",
    )
    _require(
        config["status"] == "APPEND_ONLY_PHYSICAL_UNIT_REPAIR_FROZEN_NO_ARRAY_DECODE",
        "freeze status changed",
    )
    _require(config["law_binding"] == EXPECTED_LAW_BINDING, "law binding mutated")
    _require(config["law_sha256"] == EXPECTED_LAW_SHA256, "law hash changed")
    _require(_canonical_sha256(config["law_binding"]) == EXPECTED_LAW_SHA256, "law hash fails")
    _require(config["source_binding"] == EXPECTED_SOURCE_BINDING, "source binding mutated")
    _require(config["source_binding_sha256"] == EXPECTED_SOURCE_SHA256, "source hash changed")
    _require(
        _canonical_sha256(config["source_binding"]) == EXPECTED_SOURCE_SHA256,
        "source manifest hash fails",
    )
    _require(config["contract_hashes"] == EXPECTED_CONTRACT_HASHES, "contract hashes changed")
    for key, expected in EXPECTED_CONTRACT_HASHES.items():
        _require(_canonical_sha256(config[key]) == expected, f"contract mutation: {key}")
    constants = config["law_binding"]["constants"]
    _require(constants["G_kpc_km2_s2_Msun"] == 4.300917270036279e-6, "G mutated")
    _require(constants["c_km_s"] == 299792.458, "c mutated")
    _require(
        config["law_binding"]["coefficients"]
        == {
            "phi_yukawa": 4.0 / 3.0,
            "psi_yukawa": 2.0 / 3.0,
            "photon_integral_prefactor": 1.0,
        },
        "state coefficients mutated",
    )
    _require("/100" not in json.dumps(config["law_binding"]), "unexplained normalization returned")
    cosmology = config["cosmology"]
    distances = angular_diameter_distances(config)
    for key, observed in distances.items():
        expected = cosmology["frozen_distances"][key]
        _require(math.isclose(observed, expected, rel_tol=2e-14, abs_tol=2e-12), key)
    readiness = config["reduction_readiness"]
    _require(len(readiness["missing_external_inputs_before_any_scientific_array_decode"]) == 4,
             "external input split changed")
    _require(
        len(readiness["empirical_gates_only_after_a_separately_authorized_development_array_decode"])
        == 5,
        "post-decode gate split changed",
    )
    seal = config["slacs_seal"]
    _require(seal["status"] == "SEALED_UNCHANGED", "SLACS seal opened")
    _require(seal["reserved_confirmation"] == 12, "SLACS count changed")
    _require(seal["response_manifest_deserialized_by_v5"] is False, "response deserialized")
    _require(seal["response_values_opened_by_v5"] == 0, "response value opened")
    accounting = config["access_accounting"]
    _require(all(value == 0 for value in accounting.values()), "access accounting widened")


def angular_diameter_distances(config: Mapping[str, Any]) -> dict[str, float]:
    cosmology = config["cosmology"]
    constants = config["law_binding"]["constants"]
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["Omega_m"])
    omega_lambda = float(cosmology["Omega_lambda"])
    c = float(constants["c_km_s"])
    tolerance = cosmology["integration_tolerance"]

    def comoving(redshift: float) -> float:
        integrand = lambda z: 1.0 / math.sqrt(omega_m * (1.0 + z) ** 3 + omega_lambda)
        integral = quad(
            integrand,
            0.0,
            redshift,
            epsabs=float(tolerance["absolute_Mpc"]) * h0 / c,
            epsrel=float(tolerance["relative"]),
        )[0]
        return c * integral / h0

    lens_redshift = float(cosmology["lens_redshift"])
    source_redshift = float(cosmology["source_redshift"])
    dc_lens = comoving(lens_redshift)
    dc_source = comoving(source_redshift)
    d_lens = dc_lens / (1.0 + lens_redshift)
    d_source = dc_source / (1.0 + source_redshift)
    d_lens_source = (dc_source - dc_lens) / (1.0 + source_redshift)
    ratio = d_lens_source / d_source
    kpc_per_arcsec = (
        d_lens * 1000.0 / float(constants["arcsec_per_radian"])
    )
    return {
        "D_l_Mpc": d_lens,
        "D_s_Mpc": d_source,
        "D_ls_Mpc": d_lens_source,
        "D_ls_over_D_s": ratio,
        "lens_kpc_per_arcsec": kpc_per_arcsec,
    }


def verify_v4_preservation_and_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    preservation_binding = config["v4_preservation"]
    preservation_path = REPOSITORY_ROOT / preservation_binding["path"]
    _require(file_sha256(preservation_path) == preservation_binding["sha256"], "preservation drift")
    preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
    _require(preservation["status"].startswith("V4_BYTE_EXACT"), "V4 preservation status")
    preserved_root = preservation_path.parent
    preserved_rows = []
    for item in preservation["files"]:
        path = preserved_root / item["path"]
        _require(path.stat().st_size == item["bytes"], f"V4 preserved bytes: {item['path']}")
        observed = file_sha256(path)
        _require(observed == item["sha256"], f"V4 preserved hash: {item['path']}")
        preserved_rows.append({"path": item["path"], "bytes": item["bytes"], "sha256": observed})
    v4_receipt_path = REPOSITORY_ROOT / preservation_binding["v4_receipt_path"]
    _require(
        file_sha256(v4_receipt_path) == preservation_binding["v4_receipt_sha256"],
        "V4 receipt drift",
    )
    source_rows = []
    for item in config["source_binding"]:
        path = REPOSITORY_ROOT / item["path"]
        _require(path.is_file(), f"source missing: {item['role']}")
        _require(path.stat().st_size == item["bytes"], f"source byte drift: {item['role']}")
        observed = file_sha256(path)
        _require(observed == item["sha256"], f"source hash drift: {item['role']}")
        source_rows.append({**item, "hash_pass": True})
    seal_rows = []
    for key in ("sample_manifest", "predictor_manifest", "response_manifest"):
        item = config["slacs_seal"][key]
        observed = file_sha256(REPOSITORY_ROOT / item["path"])
        _require(observed == item["sha256"], f"SLACS sealed hash drift: {key}")
        seal_rows.append({"role": key, "path": item["path"], "sha256": observed})
    return {
        "status": "PASS_V4_BYTE_EXACT_AND_V5_SOURCE_BINDINGS",
        "v4_preserved_files": preserved_rows,
        "source_rows": source_rows,
        "slacs_manifest_hashes_only": seal_rows,
        "slacs_response_manifest_deserialized": False,
        "scientific_array_elements_decoded": 0,
        "scientific_response_values_opened": 0,
    }


def unit_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    # Dimension vector order is (L, M, T).
    dimensions = {
        "G": (3, -1, -2),
        "rho": (-3, 1, 0),
        "coordinate": (1, 0, 0),
        "potential": (2, 0, -2),
        "gradient": (1, 0, -2),
        "c_squared": (2, 0, -2),
        "angle": (0, 0, 0),
    }

    def add(*vectors: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(sum(vector[index] for vector in vectors) for index in range(3))

    def subtract(left: tuple[int, int, int], right: tuple[int, int, int]):
        return tuple(left[index] - right[index] for index in range(3))

    rows = [
        {
            "equation": "nabla2_U_equals_G_rho",
            "left": subtract(dimensions["potential"], (2, 0, 0)),
            "right": add(dimensions["G"], dimensions["rho"]),
        },
        {
            "equation": "matter_acceleration_gradient_Phi",
            "left": subtract(dimensions["potential"], dimensions["coordinate"]),
            "right": dimensions["gradient"],
        },
        {
            "equation": "photon_deflection",
            "left": subtract(
                add(dimensions["gradient"], dimensions["coordinate"]),
                dimensions["c_squared"],
            ),
            "right": dimensions["angle"],
        },
        {
            "equation": "lens_plane_coordinate_Dl_theta",
            "left": add(dimensions["coordinate"], dimensions["angle"]),
            "right": dimensions["coordinate"],
        },
    ]
    for row in rows:
        row["pass"] = row["left"] == row["right"]
    return {
        "all_pass": all(row["pass"] for row in rows),
        "dimension_vector_order": ["L", "M", "T"],
        "rows": rows,
        "constants": config["law_binding"]["constants"],
        "unexplained_numeric_photon_normalization": None,
    }


@dataclass(frozen=True)
class PhysicalState:
    coordinates_kpc: np.ndarray
    cell_kpc: float
    padding_factor: int
    density_msun_kpc3: np.ndarray
    U_km2_s2: np.ndarray
    Y_km2_s2: np.ndarray
    Phi_km2_s2: np.ndarray
    Psi_km2_s2: np.ndarray


def asymmetric_density(
    cells_per_axis: int, physical_extent_kpc: float, total_mass_msun: float
) -> tuple[np.ndarray, np.ndarray, float]:
    _require(cells_per_axis % 2 == 1, "odd cell count required")
    coordinates = np.linspace(
        -physical_extent_kpc / 2.0, physical_extent_kpc / 2.0, cells_per_axis
    )
    cell = float(coordinates[1] - coordinates[0])
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    components = (
        (0.55, (0.38, -0.22, 0.17), (0.85, 1.15, 1.35)),
        (0.30, (-0.71, 0.43, -0.31), (0.70, 0.90, 1.05)),
        (0.15, (0.12, 0.66, 0.28), (1.80, 2.20, 1.60)),
    )
    density = np.zeros((cells_per_axis,) * 3, dtype=float)
    for fraction, centre, sigma in components:
        exponent = (
            ((x - centre[0]) / sigma[0]) ** 2
            + ((y - centre[1]) / sigma[1]) ** 2
            + ((z - centre[2]) / sigma[2]) ** 2
        )
        raw = np.exp(-0.5 * exponent)
        component_mass = fraction * total_mass_msun
        density += component_mass * raw / (raw.sum() * cell**3)
    return coordinates, density, cell


def solve_physical_state(
    density_msun_kpc3: np.ndarray,
    coordinates_kpc: np.ndarray,
    cell_kpc: float,
    *,
    g: float,
    range_kpc: float,
    padding_factor: int,
    config: Mapping[str, Any],
) -> PhysicalState:
    validate_config(config)
    _require(density_msun_kpc3.ndim == 3, "density must be 3D")
    _require(len(set(density_msun_kpc3.shape)) == 1, "density must be cubic")
    _require(density_msun_kpc3.shape[0] % 2 == 1, "odd grid required")
    _require(padding_factor in {2, 4}, "padding factor not frozen")
    _require(range_kpc > 0.0, "range must be positive")
    _require(np.all(np.isfinite(density_msun_kpc3)), "density not finite")
    _require(np.all(density_msun_kpc3 >= 0.0), "density negative")
    n = density_msun_kpc3.shape[0]
    padded_n = padding_factor * (n - 1) + 1
    start = (padded_n - n) // 2
    stop = start + n
    padded = np.zeros((padded_n,) * 3, dtype=float)
    padded[start:stop, start:stop, start:stop] = density_msun_kpc3
    transformed = rfftn(padded)
    wave_xy = 2.0 * np.pi * np.fft.fftfreq(padded_n, d=cell_kpc)
    wave_z = 2.0 * np.pi * np.fft.rfftfreq(padded_n, d=cell_kpc)
    k_squared = (
        wave_xy[:, None, None] ** 2
        + wave_xy[None, :, None] ** 2
        + wave_z[None, None, :] ** 2
    )
    constants = config["law_binding"]["constants"]
    coefficients = config["law_binding"]["coefficients"]
    gravitational_constant = float(constants["G_kpc_km2_s2_Msun"])
    newton_kernel = np.zeros_like(k_squared)
    nonzero = k_squared > 0.0
    newton_kernel[nonzero] = -4.0 * np.pi * gravitational_constant / k_squared[nonzero]
    inverse_range_squared = 1.0 / range_kpc**2
    yukawa_kernel = (
        -4.0 * np.pi * gravitational_constant / (k_squared + inverse_range_squared)
    )
    yukawa_kernel[0, 0, 0] = 0.0
    U_full = irfftn(transformed * newton_kernel, s=padded.shape).real
    Y_full = irfftn(transformed * yukawa_kernel, s=padded.shape).real
    U = U_full[start:stop, start:stop, start:stop]
    Y = Y_full[start:stop, start:stop, start:stop]
    phi = U + float(coefficients["phi_yukawa"]) * g * Y
    psi = U + float(coefficients["psi_yukawa"]) * g * Y
    arrays = [density_msun_kpc3, U, Y, phi, psi]
    for array in arrays:
        array.setflags(write=False)
    coordinates_kpc.setflags(write=False)
    return PhysicalState(
        coordinates_kpc,
        cell_kpc,
        padding_factor,
        density_msun_kpc3,
        U,
        Y,
        phi,
        psi,
    )


def matter_acceleration(
    state: PhysicalState, points_kpc: np.ndarray
) -> np.ndarray:
    gradients = np.gradient(state.Phi_km2_s2, state.cell_kpc, edge_order=2)
    return np.stack(
        [
            RegularGridInterpolator(
                (state.coordinates_kpc,) * 3, -component, bounds_error=True
            )(points_kpc)
            for component in gradients
        ],
        axis=1,
    )


def reduced_photon_deflection(
    state: PhysicalState,
    lens_plane_points_kpc: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    validate_config(config)
    combined = state.Phi_km2_s2 + state.Psi_km2_s2
    gradient_x, gradient_y, _ = np.gradient(combined, state.cell_kpc, edge_order=2)
    integrated_x = np.sum(gradient_x, axis=2) * state.cell_kpc
    integrated_y = np.sum(gradient_y, axis=2) * state.cell_kpc
    constants = config["law_binding"]["constants"]
    coefficient = config["law_binding"]["coefficients"]["photon_integral_prefactor"]
    ratio = config["cosmology"]["frozen_distances"]["D_ls_over_D_s"]
    prefactor = float(coefficient) * float(ratio) / float(constants["c_km_s"]) ** 2
    axes = (state.coordinates_kpc, state.coordinates_kpc)
    alpha_x = prefactor * RegularGridInterpolator(
        axes, integrated_x, bounds_error=True
    )(lens_plane_points_kpc)
    alpha_y = prefactor * RegularGridInterpolator(
        axes, integrated_y, bounds_error=True
    )(lens_plane_points_kpc)
    return np.stack((alpha_x, alpha_y), axis=1)


def extended_source_image(
    state: PhysicalState, image_axis_arcsec: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    constants = config["law_binding"]["constants"]
    distances = config["cosmology"]["frozen_distances"]
    xx_arcsec, yy_arcsec = np.meshgrid(image_axis_arcsec, image_axis_arcsec, indexing="ij")
    theta_arcsec = np.stack((xx_arcsec.ravel(), yy_arcsec.ravel()), axis=1)
    theta_rad = theta_arcsec / float(constants["arcsec_per_radian"])
    lens_points_kpc = theta_rad * float(distances["D_l_Mpc"]) * 1000.0
    alpha_reduced_rad = reduced_photon_deflection(state, lens_points_kpc, config)
    beta_rad = theta_rad - alpha_reduced_rad
    clumps_arcsec = (
        (1.0, -0.25, 0.10, 0.70),
        (0.65, 0.38, -0.22, 0.50),
        (0.40, 0.05, 0.48, 0.38),
    )
    brightness = np.zeros(len(beta_rad), dtype=float)
    arcsec_per_radian = float(constants["arcsec_per_radian"])
    beta_arcsec = beta_rad * arcsec_per_radian
    for amplitude, bx, by, sigma in clumps_arcsec:
        radius_squared = (beta_arcsec[:, 0] - bx) ** 2 + (beta_arcsec[:, 1] - by) ** 2
        brightness += amplitude * np.exp(-0.5 * radius_squared / sigma**2)
    return brightness.reshape(xx_arcsec.shape)


def _relative_rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-300))


def _observable_vector(
    state: PhysicalState,
    points_kpc: np.ndarray,
    image_axis_arcsec: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    matter = matter_acceleration(state, points_kpc)
    lens = reduced_photon_deflection(state, points_kpc[:, :2], config)
    image = extended_source_image(state, image_axis_arcsec, config)
    matter_scale = max(float(np.sqrt(np.mean(matter**2))), 1e-300)
    lens_scale = max(float(np.sqrt(np.mean(lens**2))), 1e-300)
    image_scale = max(float(np.sqrt(np.mean(image**2))), 1e-300)
    return np.concatenate(
        ((matter / matter_scale).ravel(), (lens / lens_scale).ravel(), (image / image_scale).ravel())
    )


def gaussian_log_predictive_density(
    observed: np.ndarray, predicted: np.ndarray, sigma: np.ndarray, indices: np.ndarray
) -> float:
    selected_observed = observed[indices]
    selected_prediction = predicted[indices]
    selected_sigma = sigma[indices]
    _require(np.all(selected_sigma > 0.0), "predictive sigma must be positive")
    standardized = (selected_observed - selected_prediction) / selected_sigma
    return float(
        np.sum(
            -0.5 * standardized**2
            - np.log(selected_sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )
    )


def _synthetic_measurements(
    state: PhysicalState,
    points_kpc: np.ndarray,
    image_axis_arcsec: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, slice]]:
    matter = matter_acceleration(state, points_kpc).ravel()
    lens = reduced_photon_deflection(state, points_kpc[:, :2], config).ravel()
    image = extended_source_image(state, image_axis_arcsec, config).ravel()
    pieces = (matter, lens, image)
    floors = (
        max(float(np.median(np.abs(matter))) * 0.1, 1e-12),
        max(float(np.median(np.abs(lens))) * 0.1, 1e-18),
        0.01,
    )
    fraction = float(config["target_free_gate"]["deterministic_noise_fraction"])
    sigmas = [fraction * np.maximum(np.abs(piece), floor) for piece, floor in zip(pieces, floors)]
    truth = np.concatenate(pieces)
    sigma = np.concatenate(sigmas)
    noise = 0.25 * sigma * np.sin(np.arange(len(truth), dtype=float) * math.sqrt(2.0) + 0.3)
    observed = truth + noise
    matter_stop = len(matter)
    lens_stop = matter_stop + len(lens)
    slices = {
        "matter": slice(0, matter_stop),
        "lensing": slice(matter_stop, lens_stop),
        "extended_image": slice(lens_stop, len(truth)),
    }
    return observed, sigma, slices


def target_free_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    gate = config["target_free_gate"]
    extent = float(gate["physical_extent_kpc"])
    mass = float(gate["total_mass_Msun"])
    coarse_n = int(gate["coarse_cells_per_axis"])
    fine_n = int(gate["fine_cells_per_axis"])
    injected_g = float(gate["injected_g"])
    injected_range = float(gate["injected_range_kpc"])
    coarse_coordinates, coarse_density, coarse_cell = asymmetric_density(coarse_n, extent, mass)
    fine_coordinates, fine_density, fine_cell = asymmetric_density(fine_n, extent, mass)
    _require(math.isclose(coarse_cell, 2.0 * fine_cell, rel_tol=0.0, abs_tol=1e-14),
             "cell size was not halved")
    coarse = solve_physical_state(
        coarse_density,
        coarse_coordinates,
        coarse_cell,
        g=injected_g,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    coarse_pad4 = solve_physical_state(
        coarse_density.copy(),
        coarse_coordinates.copy(),
        coarse_cell,
        g=injected_g,
        range_kpc=injected_range,
        padding_factor=4,
        config=config,
    )
    fine_truth = solve_physical_state(
        fine_density,
        fine_coordinates,
        fine_cell,
        g=injected_g,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    points = np.array(
        [
            [0.8, 0.0, 0.0],
            [1.2, 0.3, 0.0],
            [1.8, -0.4, 0.2],
            [2.4, 0.5, -0.3],
            [-1.1, 0.7, 0.4],
            [0.4, -1.9, -0.5],
        ]
    )
    image_axis = np.linspace(-4.0, 4.0, 33)

    # The fixture itself must be asymmetric before transformed-law tests mean anything.
    asymmetry_errors = {
        f"axis_{axis}": _relative_rms(coarse_density, np.flip(coarse_density, axis=axis))
        for axis in range(3)
    }
    reflection_errors = {}
    for axis in range(3):
        transformed_density = np.flip(coarse_density, axis=axis).copy()
        transformed = solve_physical_state(
            transformed_density,
            coarse_coordinates.copy(),
            coarse_cell,
            g=injected_g,
            range_kpc=injected_range,
            padding_factor=2,
            config=config,
        )
        errors = [
            _relative_rms(np.flip(getattr(coarse, field), axis=axis), getattr(transformed, field))
            for field in ("U_km2_s2", "Y_km2_s2", "Phi_km2_s2", "Psi_km2_s2")
        ]
        reflection_errors[f"axis_{axis}"] = max(errors)
    permutation_errors = {}
    for permutation in itertools.permutations(range(3)):
        transformed_density = np.transpose(coarse_density, axes=permutation).copy()
        transformed = solve_physical_state(
            transformed_density,
            coarse_coordinates.copy(),
            coarse_cell,
            g=injected_g,
            range_kpc=injected_range,
            padding_factor=2,
            config=config,
        )
        errors = [
            _relative_rms(
                np.transpose(getattr(coarse, field), axes=permutation), getattr(transformed, field)
            )
            for field in ("U_km2_s2", "Y_km2_s2", "Phi_km2_s2", "Psi_km2_s2")
        ]
        permutation_errors["".join(str(value) for value in permutation)] = max(errors)

    coarse_observables = _observable_vector(coarse, points, image_axis, config)
    pad4_observables = _observable_vector(coarse_pad4, points, image_axis, config)
    fine_observables = _observable_vector(fine_truth, points, image_axis, config)
    padding_error = _relative_rms(coarse_observables, pad4_observables)
    resolution_error = _relative_rms(fine_observables, coarse_observables)

    fine_gr = solve_physical_state(
        fine_density.copy(),
        fine_coordinates.copy(),
        fine_cell,
        g=0.0,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    fine_short = solve_physical_state(
        fine_density.copy(),
        fine_coordinates.copy(),
        fine_cell,
        g=injected_g,
        range_kpc=1e-4,
        padding_factor=2,
        config=config,
    )
    short_range_error = _relative_rms(
        _observable_vector(fine_gr, points, image_axis, config),
        _observable_vector(fine_short, points, image_axis, config),
    )

    observed, sigma, channel_slices = _synthetic_measurements(
        fine_truth, points, image_axis, config
    )
    gr_prediction = np.concatenate(
        (
            matter_acceleration(fine_gr, points).ravel(),
            reduced_photon_deflection(fine_gr, points[:, :2], config).ravel(),
            extended_source_image(fine_gr, image_axis, config).ravel(),
        )
    )
    unit_state = solve_physical_state(
        fine_density.copy(),
        fine_coordinates.copy(),
        fine_cell,
        g=1.0,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    unit_prediction = np.concatenate(
        (
            matter_acceleration(unit_state, points).ravel(),
            reduced_photon_deflection(unit_state, points[:, :2], config).ravel(),
            extended_source_image(unit_state, image_axis, config).ravel(),
        )
    )
    # The image route is nonlinear in g, so recover g from the two field-linear routes only.
    linear_stop = channel_slices["lensing"].stop
    basis = unit_prediction[:linear_stop] - gr_prediction[:linear_stop]
    weights = 1.0 / sigma[:linear_stop] ** 2
    recovered_g = float(
        np.sum(weights * basis * (observed[:linear_stop] - gr_prediction[:linear_stop]))
        / np.sum(weights * basis**2)
    )
    candidate_state = solve_physical_state(
        fine_density.copy(),
        fine_coordinates.copy(),
        fine_cell,
        g=recovered_g,
        range_kpc=injected_range,
        padding_factor=2,
        config=config,
    )
    candidate_prediction = np.concatenate(
        (
            matter_acceleration(candidate_state, points).ravel(),
            reduced_photon_deflection(candidate_state, points[:, :2], config).ravel(),
            extended_source_image(candidate_state, image_axis, config).ravel(),
        )
    )
    candidate_standardized = (observed - candidate_prediction) / sigma
    gr_standardized = (observed - gr_prediction) / sigma
    candidate_chi2_per_datum = float(np.mean(candidate_standardized**2))
    gr_chi2_per_datum = float(np.mean(gr_standardized**2))
    holdout = np.arange(4, len(observed), 5, dtype=int)
    candidate_lpd = gaussian_log_predictive_density(observed, candidate_prediction, sigma, holdout)
    gr_lpd = gaussian_log_predictive_density(observed, gr_prediction, sigma, holdout)
    candidate_minus_gr_lpd = candidate_lpd - gr_lpd
    channel_residuals = {}
    for channel, selected_slice in channel_slices.items():
        channel_residuals[channel] = {
            "candidate_weighted_rms": float(
                np.sqrt(np.mean(candidate_standardized[selected_slice] ** 2))
            ),
            "gr_weighted_rms": float(np.sqrt(np.mean(gr_standardized[selected_slice] ** 2))),
        }

    mass_observed = float(coarse_density.sum() * coarse_cell**3)
    mass_error = abs(mass_observed - mass) / mass
    audit = unit_audit(config)
    parameters = {
        "matter_acceleration": list(inspect.signature(matter_acceleration).parameters),
        "reduced_photon_deflection": list(
            inspect.signature(reduced_photon_deflection).parameters
        ),
        "extended_source_image": list(inspect.signature(extended_source_image).parameters),
    }
    forbidden_names = {"g", "range_kpc", "photon_multiplier", "lens_multiplier"}
    no_observable_knob = all(
        name not in forbidden_names for names in parameters.values() for name in names
    )
    limits = gate["required"]
    pass_gate = (
        mass_error <= float(limits["mass_relative_error_max"])
        and audit["all_pass"] is True
        and min(asymmetry_errors.values()) > 0.01
        and max(reflection_errors.values())
        <= float(limits["all_three_reflection_relative_error_max"])
        and max(permutation_errors.values())
        <= float(limits["all_six_axis_permutation_relative_error_max"])
        and padding_error <= float(limits["doubled_padding_observable_relative_rms_max"])
        and resolution_error <= float(limits["halved_cell_observable_relative_rms_max"])
        and short_range_error <= float(limits["short_range_to_gr_relative_rms_max"])
        and abs(recovered_g - injected_g) <= float(limits["recovered_g_absolute_error_max"])
        and candidate_chi2_per_datum <= float(limits["candidate_chi2_per_datum_max"])
        and candidate_minus_gr_lpd >= float(limits["candidate_minus_gr_holdout_lpd_min"])
        and no_observable_knob
    )
    return {
        "status": (
            "PASS_PHYSICAL_UNIT_ASYMMETRIC_TARGET_FREE_GATES"
            if pass_gate
            else "FAIL_PHYSICAL_UNIT_ASYMMETRIC_TARGET_FREE_GATES"
        ),
        "pass": pass_gate,
        "units": config["law_binding"]["unit_ledger"],
        "unit_audit": audit,
        "grid": {
            "coarse_cells": coarse_n,
            "fine_cells": fine_n,
            "coarse_cell_kpc": coarse_cell,
            "fine_cell_kpc": fine_cell,
            "cell_ratio": coarse_cell / fine_cell,
            "primary_padding_factor": coarse.padding_factor,
            "comparison_padding_factor": coarse_pad4.padding_factor,
        },
        "metrics": {
            "relative_mass_error": mass_error,
            "asymmetry_relative_errors": asymmetry_errors,
            "all_three_reflection_relative_errors": reflection_errors,
            "all_six_permutation_relative_errors": permutation_errors,
            "doubled_padding_observable_relative_rms": padding_error,
            "halved_cell_observable_relative_rms": resolution_error,
            "short_range_to_gr_observable_relative_rms": short_range_error,
            "injected_g": injected_g,
            "recovered_g": recovered_g,
            "recovered_g_absolute_error": abs(recovered_g - injected_g),
            "candidate_chi2_per_datum_calculated": candidate_chi2_per_datum,
            "gr_chi2_per_datum_calculated": gr_chi2_per_datum,
            "candidate_holdout_log_predictive_density_calculated": candidate_lpd,
            "gr_holdout_log_predictive_density_calculated": gr_lpd,
            "candidate_minus_gr_holdout_lpd": candidate_minus_gr_lpd,
            "channel_weighted_residuals": channel_residuals,
            "observable_function_parameters": parameters,
            "no_observable_specific_law_parameter": no_observable_knob,
        },
        "countermodels_retained": {
            "GR_IDENTICAL_DENSITY": {
                "chi2_per_datum": gr_chi2_per_datum,
                "holdout_lpd": gr_lpd,
            },
            "SPLIT_STATE": "FORBIDDEN_IDENTIFIABILITY_CONTROL_NOT_ADMITTED",
        },
        "claim_boundary": "Synthetic target-free implementation evidence only; not evidence about ESO 325-G004 or gravity.",
    }


def source_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    readiness = config["reduction_readiness"]
    return {
        "status": readiness["current_status"],
        "missing_external_inputs": readiness[
            "missing_external_inputs_before_any_scientific_array_decode"
        ],
        "deferred_empirical_gates": readiness[
            "empirical_gates_only_after_a_separately_authorized_development_array_decode"
        ],
        "external_missing_count": 4,
        "post_decode_gate_count": 5,
        "scientific_array_elements_decoded": 0,
        "eso_response_values_opened": 0,
        "eso_score_computed": False,
        "slacs_response_values_opened": 0,
    }


def build_artifacts(config: Mapping[str, Any]) -> dict[str, bytes]:
    preservation = verify_v4_preservation_and_sources(config)
    units = unit_audit(config)
    target_gate = target_free_gate(config)
    readiness = source_readiness(config)
    report = (
        "# Lane 7 V5 strict repair\n\n"
        f"- V4 preservation and exact sources: **{preservation['status']}**.\n"
        f"- Physical-unit law audit: **{'PASS' if units['all_pass'] else 'FAIL'}**.\n"
        f"- Asymmetric target-free gates: **{target_gate['status']}**.\n"
        f"- ESO readiness: **{readiness['status']}**.\n"
        "- Scientific FITS arrays decoded: **0**.\n"
        "- SLACS response values opened: **0**.\n\n"
        "V5 uses G and c in declared physical units and the reduced deflection "
        "(D_ls/D_s)c^-2 integral grad(Phi+Psi) dl; it has no arbitrary photon normalization. "
        "The source block now names only genuinely missing external bindings. Field-star, LSF, "
        "registration and covariance checks are correctly deferred empirical gates, not mislabeled "
        "as missing files. The target-free pass is software evidence only.\n"
    ).encode()
    return {
        "v4-preservation-and-exact-source-receipt.json": _json_bytes(preservation),
        "physical-unit-and-distance-audit.json": _json_bytes(
            {"unit_audit": units, "angular_diameter_distances": angular_diameter_distances(config)}
        ),
        "target-free-physical-shared-state-gate.json": _json_bytes(target_gate),
        "source-readiness-split.json": _json_bytes(readiness),
        "frozen-density-and-predictive-contract.json": _json_bytes(
            {
                "extended_density_contract": config["extended_density_contract"],
                "shared_state_contract": config["shared_state_contract"],
                "predictive_likelihood_contract": config["predictive_likelihood_contract"],
                "contract_hashes": config["contract_hashes"],
            }
        ),
        "report.md": report,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    artifacts = build_artifacts(config)
    preservation = json.loads(artifacts["v4-preservation-and-exact-source-receipt.json"])
    target_gate = json.loads(artifacts["target-free-physical-shared-state-gate.json"])
    readiness = json.loads(artifacts["source-readiness-split.json"])
    _require(preservation["status"].startswith("PASS_"), "preservation failed")
    _require(target_gate["pass"] is True, "target-free gate failed")
    _require(readiness["status"].startswith("SOURCE_BLOCKED_"), "readiness widened")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-same-law-eso325-extended-source-receipt-5.0",
        "package_id": config["package_id"],
        "status": "SOURCE_BLOCKED_EXTERNAL_INPUTS_AFTER_PHYSICAL_TARGET_FREE_PASS",
        "decision": "NO_ESO_ARRAY_DECODE_NO_ESO_SCORE_KEEP_SLACS_SEALED",
        "v4_preservation_and_sources": preservation,
        "law_binding_sha256": config["law_sha256"],
        "source_binding_sha256": config["source_binding_sha256"],
        "contract_hashes": config["contract_hashes"],
        "angular_diameter_distances": angular_diameter_distances(config),
        "target_free_gate": target_gate,
        "source_readiness": readiness,
        "claim_boundary": {
            "establishes": [
                "V4 is byte-exact preserved as blocked evidence",
                "the exact V5 law, metric, coefficients, constants, units and sources reject mutation",
                "the physical reduced-deflection implementation contains no arbitrary photon normalization",
                "an asymmetric synthetic density passes all-axis reflection/permutation, doubled-padding and halved-cell gates",
                "candidate residuals and held-out predictive density are calculated rather than assigned",
                "the pseudo-NFW and scientific held-out predictive metric are fully specified",
            ],
            "does_not_establish": [
                "a decode, reduction, fit or score for ESO 325-G004",
                "a reproduction of the 2018 paper likelihood",
                "evidence for the Yukawa comparator or modified gravity",
                "a nonlinear healthy theory completion",
                "any result on a SLACS response",
            ],
        },
        "access_accounting": config["access_accounting"],
        "artifact_manifest": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(artifacts.items())
        },
        "artifact_bindings": {
            "config": {
                "path": CONFIG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": file_sha256(CONFIG_PATH),
            },
            "module": {
                "path": MODULE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": file_sha256(MODULE_PATH),
            },
            "test": {
                "path": TEST_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": file_sha256(TEST_PATH),
            },
        },
    }
    receipt["content_sha256"] = _canonical_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing artifact differs: {path}")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        _require(path.read_bytes() == payload, f"concurrent artifact differs: {path}")
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
