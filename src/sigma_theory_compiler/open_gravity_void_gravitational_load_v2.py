"""Repaired 3-D void-load laws and response-unopened CF4/VAST preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

CONFIG_PATH = Path("configs/open_gravity_void_gravitational_load_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_gravitational_load_v2.py")
TEST_PATH = Path("tests/test_open_gravity_void_gravitational_load_v2.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-gravitational-load-v2/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_SCHEMA = "invariant-open-gravity-void-gravitational-load-2.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-void-gravitational-load-receipt-2.0"
_CONFIG_RAW_SHA256 = "93004d40c7da85ecfda7a0fab9f99ae1860faa64cd7b6c187b81800e8fb534e6"
_CONFIG_CONTENT_SHA256 = "33e091a436267f2308b473770346a6a51ca5794e2519478e18bf562a512a1d30"
_MODULE_SEMANTIC_SHA256 = "afda3ead274a44e69478c770b865b26da314abdc6c481f31040503c7c6ade956"
_TEST_RAW_SHA256 = "25f932a51189b7b88a69e7c63d3947dcbbb4b902b0714e98cfa5b1e56d8f4cdc"
_BRANCH_IDS = tuple(f"VQ{index:02d}" for index in range(11))

Array = NDArray[np.float64]


class VoidLoadV2Error(RuntimeError):
    """Raised when a frozen v2 void-load invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidLoadV2Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in (
        "_CONFIG_RAW_SHA256",
        "_CONFIG_CONTENT_SHA256",
        "_MODULE_SEMANTIC_SHA256",
        "_TEST_RAW_SHA256",
    ):
        marker = f'{name} = "'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        text = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoidLoadV2Error(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config.get("schema") == _SCHEMA, "schema changed")
    _require(config.get("package_id") == "open-gravity-void-gravitational-load-v2", "ID changed")
    _require(
        config.get("status") == "FROZEN_REPAIRED_3D_LAWS_SOURCE_BYTES_READY_ROWS_UNOPENED",
        "status changed",
    )
    branch_ids = tuple(str(row["id"]).split("_")[0] for row in config["branches"])
    _require(branch_ids == _BRANCH_IDS, "branch registry changed")
    _require(config["synthetic_grid"]["shape"] == [9, 9, 9], "grid changed")
    _require(
        config["geometry_contract"]["spatial_ray_arclength"].startswith("dell_b=sqrt"),
        "physical path measure changed",
    )
    _require(
        "once, and only once" in config["geometry_contract"]["rule"],
        "clock conversion changed",
    )
    dimensions = config["dimensions"]
    _require(dimensions["photon_eta"] == [0, 0, -1], "photon eta units changed")
    _require(dimensions["sigma_g"] == [2, -1, 0], "column opacity units changed")
    params = config["parameters"]
    for key in (
        "eta",
        "beta",
        "H_ref",
        "Gamma_0",
        "Gamma_b",
        "A_feed",
        "A_load",
        "L_g",
        "D_g",
        "sigma_g",
        "photon_a",
        "photon_b0",
        "photon_b_b",
        "photon_eta",
    ):
        _require(type(params[key]) in (int, float) and math.isfinite(params[key]), f"bad {key}")
    _require(params["c"] > 0 and params["L_g"] > 0, "invalid positive scale")
    _require(params["diffusion_dt"] > 0 and params["diffusion_steps"] > 0, "bad IVP grid")
    real = config["real_data_contract"]
    _require(
        "published Vpec columns forbidden" in real["response"], "response leakage guard changed"
    )
    _require(real["status"].endswith("EXECUTOR_NOT_YET_AUTHORED"), "source gate changed")
    access = config["access_accounting"]
    _require(access["scientific_rows_decoded_by_this_packet"] == 0, "rows opened")
    _require(access["response_values_inspected_by_this_packet"] == 0, "responses inspected")
    _require(access["real_scores"] == 0, "real scores claimed")
    claims = config["claim_boundary"]
    for key in (
        "scientific_rows_opened",
        "real_data_fit",
        "covariant_action",
        "known_physics_consistency",
        "novelty",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim widened: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH, "v2 config")
    _require(type(config) is dict, "config must be an object")
    validate_config(config)
    return config


def _validate_predecessors(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for label in ("supersedes", "source_packet"):
        row = config[label]
        path = Path(row["path"])
        _require(path.is_file(), f"missing {label}")
        observed[label] = file_sha256(path)
        _require(observed[label] == row["raw_sha256"], f"{label} raw bytes changed")
        value = _read_json(path, label)
        _require(value.get("content_sha256") == row["content_sha256"], f"{label} seal changed")
        _require(_self_hash(value) == row["content_sha256"], f"{label} self-hash invalid")
    source = _read_json(Path(config["source_packet"]["path"]), "source packet")
    _require(
        source["source_bundle_root_sha256"] == config["source_packet"]["source_bundle_root_sha256"],
        "source bundle root changed",
    )
    return observed


def fixture_densities(config: Mapping[str, Any]) -> dict[str, Array]:
    shape = tuple(config["synthetic_grid"]["shape"])
    x, y, z = np.indices(shape, dtype=float)
    homogeneous = np.ones(shape, dtype=float)
    void = np.full(shape, 0.12, dtype=float)
    void += 2.4 * np.exp(-((x - 2.0) ** 2 + (y - 2.0) ** 2 + (z - 6.0) ** 2) / 1.6)
    void += 1.8 * np.exp(-((x - 6.0) ** 2 + (y - 6.0) ** 2 + (z - 2.0) ** 2) / 2.2)
    bar = 0.08 + 1.8 * np.exp(-((x - 4.0) ** 2 / 8.0 + (y - 4.0) ** 2 / 0.7 + (z - 4.0) ** 2 / 1.2))
    bar += 1.2 * np.exp(-((x - 7.0) ** 2 + (y - 6.0) ** 2 + (z - 5.0) ** 2) / 0.8)
    zero = np.zeros(shape, dtype=float)
    return {
        "HOMOGENEOUS": homogeneous,
        "VOID_WITH_TWO_OFF_AXIS_SOURCES": void,
        "ASYMMETRIC_BAR_AND_CLUMP": bar,
        "ZERO_SOURCE": zero,
    }


def periodic_laplacian(field: Array, dx: float) -> Array:
    result = np.zeros_like(field)
    for axis in range(3):
        result += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2.0 * field
    return result / dx**2


def helmholtz_feed(rho: Array, *, amplitude: float, length: float, dx: float) -> Array:
    _require(rho.ndim == 3, "Helmholtz input must be 3-D")
    if length == 0.0:
        return amplitude * rho.copy()
    wave_axes = [2.0 * math.pi * np.fft.fftfreq(size, d=dx) for size in rho.shape]
    kx, ky, kz = np.meshgrid(*wave_axes, indexing="ij")
    denominator = 1.0 + length**2 * (kx**2 + ky**2 + kz**2)
    return np.fft.ifftn(amplitude * np.fft.fftn(rho) / denominator).real


def helmholtz_spectral_residual(
    field: Array, rho: Array, *, amplitude: float, length: float, dx: float
) -> float:
    wave_axes = [2.0 * math.pi * np.fft.fftfreq(size, d=dx) for size in rho.shape]
    kx, ky, kz = np.meshgrid(*wave_axes, indexing="ij")
    lhs = (1.0 + length**2 * (kx**2 + ky**2 + kz**2)) * np.fft.fftn(field)
    rhs = amplitude * np.fft.fftn(rho)
    scale = max(float(np.max(np.abs(rhs))), 1.0)
    return float(np.max(np.abs(lhs - rhs)) / scale)


def local_equilibrium(rho: Array, *, source: Array, gamma0: float, gamma_b: float) -> Array:
    denominator = gamma0 + gamma_b * rho
    _require(bool(np.all(denominator > 0.0)), "nonpositive local relaxation")
    return source / denominator


def solve_diffusive_reservoir(
    rho: Array,
    *,
    source: Array,
    diffusivity: float,
    gamma0: float,
    gamma_b: float,
    dx: float,
    dt: float,
    steps: int,
) -> Array:
    _require(rho.ndim == 3 and source.shape == rho.shape, "reservoir shape mismatch")
    _require(diffusivity >= 0.0 and dt > 0.0 and steps > 0, "invalid reservoir grid")
    if diffusivity == 0.0:
        return local_equilibrium(rho, source=source, gamma0=gamma0, gamma_b=gamma_b)
    gamma = gamma0 + gamma_b * rho
    q = np.zeros_like(rho)

    def rhs(state: Array) -> Array:
        return diffusivity * periodic_laplacian(state, dx) + source - gamma * state

    for _ in range(steps):
        k1 = rhs(q)
        k2 = rhs(q + 0.5 * dt * k1)
        k3 = rhs(q + 0.5 * dt * k2)
        k4 = rhs(q + dt * k3)
        q += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return q


def _trilinear(field: Array, points: Array) -> Array:
    upper = np.asarray(field.shape, dtype=float) - 1.0
    clipped = np.clip(points, 0.0, upper)
    lo = np.floor(clipped).astype(int)
    hi = np.minimum(lo + 1, np.asarray(field.shape) - 1)
    frac = clipped - lo
    values = np.zeros(points.shape[0], dtype=float)
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                ix = hi[:, 0] if bx else lo[:, 0]
                iy = hi[:, 1] if by else lo[:, 1]
                iz = hi[:, 2] if bz else lo[:, 2]
                weight = (
                    (frac[:, 0] if bx else 1.0 - frac[:, 0])
                    * (frac[:, 1] if by else 1.0 - frac[:, 1])
                    * (frac[:, 2] if bz else 1.0 - frac[:, 2])
                )
                values += weight * field[ix, iy, iz]
    return values


def column_attenuated_feed(
    rho: Array,
    *,
    amplitude: float,
    length: float,
    sigma: float,
    dx: float,
    samples: int,
) -> Array:
    _require(rho.ndim == 3 and length > 0.0 and samples >= 3, "invalid column solve")
    coordinates = np.stack(np.indices(rho.shape), axis=-1).reshape(-1, 3).astype(float)
    masses = rho.reshape(-1) * dx**3
    output = np.zeros(coordinates.shape[0], dtype=float)
    self_radius = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * dx
    fractions = np.linspace(0.0, 1.0, samples)
    for target_index, target in enumerate(coordinates):
        delta = target[None, :] - coordinates
        distance = np.linalg.norm(delta, axis=1) * dx
        kernel_radius = np.where(distance > 0.0, distance, self_radius)
        kernel = np.exp(-kernel_radius / length) / (4.0 * math.pi * length**2 * kernel_radius)
        columns = np.zeros_like(distance)
        nonself = distance > 0.0
        if np.any(nonself):
            segment_points = coordinates[nonself, None, :] + fractions[None, :, None] * (
                target[None, None, :] - coordinates[nonself, None, :]
            )
            sampled = _trilinear(rho, segment_points.reshape(-1, 3)).reshape(-1, samples)
            columns[nonself] = np.trapezoid(sampled, fractions, axis=1) * distance[nonself]
        attenuation = np.exp(-sigma * columns)
        output[target_index] = amplitude * float(np.sum(masses * kernel * attenuation))
    return output.reshape(rho.shape)


def central_ray(field: Array) -> Array:
    _require(field.shape == (9, 9, 9), "central ray requires frozen cube")
    return field[:, 4, 4].copy()


def path_integral(values: Array, dx: float) -> float:
    return float(np.trapezoid(values, dx=dx))


def direct_log_shift(q_ray: Array, *, eta: float, c: float, dx: float) -> float:
    return -(eta / c) * path_integral(q_ray, dx)


def slowed_ray_observables(
    q_ray: Array, *, eta: float, beta: float, c: float, dx: float
) -> tuple[float, float]:
    _require(bool(np.all(1.0 + beta * q_ray > 0.0)), "nonpositive ray speed")
    i1 = path_integral(q_ray, dx)
    i2 = path_integral(q_ray**2, dx)
    return -(eta / c) * (i1 + beta * i2), (beta / c) * i1


def photon_memory_log_shift(
    q_ray: Array,
    rho_ray: Array,
    *,
    drive: float,
    relax0: float,
    relax_b: float,
    eta: float,
    c: float,
    dx: float,
) -> tuple[float, Array]:
    _require(q_ray.shape == rho_ray.shape and q_ray.ndim == 1, "memory ray mismatch")
    state = np.zeros_like(q_ray)
    dt = dx / c
    for index in range(q_ray.size - 1):
        q_mid = 0.5 * (q_ray[index] + q_ray[index + 1])
        rho_mid = 0.5 * (rho_ray[index] + rho_ray[index + 1])
        rate = relax0 + relax_b * rho_mid
        _require(rate > 0.0, "nonpositive photon relaxation")
        equilibrium = drive * q_mid / rate
        state[index + 1] = equilibrium + (state[index] - equilibrium) * math.exp(-rate * dt)
    return -eta * path_integral(state, dx) / c, state


def _field_bundle(rho: Array, config: Mapping[str, Any]) -> dict[str, Array]:
    p = config["parameters"]
    dx = float(config["synthetic_grid"]["dx"])
    source = p["A_feed"] * rho
    local = local_equilibrium(rho, source=source, gamma0=p["Gamma_0"], gamma_b=p["Gamma_b"])
    feed = helmholtz_feed(rho, amplitude=p["A_feed"], length=p["L_g"], dx=dx)
    helmholtz_q = local_equilibrium(rho, source=feed, gamma0=p["Gamma_0"], gamma_b=p["Gamma_b"])
    reservoir = solve_diffusive_reservoir(
        rho,
        source=source,
        diffusivity=p["D_g"],
        gamma0=p["Gamma_0"],
        gamma_b=p["Gamma_b"],
        dx=dx,
        dt=p["diffusion_dt"],
        steps=p["diffusion_steps"],
    )
    column = column_attenuated_feed(
        rho,
        amplitude=p["A_load"],
        length=p["L_g"],
        sigma=p["sigma_g"],
        dx=dx,
        samples=p["column_samples_per_segment"],
    )
    rho_mean = float(np.mean(rho))
    inverse = ((rho_mean + p["rho_star"]) / (rho + p["rho_star"])) ** p["inverse_density_n"]
    return {
        "external": rho.copy(),
        "local": local,
        "helmholtz_feed": feed,
        "helmholtz_q": helmholtz_q,
        "reservoir": reservoir,
        "column": column,
        "inverse": inverse,
        "source": source,
    }


def synthetic_report(config: Mapping[str, Any]) -> dict[str, Any]:
    p = config["parameters"]
    dx = float(config["synthetic_grid"]["dx"])
    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    fixtures = fixture_densities(config)
    for fixture_id, rho in fixtures.items():
        fields = _field_bundle(rho, config)
        rho_ray = central_ray(rho)
        branch_fields = {
            "VQ01": fields["external"],
            "VQ02": fields["external"],
            "VQ03": fields["local"],
            "VQ04": fields["helmholtz_q"],
            "VQ05": fields["reservoir"],
            "VQ06": fields["column"],
            "VQ07": fields["inverse"],
            "VQ09": fields["reservoir"],
            "VQ10": fields["reservoir"],
        }
        predictions.append(
            {
                "fixture_id": fixture_id,
                "branch_id": "VQ00",
                "log_frequency_shift": 0.0,
                "extra_delay": 0.0,
                "field_max": 0.0,
            }
        )
        for branch_id, field in branch_fields.items():
            ray = central_ray(field)
            delay = 0.0
            if branch_id == "VQ02":
                shift, delay = slowed_ray_observables(
                    ray, eta=p["eta"], beta=p["beta"], c=p["c"], dx=dx
                )
            elif branch_id == "VQ07":
                shift = direct_log_shift(ray, eta=p["H_g"], c=p["c"], dx=dx)
            elif branch_id == "VQ09":
                shift, _ = photon_memory_log_shift(
                    ray,
                    rho_ray,
                    drive=p["photon_a"],
                    relax0=p["photon_b0"],
                    relax_b=p["photon_b_b"],
                    eta=p["photon_eta"],
                    c=p["c"],
                    dx=dx,
                )
            elif branch_id == "VQ10":
                shift, delay = slowed_ray_observables(
                    ray, eta=p["H_ref"] * p["beta"], beta=p["beta"], c=p["c"], dx=dx
                )
            else:
                shift = direct_log_shift(ray, eta=p["eta"], c=p["c"], dx=dx)
            predictions.append(
                {
                    "fixture_id": fixture_id,
                    "branch_id": branch_id,
                    "log_frequency_shift": float(shift),
                    "extra_delay": float(delay),
                    "field_max": float(np.max(np.abs(field))),
                }
            )
        path_length = dx * (rho_ray.size - 1)
        void_fraction = float(np.mean(rho_ray < p["void_threshold"]))
        vq08_shift = (
            -(
                p["H_v"] * path_length * void_fraction
                + p["H_m"] * path_length * (1.0 - void_fraction)
            )
            / p["c"]
        )
        predictions.append(
            {
                "fixture_id": fixture_id,
                "branch_id": "VQ08",
                "log_frequency_shift": float(vq08_shift),
                "extra_delay": 0.0,
                "field_max": void_fraction,
            }
        )
        helmholtz_residual = helmholtz_spectral_residual(
            fields["helmholtz_feed"],
            rho,
            amplitude=p["A_feed"],
            length=p["L_g"],
            dx=dx,
        )
        steady_residual = p["D_g"] * periodic_laplacian(fields["reservoir"], dx)
        steady_residual += (
            fields["source"] - (p["Gamma_0"] + p["Gamma_b"] * rho) * fields["reservoir"]
        )
        steady_scale = max(float(np.max(np.abs(fields["source"]))), 1.0)
        diagnostics.append(
            {
                "fixture_id": fixture_id,
                "helmholtz_relative_residual": helmholtz_residual,
                "reservoir_steady_relative_residual": float(
                    np.max(np.abs(steady_residual)) / steady_scale
                ),
            }
        )
    return {"predictions": predictions, "diagnostics": diagnostics}


def exact_gate_report(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    p = config["parameters"]
    dx = float(config["synthetic_grid"]["dx"])
    rho = fixture_densities(config)["ASYMMETRIC_BAR_AND_CLUMP"]
    fields = _field_bundle(rho, config)
    ray = central_ray(fields["reservoir"])
    rho_ray = central_ray(rho)
    direct = direct_log_shift(ray, eta=p["eta"], c=p["c"], dx=dx)
    beta_zero, beta_zero_delay = slowed_ray_observables(
        ray, eta=p["eta"], beta=0.0, c=p["c"], dx=dx
    )
    feed_zero = helmholtz_feed(rho, amplitude=p["A_feed"], length=0.0, dx=dx)
    local = local_equilibrium(
        rho, source=p["A_feed"] * rho, gamma0=p["Gamma_0"], gamma_b=p["Gamma_b"]
    )
    diffusion_zero = solve_diffusive_reservoir(
        rho,
        source=p["A_feed"] * rho,
        diffusivity=0.0,
        gamma0=p["Gamma_0"],
        gamma_b=p["Gamma_b"],
        dx=dx,
        dt=p["diffusion_dt"],
        steps=p["diffusion_steps"],
    )
    column_zero = column_attenuated_feed(
        rho,
        amplitude=p["A_load"],
        length=p["L_g"],
        sigma=0.0,
        dx=dx,
        samples=p["column_samples_per_segment"],
    )
    memory_forward, _ = photon_memory_log_shift(
        ray,
        rho_ray,
        drive=p["photon_a"],
        relax0=p["photon_b0"],
        relax_b=p["photon_b_b"],
        eta=p["photon_eta"],
        c=p["c"],
        dx=dx,
    )
    memory_reverse, _ = photon_memory_log_shift(
        ray[::-1],
        rho_ray[::-1],
        drive=p["photon_a"],
        relax0=p["photon_b0"],
        relax_b=p["photon_b_b"],
        eta=p["photon_eta"],
        c=p["c"],
        dx=dx,
    )
    optical_forward = path_integral(ray, dx)
    optical_reverse = path_integral(ray[::-1], dx)
    tied_shift, tied_delay = slowed_ray_observables(
        ray, eta=p["H_ref"] * p["beta"], beta=p["beta"], c=p["c"], dx=dx
    )
    i1 = path_integral(ray, dx)
    i2 = path_integral(ray**2, dx)
    tied_expected = p["H_ref"] * p["beta"] * (i1 + p["beta"] * i2) / p["c"]
    zero = np.zeros((9, 9, 9), dtype=float)
    zero_fields = _field_bundle(zero, config)
    zero_max = max(
        float(np.max(np.abs(zero_fields[key])))
        for key in (
            "external",
            "local",
            "helmholtz_feed",
            "helmholtz_q",
            "reservoir",
            "column",
            "source",
        )
    )
    checks = [
        ("CLOCK_TO_SPACE_ONCE_BETA0", beta_zero == direct and beta_zero_delay == 0.0, 0.0),
        (
            "HELMHOLTZ_3D_OPERATOR",
            helmholtz_spectral_residual(
                fields["helmholtz_feed"],
                rho,
                amplitude=p["A_feed"],
                length=p["L_g"],
                dx=dx,
            )
            < 1e-12,
            helmholtz_spectral_residual(
                fields["helmholtz_feed"],
                rho,
                amplitude=p["A_feed"],
                length=p["L_g"],
                dx=dx,
            ),
        ),
        (
            "HELMHOLTZ_L0_BOUNDARY",
            np.array_equal(feed_zero, p["A_feed"] * rho),
            float(np.max(np.abs(feed_zero - p["A_feed"] * rho))),
        ),
        (
            "DIFFUSION_D0_LOCAL_BOUNDARY",
            np.array_equal(diffusion_zero, local),
            float(np.max(np.abs(diffusion_zero - local))),
        ),
        (
            "COLUMN_SIGMA0_FINITE",
            bool(np.all(np.isfinite(column_zero))) and float(np.max(column_zero)) > 0.0,
            float(np.max(column_zero)),
        ),
        (
            "STATIC_REVERSAL_INVARIANCE",
            optical_forward == optical_reverse,
            abs(optical_forward - optical_reverse),
        ),
        (
            "MEMORY_ORDER_SENSITIVITY",
            abs(memory_forward - memory_reverse) > 1e-6,
            abs(memory_forward - memory_reverse),
        ),
        (
            "TIED_REDSHIFT_DELAY_IDENTITY",
            tied_delay > 0.0 and math.isclose(-tied_shift, tied_expected, abs_tol=1e-14),
            abs(-tied_shift - tied_expected),
        ),
        (
            "ZERO_SOURCE_AND_ZERO_COUPLING_NULLS",
            zero_max == 0.0 and direct_log_shift(ray, eta=0.0, c=p["c"], dx=dx) == 0.0,
            zero_max,
        ),
        ("RESPONSES_UNOPENED", True, 0.0),
    ]
    return [
        {"check_id": check_id, "passed": bool(passed), "diagnostic": float(diagnostic)}
        for check_id, passed, diagnostic in checks
    ]


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _report_markdown(report: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]) -> bytes:
    max_helmholtz = max(row["helmholtz_relative_residual"] for row in report["diagnostics"])
    max_reservoir = max(row["reservoir_steady_relative_residual"] for row in report["diagnostics"])
    memory = next(row for row in gates if row["check_id"] == "MEMORY_ORDER_SENSITIVITY")
    tied = next(row for row in gates if row["check_id"] == "TIED_REDSHIFT_DELAY_IDENTITY")
    text = f"""# Repaired void gravitational-load laws v2

This packet repairs the v1 geometry and operator defects before any CF4 or VAST scientific row is decoded.

## What is established

- Eleven reduced branches are dimensioned and executable on four 9x9x9 target-free cubes.
- VQ04 solves a true periodic three-dimensional Helmholtz equation; maximum spectral residual is `{max_helmholtz:.6g}`.
- VQ05 solves a three-dimensional causal diffusion/source/sink initial-value problem; maximum final steady residual is `{max_reservoir:.6g}`.
- VQ06 evaluates source-to-target columns through all three coordinates on a finite cube.
- Static scalar exposure is reversal-invariant, while the executable photon-memory ODE changes by `{memory["diagnostic"]:.6g}` when the same samples are reversed.
- VQ10 uses one beta and one solved Q field for both delay and redshift; identity error is `{tied["diagnostic"]:.6g}`.

## What is not established

No CF4 or VAST scientific row was decoded, no observed velocity or distance was inspected, and no real score was computed. These reduced laws are not a covariant action, a consistency proof, a novelty result, or evidence for an anomalous void effect. The next packet must freeze the CF4/VAST row parser, masks, nuisance model, effect grid, likelihood and split before decoding scientific tables.
"""
    return text.encode()


def _artifact_payloads(
    report: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]
) -> dict[str, bytes]:
    return {
        (ARTIFACT_DIR / "predictions.csv").as_posix(): _csv_bytes(report["predictions"]),
        (ARTIFACT_DIR / "operator-diagnostics.json").as_posix(): _pretty(report["diagnostics"]),
        (ARTIFACT_DIR / "exact-gates.json").as_posix(): _pretty(list(gates)),
        (ARTIFACT_DIR / "report.md").as_posix(): _report_markdown(report, gates),
    }


def build_receipt() -> tuple[dict[str, Any], dict[str, bytes]]:
    config = load_config()
    predecessors = _validate_predecessors(config)
    report = synthetic_report(config)
    gates = exact_gate_report(config)
    _require(all(row["passed"] for row in gates), "one or more exact gates failed")
    payloads = _artifact_payloads(report, gates)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_REPAIRED_3D_REDUCED_LAWS_SOURCE_BYTES_READY_ROWS_UNOPENED",
        "decision": "ADVANCE_VQ08_TO_SEPARATELY_FROZEN_CF4_VAST_ROW_PARSER_AND_TARGET_BLIND_SCORE",
        "content_sha256": "",
        "bindings": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
            **predecessors,
        },
        "counts": {
            "branches": len(config["branches"]),
            "fixtures": len(fixture_densities(config)),
            "predictions": len(report["predictions"]),
            "operator_diagnostics": len(report["diagnostics"]),
            "exact_gates": len(gates),
            "gates_passed": sum(bool(row["passed"]) for row in gates),
        },
        "operator_diagnostics": report["diagnostics"],
        "exact_gates": gates,
        "access_accounting": config["access_accounting"],
        "real_data_contract": config["real_data_contract"],
        "claim_boundary": config["claim_boundary"],
        "artifact_index": [
            {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for path, payload in sorted(payloads.items())
        ],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    validate_receipt(receipt)
    return receipt, payloads


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    _require(receipt.get("schema") == _RECEIPT_SCHEMA, "receipt schema changed")
    _require(_self_hash(receipt) == receipt.get("content_sha256"), "receipt self-hash invalid")
    bindings = receipt["bindings"]
    _require(bindings["config_raw_sha256"] == _CONFIG_RAW_SHA256, "config raw pin changed")
    _require(bindings["config_content_sha256"] == _CONFIG_CONTENT_SHA256, "config seal changed")
    _require(bindings["module_semantic_sha256"] == _MODULE_SEMANTIC_SHA256, "module changed")
    _require(bindings["test_raw_sha256"] == _TEST_RAW_SHA256, "tests changed")
    _require(
        receipt["counts"]
        == {
            "branches": 11,
            "fixtures": 4,
            "predictions": 44,
            "operator_diagnostics": 4,
            "exact_gates": 10,
            "gates_passed": 10,
        },
        "receipt counts changed",
    )
    _require(len(receipt["artifact_index"]) == 4, "artifact inventory changed")
    _require(
        receipt["access_accounting"]["scientific_rows_decoded_by_this_packet"] == 0, "rows opened"
    )
    _require(receipt["claim_boundary"]["real_data_fit"] is False, "real fit overclaim")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing to overwrite {path}")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, f"concurrent output differs: {path}")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_package() -> str:
    receipt, payloads = build_receipt()
    for path_text, payload in sorted(payloads.items()):
        _atomic_no_clobber(Path(path_text), payload)
    return _atomic_no_clobber(OUTPUT_PATH, _pretty(receipt))


def check_package() -> dict[str, Any]:
    _require(OUTPUT_PATH.is_file(), "canonical receipt missing")
    observed = _read_json(OUTPUT_PATH, "receipt")
    _require(type(observed) is dict, "receipt must be an object")
    validate_receipt(observed)
    expected, payloads = build_receipt()
    _require(observed == expected, "receipt does not reproduce")
    for path_text, payload in payloads.items():
        path = Path(path_text)
        _require(path.is_file() and path.read_bytes() == payload, f"artifact changed: {path}")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(write_package())
        return 0
    receipt = check_package()
    if args.command == "check":
        print("VALID")
    else:
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "branches": receipt["counts"]["branches"],
                    "gates": receipt["counts"]["gates_passed"],
                    "scientific_rows_decoded": receipt["access_accounting"][
                        "scientific_rows_decoded_by_this_packet"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
