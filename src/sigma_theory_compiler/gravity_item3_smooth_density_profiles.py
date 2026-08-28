"""Frozen smooth-profile derivation and source boundary for gravity Item 3 attempt 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item3_smooth_density_profiles_v2.json"
SAMPLE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-03-smooth-density-profiles-v2-source/sample-manifest.json"
)
SOURCE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-03-smooth-density-profiles-v2-source/"
    "exploration-source-manifest.json"
)
RADIAL_FEATURE_PATH = (
    "runs/gravity/roadmap/item-03-smooth-density-profiles-v2-source/"
    "radial-features.tsv"
)

# X-COP gas-density conversion used by the primary-source cluster literature
# (rho_gas = mu_e m_p n_e).  It is a unit/composition convention, not a fitted
# parameter.  Pratt et al. 2022, A&A 665 A24 quote mu_e=1.148.
MEAN_MOLECULAR_WEIGHT_PER_ELECTRON = 1.148


class GravityItem3SmoothDensityError(RuntimeError):
    """Raised when the frozen attempt-2 derivation or source boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(root: Path) -> dict[str, Any]:
    """Load the frozen contract and verify its immutable predecessors."""

    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item3-smooth-density-profiles-config-2.0"
    ):
        raise GravityItem3SmoothDensityError("unexpected Item 3 attempt-2 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem3SmoothDensityError("stable roadmap binding changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem3SmoothDensityError("attempt-1 receipt file changed")
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor_receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem3SmoothDensityError("attempt-1 receipt content changed")
    if predecessor_receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem3SmoothDensityError("attempt-1 decision changed")
    authorization = config["authorization"]
    if authorization["paid_model_calls_allowed"]:
        raise GravityItem3SmoothDensityError("paid model calls are forbidden")
    if authorization["reserved_xcop_confirmation_profile_accesses_allowed"] != 0:
        raise GravityItem3SmoothDensityError("confirmation access budget must be zero")
    if config["cluster_lane"]["reserved_profile_accesses_allowed"] != 0:
        raise GravityItem3SmoothDensityError("cluster confirmation access budget drifted")
    return config


def effective_density_pair(
    radius: np.ndarray,
    q_density: np.ndarray,
    *,
    support_dimension: int,
    gravity_constant: float,
    transition_acceleration: float,
    scale_clip: tuple[float, float] = (0.05, 20.0),
) -> dict[str, np.ndarray]:
    """Convert a smooth radial d-density into common surface/volume sources.

    ``q_density`` has units mass/length**d and ``radius`` uses the matching
    length unit.  ``gravity_constant`` must therefore use compatible mass,
    length, acceleration units.  No dynamics target enters this function.
    """

    radius = np.asarray(radius, dtype=np.float64)
    q_density = np.asarray(q_density, dtype=np.float64)
    if (
        support_dimension not in {2, 3}
        or radius.ndim != 1
        or radius.shape != q_density.shape
        or radius.size < 5
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(q_density))
        or np.any(radius <= 0)
        or np.any(q_density <= 0)
        or np.any(np.diff(radius) <= 0)
        or not math.isfinite(gravity_constant)
        or gravity_constant <= 0
        or not math.isfinite(transition_acceleration)
        or transition_acceleration <= 0
    ):
        raise GravityItem3SmoothDensityError("invalid smooth baryonic profile")
    lower, upper = (float(value) for value in scale_clip)
    if not 0 < lower < upper:
        raise GravityItem3SmoothDensityError("invalid scale-length clip")

    log_gradient = np.gradient(np.log(q_density), radius, edge_order=2)
    raw_h = np.divide(
        1.0,
        np.abs(log_gradient),
        out=np.full_like(log_gradient, np.inf),
        where=np.abs(log_gradient) > 0.0,
    )
    raw_h_over_r = raw_h / radius
    h_over_r = np.clip(raw_h_over_r, lower, upper)
    scale_length = h_over_r * radius
    sigma_eff = q_density * (2.0 * scale_length) ** (support_dimension - 2)
    rho_eff = q_density / (2.0 * scale_length) ** (3 - support_dimension)
    u_surface = (
        math.pi * gravity_constant * sigma_eff / transition_acceleration
    )
    u_volume = (
        4.0
        * math.pi
        * gravity_constant
        * radius
        * rho_eff
        / (3.0 * transition_acceleration)
    )
    if np.any(~np.isfinite(u_surface)) or np.any(~np.isfinite(u_volume)):
        raise GravityItem3SmoothDensityError("non-finite density source")
    if np.any(u_surface <= 0) or np.any(u_volume <= 0):
        raise GravityItem3SmoothDensityError("non-positive density source")

    s = np.log10(u_surface)
    v = np.log10(u_volume)
    mean = 0.5 * (s + v)
    contrast = s - v
    surface_transition = u_surface / (1.0 + u_surface) ** 2
    volume_transition = u_volume / (1.0 + u_volume) ** 2
    transition_sum = surface_transition + volume_transition
    values = {
        "scale_length": scale_length,
        "scale_clip_mask": raw_h_over_r != h_over_r,
        "sigma_eff": sigma_eff,
        "rho_eff": rho_eff,
        "u_surface": u_surface,
        "u_volume": u_volume,
        "s": s,
        "v": v,
        "m": mean,
        "c": contrast,
        "m_x_c": mean * contrast,
        "transition_product": surface_transition * volume_transition,
        "transition_balance": (surface_transition - volume_transition)
        / transition_sum,
    }
    if any(np.any(~np.isfinite(value)) for value in values.values()):
        raise GravityItem3SmoothDensityError("non-finite derived density feature")
    return values


def radial_feature_basis(
    *,
    gbar: np.ndarray,
    density_pair: Mapping[str, np.ndarray],
    transition_acceleration: float,
    population_proxy: float,
) -> dict[str, np.ndarray]:
    """Build the frozen model basis without accepting a dynamics target."""

    gbar = np.asarray(gbar, dtype=np.float64)
    if (
        gbar.ndim != 1
        or np.any(~np.isfinite(gbar))
        or np.any(gbar <= 0)
        or not math.isfinite(population_proxy)
    ):
        raise GravityItem3SmoothDensityError("invalid baryonic acceleration")
    required = {"s", "v", "m", "c", "m_x_c", "transition_product", "transition_balance"}
    if not required.issubset(density_pair):
        raise GravityItem3SmoothDensityError("incomplete density pair")
    if any(np.asarray(density_pair[key]).shape != gbar.shape for key in required):
        raise GravityItem3SmoothDensityError("density/baryonic profile mismatch")
    a = np.log10(gbar / float(transition_acceleration))
    mean = np.asarray(density_pair["m"], dtype=np.float64)
    contrast = np.asarray(density_pair["c"], dtype=np.float64)
    return {
        "a": a,
        "a2": a**2,
        "a3": a**3,
        "s": np.asarray(density_pair["s"], dtype=np.float64),
        "v": np.asarray(density_pair["v"], dtype=np.float64),
        "m": mean,
        "c": contrast,
        "m_x_c": np.asarray(density_pair["m_x_c"], dtype=np.float64),
        "a_x_m": a * mean,
        "a_x_c": a * contrast,
        "transition_product": np.asarray(
            density_pair["transition_product"], dtype=np.float64
        ),
        "transition_balance": np.asarray(
            density_pair["transition_balance"], dtype=np.float64
        ),
        "population_proxy": np.full_like(a, float(population_proxy)),
    }


def _interpolate_density_pair(
    source_radius: np.ndarray,
    density_pair: Mapping[str, np.ndarray],
    target_radius: np.ndarray,
) -> dict[str, np.ndarray]:
    source_radius = np.asarray(source_radius, dtype=np.float64)
    target_radius = np.asarray(target_radius, dtype=np.float64)
    if (
        source_radius.ndim != 1
        or target_radius.ndim != 1
        or np.any(np.diff(source_radius) <= 0)
        or np.any(target_radius < source_radius[0])
        or np.any(target_radius > source_radius[-1])
    ):
        raise GravityItem3SmoothDensityError("invalid density interpolation range")
    interpolated: dict[str, np.ndarray] = {}
    log_radius = np.log(source_radius)
    log_target = np.log(target_radius)
    for key, raw in density_pair.items():
        if key == "scale_clip_mask":
            continue
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != source_radius.shape:
            raise GravityItem3SmoothDensityError("invalid density-pair shape")
        if key in {"scale_length", "sigma_eff", "rho_eff", "u_surface", "u_volume"}:
            interpolated[key] = np.exp(np.interp(log_target, log_radius, np.log(values)))
        else:
            interpolated[key] = np.interp(log_target, log_radius, values)
    return interpolated


def _canonical_galaxy_name(value: str) -> str:
    return value.strip().replace(" ", "")


def _parse_leroy_profiles(path: Path, admitted: set[str]) -> dict[str, dict[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith("recno\tName\t"))
    except StopIteration as exc:
        raise GravityItem3SmoothDensityError("Leroy table header changed") from exc
    grouped: defaultdict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    for row in csv.DictReader(lines[start:], delimiter="\t"):
        name = _canonical_galaxy_name(str(row.get("Name", "")))
        if name not in admitted:
            continue
        try:
            radius = float(row["r"])
            sigma_star = float(row["Sigma*"])
        except (TypeError, ValueError) as exc:
            raise GravityItem3SmoothDensityError(
                f"missing required Leroy density for {name}"
            ) from exc
        raw_hi = str(row.get("SigmaHI", "")).strip()
        raw_h2 = str(row.get("SigmaH2", "")).strip()
        sigma_hi = float(raw_hi) if raw_hi else 0.0
        sigma_h2 = float(raw_h2) if raw_h2 else 0.0
        grouped[name].append(
            (
                radius,
                sigma_hi + sigma_h2 + sigma_star,
                int(not raw_h2),
                int(not raw_hi or not raw_h2),
            )
        )
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for name in sorted(admitted):
        rows = sorted(grouped.get(name, []))
        if not rows:
            raise GravityItem3SmoothDensityError(f"missing Leroy profile: {name}")
        radius = np.asarray([row[0] for row in rows], dtype=np.float64)
        density = np.asarray([row[1] for row in rows], dtype=np.float64)
        if len(set(radius)) != len(radius):
            raise GravityItem3SmoothDensityError(f"duplicate Leroy radius: {name}")
        profiles[name] = {
            "radius_kpc": radius,
            "sigma_msun_pc2": density,
            "missing_h2_fraction": np.asarray(
                [sum(row[2] for row in rows) / len(rows)], dtype=np.float64
            ),
            "missing_gas_component_fraction": np.asarray(
                [sum(row[3] for row in rows) / len(rows)], dtype=np.float64
            ),
        }
    return profiles


def _parse_sparc_profile(path: Path) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise GravityItem3SmoothDensityError(f"SPARC row schema changed: {path.name}")
        rows.append([float(value) for value in fields])
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or len(values) < 5:
        raise GravityItem3SmoothDensityError(f"invalid SPARC profile: {path.name}")
    return {
        "radius_kpc": values[:, 0],
        "vobs_km_s": values[:, 1],
        "err_v_km_s": values[:, 2],
        "vgas_km_s": values[:, 3],
        "vdisk_km_s": values[:, 4],
        "vbulge_km_s": values[:, 5],
    }


def _cumulative_spherical_mass(radius_m: np.ndarray, rho_kg_m3: np.ndarray) -> np.ndarray:
    radius_m = np.asarray(radius_m, dtype=np.float64)
    rho_kg_m3 = np.asarray(rho_kg_m3, dtype=np.float64)
    integrand = 4.0 * math.pi * rho_kg_m3 * radius_m**2
    mass = np.empty_like(radius_m)
    mass[0] = 4.0 * math.pi * rho_kg_m3[0] * radius_m[0] ** 3 / 3.0
    mass[1:] = mass[0] + np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(radius_m)
    )
    return mass


def _read_xcop_cluster(
    directory: Path,
    cluster: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    density_path = directory / cluster / f"{cluster}_density_L1.fits"
    pressure_path = directory / cluster / f"{cluster}_pressure.fits"
    with fits.open(density_path, memmap=False) as handle:
        density_hdu = handle[2]
        if list(density_hdu.columns.names) != ["RW_X", "NE", "ERR_NE_LO", "ERR_NE_HI"]:
            raise GravityItem3SmoothDensityError(f"X-COP density schema changed: {cluster}")
        r500_kpc = float(density_hdu.header["R500"])
        density_radius_kpc = np.asarray(density_hdu.data["RW_X"], dtype=np.float64) * r500_kpc
        ne_cm3 = np.asarray(density_hdu.data["NE"], dtype=np.float64)
    with fits.open(pressure_path, memmap=False) as handle:
        pressure_hdu = handle[2]
        if list(pressure_hdu.columns.names) != ["RW_SZ", "P_SZ", "eP_SZ"]:
            raise GravityItem3SmoothDensityError(f"X-COP pressure schema changed: {cluster}")
        if not math.isclose(float(pressure_hdu.header["R500"]), r500_kpc):
            raise GravityItem3SmoothDensityError(f"X-COP R500 mismatch: {cluster}")
        pressure_radius_kpc = (
            np.asarray(pressure_hdu.data["RW_SZ"], dtype=np.float64) * r500_kpc
        )
        p500_kev_cm3 = float(pressure_hdu.header["P500"])
        pressure_kev_cm3 = (
            np.asarray(pressure_hdu.data["P_SZ"], dtype=np.float64) * p500_kev_cm3
        )
        pressure_error_kev_cm3 = (
            np.asarray(pressure_hdu.data["eP_SZ"], dtype=np.float64) * p500_kev_cm3
        )
    if (
        np.any(np.diff(density_radius_kpc) <= 0)
        or np.any(np.diff(pressure_radius_kpc) <= 0)
        or np.any(ne_cm3 <= 0)
        or np.any(pressure_kev_cm3 <= 0)
    ):
        raise GravityItem3SmoothDensityError(f"invalid X-COP radial profile: {cluster}")

    constants = config["constants"]
    kpc_m = float(constants["kiloparsec_m"])
    proton_mass = float(constants["proton_mass_kg"])
    solar_mass = float(constants["solar_mass_kg"])
    gravity_si = float(constants["gravity_constant_si"])
    transition = float(constants["transition_acceleration_m_s2"])
    density_radius_m = density_radius_kpc * kpc_m
    rho_gas = ne_cm3 * 1.0e6 * MEAN_MOLECULAR_WEIGHT_PER_ELECTRON * proton_mass
    gas_mass = _cumulative_spherical_mass(density_radius_m, rho_gas)
    density_pair = effective_density_pair(
        density_radius_m,
        rho_gas,
        support_dimension=3,
        gravity_constant=gravity_si,
        transition_acceleration=transition,
    )

    overlap = (pressure_radius_kpc >= density_radius_kpc[0]) & (
        pressure_radius_kpc <= density_radius_kpc[-1]
    )
    target_radius_kpc = pressure_radius_kpc[overlap]
    target_radius_m = target_radius_kpc * kpc_m
    target_pressure = pressure_kev_cm3[overlap]
    target_pressure_error = pressure_error_kev_cm3[overlap]
    ne_target = np.exp(
        np.interp(
            np.log(target_radius_kpc),
            np.log(density_radius_kpc),
            np.log(ne_cm3),
        )
    )
    gas_mass_target = np.interp(target_radius_m, density_radius_m, gas_mass)
    pair_target = _interpolate_density_pair(
        density_radius_m, density_pair, target_radius_m
    )
    log_pressure_slope = np.gradient(
        np.log(target_pressure), np.log(target_radius_m), edge_order=2
    )
    pressure_j_m3 = target_pressure * float(constants["kev_joule"]) * 1.0e6
    gdyn = -(
        pressure_j_m3
        * log_pressure_slope
        / (
            float(constants["mean_molecular_weight"])
            * proton_mass
            * ne_target
            * 1.0e6
            * target_radius_m
        )
    )
    gbar = gravity_si * gas_mass_target / target_radius_m**2
    gbar_with_stars: np.ndarray | None = None
    basis_with_stars: dict[str, np.ndarray] | None = None
    stellar_path = directory / cluster / f"{cluster}_mstar.fits"
    if stellar_path.exists():
        with fits.open(stellar_path, memmap=False) as handle:
            stellar_hdu = handle[2]
            stellar_radius_kpc = np.asarray(
                stellar_hdu.data["RADIUS"], dtype=np.float64
            )
            stellar_mass_kg = (
                np.asarray(stellar_hdu.data["MSTAR"], dtype=np.float64) * solar_mass
            )
        stellar_mass_kg = np.maximum.accumulate(stellar_mass_kg)
        stellar_mass_density_grid = np.interp(
            density_radius_kpc,
            stellar_radius_kpc,
            stellar_mass_kg,
            left=stellar_mass_kg[0],
            right=stellar_mass_kg[-1],
        )
        stellar_density = np.maximum(
            np.gradient(stellar_mass_density_grid, density_radius_m, edge_order=2)
            / (4.0 * math.pi * density_radius_m**2),
            0.0,
        )
        stellar_mass_target = np.interp(
            target_radius_kpc,
            stellar_radius_kpc,
            stellar_mass_kg,
            left=stellar_mass_kg[0],
            right=stellar_mass_kg[-1],
        )
        gbar_with_stars = gravity_si * (
            gas_mass_target + stellar_mass_target
        ) / target_radius_m**2
        total_pair = effective_density_pair(
            density_radius_m,
            rho_gas + stellar_density,
            support_dimension=3,
            gravity_constant=gravity_si,
            transition_acceleration=transition,
        )
        total_pair_target = _interpolate_density_pair(
            density_radius_m, total_pair, target_radius_m
        )
        basis_with_stars = radial_feature_basis(
            gbar=gbar_with_stars,
            density_pair=total_pair_target,
            transition_acceleration=transition,
            population_proxy=1.0,
        )

    basis = radial_feature_basis(
        gbar=gbar,
        density_pair=pair_target,
        transition_acceleration=transition,
        population_proxy=1.0,
    )
    return {
        "radius_kpc": target_radius_kpc,
        "gbar": gbar,
        "gbar_with_stars": gbar_with_stars,
        "basis_with_stars": basis_with_stars,
        "gdyn": gdyn,
        "pressure_fractional_error": target_pressure_error / target_pressure,
        "pressure_decreasing_fraction": float(np.mean(np.diff(pressure_kev_cm3) < 0)),
        "density_points": len(density_radius_kpc),
        "pressure_points": len(pressure_radius_kpc),
        "scale_clip_fraction": float(np.mean(density_pair["scale_clip_mask"])),
        "missing_h2_fraction": 0.0,
        "missing_gas_component_fraction": 0.0,
        "basis": basis,
        "mu_e": MEAN_MOLECULAR_WEIGHT_PER_ELECTRON,
    }


def _read_galaxy(
    name: str,
    leroy: Mapping[str, Mapping[str, np.ndarray]],
    sparc_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    profile = leroy[name]
    dynamics = _parse_sparc_profile(sparc_dir / f"{name}_rotmod.dat")
    density_radius_kpc = np.asarray(profile["radius_kpc"], dtype=np.float64)
    sigma_msun_pc2 = np.asarray(profile["sigma_msun_pc2"], dtype=np.float64)
    target_radius_kpc_all = np.asarray(dynamics["radius_kpc"], dtype=np.float64)
    overlap = (target_radius_kpc_all >= density_radius_kpc[0]) & (
        target_radius_kpc_all <= density_radius_kpc[-1]
    )
    constants = config["constants"]
    kpc_m = float(constants["kiloparsec_m"])
    solar_mass = float(constants["solar_mass_kg"])
    transition = float(constants["transition_acceleration_m_s2"])
    gravity_si = float(constants["gravity_constant_si"])
    radius_m = density_radius_kpc * kpc_m
    parsec_m = kpc_m / 1000.0
    sigma_kg_m2 = sigma_msun_pc2 * solar_mass / parsec_m**2
    pair = effective_density_pair(
        radius_m,
        sigma_kg_m2,
        support_dimension=2,
        gravity_constant=gravity_si,
        transition_acceleration=transition,
    )
    target_radius_kpc = target_radius_kpc_all[overlap]
    target_radius_m = target_radius_kpc * kpc_m
    pair_target = _interpolate_density_pair(radius_m, pair, target_radius_m)
    vobs = np.asarray(dynamics["vobs_km_s"], dtype=np.float64)[overlap]
    err_v = np.asarray(dynamics["err_v_km_s"], dtype=np.float64)[overlap]
    vgas = np.asarray(dynamics["vgas_km_s"], dtype=np.float64)[overlap]
    vdisk = np.asarray(dynamics["vdisk_km_s"], dtype=np.float64)[overlap]
    vbulge = np.asarray(dynamics["vbulge_km_s"], dtype=np.float64)[overlap]
    baryonic_v2 = (
        np.sign(vgas) * vgas**2
        + float(constants["fixed_sparc_disk_mass_to_light"]) * vdisk**2
        + float(constants["fixed_sparc_bulge_mass_to_light"]) * vbulge**2
    )
    acceleration_conversion = 1.0e6 / kpc_m
    gbar = baryonic_v2 / target_radius_kpc * acceleration_conversion
    gdyn = vobs**2 / target_radius_kpc * acceleration_conversion
    basis = radial_feature_basis(
        gbar=gbar,
        density_pair=pair_target,
        transition_acceleration=transition,
        population_proxy=0.0,
    )
    return {
        "radius_kpc": target_radius_kpc,
        "gbar": gbar,
        "gbar_with_stars": None,
        "basis_with_stars": None,
        "gdyn": gdyn,
        "velocity_fractional_error": err_v / vobs,
        "density_points": len(density_radius_kpc),
        "pressure_points": 0,
        "pressure_decreasing_fraction": 1.0,
        "scale_clip_fraction": float(np.mean(pair["scale_clip_mask"])),
        "missing_h2_fraction": float(profile["missing_h2_fraction"][0]),
        "missing_gas_component_fraction": float(
            profile["missing_gas_component_fraction"][0]
        ),
        "basis": basis,
    }


def _quality_reason(
    domain: str, profile: Mapping[str, Any], config: Mapping[str, Any]
) -> str | None:
    quality = config["radial_quality"]
    if domain == "galaxy":
        if int(profile["density_points"]) < int(quality["minimum_galaxy_density_points"]):
            return "insufficient_density_points"
        if len(profile["radius_kpc"]) < int(quality["minimum_galaxy_matched_dynamics_points"]):
            return "insufficient_matched_dynamics_points"
        if float(np.median(profile["velocity_fractional_error"])) > float(
            quality["maximum_median_fractional_velocity_error"]
        ):
            return "velocity_error_too_large"
    else:
        if int(profile["density_points"]) < int(quality["minimum_cluster_density_points"]):
            return "insufficient_density_points"
        if int(profile["pressure_points"]) < int(quality["minimum_cluster_pressure_points"]):
            return "insufficient_pressure_points"
        if len(profile["radius_kpc"]) < int(quality["minimum_cluster_matched_points"]):
            return "insufficient_matched_points"
        if float(profile["pressure_decreasing_fraction"]) < float(
            quality["minimum_fraction_pressure_steps_decreasing"]
        ):
            return "pressure_not_mostly_decreasing"
    if float(profile["scale_clip_fraction"]) > float(quality["maximum_scale_clip_fraction"]):
        return "scale_clip_fraction_too_large"
    if np.any(np.asarray(profile["gbar"]) <= 0) or np.any(np.asarray(profile["gdyn"]) <= 0):
        return "nonpositive_acceleration"
    return None


def build_radial_features(root: Path, raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build permitted exploration rows; confirmation names are rejected on sight."""

    root = root.resolve()
    raw_dir = raw_dir.resolve()
    config = load_config(root)
    manifest = build_sample_manifest(root)
    raw_cluster_names = {path.name for path in (raw_dir / "xcop").iterdir() if path.is_dir()}
    forbidden = raw_cluster_names & set(manifest["cluster_reserved_confirmation"])
    if forbidden:
        raise GravityItem3SmoothDensityError(
            f"confirmation cluster extracted: {sorted(forbidden)}"
        )
    expected_clusters = set(manifest["cluster_exploration"])
    if raw_cluster_names != expected_clusters:
        raise GravityItem3SmoothDensityError("exploration cluster extraction set drifted")
    galaxy_names = set(manifest["galaxy_development"])
    leroy = _parse_leroy_profiles(raw_dir / "leroy-table7.tsv", galaxy_names)

    profiles: list[tuple[str, str, dict[str, Any]]] = []
    for name in sorted(galaxy_names):
        profiles.append(("galaxy", name, _read_galaxy(name, leroy, raw_dir / "sparc", config)))
    for name in sorted(expected_clusters):
        profiles.append(("cluster", name, _read_xcop_cluster(raw_dir / "xcop", name, config)))

    rows: list[dict[str, Any]] = []
    quality_records: list[dict[str, Any]] = []
    for domain, name, profile in profiles:
        reason = _quality_reason(domain, profile, config)
        quality_records.append(
            {
                "domain": domain,
                "name": name,
                "quality_pass": reason is None,
                "failure_reason": reason,
                "density_points": int(profile["density_points"]),
                "matched_points": len(profile["radius_kpc"]),
                "scale_clip_fraction": f"{float(profile['scale_clip_fraction']):.12e}",
                "missing_h2_fraction": f"{float(profile['missing_h2_fraction']):.12e}",
                "missing_gas_component_fraction": (
                    f"{float(profile['missing_gas_component_fraction']):.12e}"
                ),
            }
        )
        if reason is not None:
            continue
        variants: list[tuple[str, np.ndarray, Mapping[str, np.ndarray]]] = [
            ("primary", profile["gbar"], profile["basis"])
        ]
        if profile["gbar_with_stars"] is not None:
            variants.append(
                (
                    "stellar_augmented",
                    profile["gbar_with_stars"],
                    profile["basis_with_stars"],
                )
            )
        for variant, variant_gbar, basis in variants:
            for index, radius in enumerate(profile["radius_kpc"]):
                row: dict[str, Any] = {
                    "variant": variant,
                    "domain": domain,
                    "name": name,
                    "radius_kpc": float(radius),
                    "gbar_m_s2": float(variant_gbar[index]),
                    "gdyn_m_s2": float(profile["gdyn"][index]),
                    "response_log10_ratio": float(
                        math.log10(profile["gdyn"][index] / variant_gbar[index])
                    ),
                    "has_stellar_robustness": int(profile["gbar_with_stars"] is not None),
                    "gbar_with_stars_m_s2": (
                        float(profile["gbar_with_stars"][index])
                        if profile["gbar_with_stars"] is not None
                        else float("nan")
                    ),
                }
                row.update({key: float(value[index]) for key, value in basis.items()})
                rows.append(row)
    valid_galaxies = sum(
        record["quality_pass"] and record["domain"] == "galaxy"
        for record in quality_records
    )
    valid_clusters = sum(
        record["quality_pass"] and record["domain"] == "cluster"
        for record in quality_records
    )
    quality_pass = valid_galaxies >= int(config["radial_quality"]["minimum_valid_galaxies"]) and valid_clusters >= int(
        config["radial_quality"]["minimum_valid_exploration_clusters"]
    )
    summary = {
        "schema_version": "invariant-gravity-item3-smooth-density-extraction-summary-2.0",
        "quality_pass": bool(quality_pass),
        "valid_galaxies": int(valid_galaxies),
        "valid_clusters": int(valid_clusters),
        "radial_rows": len(rows),
        "quality_records": quality_records,
        "confirmation_profiles_opened": 0,
        "mu_e_unit_convention": f"{MEAN_MOLECULAR_WEIGHT_PER_ELECTRON:.12e}",
    }
    return rows, summary


def _metric(value: float) -> str:
    return f"{float(value):.12e}"


def write_exploration_sources(root: Path, raw_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    raw_dir = raw_dir.resolve()
    config = load_config(root)
    rows, summary = build_radial_features(root, raw_dir)
    output = root / RADIAL_FEATURE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "domain",
        "name",
        "radius_kpc",
        "gbar_m_s2",
        "gdyn_m_s2",
        "response_log10_ratio",
        "has_stellar_robustness",
        "gbar_with_stars_m_s2",
    ] + sorted(
        key
        for key in rows[0]
        if key
        not in {
            "domain",
            "name",
            "variant",
            "radius_kpc",
            "gbar_m_s2",
            "gdyn_m_s2",
            "response_log10_ratio",
            "has_stellar_robustness",
            "gbar_with_stars_m_s2",
        }
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        row[key]
                        if key in {"variant", "domain", "name", "has_stellar_robustness"}
                        else _metric(row[key])
                    )
                    for key in fields
                }
            )
    raw_files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    source_manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-smooth-density-source-manifest-2.0",
        "goal": config["goal"],
        "freeze_commit": "68487eb9d9f00adfb1238c8f904b12e76b9bc9c2",
        "files": [
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in raw_files
        ],
        "file_count": len(raw_files),
        "raw_bytes": sum(path.stat().st_size for path in raw_files),
        "extraction": summary,
        "reserved_confirmation_profiles_opened": 0,
        "content_sha256": None,
    }
    content = dict(source_manifest)
    content.pop("content_sha256")
    source_manifest["content_sha256"] = canonical_sha256(content)
    manifest_path = root / SOURCE_MANIFEST_PATH
    manifest_path.write_bytes(canonical_json_bytes(source_manifest) + b"\n")
    return output, manifest_path


def _salted_order(names: Sequence[str], *, salt: str, stratum: str) -> list[str]:
    return sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{salt}|{stratum}|{name}".encode()
        ).hexdigest(),
    )


def build_sample_manifest(root: Path) -> dict[str, Any]:
    """Build the target-blind manifest solely from frozen names and source metadata."""

    root = root.resolve()
    config = load_config(root)
    lane = config["cluster_lane"]
    with_stars = sorted(
        set(lane["exploration_with_stellar_profile"])
        | set(lane["reserved_with_stellar_profile"])
    )
    without_stars = sorted(
        set(lane["exploration_without_stellar_profile"])
        | set(lane["reserved_without_stellar_profile"])
    )
    salt = str(lane["selection_salt"])
    expected_with = _salted_order(
        with_stars, salt=salt, stratum="with_mstar"
    )
    expected_without = _salted_order(
        without_stars, salt=salt, stratum="without_mstar"
    )
    if set(expected_with[:5]) != set(lane["exploration_with_stellar_profile"]):
        raise GravityItem3SmoothDensityError("with-star exploration split drifted")
    if set(expected_with[5:]) != set(lane["reserved_with_stellar_profile"]):
        raise GravityItem3SmoothDensityError("with-star confirmation split drifted")
    if set(expected_without[:3]) != set(lane["exploration_without_stellar_profile"]):
        raise GravityItem3SmoothDensityError("without-star exploration split drifted")
    if set(expected_without[3:]) != set(lane["reserved_without_stellar_profile"]):
        raise GravityItem3SmoothDensityError("without-star confirmation split drifted")

    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-smooth-density-sample-manifest-2.0",
        "goal": config["goal"],
        "selection_used_response_values": False,
        "galaxy_development": list(config["galaxy_lane"]["matched_development_objects"]),
        "galaxy_independent_confirmation": [],
        "cluster_exploration": sorted(lane["exploration_objects"]),
        "cluster_reserved_confirmation": sorted(lane["reserved_confirmation_objects"]),
        "cluster_selection_salt": salt,
        "cluster_selection_strata": {
            "with_mstar": {
                "salted_order": expected_with,
                "exploration_count": 5,
                "reserved_count": 2,
            },
            "without_mstar": {
                "salted_order": expected_without,
                "exploration_count": 3,
                "reserved_count": 2,
            },
        },
        "counts": {
            "galaxy_development": len(
                config["galaxy_lane"]["matched_development_objects"]
            ),
            "cluster_exploration": len(lane["exploration_objects"]),
            "cluster_reserved_confirmation": len(lane["reserved_confirmation_objects"]),
        },
        "confirmation_access_budget": 0,
        "content_sha256": None,
    }
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = canonical_sha256(content)
    return manifest


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    destination = root / SAMPLE_MANIFEST_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return destination


def verify_local_source_archives(root: Path, cache_dir: Path) -> dict[str, Any]:
    """Verify archive hashes and names without extracting any profile rows."""

    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    xcop_path = cache_dir / "xcop-allfiles.tar.gz"
    sparc_path = cache_dir / "sparc-rotmod.zip"
    if _sha256_file(xcop_path) != config["cluster_lane"]["archive_sha256"]:
        raise GravityItem3SmoothDensityError("X-COP archive hash changed")
    if _sha256_file(sparc_path) != config["galaxy_lane"]["dynamics_catalog"][
        "archive_sha256"
    ]:
        raise GravityItem3SmoothDensityError("SPARC archive hash changed")
    with tarfile.open(xcop_path, mode="r:gz") as archive:
        xcop_names = set(archive.getnames())
    with zipfile.ZipFile(sparc_path) as archive:
        sparc_names = set(archive.namelist())
    clusters = config["cluster_lane"]["exploration_objects"] + config["cluster_lane"][
        "reserved_confirmation_objects"
    ]
    for cluster in clusters:
        for suffix in ("density_L1.fits", "pressure.fits"):
            expected = f"{cluster}/{cluster}_{suffix}"
            if expected not in xcop_names:
                raise GravityItem3SmoothDensityError(f"missing X-COP member: {expected}")
    for galaxy in config["galaxy_lane"]["matched_development_objects"]:
        expected = f"{galaxy}_rotmod.dat"
        if expected not in sparc_names:
            raise GravityItem3SmoothDensityError(f"missing SPARC member: {expected}")
    return {
        "xcop_archive_sha256": _sha256_file(xcop_path),
        "xcop_member_count": len(xcop_names),
        "sparc_archive_sha256": _sha256_file(sparc_path),
        "sparc_member_count": len(sparc_names),
        "profile_rows_opened": 0,
        "confirmation_profiles_opened": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write-sample-manifest")
    verify = subparsers.add_parser("verify-archives")
    verify.add_argument("--cache-dir", type=Path, required=True)
    build = subparsers.add_parser("build-exploration-sources")
    build.add_argument("--raw-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if args.command == "write-sample-manifest":
        print(write_sample_manifest(root))
    elif args.command == "verify-archives":
        print(json.dumps(verify_local_source_archives(root, args.cache_dir), sort_keys=True))
    elif args.command == "build-exploration-sources":
        print(
            "\n".join(
                str(path)
                for path in write_exploration_sources(root, args.raw_dir)
            )
        )
    else:  # pragma: no cover
        raise GravityItem3SmoothDensityError("unknown command")


if __name__ == "__main__":
    main()
