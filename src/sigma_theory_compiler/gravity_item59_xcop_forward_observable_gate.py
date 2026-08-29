"""Item 59: forward-predict X-COP SZ pressure and X-ray temperature."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_g4_first_principles_mechanism_search import (
    _kernel_matrix,
    _log_radius_cell_widths,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item59_xcop_forward_observable_gate_v1.json")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM58_PATH = Path("runs/gravity/roadmap/item-58-cluster-coefficient-gate-v1.json")


class GravityItem59Error(RuntimeError):
    """Raised when the Item 59 freeze, source boundary, or replay changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item59-xcop-forward-observable-config-1.0"
        or config.get("item") != 59
        or config.get("status") != "scientific_freeze_before_reserved_xcop_response_access"
    ):
        raise GravityItem59Error("unsupported Item 59 config")
    freeze = str(config.get("scientific_freeze_commit", ""))
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem59Error("Item 59 scientific freeze is not bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem59Error("invalid Item 59 freeze marker")
    for relative, expected in config["scientific_dependencies"].items():
        path = root / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise GravityItem59Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / ITEM58_PATH)
    required = config["required_predecessor"]
    if (
        predecessor.get("decision") != required["decision"]
        or predecessor.get("claims", {}).get("formula_or_feature_family_pruned") is not False
    ):
        raise GravityItem59Error("Item 58 predecessor boundary changed")
    population = config["population"]
    development = list(map(str, population["development_clusters_already_exposed"]))
    confirmation = list(
        map(str, population["independent_confirmation_clusters_sealed_until_freeze"])
    )
    with_stars = set(map(str, population["clusters_with_published_stellar_profile"]))
    without_stars = set(
        map(str, population["clusters_without_published_stellar_profile"])
    )
    if (
        len(development) != 8
        or len(confirmation) != 4
        or set(development) & set(confirmation)
        or with_stars & without_stars
        or with_stars | without_stars != set(development) | set(confirmation)
    ):
        raise GravityItem59Error("Item 59 sample split changed")
    if population["confirmation_response_rows_allowed_before_freeze"] is not False:
        raise GravityItem59Error("pre-freeze confirmation response access is forbidden")
    if (
        int(population["direct_lensing_likelihood_evaluations_allowed"]) != 0
        or int(population["inferred_total_mass_rows_allowed"]) != 0
        or config["observable_contract"]["total_mass_used_anywhere"] is not False
        or config["observable_contract"]["pressure_gradient_or_hydrostatic_mass_used_as_target"]
        is not False
    ):
        raise GravityItem59Error("forbidden total-mass or lensing access enabled")
    counter = config["counterexample_policy"]
    if (
        counter["single_counterexample_terminal"] is not False
        or counter["counterexample_count_alone_terminal"] is not False
        or counter["finite_sample_may_prune_formula_family"] is not False
        or counter["global_family_pruning_allowed"] is not False
    ):
        raise GravityItem59Error("empirical over-pruning is forbidden")
    families = list(config["law_families"])
    if len(families) != 6 or len({str(row["id"]) for row in families}) != 6:
        raise GravityItem59Error("Item 59 law-family set changed")
    if sum(bool(row["qualifying"]) for row in families) != 4:
        raise GravityItem59Error("Item 59 qualifying-family boundary changed")
    nuisance = config["nuisance_grid"]
    if nuisance["all_nuisances_global_not_per_cluster"] is not True:
        raise GravityItem59Error("per-cluster fitted nuisance is forbidden")


def _all_clusters(config: Mapping[str, Any]) -> list[str]:
    population = config["population"]
    return [
        *map(str, population["development_clusters_already_exposed"]),
        *map(str, population["independent_confirmation_clusters_sealed_until_freeze"]),
    ]


def _expected_members(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    with_stars = set(
        map(str, config["population"]["clusters_with_published_stellar_profile"])
    )
    rows: dict[str, dict[str, str]] = {}
    for cluster in _all_clusters(config):
        members = {
            "density": f"{cluster}/{cluster}_density_L1.fits",
            "pressure": f"{cluster}/{cluster}_pressure.fits",
            "temperature": f"{cluster}/{cluster}_temperature.fits",
        }
        if cluster in with_stars:
            members["stellar_mass"] = f"{cluster}/{cluster}_mstar.fits"
        rows[cluster] = members
    return rows


def _source_dir(root: Path, config: Mapping[str, Any]) -> Path:
    return root / str(config["paths"]["source_dir"])


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return _source_dir(root, config) / str(config["paths"][key])


def _raw_path(root: Path, config: Mapping[str, Any], member: str) -> Path:
    return _source_dir(root, config) / str(config["paths"]["raw_dir"]) / member


def _contract_digest(config: Mapping[str, Any]) -> str:
    contract = dict(config)
    contract.pop("scientific_freeze_commit", None)
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    source = config["source"]
    archive = root / str(source["archive_cache"])
    if not archive.is_file():
        raise GravityItem59Error("frozen X-COP archive cache is missing")
    if archive.stat().st_size != int(source["archive_bytes"]):
        raise GravityItem59Error("X-COP archive byte count changed")
    if _sha256_file(archive) != str(source["archive_sha256"]):
        raise GravityItem59Error("X-COP archive digest changed")
    expected = _expected_members(config)
    expected_names = {name for rows in expected.values() for name in rows.values()}
    with tarfile.open(archive, mode="r:gz") as handle:
        names = set(handle.getnames())
    missing = sorted(expected_names - names)
    if missing:
        raise GravityItem59Error(f"X-COP archive lacks frozen members: {missing}")
    forbidden = tuple(map(str, source["forbidden_member_substrings"]))
    if any(any(token in member for token in forbidden) for member in expected_names):
        raise GravityItem59Error("forbidden total-mass-adjacent source requested")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item59-preflight-1.0",
            "item": 59,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "archive": {
                "path": str(source["archive_cache"]),
                "bytes": archive.stat().st_size,
                "sha256": _sha256_file(archive),
            },
            "expected_members": expected,
            "development_clusters": 8,
            "confirmation_clusters_still_unparsed_at_preflight": 4,
            "confirmation_response_rows_read": 0,
            "inferred_total_mass_rows_read": 0,
            "direct_lensing_likelihood_evaluations": 0,
            "paid_model_calls": 0,
        }
    )


def write_preflight(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def extract_sources(root: Path) -> Path:
    config = load_config(root)
    preflight_path = _source_path(root, config, "preflight_manifest")
    if not preflight_path.is_file() or _read_json(preflight_path) != build_preflight_manifest(root):
        raise GravityItem59Error("exact preflight is required before source extraction")
    archive = root / str(config["source"]["archive_cache"])
    members = _expected_members(config)
    records = []
    confirmation = set(
        map(
            str,
            config["population"]["independent_confirmation_clusters_sealed_until_freeze"],
        )
    )
    with tarfile.open(archive, mode="r:gz") as handle:
        for cluster in _all_clusters(config):
            for role, member in sorted(members[cluster].items()):
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise GravityItem59Error(f"unable to read X-COP member: {member}")
                payload = extracted.read()
                destination = _raw_path(root, config, member)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                records.append(
                    {
                        "cluster": cluster,
                        "role": role,
                        "member": member,
                        "bytes": len(payload),
                        "sha256": _sha256_bytes(payload),
                        "confirmation_response_opened_after_scientific_freeze": (
                            cluster in confirmation and role in {"pressure", "temperature"}
                        ),
                    }
                )
    receipt = _content_hashed(
        {
            "schema_version": "invariant-gravity-item59-source-receipt-1.0",
            "item": 59,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "archive_sha256": config["source"]["archive_sha256"],
            "files": records,
            "confirmation_response_files_opened_after_scientific_freeze": 8,
            "inferred_total_mass_files_opened": 0,
            "direct_lensing_likelihood_evaluations": 0,
        }
    )
    path = _source_path(root, config, "source_receipt")
    _write_json(path, receipt)
    return path


def _table(path: Path, hdu_index: int) -> tuple[Any, Mapping[str, Any]]:
    with fits.open(path, memmap=False) as handle:
        hdu = handle[hdu_index]
        data = hdu.data.copy()
        header = dict(hdu.header)
    return data, header


def _parse_cluster(root: Path, config: Mapping[str, Any], cluster: str) -> dict[str, Any]:
    members = _expected_members(config)[cluster]
    density, density_header = _table(
        _raw_path(root, config, members["density"]), int(config["source"]["density_hdu"])
    )
    pressure, pressure_header = _table(
        _raw_path(root, config, members["pressure"]),
        int(config["source"]["sz_pressure_hdu"]),
    )
    temperature, temperature_header = _table(
        _raw_path(root, config, members["temperature"]),
        int(config["source"]["xray_temperature_hdu"]),
    )
    if list(density.dtype.names or ()) != ["RW_X", "NE", "ERR_NE_LO", "ERR_NE_HI"]:
        raise GravityItem59Error(f"density schema changed: {cluster}")
    if list(pressure.dtype.names or ()) != ["RW_SZ", "P_SZ", "eP_SZ"]:
        raise GravityItem59Error(f"SZ pressure schema changed: {cluster}")
    if list(temperature.dtype.names or ()) != ["RW_X", "T_X", "eT_X"]:
        raise GravityItem59Error(f"X-ray temperature schema changed: {cluster}")
    r500 = float(density_header["R500"])
    if not math.isclose(float(pressure_header["R500"]), r500) or not math.isclose(
        float(temperature_header["R500"]), r500
    ):
        raise GravityItem59Error(f"R500 mismatch: {cluster}")
    packet: dict[str, Any] = {
        "cluster": cluster,
        "r500_kpc": r500,
        "density_radius_kpc": np.asarray(density["RW_X"], dtype=float) * r500,
        "ne_cm3": np.asarray(density["NE"], dtype=float),
        "ne_error_low_cm3": np.asarray(density["ERR_NE_LO"], dtype=float),
        "ne_error_high_cm3": np.asarray(density["ERR_NE_HI"], dtype=float),
        "pressure_radius_kpc": np.asarray(pressure["RW_SZ"], dtype=float) * r500,
        "pressure_kev_cm3": np.asarray(pressure["P_SZ"], dtype=float)
        * float(pressure_header["P500"]),
        "pressure_error_kev_cm3": np.asarray(pressure["eP_SZ"], dtype=float)
        * float(pressure_header["P500"]),
        "temperature_radius_kpc": np.asarray(temperature["RW_X"], dtype=float) * r500,
        "temperature_kev": np.asarray(temperature["T_X"], dtype=float)
        * float(temperature_header["T500"]),
        "temperature_error_kev": np.asarray(temperature["eT_X"], dtype=float)
        * float(temperature_header["T500"]),
        "stellar": None,
    }
    if "stellar_mass" in members:
        stellar, _stellar_header = _table(
            _raw_path(root, config, members["stellar_mass"]),
            int(config["source"]["stellar_mass_hdu"]),
        )
        expected = ["RADIUS", "MSTAR", "MSTAR_LO", "MSTAR_HI"]
        if list(stellar.dtype.names or ()) != expected:
            raise GravityItem59Error(f"stellar schema changed: {cluster}")
        packet["stellar"] = {
            "radius_kpc": np.asarray(stellar["RADIUS"], dtype=float),
            "mass_msun": np.asarray(stellar["MSTAR"], dtype=float),
            "mass_low_msun": np.asarray(stellar["MSTAR_LO"], dtype=float),
            "mass_high_msun": np.asarray(stellar["MSTAR_HI"], dtype=float),
        }
    for key in (
        "density_radius_kpc",
        "pressure_radius_kpc",
        "temperature_radius_kpc",
    ):
        values = np.asarray(packet[key])
        if len(values) < 5 or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
            raise GravityItem59Error(f"invalid radial source: {cluster}:{key}")
    for key in (
        "ne_cm3",
        "pressure_kev_cm3",
        "pressure_error_kev_cm3",
        "temperature_kev",
        "temperature_error_kev",
    ):
        values = np.asarray(packet[key])
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise GravityItem59Error(f"invalid observable source: {cluster}:{key}")
    return packet


def _split_order(cluster: str, observable: str, index: int, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{cluster}|{observable}|{index}".encode()).hexdigest()


def prepare_packets(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipt_path = _source_path(root, config, "source_receipt")
    if not receipt_path.is_file():
        raise GravityItem59Error("Item 59 source extraction receipt is missing")
    packets = [_parse_cluster(root, config, cluster) for cluster in _all_clusters(config)]
    development = set(
        map(str, config["population"]["development_clusters_already_exposed"])
    )
    radial = config["radial_split"]
    salt = str(radial["salt"])
    minimum_train = int(radial["minimum_train_rows_per_cluster_observable"])
    minimum_holdout = int(radial["minimum_holdout_rows_per_cluster_observable"])
    for packet in packets:
        density_radius = np.asarray(packet["density_radius_kpc"])
        pressure_radius = np.asarray(packet["pressure_radius_kpc"])
        usable_pressure = np.flatnonzero(
            (pressure_radius >= density_radius[0]) & (pressure_radius <= density_radius[-1])
        )
        if len(usable_pressure) < 5:
            raise GravityItem59Error(f"insufficient usable SZ pressure rows: {packet['cluster']}")
        anchor_index = int(usable_pressure[-1])
        anchor_radius = float(pressure_radius[anchor_index])
        packet["anchor"] = {
            "index": anchor_index,
            "radius_kpc": anchor_radius,
            "pressure_kev_cm3": float(packet["pressure_kev_cm3"][anchor_index]),
            "error_kev_cm3": float(packet["pressure_error_kev_cm3"][anchor_index]),
        }
        definitions = {
            "pressure": (
                np.asarray(packet["pressure_radius_kpc"]),
                np.asarray(packet["pressure_kev_cm3"]),
                np.asarray(packet["pressure_error_kev_cm3"]),
                [int(index) for index in usable_pressure[:-1]],
            ),
            "temperature": (
                np.asarray(packet["temperature_radius_kpc"]),
                np.asarray(packet["temperature_kev"]),
                np.asarray(packet["temperature_error_kev"]),
                [
                    int(index)
                    for index, radius in enumerate(packet["temperature_radius_kpc"])
                    if density_radius[0] <= float(radius) < anchor_radius
                ],
            ),
        }
        rows = []
        for observable, (radii, values, errors, indices) in definitions.items():
            if len(indices) < minimum_train + minimum_holdout:
                raise GravityItem59Error(
                    f"insufficient scorable {observable} rows: {packet['cluster']}"
                )
            if str(packet["cluster"]) in development:
                ordered = sorted(
                    indices,
                    key=lambda index: _split_order(
                        str(packet["cluster"]), observable, index, salt
                    ),
                )
                train_count = round(len(indices) * float(radial["development_train_fraction"]))
                train_count = max(minimum_train, min(len(indices) - minimum_holdout, train_count))
                training = set(ordered[:train_count])
            else:
                training = set()
            for index in indices:
                split = (
                    "development_train"
                    if index in training
                    else (
                        "development_holdout"
                        if str(packet["cluster"]) in development
                        else "confirmation"
                    )
                )
                rows.append(
                    {
                        "row_id": f"{packet['cluster']}:{observable}:{index}",
                        "cluster": str(packet["cluster"]),
                        "observable": observable,
                        "index": index,
                        "radius_kpc": float(radii[index]),
                        "observed": float(values[index]),
                        "error": float(errors[index]),
                        "split": split,
                    }
                )
        packet["rows"] = sorted(rows, key=lambda row: str(row["row_id"]))
    if len(packets) != 12 or sum(packet["cluster"] in development for packet in packets) != 8:
        raise GravityItem59Error("Item 59 parsed sample changed")
    return packets


def _variant_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def enumerate_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    nuisance = config["nuisance_grid"]
    nuisance_rows = itertools.product(
        nuisance["outer_nonthermal_fraction"],
        nuisance["published_stellar_mass_scale"],
        nuisance["missing_stellar_to_gas_mass_ratio"],
        nuisance["xray_temperature_cross_calibration"],
    )
    combinations = [
        {
            "outer_nonthermal_fraction": float(row[0]),
            "published_stellar_mass_scale": float(row[1]),
            "missing_stellar_to_gas_mass_ratio": float(row[2]),
            "xray_temperature_cross_calibration": float(row[3]),
        }
        for row in nuisance_rows
    ]
    variants = []
    for family in config["law_families"]:
        for parameters in family["parameter_grid"]:
            for nuisance_row in combinations:
                body = {
                    "family_id": str(family["id"]),
                    "origin_label": str(family["origin_label"]),
                    "qualifying": bool(family["qualifying"]),
                    "parameters": {
                        str(key): float(value) for key, value in parameters.items()
                    },
                    "nuisances": nuisance_row,
                }
                body["variant_id"] = f"{body['family_id']}:{_variant_id(body)}"
                variants.append(body)
    expected = (1 + 1 + 5 + 6 + 6 + 6) * 3**4
    if len(variants) != expected or len({row["variant_id"] for row in variants}) != expected:
        raise GravityItem59Error("Item 59 variant enumeration changed")
    return variants


def _cumulative_mass(radius_m: np.ndarray, rho_kg_m3: np.ndarray) -> np.ndarray:
    integrand = 4.0 * np.pi * rho_kg_m3 * radius_m**2
    mass = np.empty_like(radius_m)
    mass[0] = 4.0 * np.pi * rho_kg_m3[0] * radius_m[0] ** 3 / 3.0
    mass[1:] = mass[0] + np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(radius_m)
    )
    return mass


def _member_mass(
    packet: Mapping[str, Any],
    radius_kpc: np.ndarray,
    gas_mass_kg: np.ndarray,
    variant: Mapping[str, Any],
    member_mode: str,
    config: Mapping[str, Any],
) -> np.ndarray:
    nuisance = variant["nuisances"]
    stellar = packet["stellar"]
    if stellar is None:
        ratio = float(nuisance["missing_stellar_to_gas_mass_ratio"])
        if member_mode == "low":
            ratio = min(map(float, config["nuisance_grid"]["missing_stellar_to_gas_mass_ratio"]))
        elif member_mode == "high":
            ratio = max(map(float, config["nuisance_grid"]["missing_stellar_to_gas_mass_ratio"]))
        return ratio * gas_mass_kg
    scale = float(nuisance["published_stellar_mass_scale"])
    mass_key = "mass_msun"
    if member_mode == "low":
        scale = min(map(float, config["nuisance_grid"]["published_stellar_mass_scale"]))
        mass_key = "mass_low_msun"
    elif member_mode == "high":
        scale = max(map(float, config["nuisance_grid"]["published_stellar_mass_scale"]))
        mass_key = "mass_high_msun"
    mass = np.maximum.accumulate(np.asarray(stellar[mass_key], dtype=float))
    values = np.interp(
        radius_kpc,
        np.asarray(stellar["radius_kpc"], dtype=float),
        mass,
        left=mass[0],
        right=mass[-1],
    )
    return scale * values * float(config["constants"]["solar_mass_kg"])


def _law_acceleration(
    family_id: str,
    parameters: Mapping[str, Any],
    radius_kpc: np.ndarray,
    r500_kpc: float,
    gbar: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    a0 = float(config["constants"]["transition_acceleration_m_s2"])
    safe_gbar = np.maximum(gbar, np.finfo(float).tiny)
    if family_id == "newtonian_baryons":
        result = safe_gbar
    elif family_id == "empirical_rar":
        result = safe_gbar / -np.expm1(-np.sqrt(safe_gbar / a0))
    elif family_id == "cross_scale_boundary":
        log_radius = np.log(radius_kpc)
        widths = _log_radius_cell_widths(log_radius)
        interior = _kernel_matrix(log_radius, widths, "interior_exponential", 0.25)
        symmetric = _kernel_matrix(log_radius, widths, "symmetric_exponential", 0.25)
        occupancy = (safe_gbar / a0) / (safe_gbar / a0 + 0.1)
        component = safe_gbar * (interior @ occupancy) + a0 * (symmetric @ occupancy)
        result = safe_gbar + float(parameters["beta"]) * component
    elif family_id == "distance_running_gravity":
        x = np.maximum(radius_kpc / r500_kpc, 0.0)
        power = float(parameters["power"])
        flow = x**power / (1.0 + x**power)
        result = safe_gbar * (1.0 + float(parameters["amplitude"]) * flow)
    elif family_id == "qed_like_screened_coupling":
        power = float(parameters["power"])
        response = 1.0 / (1.0 + (safe_gbar / a0) ** power)
        result = safe_gbar * (1.0 + float(parameters["amplitude"]) * response)
    elif family_id == "interior_resonance_equalization":
        log_radius = np.log(radius_kpc)
        widths = _log_radius_cell_widths(log_radius)
        scale = float(parameters["log_radius_scale"])
        interior = _kernel_matrix(log_radius, widths, "interior_exponential", scale)
        result = safe_gbar + float(parameters["amplitude"]) * (interior @ safe_gbar - safe_gbar)
    else:
        raise GravityItem59Error(f"unknown Item 59 law family: {family_id}")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise GravityItem59Error(f"invalid acceleration from {family_id}")
    return result


def _predict_variant(
    packet: Mapping[str, Any],
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    density_mode: str = "nominal",
    anchor_mode: str = "nominal",
    member_mode: str = "nominal",
) -> dict[str, float]:
    constants = config["constants"]
    density_radius = np.asarray(packet["density_radius_kpc"], dtype=float)
    ne = np.asarray(packet["ne_cm3"], dtype=float).copy()
    if density_mode == "low":
        ne -= np.asarray(packet["ne_error_low_cm3"], dtype=float)
    elif density_mode == "high":
        ne += np.asarray(packet["ne_error_high_cm3"], dtype=float)
    ne = np.maximum(ne, np.finfo(float).tiny)
    anchor = packet["anchor"]
    anchor_pressure = float(anchor["pressure_kev_cm3"])
    if anchor_mode == "low":
        anchor_pressure = max(
            np.finfo(float).tiny, anchor_pressure - float(anchor["error_kev_cm3"])
        )
    elif anchor_mode == "high":
        anchor_pressure += float(anchor["error_kev_cm3"])
    target_radii = [float(row["radius_kpc"]) for row in packet["rows"]]
    calc_radius = np.unique(
        np.asarray(
            [
                *density_radius[
                    (density_radius >= density_radius[0])
                    & (density_radius <= float(anchor["radius_kpc"]))
                ],
                *target_radii,
                float(anchor["radius_kpc"]),
            ],
            dtype=float,
        )
    )
    calc_ne = np.exp(np.interp(np.log(calc_radius), np.log(density_radius), np.log(ne)))
    radius_m = calc_radius * float(constants["kiloparsec_m"])
    rho = (
        calc_ne
        * 1.0e6
        * float(constants["mean_molecular_weight_per_electron"])
        * float(constants["proton_mass_kg"])
    )
    gas_mass = _cumulative_mass(radius_m, rho)
    member_mass = _member_mass(
        packet, calc_radius, gas_mass, variant, member_mode, config
    )
    gbar = (
        float(constants["gravity_si"])
        * (gas_mass + member_mass)
        / np.maximum(radius_m**2, np.finfo(float).tiny)
    )
    acceleration = _law_acceleration(
        str(variant["family_id"]),
        variant["parameters"],
        calc_radius,
        float(packet["r500_kpc"]),
        gbar,
        config,
    )
    nonthermal = float(variant["nuisances"]["outer_nonthermal_fraction"])
    radial_power = float(config["nuisance_grid"]["nonthermal_radial_power"])
    thermal_fraction = 1.0 - nonthermal * (calc_radius / float(packet["r500_kpc"])) ** radial_power
    thermal_fraction = np.clip(thermal_fraction, 0.25, 1.0)
    gradient = (
        float(constants["mean_molecular_weight"])
        * float(constants["proton_mass_kg"])
        * calc_ne
        * 1.0e6
        * acceleration
        * thermal_fraction
    )
    integral = np.zeros_like(radius_m)
    for index in range(len(radius_m) - 2, -1, -1):
        integral[index] = integral[index + 1] + 0.5 * (
            gradient[index + 1] + gradient[index]
        ) * (radius_m[index + 1] - radius_m[index])
    pressure = anchor_pressure + integral / float(
        constants["kev_per_cubic_centimeter_j_per_cubic_meter"]
    )
    predictions = {}
    cross_calibration = float(
        variant["nuisances"]["xray_temperature_cross_calibration"]
    )
    for row in packet["rows"]:
        radius = float(row["radius_kpc"])
        pressure_value = float(np.interp(radius, calc_radius, pressure))
        if row["observable"] == "pressure":
            value = pressure_value
        else:
            ne_value = float(
                np.exp(np.interp(np.log(radius), np.log(density_radius), np.log(ne)))
            )
            value = pressure_value / ne_value * cross_calibration
        if not math.isfinite(value) or value <= 0.0:
            raise GravityItem59Error(f"invalid predicted observable: {row['row_id']}")
        predictions[str(row["row_id"])] = value
    return predictions


def _rows(packets: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    return [
        row
        for packet in packets
        for row in packet["rows"]
        if str(row["split"]) == split
    ]


def _score_predictions(
    packets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    split: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _rows(packets, split)
    if not selected:
        raise GravityItem59Error(f"no rows for split: {split}")
    floor = float(config["scoring"]["minimum_fractional_error"])
    grouped: dict[tuple[str, str], list[float]] = {}
    residuals = []
    per_row = []
    for row in selected:
        row_id = str(row["row_id"])
        predicted = float(predictions[row_id])
        observed = float(row["observed"])
        fractional = max(float(row["error"]) / observed, floor)
        log_residual = math.log(predicted / observed)
        standardized = log_residual / fractional
        grouped.setdefault((str(row["cluster"]), str(row["observable"])), []).append(
            standardized**2
        )
        residuals.append(log_residual)
        per_row.append(
            {
                "row_id": row_id,
                "cluster": str(row["cluster"]),
                "observable": str(row["observable"]),
                "radius_kpc": float(row["radius_kpc"]),
                "observed": observed,
                "error": float(row["error"]),
                "predicted": predicted,
                "log_residual": log_residual,
                "standardized_square": standardized**2,
            }
        )
    group_scores = {key: float(np.mean(values)) for key, values in grouped.items()}
    by_observable = {
        observable: float(
            np.mean(
                [value for (_cluster, kind), value in group_scores.items() if kind == observable]
            )
        )
        for observable in ("pressure", "temperature")
    }
    clusters = sorted({cluster for cluster, _observable in group_scores})
    by_cluster = {
        cluster: float(
            np.mean(
                [value for (name, _observable), value in group_scores.items() if name == cluster]
            )
        )
        for cluster in clusters
    }
    return {
        "score": float(np.mean(list(group_scores.values()))),
        "rows": len(selected),
        "cluster_observable_groups": len(group_scores),
        "by_observable": by_observable,
        "by_cluster": by_cluster,
        "median_absolute_log_residual": float(np.median(np.abs(residuals))),
        "root_mean_square_log_residual": float(np.sqrt(np.mean(np.square(residuals)))),
        "per_row": per_row,
    }


def _variant_predictions(
    packets: Sequence[Mapping[str, Any]],
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    density_mode: str = "nominal",
    anchor_mode: str = "nominal",
    member_mode: str = "nominal",
) -> dict[str, float]:
    predictions = {}
    for packet in packets:
        packet_predictions = _predict_variant(
            packet,
            variant,
            config,
            density_mode=density_mode,
            anchor_mode=anchor_mode,
            member_mode=member_mode,
        )
        overlap = set(predictions) & set(packet_predictions)
        if overlap:
            raise GravityItem59Error(f"duplicate Item 59 prediction rows: {sorted(overlap)}")
        predictions.update(packet_predictions)
    return predictions


def _empirical_shape_fit(
    packets: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    degree = int(config["empirical_pressure_shape_comparator"]["degree"])
    training = [row for row in _rows(packets, "development_train") if row["observable"] == "pressure"]
    counts = {
        cluster: sum(row["cluster"] == cluster for row in training)
        for cluster in {str(row["cluster"]) for row in training}
    }
    packet_by_name = {str(packet["cluster"]): packet for packet in packets}
    design = []
    target = []
    weights = []
    for row in training:
        packet = packet_by_name[str(row["cluster"])]
        x = math.log(float(row["radius_kpc"]) / float(packet["anchor"]["radius_kpc"]))
        design.append([x**power for power in range(1, degree + 1)])
        target.append(
            math.log(float(row["observed"]) / float(packet["anchor"]["pressure_kev_cm3"]))
        )
        weights.append(1.0 / counts[str(row["cluster"])])
    matrix = np.asarray(design, dtype=float)
    response = np.asarray(target, dtype=float)
    root_weight = np.sqrt(np.asarray(weights, dtype=float))
    coefficients, *_ = np.linalg.lstsq(
        matrix * root_weight[:, None], response * root_weight, rcond=None
    )
    if np.any(~np.isfinite(coefficients)):
        raise GravityItem59Error("invalid empirical pressure-shape fit")
    candidates = []
    for cross_calibration in config["empirical_pressure_shape_comparator"][
        "temperature_cross_calibration_grid"
    ]:
        fit = {
            "id": str(config["empirical_pressure_shape_comparator"]["id"]),
            "degree": degree,
            "coefficients": [float(value) for value in coefficients],
            "xray_temperature_cross_calibration": float(cross_calibration),
        }
        predictions = _empirical_shape_predictions(packets, fit, config)
        score = _score_predictions(packets, predictions, "development_train", config)
        candidates.append((float(score["score"]), float(cross_calibration), fit))
    return min(candidates, key=lambda row: (row[0], row[1]))[2]


def _empirical_shape_predictions(
    packets: Sequence[Mapping[str, Any]],
    fit: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    density_mode: str = "nominal",
    anchor_mode: str = "nominal",
) -> dict[str, float]:
    predictions = {}
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    cross_calibration = float(fit["xray_temperature_cross_calibration"])
    for packet in packets:
        anchor_pressure = float(packet["anchor"]["pressure_kev_cm3"])
        if anchor_mode == "low":
            anchor_pressure = max(
                np.finfo(float).tiny,
                anchor_pressure - float(packet["anchor"]["error_kev_cm3"]),
            )
        elif anchor_mode == "high":
            anchor_pressure += float(packet["anchor"]["error_kev_cm3"])
        density_radius = np.asarray(packet["density_radius_kpc"], dtype=float)
        ne = np.asarray(packet["ne_cm3"], dtype=float).copy()
        if density_mode == "low":
            ne -= np.asarray(packet["ne_error_low_cm3"], dtype=float)
        elif density_mode == "high":
            ne += np.asarray(packet["ne_error_high_cm3"], dtype=float)
        ne = np.maximum(ne, np.finfo(float).tiny)
        for row in packet["rows"]:
            radius = float(row["radius_kpc"])
            x = math.log(radius / float(packet["anchor"]["radius_kpc"]))
            log_ratio = sum(
                float(coefficient) * x**power
                for power, coefficient in enumerate(coefficients, start=1)
            )
            pressure = anchor_pressure * math.exp(log_ratio)
            if row["observable"] == "pressure":
                value = pressure
            else:
                ne_value = float(
                    np.exp(np.interp(np.log(radius), np.log(density_radius), np.log(ne)))
                )
                value = pressure / ne_value * cross_calibration
            predictions[str(row["row_id"])] = value
    return predictions


def _select_variants(
    development_packets: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    best_by_family: dict[str, dict[str, Any]] = {}
    evaluated = 0
    for variant in enumerate_variants(config):
        predictions = _variant_predictions(development_packets, variant, config)
        metrics = _score_predictions(
            development_packets, predictions, "development_train", config
        )
        evaluated += 1
        row = {
            "variant": variant,
            "training_score": float(metrics["score"]),
            "training_by_observable": metrics["by_observable"],
        }
        family = str(variant["family_id"])
        current = best_by_family.get(family)
        if current is None or (
            float(row["training_score"]), str(variant["variant_id"])
        ) < (
            float(current["training_score"]),
            str(current["variant"]["variant_id"]),
        ):
            best_by_family[family] = row
    qualifying = [
        row for row in best_by_family.values() if bool(row["variant"]["qualifying"])
    ]
    selected = min(
        qualifying,
        key=lambda row: (float(row["training_score"]), str(row["variant"]["variant_id"])),
    )
    return {
        "evaluated_variants": evaluated,
        "best_by_family": best_by_family,
        "selected_qualifying": selected,
    }


def _systematic_modes(name: str) -> tuple[str, str, str]:
    mapping = {
        "density_minus_error": ("low", "nominal", "nominal"),
        "density_plus_error": ("high", "nominal", "nominal"),
        "boundary_minus_error": ("nominal", "low", "nominal"),
        "boundary_plus_error": ("nominal", "high", "nominal"),
        "member_baryons_low": ("nominal", "nominal", "low"),
        "member_baryons_high": ("nominal", "nominal", "high"),
    }
    if name not in mapping:
        raise GravityItem59Error(f"unknown Item 59 systematic: {name}")
    return mapping[name]


def _minimum_improvements(
    candidate: Mapping[str, Any], baselines: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    aggregate = {
        name: 1.0 - float(candidate["score"]) / float(row["score"])
        for name, row in baselines.items()
    }
    by_observable = {
        observable: {
            name: 1.0
            - float(candidate["by_observable"][observable])
            / float(row["by_observable"][observable])
            for name, row in baselines.items()
        }
        for observable in ("pressure", "temperature")
    }
    return {
        "by_baseline": aggregate,
        "minimum": min(aggregate.values()),
        "by_observable_and_baseline": by_observable,
        "minimum_by_observable": {
            observable: min(values.values())
            for observable, values in by_observable.items()
        },
    }


def _public_score(score: Mapping[str, Any], *, include_rows: bool) -> dict[str, Any]:
    return {
        key: value
        for key, value in score.items()
        if include_rows or key != "per_row"
    }


def _source_receipt_valid(root: Path, config: Mapping[str, Any]) -> bool:
    path = _source_path(root, config, "source_receipt")
    if not path.is_file():
        return False
    receipt = _read_json(path)
    expected = _expected_members(config)
    expected_keys = {
        (cluster, role, member)
        for cluster, roles in expected.items()
        for role, member in roles.items()
    }
    rows = receipt.get("files", [])
    observed_keys = {
        (str(row["cluster"]), str(row["role"]), str(row["member"])) for row in rows
    }
    if observed_keys != expected_keys:
        return False
    for row in rows:
        raw = _raw_path(root, config, str(row["member"]))
        if (
            not raw.is_file()
            or raw.stat().st_size != int(row["bytes"])
            or _sha256_file(raw) != str(row["sha256"])
        ):
            return False
    content = dict(receipt)
    expected_content = str(content.pop("content_sha256", ""))
    return _content_hashed(content)["content_sha256"] == expected_content


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    if not _source_receipt_valid(root, config):
        raise GravityItem59Error("Item 59 raw-source receipt failed validation")
    packets = prepare_packets(root, config)
    development_names = set(
        map(str, config["population"]["development_clusters_already_exposed"])
    )
    development = [packet for packet in packets if packet["cluster"] in development_names]
    selection = _select_variants(development, config)
    selected_variant = selection["selected_qualifying"]["variant"]
    family_winners = selection["best_by_family"]
    baseline_variants = {
        "newtonian_baryons": family_winners["newtonian_baryons"]["variant"],
        "empirical_rar": family_winners["empirical_rar"]["variant"],
    }
    shape_fit = _empirical_shape_fit(development, config)

    selected_predictions = _variant_predictions(packets, selected_variant, config)
    baseline_predictions = {
        name: _variant_predictions(packets, variant, config)
        for name, variant in baseline_variants.items()
    }
    baseline_predictions["training_only_cubic_log_pressure_shape"] = (
        _empirical_shape_predictions(packets, shape_fit, config)
    )
    splits = ("development_train", "development_holdout", "confirmation")
    evaluations: dict[str, Any] = {}
    for split in splits:
        candidate_score = _score_predictions(packets, selected_predictions, split, config)
        baseline_scores = {
            name: _score_predictions(packets, prediction, split, config)
            for name, prediction in baseline_predictions.items()
        }
        evaluations[split] = {
            "candidate": _public_score(candidate_score, include_rows=True),
            "baselines": {
                name: _public_score(score, include_rows=False)
                for name, score in baseline_scores.items()
            },
            "improvements": _minimum_improvements(candidate_score, baseline_scores),
        }

    systematic_results = {}
    for name in config["scoring"]["systematic_variants"]:
        density_mode, anchor_mode, member_mode = _systematic_modes(str(name))
        candidate_predictions = _variant_predictions(
            packets,
            selected_variant,
            config,
            density_mode=density_mode,
            anchor_mode=anchor_mode,
            member_mode=member_mode,
        )
        physical_baselines = {
            baseline_name: _variant_predictions(
                packets,
                variant,
                config,
                density_mode=density_mode,
                anchor_mode=anchor_mode,
                member_mode=member_mode,
            )
            for baseline_name, variant in baseline_variants.items()
        }
        physical_baselines["training_only_cubic_log_pressure_shape"] = (
            _empirical_shape_predictions(
                packets,
                shape_fit,
                config,
                density_mode=density_mode,
                anchor_mode=anchor_mode,
            )
        )
        candidate_score = _score_predictions(
            packets, candidate_predictions, "confirmation", config
        )
        baseline_scores = {
            baseline_name: _score_predictions(
                packets, predictions, "confirmation", config
            )
            for baseline_name, predictions in physical_baselines.items()
        }
        systematic_results[str(name)] = {
            "candidate": _public_score(candidate_score, include_rows=False),
            "baselines": {
                baseline_name: _public_score(score, include_rows=False)
                for baseline_name, score in baseline_scores.items()
            },
            "improvements": _minimum_improvements(candidate_score, baseline_scores),
        }

    confirmation = evaluations["confirmation"]
    holdout = evaluations["development_holdout"]
    candidate_cluster_scores = confirmation["candidate"]["by_cluster"]
    baseline_cluster_scores = {
        name: row["by_cluster"] for name, row in confirmation["baselines"].items()
    }
    cluster_wins = [
        cluster
        for cluster, score in candidate_cluster_scores.items()
        if all(
            float(score) < float(values[cluster])
            for values in baseline_cluster_scores.values()
        )
    ]
    counterexamples = [
        cluster for cluster in sorted(candidate_cluster_scores) if cluster not in cluster_wins
    ]
    stable_counterexamples = []
    for cluster in counterexamples:
        stable = True
        for systematic in systematic_results.values():
            candidate_score = float(systematic["candidate"]["by_cluster"][cluster])
            best_baseline = min(
                float(row["by_cluster"][cluster])
                for row in systematic["baselines"].values()
            )
            if candidate_score < best_baseline:
                stable = False
        if stable:
            stable_counterexamples.append(cluster)
    strongest_baseline = min(
        confirmation["baselines"],
        key=lambda name: (
            float(confirmation["baselines"][name]["score"]),
            str(name),
        ),
    )
    advantages = {
        cluster: float(confirmation["baselines"][strongest_baseline]["by_cluster"][cluster])
        - float(candidate_cluster_scores[cluster])
        for cluster in candidate_cluster_scores
    }
    advantage_values = np.asarray(list(advantages.values()), dtype=float)
    leave_one = {
        cluster: float(np.mean([value for name, value in advantages.items() if name != cluster]))
        for cluster in advantages
    }
    trim_each = math.floor(
        len(advantage_values) * float(config["scoring"]["influence_trim_fraction"])
    )
    ordered_advantages = np.sort(advantage_values)
    trimmed = (
        ordered_advantages[trim_each:-trim_each]
        if trim_each and 2 * trim_each < len(ordered_advantages)
        else ordered_advantages
    )
    influence = {
        "strongest_confirmation_baseline": strongest_baseline,
        "per_cluster_advantage": advantages,
        "mean_advantage": float(np.mean(advantage_values)),
        "leave_one_minimum_advantage": min(leave_one.values()),
        "leave_one_changes_sign": any(value <= 0.0 for value in leave_one.values()),
        "trim_each_tail": trim_each,
        "trimmed_mean_advantage": float(np.mean(trimmed)),
        "trim_changes_sign": float(np.mean(trimmed)) <= 0.0,
    }
    threshold = config["admission"]
    gates = {
        "all_12_clusters_evaluable": len(packets) == 12,
        "selected_law_qualifying": bool(selected_variant["qualifying"]),
        "development_radial_holdout_improvement_over_every_baseline_minimum": float(
            holdout["improvements"]["minimum"]
        )
        >= float(
            threshold[
                "development_radial_holdout_improvement_over_every_baseline_minimum"
            ]
        ),
        "confirmation_cluster_improvement_over_every_baseline_minimum": float(
            confirmation["improvements"]["minimum"]
        )
        >= float(
            threshold["confirmation_cluster_improvement_over_every_baseline_minimum"]
        ),
        "candidate_beats_every_baseline_in_each_observable_on_radial_holdout": all(
            value > 0.0
            for value in holdout["improvements"]["minimum_by_observable"].values()
        ),
        "candidate_beats_every_baseline_in_each_observable_on_confirmation": all(
            value > 0.0
            for value in confirmation["improvements"]["minimum_by_observable"].values()
        ),
        "systematic_variants_preserve_positive_confirmation_improvement": all(
            float(row["improvements"]["minimum"]) > 0.0
            for row in systematic_results.values()
        ),
        "leave_one_confirmation_cluster_preserves_positive_improvement": not bool(
            influence["leave_one_changes_sign"]
        ),
        "confirmation_cluster_wins_minimum": len(cluster_wins)
        >= int(threshold["confirmation_cluster_wins_minimum"]),
        "direct_lensing_likelihood_evaluations_zero": int(
            threshold["direct_lensing_likelihood_evaluations"]
        )
        == 0,
        "inferred_total_mass_rows_zero": int(threshold["inferred_total_mass_rows"]) == 0,
    }
    independent_failure_strata = sum(
        float(value) <= 0.0
        for value in confirmation["improvements"]["minimum_by_observable"].values()
    )
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": 4,
        "raw_counterexample_count": len(counterexamples),
        "quality_verified_counterexample_count": len(counterexamples),
        "uncertainty_resolved_counterexample_count": len(stable_counterexamples),
        "aggregate_improvement_percent": 100.0
        * float(confirmation["improvements"]["minimum"]),
        "quality_gate_passed": False,
        "strongest_baseline_failed": float(confirmation["improvements"]["minimum"]) < 0.0,
        "leave_one_changes_sign": bool(influence["leave_one_changes_sign"]),
        "trim_changes_sign": bool(influence["trim_changes_sign"]),
        "independent_failure_strata": int(independent_failure_strata),
        "unchanged_independent_replication_failures": len(stable_counterexamples),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item59-forward-observable-evaluation-1.0",
            "item": 59,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "source_receipt_content_sha256": _read_json(
                _source_path(root, config, "source_receipt")
            )["content_sha256"],
            "selection": {
                "evaluated_variants": selection["evaluated_variants"],
                "selected_qualifying": selection["selected_qualifying"],
                "best_by_family": selection["best_by_family"],
                "baseline_variants": baseline_variants,
                "empirical_pressure_shape_fit": shape_fit,
            },
            "splits": evaluations,
            "systematic_confirmation": systematic_results,
            "confirmation_cluster_wins": cluster_wins,
            "confirmation_counterexamples": counterexamples,
            "counterexamples_stable_across_all_systematics": stable_counterexamples,
            "influence": influence,
            "gates": gates,
            "gate_passed": all(gates.values()),
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": assessment,
            "counts": {
                "clusters": len(packets),
                "development_clusters": 8,
                "confirmation_clusters": 4,
                "development_train_rows": len(_rows(packets, "development_train")),
                "development_holdout_rows": len(
                    _rows(packets, "development_holdout")
                ),
                "confirmation_rows": len(_rows(packets, "confirmation")),
                "evaluated_variants": selection["evaluated_variants"],
                "confirmation_response_files_opened_after_freeze": 8,
                "inferred_total_mass_rows": 0,
                "direct_lensing_likelihood_evaluations": 0,
                "paid_model_calls": 0,
            },
            "compute": {
                "backend": "numpy_cpu",
                "gpu_used": False,
                "paid_api_cost_usd": 0.0,
            },
            "claims": {
                "xcop_forward_observable_development_gate_passed": all(gates.values()),
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
                "direct_lensing_completed": False,
            },
            "limitations": [
                "One outer Planck pressure point per cluster is a measured boundary condition, so this predicts profile shape and the interior normalization conditional on that anchor, not an absolute pressure profile from baryons alone.",
                "Hydrostatic equilibrium, spherical symmetry, constant-composition conversion, and the declared nonthermal-pressure form are assumptions rather than gravity-theory-independent facts.",
                "Released diagonal pressure and temperature errors are used; a complete joint deprojection, calibration, density, pressure, and temperature covariance is unavailable to this evaluator.",
                "Five clusters lack a released member-galaxy stellar-mass profile and use one globally selected stellar-to-gas nuisance bracket.",
                "The four-cluster confirmation is independent of law and nuisance selection but remains a small sample from the same X-COP release.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    source = _read_json(_source_path(root, config, "source_receipt"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    gate_passed = bool(evaluation["gate_passed"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item59-forward-observable-gate-1.0",
            "goal": "GRAVITY_ROADMAP_ITEM_59_XCOP_FORWARD_OBSERVABLE_GATE",
            "item": 59,
            "decision": (
                "ITEM59_XCOP_FORWARD_OBSERVABLE_GATE_PASSED_DEVELOPMENT_EVIDENCE"
                if gate_passed
                else "ITEM59_XCOP_FORWARD_OBSERVABLE_GATE_NOT_PASSED_FAMILIES_RETAINED"
            ),
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "preflight": preflight,
            "source_receipt": source,
            "selection": evaluation["selection"],
            "splits": evaluation["splits"],
            "systematic_confirmation": evaluation["systematic_confirmation"],
            "confirmation_cluster_wins": evaluation["confirmation_cluster_wins"],
            "confirmation_counterexamples": evaluation["confirmation_counterexamples"],
            "counterexamples_stable_across_all_systematics": evaluation[
                "counterexamples_stable_across_all_systematics"
            ],
            "influence": evaluation["influence"],
            "gates": evaluation["gates"],
            "gate_passed": gate_passed,
            "counterexample_policy_assessment": evaluation[
                "counterexample_policy_assessment"
            ],
            "counts": evaluation["counts"],
            "compute": evaluation["compute"],
            "claims": {
                **evaluation["claims"],
                "roadmap_item_59_execution_complete": True,
            },
            "limitations": evaluation["limitations"],
            "next_action": (
                "Advance to Item 60 direct-CLASH lensing. Preserve all Item 59 families and object-level failures; do not refit on the four X-COP confirmations."
            ),
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    checks = {
        "preflight": _read_json(_source_path(root, config, "preflight_manifest"))
        == build_preflight_manifest(root),
        "source_receipt": _source_receipt_valid(root, config),
        "evaluation": _read_json(_source_path(root, config, "evaluation_result"))
        == build_evaluation_result(root),
        "aggregate": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "extract", "evaluate", "aggregate", "replay")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight(root))
    elif args.command == "extract":
        result = str(extract_sources(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
