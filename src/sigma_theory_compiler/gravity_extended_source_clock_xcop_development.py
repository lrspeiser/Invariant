"""One authorized development-only X-COP test of the frozen extended-source clock."""

from __future__ import annotations

import argparse
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
from astropy.io import fits

from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    _cumulative_mass,
    _law_acceleration,
    _member_mass,
)

CONFIG_PATH = Path("configs/gravity_extended_source_clock_xcop_development_v1.json")
IMPLEMENTATION_PATH = Path(
    "src/sigma_theory_compiler/gravity_extended_source_clock_xcop_development.py"
)
RESULT_SCHEMA = "invariant-gravity-extended-source-clock-xcop-development-result-1.0"


class ClockDevelopmentError(RuntimeError):
    """Raised when the frozen development-only contract changes."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _with_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("content_sha256", None)
    body["content_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-extended-source-clock-xcop-development-config-1.0"
        or config.get("analysis_id") != "gravity-extended-source-clock-xcop-development-v1"
        or config.get("status") != "authorized_development_only_single_run"
    ):
        raise ClockDevelopmentError("unsupported clock development config")
    for name, binding in config["upstream_bindings"].items():
        path = root / str(binding["path"])
        if not path.is_file() or _sha256_file(path) != str(binding["sha256"]):
            raise ClockDevelopmentError(f"upstream binding changed: {name}")
    population = config["population"]
    development = list(map(str, population["development_clusters"]))
    forbidden = list(map(str, population["forbidden_confirmation_clusters"]))
    stellar = set(map(str, population["clusters_with_stellar_profile"]))
    expected_development = [
        "A1644",
        "A1795",
        "A2142",
        "A2255",
        "A2319",
        "A3266",
        "A85",
        "ZW1215",
    ]
    if (
        development != expected_development
        or forbidden != ["A2029", "A3158", "A644", "RXC1825"]
        or set(development) & set(forbidden)
        or stellar != {"A1795", "A2142", "A2319", "A85", "ZW1215"}
    ):
        raise ClockDevelopmentError("development population changed")
    inputs = config["input_contract"]
    files = list(inputs["files"])
    if (
        int(inputs["unique_files"]) != 29
        or len(files) != 29
        or len({str(row["member"]) for row in files}) != 29
        or sum(int(row["bytes"]) for row in files) != int(inputs["total_bytes"])
        or any(str(row["cluster"]) not in development for row in files)
        or any(str(row["cluster"]) in forbidden for row in files)
    ):
        raise ClockDevelopmentError("input allowlist changed")
    by_cluster: dict[str, set[str]] = {cluster: set() for cluster in development}
    for row in files:
        by_cluster[str(row["cluster"])].add(str(row["role"]))
        if re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None:
            raise ClockDevelopmentError("invalid input hash")
    for cluster, roles in by_cluster.items():
        expected = {"density", "pressure", "temperature"}
        if cluster in stellar:
            expected.add("stellar_mass")
        if roles != expected:
            raise ClockDevelopmentError(f"input roles changed: {cluster}")
    law = config["frozen_clock_law"]
    if (
        float(law["a0_m_s2"]) != 1.2e-10
        or law["eta_definition"] != "max(0,d ln M_b(<r)/d ln r)"
        or law["extended_multiplier"] != "2*eta_plus"
        or law["transition"] != "max(rar_multiplier,extended_multiplier)"
    ):
        raise ClockDevelopmentError("clock law changed")
    nuisance = config["fixed_nuisances"]
    if nuisance != {
        "outer_nonthermal_fraction": 0.15,
        "published_stellar_mass_scale": 1.0,
        "missing_stellar_to_gas_mass_ratio": 0.1,
        "xray_temperature_cross_calibration": 1.0,
        "nonthermal_radial_power": 1.0,
        "selection_or_tuning_calls": 0,
    }:
        raise ClockDevelopmentError("fixed nuisance contract changed")
    access = config["access_ceiling"]
    if any(
        int(access[key]) != 0
        for key in (
            "confirmation_files",
            "independent_rows",
            "group_rows",
            "lensing_rows",
            "network_calls",
            "model_calls",
            "paid_calls",
            "tuning_calls",
        )
    ):
        raise ClockDevelopmentError("forbidden access enabled")


def validate_authorization(
    root: Path, config: Mapping[str, Any], authorization_path: Path
) -> dict[str, Any]:
    expected = root / str(config["authorization"]["path"])
    if authorization_path.resolve() != expected.resolve():
        raise ClockDevelopmentError("authorization path changed")
    authorization = _read_json(authorization_path)
    expected_keys = {
        "schema_version",
        "authorization_id",
        "run_id",
        "authorized",
        "authorized_by",
        "approved_at_utc",
        "approval_phrase",
        "config_path",
        "config_sha256",
        "allowed_clusters",
        "allowed_unique_files",
        "allowed_total_bytes",
        "confirmation_clusters_allowed",
        "independent_rows_allowed",
        "group_rows_allowed",
        "lensing_rows_allowed",
        "network_calls_allowed",
        "model_calls_allowed",
        "paid_calls_allowed",
        "tuning_calls_allowed",
        "single_run_only",
    }
    if set(authorization) != expected_keys:
        raise ClockDevelopmentError("authorization schema changed")
    if (
        authorization["schema_version"]
        != "invariant-gravity-extended-source-clock-xcop-development-authorization-1.0"
        or authorization["authorization_id"] != "extended-source-clock-xcop-development-v1"
        or authorization["run_id"] != config["authorization"]["run_id"]
        or authorization["authorized"] is not True
        or authorization["authorized_by"] != "Henry"
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            str(authorization["approved_at_utc"]),
        )
        is None
        or authorization["approval_phrase"] != config["authorization"]["required_phrase"]
        or authorization["config_path"] != str(CONFIG_PATH).replace("\\", "/")
        or authorization["config_sha256"] != _sha256_file(root / CONFIG_PATH)
        or authorization["allowed_clusters"] != config["population"]["development_clusters"]
        or int(authorization["allowed_unique_files"]) != 29
        or int(authorization["allowed_total_bytes"]) != 538560
        or authorization["single_run_only"] is not True
    ):
        raise ClockDevelopmentError("authorization does not match the frozen run")
    for key in (
        "confirmation_clusters_allowed",
        "independent_rows_allowed",
        "group_rows_allowed",
        "lensing_rows_allowed",
        "network_calls_allowed",
        "model_calls_allowed",
        "paid_calls_allowed",
        "tuning_calls_allowed",
    ):
        if int(authorization[key]) != 0:
            raise ClockDevelopmentError(f"authorization enabled forbidden scope: {key}")
    return authorization


def _load_allowed_payloads(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[tuple[str, str], bytes], list[dict[str, Any]]]:
    raw_root = root / str(config["input_contract"]["raw_root"])
    payloads: dict[tuple[str, str], bytes] = {}
    ledger = []
    for row in config["input_contract"]["files"]:
        cluster = str(row["cluster"])
        role = str(row["role"])
        path = raw_root / str(row["member"])
        payload = path.read_bytes()
        actual_hash = _sha256_bytes(payload)
        if len(payload) != int(row["bytes"]) or actual_hash != str(row["sha256"]):
            raise ClockDevelopmentError(f"allowed input changed: {cluster}:{role}")
        payloads[(cluster, role)] = payload
        ledger.append(
            {
                "cluster": cluster,
                "role": role,
                "member": str(row["member"]),
                "bytes": len(payload),
                "sha256": actual_hash,
            }
        )
    if len(payloads) != 29:
        raise ClockDevelopmentError("allowed payload count changed")
    return payloads, ledger


def _fits_table(payload: bytes, hdu_index: int) -> tuple[Any, Mapping[str, Any]]:
    with fits.open(io.BytesIO(payload), memmap=False) as handle:
        hdu = handle[hdu_index]
        return hdu.data.copy(), dict(hdu.header)


def _parse_packet(
    cluster: str,
    payloads: Mapping[tuple[str, str], bytes],
    item59: Mapping[str, Any],
) -> dict[str, Any]:
    source = item59["source"]
    density, density_header = _fits_table(
        payloads[(cluster, "density")], int(source["density_hdu"])
    )
    pressure, pressure_header = _fits_table(
        payloads[(cluster, "pressure")], int(source["sz_pressure_hdu"])
    )
    temperature, temperature_header = _fits_table(
        payloads[(cluster, "temperature")], int(source["xray_temperature_hdu"])
    )
    if list(density.dtype.names or ()) != ["RW_X", "NE", "ERR_NE_LO", "ERR_NE_HI"]:
        raise ClockDevelopmentError(f"density schema changed: {cluster}")
    if list(pressure.dtype.names or ()) != ["RW_SZ", "P_SZ", "eP_SZ"]:
        raise ClockDevelopmentError(f"pressure schema changed: {cluster}")
    if list(temperature.dtype.names or ()) != ["RW_X", "T_X", "eT_X"]:
        raise ClockDevelopmentError(f"temperature schema changed: {cluster}")
    r500 = float(density_header["R500"])
    if not math.isclose(float(pressure_header["R500"]), r500) or not math.isclose(
        float(temperature_header["R500"]), r500
    ):
        raise ClockDevelopmentError(f"R500 mismatch: {cluster}")
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
        "source_rows": {
            "density": len(density),
            "pressure": len(pressure),
            "temperature": len(temperature),
            "stellar_mass": 0,
        },
    }
    stellar_payload = payloads.get((cluster, "stellar_mass"))
    if stellar_payload is not None:
        stellar, _header = _fits_table(stellar_payload, int(source["stellar_mass_hdu"]))
        if list(stellar.dtype.names or ()) != ["RADIUS", "MSTAR", "MSTAR_LO", "MSTAR_HI"]:
            raise ClockDevelopmentError(f"stellar schema changed: {cluster}")
        packet["stellar"] = {
            "radius_kpc": np.asarray(stellar["RADIUS"], dtype=float),
            "mass_msun": np.asarray(stellar["MSTAR"], dtype=float),
            "mass_low_msun": np.asarray(stellar["MSTAR_LO"], dtype=float),
            "mass_high_msun": np.asarray(stellar["MSTAR_HI"], dtype=float),
        }
        packet["source_rows"]["stellar_mass"] = len(stellar)
    for key in ("density_radius_kpc", "pressure_radius_kpc", "temperature_radius_kpc"):
        values = np.asarray(packet[key])
        if len(values) < 5 or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0):
            raise ClockDevelopmentError(f"invalid radial source: {cluster}:{key}")
    return packet


def _split_order(cluster: str, observable: str, index: int, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{cluster}|{observable}|{index}".encode()).hexdigest()


def _add_rows(packet: dict[str, Any], item59: Mapping[str, Any]) -> None:
    density_radius = np.asarray(packet["density_radius_kpc"])
    pressure_radius = np.asarray(packet["pressure_radius_kpc"])
    usable_pressure = np.flatnonzero(
        (pressure_radius >= density_radius[0]) & (pressure_radius <= density_radius[-1])
    )
    if len(usable_pressure) < 5:
        raise ClockDevelopmentError(f"insufficient pressure rows: {packet['cluster']}")
    anchor_index = int(usable_pressure[-1])
    anchor_radius = float(pressure_radius[anchor_index])
    packet["anchor"] = {
        "index": anchor_index,
        "radius_kpc": anchor_radius,
        "pressure_kev_cm3": float(packet["pressure_kev_cm3"][anchor_index]),
        "error_kev_cm3": float(packet["pressure_error_kev_cm3"][anchor_index]),
    }
    radial = item59["radial_split"]
    minimum_train = int(radial["minimum_train_rows_per_cluster_observable"])
    minimum_holdout = int(radial["minimum_holdout_rows_per_cluster_observable"])
    definitions = {
        "pressure": (
            pressure_radius,
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
            raise ClockDevelopmentError(
                f"insufficient scorable {observable} rows: {packet['cluster']}"
            )
        ordered = sorted(
            indices,
            key=lambda index: _split_order(
                str(packet["cluster"]), observable, index, str(radial["salt"])
            ),
        )
        train_count = round(len(indices) * float(radial["development_train_fraction"]))
        train_count = max(minimum_train, min(len(indices) - minimum_holdout, train_count))
        training = set(ordered[:train_count])
        for index in indices:
            rows.append(
                {
                    "row_id": f"{packet['cluster']}:{observable}:{index}",
                    "cluster": str(packet["cluster"]),
                    "observable": observable,
                    "index": index,
                    "radius_kpc": float(radii[index]),
                    "observed": float(values[index]),
                    "error": float(errors[index]),
                    "split": ("development_train" if index in training else "development_holdout"),
                }
            )
    packet["rows"] = sorted(rows, key=lambda row: str(row["row_id"]))


def _predict_law(
    packet: Mapping[str, Any],
    law_id: str,
    item59: Mapping[str, Any],
    fixed_nuisances: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    constants = item59["constants"]
    density_radius = np.asarray(packet["density_radius_kpc"], dtype=float)
    ne = np.maximum(np.asarray(packet["ne_cm3"], dtype=float), np.finfo(float).tiny)
    anchor = packet["anchor"]
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
    variant = {"nuisances": dict(fixed_nuisances)}
    member_mass = _member_mass(packet, calc_radius, gas_mass, variant, "nominal", item59)
    baryonic_mass = gas_mass + member_mass
    gbar = (
        float(constants["gravity_si"])
        * baryonic_mass
        / np.maximum(radius_m**2, np.finfo(float).tiny)
    )
    a0 = float(constants["transition_acceleration_m_s2"])
    rar_multiplier = 1.0 / -np.expm1(-np.sqrt(np.maximum(gbar, np.finfo(float).tiny) / a0))
    eta = np.gradient(np.log(baryonic_mass), np.log(radius_m), edge_order=2)
    eta_plus = np.maximum(eta, 0.0)
    extended_multiplier = 2.0 * eta_plus
    if law_id == "extended_source_clock":
        multiplier = np.maximum(rar_multiplier, extended_multiplier)
        acceleration = multiplier * gbar
    elif law_id == "newtonian_baryons":
        multiplier = np.ones_like(gbar)
        acceleration = gbar
    elif law_id == "empirical_rar":
        multiplier = rar_multiplier
        acceleration = multiplier * gbar
    elif law_id == "previous_cross_scale_candidate":
        acceleration = _law_acceleration(
            "cross_scale_boundary",
            {"beta": 1.5},
            calc_radius,
            float(packet["r500_kpc"]),
            gbar,
            item59,
        )
        multiplier = acceleration / gbar
    else:
        raise ClockDevelopmentError(f"unknown law: {law_id}")
    if np.any(~np.isfinite(acceleration)) or np.any(acceleration <= 0):
        raise ClockDevelopmentError(f"invalid acceleration: {packet['cluster']}:{law_id}")
    nonthermal = float(fixed_nuisances["outer_nonthermal_fraction"])
    radial_power = float(fixed_nuisances["nonthermal_radial_power"])
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
        integral[index] = integral[index + 1] + 0.5 * (gradient[index + 1] + gradient[index]) * (
            radius_m[index + 1] - radius_m[index]
        )
    pressure = float(anchor["pressure_kev_cm3"]) + integral / float(
        constants["kev_per_cubic_centimeter_j_per_cubic_meter"]
    )
    predictions = {}
    cross_calibration = float(fixed_nuisances["xray_temperature_cross_calibration"])
    for row in packet["rows"]:
        radius = float(row["radius_kpc"])
        pressure_value = float(np.interp(radius, calc_radius, pressure))
        if row["observable"] == "pressure":
            value = pressure_value
        else:
            ne_value = float(np.exp(np.interp(np.log(radius), np.log(density_radius), np.log(ne))))
            value = pressure_value / ne_value * cross_calibration
        if not math.isfinite(value) or value <= 0:
            raise ClockDevelopmentError(f"invalid prediction: {row['row_id']}:{law_id}")
        predictions[str(row["row_id"])] = value
    diagnostics = {
        "eta_plus_min": float(np.min(eta_plus)),
        "eta_plus_max": float(np.max(eta_plus)),
        "rar_multiplier_min": float(np.min(rar_multiplier)),
        "rar_multiplier_max": float(np.max(rar_multiplier)),
        "law_multiplier_min": float(np.min(multiplier)),
        "law_multiplier_max": float(np.max(multiplier)),
        "extended_channel_fraction": float(np.mean(extended_multiplier > rar_multiplier)),
        "clock_ratio_min": float(np.min(multiplier**-0.5)),
        "clock_ratio_max": float(np.max(multiplier**-0.5)),
        "stellar_profile_available": packet["stellar"] is not None,
    }
    return predictions, diagnostics


def _score(
    packets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    split: str,
    minimum_fractional_error: float,
) -> dict[str, Any]:
    selected = [row for packet in packets for row in packet["rows"] if str(row["split"]) == split]
    grouped: dict[tuple[str, str], list[float]] = {}
    per_row = []
    for row in selected:
        row_id = str(row["row_id"])
        observed = float(row["observed"])
        predicted = float(predictions[row_id])
        fractional = max(float(row["error"]) / observed, minimum_fractional_error)
        log_residual = math.log(predicted / observed)
        standardized_square = (log_residual / fractional) ** 2
        grouped.setdefault((str(row["cluster"]), str(row["observable"])), []).append(
            standardized_square
        )
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
                "standardized_square": standardized_square,
            }
        )
    group_scores = {key: float(np.mean(values)) for key, values in grouped.items()}
    clusters = sorted({cluster for cluster, _observable in group_scores})
    return {
        "score": float(np.mean(list(group_scores.values()))),
        "rows": len(selected),
        "by_observable": {
            observable: float(
                np.mean(
                    [
                        value
                        for (_cluster, kind), value in group_scores.items()
                        if kind == observable
                    ]
                )
            )
            for observable in ("pressure", "temperature")
        },
        "by_cluster": {
            cluster: float(
                np.mean(
                    [
                        value
                        for (name, _observable), value in group_scores.items()
                        if name == cluster
                    ]
                )
            )
            for cluster in clusters
        },
        "by_cluster_observable": {
            f"{cluster}:{observable}": value
            for (cluster, observable), value in sorted(group_scores.items())
        },
        "per_row": sorted(per_row, key=lambda row: str(row["row_id"])),
    }


def _comparison(clock: Mapping[str, Any], comparator: Mapping[str, Any]) -> dict[str, Any]:
    clock_score = float(clock["score"])
    comparator_score = float(comparator["score"])
    clusters = sorted(clock["by_cluster"])
    observables = sorted(clock["by_cluster_observable"])
    return {
        "fractional_improvement": 1.0 - clock_score / comparator_score,
        "clock_better_overall": clock_score < comparator_score,
        "cluster_wins": [
            cluster
            for cluster in clusters
            if float(clock["by_cluster"][cluster]) < float(comparator["by_cluster"][cluster])
        ],
        "cluster_counterexamples": [
            cluster
            for cluster in clusters
            if float(clock["by_cluster"][cluster]) >= float(comparator["by_cluster"][cluster])
        ],
        "cluster_observable_wins": [
            key
            for key in observables
            if float(clock["by_cluster_observable"][key])
            < float(comparator["by_cluster_observable"][key])
        ],
        "cluster_observable_counterexamples": [
            key
            for key in observables
            if float(clock["by_cluster_observable"][key])
            >= float(comparator["by_cluster_observable"][key])
        ],
    }


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ClockDevelopmentError(f"output already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def execute(root: Path, authorization_path: Path) -> Path:
    config = load_config(root)
    result_path = root / str(config["output"])
    if result_path.exists():
        raise ClockDevelopmentError("single-run result already exists")
    authorization = validate_authorization(root, config, authorization_path)
    item59 = _read_json(root / str(config["upstream_bindings"]["item59_config"]["path"]))
    payloads, file_ledger = _load_allowed_payloads(root, config)
    packets = []
    for cluster in config["population"]["development_clusters"]:
        packet = _parse_packet(str(cluster), payloads, item59)
        _add_rows(packet, item59)
        packets.append(packet)
    laws = [
        "extended_source_clock",
        "newtonian_baryons",
        "empirical_rar",
        "previous_cross_scale_candidate",
    ]
    all_predictions: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for law_id in laws:
        all_predictions[law_id] = {}
        diagnostics[law_id] = {}
        for packet in packets:
            prediction, diagnostic = _predict_law(packet, law_id, item59, config["fixed_nuisances"])
            all_predictions[law_id].update(prediction)
            diagnostics[law_id][str(packet["cluster"])] = diagnostic
    scores = {
        split: {
            law_id: _score(
                packets,
                all_predictions[law_id],
                split,
                float(config["scoring"]["minimum_fractional_error"]),
            )
            for law_id in laws
        }
        for split in ("development_train", "development_holdout")
    }
    comparisons = {
        split: {
            law_id: _comparison(scores[split]["extended_source_clock"], scores[split][law_id])
            for law_id in laws
            if law_id != "extended_source_clock"
        }
        for split in scores
    }
    primary = scores["development_holdout"]
    ranking = sorted(laws, key=lambda law_id: (float(primary[law_id]["score"]), law_id))
    clock_rank = ranking.index("extended_source_clock") + 1
    decision = (
        "EXTENDED_SOURCE_CLOCK_RANKS_FIRST_ON_FROZEN_DEVELOPMENT_HOLDOUT"
        if clock_rank == 1
        else "EXTENDED_SOURCE_CLOCK_DOES_NOT_RANK_FIRST_ON_FROZEN_DEVELOPMENT_HOLDOUT"
    )
    source_rows = {
        role: sum(int(packet["source_rows"][role]) for packet in packets)
        for role in ("density", "pressure", "temperature", "stellar_mass")
    }
    result = _with_content_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "analysis_id": config["analysis_id"],
            "status": "completed_development_only_real_xcop_run",
            "decision": decision,
            "bindings": {
                "config_path": str(CONFIG_PATH).replace("\\", "/"),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "implementation_path": str(IMPLEMENTATION_PATH).replace("\\", "/"),
                "implementation_sha256": _sha256_file(root / IMPLEMENTATION_PATH),
                "authorization_path": str(authorization_path.relative_to(root)).replace("\\", "/"),
                "authorization_sha256": _sha256_file(authorization_path),
                "authorization_id": authorization["authorization_id"],
                "upstream": config["upstream_bindings"],
            },
            "frozen_law": config["frozen_clock_law"],
            "fixed_nuisances": config["fixed_nuisances"],
            "population": config["population"],
            "input_file_ledger": file_ledger,
            "source_rows_opened": source_rows,
            "scoring": {
                "contract": config["scoring"],
                "scores": scores,
                "comparisons": comparisons,
                "primary_holdout_ranking": ranking,
                "extended_source_clock_rank": clock_rank,
            },
            "clock_diagnostics": diagnostics["extended_source_clock"],
            "counterexamples": {
                split: {
                    law_id: comparisons[split][law_id]["cluster_counterexamples"]
                    for law_id in comparisons[split]
                }
                for split in comparisons
            },
            "access_and_compute": {
                "unique_development_files_opened": len(file_ledger),
                "development_file_bytes_opened": sum(int(row["bytes"]) for row in file_ledger),
                "development_clusters_opened": len(packets),
                "development_train_rows_scored_per_law": int(
                    scores["development_train"]["extended_source_clock"]["rows"]
                ),
                "development_holdout_rows_scored_per_law": int(
                    scores["development_holdout"]["extended_source_clock"]["rows"]
                ),
                "laws_scored": len(laws),
                "confirmation_files_opened": 0,
                "confirmation_rows_opened": 0,
                "independent_rows_opened": 0,
                "group_rows_opened": 0,
                "lensing_rows_opened": 0,
                "network_calls": 0,
                "model_calls": 0,
                "paid_calls": 0,
                "tuning_calls": 0,
            },
            "claims": {
                "development_only_real_cluster_evidence_completed": True,
                "confirmation_evidence_completed": False,
                "independent_replication_completed": False,
                "absolute_pressure_prediction_established": False,
                "covariant_time_theory_established": False,
                "same_action_lensing_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "publication_readiness_changed": False,
                "scientific_claim_allowed": False,
            },
            "limitations": [
                "One measured outer pressure point per cluster is an unscored boundary condition, so the run tests interior profile shape conditional on that boundary rather than absolute pressure normalization.",
                "The fixed spherical hydrostatic, nonthermal, composition, stellar-profile, and missing-stellar assumptions remain model limitations.",
                "The factor 2 and max transition were frozen from analytic synthetic controls, not derived from a covariant action.",
                "The previous cross-scale comparator was selected on earlier Item59 development training rows and is contextual rather than an untouched baseline.",
                "No confirmation, independent, group, or lensing data were opened; this result cannot establish generalization or publication readiness.",
            ],
        }
    )
    _atomic_no_clobber(result_path, _canonical_bytes(result) + b"\n")
    return result_path


def check_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = root / str(config["output"])
    result = _read_json(path)
    body = dict(result)
    content_hash = str(body.pop("content_sha256", ""))
    valid = (
        result.get("schema_version") == RESULT_SCHEMA
        and content_hash == _sha256_bytes(_canonical_bytes(body))
        and result.get("bindings", {}).get("config_sha256") == _sha256_file(root / CONFIG_PATH)
        and result.get("bindings", {}).get("implementation_sha256")
        == _sha256_file(root / IMPLEMENTATION_PATH)
        and result.get("access_and_compute", {}).get("confirmation_rows_opened") == 0
        and result.get("access_and_compute", {}).get("network_calls") == 0
        and result.get("claims", {}).get("scientific_claim_allowed") is False
    )
    if not valid:
        raise ClockDevelopmentError("stored result failed validation")
    return {
        "valid": True,
        "decision": result["decision"],
        "clock_rank": result["scoring"]["extended_source_clock_rank"],
        "holdout_clock_score": result["scoring"]["scores"]["development_holdout"][
            "extended_source_clock"
        ]["score"],
        "confirmation_rows_opened": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--authorization", required=True, type=Path)
    subparsers.add_parser("check-result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "execute":
        path = execute(root, root / args.authorization)
        print(path)
    else:
        print(json.dumps(check_result(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
