"""Source-only X-COP eligibility preflight for GAIN-PERSISTENCE-01.

This module is deliberately unable to open pressure or temperature files.  It reads the
eight frozen density profiles and five available stellar profiles, reconstructs only
baryonic source fields, and checks whether the preregistered y=100 transport anchor exists.
It computes no scientific response score.
"""

from __future__ import annotations

import argparse
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
from astropy.io import fits

from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    _cumulative_mass,
    _member_mass,
)

CONFIG_PATH = Path("configs/gravity_gain_persistence_gp01_xcop_source_preflight_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/gravity_gain_persistence_gp01_xcop_source_preflight.py"
)
TEST_PATH = Path("tests/test_gravity_gain_persistence_gp01_xcop_source_preflight.py")
OUTPUT_PATH = Path("runs/gravity/theory/gain-persistence-gp01-xcop-source-preflight-v1.json")

CONFIG_SHA256 = "4ac4bd034cfd6339dd8bdcb6fd946af06b254da77c2a9eeba682ea0fac50e40d"
CONFIG_CONTENT_SHA256 = "1616b528f3784116316590a48b30ff4ac56345f1dcdc30d5a6c1b0b6985cc7d7"
CONFIG_SCHEMA = "invariant-gravity-gain-persistence-gp01-xcop-source-preflight-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-gain-persistence-gp01-xcop-source-preflight-receipt-1.0"
DECISION = "SOURCE_ONLY_PREFLIGHT_LOCAL_AND_ELLIPTIC_READY_T1_T2_BLOCKED_NO_Y100_ANCHOR"


class GP01XcopSourcePreflightError(RuntimeError):
    """Raised when the frozen source-only preflight fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GP01XcopSourcePreflightError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GP01XcopSourcePreflightError(f"could not load frozen JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == CONFIG_CONTENT_SHA256, "config semantics changed")
    _exact_keys(
        config,
        {
            "schema_version",
            "preflight_id",
            "version",
            "status",
            "purpose",
            "upstream_bindings",
            "population",
            "source_contract",
            "frozen_source_mapping",
            "branch_adjudication",
            "claim_boundary",
            "zero_access",
            "output_path",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["status"] == "FROZEN_SOURCE_ONLY_ZERO_RESPONSE_ACCESS",
        "config status changed",
    )
    population = config["population"]
    clusters = list(population["development_clusters"])
    _require(
        clusters == ["A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"],
        "development clusters changed",
    )
    _require(
        set(population["clusters_with_stellar_profile"])
        == {"A1795", "A2142", "A2319", "A85", "ZW1215"},
        "stellar-profile population changed",
    )
    _require(
        set(population["clusters_with_shared_missing_stellar_rule"]) == {"A1644", "A2255", "A3266"},
        "missing-stellar population changed",
    )
    source = config["source_contract"]
    _require(source["allowed_roles"] == ["density", "stellar_mass"], "allowed roles changed")
    _require(
        {"pressure", "temperature"} <= set(source["forbidden_roles"]),
        "response role is not forbidden",
    )
    _require(
        (
            source["expected_density_files"],
            source["expected_stellar_files"],
            source["expected_source_files"],
        )
        == (8, 5, 13),
        "source file counts changed",
    )
    _require(source["expected_source_bytes"] == 308160, "source byte ceiling changed")
    mapping = config["frozen_source_mapping"]
    _require(mapping["a_star_m_s2"] == 1.2e-10, "a_star changed")
    _require(mapping["transport_anchor_y"] == 100.0, "transport anchor changed")
    _require(mapping["published_stellar_mass_scale"] == 1.0, "stellar scale changed")
    _require(mapping["missing_stellar_to_gas_mass_ratio"] == 0.1, "missing-star rule changed")
    claims = config["claim_boundary"]
    _require(claims["source_eligibility_only"] is True, "source-only claim changed")
    _require(claims["response_scoring_authorized"] is False, "response scoring unlocked")
    _require(claims["pressure_or_temperature_opened"] is False, "response access claimed")
    _require(
        claims["missing_anchor_is_source_blocked_not_empirical_failure"] is True,
        "missing-anchor interpretation changed",
    )
    _require(not any(config["zero_access"].values()), "zero-access contract changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    _require(file_sha256(path) == CONFIG_SHA256, "preflight config bytes changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _verify_upstream(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for binding in config["upstream_bindings"]:
        path = root / str(binding["path"])
        digest = file_sha256(path)
        _require(digest == binding["sha256"], f"upstream changed: {binding['id']}")
        observed[str(binding["id"])] = digest
    return observed


def _source_records(
    root: Path, config: Mapping[str, Any]
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    parent = _read_json(root / "configs/gravity_extended_source_clock_xcop_development_v1.json")
    item59 = _read_json(root / "configs/gravity_item59_xcop_forward_observable_gate_v1.json")
    clusters = set(config["population"]["development_clusters"])
    allowed_roles = set(config["source_contract"]["allowed_roles"])
    records = [
        dict(row)
        for row in parent["input_contract"]["files"]
        if row["cluster"] in clusters and row["role"] in allowed_roles
    ]
    _require(len(records) == 13, "source record count changed")
    _require(sum(int(row["bytes"]) for row in records) == 308160, "source bytes changed")
    _require(
        sum(row["role"] == "density" for row in records) == 8
        and sum(row["role"] == "stellar_mass" for row in records) == 5,
        "source role counts changed",
    )
    _require(
        not ({"pressure", "temperature"} & {str(row["role"]) for row in records}),
        "response file admitted",
    )
    return root / parent["input_contract"]["raw_root"], records, item59


def _fits_table(payload: bytes, hdu_index: int) -> tuple[Any, Mapping[str, Any]]:
    try:
        with fits.open(io.BytesIO(payload), memmap=False) as hdus:
            return hdus[hdu_index].data.copy(), dict(hdus[hdu_index].header)
    except Exception as error:
        raise GP01XcopSourcePreflightError("source FITS parse failed") from error


def _effective_density(radius_m: np.ndarray, mass_kg: np.ndarray) -> np.ndarray:
    derivative = np.gradient(mass_kg, radius_m, edge_order=2)
    density = np.maximum(derivative / (4.0 * np.pi * radius_m**2), 0.0)
    inner = 3.0 * mass_kg[0] / (4.0 * np.pi * radius_m[0] ** 3)
    density[0] = max(float(density[0]), float(inner))
    _require(np.all(np.isfinite(density) & (density >= 0.0)), "effective density invalid")
    return density


def _first_r90(radius_kpc: np.ndarray, mass_kg: np.ndarray) -> float:
    target = 0.9 * float(mass_kg[-1])
    index = int(np.flatnonzero(mass_kg >= target)[0])
    return float(radius_kpc[index])


def _positive_median(values: np.ndarray, label: str) -> float:
    selected = values[np.isfinite(values) & (values > 0.0)]
    _require(selected.size > 0, f"no positive {label} reference")
    return float(np.median(selected))


def _cluster_report(
    cluster: str,
    raw_root: Path,
    record_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    item59: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opened: list[dict[str, Any]] = []

    def read_role(role: str) -> bytes:
        row = record_by_key[(cluster, role)]
        path = raw_root / str(row["member"])
        payload = path.read_bytes()
        _require(len(payload) == int(row["bytes"]), "source file byte count changed")
        _require(hashlib.sha256(payload).hexdigest() == row["sha256"], "source file hash changed")
        opened.append(
            {
                "cluster": cluster,
                "role": role,
                "member": str(row["member"]),
                "bytes": len(payload),
                "sha256": str(row["sha256"]),
            }
        )
        return payload

    density, density_header = _fits_table(
        read_role("density"), int(config["source_contract"]["density_hdu"])
    )
    _require(
        list(density.dtype.names or ()) == config["source_contract"]["density_columns"],
        "density schema changed",
    )
    r500 = float(density_header["R500"])
    radius_kpc = np.asarray(density["RW_X"], dtype=float) * r500
    ne_cm3 = np.maximum(np.asarray(density["NE"], dtype=float), np.finfo(float).tiny)
    _require(
        radius_kpc.size >= 3
        and np.all(np.isfinite(radius_kpc))
        and np.all(np.diff(radius_kpc) > 0.0),
        "density radii invalid",
    )
    constants = item59["constants"]
    radius_m = radius_kpc * float(constants["kiloparsec_m"])
    gas_density = (
        ne_cm3
        * 1.0e6
        * float(constants["mean_molecular_weight_per_electron"])
        * float(constants["proton_mass_kg"])
    )
    gas_mass = _cumulative_mass(radius_m, gas_density)
    stellar = None
    if (cluster, "stellar_mass") in record_by_key:
        stellar_table, _stellar_header = _fits_table(
            read_role("stellar_mass"), int(config["source_contract"]["stellar_hdu"])
        )
        _require(
            list(stellar_table.dtype.names or ()) == config["source_contract"]["stellar_columns"],
            "stellar schema changed",
        )
        stellar = {
            "radius_kpc": np.asarray(stellar_table["RADIUS"], dtype=float),
            "mass_msun": np.asarray(stellar_table["MSTAR"], dtype=float),
            "mass_low_msun": np.asarray(stellar_table["MSTAR_LO"], dtype=float),
            "mass_high_msun": np.asarray(stellar_table["MSTAR_HI"], dtype=float),
        }
    variant = {
        "nuisances": {
            "published_stellar_mass_scale": float(
                config["frozen_source_mapping"]["published_stellar_mass_scale"]
            ),
            "missing_stellar_to_gas_mass_ratio": float(
                config["frozen_source_mapping"]["missing_stellar_to_gas_mass_ratio"]
            ),
        }
    }
    member_mass = _member_mass(
        {"stellar": stellar}, radius_kpc, gas_mass, variant, "nominal", item59
    )
    baryonic_mass = gas_mass + member_mass
    gravity = float(constants["gravity_si"])
    g_b = gravity * baryonic_mass / radius_m**2
    y = g_b / float(config["frozen_source_mapping"]["a_star_m_s2"])
    anchor = float(config["frozen_source_mapping"]["transport_anchor_y"])
    outward_crossings = np.flatnonzero((y[:-1] >= anchor) & (y[1:] < anchor))
    effective_density = _effective_density(radius_m, baryonic_mass)
    tidal = math.sqrt(2.0 / 3.0) * np.abs(
        4.0 * np.pi * gravity * effective_density - 3.0 * g_b / radius_m
    )
    anchor_count = int(outward_crossings.size)
    transport_status = (
        "SOURCE_READY_UNIQUE_Y100_ANCHOR"
        if anchor_count == 1
        else "SOURCE_BLOCKED_NO_UNIQUE_Y100_ANCHOR"
    )
    report = {
        "cluster": cluster,
        "density_rows": int(radius_kpc.size),
        "stellar_rows": 0 if stellar is None else int(stellar["radius_kpc"].size),
        "stellar_profile_available": stellar is not None,
        "missing_stellar_rule_applied": stellar is None,
        "r500_kpc": r500,
        "R_b_r90_kpc": _first_r90(radius_kpc, baryonic_mass),
        "g_b_min_m_s2": float(np.min(g_b)),
        "g_b_max_m_s2": float(np.max(g_b)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "nodes_at_or_above_y100": int(np.count_nonzero(y >= anchor)),
        "outward_y100_crossing_count": anchor_count,
        "transport_status": transport_status,
        "rho_reference_kg_m3": _positive_median(effective_density, "density"),
        "tidal_reference_s_minus_2": _positive_median(tidal, "tidal"),
        "local_status": "SOURCE_READY_LOCAL_RADIAL_CONTROL",
        "aqual_status": "EQUIVALENCE_LINK_SPHERICAL_SCORE_ONCE",
        "elliptic_status": "SOURCE_READY_PENDING_EXACT_SPHERICAL_SOLVER",
        "telegraph_status": "SOURCE_BLOCKED_NO_SOURCE_HISTORY",
    }
    return report, opened


def receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("content_sha256", None)
    return content_sha256(payload)


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    upstream = _verify_upstream(root, config)
    raw_root, records, item59 = _source_records(root, config)
    record_by_key = {(str(row["cluster"]), str(row["role"])): row for row in records}
    reports: list[dict[str, Any]] = []
    opened: list[dict[str, Any]] = []
    for cluster in config["population"]["development_clusters"]:
        report, cluster_opened = _cluster_report(
            str(cluster), raw_root, record_by_key, item59, config
        )
        reports.append(report)
        opened.extend(cluster_opened)
    _require(len(opened) == 13, "opened source count changed")
    _require(sum(row["bytes"] for row in opened) == 308160, "opened source bytes changed")
    _require(all(row["role"] in {"density", "stellar_mass"} for row in opened), "response opened")
    transport_ready = sum(row["transport_status"].startswith("SOURCE_READY") for row in reports)
    _require(transport_ready == 0, "a transport anchor unexpectedly became ready")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "preflight_id": config["preflight_id"],
        "status": "SOURCE_ONLY_PREFLIGHT_COMPLETE_ZERO_RESPONSE_ACCESS",
        "decision": DECISION,
        "bindings": {
            "config_sha256": file_sha256(root / CONFIG_PATH),
            "module_sha256": file_sha256(root / MODULE_PATH),
            "test_sha256": file_sha256(root / TEST_PATH),
            "upstream_sha256": upstream,
        },
        "source_access": {
            "opened_files": sorted(opened, key=lambda row: (row["cluster"], row["role"])),
            "density_files_opened": 8,
            "stellar_files_opened": 5,
            "source_files_opened": 13,
            "source_bytes_opened": 308160,
            "density_rows_opened": sum(row["density_rows"] for row in reports),
            "stellar_rows_opened": sum(row["stellar_rows"] for row in reports),
            "pressure_files_opened": 0,
            "temperature_files_opened": 0,
            "response_rows_opened": 0,
            "scientific_scores_computed": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "clusters": reports,
        "adjudication": {
            "clusters_total": 8,
            "local_source_ready": 8,
            "aqual_spherical_equivalence_links": 8,
            "transport_source_ready": transport_ready,
            "transport_source_blocked": 8 - transport_ready,
            "elliptic_source_ready_pending_solver": 8,
            "telegraph_source_blocked": 8,
            "action_quarantined": 8,
            "missing_anchor_interpretation": "SOURCE_BLOCKED_NOT_EMPIRICAL_FAILURE",
        },
        "claim_boundary": dict(config["claim_boundary"]),
        "limitations": [
            "The preflight uses only spherical source profiles and does not establish a general three-dimensional history law.",
            "The three missing stellar profiles use one shared frozen 0.1 gas-mass rule; no object-specific value is fitted.",
            "No pressure, temperature, motion, lensing, confirmation, or independent row was opened or scored.",
            "Absence of the preregistered y=100 anchor is a source-eligibility result, not an empirical falsification of T1 or T2.",
            "Elliptic source readiness does not imply that its numerical solver or physical theory is complete.",
        ],
    }
    receipt["content_sha256"] = receipt_content_sha256(receipt)
    return receipt


def validate_receipt(root: Path, output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    target = output_path if output_path.is_absolute() else root / output_path
    stored = _read_json(target)
    _require(stored.get("content_sha256") == receipt_content_sha256(stored), "receipt hash failed")
    _require(stored == build_receipt(root), "receipt differs from deterministic rebuild")
    return stored


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing to overwrite nonidentical receipt")
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
            _require(path.read_bytes() == payload, "concurrent nonidentical receipt exists")
            return "EXISTING_IDENTICAL"
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except PermissionError:
            _require(os.name == "nt", "receipt directory could not be opened for fsync")
        else:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path) -> str:
    receipt = build_receipt(root)
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return _atomic_no_clobber(root / OUTPUT_PATH, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        publication = write_receipt(root)
        receipt = validate_receipt(root)
    elif args.command == "check":
        publication = None
        receipt = validate_receipt(root)
    else:
        publication = None
        receipt = build_receipt(root)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "transport_source_ready": receipt["adjudication"]["transport_source_ready"],
                "transport_source_blocked": receipt["adjudication"]["transport_source_blocked"],
                "response_rows_opened": receipt["source_access"]["response_rows_opened"],
                "publication": publication,
                "content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
