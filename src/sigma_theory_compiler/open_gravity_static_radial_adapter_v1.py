"""Append-only, target-blind static-radial adapter for the open-gravity campaign.

The module has no observational loader.  Its public adapter functions accept in-memory
source-only physical arrays, while its receipt executes analytic synthetic fixtures only.
It never accepts a response vector and has no scoring or likelihood surface.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_static_radial_adapter_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_static_radial_adapter_v1.py")
TEST_PATH = Path("tests/test_open_gravity_static_radial_adapter_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-static-radial-adapter-v1/receipt.json")
BLOCKED_RECEIPT_PATH = Path(
    "work/open-gravity-static-radial-adapter-v1-audit-blocked/receipt-pre-repair.json"
)
BLOCKED_RECEIPT_SHA256 = "8162b16fd78afc81d105a2ac7394c01af2d4665efce2dbef70887b8dc9a5eabc"
CONFIG_SCHEMA = "invariant-open-gravity-static-radial-adapter-config-1.0"
RECEIPT_SCHEMA = "invariant-open-gravity-static-radial-adapter-receipt-1.0"
DECISION = "ADAPTER_MECHANICS_SYNTHETICALLY_VERIFIED_NO_SCIENTIFIC_PASS_NO_SCORING_AUTHORITY"
EXPECTED_CONFIG_CONTENT_SHA256 = "6508503882159b3b684b02f71a5e6c897a28d972c707ddd1afe6e82a123b66b8"

G_SI = 6.67430e-11
KPC_M = 3.085677581491367e19
PRIMARY_POINTS = 257
CONVERGENCE_POINTS = 129
FINITE_TOLERANCE = 1.0e-12
OPERATOR_RESIDUAL_TOLERANCE = 1.0e-9
BOUNDARY_RESIDUAL_TOLERANCE = 1.0e-10
CONVERGENCE_MAX_ABS_TOLERANCE = 0.02
STATIC_ARCHITECTURES = (
    "A01_LAPSE",
    "A02_CLOCK",
    "A03_CONFORMAL",
    "A04_DISFORMAL",
    "A05_SLIP",
    "A06_SPATIAL_KERNEL",
    "A07_BOUNDARY",
    "A08_PERMITTIVITY",
    "A09_ENTROPIC",
    "A10_DENSITY_SCREEN",
    "A11_DERIV_SCREEN",
    "A12_MASSIVE",
    "A13_MIXED_MODE",
    "A14_PHASE",
    "A19_FEEDBACK",
)
TIME_SOURCE_BLOCKS = (
    "A15_RETARDED",
    "A16_MEMORY",
    "A17_RESONANCE",
    "A18_STOCHASTIC",
)
COMPOUND_IDS = ("X01", "X05", "X10", "X13", "X17", "X18")
SPARC_DRIVERS = ("D01_ACC", "D03_RAD", "D06_SLOPE", "D13_GASF")
XCOP_DRIVERS = (
    "D01_ACC",
    "D02_POT",
    "D03_RAD",
    "D04_RHO",
    "D05_SIG",
    "D06_SLOPE",
    "D07_TIDE",
    "D13_GASF",
)
DRIVER_REFERENCES = {
    "D01_ACC": 1.0e-10,
    "D02_POT": 1.0e12,
    "D03_RAD": KPC_M,
    "D04_RHO": 1.0e-21,
    "D05_SIG": 1.0,
    "D06_SLOPE": 1.0,
    "D07_TIDE": 1.0e-30,
    "D13_GASF": 1.0,
}
ARCHITECTURE_PARAMETERS: dict[str, dict[str, tuple[float | int, ...]]] = {
    "A01_LAPSE": {"lambda": (0.0, 0.25)},
    "A02_CLOCK": {"lambda": (0.0, 0.25)},
    "A03_CONFORMAL": {"lambda": (0.0, 0.25)},
    "A04_DISFORMAL": {"lambda": (0.0, 0.25)},
    "A05_SLIP": {"lambda": (0.0, 0.25)},
    "A06_SPATIAL_KERNEL": {"lambda": (0.0, 0.25), "ell": (0.1, 0.25)},
    "A07_BOUNDARY": {"lambda": (0.0, 0.25)},
    "A08_PERMITTIVITY": {"lambda": (0.0, 0.25)},
    "A09_ENTROPIC": {"lambda": (0.0, 0.25)},
    "A10_DENSITY_SCREEN": {"lambda": (0.0, 0.25), "u_c": (0.25, 0.5), "n": (2,)},
    "A11_DERIV_SCREEN": {"lambda": (0.0, 0.25), "s_c": (0.5, 1.0), "n": (2,)},
    "A12_MASSIVE": {"lambda": (0.0, 0.25), "mu": (1.0, 4.0)},
    "A13_MIXED_MODE": {
        "lambda": (0.0, 0.25),
        "theta": (0.0, math.pi / 4.0),
        "ell": (0.25,),
    },
    "A14_PHASE": {"lambda": (0.0, 0.25), "k": (1.0, 2.0), "phi0": (0.0,)},
    "A19_FEEDBACK": {"lambda": (0.0, 0.25), "kappa": (0.0, 0.5)},
}
GP01_GRID = {
    "n": (1, 2, 4),
    "A_max": (2.0, 4.0, 8.0),
    "rho_ratio": (0.1, 1.0, 10.0),
    "tide_ratio": (0.1, 1.0, 10.0),
    "q": (1, 2),
    "tide_power": (1, 2),
    "L_ratio": (0.0, 0.25, 1.0, 4.0),
}


class OpenGravityStaticRadialAdapterError(RuntimeError):
    """Raised when the frozen adapter contract fails closed."""


class StaticSourceBlockedError(OpenGravityStaticRadialAdapterError):
    """Raised for a declared source-ineligible domain/architecture combination."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenGravityStaticRadialAdapterError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} keys changed")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OpenGravityStaticRadialAdapterError(f"noncanonical value: {error}") from error


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def registry_content_sha256(value: Any) -> str:
    """Use the live registry's canonical JSON rule (no trailing newline)."""

    return hashlib.sha256(canonical_bytes(value)[:-1]).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(value: Sequence[float] | np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    header = canonical_bytes({"dtype": "float64-le", "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenGravityStaticRadialAdapterError(f"could not load frozen JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _program_root(config: Mapping[str, Any]) -> str:
    return content_sha256(
        {
            "driver_contract": config["driver_contract"],
            "architecture_programs": config["architecture_programs"],
            "compound_programs": config["compound_programs"],
            "gp01_program": config["gp01_program"],
            "wrong_control_programs": config["wrong_control_programs"],
        }
    )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate every frozen nested semantic through an immutable canonical seal."""

    _exact_keys(
        config,
        {
            "schema_version",
            "adapter_id",
            "semantic_version",
            "status",
            "purpose",
            "identity",
            "committed_upstream_bindings",
            "access_contract",
            "grid_contract",
            "driver_contract",
            "architecture_programs",
            "static_real_source_blocks",
            "compound_programs",
            "gp01_program",
            "wrong_control_programs",
            "gate_contract",
            "twell_rebind_gate",
            "typed_card_contract",
            "output_contract",
            "claim_boundary",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "immutable config content changed",
    )
    _require(config["adapter_id"] == "OPEN-GRAVITY-STATIC-RADIAL-ADAPTER-v1", "ID changed")
    _require(config["semantic_version"] == "1.0.0", "version changed")
    identity = config["identity"]
    _require(
        identity["append_only"] is True and identity["target_blind"] is True, "identity changed"
    )
    _require(identity["campaign_manifest_authority"] is False, "manifest authority enabled")
    _require(identity["scientific_scoring_authority"] is False, "score authority enabled")

    bindings = config["committed_upstream_bindings"]
    _require(len(bindings) == 2, "committed binding count changed")
    _require(
        [row["commit"] for row in bindings]
        == [
            "35f70938f158c81971b2e1b838371b09d9fcee2c",
            "ed2988546fb1165d9efe5e62d52cddebc7b1a79d",
        ],
        "committed upstream changed",
    )
    _require(all(len(row["files"]) == 4 for row in bindings), "upstream file set changed")
    _require(
        all(len(file["sha256"]) == 64 for row in bindings for file in row["files"]),
        "bad upstream seal",
    )

    access = config["access_contract"]
    forbidden = set(access["forbidden_inputs"])
    _require(
        {
            "SPARC_VOBS_OR_ANY_MOTION_RESPONSE",
            "XCOP_PRESSURE_OR_TEMPERATURE",
            "RESPONSE_BEARING_RECEIPTS",
            "RESIDUALS_RANKINGS_SCORES_OR_LIKELIHOODS",
        }
        <= forbidden,
        "response boundary weakened",
    )
    _require(access["loader_surface"] == "NONE_ARRAYS_ONLY", "loader surface changed")
    _require(all(value == 0 for value in access["zero_access"].values()), "nonzero access declared")

    grid = config["grid_contract"]
    _require(grid["primary_points"] == PRIMARY_POINTS, "primary grid changed")
    _require(grid["convergence_points"] == CONVERGENCE_POINTS, "convergence grid changed")
    _require(grid["domain"] == [0.0, 1.0], "radial domain changed")

    references = config["driver_contract"]["reference_values_si"]
    _require(references == DRIVER_REFERENCES, "driver SI references changed")
    _require(
        tuple(config["driver_contract"]["sparc"]["available_drivers"]) == SPARC_DRIVERS,
        "SPARC driver set changed",
    )
    _require(
        tuple(config["driver_contract"]["xcop_spherical"]["available_drivers"]) == XCOP_DRIVERS,
        "X-COP driver set changed",
    )

    architectures = config["architecture_programs"]
    _require(
        tuple(row["id"] for row in architectures) == STATIC_ARCHITECTURES,
        "architecture order changed",
    )
    for row in architectures:
        expected = {
            name: list(values) for name, values in ARCHITECTURE_PARAMETERS[row["id"]].items()
        }
        _require(row["parameters"] == expected, f"parameter grid changed: {row['id']}")
    _require(
        tuple(config["static_real_source_blocks"]) == TIME_SOURCE_BLOCKS,
        "time-source block set changed",
    )
    compounds = config["compound_programs"]
    _require(tuple(row["id"] for row in compounds) == COMPOUND_IDS, "compound set changed")
    _require(compounds[0]["sparc_status"].startswith("SOURCE_BLOCKED_"), "SPARC X01 unlocked")

    gp01 = config["gp01_program"]
    _require(gp01["n_grid"] == [1, 2, 4], "GP01 n grid changed")
    _require(gp01["L_g_over_R_b_grid_including_zero"][0] == 0.0, "L=0 control removed")
    expected_count = math.prod(len(values) for values in GP01_GRID.values())
    _require(expected_count == 1296 == gp01["exact_cell_count"], "GP01 enumeration changed")

    gates = config["gate_contract"]
    _require(gates["finite_tolerance"] == FINITE_TOLERANCE, "finite tolerance changed")
    _require(
        gates["operator_residual_tolerance"] == OPERATOR_RESIDUAL_TOLERANCE,
        "operator tolerance changed",
    )
    _require(
        gates["boundary_residual_tolerance"] == BOUNDARY_RESIDUAL_TOLERANCE,
        "boundary tolerance changed",
    )
    _require(
        gates["convergence_max_abs_tolerance"] == CONVERGENCE_MAX_ABS_TOLERANCE,
        "convergence tolerance changed",
    )

    controls = config["wrong_control_programs"]
    _require(
        [row["id"] for row in controls] == ["IDENTITY", "RADIAL_FACTOR_REVERSAL"],
        "controls changed",
    )
    _require(all(row["target_free"] is True for row in controls), "target-free control changed")

    rebind = config["twell_rebind_gate"]
    _require(rebind["current_repair_hashes_authoritative"] is False, "repair became authoritative")
    _require(rebind["repair_paths_may_be_opened_by_this_adapter"] is False, "repair read enabled")
    cards = config["typed_card_contract"]
    _require(
        cards["atomic_architectures"] == list(STATIC_ARCHITECTURES),
        "typed-card architectures changed",
    )
    _require(cards["atomic_drivers"] == list(XCOP_DRIVERS), "typed-card drivers changed")
    _require(cards["atomic_card_count"] == 120, "typed atomic card count changed")
    _require(cards["gp01_card_count"] == 7, "GP01 typed card count changed")
    _require(
        cards["provisional_twell_adapter_card_count"] == 126, "provisional TWELL count changed"
    )
    _require(cards["exact_total_card_count"] == 133, "typed card count changed")
    _require(
        cards["provisional_twell_cards_manifest_authority_before_rebind"] is False,
        "provisional TWELL cards gained manifest authority",
    )
    _require(cards["lane_assignment_authority"] is False, "adapter gained lane authority")
    _require(cards["orthogonal_or_wildcard_fillers_invented"] is False, "lane filler invented")
    _require(config["output_contract"]["path"] == OUTPUT_PATH.as_posix(), "output changed")
    claims = config["claim_boundary"]
    _require(claims["scientific_pass_claimed"] is False, "scientific PASS claimed")
    _require(claims["campaign_manifest_created"] is False, "manifest claimed")
    _require(claims["campaign_scoring_authorized"] is False, "scoring authorized")
    _require(claims["twell_final_rebind_complete"] is False, "TWELL rebind claimed")


def load_config(root: Path = Path("."), config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    _require(config_path == CONFIG_PATH, "only the frozen config path is allowed")
    config = _read_json(root / config_path)
    validate_config(config)
    return config


def verify_committed_upstreams(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash exact committed files; receipt roles are never parsed or semantically opened."""

    verified: list[dict[str, Any]] = []
    for binding in config["committed_upstream_bindings"]:
        for item in binding["files"]:
            path = root / item["path"]
            _require(path.is_file(), f"missing committed upstream: {item['path']}")
            observed = file_sha256(path)
            _require(observed == item["sha256"], f"upstream hash changed: {item['path']}")
            verified.append(
                {
                    "binding_id": binding["binding_id"],
                    "commit": binding["commit"],
                    "path": item["path"],
                    "role": item["role"],
                    "sha256": observed,
                    "semantic_open": "RECEIPT" not in item["role"],
                }
            )
    return {
        "commits": [row["commit"] for row in config["committed_upstream_bindings"]],
        "files": verified,
        "all_exact": True,
        "response_bearing_receipts_opened": 0,
    }


def verify_blocked_receipt_preservation(root: Path) -> dict[str, Any]:
    """Bind the audit-blocked predecessor bytewise without parsing it."""

    path = root / BLOCKED_RECEIPT_PATH
    _require(path.is_file(), "audit-blocked predecessor receipt is not preserved")
    observed = file_sha256(path)
    _require(observed == BLOCKED_RECEIPT_SHA256, "audit-blocked predecessor receipt changed")
    return {
        "path": BLOCKED_RECEIPT_PATH.as_posix(),
        "sha256": observed,
        "status": "BLOCKED_SUPERSEDED_PRESERVED_AS_COUNTEREVIDENCE",
        "semantic_open": False,
    }


def evaluate_twell_rebind_gate(
    config: Mapping[str, Any], candidate: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Evaluate an explicit future hash rebind without reading any deferred repair path."""

    required = tuple(config["twell_rebind_gate"]["required_final_fields"])
    if candidate is None:
        return {
            "status": "DEFERRED_NOT_REBOUND",
            "authoritative": False,
            "missing_fields": list(required),
            "repair_paths_opened": 0,
        }
    _exact_keys(candidate, set(required), "TWELL rebind candidate")
    _require(
        candidate["independent_audit_status"]
        == config["twell_rebind_gate"]["required_independent_audit_status"],
        "TWELL independent adapter audit has not passed",
    )
    for name in required:
        if name.endswith("_sha256"):
            value = candidate[name]
            _require(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"bad TWELL final hash: {name}",
            )
    _require(
        candidate["final_formula_program_root_sha256"] == _program_root(config),
        "TWELL final operator program is not equivalent to the frozen adapter",
    )
    return {
        "status": "REBIND_MECHANICS_EQUIVALENCE_SATISFIED_NO_CAMPAIGN_AUTHORITY",
        "authoritative": True,
        "campaign_authority": False,
        "repair_paths_opened": 0,
        "final_hashes": dict(candidate),
    }


def _vector(
    value: Sequence[float] | np.ndarray, label: str, *, nonnegative: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    _require(array.ndim == 1 and array.size >= 3, f"{label} must be a one-dimensional source array")
    _require(bool(np.all(np.isfinite(array))), f"{label} is nonfinite")
    if nonnegative:
        _require(bool(np.all(array >= 0.0)), f"{label} must be nonnegative")
    return array


def _source_radius(value: Sequence[float] | np.ndarray, *, allow_zero: bool) -> np.ndarray:
    radius = _vector(value, "radius_m", nonnegative=True)
    _require(bool(np.all(np.diff(radius) > 0.0)), "source radii must strictly increase")
    if allow_zero:
        _require(radius[0] >= 0.0, "source radius is invalid")
    else:
        _require(radius[0] > 0.0, "first source radius must be positive")
    _require(radius[-1] > 0.0, "outer source radius must be positive")
    return radius


def _uniform_radius(r_out: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    _require(points in (PRIMARY_POINTS, CONVERGENCE_POINTS), "grid size is not frozen")
    _require(math.isfinite(r_out) and r_out > 0.0, "outer radius must be positive")
    xi = np.linspace(0.0, 1.0, points, dtype=float)
    return xi, xi * r_out


def _prepend_regular_origin(radius: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    if radius[0] == 0.0:
        return (radius, *values)
    return (np.concatenate(([0.0], radius)), *(np.concatenate(([0.0], row)) for row in values))


def _frozen_derivative(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    _require(values.shape == coordinate.shape and values.size >= 3, "bad derivative arrays")
    spacing = np.diff(coordinate)
    _require(bool(np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-14)), "grid is not uniform")
    h = float(spacing[0])
    derivative = np.empty_like(values)
    derivative[0] = (values[1] - values[0]) / h
    derivative[-1] = (values[-1] - values[-2]) / h
    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * h)
    return derivative


def _log_slope(radius: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    _require(radius.shape == acceleration.shape, "log-slope arrays differ")
    positive = (radius > 0.0) & (acceleration > 0.0)
    indices = np.flatnonzero(positive)
    _require(indices.size >= 3, "log slope requires three positive source points")
    _require(
        bool(np.all(positive[indices[0] :])), "only the regular origin may have zero acceleration"
    )
    log_radius = np.log(radius[indices])
    log_acceleration = np.log(acceleration[indices])
    edge_order = 2 if indices.size >= 3 else 1
    slope_positive = -np.gradient(log_acceleration, log_radius, edge_order=edge_order)
    slope = np.empty_like(acceleration)
    slope[indices] = slope_positive
    slope[: indices[0]] = slope_positive[0]
    return slope


def normalize_driver(driver_id: str, physical: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply the exact dimensionless normalization for one admitted driver."""

    _require(driver_id in DRIVER_REFERENCES, f"unknown driver: {driver_id}")
    value = np.asarray(physical, dtype=float)
    _require(bool(np.all(np.isfinite(value))), f"nonfinite driver: {driver_id}")
    if driver_id == "D06_SLOPE":
        scaled = value
    else:
        _require(bool(np.all(value >= 0.0)), f"negative physical driver: {driver_id}")
        scaled = value / DRIVER_REFERENCES[driver_id]
    result = np.tanh(scaled)
    _require(bool(np.all(np.isfinite(result))), f"normalization failed: {driver_id}")
    return result


def _driver_bundle(
    *,
    domain: str,
    xi: np.ndarray,
    radius_m: np.ndarray,
    physical: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = {
        driver_id: normalize_driver(driver_id, values) for driver_id, values in physical.items()
    }
    return {
        "domain": domain,
        "xi": xi,
        "radius_m": radius_m,
        "physical": dict(physical),
        "normalized": normalized,
        "metadata": dict(metadata),
        "response_inputs": 0,
        "scores_computed": 0,
    }


def compile_sparc_source_drivers(
    radius_m: Sequence[float] | np.ndarray,
    gas_acceleration_m_s2: Sequence[float] | np.ndarray,
    stellar_acceleration_m_s2: Sequence[float] | np.ndarray,
    *,
    points: int = PRIMARY_POINTS,
) -> dict[str, Any]:
    """Compile SPARC source-model accelerations without accepting any observed velocity."""

    radius = _source_radius(radius_m, allow_zero=True)
    gas = _vector(gas_acceleration_m_s2, "gas_acceleration_m_s2", nonnegative=True)
    stars = _vector(stellar_acceleration_m_s2, "stellar_acceleration_m_s2", nonnegative=True)
    _require(radius.shape == gas.shape == stars.shape, "SPARC source array shapes differ")
    radius, gas, stars = _prepend_regular_origin(radius, gas, stars)
    xi, grid_radius = _uniform_radius(float(radius[-1]), points)
    gas_grid = np.interp(grid_radius, radius, gas)
    star_grid = np.interp(grid_radius, radius, stars)
    g_b = gas_grid + star_grid
    _require(
        bool(np.all(g_b[1:] > 0.0)), "SPARC baryonic acceleration must be positive away from origin"
    )
    gas_fraction = np.divide(gas_grid, g_b, out=np.zeros_like(g_b), where=g_b > 0.0)
    slope = _log_slope(grid_radius, g_b)
    physical = {
        "D01_ACC": g_b,
        "D03_RAD": grid_radius,
        "D06_SLOPE": slope,
        "D13_GASF": gas_fraction,
    }
    return _driver_bundle(
        domain="SPARC",
        xi=xi,
        radius_m=grid_radius,
        physical=physical,
        metadata={
            "source_rows_supplied": len(radius_m),
            "grid_points": points,
            "source_only_component_accelerations": True,
            "vobs_accepted": False,
            "r_out_m": float(grid_radius[-1]),
        },
    )


def _observed_shell_mass(radius: np.ndarray, density: np.ndarray) -> np.ndarray:
    _require(radius[0] > 0.0 and radius.shape == density.shape, "bad spherical density source")
    mass = np.empty_like(radius)
    mass[0] = 4.0 * math.pi * density[0] * radius[0] ** 3 / 3.0
    integrand = 4.0 * math.pi * density * radius * radius
    increments = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(radius)
    mass[1:] = mass[0] + np.cumsum(increments)
    _require(bool(np.all(np.diff(mass) >= 0.0)), "gas mass is not monotone")
    return mass


def _interpolate_enclosed_mass(
    radius: np.ndarray, mass: np.ndarray, grid_radius: np.ndarray
) -> np.ndarray:
    _require(radius[0] > 0.0 and mass[0] >= 0.0, "bad enclosed mass source")
    result = np.interp(grid_radius, radius, mass)
    inner = grid_radius < radius[0]
    result[inner] = mass[0] * (grid_radius[inner] / radius[0]) ** 3
    result[0] = 0.0
    return result


def _effective_density(radius: np.ndarray, mass: np.ndarray) -> np.ndarray:
    derivative = np.gradient(mass, radius, edge_order=2)
    density = np.empty_like(mass)
    density[1:] = derivative[1:] / (4.0 * math.pi * radius[1:] ** 2)
    density[1:] = np.maximum(density[1:], 0.0)
    density[0] = 3.0 * float(mass[1]) / (4.0 * math.pi * radius[1] ** 3)
    return density


def _potential_depth(radius: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    potential = np.zeros_like(radius)
    for index in range(radius.size - 2, -1, -1):
        potential[index] = potential[index + 1] + 0.5 * (
            acceleration[index] + acceleration[index + 1]
        ) * (radius[index + 1] - radius[index])
    return potential


def _positive_median(value: np.ndarray, label: str) -> float:
    admitted = value[np.isfinite(value) & (value > 0.0)]
    _require(admitted.size > 0, f"{label} has no positive reference nodes")
    return float(np.median(admitted))


def compile_xcop_spherical_source_drivers(
    radius_m: Sequence[float] | np.ndarray,
    gas_density_kg_m3: Sequence[float] | np.ndarray,
    stellar_enclosed_mass_kg: Sequence[float] | np.ndarray | None = None,
    *,
    missing_stellar_to_gas_ratio: float = 0.1,
    points: int = PRIMARY_POINTS,
) -> dict[str, Any]:
    """Compile spherical X-COP cause fields from density and optional stellar source mass."""

    radius = _source_radius(radius_m, allow_zero=False)
    gas_density = _vector(gas_density_kg_m3, "gas_density_kg_m3", nonnegative=True)
    _require(radius.shape == gas_density.shape, "X-COP source array shapes differ")
    _require(bool(np.all(gas_density > 0.0)), "gas density must be strictly positive")
    gas_mass_source = _observed_shell_mass(radius, gas_density)
    if stellar_enclosed_mass_kg is None:
        _require(
            math.isfinite(missing_stellar_to_gas_ratio) and missing_stellar_to_gas_ratio == 0.1,
            "only the frozen shared missing-stellar rule is allowed",
        )
        stellar_mass_source = missing_stellar_to_gas_ratio * gas_mass_source
        stellar_rule = "SHARED_0.1_GAS_MASS"
    else:
        stellar_mass_source = _vector(
            stellar_enclosed_mass_kg, "stellar_enclosed_mass_kg", nonnegative=True
        )
        _require(stellar_mass_source.shape == radius.shape, "stellar mass shape differs")
        _require(
            bool(np.all(np.diff(stellar_mass_source) >= 0.0)), "stellar mass must be cumulative"
        )
        stellar_rule = "SUPPLIED_CUMULATIVE_SOURCE_PROFILE"

    xi, grid_radius = _uniform_radius(float(radius[-1]), points)
    gas_mass = _interpolate_enclosed_mass(radius, gas_mass_source, grid_radius)
    stellar_mass = _interpolate_enclosed_mass(radius, stellar_mass_source, grid_radius)
    baryonic_mass = gas_mass + stellar_mass
    g_b = np.zeros_like(grid_radius)
    g_b[1:] = G_SI * baryonic_mass[1:] / grid_radius[1:] ** 2
    rho_b = _effective_density(grid_radius, baryonic_mass)
    potential = _potential_depth(grid_radius, g_b)
    surface_density = np.zeros_like(grid_radius)
    surface_density[1:] = baryonic_mass[1:] / (math.pi * grid_radius[1:] ** 2)
    slope = _log_slope(grid_radius, g_b)
    g_over_r = np.empty_like(g_b)
    g_over_r[1:] = g_b[1:] / grid_radius[1:]
    g_over_r[0] = g_over_r[1]
    tide = math.sqrt(2.0 / 3.0) * np.abs(4.0 * math.pi * G_SI * rho_b - 3.0 * g_over_r)
    gas_fraction = np.divide(
        gas_mass, baryonic_mass, out=np.zeros_like(gas_mass), where=baryonic_mass > 0.0
    )
    gas_fraction[0] = gas_fraction[1]
    threshold = 0.9 * baryonic_mass[-1]
    r_b_index = int(np.flatnonzero(baryonic_mass >= threshold)[0])
    rho_reference = _positive_median(rho_b, "rho_b")
    tide_reference = _positive_median(tide, "T_b")

    physical = {
        "D01_ACC": g_b,
        "D02_POT": potential,
        "D03_RAD": grid_radius,
        "D04_RHO": rho_b,
        "D05_SIG": surface_density,
        "D06_SLOPE": slope,
        "D07_TIDE": tide,
        "D13_GASF": gas_fraction,
    }
    bundle = _driver_bundle(
        domain="XCOP_SPHERICAL",
        xi=xi,
        radius_m=grid_radius,
        physical=physical,
        metadata={
            "source_rows_supplied": len(radius_m),
            "grid_points": points,
            "stellar_rule": stellar_rule,
            "r_out_m": float(grid_radius[-1]),
            "R_b_m": float(grid_radius[r_b_index]),
            "rho_reference_kg_m3": rho_reference,
            "tidal_reference_s_minus_2": tide_reference,
            "potential_outer_zero": float(potential[-1]),
        },
    )
    bundle["mass"] = {
        "gas_enclosed_kg": gas_mass,
        "stellar_enclosed_kg": stellar_mass,
        "baryonic_enclosed_kg": baryonic_mass,
    }
    return bundle


def architecture_parameter_cells(architecture_id: str) -> list[dict[str, float | int]]:
    _require(
        architecture_id in ARCHITECTURE_PARAMETERS,
        f"unknown static architecture: {architecture_id}",
    )
    grid = ARCHITECTURE_PARAMETERS[architecture_id]
    names = tuple(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def _validate_architecture_parameters(
    architecture_id: str, parameters: Mapping[str, float | int]
) -> dict[str, float | int]:
    expected = ARCHITECTURE_PARAMETERS[architecture_id]
    _exact_keys(parameters, set(expected), f"{architecture_id} parameters")
    result: dict[str, float | int] = {}
    for name, allowed in expected.items():
        value = parameters[name]
        _require(
            not isinstance(value, bool) and value in allowed, f"unfrozen {architecture_id}.{name}"
        )
        result[name] = value
    return result


def _tridiagonal_solve(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Deterministic Thomas solve for a nonsingular frozen tridiagonal system."""

    n = diagonal.size
    _require(
        lower.shape == upper.shape == (n - 1,) and rhs.shape == (n,),
        "bad tridiagonal system",
    )
    a = np.array(lower, dtype=float, copy=True)
    b = np.array(diagonal, dtype=float, copy=True)
    c = np.array(upper, dtype=float, copy=True)
    d = np.array(rhs, dtype=float, copy=True)
    for index in range(1, n):
        _require(abs(b[index - 1]) > 1e-30, "zero tridiagonal pivot")
        multiplier = a[index - 1] / b[index - 1]
        b[index] -= multiplier * c[index - 1]
        d[index] -= multiplier * d[index - 1]
    _require(abs(b[-1]) > 1e-30, "zero terminal tridiagonal pivot")
    solution = np.empty(n, dtype=float)
    solution[-1] = d[-1] / b[-1]
    for index in range(n - 2, -1, -1):
        solution[index] = (d[index] - c[index] * solution[index + 1]) / b[index]
    _require(bool(np.all(np.isfinite(solution))), "tridiagonal solve became nonfinite")
    return solution


def _a06_kernel(u: np.ndarray, xi: np.ndarray, ell: float) -> tuple[np.ndarray, float, float]:
    n = u.size
    h = float(xi[1] - xi[0])
    coefficient = (ell / h) ** 2
    lower = np.zeros(n - 1)
    diagonal = np.zeros(n)
    upper = np.zeros(n - 1)
    rhs = np.array(u, copy=True)
    diagonal[0] = 1.0
    upper[0] = -1.0
    rhs[0] = 0.0
    lower[:-1] = -coefficient
    diagonal[1:-1] = 1.0 + 2.0 * coefficient
    upper[1:] = -coefficient
    lower[-1] = -1.0
    diagonal[-1] = 1.0
    rhs[-1] = 0.0
    q = _tridiagonal_solve(lower, diagonal, upper, rhs)
    interior = q[1:-1] - coefficient * (q[:-2] - 2.0 * q[1:-1] + q[2:]) - u[1:-1]
    operator_residual = float(np.max(np.abs(interior), initial=0.0))
    boundary_residual = float(max(abs(q[1] - q[0]), abs(q[-1] - q[-2])))
    return q, operator_residual, boundary_residual


def _a12_massive(u: np.ndarray, xi: np.ndarray, mu: float) -> tuple[np.ndarray, float, float]:
    n = u.size
    h = float(xi[1] - xi[0])
    inv_h2 = 1.0 / (h * h)
    lower = np.zeros(n - 1)
    diagonal = np.zeros(n)
    upper = np.zeros(n - 1)
    rhs = np.array(u, copy=True)
    diagonal[0] = 1.0
    upper[0] = -1.0
    rhs[0] = 0.0
    lower[:-1] = -inv_h2
    diagonal[1:-1] = 2.0 * inv_h2 + mu * mu
    upper[1:] = -inv_h2
    lower[-1] = 0.0
    diagonal[-1] = 1.0
    rhs[-1] = 0.0
    q = _tridiagonal_solve(lower, diagonal, upper, rhs)
    canonical = (q[:-2] - 2.0 * q[1:-1] + q[2:]) * inv_h2 - mu * mu * q[1:-1] + u[1:-1]
    operator_residual = float(np.max(np.abs(canonical), initial=0.0))
    boundary_residual = float(max(abs(q[1] - q[0]), abs(q[-1])))
    return q, operator_residual, boundary_residual


def apply_static_architecture(
    architecture_id: str,
    u: Sequence[float] | np.ndarray,
    g_b_m_s2: Sequence[float] | np.ndarray,
    xi: Sequence[float] | np.ndarray,
    parameters: Mapping[str, float | int],
) -> dict[str, Any]:
    """Execute one exact static TWELL operator on an already uniform radial grid."""

    if architecture_id in TIME_SOURCE_BLOCKS:
        raise StaticSourceBlockedError(f"{architecture_id} is source-blocked on static real data")
    _require(architecture_id in STATIC_ARCHITECTURES, f"unknown architecture: {architecture_id}")
    p = _validate_architecture_parameters(architecture_id, parameters)
    driver = _vector(u, "normalized driver")
    acceleration = _vector(g_b_m_s2, "baryonic acceleration", nonnegative=True)
    coordinate = _vector(xi, "xi", nonnegative=True)
    _require(driver.shape == acceleration.shape == coordinate.shape, "architecture arrays differ")
    _require(coordinate[0] == 0.0 and coordinate[-1] == 1.0, "xi endpoints changed")
    spacing = np.diff(coordinate)
    _require(bool(np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-14)), "xi must be uniform")
    lam = float(p["lambda"])
    q: np.ndarray | None = None
    operator_residual = 0.0
    boundary_residual = 0.0
    iterations = 0
    auxiliary: dict[str, Any] = {}

    if architecture_id in ("A01_LAPSE", "A02_CLOCK", "A08_PERMITTIVITY"):
        factor = np.exp(lam * driver)
        if architecture_id == "A02_CLOCK":
            auxiliary["clock_rate"] = np.exp(-lam * driver)
        if architecture_id == "A08_PERMITTIVITY":
            auxiliary["epsilon"] = np.exp(-lam * driver)
    elif architecture_id == "A03_CONFORMAL":
        factor = 1.0 + lam * driver
    elif architecture_id == "A04_DISFORMAL":
        factor = 1.0 + lam * driver * driver / (1.0 + driver * driver)
    elif architecture_id == "A05_SLIP":
        factor = 1.0 + 0.5 * lam * driver
        auxiliary["phi_factor"] = factor
        auxiliary["psi_factor"] = 1.0 - 0.5 * lam * driver
    elif architecture_id == "A06_SPATIAL_KERNEL":
        q, operator_residual, boundary_residual = _a06_kernel(driver, coordinate, float(p["ell"]))
        factor = np.exp(lam * q)
    elif architecture_id == "A07_BOUNDARY":
        q = np.zeros_like(driver)
        h = float(spacing[0])
        for index in range(driver.size - 2, -1, -1):
            q[index] = q[index + 1] + 0.5 * h * (driver[index] + driver[index + 1])
        residual = q[:-1] - q[1:] - 0.5 * h * (driver[:-1] + driver[1:])
        operator_residual = float(np.max(np.abs(residual), initial=0.0))
        boundary_residual = float(abs(q[-1]))
        factor = np.exp(lam * q)
    elif architecture_id == "A09_ENTROPIC":
        factor = 1.0 + lam * driver / (1.0 + np.abs(driver))
    elif architecture_id == "A10_DENSITY_SCREEN":
        screen = 1.0 / (1.0 + (np.abs(driver) / float(p["u_c"])) ** int(p["n"]))
        auxiliary["screen"] = screen
        factor = 1.0 + lam * screen
    elif architecture_id == "A11_DERIV_SCREEN":
        slope = np.abs(_frozen_derivative(driver, coordinate))
        screen = 1.0 / (1.0 + (slope / float(p["s_c"])) ** int(p["n"]))
        auxiliary["derivative_magnitude"] = slope
        auxiliary["screen"] = screen
        factor = 1.0 + lam * screen
    elif architecture_id == "A12_MASSIVE":
        q, operator_residual, boundary_residual = _a12_massive(driver, coordinate, float(p["mu"]))
        factor = 1.0 + lam * q
    elif architecture_id == "A13_MIXED_MODE":
        q, operator_residual, boundary_residual = _a06_kernel(driver, coordinate, float(p["ell"]))
        mixed = math.cos(float(p["theta"])) * driver + math.sin(float(p["theta"])) * q
        auxiliary["q1"] = driver
        auxiliary["q2"] = q
        factor = 1.0 + lam * mixed
    elif architecture_id == "A14_PHASE":
        phase = 2.0 * math.pi * float(p["k"]) * coordinate + float(p["phi0"])
        auxiliary["phase"] = phase
        factor = 1.0 + lam * driver * np.cos(phase)
    else:
        _require(architecture_id == "A19_FEEDBACK", "unreachable architecture branch")
        kappa = float(p["kappa"])
        q = np.zeros_like(driver)
        tolerance = 1.0e-12
        for iterations in range(1, 129):
            updated = np.tanh(driver + kappa * q)
            delta = float(np.max(np.abs(updated - q)))
            q = updated
            if delta <= tolerance:
                break
        _require(iterations <= 128 and delta <= tolerance, "feedback did not converge")
        operator_residual = float(np.max(np.abs(q - np.tanh(driver + kappa * q))))
        factor = np.exp(lam * q)

    _require(bool(np.all(np.isfinite(factor))), f"{architecture_id} factor is nonfinite")
    effective = factor * acceleration
    state_arrays = {"q": q} if q is not None else {}
    state_arrays.update(auxiliary)
    return {
        "architecture_id": architecture_id,
        "parameters": dict(p),
        "factor": factor,
        "g_eff_m_s2": effective,
        "state": state_arrays,
        "diagnostics": {
            "operator_residual_max_abs": operator_residual,
            "boundary_residual_max_abs": boundary_residual,
            "iterations": iterations,
            "finite": bool(np.all(np.isfinite(effective))),
            "positive_factor": bool(np.all(factor > 0.0)),
            "null_parameter_gr_limit": bool(np.array_equal(factor, np.ones_like(factor)))
            if lam == 0.0
            else None,
        },
    }


def compile_static_architecture(
    architecture_id: str,
    source_xi: Sequence[float] | np.ndarray,
    source_u: Sequence[float] | np.ndarray,
    source_g_b_m_s2: Sequence[float] | np.ndarray,
    parameters: Mapping[str, float | int],
) -> dict[str, Any]:
    """Interpolate one source and run aligned 257/129 static convergence grids."""

    xi_source = _vector(source_xi, "source_xi", nonnegative=True)
    u_source = _vector(source_u, "source_u")
    g_source = _vector(source_g_b_m_s2, "source_g_b", nonnegative=True)
    _require(
        xi_source.shape == u_source.shape == g_source.shape, "source interpolation arrays differ"
    )
    _require(xi_source[0] == 0.0 and xi_source[-1] == 1.0, "source must cover xi=[0,1]")
    _require(bool(np.all(np.diff(xi_source) > 0.0)), "source xi must increase")
    primary_xi = np.linspace(0.0, 1.0, PRIMARY_POINTS)
    convergence_xi = np.linspace(0.0, 1.0, CONVERGENCE_POINTS)
    primary = apply_static_architecture(
        architecture_id,
        np.interp(primary_xi, xi_source, u_source),
        np.interp(primary_xi, xi_source, g_source),
        primary_xi,
        parameters,
    )
    convergence = apply_static_architecture(
        architecture_id,
        np.interp(convergence_xi, xi_source, u_source),
        np.interp(convergence_xi, xi_source, g_source),
        convergence_xi,
        parameters,
    )
    difference = np.asarray(primary["factor"])[::2] - np.asarray(convergence["factor"])
    convergence_max = float(np.max(np.abs(difference)))
    gates = {
        "SOURCE_INTERPOLATION": True,
        "FINITE": primary["diagnostics"]["finite"] and convergence["diagnostics"]["finite"],
        "POSITIVE_FACTOR": primary["diagnostics"]["positive_factor"]
        and convergence["diagnostics"]["positive_factor"],
        "OPERATOR_RESIDUAL": max(
            primary["diagnostics"]["operator_residual_max_abs"],
            convergence["diagnostics"]["operator_residual_max_abs"],
        )
        <= OPERATOR_RESIDUAL_TOLERANCE,
        "BOUNDARY": max(
            primary["diagnostics"]["boundary_residual_max_abs"],
            convergence["diagnostics"]["boundary_residual_max_abs"],
        )
        <= BOUNDARY_RESIDUAL_TOLERANCE,
        "PRIMARY_VS_CONVERGENCE": convergence_max <= CONVERGENCE_MAX_ABS_TOLERANCE,
    }
    _require(all(gates.values()), f"static architecture gate failed: {architecture_id}: {gates}")
    return {
        "architecture_id": architecture_id,
        "parameters": dict(parameters),
        "primary": primary,
        "convergence": convergence,
        "convergence_max_abs": convergence_max,
        "gates": gates,
        "factor_sha256": array_sha256(primary["factor"]),
    }


def compound_source_status(domain: str, compound_id: str) -> str:
    _require(domain in {"SPARC", "XCOP_SPHERICAL"}, "unknown source domain")
    _require(compound_id in COMPOUND_IDS, "unknown compound")
    if domain == "XCOP_SPHERICAL":
        return "SOURCE_AVAILABLE"
    status = {
        "X01": "SOURCE_BLOCKED_NO_HONEST_SPHERICAL_MASS_HISTORY",
        "X05": "SOURCE_BLOCKED_D02_UNAVAILABLE",
        "X10": "SOURCE_BLOCKED_D07_UNAVAILABLE",
        "X13": "SOURCE_BLOCKED_D02_UNAVAILABLE",
        "X17": "SOURCE_BLOCKED_D04_UNAVAILABLE",
        "X18": "SOURCE_BLOCKED_D02_UNAVAILABLE",
    }
    return status[compound_id]


def combine_compound_drivers(
    domain: str, compound_id: str, normalized_drivers: Mapping[str, Sequence[float] | np.ndarray]
) -> np.ndarray:
    """Execute one exact source-only compound program, honoring domain blocks."""

    status = compound_source_status(domain, compound_id)
    if status != "SOURCE_AVAILABLE":
        raise StaticSourceBlockedError(f"{compound_id} on {domain}: {status}")
    required = {
        "X01": ("D01_ACC", "D06_SLOPE"),
        "X05": ("D02_POT", "D06_SLOPE"),
        "X10": ("D01_ACC", "D07_TIDE"),
        "X13": ("D02_POT", "D13_GASF"),
        "X17": ("D04_RHO", "D03_RAD"),
        "X18": ("D01_ACC", "D02_POT"),
    }[compound_id]
    _require(set(normalized_drivers) == set(required), f"{compound_id} driver keys changed")
    u1 = _vector(normalized_drivers[required[0]], f"{compound_id}.u1")
    u2 = _vector(normalized_drivers[required[1]], f"{compound_id}.u2")
    _require(u1.shape == u2.shape, "compound driver shapes differ")
    if compound_id == "X01":
        result = np.clip(u1 * (1.0 + u2) / 2.0, -1.0, 1.0)
    elif compound_id == "X05":
        result = (u1 - u2) / 2.0
    elif compound_id == "X10":
        result = u1 / (1.0 + np.abs(u2))
    elif compound_id == "X13":
        result = (u1 + 2.0 * u2) / 3.0
    elif compound_id == "X17":
        result = u1 / (1.0 + np.abs(u2))
    else:
        result = (u1 + u2) / 2.0
    _require(bool(np.all(np.isfinite(result))), f"compound became nonfinite: {compound_id}")
    return result


def compile_compound_static(
    domain: str,
    compound_id: str,
    source_bundle: Mapping[str, Any],
    parameters: Mapping[str, float | int],
) -> dict[str, Any]:
    architecture = {
        "X01": "A02_CLOCK",
        "X05": "A01_LAPSE",
        "X10": "A11_DERIV_SCREEN",
        "X13": "A08_PERMITTIVITY",
        "X17": "A12_MASSIVE",
        "X18": "A13_MIXED_MODE",
    }[compound_id]
    required = {
        "X01": ("D01_ACC", "D06_SLOPE"),
        "X05": ("D02_POT", "D06_SLOPE"),
        "X10": ("D01_ACC", "D07_TIDE"),
        "X13": ("D02_POT", "D13_GASF"),
        "X17": ("D04_RHO", "D03_RAD"),
        "X18": ("D01_ACC", "D02_POT"),
    }[compound_id]
    normalized = source_bundle["normalized"]
    compound_u = combine_compound_drivers(
        domain, compound_id, {name: normalized[name] for name in required}
    )
    result = compile_static_architecture(
        architecture,
        source_bundle["xi"],
        compound_u,
        source_bundle["physical"]["D01_ACC"],
        parameters,
    )
    result["compound_id"] = compound_id
    result["compound_driver_sha256"] = array_sha256(compound_u)
    result["source_status"] = "SOURCE_AVAILABLE"
    return result


def wrong_control_program_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {row["id"]: content_sha256(row) for row in config["wrong_control_programs"]}


def apply_wrong_control(
    control_id: str,
    factor: Sequence[float] | np.ndarray,
    ordered_radius: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply one target-free factor transform; no response or object label is accepted."""

    gain = _vector(factor, "factor")
    radius = _vector(ordered_radius, "ordered_radius", nonnegative=True)
    _require(gain.shape == radius.shape, "wrong-control arrays differ")
    _require(bool(np.all(gain > 0.0)), "wrong-control factor must be positive")
    _require(bool(np.all(np.diff(radius) > 0.0)), "wrong control requires ordered source radius")
    if control_id == "IDENTITY":
        return gain.copy()
    if control_id == "RADIAL_FACTOR_REVERSAL":
        return gain[::-1].copy()
    raise OpenGravityStaticRadialAdapterError(f"unknown wrong control: {control_id}")


def gp01_nu_n(y: float | Sequence[float] | np.ndarray, n: int) -> float | np.ndarray:
    """Exact GP01-L interpolation bound to the committed foundation."""

    value = np.asarray(y, dtype=float)
    _require(n in GP01_GRID["n"] and isinstance(n, int), "unfrozen GP01 n")
    _require(bool(np.all(np.isfinite(value) & (value > 0.0))), "GP01 y must be finite and positive")
    result = np.exp(np.logaddexp(0.0, -n * np.log(value)) / (2.0 * n))
    return float(result) if result.ndim == 0 else result


def gp01_l_acceleration(
    g_b_m_s2: Sequence[float] | np.ndarray, *, n: int, a_star_m_s2: float = 1.2e-10
) -> np.ndarray:
    source = np.asarray(g_b_m_s2, dtype=float)
    _require(bool(np.all(np.isfinite(source) & (source >= 0.0))), "bad GP01-L source")
    _require(a_star_m_s2 == 1.2e-10, "GP01 acceleration reference changed")
    result = np.zeros_like(source)
    positive = source > 0.0
    if np.any(positive):
        result[positive] = source[positive] * np.asarray(
            gp01_nu_n(source[positive] / a_star_m_s2, n), dtype=float
        )
    return result


def gp01_environment_weight(
    rho_b: Sequence[float] | np.ndarray,
    tide_b: Sequence[float] | np.ndarray,
    *,
    rho_star: float,
    tide_star: float,
    q: int,
    tide_power: int,
) -> np.ndarray:
    density = np.asarray(rho_b, dtype=float)
    tide = np.asarray(tide_b, dtype=float)
    _require(density.shape == tide.shape, "GP01 environment arrays differ")
    _require(bool(np.all(np.isfinite(density) & (density >= 0.0))), "bad GP01 density")
    _require(bool(np.all(np.isfinite(tide) & (tide >= 0.0))), "bad GP01 tide")
    _require(math.isfinite(rho_star) and rho_star > 0.0, "bad GP01 rho_star")
    _require(math.isfinite(tide_star) and tide_star > 0.0, "bad GP01 tide_star")
    _require(q in (1, 2) and tide_power in (1, 2), "unfrozen GP01 gate exponent")
    return 1.0 / (1.0 + (density / rho_star) ** q + (tide / tide_star) ** tide_power)


def gp01_bounded_target(
    g_b_m_s2: Sequence[float] | np.ndarray,
    weight: Sequence[float] | np.ndarray,
    *,
    n: int,
    A_max: float,
    a_star_m_s2: float = 1.2e-10,
) -> np.ndarray:
    source = np.asarray(g_b_m_s2, dtype=float)
    gate = np.asarray(weight, dtype=float)
    _require(source.shape == gate.shape, "GP01 target arrays differ")
    _require(bool(np.all(np.isfinite(source) & (source >= 0.0))), "bad GP01 target source")
    _require(bool(np.all(np.isfinite(gate) & (gate >= 0.0) & (gate <= 1.0))), "bad GP01 W")
    _require(n in GP01_GRID["n"], "unfrozen GP01 n")
    _require(A_max in GP01_GRID["A_max"], "unfrozen GP01 A_max")
    _require(a_star_m_s2 == 1.2e-10, "GP01 acceleration reference changed")
    gamma_max = math.log(A_max)
    target = np.empty_like(source)
    positive = source > 0.0
    log_one_plus = np.logaddexp(0.0, n * (math.log(a_star_m_s2) - np.log(source[positive])))
    target[positive] = gate[positive] * gamma_max * np.tanh(log_one_plus / (2.0 * n * gamma_max))
    target[~positive] = gate[~positive] * gamma_max
    return target


def solve_spherical_gamma(
    radius_m: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    L_g_m: float,
) -> dict[str, Any]:
    """Solve the frozen radial Helmholtz equation with regular inner and zero outer rows."""

    radius = _vector(radius_m, "spherical radius", nonnegative=True)
    source = _vector(target, "Gamma target", nonnegative=True)
    _require(radius.shape == source.shape, "spherical solver arrays differ")
    _require(radius[0] == 0.0 and radius[-1] > 0.0, "spherical solver domain changed")
    spacing = np.diff(radius)
    _require(bool(np.allclose(spacing, spacing[0], rtol=1e-12, atol=0.0)), "radius is not uniform")
    _require(math.isfinite(L_g_m) and L_g_m >= 0.0, "bad GP01 length")
    h = float(spacing[0])
    n_points = radius.size
    lower = np.zeros(n_points - 1)
    diagonal = np.zeros(n_points)
    upper = np.zeros(n_points - 1)
    rhs = np.array(source, copy=True)
    diagonal[0] = 1.0
    upper[0] = -1.0
    rhs[0] = 0.0
    length2 = L_g_m * L_g_m
    for index in range(1, n_points - 1):
        radial = radius[index]
        laplace_lower = 1.0 / (h * h) - 1.0 / (radial * h)
        laplace_upper = 1.0 / (h * h) + 1.0 / (radial * h)
        lower[index - 1] = -length2 * laplace_lower
        diagonal[index] = 1.0 + 2.0 * length2 / (h * h)
        upper[index] = -length2 * laplace_upper
    lower[-1] = 0.0
    diagonal[-1] = 1.0
    rhs[-1] = 0.0
    gamma = _tridiagonal_solve(lower, diagonal, upper, rhs)
    laplacian = (gamma[2:] - 2.0 * gamma[1:-1] + gamma[:-2]) / (h * h) + (
        gamma[2:] - gamma[:-2]
    ) / (radius[1:-1] * h)
    residual = gamma[1:-1] - length2 * laplacian - source[1:-1]
    operator_residual = float(np.max(np.abs(residual), initial=0.0))
    boundary_residual = float(max(abs(gamma[0] - gamma[1]), abs(gamma[-1])))
    return {
        "gamma": gamma,
        "operator_residual_max_abs": operator_residual,
        "inner_regularity_residual": float(abs(gamma[0] - gamma[1])),
        "outer_dirichlet_residual": float(abs(gamma[-1])),
        "boundary_residual_max_abs": boundary_residual,
        "positive": bool(np.all(gamma >= -1.0e-12)),
        "finite": bool(np.all(np.isfinite(gamma))),
        "zero_length_recovers_interior_target": bool(np.array_equal(gamma[1:-1], source[1:-1]))
        if L_g_m == 0.0
        else None,
    }


def spherical_constant_target_solution(
    radius_m: Sequence[float] | np.ndarray, *, target: float, L_g_m: float
) -> np.ndarray:
    """Continuous regular analytic control for a constant target and zero outer Gamma."""

    radius = np.asarray(radius_m, dtype=float)
    _require(target >= 0.0 and math.isfinite(target), "bad analytic target")
    _require(L_g_m > 0.0 and math.isfinite(L_g_m), "analytic control requires L_g>0")
    outer = float(radius[-1])
    denominator = math.sinh(outer / L_g_m)
    result = np.empty_like(radius)
    result[0] = target * (1.0 - (outer / L_g_m) / denominator)
    positive = radius > 0.0
    result[positive] = target * (
        1.0 - outer * np.sinh(radius[positive] / L_g_m) / (radius[positive] * denominator)
    )
    return result


def integrated_spherical_flux(
    radius_m: Sequence[float] | np.ndarray,
    enclosed_mass_kg: Sequence[float] | np.ndarray,
    gamma: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Apply exp(Gamma) only after the spherical source flux has been integrated."""

    radius = _vector(radius_m, "flux radius", nonnegative=True)
    mass = _vector(enclosed_mass_kg, "enclosed mass", nonnegative=True)
    gain_log = _vector(gamma, "Gamma")
    _require(radius.shape == mass.shape == gain_log.shape, "flux arrays differ")
    _require(radius[0] == 0.0 and mass[0] == 0.0, "spherical flux origin changed")
    _require(bool(np.all(np.diff(mass) >= 0.0)), "enclosed mass is not monotone")
    g_b = np.zeros_like(radius)
    g_b[1:] = G_SI * mass[1:] / radius[1:] ** 2
    factor = np.exp(gain_log)
    g_eff = factor * g_b
    lhs = np.exp(-gain_log[1:]) * g_eff[1:] * radius[1:] ** 2
    rhs = G_SI * mass[1:]
    scale = np.maximum(np.abs(rhs), 1.0e-300)
    residual = float(np.max(np.abs(lhs - rhs) / scale, initial=0.0))
    return {
        "g_b_m_s2": g_b,
        "factor": factor,
        "g_eff_m_s2": g_eff,
        "integrated_flux_relative_residual": residual,
    }


def _gp01_cell_parameters() -> list[dict[str, float | int]]:
    names = tuple(GP01_GRID)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(GP01_GRID[name] for name in names))
    ]


def enumerate_gp01_spherical_cells(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate the exact 1,296 target-free spherical GP01-ELLIPTIC cells."""

    _require(source_bundle["domain"] == "XCOP_SPHERICAL", "GP01 elliptic requires spherical source")
    source_radius = np.asarray(source_bundle["radius_m"], dtype=float)
    source_mass = np.asarray(source_bundle["mass"]["baryonic_enclosed_kg"], dtype=float)
    source_rho = np.asarray(source_bundle["physical"]["D04_RHO"], dtype=float)
    source_tide = np.asarray(source_bundle["physical"]["D07_TIDE"], dtype=float)
    r_out = float(source_radius[-1])
    r_b = float(source_bundle["metadata"]["R_b_m"])
    rho_reference = float(source_bundle["metadata"]["rho_reference_kg_m3"])
    tide_reference = float(source_bundle["metadata"]["tidal_reference_s_minus_2"])
    cells: list[dict[str, Any]] = []
    maximum_operator = 0.0
    maximum_boundary = 0.0
    maximum_convergence = 0.0
    maximum_flux = 0.0
    for parameters in _gp01_cell_parameters():
        cell_solutions: dict[int, dict[str, Any]] = {}
        for points in (PRIMARY_POINTS, CONVERGENCE_POINTS):
            _, radius = _uniform_radius(r_out, points)
            mass = np.interp(radius, source_radius, source_mass)
            mass[0] = 0.0
            rho = np.interp(radius, source_radius, source_rho)
            tide = np.interp(radius, source_radius, source_tide)
            g_b = np.zeros_like(radius)
            g_b[1:] = G_SI * mass[1:] / radius[1:] ** 2
            weight = gp01_environment_weight(
                rho,
                tide,
                rho_star=float(parameters["rho_ratio"]) * rho_reference,
                tide_star=float(parameters["tide_ratio"]) * tide_reference,
                q=int(parameters["q"]),
                tide_power=int(parameters["tide_power"]),
            )
            target = gp01_bounded_target(
                g_b, weight, n=int(parameters["n"]), A_max=float(parameters["A_max"])
            )
            solved = solve_spherical_gamma(radius, target, L_g_m=float(parameters["L_ratio"]) * r_b)
            flux = integrated_spherical_flux(radius, mass, solved["gamma"])
            cell_solutions[points] = {
                "radius": radius,
                "target": target,
                "solved": solved,
                "flux": flux,
            }
        primary = cell_solutions[PRIMARY_POINTS]
        convergence = cell_solutions[CONVERGENCE_POINTS]
        # Boundary rows are gated independently.  Compare the aligned PDE rows only;
        # the L_g=0 inner first-difference row samples target at h and 2h by design.
        convergence_max = float(
            np.max(
                np.abs(primary["solved"]["gamma"][::2][1:-1] - convergence["solved"]["gamma"][1:-1])
            )
        )
        operator = max(
            primary["solved"]["operator_residual_max_abs"],
            convergence["solved"]["operator_residual_max_abs"],
        )
        boundary = max(
            primary["solved"]["boundary_residual_max_abs"],
            convergence["solved"]["boundary_residual_max_abs"],
        )
        flux_residual = max(
            primary["flux"]["integrated_flux_relative_residual"],
            convergence["flux"]["integrated_flux_relative_residual"],
        )
        gamma = primary["solved"]["gamma"]
        gamma_max = math.log(float(parameters["A_max"]))
        gates = {
            "FINITE": primary["solved"]["finite"] and convergence["solved"]["finite"],
            "POSITIVE": primary["solved"]["positive"] and convergence["solved"]["positive"],
            "BOUNDED": bool(np.min(gamma) >= -1.0e-12 and np.max(gamma) <= gamma_max + 1.0e-12),
            "OPERATOR_RESIDUAL": operator <= 1.0e-9,
            "BOUNDARY": boundary <= 1.0e-10,
            "PRIMARY_VS_CONVERGENCE": convergence_max <= 0.02,
            "INTEGRATED_SPHERICAL_FLUX": flux_residual <= 1.0e-12,
        }
        _require(all(gates.values()), f"GP01 cell gate failed: {parameters}: {gates}")
        cell_id = (
            f"GP01E-n{parameters['n']}-A{parameters['A_max']:g}"
            f"-rho{parameters['rho_ratio']:g}-T{parameters['tide_ratio']:g}"
            f"-q{parameters['q']}-p{parameters['tide_power']}-L{parameters['L_ratio']:g}"
        )
        cells.append(
            {
                "cell_id": cell_id,
                "parameters": parameters,
                "target_sha256": array_sha256(primary["target"]),
                "Gamma_sha256": array_sha256(gamma),
                "prediction_sha256": array_sha256(primary["flux"]["g_eff_m_s2"]),
                "factor_sha256": array_sha256(primary["flux"]["factor"]),
                "operator_residual_max_abs": operator,
                "boundary_residual_max_abs": boundary,
                "primary_vs_convergence_max_abs": convergence_max,
                "integrated_flux_relative_residual": flux_residual,
                "Gamma_min": float(np.min(gamma)),
                "Gamma_max": float(np.max(gamma)),
                "zero_length_recovers_interior_target": primary["solved"][
                    "zero_length_recovers_interior_target"
                ],
                "gates": gates,
            }
        )
        maximum_operator = max(maximum_operator, operator)
        maximum_boundary = max(maximum_boundary, boundary)
        maximum_convergence = max(maximum_convergence, convergence_max)
        maximum_flux = max(maximum_flux, flux_residual)
    _require(len(cells) == 1296, "GP01 exact cell count changed")
    return {
        "cell_count": len(cells),
        "cells": cells,
        "ordered_cell_ids_sha256": content_sha256([row["cell_id"] for row in cells]),
        "ordered_cell_program_sha256": content_sha256(
            [{"cell_id": row["cell_id"], "parameters": row["parameters"]} for row in cells]
        ),
        "prediction_root_sha256": content_sha256([row["prediction_sha256"] for row in cells]),
        "factor_root_sha256": content_sha256([row["factor_sha256"] for row in cells]),
        "maximum_operator_residual": maximum_operator,
        "maximum_boundary_residual": maximum_boundary,
        "maximum_primary_vs_convergence": maximum_convergence,
        "maximum_integrated_flux_relative_residual": maximum_flux,
        "all_gates_pass": all(all(row["gates"].values()) for row in cells),
    }


def program_hash_report(config: Mapping[str, Any]) -> dict[str, Any]:
    driver_programs = config["driver_contract"]["normalized_programs"]
    architecture_programs = config["architecture_programs"]
    compound_programs = config["compound_programs"]
    controls = config["wrong_control_programs"]
    return {
        "full_formula_program_root_sha256": _program_root(config),
        "driver_program_sha256": {
            name: content_sha256({"id": name, "program": program})
            for name, program in sorted(driver_programs.items())
        },
        "architecture_program_sha256": {
            row["id"]: content_sha256(row) for row in architecture_programs
        },
        "compound_program_sha256": {row["id"]: content_sha256(row) for row in compound_programs},
        "gp01_program_sha256": content_sha256(config["gp01_program"]),
        "wrong_control_program_sha256": {row["id"]: content_sha256(row) for row in controls},
    }


FORMULA_PAYLOAD_FIELDS = (
    "source",
    "coupling",
    "action_or_equations",
    "initial_conditions",
    "boundaries",
    "degrees_of_freedom",
    "propagation",
    "state_rule",
    "closures",
    "ledgers",
    "structure",
    "dimensions",
    "parameter_cells",
    "priors",
    "screens",
    "limiting_cases",
)


def mechanism_formula_sha256(card: Mapping[str, Any]) -> str:
    return registry_content_sha256({name: card[name] for name in FORMULA_PAYLOAD_FIELDS})


def _parameter_rows(
    stable_id: str, grid: Mapping[str, Sequence[float | int]]
) -> list[dict[str, Any]]:
    units = {
        "lambda": "1",
        "ell": "normalized_radius",
        "u_c": "1",
        "n": "1",
        "s_c": "1",
        "mu": "inverse_normalized_radius",
        "theta": "rad",
        "k": "cycle_per_normalized_radius",
        "phi0": "rad",
        "kappa": "1",
        "A_max": "1",
        "rho_ratio": "1",
        "tide_ratio": "1",
        "q": "1",
        "tide_power": "1",
        "L_ratio": "1",
        "L_reset_ratio": "1",
        "c_Gamma_over_c": "1",
    }
    rows = []
    for parameter, values in grid.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "cell_id": f"{stable_id}-{parameter.upper()}-{index:02d}",
                    "parameter": parameter,
                    "value": value,
                    "unit": units[parameter],
                    "frozen": True,
                }
            )
    return rows


def _base_mechanism_card(
    *,
    stable_id: str,
    identity_class: str,
    ontology: str,
    scientific_status: str,
    kind: str,
    executable: bool,
    exact_expressions: Sequence[str],
    source: str,
    allowed_inputs: Sequence[str],
    state_mode: str,
    parameter_rows: Sequence[Mapping[str, Any]],
    code_sha256: str,
    configuration_sha256: str,
    program_fingerprint: str,
    synthetic_fingerprint: str,
    boundaries: Sequence[str],
    limiting_cases: Sequence[str],
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "schema_version": "invariant-open-gravity-mechanism-card-1.0",
        "card_id": f"{stable_id}@1.0.0",
        "stable_concept_id": stable_id,
        "semantic_version": "1.0.0",
        "identity_class": identity_class,
        "parents": [],
        "author_agent": "open-gravity-static-radial-adapter-v1",
        "provenance": {
            "created_at_utc": "2026-08-30T00:00:00Z",
            "origin_timing": "PRE_RESPONSE",
            "origin_artifacts": [
                CONFIG_PATH.as_posix(),
                "commits:35f70938f158c81971b2e1b838371b09d9fcee2c,ed2988546fb1165d9efe5e62d52cddebc7b1a79d",
            ],
            "residual_access_lineage": [],
        },
        "lay_mechanism": "A frozen source-derived radial program maps declared baryonic cause fields to an acceleration factor without consulting a response.",
        "novelty_claim": "Formula-variant registration only; historical novelty and scientific validity are not claimed.",
        "ontology": [ontology],
        "scientific_status": scientific_status,
        "operational_variables": [
            {
                "symbol": "u_D",
                "operational_definition": "dimensionless source-only driver on the frozen radial grid",
                "dimension": "1",
                "observable_or_latent": "SOURCE_DERIVED",
            },
            {
                "symbol": "A",
                "operational_definition": "positive radial acceleration factor",
                "dimension": "1",
                "observable_or_latent": "LATENT_FIELD",
            },
        ],
        "source": source,
        "coupling": "g_eff=A*g_b in the radial matter-acceleration phenomenology only",
        "action_or_equations": {
            "kind": kind,
            "exact_expressions": list(exact_expressions),
            "executable": executable,
        },
        "initial_conditions": [
            "static local states have no time initial data; iterative states start from the frozen zero state"
        ],
        "boundaries": list(boundaries),
        "degrees_of_freedom": {
            "fields": ["source-derived radial factor A"],
            "spin_helicity": "not assigned by this effective radial adapter",
            "mass": "not assigned unless explicitly present in the frozen operator",
            "statistics": "classical deterministic",
            "state": "static source-derived radial profile",
            "quantum_applicability": "NOT_APPLICABLE",
        },
        "propagation": {
            "speed": "not established by a static radial adapter",
            "dispersion": "not established",
            "polarization": "not established",
            "attenuation": "not established",
            "range": "the frozen finite radial source domain",
            "static_limit": "the exact expressions in this card",
        },
        "state_rule": {"mode": state_mode, "exact_rule": ";".join(exact_expressions)},
        "closures": {
            "matter": "radial effective acceleration only",
            "photon": "L0_NO_LIGHT_CLAIM",
            "gravitational_wave": "GW0_REQUIRE_GR_RECOVERY_NO_NEW_WAVE_CLAIM",
            "quantum_laboratory": "Q0_NO_QUANTUM_LAB_CLAIM",
            "capture": "C0_ISOLATED_CONSERVATIVE",
            "cosmology": "COS0_NO_COSMOLOGY_CLAIM",
        },
        "ledgers": {
            "energy": "no covariant energy completion claimed",
            "momentum": "no covariant momentum completion claimed",
            "entropy": "no entropy production claimed",
            "information": "source-only deterministic program with zero response lineage",
        },
        "structure": {
            "symmetries": ["frozen one-dimensional radial source ordering"],
            "covariance_or_frame": "static radial phenomenology; no covariant completion claimed",
            "equivalence_behavior": "universal frozen parameters; prediction degeneracies are explicitly linked",
            "causal_structure": "not inferred from a static source profile",
        },
        "dimensions": ["[u_D]=[A]=1", "[g_eff]=[g_b]=m s^-2"],
        "parameter_cells": [dict(row) for row in parameter_rows],
        "priors": ["all listed parameter cells are frozen before response access"],
        "screens": [
            "finite",
            "positive factor",
            "operator residual",
            "boundary",
            "257-versus-129 convergence",
        ],
        "limiting_cases": list(limiting_cases),
        "source_only_data_contract": {
            "allowed_inputs": list(allowed_inputs),
            "forbidden_response_inputs": [
                "SPARC Vobs or any motion response",
                "X-COP pressure or temperature",
                "lensing or inferred total mass",
                "residuals scores likelihoods confirmation or independent responses",
            ],
            "construction_before_response": True,
            "missing_data_action": "SOURCE_BLOCKED",
        },
        "synthetic_falsifier": "Reject if dimensions, finiteness, positivity, exact operator, boundary, replay, or aligned-grid convergence fails.",
        "real_data_discriminator": "Deferred target-blind executor may apply this source-only factor only after manifest freeze; this card authorizes no score.",
        "prior_art": [
            {
                "citation": "open-gravity committed GP01 foundation and primary-source metadata",
                "relationship": "operational boundary and known-family comparison; no novelty inference",
            }
        ],
        "equivalence_fingerprint": {
            "canonical_symbolic_sha256": program_fingerprint,
            "analytic_limits_sha256": content_sha256(list(limiting_cases)),
            "synthetic_fingerprint_sha256": synthetic_fingerprint,
            "observable_fingerprint_sha256": content_sha256(
                {"matter": "radial effective acceleration only", "photon": "L0_NO_LIGHT_CLAIM"}
            ),
        },
        "version_change": {
            "kind": "INITIAL_REGISTRATION",
            "previous_card_id": None,
            "previous_card_sha256": None,
            "changed_facets": [],
            "prior_result_retained": True,
            "replay_all_affected": False,
        },
        "hashes": {
            "code_sha256": code_sha256,
            "data_sha256": content_sha256(
                {"allowed_inputs": list(allowed_inputs), "source": source, "response_inputs": 0}
            ),
            "environment_sha256": content_sha256(
                {"grid": [PRIMARY_POINTS, CONVERGENCE_POINTS], "float": "IEEE754_BINARY64"}
            ),
            "configuration_sha256": configuration_sha256,
            "formula_sha256": "0" * 64,
        },
    }
    card["hashes"]["formula_sha256"] = mechanism_formula_sha256(card)
    return card


def _domain_execution_rows(
    *, source_status: Mapping[str, str], candidate_status: str
) -> dict[str, dict[str, Any]]:
    source_by_campaign_domain = {
        "GALAXIES": "SPARC",
        "GROUPS": "NOT_IN_ADAPTER",
        "CLUSTERS": "XCOP_SPHERICAL",
        "LENSING": "NOT_IN_ADAPTER",
    }
    rows: dict[str, dict[str, Any]] = {}
    for domain, source_domain in source_by_campaign_domain.items():
        status = source_status.get(source_domain, "NOT_APPLICABLE")
        if candidate_status == "QUARANTINED_REVISION_REQUIRED":
            disposition = "QUARANTINED" if status == "INCOMPLETE_QUARANTINE" else "NOT_APPLICABLE"
        elif candidate_status == "SOURCE_BLOCKED":
            disposition = (
                "SOURCE_BLOCKED" if status.startswith("SOURCE_BLOCKED") else "NOT_APPLICABLE"
            )
        elif candidate_status == "KNOWN_REWRITE_NONINDEPENDENT":
            disposition = (
                "KNOWN_REWRITE_NONINDEPENDENT" if status == "SOURCE_AVAILABLE" else "NOT_APPLICABLE"
            )
        else:
            disposition = "THEORY_ONLY" if status == "SOURCE_AVAILABLE" else "NOT_APPLICABLE"
        rows[domain] = {
            "eligible": False,
            "execution_disposition": disposition,
            "scored": False,
            "source_contract_sha256": content_sha256(
                {"campaign_domain": domain, "source_domain": source_domain, "source_status": status}
            ),
        }
    return rows


def _typed_card_wrapper(
    card: Mapping[str, Any],
    *,
    source_status: Mapping[str, str],
    lane_hint: str,
    prediction_group: str,
    manifest_authority: bool,
) -> dict[str, Any]:
    if card["action_or_equations"]["kind"] == "ACTION_PLACEHOLDER":
        admission = "QUARANTINED_REVISION_REQUIRED"
        candidate_status = "QUARANTINED_REVISION_REQUIRED"
    elif card["identity_class"] == "KNOWN_REWRITE":
        admission = "KNOWN_REWRITE_NONINDEPENDENT"
        candidate_status = "KNOWN_REWRITE_NONINDEPENDENT"
    elif card["action_or_equations"]["executable"] is not True:
        admission = "SOURCE_BLOCKED"
        candidate_status = "SOURCE_BLOCKED"
    else:
        admission = "READY_FOR_THEORY_GATES"
        candidate_status = "REGISTERED_THEORY_ONLY"
    formula_sha = str(card["hashes"]["formula_sha256"])
    return {
        "card": dict(card),
        "card_sha256": registry_content_sha256(card),
        "formula_sha256": formula_sha,
        "configuration_sha256": card["hashes"]["configuration_sha256"],
        "equivalence_fingerprint_sha256": registry_content_sha256(card["equivalence_fingerprint"]),
        "formula_equivalence_family_id": f"EQ-{formula_sha[:24]}",
        "prediction_degeneracy_sha256": content_sha256({"prediction_group": prediction_group}),
        "registry_admission_status": admission,
        "candidate_status_hint": candidate_status,
        "domain_source_status": dict(source_status),
        "domain_execution": _domain_execution_rows(
            source_status=source_status, candidate_status=candidate_status
        ),
        "lane_hint": lane_hint,
        "lane_assignment_authority": False,
        "manifest_authority_after_required_bindings": manifest_authority,
        "anonymous_formula_seed": f"F-{formula_sha[:20]}",
    }


def typed_mechanism_card_catalog(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete schema-shaped adapter card set for manifest composition."""

    code_sha = file_sha256(root / MODULE_PATH)
    configuration_sha = content_sha256(config)
    architecture_rows = {row["id"]: row for row in config["architecture_programs"]}
    compound_rows = {row["id"]: row for row in config["compound_programs"]}
    driver_programs = config["driver_contract"]["normalized_programs"]
    state_modes = {
        **{name: "LOCAL" for name in STATIC_ARCHITECTURES},
        "A06_SPATIAL_KERNEL": "SPATIAL_HISTORY",
        "A07_BOUNDARY": "SPATIAL_HISTORY",
        "A12_MASSIVE": "SPATIAL_HISTORY",
        "A13_MIXED_MODE": "SPATIAL_HISTORY",
        "A14_PHASE": "PHASE",
        "A19_FEEDBACK": "FEEDBACK",
    }
    field_architectures = {
        "A06_SPATIAL_KERNEL",
        "A07_BOUNDARY",
        "A12_MASSIVE",
        "A13_MIXED_MODE",
        "A19_FEEDBACK",
    }
    cards: list[dict[str, Any]] = []

    for architecture_id in STATIC_ARCHITECTURES:
        architecture_number = int(architecture_id[1:3])
        architecture = architecture_rows[architecture_id]
        for driver_id in XCOP_DRIVERS:
            driver_number = int(driver_id[1:3])
            stable_id = f"TW2-A{architecture_number:02d}-D{driver_number:02d}"
            expressions = [driver_programs[driver_id], architecture["operator"], "g_eff=A*g_b"]
            program = {
                "stable_id": stable_id,
                "driver": driver_id,
                "architecture": architecture,
                "expressions": expressions,
            }
            prediction_group = (
                f"EXPONENTIAL-LOCAL-{driver_id}"
                if architecture_id in {"A01_LAPSE", "A02_CLOCK", "A08_PERMITTIVITY"}
                else stable_id
            )
            card = _base_mechanism_card(
                stable_id=stable_id,
                identity_class="FORMULA_VARIANT",
                ontology="TWELL-400-v2",
                scientific_status="H_HYPOTHESIS",
                kind="FIELD_EQUATIONS"
                if architecture_id in field_architectures
                else "EFFECTIVE_LAW",
                executable=True,
                exact_expressions=expressions,
                source=f"{driver_id} source-only radial cause field",
                allowed_inputs=[driver_id, "ordered source radius", "baryonic acceleration"],
                state_mode=state_modes[architecture_id],
                parameter_rows=_parameter_rows(stable_id, ARCHITECTURE_PARAMETERS[architecture_id]),
                code_sha256=code_sha,
                configuration_sha256=configuration_sha,
                program_fingerprint=content_sha256(program),
                synthetic_fingerprint=content_sha256(
                    {"prediction_group": prediction_group, "grid": [257, 129]}
                ),
                boundaries=[architecture["operator"], "xi=r/r_out on [0,1]"],
                limiting_cases=["lambda=0 gives A=1", "source=0 remains finite"],
            )
            source_status = {
                "SPARC": "SOURCE_AVAILABLE"
                if driver_id in SPARC_DRIVERS
                else "SOURCE_BLOCKED_DRIVER_UNAVAILABLE",
                "XCOP_SPHERICAL": "SOURCE_AVAILABLE",
            }
            cards.append(
                _typed_card_wrapper(
                    card,
                    source_status=source_status,
                    lane_hint="CORE",
                    prediction_group=prediction_group,
                    manifest_authority=False,
                )
            )

    compound_architecture = {
        "X01": "A02_CLOCK",
        "X05": "A01_LAPSE",
        "X10": "A11_DERIV_SCREEN",
        "X13": "A08_PERMITTIVITY",
        "X17": "A12_MASSIVE",
        "X18": "A13_MIXED_MODE",
    }
    for compound_id in COMPOUND_IDS:
        row = compound_rows[compound_id]
        architecture_id = compound_architecture[compound_id]
        stable_id = compound_id
        expressions = [
            row["operator"],
            architecture_rows[architecture_id]["operator"],
            "g_eff=A*g_b",
        ]
        card = _base_mechanism_card(
            stable_id=stable_id,
            identity_class="FORMULA_VARIANT",
            ontology="TWELL-400-v2",
            scientific_status="H_HYPOTHESIS",
            kind="FIELD_EQUATIONS" if architecture_id in field_architectures else "EFFECTIVE_LAW",
            executable=True,
            exact_expressions=expressions,
            source=f"compound source-only drivers {','.join(row['drivers'])}",
            allowed_inputs=list(row["drivers"])
            + ["ordered source radius", "baryonic acceleration"],
            state_mode=state_modes[architecture_id],
            parameter_rows=_parameter_rows(stable_id, ARCHITECTURE_PARAMETERS[architecture_id]),
            code_sha256=code_sha,
            configuration_sha256=configuration_sha,
            program_fingerprint=content_sha256(row),
            synthetic_fingerprint=content_sha256({"compound": compound_id, "grid": [257, 129]}),
            boundaries=[architecture_rows[architecture_id]["operator"], "xi=r/r_out on [0,1]"],
            limiting_cases=["lambda=0 gives A=1", "compound driver remains bounded"],
        )
        cards.append(
            _typed_card_wrapper(
                card,
                source_status={
                    "SPARC": row["sparc_status"],
                    "XCOP_SPHERICAL": row["xcop_status"],
                },
                lane_hint="CORE",
                prediction_group=compound_id,
                manifest_authority=False,
            )
        )

    gp = config["gp01_program"]
    environment_grid = {
        "n": GP01_GRID["n"],
        "A_max": GP01_GRID["A_max"],
        "rho_ratio": GP01_GRID["rho_ratio"],
        "tide_ratio": GP01_GRID["tide_ratio"],
        "q": GP01_GRID["q"],
        "tide_power": GP01_GRID["tide_power"],
    }
    gp_specs = [
        {
            "stable_id": "GP01-L",
            "identity": "FORMULA_VARIANT",
            "kind": "EFFECTIVE_LAW",
            "executable": True,
            "expressions": [gp["local_gain"], "g=nu_n(g_b/a_star)*g_b"],
            "source": "D01_ACC baryonic source acceleration",
            "allowed": ["D01_ACC", "baryonic acceleration"],
            "mode": "LOCAL",
            "grid": {"n": GP01_GRID["n"]},
            "boundaries": ["local algebraic source point; no outer fit"],
            "limits": ["high field gives g=g_b", "deep field gives g=sqrt(a_star*g_b)"],
            "status": {"SPARC": "SOURCE_AVAILABLE", "XCOP_SPHERICAL": "SOURCE_AVAILABLE"},
            "lane": "CORE",
            "prediction": "GP01-L-LOCAL",
        },
        {
            "stable_id": "GP01-AQUAL",
            "identity": "KNOWN_REWRITE",
            "kind": "FIELD_EQUATIONS",
            "executable": True,
            "expressions": [gp["aqual_equation"]],
            "source": "three-dimensional baryonic density or explicitly spherical source reduction",
            "allowed": ["D04_RHO", "baryonic source density", "declared field boundary"],
            "mode": "SPATIAL_HISTORY",
            "grid": {"n": GP01_GRID["n"]},
            "boundaries": ["finite inner gradient", "declared outer flux and one gauge point"],
            "limits": ["spherical prediction equals GP01-L and is scored once"],
            "status": {
                "SPARC": "SOURCE_BLOCKED_NO_THREE_DIMENSIONAL_BARYON_DENSITY_BOUNDARY",
                "XCOP_SPHERICAL": "SOURCE_AVAILABLE",
            },
            "lane": "CORE",
            "prediction": "GP01-L-AQUAL-SPHERICAL",
        },
        {
            "stable_id": "GP01-T1",
            "identity": "FORMULA_VARIANT",
            "kind": "FIELD_EQUATIONS",
            "executable": False,
            "expressions": [gp["t1_equation"], gp["environment_gate"], gp["bounded_target"]],
            "source": "one anchored outward baryonic field line with no null or separatrix",
            "allowed": ["D01_ACC", "D04_RHO", "D07_TIDE", "unique y=100 anchor"],
            "mode": "SPATIAL_HISTORY",
            "grid": {**environment_grid, "L_reset_ratio": tuple(gp["L_reset_over_R_b_grid"])},
            "boundaries": [
                "first outward y=100 anchor",
                "quarantine nulls separatrices and multiple anchors",
            ],
            "limits": ["static profiles without a unique anchor remain source-blocked"],
            "status": {
                "SPARC": "SOURCE_BLOCKED_NO_HONEST_FIELD_LINE_HISTORY",
                "XCOP_SPHERICAL": "SOURCE_BLOCKED_NO_UNIQUE_Y100_ANCHOR",
            },
            "lane": "ADJACENT",
            "prediction": "GP01-T1-BLOCKED",
        },
        {
            "stable_id": "GP01-T2",
            "identity": "FORMULA_VARIANT",
            "kind": "FIELD_EQUATIONS",
            "executable": False,
            "expressions": [gp["t2_equation"], gp["environment_gate"], gp["bounded_target"]],
            "source": "one anchored outward baryonic field line with no null or separatrix",
            "allowed": ["D01_ACC", "D04_RHO", "D07_TIDE", "unique y=100 anchor"],
            "mode": "SPATIAL_HISTORY",
            "grid": {**environment_grid, "L_ratio": (0.25, 1.0, 4.0)},
            "boundaries": [
                "first outward y=100 anchor",
                "quarantine nulls separatrices and multiple anchors",
            ],
            "limits": ["static profiles without a unique anchor remain source-blocked"],
            "status": {
                "SPARC": "SOURCE_BLOCKED_NO_HONEST_FIELD_LINE_HISTORY",
                "XCOP_SPHERICAL": "SOURCE_BLOCKED_NO_UNIQUE_Y100_ANCHOR",
            },
            "lane": "ADJACENT",
            "prediction": "GP01-T2-BLOCKED",
        },
        {
            "stable_id": "GP01-ELLIPTIC",
            "identity": "FORMULA_VARIANT",
            "kind": "FIELD_EQUATIONS",
            "executable": True,
            "expressions": [
                gp["environment_gate"],
                gp["bounded_target"],
                gp["elliptic_equation"],
                gp["coupled_flux"],
            ],
            "source": "spherical baryonic mass, acceleration, density, and trace-free tide",
            "allowed": ["D01_ACC", "D04_RHO", "D07_TIDE", "baryonic enclosed mass"],
            "mode": "SPATIAL_HISTORY",
            "grid": GP01_GRID,
            "boundaries": [gp["inner_boundary"], gp["outer_boundary"]],
            "limits": [
                "L_g=0 recovers the bounded interior target",
                "Gamma=0 recovers spherical source flux",
            ],
            "status": {
                "SPARC": "SOURCE_BLOCKED_NO_SPHERICAL_MASS_DENSITY_TIDE_HISTORY",
                "XCOP_SPHERICAL": "SOURCE_AVAILABLE",
            },
            "lane": "CORE",
            "prediction": "GP01-ELLIPTIC-SPHERICAL",
        },
        {
            "stable_id": "GP01-TELEGRAPH",
            "identity": "FORMULA_VARIANT",
            "kind": "FIELD_EQUATIONS",
            "executable": False,
            "expressions": [gp["telegraph_equation"], gp["bounded_target"]],
            "source": "time-resolved causal baryonic source history absent from this static adapter",
            "allowed": ["time-resolved independent baryonic source history"],
            "mode": "TEMPORAL_MEMORY",
            "grid": {
                **environment_grid,
                "L_ratio": (0.25, 1.0, 4.0),
                "c_Gamma_over_c": tuple(gp["c_Gamma_over_c_grid"]),
            },
            "boundaries": [gp["outer_boundary"], "initial Gamma=Gamma_target and D_t Gamma=0"],
            "limits": ["static source profiles cannot supply temporal memory"],
            "status": {
                "SPARC": "SOURCE_BLOCKED_NO_SOURCE_HISTORY",
                "XCOP_SPHERICAL": "SOURCE_BLOCKED_NO_SOURCE_HISTORY",
            },
            "lane": "ADJACENT",
            "prediction": "GP01-TELEGRAPH-BLOCKED",
        },
        {
            "stable_id": "GP01-ACTION-PLACEHOLDER",
            "identity": "FORMULA_VARIANT",
            "kind": "ACTION_PLACEHOLDER",
            "executable": False,
            "expressions": [gp["action_placeholder"]],
            "source": "incomplete preferred-frame action seed with no closed energy ledger",
            "allowed": ["synthetic source only after a complete action is supplied"],
            "mode": "TEMPORAL_MEMORY",
            "grid": {"n": GP01_GRID["n"]},
            "boundaries": ["incomplete action and causal source completion remain quarantined"],
            "limits": ["placeholder cannot enter a scored slot"],
            "status": {"SPARC": "INCOMPLETE_QUARANTINE", "XCOP_SPHERICAL": "INCOMPLETE_QUARANTINE"},
            "lane": "ADJACENT",
            "prediction": "GP01-ACTION-QUARANTINE",
        },
    ]
    for spec in gp_specs:
        stable_id = str(spec["stable_id"])
        card = _base_mechanism_card(
            stable_id=stable_id,
            identity_class=str(spec["identity"]),
            ontology="GAIN-PERSISTENCE-01",
            scientific_status="H_HYPOTHESIS",
            kind=str(spec["kind"]),
            executable=bool(spec["executable"]),
            exact_expressions=spec["expressions"],
            source=str(spec["source"]),
            allowed_inputs=spec["allowed"],
            state_mode=str(spec["mode"]),
            parameter_rows=_parameter_rows(stable_id, spec["grid"]),
            code_sha256=code_sha,
            configuration_sha256=configuration_sha,
            program_fingerprint=content_sha256(
                {"stable_id": stable_id, "expressions": spec["expressions"], "grid": spec["grid"]}
            ),
            synthetic_fingerprint=content_sha256({"prediction_group": spec["prediction"]}),
            boundaries=spec["boundaries"],
            limiting_cases=spec["limits"],
        )
        cards.append(
            _typed_card_wrapper(
                card,
                source_status=spec["status"],
                lane_hint=str(spec["lane"]),
                prediction_group=str(spec["prediction"]),
                manifest_authority=True,
            )
        )

    _require(len(cards) == 133, "typed adapter card count changed")
    card_ids = [row["card"]["card_id"] for row in cards]
    _require(len(card_ids) == len(set(card_ids)), "duplicate typed adapter card")
    ordered = sorted(cards, key=lambda row: row["card"]["card_id"])
    gp01_rows = [row for row in ordered if row["card"]["stable_concept_id"].startswith("GP01-")]
    provisional_twell_rows = [row for row in ordered if row not in gp01_rows]
    _require(len(gp01_rows) == 7 and len(provisional_twell_rows) == 126, "card partition changed")
    control_comparators = [
        {
            "transformation_id": row["id"],
            "operator": row["operator"],
            "program_sha256": content_sha256(row),
            "candidate_card": False,
            "target_free": True,
            "lane_hint_if_manifest_uses_as_transformation": "RIVALS_CONTROLS",
        }
        for row in config["wrong_control_programs"]
    ]
    return {
        "card_count": len(ordered),
        "cards": ordered,
        "ordered_card_set_sha256": registry_content_sha256(
            [
                {"card_id": row["card"]["card_id"], "card_sha256": row["card_sha256"]}
                for row in ordered
            ]
        ),
        "ordered_formula_sha256_root": content_sha256([row["formula_sha256"] for row in ordered]),
        "ordered_equivalence_sha256_root": content_sha256(
            [row["equivalence_fingerprint_sha256"] for row in ordered]
        ),
        "admission_counts": {
            status: sum(row["registry_admission_status"] == status for row in ordered)
            for status in (
                "READY_FOR_THEORY_GATES",
                "SOURCE_BLOCKED",
                "KNOWN_REWRITE_NONINDEPENDENT",
                "QUARANTINED_REVISION_REQUIRED",
            )
        },
        "gp01_live_card_count": len(gp01_rows),
        "gp01_live_card_set_sha256": registry_content_sha256(
            [
                {"card_id": row["card"]["card_id"], "card_sha256": row["card_sha256"]}
                for row in gp01_rows
            ]
        ),
        "provisional_twell_adapter_card_count": len(provisional_twell_rows),
        "provisional_twell_cards_manifest_authority": False,
        "provisional_twell_required_rebind_root_sha256": _program_root(config),
        "control_comparators": control_comparators,
        "controls_are_candidate_cards": False,
        "lane_assignment_authority": False,
        "lane_hints_present": sorted({row["lane_hint"] for row in ordered}),
        "orthogonal_or_wildcard_fillers_invented": False,
    }


def _synthetic_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    sparc_radius = np.linspace(0.05, 1.0, 41) * (20.0 * KPC_M)
    sparc_x = sparc_radius / sparc_radius[-1]
    gas_acceleration = 3.0e-11 * sparc_x / (0.08 + sparc_x * sparc_x)
    stellar_acceleration = 6.0e-11 * sparc_x / (0.04 + sparc_x * sparc_x) ** 0.75
    sparc = compile_sparc_source_drivers(
        sparc_radius, gas_acceleration, stellar_acceleration, points=PRIMARY_POINTS
    )

    # Positive source nodes align with the 257-point adapter grid after its regular origin.
    xcop_radius = np.linspace(1.0 / 256.0, 1.0, 256) * 2.0e22
    xcop_x = xcop_radius / xcop_radius[-1]
    gas_density = 2.0e-23 * (1.0 - 0.45 * xcop_x * xcop_x)
    xcop = compile_xcop_spherical_source_drivers(
        xcop_radius, gas_density, None, points=PRIMARY_POINTS
    )
    return sparc, xcop


def _architecture_synthetic_report() -> dict[str, Any]:
    source_xi = np.linspace(0.0, 1.0, 65)
    source_u = 0.30 + 0.22 * np.sin(math.pi * source_xi) + 0.07 * np.cos(2.0 * math.pi * source_xi)
    source_g = 2.0e-11 * (0.2 + source_xi)
    cells: list[dict[str, Any]] = []
    for architecture_id in STATIC_ARCHITECTURES:
        for parameters in architecture_parameter_cells(architecture_id):
            result = compile_static_architecture(
                architecture_id, source_xi, source_u, source_g, parameters
            )
            cells.append(
                {
                    "architecture_id": architecture_id,
                    "parameters": parameters,
                    "factor_sha256": result["factor_sha256"],
                    "operator_residual_max_abs": result["primary"]["diagnostics"][
                        "operator_residual_max_abs"
                    ],
                    "boundary_residual_max_abs": result["primary"]["diagnostics"][
                        "boundary_residual_max_abs"
                    ],
                    "primary_vs_convergence_max_abs": result["convergence_max_abs"],
                    "gates": result["gates"],
                }
            )
    return {
        "cell_count": len(cells),
        "cells": cells,
        "cell_root_sha256": content_sha256(cells),
        "all_parameter_branches_exercised": len(cells)
        == sum(len(architecture_parameter_cells(name)) for name in STATIC_ARCHITECTURES),
        "all_gates_pass": all(all(row["gates"].values()) for row in cells),
        "max_operator_residual": max(row["operator_residual_max_abs"] for row in cells),
        "max_boundary_residual": max(row["boundary_residual_max_abs"] for row in cells),
        "max_primary_vs_convergence": max(row["primary_vs_convergence_max_abs"] for row in cells),
    }


def _compound_synthetic_report(sparc: Mapping[str, Any], xcop: Mapping[str, Any]) -> dict[str, Any]:
    architecture_by_compound = {
        "X01": "A02_CLOCK",
        "X05": "A01_LAPSE",
        "X10": "A11_DERIV_SCREEN",
        "X13": "A08_PERMITTIVITY",
        "X17": "A12_MASSIVE",
        "X18": "A13_MIXED_MODE",
    }
    rows = []
    for compound_id in COMPOUND_IDS:
        architecture = architecture_by_compound[compound_id]
        parameters = architecture_parameter_cells(architecture)[-1]
        result = compile_compound_static("XCOP_SPHERICAL", compound_id, xcop, parameters)
        rows.append(
            {
                "compound_id": compound_id,
                "architecture_id": architecture,
                "parameters": parameters,
                "driver_sha256": result["compound_driver_sha256"],
                "factor_sha256": result["factor_sha256"],
                "all_gates_pass": all(result["gates"].values()),
                "xcop_status": result["source_status"],
                "sparc_status": compound_source_status("SPARC", compound_id),
            }
        )
    try:
        compile_compound_static(
            "SPARC", "X01", sparc, architecture_parameter_cells("A02_CLOCK")[-1]
        )
    except StaticSourceBlockedError:
        sparc_x01_block_observed = True
    else:
        sparc_x01_block_observed = False
    _require(sparc_x01_block_observed, "SPARC X01 block was not enforced")
    return {
        "rows": rows,
        "compound_count": len(rows),
        "all_xcop_static_compounds_pass": all(row["all_gates_pass"] for row in rows),
        "sparc_X01_block_observed": sparc_x01_block_observed,
        "compound_prediction_root_sha256": content_sha256([row["factor_sha256"] for row in rows]),
    }


def _time_block_report() -> dict[str, Any]:
    xi = np.linspace(0.0, 1.0, 5)
    observed: dict[str, str] = {}
    for architecture_id in TIME_SOURCE_BLOCKS:
        try:
            apply_static_architecture(architecture_id, xi, xi, xi, {})
        except StaticSourceBlockedError as error:
            observed[architecture_id] = str(error)
        else:
            raise OpenGravityStaticRadialAdapterError(
                f"time architecture unlocked: {architecture_id}"
            )
    return {
        "blocked": observed,
        "blocked_count": len(observed),
        "all_static_real_time_branches_blocked": len(observed) == 4,
    }


def _prediction_equivalence_report() -> dict[str, Any]:
    xi = np.linspace(0.0, 1.0, PRIMARY_POINTS)
    u = 0.2 + 0.6 * xi
    g = 1.0e-11 * (1.0 + xi)
    exponential = {
        name: array_sha256(apply_static_architecture(name, u, g, xi, {"lambda": 0.25})["factor"])
        for name in ("A01_LAPSE", "A02_CLOCK", "A08_PERMITTIVITY")
    }
    _require(len(set(exponential.values())) == 1, "exact exponential degeneracy changed")
    linear = {
        "A03_CONFORMAL": array_sha256(
            apply_static_architecture("A03_CONFORMAL", u, g, xi, {"lambda": 0.25})["factor"]
        ),
        "A13_MIXED_MODE_THETA0": array_sha256(
            apply_static_architecture(
                "A13_MIXED_MODE",
                u,
                g,
                xi,
                {"lambda": 0.25, "theta": 0.0, "ell": 0.25},
            )["factor"]
        ),
    }
    _require(len(set(linear.values())) == 1, "exact linear degeneracy changed")
    y = np.logspace(-7.0, 7.0, 129)
    spherical_links = []
    for n in GP01_GRID["n"]:
        prediction = y * np.asarray(gp01_nu_n(y, int(n)), dtype=float)
        prediction_hash = array_sha256(prediction)
        spherical_links.append(
            {
                "n": n,
                "GP01-L_prediction_sha256": prediction_hash,
                "AQUAL_spherical_prediction_sha256": prediction_hash,
                "score_once": True,
            }
        )
    groups = [
        {
            "kind": "EXACT_PREDICTION_DEGENERACY",
            "members": list(exponential),
            "shared_prediction_sha256": next(iter(exponential.values())),
            "degeneracy_sha256": content_sha256(sorted(exponential)),
        },
        {
            "kind": "PARAMETER_BRANCH_EXACT_PREDICTION_DEGENERACY",
            "members": list(linear),
            "shared_prediction_sha256": next(iter(linear.values())),
            "degeneracy_sha256": content_sha256(sorted(linear)),
        },
    ]
    return {
        "groups": groups,
        "gp01_spherical_equivalence_links": spherical_links,
        "equivalence_root_sha256": content_sha256(
            {"groups": groups, "gp01_spherical_equivalence_links": spherical_links}
        ),
        "matching_synthetic_predictions_do_not_imply_formula_identity": True,
    }


def _wrong_control_synthetic_report(config: Mapping[str, Any]) -> dict[str, Any]:
    radius = np.linspace(0.0, 1.0, 17)
    factor = 1.0 + 0.1 * radius + 0.03 * radius * radius
    rows = []
    for control_id in ("IDENTITY", "RADIAL_FACTOR_REVERSAL"):
        transformed = apply_wrong_control(control_id, factor, radius)
        rows.append(
            {
                "control_id": control_id,
                "program_sha256": wrong_control_program_hashes(config)[control_id],
                "input_factor_sha256": array_sha256(factor),
                "output_factor_sha256": array_sha256(transformed),
                "positive": bool(np.all(transformed > 0.0)),
                "target_inputs": 0,
            }
        )
    _require(
        np.array_equal(apply_wrong_control("IDENTITY", factor, radius), factor), "identity failed"
    )
    _require(
        np.array_equal(apply_wrong_control("RADIAL_FACTOR_REVERSAL", factor, radius), factor[::-1]),
        "radial reversal failed",
    )
    return {
        "rows": rows,
        "control_count": 2,
        "target_inputs": 0,
        "environment_or_object_shuffles_in_adapter": 0,
        "control_root_sha256": content_sha256(rows),
    }


def build_synthetic_report(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    sparc, xcop = _synthetic_sources()
    architecture = _architecture_synthetic_report()
    compounds = _compound_synthetic_report(sparc, xcop)
    gp01 = enumerate_gp01_spherical_cells(xcop)
    local_rows = []
    for n in GP01_GRID["n"]:
        y = np.logspace(-10.0, 10.0, 257)
        acceleration = gp01_l_acceleration(y * 1.2e-10, n=int(n)) / 1.2e-10
        local_rows.append(
            {
                "n": n,
                "prediction_sha256": array_sha256(acceleration),
                "high_field_ratio_to_y": float(acceleration[-1] / y[-1]),
                "deep_field_ratio_to_sqrt_y": float(acceleration[0] / math.sqrt(y[0])),
            }
        )
    source_summary = {
        "SPARC": {
            "driver_ids": sorted(sparc["normalized"]),
            "dimensions": {name: list(value.shape) for name, value in sparc["normalized"].items()},
            "driver_root_sha256": content_sha256(
                {name: array_sha256(value) for name, value in sparc["normalized"].items()}
            ),
            "response_inputs": sparc["response_inputs"],
        },
        "XCOP_SPHERICAL": {
            "driver_ids": sorted(xcop["normalized"]),
            "dimensions": {name: list(value.shape) for name, value in xcop["normalized"].items()},
            "driver_root_sha256": content_sha256(
                {name: array_sha256(value) for name, value in xcop["normalized"].items()}
            ),
            "R_b_m": xcop["metadata"]["R_b_m"],
            "rho_reference_kg_m3": xcop["metadata"]["rho_reference_kg_m3"],
            "tidal_reference_s_minus_2": xcop["metadata"]["tidal_reference_s_minus_2"],
            "response_inputs": xcop["response_inputs"],
        },
    }
    return {
        "source_drivers": source_summary,
        "static_architectures": architecture,
        "static_compounds": compounds,
        "static_time_source_blocks": _time_block_report(),
        "gp01_local": {
            "rows": local_rows,
            "n_count": len(local_rows),
            "all_high_and_deep_limits": all(
                abs(row["high_field_ratio_to_y"] - 1.0) < 1.0e-5
                and abs(row["deep_field_ratio_to_sqrt_y"] - 1.0) < 1.0e-5
                for row in local_rows
            ),
        },
        "gp01_elliptic": {key: value for key, value in gp01.items() if key != "cells"},
        "gp01_elliptic_cell_ledger_sha256": content_sha256(gp01["cells"]),
        "prediction_equivalence_and_degeneracy": _prediction_equivalence_report(),
        "wrong_controls": _wrong_control_synthetic_report(config),
        "response_inputs": 0,
        "scores_computed": 0,
    }


def receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("content_sha256", None)
    return content_sha256(payload)


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    config = load_config(root)
    upstream = verify_committed_upstreams(root, config)
    blocked_predecessor = verify_blocked_receipt_preservation(root)
    synthetic = build_synthetic_report(config)
    typed_cards = typed_mechanism_card_catalog(root, config)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "adapter_id": config["adapter_id"],
        "status": "SYNTHETIC_ADAPTER_MECHANICS_COMPLETE_ZERO_RESPONSE_ACCESS",
        "decision": DECISION,
        "bindings": {
            "config_sha256": file_sha256(root / CONFIG_PATH),
            "module_sha256": file_sha256(root / MODULE_PATH),
            "test_sha256": file_sha256(root / TEST_PATH),
            "committed_upstreams": upstream,
            "audit_blocked_predecessor": blocked_predecessor,
        },
        "program_hashes": program_hash_report(config),
        "typed_mechanism_card_catalog": typed_cards,
        "twell_rebind": evaluate_twell_rebind_gate(config, None),
        "synthetic_verification": synthetic,
        "access_ledger": dict(config["access_contract"]["zero_access"]),
        "claim_boundary": dict(config["claim_boundary"]),
        "limitations": [
            "This verifies adapter mechanics on analytic synthetic source arrays only.",
            "The independently audit-blocked predecessor receipt is preserved byte-for-byte under work and cannot authorize any campaign use.",
            "No SPARC Vobs, X-COP pressure or temperature, response-bearing receipt, response row, score, likelihood, ranking, confirmation row, or independent row was opened.",
            "The X-COP D05 surface density is a labeled mean-enclosed spherical proxy, not a projected-data measurement.",
            "A15-A18 remain source-blocked on static real profiles; SPARC X01 remains source-blocked.",
            "TWELL repair outputs remain deferred and non-authoritative until exact final hashes and an independent operator-equivalence audit are supplied to the rebind gate.",
            "Environment and object shuffles remain campaign-level controls and are not implemented here.",
            "Prediction equality or synthetic degeneracy is not formula identity and confers no scientific authority.",
            "An independent audit is requested; this receipt does not self-declare a scientific PASS.",
        ],
    }
    _require(
        all(value == 0 for value in receipt["access_ledger"].values()), "access ledger changed"
    )
    receipt["content_sha256"] = receipt_content_sha256(receipt)
    return receipt


def validate_receipt_payload(root: Path, stored: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an already-loaded receipt value without granting another file-read surface."""

    _require(isinstance(stored, dict), "receipt payload must be a JSON object")
    _require(stored.get("content_sha256") == receipt_content_sha256(stored), "receipt seal failed")
    _require(stored == build_receipt(root), "receipt differs from deterministic rebuild")
    return dict(stored)


def validate_receipt(root: Path = Path("."), output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    repo = root.resolve(strict=True)
    expected = (repo / OUTPUT_PATH).resolve(strict=False)
    provided = output_path if output_path.is_absolute() else repo / output_path
    target = provided.resolve(strict=False)
    _require(target == expected, "receipt validation path is not the canonical frozen OUTPUT_PATH")
    stored = _read_json(target)
    return validate_receipt_payload(repo, stored)


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
        except (OSError, PermissionError):
            _require(os.name == "nt", "could not fsync receipt directory")
        else:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path = Path(".")) -> str:
    receipt = build_receipt(root)
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return _atomic_no_clobber(root / OUTPUT_PATH, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "write", "check"))
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
                "GP01_elliptic_cells": receipt["synthetic_verification"]["gp01_elliptic"][
                    "cell_count"
                ],
                "response_inputs": receipt["synthetic_verification"]["response_inputs"],
                "scientific_scores_computed": receipt["access_ledger"][
                    "scientific_scores_computed"
                ],
                "twell_rebind_status": receipt["twell_rebind"]["status"],
                "publication": publication,
                "content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
