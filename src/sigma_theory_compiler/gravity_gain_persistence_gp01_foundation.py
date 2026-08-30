"""No-response GP01 formula, transport, PDE-contract, and synthetic preflight.

This module deliberately does not know how to load an observational table.  It compiles
source-only formula contracts and runs deterministic analytic/dimensionless synthetic checks.
The incomplete action and causal source sector remain fail-closed quarantines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/gravity_gain_persistence_gp01_foundation_v1.json")
OUTPUT_PATH = Path("runs/gravity/theory/gain-persistence-gp01-foundation-v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_gain_persistence_gp01_foundation.py")
TEST_PATH = Path("tests/test_gravity_gain_persistence_gp01_foundation.py")
CONFIG_SCHEMA = "invariant-gravity-gain-persistence-gp01-foundation-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-gain-persistence-gp01-foundation-receipt-1.0"
DECISION = "GP01_FOUNDATION_PASS_SYNTHETIC_ONLY_ACTION_AND_CAUSAL_COMPLETION_QUARANTINED"
EXPECTED_CONFIG_CONTENT_SHA256 = "6a4e2acb1865e0e7837f07c55ad8908546cc1eaebe9a7193966d459199a1bcb5"
EXPECTED_CONFIG_SECTION_SHA256 = {
    "scope": "2fc35bffb8389c2ab9e64e1bda717c844cc03488d234157a6b43794e43378cfb",
    "source_boundary": "29244d8ba8ed094a6b06f5d062f99bf276f0920bb2b79ca6cd5cd26dfac332f3",
    "variants": "c928b9450cac5ba736d2ede0f6fae72e90a012998891c1da22d089fd1f6e49ac",
    "equations": "c29db59f8f7cad6047d60d116babe90e23f9880235897f4dc4772855f2df4305",
    "dimensions": "eb083b7cd04506e8c496428304ae4089570e65cab9f0ccd12912f5e79cdf3e05",
    "parameters": "6a13dfae2281b90c9df49b77f6d33564d7a0f165281db12d267c1fd36934dffd",
    "boundaries_and_initial_data": "12c6d87a99de978e56f890abde2ced195c0229bdc0e02f35f1db92680d2095c7",
    "theory_filters": "b47afd82ab1647bcee53924b6affe9dc7c100043910f13458584fedf1149ba5e",
    "synthetic_contract": "d88cdc69d41401c979520dc2cd7a979e5a27bfabf51d7e9ec8d19559aa9fd6ce",
    "closures": "38435e132cc1db0e8118610ebe21ae0228c2879e7663bfd9dc32c5022eeb6710",
    "action_placeholder": "7739f2ed9ea759f71865ad088170b84e6cdad6726ae4835c15eb49dcb424b482",
    "claim_boundary": "4d82b4f598af7ca6f869c7e3dfcaf5920f788bcf85d3d7b853dfe8c5447417d2",
    "zero_access": "d65dd314a39ca29d19f539f9acc8c8a25bb123bd9e6dc62dbfd370de7deda733",
}

VARIANT_IDS = (
    "GP01-L",
    "GP01-AQUAL",
    "GP01-T1",
    "GP01-T2",
    "GP01-ELLIPTIC",
    "GP01-TELEGRAPH",
    "GP01-ACTION_PLACEHOLDER",
)
FILTER_IDS = tuple(
    f"F{index:02d}_{name}"
    for index, name in enumerate(
        (
            "DIMENSIONS",
            "HIGH_AND_DEEP_LIMITS",
            "SPHERICAL_BTF_AND_TRANSITION",
            "EXACT_AQUAL_MAPPING",
            "GENERAL_3D_CURL",
            "PATH_LOCAL_REDUCTION",
            "ENVIRONMENTAL_CLOSED_PATH",
            "FIELD_NULL_AND_SEPARATRIX",
            "CONDITIONAL_BOUNDED_ELLIPTICITY",
            "TELEGRAPH_NECESSARY_SPEED",
            "CAUSAL_SOURCE_COMPLETION",
            "ENERGY_AND_ACTION",
            "LIGHT_CAPTURE_CLOSURES",
        ),
        start=1,
    )
)
FIXTURE_IDS = (
    "SYN-GP01-SPHERE",
    "SYN-GP01-DISK",
    "SYN-GP01-MULTISOURCE",
    "SYN-GP01-SADDLE",
    "SYN-GP01-VOID",
)
ZERO_ACCESS_KEYS = (
    "observational_files_opened",
    "predictor_rows_opened",
    "response_rows_opened",
    "confirmation_rows_opened",
    "independent_rows_opened",
    "lensing_rows_opened",
    "formula_scores_computed",
    "likelihood_calls",
    "network_calls",
    "model_calls",
    "paid_calls",
    "gpu_calls",
)
LIMITATIONS = (
    "This receipt establishes deterministic formula and synthetic-preflight behavior only.",
    "GP01-L is a radial algebraic control; its nonspherical curl prevents promotion as a complete conservative field theory.",
    "The AQUAL entry is a known-family comparator with equivalence to GP01-L only in spherical or valid one-dimensional curl-free symmetry.",
    "T1 and T2 remain one-dimensional anchored field-line phenomenology and refuse nulls, separatrices, multiple or missing anchors, closed lines, and undeclared exits.",
    "The elliptic coefficient bound is conditional on a bounded solution; only a declared one-dimensional M-matrix control was solved, and telegraph overshoot is not bounded.",
    "A subluminal preferred-slice telegraph speed does not cure the instantaneous baryonic Poisson target.",
    "The action target is high-field singular in every declared n cell; the placeholder supplies neither damping nor a closed energy ledger.",
    "No matter-light, lensing, redshift, capture, gravitational-wave, quantum, Solar, or cosmological claim is earned.",
    "No observational file, predictor row, response row, score, likelihood, network, model, paid service, or GPU was used.",
)


class GravityGainPersistenceFoundationError(RuntimeError):
    """Raised when the frozen GP01 foundation contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GravityGainPersistenceFoundationError(message)


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityGainPersistenceFoundationError(
            f"{label} keys changed: expected {sorted(keys)}, got {sorted(value)}"
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GravityGainPersistenceFoundationError(f"noncanonical JSON value: {exc}") from exc
    return serialized.encode("utf-8") + b"\n"


def _content_sha256(value: Any) -> str:
    """Hash canonical JSON including finite floats used by numerical receipts."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GravityGainPersistenceFoundationError(
            f"cannot read JSON {path.as_posix()}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GravityGainPersistenceFoundationError(f"JSON object required: {path.as_posix()}")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen semantics without consulting any response-bearing artifact."""

    _require_exact_keys(
        config,
        {
            "schema_version",
            "foundation_id",
            "version",
            "status",
            "purpose",
            "scope",
            "source_boundary",
            "variants",
            "equations",
            "dimensions",
            "parameters",
            "boundaries_and_initial_data",
            "theory_filters",
            "synthetic_contract",
            "closures",
            "action_placeholder",
            "claim_boundary",
            "zero_access",
            "output_path",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        _content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "immutable config content hash changed",
    )
    observed_section_hashes = {
        section: _content_sha256(config[section]) for section in EXPECTED_CONFIG_SECTION_SHA256
    }
    _require(
        observed_section_hashes == EXPECTED_CONFIG_SECTION_SHA256,
        "immutable config section hash changed",
    )
    _require(config["foundation_id"] == "GAIN-PERSISTENCE-01-FOUNDATION-v1", "ID changed")
    _require(config["version"] == "1.0.0", "semantic version changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")

    scope = config["scope"]
    _require(scope["synthetic_execution_only"] is True, "synthetic-only scope changed")
    for key in ("campaign_scoring", "observational_execution", "historical_novelty_claim"):
        _require(scope[key] is False, f"forbidden scope enabled: {key}")

    source = config["source_boundary"]
    _require(source["real_data_paths"] == [], "real data path was admitted")
    _require(
        source["gain_is_constructed_only_from_baryonic_cause_fields"] is True,
        "source-only gain contract changed",
    )
    forbidden = set(source["forbidden_inputs"])
    _require(
        {
            "rotation_or_velocity_responses",
            "cluster_pressure_or_temperature",
            "lensing_or_inferred_total_mass",
            "redshift_residuals",
            "confirmation_holdout_or_independent_rows",
        }
        <= forbidden,
        "response exclusion list weakened",
    )

    variants = config["variants"]
    _require(tuple(item["variant_id"] for item in variants) == VARIANT_IDS, "variants changed")
    _require(
        all(item["response_scoring_eligible"] is False for item in variants),
        "response scoring was enabled",
    )
    local = variants[0]
    _require(local["general_three_dimensional_theory"] is False, "curl ceiling erased")
    aqual = variants[1]
    _require(aqual["identity"] == "KNOWN_FAMILY_COMPARATOR", "AQUAL identity changed")
    _require(aqual["general_3d_equivalence"] is False, "AQUAL equivalence overclaimed")
    _require(
        aqual["equivalent_to_GP01_L_only_in"] == "spherical_or_valid_1D_curl_free_symmetry",
        "AQUAL symmetry ceiling changed",
    )
    action_variant = variants[-1]
    _require(
        action_variant["status"] == "ACTION_PLACEHOLDER_QUARANTINED",
        "action variant escaped quarantine",
    )

    equations = config["equations"]
    _require("y/x(y) = 1/nu_n(y)" in equations["aqual_parametric_mapping"], "AQUAL map changed")
    _require(
        "closed integral W d ln(f) = 0" in equations["path_independence_gate"],
        "path gate changed",
    )
    _require("exp(-Gamma)" in equations["coupled_potential"], "elliptic coefficient changed")
    _require(
        "c_Gamma = L_g/tau_g <= c" == equations["necessary_characteristic_speed"],
        "speed gate changed",
    )

    dimensions = config["dimensions"]
    _require(dimensions["elliptic_equation_each_side"] == "s^-2", "elliptic units changed")
    _require(
        dimensions["telegraph_equation_each_term"] == "dimensionless",
        "telegraph units changed",
    )

    parameters = config["parameters"]
    _require(parameters["beta_principal"] == 0.5, "principal beta changed")
    _require(parameters["a_star_m_s2"] == 1.2e-10, "reference acceleration changed")
    _require(parameters["n_grid"] == [1, 2, 4], "smoothness grid changed")
    _require(parameters["synthetic_R_b_m"] > 0.0, "synthetic R_b must be positive")
    _require(parameters["L_g_zero_control"] == 0.0, "L_g=0 control changed")
    _require(parameters["field_null_tolerance"] > 0.0, "field-null tolerance changed")
    _require(parameters["object_specific_gravity_parameters"] is False, "object tuning enabled")
    _require(
        all(0.0 < value <= 1.0 for value in parameters["c_Gamma_over_c_grid"]),
        "superluminal characteristic cell admitted",
    )

    boundaries = config["boundaries_and_initial_data"]
    _require(
        boundaries["GP01-T1_and_T2"]["field_null"] == "TRANSPORT_QUARANTINED_AT_NULL",
        "field-null quarantine changed",
    )
    _require(
        boundaries["GP01-TELEGRAPH"]["source_completion"].startswith("BLOCKED_"),
        "instantaneous-source causal blocker removed",
    )

    _require(tuple(config["theory_filters"]) == FILTER_IDS, "theory filters changed")
    fixtures = config["synthetic_contract"]["fixtures"]
    _require(tuple(item["fixture_id"] for item in fixtures) == FIXTURE_IDS, "fixtures changed")

    closures = config["closures"]
    _require(closures["light"] == "L0_NO_LIGHT_CLAIM", "light claim unlocked")
    _require(closures["capture"] == "C0_ISOLATED_CONSERVATIVE", "capture claim unlocked")

    action = config["action_placeholder"]
    _require(action["label"] == "ACTION_PLACEHOLDER", "action label changed")
    _require(action["executable"] is False, "placeholder became executable")
    _require(action["response_scoring_eligible"] is False, "placeholder became scoreable")
    _require(len(action["required_missing_definitions"]) == 7, "action blockers changed")

    claims = config["claim_boundary"]
    for key in (
        "causal_theory_completed",
        "healthy_action_completed",
        "matter_light_unified",
        "capture_mechanism_established",
        "observational_signal_measured",
        "historical_novelty_established",
        "response_scoring_unlocked",
        "confirmation_opened",
    ):
        _require(claims[key] is False, f"claim boundary overstates {key}")

    zero = config["zero_access"]
    _require_exact_keys(zero, set(ZERO_ACCESS_KEYS), "zero-access ledger")
    _require(all(zero[key] == 0 for key in ZERO_ACCESS_KEYS), "nonzero access declared")


def load_config(root: Path = Path("."), config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    _require(config_path == CONFIG_PATH, "config path is not the frozen path")
    path = root / config_path
    _require(path.is_file(), f"config is missing: {path.as_posix()}")
    config = _read_json(path)
    validate_config(config)
    return config


def nu_n(y: float | np.ndarray, n: int) -> float | np.ndarray:
    """Evaluate the exact GP01-L interpolation without fitting anything."""

    array = np.asarray(y, dtype=float)
    _require(isinstance(n, int) and not isinstance(n, bool) and n > 0, "n must be positive")
    _require(bool(np.all(np.isfinite(array) & (array > 0.0))), "y must be finite and positive")
    result = np.exp(np.logaddexp(0.0, -n * np.log(array)) / (2.0 * n))
    return float(result) if result.ndim == 0 else result


def aqual_parametric_mapping(
    y: float | np.ndarray, n: int
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return x(y), mu(x(y)) for the exact AQUAL comparator map."""

    array = np.asarray(y, dtype=float)
    nu = np.asarray(nu_n(array, n), dtype=float)
    x = array * nu
    mu = array / x
    if array.ndim == 0:
        return float(x), float(mu)
    return x, mu


def local_algebraic_field(g_b: np.ndarray, *, a_star: float, n: int) -> np.ndarray:
    """Apply GP01-L to a final vector axis; this is only a phenomenology control."""

    field = np.asarray(g_b, dtype=float)
    _require(field.ndim >= 1 and field.shape[-1] in (2, 3), "vector field required")
    _require(a_star > 0.0 and math.isfinite(a_star), "a_star must be finite and positive")
    magnitude = np.linalg.norm(field, axis=-1)
    result = np.zeros_like(field)
    nonzero = magnitude > 0.0
    if np.any(nonzero):
        factor = np.asarray(nu_n(magnitude[nonzero] / a_star, n), dtype=float)
        result[nonzero] = field[nonzero] * factor[..., None]
    return result


def bounded_gamma_target(
    f: float | np.ndarray,
    *,
    w: float | np.ndarray,
    a_star: float,
    n: int,
    gamma_max: float,
) -> float | np.ndarray:
    """Evaluate the bounded target, including the declared continuous exact-null limit."""

    force = np.asarray(f, dtype=float)
    weight = np.asarray(w, dtype=float)
    _require(np.all(np.isfinite(force) & (force >= 0.0)), "f must be finite and nonnegative")
    _require(np.all(np.isfinite(weight) & (weight >= 0.0) & (weight <= 1.0)), "bad W")
    _require(
        math.isfinite(a_star)
        and a_star > 0.0
        and isinstance(n, int)
        and n > 0
        and math.isfinite(gamma_max)
        and gamma_max > 0.0,
        "bad target parameter",
    )
    force, weight = np.broadcast_arrays(force, weight)
    target = np.empty_like(force)
    nonzero = force > 0.0
    log_ratio_power = n * (math.log(a_star) - np.log(force[nonzero]))
    log_one_plus = np.logaddexp(0.0, log_ratio_power)
    target[nonzero] = weight[nonzero] * gamma_max * np.tanh(log_one_plus / (2.0 * n * gamma_max))
    target[~nonzero] = weight[~nonzero] * gamma_max
    return float(target) if target.ndim == 0 else target


def _transport_inputs(
    s: Sequence[float], values: Sequence[float], label: str
) -> tuple[np.ndarray, np.ndarray]:
    arc = np.asarray(s, dtype=float)
    data = np.asarray(values, dtype=float)
    _require(arc.ndim == data.ndim == 1 and arc.size == data.size, f"bad {label} arrays")
    _require(arc.size >= 2 and np.all(np.isfinite(arc)), f"bad {label} arc")
    _require(np.all(np.diff(arc) > 0.0), f"{label} arc must increase")
    _require(np.all(np.isfinite(data)), f"bad {label} values")
    return arc, data


def _validate_transport_line(
    field_magnitude: Sequence[float],
    *,
    anchored: bool,
    anchor_count: int,
    crosses_separatrix: bool,
    closed_field_line: bool,
    domain_exit: str,
    field_null_tolerance: float,
    label: str,
) -> np.ndarray:
    """Apply the frozen multiple-source field-line validity contract."""

    force = np.asarray(field_magnitude, dtype=float)
    _require(force.ndim == 1 and force.size >= 2, f"{label} field magnitude is invalid")
    _require(np.all(np.isfinite(force) & (force >= 0.0)), f"{label} field is nonfinite")
    _require(
        math.isfinite(field_null_tolerance) and field_null_tolerance > 0.0, "bad null tolerance"
    )
    _require(anchored, f"{label} requires a source-defined anchor")
    _require(
        isinstance(anchor_count, int) and not isinstance(anchor_count, bool) and anchor_count == 1,
        f"{label} multiple or missing anchors are quarantined",
    )
    _require(not crosses_separatrix, f"{label} separatrix continuation is quarantined")
    _require(not closed_field_line, f"{label} closed field lines are quarantined")
    _require(
        domain_exit in {"NONE", "DECLARED_OUTER_BOUNDARY"},
        f"{label} domain exit is not declared",
    )
    _require(np.all(force > field_null_tolerance), f"{label} cannot cross a field null")
    return force


def transport_t1(
    s: Sequence[float],
    f: Sequence[float],
    w: Sequence[float],
    *,
    beta: float,
    l_reset: float,
    gamma_anchor: float,
    anchored: bool = True,
    anchor_count: int = 1,
    crosses_separatrix: bool = False,
    closed_field_line: bool = False,
    domain_exit: str = "NONE",
    field_null_tolerance: float = 1e-12,
) -> np.ndarray:
    """Exact segment propagation for linear ln(f) and midpoint-constant W.

    This one-dimensional integrator is valid only on an already declared, anchored field line.
    It intentionally refuses to invent a continuation through a null or an unanchored line.
    """

    arc, force = _transport_inputs(s, f, "T1")
    _, weight = _transport_inputs(s, w, "T1")
    force = _validate_transport_line(
        force,
        anchored=anchored,
        anchor_count=anchor_count,
        crosses_separatrix=crosses_separatrix,
        closed_field_line=closed_field_line,
        domain_exit=domain_exit,
        field_null_tolerance=field_null_tolerance,
        label="T1",
    )
    _require(np.all((weight >= 0.0) & (weight <= 1.0)), "T1 W outside [0,1]")
    _require(
        math.isfinite(beta)
        and beta >= 0.0
        and math.isfinite(l_reset)
        and l_reset > 0.0
        and math.isfinite(gamma_anchor)
        and gamma_anchor >= 0.0,
        "bad T1 parameter",
    )
    gamma = np.empty_like(arc)
    gamma[0] = gamma_anchor
    log_force = np.log(force)
    for index, ds in enumerate(np.diff(arc)):
        midpoint_w = 0.5 * (weight[index] + weight[index + 1])
        source = -beta * midpoint_w * (log_force[index + 1] - log_force[index]) / ds
        damping = (1.0 - midpoint_w) / l_reset
        if damping == 0.0:
            gamma[index + 1] = gamma[index] + source * ds
        else:
            decay = math.exp(-damping * ds)
            gamma[index + 1] = gamma[index] * decay + source * (1.0 - decay) / damping
    return gamma


def transport_t2(
    s: Sequence[float],
    gamma_target: Sequence[float],
    field_magnitude: Sequence[float],
    *,
    l_g: float,
    gamma_anchor: float,
    anchored: bool = True,
    anchor_count: int = 1,
    crosses_separatrix: bool = False,
    closed_field_line: bool = False,
    domain_exit: str = "NONE",
    field_null_tolerance: float = 1e-12,
) -> np.ndarray:
    """Exact segment propagation for a midpoint-constant target on an anchored line."""

    arc, target = _transport_inputs(s, gamma_target, "T2")
    _require(np.all(target >= 0.0), "T2 target must be nonnegative")
    force = _validate_transport_line(
        field_magnitude,
        anchored=anchored,
        anchor_count=anchor_count,
        crosses_separatrix=crosses_separatrix,
        closed_field_line=closed_field_line,
        domain_exit=domain_exit,
        field_null_tolerance=field_null_tolerance,
        label="T2",
    )
    _require(force.size == arc.size, "T2 field magnitude length changed")
    _require(
        math.isfinite(l_g) and l_g > 0.0 and math.isfinite(gamma_anchor) and gamma_anchor >= 0.0,
        "T2 L_g and anchor must be finite and positive where required",
    )
    gamma = np.empty_like(arc)
    gamma[0] = gamma_anchor
    for index, ds in enumerate(np.diff(arc)):
        midpoint_target = 0.5 * (target[index] + target[index + 1])
        decay = math.exp(-ds / l_g)
        gamma[index + 1] = midpoint_target + (gamma[index] - midpoint_target) * decay
    return gamma


def closed_path_integral(log_f: Sequence[float], w: Sequence[float]) -> float:
    """Trapezoidal discrete evaluation of the declared closed W d ln(f) gate."""

    force_log = np.asarray(log_f, dtype=float)
    weight = np.asarray(w, dtype=float)
    _require(force_log.ndim == weight.ndim == 1, "closed path arrays must be one-dimensional")
    _require(force_log.size == weight.size and force_log.size >= 3, "bad closed path arrays")
    _require(np.all(np.isfinite(force_log)) and np.all(np.isfinite(weight)), "nonfinite path")
    _require(abs(force_log[0] - force_log[-1]) <= 1e-12, "path is not closed in f")
    return float(np.sum(0.5 * (weight[:-1] + weight[1:]) * np.diff(force_log)))


def ellipticity_bounds(gamma_max: float) -> tuple[float, float]:
    """Uniform coefficient bounds for exp(-Gamma) when 0 <= Gamma <= Gamma_max."""

    _require(gamma_max > 0.0 and math.isfinite(gamma_max), "Gamma_max must be positive")
    return math.exp(-gamma_max), 1.0


def telegraph_characteristic_speed(l_g: float, tau_g: float) -> float:
    """Return the preferred-slice principal speed; this is necessary, not sufficient."""

    _require(
        math.isfinite(l_g) and l_g > 0.0 and math.isfinite(tau_g) and tau_g > 0.0,
        "telegraph scales must be finite and positive",
    )
    return l_g / tau_g


def trace_free_tidal_norm(jacobian: np.ndarray) -> np.ndarray:
    """Return the Frobenius norm of the trace-free 2D synthetic source Jacobian."""

    tensor = np.asarray(jacobian, dtype=float)
    _require(tensor.shape[-2:] == (2, 2), "2D source Jacobian required")
    _require(np.all(np.isfinite(tensor)), "tidal tensor is nonfinite")
    trace = np.trace(tensor, axis1=-2, axis2=-1)
    identity = np.eye(2)
    trace_free = tensor - 0.5 * trace[..., None, None] * identity
    return np.sqrt(np.sum(trace_free * trace_free, axis=(-2, -1)))


def environment_gate(
    rho_b: float | np.ndarray,
    tidal: float | np.ndarray,
    *,
    rho_star: float,
    tidal_star: float,
    q: int,
    r: int,
) -> float | np.ndarray:
    """Evaluate the exact bounded synthetic environment gate."""

    density = np.asarray(rho_b, dtype=float)
    tide = np.asarray(tidal, dtype=float)
    _require(np.all(np.isfinite(density) & (density >= 0.0)), "density is invalid")
    _require(np.all(np.isfinite(tide) & (tide >= 0.0)), "tidal magnitude is invalid")
    _require(math.isfinite(rho_star) and rho_star > 0.0, "rho_star must be positive")
    _require(math.isfinite(tidal_star) and tidal_star > 0.0, "T_star must be positive")
    _require(
        isinstance(q, int)
        and not isinstance(q, bool)
        and q > 0
        and isinstance(r, int)
        and not isinstance(r, bool)
        and r > 0,
        "bad gate exponent",
    )
    density, tide = np.broadcast_arrays(density, tide)
    result = 1.0 / (1.0 + (density / rho_star) ** q + (tide / tidal_star) ** r)
    return float(result) if result.ndim == 0 else result


def quasi_static_gain_1d(
    target: Sequence[float], *, l_over_domain: float, spacing: float
) -> np.ndarray:
    """Solve the frozen 1D Dirichlet M-matrix control for (1-L^2 D^2)Gamma=target."""

    source = np.asarray(target, dtype=float)
    _require(source.ndim == 1 and source.size >= 3, "quasi-static target is invalid")
    _require(np.all(np.isfinite(source) & (source >= 0.0)), "quasi-static target is invalid")
    _require(math.isfinite(l_over_domain) and l_over_domain >= 0.0, "bad L_g control")
    _require(math.isfinite(spacing) and spacing > 0.0, "bad quasi-static spacing")
    if l_over_domain == 0.0:
        return source.copy()
    interior = source.size - 2
    coefficient = (l_over_domain / spacing) ** 2
    matrix = np.diag(np.full(interior, 1.0 + 2.0 * coefficient))
    if interior > 1:
        off_diagonal = np.full(interior - 1, -coefficient)
        matrix += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    rhs = source[1:-1].copy()
    rhs[0] += coefficient * source[0]
    rhs[-1] += coefficient * source[-1]
    solution = source.copy()
    solution[1:-1] = np.linalg.solve(matrix, rhs)
    return solution


def action_target(y: float | np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gamma_n and the required dimensionless V'_n along the target branch."""

    acceleration = np.asarray(y, dtype=float)
    gain = np.asarray(nu_n(acceleration, n), dtype=float)
    return np.log(gain), acceleration * acceleration * gain


def action_regularity_class(n: int) -> str:
    """Classify the high-field Gamma->0+ singularity implied by the target branch."""

    _require(isinstance(n, int) and not isinstance(n, bool) and n > 0, "n must be positive")
    if n < 2:
        return "V_POWER_DIVERGENCE_AND_VPRIME_DIVERGENCE"
    if n == 2:
        return "V_LOG_DIVERGENCE_AND_VPRIME_DIVERGENCE"
    return "V_FINITE_BUT_VPRIME_DIVERGENT_NONANALYTIC"


def _source_positions(fixture: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if fixture["kind"] == "sparse_center_inside_baryonic_ring":
        count = int(fixture["ring_source_count"])
        angles = np.arange(count, dtype=float) * (2.0 * math.pi / count)
        radius = float(fixture["ring_radius"])
        positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        masses = np.full(count, float(fixture["total_mass"]) / count)
        return positions, masses
    return np.asarray(fixture["positions"], dtype=float), np.asarray(fixture["masses"], dtype=float)


def softened_point_field(
    x: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    softening: float,
) -> np.ndarray:
    """Construct a dimensionless, source-only softened Newtonian field."""

    field, _ = softened_point_field_and_jacobian(
        x,
        y,
        positions,
        masses,
        softening=softening,
    )
    return field


def softened_point_field_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the analytic field and its symmetric spatial Jacobian."""

    _require(x.shape == y.shape, "mesh shapes differ")
    _require(positions.ndim == 2 and positions.shape[1] == 2, "2D positions required")
    _require(masses.shape == (positions.shape[0],), "source mass shape changed")
    _require(np.all(masses > 0.0) and softening > 0.0, "bad synthetic source")
    gx = np.zeros_like(x, dtype=float)
    gy = np.zeros_like(y, dtype=float)
    jxx = np.zeros_like(x, dtype=float)
    jxy = np.zeros_like(x, dtype=float)
    jyx = np.zeros_like(x, dtype=float)
    jyy = np.zeros_like(x, dtype=float)
    for (source_x, source_y), mass in zip(positions, masses, strict=True):
        dx = x - source_x
        dy = y - source_y
        radius_squared = dx * dx + dy * dy + softening * softening
        inverse_three = radius_squared**-1.5
        inverse_five = radius_squared**-2.5
        gx -= mass * dx * inverse_three
        gy -= mass * dy * inverse_three
        jxx -= mass * (inverse_three - 3.0 * dx * dx * inverse_five)
        jxy += mass * (3.0 * dx * dy * inverse_five)
        jyx += mass * (3.0 * dy * dx * inverse_five)
        jyy -= mass * (inverse_three - 3.0 * dy * dy * inverse_five)
    field = np.stack((gx, gy), axis=-1)
    jacobian = np.stack(
        (
            np.stack((jxx, jxy), axis=-1),
            np.stack((jyx, jyy), axis=-1),
        ),
        axis=-2,
    )
    return field, jacobian


def _curl_statistics(
    field: np.ndarray,
    jacobian: np.ndarray,
    *,
    a_star: float,
    n: int,
) -> dict[str, float]:
    """Use the analytic source Hessian to isolate curl induced by algebraic gain."""

    gx, gy = field[..., 0], field[..., 1]
    magnitude = np.linalg.norm(field, axis=-1)
    mask = magnitude > 1e-10
    _require(bool(np.any(mask)), "curl diagnostic mask is empty")
    grad_f_x = np.zeros_like(magnitude)
    grad_f_y = np.zeros_like(magnitude)
    grad_f_x[mask] = (
        gx[mask] * jacobian[..., 0, 0][mask] + gy[mask] * jacobian[..., 1, 0][mask]
    ) / magnitude[mask]
    grad_f_y[mask] = (
        gx[mask] * jacobian[..., 0, 1][mask] + gy[mask] * jacobian[..., 1, 1][mask]
    ) / magnitude[mask]
    nu = np.ones_like(magnitude)
    nu[mask] = np.asarray(nu_n(magnitude[mask] / a_star, n))
    low_acceleration_fraction = np.zeros_like(magnitude)
    low_acceleration_fraction[mask] = 1.0 / (1.0 + (magnitude[mask] / a_star) ** n)
    derivative = np.zeros_like(magnitude)
    derivative[mask] = -nu[mask] * low_acceleration_fraction[mask] / (2.0 * magnitude[mask])
    dnu_dx = derivative * grad_f_x
    dnu_dy = derivative * grad_f_y
    induced_curl = dnu_dx * gy - dnu_dy * gx
    baryonic_curl = jacobian[..., 1, 0] - jacobian[..., 0, 1]
    return {
        "baryonic_curl_analytic_max_abs": float(np.max(np.abs(baryonic_curl))),
        "induced_curl_analytic_rms": float(np.sqrt(np.mean(induced_curl[mask] ** 2))),
        "induced_curl_analytic_max_abs": float(np.max(np.abs(induced_curl[mask]))),
    }


def _field_and_jacobian_at_origin(
    positions: np.ndarray, masses: np.ndarray, *, softening: float
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.zeros((1, 1), dtype=float)
    field, jacobian = softened_point_field_and_jacobian(
        origin, origin, positions, masses, softening=softening
    )
    return field[0, 0], jacobian[0, 0]


def _computed_check(check_id: str, passed: bool, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed), "evidence": dict(evidence)}


def _finalize_fixture(
    fixture: Mapping[str, Any],
    payload: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    required = tuple(fixture["required_checks"])
    observed = tuple(check["check_id"] for check in checks)
    _require(observed == required, f"required checks were not computed for {fixture['fixture_id']}")
    _require(
        all(check["passed"] is True for check in checks),
        f"fixture gate failed: {fixture['fixture_id']}",
    )
    return {
        "fixture_id": fixture["fixture_id"],
        "source_kind": fixture["kind"],
        "real_rows": 0,
        **dict(payload),
        "required_checks": list(required),
        "checks": [dict(check) for check in checks],
        "all_required_checks_passed": True,
        "status": status,
    }


def _bisect_transition_root(mass: float, a_star: float, lower: float, upper: float) -> float:
    def residual(radius: float) -> float:
        return mass / (radius * radius) - a_star

    _require(residual(lower) > 0.0 and residual(upper) < 0.0, "transition root not bracketed")
    left, right = lower, upper
    for _ in range(100):
        midpoint = 0.5 * (left + right)
        if residual(midpoint) > 0.0:
            left = midpoint
        else:
            right = midpoint
    return 0.5 * (left + right)


def _sphere_fixture(config: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    synthetic = config["synthetic_contract"]
    grid = synthetic["grid"]
    a_star = 0.01
    mass = float(fixture["mass"])
    radii = np.geomspace(
        float(fixture["radial_min"]),
        float(fixture["radial_max"]),
        int(fixture["radial_points"]),
    )
    g_b = mass / radii**2
    analytic_transition = math.sqrt(mass / a_star)
    computed_transition = _bisect_transition_root(mass, a_star, float(radii[0]), float(radii[-1]))
    transition_error = abs(computed_transition / analytic_transition - 1.0)
    n_results = []
    for n in config["parameters"]["n_grid"]:
        g = np.asarray(nu_n(g_b / a_star, n)) * g_b
        deep_btf_ratio = float((radii[-1] * g[-1]) ** 2 / (mass * a_star))
        high_gain = float(nu_n(1e12, n))
        deep_ratio = float(nu_n(1e-12, n) * math.sqrt(1e-12))
        n_results.append(
            {
                "n": n,
                "high_field_nu_minus_one": high_gain - 1.0,
                "deep_field_scaled_ratio": deep_ratio,
                "deep_outer_btf_ratio": deep_btf_ratio,
                "deep_outer_btf_relative_error": abs(deep_btf_ratio - 1.0),
            }
        )

    coordinates = np.linspace(
        float(grid["minimum"]), float(grid["maximum"]), int(grid["points_per_axis"])
    )
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    radial_field, radial_jacobian = softened_point_field_and_jacobian(
        x,
        y,
        np.asarray([[0.0, 0.0]]),
        np.asarray([mass]),
        softening=float(grid["softening"]),
    )
    radial_curls = _curl_statistics(radial_field, radial_jacobian, a_star=a_star, n=2)
    local_limits_pass = all(
        abs(row["high_field_nu_minus_one"]) <= 1e-12
        and abs(row["deep_field_scaled_ratio"] - 1.0) <= 1e-6
        for row in n_results
    )
    btf_pass = all(row["deep_outer_btf_relative_error"] <= 0.011 for row in n_results)
    radial_curl_pass = radial_curls["induced_curl_analytic_max_abs"] <= 1e-14
    checks = [
        _computed_check("local_limits", local_limits_pass, {"n_results": n_results}),
        _computed_check(
            "transition_root_numerical",
            transition_error <= 1e-14,
            {
                "analytic_radius": analytic_transition,
                "bisection_radius": computed_transition,
                "relative_error": transition_error,
                "iterations": 100,
            },
        ),
        _computed_check(
            "baryonic_tully_fisher",
            btf_pass,
            {
                "maximum_relative_error": max(
                    row["deep_outer_btf_relative_error"] for row in n_results
                )
            },
        ),
        _computed_check("radial_curl_computed", radial_curl_pass, radial_curls),
    ]
    return _finalize_fixture(
        fixture,
        {
            "synthetic_units": synthetic["units"],
            "analytic_transition_radius": analytic_transition,
            "computed_transition_radius": computed_transition,
            "transition_relative_error": transition_error,
            "n_results": n_results,
            **radial_curls,
        },
        checks,
        status="PASS_COMPUTED_LOCAL_CONTROL",
    )


def _multisource_loop_evidence(
    config: Mapping[str, Any], positions: np.ndarray, masses: np.ndarray
) -> dict[str, Any]:
    contract = config["synthetic_contract"]["multisource_closed_loop"]
    count = int(contract["points_including_closure"])
    theta = np.linspace(0.0, 2.0 * math.pi, count)
    x = float(contract["center"][0]) + float(contract["x_radius"]) * np.cos(theta)
    y = float(contract["center"][1]) + float(contract["y_radius"]) * np.sin(theta)
    x[-1], y[-1] = x[0], y[0]
    field, jacobian = softened_point_field_and_jacobian(
        x,
        y,
        positions,
        masses,
        softening=float(config["synthetic_contract"]["grid"]["softening"]),
    )
    force = np.linalg.norm(field, axis=-1)
    _require(np.all(force > 0.0), "declared multisource loop crosses a field null")
    tidal = trace_free_tidal_norm(jacobian)
    positive_tidal = tidal[tidal > 0.0]
    _require(positive_tidal.size > 0, "declared loop has no tidal reference")
    tidal_reference = float(np.median(positive_tidal))
    weight = np.asarray(
        environment_gate(
            np.full(count, float(contract["rho_b_on_loop"])),
            tidal,
            rho_star=float(contract["rho_reference"]),
            tidal_star=tidal_reference,
            q=int(contract["q"]),
            r=int(contract["r"]),
        )
    )
    integral = closed_path_integral(np.log(force), weight)
    return {
        "loop_points_including_closure": count,
        "force_closure_error": float(abs(force[0] - force[-1])),
        "tidal_reference": tidal_reference,
        "weight_minimum": float(np.min(weight)),
        "weight_maximum": float(np.max(weight)),
        "closed_integral_W_dlnf": integral,
        "path_independence_passed": abs(integral) <= 1e-8,
    }


def _observe_transport_refusal(call: Any, expected_fragment: str) -> dict[str, Any]:
    try:
        call()
    except GravityGainPersistenceFoundationError as exc:
        message = str(exc)
        return {
            "refusal_observed": expected_fragment in message,
            "exception_type": type(exc).__name__,
            "message": message,
        }
    return {"refusal_observed": False, "exception_type": None, "message": None}


def _spatial_fixture(config: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    grid = config["synthetic_contract"]["grid"]
    coordinates = np.linspace(
        float(grid["minimum"]), float(grid["maximum"]), int(grid["points_per_axis"])
    )
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    positions, masses = _source_positions(fixture)
    field, jacobian = softened_point_field_and_jacobian(
        x, y, positions, masses, softening=float(grid["softening"])
    )
    curls = _curl_statistics(field, jacobian, a_star=0.01, n=2)
    center_vector, center_jacobian = _field_and_jacobian_at_origin(
        positions, masses, softening=float(grid["softening"])
    )
    center_field = float(np.linalg.norm(center_vector))
    fixture_id = fixture["fixture_id"]
    common = {
        "source_count": len(masses),
        "source_mass_sum": float(np.sum(masses)),
        "origin_field_vector": center_vector.tolist(),
        "origin_field_magnitude": center_field,
        **curls,
    }
    if fixture_id == "SYN-GP01-DISK":
        checks = [
            _computed_check(
                "baryonic_field_curl_analytic_control",
                curls["baryonic_curl_analytic_max_abs"] <= 1e-14,
                {"maximum_absolute_curl": curls["baryonic_curl_analytic_max_abs"]},
            ),
            _computed_check(
                "algebraic_gain_induced_curl_computed_counterexample",
                curls["induced_curl_analytic_rms"] > 1e-6,
                {"rms_curl": curls["induced_curl_analytic_rms"]},
            ),
        ]
        return _finalize_fixture(
            fixture,
            common,
            checks,
            status="DESIGNED_GP01_L_GENERAL_3D_CURL_COUNTEREXAMPLE",
        )
    if fixture_id == "SYN-GP01-MULTISOURCE":
        loop = _multisource_loop_evidence(config, positions, masses)
        checks = [
            _computed_check(
                "algebraic_gain_induced_curl_computed_counterexample",
                curls["induced_curl_analytic_rms"] > 1e-6,
                {"rms_curl": curls["induced_curl_analytic_rms"]},
            ),
            _computed_check(
                "source_derived_environmental_closed_path_computed_failure",
                not loop["path_independence_passed"],
                loop,
            ),
        ]
        return _finalize_fixture(
            fixture,
            {**common, "source_derived_loop": loop},
            checks,
            status="DESIGNED_CURL_AND_ENVIRONMENT_PATH_COUNTEREXAMPLES",
        )

    null_tolerance = float(config["parameters"]["field_null_tolerance"])
    exact_null_recovered = center_field <= null_tolerance
    center_tidal = float(trace_free_tidal_norm(center_jacobian))
    center_weight = float(
        environment_gate(0.0, center_tidal, rho_star=1.0, tidal_star=1.0, q=1, r=1)
    )
    gamma_max = math.log(4.0)
    null_target = float(
        bounded_gamma_target(0.0, w=center_weight, a_star=0.01, n=2, gamma_max=gamma_max)
    )
    target_expected = center_weight * gamma_max
    arc = np.asarray([0.0, 1.0])
    if fixture_id == "SYN-GP01-SADDLE":
        force = np.asarray([1.0, center_field])
        refusal = _observe_transport_refusal(
            lambda: transport_t1(
                arc,
                force,
                np.ones(2),
                beta=0.5,
                l_reset=1.0,
                gamma_anchor=0.0,
                field_null_tolerance=null_tolerance,
            ),
            "field null",
        )
        refusal_check = "transport_null_refusal_observed"
    else:
        refusal = _observe_transport_refusal(
            lambda: transport_t1(
                arc,
                np.asarray([1.0, 0.5]),
                np.ones(2),
                beta=0.5,
                l_reset=1.0,
                gamma_anchor=0.0,
                anchored=False,
                field_null_tolerance=null_tolerance,
            ),
            "anchor",
        )
        refusal_check = "transport_unanchored_refusal_observed"
    checks = [
        _computed_check(
            "exact_center_field_null",
            exact_null_recovered,
            {"field_magnitude": center_field, "tolerance": null_tolerance},
        ),
        _computed_check(
            "source_derived_W_and_bounded_target_at_null",
            0.0 < center_weight <= 1.0 and abs(null_target - target_expected) <= 1e-14,
            {
                "center_trace_free_tidal_norm": center_tidal,
                "center_weight": center_weight,
                "computed_null_target": null_target,
                "expected_null_target": target_expected,
            },
        ),
        _computed_check(refusal_check, refusal["refusal_observed"], refusal),
    ]
    return _finalize_fixture(
        fixture,
        {
            **common,
            "declared_exact_null_fixture": True,
            "exact_null_recovered": exact_null_recovered,
            "center_trace_free_tidal_norm": center_tidal,
            "center_environment_weight": center_weight,
            "center_bounded_null_target": null_target,
            "transport_refusal": refusal,
        },
        checks,
        status="PASS_COMPUTED_NULL_RECOVERY_AND_TRANSPORT_QUARANTINE",
    )


def _transport_report(config: Mapping[str, Any]) -> dict[str, Any]:
    arc = np.linspace(0.0, 4.0, 41)
    force = np.exp(-arc)
    full_weight = np.ones_like(arc)
    beta = float(config["parameters"]["beta_principal"])
    t1 = transport_t1(arc, force, full_weight, beta=beta, l_reset=1.0, gamma_anchor=0.0)
    expected_t1 = beta * arc
    reset_cells = []
    for length in config["parameters"]["L_reset_over_R_b_grid"]:
        reset = transport_t1(
            arc,
            np.ones_like(arc),
            np.zeros_like(arc),
            beta=beta,
            l_reset=float(length),
            gamma_anchor=1.0,
        )
        error = float(np.max(np.abs(reset - np.exp(-arc / float(length)))))
        reset_cells.append({"L_reset_over_R_b": length, "max_abs_error": error})
    t2_cells = []
    for length in config["parameters"]["L_g_over_R_b_grid"]:
        target = np.ones_like(arc)
        t2 = transport_t2(
            arc,
            target,
            force,
            l_g=float(length),
            gamma_anchor=0.0,
        )
        error = float(np.max(np.abs(t2 - (1.0 - np.exp(-arc / float(length))))))
        t2_cells.append({"L_g_over_R_b": length, "max_abs_error": error})
    closed_log_f = np.asarray([0.0, -1.0, -2.0, -1.0, 0.0])
    constant_loop = closed_path_integral(closed_log_f, np.ones(5))
    quarantine_calls = {
        "unanchored": lambda: transport_t2(
            arc, np.ones_like(arc), force, l_g=1.0, gamma_anchor=0.0, anchored=False
        ),
        "multiple_anchors": lambda: transport_t2(
            arc, np.ones_like(arc), force, l_g=1.0, gamma_anchor=0.0, anchor_count=2
        ),
        "separatrix": lambda: transport_t2(
            arc,
            np.ones_like(arc),
            force,
            l_g=1.0,
            gamma_anchor=0.0,
            crosses_separatrix=True,
        ),
        "closed_field_line": lambda: transport_t2(
            arc,
            np.ones_like(arc),
            force,
            l_g=1.0,
            gamma_anchor=0.0,
            closed_field_line=True,
        ),
        "undeclared_domain_exit": lambda: transport_t2(
            arc,
            np.ones_like(arc),
            force,
            l_g=1.0,
            gamma_anchor=0.0,
            domain_exit="GRID_EDGE",
        ),
    }
    expected_fragments = {
        "unanchored": "anchor",
        "multiple_anchors": "anchors",
        "separatrix": "separatrix",
        "closed_field_line": "closed field",
        "undeclared_domain_exit": "domain exit",
    }
    quarantine_evidence = {
        key: _observe_transport_refusal(call, expected_fragments[key])
        for key, call in quarantine_calls.items()
    }
    return {
        "segment_rule": "ln(f) linear and W or target midpoint-constant per segment",
        "t1_full_weight_max_abs_error": float(np.max(np.abs(t1 - expected_t1))),
        "t1_reset_cells": reset_cells,
        "t2_relaxation_cells": t2_cells,
        "constant_weight_closed_integral": constant_loop,
        "validity_quarantine_evidence": quarantine_evidence,
        "all_validity_refusals_observed": all(
            item["refusal_observed"] for item in quarantine_evidence.values()
        ),
    }


def _limit_and_mapping_report(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    y = np.geomspace(1e-12, 1e12, 257)
    for n in config["parameters"]["n_grid"]:
        high = float(nu_n(y[-1], n))
        deep_gain = float(nu_n(y[0], n)) * y[0]
        x, mu = aqual_parametric_mapping(y, n)
        x_array, mu_array = np.asarray(x), np.asarray(mu)
        relative_residual = np.abs(mu_array * x_array - y) / y
        gamma, v_prime = action_target(y, n)
        rows.append(
            {
                "n": n,
                "high_field_nu_minus_one": high - 1.0,
                "deep_field_g_over_a_star_divided_by_sqrt_y": deep_gain / math.sqrt(y[0]),
                "aqual_mu_x_minus_y_max_relative": float(np.max(relative_residual)),
                "aqual_mu_min": float(np.min(mu_array)),
                "aqual_mu_max": float(np.max(mu_array)),
                "aqual_mu_positive": bool(np.all(mu_array > 0.0)),
                "aqual_x_strictly_monotonic": bool(np.all(np.diff(x_array) > 0.0)),
                "aqual_deep_mu_over_x": float(mu_array[0] / x_array[0]),
                "aqual_high_mu": float(mu_array[-1]),
                "action_gamma_minimum": float(np.min(gamma)),
                "action_V_prime_minimum": float(np.min(v_prime)),
                "action_target_residual_max_abs": float(
                    np.max(np.abs(v_prime - y * y * np.asarray(nu_n(y, n))))
                ),
                "action_high_field_regularity": action_regularity_class(n),
            }
        )
    return {
        "rows": rows,
        "exact_relation": "mu(x(y))*x(y)=y",
        "equivalence_scope": "spherical_or_valid_1D_curl_free_symmetry_only",
        "general_3d_equivalence": False,
    }


def _parameter_grid_report(
    config: Mapping[str, Any],
    *,
    n_rows: Sequence[Mapping[str, Any]],
    reset_cells: Sequence[Mapping[str, Any]],
    t2_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parameters = config["parameters"]
    amax_cells = []
    quasi_static_cells = []
    grid_points = 65
    coordinate = np.linspace(0.0, 1.0, grid_points)
    spacing = float(coordinate[1] - coordinate[0])
    length_cells = [float(parameters["L_g_zero_control"])] + [
        float(value) for value in parameters["L_g_over_R_b_grid"]
    ]
    for a_max in parameters["A_max_grid"]:
        gamma_max = math.log(float(a_max))
        lower, upper = ellipticity_bounds(gamma_max)
        null_target = float(bounded_gamma_target(0.0, w=0.7, a_star=0.01, n=2, gamma_max=gamma_max))
        amax_cells.append(
            {
                "A_max": a_max,
                "Gamma_max": gamma_max,
                "coefficient_minimum": lower,
                "coefficient_maximum": upper,
                "null_target": null_target,
                "null_target_expected": 0.7 * gamma_max,
            }
        )
        target = gamma_max * 0.8 * np.sin(math.pi * coordinate) ** 2
        for length in length_cells:
            solution = quasi_static_gain_1d(target, l_over_domain=length, spacing=spacing)
            quasi_static_cells.append(
                {
                    "A_max": a_max,
                    "L_g_over_R_b": length,
                    "solution_minimum": float(np.min(solution)),
                    "solution_maximum": float(np.max(solution)),
                    "left_Dirichlet_Gamma": float(solution[0]),
                    "right_Dirichlet_Gamma": float(solution[-1]),
                    "bounded_under_declared_M_matrix_conditions": bool(
                        np.min(solution) >= -1e-14 and np.max(solution) <= gamma_max + 1e-14
                    ),
                    "zero_length_recovers_target": (
                        bool(np.array_equal(solution, target)) if length == 0.0 else None
                    ),
                }
            )

    environment_cells = []
    for rho_star in parameters["rho_star_over_source_reference_grid"]:
        for tidal_star in parameters["T_star_over_source_reference_grid"]:
            for q in parameters["q_grid"]:
                for r in parameters["r_grid"]:
                    weight = float(
                        environment_gate(
                            0.5,
                            0.5,
                            rho_star=float(rho_star),
                            tidal_star=float(tidal_star),
                            q=int(q),
                            r=int(r),
                        )
                    )
                    environment_cells.append(
                        {
                            "rho_star_over_source_reference": rho_star,
                            "T_star_over_source_reference": tidal_star,
                            "q": q,
                            "r": r,
                            "computed_W": weight,
                            "bounded": 0.0 < weight <= 1.0,
                        }
                    )

    speed_of_light = 299_792_458.0
    radius_m = float(parameters["synthetic_R_b_m"])
    causal_cells = []
    for length_ratio in parameters["L_g_over_R_b_grid"]:
        for fraction in parameters["c_Gamma_over_c_grid"]:
            length_m = float(length_ratio) * radius_m
            tau_s = length_m / (float(fraction) * speed_of_light)
            speed = telegraph_characteristic_speed(length_m, tau_s)
            causal_cells.append(
                {
                    "L_g_over_R_b": length_ratio,
                    "c_Gamma_over_c": fraction,
                    "R_b_m": radius_m,
                    "L_g_m": length_m,
                    "tau_g_s": tau_s,
                    "hat_tau_c_tau_over_R_b": speed_of_light * tau_s / radius_m,
                    "computed_c_Gamma_over_c": speed / speed_of_light,
                    "finite_positive_scales": math.isfinite(length_m)
                    and length_m > 0.0
                    and math.isfinite(tau_s)
                    and tau_s > 0.0,
                    "necessary_speed_gate": speed <= speed_of_light * (1.0 + 1e-15),
                }
            )

    declared_counts = {
        "n": len(parameters["n_grid"]),
        "A_max": len(parameters["A_max_grid"]),
        "L_reset": len(parameters["L_reset_over_R_b_grid"]),
        "T2_transport_L_g": len(parameters["L_g_over_R_b_grid"]),
        "environment": len(parameters["rho_star_over_source_reference_grid"])
        * len(parameters["T_star_over_source_reference_grid"])
        * len(parameters["q_grid"])
        * len(parameters["r_grid"]),
        "quasi_static_A_max_by_L_g_including_zero": len(parameters["A_max_grid"])
        * (len(parameters["L_g_over_R_b_grid"]) + 1),
        "telegraph_L_g_by_speed": len(parameters["L_g_over_R_b_grid"])
        * len(parameters["c_Gamma_over_c_grid"]),
    }
    exercised_counts = {
        "n": len(n_rows),
        "A_max": len(amax_cells),
        "L_reset": len(reset_cells),
        "T2_transport_L_g": len(t2_cells),
        "environment": len(environment_cells),
        "quasi_static_A_max_by_L_g_including_zero": len(quasi_static_cells),
        "telegraph_L_g_by_speed": len(causal_cells),
    }
    exact_declared_values_exercised = (
        [row["n"] for row in n_rows] == parameters["n_grid"]
        and [cell["L_reset_over_R_b"] for cell in reset_cells]
        == parameters["L_reset_over_R_b_grid"]
        and [cell["L_g_over_R_b"] for cell in t2_cells] == parameters["L_g_over_R_b_grid"]
        and [cell["A_max"] for cell in amax_cells] == parameters["A_max_grid"]
    )
    return {
        "A_max_cells": amax_cells,
        "environment_cells": environment_cells,
        "quasi_static_cells": quasi_static_cells,
        "causal_cells": causal_cells,
        "declared_cell_counts": declared_counts,
        "exercised_cell_counts": exercised_counts,
        "exact_declared_values_exercised": exact_declared_values_exercised,
        "all_declared_cells_exercised": (
            declared_counts == exercised_counts and exact_declared_values_exercised
        ),
    }


def build_synthetic_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the complete declared synthetic battery with no observational loader."""

    validate_config(config)
    fixtures = config["synthetic_contract"]["fixtures"]
    fixture_results = [_sphere_fixture(config, fixtures[0])]
    fixture_results.extend(_spatial_fixture(config, fixture) for fixture in fixtures[1:])
    transport = _transport_report(config)
    limits = _limit_and_mapping_report(config)
    grids = _parameter_grid_report(
        config,
        n_rows=limits["rows"],
        reset_cells=transport["t1_reset_cells"],
        t2_cells=transport["t2_relaxation_cells"],
    )
    spatial = {item["fixture_id"]: item for item in fixture_results}
    limit_pass = all(
        abs(row["high_field_nu_minus_one"]) <= 1e-12
        and abs(row["deep_field_g_over_a_star_divided_by_sqrt_y"] - 1.0) <= 1e-6
        for row in limits["rows"]
    )
    mapping_pass = all(
        row["aqual_mu_x_minus_y_max_relative"] <= 5e-16
        and row["aqual_mu_positive"]
        and row["aqual_x_strictly_monotonic"]
        and abs(row["aqual_deep_mu_over_x"] - 1.0) <= 1e-6
        and abs(row["aqual_high_mu"] - 1.0) <= 1e-12
        for row in limits["rows"]
    )
    sphere_pass = spatial["SYN-GP01-SPHERE"]["all_required_checks_passed"]
    curl_detected = all(
        spatial[fixture_id]["induced_curl_analytic_rms"] > 1e-6
        for fixture_id in ("SYN-GP01-DISK", "SYN-GP01-MULTISOURCE")
    )
    environment_path_failure = not spatial["SYN-GP01-MULTISOURCE"]["source_derived_loop"][
        "path_independence_passed"
    ]
    nulls_and_refusals = (
        all(
            spatial[fixture_id]["all_required_checks_passed"]
            for fixture_id in ("SYN-GP01-SADDLE", "SYN-GP01-VOID")
        )
        and transport["all_validity_refusals_observed"]
    )
    conditional_bound_pass = (
        all(
            0.0 < cell["coefficient_minimum"] <= cell["coefficient_maximum"]
            and abs(cell["null_target"] - cell["null_target_expected"]) <= 1e-14
            for cell in grids["A_max_cells"]
        )
        and all(
            cell["bounded_under_declared_M_matrix_conditions"]
            and abs(cell["left_Dirichlet_Gamma"]) <= 1e-14
            and abs(cell["right_Dirichlet_Gamma"]) <= 1e-14
            for cell in grids["quasi_static_cells"]
        )
        and all(
            cell["zero_length_recovers_target"] is True
            for cell in grids["quasi_static_cells"]
            if cell["L_g_over_R_b"] == 0.0
        )
    )
    necessary_speed_pass = all(
        cell["finite_positive_scales"] and cell["necessary_speed_gate"]
        for cell in grids["causal_cells"]
    )
    parameter_coverage_pass = (
        grids["all_declared_cells_exercised"]
        and all(cell["bounded"] for cell in grids["environment_cells"])
        and all(cell["max_abs_error"] <= 1e-14 for cell in transport["t1_reset_cells"])
        and all(cell["max_abs_error"] <= 1e-14 for cell in transport["t2_relaxation_cells"])
    )
    filter_results = [
        {
            "filter_id": "F01_DIMENSIONS",
            "status": "PASS_DECLARED_UNIT_CONTRACT_AND_PHYSICAL_TELEGRAPH_SCALES",
            "observed_as_designed": necessary_speed_pass,
        },
        {
            "filter_id": "F02_HIGH_AND_DEEP_LIMITS",
            "status": "PASS" if limit_pass else "INVALID_LIMIT",
            "observed_as_designed": limit_pass,
        },
        {
            "filter_id": "F03_SPHERICAL_BTF_AND_TRANSITION",
            "status": "PASS_COMPUTED" if sphere_pass else "INVALID_SPHERICAL_LIMIT",
            "observed_as_designed": sphere_pass,
        },
        {
            "filter_id": "F04_EXACT_AQUAL_MAPPING",
            "status": (
                "PASS_KNOWN_FAMILY_COMPARATOR_SPHERICAL_OR_VALID_1D_ONLY"
                if mapping_pass
                else "INVALID_MAPPING"
            ),
            "observed_as_designed": mapping_pass,
        },
        {
            "filter_id": "F05_GENERAL_3D_CURL",
            "status": (
                "DESIGNED_FAIL_GP01_L_GENERAL_3D_THEORY" if curl_detected else "UNEXPECTED_PASS"
            ),
            "observed_as_designed": curl_detected,
        },
        {
            "filter_id": "F06_PATH_LOCAL_REDUCTION",
            "status": "PASS_LOCAL_EXACT_DIFFERENTIAL_CONTROL",
            "observed_as_designed": abs(transport["constant_weight_closed_integral"]) <= 1e-12,
        },
        {
            "filter_id": "F07_ENVIRONMENTAL_CLOSED_PATH",
            "status": "DESIGNED_FAIL_SOURCE_DERIVED_ARBITRARY_ROUTE_HISTORY",
            "observed_as_designed": environment_path_failure,
        },
        {
            "filter_id": "F08_FIELD_NULL_AND_SEPARATRIX",
            "status": "QUARANTINED_WITH_OBSERVED_TRANSPORT_REFUSALS",
            "observed_as_designed": nulls_and_refusals,
        },
        {
            "filter_id": "F09_CONDITIONAL_BOUNDED_ELLIPTICITY",
            "status": "CONDITIONAL_COEFFICIENT_BOUND_WITH_1D_M_MATRIX_CONTROL",
            "observed_as_designed": conditional_bound_pass,
        },
        {
            "filter_id": "F10_TELEGRAPH_NECESSARY_SPEED",
            "status": "PASS_NECESSARY_NOT_SUFFICIENT_PHYSICAL_UNITS",
            "observed_as_designed": necessary_speed_pass,
        },
        {
            "filter_id": "F11_CAUSAL_SOURCE_COMPLETION",
            "status": "BLOCKED_INSTANTANEOUS_POISSON_TARGET",
            "observed_as_designed": True,
        },
        {
            "filter_id": "F12_ENERGY_AND_ACTION",
            "status": "ACTION_PLACEHOLDER_TARGET_DERIVED_BUT_ALL_N_GRID_CELLS_SINGULAR",
            "observed_as_designed": all(
                row["action_V_prime_minimum"] > 0.0 and row["action_target_residual_max_abs"] == 0.0
                for row in limits["rows"]
            ),
        },
        {
            "filter_id": "F13_LIGHT_CAPTURE_CLOSURES",
            "status": "L0_NO_LIGHT_CLAIM_AND_C0_ISOLATED_CONSERVATIVE",
            "observed_as_designed": True,
        },
    ]
    _require(
        tuple(item["filter_id"] for item in filter_results) == FILTER_IDS, "filter order drift"
    )
    _require(parameter_coverage_pass, "declared parameter grid was not fully exercised")
    _require(all(item["observed_as_designed"] for item in filter_results), "synthetic gate drift")
    return {
        "status": "FOUNDATION_PASS_WITH_DESIGNED_FAILURES_AND_QUARANTINES",
        "real_rows": 0,
        "fixture_results": fixture_results,
        "transport": transport,
        "limits_and_aqual_mapping": limits,
        "parameter_grid_coverage": grids,
        "pde_contracts": {
            "elliptic_bound_status": "CONDITIONAL_ON_DECLARED_MAXIMUM_PRINCIPLE_AND_1D_M_MATRIX_CONTROL",
            "quasi_static_cells": grids["quasi_static_cells"],
            "causal_cells": grids["causal_cells"],
            "L_g_zero_is_quasi_static_control_not_telegraph_cell": True,
            "telegraph_overshoot_is_bounded": False,
            "telegraph_speed_is_sufficient_for_causality": False,
            "instantaneous_baryonic_poisson_source_remains": True,
            "quasi_static_equation_is_temporal_memory": False,
        },
        "action_audit": {
            "target_state": config["action_placeholder"]["target_state"],
            "target_potential_derivative": config["action_placeholder"][
                "target_potential_derivative"
            ],
            "potential_dimension": config["action_placeholder"]["potential_dimension"],
            "phi_euler_lagrange": config["action_placeholder"]["phi_euler_lagrange"],
            "gamma_euler_lagrange": config["action_placeholder"]["gamma_euler_lagrange"],
            "n_grid_regularity": [
                {"n": row["n"], "classification": row["action_high_field_regularity"]}
                for row in limits["rows"]
            ],
            "healthy_action_completed": False,
            "damping_derived_from_action": False,
            "status": "ACTION_PLACEHOLDER_QUARANTINED",
        },
        "filter_results": filter_results,
        "designed_failures": [
            "GP01-L develops curl in generic nonspherical fields and is not a general conservative 3D theory.",
            "Environment-weighted T1 transport has a computed nonzero source-derived closed-path integral, so arbitrary-route history is rejected.",
            "T1/T2 refuse field nulls, saddles, separatrices, multiple or missing anchors, closed lines, and undeclared exits.",
            "The telegraph principal speed bound is necessary only; the instantaneous Poisson-built target blocks a wholly causal claim.",
            "Phenomenological damping has no frozen receiver or energy ledger.",
            "The action target has a high-field singularity in every declared n cell and the ACTION_PLACEHOLDER remains incomplete.",
        ],
    }


def build_receipt(root: Path = Path("."), config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_config(root, config_path)
    source_path = root / SOURCE_PATH
    test_path = root / TEST_PATH
    _require(source_path.is_file(), "implementation source is missing")
    _require(test_path.is_file(), "implementation test is missing")
    synthetic = build_synthetic_report(config)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "foundation_id": config["foundation_id"],
        "version": config["version"],
        "status": synthetic["status"],
        "decision": DECISION,
        "config_binding": {
            "path": config_path.as_posix(),
            "file_sha256": _file_sha256(root / config_path),
            "content_sha256": _content_sha256(config),
            "section_sha256": dict(EXPECTED_CONFIG_SECTION_SHA256),
        },
        "implementation_binding": {
            "source_path": SOURCE_PATH.as_posix(),
            "source_sha256": _file_sha256(source_path),
            "test_path": TEST_PATH.as_posix(),
            "test_sha256": _file_sha256(test_path),
        },
        "variant_status": [
            {
                "variant_id": item["variant_id"],
                "status": item["status"],
                "response_scoring_eligible": item["response_scoring_eligible"],
            }
            for item in config["variants"]
        ],
        "formula_contract": config["equations"],
        "parameter_contract": config["parameters"],
        "boundary_contract": config["boundaries_and_initial_data"],
        "closures": config["closures"],
        "action_quarantine": config["action_placeholder"],
        "synthetic_report": synthetic,
        "counts": {
            "variants": len(VARIANT_IDS),
            "synthetic_fixtures": len(synthetic["fixture_results"]),
            "theory_filters": len(synthetic["filter_results"]),
            "designed_failures_and_quarantines": len(synthetic["designed_failures"]),
            "real_rows": 0,
            "response_scores": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access": config["zero_access"],
        "limitations": list(LIMITATIONS),
    }
    receipt["content_sha256"] = _content_sha256(receipt)
    validate_receipt(receipt, config, root=root)
    return receipt


def validate_receipt(
    receipt: Mapping[str, Any], config: Mapping[str, Any], *, root: Path = Path(".")
) -> None:
    validate_config(config)
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "foundation_id",
            "version",
            "status",
            "decision",
            "config_binding",
            "implementation_binding",
            "variant_status",
            "formula_contract",
            "parameter_contract",
            "boundary_contract",
            "closures",
            "action_quarantine",
            "synthetic_report",
            "counts",
            "claim_boundary",
            "zero_access",
            "limitations",
            "content_sha256",
        },
        "receipt",
    )
    _require(receipt["schema_version"] == RECEIPT_SCHEMA, "receipt schema changed")
    body = dict(receipt)
    observed_hash = body.pop("content_sha256", None)
    _require(observed_hash == _content_sha256(body), "receipt content hash invalid")
    _require(receipt["decision"] == DECISION, "receipt decision changed")
    _require(receipt["foundation_id"] == config["foundation_id"], "receipt ID unbound")
    _require(receipt["version"] == config["version"], "receipt version changed")
    binding = receipt["config_binding"]
    _require(isinstance(binding, dict), "config binding missing")
    _require_exact_keys(
        binding,
        {"path", "file_sha256", "content_sha256", "section_sha256"},
        "config binding",
    )
    _require(binding["path"] == CONFIG_PATH.as_posix(), "config path changed")
    _require(binding["content_sha256"] == EXPECTED_CONFIG_CONTENT_SHA256, "config content unbound")
    _require(binding["section_sha256"] == EXPECTED_CONFIG_SECTION_SHA256, "config sections unbound")
    config_file = root / CONFIG_PATH
    _require(config_file.is_file(), "bound config file is missing")
    _require(binding["file_sha256"] == _file_sha256(config_file), "config file hash changed")
    disk_config = _read_json(config_file)
    validate_config(disk_config)
    _require(disk_config == config, "bound config object differs from the on-disk config")

    implementation = receipt["implementation_binding"]
    _require(isinstance(implementation, dict), "implementation binding missing")
    _require_exact_keys(
        implementation,
        {"source_path", "source_sha256", "test_path", "test_sha256"},
        "implementation binding",
    )
    _require(implementation["source_path"] == SOURCE_PATH.as_posix(), "source path changed")
    _require(implementation["test_path"] == TEST_PATH.as_posix(), "test path changed")
    source_file, test_file = root / SOURCE_PATH, root / TEST_PATH
    _require(source_file.is_file(), "bound implementation source is missing")
    _require(test_file.is_file(), "bound implementation test is missing")
    _require(
        implementation["source_sha256"] == _file_sha256(source_file),
        "implementation source hash changed",
    )
    _require(
        implementation["test_sha256"] == _file_sha256(test_file),
        "implementation test hash changed",
    )

    _require(receipt["formula_contract"] == config["equations"], "formula contract changed")
    _require(receipt["parameter_contract"] == config["parameters"], "parameters changed")
    _require(
        receipt["boundary_contract"] == config["boundaries_and_initial_data"],
        "boundaries changed",
    )
    _require(receipt["closures"] == config["closures"], "closures changed")
    _require(receipt["action_quarantine"] == config["action_placeholder"], "action changed")
    _require(receipt["zero_access"] == config["zero_access"], "access ledger changed")
    _require(receipt["claim_boundary"] == config["claim_boundary"], "claims changed")
    _require(receipt["limitations"] == list(LIMITATIONS), "limitations changed")
    expected_variant_status = [
        {
            "variant_id": item["variant_id"],
            "status": item["status"],
            "response_scoring_eligible": item["response_scoring_eligible"],
        }
        for item in config["variants"]
    ]
    variant_status = receipt["variant_status"]
    _require(
        variant_status == expected_variant_status,
        "receipt variant status changed",
    )
    report = receipt["synthetic_report"]
    _require(isinstance(report, dict), "synthetic report missing")
    expected_report = build_synthetic_report(config)
    _require(report == expected_report, "synthetic report is not an exact recomputation")
    _require(receipt["status"] == expected_report["status"], "receipt status changed")
    counts = receipt["counts"]
    _require(
        counts
        == {
            "variants": 7,
            "synthetic_fixtures": 5,
            "theory_filters": 13,
            "designed_failures_and_quarantines": 6,
            "real_rows": 0,
            "response_scores": 0,
        },
        "receipt counts changed",
    )


def _atomic_no_replace(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return "EXISTING_IDENTICAL"
        raise GravityGainPersistenceFoundationError(
            f"refusing to overwrite non-identical output: {path.as_posix()}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise GravityGainPersistenceFoundationError(
                f"concurrent creator won; output preserved: {path.as_posix()}"
            ) from exc
        return "CREATED"
    finally:
        temporary_path.unlink(missing_ok=True)


def write_receipt(
    root: Path = Path("."),
    config_path: Path = CONFIG_PATH,
    output_path: Path = OUTPUT_PATH,
) -> tuple[dict[str, Any], str]:
    _require(output_path == OUTPUT_PATH, "output path is not the frozen confined path")
    receipt = build_receipt(root, config_path)
    publication = _atomic_no_replace(root / output_path, _canonical_bytes(receipt))
    return receipt, publication


def check_receipt(
    root: Path = Path("."),
    config_path: Path = CONFIG_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    config = load_config(root, config_path)
    stored = _read_json(root / output_path)
    validate_receipt(stored, config, root=root)
    expected = build_receipt(root, config_path)
    _require(stored == expected, "stored receipt differs from deterministic rebuild")
    return stored


def _summary(receipt: Mapping[str, Any], publication: str | None = None) -> dict[str, Any]:
    summary = {
        "valid": True,
        "decision": receipt["decision"],
        "content_sha256": receipt["content_sha256"],
        "synthetic_fixtures": receipt["counts"]["synthetic_fixtures"],
        "theory_filters": receipt["counts"]["theory_filters"],
        "designed_failures_and_quarantines": receipt["counts"]["designed_failures_and_quarantines"],
        "response_rows_opened": receipt["zero_access"]["response_rows_opened"],
        "response_scoring_unlocked": receipt["claim_boundary"]["response_scoring_unlocked"],
    }
    if publication is not None:
        summary["publication"] = publication
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "write", "check", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("."))
        command.add_argument("--config", type=Path, default=CONFIG_PATH)
        command.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            print(json.dumps(build_receipt(args.root, args.config), indent=2, sort_keys=True))
            return 0
        if args.command == "write":
            receipt, publication = write_receipt(
                args.root,
                args.config,
                args.output,
            )
            result = _summary(receipt, publication)
        else:
            result = _summary(check_receipt(args.root, args.config, args.output))
        print(json.dumps(result, sort_keys=True))
        return 0
    except GravityGainPersistenceFoundationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
