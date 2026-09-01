"""Executable void gravitational-load branches and response-unopened data contract."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_void_gravitational_load_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_gravitational_load_v1.py")
TEST_PATH = Path("tests/test_open_gravity_void_gravitational_load_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-gravitational-load-v1/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_SCHEMA = "invariant-open-gravity-void-gravitational-load-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-void-gravitational-load-receipt-1.0"
_CONFIG_RAW_SHA256 = "d32e5696a83d0646f5ab42fd8c15c9b6d0f7ff6bd4f96901c30f0933de4e560b"
_CONFIG_CONTENT_SHA256 = "a81db51ef9a079dd755e893cc39fdc4abe6dc80dded451d0119e592e7a000b63"
_MODULE_SEMANTIC_SHA256 = "20d47c667c1897b0a98e688818ba3f56dc904c1a4c38d0c5a1dec1a134001bc4"
_TEST_RAW_SHA256 = "4aa4b75a8d2bd33c537aa65aaeee08a35b71ff0c67fd920e6edc0adee2009379"
_BRANCH_IDS = (
    "VQ00_STANDARD_FLRW_PECULIAR_CONTROL",
    "VQ01_DIRECT_LOAD_OPTICAL_DEPTH",
    "VQ02_SLOWED_LIGHT_NONLINEAR_EXPOSURE",
    "VQ03_LOCAL_EQUILIBRIUM_LOAD",
    "VQ04_YUKAWA_NONLOCAL_BARYON_FEED",
    "VQ05_DIFFUSIVE_SOURCE_SINK_RESERVOIR",
    "VQ06_BARYONIC_COLUMN_ATTENUATION",
    "VQ07_INVERSE_DENSITY_HUBBLE_MIMIC",
    "VQ08_TWO_PHASE_VOID_MATTER",
    "VQ09_PHOTON_CARRIED_LOAD_MEMORY",
    "VQ10_RESERVOIR_PLUS_SLOWED_PHOTON",
)


class VoidLoadError(RuntimeError):
    """Raised when a frozen gravitational-load invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidLoadError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    replacements = (
        "_CONFIG_RAW_SHA256",
        "_CONFIG_CONTENT_SHA256",
        "_MODULE_SEMANTIC_SHA256",
        "_TEST_RAW_SHA256",
    )
    for name in replacements:
        marker = f'{name} = "'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        text = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoidLoadError(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-void-gravitational-load-v1", "ID changed")
    _require(
        config["status"] == "FROZEN_RESPONSE_UNOPENED_THEORY_AND_EMPIRICAL_PREFLIGHT",
        "status changed",
    )
    _require(tuple(row["id"] for row in config["branches"]) == _BRANCH_IDS, "branches changed")
    _require(len(config["synthetic_fixtures"]) == 9, "fixture registry changed")
    _require(len(config["separate_consistency_axes"]) == 11, "consistency axes changed")
    _require(len(config["published_neighbors"]) >= 13, "literature boundary shrank")
    _require(config["origin"]["path_measure"].startswith("dell=-u_a dx^a>0"), "path changed")
    _require(config["parameters"]["beta"] >= 0.0, "frozen beta changed sign")
    _require(config["parameters"]["rho_star"] > 0.0, "rho_star must regularize voids")
    _require(config["parameters"]["reservoir_dt"] > 0.0, "dt must be positive")
    _require(config["decision_policy"]["retain_every_failure"] is True, "failures dropped")
    preflight = config["empirical_preflight"]
    _require(preflight["response_status"] == "NOT_OPENED_NOT_SCORED", "response opened")
    _require("SOURCE_BLOCKED" in preflight["contract_status"], "source gate widened")
    _require(len(preflight["sources"]) == 3, "empirical sources changed")
    for source in preflight["sources"]:
        _require(
            source["revision"] is None and source["sha256"] is None, "unreceipted hash claimed"
        )
        _require(source["gate"].startswith("BLOCKED_"), "source prematurely admitted")
    _require(set(config["access_contract"].values()) == {0}, "access contract changed")
    claims = config["claim_boundary"]
    for key in (
        "exact_observational_payload_receipts",
        "observational_rows_opened",
        "real_data_fit",
        "historical_novelty_established",
        "covariant_action_established",
        "gravity_discovery",
    ):
        _require(claims[key] is False, f"claim widened: {key}")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "receipt path changed")
    _require(
        config["outputs"]["artifact_directory"] == ARTIFACT_DIR.as_posix(),
        "artifact path changed",
    )


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH, "void-load config")
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["local_bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing local binding: {row['role']}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"local binding changed: {row['role']}")
        observed[row["role"]] = digest
    _require(len(observed) == 2, "binding count changed")
    return observed


@dataclass(frozen=True)
class PathFixture:
    """One baryon-frame ray fixture sampled at equal invariant path increments."""

    fixture_id: str
    rho: tuple[float, ...]
    q_external: tuple[float, ...]
    source_multiplier_history: tuple[float, ...]
    scale_factor_ratio: float = 1.0
    peculiar_v_over_c: float = 0.0


@dataclass(frozen=True)
class Prediction:
    """Separate known-background and proposed-load predictions for one ray."""

    branch_id: str
    fixture_id: str
    control_log1pz: float
    load_log1pz: float
    arrival_delay_over_dx_c: float
    final_photon_load: float
    mean_field_load: float
    minimum_c_eff_over_c: float
    load_source_time_stretch: float


def _fixture_profiles(config: Mapping[str, Any]) -> list[PathFixture]:
    size = 64
    low, high = 0.15, 2.0

    def q_for(rho: Sequence[float]) -> tuple[float, ...]:
        return tuple(1.45 if value == low else 0.55 if value == high else 1.0 for value in rho)

    homogeneous = (1.0,) * size
    void_rich = (low,) * 44 + (high,) * 20
    matter_rich = (high,) * 44 + (low,) * 20
    void_then_matter = (low,) * 32 + (high,) * 32
    matter_then_void = tuple(reversed(void_then_matter))
    transient = tuple(low if 16 <= index < 48 else high for index in range(size))
    zero = (0.0,) * size
    histories = {
        "steady": (1.0,) * int(config["parameters"]["reservoir_steps"]),
        "transient": (0.0,) * 200 + (1.0,) * 200,
        "zero": (0.0,) * int(config["parameters"]["reservoir_steps"]),
    }
    rows = [
        PathFixture(
            "VF00_HOMOGENEOUS_STEADY", homogeneous, q_for(homogeneous), histories["steady"]
        ),
        PathFixture("VF01_VOID_RICH_PATH", void_rich, q_for(void_rich), histories["steady"]),
        PathFixture(
            "VF02_MATTER_RICH_MATCHED_LENGTH", matter_rich, q_for(matter_rich), histories["steady"]
        ),
        PathFixture(
            "VF03_ORDER_VOID_THEN_MATTER",
            void_then_matter,
            q_for(void_then_matter),
            histories["steady"],
        ),
        PathFixture(
            "VF04_ORDER_MATTER_THEN_VOID",
            matter_then_void,
            q_for(matter_then_void),
            histories["steady"],
        ),
        PathFixture(
            "VF05_TRANSIENT_SOURCE_ON", transient, q_for(transient), histories["transient"]
        ),
        PathFixture("VF06_ZERO_SOURCE_NULL", zero, zero, histories["zero"]),
        PathFixture(
            "VF07_FLRW_TIME_DILATION_CONTROL",
            homogeneous,
            q_for(homogeneous),
            histories["steady"],
            scale_factor_ratio=1.5,
            peculiar_v_over_c=0.002,
        ),
        PathFixture(
            "VF08_REVERSE_PATH_RECIPROCITY",
            matter_then_void,
            q_for(matter_then_void),
            histories["steady"],
        ),
    ]
    _require(
        tuple(row.fixture_id for row in rows) == tuple(config["synthetic_fixtures"]), "fixtures"
    )
    return rows


def _control_log1pz(fixture: PathFixture) -> float:
    velocity = fixture.peculiar_v_over_c
    _require(abs(velocity) < 1.0, "unphysical peculiar velocity")
    return math.log(fixture.scale_factor_ratio) + math.atanh(velocity)


def _integral(values: Sequence[float], dx: float) -> float:
    return math.fsum(values) * dx


def _periodic_yukawa_feed(
    rho: Sequence[float], *, amplitude: float, length: float
) -> tuple[float, ...]:
    """Normalized one-dimensional ray projection of the frozen 3D Yukawa convolution."""
    _require(length > 0.0 and amplitude >= 0.0, "invalid Yukawa parameters")
    size = len(rho)
    weights = [math.exp(-min(offset, size - offset) / length) for offset in range(size)]
    normalization = math.fsum(weights)
    return tuple(
        amplitude
        * math.fsum(weights[(index - source) % size] * rho[source] for source in range(size))
        / normalization
        for index in range(size)
    )


def _local_equilibrium(
    j_values: Sequence[float], rho: Sequence[float], *, gamma0: float, gamma_b: float
) -> tuple[float, ...]:
    _require(len(j_values) == len(rho), "local arrays mismatch")
    values = tuple(j / (gamma0 + gamma_b * density) for j, density in zip(j_values, rho))
    _require(all(math.isfinite(value) and value >= 0.0 for value in values), "bad local Q")
    return values


def _solve_reservoir(
    rho: Sequence[float],
    j_values: Sequence[float],
    source_history: Sequence[float],
    *,
    diffusion: float,
    gamma0: float,
    gamma_b: float,
    dx: float,
    dt: float,
) -> tuple[float, ...]:
    """Solve the declared source/diffusion/sink PDE by an explicit finite-difference IVP."""
    _require(len(rho) == len(j_values) and bool(rho), "reservoir arrays mismatch")
    courant = diffusion * dt / dx**2
    _require(0.0 <= courant <= 0.5, "explicit diffusion stability gate failed")
    state = [0.0] * len(rho)
    for multiplier in source_history:
        _require(0.0 <= multiplier <= 1.0, "source multiplier outside [0,1]")
        updated: list[float] = []
        for index, value in enumerate(state):
            laplacian = (
                state[(index - 1) % len(state)] - 2.0 * value + state[(index + 1) % len(state)]
            ) / dx**2
            derivative = (
                diffusion * laplacian
                + multiplier * j_values[index]
                - (gamma0 + gamma_b * rho[index]) * value
            )
            candidate = value + dt * derivative
            _require(
                candidate >= -1.0e-14 and math.isfinite(candidate), "reservoir lost positivity"
            )
            updated.append(max(0.0, candidate))
        state = updated
    return tuple(state)


def _column_attenuated_load(
    rho: Sequence[float], *, amplitude: float, length: float, sigma: float, dx: float
) -> tuple[float, ...]:
    _require(amplitude >= 0.0 and length > 0.0 and sigma >= 0.0, "bad attenuation")
    result: list[float] = []
    for receiver in range(len(rho)):
        weighted = 0.0
        norm = 0.0
        for source, density in enumerate(rho):
            distance_cells = abs(receiver - source)
            kernel = math.exp(-distance_cells * dx / length)
            left, right = sorted((receiver, source))
            column = math.fsum(rho[left + 1 : right]) * dx
            attenuation = math.exp(-sigma * column)
            weighted += density * kernel * attenuation
            norm += kernel
        result.append(amplitude * weighted / norm)
    return tuple(result)


def _photon_memory(
    q_field: Sequence[float],
    rho: Sequence[float],
    *,
    a: float,
    b0: float,
    b_b: float,
    eta: float,
    dx: float,
    c: float,
) -> tuple[float, float]:
    """Exact cellwise solution of the declared linear photon-state ODE."""
    photon_state = 0.0
    integrated_state = 0.0
    for field, density in zip(q_field, rho):
        rate = b0 + b_b * density
        _require(rate > 0.0, "photon relaxation must be positive")
        dt = dx / c
        decay = math.exp(-rate * dt)
        equilibrium = a * field / rate
        integral = photon_state * (1.0 - decay) / rate
        integral += equilibrium * (dt - (1.0 - decay) / rate)
        integrated_state += integral
        photon_state = decay * photon_state + (1.0 - decay) * equilibrium
    return photon_state, eta * integrated_state


def _fields_for(fixture: PathFixture, config: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
    p = config["parameters"]
    rho = fixture.rho
    feed = _periodic_yukawa_feed(rho, amplitude=float(p["A_feed"]), length=float(p["L_g"]))
    prescribed_feed = tuple(1.0 if density > 0.0 else 0.0 for density in rho)
    local = _local_equilibrium(
        prescribed_feed,
        rho,
        gamma0=float(p["Gamma_0"]),
        gamma_b=float(p["Gamma_b"]),
    )
    nonlocal_field = _local_equilibrium(
        feed, rho, gamma0=float(p["Gamma_0"]), gamma_b=float(p["Gamma_b"])
    )
    reservoir = _solve_reservoir(
        rho,
        feed,
        fixture.source_multiplier_history,
        diffusion=float(p["D_g"]),
        gamma0=float(p["Gamma_0"]),
        gamma_b=float(p["Gamma_b"]),
        dx=float(p["dx"]),
        dt=float(p["reservoir_dt"]),
    )
    attenuated = _column_attenuated_load(
        rho,
        amplitude=float(p["A_load"]),
        length=float(p["L_g"]),
        sigma=float(p["sigma_g"]),
        dx=float(p["dx"]),
    )
    inverse = tuple(
        ((1.0 + float(p["rho_star"])) / (value + float(p["rho_star"])))
        ** float(p["inverse_density_n"])
        for value in rho
    )
    return {
        "external": fixture.q_external,
        "feed": feed,
        "local": local,
        "nonlocal": nonlocal_field,
        "reservoir": reservoir,
        "attenuated": attenuated,
        "inverse": inverse,
    }


def predict(branch_id: str, fixture: PathFixture, config: Mapping[str, Any]) -> Prediction:
    _require(branch_id in _BRANCH_IDS, "unknown branch")
    p = config["parameters"]
    dx, c = float(p["dx"]), float(p["c"])
    eta, beta = float(p["eta"]), float(p["beta"])
    fields = _fields_for(fixture, config)
    control = _control_log1pz(fixture)
    load = 0.0
    arrival_delay = 0.0
    final_photon_load = 0.0
    field = (0.0,) * len(fixture.rho)
    minimum_speed = 1.0

    if branch_id == _BRANCH_IDS[0]:
        pass
    elif branch_id == _BRANCH_IDS[1]:
        field = fields["external"]
        load = eta / c * _integral(field, dx)
    elif branch_id == _BRANCH_IDS[2]:
        field = fields["external"]
        _require(all(1.0 + beta * value > 0.0 for value in field), "c_eff pole")
        load = eta / c * _integral([value + beta * value**2 for value in field], dx)
        arrival_delay = beta / c * _integral(field, dx)
        minimum_speed = min(1.0 / (1.0 + beta * value) for value in field)
    elif branch_id == _BRANCH_IDS[3]:
        field = fields["local"]
        load = eta / c * _integral(field, dx)
    elif branch_id == _BRANCH_IDS[4]:
        field = fields["nonlocal"]
        load = eta / c * _integral(field, dx)
    elif branch_id == _BRANCH_IDS[5]:
        field = fields["reservoir"]
        load = eta / c * _integral(field, dx)
    elif branch_id == _BRANCH_IDS[6]:
        field = fields["attenuated"]
        load = eta / c * _integral(field, dx)
    elif branch_id == _BRANCH_IDS[7]:
        field = fields["inverse"]
        load = float(p["H_g"]) / c * _integral(field, dx)
    elif branch_id == _BRANCH_IDS[8]:
        threshold = float(p["void_density_threshold_over_mean"])
        void_cells = sum(value < threshold for value in fixture.rho)
        matter_cells = len(fixture.rho) - void_cells
        load = (float(p["H_v"]) * void_cells * dx + float(p["H_m"]) * matter_cells * dx) / c
        field = tuple(1.0 if value < threshold else 0.0 for value in fixture.rho)
    elif branch_id == _BRANCH_IDS[9]:
        field = fields["local"]
        final_photon_load, load = _photon_memory(
            field,
            fixture.rho,
            a=float(p["photon_a"]),
            b0=float(p["photon_b0"]),
            b_b=float(p["photon_b_b"]),
            eta=float(p["photon_eta"]),
            dx=dx,
            c=c,
        )
    else:
        field = fields["reservoir"]
        _require(all(1.0 + beta * value > 0.0 for value in field), "c_eff pole")
        load = eta / c * _integral([value + beta * value**2 for value in field], dx)
        arrival_delay = beta / c * _integral(field, dx)
        minimum_speed = min(1.0 / (1.0 + beta * value) for value in field)

    _require(load >= 0.0 and math.isfinite(load), "invalid redshift exposure")
    return Prediction(
        branch_id=branch_id,
        fixture_id=fixture.fixture_id,
        control_log1pz=control,
        load_log1pz=load,
        arrival_delay_over_dx_c=arrival_delay,
        final_photon_load=final_photon_load,
        mean_field_load=math.fsum(field) / len(field),
        minimum_c_eff_over_c=minimum_speed,
        load_source_time_stretch=1.0,
    )


def synthetic_predictions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        asdict(predict(branch_id, fixture, config))
        for fixture in _fixture_profiles(config)
        for branch_id in _BRANCH_IDS
    ]
    _require(len(rows) == 99, "synthetic matrix changed")
    return rows


def _branch_controls(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grades: dict[str, tuple[str, ...]] = {
        _BRANCH_IDS[0]: (
            "KNOWN_GR_STRESS_ENERGY",
            "CAUSAL_METRIC_IVP",
            "COMMON_METRIC_CLOSURE",
            "DERIVES_1_PLUS_Z_STRETCH",
            "PRESERVED_FOR_GEODESIC_PHOTON_NUMBER",
            "KNOWN_GEOMETRIC_DELAY",
            "ACHROMATIC_VACUUM",
            "CMB_CONTROL_BASELINE",
            "TOLMAN_CONTROL_BASELINE",
            "KNOWN_NOT_NOVEL",
        ),
        _BRANCH_IDS[1]: (
            "MISSING_RECEIVER",
            "CONDITIONAL_ON_Q",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "TIRED_LIGHT_ADJACENT",
        ),
        _BRANCH_IDS[2]: (
            "MISSING_MEDIUM_ACTION",
            "CONDITIONAL_ON_Q",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "POSITIVE_BETA_Q_DELAY",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "SLOWED_LIGHT_ADJACENT",
        ),
        _BRANCH_IDS[3]: (
            "MISSING_FIELD_ENERGY",
            "INSTANTANEOUS_LOCAL_CLOSURE",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "SOURCE_SINK_SYNTHESIS_UNAUDITED",
        ),
        _BRANCH_IDS[4]: (
            "MISSING_FIELD_ENERGY",
            "NONLOCAL_KERNEL_WITHOUT_RETARDATION",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "YUKAWA_NONLOCAL_NEIGHBORS",
        ),
        _BRANCH_IDS[5]: (
            "MISSING_FIELD_ACTION",
            "PARABOLIC_INSTANTANEOUS_TAIL",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "REACTION_DIFFUSION_NEIGHBOR",
        ),
        _BRANCH_IDS[6]: (
            "MISSING_ABSORBER_LEDGER",
            "NONLOCAL_COLUMN_WITHOUT_RETARDATION",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "DISTINCT_GEOMETRIC_SYNTHESIS_UNAUDITED",
        ),
        _BRANCH_IDS[7]: (
            "PHENOMENOLOGICAL_NO_RECEIVER",
            "LOCAL_ALGEBRAIC",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "EMPIRICAL_ANSATZ",
        ),
        _BRANCH_IDS[8]: (
            "PHENOMENOLOGICAL_NO_RECEIVER",
            "PHASE_LABEL_STATIC",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "COARSE_GRAINED_ANSATZ",
        ),
        _BRANCH_IDS[9]: (
            "QGAMMA_STATE_NOT_ENERGY_RECEIVER",
            "PHOTON_ODE_CAUSAL_Q_NOT_CLOSED",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "NONE_BEYOND_C",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "ORDER_MEMORY_SYNTHESIS_CANDIDATE",
        ),
        _BRANCH_IDS[10]: (
            "MISSING_FIELD_AND_MEDIUM_ACTION",
            "INHERITS_PARABOLIC_TAIL",
            "MISSING",
            "NOT_DERIVED",
            "AT_RISK",
            "POSITIVE_BETA_Q_DELAY",
            "ACHROMATIC_BY_ASSERTION",
            "AT_RISK",
            "AT_RISK",
            "COMBINED_SYNTHESIS_CANDIDATE",
        ),
    }
    axes = (
        "receiver_or_action_energy_accounting",
        "causality_and_initial_value_problem",
        "common_matter_light_closure",
        "source_lightcurve_time_dilation",
        "distance_duality_and_photon_number",
        "arrival_time_delay",
        "chromaticity",
        "CMB_blackbody_and_anisotropy",
        "Tolman_surface_brightness",
        "historical_novelty",
    )
    branch_by_id = {row["id"]: row for row in config["branches"]}
    rows: list[dict[str, Any]] = []
    for branch_id in _BRANCH_IDS:
        row = {
            "branch_id": branch_id,
            "class": branch_by_id[branch_id]["class"],
            "observational_fit": "UNOPENED_NOT_SCORED",
        }
        row.update(dict(zip(axes, grades[branch_id])))
        rows.append(row)
    return rows


def _priority_ledger() -> list[dict[str, Any]]:
    order = (
        (_BRANCH_IDS[9], 1, "LEAD_ORDER_SENSITIVE_FALSIFIER_ACTION_REQUIRED"),
        (_BRANCH_IDS[10], 2, "LEAD_JOINT_REDSHIFT_DELAY_FALSIFIER_ACTION_REQUIRED"),
        (_BRANCH_IDS[6], 3, "RETAIN_GEOMETRIC_ATTENUATION_FALSIFIER"),
        (_BRANCH_IDS[5], 4, "RETAIN_TRANSIENT_RESERVOIR_FALSIFIER"),
        (_BRANCH_IDS[8], 5, "FIRST_LOW_COMPLEXITY_EMPIRICAL_SCREEN"),
        (_BRANCH_IDS[7], 6, "RETAIN_CONTINUOUS_DENSITY_SCREEN"),
        (_BRANCH_IDS[2], 7, "RETAIN_NONLINEAR_DELAY_SCREEN"),
        (_BRANCH_IDS[4], 8, "RETAIN_NONLOCAL_FEED_SCREEN"),
        (_BRANCH_IDS[3], 9, "RETAIN_LOCAL_EQUILIBRIUM_BASELINE"),
        (_BRANCH_IDS[1], 10, "RETAIN_DIRECT_OPTICAL_BASELINE"),
        (_BRANCH_IDS[0], 0, "MANDATORY_KNOWN_CONTROL"),
    )
    return [
        {
            "branch_id": branch,
            "next_test_priority": rank,
            "disposition": disposition,
            "truth_score": "NOT_ASSIGNED",
            "data_fit_score": "UNOPENED",
        }
        for branch, rank, disposition in order
    ]


def _equivalence_ledger(config: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for branch in config["branches"]:
        for index, statement in enumerate(branch["limits"]):
            rows.append(
                {
                    "branch_id": branch["id"],
                    "kind": "LIMIT",
                    "index": str(index),
                    "statement": statement,
                }
            )
        for index, statement in enumerate(branch["equivalences"]):
            rows.append(
                {
                    "branch_id": branch["id"],
                    "kind": "EQUIVALENCE",
                    "index": str(index),
                    "statement": statement,
                }
            )
    return rows


def _counterexamples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixtures = {row.fixture_id: row for row in _fixture_profiles(config)}
    predictions = {
        (branch, fixture): predict(branch, fixtures[fixture], config)
        for branch in _BRANCH_IDS
        for fixture in fixtures
    }
    order_a = predictions[(_BRANCH_IDS[9], "VF03_ORDER_VOID_THEN_MATTER")].load_log1pz
    order_b = predictions[(_BRANCH_IDS[9], "VF04_ORDER_MATTER_THEN_VOID")].load_log1pz
    direct_a = predictions[(_BRANCH_IDS[1], "VF03_ORDER_VOID_THEN_MATTER")].load_log1pz
    direct_b = predictions[(_BRANCH_IDS[1], "VF04_ORDER_MATTER_THEN_VOID")].load_log1pz
    return [
        {
            "id": "CX00_CONSTANT_Q_HUBBLE_DEGENERACY",
            "branch": "VQ01/VQ07/VQ08",
            "synthetic_result": "EXACT_ON_ONE_CONSTANT_EXPOSURE_DOMAIN",
            "meaning": "A distance-redshift fit alone cannot identify load rather than expansion.",
        },
        {
            "id": "CX01_ORDER_BLIND_DIRECT_COLUMN",
            "branch": _BRANCH_IDS[1],
            "synthetic_result": abs(direct_a - direct_b),
            "meaning": "Direct optical depth cannot distinguish rearrangements with the same Q column.",
        },
        {
            "id": "CX02_ORDER_SENSITIVE_PHOTON_MEMORY",
            "branch": _BRANCH_IDS[9],
            "synthetic_result": order_a - order_b,
            "meaning": "Finite photon memory preserves ordering at identical total phase columns.",
        },
        {
            "id": "CX03_DIFFUSION_CAUSALITY",
            "branch": "VQ05/VQ10",
            "synthetic_result": "PARABOLIC_KERNEL_HAS_INSTANTANEOUS_SUPPORT",
            "meaning": "The executable PDE is well posed but not a relativistic causal completion.",
        },
        {
            "id": "CX04_TIME_DILATION",
            "branch": "VQ01-VQ10",
            "synthetic_result": "LOAD_SOURCE_TIME_STRETCH_EQUALS_1",
            "meaning": "No load branch derives the observed 1+z source-clock stretch; an external metric background can preserve it but then load is residual, not a replacement.",
        },
        {
            "id": "CX05_ENERGY_RECEIVER",
            "branch": "VQ01-VQ10",
            "synthetic_result": "NO_ACTION_CLOSES_LOST_PHOTON_FOUR_MOMENTUM",
            "meaning": "q_gamma records memory but is not yet an energy-momentum receiver.",
        },
        {
            "id": "CX06_DISTANCE_DUALITY_TOLMAN_CMB",
            "branch": "VQ01-VQ10",
            "synthetic_result": "NOT_DERIVED",
            "meaning": "A redshift law is not enough; phase-space occupation, angular distance, surface brightness and blackbody preservation remain separate tests.",
        },
        {
            "id": "CX07_REDSHIFT_SPACE_VOID_CIRCULARITY",
            "branch": "EMPIRICAL_PREFLIGHT",
            "synthetic_result": "MANDATORY_CONTROL",
            "meaning": "Void geometry inferred from galaxy redshifts can correlate with the same redshift response by construction.",
        },
        {
            "id": "CX08_ZERO_BARYON_INVERSE_DENSITY",
            "branch": _BRANCH_IDS[7],
            "synthetic_result": predictions[(_BRANCH_IDS[7], "VF06_ZERO_SOURCE_NULL")].load_log1pz,
            "meaning": "VQ07 predicts load even with no baryonic feed; it is explicitly phenomenology, not a derivation of matter-fed Q.",
        },
    ]


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    _require(bool(rows), "empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().replace("\r\n", "\n").encode()


def _report(config: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> bytes:
    by_key = {(row["branch_id"], row["fixture_id"]): row for row in predictions}
    order_delta = (
        by_key[(_BRANCH_IDS[9], "VF03_ORDER_VOID_THEN_MATTER")]["load_log1pz"]
        - by_key[(_BRANCH_IDS[9], "VF04_ORDER_MATTER_THEN_VOID")]["load_log1pz"]
    )
    vq10 = by_key[(_BRANCH_IDS[10], "VF01_VOID_RICH_PATH")]
    text = f"""# Void gravitational-load theory packet

## Outcome

**PASS** for eleven dimensionally closed, executable branches and a response-unopened empirical contract. **SOURCE BLOCKED** for a real-data score because exact immutable Cosmicflows-4, VAST/VIDE, and Pantheon+ payload revisions and SHA-256 receipts have not been admitted. No observational response row was opened.

The ray variable is the invariant baryon-frame measure `dell=-u_a dx^a`, with `dt_b=dell/c`; no null affine interval or photon proper time is used. The synthetic matrix contains {len(predictions)} predictions from laws solved over nine fixtures. The reaction-diffusion state is time-stepped from an initial condition, Yukawa and attenuation fields are convolved from baryons, and photon memory is integrated by the exact cellwise ODE solution.

## Strongest distinctive predictions

1. **VQ09 ordering memory:** two rays with identical total void and matter columns but reversed order differ by `{order_delta:.12g}` in synthetic load `ln(1+z)`. VQ01, VQ07 and VQ08 are order blind. A confirmed ordering residual at fixed independent distance is more diagnostic than another distance-only fit.
2. **VQ10 redshift-delay pairing:** in the void-rich fixture it predicts load `ln(1+z)={vq10["load_log1pz"]:.12g}` and a positive dimensionless extra delay `{vq10["arrival_delay_over_dx_c"]:.12g}` from the same solved Q with no independent lens or delay multiplier. This pairing is a stronger falsifier than redshift alone.
3. **VQ08 first empirical screen:** at matched independently measured distance, larger foreground void chord fraction must mean larger signed redshift residual when `H_v>H_m`. It is easy to test but is only a coarse ansatz and has low mechanism uniqueness.

## What is not being hidden

- VQ05 and VQ10 use a parabolic diffusion equation with instantaneous mathematical tails. That is a retained causal-completion failure, not a data failure.
- VQ01-VQ10 do not derive supernova source-clock time dilation. If used as replacements for expansion they face that control; if used as residual additions, the external metric background supplies the stretch.
- No branch yet supplies an action or receiver that gains exactly the photon four-momentum lost. The photon state in VQ09 is memory, not an energy ledger.
- Etherington reciprocity, Tolman surface brightness, CMB blackbody/anisotropy, chromaticity, and arrival delay are separately graded rather than compressed into a fit score.
- A constant load is Hubble-degenerate, VQ07 creates load even with no baryonic source, and redshift-space void geometry risks circularity. All remain in `counterexamples.csv`.

## Frozen real-data test

Cross Cosmicflows-4 independent group distances with public SDSS DR7 VAST/VIDE void geometry. Compute exact ordered ray/zone intersections inside the released mask; never substitute effective-radius spheres for missing topology except as a labeled tertiary sensitivity. Calibrate redshift-space-to-real-space chord exposure and void-beta/RSD uncertainty on frozen mocks. Fit signed `ln(1+z)` residuals after the registered FLRW, Hoffman bias-Gaussianized/lognormal distance, peculiar-flow, Local-Void outflow, bulk/shear, distance-method, sky-sector, local-density, selection, random-rotation, scrambled-void, alternate-finder, ISW and timescape controls. Group-hash folds 8-9 remain sealed. Pantheon+ is a covariance-aware cross-check and cannot tune the CF4 analysis.

## Publication boundary

The executable taxonomy, equivalence ledger, and two joint discriminators can support a theory/methods note after independent code and historical audit. No historical novelty, action completion, empirical fit, or gravity discovery is claimed. The nearest-neighbor ledger explicitly covers tired light, timescape/backreaction, ISW, refracted/nonlocal gravity, supernova time dilation, distance duality, Tolman surface brightness, Cosmicflows-4, VAST/VIDE and Pantheon+.
"""
    return text.encode()


def _artifact_payloads(config: Mapping[str, Any]) -> dict[str, bytes]:
    predictions = synthetic_predictions(config)
    branch_cards = []
    controls = {row["branch_id"]: row for row in _branch_controls(config)}
    priority = {row["branch_id"]: row for row in _priority_ledger()}
    for branch in config["branches"]:
        branch_cards.append(
            {
                "branch_id": branch["id"],
                "class": branch["class"],
                "field_law": branch["field_law"],
                "photon_law": branch["photon_law"],
                "dimension_check": branch["dimension_check"],
                "next_test_priority": priority[branch["id"]]["next_test_priority"],
                "disposition": priority[branch["id"]]["disposition"],
                "observational_fit": controls[branch["id"]]["observational_fit"],
            }
        )
    preflight = {
        "schema": "invariant-open-gravity-void-gravitational-load-preflight-1.0",
        **config["empirical_preflight"],
        "content_sha256": "",
    }
    preflight["content_sha256"] = content_sha256({**preflight, "content_sha256": ""})
    fixture_payload = {
        "schema": "invariant-open-gravity-void-gravitational-load-fixtures-1.0",
        "path_measure": config["origin"]["path_measure"],
        "solver": {
            "reservoir": "explicit periodic finite difference from Q(t=0)=0 with frozen stable dt",
            "photon_memory": "exact cellwise constant-coefficient ODE update and exact integrated q_gamma",
            "nonlocal_feed": "normalized 1D periodic ray-projection fixture of the declared 3D Yukawa kernel",
            "attenuation": "discrete intervening baryonic column between every source-receiver pair",
        },
        "fixtures": [
            {
                "fixture_id": row.fixture_id,
                "cells": len(row.rho),
                "rho_sum": math.fsum(row.rho),
                "q_external_sum": math.fsum(row.q_external),
                "source_history_sum": math.fsum(row.source_multiplier_history),
                "scale_factor_ratio": row.scale_factor_ratio,
                "peculiar_v_over_c": row.peculiar_v_over_c,
                "rho_sha256": content_sha256(row.rho),
                "q_external_sha256": content_sha256(row.q_external),
            }
            for row in _fixture_profiles(config)
        ],
    }
    return {
        "branch-cards.csv": _csv_bytes(branch_cards),
        "consistency-ledger.csv": _csv_bytes(_branch_controls(config)),
        "equivalence-ledger.csv": _csv_bytes(_equivalence_ledger(config)),
        "synthetic-predictions.csv": _csv_bytes(predictions),
        "fixtures.json": _canonical(fixture_payload),
        "empirical-preflight.json": _canonical(preflight),
        "counterexamples.csv": _csv_bytes(_counterexamples(config)),
        "report.md": _report(config, predictions),
    }


def _artifact_index(payloads: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "path": (ARTIFACT_DIR / name).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]


def build_receipt() -> tuple[dict[str, Any], dict[str, bytes]]:
    config = load_config()
    bindings = _validate_bindings(config)
    predictions = synthetic_predictions(config)
    payloads = _artifact_payloads(config)
    controls = _branch_controls(config)
    counterexamples = _counterexamples(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_EXECUTABLE_VOID_LOAD_BRANCHES_EMPIRICAL_SCORE_SOURCE_BLOCKED_RESPONSE_UNOPENED",
        "decision": "ADVANCE_VQ09_ORDER_MEMORY_AND_VQ10_COUPLED_REDSHIFT_DELAY_TO_INDEPENDENT_AUDIT_USE_VQ08_AS_FIRST_DATA_SCREEN_RETAIN_ALL_BRANCHES_AND_FAILURES",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
            "local_receipts": bindings,
            "user_attachment_sha256": config["origin"]["user_attachment_sha256"],
        },
        "summary": {
            "branches": len(_BRANCH_IDS),
            "synthetic_fixtures": len(config["synthetic_fixtures"]),
            "synthetic_predictions": len(predictions),
            "separately_graded_consistency_cells": len(controls) * 11,
            "equivalence_and_limit_rows": len(_equivalence_ledger(config)),
            "retained_counterexamples": len(counterexamples),
            "published_neighbors": len(config["published_neighbors"]),
            "observational_response_rows_opened": 0,
            "observational_response_rows_scored": 0,
            "artifact_index": _artifact_index(payloads),
        },
        "strongest_unique_testable_discriminator": config["decision_policy"]["lead_discriminator"],
        "next_empirical_test": config["empirical_preflight"]["primary_question"],
        "source_status": config["empirical_preflight"]["contract_status"],
        "priority_ledger": _priority_ledger(),
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "independent_audit_required": True,
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt, payloads


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing to overwrite different file: {path}")
        return "EXISTING_IDENTICAL"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "CREATED"


def write_package() -> str:
    receipt, payloads = build_receipt()
    statuses = [
        _atomic_no_clobber(ARTIFACT_DIR / name, payload) for name, payload in payloads.items()
    ]
    statuses.append(_atomic_no_clobber(OUTPUT_PATH, _canonical(receipt)))
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def check_package() -> dict[str, Any]:
    observed = _read_json(OUTPUT_PATH, "void-load receipt")
    expected, payloads = build_receipt()
    _require(observed == expected, "receipt differs from deterministic rebuild")
    _require(
        observed["content_sha256"] == content_sha256({**observed, "content_sha256": ""}),
        "receipt hash changed",
    )
    for name, payload in payloads.items():
        path = ARTIFACT_DIR / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == payload, f"artifact changed: {name}")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.action == "build":
        print(write_package())
        return 0
    if arguments.action == "check":
        check_package()
        print("VALID")
        return 0
    receipt, _ = build_receipt()
    print(receipt["status"])
    print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
