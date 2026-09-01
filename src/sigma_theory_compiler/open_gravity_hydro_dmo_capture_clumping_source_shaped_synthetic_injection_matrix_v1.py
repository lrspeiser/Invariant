"""Response-blind hydro/DMO source-shaped synthetic matrix for Lane 4.

The population cadence and field vocabulary are bound to public TNG100 and
CAMELS metadata.  Every merger history, hydro/DMO pair, truth injection, and
response is synthetic.  This module never opens a simulation payload.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    Availability,
    DataElement,
    DataRole,
    ExperimentRole,
    UncertaintyKind,
    catalogue_from_elements,
)
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import (
    AdapterRegistration,
    validate_adapter_registry,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    FormulaExecutionBinding,
    ResourceBounds,
    validate_binding_catalogue,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 import (
    ObservableComparison,
    ParameterCell,
    ScenarioRuntimeValues,
)
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v2 import (
    run_discovery_matrix_v2,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    SyntheticSuiteRelease,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    AnchorBinding,
    AxisSpec,
    EmittedPredictionSpec,
    FeatureValueRef,
    ScenarioDescriptor,
    UncertaintyRef,
    array_sha256,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path(
    "configs/open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1.json"
)
PARAMETER_SCHEMA_PATH = Path(
    "configs/open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1.parameters.schema.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v1"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
MATRIX_PATH = OUTPUT_DIR / "matrix-result.json"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "invariance-identifiability-and-blocks.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]

_DOMAIN = "capture-clumping-planar-encounter"
_GEOMETRY = "center-of-mass-planar-encounter"
_TIME_MODE = "public-cadence-shaped-interval-censored"
_FRAME = "center-of-mass-planar-encounter"

_EXECUTABLE = (
    "CM01_CONSERVATIVE_THREE_BODY_CAPTURE",
    "DC00_NEWTONIAN_FOCUSING_CONTROL",
    "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL",
    "DC05_TIMEWELL_SINGLE_MEMORY_BATH",
    "DC06_TIMEWELL_BIMODAL_MEMORY_BATH",
    "DC07_COMPRESSION_GATED_BATH",
)
_ENTRYPOINTS = {
    "CM01_CONSERVATIVE_THREE_BODY_CAPTURE": "cm01_conservative_three_body_adapter",
    "DC00_NEWTONIAN_FOCUSING_CONTROL": "dc00_newtonian_focusing_adapter",
    "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL": "dc01_static_force_amplification_adapter",
    "DC05_TIMEWELL_SINGLE_MEMORY_BATH": "dc05_single_memory_bath_adapter",
    "DC06_TIMEWELL_BIMODAL_MEMORY_BATH": "dc06_bimodal_memory_bath_adapter",
    "DC07_COMPRESSION_GATED_BATH": "dc07_compression_gated_bath_adapter",
}
_SOURCE_FEATURES = (
    "source.scalar.activation-scale",
    "source.scalar.cooling-control",
    "source.scalar.gas-fraction",
    "source.scalar.history-memory",
    "source.scalar.impact-parameter",
    "source.scalar.initial-radial-velocity",
    "source.scalar.initial-separation",
    "source.scalar.mass-ratio",
    "source.scalar.pericenter-proxy",
    "source.scalar.relaxation-time",
    "source.scalar.role-code",
    "source.scalar.shock-mach-control",
    "source.scalar.temperature",
    "source.scalar.total-mass",
    "source.scalar.wake-coulomb-log",
    "source.vector.cadence-coordinate",
    "source.vector.cadence-interval-lower",
    "source.vector.cadence-interval-upper",
    "source.vector.encounter-time",
)
_DYNAMIC_FEATURES = (
    "source.scalar.activation-scale",
    "source.scalar.history-memory",
    "source.scalar.impact-parameter",
    "source.scalar.initial-radial-velocity",
    "source.scalar.initial-separation",
    "source.scalar.mass-ratio",
    "source.scalar.temperature",
    "source.scalar.total-mass",
    "source.vector.encounter-time",
)
_INACTIVE_SOURCE_FEATURES = tuple(sorted(set(_SOURCE_FEATURES) - set(_DYNAMIC_FEATURES)))
_PREDICTIONS = (
    "prediction.vector.entropy",
    "prediction.vector.receiver-energy",
    "prediction.vector.separation",
    "prediction.vector.visible-pair-energy",
)
_RESPONSES = {
    value: value.replace("prediction.", "response.synthetic-", 1) for value in _PREDICTIONS
}
_UNITS = {
    "prediction.vector.entropy": "encounter entropy",
    "prediction.vector.receiver-energy": "encounter energy",
    "prediction.vector.separation": "encounter length",
    "prediction.vector.visible-pair-energy": "encounter energy",
}
_SIGMA_KEY = {
    "prediction.vector.entropy": "entropy_sigma",
    "prediction.vector.receiver-energy": "receiver_energy_sigma",
    "prediction.vector.separation": "separation_sigma",
    "prediction.vector.visible-pair-energy": "energy_sigma",
}
_MECHANISM_CODE = {formula_id: index for index, formula_id in enumerate(_EXECUTABLE)}
_PREDECESSOR_MODULE_SHA256 = "4b92060be6314e5721aa43b1112a89bd17ab2ad4e1fbe7ddeece0992284c9a82"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaViolation(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _repo_path(value: str | Path) -> Path:
    parsed = PurePosixPath(str(value).replace("\\", "/"))
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SchemaViolation("Lane 4 synthetic path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("Lane 4 synthetic path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_upstreams: bool = True) -> None:
    expected = {
        "schema",
        "package_id",
        "version",
        "status",
        "claim_class",
        "empirical_authority",
        "experiment_id",
        "suite_seed",
        "output_directory",
        "parameter_schema_path",
        "source_families",
        "analytic_design",
        "noise",
        "scoring",
        "executable_formulas",
        "nonexecutable_formulas",
        "formula_aliases",
        "units_and_boundaries",
        "upstream_bindings",
        "source_blocks",
        "access_contract",
    }
    _require(set(config) == expected, "Lane 4 synthetic config keys changed")
    _require(
        config["schema"]
        == "open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-1.0"
        and config["package_id"]
        == "open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v1"
        and config["version"] == "v1.0.0",
        "Lane 4 synthetic identity changed",
    )
    _require(
        config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE"
        and config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL"
        and config["empirical_authority"] == "NONE",
        "Lane 4 synthetic claim boundary changed",
    )
    _require(tuple(sorted(config["executable_formulas"])) == _EXECUTABLE, "formula set changed")
    _require(len(config["nonexecutable_formulas"]) == 10, "blocked inventory changed")
    _require(
        sorted(config["source_families"]) == ["camels", "tng100"],
        "source family set changed",
    )
    _require(
        len(config["source_families"]["camels"]["cadence_scale_factors"]) == 15
        and len(config["source_families"]["tng100"]["cadence_age_gyr"]) == 8,
        "public cadence shapes changed",
    )
    design = config["analytic_design"]
    _require(
        design["mass_ratio"] == [0.25, 0.75]
        and design["pericenter_proxy"] == [0.8, 1.6]
        and sorted(design["history_initial_memory"]) == ["activated", "quiet"]
        and design["roles"] == ["dmo", "hydro"],
        "analytic population changed",
    )
    _require(design["integration_dt"] > 0 and design["encounter_time_span"] > 0, "bad grid")
    _require(
        config["noise"]["families"] == ["analytic-diagonal", "zero-draw"]
        and config["noise"]["single_draw_per_response_vector"] is True
        and config["noise"]["response_calibrated"] is False,
        "noise boundary changed",
    )
    _require(
        config["scoring"]["metric"] == "joint_diagonal_whitened_rmse"
        and config["scoring"]["no_hand_ranking"] is True,
        "scoring contract changed",
    )
    _require(
        _repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve()
        and _repo_path(config["parameter_schema_path"])
        == (_ROOT / PARAMETER_SCHEMA_PATH).resolve(),
        "output or parameter path changed",
    )
    _require(all(value == 0 for value in config["access_contract"].values()), "response access")
    statuses = Counter(row["status"] for row in config["nonexecutable_formulas"].values())
    _require(statuses == {"SOURCE_BLOCKED": 2, "UNADAPTED": 8}, "block status changed")
    if verify_upstreams:
        for row in config["upstream_bindings"]:
            path = _repo_path(row["path"])
            _require(path.is_file(), f"missing upstream: {row['path']}")
            _require(_file_sha256(path) == row["sha256"], f"upstream drift: {row['role']}")
        for source in config["source_families"].values():
            _require(
                _file_sha256(_repo_path(source["source_manifest_path"]))
                == source["source_manifest_sha256"],
                "source manifest drift",
            )


def _metadata(
    element_id: str,
) -> tuple[str, tuple[str, ...], tuple[int, int, int, int, int, int, int], int]:
    scalar = ("object",)
    time = ("cadence",)
    dimensionless = (0, 0, 0, 0, 0, 0, 0)
    length = (0, 1, 0, 0, 0, 0, 0)
    time_dimension = (0, 0, 1, 0, 0, 0, 0)
    energy = (1, 2, -2, 0, 0, 0, 0)
    entropy = (1, 2, -2, 0, -1, 0, 0)
    if element_id.endswith(".entropy"):
        return "encounter entropy", time, entropy, 0
    if element_id.endswith(("receiver-energy", "visible-pair-energy")):
        return "encounter energy", time, energy, 0
    if element_id.endswith(".separation"):
        return "encounter length", time, length, 0
    if element_id in {
        "source.scalar.activation-scale",
        "source.scalar.impact-parameter",
        "source.scalar.initial-separation",
        "source.scalar.pericenter-proxy",
    }:
        return "encounter length", scalar, length, 0
    if element_id in {
        "source.scalar.relaxation-time",
        "source.vector.cadence-interval-lower",
        "source.vector.cadence-interval-upper",
        "source.vector.encounter-time",
    }:
        return "encounter time", time if ".vector." in element_id else scalar, time_dimension, 0
    if element_id == "source.scalar.initial-radial-velocity":
        return "encounter velocity", scalar, (0, 1, -1, 0, 0, 0, 0), 0
    if element_id == "source.scalar.total-mass":
        return "encounter mass", scalar, (1, 0, 0, 0, 0, 0, 0), 0
    if element_id == "source.scalar.temperature":
        return "encounter temperature", scalar, (0, 0, 0, 0, 1, 0, 0), 0
    if element_id == "truth.scalar.formula-code":
        return "integer code", scalar, dimensionless, 0
    if element_id == "source.vector.cadence-coordinate":
        return "normalized public cadence coordinate", time, dimensionless, 0
    return "1", scalar, dimensionless, 0


def _catalogue(config: Mapping[str, Any]):
    experiment = config["experiment_id"]
    config_hash = _json_sha256(config)
    cadence_hash = canonical_sha256(
        {key: value["source_manifest_sha256"] for key, value in config["source_families"].items()}
    )
    derivation_hash = canonical_sha256(
        {"rule": "normalize source cadence to finite encounter time and midpoint intervals"}
    )
    elements: list[DataElement] = []
    for element_id in _SOURCE_FEATURES:
        unit, axes, dimension, rank = _metadata(element_id)
        derived = element_id in {
            "source.vector.cadence-interval-lower",
            "source.vector.cadence-interval-upper",
            "source.vector.encounter-time",
        }
        parents = (
            ("source.vector.cadence-coordinate",)
            if element_id == "source.vector.encounter-time"
            else (("source.vector.encounter-time",) if derived else ())
        )
        elements.append(
            DataElement(
                element_id,
                "source",
                element_id.replace("source.", "").replace("-", " "),
                rank,
                dimension,
                unit,
                _FRAME,
                "per-synthetic-pair-cadence",
                axes,
                "scalar",
                parents,
                UncertaintyKind.CENSORING if "interval" in element_id else UncertaintyKind.NONE,
                Availability.PUBLIC_SOURCE
                if element_id == "source.vector.cadence-coordinate"
                else Availability.ANALYTIC,
                (
                    ExperimentRole(
                        experiment, DataRole.SOURCE_DERIVED if derived else DataRole.FORMULA_INPUT
                    ),
                ),
                cadence_hash if ".vector.cadence" in element_id else config_hash,
                derivation_hash if derived else None,
            )
        )
    for prediction in _PREDICTIONS:
        unit, axes, dimension, rank = _metadata(prediction)
        response = _RESPONSES[prediction]
        elements.extend(
            [
                DataElement(
                    prediction,
                    "prediction",
                    prediction.replace("prediction.", "").replace("-", " "),
                    rank,
                    dimension,
                    unit,
                    _FRAME,
                    "per-synthetic-pair-cadence",
                    axes,
                    "scalar",
                    (),
                    UncertaintyKind.NONE,
                    Availability.SYNTHETIC_ONLY,
                    (ExperimentRole(experiment, DataRole.DERIVED),),
                    _PREDECESSOR_MODULE_SHA256,
                ),
                DataElement(
                    response,
                    "response",
                    response.replace("response.", "").replace("-", " "),
                    rank,
                    dimension,
                    unit,
                    _FRAME,
                    "per-synthetic-pair-cadence",
                    axes,
                    "scalar",
                    (),
                    UncertaintyKind.PER_VALUE,
                    Availability.SYNTHETIC_ONLY,
                    (ExperimentRole(experiment, DataRole.SCORING_ONLY_RESPONSE),),
                    config_hash,
                ),
            ]
        )
    unit, axes, dimension, rank = _metadata("truth.scalar.formula-code")
    elements.append(
        DataElement(
            "truth.scalar.formula-code",
            "truth",
            "hidden injected formula code",
            rank,
            dimension,
            unit,
            "latent",
            "per-synthetic-scenario",
            axes,
            "scalar",
            (),
            UncertaintyKind.NONE,
            Availability.SYNTHETIC_ONLY,
            (ExperimentRole(experiment, DataRole.LATENT_SYNTHETIC_TRUTH),),
            config_hash,
        )
    )
    return catalogue_from_elements(
        "capture.hydro-dmo-source-shaped-synthetic.v1", "v1.0.0", elements
    )


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    schema_hash = _file_sha256(_repo_path(config["parameter_schema_path"]))
    entrypoint_prefix = (
        "sigma_theory_compiler."
        "open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1:"
    )
    rows: list[FormulaExecutionBinding] = []
    for formula_id in _EXECUTABLE:
        rows.append(
            FormulaExecutionBinding(
                f"binding.{formula_id}",
                formula_id,
                "v2.0.0",
                _PREDECESSOR_MODULE_SHA256,
                BindingStatus.EXECUTABLE,
                entrypoint_prefix + _ENTRYPOINTS[formula_id],
                _DYNAMIC_FEATURES,
                _INACTIVE_SOURCE_FEATURES,
                _PREDICTIONS,
                (_DOMAIN,),
                (_GEOMETRY,),
                (_TIME_MODE,),
                config["parameter_schema_path"],
                schema_hash,
                "Finite planar encounter ODE in dimensionless G=1 units; no cosmological lift.",
                (
                    "causal-current-state-rhs",
                    "finite-state-and-positive-separation",
                    "state-derived-energy-ledger",
                ),
                ResourceBounds(20, 268435456, 8388608),
            )
        )
    status_map = {
        "SOURCE_BLOCKED": BindingStatus.SOURCE_BLOCKED,
        "UNADAPTED": BindingStatus.UNADAPTED,
    }
    for formula_id, block in config["nonexecutable_formulas"].items():
        rows.append(
            FormulaExecutionBinding(
                f"binding.{formula_id}",
                formula_id,
                "v1.0.0",
                canonical_sha256({"formula_id": formula_id, **block}),
                status_map[block["status"]],
                None,
                (),
                (),
                (),
                (_DOMAIN,),
                (_GEOMETRY,),
                (_TIME_MODE,),
                config["parameter_schema_path"],
                schema_hash,
                block["reason"],
                ("explicit-nonexecution",),
                ResourceBounds(1, 1048576, 1048576),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.binding_id))


def _adapters(bindings: Sequence[FormulaExecutionBinding]) -> tuple[AdapterRegistration, ...]:
    rows = tuple(
        AdapterRegistration.create(f"adapter.{row.formula_id}", row)
        for row in bindings
        if row.status is BindingStatus.EXECUTABLE
    )
    rows = tuple(sorted(rows, key=lambda row: row.adapter_id))
    validate_adapter_registry(rows)
    return rows


def _parameter_cells(
    config: Mapping[str, Any], bindings: Sequence[FormulaExecutionBinding]
) -> dict[str, tuple[ParameterCell, ...]]:
    return {
        binding.binding_id: (
            (
                ParameterCell(
                    f"cell.{binding.formula_id.lower()}",
                    config["executable_formulas"][binding.formula_id],
                ),
            )
            if binding.status is BindingStatus.EXECUTABLE
            else ()
        )
        for binding in bindings
    }


def _scalar(features: Mapping[str, Any], element_id: str) -> float:
    array = np.asarray(features[element_id], dtype=np.float64)
    if array.shape != (1,) or not np.all(np.isfinite(array)):
        raise SchemaViolation(f"{element_id} must be one finite scalar")
    return float(array[0])


def _parameter_number(value: Any) -> float:
    if not isinstance(value, str):
        raise SchemaViolation("adapter parameter must use exact hexadecimal float text")
    result = float.fromhex(value)
    if not math.isfinite(result):
        raise SchemaViolation("adapter parameter is nonfinite")
    return result


def _validate_adapter_features(features: Mapping[str, Any]) -> tuple[float, ...]:
    _require(set(features) == set(_SOURCE_FEATURES), "adapter feature projection changed")
    times = np.asarray(features["source.vector.encounter-time"], dtype=np.float64)
    lower = np.asarray(features["source.vector.cadence-interval-lower"], dtype=np.float64)
    upper = np.asarray(features["source.vector.cadence-interval-upper"], dtype=np.float64)
    coordinate = np.asarray(features["source.vector.cadence-coordinate"], dtype=np.float64)
    _require(
        times.ndim == 1
        and len(times) >= 2
        and times.shape == lower.shape == upper.shape == coordinate.shape
        and np.all(np.isfinite(times))
        and np.all(np.diff(times) > 0),
        "cadence arrays changed",
    )
    _require(np.all(lower <= times) and np.all(times <= upper), "cadence censoring invalid")
    _require(float(times[0]) == 0.0 and float(coordinate[0]) == 0.0, "cadence origin changed")
    for element_id in _SOURCE_FEATURES:
        if ".scalar." in element_id:
            _scalar(features, element_id)
    return tuple(float(value) for value in times)


def _two_body_derivative(
    state: np.ndarray,
    *,
    m1: float,
    m2: float,
    activation_scale: float,
    temperature: float,
    force_scale: float,
    gamma: float,
    mode: str,
    tau: tuple[float, ...],
    weights: tuple[float, ...],
    compression_speed_scale: float,
) -> np.ndarray:
    r, radial_momentum, angular_momentum, _receiver, _entropy, *memory = state
    if r <= 0.05 or not np.all(np.isfinite(state)):
        raise FloatingPointError("nonpositive or nonfinite two-body state")
    mu = m1 * m2 / (m1 + m2)
    radial_velocity = radial_momentum / mu
    activation = math.exp(-((r / activation_scale) ** 2))
    if mode in {"DC00", "DC01"}:
        effective_memory = 0.0
    elif mode in {"DC05", "DC06"}:
        effective_memory = sum(weight * value for weight, value in zip(weights, memory))
    elif mode == "DC07":
        effective_memory = max(0.0, -radial_velocity / compression_speed_scale) * activation
    else:
        raise SchemaViolation("unknown two-body mechanism")
    heat_rate = gamma * effective_memory * radial_velocity * radial_velocity
    derivative = np.asarray(
        [
            radial_velocity,
            angular_momentum * angular_momentum / (mu * r**3)
            - force_scale * m1 * m2 / r**2
            - gamma * effective_memory * radial_velocity,
            0.0,
            heat_rate,
            heat_rate / temperature,
            *((activation - value) / scale for value, scale in zip(memory, tau)),
        ],
        dtype=np.float64,
    )
    return derivative


def _rk4_two_body(state: np.ndarray, dt: float, **kwargs: Any) -> np.ndarray:
    k1 = _two_body_derivative(state, **kwargs)
    k2 = _two_body_derivative(state + 0.5 * dt * k1, **kwargs)
    k3 = _two_body_derivative(state + 0.5 * dt * k2, **kwargs)
    k4 = _two_body_derivative(state + dt * k3, **kwargs)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _visible_pair_energy(state: np.ndarray, *, m1: float, m2: float, force_scale: float) -> float:
    r, radial_momentum, angular_momentum = state[:3]
    mu = m1 * m2 / (m1 + m2)
    return float(
        radial_momentum**2 / (2.0 * mu)
        + angular_momentum**2 / (2.0 * mu * r**2)
        - force_scale * m1 * m2 / r
    )


@lru_cache(maxsize=4096)
def _simulate_two_body_cached(
    mode: str,
    times: tuple[float, ...],
    mass_ratio: float,
    total_mass: float,
    initial_separation: float,
    initial_radial_velocity: float,
    impact_parameter: float,
    activation_scale: float,
    temperature: float,
    history_memory: float,
    dt: float,
    force_scale: float,
    gamma: float,
    tau: tuple[float, ...],
    weights: tuple[float, ...],
    compression_speed_scale: float,
) -> tuple[tuple[float, ...], ...]:
    _require(0 < mass_ratio <= 1 and total_mass > 0, "mass controls outside domain")
    _require(initial_separation > 0.05 and activation_scale > 0, "length control invalid")
    _require(initial_radial_velocity < 0 and impact_parameter > 0, "encounter control invalid")
    _require(temperature > 0 and dt > 0, "temperature or time step invalid")
    _require(gamma >= 0 and all(value > 0 for value in tau), "dissipation control invalid")
    _require(
        len(tau) == len(weights) and (not tau or math.isclose(sum(weights), 1.0)),
        "memory invalid",
    )
    m1 = total_mass / (1.0 + mass_ratio)
    m2 = total_mass - m1
    mu = m1 * m2 / total_mass
    state = np.asarray(
        [
            initial_separation,
            mu * initial_radial_velocity,
            mu * impact_parameter * abs(initial_radial_velocity),
            0.0,
            0.0,
            *(history_memory for _ in tau),
        ],
        dtype=np.float64,
    )
    derivative_kwargs = {
        "m1": m1,
        "m2": m2,
        "activation_scale": activation_scale,
        "temperature": temperature,
        "force_scale": force_scale,
        "gamma": gamma,
        "mode": mode,
        "tau": tau,
        "weights": weights,
        "compression_speed_scale": compression_speed_scale,
    }
    energy: list[float] = []
    separation: list[float] = []
    receiver: list[float] = []
    entropy: list[float] = []
    current_time = 0.0
    for target in times:
        while current_time + 1.0e-14 < target:
            step = min(dt, target - current_time)
            state = _rk4_two_body(state, step, **derivative_kwargs)
            current_time += step
        energy.append(_visible_pair_energy(state, m1=m1, m2=m2, force_scale=force_scale))
        separation.append(float(state[0]))
        receiver.append(float(state[3]))
        entropy.append(float(state[4]))
    return tuple(map(tuple, (entropy, receiver, separation, energy)))


def _two_body_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    times = _validate_adapter_features(features)
    mode = str(parameters.get("mechanism"))
    _require(mode in {"DC00", "DC01", "DC05", "DC06", "DC07"}, "parameter branch")
    tau = tuple(_parameter_number(value) for value in parameters.get("tau", ()))
    weights = tuple(_parameter_number(value) for value in parameters.get("weights", ()))
    if not tau:
        weights = ()
    values = _simulate_two_body_cached(
        mode,
        times,
        _scalar(features, "source.scalar.mass-ratio"),
        _scalar(features, "source.scalar.total-mass"),
        _scalar(features, "source.scalar.initial-separation"),
        _scalar(features, "source.scalar.initial-radial-velocity"),
        _scalar(features, "source.scalar.impact-parameter"),
        _scalar(features, "source.scalar.activation-scale"),
        _scalar(features, "source.scalar.temperature"),
        _scalar(features, "source.scalar.history-memory"),
        0.01,
        _parameter_number(parameters["force_scale"]),
        _parameter_number(parameters["gamma"]),
        tau,
        weights,
        _parameter_number(parameters.get("compression_speed_scale", "0x1.0000000000000p+0")),
    )
    return {
        prediction: np.asarray(value, dtype=np.float64)
        for prediction, value in zip(_PREDICTIONS, values, strict=True)
    }


def dc00_newtonian_focusing_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _two_body_adapter(features, parameters)


def dc01_static_force_amplification_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _two_body_adapter(features, parameters)


def dc05_single_memory_bath_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _two_body_adapter(features, parameters)


def dc06_bimodal_memory_bath_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _two_body_adapter(features, parameters)


def dc07_compression_gated_bath_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _two_body_adapter(features, parameters)


def _three_body_acceleration(
    positions: np.ndarray, masses: np.ndarray, softening: float
) -> np.ndarray:
    acceleration = np.zeros_like(positions)
    for first in range(3):
        for second in range(first + 1, 3):
            displacement = positions[second] - positions[first]
            denominator = (float(displacement @ displacement) + softening**2) ** 1.5
            acceleration[first] += masses[second] * displacement / denominator
            acceleration[second] -= masses[first] * displacement / denominator
    return acceleration


def _three_body_total_energy(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, softening: float
) -> float:
    energy = 0.5 * float(np.sum(masses[:, None] * velocities * velocities))
    for first in range(3):
        for second in range(first + 1, 3):
            displacement = positions[first] - positions[second]
            energy -= (
                masses[first]
                * masses[second]
                / math.sqrt(float(displacement @ displacement) + softening**2)
            )
    return energy


def _three_body_pair_observables(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float,
    total_energy: float,
) -> tuple[float, float, float]:
    displacement = positions[0] - positions[1]
    relative_velocity = velocities[0] - velocities[1]
    separation = math.sqrt(float(displacement @ displacement) + softening**2)
    reduced_mass = masses[0] * masses[1] / (masses[0] + masses[1])
    pair_energy = 0.5 * reduced_mass * float(relative_velocity @ relative_velocity) - (
        masses[0] * masses[1] / separation
    )
    return pair_energy, separation, total_energy - pair_energy


@lru_cache(maxsize=4096)
def _simulate_three_body_cached(
    times: tuple[float, ...],
    mass_ratio: float,
    total_mass: float,
    initial_separation: float,
    initial_radial_velocity: float,
    impact_parameter: float,
    dt: float,
    softening: float,
    third_mass_fraction: float,
) -> tuple[tuple[float, ...], ...]:
    _require(0 < mass_ratio <= 1 and total_mass > 0, "three-body mass controls invalid")
    _require(initial_separation > 0 and initial_radial_velocity < 0, "three-body encounter invalid")
    _require(dt > 0 and softening > 0 and third_mass_fraction > 0, "three-body grid invalid")
    m1 = total_mass / (1.0 + mass_ratio)
    m2 = total_mass - m1
    m3 = third_mass_fraction * total_mass
    masses = np.asarray([m1, m2, m3], dtype=np.float64)
    pair_mass = m1 + m2
    positions = np.asarray(
        [
            [-initial_separation * m2 / pair_mass, 0.0],
            [initial_separation * m1 / pair_mass, 0.0],
            [0.25 * initial_separation, -1.5 * initial_separation],
        ],
        dtype=np.float64,
    )
    transverse = impact_parameter * abs(initial_radial_velocity) / initial_separation
    relative_velocity = np.asarray([-initial_radial_velocity, transverse], dtype=np.float64)
    velocities = np.asarray(
        [
            relative_velocity * (m2 / pair_mass),
            -relative_velocity * (m1 / pair_mass),
            [0.2 * abs(initial_radial_velocity), 0.3 * abs(initial_radial_velocity)],
        ],
        dtype=np.float64,
    )
    velocities -= np.sum(masses[:, None] * velocities, axis=0) / float(np.sum(masses))
    initial_total = _three_body_total_energy(positions, velocities, masses, softening)
    acceleration = _three_body_acceleration(positions, masses, softening)
    energy: list[float] = []
    separation: list[float] = []
    receiver: list[float] = []
    entropy: list[float] = []
    current_time = 0.0
    for target in times:
        while current_time + 1.0e-14 < target:
            step = min(dt, target - current_time)
            velocities += 0.5 * step * acceleration
            positions += step * velocities
            acceleration = _three_body_acceleration(positions, masses, softening)
            velocities += 0.5 * step * acceleration
            current_time += step
        pair_energy, pair_separation, complement = _three_body_pair_observables(
            positions, velocities, masses, softening, initial_total
        )
        energy.append(pair_energy)
        separation.append(pair_separation)
        receiver.append(complement)
        entropy.append(0.0)
    return tuple(map(tuple, (entropy, receiver, separation, energy)))


def cm01_conservative_three_body_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    times = _validate_adapter_features(features)
    _require(parameters.get("mechanism") == "CM01", "three-body parameter branch changed")
    values = _simulate_three_body_cached(
        times,
        _scalar(features, "source.scalar.mass-ratio"),
        _scalar(features, "source.scalar.total-mass"),
        _scalar(features, "source.scalar.initial-separation"),
        _scalar(features, "source.scalar.initial-radial-velocity"),
        _scalar(features, "source.scalar.impact-parameter"),
        0.01,
        _parameter_number(parameters["softening"]),
        _parameter_number(parameters["third_mass_fraction"]),
    )
    return {
        prediction: np.asarray(value, dtype=np.float64)
        for prediction, value in zip(_PREDICTIONS, values, strict=True)
    }


def _cadence_arrays(source: Mapping[str, Any], span: float) -> tuple[np.ndarray, ...]:
    raw = np.asarray(
        source.get("cadence_scale_factors", source.get("cadence_age_gyr")), dtype=np.float64
    )
    coordinate = (raw - raw[0]) / (raw[-1] - raw[0])
    times = span * coordinate
    edges = np.empty(len(times) + 1, dtype=np.float64)
    edges[0] = times[0]
    edges[-1] = times[-1]
    edges[1:-1] = 0.5 * (times[:-1] + times[1:])
    return coordinate, times, edges[:-1], edges[1:]


def _feature_ref(element_id: str, value: np.ndarray) -> FeatureValueRef:
    unit, axes, _dimension, _rank = _metadata(element_id)
    return FeatureValueRef(
        element_id,
        VALUES_PATH.as_posix(),
        array_sha256(value),
        value.dtype.name,
        value.shape,
        axes,
        unit,
        _FRAME,
    )


def _prediction_spec(element_id: str, length: int) -> EmittedPredictionSpec:
    return EmittedPredictionSpec(
        element_id,
        VALUES_PATH.as_posix(),
        "float64",
        (length,),
        ("cadence",),
        _UNITS[element_id],
        _FRAME,
    )


def _response_ref(element_id: str, value: np.ndarray) -> FeatureValueRef:
    prediction = next(key for key, response in _RESPONSES.items() if response == element_id)
    return FeatureValueRef(
        element_id,
        VALUES_PATH.as_posix(),
        array_sha256(value),
        value.dtype.name,
        value.shape,
        ("cadence",),
        _UNITS[prediction],
        _FRAME,
    )


def _array_key(kind: str, scenario_id: str, element_id: str) -> str:
    return "__".join(
        [kind, scenario_id.replace(".", "_"), element_id.replace(".", "_").replace("-", "_")]
    )


def _design_features(
    config: Mapping[str, Any],
    source_id: str,
    mass_index: int,
    pericenter_index: int,
    history_id: str,
    role: str,
) -> dict[str, np.ndarray]:
    design = config["analytic_design"]
    coordinate, times, lower, upper = _cadence_arrays(
        config["source_families"][source_id], float(design["encounter_time_span"])
    )
    mass_ratio = float(design["mass_ratio"][mass_index])
    pericenter = float(design["pericenter_proxy"][pericenter_index])
    hydro = role == "hydro"
    scalar = lambda value: np.asarray([value], dtype=np.float64)
    return {
        "source.scalar.activation-scale": scalar(design["activation_scale"]),
        "source.scalar.cooling-control": scalar(
            design["hydro_cooling_control_by_pericenter"][pericenter_index] if hydro else 0.0
        ),
        "source.scalar.gas-fraction": scalar(
            design["hydro_gas_fraction_by_mass_ratio"][mass_index] if hydro else 0.0
        ),
        "source.scalar.history-memory": scalar(design["history_initial_memory"][history_id]),
        "source.scalar.impact-parameter": scalar(1.25 + pericenter),
        "source.scalar.initial-radial-velocity": scalar(design["initial_radial_velocity"]),
        "source.scalar.initial-separation": scalar(design["initial_separation"]),
        "source.scalar.mass-ratio": scalar(mass_ratio),
        "source.scalar.pericenter-proxy": scalar(pericenter),
        "source.scalar.relaxation-time": scalar(
            design["relaxation_time_by_mass_ratio"][mass_index]
        ),
        "source.scalar.role-code": scalar(1.0 if hydro else 0.0),
        "source.scalar.shock-mach-control": scalar(
            design["hydro_shock_mach_control_by_pericenter"][pericenter_index] if hydro else 0.0
        ),
        "source.scalar.temperature": scalar(design["temperature"]),
        "source.scalar.total-mass": scalar(design["total_mass"]),
        "source.scalar.wake-coulomb-log": scalar(
            design["wake_coulomb_log_by_mass_ratio"][mass_index]
        ),
        "source.vector.cadence-coordinate": coordinate.astype(np.float64),
        "source.vector.cadence-interval-lower": lower.astype(np.float64),
        "source.vector.cadence-interval-upper": upper.astype(np.float64),
        "source.vector.encounter-time": times.astype(np.float64),
    }


def _truth_prediction(
    formula_id: str,
    features: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    function = globals()[_ENTRYPOINTS[formula_id]]
    result = function(features, parameters)
    return {key: np.asarray(result[key], dtype=np.float64) for key in _PREDICTIONS}


def _anchors(config: Mapping[str, Any]) -> tuple[AnchorBinding, ...]:
    selected_roles = {
        "lane4-v2-receipt",
        "runner-v2-independent-audit",
    }
    rows = [
        AnchorBinding(f"anchor.{row['role']}", row["path"], row["sha256"])
        for row in config["upstream_bindings"]
        if row["role"] in selected_roles
    ]
    rows.extend(
        AnchorBinding(
            f"anchor.source-{source_id}",
            source["source_manifest_path"],
            source["source_manifest_sha256"],
        )
        for source_id, source in config["source_families"].items()
    )
    return tuple(sorted(rows, key=lambda row: row.anchor_id))


def _scenario_population(
    config: Mapping[str, Any],
) -> tuple[
    tuple[ScenarioDescriptor, ...],
    dict[str, ScenarioRuntimeValues],
    dict[str, str],
    dict[str, tuple[ObservableComparison, ...]],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    scenarios: list[ScenarioDescriptor] = []
    runtime: dict[str, ScenarioRuntimeValues] = {}
    truths: dict[str, str] = {}
    comparisons: dict[str, tuple[ObservableComparison, ...]] = {}
    stored: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    anchors = _anchors(config)
    noise = config["noise"]
    sequence = 0
    for source_id in sorted(config["source_families"]):
        for mass_index, mass_ratio in enumerate(config["analytic_design"]["mass_ratio"]):
            for pericenter_index, pericenter in enumerate(
                config["analytic_design"]["pericenter_proxy"]
            ):
                for history_id in sorted(config["analytic_design"]["history_initial_memory"]):
                    pair_id = (
                        f"{source_id}-q{int(100 * mass_ratio):03d}-p{int(10 * pericenter):02d}"
                        f"-{history_id}"
                    )
                    for role in config["analytic_design"]["roles"]:
                        features = _design_features(
                            config,
                            source_id,
                            mass_index,
                            pericenter_index,
                            history_id,
                            role,
                        )
                        length = len(features["source.vector.encounter-time"])
                        for truth_formula in _EXECUTABLE:
                            truth_prediction = _truth_prediction(
                                truth_formula,
                                features,
                                config["executable_formulas"][truth_formula],
                            )
                            for noise_index, noise_family in enumerate(noise["families"]):
                                object_id = f"pair-{pair_id}-{role}"
                                scenario_id = (
                                    f"capture.{pair_id}.{role}.{truth_formula.lower()}."
                                    f"{noise_family}.v1"
                                )
                                lineage = SeedLineage(
                                    int(config["suite_seed"]),
                                    scenario_id,
                                    object_id,
                                    f"truth.{truth_formula.lower()}",
                                    noise_index,
                                    0,
                                )
                                rng = np.random.default_rng(lineage.derived_seed)
                                response_values: dict[str, np.ndarray] = {}
                                uncertainty_values: dict[str, np.ndarray] = {}
                                response_refs: list[FeatureValueRef] = []
                                uncertainty_refs: list[UncertaintyRef] = []
                                for prediction in _PREDICTIONS:
                                    sigma = float(noise[_SIGMA_KEY[prediction]])
                                    draw = (
                                        np.zeros(length, dtype=np.float64)
                                        if noise_family == "zero-draw"
                                        else rng.normal(0.0, sigma, size=length).astype(np.float64)
                                    )
                                    response_id = _RESPONSES[prediction]
                                    response = truth_prediction[prediction] + draw
                                    variance = np.full(length, sigma * sigma, dtype=np.float64)
                                    uncertainty_id = f"uncertainty.{response_id}"
                                    response_values[response_id] = response
                                    uncertainty_values[uncertainty_id] = variance
                                    response_refs.append(_response_ref(response_id, response))
                                    uncertainty_refs.append(
                                        UncertaintyRef(
                                            uncertainty_id,
                                            response_id,
                                            "diagonal-covariance",
                                            VALUES_PATH.as_posix(),
                                            array_sha256(variance),
                                        )
                                    )
                                    stored[_array_key("response", scenario_id, response_id)] = (
                                        response
                                    )
                                    stored[_array_key("variance", scenario_id, prediction)] = (
                                        variance
                                    )
                                truth_value = np.asarray(
                                    [_MECHANISM_CODE[truth_formula]], dtype=np.int64
                                )
                                truth_ref = FeatureValueRef(
                                    "truth.scalar.formula-code",
                                    VALUES_PATH.as_posix(),
                                    array_sha256(truth_value),
                                    "int64",
                                    (1,),
                                    ("object",),
                                    "integer code",
                                    "latent",
                                )
                                descriptor = ScenarioDescriptor(
                                    scenario_id,
                                    object_id,
                                    config["experiment_id"],
                                    _DOMAIN,
                                    _GEOMETRY,
                                    _TIME_MODE,
                                    _FRAME,
                                    (
                                        AxisSpec(
                                            "cadence",
                                            length,
                                            "source.vector.encounter-time",
                                            array_sha256(features["source.vector.encounter-time"]),
                                        ),
                                        AxisSpec("object", 1, None, None),
                                    ),
                                    tuple(
                                        sorted(
                                            (
                                                _feature_ref(key, value)
                                                for key, value in features.items()
                                            ),
                                            key=lambda row: row.element_id,
                                        )
                                    ),
                                    tuple(sorted(response_refs, key=lambda row: row.element_id)),
                                    (truth_ref,),
                                    tuple(_prediction_spec(key, length) for key in _PREDICTIONS),
                                    tuple(
                                        sorted(uncertainty_refs, key=lambda row: row.uncertainty_id)
                                    ),
                                    anchors,
                                    lineage,
                                )
                                for key, value in features.items():
                                    stored.setdefault(
                                        _array_key("feature", scenario_id, key), value
                                    )
                                stored[_array_key("truth", scenario_id, "formula-code")] = (
                                    truth_value
                                )
                                scenarios.append(descriptor)
                                runtime[scenario_id] = ScenarioRuntimeValues(
                                    features,
                                    response_values,
                                    {"truth.scalar.formula-code": truth_value},
                                    uncertainty_values,
                                )
                                truths[scenario_id] = truth_formula
                                comparisons[scenario_id] = tuple(
                                    ObservableComparison(
                                        prediction,
                                        _RESPONSES[prediction],
                                        f"uncertainty.{_RESPONSES[prediction]}",
                                    )
                                    for prediction in _PREDICTIONS
                                )
                                records.append(
                                    {
                                        "scenario": descriptor.to_dict(),
                                        "scenario_sha256": descriptor.content_sha256,
                                        "source_family": source_id,
                                        "synthetic_pair_id": pair_id,
                                        "pair_role": role,
                                        "mass_ratio": mass_ratio,
                                        "pericenter_proxy": pericenter,
                                        "history": history_id,
                                        "truth_formula_id": truth_formula,
                                        "noise_family": noise_family,
                                        "noise_draws_per_response_vector": int(
                                            noise_family != "zero-draw"
                                        ),
                                        "public_object_id_claimed": False,
                                        "real_response_used": False,
                                    }
                                )
                                sequence += 1
    _require(sequence == 384, "scenario Cartesian population changed")
    ordered = tuple(sorted(scenarios, key=lambda row: row.scenario_id))
    records.sort(key=lambda row: row["scenario"]["scenario_id"])
    return ordered, runtime, truths, comparisons, stored, records


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return target.getvalue()


def _invariant_diagnostics(config: Mapping[str, Any]) -> dict[str, Any]:
    base = _design_features(config, "camels", 0, 0, "quiet", "hydro")
    times = tuple(float(value) for value in base["source.vector.encounter-time"])
    common = (
        times,
        0.25,
        5.0,
        8.0,
        -1.2,
        2.05,
        2.5,
        1.0,
        0.0,
        0.01,
    )
    newton = _simulate_two_body_cached("DC00", *common, 1.0, 0.0, (), (), 1.0)
    static_limit = _simulate_two_body_cached("DC01", *common, 1.0, 0.0, (), (), 1.0)
    single_limit = _simulate_two_body_cached("DC05", *common, 1.0, 0.0, (3.0,), (1.0,), 1.0)
    compression_limit = _simulate_two_body_cached("DC07", *common, 1.0, 0.0, (), (), 1.0)
    limit_residual = max(
        float(np.max(np.abs(np.asarray(candidate) - np.asarray(newton))))
        for candidate in (static_limit, single_limit, compression_limit)
    )

    conservation: dict[str, float] = {}
    for formula_id in _EXECUTABLE:
        prediction = _truth_prediction(formula_id, base, config["executable_formulas"][formula_id])
        if formula_id == "CM01_CONSERVATIVE_THREE_BODY_CAPTURE":
            total = (
                prediction["prediction.vector.visible-pair-energy"]
                + prediction["prediction.vector.receiver-energy"]
            )
        else:
            total = (
                prediction["prediction.vector.visible-pair-energy"]
                + prediction["prediction.vector.receiver-energy"]
            )
        conservation[formula_id] = float(np.max(np.abs(total - total[0])))

    dmo = _design_features(config, "camels", 0, 0, "quiet", "dmo")
    hydro = _design_features(config, "camels", 0, 0, "quiet", "hydro")
    inactive_residual = 0.0
    for formula_id in _EXECUTABLE:
        left = _truth_prediction(formula_id, dmo, config["executable_formulas"][formula_id])
        right = _truth_prediction(formula_id, hydro, config["executable_formulas"][formula_id])
        inactive_residual = max(
            inactive_residual,
            *(float(np.max(np.abs(left[key] - right[key]))) for key in _PREDICTIONS),
        )
    return {
        "state_derived_max_total_energy_residual_by_formula": conservation,
        "maximum_zero_gamma_or_unit_force_limit_residual": limit_residual,
        "maximum_inactive_hydro_dmo_role_feature_residual": inactive_residual,
        "inactive_formula_features": list(_INACTIVE_SOURCE_FEATURES),
        "future_cadence_or_response_accessed_by_rhs": False,
        "real_response_accessed": False,
        "ordinary_control_aliases": config["formula_aliases"],
    }


def _diagnostics(
    config: Mapping[str, Any], matrix: Any, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    record_by_id = {row["scenario"]["scenario_id"]: row for row in records}
    status_counts = Counter(cell.discovery_status for cell in matrix.cells)
    eligibility_counts = Counter(cell.eligibility for cell in matrix.cells)
    by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for cell in matrix.cells:
        by_truth[cell.truth_formula_id][cell.discovery_status] += 1
        metadata = record_by_id[cell.scenario_id]
        by_role[metadata["pair_role"]][cell.discovery_status] += 1
        by_source[metadata["source_family"]][cell.discovery_status] += 1
    invariant = _invariant_diagnostics(config)
    return {
        "schema": "open-gravity-hydro-dmo-capture-clumping-synthetic-diagnostics-1.0",
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "empirical_authority": "NONE",
        "counts": {
            "scenarios": matrix.scenario_count,
            "attempted_cells": matrix.attempted_cell_count,
            "scored_cells": matrix.scored_cell_count,
            "blocked_or_unadapted_cells": sum(
                value
                for key, value in eligibility_counts.items()
                if key in {"SOURCE_BLOCKED", "UNADAPTED"}
            ),
            "truth_recovered": matrix.truth_recovery_count,
            "truth_distinctly_recovered": matrix.distinct_truth_recovery_count,
        },
        "discovery_status_counts": dict(sorted(status_counts.items())),
        "eligibility_counts": dict(sorted(eligibility_counts.items())),
        "by_truth": {key: dict(sorted(value.items())) for key, value in sorted(by_truth.items())},
        "by_pair_role": {
            key: dict(sorted(value.items())) for key, value in sorted(by_role.items())
        },
        "by_source_family": {
            key: dict(sorted(value.items())) for key, value in sorted(by_source.items())
        },
        "cross_hydro_dmo_identifiability": {
            "synthetic_pair_roles_only": True,
            "public_object_level_matching_claimed": False,
            "inactive_role_field_prediction_residual": invariant[
                "maximum_inactive_hydro_dmo_role_feature_residual"
            ],
            "interpretation": (
                "The current executable laws do not consume gas/cooling/shock/wake/relaxation "
                "controls, so paired hydro/DMO role changes alone are intentionally non-identifying."
            ),
        },
        "invariants_and_limits": invariant,
        "explicit_blocks": config["nonexecutable_formulas"],
        "source_blocks": config["source_blocks"],
        "access_contract": config["access_contract"],
    }


def _confusion(matrix: Any) -> dict[str, Any]:
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for cell in matrix.cells:
        if cell.winner:
            rows[cell.truth_formula_id][cell.formula_id] += 1
    return {
        "schema": "open-gravity-synthetic-confusion-matrix-1.0",
        "truth_rows": {key: dict(sorted(value.items())) for key, value in sorted(rows.items())},
        "scenario_count": matrix.scenario_count,
        "truth_recovery_count": matrix.truth_recovery_count,
        "distinct_truth_recovery_count": matrix.distinct_truth_recovery_count,
        "empirical_authority": "NONE",
    }


def derive_release() -> tuple[dict[str, Any], dict[str, bytes]]:
    config = load_config()
    validate_config(config)
    catalogue = _catalogue(config)
    bindings = _bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    adapters = _adapters(bindings)
    parameter_cells = _parameter_cells(config, bindings)
    scenarios, runtime, truths, comparisons, arrays, records = _scenario_population(config)
    release = SyntheticSuiteRelease(
        config["package_id"],
        config["version"],
        canonical_sha256(
            {
                "config": _json_sha256(config),
                "source_manifests": {
                    key: value["source_manifest_sha256"]
                    for key, value in config["source_families"].items()
                },
            }
        ),
        catalogue.content_sha256,
        _file_sha256(Path(__file__)),
        _json_sha256(config["noise"]),
        tuple(sorted(_SOURCE_FEATURES)),
        "MAJOR",
        False,
        True,
    )
    matrix = run_discovery_matrix_v2(
        catalogue=catalogue,
        release=release,
        scenarios=scenarios,
        scenario_values=runtime,
        truth_formula_by_scenario=truths,
        bindings=bindings,
        adapters=adapters,
        parameter_cells=parameter_cells,
        comparisons=comparisons,
        distinct_gap=float(config["scoring"]["minimum_whitened_gap_for_distinct_signature"]),
        ledger_id="capture.hydro-dmo-source-shaped-synthetic.v1",
    )
    diagnostics = _diagnostics(config, matrix, records)
    artifacts = {
        VALUES_PATH.as_posix(): _npz_bytes(arrays),
        SCENARIOS_PATH.as_posix(): b"".join(_json_bytes(row) + b"\n" for row in records),
        MATRIX_PATH.as_posix(): _json_bytes(matrix.to_dict(), indent=2),
        LEDGER_PATH.as_posix(): _json_bytes(matrix.ledger.to_dict(), indent=2),
        CONFUSION_PATH.as_posix(): _json_bytes(_confusion(matrix), indent=2),
        DIAGNOSTICS_PATH.as_posix(): _json_bytes(diagnostics, indent=2),
    }
    result = {
        "config": config,
        "catalogue": catalogue,
        "bindings": bindings,
        "adapters": adapters,
        "parameter_cells": parameter_cells,
        "scenarios": scenarios,
        "runtime": runtime,
        "truths": truths,
        "comparisons": comparisons,
        "release": release,
        "matrix": matrix,
        "diagnostics": diagnostics,
        "records": records,
    }
    return result, artifacts


def _receipt(result: Mapping[str, Any], artifacts: Mapping[str, bytes]) -> dict[str, Any]:
    config = result["config"]
    matrix = result["matrix"]
    body = {
        "schema": "open-gravity-hydro-dmo-capture-clumping-synthetic-receipt-1.0",
        "package_id": config["package_id"],
        "status": config["status"],
        "claim_class": config["claim_class"],
        "empirical_authority": "NONE",
        "subject": {
            "config_path": CONFIG_PATH.as_posix(),
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "parameter_schema_path": PARAMETER_SCHEMA_PATH.as_posix(),
            "parameter_schema_raw_sha256": _file_sha256(_ROOT / PARAMETER_SCHEMA_PATH),
            "module_path": Path(__file__).resolve().relative_to(_ROOT).as_posix(),
            "module_raw_sha256": _file_sha256(Path(__file__)),
            "test_path": TEST_PATH.as_posix(),
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
        },
        "upstream_bindings": config["upstream_bindings"],
        "source_manifest_bindings": {
            key: {
                "path": value["source_manifest_path"],
                "sha256": value["source_manifest_sha256"],
            }
            for key, value in config["source_families"].items()
        },
        "formula_inventory": {
            "executable": list(_EXECUTABLE),
            "nonexecutable": config["nonexecutable_formulas"],
            "aliases": config["formula_aliases"],
        },
        "matrix_counts": {
            "scenarios": matrix.scenario_count,
            "attempted_cells": matrix.attempted_cell_count,
            "scored_cells": matrix.scored_cell_count,
            "retained_nonexecutable_cells": matrix.attempted_cell_count - matrix.scored_cell_count,
            "truth_recovered": matrix.truth_recovery_count,
            "truth_distinctly_recovered": matrix.distinct_truth_recovery_count,
            "replay_entries": len(matrix.ledger.entries),
        },
        "artifacts": [
            {
                "path": path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(artifacts.items())
        ],
        "invariants_and_limits": result["diagnostics"]["invariants_and_limits"],
        "source_blocks": config["source_blocks"],
        "access_contract": config["access_contract"],
        "claim_boundary": {
            "synthetic_only": True,
            "public_metadata_source_shaped": True,
            "simulation_payload_decoded": False,
            "real_hydro_dmo_pair_tested": False,
            "real_response_scored": False,
            "empirical_support_or_rejection": False,
        },
        "decision": "FREEZE_FOR_DISTINCT_INDEPENDENT_AUDIT_BEFORE_ANY_EMPIRICAL_PROMOTION",
    }
    return {**body, "content_sha256": _json_sha256(body)}


def _write_once(path: Path, payload: bytes) -> str:
    absolute = _ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        _require(absolute.read_bytes() == payload, f"append-only artifact drift: {path}")
        return hashlib.sha256(payload).hexdigest()
    descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def freeze() -> str:
    result, artifacts = derive_release()
    for path, payload in sorted(artifacts.items()):
        _write_once(Path(path), payload)
    receipt = _receipt(result, artifacts)
    return _write_once(RECEIPT_PATH, _json_bytes(receipt, indent=2))


def check() -> dict[str, Any]:
    result, artifacts = derive_release()
    for path, expected in artifacts.items():
        absolute = _repo_path(path)
        _require(absolute.is_file(), f"missing frozen artifact: {path}")
        _require(absolute.read_bytes() == expected, f"frozen artifact drift: {path}")
    expected_receipt = _json_bytes(_receipt(result, artifacts), indent=2)
    _require((_ROOT / RECEIPT_PATH).read_bytes() == expected_receipt, "receipt drift")
    return json.loads(expected_receipt)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(argv if argv is not None else ())
    if arguments == ("--check",):
        check()
    elif arguments in {(), ("--freeze",)}:
        freeze()
    else:
        raise SystemExit("usage: module [--freeze|--check]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(os.sys.argv[1:])))


__all__ = [
    "check",
    "cm01_conservative_three_body_adapter",
    "dc00_newtonian_focusing_adapter",
    "dc01_static_force_amplification_adapter",
    "dc05_single_memory_bath_adapter",
    "dc06_bimodal_memory_bath_adapter",
    "dc07_compression_gated_bath_adapter",
    "derive_release",
    "freeze",
    "load_config",
    "validate_config",
]
