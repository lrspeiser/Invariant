"""State-derived energy repair for the Lane 4 hydro/DMO synthetic matrix."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1 as base,
)
from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import catalogue_from_elements
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
    "configs/open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v2.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v2.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v2"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
MATRIX_PATH = OUTPUT_DIR / "matrix-result.json"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "invariance-identifiability-and-blocks.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]

_EXECUTABLE = base._EXECUTABLE
_PREDICTIONS = base._PREDICTIONS
_RESPONSES = base._RESPONSES
_SOURCE_FEATURES = base._SOURCE_FEATURES
_DYNAMIC_FEATURES = base._DYNAMIC_FEATURES
_INACTIVE_SOURCE_FEATURES = base._INACTIVE_SOURCE_FEATURES
_UNITS = base._UNITS
_SIGMA_KEY = base._SIGMA_KEY
_MECHANISM_CODE = base._MECHANISM_CODE
_DOMAIN = base._DOMAIN
_GEOMETRY = base._GEOMETRY
_TIME_MODE = base._TIME_MODE
_FRAME = base._FRAME
_ENTRYPOINTS = {
    "CM01_CONSERVATIVE_THREE_BODY_CAPTURE": "cm01_conservative_three_body_adapter",
    "DC00_NEWTONIAN_FOCUSING_CONTROL": "dc00_newtonian_focusing_adapter",
    "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL": "dc01_static_force_amplification_adapter",
    "DC05_TIMEWELL_SINGLE_MEMORY_BATH": "dc05_single_memory_bath_adapter",
    "DC06_TIMEWELL_BIMODAL_MEMORY_BATH": "dc06_bimodal_memory_bath_adapter",
    "DC07_COMPRESSION_GATED_BATH": "dc07_compression_gated_bath_adapter",
}


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
        raise SchemaViolation("Lane 4 v2 path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("Lane 4 v2 path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def _base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = json.loads(
        _repo_path(config["predecessor"]["config_path"]).read_text(encoding="utf-8")
    )
    predecessor.update(
        {
            "package_id": config["package_id"],
            "version": config["version"],
            "experiment_id": config["experiment_id"],
            "suite_seed": config["suite_seed"],
            "output_directory": config["output_directory"],
            "access_contract": config["access_contract"],
        }
    )
    return predecessor


def validate_config(config: Mapping[str, Any], *, verify_upstreams: bool = True) -> None:
    _require(
        set(config)
        == {
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
            "predecessor",
            "blocked_audit",
            "repair",
            "expected_unchanged",
            "access_contract",
        },
        "Lane 4 v2 config keys changed",
    )
    _require(
        config["schema"]
        == "open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-2.0"
        and config["package_id"]
        == "open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v2"
        and config["version"] == "v1.1.0",
        "Lane 4 v2 identity changed",
    )
    _require(
        config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE"
        and config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL"
        and config["empirical_authority"] == "NONE",
        "Lane 4 v2 claim boundary changed",
    )
    repair = config["repair"]
    _require(
        repair["cm01_receiver_energy"]
        == "current three-body Hamiltonian minus current visible-pair energy"
        and repair["cm01_integration_dt"] == 0.0004
        and repair["two_body_integration_dt"] == 0.005
        and repair["full_grid_energy_tolerance"] == 0.000005
        and repair["retain_changed_recovery_without_tuning"] is True,
        "Lane 4 v2 repair contract changed",
    )
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")
    _require(
        _repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(),
        "v2 output changed",
    )
    if not verify_upstreams:
        return
    predecessor = config["predecessor"]
    for prefix in ("config", "parameter_schema", "module", "test", "receipt"):
        _require(
            _file_sha256(_repo_path(predecessor[f"{prefix}_path"]))
            == predecessor[f"{prefix}_raw_sha256"],
            f"v1 {prefix} drift",
        )
    receipt = json.loads(_repo_path(predecessor["receipt_path"]).read_text(encoding="utf-8"))
    _require(
        receipt["content_sha256"] == predecessor["receipt_content_sha256"],
        "v1 receipt content drift",
    )
    for name, expected in predecessor["artifact_sha256"].items():
        path = _repo_path(
            f"runs/gravity/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v1/{name}"
        )
        _require(_file_sha256(path) == expected, f"v1 artifact drift: {name}")
    audit = config["blocked_audit"]
    _require(audit["status"] == "BLOCK", "v1 audit disposition changed")
    _require(audit["raw_sha256"] != "PENDING_AUDITOR_FINALIZATION", "v1 audit not finalized")
    _require(_file_sha256(_repo_path(audit["path"])) == audit["raw_sha256"], "v1 audit drift")
    predecessor_config = json.loads(
        _repo_path(predecessor["config_path"]).read_text(encoding="utf-8")
    )
    base.validate_config(predecessor_config, verify_upstreams=True)


def _catalogue(config: Mapping[str, Any]):
    inherited = base._catalogue(_base_config(config))
    return catalogue_from_elements(
        "capture.hydro-dmo-source-shaped-synthetic.v2", "v1.1.0", inherited.elements
    )


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    inherited = _base_config(config)
    schema_hash = _file_sha256(_repo_path(config["parameter_schema_path"]))
    prefix = (
        "sigma_theory_compiler."
        "open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v2:"
    )
    rows: list[FormulaExecutionBinding] = []
    for formula_id in _EXECUTABLE:
        rows.append(
            FormulaExecutionBinding(
                f"binding.{formula_id}",
                formula_id,
                "v2.1.0",
                base._PREDECESSOR_MODULE_SHA256,
                BindingStatus.EXECUTABLE,
                prefix + _ENTRYPOINTS[formula_id],
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
                    "current-state-derived-energy-ledger",
                    "finite-state-and-positive-separation",
                ),
                ResourceBounds(30, 268435456, 8388608),
            )
        )
    status_map = {
        "SOURCE_BLOCKED": BindingStatus.SOURCE_BLOCKED,
        "UNADAPTED": BindingStatus.UNADAPTED,
    }
    for formula_id, block in inherited["nonexecutable_formulas"].items():
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
        sorted(
            (
                AdapterRegistration.create(f"adapter.{row.formula_id}.v2", row)
                for row in bindings
                if row.status is BindingStatus.EXECUTABLE
            ),
            key=lambda row: row.adapter_id,
        )
    )
    validate_adapter_registry(rows)
    return rows


def _parameter_cells(
    config: Mapping[str, Any], bindings: Sequence[FormulaExecutionBinding]
) -> dict[str, tuple[ParameterCell, ...]]:
    inherited = _base_config(config)
    return {
        binding.binding_id: (
            (
                ParameterCell(
                    f"cell.{binding.formula_id.lower()}",
                    inherited["executable_formulas"][binding.formula_id],
                ),
            )
            if binding.status is BindingStatus.EXECUTABLE
            else ()
        )
        for binding in bindings
    }


def _two_body_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    times = base._validate_adapter_features(features)
    mode = str(parameters.get("mechanism"))
    _require(mode in {"DC00", "DC01", "DC05", "DC06", "DC07"}, "parameter branch")
    tau = tuple(base._parameter_number(value) for value in parameters.get("tau", ()))
    weights = tuple(base._parameter_number(value) for value in parameters.get("weights", ()))
    if not tau:
        weights = ()
    values = base._simulate_two_body_cached(
        mode,
        times,
        base._scalar(features, "source.scalar.mass-ratio"),
        base._scalar(features, "source.scalar.total-mass"),
        base._scalar(features, "source.scalar.initial-separation"),
        base._scalar(features, "source.scalar.initial-radial-velocity"),
        base._scalar(features, "source.scalar.impact-parameter"),
        base._scalar(features, "source.scalar.activation-scale"),
        base._scalar(features, "source.scalar.temperature"),
        base._scalar(features, "source.scalar.history-memory"),
        0.005,
        base._parameter_number(parameters["force_scale"]),
        base._parameter_number(parameters["gamma"]),
        tau,
        weights,
        base._parameter_number(parameters.get("compression_speed_scale", "0x1.0000000000000p+0")),
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
    acceleration = base._three_body_acceleration(positions, masses, softening)
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
            acceleration = base._three_body_acceleration(positions, masses, softening)
            velocities += 0.5 * step * acceleration
            current_time += step
        current_total = base._three_body_total_energy(positions, velocities, masses, softening)
        pair_energy, pair_separation, _old_complement = base._three_body_pair_observables(
            positions, velocities, masses, softening, current_total
        )
        energy.append(pair_energy)
        separation.append(pair_separation)
        receiver.append(current_total - pair_energy)
        entropy.append(0.0)
    return tuple(map(tuple, (entropy, receiver, separation, energy)))


def cm01_conservative_three_body_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    times = base._validate_adapter_features(features)
    _require(parameters.get("mechanism") == "CM01", "three-body parameter branch changed")
    values = _simulate_three_body_cached(
        times,
        base._scalar(features, "source.scalar.mass-ratio"),
        base._scalar(features, "source.scalar.total-mass"),
        base._scalar(features, "source.scalar.initial-separation"),
        base._scalar(features, "source.scalar.initial-radial-velocity"),
        base._scalar(features, "source.scalar.impact-parameter"),
        0.0004,
        base._parameter_number(parameters["softening"]),
        base._parameter_number(parameters["third_mass_fraction"]),
    )
    return {
        prediction: np.asarray(value, dtype=np.float64)
        for prediction, value in zip(_PREDICTIONS, values, strict=True)
    }


def _truth_prediction(
    formula_id: str,
    features: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    function = globals()[_ENTRYPOINTS[formula_id]]
    result = function(features, parameters)
    return {key: np.asarray(result[key], dtype=np.float64) for key in _PREDICTIONS}


def _feature_ref(element_id: str, value: np.ndarray) -> FeatureValueRef:
    unit, axes, _dimension, _rank = base._metadata(element_id)
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


def _anchors(config: Mapping[str, Any]) -> tuple[AnchorBinding, ...]:
    inherited = _base_config(config)
    selected_roles = {"lane4-v2-receipt", "runner-v2-independent-audit"}
    rows = [
        AnchorBinding(f"anchor.{row['role']}", row["path"], row["sha256"])
        for row in inherited["upstream_bindings"]
        if row["role"] in selected_roles
    ]
    rows.extend(
        AnchorBinding(
            f"anchor.source-{source_id}",
            source["source_manifest_path"],
            source["source_manifest_sha256"],
        )
        for source_id, source in inherited["source_families"].items()
    )
    rows.extend(
        [
            AnchorBinding(
                "anchor.synthetic-v1-receipt",
                config["predecessor"]["receipt_path"],
                config["predecessor"]["receipt_raw_sha256"],
            ),
            AnchorBinding(
                "anchor.synthetic-v1-blocked-audit",
                config["blocked_audit"]["path"],
                config["blocked_audit"]["raw_sha256"],
            ),
        ]
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
    inherited = _base_config(config)
    scenarios: list[ScenarioDescriptor] = []
    runtime: dict[str, ScenarioRuntimeValues] = {}
    truths: dict[str, str] = {}
    comparisons: dict[str, tuple[ObservableComparison, ...]] = {}
    stored: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    anchors = _anchors(config)
    noise = inherited["noise"]
    sequence = 0
    for source_id in sorted(inherited["source_families"]):
        for mass_index, mass_ratio in enumerate(inherited["analytic_design"]["mass_ratio"]):
            for pericenter_index, pericenter in enumerate(
                inherited["analytic_design"]["pericenter_proxy"]
            ):
                for history_id in sorted(inherited["analytic_design"]["history_initial_memory"]):
                    pair_id = (
                        f"{source_id}-q{int(100 * mass_ratio):03d}-p{int(10 * pericenter):02d}"
                        f"-{history_id}"
                    )
                    for role in inherited["analytic_design"]["roles"]:
                        features = base._design_features(
                            inherited,
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
                                inherited["executable_formulas"][truth_formula],
                            )
                            for noise_index, noise_family in enumerate(noise["families"]):
                                object_id = f"pair-{pair_id}-{role}"
                                scenario_id = (
                                    f"capture.{pair_id}.{role}.{truth_formula.lower()}."
                                    f"{noise_family}.v2"
                                )
                                lineage = SeedLineage(
                                    int(config["suite_seed"]),
                                    scenario_id,
                                    object_id,
                                    f"truth.{truth_formula.lower()}.v2",
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
                                    stored[_array_key("feature", scenario_id, key)] = value
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
                                        "v1_recovery_status_not_presumed": True,
                                    }
                                )
                                sequence += 1
    _require(sequence == config["repair"]["expected_scenarios"], "scenario count changed")
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
    inherited = _base_config(config)
    maximum_by_formula = {formula_id: 0.0 for formula_id in _EXECUTABLE}
    entropy_identity_max = 0.0
    role_residual = 0.0
    scenario_profiles = 0
    for source_id in sorted(inherited["source_families"]):
        for mass_index in range(2):
            for pericenter_index in range(2):
                for history_id in ("activated", "quiet"):
                    predictions_by_role: dict[str, dict[str, dict[str, np.ndarray]]] = {}
                    for role in ("dmo", "hydro"):
                        features = base._design_features(
                            inherited,
                            source_id,
                            mass_index,
                            pericenter_index,
                            history_id,
                            role,
                        )
                        predictions_by_role[role] = {}
                        for formula_id in _EXECUTABLE:
                            prediction = _truth_prediction(
                                formula_id,
                                features,
                                inherited["executable_formulas"][formula_id],
                            )
                            predictions_by_role[role][formula_id] = prediction
                            total = (
                                prediction["prediction.vector.visible-pair-energy"]
                                + prediction["prediction.vector.receiver-energy"]
                            )
                            maximum_by_formula[formula_id] = max(
                                maximum_by_formula[formula_id],
                                float(np.max(np.abs(total - total[0]))),
                            )
                            if formula_id.startswith(("DC05", "DC06", "DC07")):
                                entropy_identity_max = max(
                                    entropy_identity_max,
                                    float(
                                        np.max(
                                            np.abs(
                                                prediction["prediction.vector.receiver-energy"]
                                                - prediction["prediction.vector.entropy"]
                                            )
                                        )
                                    ),
                                )
                    for formula_id in _EXECUTABLE:
                        for prediction in _PREDICTIONS:
                            role_residual = max(
                                role_residual,
                                float(
                                    np.max(
                                        np.abs(
                                            predictions_by_role["dmo"][formula_id][prediction]
                                            - predictions_by_role["hydro"][formula_id][prediction]
                                        )
                                    )
                                ),
                            )
                    scenario_profiles += 2
    base_features = base._design_features(inherited, "camels", 0, 0, "quiet", "hydro")
    times = tuple(float(value) for value in base_features["source.vector.encounter-time"])
    common = (times, 0.25, 5.0, 8.0, -1.2, 2.05, 2.5, 1.0, 0.0, 0.005)
    newton = base._simulate_two_body_cached("DC00", *common, 1.0, 0.0, (), (), 1.0)
    limits = (
        base._simulate_two_body_cached("DC01", *common, 1.0, 0.0, (), (), 1.0),
        base._simulate_two_body_cached("DC05", *common, 1.0, 0.0, (3.0,), (1.0,), 1.0),
        base._simulate_two_body_cached("DC07", *common, 1.0, 0.0, (), (), 1.0),
    )
    limit_residual = max(
        float(np.max(np.abs(np.asarray(candidate) - np.asarray(newton)))) for candidate in limits
    )
    return {
        "full_source_design_role_profiles_checked": scenario_profiles,
        "state_derived_max_total_energy_residual_by_formula": maximum_by_formula,
        "maximum_state_derived_total_energy_residual": max(maximum_by_formula.values()),
        "full_grid_energy_tolerance": config["repair"]["full_grid_energy_tolerance"],
        "all_full_grid_energy_gates_pass": max(maximum_by_formula.values())
        < config["repair"]["full_grid_energy_tolerance"],
        "maximum_receiver_temperature_entropy_identity_residual": entropy_identity_max,
        "maximum_zero_gamma_or_unit_force_limit_residual": limit_residual,
        "maximum_inactive_hydro_dmo_role_feature_residual": role_residual,
        "cm01_receiver_energy_uses_current_three_body_hamiltonian": True,
        "cm01_receiver_energy_is_initial_total_deficit": False,
        "future_cadence_or_response_accessed_by_rhs": False,
        "real_response_accessed": False,
    }


def _diagnostics(
    config: Mapping[str, Any], matrix: Any, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    inherited = _base_config(config)
    record_by_id = {row["scenario"]["scenario_id"]: row for row in records}
    status_counts = Counter(cell.discovery_status for cell in matrix.cells)
    eligibility_counts = Counter(cell.eligibility for cell in matrix.cells)
    by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    nonrecoveries: set[str] = set()
    nondistinct_recoveries: set[str] = set()
    for cell in matrix.cells:
        by_truth[cell.truth_formula_id][cell.discovery_status] += 1
        if not cell.truth_recovered:
            nonrecoveries.add(cell.scenario_id)
        elif not cell.distinct and cell.eligibility == "ELIGIBLE":
            nondistinct_recoveries.add(cell.scenario_id)
    invariants = _invariant_diagnostics(config)
    return {
        "schema": "open-gravity-hydro-dmo-capture-clumping-synthetic-diagnostics-2.0",
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "empirical_authority": "NONE",
        "repair": config["repair"],
        "counts": {
            "scenarios": matrix.scenario_count,
            "attempted_cells": matrix.attempted_cell_count,
            "scored_cells": matrix.scored_cell_count,
            "source_blocked_cells": eligibility_counts["SOURCE_BLOCKED"],
            "unadapted_cells": eligibility_counts["UNADAPTED"],
            "truth_recovered": matrix.truth_recovery_count,
            "truth_distinctly_recovered": matrix.distinct_truth_recovery_count,
        },
        "discovery_status_counts": dict(sorted(status_counts.items())),
        "eligibility_counts": dict(sorted(eligibility_counts.items())),
        "by_truth": {key: dict(sorted(value.items())) for key, value in sorted(by_truth.items())},
        "nonrecovered_scenarios": [
            {
                "scenario_id": scenario_id,
                "source_family": record_by_id[scenario_id]["source_family"],
                "pair_role": record_by_id[scenario_id]["pair_role"],
                "truth_formula_id": record_by_id[scenario_id]["truth_formula_id"],
                "noise_family": record_by_id[scenario_id]["noise_family"],
            }
            for scenario_id in sorted(nonrecoveries)
        ],
        "truth_recovered_but_nondistinct_scenarios": sorted(nondistinct_recoveries),
        "cross_hydro_dmo_identifiability": {
            "synthetic_pair_roles_only": True,
            "public_object_level_matching_claimed": False,
            "inactive_role_field_prediction_residual": invariants[
                "maximum_inactive_hydro_dmo_role_feature_residual"
            ],
            "interpretation": (
                "The executable laws do not consume gas/cooling/shock/wake/relaxation controls, "
                "so paired role changes alone remain intentionally non-identifying."
            ),
        },
        "invariants_and_limits": invariants,
        "explicit_blocks": inherited["nonexecutable_formulas"],
        "source_blocks": inherited["source_blocks"],
        "access_contract": config["access_contract"],
        "v1_blocked_audit": config["blocked_audit"],
    }


def _confusion(matrix: Any) -> dict[str, Any]:
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for cell in matrix.cells:
        if cell.winner:
            rows[cell.truth_formula_id][cell.formula_id] += 1
    return {
        "schema": "open-gravity-synthetic-confusion-matrix-2.0",
        "truth_rows": {key: dict(sorted(value.items())) for key, value in sorted(rows.items())},
        "scenario_count": matrix.scenario_count,
        "truth_recovery_count": matrix.truth_recovery_count,
        "distinct_truth_recovery_count": matrix.distinct_truth_recovery_count,
        "empirical_authority": "NONE",
    }


def derive_release() -> tuple[dict[str, Any], dict[str, bytes]]:
    config = load_config()
    validate_config(config)
    inherited = _base_config(config)
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
                "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
                "v1_receipt_raw_sha256": config["predecessor"]["receipt_raw_sha256"],
                "v1_blocked_audit_raw_sha256": config["blocked_audit"]["raw_sha256"],
            }
        ),
        catalogue.content_sha256,
        _file_sha256(Path(__file__)),
        _json_sha256(inherited["noise"]),
        tuple(sorted(_PREDICTIONS)),
        "MINOR",
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
        distinct_gap=float(inherited["scoring"]["minimum_whitened_gap_for_distinct_signature"]),
        ledger_id="capture.hydro-dmo-source-shaped-synthetic.v2",
    )
    repair = config["repair"]
    _require(matrix.scenario_count == repair["expected_scenarios"], "scenario count drift")
    _require(matrix.attempted_cell_count == repair["expected_attempted_cells"], "cell count drift")
    _require(matrix.scored_cell_count == repair["expected_scored_cells"], "score count drift")
    eligibility = Counter(cell.eligibility for cell in matrix.cells)
    _require(
        eligibility["SOURCE_BLOCKED"] == repair["expected_source_blocked_cells"]
        and eligibility["UNADAPTED"] == repair["expected_unadapted_cells"],
        "blocked cell count drift",
    )
    _require(len(matrix.ledger.entries) == repair["expected_replay_entries"], "ledger drift")
    diagnostics = _diagnostics(config, matrix, records)
    _require(
        diagnostics["invariants_and_limits"]["all_full_grid_energy_gates_pass"],
        "full-grid state-derived energy gate failed",
    )
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
        "inherited": inherited,
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
    diagnostics = result["diagnostics"]
    body = {
        "schema": "open-gravity-hydro-dmo-capture-clumping-synthetic-receipt-2.0",
        "package_id": config["package_id"],
        "status": config["status"],
        "claim_class": config["claim_class"],
        "empirical_authority": "NONE",
        "subject": {
            "config_path": CONFIG_PATH.as_posix(),
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "parameter_schema_path": config["parameter_schema_path"],
            "parameter_schema_raw_sha256": _file_sha256(
                _repo_path(config["parameter_schema_path"])
            ),
            "module_path": Path(__file__).resolve().relative_to(_ROOT).as_posix(),
            "module_raw_sha256": _file_sha256(Path(__file__)),
            "test_path": TEST_PATH.as_posix(),
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
        },
        "predecessor": config["predecessor"],
        "blocked_audit": config["blocked_audit"],
        "repair": config["repair"],
        "matrix_counts": diagnostics["counts"],
        "replay_entries": len(matrix.ledger.entries),
        "recovery_failures_retained": diagnostics["nonrecovered_scenarios"],
        "artifacts": [
            {
                "path": path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(artifacts.items())
        ],
        "invariants_and_limits": diagnostics["invariants_and_limits"],
        "source_blocks": result["inherited"]["source_blocks"],
        "access_contract": config["access_contract"],
        "claim_boundary": {
            "synthetic_only": True,
            "simulation_payload_decoded": False,
            "real_hydro_dmo_pair_tested": False,
            "real_response_scored": False,
            "empirical_support_or_rejection": False,
        },
        "decision": "FREEZE_REPAIR_FOR_DISTINCT_INDEPENDENT_REAUDIT",
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
    return _write_once(RECEIPT_PATH, _json_bytes(_receipt(result, artifacts), indent=2))


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
