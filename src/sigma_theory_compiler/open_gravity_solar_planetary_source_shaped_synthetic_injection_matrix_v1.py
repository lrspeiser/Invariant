"""Response-blind Solar/planetary real-source-shaped synthetic matrix.

The only physical source values used to generate the synthetic population are
the frozen Lane-6 JPL approximate-element and DE440-mass source contract.  The
official local DE440 kernel is hash-bound as a public source anchor, but no SPK
state is extracted and no observational residual or response is opened.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_lane6_gqns_solar_source_domain_v1 as solar
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
    EligibilityStatus,
    FormulaExecutionBinding,
    ResourceBounds,
    validate_binding_catalogue,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    SyntheticReplayLedger,
    SyntheticSuiteRelease,
    status_from_result,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    AnchorBinding,
    AxisSpec,
    EmittedPredictionSpec,
    FeatureValueRef,
    ScenarioDescriptor,
    UncertaintyRef,
    array_sha256,
    decide_scenario_eligibility,
    execute_binding_in_process,
    validate_scenario_catalogue,
    validate_scenario_values,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path(
    "configs/open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v1.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v1.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-v1"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "invariance-and-identifiability.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]

_TARGETS = ("EARTH", "JUPITER", "MARS", "MERCURY", "NEPTUNE", "SATURN", "URANUS", "VENUS")
_BODY_NAMES = (
    "SUN",
    "MERCURY",
    "VENUS",
    "EMB",
    "EARTH",
    "MOON",
    "MARS",
    "JUPITER",
    "SATURN",
    "URANUS",
    "NEPTUNE",
    *(f"ASTEROID_RING_{index:02d}" for index in range(36)),
)
_DOMAIN_MEMBERS = {
    "D02_SUN_INNER_BARYCENTERS": ("SUN", "MERCURY", "VENUS", "EMB", "MARS"),
    "D05_SUN_EIGHT_PLANETS": (
        "SUN", "MERCURY", "VENUS", "EMB", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"
    ),
    "D06_MOON_SPLIT": (
        "SUN", "MERCURY", "VENUS", "EARTH", "MOON", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"
    ),
    "D07_ASTEROID_RING": (
        "SUN", "MERCURY", "VENUS", "EARTH", "MOON", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE",
        *(f"ASTEROID_RING_{index:02d}" for index in range(36)),
    ),
}
_FORMULA_DOMAIN = {
    "GQNS_GLOBAL_D02_INNER": "D02_SUN_INNER_BARYCENTERS",
    "GQNS_GLOBAL_D05_EIGHT_PLANETS": "D05_SUN_EIGHT_PLANETS",
    "GQNS_GLOBAL_D06_MOON_SPLIT": "D06_MOON_SPLIT",
    "GQNS_GLOBAL_D07_ASTEROID_RING": "D07_ASTEROID_RING",
    "NEWTON_D07_COMPARATOR": "D07_ASTEROID_RING",
}
_ENTRYPOINTS = {
    "GQNS_GLOBAL_D02_INNER": "gqns_global_d02_adapter",
    "GQNS_GLOBAL_D05_EIGHT_PLANETS": "gqns_global_d05_adapter",
    "GQNS_GLOBAL_D06_MOON_SPLIT": "gqns_global_d06_adapter",
    "GQNS_GLOBAL_D07_ASTEROID_RING": "gqns_global_d07_adapter",
    "NEWTON_D07_COMPARATOR": "newton_d07_adapter",
}
_FEATURES = (
    "source.matrix.body-position-au",
    "source.scalar.target-code",
    "source.tensor.body-internal-covariance-au2",
    "source.vector.body-gm-km3-s2",
    "source.vector.epoch-centuries",
    "source.vector.target-heliocentric-radius-au",
    "source.vector.target-position-au",
)
_BLOCK_STATUS = {
    "AQUAL_SIMPLE_MU": BindingStatus.SOURCE_BLOCKED,
    "DPEL01_DISK_POLAR_ESCAPE_LOAD": BindingStatus.UNADAPTED,
    "GP01_ELLIPTIC_N2_L035": BindingStatus.SOURCE_BLOCKED,
    "MASHHOON_RAHVAR_NLG_Q0": BindingStatus.SOURCE_BLOCKED,
    "NFW_SOURCE_MATCHED_CONTROL": BindingStatus.THEORY_ONLY,
    "QUMOND_SIMPLE_NU": BindingStatus.SOURCE_BLOCKED,
    "REFRACTED_GRAVITY_DISKMASS_MEDIAN": BindingStatus.SOURCE_BLOCKED,
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
        raise SchemaViolation("Solar synthetic path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("Solar synthetic path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_upstreams: bool = True) -> None:
    expected = {
        "schema", "package_id", "version", "status", "claim_class", "experiment_id",
        "suite_seed", "targets", "epochs_centuries_from_j2000",
        "refined_epochs_centuries_from_j2000", "mechanisms", "source_domains",
        "noise_families", "noise", "scoring", "geometry_mode", "time_mode",
        "coordinate_frame", "parameter_schema_path", "output_directory",
        "upstream_bindings", "adapter_blocks", "access_contract",
    }
    _require(set(config) == expected, "Solar synthetic config keys changed")
    _require(
        config["schema"] == "open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-1.0",
        "Solar synthetic schema changed",
    )
    _require(
        config["package_id"] == "open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-v1"
        and config["version"] == "v1.0.0",
        "Solar synthetic identity changed",
    )
    _require(config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE", "status changed")
    _require(config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim changed")
    _require(tuple(config["targets"]) == _TARGETS, "target population changed")
    _require(config["mechanisms"] == sorted(_FORMULA_DOMAIN), "mechanisms changed")
    _require(config["source_domains"] == sorted(_DOMAIN_MEMBERS), "source domains changed")
    epochs = np.asarray(config["epochs_centuries_from_j2000"], dtype=float)
    refined = np.asarray(config["refined_epochs_centuries_from_j2000"], dtype=float)
    _require(epochs.shape == (13,) and refined.shape == (25,), "epoch grids changed")
    _require(np.array_equal(refined[::2], epochs), "refinement grid is not nested")
    _require(float(epochs[0]) == 0.0 and float(epochs[-1]) == 0.5, "epoch span changed")
    _require(
        config["noise_families"]
        == ["independent-source-fraction", "orbital-phase-correlated", "zero-noise"],
        "noise families changed",
    )
    _require(
        config["noise"]
        == {
            "fractional_sigma": 0.02,
            "floor_fraction_of_truth_rms": 0.0001,
            "correlation_length_epochs": 2.5,
        },
        "noise contract changed",
    )
    scoring = config["scoring"]
    _require(
        scoring["primary_metric"] == "profiled_whitened_rmse"
        and scoring["winner_absolute_tolerance"] == 1.0e-12
        and scoring["minimum_whitened_gap_for_distinct_signature"] == 0.1
        and scoring["pairwise_degeneracy_relative_rms_max"] == 1.0e-8
        and scoring["no_hand_ranking"] is True,
        "scoring contract changed",
    )
    _require(
        config["geometry_mode"] == "compact-body-point-source"
        and config["time_mode"] == "orbital-phase-static-snapshots",
        "geometry/time contract changed",
    )
    _require(_repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(), "output changed")
    access = config["access_contract"]
    _require(
        access["de440_kernel_bytes_hash_read"] == 119799808
        and access["de440_state_values_extracted"] == 0
        and access["lane6_public_source_config_bytes_read"] == 11017,
        "source access accounting changed",
    )
    _require(
        all(
            access[key] == 0
            for key in (
                "observational_response_files_opened", "observational_response_rows_opened",
                "observational_residual_values_opened", "response_calibrated_parameters",
                "network_calls", "model_calls", "paid_calls",
            )
        ),
        "response-blind access boundary changed",
    )
    _require(len(config["adapter_blocks"]) == len(_BLOCK_STATUS), "adapter blocks changed")
    _require(
        [row["formula_id"] for row in config["adapter_blocks"]] == sorted(_BLOCK_STATUS),
        "adapter block order changed",
    )
    upstream_ids = [row["id"] for row in config["upstream_bindings"]]
    _require(upstream_ids == sorted(set(upstream_ids)), "upstream bindings must be sorted")
    for row in config["upstream_bindings"]:
        _require(set(row) == {"id", "path", "sha256"}, "upstream binding schema changed")
        if verify_upstreams:
            path = _repo_path(row["path"])
            _require(path.is_file() and _file_sha256(path) == row["sha256"], f"upstream changed: {row['id']}")


def _upstream_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {row["id"]: _file_sha256(_repo_path(row["path"])) for row in config["upstream_bindings"]}


def _catalogue(config: Mapping[str, Any]):
    provenance = canonical_sha256(config["upstream_bindings"])
    specs = (
        ("prediction.vector.relative-acceleration", "candidate target-minus-Sun acceleration", 1, "m s^-2", ("epoch", "component")),
        ("response.vector.synthetic-relative-acceleration", "synthetic target-minus-Sun acceleration", 1, "m s^-2", ("epoch", "component")),
        ("source.matrix.body-position-au", "compact-body source positions", 1, "au", ("epoch", "body", "component")),
        ("source.scalar.target-code", "fixed target identity code", 0, "integer code", ("object",)),
        ("source.tensor.body-internal-covariance-au2", "body internal second moments", 2, "au2", ("body", "component", "matrixrow")),
        ("source.vector.body-gm-km3-s2", "source gravitational parameters", 0, "km3 s^-2", ("body",)),
        ("source.vector.epoch-centuries", "Julian centuries from J2000", 0, "century", ("epoch",)),
        ("source.vector.target-heliocentric-radius-au", "instantaneous target orbital radius", 0, "au", ("epoch",)),
        ("source.vector.target-position-au", "target positions", 1, "au", ("epoch", "component")),
        ("truth.scalar.injection-id", "synthetic mechanism identity", 0, "integer code", ("object",)),
    )
    dimensions = {
        "m s^-2": (0, 1, -2, 0, 0, 0, 0), "au": (0, 1, 0, 0, 0, 0, 0),
        "au2": (0, 2, 0, 0, 0, 0, 0), "km3 s^-2": (0, 3, -2, 0, 0, 0, 0),
        "century": (0, 0, 1, 0, 0, 0, 0), "integer code": (0, 0, 0, 0, 0, 0, 0),
    }
    elements = []
    for element_id, quantity, rank, unit, axes in specs:
        if element_id.startswith("response."):
            role, availability = DataRole.SCORING_ONLY_RESPONSE, Availability.SYNTHETIC_ONLY
        elif element_id.startswith("prediction."):
            role, availability = DataRole.DERIVED, Availability.SYNTHETIC_ONLY
        elif element_id.startswith("truth."):
            role, availability = DataRole.LATENT_SYNTHETIC_TRUTH, Availability.SYNTHETIC_ONLY
        else:
            role, availability = DataRole.FORMULA_INPUT, Availability.PUBLIC_SOURCE
        elements.append(
            DataElement(
                element_id=element_id,
                namespace=element_id.rsplit(".", 1)[0],
                physical_quantity=quantity,
                tensor_rank=rank,
                si_dimension=dimensions[unit],
                canonical_unit=unit,
                frame="latent" if element_id.startswith("truth.") else config["coordinate_frame"],
                support="13 Lane-6 public JPL approximate-element snapshots from J2000 through 2050",
                axes=axes,
                component="total",
                derivation_parents=(),
                uncertainty=(UncertaintyKind.COVARIANCE if element_id.startswith(("response.", "prediction.")) else UncertaintyKind.NONE),
                availability=availability,
                experiment_roles=(ExperimentRole(config["experiment_id"], role),),
                provenance_sha256=provenance,
            )
        )
    return catalogue_from_elements("open-gravity-solar-planetary-source-shaped-synthetic", "v1.0.0", elements)


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    upstream = _upstream_hashes(config)
    schema_sha = _file_sha256(_repo_path(config["parameter_schema_path"]))
    rows = []
    for formula_id in sorted((*_FORMULA_DOMAIN, *_BLOCK_STATUS)):
        executable = formula_id in _FORMULA_DOMAIN
        status = BindingStatus.EXECUTABLE if executable else _BLOCK_STATUS[formula_id]
        entrypoint = (
            f"sigma_theory_compiler.open_gravity_solar_planetary_source_shaped_synthetic_injection_matrix_v1:{_ENTRYPOINTS[formula_id]}"
            if executable else None
        )
        formula_sha = canonical_sha256(
            {
                "formula_id": formula_id,
                "lane6_unchanged_law_module_sha256": upstream["LANE6_SOLAR_V1_MODULE"],
                "source_domain": _FORMULA_DOMAIN.get(formula_id),
                "blocked_reason": next((row["reason"] for row in config["adapter_blocks"] if row["formula_id"] == formula_id), None),
            }
        )
        required = _FEATURES if executable else ("source.scalar.mass-density",)
        rows.append(
            FormulaExecutionBinding(
                binding_id=f"binding.solar.{formula_id.lower()}.v1",
                formula_id=formula_id,
                formula_version="v1.0.0-lane6-source-domain-adapter",
                formula_sha256=formula_sha,
                status=status,
                entrypoint=entrypoint,
                required_features=tuple(sorted(required)),
                optional_features=(),
                emitted_features=("prediction.vector.relative-acceleration",),
                domains=(("solar-system",) if executable else ("cluster", "disk-galaxy")),
                geometry_support=((config["geometry_mode"],) if executable else ("density-grid",)),
                time_support=((config["time_mode"],) if executable else ("static",)),
                parameter_schema_path=config["parameter_schema_path"],
                parameter_schema_sha256=schema_sha,
                approximation_ceiling=(
                    "source-only static snapshots; no propagation, orbit fit, relativistic correction, or DE440 state extraction"
                    if executable else next(row["reason"] for row in config["adapter_blocks"] if row["formula_id"] == formula_id)
                ),
                health_gates=("determinism", "finite-output", "rotation", "source-hash", "translation", "typed-output"),
                resource_bounds=ResourceBounds(30, 500_000_000, 1_000_000),
            )
        )
    return tuple(rows)


def _check_feature_shapes(features: Mapping[str, Any]) -> None:
    _require(set(features) == set(_FEATURES), "Solar adapter feature projection changed")
    positions = np.asarray(features["source.matrix.body-position-au"])
    covariances = np.asarray(features["source.tensor.body-internal-covariance-au2"])
    gm = np.asarray(features["source.vector.body-gm-km3-s2"])
    epochs = np.asarray(features["source.vector.epoch-centuries"])
    target_position = np.asarray(features["source.vector.target-position-au"])
    radius = np.asarray(features["source.vector.target-heliocentric-radius-au"])
    code = np.asarray(features["source.scalar.target-code"])
    _require(positions.ndim == 3 and positions.shape[1:] == (len(_BODY_NAMES), 3), "body positions changed")
    _require(covariances.shape == (len(_BODY_NAMES), 3, 3), "body covariances changed")
    _require(gm.shape == (len(_BODY_NAMES),), "body GM changed")
    _require(epochs.shape == (positions.shape[0],), "epoch vector changed")
    _require(target_position.shape == (positions.shape[0], 3), "target position changed")
    _require(radius.shape == (positions.shape[0],), "target radius changed")
    _require(code.shape == (1,) and int(code[0]) in range(len(_TARGETS)), "target code changed")
    sun_position = positions[:, _BODY_NAMES.index("SUN"), :]
    _require(
        np.allclose(
            np.linalg.norm(target_position - sun_position, axis=1),
            radius,
            rtol=0.0,
            atol=1e-12,
        ),
        "target-minus-Sun radius inconsistent",
    )


def _execute_formula(features: Mapping[str, Any], domain_id: str, *, include_gqns: bool) -> Mapping[str, Any]:
    _check_feature_shapes(features)
    positions = np.asarray(features["source.matrix.body-position-au"], dtype=np.float64)
    covariances = np.asarray(features["source.tensor.body-internal-covariance-au2"], dtype=np.float64)
    gm = np.asarray(features["source.vector.body-gm-km3-s2"], dtype=np.float64)
    target_positions = np.asarray(features["source.vector.target-position-au"], dtype=np.float64)
    target = _TARGETS[int(np.asarray(features["source.scalar.target-code"])[0])]
    indices = tuple(_BODY_NAMES.index(name) for name in _DOMAIN_MEMBERS[domain_id])
    config = solar.load_config()
    prediction = np.empty((positions.shape[0], 3), dtype=np.float64)
    for epoch in range(positions.shape[0]):
        bodies = [
            {
                "name": _BODY_NAMES[index],
                "position": positions[epoch, index],
                "gm": float(gm[index]),
                "internal": covariances[index],
            }
            for index in indices
        ]
        target_position = target_positions[epoch]
        sun_position = positions[epoch, _BODY_NAMES.index("SUN")]
        metrics = solar.geometry_metrics(bodies)
        target_newton = solar.acceleration(target, target_position, bodies, anisotropy=None, length_au=None, config=config)
        sun_newton = solar.acceleration("SUN", sun_position, bodies, anisotropy=None, length_au=None, config=config)
        total = target_newton - sun_newton
        if include_gqns:
            target_dark = solar.acceleration(
                target, target_position, bodies,
                anisotropy=float(metrics["A_Q"]), length_au=float(metrics["L_au"]), config=config,
            )
            sun_dark = solar.acceleration(
                "SUN", sun_position, bodies,
                anisotropy=float(metrics["A_Q"]), length_au=float(metrics["L_au"]), config=config,
            )
            total = total + target_dark - sun_dark
        prediction[epoch] = total
    _require(np.all(np.isfinite(prediction)), "Solar adapter produced nonfinite output")
    return {"prediction.vector.relative-acceleration": prediction}


def gqns_global_d02_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "GQNS adapter has no tunable parameters")
    return _execute_formula(features, "D02_SUN_INNER_BARYCENTERS", include_gqns=True)


def gqns_global_d05_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "GQNS adapter has no tunable parameters")
    return _execute_formula(features, "D05_SUN_EIGHT_PLANETS", include_gqns=True)


def gqns_global_d06_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "GQNS adapter has no tunable parameters")
    return _execute_formula(features, "D06_MOON_SPLIT", include_gqns=True)


def gqns_global_d07_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "GQNS adapter has no tunable parameters")
    return _execute_formula(features, "D07_ASTEROID_RING", include_gqns=True)


def newton_d07_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "Newton adapter has no tunable parameters")
    return _execute_formula(features, "D07_ASTEROID_RING", include_gqns=False)


def _source_arrays(config: Mapping[str, Any], epochs: np.ndarray, target: str) -> dict[str, np.ndarray]:
    source_config = solar.load_config()
    positions = np.empty((len(epochs), len(_BODY_NAMES), 3), dtype=np.float64)
    gm = np.empty(len(_BODY_NAMES), dtype=np.float64)
    covariances = np.empty((len(_BODY_NAMES), 3, 3), dtype=np.float64)
    target_positions = np.empty((len(epochs), 3), dtype=np.float64)
    d07 = next(row for row in source_config["source_domains"] if row["id"] == "D07_ASTEROID_RING")
    for epoch_index, centuries in enumerate(epochs):
        state = solar._base_state(source_config, float(centuries))
        bodies = {row["name"]: row for row in solar._domain_bodies(source_config, d07, float(centuries))}
        bodies["EMB"] = {"name": "EMB", **state["EMB"]}
        for body_index, name in enumerate(_BODY_NAMES):
            body = bodies[name]
            positions[epoch_index, body_index] = np.asarray(body["position"], dtype=np.float64)
            if epoch_index == 0:
                gm[body_index] = float(body["gm"])
                covariances[body_index] = np.asarray(body["internal"], dtype=np.float64)
            else:
                _require(gm[body_index] == float(body["gm"]), "source GM changed with epoch")
                _require(np.array_equal(covariances[body_index], np.asarray(body["internal"])), "source covariance changed with epoch")
        target_positions[epoch_index] = np.asarray(state[target]["position"], dtype=np.float64)
    radius = np.linalg.norm(target_positions, axis=1).astype(np.float64)
    return {
        "source.matrix.body-position-au": positions,
        "source.scalar.target-code": np.asarray([_TARGETS.index(target)], dtype=np.int64),
        "source.tensor.body-internal-covariance-au2": covariances,
        "source.vector.body-gm-km3-s2": gm,
        "source.vector.epoch-centuries": np.asarray(epochs, dtype=np.float64),
        "source.vector.target-heliocentric-radius-au": radius,
        "source.vector.target-position-au": target_positions,
    }


def _scenario(
    config: Mapping[str, Any], target: str, scenario_id: str, truth_world_id: str,
    nuisance_draw: int, values: Mapping[str, np.ndarray], response: np.ndarray,
    variance: np.ndarray, truth_index: int,
) -> ScenarioDescriptor:
    units = {
        "source.matrix.body-position-au": "au", "source.scalar.target-code": "integer code",
        "source.tensor.body-internal-covariance-au2": "au2", "source.vector.body-gm-km3-s2": "km3 s^-2",
        "source.vector.epoch-centuries": "century", "source.vector.target-heliocentric-radius-au": "au",
        "source.vector.target-position-au": "au",
    }
    axes = {
        "source.matrix.body-position-au": ("epoch", "body", "component"),
        "source.scalar.target-code": ("object",),
        "source.tensor.body-internal-covariance-au2": ("body", "component", "matrixrow"),
        "source.vector.body-gm-km3-s2": ("body",), "source.vector.epoch-centuries": ("epoch",),
        "source.vector.target-heliocentric-radius-au": ("epoch",),
        "source.vector.target-position-au": ("epoch", "component"),
    }
    frame = config["coordinate_frame"]
    truth = np.asarray([truth_index], dtype=np.int64)
    anchors = {row["id"]: row for row in config["upstream_bindings"]}
    return ScenarioDescriptor(
        scenario_id=scenario_id, object_id=target.lower(), experiment_id=config["experiment_id"],
        domain="solar-system", geometry_mode=config["geometry_mode"], time_mode=config["time_mode"],
        coordinate_frame=frame,
        axes=(
            AxisSpec("body", len(_BODY_NAMES), None, None), AxisSpec("component", 3, None, None),
            AxisSpec("epoch", len(response), "source.vector.epoch-centuries", array_sha256(values["source.vector.epoch-centuries"])),
            AxisSpec("matrixrow", 3, None, None), AxisSpec("object", 1, None, None),
        ),
        formula_features=tuple(
            FeatureValueRef(element_id, VALUES_PATH.as_posix(), array_sha256(values[element_id]), values[element_id].dtype.name,
                            values[element_id].shape, axes[element_id], units[element_id], frame)
            for element_id in _FEATURES
        ),
        scoring_responses=(FeatureValueRef(
            "response.vector.synthetic-relative-acceleration", VALUES_PATH.as_posix(), array_sha256(response),
            "float64", response.shape, ("epoch", "component"), "m s^-2", frame,
        ),),
        hidden_truth=(FeatureValueRef(
            "truth.scalar.injection-id", VALUES_PATH.as_posix(), array_sha256(truth), "int64", truth.shape,
            ("object",), "integer code", "latent",
        ),),
        expected_predictions=(EmittedPredictionSpec(
            "prediction.vector.relative-acceleration", VALUES_PATH.as_posix(), "float64", response.shape,
            ("epoch", "component"), "m s^-2", frame,
        ),),
        uncertainties=(UncertaintyRef(
            "synthetic-relative-acceleration.diagonal-covariance",
            "response.vector.synthetic-relative-acceleration", "diagonal-covariance",
            VALUES_PATH.as_posix(), array_sha256(variance),
        ),),
        anchors=(
            AnchorBinding("anchor.de440-public-kernel", anchors["DE440_PUBLIC_KERNEL"]["path"], anchors["DE440_PUBLIC_KERNEL"]["sha256"]),
            AnchorBinding("anchor.lane6-solar-v2-receipt", anchors["LANE6_SOLAR_V2_RECEIPT"]["path"], anchors["LANE6_SOLAR_V2_RECEIPT"]["sha256"]),
        ),
        seed_lineage=SeedLineage(config["suite_seed"], scenario_id, target.lower(), truth_world_id, nuisance_draw, 0),
    )


def _noise_response(
    truth: np.ndarray, family: str, lineage: SeedLineage, config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rms = float(np.sqrt(np.mean(truth * truth)))
    floor = max(np.finfo(float).tiny, config["noise"]["floor_fraction_of_truth_rms"] * rms)
    sigma = np.maximum(config["noise"]["fractional_sigma"] * np.abs(truth), floor)
    variance = (sigma * sigma).astype(np.float64)
    if family == "zero-noise":
        noise = np.zeros_like(truth)
    else:
        rng = np.random.default_rng(lineage.derived_seed)
        if family == "independent-source-fraction":
            noise = rng.normal(size=truth.shape) * sigma
        elif family == "orbital-phase-correlated":
            index = np.arange(truth.shape[0], dtype=float)
            correlation = np.exp(-np.abs(index[:, None] - index[None, :]) / config["noise"]["correlation_length_epochs"])
            root = np.linalg.cholesky(correlation)
            noise = np.column_stack([root @ rng.normal(size=len(index)) for _ in range(3)]) * sigma
        else:
            raise SchemaViolation("unknown Solar synthetic noise family")
    return (truth + noise).astype(np.float64), variance, {
        "family": family, "seed": lineage.derived_seed, "truth_rms_m_s2": rms,
        "noise_rms_m_s2": float(np.sqrt(np.mean(noise * noise))),
        "variance_representation": "diagonal-marginal",
    }


def _profiled_metrics(candidate: np.ndarray, response: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    weight = 1.0 / variance.reshape(-1)
    x = candidate.reshape(-1)
    y = response.reshape(-1)
    denominator = float(np.dot(weight * x, x))
    _require(math.isfinite(denominator) and denominator > 0.0, "invalid nuisance projection")
    scale = float(np.dot(weight * x, y) / denominator)
    fitted = scale * candidate
    residual = fitted - response
    return {
        "profiled_whitened_rmse": float(np.sqrt(np.mean(residual * residual / variance))),
        "relative_rmse": float(np.linalg.norm(residual) / max(np.linalg.norm(response), np.finfo(float).tiny)),
        "raw_relative_rmse": float(np.linalg.norm(candidate - response) / max(np.linalg.norm(response), np.finfo(float).tiny)),
        "fitted_common_acceleration_scale": scale,
    }


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **{key: arrays[key] for key in sorted(arrays)})
    return stream.getvalue()


def _array_key(*parts: str) -> str:
    return "__".join(part.lower().replace("-", "_").replace(".", "_") for part in parts)


def _invariance_and_identifiability(
    config: Mapping[str, Any], source_items: Mapping[str, Mapping[str, np.ndarray]],
    predictions: Mapping[tuple[str, str], np.ndarray],
) -> dict[str, Any]:
    rotation = solar._rotation_z(0.37) @ solar._rotation_x(-0.22)
    translation = np.asarray((0.123, -0.456, 0.789), dtype=float)
    invariance_rows = []
    refinement_rows = []
    refined_epochs = np.asarray(config["refined_epochs_centuries_from_j2000"], dtype=np.float64)
    pairwise_rows = []
    formula_ids = tuple(config["mechanisms"])
    for target in _TARGETS:
        values = source_items[target]
        translated = {key: np.array(value, copy=True) for key, value in values.items()}
        translated["source.matrix.body-position-au"] += translation
        translated["source.vector.target-position-au"] += translation
        rotated = {key: np.array(value, copy=True) for key, value in values.items()}
        rotated["source.matrix.body-position-au"] = values["source.matrix.body-position-au"] @ rotation.T
        rotated["source.vector.target-position-au"] = values["source.vector.target-position-au"] @ rotation.T
        rotated["source.tensor.body-internal-covariance-au2"] = np.einsum(
            "ij,bjk,lk->bil", rotation, values["source.tensor.body-internal-covariance-au2"], rotation
        )
        refined_values = _source_arrays(config, refined_epochs, target)
        for formula_id in formula_ids:
            domain_id = _FORMULA_DOMAIN[formula_id]
            gqns = formula_id != "NEWTON_D07_COMPARATOR"
            baseline = predictions[(target, formula_id)]
            translated_prediction = _execute_formula(translated, domain_id, include_gqns=gqns)["prediction.vector.relative-acceleration"]
            rotated_prediction = _execute_formula(rotated, domain_id, include_gqns=gqns)["prediction.vector.relative-acceleration"]
            refined_prediction = _execute_formula(refined_values, domain_id, include_gqns=gqns)["prediction.vector.relative-acceleration"]
            invariance_rows.append({
                "target": target, "formula_id": formula_id,
                "translation_max_abs_error_m_s2": float(np.max(np.abs(translated_prediction - baseline))),
                "rotation_max_abs_error_m_s2": float(np.max(np.abs(rotated_prediction - baseline @ rotation.T))),
            })
            refinement_rows.append({
                "target": target, "formula_id": formula_id,
                "nested_epoch_max_abs_error_m_s2": float(np.max(np.abs(refined_prediction[::2] - baseline))),
            })
        for left_index, left in enumerate(formula_ids):
            for right in formula_ids[left_index + 1 :]:
                lhs = predictions[(target, left)]
                rhs = predictions[(target, right)]
                scale = float(np.vdot(rhs, lhs).real / np.vdot(rhs, rhs).real)
                residual = lhs - scale * rhs
                relative = float(np.linalg.norm(residual) / max(np.linalg.norm(lhs), np.finfo(float).tiny))
                pairwise_rows.append({
                    "target": target, "left_formula_id": left, "right_formula_id": right,
                    "profiled_relative_rms": relative,
                    "degenerate_by_frozen_threshold": relative <= config["scoring"]["pairwise_degeneracy_relative_rms_max"],
                })
    source_domain_rows = []
    source_config = solar.load_config()
    domains = {row["id"]: row for row in source_config["source_domains"]}
    for domain_id in config["source_domains"]:
        metrics = [solar.geometry_metrics(solar._domain_bodies(source_config, domains[domain_id], float(epoch))) for epoch in config["epochs_centuries_from_j2000"]]
        source_domain_rows.append({
            "source_domain_id": domain_id,
            "A_Q_min": min(float(row["A_Q"]) for row in metrics), "A_Q_max": max(float(row["A_Q"]) for row in metrics),
            "L_au_min": min(float(row["L_au"]) for row in metrics), "L_au_max": max(float(row["L_au"]) for row in metrics),
        })
    radii = [
        {"target": target, "minimum_heliocentric_radius_au": float(np.min(source_items[target]["source.vector.target-heliocentric-radius-au"])),
         "maximum_heliocentric_radius_au": float(np.max(source_items[target]["source.vector.target-heliocentric-radius-au"]))}
        for target in _TARGETS
    ]
    return {
        "schema": "open-gravity-solar-planetary-invariance-identifiability-1.0",
        "translation_rotation": invariance_rows, "nested_epoch_refinement": refinement_rows,
        "pairwise_profiled_identifiability": pairwise_rows, "source_domain_moment_ranges": source_domain_rows,
        "planet_orbital_radius_ranges": radii,
        "maximum_translation_error_m_s2": max(row["translation_max_abs_error_m_s2"] for row in invariance_rows),
        "maximum_rotation_error_m_s2": max(row["rotation_max_abs_error_m_s2"] for row in invariance_rows),
        "maximum_nested_refinement_error_m_s2": max(row["nested_epoch_max_abs_error_m_s2"] for row in refinement_rows),
        "degenerate_pair_count": sum(row["degenerate_by_frozen_threshold"] for row in pairwise_rows),
        "pair_count": len(pairwise_rows),
    }


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes]:
    config = load_config()
    validate_config(config)
    lane2_receipt = json.loads(_repo_path(next(row["path"] for row in config["upstream_bindings"] if row["id"] == "LANE6_SOLAR_V2_RECEIPT")).read_text(encoding="utf-8"))
    _require(lane2_receipt["observational_decision"] == "NOT_EVALUATED__MATCHED_N_BODY_REFIT_REQUIRED", "Lane6 response boundary changed")
    _require(all(value == 0 for value in lane2_receipt["access_contract"].values()), "Lane6 receipt opened a response")
    catalogue = _catalogue(config)
    bindings = _bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    executable = tuple(row for row in bindings if row.status is BindingStatus.EXECUTABLE)
    registrations = tuple(AdapterRegistration.create(f"adapter.solar.{row.formula_id.lower()}.v1", row) for row in executable)
    validate_adapter_registry(registrations)
    registration_by_formula = {row.formula_binding.formula_id: row for row in registrations}
    module_sha = _file_sha256(Path(__file__))
    release = SyntheticSuiteRelease(
        suite_id=config["package_id"], version=config["version"],
        release_sha256=canonical_sha256({
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH), "generator_raw_sha256": module_sha,
            "lane6_source_receipt_sha256": next(row["sha256"] for row in config["upstream_bindings"] if row["id"] == "LANE6_SOLAR_V2_RECEIPT"),
            "de440_public_kernel_sha256": next(row["sha256"] for row in config["upstream_bindings"] if row["id"] == "DE440_PUBLIC_KERNEL"),
        }),
        ontology_sha256=catalogue.content_sha256, generator_sha256=module_sha,
        observation_operator_sha256=_json_sha256({"noise": config["noise"], "noise_families": config["noise_families"], "scoring": config["scoring"]}),
        changed_feature_ids=("prediction.vector.relative-acceleration", "response.vector.synthetic-relative-acceleration", "truth.scalar.injection-id"),
        change_level="MAJOR", response_calibrated=False, prediction_semantics_changed=True,
    )
    epochs = np.asarray(config["epochs_centuries_from_j2000"], dtype=np.float64)
    source_items = {target: _source_arrays(config, epochs, target) for target in _TARGETS}
    arrays: dict[str, np.ndarray] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    numerical_failures = []
    for target, values in source_items.items():
        for feature_id, value in values.items():
            arrays[_array_key("source", target, feature_id)] = value
        scaffold_response = np.zeros((len(epochs), 3), dtype=np.float64)
        scaffold_variance = np.ones_like(scaffold_response)
        scaffold = _scenario(config, target, f"solar.{target.lower()}.execution-scaffold.v1", "truth.execution-scaffold", 0, values, scaffold_response, scaffold_variance, 0)
        validate_scenario_catalogue(scaffold, catalogue)
        for binding in executable:
            decision = decide_scenario_eligibility(binding, catalogue, scaffold)
            _require(decision.status is EligibilityStatus.ELIGIBLE, "executable Solar binding became ineligible")
            try:
                result = execute_binding_in_process(binding, catalogue, scaffold, {key: values[key] for key in binding.required_features}, {})
                prediction = np.asarray(result.output_values["prediction.vector.relative-acceleration"])
                value_key = _array_key("candidate", target, binding.formula_id)
                arrays[value_key] = prediction
                predictions[(target, binding.formula_id)] = prediction
                cache[(target, binding.formula_id)] = {
                    "success": True, "prediction": prediction, "value_key": value_key,
                    "value_sha256": array_sha256(prediction), "output_sha256": result.output_sha256,
                    "scaffold_scenario_sha256": scaffold.content_sha256,
                }
            except Exception as error:
                failure = {"type": type(error).__name__, "message": str(error)}
                failure_sha = _json_sha256(failure)
                cache[(target, binding.formula_id)] = {"success": False, "failure": failure, "output_sha256": failure_sha, "scaffold_scenario_sha256": scaffold.content_sha256}
                numerical_failures.append({"target": target, "formula_id": binding.formula_id, "failure": failure, "failure_sha256": failure_sha})
    _require(not numerical_failures, "an executable Solar truth adapter failed before matrix generation")
    diagnostics = _invariance_and_identifiability(config, source_items, predictions)
    diagnostics_bytes = _json_bytes(diagnostics, indent=2)
    ledger = SyntheticReplayLedger("gravity.synthetic.solar-planetary-source-shaped-matrix.v1", ())
    scenario_rows = []
    confusion_counts = {truth: {candidate.formula_id: 0 for candidate in executable} for truth in config["mechanisms"]}
    recovery_by_truth = {truth: {"scenarios": 0, "recovered": 0, "distinct": 0} for truth in config["mechanisms"]}
    candidate_comparison_count = 0
    blocked_entry_count = 0
    truth_recovered_count = 0
    distinct_truth_recovered_count = 0
    for target in _TARGETS:
        values = source_items[target]
        for truth_index, truth_formula in enumerate(config["mechanisms"]):
            truth_prediction = predictions[(target, truth_formula)]
            for nuisance_draw, family in enumerate(config["noise_families"]):
                truth_world_id = f"truth.{truth_formula.lower()}"
                scenario_id = f"solar.{target.lower()}.{truth_world_id}.noise-{family}.v1"
                lineage = SeedLineage(config["suite_seed"], scenario_id, target.lower(), truth_world_id, nuisance_draw, 0)
                response, variance, noise_diagnostics = _noise_response(truth_prediction, family, lineage, config)
                scenario = _scenario(config, target, scenario_id, truth_world_id, nuisance_draw, values, response, variance, truth_index)
                truth_value = np.asarray([truth_index], dtype=np.int64)
                validate_scenario_catalogue(scenario, catalogue)
                validate_scenario_values(
                    scenario, formula_values=values,
                    response_values={"response.vector.synthetic-relative-acceleration": response},
                    truth_values={"truth.scalar.injection-id": truth_value},
                    uncertainty_values={"synthetic-relative-acceleration.diagonal-covariance": variance},
                )
                response_key = _array_key("response", target, truth_formula, family)
                variance_key = _array_key("variance", target, truth_formula, family)
                truth_key = _array_key("truth", target, truth_formula, family)
                arrays[response_key], arrays[variance_key], arrays[truth_key] = response, variance, truth_value
                comparisons = []
                for binding in executable:
                    candidate_comparison_count += 1
                    cached = cache[(target, binding.formula_id)]
                    metric = _profiled_metrics(cached["prediction"], response, variance)
                    comparisons.append({
                        "candidate_formula_id": binding.formula_id, "binding_sha256": binding.content_sha256,
                        "adapter_sha256": registration_by_formula[binding.formula_id].adapter_sha256,
                        "numerical_valid": True, "metrics": metric, "value_key": cached["value_key"],
                        "value_sha256": cached["value_sha256"], "output_sha256": cached["output_sha256"],
                        "source_cache_scenario_sha256": cached["scaffold_scenario_sha256"],
                    })
                ordered = sorted(comparisons, key=lambda row: (row["metrics"]["profiled_whitened_rmse"], row["candidate_formula_id"]))
                minimum = ordered[0]["metrics"]["profiled_whitened_rmse"]
                tolerance = config["scoring"]["winner_absolute_tolerance"]
                winners = sorted(row["candidate_formula_id"] for row in ordered if math.isclose(row["metrics"]["profiled_whitened_rmse"], minimum, abs_tol=tolerance, rel_tol=0.0))
                second = ordered[1]["metrics"]["profiled_whitened_rmse"]
                gap = float(second - minimum)
                distinct = len(winners) == 1 and gap >= config["scoring"]["minimum_whitened_gap_for_distinct_signature"]
                recovered = truth_formula in winners
                truth_recovered_count += int(recovered)
                distinct_truth_recovered_count += int(recovered and distinct)
                recovery_by_truth[truth_formula]["scenarios"] += 1
                recovery_by_truth[truth_formula]["recovered"] += int(recovered)
                recovery_by_truth[truth_formula]["distinct"] += int(recovered and distinct)
                for winner in winners:
                    confusion_counts[truth_formula][winner] += 1
                completed_rows = []
                for binding in bindings:
                    decision = decide_scenario_eligibility(binding, catalogue, scenario)
                    ledger = ledger.append(
                        release=release, binding=binding, eligibility=decision,
                        adapter_sha256=(registration_by_formula[binding.formula_id].adapter_sha256 if binding.formula_id in registration_by_formula else None),
                        domain="solar-system", experiment_id=config["experiment_id"],
                    )
                    if decision.status is not EligibilityStatus.ELIGIBLE:
                        blocked_entry_count += 1
                        continue
                    comparison = next(row for row in comparisons if row["candidate_formula_id"] == binding.formula_id)
                    status = status_from_result(
                        distinct_from_comparators=distinct,
                        self_injection_recovered=binding.formula_id == truth_formula and recovered,
                        numerical_valid=True,
                        powered=gap >= config["scoring"]["minimum_whitened_gap_for_distinct_signature"],
                    )
                    metric_payload = comparison["metrics"]
                    diagnostics_payload = {
                        "truth_formula_id": truth_formula, "candidate_formula_id": binding.formula_id,
                        "noise_family": family, "winner_formula_ids": winners, "profiled_whitened_gap": gap,
                        "distinct_by_frozen_threshold": distinct, "real_response_used": False,
                        "orbit_fit_performed": False, "symmetric_common_scale_profiled": True,
                    }
                    ledger = ledger.complete_last_eligible(
                        release=release, binding=binding,
                        adapter_sha256=registration_by_formula[binding.formula_id].adapter_sha256,
                        domain="solar-system", experiment_id=config["experiment_id"], status=status,
                        scenario_id=scenario.scenario_id, object_id=scenario.object_id,
                        truth_world_id=truth_world_id,
                        seed_lineage_sha256=canonical_sha256(scenario.seed_lineage.to_dict()), nuisance_draw=nuisance_draw,
                        parameter_cell_id="profiled-common-acceleration-scale", observable_ids=("response.vector.synthetic-relative-acceleration",),
                        result_sha256=comparison["output_sha256"], metrics_sha256=_json_sha256(metric_payload),
                        diagnostics_sha256=_json_sha256(diagnostics_payload),
                        reason_codes=("no-orbit-fit", "response-blind", "source-cached-common-abi-execution", "synthetic-only"),
                    )
                    completed_rows.append({**comparison, "discovery_status": status.value, "completed_ledger_sequence": ledger.entries[-1].sequence, "completed_ledger_entry_sha256": ledger.entries[-1].entry_sha256})
                scenario_rows.append({
                    "scenario": scenario.to_dict(), "scenario_sha256": scenario.content_sha256,
                    "target": target, "truth_formula_id": truth_formula, "truth_world_id": truth_world_id,
                    "noise": noise_diagnostics,
                    "value_locators": {
                        "response": {"path": VALUES_PATH.as_posix(), "key": response_key, "sha256": array_sha256(response)},
                        "variance": {"path": VALUES_PATH.as_posix(), "key": variance_key, "sha256": array_sha256(variance)},
                        "truth": {"path": VALUES_PATH.as_posix(), "key": truth_key, "sha256": array_sha256(truth_value)},
                    },
                    "candidate_comparisons": completed_rows,
                    "injection_recovery": {
                        "primary_metric": config["scoring"]["primary_metric"], "winner_formula_ids": winners,
                        "minimum_profiled_whitened_rmse": minimum, "second_best_profiled_whitened_rmse": second,
                        "profiled_whitened_gap": gap, "distinct_by_frozen_threshold": distinct,
                        "truth_recovered": recovered, "truth_distinctly_recovered": recovered and distinct,
                    },
                })
    values_bytes = _npz_bytes(arrays)
    _require(values_bytes == _npz_bytes(arrays), "NPZ serialization is nondeterministic")
    scenarios_bytes = b"".join(_json_bytes(row) + b"\n" for row in scenario_rows)
    ledger_bytes = _json_bytes(ledger.to_dict(), indent=2)
    confusion = {
        "schema": "open-gravity-solar-planetary-confusion-matrix-1.0",
        "truth_formula_ids": config["mechanisms"], "candidate_formula_ids": [row.formula_id for row in executable],
        "winner_membership_counts": confusion_counts, "recovery_by_truth": recovery_by_truth,
        "scenario_count": len(scenario_rows), "truth_recovered_count": truth_recovered_count,
        "distinct_truth_recovered_count": distinct_truth_recovered_count,
        "candidate_comparison_count": candidate_comparison_count, "numerical_failure_count": len(numerical_failures),
        "no_hand_ranking": True,
    }
    confusion_bytes = _json_bytes(confusion, indent=2)
    status = "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT"
    receipt_body = {
        "schema": "open-gravity-solar-planetary-source-shaped-synthetic-injection-matrix-receipt-1.0",
        "package_id": config["package_id"], "version": config["version"], "status": status,
        "claim_class": config["claim_class"], "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False, "distinct_independent_audit_required": True,
        "target_count": len(_TARGETS), "epoch_count": len(epochs), "body_slot_count": len(_BODY_NAMES),
        "source_domain_count": len(config["source_domains"]), "mechanism_count": len(executable),
        "noise_family_count": len(config["noise_families"]), "scenario_count": len(scenario_rows),
        "common_abi_execution_count": len(cache), "successful_common_abi_execution_count": sum(row["success"] for row in cache.values()),
        "candidate_comparison_count": candidate_comparison_count,
        "confusion_matrix_cell_count": len(config["mechanisms"]) * len(executable),
        "replay_entry_count": len(ledger.entries), "blocked_ledger_entry_count": blocked_entry_count,
        "truth_recovered_count": truth_recovered_count, "distinct_truth_recovered_count": distinct_truth_recovered_count,
        "recovery_by_truth": recovery_by_truth, "targets": list(_TARGETS), "mechanism_ids": [row.formula_id for row in executable],
        "source_domain_ids": config["source_domains"], "body_order": list(_BODY_NAMES),
        "formula_bindings": {row.formula_id: row.to_dict() for row in bindings},
        "formula_binding_sha256": {row.formula_id: row.content_sha256 for row in bindings},
        "adapter_sha256": {row.formula_binding.formula_id: row.adapter_sha256 for row in registrations},
        "adapter_blocks": config["adapter_blocks"], "numerical_failures": numerical_failures,
        "release": release.to_dict(), "catalogue_sha256": catalogue.content_sha256,
        "source_anchors": {
            "lane6_v1_config_sha256": next(row["sha256"] for row in config["upstream_bindings"] if row["id"] == "LANE6_SOLAR_V1_CONFIG"),
            "lane6_v2_receipt_sha256": next(row["sha256"] for row in config["upstream_bindings"] if row["id"] == "LANE6_SOLAR_V2_RECEIPT"),
            "de440_public_kernel_sha256": next(row["sha256"] for row in config["upstream_bindings"] if row["id"] == "DE440_PUBLIC_KERNEL"),
            "de440_states_extracted": 0,
        },
        "invariance_gates": {
            "maximum_translation_error_m_s2": diagnostics["maximum_translation_error_m_s2"],
            "maximum_rotation_error_m_s2": diagnostics["maximum_rotation_error_m_s2"],
            "maximum_nested_refinement_error_m_s2": diagnostics["maximum_nested_refinement_error_m_s2"],
            "pass": diagnostics["maximum_translation_error_m_s2"] < 1e-14 and diagnostics["maximum_rotation_error_m_s2"] < 1e-14 and diagnostics["maximum_nested_refinement_error_m_s2"] < 1e-14,
        },
        "identifiability": {"degenerate_pair_count": diagnostics["degenerate_pair_count"], "pair_count": diagnostics["pair_count"], "threshold": config["scoring"]["pairwise_degeneracy_relative_rms_max"]},
        "package_hashes": {
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH), "module_raw_sha256": module_sha,
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH), "upstream_raw_sha256": _upstream_hashes(config),
        },
        "artifact_sha256": {
            "values.npz": hashlib.sha256(values_bytes).hexdigest(), "scenarios.jsonl": hashlib.sha256(scenarios_bytes).hexdigest(),
            "ledger.json": hashlib.sha256(ledger_bytes).hexdigest(), "confusion-matrix.json": hashlib.sha256(confusion_bytes).hexdigest(),
            "invariance-and-identifiability.json": hashlib.sha256(diagnostics_bytes).hexdigest(),
        },
        "access_accounting": {
            "de440_kernel_hash_bytes_read": config["access_contract"]["de440_kernel_bytes_hash_read"],
            "de440_state_values_extracted": 0, "lane6_public_source_config_bytes_read": config["access_contract"]["lane6_public_source_config_bytes_read"],
            "observational_response_files_opened": 0, "observational_response_rows_opened": 0,
            "observational_residual_values_opened": 0, "response_calibrated_parameters": 0,
            "orbit_fits_performed": 0, "network_calls": 0, "model_calls": 0, "paid_calls": 0,
        },
        "limitations": [
            "The official DE440 kernel is an exact hash-bound public anchor, but no SPK state is extracted in this slice.",
            "Positions are the frozen Lane-6 JPL 1800-2050 approximate-element construction, not precision DE440 states.",
            "Each epoch is a static target-minus-Sun force snapshot; no trajectory is propagated and no orbit is fitted.",
            "Planet radii in this package mean instantaneous heliocentric orbital radii, not physical body radii.",
            "The one fitted common acceleration scale is a symmetric synthetic nuisance projection, not a response-calibrated parameter.",
            "D05, D06, and D07 may be numerically degenerate; ties and all underpowered cells are retained.",
            "Galaxy/cluster formulas without an audited compact-body feature and self-field adapter remain explicitly blocked.",
            "Synthetic recovery and confusion cannot support or reject any gravity theory.",
        ],
    }
    receipt = {**receipt_body, "content_sha256": _json_sha256(receipt_body)}
    return receipt, values_bytes, scenarios_bytes, ledger_bytes, confusion_bytes, diagnostics_bytes


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing to overwrite changed artifact: {path}")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "CREATED"


def freeze() -> str:
    receipt, values, scenarios, ledger, confusion, diagnostics = derive_release()
    statuses = [
        _write_once(_ROOT / VALUES_PATH, values), _write_once(_ROOT / SCENARIOS_PATH, scenarios),
        _write_once(_ROOT / LEDGER_PATH, ledger), _write_once(_ROOT / CONFUSION_PATH, confusion),
        _write_once(_ROOT / DIAGNOSTICS_PATH, diagnostics),
        _write_once(_ROOT / RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    ]
    return ",".join(statuses)


def check() -> dict[str, Any]:
    receipt, values, scenarios, ledger, confusion, diagnostics = derive_release()
    expected = (
        (VALUES_PATH, values), (SCENARIOS_PATH, scenarios), (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion), (DIAGNOSTICS_PATH, diagnostics),
        (RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    for relative, payload in expected:
        path = _ROOT / relative
        _require(path.is_file() and path.read_bytes() == payload, f"frozen Solar artifact changed: {relative}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze())
    else:
        receipt = check()
        print(receipt["status"], receipt["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
