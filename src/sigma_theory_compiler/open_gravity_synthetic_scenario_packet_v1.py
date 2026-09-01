"""Typed scenario/value packets and narrow deterministic formula execution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from numpy.typing import NDArray

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    DataElementCatalogue,
    DataRole,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    EligibilityDecision,
    EligibilityStatus,
    FormulaExecutionBinding,
    decide_eligibility,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

SCHEMA_VERSION = "open-gravity-synthetic-scenario-packet-1.0"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DTYPES = {"bool", "float32", "float64", "int32", "int64", "uint8", "uint32"}
_ROOT = Path(__file__).resolve().parents[2]


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256")
    return value


def _path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SchemaViolation(f"{label} must be a canonical relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise SchemaViolation(f"{label} escaped the packet root")
    return value


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.name not in _DTYPES or not np.all(np.isfinite(array)):
        raise SchemaViolation("feature array dtype or values are invalid")
    digest = hashlib.sha256()
    digest.update(array.dtype.name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AxisSpec:
    axis_id: str
    length: int
    coordinate_element_id: str | None
    coordinate_sha256: str | None

    def __post_init__(self) -> None:
        _identifier(self.axis_id, "axis_id")
        if type(self.length) is not int or self.length <= 0:
            raise SchemaViolation("axis length must be a positive integer")
        if (self.coordinate_element_id is None) != (self.coordinate_sha256 is None):
            raise SchemaViolation("axis coordinates require both element ID and hash")
        if self.coordinate_element_id is not None:
            _identifier(self.coordinate_element_id, "coordinate element ID")
            _hash(self.coordinate_sha256, "coordinate sha256")  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "length": self.length,
            "coordinate_element_id": self.coordinate_element_id,
            "coordinate_sha256": self.coordinate_sha256,
        }


@dataclass(frozen=True, slots=True)
class FeatureValueRef:
    element_id: str
    artifact_path: str
    value_sha256: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    unit: str
    frame: str

    def __post_init__(self) -> None:
        _identifier(self.element_id, "feature element_id")
        _path(self.artifact_path, "feature artifact path")
        _hash(self.value_sha256, "feature value_sha256")
        if self.dtype not in _DTYPES:
            raise SchemaViolation("feature dtype is not registered")
        if not self.shape or any(type(value) is not int or value <= 0 for value in self.shape):
            raise SchemaViolation("feature shape must contain positive integers")
        if len(self.shape) != len(self.axes) or len(set(self.axes)) != len(self.axes):
            raise SchemaViolation("feature axes must uniquely match shape in physical order")
        for axis in self.axes:
            _identifier(axis, "feature axis")
        for label, value in (("unit", self.unit), ("frame", self.frame)):
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise SchemaViolation(f"feature {label} is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "artifact_path": self.artifact_path,
            "value_sha256": self.value_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "unit": self.unit,
            "frame": self.frame,
        }


@dataclass(frozen=True, slots=True)
class EmittedPredictionSpec:
    element_id: str
    artifact_path: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    unit: str
    frame: str

    def __post_init__(self) -> None:
        _identifier(self.element_id, "prediction element_id")
        _path(self.artifact_path, "prediction artifact path")
        if self.dtype not in _DTYPES:
            raise SchemaViolation("prediction dtype is not registered")
        if not self.shape or any(type(value) is not int or value <= 0 for value in self.shape):
            raise SchemaViolation("prediction shape must contain positive integers")
        if len(self.shape) != len(self.axes) or len(set(self.axes)) != len(self.axes):
            raise SchemaViolation("prediction axes must uniquely match shape")
        for axis in self.axes:
            _identifier(axis, "prediction axis")
        for label, value in (("unit", self.unit), ("frame", self.frame)):
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise SchemaViolation(f"prediction {label} is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "artifact_path": self.artifact_path,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "unit": self.unit,
            "frame": self.frame,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyRef:
    uncertainty_id: str
    applies_to_element_id: str
    representation: str
    artifact_path: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.uncertainty_id, "uncertainty_id")
        _identifier(self.applies_to_element_id, "uncertainty target")
        _identifier(self.representation, "uncertainty representation")
        _path(self.artifact_path, "uncertainty artifact path")
        _hash(self.artifact_sha256, "uncertainty artifact sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "uncertainty_id": self.uncertainty_id,
            "applies_to_element_id": self.applies_to_element_id,
            "representation": self.representation,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class AnchorBinding:
    anchor_id: str
    artifact_path: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.anchor_id, "anchor_id")
        _path(self.artifact_path, "anchor artifact path")
        _hash(self.artifact_sha256, "anchor artifact sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "anchor_id": self.anchor_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class ScenarioDescriptor:
    scenario_id: str
    object_id: str
    experiment_id: str
    domain: str
    geometry_mode: str
    time_mode: str
    coordinate_frame: str
    axes: tuple[AxisSpec, ...]
    formula_features: tuple[FeatureValueRef, ...]
    scoring_responses: tuple[FeatureValueRef, ...]
    hidden_truth: tuple[FeatureValueRef, ...]
    expected_predictions: tuple[EmittedPredictionSpec, ...]
    uncertainties: tuple[UncertaintyRef, ...]
    anchors: tuple[AnchorBinding, ...]
    seed_lineage: SeedLineage
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("scenario packet schema changed")
        for label, value in (
            ("scenario_id", self.scenario_id),
            ("object_id", self.object_id),
            ("experiment_id", self.experiment_id),
            ("domain", self.domain),
            ("geometry_mode", self.geometry_mode),
            ("time_mode", self.time_mode),
            ("coordinate_frame", self.coordinate_frame),
        ):
            _identifier(value, label)
        if self.seed_lineage.scenario_id != self.scenario_id or (
            self.seed_lineage.object_id != self.object_id
        ):
            raise SchemaViolation("scenario and seed lineage identity differ")
        for label, rows, key in (
            ("axes", self.axes, lambda row: row.axis_id),
            ("formula features", self.formula_features, lambda row: row.element_id),
            ("scoring responses", self.scoring_responses, lambda row: row.element_id),
            ("hidden truth", self.hidden_truth, lambda row: row.element_id),
            ("expected predictions", self.expected_predictions, lambda row: row.element_id),
            ("uncertainties", self.uncertainties, lambda row: row.uncertainty_id),
            ("anchors", self.anchors, lambda row: row.anchor_id),
        ):
            identifiers = tuple(key(row) for row in rows)
            if not identifiers or identifiers != tuple(sorted(set(identifiers))):
                raise SchemaViolation(f"scenario {label} must be nonempty, unique, and sorted")
        feature_sets = (
            {row.element_id for row in self.formula_features},
            {row.element_id for row in self.scoring_responses},
            {row.element_id for row in self.hidden_truth},
            {row.element_id for row in self.expected_predictions},
        )
        if any(feature_sets[i] & feature_sets[j] for i in range(4) for j in range(i + 1, 4)):
            raise SchemaViolation("scenario feature visibility partitions overlap")
        axis_map = {row.axis_id: row.length for row in self.axes}
        for feature in (*self.formula_features, *self.scoring_responses, *self.hidden_truth):
            if tuple(axis_map[axis] for axis in feature.axes) != feature.shape:
                raise SchemaViolation("feature shape differs from scenario axes")
        for prediction in self.expected_predictions:
            if tuple(axis_map[axis] for axis in prediction.axes) != prediction.shape:
                raise SchemaViolation("prediction shape differs from scenario axes")
        targets = {row.element_id for row in (*self.formula_features, *self.scoring_responses)}
        if any(row.applies_to_element_id not in targets for row in self.uncertainties):
            raise SchemaViolation("uncertainty target is absent from observable packet")

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "object_id": self.object_id,
            "experiment_id": self.experiment_id,
            "domain": self.domain,
            "geometry_mode": self.geometry_mode,
            "time_mode": self.time_mode,
            "coordinate_frame": self.coordinate_frame,
            "axes": [row.to_dict() for row in self.axes],
            "formula_features": [row.to_dict() for row in self.formula_features],
            "scoring_responses": [row.to_dict() for row in self.scoring_responses],
            "hidden_truth": [row.to_dict() for row in self.hidden_truth],
            "expected_predictions": [row.to_dict() for row in self.expected_predictions],
            "uncertainties": [row.to_dict() for row in self.uncertainties],
            "anchors": [row.to_dict() for row in self.anchors],
            "seed_lineage": self.seed_lineage.to_dict(),
        }


def validate_scenario_catalogue(
    scenario: ScenarioDescriptor, catalogue: DataElementCatalogue
) -> None:
    known = catalogue.by_id()

    def validate_metadata(feature: FeatureValueRef | EmittedPredictionSpec, label: str) -> None:
        if feature.element_id not in known:
            raise SchemaViolation(f"{label} is absent from ontology")
        element = known[feature.element_id]
        if (
            feature.unit != element.canonical_unit
            or feature.frame != element.frame
            or feature.axes != element.axes
        ):
            raise SchemaViolation(f"{label} unit, frame, or axes differ from ontology")

    for feature in scenario.formula_features:
        validate_metadata(feature, "formula feature")
        if not known[feature.element_id].visible_to_formula(scenario.experiment_id):
            raise SchemaViolation("formula feature is hidden in this experiment")
    for feature in scenario.scoring_responses:
        validate_metadata(feature, "scoring response")
        if (
            known[feature.element_id].role_for(scenario.experiment_id)
            is not DataRole.SCORING_ONLY_RESPONSE
        ):
            raise SchemaViolation("scoring response role differs from ontology")
    for feature in scenario.hidden_truth:
        validate_metadata(feature, "hidden truth")
        if (
            known[feature.element_id].role_for(scenario.experiment_id)
            is not DataRole.LATENT_SYNTHETIC_TRUTH
        ):
            raise SchemaViolation("hidden truth role differs from ontology")
    for prediction in scenario.expected_predictions:
        validate_metadata(prediction, "expected prediction")
        role = known[prediction.element_id].role_for(scenario.experiment_id)
        if role not in {DataRole.DERIVED, DataRole.SOURCE_DERIVED}:
            raise SchemaViolation("expected prediction role differs from ontology")

    values = {
        row.element_id: row
        for row in (*scenario.formula_features, *scenario.scoring_responses, *scenario.hidden_truth)
    }
    for axis in scenario.axes:
        if axis.coordinate_element_id is None:
            continue
        if axis.coordinate_element_id not in values:
            raise SchemaViolation("axis coordinate reference is absent from scenario values")
        coordinate = values[axis.coordinate_element_id]
        if coordinate.value_sha256 != axis.coordinate_sha256:
            raise SchemaViolation("axis coordinate hash differs from scenario value")
        coordinate_element = known[axis.coordinate_element_id]
        if (
            coordinate.axes != (axis.axis_id,)
            or coordinate.shape != (axis.length,)
            or coordinate_element.tensor_rank != 0
            or not coordinate_element.visible_to_formula(scenario.experiment_id)
        ):
            raise SchemaViolation(
                "axis coordinate must be a formula-visible scalar on exactly that axis"
            )


def decide_scenario_eligibility(
    binding: FormulaExecutionBinding,
    catalogue: DataElementCatalogue,
    scenario: ScenarioDescriptor,
) -> EligibilityDecision:
    base = decide_eligibility(binding, catalogue, scenario.experiment_id, scenario.domain)
    if base.status is not EligibilityStatus.ELIGIBLE:
        return base
    if scenario.geometry_mode not in binding.geometry_support:
        return EligibilityDecision(
            EligibilityStatus.INCOMPATIBLE_FEATURE_SET,
            (f"geometry.{scenario.geometry_mode}",),
            (),
        )
    if scenario.time_mode not in binding.time_support:
        return EligibilityDecision(
            EligibilityStatus.INCOMPATIBLE_FEATURE_SET,
            (f"time.{scenario.time_mode}",),
            (),
        )
    available = {row.element_id for row in scenario.formula_features}
    missing = tuple(sorted(set(binding.required_features) - available))
    if missing:
        return EligibilityDecision(EligibilityStatus.INCOMPATIBLE_FEATURE_SET, missing, ())
    return EligibilityDecision(EligibilityStatus.ELIGIBLE, (), ())


@dataclass(frozen=True, slots=True)
class FormulaExecutionResult:
    binding_sha256: str
    scenario_sha256: str
    parameter_values_sha256: str
    output_sha256: str
    output_predictions: Mapping[str, PredictionArtifact]
    output_values: Mapping[str, NDArray[Any]]
    deterministic_replay: bool


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    element_id: str
    artifact_path: str
    value_sha256: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    unit: str
    frame: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "artifact_path": self.artifact_path,
            "value_sha256": self.value_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "unit": self.unit,
            "frame": self.frame,
        }


def _read_only_array(value: Any, reference: FeatureValueRef) -> NDArray[Any]:
    array = np.asarray(value)
    if array.dtype.name != reference.dtype or array.shape != reference.shape:
        raise SchemaViolation("loaded feature dtype or shape differs from reference")
    if array_sha256(array) != reference.value_sha256:
        raise SchemaViolation("loaded feature content differs from reference")
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ValidatedScenarioValues:
    formula_features: Mapping[str, NDArray[Any]]
    scoring_responses: Mapping[str, NDArray[Any]]
    hidden_truth: Mapping[str, NDArray[Any]]
    uncertainties: Mapping[str, NDArray[Any]]


def validate_scenario_values(
    scenario: ScenarioDescriptor,
    *,
    formula_values: Mapping[str, Any],
    response_values: Mapping[str, Any],
    truth_values: Mapping[str, Any],
    uncertainty_values: Mapping[str, Any],
) -> ValidatedScenarioValues:
    """Validate every stored value partition, including hidden values.

    Formula execution still receives only the projected formula partition; this
    validator belongs to the trusted generator/adjudicator boundary.
    """

    def validate_partition(
        values: Mapping[str, Any], references: tuple[FeatureValueRef, ...], label: str
    ) -> Mapping[str, NDArray[Any]]:
        expected = {row.element_id: row for row in references}
        if set(values) != set(expected):
            raise SchemaViolation(f"{label} values differ from the scenario packet")
        return MappingProxyType(
            {
                element_id: _read_only_array(values[element_id], expected[element_id])
                for element_id in sorted(expected)
            }
        )

    formula = validate_partition(formula_values, scenario.formula_features, "formula")
    response = validate_partition(response_values, scenario.scoring_responses, "response")
    truth = validate_partition(truth_values, scenario.hidden_truth, "truth")
    uncertainty_refs = {row.uncertainty_id: row for row in scenario.uncertainties}
    if set(uncertainty_values) != set(uncertainty_refs):
        raise SchemaViolation("uncertainty values differ from the scenario packet")
    targets = {
        row.element_id: row for row in (*scenario.formula_features, *scenario.scoring_responses)
    }
    uncertainties: dict[str, NDArray[Any]] = {}
    for uncertainty_id in sorted(uncertainty_refs):
        reference = uncertainty_refs[uncertainty_id]
        array = np.asarray(uncertainty_values[uncertainty_id])
        if array.dtype.name not in _DTYPES or not np.all(np.isfinite(array)):
            raise SchemaViolation("uncertainty dtype or values are invalid")
        if array_sha256(array) != reference.artifact_sha256:
            raise SchemaViolation("uncertainty content differs from reference")
        target = targets[reference.applies_to_element_id]
        if reference.representation == "diagonal-covariance":
            if array.shape != target.shape or np.any(array < 0):
                raise SchemaViolation("diagonal covariance shape or sign is invalid")
        elif reference.representation == "covariance":
            size = int(np.prod(target.shape))
            if array.shape != (size, size) or not np.allclose(array, array.T):
                raise SchemaViolation("full covariance shape or symmetry is invalid")
            if float(np.min(np.linalg.eigvalsh(array.astype(np.float64)))) < -1e-12:
                raise SchemaViolation("full covariance is not positive semidefinite")
        else:
            raise SchemaViolation("uncertainty representation has no executable validator")
        copy = np.array(array, copy=True, order="C")
        copy.setflags(write=False)
        uncertainties[uncertainty_id] = copy
    return ValidatedScenarioValues(
        formula,
        response,
        truth,
        MappingProxyType(uncertainties),
    )


def execute_binding_in_process(
    binding: FormulaExecutionBinding,
    catalogue: DataElementCatalogue,
    scenario: ScenarioDescriptor,
    loaded_feature_values: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> FormulaExecutionResult:
    """Execute a deterministic foundation adapter.

    This function enforces feature projection and output validation.  It is not
    the future resource-isolated production runner; that runner must execute the
    same packet in a bounded subprocess before broad formula replay.
    """

    validate_scenario_catalogue(scenario, catalogue)
    decision = decide_scenario_eligibility(binding, catalogue, scenario)
    if decision.status is not EligibilityStatus.ELIGIBLE:
        raise SchemaViolation(f"formula is not scenario-eligible: {decision.status.value}")
    schema_path = (_ROOT / binding.parameter_schema_path).resolve()
    if not schema_path.is_relative_to(_ROOT) or not schema_path.is_file():
        raise SchemaViolation("parameter schema path is missing or escaped repository")
    schema_bytes = schema_path.read_bytes()
    if hashlib.sha256(schema_bytes).hexdigest() != binding.parameter_schema_sha256:
        raise SchemaViolation("parameter schema bytes differ from binding")
    try:
        parameter_schema = json.loads(schema_bytes)
        Draft202012Validator.check_schema(parameter_schema)
    except Exception as error:
        raise SchemaViolation("registered parameter schema is invalid") from error
    parameter_payload = dict(parameters)
    errors = sorted(
        Draft202012Validator(parameter_schema).iter_errors(parameter_payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        raise SchemaViolation(f"parameters violate registered schema: {errors[0].message}")
    references = {row.element_id: row for row in scenario.formula_features}
    selected_ids = tuple(
        sorted(set(binding.required_features) | (set(binding.optional_features) & set(references)))
    )
    if set(loaded_feature_values) != set(selected_ids):
        raise SchemaViolation("loaded feature values differ from exact binding projection")
    projected = MappingProxyType(
        {
            element_id: _read_only_array(loaded_feature_values[element_id], references[element_id])
            for element_id in selected_ids
        }
    )
    entrypoint = binding.resolve()
    prediction_specs = {row.element_id: row for row in scenario.expected_predictions}
    if not set(binding.emitted_features) <= set(prediction_specs):
        raise SchemaViolation("binding output lacks a typed prediction contract")

    def run_once() -> tuple[dict[str, PredictionArtifact], dict[str, NDArray[Any]], str]:
        raw = entrypoint(projected, MappingProxyType(parameter_payload))
        if not isinstance(raw, Mapping) or set(raw) != set(binding.emitted_features):
            raise SchemaViolation("formula outputs differ from binding")
        artifacts: dict[str, PredictionArtifact] = {}
        values: dict[str, NDArray[Any]] = {}
        for element_id in sorted(raw):
            spec = prediction_specs[element_id]
            array = np.asarray(raw[element_id])
            if array.dtype.name != spec.dtype or array.shape != spec.shape:
                raise SchemaViolation(
                    "formula output dtype or shape differs from prediction contract"
                )
            value = np.array(array, copy=True, order="C")
            value.setflags(write=False)
            digest = array_sha256(value)
            values[element_id] = value
            artifacts[element_id] = PredictionArtifact(
                element_id,
                spec.artifact_path,
                digest,
                spec.dtype,
                spec.shape,
                spec.axes,
                spec.unit,
                spec.frame,
            )
        encoded_bytes = sum(value.nbytes for value in values.values())
        if encoded_bytes > binding.resource_bounds.max_output_bytes:
            raise SchemaViolation("formula output exceeds registered resource bound")
        return (
            artifacts,
            values,
            canonical_sha256({key: artifact.to_dict() for key, artifact in artifacts.items()}),
        )

    first_artifacts, first_values, first_root = run_once()
    second_artifacts, _, second_root = run_once()
    if first_root != second_root or first_artifacts != second_artifacts:
        raise SchemaViolation("formula is nondeterministic under exact replay")
    return FormulaExecutionResult(
        binding_sha256=binding.content_sha256,
        scenario_sha256=scenario.content_sha256,
        parameter_values_sha256=canonical_sha256(parameter_payload),
        output_sha256=first_root,
        output_predictions=MappingProxyType(first_artifacts),
        output_values=MappingProxyType(first_values),
        deterministic_replay=True,
    )


__all__ = [
    "SCHEMA_VERSION",
    "AnchorBinding",
    "AxisSpec",
    "EmittedPredictionSpec",
    "FeatureValueRef",
    "FormulaExecutionResult",
    "PredictionArtifact",
    "ScenarioDescriptor",
    "UncertaintyRef",
    "ValidatedScenarioValues",
    "array_sha256",
    "decide_scenario_eligibility",
    "execute_binding_in_process",
    "validate_scenario_catalogue",
    "validate_scenario_values",
]
