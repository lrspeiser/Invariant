"""Deterministic observation operators for real-shaped synthetic discovery.

The functions transform hidden latent values into an analysis product.  They do
not score formulas and the returned product never contains the latent truth or
injection identity.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

SCHEMA_VERSION = "open-gravity-observation-operator-1.0"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CalibrationTier(str, Enum):
    ANALYTIC = "ANALYTIC"
    PUBLIC_SOURCE_ONLY = "PUBLIC_SOURCE_ONLY"
    REAL_RESPONSE_CALIBRATED_DEVELOPMENT_ONLY = "REAL_RESPONSE_CALIBRATED_DEVELOPMENT_ONLY"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256")
    return value


def _finite_vector(values: Any, label: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise SchemaViolation(f"{label} must be a nonempty finite vector")
    return array


def _finite_matrix(values: Any, label: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape or not np.all(np.isfinite(array)):
        raise SchemaViolation(f"{label} must be a nonempty finite matrix")
    return array


@dataclass(frozen=True, slots=True)
class SeedLineage:
    suite_seed: int
    scenario_id: str
    object_id: str
    truth_world_id: str
    nuisance_draw: int
    operator_draw: int

    def __post_init__(self) -> None:
        for label, value in (
            ("suite_seed", self.suite_seed),
            ("nuisance_draw", self.nuisance_draw),
            ("operator_draw", self.operator_draw),
        ):
            if type(value) is not int or value < 0:
                raise SchemaViolation(f"{label} must be a nonnegative integer")
        for label, value in (
            ("scenario_id", self.scenario_id),
            ("object_id", self.object_id),
            ("truth_world_id", self.truth_world_id),
        ):
            _identifier(value, label)

    @property
    def derived_seed(self) -> int:
        return int(canonical_sha256(self.to_dict())[:16], 16)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_seed": self.suite_seed,
            "scenario_id": self.scenario_id,
            "object_id": self.object_id,
            "truth_world_id": self.truth_world_id,
            "nuisance_draw": self.nuisance_draw,
            "operator_draw": self.operator_draw,
        }


@dataclass(frozen=True, slots=True)
class ObservationOperatorSpec:
    operator_id: str
    domain: str
    input_element_id: str
    output_element_id: str
    calibration_tier: CalibrationTier
    calibration_sha256: str
    transform_ids: tuple[str, ...]
    seed_lineage: SeedLineage
    contaminated_confirmation_dataset_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("observation-operator schema changed")
        for label, value in (
            ("operator_id", self.operator_id),
            ("domain", self.domain),
            ("input_element_id", self.input_element_id),
            ("output_element_id", self.output_element_id),
        ):
            _identifier(value, label)
        _hash(self.calibration_sha256, "calibration_sha256")
        if self.transform_ids != tuple(dict.fromkeys(self.transform_ids)) or not self.transform_ids:
            raise SchemaViolation("transform IDs must be nonempty and ordered-unique")
        for transform in self.transform_ids:
            _identifier(transform, "transform ID")
        if self.contaminated_confirmation_dataset_ids != tuple(
            sorted(set(self.contaminated_confirmation_dataset_ids))
        ):
            raise SchemaViolation("contaminated dataset IDs must be unique and sorted")
        for dataset in self.contaminated_confirmation_dataset_ids:
            _identifier(dataset, "contaminated dataset ID")
        response_calibrated = (
            self.calibration_tier is CalibrationTier.REAL_RESPONSE_CALIBRATED_DEVELOPMENT_ONLY
        )
        if response_calibrated != bool(self.contaminated_confirmation_dataset_ids):
            raise SchemaViolation("response calibration must name every contaminated dataset")

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operator_id": self.operator_id,
            "domain": self.domain,
            "input_element_id": self.input_element_id,
            "output_element_id": self.output_element_id,
            "calibration_tier": self.calibration_tier.value,
            "calibration_sha256": self.calibration_sha256,
            "transform_ids": list(self.transform_ids),
            "seed_lineage": self.seed_lineage.to_dict(),
            "contaminated_confirmation_dataset_ids": list(
                self.contaminated_confirmation_dataset_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class AnalysisProduct:
    operator_sha256: str
    transform_input_sha256: str
    output_element_id: str
    values: NDArray[np.float64]
    covariance: NDArray[np.float64]
    observed_indices: NDArray[np.int64]
    product_sha256: str

    def __post_init__(self) -> None:
        _hash(self.operator_sha256, "operator_sha256")
        _hash(self.transform_input_sha256, "transform_input_sha256")
        _identifier(self.output_element_id, "output_element_id")
        values = _finite_vector(self.values, "analysis values")
        covariance = _finite_matrix(self.covariance, "analysis covariance")
        indices = np.asarray(self.observed_indices, dtype=np.int64)
        if covariance.shape != (values.size, values.size):
            raise SchemaViolation("analysis covariance shape changed")
        if indices.shape != (values.size,) or len(set(indices.tolist())) != values.size:
            raise SchemaViolation("observed indices must be a unique vector matching values")
        if np.any(indices < 0) or not np.array_equal(indices, np.sort(indices)):
            raise SchemaViolation("observed indices must be nonnegative and sorted")
        _hash(self.product_sha256, "product_sha256")
        expected = canonical_sha256(
            {
                "operator_sha256": self.operator_sha256,
                "transform_input_sha256": self.transform_input_sha256,
                "output_element_id": self.output_element_id,
                "values_hex": [value.hex() for value in values],
                "covariance_hex": [[value.hex() for value in row] for row in covariance],
                "observed_indices": indices.tolist(),
            }
        )
        if expected != self.product_sha256:
            raise SchemaViolation("analysis product hash changed")
        for array in (values, covariance, indices):
            array.setflags(write=False)


def normalized_convolution_matrix(kernel: Any, sample_count: int) -> NDArray[np.float64]:
    weights = _finite_vector(kernel, "convolution kernel")
    if np.any(weights < 0.0) or not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1e-12):
        raise SchemaViolation("convolution kernel must be nonnegative and normalized")
    if type(sample_count) is not int or sample_count <= 0:
        raise SchemaViolation("sample_count must be positive")
    centre = weights.size // 2
    matrix = np.zeros((sample_count, sample_count), dtype=np.float64)
    for row in range(sample_count):
        for offset, weight in enumerate(weights):
            column = row + offset - centre
            if 0 <= column < sample_count:
                matrix[row, column] += weight
        row_sum = float(np.sum(matrix[row]))
        if row_sum > 0.0:
            matrix[row] /= row_sum
    return matrix


def project_line_of_sight(vector_field: Any, unit_directions: Any) -> NDArray[np.float64]:
    vectors = np.asarray(vector_field, dtype=np.float64)
    directions = np.asarray(unit_directions, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape != directions.shape:
        raise SchemaViolation("line-of-sight vectors and directions must have shape (n, 3)")
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(directions)):
        raise SchemaViolation("line-of-sight inputs must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise SchemaViolation("line-of-sight direction cannot be zero")
    normalized = directions / norms[:, None]
    return np.einsum("ij,ij->i", vectors, normalized, dtype=np.float64)


def observe_linear(
    spec: ObservationOperatorSpec,
    latent_values: Any,
    response_matrix: Any,
    covariance: Any,
    observed_mask: Any,
) -> AnalysisProduct:
    latent = _finite_vector(latent_values, "latent values")
    response = _finite_matrix(response_matrix, "response matrix")
    full_covariance = _finite_matrix(covariance, "covariance")
    mask = np.asarray(observed_mask)
    if response.shape[1] != latent.size:
        raise SchemaViolation("response matrix and latent vector are incompatible")
    if full_covariance.shape != (response.shape[0], response.shape[0]):
        raise SchemaViolation("covariance and response matrix are incompatible")
    if mask.dtype != np.bool_ or mask.shape != (response.shape[0],) or not np.any(mask):
        raise SchemaViolation("observed mask must be a nonempty boolean vector")
    if not np.allclose(full_covariance, full_covariance.T, atol=0.0, rtol=0.0):
        raise SchemaViolation("covariance must be exactly symmetric")
    eigenvalues = np.linalg.eigvalsh(full_covariance)
    if float(np.min(eigenvalues)) < -1e-12:
        raise SchemaViolation("covariance is not positive semidefinite")
    indices = np.flatnonzero(mask).astype(np.int64)
    mean = response @ latent
    selected_mean = mean[indices]
    selected_covariance = full_covariance[np.ix_(indices, indices)]
    rng = np.random.Generator(np.random.PCG64(spec.seed_lineage.derived_seed))
    eigenvalues, eigenvectors = np.linalg.eigh(selected_covariance)
    noise = eigenvectors @ (
        np.sqrt(np.clip(eigenvalues, 0.0, None))
        * rng.standard_normal(indices.size, dtype=np.float64)
    )
    values = np.asarray(selected_mean + noise, dtype=np.float64)
    transform_input_sha256 = canonical_sha256(
        {
            "latent_hex": [value.hex() for value in latent],
            "response_matrix_hex": [[value.hex() for value in row] for row in response],
            "full_covariance_hex": [[value.hex() for value in row] for row in full_covariance],
            "observed_mask": mask.tolist(),
        }
    )
    body = {
        "operator_sha256": spec.content_sha256,
        "transform_input_sha256": transform_input_sha256,
        "output_element_id": spec.output_element_id,
        "values_hex": [value.hex() for value in values],
        "covariance_hex": [[value.hex() for value in row] for row in selected_covariance],
        "observed_indices": indices.tolist(),
    }
    return AnalysisProduct(
        operator_sha256=spec.content_sha256,
        transform_input_sha256=transform_input_sha256,
        output_element_id=spec.output_element_id,
        values=values,
        covariance=np.asarray(selected_covariance, dtype=np.float64),
        observed_indices=indices,
        product_sha256=canonical_sha256(body),
    )


__all__ = [
    "SCHEMA_VERSION",
    "AnalysisProduct",
    "CalibrationTier",
    "ObservationOperatorSpec",
    "SeedLineage",
    "normalized_convolution_matrix",
    "observe_linear",
    "project_line_of_sight",
]
