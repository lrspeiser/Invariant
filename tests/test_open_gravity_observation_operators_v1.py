from __future__ import annotations

import numpy as np
import pytest

from sigma_theory_compiler.open_gravity_observation_operators_v1 import (
    CalibrationTier,
    ObservationOperatorSpec,
    SeedLineage,
    normalized_convolution_matrix,
    observe_linear,
    project_line_of_sight,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

HASH = "4" * 64


def spec(tier: CalibrationTier = CalibrationTier.PUBLIC_SOURCE_ONLY) -> ObservationOperatorSpec:
    return ObservationOperatorSpec(
        operator_id="galaxy.rotation.observe.v1",
        domain="galaxy",
        input_element_id="truth.vector.acceleration",
        output_element_id="response.scalar.velocity",
        calibration_tier=tier,
        calibration_sha256=HASH,
        transform_ids=("projection.los", "instrument.beam", "noise.covariance"),
        seed_lineage=SeedLineage(
            suite_seed=1729,
            scenario_id="galaxy.disk.v1",
            object_id="ngc2903",
            truth_world_id="newton.nfw",
            nuisance_draw=0,
            operator_draw=0,
        ),
        contaminated_confirmation_dataset_ids=("sparc",)
        if tier is CalibrationTier.REAL_RESPONSE_CALIBRATED_DEVELOPMENT_ONLY
        else (),
    )


def test_linear_observation_is_deterministic_and_does_not_return_truth() -> None:
    operator = spec()
    latent = np.array([1.0, 2.0, 3.0])
    response = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    covariance = np.diag([0.01, 0.04, 0.09])
    mask = np.array([True, False, True])
    first = observe_linear(operator, latent, response, covariance, mask)
    second = observe_linear(operator, latent, response, covariance, mask)
    assert first.product_sha256 == second.product_sha256
    assert np.array_equal(first.values, second.values)
    assert first.observed_indices.tolist() == [0, 2]
    assert not hasattr(first, "latent_values")
    with pytest.raises(ValueError):
        first.values[0] = 0.0


def test_response_calibration_contaminates_named_confirmation_dataset() -> None:
    calibrated = spec(CalibrationTier.REAL_RESPONSE_CALIBRATED_DEVELOPMENT_ONLY)
    assert calibrated.contaminated_confirmation_dataset_ids == ("sparc",)
    with pytest.raises(SchemaViolation, match="contaminated"):
        ObservationOperatorSpec(
            operator_id="galaxy.bad.v1",
            domain="galaxy",
            input_element_id="truth.vector.acceleration",
            output_element_id="response.scalar.velocity",
            calibration_tier=CalibrationTier.REAL_RESPONSE_CALIBRATED_DEVELOPMENT_ONLY,
            calibration_sha256=HASH,
            transform_ids=("projection.los",),
            seed_lineage=SeedLineage(1, "galaxy.disk.v1", "ngc2903", "newton.nfw", 0, 0),
        )


def test_projection_convolution_and_covariance_fail_closed() -> None:
    vectors = np.eye(3)
    directions = 2.0 * np.eye(3)
    assert np.array_equal(project_line_of_sight(vectors, directions), np.ones(3))
    matrix = normalized_convolution_matrix([0.25, 0.5, 0.25], 5)
    assert np.allclose(np.sum(matrix, axis=1), 1.0)
    with pytest.raises(SchemaViolation, match="positive semidefinite"):
        observe_linear(
            spec(),
            [1.0],
            [[1.0], [1.0]],
            [[1.0, 2.0], [2.0, 1.0]],
            np.array([True, True]),
        )


def test_singular_positive_semidefinite_covariance_is_supported() -> None:
    result = observe_linear(
        spec(),
        [1.0],
        [[1.0], [1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        np.array([True, True]),
    )
    assert result.values.shape == (2,)
    assert result.transform_input_sha256
