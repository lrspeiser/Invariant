from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_rg_things_2d_projection_benchmark_v1 as projection


@pytest.fixture(scope="module")
def config() -> dict:
    return projection.load_config(verify_package=False)


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return projection.build_receipt(config)


def test_builder_admission_binds_data_papers_and_benchmarks(config: dict) -> None:
    admission = config["builder_admission"]
    assert admission["real_public_data_bound"] is True
    assert admission["primary_data_and_kinematic_papers_bound"] is True
    assert admission["independent_target_free_benchmarks_required"] is True
    assert admission["missing_data_disposition"] == "SOURCE_BLOCKED"
    assert admission["benchmark_failure_disposition"] == "BUILDER_BLOCKED_RETAIN_FAILURE"
    source = projection._load_source_binding(config)
    assert {row["id"] for row in source["primary_sources"]} == {
        "THINGS_HIGH_RESOLUTION_KINEMATICS",
        "THINGS_SURVEY_DATA_RELEASE",
    }
    assert source["file_count"] == 4
    assert source["byte_count"] == 16974720


def test_disk_coordinate_convention() -> None:
    x = np.asarray([2.0, 0.0])
    y = np.asarray([0.0, 3.0])
    major, disk_y, radius, cosine = projection.disk_coordinates(
        x,
        y,
        position_angle_deg=0.0,
        inclination_deg=60.0,
    )
    assert major == pytest.approx([0.0, 3.0])
    assert disk_y == pytest.approx([-4.0, 0.0])
    assert radius == pytest.approx([4.0, 3.0])
    assert cosine == pytest.approx([0.0, 1.0])


def test_projection_has_face_on_and_rotation_sign_limits() -> None:
    x = np.asarray([1.0e18, -1.0e18])
    y = np.asarray([0.0, 0.0])
    acceleration = np.asarray([4.0e-8, 4.0e-8])
    face = projection.project_quasi_circular(
        x,
        y,
        acceleration,
        position_angle_deg=90.0,
        inclination_deg=0.0,
        systemic_velocity_m_s=0.0,
        rotation_sign=1.0,
    )
    forward = projection.project_quasi_circular(
        x,
        y,
        acceleration,
        position_angle_deg=90.0,
        inclination_deg=55.0,
        systemic_velocity_m_s=0.0,
        rotation_sign=1.0,
    )
    reverse = projection.project_quasi_circular(
        x,
        y,
        acceleration,
        position_angle_deg=90.0,
        inclination_deg=55.0,
        systemic_velocity_m_s=0.0,
        rotation_sign=-1.0,
    )
    assert face == pytest.approx([0.0, 0.0])
    assert reverse == pytest.approx(-forward)


def test_beam_preserves_constant_field_and_flux() -> None:
    kernel = projection.elliptical_gaussian_kernel(
        65,
        beam_major_pixels=6.0,
        beam_minor_pixels=4.0,
        beam_position_angle_deg=27.0,
    )
    intensity = np.ones((65, 65), dtype=np.float64)
    velocity = np.full((65, 65), 42125.0)
    convolved, denominator = projection.intensity_weighted_beam(velocity, intensity, kernel)
    assert convolved[denominator > 1.0e-12] == pytest.approx(42125.0, abs=1.0e-9)
    delta = np.zeros((65, 65), dtype=np.float64)
    delta[32, 32] = 1.0
    from scipy.signal import fftconvolve

    assert float(np.sum(fftconvolve(delta, kernel, mode="same"))) == pytest.approx(1.0, rel=1.0e-12)


def test_analytic_systemic_offset() -> None:
    predicted = np.asarray([10.0, 20.0, 30.0])
    observed = predicted + 7.5
    assert projection.analytic_systemic_offset(
        predicted, observed, np.asarray([1.0, 2.0, 3.0])
    ) == pytest.approx(7.5)


def test_all_target_free_benchmarks_pass(receipt: dict) -> None:
    assert receipt["benchmarks"]["all_pass"] is True
    assert len(receipt["benchmarks"]["checks"]) == 7
    assert all(receipt["benchmarks"]["checks"].values())
    assert receipt["status"] == "PASS_TARGET_FREE_2D_PROJECTION_RESPONSE_PIXEL_DECODE_ALLOWED"
    assert receipt["decision"] == "READY_TO_BUILD_FIXED_MATCHED_PAIR_2D_SOURCE_PREDICTIONS"


def test_receipt_preserves_zero_response_access_and_claim_limits(receipt: dict) -> None:
    boundary = receipt["scientific_boundary"]
    assert boundary["velocity_pixel_values_decoded"] == 0
    assert boundary["dispersion_pixel_values_decoded"] == 0
    assert boundary["scientific_scores_computed"] == 0
    assert boundary["quasi_circular_projection_not_gas_dynamics"] is True
    assert receipt["claim_boundary"]["scientific_fit_tested"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_receipt_is_deterministic(config: dict, receipt: dict) -> None:
    assert projection.build_receipt(config) == receipt
    assert receipt["content_sha256"] == projection.content_sha256({**receipt, "content_sha256": ""})


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "READY_TO_SCORE"),
        (("builder_admission", "real_public_data_bound"), False),
        (("builder_admission", "missing_data_disposition"), "READY"),
        (("builder_admission", "response_pixel_decode_allowed_only_if_all_benchmarks_pass"), False),
        (("scientific_boundary", "general_3d_motion_predicted"), True),
        (("scientific_boundary", "velocity_pixel_values_decoded"), 1),
        (("claim_boundary", "publication_ready"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(projection.ProjectionBenchmarkError):
            projection.validate_config(mutated)


def test_receipt_mutation_fails(config: dict, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["new_gravity_law_supported"] = True
    mutated["content_sha256"] = projection.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(projection.ProjectionBenchmarkError):
        projection.validate_receipt_payload(config, mutated)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert projection._atomic_no_clobber(output, b"one\n") == "CREATED"
    assert projection._atomic_no_clobber(output, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(projection.ProjectionBenchmarkError):
        projection._atomic_no_clobber(output, b"two\n")
