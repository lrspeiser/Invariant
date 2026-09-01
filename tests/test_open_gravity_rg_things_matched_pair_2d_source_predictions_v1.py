from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_rg_things_matched_pair_2d_source_predictions_v1 as predictions,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return predictions.load_config(verify_package=False)


@pytest.fixture(scope="module")
def packet(config: dict) -> tuple[dict, dict, dict]:
    return predictions.build_packet(config)


def test_admission_requires_data_papers_benchmarks_and_control(config: dict) -> None:
    admission = config["admission_rule"]
    assert admission["real_public_source_data_required"] is True
    assert admission["primary_measurement_and_data_release_papers_required"] is True
    assert admission["independent_operator_benchmarks_required"] is True
    assert admission["known_newtonian_control_required"] is True
    assert admission["missing_data_disposition"] == "SOURCE_BLOCKED"
    assert admission["spherical_or_1d_data_cannot_validate_general_3d"] is True
    assert admission["model_lifted_vertical_structure_is_not_observed_3d"] is True
    source_ids = {row["id"] for row in config["primary_sources"]}
    assert {"THINGS_SURVEY_DATA_RELEASE", "THINGS_HIGH_RESOLUTION_KINEMATICS"} <= source_ids
    assert {"FREEMAN_DISK", "CASERTANO_FINITE_THICKNESS"} <= source_ids
    assert config["operator"]["models"][0] == "NEWTON_3D_DST"


def test_ngc4214_is_retained_as_low_inclination_counterexample(config: dict) -> None:
    row = next(item for item in config["objects"] if item["object_id"] == "NGC4214")
    assert row["kinematic_disposition"] == (
        "RETAINED_LOW_INCLINATION_COUNTEREXAMPLE_NOT_STANDARD_ROTATION_CURVE"
    )
    assert "excluded a standard rotation curve" in row["paper_inclination_note"]


def test_response_access_is_header_only(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _row = predictions._response_header_path(config, "NGC2976")

    def forbidden(*args, **kwargs):
        raise AssertionError("response pixel loader called")

    monkeypatch.setattr(predictions.fits, "getdata", forbidden)
    header = predictions._response_header_only(path)
    assert int(header["NAXIS1"]) == 1024
    assert int(header["NAXIS2"]) == 1024
    assert config["scientific_boundary"]["velocity_pixel_values_decoded"] == 0
    assert config["scientific_boundary"]["response_bytes_hashed"] == 8_487_360


def test_radial_force_sampling_has_known_answer() -> None:
    nodes = 9
    coordinate = np.linspace(-1.0, 1.0, nodes)
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    radial, tangential = predictions._sample_force(
        (-x, -y),
        np.asarray([0.5, 0.0]),
        np.asarray([0.0, 0.5]),
        np.asarray([0.5, 0.5]),
        half_box_kpc=1.0,
        a0_m_s2=2.0,
    )
    assert radial == pytest.approx([1.0, 1.0])
    assert tangential == pytest.approx([0.0, 0.0], abs=1.0e-15)


def test_beam_covariance_difference_reconstructs_target() -> None:
    source = (0.0014, 0.0011, 25.0)
    target = (0.0022, 0.0018, 67.0)
    pixel = 0.0004
    additional = predictions.additional_beam(source, target, pixel)
    reconstructed = predictions._beam_covariance(
        source[0] / pixel, source[1] / pixel, source[2]
    ) + predictions._beam_covariance(
        float(additional["major_pixels"]),
        float(additional["minor_pixels"]),
        float(additional["position_angle_deg"]),
    )
    expected = predictions._beam_covariance(target[0] / pixel, target[1] / pixel, target[2])
    assert reconstructed == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
    assert float(np.sum(additional["kernel"])) == pytest.approx(1.0, rel=1.0e-14)


def test_real_source_predictions_build_without_response_values(
    config: dict, packet: tuple[dict, dict, dict]
) -> None:
    arrays, manifest, receipt = packet
    assert set(arrays) == {"NGC2976", "NGC4214"}
    assert manifest["array_count"] == 18
    assert receipt["private_array_count"] == 18
    assert all(row["all_solver_gates_pass"] for row in receipt["objects"])
    assert all(row["eligible_prediction_pixels"] > 0 for row in receipt["objects"])
    assert all(row["response_header"]["data_values_decoded"] == 0 for row in receipt["objects"])
    assert receipt["scientific_boundary"]["scientific_scores_computed"] == 0
    assert receipt["claim_boundary"]["scientific_fit_tested"] is False
    for object_arrays in arrays.values():
        assert set(object_arrays) == {
            "radius_kpc",
            "newton_vlos_plus_m_s",
            "newton_tangential_ratio",
            "newton_convergence_relative",
            "rg_vlos_plus_m_s",
            "rg_tangential_ratio",
            "rg_convergence_relative",
            "source_intensity",
            "source_eligibility",
        }


def test_packet_is_deterministically_self_hashed(packet: tuple[dict, dict, dict]) -> None:
    _arrays, manifest, receipt = packet
    assert manifest["content_sha256"] == predictions.content_sha256(
        {**manifest, "content_sha256": ""}
    )
    assert receipt["content_sha256"] == predictions.content_sha256(
        {**receipt, "content_sha256": ""}
    )


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "READY_TO_PUBLISH"),
        (("admission_rule", "real_public_source_data_required"), False),
        (("admission_rule", "missing_data_disposition"), "READY"),
        (("operator", "epsilon_0"), 0.7),
        (("scientific_boundary", "velocity_pixel_values_decoded"), 1),
        (("scientific_boundary", "response_bytes_hashed"), 0),
        (("claim_boundary", "general_3d_validated"), True),
        (("claim_boundary", "publication_ready"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(predictions.MatchedPairPredictionError):
            predictions.validate_config(mutated)


def test_receipt_mutation_fails(config: dict, packet: tuple[dict, dict, dict]) -> None:
    _arrays, _manifest, receipt = packet
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["scientific_fit_tested"] = True
    mutated["content_sha256"] = predictions.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(predictions.MatchedPairPredictionError):
        predictions.validate_receipt_payload(config, mutated, receipt)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert predictions._atomic_no_clobber(output, b"one\n") == "CREATED"
    assert predictions._atomic_no_clobber(output, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(predictions.MatchedPairPredictionError):
        predictions._atomic_no_clobber(output, b"two\n")
