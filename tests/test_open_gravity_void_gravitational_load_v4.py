from __future__ import annotations

import math

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_void_gravitational_load_v4 as lane


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_rays_distances_radii_and_intervals_fail_closed(bad: float) -> None:
    with pytest.raises(lane.VoidLoadV4Error):
        lane.ray_sphere_intervals([bad, 0.0, 0.0], 1.0, [])
    with pytest.raises(lane.VoidLoadV4Error):
        lane.ray_sphere_intervals([1.0, 0.0, 0.0], bad, [])
    with pytest.raises(lane.VoidLoadV4Error):
        lane.ray_sphere_intervals([1.0, 0.0, 0.0], 1.0, [([0.0, 0.0, 0.0], bad)])
    with pytest.raises(lane.VoidLoadV4Error):
        lane.union_intervals([(0.0, bad)])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_delta_h_and_prediction_inputs_fail_closed(bad: float) -> None:
    with pytest.raises(lane.VoidLoadV4Error):
        lane.identifiable_void_prediction(bad, 1.0, 1.0)
    with pytest.raises(lane.VoidLoadV4Error):
        lane.identifiable_void_prediction(1.0, bad, 1.0)
    with pytest.raises(lane.VoidLoadV4Error):
        lane.identifiable_void_prediction(1.0, 1.0, bad)
    with pytest.raises(lane.VoidLoadV4Error):
        lane.observed_log_redshift(bad, 299792.458)


def test_finite_inputs_that_overflow_prediction_fail_closed() -> None:
    with pytest.raises(lane.VoidLoadV4Error, match="nonfinite prediction"):
        lane.identifiable_void_prediction(1e308, 1e308, 1.0)


def test_valid_partition_and_identifiable_sign() -> None:
    spheres = [(np.array([3.0, 0.0, 0.0]), 2.0)]
    partition = lane.path_partition([1.0, 0.0, 0.0], 10.0, spheres, [(0.0, 10.0)])
    assert partition == pytest.approx({"L_void": 4.0, "L_observed_matter": 6.0, "L_unobserved": 0.0, "D": 10.0})
    logz, velocity = lane.identifiable_void_prediction(0.02, partition["L_void"], 1.0)
    assert logz == pytest.approx(0.08) and velocity == pytest.approx(0.08)


def test_absorbed_baseline_is_hm_times_full_distance() -> None:
    law = lane.load_config()["identifiable_law"]
    assert "H_m*D" in law["full_two_phase_log_redshift"]
    assert "H_m*D/c" in law["absorbed_baseline"]
    assert "L_observed" not in law["absorbed_baseline"]


def test_pins_gates_and_receipt_replay() -> None:
    lane.validate_code_pins()
    assert all(row["passed"] for row in lane.gates())
    first = lane.build_receipt()
    second = lane.build_receipt()
    assert first == second
    assert first["content_sha256"] == lane._self_hash(first)
    assert first["access_accounting"]["scientific_rows_decoded"] == 0
    assert lane.check_receipt() == first
