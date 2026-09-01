from __future__ import annotations

import math

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_void_geometry_source_completion_v3 as lane


def test_dist_is_the_only_mpc_input_and_dmzp_is_mag_consistency() -> None:
    dmzp = 35.0
    dist = round(10.0 ** ((dmzp - 25.0) / 5.0), 1)
    row = lane.validate_cf4_distance(dmzp, dist)
    assert row["DMzp_mag"] == dmzp
    assert row["Dist_Mpc"] == dist
    assert row["difference_Mpc"] <= row["tolerance_Mpc"]
    with pytest.raises(lane.VoidGeometryV3Error, match="rounding mismatch"):
        lane.validate_cf4_distance(dmzp, dist + 0.3)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_distances_fail_closed(bad: float) -> None:
    with pytest.raises(lane.VoidGeometryV3Error):
        lane.validate_cf4_distance(bad, 10.0)
    with pytest.raises(lane.VoidGeometryV3Error):
        lane.validate_cf4_distance(30.0, bad)
    with pytest.raises(lane.VoidGeometryV3Error):
        lane.luminosity_to_comoving_hinv(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinates_and_radius_fail_closed(bad: float) -> None:
    with pytest.raises(lane.VoidGeometryV3Error):
        lane.mask_index(bad, 0.0)
    with pytest.raises(lane.VoidGeometryV3Error):
        lane.radec_to_xyz(0.0, bad, 1.0)
    with pytest.raises(lane.VoidGeometryV3Error):
        lane.radec_to_xyz(0.0, 0.0, bad)


def test_code_pins_are_enforced_before_build_and_receipt_replays() -> None:
    lane.validate_code_pins()
    assert all(row["passed"] for row in lane.gates())
    first = lane.build_receipt()
    second = lane.build_receipt()
    assert first == second
    assert first["content_sha256"] == lane._self_hash(first)
    assert first["access_accounting"]["scientific_rows_decoded"] == 0
    assert lane.check_receipt() == first


def test_valid_transform_and_distance_stay_finite() -> None:
    assert np.all(np.isfinite(lane.radec_to_xyz(120.0, 20.0, 10.0)))
    assert all(math.isfinite(value) for value in lane.luminosity_to_comoving_hinv(100.0))
