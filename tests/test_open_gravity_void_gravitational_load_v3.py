from __future__ import annotations

import math

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_void_gravitational_load_v3 as lane


def test_vq05_d0_is_the_finite_time_ivp_not_instant_equilibrium() -> None:
    rho = np.array([[[0.2, 0.8]]])
    source = 1.2 * rho
    result = lane.solve_parabolic_reservoir(rho, source, diffusivity=0.0, gamma0=0.5, gamma_b=0.8, dx=1.0, dt=0.01, steps=100)
    gamma = 0.5 + 0.8 * rho
    expected = source / gamma * (1.0 - np.exp(-gamma))
    assert np.array_equal(result, expected)
    assert not np.array_equal(result, source / gamma)


def test_vq06_self_kernel_is_the_declared_tensor_average() -> None:
    first = lane.self_cell_yukawa_average(1.6, 1.0, 10)
    second = lane.self_cell_yukawa_average(1.6, 1.0, 10)
    assert first == second and first > 0.0
    with pytest.raises(lane.VoidLoadV3Error):
        lane.self_cell_yukawa_average(1.6, 1.0, 9)


def test_vq06_midpoint_column_is_three_dimensional_and_rotation_covariant() -> None:
    rho = np.zeros((5, 5, 5))
    rho[1, 1, 3] = 2.0
    rho[4, 3, 1] = 1.0
    kwargs = {
        "amplitude": 1.1,
        "length": 1.6,
        "sigma": 0.25,
        "dx": 1.0,
        "midpoint_samples": 12,
        "self_order": 10,
    }
    field = lane.column_attenuated_feed(rho, **kwargs)
    rotated = lane.column_attenuated_feed(np.rot90(rho, axes=(1, 2)), **kwargs)
    assert np.allclose(rotated, np.rot90(field, axes=(1, 2)), atol=1e-13)


def test_vq08_analytic_sphere_intervals_union_and_partition() -> None:
    spheres = [(np.array([3.0, 0.0, 0.0]), 2.0), (np.array([5.0, 0.0, 0.0]), 2.0), (np.array([9.0, 0.6, 0.0]), 1.0)]
    intervals = lane.ray_sphere_intervals(np.array([1.0, 0.0, 0.0]), 12.0, spheres)
    assert intervals[0] == (1.0, 7.0)
    assert intervals[1] == pytest.approx((8.2, 9.8))
    partition = lane.path_partition(np.array([1.0, 0.0, 0.0]), 12.0, spheres, [(0.5, 10.0)])
    assert partition["L_void"] == pytest.approx(7.6)
    assert partition["L_observed_matter"] == pytest.approx(1.9)
    assert partition["L_unobserved"] == pytest.approx(2.5)
    assert sum(partition[key] for key in ("L_void", "L_observed_matter", "L_unobserved")) == pytest.approx(12.0)


def test_only_delta_h_survives_and_sign_units_are_explicit() -> None:
    logz, velocity = lane.identifiable_void_prediction(0.02, 7.6, 1.0)
    assert logz == pytest.approx(0.152)
    assert velocity == pytest.approx(0.152)
    assert logz > 0.0


def test_catalog_velocity_is_mapped_to_log_redshift() -> None:
    c = 299792.458
    assert lane.observed_log_redshift(0.0, c) == 0.0
    assert lane.observed_log_redshift(3000.0, c) == pytest.approx(math.log1p(3000.0 / c))


def test_1pgc_split_bytes_and_byte_order_are_exact() -> None:
    assert lane.canonical_1pgc_bytes(12345) == b"12345"
    assert lane.split_bucket(12345) == lane.split_bucket("12345")
    with pytest.raises(lane.VoidLoadV3Error):
        lane.canonical_1pgc_bytes("012345")
    digest = lane.hashlib.sha256(b"12345").digest()
    assert lane.split_bucket(12345)[0] == int.from_bytes(digest[:8], "big") % 10


def test_all_repair_gates_and_deterministic_receipt() -> None:
    config = lane.load_config()
    gates = lane.synthetic_gates(config)
    assert len(gates) == 8 and all(row["passed"] for row in gates)
    first, first_payloads = lane.build_receipt()
    second, second_payloads = lane.build_receipt()
    assert first == second and first_payloads == second_payloads
    assert first["content_sha256"] == lane._self_hash(first)
    assert first["access_accounting"]["scientific_rows_decoded"] == 0
    assert lane.check_package() == first
