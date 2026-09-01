from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_disk_polar_escape_load_v1 as lane


def test_existing_registry_gap_and_formula_are_explicit() -> None:
    config = lane.load_config()
    assert set(config["existing_formula_audit"]) == {
        "GQNS",
        "REFRACTED_GRAVITY",
        "VOID_LOAD_VQ00_VQ10",
        "TWELL_400",
        "QUANTUM_ENTITY_WAVE_ATLAS",
        "DIFFERENTIAL_PROPAGATION_KERNEL",
    }
    assert [row["id"] for row in config["formula_catalog"]] == [
        "DPEL01_DISK_POLAR_ESCAPE_LOAD",
        "DGKT01_DIRECTIONAL_GRAVITY_KINETIC_TRANSPORT",
        "GRRAD00_STATIC_FIELD_VS_RADIATION_CONTROL",
    ]
    law = config["law"]
    assert "D_parallel" in law["diffusion_tensor"]
    assert "D_perp" in law["diffusion_tensor"]
    assert "v_g=-chi" in law["drift"]
    assert "beta rho_b/rho_star" in law["evolution"]


def test_dimension_ledger_closes_the_pde_and_observables() -> None:
    ledger = lane.load_config()["dimension_ledger"]
    assert ledger["U_Q"] == "L^2 T^-2"
    assert ledger["D_parallel_D_perp_D_iso_chi"] == "L^2 T^-1"
    assert ledger["v_g"] == "L T^-1"
    assert ledger["G_rho_b_D_iso"] == "L^2 T^-3"
    assert ledger["eta_Q_integral_U_Q_dell_over_c3"] == "1"


def test_sphere_switches_off_orientation_and_disk_finds_normal() -> None:
    disk, sphere, _ = lane._fixture_density()
    disk_shape = lane.disk_shape_from_density(disk, 0.25)
    sphere_shape = lane.disk_shape_from_density(sphere, 0.25)
    assert disk_shape["activation"] > 0.75
    assert abs(abs(disk_shape["normal"][2]) - 1.0) < 1e-12
    assert sphere_shape["activation"] < 1e-12
    tensor = lane.diffusion_tensor(sphere_shape["normal"], sphere_shape["activation"], 0.5, 1.5)
    assert tensor == pytest.approx(np.eye(3) * np.trace(tensor) / 3.0, abs=1e-12)


def test_bulge_reduces_directional_activation() -> None:
    disk, _, bulged = lane._fixture_density()
    assert (
        0.0
        < lane.disk_shape_from_density(bulged, 0.25)["activation"]
        < lane.disk_shape_from_density(disk, 0.25)["activation"]
    )


def test_tensor_is_positive_trace_preserving_and_isotropic_at_zero_activation() -> None:
    tensor = lane.diffusion_tensor([0.0, 0.0, 1.0], 0.8, 0.5, 1.5)
    assert np.all(np.linalg.eigvalsh(tensor) > 0.0)
    assert np.trace(tensor) == pytest.approx(2.5)
    isotropic = lane.diffusion_tensor([0.0, 0.0, 1.0], 0.0, 0.5, 1.5)
    assert isotropic == pytest.approx(np.eye(3) * (2.5 / 3.0))
    equal = lane.diffusion_tensor([1.0, 2.0, 3.0], 1.0, 0.7, 0.7)
    assert equal == pytest.approx(np.eye(3) * 0.7)


def test_so3_covariance_uses_projector_not_normal_sign() -> None:
    rng = np.random.default_rng(14021)
    points = rng.normal(size=(250, 3)) @ np.diag([1.4, 0.9, 0.18])
    weights = np.exp(-np.sum(points**2, axis=1) / 4.0)
    first = lane.disk_shape_from_points(points, weights)
    rotation = lane._rotation_matrix()
    second = lane.disk_shape_from_points(points @ rotation.T, weights)
    d1 = lane.diffusion_tensor(first["normal"], first["activation"], 0.5, 1.5)
    d2 = lane.diffusion_tensor(second["normal"], second["activation"], 0.5, 1.5)
    assert second["activation"] == pytest.approx(first["activation"], abs=2e-15)
    assert d2 == pytest.approx(rotation @ d1 @ rotation.T, abs=2e-14)


def test_constant_coefficient_green_has_polar_signature_and_reversal() -> None:
    tensor = lane.diffusion_tensor([0.0, 0.0, 1.0], 1.0, 0.5, 1.5)
    polar = lane.anisotropic_green([0.0, 0.0, 1.0], tensor, 0.2)
    equatorial = lane.anisotropic_green([1.0, 0.0, 0.0], tensor, 0.2)
    assert polar > equatorial
    reverse = lane.diffusion_tensor([0.0, 0.0, 1.0], 1.0, 1.5, 0.5)
    assert lane.anisotropic_green([0.0, 0.0, 1.0], reverse, 0.2) < lane.anisotropic_green(
        [1.0, 0.0, 0.0], reverse, 0.2
    )
    assert lane.anisotropic_green_residual([0.8, -0.5, 0.7], tensor, 0.2) < 2e-6


def test_periodic_rhs_closes_source_sink_budget() -> None:
    disk, _, _ = lane._fixture_density(size=11, spacing=0.35)
    shape = lane.disk_shape_from_density(disk, 0.35)
    tensor = lane.diffusion_tensor(shape["normal"], shape["activation"], 0.5, 1.5)
    load = 0.07 + 0.03 * disk
    rhs, budget = lane.load_rhs(load, disk, 0.35, tensor, 0.08, 1.0, 0.2, 0.35)
    assert np.all(np.isfinite(rhs))
    assert budget["periodic_budget_relative_residual"] < 2e-14


def test_zero_source_is_exact_zero_and_absorption_reduces_load() -> None:
    disk, _, _ = lane._fixture_density(size=11, spacing=0.35)
    shape = lane.disk_shape_from_density(disk, 0.35)
    tensor = lane.diffusion_tensor(shape["normal"], shape["activation"], 0.5, 1.5)
    common = {
        "spacing": 0.35,
        "tensor": tensor,
        "chi": 0.04,
        "source_gain": 1.0,
        "gamma0": 0.2,
        "dt": 0.002,
        "steps": 80,
    }
    zero, zero_summary = lane.integrate_load(np.zeros_like(disk), beta=0.35, **common)
    weak, weak_summary = lane.integrate_load(disk, beta=0.35, **common)
    strong, strong_summary = lane.integrate_load(disk, beta=1.2, **common)
    assert np.max(np.abs(zero)) == 0.0
    assert zero_summary["total_load"] == 0.0
    assert strong_summary["total_load"] < weak_summary["total_load"]
    assert np.min(weak) >= 0.0 and np.min(strong) >= 0.0


def test_same_state_drives_matter_and_path_observables() -> None:
    axis = np.linspace(-2.0, 2.0, 17)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    load = np.exp(-(x**2 + y**2 + z**2))
    acceleration = lane.matter_acceleration_from_load(load, axis[1] - axis[0])
    redshift = lane.photon_path_log_redshift(load[8, 8, :], axis[1] - axis[0], 0.03)
    assert acceleration.shape == (3, 17, 17, 17)
    assert np.max(np.abs(acceleration)) > 0.0
    assert redshift > 0.0


def test_kinetic_angular_moments_separate_density_flux_and_pressure() -> None:
    directions, weights = lane.cartesian_angular_quadrature()
    isotropic = np.full((6, 2, 3, 4), 2.5)
    scalar, flux, pressure = lane.angular_moments(isotropic, directions, weights, 3.0)
    assert scalar == pytest.approx(np.full((2, 3, 4), 2.5))
    assert flux == pytest.approx(np.zeros((3, 2, 3, 4)), abs=1e-15)
    assert pressure == pytest.approx(
        np.eye(3).reshape(3, 3, 1, 1, 1) * scalar * 3.0,
        abs=1e-14,
    )
    pencil = np.zeros_like(isotropic)
    pencil[4] = 6.0
    q_beam, f_beam, p_beam = lane.angular_moments(pencil, directions, weights, 3.0)
    assert f_beam[2] == pytest.approx(3.0 * q_beam)
    assert p_beam[2, 2] == pytest.approx(9.0 * q_beam)


def test_kinetic_transport_closes_exact_zeroth_moment() -> None:
    directions, weights = lane.cartesian_angular_quadrature()
    grid = np.indices((5, 5, 5), dtype=float)
    source = 0.2 + 0.03 * (grid[0] + grid[1] + grid[2])
    distribution = np.stack(
        [0.4 + source * (1.0 + 0.05 * ray) + 0.01 * grid[ray % 3] for ray in range(6)]
    )
    rhs, budget = lane.kinetic_transport_rhs(
        distribution,
        source,
        0.4,
        directions,
        weights,
        carrier_speed=1.7,
        source_gain=0.8,
        absorption0=0.1,
        absorption_baryonic=0.25,
        collision_rates=0.6,
    )
    assert np.all(np.isfinite(rhs))
    assert budget["zeroth_moment_max_relative_residual"] < 2e-15


def test_disk_aligned_collision_derives_registered_diffusion_and_drift_moment() -> None:
    normal = np.array([0.3, -0.4, 0.8660254037844386])
    directions, weights = lane.disk_aligned_angular_quadrature(normal)
    assert directions[4] == pytest.approx(normal / np.linalg.norm(normal), abs=2e-15)
    rates, summary = lane.disk_aligned_collision_rates(0.72, 0.5, 1.5, 2.0)
    assert 2.0**2 / (3.0 * rates[0]) == pytest.approx(summary["effective_parallel_diffusion"])
    assert 2.0**2 / (3.0 * rates[4]) == pytest.approx(summary["effective_perpendicular_diffusion"])
    assert summary["effective_perpendicular_diffusion"] > summary["effective_parallel_diffusion"]
    grid = np.indices((5, 5, 5), dtype=float)
    source = 0.2 + 0.01 * np.sum(grid, axis=0)
    distribution = np.stack([0.6 + source * (1.0 + 0.03 * ray) for ray in range(6)])
    drift = lane.density_gradient_drift(source, 0.4, 0.07)
    _, budget = lane.kinetic_transport_rhs(
        distribution,
        source,
        0.4,
        directions,
        weights,
        carrier_speed=2.0,
        source_gain=0.3,
        absorption0=0.1,
        absorption_baryonic=0.2,
        collision_rates=rates,
        drift_velocity=drift,
    )
    assert budget["zeroth_moment_max_relative_residual"] < 3e-15


def test_markov_collision_is_positive_at_registered_boundary_state() -> None:
    directions, weights = lane.cartesian_angular_quadrature()
    rates, _ = lane.disk_aligned_collision_rates(0.931207, 0.5, 1.5, 1.0)
    population = np.zeros((6, 3, 3, 3))
    population[4] = 6.0
    rhs, budget = lane.kinetic_transport_rhs(
        population,
        np.zeros((3, 3, 3)),
        1.0,
        directions,
        weights,
        carrier_speed=1.0,
        source_gain=0.0,
        absorption0=0.0,
        absorption_baryonic=0.0,
        collision_rates=rates,
    )
    assert np.min(rhs[[0, 1, 2, 3, 5]]) >= 0.0
    assert budget["minimum_off_diagonal_transition_rate"] >= 0.0
    assert budget["weighted_collision_source_sum"] == pytest.approx(0.0, abs=2e-15)


def test_static_quadrupole_is_not_radiation_and_carrier_energy_is_variable() -> None:
    static = np.broadcast_to(np.diag([2.0, -1.0, -1.0]), (20, 3, 3)).copy()
    assert np.max(lane.quadrupole_radiation_measure(static, 0.25)) == 0.0
    time = np.arange(20) * 0.25
    changing = np.sin(time)[:, None, None] * np.diag([2.0, -1.0, -1.0])
    assert np.max(lane.quadrupole_radiation_measure(changing, 0.25)) > 0.0
    assert lane.carrier_quantum_energy(2.0) / lane.carrier_quantum_energy(0.5) == 4.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_and_invalid_inputs_fail_closed(bad: float) -> None:
    with pytest.raises(lane.DiskPolarEscapeError):
        lane.diffusion_tensor([0.0, 0.0, 1.0], bad, 0.5, 1.5)
    with pytest.raises(lane.DiskPolarEscapeError):
        lane.diffusion_tensor([0.0, 0.0, 1.0], 0.5, bad, 1.5)
    with pytest.raises(lane.DiskPolarEscapeError):
        lane.anisotropic_green([1.0, 0.0, 0.0], np.eye(3), bad)
    with pytest.raises(lane.DiskPolarEscapeError):
        lane.photon_path_log_redshift([0.0, 1.0], 1.0, bad)


def test_source_predecessor_is_public_source_only_and_response_remains_blocked() -> None:
    config = lane.load_config()
    preflight = config["source_and_response_preflight"]
    assert preflight["source_status"] == "EXACT_PUBLIC_SOURCE_PREDECESSOR_AVAILABLE"
    assert preflight["source_objects"] == ["NGC2903", "NGC3351", "NGC3627"]
    assert preflight["response_status"].startswith("SOURCE_BLOCKED")
    assert preflight["scientific_response_rows_opened"] == 0
    assert preflight["scores"] == 0
    assert lane._bind_source_predecessor(config)


def test_target_free_suite_is_deterministic_and_all_checks_pass() -> None:
    first = lane.target_free_suite()
    second = lane.target_free_suite()
    assert first == second
    assert all(row["passed"] for row in first["checks"])
    assert first["constant_coefficient_signature"]["polar_over_equatorial"] > 1.0
    assert first["constant_coefficient_signature"]["reversed_polar_over_equatorial"] < 1.0
    assert first["kinetic_signature"]["static_quadrupole_radiation_measure"] == 0.0
    assert first["kinetic_signature"]["zeroth_moment_max_relative_residual"] < 2e-15
    assert first["kinetic_signature"]["empty_ray_minimum_collision_derivative"] >= 0.0
    assert first["kinetic_signature"]["finite_six_ray_fourth_moment_rotation_error"] > 1e-6


def test_receipt_rebuild_self_hash_and_zero_access() -> None:
    receipt = lane.build_receipt()
    assert receipt == lane.build_receipt()
    assert receipt["content_sha256"] == lane._self_hash(receipt)
    assert receipt["formula_ids"] == [
        "DPEL01_DISK_POLAR_ESCAPE_LOAD",
        "DGKT01_DIRECTIONAL_GRAVITY_KINETIC_TRANSPORT",
        "GRRAD00_STATIC_FIELD_VS_RADIATION_CONTROL",
    ]
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert receipt["source_and_response_preflight"]["scientific_response_rows_opened"] == 0


def test_config_mutations_do_not_validate_as_the_frozen_contract() -> None:
    config = lane.load_config()
    forged = copy.deepcopy(config)
    forged["registered_parameterization"]["dimensionless_target_free"]["D_perp_over_D0"] = 99.0
    assert lane.content_sha256(forged) != lane._CONFIG_CONTENT_SHA256
