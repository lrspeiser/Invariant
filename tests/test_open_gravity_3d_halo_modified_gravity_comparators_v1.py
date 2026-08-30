from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_3d_halo_modified_gravity_comparators_v1 as comp
from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = comp.load_config()
    suite = comp.run_suite(config)
    return config, suite


def test_config_and_committed_bindings(packet: tuple) -> None:
    config, _suite = packet
    comp.validate_config(config)
    assert [row["role"] for row in config["bindings"]] == [
        "NEWTON_AQUAL_QUMOND_BASELINES",
        "THEORY_CARDS",
    ]
    assert all(len(row["commit"]) == 40 for row in config["bindings"])


def test_exact_comparator_inventory(packet: tuple) -> None:
    config, _suite = packet
    assert tuple(row["id"] for row in config["comparators"]) == comp._COMPARATOR_IDS
    assert len(config["primary_sources"]) == 6
    assert len(config["blocked_comparators"]) == 4


def test_three_halo_profiles_are_distinct_and_positive(packet: tuple) -> None:
    _config, suite = packet
    metrics = suite["gates"]["HALO_POSITIVITY_MONOTONICITY_AND_MASS"]["metrics"]
    assert set(metrics) == {"HALO_NFW", "HALO_BURKERT", "HALO_EINASTO"}
    assert all(row["mass_within_unit_radius"] > 0.0 for row in metrics.values())
    assert len({row["mass_within_unit_radius"] for row in metrics.values()}) == 3


def test_nfw_enclosed_mass_matches_numerical_integral(packet: tuple) -> None:
    _config, suite = packet
    result = suite["gates"]["NFW_ANALYTIC_ENCLOSED_MASS"]
    assert result["passed"] is True
    assert result["metrics"]["relative_error"] < 1.0e-10


def test_nfw_central_cell_uses_exact_volume_average() -> None:
    grid = base.make_grid(13)
    density = comp.halo_density_on_grid("HALO_NFW", grid)
    centre = tuple(size // 2 for size in grid.shape)
    radius = grid.spacing / 2.0
    expected = comp.nfw_enclosed_mass(radius) / (4.0 * math.pi * radius**3 / 3.0)
    assert density[centre] == expected


def test_cusp_and_core_are_not_equivalence_collapsed(packet: tuple) -> None:
    _config, suite = packet
    metrics = suite["gates"]["HALO_CUSP_CORE_DISTINCTION"]["metrics"]
    assert metrics["HALO_NFW"] > metrics["HALO_BURKERT"]
    assert metrics["HALO_EINASTO"] > metrics["HALO_BURKERT"]


def test_halo_poisson_solution_has_small_residual(packet: tuple) -> None:
    _config, suite = packet
    result = suite["gates"]["POISSON_HALO_RESIDUAL"]
    assert result["passed"] is True
    assert result["metrics"]["relative_residual"] < 1.0e-11


def test_helmholtz_manufactured_solution_is_exact_to_roundoff(packet: tuple) -> None:
    _config, suite = packet
    metrics = suite["gates"]["HELMHOLTZ_MANUFACTURED"]["metrics"]
    assert metrics["relative_residual"] < 1.0e-11
    assert metrics["maximum_solution_error"] < 1.0e-11


def test_mog_point_kernel_has_newton_and_enhanced_limits(packet: tuple) -> None:
    config, suite = packet
    alpha = config["numerical_contract"]["mog_parameters"]["alpha"]
    metrics = suite["gates"]["MOG_POINT_KERNEL_LIMITS"]["metrics"]
    assert metrics["near_ratio"] == pytest.approx(1.0, abs=1.0e-10)
    assert metrics["far_ratio"] == pytest.approx(1.0 + alpha, abs=1.0e-10)


def test_mog_solver_is_three_dimensional_and_rotation_covariant(packet: tuple) -> None:
    _config, suite = packet
    result = suite["gates"]["MOG_ROTATION_COVARIANCE"]
    assert result["passed"] is True
    assert result["metrics"]["relative_rotation_error"] < 1.0e-12


def test_mog_zero_alpha_is_exact_newton_control() -> None:
    grid = base.make_grid(9)
    density = np.exp(-4.0 * (grid.x**2 + 2.0 * grid.y**2 + 3.0 * grid.z**2))
    zero = np.zeros(grid.shape)
    mog, newton, _yukawa = comp.solve_mog_weak_field(density, zero, grid.spacing, alpha=0.0, mu=0.7)
    assert np.array_equal(mog.potential, newton.potential)


def test_refracted_constant_permittivity_limit(packet: tuple) -> None:
    _config, suite = packet
    result = suite["gates"]["REFRACTED_CONSTANT_EPSILON_LIMIT"]
    assert result["passed"] is True
    assert result["metrics"]["relative_error"] < 1.0e-12


def test_refracted_gravity_preserves_geometry_dependence(packet: tuple) -> None:
    _config, suite = packet
    metrics = suite["gates"]["REFRACTED_DISK_SPHERE_GEOMETRY_DISCRIMINATOR"]["metrics"]
    assert metrics["absolute_gain_difference"] > 1.0e-3
    assert metrics["disk_potential_gain"] != metrics["sphere_potential_gain"]


def test_refracted_solver_is_rotation_covariant(packet: tuple) -> None:
    _config, suite = packet
    result = suite["gates"]["REFRACTED_ROTATION_COVARIANCE"]
    assert result["passed"] is True
    assert result["metrics"]["relative_rotation_error"] < 1.0e-12


def test_permittivity_is_bounded_and_monotonic() -> None:
    density = np.geomspace(1.0e-6, 1.0e6, 1000)
    epsilon = comp.permittivity(density, epsilon_0=0.2, rho_c=1.0, q_slope=1.5)
    assert np.all(epsilon >= 0.2)
    assert np.all(epsilon <= 1.0)
    assert np.all(np.diff(epsilon) >= 0.0)
    assert epsilon[-1] > epsilon[0]


def test_invalid_parameters_fail_closed(packet: tuple) -> None:
    _config, suite = packet
    result = suite["gates"]["DESIGNED_NEGATIVE_PARAMETER_FAILURES"]
    assert result["metrics"] == {"rejected": 8, "attempted": 8}


@pytest.mark.parametrize(
    "call",
    (
        lambda: comp._profile_density("HALO_NFW", 1.0, density_scale=0.0),
        lambda: comp._profile_density("HALO_BURKERT", 1.0, radius_scale=-1.0),
        lambda: comp._profile_density("HALO_EINASTO", 1.0, alpha=-0.1),
        lambda: comp.mog_point_acceleration_ratio(0.0, alpha=1.0, mu=1.0),
        lambda: comp.permittivity(np.ones(1), epsilon_0=1.1, rho_c=1.0, q_slope=1.0),
    ),
)
def test_public_parameter_guards(call) -> None:
    with pytest.raises(comp.ComparatorError):
        call()


def test_all_twelve_declared_gates_pass_once(packet: tuple) -> None:
    config, suite = packet
    assert list(suite["gates"]) == config["required_target_free_gates"]
    assert suite["passed"] == 12
    assert suite["failed"] == 0
    assert all(row["passed"] is True for row in suite["gates"].values())


def test_full_covariant_and_unimplemented_comparators_stay_blocked(packet: tuple) -> None:
    config, _suite = packet
    blocked = {row["id"] for row in config["blocked_comparators"]}
    assert blocked == {
        "FULL_COVARIANT_MOG",
        "FULL_COVARIANT_REFRACTED_GRAVITY",
        "PUBLISHED_NONLOCAL_GRAVITY",
        "EMOND",
    }


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "bindings",
        "primary_sources",
        "comparators",
        "numerical_contract",
        "required_target_free_gates",
        "blocked_comparators",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple, section: str) -> None:
    config, _suite = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(comp.ComparatorError, match="config semantics changed"):
        comp.validate_config(changed)


def test_noncanonical_receipt_path_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(comp, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(comp, "_read_json", forbidden)
    with pytest.raises(comp.ComparatorError, match="output path changed"):
        comp.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple) -> None:
    _config, _suite = packet
    receipt = comp.build_receipt()
    comp.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["claim_boundary"]["does_not_establish"] = []
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = comp.content_sha256(body)
    with pytest.raises(comp.ComparatorError, match="not reproducible"):
        comp.validate_receipt_payload(forged)


def test_zero_access_and_narrow_claim(packet: tuple) -> None:
    config, _suite = packet
    receipt = comp.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert "a halo fit" in config["claim_boundary"]["does_not_establish"]
    assert "full covariant theory health" in config["claim_boundary"]["does_not_establish"]
