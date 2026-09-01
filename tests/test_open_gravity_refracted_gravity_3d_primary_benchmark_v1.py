from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_3d_primary_benchmark_v1 as benchmark,
)


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = benchmark.load_config(verify_package=False)
    suite = benchmark.run_suite(config)
    return config, suite


def test_config_is_primary_paper_bound_and_target_free(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    benchmark.validate_config(config)
    assert [row["id"] for row in config["primary_sources"]] == [
        "RG_ORIGINAL_2016",
        "RG_DISKMASS_2021_V2",
        "RG_REVIEW_2024",
    ]
    assert config["benchmark_contract"]["response_tuning"] is False
    assert config["admission_rule"]["missing_public_source_or_paper"].startswith("SOURCE_BLOCKED")


def test_predecessor_commit_and_all_artifacts_are_byte_bound(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    benchmark.validate_predecessor(config)
    binding = config["predecessor_binding"]
    assert binding["commit"] == "009c2e0d0c7d29b1ce0e00d776f7b8b7d246ccc4"
    assert len(binding["artifacts"]) == 4


def test_published_permittivity_matches_equation_five_directly() -> None:
    density = np.geomspace(1.0e-8, 1.0e8, 301)
    epsilon_0 = 0.661
    q_slope = 1.79
    expected = epsilon_0 + (1.0 - epsilon_0) * (1.0 + np.tanh(q_slope * np.log(density))) / 2.0
    actual = benchmark.published_permittivity(
        density, epsilon_0=epsilon_0, rho_c=1.0, q_slope=q_slope
    )
    assert np.array_equal(actual, expected)


def test_exact_published_median_and_prior_corner_inventory(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    cells = benchmark.published_parameter_cells(config)
    assert len(cells) == 9
    assert len({row["id"] for row in cells}) == 9
    assert cells[0] == {
        "id": "DISKMASS_UNIVERSAL_MEDIAN",
        "epsilon_0": 0.661,
        "Q": 1.79,
        "log10_rho_c_g_cm3": -24.54,
    }
    assert sum(row["id"].startswith("PRIOR_CORNER_") for row in cells) == 8


def test_permittivity_bounds_transition_and_derivative(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    derivative = suite["gates"]["PERMITTIVITY_BOUNDS_MONOTONICITY_AND_DERIVATIVE"]
    limits = suite["gates"]["LOW_HIGH_DENSITY_AND_TRANSITION_LIMITS"]
    assert derivative["passed"] is True
    assert derivative["metrics"]["finite_difference_relative_error"] < 1.0e-8
    assert limits["metrics"]["transition_epsilon"] == pytest.approx((1.0 + 0.661) / 2.0)


def test_epsilon_one_and_constant_epsilon_limits_are_exact(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    newton = suite["gates"]["NEWTON_EPSILON_ONE_LIMIT"]["metrics"]
    constant = suite["gates"]["CONSTANT_EPSILON_FIELD_SCALE"]["metrics"]
    assert newton["relative_potential_error"] < 1.0e-13
    assert constant["field_enhancement"] == 2.5
    assert constant["relative_potential_error"] < 1.0e-12


def test_variable_coefficient_manufactured_solution_converges_second_order(
    packet: tuple[dict, dict],
) -> None:
    config, suite = packet
    metrics = suite["gates"]["VARIABLE_COEFFICIENT_MANUFACTURED_SECOND_ORDER"]["metrics"]
    errors = [row["maximum_solution_error"] for row in metrics["grids"]]
    assert errors == sorted(errors, reverse=True)
    assert metrics["observed_order"] >= config["benchmark_contract"]["required_order"]
    assert errors[-1] <= config["benchmark_contract"]["maximum_finest_manufactured_error"]


def test_cartesian_solver_recovers_published_spherical_gauss_law(
    packet: tuple[dict, dict],
) -> None:
    config, suite = packet
    metrics = suite["gates"]["SPHERICAL_GAUSS_LAW_SECOND_ORDER"]["metrics"]
    errors = [row["relative_potential_error"] for row in metrics["grids"]]
    assert errors == sorted(errors, reverse=True)
    assert metrics["observed_order"] >= config["benchmark_contract"]["required_order"]
    assert errors[-1] <= config["benchmark_contract"]["maximum_finest_spherical_potential_error"]


def test_nonspherical_density_gradient_term_is_not_local_rescaling(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    metrics = suite["gates"]["NONSPHERICAL_DENSITY_GRADIENT_TERM_ACTIVE"]["metrics"]
    assert metrics["maximum_abs_grad_epsilon_dot_grad_phi"] > 1.0e-4
    assert metrics["relative_difference_from_pointwise_newton_rescale"] > 1.0e-3


def test_operator_is_rotation_covariant(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["gates"]["ROTATION_COVARIANCE"]["metrics"]
    assert metrics["relative_rotation_error"] < 1.0e-12


def test_all_published_parameter_cells_remain_positive_elliptic(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    gate = suite["gates"]["POSITIVE_ELLIPTICITY_PUBLISHED_SENSITIVITY_GRID"]
    assert gate["metrics"]["count"] == 9
    assert all(row["positive_elliptic"] is True for row in gate["metrics"]["cells"])


def test_published_adverse_evidence_is_retained_without_pruning(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    metrics = suite["gates"]["PUBLISHED_COUNTEREVIDENCE_RETAINED"]["metrics"]
    assert metrics["count"] == 3
    assert metrics["pruning_from_published_failure"] is False
    assert "LOW_ACCELERATION_RAR_UNDERESTIMATE" in metrics["ids"]


@pytest.mark.parametrize(
    "call",
    (
        lambda: benchmark.published_permittivity(np.ones(1), epsilon_0=0.0, rho_c=1.0, q_slope=1.0),
        lambda: benchmark.published_permittivity(
            np.ones(1), epsilon_0=0.5, rho_c=-1.0, q_slope=1.0
        ),
        lambda: benchmark.published_permittivity(
            np.ones(1), epsilon_0=0.5, rho_c=1.0, q_slope=math.inf
        ),
        lambda: benchmark.published_permittivity(
            np.array([-1.0]), epsilon_0=0.5, rho_c=1.0, q_slope=1.0
        ),
    ),
)
def test_public_parameter_guards_fail_closed(call) -> None:
    with pytest.raises(benchmark.RefractedGravityBenchmarkError):
        call()


def test_all_declared_gates_pass_once(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert list(suite["gates"]) == config["required_gates"]
    assert suite["passed"] == 14
    assert suite["failed"] == 0
    assert all(row["passed"] is True for row in suite["gates"].values())


def test_zero_response_and_no_scoring_authority(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert suite["gates"]["ZERO_RESPONSE_ACCESS"]["passed"] is True
    access = config["access_contract"]
    assert access["scientific_response_files_opened"] == 0
    assert access["scientific_rows_opened"] == 0
    assert access["scores_computed"] == 0
    claims = config["claim_boundary"]
    assert claims["real_galaxy_or_cluster_fit"] is False
    assert claims["observational_preference_established"] is False
    assert claims["publication_ready"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("status",), "CONFIRMED"),
        (("published_parameters", "universal_diskmass_median", "epsilon_0"), 0.2),
        (("benchmark_contract", "required_order"), 0.0),
        (("benchmark_contract", "response_tuning"), True),
        (("claim_boundary", "observational_preference_established"), True),
        (("access_contract", "scientific_response_files_opened"), 1),
    ),
)
def test_semantic_config_mutations_are_rejected(
    packet: tuple[dict, dict], path: tuple[str, ...], replacement: object
) -> None:
    config, _suite = packet
    forged = copy.deepcopy(config)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(benchmark.RefractedGravityBenchmarkError):
        benchmark.validate_config(forged)


def test_receipt_rebuild_and_coherent_forgery_rejection() -> None:
    receipt = benchmark.build_receipt()
    benchmark.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["claim_boundary"]["observational_preference_established"] = True
    forged["content_sha256"] = benchmark.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(benchmark.RefractedGravityBenchmarkError):
        benchmark.validate_receipt_payload(forged)
