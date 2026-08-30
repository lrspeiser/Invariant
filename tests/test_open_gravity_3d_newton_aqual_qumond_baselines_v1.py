from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as baseline


@pytest.fixture(scope="module")
def config() -> dict:
    return baseline.load_config()


@pytest.fixture(scope="module")
def suite(config: dict) -> dict:
    return baseline.run_suite(config)


def test_config_and_foundation_are_exactly_bound(config: dict) -> None:
    baseline.validate_config(config)
    assert config["foundation_binding"]["commit"] == "95a8d3d85bc1f064d3dd654e7c430d07786113c9"
    assert len(config["foundation_binding"]["artifacts"]) == 4


def test_simple_mu_nu_are_spherical_inverse_pair() -> None:
    x = np.geomspace(1e-2, 1e3, 100)
    y = x * baseline.mu_simple(x)
    reconstructed_x = y * baseline.nu_simple(y)
    assert np.allclose(reconstructed_x, x, rtol=2e-13, atol=2e-13)


def test_grid_requires_odd_size_and_exposes_cartesian_shape() -> None:
    grid = baseline.make_grid(9)
    assert grid.shape == (9, 9, 9)
    assert math.isclose(grid.spacing, 0.25)
    with pytest.raises(baseline.BaselineSolverError):
        baseline.make_grid(8)


def test_newton_manufactured_gate(suite: dict) -> None:
    assert suite["checks"]["B3D01_NEWTON_MANUFACTURED"]
    assert suite["metrics"]["newton_manufactured_relative_error"] < 2e-6
    assert suite["metrics"]["newton_linear_relative_residual"] < 1e-10


def test_aqual_manufactured_gate(suite: dict) -> None:
    assert suite["checks"]["B3D02_AQUAL_MANUFACTURED"]
    assert suite["metrics"]["aqual_manufactured_relative_error"] < 2e-6
    assert suite["metrics"]["aqual_relative_residual"] < 2e-7


def test_spherical_limit_is_close_but_not_identity_linked(suite: dict) -> None:
    value = suite["metrics"]["spherical_aqual_qumond_relative_difference"]
    assert suite["checks"]["B3D03_SPHERE"]
    assert 0.0 < value < 0.2


def test_shell_and_pair_symmetry(suite: dict) -> None:
    assert suite["checks"]["B3D04_SHELL"]
    assert suite["checks"]["B3D07_PAIR_SADDLE"]
    assert suite["metrics"]["shell_centre_acceleration"] < 1e-10
    assert suite["metrics"]["pair_centre_acceleration"] < 1e-10


def test_nonspherical_disk_distinguishes_aqual_from_qumond(suite: dict) -> None:
    assert suite["checks"]["B3D05_DISK"]
    assert suite["metrics"]["nonspherical_disk_aqual_qumond_relative_difference"] > 1e-5


def test_grid_rotation_covariance(suite: dict) -> None:
    assert suite["checks"]["B3D06_BAR_TRIAXIAL"]
    assert suite["metrics"]["bar_rotation_relative_error"] < 1e-9


def test_external_field_changes_internal_solution(suite: dict) -> None:
    assert suite["checks"]["B3D08_EXTERNAL_FIELD"]
    assert suite["metrics"]["external_field_internal_relative_change"] > 1e-5
    assert suite["metrics"]["qumond_external_field_internal_relative_change"] > 1e-5


def test_high_acceleration_limit_returns_to_newton(suite: dict) -> None:
    assert suite["checks"]["B3D09_HIGH_ACCELERATION"]
    assert suite["metrics"]["high_acceleration_aqual_newton_relative_difference"] < 0.02
    assert suite["metrics"]["high_acceleration_qumond_newton_relative_difference"] < 0.02


def test_three_grid_error_strictly_improves(suite: dict) -> None:
    assert suite["checks"]["B3D10_RESOLUTION"]
    errors = suite["metrics"]["resolution_relative_errors"]
    assert errors["17"] < errors["13"] < errors["9"]


def test_all_ten_frozen_gates_pass(suite: dict) -> None:
    assert suite["all_pass"] is True
    assert suite["passed"] == suite["total"] == 10
    assert suite["metrics"]["qumond_linear_relative_residual_max"] < 1e-10


def test_aqual_does_not_call_qumond_builder(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("QUMOND path called")

    monkeypatch.setattr(baseline, "_qumond_flux_divergence", forbidden)
    grid = baseline.make_grid(7)
    exact = (
        np.cos(math.pi * grid.x / 2) * np.cos(math.pi * grid.y / 2) * np.cos(math.pi * grid.z / 2)
    )
    gradient = baseline._gradient(exact, grid.spacing)
    coefficient = baseline.mu_simple(np.sqrt(sum(value * value for value in gradient)))
    rhs = baseline._variable_divergence(exact, coefficient, grid.spacing)
    result = baseline.solve_aqual(
        rhs,
        exact,
        grid.spacing,
        **baseline._aqual_options(config),
    )
    assert result.converged


def test_qumond_does_not_call_aqual_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> baseline.SolverResult:
        raise AssertionError("AQUAL path called")

    monkeypatch.setattr(baseline, "solve_aqual", forbidden)
    grid = baseline.make_grid(7)
    density = baseline._gaussian(grid, [(0.0, 0.0, 0.0)], [(0.2, 0.2, 0.2)], [1.0])
    zero = baseline._zero_boundary(grid)
    newton, qumond, _rhs = baseline.solve_qumond(4.0 * math.pi * density, zero, zero, grid.spacing)
    assert newton.converged and qumond.converged


@pytest.mark.parametrize(
    "section",
    (
        "foundation_binding",
        "primary_sources",
        "equation_contract",
        "normalized_units",
        "grid_contract",
        "boundary_contract",
        "fixture_contract",
        "gate_contract",
        "independence_contract",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_frozen_section_rejects_mutation(config: dict, section: str) -> None:
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(baseline.BaselineSolverError, match=f"section {section} changed"):
        baseline.validate_config(changed)


def test_coherent_authority_mutation_is_rejected(config: dict) -> None:
    changed = copy.deepcopy(config)
    changed["access_contract"]["scientific_response_files"] = 1
    changed["section_sha256"]["access_contract"] = baseline.content_sha256(
        changed["access_contract"]
    )
    with pytest.raises(baseline.BaselineSolverError, match="access contract changed"):
        baseline.validate_config(changed)


def test_noncanonical_receipt_path_rejects_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attacker = tmp_path / "response.json"
    attacker.write_text('{"secret":1}', encoding="utf-8")
    reads = 0

    def forbidden_read(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(baseline, "OUTPUT_PATH", attacker)
    monkeypatch.setattr(baseline, "_read_json", forbidden_read)
    with pytest.raises(baseline.BaselineSolverError, match="output path changed"):
        baseline.validate_receipt()
    assert reads == 0


def test_receipt_is_deterministic_and_forgery_fails(config: dict) -> None:
    first = baseline.build_receipt()
    second = baseline.build_receipt()
    assert first == second
    baseline.validate_receipt_payload(first)
    forged = copy.deepcopy(first)
    forged["status"] = "OBSERVATIONALLY_CONFIRMED"
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = baseline.content_sha256(body)
    with pytest.raises(baseline.BaselineSolverError, match="not reproducible"):
        baseline.validate_receipt_payload(forged)


def test_zero_access_and_narrow_claim(config: dict) -> None:
    receipt = baseline.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert "fit to any galaxy or cluster" in receipt["claim_boundary"]["does_not_establish"]
    assert config["boundary_contract"]["response_fitted_boundary_forbidden"] is True


def test_config_bytes_are_json_and_no_response_path_is_registered(config: dict) -> None:
    raw = (baseline._ROOT / baseline.CONFIG_PATH).read_text(encoding="utf-8")
    assert json.loads(raw) == config
    lowered = raw.lower()
    assert "vobs" not in lowered
    assert "pressure.csv" not in lowered
    assert "temperature.csv" not in lowered
