"""Prospective unit/analytic controls for the compatibility-first successor.

These are implementation tests, not an independently authored discovery audit.
No astronomical file or historical response loader is imported.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from invariant_gravity_extensions.actions import (
    ActionSpec,
    H,
    X,
    Y,
    Z,
    action_certificates,
    generate_specs,
)
from invariant_gravity_extensions.cli import main, read_config, run_demo
from invariant_gravity_extensions.dynamics import InertiaMemory, evolve_auxiliary
from invariant_gravity_extensions.fields import PeriodicGrid, joint_density, solve_fields
from invariant_gravity_extensions.observables import (
    UnsupportedSectorError,
    assumed_metric,
    born_lensing,
    member_relative_acceleration,
    require_supported_sector,
)
from invariant_gravity_extensions.policy import (
    CompatibilityPolicy,
    assess_compatibility,
    next_stage,
    rank_experiments,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gravity_extension_discovery_v1.json"


@pytest.fixture
def grid():
    return PeriodicGrid(9, 10.0)


@pytest.fixture
def density(grid):
    xyz = grid.coordinates()
    r = np.exp(-np.sum((xyz/np.array([1.3, 0.9, 0.7])[:, None, None, None])**2, axis=0))
    return r-r.mean()


def test_identical_prediction_survives_without_improvement():
    y = np.arange(12)*0.01
    b = y+0.03
    result = assess_compatibility(y, b, b, [str(i//2) for i in range(12)],
                                  CompatibilityPolicy(resamples=100))
    assert result["status"] == "COMPATIBLE"
    assert result["delta_rms_dex"] == 0
    assert result["one_sided_upper_dex"] == 0
    assert not result["baseline_improvement_required"]
    assert not result["discovery_claim_allowed"]
    assert next_stage("COMPATIBLE", False) == "RETAIN_CURRENT_OBSERVABLE_EQUIVALENCE"
    assert next_stage("COMPATIBLE", True).startswith("ELIGIBLE_FOR_SEPARATELY")
    assert next_stage("COMPATIBLE", True, False) == "UNSUPPORTED_BY_CURRENT_SCORER"


def test_bad_candidate_does_not_hide_behind_nonsignificance():
    y = np.zeros(6)
    result = assess_compatibility(y, y, y+1, list("abcdef"),
                                  CompatibilityPolicy(resamples=100))
    assert result["status"] == "INCOMPATIBLE"
    assert result["one_sided_lower_dex"] == 1


@pytest.mark.parametrize("role", ["confirmation", "sealed", "validation", "heldout"])
def test_compatibility_cannot_authorize_protected_data(role):
    with pytest.raises(ValueError, match="authorize"):
        assess_compatibility(np.zeros(3), np.zeros(3), np.zeros(3), list("abc"), role=role)


@pytest.mark.parametrize("kwargs", [{"margin_dex": -1}, {"confidence": 1},
                                  {"resamples": 0}, {"seed": -1}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        CompatibilityPolicy(**kwargs)


def test_object_weighting_not_radial_point_weighting():
    y = np.zeros(3)
    b = np.array([0.01, 0.02, 0.03])
    c = np.array([0.02, 0.04, 0.01])
    policy = CompatibilityPolicy(resamples=100)
    a = assess_compatibility(y, b, c, list("abc"), policy)
    r = np.array([0]*100+[1, 2])
    z = assess_compatibility(y[r], b[r], c[r], [list("abc")[i] for i in r], policy)
    for key in ("baseline_rms_dex", "candidate_rms_dex", "one_sided_upper_dex"):
        assert z[key] == pytest.approx(a[key], abs=1e-14)


def test_active_design_projects_covariant_nuisance():
    p = {"normalization": np.array([[0., 0.], [1., 1.]]),
         "shape": np.array([[0., 0.], [1., -1.]])}
    rows = rank_experiments(p, np.array([[1., 0.3], [0.3, 1.]]), np.ones((2, 1)))
    assert rows[0]["experiment"] == "shape"
    assert rows[1]["utility"] < 1e-25
    with pytest.raises(np.linalg.LinAlgError):
        rank_experiments(p, np.ones((2, 2)))


def test_symbolic_variations_not_only_values():
    result = action_certificates()
    assert result["all_pass"]
    assert result["value_only_negative_control_rejected"]
    assert result["trimond_collinear_physical_flux_residual"] == "0"
    assert result["deep_Q_coefficient"] == "4/3"
    assert result["high_Q_derivative"] == "1"


def test_catalog_hashes_and_zero_amplitude_deduplication():
    cfg = read_config(CONFIG)
    specs = generate_specs(cfg["grammar"])
    assert len(specs) == 15
    assert len({s.card()["content_sha256"] for s in specs}) == 15
    assert specs[0].family == "qumond"
    assert all(not s.card()["historical_novelty_claimed"] for s in specs)
    assert specs[0].card() == ActionSpec("qumond").card()


@pytest.mark.parametrize("spec", [{"family": "bogus"}, {"family": "qumond", "beta": 1},
                                  {"family": "gqumond_length", "length": -1},
                                  {"family": "qumond", "epsilon": 0}])
def test_invalid_action_parameters(spec):
    with pytest.raises(ValueError):
        ActionSpec(**spec)


def test_periodic_poisson_manufactured_solution(grid):
    xyz = grid.coordinates()
    k = 2*np.pi/grid.length
    truth = np.cos(k*xyz[0])*np.sin(2*k*xyz[1])
    solved = grid.poisson(-5*k*k*truth)
    assert np.max(np.abs(solved-truth)) < 1e-13
    assert abs(solved.mean()) < 1e-14
    with pytest.raises(ValueError, match="density contrast"):
        grid.poisson(np.ones(grid.shape))
    with pytest.raises(ValueError, match="odd"):
        PeriodicGrid(8)


def test_discrete_integration_by_parts(grid):
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(2, *grid.shape))
    for axis in range(3):
        assert np.sum(a*grid.derivative(b, axis)) == pytest.approx(
            -np.sum(grid.derivative(a, axis)*b), abs=1e-10)


def test_null_extensions_reduce_to_identical_qumond(grid, density):
    base = solve_fields(grid, density, ActionSpec("qumond"))
    for spec in [ActionSpec("gqumond_length", length=0),
                 ActionSpec("trimond_alignment", mixing=0, beta=3)]:
        candidate = solve_fields(grid, density, spec)
        assert np.allclose(candidate.physical, base.physical, atol=2e-13, rtol=1e-12)


@pytest.mark.parametrize("spec", [ActionSpec("qumond"),
                                  ActionSpec("trimond_alignment", .6, .7),
                                  ActionSpec("gqumond_length", length=.4)])
def test_action_euler_derivative_matches_physical_source(grid, density, spec):
    sol = solve_fields(grid, density, spec, tolerance=1e-11)
    # A smooth perturbation independent of the action-derivative implementation.
    d = np.sin(
        2*np.pi*grid.coordinates()[0]/grid.length)
    fn = sp.lambdify((X, Y, Z, H), spec.expression(), "numpy")
    q = grid.gradient(sol.auxiliary) if sol.auxiliary is not None else np.zeros((3, *grid.shape))

    def energy(psi):
        p = grid.gradient(psi)
        x = np.sum(p*p, axis=0)
        y = np.sum(q*q, axis=0)
        z = 2*np.sum(p*q, axis=0)
        h = spec.length**2*np.sum(grid.hessian(psi)**2, axis=(0, 1))
        return np.sum(fn(x, y, z, h))*grid.dx**3

    delta = 1e-6
    measured = (energy(sol.newtonian+delta*d)-energy(sol.newtonian-delta*d))/(2*delta)
    expected = -2*np.sum(grid.laplacian(sol.physical)*d)*grid.dx**3
    assert measured == pytest.approx(expected, rel=5e-5, abs=5e-6)


def test_gqumond_requires_double_divergence(grid, density):
    spec = ActionSpec("gqumond_length", length=.7)
    sol = solve_fields(grid, density, spec)
    p = grid.gradient(sol.newtonian)
    h = spec.length**2*np.sum(grid.hessian(sol.newtonian)**2, axis=(0, 1))
    px, _, _, _ = spec.partials(np.sum(p*p, axis=0), h=h)
    wrong = grid.poisson(grid.divergence(px[None]*p))
    assert np.linalg.norm(wrong-sol.physical) > 1e-3


def test_auxiliary_is_not_trivial_in_nonspherical_system(grid, density):
    base = solve_fields(grid, density, ActionSpec("qumond"))
    sol = solve_fields(grid, density, ActionSpec("trimond_alignment", .8, .7))
    assert sol.diagnostics["auxiliary_relative_residual"] < 2e-8
    assert np.linalg.norm(sol.physical-base.physical) > 1e-7
    # Final physical acceleration is curl-free, not necessarily the intermediate flux.
    a = sol.acceleration
    curl = grid.derivative(a[0], 1)-grid.derivative(a[1], 0)
    assert np.max(np.abs(curl)) < 1e-12


def test_rotation_covariance_axis_permutation(grid, density):
    spec = ActionSpec("trimond_alignment", .5, .8)
    original = solve_fields(grid, density, spec)
    rotated = solve_fields(grid, np.transpose(density, (2, 0, 1)), spec)
    assert np.allclose(rotated.physical, np.transpose(original.physical, (2, 0, 1)), atol=2e-10)


def test_source_partition_keeps_field_but_not_nonlinear_superposition(grid):
    xyz = grid.coordinates()
    a = np.exp(-np.sum((xyz-np.array([1., 0, 0])[:, None, None, None])**2, axis=0))
    b = np.exp(-np.sum((xyz+np.array([1., 0, 0])[:, None, None, None])**2, axis=0))
    r1, meta = joint_density(grid, {"a": a, "b": b}, subtract_background=True)
    r2, _ = joint_density(grid, {"a1": a/2, "a2": a/2, "b": b}, subtract_background=True)
    assert np.allclose(r1, r2, atol=1e-15)
    assert meta["component_ids"] == ["a", "b"]
    spec = ActionSpec("qumond")
    full = solve_fields(grid, r1, spec).physical
    wrong = solve_fields(grid, a-a.mean(), spec).physical+solve_fields(grid, b-b.mean(), spec).physical
    assert np.linalg.norm(wrong-full) > .01


def test_member_com_subtraction_removes_uniform_acceleration(grid, density):
    sol = solve_fields(grid, density, ActionSpec("qumond"))
    member = density-density.min()+1e-6
    a, c = member_relative_acceleration(sol, member)
    b, d = member_relative_acceleration(sol, member,
                                      uniform_external_acceleration=np.array([4., -2., 9.]))
    assert np.allclose(a, b, atol=1e-13)
    assert np.allclose(d-c, [4, -2, 9])


def test_lensing_explicit_opt_in_and_same_metric(grid, density):
    sol = solve_fields(grid, density, ActionSpec("qumond"))
    with pytest.raises(UnsupportedSectorError):
        assumed_metric(sol)
    metric = assumed_metric(sol, closure="assumed_no_slip")
    predicted = born_lensing(metric, distance_factor=2)
    assert not predicted["metadata"]["derived_from_action"]
    assert predicted["shear_1"].shape == (grid.n, grid.n)
    assert np.allclose(metric.psi, metric.phi)
    with pytest.raises(UnsupportedSectorError):
        require_supported_sector("isolated_nested_cluster")


def test_born_manufactured_slab(grid, density):
    sol = solve_fields(grid, density, ActionSpec("qumond"))
    k = 2*np.pi/grid.length
    metric = assumed_metric(sol, closure="assumed_no_slip", speed_of_light=20)
    metric.psi = np.cos(k*grid.coordinates()[0])
    metric.phi = metric.psi.copy()
    result = born_lensing(metric, distance_factor=3)
    expected = -3*grid.length*k*k*np.cos(k*grid.coordinates()[0][:, :, 0])/20**2
    assert np.allclose(result["convergence"], expected, atol=1e-13)
    assert np.allclose(result["shear_1"], expected, atol=1e-13)
    assert np.max(np.abs(result["shear_2"])) < 1e-13


def test_wave_no_future_driver_leak(grid):
    t = np.linspace(0, 2, 21)
    drive = np.zeros((len(t), *grid.shape))
    drive[10:] = 1
    q, _ = evolve_auxiliary(grid, t, drive, np.zeros(grid.shape), np.zeros(grid.shape))
    assert np.max(np.abs(q[:11])) == 0
    assert np.mean(q[-1]) == pytest.approx(1-np.cos(1), abs=1e-12)
    drive[-1] = 1e6
    r, _ = evolve_auxiliary(grid, t, drive, np.zeros(grid.shape), np.zeros(grid.shape))
    assert np.array_equal(q, r)


def test_wave_zero_mode_and_free_oscillator(grid):
    t = np.array([0., .2, .7])
    drive = np.ones((3, *grid.shape))
    q, v = evolve_auxiliary(grid, t, drive, np.zeros(grid.shape), np.zeros(grid.shape), mass=0)
    assert np.allclose(q[-1], .5*.7**2)
    assert np.allclose(v[-1], .7)
    q, v = evolve_auxiliary(grid, t, drive*0, np.ones(grid.shape), np.zeros(grid.shape), mass=2)
    assert np.allclose(q[-1], np.cos(1.4))
    assert np.allclose(v[-1], -2*np.sin(1.4))


def test_worldline_null_and_energy():
    t = np.linspace(0, 8, 161)
    initial = (np.array([1., 0.]), np.array([0., .8]), np.zeros(2), np.zeros(2))
    base = InertiaMemory().integrate(lambda t, x: -x, t, *initial)
    assert np.max(np.abs(base[:, 0, 0]-np.cos(t))) < 2e-8
    assert np.max(np.abs(base[:, 0, 1]-.8*np.sin(t))) < 2e-8
    model = InertiaMemory(.4, 1., 2.)
    trajectory = model.integrate(lambda t, x: -x, t, *initial)
    energy = model.energy(trajectory, lambda x: float(x@x/2))
    assert np.max(np.abs(energy-energy[0])) < 2e-8
    assert np.max(np.abs(trajectory[:, 0]-base[:, 0])) > .01
    with pytest.raises(ValueError, match="kinetic"):
        InertiaMemory(1, 1, 2)


def test_module_import_has_no_data_or_network_effects():
    code = """
import builtins, socket
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected observation/network access')
socket.create_connection = forbidden
builtins.open = forbidden
import invariant_gravity_extensions
assert invariant_gravity_extensions.__version__ == '0.1.0'
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_config_cannot_enable_real_data(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    cfg["access"]["observational_inputs"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="authorize"):
        read_config(path)


def test_demo_end_to_end_and_deterministic():
    cfg = read_config(CONFIG)
    cfg["synthetic"].update(grid_n=9, scene_count=3)
    cfg["compatibility"]["resamples"] = 100
    cfg["grammar"].update(mixing=[.4], beta=[.5], powers=[1], lengths=[.3])
    a = run_demo(cfg, include_lensing=True)
    b = run_demo(cfg, include_lensing=True)
    assert a == b
    assert a["candidates"][0]["compatibility"]["status"] == "COMPATIBLE"
    assert a["action_certificates"]["all_pass"]
    assert a["confirmation_accesses"] == 0
    assert a["dynamics_controls"]["relative_energy_drift"] < 1e-7
    assert a["dynamics_controls"]["auxiliary_pre_drive_max"] == 0
    assert len(a["lensing"]) == 9


def test_cli_append_only_catalog(tmp_path):
    output = tmp_path / "run"
    assert main(["catalog", "--config", str(CONFIG), "--output", str(output)]) == 0
    first = (output / "result.json").read_bytes()
    assert main(["catalog", "--config", str(CONFIG), "--output", str(output)]) == 2
    assert (output / "result.json").read_bytes() == first
    assert not (output / "failure.json").exists()


@pytest.mark.parametrize("kwargs", [{"power": 1.5}, {"power": True}])
def test_discrete_action_powers_are_not_silently_rounded(kwargs):
    with pytest.raises(ValueError):
        ActionSpec("trimond_alignment", mixing=.3, **kwargs)


def test_empty_source_and_gqumond_rotated_scene(grid):
    for spec in [ActionSpec("qumond"), ActionSpec("trimond_alignment", .5, .2),
                 ActionSpec("gqumond_length", length=.5)]:
        sol = solve_fields(grid, np.zeros(grid.shape), spec)
        assert np.max(np.abs(sol.physical)) == 0
    x = grid.coordinates()
    rho = np.sin(2*np.pi*x[0]/grid.length) + .3*np.cos(4*np.pi*x[2]/grid.length)
    spec = ActionSpec("gqumond_length", length=.4)
    first = solve_fields(grid, rho, spec)
    second = solve_fields(grid, np.transpose(rho, (1, 2, 0)), spec)
    assert np.allclose(second.physical, np.transpose(first.physical, (1, 2, 0)), atol=1e-12)


def test_auxiliary_failure_is_not_a_physical_result(grid, density):
    with pytest.raises(RuntimeError, match="converge"):
        solve_fields(grid, density, ActionSpec("trimond_alignment", .8, 100),
                     tolerance=1e-13, maxiter=1)


def test_config_refuses_fractional_grid_and_seed(tmp_path):
    for key, val in [("grid_n", 9.5), ("scene_count", 3.5), ("seed", 1.1)]:
        cfg = json.loads(CONFIG.read_text())
        cfg["synthetic"][key] = val
        path = tmp_path / f"bad-{key}.json"
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError):
            read_config(path)


def test_invalid_light_state_and_worldline_tolerance(grid, density):
    sol = solve_fields(grid, density, ActionSpec("qumond"))
    metric = assumed_metric(sol, closure="assumed_no_slip")
    metric.speed_of_light = 0
    with pytest.raises(ValueError, match="light speed"):
        born_lensing(metric, distance_factor=1)
    with pytest.raises(ValueError, match="tolerances"):
        InertiaMemory().integrate(lambda t, x: -x, np.array([0., 1.]),
                                  np.ones(1), np.zeros(1), np.zeros(1), np.zeros(1), rtol=np.nan)


def test_cli_retains_failure_and_never_writes_success_receipt(tmp_path, monkeypatch):
    from invariant_gravity_extensions import cli

    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(cli, "run_demo", fail)
    output = tmp_path / "failed-run"
    assert main(["demo", "--config", str(CONFIG), "--output", str(output)]) == 2
    assert (output / "started.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "NUMERICAL_OR_INPUT_BLOCK_NOT_THEORY_REJECTION"
    assert not (output / "receipt.json").exists()


def test_cli_quarantines_config_changed_mid_run(tmp_path, monkeypatch):
    from invariant_gravity_extensions import cli

    config = tmp_path / "config.json"
    config.write_bytes(CONFIG.read_bytes())

    def mutate(*args, **kwargs):
        config.write_bytes(config.read_bytes()+b"\n")
        return {"claim_ceiling": "TEST"}

    monkeypatch.setattr(cli, "run_demo", mutate)
    output = tmp_path / "quarantined"
    assert main(["demo", "--config", str(config), "--output", str(output)]) == 2
    assert "changed during execution" in (output / "failure.json").read_text()
    assert not (output / "result.json").exists()
