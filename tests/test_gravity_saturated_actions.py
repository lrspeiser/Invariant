import numpy as np
import pytest

from invariant_gravity_extensions.actions import ActionSpec
from invariant_gravity_extensions.fields import PeriodicGrid, solve_fields
from invariant_gravity_extensions.local_limits import Orbit, perihelion_first_order
from invariant_gravity_extensions.saturated_actions import (
    SaturatedActionSpec,
    saturated_certificates,
)


@pytest.mark.parametrize("shape", [0.5, 1, 2])
def test_saturated_exact_limits_and_variations(shape):
    assert saturated_certificates(shape)["all_pass"]


@pytest.mark.parametrize("shape", [0.5, 1, 2])
def test_stable_response_matches_action_and_scaling(shape):
    spec = SaturatedActionSpec("qumond", shape=shape)
    y = np.logspace(-5, 2, 100)
    np.testing.assert_allclose(spec.partials(y*y)[0] - 1, spec.delta_nu(y), rtol=1e-5, atol=4e-14)
    small = SaturatedActionSpec("qumond", shape=shape, epsilon=1e-12)
    assert small.delta_nu(1e-7) * np.sqrt(1e-7) == pytest.approx(1, rel=1e-5)
    assert spec.delta_nu(1e6) * 1e6**(2*(1+shape)) == pytest.approx(1, rel=5e-6)
    assert np.all(np.isfinite(spec.delta_nu(np.array([1e-200, 1e100]))))


def test_new_kernel_cards_do_not_alias_old_or_other_shapes():
    old = ActionSpec("qumond").card()
    new = [SaturatedActionSpec("qumond", shape=m).card() for m in (0.5, 1, 2)]
    assert len({old["content_sha256"], *(c["content_sha256"] for c in new)}) == 4
    assert "kernel" not in old
    assert all(c["historical_novelty_claimed"] is False for c in new)


@pytest.mark.parametrize("family", ["qumond", "trimond_alignment"])
def test_saturated_discrete_action_variation_matches_field_equation(family):
    grid = PeriodicGrid(9, 10)
    xyz = grid.coordinates()
    rho = np.exp(-np.sum((xyz/np.array([1.4, 1, .8])[:, None, None, None])**2, axis=0))
    rho -= rho.mean()
    spec = (SaturatedActionSpec(family, shape=1) if family == "qumond" else
            SaturatedActionSpec(family, .75, .5, shape=1))
    sol = solve_fields(grid, rho, spec)
    direction = np.random.default_rng(82).normal(size=grid.shape)
    direction -= direction.mean()
    q = np.zeros((3, *grid.shape)) if sol.auxiliary is None else grid.gradient(sol.auxiliary)

    def action(psi):
        import sympy as sp

        from invariant_gravity_extensions.actions import H, X, Y, Z
        p = grid.gradient(psi)
        expression = sp.lambdify((X, Y, Z, H), spec.expression(), "numpy")
        return np.sum(expression(np.sum(p*p, axis=0), np.sum(q*q, axis=0),
                                 2*np.sum(p*q, axis=0), 0)) * grid.dx**3

    step = 1e-6
    measured = (action(sol.newtonian+step*direction)-action(sol.newtonian-step*direction))/(2*step)
    expected = -2*np.sum(grid.laplacian(sol.physical)*direction)*grid.dx**3
    assert measured == pytest.approx(expected, rel=2e-5, abs=2e-6)


def test_auxiliary_interaction_and_rotation_survive_saturation():
    grid = PeriodicGrid(9, 10)
    xyz = grid.coordinates()
    rho = np.exp(-np.sum((xyz/np.array([2, 1, .8])[:, None, None, None])**2, axis=0))
    rho -= rho.mean()
    spec = SaturatedActionSpec("trimond_alignment", .75, 2, shape=1)
    base = solve_fields(grid, rho, SaturatedActionSpec("qumond", shape=1))
    sol = solve_fields(grid, rho, spec)
    assert np.linalg.norm(sol.physical-base.physical) > 1e-6
    rotated = solve_fields(grid, np.transpose(rho, (2, 0, 1)), spec)
    np.testing.assert_allclose(rotated.physical, np.transpose(sol.physical, (2, 0, 1)), atol=1e-10)


def test_solar_tail_is_small_but_does_not_authorize_complete_local_pass():
    orbit = Orbit(9.53667594 * 149597870700, .05386179, 1.32712440041279419e20)
    for shape in (.5, 1, 2):
        spec = SaturatedActionSpec("qumond", shape=shape)
        angle = perihelion_first_order(orbit, 1.2e-10, spec.delta_nu)
        assert abs(angle) < 1e-15
        assert "local_gravity_constraints" in spec.card()["open_obligations"]


@pytest.mark.parametrize("kwargs", [{"shape": 3}, {"family": "gqumond_length", "length": .1}])
def test_unimplemented_kernel_variants_refused(kwargs):
    with pytest.raises(ValueError):
        SaturatedActionSpec(**({"family": "qumond"} | kwargs))
