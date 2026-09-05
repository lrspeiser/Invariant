import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from invariant_gravity_extensions.cluster_pressure import (
    MU,
    PROTON_MASS,
    PowerLawDensity,
    boundary_residual_covariance,
    covariance_loss,
    integrate_electron_pressure,
    load_development_packet,
    pressure_indices,
)


@pytest.mark.parametrize("power", [0, -2, -3, -4.2])
def test_source_mass_against_independent_quadrature(power):
    nodes = np.array([1, 2, 4, 8.])
    density = 3*nodes**power
    source = PowerLawDensity(nodes, density)
    radii = np.array([.2, 1, 1.7, 3.4, 7.9])
    _, mass = source.evaluate(radii)
    expected = [4*np.pi*quad(lambda r: 3*(1 if r < 1 else r**power)*r*r, 0, end,
                             points=[1] if end > 1 else None, epsabs=1e-10)[0] for end in radii]
    np.testing.assert_allclose(mass, expected, rtol=1e-11)
    joined = np.unique(np.concatenate([radii, [1.1, 2.4, 5]]))
    np.testing.assert_allclose(source.evaluate(joined)[1][np.searchsorted(joined, radii)], mass, rtol=1e-14)
    with pytest.raises(ValueError, match="support"):
        source.evaluate(np.array([9]))


@pytest.mark.parametrize("outer_fraction", [0, .15, .3])
def test_hydrostatic_pressure_fraction_against_exact_solution_and_ode(outer_fraction):
    r = np.linspace(1, 3, 501)
    f = outer_fraction*r/3
    n = np.ones_like(r)/(MU*PROTON_MASS)
    pressure, k = integrate_electron_pressure(r, n, np.ones_like(r), f, 2)
    exact = (1-f)*(2/(1-outer_fraction)+3-r)
    np.testing.assert_allclose(pressure, exact, rtol=1e-14)
    assert k[-1] == 1
    ode = solve_ivp(lambda x, p: [-(1-outer_fraction*x/3)-p[0]*(outer_fraction/3)/(1-outer_fraction*x/3)],
                    [3, 1], [2], t_eval=r[::-1], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(pressure, ode.y[0][::-1], rtol=2e-10)
    if outer_fraction:
        gradient_fraction_answer = 2+(3-r)-outer_fraction*(9-r*r)/6
        assert np.max(abs(pressure/gradient_fraction_answer-1)) > .02


def test_boundary_covariance_includes_correlations_and_boundary_noise():
    c = np.array([[4, 1, 1], [1, 3, .5], [1, .5, 2.]])
    k = np.array([1.2, 1.1])
    predicted = boundary_residual_covariance(c, [0, 1], 2, k)
    noise = np.random.default_rng(421).multivariate_normal(np.zeros(3), c, 150000)
    empirical = np.cov((noise[:, :2]-k*noise[:, 2, None]).T)
    np.testing.assert_allclose(predicted, empirical, rtol=.02)
    assert not np.allclose(predicted, c[:2, :2])
    with pytest.raises(ValueError):
        boundary_residual_covariance(c, [0, 2], 2, k)


def test_covariance_loss_whitens_and_is_unit_invariant():
    c = np.array([[4, 1], [1, 2.]])
    residual = np.array([2, 3.])
    expected = residual@np.linalg.solve(c, residual)/2
    assert covariance_loss(residual, c) == pytest.approx(expected)
    assert covariance_loss(residual*1e-10, c*1e-20) == pytest.approx(expected)


def test_reserved_cluster_rejected_before_any_file_access(tmp_path):
    with pytest.raises(PermissionError, match="reserved"):
        load_development_packet(tmp_path, "A2029", {}, {})


def test_radius_and_boundary_gate_is_independent_of_pressure_values():
    packet = {"density_radius_kpc": np.array([.1, 8.]), "pressure_radius_kpc": np.arange(1, 11.)}
    ids, anchor, rows = pressure_indices(packet)
    np.testing.assert_array_equal(ids, [3, 4, 5, 6])
    assert anchor == 7
    assert rows[0]["status"] == "EXCLUDED_PUBLISHED_INNER_BEAM_LIMIT"
    assert rows[-1]["status"] == "OUTSIDE_MEASURED_DENSITY_SUPPORT"
    packet["pressure"] = np.full(10, np.nan)
    np.testing.assert_array_equal(pressure_indices(packet)[0], ids)


def test_invalid_nonthermal_fraction_is_not_clipped():
    with pytest.raises(ValueError):
        integrate_electron_pressure([1, 2], [1, 1], [1, 1], [.2, 1.1], 2)
