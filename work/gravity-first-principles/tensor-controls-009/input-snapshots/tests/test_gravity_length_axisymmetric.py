"""Independent Green, Cartesian derivative and smooth source controls."""
from functools import lru_cache

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad

from invariant_gravity_extensions.external_multifield import FluxPoissonSolver
from invariant_gravity_extensions.isolated_axisymmetric import MassComponent, MultipoleGrid
from invariant_gravity_extensions.length_axisymmetric import (
    C3MultipolePotential,
    GreenRadialInterpolator,
    RegularSurfaceDensityDisk,
    full_length_flux,
)
from invariant_gravity_extensions.length_screening import LengthScreening
from invariant_gravity_extensions.reconstructed_axisymmetric import SurfaceDensityDisk
from invariant_gravity_extensions.smooth_spherical_source import spherical_length_anomaly


def test_partial_green_integrals_against_independent_quadrature():
    t = np.linspace(-4, 3, 65)
    f = np.exp(-t[:, None]**2/2)*(1+np.arange(4)[None, :])
    model = GreenRadialInterpolator(t, f)
    for probe in [-4., -3.51, -.23, 1.73, 3.]:
        actual = model.jet(probe)
        for order in range(4):
            inner = quad(lambda v, l=order, q=probe: model.source(v)[l]*np.exp(2*v-(l+1)*(q-v)),
                         t[0], probe, points=t[(t > t[0]) & (t < probe)], epsabs=1e-11, limit=200)[0]
            outer = quad(lambda v, l=order, q=probe: model.source(v)[l]*np.exp(2*v-l*(v-q)),
                         probe, t[-1], points=t[(t > probe) & (t < t[-1])], epsabs=1e-11, limit=200)[0]
            assert actual[0][order] == pytest.approx(-(inner+outer)/(2*order+1), rel=3e-12, abs=1e-12)
            assert actual[1][order] == pytest.approx(((order+1)*inner-order*outer)/(2*order+1), rel=3e-12, abs=1e-12)
        h = 1e-5
        if t[0] < probe < t[-1]:
            for n in [0, 1, 2]:
                numerical = (model.jet(probe+h)[n]-model.jet(probe-h)[n])/(2*h)
                np.testing.assert_allclose(numerical, actual[n+1], atol=2e-9, rtol=2e-8)


@lru_cache
def manufactured_functions(order):
    x, y, z = sp.symbols('x y z', real=True)
    xyz = (x, y, z)
    solid = [sp.Integer(1), z, (2*z*z-x*x-y*y)/2, (2*z**3-3*z*(x*x+y*y))/2][order]
    phi = solid*sp.exp(-(x*x+y*y+z*z)/2)
    p = [sp.diff(phi, c) for c in xyz]
    H = [[sp.diff(phi, c, d) for d in xyz] for c in xyz]
    lap = sum(H[i][i] for i in range(3))
    norm = sum(value**2 for row in H for value in row)
    # Scalar lambdas avoid ambiguous mixed scalar/array lambdified matrices.
    expressions = [phi, *p, *[v for row in H for v in row], *[sp.diff(norm, c) for c in xyz],
                   *[sp.diff(lap, c) for c in xyz], lap]
    return [sp.lambdify((x, y, z), value, 'numpy') for value in expressions]


@pytest.mark.parametrize('order', [0, 2, 3])
def test_third_derivative_invariants_against_manufactured_cartesian_potential(order):
    exact = manufactured_functions(order)

    def source(R, z):
        return exact[-1](R, 0., z)

    model = C3MultipolePotential.build(MultipoleGrid(1e-5, 30, 2049, 32, 5), source)
    R, z = np.array([.2, .5, 1, 2, 3, 0, 0]), np.array([.1, -.3, .8, 1, -2, 1.3, -1.7])
    actual = model.fields(R, z)
    values = np.array([np.broadcast_to(fun(R, 0., z), R.shape) for fun in exact])
    r = np.hypot(R, z)
    s, mu = R/r, z/r
    rotation = np.zeros((3, 3, len(r)))
    rotation[0, 0], rotation[0, 2] = s, mu
    rotation[1, 0], rotation[1, 2], rotation[2, 1] = mu, -s, 1.
    p = np.einsum('ij...,j...->i...', rotation, values[1:4])
    H = np.einsum('ik...,kl...,jl...->ij...', rotation, values[4:13].reshape(3, 3, -1), rotation)
    hn = np.einsum('ij...,j...->i...', rotation, values[13:16])
    dl = np.einsum('ij...,j...->i...', rotation, values[16:19])
    np.testing.assert_allclose(actual['potential'], values[0], atol=3e-8, rtol=2e-6)
    np.testing.assert_allclose(actual['gradient_r_theta'], p[:2], atol=3e-8, rtol=2e-6)
    np.testing.assert_allclose(actual['hessian_rr_rt_tt_pp'], np.array([H[0, 0], H[0, 1], H[1, 1], H[2, 2]]), atol=3e-7, rtol=2e-5)
    np.testing.assert_allclose(actual['gradient_hessian_norm_r_theta'], hn[:2], atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(actual['gradient_laplacian_r_theta'], dl[:2], atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(actual['laplacian'], values[19], atol=3e-7, rtol=3e-5)


def test_green_jets_are_continuous_at_radial_knots_and_small_core_is_stable():
    source = MassComponent('plummer', 1., 0., 1.)
    model = C3MultipolePotential.build(MultipoleGrid(1e-8, 100, 2049, 8, 0), lambda R, z: source.fields(R, z)['laplacian'])
    r = np.geomspace(1e-4, .01, 31)
    actual = model.fields(r, r*0)
    np.testing.assert_allclose(actual['gradient_r_theta'][0], r/(1+r*r)**1.5, rtol=2e-7)
    expected = -15*r/(1+r*r)**3.5
    np.testing.assert_allclose(actual['gradient_laplacian_r_theta'][0], expected, rtol=3e-3, atol=3e-6)
    knots = model.spline.x[::101][1:-1]
    for derivative in [0, 1, 2, 3]:
        left, right = model.spline(knots-1e-10, derivative), model.spline(knots+1e-10, derivative)
        np.testing.assert_allclose(left, right, atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize('length', [0., .03, 1.])
def test_full_length_flux_and_spherical_green_response(length):
    component = MassComponent('plummer', 2., 0., .7)
    grid = MultipoleGrid(1e-5, 1e3, 2049, 8, 0)
    model = C3MultipolePotential.build(grid, lambda R, z: component.fields(R, z)['laplacian'])
    solver = FluxPoissonSolver(grid)
    r, mu = solver.radius[:, None], solver.mu
    fields = model.fields(r*np.sqrt(1-mu*mu), r*mu)
    spec = LengthScreening(1.)
    flux = full_length_flux(fields, spec, length, .2)
    solution = solver.solve(flux)
    probes = np.geomspace(.01, 100, 60)
    g = 2*probes/(probes**2+.7**2)**1.5
    gp = 2*(.7**2-2*probes**2)/(probes**2+.7**2)**2.5
    gpp = 6*probes*(2*probes**2-3*.7**2)/(probes**2+.7**2)**3.5
    expected = spherical_length_anomaly(spec, probes, g, gp, gpp, length, .2)
    measured = -solution.evaluate(probes, probes*0)['acceleration'][0]
    np.testing.assert_allclose(measured, expected, rtol=5e-4, atol=2e-6)


def test_core_join_positive_and_c1_with_analytic_density_derivative():
    radius, surface = np.arange(.25, 8, .5), np.exp(-np.arange(.25, 8, .5))
    old = SurfaceDensityDisk(radius, surface, .3, 7, 1)
    new = RegularSurfaceDensityDisk(radius, surface, .3, 7, 1)
    np.testing.assert_allclose(new.surface(radius), old.surface(radius), atol=1e-15)
    assert new.surface_and_derivative(0.)[1] == 0.
    for point in [radius[0], 6., 7.]:
        left = new.surface_and_derivative(point-1e-8)
        right = new.surface_and_derivative(point+1e-8)
        np.testing.assert_allclose(left, right, atol=2e-8)
    R, z = np.array([0., .1, .25, .9, 3, 6.5, 7]), np.array([.1, -.3, .2, .7, -.1, .2, 0.])
    density, gradient = new.density_and_gradient(R, z)
    assert np.all(density >= 0)
    h = 1e-6
    np.testing.assert_allclose((new.density(R+h, z)-new.density(np.maximum(R-h, 0), z))/(np.where(R == 0, h, 2*h)), gradient[0], atol=3e-6, rtol=1e-4)
    np.testing.assert_allclose((new.density(R, z+h)-new.density(R, z-h))/(2*h), gradient[1], atol=1e-8, rtol=1e-7)


def test_source_partition_and_coordinate_units():
    source = MassComponent('disk', 2., .7, .3)
    grid = MultipoleGrid(1e-4, 100, 1025, 96, 32, .3)
    whole = C3MultipolePotential.build(grid, lambda R, z: source.fields(R, z)['laplacian'])
    split = C3MultipolePotential.build(grid, lambda R, z: .3*source.fields(R, z)['laplacian']+.7*source.fields(R, z)['laplacian'])
    R, z = np.array([.3, 1., 5.]), np.array([.2, -.5, .3])
    a, b = whole.fields(R, z), split.fields(R, z)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-10, rtol=2e-10)
    d = 3.
    scaled_grid = MultipoleGrid(grid.r_min*d, grid.r_max*d, 1025, 96, 32, .3*d)
    # rho transforms as 1/d for the fixed angular source / flux-distance scaling.
    scaled = C3MultipolePotential.build(scaled_grid, lambda R, z: source.fields(R/d, z/d)['laplacian']/d)
    c = scaled.fields(R*d, z*d)
    for key, exponent in [('potential', 1), ('gradient_r_theta', 0), ('hessian_rr_rt_tt_pp', -1),
                           ('gradient_hessian_norm_r_theta', -3), ('gradient_laplacian_r_theta', -2), ('laplacian', -1)]:
        np.testing.assert_allclose(c[key], a[key]*d**exponent, atol=1e-10, rtol=3e-10)
    spec = LengthScreening(1)
    np.testing.assert_allclose(full_length_flux(c, spec, .1, .2), full_length_flux(a, spec, .1/d, .2), atol=1e-10, rtol=1e-9)


def test_invalid_coordinates_and_derivative_requests_fail():
    model = GreenRadialInterpolator(np.linspace(0, 1, 9), np.ones((9, 2)))
    with pytest.raises(ValueError):
        model(-1.)
    with pytest.raises(ValueError):
        model(.5, 4)
    with pytest.raises(ValueError):
        RegularSurfaceDensityDisk([0, 1, 2], [1, 1, 1], .1, 2, 1)
