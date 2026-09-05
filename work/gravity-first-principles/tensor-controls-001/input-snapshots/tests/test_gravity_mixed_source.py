"""Independent Gaussian Hankel identities and direct omitted-tail integrals."""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import jvp

from invariant_gravity_extensions.hankel_axisymmetric import cylindrical_jet
from invariant_gravity_extensions.hankel_midplane import piecewise_gauss
from invariant_gravity_extensions.mixed_source import hankel_mixed_jet, leading_tail_mixed_jet


def test_all_sixteen_hankel_partials_against_exact_gaussian_integral():
    R, Z = sp.symbols('R Z', real=True)
    r, z = np.array([0., .07, .6, 1.3, 3.]), np.array([-.7, 0., .2, 1.4])
    k, w = piecewise_gauss(np.linspace(0, 16, 33), 24)
    transform = k*np.exp(-k*k/2)
    vertical = np.array([np.broadcast_to(sp.lambdify(Z, sp.diff(sp.exp(-Z**2/2), Z, j), 'numpy')(z)[:, None],
        (len(z), len(k))) for j in range(4)])[None]
    actual = hankel_mixed_jet(k, w, transform[None], vertical, r, z, .7, batch_size=2)
    expression = -2*sp.pi*sp.Rational(7, 10)*sp.exp(-(R**2+Z**2)/2)
    expected = np.array([[sp.lambdify((R, Z), sp.diff(expression, R, i, Z, j), 'numpy')(r[:, None], z[None, :])
        for j in range(4)] for i in range(4)])
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=3e-12)
    np.testing.assert_array_equal(actual[1::2, :, 0], 0.)
    np.testing.assert_array_equal(actual[:, 1::2, :, 1], 0.)
    old = cylindrical_jet(k, w, transform[None], vertical, r, z, .7)
    for i, j, reference in [(0, 0, old['potential']), (1, 0, old['gradient_R_z'][0]),
                          (2, 1, old['third_RRR_RRz_Rzz_zzz_Rpp_zpp'][1])]:
        np.testing.assert_allclose(actual[i, j], reference, rtol=2e-13, atol=2e-13)


def test_all_sixteen_tail_partials_against_independent_integrals():
    class Disk:
        radius = np.array([.25, 1., 2., 4., 8., 12.])
        surface_density = np.exp(-radius**2/2)
        outer_radius, taper_width, core_coefficient, height = 12., 0., -.5, .4

        def surface(self, r):
            return np.exp(-np.asarray(r)**2/2)

        def surface_and_derivative(self, r):
            return self.surface(r), -np.asarray(r)*self.surface(r)

    r, z = np.array([0., .3, 1.7]), np.array([-.3, 0., .2])
    k, w = piecewise_gauss(np.linspace(0, 3, 13), 32)
    actual, _ = leading_tail_mixed_jet({'disk': Disk()}, ['disk'], k, w, np.exp(-k*k/2)[None], r, z, 1., 3.)
    Z = sp.Symbol('Z', real=True)
    f = 1/(2*sp.Rational(2, 5)*sp.cosh(Z/sp.Rational(2, 5))**2)
    expected = np.empty_like(actual)
    for i in range(4):
        radial = np.array([quad(lambda q, rr=rr, i=i: np.exp(-q*q/2)*q**(i-1)*jvp(0, q*rr, i),
            3, np.inf, epsabs=1e-13, epsrel=1e-12)[0] for rr in r])
        for j in range(4):
            vertical = sp.lambdify(Z, sp.diff(f, Z, j), 'numpy')(z)
            expected[i, j] = -4*np.pi*radial[:, None]*vertical
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=3e-10)
