"""Independent log-source integrals and a large-cancellation derivative witness."""
import mpmath as mp
import numpy as np
from scipy.special import j0

from invariant_gravity_extensions.hankel_midplane import piecewise_gauss
from invariant_gravity_extensions.length_axisymmetric import RegularSurfaceDensityDisk
from invariant_gravity_extensions.precise_tail import PreciseLogGreen, precise_radial_integrals


def disk():
    return RegularSurfaceDensityDisk(np.array([.25, .5, 1., 2., 3.]), np.full(5, 1e10), .2, 3., 1.)


def test_precise_log_source_against_independent_taper_integrals():
    d = disk()
    green = PreciseLogGreen(d)
    c = mp.mp.clone()
    c.dps = 65
    expected_mass = c.mpf('1e10')*(c.mpf('3.25')-1/c.pi**2)
    assert abs(green.mass[-1]-expected_mass) < c.mpf('0.00001')
    for radius in [0., .1, .25, 1., 2.5, 3., 4.]:
        r = c.mpf(radius)

        def integrand(s, r=r):
            taper = 1 if s <= 2 else (1+c.cos(c.pi*(s-2)))/2
            return c.mpf('1e10')*s*taper*c.log(max(r, s))

        edges = sorted({c.mpf(0), c.mpf(2), c.mpf(3), min(r, c.mpf(3))})
        expected = c.quad(integrand, edges)
        _, actual = green.evaluate(radius)
        assert abs(actual-expected) < c.mpf('0.00001')


def test_precise_tail_potential_derivative_under_large_cancellation():
    d = disk()
    k, w = piecewise_gauss(np.arange(0, 20.01, .5), 16)
    r, rw = piecewise_gauss(np.r_[0., d.radius], 64)
    transform = j0(k[:, None]*r)@(rw*r*d.surface(r))
    centers = np.array([3.5, 5., 8.])
    h = 1e-4
    points = np.unique((centers[:, None]+h*np.arange(-2, 3)).ravel())
    value, first = precise_radial_integrals(d, k, w, transform, points, 20., low_k_limit=20.)
    derivative = sum(coefficient*value[np.searchsorted(points, centers+offset*h)]
        for offset, coefficient in [(-2, 1), (-1, -8), (1, 8), (2, -1)])/(12*h)
    # Absolute units matter: a relative test normalized to the large source
    # mass would hide the cancellation error in this tiny potential tail.
    np.testing.assert_allclose(derivative, first[np.searchsorted(points, centers)], atol=1e-4, rtol=1e-9)
