"""Independent Gaussian integrals and derivative checks of tail completion."""
import numpy as np
from scipy.integrate import quad
from scipy.special import exp1, j0, j1, jv

from invariant_gravity_extensions.hankel_midplane import piecewise_gauss
from invariant_gravity_extensions.hankel_tail import (
    DiskLogGreen,
    complete_leading_tail,
    radial_tail_jet,
)


class GaussianDisk:
    radius = np.array([.25, 1., 2., 4., 8., 12.])
    surface_density = np.exp(-radius**2/2)
    outer_radius = 12.
    taper_width = 0.
    core_coefficient = -.5
    height = .4

    def surface(self, R):
        return np.exp(-np.asarray(R)**2/2)

    def surface_and_derivative(self, R):
        v = self.surface(R)
        return v, -np.asarray(R)*v


def setup():
    k, w = piecewise_gauss(np.arange(0, 3.01, .25), 32)
    return GaussianDisk(), k, w, np.exp(-k*k/2)


def test_log_green_and_tail_against_independent_gaussian_integrals():
    disk, k, w, S = setup()
    green = DiskLogGreen(disk)
    np.testing.assert_allclose(green.prefix_mass[-1], 1., atol=1e-14)
    np.testing.assert_allclose(green.prefix_log[-1], (np.log(2)-np.euler_gamma)/2, atol=1e-14)
    R = np.array([0., 1e-10, .05, .5, 1., 2., 4.])
    value = radial_tail_jet(disk, k, w, S, R, 3.)
    expected = []
    for r in R:
        kernels = [lambda q, r=r: j0(q*r)/q, lambda q, r=r: -j1(q*r),
            lambda q, r=r: -.5*q*(j0(q*r)-jv(2, q*r)),
            lambda q, r=r: -.25*q*q*(jv(3, q*r)-3*j1(q*r))]
        expected.append([quad(lambda q, f=f: np.exp(-q*q/2)*f(q), 3., np.inf, epsabs=1e-13, epsrel=1e-12)[0] for f in kernels])
    expected = np.array(expected).T
    for key, e in zip(['potential', 'first', 'second', 'third'], expected, strict=True):
        np.testing.assert_allclose(value[key], e, atol=2e-12, rtol=2e-9, err_msg=key)
    np.testing.assert_allclose(value['potential'][0], exp1(4.5)/2, atol=1e-14)
    np.testing.assert_allclose(value['first_over_radius'][0], value['second'][0], atol=1e-15)
    assert value['derivative_first_over_radius'][0] == 0.


def test_completed_tail_has_consistent_potential_hessian_and_third_derivatives():
    disk, k, w, S = setup()

    def evaluate(R, z):
        shape = (len(R), len(z))
        zero = {'potential': np.zeros(shape), 'gradient_R_z': np.zeros((2,)+shape),
            'hessian_RR_Rz_zz_pp': np.zeros((4,)+shape),
            'third_RRR_RRz_Rzz_zzz_Rpp_zpp': np.zeros((6,)+shape)}
        return complete_leading_tail(zero, {'gaussian': disk}, ['gaussian'], k, w, S[None, :], R, z, 1., 3.)[0]

    R, z = np.array([.2, 1., 3.]), np.array([-.5, 0., .3])
    base = evaluate(R, z)
    h = 2e-4
    for axis in [0, 1]:
        derivatives = {key: np.zeros_like(base[key]) for key in ['potential', 'gradient_R_z', 'hessian_RR_Rz_zz_pp']}
        for shift, coefficient in [(-2, 1), (-1, -8), (1, 8), (2, -1)]:
            f = evaluate(R+shift*h if axis == 0 else R, z+shift*h if axis == 1 else z)
            for key in derivatives:
                derivatives[key] += coefficient*f[key]/(12*h)
        H = base['hessian_RR_Rz_zz_pp']
        T = base['third_RRR_RRz_Rzz_zzz_Rpp_zpp']
        expected = [base['gradient_R_z'][axis], H[[0, 1] if axis == 0 else [1, 2]], T[[0, 1, 2, 4] if axis == 0 else [1, 2, 3, 5]]]
        for key, value in zip(derivatives, expected, strict=True):
            np.testing.assert_allclose(derivatives[key], value, atol=3e-8, rtol=3e-7, err_msg=key)
