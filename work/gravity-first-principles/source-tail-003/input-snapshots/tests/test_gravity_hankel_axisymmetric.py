"""Off-plane Green integrals, Cartesian Gaussian tensors and symmetry controls."""
import numpy as np
import pytest
from scipy.integrate import quad, quad_vec
from scipy.special import erf, erfcx

from invariant_gravity_extensions.hankel_axisymmetric import cylindrical_jet
from invariant_gravity_extensions.hankel_midplane import piecewise_gauss, sech2_midplane_laplace
from invariant_gravity_extensions.vertical_green import (
    Sech2VerticalGreen,
    exponential_convolution,
    exponential_moments,
)


def gaussian_vertical(k, z, width):
    kz, zz = k[None, :]*width, z[:, None]/width
    common = .5*np.exp(-zz*zz/2)
    left = common*erfcx((kz-zz)/np.sqrt(2))
    right = common*erfcx((kz+zz)/np.sqrt(2))
    Z, dz = left+right, k*(right-left)
    density = np.exp(-zz*zz/2)/(np.sqrt(2*np.pi)*width)
    derivative = -zz/width*density
    return np.array([Z, dz, k*k*Z-2*k*density, k*k*dz-2*k*derivative])


def test_exponential_moments_and_equal_rate_continuation():
    a = np.array([0., 1e-6, 1., 2., 100.])
    for span in [0., .01, 1.7]:
        moments = exponential_moments(a, span)
        for j in range(4):
            for i, decay in enumerate(a):
                expected = quad(lambda t, decay=decay, j=j: t**j*np.exp(-decay*t), 0, span, epsabs=1e-14)[0]
                assert moments[j, i] == pytest.approx(expected, rel=1e-11, abs=1e-14)
        actual = exponential_convolution(a, 2., span)
        expected = [quad(lambda t, decay=decay, span=span: np.exp(-2*t-decay*(span-t)), 0, span, epsabs=1e-14)[0] for decay in a]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_vertical_spline_green_matches_independent_direct_integrals_and_derivatives():
    # Short/rough profile makes the spline-to-tail f'' jump measurable. The
    # third derivative must include its weak point contribution at both ends.
    source = Sech2VerticalGreen(intervals=64, extent=4.)
    a = np.array([0., .01, .2, 1., 2., 20., 200.])
    probes = np.array([0., .17, 1., 3.97, 4., 4.2, 8.])
    result = source.jet(a, probes)
    for index, u in enumerate(probes):
        def integrand(v, u=u):
            f = source.source(v)[0]
            e1, e2 = np.exp(-a*abs(u-v)), np.exp(-a*(u+v))
            return np.r_[f*(e1+e2), -a*f*(np.sign(u-v)*e1+e2)]

        direct, _ = quad_vec(integrand, 0, np.inf, points=[*source.nodes, u], epsabs=2e-13, epsrel=2e-13, limit=2000)
        f, fp = source.source(u)
        Z, dz = direct[:len(a)], direct[len(a):]
        expected = np.array([Z, dz, a*a*Z-2*a*f, a*a*dz-2*a*fp])
        np.testing.assert_allclose(result[:, index], expected, rtol=2e-7, atol=3e-9)
    reflected = source.jet(a, -probes)
    np.testing.assert_array_equal(reflected[[0, 2]], result[[0, 2]])
    np.testing.assert_array_equal(reflected[[1, 3]], -result[[1, 3]])
    np.testing.assert_allclose(result[0, :, 0], 1., atol=3e-14)
    np.testing.assert_allclose(result[1:, :, 0], 0., atol=3e-14)


def test_vertical_reference_converges_to_exact_sech_midplane():
    source = Sech2VerticalGreen(intervals=2400, extent=24.)
    a = np.r_[0., np.geomspace(.001, 300., 30)]
    result = source.jet(a, [0.])[:, 0]
    Z = sech2_midplane_laplace(a)
    np.testing.assert_allclose(result[0], Z, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(result[1], 0., atol=1e-15)
    np.testing.assert_allclose(result[3], 0., atol=1e-15)
    f0 = source.source(0.)[0]
    # Same-source differential identity; stable convolution is used in code.
    np.testing.assert_allclose(result[2], a*a*result[0]-2*a*f0, rtol=1e-7, atol=3e-11)
    assert abs(source.unnormalized_mass-1) < 1e-8
    f = source.source(np.linspace(-40, 40, 40001))[0]
    assert np.min(f) > 0


def test_full_cylindrical_jet_against_cartesian_spherical_gaussian():
    width, mass, G = 1.3, 2.7, .8
    R, z = np.array([0., .25, 1., 3., 7.]), np.array([-4., -1., 0., .3, 2., 6.])
    k, w = piecewise_gauss(np.linspace(0, 16, 65), 24)
    S = mass/(2*np.pi)*np.exp(-.5*width*width*k*k)
    got = cylindrical_jet(k, w, S[None, :], gaussian_vertical(k, z, width)[None, :], R, z, G)
    x = np.array(np.broadcast_arrays(R[:, None], z[None, :], np.zeros((len(R), len(z)))))
    r = np.linalg.norm(x, axis=0)
    rho = mass/(2*np.pi*width*width)**1.5*np.exp(-.5*(r/width)**2)
    drho = -r/width**2*rho
    enclosed = mass*(erf(r/(np.sqrt(2)*width))-np.sqrt(2/np.pi)*r/width*np.exp(-.5*(r/width)**2))
    g = np.divide(G*enclosed, r*r, out=np.zeros_like(r), where=r > 0)
    aa = np.divide(g, r, out=4*np.pi*G*rho/3, where=r > 0)
    gp = 4*np.pi*G*rho-2*aa
    b = np.divide(gp-aa, r*r, out=np.zeros_like(r), where=r > 0)
    gpp = 4*np.pi*G*drho-2*np.divide(gp-aa, r, out=np.zeros_like(r), where=r > 0)
    c = np.divide(gpp-3*np.divide(gp-aa, r, out=np.zeros_like(r), where=r > 0), r**3,
                  out=np.zeros_like(r), where=r > 0)
    H = np.empty((3, 3, len(R), len(z)))
    T = np.empty((3, 3, 3, len(R), len(z)))
    for i in range(3):
        for j in range(3):
            H[i, j] = (i == j)*aa+b*x[i]*x[j]
            for l in range(3):
                T[i, j, l] = b*((i == j)*x[l]+(i == l)*x[j]+(j == l)*x[i])+c*x[i]*x[j]*x[l]
    expected_H = np.array([H[0, 0], H[0, 1], H[1, 1], H[2, 2]])
    expected_T = np.array([T[0, 0, 0], T[0, 0, 1], T[0, 1, 1], T[1, 1, 1], T[0, 2, 2], T[1, 2, 2]])
    np.testing.assert_allclose(got['gradient_R_z'], (aa*x)[:2], atol=4e-14, rtol=2e-11)
    np.testing.assert_allclose(got['hessian_RR_Rz_zz_pp'], expected_H, atol=5e-13, rtol=2e-10)
    np.testing.assert_allclose(got['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], expected_T, atol=2e-12, rtol=2e-9)
    np.testing.assert_allclose(got['third_tensor_norm'], np.sqrt(np.sum(T*T, axis=(0, 1, 2))), atol=3e-12, rtol=2e-9)
    np.testing.assert_allclose(got['gradient_hessian_norm_R_z'], 2*np.einsum('ijrz,ijlrz->lrz', H, T)[:2], atol=3e-12, rtol=2e-9)
    np.testing.assert_allclose(got['laplacian'], 4*np.pi*G*rho, atol=3e-14, rtol=2e-10)
    np.testing.assert_allclose(got['gradient_laplacian_R_z'], -4*np.pi*G*rho*x[:2]/width**2, atol=3e-14, rtol=2e-10)


def test_joint_source_partition_and_distance_homology():
    R, z = np.array([0., .5, 2., 5.]), np.array([-1., 0., .3, 2.])
    k, w = piecewise_gauss(np.linspace(0, 16, 33), 16)
    S = np.exp(-k*k/2)/(2*np.pi)
    Z = gaussian_vertical(k, z, 1.)
    original = cylindrical_jet(k, w, S[None, :], Z[None, :], R, z, 1.)
    partition = cylindrical_jet(k, w, np.array([.3*S, .7*S]), np.array([Z, Z]), R, z, 1., batch_size=1)
    for key in original:
        np.testing.assert_allclose(partition[key], original[key], atol=2e-13, rtol=2e-11)
    distance = 1.27
    scaled = cylindrical_jet(k/distance, w/distance, (S*distance**2)[None, :],
        (Z/distance**np.arange(4)[:, None, None])[None, :], R*distance, z*distance, 1.)
    for key, power in [('potential', 1), ('gradient_R_z', 0), ('hessian_RR_Rz_zz_pp', -1),
                       ('third_RRR_RRz_Rzz_zzz_Rpp_zpp', -2), ('gradient_hessian_norm_R_z', -3)]:
        np.testing.assert_allclose(scaled[key], original[key]*distance**power, atol=3e-13, rtol=2e-10)


def test_vertical_green_invalid_inputs_fail():
    with pytest.raises(ValueError):
        Sech2VerticalGreen(intervals=8)
    with pytest.raises(ValueError):
        exponential_moments([-1.], 1.)
    with pytest.raises(ValueError):
        Sech2VerticalGreen(intervals=32).jet([1.], [np.nan])
