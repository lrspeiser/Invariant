"""Independent direct spatial integration of a spherical Gaussian at large r."""
import numpy as np
from scipy.special import roots_hermite, roots_laguerre

from invariant_gravity_extensions.direct_exterior import point_jet_sum, unpack_point_jet


def test_direct_three_dimensional_gaussian_matches_exterior_monopole():
    # Gaussian radial and height quadratures are independent of the disk lift.
    # Beyond ten widths its exact enclosed-mass correction is <2e-20.
    width, mass, G = .7, 2.4, 1.3
    t, wt = roots_laguerre(24)
    u, wu = roots_hermite(24)
    phi = np.arange(48)*2*np.pi/48
    radius = width*np.sqrt(2*t)
    shape = (len(t), len(u), len(phi))
    x = np.broadcast_to(radius[:, None, None]*np.cos(phi), shape).ravel()
    y = np.broadcast_to(radius[:, None, None]*np.sin(phi), shape).ravel()
    z = np.broadcast_to(np.sqrt(2)*width*u[None, :, None], shape).ravel()
    gm = np.broadcast_to((G*mass/np.sqrt(np.pi)/len(phi))*wt[:, None, None]*wu[None, :, None], shape).ravel()
    R, Z = width*np.array([0., 6., 10., 12.]), width*np.array([12., 8., 0., -5.])
    field = unpack_point_jet(point_jet_sum(R, Z, x, y, z, gm))
    r = np.hypot(R, Z)
    p = np.array([R, Z, np.zeros_like(R)])
    H = np.empty((3, 3, len(R)))
    T = np.empty((3, 3, 3, len(R)))
    for i in range(3):
        for j in range(3):
            H[i, j] = G*mass*((i == j)/r**3-3*p[i]*p[j]/r**5)
            for k in range(3):
                T[i, j, k] = G*mass*(-3*((i == j)*p[k]+(i == k)*p[j]+(j == k)*p[i])/r**5+15*p[i]*p[j]*p[k]/r**7)
    np.testing.assert_allclose(field['potential'], -G*mass/r, rtol=3e-11, atol=1e-14)
    np.testing.assert_allclose(field['gradient_R_z'], G*mass*p[:2]/r**3, rtol=3e-10, atol=1e-14)
    np.testing.assert_allclose(field['hessian_RR_Rz_zz_pp'], [H[0, 0], H[0, 1], H[1, 1], H[2, 2]], rtol=3e-9, atol=1e-13)
    expected = np.array([T[0, 0, 0], T[0, 0, 1], T[0, 1, 1], T[1, 1, 1], T[0, 2, 2], T[1, 2, 2]])
    np.testing.assert_allclose(field['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], expected, rtol=3e-8, atol=1e-13)
    assert max(abs(field['azimuthal_gradient'])) < 1e-15
