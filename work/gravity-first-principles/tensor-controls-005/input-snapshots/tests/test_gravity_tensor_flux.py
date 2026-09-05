"""Coordinate-covariant transfer of complete cylindrical fields to the action."""
import runpy
from pathlib import Path

import numpy as np
import pytest

from invariant_gravity_extensions.length_axisymmetric import full_length_flux
from invariant_gravity_extensions.length_screening import LengthScreening


@pytest.mark.parametrize('shape', [.5, 1., 2.])
@pytest.mark.parametrize('length', [0., .7])
def test_cylindrical_flux_agrees_with_rotated_existing_action(shape, length):
    root = Path(__file__).resolve().parents[1]
    full_flux = runpy.run_path(str(root/'scripts/audit_gravity_tensor_flux.py'))['full_flux']
    # Local components of a smooth axisymmetric polynomial potential.
    r, z = np.array([0., .3, 1.2]), np.array([.7, -.4, 0.])
    p = np.array([2*r+2*r*z*z, 4*z+2*r*r*z])
    H = np.zeros((3, 3, len(r)))
    H[0, 0], H[1, 1], H[2, 2] = 2+2*z*z, 4+2*r*r, 2+2*z*z
    H[0, 1] = H[1, 0] = 4*r*z
    dnorm = np.array([64*r*z*z+8*r*(4+2*r*r), 16*z*(2+2*z*z)+64*r*r*z])
    dlap = np.array([4*r, 8*z])
    fields = {'potential': r*r+2*z*z+r*r*z*z, 'gradient_R_z': p,
        'hessian_RR_Rz_zz_pp': H[[0, 0, 1, 2], [0, 1, 1, 2]],
        'third_RRR_RRz_Rzz_zzz_Rpp_zpp': np.array([0*r, 4*z, 4*r, 0*r, 0*r, 4*z]),
        'gradient_hessian_norm_R_z': dnorm, 'gradient_laplacian_R_z': dlap}
    radius = np.hypot(r, z)
    q = np.array([[r/radius, z/radius], [z/radius, -r/radius]])
    rotated_p = np.einsum('ijn,jn->in', q, p)
    rotated_H = np.einsum('ain,ijn,bjn->abn', q, H[:2, :2], q)
    polar_fields = {'gradient_r_theta': rotated_p,
        'hessian_rr_rt_tt_pp': np.array([rotated_H[0, 0], rotated_H[0, 1], rotated_H[1, 1], H[2, 2]]),
        'gradient_hessian_norm_r_theta': np.einsum('ijn,jn->in', q, dnorm),
        'gradient_laplacian_r_theta': np.einsum('ijn,jn->in', q, dlap)}
    spec = LengthScreening(shape, 1e-6)
    a0 = .8
    from invariant_gravity_extensions.galaxy_development import SI_ACCELERATION_TO_KMS2_KPC

    total, anomaly = full_flux(fields, {'shape': shape, 'epsilon': 1e-6,
        'length_pc': length*1000, 'a0_m_s2': a0/SI_ACCELERATION_TO_KMS2_KPC})
    expected_polar = full_length_flux(polar_fields, spec, length, a0)
    expected = np.einsum('jin,jn->in', q, expected_polar)
    np.testing.assert_allclose(anomaly[:2], expected, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(total[:2], p+expected, rtol=3e-13, atol=3e-14)
    np.testing.assert_array_equal(total[2], 0.)
