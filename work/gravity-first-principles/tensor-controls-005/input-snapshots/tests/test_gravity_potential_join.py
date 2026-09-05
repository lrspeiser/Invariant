"""Independent symbolic checks of the full joined Cartesian potential."""
import numpy as np
import pytest
import sympy as sp

from invariant_gravity_extensions.potential_join import (
    blend_potential_jets,
    cartesian_tensors,
    pack_cartesian,
    radial_weight_jet,
)


def symbolic_jet(expression, xyz, R, z):
    values = [expression]
    values += [sp.diff(expression, x) for x in xyz]
    values += [sp.diff(expression, x, y) for x in xyz for y in xyz]
    values += [sp.diff(expression, x, y, q) for x in xyz for y in xyz for q in xyz]
    f = sp.lambdify(xyz, values, 'numpy', cse=True)
    v = np.array([np.broadcast_to(a, R.shape) for a in f(R, z, np.zeros_like(R))])
    return pack_cartesian(v[0], v[1:4], v[4:13].reshape((3, 3)+R.shape),
                          v[13:].reshape((3, 3, 3)+R.shape), R, z)


def test_join_matches_derivatives_of_symbolic_potential():
    x, zsym, y = xyz = sp.symbols('x z y', real=True)
    r2 = x*x+y*y+zsym*zsym
    r = sp.sqrt(r2)
    near = sp.exp(-r2/10)+sp.Rational(1, 20)*(x*x+y*y)*zsym*zsym
    far = -1/r+sp.Rational(3, 100)*(zsym*zsym-(x*x+y*y)/2)/r**5+sp.Rational(1, 5)
    q = (r-2)/3
    w = 35*q**4-84*q**5+70*q**6-20*q**7
    R, z = np.array([0., 2.5, 2., 4.2, .3]), np.array([3., 0., 2., -.5, -4.])
    expected = symbolic_jet(near+w*(far-near), xyz, R, z)
    actual = blend_potential_jets(symbolic_jet(near, xyz, R, z), symbolic_jet(far, xyz, R, z), R, z, inner=2., outer=5.)
    for key in ['potential', 'gradient_R_z', 'hessian_RR_Rz_zz_pp', 'third_RRR_RRz_Rzz_zzz_Rpp_zpp',
                'laplacian', 'gradient_laplacian_R_z', 'gradient_hessian_norm_R_z']:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-11, atol=3e-12, err_msg=key)


def test_join_weight_has_exact_plateaus_and_c3_endpoints():
    R = np.array([0., 1., 2., 5., 6., 100.])
    w, p, h, t = radial_weight_jet(R, np.zeros_like(R), 2., 5.)
    np.testing.assert_array_equal(w, [0, 0, 0, 1, 1, 1])
    for derivative in (p, h, t):
        np.testing.assert_array_equal(derivative, np.zeros_like(derivative))
    q = np.array([.00001, .1, .5, .9, .99999])
    w, p, h, t = radial_weight_jet(2+3*q, np.zeros_like(q), 2., 5.)
    np.testing.assert_allclose(w+w[::-1], 1., atol=1e-15)


def test_join_retains_gauge_mismatch_source_terms():
    R, z = np.array([3.5]), np.array([0.])
    zero = pack_cartesian(np.zeros(1), np.zeros((3, 1)), np.zeros((3, 3, 1)), np.zeros((3, 3, 3, 1)), R, z)
    constant = {**zero, 'potential': np.ones(1)}
    joined = blend_potential_jets(zero, constant, R, z, inner=2., outer=5.)
    expected = radial_weight_jet(R, z, 2., 5.)
    for value, reference in zip(cartesian_tensors(joined), expected, strict=True):
        np.testing.assert_allclose(value, reference, atol=1e-15)
    assert joined['laplacian'][0] > .1
    with pytest.raises(ValueError):
        radial_weight_jet(R, z, 5., 2.)
