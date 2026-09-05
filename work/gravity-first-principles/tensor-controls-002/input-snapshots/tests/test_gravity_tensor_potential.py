"""Independent symbolic fields, axis quotients and interpolation convergence."""
import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from invariant_gravity_extensions.tensor_potential import C3TensorPotential

R, Z = sp.symbols('R Z', real=True)


def evaluate(expression, radius, height):
    r, z = np.broadcast_arrays(radius, height)
    return np.broadcast_to(sp.lambdify((R, Z), expression, 'numpy')(r, z), r.shape).copy()


def table(expression, radius, height):
    return np.array([[evaluate(sp.diff(expression, R, i, Z, j), radius[:, None], height[None, :])
        for j in range(4)] for i in range(4)])


def expected_fields(expression, radius, height):
    pr, pz = sp.diff(expression, R), sp.diff(expression, Z)
    hpp = sp.cancel(pr/R)
    expressions = [expression, pr, pz, sp.diff(pr, R), sp.diff(pr, Z), sp.diff(pz, Z), hpp,
        sp.diff(pr, R, 2), sp.diff(pr, R, Z), sp.diff(pr, Z, 2), sp.diff(pz, Z, 2), sp.diff(hpp, R), sp.diff(hpp, Z)]
    return np.array([evaluate(x, radius, height) for x in expressions])


def packed(fields):
    return np.concatenate([fields['potential'][None], fields['gradient_R_z'], fields['hessian_RR_Rz_zz_pp'],
        fields['third_RRR_RRz_Rzz_zzz_Rpp_zpp']])


def test_tensor_polynomial_exactness_and_regular_axis_quotients():
    expression = 17+R**2/5+3*Z**2/10+7*R**2*Z**2/10-13*R**4*Z**2/100+R**6*Z**4/20+R**4*Z**6/100
    r, z = np.array([0., .4, 1., 2.3]), np.array([0., .3, .9, 1.7])
    potential = C3TensorPotential(r, z, table(expression, r, z))
    qr = np.array([0., 1e-14, .073, .4-1e-12, .4, .4+1e-12, .73, 1.8, 2.3])
    qz = np.array([0., .23, -.17, .37, -.37, .37, .9, -1.4, 1.7])
    actual = packed(potential.fields(qr, qz, batch_size=3))
    expected = expected_fields(expression, qr, qz)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=3e-10)
    # Sampling a large additive constant loses low-order bits before the
    # interpolator receives them. Retain the strict analytic axis check using
    # a source gauge that preserves those bits; test rounded inputs separately.
    reduced = C3TensorPotential(r, z, table(expression-17, r, z))
    axis = packed(reduced.fields(qr[1:2], qz[1:2]))
    np.testing.assert_allclose(axis[11, 0]/qr[1], expected[11, 1]/qr[1], rtol=2e-10, atol=1e-12)
    assert actual[11, 0] == 0.
    assert actual[6, 0] == actual[3, 0]


def test_axis_quotient_matches_independent_interpolant_of_rounded_inputs():
    expression = 17+R**2/5+3*Z**2/10+7*R**2*Z**2/10-13*R**4*Z**2/100+R**6*Z**4/20+R**4*Z**6/100
    grid_r, grid_z = np.array([0., .4]), np.array([0., .3])
    data = table(expression, grid_r, grid_z)
    ctx = mp.mp.clone()
    ctx.dps = 60

    def interpolate(values, width):
        # Independent monomial boundary-condition solve; production uses
        # factored basis polynomials and tensor contractions.
        matrix = ctx.matrix(8, 8)
        for endpoint in range(2):
            for derivative in range(4):
                for power in range(derivative, 8):
                    matrix[endpoint*4+derivative, power] = (ctx.factorial(power)/ctx.factorial(power-derivative)
                        * (endpoint*width)**(power-derivative))
        return ctx.lu_solve(matrix, ctx.matrix(values))

    radial_values = []
    for endpoint in range(2):
        for derivative in range(4):
            vertical = interpolate([ctx.mpf(float(data[derivative, j, endpoint, f]))
                for f in range(2) for j in range(4)], ctx.mpf(float(grid_z[1])))
            radial_values.append(sum(c*ctx.mpf(.23)**j for j, c in enumerate(vertical)))
    radial = interpolate(radial_values, ctx.mpf(float(grid_r[1])))
    expected_limit = float(8*radial[4])
    actual = packed(C3TensorPotential(grid_r, grid_z, data).fields([1e-14], [.23]))[11, 0]/1e-14
    np.testing.assert_allclose(actual, expected_limit, rtol=0, atol=2e-12)


def test_nonpolynomial_convergence_and_third_derivative_continuity():
    expression = -sp.exp(-R**2-Z**2)
    qr = np.array([.073, .317, .613, .941])
    qz = np.array([-.17, .29, .53, .87])
    expected = expected_fields(expression, qr, qz)
    errors = []
    for n in [4, 7]:
        grid = np.linspace(0, 1.2, n)
        potential = C3TensorPotential(grid, grid, table(expression, grid, grid))
        errors.append(np.max(abs(packed(potential.fields(qr, qz))-expected)))
    assert errors[1] < errors[0]/8
    assert errors[1] < .001
    left = packed(potential.fields(np.array([.6-1e-8]), np.array([.37])))
    right = packed(potential.fields(np.array([.6+1e-8]), np.array([.37])))
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-5)


def test_symmetry_and_domain_are_explicit_requirements():
    grid = np.array([0., 1.])
    data = table(R**2+Z**2, grid, grid)
    bad = data.copy()
    bad[1, 0, 0, 1] = 1e-15
    with pytest.raises(ValueError, match='symmetry'):
        C3TensorPotential(grid, grid, bad)
    potential = C3TensorPotential(grid, grid, data)
    with pytest.raises(ValueError, match='domain'):
        potential.fields([1.01], [0.])
    with pytest.raises(ValueError, match='domain'):
        potential.fields([.5], [float('nan')])
