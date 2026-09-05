"""Independent finite-polynomial controls for the source-only projection."""
import numpy as np
import pytest
from numpy.polynomial.legendre import legval

from invariant_gravity_extensions.source_projection import (
    half_gauss,
    project_even_source,
    projection_metrics,
)


class PolynomialSource:
    def density_and_gradient(self, R, z):
        # rho=2+R²+3z²=2+(5/3)r²+(4/3)r² P2(mu), positive and even.
        return 2+R*R+3*z*z, np.array([2*R, 6*z])


def test_even_projection_matches_exact_cartesian_polynomial_and_radial_derivative():
    radii = np.array([.1, 1., 3.])
    projection = project_even_source({'p': PolynomialSource()}, radii, 8, 32)
    c, d = projection['coefficients'], projection['radial_derivative_coefficients']
    expected = np.zeros_like(c)
    expected[:, 0], expected[:, 2] = 2+5*radii**2/3, 4*radii**2/3
    derivative = np.zeros_like(d)
    derivative[:, 0], derivative[:, 2] = 10*radii/3, 8*radii/3
    np.testing.assert_allclose(c, expected, atol=3e-12)
    np.testing.assert_allclose(d, derivative, atol=3e-12)
    metrics = projection_metrics({'p': PolynomialSource()}, projection, [0, 2, 8], 64)
    assert max(metrics[0]['density_L1_fraction_error']) > .1
    for record in metrics[1:]:
        assert max(record['density_L1_fraction_error']) < 1e-12
        assert max(record['gradient_L1_fraction_error']) < 1e-12
        assert max(abs(record['relative_shell_mass_error'])) < 1e-12
        assert max(record['negative_density_fraction']) == 0


def test_negative_projection_is_measured_without_clipping():
    p = project_even_source({'p': PolynomialSource()}, np.array([1.]), 4, 32)
    p['coefficients'][0, 4] = 100.
    record = projection_metrics({'p': PolynomialSource()}, p, [4], 256)[0]
    assert record['negative_density_fraction'][0] > 1.
    assert record['density_L1_fraction_error'][0] > 1.
    assert abs(record['relative_shell_mass_error'][0]) < 1e-10
    mu, w = half_gauss(256)
    expected = legval(mu, p['coefficients'][0])
    actual = (np.maximum(-expected, 0)@w)/record['physical_hemisphere_density_integral'][0]
    assert record['negative_density_fraction'][0] == pytest.approx(actual)


def test_projection_units_and_invalid_inputs():
    with pytest.raises(ValueError):
        project_even_source({'p': PolynomialSource()}, [1.], 3, 32)
    with pytest.raises(ValueError):
        project_even_source({'p': PolynomialSource()}, [0.], 2, 32)
    with pytest.raises(ValueError):
        half_gauss(2)
