"""Independent shell, volume, symmetry and translation checks; synthetic only."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from invariant_gravity_extensions.coherent_momentum import (
    audit_scene,
    coherent_excess,
    integrate_axisymmetric,
    plummer_shell_acceleration,
)
from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    total_newtonian,
)


def parts(offset=0, reflection=1):
    return (MassComponent("a", 1, 0, .5, offset+reflection*2),
            MassComponent("b", 2, 0, .5, offset-reflection))


def test_closed_shell_field_against_independent_gauss_quadrature():
    r = np.geomspace(.01, 100, 71)
    mu, w = leggauss(240)
    R, z = r[:, None]*np.sqrt(1-mu*mu), r[:, None]*mu
    grad = total_newtonian(parts(), R, z)["gradient"]
    direct = ((grad[0]*np.sqrt(1-mu*mu)+grad[1]*mu)*w).sum(axis=1)/2
    np.testing.assert_allclose(plummer_shell_acceleration(parts(), r), direct, rtol=1e-9)


def test_centered_plummer_shell_exact():
    r = np.geomspace(1e-4, 1e4, 201)
    source = (MassComponent("central", 3, 0, .7, 4),)
    np.testing.assert_allclose(plummer_shell_acceleration(source, r), 3*r/(r*r+.7**2)**1.5, rtol=1e-14)


def test_volume_measure_full_azimuth_and_radial_jacobian():
    grid = MultipoleGrid(.1, 3, 1025, 16, 0)
    integral = integrate_axisymmetric(grid, np.ones((1025, 16)))
    assert integral == pytest.approx(4*np.pi*(3**3-.1**3)/3, rel=1e-9)


def test_simple_mond_limits_and_nonnegative_excess():
    g = np.array([0, 1e-14, 1, 1e14])
    excess = coherent_excess(g, 1)
    assert excess[0] == 0
    assert excess[1] == pytest.approx(np.sqrt(g[1]), rel=1e-6)
    assert excess[-1] == pytest.approx(1, rel=1e-12)
    np.testing.assert_array_equal(coherent_excess(g, 0), 0)
    with pytest.raises(ValueError):
        coherent_excess([-1], 1)


def test_translation_reflection_and_newtonian_internal_balance():
    grid = MultipoleGrid(.0001, 100, 513, 96, 0)
    basic = audit_scene(parts(), 1, grid)
    shifted = audit_scene(parts(offset=7), 1, grid)
    mirror = audit_scene(parts(reflection=-1), 1, grid)
    assert shifted["correction_net_force_z"] == pytest.approx(basic["correction_net_force_z"], rel=1e-12)
    assert mirror["correction_net_force_z"] == pytest.approx(-basic["correction_net_force_z"], rel=1e-12)
    assert basic["normalized_newtonian_net_force"] < 1e-6


def test_symmetric_sources_have_zero_correction_force():
    grid = MultipoleGrid(.0001, 100, 513, 96, 0)
    symmetric = (MassComponent("a", 1, 0, .5, 1.5), MassComponent("b", 1, 0, .5, -1.5))
    assert audit_scene(symmetric, 1, grid)["normalized_correction_net_force"] < 1e-13


def test_invalid_nonplummer_and_bad_radius_rejected():
    with pytest.raises(ValueError):
        plummer_shell_acceleration((MassComponent("disk", 1, 1, .5),), [1])
    with pytest.raises(ValueError):
        plummer_shell_acceleration(parts(), [0])
