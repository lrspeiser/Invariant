import numpy as np
import pytest
from scipy.integrate import quad

from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    solve_isolated,
)
from invariant_gravity_extensions.reconstructed_axisymmetric import (
    ReconstructedNewtonianSource,
    SurfaceDensityDisk,
    multipole_fields,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


@pytest.mark.parametrize("a,b", [(0, 1), (1, .3)])
def test_reconstructed_fields_against_analytic_mass_and_finite_differences(a, b):
    exact = MassComponent("control", 1, a, b)
    grid = MultipoleGrid(1e-4, 1e3, 1025, 160, 48, plane_scale=b)
    source = ReconstructedNewtonianSource.build("reconstructed", lambda R, z: exact.fields(R, z)["laplacian"], grid)
    R, z = np.array([0, .5, 1, 2, 4]), np.array([1, .2, 0, -.4, .7])
    predicted = multipole_fields(source.potential, R, z, batch_size=2)
    truth = exact.fields(R, z)
    assert np.linalg.norm(predicted["gradient"]-truth["gradient"])/np.linalg.norm(truth["gradient"]) < .003
    assert np.linalg.norm(predicted["hessian"]-truth["hessian"])/np.linalg.norm(truth["hessian"]) < .015
    # Differentiate returned forces independently; mixed components must agree.
    point = np.array([1.3, .4])
    step = 1e-5
    hessian = np.column_stack([(multipole_fields(source.potential, *(point+step*direction))["gradient"]-
                               multipole_fields(source.potential, *(point-step*direction))["gradient"])/(2*step)
                              for direction in np.eye(2)])
    np.testing.assert_allclose(multipole_fields(source.potential, *point)["hessian"], hessian, rtol=2e-7, atol=1e-9)
    mirrored = multipole_fields(source.potential, R, -z)
    np.testing.assert_allclose(mirrored["gradient"], predicted["gradient"]*np.array([[1], [-1]]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(source.fields(R, z)["laplacian"], truth["laplacian"], rtol=1e-14)


def test_reconstructed_sphere_qumond_against_exact_algebraic_solution():
    exact = MassComponent("plummer", 1, 0, 1)
    grid = MultipoleGrid(1e-5, 1e4, 1537, 24, 0)
    source = ReconstructedNewtonianSource.build("density", lambda R, z: exact.fields(R, z)["laplacian"], grid)
    spec = SaturatedActionSpec("qumond", shape=1)
    solution = solve_isolated((source,), spec, .1, grid)
    r = np.array([.3, 1, 3, 10, 30.])
    gn = exact.fields(r, 0)["gradient"][0]
    expected = gn*(1+spec.delta_nu(gn/.1))
    actual = -solution.evaluate(r, 0)["acceleration"][0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_surface_lift_normalization_and_aperture_are_independent_of_queries():
    disk = SurfaceDensityDisk(np.array([.1, 1, 2, 3.]), np.array([4, 3, 2, 1.]), .2, 3, .5)
    for R in [.01, .8, 2.8]:
        integral = quad(lambda z, radius=R: float(disk.density(radius, z)), -5, 5, epsabs=1e-11)[0]
        assert integral == pytest.approx(disk.surface(R), rel=1e-10)
    assert disk.surface(3.1) == 0
    assert disk.surface(0) == disk.surface(.1)
    assert np.all(disk.density(np.linspace(0, 4), 1e5) == 0)
    with pytest.raises(ValueError):
        SurfaceDensityDisk([1, 2, 3], [1, -1, 1], .2, 3, .5)


def test_reconstruction_domain_does_not_silently_extrapolate():
    exact = MassComponent("p", 1, 0, 1)
    source = ReconstructedNewtonianSource.build("p", lambda R, z: exact.fields(R, z)["laplacian"],
                                                 MultipoleGrid(.01, 100, 65, 8, 0))
    with pytest.raises(ValueError, match="domain"):
        source.fields(101, 0)
