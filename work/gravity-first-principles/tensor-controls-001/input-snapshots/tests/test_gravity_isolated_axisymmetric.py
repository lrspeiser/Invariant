import numpy as np
import pytest
from numpy.polynomial.legendre import legval

from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    anomalous_source,
    solve_isolated,
    solve_poisson,
    total_newtonian,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def test_analytic_mass_gradient_hessian_and_density():
    component = MassComponent("disk", 3, 1.2, .3, z_center=.2)
    R, z = np.array([.2, 1, 3]), np.array([.6, -.4, 1])
    fields = component.fields(R, z)
    step = 1e-4
    for axis in (0, 1):
        plus = component.fields(R+(axis == 0)*step, z+(axis == 1)*step)
        minus = component.fields(R-(axis == 0)*step, z-(axis == 1)*step)
        plus2 = component.fields(R+(axis == 0)*2*step, z+(axis == 1)*2*step)
        minus2 = component.fields(R-(axis == 0)*2*step, z-(axis == 1)*2*step)
        np.testing.assert_allclose((8*(plus["potential"]-minus["potential"])-plus2["potential"]+minus2["potential"])/(12*step),
                                   fields["gradient"][axis], rtol=1e-8)
        np.testing.assert_allclose((8*(plus["gradient"]-minus["gradient"])-plus2["gradient"]+minus2["gradient"])/(12*step),
                                   fields["hessian"][:, axis], rtol=1e-8)
    lap = fields["hessian"][0, 0]+fields["hessian"][1, 1]+fields["gradient"][0]/R
    np.testing.assert_allclose(lap, fields["laplacian"], rtol=1e-13)
    assert np.all(fields["laplacian"] > 0)


@pytest.mark.parametrize("order", [0, 2, 3])
@pytest.mark.parametrize("plane_scale", [None, .2])
def test_manufactured_poisson_multipoles(order, plane_scale):
    # Exact potential r^l exp(-r^2/2) P_l(mu), which vanishes at infinity.
    def source(R, z):
        radius = np.hypot(R, z)
        return (radius**order*np.exp(-radius*radius/2)*(radius*radius-2*order-3)*
                legval(z/radius, np.eye(order+1)[order]))

    grid = MultipoleGrid(1e-4, 30, 4097, 64, 5, plane_scale)
    solution = solve_poisson(grid, source)
    R, z = np.array([.2, .5, 1, 2, 3]), np.array([.1, -.3, .8, 1, -2])

    def exact(R, z):
        radius = np.hypot(R, z)
        return radius**order*np.exp(-radius*radius/2)*legval(z/radius, np.eye(order+1)[order])

    predicted = solution.evaluate(R, z)
    np.testing.assert_allclose(predicted["potential"], exact(R, z), atol=2e-8, rtol=1e-6)
    step = 1e-5
    accel = -np.array([(exact(R+step, z)-exact(R-step, z))/(2*step),
                       (exact(R, z+step)-exact(R, z-step))/(2*step)])
    np.testing.assert_allclose(predicted["acceleration"], accel, atol=2e-8, rtol=1e-5)


@pytest.mark.parametrize("shape", [.5, 1, 2])
def test_spherical_numerical_field_matches_exact_qumond(shape):
    components = (MassComponent("plummer", 2, 0, 1),)
    spec = SaturatedActionSpec("qumond", shape=shape)
    solution = solve_isolated(components, spec, 1, MultipoleGrid(1e-5, 1e4, 2049, 8, 0))
    R = np.geomspace(.03, 100, 40)
    predicted = solution.evaluate(R, np.zeros_like(R))
    p = total_newtonian(components, R, 0)["gradient"]
    expected = -p*(1+spec.delta_nu(np.sqrt(np.sum(p*p, axis=0))))[None, :]
    np.testing.assert_allclose(predicted["acceleration"], expected, rtol=2e-6, atol=1e-10)


def test_joint_source_partition_invariance_and_nonlinear_response():
    whole = (MassComponent("whole", 2, 1, .3),)
    split = (MassComponent("stars", .5, 1, .3), MassComponent("gas", 1.5, 1, .3))
    spec = SaturatedActionSpec("qumond")
    R, z = np.array([1, 2, 5]), np.array([.1, .3, 2])
    expected = anomalous_source(whole, spec, 1, R, z)
    np.testing.assert_allclose(anomalous_source(split, spec, 1, R, z), expected, rtol=1e-14)
    wrongly_separated = sum(anomalous_source((c,), spec, 1, R, z) for c in split)
    assert np.linalg.norm(wrongly_separated-expected)/np.linalg.norm(expected) > .1


def test_disk_newtonian_inverse_and_curl_control():
    component = MassComponent("disk", 3, 1, .5)
    grid = MultipoleGrid(1e-4, 1e3, 2049, 192, 64, .5)
    solution = solve_poisson(grid, lambda R, z: component.fields(R, z)["laplacian"])
    R = np.array([.2, .5, 1, 2, 5])
    z = .3*R
    exact = -component.fields(R, z)["gradient"]
    measured = solution.evaluate(R, z)["acceleration"]
    errors = np.linalg.norm(measured-exact, axis=0)/np.linalg.norm(exact, axis=0)
    assert np.max(errors) < 2e-4
    h = 1e-4
    dR_az = (solution.evaluate(R+h, z)["acceleration"][1]-solution.evaluate(R-h, z)["acceleration"][1])/(2*h)
    dz_aR = (solution.evaluate(R, z+h)["acceleration"][0]-solution.evaluate(R, z-h)["acceleration"][0])/(2*h)
    np.testing.assert_allclose(dR_az, dz_aR, atol=1e-6, rtol=1e-4)


def test_anomalous_source_matches_independent_flux_divergence():
    components = (MassComponent("disk", 2, 1, .2), MassComponent("gas", .5, 2, .4))
    spec = SaturatedActionSpec("qumond", shape=2)
    R, z = np.array([.5, 1, 3]), np.array([.3, .5, 1])

    def flux(R, z):
        p = total_newtonian(components, R, z)["gradient"]
        return spec.delta_nu(np.linalg.norm(p, axis=0))*p

    h = 1e-5
    numerical = ((flux(R+h, z)[0]-flux(R-h, z)[0])/(2*h)+flux(R, z)[0]/R+
                 (flux(R, z+h)[1]-flux(R, z-h)[1])/(2*h))
    np.testing.assert_allclose(anomalous_source(components, spec, 1, R, z), numerical, rtol=1e-8)


def test_no_implicit_boundary_extrapolation_or_multifield_claim():
    component = MassComponent("body", 1, 0, 1)
    grid = MultipoleGrid(.01, 100, 257, 8, 0)
    solution = solve_isolated((component,), SaturatedActionSpec("qumond"), 1, grid)
    with pytest.raises(ValueError, match="domain"):
        solution.evaluate(200, 0)
    with pytest.raises(NotImplementedError, match="scalar"):
        solve_isolated((component,), SaturatedActionSpec("trimond_alignment", .75, 2), 1, grid)
    with pytest.raises(ValueError, match="uniquely"):
        total_newtonian((component, component), 1, 0)
    with pytest.raises(ValueError, match="thickness"):
        MassComponent("sheet", 1, 1, 0)


def test_dimensional_rescaling_preserves_gravity_prediction():
    spec = SaturatedActionSpec("qumond", shape=1)
    component = MassComponent("sphere", 2, 0, 1)
    grid = MultipoleGrid(1e-4, 1e3, 1025, 8, 0)
    solution = solve_isolated((component,), spec, 1, grid)
    length, acceleration = 17.0, 5.0
    transformed = MassComponent("sphere", component.gm*acceleration*length**2, 0, length)
    transformed_grid = MultipoleGrid(grid.r_min*length, grid.r_max*length, 1025, 8, 0)
    other = solve_isolated((transformed,), spec, acceleration, transformed_grid)
    R, z = np.array([.1, 1, 10]), np.array([.2, -.3, 3])
    expected = solution.evaluate(R, z)["acceleration"]*acceleration
    np.testing.assert_allclose(other.evaluate(R*length, z*length)["acceleration"], expected, rtol=1e-10)
