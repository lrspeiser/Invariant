"""Source transfer, unit scaling and internal-force checks without observations."""
import numpy as np
import pytest

from invariant_gravity_extensions.coherent_momentum import integrate_axisymmetric
from invariant_gravity_extensions.external_multifield import FluxPoissonSolver
from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    solve_poisson,
    total_newtonian,
)
from invariant_gravity_extensions.isolated_multifield import (
    beta_zero_density_source,
    gradient_on_flux_grid,
    normalized_newtonian_gradient,
    solve_isolated_auxiliary,
)
from invariant_gravity_extensions.reconstructed_axisymmetric import multipole_fields


def source(solver):
    r, mu, _ = solver.grid.nodes()
    R, z = r[:, None]*np.sqrt(1-mu*mu), r[:, None]*mu
    parts = (MassComponent("a", 1, 0, .5, 2), MassComponent("b", 2, 0, .5, -1))
    fields = total_newtonian(parts, R, z)
    return fields, normalized_newtonian_gradient(fields, r, mu, 1)


def test_angular_quadrature_change_evaluates_same_potential():
    grid = MultipoleGrid(1e-4, 100, 257, 64, 24, plane_scale=.3)
    part = MassComponent("disk", 1, 1, .3)
    potential = solve_poisson(grid, lambda R, z: part.fields(R, z)["laplacian"])
    flux = FluxPoissonSolver(MultipoleGrid(1e-4, 100, 257, 64, 24))
    r, mu, _ = flux.grid.nodes()
    fields = multipole_fields(potential, r[:, None]*np.sqrt(1-mu*mu), r[:, None]*mu)
    expected = normalized_newtonian_gradient(fields, r, mu, 1)
    np.testing.assert_allclose(gradient_on_flux_grid(potential, flux), expected, atol=1e-12, rtol=1e-10)


def test_spherical_source_has_no_physical_auxiliary_force():
    s = FluxPoissonSolver(MultipoleGrid(1e-4, 100, 257, 64, 24))
    r, mu, _ = s.grid.nodes()
    radial = np.broadcast_to(r[:, None]/(r[:, None]**2+.5**2)**1.5, (len(r), mu.size))
    p = np.array([radial, np.zeros_like(radial)])
    a = solve_isolated_auxiliary(s, p, 2, 2)
    assert np.max(abs(s.gradient(a.physical_flux_potential))) < 1e-12


def test_nonspherical_action_conserves_internal_force():
    s = FluxPoissonSolver(MultipoleGrid(1e-4, 200, 1025, 160, 64))
    fields, p = source(s)
    a = solve_isolated_auxiliary(s, p, 2, 1)
    gradient = s.gradient(a.physical_flux_potential)
    rho = fields["laplacian"]/(4*np.pi)
    az = -gradient[0]*s.mu+gradient[1]*s.sine
    norm = integrate_axisymmetric(s.grid, rho*np.linalg.norm(p, axis=0))
    assert abs(integrate_axisymmetric(s.grid, rho*az))/norm < 1e-4
    assert a.relative_equation_residual < 1e-9
    assert a.maximum_equation_residual < 1e-9


def test_homologous_length_rescaling_preserves_acceleration():
    s = FluxPoissonSolver(MultipoleGrid(1e-4, 100, 257, 64, 24))
    _, p = source(s)
    scaled = FluxPoissonSolver(MultipoleGrid(2e-4, 200, 257, 64, 24))
    a = solve_isolated_auxiliary(s, p, .5, 2)
    b = solve_isolated_auxiliary(scaled, p, .5, 2)
    np.testing.assert_allclose(b.q, a.q, atol=2e-12, rtol=1e-9)
    np.testing.assert_allclose(scaled.gradient(b.physical_flux_potential), s.gradient(a.physical_flux_potential), atol=2e-12, rtol=1e-8)


def test_unsupported_geometry_or_invalid_inputs_fail_explicitly():
    s = FluxPoissonSolver(MultipoleGrid(1e-4, 100, 257, 64, 24))
    _, p = source(s)
    with pytest.raises(ValueError):
        solve_isolated_auxiliary(s, p, 3, 1)
    with pytest.raises(ValueError):
        solve_isolated_auxiliary(s, p[:, :-1], 0, 1)
    with pytest.raises(RuntimeError):
        solve_isolated_auxiliary(s, p, 2, 1, max_iterations=2)


def test_beta_zero_density_source_uses_cylindrical_hessian_and_a0_units():
    s = FluxPoissonSolver(MultipoleGrid(1e-4, 200, 1025, 160, 64))
    fields, p = source(s)
    a0 = .3
    a = solve_isolated_auxiliary(s, p/a0, 0, 2)
    values = beta_zero_density_source(fields, a0, 2)
    independent = solve_poisson(s.grid, lambda R, z: values)
    reference = s.gradient(independent)
    assert s.energy_norm(a.q-reference)/s.energy_norm(a.q) < 1e-4
