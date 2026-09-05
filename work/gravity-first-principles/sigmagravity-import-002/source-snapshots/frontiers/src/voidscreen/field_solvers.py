"""Cartesian Newtonian, QUMOND, and AQUAL field solvers.

The solvers operate on a common three-dimensional uniform density grid. A
resolved two-dimensional surface-density map can be lifted onto that grid with
``surface_density_to_volume``. Dirichlet Poisson equations are solved with the
eigenvectors of the second-order finite-difference Laplacian; AQUAL uses a
finite-volume Picard iteration with a conjugate-gradient linear solve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dstn, idstn
from scipy.integrate import cumulative_trapezoid
from scipy.sparse.linalg import LinearOperator, cg

Array = np.ndarray


@dataclass(frozen=True)
class FieldSolution:
    potential: Array
    acceleration: tuple[Array, Array, Array]
    equation_source: Array
    normalized_residual_rms: float
    converged: bool
    nonlinear_iterations: int = 0
    metadata: dict = field(default_factory=dict)


def _spacing3(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        values = (float(spacing),) * 3
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("spacing must contain three positive values")
    return values


def _validate_grid(values: Array) -> Array:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or min(array.shape) < 5:
        raise ValueError("field grids must be three-dimensional with at least five cells per axis")
    if not np.all(np.isfinite(array)):
        raise ValueError("field grids must be finite")
    return array


def cell_coordinates(
    shape: Sequence[int], spacing: float | Sequence[float]
) -> tuple[Array, Array, Array]:
    steps = _spacing3(spacing)
    if len(shape) != 3:
        raise ValueError("shape must contain three axes")
    axes = [
        (np.arange(int(count), dtype=float) - (int(count) - 1.0) / 2.0) * step
        for count, step in zip(shape, steps, strict=True)
    ]
    return tuple(np.meshgrid(*axes, indexing="ij"))


def boundary_mask(shape: Sequence[int]) -> Array:
    mask = np.zeros(tuple(shape), dtype=bool)
    for axis in range(3):
        leading = [slice(None)] * 3
        trailing = [slice(None)] * 3
        leading[axis] = 0
        trailing[axis] = -1
        mask[tuple(leading)] = True
        mask[tuple(trailing)] = True
    return mask


def surface_density_to_volume(
    surface_density: Array,
    z_coordinates: Array,
    *,
    scale_height: float,
) -> Array:
    """Lift a 2D column-density map into a normalized sech-squared layer."""

    surface = np.asarray(surface_density, dtype=float)
    z = np.asarray(z_coordinates, dtype=float)
    if surface.ndim != 2 or z.ndim != 1 or z.size < 5:
        raise ValueError("surface_density must be 2D and z_coordinates must be a 1D grid")
    if not np.all(np.isfinite(surface)) or np.any(surface < 0.0):
        raise ValueError("surface density must be finite and non-negative")
    if not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0.0):
        raise ValueError("z_coordinates must be finite and strictly increasing")

    dz = float(np.median(np.diff(z)))
    if scale_height <= 0.0:
        weights = np.zeros_like(z)
        weights[int(np.argmin(np.abs(z)))] = 1.0 / dz
    else:
        weights = 1.0 / np.square(np.cosh(z / float(scale_height)))
        weights /= float(np.sum(weights) * dz)
    return surface[:, :, None] * weights[None, None, :]


def gradient(potential: Array, spacing: float | Sequence[float]) -> tuple[Array, Array, Array]:
    steps = _spacing3(spacing)
    values = _validate_grid(potential)
    return tuple(np.gradient(values, *steps, edge_order=2))


def acceleration_from_potential(
    potential: Array, spacing: float | Sequence[float]
) -> tuple[Array, Array, Array]:
    return tuple(-component for component in gradient(potential, spacing))


def acceleration_magnitude(acceleration: Sequence[Array]) -> Array:
    if len(acceleration) != 3:
        raise ValueError("acceleration must contain three components")
    return np.sqrt(sum(np.square(np.asarray(component, dtype=float)) for component in acceleration))


def laplacian(potential: Array, spacing: float | Sequence[float]) -> Array:
    values = _validate_grid(potential)
    steps = _spacing3(spacing)
    result = np.zeros_like(values)
    interior = (slice(1, -1),) * 3
    center = values[interior]
    for axis, step in enumerate(steps):
        before = [slice(1, -1)] * 3
        after = [slice(1, -1)] * 3
        before[axis] = slice(0, -2)
        after[axis] = slice(2, None)
        result[interior] += (
            values[tuple(before)] - 2.0 * center + values[tuple(after)]
        ) / step**2
    return result


def normalized_residual_rms(residual: Array, source: Array) -> float:
    interior = (slice(1, -1),) * 3
    numerator = float(np.sqrt(np.mean(np.square(residual[interior]))))
    denominator = float(np.sqrt(np.mean(np.square(source[interior]))))
    return numerator / max(denominator, np.finfo(float).tiny)


def solve_poisson_dirichlet(
    source: Array,
    spacing: float | Sequence[float],
    boundary_potential: Array,
) -> Array:
    """Solve the second-order Cartesian Poisson equation with fixed boundaries."""

    rhs_full = _validate_grid(source)
    boundary = _validate_grid(boundary_potential)
    if rhs_full.shape != boundary.shape:
        raise ValueError("source and boundary_potential must have the same shape")
    steps = _spacing3(spacing)
    shape = rhs_full.shape
    rhs = rhs_full[1:-1, 1:-1, 1:-1].copy()

    rhs[0, :, :] -= boundary[0, 1:-1, 1:-1] / steps[0] ** 2
    rhs[-1, :, :] -= boundary[-1, 1:-1, 1:-1] / steps[0] ** 2
    rhs[:, 0, :] -= boundary[1:-1, 0, 1:-1] / steps[1] ** 2
    rhs[:, -1, :] -= boundary[1:-1, -1, 1:-1] / steps[1] ** 2
    rhs[:, :, 0] -= boundary[1:-1, 1:-1, 0] / steps[2] ** 2
    rhs[:, :, -1] -= boundary[1:-1, 1:-1, -1] / steps[2] ** 2

    transformed = dstn(rhs, type=1, norm="ortho")
    eigenvalues = []
    for count, step in zip(shape, steps, strict=True):
        mode = np.arange(1, count - 1, dtype=float)
        eigenvalues.append(2.0 * (np.cos(np.pi * mode / (count - 1)) - 1.0) / step**2)
    denominator = (
        eigenvalues[0][:, None, None]
        + eigenvalues[1][None, :, None]
        + eigenvalues[2][None, None, :]
    )
    interior_solution = idstn(transformed / denominator, type=1, norm="ortho")
    potential = np.asarray(boundary, dtype=float).copy()
    potential[1:-1, 1:-1, 1:-1] = interior_solution
    return potential


def _center_of_mass(density: Array, spacing: tuple[float, float, float]) -> tuple[float, ...]:
    coordinates = cell_coordinates(density.shape, spacing)
    mass = float(np.sum(density))
    if mass <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(float(np.sum(density * coordinate) / mass) for coordinate in coordinates)


def newtonian_monopole_boundary(
    density: Array,
    spacing: float | Sequence[float],
    *,
    gravitational_constant: float,
) -> Array:
    rho = _validate_grid(density)
    steps = _spacing3(spacing)
    mass = float(np.sum(rho) * np.prod(steps))
    center = _center_of_mass(rho, steps)
    coordinates = cell_coordinates(rho.shape, steps)
    radius = np.sqrt(
        sum(np.square(coordinate - offset) for coordinate, offset in zip(coordinates, center))
    )
    safe_radius = np.maximum(radius, min(steps) / 2.0)
    potential = -float(gravitational_constant) * mass / safe_radius
    return np.where(boundary_mask(rho.shape), potential, 0.0)


def radial_boundary_from_acceleration(
    shape: Sequence[int],
    spacing: float | Sequence[float],
    acceleration_function: Callable[[Array], Array],
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    samples: int = 8192,
) -> Array:
    """Construct a Dirichlet boundary whose radial derivative is ``g(r)``."""

    steps = _spacing3(spacing)
    coordinates = cell_coordinates(shape, steps)
    radius = np.sqrt(
        sum(
            np.square(coordinate - float(offset))
            for coordinate, offset in zip(coordinates, center, strict=True)
        )
    )
    mask = boundary_mask(shape)
    lower = float(np.min(radius[mask]))
    upper = float(np.max(radius[mask]))
    radial_grid = np.linspace(lower, upper, int(samples))
    radial_acceleration = np.asarray(acceleration_function(radial_grid), dtype=float)
    if np.any(radial_acceleration < 0.0) or not np.all(np.isfinite(radial_acceleration)):
        raise ValueError("radial acceleration must be finite and non-negative")
    potential_grid = cumulative_trapezoid(radial_acceleration, radial_grid, initial=0.0)
    potential = np.interp(radius, radial_grid, potential_grid)
    return np.where(mask, potential, 0.0)


def simple_mu(x: Array) -> Array:
    values = np.maximum(np.asarray(x, dtype=float), 0.0)
    return values / (1.0 + values)


def simple_nu(y: Array) -> Array:
    values = np.maximum(np.asarray(y, dtype=float), np.finfo(float).tiny)
    return 0.5 + np.sqrt(0.25 + 1.0 / values)


def simple_mond_acceleration(newtonian_acceleration: Array, a0: float) -> Array:
    values = np.maximum(np.asarray(newtonian_acceleration, dtype=float), 0.0)
    return 0.5 * (values + np.sqrt(np.square(values) + 4.0 * float(a0) * values))


def simple_mond_monopole_boundary(
    density: Array,
    spacing: float | Sequence[float],
    *,
    gravitational_constant: float,
    a0: float,
) -> Array:
    rho = _validate_grid(density)
    steps = _spacing3(spacing)
    mass = float(np.sum(rho) * np.prod(steps))
    center = _center_of_mass(rho, steps)

    def radial_g(radius: Array) -> Array:
        g_newton = float(gravitational_constant) * mass / np.square(radius)
        return simple_mond_acceleration(g_newton, a0)

    return radial_boundary_from_acceleration(rho.shape, steps, radial_g, center=center)


def solve_newtonian(
    density: Array,
    spacing: float | Sequence[float],
    *,
    gravitational_constant: float = 6.67430e-11,
    boundary_potential: Array | None = None,
) -> FieldSolution:
    rho = _validate_grid(density)
    steps = _spacing3(spacing)
    boundary = (
        newtonian_monopole_boundary(
            rho,
            steps,
            gravitational_constant=gravitational_constant,
        )
        if boundary_potential is None
        else _validate_grid(boundary_potential)
    )
    source = 4.0 * np.pi * float(gravitational_constant) * rho
    potential = solve_poisson_dirichlet(source, steps, boundary)
    residual = laplacian(potential, steps) - source
    score = normalized_residual_rms(residual, source)
    return FieldSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=source,
        normalized_residual_rms=score,
        converged=bool(np.isfinite(score)),
        metadata={"law": "Newtonian Poisson", "spacing": steps},
    )


def divergence_scaled_gradient(
    potential: Array,
    coefficient: Array,
    spacing: float | Sequence[float],
) -> Array:
    """Finite-volume divergence of a cell-centered coefficient times grad(phi)."""

    phi = _validate_grid(potential)
    scale = _validate_grid(coefficient)
    if phi.shape != scale.shape:
        raise ValueError("potential and coefficient must have the same shape")
    steps = _spacing3(spacing)
    result = np.zeros_like(phi)
    interior = (slice(1, -1),) * 3
    for axis, step in enumerate(steps):
        left_neighbor = [slice(1, -1)] * 3
        right_neighbor = [slice(1, -1)] * 3
        left_neighbor[axis] = slice(0, -2)
        right_neighbor[axis] = slice(2, None)
        center = phi[interior]
        left_mu = 0.5 * (scale[interior] + scale[tuple(left_neighbor)])
        right_mu = 0.5 * (scale[interior] + scale[tuple(right_neighbor)])
        left_flux = left_mu * (center - phi[tuple(left_neighbor)]) / step
        right_flux = right_mu * (phi[tuple(right_neighbor)] - center) / step
        result[interior] += (right_flux - left_flux) / step
    return result


def divergence_qumond_gradient(
    potential: Array,
    spacing: float | Sequence[float],
    *,
    a0: float,
    nu_function: Callable[[Array], Array] = simple_nu,
) -> Array:
    """Finite-volume divergence of ``nu(|grad(phi)|/a0) grad(phi)``.

    QUMOND's simple ``nu`` function diverges at a zero Newtonian field, while
    the boosted flux itself has a finite zero-field limit.  Building the flux
    on cell faces avoids multiplying a divergent cell-centred coefficient by
    a neighbouring non-zero gradient at symmetry points.
    """

    phi = _validate_grid(potential)
    steps = _spacing3(spacing)
    cell_gradient = gradient(phi, steps)
    result = np.zeros_like(phi)
    interior = (slice(1, -1),) * 3

    def face_flux(
        axis: int, neighbor: list[slice], step: float, direction: int
    ) -> Array:
        normal_gradient = direction * (phi[tuple(neighbor)] - phi[interior]) / step
        magnitude_squared = np.square(normal_gradient)
        for tangent_axis in range(3):
            if tangent_axis == axis:
                continue
            tangent_gradient = 0.5 * (
                cell_gradient[tangent_axis][interior]
                + cell_gradient[tangent_axis][tuple(neighbor)]
            )
            magnitude_squared += np.square(tangent_gradient)
        magnitude = np.sqrt(magnitude_squared)
        multiplier = np.ones_like(magnitude)
        active = magnitude > 0.0
        multiplier[active] = np.asarray(
            nu_function(magnitude[active] / float(a0)), dtype=float
        )
        return multiplier * normal_gradient

    for axis, step in enumerate(steps):
        left_neighbor = [slice(1, -1)] * 3
        right_neighbor = [slice(1, -1)] * 3
        left_neighbor[axis] = slice(0, -2)
        right_neighbor[axis] = slice(2, None)
        left_flux = face_flux(axis, left_neighbor, step, -1)
        right_flux = face_flux(axis, right_neighbor, step, 1)
        result[interior] += (right_flux - left_flux) / step
    return result


def solve_qumond(
    density: Array,
    spacing: float | Sequence[float],
    *,
    a0: float = 1.2e-10,
    gravitational_constant: float = 6.67430e-11,
    newtonian_boundary: Array | None = None,
    mond_boundary: Array | None = None,
    nu_function: Callable[[Array], Array] = simple_nu,
) -> FieldSolution:
    """Solve QUMOND's Newtonian and modified Poisson equations on one grid."""

    rho = _validate_grid(density)
    steps = _spacing3(spacing)
    newtonian = solve_newtonian(
        rho,
        steps,
        gravitational_constant=gravitational_constant,
        boundary_potential=newtonian_boundary,
    )
    qumond_source = divergence_qumond_gradient(
        newtonian.potential,
        steps,
        a0=a0,
        nu_function=nu_function,
    )
    boundary = (
        simple_mond_monopole_boundary(
            rho,
            steps,
            gravitational_constant=gravitational_constant,
            a0=a0,
        )
        if mond_boundary is None
        else _validate_grid(mond_boundary)
    )
    potential = solve_poisson_dirichlet(qumond_source, steps, boundary)
    residual = laplacian(potential, steps) - qumond_source
    score = normalized_residual_rms(residual, qumond_source)
    return FieldSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=qumond_source,
        normalized_residual_rms=score,
        converged=bool(newtonian.converged and np.isfinite(score)),
        metadata={
            "law": "QUMOND",
            "a0": float(a0),
            "spacing": steps,
            "newtonian_residual_rms": newtonian.normalized_residual_rms,
        },
    )


def _zero_boundary_field(interior_values: Array, shape: tuple[int, int, int]) -> Array:
    values = np.zeros(shape, dtype=float)
    values[1:-1, 1:-1, 1:-1] = np.asarray(interior_values).reshape(
        tuple(count - 2 for count in shape)
    )
    return values


def _linear_variable_coefficient_solve(
    source: Array,
    coefficient: Array,
    spacing: tuple[float, float, float],
    boundary: Array,
    initial: Array,
    *,
    relative_tolerance: float,
    maximum_iterations: int,
) -> tuple[Array, int]:
    shape = source.shape
    interior_shape = tuple(count - 2 for count in shape)
    unknowns = int(np.prod(interior_shape))

    boundary_field = np.zeros(shape, dtype=float)
    boundary_field[boundary_mask(shape)] = boundary[boundary_mask(shape)]
    boundary_effect = -divergence_scaled_gradient(boundary_field, coefficient, spacing)[
        1:-1, 1:-1, 1:-1
    ]
    rhs = -source[1:-1, 1:-1, 1:-1] - boundary_effect

    def matvec(vector: Array) -> Array:
        field_values = _zero_boundary_field(vector, shape)
        applied = -divergence_scaled_gradient(field_values, coefficient, spacing)
        return applied[1:-1, 1:-1, 1:-1].ravel()

    diagonal = np.zeros(interior_shape, dtype=float)
    interior = (slice(1, -1),) * 3
    for axis, step in enumerate(spacing):
        left_neighbor = [slice(1, -1)] * 3
        right_neighbor = [slice(1, -1)] * 3
        left_neighbor[axis] = slice(0, -2)
        right_neighbor[axis] = slice(2, None)
        diagonal += (
            0.5 * (coefficient[interior] + coefficient[tuple(left_neighbor)])
            + 0.5 * (coefficient[interior] + coefficient[tuple(right_neighbor)])
        ) / step**2
    inverse_diagonal = 1.0 / np.maximum(diagonal.ravel(), np.finfo(float).tiny)
    operator = LinearOperator((unknowns, unknowns), matvec=matvec, dtype=float)
    preconditioner = LinearOperator(
        (unknowns, unknowns), matvec=lambda vector: inverse_diagonal * vector, dtype=float
    )
    solution, info = cg(
        operator,
        rhs.ravel(),
        x0=initial[1:-1, 1:-1, 1:-1].ravel(),
        rtol=float(relative_tolerance),
        atol=0.0,
        maxiter=int(maximum_iterations),
        M=preconditioner,
    )
    if info < 0:
        raise RuntimeError("AQUAL conjugate-gradient solve failed")
    potential = np.asarray(boundary, dtype=float).copy()
    potential[1:-1, 1:-1, 1:-1] = solution.reshape(interior_shape)
    return potential, int(info)


def solve_aqual(
    density: Array,
    spacing: float | Sequence[float],
    *,
    a0: float = 1.2e-10,
    gravitational_constant: float = 6.67430e-11,
    boundary_potential: Array | None = None,
    mu_function: Callable[[Array], Array] = simple_mu,
    residual_tolerance: float = 1e-5,
    maximum_nonlinear_iterations: int = 80,
    maximum_linear_iterations: int = 2000,
    linear_relative_tolerance: float = 1e-9,
    damping: float = 0.65,
    mu_floor: float = 1e-6,
) -> FieldSolution:
    """Solve AQUAL with finite-volume Picard iterations and fixed boundaries."""

    rho = _validate_grid(density)
    steps = _spacing3(spacing)
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    source = 4.0 * np.pi * float(gravitational_constant) * rho
    boundary = (
        simple_mond_monopole_boundary(
            rho,
            steps,
            gravitational_constant=gravitational_constant,
            a0=a0,
        )
        if boundary_potential is None
        else _validate_grid(boundary_potential)
    )
    potential = solve_qumond(
        rho,
        steps,
        a0=a0,
        gravitational_constant=gravitational_constant,
        mond_boundary=boundary,
    ).potential

    score = np.inf
    linear_info: list[int] = []
    residual_history: list[float] = []
    for iteration in range(1, int(maximum_nonlinear_iterations) + 1):
        field_strength = acceleration_magnitude(gradient(potential, steps))
        coefficient = np.maximum(
            np.asarray(mu_function(field_strength / float(a0)), dtype=float),
            float(mu_floor),
        )
        candidate, info = _linear_variable_coefficient_solve(
            source,
            coefficient,
            steps,
            boundary,
            potential,
            relative_tolerance=linear_relative_tolerance,
            maximum_iterations=maximum_linear_iterations,
        )
        linear_info.append(info)
        potential = (1.0 - float(damping)) * potential + float(damping) * candidate
        potential[boundary_mask(rho.shape)] = boundary[boundary_mask(rho.shape)]

        updated_strength = acceleration_magnitude(gradient(potential, steps))
        updated_coefficient = np.maximum(
            np.asarray(mu_function(updated_strength / float(a0)), dtype=float),
            float(mu_floor),
        )
        residual = divergence_scaled_gradient(potential, updated_coefficient, steps) - source
        score = normalized_residual_rms(residual, source)
        residual_history.append(float(score))
        if score <= float(residual_tolerance):
            break

    return FieldSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=source,
        normalized_residual_rms=float(score),
        converged=bool(score <= float(residual_tolerance) and all(info == 0 for info in linear_info)),
        nonlinear_iterations=iteration,
        metadata={
            "law": "AQUAL",
            "a0": float(a0),
            "spacing": steps,
            "mu_floor": float(mu_floor),
            "linear_info": linear_info,
            "residual_history": residual_history,
        },
    )
