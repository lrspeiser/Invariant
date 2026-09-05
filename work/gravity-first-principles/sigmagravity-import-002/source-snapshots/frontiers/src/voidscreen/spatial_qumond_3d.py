"""Registered 3D path-diluted QUMOND development operator.

This module implements the P0685 finite-volume equation

    laplacian(Phi) = div[nu_0(|grad Phi_N|/a0)^p grad Phi_N]

where ``p`` is calculated only from the baryonic Newtonian potential and
acceleration.  It is a phenomenological weak-field solver, not a covariant
gravity theory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voidscreen.field_solvers import (
    FieldSolution,
    acceleration_from_potential,
    acceleration_magnitude,
    cell_coordinates,
    gradient,
    laplacian,
    normalized_residual_rms,
    radial_boundary_from_acceleration,
    solve_newtonian,
    solve_poisson_dirichlet,
)
from voidscreen.potential_channel_qumond import (
    path_diluted_channel_exponent,
    rar_qumond_boost,
)

C_M_S = 299792458.0


@dataclass(frozen=True)
class SpatialQumondSolution:
    field: FieldSolution
    newtonian: FieldSolution
    potential_depth: np.ndarray
    potential_path_ratio: np.ndarray
    path_survival: np.ndarray
    channel_exponent: np.ndarray
    boundary_potential: np.ndarray
    center_of_mass: tuple[float, float, float]


def _spacing3(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        values = (float(spacing),) * 3
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("spacing must contain three positive values")
    return values


def _grid(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or min(array.shape) < 5:
        raise ValueError(f"{name} must be a 3D grid with at least five cells per axis")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def baryonic_center_of_mass(
    density: np.ndarray,
    spacing: float | Sequence[float],
) -> tuple[float, float, float]:
    rho = _grid(density, name="density")
    steps = _spacing3(spacing)
    weight = float(np.sum(rho))
    if weight <= 0.0:
        return (0.0, 0.0, 0.0)
    coordinates = cell_coordinates(rho.shape, steps)
    return tuple(float(np.sum(rho * coordinate) / weight) for coordinate in coordinates)


def path_geometry_from_newtonian(
    potential: np.ndarray,
    acceleration: Sequence[np.ndarray],
    spacing: float | Sequence[float],
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    transition_depth: float,
    transition_power: float,
    extra_spatial_channels: float,
    path_power: float,
    light_speed: float = C_M_S,
) -> dict[str, np.ndarray]:
    phi = _grid(potential, name="potential")
    steps = _spacing3(spacing)
    if len(acceleration) != 3 or any(np.asarray(item).shape != phi.shape for item in acceleration):
        raise ValueError("acceleration must contain three grids matching potential")
    if len(center) != 3 or light_speed <= 0.0:
        raise ValueError("center and light_speed are invalid")
    coordinates = cell_coordinates(phi.shape, steps)
    radius = np.sqrt(
        sum(
            np.square(coordinate - float(offset))
            for coordinate, offset in zip(coordinates, center, strict=True)
        )
    )
    magnitude = acceleration_magnitude(acceleration)
    absolute_potential = np.abs(phi)
    denominator_floor = max(
        float(np.max(absolute_potential)) * 1e-14,
        np.finfo(float).tiny,
    )
    path_ratio = absolute_potential / np.maximum(radius * magnitude, denominator_floor)
    path_ratio = np.maximum(path_ratio, np.finfo(float).tiny)
    depth = absolute_potential / float(light_speed) ** 2
    geometry = path_diluted_channel_exponent(
        depth,
        path_ratio,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=extra_spatial_channels,
        path_power=path_power,
    )
    return {
        "radius": radius,
        "potential_depth": depth,
        "potential_path_ratio": path_ratio,
        **geometry,
    }


def divergence_powered_qumond_gradient(
    potential: np.ndarray,
    channel_exponent: np.ndarray,
    spacing: float | Sequence[float],
    *,
    a0: float,
) -> np.ndarray:
    """Finite-volume divergence of ``nu_0^p grad(Phi_N)``.

    ``p`` is averaged to a face before the boost is evaluated there.  A zero
    face gradient has exactly zero flux, avoiding the divergent value of
    ``nu_0`` at the origin.
    """

    phi = _grid(potential, name="potential")
    exponent = _grid(channel_exponent, name="channel_exponent")
    if phi.shape != exponent.shape or a0 <= 0.0:
        raise ValueError("potential, exponent, or a0 is invalid")
    if np.any(exponent < 1.0):
        raise ValueError("channel_exponent must be at least one")
    steps = _spacing3(spacing)
    cell_gradient = gradient(phi, steps)
    result = np.zeros_like(phi)
    interior = (slice(1, -1),) * 3

    def face_flux(axis: int, neighbor: list[slice], step: float, direction: int):
        normal = direction * (phi[tuple(neighbor)] - phi[interior]) / step
        magnitude_squared = np.square(normal)
        for tangent_axis in range(3):
            if tangent_axis == axis:
                continue
            tangent = 0.5 * (
                cell_gradient[tangent_axis][interior] + cell_gradient[tangent_axis][tuple(neighbor)]
            )
            magnitude_squared += np.square(tangent)
        magnitude = np.sqrt(magnitude_squared)
        face_exponent = 0.5 * (exponent[interior] + exponent[tuple(neighbor)])
        multiplier = np.ones_like(magnitude)
        active = magnitude > 0.0
        multiplier[active] = np.power(
            rar_qumond_boost(magnitude[active], float(a0)),
            face_exponent[active],
        )
        return multiplier * normal

    for axis, step in enumerate(steps):
        left = [slice(1, -1)] * 3
        right = [slice(1, -1)] * 3
        left[axis] = slice(0, -2)
        right[axis] = slice(2, None)
        left_flux = face_flux(axis, left, step, -1)
        right_flux = face_flux(axis, right, step, 1)
        result[interior] += (right_flux - left_flux) / step
    return result


def path_qumond_monopole_boundary(
    density: np.ndarray,
    spacing: float | Sequence[float],
    *,
    gravitational_constant: float,
    a0: float,
    transition_depth: float,
    transition_power: float,
    extra_spatial_channels: float,
    path_power: float,
    light_speed: float = C_M_S,
) -> np.ndarray:
    rho = _grid(density, name="density")
    steps = _spacing3(spacing)
    total_mass = float(np.sum(rho) * np.prod(steps))
    center = baryonic_center_of_mass(rho, steps)

    def radial_acceleration(radius: np.ndarray) -> np.ndarray:
        newtonian = float(gravitational_constant) * total_mass / np.square(radius)
        depth = float(gravitational_constant) * total_mass / (radius * float(light_speed) ** 2)
        geometry = path_diluted_channel_exponent(
            depth,
            np.ones_like(radius),
            transition_depth=transition_depth,
            transition_power=transition_power,
            extra_spatial_channels=extra_spatial_channels,
            path_power=path_power,
        )
        return newtonian * np.power(
            rar_qumond_boost(newtonian, float(a0)),
            geometry["channel_exponent"],
        )

    return radial_boundary_from_acceleration(
        rho.shape,
        steps,
        radial_acceleration,
        center=center,
    )


def solve_path_diluted_qumond(
    density: np.ndarray,
    spacing: float | Sequence[float],
    *,
    gravitational_constant: float = 6.67430e-11,
    a0: float = 1.2e-10,
    transition_depth: float = 1e-6,
    transition_power: float = 4.0,
    extra_spatial_channels: float = 2.0,
    path_power: float = 0.5,
    light_speed: float = C_M_S,
    newtonian_boundary: np.ndarray | None = None,
    modified_boundary: np.ndarray | None = None,
) -> SpatialQumondSolution:
    rho = _grid(density, name="density")
    steps = _spacing3(spacing)
    center = baryonic_center_of_mass(rho, steps)
    newtonian = solve_newtonian(
        rho,
        steps,
        gravitational_constant=gravitational_constant,
        boundary_potential=newtonian_boundary,
    )
    geometry = path_geometry_from_newtonian(
        newtonian.potential,
        newtonian.acceleration,
        steps,
        center=center,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=extra_spatial_channels,
        path_power=path_power,
        light_speed=light_speed,
    )
    source = divergence_powered_qumond_gradient(
        newtonian.potential,
        geometry["channel_exponent"],
        steps,
        a0=a0,
    )
    boundary = (
        path_qumond_monopole_boundary(
            rho,
            steps,
            gravitational_constant=gravitational_constant,
            a0=a0,
            transition_depth=transition_depth,
            transition_power=transition_power,
            extra_spatial_channels=extra_spatial_channels,
            path_power=path_power,
            light_speed=light_speed,
        )
        if modified_boundary is None
        else _grid(modified_boundary, name="modified_boundary")
    )
    potential = solve_poisson_dirichlet(source, steps, boundary)
    residual = laplacian(potential, steps) - source
    residual_rms = normalized_residual_rms(residual, source)
    field = FieldSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=source,
        normalized_residual_rms=residual_rms,
        converged=bool(newtonian.converged and np.isfinite(residual_rms)),
        metadata={
            "law": "path-diluted potential-channel QUMOND",
            "spacing": steps,
            "a0": float(a0),
            "transition_depth": float(transition_depth),
            "transition_power": float(transition_power),
            "extra_spatial_channels": float(extra_spatial_channels),
            "path_power": float(path_power),
            "newtonian_residual_rms": newtonian.normalized_residual_rms,
        },
    )
    return SpatialQumondSolution(
        field=field,
        newtonian=newtonian,
        potential_depth=geometry["potential_depth"],
        potential_path_ratio=geometry["potential_path_ratio"],
        path_survival=geometry["path_survival"],
        channel_exponent=geometry["channel_exponent"],
        boundary_potential=boundary,
        center_of_mass=center,
    )
