"""Coherent-monopole completion and zero-boundary routing corrections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from voidscreen.field_solvers import (
    acceleration_from_potential,
    cell_coordinates,
    laplacian,
    simple_mond_acceleration,
)
from voidscreen.spatial_qumond_3d import baryonic_center_of_mass


@dataclass(frozen=True)
class CoherentMonopoleSolution:
    potential: np.ndarray
    acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    equation_source: np.ndarray
    correction_potential: np.ndarray
    correction_acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    center_of_mass: tuple[float, float, float]
    shell_radius: np.ndarray
    coherent_newtonian_acceleration: np.ndarray
    coherent_completed_acceleration: np.ndarray
    coherent_acceleration_correction: np.ndarray


@dataclass(frozen=True)
class HybridCoherentRoutingSolution:
    potential: np.ndarray
    acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    equation_source: np.ndarray
    routing_correction_potential: np.ndarray
    routing_fraction: float


def _grid(values, *, name: str, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or min(array.shape) < 5 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3D grid with at least five cells per axis")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _isotropic_spacing(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        steps = (float(spacing),) * 3
    else:
        steps = tuple(float(value) for value in spacing)
    if len(steps) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("spacing must contain three finite positive values")
    if not np.allclose(steps, steps[0], rtol=0.0, atol=1e-14 * steps[0]):
        raise ValueError("coherent native-grid shells require isotropic spacing")
    return steps


def coherent_monopole_potential(
    density: np.ndarray,
    newtonian_potential: np.ndarray,
    newtonian_acceleration: Sequence[np.ndarray],
    spacing: float | Sequence[float],
    *,
    a0: float,
) -> CoherentMonopoleSolution:
    """Boost only the shell-shared inward Newtonian monopole.

    The Newtonian potential supplies every measured multipole.  On each native
    radial shell, the mean inward Newtonian acceleration is completed with the
    simple algebraic low-acceleration relation.  Integrating only that scalar
    difference creates a curl-free radial correction without a smoothing or
    fitting scale.
    """

    rho = _grid(density, name="density", nonnegative=True)
    potential_n = _grid(newtonian_potential, name="newtonian_potential")
    if rho.shape != potential_n.shape:
        raise ValueError("density and newtonian_potential must have the same shape")
    if len(newtonian_acceleration) != 3:
        raise ValueError("newtonian_acceleration must contain three components")
    acceleration_n = tuple(
        _grid(component, name=f"newtonian_acceleration[{index}]")
        for index, component in enumerate(newtonian_acceleration)
    )
    if any(component.shape != rho.shape for component in acceleration_n):
        raise ValueError("newtonian acceleration components must match density")
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    steps = _isotropic_spacing(spacing)
    step = steps[0]
    center = baryonic_center_of_mass(rho, steps)
    coordinates = cell_coordinates(rho.shape, steps)
    displacement = tuple(
        coordinate - offset
        for coordinate, offset in zip(coordinates, center, strict=True)
    )
    radius = np.sqrt(sum(component * component for component in displacement))
    safe_radius = np.maximum(radius, np.finfo(float).tiny)
    radial_unit = tuple(component / safe_radius for component in displacement)
    inward_acceleration = -sum(
        component * direction
        for component, direction in zip(acceleration_n, radial_unit, strict=True)
    )
    shell_index = np.rint(radius / step).astype(int)
    shell_count = int(np.max(shell_index)) + 1
    shell_radius = np.arange(shell_count, dtype=float) * step
    coherent_newtonian = np.full(shell_count, np.nan, dtype=float)
    for shell in range(shell_count):
        members = shell_index == shell
        if np.any(members):
            coherent_newtonian[shell] = max(float(np.mean(inward_acceleration[members])), 0.0)
    coherent_newtonian[0] = 0.0
    finite_shells = np.flatnonzero(np.isfinite(coherent_newtonian))
    if finite_shells.size < 2:
        raise ValueError("native grid does not contain enough occupied radial shells")
    missing_shells = np.flatnonzero(~np.isfinite(coherent_newtonian))
    if missing_shells.size:
        coherent_newtonian[missing_shells] = np.interp(
            missing_shells,
            finite_shells,
            coherent_newtonian[finite_shells],
        )
    coherent_completed = simple_mond_acceleration(coherent_newtonian, float(a0))
    coherent_correction = coherent_completed - coherent_newtonian
    radial_correction_potential = cumulative_trapezoid(
        coherent_correction,
        shell_radius,
        initial=0.0,
    )
    correction_potential = np.interp(radius, shell_radius, radial_correction_potential)
    potential = potential_n + correction_potential
    correction_acceleration = acceleration_from_potential(correction_potential, steps)
    acceleration = acceleration_from_potential(potential, steps)
    return CoherentMonopoleSolution(
        potential=potential,
        acceleration=acceleration,
        equation_source=laplacian(potential, steps),
        correction_potential=correction_potential,
        correction_acceleration=correction_acceleration,
        center_of_mass=center,
        shell_radius=shell_radius,
        coherent_newtonian_acceleration=coherent_newtonian,
        coherent_completed_acceleration=coherent_completed,
        coherent_acceleration_correction=coherent_correction,
    )


def hybrid_coherent_routing_potential(
    coherent_solution: CoherentMonopoleSolution,
    local_potential: np.ndarray,
    routed_potential: np.ndarray,
    spacing: float | Sequence[float],
    routing_fraction: float,
) -> HybridCoherentRoutingSolution:
    """Add a declared fraction of routed-minus-local potential to the completion."""

    local = _grid(local_potential, name="local_potential")
    routed = _grid(routed_potential, name="routed_potential")
    if local.shape != coherent_solution.potential.shape or routed.shape != local.shape:
        raise ValueError("coherent, local, and routed potentials must share one grid")
    fraction = float(routing_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("routing_fraction must be finite and lie in [0, 1]")
    steps = _isotropic_spacing(spacing)
    correction = routed - local
    potential = coherent_solution.potential + fraction * correction
    return HybridCoherentRoutingSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=laplacian(potential, steps),
        routing_correction_potential=correction,
        routing_fraction=fraction,
    )
