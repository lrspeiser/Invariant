"""Gauge-safe source gating from baryonic Newtonian vector coherence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from voidscreen.coherent_monopole import CoherentMonopoleSolution
from voidscreen.field_solvers import (
    acceleration_from_potential,
    boundary_mask,
    laplacian,
    normalized_residual_rms,
    solve_poisson_dirichlet,
)


@dataclass(frozen=True)
class VectorCoherenceSolution:
    coherence: np.ndarray
    raw_coherence: np.ndarray
    direct_acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    unsummed_acceleration_strength: np.ndarray
    maximum_triangle_inequality_excess: float


@dataclass(frozen=True)
class CoherenceGatedSourceSolution:
    potential: np.ndarray
    acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    equation_source: np.ndarray
    coherence: np.ndarray
    coherent_source: np.ndarray
    local_source: np.ndarray
    normalized_residual_rms: float


@dataclass(frozen=True)
class HybridCoherenceRoutingSolution:
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
        raise ValueError("native pairwise coherence requires isotropic spacing")
    return steps


def baryonic_vector_coherence(
    density: np.ndarray,
    spacing: float | Sequence[float],
    *,
    gravitational_constant: float,
) -> VectorCoherenceSolution:
    """Return the fraction of pairwise Newtonian vector strength that survives summation.

    Both numerator and denominator use the same discrete, unsoftened pair
    kernel.  The self pair is zero.  Consequently the raw ratio obeys the
    triangle inequality apart from floating-point convolution roundoff.
    """

    rho = _grid(density, name="density", nonnegative=True)
    if float(np.sum(rho)) <= 0.0:
        raise ValueError("density must have positive mass")
    steps = _isotropic_spacing(spacing)
    gravitational_constant = float(gravitational_constant)
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0.0:
        raise ValueError("gravitational_constant must be finite and positive")
    kernel_axes = [
        np.arange(-(count - 1), count, dtype=float) * step
        for count, step in zip(rho.shape, steps, strict=True)
    ]
    displacement = np.meshgrid(*kernel_axes, indexing="ij")
    radius_squared = sum(component * component for component in displacement)
    active = radius_squared > 0.0
    inverse_radius_squared = np.zeros_like(radius_squared)
    inverse_radius_squared[active] = 1.0 / radius_squared[active]
    scalar_kernel = gravitational_constant * inverse_radius_squared
    inverse_radius_cubed = np.zeros_like(radius_squared)
    inverse_radius_cubed[active] = inverse_radius_squared[active] / np.sqrt(
        radius_squared[active]
    )
    vector_kernels = tuple(
        -gravitational_constant * component * inverse_radius_cubed
        for component in displacement
    )
    cell_volume = float(np.prod(steps))
    unsummed_strength = (
        fftconvolve(rho, scalar_kernel, mode="same") * cell_volume
    )
    direct_acceleration = tuple(
        fftconvolve(rho, kernel, mode="same") * cell_volume
        for kernel in vector_kernels
    )
    direct_magnitude = np.sqrt(
        sum(component * component for component in direct_acceleration)
    )
    raw = np.divide(
        direct_magnitude,
        unsummed_strength,
        out=np.zeros_like(direct_magnitude),
        where=unsummed_strength > 0.0,
    )
    maximum_excess = float(max(np.max(raw - 1.0), 0.0))
    coherence = np.clip(raw, 0.0, 1.0)
    return VectorCoherenceSolution(
        coherence=coherence,
        raw_coherence=raw,
        direct_acceleration=direct_acceleration,
        unsummed_acceleration_strength=unsummed_strength,
        maximum_triangle_inequality_excess=maximum_excess,
    )


def coherence_gated_source_potential(
    coherent_solution: CoherentMonopoleSolution,
    local_source: np.ndarray,
    coherence: np.ndarray,
    spacing: float | Sequence[float],
) -> CoherenceGatedSourceSolution:
    """Mix equation sources by coherence and solve with the coherent boundary."""

    local = _grid(local_source, name="local_source")
    controller = _grid(coherence, name="coherence")
    if local.shape != coherent_solution.potential.shape or controller.shape != local.shape:
        raise ValueError("coherent solution, local source, and coherence must share one grid")
    tolerance = 1e-12
    if np.min(controller) < -tolerance or np.max(controller) > 1.0 + tolerance:
        raise ValueError("coherence must lie in [0, 1] apart from roundoff")
    controller = np.clip(controller, 0.0, 1.0)
    steps = _isotropic_spacing(spacing)
    coherent_source = coherent_solution.equation_source
    mixed_source = controller * coherent_source + (1.0 - controller) * local
    potential = solve_poisson_dirichlet(
        mixed_source,
        steps,
        coherent_solution.potential,
    )
    residual = laplacian(potential, steps) - mixed_source
    return CoherenceGatedSourceSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=mixed_source,
        coherence=controller,
        coherent_source=coherent_source,
        local_source=local,
        normalized_residual_rms=normalized_residual_rms(residual, mixed_source),
    )


def hybrid_coherence_routing_potential(
    base_solution: CoherenceGatedSourceSolution,
    local_potential: np.ndarray,
    routed_potential: np.ndarray,
    spacing: float | Sequence[float],
    routing_fraction: float,
) -> HybridCoherenceRoutingSolution:
    """Add the zero-boundary routed-minus-local topology correction."""

    local = _grid(local_potential, name="local_potential")
    routed = _grid(routed_potential, name="routed_potential")
    if local.shape != base_solution.potential.shape or routed.shape != local.shape:
        raise ValueError("base, local, and routed potentials must share one grid")
    fraction = float(routing_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("routing_fraction must be finite and lie in [0, 1]")
    steps = _isotropic_spacing(spacing)
    correction = routed - local
    potential = base_solution.potential + fraction * correction
    return HybridCoherenceRoutingSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=laplacian(potential, steps),
        routing_correction_potential=correction,
        routing_fraction=fraction,
    )


def base_boundary_relative_mismatch(
    base_solution: CoherenceGatedSourceSolution,
    coherent_solution: CoherentMonopoleSolution,
) -> float:
    """Return the maximum relative boundary mismatch for audit code."""

    if base_solution.potential.shape != coherent_solution.potential.shape:
        raise ValueError("base and coherent solutions must share one grid")
    edge = boundary_mask(base_solution.potential.shape)
    scale = max(
        float(np.max(np.abs(coherent_solution.potential[edge]))),
        np.finfo(float).tiny,
    )
    return float(
        np.max(np.abs(base_solution.potential[edge] - coherent_solution.potential[edge]))
        / scale
    )
