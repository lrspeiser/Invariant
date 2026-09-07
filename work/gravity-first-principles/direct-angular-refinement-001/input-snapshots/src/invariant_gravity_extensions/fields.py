"""Variationally consistent periodic 3-D controls for QUMOND extensions.

These are density-CONTRAST problems, not isolated clusters. A nonzero source
mean is an error, never silently turned into a compensating background. Odd
meshes avoid the real-valued Nyquist derivative ambiguity. Real observations
and isolated/nested boundary conditions are outside this adapter's scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .actions import ActionSpec


@dataclass(frozen=True)
class PeriodicGrid:
    n: int = 17
    length: float = 12.0

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n < 5 or self.n % 2 != 1:
            raise ValueError("use an odd grid size >=5")
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("length must be positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n,) * 3

    @property
    def dx(self) -> float:
        return self.length / self.n

    def coordinates(self) -> np.ndarray:
        x = (np.arange(self.n) - (self.n-1)/2) * self.dx
        return np.array(np.meshgrid(x, x, x, indexing="ij"))

    def wavevectors(self) -> np.ndarray:
        k = 2*np.pi*np.fft.fftfreq(self.n, self.dx)
        return np.array(np.meshgrid(k, k, k, indexing="ij"))

    def check(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.shape != self.shape or not np.all(np.isfinite(a)):
            raise ValueError("field has wrong shape or nonfinite entries")
        return a

    def derivative(self, a: np.ndarray, axis: int) -> np.ndarray:
        if axis not in {0, 1, 2}:
            raise ValueError("invalid derivative axis")
        a = self.check(a)
        return np.fft.ifftn(1j*self.wavevectors()[axis]*np.fft.fftn(a)).real

    def gradient(self, a: np.ndarray) -> np.ndarray:
        return np.array([self.derivative(a, i) for i in range(3)])

    def divergence(self, v: np.ndarray) -> np.ndarray:
        if np.shape(v) != (3, *self.shape):
            raise ValueError("vector field has wrong shape")
        return sum(self.derivative(v[i], i) for i in range(3))

    def hessian(self, a: np.ndarray) -> np.ndarray:
        g = self.gradient(a)
        return np.array([[self.derivative(g[i], j) for j in range(3)] for i in range(3)])

    def laplacian(self, a: np.ndarray) -> np.ndarray:
        a = self.check(a)
        return np.fft.ifftn(-np.sum(self.wavevectors()**2, axis=0)*np.fft.fftn(a)).real

    def poisson(self, source: np.ndarray) -> np.ndarray:
        """Solve lap(phi)=source with zero potential mean; require zero source mean."""
        source = self.check(source)
        scale = max(float(np.sqrt(np.mean(source**2))), np.finfo(float).tiny)
        if abs(float(source.mean())) > 1e-11*scale:
            raise ValueError("periodic Poisson needs zero-mean density contrast; no implicit subtraction")
        k2 = np.sum(self.wavevectors()**2, axis=0)
        rhs = np.fft.fftn(source)
        out = np.zeros_like(rhs)
        np.divide(-rhs, k2, out=out, where=k2 > 0)
        return np.fft.ifftn(out).real


@dataclass
class FieldSolution:
    spec: ActionSpec
    grid: PeriodicGrid
    newtonian: np.ndarray
    physical: np.ndarray
    auxiliary: np.ndarray | None
    diagnostics: dict[str, Any]

    @property
    def acceleration(self) -> np.ndarray:
        return -self.grid.gradient(self.physical)


def solve_fields(
    grid: PeriodicGrid, density_contrast: np.ndarray, spec: ActionSpec,
    *, a0: float = 1.0, four_pi_G: float = 1.0, tolerance: float = 1e-9,
    maxiter: int = 1000,
) -> FieldSolution:
    """Solve the whole source jointly; no superposition of modified member fields.

    TRIMOND solves div(F_y grad chi + F_z grad psi)=0 followed by the
    physical Poisson solve. For this specific quadratic auxiliary subclass,
    the chi equation is SPD after reversing sign and fixing its zero mode.

    GQUMOND includes BOTH first- and second-derivative Euler terms. Dropping
    the Hessian double divergence is not an allowed approximation.
    """
    for value in (a0, four_pi_G, tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("a0, four_pi_G and tolerance must be positive")
    if type(maxiter) is not int or maxiter <= 0:
        raise ValueError("maxiter must be positive")
    rho = grid.check(density_contrast)
    psi = grid.poisson(four_pi_G*rho)
    p = grid.gradient(psi)
    x = np.sum(p*p, axis=0)/a0**2
    chi = None
    aux_residual = 0.0
    iterations = 0
    if spec.family == "trimond_alignment":
        s = spec.mixing/(1+x)**spec.power
        weight = spec.beta/(1+x)**2

        def multiply(v: np.ndarray) -> np.ndarray:
            dot = np.sum(p*v, axis=0)/a0**2
            return v + weight[None, ...]*(x[None, ...]*v-p*dot[None, ...])

        def op(flat: np.ndarray) -> np.ndarray:
            a = flat.reshape(grid.shape)
            return (-grid.divergence(multiply(grid.gradient(a))) + a.mean()).ravel()

        def precondition(flat: np.ndarray) -> np.ndarray:
            a = flat.reshape(grid.shape)
            return (-grid.poisson(a-a.mean()) + a.mean()).ravel()

        rhs = -grid.divergence(s[None, ...]*p)
        count = [0]

        def tick(_: np.ndarray) -> None:
            count[0] += 1

        size = grid.n**3
        operator = LinearOperator((size, size), matvec=op, dtype=float)
        pre = LinearOperator((size, size), matvec=precondition, dtype=float)
        flat, info = cg(operator, rhs.ravel(), M=pre, rtol=tolerance, atol=0,
                        maxiter=maxiter, callback=tick)
        if info != 0:
            raise RuntimeError(f"auxiliary solve did not converge: cg info={info}")
        chi = flat.reshape(grid.shape)
        chi -= chi.mean()
        q = grid.gradient(chi)
        y = np.sum(q*q, axis=0)/a0**2
        z = 2*np.sum(p*q, axis=0)/a0**2
        fx, fy, fz, _ = spec.partials(x, y, z)
        source = grid.divergence(fx[None, ...]*p + fz[None, ...]*q)
        err = grid.divergence(fy[None, ...]*q+fz[None, ...]*p)
        aux_residual = float(np.linalg.norm(err)/max(np.linalg.norm(rhs), 1e-30))
        iterations = count[0]
    elif spec.family == "gqumond_length":
        hessian = grid.hessian(psi)
        h = spec.length**2*np.sum(hessian**2, axis=(0, 1))/a0**2
        px, _, _, ph = spec.partials(x, h=h)
        source = grid.divergence(px[None, ...]*p)
        for i in range(3):
            for j in range(3):
                source -= spec.length**2 * grid.derivative(
                    grid.derivative(ph*hessian[i, j], j), i)
    else:
        px, _, _, _ = spec.partials(x)
        source = grid.divergence(px[None, ...]*p)
    physical = grid.poisson(source)
    residual = float(np.linalg.norm(grid.laplacian(physical)-source) /
                     max(np.linalg.norm(source), 1e-30))
    if not np.isfinite(residual) or residual > max(1e-10, 10*tolerance):
        raise RuntimeError("physical Poisson residual exceeded tolerance")
    if aux_residual > 20*tolerance:
        raise RuntimeError("auxiliary Euler residual exceeded tolerance")
    return FieldSolution(spec, grid, psi, physical, chi, {
        "poisson_relative_residual": residual,
        "auxiliary_relative_residual": aux_residual,
        "auxiliary_iterations": iterations,
        "boundary": "periodic_zero_mean_density_contrast",
        "source_mean": float(rho.mean()),
        "claim_ceiling": "static_discrete_synthetic_solution_only",
        "relativistic_completion": "unsupported",
        "isolated_cluster_prediction": False,
    })


def joint_density(
    grid: PeriodicGrid, components: dict[str, np.ndarray], *, subtract_background: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve named components but solve their SUM, not a sum of modified forces.

    subtract_background is explicit and restricted to synthetic periodic scenes.
    The returned background is recorded; no isolated-mass claim is possible.
    """
    if not components or any(not isinstance(k, str) or not k for k in components):
        raise ValueError("named source components are required")
    arrays = [grid.check(components[k]) for k in sorted(components)]
    if any(np.any(v < 0) for v in arrays):
        raise ValueError("component densities must be nonnegative")
    total = np.sum(arrays, axis=0)
    background = float(total.mean()) if subtract_background else 0.0
    return total-background, {
        "component_ids": sorted(components),
        "component_integrals": {k: float(np.sum(components[k])*grid.dx**3)
                                for k in sorted(components)},
        "subtracted_homogeneous_background": background,
        "scope": "synthetic_periodic_scene" if subtract_background else "unsubtracted_sources",
    }
