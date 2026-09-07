"""Time-domain auxiliary-field and trajectory controls, not snapshot proxies.

These controls are not MOND fits or relativistic completions. They expose time
support, initial conditions, and an energy-conserving example for discovery.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .fields import PeriodicGrid


def evolve_auxiliary(
    grid: PeriodicGrid, times: np.ndarray, sources: np.ndarray,
    q0: np.ndarray, velocity0: np.ndarray, *, speed: float = 1.0, mass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """q_tt + c_q^2(-lap + m^2)q=J with explicit initial state.

    sources[j] is held constant on [times[j], times[j+1]); the last source
    is intentionally unused. Exact oscillator updates have no unstable CFL
    branch. Spectral periodic discretization is not a causal-cone proof for
    a continuum/relativistic theory. No future source samples are read.
    """
    t = np.asarray(times, dtype=float)
    j = np.asarray(sources, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or
            np.any(np.diff(t) <= 0)):
        raise ValueError("strictly increasing finite times are required")
    if j.shape != (len(t), *grid.shape) or not np.all(np.isfinite(j)):
        raise ValueError("sources must cover every declared time on the grid")
    if not np.isfinite(speed) or speed <= 0 or not np.isfinite(mass) or mass < 0:
        raise ValueError("speed > 0 and mass >= 0 required")
    q = np.fft.fftn(grid.check(q0))
    v = np.fft.fftn(grid.check(velocity0))
    omega = speed*np.sqrt(np.sum(grid.wavevectors()**2, axis=0)+mass**2)
    qs, vs = [q0.copy()], [velocity0.copy()]
    for i, dt in enumerate(np.diff(t)):
        drive = np.fft.fftn(j[i])
        cs = np.cos(omega*dt)
        sv = dt*np.sinc(omega*dt/np.pi)
        # (1-cos(w*dt))/w^2, stable even at w=0.
        cv = 0.5*dt**2*np.sinc(omega*dt/(2*np.pi))**2
        q, v = cs*q+sv*v+cv*drive, -omega**2*sv*q+cs*v+sv*drive
        qs.append(np.fft.ifftn(q).real)
        vs.append(np.fft.ifftn(v).real)
    return np.asarray(qs), np.asarray(vs)


@dataclass(frozen=True)
class InertiaMemory:
    """Positive-kinetic local worldline control with an internal vector q.

    L/m = |v|^2/2 + mu|qdot|^2/2 - coupling*v.qdot
          - mu*omega^2|q|^2/2 - Phi(x).

    Eliminating q generates frequency/history dependence. This is a controlled
    auxiliary-oscillator example, NOT a derived relativistic MOND-inertia law.
    """
    coupling: float = 0.0
    mu: float = 1.0
    omega: float = 1.0

    def __post_init__(self) -> None:
        if any(not np.isfinite(v) for v in (self.coupling, self.mu, self.omega)):
            raise ValueError("finite coefficients required")
        if self.mu <= self.coupling**2 or self.omega <= 0:
            raise ValueError("positive kinetic matrix requires mu > coupling^2; omega > 0")

    def accelerations(self, force: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        force, q = np.asarray(force, float), np.asarray(q, float)
        if force.shape != q.shape or not np.all(np.isfinite(force)) or not np.all(np.isfinite(q)):
            raise ValueError("finite force and internal displacement with matching shape required")
        acc = (force-self.coupling*self.omega**2*q)/(1-self.coupling**2/self.mu)
        return acc, (self.coupling/self.mu)*acc-self.omega**2*q

    def integrate(
        self, force: Callable[[float, np.ndarray], np.ndarray], times: np.ndarray,
        position: np.ndarray, velocity: np.ndarray,
        q0: np.ndarray, qdot0: np.ndarray, *, rtol: float = 1e-9, atol: float = 1e-11,
    ) -> np.ndarray:
        """Return (time, [x,v,q,qdot], spatial dimension); no inferred initial state."""
        t = np.asarray(times, float)
        states = [np.asarray(a, float) for a in (position, velocity, q0, qdot0)]
        if (t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or
                np.any(np.diff(t) <= 0)):
            raise ValueError("ordered finite trajectory times are required")
        shape = states[0].shape
        if len(shape) != 1 or not shape[0] or any(a.shape != shape for a in states):
            raise ValueError("matching vector initial states required")
        if (any(not np.all(np.isfinite(a)) for a in states) or
                not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0 or atol <= 0):
            raise ValueError("invalid initial data or tolerances")
        dim = shape[0]

        def rhs(time: float, flat: np.ndarray) -> np.ndarray:
            x, v, q, w = flat.reshape(4, dim)
            a, qa = self.accelerations(force(time, x), q)
            return np.array([v, a, w, qa]).ravel()

        solved = solve_ivp(rhs, (t[0], t[-1]), np.asarray(states).ravel(), t_eval=t,
                           method="DOP853", rtol=rtol, atol=atol)
        if not solved.success or not np.all(np.isfinite(solved.y)):
            raise RuntimeError(f"trajectory integration failed: {solved.message}")
        return solved.y.T.reshape(len(t), 4, dim)

    def energy(self, trajectory: np.ndarray, potential: Callable[[np.ndarray], float]) -> np.ndarray:
        x, v, q, w = np.moveaxis(np.asarray(trajectory), 1, 0)
        return (0.5*np.sum(v*v, axis=1) + 0.5*self.mu*np.sum(w*w, axis=1)
                - self.coupling*np.sum(v*w, axis=1)
                + 0.5*self.mu*self.omega**2*np.sum(q*q, axis=1)
                + np.array([potential(xx) for xx in x]))
