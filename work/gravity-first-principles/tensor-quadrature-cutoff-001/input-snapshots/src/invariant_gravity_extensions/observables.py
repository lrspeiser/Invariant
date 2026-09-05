"""The same solved field drives matter and explicitly declared photon closures.

A static action is never silently treated as a relativistic lensing theory.
Born projection here is a synthetic slab check, not a cluster likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fields import FieldSolution, PeriodicGrid


class UnsupportedSectorError(ValueError):
    """The requested observable has no implemented physical closure."""


@dataclass
class WeakMetric:
    grid: PeriodicGrid
    psi: np.ndarray
    phi: np.ndarray
    speed_of_light: float
    provenance: dict[str, Any]


def assumed_metric(solution: FieldSolution, *, closure: str | None = None,
                   speed_of_light: float = 100.0) -> WeakMetric:
    """Opt-in only. 'assumed_no_slip' is not an action-derived photon sector."""
    if closure != "assumed_no_slip":
        raise UnsupportedSectorError("explicit assumed_no_slip required; covariant photon sector absent")
    if not np.isfinite(speed_of_light) or speed_of_light <= 0:
        raise ValueError("positive light speed required")
    psi = solution.physical
    if np.max(np.abs(psi))/speed_of_light**2 >= 0.01:
        raise ValueError("weak-metric expansion is outside its declared domain")
    return WeakMetric(solution.grid, psi, psi.copy(), speed_of_light, {
        "closure": closure,
        "derived_from_action": False,
        "parent_action_sha256": solution.spec.card()["content_sha256"],
        "scope": "synthetic_weak_metric_Born_projection",
    })


def born_lensing(metric: WeakMetric, *, axis: int = 2, distance_factor: float) -> dict[str, Any]:
    """Project one periodic box; distance_factor is D_l D_ls/D_s in grid units.

    No source redshift, distance geometry or closure is fitted or inferred here.
    The returned shear is not reduced shear. Periodic mock boundary conditions
    are carried in metadata and cannot be used as an isolated-lens prediction.
    """
    if axis not in {0, 1, 2} or not np.isfinite(distance_factor) or distance_factor <= 0:
        raise ValueError("valid line-of-sight axis and positive distance factor required")
    if not np.isfinite(metric.speed_of_light) or metric.speed_of_light <= 0:
        raise ValueError("positive light speed required")
    if max(np.max(np.abs(metric.psi)), np.max(np.abs(metric.phi)))/metric.speed_of_light**2 >= 0.01:
        raise ValueError("weak-metric expansion is outside its declared domain")
    g = metric.grid
    u = g.check(metric.psi)+g.check(metric.phi)
    transverse = [i for i in range(3) if i != axis]
    def project(a: np.ndarray) -> np.ndarray:
        return a.sum(axis=axis)*g.dx/metric.speed_of_light**2
    alpha = np.array([project(g.derivative(u, i)) for i in transverse])
    hh = np.array([[distance_factor*project(g.derivative(g.derivative(u, i), j))
                    for j in transverse] for i in transverse])
    return {
        "deflection": alpha,
        "convergence": 0.5*(hh[0, 0]+hh[1, 1]),
        "shear_1": 0.5*(hh[0, 0]-hh[1, 1]),
        "shear_2": hh[0, 1],
        "metadata": {**metric.provenance, "boundary": "periodic_synthetic_slab",
                     "distance_factor": distance_factor, "line_of_sight_axis": axis},
    }


def member_relative_acceleration(
    solution: FieldSolution, member_density: np.ndarray,
    *, uniform_external_acceleration: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the mass-weighted galaxy COM acceleration, not just its central value."""
    density = solution.grid.check(member_density)
    if np.any(density < 0) or density.sum() <= 0:
        raise ValueError("positive member mass required")
    acc = solution.acceleration.copy()
    if uniform_external_acceleration is not None:
        extra = np.asarray(uniform_external_acceleration, float)
        if extra.shape != (3,) or not np.all(np.isfinite(extra)):
            raise ValueError("external acceleration must be a finite 3-vector")
        acc += extra[:, None, None, None]
    centre = np.sum(acc*density[None, ...], axis=(1, 2, 3))/density.sum()
    return acc-centre[:, None, None, None], centre


def require_supported_sector(sector: str) -> None:
    """Do not silently manufacture unsupported links to the covariant compiler."""
    if sector not in {"static_periodic_matter", "explicit_assumed_Born_lensing"}:
        raise UnsupportedSectorError(f"{sector} has no executable adapter in this release")
