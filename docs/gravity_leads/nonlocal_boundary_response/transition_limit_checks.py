"""Analytic-limit smoke checks for the screened nonlocal boundary scaffold.

The numbers below are illustrative dimensionless regimes, not fitted astrophysical
parameters and not empirical evidence. The checks only verify the intended limiting
behavior of the proposed transition and lensing branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite


@dataclass(frozen=True)
class TransitionExponents:
    compactness: float = 1.0
    boundary: float = 1.0
    thermal: float = 1.0
    high_acceleration: float = 2.0
    environment: float = 2.0


def environmental_state(
    compactness_ratio: float,
    boundary_ratio: float,
    thermal_ratio: float,
    exponents: TransitionExponents = TransitionExponents(),
) -> float:
    """Return zeta using already normalized, dimensionless source variables."""
    values = (compactness_ratio, boundary_ratio, thermal_ratio)
    if any(value < 0.0 or not isfinite(value) for value in values):
        raise ValueError("dimensionless environmental ratios must be finite and nonnegative")
    return (
        compactness_ratio**exponents.compactness
        * (1.0 + boundary_ratio) ** exponents.boundary
        * (1.0 + thermal_ratio) ** exponents.thermal
    )


def transition_strength(
    acceleration_to_screen_ratio: float,
    compactness_ratio: float,
    boundary_ratio: float,
    thermal_ratio: float,
    exponents: TransitionExponents = TransitionExponents(),
) -> float:
    """Return T = S_high * S_env in the closed interval [0, 1]."""
    if acceleration_to_screen_ratio < 0.0 or not isfinite(acceleration_to_screen_ratio):
        raise ValueError("acceleration ratio must be finite and nonnegative")
    zeta = environmental_state(
        compactness_ratio,
        boundary_ratio,
        thermal_ratio,
        exponents,
    )
    high_screen = 1.0 / (
        1.0 + acceleration_to_screen_ratio**exponents.high_acceleration
    )
    zeta_power = zeta**exponents.environment
    environment_activation = zeta_power / (1.0 + zeta_power)
    return high_screen * environment_activation


def permittivity(alpha: float, transition: float, response: float) -> float:
    """Positive weak-field gradient coefficient epsilon."""
    if not 0.0 <= transition <= 1.0:
        raise ValueError("transition must lie in [0, 1]")
    if not all(isfinite(value) for value in (alpha, response)):
        raise ValueError("alpha and response must be finite")
    return exp(-2.0 * alpha * transition * response)


def extra_lensing_fraction(branch: str, transition: float, power: float = 1.0) -> float:
    """Return the fraction of the extra dynamical potential seen by light."""
    if not 0.0 <= transition <= 1.0:
        raise ValueError("transition must lie in [0, 1]")
    if power <= 0.0 or not isfinite(power):
        raise ValueError("power must be finite and positive")
    if branch == "scalar_conformal_control":
        return 0.0
    if branch == "transition_linked_mixed_mode":
        return transition**power
    if branch == "metric_tensor_control":
        return 1.0
    raise ValueError(f"unknown lensing branch: {branch}")


def run_checks() -> dict[str, float]:
    """Exercise the declared high-acceleration and environmental limits."""
    active_cluster_like = transition_strength(0.01, 10.0, 2.0, 10.0)
    low_environment_disk_like = transition_strength(0.01, 0.01, 0.1, 0.01)
    high_acceleration_local = transition_strength(1.0e6, 10.0, 2.0, 10.0)

    assert 0.95 < active_cluster_like < 1.0
    assert 0.0 <= low_environment_disk_like < 0.001
    assert 0.0 <= high_acceleration_local < 1.0e-10

    epsilon = permittivity(alpha=0.5, transition=active_cluster_like, response=1.0)
    assert epsilon > 0.0 and isfinite(epsilon)

    assert extra_lensing_fraction("scalar_conformal_control", active_cluster_like) == 0.0
    assert (
        extra_lensing_fraction("transition_linked_mixed_mode", active_cluster_like)
        == active_cluster_like
    )
    assert extra_lensing_fraction("metric_tensor_control", active_cluster_like) == 1.0

    return {
        "active_cluster_like": active_cluster_like,
        "low_environment_disk_like": low_environment_disk_like,
        "high_acceleration_local": high_acceleration_local,
        "positive_permittivity": epsilon,
    }


if __name__ == "__main__":
    for name, value in run_checks().items():
        print(f"{name}={value:.12g}")

