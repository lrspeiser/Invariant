"""Potential-depth channel exponent for a QUMOND-style source boost."""

from __future__ import annotations

import numpy as np


def rar_qumond_boost(gbar_m_s2, a0_m_s2: float) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if not np.isfinite(a0_m_s2) or a0_m_s2 <= 0.0:
        raise ValueError("a0_m_s2 must be finite and positive")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar_m_s2 must be finite and positive")
    denominator = -np.expm1(-np.sqrt(gbar / float(a0_m_s2)))
    return 1.0 / denominator


def potential_channel_exponent(
    potential_depth,
    *,
    transition_depth: float,
    transition_power: float,
    endpoint_exponent: float,
) -> np.ndarray:
    depth = np.asarray(potential_depth, dtype=float)
    if np.any(~np.isfinite(depth)) or np.any(depth < 0.0):
        raise ValueError("potential_depth must be finite and non-negative")
    if not np.isfinite(transition_depth) or transition_depth <= 0.0:
        raise ValueError("transition_depth must be finite and positive")
    if not np.isfinite(transition_power) or transition_power <= 0.0:
        raise ValueError("transition_power must be finite and positive")
    if not np.isfinite(endpoint_exponent) or endpoint_exponent < 1.0:
        raise ValueError("endpoint_exponent must be finite and at least one")
    coordinate = np.power(depth / float(transition_depth), float(transition_power))
    activation = coordinate / (1.0 + coordinate)
    return 1.0 + (float(endpoint_exponent) - 1.0) * activation


def potential_channel_acceleration(
    gbar_m_s2,
    potential_depth,
    *,
    a0_m_s2: float,
    transition_depth: float,
    transition_power: float,
    endpoint_exponent: float,
) -> dict[str, np.ndarray]:
    gbar, depth = np.broadcast_arrays(
        np.asarray(gbar_m_s2, dtype=float),
        np.asarray(potential_depth, dtype=float),
    )
    boost = rar_qumond_boost(gbar, a0_m_s2)
    exponent = potential_channel_exponent(
        depth,
        transition_depth=transition_depth,
        transition_power=transition_power,
        endpoint_exponent=endpoint_exponent,
    )
    enhancement = np.power(boost, exponent)
    predicted = gbar * enhancement
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        raise ValueError("predicted acceleration must be finite and positive")
    return {
        "base_qumond_boost": boost,
        "channel_exponent": exponent,
        "enhancement": enhancement,
        "predicted_acceleration_m_s2": predicted,
    }


def path_diluted_channel_exponent(
    potential_depth,
    potential_path_ratio,
    *,
    transition_depth: float,
    transition_power: float,
    extra_spatial_channels: float,
    path_power: float,
) -> dict[str, np.ndarray]:
    depth, path_ratio = np.broadcast_arrays(
        np.asarray(potential_depth, dtype=float),
        np.asarray(potential_path_ratio, dtype=float),
    )
    if np.any(~np.isfinite(path_ratio)) or np.any(path_ratio <= 0.0):
        raise ValueError("potential_path_ratio must be finite and positive")
    if not np.isfinite(extra_spatial_channels) or extra_spatial_channels < 0.0:
        raise ValueError("extra_spatial_channels must be finite and non-negative")
    if not np.isfinite(path_power) or path_power < 0.0:
        raise ValueError("path_power must be finite and non-negative")
    onset_exponent = potential_channel_exponent(
        depth,
        transition_depth=transition_depth,
        transition_power=transition_power,
        endpoint_exponent=2.0,
    )
    onset = onset_exponent - 1.0
    clipped_path = np.maximum(path_ratio, 1.0)
    survival = np.power(clipped_path, -float(path_power))
    exponent = 1.0 + float(extra_spatial_channels) * onset * survival
    return {
        "potential_onset": onset,
        "clipped_potential_path_ratio": clipped_path,
        "path_survival": survival,
        "channel_exponent": exponent,
    }


def path_diluted_potential_channel_acceleration(
    gbar_m_s2,
    potential_depth,
    potential_path_ratio,
    *,
    a0_m_s2: float,
    transition_depth: float,
    transition_power: float,
    extra_spatial_channels: float,
    path_power: float,
) -> dict[str, np.ndarray]:
    gbar, depth, path_ratio = np.broadcast_arrays(
        np.asarray(gbar_m_s2, dtype=float),
        np.asarray(potential_depth, dtype=float),
        np.asarray(potential_path_ratio, dtype=float),
    )
    boost = rar_qumond_boost(gbar, a0_m_s2)
    geometry = path_diluted_channel_exponent(
        depth,
        path_ratio,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=extra_spatial_channels,
        path_power=path_power,
    )
    enhancement = np.power(boost, geometry["channel_exponent"])
    predicted = gbar * enhancement
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        raise ValueError("predicted acceleration must be finite and positive")
    return {
        "base_qumond_boost": boost,
        **geometry,
        "enhancement": enhancement,
        "predicted_acceleration_m_s2": predicted,
    }


def system_potential_path_coordinate(
    potential_depth,
    radius_m,
    gbar_m_s2,
    *,
    light_speed_m_s: float = 299792458.0,
) -> float:
    """Return ``max(|Phi|) / max(r gbar)`` for one baryonic system."""

    depth, radius, gbar = np.broadcast_arrays(
        np.asarray(potential_depth, dtype=float),
        np.asarray(radius_m, dtype=float),
        np.asarray(gbar_m_s2, dtype=float),
    )
    if (
        np.any(~np.isfinite(depth))
        or np.any(depth < 0.0)
        or np.any(~np.isfinite(radius))
        or np.any(radius <= 0.0)
        or np.any(~np.isfinite(gbar))
        or np.any(gbar <= 0.0)
        or not np.isfinite(light_speed_m_s)
        or light_speed_m_s <= 0.0
    ):
        raise ValueError("system path-coordinate inputs must be finite and positive")
    numerator = float(np.max(depth) * float(light_speed_m_s) ** 2)
    denominator = float(np.max(radius * gbar))
    coordinate = numerator / denominator
    if not np.isfinite(coordinate) or coordinate <= 0.0:
        raise ValueError("system path coordinate must be finite and positive")
    return coordinate


def inward_monotone_majorant(values) -> np.ndarray:
    """Smallest pointwise majorant that is nonincreasing along a 1D profile."""

    profile = np.asarray(values, dtype=float)
    if profile.ndim != 1 or profile.size == 0 or np.any(~np.isfinite(profile)):
        raise ValueError("values must be a nonempty finite one-dimensional profile")
    return np.maximum.accumulate(profile[::-1])[::-1]
