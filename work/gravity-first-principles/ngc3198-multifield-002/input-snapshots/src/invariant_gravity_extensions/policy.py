"""Separate compatibility, discrimination and discovery claims.

Nothing in this module authorizes access to protected observations. Inputs are
already-produced development predictions, with whole-object resampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompatibilityPolicy:
    """Non-inferiority margin is an explicit engineering choice, not a measurement."""

    margin_dex: float = 0.015
    confidence: float = 0.95
    resamples: int = 2000
    seed: int = 1729

    def __post_init__(self) -> None:
        if not np.isfinite(self.margin_dex) or self.margin_dex < 0:
            raise ValueError("margin_dex must be finite and nonnegative")
        if not 0.5 < self.confidence < 1:
            raise ValueError("confidence must be between 0.5 and 1")
        if (type(self.resamples) is not int or type(self.seed) is not int or
                self.resamples < 100 or self.seed < 0):
            raise ValueError("use at least 100 resamples and a nonnegative seed")


DEFAULT_COMPATIBILITY_POLICY = CompatibilityPolicy()


def assess_compatibility(
    observed_log10: np.ndarray,
    baseline_log10: np.ndarray,
    candidate_log10: np.ndarray,
    object_ids: list[str],
    policy: CompatibilityPolicy = DEFAULT_COMPATIBILITY_POLICY,
    *,
    role: str = "development",
) -> dict[str, Any]:
    """Paired object-bootstrap upper bound on candidate-minus-baseline RMS.

    Arrays are log10 observables in identical units, with frozen predictions.
    All radial points of an object stay together and each object has equal
    weight. The confidence bound is a percentile-bootstrap approximation,
    not a universal significance calibration or a model-selection correction.
    """
    if role not in {"synthetic", "development"}:
        raise ValueError("this successor does not authorize validation/confirmation access")
    arrays = [np.asarray(v, dtype=float) for v in
              (observed_log10, baseline_log10, candidate_log10)]
    y, b, c = arrays
    if y.ndim != 1 or y.size == 0 or any(v.shape != y.shape for v in arrays):
        raise ValueError("predictions must be matching nonempty vectors")
    if any(not np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("nonfinite predictions are not compatible")
    if len(object_ids) != y.size or any(not isinstance(i, str) or not i for i in object_ids):
        raise ValueError("every point requires a stable nonempty object identifier")
    ids = np.asarray(object_ids)
    names = sorted(set(object_ids))
    if len(names) < 3:
        raise ValueError("at least three independent objects are required")
    bm = np.array([np.mean((y[ids == i] - b[ids == i]) ** 2) for i in names])
    cm = np.array([np.mean((y[ids == i] - c[ids == i]) ** 2) for i in names])
    rng = np.random.default_rng(policy.seed)
    # One draw at a time bounds memory even on large source catalogues.
    draws = np.empty(policy.resamples)
    for j in range(policy.resamples):
        k = rng.integers(0, len(names), len(names))
        draws[j] = np.sqrt(cm[k].mean()) - np.sqrt(bm[k].mean())
    low, high = np.quantile(draws, [1 - policy.confidence, policy.confidence])
    status = ("COMPATIBLE" if high <= policy.margin_dex else
              "INCOMPATIBLE" if low > policy.margin_dex else "INDETERMINATE")
    return {
        "status": status,
        "role": role,
        "objects": len(names),
        "baseline_rms_dex": float(np.sqrt(bm.mean())),
        "candidate_rms_dex": float(np.sqrt(cm.mean())),
        "delta_rms_dex": float(np.sqrt(cm.mean()) - np.sqrt(bm.mean())),
        "one_sided_lower_dex": float(low),
        "one_sided_upper_dex": float(high),
        "confidence": policy.confidence,
        "margin_dex": policy.margin_dex,
        "baseline_improvement_required": False,
        "discovery_claim_allowed": False,
        "confirmation_access_authorized": False,
    }


def next_stage(compatibility: str, distinguishable: bool, supported: bool = True) -> str:
    """Equal-to-baseline candidates are retained, not automatically discoveries."""
    if compatibility not in {"COMPATIBLE", "INCOMPATIBLE", "INDETERMINATE"}:
        raise ValueError("unknown compatibility status")
    if compatibility == "INCOMPATIBLE":
        return "RETAIN_FAILED_COMPATIBILITY"
    if compatibility == "INDETERMINATE":
        return "NEEDS_COMPATIBILITY_INFORMATION"
    if not supported:
        return "UNSUPPORTED_BY_CURRENT_SCORER"
    if not distinguishable:
        return "RETAIN_CURRENT_OBSERVABLE_EQUIVALENCE"
    return "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_DEVELOPMENT_TEST"


def rank_experiments(
    predictions: dict[str, np.ndarray],
    covariance: np.ndarray,
    nuisance_jacobian: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Gaussian design utility on synthetic predictions, not a p-value.

    Each value has shape (n_models, n_observables). Differences are whitened
    with the FULL covariance, then projected off the whitened nuisance span.
    Utility is the mean log(1+d^2) over model pairs. Also report the least
    distinguishable pair so an average cannot hide an equivalence class.
    """
    cov = np.asarray(covariance, dtype=float)
    if (cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.size == 0 or
            not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T)):
        raise ValueError("covariance must be finite, symmetric and square")
    chol = np.linalg.cholesky(cov)  # Singular covariance is never silently regularized.
    basis = np.empty((len(cov), 0))
    if nuisance_jacobian is not None:
        j = np.asarray(nuisance_jacobian, dtype=float)
        if j.ndim != 2 or j.shape[0] != len(cov) or not np.all(np.isfinite(j)):
            raise ValueError("invalid nuisance Jacobian")
        if j.shape[1]:
            u, s, _ = np.linalg.svd(np.linalg.solve(chol, j), full_matrices=False)
            cutoff = np.finfo(float).eps * max(j.shape) * (s[0] if s.size else 0)
            basis = u[:, s > cutoff]
    rows = []
    for name, value in sorted(predictions.items()):
        p = np.asarray(value, dtype=float)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != len(cov):
            raise ValueError("predictions need at least two models and matching observables")
        if not np.all(np.isfinite(p)):
            raise ValueError("nonfinite predictions")
        w = np.linalg.solve(chol, p.T).T
        w -= (w @ basis) @ basis.T
        d2 = [float(np.sum((w[i] - w[k]) ** 2))
              for i in range(len(w)) for k in range(i)]
        rows.append({
            "experiment": name,
            "utility": float(np.mean(np.log1p(d2))),
            "minimum_pair_distance_squared": min(d2),
            "pair_distances_squared": d2,
            "nuisance_rank": basis.shape[1],
            "scope": "synthetic_design_only_not_detection_significance",
        })
    return sorted(rows, key=lambda row: (-row["utility"], row["experiment"]))
