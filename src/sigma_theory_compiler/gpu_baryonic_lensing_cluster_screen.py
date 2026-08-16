"""GPU lensing (P1) and cluster (P2) gates over the baryonic acceleration-law family.

The billion-scale baryonic screen (`gpu_baryonic_interpolation_screen`) asks whether a
universal law ``g_obs = nu(y) * g_bar`` (``y = g_bar / a0``, one shared constant, zero
per-object freedom) produces flat rotation curves and Tully-Fisher from baryons alone.
Roadmap items P1 and P2 ask the two questions that historically kill such laws next:

* **P1 (lensing):** does the same frozen law reproduce the galaxy-galaxy lensing fact —
  a flat deflection profile consistent with the *same candidate's own* rotation curve —
  with zero per-object mass?  A nonrelativistic law needs a declared lensing
  prescription; the one frozen here (an assumption, recorded as such) is that photons
  respond to the same ``g_obs`` field with the standard weak-field GR factor 2, in Born
  approximation along straight-line paths (c = 1 units).
* **P2 (clusters):** does the law carry a hydrostatic gas cluster with no invisible
  mass?  This is where MOND-like laws historically fail; the gate exists to measure
  that per family, and the sealed negative "no candidate in this grammar passes
  clusters" is itself a deliverable, with exact margins.

Candidates are the *same ordinal family* as the screen — the codec, batch decoder, and
``nu`` evaluator are imported, not forked — so every ordinal means the same formula in
both receipts.

**P1 controls.**  The three screen disk masses (1/250, 8/125, 128/125) are recast as
spherical-equivalent Hernquist baryon profiles ``M(<r) = M r^2 / (r + a_h)^2`` with
``a_h = 1``, giving ``g_bar(r) = M / (r + 1)^2`` exactly.  The deflection

    alpha(b) = 2 * Integral_{-L}^{+L} g_obs(r(l)) * (b / r(l)) dl,   r = sqrt(b^2+l^2)

is evaluated with L = 200 by composite Simpson on the substitution ``l = b sinh t``
(uniform t in [0, asinh(L/b)], symmetric doubling for l < 0), which makes the integrand
smooth: 33 nodes reproduce 50-digit ``mpmath.quad`` of the raw formula to ~3e-8
relative, four orders under the declared 0.5% budget (verified by test).  All node
accelerations and weights are candidate-independent, computed once at 50 digits, and
frozen into the receipt as exact decimal strings; per-candidate work is one ``nu``
evaluation per node plus weighted sums, batched exactly like the screen.
Criterion, per mass, at deep impact parameters b in {8, 10, 12, 16, 20} (largest path
``y`` is 0.01264 < 0.02): alpha(b) flat to 8% (fp64; 10% fp32) of its mean, and
``|alpha(b) / (2 pi v_flat^2) - 1| <= 0.15`` (fp64; 0.18 fp32) at every b, where
``v_flat`` is measured from the same candidate's rotation curve on the screen's own
frozen disk grid.  No mass is fitted anywhere.

**P2 control.**  An isothermal beta-model gas sphere, ``beta_g = 2/3``, ``rc = 1``:
``rho ~ (1 + r^2)^(-1)``, amplitude ``4 pi rho0 = 9`` and temperature ``T0 = 9`` (so
``g_dyn(r) = 2 T0 r / (1 + r^2)`` by hydrostatic equilibrium).  Calibration facts,
fixed by 50-digit computation and enforced by known-answer controls: under pure Newton
the gas alone falls short of ``g_dyn`` by 2.40x-2.82x at the outer probes (5.50x at the
innermost; 3.58x at r = 2 rc, above the required ~2x), i.e. the real cluster
missing-mass factor with no dark matter expressible.  Probe accelerations span
``y = 0.92-2.01``: the beta = 2/3 profile's ``g_bar`` varies only 2.2x across
r in {0.5, 1, 2, 4, 8} rc, so a 0.5-20 span is geometrically unattainable for this
profile; the calibration instead pins the missing-mass factors above, which is what the
known-answer controls actually require.  Criterion:
``max_r |g_obs(g_bar(r)) / g_dyn(r) - 1| <= 0.15`` (fp64; 0.20 fp32).

**Known-answer controls (mandatory, the run fails closed if they break):**
Newton (``nu = 1``) fails P2 with recorded shortfall >= 1.5 (measured min 2.404) and
fails P1 (its deflection falls as 1/b: flatness 0.79).  The sqrt-family survivor
``nu = sqrt(1 + u^2)`` passes P1 (worst consistency deviation 0.118) and fails P2
(closest probe still 0.399 from unity, 2.7x the tolerance) — the documented MOND
cluster shortfall.  Bonus finding recorded as a third control: the galaxy-screen
survivor ``nu = 1 + u`` *fails* P1 (worst deviation 0.158): its slowly-decaying
Newtonian residual adds a non-flat deflection component inconsistent with its own
rotation curve — a family that flattens curves but fails lensing, named per roadmap.

Three-layer honesty, matching the screen: an fp32 sweep with slack thresholds, an fp64
recheck of survivors at strict thresholds, and 50-digit mpmath re-verification of every
reported candidate, plus an fp64 CPU/GPU decision cross-check on a 4096 sample (GPU
parts skip gracefully when cupy is unavailable).

Claim boundary: no observational data is opened; the controls are analytic; the lensing
prescription is an assumption, not a derivation; survivors are search priorities, not
validated theories; and a zero-survivor cluster verdict is a sealed negative result,
not a failure of the run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from .gpu_baryonic_interpolation_screen import (
    FAMILY_SIZE,
    SCREEN_CONFIG,
    _digits_from_ordinals,
    _nu_at,
    build_probe_grid,
    decode_ordinal,
    encode_candidate,
    render_candidate,
    screen_batch,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-gpu-baryonic-lensing-cluster-screen-result-1.0"

#: Hard bounds.  Exceeding any of these raises; nothing is silently truncated except the
#: explicitly recorded exact-verification cap below.
SYSTEM_CAPS = {
    "min_batch_size": 1 << 10,
    "max_batch_size": 1 << 24,
    "max_pareto_reported": 64,
    "max_exact_verifications": 192,
    "cluster_min_tracked": 16,
}

#: Frozen gate configuration.  Changing any value changes the claim and the receipt
#: hash.  The cluster probe table is frozen as 50-digit decimal strings (g_bar) and
#: exact rationals (g_dyn); a test and `validate_receipt` recompute both.
GATE_CONFIG: dict[str, Any] = {
    "a0": 1,
    "mpmath_dps": 50,
    "crosscheck_sample": 4096,
    "lensing": {
        "prescription": (
            "photons respond to the same g_obs field as dynamics with the standard "
            "weak-field GR factor 2, Born approximation, straight-line paths, c = 1; "
            "declared assumption, not a derivation"
        ),
        "profile": "hernquist-sphere",
        "hernquist_scale": 1,
        "masses": ["1/250", "8/125", "128/125"],
        "impact_parameters": [8, 10, 12, 16, 20],
        "path_half_length": 200,
        "path_nodes": 33,
        "path_substitution": (
            "l = b*sinh(t), composite Simpson on uniform t in [0, asinh(L/b)], "
            "factor 2 for the symmetric l < 0 half"
        ),
        "deep_y_ceiling": "2e-2",
        "fp32_thresholds": {"flatness": "1e-1", "consistency": "18e-2"},
        "fp64_thresholds": {"flatness": "8e-2", "consistency": "15e-2"},
    },
    "cluster": {
        "gas_profile": "isothermal-beta-model",
        "beta_g": "2/3",
        "core_radius": 1,
        "amplitude_4pi_rho0": 9,
        "temperature_T0": 9,
        "hydrostatic_gdyn": "g_dyn(r) = 3*beta_g*T0*r/(rc^2*(1+(r/rc)^2)) = 2*T0*r/(1+r^2)",
        "probe_radii": ["1/2", "1", "2", "4", "8"],
        "gbar_50dps": [
            "1.3086860759709798162867756673962815269726660456997",
            "1.931416529422965213459052387621118510556368851406",
            "2.0089153849632963682116027145982916598423927978468",
            "1.5042275641867317384041779441339824574587521617652",
            "0.92159418765260598972187966277682931642611358482407",
        ],
        "gdyn_exact": ["36/5", "9", "36/5", "72/17", "144/65"],
        "newton_shortfall_floor": "3/2",
        "fp32_thresholds": {"consistency": "2e-1"},
        "fp64_thresholds": {"consistency": "15e-2"},
    },
}

CLAIMS = {
    "cluster_negative_is_a_valid_deliverable": True,
    "corpus_absence_establishes_novelty": False,
    "invisible_mass_used_as_target_or_rescue": False,
    "lensing_prescription_is_an_assumption": True,
    "observational_data_opened": False,
    "per_object_free_parameters_expressible": False,
    "scalar_truth_or_probability_score": False,
    "sealed_validation_ladder_bypassed": False,
    "survivor_is_validated_theory": False,
    "synthetic_analytic_controls_only": True,
}

CONTROL_ORDINALS = {
    "newton_nu1": encode_candidate(2, [0] * 5, [0] * 5),
    "sqrt_family": encode_candidate(1, [0, 1, 0, 0, 0], [0] * 5),
    "linear_u": encode_candidate(2, [1, 0, 0, 0, 0], [0] * 5),
}


class LensingClusterScreenError(ValueError):
    """Raised on malformed input, a broken known-answer control, or receipt tamper."""


def _fraction(text: str) -> mp.mpf:
    if "/" in text:
        numerator, denominator = text.split("/")
        return mp.mpf(numerator) / mp.mpf(denominator)
    return mp.mpf(text)


# ---------------------------------------------------------------------------
# Frozen candidate-independent grids (50-digit mpmath, decimal strings)
# ---------------------------------------------------------------------------


def build_lensing_grid(path_nodes: int | None = None) -> dict[str, Any]:
    """Hernquist path nodes and Simpson weights for every (mass, b), as decimals.

    ``alpha(b) = sum_i weight_i * nu(y_i)`` where ``weight_i`` already contains the
    prescription factor 2, the symmetric-path factor, the Simpson coefficient, and the
    local ``g_bar * b / r`` integrand factor, so per-candidate work is one ``nu`` per
    node.  ``path_nodes`` may be overridden only to *increase* resolution for the
    convergence test; the frozen default is part of the claim.
    """

    mp.mp.dps = GATE_CONFIG["mpmath_dps"]
    config = GATE_CONFIG["lensing"]
    nodes = config["path_nodes"] if path_nodes is None else path_nodes
    if nodes < config["path_nodes"] or nodes % 2 == 0:
        raise LensingClusterScreenError("path_nodes must be odd and >= the frozen default")
    length = mp.mpf(config["path_half_length"])
    scale = mp.mpf(config["hernquist_scale"])
    ceiling = mp.mpf(config["deep_y_ceiling"])
    integrals: list[dict[str, Any]] = []
    max_path_y = mp.mpf(0)
    for mass_index, mass_text in enumerate(config["masses"]):
        mass = _fraction(mass_text)
        for impact in config["impact_parameters"]:
            b = mp.mpf(impact)
            t_max = mp.asinh(length / b)
            step = t_max / (nodes - 1)
            entries = []
            for index in range(nodes):
                t = index * step
                radius = b * mp.cosh(t)
                y = mass / (radius + scale) ** 2
                if y > ceiling:
                    raise LensingClusterScreenError(
                        f"path node leaves the deep regime: y={float(y)} at b={impact}"
                    )
                max_path_y = max(max_path_y, y)
                simpson = 1 if index in (0, nodes - 1) else (4 if index % 2 == 1 else 2)
                # 2 (prescription) * 2 (symmetric halves) * (h/3 Simpson) * g_bar * b/r * dl/dt
                # with dl = b cosh(t) dt and b/r = 1/cosh(t): the cosh factors cancel.
                weight = 4 * b * (step / 3) * simpson * y
                entries.append({"y": mp.nstr(y, 50), "weight": mp.nstr(weight, 50)})
            integrals.append(
                {
                    "mass_text": mass_text,
                    "mass_index": mass_index,
                    "impact_parameter": impact,
                    "t_max": mp.nstr(t_max, 50),
                    "nodes": entries,
                }
            )
    return {"integrals": integrals, "max_path_y": mp.nstr(max_path_y, 50)}


def recompute_cluster_table() -> dict[str, list[str]]:
    """Recompute the frozen cluster probe table at 50 digits (used by validation).

    ``M(<r)`` is integrated numerically from the beta-model density and cross-checked
    against the closed form ``(4 pi rho0) * (r - atan r)`` before freezing.
    """

    mp.mp.dps = GATE_CONFIG["mpmath_dps"]
    config = GATE_CONFIG["cluster"]
    amplitude = mp.mpf(config["amplitude_4pi_rho0"])
    t0 = mp.mpf(config["temperature_T0"])
    gbar, gdyn = [], []
    for radius_text in config["probe_radii"]:
        radius = _fraction(radius_text)
        mass = mp.quad(lambda rp: amplitude * rp**2 / (1 + rp**2), [0, radius])
        closed = amplitude * (radius - mp.atan(radius))
        if abs(mass - closed) > mp.mpf(10) ** (-40):
            raise LensingClusterScreenError("beta-model mass integral disagrees with closed form")
        gbar.append(mp.nstr(closed / radius**2, 50))
        gdyn.append(2 * t0 * radius / (1 + radius**2))
    return {"gbar_50dps": gbar, "gdyn_values": [mp.nstr(value, 50) for value in gdyn]}


def _compile_grids(lensing_grid: Mapping[str, Any]) -> dict[str, Any]:
    """Parse frozen decimal strings into float probe lists for the batched kernels."""

    by_mass: list[list[list[tuple[float, float]]]] = [
        [] for _ in GATE_CONFIG["lensing"]["masses"]
    ]
    for integral in lensing_grid["integrals"]:
        nodes = [(float(node["y"]), float(node["weight"])) for node in integral["nodes"]]
        by_mass[integral["mass_index"]].append(nodes)
    config = GATE_CONFIG["cluster"]
    cluster = [
        (float(gbar), float(_fraction(gdyn)))
        for gbar, gdyn in zip(config["gbar_50dps"], config["gdyn_exact"], strict=True)
    ]
    disk_grid = build_probe_grid()
    disks = [
        [(point["gbar"], point["radius"]) for point in disk["points"] if point["outer"]]
        for disk in disk_grid["disks"]
    ]
    return {"lensing_by_mass": by_mass, "cluster": cluster, "disks": disks, "disk_grid": disk_grid}


# ---------------------------------------------------------------------------
# Vectorized gate evaluation (shared numpy/cupy code path)
# ---------------------------------------------------------------------------


def evaluate_gate_batch(
    xp: Any, ordinals: Any, compiled: Mapping[str, Any], *, dtype: Any, tier: str
) -> dict[str, Any]:
    """Both gates for one ordinal batch.  ``tier`` picks fp32 or fp64 thresholds."""

    lens_thresholds = GATE_CONFIG["lensing"][tier]
    cluster_thresholds = GATE_CONFIG["cluster"][tier]
    beta_index, coefficients = _digits_from_ordinals(xp, ordinals)
    count = ordinals.shape[0]
    two_pi = dtype(2.0 * math.pi)

    # v_flat^2 per mass, measured exactly as the screen does on its frozen disk grid.
    lens_valid = xp.ones(count, dtype=bool)
    vflat2 = []
    for outer_points in compiled["disks"]:
        speed_sum = xp.zeros(count, dtype=dtype)
        for gbar, radius in outer_points:
            nu, valid = _nu_at(xp, beta_index, coefficients, gbar, dtype)
            lens_valid &= valid
            speed_sum = speed_sum + xp.sqrt(xp.maximum(dtype(gbar) * nu * dtype(radius), dtype(0)))
        mean = speed_sum / dtype(len(outer_points))
        vflat2.append(mean * mean)

    # Deflection profile per mass: flat across b and consistent with 2*pi*v_flat^2.
    flat_ok = xp.ones(count, dtype=bool)
    consistent = xp.ones(count, dtype=bool)
    positive = xp.ones(count, dtype=bool)
    worst_consistency = xp.zeros(count, dtype=dtype)
    for mass_index, integrals in enumerate(compiled["lensing_by_mass"]):
        expected = two_pi * vflat2[mass_index]
        positive &= expected > 0
        safe_expected = xp.where(expected > 0, expected, dtype(1))
        alpha_max = None
        alpha_min = None
        alpha_sum = xp.zeros(count, dtype=dtype)
        for nodes in integrals:
            alpha = xp.zeros(count, dtype=dtype)
            for y_value, weight in nodes:
                nu, valid = _nu_at(xp, beta_index, coefficients, y_value, dtype)
                lens_valid &= valid
                alpha = alpha + dtype(weight) * nu
            deviation = xp.abs(alpha / safe_expected - 1)
            worst_consistency = xp.maximum(worst_consistency, deviation)
            consistent &= deviation <= dtype(float(lens_thresholds["consistency"]))
            alpha_max = alpha if alpha_max is None else xp.maximum(alpha_max, alpha)
            alpha_min = alpha if alpha_min is None else xp.minimum(alpha_min, alpha)
            alpha_sum = alpha_sum + alpha
        mean_alpha = alpha_sum / dtype(len(integrals))
        safe_mean = xp.where(mean_alpha > 0, mean_alpha, dtype(1))
        spread_ok = (alpha_max - alpha_min) <= dtype(float(lens_thresholds["flatness"])) * safe_mean
        flat_ok &= (mean_alpha > 0) & spread_ok
    lensing_pass = lens_valid & positive & flat_ok & consistent

    # Cluster: worst hydrostatic ratio deviation over the frozen probe table.
    cluster_valid = xp.ones(count, dtype=bool)
    cluster_deviation = xp.zeros(count, dtype=dtype)
    for y_value, gdyn in compiled["cluster"]:
        nu, valid = _nu_at(xp, beta_index, coefficients, y_value, dtype)
        cluster_valid &= valid
        ratio = (dtype(y_value) * nu) / dtype(gdyn)
        cluster_deviation = xp.maximum(cluster_deviation, xp.abs(ratio - 1))
    within = cluster_deviation <= dtype(float(cluster_thresholds["consistency"]))
    cluster_pass = cluster_valid & within

    simplicity = (coefficients != 0).sum(axis=1).astype(xp.int32)
    simplicity = simplicity + (beta_index != 2).astype(xp.int32)
    infinity = dtype(float("inf"))
    return {
        "lensing_pass": lensing_pass,
        "cluster_pass": cluster_pass,
        "lensing_consistency": xp.where(lens_valid, worst_consistency, infinity),
        "cluster_deviation": xp.where(cluster_valid, cluster_deviation, infinity),
        "simplicity": simplicity,
    }


# ---------------------------------------------------------------------------
# Exact re-verification (mpmath, 50 digits)
# ---------------------------------------------------------------------------


def _exact_context(lensing_grid: Mapping[str, Any], disk_grid: Mapping[str, Any]) -> dict[str, Any]:
    mp.mp.dps = GATE_CONFIG["mpmath_dps"]
    lensing = []
    for integral in lensing_grid["integrals"]:
        lensing.append(
            {
                "mass_index": integral["mass_index"],
                "mass_text": integral["mass_text"],
                "impact_parameter": integral["impact_parameter"],
                "nodes": [
                    (mp.mpf(node["y"]), mp.mpf(node["weight"])) for node in integral["nodes"]
                ],
            }
        )
    config = GATE_CONFIG["cluster"]
    cluster = [
        (mp.mpf(gbar), _fraction(gdyn))
        for gbar, gdyn in zip(config["gbar_50dps"], config["gdyn_exact"], strict=True)
    ]
    disks = [
        [(mp.mpf(point["gbar"]), point["radius"]) for point in disk["points"] if point["outer"]]
        for disk in disk_grid["disks"]
    ]
    return {"lensing": lensing, "cluster": cluster, "disks": disks}


def _nu_exact(candidate: Mapping[str, Any], y: mp.mpf) -> mp.mpf | None:
    """50-digit nu(y); None when the ratio is undefined or nonpositive (fail closed)."""

    beta = {"1/3": mp.mpf(1) / 3, "1/2": mp.mpf(1) / 2, "1": mp.mpf(1), "2": mp.mpf(2)}[
        candidate["beta"]
    ]
    u = 1 / mp.sqrt(y)
    numerator = mp.mpf(1)
    denominator = mp.mpf(1)
    for power, (a_k, b_k) in enumerate(zip(candidate["a"], candidate["b"], strict=True), start=1):
        numerator += a_k * u**power
        denominator += b_k * u**power
    if denominator == 0:
        return None
    ratio = numerator / denominator
    if ratio <= 0:
        return None
    return ratio**beta


def verify_candidate_exact_gates(ordinal: int, context: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run both gates for one candidate at 50 digits, with decimal-string margins."""

    mp.mp.dps = GATE_CONFIG["mpmath_dps"]
    candidate = decode_ordinal(ordinal)
    lens_thresholds = GATE_CONFIG["lensing"]["fp64_thresholds"]
    cluster_tolerance = mp.mpf(GATE_CONFIG["cluster"]["fp64_thresholds"]["consistency"])

    lens_valid = True
    per_mass: list[dict[str, Any]] = []
    vflat_by_mass: list[mp.mpf | None] = []
    for outer_points in context["disks"]:
        speeds = []
        for gbar, radius in outer_points:
            nu = _nu_exact(candidate, gbar)
            if nu is None:
                lens_valid = False
                break
            speeds.append(mp.sqrt(gbar * nu * radius))
        vflat_by_mass.append(sum(speeds) / len(speeds) if lens_valid else None)
        if not lens_valid:
            break

    worst_flatness: mp.mpf | None = None
    worst_consistency: mp.mpf | None = None
    if lens_valid:
        worst_flatness = mp.mpf(0)
        worst_consistency = mp.mpf(0)
        alphas_by_mass: dict[int, list[mp.mpf]] = {}
        for integral in context["lensing"]:
            total = mp.mpf(0)
            for y, weight in integral["nodes"]:
                nu = _nu_exact(candidate, y)
                if nu is None:
                    lens_valid = False
                    break
                total += weight * nu
            if not lens_valid:
                break
            alphas_by_mass.setdefault(integral["mass_index"], []).append(total)
        if lens_valid:
            for mass_index, alphas in sorted(alphas_by_mass.items()):
                vflat = vflat_by_mass[mass_index]
                expected = 2 * mp.pi * vflat**2
                if expected <= 0:
                    lens_valid = False
                    break
                mean = sum(alphas) / len(alphas)
                if mean <= 0:
                    lens_valid = False
                    break
                spread = (max(alphas) - min(alphas)) / mean
                worst_flatness = max(worst_flatness, spread)
                for alpha in alphas:
                    worst_consistency = max(worst_consistency, abs(alpha / expected - 1))
                per_mass.append(
                    {
                        "mass_text": GATE_CONFIG["lensing"]["masses"][mass_index],
                        "v_flat": mp.nstr(vflat, 30),
                        "flatness": format(float(spread), ".9e"),
                    }
                )
    lensing_passes = (
        lens_valid
        and worst_flatness is not None
        and worst_flatness <= mp.mpf(lens_thresholds["flatness"])
        and worst_consistency <= mp.mpf(lens_thresholds["consistency"])
    )

    cluster_valid = True
    ratios: list[mp.mpf] = []
    shortfalls: list[mp.mpf] = []
    for gbar, gdyn in context["cluster"]:
        nu = _nu_exact(candidate, gbar)
        if nu is None:
            cluster_valid = False
            break
        gobs = gbar * nu
        ratios.append(gobs / gdyn)
        shortfalls.append(gdyn / gobs)
    deviations = [abs(ratio - 1) for ratio in ratios] if cluster_valid else []
    cluster_passes = cluster_valid and max(deviations) <= cluster_tolerance

    def _text(value: mp.mpf | None) -> str | None:
        return None if value is None else format(float(value), ".9e")

    return {
        "ordinal": ordinal,
        "candidate": candidate,
        "formula": render_candidate(candidate),
        "lensing": {
            "valid": lens_valid,
            "passes": bool(lensing_passes),
            "worst_flatness": _text(worst_flatness if lens_valid else None),
            "worst_consistency": _text(worst_consistency if lens_valid else None),
            "per_mass": per_mass if lens_valid else [],
        },
        "cluster": {
            "valid": cluster_valid,
            "passes": bool(cluster_passes),
            "max_deviation": _text(max(deviations) if deviations else None),
            "closest_probe_deviation": _text(min(deviations) if deviations else None),
            "ratio_by_probe": [format(float(ratio), ".9e") for ratio in ratios],
            "shortfall_by_probe": [format(float(value), ".9e") for value in shortfalls],
            "shortfall_min": _text(min(shortfalls) if shortfalls else None),
        },
        "both_pass": bool(lensing_passes and cluster_passes),
    }


def _assert_known_answer_controls(controls: Mapping[str, Mapping[str, Any]]) -> None:
    """The calibration is part of the claim: a broken control aborts the run."""

    newton = controls["newton_nu1"]
    shortfall_floor = float(_fraction(GATE_CONFIG["cluster"]["newton_shortfall_floor"]))
    if newton["cluster"]["passes"] or newton["lensing"]["passes"]:
        raise LensingClusterScreenError("Newton control unexpectedly passed a gate")
    if float(newton["cluster"]["shortfall_min"]) < shortfall_floor:
        raise LensingClusterScreenError("cluster calibration too weak: Newton shortfall < 3/2")
    sqrt_family = controls["sqrt_family"]
    if not sqrt_family["lensing"]["passes"]:
        raise LensingClusterScreenError("sqrt-family control failed the lensing gate")
    if sqrt_family["cluster"]["passes"]:
        raise LensingClusterScreenError(
            "cluster calibration too weak: the MOND-like control passed the cluster gate"
        )


# ---------------------------------------------------------------------------
# Campaign driver
# ---------------------------------------------------------------------------


def _array_module(use_gpu: bool) -> tuple[Any, str, bool]:
    if use_gpu:
        try:
            import cupy as xp

            xp.arange(4).sum()
            name = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            return xp, name, True
        except Exception:  # noqa: BLE001 - any CUDA absence degrades to CPU gracefully
            return np, "cpu-numpy (cupy unavailable)", False
    return np, "cpu-numpy", False


def run_screen(
    *,
    limit: int | None = None,
    batch_size: int = 1 << 22,
    use_gpu: bool = True,
    pareto_cap: int = 64,
) -> dict[str, Any]:
    """Run both gates over the family (or its first ``limit`` ordinals), seal a receipt."""

    if not SYSTEM_CAPS["min_batch_size"] <= batch_size <= SYSTEM_CAPS["max_batch_size"]:
        raise LensingClusterScreenError(f"batch_size outside system caps: {batch_size}")
    if not 1 <= pareto_cap <= SYSTEM_CAPS["max_pareto_reported"]:
        raise LensingClusterScreenError(f"pareto_cap outside system caps: {pareto_cap}")
    if limit is not None and limit < 1:
        raise LensingClusterScreenError(f"limit must be positive: {limit}")
    if GATE_CONFIG["lensing"]["masses"] != SCREEN_CONFIG["disk_masses"]:
        raise LensingClusterScreenError("lensing masses drifted from the screen disk masses")

    xp, device, gpu = _array_module(use_gpu)
    lensing_grid = build_lensing_grid()
    compiled = _compile_grids(lensing_grid)
    exact_context = _exact_context(lensing_grid, compiled["disk_grid"])
    total = FAMILY_SIZE if limit is None else min(limit, FAMILY_SIZE)

    started = time.perf_counter()
    union_parts: list[np.ndarray] = []
    tracked: list[tuple[float, int]] = []
    processed = 0
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        ordinals = xp.arange(start, stop, dtype=xp.int64)
        sweep = evaluate_gate_batch(
            xp, ordinals, compiled, dtype=xp.float32, tier="fp32_thresholds"
        )
        union_mask = sweep["lensing_pass"] | sweep["cluster_pass"]
        chosen = ordinals[union_mask]
        if chosen.shape[0]:
            union_parts.append(chosen.get() if gpu else np.asarray(chosen))
        deviation = sweep["cluster_deviation"]
        keep = min(SYSTEM_CAPS["cluster_min_tracked"], int(deviation.shape[0]))
        if keep:
            index = xp.argpartition(deviation, keep - 1)[:keep]
            best_dev = deviation[index]
            best_ord = ordinals[index]
            if gpu:
                best_dev, best_ord = best_dev.get(), best_ord.get()
            tracked.extend(
                (float(d), int(o))
                for d, o in zip(np.asarray(best_dev), np.asarray(best_ord), strict=True)
                if math.isfinite(float(d))
            )
        processed = stop
    elapsed = time.perf_counter() - started

    union = (
        np.unique(np.concatenate(union_parts)) if union_parts else np.empty(0, dtype=np.int64)
    )

    # fp64 strict recheck of every fp32 survivor, keeping the per-gate decisions.
    rows: list[dict[str, np.ndarray]] = []
    for start in range(0, union.size, batch_size):
        chunk = xp.asarray(union[start : start + batch_size])
        recheck = evaluate_gate_batch(
            xp, chunk, compiled, dtype=xp.float64, tier="fp64_thresholds"
        )
        rows.append(
            {key: (value.get() if gpu else np.asarray(value)) for key, value in recheck.items()}
        )

    def _collect(key: str, dtype: Any) -> np.ndarray:
        if not rows:
            return np.empty(0, dtype=dtype)
        return np.concatenate([row[key] for row in rows]).astype(dtype)

    lens_ok = _collect("lensing_pass", bool)
    cluster_ok = _collect("cluster_pass", bool)
    lens_metric = _collect("lensing_consistency", np.float64)
    cluster_metric = _collect("cluster_deviation", np.float64)
    simplicity = _collect("simplicity", np.int32)
    lensing_pass = int(lens_ok.sum())
    cluster_pass = int(cluster_ok.sum())
    both_ordinals = union[lens_ok & cluster_ok]

    # Pareto front over fp64 survivors of either gate: (simplicity, lensing, cluster).
    pool = np.flatnonzero(lens_ok | cluster_ok)
    if pool.size:
        prefilter = min(4096, pool.size)
        axes_source = (simplicity[pool], lens_metric[pool], cluster_metric[pool])
        chosen = np.unique(
            np.concatenate(
                [
                    np.lexsort((axes_source[1], axes_source[2], axes_source[0]))[:prefilter],
                    np.lexsort((axes_source[0], axes_source[2], axes_source[1]))[:prefilter],
                    np.lexsort((axes_source[0], axes_source[1], axes_source[2]))[:prefilter],
                ]
            )
        )
        axes = np.stack(
            [
                axes_source[0][chosen].astype(np.float64),
                axes_source[1][chosen],
                axes_source[2][chosen],
            ],
            axis=1,
        )
        not_worse = (axes[:, None, :] <= axes[None, :, :]).all(axis=2)
        strictly_better = (axes[:, None, :] < axes[None, :, :]).any(axis=2)
        dominated = (not_worse & strictly_better).any(axis=0)
        front_local = chosen[~dominated]
        order = np.lexsort(
            (axes_source[2][front_local], axes_source[1][front_local], axes_source[0][front_local])
        )
        front_rows = pool[front_local[order][:pareto_cap]]
    else:
        front_rows = np.empty(0, dtype=np.int64)
    front_ordinals = union[front_rows]

    def _union_row(ordinal: int) -> int:
        """Row of an ordinal in the sorted union; fail closed on a phantom entry."""

        row = int(np.searchsorted(union, ordinal))
        if row >= union.size or int(union[row]) != ordinal:
            raise LensingClusterScreenError(f"ordinal {ordinal} missing from the fp32 union")
        return row

    # Sealed-negative margin: refine the fp32-tracked closest cluster approaches at fp64,
    # then verify the single best at 50 digits.
    closest_cluster: dict[str, Any] | None = None
    tracked_ordinals = sorted({ordinal for _, ordinal in tracked})
    if tracked_ordinals:
        refine = evaluate_gate_batch(
            np,
            np.asarray(tracked_ordinals, dtype=np.int64),
            compiled,
            dtype=np.float64,
            tier="fp64_thresholds",
        )
        deviations = np.asarray(refine["cluster_deviation"], dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(deviations))
        if finite.size:
            best_row = finite[np.lexsort((np.asarray(tracked_ordinals)[finite],
                                          deviations[finite]))[0]]
            best_ordinal = int(tracked_ordinals[best_row])
            verdict = verify_candidate_exact_gates(best_ordinal, exact_context)
            closest_cluster = {
                "ordinal": best_ordinal,
                "formula": verdict["formula"],
                "max_deviation": verdict["cluster"]["max_deviation"],
                "lensing_passes": verdict["lensing"]["passes"],
                "located_by": "fp32 sweep argmin, fp64 refined, sealed at 50 digits",
            }

    # Known-answer controls, always at 50 digits; a broken control aborts the run.
    controls = {
        name: verify_candidate_exact_gates(ordinal, exact_context)
        for name, ordinal in CONTROL_ORDINALS.items()
    }
    _assert_known_answer_controls(controls)

    # Exact verification of every reported candidate (front plus any both-gate
    # survivors beyond it), with an explicitly recorded cap.
    verify_ordinals: list[int] = [int(v) for v in front_ordinals]
    front_set = set(verify_ordinals)
    for value in both_ordinals:
        if int(value) not in front_set:
            verify_ordinals.append(int(value))
    budget = SYSTEM_CAPS["max_exact_verifications"]
    exact_truncated = max(0, len(verify_ordinals) - budget)
    verify_ordinals = verify_ordinals[:budget]
    galaxy_pass_lookup: dict[int, bool] = {}
    if verify_ordinals:
        galaxy = screen_batch(
            np,
            np.asarray(verify_ordinals, dtype=np.int64),
            compiled["disk_grid"],
            dtype=np.float64,
            thresholds=SCREEN_CONFIG["fp64_thresholds"],
        )
        galaxy_pass_lookup = {
            ordinal: bool(flag) for ordinal, flag in zip(verify_ordinals, galaxy, strict=True)
        }
    verified: list[dict[str, Any]] = []
    for ordinal in verify_ordinals:
        verdict = verify_candidate_exact_gates(ordinal, exact_context)
        row = _union_row(ordinal)
        fp64_decisions = {
            "lensing_pass": bool(lens_ok[row]),
            "cluster_pass": bool(cluster_ok[row]),
        }
        verdict["fp64"] = fp64_decisions
        verdict["role"] = "pareto_front" if ordinal in front_set else "both_pass"
        verdict["galaxy_screen_pass"] = galaxy_pass_lookup[ordinal]
        verdict["exact_confirmed"] = (
            verdict["lensing"]["passes"] == fp64_decisions["lensing_pass"]
            and verdict["cluster"]["passes"] == fp64_decisions["cluster_pass"]
        )
        verified.append(verdict)
    confirmed = sum(1 for item in verified if item["exact_confirmed"])

    front_entries = []
    for row, ordinal in zip(front_rows, map(int, front_ordinals), strict=True):
        front_entries.append(
            {
                "ordinal": ordinal,
                "simplicity": int(simplicity[row]),
                "lensing_consistency": format(float(lens_metric[row]), ".9e"),
                "cluster_deviation": format(float(cluster_metric[row]), ".9e"),
                "lensing_pass": bool(lens_ok[row]),
                "cluster_pass": bool(cluster_ok[row]),
                "formula": render_candidate(decode_ordinal(ordinal)),
                "galaxy_screen_pass": galaxy_pass_lookup[ordinal],
                "exact_confirmed": next(
                    item["exact_confirmed"] for item in verified if item["ordinal"] == ordinal
                ),
            }
        )

    # CPU/GPU decision cross-check on a deterministic sample (GPU runs only).
    crosscheck: dict[str, Any] = {"performed": False}
    if gpu:
        rng = np.random.default_rng(20260814)
        sample = np.sort(
            rng.choice(total, size=min(GATE_CONFIG["crosscheck_sample"], total), replace=False)
        ).astype(np.int64)
        import cupy as cp

        gpu_result = evaluate_gate_batch(
            cp, cp.asarray(sample), compiled, dtype=cp.float64, tier="fp64_thresholds"
        )
        cpu_result = evaluate_gate_batch(
            np, sample, compiled, dtype=np.float64, tier="fp64_thresholds"
        )
        crosscheck = {
            "performed": True,
            "sample": int(sample.shape[0]),
            "lensing_disagreements": int(
                (gpu_result["lensing_pass"].get() != cpu_result["lensing_pass"]).sum()
            ),
            "cluster_disagreements": int(
                (gpu_result["cluster_pass"].get() != cpu_result["cluster_pass"]).sum()
            ),
        }

    both_pass = int(both_ordinals.size)
    if both_pass == 0:
        margin = closest_cluster["max_deviation"] if closest_cluster else None
        decision = (
            "SCREENED-SEALED-NEGATIVE: no processed candidate passes the lensing and "
            f"cluster gates jointly; closest cluster approach deviation {margin} against "
            f"tolerance {GATE_CONFIG['cluster']['fp64_thresholds']['consistency']}"
        )
        cluster_negative = {
            "sealed": True,
            "statement": (
                "within the processed ordinal range, no universal nu(y) in this grammar "
                "carries the hydrostatic cluster control without invisible mass"
            ),
            "closest_cluster_approach": closest_cluster,
            "lensing_pass_count": lensing_pass,
        }
    else:
        decision = f"SCREENED: {both_pass} candidates pass both gates"
        cluster_negative = None

    body: dict[str, Any] = {
        "assumptions": {
            "lensing_prescription": GATE_CONFIG["lensing"]["prescription"],
            "cluster_control": (
                "isothermal beta-model gas sphere in hydrostatic equilibrium; the true "
                "dynamical field g_dyn is fixed by the declared temperature, not by any "
                "candidate; missing-mass factors 2.40-5.50 across probes under Newton"
            ),
            "probe_span_note": (
                "beta = 2/3 gas g_bar varies only 2.2x over the probe radii, so probe "
                "y sits at 0.92-2.01 (transition regime); calibration pins the Newton "
                "missing-mass factors (>= 2 at 2 rc) rather than a wider y span, which "
                "this profile cannot express"
            ),
        },
        "claims": CLAIMS,
        "config": GATE_CONFIG,
        "config_sha256": canonical_sha256(GATE_CONFIG),
        "screen_config_sha256": canonical_sha256(SCREEN_CONFIG),
        "controls": controls,
        "counts": {
            "family_size": FAMILY_SIZE,
            "processed": processed,
            "fp32_union_survivors": int(union.size),
            "lensing_pass": lensing_pass,
            "cluster_pass": cluster_pass,
            "both_pass": both_pass,
            "pareto_front": len(front_entries),
            "exact_verified": len(verified),
            "exact_confirmed": confirmed,
            "exact_refuted": len(verified) - confirmed,
            "exact_verification_truncated": exact_truncated,
        },
        "crosscheck": crosscheck,
        "decision": decision,
        "cluster_negative": cluster_negative,
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "frozen_grids": {"lensing": lensing_grid, "cluster": recompute_cluster_table()},
        "pareto_front": front_entries,
        "exact_verification": verified,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "GPU screen of the ordinal-indexed universal baryonic acceleration-law "
            "family against two synthetic gates: P1, deflection flatness and "
            "dynamics-lensing consistency on Hernquist spherical equivalents of the "
            "screen's disk masses under a declared factor-2 lensing prescription; and "
            "P2, a hydrostatic isothermal beta-model cluster whose dynamical field the "
            "gas alone cannot supply under Newton. One universal constant, zero "
            "per-object freedom; no observational data opened; survivors are search "
            "priorities and a zero-survivor cluster verdict is a sealed negative "
            "deliverable, with margins. The sealed covariant/formal/observational "
            "ladder is untouched."
        ),
        "throughput_candidates_per_second": int(processed / elapsed) if elapsed > 0 else None,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, binding, frozen-grid, and control replay checks; fail closed on any drift."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise LensingClusterScreenError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise LensingClusterScreenError("receipt seal changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise LensingClusterScreenError("config binding changed")
    if value.get("config_sha256") != canonical_sha256(GATE_CONFIG):
        raise LensingClusterScreenError("receipt config does not match this module")
    if value.get("screen_config_sha256") != canonical_sha256(SCREEN_CONFIG):
        raise LensingClusterScreenError("screen config binding changed")
    if value.get("claims") != CLAIMS:
        raise LensingClusterScreenError("claims block changed")
    frozen = value.get("frozen_grids", {})
    if canonical_sha256(frozen.get("lensing")) != canonical_sha256(build_lensing_grid()):
        raise LensingClusterScreenError("frozen lensing grid does not replay")
    if canonical_sha256(frozen.get("cluster")) != canonical_sha256(recompute_cluster_table()):
        raise LensingClusterScreenError("frozen cluster table does not replay")
    lensing_grid = build_lensing_grid()
    context = _exact_context(lensing_grid, build_probe_grid())
    controls = {
        name: verify_candidate_exact_gates(ordinal, context)
        for name, ordinal in CONTROL_ORDINALS.items()
    }
    _assert_known_answer_controls(controls)
    for name, verdict in controls.items():
        recorded = value.get("controls", {}).get(name, {})
        for gate in ("lensing", "cluster"):
            if recorded.get(gate, {}).get("passes") != verdict[gate]["passes"]:
                raise LensingClusterScreenError(f"control replay changed for {name}.{gate}")
    for entry in value.get("exact_verification", []):
        if not entry.get("exact_confirmed", False):
            continue
        replay = verify_candidate_exact_gates(entry["ordinal"], context)
        if (
            replay["lensing"]["passes"] != entry["lensing"]["passes"]
            or replay["cluster"]["passes"] != entry["cluster"]["passes"]
        ):
            raise LensingClusterScreenError(f"exact replay failed for ordinal {entry['ordinal']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU lensing (P1) + cluster (P2) gates.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1 << 22)
    parser.add_argument("--cpu", action="store_true", help="force the numpy path")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        if not args.output:
            raise LensingClusterScreenError("--validate-checked requires --output")
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = run_screen(limit=args.limit, batch_size=args.batch_size, use_gpu=not args.cpu)
    if args.output:
        path = Path(args.output)
        encoded = canonical_json_bytes(result) + b"\n"
        if path.exists() and path.read_bytes() != encoded:
            raise LensingClusterScreenError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    print(
        json.dumps(
            {
                "processed": result["counts"]["processed"],
                "fp32_union_survivors": result["counts"]["fp32_union_survivors"],
                "lensing_pass": result["counts"]["lensing_pass"],
                "cluster_pass": result["counts"]["cluster_pass"],
                "both_pass": result["counts"]["both_pass"],
                "decision": result["decision"],
                "elapsed_seconds": result["elapsed_seconds"],
                "throughput_candidates_per_second": result[
                    "throughput_candidates_per_second"
                ],
                "device": result["device"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
