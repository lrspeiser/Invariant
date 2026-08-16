"""Billion-scale GPU screen: baryonic-only rotation phenomenology.

The existing funnel screens ~10^9 static actions cheaply, then jumps straight to
symbolic formal gates that run at minutes per candidate.  Nothing in between ever asks
the physics question the search exists for: *does this candidate produce flat galaxy
rotation curves and the baryonic Tully-Fisher relation from baryons alone?*  The last
production run pushed 5,855 static survivors into the covariant stage and every one
died formally — 70/70 rejected — without a single rotation curve ever being computed.

This module is that missing gate.  A candidate is a universal acceleration law

    g_obs = nu(y) * g_bar,      y = g_bar / a0,

with exactly one universal constant `a0` and **zero per-galaxy freedom**.  The grammar
cannot express a per-galaxy mass rescue, which is the project's no-invisible-halo rule
enforced structurally rather than by policy text.

Candidate family (ordinal-indexed, decodable on GPU):

    nu(y) = (P(u) / Q(u))^beta,   u = y^(-1/2),
    P(u) = 1 + a1 u + ... + a5 u^5,   Q(u) = 1 + b1 u + ... + b5 u^5,
    a_k, b_k in {-3..3},   beta in {1/3, 1/2, 1, 2}

for 4 * 7^10 = 1,129,900,996 candidates.  Deep-limit flatness requires
beta * (deg P - deg Q) = 1/2 * ... to hit nu ~ y^(-1/2); beta = 2 candidates cannot
satisfy it at all.  They are enumerated anyway: the tests must discriminate, the grammar
must not smuggle the answer in.

Screening criteria, all evaluated on frozen synthetic exponential-disk controls
(Freeman 1970 razor-thin disk, g_bar precomputed once with 50-digit mpmath):

  1. definedness/positivity of P/Q at every probe;
  2. Newtonian recovery: nu -> 1 at high acceleration (Solar-System survival);
  3. monotone g_obs(g_bar) (single-valued radial-acceleration relation);
  4. flat outer rotation curves for three disks spanning a 256x baryonic mass range;
  5. baryonic Tully-Fisher slope: d log M / d log v_flat within tolerance of 4.

Three-layer honesty, matching project GPU discipline: an fp32 GPU sweep with slack
thresholds, an fp64 GPU recheck of survivors at strict thresholds, and a 50-digit
mpmath re-verification of every reported Pareto candidate plus an fp64 CPU/GPU
decision cross-check on a random sample.

Claim boundary: survivors are **search priorities, not validated theories**.  No
observational data is opened; the synthetic controls are analytic; the sealed
covariant/formal/observational ladder is untouched and is still the only path to any
physical claim.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-gpu-baryonic-interpolation-screen-result-1.0"

COEFFICIENT_VALUES = tuple(range(-3, 4))  # 7 values
COEFFICIENT_SLOTS = 10  # a1..a5, b1..b5
BETA_VALUES = ("1/3", "1/2", "1", "2")
BETA_FLOATS = (1.0 / 3.0, 0.5, 1.0, 2.0)
FAMILY_SIZE = len(BETA_VALUES) * len(COEFFICIENT_VALUES) ** COEFFICIENT_SLOTS

#: Frozen screen configuration.  Changing any value changes the search claim and the
#: receipt hash.  Disk masses span 256x so the Tully-Fisher lever arm is long; outer
#: radii sit in the deep-acceleration regime where v_flat is defined.
SCREEN_CONFIG: dict[str, Any] = {
    "a0": 1,
    "disk_masses": ["1/250", "8/125", "128/125"],
    "disk_scale_length": 1,
    "outer_radii": [8, 10, 12, 16, 20],
    "inner_radii": [1, 2, 3, 4, 6],
    "newton_probe_y": [10000, 1000000],
    "fp32_thresholds": {
        "newton_far": "3e-3",
        "newton_near": "3e-2",
        "flatness": "8e-2",
        "btfr_slope": "45e-2",
        "positivity_floor": "1e-6",
    },
    "fp64_thresholds": {
        "newton_far": "2e-3",
        "newton_near": "2e-2",
        "flatness": "6e-2",
        "btfr_slope": "30e-2",
        "positivity_floor": "1e-9",
    },
    "mpmath_dps": 50,
    "crosscheck_sample": 4096,
}

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "invisible_mass_used_as_target_or_rescue": False,
    "observational_data_opened": False,
    "per_galaxy_free_parameters_expressible": False,
    "scalar_truth_or_probability_score": False,
    "survivor_is_validated_theory": False,
    "synthetic_analytic_controls_only": True,
    "sealed_validation_ladder_bypassed": False,
}


class BaryonicScreenError(ValueError):
    """Raised on malformed input, decode error, or receipt tamper."""


# ---------------------------------------------------------------------------
# Ordinal codec
# ---------------------------------------------------------------------------


def decode_ordinal(ordinal: int) -> dict[str, Any]:
    """Ordinal -> {beta_index, a[5], b[5]}.  Beta is the most significant digit."""

    if not 0 <= ordinal < FAMILY_SIZE:
        raise BaryonicScreenError(f"ordinal out of range: {ordinal}")
    base = len(COEFFICIENT_VALUES)
    digits: list[int] = []
    value = ordinal
    for _ in range(COEFFICIENT_SLOTS):
        digits.append(value % base - 3)
        value //= base
    beta_index = value
    return {
        "beta_index": beta_index,
        "beta": BETA_VALUES[beta_index],
        "a": digits[:5],
        "b": digits[5:],
    }


def encode_candidate(beta_index: int, a: Sequence[int], b: Sequence[int]) -> int:
    """Inverse of `decode_ordinal`; used by tests and known-answer controls."""

    if beta_index not in range(len(BETA_VALUES)):
        raise BaryonicScreenError("beta index out of range")
    if len(a) != 5 or len(b) != 5:
        raise BaryonicScreenError("need exactly five numerator and denominator digits")
    base = len(COEFFICIENT_VALUES)
    ordinal = beta_index
    for digit in reversed(list(a) + list(b)):
        if digit not in COEFFICIENT_VALUES:
            raise BaryonicScreenError(f"coefficient out of range: {digit}")
        ordinal = ordinal * base + (digit + 3)
    return ordinal


def render_candidate(candidate: Mapping[str, Any]) -> str:
    def poly(coeffs: Sequence[int]) -> str:
        parts = ["1"]
        for power, coefficient in enumerate(coeffs, start=1):
            if coefficient == 0:
                continue
            term = "u" if power == 1 else f"u^{power}"
            if coefficient == 1:
                parts.append(f"+ {term}")
            elif coefficient == -1:
                parts.append(f"- {term}")
            elif coefficient > 0:
                parts.append(f"+ {coefficient}{term}")
            else:
                parts.append(f"- {-coefficient}{term}")
        return " ".join(parts)

    body = f"({poly(candidate['a'])}) / ({poly(candidate['b'])})"
    beta = candidate["beta"]
    return f"nu(y) = [{body}]^{beta},  u = y^(-1/2)"


# ---------------------------------------------------------------------------
# Frozen synthetic-disk grid (candidate-independent; computed once with mpmath)
# ---------------------------------------------------------------------------


def _disk_gbar(mass: mp.mpf, radius: mp.mpf, scale: mp.mpf) -> mp.mpf:
    """Freeman razor-thin exponential disk: g_bar(r) = (M/Rd^2) y B(y), y = r/(2 Rd)."""

    y = radius / (2 * scale)
    bessel = mp.besseli(0, y) * mp.besselk(0, y) - mp.besseli(1, y) * mp.besselk(1, y)
    return (mass / scale**2) * y * bessel


def build_probe_grid() -> dict[str, Any]:
    """All candidate-independent inputs, at 50 digits, frozen into the receipt."""

    mp.mp.dps = SCREEN_CONFIG["mpmath_dps"]
    scale = mp.mpf(SCREEN_CONFIG["disk_scale_length"])
    disks = []
    for mass_text in SCREEN_CONFIG["disk_masses"]:
        mass = mp.mpf(mass_text.split("/")[0]) / mp.mpf(mass_text.split("/")[1])
        radii = [*SCREEN_CONFIG["inner_radii"], *SCREEN_CONFIG["outer_radii"]]
        points = []
        for radius in radii:
            gbar = _disk_gbar(mass, mp.mpf(radius), scale)
            points.append(
                {
                    "radius": radius,
                    "gbar": float(gbar),
                    "outer": radius in SCREEN_CONFIG["outer_radii"],
                }
            )
        disks.append({"mass": float(mass), "mass_text": mass_text, "points": points})
    monotone_y = sorted(
        {point["gbar"] for disk in disks for point in disk["points"]}
        | {float(value) for value in SCREEN_CONFIG["newton_probe_y"]}
    )
    return {"disks": disks, "monotone_y": monotone_y}


# ---------------------------------------------------------------------------
# Vectorized evaluation (shared numpy/cupy code path)
# ---------------------------------------------------------------------------


def _digits_from_ordinals(xp: Any, ordinals: Any) -> tuple[Any, Any]:
    """GPU/CPU mixed-radix decode: ordinals -> (beta_index[B], coeffs[B, 10])."""

    base = len(COEFFICIENT_VALUES)
    value = ordinals.astype(xp.int64)
    coefficients = xp.empty((ordinals.shape[0], COEFFICIENT_SLOTS), dtype=xp.int8)
    for slot in range(COEFFICIENT_SLOTS):
        coefficients[:, slot] = (value % base - 3).astype(xp.int8)
        value //= base
    return value.astype(xp.int8), coefficients


def _nu_at(xp: Any, beta_index: Any, coefficients: Any, y: float, dtype: Any) -> tuple[Any, Any]:
    """Evaluate nu(y) for a whole batch at one scalar probe.  Returns (nu, valid)."""

    u = dtype(y) ** dtype(-0.5)
    numerator = xp.ones(coefficients.shape[0], dtype=dtype)
    denominator = xp.ones(coefficients.shape[0], dtype=dtype)
    power = dtype(1)
    for slot in range(5):
        power = power * u
        numerator = numerator + coefficients[:, slot].astype(dtype) * power
        denominator = denominator + coefficients[:, 5 + slot].astype(dtype) * power
    floor = dtype(
        float(
            SCREEN_CONFIG["fp32_thresholds" if dtype == xp.float32 else "fp64_thresholds"][
                "positivity_floor"
            ]
        )
    )
    valid = (xp.abs(denominator) > floor)
    ratio = numerator / xp.where(valid, denominator, dtype(1))
    valid = valid & (ratio > floor)
    safe = xp.where(valid, ratio, dtype(1))
    nu = xp.where(
        beta_index == 0,
        xp.cbrt(safe),
        xp.where(beta_index == 1, xp.sqrt(safe), xp.where(beta_index == 2, safe, safe * safe)),
    )
    return nu, valid


def screen_batch(
    xp: Any, ordinals: Any, grid: Mapping[str, Any], *, dtype: Any, thresholds: Mapping[str, float]
) -> Any:
    """Full criteria for one ordinal batch.  Returns a boolean survivor mask."""

    beta_index, coefficients = _digits_from_ordinals(xp, ordinals)
    count = ordinals.shape[0]
    alive = xp.ones(count, dtype=bool)

    # 1+2: definedness everywhere and Newtonian recovery at the far probes.
    near_y, far_y = (float(v) for v in SCREEN_CONFIG["newton_probe_y"])
    nu_near, valid = _nu_at(xp, beta_index, coefficients, near_y, dtype)
    alive &= valid & (xp.abs(nu_near - 1) <= dtype(float(thresholds["newton_near"])))
    nu_far, valid = _nu_at(xp, beta_index, coefficients, far_y, dtype)
    alive &= valid & (xp.abs(nu_far - 1) <= dtype(float(thresholds["newton_far"])))

    # 3: monotone g_obs over the full sorted acceleration grid.
    previous = None
    for y in grid["monotone_y"]:
        nu, valid = _nu_at(xp, beta_index, coefficients, y, dtype)
        alive &= valid
        gobs = dtype(y) * nu
        if previous is not None:
            alive &= gobs > previous
        previous = gobs

    # 4+5: outer-curve flatness per disk and the cross-disk Tully-Fisher slope.
    vflat = []
    for disk in grid["disks"]:
        vmax = None
        vmin = None
        vsum = xp.zeros(count, dtype=dtype)
        outer_count = 0
        for point in disk["points"]:
            if not point["outer"]:
                continue
            nu, valid = _nu_at(xp, beta_index, coefficients, point["gbar"], dtype)
            alive &= valid
            vsq = dtype(point["gbar"]) * nu * dtype(point["radius"])
            speed = xp.sqrt(xp.maximum(vsq, dtype(0)))
            vmax = speed if vmax is None else xp.maximum(vmax, speed)
            vmin = speed if vmin is None else xp.minimum(vmin, speed)
            vsum = vsum + speed
            outer_count += 1
        mean = vsum / dtype(outer_count)
        alive &= (vmax - vmin) <= dtype(float(thresholds["flatness"])) * mean
        vflat.append(mean)

    mass_low = grid["disks"][0]["mass"]
    mass_high = grid["disks"][-1]["mass"]
    log_mass_span = dtype(float(np.log(mass_high / mass_low)))
    ratio = xp.log(xp.maximum(vflat[-1], dtype(1e-30)) / xp.maximum(vflat[0], dtype(1e-30)))
    slope_ok = xp.abs(log_mass_span / xp.where(ratio > 0, ratio, dtype(1e-30)) - 4) <= dtype(
        float(thresholds["btfr_slope"])
    )
    alive &= slope_ok & (ratio > 0)
    return alive


def _candidate_metrics(xp: Any, ordinals: Any, grid: Mapping[str, Any], dtype: Any) -> dict[str, Any]:
    """Pareto axes for surviving ordinals (fp64): simplicity, convergence, flatness."""

    beta_index, coefficients = _digits_from_ordinals(xp, ordinals)
    nonzero = (coefficients != 0).sum(axis=1).astype(xp.int32) + (beta_index != 2).astype(
        xp.int32
    )
    near_y = float(SCREEN_CONFIG["newton_probe_y"][0])
    nu_near, _ = _nu_at(xp, beta_index, coefficients, near_y, dtype)
    newton_error = xp.abs(nu_near - 1)
    worst_flatness = xp.zeros(ordinals.shape[0], dtype=dtype)
    for disk in grid["disks"]:
        vmax = None
        vmin = None
        vsum = xp.zeros(ordinals.shape[0], dtype=dtype)
        outer_count = 0
        for point in disk["points"]:
            if not point["outer"]:
                continue
            nu, _ = _nu_at(xp, beta_index, coefficients, point["gbar"], dtype)
            vsq = dtype(point["gbar"]) * nu * dtype(point["radius"])
            speed = xp.sqrt(xp.maximum(vsq, dtype(0)))
            vmax = speed if vmax is None else xp.maximum(vmax, speed)
            vmin = speed if vmin is None else xp.minimum(vmin, speed)
            vsum = vsum + speed
            outer_count += 1
        worst_flatness = xp.maximum(worst_flatness, (vmax - vmin) / (vsum / dtype(outer_count)))
    return {"simplicity": nonzero, "newton_error": newton_error, "flatness": worst_flatness}


# ---------------------------------------------------------------------------
# Exact re-verification (mpmath, 50 digits)
# ---------------------------------------------------------------------------


def verify_candidate_exact(ordinal: int, grid: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run every criterion for one candidate at 50-digit precision."""

    mp.mp.dps = SCREEN_CONFIG["mpmath_dps"]
    candidate = decode_ordinal(ordinal)
    beta = {"1/3": mp.mpf(1) / 3, "1/2": mp.mpf(1) / 2, "1": mp.mpf(1), "2": mp.mpf(2)}[
        candidate["beta"]
    ]

    def nu(y: mp.mpf) -> mp.mpf | None:
        u = 1 / mp.sqrt(y)
        numerator = mp.mpf(1)
        denominator = mp.mpf(1)
        for power, (a_k, b_k) in enumerate(
            zip(candidate["a"], candidate["b"], strict=True), start=1
        ):
            numerator += a_k * u**power
            denominator += b_k * u**power
        if denominator == 0:
            return None
        ratio = numerator / denominator
        if ratio <= 0:
            return None
        return ratio**beta

    thresholds = {key: mp.mpf(value) for key, value in SCREEN_CONFIG["fp64_thresholds"].items()}
    checks: dict[str, bool] = {}
    near_y, far_y = (mp.mpf(v) for v in SCREEN_CONFIG["newton_probe_y"])
    nu_near, nu_far = nu(near_y), nu(far_y)
    checks["defined"] = nu_near is not None and nu_far is not None
    if checks["defined"]:
        checks["newton_near"] = abs(nu_near - 1) <= thresholds["newton_near"]
        checks["newton_far"] = abs(nu_far - 1) <= thresholds["newton_far"]
    previous = None
    monotone = True
    for y in grid["monotone_y"]:
        value = nu(mp.mpf(y))
        if value is None:
            monotone = False
            break
        gobs = mp.mpf(y) * value
        if previous is not None and gobs <= previous:
            monotone = False
            break
        previous = gobs
    checks["monotone"] = monotone
    vflat = []
    flat = True
    for disk in grid["disks"]:
        speeds = []
        for point in disk["points"]:
            if not point["outer"]:
                continue
            value = nu(mp.mpf(point["gbar"]))
            if value is None:
                flat = False
                break
            speeds.append(mp.sqrt(mp.mpf(point["gbar"]) * value * point["radius"]))
        if not flat:
            break
        mean = sum(speeds) / len(speeds)
        if (max(speeds) - min(speeds)) > thresholds["flatness"] * mean:
            flat = False
            break
        vflat.append(mean)
    checks["flat_curves"] = flat
    if flat and len(vflat) == len(grid["disks"]):
        slope = mp.log(
            mp.mpf(grid["disks"][-1]["mass"]) / mp.mpf(grid["disks"][0]["mass"])
        ) / mp.log(vflat[-1] / vflat[0])
        checks["btfr_slope"] = abs(slope - 4) <= thresholds["btfr_slope"]
        slope_value = float(slope)  # stringified at receipt boundary
    else:
        checks["btfr_slope"] = False
        slope_value = None
    return {
        "ordinal": ordinal,
        "candidate": candidate,
        "formula": render_candidate(candidate),
        "checks": checks,
        "passes": all(checks.values()),
        "btfr_slope": None if slope_value is None else format(slope_value, ".9e"),
    }


# ---------------------------------------------------------------------------
# Campaign driver
# ---------------------------------------------------------------------------


def run_screen(
    *,
    limit: int | None = None,
    batch_size: int = 1 << 22,
    use_gpu: bool = True,
    pareto_cap: int = 64,
) -> dict[str, Any]:
    """Screen the family (or its first `limit` ordinals) and seal a receipt."""

    if use_gpu:
        import cupy as xp

        device_properties = xp.cuda.runtime.getDeviceProperties(0)
        device = device_properties["name"].decode()
    else:
        xp = np
        device = "cpu-numpy"

    grid = build_probe_grid()
    total = FAMILY_SIZE if limit is None else min(limit, FAMILY_SIZE)
    started = time.perf_counter()

    survivor_ordinals: list[int] = []
    processed = 0
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        ordinals = xp.arange(start, stop, dtype=xp.int64)
        alive32 = screen_batch(
            xp,
            ordinals,
            grid,
            dtype=xp.float32,
            thresholds=SCREEN_CONFIG["fp32_thresholds"],
        )
        candidates = ordinals[alive32]
        if candidates.shape[0]:
            alive64 = screen_batch(
                xp,
                candidates,
                grid,
                dtype=xp.float64,
                thresholds=SCREEN_CONFIG["fp64_thresholds"],
            )
            final = candidates[alive64]
            if final.shape[0]:
                survivor_ordinals.extend(
                    int(v) for v in (final.get() if use_gpu else final)
                )
        processed = stop
    elapsed = time.perf_counter() - started

    # Pareto metrics for all fp64 survivors, computed vectorized end to end.
    # Building one Python dict per survivor and running an O(n^2) Python front does
    # not survive contact with millions of survivors; everything stays in arrays and
    # the exact front is computed by numpy broadcasting over an axis-wise prefilter.
    survivors = np.asarray(survivor_ordinals, dtype=np.int64)
    simplicity_all = np.empty(0, dtype=np.int32)
    newton_all = np.empty(0, dtype=np.float64)
    flatness_all = np.empty(0, dtype=np.float64)
    if survivors.size:
        parts_s, parts_n, parts_f = [], [], []
        for start in range(0, survivors.size, batch_size):
            chunk = xp.asarray(survivors[start : start + batch_size])
            metrics = _candidate_metrics(xp, chunk, grid, xp.float64)
            parts_s.append(metrics["simplicity"].get() if use_gpu else metrics["simplicity"])
            parts_n.append(metrics["newton_error"].get() if use_gpu else metrics["newton_error"])
            parts_f.append(metrics["flatness"].get() if use_gpu else metrics["flatness"])
        simplicity_all = np.concatenate(parts_s)
        newton_all = np.concatenate(parts_n)
        flatness_all = np.concatenate(parts_f)

    prefilter = min(4096, survivors.size)
    if survivors.size:
        chosen = np.unique(
            np.concatenate(
                [
                    np.lexsort((newton_all, flatness_all, simplicity_all))[:prefilter],
                    np.lexsort((simplicity_all, flatness_all, newton_all))[:prefilter],
                    np.lexsort((simplicity_all, newton_all, flatness_all))[:prefilter],
                ]
            )
        )
        axes = np.stack(
            [
                simplicity_all[chosen].astype(np.float64),
                newton_all[chosen],
                flatness_all[chosen],
            ],
            axis=1,
        )
        not_worse = (axes[:, None, :] <= axes[None, :, :]).all(axis=2)
        strictly_better = (axes[:, None, :] < axes[None, :, :]).any(axis=2)
        dominated = (not_worse & strictly_better).any(axis=0)
        front_index = chosen[~dominated]
        order = np.lexsort((newton_all[front_index], simplicity_all[front_index]))
        front_index = front_index[order][:pareto_cap]
        front = [
            {
                "ordinal": int(survivors[i]),
                "simplicity": int(simplicity_all[i]),
                "newton_error": float(newton_all[i]),
                "flatness": float(flatness_all[i]),
            }
            for i in front_index
        ]
    else:
        front = []

    # Layer 3: exact re-verification of every reported candidate.
    verified = [verify_candidate_exact(entry["ordinal"], grid) for entry in front]
    confirmed = [item for item in verified if item["passes"]]
    refuted = [item for item in verified if not item["passes"]]

    # CPU/GPU decision cross-check on a deterministic sample.
    crosscheck: dict[str, Any] = {"performed": False}
    if use_gpu:
        rng = np.random.default_rng(20260814)
        sample = np.sort(
            rng.choice(total, size=min(SCREEN_CONFIG["crosscheck_sample"], total), replace=False)
        ).astype(np.int64)
        import cupy as cp

        gpu_decision = screen_batch(
            cp,
            cp.asarray(sample),
            grid,
            dtype=cp.float64,
            thresholds=SCREEN_CONFIG["fp64_thresholds"],
        ).get()
        cpu_decision = screen_batch(
            np,
            sample,
            grid,
            dtype=np.float64,
            thresholds=SCREEN_CONFIG["fp64_thresholds"],
        )
        disagreements = int((gpu_decision != cpu_decision).sum())
        crosscheck = {
            "performed": True,
            "sample": int(sample.shape[0]),
            "disagreements": disagreements,
        }

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "config": SCREEN_CONFIG,
        "config_sha256": canonical_sha256(SCREEN_CONFIG),
        "counts": {
            "family_size": FAMILY_SIZE,
            "processed": processed,
            "fp64_survivors": len(survivor_ordinals),
            "pareto_front": len(front),
            "exact_confirmed": len(confirmed),
            "exact_refuted": len(refuted),
        },
        "crosscheck": crosscheck,
        "decision": "SCREENED",
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "pareto_front": [
            {
                "ordinal": entry["ordinal"],
                "simplicity": entry["simplicity"],
                "newton_error": format(entry["newton_error"], ".9e"),
                "flatness": format(entry["flatness"], ".9e"),
                "formula": render_candidate(decode_ordinal(entry["ordinal"])),
                "exact_confirmed": any(
                    item["ordinal"] == entry["ordinal"] for item in confirmed
                ),
            }
            for entry in front
        ],
        "exact_verification": verified,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "GPU phenomenological screen over an ordinal-indexed family of universal "
            "baryonic acceleration laws with one shared constant and no per-galaxy "
            "freedom, tested on frozen synthetic exponential-disk controls for "
            "Newtonian recovery, monotone dynamics, flat outer rotation curves, and "
            "the baryonic Tully-Fisher slope. Survivors are search priorities for the "
            "sealed covariant/formal/observational ladder. Nothing here opens "
            "observational data, uses invisible mass, or validates a theory."
        ),
        "throughput_candidates_per_second": (
            int(processed / elapsed) if elapsed > 0 else None
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal and schema check.  A full replay is a deliberate re-run, not a validation."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise BaryonicScreenError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise BaryonicScreenError("receipt seal changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise BaryonicScreenError("config binding changed")
    for entry in value.get("pareto_front", []):
        if not entry.get("exact_confirmed", False):
            continue
        # Every confirmed front entry must replay exactly under mpmath.
        replay = verify_candidate_exact(entry["ordinal"], build_probe_grid())
        if not replay["passes"]:
            raise BaryonicScreenError(f"exact replay failed for ordinal {entry['ordinal']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Billion-scale baryonic GPU screen.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1 << 22)
    parser.add_argument("--cpu", action="store_true", help="force the numpy path")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = run_screen(
        limit=args.limit, batch_size=args.batch_size, use_gpu=not args.cpu
    )
    if args.output:
        path = Path(args.output)
        encoded = canonical_json_bytes(result) + b"\n"
        if path.exists() and path.read_bytes() != encoded:
            raise BaryonicScreenError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    print(
        json.dumps(
            {
                "processed": result["counts"]["processed"],
                "fp64_survivors": result["counts"]["fp64_survivors"],
                "pareto_front": result["counts"]["pareto_front"],
                "exact_confirmed": result["counts"]["exact_confirmed"],
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
