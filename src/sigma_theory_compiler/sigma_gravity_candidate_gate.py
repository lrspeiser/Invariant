"""Sigma-Gravity published-point gauntlet and GPU parameter-neighborhood scan.

An external candidate law is evaluated through the same three synthetic gate families
the ordinal screens use, without touching the sealed validation ladder.  The candidate
is the published Sigma-Gravity formula (github.com/lrspeiser/sigmagravity, README
II.A-II.E):

    Sigma(g_N, C, L) = 1 + A(L) * C * h(g_N),        g_eff = Sigma * g_N,
    h(g_N) = sqrt(g_dagger/g_N) * g_dagger/(g_dagger + g_N),
    g_dagger = c*H0/(4*sqrt(pi)) = 9.6e-11 m/s^2,
    A(L) = A0 * (L/L0)^n,  A0 = exp(1/(2*pi)),  L0 = 0.4 kpc,  n = 0.27,
    disk coherence C = W(r) = r/(xi + r),  xi = R_d/(2*pi).

Unit mapping, declared as exact assumptions in the receipt: the screens set
a0 = 1.2e-10 m/s^2 = 1, so g_dagger = 4/5 exactly; R_d = 1 (the screen disk scale
length) so xi = 1/(2*pi); thin disks have L = L0 so A_disk = A0 for every n; clusters
have L = 600 kpc so A_cluster = A0 * 1500^n (~8.446 at n = 0.27).  The published
cluster coherence is C = 1: SI 13.3 states "Galaxy Clusters | W = 1 at r = 200 kpc
(xi irrelevant)" and the repository's cluster code sets W = 1.0; the README V.A.2
narrative that dispersion-dominated systems have reduced C is therefore covered by
scanning C_cluster over [0, 1] rather than by trusting one reading.

Stage 1 (gauntlet) re-runs the published point at 50-digit mpmath against the frozen
synthetic controls of the existing screens (disk grid, Hernquist lensing path nodes,
beta-model cluster table -- imported, not reinvented) with the same fp64 thresholds:
Newtonian recovery, monotone g_eff (both on the real (r, g_bar) disk pairs with W(r)
per point, and separately at the declared coherence bound C = 1), flat outer curves,
the Tully-Fisher slope, the P1 Born-deflection lensing consistency, and the P2
hydrostatic cluster ratio, plus a report-only Solar-System proxy at y = 1e8.

Stage 2 (neighborhood) scans 9*13*13*11*5*4 = 334,620 ordinal-indexed parameter
combinations around the published values (A0, n, g_dagger, C_cluster, and the h-shape
generalization h = (gd/g)^p * (gd/(gd+g))^q) through all the same gates, vectorized in
fp64 on GPU with a CPU cross-check, a Pareto front over (cluster deviation, lensing
consistency, Newton-far error, distance-from-published), and 50-digit re-verification
of every reported candidate.  A zero-survivor outcome is a sealed negative for this
neighborhood, reported with the margins of the closest candidates.

Claim boundary: the controls are synthetic and analytic; no observational data is
opened; a gate pass is not physical validation; a gate failure is not a refutation of
the paper's SPARC/Fox fits (real-data fits answer a different question than these
synthetic single-valuedness and shape gates); the scan is not a calibration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from .gpu_baryonic_interpolation_screen import SCREEN_CONFIG, build_probe_grid
from .gpu_baryonic_lensing_cluster_screen import (
    GATE_CONFIG,
    _fraction,
    build_lensing_grid,
    recompute_cluster_table,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

GAUNTLET_SCHEMA = "invariant-sigma-gravity-gauntlet-1.0"
NEIGHBORHOOD_SCHEMA = "invariant-sigma-gravity-neighborhood-scan-1.0"

#: Frozen candidate description.  Every value is an exact string; changing any value
#: changes the claim and the receipt hash.
CANDIDATE_CONFIG: dict[str, Any] = {
    "mpmath_dps": 50,
    "formula": (
        "Sigma(g_N, C, L) = 1 + A(L)*C*h(g_N); h(g_N) = (g_dagger/g_N)^(1/2) * "
        "g_dagger/(g_dagger + g_N); A(L) = A0*(L/L0)^n; g_eff = Sigma*g_N; "
        "disk coherence C = W(r) = r/(xi + r)"
    ),
    "source": "https://github.com/lrspeiser/sigmagravity README II.A-II.E, SI 6, SI 13.3",
    "published_parameters": {
        "A0": "exp(1/(2*pi))",
        "n": "27/100",
        "g_dagger_si": "9.6e-11 m/s^2 = c*H0/(4*sqrt(pi)) at H0 = 70 km/s/Mpc",
        "g_dagger_code": "4/5",
        "L0_kpc": "2/5",
        "L_disk_kpc": "2/5",
        "L_cluster_kpc": "600",
        "cluster_L_ratio": "1500",
        "R_d_code": "1",
        "xi_code": "1/(2*pi)",
        "C_cluster_published": "1",
        "h_shape_p": "1/2",
        "h_shape_q": "1",
    },
    "unit_mapping": {"a0_si": "1.2e-10 m/s^2", "a0_code": "1", "g_dagger_over_a0": "4/5"},
    "newton_probe_y": [10000, 1000000],
    "solar_probe_y": 100000000,
    "fp64_thresholds": {
        "newton_near": "2e-2",
        "newton_far": "2e-3",
        "flatness": "6e-2",
        "btfr_slope": "30e-2",
        "lensing_flatness": "8e-2",
        "lensing_consistency": "15e-2",
        "cluster_consistency": "15e-2",
    },
}

#: Frozen scan configuration.  Axes are ordinal-indexed most-significant-first in
#: ``axis_order``; every axis brackets the published value, and the published A0 and n
#: are deliberately off-grid (recorded as null), so exact-equality distance counts at
#: most 4 of 6 axes for any grid candidate.
SCAN_CONFIG: dict[str, Any] = {
    "axis_order": ["A0", "n", "g_dagger", "C_cluster", "p", "q"],
    "axes": {
        "A0": ["0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"],
        "n": [
            "0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30",
            "0.35", "0.40", "0.45", "0.50", "0.55", "0.60",
        ],
        "g_dagger": [
            "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0",
            "1.1", "1.2", "1.3", "1.4", "1.5", "1.6",
        ],
        "C_cluster": [
            "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0",
        ],
        "p": ["0.40", "0.45", "0.50", "0.55", "0.60"],
        "q": ["0.5", "1.0", "1.5", "2.0"],
    },
    "published_on_grid_axes": {
        "A0": None,
        "n": None,
        "g_dagger": "0.8",
        "C_cluster": "1.0",
        "p": "0.50",
        "q": "1.0",
    },
    "total_candidates": 334620,
    "crosscheck_sample": 2048,
    "crosscheck_seed": 20260815,
    "pareto": {
        "axes": [
            "cluster_max_deviation",
            "lensing_worst_consistency",
            "newton_far_error",
            "published_distance",
        ],
        "prefilter_per_axis": 4096,
        "reported_cap": 64,
    },
    "closest_report": {
        "count": 20,
        "order": "fewest failed top-level gates, then cluster_max_deviation, then ordinal",
    },
    "max_exact_verifications": 64,
    "mpmath_dps": 50,
}

CLAIMS_GAUNTLET = {
    "external_formula_under_test": True,
    "gate_failure_refutes_paper_observational_fits": False,
    "observational_data_opened": False,
    "pass_is_not_physical_validation": True,
    "per_object_free_parameters_expressible": False,
    "real_observational_data_used": False,
    "scan_is_not_calibration": True,
    "synthetic_controls_only": True,
}

CLAIMS_SCAN = {
    "external_formula_under_test": True,
    "observational_data_opened": False,
    "pass_is_not_physical_validation": True,
    "real_observational_data_used": False,
    "scan_is_not_calibration": True,
    "scan_is_refutation_of_paper_sparc_fox_fits": False,
    "survivor_is_validated_theory": False,
    "synthetic_controls_only": True,
}

#: Declared assumptions.  Everything a reader needs to reproduce the unit mapping and
#: the coherence readings without opening the Sigma-Gravity repository.
ASSUMPTIONS: dict[str, str] = {
    "unit_mapping": (
        "a0 = 1.2e-10 m/s^2 = 1 code unit; g_dagger = 9.6e-11 m/s^2 = 4/5 exactly "
        "(9.6/12); every acceleration below is in a0 units, so the screens' y is g_N"
    ),
    "disk_controls": (
        "R_d = 1 code unit (the screen disk_scale_length), so xi = 1/(2*pi); thin disks "
        "take L = L0 = 0.4 kpc (README II.E), so A_disk = A0*(L/L0)^n = A0 for every n"
    ),
    "cluster_amplitude": (
        "cluster path length L = 600 kpc (README II.E, SI 6), so A_cluster = "
        "A0*(600/0.4)^n = A0*1500^n, computed exactly at runtime (about 8.446 at "
        "published A0 and n = 0.27)"
    ),
    "c_cluster_published": (
        "published cluster coherence is 1: SI 13.3 row 'Galaxy Clusters | W = 1 at "
        "r = 200 kpc (xi irrelevant)' and derivations/cluster_holdout_validation.py "
        "predict_cluster_mass sets W = 1.0; README V.A.2 narrates reduced coherence for "
        "dispersion-dominated systems, so the neighborhood stage scans C_cluster over "
        "[0, 1] instead of trusting a single reading"
    ),
    "lensing_coherence": (
        "the P1 Hernquist controls are galaxy-scale, so coherence at each frozen path "
        "node is the disk closed form W(r) = r/(xi + r) at the node's spherical radius "
        "r = sqrt(mass/y) - 1 (recovered exactly from the frozen node y), with amplitude "
        "A(L0) = A0; every node radius is >= 8, which is 50x xi, so W is within 2% of 1"
    ),
    "newton_and_solar_probes": (
        "the pure-acceleration probes (Newton y = 1e4 and 1e6, the Solar proxy y = 1e8, "
        "and the C = 1 monotonicity chain) carry no radius, so the declared coherence "
        "upper bound C = 1 is used: Newtonian recovery must hold even at full coherence"
    ),
    "boost_semantics": (
        "Sigma multiplies g_N exactly where the screens apply nu: g_eff = Sigma*g_N; "
        "for disks Sigma depends on radius through W(r), so the radial-acceleration "
        "relation is evaluated on the frozen (r, g_bar) pairs with W(r) per point, and "
        "the screens' pooled monotone criterion becomes a cross-disk single-valuedness "
        "test of that relation"
    ),
    "threshold_tier": (
        "only the strict fp64 thresholds of the existing screens are used; no fp32 "
        "slack tier exists in this gauntlet"
    ),
}

_AXIS_SIZES = (9, 13, 13, 11, 5, 4)
TOTAL_SCAN_CANDIDATES = 334620


class SigmaGravityGateError(ValueError):
    """Raised on malformed input, a broken control, or receipt tamper."""


def _dps() -> None:
    mp.mp.dps = CANDIDATE_CONFIG["mpmath_dps"]


def published_exact_parameters() -> dict[str, mp.mpf]:
    """The published point at 50 digits, from the exact strings in the config."""

    _dps()
    return {
        "A0": mp.e ** (1 / (2 * mp.pi)),
        "n": mp.mpf(27) / 100,
        "g_dagger": mp.mpf(4) / 5,
        "p": mp.mpf(1) / 2,
        "q": mp.mpf(1),
        "C_cluster": mp.mpf(1),
    }


# ---------------------------------------------------------------------------
# Frozen candidate-independent control pack (reused grids + coherence window)
# ---------------------------------------------------------------------------


def build_control_pack() -> dict[str, Any]:
    """All candidate-independent inputs at 50 digits, from the existing frozen grids.

    The disk grid, lensing path nodes, and cluster probe table are imported from the
    two GPU screens verbatim; the only additions are the coherence window W(r) per
    point (candidate-independent) and the node radii recovered exactly from the frozen
    node accelerations.
    """

    _dps()
    xi = 1 / (2 * mp.pi)
    disk_grid = build_probe_grid()
    lensing_grid = build_lensing_grid()
    pooled: list[tuple[mp.mpf, int, mp.mpf, int]] = []
    disks: list[dict[str, Any]] = []
    for disk_index, disk in enumerate(disk_grid["disks"]):
        outer = []
        for point in disk["points"]:
            gbar = mp.mpf(point["gbar"])
            radius = mp.mpf(point["radius"])
            window = radius / (xi + radius)
            pooled.append((gbar, point["radius"], window, disk_index))
            if point["outer"]:
                outer.append((gbar, point["radius"], window))
        disks.append({"mass": mp.mpf(disk["mass"]), "mass_text": disk["mass_text"], "outer": outer})
    pooled.sort(key=lambda item: item[0])
    scale = mp.mpf(GATE_CONFIG["lensing"]["hernquist_scale"])
    lensing: list[dict[str, Any]] = []
    for integral in lensing_grid["integrals"]:
        mass = _fraction(integral["mass_text"])
        nodes = []
        for node in integral["nodes"]:
            y = mp.mpf(node["y"])
            radius = mp.sqrt(mass / y) - scale
            nodes.append((y, mp.mpf(node["weight"]), radius / (xi + radius)))
        lensing.append(
            {
                "mass_index": integral["mass_index"],
                "mass_text": integral["mass_text"],
                "impact_parameter": integral["impact_parameter"],
                "nodes": nodes,
            }
        )
    cluster_config = GATE_CONFIG["cluster"]
    cluster = [
        (mp.mpf(gbar), _fraction(gdyn))
        for gbar, gdyn in zip(
            cluster_config["gbar_50dps"], cluster_config["gdyn_exact"], strict=True
        )
    ]
    return {
        "xi": xi,
        "pooled": pooled,
        "disks": disks,
        "lensing": lensing,
        "cluster": cluster,
        "monotone_c1_y": [mp.mpf(y) for y in disk_grid["monotone_y"]],
        "disk_grid": disk_grid,
        "lensing_grid": lensing_grid,
    }


def _stringified_disk_grid(disk_grid: Mapping[str, Any]) -> dict[str, Any]:
    """Float-free binding of the frozen disk grid (17 digits round-trips float64)."""

    return {
        "disks": [
            {
                "mass_text": disk["mass_text"],
                "mass": format(disk["mass"], ".17e"),
                "points": [
                    {
                        "radius": point["radius"],
                        "gbar": format(point["gbar"], ".17e"),
                        "outer": point["outer"],
                    }
                    for point in disk["points"]
                ],
            }
            for disk in disk_grid["disks"]
        ],
        "monotone_y": [format(value, ".17e") for value in disk_grid["monotone_y"]],
    }


def frozen_grid_bindings(pack: Mapping[str, Any]) -> dict[str, str]:
    return {
        "disk_grid_sha256": canonical_sha256(_stringified_disk_grid(pack["disk_grid"])),
        "lensing_grid_sha256": canonical_sha256(pack["lensing_grid"]),
        "cluster_table_sha256": canonical_sha256(recompute_cluster_table()),
    }


# ---------------------------------------------------------------------------
# Exact evaluation (mpmath, 50 digits)
# ---------------------------------------------------------------------------


def _h_exact(y: mp.mpf, gd: mp.mpf, p: mp.mpf, q: mp.mpf) -> mp.mpf:
    return (gd / y) ** p * (gd / (gd + y)) ** q


def evaluate_candidate_exact(
    parameters: Mapping[str, Any] | None = None, pack: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """All gates for one candidate at 50 digits, with decimal-string margins.

    ``parameters`` overrides any of A0, n, g_dagger, p, q, C_cluster with exact
    strings; the default is the published point.
    """

    _dps()
    pack = pack if pack is not None else build_control_pack()
    values = published_exact_parameters()
    for key, text in (parameters or {}).items():
        if key not in values:
            raise SigmaGravityGateError(f"unknown candidate parameter: {key}")
        values[key] = _fraction(str(text))
    a0, gd, p, q = values["A0"], values["g_dagger"], values["p"], values["q"]
    c_cluster = values["C_cluster"]
    a_cluster = a0 * mp.mpf(1500) ** values["n"]
    thresholds = {
        key: mp.mpf(text) for key, text in CANDIDATE_CONFIG["fp64_thresholds"].items()
    }

    def sigma(y: mp.mpf, coherence: mp.mpf, amplitude: mp.mpf) -> mp.mpf:
        return 1 + amplitude * coherence * _h_exact(y, gd, p, q)

    def _text(value: mp.mpf) -> str:
        return format(float(value), ".9e")

    # Newtonian recovery and the Solar proxy at the declared coherence bound C = 1.
    near_y, far_y = (mp.mpf(v) for v in CANDIDATE_CONFIG["newton_probe_y"])
    near_error = abs(sigma(near_y, mp.mpf(1), a0) - 1)
    far_error = abs(sigma(far_y, mp.mpf(1), a0) - 1)
    solar_error = abs(sigma(mp.mpf(CANDIDATE_CONFIG["solar_probe_y"]), mp.mpf(1), a0) - 1)
    newton = {
        "near": {
            "y": CANDIDATE_CONFIG["newton_probe_y"][0],
            "error": _text(near_error),
            "threshold": CANDIDATE_CONFIG["fp64_thresholds"]["newton_near"],
            "pass": bool(near_error <= thresholds["newton_near"]),
        },
        "far": {
            "y": CANDIDATE_CONFIG["newton_probe_y"][1],
            "error": _text(far_error),
            "threshold": CANDIDATE_CONFIG["fp64_thresholds"]["newton_far"],
            "pass": bool(far_error <= thresholds["newton_far"]),
        },
    }
    newton_pass = newton["near"]["pass"] and newton["far"]["pass"]

    # Monotonicity at the declared C = 1 bound over the screens' pooled y grid.
    previous = None
    c1_min_margin = None
    c1_monotone = True
    for y in pack["monotone_c1_y"]:
        geff = y * sigma(y, mp.mpf(1), a0)
        if previous is not None:
            margin = geff / previous - 1
            c1_min_margin = margin if c1_min_margin is None else min(c1_min_margin, margin)
            if geff <= previous:
                c1_monotone = False
        previous = geff

    # Radial-acceleration-relation single-valuedness on the real (r, g_bar) pairs.
    pooled_geff = [
        (gbar * sigma(gbar, window, a0), gbar, radius, disk_index)
        for gbar, radius, window, disk_index in pack["pooled"]
    ]
    violations = []
    rar_min_margin = None
    for before, after in itertools.pairwise(pooled_geff):
        margin = after[0] / before[0] - 1
        rar_min_margin = margin if rar_min_margin is None else min(rar_min_margin, margin)
        if after[0] <= before[0]:
            violations.append(
                {
                    "before": {
                        "disk": before[3],
                        "radius": before[2],
                        "gbar": _text(before[1]),
                        "geff": _text(before[0]),
                    },
                    "after": {
                        "disk": after[3],
                        "radius": after[2],
                        "gbar": _text(after[1]),
                        "geff": _text(after[0]),
                    },
                    "geff_ratio_minus_1": _text(margin),
                }
            )
    per_disk_monotone = True
    for disk_index in range(len(pack["disks"])):
        chain = [entry for entry in pooled_geff if entry[3] == disk_index]
        for before, after in itertools.pairwise(chain):
            if after[0] <= before[0]:
                per_disk_monotone = False

    # Flat outer curves and the Tully-Fisher slope, with W(r) per point.
    per_disk = []
    vflat = []
    flat_pass = True
    for disk in pack["disks"]:
        speeds = [
            mp.sqrt(gbar * sigma(gbar, window, a0) * radius)
            for gbar, radius, window in disk["outer"]
        ]
        mean = sum(speeds) / len(speeds)
        spread = (max(speeds) - min(speeds)) / mean
        passed = bool(spread <= thresholds["flatness"])
        flat_pass = flat_pass and passed
        vflat.append(mean)
        per_disk.append(
            {
                "mass_text": disk["mass_text"],
                "spread": _text(spread),
                "v_flat": _text(mean),
                "pass": passed,
            }
        )
    ratio = mp.log(vflat[-1] / vflat[0])
    if ratio > 0:
        slope = mp.log(pack["disks"][-1]["mass"] / pack["disks"][0]["mass"]) / ratio
        slope_error = abs(slope - 4)
        btfr_pass = bool(slope_error <= thresholds["btfr_slope"])
        slope_text = _text(slope)
    else:
        btfr_pass = False
        slope_text = None
        slope_error = mp.inf

    # P1: Born deflection over the frozen Hernquist paths against 2*pi*v_flat^2.
    worst_flatness = mp.mpf(0)
    worst_consistency = mp.mpf(0)
    alphas_by_mass: dict[int, list[mp.mpf]] = {}
    for integral in pack["lensing"]:
        total = mp.mpf(0)
        for y, weight, window in integral["nodes"]:
            total += weight * sigma(y, window, a0)
        alphas_by_mass.setdefault(integral["mass_index"], []).append(total)
    lensing_per_mass = []
    lensing_valid = True
    for mass_index, alphas in sorted(alphas_by_mass.items()):
        expected = 2 * mp.pi * vflat[mass_index] ** 2
        mean = sum(alphas) / len(alphas)
        if expected <= 0 or mean <= 0:
            lensing_valid = False
            break
        spread = (max(alphas) - min(alphas)) / mean
        consistency = max(abs(alpha / expected - 1) for alpha in alphas)
        worst_flatness = max(worst_flatness, spread)
        worst_consistency = max(worst_consistency, consistency)
        lensing_per_mass.append(
            {
                "mass_text": pack["disks"][mass_index]["mass_text"],
                "alpha_spread": _text(spread),
                "worst_consistency": _text(consistency),
            }
        )
    lensing_pass = bool(
        lensing_valid
        and worst_flatness <= thresholds["lensing_flatness"]
        and worst_consistency <= thresholds["lensing_consistency"]
    )

    # P2: hydrostatic cluster ratio with the declared cluster amplitude and coherence.
    ratios = []
    deviations = []
    shortfalls = []
    for gbar, gdyn in pack["cluster"]:
        boost = sigma(gbar, c_cluster, a_cluster)
        value = gbar * boost / gdyn
        ratios.append(value)
        deviations.append(abs(value - 1))
        shortfalls.append(1 / value)
    cluster_max = max(deviations)
    cluster_pass = bool(cluster_max <= thresholds["cluster_consistency"])

    galaxy_pass = bool(
        newton_pass and c1_monotone and not violations and flat_pass and btfr_pass
    )
    return {
        "parameters": {
            "A0": mp.nstr(a0, 30),
            "n": mp.nstr(values["n"], 30),
            "g_dagger": mp.nstr(gd, 30),
            "p": mp.nstr(p, 30),
            "q": mp.nstr(q, 30),
            "C_cluster": mp.nstr(c_cluster, 30),
            "A_disk": mp.nstr(a0, 30),
            "A_cluster": mp.nstr(a_cluster, 30),
        },
        "galaxy": {
            "newton": newton,
            "monotone_c1": {
                "pass": c1_monotone,
                "min_pair_margin": _text(c1_min_margin),
                "probes": len(pack["monotone_c1_y"]),
            },
            "monotone_disk_rar": {
                "pass": not violations,
                "min_pair_margin": _text(rar_min_margin),
                "pooled_points": len(pooled_geff),
                "violations": violations,
                "per_disk_pass": per_disk_monotone,
            },
            "flat_outer_curves": {
                "pass": flat_pass,
                "threshold": CANDIDATE_CONFIG["fp64_thresholds"]["flatness"],
                "per_disk": per_disk,
            },
            "btfr": {
                "pass": btfr_pass,
                "slope": slope_text,
                "error": _text(slope_error) if slope_error != mp.inf else "inf",
                "tolerance": CANDIDATE_CONFIG["fp64_thresholds"]["btfr_slope"],
            },
            "pass": galaxy_pass,
        },
        "lensing": {
            "pass": lensing_pass,
            "worst_flatness": _text(worst_flatness),
            "worst_consistency": _text(worst_consistency),
            "thresholds": {
                "flatness": CANDIDATE_CONFIG["fp64_thresholds"]["lensing_flatness"],
                "consistency": CANDIDATE_CONFIG["fp64_thresholds"]["lensing_consistency"],
            },
            "per_mass": lensing_per_mass,
        },
        "cluster": {
            "pass": cluster_pass,
            "max_deviation": _text(cluster_max),
            "closest_probe_deviation": _text(min(deviations)),
            "ratio_by_probe": [_text(value) for value in ratios],
            "shortfall_by_probe": [_text(value) for value in shortfalls],
            "tolerance": CANDIDATE_CONFIG["fp64_thresholds"]["cluster_consistency"],
            "probe_radii": GATE_CONFIG["cluster"]["probe_radii"],
        },
        "solar_proxy": {
            "y": CANDIDATE_CONFIG["solar_probe_y"],
            "sigma_minus_1": _text(solar_error),
            "report_only": True,
        },
        "verdict": {
            "galaxy_pass": galaxy_pass,
            "lensing_pass": lensing_pass,
            "cluster_pass": cluster_pass,
            "all_pass": bool(galaxy_pass and lensing_pass and cluster_pass),
        },
    }


# ---------------------------------------------------------------------------
# Vectorized fp64 evaluation (shared numpy/cupy code path)
# ---------------------------------------------------------------------------


def build_float_pack(pack: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Float64 projection of the exact control pack for the batched kernels."""

    pack = pack if pack is not None else build_control_pack()
    lensing_by_mass: dict[int, list[list[tuple[float, float, float]]]] = {}
    for integral in pack["lensing"]:
        nodes = [(float(y), float(w), float(win)) for y, w, win in integral["nodes"]]
        lensing_by_mass.setdefault(integral["mass_index"], []).append(nodes)
    return {
        "pooled": [(float(g), float(w)) for g, _, w, _ in pack["pooled"]],
        "disks": [
            [(float(g), float(r), float(w)) for g, r, w in disk["outer"]]
            for disk in pack["disks"]
        ],
        "disk_offsets": [
            sum(1 for entry in pack["pooled"] if entry[3] == index)
            for index in range(len(pack["disks"]))
        ],
        "pooled_disk_index": [entry[3] for entry in pack["pooled"]],
        "log_mass_span": float(mp.log(pack["disks"][-1]["mass"] / pack["disks"][0]["mass"])),
        "monotone_c1_y": [float(y) for y in pack["monotone_c1_y"]],
        "lensing_by_mass": [lensing_by_mass[key] for key in sorted(lensing_by_mass)],
        "cluster": [(float(g), float(d)) for g, d in pack["cluster"]],
        "thresholds": {
            key: float(mp.mpf(text))
            for key, text in CANDIDATE_CONFIG["fp64_thresholds"].items()
        },
        "newton_probe_y": [float(v) for v in CANDIDATE_CONFIG["newton_probe_y"]],
        "solar_probe_y": float(CANDIDATE_CONFIG["solar_probe_y"]),
    }


def axis_float_values() -> dict[str, np.ndarray]:
    return {
        name: np.array([float(text) for text in SCAN_CONFIG["axes"][name]], dtype=np.float64)
        for name in SCAN_CONFIG["axis_order"]
    }


def encode_scan_indices(indices: Sequence[int]) -> int:
    if len(indices) != len(_AXIS_SIZES):
        raise SigmaGravityGateError("need exactly one index per axis")
    ordinal = 0
    for index, size in zip(indices, _AXIS_SIZES, strict=True):
        if not 0 <= index < size:
            raise SigmaGravityGateError(f"axis index out of range: {index}")
        ordinal = ordinal * size + index
    return ordinal


def decode_scan_ordinal(ordinal: int) -> dict[str, Any]:
    if not 0 <= ordinal < TOTAL_SCAN_CANDIDATES:
        raise SigmaGravityGateError(f"scan ordinal out of range: {ordinal}")
    indices = []
    value = ordinal
    for size in reversed(_AXIS_SIZES):
        indices.append(value % size)
        value //= size
    indices.reverse()
    values = {
        name: SCAN_CONFIG["axes"][name][index]
        for name, index in zip(SCAN_CONFIG["axis_order"], indices, strict=True)
    }
    return {"indices": indices, "values": values}


def _decode_indices_batch(xp: Any, ordinals: Any) -> list[Any]:
    value = ordinals.astype(xp.int64)
    reversed_indices = []
    for size in reversed(_AXIS_SIZES):
        reversed_indices.append(value % size)
        value //= size
    return list(reversed(reversed_indices))


def _params_from_indices(xp: Any, indices: Sequence[Any], dtype: Any) -> dict[str, Any]:
    axes = axis_float_values()
    tables = {
        name: xp.asarray(axes[name], dtype=dtype) for name in SCAN_CONFIG["axis_order"]
    }
    named = dict(zip(SCAN_CONFIG["axis_order"], indices, strict=True))
    params = {
        "a0": tables["A0"][named["A0"]],
        "n": tables["n"][named["n"]],
        "gd": tables["g_dagger"][named["g_dagger"]],
        "c_cluster": tables["C_cluster"][named["C_cluster"]],
        "p": tables["p"][named["p"]],
        "q": tables["q"][named["q"]],
    }
    published = SCAN_CONFIG["published_on_grid_axes"]
    at_published = xp.zeros(indices[0].shape[0], dtype=xp.int32)
    for name in SCAN_CONFIG["axis_order"]:
        target = published[name]
        if target is None:
            continue
        at_published = at_published + (
            named[name] == SCAN_CONFIG["axes"][name].index(target)
        ).astype(xp.int32)
    params["axes_at_published"] = at_published
    return params


def evaluate_params_batch(
    xp: Any, params: Mapping[str, Any], floats: Mapping[str, Any], *, dtype: Any
) -> dict[str, Any]:
    """All gates for one batch of explicit parameter arrays.  Returns arrays."""

    a0, gd = params["a0"], params["gd"]
    p, q = params["p"], params["q"]
    c_cluster = params["c_cluster"]
    a_cluster = a0 * dtype(1500.0) ** params["n"]
    count = a0.shape[0]
    thresholds = floats["thresholds"]

    def h_of(y: float) -> Any:
        return (gd / dtype(y)) ** p * (gd / (gd + dtype(y))) ** q

    defined = xp.ones(count, dtype=bool)

    def sane(value: Any) -> Any:
        nonlocal defined
        defined = defined & xp.isfinite(value)
        return value

    near_y, far_y = floats["newton_probe_y"]
    newton_near_error = sane(a0 * h_of(near_y))
    newton_far_error = sane(a0 * h_of(far_y))
    solar_error = sane(a0 * h_of(floats["solar_probe_y"]))
    newton_pass = (newton_near_error <= dtype(thresholds["newton_near"])) & (
        newton_far_error <= dtype(thresholds["newton_far"])
    )

    monotone_c1 = xp.ones(count, dtype=bool)
    previous = None
    for y in floats["monotone_c1_y"]:
        geff = dtype(y) * (1 + a0 * h_of(y))
        sane(geff)
        if previous is not None:
            monotone_c1 &= geff > previous
        previous = geff

    rar_min_margin = xp.full(count, xp.inf, dtype=dtype)
    per_disk_ok = xp.ones(count, dtype=bool)
    previous = None
    previous_disk: dict[int, Any] = {}
    for (gbar, window), disk_index in zip(
        floats["pooled"], floats["pooled_disk_index"], strict=True
    ):
        geff = dtype(gbar) * (1 + a0 * dtype(window) * h_of(gbar))
        sane(geff)
        if previous is not None:
            rar_min_margin = xp.minimum(rar_min_margin, geff / previous - 1)
        if disk_index in previous_disk:
            per_disk_ok &= geff > previous_disk[disk_index]
        previous_disk[disk_index] = geff
        previous = geff
    disk_rar_pass = rar_min_margin > 0

    flat_worst = xp.zeros(count, dtype=dtype)
    vflat = []
    for outer in floats["disks"]:
        vmax = None
        vmin = None
        vsum = xp.zeros(count, dtype=dtype)
        for gbar, radius, window in outer:
            vsq = dtype(gbar) * (1 + a0 * dtype(window) * h_of(gbar)) * dtype(radius)
            speed = xp.sqrt(xp.maximum(vsq, dtype(0)))
            sane(speed)
            vmax = speed if vmax is None else xp.maximum(vmax, speed)
            vmin = speed if vmin is None else xp.minimum(vmin, speed)
            vsum = vsum + speed
        mean = vsum / dtype(len(outer))
        safe_mean = xp.where(mean > 0, mean, dtype(1))
        flat_worst = xp.maximum(flat_worst, (vmax - vmin) / safe_mean)
        vflat.append(mean)
    flat_pass = flat_worst <= dtype(thresholds["flatness"])

    ratio = xp.log(
        xp.maximum(vflat[-1], dtype(1e-300)) / xp.maximum(vflat[0], dtype(1e-300))
    )
    slope = dtype(floats["log_mass_span"]) / xp.where(ratio > 0, ratio, dtype(1))
    btfr_error = xp.where(ratio > 0, xp.abs(slope - 4), dtype(float("inf")))
    btfr_pass = (ratio > 0) & (btfr_error <= dtype(thresholds["btfr_slope"]))

    lensing_flat_worst = xp.zeros(count, dtype=dtype)
    lensing_cons_worst = xp.zeros(count, dtype=dtype)
    lensing_valid = xp.ones(count, dtype=bool)
    for mass_index, integrals in enumerate(floats["lensing_by_mass"]):
        expected = dtype(2.0 * float(np.pi)) * vflat[mass_index] * vflat[mass_index]
        lensing_valid &= expected > 0
        safe_expected = xp.where(expected > 0, expected, dtype(1))
        alpha_max = None
        alpha_min = None
        alpha_sum = xp.zeros(count, dtype=dtype)
        for nodes in integrals:
            alpha = xp.zeros(count, dtype=dtype)
            for y, weight, window in nodes:
                alpha = alpha + dtype(weight) * (1 + a0 * dtype(window) * h_of(y))
            sane(alpha)
            lensing_cons_worst = xp.maximum(
                lensing_cons_worst, xp.abs(alpha / safe_expected - 1)
            )
            alpha_max = alpha if alpha_max is None else xp.maximum(alpha_max, alpha)
            alpha_min = alpha if alpha_min is None else xp.minimum(alpha_min, alpha)
            alpha_sum = alpha_sum + alpha
        mean_alpha = alpha_sum / dtype(len(integrals))
        lensing_valid &= mean_alpha > 0
        safe_mean = xp.where(mean_alpha > 0, mean_alpha, dtype(1))
        lensing_flat_worst = xp.maximum(lensing_flat_worst, (alpha_max - alpha_min) / safe_mean)
    lensing_pass = (
        lensing_valid
        & (lensing_flat_worst <= dtype(thresholds["lensing_flatness"]))
        & (lensing_cons_worst <= dtype(thresholds["lensing_consistency"]))
    )

    cluster_dev = xp.zeros(count, dtype=dtype)
    for gbar, gdyn in floats["cluster"]:
        boost = 1 + a_cluster * c_cluster * h_of(gbar)
        value = dtype(gbar) * boost / dtype(gdyn)
        sane(value)
        cluster_dev = xp.maximum(cluster_dev, xp.abs(value - 1))
    cluster_pass = cluster_dev <= dtype(thresholds["cluster_consistency"])

    galaxy_pass = defined & newton_pass & monotone_c1 & disk_rar_pass & flat_pass & btfr_pass
    return {
        "defined": defined,
        "newton_pass": defined & newton_pass,
        "monotone_c1_pass": defined & monotone_c1,
        "disk_rar_pass": defined & disk_rar_pass,
        "disk_rar_per_disk_pass": defined & per_disk_ok,
        "flat_pass": defined & flat_pass,
        "btfr_pass": defined & btfr_pass,
        "galaxy_pass": galaxy_pass,
        "lensing_pass": defined & lensing_pass,
        "cluster_pass": defined & cluster_pass,
        "all_pass": galaxy_pass & lensing_pass & cluster_pass,
        "newton_near_error": newton_near_error,
        "newton_far_error": newton_far_error,
        "solar_error": solar_error,
        "rar_min_margin": rar_min_margin,
        "flat_worst_spread": flat_worst,
        "btfr_error": btfr_error,
        "lensing_worst_flatness": lensing_flat_worst,
        "lensing_worst_consistency": lensing_cons_worst,
        "cluster_max_deviation": cluster_dev,
    }


def evaluate_batch(
    xp: Any, ordinals: Any, floats: Mapping[str, Any], *, dtype: Any
) -> dict[str, Any]:
    """Decode scan ordinals and run every gate; adds the published-distance metric."""

    indices = _decode_indices_batch(xp, ordinals)
    params = _params_from_indices(xp, indices, dtype)
    result = evaluate_params_batch(xp, params, floats, dtype=dtype)
    result["axes_at_published"] = params["axes_at_published"]
    return result


def published_params_row() -> dict[str, np.ndarray]:
    """The exact published point as a one-row fp64 parameter batch."""

    exact = published_exact_parameters()
    return {
        "a0": np.array([float(exact["A0"])]),
        "n": np.array([float(exact["n"])]),
        "gd": np.array([float(exact["g_dagger"])]),
        "c_cluster": np.array([float(exact["C_cluster"])]),
        "p": np.array([float(exact["p"])]),
        "q": np.array([float(exact["q"])]),
    }


# ---------------------------------------------------------------------------
# Stage 1: the gauntlet receipt
# ---------------------------------------------------------------------------


def _assert_gauntlet_controls(newton_control: Mapping[str, Any]) -> None:
    """A0 = 0 is Newtonian gravity: it must fail exactly where Newton fails."""

    galaxy = newton_control["galaxy"]
    if galaxy["flat_outer_curves"]["pass"] or galaxy["btfr"]["pass"]:
        raise SigmaGravityGateError("Newton control unexpectedly passed a galaxy gate")
    if newton_control["cluster"]["pass"]:
        raise SigmaGravityGateError("Newton control unexpectedly passed the cluster gate")
    if not galaxy["newton"]["near"]["pass"] or not galaxy["newton"]["far"]["pass"]:
        raise SigmaGravityGateError("Newton control failed Newtonian recovery")
    if not galaxy["monotone_disk_rar"]["pass"]:
        raise SigmaGravityGateError("Newton control failed RAR monotonicity")


def run_gauntlet() -> dict[str, Any]:
    """Evaluate the published point through every gate and seal the receipt."""

    started = time.perf_counter()
    pack = build_control_pack()
    published = evaluate_candidate_exact(pack=pack)
    newton_control = evaluate_candidate_exact({"A0": "0"}, pack=pack)
    _assert_gauntlet_controls(newton_control)
    verdict = published["verdict"]
    labels = ", ".join(
        f"{name} {'PASS' if verdict[f'{name}_pass'] else 'FAIL'}"
        for name in ("galaxy", "lensing", "cluster")
    )
    elapsed = time.perf_counter() - started
    body: dict[str, Any] = {
        "assumptions": ASSUMPTIONS,
        "candidate_config": CANDIDATE_CONFIG,
        "candidate_config_sha256": canonical_sha256(CANDIDATE_CONFIG),
        "claims": CLAIMS_GAUNTLET,
        "controls": {"newton_a0_zero": newton_control},
        "decision": (
            f"GAUNTLET-EVALUATED: {labels}; all_pass={verdict['all_pass']}; "
            f"solar proxy |Sigma-1| = {published['solar_proxy']['sigma_minus_1']}"
        ),
        "elapsed_seconds": format(elapsed, ".3f"),
        "frozen_grids": {
            "disk_grid": _stringified_disk_grid(pack["disk_grid"]),
            "lensing_grid": pack["lensing_grid"],
            "cluster_table": recompute_cluster_table(),
        },
        "gate_config_sha256": canonical_sha256(GATE_CONFIG),
        "published_point": published,
        "schema_version": GAUNTLET_SCHEMA,
        "scope": (
            "Exact 50-digit evaluation of the published Sigma-Gravity formula through "
            "the frozen synthetic galaxy, lensing, and cluster gates of the existing "
            "GPU screens, under a declared unit mapping and coherence reading. The "
            "controls are analytic; nothing here opens observational data, validates a "
            "theory, or refutes the paper's fits to real SPARC/Fox data."
        ),
        "screen_config_sha256": canonical_sha256(SCREEN_CONFIG),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_gauntlet_receipt(value: Mapping[str, Any]) -> None:
    """Seal, binding, frozen-grid, and full exact-replay checks; fail closed."""

    if value.get("schema_version") != GAUNTLET_SCHEMA:
        raise SigmaGravityGateError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise SigmaGravityGateError("receipt seal changed")
    if value.get("claims") != CLAIMS_GAUNTLET:
        raise SigmaGravityGateError("claims block changed")
    if value.get("candidate_config_sha256") != canonical_sha256(CANDIDATE_CONFIG):
        raise SigmaGravityGateError("candidate config binding changed")
    if value.get("candidate_config") != CANDIDATE_CONFIG:
        raise SigmaGravityGateError("candidate config does not match this module")
    if value.get("screen_config_sha256") != canonical_sha256(SCREEN_CONFIG):
        raise SigmaGravityGateError("screen config binding changed")
    if value.get("gate_config_sha256") != canonical_sha256(GATE_CONFIG):
        raise SigmaGravityGateError("gate config binding changed")
    pack = build_control_pack()
    frozen = value.get("frozen_grids", {})
    if canonical_sha256(frozen.get("disk_grid")) != canonical_sha256(
        _stringified_disk_grid(pack["disk_grid"])
    ):
        raise SigmaGravityGateError("frozen disk grid does not replay")
    if canonical_sha256(frozen.get("lensing_grid")) != canonical_sha256(pack["lensing_grid"]):
        raise SigmaGravityGateError("frozen lensing grid does not replay")
    if canonical_sha256(frozen.get("cluster_table")) != canonical_sha256(
        recompute_cluster_table()
    ):
        raise SigmaGravityGateError("frozen cluster table does not replay")
    replay = evaluate_candidate_exact(pack=pack)
    if canonical_sha256(replay) != canonical_sha256(value.get("published_point")):
        raise SigmaGravityGateError("published-point exact replay changed")
    control = evaluate_candidate_exact({"A0": "0"}, pack=pack)
    _assert_gauntlet_controls(control)
    if canonical_sha256(control) != canonical_sha256(
        value.get("controls", {}).get("newton_a0_zero")
    ):
        raise SigmaGravityGateError("Newton control exact replay changed")


# ---------------------------------------------------------------------------
# Stage 2: the neighborhood scan receipt
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


def _pareto_front(axes: np.ndarray, prefilter: int, block: int = 512) -> np.ndarray:
    """Row indices of the non-dominated set within an axis-wise prefilter (minimize)."""

    if axes.size == 0:
        return np.empty(0, dtype=np.int64)
    heads = []
    for primary in range(axes.shape[1]):
        others = [axes[:, k] for k in range(axes.shape[1]) if k != primary]
        heads.append(np.lexsort((*reversed(others), axes[:, primary]))[:prefilter])
    chosen = np.unique(np.concatenate(heads))
    sub = axes[chosen]
    dominated = np.zeros(chosen.size, dtype=bool)
    for start in range(0, chosen.size, block):
        stop = min(start + block, chosen.size)
        piece = sub[start:stop]
        not_worse = (sub[:, None, :] <= piece[None, :, :]).all(axis=2)
        strictly = (sub[:, None, :] < piece[None, :, :]).any(axis=2)
        dominated[start:stop] = (not_worse & strictly).any(axis=0)
    return chosen[~dominated]


def _metric_text(value: float) -> str:
    return format(float(value), ".9e")


def _candidate_entry(
    ordinal: int, row: int, decisions: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    decoded = decode_scan_ordinal(ordinal)
    return {
        "ordinal": ordinal,
        "parameters": decoded["values"],
        "axes_at_published": int(decisions["axes_at_published"][row]),
        "gates": {
            name: bool(decisions[f"{name}_pass"][row])
            for name in (
                "newton",
                "monotone_c1",
                "disk_rar",
                "flat",
                "btfr",
                "galaxy",
                "lensing",
                "cluster",
                "all",
            )
        },
        "metrics": {
            name: _metric_text(decisions[name][row])
            for name in (
                "newton_far_error",
                "rar_min_margin",
                "flat_worst_spread",
                "btfr_error",
                "lensing_worst_flatness",
                "lensing_worst_consistency",
                "cluster_max_deviation",
            )
        },
    }


def _exact_verify_scan_candidate(
    ordinal: int, pack: Mapping[str, Any]
) -> dict[str, Any]:
    decoded = decode_scan_ordinal(ordinal)
    values = decoded["values"]
    verdict = evaluate_candidate_exact(
        {
            "A0": values["A0"],
            "n": values["n"],
            "g_dagger": values["g_dagger"],
            "C_cluster": values["C_cluster"],
            "p": values["p"],
            "q": values["q"],
        },
        pack=pack,
    )
    return {
        "ordinal": ordinal,
        "parameters": values,
        "verdict": verdict["verdict"],
        "cluster_max_deviation": verdict["cluster"]["max_deviation"],
        "lensing_worst_consistency": verdict["lensing"]["worst_consistency"],
    }


def scan_neighborhood(
    *, use_gpu: bool = True, limit: int | None = None, chunk_size: int = 1 << 19
) -> dict[str, Any]:
    """Run every gate over the ordinal neighborhood grid and seal a receipt."""

    total = TOTAL_SCAN_CANDIDATES
    if int(np.prod(_AXIS_SIZES)) != total or SCAN_CONFIG["total_candidates"] != total:
        raise SigmaGravityGateError("axis sizes drifted from the frozen total")
    if limit is not None and not 1 <= limit <= total:
        raise SigmaGravityGateError(f"limit outside the family: {limit}")
    processed = total if limit is None else limit
    xp, device, gpu = _array_module(use_gpu)
    pack = build_control_pack()
    floats = build_float_pack(pack)

    started = time.perf_counter()
    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, processed, chunk_size):
        stop = min(start + chunk_size, processed)
        ordinals = xp.arange(start, stop, dtype=xp.int64)
        chunk = evaluate_batch(xp, ordinals, floats, dtype=xp.float64)
        parts.append(
            {key: (item.get() if gpu else np.asarray(item)) for key, item in chunk.items()}
        )
    decisions = {
        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    elapsed = time.perf_counter() - started

    gate_counts = {
        f"{name}_pass": int(decisions[f"{name}_pass"].sum())
        for name in (
            "newton",
            "monotone_c1",
            "disk_rar",
            "disk_rar_per_disk",
            "flat",
            "btfr",
            "galaxy",
            "lensing",
            "cluster",
            "all",
        )
    }
    counts = {
        "total_candidates": total,
        "processed": processed,
        "defined": int(decisions["defined"].sum()),
        **gate_counts,
    }

    distance = (6 - decisions["axes_at_published"]).astype(np.float64)
    axes = np.stack(
        [
            decisions["cluster_max_deviation"],
            decisions["lensing_worst_consistency"],
            decisions["newton_far_error"],
            distance,
        ],
        axis=1,
    )
    front_rows = _pareto_front(axes, SCAN_CONFIG["pareto"]["prefilter_per_axis"])
    order = np.lexsort(
        (
            front_rows,
            axes[front_rows, 3],
            axes[front_rows, 2],
            axes[front_rows, 1],
            axes[front_rows, 0],
        )
    )
    front_rows = front_rows[order]
    front_total = int(front_rows.size)
    front_rows = front_rows[: SCAN_CONFIG["pareto"]["reported_cap"]]
    pareto_front = [
        {
            **_candidate_entry(int(row), int(row), decisions),
            "pareto_axes": {
                "cluster_max_deviation": _metric_text(axes[row, 0]),
                "lensing_worst_consistency": _metric_text(axes[row, 1]),
                "newton_far_error": _metric_text(axes[row, 2]),
                "published_distance": int(axes[row, 3]),
            },
        }
        for row in front_rows
    ]

    # Best all-gate passers, or the closest candidates when the verdict is negative.
    passer_rows = np.flatnonzero(decisions["all_pass"])
    passer_order = np.lexsort(
        (
            passer_rows,
            decisions["lensing_worst_consistency"][passer_rows],
            decisions["cluster_max_deviation"][passer_rows],
        )
    )
    best_passers = [
        _candidate_entry(int(row), int(row), decisions)
        for row in passer_rows[passer_order][: SCAN_CONFIG["closest_report"]["count"]]
    ]
    failed_top = (
        (~decisions["galaxy_pass"]).astype(np.int64)
        + (~decisions["lensing_pass"]).astype(np.int64)
        + (~decisions["cluster_pass"]).astype(np.int64)
    )
    closest_order = np.lexsort(
        (
            np.arange(processed),
            decisions["cluster_max_deviation"],
            failed_top,
        )
    )
    closest = [
        {
            **_candidate_entry(int(row), int(row), decisions),
            "failed_top_gates": int(failed_top[row]),
        }
        for row in closest_order[: SCAN_CONFIG["closest_report"]["count"]]
    ]

    # Published point: exact params through the same fp64 kernels, checked against the
    # 50-digit gauntlet evaluator, and its dominance status against the whole grid.
    published_exact = evaluate_candidate_exact(pack=pack)
    published_fp64 = evaluate_params_batch(np, published_params_row(), floats, dtype=np.float64)
    fp64_verdict = {
        f"{name}_pass": bool(published_fp64[f"{name}_pass"][0])
        for name in ("galaxy", "lensing", "cluster", "all")
    }
    agreement = fp64_verdict == published_exact["verdict"]
    if not agreement:
        raise SigmaGravityGateError("published point: fp64 and 50-digit verdicts disagree")
    published_axes = np.array(
        [
            float(published_fp64["cluster_max_deviation"][0]),
            float(published_fp64["lensing_worst_consistency"][0]),
            float(published_fp64["newton_far_error"][0]),
            0.0,
        ]
    )
    not_worse_full = (axes <= published_axes).all(axis=1)
    strictly_full = (axes < published_axes).any(axis=1)
    dominators_full = np.flatnonzero(not_worse_full & strictly_full)
    physics = axes[:, :3]
    published_physics = published_axes[:3]
    not_worse_physics = (physics <= published_physics).all(axis=1)
    strictly_physics = (physics < published_physics).any(axis=1)
    dominators_physics = np.flatnonzero(not_worse_physics & strictly_physics)
    published_point = {
        "parameters": published_exact["parameters"],
        "on_scan_grid": False,
        "off_grid_axes": ["A0", "n"],
        "passes_all_gates": published_exact["verdict"]["all_pass"],
        "verdict": published_exact["verdict"],
        "fp64_verdict": fp64_verdict,
        "fp64_metrics": {
            "cluster_max_deviation": _metric_text(published_axes[0]),
            "lensing_worst_consistency": _metric_text(published_axes[1]),
            "newton_far_error": _metric_text(published_axes[2]),
        },
        "exact_agrees_with_fp64": agreement,
        "pareto": {
            "with_published_distance_axis": {
                "dominated_by_count": int(dominators_full.size),
                "on_front": bool(dominators_full.size == 0),
                "note": (
                    "published distance is 0 by definition and grid distance is >= 2 "
                    "(A0 and n are off-grid), so this membership is vacuous"
                ),
            },
            "physics_axes_only": {
                "dominated_by_count": int(dominators_physics.size),
                "on_front": bool(dominators_physics.size == 0),
                "example_dominators": [int(v) for v in dominators_physics[:3]],
            },
        },
    }

    # 50-digit re-verification of every reported candidate, within a recorded budget.
    reported: list[int] = []
    for entry in (*closest, *best_passers, *pareto_front):
        if entry["ordinal"] not in reported:
            reported.append(entry["ordinal"])
    budget = SCAN_CONFIG["max_exact_verifications"]
    exact_truncated = max(0, len(reported) - budget)
    exact_verification = []
    for ordinal in reported[:budget]:
        verdict = _exact_verify_scan_candidate(ordinal, pack)
        row = ordinal
        verdict["exact_confirmed"] = all(
            verdict["verdict"][f"{name}_pass"] == bool(decisions[f"{name}_pass"][row])
            for name in ("galaxy", "lensing", "cluster", "all")
        )
        exact_verification.append(verdict)
    if any(not entry["exact_confirmed"] for entry in exact_verification):
        raise SigmaGravityGateError("fp64 and 50-digit scan verdicts disagree")

    # CPU/GPU cross-check on a deterministic sample; the CPU decisions are also the
    # deterministic replay target for validation.
    rng = np.random.default_rng(SCAN_CONFIG["crosscheck_seed"])
    sample = np.sort(
        rng.choice(processed, size=min(SCAN_CONFIG["crosscheck_sample"], processed), replace=False)
    ).astype(np.int64)
    cpu_sample = evaluate_batch(np, sample, floats, dtype=np.float64)
    sample_digest = canonical_sha256(
        {
            "ordinals": [int(v) for v in sample],
            "galaxy_pass": [bool(v) for v in cpu_sample["galaxy_pass"]],
            "lensing_pass": [bool(v) for v in cpu_sample["lensing_pass"]],
            "cluster_pass": [bool(v) for v in cpu_sample["cluster_pass"]],
            "all_pass": [bool(v) for v in cpu_sample["all_pass"]],
        }
    )
    crosscheck: dict[str, Any] = {
        "performed": gpu,
        "sample": int(sample.size),
        "sample_decisions_sha256": sample_digest,
    }
    if gpu:
        import cupy as cp

        gpu_sample = evaluate_batch(cp, cp.asarray(sample), floats, dtype=cp.float64)
        for name in ("galaxy_pass", "lensing_pass", "cluster_pass", "all_pass"):
            crosscheck[f"{name}_disagreements"] = int(
                (gpu_sample[name].get() != cpu_sample[name]).sum()
            )

    if counts["all_pass"] == 0:
        best_cluster = _metric_text(decisions["cluster_max_deviation"].min())
        decision = (
            "SCANNED-SEALED-NEGATIVE: 0 of "
            f"{processed} neighborhood candidates pass all gates; disk-RAR "
            f"single-valuedness pass count {counts['disk_rar_pass']}, cluster pass "
            f"count {counts['cluster_pass']}; closest cluster approach deviation "
            f"{best_cluster} against tolerance "
            f"{CANDIDATE_CONFIG['fp64_thresholds']['cluster_consistency']}"
        )
    else:
        decision = f"SCANNED: {counts['all_pass']} of {processed} candidates pass all gates"

    body: dict[str, Any] = {
        "assumptions": ASSUMPTIONS,
        "best_all_gate_passers": best_passers,
        "candidate_config": CANDIDATE_CONFIG,
        "candidate_config_sha256": canonical_sha256(CANDIDATE_CONFIG),
        "claims": CLAIMS_SCAN,
        "closest_candidates": closest,
        "counts": counts,
        "crosscheck": crosscheck,
        "decision": decision,
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "exact_verification": exact_verification,
        "exact_verification_truncated": exact_truncated,
        "frozen_grid_sha256": frozen_grid_bindings(pack),
        "gate_config_sha256": canonical_sha256(GATE_CONFIG),
        "pareto_front": pareto_front,
        "pareto_front_total": front_total,
        "published_point": published_point,
        "scan_config": SCAN_CONFIG,
        "scan_config_sha256": canonical_sha256(SCAN_CONFIG),
        "schema_version": NEIGHBORHOOD_SCHEMA,
        "scope": (
            "fp64 GPU scan of 334,620 ordinal-indexed parameter combinations around "
            "the published Sigma-Gravity point through the frozen synthetic galaxy, "
            "lensing, and cluster gates, with CPU cross-check and 50-digit "
            "re-verification of every reported candidate. The controls are synthetic; "
            "varying a published formula's parameters against them is neither a "
            "calibration against real data nor a refutation of the paper's SPARC/Fox "
            "fits; a zero-survivor verdict is a sealed negative for this neighborhood "
            "only."
        ),
        "screen_config_sha256": canonical_sha256(SCREEN_CONFIG),
        "throughput_candidates_per_second": int(processed / elapsed) if elapsed > 0 else None,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_neighborhood_receipt(value: Mapping[str, Any]) -> None:
    """Seal, binding, published replay, and sample replay checks; fail closed."""

    if value.get("schema_version") != NEIGHBORHOOD_SCHEMA:
        raise SigmaGravityGateError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise SigmaGravityGateError("receipt seal changed")
    if value.get("claims") != CLAIMS_SCAN:
        raise SigmaGravityGateError("claims block changed")
    if value.get("scan_config_sha256") != canonical_sha256(SCAN_CONFIG):
        raise SigmaGravityGateError("scan config binding changed")
    if value.get("scan_config") != SCAN_CONFIG:
        raise SigmaGravityGateError("scan config does not match this module")
    if value.get("candidate_config_sha256") != canonical_sha256(CANDIDATE_CONFIG):
        raise SigmaGravityGateError("candidate config binding changed")
    if value.get("screen_config_sha256") != canonical_sha256(SCREEN_CONFIG):
        raise SigmaGravityGateError("screen config binding changed")
    if value.get("gate_config_sha256") != canonical_sha256(GATE_CONFIG):
        raise SigmaGravityGateError("gate config binding changed")
    pack = build_control_pack()
    if value.get("frozen_grid_sha256") != frozen_grid_bindings(pack):
        raise SigmaGravityGateError("frozen grid binding does not replay")
    published = evaluate_candidate_exact(pack=pack)
    recorded = value.get("published_point", {})
    if recorded.get("verdict") != published["verdict"]:
        raise SigmaGravityGateError("published-point verdict does not replay")
    if recorded.get("parameters") != published["parameters"]:
        raise SigmaGravityGateError("published-point parameters do not replay")
    floats = build_float_pack(pack)
    processed = value.get("counts", {}).get("processed")
    if not isinstance(processed, int) or processed < 1:
        raise SigmaGravityGateError("receipt processed count is malformed")
    rng = np.random.default_rng(SCAN_CONFIG["crosscheck_seed"])
    sample = np.sort(
        rng.choice(processed, size=min(SCAN_CONFIG["crosscheck_sample"], processed), replace=False)
    ).astype(np.int64)
    cpu_sample = evaluate_batch(np, sample, floats, dtype=np.float64)
    digest = canonical_sha256(
        {
            "ordinals": [int(v) for v in sample],
            "galaxy_pass": [bool(v) for v in cpu_sample["galaxy_pass"]],
            "lensing_pass": [bool(v) for v in cpu_sample["lensing_pass"]],
            "cluster_pass": [bool(v) for v in cpu_sample["cluster_pass"]],
            "all_pass": [bool(v) for v in cpu_sample["all_pass"]],
        }
    )
    if value.get("crosscheck", {}).get("sample_decisions_sha256") != digest:
        raise SigmaGravityGateError("crosscheck sample decisions do not replay")
    for entry in value.get("exact_verification", []):
        if not entry.get("exact_confirmed", False):
            raise SigmaGravityGateError("receipt contains an unconfirmed exact verification")
        replay = _exact_verify_scan_candidate(entry["ordinal"], pack)
        if replay["verdict"] != entry.get("verdict"):
            raise SigmaGravityGateError(
                f"exact replay changed for scan ordinal {entry['ordinal']}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise SigmaGravityGateError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sigma-Gravity published-point gauntlet and neighborhood scan."
    )
    parser.add_argument("--stage", choices=("gauntlet", "scan"), required=True)
    parser.add_argument("--output")
    parser.add_argument("--cpu", action="store_true", help="force the numpy path")
    parser.add_argument("--limit", type=int, default=None, help="scan only: ordinal cap")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        if not args.output:
            raise SigmaGravityGateError("--validate-checked requires --output")
        value = json.loads(Path(args.output).read_text(encoding="utf-8"))
        if args.stage == "gauntlet":
            validate_gauntlet_receipt(value)
        else:
            validate_neighborhood_receipt(value)
        return 0
    if args.stage == "gauntlet":
        result = run_gauntlet()
        summary = {
            "decision": result["decision"],
            "verdict": result["published_point"]["verdict"],
            "cluster_max_deviation": result["published_point"]["cluster"]["max_deviation"],
            "lensing_worst_consistency": result["published_point"]["lensing"][
                "worst_consistency"
            ],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    else:
        result = scan_neighborhood(use_gpu=not args.cpu, limit=args.limit)
        summary = {
            "decision": result["decision"],
            "counts": result["counts"],
            "published_passes_all": result["published_point"]["passes_all_gates"],
            "device": result["device"],
            "elapsed_seconds": result["elapsed_seconds"],
            "throughput_candidates_per_second": result["throughput_candidates_per_second"],
        }
    if args.output:
        _write_receipt(result, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
