"""Nonlocal-kernel gravity screen (v2): the grammar the two sealed negatives point at.

Two sealed negatives identified one exact failure mode.  The billion-candidate
pointwise screen (`runs/gpu-baryonic-screen/lensing-cluster-v1.json`) proved that no
universal ``nu(y)`` carries the hydrostatic cluster control (closest approach 0.170 vs
0.15), and the Sigma-Gravity neighborhood scan (`runs/sigma-gravity/neighborhood-v1.json`)
proved that amplitude rescaling does not rescue a pointwise law (closest 0.454): the
cluster's needed boost is a *radial shape* — the ratio ``g_dyn/g_bar`` runs 5.50 at
``r = rc/2`` down to 2.40 at ``8 rc`` while the local acceleration barely moves
(``y = 0.92-2.01``) and is not even monotone in radius.  No function of the local
acceleration, however rescaled, can produce two different boosts at the same ``y``.
A *kernel* can: this module screens laws whose boost is a declared convolution over the
source mass distribution,

    g_obs(x) = nu_loc(y) * g_N(x) + B(x),          y = g_N / a0,  a0 = 1
    B(x)     = Int rho_b(x') K(|x - x'|; theta) dV'
    K(s)     = (1/s^2) * [ w_Y * exp(-s/L1)
                           + w_P * (s/L2)^p * (1 + s/L2)^(-(p+t)) ]

with an ordinal-indexed integer grid over (nu_loc, w_Y, L1, w_P, L2, p, t):
3 * 49 * 12 * 49 * 12 * 16 * 8 = 132,765,696 candidates (declared; inside the
10^8-10^9 target band).  The Yukawa term is massive-scalar exchange (mediator mass
1/L1); the power term is a solar-safe rising tail: short range ~ s^(p-2) (suppressed
for p > 0), long range ~ s^(-2-t) (t = -1 is a MOND-like 1/s tail, t = 0 a long-range
rescaling of G, t > 0 a decaying correction).  The prior pointwise family is *embedded*
(w_Y = w_P = 0 leaves ``nu_loc(y) * g_N`` with nu_loc in {1, 1+u, sqrt(1+u^2)},
u = y^(-1/2)), so the old sealed negatives must be reproducible inside v2 — a
known-answer control, not an assumption.

**The enabling trick.**  For each synthetic control system the source mass distribution
is compiled once, at 50-digit mpmath, into frozen distance-distribution quadrature
nodes {(s_n, w_n)} per probe, using the swapped-order representation
``B(r) = Int K(s) * Phi_r(s) ds`` (spheres: closed-form shell-theorem Phi; razor-thin
disk, in-plane probe: Phi = s * A_r(s) with A_r an mp.quad angular integral), so *any*
kernel evaluates as ``B = sum_n w_n K(s_n)``.  Because K is linear in (w_Y, w_P), the
whole family reduces to two per-probe basis tables — one per L1 value, one per
(L2, p, t) combination — and the 1.3e8-ordinal sweep is table combination, not
quadrature.  Validation: the Newton kernel ``K = 1/s^2`` must reproduce the frozen
``g_bar`` grids of the existing screens to 1e-10 relative at every probe; that single
test validates all geometry.

**Controls (reused exactly, never reinvented).**  Galaxy: the three frozen Freeman
disks of the billion screen (masses 1/250, 8/125, 128/125, Rd = 1) probed at the frozen
outer radii {8, 10, 12, 16, 20}; flat outer curves and the baryonic Tully-Fisher slope
at the existing thresholds.  Lensing: the existing Hernquist path-integral grid of the
P1 gate (33 Simpson nodes per (mass, b), factor-2 Born prescription), with ``g_obs``
now including the kernel boost of the same Hernquist sphere (truncated at 1024 code
units; declared); flatness and dynamics-lensing consistency at the existing thresholds.
Cluster: the existing frozen isothermal beta-model probe table and exact hydrostatic
``g_dyn``; gas truncated at 32 rc (declared); the decisive 5-probe criterion
``|g_obs/g_dyn - 1| <= 0.15`` that no candidate has ever passed.  Solar/Newton: a
declared unit point source probed at y = 1e4 and 1e6 (existing thresholds), y = 1e8
report-only, plus a Yukawa-safety probe ``|B/g_N| < 1e-8`` at s = 1e-5 so long-range
kernels that contaminate short scales die regardless of their galaxy behavior.

One declared limitation: the synthetic controls share one code-unit length scale
(disk Rd = cluster rc = Hernquist a = 1), so a kernel scale L means the same code
length in every system, while real galaxies and clusters are ~50x apart; survivors and
negatives are statements about these controls, not about nature.

Three-layer honesty, matching the existing screens: an fp32 sweep with slack
thresholds (the Yukawa-safety margin is always evaluated in fp64), an fp64 strict
recheck of survivors, 50-digit re-verification of every reported candidate, known-
answer controls that abort the run if broken, a CPU/GPU decision cross-check, and a
sealed receipt with no floating-point values.  Survivors, if any, are classified into
grid-neighbor equivalence families and emitted as typed ``covariant_lift_candidate``
blocks (field-theory ansatz strings with ``first_principles_derivation_pending``) for
the existing covariant machinery; a zero-survivor verdict is a sealed negative with
per-gate closest-approach margins and a named next structural direction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from .gpu_baryonic_interpolation_screen import SCREEN_CONFIG, build_probe_grid
from .gpu_baryonic_lensing_cluster_screen import (
    GATE_CONFIG,
    _fraction,
    build_lensing_grid,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-gpu-baryonic-kernel-screen-result-1.0"
PRECOMPUTE_SCHEMA = "invariant-gpu-baryonic-kernel-screen-precompute-1.0"

#: Ordinal axes, most-significant first.  Every value is an exact string; the float
#: grids used on GPU are exact binary representations of these rationals.
AXIS_ORDER = ("local", "w_yukawa", "L1", "w_power", "L2", "p", "t")
AXES: dict[str, tuple[str, ...]] = {
    "local": ("identity", "one_plus_u", "sqrt_one_plus_u_squared"),
    "w_yukawa": tuple(str(Fraction(k, 8)) for k in range(49)),
    "L1": ("1/2", "3/4", "1", "3/2", "2", "3", "4", "6", "8", "12", "16", "24"),
    "w_power": tuple(str(Fraction(k, 8)) for k in range(49)),
    "L2": ("1/2", "3/4", "1", "3/2", "2", "3", "4", "6", "8", "12", "16", "24"),
    "p": tuple(str(Fraction(k, 4)) for k in range(1, 17)),
    "t": ("-1", "-1/2", "0", "1/2", "1", "3/2", "2", "3"),
}
AXIS_SIZES = tuple(len(AXES[name]) for name in AXIS_ORDER)
FAMILY_SIZE = math.prod(AXIS_SIZES)  # 3*49*12*49*12*16*8 = 132,765,696

SYSTEM_CAPS = {
    "min_batch_size": 1 << 10,
    "max_batch_size": 1 << 23,
    "max_pareto_reported": 64,
    "max_exact_verifications": 24,
    "max_families_reported": 16,
    "family_analysis_cap": 100_000,
}

#: Frozen screen configuration.  Changing any value changes the claim and the receipt
#: hash.  Thresholds are the existing screens' tiers verbatim, plus the Yukawa-safety
#: probe; geometry declarations are the truncations and quadrature layout.
KERNEL_CONFIG: dict[str, Any] = {
    "a0": 1,
    "mpmath_dps": 50,
    "law": (
        "g_obs = nu_loc(g_N/a0) * g_N + B;  B(x) = Int rho_b(x') K(|x-x'|) dV';  "
        "K(s) = (1/s^2) * [w_Y*exp(-s/L1) + w_P*(s/L2)^p*(1+s/L2)^(-(p+t))]"
    ),
    "axis_order": list(AXIS_ORDER),
    "axes": {name: list(values) for name, values in AXES.items()},
    "family_size": FAMILY_SIZE,
    "geometry": {
        "disk": {
            "profile": "freeman-exponential-disk, razor-thin, in-plane probes",
            "scale_length": 1,
            "truncation_radius": 60,
            "probe_radii": [8, 10, 12, 16, 20],
            "representation": (
                "swapped-order distance nodes: B(r) = Int K(s) s A_r(s) ds with "
                "A_r(s) = Int (-cos psi) Sigma(rho(s,psi)) dpsi by 50-dps mp.quad; "
                "Gauss-Legendre panels with geometric side ladders in |s-r|"
            ),
        },
        "hernquist": {
            "profile": "hernquist-sphere (the existing P1 lensing deflectors)",
            "scale": 1,
            "truncation_radius": 1024,
            "probe_radii": "the 165 frozen lensing path-node radii b*cosh(t)",
            "representation": (
                "swapped-order distance nodes with the closed-form shell factor "
                "Phi_r(s) = [F(a2)-F(a1)]/(2 r^2), a1 = |r-s|, a2 = min(rt, r+s)"
            ),
        },
        "cluster": {
            "profile": "isothermal-beta-model gas (the existing P2 control)",
            "amplitude_4pi_rho0": 9,
            "core_radius": 1,
            "truncation_radius": 32,
            "probe_radii": ["1/2", "1", "2", "4", "8"],
            "representation": "swapped-order distance nodes with closed-form Phi_r(s)",
        },
        "ladder_ratio": 4,
        "head_halfwidth": "0.05*min(r, 1)",
        "panel_orders": {"head": 12, "side": 16},
        "newton_validation_max_relative_error": "1e-10",
        "kernel_convergence_budget": "1e-6",
    },
    "newton_control": {
        "source": "unit point mass; B/g_N = K(s)*s^2 exactly, mass-independent",
        "probe_y": [10000, 1000000],
        "report_probe_y": 100000000,
        "yukawa_safety": {"s": "1e-5", "max_abs_boost_ratio": "1e-8", "tier": "fp64 always"},
    },
    "fp32_thresholds": {
        "newton_near": "3e-2",
        "newton_far": "3e-3",
        "flatness": "8e-2",
        "btfr_slope": "45e-2",
        "lensing_flatness": "1e-1",
        "lensing_consistency": "18e-2",
        "cluster_consistency": "2e-1",
    },
    "fp64_thresholds": {
        "newton_near": "2e-2",
        "newton_far": "2e-3",
        "flatness": "6e-2",
        "btfr_slope": "30e-2",
        "lensing_flatness": "8e-2",
        "lensing_consistency": "15e-2",
        "cluster_consistency": "15e-2",
    },
    "crosscheck_sample": 2048,
    "crosscheck_seed": 20260816,
}

CLAIMS = {
    "cluster_negative_is_a_valid_deliverable": True,
    "corpus_absence_establishes_novelty": False,
    "embedded_pointwise_family_reproduces_prior_negative": True,
    "first_principles_derivation_claimed": False,
    "invisible_mass_used_as_target_or_rescue": False,
    "kernel_scales_share_code_units_across_systems": True,
    "lensing_prescription_is_an_assumption": True,
    "observational_data_opened": False,
    "per_object_free_parameters_expressible": False,
    "scalar_truth_or_probability_score": False,
    "sealed_validation_ladder_bypassed": False,
    "survivor_is_validated_theory": False,
    "synthetic_controls_only": True,
}


class KernelScreenError(ValueError):
    """Raised on malformed input, a broken known-answer control, or receipt tamper."""


def _dps() -> None:
    mp.mp.dps = KERNEL_CONFIG["mpmath_dps"]


def _text(value: Any) -> str:
    return format(float(value), ".9e")


# ---------------------------------------------------------------------------
# Ordinal codec
# ---------------------------------------------------------------------------


def decode_ordinal(ordinal: int) -> dict[str, Any]:
    """Ordinal -> {indices, values}; ``local`` is the most significant digit."""

    if not 0 <= ordinal < FAMILY_SIZE:
        raise KernelScreenError(f"ordinal out of range: {ordinal}")
    indices: list[int] = []
    value = ordinal
    for size in reversed(AXIS_SIZES):
        indices.append(value % size)
        value //= size
    indices.reverse()
    values = {
        name: AXES[name][index] for name, index in zip(AXIS_ORDER, indices, strict=True)
    }
    return {"indices": indices, "values": values}


def encode_indices(indices: Sequence[int]) -> int:
    """Inverse of `decode_ordinal`; used by tests and known-answer controls."""

    if len(indices) != len(AXIS_SIZES):
        raise KernelScreenError("need exactly one index per axis")
    ordinal = 0
    for index, size in zip(indices, AXIS_SIZES, strict=True):
        if not 0 <= index < size:
            raise KernelScreenError(f"axis index out of range: {index}")
        ordinal = ordinal * size + index
    return ordinal


def encode_named(**named: str) -> int:
    """Encode from axis-value strings (defaults: identity local, all-zero kernel)."""

    defaults = {
        "local": "identity",
        "w_yukawa": "0",
        "L1": "1",
        "w_power": "0",
        "L2": "1",
        "p": "1",
        "t": "0",
    }
    defaults.update(named)
    indices = []
    for name in AXIS_ORDER:
        value = defaults[name]
        if value not in AXES[name]:
            raise KernelScreenError(f"{name} value not on the grid: {value}")
        indices.append(AXES[name].index(value))
    return encode_indices(indices)


def render_candidate(decoded: Mapping[str, Any]) -> str:
    values = decoded["values"]
    local = {
        "identity": "1",
        "one_plus_u": "1 + u",
        "sqrt_one_plus_u_squared": "sqrt(1 + u^2)",
    }[values["local"]]
    terms = []
    if values["w_yukawa"] != "0":
        terms.append(f"{values['w_yukawa']}*exp(-s/{values['L1']})")
    if values["w_power"] != "0":
        terms.append(
            f"{values['w_power']}*(s/{values['L2']})^{values['p']}"
            f"*(1+s/{values['L2']})^-({values['p']}+{values['t']})"
        )
    kernel = " + ".join(terms) if terms else "0"
    return f"g_obs = [{local}]*g_N + conv(rho_b, K);  K(s) = ({kernel})/s^2,  u = (g_N/a0)^(-1/2)"


# ---------------------------------------------------------------------------
# Frozen geometry: distance-distribution quadrature nodes at 50 digits
# ---------------------------------------------------------------------------

_GL_CACHE: dict[int, list[tuple[mp.mpf, mp.mpf]]] = {}
_GEOMETRY_CACHE: dict[int, dict[str, Any]] = {}


def _gauss_legendre(n: int) -> list[tuple[mp.mpf, mp.mpf]]:
    """Gauss-Legendre nodes/weights on [-1, 1] at 50 digits (Newton on P_n)."""

    if n not in _GL_CACHE:
        _dps()
        pairs: list[tuple[mp.mpf, mp.mpf]] = []
        for k in range(1, n + 1):
            x = mp.cos(mp.pi * (k - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            derivative = mp.mpf(1)
            for _ in range(80):
                p0, p1 = mp.mpf(1), x
                for j in range(2, n + 1):
                    p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
                derivative = n * (x * p1 - p0) / (x * x - 1)
                step = p1 / derivative
                x -= step
                if abs(step) < mp.mpf(10) ** (-52):
                    break
            pairs.append((x, 2 / ((1 - x * x) * derivative * derivative)))
        _GL_CACHE[n] = pairs
    return _GL_CACHE[n]


def _panel(a: mp.mpf, b: mp.mpf, n: int) -> list[tuple[mp.mpf, mp.mpf]]:
    """(s, ds-weight) Gauss-Legendre pairs for the interval [a, b]."""

    half = (b - a) / 2
    mid = (a + b) / 2
    return [(mid + half * x, w * half) for x, w in _gauss_legendre(n)]


def _side_ladder(limit: mp.mpf, head: mp.mpf, ratio: int, extra: Iterable[mp.mpf]) -> list[mp.mpf]:
    """Geometric breakpoints in x = |s - r| from the head width out to ``limit``."""

    points = {limit}
    for value in extra:
        if head < value < limit:
            points.add(value)
    x = head
    while x * ratio < limit:
        x = x * ratio
        points.add(x)
    return sorted(points)


def _probe_panels(radius: mp.mpf, truncation: mp.mpf, refine: int) -> list[tuple[mp.mpf, mp.mpf, int]]:
    """Panels (a, b, order) covering s in (0, radius + truncation) for one probe.

    Head panels bracket the kink at s = r; geometric side ladders in x = |s - r|
    resolve the source-scale structure on both sides; declared kinks (the far mass
    edge and the truncation switch) are explicit breakpoints.  ``refine`` = 2 halves
    the ladder ratio and raises panel orders for the convergence control.
    """

    config = KERNEL_CONFIG["geometry"]
    ratio = max(2, config["ladder_ratio"] // refine)
    head_order = config["panel_orders"]["head"] * refine
    side_order = config["panel_orders"]["side"] * refine
    head = mp.mpf("0.05") * min(radius, mp.mpf(1))
    panels = [(radius - head, radius, head_order), (radius, radius + head, head_order)]
    low = head
    for x in _side_ladder(radius, head, ratio, ()):
        panels.append((radius - x, radius - low, side_order))
        low = x
    kinks = [truncation - radius]
    if truncation > 2 * radius:
        kinks.append(truncation - 2 * radius)
    low = head
    for x in _side_ladder(truncation, head, ratio, kinks):
        panels.append((radius + low, radius + x, side_order))
        low = x
    return [(a, b, order) for a, b, order in panels if b > a]


def _phi_cluster(radius: mp.mpf, s: mp.mpf) -> mp.mpf:
    """Beta-model gas Phi_r(s): rho = amplitude/(4 pi (1 + a^2)), truncated at 32."""

    truncation = mp.mpf(KERNEL_CONFIG["geometry"]["cluster"]["truncation_radius"])
    amplitude = mp.mpf(KERNEL_CONFIG["geometry"]["cluster"]["amplitude_4pi_rho0"])
    a1 = abs(radius - s)
    a2 = min(truncation, radius + s)
    if a1 >= a2:
        return mp.mpf(0)
    c = radius * radius + s * s
    value = (c + 1) * (mp.log(1 + a2 * a2) - mp.log(1 + a1 * a1)) - (a2 * a2 - a1 * a1)
    return amplitude / (8 * radius * radius) * value


def _phi_hernquist(radius: mp.mpf, s: mp.mpf) -> mp.mpf:
    """Unit-mass Hernquist Phi_r(s), scale 1, truncated at 1024."""

    truncation = mp.mpf(KERNEL_CONFIG["geometry"]["hernquist"]["truncation_radius"])
    a1 = abs(radius - s)
    a2 = min(truncation, radius + s)
    if a1 >= a2:
        return mp.mpf(0)
    c = radius * radius + s * s

    def antiderivative(a: mp.mpf) -> mp.mpf:
        return -(c - 1) / (2 * (1 + a) ** 2) - 2 / (1 + a) - mp.log(1 + a)

    return (antiderivative(a2) - antiderivative(a1)) / (2 * radius * radius)


def _disk_projection(radius: mp.mpf, s: mp.mpf) -> mp.mpf:
    """Razor-thin unit Freeman disk: Phi = s * A_r(s), A_r by angular mp.quad."""

    truncation = mp.mpf(KERNEL_CONFIG["geometry"]["disk"]["truncation_radius"])
    if s >= truncation + radius:
        return mp.mpf(0)
    cut = (truncation * truncation - radius * radius - s * s) / (2 * radius * s)
    lower = mp.mpf(0) if cut >= 1 else mp.acos(cut)

    def integrand(psi: mp.mpf) -> mp.mpf:
        rho = mp.sqrt(radius * radius + s * s + 2 * radius * s * mp.cos(psi))
        return -mp.cos(psi) * mp.e ** (-rho) / (2 * mp.pi)

    return s * 2 * mp.quad(integrand, [lower, mp.pi])


def _lensing_path_radii() -> list[mp.mpf]:
    """The 165 frozen path-node radii b*cosh(t), mass-independent, sorted unique."""

    _dps()
    config = GATE_CONFIG["lensing"]
    length = mp.mpf(config["path_half_length"])
    radii: set[str] = set()
    values: list[mp.mpf] = []
    for impact in config["impact_parameters"]:
        b = mp.mpf(impact)
        t_max = mp.asinh(length / b)
        step = t_max / (config["path_nodes"] - 1)
        for index in range(config["path_nodes"]):
            radius = b * mp.cosh(index * step)
            key = mp.nstr(radius, 40)
            if key not in radii:
                radii.add(key)
                values.append(radius)
    return sorted(values)


def build_kernel_geometry(refine: int = 1) -> dict[str, Any]:
    """All frozen distance-distribution nodes for the three control systems.

    Returns per system a list of rows ``{radius, nodes: [(s, w)], newton_exact}``,
    everything as 50-digit mpmath values.  ``refine`` > 1 builds the denser geometry
    used only by the convergence control; the frozen claim is ``refine = 1``.
    """

    if refine not in (1, 2):
        raise KernelScreenError("refine must be 1 (frozen) or 2 (convergence control)")
    if refine in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[refine]
    _dps()
    geometry: dict[str, Any] = {}

    def rows_for(
        radii: Sequence[mp.mpf],
        phi: Any,
        truncation: mp.mpf,
        newton_exact: Any,
    ) -> list[dict[str, Any]]:
        rows = []
        for radius in radii:
            nodes = []
            for a, b, order in _probe_panels(radius, truncation, refine):
                for s, weight in _panel(a, b, order):
                    factor = phi(radius, s)
                    if factor != 0:
                        nodes.append((s, weight * factor))
            rows.append({"radius": radius, "nodes": nodes, "newton_exact": newton_exact(radius)})
        return rows

    disk_config = KERNEL_CONFIG["geometry"]["disk"]
    scale = mp.mpf(disk_config["scale_length"])

    def disk_gbar(radius: mp.mpf) -> mp.mpf:
        y = radius / (2 * scale)
        return y * (
            mp.besseli(0, y) * mp.besselk(0, y) - mp.besseli(1, y) * mp.besselk(1, y)
        )

    geometry["disk"] = rows_for(
        [mp.mpf(r) for r in disk_config["probe_radii"]],
        _disk_projection,
        mp.mpf(disk_config["truncation_radius"]),
        disk_gbar,
    )
    geometry["hernquist"] = rows_for(
        _lensing_path_radii(),
        _phi_hernquist,
        mp.mpf(KERNEL_CONFIG["geometry"]["hernquist"]["truncation_radius"]),
        lambda radius: 1 / (radius + 1) ** 2,
    )
    geometry["cluster"] = rows_for(
        [_fraction(text) for text in KERNEL_CONFIG["geometry"]["cluster"]["probe_radii"]],
        _phi_cluster,
        mp.mpf(KERNEL_CONFIG["geometry"]["cluster"]["truncation_radius"]),
        lambda radius: mp.mpf(9) * (radius - mp.atan(radius)) / (radius * radius),
    )
    _GEOMETRY_CACHE[refine] = geometry
    return geometry


def geometry_newton_residuals(geometry: Mapping[str, Any]) -> dict[str, str]:
    """Max relative error of the Newton kernel sum against the exact g_bar, per system.

    This is the validation the whole build stands on: sum w/s^2 must reproduce the
    frozen 50-digit baryonic fields to the declared 1e-10 at every probe.
    """

    _dps()
    worst: dict[str, str] = {}
    for system, rows in geometry.items():
        largest = mp.mpf(0)
        for row in rows:
            total = mp.fsum(w / (s * s) for s, w in row["nodes"])
            largest = max(largest, abs(total / row["newton_exact"] - 1))
        worst[system] = mp.nstr(largest, 6)
    return worst


def geometry_sha256(geometry: Mapping[str, Any]) -> str:
    """Deterministic content hash of the frozen node tables (40-digit strings)."""

    payload = {
        system: [
            {
                "radius": mp.nstr(row["radius"], 40),
                "newton_exact": mp.nstr(row["newton_exact"], 40),
                "nodes": [[mp.nstr(s, 40), mp.nstr(w, 40)] for s, w in row["nodes"]],
            }
            for row in rows
        ]
        for system, rows in sorted(geometry.items())
    }
    return canonical_sha256(payload)


def _kernel_exact(s: mp.mpf, values: Mapping[str, str]) -> mp.mpf:
    """K(s) at 50 digits for exact re-verification."""

    total = mp.mpf(0)
    w_yukawa = _fraction(values["w_yukawa"])
    if w_yukawa != 0:
        total += w_yukawa * mp.e ** (-s / _fraction(values["L1"]))
    w_power = _fraction(values["w_power"])
    if w_power != 0:
        ratio = s / _fraction(values["L2"])
        p = _fraction(values["p"])
        t = _fraction(values["t"])
        total += w_power * ratio**p * (1 + ratio) ** (-(p + t))
    return total / (s * s)


def _nu_local_exact(local: str, y: mp.mpf) -> mp.mpf:
    u = 1 / mp.sqrt(y)
    if local == "identity":
        return mp.mpf(1)
    if local == "one_plus_u":
        return 1 + u
    if local == "sqrt_one_plus_u_squared":
        return mp.sqrt(1 + u * u)
    raise KernelScreenError(f"unknown local factor: {local}")


def boost_exact(rows: Sequence[Mapping[str, Any]], values: Mapping[str, str]) -> list[mp.mpf]:
    """B at every row of one geometry system, summed at 50 digits."""

    if values["w_yukawa"] == "0" and values["w_power"] == "0":
        return [mp.mpf(0) for _ in rows]
    return [
        mp.fsum(w * _kernel_exact(s, values) for s, w in row["nodes"]) for row in rows
    ]


# ---------------------------------------------------------------------------
# Exact gate pack: probe tables bound to the reused frozen controls
# ---------------------------------------------------------------------------


def build_exact_pack(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """All candidate-independent gate inputs at 50 digits, bound to the frozen grids.

    Binding is fail-closed: the recomputed disk ``g_bar`` must round to the frozen
    screen floats, the recovered lensing node radii must reproduce the frozen node
    accelerations, and the cluster geometry's Newtonian field must equal the frozen
    50-digit probe table.
    """

    _dps()
    scale = mp.mpf(SCREEN_CONFIG["disk_scale_length"])
    disk_grid = build_probe_grid()
    galaxy: list[dict[str, Any]] = []
    for disk in disk_grid["disks"]:
        mass = _fraction(disk["mass_text"])
        points = []
        for point in disk["points"]:
            if not point["outer"]:
                continue
            radius = mp.mpf(point["radius"])
            y = radius / (2 * scale)
            gbar = (mass / scale**2) * y * (
                mp.besseli(0, y) * mp.besselk(0, y) - mp.besseli(1, y) * mp.besselk(1, y)
            )
            if float(gbar) != point["gbar"]:
                raise KernelScreenError("disk g_bar drifted from the frozen screen grid")
            points.append((radius, gbar))
        galaxy.append({"mass_text": disk["mass_text"], "mass": mass, "points": points})
    for row, (radius, _) in zip(geometry["disk"], galaxy[0]["points"], strict=True):
        if row["radius"] != radius:
            raise KernelScreenError("disk geometry rows do not match the frozen probe radii")

    lensing_grid = build_lensing_grid()
    row_index = {
        mp.nstr(row["radius"], 40): index for index, row in enumerate(geometry["hernquist"])
    }
    config = GATE_CONFIG["lensing"]
    length = mp.mpf(config["path_half_length"])
    pairs: list[dict[str, Any]] = []
    for integral in lensing_grid["integrals"]:
        mass = _fraction(integral["mass_text"])
        b = mp.mpf(integral["impact_parameter"])
        step = mp.asinh(length / b) / (config["path_nodes"] - 1)
        nodes = []
        for index, node in enumerate(integral["nodes"]):
            y = mp.mpf(node["y"])
            weight = mp.mpf(node["weight"])
            radius = b * mp.cosh(index * step)
            if abs(y * (radius + 1) ** 2 / mass - 1) > mp.mpf(10) ** (-40):
                raise KernelScreenError("recovered lensing radius contradicts the frozen node")
            key = mp.nstr(radius, 40)
            if key not in row_index:
                raise KernelScreenError("lensing node radius missing from the geometry rows")
            nodes.append((weight / y, y, row_index[key]))
        pairs.append(
            {
                "mass_index": integral["mass_index"],
                "mass_text": integral["mass_text"],
                "mass": mass,
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
    for row, (gbar, _) in zip(geometry["cluster"], cluster, strict=True):
        if abs(row["newton_exact"] / gbar - 1) > mp.mpf(10) ** (-45):
            raise KernelScreenError("cluster geometry contradicts the frozen probe table")

    newton_config = KERNEL_CONFIG["newton_control"]
    probes = []
    for y_value in (*newton_config["probe_y"], newton_config["report_probe_y"]):
        y = mp.mpf(y_value)
        probes.append((y, 1 / mp.sqrt(y)))
    return {
        "galaxy": galaxy,
        "lensing": pairs,
        "cluster": cluster,
        "newton": probes,
        "safety_s": mp.mpf(newton_config["yukawa_safety"]["s"]),
        "log_mass_span": mp.log(galaxy[-1]["mass"] / galaxy[0]["mass"]),
    }


# ---------------------------------------------------------------------------
# Float tables: basis boosts per (L1) and (L2, p, t), pregathers, local scalars
# ---------------------------------------------------------------------------

N_POWER_COMBOS = len(AXES["L2"]) * len(AXES["p"]) * len(AXES["t"])


def _axis_floats(name: str) -> np.ndarray:
    return np.array([float(_fraction(v)) for v in AXES[name]], dtype=np.float64)


def _padded_nodes(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(row["nodes"]) for row in rows)
    s = np.ones((len(rows), width), dtype=np.float64)
    w = np.zeros((len(rows), width), dtype=np.float64)
    for index, row in enumerate(rows):
        for column, (s_value, w_value) in enumerate(row["nodes"]):
            s[index, column] = float(s_value)
            w[index, column] = float(w_value)
    return s, w


def _basis_for(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """(BY[12, rows], BP[1536, rows]): unit-weight kernel boosts at every probe row."""

    s, w = _padded_nodes(rows)
    inv_s2 = np.where(w != 0, 1.0 / (s * s), 0.0)
    yukawa = np.empty((len(AXES["L1"]), len(rows)), dtype=np.float64)
    for index, l1 in enumerate(_axis_floats("L1")):
        yukawa[index] = (w * np.exp(-s / l1) * inv_s2).sum(axis=1)
    power = np.empty((N_POWER_COMBOS, len(rows)), dtype=np.float64)
    p_values = _axis_floats("p")
    t_values = _axis_floats("t")
    combo = 0
    for l2 in _axis_floats("L2"):
        ratio = s / l2
        log_ratio = np.log(ratio)
        log_one_plus = np.log1p(ratio)
        for p_value in p_values:
            for t_value in t_values:
                shape = np.exp(p_value * log_ratio - (p_value + t_value) * log_one_plus)
                power[combo] = (w * shape * inv_s2).sum(axis=1)
                combo += 1
    return yukawa, power


def _point_kernel_tables(s_value: float) -> tuple[np.ndarray, np.ndarray]:
    """K(s)*s^2 factors for a unit point source at one scalar separation."""

    yukawa = np.exp(-s_value / _axis_floats("L1"))
    power = np.empty(N_POWER_COMBOS, dtype=np.float64)
    combo = 0
    for l2 in _axis_floats("L2"):
        ratio = s_value / l2
        for p_value in _axis_floats("p"):
            for t_value in _axis_floats("t"):
                power[combo] = ratio**p_value * (1.0 + ratio) ** (-(p_value + t_value))
                combo += 1
    return yukawa, power


def build_sweep_tables(geometry: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, Any]:
    """Every array the batched sweep needs, built deterministically on CPU float64."""

    _dps()
    by_disk, bp_disk = _basis_for(geometry["disk"])
    by_hern, bp_hern = _basis_for(geometry["hernquist"])
    by_clu, bp_clu = _basis_for(geometry["cluster"])

    pair_count = len(pack["lensing"])
    gather = np.zeros((pair_count, len(geometry["hernquist"])), dtype=np.float64)
    for pair_index, pair in enumerate(pack["lensing"]):
        for gw, _, row in pair["nodes"]:
            gather[pair_index, row] += float(gw)
    ay = by_hern @ gather.T
    ap = bp_hern @ gather.T

    locals_ = AXES["local"]
    galaxy_scalar = np.zeros((len(locals_), 3, 5), dtype=np.float64)
    radius_mass = np.zeros((3, 5), dtype=np.float64)
    for m, disk in enumerate(pack["galaxy"]):
        for k, (radius, gbar) in enumerate(disk["points"]):
            radius_mass[m, k] = float(disk["mass"] * radius)
            for index, local in enumerate(locals_):
                galaxy_scalar[index, m, k] = float(_nu_local_exact(local, gbar) * gbar * radius)

    alpha_local = np.zeros((len(locals_), pair_count), dtype=np.float64)
    pair_mass = np.zeros(pair_count, dtype=np.float64)
    for pair_index, pair in enumerate(pack["lensing"]):
        pair_mass[pair_index] = float(pair["mass"])
        for index, local in enumerate(locals_):
            alpha_local[index, pair_index] = float(
                mp.fsum(gw * _nu_local_exact(local, y) * y for gw, y, _ in pair["nodes"])
            )

    cluster_scalar = np.zeros((len(locals_), 5), dtype=np.float64)
    gdyn = np.zeros(5, dtype=np.float64)
    for k, (gbar, gdyn_value) in enumerate(pack["cluster"]):
        gdyn[k] = float(gdyn_value)
        for index, local in enumerate(locals_):
            cluster_scalar[index, k] = float(_nu_local_exact(local, gbar) * gbar)

    newton_local = np.zeros((len(locals_), 3), dtype=np.float64)
    ky_point = np.zeros((len(AXES["L1"]), 3), dtype=np.float64)
    kp_point = np.zeros((N_POWER_COMBOS, 3), dtype=np.float64)
    for k, (y, s) in enumerate(pack["newton"]):
        for index, local in enumerate(locals_):
            newton_local[index, k] = float(_nu_local_exact(local, y) - 1)
        ky, kp = _point_kernel_tables(float(s))
        ky_point[:, k] = ky
        kp_point[:, k] = kp
    ky_safety, kp_safety = _point_kernel_tables(float(pack["safety_s"]))

    return {
        "BY_gal": np.ascontiguousarray(by_disk.T),  # (5, 12) for per-probe gathers
        "BP_gal": np.ascontiguousarray(bp_disk.T),
        "BY_clu": np.ascontiguousarray(by_clu.T),
        "BP_clu": np.ascontiguousarray(bp_clu.T),
        "AY": np.ascontiguousarray(ay.T),  # (15, 12)
        "AP": np.ascontiguousarray(ap.T),
        "KY_pt": ky_point,
        "KP_pt": kp_point,
        "KY_saf": ky_safety,
        "KP_saf": kp_safety,
        "S_gal": galaxy_scalar,
        "R_gal": radius_mass,
        "A_loc": alpha_local,
        "M_pair": pair_mass,
        "S_clu": cluster_scalar,
        "gdyn": gdyn,
        "N_loc": newton_local,
        "log_mass_span": float(pack["log_mass_span"]),
        "WY": _axis_floats("w_yukawa"),
        "WP": _axis_floats("w_power"),
    }


def _device_tables(tables: Mapping[str, Any], xp: Any) -> dict[str, Any]:
    """fp64 and fp32 device copies; the safety tables stay fp64 by construction."""

    out: dict[str, Any] = {"log_mass_span": tables["log_mass_span"]}
    for key, value in tables.items():
        if key == "log_mass_span":
            continue
        array = xp.asarray(value)
        out[key] = array
        out[key + "_f32"] = array.astype(xp.float32)
    return out


# ---------------------------------------------------------------------------
# Vectorized gate evaluation (shared numpy/cupy code path)
# ---------------------------------------------------------------------------


def _decode_batch(xp: Any, ordinals: Any) -> dict[str, Any]:
    value = ordinals.astype(xp.int64)
    indices: dict[str, Any] = {}
    for name, size in zip(reversed(AXIS_ORDER), reversed(AXIS_SIZES), strict=True):
        indices[name] = value % size
        value //= size
    indices["combo"] = (
        indices["L2"] * len(AXES["p"]) + indices["p"]
    ) * len(AXES["t"]) + indices["t"]
    return indices


def evaluate_batch(
    xp: Any, ordinals: Any, tables: Mapping[str, Any], *, dtype: Any, tier: str
) -> dict[str, Any]:
    """Every gate for one ordinal batch.  ``tier`` picks fp32 or fp64 thresholds.

    The Yukawa-safety margin is always computed in fp64 (declared in the config): the
    1e-8 criterion sits below fp32 resolution against O(1) table entries.
    """

    thresholds = {key: float(value) for key, value in KERNEL_CONFIG[tier].items()}
    suffix = "_f32" if dtype == xp.float32 else ""

    def table(name: str) -> Any:
        return tables[name + suffix]

    ix = _decode_batch(xp, ordinals)
    il, il1, ic = ix["local"], ix["L1"], ix["combo"]
    w_yukawa = table("WY")[ix["w_yukawa"]]
    w_power = table("WP")[ix["w_power"]]

    # Galaxy: flat outer curves and the Tully-Fisher slope on the frozen disks.
    valid = xp.ones(ordinals.shape[0], dtype=bool)
    flat_worst = xp.zeros(ordinals.shape[0], dtype=dtype)
    vflat = []
    for m in range(3):
        speed_sum = None
        vmax = None
        vmin = None
        for k in range(5):
            boost = w_yukawa * table("BY_gal")[k][il1] + w_power * table("BP_gal")[k][ic]
            v2 = table("S_gal")[:, m, k][il] + boost * dtype(float(tables["R_gal"][m, k]))
            valid &= v2 > 0
            speed = xp.sqrt(xp.maximum(v2, dtype(0)))
            vmax = speed if vmax is None else xp.maximum(vmax, speed)
            vmin = speed if vmin is None else xp.minimum(vmin, speed)
            speed_sum = speed if speed_sum is None else speed_sum + speed
        mean = speed_sum / dtype(5)
        safe_mean = xp.where(mean > 0, mean, dtype(1))
        flat_worst = xp.maximum(flat_worst, (vmax - vmin) / safe_mean)
        vflat.append(mean)
    flat_pass = flat_worst <= dtype(thresholds["flatness"])
    ratio = xp.log(
        xp.maximum(vflat[2], dtype(1e-30)) / xp.maximum(vflat[0], dtype(1e-30))
    )
    btfr_err = xp.where(
        ratio > 0,
        xp.abs(dtype(tables["log_mass_span"]) / xp.where(ratio > 0, ratio, dtype(1)) - 4),
        dtype(float("inf")),
    )
    btfr_pass = (ratio > 0) & (btfr_err <= dtype(thresholds["btfr_slope"]))
    galaxy_pass = valid & flat_pass & btfr_pass

    # Lensing: deflection flatness and dynamics-lensing consistency per mass.
    lens_valid = xp.ones(ordinals.shape[0], dtype=bool)
    worst_cons = xp.zeros(ordinals.shape[0], dtype=dtype)
    worst_flat = xp.zeros(ordinals.shape[0], dtype=dtype)
    two_pi = dtype(2.0 * math.pi)
    for m in range(3):
        expected = two_pi * vflat[m] * vflat[m]
        lens_valid &= expected > 0
        safe_expected = xp.where(expected > 0, expected, dtype(1))
        alpha_max = None
        alpha_min = None
        alpha_sum = None
        for b in range(5):
            pair = m * 5 + b
            alpha = table("A_loc")[:, pair][il] + dtype(float(tables["M_pair"][pair])) * (
                w_yukawa * table("AY")[pair][il1] + w_power * table("AP")[pair][ic]
            )
            deviation = xp.abs(alpha / safe_expected - 1)
            worst_cons = xp.maximum(worst_cons, deviation)
            alpha_max = alpha if alpha_max is None else xp.maximum(alpha_max, alpha)
            alpha_min = alpha if alpha_min is None else xp.minimum(alpha_min, alpha)
            alpha_sum = alpha if alpha_sum is None else alpha_sum + alpha
        mean_alpha = alpha_sum / dtype(5)
        lens_valid &= mean_alpha > 0
        safe_mean = xp.where(mean_alpha > 0, mean_alpha, dtype(1))
        worst_flat = xp.maximum(worst_flat, (alpha_max - alpha_min) / safe_mean)
    lensing_pass = (
        lens_valid
        & (worst_flat <= dtype(thresholds["lensing_flatness"]))
        & (worst_cons <= dtype(thresholds["lensing_consistency"]))
    )

    # Cluster: the decisive 5-probe hydrostatic ratio criterion.
    cluster_dev = xp.zeros(ordinals.shape[0], dtype=dtype)
    for k in range(5):
        boost = w_yukawa * table("BY_clu")[k][il1] + w_power * table("BP_clu")[k][ic]
        g_obs = table("S_clu")[:, k][il] + boost
        cluster_dev = xp.maximum(
            cluster_dev, xp.abs(g_obs / dtype(float(tables["gdyn"][k])) - 1)
        )
    cluster_pass = cluster_dev <= dtype(thresholds["cluster_consistency"])

    # Newton/solar: point-source probes plus the fp64 Yukawa-safety margin.
    newton_devs = []
    for k in range(3):
        deviation = xp.abs(
            table("N_loc")[:, k][il]
            + w_yukawa * table("KY_pt")[:, k][il1]
            + w_power * table("KP_pt")[:, k][ic]
        )
        newton_devs.append(deviation)
    w_yukawa64 = tables["WY"][ix["w_yukawa"]]
    w_power64 = tables["WP"][ix["w_power"]]
    safety = xp.abs(
        w_yukawa64 * tables["KY_saf"][il1] + w_power64 * tables["KP_saf"][ic]
    )
    safety_limit = float(
        KERNEL_CONFIG["newton_control"]["yukawa_safety"]["max_abs_boost_ratio"]
    )
    safety_pass = safety < safety_limit
    newton_pass = (
        (newton_devs[0] <= dtype(thresholds["newton_near"]))
        & (newton_devs[1] <= dtype(thresholds["newton_far"]))
        & safety_pass
    )

    all_pass = newton_pass & galaxy_pass & lensing_pass & cluster_pass
    infinity = dtype(float("inf"))
    return {
        "newton_pass": newton_pass,
        "safety_pass": safety_pass,
        "galaxy_pass": galaxy_pass,
        "lensing_pass": lensing_pass,
        "cluster_pass": cluster_pass,
        "all_pass": all_pass,
        "newton_near": newton_devs[0],
        "newton_far": newton_devs[1],
        "newton_report_y8": newton_devs[2],
        "safety_margin": safety,
        "flat_worst": xp.where(valid, flat_worst, infinity),
        "btfr_err": xp.where(valid, btfr_err, infinity),
        "lensing_flat": xp.where(lens_valid, worst_flat, infinity),
        "lensing_cons": xp.where(lens_valid, worst_cons, infinity),
        "cluster_dev": cluster_dev,
    }


# ---------------------------------------------------------------------------
# Exact re-verification (mpmath, 50 digits)
# ---------------------------------------------------------------------------


def build_exact_context() -> dict[str, Any]:
    """Geometry plus gate pack, cached; the single source for exact evaluation."""

    geometry = build_kernel_geometry()
    return {"geometry": geometry, "pack": build_exact_pack(geometry)}


def evaluate_candidate_exact(ordinal: int, context: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run every gate for one candidate at 50 digits, with decimal-string margins."""

    _dps()
    decoded = decode_ordinal(ordinal)
    values = decoded["values"]
    geometry = context["geometry"]
    pack = context["pack"]
    thresholds = {
        key: mp.mpf(value) for key, value in KERNEL_CONFIG["fp64_thresholds"].items()
    }

    disk_boost = boost_exact(geometry["disk"], values)
    hern_boost = boost_exact(geometry["hernquist"], values)
    cluster_boost = boost_exact(geometry["cluster"], values)
    local = values["local"]

    # Galaxy.
    galaxy_valid = True
    flat_worst = mp.mpf(0)
    vflat: list[mp.mpf] = []
    per_disk = []
    for disk in pack["galaxy"]:
        speeds = []
        for k, (radius, gbar) in enumerate(disk["points"]):
            v2 = (_nu_local_exact(local, gbar) * gbar + disk["mass"] * disk_boost[k]) * radius
            if v2 <= 0:
                galaxy_valid = False
                break
            speeds.append(mp.sqrt(v2))
        if not galaxy_valid:
            break
        mean = sum(speeds) / len(speeds)
        spread = (max(speeds) - min(speeds)) / mean
        flat_worst = max(flat_worst, spread)
        vflat.append(mean)
        per_disk.append(
            {
                "mass_text": disk["mass_text"],
                "spread": _text(spread),
                "v_flat": _text(mean),
            }
        )
    if galaxy_valid:
        ratio = mp.log(vflat[-1] / vflat[0])
        if ratio > 0:
            slope = pack["log_mass_span"] / ratio
            btfr_err = abs(slope - 4)
            slope_text = _text(slope)
        else:
            btfr_err = mp.inf
            slope_text = None
        galaxy_passes = bool(
            flat_worst <= thresholds["flatness"] and btfr_err <= thresholds["btfr_slope"]
        )
    else:
        btfr_err = mp.inf
        slope_text = None
        galaxy_passes = False

    # Lensing.
    lens_valid = galaxy_valid
    worst_flat = mp.mpf(0)
    worst_cons = mp.mpf(0)
    if lens_valid:
        alphas_by_mass: dict[int, list[mp.mpf]] = {}
        for pair in pack["lensing"]:
            total = mp.fsum(
                gw * (_nu_local_exact(local, y) * y + pair["mass"] * hern_boost[row])
                for gw, y, row in pair["nodes"]
            )
            alphas_by_mass.setdefault(pair["mass_index"], []).append(total)
        for mass_index, alphas in sorted(alphas_by_mass.items()):
            expected = 2 * mp.pi * vflat[mass_index] ** 2
            mean = sum(alphas) / len(alphas)
            if expected <= 0 or mean <= 0:
                lens_valid = False
                break
            worst_flat = max(worst_flat, (max(alphas) - min(alphas)) / mean)
            for alpha in alphas:
                worst_cons = max(worst_cons, abs(alpha / expected - 1))
    lensing_passes = bool(
        lens_valid
        and worst_flat <= thresholds["lensing_flatness"]
        and worst_cons <= thresholds["lensing_consistency"]
    )

    # Cluster.
    ratios = []
    shortfalls = []
    for k, (gbar, gdyn) in enumerate(pack["cluster"]):
        g_obs = _nu_local_exact(local, gbar) * gbar + cluster_boost[k]
        ratios.append(g_obs / gdyn)
        shortfalls.append(gdyn / g_obs if g_obs > 0 else mp.inf)
    deviations = [abs(ratio - 1) for ratio in ratios]
    cluster_passes = bool(max(deviations) <= thresholds["cluster_consistency"])

    # Newton/solar point-source probes and the Yukawa-safety margin.
    newton_entries = []
    for y, s in pack["newton"]:
        boost_ratio = _kernel_exact(s, values) * s * s if (
            values["w_yukawa"] != "0" or values["w_power"] != "0"
        ) else mp.mpf(0)
        newton_entries.append(abs((_nu_local_exact(local, y) - 1) + boost_ratio))
    safety_s = pack["safety_s"]
    safety_margin = (
        abs(_kernel_exact(safety_s, values) * safety_s * safety_s)
        if (values["w_yukawa"] != "0" or values["w_power"] != "0")
        else mp.mpf(0)
    )
    safety_limit = mp.mpf(
        KERNEL_CONFIG["newton_control"]["yukawa_safety"]["max_abs_boost_ratio"]
    )
    newton_passes = bool(
        newton_entries[0] <= thresholds["newton_near"]
        and newton_entries[1] <= thresholds["newton_far"]
        and safety_margin < safety_limit
    )

    return {
        "ordinal": ordinal,
        "indices": decoded["indices"],
        "values": values,
        "formula": render_candidate(decoded),
        "newton": {
            "passes": newton_passes,
            "near_deviation": _text(newton_entries[0]),
            "far_deviation": _text(newton_entries[1]),
            "report_y8_deviation": _text(newton_entries[2]),
            "safety_margin": _text(safety_margin),
            "safety_passes": bool(safety_margin < safety_limit),
        },
        "galaxy": {
            "passes": galaxy_passes,
            "valid": galaxy_valid,
            "flat_worst": _text(flat_worst) if galaxy_valid else None,
            "btfr_slope": slope_text,
            "btfr_error": _text(btfr_err) if btfr_err != mp.inf else "inf",
            "per_disk": per_disk,
        },
        "lensing": {
            "passes": lensing_passes,
            "valid": lens_valid,
            "worst_flatness": _text(worst_flat) if lens_valid else None,
            "worst_consistency": _text(worst_cons) if lens_valid else None,
        },
        "cluster": {
            "passes": cluster_passes,
            "max_deviation": _text(max(deviations)),
            "closest_probe_deviation": _text(min(deviations)),
            "ratio_by_probe": [_text(value) for value in ratios],
            "shortfall_by_probe": [
                "inf" if value == mp.inf else _text(value) for value in shortfalls
            ],
            "shortfall_min": (
                "inf" if min(shortfalls) == mp.inf else _text(min(shortfalls))
            ),
        },
        "all_pass": bool(newton_passes and galaxy_passes and lensing_passes and cluster_passes),
    }


CONTROL_ORDINALS = {
    "newton_identity": encode_named(),
    "sqrt_local": encode_named(local="sqrt_one_plus_u_squared"),
    "linear_u_local": encode_named(local="one_plus_u"),
    "yukawa_hand": encode_named(w_yukawa="1", L1="4"),
    "power_hand": encode_named(w_power="1", L2="2", p="2", t="-1"),
}


def _assert_known_answer_controls(controls: Mapping[str, Mapping[str, Any]]) -> None:
    """The calibration is part of the claim: a broken control aborts the run.

    The first three controls are the embedded pointwise family and must reproduce the
    prior screens' verdicts exactly (including the sealed cluster negative); the two
    hand-built kernels pin the Solar gates from both sides.
    """

    newton = controls["newton_identity"]
    if not newton["newton"]["passes"]:
        raise KernelScreenError("Newton control failed Newtonian recovery")
    if newton["galaxy"]["passes"] or newton["cluster"]["passes"] or newton["lensing"]["passes"]:
        raise KernelScreenError("Newton control unexpectedly passed a physics gate")
    if float(newton["cluster"]["shortfall_min"]) < 1.5:
        raise KernelScreenError("cluster calibration too weak: Newton shortfall < 3/2")
    sqrt_local = controls["sqrt_local"]
    if not (sqrt_local["galaxy"]["passes"] and sqrt_local["lensing"]["passes"]):
        raise KernelScreenError("sqrt-local control lost its galaxy/lensing passes")
    if sqrt_local["cluster"]["passes"]:
        raise KernelScreenError(
            "embedded pointwise family no longer reproduces the sealed cluster negative"
        )
    if float(sqrt_local["cluster"]["closest_probe_deviation"]) < 0.3:
        raise KernelScreenError("embedded sqrt-local cluster margin drifted from the prior screen")
    linear = controls["linear_u_local"]
    if not linear["galaxy"]["passes"] or linear["lensing"]["passes"]:
        raise KernelScreenError("linear-u control drifted: must flatten curves yet fail lensing")
    if linear["cluster"]["passes"]:
        raise KernelScreenError("linear-u control unexpectedly passed the cluster gate")
    yukawa = controls["yukawa_hand"]
    if yukawa["newton"]["safety_passes"]:
        raise KernelScreenError("unscreened Yukawa slipped past the safety probe")
    power = controls["power_hand"]
    if not power["newton"]["passes"]:
        raise KernelScreenError("the solar-safe rising-tail witness died at the Newton gate")


# ---------------------------------------------------------------------------
# Survivor structure: equivalence families, Pareto, covariant lifts
# ---------------------------------------------------------------------------


def passer_families(ordinals: Sequence[int]) -> list[list[int]]:
    """Grid-neighbor equivalence families (Chebyshev distance 1 across all axes)."""

    tuples = {tuple(decode_ordinal(int(o))["indices"]): int(o) for o in ordinals}
    seen: set[tuple[int, ...]] = set()
    families: list[list[int]] = []
    offsets = [(-1, 0, 1)] * len(AXIS_SIZES)
    for start in sorted(tuples):
        if start in seen:
            continue
        component = []
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(tuples[node])
            candidates = [node]
            for axis, choices in enumerate(offsets):
                candidates = [
                    (*c[:axis], c[axis] + delta, *c[axis + 1 :])
                    for c in candidates
                    for delta in choices
                    if 0 <= c[axis] + delta < AXIS_SIZES[axis]
                ]
            for neighbor in candidates:
                if neighbor in tuples and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        families.append(sorted(component))
    return families


def active_parameter_count(values: Mapping[str, str]) -> int:
    """Pareto simplicity axis: parameters doing work in this candidate."""

    count = 0
    if values["local"] != "identity":
        count += 1
    if values["w_yukawa"] != "0":
        count += 2  # amplitude + range
    if values["w_power"] != "0":
        count += 4  # amplitude + scale + rise + tail
    return count


def emit_covariant_lift(values: Mapping[str, str]) -> dict[str, Any]:
    """Typed covariant-lift candidate for the existing machinery; no derivation claimed."""

    components: list[dict[str, str]] = []
    if values["w_yukawa"] != "0":
        mass = str(1 / Fraction(values["L1"]))
        components.append(
            {
                "mechanism": "massive_scalar_exchange",
                "static_kernel": f"{values['w_yukawa']} * exp(-s/{values['L1']}) / s^2",
                "field_theory_ansatz": (
                    "S_int = g*phi*rho with (Box - m^2)phi = 4*pi*G*g*rho; "
                    f"mediator mass m = 1/L1 = {mass} (code units); "
                    f"coupling g^2 proportional to w_Y = {values['w_yukawa']}"
                ),
            }
        )
    if values["w_power"] != "0":
        t = Fraction(_to_fraction_text(values["t"]))
        alpha = 1 - t / 2
        tail = -(2 + t)
        components.append(
            {
                "mechanism": "nonlocal_propagator_correction",
                "static_kernel": (
                    f"{values['w_power']} * (s/{values['L2']})^{values['p']} * "
                    f"(1+s/{values['L2']})^-({values['p']}+{values['t']}) / s^2"
                ),
                "field_theory_ansatz": (
                    f"long-range force tail s^({tail}) <-> static Green's function of a "
                    f"nonlocal operator (-Box)^alpha with alpha = 1 - t/2 = {alpha}; "
                    "t = 0 is a long-range renormalization of G (running coupling); "
                    f"short-range suppression exponent p = {values['p']} <-> UV form "
                    f"factor turning on at L2 = {values['L2']}"
                ),
            }
        )
    if values["local"] != "identity":
        components.append(
            {
                "mechanism": "pointwise_modified_dynamics",
                "static_kernel": f"multiplicative nu_loc = {values['local']} on g_N",
                "field_theory_ansatz": (
                    "AQUAL/k-essence lane of the existing covariant machinery "
                    "(pointwise nu(y) factor retained alongside the kernel)"
                ),
            }
        )
    return {
        "kernel_form": render_candidate({"values": dict(values)}),
        "components": components,
        "claims": {"first_principles_derivation_pending": True},
    }


def _to_fraction_text(text: str) -> str:
    return text if "/" in text or text.lstrip("-").isdigit() else str(Fraction(text))


def _pareto_front(axes: np.ndarray, cap: int) -> np.ndarray:
    """Row indices of the non-dominated set (minimize every axis), deterministic order."""

    if axes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    dominated = np.zeros(axes.shape[0], dtype=bool)
    for start in range(0, axes.shape[0], 512):
        stop = min(start + 512, axes.shape[0])
        piece = axes[start:stop]
        not_worse = (axes[:, None, :] <= piece[None, :, :]).all(axis=2)
        strictly = (axes[:, None, :] < piece[None, :, :]).any(axis=2)
        dominated[start:stop] = (not_worse & strictly).any(axis=0)
    front = np.flatnonzero(~dominated)
    order = np.lexsort((front, axes[front, 1], axes[front, 0]))
    return front[order][:cap]


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


class _ArgminTracker:
    """Deterministic (value, ordinal) minimum across batches; ties -> lowest ordinal."""

    def __init__(self) -> None:
        self.value = float("inf")
        self.ordinal: int | None = None

    def update(self, xp: Any, metric: Any, ordinals: Any, gpu: bool) -> None:
        if metric.shape[0] == 0:
            return
        row = int(xp.argmin(metric))
        value = float(metric[row])
        ordinal = int(ordinals[row])
        if not math.isfinite(value):
            return
        if value < self.value or (value == self.value and (
            self.ordinal is None or ordinal < self.ordinal
        )):
            self.value = value
            self.ordinal = ordinal


def run_screen(
    *,
    limit: int | None = None,
    batch_size: int = 1 << 21,
    use_gpu: bool = True,
    pareto_cap: int = 64,
) -> dict[str, Any]:
    """Screen the family (or its first ``limit`` ordinals) and seal a receipt."""

    if not SYSTEM_CAPS["min_batch_size"] <= batch_size <= SYSTEM_CAPS["max_batch_size"]:
        raise KernelScreenError(f"batch_size outside system caps: {batch_size}")
    if not 1 <= pareto_cap <= SYSTEM_CAPS["max_pareto_reported"]:
        raise KernelScreenError(f"pareto_cap outside system caps: {pareto_cap}")
    if limit is not None and limit < 1:
        raise KernelScreenError(f"limit must be positive: {limit}")
    if FAMILY_SIZE != math.prod(AXIS_SIZES) or not 10**8 <= FAMILY_SIZE < 10**9:
        raise KernelScreenError("family size drifted outside the declared 10^8-10^9 band")

    xp, device, gpu = _array_module(use_gpu)
    context = build_exact_context()
    geometry = context["geometry"]
    pack = context["pack"]
    newton_validation = geometry_newton_residuals(geometry)
    budget = mp.mpf(KERNEL_CONFIG["geometry"]["newton_validation_max_relative_error"])
    for system, residual in newton_validation.items():
        if mp.mpf(residual) > budget:
            raise KernelScreenError(f"geometry validation failed for {system}: {residual}")
    tables = build_sweep_tables(geometry, pack)
    device_tables = _device_tables(tables, xp)

    total = FAMILY_SIZE if limit is None else min(limit, FAMILY_SIZE)
    gate_names = ("newton", "safety", "galaxy", "lensing", "cluster", "all")
    fp32_counts = dict.fromkeys(gate_names, 0)
    trackers = {
        "cluster_any": _ArgminTracker(),
        "cluster_solar_safe": _ArgminTracker(),
        "lensing_among_galaxy_passers": _ArgminTracker(),
        "galaxy": _ArgminTracker(),
    }
    survivor_parts: list[np.ndarray] = []
    started = time.perf_counter()
    processed = 0
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        ordinals = xp.arange(start, stop, dtype=xp.int64)
        sweep = evaluate_batch(
            xp, ordinals, device_tables, dtype=xp.float32, tier="fp32_thresholds"
        )
        for name in gate_names:
            fp32_counts[name] += int(sweep[f"{name}_pass"].sum())
        chosen = ordinals[sweep["all_pass"]]
        if chosen.shape[0]:
            survivor_parts.append(chosen.get() if gpu else np.asarray(chosen))
        infinity = xp.float32(float("inf"))
        trackers["cluster_any"].update(xp, sweep["cluster_dev"], ordinals, gpu)
        safe = sweep["newton_pass"]
        trackers["cluster_solar_safe"].update(
            xp, xp.where(safe, sweep["cluster_dev"], infinity), ordinals, gpu
        )
        trackers["lensing_among_galaxy_passers"].update(
            xp,
            xp.where(sweep["galaxy_pass"], sweep["lensing_cons"], infinity),
            ordinals,
            gpu,
        )
        flat_limit = xp.float32(float(KERNEL_CONFIG["fp32_thresholds"]["flatness"]))
        btfr_limit = xp.float32(float(KERNEL_CONFIG["fp32_thresholds"]["btfr_slope"]))
        galaxy_metric = xp.maximum(
            sweep["flat_worst"] / flat_limit, sweep["btfr_err"] / btfr_limit
        )
        trackers["galaxy"].update(xp, galaxy_metric, ordinals, gpu)
        processed = stop
    elapsed_sweep = time.perf_counter() - started

    survivors32 = (
        np.unique(np.concatenate(survivor_parts))
        if survivor_parts
        else np.empty(0, dtype=np.int64)
    )

    # fp64 strict recheck of every fp32 survivor, keeping per-gate decisions.
    fp64_counts = dict.fromkeys(gate_names, 0)
    passer_ordinals: list[int] = []
    passer_metrics: dict[int, dict[str, float]] = {}
    for start in range(0, survivors32.size, batch_size):
        chunk = xp.asarray(survivors32[start : start + batch_size])
        recheck = evaluate_batch(
            xp, chunk, device_tables, dtype=xp.float64, tier="fp64_thresholds"
        )
        host = {
            key: (value.get() if gpu else np.asarray(value)) for key, value in recheck.items()
        }
        for name in gate_names:
            fp64_counts[name] += int(host[f"{name}_pass"].sum())
        rows = np.flatnonzero(host["all_pass"])
        for row in rows:
            ordinal = int(survivors32[start + row])
            passer_ordinals.append(ordinal)
            passer_metrics[ordinal] = {
                "cluster_dev": float(host["cluster_dev"][row]),
                "lensing_cons": float(host["lensing_cons"][row]),
                "newton_far": float(host["newton_far"][row]),
            }

    # fp64 refinement of the tracked closest approaches.
    tracked_ordinals = sorted(
        {tracker.ordinal for tracker in trackers.values() if tracker.ordinal is not None}
    )
    refined: dict[int, dict[str, Any]] = {}
    if tracked_ordinals:
        batch = evaluate_batch(
            np,
            np.asarray(tracked_ordinals, dtype=np.int64),
            _device_tables(tables, np),
            dtype=np.float64,
            tier="fp64_thresholds",
        )
        for row, ordinal in enumerate(tracked_ordinals):
            refined[ordinal] = {
                key: float(np.asarray(batch[key])[row])
                for key in (
                    "cluster_dev",
                    "lensing_cons",
                    "lensing_flat",
                    "flat_worst",
                    "btfr_err",
                    "newton_near",
                    "newton_far",
                    "safety_margin",
                )
            }

    # Known-answer controls, always at 50 digits; a broken control aborts the run.
    controls = {
        name: evaluate_candidate_exact(ordinal, context)
        for name, ordinal in CONTROL_ORDINALS.items()
    }
    _assert_known_answer_controls(controls)

    # 50-digit re-verification of every reported candidate, within a recorded budget.
    verify_ordinals: list[int] = list(passer_ordinals)
    for ordinal in tracked_ordinals:
        if ordinal not in verify_ordinals:
            verify_ordinals.append(ordinal)
    exact_budget = SYSTEM_CAPS["max_exact_verifications"]
    exact_truncated = max(0, len(verify_ordinals) - exact_budget)
    verify_ordinals = verify_ordinals[:exact_budget]
    exact_verification = []
    passer_set = set(passer_ordinals)
    for ordinal in verify_ordinals:
        verdict = evaluate_candidate_exact(ordinal, context)
        verdict["role"] = "all_gate_passer" if ordinal in passer_set else "closest_approach"
        verdict["exact_confirmed"] = (
            verdict["all_pass"] == (ordinal in passer_set)
            if verdict["role"] == "all_gate_passer"
            else True
        )
        exact_verification.append(verdict)
    confirmed = sum(1 for entry in exact_verification if entry["exact_confirmed"])

    # Survivor structure: families, Pareto, covariant lifts (the historic branch).
    families_block: list[dict[str, Any]] = []
    pareto_entries: list[dict[str, Any]] = []
    family_truncated = False
    if passer_ordinals:
        family_input = passer_ordinals
        if len(family_input) > SYSTEM_CAPS["family_analysis_cap"]:
            family_truncated = True
            family_input = sorted(
                family_input, key=lambda o: (passer_metrics[o]["cluster_dev"], o)
            )[: SYSTEM_CAPS["family_analysis_cap"]]
        families = passer_families(family_input)
        for family in families[: SYSTEM_CAPS["max_families_reported"]]:
            representative = min(
                family, key=lambda o: (passer_metrics[o]["cluster_dev"], o)
            )
            values = decode_ordinal(representative)["values"]
            families_block.append(
                {
                    "size": len(family),
                    "representative_ordinal": representative,
                    "representative_values": values,
                    "representative_formula": render_candidate({"values": values}),
                    "cluster_deviation": _text(passer_metrics[representative]["cluster_dev"]),
                    "lensing_consistency": _text(
                        passer_metrics[representative]["lensing_cons"]
                    ),
                    "covariant_lift_candidate": emit_covariant_lift(values),
                }
            )
        axes = np.array(
            [
                [
                    passer_metrics[o]["cluster_dev"],
                    passer_metrics[o]["lensing_cons"],
                    passer_metrics[o]["newton_far"],
                    float(active_parameter_count(decode_ordinal(o)["values"])),
                ]
                for o in passer_ordinals
            ],
            dtype=np.float64,
        )
        front_rows = _pareto_front(axes, pareto_cap)
        for row in front_rows:
            ordinal = passer_ordinals[int(row)]
            values = decode_ordinal(ordinal)["values"]
            pareto_entries.append(
                {
                    "ordinal": ordinal,
                    "values": values,
                    "formula": render_candidate({"values": values}),
                    "cluster_deviation": _text(axes[row, 0]),
                    "lensing_consistency": _text(axes[row, 1]),
                    "newton_far_deviation": _text(axes[row, 2]),
                    "active_parameters": int(axes[row, 3]),
                }
            )
        family_count = len(families)
    else:
        family_count = 0

    # Sealed negative: closest approaches per gate plus the named structural direction.
    sealed_negative: dict[str, Any] | None = None
    if not passer_ordinals:
        tolerance = float(KERNEL_CONFIG["fp64_thresholds"]["cluster_consistency"])

        def approach(name: str, metric_key: str, threshold: str) -> dict[str, Any] | None:
            tracker = trackers[name]
            if tracker.ordinal is None:
                return None
            values = decode_ordinal(tracker.ordinal)["values"]
            return {
                "ordinal": tracker.ordinal,
                "values": values,
                "formula": render_candidate({"values": values}),
                "fp64_metric": _text(refined[tracker.ordinal][metric_key]),
                "threshold": threshold,
                "fp64_metrics": {
                    key: _text(value) for key, value in refined[tracker.ordinal].items()
                },
            }

        closest = {
            "cluster_any": approach(
                "cluster_any", "cluster_dev", KERNEL_CONFIG["fp64_thresholds"]["cluster_consistency"]
            ),
            "cluster_solar_safe": approach(
                "cluster_solar_safe",
                "cluster_dev",
                KERNEL_CONFIG["fp64_thresholds"]["cluster_consistency"],
            ),
            "lensing_among_galaxy_passers": approach(
                "lensing_among_galaxy_passers",
                "lensing_cons",
                KERNEL_CONFIG["fp64_thresholds"]["lensing_consistency"],
            ),
            "galaxy": approach(
                "galaxy", "flat_worst", KERNEL_CONFIG["fp64_thresholds"]["flatness"]
            ),
        }
        best_any = (
            refined[trackers["cluster_any"].ordinal]["cluster_dev"]
            if trackers["cluster_any"].ordinal is not None
            else float("inf")
        )
        best_safe = (
            refined[trackers["cluster_solar_safe"].ordinal]["cluster_dev"]
            if trackers["cluster_solar_safe"].ordinal is not None
            else float("inf")
        )
        if best_any <= tolerance < best_safe:
            direction = (
                "needed: density- or environment-dependent screening. The kernel family "
                "does express the cluster's radial boost shape (best unconstrained "
                f"cluster deviation {_text(best_any)} within tolerance), but every such "
                f"ordinal violates the Solar/Yukawa-safety gates (best solar-safe "
                f"deviation {_text(best_safe)}): the amplitude a cluster needs is the "
                "amplitude the Solar System forbids. The next structural axis is a "
                "screening mechanism (chameleon/Vainshtein-like) that suppresses the "
                "kernel in high-density or high-acceleration environments."
            )
        elif best_safe <= tolerance:
            direction = (
                "needed: sublinear source-mass response or environment-gated amplitude. "
                f"A solar-safe ordinal matches the cluster shape (deviation "
                f"{_text(best_safe)}), but no ordinal carries the galaxy and lensing "
                "gates simultaneously: a kernel linear in the source mass over-boosts "
                "or under-boosts the disks (BTFR forces slope 4 where linear kernels "
                "give 2)."
            )
        else:
            direction = (
                "needed: kernel anisotropy or density-dependent screening. Even with "
                "the Solar gates ignored, no isotropic distance kernel on the declared "
                f"grids reproduces the cluster radial shape (best deviation "
                f"{_text(best_any)} vs tolerance {_text(tolerance)}): the boost must "
                "depend on direction, environment, or source kinematics, not on "
                "separation distance alone."
            )
        best_exemplar = None
        safe_ordinal = trackers["cluster_solar_safe"].ordinal
        if safe_ordinal is not None:
            best_exemplar = next(
                (
                    entry
                    for entry in exact_verification
                    if entry["ordinal"] == safe_ordinal
                ),
                None,
            )
        sealed_negative = {
            "sealed": True,
            "statement": (
                "within the processed ordinal range, no candidate of the nonlocal "
                "kernel grammar passes the Solar, galaxy, lensing, and cluster gates "
                "jointly on the frozen synthetic controls"
            ),
            "closest_approach": closest,
            "best_cluster_shape_achiever": best_exemplar,
            "structural_direction": direction,
        }

    # CPU/GPU decision cross-check on a deterministic sample; the CPU decisions are
    # also the deterministic replay target for validation.
    rng = np.random.default_rng(KERNEL_CONFIG["crosscheck_seed"])
    sample = np.sort(
        rng.choice(
            total, size=min(KERNEL_CONFIG["crosscheck_sample"], total), replace=False
        )
    ).astype(np.int64)
    cpu_tables = _device_tables(tables, np)
    cpu_sample = evaluate_batch(
        np, sample, cpu_tables, dtype=np.float64, tier="fp64_thresholds"
    )
    sample_digest = canonical_sha256(
        {
            "ordinals": [int(v) for v in sample],
            **{
                f"{name}_pass": [bool(v) for v in cpu_sample[f"{name}_pass"]]
                for name in gate_names
            },
        }
    )
    crosscheck: dict[str, Any] = {
        "performed": gpu,
        "sample": int(sample.size),
        "sample_decisions_sha256": sample_digest,
    }
    if gpu:
        gpu_sample = evaluate_batch(
            xp, xp.asarray(sample), device_tables, dtype=xp.float64, tier="fp64_thresholds"
        )
        for name in gate_names:
            crosscheck[f"{name}_disagreements"] = int(
                (gpu_sample[f"{name}_pass"].get() != cpu_sample[f"{name}_pass"]).sum()
            )

    elapsed = time.perf_counter() - started
    all_gate_passers = len(passer_ordinals)
    if all_gate_passers:
        decision = (
            f"SCREENED: {all_gate_passers} candidates in {family_count} grid-neighbor "
            "equivalence families pass every gate on the synthetic controls; survivors "
            "are search priorities for the covariant lift, not validated theories"
        )
    else:
        margin = (
            sealed_negative["closest_approach"]["cluster_solar_safe"]["fp64_metric"]
            if sealed_negative and sealed_negative["closest_approach"]["cluster_solar_safe"]
            else None
        )
        decision = (
            "SCREENED-SEALED-NEGATIVE: no processed candidate passes all gates; "
            f"closest solar-safe cluster approach deviation {margin} against tolerance "
            f"{KERNEL_CONFIG['fp64_thresholds']['cluster_consistency']}"
        )

    body: dict[str, Any] = {
        "assumptions": {
            "lensing_prescription": GATE_CONFIG["lensing"]["prescription"],
            "cluster_control": (
                "the existing isothermal beta-model probe table and exact hydrostatic "
                "g_dyn; the kernel boost integrates the same gas density truncated at "
                "32 core radii (declared)"
            ),
            "newton_control": (
                "a declared unit point source: B/g_N = K(s)s^2 exactly, so the Solar "
                "gates constrain the kernel itself, independent of any extended control"
            ),
            "shared_code_units": (
                "disk Rd = cluster rc = Hernquist a = 1 in code units, so one kernel "
                "scale L means the same code length in every system; real galaxies and "
                "clusters are ~50x apart in physical scale. Survivors and negatives "
                "are statements about these synthetic controls only."
            ),
        },
        "claims": CLAIMS,
        "config": KERNEL_CONFIG,
        "config_sha256": canonical_sha256(KERNEL_CONFIG),
        "screen_config_sha256": canonical_sha256(SCREEN_CONFIG),
        "gate_config_sha256": canonical_sha256(GATE_CONFIG),
        "geometry_sha256": geometry_sha256(geometry),
        "geometry_newton_validation": newton_validation,
        "controls": controls,
        "counts": {
            "family_size": FAMILY_SIZE,
            "processed": processed,
            "fp32": fp32_counts,
            "fp32_all_pass_survivors": int(survivors32.size),
            "fp64_of_fp32_survivors": fp64_counts,
            "all_gate_passers": all_gate_passers,
            "passer_families": family_count,
            "passer_family_analysis_truncated": family_truncated,
            "exact_verified": len(exact_verification),
            "exact_confirmed": confirmed,
            "exact_refuted": len(exact_verification) - confirmed,
            "exact_verification_truncated": exact_truncated,
        },
        "crosscheck": crosscheck,
        "decision": decision,
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "sweep_elapsed_seconds": format(elapsed_sweep, ".3f"),
        "exact_verification": exact_verification,
        "passer_families_reported": families_block,
        "pareto_front": pareto_entries,
        "sealed_negative": sealed_negative,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "GPU screen of an ordinal-indexed nonlocal-kernel gravity grammar "
            "(boost = convolution of the baryonic source with a declared two-term "
            "kernel, times an optional embedded pointwise factor) against the frozen "
            "synthetic Solar, galaxy, lensing, and cluster controls of the existing "
            "screens. The grammar answers the sealed negatives of the pointwise "
            "billion-candidate screen and the Sigma-Gravity neighborhood scan, which "
            "identified the cluster boost's radial shape as unreachable by any local "
            "law. One shared constant per candidate, zero per-object freedom; no "
            "observational data opened; survivors are search priorities and a "
            "zero-survivor verdict is a sealed negative deliverable with margins and "
            "a named next structural direction."
        ),
        "throughput_candidates_per_second": (
            int(processed / elapsed_sweep) if elapsed_sweep > 0 else None
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Precompute-validation receipt
# ---------------------------------------------------------------------------


def _convergence_deltas() -> dict[str, str]:
    """Refined-geometry deltas for the two hand kernels; the quadrature convergence claim."""

    frozen = build_kernel_geometry(refine=1)
    refined = build_kernel_geometry(refine=2)
    deltas: dict[str, str] = {}
    for name in ("yukawa_hand", "power_hand"):
        values = decode_ordinal(CONTROL_ORDINALS[name])["values"]
        worst = mp.mpf(0)
        for system in ("disk", "hernquist", "cluster"):
            coarse = boost_exact(frozen[system], values)
            fine = boost_exact(refined[system], values)
            for a, b in zip(coarse, fine, strict=True):
                if abs(b) > mp.mpf(10) ** (-30):
                    worst = max(worst, abs(a / b - 1))
                else:
                    worst = max(worst, abs(a - b))
        deltas[name] = mp.nstr(worst, 6)
    return deltas


def build_precompute_receipt() -> dict[str, Any]:
    """Small sealed receipt: geometry hashes, Newton residuals, convergence evidence."""

    geometry = build_kernel_geometry()
    residuals = geometry_newton_residuals(geometry)
    budget = mp.mpf(KERNEL_CONFIG["geometry"]["newton_validation_max_relative_error"])
    for system, residual in residuals.items():
        if mp.mpf(residual) > budget:
            raise KernelScreenError(f"geometry validation failed for {system}: {residual}")
    deltas = _convergence_deltas()
    convergence_budget = mp.mpf(KERNEL_CONFIG["geometry"]["kernel_convergence_budget"])
    for name, delta in deltas.items():
        if mp.mpf(delta) > convergence_budget:
            raise KernelScreenError(f"kernel quadrature convergence failed for {name}: {delta}")
    body = {
        "schema_version": PRECOMPUTE_SCHEMA,
        "config_sha256": canonical_sha256(KERNEL_CONFIG),
        "geometry_sha256": geometry_sha256(geometry),
        "refined_geometry_sha256": geometry_sha256(build_kernel_geometry(refine=2)),
        "systems": {
            system: {
                "rows": len(rows),
                "nodes": sum(len(row["nodes"]) for row in rows),
                "newton_max_relative_error": residuals[system],
            }
            for system, rows in geometry.items()
        },
        "newton_validation_budget": KERNEL_CONFIG["geometry"][
            "newton_validation_max_relative_error"
        ],
        "kernel_convergence": {
            **deltas,
            "budget": KERNEL_CONFIG["geometry"]["kernel_convergence_budget"],
            "refinement": "ladder ratio 4 -> 2 and all panel orders doubled",
        },
        "scope": (
            "Validation of the frozen distance-distribution quadrature: the Newton "
            "kernel reproduces the frozen g_bar grids to the declared budget at every "
            "probe of every system, and the two hand-built kernels are stable under "
            "quadrature refinement."
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_precompute_receipt(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != PRECOMPUTE_SCHEMA:
        raise KernelScreenError("precompute receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise KernelScreenError("precompute receipt seal changed")
    if value.get("config_sha256") != canonical_sha256(KERNEL_CONFIG):
        raise KernelScreenError("precompute receipt config binding changed")
    geometry = build_kernel_geometry()
    if value.get("geometry_sha256") != geometry_sha256(geometry):
        raise KernelScreenError("frozen geometry does not replay")
    residuals = geometry_newton_residuals(geometry)
    for system, block in value.get("systems", {}).items():
        if block.get("newton_max_relative_error") != residuals.get(system):
            raise KernelScreenError(f"newton validation does not replay for {system}")


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, binding, geometry, control, and sample replay checks; fail closed."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise KernelScreenError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise KernelScreenError("receipt seal changed")
    if value.get("claims") != CLAIMS:
        raise KernelScreenError("claims block changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise KernelScreenError("config binding changed")
    if value.get("config_sha256") != canonical_sha256(KERNEL_CONFIG):
        raise KernelScreenError("receipt config does not match this module")
    if value.get("screen_config_sha256") != canonical_sha256(SCREEN_CONFIG):
        raise KernelScreenError("screen config binding changed")
    if value.get("gate_config_sha256") != canonical_sha256(GATE_CONFIG):
        raise KernelScreenError("gate config binding changed")
    context = build_exact_context()
    if value.get("geometry_sha256") != geometry_sha256(context["geometry"]):
        raise KernelScreenError("frozen geometry does not replay")
    if value.get("geometry_newton_validation") != geometry_newton_residuals(
        context["geometry"]
    ):
        raise KernelScreenError("geometry Newton validation does not replay")
    controls = {
        name: evaluate_candidate_exact(ordinal, context)
        for name, ordinal in CONTROL_ORDINALS.items()
    }
    _assert_known_answer_controls(controls)
    for name, verdict in controls.items():
        recorded = value.get("controls", {}).get(name, {})
        for gate in ("newton", "galaxy", "lensing", "cluster"):
            if recorded.get(gate, {}).get("passes") != verdict[gate]["passes"]:
                raise KernelScreenError(f"control replay changed for {name}.{gate}")
    for entry in value.get("exact_verification", []):
        if not entry.get("exact_confirmed", False):
            continue
        replay = evaluate_candidate_exact(entry["ordinal"], context)
        for gate in ("newton", "galaxy", "lensing", "cluster"):
            if replay[gate]["passes"] != entry[gate]["passes"]:
                raise KernelScreenError(
                    f"exact replay failed for ordinal {entry['ordinal']} at {gate}"
                )
    processed = value.get("counts", {}).get("processed")
    if not isinstance(processed, int) or processed < 1:
        raise KernelScreenError("receipt processed count is malformed")
    tables = build_sweep_tables(context["geometry"], context["pack"])
    rng = np.random.default_rng(KERNEL_CONFIG["crosscheck_seed"])
    sample = np.sort(
        rng.choice(
            processed,
            size=min(KERNEL_CONFIG["crosscheck_sample"], processed),
            replace=False,
        )
    ).astype(np.int64)
    cpu_sample = evaluate_batch(
        np, sample, _device_tables(tables, np), dtype=np.float64, tier="fp64_thresholds"
    )
    digest = canonical_sha256(
        {
            "ordinals": [int(v) for v in sample],
            **{
                f"{name}_pass": [bool(v) for v in cpu_sample[f"{name}_pass"]]
                for name in ("newton", "safety", "galaxy", "lensing", "cluster", "all")
            },
        }
    )
    if value.get("crosscheck", {}).get("sample_decisions_sha256") != digest:
        raise KernelScreenError("crosscheck sample decisions do not replay")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise KernelScreenError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Nonlocal-kernel gravity screen v2 (GPU, ordinal-indexed)."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1 << 21)
    parser.add_argument("--cpu", action="store_true", help="force the numpy path")
    parser.add_argument("--output")
    parser.add_argument("--precompute-output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        if not args.output and not args.precompute_output:
            raise KernelScreenError("--validate-checked requires --output or --precompute-output")
        if args.output:
            validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        if args.precompute_output:
            validate_precompute_receipt(
                json.loads(Path(args.precompute_output).read_text(encoding="utf-8"))
            )
        return 0
    result = run_screen(limit=args.limit, batch_size=args.batch_size, use_gpu=not args.cpu)
    if args.output:
        _write_receipt(result, args.output)
    if args.precompute_output:
        _write_receipt(build_precompute_receipt(), args.precompute_output)
    print(
        json.dumps(
            {
                "processed": result["counts"]["processed"],
                "fp32_all_pass_survivors": result["counts"]["fp32_all_pass_survivors"],
                "all_gate_passers": result["counts"]["all_gate_passers"],
                "passer_families": result["counts"]["passer_families"],
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
