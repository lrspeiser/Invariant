"""Screened-kernel gravity screen (v3): the grammar that answers v2's measured tension.

The v2 nonlocal-kernel screen (`runs/gpu-baryonic-screen/kernel-v2.json`) sealed a
negative with a precise shape.  Its kernel family *can* express the cluster's radial
boost (best unconstrained cluster deviation 0.0237 at ordinal 86,917,636), but every
ordinal that does so violates the Solar/Yukawa-safety gates (best solar-safe deviation
0.405): **the amplitude a cluster needs is the amplitude the Solar System forbids.**
v2's receipt named the next structural axis itself -- "density- or environment-dependent
screening -- chameleon/Vainshtein-like mechanism".  This module builds exactly that
grammar and screens it.

The law keeps v2's kernel ``K(s; theta)`` *verbatim* and multiplies the boost by a
screening factor that depends on the local environment of the **field point**, so the
same law can be loud in a cluster outskirt and silent in the Solar System::

    g_obs(x) = nu_loc(y) * g_N(x) + S(x) * Int rho_b(x') K(|x - x'|; theta) dV'
    K(s)     = (1/s^2) * [ w_Y * exp(-s/L1)
                           + w_P * (s/L2)^p * (1 + s/L2)^(-(p+t)) ]
    S(x)     = [ 1 + X(x)^k ]^(-1)

with the screening argument ``X`` drawn from a declared three-family axis (plus
``none``, under which ``S == 1`` and the law is *exactly* v2):

  (a) DENSITY / chameleon-like       X = rho_local / rho_star
  (b) ACCELERATION / Vainshtein-like X = g_N / g_star
  (c) CURVATURE-PROXY                X = |grad g_N| * Lc / g_N

``rho_local`` is written throughout in the Poisson-source convention
``rho_local := 4 pi G rho_b`` (G = 1 in code units), which removes every factor of pi
from the declared grids; ``|grad g_N|`` is the radial derivative ``|d g_N/dr|`` of the
Newtonian field magnitude, the natural scalar for the spherically symmetric and
in-plane-axisymmetric controls.

**The enabling trick survives.**  ``S`` depends only on the field point and the three
screening parameters -- never on the kernel parameters -- so the boost stays linear in
``(w_Y, w_P)`` and the whole family still reduces to per-probe basis tables, now carrying
one extra (screening) index.  The 8.35e8-ordinal sweep is table combination, not
quadrature.  The frozen 46,392-node distance-distribution tables, the four gates, every
threshold, the fp32/fp64/mpmath ladder, and the receipt discipline are v2's, imported and
reused rather than reimplemented.

**Screening is applied in every gate.**  The galaxy boost, the *lensing* path integral
(photons see the screened field, node by node), the cluster probes and the Solar
point-source probes all multiply by ``S`` evaluated at the field point in question.  No
gate quietly uses the unscreened boost, and the receipt carries a control candidate whose
lensing verdict flips when the screening index is moved to ``none``.

**Declared environment of the Solar probes.**  The Newton/Solar control is a unit point
mass probed at ``y = 1e4, 1e6`` (plus ``y = 1e8`` report-only and the ``s = 1e-5`` safety
probe).  Its ``g_N`` and ``|grad g_N|`` come from the point mass itself.  Its
``rho_local`` cannot: a point source is surrounded by vacuum, and a vacuum probe is
unscreenable by construction in family (a).  The declared choice is that the Solar probe
sits inside a galaxy, at the frozen inner radius ``r = 3`` of the largest control disk
(the Sun sits at ~2.7 disk scale lengths; 3 is the nearest frozen inner probe of the
existing screen, so no new number enters).  This is an assumption, flagged in the claims
block, and it is what gives family (a) a fair rather than a rigged test.

**Known-answer controls (run-aborting).**  v2 is embedded as the leading 5,760,000
ordinals (screening index 0 = ``none``): the four v2 closest-approach exemplars are
re-evaluated through the v3 pipeline and must reproduce v2's *recorded digits*.  Two of
them are on the trimmed v3 grid and are additionally checked by ordinal round-trip; the
other two carry eighth-grid amplitudes that the trimmed grid does not contain, so they
are evaluated from their exact parameter values -- a strictly stronger check than a grid
lookup.  Newton recovery, the two hand kernels, a strong-screening ordinal that must pass
Solar and fail galaxy, and an unscreened strong kernel that must fail Solar complete the
calibration.

**The tension map.**  Regardless of the verdict the receipt carries a monotone frontier
table: the best achievable cluster deviation as a function of how badly the Solar System
is violated (``solar_ratio`` = the worst of the three Solar margins in units of its own
threshold).  That turns v2's headline sentence from an anecdote into a measured object.

One declared limitation is inherited unchanged: the synthetic controls share one code-unit
length scale (disk Rd = cluster rc = Hernquist a = 1), so a kernel scale L means the same
code length in every system while real galaxies and clusters are ~50x apart.  Survivors
and negatives are statements about these controls, not about nature.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from .gpu_baryonic_interpolation_screen import SCREEN_CONFIG
from .gpu_baryonic_kernel_screen import (
    KERNEL_CONFIG as V2_CONFIG,
)
from .gpu_baryonic_kernel_screen import (
    _fraction,
    _kernel_exact,
    _nu_local_exact,
    boost_exact,
    build_exact_pack,
    build_kernel_geometry,
    emit_covariant_lift,
    geometry_newton_residuals,
    geometry_sha256,
)
from .gpu_baryonic_lensing_cluster_screen import GATE_CONFIG
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-gpu-screened-kernel-screen-result-1.0"
PRECOMPUTE_SCHEMA = "invariant-gpu-screened-kernel-screen-precompute-1.0"


# ---------------------------------------------------------------------------
# Declared axes
# ---------------------------------------------------------------------------

#: Screening families.  ``none`` embeds v2 exactly (S == 1 everywhere).
SCREEN_FAMILIES = ("none", "density", "acceleration", "curvature")

#: One log grid per family, 12 points each, on the exact half-decade {1,3}x10^n ladder.
#: Density/acceleration scales divide the environment (larger scale = less screening);
#: the curvature length Lc multiplies it (larger Lc = more screening).
SCREEN_SCALES: dict[str, tuple[str, ...]] = {
    "density": (
        "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3",
        "1e-2", "3e-2", "1e-1", "3e-1", "1", "3",
    ),
    "acceleration": (
        "1e-2", "3e-2", "1e-1", "3e-1", "1", "3",
        "10", "30", "100", "300", "1000", "3000",
    ),
    "curvature": (
        "1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1",
        "1", "3", "10", "30", "100", "300",
    ),
}

#: Screening sharpness.
SCREEN_SHARPNESS = ("1", "2", "3", "4")


def _build_screen_entries() -> tuple[tuple[str, str, str], ...]:
    """(family, scale, sharpness) triples; entry 0 is ``none`` (no duplicates)."""

    entries: list[tuple[str, str, str]] = [("none", "0", "0")]
    for family in SCREEN_FAMILIES[1:]:
        for scale in SCREEN_SCALES[family]:
            for sharpness in SCREEN_SHARPNESS:
                entries.append((family, scale, sharpness))
    return tuple(entries)


#: 1 + 3 * 12 * 4 = 145 screening configurations, degeneracy-free.
SCREEN_ENTRIES = _build_screen_entries()

AXIS_ORDER = ("screen", "local", "w_yukawa", "L1", "w_power", "L2", "p", "t")
AXES: dict[str, tuple[str, ...]] = {
    "screen": tuple(f"{family}:{scale}:{k}" for family, scale, k in SCREEN_ENTRIES),
    "local": ("identity", "one_plus_u", "sqrt_one_plus_u_squared"),
    "w_yukawa": tuple(str(Fraction(k, 4)) for k in range(25)),
    "L1": ("1/2", "1", "2", "3", "4", "6", "12", "24"),
    "w_power": tuple(str(Fraction(k, 4)) for k in range(25)),
    "L2": ("1/2", "1", "2", "3", "4", "6", "12", "24"),
    "p": ("1/4", "1/2", "3/4", "1", "3/2", "2", "5/2", "3", "7/2", "4", "9/2", "5"),
    "t": ("-1", "0", "1/2", "1"),
}
AXIS_SIZES = tuple(len(AXES[name]) for name in AXIS_ORDER)
FAMILY_SIZE = math.prod(AXIS_SIZES)  # 145*3*25*8*25*8*12*4 = 835,200,000
KERNEL_SUBFAMILY_SIZE = FAMILY_SIZE // len(AXES["screen"])  # the embedded v2 block
N_POWER_COMBOS = len(AXES["L2"]) * len(AXES["p"]) * len(AXES["t"])  # 384

SYSTEM_CAPS = {
    "min_batch_size": 1 << 10,
    "max_batch_size": 1 << 23,
    "max_pareto_reported": 64,
    "max_exact_verifications": 256,
    "max_families_reported": 16,
    "family_analysis_cap": 100_000,
    "bulk_passer_sample": 32,
}

#: The monotone tension-map ladder: bounds on ``solar_ratio`` (the worst Solar margin in
#: units of its own fp64 threshold).  ``solar_ratio <= 1`` means fully Solar-safe.
TENSION_MAP_BOUNDS = (
    "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1",
    "1e1", "1e2", "1e3", "1e4", "1e6", "1e8", "1e10",
)

#: Frozen screen configuration.  Changing any value changes the claim and the receipt
#: hash.  Every gate threshold is v2's verbatim; the new material is the screening axis
#: and the environment-field declarations.
SCREENED_CONFIG: dict[str, Any] = {
    "a0": 1,
    "mpmath_dps": 50,
    "law": (
        "g_obs = nu_loc(g_N/a0) * g_N + S(x) * B_raw;  "
        "B_raw(x) = Int rho_b(x') K(|x-x'|) dV';  "
        "K(s) = (1/s^2) * [w_Y*exp(-s/L1) + w_P*(s/L2)^p*(1+s/L2)^(-(p+t))];  "
        "S(x) = [1 + X(x)^k]^(-1)"
    ),
    "screening": {
        "families": {
            "none": "S = 1 (the embedded v2 grammar; mandatory control)",
            "density": "X = rho_local/rho_star  (chameleon-like: screens dense regions)",
            "acceleration": "X = g_N/g_star  (Vainshtein-like: screens high-acceleration regions)",
            "curvature": (
                "X = |grad g_N| * Lc / g_N  (screens where the Newtonian field varies sharply)"
            ),
        },
        "scale_grids": {name: list(values) for name, values in SCREEN_SCALES.items()},
        "sharpness_grid": list(SCREEN_SHARPNESS),
        "screen_axis_size": len(SCREEN_ENTRIES),
        "density_convention": (
            "rho_local := 4*pi*G*rho_b with G = 1 in code units (the Poisson source "
            "density), so every declared grid value is pi-free"
        ),
        "gradient_convention": (
            "|grad g_N| := |d g_N/dr|, the radial derivative of the Newtonian field "
            "magnitude; exact for the spherical controls and for in-plane disk probes"
        ),
        "disk_screening_thickness": "1/10",
        "disk_screening_thickness_note": (
            "the razor-thin Freeman disk has no volume density; for the screening field "
            "only, its surface density is smeared over a declared full thickness 2h with "
            "h = 1/10 Rd, so rho_local = Sigma/(2h).  The kernel geometry stays razor-thin."
        ),
        "solar_ambient_source": (
            "the Newton/Solar point-source probes sit inside a galaxy: rho_local there is "
            "the largest control disk (mass 128/125) at the frozen inner radius 3 of the "
            "existing screen (the Sun sits at ~2.7 disk scale lengths).  g_N and "
            "|grad g_N| at those probes come from the point mass itself.  Declared "
            "assumption, not a derivation."
        ),
        "solar_ambient_radius": 3,
        "applied_in_gates": ["newton", "safety", "galaxy", "lensing", "cluster"],
    },
    "axis_order": list(AXIS_ORDER),
    "axes": {name: list(values) for name, values in AXES.items()},
    "family_size": FAMILY_SIZE,
    "embedded_v2_block_size": KERNEL_SUBFAMILY_SIZE,
    "geometry": V2_CONFIG["geometry"],
    "environment_validation": {
        "max_relative_error": "1e-10",
        "finite_difference_dps": 80,
        "finite_difference_step": "1e-25",
        "density_route": (
            "spheres: 4 pi rho = (1/r^2) d/dr [r^2 g_N] by high-precision central "
            "difference of the frozen Newtonian field; disk: Sigma = (1/(2 pi r)) dM/dr "
            "by high-precision central difference of the frozen enclosed mass"
        ),
        "gradient_route": (
            "closed form versus a high-precision central difference of g_N at the "
            "declared step and working precision"
        ),
    },
    "newton_control": V2_CONFIG["newton_control"],
    "fp32_thresholds": V2_CONFIG["fp32_thresholds"],
    "fp64_thresholds": V2_CONFIG["fp64_thresholds"],
    "solar_block_precision": (
        "the whole Solar block (near, far, report, safety) is evaluated in fp64 in both "
        "tiers; only the thresholds differ, so the fp32 tier stays a strict superset"
    ),
    "tension_map_bounds": list(TENSION_MAP_BOUNDS),
    "crosscheck_sample": 2048,
    "crosscheck_seed": 20260816,
}

CLAIMS = {
    "cluster_negative_is_a_valid_deliverable": True,
    "corpus_absence_establishes_novelty": False,
    "embedded_v2_family_reproduces_prior_negative": True,
    "first_principles_derivation_claimed": False,
    "invisible_mass_used_as_target_or_rescue": False,
    "kernel_scales_share_code_units_across_systems": True,
    "lensing_prescription_is_an_assumption": True,
    "observational_data_opened": False,
    "per_object_free_parameters_expressible": False,
    "scalar_truth_or_probability_score": False,
    "screening_applied_in_every_gate_including_lensing": True,
    "screening_is_phenomenological_not_derived": True,
    "sealed_validation_ladder_bypassed": False,
    "solar_probe_ambient_density_is_a_declared_assumption": True,
    "survivor_is_validated_theory": False,
    "synthetic_controls_only": True,
}

#: v2's four recorded closest approaches.  The embedded-v2 control re-evaluates each of
#: them through the v3 pipeline at screening index 0 and demands v2's recorded digits.
#: ``exact`` entries were recorded by v2's 50-digit layer (string equality required);
#: ``fp64`` entries were recorded by v2's float64 batch layer (9-digit agreement).
V2_RECORDED: dict[str, dict[str, Any]] = {
    "cluster_any": {
        "v2_ordinal": 86917636,
        "values": {
            "local": "one_plus_u", "w_yukawa": "47/8", "L1": "1",
            "w_power": "41/8", "L2": "1/2", "p": "1/4", "t": "1",
        },
        "tier": "fp64",
        "metrics": {
            "cluster_dev": "2.372244161e-02",
            "flat_worst": "8.108148123e-02",
            "btfr_err": "1.384239599e-01",
            "lensing_cons": "1.742199260e-01",
            "lensing_flat": "5.011178214e-02",
            "newton_near": "7.706730243e+00",
            "newton_far": "6.951230051e+00",
            "safety_margin": "6.217662089e+00",
        },
    },
    "cluster_solar_safe": {
        "v2_ordinal": 44329019,
        "values": {
            "local": "one_plus_u", "w_yukawa": "0", "L1": "1/2",
            "w_power": "6", "L2": "1/2", "p": "2", "t": "1/2",
        },
        "tier": "exact",
        "metrics": {
            "cluster_dev": "4.050292865e-01",
            "cluster_closest_probe": "8.744436394e-02",
            "flat_worst": "1.102681038e-01",
            "btfr_slope": "3.780138058e+00",
            "btfr_err": "2.198619425e-01",
            "lensing_cons": "2.104764831e-01",
            "lensing_flat": "8.734259361e-02",
            "newton_near": "1.228407738e-02",
            "newton_far": "1.023880419e-03",
            "newton_report_y8": "1.002398800e-04",
            "safety_margin": "2.399880004e-09",
        },
        "cluster_ratio_by_probe": [
            "5.949707135e-01", "6.873838984e-01", "9.125556361e-01",
            "1.176597895e+00", "1.379214556e+00",
        ],
    },
    "galaxy": {
        "v2_ordinal": 88568552,
        "values": {
            "local": "sqrt_one_plus_u_squared", "w_yukawa": "0", "L1": "1/2",
            "w_power": "37/8", "L2": "12", "p": "7/2", "t": "-1",
        },
        "tier": "fp64",
        "metrics": {
            "cluster_dev": "7.524788061e-01",
            "flat_worst": "1.818145401e-02",
            "btfr_err": "9.859662934e-02",
            "lensing_cons": "1.164332352e-01",
            "lensing_flat": "7.287388643e-02",
            "newton_near": "4.999882717e-05",
            "newton_far": "4.999998994e-07",
            "safety_margin": "2.443298178e-21",
        },
    },
    "lensing_among_galaxy_passers": {
        "v2_ordinal": 94380536,
        "values": {
            "local": "sqrt_one_plus_u_squared", "w_yukawa": "3/4", "L1": "3",
            "w_power": "6", "L2": "6", "p": "4", "t": "-1",
        },
        "tier": "fp64",
        "metrics": {
            "cluster_dev": "7.963416431e-01",
            "flat_worst": "7.929311687e-02",
            "btfr_err": "3.339707547e-01",
            "lensing_cons": "1.154747818e-01",
            "lensing_flat": "1.604446294e-01",
            "newton_near": "7.475541608e-01",
            "newton_far": "7.497505417e-01",
            "safety_margin": "7.499975000e-01",
        },
    },
}

#: v2's sealed headline, restated so the receipt records what v3 is answering.
V2_TENSION_STATEMENT = (
    "v2 sealed negative: best unconstrained cluster deviation 2.372244161e-02 (inside "
    "the 15e-2 tolerance) but best solar-safe cluster deviation 4.050292865e-01 -- the "
    "amplitude a cluster needs is the amplitude the Solar System forbids"
)


class ScreenedKernelError(ValueError):
    """Raised on malformed input, a broken known-answer control, or receipt tamper."""


def _dps() -> None:
    mp.mp.dps = SCREENED_CONFIG["mpmath_dps"]


def _text(value: Any) -> str:
    return format(float(value), ".9e")


# ---------------------------------------------------------------------------
# Ordinal codec
# ---------------------------------------------------------------------------


def decode_ordinal(ordinal: int) -> dict[str, Any]:
    """Ordinal -> {indices, values}; ``screen`` is the most significant digit."""

    if not 0 <= ordinal < FAMILY_SIZE:
        raise ScreenedKernelError(f"ordinal out of range: {ordinal}")
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
        raise ScreenedKernelError("need exactly one index per axis")
    ordinal = 0
    for index, size in zip(indices, AXIS_SIZES, strict=True):
        if not 0 <= index < size:
            raise ScreenedKernelError(f"axis index out of range: {index}")
        ordinal = ordinal * size + index
    return ordinal


def screen_label(family: str, scale: str = "0", sharpness: str = "0") -> str:
    label = f"{family}:{scale}:{sharpness}"
    if label not in AXES["screen"]:
        raise ScreenedKernelError(f"screen value not on the grid: {label}")
    return label


def encode_named(**named: str) -> int:
    """Encode from axis-value strings (defaults: no screening, Newton kernel)."""

    defaults = {
        "screen": "none:0:0",
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
            raise ScreenedKernelError(f"{name} value not on the grid: {value}")
        indices.append(AXES[name].index(value))
    return encode_indices(indices)


def screen_entry(values: Mapping[str, str]) -> tuple[str, str, str]:
    family, scale, sharpness = values["screen"].split(":")
    return family, scale, sharpness


def render_screening(values: Mapping[str, str]) -> str:
    family, scale, sharpness = screen_entry(values)
    if family == "none":
        return "S = 1"
    argument = {
        "density": f"rho_local/{scale}",
        "acceleration": f"g_N/{scale}",
        "curvature": f"{scale}*|grad g_N|/g_N",
    }[family]
    return f"S = [1 + ({argument})^{sharpness}]^-1"


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
    return (
        f"g_obs = [{local}]*g_N + S(x)*conv(rho_b, K);  K(s) = ({kernel})/s^2,  "
        f"u = (g_N/a0)^(-1/2);  {render_screening(values)}"
    )


def kernel_values(values: Mapping[str, str]) -> dict[str, str]:
    """The v2 sub-dictionary (everything except the screening axis)."""

    return {name: values[name] for name in AXIS_ORDER if name != "screen"}


# ---------------------------------------------------------------------------
# Environment fields: rho_local, g_N, |grad g_N| at every probe, 50 digits
# ---------------------------------------------------------------------------

_ENVIRONMENT_CACHE: dict[int, dict[str, Any]] = {}


def _disk_g_unit(radius: mp.mpf) -> mp.mpf:
    """Unit-mass Freeman disk in-plane g_N (Rd = 1)."""

    y = radius / 2
    return y * (
        mp.besseli(0, y) * mp.besselk(0, y) - mp.besseli(1, y) * mp.besselk(1, y)
    )


def _disk_grad_unit(radius: mp.mpf) -> mp.mpf:
    """Closed-form d/dr of `_disk_g_unit`, from the Bessel derivative identities."""

    y = radius / 2
    i0, i1 = mp.besseli(0, y), mp.besseli(1, y)
    k0, k1 = mp.besselk(0, y), mp.besselk(1, y)
    return (i0 * k0 + i1 * k1 + 2 * y * (i1 * k0 - i0 * k1)) / 2


def _disk_rho_unit(radius: mp.mpf) -> mp.mpf:
    """Unit-mass 4 pi G rho for the smeared razor-thin disk: exp(-r) / h."""

    thickness = _fraction(SCREENED_CONFIG["screening"]["disk_screening_thickness"])
    return mp.e ** (-radius) / thickness


def _hernquist_g_unit(radius: mp.mpf) -> mp.mpf:
    return 1 / (radius + 1) ** 2


def _hernquist_grad_unit(radius: mp.mpf) -> mp.mpf:
    return 2 / (radius + 1) ** 3


def _hernquist_rho_unit(radius: mp.mpf) -> mp.mpf:
    return 2 / (radius * (1 + radius) ** 3)


def _cluster_g(radius: mp.mpf) -> mp.mpf:
    amplitude = mp.mpf(V2_CONFIG["geometry"]["cluster"]["amplitude_4pi_rho0"])
    return amplitude * (radius - mp.atan(radius)) / radius**2


def _cluster_grad(radius: mp.mpf) -> mp.mpf:
    amplitude = mp.mpf(V2_CONFIG["geometry"]["cluster"]["amplitude_4pi_rho0"])
    return amplitude * (
        1 / (1 + radius * radius) - 2 * (radius - mp.atan(radius)) / radius**3
    )


def _cluster_rho(radius: mp.mpf) -> mp.mpf:
    amplitude = mp.mpf(V2_CONFIG["geometry"]["cluster"]["amplitude_4pi_rho0"])
    return amplitude / (1 + radius * radius)


_UNIT_FIELDS = {
    "disk": (_disk_g_unit, _disk_grad_unit, _disk_rho_unit),
    "hernquist": (_hernquist_g_unit, _hernquist_grad_unit, _hernquist_rho_unit),
    "cluster": (_cluster_g, _cluster_grad, _cluster_rho),
}


def build_environment(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Unit-mass ``(g_N, |grad g_N|, rho_local, curv)`` at every frozen probe row.

    ``curv = |grad g_N| / g_N`` is mass independent (both scale linearly with the source
    mass), so the curvature family needs no per-mass table; density and acceleration do.
    The cluster carries its declared amplitude, so its "unit" values are its actual ones.
    """

    key = id(geometry)
    if key in _ENVIRONMENT_CACHE:
        return _ENVIRONMENT_CACHE[key]
    _dps()
    environment: dict[str, Any] = {}
    for system, rows in geometry.items():
        g_fn, grad_fn, rho_fn = _UNIT_FIELDS[system]
        entries = []
        for row in rows:
            radius = row["radius"]
            g_value = g_fn(radius)
            if g_value <= 0:
                raise ScreenedKernelError(f"non-positive g_N at {system} r={radius}")
            entries.append(
                {
                    "radius": radius,
                    "g": g_value,
                    "grad": abs(grad_fn(radius)),
                    "rho": rho_fn(radius),
                    "curv": abs(grad_fn(radius)) / g_value,
                }
            )
        environment[system] = entries

    # The declared Solar ambient: largest control disk at the frozen inner radius 3.
    ambient_radius = mp.mpf(SCREENED_CONFIG["screening"]["solar_ambient_radius"])
    if int(ambient_radius) not in SCREEN_CONFIG["inner_radii"]:
        raise ScreenedKernelError("solar ambient radius is not a frozen inner probe")
    ambient_mass = _fraction(SCREEN_CONFIG["disk_masses"][-1])
    environment["solar_ambient_rho"] = ambient_mass * _disk_rho_unit(ambient_radius)
    _ENVIRONMENT_CACHE[key] = environment
    return environment


def _central_difference(function: Any, point: mp.mpf, dps: int, step: mp.mpf) -> mp.mpf:
    previous = mp.mp.dps
    mp.mp.dps = dps
    try:
        return (function(point + step) - function(point - step)) / (2 * step)
    finally:
        mp.mp.dps = previous


def environment_residuals(geometry: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Independent-route validation of the two new fields, per system.

    ``rho`` is re-derived from the frozen Newtonian field (spheres: the Poisson
    divergence ``(1/r^2) d/dr[r^2 g_N]``; disk: ``Sigma = (1/(2 pi r)) dM/dr`` from the
    frozen enclosed mass), and ``grad`` is re-derived from a high-precision central
    difference of ``g_N``.  Both routes use only the declared source models, never the
    closed forms under test.
    """

    _dps()
    config = SCREENED_CONFIG["environment_validation"]
    dps = int(config["finite_difference_dps"])
    step = mp.mpf(config["finite_difference_step"])
    thickness = _fraction(SCREENED_CONFIG["screening"]["disk_screening_thickness"])
    environment = build_environment(geometry)
    residuals: dict[str, dict[str, str]] = {}
    for system, entries in environment.items():
        if system == "solar_ambient_rho":
            continue
        g_fn = _UNIT_FIELDS[system][0]
        worst_rho = mp.mpf(0)
        worst_grad = mp.mpf(0)
        for entry in entries:
            radius = entry["radius"]
            derivative = _central_difference(g_fn, radius, dps, step)
            worst_grad = max(worst_grad, abs(abs(derivative) / entry["grad"] - 1))
            if system == "disk":
                # Sigma from the frozen enclosed mass M(<R) = 1 - (1+R) exp(-R).
                enclosed = _central_difference(
                    lambda r: 1 - (1 + r) * mp.e ** (-r), radius, dps, step
                )
                sigma = enclosed / (2 * mp.pi * radius)
                rho_check = 4 * mp.pi * sigma / (2 * thickness)
            else:
                flux = _central_difference(
                    lambda r, fn=g_fn: r * r * fn(r), radius, dps, step
                )
                rho_check = flux / (radius * radius)
            worst_rho = max(worst_rho, abs(rho_check / entry["rho"] - 1))
        residuals[system] = {
            "rho_max_relative_error": mp.nstr(worst_rho, 6),
            "grad_max_relative_error": mp.nstr(worst_grad, 6),
        }
    return residuals


def environment_sha256(environment: Mapping[str, Any]) -> str:
    """Deterministic content hash of the frozen environment fields (40-digit strings)."""

    payload: dict[str, Any] = {}
    for system, entries in sorted(environment.items()):
        if system == "solar_ambient_rho":
            payload[system] = mp.nstr(entries, 40)
            continue
        payload[system] = [
            {
                "radius": mp.nstr(entry["radius"], 40),
                "g": mp.nstr(entry["g"], 40),
                "grad": mp.nstr(entry["grad"], 40),
                "rho": mp.nstr(entry["rho"], 40),
            }
            for entry in entries
        ]
    return canonical_sha256(payload)


# ---------------------------------------------------------------------------
# The screening factor
# ---------------------------------------------------------------------------


def screening_argument(
    entry: tuple[str, str, str],
    environment_entry: Mapping[str, Any],
    mass: mp.mpf,
) -> mp.mpf | None:
    """``X`` for one screening configuration at one probe; ``None`` for ``none``."""

    family, scale_text, _ = entry
    if family == "none":
        return None
    scale = mp.mpf(scale_text)
    if family == "density":
        return mass * environment_entry["rho"] / scale
    if family == "acceleration":
        return mass * environment_entry["g"] / scale
    if family == "curvature":
        return environment_entry["curv"] * scale
    raise ScreenedKernelError(f"unknown screening family: {family}")


def screen_factor_exact(
    entry: tuple[str, str, str],
    environment_entry: Mapping[str, Any],
    mass: mp.mpf,
) -> mp.mpf:
    """``S = [1 + X^k]^(-1)`` at 50 digits (exactly 1 for the ``none`` family)."""

    argument = screening_argument(entry, environment_entry, mass)
    if argument is None:
        return mp.mpf(1)
    sharpness = int(entry[2])
    return 1 / (1 + argument**sharpness)


def _point_environment(separation: mp.mpf, ambient_rho: mp.mpf) -> dict[str, mp.mpf]:
    """Environment at a Solar probe: point-source field, declared ambient density."""

    g_value = 1 / (separation * separation)
    return {
        "g": g_value,
        "grad": 2 / separation**3,
        "rho": ambient_rho,
        "curv": 2 / separation,
    }


def _screen_matrix(
    entries: Sequence[Mapping[str, Any]], mass: mp.mpf
) -> np.ndarray:
    """(145, rows) float64 screening factors for one system at one source mass."""

    table = np.empty((len(SCREEN_ENTRIES), len(entries)), dtype=np.float64)
    for index, entry in enumerate(SCREEN_ENTRIES):
        for row, environment_entry in enumerate(entries):
            table[index, row] = float(screen_factor_exact(entry, environment_entry, mass))
    return table


# ---------------------------------------------------------------------------
# Exact gate context and evaluation (mpmath, 50 digits)
# ---------------------------------------------------------------------------


def build_exact_context() -> dict[str, Any]:
    """Geometry, gate pack, and environment fields; the single source for exactness."""

    geometry = build_kernel_geometry()
    pack = build_exact_pack(geometry)
    environment = build_environment(geometry)
    _dps()
    newton_config = SCREENED_CONFIG["newton_control"]
    ambient = environment["solar_ambient_rho"]
    point_environments = [
        _point_environment(separation, ambient) for _, separation in pack["newton"]
    ]
    safety_environment = _point_environment(pack["safety_s"], ambient)
    return {
        "geometry": geometry,
        "pack": pack,
        "environment": environment,
        "newton_environments": point_environments,
        "safety_environment": safety_environment,
        "report_probe_y": newton_config["report_probe_y"],
    }


#: Bounded memo for the expensive 50-digit node sums.  The screening factor never
#: touches the kernel parameters, so one raw boost per (system, kernel) serves all three
#: source masses, both the campaign and the receipt replay.
_RAW_BOOST_CACHE: dict[tuple[Any, ...], list[mp.mpf]] = {}
_RAW_BOOST_CACHE_LIMIT = 512


def _raw_boost(
    system: str, rows: Sequence[Mapping[str, Any]], values: Mapping[str, str]
) -> list[mp.mpf]:
    """Unscreened ``Int rho_b K`` per row -- v2's `boost_exact`, memoized per system."""

    kernel = kernel_values(values)
    key = (system, len(rows), *(kernel[name] for name in sorted(kernel)))
    cached = _RAW_BOOST_CACHE.get(key)
    if cached is None:
        cached = boost_exact(rows, kernel)
        if len(_RAW_BOOST_CACHE) >= _RAW_BOOST_CACHE_LIMIT:
            _RAW_BOOST_CACHE.pop(next(iter(_RAW_BOOST_CACHE)))
        _RAW_BOOST_CACHE[key] = cached
    return cached


def _apply_screening(
    raw: Sequence[mp.mpf],
    environment_entries: Sequence[Mapping[str, Any]],
    entry: tuple[str, str, str],
    mass: mp.mpf,
) -> list[mp.mpf]:
    """``S(r) * B_raw(r)`` at every row of one geometry system."""

    return [
        screen_factor_exact(entry, environment_entry, mass) * value
        for value, environment_entry in zip(raw, environment_entries, strict=True)
    ]


def _boost_exact_screened(
    system: str,
    rows: Sequence[Mapping[str, Any]],
    environment_entries: Sequence[Mapping[str, Any]],
    values: Mapping[str, str],
    entry: tuple[str, str, str],
    mass: mp.mpf,
) -> list[mp.mpf]:
    """Screened boost ``S(r) * Int rho_b K`` at every row of one geometry system."""

    return _apply_screening(
        _raw_boost(system, rows, values), environment_entries, entry, mass
    )


def evaluate_values_exact(
    values: Mapping[str, str], context: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-run every gate for one candidate at 50 digits, with decimal-string margins.

    ``values`` may carry off-grid kernel amplitudes: the embedded-v2 control uses that to
    evaluate v2's recorded exemplars whose eighth-grid weights the trimmed v3 axes do not
    contain.  Screening enters every gate -- galaxy, lensing (node by node), cluster, and
    the Solar point probes.
    """

    _dps()
    geometry = context["geometry"]
    pack = context["pack"]
    environment = context["environment"]
    entry = screen_entry(values)
    local = values["local"]
    thresholds = {
        key: mp.mpf(value) for key, value in SCREENED_CONFIG["fp64_thresholds"].items()
    }

    # The three expensive 50-digit node sums, each computed exactly once.
    raw_disk = _raw_boost("disk", geometry["disk"], values)
    raw_hernquist = _raw_boost("hernquist", geometry["hernquist"], values)
    raw_cluster = _raw_boost("cluster", geometry["cluster"], values)

    # Galaxy: screened boost per (disk mass, probe).
    galaxy_valid = True
    flat_worst = mp.mpf(0)
    vflat: list[mp.mpf] = []
    per_disk = []
    for disk in pack["galaxy"]:
        boost = _apply_screening(raw_disk, environment["disk"], entry, disk["mass"])
        speeds = []
        for k, (radius, gbar) in enumerate(disk["points"]):
            v2_value = (
                _nu_local_exact(local, gbar) * gbar + disk["mass"] * boost[k]
            ) * radius
            if v2_value <= 0:
                galaxy_valid = False
                break
            speeds.append(mp.sqrt(v2_value))
        if not galaxy_valid:
            break
        mean = sum(speeds) / len(speeds)
        spread = (max(speeds) - min(speeds)) / mean
        flat_worst = max(flat_worst, spread)
        vflat.append(mean)
        per_disk.append(
            {"mass_text": disk["mass_text"], "spread": _text(spread), "v_flat": _text(mean)}
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

    # Lensing: the screening multiplies the boost at every path node.
    lens_valid = galaxy_valid
    worst_flat = mp.mpf(0)
    worst_cons = mp.mpf(0)
    if lens_valid:
        boost_by_mass: dict[str, list[mp.mpf]] = {}
        alphas_by_mass: dict[int, list[mp.mpf]] = {}
        for pair in pack["lensing"]:
            if pair["mass_text"] not in boost_by_mass:
                boost_by_mass[pair["mass_text"]] = _apply_screening(
                    raw_hernquist, environment["hernquist"], entry, pair["mass"]
                )
            boost = boost_by_mass[pair["mass_text"]]
            total = mp.fsum(
                gw * (_nu_local_exact(local, y) * y + pair["mass"] * boost[row])
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

    # Cluster: the decisive 5-probe hydrostatic criterion, screened.
    cluster_boost = _apply_screening(
        raw_cluster, environment["cluster"], entry, mp.mpf(1)
    )
    ratios = []
    shortfalls = []
    for k, (gbar, gdyn) in enumerate(pack["cluster"]):
        g_obs = _nu_local_exact(local, gbar) * gbar + cluster_boost[k]
        ratios.append(g_obs / gdyn)
        shortfalls.append(gdyn / g_obs if g_obs > 0 else mp.inf)
    deviations = [abs(ratio - 1) for ratio in ratios]
    cluster_passes = bool(max(deviations) <= thresholds["cluster_consistency"])

    # Solar: screened point-source probes plus the screened safety margin.
    kernel = kernel_values(values)
    has_kernel = values["w_yukawa"] != "0" or values["w_power"] != "0"
    newton_entries = []
    screen_factors = []
    for index, (y, s) in enumerate(pack["newton"]):
        factor = screen_factor_exact(entry, context["newton_environments"][index], mp.mpf(1))
        screen_factors.append(factor)
        boost_ratio = factor * _kernel_exact(s, kernel) * s * s if has_kernel else mp.mpf(0)
        newton_entries.append(abs((_nu_local_exact(local, y) - 1) + boost_ratio))
    safety_s = pack["safety_s"]
    safety_factor = screen_factor_exact(entry, context["safety_environment"], mp.mpf(1))
    safety_margin = (
        abs(safety_factor * _kernel_exact(safety_s, kernel) * safety_s * safety_s)
        if has_kernel
        else mp.mpf(0)
    )
    safety_limit = mp.mpf(
        SCREENED_CONFIG["newton_control"]["yukawa_safety"]["max_abs_boost_ratio"]
    )
    newton_passes = bool(
        newton_entries[0] <= thresholds["newton_near"]
        and newton_entries[1] <= thresholds["newton_far"]
        and safety_margin < safety_limit
    )
    solar_ratio = max(
        newton_entries[0] / thresholds["newton_near"],
        newton_entries[1] / thresholds["newton_far"],
        safety_margin / safety_limit,
    )

    return {
        "values": dict(values),
        "formula": render_candidate({"values": values}),
        "newton": {
            "passes": newton_passes,
            "near_deviation": _text(newton_entries[0]),
            "far_deviation": _text(newton_entries[1]),
            "report_y8_deviation": _text(newton_entries[2]),
            "safety_margin": _text(safety_margin),
            "safety_passes": bool(safety_margin < safety_limit),
            "solar_ratio": _text(solar_ratio),
            "screen_factor_near": _text(screen_factors[0]),
            "screen_factor_safety": _text(safety_factor),
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
            "shortfall_min": ("inf" if min(shortfalls) == mp.inf else _text(min(shortfalls))),
            "screen_factor_by_probe": [
                _text(screen_factor_exact(entry, item, mp.mpf(1)))
                for item in environment["cluster"]
            ],
        },
        "all_pass": bool(newton_passes and galaxy_passes and lensing_passes and cluster_passes),
    }


def evaluate_candidate_exact(ordinal: int, context: Mapping[str, Any]) -> dict[str, Any]:
    """`evaluate_values_exact` for one grid ordinal, tagged with its indices."""

    decoded = decode_ordinal(ordinal)
    verdict = evaluate_values_exact(decoded["values"], context)
    return {"ordinal": ordinal, "indices": decoded["indices"], **verdict}


# ---------------------------------------------------------------------------
# Float tables: screened basis boosts, pregathers, local scalars
# ---------------------------------------------------------------------------


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
    """(BY[8, rows], BP[384, rows]): unit-weight, unscreened kernel boosts per probe."""

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


def build_sweep_tables(context: Mapping[str, Any]) -> dict[str, Any]:
    """Every array the batched sweep needs, built deterministically on CPU float64.

    The screening index is folded into the basis tables (it multiplies the boost, never
    the kernel parameters), so a candidate still costs two table lookups per probe.
    """

    _dps()
    geometry = context["geometry"]
    pack = context["pack"]
    environment = context["environment"]
    screens = len(SCREEN_ENTRIES)

    by_disk, bp_disk = _basis_for(geometry["disk"])
    by_hern, bp_hern = _basis_for(geometry["hernquist"])
    by_clu, bp_clu = _basis_for(geometry["cluster"])

    masses = [disk["mass"] for disk in pack["galaxy"]]
    s_disk = np.stack([_screen_matrix(environment["disk"], mass) for mass in masses], axis=1)
    s_hern = np.stack(
        [_screen_matrix(environment["hernquist"], mass) for mass in masses], axis=1
    )
    s_clu = _screen_matrix(environment["cluster"], mp.mpf(1))

    # Galaxy: (screen, mass, probe, {L1 | combo}) flattened row-major.
    byg = np.einsum("smk,lk->smkl", s_disk, by_disk).reshape(screens * 3 * 5, -1)
    bpg = np.einsum("smk,ck->smkc", s_disk, bp_disk).reshape(screens * 3 * 5, -1)

    # Lensing: fold the path-node gather and the per-node screening into one table.
    pair_count = len(pack["lensing"])
    gather = np.zeros((pair_count, len(geometry["hernquist"])), dtype=np.float64)
    mass_index = np.zeros(pair_count, dtype=np.int64)
    for pair_index, pair in enumerate(pack["lensing"]):
        mass_index[pair_index] = pair["mass_index"]
        for gw, _, row in pair["nodes"]:
            gather[pair_index, row] += float(gw)
    gather_screened = gather[None, :, :] * s_hern[:, mass_index, :]
    ay = np.einsum("spr,lr->spl", gather_screened, by_hern).reshape(screens * pair_count, -1)
    ap = np.einsum("spr,cr->spc", gather_screened, bp_hern).reshape(screens * pair_count, -1)

    # Cluster: (screen, probe, {L1 | combo}).
    byc = np.einsum("sk,lk->skl", s_clu, by_clu).reshape(screens * 5, -1)
    bpc = np.einsum("sk,ck->skc", s_clu, bp_clu).reshape(screens * 5, -1)

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
    screen_newton = np.zeros((screens, 3), dtype=np.float64)
    for k, (y, s) in enumerate(pack["newton"]):
        for index, local in enumerate(locals_):
            newton_local[index, k] = float(_nu_local_exact(local, y) - 1)
        ky, kp = _point_kernel_tables(float(s))
        ky_point[:, k] = ky
        kp_point[:, k] = kp
        for index, entry in enumerate(SCREEN_ENTRIES):
            screen_newton[index, k] = float(
                screen_factor_exact(entry, context["newton_environments"][k], mp.mpf(1))
            )
    ky_safety, kp_safety = _point_kernel_tables(float(pack["safety_s"]))
    screen_safety = np.array(
        [
            float(screen_factor_exact(entry, context["safety_environment"], mp.mpf(1)))
            for entry in SCREEN_ENTRIES
        ],
        dtype=np.float64,
    )

    return {
        "BYG": np.ascontiguousarray(byg.ravel()),
        "BPG": np.ascontiguousarray(bpg.ravel()),
        "AY": np.ascontiguousarray(ay.ravel()),
        "AP": np.ascontiguousarray(ap.ravel()),
        "BYC": np.ascontiguousarray(byc.ravel()),
        "BPC": np.ascontiguousarray(bpc.ravel()),
        "KY_pt": ky_point,
        "KP_pt": kp_point,
        "KY_saf": ky_safety,
        "KP_saf": kp_safety,
        "SN": np.ascontiguousarray(screen_newton.ravel()),
        "SS": screen_safety,
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
    """fp64 and fp32 device copies; the Solar block always reads the fp64 copies."""

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

    The extended-system gates run in ``dtype``; the whole Solar block runs in fp64 in both
    tiers (the 1e-8 safety criterion and the screened 1e-16 factors sit far below fp32
    resolution), so the fp32 tier remains a strict superset of the fp64 tier.
    """

    thresholds = {key: float(value) for key, value in SCREENED_CONFIG[tier].items()}
    suffix = "_f32" if dtype == xp.float32 else ""

    def table(name: str) -> Any:
        return tables[name + suffix]

    ix = _decode_batch(xp, ordinals)
    il, il1, ic = ix["local"], ix["L1"], ix["combo"]
    screen = ix["screen"]
    w_yukawa = table("WY")[ix["w_yukawa"]]
    w_power = table("WP")[ix["w_power"]]
    n_l1 = len(AXES["L1"])
    n_combo = N_POWER_COMBOS

    base_gal_y = screen * (15 * n_l1) + il1
    base_gal_p = screen * (15 * n_combo) + ic
    base_len_y = screen * (15 * n_l1) + il1
    base_len_p = screen * (15 * n_combo) + ic
    base_clu_y = screen * (5 * n_l1) + il1
    base_clu_p = screen * (5 * n_combo) + ic

    # Galaxy: flat outer curves and the Tully-Fisher slope on the frozen disks.
    valid = xp.ones(ordinals.shape[0], dtype=bool)
    flat_worst = xp.zeros(ordinals.shape[0], dtype=dtype)
    vflat = []
    for m in range(3):
        speed_sum = None
        vmax = None
        vmin = None
        for k in range(5):
            slot = m * 5 + k
            boost = (
                w_yukawa * table("BYG")[base_gal_y + slot * n_l1]
                + w_power * table("BPG")[base_gal_p + slot * n_combo]
            )
            v2_value = table("S_gal")[:, m, k][il] + boost * dtype(
                float(tables["R_gal"][m, k])
            )
            valid &= v2_value > 0
            speed = xp.sqrt(xp.maximum(v2_value, dtype(0)))
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

    # Lensing: deflection flatness and dynamics-lensing consistency, screened per node.
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
                w_yukawa * table("AY")[base_len_y + pair * n_l1]
                + w_power * table("AP")[base_len_p + pair * n_combo]
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

    # Cluster: the decisive 5-probe hydrostatic ratio criterion, screened.
    cluster_dev = xp.zeros(ordinals.shape[0], dtype=dtype)
    for k in range(5):
        boost = (
            w_yukawa * table("BYC")[base_clu_y + k * n_l1]
            + w_power * table("BPC")[base_clu_p + k * n_combo]
        )
        g_obs = table("S_clu")[:, k][il] + boost
        cluster_dev = xp.maximum(
            cluster_dev, xp.abs(g_obs / dtype(float(tables["gdyn"][k])) - 1)
        )
    cluster_pass = cluster_dev <= dtype(thresholds["cluster_consistency"])

    # Solar block: fp64 in both tiers, screened at the point-source probes.
    w_yukawa64 = tables["WY"][ix["w_yukawa"]]
    w_power64 = tables["WP"][ix["w_power"]]
    newton_devs = []
    for k in range(3):
        factor = tables["SN"][screen * 3 + k]
        newton_devs.append(
            xp.abs(
                tables["N_loc"][:, k][il]
                + factor
                * (
                    w_yukawa64 * tables["KY_pt"][:, k][il1]
                    + w_power64 * tables["KP_pt"][:, k][ic]
                )
            )
        )
    safety = xp.abs(
        tables["SS"][screen]
        * (w_yukawa64 * tables["KY_saf"][il1] + w_power64 * tables["KP_saf"][ic])
    )
    safety_limit = float(
        SCREENED_CONFIG["newton_control"]["yukawa_safety"]["max_abs_boost_ratio"]
    )
    safety_pass = safety < safety_limit
    newton_pass = (
        (newton_devs[0] <= thresholds["newton_near"])
        & (newton_devs[1] <= thresholds["newton_far"])
        & safety_pass
    )
    strict = SCREENED_CONFIG["fp64_thresholds"]
    solar_ratio = xp.maximum(
        xp.maximum(
            newton_devs[0] / float(strict["newton_near"]),
            newton_devs[1] / float(strict["newton_far"]),
        ),
        safety / safety_limit,
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
        "solar_ratio": solar_ratio,
        "flat_worst": xp.where(valid, flat_worst, infinity),
        "btfr_err": xp.where(valid, btfr_err, infinity),
        "lensing_flat": xp.where(lens_valid, worst_flat, infinity),
        "lensing_cons": xp.where(lens_valid, worst_cons, infinity),
        "cluster_dev": cluster_dev,
    }


# ---------------------------------------------------------------------------
# Known-answer controls
# ---------------------------------------------------------------------------

CONTROL_ORDINALS = {
    "newton_identity": encode_named(),
    "sqrt_local": encode_named(local="sqrt_one_plus_u_squared"),
    "linear_u_local": encode_named(local="one_plus_u"),
    "unscreened_yukawa": encode_named(w_yukawa="1", L1="4"),
    "unscreened_power": encode_named(w_power="1", L2="2", p="2", t="-1"),
    "strong_acceleration_screen": encode_named(
        screen="acceleration:1e-2:4", w_yukawa="6", L1="4"
    ),
    "solar_free_acceleration_screen": encode_named(
        screen="acceleration:100:2", w_yukawa="6", L1="4"
    ),
    "strong_curvature_screen": encode_named(screen="curvature:300:4", w_yukawa="6", L1="4"),
}


def _assert_known_answer_controls(
    controls: Mapping[str, Mapping[str, Any]], embedded: Mapping[str, Any]
) -> None:
    """The calibration is part of the claim: a broken control aborts the run."""

    newton = controls["newton_identity"]
    if not newton["newton"]["passes"]:
        raise ScreenedKernelError("Newton control failed Newtonian recovery")
    if newton["galaxy"]["passes"] or newton["cluster"]["passes"] or newton["lensing"]["passes"]:
        raise ScreenedKernelError("Newton control unexpectedly passed a physics gate")
    if float(newton["cluster"]["shortfall_min"]) < 1.5:
        raise ScreenedKernelError("cluster calibration too weak: Newton shortfall < 3/2")
    sqrt_local = controls["sqrt_local"]
    if not (sqrt_local["galaxy"]["passes"] and sqrt_local["lensing"]["passes"]):
        raise ScreenedKernelError("sqrt-local control lost its galaxy/lensing passes")
    if sqrt_local["cluster"]["passes"]:
        raise ScreenedKernelError(
            "embedded pointwise family no longer reproduces the sealed cluster negative"
        )
    linear = controls["linear_u_local"]
    if not linear["galaxy"]["passes"] or linear["lensing"]["passes"]:
        raise ScreenedKernelError("linear-u control drifted: must flatten curves yet fail lensing")

    # Screening sanity, both directions.
    unscreened = controls["unscreened_yukawa"]
    if unscreened["newton"]["safety_passes"]:
        raise ScreenedKernelError("unscreened Yukawa slipped past the safety probe")
    if not controls["unscreened_power"]["newton"]["passes"]:
        raise ScreenedKernelError("the solar-safe rising-tail witness died at the Newton gate")
    for name in ("strong_acceleration_screen", "strong_curvature_screen"):
        strong = controls[name]
        if not strong["newton"]["passes"]:
            raise ScreenedKernelError(f"{name} failed the Solar gates it must pass")
        if strong["galaxy"]["passes"]:
            raise ScreenedKernelError(f"{name} must screen the galaxy boost away and fail")
    free = controls["solar_free_acceleration_screen"]
    if not free["newton"]["passes"]:
        raise ScreenedKernelError(
            "an acceleration screen at g_star = 100 must free the Solar gates"
        )
    if float(free["cluster"]["screen_factor_by_probe"][4]) < 0.99:
        raise ScreenedKernelError(
            "an acceleration screen at g_star = 100 must leave the cluster unscreened"
        )

    for name, block in embedded.items():
        if not block["reproduced"]:
            raise ScreenedKernelError(
                f"embedded v2 reproduction failed for {name}: {block['mismatches']}"
            )


def reproduce_v2_exemplars(context: Mapping[str, Any]) -> dict[str, Any]:
    """Re-evaluate v2's four recorded closest approaches at screening index 0.

    ``exact``-tier metrics must match v2's recorded strings verbatim; ``fp64``-tier
    metrics (recorded by v2's float64 batch layer) must agree to 1e-9 relative.  On-grid
    exemplars are additionally checked through the ordinal codec, and the two routes must
    agree exactly.
    """

    out: dict[str, Any] = {}
    for name, record in V2_RECORDED.items():
        values = {"screen": "none:0:0", **record["values"]}
        verdict = evaluate_values_exact(values, context)
        observed = {
            "cluster_dev": verdict["cluster"]["max_deviation"],
            "cluster_closest_probe": verdict["cluster"]["closest_probe_deviation"],
            "flat_worst": verdict["galaxy"]["flat_worst"],
            "btfr_slope": verdict["galaxy"]["btfr_slope"],
            "btfr_err": verdict["galaxy"]["btfr_error"],
            "lensing_cons": verdict["lensing"]["worst_consistency"],
            "lensing_flat": verdict["lensing"]["worst_flatness"],
            "newton_near": verdict["newton"]["near_deviation"],
            "newton_far": verdict["newton"]["far_deviation"],
            "newton_report_y8": verdict["newton"]["report_y8_deviation"],
            "safety_margin": verdict["newton"]["safety_margin"],
        }
        mismatches: list[str] = []
        for key, recorded in record["metrics"].items():
            got = observed[key]
            if record["tier"] == "exact":
                if got != recorded:
                    mismatches.append(f"{key}: {got} != {recorded}")
            elif abs(float(got) / float(recorded) - 1) > 1e-9:
                mismatches.append(f"{key}: {got} !~ {recorded}")
        for index, recorded in enumerate(record.get("cluster_ratio_by_probe", [])):
            got = verdict["cluster"]["ratio_by_probe"][index]
            if got != recorded:
                mismatches.append(f"cluster_ratio[{index}]: {got} != {recorded}")
        on_grid = all(
            record["values"][axis] in AXES[axis] for axis in record["values"]
        )
        ordinal = None
        if on_grid:
            ordinal = encode_named(**record["values"])
            by_ordinal = evaluate_candidate_exact(ordinal, context)
            if by_ordinal["cluster"]["max_deviation"] != observed["cluster_dev"]:
                mismatches.append("ordinal route disagrees with the value route")
        out[name] = {
            "v2_ordinal": record["v2_ordinal"],
            "v3_ordinal": ordinal,
            "on_v3_grid": on_grid,
            "tier": record["tier"],
            "reproduced": not mismatches,
            "mismatches": mismatches,
            "observed": {key: value for key, value in observed.items() if value is not None},
            "recorded": record["metrics"],
        }
    return out


# ---------------------------------------------------------------------------
# Survivor structure: families, Pareto, covariant lifts, predictions
# ---------------------------------------------------------------------------


_STRIDES = np.array(
    [math.prod(AXIS_SIZES[index + 1 :]) for index in range(len(AXIS_SIZES))],
    dtype=np.int64,
)


def _neighbor_offsets() -> np.ndarray:
    """The 3280 lexicographically positive Chebyshev-1 offsets in 8 dimensions.

    Adjacency is symmetric, so only half the 3^8 - 1 non-zero offsets are needed; the
    edge list they produce is the same undirected graph.
    """

    grid = np.stack(
        np.meshgrid(*([np.array([-1, 0, 1], dtype=np.int8)] * len(AXIS_SIZES)), indexing="ij"),
        axis=-1,
    ).reshape(-1, len(AXIS_SIZES))
    keep = []
    for offset in grid:
        for value in offset:
            if value != 0:
                keep.append(value > 0)
                break
        else:
            keep.append(False)
    return grid[np.asarray(keep, dtype=bool)].astype(np.int64)


def passer_families(ordinals: Sequence[int]) -> list[list[int]]:
    """Equivalence families: same screening family, grid-neighbor everywhere else.

    Two passers join a family when their screening *family* label matches and every axis
    index (including the screening scale and sharpness) differs by at most one grid step.
    Edges are found by vectorized ordinal arithmetic over the 3280 positive Chebyshev-1
    offsets, then merged with union-find; the result is identical to a per-node BFS.
    """

    sorted_ordinals = np.unique(np.asarray(list(ordinals), dtype=np.int64))
    count = sorted_ordinals.size
    if count == 0:
        return []
    indices = np.empty((count, len(AXIS_SIZES)), dtype=np.int64)
    remainder = sorted_ordinals.copy()
    for axis in range(len(AXIS_SIZES) - 1, -1, -1):
        indices[:, axis] = remainder % AXIS_SIZES[axis]
        remainder //= AXIS_SIZES[axis]
    screen_family = np.array(
        [SCREEN_ENTRIES[index][0] for index in indices[:, 0]], dtype=object
    )

    parent = np.arange(count, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    sizes = np.asarray(AXIS_SIZES, dtype=np.int64)
    for offset in _neighbor_offsets():
        shifted = indices + offset
        inside = np.all((shifted >= 0) & (shifted < sizes), axis=1)
        if not inside.any():
            continue
        rows = np.flatnonzero(inside)
        targets = shifted[rows] @ _STRIDES
        position = np.searchsorted(sorted_ordinals, targets)
        position = np.minimum(position, count - 1)
        hit = sorted_ordinals[position] == targets
        for left, right in zip(rows[hit], position[hit], strict=True):
            if screen_family[left] != screen_family[right]:
                continue
            root_left, root_right = find(int(left)), find(int(right))
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)

    groups: dict[int, list[int]] = {}
    for node in range(count):
        groups.setdefault(find(node), []).append(int(sorted_ordinals[node]))
    return [sorted(group) for _, group in sorted(groups.items())]


def active_parameter_count(values: Mapping[str, str]) -> int:
    """Pareto simplicity axis: parameters doing work in this candidate."""

    count = 0
    if values["local"] != "identity":
        count += 1
    if values["w_yukawa"] != "0":
        count += 2  # amplitude + range
    if values["w_power"] != "0":
        count += 4  # amplitude + scale + rise + tail
    if screen_entry(values)[0] != "none":
        count += 2  # screening scale + sharpness
    return count


SCREENING_LIFTS = {
    "density": {
        "mechanism": "chameleon_density_screening",
        "field_theory_ansatz": (
            "chameleon scalar with a runaway potential V(phi) = M^(4+n) phi^(-n) and a "
            "conformal matter coupling A(phi) = exp(beta phi/Mpl): the effective mass "
            "m_eff^2 = V'' + beta^2 rho/Mpl^2 grows with the local baryon density, so the "
            "fifth force is thin-shell suppressed where rho >> rho_star.  S = "
            "[1+(rho/rho_star)^k]^(-1) is the phenomenological stand-in for that "
            "suppression, not its derivation"
        ),
    },
    "acceleration": {
        "mechanism": "vainshtein_kinetic_braiding",
        "field_theory_ansatz": (
            "cubic-Galileon / kinetic-braiding sector L = -(dphi)^2/2 - "
            "(Box phi)(dphi)^2/Lambda^3 + phi T/Mpl: the nonlinear derivative "
            "self-interaction makes the scalar kinetic term large where the Newtonian "
            "acceleration exceeds g_star, suppressing the fifth force inside the "
            "Vainshtein radius.  S = [1+(g_N/g_star)^k]^(-1) is the phenomenological "
            "stand-in for that suppression, not its derivation"
        ),
    },
    "curvature": {
        "mechanism": "kmouflage_gradient_screening",
        "field_theory_ansatz": (
            "K-mouflage sector L = M^4 K(X), X = (dphi)^2/(2 M^4), with K nonlinear at "
            "large X: screening keys on the gradient scale of the Newtonian field rather "
            "than on its amplitude, so S = [1+(Lc |grad g_N|/g_N)^k]^(-1) with Lc the "
            "declared crossover length.  Phenomenological stand-in, not a derivation"
        ),
    },
}


def emit_screened_covariant_lift(values: Mapping[str, str]) -> dict[str, Any]:
    """v2's covariant-lift emitter plus the screening mechanism component."""

    lift = emit_covariant_lift(kernel_values(values))
    family, scale, sharpness = screen_entry(values)
    components = list(lift["components"])
    if family == "none":
        components.append(
            {
                "mechanism": "no_screening",
                "static_kernel": "S = 1",
                "field_theory_ansatz": (
                    "the embedded v2 grammar: no environment dependence (the mandatory "
                    "control, whose sealed negative v3 must reproduce)"
                ),
            }
        )
    else:
        template = SCREENING_LIFTS[family]
        components.append(
            {
                "mechanism": template["mechanism"],
                "static_kernel": render_screening(values),
                "field_theory_ansatz": (
                    f"{template['field_theory_ansatz']}; declared crossover "
                    f"{'Lc' if family == 'curvature' else 'scale'} = {scale} (code units), "
                    f"sharpness k = {sharpness}"
                ),
            }
        )
    return {
        "kernel_form": render_candidate({"values": dict(values)}),
        "components": components,
        "claims": {"first_principles_derivation_pending": True},
    }


#: The declared intermediate scales at which a family's deviation from GR is predicted.
PREDICTION_PROBES = {
    "isolated_point_source_y100": "separation s = 1/10 from a unit point mass (y = 100)",
    "cluster_r4": "the frozen cluster probe at r = 4 core radii",
    "disk_r12": "the largest control disk at r = 12 scale lengths",
}


def _transition_radius(
    entry: tuple[str, str, str], system: str, mass: mp.mpf, environment: Mapping[str, Any]
) -> str:
    """Radius where the screening argument crosses 1 (S = 1/2), by bisection."""

    family = entry[0]
    if family == "none":
        return "no_screening"
    if system == "point_source":
        # X(s) is monotone decreasing in s for every family; solve on s in [1e-8, 1e4].
        def argument(radius: mp.mpf) -> mp.mpf:
            ambient = environment["solar_ambient_rho"]
            return screening_argument(entry, _point_environment(radius, ambient), mp.mpf(1))

        low, high = mp.mpf("1e-8"), mp.mpf("1e4")
    else:
        g_fn, grad_fn, rho_fn = _UNIT_FIELDS[system]

        def argument(radius: mp.mpf) -> mp.mpf:
            item = {
                "g": g_fn(radius),
                "grad": abs(grad_fn(radius)),
                "rho": rho_fn(radius),
                "curv": abs(grad_fn(radius)) / g_fn(radius),
            }
            return screening_argument(entry, item, mass)

        low, high = mp.mpf("1e-4"), mp.mpf("1e3")
    low_value, high_value = argument(low) - 1, argument(high) - 1
    if low_value * high_value > 0:
        return "screened_everywhere" if low_value > 0 else "unscreened_everywhere"
    for _ in range(120):
        mid = mp.sqrt(low * high)
        if (argument(mid) - 1) * low_value > 0:
            low = mid
        else:
            high = mid
    return _text(mp.sqrt(low * high))


def falsifiable_predictions(
    values: Mapping[str, str], context: Mapping[str, Any]
) -> dict[str, Any]:
    """Screening transition radii plus predicted GR deviations at declared scales."""

    _dps()
    entry = screen_entry(values)
    environment = context["environment"]
    geometry = context["geometry"]
    pack = context["pack"]
    kernel = kernel_values(values)
    has_kernel = values["w_yukawa"] != "0" or values["w_power"] != "0"
    biggest = pack["galaxy"][-1]

    transitions = {
        "point_source_separation": _transition_radius(
            entry, "point_source", mp.mpf(1), environment
        ),
        "cluster_radius": _transition_radius(entry, "cluster", mp.mpf(1), environment),
        "disk_radius_largest_mass": _transition_radius(
            entry, "disk", biggest["mass"], environment
        ),
        "hernquist_radius_largest_mass": _transition_radius(
            entry, "hernquist", biggest["mass"], environment
        ),
    }

    # Point source at the declared intermediate scale y = 100 (s = 1/10).
    separation = mp.mpf(1) / 10
    point_env = _point_environment(separation, environment["solar_ambient_rho"])
    factor = screen_factor_exact(entry, point_env, mp.mpf(1))
    point_dev = (_nu_local_exact(values["local"], 1 / separation**2) - 1) + (
        factor * _kernel_exact(separation, kernel) * separation * separation
        if has_kernel
        else mp.mpf(0)
    )

    cluster_boost = _boost_exact_screened(
        "cluster", geometry["cluster"], environment["cluster"], values, entry, mp.mpf(1)
    )
    gbar_cluster = pack["cluster"][3][0]
    cluster_dev = (
        _nu_local_exact(values["local"], gbar_cluster) * gbar_cluster + cluster_boost[3]
    ) / gbar_cluster - 1

    disk_boost = _boost_exact_screened(
        "disk", geometry["disk"], environment["disk"], values, entry, biggest["mass"]
    )
    _, gbar_disk = biggest["points"][2]
    disk_dev = (
        _nu_local_exact(values["local"], gbar_disk) * gbar_disk
        + biggest["mass"] * disk_boost[2]
    ) / gbar_disk - 1

    return {
        "screening_transition_radii": transitions,
        "transition_definition": "the radius at which the screening argument X = 1, i.e. S = 1/2",
        "predicted_gr_deviations": {
            "isolated_point_source_y100": _text(point_dev),
            "cluster_r4": _text(cluster_dev),
            "disk_r12": _text(disk_dev),
        },
        "prediction_probes": PREDICTION_PROBES,
        "falsifier": (
            "measure g_obs/g_N - 1 at a separation s = 1/10 from an isolated point mass "
            f"(code units, y = 100): this family predicts {_text(point_dev)}.  An "
            "observation bounding that deviation below the predicted value, or a "
            "hydrostatic cluster at r = 4 core radii inconsistent with "
            f"{_text(cluster_dev)}, kills the family outright -- the screening scale "
            "cannot be retuned per object (zero per-object freedom is a claim of this run)"
        ),
        "claims": {"first_principles_derivation_pending": True},
    }


def _pareto_front(axes: np.ndarray, cap: int) -> np.ndarray:
    """Row indices of the non-dominated set (minimize every axis), deterministic order.

    Lexicographic sweep: if B dominates A then B precedes A in lexicographic order, and
    domination is transitive, so testing each point against the accepted front is exact
    and avoids the quadratic all-pairs comparison at large passer counts.
    """

    if axes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort(tuple(axes[:, index] for index in range(axes.shape[1] - 1, -1, -1)))
    front_rows: list[int] = []
    front_values = np.empty((0, axes.shape[1]), dtype=np.float64)
    for row in order:
        point = axes[row]
        if front_values.shape[0]:
            not_worse = (front_values <= point).all(axis=1)
            strictly = (front_values < point).any(axis=1)
            if bool((not_worse & strictly).any()):
                continue
        front_rows.append(int(row))
        front_values = np.vstack([front_values, point[None, :]])
    front = np.asarray(sorted(front_rows), dtype=np.int64)
    ranking = np.lexsort((front, axes[front, 1], axes[front, 0]))
    return front[ranking][:cap]


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

    def update(self, xp: Any, metric: Any, ordinals: Any) -> None:
        if metric.shape[0] == 0:
            return
        row = int(xp.argmin(metric))
        value = float(metric[row])
        ordinal = int(ordinals[row])
        if not math.isfinite(value):
            return
        if value < self.value or (
            value == self.value and (self.ordinal is None or ordinal < self.ordinal)
        ):
            self.value = value
            self.ordinal = ordinal


def run_screen(
    *,
    limit: int | None = None,
    batch_size: int = 1 << 22,
    use_gpu: bool = True,
    pareto_cap: int = 64,
) -> dict[str, Any]:
    """Screen the family (or its first ``limit`` ordinals) and seal a receipt."""

    if not SYSTEM_CAPS["min_batch_size"] <= batch_size <= SYSTEM_CAPS["max_batch_size"]:
        raise ScreenedKernelError(f"batch_size outside system caps: {batch_size}")
    if not 1 <= pareto_cap <= SYSTEM_CAPS["max_pareto_reported"]:
        raise ScreenedKernelError(f"pareto_cap outside system caps: {pareto_cap}")
    if limit is not None and limit < 1:
        raise ScreenedKernelError(f"limit must be positive: {limit}")
    if FAMILY_SIZE != math.prod(AXIS_SIZES) or not 10**8 <= FAMILY_SIZE < 10**9:
        raise ScreenedKernelError("family size drifted outside the declared 10^8-10^9 band")

    xp, device, gpu = _array_module(use_gpu)
    context = build_exact_context()
    geometry = context["geometry"]
    newton_validation = geometry_newton_residuals(geometry)
    budget = mp.mpf(V2_CONFIG["geometry"]["newton_validation_max_relative_error"])
    for system, residual in newton_validation.items():
        if mp.mpf(residual) > budget:
            raise ScreenedKernelError(f"geometry validation failed for {system}: {residual}")
    env_validation = environment_residuals(geometry)
    env_budget = mp.mpf(SCREENED_CONFIG["environment_validation"]["max_relative_error"])
    for system, block in env_validation.items():
        for key, residual in block.items():
            if mp.mpf(residual) > env_budget:
                raise ScreenedKernelError(
                    f"environment validation failed for {system}.{key}: {residual}"
                )
    tables = build_sweep_tables(context)
    device_tables = _device_tables(tables, xp)

    total = FAMILY_SIZE if limit is None else min(limit, FAMILY_SIZE)
    gate_names = ("newton", "safety", "galaxy", "lensing", "cluster", "all")
    fp32_counts = dict.fromkeys(gate_names, 0)
    trackers = {
        "cluster_any": _ArgminTracker(),
        "cluster_solar_safe": _ArgminTracker(),
        "cluster_among_galaxy_lensing_passers": _ArgminTracker(),
        "lensing_among_galaxy_passers": _ArgminTracker(),
        "galaxy": _ArgminTracker(),
    }
    bounds = [float(text) for text in TENSION_MAP_BOUNDS]
    frontier_any = [_ArgminTracker() for _ in bounds]
    frontier_gl = [_ArgminTracker() for _ in bounds]
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
        cluster_dev = sweep["cluster_dev"]
        trackers["cluster_any"].update(xp, cluster_dev, ordinals)
        trackers["cluster_solar_safe"].update(
            xp, xp.where(sweep["newton_pass"], cluster_dev, infinity), ordinals
        )
        galaxy_lensing = sweep["galaxy_pass"] & sweep["lensing_pass"]
        trackers["cluster_among_galaxy_lensing_passers"].update(
            xp, xp.where(galaxy_lensing, cluster_dev, infinity), ordinals
        )
        trackers["lensing_among_galaxy_passers"].update(
            xp, xp.where(sweep["galaxy_pass"], sweep["lensing_cons"], infinity), ordinals
        )
        flat_limit = xp.float32(float(SCREENED_CONFIG["fp32_thresholds"]["flatness"]))
        btfr_limit = xp.float32(float(SCREENED_CONFIG["fp32_thresholds"]["btfr_slope"]))
        trackers["galaxy"].update(
            xp,
            xp.maximum(sweep["flat_worst"] / flat_limit, sweep["btfr_err"] / btfr_limit),
            ordinals,
        )
        solar_ratio = sweep["solar_ratio"]
        for index, bound in enumerate(bounds):
            within = solar_ratio <= bound
            frontier_any[index].update(xp, xp.where(within, cluster_dev, infinity), ordinals)
            frontier_gl[index].update(
                xp, xp.where(within & galaxy_lensing, cluster_dev, infinity), ordinals
            )
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
        for row in np.flatnonzero(host["all_pass"]):
            ordinal = int(survivors32[start + row])
            passer_ordinals.append(ordinal)
            passer_metrics[ordinal] = {
                "cluster_dev": float(host["cluster_dev"][row]),
                "lensing_cons": float(host["lensing_cons"][row]),
                "newton_far": float(host["newton_far"][row]),
                "solar_ratio": float(host["solar_ratio"][row]),
            }

    # fp64 refinement of every tracked exemplar (closest approaches and frontier bins).
    tracked = sorted(
        {tracker.ordinal for tracker in trackers.values() if tracker.ordinal is not None}
        | {t.ordinal for t in frontier_any if t.ordinal is not None}
        | {t.ordinal for t in frontier_gl if t.ordinal is not None}
    )
    refined: dict[int, dict[str, Any]] = {}
    if tracked:
        cpu_tables = _device_tables(tables, np)
        batch = evaluate_batch(
            np,
            np.asarray(tracked, dtype=np.int64),
            cpu_tables,
            dtype=np.float64,
            tier="fp64_thresholds",
        )
        for row, ordinal in enumerate(tracked):
            refined[ordinal] = {
                key: float(np.asarray(batch[key])[row])
                for key in (
                    "cluster_dev", "lensing_cons", "lensing_flat", "flat_worst",
                    "btfr_err", "newton_near", "newton_far", "safety_margin", "solar_ratio",
                )
            }

    # Known-answer controls, always at 50 digits; a broken control aborts the run.
    controls = {
        name: evaluate_candidate_exact(ordinal, context)
        for name, ordinal in CONTROL_ORDINALS.items()
    }
    embedded = reproduce_v2_exemplars(context)
    _assert_known_answer_controls(controls, embedded)

    # The tension map: best cluster deviation versus the Solar violation factor.
    tension_rows = []
    running_any = float("inf")
    running_gl = float("inf")
    for index, bound_text in enumerate(TENSION_MAP_BOUNDS):
        any_tracker, gl_tracker = frontier_any[index], frontier_gl[index]
        any_value = (
            refined[any_tracker.ordinal]["cluster_dev"]
            if any_tracker.ordinal is not None
            else float("inf")
        )
        gl_value = (
            refined[gl_tracker.ordinal]["cluster_dev"]
            if gl_tracker.ordinal is not None
            else float("inf")
        )
        running_any = min(running_any, any_value)
        running_gl = min(running_gl, gl_value)
        bound = float(bound_text)
        under = [
            metrics["cluster_dev"]
            for metrics in passer_metrics.values()
            if metrics["solar_ratio"] <= bound
        ]
        strict_count = len(under)
        strict_best = min(under) if under else float("inf")
        tension_rows.append(
            {
                "solar_ratio_bound": bound_text,
                "solar_safe": bool(float(bound_text) <= 1.0),
                "best_cluster_deviation": (
                    "inf" if running_any == float("inf") else _text(running_any)
                ),
                "best_cluster_deviation_with_galaxy_and_lensing": (
                    "inf" if running_gl == float("inf") else _text(running_gl)
                ),
                # The strict column: the best cluster deviation among fp64-confirmed
                # all-gate passers whose fp64 Solar ratio is under this bound.  No tier
                # caveat attaches to it -- these candidates passed every gate.
                "best_cluster_deviation_all_gates_fp64": (
                    "none" if strict_best == float("inf") else _text(strict_best)
                ),
                "all_gate_passers_under_bound": strict_count,
                "exemplar_ordinal": any_tracker.ordinal,
                "exemplar_with_galaxy_and_lensing_ordinal": gl_tracker.ordinal,
                # The bins are selected in the slack fp32 tier, so the galaxy/lensing
                # column is only a *claim* about the strict gates until the exemplar is
                # rechecked in fp64.  This flag is that recheck, per row.
                "galaxy_and_lensing_confirmed_fp64": bool(
                    gl_tracker.ordinal is not None
                    and _passes_galaxy_and_lensing_fp64(refined[gl_tracker.ordinal])
                ),
            }
        )

    # Survivor structure: families, Pareto, covariant lifts, falsifiable predictions.
    families_block: list[dict[str, Any]] = []
    pareto_entries: list[dict[str, Any]] = []
    family_truncated = False
    family_count = 0
    representatives: list[int] = []
    pareto_ordinals: list[int] = []
    ordered: list[list[int]] = []
    if passer_ordinals:
        family_input = passer_ordinals
        if len(family_input) > SYSTEM_CAPS["family_analysis_cap"]:
            family_truncated = True
            family_input = sorted(
                family_input, key=lambda o: (passer_metrics[o]["cluster_dev"], o)
            )[: SYSTEM_CAPS["family_analysis_cap"]]
        families = passer_families(family_input)
        family_count = len(families)
        ordered = sorted(
            families, key=lambda f: min(passer_metrics[o]["cluster_dev"] for o in f)
        )
        representatives = [
            min(family, key=lambda o: (passer_metrics[o]["cluster_dev"], o))
            for family in ordered
        ]
        for family, representative in zip(ordered, representatives, strict=True):
            values = decode_ordinal(representative)["values"]
            if len(families_block) >= SYSTEM_CAPS["max_families_reported"]:
                break
            families_block.append(
                {
                    "size": len(family),
                    "screening_family": screen_entry(values)[0],
                    "representative_ordinal": representative,
                    "representative_values": values,
                    "representative_formula": render_candidate({"values": values}),
                    "cluster_deviation": _text(passer_metrics[representative]["cluster_dev"]),
                    "lensing_consistency": _text(passer_metrics[representative]["lensing_cons"]),
                    "solar_ratio": _text(passer_metrics[representative]["solar_ratio"]),
                    "covariant_lift_candidate": emit_screened_covariant_lift(values),
                    "falsifiable_prediction": falsifiable_predictions(values, context),
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
        for row in _pareto_front(axes, pareto_cap):
            ordinal = passer_ordinals[int(row)]
            pareto_ordinals.append(ordinal)
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
    breakdown = _screening_family_breakdown(passer_ordinals, passer_metrics, ordered)

    # 50-digit re-verification.  The verification set is chosen by *role*, not by
    # ordinal order: every reported family representative, every Pareto-front member,
    # every closest-approach and tension-map exemplar, then a seeded random sample of
    # the remaining bulk passers.  Exhaustive verification of all passers is recorded
    # as truncated with the measured per-candidate cost implied by the budget.
    verify_ordinals: list[int] = []
    selection: dict[int, str] = {}
    for ordinal in representatives:
        selection.setdefault(ordinal, "family_representative")
    for ordinal in pareto_ordinals:
        selection.setdefault(ordinal, "pareto_front")
    for ordinal in tracked:
        selection.setdefault(ordinal, "closest_approach_or_frontier_exemplar")
    remaining = sorted(set(passer_ordinals) - set(selection))
    if remaining:
        rng_sample = np.random.default_rng(SCREENED_CONFIG["crosscheck_seed"])
        picked = rng_sample.choice(
            len(remaining),
            size=min(SYSTEM_CAPS["bulk_passer_sample"], len(remaining)),
            replace=False,
        )
        for index in sorted(int(value) for value in picked):
            selection.setdefault(remaining[index], "bulk_passer_random_sample")
    verify_ordinals = sorted(selection)
    exact_budget = SYSTEM_CAPS["max_exact_verifications"]
    selection_truncated = max(0, len(verify_ordinals) - exact_budget)
    verify_ordinals = verify_ordinals[:exact_budget]
    passer_set = set(passer_ordinals)
    exact_verification = []
    for ordinal in verify_ordinals:
        verdict = evaluate_candidate_exact(ordinal, context)
        verdict["role"] = "all_gate_passer" if ordinal in passer_set else "closest_approach"
        verdict["selection"] = selection[ordinal]
        verdict["exact_confirmed"] = (
            verdict["all_pass"] if ordinal in passer_set else True
        )
        exact_verification.append(verdict)
    confirmed = sum(1 for entry in exact_verification if entry["exact_confirmed"])
    exact_truncated = max(0, len(passer_set) - len(passer_set & set(verify_ordinals)))

    sealed_negative = None
    if not passer_ordinals:
        sealed_negative = _seal_negative(
            trackers, refined, tension_rows, exact_verification
        )

    # CPU/GPU decision cross-check on a deterministic sample.
    rng = np.random.default_rng(SCREENED_CONFIG["crosscheck_seed"])
    sample = np.sort(
        rng.choice(total, size=min(SCREENED_CONFIG["crosscheck_sample"], total), replace=False)
    ).astype(np.int64)
    cpu_sample = evaluate_batch(
        np, sample, _device_tables(tables, np), dtype=np.float64, tier="fp64_thresholds"
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
        carriers = ", ".join(
            f"{row['screening_family']} {row['all_gate_passers']}"
            for row in breakdown
            if row["all_gate_passers"]
        )
        unconstrained = tension_rows[-1]["best_cluster_deviation"]
        safe = tension_rows[TENSION_MAP_BOUNDS.index("1")]["best_cluster_deviation"]
        decision = (
            f"SCREENED: {all_gate_passers} screened-kernel candidates in {family_count} "
            f"equivalence families pass every gate on the synthetic controls ({carriers}); "
            f"v2's tension is dissolved -- the best cluster deviation costs {safe} under "
            f"full Solar safety against {unconstrained} with the Solar gates ignored, "
            "where v2 paid 4.050292865e-01 against 2.372244161e-02; survivors are search "
            "priorities for the covariant lift, not validated theories"
        )
    else:
        margin = tension_rows[TENSION_MAP_BOUNDS.index("1")]["best_cluster_deviation"]
        decision = (
            "SCREENED-SEALED-NEGATIVE: no processed candidate passes all gates; best "
            f"solar-safe cluster deviation {margin} against tolerance "
            f"{SCREENED_CONFIG['fp64_thresholds']['cluster_consistency']}"
        )

    body: dict[str, Any] = {
        "answers": V2_TENSION_STATEMENT,
        "assumptions": {
            "lensing_prescription": GATE_CONFIG["lensing"]["prescription"],
            "screening_is_phenomenological": (
                "S(x) is a declared environment-dependent suppression factor, not a "
                "solution of a scalar-field equation; the covariant-lift blocks name the "
                "mechanism each family stands in for and flag the derivation as pending"
            ),
            "solar_probe_environment": SCREENED_CONFIG["screening"]["solar_ambient_source"],
            "disk_screening_thickness": (
                SCREENED_CONFIG["screening"]["disk_screening_thickness_note"]
            ),
            "shared_code_units": (
                "disk Rd = cluster rc = Hernquist a = 1 in code units, so one kernel "
                "scale L and one screening length mean the same code length in every "
                "system; real galaxies and clusters are ~50x apart in physical scale. "
                "Survivors and negatives are statements about these synthetic controls."
            ),
        },
        "claims": CLAIMS,
        "config": SCREENED_CONFIG,
        "config_sha256": canonical_sha256(SCREENED_CONFIG),
        "v2_config_sha256": canonical_sha256(V2_CONFIG),
        "screen_config_sha256": canonical_sha256(SCREEN_CONFIG),
        "gate_config_sha256": canonical_sha256(GATE_CONFIG),
        "geometry_sha256": geometry_sha256(geometry),
        "geometry_newton_validation": newton_validation,
        "environment_sha256": environment_sha256(context["environment"]),
        "environment_validation": env_validation,
        "controls": controls,
        "embedded_v2_reproduction": embedded,
        "counts": {
            "family_size": FAMILY_SIZE,
            "embedded_v2_block_size": KERNEL_SUBFAMILY_SIZE,
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
            "passers_not_exactly_verified": exact_truncated,
            "exact_verification_selection_truncated": selection_truncated,
        },
        "exact_verification_policy": (
            "every reported family representative, every Pareto-front member, and every "
            "closest-approach / tension-map exemplar is re-run at 50 digits, plus a "
            f"seeded random sample of {SYSTEM_CAPS['bulk_passer_sample']} of the "
            "remaining passers.  Exhaustive 50-digit verification of every passer is not "
            "run: one exact evaluation costs a full 46,392-node mpmath convolution, so "
            "the bulk set would take hours.  'passers_not_exactly_verified' records "
            "exactly how many were left to the fp64 tier."
        ),
        "screening_family_breakdown": breakdown,
        "crosscheck": crosscheck,
        "decision": decision,
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "sweep_elapsed_seconds": format(elapsed_sweep, ".3f"),
        "exact_verification": exact_verification,
        "passer_families_reported": families_block,
        "pareto_front": pareto_entries,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "GPU screen of an ordinal-indexed screened-kernel gravity grammar: v2's "
            "nonlocal kernel boost multiplied by a declared environment-dependent "
            "screening factor S(x) drawn from a density (chameleon-like), acceleration "
            "(Vainshtein-like), curvature-proxy, or none family.  The grammar answers "
            "v2's sealed negative, which measured that the cluster's needed amplitude is "
            "the amplitude the Solar System forbids.  Screening is applied in every gate, "
            "lensing included.  One shared constant set per candidate, zero per-object "
            "freedom; no observational data opened; survivors are search priorities and a "
            "zero-survivor verdict is a sealed negative with margins, a measured tension "
            "frontier, and a named next structural direction."
        ),
        "sealed_negative": sealed_negative,
        "tension_map": {
            "definition": (
                "solar_ratio = max(newton_near/2e-2, newton_far/2e-3, safety/1e-8); "
                "solar_ratio <= 1 is fully Solar-safe.  Each row reports the best cluster "
                "deviation achievable anywhere in the family under that Solar bound "
                "(cumulative, hence monotone non-increasing in the bound), and the best "
                "achievable while also passing the galaxy and lensing gates."
            ),
            "cluster_tolerance": SCREENED_CONFIG["fp64_thresholds"]["cluster_consistency"],
            "selection_tier": (
                "bins selected in the fp32 sweep tier (Solar block fp64); the reported "
                "values are fp64 refinements of the selected exemplars, accumulated as a "
                "running minimum so the frontier is monotone by construction.  Because "
                "fp32 is the slack tier, the galaxy-and-lensing column is a claim about "
                "the strict gates only where 'galaxy_and_lensing_confirmed_fp64' is true; "
                "the strict all-gate answer is the "
                "'best_cluster_deviation_all_gates_fp64' column, which is computed "
                "directly from the fp64-confirmed passer set and carries no tier caveat."
            ),
            "strict_all_gate_best_cluster_deviation": (
                _text(min(passer_metrics[o]["cluster_dev"] for o in passer_ordinals))
                if passer_ordinals
                else "none"
            ),
            "v2_frontier_for_comparison": {
                "best_cluster_deviation_unconstrained": "2.372244161e-02",
                "best_cluster_deviation_solar_safe": "4.050292865e-01",
                "solar_safety_penalty_factor": "1.707527732e+01",
            },
            "solar_safety_penalty_factor": (
                _text(
                    float(tension_rows[TENSION_MAP_BOUNDS.index("1")]["best_cluster_deviation"])
                    / float(tension_rows[-1]["best_cluster_deviation"])
                )
                if tension_rows[-1]["best_cluster_deviation"] != "inf"
                and tension_rows[TENSION_MAP_BOUNDS.index("1")]["best_cluster_deviation"]
                != "inf"
                else "inf"
            ),
            "penalty_definition": (
                "the factor by which demanding full Solar safety degrades the best "
                "achievable cluster deviation.  v2 measured 17.1; this run measures the "
                "same quantity for the screened grammar, and that ratio is the object "
                "v2's headline sentence was an anecdote about."
            ),
            "frontier": tension_rows,
        },
        "throughput_candidates_per_second": (
            int(processed / elapsed_sweep) if elapsed_sweep > 0 else None
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _passes_galaxy_and_lensing_fp64(metrics: Mapping[str, float]) -> bool:
    """The strict galaxy and lensing verdicts, read off fp64-refined metrics."""

    strict = SCREENED_CONFIG["fp64_thresholds"]
    return (
        metrics["flat_worst"] <= float(strict["flatness"])
        and metrics["btfr_err"] <= float(strict["btfr_slope"])
        and metrics["lensing_flat"] <= float(strict["lensing_flatness"])
        and metrics["lensing_cons"] <= float(strict["lensing_consistency"])
    )


def _screening_family_breakdown(
    passers: Sequence[int],
    metrics: Mapping[int, Mapping[str, float]],
    families: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Which screening families actually carry the gates, and how well.

    Reported for all four families -- including the ones with zero passers, because a
    zero is a measured statement about that mechanism on these controls.
    """

    per_family: dict[str, list[int]] = {name: [] for name in SCREEN_FAMILIES}
    for ordinal in passers:
        per_family[SCREEN_ENTRIES[ordinal // KERNEL_SUBFAMILY_SIZE][0]].append(ordinal)
    family_counts = dict.fromkeys(SCREEN_FAMILIES, 0)
    for family in families:
        family_counts[SCREEN_ENTRIES[family[0] // KERNEL_SUBFAMILY_SIZE][0]] += 1
    rows = []
    for name in SCREEN_FAMILIES:
        members = per_family[name]
        rows.append(
            {
                "screening_family": name,
                "all_gate_passers": len(members),
                "equivalence_families": family_counts[name],
                "best_cluster_deviation": (
                    _text(min(metrics[o]["cluster_dev"] for o in members))
                    if members
                    else "none"
                ),
                "best_solar_ratio": (
                    _text(min(metrics[o]["solar_ratio"] for o in members))
                    if members
                    else "none"
                ),
            }
        )
    return rows


def _seal_negative(
    trackers: Mapping[str, _ArgminTracker],
    refined: Mapping[int, Mapping[str, Any]],
    tension_rows: Sequence[Mapping[str, Any]],
    exact_verification: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-gate closest approaches plus the next structural direction from the margins."""

    tolerance = float(SCREENED_CONFIG["fp64_thresholds"]["cluster_consistency"])

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

    fp64 = SCREENED_CONFIG["fp64_thresholds"]
    closest = {
        "cluster_any": approach("cluster_any", "cluster_dev", fp64["cluster_consistency"]),
        "cluster_solar_safe": approach(
            "cluster_solar_safe", "cluster_dev", fp64["cluster_consistency"]
        ),
        "cluster_among_galaxy_lensing_passers": approach(
            "cluster_among_galaxy_lensing_passers", "cluster_dev", fp64["cluster_consistency"]
        ),
        "lensing_among_galaxy_passers": approach(
            "lensing_among_galaxy_passers", "lensing_cons", fp64["lensing_consistency"]
        ),
        "galaxy": approach("galaxy", "flat_worst", fp64["flatness"]),
    }

    safe_row = tension_rows[TENSION_MAP_BOUNDS.index("1")]
    best_safe_text = safe_row["best_cluster_deviation"]
    best_safe = float("inf") if best_safe_text == "inf" else float(best_safe_text)
    best_safe_gl_text = safe_row["best_cluster_deviation_with_galaxy_and_lensing"]
    best_safe_gl = float("inf") if best_safe_gl_text == "inf" else float(best_safe_gl_text)
    best_any = (
        refined[trackers["cluster_any"].ordinal]["cluster_dev"]
        if trackers["cluster_any"].ordinal is not None
        else float("inf")
    )
    best_gl = (
        refined[trackers["cluster_among_galaxy_lensing_passers"].ordinal]["cluster_dev"]
        if trackers["cluster_among_galaxy_lensing_passers"].ordinal is not None
        else float("inf")
    )

    if best_safe <= tolerance < best_safe_gl:
        direction = (
            "screening WORKS on the Solar axis and the wall has moved.  A solar-safe "
            f"screened kernel reaches cluster deviation {_text(best_safe)} inside the "
            f"{tolerance} tolerance -- v2's headline tension (the cluster amplitude the "
            "Solar System forbids) is dissolved by environment-dependent screening.  What "
            "now blocks is the galaxy/lensing pair: the best solar-safe cluster match that "
            f"also carries flat curves and consistent deflection is {_text(best_safe_gl)}.  "
            "The next structural axis is therefore NOT more screening but a "
            "SCALE-DEPENDENT COUPLING: the boost must respond differently to a disk and to "
            "a sphere at the same separation and the same environment.  Concretely, a "
            "two-field kernel (one component sourced by surface density, one by volume "
            "density) or an anisotropic/tensor kernel K(s, n.n') that a razor-thin disk "
            "and an isothermal sphere sample differently."
        )
    elif best_safe > tolerance and best_any <= tolerance:
        direction = (
            "screening did NOT dissolve v2's tension on the declared grids.  The family "
            f"still expresses the cluster shape unconstrained (best deviation "
            f"{_text(best_any)}) but the best solar-safe deviation is {_text(best_safe)}, "
            f"outside the {tolerance} tolerance.  The measured tension map shows how much "
            "of the gap each order of Solar violation buys, and the closest-approach "
            "exemplars name which family gets furthest.  The next structural axis is a "
            "TWO-FIELD or SCALE-DEPENDENT COUPLING rather than a sharper single screen: a "
            "single scalar S(x) built from one environment variable cannot be small at the "
            "Solar probe and order unity across all five cluster radii at once, because "
            "the cluster's own environment brackets the Solar probe's on the density axis "
            "and the required boost is not monotone in either acceleration or curvature."
        )
    else:
        direction = (
            "no ordinal reproduces the cluster radial shape even with the Solar gates "
            f"ignored (best deviation {_text(best_any)} against tolerance {tolerance}), so "
            "the obstruction is upstream of screening: an isotropic distance kernel times "
            "any scalar function of the local environment cannot make the boost depend on "
            "direction or on source kinematics.  The next structural axis is an "
            "ANISOTROPIC/TENSOR KERNEL K(s, n.n'), not a sharper screen."
        )

    best_exemplar = None
    safe_ordinal = trackers["cluster_solar_safe"].ordinal
    if safe_ordinal is not None:
        best_exemplar = next(
            (entry for entry in exact_verification if entry["ordinal"] == safe_ordinal), None
        )
    return {
        "sealed": True,
        "statement": (
            "within the processed ordinal range, no candidate of the screened-kernel "
            "grammar passes the Solar, galaxy, lensing, and cluster gates jointly on the "
            "frozen synthetic controls"
        ),
        "closest_approach": closest,
        "best_cluster_shape_achiever": best_exemplar,
        "best_cluster_among_galaxy_and_lensing_passers": (
            "inf" if best_gl == float("inf") else _text(best_gl)
        ),
        "structural_direction": direction,
    }


# ---------------------------------------------------------------------------
# Precompute-validation receipt
# ---------------------------------------------------------------------------


def build_precompute_receipt() -> dict[str, Any]:
    """Sealed receipt for the frozen geometry and the two new environment fields."""

    geometry = build_kernel_geometry()
    environment = build_environment(geometry)
    residuals = geometry_newton_residuals(geometry)
    budget = mp.mpf(V2_CONFIG["geometry"]["newton_validation_max_relative_error"])
    for system, residual in residuals.items():
        if mp.mpf(residual) > budget:
            raise ScreenedKernelError(f"geometry validation failed for {system}: {residual}")
    env_residuals = environment_residuals(geometry)
    env_budget = mp.mpf(SCREENED_CONFIG["environment_validation"]["max_relative_error"])
    for system, block in env_residuals.items():
        for key, residual in block.items():
            if mp.mpf(residual) > env_budget:
                raise ScreenedKernelError(
                    f"environment validation failed for {system}.{key}: {residual}"
                )
    _dps()
    ranges = {}
    for system in ("disk", "hernquist", "cluster"):
        entries = environment[system]
        ranges[system] = {
            field: {
                "min": _text(min(entry[field] for entry in entries)),
                "max": _text(max(entry[field] for entry in entries)),
            }
            for field in ("g", "rho", "curv")
        }
    body = {
        "schema_version": PRECOMPUTE_SCHEMA,
        "config_sha256": canonical_sha256(SCREENED_CONFIG),
        "geometry_sha256": geometry_sha256(geometry),
        "environment_sha256": environment_sha256(environment),
        "systems": {
            system: {
                "rows": len(rows),
                "nodes": sum(len(row["nodes"]) for row in rows),
                "newton_max_relative_error": residuals[system],
                **env_residuals[system],
            }
            for system, rows in geometry.items()
        },
        "total_nodes": sum(
            len(row["nodes"]) for rows in geometry.values() for row in rows
        ),
        "unit_mass_field_ranges": ranges,
        "solar_ambient_rho": _text(environment["solar_ambient_rho"]),
        "newton_validation_budget": V2_CONFIG["geometry"][
            "newton_validation_max_relative_error"
        ],
        "environment_validation": SCREENED_CONFIG["environment_validation"],
        "scope": (
            "Validation of the frozen precompute this screen stands on: the Newton kernel "
            "reproduces v2's frozen g_bar grids at every probe of every system, and the "
            "two new screening fields (rho_local and |grad g_N|) are reproduced by "
            "independent routes -- the Poisson divergence of the frozen Newtonian field "
            "and a high-precision central difference -- to the declared budget."
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_precompute_receipt(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != PRECOMPUTE_SCHEMA:
        raise ScreenedKernelError("precompute receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ScreenedKernelError("precompute receipt seal changed")
    if value.get("config_sha256") != canonical_sha256(SCREENED_CONFIG):
        raise ScreenedKernelError("precompute receipt config binding changed")
    geometry = build_kernel_geometry()
    if value.get("geometry_sha256") != geometry_sha256(geometry):
        raise ScreenedKernelError("frozen geometry does not replay")
    if value.get("environment_sha256") != environment_sha256(build_environment(geometry)):
        raise ScreenedKernelError("frozen environment fields do not replay")
    residuals = geometry_newton_residuals(geometry)
    env_residuals = environment_residuals(geometry)
    for system, block in value.get("systems", {}).items():
        if block.get("newton_max_relative_error") != residuals.get(system):
            raise ScreenedKernelError(f"newton validation does not replay for {system}")
        for key in ("rho_max_relative_error", "grad_max_relative_error"):
            if block.get(key) != env_residuals.get(system, {}).get(key):
                raise ScreenedKernelError(f"{key} does not replay for {system}")


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, binding, geometry, environment, control, and sample replay; fail closed."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise ScreenedKernelError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ScreenedKernelError("receipt seal changed")
    if value.get("claims") != CLAIMS:
        raise ScreenedKernelError("claims block changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise ScreenedKernelError("config binding changed")
    if value.get("config_sha256") != canonical_sha256(SCREENED_CONFIG):
        raise ScreenedKernelError("receipt config does not match this module")
    if value.get("v2_config_sha256") != canonical_sha256(V2_CONFIG):
        raise ScreenedKernelError("v2 kernel config binding changed")
    if value.get("screen_config_sha256") != canonical_sha256(SCREEN_CONFIG):
        raise ScreenedKernelError("screen config binding changed")
    if value.get("gate_config_sha256") != canonical_sha256(GATE_CONFIG):
        raise ScreenedKernelError("gate config binding changed")
    context = build_exact_context()
    if value.get("geometry_sha256") != geometry_sha256(context["geometry"]):
        raise ScreenedKernelError("frozen geometry does not replay")
    if value.get("environment_sha256") != environment_sha256(context["environment"]):
        raise ScreenedKernelError("frozen environment fields do not replay")
    if value.get("geometry_newton_validation") != geometry_newton_residuals(
        context["geometry"]
    ):
        raise ScreenedKernelError("geometry Newton validation does not replay")
    if value.get("environment_validation") != environment_residuals(context["geometry"]):
        raise ScreenedKernelError("environment validation does not replay")
    controls = {
        name: evaluate_candidate_exact(ordinal, context)
        for name, ordinal in CONTROL_ORDINALS.items()
    }
    embedded = reproduce_v2_exemplars(context)
    _assert_known_answer_controls(controls, embedded)
    for name, verdict in controls.items():
        recorded = value.get("controls", {}).get(name, {})
        for gate in ("newton", "galaxy", "lensing", "cluster"):
            if recorded.get(gate, {}).get("passes") != verdict[gate]["passes"]:
                raise ScreenedKernelError(f"control replay changed for {name}.{gate}")
    for name, block in embedded.items():
        if value.get("embedded_v2_reproduction", {}).get(name, {}).get("observed") != block[
            "observed"
        ]:
            raise ScreenedKernelError(f"embedded v2 reproduction does not replay for {name}")
    for entry in value.get("exact_verification", []):
        if not entry.get("exact_confirmed", False):
            continue
        replay = evaluate_candidate_exact(entry["ordinal"], context)
        for gate in ("newton", "galaxy", "lensing", "cluster"):
            if replay[gate]["passes"] != entry[gate]["passes"]:
                raise ScreenedKernelError(
                    f"exact replay failed for ordinal {entry['ordinal']} at {gate}"
                )
    frontier = value.get("tension_map", {}).get("frontier", [])
    previous = dict.fromkeys(
        (
            "best_cluster_deviation",
            "best_cluster_deviation_with_galaxy_and_lensing",
            "best_cluster_deviation_all_gates_fp64",
        ),
        float("inf"),
    )
    for row in frontier:
        for column, last in previous.items():
            current = row[column]
            numeric = float("inf") if current in ("inf", "none") else float(current)
            if numeric > last + 1e-15:
                raise ScreenedKernelError(f"tension map frontier is not monotone: {column}")
            previous[column] = numeric
    processed = value.get("counts", {}).get("processed")
    if not isinstance(processed, int) or processed < 1:
        raise ScreenedKernelError("receipt processed count is malformed")
    tables = build_sweep_tables(context)
    rng = np.random.default_rng(SCREENED_CONFIG["crosscheck_seed"])
    sample = np.sort(
        rng.choice(
            processed,
            size=min(SCREENED_CONFIG["crosscheck_sample"], processed),
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
        raise ScreenedKernelError("crosscheck sample decisions do not replay")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise ScreenedKernelError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screened-kernel gravity screen v3 (GPU, ordinal-indexed)."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1 << 22)
    parser.add_argument("--cpu", action="store_true", help="force the numpy path")
    parser.add_argument("--output")
    parser.add_argument("--precompute-output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        if not args.output and not args.precompute_output:
            raise ScreenedKernelError(
                "--validate-checked requires --output or --precompute-output"
            )
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
