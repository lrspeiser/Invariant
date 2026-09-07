"""halo.py -- the dark-matter universe's PRIOR, with every nuisance an explicit knob.

Run BK's binding finding: the CDM-vs-class separation BF reported "is a
statement about the alignment prior, not about gravity" -- BF's generator gave
its haloes no knowledge of the surrounding structure (f_lss = 0), and above
f_lss = 0.38 the separation collapsed.  The same trap exists for every other
property of a halo that the visible baryons do not fix.  So this module does
not hide any of them inside a generator: each is a named knob with a declared
nominal prior, and every separation in this lane is reported as a function
of them.

Galaxy haloes (BF's are spherical NFW with SHMR scatter; here in addition):
    s_shmr     multiplier on the 0.16 dex halo-mass scatter at fixed M*
    s_conc     multiplier on the 0.11 dex concentration scatter
    q_amp      in-plane m=2 modulation of v_c^2 from a flattened halo,
               U(0, q_max); its axis sits near the DISC axis with mis_deg of
               scatter and takes a fraction f_lss from the external axis
    q_h        oblateness of the halo potential (vertical force x 1/q_h^2)
    f_dd       DARK DISC: fraction of the halo's enclosed-mass profile placed
               in a razor-thin disc.  Radial force unchanged; vertical force
               at the stellar disc raised from G M z / r^3 to G M'(R)/R.
               This is the one halo degree of freedom that can mimic an
               isotropic boost of the baryons' own field, so it is the axis
               along which the vertical answer changes.
Cluster haloes (BF: NFW with ell = 0.72 ell_bar + N(0, 0.10), pa = pa_bar +
N(0, 22 deg), f_lss = 0):
    s_mclu     multiplier on the 0.05 dex halo-mass scatter at fixed M_bar
    s_conc     multiplier on the 0.13 dex concentration scatter
    s_shape    multiplier on the 0.10 shape scatter; shape_corr the slope
    f_lss      fraction of the projected alignment inherited from the external
               axis (BK's mixture); None -> drawn per cluster from the prior
    mis_deg    misalignment scatter about the mixed axis

Nominal priors are DECLARED here.  They are inputs to the null, not
measurements, and the report quotes the separation as a function of them.
"""
from __future__ import annotations

import numpy as np

from universes import corpus as cp
from universes.baryons import G

NOMINAL = dict(
    s_shmr=1.0, s_conc=1.0, s_mclu=1.0, s_shape=1.0, shape_corr=0.72,
    q_max=0.10, gal_mis_deg=25.0, q_h_range=(0.70, 1.00), f_dd_range=(0.0, 0.05),
    f_lss=None, f_lss_prior=(2.0, 2.0),      # Beta(2,2): mean 0.5, sd 0.22
    mis_deg=22.0, fix_halo_per_object=False,
)


def knobs(**over):
    k = dict(NOMINAL)
    k.update(over)
    return k


def _f_lss(k, rng):
    if k["f_lss"] is not None:
        return float(k["f_lss"])
    a, b = k["f_lss_prior"]
    return float(rng.beta(a, b))


def mixed_axis(pa_local_deg, pa_ext_deg, f_lss, rng, mis_deg):
    """BK's halo_axis: the projected major axis as an f_lss mixture of the
    local (baryon / disc) axis and the external axis, plus scatter."""
    a_b = np.deg2rad(2.0 * pa_local_deg)
    a_e = np.deg2rad(2.0 * pa_ext_deg)
    v = (1.0 - f_lss) * np.array([np.cos(a_b), np.sin(a_b)]) \
        + f_lss * np.array([np.cos(a_e), np.sin(a_e)])
    base = 0.5 * np.rad2deg(np.arctan2(v[1], v[0]))
    return float((base + rng.normal(0.0, mis_deg)) % 180.0)


def draw_galaxy_halo(gal, p, rng, k):
    """A galaxy's collisionless halo under the declared prior.

    ``p`` is the CDM universe's parameter dict (fbar, c_norm, shmr_scatter);
    ``k`` the knob dict.  Returns the dm dict consumed by paired.galaxy_field_cdm.
    """
    M200 = cp.shmr_M200(gal.Md + gal.Mb) * 10 ** rng.normal(0, p["shmr_scatter"] * k["s_shmr"])
    c = p["c_norm"] * 2.2 * (M200 / 1e12) ** -0.10 * 10 ** rng.normal(0, 0.11 * k["s_conc"])
    r2 = cp.r200_of(M200)
    rs = r2 / c
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    norm = (1.0 - 0.157) * M200 / mu(c)
    f_lss = _f_lss(k, rng)
    q_amp = float(rng.uniform(0.0, k["q_max"]))
    # in-plane axis of the flattened halo, in the DISC frame (disc axis = 0)
    psi = mixed_axis(0.0, gal.axis_ext_deg - gal.pa_deg, f_lss, rng, k["gal_mis_deg"])
    q_h = float(rng.uniform(*k["q_h_range"]))
    f_dd = float(rng.uniform(*k["f_dd_range"]))
    return {"Mdm": lambda r: norm * mu(np.asarray(r, float) / rs),
            "M200": float(M200), "c": float(c), "r200": float(r2), "rs": float(rs),
            "norm": float(norm),
            "q_amp": q_amp, "psi_deg": psi, "q_h": q_h, "f_dd": f_dd,
            "f_lss": f_lss}


def draw_cluster_halo(clu, p, rng, k):
    M200 = clu.Mbar500 / p["fbar"] * 1.35 * 10 ** rng.normal(0, 0.05 * k["s_mclu"])
    c = p["c_norm"] * 10 ** rng.normal(0, 0.13 * k["s_conc"])
    r2 = cp.r200_of(M200)
    rs = r2 / c
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    norm = (1.0 - p["fbar"]) * M200 / mu(c)
    e_h = float(np.clip(k["shape_corr"] * clu.ell_bar
                        + rng.normal(0, 0.10 * k["s_shape"]), 0.0, 0.55))
    f_lss = _f_lss(k, rng)
    pa_h = mixed_axis(clu.pa_bar_deg, clu.axis_ext_deg, f_lss, rng, k["mis_deg"])
    return {"Mdm": lambda r: norm * mu(np.asarray(r, float) / rs),
            "M200": float(M200), "c": float(c), "r200": float(r2), "rs": float(rs),
            "norm": float(norm), "ell": e_h, "pa": pa_h, "f_lss": f_lss}


def rescale_halo(dm, dlogM=0.0, dlogc=0.0, dell=0.0, dpa=0.0):
    """A COUNTERFACTUAL halo: the same halo with one property moved.

    Used for 'move the halo holding the baryons fixed'.  The NFW normalisation
    is recomputed so the enclosed-mass profile is self-consistent.
    """
    M200 = dm["M200"] * 10 ** dlogM
    c = dm["c"] * 10 ** dlogc
    r2 = cp.r200_of(M200)
    rs = r2 / c
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    # keep the (1 - fbar) factor implied by the original normalisation
    fb = dm["norm"] * mu(dm["c"]) / dm["M200"]
    norm = fb * M200 / mu(c)
    out = dict(dm)
    out.update({"Mdm": lambda r: norm * mu(np.asarray(r, float) / rs),
                "M200": float(M200), "c": float(c), "r200": float(r2),
                "rs": float(rs), "norm": float(norm)})
    if "ell" in dm:
        out["ell"] = float(np.clip(dm["ell"] + dell, 0.0, 0.9))
        out["pa"] = float((dm["pa"] + dpa) % 180.0)
    if "psi_deg" in dm:
        out["psi_deg"] = float((dm["psi_deg"] + dpa) % 180.0)
    return out


def halo_vertical_field(R, hz, dm):
    """Vertical acceleration of the halo at height hz above the disc plane.

    spherical part   g_z = G M(<r) z / r^3          -> G M(<R) hz / R^3 at z << R
    oblateness       potential-flattened halo, m^2 = R^2 + z^2/q^2:
                     g_z = g_sph(m) z / (q^2 m)     -> a 1/q_h^2 multiplier
    dark disc        fraction f_dd of the enclosed-mass profile in a razor-thin
                     disc with Sigma_dd = M'(R) / (2 pi R):  g_z = G M'(R) / R
    The RADIAL force is the same in every case to leading order, which is what
    makes f_dd the clean 'change the vertical structure holding the radial
    force' counterfactual.
    """
    R = np.asarray(R, float)
    M = dm["Mdm"](R)
    g_sph = G * M * hz / R ** 3 / dm.get("q_h", 1.0) ** 2
    dR = 1e-3 * R
    dM = (dm["Mdm"](R + dR) - dm["Mdm"](R - dR)) / (2 * dR)
    g_dd = G * np.maximum(dM, 0.0) / R
    f = dm.get("f_dd", 0.0)
    return (1.0 - f) * g_sph + f * g_dd
