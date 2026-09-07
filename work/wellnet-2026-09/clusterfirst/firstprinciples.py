"""firstprinciples.py -- gravity predicted from where the baryons actually are.

Eleven clusters carry all three channels at once, which is what makes this
possible and is why the test has waited so long:

  * ACCEPT deprojected electron density n_e(r), calibrated, from Chandra
  * deep Chandra event lists (this programme's own, Run BR)
  * public per-source weak-lensing shear from DECADE, in the OPEN half of the
    holdout

One of them is 1E0657-56, the Bullet Cluster.

WHAT IS COMPUTED, and in which direction. The baryons are turned into a
gravitational field from first principles -- no fitted halo, no mass model, no
scaling relation:

    rho_gas(r) = MU_E * m_p * n_e(r)          gas mass density
    M_gas(<r)  = int 4 pi r^2 rho_gas dr      enclosed baryonic mass
    g_bar(r)   = G M_gas(<r) / r^2            Newtonian field OF THE BARYONS
    Sigma_gas(R) = Abel projection of rho_gas
    DeltaSigma_bar(R) = mean Sigma inside R - Sigma at R

and DeltaSigma_bar is compared directly against DeltaSigma_obs from the shear.
The comparison happens IN PROJECTION, which is where lensing actually measures,
so no new deprojection assumption is introduced -- the only spherical assumption
is the one ACCEPT already made when it produced n_e(r), and re-projecting it is
self-consistent with that.

WHAT IS DELIBERATELY NOT USED. eRASS1's M500, R500 and Mgas500 all come from
eROCOP, which is calibrated on weak lensing. Using any of them to predict a weak
-lensing signal would be circular, and the aperture is circular even where the
measurement is not -- the same identity that killed "organised by r/R500"
(failures/artefacts/08-r500-is-r.md). Only fixed-physical-aperture X-ray
quantities and the deprojected profile are admissible here.

STARS ARE MISSING, and it matters in a known direction. Hot gas is roughly
85-90% of a cluster's baryons; the BCG and member galaxies make up the rest, and
no stellar mass profile exists for these eleven. So g_bar is underestimated by
of order 10-15%, which makes any inferred missing gravity slightly WORSE than
reported here. The bias is conservative with respect to the headline.

    python firstprinciples.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))
sys.path.insert(0, WELLNET)

import cosmo as C                                              # noqa: E402

OUT = os.path.join(HERE, "firstprinciples.json")
OVERLAP = os.path.join(HERE, "accept_overlap.json")
PROFILES = os.path.join(WELLNET, "clustershear", "profiles.jsonl")
ACCEPT = os.path.join(HERE, "accept_profiles.json")

M_P = 1.67262192e-27          # kg
MU_E = 1.15                   # gas mass per electron, in m_p; fully ionised ICM
MPC = C.MPC
G = C.G
A0 = 1.2e-10                  # m/s^2, the MOND acceleration scale
SIGMA_CRIT_PREF = 4.0 * math.pi * G / C.CLIGHT ** 2

#: the laws. Each maps g_bar -> predicted g_obs. All are one-parameter-free
#: given a0; none is fitted to these data.
def law_newton(gb):
    return gb


def law_rar(gb):
    """McGaugh+2016 radial acceleration relation."""
    y = np.maximum(gb / A0, 1e-12)
    return gb / (1.0 - np.exp(-np.sqrt(y)))


def law_mond_simple(gb):
    y = np.maximum(gb / A0, 1e-12)
    return gb * (0.5 + np.sqrt(0.25 + 1.0 / y))


def law_mond_standard(gb):
    y = np.maximum(gb / A0, 1e-12)
    return gb * np.sqrt(0.5 + np.sqrt(0.25 + 1.0 / y ** 2))


LAWS = [("Newton", law_newton), ("RAR (McGaugh+2016)", law_rar),
        ("MOND simple nu", law_mond_simple),
        ("MOND standard nu", law_mond_standard)]


# --------------------------------------------------------------- the baryons
def gas_model(prof, r_out_mpc):
    """Return interpolators for rho_gas(r) and M_gas(<r), plus the extrapolated
    fraction of the projection integral.

    The ACCEPT profile stops well inside the lensing aperture. Beyond the last
    measured bin the density is continued as a power law fitted to the outer
    third of the data. `temperature_support` v2 in this programme forbids silent
    clamping and requires the extrapolated fraction to be printed; the same rule
    is applied here to density.
    """
    r = np.array([p[0] for p in prof]) * MPC                   # m
    ne = np.array([p[1] for p in prof])                        # cm^-3
    rho = MU_E * M_P * ne * 1e6                                # kg m^-3

    k = max(4, len(r) // 3)
    lr, lp = np.log(r[-k:]), np.log(rho[-k:])
    slope = np.polyfit(lr, lp, 1)[0]
    slope = min(slope, -1.2)          # force a convergent mass at large radius

    def rho_of(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.interp(x, r, rho, left=rho[0], right=np.nan)
        far = x > r[-1]
        out[far] = rho[-1] * (x[far] / r[-1]) ** slope
        return out

    rr = np.geomspace(r[0], max(r[-1] * 12.0, 6.0 * MPC), 900)
    dens = rho_of(rr)
    mass = np.concatenate(([0.0], np.cumsum(
        0.5 * (4 * math.pi * rr[:-1] ** 2 * dens[:-1]
               + 4 * math.pi * rr[1:] ** 2 * dens[1:]) * np.diff(rr))))
    frac_extrap = float(np.interp(r_out_mpc * MPC, rr, mass)
                        / max(np.interp(r[-1], rr, mass), 1e-30) - 1.0)
    return rho_of, rr, mass, r[-1] / MPC, slope, frac_extrap


def sigma_gas(rho_of, R, r_max_m):
    """Abel projection: Sigma(R) = 2 int rho(sqrt(R^2+l^2)) dl."""
    out = np.empty_like(R)
    for i, Rv in enumerate(R):
        l = np.geomspace(1e-4 * MPC, max(r_max_m, 8.0 * MPC), 500)
        rr = np.sqrt(Rv ** 2 + l ** 2)
        d = rho_of(rr)
        d = np.where(np.isfinite(d), d, 0.0)
        out[i] = 2.0 * np.trapezoid(d, l)
    return out


def delta_sigma_bar(rho_of, R, r_max_m):
    """DeltaSigma(R) = mean Sigma inside R - Sigma(R), computed on a fine grid."""
    grid = np.geomspace(R.min() * 0.05, R.max() * 1.05, 260)
    sg = sigma_gas(rho_of, grid, r_max_m)
    cum = np.concatenate(([0.0], np.cumsum(
        0.5 * (grid[:-1] * sg[:-1] + grid[1:] * sg[1:]) * np.diff(grid))))
    mean_in = 2.0 * np.interp(R, grid, cum) / R ** 2
    return mean_in - np.interp(R, grid, sg)


# ------------------------------------------------------------------ the data
def observed(rec):
    """DeltaSigma_obs and its error, per radial bin, from the shear profile."""
    Dl = float(C.d_ang(rec["z"]))
    out = []
    for b in rec["profile"]:
        if b["gt"] is None or not (b["err"] or 0) > 0:
            continue
        beta = b.get("beta")
        if not beta or beta <= 0:
            continue
        inv_sc = SIGMA_CRIT_PREF * Dl * beta
        out.append((b["R"] * MPC, b["gt"] / inv_sc, b["err"] / inv_sc, b["n"]))
    return out


def main():
    from holdout import loader                                 # noqa: E402
    loader.verify()

    match = json.load(io.open(OVERLAP, encoding="utf-8"))
    acc = json.load(io.open(ACCEPT, encoding="utf-8"))
    loader.assert_not_sealed([m["erass"] for m in match], "first-principles input")

    prof = {}
    for line in io.open(PROFILES, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("profile"):
                prof[r["name"]] = r
    print("clusters with gas + shear: %d (sealed %d untouched)\n"
          % (len(match), len(loader.sealed_names())), flush=True)

    results = []
    for m in sorted(match, key=lambda m: -m["shear"]):
        rec = prof.get(m["erass"])
        p = acc.get(m["accept"])
        if rec is None or not p:
            continue
        obs = observed(rec)
        if len(obs) < 4:
            continue
        R = np.array([o[0] for o in obs])
        ds_obs = np.array([o[1] for o in obs])
        ds_err = np.array([o[2] for o in obs])

        rho_of, rr, mass, r_last_mpc, slope, fex = gas_model(p, R.max() / MPC)
        ds_bar = delta_sigma_bar(rho_of, R, rr[-1])
        gbar = G * np.interp(R, rr, mass) / R ** 2

        # HARD CUT. The gas profile stops long before the shear does. Beyond the
        # last measured bin the density is a fitted power law, and at the outer
        # lensing radii that extrapolation supplies 3x to 140x more mass than
        # ACCEPT ever measured -- the prediction would be a statement about the
        # fit, not about the baryons. `temperature_support` v2 forbids exactly
        # this, so every bin outside the measured range is dropped rather than
        # clamped, and the count that survives is reported.
        inside = R <= r_last_mpc * MPC
        n_in = int(inside.sum())

        # inverse-variance ratio of observed to baryon-predicted lensing,
        # measured radii only
        wgt = 1.0 / ds_err ** 2
        good = inside & (ds_bar > 0)
        ratio = (float(np.sum(wgt[good] * ds_obs[good] / ds_bar[good])
                       / np.sum(wgt[good])) if good.any() else float("nan"))
        results.append(dict(
            n_bins_within_measured_gas=n_in,
            n_bins_total=int(len(R)),
            within=[bool(x) for x in inside],
            cluster=m["erass"], accept=m["accept"], chandra=m["chandra"],
            z=m["z"], kt=m["kt"], n_shear=m["shear"],
            gas_r_max_mpc=round(r_last_mpc, 3),
            outer_slope=round(slope, 2),
            mass_extrapolated_fraction=round(fex, 3),
            R_mpc=[round(x / MPC, 3) for x in R],
            ds_obs=[float(x) for x in ds_obs],
            ds_err=[float(x) for x in ds_err],
            ds_bar=[float(x) for x in ds_bar],
            g_bar=[float(x) for x in gbar],
            ratio_obs_over_baryon=ratio))
        print("  %-24s z=%.3f  gas to %.2f Mpc  %d of %d lensing bins inside it  "
              "obs/baryon = %s"
              % (m["erass"], m["z"], r_last_mpc, n_in, len(R),
                 ("%.1f" % ratio) if n_in else "n/a"), flush=True)

    # ---- the laws, scored on the pooled sample
    print("\n  law                        median predicted/observed   scatter")
    scored = []
    for name, fn in LAWS:
        rat = []
        for r in results:
            gb = np.array(r["g_bar"])
            dsb = np.array(r["ds_bar"])
            dso = np.array(r["ds_obs"])
            inr = np.array(r["within"])
            ok = inr & (dsb > 0) & (gb > 0) & np.isfinite(dso)
            if not ok.any():
                continue
            boost = fn(gb[ok]) / gb[ok]          # the law's enhancement factor
            pred = dsb[ok] * boost               # predicted lensing signal
            rat.extend((pred / dso[ok]).tolist())
        rat = np.array([x for x in rat if np.isfinite(x) and x > 0])
        med = float(np.median(rat)) if len(rat) else float("nan")
        sc = float(np.std(np.log10(rat))) if len(rat) else float("nan")
        scored.append(dict(law=name, median_pred_over_obs=med,
                           log_scatter_dex=sc, n=int(len(rat))))
        print("  %-26s %8.3f                   %.2f dex" % (name, med, sc))

    out = dict(lane="clusterfirst", n_clusters=len(results),
               a0=A0, mu_e=MU_E,
               stars_included=False,
               note=("g_bar is the Newtonian field of the OBSERVED gas, computed "
                     "by direct integration of the ACCEPT deprojected density. No "
                     "halo, no mass model, no scaling relation. eRASS1 M500/R500/"
                     "Mgas500 excluded as weak-lensing calibrated."),
               clusters=results, laws=scored,
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("\nwrote firstprinciples.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
