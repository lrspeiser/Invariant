"""
JOB 2.2 -- does a cancellation lemma exist for CLASH?

Run AT's X-COP lemma:
    X-COP tabulates T/T500 against R/R500, so reconstructing
    T_X x T500(R500) at RW_X x R500 returns the observed physical temperature
    for ANY R500.  Scaling R500 by 0.55x and 2.30x moved g_obs by 1.6e-13.
    R500 enters the x-axis ONLY.

That lemma has a PRECONDITION: the numerator must be tabulated in R500-scaled
units, so that the R500 used to scale and the R500 used to unscale are the same
number and cancel.  CLASH does not meet it.  Tian+2020 fig2.dat tabulates
log g_tot in absolute m/s^2 against an absolute radius in kpc, so R500 is not an
input to the numerator's tabulation at all.

That sounds protective and is not.  It removes the cancellation without removing
the coupling, because for CLASH the coupling is at the ESTIMATOR level rather
than the coordinate level: g_tot(r) and R500 are two functionals of the SAME
two-parameter NFW fit.  This module measures both statements:

  (a) TABLE level   : substitute a different R500 on the x-axis and recompute the
                      numerator from the published table.  Movement is exactly
                      zero -- and that is a statement about the table, not about
                      the measurement.
  (b) ESTIMATOR level: move the lensing mass that GENERATES R500, and measure how
                      far the numerator moves with it.  This is the number that
                      is 1.6e-13 for X-COP.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import stats as S

KPC, MPC, MSUN, G = I.KPC, I.MPC, I.MSUN, I.G
OUT = {}


def r500_of_scaled_profile(M200, c200, z, f):
    """R500 when the whole reconstructed mass profile is multiplied by f
    (a multiplicative bias in kappa -- shear calibration, source redshifts,
    Sigma_crit, the dominant lensing systematic).  Solves
        f M_NFW(<R) = (4/3) pi 500 rho_c(z) R^3   exactly."""
    A = (4 / 3) * math.pi * 500 * I.rhoc(z)
    g = lambda R: f * float(I.nfw_mass(R, M200, c200, z)) * MSUN - A * R ** 3
    lo, hi = 1e-3 * MPC, 30.0 * MPC
    assert g(lo) > 0 > g(hi)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def main():
    D = I.load_all(verbose=False)
    T = I.points_table(D)
    C = D["clusters"]
    nm, r = T["name"], T["r"]

    # ------------------------------------------------------------------ (a)
    print("=== (a) TABLE level: rescale the assumed R500, recompute the numerator ===")
    base = np.log10(T["go"])
    moved = []
    for f in (0.55, 0.80, 1.25, 2.30):
        # exactly what the lane does: r/R500 with a rescaled R500.  The numerator
        # is read straight from fig2.dat and cannot move.
        g2 = np.log10(T["go"])            # unchanged by construction
        moved.append(float(np.max(np.abs(g2 - base))))
    OUT["table_level"] = dict(
        scale_factors=[0.55, 0.80, 1.25, 2.30],
        max_abs_move_dex=moved,
        xcop_comparison=1.6e-13,
        interpretation=(
            "Exactly zero, but for the OPPOSITE reason to X-COP.  In X-COP the "
            "numerator is reconstructed THROUGH R500 and R500 cancels.  In CLASH "
            "R500 is never an input to the tabulated numerator, so there is "
            "nothing to cancel -- and therefore no lemma bounding how far the "
            "numerator moves when the underlying mass moves."))
    print(f"  max |d log10 g_obs| = {max(moved):.3e} for every scale factor")
    print("  -> zero, but this is a property of the TABLE, not of the measurement.")

    # ------------------------------------------------------------------ (b)
    print("\n=== (b) ESTIMATOR level: move the mass that GENERATES R500 ===")
    facs = [0.55, 0.70, 0.85, 1.0, 1.20, 1.50, 2.30]
    rows = []
    for f in facs:
        dlg, dlR = [], []
        for n in sorted(C):
            c = C[n]
            R0 = c["R500_nfw"]
            R1 = r500_of_scaled_profile(c["M200"], c["c200"], c["z"], f)
            dlR.append(math.log10(R1 / R0))
            dlg.append(math.log10(f))   # g_tot(r) -> f g_tot(r) at every r
        rows.append(dict(f=f, dlog10_gobs=float(np.mean(dlg)),
                         dlog10_R500=float(np.mean(dlR)),
                         dlog10_R500_sd=float(np.std(dlR))))
        print(f"  mass x {f:4.2f}:  d log10 g_obs = {np.mean(dlg):+.4f} dex, "
              f"d log10 R500 = {np.mean(dlR):+.4f} dex")
    # local derivative at f = 1
    eps = 0.02
    dg, dR = [], []
    for n in sorted(C):
        c = C[n]
        Rp = r500_of_scaled_profile(c["M200"], c["c200"], c["z"], 1 + eps)
        Rm = r500_of_scaled_profile(c["M200"], c["c200"], c["z"], 1 - eps)
        dR.append((math.log10(Rp) - math.log10(Rm)) / (math.log10(1 + eps) - math.log10(1 - eps)))
        dg.append(1.0)
    dRdf = float(np.mean(dR))
    OUT["estimator_level"] = dict(
        scan=rows,
        dlog10_gobs_per_dex_of_mass=1.0,
        dlog10_R500_per_dex_of_mass=dRdf,
        dlog10_R500_per_dex_of_mass_sd=float(np.std(dR)),
        induced_slope_dy_dx=-1.0 / dRdf,
        induced_slope_da0_dx=-2.0 / dRdf,
        xcop_numerator_move=1.6e-13,
        ratio_to_xcop=float(1.0 / 1.6e-13))
    print(f"\n  d log10 g_obs / d log10 (mass) = 1.000 exactly")
    print(f"  d log10 R500   / d log10 (mass) = {dRdf:.4f} +- {np.std(dR):.4f}")
    print(f"  -> along the common-mode mass direction the numerator moves "
          f"{1.0/dRdf:.3f} dex for every dex R500 moves.")
    print(f"  -> INDUCED SLOPE  dy/dlog10(r/R500)      = {-1.0/dRdf:+.3f}")
    print(f"  -> INDUCED SLOPE  d(log a0)/dlog10(r/R500) = {-2.0/dRdf:+.3f}")
    print(f"\n  X-COP moved the numerator by 1.6e-13 dex.  CLASH moves it by 1.0 dex")
    print(f"  per dex of mass.  There is NO cancellation lemma: the ratio is "
          f"{1.0/1.6e-13:.1e}.")

    # ------------------------------------------------------- parameter-wise
    print("\n=== (c) which parameter carries it: M200 or c200? ===")
    par = {}
    for pname in ("M200", "c200"):
        dg_all, dR_all = [], []
        for n in sorted(C):
            c = C[n]
            v = c[pname]
            out_g, out_R = [], []
            for s in (0.98, 1.02):
                kw = dict(M200=c["M200"], c200=c["c200"])
                kw[pname] = v * s
                R = I.r_delta(kw["M200"], kw["c200"], c["z"], 500.0)
                rr = r[nm == n]
                gg = np.log10(G * I.nfw_mass(rr, kw["M200"], kw["c200"], c["z"])
                              * MSUN / rr ** 2)
                out_g.append(gg); out_R.append(math.log10(R))
            dg_all.append(np.mean(out_g[1] - out_g[0]) / (math.log10(1.02 / 0.98)))
            dR_all.append((out_R[1] - out_R[0]) / (math.log10(1.02 / 0.98)))
        par[pname] = dict(dlog10_gobs_per_dex=float(np.mean(dg_all)),
                          dlog10_R500_per_dex=float(np.mean(dR_all)),
                          ratio=float(np.mean(dg_all) / np.mean(dR_all)))
        print(f"  {pname}: d log g_obs/d log {pname} = {np.mean(dg_all):+.4f}, "
              f"d log R500/d log {pname} = {np.mean(dR_all):+.4f}, "
              f"ratio {np.mean(dg_all)/np.mean(dR_all):+.3f}")
    OUT["by_parameter"] = par

    # ------------------------------- how much R500 scatter could be error?
    print("\n=== (d) leverage: is there enough R500 scatter for the channel to bite? ===")
    lR = np.array([math.log10(C[n]["R500_lens"]) for n in sorted(C)])
    eR = np.array([C[n]["e_M500"] / C[n]["M500"] / 3.0 / math.log(10)
                   for n in sorted(C)])          # e_R500/R500 = (1/3) e_M500/M500
    OUT["leverage"] = dict(
        sd_log10_R500_observed=float(lR.std(ddof=1)),
        mean_e_log10_R500=float(eR.mean()),
        error_to_scatter_ratio=float(eR.mean() / lR.std(ddof=1)),
        R500_span_factor=float(10 ** (lR.max() - lR.min())),
        sd_ln_R500=float(lR.std(ddof=1) * math.log(10)),
        xcop_sd_ln_R500=0.109, xcop_span=1.36)
    print(f"  sd(log10 R500) across 20 clusters = {lR.std(ddof=1):.4f} dex "
          f"(span factor {10**(lR.max()-lR.min()):.2f})")
    print(f"  mean quoted e(log10 R500)          = {eR.mean():.4f} dex")
    print(f"  -> {100*eR.mean()/lR.std(ddof=1):.0f}% of the R500 spread is "
          f"measurement error, and that error is SHARED with the numerator.")

    json.dump(OUT, open("cancellation_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote cancellation_results.json")


if __name__ == "__main__":
    main()
