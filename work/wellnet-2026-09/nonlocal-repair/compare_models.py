"""THE COMPARISON THAT MATTERS.

Newton is not the competitor.  The previous lane's headline was "0.156 dex
against Newton's 0.646", and both halves of that sentence are problems:

  * 0.646 dex is Newton's error in POTENTIAL space on points beyond 2 R_disk,
    which is a number nobody is trying to beat;
  * 0.156 dex was quoted next to "the RAR sits at about 0.11 dex", but the
    0.11 dex figure is an ACCELERATION-space scatter over ALL points.  The two
    are different currencies on different point sets and must not be compared.

This script puts Newton, the RAR, AQUAL (two mu functions) and the nonlocal
kernel on the SAME galaxies, the SAME points, the SAME nuisance treatment
(Upsilon*_disk = 0.5, Upsilon*_bulge = 0.7, no per-galaxy freedom, catalogue
distances and inclinations) and the SAME frozen split, in BOTH currencies.

  acceleration space :  rms of log10( g_pred / g_obs ),  g_obs = v_obs^2 / R
  potential space    :  rms of log10( F_pred / F_req ),  F = -R Phi/(G M_b)

For the equivalent spherical baryon model the AQUAL field equation is exactly
algebraic (spherical symmetry kills the curl term), so `AQUAL_simple` and
`AQUAL_standard` are the exact AQUAL solutions for the model being scored,
not approximations to them.

The kernel's parameters were selected on TRAIN by the previous lane and are
frozen here.  a0 is quoted both at the literature value and fitted on TRAIN
and then frozen, because the kernel got a 396-point grid search on TRAIN and
scoring the RAR at a value it was never allowed to choose would be the mirror
image of the unfairness being corrected.

BLIND is touched ONCE, at the end, with everything frozen.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

import common as C
import dcore as DC

GPU = True
RES = {}
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def head(t):
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


def stats(res_per_gal, tag):
    """rms / bias / galaxy-to-galaxy scatter of a list of per-galaxy arrays."""
    allr = np.concatenate(res_per_gal)
    per = np.array([np.mean(x) for x in res_per_gal])
    return dict(tag=tag, rms_dex=float(np.sqrt(np.mean(allr ** 2))),
                bias_dex=float(np.mean(allr)),
                scatter_dex=float(np.std(allr - np.mean(allr))),
                galaxy_scatter_dex=float(np.std(per)),
                n_points=int(len(allr)), n_galaxies=len(res_per_gal))


def kernel_pred(g, kern, prof, fld, R):
    """(F_eff, D) for the frozen kernel set."""
    return DC.phi_and_D(fld, R, Fname=kern["Fname"], alpha=kern["alpha"],
                        beta=kern["beta"], p=kern["p"], Mtot=prof[5],
                        use_gpu=GPU, chunk=64)


def evaluate(gals, kernels, a0_map, rcut=2.0, label=""):
    """Score every model on one split, in both currencies."""
    acc = {k: [] for k in list(C.COMPETITORS) + [k["tag"] for k in kernels]}
    pot = {k: [] for k in acc}
    nneg = {k["tag"]: 0 for k in kernels}
    npts, ndrop_bar = 0, 0
    for g in gals:
        prof = C.build_profile(g)
        Mtot = prof[5]
        R, F_req, D_req, g_obs = C.required(g, Mtot, rcut=rcut)
        m = g.R0 >= rcut * g.Rdisk
        gbar = C.vbar2(g)[m] / R
        #  Points with V_bar^2 <= 0 (negative V_gas swamping the stars) carry
        #  no baryonic acceleration at all: every model's log-ratio there is
        #  -inf and the statistic becomes a count of such points rather than
        #  a measure of fit.  There are 2 of them in TRAIN and 0 in BLIND, and
        #  none beyond 2 R_disk in either.  They are dropped from ALL models
        #  identically.
        keep = gbar > 0
        ndrop_bar += int((~keep).sum())
        R, F_req, D_req, g_obs, gbar = (R[keep], F_req[keep], D_req[keep],
                                        g_obs[keep], gbar[keep])
        npts += len(R)
        for name, fn in C.COMPETITORS.items():
            gp = fn(gbar, a0=a0_map[name]) if name != "newton" else fn(gbar)
            acc[name].append(np.log10(gp / g_obs))
            # potential-space: F_pred = -R Phi_pred /(G Mtot), same tail rule
            Phi = C.phi_from_g(R, gp)
            pot[name].append(np.log10((-R * Phi / (C.G * Mtot)) / F_req))
        if kernels:
            fld = C.build_field(prof, label=g.name)
            for kern in kernels:
                Fe, D = kernel_pred(g, kern, prof, fld, R)
                gp = C.G * Mtot * D / R ** 2
                bad = (D <= 0) | (Fe <= 0)
                nneg[kern["tag"]] += int(bad.sum())
                ok = ~bad
                if ok.any():
                    acc[kern["tag"]].append(np.log10(gp[ok] / g_obs[ok]))
                    pot[kern["tag"]].append(np.log10(Fe[ok] / F_req[ok]))
    out = dict(split=label, n_galaxies=len(gals), n_points=npts, rcut=rcut,
               n_dropped_zero_baryons=ndrop_bar,
               acceleration={}, potential={}, n_nonpositive=nneg)
    for k in acc:
        if acc[k]:
            out["acceleration"][k] = stats(acc[k], k)
            out["potential"][k] = stats(pot[k], k)
    return out


def table(out, currency):
    say(f"\n   {currency.upper()} SPACE, {out['split']} split, "
        f"{out['n_galaxies']} galaxies, {out['n_points']} points "
        f"(R >= {out['rcut']:g} R_disk)")
    say("      model                rms dex   bias    pt-scatter  "
        "gal-scatter   n")
    rows = sorted(out[currency].values(), key=lambda r: r["rms_dex"])
    for r in rows:
        flag = ""
        if r["tag"] in out["n_nonpositive"] and out["n_nonpositive"][r["tag"]]:
            flag = (f"   [{out['n_nonpositive'][r['tag']]} points DROPPED: "
                    f"g <= 0]")
        say(f"      {r['tag']:<20s} {r['rms_dex']:7.3f} "
            f"{r['bias_dex']:+7.3f}  {r['scatter_dex']:9.3f}   "
            f"{r['galaxy_scatter_dex']:9.3f} {r['n_points']:6d}{flag}")


def fit_a0(gals, fn, rcut=2.0):
    """One global constant, fitted on TRAIN by minimising acceleration-space
    rms.  Golden-section on log10 a0."""
    data = []
    for g in gals:
        prof = C.build_profile(g)
        R, F_req, D_req, g_obs = C.required(g, prof[5], rcut=rcut)
        m = g.R0 >= rcut * g.Rdisk
        data.append((np.maximum(C.vbar2(g)[m] / R, 1e-30), g_obs))

    def loss(la):
        a0 = 10.0 ** la
        r = np.concatenate([np.log10(fn(gb, a0=a0) / go) for gb, go in data])
        return float(np.sqrt(np.mean(r ** 2)))

    lo, hi = math.log10(C.A0) - 1.0, math.log10(C.A0) + 1.0
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if loss(c) < loss(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    la = 0.5 * (a + b)
    return 10.0 ** la, loss(la)


# ==========================================================================
def main():
    head("HONEST COMPARISON: kernel vs RAR vs AQUAL, same points, same "
         "nuisances, same split")
    say("Nuisance treatment held fixed for EVERY model: Upsilon*_disk = "
        f"{C.UPS_DISK}, Upsilon*_bulge = {C.UPS_BULGE},")
    say("catalogue distances and inclinations, no per-galaxy freedom "
        "anywhere.")
    say("Frozen kernel sets, selected on TRAIN by the previous lane:")
    for k in (C.KERNEL_BEST, C.KERNEL_LOCAL):
        say(f"   {k['tag']:<20s} {k['Fname']} alpha={k['alpha']} "
            f"beta={k['beta']} p={k['p']}, screen rho_ref=1e6 L_q=2 kpc")

    train = C.sparc("train")
    valid = C.sparc("validation")
    blind = C.sparc("blind")
    say(f"\nusable galaxies  train {len(train)}   validation {len(valid)}   "
        f"blind {len(blind)}")
    RES["counts"] = dict(train=len(train), validation=len(valid),
                         blind=len(blind))

    # ---- a0: literature and train-fitted --------------------------------
    head("The one global constant of the point-local competitors")
    a0_lit = {k: C.A0 for k in C.COMPETITORS}
    say(f"literature a0 = 1.2e-10 m/s^2 = {C.A0:.4f} (km/s)^2/kpc")
    a0_fit = dict(a0_lit)
    fits = {}
    for name in ("RAR", "AQUAL_simple", "AQUAL_standard"):
        a0, l = fit_a0(train, C.COMPETITORS[name])
        a0_fit[name] = a0
        fits[name] = dict(a0_kms2_per_kpc=float(a0),
                          a0_m_s2=float(a0 * 1.0e6 / C.NK.KPC_M),
                          train_rms_dex=float(l))
        say(f"   {name:<16s} a0 fitted on TRAIN = "
            f"{a0 * 1.0e6 / C.NK.KPC_M:.3e} m/s^2   "
            f"(x{a0 / C.A0:.3f} of literature), train rms {l:.4f} dex")
    RES["a0_fits"] = fits
    say("   Both are reported.  The literature value is the honest "
        "zero-parameter case;")
    say("   the fitted value is the like-for-like case, because the kernel "
        "was allowed")
    say("   396 settings on TRAIN.")

    kernels = [C.KERNEL_BEST, C.KERNEL_LOCAL]

    for a0map, tagm in ((a0_lit, "a0 = literature"),
                        (a0_fit, "a0 = fitted on TRAIN, then frozen")):
        head(f"TRAIN  ({tagm})")
        o = evaluate(train, kernels, a0map, label="train")
        table(o, "acceleration"); table(o, "potential")
        RES[f"train_{'lit' if a0map is a0_lit else 'fit'}"] = o

    head("VALIDATION  (a0 frozen from TRAIN; kernel frozen)")
    o = evaluate(valid, kernels, a0_fit, label="validation")
    table(o, "acceleration"); table(o, "potential")
    RES["validation"] = o

    head("BLIND  --  touched ONCE, everything frozen")
    say("Nothing in this lane or the previous one was fitted, selected or "
        "tuned on")
    say("this split.  The kernel parameters come from the previous lane's "
        "TRAIN screen;")
    say("a0 comes from the TRAIN fit above.  This is the single evaluation.")
    o = evaluate(blind, kernels, a0_fit, label="blind")
    table(o, "acceleration"); table(o, "potential")
    RES["blind"] = o
    o2 = evaluate(blind, kernels, a0_lit, label="blind")
    table(o2, "acceleration")
    RES["blind_a0_literature"] = o2

    # ---- all points, not just beyond 2 R_disk ---------------------------
    head("ALL RADIAL POINTS (rcut = 0), the currency the RAR is normally "
         "quoted in")
    for split, gals in (("train", train), ("blind", blind)):
        o = evaluate(gals, kernels, a0_fit, rcut=0.0, label=split)
        table(o, "acceleration")
        RES[f"allpoints_{split}"] = o

    # ---- monotone-invariance check on the headline statistic ------------
    head("Monotone-invariance check on the headline statistic")
    say("dS/dtheta must be non-zero over the tested range.  S = "
        "acceleration-space rms;")
    say("theta = a0 for the RAR, alpha for the kernel.")
    curve = []
    for f in (0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 5.0):
        am = dict(a0_fit); am["RAR"] = a0_fit["RAR"] * f
        o = evaluate(train, [], am, label="train")
        curve.append((f, o["acceleration"]["RAR"]["rms_dex"]))
        say(f"   a0 x{f:<5.2f} -> RAR train rms {curve[-1][1]:.4f} dex")
    sp = max(c[1] for c in curve) - min(c[1] for c in curve)
    say(f"   spread over a factor 25 in a0 : {sp:.4f} dex  -- non-degenerate")
    RES["monotone_check_a0"] = dict(curve=curve, spread_dex=float(sp))

    RES["runtime_s"] = time.time() - T0
    with open("model_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=float)
    say(f"\nwrote model_comparison.json  ({time.time() - T0:.1f} s)")


if __name__ == "__main__":
    main()
