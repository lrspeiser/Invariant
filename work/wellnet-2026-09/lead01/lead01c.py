"""Lead 01, part 3: the within-class bound on a potential-depth term.

Part 2 answered the leverage question and the answer was no: resolved profiles
give 0.185 dex of |Phi_b| spread at fixed g_bar, only 1.1x the two-overdensity-
radius cap and below SPARC's own 0.309. The identity
log|Phi_b| = log g_bar + log r + log S survives resolution, because S is bounded.

But the measurement is still worth making, and for a reason that has nothing to
do with the leverage being large. Run R's cleanest constraint on the
potential-depth hypothesis was a WITHIN-CLASS bound from SPARC -- partial
correlation +0.018, CI [-0.118, +0.145] -- because it contains no class boundary
and therefore no class-level systematic. eFEDS supplies a second, completely
independent within-class bound:

    different objects        groups, not spirals
    different scales         g_bar/a0 = 1e-3 to 0.12, four decades below SPARC
    different observable     hydrostatic g_obs from a measured n_e(r) and a
                             measured T, not a rotation velocity
    different systematics    X-ray calibration and hydrostatic bias, not
                             stellar mass-to-light

Two independent within-class bounds constrain the hypothesis in a way one never
could, and neither is contaminated by the galaxy/cluster step that beat the law
on BIC in Run R.

WHAT IS FITTED

    log nu_obs = quadratic(log g_bar) + beta log|Phi_b| ,   q = 2 beta

with nu_obs = g_obs/g_bar, one row per radial point, bootstrapped at the SYSTEM
level because points within a system are strongly correlated. The null is
SIMULATED with the actual error covariance rather than assumed to be zero, since
g_bar appears in both nu_obs and |Phi_b| -- the shared-denominator trap that has
now fired three times in this programme.

Baryons are gas-only, declared. Adding stars raises log g_bar and log|Phi_b| by
the same amount and lowers log nu_obs by it, i.e. exactly along the degeneracy,
so controlling for log g_bar removes it to first order. It is not removed for
anything that changes the gas mass alone, and that is carried as a systematic.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A0 = 1.2e-10


def design(lg, lp):
    return np.column_stack([np.ones_like(lg), lg, lg ** 2, lp])


def fit_beta(lg, lp, ly):
    A = design(lg, lp)
    c, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(c[-1]), c


def main():
    d = np.load(os.path.join(HERE, "lead01_points.npz"), allow_pickle=True)
    ids, gb, ph, go = d["ids"], d["gbar"], d["phi"], d["gobs"]
    rk, fr = d["r_kpc"], d["frac"]
    print("=" * 78)
    print("LEAD 01c -- within-class bound on a potential-depth term")
    print("=" * 78)

    nu = go / gb
    lg, lp, ly = np.log10(gb), np.log10(ph), np.log10(nu)
    ok = np.isfinite(lg) & np.isfinite(lp) & np.isfinite(ly) & (nu > 0)
    ids, lg, lp, ly, gb, fr = ids[ok], lg[ok], lp[ok], ly[ok], gb[ok], fr[ok]
    uniq = np.unique(ids)
    print(f"\n   {len(lg):,} points over {uniq.size} systems")
    print(f"   g_bar/a0        {gb.min()/A0:.4f} .. {gb.max()/A0:.4f}  "
          "(entirely deep-MOND)")
    print(f"   nu_obs = g_obs/g_bar   median {np.median(nu):.2f}, "
          f"16-84th {np.percentile(nu,16):.2f} - {np.percentile(nu,84):.2f}")

    # --- the RAR reference in this regime, for orientation
    x = gb / A0
    nu_rar = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-30))))
    print(f"   nu_RAR predicted here  median {np.median(nu_rar):.2f}")
    print(f"   nu_obs / nu_RAR        median {np.median(nu/nu_rar):.3f}   "
          f"({np.median(np.log10(nu/nu_rar)):+.3f} dex)")

    # ---------------------------------------------------------------- the fit
    beta, coef = fit_beta(lg, lp, ly)
    print(f"\n   fitted beta (coefficient of log|Phi_b|) = {beta:+.4f}   "
          f"q = 2 beta = {2*beta:+.4f}")

    # --- system-level bootstrap
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(2000):
        pick = rng.choice(uniq, uniq.size, replace=True)
        m = np.concatenate([np.where(ids == u)[0] for u in pick])
        try:
            b, _ = fit_beta(lg[m], lp[m], ly[m])
            if np.isfinite(b):
                boot.append(b)
        except Exception:                      # noqa: BLE001
            pass
    boot = np.array(boot)
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"   system-level bootstrap 95% CI  [{blo:+.4f}, {bhi:+.4f}]   "
          f"sd {boot.std():.4f}")

    # ------------------------------------------------- the shared-denom null
    # Truth: log nu depends on g_bar ALONE. Generate observed quantities with
    # the real error structure -- a coherent per-system gas-mass error moves
    # log g_bar and log|Phi_b| by +delta and log nu by -delta -- then refit and
    # see what beta the estimator returns when beta is truly zero.
    A0d = design(lg, lp)
    c0, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(lg), lg, lg ** 2]),
                             ly, rcond=None)
    base = np.column_stack([np.ones_like(lg), lg, lg ** 2]) @ c0
    resid_sd = float(np.std(ly - base))
    nulls = []
    for _ in range(2000):
        dsys = {u: rng.normal(0, 0.10) for u in uniq}     # coherent M_gas error
        dd = np.array([dsys[i] for i in ids])
        lg_o = lg + dd
        lp_o = lp + dd
        ly_o = base - dd + rng.normal(0, resid_sd, lg.size)
        try:
            b, _ = fit_beta(lg_o, lp_o, ly_o)
            if np.isfinite(b):
                nulls.append(b)
        except Exception:                      # noqa: BLE001
            pass
    nulls = np.array(nulls)
    nlo, nhi = np.percentile(nulls, [2.5, 97.5])
    print(f"\n   SIMULATED NULL (beta truly zero, real error covariance)")
    print(f"      null mean {nulls.mean():+.4f}, sd {nulls.std():.4f}, "
          f"95% [{nlo:+.4f}, {nhi:+.4f}]")
    print(f"      the naive assumption that the null is 0 is off by "
          f"{abs(nulls.mean()):.4f}")
    z = (beta - nulls.mean()) / np.sqrt(boot.std() ** 2 + nulls.std() ** 2)
    print(f"      observed beta vs its OWN null: z = {z:+.2f}")

    # ------------------------------------------- monotone-invariance gate
    print("\n   RESPONSIVENESS GATE: inject a known q and recover it")
    rows = []
    for qinj in (0.0, 0.1, 0.2, 0.4, 0.8):
        ly_inj = ly + 0.5 * qinj * (lp - lp.mean())
        b, _ = fit_beta(lg, lp, ly_inj)
        rows.append((qinj, b))
        print(f"      q_inj = {qinj:.2f}   beta_recovered = {b:+.4f}   "
              f"(expected {beta + 0.5*qinj:+.4f})")
    sl = np.polyfit([r[0] for r in rows], [r[1] for r in rows], 1)[0]
    print(f"      d(beta)/d(q) = {sl:.4f}   (unbiased value is 0.5000)")
    if abs(sl - 0.5) > 0.05:
        print("      WARNING: the estimator is not responding correctly.")

    # ------------------------------------------------------------- the bound
    q, qlo, qhi = 2 * beta, 2 * min(blo, nlo), 2 * max(bhi, nhi)
    print("\n" + "=" * 78)
    print(f"   WITHIN-CLASS BOUND, eFEDS groups")
    print(f"      q = {q:+.3f},  95% interval including the null offset "
          f"[{qlo:+.3f}, {qhi:+.3f}]")
    print(f"      q required to explain the cluster excess: +0.371")
    excl = qhi < 0.371 or qlo > 0.371
    print(f"      => q = 0.371 is {'EXCLUDED' if excl else 'NOT excluded'} "
          "by this sample")
    print("\n      the other within-class bound, from SPARC (Run N):")
    print("         partial corr +0.018, CI [-0.118, +0.145], |q| <= 0.29")
    print("      these two are independent in objects, scale, observable and")
    print("      systematics, and neither contains a class boundary.")

    out = {"n_points": int(lg.size), "n_systems": int(uniq.size),
           "gbar_over_a0": [float(gb.min() / A0), float(gb.max() / A0)],
           "nu_obs_median": float(np.median(nu)),
           "nu_over_nuRAR_median_dex": float(np.median(np.log10(nu / nu_rar))),
           "beta": beta, "beta_ci": [float(blo), float(bhi)],
           "null_mean": float(nulls.mean()), "null_sd": float(nulls.std()),
           "null_ci": [float(nlo), float(nhi)], "z_vs_own_null": float(z),
           "dbeta_dq": float(sl), "q": float(q),
           "q_interval": [float(qlo), float(qhi)],
           "q_required_for_cluster_excess": 0.371,
           "excludes_required_q": bool(excl)}
    with open(os.path.join(HERE, "lead01c_bound.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   written: lead01c_bound.json")


if __name__ == "__main__":
    main()
