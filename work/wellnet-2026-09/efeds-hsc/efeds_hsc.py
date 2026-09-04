"""The potential-depth test on eFEDS, scored against weak-lensing REDUCED SHEAR.

Run Z established that the hydrostatic route is structurally vacuous: the test
variable and the observable were both the gas density profile shape.  The repair
named there is to score against an observable that does not contain the density
profile.  Weak-lensing shear is that observable.

WHAT IS ACTUALLY AVAILABLE (see REPORT.md Sect. 1 and acquire/access_probes.json)
The HSC per-source shape catalogue is behind an account (HTTP 401 on every
archive path).  No per-cluster eFEDS shear profile is published anywhere.  The
ONE public shear measurement is Chiu+2022's stacked profile, recovered exactly
from the vector PDF by extract_shear_pdf.py.  So the within-class differential
test the brief specifies CANNOT be run, and this file does the three things that
can be done honestly instead:

  A  the design measurement -- how much within-class DeltaPhi_b leverage the
     eFEDS sample carries at matched g_b, under four declared boundary rules,
     and how collinear it is with every named competitor.  If this fails, the
     experiment is dead whatever the shear quality.
  B  the amplitude-and-shape test against the one public stacked profile, with
     a free amplitude (= the class step, which within a single class has no
     other content) as the primary null.
  C  the power projection for the experiment that the private per-cluster
     profiles would allow, with the noise model VALIDATED against the published
     stacked error bars.

DECLARED IN ADVANCE, BEFORE ANY RESIDUAL WAS LOOKED AT
  boundary rules   primary 'fixed10Mpc'; sensitivity 'fixed5Mpc', 'fixed3Mpc',
                   '2xR500', '10xrs'
  Phi_0            1e12 m^2/s^2   (Run AA gate scale, not fitted here)
  a0               1.2e-10 m/s^2  (RAR, not fitted here)
  stellar fraction M_b = M_gas x (1 + f_star), f_star = 0 primary, 0.15/0.30 sens
  truncation       r_t = 20 Mpc primary, 10/40 Mpc sensitivity
  cuts             finite density params; n0^2 > 0; rs > 0; z > 0; R500 > 0;
                   Vikhlinin beta > 1/3
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

import pipeline as P
import lead01 as L

HERE = os.path.dirname(os.path.abspath(__file__))
MPC, MSUN, G = P.MPC, P.MSUN, P.G
RULES = ["fixed10Mpc", "fixed5Mpc", "fixed3Mpc", "2xR500", "10xrs"]
PRIMARY_RULE = "fixed10Mpc"
SIGMA_E = 0.4            # HSC per-component shape dispersion, sqrt(sig^2+erms^2)
RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# --------------------------------------------------------------------- ingest
def load_efeds(require_hsc=False):
    d1 = L.read_tsv(os.path.join(P.LEAD01,
                                 "efeds_bahar2022_table1_density.tsv"))[1]
    d2 = L.read_tsv(os.path.join(P.LEAD01, "efeds_bahar2022_table2.tsv"))[1]
    assert len(d1) == 542 and len(d2) == 542, (len(d1), len(d2))
    t2 = {r["ID"]: r for r in d2}
    cov = None
    mp = os.path.join(HERE, "efeds_hsc_coverage_mask.npy")
    if require_hsc and os.path.exists(mp):
        cov = np.load(mp)
    cuts = {"ingested": len(d1), "finite_density_params": 0,
            "finite_z_R500": 0, "vikhlinin_beta_gt_1_3": 0}
    if cov is not None:
        cuts["on_sampled_HSC_coverage"] = 0
    recs = []
    for r1 in d1:
        r2 = t2.get(r1["ID"])
        if r2 is None:
            continue
        n0sq, rs_as = L.num(r1, "n0"), L.num(r1, "rs")
        eps, bet, alp = (L.num(r1, "epsilon"), L.num(r1, "beta"),
                         L.num(r1, "alpha"))
        if not np.all(np.isfinite([n0sq, rs_as, eps, bet, alp])) \
           or n0sq <= 0 or rs_as <= 0:
            continue
        cuts["finite_density_params"] += 1
        z, R500_am = L.num(r2, "z"), L.num(r2, "R500")
        if not np.all(np.isfinite([z, R500_am])) or z <= 0 or R500_am <= 0:
            continue
        cuts["finite_z_R500"] += 1
        if not bet > 1.0 / 3.0:
            continue
        cuts["vikhlinin_beta_gt_1_3"] += 1
        ra_, de_ = L.num(r2, "RAJ2000"), L.num(r2, "DEJ2000")
        if cov is not None:
            i, j = int((ra_ - 126.0) / 0.1), int((de_ + 3.0) / 0.1)
            if not (0 <= i < cov.shape[0] and 0 <= j < cov.shape[1]
                    and cov[i, j]):
                continue
            cuts["on_sampled_HSC_coverage"] += 1
        DA = float(P.d_ang(z))
        recs.append(dict(
            id=r1["ID"], z=z, DA=DA,
            rs=rs_as * P.ARCSEC * DA, R500=R500_am * P.ARCMIN * DA,
            n0sq=n0sq, eps=eps, beta=bet, alpha=alp,
            e_n0sq=max(L.num(r1, "e_n0"), L.num(r1, "E_n0")),
            e_rs=max(L.num(r1, "e_rs"), L.num(r1, "E_rs")) * P.ARCSEC * DA,
            e_eps=max(L.num(r1, "e_epsilon"), L.num(r1, "E_epsilon")),
            e_beta=max(L.num(r1, "e_beta"), L.num(r1, "E_beta")),
            e_alpha=max(L.num(r1, "e_alpha"), L.num(r1, "E_alpha")),
            Mgas500_pub=L.num(r2, "Mgas500"),
            l_Mgas=r2.get("l_Mgas500", "").strip(),
            T=L.num(r2, "Tcex500") if L.num(r2, "Tcex500") > 0
            else L.num(r2, "T500"),
            RA=L.num(r2, "RAJ2000"), DE=L.num(r2, "DEJ2000")))
    return recs, cuts


def gate_mgas(sysd, recs):
    """Reproduce Bahar+2022's own published M_gas,500.  Reused from Run Z."""
    rat = []
    for s, r in zip(sysd, recs):
        if r["l_Mgas"] == "<" or not (r["Mgas500_pub"] > 0):
            continue
        mine = np.interp(s.R500, s.r, s.M_gas) / MSUN / 1e12
        rat.append(mine / r["Mgas500_pub"])
    rat = np.array(rat)
    med, sc = float(np.median(rat)), float(np.std(np.log10(rat)))
    ok = rat.size >= 20 and 0.8 < med < 1.25 and sc < 0.15
    print(f"   GATE M_gas,500: n = {rat.size}, median mine/published = "
          f"{med:.4f}, scatter {sc:.4f} dex  ->  {'PASS' if ok else 'FAIL'}")
    RES["gate_mgas500"] = {"n": int(rat.size), "median_ratio": med,
                           "scatter_dex": sc, "passed": bool(ok)}
    if not ok:
        raise SystemExit("gas-mass gate failed; nothing downstream is valid")


# ------------------------------------------------------- A. design measurement
def section_A(sysd):
    hdr("A.  DESIGN -- within-class leverage and competitor collinearity")
    R_eval = np.geomspace(0.3, 4.5, 12) * MPC        # the shear-measured range
    rows = {k: [] for k in ("id", "logr", "loggb", "logS", "logMb", "logT",
                            "fgas", "z", "logR500", "slope")}
    for k in RULES:
        rows[k] = []
    rows["ok"] = {k: [] for k in RULES}
    for s in sysd:
        gb = np.interp(R_eval, s.r, s.g_b)
        if not np.all(np.isfinite(gb)) or np.any(gb <= 0):
            continue
        dph, okr = {}, {}
        for k in RULES:
            d, r_ref = s.dphi(k)
            dph[k] = np.interp(R_eval, s.r, d)
            # DeltaPhi is only DEFINED inside the reference radius; a rule
            # whose r_ref falls inside the shear-measured range simply cannot
            # be evaluated there, and those points are masked rather than
            # clipped to a floor
            okr[k] = (R_eval < 0.9 * r_ref) & (dph[k] > 0)
            dph[k] = np.where(okr[k], dph[k], np.nan)
        for k in RULES:
            rows["ok"][k] += list(okr[k])
        rows["id"] += [s.id] * len(R_eval)
        rows["logr"] += list(np.log10(R_eval / MPC))
        rows["loggb"] += list(np.log10(gb))
        # the Run Z shape factor S = |Phi_b| / (g_b r), recomputed here so the
        # identity can be re-tested on the DIFFERENCE rather than the absolute
        rows["logS"] += list(np.log10(dph[PRIMARY_RULE] / (gb * R_eval)))
        rows["logMb"] += list(np.log10(np.full(len(R_eval),
                                               np.interp(s.R500, s.r, s.M_b)
                                               / MSUN)))
        rows["logT"] += list(np.full(len(R_eval),
                                     math.log10(max(s.T, 1e-3))))
        # f_gas from the catalogue's own R500: M500 = 500 rho_c(z) (4pi/3) R500^3
        rho_c = 3.0 * (L.H0 * L.E(s.z)) ** 2 / (8.0 * math.pi * G)
        M500 = 500.0 * rho_c * 4.0 / 3.0 * math.pi * s.R500 ** 3
        rows["fgas"] += list(np.full(len(R_eval),
                                     np.interp(s.R500, s.r, s.M_gas) / M500))
        rows["z"] += list(np.full(len(R_eval), s.z))
        rows["logR500"] += list(np.full(len(R_eval),
                                        math.log10(s.R500 / MPC)))
        rows["slope"] += list(np.log10(np.abs(np.interp(
            R_eval, s.r, s.slope))))
        for k in RULES:
            rows[k] += list(np.log10(dph[k] / P.PHI0))
    okmask = {k: np.array(v, bool) for k, v in rows.pop("ok").items()}
    D = {k: np.array(v) for k, v in rows.items()}
    n_sys = len(set(D["id"]))
    print(f"\n   {n_sys} systems x {len(R_eval)} radii = {len(D['logr'])} "
          f"points, over R = 0.3-4.5 Mpc (the shear-measured range)")

    # A1 leverage at matched g_bar, per boundary rule
    print("\n   A1  within-class spread of x_Phi at matched g_b "
          "(0.1 dex bins in log g_b, median over bins)")
    lev = {}
    bins = np.floor(D["loggb"] / 0.1)
    for k in RULES:
        mk = okmask[k]
        sp = [np.nanstd(D[k][(bins == b) & mk]) for b in np.unique(bins)
              if np.sum((bins == b) & mk) >= 20]
        lev[k] = float(np.median(sp)) if sp else float("nan")
        print(f"       {k:12s}  {lev[k]:.3f} dex   "
              f"(x_Phi {np.nanmin(D[k]):+.2f} to {np.nanmax(D[k]):+.2f}; "
              f"{100 * mk.mean():.0f}% of points inside r_ref)")
    print(f"       reference points: SPARC alone 0.309, eFEDS |Phi_b| "
          f"absolute (Run Z) 0.185, full six-rung ladder 0.766 (86% class)")
    RES["A1_leverage_dex"] = lev

    # A2 collinearity with the named competitors
    print("\n   A2  R^2 of x_Phi (primary rule) on competitor sets.  Every one"
          "\n       of these is a COMPETITOR the standing brief names, run "
          "through\n       the identical pipeline.")
    D["loggb2"] = D["loggb"] ** 2
    D["logr2"] = D["logr"] ** 2
    D["loggbr"] = D["loggb"] * D["logr"]
    fin = np.isfinite(D[PRIMARY_RULE])
    y = D[PRIMARY_RULE][fin]
    sets = {
        "quadratic in (log g_b, log r)":
            ["loggb", "loggb2", "logr", "logr2", "loggbr"],
        "log M_b": ["logMb"], "log R500": ["logR500"], "log T": ["logT"],
        "f_gas": ["fgas"], "redshift": ["z"],
        "log M_b + log r": ["logMb", "logr"],
        "ALL competitors": ["loggb", "loggb2", "logr", "logr2", "loggbr",
                            "logMb", "logT", "fgas", "z", "logR500"],
    }
    r2 = {}
    for name, cols in sets.items():
        X = np.column_stack([np.ones_like(y)] + [D[c][fin] for c in cols])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        r2[name] = float(1 - np.var(res) / np.var(y))
        print(f"       {name:32s} R^2 {r2[name]:.4f}   residual "
              f"{np.std(res):.3f} dex")
    RES["A2_R2_of_xPhi_on_competitors"] = r2

    # A3 the Run Z identity, re-tested on the potential DIFFERENCE
    X = np.column_stack([np.ones_like(y), D["loggb"][fin],
                         D["loggb"][fin] ** 2, D["logr"][fin]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    ry = y - X @ b
    b2, *_ = np.linalg.lstsq(X, D["logS"][fin], rcond=None)
    rs_ = D["logS"][fin] - X @ b2
    c_id = float(np.corrcoef(ry, rs_)[0, 1])
    b3, *_ = np.linalg.lstsq(X, D["slope"][fin], rcond=None)
    r_sl = D["slope"][fin] - X @ b3
    c_sl = float(np.corrcoef(rs_, r_sl)[0, 1])
    print("\n   A3  the Run Z identity, re-measured on the DIFFERENCE")
    print(f"       corr(resid x_Phi, resid log S | g_b, g_b^2, r) = "
          f"{c_id:+.4f}   (Run Z on |Phi_b|: +1.0000)")
    print(f"       corr(log S, log |dln n_e/dln r|)               = "
          f"{float(np.corrcoef(D['logS'], D['slope'])[0, 1]):+.4f}   "
          f"(Run Z: -0.8735)")
    print(f"       ... and the hydrostatic observable IS that slope.  The "
          f"shear observable is NOT.")
    RES["A3_identity"] = {
        "corr_resid_xPhi_resid_logS": c_id,
        "corr_logS_logdensityslope":
            float(np.corrcoef(D["logS"], D["slope"])[0, 1]),
        "corr_resid_logS_resid_slope": c_sl}
    return D, R_eval


# --------------------------------------- B. the test against the public stack
def read_stack():
    R, g, elo, ehi, used = [], [], [], [], []
    with open(os.path.join(HERE, "efeds_stacked_shear.tsv"),
              encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#") or ln.startswith("R_hinv"):
                continue
            p = ln.split("\t")
            R.append(float(p[0]))
            g.append(float(p[1]))
            elo.append(float(p[2]))
            ehi.append(float(p[3]))
            used.append(int(p[4]))
    R = np.array(R) / P.H_LITTLE * MPC              # h^-1 Mpc -> proper Mpc
    return (R, np.array(g), 0.5 * (np.array(elo) + np.array(ehi)),
            np.array(used, bool))


DLNR = math.log(3.5 / 0.2) / 10.0          # Chiu's ten log bins


def source_counts(sysd, src, R, theta_cap_arcmin):
    """N_ij, the number of selected sources in bin i around cluster j.

    Chiu weights each cluster by the DIAGONAL of its lensing covariance, which
    for shape noise is the source count in the annulus.  The angular aperture
    has to be capped: a z = 0.017 system subtends 5 Mpc at 4 degrees, and there
    is no HSC coverage at 4 degrees around one cluster, so the uncapped count
    hands 19% of the whole stack to a single nearby group.  The cap is
    calibrated below against the PUBLISHED stacked error bars -- error bars
    only, never the shear values, so it cannot bias the test.
    """
    N = np.zeros((len(sysd), len(R)))
    for j, s in enumerate(sysd):
        th = R / s.DA / P.ARCMIN
        A = 2 * math.pi * th ** 2 * DLNR
        N[j] = np.where(th <= theta_cap_arcmin, src.neff(s.z) * A, 0.0)
    return N


def stack_weights(sysd, src, R, theta_cap_arcmin=45.0):
    N = source_counts(sysd, src, R, theta_cap_arcmin)
    col = N.sum(axis=0)
    return N / np.where(col > 0, col, 1.0)      # (n_sys, n_R), columns sum to 1


def predict_stack(sysd, src, R, W, beta_pd=0.0, amp=0.0, rule=PRIMARY_RULE,
                  law="rar", r_trunc=20.0, tilt=0.0, a0_scale=1.0,
                  want_parts=False):
    acc = np.zeros(len(R))
    bad = 0
    for s, wj in zip(sysd, W):
        gp, kap, gam = s.reduced_shear(R, src, r_trunc_mpc=r_trunc,
                                       beta_pd=beta_pd, amp=amp, rule=rule,
                                       law=law, tilt=tilt, a0_scale=a0_scale)
        if np.any(s.last_dM < -1e-6 * np.abs(s.last_dM).max()):
            bad += 1
        acc += wj * gp
    if want_parts:
        return acc, bad
    return acc


def profile_grid(sysd, src, R, w, grid_b, **kw):
    """Stacked prediction on a grid of beta, computed once and reused."""
    return np.array([predict_stack(sysd, src, R, w, beta_pd=b, **kw)
                     for b in grid_b])


def fit_from_grid(P_grid, grid_b, y, err, m, grid_amp):
    """Profile chi^2 over the free amplitude; return (beta_hat, ci68, prof)."""
    prof = np.empty(len(grid_b))
    for i, p in enumerate(P_grid):
        c = [(float(np.sum((((p * 10 ** a) - y) / err)[m] ** 2)))
             for a in grid_amp]
        prof[i] = min(c)
    ok = grid_b[prof <= prof.min() + 1.0]
    return (float(grid_b[int(np.argmin(prof))]),
            ([float(ok.min()), float(ok.max())] if ok.size else None), prof)


def section_B(sysd, src):
    hdr("B.  THE TEST -- baryons + RAR forward-modelled into reduced shear")
    R, gobs, gerr, used = read_stack()
    m = used

    # B0  the aperture cap and the noise model, calibrated on the PUBLISHED
    #     stacked error bars alone
    print("\n   B0  noise model.  sigma_stack(R)^-2 = sum_j N_ij / sigma_e^2 "
          "must\n       reproduce the published error bars; the one free "
          "quantity is the\n       angular aperture cap, and it is fitted to "
          "the ERROR BARS ONLY.")
    scan = []
    for cap in (15, 20, 25, 30, 35, 40, 50, 60, 80, 120, 1e9):
        N = source_counts(sysd, src, R, cap)
        pe = 1.0 / np.sqrt(np.maximum(N.sum(axis=0), 1e-9)) * SIGMA_E
        scan.append((cap, float(np.median(pe / gerr)),
                     float(np.std(pe / gerr))))
    for cap, r, sd in scan:
        print(f"       theta_cap {cap:>9.0f}'  predicted/published error "
              f"{r:.3f} +- {sd:.3f}")
    # The cap is calibrated so that the predicted stacked errors MATCH the
    # published ones.  Only error bars enter, never the shear values, so this
    # cannot bias the test.  It matters: uncapped weights hand the whole stack
    # to a handful of very nearby systems whose 3.5 h^-1 Mpc aperture subtends
    # several degrees, which no cluster weak-lensing analysis actually uses.
    cap, ratio, _sd = min(scan, key=lambda t: abs(t[1] - 1.0))
    print(f"       -> adopting theta_cap = {cap:.0f} arcmin "
          f"(predicted/published error {ratio:.3f}); "
          f"C4 reports the uncapped\n          and equal-weight variants as "
          f"the systematic")
    W = stack_weights(sysd, src, R, cap)
    neff_sys = 1.0 / np.sum(W ** 2, axis=0)
    zbar = np.sum(W * np.array([s.z for s in sysd])[:, None], axis=0)
    print(f"       effective number of systems per bin "
          f"{neff_sys.min():.0f} - {neff_sys.max():.0f}; effective redshift "
          f"{zbar.min():.3f} - {zbar.max():.3f}")
    RES["B0_noise_and_weights"] = {
        "theta_cap_arcmin": float(cap),
        "scan": [{"cap": float(a), "ratio": b, "sd": c} for a, b, c in scan],
        "N_eff_systems_per_bin": [float(x) for x in neff_sys],
        "effective_redshift_per_bin": [float(x) for x in zbar]}

    # B1  the fixed-law predictions
    print("\n   B1  predicted stacked reduced shear, no free parameters")
    preds = {}
    for lab, kw in [("Newtonian baryons", dict(law="newton")),
                    ("RAR, beta = 0", dict(law="rar")),
                    ("RAR, beta = +0.17188 (Run R, frozen)",
                     dict(law="rar", beta_pd=0.17188))]:
        p, bad = predict_stack(sysd, src, R, W, want_parts=True, **kw)
        preds[lab] = p
        chi2 = float(np.sum(((p - gobs) / gerr)[m] ** 2))
        print(f"       {lab:38s} chi2 = {chi2:8.1f}/{m.sum()}  "
              f"mean g_pred/g_obs = {np.mean((p / gobs)[m]):.4f}  "
              f"(non-monotone M_dyn: {bad})")
        RES.setdefault("B1_models", {})[lab] = {
            "chi2": chi2, "ndof": int(m.sum()),
            "mean_ratio_to_obs": float(np.mean((p / gobs)[m])),
            "nonmonotone_Mdyn": int(bad),
            "g_pred": [float(x) for x in p]}
    RES["B1_observed"] = {"R_Mpc": [float(x / MPC) for x in R],
                          "g_obs": [float(x) for x in gobs],
                          "g_err": [float(x) for x in gerr],
                          "used_fiducial": [bool(x) for x in used]}

    # B2  model comparison.  Inside ONE class the class step has no content
    #     beyond a free amplitude, so M1 is the strongest possible null; the
    #     radius tilt and the free a0 are the two other one-parameter
    #     competitors that can reshape the profile.
    print("\n   B2  model comparison at matched parameter count.")
    print("       Inside one class a class indicator IS a free amplitude, so "
          "M1 is\n       the primary null.  M3 and M4 are the competitors "
          "that can reshape\n       the radial profile with one parameter, "
          "exactly as beta does.")
    grid_amp = np.linspace(-0.5, 2.5, 151)
    grid_b = np.linspace(-1.5, 2.0, 71)
    grid_t = np.linspace(-1.5, 1.5, 61)
    grid_a0 = np.geomspace(0.05, 200.0, 61)

    def chi2_of(p):
        return float(np.sum(((p - gobs) / gerr)[m] ** 2))

    def best_amp(p):
        c = [chi2_of(p * 10 ** a) for a in grid_amp]
        i = int(np.argmin(c))
        return float(c[i]), float(grid_amp[i])

    Pg = profile_grid(sysd, src, R, W, grid_b)
    Pt = np.array([predict_stack(sysd, src, R, W, tilt=t) for t in grid_t])
    Pa = np.array([predict_stack(sysd, src, R, W, a0_scale=x)
                   for x in grid_a0])

    p0 = preds["RAR, beta = 0"]
    c0 = chi2_of(p0)
    c1, amp_hat = best_amp(p0)
    prof_b = np.array([best_amp(p)[0] for p in Pg])
    prof_t = np.array([best_amp(p)[0] for p in Pt])
    prof_a = np.array([best_amp(p)[0] for p in Pa])
    ib, it, ia = (int(np.argmin(prof_b)), int(np.argmin(prof_t)),
                  int(np.argmin(prof_a)))
    n = int(m.sum())
    tab = [("M0  RAR only", 0, c0),
           ("M1  + free amplitude  (CLASS STEP)", 1, c1),
           ("M2  + beta x_Phi      (HYPOTHESIS)", 2, float(prof_b[ib])),
           ("M3  + gamma log r     (competitor)", 2, float(prof_t[it])),
           ("M4  + free a0         (competitor)", 2, float(prof_a[ia]))]
    bics = [c + k * math.log(n) for _, k, c in tab]
    b0 = min(bics)
    print(f"\n       {'model':38s} {'k':>2s} {'chi2':>8s} {'BIC':>8s} "
          f"{'dBIC':>7s}")
    for (lab, k, c), bic in zip(tab, bics):
        print(f"       {lab:38s} {k:2d} {c:8.2f} {bic:8.2f} {bic - b0:+7.2f}")
    print(f"\n       best fits: amplitude {amp_hat:+.3f} dex; "
          f"beta {grid_b[ib]:+.3f}; gamma {grid_t[it]:+.3f}; "
          f"a0/a0_RAR {grid_a0[ia]:.3g}")

    def ci(grid, prof):
        ok = grid[prof <= prof.min() + 1.0]
        if not ok.size:
            return None, False
        return ([float(ok.min()), float(ok.max())],
                bool(ok.min() <= grid[0] + 1e-9 or ok.max() >= grid[-1] - 1e-9))

    cb, edge_b = ci(grid_b, prof_b)
    print(f"       beta 68% "
          + ("UNCONSTRAINED" if cb is None else
             f"[{cb[0]:+.3f}, {cb[1]:+.3f}]")
          + ("  (hits the grid edge)" if edge_b else ""))
    RES["B2_model_comparison"] = {
        "n_bins_used": n,
        "table": [{"model": t[0], "k": t[1], "chi2": t[2], "bic": float(b)}
                  for t, b in zip(tab, bics)],
        "amp_hat_dex": amp_hat,
        "beta_hat": float(grid_b[ib]), "beta_ci68": cb,
        "beta_ci_hits_edge": edge_b,
        "gamma_hat": float(grid_t[it]),
        "a0_scale_hat": float(grid_a0[ia]),
        "beta_profile_grid": [float(x) for x in grid_b],
        "beta_profile_chi2": [float(x) for x in prof_b],
        "dBIC_hypothesis_minus_best_competitor":
            float(bics[2] - min(bics[1], bics[3], bics[4]))}
    verdict = ("potential depth WINS on BIC"
               if bics[2] == b0 else
               f"potential depth LOSES on BIC by {bics[2] - b0:.2f}")
    print(f"\n       VERDICT on the one public stacked profile: {verdict}")
    return R, gobs, gerr, used, W, preds, Pg, grid_b, grid_amp


# ------------------------------------------------------------ C. sensitivity
def section_C(sysd, recs, src, R, gobs, gerr, used, w, grid_b, grid_amp):
    hdr("C.  SENSITIVITY -- boundary rule, stars, truncation, weights")
    m = used
    out = {}
    gb = grid_b[::5]        # coarse: this section is a systematic sweep,
                          # not the headline fit

    def run(tag, **kw):
        Pg = profile_grid(sysd, src, R, w, gb, **kw)
        bh, ci, prof = fit_from_grid(Pg, gb, gobs, gerr, m, grid_amp)
        edge = ci is None or ci[0] <= gb[0] + 1e-9 or ci[1] >= gb[-1] - 1e-9
        print(f"       {tag:26s} beta = {bh:+.3f}   68% "
              + ("UNCONSTRAINED" if ci is None
                 else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
              + ("  (edge)" if edge else ""))
        return {"beta": bh, "ci68": ci, "hits_grid_edge": bool(edge),
                "chi2_min": float(prof.min())}

    print("\n   C1  boundary rule.  Run Z showed the rule DEFINES the "
          "variable, so\n       every defensible rule is run and the primary "
          "was declared first.")
    for rule in RULES:
        out.setdefault("boundary_rule", {})[rule] = run(rule, rule=rule)

    print("\n   C2  truncation radius of the dynamical density.  Under the "
          "RAR the\n       outer dynamical density falls as r^-2, so Sigma "
          "depends on r_t\n       logarithmically -- this is a real "
          "systematic, not a grid choice.")
    for rt in (10.0, 20.0, 40.0):
        out.setdefault("truncation_Mpc", {})[str(rt)] = run(
            f"r_t = {rt:.0f} Mpc", r_trunc=rt)

    print("\n   C3  stellar baryons.  The eFEDS tables give no stellar mass, "
          "so M_b\n       is gas only in the primary; these are uniform "
          "rescalings of M_b.")
    for fs in (0.0, 0.15, 0.30):
        alt = [P.System(rc, f_star=fs) for rc in recs]
        Pg = profile_grid(alt, src, R, w, gb)
        bh, ci, prof = fit_from_grid(Pg, gb, gobs, gerr, m, grid_amp)
        out.setdefault("f_star", {})[str(fs)] = {"beta": bh, "ci68": ci}
        print(f"       f_star = {fs:4.2f}            beta = {bh:+.3f}   68% "
              + ("UNCONSTRAINED" if ci is None
                 else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"))

    print("\n   C4  stacking weights")
    for tag, ww in (("inverse shape noise", w),
                    ("equal weights",
                     np.full((len(sysd), len(R)), 1.0 / len(sysd))),
                    ("uncapped aperture",
                     stack_weights(sysd, src, R, 1e9))):
        Pg = profile_grid(sysd, src, R, ww, gb)
        bh, ci, prof = fit_from_grid(Pg, gb, gobs, gerr, m, grid_amp)
        out.setdefault("weights", {})[tag] = {"beta": bh, "ci68": ci}
        print(f"       {tag:26s} beta = {bh:+.3f}")

    print("\n   C5  radial range (the paper excludes R < 0.5 h^-1 Mpc from "
          "its own\n       fiducial fit because of miscentering)")
    Pg = profile_grid(sysd, src, R, w, gb)
    for tag, mm in (("R > 0.5 h^-1 Mpc (fiducial)", used),
                    ("all ten bins", np.ones(len(R), bool)),
                    ("R > 1 h^-1 Mpc", R / MPC * P.H_LITTLE > 1.0)):
        bh, ci, prof = fit_from_grid(Pg, gb, gobs, gerr, mm, grid_amp)
        out.setdefault("radial_range", {})[tag] = {"beta": bh, "ci68": ci,
                                                   "nbins": int(mm.sum())}
        print(f"       {tag:26s} beta = {bh:+.3f}  ({int(mm.sum())} bins)")
    RES["C_sensitivity"] = out


# --------------------------------- D. shared-quantity null with real covariance
def section_D(sysd, recs, src, R, gobs, gerr, used, w, n_mc=200):
    hdr("D.  SHARED-QUANTITY NULL, simulated with the actual error covariance")
    print("""
   Writing out the construction expression for BOTH axes is the check that
   caught all four previous artefacts in this programme:

       x_Phi(r)       = f( n0^2, r_s, alpha, beta_V, epsilon, z )  X-ray fit only
       g_+^obs(theta) = (1/2R(1+K)) sum_s w_s e_+,s / sum_s w_s    HSC shapes only

   They share NO input quantity.  In Run Z the two expressions shared the gas
   density fit -- the hydrostatic g_obs IS the density log-slope up to kT/r --
   so the naive estimator was guaranteed to find something.  Here that channel
   is absent by construction, not by assumption.  The Monte Carlo below is the
   numerical confirmation: the X-ray parameters are perturbed within their
   published errors, which moves g_b and x_Phi COHERENTLY, and the shear is
   drawn independently.""")
    m = used
    rng = np.random.default_rng(20260904)
    db = 0.10
    p0 = predict_stack(sysd, src, R, w)
    a_true = float(np.log10(np.sum(gobs[m] * p0[m] / gerr[m] ** 2)
                            / np.sum(p0[m] ** 2 / gerr[m] ** 2)))
    print(f"\n   truth for the null: RAR, beta = 0, amplitude "
          f"{a_true:+.4f} dex (the M1 fit)")

    def linear_beta(pb0, pbd, y):
        """Least-squares beta with the free amplitude projected out."""
        d = ((pbd - pb0) / db)[m]
        a = (pb0 * math.log(10.0))[m]
        z = (y - pb0)[m]
        iv = 1.0 / gerr[m] ** 2
        A = np.array([[np.sum(d * d * iv), np.sum(d * a * iv)],
                      [np.sum(d * a * iv), np.sum(a * a * iv)]])
        b = np.array([np.sum(d * z * iv), np.sum(a * z * iv)])
        return float(np.linalg.solve(A, b)[0])

    # validate the linearised estimator against the grid estimator on the data
    pbd = predict_stack(sysd, src, R, w, beta_pd=db)
    p_true = p0 * 10 ** a_true
    pbd_t = pbd * 10 ** a_true
    lin_on_data = linear_beta(p_true, pbd_t, gobs)
    print(f"   linearised estimator on the real data: beta = "
          f"{lin_on_data:+.4f}  (grid estimator gave "
          f"{RES['B2_fits']['joint']['beta']:+.3f})")

    bhat = []
    for it in range(n_mc):
        pert = []
        for s, rc in zip(sysd, recs):
            pert.append(P.System(dict(
                z=s.z, DA=s.DA, R500=s.R500, id=s.id, T=rc["T"],
                rs=max(s.rs + rng.normal(0, rc["e_rs"]), 0.05 * s.rs),
                n0sq=max(rc["n0sq"] + rng.normal(0, rc["e_n0sq"]),
                         1e-3 * rc["n0sq"]),
                eps=rc["eps"] + rng.normal(0, rc["e_eps"]),
                beta=max(rc["beta"] + rng.normal(0, rc["e_beta"]), 0.34),
                alpha=rc["alpha"] + rng.normal(0, rc["e_alpha"]))))
        q0 = predict_stack(pert, src, R, w) * 10 ** a_true
        qd = predict_stack(pert, src, R, w, beta_pd=db) * 10 ** a_true
        y = p_true + rng.normal(0, gerr)
        bhat.append(linear_beta(q0, qd, y))
        if it % 20 == 0:
            print(f"      ... {it + 1}/{n_mc}  running mean "
                  f"{np.mean(bhat):+.4f}")
    bhat = np.array(bhat)
    se = float(np.std(bhat) / math.sqrt(len(bhat)))
    print(f"\n   null expectation of beta-hat = {bhat.mean():+.4f} +- {se:.4f}"
          f"   (sd {bhat.std():.4f}, n = {len(bhat)})")
    print(f"   95% of the null lies in [{np.percentile(bhat, 2.5):+.3f}, "
          f"{np.percentile(bhat, 97.5):+.3f}]")
    z_null = bhat.mean() / se
    print(f"   offset of the null from zero: {z_null:+.2f} sigma_MC  ->  "
          f"{'no shared-quantity bias detected' if abs(z_null) < 3 else 'BIAS'}")
    RES["D_shared_quantity_null"] = {
        "n_mc": int(n_mc), "mean": float(bhat.mean()), "sem": se,
        "sd": float(bhat.std()),
        "p2.5": float(np.percentile(bhat, 2.5)),
        "p97.5": float(np.percentile(bhat, 97.5)),
        "z_of_null_from_zero": float(z_null),
        "amplitude_of_truth_dex": a_true,
        "linear_estimator_on_data": lin_on_data}


# ------------------------------------------------ E. responsiveness / injection
def section_E(sysd, src, R, gerr, used, w, Pg, grid_b, grid_amp, n_real=60):
    hdr("E.  RESPONSIVENESS -- d beta-hat / d beta_injected must not be zero")
    print("\n   A rank statistic in this programme was once bit-identical "
          "across three\n   decades of the parameter it was supposed to "
          "measure.  This is the check.")
    m = used
    rng = np.random.default_rng(7)
    inj = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
    rec = []
    for bt in inj:
        pt = np.array([np.interp(bt, grid_b, Pg[:, i])
                       for i in range(Pg.shape[1])]) * 10 ** 0.3
        hats = []
        for _ in range(n_real):
            y = pt + rng.normal(0, gerr)
            bh, _, _ = fit_from_grid(Pg, grid_b, y, gerr, m, grid_amp)
            hats.append(bh)
        rec.append((bt, float(np.mean(hats)), float(np.std(hats))))
        print(f"       beta_inj {bt:+.2f}  ->  beta_hat "
              f"{rec[-1][1]:+.4f} +- {rec[-1][2]:.4f}")
    x = np.array([r[0] for r in rec])
    y = np.array([r[1] for r in rec])
    slope = float(np.polyfit(x, y, 1)[0])
    print(f"\n   d beta-hat / d beta_inj = {slope:.4f}, spread of beta-hat "
          f"over the tested range {y.max() - y.min():.4f}")
    print(f"   -> {'PASS, the statistic responds' if abs(slope) > 0.5 else 'FAIL, statistic is degenerate'}")
    print(f"   the per-realisation sd {np.mean([r[2] for r in rec]):.3f} IS "
          f"the achievable sigma(beta) from one stacked profile")
    RES["E_responsiveness"] = {
        "slope": slope, "spread": float(y.max() - y.min()),
        "points": [{"inj": a, "hat": b, "sd": c} for a, b, c in rec],
        "sigma_beta_one_stacked_profile":
            float(np.mean([r[2] for r in rec]))}


def section_F(sysd, src, theta_cap):
    hdr("F.  POWER -- what the PRIVATE per-cluster profiles would deliver")
    print("""
   The 313 per-cluster shear profiles exist; they are simply not published.
   This is the Fisher forecast for the experiment they would allow, with the
   noise model already validated in B0 against the published stacked errors.
   The competitor is carried in the SAME Fisher matrix: if beta and a bare
   radius tilt gamma are degenerate even with per-cluster data, the experiment
   cannot decide the question no matter how much shear is available.""")
    Rb = np.geomspace(0.2, 3.5, 11) / P.H_LITTLE * MPC
    R = np.sqrt(Rb[1:] * Rb[:-1])
    db, dt = 0.05, 0.05
    F = np.zeros((3, 3))
    tot_sn2, nu = 0.0, 0
    for s in sysd:
        g0, _, _ = s.reduced_shear(R, src, beta_pd=0.0)
        gb_, _, _ = s.reduced_shear(R, src, beta_pd=db)
        gt_, _, _ = s.reduced_shear(R, src, tilt=dt)
        if not np.all(np.isfinite([g0, gb_, gt_])):
            continue
        d = np.array([(gb_ - g0) / db,            # beta   (hypothesis)
                      g0 * math.log(10.0),        # amp    (class step)
                      (gt_ - g0) / dt])           # gamma  (competitor)
        th = R / s.DA / P.ARCMIN
        N = np.where(th <= theta_cap,
                     src.neff(s.z) * 2 * math.pi * th ** 2 * DLNR, 0.0)
        iv = N / SIGMA_E ** 2
        F += (d * iv) @ d.T
        tot_sn2 += float(np.sum(g0 ** 2 * iv))
        nu += 1
    C = np.linalg.inv(F)
    C2 = np.linalg.inv(F[:2, :2])
    sb_fix = 1.0 / math.sqrt(F[0, 0])
    sb_amp = math.sqrt(C2[0, 0])
    sb_all = math.sqrt(C[0, 0])
    rho_bt = C[0, 2] / math.sqrt(C[0, 0] * C[2, 2])
    print(f"\n   {nu} systems, 10 radial bins each, shape noise sigma_e = "
          f"{SIGMA_E}, aperture cap {theta_cap:.0f}'")
    print(f"   predicted total S/N of the per-cluster ensemble "
          f"{math.sqrt(tot_sn2):.1f}   (published stacked S/N 29.0)")
    print(f"\n   sigma(beta), nothing else free                {sb_fix:.4f}")
    print(f"   sigma(beta), free amplitude (class step)     {sb_amp:.4f}")
    print(f"   sigma(beta), + free radius tilt (competitor) {sb_all:.4f}")
    print(f"   correlation(beta, gamma) after marginalising {rho_bt:+.4f}")
    print(f"\n   Run R fitted beta = +0.17188 with a systematic floor of "
          f"beta_spurious ~ 0.096.")
    print(f"   statistical detection of beta = 0.17188:  "
          f"{0.17188 / sb_amp:5.1f} sigma with the class step free, "
          f"{0.17188 / sb_all:5.1f} sigma with the competitor free")
    print(f"   the experiment is "
          f"{'SYSTEMATICS-limited' if sb_all < 0.096 else 'STATISTICS-limited'}"
          f": sigma_stat {sb_all:.3f} vs sigma_syst 0.096")
    RES["F_power"] = {
        "n_systems": nu, "theta_cap_arcmin": float(theta_cap),
        "sigma_beta_alone": sb_fix,
        "sigma_beta_marg_amplitude": sb_amp,
        "sigma_beta_marg_amplitude_and_tilt": sb_all,
        "corr_beta_gamma": float(rho_bt),
        "predicted_ensemble_SN": float(math.sqrt(tot_sn2)),
        "sigma_stat_over_sigma_syst": float(sb_all / 0.096),
        "detection_sigma_of_RunR_beta": float(0.17188 / sb_all)}


# --------------------------------------------------------------------- driver
def main():
    hdr("eFEDS x HSC -- potential depth against RAW WEAK-LENSING SHEAR")
    recs, cuts = load_efeds(require_hsc=True)
    print("\n   cuts, declared in the module docstring before any residual:")
    prev = cuts["ingested"]
    for k, v in cuts.items():
        print(f"      {k:26s} {v:4d}   (dropped {prev - v})")
        prev = v
    RES["cuts"] = cuts
    sysd = [P.System(r) for r in recs]
    gate_mgas(sysd, recs)

    print("\n   GATE: monotone M_dyn and non-negative rho_dyn under every law")
    bad = 0
    for s in sysd:
        for kw in (dict(law="newton"), dict(law="rar"),
                   dict(law="rar", beta_pd=0.17188)):
            g = s.g_pred(**kw)
            M = g * s.r ** 2 / G
            rt = s.trunc_for(kw.get("rule", PRIMARY_RULE), 20.0) * MPC
            ins = s.r[1:] <= rt
            if np.any(np.diff(M)[ins] < -1e-9 * np.abs(M).max()):
                bad += 1
                break
    print(f"      non-monotone M_dyn in {bad} of {len(sysd)} systems  "
          f"->  {'PASS' if bad == 0 else 'CHECK'}")
    RES["gate_monotone_Mdyn_failures"] = bad

    print("\n   HSC source population, calibrated on Chiu+2022's own quoted "
          "source densities")
    src = P.Sources(P.fit_source_nz())

    D, R_eval = section_A(sysd)
    R, gobs, gerr, used, w, preds, Pg, grid_b, grid_amp = section_B(sysd, src)
    section_C(sysd, recs, src, R, gobs, gerr, used, w, grid_b, grid_amp)
    section_F(sysd, src, RES["B0_noise_and_weights"]["theta_cap_arcmin"])
    # Sections D (shared-quantity null) and E (responsiveness) are run on the
    # DECADE per-cluster data instead -- see decade_test.py -- because there
    # they act on the real observable rather than on a single stacked profile.

    with open(os.path.join(HERE, "hsc_stack_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(RES, f, indent=1)
    print("\n   wrote efeds_hsc_results.json")


if __name__ == "__main__":
    main()
