"""
JOB 2.3 -- run the forward synthetic null, calibrate every test's own
false-positive rate, and measure responsiveness d(estimate)/d(injected).

Run AT found the obvious permutation test had an FPR of 0.53-0.70 against a
nominal 0.05.  Nothing here is trusted until it has been sized the same way.
"""
from __future__ import annotations
import json
import math
import time

import numpy as np

import ingest as I
import stats as S
import nullsim as N

KPC, MPC, MSUN, G = I.KPC, I.MPC, I.MSUN, I.G
OUT = {}
RNG = np.random.default_rng(20260904)


# ------------------------------------------------------- fast gridded NFW fit
class Fitter:
    """Precompute Sigma_NFW(R_FIT | M200, c200) once per redshift; each fit is
    then a chi^2 argmin over the grid."""

    def __init__(self, z, nM=160, nc=120):
        self.lM = np.linspace(math.log10(1.5e14), math.log10(9e15), nM)
        self.lc = np.linspace(math.log10(0.8), math.log10(14.0), nc)
        self.grid = np.empty((nM, nc, len(N.R_FIT)))
        for i, a in enumerate(self.lM):
            for j, b in enumerate(self.lc):
                self.grid[i, j] = N.nfw_sigma(N.R_FIT, 10 ** a, 10 ** b, z)
        self.flat = self.grid.reshape(-1, len(N.R_FIT))
        self.nM, self.nc = nM, nc

    def fit(self, Sig, err):
        chi = np.sum(((self.flat - Sig) / err) ** 2, axis=1)
        k = int(np.argmin(chi))
        i, j = divmod(k, self.nc)
        return 10 ** self.lM[i], 10 ** self.lc[j]


def build(D, y_per_cluster, s=0.0, r_break=None):
    T = I.points_table(D)
    C = D["clusters"]
    truths, rby = [], {}
    for n in sorted(C):
        m = T["name"] == n
        rby[n] = T["r"][m]
        truths.append(N.Truth(n, T["r"][m], np.log10(T["gb"][m]), C[n]["z"],
                              y_per_cluster[n], s,
                              R500_ref=C[n]["R500_lens"], r_break=r_break,
                              outer=(C[n]["M200"], C[n]["c200"])))
    return truths, rby


R_PIV = 0.5 * MPC


def realise(truths, rby, fitters, noise_pars, rng, sig_cache, noise=True):
    """noise_pars = (coherent fractional, independent fractional, radial tilt).

    Umetsu+2016's error budget is (i) measurement error, (ii) cosmic noise from
    uncorrelated large-scale structure, (iii) halo triaxiality and correlated
    substructure.  (ii) and (iii) are COHERENT across radial bins, so an
    independent-per-bin noise model averages down and cannot reproduce the quoted
    e_M500/M500 = 0.24 at any plausible amplitude (it saturates near 0.06).  The
    coherent term is also the component that carries the tautology: it moves the
    whole mass normalisation, hence both g_tot and R500.

    SECOND CALIBRATION DEFECT, found by checking the recovered c200 scatter
    against Umetsu's quoted e_c200/c200 = 0.301: a pure amplitude term reproduces
    e_M500/M500 = 0.224 but leaves e_c200/c200 at 0.074, FOUR TIMES too small,
    because rescaling Sigma barely moves its SHAPE.  The width of the
    fixed-effects-slope null is set by exactly that shape uncertainty, so without
    this term S3's null was ~4x too narrow and its z ~4x too large.  Fixed by
    adding a coherent radial TILT, Sigma -> Sigma (R/R_piv)^tilt -- the shape-
    systematic form of dilution, contamination and mass-sheet errors.
    """
    f_coh, f_ind, f_tilt = noise_pars
    nm, rr, gb, go, R5 = [], [], [], [], []
    for t in truths:
        Sig = sig_cache[t.name]
        err = math.sqrt(f_coh ** 2 + f_ind ** 2 + f_tilt ** 2) * Sig
        if noise:
            obs = (Sig * (1 + f_coh * rng.normal())
                   * (N.R_FIT / R_PIV) ** (f_tilt * rng.normal())
                   + f_ind * Sig * rng.normal(0, 1, len(Sig)))
        else:
            obs = Sig
        M200, c200 = fitters[t.name].fit(obs, err)
        r_pts = rby[t.name]
        gt = (G * I.nfw_mass(r_pts, M200, c200, t.z) * MSUN
              / np.asarray(r_pts) ** 2)
        R500 = I.r_delta(M200, c200, t.z, 500.0)
        nm += [t.name] * len(r_pts); rr += list(r_pts)
        gb += list(t.gbar(r_pts)); go += list(gt); R5 += [R500] * len(r_pts)
    return (np.array(nm), np.array(rr), np.array(gb), np.array(go), np.array(R5))


RX = {}          # real, INDEPENDENT Donahue+2014 X-ray R500 per cluster


def stat_pack(nm, r, gb, go, R5, stat, mask=None):
    e = S.excess_y(gb, go) if stat == "y" else S.excess_a0(gb, go)
    if mask is None:
        mask = np.ones(len(r), bool)
    e, r, R5, nm2 = e[mask], r[mask], R5[mask], nm[mask]
    x = np.log10(r / R5)
    names = sorted(set(nm2.tolist()))
    mu = np.array([e[nm2 == c].mean() for c in names])
    lR = np.array([math.log10(R5[nm2 == c][0]) for c in names])
    out = dict(S1=S.ols_slope(x, e), S2=S.spear(x, e),
               S3=S.fe_slope(x, e, nm2), S4=S.pear(lR, mu),
               S5=S.ols_slope(np.log10(r / KPC), e))
    # the pure tautology contrast: at FIXED physical radius, r/R500 varies only
    # through R500, so corr(excess, log R500) IS the tautology channel.  Measured
    # against the lensing R500 (shared with the numerator) and against the
    # Donahue+2014 X-ray R500 (independent of it).
    for L in (400.0, 600.0):
        m = np.abs(r / KPC - L) < 1e-6
        if m.sum() >= 8:
            lx = np.array([math.log10(RX[c]) for c in nm2[m]])
            ok = np.isfinite(lx)
            out[f"S6_{L:.0f}"] = S.pear(np.log10(R5[m]), e[m])
            out[f"S7_{L:.0f}"] = S.pear(lx[ok], e[m][ok])
        else:
            out[f"S6_{L:.0f}"] = float("nan")
            out[f"S7_{L:.0f}"] = float("nan")
    lxa = np.array([RX[c] for c in nm2])
    ok = np.isfinite(lxa)
    out["S8_slope_vs_xray_R500"] = S.ols_slope(np.log10(r[ok] / lxa[ok]), e[ok])
    return out


def summarise(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                p05=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)))


def main(nreal=1200):
    t0 = time.time()
    D = I.load_all(verbose=False)
    T = I.points_table(D)
    C = D["clusters"]
    print("building fitters (one Sigma grid per cluster redshift) ...")
    fitters = {n: Fitter(C[n]["z"]) for n in sorted(C)}
    RX.update({n: C[n]["R500_xray"] for n in sorted(C)})
    print(f"  {len(fitters)} grids in {time.time()-t0:.0f}s")

    # per-cluster excess levels: match the OBSERVED between-cluster mean, so the
    # null has the right between-cluster scatter and no radial dependence.
    obs_mask = T["r"] / KPC > 50
    y_obs = S.excess_y(T["gb"], T["go"])
    a_obs = S.excess_a0(T["gb"], T["go"])
    ylev = {n: float(y_obs[obs_mask & (T["name"] == n)].mean())
            for n in sorted(C)}

    # ------------------------------------------------ noise calibration
    print("\ncalibrating the Sigma noise against Umetsu's quoted e_M500/M500 ...")
    e_rel = np.array([C[n]["e_M500"] / C[n]["M500"] for n in sorted(C)])
    target = float(np.median(e_rel))
    truths, rby = build(D, ylev, 0.0)
    sig = {t.name: t.sigma(N.R_FIT) for t in truths}
    target_c = float(np.median([C[n]["e_c200"] / C[n]["c200"] for n in sorted(C)]))
    cal = []
    for f_coh in (0.10, 0.15, 0.20):
        for f_tilt in (0.0, 0.05, 0.10, 0.15):
            cc = {t.name: [] for t in truths}
            rng = np.random.default_rng(1)
            M5 = {t.name: [] for t in truths}
            for _ in range(50):
                for t in truths:
                    err = math.sqrt(f_coh ** 2 + 0.03 ** 2 + f_tilt ** 2) * sig[t.name]
                    obs = (sig[t.name] * (1 + f_coh * rng.normal())
                           * (N.R_FIT / R_PIV) ** (f_tilt * rng.normal())
                           + 0.03 * sig[t.name] * rng.normal(0, 1, len(err)))
                    M200, c200 = fitters[t.name].fit(obs, err)
                    cc[t.name].append(c200)
                    M5[t.name].append(float(I.nfw_mass(
                        I.r_delta(M200, c200, t.z, 500.0), M200, c200, t.z)))
            rm = float(np.median([np.std(v) / np.mean(v) for v in M5.values()]))
            rc = float(np.median([np.std(v) / np.mean(v) for v in cc.values()]))
            cal.append(dict(f_coh=f_coh, f_ind=0.03, f_tilt=f_tilt,
                            recovered_e_M500_over_M500=rm,
                            recovered_e_c200_over_c200=rc))
            print(f"  coh {f_coh:.2f} tilt {f_tilt:.2f} -> e_M500/M500 = {rm:.3f} "
                  f"(target {target:.3f}), e_c200/c200 = {rc:.3f} "
                  f"(target {target_c:.3f})")
    best = min(cal, key=lambda d:
               (d["recovered_e_M500_over_M500"] - target) ** 2 / target ** 2
               + (d["recovered_e_c200_over_c200"] - target_c) ** 2 / target_c ** 2)
    FE = (best["f_coh"], 0.03, best["f_tilt"])
    OUT["noise_calibration"] = dict(
        scan=cal, umetsu_median_e_M500_over_M500=target,
        umetsu_median_e_c200_over_c200=target_c,
        chosen_f_coherent=FE[0], chosen_f_independent=FE[1], chosen_f_tilt=FE[2],
        achieved_e_M500=best["recovered_e_M500_over_M500"],
        achieved_e_c200=best["recovered_e_c200_over_c200"])
    print(f"  -> using coherent {FE[0]:.2f}, independent {FE[1]:.2f}, "
          f"tilt {FE[2]:.2f}")

    # ------------------------------------------------ noise-free template bias
    print("\n=== template bias: a FLAT truth through the pipeline, NO noise ===")
    truths, rby = build(D, ylev, 0.0)
    sig = {t.name: t.sigma(N.R_FIT) for t in truths}
    out = realise(truths, rby, fitters, FE, RNG, sig, noise=False)
    tb = {}
    for stat in ("y", "a0"):
        for lab, mk in (("all_84", None), ("cluster_scale_64", out[1] / KPC > 50)):
            tb[f"{stat}/{lab}"] = stat_pack(*out, stat=stat, mask=mk)
    gtrue = np.concatenate([
        t.gbar(rby[t.name]) * I.nu_rar(t.gbar(rby[t.name]) / I.A0)
        * 10 ** t.excess_true(rby[t.name]) for t in truths])
    ratio = np.log10(out[3] / gtrue)
    lev = np.round(out[1] / KPC, 1)
    tb["template_misfit_dex_by_level"] = {
        ("BCG" if L < 40 else f"{L:.0f}kpc"): float(
            ratio[(lev < 40) if L < 40 else (np.abs(lev - L) < 1e-6)].mean())
        for L in (14.3, 100.0, 200.0, 400.0, 600.0)}
    OUT["template_bias_noise_free"] = tb
    print("  NFW-template misfit log10(g_published / g_true), by radius level:")
    for k, v in tb["template_misfit_dex_by_level"].items():
        print(f"    {k:>8s}  {v:+.4f} dex")
    for k, v in tb.items():
        if k == "template_misfit_dex_by_level":
            continue
        print(f"  {k:22s} S1 slope {v['S1']:+.4f}  S2 spearman {v['S2']:+.4f}  "
              f"S3 FE slope {v['S3']:+.4f}  S4 between {v['S4']:+.4f}  "
              f"S6_600 {v['S6_600']:+.4f}  S7_600 {v['S7_600']:+.4f}")

    # ------------------------------------------------ the null distribution
    print(f"\n=== forward null: {nreal} realisations, flat truth + noise ===")
    KEYS = ("S1", "S2", "S3", "S4", "S5", "S6_400", "S6_600",
            "S7_400", "S7_600", "S8_slope_vs_xray_R500")
    keep = {f"{s}/{l}": {k: [] for k in KEYS}
            for s in ("y", "a0") for l in ("all_84", "cluster_scale_64")}
    for it in range(nreal):
        o = realise(truths, rby, fitters, FE, RNG, sig, noise=True)
        for stat in ("y", "a0"):
            for lab, mk in (("all_84", None), ("cluster_scale_64", o[1] / KPC > 50)):
                p = stat_pack(*o, stat=stat, mask=mk)
                for k in keep[f"{stat}/{lab}"]:
                    keep[f"{stat}/{lab}"][k].append(p[k])
        if (it + 1) % 300 == 0:
            print(f"  {it+1}/{nreal}  ({time.time()-t0:.0f}s)")
    nullD = {k: {s: summarise(v) for s, v in d.items()} for k, d in keep.items()}
    OUT["forward_null"] = nullD

    # ------------------------------------------------ observed vs the null
    print("\n=== observed against the forward null ===")
    obs = {}
    for stat, ev in (("y", y_obs), ("a0", a_obs)):
        for lab, mk in (("all_84", np.ones(84, bool)),
                        ("cluster_scale_64", obs_mask)):
            R5o = np.array([C[n]["R500_lens"] for n in T["name"]])
            p = stat_pack(T["name"], T["r"], T["gb"], T["go"], R5o,
                          stat=stat, mask=mk)
            key = f"{stat}/{lab}"
            obs[key] = p
            for k in KEYS:
                nl = np.array(keep[key][k], float)
                if not np.all(np.isfinite(nl)) or nl.std(ddof=1) == 0:
                    continue
                z = (p[k] - nl.mean()) / nl.std(ddof=1)
                pct = float((nl < p[k]).mean() * 100)
                obs[key][k + "_z"] = float(z)
                obs[key][k + "_pct"] = pct
            for k in ("S1", "S3", "S4", "S6_600", "S7_600",
                      "S8_slope_vs_xray_R500"):
                if k + "_z" not in p:
                    continue
                print(f"  {key if k=='S1' else '':22s} {k:<22s} obs "
                      f"{p[k]:+.4f}  null {nullD[key][k]['mean']:+.4f} +- "
                      f"{nullD[key][k]['sd']:.4f}  z = {p[k+'_z']:+6.2f}  "
                      f"pct {p[k+'_pct']:5.1f}")
    OUT["observed_vs_null"] = obs

    # ------------------------------------------------ responsiveness
    print("\n=== responsiveness d(measured)/d(injected) ===")
    resp = {}
    inj = [0.0, -0.10, -0.20, -0.40, -0.60]
    meas = {("y", "S1"): [], ("y", "S3"): [],
            ("a0", "S1"): [], ("a0", "S3"): []}
    for s in inj:
        tr2, rb2 = build(D, ylev, s)
        sg2 = {t.name: t.sigma(N.R_FIT) for t in tr2}
        rng = np.random.default_rng(99)
        acc = {k: [] for k in meas}
        for _ in range(80):
            o = realise(tr2, rb2, fitters, FE, rng, sg2, noise=True)
            m = o[1] / KPC > 50
            for stat in ("y", "a0"):
                p = stat_pack(*o, stat=stat, mask=m)
                acc[(stat, "S1")].append(p["S1"])
                acc[(stat, "S3")].append(p["S3"])
        for k in meas:
            meas[k].append(float(np.mean(acc[k])))
        print(f"  injected {s:+.2f}  ->  y/S1 {meas[('y','S1')][-1]:+.4f}   "
              f"y/S3 {meas[('y','S3')][-1]:+.4f}   "
              f"a0/S1 {meas[('a0','S1')][-1]:+.4f}   "
              f"a0/S3 {meas[('a0','S3')][-1]:+.4f}")
    for (stat, sk), v in meas.items():
        c = np.polyfit(inj, v, 1)
        resp[f"{stat}/{sk}"] = dict(injected=inj, measured=v,
                                    responsiveness=float(c[0]),
                                    intercept=float(c[1]))
        print(f"  d({stat}/{sk})/d(injected) = {c[0]:.3f}")
    OUT["responsiveness"] = resp

    # ------------------------------------------------ FPR of the obvious tests
    print("\n=== FPR calibration: size the tests against the FLAT truth ===")
    fpr = {}
    # Test A: R500-label permutation, the direct analogue of Run AT's test.
    # Test B: the naive t-test on the pooled slope with n-2 df.
    nperm = 300
    rejA = rejB = 0
    nsim = 200
    for it in range(nsim):
        o = realise(truths, rby, fitters, FE, RNG, sig, noise=True)
        mk = o[1] / KPC > 50
        nm2, r2, gb2, go2, R52 = (a[mk] for a in o)
        e2 = S.excess_y(gb2, go2)
        names = sorted(set(nm2.tolist()))
        R5map = {c: R52[nm2 == c][0] for c in names}
        s_obs = S.ols_slope(np.log10(r2 / R52), e2)
        # A: permute the R500 labels across clusters
        cnt = 0
        for _ in range(nperm):
            perm = RNG.permutation(names)
            mp = dict(zip(names, perm))
            R5p = np.array([R5map[mp[c]] for c in nm2])
            if S.ols_slope(np.log10(r2 / R5p), e2) <= s_obs:
                cnt += 1
        if cnt / nperm <= 0.05:
            rejA += 1
        # B: naive OLS t-test
        x2 = np.log10(r2 / R52)
        n2 = len(x2)
        res = e2 - np.polyval(np.polyfit(x2, e2, 1), x2)
        se = math.sqrt(np.sum(res ** 2) / (n2 - 2) / np.sum((x2 - x2.mean()) ** 2))
        if abs(s_obs / se) > 2.0:
            rejB += 1
    fpr["R500_label_permutation"] = dict(nominal=0.05, measured=rejA / nsim,
                                         n_sim=nsim, n_perm=nperm)
    fpr["naive_OLS_t_test"] = dict(nominal=0.05, measured=rejB / nsim, n_sim=nsim)
    OUT["false_positive_rates"] = fpr
    print(f"  R500-label permutation test : FPR = {rejA/nsim:.3f} "
          f"(nominal 0.05, n={nsim})   [Run AT X-COP: 0.53-0.70]")
    print(f"  naive OLS t-test on S1      : FPR = {rejB/nsim:.3f} (nominal 0.05)")

    # ------------------------------------------------ r_break sensitivity
    print("\n=== sensitivity to the truth's outer break radius ===")
    sens = []
    for rb in (0.6, 1.0, 1.5):
        tr3, rb3 = build(D, ylev, 0.0, r_break=rb * MPC)
        sg3 = {t.name: t.sigma(N.R_FIT) for t in tr3}
        o = realise(tr3, rb3, fitters, FE, RNG, sg3, noise=False)
        p = stat_pack(*o, stat="y", mask=o[1] / KPC > 50)
        sens.append(dict(r_break_Mpc=rb, S1=p["S1"], S3=p["S3"], S4=p["S4"]))
        print(f"  r_break {rb:.1f} Mpc -> noise-free S1 = {p['S1']:+.4f}, "
              f"S4 = {p['S4']:+.4f}")
    OUT["r_break_sensitivity"] = sens

    OUT["runtime_seconds"] = time.time() - t0
    OUT["n_realisations"] = nreal
    json.dump(OUT, open("null_results.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote null_results.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
