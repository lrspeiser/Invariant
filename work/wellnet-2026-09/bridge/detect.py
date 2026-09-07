"""detect.py -- Job 4: detectability under BF's declared detector, sized on an
untouched null half; responsiveness; the 3-sigma amplitude, source density
and separation; and the member-safe eps window.

The estimator is LINEAR in the shear map, so for a signal s and a noise draw n
    stats(s + n) = stats(s) + stats(n)          (verified below to round-off)
and the noise distribution of every statistic is independent of the signal.
The null is  endpoints + BF noise  (shape noise per pixel, additive bias c,
the spatially coherent PSF residual); the multiplicative bias multiplies the
signal and is carried as a nuisance in certify.py.  N_pairs independent pairs
of identical geometry stack coherently, so sd -> sd / sqrt(N_pairs).

    python detect.py            # N_NULL=600 by default (300 + 300 untouched)
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict

import numpy as np

import guard
import scene as S
import lensing as L
import estimator as E
import build_cache as BC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KPC, MPC, MSUN = S.KPC, S.MPC, S.MSUN
EPS_REF = 0.03
N_NULL = int(os.environ.get("N_NULL", "600"))
N_SCAN = int(os.environ.get("N_SCAN", "160"))
N_PAIRS_LADDER = (1, 10, 30, 100, 300, 1000, 3000)
TOL_LADDER = ("tol_0.02", "tol_0.05", "tol_0.10", "tol_0.20")


def null_draws(g1e, g2e, sky, sv, geom, scr, template, n, seed, systematics=True):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        nz = L.noise_maps(sky, sv, rng, systematics=systematics)
        res = E.bridge_statistics(g1e + nz["e1"], g2e + nz["e2"], sky, geom, scr,
                                  template=template)
        rows.append(E.stat_vector(res))
    return np.array(rows)


def summarise(cal: np.ndarray, test: np.ndarray, nominal: float = 0.05) -> Dict:
    """Thresholds from the CALIBRATION half; realised rates on the UNTOUCHED
    half.  Two-sided at `nominal`, and one-sided (Q < 0, the P sign)."""
    out = {}
    for i, k in enumerate(E.STAT_KEYS):
        c, t = cal[:, i], test[:, i]
        lo, hi = np.quantile(c, nominal / 2), np.quantile(c, 1 - nominal / 2)
        one = np.quantile(c, nominal)
        out[k] = dict(cal_mean=float(c.mean()), cal_sd=float(c.std(ddof=1)),
                      test_mean=float(t.mean()), test_sd=float(t.std(ddof=1)),
                      realised_fpr_two_sided=float(np.mean((t < lo) | (t > hi))),
                      realised_fpr_one_sided_low=float(np.mean(t < one)),
                      nominal=nominal, n_cal=int(len(c)), n_test=int(len(t)))
    return out


def main():
    guard.arm()
    t0 = time.perf_counter()
    pf = BC.load("fid")
    sc = pf.scene
    D = sc.D
    sky = L.default_sky(D)
    sv = L.Survey()
    scr = sv.sigma_crit_eff
    geom = E.Geometry(xA=-0.5 * D, xB=0.5 * D)
    template = np.load(os.path.join(RES, "template_kres.npy"))
    iso: Dict = {}
    members = json.load(open(os.path.join(RES, "members.json"), encoding="utf-8"))

    # ------------------------------------------------ signal responses (noise-free)
    signal = {}
    for rtag, rs in S.RHO_STAR_LADDER.items():
        m = E.law_kappa_maps("P", pf, sky, scr, eps=EPS_REF, rho_star=rs, iso_cache=iso)
        r = E.bridge_statistics(m["g1"], m["g2"], sky, geom, scr, template=template)
        signal[rtag] = dict(maps=m, stats=E.stat_vector(r))
    ends = signal["1e-24"]["maps"]
    g1e, g2e = ends["g1_ends"], ends["g2_ends"]

    # ------------------------------------------------ linearity check
    rng = np.random.default_rng(1)
    nz = L.noise_maps(sky, sv, rng)
    m = signal["1e-24"]["maps"]
    a = E.stat_vector(E.bridge_statistics(m["g1"] + nz["e1"], m["g2"] + nz["e2"],
                                          sky, geom, scr, template=template))
    b = E.stat_vector(E.bridge_statistics(g1e + nz["e1"], g2e + nz["e2"],
                                          sky, geom, scr, template=template))
    ends_only = E.stat_vector(E.bridge_statistics(g1e, g2e, sky, geom, scr, template=template))
    sig_only = signal["1e-24"]["stats"] - ends_only
    lin = np.abs((a - b) - sig_only) / np.maximum(np.abs(sig_only), 1e-300)
    print("linearity |stats(s+n) - stats(n) - stats(s)| / |stats(s)|:", lin.max())

    # ------------------------------------------------ the null, two halves
    t1 = time.perf_counter()
    cal = null_draws(g1e, g2e, sky, sv, geom, scr, template, N_NULL // 2, seed=101)
    test = null_draws(g1e, g2e, sky, sv, geom, scr, template, N_NULL - N_NULL // 2,
                      seed=202)
    null = summarise(cal, test)
    print(f"null: {N_NULL} draws in {time.perf_counter() - t1:.0f}s")
    for k in E.STAT_KEYS:
        print(f"  {k:<10} cal sd {null[k]['cal_sd']:.3e}  untouched sd {null[k]['test_sd']:.3e} "
              f"mean {null[k]['test_mean']:+.2e}  FPR two-sided {null[k]['realised_fpr_two_sided']:.3f} "
              f"one-sided {null[k]['realised_fpr_one_sided_low']:.3f}")
    # shape noise only (no systematics): how much of the sd is the PSF/c terms?
    cal_ns = null_draws(g1e, g2e, sky, sv, geom, scr, template, N_SCAN, seed=303,
                        systematics=False)
    sd_ns = {k: float(cal_ns[:, i].std(ddof=1)) for i, k in enumerate(E.STAT_KEYS)}

    # ------------------------------------------------ responsiveness, 3 sigma
    sd = {k: null[k]["test_sd"] for k in E.STAT_KEYS}
    resp = {}
    for rtag, s in signal.items():
        per_eps = s["stats"] / EPS_REF
        resp[rtag] = {}
        for i, k in enumerate(E.STAT_KEYS):
            R = float(per_eps[i])
            z1 = abs(R) * EPS_REF / sd[k]            # z per pair at eps_ref
            resp[rtag][k] = dict(
                d_stat_d_eps=R, se_per_pair=sd[k] / EPS_REF,
                z_per_pair_at_eps_ref=z1,
                eps_3sigma_by_N_pairs={str(n): 3.0 * sd[k] / (abs(R) * math.sqrt(n))
                                       if R != 0 else float("inf") for n in N_PAIRS_LADDER},
                N_pairs_for_3sigma_at_eps={str(e): (3.0 * sd[k] / (abs(R) * e)) ** 2
                                           for e in (0.3, 0.03, 0.01, 0.003, 0.001)})
    # ------------------------------------------------ member-safe window
    window = {}
    for rtag in signal:
        mrow = members["rows"][rtag]
        window[rtag] = {}
        for tol in TOL_LADDER:
            es = mrow["eps_safe"][tol]
            w = {}
            for k in ("A_mf", "Q", "DQ"):
                e3 = resp[rtag][k]["eps_3sigma_by_N_pairs"]
                w[k] = {n: dict(eps_3sigma=e3[n], eps_safe=es,
                                window_open=bool(e3[n] < es),
                                window=[e3[n], es] if e3[n] < es else None)
                        for n in e3}
                w[k]["N_pairs_to_open"] = (3.0 * sd[k] / (abs(resp[rtag][k]["d_stat_d_eps"]) * es)) ** 2
            window[rtag][tol] = w
    for rtag in signal:
        print(f"window rho_*={rtag}: eps_safe(5%) = {members['rows'][rtag]['eps_safe']['tol_0.05']:.2e}; "
              f"A_mf: eps_3sigma(100 pairs) = {resp[rtag]['A_mf']['eps_3sigma_by_N_pairs']['100']:.2e}, "
              f"N to open at 5% = {window[rtag]['tol_0.05']['A_mf']['N_pairs_to_open']:.0f}")

    # ------------------------------------------------ source density scan
    dens = {}
    for n_arc in (10.0, 20.0, 40.0, 80.0):
        sv2 = L.Survey(n_arcmin2=n_arc)
        draws = null_draws(g1e, g2e, sky, sv2, geom, scr, template, N_SCAN, seed=404)
        dens[str(n_arc)] = {k: float(draws[:, i].std(ddof=1)) for i, k in enumerate(E.STAT_KEYS)}
        dens[str(n_arc)]["eps_3sigma_A_mf_100pairs"] = (
            3.0 * dens[str(n_arc)]["A_mf"] / (abs(resp["1e-24"]["A_mf"]["d_stat_d_eps"]) * 10.0))
        print(f"  n = {n_arc:4.0f}/arcmin2: sd(A_mf) {dens[str(n_arc)]['A_mf']:.3e}, "
              f"eps_3sigma(A_mf, 100 pairs) {dens[str(n_arc)]['eps_3sigma_A_mf_100pairs']:.2e}")

    # ------------------------------------------------ separation scan
    sep = {}
    for tag in ("D2", "D3", "fid", "D6"):
        pfd = BC.load(tag)
        Dd = pfd.scene.D
        skyd = L.default_sky(Dd, half1=1.5, half2=min(1.4, 5.6 * MPC / Dd))
        geod = E.Geometry(xA=-0.5 * Dd, xB=0.5 * Dd)
        isod: Dict = {}
        md = E.law_kappa_maps("P", pfd, skyd, scr, eps=EPS_REF, rho_star=S.RHO_STAR_FID,
                              iso_cache=isod)
        rd = E.bridge_statistics(md["g1"], md["g2"], skyd, geod, scr, return_maps=True)
        T = rd["kappa_res"]
        rd = E.bridge_statistics(md["g1"], md["g2"], skyd, geod, scr, template=T)
        draws = null_draws(md["g1_ends"], md["g2_ends"], skyd, sv, geod, scr, T, N_SCAN,
                           seed=505)
        sdd = {k: float(draws[:, i].std(ddof=1)) for i, k in enumerate(E.STAT_KEYS)}
        j0 = int(np.argmin(np.abs(skyd.x1)))
        k0 = int(np.argmin(np.abs(skyd.x2)))
        sep[tag] = dict(D_Mpc=Dd / MPC, Sigma_axis=float(md["Sigma_spec"][j0, k0]),
                        stats_at_eps_ref={k: float(rd[k]) for k in E.STAT_KEYS},
                        null_sd=sdd,
                        eps_3sigma_100pairs={k: 3.0 * sdd[k] / (abs(rd[k] / EPS_REF) * 10.0)
                                             if rd[k] != 0 else float("inf")
                                             for k in ("Q", "DQ", "A_mf")},
                        note="A_mf uses this separation's own template")
        print(f"  D = {Dd / MPC:.0f} Mpc: Sigma_axis {sep[tag]['Sigma_axis']:+.3e}, "
              f"Q {rd['Q']:+.2e} +- {sdd['Q']:.2e} per pair; eps_3sigma(A_mf,100) "
              f"{sep[tag]['eps_3sigma_100pairs']['A_mf']:.2e}")

    # ------------------------------------------------ competitor separation at detector level
    comp_sep = {}
    for law, kw in (("qumond", {}), ("tensor", dict(fE=0.3)), ("cdm", dict(rho_f_dm=1e-26)),
                    ("newton", {})):
        mm = E.law_kappa_maps(law, pf, sky, scr, iso_cache=iso, **kw)
        rr = E.stat_vector(E.bridge_statistics(mm["g1"], mm["g2"], sky, geom, scr,
                                               template=template))
        comp_sep[law] = {}
        for rtag in signal:
            for eps in (0.3, 0.03, 0.01, 0.003):
                sP = signal[rtag]["stats"] * eps / EPS_REF
                diff = {k: float(sP[i] - rr[i]) for i, k in enumerate(E.STAT_KEYS)}
                nreq = {k: (3.0 * sd[k] / abs(diff[k])) ** 2 if diff[k] != 0 else float("inf")
                        for k in E.STAT_KEYS}
                comp_sep[law][f"{rtag}|eps={eps}"] = dict(
                    difference=diff, N_pairs_for_3sigma_separation=nreq)
    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        survey=dict(z_l=sv.z_l, sigma_e=sv.sigma_e, n_arcmin2=sv.n_arcmin2,
                    kpc_per_arcmin=sv.kpc_per_arcmin, sigma_crit_eff=scr,
                    pixel_kpc=sky.pix / KPC, pixel_shear_sd=sv.pixel_noise(sky.pix),
                    field_Mpc=[2 * sky.x1[-1] / MPC, 2 * sky.x2[-1] / MPC],
                    sources_per_field=sv.sources_per_pixel(sky.pix) * sky.x1.size * sky.x2.size,
                    systematics="BF: additive c ~ N(0, 5e-4) per component plus a "
                                "coherent PSF residual of amplitude 1e-3 at a random "
                                "wavevector ~ 2/r_max; the multiplicative bias "
                                "N(0, 0.02) multiplies the signal and is a nuisance in "
                                "certify.py"),
        eps_ref=EPS_REF, n_null=N_NULL, n_scan=N_SCAN,
        linearity_max_rel=float(lin.max()),
        null=null, null_sd_shape_noise_only=sd_ns,
        signal_stats_at_eps_ref={k: {s: float(v["stats"][i]) for i, s in enumerate(E.STAT_KEYS)}
                                 for k, v in signal.items()},
        responsiveness=resp, member_safe_window=window,
        members_eps_safe={k: v["eps_safe"] for k, v in members["rows"].items()},
        source_density_scan=dens, separation_scan=sep,
        competitor_separation=comp_sep,
        provenance=guard.summary(), wall_seconds=time.perf_counter() - t0)
    with open(os.path.join(RES, "detect.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote detect.json in {out['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
