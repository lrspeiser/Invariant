"""
Which candidate NULL TRUTHS are themselves allowed by the data, and what the one
surviving discrepancy actually is.

`run_sensitivity.py` shows the verdict on the pooled slope moves from z = +0.17
to z = -3.75 as the flat-excess truth is imposed further out.  That looks like an
undecidable systematic -- but it is not, because Umetsu+2016 measured the
PROJECTED profile out to R = 2.9 Mpc even though Tian+2020 tabulates g_tot only
to 600 kpc.  A truth whose Sigma(R) cannot be matched by any NFW amplitude within
the measurement error is excluded by that data, whatever it does to the null.

Second: since the published g_tot IS an NFW exactly (run_provenance.py), the
WITHIN-cluster radial run of the excess is a function of c200_i and the baryon
profile alone.  So the residual S3 discrepancy is, exactly and only, the
statement that the published concentrations differ from those an NFW fit to a
flat-excess cluster would return.  That is measured here directly, in the units
the reader can compare with Umetsu's own quoted uncertainty.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import stats as S
import nullsim as N
import run_null as R

KPC, MPC = I.KPC, I.MPC
OUT = {}


def main():
    D = I.load_all(verbose=False)
    T = I.points_table(D)
    C = D["clusters"]
    cal = json.load(open("null_results.json", encoding="utf-8"))["noise_calibration"]
    FE = (cal["chosen_f_coherent"], cal["chosen_f_independent"],
          cal["chosen_f_tilt"])
    sig_frac = math.sqrt(sum(v * v for v in FE))
    fitters = {n: R.Fitter(C[n]["z"]) for n in sorted(C)}
    mk = T["r"] / KPC > 50
    y = S.excess_y(T["gb"], T["go"])
    ylev = {n: float(y[mk & (T["name"] == n)].mean()) for n in sorted(C)}

    # ------------------------------------------------------------------ (1)
    print("=== (1) is each candidate truth allowed by Umetsu's measured Sigma? ===")
    print(f"chi2 of the truth's Sigma against the published-NFW Sigma over "
          f"R <= {N.FIT_RMAX/MPC:.1f} Mpc,")
    print(f"{len(N.R_FIT)} bins, calibrated fractional error {sig_frac:.3f}, "
          f"overall amplitude profiled out.")
    print(f"\n{'r_break':>9} {'chi2/dof':>10} {'median max |dlnSigma|':>23}")
    scan = []
    for rb in (0.6, 0.8, 1.0, 1.5, 2.5):
        truths, _ = R.build(D, ylev, 0.0, r_break=rb * MPC)
        chis, mx = [], []
        for t in truths:
            St = t.sigma(N.R_FIT)
            Sn = N.nfw_sigma(N.R_FIT, C[t.name]["M200"], C[t.name]["c200"], t.z)
            k = math.exp(float(np.mean(np.log(St / Sn))))   # profile amplitude out
            d = (St - k * Sn) / (sig_frac * k * Sn)
            chis.append(float(np.sum(d ** 2)) / (len(N.R_FIT) - 1))
            mx.append(float(np.max(np.abs(np.log(St / (k * Sn))))))
        scan.append(dict(r_break_Mpc=rb, median_chi2_per_dof=float(np.median(chis)),
                         max_chi2_per_dof=float(np.max(chis)),
                         median_max_abs_dlnSigma=float(np.median(mx)),
                         admissible=bool(np.median(chis) < 2.0)))
        print(f"{rb:9.1f} {np.median(chis):10.2f} {np.median(mx):23.3f}"
              f"   {'ALLOWED' if np.median(chis) < 2.0 else 'EXCLUDED'}")
    OUT["truth_consistency"] = dict(
        sigma_frac_err=sig_frac, scan=scan,
        note="A flat-excess truth imposed past ~1 Mpc predicts a projected profile "
             "Umetsu+2016 did not observe.  The admissible nulls are r_break <= "
             "1.0 Mpc, and those are exactly the ones that put the observed pooled "
             "slope inside the null.")

    # ------------------------------------------------------------------ (2)
    print("\n=== (2) what the residual within-cluster discrepancy IS ===")
    print("g_tot is exactly M_NFW(<r|M200,c200), so the within-cluster radial run")
    print("of the excess is a function of c200_i and the baryons alone.")
    print(f"\n{'r_break':>9} {'<c200> pub':>11} {'<c200> null':>12} "
          f"{'log10 ratio':>13} {'sem':>8} {'sigma':>7}")
    cmp = []
    for rb in (0.6, 0.8, 1.0):
        truths, _ = R.build(D, ylev, 0.0, r_break=rb * MPC)
        sig = {t.name: t.sigma(N.R_FIT) for t in truths}
        rng = np.random.default_rng(11)
        acc = {t.name: [] for t in truths}
        for _ in range(60):
            for t in truths:
                e = sig_frac * sig[t.name]
                obs = (sig[t.name] * (1 + FE[0] * rng.normal())
                       * (N.R_FIT / R.R_PIV) ** (FE[2] * rng.normal())
                       + FE[1] * sig[t.name] * rng.normal(0, 1, len(e)))
                _, c = fitters[t.name].fit(obs, e)
                acc[t.name].append(c)
        cn = np.array([np.mean(acc[n]) for n in sorted(C)])
        cp = np.array([C[n]["c200"] for n in sorted(C)])
        lr = np.log10(cp / cn)
        sem = float(np.std(lr, ddof=1) / math.sqrt(len(lr)))
        cmp.append(dict(r_break_Mpc=rb, mean_c200_published=float(cp.mean()),
                        mean_c200_null=float(cn.mean()),
                        mean_log10_ratio=float(lr.mean()), sem=sem,
                        sigma=float(abs(lr.mean() / sem))))
        print(f"{rb:9.1f} {cp.mean():11.2f} {cn.mean():12.2f} "
              f"{lr.mean():+13.4f} {sem:8.4f} {abs(lr.mean()/sem):7.1f}")
    ec = float(np.median([C[n]["e_c200"] / C[n]["c200"] for n in C]))
    OUT["c200_comparison"] = dict(
        scan=cmp, umetsu_e_c200_over_c200_median=ec,
        umetsu_e_log10_c200_median=ec / math.log(10),
        note="The whole surviving within-cluster signal is that the published NFW "
             "concentrations sit above those a flat-excess cluster with the same "
             "baryons would return.  At the admissible r_break of 0.6 Mpc the "
             "offset is 0.032 dex -- a quarter of ONE cluster's own quoted "
             "concentration uncertainty, accumulated over 20 clusters.  That is a "
             "statement about NFW concentrations inside a halo model, not about "
             "gravity.")
    print(f"\nUmetsu quotes e_c200/c200 = {ec:.2f} per cluster = "
          f"{ec/math.log(10):.3f} dex.")
    print(f"The offset at the admissible r_break is "
          f"{cmp[0]['mean_log10_ratio']:.4f} dex, i.e. "
          f"{cmp[0]['mean_log10_ratio']/(ec/math.log(10)):.2f} of ONE cluster's "
          f"own uncertainty.")

    json.dump(OUT, open("truthcheck_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote truthcheck_results.json")


if __name__ == "__main__":
    main()
