"""
The verdict's dependence on the one thing nothing measures.

The forward null's CENTRE is set by how far out the flat-excess truth is imposed
before it is handed over to the published NFW.  Beyond 600 kpc -- the outermost
CLASH datum -- nothing in this data set constrains that choice, so it has to be
scanned rather than picked, and the observed z quoted under each.

  r_break = 0.6 Mpc  the data edge.  Grants the published NFW everywhere outside
                     the measurements and asks only whether a flat excess INSIDE
                     them comes back out sloped.  The conservative choice.
  r_break = 1.5 Mpc  asserts the flat-excess law over a region where nothing
                     measures it, i.e. assumes part of the hypothesis.
"""
from __future__ import annotations
import json
import time

import numpy as np

import ingest as I
import stats as S
import nullsim as N
import run_null as R

KPC, MPC = I.KPC, I.MPC
OUT = {}


def main(nreal=400):
    t0 = time.time()
    D = I.load_all(verbose=False)
    T = I.points_table(D)
    C = D["clusters"]
    fitters = {n: R.Fitter(C[n]["z"]) for n in sorted(C)}
    R.RX.update({n: C[n]["R500_xray"] for n in sorted(C)})
    cal = json.load(open("null_results.json", encoding="utf-8"))["noise_calibration"]
    FE = (cal["chosen_f_coherent"], cal["chosen_f_independent"], cal["chosen_f_tilt"])
    print(f"noise (coherent, independent, tilt) = {FE}")

    mk = T["r"] / KPC > 50
    y_obs = S.excess_y(T["gb"], T["go"])
    ylev = {n: float(y_obs[mk & (T["name"] == n)].mean()) for n in sorted(C)}
    R5o = np.array([C[n]["R500_lens"] for n in T["name"]])
    obs = {s: R.stat_pack(T["name"], T["r"], T["gb"], T["go"], R5o,
                          stat=s, mask=mk) for s in ("y", "a0")}

    rows = []
    for rb in (0.6, 0.8, 1.0, 1.5, 2.5):
        truths, rby = R.build(D, ylev, 0.0, r_break=rb * MPC)
        sig = {t.name: t.sigma(N.R_FIT) for t in truths}
        rng = np.random.default_rng(4242)
        keep = {s: {k: [] for k in ("S1", "S2", "S3", "S4")} for s in ("y", "a0")}
        for _ in range(nreal):
            o = R.realise(truths, rby, fitters, FE, rng, sig, noise=True)
            m = o[1] / KPC > 50
            for s in ("y", "a0"):
                p = R.stat_pack(*o, stat=s, mask=m)
                for k in keep[s]:
                    keep[s][k].append(p[k])
        row = dict(r_break_Mpc=rb)
        for s in ("y", "a0"):
            for k in ("S1", "S3"):
                v = np.array(keep[s][k], float)
                row[f"{s}/{k}/null_mean"] = float(v.mean())
                row[f"{s}/{k}/null_sd"] = float(v.std(ddof=1))
                row[f"{s}/{k}/obs"] = float(obs[s][k])
                row[f"{s}/{k}/z"] = float((obs[s][k] - v.mean()) / v.std(ddof=1))
                row[f"{s}/{k}/pct"] = float((v < obs[s][k]).mean() * 100)
        rows.append(row)
        print(f"  r_break {rb:.1f} Mpc:  "
              f"y  S1 obs {row['y/S1/obs']:+.4f} null {row['y/S1/null_mean']:+.4f}"
              f" +- {row['y/S1/null_sd']:.4f}  z = {row['y/S1/z']:+5.2f}   |   "
              f"S3 z = {row['y/S3/z']:+5.2f}   |   "
              f"a0 S1 z = {row['a0/S1/z']:+5.2f}  S3 z = {row['a0/S3/z']:+5.2f}")
    OUT["r_break_scan"] = rows
    OUT["n_realisations"] = nreal
    OUT["summary"] = dict(
        y_S1_z_min=float(min(r_["y/S1/z"] for r_ in rows)),
        y_S1_z_max=float(max(r_["y/S1/z"] for r_ in rows)),
        a0_S1_z_min=float(min(r_["a0/S1/z"] for r_ in rows)),
        a0_S1_z_max=float(max(r_["a0/S1/z"] for r_ in rows)),
        y_S3_z_min=float(min(r_["y/S3/z"] for r_ in rows)),
        y_S3_z_max=float(max(r_["y/S3/z"] for r_ in rows)),
        note="The null's CENTRE, not its width, is what moves with r_break.  The "
             "conservative choice -- the data edge, granting the published NFW "
             "everywhere nothing is measured -- puts the observation at the null "
             "median.")
    print(f"\n  pooled slope S1: z ranges {min(r_['y/S1/z'] for r_ in rows):+.2f} "
          f"to {max(r_['y/S1/z'] for r_ in rows):+.2f} (RAR residual)")
    print(f"  within slope S3: z ranges {min(r_['y/S3/z'] for r_ in rows):+.2f} "
          f"to {max(r_['y/S3/z'] for r_ in rows):+.2f}")

    json.dump(OUT, open("sensitivity_results.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote sensitivity_results.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
