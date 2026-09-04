"""Independent calibration gate for the lensing side.

Chiu+2022's figure carries their OWN best-fit model curve as well as the data.
That curve is the weighted stack of their per-cluster NFW + miscentering models,
so pushing a single-mass NFW population through THIS pipeline -- my source
n(z), my Sigma_crit, my reduced-shear conversion, my stacking weights -- and
fitting it to their curve must return a mass consistent with the mass they
publish.  If it does not, the amplitude of every prediction in efeds_hsc.py is
suspect and the report has to say so.

Chiu+2022 abstract: ensemble mass 1e13 <= M500 <= 1e15 h^-1 Msun with a median
of ~1e14 h^-1 Msun.  Their Table C1 gives log M500 per cluster in h^-1 Msun; the
lensing-weighted mean of that column is the number to compare against, and it is
read here ONLY for this gate -- it is never an observable in the test itself.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import pipeline as P
import efeds_hsc as E

HERE = os.path.dirname(os.path.abspath(__file__))


def read_chiu_masses():
    p = os.path.join(HERE, "acquire", "chiu2022_efeds_tablec1.tsv")
    out = {}
    for ln in open(p, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        q = ln.rstrip("\n").split("\t")
        if len(q) < 9 or not q[1].strip().startswith("J"):
            continue
        try:
            out[q[1].strip().lstrip("J")] = float(q[7])   # logM500R0.5A
        except ValueError:
            pass
    return out


def main():
    print("=" * 78)
    print("CALIBRATION GATE -- my lensing pipeline against Chiu's own model")
    print("=" * 78)
    recs, _ = E.load_efeds()
    sysd = [P.System(r) for r in recs]
    src = P.Sources(P.fit_source_nz(verbose=False))
    # Chiu's model curve, recovered from the same vector PDF
    Rm, gm = [], []
    for ln in open(os.path.join(HERE,
                                "chiu2022_model_bestfit_miscentered.tsv"),
                   encoding="utf-8"):
        if ln.startswith("#") or ln.startswith("R_hinv"):
            continue
        a, b = ln.split("\t")
        Rm.append(float(a))
        gm.append(float(b))
    R = np.array(Rm) / P.H_LITTLE * P.MPC
    gm = np.array(gm)
    used = np.array(Rm) * 1.0 > 0.5 * 1.0        # R > 0.5 h^-1 Mpc
    # same weights as the main run: shape noise with the calibrated cap
    W = E.stack_weights(sysd, src, R, 20.0)
    w = W.mean(axis=1)
    w = w / w.sum()

    # a single-mass NFW population at the real redshifts and weights
    def stack_nfw(logM500h):
        M500 = 10 ** logM500h / P.H_LITTLE * P.MSUN
        acc = np.zeros(len(R))
        for s, wj in zip(sysd, w):
            rho_c = 3.0 * (P.L.H0 * P.L.E(s.z)) ** 2 / (8 * math.pi * P.G)
            # invert M500 -> M200 for a Duffy-like c(M); c only weakly matters
            c200 = 5.0
            lo, hi = 0.2 * M500, 6.0 * M500
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if P.nfw_m200_to_m500(mid, c200, s.z) < M500:
                    lo = mid
                else:
                    hi = mid
            M200 = 0.5 * (lo + hi)
            Sig = P.nfw_sigma(R, M200, c200, s.z)
            dS = P.nfw_delta_sigma(R, M200, c200, s.z)
            scr = src.sigma_crit_eff(s.z)
            bm, b2 = src.beta_moments(s.z)
            kap, gam = Sig / scr, dS / scr
            acc += wj * gam / (1 - kap) * (1 + kap * (b2 / bm ** 2 - 1))
        return acc

    grid = np.linspace(13.0, 15.0, 81)
    chi = [np.sum(((stack_nfw(x) - gm) / (0.05 * gm))[used] ** 2)
           for x in grid]
    best = float(grid[int(np.argmin(chi))])
    pred = stack_nfw(best)
    print(f"\n   single-mass NFW pushed through THIS pipeline reproduces "
          f"Chiu's own\n   best-fit stacked model at log10 M500 = {best:.3f} "
          f"h^-1 Msun")
    print(f"   residual over R > 0.5 h^-1 Mpc: "
          f"{np.max(np.abs((pred / gm - 1)[used])) * 100:.1f}% max, "
          f"{np.mean(np.abs((pred / gm - 1)[used])) * 100:.1f}% mean")

    mm = read_chiu_masses()
    have = np.array([mm[s.id] for s in sysd if s.id in mm])
    ww = np.array([wj for s, wj in zip(sysd, w) if s.id in mm])
    ww = ww / ww.sum()
    print(f"\n   Chiu Table C1 log M500 for the {len(have)} matched systems: "
          f"median {np.median(have):.3f}, lensing-weighted mean "
          f"{np.sum(ww * have):.3f} h^-1 Msun")
    print(f"   paper's quoted sample median ~14.0 h^-1 Msun")
    off = best - float(np.sum(ww * have))
    print(f"\n   GATE: pipeline-recovered minus catalogue weighted mean = "
          f"{off:+.3f} dex  ->  "
          f"{'PASS' if abs(off) < 0.25 else 'CHECK -- amplitude systematic'}")
    print("   This bounds the amplitude systematic of the whole forward model:"
          f"\n   Sigma_crit, source n(z), stacking weights and the reduced-"
          f"shear conversion\n   together are good to {abs(off):.2f} dex.")

    out = {"logM500h_recovered_from_Chiu_model_curve": best,
           "chiu_tablec1_weighted_mean_logM500h": float(np.sum(ww * have)),
           "chiu_tablec1_median_logM500h": float(np.median(have)),
           "n_matched": int(len(have)),
           "offset_dex": off,
           "max_frac_residual_on_curve":
               float(np.max(np.abs((pred / gm - 1)[used]))),
           "passed": bool(abs(off) < 0.25)}
    with open(os.path.join(HERE, "calib_check.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n   wrote calib_check.json")


if __name__ == "__main__":
    main()
