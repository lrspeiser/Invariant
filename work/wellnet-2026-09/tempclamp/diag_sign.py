"""Diagnostic: is the clamp bias positive or negative, and on WHICH statistic?

KEPT AS THE RECORD OF HOW THE SIGN WAS SETTLED. Its first run exposed the bug
in this lane's own synthetic truth -- negative temperatures from a mid-bin
pressure anchor -- which had flipped the reported bias sign. Both are fixed;
run it to see the two statistics side by side on the same truth.

Run AT reports a flat truth returning S3 = -0.1359 (pooled slope beyond
0.25 R500 against r/R500_inferred).  The tempclamp within-cluster experiment
returns +0.055.  Both cannot describe the same thing.  This script settles it
by computing BOTH statistics on the SAME synthetic truth, and by looking
directly at the clamped-versus-true acceleration ratio.
"""
from __future__ import annotations
import glob
import os
import numpy as np
from astropy.io import fits

import tclamp as T
from tclamp import A0, KPC, MU, MP, nu_rar, load_xcop
import run_audit as R

KEV = 1.602176634e-16


def attach(CL):
    for c in CL:
        d = os.path.join(T.XR, c["name"])
        fd = glob.glob(os.path.join(d, "*density*.fits"))[0]
        ft = glob.glob(os.path.join(d, "*temperature*.fits"))[0]
        with fits.open(fd) as h:
            da = h[1].data
            R500m = float(h[1].header["R500"]) * KPC
            rr = 0.5 * (da["R_IN"].astype(np.float64)
                        + da["R_OUT"].astype(np.float64)) * KPC
            nee = da["NE"].astype(np.float64) * 1e6
        with fits.open(ft) as h:
            rw = np.asarray(h[1].data["RW_X"], float) * R500m
        m = (rr > 120 * KPC) & (rr < 1650 * KPC)
        sel = np.argsort(rr[m])
        c["ne"] = nee[m][sel]
        c["r_m"] = rr[m][sel]
        c["rw"] = rw
        c["R500_m"] = R500m
        assert len(c["ne"]) == c["n"]
    return CL


def pooled_S3(pairs, tmin=0.25):
    """AT's statistic: pooled slope of y on log10(r/R500), points beyond 0.25."""
    t = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    m = t > tmin
    return float(np.polyfit(np.log10(t[m]), y[m], 1)[0])


def within_slope(pairs):
    x = np.concatenate([np.log10(p[0]) for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    g = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(pairs)])
    idx = np.unique(g)
    A = np.zeros((len(x), len(idx) + 1)); A[:, 0] = x
    for j, gi in enumerate(idx):
        A[g == gi, j + 1] = 1.0
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def main():
    xc, CL, tot = load_xcop("clamp")
    CL = attach(CL)

    print("=" * 78)
    print("1. Is the synthetic temperature profile realistic?")
    print("=" * 78)
    print(f"   {'cluster':<9}{'synth dlnT/dlnr':>17}{'real dlnT/dlnr':>16}"
          f"{'synth kT mid':>14}{'real kT mid':>13}")
    for c in CL:
        sy = R.synth_cluster(c, 0.0)
        r = sy["r"]
        s_syn = float(np.polyfit(np.log(r[r > 0.4 * c["R500_m"]]),
                                 np.log(sy["kT_true"][r > 0.4 * c["R500_m"]]), 1)[0])
        rw, kw = c["rw"], np.interp(c["rw"], r, c["kT"])
        ok = rw > 0.4 * c["R500_m"]
        s_real = float(np.polyfit(np.log(rw[ok]), np.log(kw[ok]), 1)[0])
        j = len(r) // 2
        print(f"   {c['name']:<9}{s_syn:>17.3f}{s_real:>16.3f}"
              f"{sy['kT_true'][j]:>14.2f}{c['kT'][j]:>13.2f}")

    print("\n" + "=" * 78)
    print("2. What the clamp does to g_obs at the outer end (flat truth)")
    print("=" * 78)
    print(f"   {'cluster':<9}{'n clamp':>8}{'kT_clamp/kT_true':>19}"
          f"{'sum_clamp/sum_true':>20}{'g_clamp/g_true':>17}")
    for c in CL:
        sy = R.synth_cluster(c, 0.0)
        r, ne = sy["r"], sy["ne"]
        rw = c["rw"]
        kT_c = np.interp(rw, r, sy["kT_true"])
        kT_hat = np.interp(r, rw, kT_c)
        ext = (r < rw.min()) | (r > rw.max())
        lr = np.log(r)
        sum_t = np.gradient(np.log(ne), lr) + np.gradient(np.log(sy["kT_true"]), lr)
        sum_c = np.gradient(np.log(ne), lr) + np.gradient(np.log(kT_hat), lr)
        g_c = -(kT_hat * KEV / (MU * MP)) * sum_c / r
        if ext.any():
            e = ext & (r > rw.max())
            print(f"   {c['name']:<9}{int(e.sum()):>8}"
                  f"{np.median(kT_hat[e]/sy['kT_true'][e]):>19.3f}"
                  f"{np.median(sum_c[e]/sum_t[e]):>20.3f}"
                  f"{np.median(g_c[e]/sy['go_true'][e]):>17.3f}")

    print("\n" + "=" * 78)
    print("3. The SAME synthetic truth, BOTH statistics")
    print("=" * 78)
    print(f"   {'truth':>7}  {'stat':<14}" + "".join(
        f"{m:>14}" for m in ("clamp", "drop", "loglinear", "full_cov", "perfect")))
    out = []
    for s_true in (0.0, -0.25, -0.48):
        row = {"s_true": s_true}
        for stat in ("pooled_S3", "within"):
            vals = {}
            for mode in ("clamp", "drop", "loglinear", "full_coverage", "perfect"):
                pr = []
                for c in CL:
                    sy = R.synth_cluster(c, s_true)
                    if mode == "perfect":
                        x, y, _ = R.reconstruct(sy, c, "clamp", rw_grid=sy["r"])
                    elif mode == "full_coverage":
                        rw = np.exp(np.linspace(np.log(sy["r"].min()),
                                                np.log(sy["r"].max()),
                                                len(c["rw"])))
                        x, y, _ = R.reconstruct(sy, c, "clamp", rw_grid=rw)
                    else:
                        x, y, _ = R.reconstruct(sy, c, mode)
                    if stat == "pooled_S3":
                        pr.append(((10 ** x) / c["R500_m"], y))
                    else:
                        pr.append((10 ** x, y))
                vals[mode] = (pooled_S3(pr) if stat == "pooled_S3"
                              else within_slope(pr))
            row[stat] = vals
            print(f"   {s_true:>+7.2f}  {stat:<14}" + "".join(
                f"{vals[m]:>+14.4f}" for m in ("clamp", "drop", "loglinear",
                                               "full_coverage", "perfect")))
        out.append(row)

    print("\n   Observed values on the REAL data, same two statistics:")
    for stat in ("pooled_S3", "within"):
        line = []
        for mode in ("clamp", "drop", "loglinear"):
            _, CLm, _ = load_xcop(mode)
            pr = [((c["r"] * KPC) / (c["R500"] * KPC), np.log10(c["exc"]))
                  if stat == "pooled_S3" else (c["r"], np.log10(c["exc"]))
                  for c in CLm]
            v = pooled_S3(pr) if stat == "pooled_S3" else within_slope(pr)
            line.append(f"{mode}={v:+.4f}")
        print(f"      {stat:<12} " + "   ".join(line))
    return out


if __name__ == "__main__":
    main()
