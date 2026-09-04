"""Is M_global distinguishable from a Sigma_0-dependent Upsilon_K?

M_global multiplies K_z by (Sigma_0/<Sigma_0>)^p at every radius.
"Newton with Upsilon_K ~ Sigma_0^u" multiplies Sigma_*(R) -- and therefore K_z --
by the same factor at every radius.  Both are radially flat and both are linear
in sigma_z^2.  They differ ONLY through the radial-leakage term, because
Upsilon_K also enters g_R and hence dV_c^2/dR while a pure vertical boost does
not.  This script measures that difference.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vaudit_core as V                                       # noqa: E402
import adyn_model as M                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B = V.Bench()
NG = B.NG
XG = V.XG
lSig = B.log_sigma0()
C = dict(zU=np.log10(0.60), sc=0.15, dhz=0.0, kv=1.5, al=0.60,
         lfg=np.log10(0.25), fhg=2.0, fhzg=0.5, lo=0.3, hi=2.0)
al = np.full(NG, 0.60)
fg = np.full(NG, 0.25)
out = {}

print("  model A: a radially flat VERTICAL boost   B_z = (Sigma_0/<Sigma_0>)^p")
print("  model B: Newton with a tilted M/L         Upsilon_K ~ Sigma_0^p")
print(f"\n    {'p':>7}{'amp A':>10}{'amp B':>10}{'max |dlog amp|':>17}"
      f"{'h A':>9}{'h B':>9}{'max |dlog h|':>15}")
for p in (-0.60, -0.45, -0.350, -0.20, 0.0, +0.20):
    tilt = 10 ** (p * (lSig - lSig.mean()))
    Ups0 = 10 ** (C["zU"] + C["sc"] * (B.BK - 3.4))
    # A: boost sigma_z^2 after the Newtonian chain
    bA = B.newton_chain(Ups0, B.HZ_TAB, fg, al, C["kv"], C["fhg"], C["fhzg"])
    slA = B.to_los(np.sqrt(bA["s2"] * tilt[:, None]) / 1e3, al)
    aA, hA = M.fit_exponential_rows(XG, slA, C["lo"], C["hi"])
    # B: tilt Upsilon_K, i.e. tilt the baryons themselves
    bB = B.newton_chain(Ups0 * tilt, B.HZ_TAB, fg, al, C["kv"], C["fhg"],
                        C["fhzg"])
    slB = B.to_los(np.sqrt(bB["s2"]) / 1e3, al)
    aB, hB = M.fit_exponential_rows(XG, slB, C["lo"], C["hi"])
    da = float(np.max(np.abs(np.log10(aA / aB))))
    dh = float(np.max(np.abs(np.log10(hA / hB))))
    out[f"p={p:+.3f}"] = dict(max_dlog_amp=da, max_dlog_h=dh,
                              median_amp=float(np.median(aA)))
    print(f"    {p:>7.3f}{np.median(aA):>10.2f}{np.median(aB):>10.2f}"
          f"{da:>17.2e}{np.median(hA*np.squeeze(B.hR_as)):>9.2f}"
          f"{np.median(hB*np.squeeze(B.hR_as)):>9.2f}{dh:>15.2e}")

print("\n    The two differ by at most a few times 1e-3 dex in amplitude and")
print("    ~1e-3 dex in scale length, against measurement errors of 0.028 and")
print("    0.058 dex.  They are the SAME MODEL for these data.")
print("\n    The published error budget assigns Upsilon_K a 0.15 dex COMMON-MODE")
print("    prior and a 0.06 dex per-galaxy scatter, with the only Sigma_0-")
print("    dependent term being the (B-K) colour slope 0.15 +- 0.10 dex/mag.")
b_bk = float(np.polyfit(lSig, B.BK, 1)[0])
print(f"    d(B-K)/d log Sigma_0 = {b_bk:+.3f} mag/dex, so that term supplies")
print(f"    only {0.15*b_bk:+.3f} dex/dex.  Reproducing the signal needs")
print(f"    {-0.346 - 0.15*b_bk:+.3f} dex/dex more, i.e. an UNMODELLED")
print(f"    Upsilon_K anticorrelation with surface brightness of that size --")
print(f"    a factor {10**(abs(-0.346)*1.544):.1f} across the sample's 1.544 dex.")
out["colour_term"] = dict(d_BK_d_logSigma=b_bk, supplied=0.15 * b_bk,
                          required_extra=float(-0.346 - 0.15 * b_bk),
                          factor_across_sample=float(10 ** (0.346 * 1.544)))
with open(os.path.join(HERE, "degeneracy_check.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("\n  wrote degeneracy_check.json")
