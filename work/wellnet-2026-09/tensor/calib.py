"""Which shell average of k reproduces the 3-D shell-averaged |g|?

The map needs a surrogate for the full nonlinear 3-D solve, and the obvious
choice -- average k over the shell, then run the 1-D reduction once -- is not
obviously the right one when k varies by orders of magnitude across the shell.
Three candidates are compared against the actual 3-D solve at a range of
amplitudes:

  A  boost( <k> )          arithmetic mean conductivity, then the reduction
  B  boost( 1/<1/k> )      harmonic mean conductivity
  C  < boost(k) >          the reduction applied cell by cell, then averaged

C is the one that matches the observable: <|g|> over the shell is the average
of the local field, and locally |g| = F/(mu k) with F the local radial flux.
This script measures which is right rather than arguing about it.
"""
from __future__ import annotations

import json
import time

import numpy as np

import field as F
import mechanism as M
import run as RUN
import wellnet as W
from wellnet import G, A0, KPC, MSUN

XP = M.XP
mu = F.Mu("simple")

clu, fld, mem = M.contexts(n=64)
c = clu["c"]
R = XP.asarray(c["R"])
KI = RUN.identity_K(c["rho"].shape, XP)
Psi0, _ = RUN.solve_with_K(c, KI, mu, xp=XP, outer=60)
g0, _ = F.gradient_mag(Psi0, c["dx"], XP)
base0 = [float(g0[XP.abs(R - rk * KPC) < c["dx"]].mean()) for rk in M.RADII]
print("baseline |g|/a0 :", " ".join(f"{v/A0:.5f}" for v in base0))

kw = dict(family="plaw", p=1.0, q=2.0, s=1.5, m=1.0, L=300 * KPC,
          M_0=1e11 * MSUN, exclude_nearest=False)
Sfull = W.S_tensor(clu["pts"], clu["wx"], clu["wm"], xp=XP,
                   gN_local=clu["gN"], **kw)
Ssub = Sfull[clu["sub_idx"]]

rows = []
print(f"\n{'A_T':>6} {'CGres':>9}   " + "  ".join(
    f"{'3D':>6} {'A':>6} {'B':>6} {'C':>6}" for _ in M.RADII))
for A_T in (-1.0, -2.0, -3.0, -4.5, -6.0, -8.0):
    K = W.sym3_expm(A_T * Sfull, XP)
    Kf = XP.moveaxis(K.reshape(c["n"], c["n"], c["n"], 6), -1, 0)
    t = time.time()
    Psi, info = RUN.solve_with_K(c, Kf, mu, xp=XP, outer=80, tol_outer=1e-6)
    gm, _ = F.gradient_mag(Psi, c["dx"], XP)
    b3 = [float(gm[XP.abs(R - rk * KPC) < c["dx"]].mean()) / b
          for rk, b in zip(M.RADII, base0)]
    kcell = W.sym3_quad(W.sym3_expm(A_T * Ssub, XP), clu["sub_rhat"], XP)
    est = {"A": [], "B": [], "C": []}
    for j, rk in enumerate(M.RADII):
        sh = clu["sub_shells"][j]
        kk = kcell[sh]
        i = min(int(np.searchsorted(clu["r_prof"], rk * KPC)),
                len(clu["r_prof"]) - 1)
        rr = float(clu["r_prof"][i])
        MM = float(clu["M_prof"][i])
        one = XP.ones_like(kk)
        gref = mu.invert(XP.full(kk.shape, G * MM / rr ** 2), one, XP)
        loc = mu.invert(XP.full(kk.shape, G * MM / rr ** 2), kk, XP)
        est["C"].append(float((loc / gref).mean()))
        for tag, kbar in (("A", float(kk.mean())),
                          ("B", float(1.0 / (1.0 / kk).mean()))):
            kb = XP.full((1,), kbar)
            est[tag].append(float(
                mu.invert(XP.full((1,), G * MM / rr ** 2), kb, XP)[0]
                / mu.invert(XP.full((1,), G * MM / rr ** 2),
                            XP.ones(1), XP)[0]))
    rows.append(dict(A_T=A_T, B_3d=b3, A=est["A"], B=est["B"], C=est["C"],
                     outer=info["outer"], seconds=time.time() - t))
    print(f"{A_T:>6.1f} {'':>9}   " + "  ".join(
        f"{b3[j]:6.3f} {est['A'][j]:6.3f} {est['B'][j]:6.3f} "
        f"{est['C'][j]:6.3f}" for j in range(len(M.RADII))))
    del K, Kf, Psi, gm
    M._free()

err = {t: max(abs(np.array(r[t]) / np.array(r["B_3d"]) - 1).max()
              for r in rows) for t in ("A", "B", "C")}
print("\nworst relative departure from the 3-D solve, over all A_T and radii:")
for t in ("A", "B", "C"):
    print(f"   {t}: {100*err[t]:.1f}%")
json.dump(dict(rows=rows, worst=err), open("calibration.json", "w"), indent=1)
print("written calibration.json")
