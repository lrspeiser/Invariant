"""Attempt a FULL nonlinear 3-D solve for the tournament's survivors.

Everything in the cluster channel is the tensor lane's calibrated spherical
surrogate (harmonic shell mean of k, pushed through the exact spherical
reduction), whose worst departure from six full 3-D solves is 20.4 per cent.
The tensor lane also recorded that above |A_T| ~ 20 the response tensor's
condition number reaches ~1e5 and Jacobi-preconditioned CG stops converging,
so those points are reported from the surrogate only.

The survivors here sit at |A| = 25 to 102.  This script measures, rather than
assumes, whether they can be solved in 3-D at all: it builds the survivor's K
on the cluster grid with the tensor lane's own `run.solve_with_K`, and reports
the condition number, the convergence and the resulting B(r) beside the
surrogate's.

Writes verify3d.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TENSOR = os.path.normpath(os.path.join(HERE, "..", "tensor"))
for p in (HERE, TENSOR):
    if p not in sys.path:
        sys.path.insert(0, p)

import field as F                                               # noqa: E402
import mechanism as MECH                                        # noqa: E402
import run as RUN                                               # noqa: E402
import wellnet as W                                             # noqa: E402
import ch_cluster as CC                                         # noqa: E402
from tw_core import Candidate, KPC                               # noqa: E402

XP = CC.XP
CASES = [
    dict(tag="survivor_1", struct="tensor_S", well="plaw_p0q1s2_L300",
         form="pow", m=2.0, I0=3e12, A=-94.66, a0=1.041e-10),
    dict(tag="survivor_4", struct="tensor_S", well="plaw_p0q1s2_L300",
         form="sat", m=2.0, I0=1e12, A=-25.00, a0=1.020e-10),
    dict(tag="scalar_best_member", struct="scalar_a0", well=None,
         form="sat", m=4.0, I0=1e12, A=30.50, a0=1.058e-10),
    dict(tag="tensor_lane_reference", struct="tensor_S",
         well="plaw_p0q1s2_L300", form="sat", m=4.0, I0=1e12, A=-24.7,
         a0=1.2e-10),
]


def K_on_grid(cand, c, pts):
    """(6, n, n, n) symmetric K field for a candidate on the cluster grid."""
    N = CC.newton_full(pts, c, xp=XP)
    fro, That = CC._traceless_norm_and_hat(N["H"], XP)
    inv = dict(gn=N["gmag"] / 1.2e-10, phi=XP.abs(N["Phi"]),
               rhobar=XP.maximum(N["rho"], 1e-40),
               tidal=XP.maximum(fro, 1e-45))
    I = XP.maximum(inv[cand.inv] / cand.I0, 1e-300)
    if cand.form == "sat":
        u = I ** cand.m
        Wf = u / (1.0 + u)
    elif cand.form == "pow":
        Wf = I ** cand.m
    elif cand.form == "log":
        Wf = XP.log1p(I ** cand.m)
    else:
        Wf = 1.0 / (1.0 + I ** cand.m)
    if cand.struct == "tensor_S":
        S = W.S_tensor(pts, XP.asarray(c["pos"]), XP.asarray(c["Mg"]),
                       family=cand.extra["well"]["family"],
                       p=cand.extra["well"]["p"], q=cand.extra["well"]["q"],
                       s=cand.extra["well"]["s"], L=cand.extra["well"]["L"],
                       exclude_nearest=cand.extra["well"]["exclude_nearest"],
                       xp=XP)
        M = (cand.A * Wf)[:, None] * S
    else:                                    # isotropic equivalent of a0 -> a0(1+A W)
        keq = (1.0 + cand.A * Wf) ** (-2.0 / 3.0)
        M = XP.log(keq)[:, None] * XP.asarray([1.0, 1.0, 1.0, 0, 0, 0])[None, :]
    K = W.sym3_expm(M, XP)
    n = c["n"]
    return XP.moveaxis(K.reshape(n, n, n, 6), -1, 0), Wf


def main():
    out = []
    c = CC.CL.build(n=64, seed=20260903)
    pts = RUN.points_of(c, XP)
    mu = F.Mu("simple", a0=1.2e-10)
    for case in CASES:
        ws = None
        if case["well"]:
            ws = [w for w in CC.WELL_SETTINGS if w["tag"] == case["well"]][0]
        cand = Candidate(case["tag"], base="aqual", a0=case["a0"], inv="phi",
                         form=case["form"], m=case["m"], I0=case["I0"],
                         struct=case["struct"], A=case["A"],
                         extra=dict(well=ws) if ws else {})
        Kf, Wf = K_on_grid(cand, c, pts)
        ev = W.sym3_eigvals(XP.moveaxis(Kf, 0, -1).reshape(-1, 6), XP)
        lo = float(W.asnumpy(XP.min(ev)))
        hi = float(W.asnumpy(XP.max(ev)))
        rec = dict(**case, K_min_eig=lo, K_max_eig=hi, K_cond=hi / max(lo, 1e-300),
                   W_max=float(W.asnumpy(XP.max(Wf))))
        print(f"{case['tag']:<24} K eig {lo:.3e} .. {hi:.3e}  cond "
              f"{hi/max(lo,1e-300):.3e}", flush=True)
        if lo <= 0 or not np.isfinite(hi / max(lo, 1e-300)) \
                or hi / max(lo, 1e-300) > 1e12:
            rec["solved"] = False
            rec["why"] = ("condition number beyond 1e12; the discrete operator "
                          "is numerically singular and there is nothing to "
                          "solve")
            out.append(rec)
            continue
        t0 = time.time()
        try:
            Psi, info = RUN.solve_with_K(c, Kf, mu, xp=XP, outer=40,
                                         tol_outer=1e-5)
            g = F.gradient_mag(Psi, c["dx"], XP)[0]
            R = XP.asarray(c["R"]).ravel()
            gm = g.ravel()
            B3 = []
            for rk in MECH.RADII:
                m = XP.abs(R - rk * KPC) < c["dx"]
                B3.append(float(W.asnumpy(gm[m].mean())))
            # B = |g|(response) / |g|(K = I), the SAME definition the
            # surrogate uses, so the two are directly comparable
            Kid = XP.zeros_like(Kf)
            Kid[0] = 1.0
            Kid[1] = 1.0
            Kid[2] = 1.0
            Psi0, _ = RUN.solve_with_K(c, Kid, mu, xp=XP, outer=40,
                                       tol_outer=1e-5)
            g0 = F.gradient_mag(Psi0, c["dx"], XP)[0].ravel()
            B0 = []
            for rk in MECH.RADII:
                m = XP.abs(R - rk * KPC) < c["dx"]
                B0.append(float(W.asnumpy(g0[m].mean())))
            rec.update(solved=True, seconds=time.time() - t0,
                       outer_iters=int(info.get("outer", -1)),
                       resid=float(info.get("rel", float("nan"))),
                       g_shell=B3, g_shell_KI=B0,
                       B_3d=[a / b for a, b in zip(B3, B0)])
            print(f"    solved in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:                        # noqa: BLE001
            rec.update(solved=False, why=repr(e)[:300],
                       seconds=time.time() - t0)
            print(f"    FAILED: {repr(e)[:160]}", flush=True)
        out.append(rec)
    with open(os.path.join(HERE, "verify3d.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote verify3d.json")


if __name__ == "__main__":
    main()
