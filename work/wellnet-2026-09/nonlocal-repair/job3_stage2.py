"""JOB 3(b), STAGE 2 -- full 3-D anisotropic solves, redone with the tidal
regulariser handled correctly.

A BUG IN MY OWN FIRST VERSION, recorded because it changes the conclusion.
`families.tidal_hat` normalises by sqrt(eps_T^2 + |T0|^2), and eps_T carries
units of s^-2.  My first run passed eps_T = 1e-30, which sounds small but is
190 TIMES LARGER than the actual |T0| = 5.3e-33 s^-2 at 50 kpc from a
6e10 Msun galaxy.  That suppressed That by a factor 200, so f_T That was
numerically zero and the "anisotropy does no independent work" reading was an
artefact of my own regulariser, not a measurement.

The lane's own default is eps_T = a0/(10 kpc) = 3.9e-31 s^-2, which is still
74 times |T0| at 50 kpc.  That is a deliberate physical choice -- "where the
tidal field is weaker than a0 per 10 kpc the direction is undefined" -- and it
means that under the existing screen's own regulariser THE ANISOTROPY IS
SWITCHED OFF THROUGHOUT GALAXY OUTSKIRTS.  Worth saying out loud.

So eps_T is screened here as a parameter: the lane's default, an intermediate
value, and a value small enough that That is the exact monopole direction
everywhere the rotation curve is measured.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

import common as C

SCREEN = ("C:/Users/henry/Documents/Codex/2026-08-21/"
          "Invariant-main-integration/work/wellnet-2026-09/screen")
GLAB = ("C:/Users/henry/Documents/Codex/2026-08-21/"
        "Invariant-main-integration/work/gravitylab")
for pth in (SCREEN, GLAB):
    if pth not in sys.path:
        sys.path.append(pth)

import families as FA          # noqa: E402
import fieldsolve as FS        # noqa: E402
from job3_tensor import (EXP_CLIP, S6, R0_SUN, f_nl, f_T, kr_kt_kz,  # noqa: E402
                         galaxy_qbar)

KPC, MSUN = FA.KPC, FA.MSUN
T0_ = time.time()


def say(*a):
    print(*a, flush=True)


def main(n=64, L=160.0):
    say("=" * 78)
    say("STAGE 2  Full 3-D anisotropic solves, eps_T screened")
    say("=" * 78)
    res = json.load(open("tensor_atom_screen.json", encoding="utf-8"))
    cands = res["stage1"]["stage2_candidates"]
    #  keep one directional and one isotropic candidate, deduplicated
    seen, keep = set(), []
    for cd in cands:
        k = (cd["f_nl"], cd["f_T"], cd["a"], cd["p"], cd["c"], cd["m"],
             cd["rho_ref"], cd["qdef"], cd["nonlocal_qbar"])
        if k in seen:
            continue
        seen.add(k)
        keep.append(cd)
    keep = keep[:2]

    from scipy.ndimage import map_coordinates
    box = FA.Box(n, L)
    Mg, Rd, hz = 6.0e10, 3.0, 1.0
    rho = FA.expdisk_rho(box.pts, Mg * MSUN, Rd * KPC,
                         hz * KPC).reshape(box.shape)
    rho = FA.normalise_mass(rho, box.vol, Mg * MSUN)
    say(f"grid {n}^3, box {L} kpc, h = {box.h / KPC:.2f} kpc; exponential "
        f"disk M = {Mg:.2g} Msun, R_d = {Rd} kpc, h_z = {hz} kpc")
    say(f"h = {box.h / KPC:.2f} kpc does NOT resolve the |z| = 1.1 kpc Oort "
        f"column, so the")
    say("vertical force is quoted at z = 3 and 6 kpc and the Oort number "
        "stays with")
    say("the Stage-1 slab proxy.")
    rN = FS.solve_newton(rho, box, tol=1e-10, maxiter=4000)
    vN = FS.vcirc_axis(rN["Psi"], box, np.array([8., 16., 24., 32., 48.])
                       * KPC)
    Rq = np.array([8., 16., 24., 32., 48.]) * KPC
    gzN = [abs(FS.force_at(rN["Psi"], box, np.array(
        [R0_SUN * KPC, 0.0, z * KPC]))[2]) for z in (3.0, 6.0)]
    say(f"Newtonian reference: {rN['iters']} iters, resid {rN['resid']:.2e}")

    rr = box.r.ravel()
    nh = (np.stack([box.X.ravel(), box.Y.ravel(), box.Z.ravel()], 1)
          / np.maximum(rr, 1e-30)[:, None])
    EPS = [("lane default a0/10kpc", FA.A0 / (10.0 * KPC)),
           ("intermediate 1e-33", 1.0e-33),
           ("unregularised 1e-37", 1.0e-37)]
    out = {"orientation": [], "runs": []}
    Thats = {}
    say("")
    say("Orientation of That in the outskirts, per eps_T.  The exterior "
        "monopole")
    say(f"value that Stage 1 assumes is n.That.n = -2/sqrt6 = {-2 / S6:+.4f}.")
    for tag, eps in EPS:
        Th = FA.tidal_hat(rN["Psi"], box.h, dict(eps_T=eps))
        Thats[tag] = Th
        row = dict(eps_T=float(eps), tag=tag)
        for lo, hi in ((10, 20), (20, 40), (40, 70)):
            sel = (rr > lo * KPC) & (rr < hi * KPC)
            t = np.einsum("pi,pij,pj->p", nh[sel], Th[sel], nh[sel])
            row[f"n_That_n_{lo}_{hi}kpc"] = float(t.mean())
        out["orientation"].append(row)
        say(f"   eps_T = {eps:.2e} ({tag:<22s}) : 10-20 kpc "
            f"{row['n_That_n_10_20kpc']:+.4f}   20-40 kpc "
            f"{row['n_That_n_20_40kpc']:+.4f}   40-70 kpc "
            f"{row['n_That_n_40_70kpc']:+.4f}")
    say("   The lane's own default switches the anisotropy OFF in galaxy "
        "outskirts:")
    say("   |T0| = 5.3e-33 s^-2 at 50 kpc for a 6e10 Msun galaxy, 74 times "
        "smaller")
    say("   than a0/(10 kpc).  Any conclusion about 'the anisotropy does no "
        "work'")
    say("   drawn at that eps_T is a statement about the regulariser.")

    for cd in keep:
        rho_ms = rho.ravel() / MSUN * KPC ** 3 + C.NK.RHO_BAR_B
        if cd["qdef"] == "delta":
            q = np.clip(cd["rho_ref"] / rho_ms - 1.0, 0.0, 1 - 1e-15)
        else:
            q = 1.0 / (1.0 + rho_ms / cd["rho_ref"])
        if cd["nonlocal_qbar"]:
            qg = q.reshape(box.shape)
            ns = 24
            acc = np.zeros(q.shape[0])
            P = np.stack([box.X.ravel(), box.Y.ravel(), box.Z.ravel()], 1)
            for sv in (np.arange(ns) + 0.5) / ns:
                idx = ((P * (1.0 - sv) / box.h) + (n - 1) / 2.0).T
                acc += map_coordinates(qg, idx, order=1, mode="nearest")
            qb = acc / ns
        else:
            qb = q
        fn = np.clip(f_nl(cd["f_nl"], qb, cd["a"], cd["p"]), -EXP_CLIP,
                     EXP_CLIP)
        say("")
        say(f"CANDIDATE f_nl={cd['f_nl']} a={cd['a']:g} p={cd['p']:g} "
            f"f_T={cd['f_T']} c={cd['c']:g} m={cd['m']:g} "
            f"rho_ref={cd['rho_ref']:g} q={cd['qdef']} "
            f"nonlocal={int(cd['nonlocal_qbar'])}")
        for tag, eps in EPS:
            Th = Thats[tag]
            for vtag, cval in ((f"f_T on (c={cd['c']:g})", cd["c"]),
                               ("f_T = 0 CONTROL", 0.0)):
                if cd["f_T"] == "zero" and cval == 0.0 and vtag.startswith(
                        "f_T on"):
                    continue
                ft = (f_T(cd["f_T"], qb, cval, cd["m"]) if cval
                      else np.zeros_like(qb))
                M = (fn[:, None, None] * np.eye(3)[None]
                     + ft[:, None, None] * Th)
                try:
                    K = FA._sym_expm(M, "K")
                    rK = FS.solve_K(rho, K, box, tol=1e-10, maxiter=6000,
                                    Mtot=float(rho.sum() * box.vol))
                except Exception as e:                  # noqa: BLE001
                    say(f"   {tag} | {vtag}: FAILED {type(e).__name__}: {e}")
                    continue
                vK = FS.vcirc_axis(rK["Psi"], box, Rq)
                gzK = [abs(FS.force_at(rK["Psi"], box, np.array(
                    [R0_SUN * KPC, 0.0, z * KPC]))[2]) for z in (3.0, 6.0)]
                b3d = (vK / vN) ** 2
                sl = float(np.mean(np.gradient(
                    np.log(np.maximum(vK, 1e-30)), np.log(Rq))))
                row = dict(setting={k: cd[k] for k in
                                    ("f_nl", "f_T", "a", "p", "c", "m",
                                     "rho_ref", "qdef", "nonlocal_qbar")},
                           eps_T=float(eps), eps_tag=tag, variant=vtag,
                           R_kpc=(Rq / KPC).tolist(),
                           radial_boost=b3d.tolist(),
                           mean_radial_boost=float(np.mean(b3d)),
                           vertical_boost_z3=float(gzK[0] / gzN[0]),
                           vertical_boost_z6=float(gzK[1] / gzN[1]),
                           vert_over_rad=float(gzK[0] / gzN[0]
                                               / np.mean(b3d)),
                           outer_logslope=sl,
                           newton_logslope=float(np.mean(np.gradient(
                               np.log(vN), np.log(Rq)))),
                           shell_spread=float(rK["shell_spread"]),
                           iters=int(rK["iters"]), resid=float(rK["resid"]))
                out["runs"].append(row)
                say(f"   eps_T {eps:.1e} | {vtag:<18s} radial boost "
                    + " ".join(f"{x:5.2f}" for x in b3d)
                    + f"   vertical {row['vertical_boost_z3']:.3f} (z=3) "
                      f"{row['vertical_boost_z6']:.3f} (z=6)   "
                      f"dlnv/dlnr {sl:+.3f}")
                del K, rK
    res["stage2_full3d"] = out
    say("")
    say("SUMMARY: vertical-to-radial response ratio, f_T on versus off.")
    say("   eps_T        variant             radial   vert(z3)  ratio")
    for r_ in out["runs"]:
        say(f"   {r_['eps_T']:.1e}  {r_['variant']:<18s} "
            f"{r_['mean_radial_boost']:7.3f} {r_['vertical_boost_z3']:8.3f} "
            f"{r_['vert_over_rad']:7.3f}")
    res["runtime_stage2_s"] = time.time() - T0_
    with open("tensor_atom_screen.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    say(f"\nupdated tensor_atom_screen.json  ({time.time() - T0_:.1f} s)")


if __name__ == "__main__":
    main()
