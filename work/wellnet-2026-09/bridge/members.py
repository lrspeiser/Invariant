"""members.py -- the binding constraint: the carrier term inside a cluster
member galaxy, as a function of rho_*.

BL.5: at the fiducial (eps = 0.3, rho_* = 1e-24) the path family multiplies
a member galaxy's internal gravity by 7.6 at 20 kpc, because a dense body in
a medium near rho_* sits in a Phi_3 trough.  Everything is linear in eps, so
the member-safe amplitude is  eps_safe = tol / k(rho_*)  with
k = (g_3 / g_N)(20 kpc) per unit eps.  This module recomputes k with THIS
lane's column code on the compiler's own member-probe caricature (5e10 Msun
galaxy at 700 kpc inside the 2e14 Msun gas + 3e12 BCG) and scans rho_*.

The data that would test member confinement (MUSE/Granata dispersions, Gaia
dynamical products) are in the confirmation reserve and are NOT opened: the
tolerance is a declared ladder, not a measurement.

    python members.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import guard
import scene as S

C = S.C
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KPC, MSUN, G = S.KPC, S.MSUN, S.G

RHO_STAR_SCAN = {"1e-24": 1.0e-24, "3e-25": 3.0e-25, "1e-25": 1.0e-25,
                 "3e-26": 3.0e-26, "1e-26": 1.0e-26, "3e-27": 3.0e-27,
                 "rho_mean": S.RHO_MEAN}
TOLERANCES = (0.02, 0.05, 0.10, 0.20)
BL_FACTOR_20KPC = 7.565          # BL's recorded total factor at eps = 0.3
BL_CARRIER_20KPC = 7.600         # BL's recorded carrier-only factor


def P_points(comps, pts, n_dir=20000, chunk=8):
    dirs = S.fib_dirs(n_dir)
    w = 4.0 * math.pi / n_dir
    out = np.zeros(len(pts))
    for s0 in range(0, len(pts), chunk):
        P = pts[s0:s0 + chunk]
        Cp = np.zeros((len(P), n_dir))
        Cm = np.zeros((len(P), n_dir))
        for c in comps:
            Cp += S.half_column(c, P[:, None, :], dirs[None, :, :])
            Cm += S.half_column(c, P[:, None, :], -dirs[None, :, :])
        out[s0:s0 + chunk] = w * (Cp * Cm).sum(1)
    return out


def member_scan(n_dir=20000):
    gal = C.Plummer(C.GAL_M, C.GAL_A, (C.MEMBER_D, 0.0, 0.0))
    gas = C.Plummer(C.CLU_GAS_M, C.CLU_GAS_A, (0.0, 0.0, 0.0))
    bcg = C.Plummer(C.BCG_M, C.BCG_A, (0.0, 0.0, 0.0))
    comps = [gal, gas, bcg]
    centre = gal.c
    rg = np.geomspace(4.0 * KPC, 80.0 * KPC, 28)
    # antipodally symmetric directions (BL's reason: the cluster's dipole
    # pull cancels in the average and only the galaxy's own monopole survives)
    half = S.fib_dirs(6)
    dirs = np.vstack([half, -half])
    pts = (centre[None, None, :] + rg[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
    t0 = time.perf_counter()
    P = P_points(comps, pts, n_dir).reshape(len(rg), len(dirs))
    rho = sum(c.rho(pts) for c in comps).reshape(len(rg), len(dirs))
    rho_gal = gal.rho(pts).reshape(len(rg), len(dirs))
    gN = G * gal.M_enc(rg) / rg ** 2
    rows = {}
    for tag, rs in RHO_STAR_SCAN.items():
        phi3 = (-0.5 * G * S.dphi_vac(rho, rs) * P).mean(1)      # per unit eps
        # INWARD carrier acceleration: a = -grad Phi_3, so a rising Phi_3
        # (dPhi_3/dr > 0) pulls inward; gN is the inward Newtonian magnitude.
        # (The first version took g3 = -dPhi_3/dr, outward-positive, against
        # an inward-positive gN and reported BL's confinement as expulsion --
        # a sign-convention slip caught by the BL comparison.)
        g3_in = np.gradient(phi3, np.log(rg)) / rg
        k = g3_in / gN
        at = {int(r): float(np.interp(r * KPC, rg, k)) for r in (10, 20, 30)}
        rows[tag] = dict(rho_star=rs, k_per_eps=at,
                         factor_at_eps0p3_20kpc=1.0 + 0.3 * at[20],
                         eps_safe={f"tol_{t:.2f}": (t / at[20] if at[20] > 0 else float("inf"))
                                   for t in TOLERANCES},
                         phi3_trough_depth_per_eps_kms2=float(
                             (phi3.max() - phi3.min()) / 1e6),
                         rho_at_20kpc=float(np.interp(20 * KPC, rg, rho.mean(1))),
                         rho_gas_at_member=float(gas.rho(centre[None, :])[0]))
    return dict(rows=rows, r_kpc=(rg / KPC).tolist(),
                seconds=time.perf_counter() - t0,
                P_at_20kpc_mean=float(np.interp(20 * KPC, rg, P.mean(1))),
                gN_20kpc=float(np.interp(20 * KPC, rg, gN)),
                rho_gal_20kpc=float(np.interp(20 * KPC, rg, rho_gal.mean(1))))


def main():
    guard.arm()
    t0 = time.perf_counter()
    res = member_scan()
    fid = res["rows"]["1e-24"]
    res["BL_comparison"] = dict(
        BL_carrier_factor_20kpc_eps0p3=BL_CARRIER_20KPC,
        this_lane_carrier_factor_20kpc_eps0p3=fid["factor_at_eps0p3_20kpc"],
        ratio=fid["factor_at_eps0p3_20kpc"] / BL_CARRIER_20KPC)
    res["statement"] = (
        "k(rho_*) = (g_3/g_N)(20 kpc) per unit eps on the compiler's member "
        "caricature; eps_safe(tol) = tol / k. The tolerance is DECLARED (the "
        "member internal-dynamics data are in the confirmation reserve, not "
        "opened).")
    res["provenance"] = guard.summary()
    res["wall_seconds"] = time.perf_counter() - t0
    with open(os.path.join(RES, "members.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(res, fh, indent=1, default=float)
    for tag, r in res["rows"].items():
        print(f"rho_* = {tag:<9} k(20 kpc) = {r['k_per_eps'][20]:9.3f} per eps ; "
              f"factor at eps=0.3: {r['factor_at_eps0p3_20kpc']:8.3f} ; "
              f"eps_safe(5%) = {r['eps_safe']['tol_0.05']:.2e}")
    print("BL comparison:", res["BL_comparison"])


if __name__ == "__main__":
    main()
