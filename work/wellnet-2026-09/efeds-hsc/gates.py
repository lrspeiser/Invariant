"""Numerical gates for the lensing forward model.

Every one of these has a closed form, so a failure is unambiguous.  The
truncation gate exists because the brief's checklist says a FLAT error curve
versus a resolution parameter means a modelling mismatch, not a discretisation
error -- and this file caught exactly that: the first version of the Abel
projection was missing the cosh Jacobian and sat at 2/pi = 0.6366 of the truth
independently of every grid parameter.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
MPC, MSUN, G = P.MPC, P.MSUN, P.G


def main():
    print("=" * 78)
    print("GATES -- lensing forward model")
    print("=" * 78)
    out = {}
    r = np.geomspace(1e-4, 200.0, 3000) * MPC
    R = np.geomspace(0.05, 5.0, 14) * MPC

    # G1 singular isothermal sphere: Sigma = (2A/R) arccos(R/r_t), exact
    A = 1.0e10
    S, dS, _ = P.sigma_from_g(r, 4 * math.pi * G * A / r, R, r_trunc_mpc=50.0)
    e1 = float(np.max(np.abs(S / ((2 * A / R) * np.arccos(R / (50 * MPC)))
                             - 1)))
    out["G1_SIS_Sigma_max_rel_err"] = e1
    print(f"\n   G1  SIS Sigma vs closed form              {e1:.2e}  "
          f"{'PASS' if e1 < 3e-3 else 'FAIL'}")

    # G2 Plummer: Sigma and DeltaSigma both closed form
    M, a = 1e14 * MSUN, 0.5 * MPC
    g = G * (M * r ** 3 / (r ** 2 + a ** 2) ** 1.5) / r ** 2
    S, dS, _ = P.sigma_from_g(r, g, R, r_trunc_mpc=150.0)
    Sa = M * a ** 2 / (math.pi * (a ** 2 + R ** 2) ** 2)
    dSa = M / (math.pi * (a ** 2 + R ** 2)) - Sa
    e2 = float(np.max(np.abs(S / Sa - 1)))
    e2d = float(np.max(np.abs(dS / dSa - 1)))
    out["G2_Plummer_Sigma_max_rel_err"] = e2
    out["G2_Plummer_DeltaSigma_max_rel_err"] = e2d
    print(f"   G2  Plummer Sigma / DeltaSigma             {e2:.2e} / "
          f"{e2d:.2e}  {'PASS' if max(e2, e2d) < 5e-3 else 'FAIL'}")

    # G3 NFW against Wright & Brainerd (2000)
    z, M200, c = 0.35, 2e14 * MSUN, 4.0
    rho_c = 3.0 * (P.L.H0 * P.L.E(z)) ** 2 / (8.0 * math.pi * G)
    r200 = (M200 / (200.0 * rho_c * 4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
    rs = r200 / c
    Mn = M200 * (np.log(1 + r / rs) - r / (rs + r)) / (math.log(1 + c)
                                                       - c / (1 + c))
    gn = G * Mn / r ** 2
    errs = {}
    for rt in (25.0, 50.0, 100.0, 200.0):
        S, dS, _ = P.sigma_from_g(r, gn, R, r_trunc_mpc=rt)
        errs[rt] = (float(np.max(np.abs(S / P.nfw_sigma(R, M200, c, z) - 1))),
                    float(np.max(np.abs(dS
                                        / P.nfw_delta_sigma(R, M200, c, z)
                                        - 1))))
    out["G3_NFW_vs_WrightBrainerd_by_truncation"] = {
        str(k): {"Sigma": v[0], "DeltaSigma": v[1]} for k, v in errs.items()}
    e3 = errs[200.0]
    print(f"   G3  NFW vs Wright-Brainerd (r_t=200 Mpc)   {e3[0]:.2e} / "
          f"{e3[1]:.2e}  {'PASS' if max(e3) < 5e-3 else 'FAIL'}")
    spread = max(v[0] for v in errs.values()) - min(v[0] for v in errs.values())
    print(f"   G3b truncation NOT flat: Sigma error moves {spread:.2e} over "
          f"r_t = 25 -> 200 Mpc  "
          f"{'PASS' if spread > 1e-3 else 'FAIL (flat => modelling mismatch)'}")
    out["G3b_truncation_error_spread"] = spread

    # G4 quadrature convergence in n_t
    conv = {}
    for nt in (125, 250, 500, 1000):
        S, dS, _ = P.sigma_from_g(r, gn, R, r_trunc_mpc=200.0, n_t=nt)
        conv[nt] = float(np.max(np.abs(dS / P.nfw_delta_sigma(R, M200, c, z)
                                       - 1)))
    out["G4_nt_convergence"] = conv
    print(f"   G4  n_t convergence (DeltaSigma error): "
          + ", ".join(f"{k}:{v:.2e}" for k, v in conv.items()))

    # G5 reduced shear reduces to gamma when kappa -> 0
    src = P.Sources(P.fit_source_nz(verbose=False))
    scr = src.sigma_crit_eff(0.35)
    kap = P.nfw_sigma(R, M200, c, z) / scr
    print(f"   G5  kappa at R = 0.05-5 Mpc for a 2e14 halo: "
          f"{kap.max():.4f} -> {kap.min():.5f}; the reduced-shear correction "
          f"is a {100 * kap.max():.1f}% effect at the innermost bin")
    out["G5_kappa_range"] = [float(kap.min()), float(kap.max())]

    # G6 monotone M_dyn and non-negative rho for the actual law on a real case
    #     is checked inside efeds_hsc.py per system.
    with open(os.path.join(HERE, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n   wrote gates.json")


if __name__ == "__main__":
    main()
