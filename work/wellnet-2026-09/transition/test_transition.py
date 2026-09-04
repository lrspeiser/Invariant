"""Tests for the transition lane.  Run: python test_transition.py

Each test prints PASS/FAIL and the number it checked.  Tests that found real
bugs are marked  [caught a bug]  in the docstring.
"""
from __future__ import annotations

import math
import sys

import numpy as np

import build as B
import common as K
import decl
import fitlib as F
import pipeline as P

MPC, MSUN = P.MPC, P.MSUN
FAILS = []


def check(name, ok, detail=""):
    print(f"   {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------- unit tests
def t_rar_limits():
    """RAR reduces to Newton at high g and to sqrt(g a0) deep in MOND."""
    a0 = decl.A0_RAR
    hi = B.g_rar(np.array([1e4 * a0]))[0]
    lo = B.g_rar(np.array([1e-6 * a0]))[0]
    check("RAR -> Newton at g >> a0", abs(hi / (1e4 * a0) - 1) < 1e-6,
          f"ratio {hi / (1e4 * a0):.9f}")
    check("RAR -> sqrt(g a0) deep MOND",
          abs(lo / math.sqrt(1e-6 * a0 * a0) - 1) < 2e-3,
          f"ratio {lo / math.sqrt(1e-6 * a0 * a0):.6f}")


def t_r500_solver():
    """R500_dyn solves M_dyn(R) = 500 rho_c (4pi/3) R^3 to round-off."""
    c = K.Bundle.__new__(K.Bundle)          # not needed; use a real cluster
    cl = _one_cluster()
    R = cl.R500
    lhs = cl.M_dyn(R)
    rhs = 500.0 * B.rho_c(cl.z) * (4 * math.pi / 3) * R ** 3
    check("R500_dyn residual", abs(lhs / rhs - 1) < 2e-3,
          f"M_dyn/target = {lhs / rhs:.8f}")


_CL = None


def _one_cluster():
    global _CL
    if _CL is None:
        Mg, _ = B.gas_from_accept("ABELL_2744")
        _CL = B.Cluster("T", "sl", 0.308, B.R_GRID, Mg,
                        np.zeros_like(Mg))
    return _CL


def t_sis_kappa():
    """kappa_bar for a singular isothermal sphere has the analytic form.

    For rho ~ r^-2 with M(r) = 2 sigma^2 r / G, Sigma(R) = sigma^2/(2 G R) and
    Sigma_bar(<R) = sigma^2/(G R) = 2 Sigma(R), so kappa_bar = theta_E/theta
    with theta_E = 4 pi sigma^2/c^2 D_ls/D_s.  This pins the whole projection
    +-geometry chain against a closed form.
    """
    sig = 1.2e6                                     # m/s
    r = B.R_GRID
    M = 2.0 * sig ** 2 * r / P.G
    g = P.G * M / r ** 2
    z_l, z_s = 0.4, 2.0
    scr, D_l = K.sigma_crit(z_l, z_s)
    th = np.array([5.0, 10.0, 20.0, 40.0])
    R = th * P.ARCSEC * D_l
    kb = K.M_2d(r, M, R) / (math.pi * R ** 2 * scr)
    D_s = float(P.d_ang(z_s))
    D_ls = float(P.d_ang12(z_l, z_s))
    thE = 4.0 * math.pi * sig ** 2 / P.CLIGHT ** 2 * D_ls / D_s / P.ARCSEC
    # the declared 20 Mpc truncation removes exactly R/(pi r_t) of the SIS's
    # projected mass; that is physics, not a discretisation error, and the
    # residual below matches it to four figures.
    pred = (thE / th) * (1.0 - R / (math.pi * 20.0 * P.MPC))
    err = float(np.max(np.abs(kb / pred - 1.0)))
    check("SIS kappa_bar vs closed form (truncation-corrected)", err < 1e-3,
          f"max rel err {err:.6f}, theta_E = {thE:.3f} arcsec")
    # the OLD route, kept as the record of the bug this test caught
    S, dS = P.sigma_from_g(r, g, R, r_trunc_mpc=20.0)[:2]
    old = float(np.max(np.abs((S + dS) / scr / pred - 1.0)))
    print(f"          (sigma_from_g Sigma_bar route: max rel err {old:.5f}"
          f" -- the bug this test caught)")


def t_slip_linearity():
    """A per-system CONSTANT slip factors out of the Abel projection exactly.

    This is what lets the fitter apply exp(c + alpha ln M) after the
    projection instead of inside it.  If it were not exact the whole speed-up
    would be wrong.
    """
    bd = _bundle()
    idx = bd.ef_idx[:20]
    fac = 1.7
    S1, dS1 = K.project_slip(bd.syss, bd.obs, idx, lambda sm: 1.0)
    S2, dS2 = K.project_slip(bd.syss, bd.obs, idx, lambda sm: fac)
    e1 = float(np.max(np.abs(S2 / S1 - fac)))
    e2 = float(np.max(np.abs(dS2 / dS1 - fac)))
    check("constant slip factors out of the projection",
          max(e1, e2) < 1e-12, f"max rel err {max(e1, e2):.3e}")


def t_gplus_nonlinear():
    """g_+ is NOT linear in the slip amplitude, and the code keeps that."""
    bd = _bundle()
    S, dS = K.project_slip(bd.syss, bd.obs, bd.ef_idx, lambda sm: 1.0)
    a = bd.F.gplus(S, dS, 1.0)
    b = bd.F.gplus(S, dS, 2.0)
    r = float(np.max(b / a))
    check("g_+ nonlinearity carried, not linearised", r > 2.0,
          f"max g_+(2)/g_+(1) = {r:.6f} (exactly 2 would mean linearised)")


def t_locuss_radius_circularity():
    """[caught a bug] ln(r/R500_dyn) for LoCuSS is a function of ln S.

    With M_dyn ~ r^m near the aperture, r500_WL/R500_dyn = S^(1/(3-m)).  A
    fit using R500_dyn as the LoCuSS radius axis would therefore 'discover'
    a radial dependence that is pure algebra.  This test is the reason the
    declaration was amended to the external catalogue aperture.
    """
    bd = _bundle()
    S = np.array([p["M_WL"] / c.M_dyn(p["r"])
                  for c, p in zip(bd.lo, bd.lop)])
    x = np.array([p["r"] / c.R500 for c, p in zip(bd.lo, bd.lop)])
    rho = float(np.corrcoef(np.log(S), np.log(x))[0, 1])
    check("LoCuSS R500_dyn radius axis is circular", rho > 0.8,
          f"corr(ln S, ln r/R500_dyn) = {rho:+.4f} -- so it is NOT used")
    lnx = np.array([r["lnx"] for r in bd.lo_rows])
    check("LoCuSS primary radius axis has zero spread",
          float(np.std(lnx)) < 1e-12,
          f"sd(ln r/R500_cat) = {np.std(lnx):.3e}  -> no leverage on beta")


def t_sl_internal_slope_is_artefact():
    """[caught a bug] The strong-lens internal radial slope is POSITIVE.

    S = 1/kappa_bar(theta) and ln(r/R500) = ln(theta D_l/R500) share theta
    with d ln x/d ln theta = 1 exactly, so a system that is not on the
    cluster's critical curve moves both up together.  The measured internal
    slope comes out positive, the opposite sign to every other probe -- which
    is why the primary fit aggregates the sample to one point per cluster.
    """
    bd = _bundle()
    rows = bd.sl_rows_full
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cid"]].append(r)
    bs, ws = [], []
    for c, v in by.items():
        if len(v) < 5:
            continue
        lx = np.array([q["lnx"] for q in v])
        if lx.max() - lx.min() < 0.7:
            continue
        ly = np.array([q["lnS"] for q in v])
        co = np.polyfit(lx, ly, 1)
        sd = float(np.std(ly - np.polyval(co, lx), ddof=2))
        se = sd / math.sqrt(np.sum((lx - lx.mean()) ** 2))
        bs.append(co[0]); ws.append(1.0 / se ** 2)
    b = float(np.sum(np.array(ws) * np.array(bs)) / np.sum(ws))
    check("SL internal radial slope is POSITIVE (the artefact)", b > 0.0,
          f"combined beta = {b:+.4f} on {len(bs)} clusters -- so the sample "
          f"is aggregated to cluster level")


def t_responsiveness_nonzero():
    """dS/dtheta != 0 for every headline parameter (the Run L check)."""
    bd = _bundle()
    J = F.Joint(bd, fixed_scatter=(0.25, 0.2, 0.2))
    S0, dS0 = J.project("H0", None)
    base = float(bd.F.chi2(S0, dS0, 1.0))
    got = []
    for beta in (-0.4, 0.0, 0.4):
        S, dS = J.project("H_R", np.array([beta]))
        got.append(float(bd.F.chi2(S, dS, 1.0)))
    spread = max(got) - min(got)
    check("eFEDS chi2 responds to beta", spread > 10.0,
          f"chi2 spread {spread:.2f} over beta in [-0.4, 0.4], base {base:.1f}")
    got = []
    for a in (-0.4, 0.0, 0.4):
        amp = np.exp(a * bd.ef_x["lnM"])
        got.append(float(bd.F.chi2(S0, dS0, amp)))
    check("eFEDS chi2 responds to alpha", max(got) - min(got) > 10.0,
          f"chi2 spread {max(got) - min(got):.2f} over alpha in [-0.4, 0.4]")


def t_block_covariance():
    """The Sherman-Morrison block chi2 equals the brute-force form."""
    rng = np.random.default_rng(7)
    n = 9
    res = rng.normal(size=n)
    d = rng.uniform(0.05, 0.3, size=n)
    blocks = [np.arange(0, 4), np.arange(4, 9)]
    sc2 = 0.07
    fast = F.Joint._block_chi2(res, d, sc2, blocks)
    slow = 0.0
    for b in blocks:
        Cm = np.diag(d[b]) + sc2
        slow += float(res[b] @ np.linalg.solve(Cm, res[b]))
        slow += float(np.log(np.linalg.det(Cm)))
    check("block chi2 == brute force", abs(fast - slow) < 1e-9,
          f"{fast:.9f} vs {slow:.9f}")


def t_efeds_reproduces_closure():
    """The rebuilt chain reproduces the closure lane's RAR no-slip chi2."""
    bd = _bundle()
    S, dS = K.project_slip(bd.syss, bd.obs, bd.ef_idx, lambda sm: 1.0)
    ch = float(bd.F.chi2(S, dS, 1.0))
    check("eFEDS chi2 matches closure lane 3588.4", abs(ch - 3588.4) < 0.5,
          f"chi2 = {ch:.2f}")


def t_locuss_reproduces_runK():
    """LoCuSS S reproduces Run K's E median 1.62 within the profile change."""
    bd = _bundle()
    S = np.array([r["S"] for r in bd.lo_rows])
    med = float(np.median(S))
    check("LoCuSS median S near Run K's 1.62", 1.4 < med < 1.9,
          f"median S = {med:.3f} on {len(S)} clusters "
          f"(Run K: 1.62 on 40, published M_gas not ACCEPT)")


def t_counts():
    """Row/column counts asserted on every ingest."""
    bd = _bundle()
    check("eFEDS 496 systems / 3365 points",
          len(bd.ef) == 496 and len(bd.F.gt) == 3365,
          f"{len(bd.ef)} / {len(bd.F.gt)}")
    check("LoCuSS 27 usable of 41", len(bd.lo_rows) == 27,
          f"{len(bd.lo_rows)}")
    check("SL 4 clusters / 49 image systems",
          len(bd.sl_rows) == 4 and len(bd.sl_rows_full) == 49,
          f"{len(bd.sl_rows)} aggregated from {len(bd.sl_rows_full)}")
    tot = sum(r["n_systems"] for r in bd.sl_rows)
    check("aggregation conserves the image systems", tot == 49, f"{tot}")


def t_star_template():
    """The declared stellar template reproduces the Refsdal lane's M*(<r)."""
    r = np.array([30.0, 60.0, 100.0, 300.0]) * P.KPC
    tot = sum(B.hernquist_M(r, M * MSUN, a * P.KPC)
              for M, a in B.STAR_TEMPLATE)
    ref = np.array([86989776325.8, 111425132149.5, 162423351417.4,
                    634213604055.5]) * MSUN
    err = float(np.max(np.abs(tot / ref - 1.0)))
    check("stellar template == Refsdal lane fit", err < 1e-4,
          f"max rel err {err:.3e}")


_BD = None


def _bundle():
    global _BD
    if _BD is None:
        _BD = K.Bundle(verbose=False, r500_mode="cat")
    return _BD


if __name__ == "__main__":
    print("\nTESTS -- transition lane\n" + "-" * 60)
    for fn in (t_rar_limits, t_sis_kappa, t_r500_solver, t_star_template,
               t_block_covariance, t_counts, t_efeds_reproduces_closure,
               t_locuss_reproduces_runK, t_slip_linearity, t_gplus_nonlinear,
               t_locuss_radius_circularity, t_sl_internal_slope_is_artefact,
               t_responsiveness_nonzero):
        try:
            fn()
        except Exception as e:                                # noqa: BLE001
            check(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
    print("-" * 60)
    print(f"   {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
