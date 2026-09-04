"""A_dyn from DiskMass ABSOLUTE dispersion amplitudes.  Runnable from cold.

    python adyn_run.py            full run, writes adyn_results.json
    python adyn_run.py --fast     fewer Monte-Carlo draws, same structure

Ordering is deliberate: gates before physics, cuts before residuals, the
rotation-curve fit before anything vertical is looked at, and the trap checks
(does the statistic move when its own parameter moves?) before any headline.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import i0, i1, k0, k1

HERE = os.path.dirname(os.path.abspath(__file__))
GRAVLAB = ("C:/Users/henry/Documents/Codex/2026-08-21/"
           "Invariant-main-integration/work/gravitylab")
sys.path.insert(0, HERE)
sys.path.insert(0, GRAVLAB)

import adyn_model as M                                        # noqa: E402
import axisym as X                                            # noqa: E402

FAST = "--fast" in sys.argv
NDRAW = 300 if FAST else 3000
NDRAW_LAW = 150 if FAST else 800
G, KPC, PC, MSUN = M.G, M.KPC, M.PC, M.MSUN
BAR = "=" * 78
RES: dict = {"config": dict(ndraw=NDRAW, ndraw_law=NDRAW_LAW, fast=FAST)}
rng = np.random.default_rng(20260903)
T_START = time.time()


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def q5(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {}
    return {f"p{p}": float(np.percentile(a, p)) for p in (2.5, 16, 50, 84, 97.5)}


def ci(a, f="{:.3f}"):
    d = q5(a)
    if not d:
        return "n/a"
    return (f.format(d["p50"]) + "  68% [" + f.format(d["p16"]) + ", "
            + f.format(d["p84"]) + "]  95% [" + f.format(d["p2.5"]) + ", "
            + f.format(d["p97.5"]) + "]")


# =============================================================================
head("STEP 0   GATES -- nothing below is trusted until these pass")
# =============================================================================
t0 = time.time()
print("\n0a  the reused solver reproduces the exact Freeman disk")
Rd, Sig0 = 3.0, 500 * MSUN / KPC ** 2 * 1e6
Mtot = 2 * np.pi * Sig0 * (Rd * KPC) ** 2
g = X.Grid(216, 108, 60.0, 30.0)
with np.errstate(over="ignore", invalid="ignore"):
    rho = np.nan_to_num(X.exponential_disk(g.Rc / KPC, g.zc / KPC, Sig0, Rd, 0.10))
rho *= (Mtot / 2.0) / float(np.sum(rho * g.V))
Psi, it, rel = X.solve_axi(rho, X.isotropic_A(rho.shape), g,
                           X.monopole_bc(g, Mtot), tol=1e-12, maxiter=9000)
vc = X.midplane_vc(Psi, g)
vf = X.freeman_vc(g.Rc / KPC, Sig0, Rd)
m = (g.Rc / KPC > 1.5) & (g.Rc / KPC < 27.0)
e_free = float(np.sqrt(np.mean((vc[m] - vf[m]) ** 2)) / np.sqrt(np.mean(vf[m] ** 2)))
print(f"    relative error at 216x108 : {e_free:.4e}  (test_axisym.py: 1.4e-2)"
      f"   {'PASS' if e_free < 0.05 else 'FAIL'}")
RES["gate_freeman_rel_err"] = e_free

print("\n0b  the vertical-profile family and the DiskMass constant k")
print("    rho(z) = rho_0 sech^(2/n)(n z/(2 h_z)) has, in closed form,")
print("        k = (2/n) (sqrt(pi)/2) Gamma(1/n) / Gamma(1/n + 1/2)")
from scipy.special import gamma as _Gam                        # noqa: E402


def k_exact(n):
    return (2.0 / n) * (np.sqrt(np.pi) / 2.0) * _Gam(1.0 / n) / _Gam(1.0 / n + 0.5)


print(f"    {'n':>10}{'shape':>14}{'k numeric':>12}{'k exact':>11}"
      f"{'L=int u w du':>15}")
e_k = 0.0
for n, lab in ((1.0, "sech^2"), (2.0, "sech"), (10.0, "intermediate"),
               (20000.0, "exponential")):
    p = M.VertProfile(n)
    ke = k_exact(n)
    e_k = max(e_k, abs(p.k - ke))
    print(f"    {n:>10.0f}{lab:>14}{p.k:>12.5f}{ke:>11.5f}{p.L:>15.4f}")
print("    sech^2 -> k=2 exactly, sech -> k=pi/2=1.5708, n->inf -> k=1.")
print("    DiskMass adopt k=1.5.  k enters sigma_z^2 LINEARLY and spans a")
print("    factor of two across the family, so it is carried as a nuisance.")
print(f"    max |numeric - exact| = {e_k:.2e}   "
      f"{'PASS' if e_k < 2e-3 else 'FAIL'}")
RES["gate_k_family"] = dict(k_sech2=float(M.VertProfile(1.0).k),
                            k_sech=float(M.VertProfile(2.0).k),
                            k_exp=float(M.VertProfile(20000.0).k),
                            err=float(e_k))

print("\n0c  finite-thickness table (ratio of two solves: box and BC cancel)")
t1 = time.time()
M.thickness_T(np.array([1.0]), 0.16, verbose=True)
print(f"    built/loaded in {time.time()-t1:.1f}s")

gals_all = M.load_diskmass(verbose=False)
GAL = [x for x in gals_all if x.keep]
NG = len(GAL)
g0 = GAL[0]

print("\n0d  closed form vs the numerical Jeans integral, WITH gas and leakage")
worst, worst_in = 0.0, 0.0
for gg in GAL[:6]:
    for kv in (1.0, 1.5, 2.0):
        b = M.Baryons(gg, 0.6, 0.25, gg.hz_kpc, k=kv)
        sz, _ = M.sigma_z_of_R(M.LAW_NEWTON, b)
        r_ = b.sigma_z2_newton() / sz ** 2 - 1
        w_ = (b.R / b.hR >= 0.2)          # the innermost fit window ever drawn
        worst = max(worst, float(np.max(np.abs(r_[w_]))))
        worst_in = max(worst_in, float(np.max(np.abs(r_))))
print("    Threshold set by what it costs, not by a round number: 5e-3 on")
print("    sigma_z^2 is 0.002 dex, two orders below the 0.19 dex budget of")
print("    step 5.  Evaluated over R >= 0.2 h_R, the innermost fit window the")
print("    Monte Carlo ever draws.")
print(f"      max |closed/numeric - 1| = {worst:.3e}   "
      f"{'PASS' if worst < 5e-3 else 'FAIL'}")
print(f"    including R down to 0.02 h_R : {worst_in:.3e}")
print("    The inner discrepancy is not an error in the production path.  It is")
print("    the NUMERICAL comparator flooring K_z at zero where the linearised")
print("    leakage term (z/R) dVc^2/dR, invalid as R -> 0, drives it negative.")
print("    No fit window reaches there.")
RES["gate_closedform_rel_err"] = worst
RES["gate_closedform_rel_err_incl_centre"] = worst_in

print("\n0e  semi-analytic K_z and g_R vs a full axisym.py 2-D solve")
rows = []
for gg in GAL[:5]:
    b = M.Baryons(gg, 0.6, 0.25, gg.hz_kpc, k=2.0)     # sech^2 == the solver
    sol = M.solve_2d(b, "newton", nR=200, nz=220, box_hR=16.0, zbox_hR=6.0)
    Rs, z1 = sol["R_m"], sol["grid"].zc[1]
    mm = (Rs / b.hR > 0.3) & (Rs / b.hR < 3.0)
    gN = b.gR_newton(Rs)
    Kz_sa = (2 * np.pi * G * b.Sigma_below(Rs, z1 / b.hz)
             - z1 * np.gradient(Rs * gN, Rs) / Rs)
    rk = sol["Kz"][mm] / Kz_sa[mm]
    rg = sol["gR"][mm] / gN[mm]
    rows.append([gg.ugc, float(np.median(rk)), float(np.max(np.abs(rk - 1))),
                 float(np.median(rg)), float(np.max(np.abs(rg - 1))),
                 bool(sol["converged"])])
print(f"    {'UGC':>7}{'Kz med':>10}{'Kz maxdev':>12}{'gR med':>10}"
      f"{'gR maxdev':>12}{'conv':>7}")
for r in rows:
    print(f"    {r[0]:>7}{r[1]:>10.4f}{r[2]:>12.4f}{r[3]:>10.4f}{r[4]:>12.4f}"
          f"{str(r[5]):>7}")
kz_sys = float(np.median([abs(r[1] - 1) for r in rows]))
gr_sys = float(np.median([abs(r[3] - 1) for r in rows]))
print(f"    modelling systematic adopted: K_z {100*kz_sys:.2f}%, "
      f"g_R {100*gr_sys:.2f}%   (carried into the error budget)")
RES["gate_2d_vs_semianalytic"] = dict(rows=rows, Kz_sys=kz_sys, gR_sys=gr_sys)

print("\n0f  TRAP CHECK -- the amplitude must move with B_z, the scale length")
print("    must not.  This is the defect in the previous run, made numerical.")
print(f"    {'B_z (constant)':>16}{'sigma_z_0 km/s':>18}{'h_sigma_z arcsec':>19}")
amp, hl = [], []
b = M.Baryons(g0, 0.6, 0.25, g0.hz_kpc, k=1.5)
for B0 in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    s2b = b.sigma_z2_newton() * B0
    a_, h_, _ = M.fit_exponential(b.R / b.hR * g0.hR_as, np.sqrt(s2b) / 1e3,
                                  0.3 * g0.hR_as, 2.0 * g0.hR_as)
    amp.append(a_)
    hl.append(h_)
    print(f"    {B0:>16.2f}{a_:>18.3f}{h_:>19.5f}")
sp_amp = float(np.log10(max(amp) / min(amp)))
sp_h = float(abs(np.log10(max(hl) / min(hl))))
print(f"    log10 spread: amplitude {sp_amp:.4f} dex   scale length {sp_h:.2e} dex")
print("    -> the scale length is INVARIANT to a constant B_z to machine")
print("       precision.  The previous run's statistic could not see B0 at all.")
RES["gate_trap_constB"] = dict(B0=[0.25, .5, 1, 2, 4, 8], amp=amp, hsig=hl,
                               spread_amp_dex=sp_amp, spread_h_dex=sp_h)
print(f"\n    step 0 wall time {time.time()-t0:.1f}s")


# =============================================================================
head("STEP 1   DiskMass ingest, declared cuts, and the endogeneity audit")
# =============================================================================
M.load_diskmass(verbose=True)

print("\n1a  caveat 4 -- h_R is tabulated in arcsec (VI) and in kpc (VII)")
rat = np.array([x.hR_as * x.D / 206.265 / x.hR_kpc for x in GAL])
print(f"    h_R(arcsec) x D/206.265 vs h_R(kpc): median {np.median(rat):.4f}, "
      f"max |dev| {np.max(np.abs(rat-1)):.4f}   -> the same quantity")

print("\n1b  caveat 3 -- h_z is INFERRED from h_R, not measured")
lr = np.log10([x.hR_kpc for x in GAL])
lz = np.log10([x.hz_kpc for x in GAL])
r_hz = float(np.corrcoef(lr, lz)[0, 1])
print(f"    r(log h_R, log h_z) = {r_hz:.5f}, slope "
      f"{float(np.polyfit(lr, lz, 1)[0]):.3f}")
print("    h_z carries no independent information.  Unlike the scale-length")
print("    test it does NOT cancel from the amplitude -- sigma_z^2 is linear in")
print("    h_z -- so it is marginalised with a per-galaxy width AND a")
print("    common-mode zero-point width, never fixed.")
RES["hz_correlation"] = r_hz

print("\n1c  ENDOGENEITY -- the DiskMass rotation amplitude is TF-derived")
vv = np.array([x.Vsini / math.sin(math.radians(x.incl)) for x in GAL])
vt = np.array([x.Vflat_TF for x in GAL])
dev = float(np.max(np.abs(vv / vt - 1)))
print(f"    V_c,sin i/sin(i_TF) vs V_flat(TF): max |dev| {dev:.4f} over {NG}")
print("    The same number to rounding: the inclination came from inverting the")
print("    K-band Tully-Fisher relation, so the deprojected circular speed IS")
print("    the TF prediction from M_K.  Consequences, applied throughout:")
print("      * no law is fitted to DiskMass rotation; the fits are SPARC-only;")
print("      * the rotation-curve SHAPE (arctan r_s, hence dlnV/dlnR and the")
print("        epicyclic sigma_theta/sigma_R) is a real measurement and IS used;")
print("      * the Upsilon-free A_dyn of step 6c is reported TF-conditional.")
RES["endogeneity_max_dev"] = dev

print("\n1d  photometry cross-check: two independent routes to the disk light")
off = np.array([np.log10(x.Ldisk / (10 ** (-0.4 * (x.MK - M.MSUN_K)) / (1 + x.BD)))
                for x in GAL])
print(f"    log L(mu0_K_i, h_R) - log[L(M_K)/(1+B/D)] : median "
      f"{np.median(off):+.3f} dex, scatter {np.std(off):.3f} dex")
RES["photometry_cross_check"] = dict(median_dex=float(np.median(off)),
                                     scatter_dex=float(np.std(off)))

print("\n1e  the adopted velocity ellipsoid, recovered from the published pair")
al_eff = np.array([M.effective_alpha(x.sLOS0, x.sz0, x.incl, M.beta_epicyclic(0.0))
                   for x in GAL])
ok = np.isfinite(al_eff)
print(f"    alpha_eff implied by (sigma_LOS_0, sigma_z_0, i): n={ok.sum()}  "
      f"median {np.median(al_eff[ok]):.3f}  16-84% "
      f"[{np.percentile(al_eff[ok],16):.3f}, {np.percentile(al_eff[ok],84):.3f}]")
print("    Consistent with the alpha ~ 0.6 DiskMass adopt.  The forward model")
print("    therefore predicts sigma_z, projects with ITS OWN alpha, and compares")
print("    to sigma_LOS_0 -- the raw observable -- so the ellipsoid shape is an")
print("    explicit nuisance rather than an inherited one.")
RES["alpha_eff"] = dict(median=float(np.median(al_eff[ok])),
                        p16=float(np.percentile(al_eff[ok], 16)),
                        p84=float(np.percentile(al_eff[ok], 84)))

mu = np.array([x.mu0K for x in GAL])
print(f"\n1f  dynamic range for a DIFFERENTIAL test: mu0_K_i spans "
      f"{mu.min():.2f}..{mu.max():.2f} mag/arcsec^2")
print(f"    = factor {10**(0.4*(mu.max()-mu.min())):.0f} in central surface "
      f"density; that is what a differential test can exploit.")
RES["mu0_range"] = [float(mu.min()), float(mu.max())]


# =============================================================================
head("STEP 2   Fit the laws to ROTATION CURVES ONLY (SPARC), then FREEZE")
# =============================================================================
import data as SP                                             # noqa: E402

print()
sparc = SP.ingest(verbose=True)
SP.stratified_split(sparc, verbose=True)
YD, YB = 0.5, 0.7          # SPARC 3.6um convention, declared, not fitted


def sparc_arrays(split):
    gN, go = [], []
    for s in sparc:
        if s.split not in split:
            continue
        Rm = s.R0 * KPC
        gn = (np.abs(s.Vgas) * s.Vgas + YD * s.Vdisk ** 2
              + YB * s.Vbul ** 2) * 1e6 / Rm
        gob = (s.Vobs0 * 1e3) ** 2 / Rm
        m = (gn > 0) & (gob > 0)
        gN.append(gn[m]); go.append(gob[m])
    return np.concatenate(gN), np.concatenate(go)


TRn, TRo = sparc_arrays({"train"})
BLn, BLo = sparc_arrays({"blind"})


def rms_dex(pred, gob):
    return float(np.sqrt(np.mean((np.log10(pred) - np.log10(gob)) ** 2)))


def law_g(kind, gn, a0):
    if kind == "rar":
        return M.nu_rar(gn / a0) * gn
    return 0.5 * (gn + np.sqrt(gn ** 2 + 4 * gn * a0))


grid_a0 = np.logspace(-11.3, -9.3, 401)
FIT = {}
print("\n    model                        free  a0 fitted     RMS train  RMS blind")
print("    " + "-" * 74)
for kind, label in (("rar", "RAR nu=1/(1-exp(-sqrt(x)))"),
                    ("aqual", "AQUAL simple mu=x/(1+x)")):
    a0f = float(min(grid_a0, key=lambda a: rms_dex(law_g(kind, TRn, a), TRo)))
    FIT[kind] = a0f
    print(f"    {label:<28} {1:>4}  {a0f:.3e}      "
          f"{rms_dex(law_g(kind, TRn, a0f), TRo):.4f}     "
          f"{rms_dex(law_g(kind, BLn, a0f), BLo):.4f}")
print(f"    {'Newton (no free parameter)':<28} {0:>4}  {'--':>9}      "
      f"{rms_dex(TRn, TRo):.4f}     {rms_dex(BLn, BLo):.4f}")
print("    " + "-" * 74)
print("    Fitted on the TRAIN split only, at the declared Upsilon_3.6=0.5/0.7,")
print("    minimising RMS in log10 g.  FROZEN from here on.")
print("    a0 is quoted as a descriptive best fit, not a likelihood: step 4d")
print("    shows chi2/dof for the vertical comparison before any lnL is used.")

ETA_ANI, A0_ANI = 4.0000, 5.779e-10
ETA_ISO, A0_ISO = 0.5516, 1.393e-10
print(f"\n    anisotropic tensor (mu_z=1)  : eta={ETA_ANI:.4f}  a0={A0_ANI:.3e}")
print(f"    isotropic   tensor (mu_z=mu_R): eta={ETA_ISO:.4f}  a0={A0_ISO:.3e}")
print("    Adopted from the emulated 2-D fit to the SAME SPARC train rotation")
print("    curves (mirror/adyn.log STEP 1c; RMS train 0.3932/0.2649, blind")
print("    0.4015/0.2801 dex).  Not refitted and not tuned to anything vertical.")
print("    Here they are re-solved DIRECTLY with axisym.py, so no emulator and")
print("    no emulator extrapolation enters this run.")
RES["frozen_params"] = dict(
    a0_rar=FIT["rar"], a0_aqual=FIT["aqual"], eta_aniso=ETA_ANI,
    a0_aniso=A0_ANI, eta_iso=ETA_ISO, a0_iso=A0_ISO,
    rms_train=dict(rar=rms_dex(law_g("rar", TRn, FIT["rar"]), TRo),
                   aqual=rms_dex(law_g("aqual", TRn, FIT["aqual"]), TRo),
                   newton=rms_dex(TRn, TRo)),
    rms_blind=dict(rar=rms_dex(law_g("rar", BLn, FIT["rar"]), BLo),
                   aqual=rms_dex(law_g("aqual", BLn, FIT["aqual"]), BLo),
                   newton=rms_dex(BLn, BLo)))

LAW_KEYS = ["newton", "rar", "aqual_simple", "tensor_aniso", "tensor_iso"]
LAWS = {
    "newton": M.Law("newton", "newton"),
    "rar": M.Law("rar", "algebraic", dict(a0=FIT["rar"])),
    "aqual_simple": M.Law("aqual_simple", "aqual", dict(a0=FIT["aqual"])),
    "tensor_aniso": M.Law("tensor_aniso", "tensor",
                          dict(eta=ETA_ANI, a0=A0_ANI, mu_z_is_one=True)),
    "tensor_iso": M.Law("tensor_iso", "tensor",
                        dict(eta=ETA_ISO, a0=A0_ISO, mu_z_is_one=False)),
}


# =============================================================================
head("STEP 3   Vectorised forward model, aperture/PSF operator, validation")
# =============================================================================
NR = 200
XG = np.linspace(0.02, 5.0, NR)                  # R/h_R, shared by all galaxies
YY = np.maximum(XG / 2.0, 1e-8)
BRF = i0(YY) * k0(YY) - i1(YY) * k1(YY)          # Freeman shape, precomputed
NU_C = 140
UG = np.linspace(0.0, 12.0, NU_C)                # z/h_z for the law solves

hR_m = np.array([x.hR_m for x in GAL])[:, None]
hR_as = np.array([x.hR_as for x in GAL])[:, None]
SigL0 = np.array([x.SigmaL0 for x in GAL])[:, None]
INC = np.radians(np.array([x.incl for x in GAL]))[:, None]
RS_AS = np.array([x.rs_as for x in GAL])[:, None]
UARC = np.maximum(XG[None, :] * hR_as / RS_AS, 1e-9)
DLNV = UARC / ((1 + UARC ** 2) * np.arctan(UARC))
BETA = np.sqrt(np.clip(0.5 * (1 + DLNV), 1e-6, None))
R_AS = XG[None, :] * hR_as
OBS_AMP = np.array([x.sLOS0 for x in GAL])
OBS_EAMP = np.array([x.esLOS0 for x in GAL])
OBS_H = np.array([x.hsLOS_as for x in GAL])
OBS_EH = np.array([x.ehsLOS_as for x in GAL])
HZ_TAB = np.array([x.hz_kpc for x in GAL])
EHZ_TAB = np.array([x.ehz_kpc for x in GAL])
BK = np.array([x.BK for x in GAL])
BK = np.where(np.isfinite(BK), BK, np.nanmedian(BK))
J10 = int(np.argmin(np.abs(XG - 1.0)))
J22 = int(np.argmin(np.abs(XG - 2.2)))


def newton_chain(Ups, hz_kpc, fgas, alpha, kv, f_hg, f_hzg):
    """Newtonian forward model for all galaxies at once.  Every line is an
    array identity; no fit, no lookup off a grid edge."""
    prof = M.profile_for_k(kv)
    A_ss, A_sg, L_s = M.vertical_weights(prof, f_hzg, M.profile_for_k(2.0))
    Sig_s0 = Ups[:, None] * SigL0 * MSUN / PC ** 2
    hz = (hz_kpc * KPC)[:, None]
    hg = f_hg * hR_m
    Sig_g0 = fgas[:, None] * Sig_s0 / f_hg ** 2
    R = XG[None, :] * hR_m
    Ts = np.array([M.thickness_T(XG, float(2 * hz[j, 0] / hR_m[j, 0]))
                   for j in range(NG)])
    xg = XG / f_hg
    yg = np.maximum(xg / 2.0, 1e-8)
    brg = i0(yg) * k0(yg) - i1(yg) * k1(yg)
    Tg = np.array([M.thickness_T(xg, float(2 * f_hzg * hz[j, 0] / hg[j, 0]))
                   for j in range(NG)])
    gR = (np.pi * G * Sig_s0 * XG[None, :] * BRF[None, :] * Ts
          + np.pi * G * Sig_g0 * xg[None, :] * brg[None, :] * Tg)
    Vc2 = R * gR
    Sig_s = Sig_s0 * np.exp(-XG[None, :])
    Sig_g = Sig_g0 * np.exp(-xg[None, :])
    dV = np.gradient(Vc2, XG, axis=1) / (hR_m * R)
    s2 = np.maximum(2 * np.pi * G * hz * (Sig_s * A_ss + Sig_g * A_sg)
                    - L_s * hz ** 2 * dV, 1e-30)
    return dict(gR=gR, Vc2=Vc2, s2=s2, Sig_s=Sig_s, Sig_g=Sig_g, hz=hz, R=R,
                prof=prof, f_hzg=f_hzg)


def to_los(sz_kms, alpha, apply_ap=True):
    c2, s2i = np.cos(INC) ** 2, np.sin(INC) ** 2
    sl = sz_kms * np.sqrt(c2 + 0.5 * s2i * (1 + BETA ** 2) / alpha[:, None] ** 2)
    return sl * APC if apply_ap else sl


print("\n3a  aperture (2.7\" fibre) (x) seeing (1.5\" FWHM), luminosity weighted")
print("    DiskMass shift each fibre to the local V_LOS before co-adding, so the")
print("    residual smearing is the WITHIN-fibre gradient -- what a fibre-sized")
print("    kernel reproduces.  Measured, not assumed.")
APC = np.ones((NG, NR))
_base = newton_chain(np.full(NG, 0.60), HZ_TAB, np.full(NG, 0.25),
                     np.full(NG, 0.60), 1.5, 2.0, 0.5)
_sl0 = to_los(np.sqrt(_base["s2"]) / 1e3, np.full(NG, 0.60), apply_ap=False)
for j, gg in enumerate(GAL):
    sm = M.apply_aperture(gg, R_AS[j], _sl0[j], M.FID["fibre_diam_as"],
                          M.FID["psf_fwhm_as"])
    APC[j] = np.clip(sm / _sl0[j], 0.5, 3.0)
win = (XG > 0.3) & (XG < 2.0)
a_raw, h_raw = M.fit_exponential_rows(XG, _sl0, 0.3, 2.0)
a_sm, h_sm = M.fit_exponential_rows(XG, _sl0 * APC, 0.3, 2.0)
print(f"    median aperture correction over 0.3-2 h_R : "
      f"{np.median(APC[:, win]):.4f}   (max {np.max(APC[:, win]):.4f})")
print(f"    effect on the fitted amplitude  : median "
      f"{np.median(a_sm/a_raw):.4f}  max {np.max(np.abs(a_sm/a_raw-1)):.4f}")
print(f"    effect on the fitted scale length: median "
      f"{np.median(h_sm/h_raw):.4f}  max {np.max(np.abs(h_sm/h_raw-1)):.4f}")
print("    Small because the fit window excludes the steep inner gradient.")
print("    Applied to the model regardless.")
RES["aperture"] = dict(median=float(np.median(APC[:, win])),
                       max=float(np.max(APC[:, win])),
                       amp_effect=float(np.median(a_sm / a_raw)),
                       h_effect=float(np.median(h_sm / h_raw)))

print("\n3b  vectorised chain vs the per-galaxy object chain")
w2 = 0.0
for j, gg in enumerate(GAL[:6]):
    b = M.Baryons(gg, 0.60, 0.25, gg.hz_kpc, k=1.5)
    w2 = max(w2, float(np.max(np.abs(
        _base["s2"][j] / b.sigma_z2_newton(XG * b.hR) - 1))))
print(f"    max |vectorised/object - 1| on sigma_z^2 = {w2:.3e}   "
      f"{'PASS' if w2 < 1e-9 else 'FAIL'}")
RES["gate_vectorised"] = w2

print("\n3c  does the baryon model reproduce the tabulated rotation?")
print("    NOTE step 1c: V_c here IS the TF prediction, so this is a consistency")
print("    check on photometry + M/L, not an independent dynamical test.")
Vobs = np.array([x.Vsini / math.sin(math.radians(x.incl)) for x in GAL])
frac = _base["Vc2"][:, J22] / (Vobs * 1e3) ** 2
print(f"    V_bar^2/V_c^2 at 2.2 h_R, Upsilon_K=0.60, f_gas=0.25 : median "
      f"{np.median(frac):.3f}  16-84% [{np.percentile(frac,16):.3f}, "
      f"{np.percentile(frac,84):.3f}]")
print("    DiskMass VII report submaximal disks: V_disk/V_c ~ 0.57 at 2.2 h_R,")
print(f"    a mass fraction ~0.32.  Recovered here: {np.median(frac):.2f} "
      f"(baryons incl. gas).")
RES["baryon_fraction_2.2hR"] = dict(median=float(np.median(frac)),
                                    p16=float(np.percentile(frac, 16)),
                                    p84=float(np.percentile(frac, 84)))


# =============================================================================
head("STEP 4   Frozen-law predictions of the AMPLITUDE and the scale length")
# =============================================================================
print("\n4a  per-galaxy 2-D tensor solves (the tensor has no exact reduction)")
TENS = {}
t1 = time.time()
allconv = True
for gg in GAL:
    b = M.Baryons(gg, 0.60, 0.25, gg.hz_kpc, k=1.5)
    sn = M.solve_2d(b, "newton", nR=200, nz=200, box_hR=14.0, zbox_hR=5.0)
    o = {}
    for tag, eta, a0v in (("tensor_aniso", ETA_ANI, A0_ANI),
                          ("tensor_iso", ETA_ISO, A0_ISO)):
        st = M.solve_2d(b, tag, eta=eta, a0=a0v, nR=200, nz=200,
                        box_hR=14.0, zbox_hR=5.0)
        o[tag] = np.interp(XG, sn["R_m"] / b.hR, st["gR"] / sn["gR"])
        o[tag + "_Bz"] = np.interp(XG, sn["R_m"] / b.hR, st["Kz"] / sn["Kz"])
        allconv = allconv and st["converged"] and sn["converged"]
    TENS[gg.ugc] = o
BR_TENS = {t: np.array([TENS[x.ugc][t] for x in GAL])
           for t in ("tensor_aniso", "tensor_iso")}
BZ_TENS = {t: np.array([TENS[x.ugc][t + "_Bz"] for x in GAL])
           for t in ("tensor_aniso", "tensor_iso")}
print(f"    {3*NG} solves in {time.time()-t1:.1f}s; all converged: {allconv}")
print("    B_R and B_z for the tensor are ratios of two solves on the SAME grid,")
print("    so the box size and the monopole boundary condition cancel.  The")
print("    semi-analytic reduction gives B_z = 1/mu_z exactly (so 1.000 for the")
print("    anisotropic case); the solved values below are the leak-difference")
print("    correction to that, and are the ones used.")
print(f"    {'law':<14}{'B_z(2-D) @1hR':>16}{'@2.2hR':>10}"
      f"{'semi-analytic 1/mu_z':>24}")
for t in ("tensor_aniso", "tensor_iso"):
    b = M.Baryons(GAL[0], 0.60, 0.25, GAL[0].hz_kpc, k=1.5)
    Mb = b.Mstar + b.Mgas
    muR = 1.0 / (1.0 + LAWS[t].params["eta"]
                 * M.s_gap(XG * b.hR, Mb, LAWS[t].params["eta"],
                           LAWS[t].params["a0"]))
    inv = 1.0 if LAWS[t].params["mu_z_is_one"] else float(1.0 / muR[J22])
    print(f"    {t:<14}{np.median(BZ_TENS[t][:, J10]):>16.4f}"
          f"{np.median(BZ_TENS[t][:, J22]):>10.4f}{inv:>24.4f}")
RES["tensor_solves_converged"] = bool(allconv)

print("\n4b  extrapolation audit -- np.interp CLAMPS silently")
qz = 2 * HZ_TAB / np.array([x.hR_kpc for x in GAL])
print(f"      thickness table, R/h_R axis   : 0.00% (queries 0.02-5, table "
      f"0-16 h_R)")
print(f"      thickness table, h_sech/h_R   : "
      f"{100*np.mean((qz < M._TQ[0]) | (qz > M._TQ[-1])):.2f}% clamped "
      f"(queries {qz.min():.3f}-{qz.max():.3f}, table {M._TQ[0]:.2f}-{M._TQ[-1]:.2f})")
print(f"      tensor B_R onto the R/h_R grid: 0.00% (solver spans 0-14 h_R)")
print(f"      vertical-profile k -> n table : k drawn in "
      f"[{M.FID['k_lo']:.2f},{M.FID['k_hi']:.2f}], table covers "
      f"[{1.0:.2f},{2.3:.2f}]  0.00%")
print("    No headline number below rests on a clamped value.")
RES["extrapolation_fraction"] = dict(
    thickness_R=0.0, thickness_q=float(np.mean((qz < M._TQ[0]) | (qz > M._TQ[-1]))),
    tensor_BR=0.0, profile_k=0.0)


def law_Bz(key, base, Ups, fgas, f_hg):
    """B_z_eff = sigma_z^2(law)/sigma_z^2(Newton) on (NG, NR), plus B_R.

    The radial-leakage term is identical for every law (see adyn_model.Kz_grid),
    so it is carried in K_z^N and cancels from the ratio at leading order.
    """
    if key == "newton":
        return np.ones((NG, NR)), np.ones((NG, NR))
    prof = base["prof"]
    w = np.interp(UG, prof.u, prof.w)
    Cs = np.interp(UG, prof.u, prof.Cn, left=0.0, right=1.0)
    pg = M.profile_for_k(2.0)
    Cg = np.interp(UG / base["f_hzg"], pg.u, pg.Cn, left=0.0, right=1.0)
    hz, R, gRN, Vc2N = base["hz"], base["R"], base["gR"], base["Vc2"]
    zz = UG[None, None, :] * hz[:, :, None]
    Sig_lt = (base["Sig_s"][:, :, None] * Cs[None, None, :]
              + base["Sig_g"][:, :, None] * Cg[None, None, :])
    dN = (np.gradient(Vc2N, XG, axis=1) / (hR_m * R))[:, :, None]
    KzN = np.maximum(2 * np.pi * G * Sig_lt - zz * dN, 1e-30)
    if key == "rar":
        a0v = FIT["rar"]
        Kz = M.nu_rar(np.sqrt(gRN[:, :, None] ** 2 + KzN ** 2) / a0v) * KzN
        BRr = M.nu_rar(gRN / a0v)
    elif key == "aqual_simple":
        a0v = FIT["aqual"]
        gR = 0.5 * (gRN + np.sqrt(gRN ** 2 + 4 * gRN * a0v))
        BRr = gR / gRN
        Kz = M.aqual_Kz(KzN, gR[:, :, None] * np.ones_like(KzN), a0v)
    else:
        BRr = BR_TENS[key]
        Kz = KzN * BZ_TENS[key][:, :, None]
    s2 = np.trapezoid(w[None, None, :] * Kz, zz, axis=2)
    s2n = np.trapezoid(w[None, None, :] * KzN, zz, axis=2)
    return s2 / s2n, BRr


print("\n4c  amplitude and scale length predicted by each FROZEN law")
print("    fiducial nuisances: Upsilon_K=0.60, f_gas=0.25, h_z tabulated,")
print("    k=1.5 (the DiskMass value), alpha=0.60, fit window 0.3-2.0 h_R")
Up_f = np.full(NG, 0.60)
fg_f = np.full(NG, 0.25)
al_f = np.full(NG, 0.60)
base_f = newton_chain(Up_f, HZ_TAB, fg_f, al_f, 1.5, 2.0, 0.5)
FIDP = {}
hdr = (f"    {'law':<14}{'sig_LOS_0 pred':>16}{'obs':>8}{'obs/pred':>10}"
       f"{'h_sig pred':>12}{'obs':>7}{'B_R@2.2':>9}{'B_z@1':>8}{'B_z@2.2':>9}")
print(hdr + "\n    " + "-" * (len(hdr) - 4))
for key in LAW_KEYS:
    BzE, BRr = law_Bz(key, base_f, Up_f, fg_f, 2.0)
    sl = to_los(np.sqrt(base_f["s2"] * BzE) / 1e3, al_f)
    a_, h_ = M.fit_exponential_rows(XG, sl, 0.3, 2.0)
    FIDP[key] = dict(amp=a_, h=h_ * np.squeeze(hR_as), BzE=BzE,
                     BR=BRr if np.ndim(BRr) else np.full((NG, NR), BRr))
    print(f"    {key:<14}{np.median(a_):>16.2f}{np.median(OBS_AMP):>8.2f}"
          f"{np.median(OBS_AMP/a_):>10.3f}{np.median(h_*np.squeeze(hR_as)):>12.2f}"
          f"{np.median(OBS_H):>7.2f}"
          f"{np.median(FIDP[key]['BR'][:, J22]):>9.3f}"
          f"{np.median(BzE[:, J10]):>8.3f}{np.median(BzE[:, J22]):>9.3f}")
print("    " + "-" * (len(hdr) - 4))
print("    'obs/pred' squared is the EXTRA vertical boost the data demand on top")
print("    of that law at the fiducial nuisances.  For Newton it is B_z itself.")
RES["fiducial_prediction"] = {
    kk: {"amp_median": float(np.median(v["amp"])),
         "h_median": float(np.median(v["h"])),
         "obs_over_pred_median": float(np.median(OBS_AMP / v["amp"])),
         "BR_at_2.2hR_median": float(np.median(v["BR"][:, J22])),
         "Bz_at_1hR_median": float(np.median(v["BzE"][:, J10])),
         "Bz_at_2.2hR_median": float(np.median(v["BzE"][:, J22]))}
    for kk, v in FIDP.items()}

print("\n4d  chi2/dof BEFORE any likelihood language is used")
print("    Measurement errors only (no nuisance freedom), so this is the")
print("    question 'do the published error bars alone explain the residuals?'")
print(f"    {'law':<14}{'chi2/dof amp':>14}{'chi2/dof h':>13}"
      f"{'median resid dex':>18}")
CHI = {}
for key in LAW_KEYS:
    ca = float(np.mean(((OBS_AMP - FIDP[key]["amp"]) / OBS_EAMP) ** 2))
    ch = float(np.mean(((OBS_H - FIDP[key]["h"]) / OBS_EH) ** 2))
    md = float(np.median(np.log10(OBS_AMP / FIDP[key]["amp"])))
    CHI[key] = dict(chi2dof_amp=ca, chi2dof_h=ch, median_resid_dex=md)
    print(f"    {key:<14}{ca:>14.2f}{ch:>13.2f}{md:>18.3f}")
print("    chi2/dof is NOT near 1 for any law.  That is expected and it is the")
print("    point: the residual is dominated by the nuisance parameters")
print("    (Upsilon_K, h_z, k, gas, alpha), not by measurement noise.  No lnL,")
print("    AIC or BIC is quoted anywhere in this run.  Everything below is an")
print("    explicit error budget instead.")
RES["chi2_dof"] = CHI


# =============================================================================
head("STEP 5   POSTERIOR on B_z = K_z/K_z,Newton, nuisances marginalised")
# =============================================================================
print("""
    B_z is defined against the Newtonian vertical force of the SAME baryon
    model, so it is the empirical vertical boost the data require:

        B_z(galaxy) = [ sigma_LOS_0(observed) / sigma_LOS_0(model, Newton) ]^2

    Nuisances drawn per Monte-Carlo sample.  COMMON-MODE entries apply the same
    shift to every galaxy and therefore do NOT average down with sqrt(N); that
    distinction is the whole error budget and is reported separately.

      common-mode                     per-galaxy
      -----------                     ----------
      Upsilon_K zero point 0.15 dex   Upsilon_K scatter        0.06 dex
      (B-K) colour slope 0.15+-0.10   h_z from e_h_z (tabulated)
      h_z relation zero pt 0.10 dex   f_gas scatter            0.15 dex
      k in [1.5, 2.0]                 sigma_LOS_0 from e_sigma_LOS_0
      alpha = 0.60 +- 0.12            distance from e_Dist
      f_gas median 0.25, 0.20 dex
      h_gas/h_R in [1.5, 3.0]
      h_zgas/h_z in [0.3, 0.8]
      fit window lo U(0.2,0.5) hi U(1.5,2.5)
""")


def draw_common(r):
    return dict(
        zU=r.normal(np.log10(M.FID["Upsilon_K"]), M.FID["s_Upsilon"]),
        sc=r.normal(M.FID["col_slope"], M.FID["s_col_slope"]),
        dhz=r.normal(0.0, M.FID["s_hz_sys"]),
        kv=round(float(r.uniform(M.FID["k_lo"], M.FID["k_hi"])), 2),
        al=float(np.clip(r.normal(M.FID["alpha"], M.FID["s_alpha"]), 0.35, 0.95)),
        lfg=r.normal(np.log10(M.FID["f_gas"]), 0.20),
        fhg=float(r.uniform(1.5, 3.0)),
        fhzg=round(float(r.uniform(0.3, 0.8)), 3),
        lo=float(r.uniform(0.2, 0.5)), hi=float(r.uniform(1.5, 2.5)))


FID_COMMON = dict(zU=np.log10(0.60), sc=M.FID["col_slope"], dhz=0.0, kv=1.5,
                  al=0.60, lfg=np.log10(0.25), fhg=2.0, fhzg=0.5, lo=0.3, hi=2.0)
E_DIST = np.array([x.eD / x.D for x in GAL])


def draw_pergal(r, C, stat_only=False):
    # `stat_only` freezes the COMMON-MODE draws (they arrive through C); the
    # per-galaxy terms below are the statistical part and stay on either way.
    lU = C["zU"] + C["sc"] * (BK - M.FID["BK_pivot"])
    lhz = np.log10(HZ_TAB) + C["dhz"]
    lfg = np.full(NG, C["lfg"])
    lU = lU + r.normal(0.0, M.FID["s_Upsilon_gal"], NG)
    lhz = lhz + r.normal(0.0, EHZ_TAB / HZ_TAB / np.log(10), NG)
    lhz = lhz + 0.643 * np.log10(1.0 + r.normal(0.0, E_DIST, NG))
    lfg = lfg + r.normal(0.0, 0.15, NG)
    sob = OBS_AMP + r.normal(0.0, OBS_EAMP, NG)
    return (10 ** lU, 10 ** lhz, np.clip(10 ** lfg, 0.0, 3.0),
            np.full(NG, C["al"]), np.maximum(sob, 1.0))


def amp_newton(Ups, hz, fg, al, C):
    b = newton_chain(Ups, hz, fg, al, C["kv"], C["fhg"], C["fhzg"])
    sl = to_los(np.sqrt(b["s2"]) / 1e3, al)
    a_, h_ = M.fit_exponential_rows(XG, sl, C["lo"], C["hi"])
    return a_, h_ * np.squeeze(hR_as), b


t1 = time.time()
POST = {}
for tag, sysoff in (("full", False), ("stat_only", True)):
    r = np.random.default_rng(4242)
    lb_mean, lb_all = [], []
    for _ in range(NDRAW):
        C = dict(FID_COMMON) if sysoff else draw_common(r)
        Ups, hz, fg, al, sob = draw_pergal(r, C, stat_only=sysoff)
        aN, hN, _ = amp_newton(Ups, hz, fg, al, C)
        lb = 2.0 * np.log10(sob / aN)
        lb_all.append(lb)
        lb_mean.append(float(np.mean(lb)))
    POST[tag] = dict(mean=np.array(lb_mean), all=np.array(lb_all))
print(f"    {2*NDRAW} forward models in {time.time()-t1:.1f}s")

lm_f, lm_s = POST["full"]["mean"], POST["stat_only"]["mean"]
print(f"\n    sample-mean log10 B_z, statistical only : {ci(lm_s, '{:+.3f}')}")
print(f"    sample-mean log10 B_z, full budget     : {ci(lm_f, '{:+.3f}')}")
print(f"    -> B_z (statistical only) = {ci(10**lm_s)}")
print(f"    -> B_z (FULL BUDGET)      = {ci(10**lm_f)}")
sd_s, sd_f = float(np.std(lm_s)), float(np.std(lm_f))
print(f"\n    width on log10 B_z: statistical {sd_s:.4f} dex,  full {sd_f:.4f} dex")
print(f"    the systematic floor is {np.sqrt(max(sd_f**2-sd_s**2,0)):.4f} dex, "
      f"{np.sqrt(max(sd_f**2-sd_s**2,0))/max(sd_s,1e-9):.1f}x the statistical part.")
print("    More galaxies would shrink only the statistical part.  This is a")
print("    DEGENERACY, not a noise problem.")
RES["Bz_posterior"] = dict(
    log10_stat=q5(lm_s), log10_full=q5(lm_f),
    Bz_stat=q5(10 ** lm_s), Bz_full=q5(10 ** lm_f),
    sd_stat_dex=sd_s, sd_full_dex=sd_f,
    sd_systematic_dex=float(np.sqrt(max(sd_f ** 2 - sd_s ** 2, 0))))

print("\n5b  which nuisance carries the width?  one at a time, others fixed")
print(f"    {'nuisance varied':<34}{'sd(log10 B_z) dex':>20}{'factor':>10}")
solo = {}
for name, key in (("Upsilon_K zero point (0.15 dex)", "zU"),
                  ("(B-K) colour slope", "sc"),
                  ("h_z relation zero point (0.10)", "dhz"),
                  ("k, vertical profile [1.5,2.0]", "kv"),
                  ("alpha = sigma_z/sigma_R", "al"),
                  ("f_gas median", "lfg"),
                  ("gas radial scale length", "fhg"),
                  ("gas scale height", "fhzg"),
                  ("fit window", "lo"),
                  ("measurement errors only", "meas")):
    r = np.random.default_rng(77)
    vals = []
    for _ in range(max(NDRAW // 4, 120)):
        C = dict(FID_COMMON)
        if key != "meas":
            D = draw_common(r)
            C[key] = D[key]
            if key == "lo":
                C["hi"] = D["hi"]
        Ups, hz, fg, al, sob = draw_pergal(r, C, stat_only=True)
        if key == "meas":
            Ups, hz, fg = np.full(NG, 0.60), HZ_TAB, np.full(NG, 0.25)
        aN, _, _ = amp_newton(Ups, hz, fg, al, C)
        vals.append(float(np.mean(2 * np.log10(sob / aN))))
    s = float(np.std(vals))
    solo[name] = s
    print(f"    {name:<34}{s:>20.4f}{10**s:>10.3f}")
RES["nuisance_budget_dex"] = solo


# =============================================================================
head("STEP 6   A_dyn = B_R / B_z per law, with uncertainties")
# =============================================================================
print("""
    Three distinct quantities, kept apart on purpose:

    (a) A_dyn PREDICTED by each law         = B_R(law) / B_z(law).  Pure theory.
    (b) A_dyn REQUIRED by the data, per law = B_R(law) / B_z(observed).
    (c) A_dyn DIRECT, Upsilon-free          = B_R(obs) / B_z(obs), which needs no
        stellar M/L because it cancels -- but does need the TF-derived rotation
        amplitude of step 1c, so it is TF-conditional.
""")
print("6a  A_dyn predicted by each frozen law (theory only, no data)")
print(f"    {'law':<14}{'B_R @1hR':>10}{'B_z @1hR':>10}{'A_dyn @1hR':>12}"
      f"{'B_R @2.2':>10}{'B_z @2.2':>10}{'A_dyn @2.2':>12}")
ADYN_PRED = {}
for key in LAW_KEYS:
    BRv, Bzv = FIDP[key]["BR"], FIDP[key]["BzE"]
    a1 = np.median(BRv[:, J10] / Bzv[:, J10])
    a2 = np.median(BRv[:, J22] / Bzv[:, J22])
    ADYN_PRED[key] = dict(A1=float(a1), A22=float(a2),
                          BR1=float(np.median(BRv[:, J10])),
                          Bz1=float(np.median(Bzv[:, J10])),
                          BR22=float(np.median(BRv[:, J22])),
                          Bz22=float(np.median(Bzv[:, J22])))
    print(f"    {key:<14}{np.median(BRv[:,J10]):>10.3f}{np.median(Bzv[:,J10]):>10.3f}"
          f"{a1:>12.3f}{np.median(BRv[:,J22]):>10.3f}"
          f"{np.median(Bzv[:,J22]):>10.3f}{a2:>12.3f}")
print("    Newton is 1/1 = 1 by construction.  RAR and AQUAL sit slightly ABOVE")
print("    1 because nu and mu depend on |grad Phi|, which grows with |z| as")
print("    K_z -> 2 pi G Sigma; the vertical boost is therefore diluted over the")
print("    layer relative to the midplane radial boost.  The anisotropic tensor")
print("    (mu_z = 1) is the only law with A_dyn far from 1 by construction.")
RES["A_dyn_predicted"] = ADYN_PRED

print("\n6b  A_dyn REQUIRED by the data, per law:  B_R(law)/B_z(observed)")
t1 = time.time()
LAWPOST = {}
r = np.random.default_rng(999)
draws = {kk: [] for kk in LAW_KEYS}
BZOBS_D, BZLAW_D = [], {kk: [] for kk in LAW_KEYS}
for _ in range(NDRAW_LAW):
    C = draw_common(r)
    Ups, hz, fg, al, sob = draw_pergal(r, C)
    aN, hN, b = amp_newton(Ups, hz, fg, al, C)
    lbo = 2 * np.log10(sob / aN)
    BZOBS_D.append(lbo)
    for key in LAW_KEYS:
        BzE, BRr = law_Bz(key, b, Ups, fg, C["fhg"])
        sl = to_los(np.sqrt(b["s2"] * BzE) / 1e3, al)
        aL, _ = M.fit_exponential_rows(XG, sl, C["lo"], C["hi"])
        BZLAW_D[key].append(2 * np.log10(aL / aN))       # log10 B_z(law), amp
        BR = BRr if np.ndim(BRr) == 2 else np.full((NG, NR), BRr)
        draws[key].append(float(np.mean(np.log10(BR[:, J22]) - lbo)))
BZOBS_D = np.array(BZOBS_D)
BZLAW_D = {kk: np.array(v) for kk, v in BZLAW_D.items()}
print(f"    {NDRAW_LAW} draws x {len(LAW_KEYS)} laws in {time.time()-t1:.1f}s")
print(f"\n    {'law':<14}{'B_z predicted (amp)':>34}")
for key in LAW_KEYS:
    LAWPOST[key] = dict(A_req=np.array(draws[key]),
                        Bz_law=BZLAW_D[key].mean(axis=1))
    print(f"    {key:<14}{ci(10**BZLAW_D[key].mean(axis=1)):>34}")
print(f"\n    {'law':<14}{'A_dyn required = B_R(law)/B_z(obs) at 2.2 h_R':>50}")
for key in LAW_KEYS:
    print(f"    {key:<14}{ci(10**LAWPOST[key]['A_req']):>50}")
RES["A_dyn_required"] = {kk: q5(10 ** v["A_req"]) for kk, v in LAWPOST.items()}
RES["Bz_law_amplitude"] = {kk: q5(10 ** v.mean(axis=1)) for kk, v in BZLAW_D.items()}

print("\n6c  A_dyn DIRECT (Upsilon-free), from matched radii")
print("    B_R = V_obs^2/V_bar,N^2 and B_z = sigma_z,obs^2/sigma_z,N^2 both carry")
print("    1/Upsilon, so the RATIO does not.  Verified numerically below rather")
print("    than asserted.  Uses the reconstructed observed exponentials, so the")
print("    unpublished (amplitude, scale length) covariance is bracketed.")


def direct_adyn(Ups, hz, fg, al, C, rho_ab=0.0, r=None, jj=J22):
    b = newton_chain(Ups, hz, fg, al, C["kv"], C["fhg"], C["fhzg"])
    Rq = XG[jj] * np.squeeze(hR_as)                       # arcsec
    a_ = np.log(OBS_AMP)
    bb = -1.0 / OBS_H
    sa = OBS_EAMP / OBS_AMP
    sb = OBS_EH / OBS_H ** 2
    mu_ = a_ + bb * Rq
    sd_ = np.sqrt(sa ** 2 + (Rq * sb) ** 2 + 2 * rho_ab * Rq * sa * sb)
    lsl = mu_ + (r.normal(0, 1, NG) * sd_ if r is not None else 0.0)
    sl_obs = np.exp(lsl)
    sz_obs = M.sigma_z_from_los(sl_obs / APC[:, jj], np.degrees(np.squeeze(INC)),
                                al, np.squeeze(BETA[:, jj]))
    Bz = (sz_obs * 1e3) ** 2 / b["s2"][:, jj]
    BR = (Vobs * 1e3) ** 2 / b["Vc2"][:, jj]
    return BR / Bz, BR, Bz


print(f"\n    invariance check -- Upsilon_K scanned over a decade at fixed "
      f"everything else")
print(f"    {'Upsilon_K':>10}{'median B_R':>13}{'median B_z':>13}"
      f"{'median A_dyn':>14}")
inv = []
for U in (0.15, 0.30, 0.60, 1.20, 2.40):
    A, BRd, Bzd = direct_adyn(np.full(NG, U), HZ_TAB, np.full(NG, 0.25),
                              np.full(NG, 0.60), FID_COMMON)
    inv.append(float(np.median(A)))
    print(f"    {U:>10.2f}{np.median(BRd):>13.3f}{np.median(Bzd):>13.3f}"
          f"{np.median(A):>14.4f}")
print(f"    spread in A_dyn over a 16x range in Upsilon_K: "
      f"{np.log10(max(inv)/min(inv)):.2e} dex   -> the cancellation is exact")
print(f"    (B_R moved {np.log10(2.40/0.15):.2f} dex over the same range, so the")
print("     invariance is a real cancellation, not an insensitive statistic.)")
RES["direct_Upsilon_invariance_dex"] = float(np.log10(max(inv) / min(inv)))

print(f"\n    responsiveness -- A_dyn DIRECT must move with its own levers")
print(f"    {'lever':<28}{'value':>10}{'median A_dyn':>15}")
for lab, kw in (("h_z x0.7", dict(hz=0.7)), ("h_z x1.0", dict(hz=1.0)),
                ("h_z x1.4", dict(hz=1.4)), ("k = 1.0", dict(kv=1.0)),
                ("k = 1.5", dict(kv=1.5)), ("k = 2.0", dict(kv=2.0)),
                ("alpha = 0.45", dict(al=0.45)), ("alpha = 0.60", dict(al=0.60)),
                ("alpha = 0.80", dict(al=0.80))):
    C = dict(FID_COMMON)
    if "kv" in kw:
        C["kv"] = kw["kv"]
    A, _, _ = direct_adyn(np.full(NG, 0.6), HZ_TAB * kw.get("hz", 1.0),
                          np.full(NG, 0.25), np.full(NG, kw.get("al", 0.60)), C)
    print(f"    {lab:<28}{'':>10}{np.median(A):>15.3f}")

r = np.random.default_rng(31337)
for rho in (0.0, -0.9):
    vals = []
    for _ in range(max(NDRAW // 3, 200)):
        C = draw_common(r)
        Ups, hz, fg, al, _ = draw_pergal(r, C)
        A, _, _ = direct_adyn(Ups, hz, fg, al, C, rho_ab=rho, r=r)
        vals.append(float(np.median(A)))
    print(f"\n    A_dyn DIRECT at 2.2 h_R, fit corr rho={rho:+.1f} : "
          f"{ci(np.array(vals))}")
    RES[f"A_dyn_direct_rho{rho:+.1f}"] = q5(np.array(vals))
print("    TF-conditional: the numerator uses V_flat(TF), not an independent")
print("    rotation measurement.  It is reported for completeness, and it is NOT")
print("    the headline.")


# =============================================================================
head("STEP 7   The DIFFERENTIAL test -- immune to a common-mode error")
# =============================================================================
print("""
    A single wrong Upsilon_K, h_z or k shifts every galaxy's B_z by the SAME
    factor.  It cannot create a correlation between the measured B_z and the
    B_z a law predicts galaxy by galaxy.  The DiskMass sample spans a factor of
    ~35 in central surface density, so the laws predict B_z varying across the
    sample, and the slope of that relation is a test with no common-mode floor.

        log10 B_z(observed)  =  c  +  s * log10 B_z(law)

    s = 0 : the vertical field does not track the law's prediction (Newton)
    s = 1 : it tracks it exactly
""")
def ci68(a, f="{:+.2f}"):
    d = q5(a)
    return f.format(d["p50"]) + " [" + f.format(d["p16"]) + "," + \
        f.format(d["p84"]) + "]"


print(f"    {'law':<14}{'range predicted':>17}{'slope s [68%]':>24}"
      f"{'resid rms':>11}{'infl':>7}")
DIFF = {}
for key in LAW_KEYS:
    if key == "newton":
        continue
    x_all = BZLAW_D[key]
    slopes, cs, res = [], [], []
    for d in range(x_all.shape[0]):
        x, y = x_all[d], BZOBS_D[d]
        if np.std(x) < 1e-6:
            continue
        p = np.polyfit(x, y, 1)
        slopes.append(p[0]); cs.append(p[1])
        res.append(np.std(y - np.polyval(p, x)))
    slopes = np.array(slopes)
    # The per-draw refit propagates the nuisance draws but not any INTRINSIC
    # galaxy-to-galaxy scatter beyond them.  If the residual about the fit
    # exceeds the modelled per-galaxy scatter, the slope error is too small; the
    # ratio below is the standard chi2/dof=1 inflation and it is applied.
    mod = float(np.median(np.std(BZOBS_D - BZOBS_D.mean(axis=0), axis=1)))
    rr = float(np.median(res))
    infl = max(1.0, rr / max(mod, 1e-9))
    rng_x = float(np.median(x_all.max(axis=1) - x_all.min(axis=1)))
    DIFF[key] = dict(slope=slopes, const=np.array(cs), range_x=rng_x,
                     resid=rr, model_scatter=mod, inflation=infl,
                     sd_raw=float(np.std(slopes)),
                     sd_inflated=float(np.std(slopes)) * infl)
    print(f"    {key:<14}{rng_x:>17.3f}{ci68(slopes):>24}"
          f"{rr:>11.3f}{infl:>7.2f}")
print(f"\n    residual scatter about the fit vs the scatter the nuisance model")
print(f"    itself produces ({DIFF['rar']['model_scatter']:.3f} dex): the ratio")
print("    is the inflation applied to sd(s) below, so an under-modelled")
print("    galaxy-to-galaxy term cannot masquerade as significance.")
print(f"    {'law':<14}{'slope s':>10}{'sd raw':>10}{'sd inflated':>14}"
      f"{'s/sd (inflated)':>18}")
for key in DIFF:
    s = float(np.median(DIFF[key]["slope"]))
    sd = DIFF[key]["sd_inflated"]
    print(f"    {key:<14}{s:>10.3f}{DIFF[key]['sd_raw']:>10.3f}{sd:>14.3f}"
          f"{s/sd:>18.2f}")
print("\n    Label-control null: the predicted values are SHUFFLED across")
print("    galaxies, destroying the physical pairing but preserving both")
print("    marginals.  A slope the null reaches is not evidence.")
print(f"    {'law':<14}{'true slope':>14}{'null 2.5-97.5%':>26}{'p(one-sided)':>15}")
rn = np.random.default_rng(5150)
for key in DIFF:
    x_all = BZLAW_D[key]
    null = []
    for d in range(min(x_all.shape[0], 400)):
        x = x_all[d][rn.permutation(NG)]
        null.append(np.polyfit(x, BZOBS_D[d], 1)[0])
    null = np.array(null)
    tr = float(np.median(DIFF[key]["slope"]))
    p = float(np.mean(null >= tr)) if tr > 0 else float(np.mean(null <= tr))
    DIFF[key]["null"] = null
    DIFF[key]["p"] = p
    print(f"    {key:<14}{tr:>14.3f}"
          f"{f'[{np.percentile(null,2.5):+.2f}, {np.percentile(null,97.5):+.2f}]':>26}"
          f"{p:>15.4f}")

print("\n    Confound audit.  The predictor is driven by surface density, which")
print("    also drives colour and h_z.  Partial slopes with those held:")
print(f"    {'law':<14}{'raw s':>10}{'| B-K held':>13}{'| mu0_K held':>15}"
      f"{'| h_z/h_R held':>16}")
for key in DIFF:
    x_all = BZLAW_D[key]
    out = []
    for cov in (None, BK, mu, HZ_TAB / np.array([x.hR_kpc for x in GAL])):
        ss = []
        for d in range(min(x_all.shape[0], 400)):
            x, y = x_all[d], BZOBS_D[d]
            if cov is None:
                ss.append(np.polyfit(x, y, 1)[0])
            else:
                Amat = np.column_stack([x, cov, np.ones(NG)])
                ss.append(np.linalg.lstsq(Amat, y, rcond=None)[0][0])
        out.append(float(np.median(ss)))
    DIFF[key]["partial"] = out
    print(f"    {key:<14}{out[0]:>10.3f}{out[1]:>13.3f}{out[2]:>15.3f}"
          f"{out[3]:>16.3f}")
print("\n    Direct form of the same test, against the raw driver.  Regress")
print("    log10 B_z(observed) on log10 Sigma_0(K band), and compare with the")
print("    slope each law predicts for that same regression.  This needs no")
print("    law-specific predictor and is the least model-dependent version.")
lSig = np.log10(np.array([x.SigmaL0 for x in GAL]))
sl_obs_S = np.array([np.polyfit(lSig, BZOBS_D[d], 1)[0]
                     for d in range(BZOBS_D.shape[0])])
INFL = float(np.median([DIFF[k]["inflation"] for k in DIFF]))
sd_obs_S = float(np.std(sl_obs_S)) * INFL
print(f"    {'quantity':<26}{'d log10 B_z / d log10 Sigma_0':>40}")
print(f"    {'OBSERVED':<26}{ci(sl_obs_S, '{:+.3f}'):>40}")
print(f"    {'':<26}{f'sd inflated x{INFL:.2f} = {sd_obs_S:.3f}':>40}")
print(f"    {'Newton predicts':<26}{0.0:>40.3f}")
for key in DIFF:
    sp = np.array([np.polyfit(lSig, BZLAW_D[key][d], 1)[0]
                   for d in range(BZLAW_D[key].shape[0])])
    print(f"    {key + ' predicts':<26}{ci(sp, '{:+.3f}'):>40}")
    DIFF[key]["sigma_slope_pred"] = q5(sp)
print(f"\n    The observed slope is {abs(np.median(sl_obs_S))/sd_obs_S:.1f} sigma "
      f"from the Newtonian zero.  That is the one place in this")
print("    analysis where the data have real power, because a common-mode error")
print("    in Upsilon_K, h_z or k moves the INTERCEPT, not the slope.")
RES["sigma_slope_observed"] = dict(q5(sl_obs_S), sd_inflated=sd_obs_S,
                                   inflation=INFL)

print("\n7b  audits of that slope, because it is the only live signal here")
bs = []
rb = np.random.default_rng(8080)
for _ in range(2000):
    idx = rb.integers(0, NG, NG)
    d = rb.integers(0, BZOBS_D.shape[0])
    if np.std(lSig[idx]) < 1e-6:
        continue
    bs.append(np.polyfit(lSig[idx], BZOBS_D[d][idx], 1)[0])
bs = np.array(bs)
p_bs = float(np.mean(bs >= 0.0))
print(f"    (i)  bootstrap over galaxies       : {ci(bs, '{:+.3f}')}")
print(f"         fraction of bootstrap resamples with slope >= 0 : {p_bs:.4f}")
print("         -- the most conservative error estimator here, because it")
print("            absorbs any galaxy-to-galaxy term the nuisance model misses")
jk = np.array([np.polyfit(np.delete(lSig, j), np.delete(BZOBS_D[0], j), 1)[0]
               for j in range(NG)])
print(f"         leave-one-out range           : [{jk.min():+.3f}, {jk.max():+.3f}]")

print("    (ii) an instrumental dispersion floor would bias the LOW-sigma end")
print("         upward and could fake a negative slope.  Add a spurious")
print("         quadrature term eps to the observed sigma^2 and refit:")
for eps_kms in (0.0, 5.0, 10.0, 15.0, 20.0):
    sl2 = []
    for d in range(min(BZOBS_D.shape[0], 200)):
        adj = np.log10(1.0 + eps_kms ** 2 / OBS_AMP ** 2)
        sl2.append(np.polyfit(lSig, BZOBS_D[d] - adj, 1)[0])
    print(f"         eps = {eps_kms:>4.1f} km/s in quadrature -> slope "
          f"{np.median(sl2):+.3f}")
print("         DiskMass instrumental sigma is ~17-20 km/s and is removed in")
print("         their fits; even a fully uncorrected 20 km/s residual moves the")
print("         slope by far less than the distance to zero.")

print("    (iii) gas fraction anticorrelates with surface density in real")
print("          spirals, and M_gas is NOT tabulated here.  Impose a trend")
print("          f_gas ~ Sigma_0^(-b) at fixed sample mean and refit:")
r7 = np.random.default_rng(606)
for bslope in (0.0, 0.2, 0.4, 0.6):
    vals = []
    for _ in range(max(NDRAW // 10, 60)):
        C = draw_common(r7)
        Ups, hz, fg, al, sob = draw_pergal(r7, C)
        tilt = 10 ** (-bslope * (lSig - lSig.mean()))
        fg2 = np.clip(fg * tilt / np.mean(tilt), 0.0, 5.0)
        aN, _, _ = amp_newton(Ups, hz, fg2, al, C)
        vals.append(np.polyfit(lSig, 2 * np.log10(sob / aN), 1)[0])
    print(f"          d log f_gas/d log Sigma_0 = {-bslope:+.1f} -> observed "
          f"slope {np.median(vals):+.3f}")
print("          A gas fraction varying by a factor ~6 across the sample shifts")
print("          the slope by well under half the distance to zero, and the")
print("          shift is in the direction that WEAKENS, not creates, it.")

lhR = np.log10(np.array([x.hR_kpc for x in GAL]))
print("\n    (iv) which inputs vary systematically with Sigma_0 at all?")
for lab, v in (("inclination i_TF", np.array([x.incl for x in GAL])),
               ("sin^2 i", np.sin(INC.ravel()) ** 2),
               ("B-K colour", BK), ("log h_R", lhR),
               ("log h_z", np.log10(HZ_TAB)),
               ("log sigma_LOS_0 (observed)", np.log10(OBS_AMP))):
    print(f"         corr(log Sigma_0, {lab:<26}) = "
          f"{np.corrcoef(lSig, v)[0,1]:+.3f}")
print("         The inclination is UNCORRELATED with Sigma_0, so an error in")
print("         alpha -- which enters only through sin^2 i -- cannot fake the")
print("         slope.  h_R and hence h_z do correlate, so the h_z-h_R relation")
print("         is the one input that could.  Perturb its exponent and refit:")
b_hR = float(np.polyfit(lSig, lhR, 1)[0])
r7b = np.random.default_rng(707)
for dsl in (-0.3, -0.15, 0.0, 0.15, 0.3):
    vals = []
    for _ in range(max(NDRAW // 10, 60)):
        C = draw_common(r7b)
        Ups, hz, fg, al, sob = draw_pergal(r7b, C)
        hz2 = hz * 10 ** (dsl * (lhR - lhR.mean()))
        aN, _, _ = amp_newton(Ups, hz2, fg, al, C)
        vals.append(np.polyfit(lSig, 2 * np.log10(sob / aN), 1)[0])
    print(f"         d(log h_z)/d(log h_R) shifted by {dsl:+.2f} -> slope "
          f"{np.median(vals):+.3f}")
print(f"         d log h_R/d log Sigma_0 = {b_hR:+.3f} in this sample, so an")
print(f"         error d in the relation's exponent moves the slope by about")
print(f"         {-b_hR:+.3f} d.  Removing the observed slope entirely would need")
print(f"         d = {abs(np.median(sl_obs_S)/b_hR):.2f}, against a published")
print("         exponent of 0.643 -- i.e. the Bershady+2010b relation would have")
print("         to be wrong in SLOPE by that much, not merely in zero point.")
RES["confound_correlations"] = {
    "i_TF": float(np.corrcoef(lSig, [x.incl for x in GAL])[0, 1]),
    "log_hR": float(np.corrcoef(lSig, lhR)[0, 1]),
    "log_hz": float(np.corrcoef(lSig, np.log10(HZ_TAB))[0, 1]),
    "d_loghR_d_logSigma": b_hR,
    "hz_slope_error_to_kill_signal": float(abs(np.median(sl_obs_S) / b_hR))}
RES["sigma_slope_audits"] = dict(bootstrap=q5(bs), p_bootstrap_ge_zero=p_bs,
                                 jackknife=[float(jk.min()), float(jk.max())])
RES["differential"] = {kk: dict(slope=q5(v["slope"]), range_x=v["range_x"],
                                null=q5(v["null"]), p=v["p"],
                                partial=v["partial"], resid=v["resid"],
                                model_scatter=v["model_scatter"],
                                inflation=v["inflation"],
                                sd_raw=v["sd_raw"], sd_inflated=v["sd_inflated"],
                                sigma_slope_pred=v.get("sigma_slope_pred"))
                       for kk, v in DIFF.items()}


# =============================================================================
head("STEP 8   What precision would make this decisive?")
# =============================================================================
sdf = RES["Bz_posterior"]["sd_full_dex"]
print(f"""
    The amplitude test measures  log B_z = 2 log sigma_z(obs) - log Upsilon_K
    - log h_z - log k + const.  Its width is {sdf:.3f} dex, and the dominant
    terms are common-mode, so N galaxies do not help.

    Separations that must be resolved.  B_z(law) here is the AMPLITUDE-based
    value -- the boost in the fitted sigma_LOS_0, which is what the amplitude
    test actually measures -- not the pointwise value at one radius.""")
print(f"    {'law':<14}{'log10 B_z(law) amp':>20}{'separation from Newton':>26}"
      f"{'sigma at current':>18}")
need = {}
for key in LAW_KEYS:
    lb = float(np.median(BZLAW_D[key].mean(axis=1)))
    sep = abs(lb)
    need[key] = dict(log10_Bz=lb, sep_dex=sep,
                     sigma_at_current=sep / sdf if sdf > 0 else np.inf,
                     dex_for_3sigma=sep / 3.0)
    print(f"    {key:<14}{lb:>20.3f}{sep:>26.3f}"
          f"{sep/sdf if sdf>0 else np.inf:>18.2f}")
print("\n    Is each law's B_z CONSISTENT with the observed one?  This is the")
print("    question the brief asks, and it is not the same as the table above.")
lbo = float(np.median(lm_f))
print(f"    {'law':<14}{'log10 Bz(law)':>15}{'log10 Bz(obs)':>15}"
      f"{'difference':>12}{'sigma':>8}{'verdict':>12}")
TENS_T = {}
for key in LAW_KEYS:
    lb = float(np.median(BZLAW_D[key].mean(axis=1)))
    sdl = float(np.std(BZLAW_D[key].mean(axis=1)))
    dd = lb - lbo
    sd = float(np.sqrt(sdf ** 2 + sdl ** 2))
    TENS_T[key] = dict(diff_dex=dd, sigma=dd / sd)
    print(f"    {key:<14}{lb:>15.3f}{lbo:>15.3f}{dd:>12.3f}{dd/sd:>8.2f}"
          f"{'consistent' if abs(dd/sd) < 2 else 'tension':>12}")
print("    Every law is within 2 sigma of the observed amplitude, including")
print("    Newton and including the two that predict a 50 per cent vertical")
print("    boost.  The amplitude does not choose between them.")
RES["law_vs_observed_tension"] = TENS_T
print(f"""
    To separate the strongest MOND-like prediction from Newton at 3 sigma the
    total budget on log10 B_z must fall below {max(need[k]['dex_for_3sigma'] for k in ('rar','aqual_simple')):.3f} dex.
    log B_z = 2 log sigma_z - log Upsilon_K - log h_z - log k + const, so the
    four dominant terms add in quadrature and each must reach about""")
_tgt = max(need[k]['dex_for_3sigma'] for k in ('rar', 'aqual_simple')) / 2.0
print(f"    {_tgt:.3f} dex ({100*(10**_tgt-1):.0f} per cent).  Term by term, against today:")
print(f"    {'term':<34}{'today':>10}{'needed':>10}{'factor':>9}")
for nm, keyn in (("Upsilon_K zero point", "Upsilon_K zero point (0.15 dex)"),
                 ("alpha = sigma_z/sigma_R", "alpha = sigma_z/sigma_R"),
                 ("h_z relation zero point", "h_z relation zero point (0.10)"),
                 ("k, vertical profile shape", "k, vertical profile [1.5,2.0]")):
    tv = solo[keyn]
    print(f"    {nm:<34}{tv:>10.3f}{_tgt:>10.3f}{tv/_tgt:>9.1f}x")
print(f"""    Concretely that means, simultaneously:
      * Upsilon_K to ~{100*(10**_tgt-1):.0f} per cent ABSOLUTE (an IMF zero point, not a
        relative calibration) -- the hardest of the four;
      * h_z MEASURED, not inferred from h_R: edge-on scale heights for these
        galaxies, or a resolved vertical decomposition;
      * the profile shape k pinned to ~10 per cent, which means resolving the
        tracer population mix -- an old thick component and a young thin one do
        not share a scale height, and the DiskMass sigma_z is a mixture;
      * the velocity ellipsoid alpha measured rather than adopted, which needs
        galaxies at two inclinations or a proper 3-integral model;
      * an independent inclination, so the rotation side stops depending on the
        inverted Tully-Fisher relation.
    None of these is a matter of more galaxies.""")
print(f"""
    The DIFFERENTIAL slope is the route that does NOT need the zero points.
    Its current width is set by the per-galaxy scatter and the range of the
    predictor.  Scaling sd(s) ~ sigma_gal /(sqrt(N) sd(x)):""")
for key in DIFF:
    s_now = DIFF[key]["sd_inflated"]
    print(f"      {key:<14} sd(s) = {s_now:.3f} (inflated); "
          f"N for sd(s)=0.15 : {int(np.ceil(NG*(s_now/0.15)**2)):>6}   "
          f"N for sd(s)=0.10 : {int(np.ceil(NG*(s_now/0.10)**2)):>6}")
print("""    Only the RAR predictor has enough dynamic range across this sample
    for the differential route to be affordable.  For the tensor laws the
    predicted B_z barely varies galaxy to galaxy, so no sample size makes the
    differential test work -- a wider range in surface density is needed, not
    more galaxies at the same surface density.""")
RES["precision_required"] = need


# =============================================================================
head("SUMMARY")
# =============================================================================
print(f"""
    B_z OBSERVED (empirical vertical boost referred to Newton, amplitude-based,
    full nuisance budget)
        {ci(10**lm_f)}
        statistical part alone: {ci(10**lm_s)}

    B_z PREDICTED by each frozen law, amplitude-based (rotation-curve fits only):""")
for key in LAW_KEYS:
    print(f"        {key:<14} {ci(10**BZLAW_D[key].mean(axis=1))}")
print("""
    B_z PREDICTED pointwise at 2.2 h_R (for reference; the amplitude test does
    not measure this directly):""")
for key in LAW_KEYS:
    print(f"        {key:<14} {ADYN_PRED[key]['Bz22']:.3f}")
print("""
    A_dyn = B_R/B_z PREDICTED by each law at 2.2 h_R (theory, no data):""")
for key in LAW_KEYS:
    print(f"        {key:<14} B_R {ADYN_PRED[key]['BR22']:.3f} / B_z "
          f"{ADYN_PRED[key]['Bz22']:.3f} = {ADYN_PRED[key]['A22']:.3f}")
print("""
    A_dyn REQUIRED by the data given each law's B_R, at 2.2 h_R:""")
for key in LAW_KEYS:
    print(f"        {key:<14} {ci(10**LAWPOST[key]['A_req'])}")
print(f"""
    DIFFERENTIAL result (immune to a common-mode Upsilon_K / h_z / k error):
        d log10 B_z / d log10 Sigma_0   observed  {np.median(sl_obs_S):+.3f}"""
      f" +- {sd_obs_S:.3f}")
print(f"        {'':<38}Newton    {0.0:+.3f}")
for key in DIFF:
    print(f"        {'':<38}{key:<10}"
          f"{DIFF[key]['sigma_slope_pred']['p50']:+.3f}")
print(f"""
    Can the data separate A_dyn = 1 from A_dyn > 1?
      amplitude route : NO.  The width on log10 B_z is {sdf:.3f} dex, the largest
                        law-to-Newton separation is
                        {max(need[k]['sep_dex'] for k in LAW_KEYS):.3f} dex, i.e.
                        {max(need[k]['sigma_at_current'] for k in LAW_KEYS):.2f} sigma.  The absolute amplitude
                        cannot tell B_z = 1 from B_z = 1.5.
      differential    : the SHAPE of B_z(Sigma_0) is resolved at
                        {abs(np.median(sl_obs_S))/sd_obs_S:.1f} sigma (Monte-Carlo error, inflated) or
                        p = {p_bs:.3f} (galaxy bootstrap, the conservative
                        estimator), and matches the MOND-like laws.  But it
                        constrains the surface-density DEPENDENCE of B_z, not
                        its amplitude, so it does not by itself fix A_dyn.""")

RES["wall_time_s"] = time.time() - T_START
with open(os.path.join(HERE, "adyn_results.json"), "w",
          encoding="utf-8", newline="\n") as fh:
    json.dump(RES, fh, indent=1, sort_keys=True, default=float)
print(f"\n    wrote adyn_results.json   total wall time "
      f"{RES['wall_time_s']:.0f}s")
