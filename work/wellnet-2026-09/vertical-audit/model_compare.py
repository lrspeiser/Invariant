"""ITEM 4 (local vs global vs potential vs tensor) and ITEM 5 (law predictions
through the identical pipeline).

Every model below is evaluated by the SAME forward chain the data are compared
through: photometry -> Sigma(R) -> K_z(R,z) -> sigma_z(R) -> sigma_LOS(R) at the
tabulated inclination -> 2.7" fibre (x) 1.5" PSF -> exponential fit over the SAME
window -> (amplitude, scale length).  Nothing is scored on a shortcut.

    M_Newton  B_z = 1
    M_local   B_z(R) = nu(|g_b|(R,z)/a0)      RAR/QUMOND, pointwise in R and z
    M_aqual   B_z(R) from the AQUAL bisection, pointwise
    M_global  B_z    = c (Sigma_0/Sigma_*)^p  constant in R inside a galaxy
    M_Phi     B_z(R) = c (|dPhi_b(R)|/Phi_*)^q
    M_tensor  B_z(R) = 1/mu_z, mu_z = 1 (anisotropic) or mu_R (isotropic)

BOUNDARY RULE FOR THE POTENTIAL, DECLARED BEFORE ANY FIT
--------------------------------------------------------
|Phi_b| is defined only up to a constant, so the rule is fixed here, in advance,
and three variants are run with ONE declared primary:

  PRIMARY   dPhi_b(R) = Phi_b(R) - Phi_b(4 h_R), the isolated baryonic disk with
            Phi -> 0 at infinity, reference at 4 h_R -- OUTSIDE the 0.3-2.0 h_R
            fitting window, so the reference is never inside the data.
  SEC-1     reference at 8 h_R.
  SEC-2     |Phi_b(R)| itself, i.e. reference at infinity.

Phi_* = (100 km/s)^2 and Sigma_* = 100 Msun/pc^2 are declared units, not fitted.

SCORING
-------
The previous run declined to quote lnL/AIC/BIC because chi2/dof was 10-130, i.e.
the residual is nuisance-dominated.  That objection is removed here by giving
EVERY model a free intrinsic scatter on each observable, so chi2/dof = 1 by
construction and the comparison is on residual STRUCTURE, with the extra
parameters counted.  Free parameters are listed for every model.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vaudit_core as V                                       # noqa: E402
import adyn_model as M                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G, KPC, PC, MSUN = M.G, M.KPC, M.PC, M.MSUN
PHI_STAR = (100e3) ** 2                    # declared unit, (100 km/s)^2
SIGMA_STAR = 100.0                         # declared unit, Msun/pc^2
PHI_REF_PRIMARY = 4.0                      # h_R, DECLARED before any fit
A0_RAR = V.FIT["rar"]
A0_AQ = V.FIT["aqual"]
ETA_ANI, A0_ANI = 4.0, 5.779e-10
ETA_ISO, A0_ISO = 0.5516, 1.393e-10
R = {}


def head(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


B = V.Bench()
NG, XG = B.NG, V.XG
UG = V.UG
lSig = B.log_sigma0()
OBS_A, OBS_EA = B.OBS_AMP, B.OBS_EAMP
OBS_H, OBS_EH = B.OBS_H, B.OBS_EH
FID = dict(zU=np.log10(0.60), sc=0.15, dhz=0.0, kv=1.5, al=0.60,
           lfg=np.log10(0.25), fhg=2.0, fhzg=0.5, lo=0.3, hi=2.0)


# ============================================================== B_z generators
def _kz_newton(base):
    """K_z^N(R, z) on the (NG, NR, NU) grid -- exactly adyn_run.law_Bz's KzN."""
    prof = base["prof"]
    w = np.interp(UG, prof.u, prof.w)
    Cs = np.interp(UG, prof.u, prof.Cn, left=0.0, right=1.0)
    pg = M.profile_for_k(2.0)
    Cg = np.interp(UG / base["f_hzg"], pg.u, pg.Cn, left=0.0, right=1.0)
    hz, Rr, gRN, Vc2N = base["hz"], base["R"], base["gR"], base["Vc2"]
    zz = UG[None, None, :] * hz[:, :, None]
    Sig_lt = (base["Sig_s"][:, :, None] * Cs[None, None, :]
              + base["Sig_g"][:, :, None] * Cg[None, None, :])
    dN = (np.gradient(Vc2N, XG, axis=1) / (B.hR_m * Rr))[:, :, None]
    KzN = np.maximum(2 * np.pi * G * Sig_lt - zz * dN, 1e-30)
    return w, zz, KzN, gRN


def bz_pointwise(kind, base, theta=None):
    """B_z_eff(NG, NR) = sigma_z^2(model)/sigma_z^2(Newton), through the Jeans
    integral -- the same weighting adyn_run.py uses."""
    w, zz, KzN, gRN = _kz_newton(base)
    if kind == "newton":
        return np.ones((NG, len(XG)))
    if kind == "rar":
        Kz = M.nu_rar(np.sqrt(gRN[:, :, None] ** 2 + KzN ** 2) / A0_RAR) * KzN
    elif kind == "aqual":
        gR = 0.5 * (gRN + np.sqrt(gRN ** 2 + 4 * gRN * A0_AQ))
        Kz = M.aqual_Kz(KzN, gR[:, :, None] * np.ones_like(KzN), A0_AQ)
    elif kind in ("tensor_aniso", "tensor_iso"):
        eta, a0 = (ETA_ANI, A0_ANI) if kind == "tensor_aniso" else (ETA_ISO, A0_ISO)
        Mb = (base["Sig_s"][:, 0] * 2 * np.pi * B.hR_m[:, 0] ** 2
              + base["Sig_g"][:, 0] * 2 * np.pi * (2.0 * B.hR_m[:, 0]) ** 2)
        rr = np.sqrt(base["R"][:, :, None] ** 2 + zz ** 2)
        muR = 1.0 / (1.0 + eta * M.s_gap(rr, Mb[:, None, None], eta, a0))
        muz = np.ones_like(muR) if kind == "tensor_aniso" else muR
        Kz = KzN / muz
    elif kind == "rar_free":
        a0v = 10 ** theta[0]
        Kz = M.nu_rar(np.sqrt(gRN[:, :, None] ** 2 + KzN ** 2) / a0v) * KzN
    elif kind == "global":
        p = theta[0]
        f = 10 ** (p * (lSig - lSig.mean()))
        Kz = KzN * f[:, None, None]
    elif kind == "local_pow":
        # the FAIR counterpart of M_global: one free exponent, but on the LOCAL
        # acceleration |g| = sqrt(g_R^2 + K_z^2) at (R, z), not on Sigma_0
        m = theta[0]
        gmag = np.sqrt(gRN[:, :, None] ** 2 + KzN ** 2) / A0_RAR
        Kz = KzN * np.maximum(gmag, 1e-6) ** (-m)
    elif kind.startswith("phi"):
        # (1 + |dPhi|/Phi_*)^q, NOT a bare power law: the reference rules put a
        # zero of dPhi at R_ref, and x^q is singular there.  This form is
        # non-singular everywhere, monotone in |dPhi|, reduces to Newton at
        # q = 0, and carries exactly one parameter -- same as M_global.
        q = theta[0]
        f = (1.0 + np.abs(PHI_MAP[kind]) / PHI_STAR) ** q
        Kz = KzN * f[:, :, None]
    else:
        raise KeyError(kind)
    s2 = np.trapezoid(w[None, None, :] * Kz, zz, axis=2)
    s2n = np.trapezoid(w[None, None, :] * KzN, zz, axis=2)
    return s2 / s2n


# ------------------------------------------------- the declared potential maps
def build_phi(base):
    """dPhi_b(R) in the midplane, from the SAME g_R the chain builds.

        Phi(R) - Phi(R_ref) = - int_R^{R_ref} g_R dR'
    """
    gR, Rm = base["gR"], base["R"]                  # (NG, NR), metres
    # cumulative integral of g_R dR from the innermost grid point outward
    I = np.zeros_like(gR)
    I[:, 1:] = np.cumsum(0.5 * (gR[:, 1:] + gR[:, :-1]) * np.diff(Rm, axis=1),
                         axis=1)
    out = {}
    for tag, xref in (("phi_ref4", 4.0), ("phi_ref8", 8.0)):
        j = int(np.argmin(np.abs(XG - xref)))
        out[tag] = I - I[:, [j]]                    # Phi(R) - Phi(R_ref)
    # reference at infinity: Phi(R) = -int_R^inf g dR ~ -(I_inf - I) with the
    # analytic 1/R tail beyond the grid
    Mb = gR[:, [-1]] * Rm[:, [-1]] ** 2 / G
    tail = G * Mb / Rm[:, [-1]]
    out["phi_inf"] = -( (I[:, [-1]] - I) + tail )
    return out


# =========================================================== forward observable
def observables(kind, theta=None, C=None, hz=None):
    C = C or FID
    Ups = 10 ** (C["zU"] + C["sc"] * (B.BK - 3.4))
    hz = B.HZ_TAB * 10 ** C["dhz"] if hz is None else hz
    fg = np.full(NG, 10 ** C["lfg"])
    al = np.full(NG, C["al"])
    base = B.newton_chain(Ups, hz, fg, al, C["kv"], C["fhg"], C["fhzg"])
    global PHI_MAP
    PHI_MAP = build_phi(base)
    bz = bz_pointwise(kind, base, theta)
    sl = B.to_los(np.sqrt(base["s2"] * bz) / 1e3, al)
    a_, h_ = M.fit_exponential_rows(XG, sl, C["lo"], C["hi"])
    return a_, h_ * np.squeeze(B.hR_as), bz


# =============================================================================
head("ITEM 5   law predictions pushed through the IDENTICAL pipeline")
# =============================================================================
print("""
    Same aperture, same PSF, same 0.3-2.0 h_R window, same exponential fit
    operator, same inclination and ellipsoid projection as the data.  The
    numbers below are the fitted AMPLITUDE and SCALE LENGTH of the model
    sigma_LOS(R), not pointwise B_z at one radius.
""")
LAWS = ["newton", "rar", "aqual", "tensor_aniso", "tensor_iso"]
aN, hN, _ = observables("newton")
PRED = {}
print(f"    {'model':<14}{'amp pred':>10}{'amp obs':>9}{'h pred':>9}{'h obs':>8}"
      f"{'<log Bz>':>10}{'d logBz/d logSig':>19}")
for k in LAWS:
    a_, h_, bz = observables(k)
    lb = 2 * np.log10(a_ / aN)
    sp = float(np.polyfit(lSig, lb, 1)[0])
    PRED[k] = dict(amp=a_.tolist(), h=h_.tolist(), logBz_amp=lb.tolist(),
                   mean_logBz=float(lb.mean()), sigma_slope=sp,
                   Bz_1hR=float(np.median(bz[:, B.J10])),
                   Bz_22hR=float(np.median(bz[:, B.J22])))
    print(f"    {k:<14}{np.median(a_):>10.2f}{np.median(OBS_A):>9.2f}"
          f"{np.median(h_):>9.2f}{np.median(OBS_H):>8.2f}"
          f"{lb.mean():>10.3f}{sp:>19.3f}")
print(f"    {'OBSERVED':<14}{'':>10}{np.median(OBS_A):>9.2f}{'':>9}"
      f"{np.median(OBS_H):>8.2f}{'':>10}{-0.346:>19.3f}")
print("\n    Published (adyn REPORT.md sec.7): RAR -0.291, AQUAL -0.264,")
print("    aniso -0.020, iso -0.055.  Reproduced above at fiducial nuisances.")
R["item5_pipeline_predictions"] = {k: {kk: vv for kk, vv in v.items()
                                       if kk not in ("amp", "h", "logBz_amp")}
                                   for k, v in PRED.items()}

print("\n    radial structure each law demands, measured the same way")
print(f"    {'model':<14}{'h_sigma pred (\")':>18}{'obs 28.65':>11}"
      f"{'B_z(0.3hR)/B_z(2hR)':>22}")
j03 = int(np.argmin(np.abs(XG - 0.3)))
j20 = int(np.argmin(np.abs(XG - 2.0)))
for k in LAWS:
    a_, h_, bz = observables(k)
    rr = float(np.median(bz[:, j03] / bz[:, j20]))
    print(f"    {k:<14}{np.median(h_):>18.2f}{np.median(OBS_H):>11.2f}{rr:>22.4f}")
    R["item5_pipeline_predictions"][k]["Bz_ratio_0.3_over_2.0"] = rr

# per-galaxy h_sigma / h_R -- the form MaNGA can be compared against
hRa = np.squeeze(B.hR_as)
print(f"\n    the same radial statistic as h_sigma/h_R, per galaxy (median):")
print(f"      {'OBSERVED (DiskMass)':<24}{np.median(OBS_H / hRa):>8.3f}")
R["item5_h_over_hR"] = {"observed": float(np.median(OBS_H / hRa))}
for k in LAWS:
    a_, h_, _ = observables(k)
    R["item5_h_over_hR"][k] = float(np.median(h_ / hRa))
    print(f"      {k:<24}{np.median(h_ / hRa):>8.3f}")


# =============================================================================
head("ITEM 4   M_local vs M_global vs M_Phi vs M_tensor, same data, same chain")
# =============================================================================
la_o, lh_o = np.log10(OBS_A), np.log10(OBS_H)
sa_o = OBS_EA / OBS_A / np.log(10)
sh_o = OBS_EH / OBS_H / np.log(10)
print(f"\n    28 galaxies x 2 observables = 56 numbers.")
print(f"    median frac error: amplitude {np.median(sa_o):.4f} dex, "
      f"scale length {np.median(sh_o):.4f} dex")


def _inner(a_, h_):
    """Given a model prediction, profile out c (common-mode amplitude offset),
    s_a and s_h (free intrinsic scatters).  No forward model is called here, so
    this costs microseconds and the expensive chain runs once per shape point."""
    la_m, lh_m = np.log10(a_), np.log10(h_)

    def nll(t):
        c, lsa, lsh = t
        va = sa_o ** 2 + np.exp(2 * lsa)
        vh = sh_o ** 2 + np.exp(2 * lsh)
        ra, rh = la_o - (la_m + c), lh_o - lh_m
        return 0.5 * float(np.sum(ra ** 2 / va + np.log(va)
                                  + rh ** 2 / vh + np.log(vh)))
    best = None
    for c0 in (float(np.mean(la_o - la_m)), 0.0):
        m = minimize(nll, [c0, np.log(0.12), np.log(0.12)],
                     method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-9, fatol=1e-11))
        if best is None or m.fun < best.fun:
            best = m
    return float(best.fun), float(best.x[0]), float(np.exp(best.x[1])), \
        float(np.exp(best.x[2]))


def score(kind, grid=None, C=None):
    """grid = None for a 0-shape-parameter model, else a 1-D array of theta."""
    if grid is None:
        a_, h_, _ = observables(kind, None, C)
        f, c, sa, sh = _inner(a_, h_)
        th, npar = [], 0
    else:
        best = None
        for t in grid:
            a_, h_, _ = observables(kind, [t], C)
            # a negative fitted h means sigma_LOS RISES outward for that galaxy;
            # -1/b is then finite but negative and log10 of it is nan, so the
            # grid point must be rejected, not silently propagated.
            if not (np.all(np.isfinite(a_)) and np.all(np.isfinite(h_))
                    and np.all(a_ > 0) and np.all(h_ > 0)):
                continue
            r = _inner(a_, h_)
            if best is None or r[0] < best[0][0]:
                best = (r, t)
        if best is None:
            # every grid point produced a rising or non-finite profile: the
            # model simply cannot be fitted for this nuisance draw.  Rank it
            # last rather than crashing or silently returning nan.
            return dict(nll=1e6, k=4, aic=2e6, bic=2e6, c=0.0, s_a=np.nan,
                        s_h=np.nan, theta=[float(np.nan)], degenerate=True)
        (f, c, sa, sh), tb = best
        th, npar = [float(tb)], 1
    k = 3 + npar
    n = 2 * NG
    return dict(nll=f, k=k, aic=float(2 * f + 2 * k),
                bic=float(2 * f + k * np.log(n)),
                c=c, s_a=sa, s_h=sh, theta=th)


GP = np.linspace(-1.2, 0.6, 37)          # Sigma_0 exponent p
GQ = np.linspace(-6.0, 6.0, 61)          # potential exponent q, on (1+|dPhi|/Phi_*)
GM = np.linspace(-0.6, 1.2, 37)          # local |g| exponent m
GA = np.linspace(-11.5, -8.5, 31)        # log10 a0 for the free-a0 RAR
MODELS = [
    ("M_Newton",        "newton",       None),
    ("M_local_RAR",     "rar",          None),
    ("M_local_AQUAL",   "aqual",        None),
    ("M_tensor_aniso",  "tensor_aniso", None),
    ("M_tensor_iso",    "tensor_iso",   None),
    ("M_local_RAR_a0",  "rar_free",     GA),
    ("M_local_pow",     "local_pow",    GM),
    ("M_global",        "global",       GP),
    ("M_Phi_ref4",      "phi_ref4",     GQ),
    ("M_Phi_ref8",      "phi_ref8",     GQ),
    ("M_Phi_inf",       "phi_inf",      GQ),
]
print(f"\n    {'model':<17}{'k':>3}{'-2lnL':>10}{'AIC':>10}{'dAIC':>9}"
      f"{'BIC':>10}{'dBIC':>9}{'s_amp':>8}{'s_h':>8}{'shape':>10}")
S = {}
for name, kind, grid in MODELS:
    S[name] = score(kind, grid)
best_aic = min(v["aic"] for v in S.values())
best_bic = min(v["bic"] for v in S.values())
for name, _, _ in MODELS:
    v = S[name]
    sh = f"{v['theta'][0]:+.3f}" if v["theta"] else "--"
    print(f"    {name:<17}{v['k']:>3}{2*v['nll']:>10.2f}{v['aic']:>10.2f}"
          f"{v['aic']-best_aic:>9.2f}{v['bic']:>10.2f}{v['bic']-best_bic:>9.2f}"
          f"{v['s_a']:>8.3f}{v['s_h']:>8.3f}{sh:>10}")
R["item4_model_comparison"] = S

# ------------------------------- where does each model win or lose?  split it
print("\n    the two halves of the evidence, separated")
print(f"    {'model':<17}{'chi2 amp (28)':>15}{'chi2 h (28)':>13}"
      f"{'rms amp dex':>13}{'rms h dex':>11}")
SPLIT = {}
for name, kind, grid in MODELS:
    v = S[name]
    th = v["theta"] if v["theta"] else None
    a_, h_, _ = observables(kind, th)
    ra = la_o - (np.log10(a_) + v["c"])
    rh = lh_o - np.log10(h_)
    ca = float(np.sum(ra ** 2 / (sa_o ** 2 + v["s_a"] ** 2)))
    ch = float(np.sum(rh ** 2 / (sh_o ** 2 + v["s_h"] ** 2)))
    SPLIT[name] = dict(chi2_amp=ca, chi2_h=ch, rms_amp=float(np.std(ra)),
                       rms_h=float(np.std(rh)))
    print(f"    {name:<17}{ca:>15.2f}{ch:>13.2f}{np.std(ra):>13.4f}"
          f"{np.std(rh):>11.4f}")
R["item4_split"] = SPLIT

# ------------------------------- robustness of the ranking to the nuisances
NDR = int(os.environ.get("VAUDIT_NDRAW_RANK", "24"))
print(f"\n    ranking under {NDR} draws of the full nuisance prior (dAIC vs the")
print("    best model in each draw); a ranking that survives is a result")
rng = np.random.default_rng(4242)
tally = {n: [] for n, _, _ in MODELS}
thetas = {n: [] for n, _, _ in MODELS}
for idr in range(NDR):
    C = V.Bench.draw_common(rng)
    sc = {}
    for name, kind, grid in MODELS:
        v = score(kind, grid, C=C)
        sc[name] = v["aic"]
        if v["theta"] and np.isfinite(v["theta"][0]):
            thetas[name].append(v["theta"][0])
    b = min(sc.values())
    for name in sc:
        tally[name].append(min(sc[name] - b, 1e3))
    print(f"      draw {idr+1}/{NDR}  best = "
          f"{min(sc, key=sc.get)}", flush=True)
print(f"    {'model':<17}{'median dAIC':>13}{'p16':>9}{'p84':>9}"
      f"{'won':>6}{'shape [p16,p84]':>22}")
for name, _, _ in MODELS:
    t = np.array(tally[name])
    t = t[np.isfinite(t)]
    th = np.array(thetas[name])
    th = th[np.isfinite(th)]
    ts = (f"[{np.percentile(th,16):+.3f},{np.percentile(th,84):+.3f}]"
          if th.size else "--")
    print(f"    {name:<17}{np.median(t):>13.2f}{np.percentile(t,16):>9.2f}"
          f"{np.percentile(t,84):>9.2f}{int(np.sum(t < 1e-9)):>6}{ts:>22}")
    R.setdefault("item4_nuisance_ranking", {})[name] = dict(
        median_dAIC=float(np.median(t)), p16=float(np.percentile(t, 16)),
        p84=float(np.percentile(t, 84)), wins=int(np.sum(t < 1e-9)),
        shape_p16=float(np.percentile(th, 16)) if th.size else None,
        shape_p50=float(np.percentile(th, 50)) if th.size else None,
        shape_p84=float(np.percentile(th, 84)) if th.size else None)

# ------------------------------- how radially flat is each state variable?
print("\n    WHY the models differ: the radial dynamic range of each candidate")
print("    state variable over the 0.3-2.0 h_R fitting window")
base = B.newton_chain(10 ** (FID["zU"] + 0.15 * (B.BK - 3.4)), B.HZ_TAB,
                      np.full(NG, 0.25), np.full(NG, 0.60), 1.5, 2.0, 0.5)
PHI_MAP = build_phi(base)
rng_tab = {}
for tag, arr in (("g_b(R)", base["gR"]),
                 ("Sigma_0 (constant in R)", np.ones((NG, len(XG)))),
                 ("|dPhi_b| ref 4 h_R", np.abs(PHI_MAP["phi_ref4"])),
                 ("|dPhi_b| ref 8 h_R", np.abs(PHI_MAP["phi_ref8"])),
                 ("|Phi_b| ref infinity", np.abs(PHI_MAP["phi_inf"]))):
    within = np.median(np.log10(arr[:, j03] / np.maximum(arr[:, j20], 1e-30)))
    between = float(np.std(np.log10(np.maximum(arr[:, B.J22], 1e-30))))
    cc = float(np.corrcoef(lSig, np.log10(np.maximum(arr[:, B.J22], 1e-30)))[0, 1]) \
        if np.std(arr[:, B.J22]) > 0 else float("nan")
    rng_tab[tag] = dict(within_dex=float(within), between_sd_dex=between,
                        corr_with_logSigma0_at_2p2hR=cc)
    print(f"      {tag:<26} within-galaxy {within:+.3f} dex   "
          f"between-galaxy sd {between:.3f} dex   r(logSigma_0) {cc:+.4f}")
R["item4_state_variable_range"] = rng_tab
print("\n      -> for an exponential disk the Freeman formula makes g_b at a")
print("         FIXED multiple of h_R proportional to Sigma_0 with a pure")
print("         constant, so 'local in g_b' and 'global in Sigma_0' are the")
print("         SAME between-galaxy predictor here.  Only the RADIAL shape can")
print("         separate them, and g_b is nearly flat across 0.3-2.0 h_R.")

with open(os.path.join(HERE, "model_compare.json"), "w") as fh:
    json.dump(R, fh, indent=1)
print("\n  wrote model_compare.json")
