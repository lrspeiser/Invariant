"""THE A_dyn PREDICTION. Fit radially, freeze, predict vertically.

The anisotropic well-mirror version replaces the scalar mu by a tensor:

    (1/R) d/dR [ R mu_R(D) dPhi/dR ] + d/dz [ mu_z(D) dPhi/dz ] = 4 pi G rho_b

with mu_R = 1/(1 + eta s(r)) < 1 from the gap and mu_z = 1. That is the whole
distinctive claim: radial gravity enhanced, VERTICAL gravity left baryonic, so

    A_dyn = (g_R/g_R,N) / (K_z/K_z,N)  >  1

This script solves that tensor equation with the validated axisymmetric solver
in gravitylab/axisym.py, fits the RADIAL side only, freezes every parameter,
and then predicts the vertical side. The vertical comparison is against
DiskMass, and it is deliberately built on the ONE combination that survives the
acquisition report's caveats:

    h_sigma_z / h_R,  both measured in ARCSEC, both from DiskMass VI.

  * sigma_z is an exponential FIT, not a resolved profile (caveat 2), so only
    its scale length is used, never a pointwise value.
  * h_z is INFERRED from h_R (caveat 3), so it is never used as an independent
    quantity; it enters only as a constant multiplying sigma_z^2, and a
    constant cancels out of a scale length.
  * Sigma_dyn is not tabulated (caveat 1). It is NOT reconstructed and no proxy
    is substituted; the test does not need it.
  * h_R in arcsec (VI) and kpc (VII) are checked against each other through the
    distance before either is used (caveat 4).

For a disk with sigma_z^2 = c h_z |K_z(R)| and h_z, c constant in R,
d ln sigma_z/dR = (1/2) d ln|K_z|/dR, so the scale length of sigma_z is
fixed by the SHAPE of the vertical force alone. Newtonian thin-disk:
|K_z| ~ Sigma ~ exp(-R/h_R), giving h_sigma_z = 2 h_R.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

GLAB = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/work/gravitylab"
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
for q in (GLAB, HERE):
    if q not in sys.path:
        sys.path.insert(0, q)

import axisym as X            # noqa: E402  the validated tensor solver
import data as DAT            # noqa: E402
import mirror_models as MM    # noqa: E402

G, KPC, MSUN, KMS = MM.G, MM.KPC, MM.MSUN, 1e3
A0 = MM.A0_CANON
ACQ = os.path.join(SCR, "acquire")
BAR = "=" * 78
MSUN_K = 3.28           # 2MASS Ks absolute magnitude of the Sun


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


# ------------------------------------------------------------------ geometry
def make_disk(Mb_msun, Rd_kpc, hz_kpc, nR=160, nz=80, box=14.0, zbox=7.0):
    """Exponential disk of total baryonic mass Mb, on a grid sized in Rd."""
    g = X.Grid(nR, nz, box * Rd_kpc, zbox * Rd_kpc)
    Sigma0 = Mb_msun * MSUN / (2 * np.pi * (Rd_kpc * KPC) ** 2)
    rho = X.exponential_disk(g.Rc / KPC, g.zc / KPC, Sigma0, Rd_kpc, hz_kpc)
    rho *= (Mb_msun * MSUN / 2.0) / float(np.sum(rho * g.V))   # z >= 0 half
    return g, rho, Sigma0


def mu_field(g, Mb_msun, eta, a0):
    """mu_R on the (R, z) grid from the postulated gap profile, evaluated at
    the SPHERICAL radius, which is the argument the profile is written in."""
    R = g.Rc[:, None] * np.ones((1, g.nz))
    z = np.ones((g.nR, 1)) * g.zc[None, :]
    rho_sph = np.sqrt(R ** 2 + z ** 2)
    s = MM.s_gap(rho_sph, Mb_msun, eta, a0)
    return 1.0 / (1.0 + eta * s)


def solve_case(Mb_msun, Rd_kpc, hz_kpc, eta, a0, mode, **gk):
    """mode: 'newton' | 'iso' (mu_R = mu_z = mu) | 'aniso' (mu_z = 1)."""
    g, rho, _ = make_disk(Mb_msun, Rd_kpc, hz_kpc, **gk)
    if mode == "newton":
        A = X.isotropic_A(rho.shape)
    else:
        mR = mu_field(g, Mb_msun, eta, a0)
        mz = mR.copy() if mode == "iso" else np.ones_like(mR)
        A = (mR, mz, np.zeros_like(mR))
    bc = X.monopole_bc(g, Mb_msun * MSUN)
    Psi, it, rel = X.solve_axi(rho, A, g, bc, tol=1e-11, maxiter=12000)
    return g, Psi, dict(iters=it, rel=rel, converged=bool(rel < 1e-9))


def profiles(g, Psi):
    """Midplane circular speed, radial force and vertical force."""
    vc = X.midplane_vc(Psi, g)
    gR = vc ** 2 / np.maximum(g.Rc, 1e-30)
    Kz = np.abs(X.vertical_Kz(Psi, g, iz=1))
    return g.Rc / KPC, gR, Kz


# ------------------------------------------------------------------ emulator
#  The radial boost B = g_R(tensor)/g_R(Newton) is a RATIO, and the field
#  equation is linear in Phi once mu is fixed, so the total mass scales out
#  exactly. B therefore depends only on the dimensionless triple
#      xi = R/r_t ,  zeta = Rd/r_t ,  eta        (with hz/Rd held global)
#  which is what makes a fit of the tensor model to 75 galaxies affordable.
#  The scale invariance is verified numerically before the emulator is used.
#  What is tabulated is NOT B itself but the CORRECTION
#      Q = B_solved / B_algebraic,   B_algebraic = 1 + eta s(r)
#  which is bounded and slowly varying, where B spans decades. That matters
#  because a lookup CLAMPS at its edges, and clamping a bounded correction is a
#  mild, stateable approximation while clamping B itself is the silent
#  constant-extrapolation failure this programme has already paid for. The
#  extrapolation fraction and the full range of Q are both reported.
ETA_GRID = np.array([0.02, 0.05, 0.12, 0.3, 0.7, 1.6, 4.0, 10.0])
ZETA_GRID = np.logspace(-1.5, 2.5, 14)
XI_GRID = np.logspace(-2.5, 3.5, 60)
HZ_OVER_RD = 0.15


def build_emulator(mode="aniso", nR=120, nz=60, verbose=True, cache=True):
    """log10 Q(xi; zeta, eta) on a grid of tensor solves.

    Cached to disk keyed on the grid definition, so the table is rebuilt only
    when the grids or the solver change, never merely because the script is
    re-run."""
    key = (f"emu_{mode}_{nR}x{nz}_{len(ETA_GRID)}_{len(ZETA_GRID)}_"
           f"{len(XI_GRID)}_{HZ_OVER_RD}.npy")
    path = os.path.join(HERE, key)
    if cache and os.path.exists(path):
        if verbose:
            print(f"      loaded cached table {key}")
        return np.load(path)
    tab = np.zeros((len(ETA_GRID), len(ZETA_GRID), len(XI_GRID)))
    Rd_, Mb = 3.0, 1e10
    for i, e in enumerate(ETA_GRID):
        for j, z in enumerate(ZETA_GRID):
            rt_kpc = Rd_ / z
            a0 = G * Mb * MSUN / (rt_kpc * KPC / e) ** 2     # invert r_t
            gN_, PsN, _ = solve_case(Mb, Rd_, HZ_OVER_RD * Rd_, e, a0,
                                     "newton", nR=nR, nz=nz)
            gg, Ps, _ = solve_case(Mb, Rd_, HZ_OVER_RD * Rd_, e, a0, mode,
                                   nR=nR, nz=nz)
            Rk, gRN, _ = profiles(gN_, PsN)
            _, gR, _ = profiles(gg, Ps)
            B = np.maximum(gR / np.maximum(gRN, 1e-300), 1e-6)
            Balg = 1.0 + e * MM.s_gap(Rk * KPC, Mb, e, a0)
            Q = np.maximum(B / Balg, 1e-6)
            xi_s = Rk / rt_kpc
            m = (Rk > 0.05 * Rd_) & (Rk < 12 * Rd_) & np.isfinite(Q)
            tab[i, j] = np.interp(np.log(XI_GRID), np.log(xi_s[m]),
                                  np.log10(Q[m]))
        if verbose:
            print(f"      eta = {e:<5.2f} done ({len(ZETA_GRID)} zeta solves)")
    if cache:
        np.save(path, tab)
    return tab


def emulate(tab, eta, a0, Mb_msun, r_t_m, R_m, Rd_kpc):
    """B = B_algebraic (exact, closed form) x Q (interpolated correction).
    Returns the boost and the fraction of queries outside the tabulated box."""
    raw_xi = R_m / r_t_m
    raw_ze = (Rd_kpc * KPC) / r_t_m
    xi = np.clip(raw_xi, XI_GRID[0], XI_GRID[-1])
    zeta = np.clip(raw_ze, ZETA_GRID[0], ZETA_GRID[-1])
    et = np.clip(eta, ETA_GRID[0], ETA_GRID[-1])
    out_frac = float(np.mean((raw_xi < XI_GRID[0]) | (raw_xi > XI_GRID[-1])
                             | (raw_ze < ZETA_GRID[0]) | (raw_ze > ZETA_GRID[-1])
                             | (eta < ETA_GRID[0]) | (eta > ETA_GRID[-1])))

    def w(gr, v):
        lg = np.log(gr)
        k = np.clip(np.searchsorted(lg, np.log(v)) - 1, 0, len(gr) - 2)
        f = (np.log(v) - lg[k]) / (lg[k + 1] - lg[k])
        return k, np.clip(f, 0.0, 1.0)

    ie, fe = w(ETA_GRID, np.atleast_1d(et))
    iz, fz = w(ZETA_GRID, np.atleast_1d(zeta) * np.ones_like(np.atleast_1d(xi)))
    ix, fx = w(XI_GRID, np.atleast_1d(xi))
    v = 0.0
    for de, we in ((0, 1 - fe), (1, fe)):
        for dz, wz in ((0, 1 - fz), (1, fz)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                v = v + we * wz * wx * tab[ie + de, iz + dz, ix + dx]
    Balg = 1.0 + eta * MM.s_gap(R_m, Mb_msun, eta, a0)
    return Balg * 10 ** v, out_frac


def exp_scale(Rk, y, lo, hi):
    """Scale length of an exponential fitted to y over R in [lo, hi] kpc,
    the same functional form DiskMass fits to sigma_z."""
    m = (Rk >= lo) & (Rk <= hi) & (y > 0) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan")
    sl = np.polyfit(Rk[m], np.log(y[m]), 1)[0]
    return float(-1.0 / sl) if sl < 0 else float("inf")


# ------------------------------------------------------------------- DiskMass
def load_diskmass():
    def rd(f):
        with open(os.path.join(ACQ, f), encoding="utf-8") as fh:
            return {r["UGC"]: r for r in csv.DictReader(fh, delimiter="\t")}
    t1, t6, t7 = (rd("dms6_table1_galaxy_properties.tsv"),
                  rd("dms6_table6_sigma_z.tsv"), rd("dms7_hR_hz.tsv"))
    out = []
    for u, a in t1.items():
        if u not in t6 or u not in t7:
            continue
        try:
            D = float(a["Dist"])
            hR_as = float(a["h_R_arcsec"]); ehR_as = float(a["e_h_R_arcsec"])
            hsz = float(t6[u]["h_sigma_z"]); ehsz = float(t6[u]["e_h_sigma_z"])
            hR_kpc = float(t7[u]["h_R"]); hz_kpc = float(t7[u]["h_z"])
            mu0 = float(a["mu0_K_i"]); BK = float(a["B_K"])
            sz0 = float(t6[u]["sigma_z_0"])
        except (ValueError, KeyError):
            continue
        out.append(dict(ugc=u, D=D, hR_as=hR_as, ehR_as=ehR_as, hsz_as=hsz,
                        ehsz_as=ehsz, hR_kpc=hR_kpc, hz_kpc=hz_kpc,
                        mu0_K_i=mu0, B_K=BK, sigma_z_0=sz0,
                        ratio=hsz / hR_as,
                        e_ratio=(hsz / hR_as) * math.hypot(ehsz / hsz,
                                                           ehR_as / hR_as)))
    return out


def stellar_mass(gal, ups_K):
    """M_* from the inclination-corrected central K-band surface brightness and
    the exponential scale length. Sigma_0 = 10^[0.4(M_sun,K + 21.572 - mu0)]
    in Lsun/pc^2; L_tot = 2 pi Sigma_0 h_R^2. ups_K is STATED, never fitted."""
    Sig0 = 10 ** (0.4 * (MSUN_K + 21.572 - gal["mu0_K_i"]))       # Lsun/pc^2
    L = 2 * math.pi * Sig0 * (gal["hR_kpc"] * 1e3) ** 2           # Lsun
    return ups_K * L


# ------------------------------------------------------------------ main
def main():
    out = {}
    head("STEP 0  the solver is the validated one -- re-check before use")
    Rd, Sig = 3.0, 500 * MSUN / KPC ** 2 * 1e6
    Mtot = 2 * np.pi * Sig * (Rd * KPC) ** 2
    g = X.Grid(216, 108, 60.0, 30.0)
    rho = X.exponential_disk(g.Rc / KPC, g.zc / KPC, Sig, Rd, 0.10)
    rho *= (Mtot / 2.0) / float(np.sum(rho * g.V))
    Psi, it, rel = X.solve_axi(rho, X.isotropic_A(rho.shape), g,
                               X.monopole_bc(g, Mtot), tol=1e-12, maxiter=9000)
    vc = X.midplane_vc(Psi, g); vf = X.freeman_vc(g.Rc / KPC, Sig, Rd)
    m = (g.Rc / KPC > 1.5) & (g.Rc / KPC < 27.0)
    err = float(np.sqrt(np.mean((vc[m] - vf[m]) ** 2)) / np.sqrt(np.mean(vf[m] ** 2)))
    print(f"   Freeman exact-disk relative error at 216x108 : {err:.3e}")
    print(f"   test_axisym.py reports                        : 1.4e-2")
    print(f"   {'PASS' if err < 0.05 else 'FAIL'}")
    out["solver_freeman_err"] = err

    head("STEP 1a  Does the tensor model's radial force equal the algebraic one?")
    print("""   The screening fitted Option 2 through its exact spherical reduction,
   g = g_N (1 + eta s). The anisotropic tensor has no such reduction, so the
   parameters cannot simply be inherited -- that has to be checked, not
   assumed.\n""")
    eta_s, a0_s = 0.3305, 6.987e-11        # Option 2, TRAIN-only, from step 3
    print(f"   Option 2 as fitted on TRAIN: eta = {eta_s}, a0 = {a0_s:.4g}\n")
    print(f"   {'log Mb':>8}{'Rd kpc':>8}{'R/Rd':>7}"
          f"{'B_iso solved':>14}{'B algebraic':>13}{'B_aniso solved':>16}"
          f"{'aniso/iso':>11}")
    print("   " + "-" * 78)
    cal = []
    for lMb, Rd_ in ((9.5, 1.5), (10.3, 3.0), (11.0, 5.0)):
        Mb = 10 ** lMb
        hz = HZ_OVER_RD * Rd_
        gN_, PsN, _ = solve_case(Mb, Rd_, hz, eta_s, a0_s, "newton")
        gI_, PsI, _ = solve_case(Mb, Rd_, hz, eta_s, a0_s, "iso")
        gA_, PsA, _ = solve_case(Mb, Rd_, hz, eta_s, a0_s, "aniso")
        Rk, gRN, _ = profiles(gN_, PsN)
        _, gRI, _ = profiles(gI_, PsI)
        _, gRA, _ = profiles(gA_, PsA)
        for rr in (1.0, 2.2, 4.0):
            j = int(np.argmin(np.abs(Rk - rr * Rd_)))
            alg = 1.0 + eta_s * MM.s_gap(Rk[j] * KPC, Mb, eta_s, a0_s)
            bi, ba = gRI[j] / gRN[j], gRA[j] / gRN[j]
            cal.append(dict(logMb=lMb, Rd=Rd_, RoverRd=rr, B_iso=float(bi),
                            B_alg=float(alg), B_aniso=float(ba)))
            print(f"   {lMb:>8.1f}{Rd_:>8.1f}{rr:>7.1f}{bi:>14.4f}{alg:>13.4f}"
                  f"{ba:>16.4f}{ba/bi:>11.4f}")
    print("   " + "-" * 78)
    d_alg = np.array([math.log10(c["B_iso"] / c["B_alg"]) for c in cal])
    d_ani = np.array([math.log10(c["B_aniso"] / c["B_iso"]) for c in cal])
    print(f"   isotropic tensor vs algebraic reduction : "
          f"{np.abs(d_alg).max():.4f} dex worst, {np.sqrt(np.mean(d_alg**2)):.4f} RMS")
    print(f"   anisotropic vs isotropic RADIAL force   : "
          f"{np.abs(d_ani).max():.4f} dex worst, {np.sqrt(np.mean(d_ani**2)):.4f} RMS")
    print("""   The anisotropic radial force is up to 0.2 dex WEAKER: suppressing only
   mu_R while leaving mu_z = 1 lets flux escape vertically, which costs radial
   force. The parameters therefore do NOT transfer and the tensor model has to
   be fitted in its own right.""")
    out["step1a_radial_calibration"] = dict(
        rows=cal, rms_iso_vs_algebraic=float(np.sqrt(np.mean(d_alg ** 2))),
        rms_aniso_vs_iso=float(np.sqrt(np.mean(d_ani ** 2))),
        parameters_inherit=False, eta_screening=eta_s, a0_screening=a0_s)

    head("STEP 1b  Emulator for the tensor radial force, and its validation")
    print("""   B = g_R(tensor)/g_R(Newton) is a ratio and the field equation is
   linear in Phi at fixed mu, so the total mass cancels exactly and B depends
   only on (xi, zeta, eta) = (R/r_t, R_d/r_t, eta) at fixed h_z/R_d. That is
   checked first, then a grid of solves is tabulated so the tensor model can be
   fitted to 75 galaxies at all.\n""")
    b1 = solve_case(1e9, 3.0, HZ_OVER_RD * 3.0, 0.5,
                    G * 1e9 * MSUN / (2.0 * KPC / 0.5) ** 2, "aniso")
    b0 = solve_case(1e9, 3.0, HZ_OVER_RD * 3.0, 0.5,
                    G * 1e9 * MSUN / (2.0 * KPC / 0.5) ** 2, "newton")
    c1 = solve_case(1e11, 3.0, HZ_OVER_RD * 3.0, 0.5,
                    G * 1e11 * MSUN / (2.0 * KPC / 0.5) ** 2, "aniso")
    c0 = solve_case(1e11, 3.0, HZ_OVER_RD * 3.0, 0.5,
                    G * 1e11 * MSUN / (2.0 * KPC / 0.5) ** 2, "newton")
    Bl = profiles(*b1[:2])[1] / profiles(*b0[:2])[1]
    Bh = profiles(*c1[:2])[1] / profiles(*c0[:2])[1]
    inv = float(np.max(np.abs(np.log10(Bl[5:-5] / Bh[5:-5]))))
    print(f"   mass invariance of B at fixed (r_t, R_d, eta), 1e9 vs 1e11 Msun:")
    print(f"     max |dlog10 B| = {inv:.2e}   {'PASS' if inv < 1e-6 else 'FAIL'}")
    print(f"   building emulator: {len(ETA_GRID)} eta x {len(ZETA_GRID)} zeta solves")
    EMU = build_emulator("aniso")
    EMU_ISO = build_emulator("iso", verbose=False)
    print("   validating against direct solves at points OFF the grid:")
    print(f"   {'eta':>7}{'zeta':>8}{'R/Rd':>7}{'B direct':>11}{'B emulated':>13}"
          f"{'dex err':>10}")
    print("   " + "-" * 58)
    verr = []
    for e, z in ((0.27, 0.6), (0.55, 2.3), (1.1, 7.0)):
        Rd_, Mb = 3.0, 1e10
        rt_kpc = Rd_ / z
        a0q = G * Mb * MSUN / (rt_kpc * KPC / e) ** 2
        gN_, PsN, _ = solve_case(Mb, Rd_, HZ_OVER_RD * Rd_, e, a0q, "newton")
        gg, Ps, _ = solve_case(Mb, Rd_, HZ_OVER_RD * Rd_, e, a0q, "aniso")
        Rk, gRN, _ = profiles(gN_, PsN); _, gR, _ = profiles(gg, Ps)
        for rr in (0.8, 2.2, 5.0):
            j = int(np.argmin(np.abs(Rk - rr * Rd_)))
            bd = gR[j] / gRN[j]
            be, _ = emulate(EMU, e, a0q, Mb, rt_kpc * KPC,
                            np.array([Rk[j] * KPC]), Rd_)
            d = float(math.log10(be[0] / bd)); verr.append(abs(d))
            print(f"   {e:>7.2f}{z:>8.2f}{rr:>7.1f}{bd:>11.4f}{be[0]:>13.4f}"
                  f"{d:>10.4f}")
    print("   " + "-" * 58)
    print(f"   worst emulator error {max(verr):.4f} dex, RMS "
          f"{math.sqrt(sum(v*v for v in verr)/len(verr)):.4f} dex")
    print(f"\n   CLAMPING AUDIT. What is tabulated is the correction")
    print(f"   Q = B_solved/B_algebraic, not B. Over the whole table:")
    print(f"     aniso  log10 Q in [{EMU.min():+.4f}, {EMU.max():+.4f}]  "
          f"= factor {10**EMU.min():.3f} .. {10**EMU.max():.3f}")
    print(f"     iso    log10 Q in [{EMU_ISO.min():+.4f}, {EMU_ISO.max():+.4f}]  "
          f"= factor {10**EMU_ISO.min():.3f} .. {10**EMU_ISO.max():.3f}")
    print(f"   so a query that falls off the grid inherits a bounded correction,")
    print(f"   not a constant value of the boost itself.")
    out["step1b_emulator"] = dict(mass_invariance_dex=inv,
                                  worst_error_dex=float(max(verr)),
                                  rms_error_dex=float(math.sqrt(
                                      sum(v * v for v in verr) / len(verr))),
                                  hz_over_Rd=HZ_OVER_RD,
                                  logQ_range_aniso=[float(EMU.min()), float(EMU.max())],
                                  logQ_range_iso=[float(EMU_ISO.min()), float(EMU_ISO.max())])

    head("STEP 1c  Fit the ANISOTROPIC tensor to ROTATION CURVES ONLY, on TRAIN")
    import mirror_run as MR
    gals = DAT.ingest(verbose=False)
    DAT.stratified_split(gals, verbose=False)
    gals = MR.build_draws(gals)
    tr = [g for g in gals if g.split == "train"]
    bl = [g for g in gals if g.split == "blind"]
    extrap = []

    def predict_aniso(p, d, tab=EMU):
        rt = MM.r_t_of(d["Mb_cat"][:, None], p["eta"], p["a0"])
        B, fo = emulate(tab, p["eta"], p["a0"], d["Mb_cat"][:, None], rt,
                        d["R"] * KPC, d["Rdisk"])
        extrap.append(fo)
        return d["gbar"] * B

    MR.MM.LAWS["aniso"] = dict(fn=None, free=("eta", "a0"), needs=())
    orig_predict = MR.predict

    def patched(law, p, d, mb_key="Mb_cat"):
        if law == "aniso":
            return predict_aniso(p, d)
        if law == "iso_tensor":
            return predict_aniso(p, d, EMU_ISO)
        return orig_predict(law, p, d, mb_key)

    MR.predict = patched
    MR.MM.LAWS["iso_tensor"] = dict(fn=None, free=("eta", "a0"), needs=())
    print("   coarse grid search first, then Nelder-Mead polish: an emulated")
    print("   surface is rugged enough that a bare simplex stops on a grid node.\n")
    print(f"   {'model':<16}{'free':>5}{'RMS train':>12}{'RMS blind':>12}"
          f"   fitted globals")
    print("   " + "-" * 72)
    fitted = {}
    for law in ("aniso", "iso_tensor"):
        best = None
        for e in (0.1, 0.2, 0.35, 0.6, 1.0, 1.8, 3.0, 5.0, 9.0):
            for la in np.arange(-10.6, -8.6, 0.25):
                v = MR.total_nll(law, [e, la], tr, "Mb_cat", False,
                                 MR.bounds_of(law))
                if best is None or v < best[0]:
                    best = (v, e, la)
        from scipy.optimize import minimize
        rr = minimize(lambda v: MR.total_nll(law, v, tr, "Mb_cat", False,
                                             MR.bounds_of(law)),
                      [best[1], best[2]], method="Nelder-Mead",
                      options=dict(maxiter=800, xatol=1e-4, fatol=1e-2))
        p = MR.unpack(law, rr.x, MR.bounds_of(law)) or dict(eta=best[1],
                                                            a0=10 ** best[2])
        mt, mb_ = MR.metrics(law, p, tr), MR.metrics(law, p, bl)
        bn = MR.bounds_of(law)
        rail = [k for k, v in p.items()
                if min(abs((math.log10(v) if k in MM.LOGPAR else v) - bn[k][0]),
                       abs((math.log10(v) if k in MM.LOGPAR else v) - bn[k][1]))
                < 0.02 * (bn[k][1] - bn[k][0])]
        eta_rail = p["eta"] > 0.9 * ETA_GRID[-1]
        fitted[law] = dict(params=p, train=mt, blind=mb_, railed=rail,
                           eta_at_emulator_edge=bool(eta_rail))
        print(f"   {law:<16}{2:>5}{mt['rms_dex']:>12.4f}{mb_['rms_dex']:>12.4f}"
              f"   eta={p['eta']:.4f}  a0={p['a0']:.4g}"
              + ("   RAILED: " + ",".join(rail) if rail else "")
              + ("   eta at emulator edge" if eta_rail else ""))
    print("   " + "-" * 72)
    print(f"   benchmarks: Newton blind 0.5544 | AQUAL simple blind 0.1590")
    print(f"   emulator extrapolation fraction over all likelihood calls: "
          f"{np.mean(extrap):.4f}")
    eta0, a00 = fitted["aniso"]["params"]["eta"], fitted["aniso"]["params"]["a0"]
    print(f"\n   FROZEN from here on: eta = {eta0:.4f}, a0 = {a00:.4g}")
    print(f"   Fitted on TRAIN rotation curves only. Not touched again.")
    out["step1c_aniso_fit"] = dict(fitted=fitted, eta_frozen=eta0, a0_frozen=a00,
                                   emulator_extrapolation_fraction=float(np.mean(extrap)))

    head("STEP 2  FROZEN PREDICTION of A_dyn = (g_R/g_R,N)/(K_z/K_z,N)")
    print("""   Nothing below is fitted. eta and a0 are the numbers above, taken from
   the rotation-curve fit on the TRAIN split and not touched again.\n""")
    print(f"   {'log Mb':>8}{'Rd kpc':>8}{'r_t kpc':>9}{'R/Rd':>7}"
          f"{'g_R/g_R,N':>12}{'K_z/K_z,N':>12}{'A_dyn':>9}   version")
    print("   " + "-" * 80)
    ad = []
    for lMb, Rd_ in ((9.5, 1.5), (10.3, 3.0), (11.0, 5.0)):
        Mb = 10 ** lMb
        hz = 0.15 * Rd_
        rt = float(MM.r_t_of(Mb, eta0, a00) / KPC)
        gN_, PsN, _ = solve_case(Mb, Rd_, hz, eta0, a00, "newton")
        Rk, gRN, KzN = profiles(gN_, PsN)
        for mode in ("aniso", "iso"):
            gg, Ps, cv = solve_case(Mb, Rd_, hz, eta0, a00, mode)
            _, gR, Kz = profiles(gg, Ps)
            for rr in (1.0, 2.2, 4.0):
                j = int(np.argmin(np.abs(Rk - rr * Rd_)))
                A = (gR[j] / gRN[j]) / (Kz[j] / KzN[j])
                ad.append(dict(logMb=lMb, Rd=Rd_, rt_kpc=rt, RoverRd=rr,
                               mode=mode, gR_ratio=float(gR[j] / gRN[j]),
                               Kz_ratio=float(Kz[j] / KzN[j]), A_dyn=float(A)))
                print(f"   {lMb:>8.1f}{Rd_:>8.1f}{rt:>9.2f}{rr:>7.1f}"
                      f"{gR[j]/gRN[j]:>12.4f}{Kz[j]/KzN[j]:>12.4f}{A:>9.4f}"
                      f"   {mode}")
    print("   " + "-" * 80)
    aa = [c["A_dyn"] for c in ad if c["mode"] == "aniso"]
    ai = [c["A_dyn"] for c in ad if c["mode"] == "iso"]
    print(f"   anisotropic A_dyn : {min(aa):.3f} .. {max(aa):.3f}   (claim: > 1)")
    print(f"   isotropic   A_dyn : {min(ai):.3f} .. {max(ai):.3f}   (claim: = 1)")
    out["step2_A_dyn"] = dict(rows=ad, aniso_range=[min(aa), max(aa)],
                              iso_range=[min(ai), max(ai)])

    head("STEP 3  DiskMass: the h_sigma_z / h_R test")
    dm = load_diskmass()
    print(f"   {len(dm)} galaxies with h_sigma_z, h_R(arcsec), h_R(kpc), h_z, "
          f"mu0_K, B-K\n")
    print("   CAVEAT 4 RESOLVED FIRST -- h_R appears in arcsec (VI) and kpc (VII)")
    print("   and the acquisition report flags them as unreconciled. Check:")
    imp = np.array([q["hR_as"] * q["D"] * 1e3 / 206265.0 for q in dm])
    tab = np.array([q["hR_kpc"] for q in dm])
    print(f"     h_R(arcsec) x D / 206265 vs h_R(kpc): median ratio "
          f"{np.median(imp/tab):.4f}, max |dev| {np.max(np.abs(imp/tab-1)):.4f}")
    print("     They ARE the same quantity. The arcsec pair is used from here")
    print("     on, so the ratio is distance-free.\n")
    print("   CAVEAT 3 QUANTIFIED -- h_z is inferred from h_R, so the two are")
    hz_ = np.array([q["hz_kpc"] for q in dm]); hR_ = np.array([q["hR_kpc"] for q in dm])
    cc = np.corrcoef(np.log10(hR_), np.log10(hz_))[0, 1]
    sl_hz = float(np.polyfit(np.log10(hR_), np.log10(hz_), 1)[0])
    print(f"     correlated by construction: r(log h_R, log h_z) = {cc:.5f}, "
          f"slope {sl_hz:.3f}")
    print("     h_z therefore contributes NO independent information and is used")
    print("     only as a constant multiplier, which cancels from a scale length.\n")

    rt_m = np.array([q["ratio"] for q in dm])
    er_m = np.array([q["e_ratio"] for q in dm])
    rng = np.random.default_rng(11)
    boot = np.array([np.median(rng.choice(rt_m, len(rt_m))) for _ in range(20000)])
    med, lo95, hi95 = np.median(rt_m), *np.percentile(boot, [2.5, 97.5])
    wmean = float(np.sum(rt_m / er_m ** 2) / np.sum(1 / er_m ** 2))
    werr = float(1 / math.sqrt(np.sum(1 / er_m ** 2)))
    print(f"   MEASURED h_sigma_z / h_R over {len(dm)} galaxies")
    print(f"     median {med:.3f}, bootstrap 95% CI [{lo95:.3f}, {hi95:.3f}]")
    print(f"     inverse-variance weighted mean {wmean:.3f} +- {werr:.3f}")
    print(f"     range {rt_m.min():.3f} .. {rt_m.max():.3f}, sd {rt_m.std(ddof=1):.3f}")
    print(f"     the values are NOT pinned at 2 h_R, so they were genuinely")
    print(f"     fitted and the test is live.")
    out["diskmass_measured"] = dict(n=len(dm), median=float(med),
                                    ci95=[float(lo95), float(hi95)],
                                    wmean=wmean, werr=werr,
                                    sd=float(rt_m.std(ddof=1)),
                                    hz_hR_corr=float(cc))

    print("\n   PREDICTIONS. Each galaxy's own h_R is used; M_b comes from the")
    print("   K-band photometry at a STATED Upsilon_K, with NO gas term because")
    print("   none is tabulated -- which makes M_b a lower bound and r_t an")
    print("   underestimate, so the isotropic boost is over-predicted if")
    print("   anything. Upsilon_K is scanned rather than chosen.\n")
    print(f"   {'Upsilon_K':>10}{'f_gas':>7}{'med log Mb':>12}"
          f"{'newton':>9}{'aniso':>9}{'iso':>9}{'meas-aniso':>12}{'meas-iso':>11}"
          f"{'sign test iso':>15}")
    print("   " + "-" * 96)
    pred = []
    RLO, RHI = 0.3, 2.0          # in units of h_R, the range DiskMass samples

    # EACH version predicts at ITS OWN frozen train-only fit. Evaluating the
    # isotropic version at the anisotropic version's parameters would be a
    # straw man, and the a0 sensitivity in step 4 is large enough to matter.
    P_ANI = (fitted["aniso"]["params"]["eta"], fitted["aniso"]["params"]["a0"])
    P_ISO = (fitted["iso_tensor"]["params"]["eta"],
             fitted["iso_tensor"]["params"]["a0"])
    print(f"   anisotropic predicts at its own fit  eta={P_ANI[0]:.4f} a0={P_ANI[1]:.4g}")
    print(f"   isotropic   predicts at its own fit  eta={P_ISO[0]:.4f} a0={P_ISO[1]:.4g}\n")

    def predict_all(ups, fgas, rlo=RLO, rhi=RHI, flare=None,
                    pani=None, piso=None):
        pani = P_ANI if pani is None else pani
        piso = P_ISO if piso is None else piso
        pn, pa, pi_, lmb = [], [], [], []
        for q in dm:
            Mb = stellar_mass(q, ups) / (1.0 - fgas)
            Rd_, hz = q["hR_kpc"], q["hz_kpc"]
            lmb.append(math.log10(Mb))
            gN_, PsN, _ = solve_case(Mb, Rd_, hz, pani[0], pani[1], "newton",
                                     nR=120, nz=60)
            Rk, _, KzN = profiles(gN_, PsN)
            fl = (np.exp(Rk / (flare * Rd_)) if flare else 1.0)   # h_z(R) flare
            pn.append(exp_scale(Rk, np.sqrt(KzN * fl), rlo * Rd_, rhi * Rd_) / Rd_)
            for mode, acc, pp in (("aniso", pa, pani), ("iso", pi_, piso)):
                gg, Ps, _ = solve_case(Mb, Rd_, hz, pp[0], pp[1], mode,
                                       nR=120, nz=60)
                _, _, Kz = profiles(gg, Ps)
                acc.append(exp_scale(Rk, np.sqrt(Kz * fl),
                                     rlo * Rd_, rhi * Rd_) / Rd_)
        return (np.array(pn), np.array(pa), np.array(pi_), np.array(lmb))

    for ups in (0.4, 0.6, 0.8):
        for fgas in (0.0, 0.3):
            pn, pa, pi_, lmb = predict_all(ups, fgas)
            d_a = rt_m - pa
            d_i = rt_m - pi_
            nb = int(np.sum(rt_m < pi_))
            row = dict(ups_K=ups, fgas=fgas, median_logMb=float(np.median(lmb)),
                       newton=float(np.median(pn)), aniso=float(np.median(pa)),
                       iso=float(np.median(pi_)),
                       median_resid_aniso=float(np.median(d_a)),
                       median_resid_iso=float(np.median(d_i)),
                       n_below_iso=nb, n=len(pn))
            pred.append(row)
            print(f"   {ups:>10.1f}{fgas:>7.1f}{np.median(lmb):>12.2f}"
                  f"{np.median(pn):>9.3f}{np.median(pa):>9.3f}"
                  f"{np.median(pi_):>9.3f}{np.median(d_a):>12.3f}"
                  f"{np.median(d_i):>11.3f}{f'{nb}/{len(pn)} below':>15}")
    print("   " + "-" * 96)
    out["diskmass_predictions"] = pred

    print("""
   NOTE ON THE NEWTONIAN COLUMN. The textbook razor-thin answer is exactly 2.0.
   The solver returns slightly more because the disk has finite thickness, the
   box is finite, and the exponential is fitted over 0.3-2.0 h_R rather than
   asymptotically. The offset applies to all three columns equally, so what
   matters is each model column against Newton, and all three against the
   measurement.""")

    head("STEP 3b  Systematics on the h_sigma_z/h_R test")
    print("""   Two things could move the measurement without any change in gravity:
   the radial range the exponential is fitted over, and disk FLARING, which
   the DiskMass data cannot constrain because h_z is a single inferred number
   per galaxy (caveat 3). Both are quantified rather than assumed away.\n""")
    print(f"   {'fit range (h_R)':>17}{'newton':>9}{'aniso':>9}{'iso':>9}")
    print("   " + "-" * 46)
    sysr = []
    for rlo, rhi in ((0.3, 2.0), (0.2, 1.5), (0.5, 2.5), (0.3, 3.0)):
        pn, pa, pi_, _ = predict_all(0.6, 0.0, rlo, rhi)
        sysr.append(dict(range=[rlo, rhi], newton=float(np.median(pn)),
                         aniso=float(np.median(pa)), iso=float(np.median(pi_))))
        print(f"   {f'{rlo}-{rhi}':>17}{np.median(pn):>9.3f}"
              f"{np.median(pa):>9.3f}{np.median(pi_):>9.3f}")
    print("   " + "-" * 46)
    print(f"\n   {'h_z flare length (h_R)':>24}{'newton':>9}{'aniso':>9}{'iso':>9}")
    print("   " + "-" * 53)
    sysf = []
    for fl in (None, 20.0, 8.0, 4.0, 2.0):
        pn, pa, pi_, _ = predict_all(0.6, 0.0, flare=fl)
        sysf.append(dict(flare_hR=fl, newton=float(np.median(pn)),
                         aniso=float(np.median(pa)), iso=float(np.median(pi_))))
        print(f"   {('none' if fl is None else f'{fl:.0f}'):>24}"
              f"{np.median(pn):>9.3f}{np.median(pa):>9.3f}{np.median(pi_):>9.3f}")
    print("   " + "-" * 53)
    print("""   Flaring only ever RAISES the predicted h_sigma_z, and it raises the
   Newtonian and anisotropic columns as much as the isotropic one. So no
   amount of flaring reconciles a measurement at ~2 with an isotropic
   prediction well above it: flaring moves the target the wrong way.""")
    out["diskmass_systematics"] = dict(fit_range=sysr, flaring=sysf)

    head("STEP 4  TRAP CHECK -- does this statistic respond to the parameters?")
    print("""   A prediction that does not move when the parameter moves cannot test
   the parameter. h_sigma_z/h_R is recomputed over a grid in eta and a0.\n""")
    print(f"   {'param':>7}{'value':>12}{'h_sz/h_R iso':>15}{'h_sz/h_R aniso':>17}"
          f"{'r_t kpc':>10}")
    print("   " + "-" * 62)
    q = dm[len(dm) // 2]
    Mb0 = stellar_mass(q, 0.6)
    Rd_, hz = q["hR_kpc"], q["hz_kpc"]
    sens = {}
    for par, vals in (("eta", (eta0 / 8, eta0 / 3, eta0, eta0 * 3, eta0 * 8)),
                      ("a0", (a00 / 25, a00 / 5, a00, a00 * 5, a00 * 25))):
        rows = []
        for v in vals:
            e, a = (v, a00) if par == "eta" else (eta0, v)
            gN_, PsN, _ = solve_case(Mb0, Rd_, hz, e, a, "newton", nR=120, nz=60)
            Rk, _, KzN = profiles(gN_, PsN)
            r2 = {}
            for mode in ("iso", "aniso"):
                gg, Ps, _ = solve_case(Mb0, Rd_, hz, e, a, mode, nR=120, nz=60)
                _, _, Kz = profiles(gg, Ps)
                r2[mode] = exp_scale(Rk, np.sqrt(Kz), RLO * Rd_, RHI * Rd_) / Rd_
            rt = float(MM.r_t_of(Mb0, e, a) / KPC)
            rows.append(dict(value=float(v), iso=r2["iso"], aniso=r2["aniso"],
                             rt_kpc=rt))
            print(f"   {par:>7}{v:>12.4g}{r2['iso']:>15.4f}{r2['aniso']:>17.4f}"
                  f"{rt:>10.3f}")
        spread_i = max(x["iso"] for x in rows) - min(x["iso"] for x in rows)
        spread_a = max(x["aniso"] for x in rows) - min(x["aniso"] for x in rows)
        sens[par] = dict(rows=rows, spread_iso=spread_i, spread_aniso=spread_a,
                         responsive_iso=bool(spread_i > 1e-9),
                         responsive_aniso=bool(spread_a > 1e-9))
        print(f"   -> spread: isotropic {spread_i:.4f}  anisotropic {spread_a:.4f}"
              f"   {'RESPONSIVE' if spread_i > 1e-9 else 'BLIND'}")
    print("   " + "-" * 62)
    out["step4_sensitivity"] = sens

    p = os.path.join(HERE, "mirror_adyn_results.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n   wrote {p}")
    return out


if __name__ == "__main__":
    main()
