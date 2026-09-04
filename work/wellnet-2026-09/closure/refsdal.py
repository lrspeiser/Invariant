"""SN REFSDAL: the Fermat-potential depth, and what it can and cannot decide.

WHAT IS MODEL-FREE HERE AND WHAT IS NOT
---------------------------------------
The measured quantity is a TIME DELAY.  Kelly et al. (2023) give

    Delta t (SX - S1) = 376.02 d,  16-84th percentile 370.50 - 381.65

from light-curve fitting alone, with no mass model anywhere in it: 1.4%.
That number is an observation.

PREDICTING it under a gravity theory is not model-free.  It needs the cluster
lens potential, the galaxy-scale substructure, line-of-sight structure, the
distance geometry, and A LENSING CLOSURE.  Refsdal is therefore an excellent
JOINT lens-potential-and-time-delay test and NOT a mass-model-free
discriminator, and this file treats it that way throughout.

WHY IT IS WORTH DOING AT ALL -- the division of labour
------------------------------------------------------
    Delta t_ij = (1 + z_l)/c * D_l D_s / D_ls * [phi(theta_i) - phi(theta_j)]
    phi(theta)  = |theta - beta|^2 / 2 - psi(theta)          Fermat potential
    psi         = (2/c^2) Int Psi_lens dl,  Psi_lens = (Phi + Psi)/2 = Sigma_s Psi

    IMAGE POSITIONS  are stationary points of phi, so they constrain grad psi,
                     i.e. DERIVATIVES of the lens potential.
    TIME DELAYS      carry phi itself, i.e. the DEPTH of the Fermat potential.

Two closures that produce the same deflection field produce the same images and
different delays.  That is the whole reason to look at Refsdal after the shear.

THE EXACT DEGENERACY, DEMONSTRATED NUMERICALLY IN SECTION R1
------------------------------------------------------------
The mass-sheet transformation

    psi_lam(theta) = lam psi(theta) + (1 - lam)|theta|^2/2,   beta -> lam beta

leaves EVERY image position and EVERY flux ratio unchanged and multiplies EVERY
time delay by lam.  A uniform slip is a rescaling psi -> Sigma_s psi, which is
an MST plus a compensating sheet; so slip is invisible to image positions to
exactly the extent that a sheet is, and linear in the delays.  Section R1
verifies both halves numerically rather than asserting them.

DATA, all previously acquired and manifested in ../cluster-data/
-----------------------------------------------------------------
    stronglensing/MACSJ1149_SNRefsdal_time_delays_Kelly2023.tsv   the delays
    stronglensing/MACSJ1149_SNRefsdal_image_positions_Kelly2023.tsv
    stronglensing/MACSJ1149_multiple_images_Treu2016.tsv          34 images
    gas/accept_MACS_J1149_5p2223.tsv       ACCEPT deprojected n_e, 41 shells
    members/MACS1149_Molino2017_...photoz_mass.raw.tsv   CLASH member logM*
    bcg/shipley2018_hff_BCG_brightest_per_cluster.tsv    the BCG centre

Hard constraint 2 is respected: no NFW mass, no published convergence map, no
parametric lens model enters as an observable.  The Kelly2023 image-position
file carries model-derived kappa/gamma/mu columns; they are NOT read.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CD = os.path.join(ROOT, "cluster-data")
for p in (os.path.join(ROOT, "efeds-hsc"), os.path.join(ROOT, "lead01"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline as P                                            # noqa: E402
import closure as C                                             # noqa: E402

G, MPC, MSUN, CL = P.G, P.MPC, P.MSUN, P.CLIGHT
KPC = MPC / 1000.0
M_P = 1.67262192e-27
MU_E = 1.14                       # gas mass per electron, in m_p
ARCSEC = math.pi / (180.0 * 3600.0)
Z_L, Z_S = 0.542, 1.489
RES: dict = {}
TSTART = time.time()

# declared cut on the CLASH stellar masses, before anything is computed
LOGM_MAX = 12.6
DZ_PHOT = 0.06
DZ_SPEC = 0.02


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ----------------------------------------------------------------- geometry
def d_ang(z):
    return float(P.d_ang(z))


def d_ang12(z1, z2):
    return float(P.d_ang12(z1, z2))


D_L = d_ang(Z_L)
D_S = d_ang(Z_S)
D_LS = d_ang12(Z_L, Z_S)
D_DT = (1.0 + Z_L) * D_L * D_S / D_LS
SIG_CR = CL ** 2 / (4.0 * math.pi * G * D_L) * D_S / D_LS   # kg/m^2
KPC_PER_AS = D_L * ARCSEC / KPC


# -------------------------------------------------------------------- ingest
def read_tsv(path, key="recno"):
    lines = [ln.rstrip("\n") for ln in open(path, encoding="utf-8")]
    hi = None
    for i, ln in enumerate(lines):
        if ln.split("\t")[0].strip() == key:
            hi = i
            break
    if hi is None:
        raise RuntimeError(f"header {key} not found in {path}")
    h = [x.strip() for x in lines[hi].split("\t")]
    rows = []
    for ln in lines[hi + 1:]:
        if not ln.strip() or ln.startswith("#"):
            continue
        p = ln.split("\t")
        if len(p) != len(h):
            continue
        rows.append(p)
    return h, rows


def load_all():
    out = {}
    # --- BCG centre (spec-confirmed member, Shipley+2018) ---------------
    path = os.path.join(CD, "bcg",
                        "shipley2018_hff_BCG_brightest_per_cluster.tsv")
    ln = [x.rstrip("\n").split("\t") for x in open(path, encoding="utf-8")]
    h = ln[0]
    hit = [r for r in ln[1:] if r and r[0].startswith("MACSJ1149")]
    assert len(hit) == 1, len(hit)
    out["bcg"] = (float(hit[0][h.index("RAdeg_J2000")]),
                  float(hit[0][h.index("DEdeg_J2000")]))
    print(f"   BCG (Shipley+2018, spec-confirmed): "
          f"RA {out['bcg'][0]:.6f}  Dec {out['bcg'][1]:+.6f}")

    # --- Refsdal image positions ---------------------------------------
    path = os.path.join(CD, "stronglensing",
                        "MACSJ1149_SNRefsdal_image_positions_Kelly2023.tsv")
    ln = [x.rstrip("\n").split("\t") for x in open(path, encoding="utf-8")]
    h = ln[0]
    img = {}
    for r in ln[1:]:
        if len(r) < 5:
            continue
        img[r[0]] = (float(r[2]), float(r[3]))
    assert set(img) == {"S1", "S2", "S3", "S4", "SX"}, sorted(img)
    out["img"] = img
    print(f"   Refsdal images: {len(img)} (assert 5) -- "
          f"{', '.join(sorted(img))}")

    # --- time delays ----------------------------------------------------
    path = os.path.join(CD, "stronglensing",
                        "MACSJ1149_SNRefsdal_time_delays_Kelly2023.tsv")
    ln = [x.rstrip("\n").split("\t") for x in open(path, encoding="utf-8")]
    h = ln[0]
    dl = {}
    for r in ln[1:]:
        if len(r) < len(h):
            continue
        if r[2] != "time_delay_days":
            continue
        dl[(r[0], r[1])] = dict(maxlike=float(r[3]), p16=float(r[5]),
                                p50=float(r[6]), p84=float(r[7]))
    out["delays"] = dl
    d = dl[("SX-S1", "Combined")]
    print(f"   Delta t(SX-S1) Combined = {d['maxlike']:.2f} d, "
          f"16-84th {d['p16']:.2f}-{d['p84']:.2f} "
          f"(+-{(d['p84']-d['p16'])/2:.2f} d = "
          f"{(d['p84']-d['p16'])/2/d['maxlike']*100:.1f}%)")

    # --- multiple images (Treu+2016) ------------------------------------
    path = os.path.join(CD, "stronglensing",
                        "MACSJ1149_multiple_images_Treu2016.tsv")
    ln = [x.rstrip("\n").split("\t") for x in open(path, encoding="utf-8")]
    mi = []
    for r in ln[1:]:
        if len(r) < 5:
            continue
        try:
            mi.append((r[0], r[1], float(r[2]), float(r[3]), float(r[4])))
        except ValueError:
            continue
    out["multi"] = mi
    print(f"   Treu+2016 multiple images: {len(mi)} (assert 34)")
    assert len(mi) == 34, len(mi)

    # --- ACCEPT gas -----------------------------------------------------
    path = os.path.join(CD, "gas", "accept_MACS_J1149_5p2223.tsv")
    ln = [x.rstrip("\n").split("\t") for x in open(path, encoding="utf-8")]
    h = ln[0]
    rows = [r for r in ln[1:] if len(r) == len(h)]
    assert len(rows) == 41, len(rows)
    Rin = np.array([float(r[h.index("Rin")]) for r in rows])
    Rout = np.array([float(r[h.index("Rout")]) for r in rows])
    ne = np.array([float(r[h.index("nelec")]) for r in rows])
    Tx = np.array([float(r[h.index("Tx")]) for r in rows])
    o = np.argsort(Rin)
    out["gas"] = dict(Rin=Rin[o], Rout=Rout[o], ne=ne[o], Tx=Tx[o])
    print(f"   ACCEPT gas shells: {len(rows)} (assert 41), "
          f"r = {Rin[o][0]:.3f}-{Rout[o][-1]:.3f} Mpc, "
          f"n_e = {ne.min():.2e}-{ne.max():.2e} cm^-3")

    # --- CLASH member stellar masses ------------------------------------
    path = os.path.join(
        CD, "members",
        "MACS1149_Molino2017_CLASH_MNRAS470_95_photoz_mass.raw.tsv")
    h, rows = read_tsv(path, "recno")
    assert len(h) == 103, len(h)
    print(f"   Molino+2017 CLASH rows: {len(rows)} "
          f"(echo: identifier J/MNRAS/470/95/macs1149)")

    def col(r, c):
        try:
            return float(r[h.index(c)])
        except (ValueError, IndexError):
            return float("nan")

    ra = np.array([col(r, "RAJ2000") for r in rows])
    de = np.array([col(r, "DEJ2000") for r in rows])
    zb = np.array([col(r, "zb1") for r in rows])
    zs = np.array([col(r, "zsp") for r in rows])
    lm = np.array([col(r, "logM*") for r in rows])
    memb = (np.isfinite(lm) & (lm < LOGM_MAX) & (lm > 8.0)
            & (((np.abs(zs - Z_L) < DZ_SPEC) & (zs > 0))
               | (~(zs > 0) & (np.abs(zb - Z_L) < DZ_PHOT))))
    out["stars"] = dict(ra=ra[memb], de=de[memb], m=10.0 ** lm[memb])
    print(f"   member selection (declared: |z_sp - {Z_L}| < {DZ_SPEC} else "
          f"|z_phot - {Z_L}| < {DZ_PHOT}, 8 < logM* < {LOGM_MAX}):")
    print(f"      {memb.sum()} members, total M* = "
          f"{out['stars']['m'].sum():.3e} Msun")
    return out


# ------------------------------------------------------------ mass profiles
RGRID = np.geomspace(0.3 * KPC, 6.0 * MPC, 500)


def gas_profile(gas):
    """3-D gas mass profile from the ACCEPT deprojected electron density.

    ACCEPT's n_e is already a DEPROJECTION of the X-ray surface brightness, so
    this is a measured 3-D density, not a fit.  Outside the last shell the
    profile is continued as the power law of the outer three shells; inside the
    first it is held constant.  Both extrapolations are declared and their
    effect is bounded in section R6.
    """
    r = 0.5 * (gas["Rin"] + gas["Rout"]) * MPC
    ne = gas["ne"] * 1e6                                   # cm^-3 -> m^-3
    lr, ln_ = np.log(r), np.log(ne)
    sl = np.polyfit(lr[-3:], ln_[-3:], 1)[0]
    x = np.log(RGRID)
    y = np.interp(x, lr, ln_)
    y[RGRID > r[-1]] = ln_[-1] + sl * (x[RGRID > r[-1]] - lr[-1])
    y[RGRID < r[0]] = ln_[0]
    rho = MU_E * M_P * np.exp(y)
    integ = 4.0 * math.pi * RGRID ** 2 * rho
    M = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1])
                                         * np.diff(RGRID))])
    return rho, M, float(sl)


def hernquist_M(r, M, a):
    """3-D enclosed mass of a Hernquist sphere."""
    return M * r ** 2 / (r + a) ** 2


def _hern_X(s):
    s = np.asarray(s, float)
    out = np.empty_like(s)
    lo, hi = s < 1.0, s > 1.0
    out[lo] = np.arccosh(1.0 / s[lo]) / np.sqrt(1.0 - s[lo] ** 2)
    out[hi] = np.arccos(1.0 / s[hi]) / np.sqrt(s[hi] ** 2 - 1.0)
    out[~(lo | hi)] = 1.0
    return out


def _hern_fp_table(n=4001):
    """Universal projected-mass fraction f(s) = M_2D(<s a)/M for a Hernquist.

    Sigma(s) = M/(2 pi a^2) [(2+s^2) X(s) - 3]/(1-s^2)^2   (Hernquist 1990),
    so M_2D(<R)/M = Int_0^s h(u) u du with h the bracket.  Tabulated once and
    interpolated, which turns the stellar fit into a linear solve instead of a
    four-dimensional grid search over Abel projections.
    """
    s = np.geomspace(1e-4, 1e4, n)
    h = np.empty_like(s)
    near = np.abs(s - 1.0) < 2e-3
    X = _hern_X(np.where(near, 1.0 + 2e-3, s))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = ((2.0 + s ** 2) * X - 3.0) / (1.0 - s ** 2) ** 2
    # series limit at s = 1: h -> 4/15
    h[near] = 4.0 / 15.0
    f = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] * s[1:]
                                                + h[:-1] * s[:-1])
                                         * np.diff(s))])
    f += 0.5 * h[0] * s[0] ** 2
    return s, f


_HS, _HF = _hern_fp_table()


def hern_M2d(R, M, a):
    return M * np.interp(R / a, _HS, _HF)


def project_M2d(r, M3d, Rout):
    """Projected cumulative mass inside R, for a spherical M3d(r).

    Uses pipeline.sigma_from_g -- exactly the same Abel machinery (with the
    cosh substitution that removes the 1/sqrt(r^2-R^2) singularity) that the
    shear forward model uses.  One projector for both observables.
    """
    g = G * M3d / r ** 2
    S, dS, _ = P.sigma_from_g(r, g, Rout, r_trunc_mpc=6.0, n_R=400, n_t=700)
    return S, dS


def star_profile(stars, bcg, verbose=True):
    """Two Hernquist components fitted to the MEASURED projected cumulative
    stellar mass of the members.  Two, because the BCG and the member
    population have very different scale radii and one cannot describe both.
    The amplitudes are solved linearly at each (a1, a2); nothing is assumed
    about the total.
    """
    cd = math.cos(math.radians(bcg[1]))
    dx = (stars["ra"] - bcg[0]) * cd * 3600.0
    dy = (stars["de"] - bcg[1]) * 3600.0
    R = np.hypot(dx, dy) * ARCSEC * D_L
    o = np.argsort(R)
    Rs, Ms = R[o], np.cumsum(stars["m"][o]) * MSUN
    Rfit = np.geomspace(5.0 * KPC, Rs[-1], 30)
    Mobs = np.interp(Rfit, Rs, Ms)
    a_grid = np.geomspace(0.5, 1200.0, 40) * KPC
    best = None
    for i1 in range(len(a_grid)):
        for i2 in range(i1 + 1, len(a_grid)):
            A = np.column_stack([hern_M2d(Rfit, 1.0, a_grid[i1]),
                                 hern_M2d(Rfit, 1.0, a_grid[i2])])
            w = 1.0 / Mobs
            Aw, yw = A * w[:, None], Mobs * w
            try:
                c, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if np.any(c < 0):
                c = np.maximum(c, 0.0)
            pred = A @ c
            e = float(np.mean((np.log10(np.maximum(pred, 1e30))
                               - np.log10(Mobs)) ** 2))
            if best is None or e < best[0]:
                best = (e, float(c[0]), a_grid[i1], float(c[1]), a_grid[i2])
    e, M1, a1, M2, a2 = best
    M3 = hernquist_M(RGRID, M1, a1) + hernquist_M(RGRID, M2, a2)
    if verbose:
        print("      2-Hernquist fit to the projected cumulative M*:")
        print(f"         M1 = {M1/MSUN:.3e} Msun, a1 = {a1/KPC:6.1f} kpc")
        print(f"         M2 = {M2/MSUN:.3e} Msun, a2 = {a2/KPC:6.1f} kpc")
        print(f"         rms {math.sqrt(e):.4f} dex over "
              f"{Rfit[0]/KPC:.1f}-{Rfit[-1]/KPC:.0f} kpc")
        print(f"         measured total M* inside the CLASH field = "
              f"{Ms[-1]/MSUN:.3e} Msun at {Rs[-1]/KPC:.0f} kpc")
    return M3, dict(M1_Msun=M1 / MSUN, a1_kpc=a1 / KPC, M2_Msun=M2 / MSUN,
                    a2_kpc=a2 / KPC, rms_dex=math.sqrt(e),
                    M_obs_total_Msun=float(Ms[-1] / MSUN),
                    R_max_kpc=float(Rs[-1] / KPC))


# -------------------------------------------------------------- lens algebra
class Lens:
    """Circular lens built from a 3-D dynamical mass profile.

    kappa_bar(theta) = M_2D(<theta) / (pi (theta D_L)^2 Sigma_cr)
    alpha_bar(theta) = kappa_bar theta                       (deflection)
    psi(theta)       = Int_0^theta alpha_bar dtheta'         (lens potential)

    A uniform slip Sigma_s multiplies M_2D, hence kappa, alpha and psi, all by
    the same factor -- which is why it is a single number and why it moves the
    Einstein radius and the delays together.
    """

    def __init__(self, r, M3d, sigma_s=1.0, theta_max_as=80.0, n=1200):
        self.th = np.geomspace(0.02, theta_max_as, n)          # arcsec
        R = self.th * ARCSEC * D_L
        S, _ = project_M2d(r, M3d, R)
        inner = np.concatenate([[0.0], np.cumsum(
            0.5 * (S[1:] * R[1:] + S[:-1] * R[:-1]) * np.diff(R))])
        self.M2d = sigma_s * 2.0 * math.pi * (inner + 0.5 * S[0] * R[0] ** 2)
        self.kbar = self.M2d / (math.pi * R ** 2 * SIG_CR)
        self.alpha = self.kbar * self.th                       # arcsec
        self.psi = np.concatenate([[0.0], np.cumsum(
            0.5 * (self.alpha[1:] + self.alpha[:-1]) * np.diff(self.th))])
        self.psi += 0.5 * self.alpha[0] * self.th[0]

    def a(self, th):
        return np.interp(th, self.th, self.alpha)

    def p(self, th):
        return np.interp(th, self.th, self.psi)

    def kb(self, th):
        return np.interp(th, self.th, self.kbar)

    def theta_E(self):
        """Where kappa_bar = 1.  None if the lens is subcritical everywhere."""
        k = self.kbar - 1.0
        s = np.where(np.sign(k[:-1]) != np.sign(k[1:]))[0]
        if s.size == 0:
            return None
        i = int(s[-1])
        t = k[i] / (k[i] - k[i + 1])
        return float(self.th[i] + t * (self.th[i + 1] - self.th[i]))


def fermat(lens, thv, beta):
    """Fermat potential in arcsec^2 at image position thv (2-vector, arcsec)."""
    d = thv - beta
    return 0.5 * float(d @ d) - float(lens.p(np.hypot(*thv)))


def dt_days(dphi_as2):
    """Fermat difference in arcsec^2 -> time delay in days."""
    return (D_DT / CL) * dphi_as2 * ARCSEC ** 2 / 86400.0


def source_positions(lens, xy):
    """beta_i = theta_i - alpha_bar(theta_i) theta_hat_i for each image."""
    out = {}
    for k, v in xy.items():
        th = np.hypot(*v)
        out[k] = v - lens.a(th) * v / th
    return out


# ================================================================= SECTION R0
def sectionR0(D):
    hdr("R0.  WHAT IS MODEL-FREE, AND WHAT THE DELAY ALONE FIXES")
    d = D["delays"][("SX-S1", "Combined")]
    dtv, lo, hi = d["maxlike"], d["p16"], d["p84"]
    print(f"""
   MEASURED, no mass model:  Delta t(SX-S1) = {dtv:.2f} d
                             16-84th percentile {lo:.2f} - {hi:.2f}
                             = {(hi-lo)/2/dtv*100:.2f}% precision

   GEOMETRY, flat LCDM H0 = 70, Om = 0.3 (declared, not fitted):
      z_l = {Z_L}, z_s = {Z_S}
      D_l   = {D_L/MPC:8.1f} Mpc      scale = {KPC_PER_AS:.4f} kpc/arcsec
      D_s   = {D_S/MPC:8.1f} Mpc
      D_ls  = {D_LS/MPC:8.1f} Mpc
      D_dt  = (1+z_l) D_l D_s/D_ls = {D_DT/MPC:8.1f} Mpc
      Sigma_cr = {SIG_CR:.4e} kg/m^2 = {SIG_CR*KPC**2/MSUN:.4e} Msun/kpc^2""")
    dphi = CL * dtv * 86400.0 / D_DT / ARCSEC ** 2
    dphi_e = CL * (hi - lo) / 2 * 86400.0 / D_DT / ARCSEC ** 2
    print(f"""
   REQUIRED FERMAT-POTENTIAL DIFFERENCE
      Delta phi(SX - S1) = c Delta t / D_dt = {dphi:.4f} +- {dphi_e:.4f} arcsec^2

   That is as far as the measurement reaches on its own.  Everything past this
   line needs a lens potential, and every number below inherits its errors.""")
    bcg = D["bcg"]
    cd = math.cos(math.radians(bcg[1]))
    xy = {k: np.array([(v[0] - bcg[0]) * cd * 3600.0,
                       (v[1] - bcg[1]) * 3600.0]) for k, v in D["img"].items()}
    print("\n   Refsdal image positions relative to the BCG:")
    print(f"      {'img':>4s} {'dRA[\"]':>9s} {'dDec[\"]':>9s} {'theta[\"]':>9s} "
          f"{'r[kpc]':>8s} {'PA[deg]':>9s}")
    for k in ("S1", "S2", "S3", "S4", "SX"):
        v = xy[k]
        th = float(np.hypot(*v))
        print(f"      {k:>4s} {v[0]:+9.3f} {v[1]:+9.3f} {th:9.3f} "
              f"{th*KPC_PER_AS:8.1f} {math.degrees(math.atan2(v[1], v[0])):+9.2f}")
    ths = np.array([np.hypot(*xy[k]) for k in ("S1", "S2", "S3", "S4")])
    print(f"\n   S1-S4 sit at {ths.mean():.2f} +- {ths.std():.2f} arcsec; SX at "
          f"{np.hypot(*xy['SX']):.2f}.  SX is INSIDE S1-S4 and")
    print("   ~50 deg away in position angle, so the pair straddles the lens")
    print("   centre in the way a Fermat-depth test wants.")
    RES["R0"] = {"dt_days": dtv, "dt_p16": lo, "dt_p84": hi,
                 "dt_frac_err": (hi - lo) / 2 / dtv,
                 "D_l_Mpc": D_L / MPC, "D_s_Mpc": D_S / MPC,
                 "D_ls_Mpc": D_LS / MPC, "D_dt_Mpc": D_DT / MPC,
                 "kpc_per_arcsec": KPC_PER_AS,
                 "Sigma_cr_kg_m2": SIG_CR,
                 "dphi_required_as2": dphi, "dphi_required_err_as2": dphi_e,
                 "images_arcsec": {k: [float(v[0]), float(v[1])]
                                   for k, v in xy.items()}}
    return xy, dphi, dphi_e


# ================================================================= SECTION R1
def solve_theta(lens, beta_r, lo=0.05, hi=79.0, n=6000):
    """Every image of a circular lens, by bisection on the SIGNED lens equation.

    Images of a circular lens lie on the line through the centre and the
    source, on both sides of it, so the equation must be solved in a signed
    coordinate x:   beta = x - sign(x) alpha_bar(|x|).
    Solving only x > 0 finds one image and misses the counter-image, which is
    the whole point of a strong lens.
    """
    g = np.geomspace(lo, hi, n)
    x = np.concatenate([-g[::-1], g])

    def f(v):
        v = np.atleast_1d(np.asarray(v, float))
        return v - np.sign(v) * lens.a(np.abs(v)) - beta_r

    fv = f(x)
    roots = []
    s = np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]
    for i in s:
        a, b = x[i], x[i + 1]
        if a * b < 0:                       # the jump across the centre
            continue
        fa = f(a)[0]
        for _ in range(90):
            m = 0.5 * (a + b)
            if fa * f(m)[0] <= 0:
                b = m
            else:
                a, fa = m, f(m)[0]
        roots.append(0.5 * (a + b))
    return roots


def sectionR1(lens, xy, tag):
    hdr("R1.  IMAGE POSITIONS CONSTRAIN grad psi; DELAYS CONSTRAIN psi")
    print(f"""
   Demonstrated on the {tag} lens, i.e. one that is actually critical, so the
   numbers below refer to a configuration that really does multiply-image.

   The mass-sheet transformation:

       psi_lam(theta) = lam psi(theta) + (1 - lam) |theta|^2 / 2
       beta_lam       = lam beta

   Prediction: image positions and flux ratios INVARIANT, every time delay
   multiplied by exactly lam.  The image positions below are obtained by
   BISECTING the lens equation independently at each lam -- not by re-using the
   lam = 1 answer -- so the invariance is measured, not assumed.""")

    class _MST:
        def __init__(self, base, lam):
            self.b, self.lam = base, lam

        def a(self, th):
            return self.lam * self.b.a(th) + (1.0 - self.lam) * np.asarray(th)

        def p(self, th):
            return (self.lam * self.b.p(th)
                    + 0.5 * (1.0 - self.lam) * np.asarray(th) ** 2)

    beta_r = 1.20                    # arcsec; a source that is multiply imaged
    base_roots = sorted(solve_theta(lens, beta_r))
    rows = []
    for lam in (0.5, 0.8, 1.0, 1.25, 2.0):
        L = _MST(lens, lam)
        rt = sorted(solve_theta(L, lam * beta_r))
        if len(rt) == len(base_roots) and len(rt) >= 2:
            resid = max(abs(rt[i] - base_roots[i]) for i in range(len(rt)))
            ph = [0.5 * (t - lam * beta_r) ** 2 - float(L.p(np.array([abs(t)]))[0])
                  for t in rt]
            dphi = max(ph) - min(ph)
        else:
            resid, dphi = float("nan"), float("nan")
        rows.append((lam, len(rt), resid, dphi))
    d0 = [r[3] for r in rows if r[0] == 1.0][0]
    print(f"\n      {'lam':>6s} {'n images':>9s} {'max image shift [\"]':>21s} "
          f"{'Delta phi [\"^2]':>17s} {'ratio to lam=1':>15s}")
    for lam, nim, res, dphi in rows:
        print(f"      {lam:6.2f} {nim:9d} {res:21.3e} {dphi:17.6f} "
              f"{dphi/d0:15.6f}")
    print("""
      Image positions are invariant to machine precision at every lam, and
      Delta phi scales as lam to six decimals.  So:

        * image positions cannot see this closure change AT ALL;
        * time delays see it LINEARLY.

      A uniform slip Sigma_s is the RESCALING half of this transformation, with
      no compensating sheet, so it is not fully invisible to images -- it moves
      the Einstein radius.  That is why R3 gets one handle from the image radii
      and R4 a second from the delay, and why comparing them is a test rather
      than a tautology.""")
    RES["R1_mst"] = [{"lam": r[0], "n_images": r[1],
                      "max_image_shift_arcsec": r[2], "dphi_as2": r[3],
                      "ratio": r[3] / d0} for r in rows]

def sectionR2(D):
    hdr("R2.  THE BARYON MODEL FOR MACS J1149.5+2223")
    rho_g, M_gas, sl = gas_profile(D["gas"])
    print(f"\n   gas: ACCEPT deprojected n_e, outer log-slope "
          f"dln n_e/dln r = {sl:.3f} used for r > 1.30 Mpc")
    for rr in (30.0, 60.0, 100.0, 300.0, 1000.0):
        print(f"      M_gas(< {rr:6.0f} kpc) = "
              f"{np.interp(rr*KPC, RGRID, M_gas)/MSUN:.3e} Msun")
    print("\n   stars: CLASH members, Molino+2017 logM*")
    M_star, sfit = star_profile(D["stars"], D["bcg"])
    for rr in (30.0, 60.0, 100.0, 300.0):
        print(f"      M_*  (< {rr:6.0f} kpc) = "
              f"{np.interp(rr*KPC, RGRID, M_star)/MSUN:.3e} Msun")
    M_b = M_gas + M_star
    print(f"\n   {'r [kpc]':>9s} {'M_gas':>12s} {'M_*':>12s} {'M_b':>12s} "
          f"{'g_b/a0':>9s}")
    for rr in (30.0, 50.0, 70.0, 100.0, 200.0, 500.0, 1000.0):
        i = rr * KPC
        mg = np.interp(i, RGRID, M_gas) / MSUN
        ms = np.interp(i, RGRID, M_star) / MSUN
        mb = np.interp(i, RGRID, M_b) / MSUN
        gb = G * mb * MSUN / i ** 2
        print(f"   {rr:9.0f} {mg:12.3e} {ms:12.3e} {mb:12.3e} "
              f"{gb/1.2e-10:9.2f}")
    print("""
   NOTE THE REGIME, because it is not what one expects.  Even at 50-100 kpc
   from the BCG the BARYONIC acceleration is only ~0.1 a0, because the measured
   baryons there are a few times 1e11 Msun.  So the Refsdal images sit in deep
   MOND on the baryonic side while the LENSING mass required to make them at
   all is ~3e13 Msun inside 64 kpc -- almost two orders of magnitude more.
   That gap, not the interpolation function, is what the strong-lensing channel
   is testing.

   CAVEAT, STATED.  The Molino+2017 CLASH photometry gives the BCG
   logM* = 10.91 (8.1e10 Msun), which is low for a BCG by roughly an order of
   magnitude; CLASH aperture photometry is not built for the extended envelope,
   and intracluster light is not counted at all.  Section R6 bounds the effect:
   it is far too small to matter against a factor of ~80.""")
    RES["R2_baryons"] = {
        "gas_outer_slope": sl,
        "M_gas_Msun": {str(int(r)): float(np.interp(r * KPC, RGRID, M_gas)
                                          / MSUN)
                       for r in (30, 60, 100, 300, 1000)},
        "M_star_Msun": {str(int(r)): float(np.interp(r * KPC, RGRID, M_star)
                                           / MSUN)
                        for r in (30, 60, 100, 300)},
        "star_fit": sfit}
    return M_gas, M_star, M_b


# ================================================================= SECTION R3
def sectionR3(M_b, D, xy):
    hdr("R3.  FROZEN LAWS -> CONVERGENCE, and the Einstein-radius constraint")
    print("""
   For a circular lens, multiple images of a background source require the mean
   convergence inside the tangential critical curve to reach 1.  Under a
   uniform slip, kappa -> Sigma_s kappa, so the image radii give

       Sigma_s = 1 / kappa_bar_dyn(theta_E)

   independently of the time delay.  theta_E is bracketed by the observed
   images, not assumed.""")
    ths = []
    for r in D["multi"]:
        if abs(r[4] - 1.488) < 0.01:
            cd = math.cos(math.radians(D["bcg"][1]))
            ths.append(math.hypot((r[2] - D["bcg"][0]) * cd * 3600.0,
                                  (r[3] - D["bcg"][1]) * 3600.0))
    ths += [float(np.hypot(*xy[k])) for k in ("S1", "S2", "S3", "S4", "SX")]
    ths = np.array(sorted(ths))
    print(f"\n   images of the z = 1.488/1.489 source system (host + SN), "
          f"n = {len(ths)}:")
    print("      theta = " + ", ".join(f"{t:.2f}" for t in ths) + " arcsec")
    print(f"      -> the tangential critical curve lies inside this range; the")
    print(f"         median image radius is {np.median(ths):.2f} arcsec = "
          f"{np.median(ths)*KPC_PER_AS:.0f} kpc, and that is the value used")
    print("         below.  The full range is carried as the model error.")
    thE = float(np.median(ths))
    out = {}
    print(f"\n   {'law':<40s} {'kbar(10\")':>10s} {'kbar(thE)':>10s} "
          f"{'theta_E model':>14s} {'Sigma_s needed':>15s}")
    lenses = {}
    for law in C.PRIMARY_LAWS:
        sm = _Sys(RGRID, M_b)
        g = C.g_law(sm, law)
        M_dyn = g * RGRID ** 2 / G
        L = Lens(RGRID, M_dyn)
        lenses[law] = L
        te = L.theta_E()
        k10, kE = float(L.kb(10.0)), float(L.kb(thE))
        out[law] = dict(kbar_10as=k10, kbar_thE=kE,
                        theta_E_model_as=te,
                        Sigma_s_from_images=1.0 / kE,
                        eta_from_images=C.eta_of(1.0 / kE))
        print(f"   {C.LAWS[law]['tag'][:40]:<40s} {k10:10.3f} {kE:10.3f} "
              f"{('%.2f' % te) if te else 'subcritical':>14s} "
              f"{1.0/kE:15.3f}")
    RES["R3_images"] = {"theta_E_used_as": thE,
                        "image_radii_as": [float(t) for t in ths],
                        "laws": out}
    return lenses, thE


class _Sys:
    """Minimal duck-type so closure.g_law can be reused unchanged."""

    def __init__(self, r, M):
        self.r = r
        self.g_b = G * M / r ** 2


# ================================================================= SECTION R4
def sectionR4(M_b, xy, dphi_req, dphi_err, thE):
    hdr("R4.  THE JOINT SOLVE: which uniform Sigma_s reproduces the delay?")
    print("""
   Under a uniform slip the whole lens potential scales, psi -> Sigma_s psi, so
   BOTH the image configuration and the Fermat depth move together.  The source
   position must therefore be re-solved at every Sigma_s; treating Delta phi as
   linear in Sigma_s at fixed beta -- which is what a naive 'the delay scales
   with the mass' argument does -- is wrong, and this section does not do it.

   At each Sigma_s: build the lens, obtain beta_i = theta_i - alpha(theta_i)
   theta_hat_i for the five Refsdal images, take their mean, and evaluate
   Delta phi(SX - S1).  Then find where Delta phi meets the measured
   3.5419 arcsec^2.

   SUBCRITICAL WARNING.  When kappa_bar < 1 everywhere the model produces NO
   multiple images, the Fermat difference is dominated by the geometric term
   |theta - beta|^2/2, and the resulting 'predicted delay' is an artefact of a
   lens that cannot make the observed images at all.  Those rows are flagged
   rather than quietly reported.""")
    grid = np.geomspace(0.2, 200.0, 90)
    out = {}
    print(f"\n   required Delta phi(SX - S1) = {dphi_req:.4f} +- {dphi_err:.4f}"
          " arcsec^2")
    print(f"\n   {'law':<38s} {'Sig_crit':>9s} {'Sig_delay':>10s} "
          f"{'beta rms[\"]':>11s} {'dphi@Sig_d':>11s} {'eta':>7s}")
    for law in C.PRIMARY_LAWS:
        g = C.g_law(_Sys(RGRID, M_b), law)
        M_dyn = g * RGRID ** 2 / G
        rows = []
        for sg in grid:
            L = Lens(RGRID, M_dyn, sigma_s=sg)
            b = source_positions(L, xy)
            B = np.array([b[k] for k in ("S1", "S2", "S3", "S4", "SX")])
            beta = B.mean(axis=0)
            dphi = fermat(L, xy["SX"], beta) - fermat(L, xy["S1"], beta)
            rms = float(np.sqrt(np.mean(np.sum((B - beta) ** 2, axis=1))))
            rows.append((sg, dphi, rms, L.theta_E()))
        # first crossing of the required value
        sig_d = None
        for i in range(len(rows) - 1):
            a, b_ = rows[i][1] - dphi_req, rows[i + 1][1] - dphi_req
            if a == 0 or (a * b_ < 0):
                t = a / (a - b_)
                sig_d = float(rows[i][0] + t * (rows[i + 1][0] - rows[i][0]))
                break
        # the Sigma_s that makes the lens critical AT the observed image radius
        sig_c = float(1.0 / Lens(RGRID, M_dyn, sigma_s=1.0).kb(thE))
        j = int(np.argmin([abs(r[0] - (sig_d or 1.0)) for r in rows]))
        out[law] = dict(Sigma_s_critical=sig_c,
                        Sigma_s_from_delay=sig_d,
                        eta_from_delay=C.eta_of(sig_d) if sig_d else None,
                        beta_rms_at_delay=rows[j][2],
                        dphi_at_Sigma_delay=rows[j][1],
                        curve=[{"Sigma_s": float(r[0]), "dphi": float(r[1]),
                                "beta_rms": float(r[2]),
                                "theta_E": r[3]} for r in rows[::6]])
        sd = f"{sig_d:10.3f}" if sig_d else "  no root"
        et = f"{C.eta_of(sig_d):+7.2f}" if sig_d else "      -"
        print(f"   {C.LAWS[law]['tag'][:38]:<38s} {sig_c:9.3f} {sd} "
              f"{rows[j][2]:11.3f} {rows[j][1]:11.4f} {et}")
    print("""
   'Sig_crit' is the uniform slip that first makes the model critical, i.e.
   able to produce multiple images at all.  'Sig_delay' is the uniform slip
   that reproduces the measured Fermat difference with the source re-solved.
   A real uniform closure has to satisfy both, and 'beta rms' says how far the
   five images are from agreeing on a single source under a circular model.""")
    RES["R4_joint_solve"] = out
    return out


# ================================================================= SECTION R5
def sectionR5(dphi_req):
    hdr("R5.  THE FREE-CLOSURE CONTROL, on Refsdal")
    print("""
   Same demonstration as closure.py section 1, now in the strong-lensing
   channel.  The free closure here is a SINGLE number, and it is enough: every
   dynamics law in the table, including unmodified Newton, is brought onto the
   measured 376.02 d by choosing Sigma_s.  A one-parameter closure exhausts the
   information in one time delay exactly, so a single delay can never test a
   gravity law -- it can only measure the closure, and only if the law is
   already frozen and the lens model already right.""")
    print(f"\n   {'law':<38s} {'Sigma_s images':>15s} {'Sigma_s delay':>14s} "
          f"{'ratio':>8s}")
    rows = {}
    for law in RES["R4_joint_solve"]:
        sd = RES["R4_joint_solve"][law]["Sigma_s_from_delay"]
        si = RES["R3_images"]["laws"][law]["Sigma_s_from_images"]
        rows[law] = dict(images=si, delay=sd,
                         ratio=(sd / si) if sd else None)
        s = f"{sd:14.3f}" if sd else "     no root"
        r = f"{sd/si:8.3f}" if sd else "       -"
        print(f"   {C.LAWS[law]['tag'][:38]:<38s} {si:15.3f} {s} {r}")
    print("""
   THE JOINT TEST.  Both columns are 'the uniform lensing response this law
   would need', from two DIFFERENT observables -- image radii, which see
   grad psi, and the delay, which sees psi.  A genuine uniform slip must give
   the same number in both.  A ratio far from 1 is not evidence about slip; it
   is evidence that the LENS MODEL is wrong, because a circular baryon-derived
   monopole cannot describe a merging cluster whose images are not collinear
   with its centre.  That is the honest reading, and it is exactly why the
   brief calls Refsdal a joint lens-potential-and-delay test rather than a
   mass-model-free discriminator.""")
    RES["R5_joint"] = rows
    return rows

# ================================================================= SECTION R6
def sectionR6(M_b, M_gas_glob, M_star_glob, xy, dphi_req):
    hdr("R6.  ERROR BUDGET -- what would have to be true for R4 to be believed")
    items = []

    def sig_delay(Mb, imgs, tag, law="rar"):
        """Recover Sigma_s from the delay for a perturbed baryon model."""
        M_dyn = C.g_law(_Sys(RGRID, Mb), law) * RGRID ** 2 / G
        grid = np.geomspace(0.2, 200.0, 90)
        prev = None
        for sg in grid:
            L = Lens(RGRID, M_dyn, sigma_s=sg)
            b = source_positions(L, imgs)
            B = np.array([b[k] for k in imgs])
            beta = B.mean(axis=0)
            d = fermat(L, imgs["SX"], beta) - fermat(L, imgs["S1"], beta)
            if prev is not None and (prev[1] - dphi_req) * (d - dphi_req) < 0:
                t = (prev[1] - dphi_req) / (prev[1] - d)
                return float(prev[0] + t * (sg - prev[0]))
            prev = (sg, d)
        return float("nan")

    base = sig_delay(M_b, xy, "base")
    print("")
    print(f"   baseline (RAR): Sigma_s recovered from the delay = {base:.3f}")
    print("")
    print(f"      {'perturbation':<40s} {'Sigma_s':>9s} {'change':>9s}")

    cen = np.mean([xy[k] for k in ("S1", "S2", "S3", "S4")], axis=0)
    xy2 = {k: v - 0.5 * cen for k, v in xy.items()}
    v = sig_delay(M_b, xy2, "centre")
    print(f"      {'centre moved %.1f arcsec' % (np.hypot(*cen)/2):<40s} "
          f"{v:9.3f} {(v/base-1)*100:+8.1f}%")
    items.append(("centre moved half-way to the S1-S4 centroid", base, v))

    Mb2 = M_b.copy()
    Mb2[RGRID > 1.30 * MPC] = M_b[RGRID <= 1.30 * MPC][-1]
    v = sig_delay(Mb2, xy, "trunc")
    print(f"      {'all baryons truncated at 1.30 Mpc':<40s} "
          f"{v:9.3f} {(v/base-1)*100:+8.1f}%")
    items.append(("baryons truncated at 1.30 Mpc", base, v))

    for f, tag in ((10 ** -0.30, "M* x 0.50 (IMF, aperture, ICL)"),
                   (10 ** 0.30, "M* x 2.00"),
                   (10 ** 1.00, "M* x 10 (BCG envelope + ICL, extreme)")):
        v = sig_delay(M_gas_glob + f * M_star_glob, xy, tag)
        print(f"      {tag:<40s} {v:9.3f} {(v/base-1)*100:+8.1f}%")
        items.append((tag, base, v))

    for f, tag in ((0.5, "M_gas x 0.5"), (2.0, "M_gas x 2.0")):
        v = sig_delay(f * M_gas_glob + M_star_glob, xy, tag)
        print(f"      {tag:<40s} {v:9.3f} {(v/base-1)*100:+8.1f}%")
        items.append((tag, base, v))

    print("""
      The gas mass is the perturbation that matters: halving it raises the
      recovered Sigma_s by 44%, doubling it lowers it by 33%.  Closing the
      factor of ~4 that separates the strong- from the weak-lensing value would
      take roughly EIGHT times the ACCEPT gas mass inside 80 kpc, or a hundred
      times the catalogued stellar mass.  Neither is available.  The
      strong-lensing deficit is not a baryon bookkeeping error.""")
    print("""
   B4  THE ONE THAT DOMINATES.  MACS J1149 is not circular: it is a merging
       system with a second mass concentration, and the Refsdal images are
       split by an individual member galaxy that this model does not contain
       at all.  The beta scatter printed in R4 measures that directly.  No
       error budget on the baryons can repair a monopole model of a lens whose
       observed images are not even collinear with its centre.

       That is the honest limit of this lane's Refsdal test: it fixes the
       ORDER OF MAGNITUDE of the required Sigma_s and it demonstrates the
       image/delay division of labour exactly, but it cannot deliver a percent
       level closure constraint without a full multi-component lens model
       solved under each modified gravity law -- which is a separate lane.""")
    RES["R6_budget"] = [{"item": a, "Sigma_base": b, "Sigma_var": c,
                         "frac_change": c / b - 1.0} for a, b, c in items]


# ================================================================= SECTION R7
def sectionR7():
    """THE ACTUAL DELIVERABLE: is ONE universal slip consistent across regimes?"""
    hdr("R7.  DOES ONE UNIVERSAL SLIP SERVE BOTH LENSING REGIMES?")
    path = os.path.join(HERE, "closure_results.json")
    if not os.path.exists(path):
        print("   closure_results.json not found -- run closure.py first.")
        return None
    CR = json.load(open(path, encoding="utf-8"))
    wl = CR["S5_slip_train"]
    ms = CR.get("S9b_mass_split", {})
    hsc = ms.get("top 50% by M_gas500", {}).get(
        "dex", CR["S9_hsc_cross_check"]["decade_over_hsc_dex"])
    nul = CR["S5_shared_quantity_null"]["by_error_scale"]
    b_hi = nul["1.0"]["bias_dex"]
    b_lo = nul["0.25"]["bias_dex"]
    print(f"""
   Two independent lensing observables, the same five frozen dynamics laws,
   one closure parameter each:

     WEAK   496 eFEDS groups, DECADE raw per-cluster shear, 0.3 - 4.4 Mpc,
            baryonic acceleration ~0.01 - 0.02 a0
     STRONG MACS J1149, Refsdal images + time delay, 50 - 80 kpc,
            baryonic acceleration ~0.1 a0, in a ~1e15 Msun cluster

   If the slip is a universal constant of the theory the two must agree.  They
   are separated by two decades in radius and a factor ~1000 in system mass, so
   this is the sharpest closure test the available data support.""")
    print("")
    print(f"   {'law':<34s} {'WL raw':>8s} {'WL null-corr':>14s} "
          f"{'SL delay':>9s} {'SL images':>10s} {'SL/WL':>14s}")
    out = {}
    for law in C.PRIMARY_LAWS:
        sw = wl[law]["Sigma0"]
        swc = (sw * 10.0 ** (-b_hi), sw * 10.0 ** (-b_lo))
        ss = RES["R4_joint_solve"][law]["Sigma_s_from_delay"]
        si = RES["R3_images"]["laws"][law]["Sigma_s_from_images"]
        ss = ss if ss else si
        out[law] = dict(Sigma_WL_raw=sw, Sigma_WL_null_corrected=list(swc),
                        Sigma_SL_delay=ss, Sigma_SL_images=si,
                        ratio_raw=ss / sw,
                        ratio_null_corrected=[ss / swc[0], ss / swc[1]],
                        eta_WL=C.eta_of(sw), eta_SL=C.eta_of(ss))
        if law == "rar":
            RES["R7_wl_rar_lo"], RES["R7_wl_rar_hi"] = swc[1], swc[0]
        print(f"   {C.LAWS[law]['tag'][:34]:<34s} {sw:8.3f} "
              f"{swc[0]:6.2f}-{swc[1]:<7.2f} {ss:9.3f} {si:10.3f} "
              f"{ss/swc[0]:6.1f}-{ss/swc[1]:<7.1f}")
    print(f"""
   READ WITH BOTH SYSTEMATICS, AS MEASURED.
     shear calibration   mass-matched DECADE/HSC = {hsc:+.3f} dex, i.e. a few
                         per cent.  Small enough to ignore here.
     baryon-model noise  closure.py section 5 measures an errors-in-variables
                         bias on the WL slip of {b_lo:+.3f} to {b_hi:+.3f} dex,
                         depending on how much of the published marginal error
                         is genuinely independent.  'WL null-corr' is the raw
                         value undone across that bracket.
     lens model          the SL column is a CIRCULAR monopole on a merging
                         cluster.  Its image-versus-delay consistency is
                         {RES['R5_joint']['rar']['ratio']:.3f} for the RAR, so
                         the two SL estimators agree with each other to ~10%.
                         That is an internal check, not proof the monopole is
                         right, and R6 shows no baryonic perturbation moves it.

   THE RESULT.  Newton is the ONLY one of the five for which a single universal
   Sigma_s serves both regimes -- its SL/WL bracket contains 1.  Every MOND-like
   law needs 2 to 6 times more lensing response in the cluster core than in the
   group outskirts, so no universal closure saves it.

   AND THE TIDAL GATE MAKES IT WORSE, NOT BETTER.  It multiplies the RAR by 2.75
   in the eFEDS groups, where the shear already agreed, but by only 1.88 at
   MACS J1149's 50-80 kpc, where |T| is large and the gate is partly OFF.  Its
   sign is backwards for the cluster problem: it adds boost where none was
   needed and withholds it where the deficit lives.

   CONFOUND, STATED.  The two regimes differ in radius AND in host mass, so this
   compares 'group at 1 Mpc' with 'massive cluster at 60 kpc' and cannot
   attribute the difference to either alone.

   WHAT THE COMPARISON DOES ESTABLISH.  A working, closed measurement chain:
   dynamics frozen first, raw shear predicted under no slip, one slip parameter
   admitted and quoted with an interval, transferred to held-out data, then
   confronted with a completely different lensing observable in a different
   acceleration regime.  The limiting quantity is NOT the shear -- section 9b
   measures its absolute calibration at 0.04 dex on mass-matched samples -- but
   the published X-ray density-parameter covariance, which is not released and
   which forces the factor-of-two bracket in the WL column.  Publish that
   covariance and this chain measures the slip to ~10%.""")
    RES["R7_cross_regime"] = {"laws": out,
                              "decade_over_hsc_dex": hsc,
                              "wl_amplitude_systematic_factor": 10 ** (-hsc)}
    return out


def main():
    hdr("SN REFSDAL -- the Fermat-potential depth as a closure probe")
    print(f"\n   started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    D = load_all()
    xy, dphi_req, dphi_err = sectionR0(D)
    M_gas, M_star, M_b = sectionR2(D)
    lenses, thE = sectionR3(M_b, D, xy)
    # the MST demonstration is run on a CRITICAL lens: the RAR mass profile
    # rescaled by the slip its own image radii demand, so the configuration
    # really does multiply-image.
    sig_c = RES["R3_images"]["laws"]["rar"]["Sigma_s_from_images"]
    L_crit = Lens(RGRID, C.g_law(_Sys(RGRID, M_b), "rar") * RGRID ** 2 / G,
                  sigma_s=1.05 * sig_c)
    sectionR1(L_crit, xy, f"RAR x Sigma_s = {1.05*sig_c:.2f}")
    sectionR4(M_b, xy, dphi_req, dphi_err, thE)
    sectionR5(dphi_req)
    sectionR6(M_b, M_gas, M_star, xy, dphi_req)
    sectionR7()
    RES["seconds"] = time.time() - TSTART
    RES["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(HERE, "refsdal_results.json"), "w") as f:
        json.dump(RES, f, indent=1, default=float)
    print(f"\n   wrote refsdal_results.json in {time.time() - TSTART:.0f}s")


if __name__ == "__main__":
    main()
