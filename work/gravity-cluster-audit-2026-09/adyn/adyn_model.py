"""Baryon models and the forward observable chain for the DiskMass A_dyn test.

WHY THIS FILE EXISTS
--------------------
The previous run compared only SCALE LENGTHS.  Under the local-sheet closure
sigma_z^2 ~ B_z(R) Sigma(R) h_z(R) with Sigma ~ exp(-R/h_R) and h_z constant,

    1/h_sigma_z = 1/(2 h_R) - (1/2) dln B_z/dR

so a CONSTANT vertical enhancement B_z = B0 leaves h_sigma_z = 2 h_R for every
B0.  The scale-length ratio is mathematically blind to the amplitude; adyn_run.py
step 0f demonstrates that numerically before anything else is computed.  This
file forward-models the AMPLITUDE, the only place a constant B_z can appear.

THE FORWARD CHAIN, in order
---------------------------
    photometry -> Sigma_*(R), Sigma_gas(R), rho(R,z)
              -> K_z(R,z) from the z-integrated field equation (EXACT)
              -> sigma_z(R) from the z-Jeans equation (EXACT for the tracer)
              -> sigma_LOS(R) at the tabulated inclination and ellipsoid shape
              -> fibre aperture (x) seeing PSF, luminosity weighted
              -> exponential fit over the same radial window
              -> compare to BOTH sigma_LOS_0 / sigma_z_0 (amplitude) and
                 h_sigma_LOS / h_sigma_z (scale length)

THE VERTICAL FORCE IS NOT APPROXIMATED
--------------------------------------
Integrate the field equation over z from the midplane, where K_z(z=0) = 0:

  Newton      K_z(R,z) = 2 pi G Sigma(<z)  -  (z/R) dV_c^2/dR
  AQUAL       mu(|grad Phi|/a0) K_z = 2 pi G Sigma(<z) - (z/R) d(mu V_c^2)/dR
  QUMOND/RAR  K_z = nu(|g_N|/a0) K_z^N  +  O((z/h_R)^2)
  tensor      mu_z K_z = 2 pi G Sigma(<z) - (z/R) d(mu_R R g_R)/dR

The only approximation is g_R(R,z) ~ g_R(R,0) inside the integrand, an error of
O((z/h_R)^2) which adyn_run.py step 0e measures against axisym.py rather than
assuming.  Everything else -- the Jeans integral, the vertical profile family,
the projection, the aperture -- is exact.

THE FREE PARAMETER k, MADE EXPLICIT
-----------------------------------
DiskMass write Sigma_dyn = sigma_z^2/(pi G k h_z) and adopt k = 1.5.  k is not a
fudge: it is fixed by the SHAPE of the vertical mass profile at fixed exponential
scale height h_z.  For the van der Kruit (1988) family

    rho(z) = rho_0 sech^(2/n)( n z / (2 h_z) )

the z-Jeans equation gives, exactly,   sigma_z^2 = pi G k Sigma h_z   with

    k = int_0^inf sech^(2/n)(n u / 2) du

    n -> inf  exponential  k = 1.0
    n = 2     sech         k = pi/2 = 1.571   (the DiskMass choice, "1.5")
    n = 1     sech^2       k = 2.0            (self-consistent isothermal)

k therefore spans a factor of two and enters sigma_z^2 linearly.  It is carried
as a declared nuisance with a stated prior, never fixed silently.  The tabulated
h_z is the EXPONENTIAL scale height from the Bershady+2010b relation, which is
the h_z that appears in the formula above.

WHAT THE AMPLITUDE ACTUALLY CONSTRAINS
--------------------------------------
    log B_z = 2 log sigma_z(obs) - log Upsilon_K - log h_z - log k + const

Upsilon_K, h_z and k enter with unit coefficient and are COMMON-MODE across the
sample (one IMF zero point, one Bershady relation, one profile shape), so they do
not average down with sqrt(N).  That is the precision floor, and it is reported
as such rather than hidden inside a combined error bar.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.special import i0, i1, k0, k1

# the validated solver, reused not rewritten
GRAVLAB = ("C:/Users/henry/Documents/Codex/2026-08-21/"
           "Invariant-main-integration/work/gravitylab")
if GRAVLAB not in sys.path:
    sys.path.insert(0, GRAVLAB)
import axisym as X                                            # noqa: E402

ACQ = ("C:/Users/henry/AppData/Local/Temp/claude/"
       "C--Users-henry-dev/a2309145-5e60-4815-97f2-bb0c877edc0d/"
       "scratchpad/acquire")

G = 6.674e-11
KPC = 3.0856775814913673e19
PC = KPC / 1e3
MSUN = 1.98892e30
A0_CANON = 1.2e-10
MSUN_K = 3.28          # 2MASS Ks absolute magnitude of the Sun
ARCSEC_PER_RAD = 206264.806

# ------------------------------------------------------------------- defaults
#: Declared BEFORE any residual is examined.  Two galaxies fail these.
CUTS = dict(max_frac_err_sigma=0.30,   # e_sigma_LOS_0 / sigma_LOS_0
            max_BD=0.35)               # bulge/disk light ratio

#: Fiducial nuisance values.  Every one is marginalised in adyn_run.py; these are
#: the CENTRES of the priors, not fixed choices.
FID = dict(
    Upsilon_K=0.60,        # K-band stellar M/L, Msun/Lsun_K
    s_Upsilon=0.15,        # dex, COMMON-MODE (IMF zero point)
    s_Upsilon_gal=0.06,    # dex, per-galaxy (memory: SPARC mid-IR route)
    col_slope=0.15,        # dex per mag of (B-K) about the sample mean
    s_col_slope=0.10,
    BK_pivot=3.4,
    f_gas=0.25,            # M_gas/M_star -- NOT tabulated, a prior only
    s_f_gas=0.25,          # dex
    h_gas_over_hR=2.0,     # gas radial scale length / h_R
    h_zgas_over_hz=0.5,    # gas vertical scale height / h_z
    s_hz_sys=0.10,         # dex, COMMON-MODE Bershady+2010b zero point
    k_lo=1.5, k_hi=2.0,    # vertical-profile constant, fiducial prior
    alpha=0.60,            # sigma_z/sigma_R
    s_alpha=0.12,          # common-mode
    fibre_diam_as=2.7,     # PPak fibre diameter
    psf_fwhm_as=1.5,       # typical seeing
    fit_lo=0.3, fit_hi=2.0,  # exponential-fit window, units of h_R
)


# ================================================== vertical profile family
class VertProfile:
    """rho(z)/rho_0 = sech^(2/n)(n u/2),  u = z/h_z.   k = int_0^inf w du."""

    def __init__(self, n, umax=16.0, nu=4001):
        self.n = float(n)
        self.u = np.linspace(0.0, umax, nu)
        x = self.n * self.u / 2.0
        # log cosh without overflow
        logcosh = x + np.log1p(np.exp(-2.0 * np.minimum(x, 700.0))) - np.log(2.0)
        logcosh = np.where(x < 1e-8, 0.5 * x ** 2, logcosh)
        self.w = np.exp(-(2.0 / self.n) * logcosh)
        c = np.concatenate(([0.0], np.cumsum(
            0.5 * (self.w[1:] + self.w[:-1]) * np.diff(self.u))))
        self.C = c
        self.k = float(c[-1])
        self.Cn = c / self.k                         # normalised cumulative
        self.L = float(np.trapezoid(self.w * self.u, self.u))


_NGRID = np.concatenate((np.linspace(0.6, 4.0, 60), np.logspace(np.log10(4.2),
                                                                3.0, 60)))
_KOF = None
_PCACHE: dict = {}


def profile_for_k(k):
    """VertProfile whose k matches the requested value (cached, 3-dp key)."""
    global _KOF
    if _KOF is None:
        _KOF = np.array([VertProfile(n, nu=1501).k for n in _NGRID])
    kk = float(np.clip(k, _KOF.min(), _KOF.max()))
    n = float(np.interp(-kk, -_KOF, _NGRID))     # k(n) is decreasing
    key = round(n, 4)
    if key not in _PCACHE:
        _PCACHE[key] = VertProfile(key)
    return _PCACHE[key]


def vertical_weights(prof_s: VertProfile, q_gas, prof_g: VertProfile = None):
    """(A_ss, A_sg, L_s) for the Jeans integral.

        sigma_z^2 = 2 pi G h_z [ Sigma_* A_ss + Sigma_g A_sg ]
                    - L_s h_z^2 (1/R) dVc^2/dR

    A_ss = k_*/2 exactly.  A_sg -> k_* as the gas layer becomes thin, i.e. a
    razor-thin gas disk contributes TWICE per unit surface density, because the
    stars feel its whole column at every height.
    """
    prof_g = prof_g or prof_s
    A_ss = prof_s.k / 2.0
    Cg = np.interp(prof_s.u / max(q_gas, 1e-6), prof_g.u, prof_g.Cn,
                   left=0.0, right=1.0)
    A_sg = float(np.trapezoid(prof_s.w * Cg, prof_s.u))
    return A_ss, A_sg, prof_s.L


# ============================================================== data ingestion
def _rows(path):
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        return [dict(zip(head, ln.rstrip("\n").split("\t")))
                for ln in fh if ln.strip()]


def _f(tok, default=float("nan")):
    tok = (tok or "").strip()
    if not tok or tok in ("--", "-"):
        return default
    try:
        return float(tok)
    except ValueError:
        return default


@dataclass
class DMGalaxy:
    ugc: str
    D: float            # Mpc
    eD: float
    hR_as: float        # arcsec  (DiskMass VI table 1)
    ehR_as: float
    hR_kpc: float       # kpc     (DiskMass VII)
    ehR_kpc: float
    hz_kpc: float       # kpc, INFERRED from h_R -- a prior, not a measurement
    ehz_kpc: float
    mu0K: float         # inclination-corrected central K SB, mag/arcsec^2
    emu0K: float
    BK: float           # B-K colour
    MK: float
    BD: float           # bulge/disk light
    incl: float         # deg, from the INVERTED Tully-Fisher relation
    eincl: float
    Vsini: float        # km/s
    eVsini: float
    Vflat_TF: float     # km/s, TF prediction from M_K
    Varot: float        # km/s, arctan asymptote, projected
    rs_as: float        # arcsec, arctan turnover radius
    sLOS0: float
    esLOS0: float
    hsLOS_as: float
    ehsLOS_as: float
    sz0: float
    esz0: float
    hsz_as: float
    ehsz_as: float
    keep: bool = True
    drop_reason: str = ""
    hR_m: float = field(default=0.0)
    SigmaL0: float = field(default=0.0)   # Lsun/pc^2, K band, disk only
    Ldisk: float = field(default=0.0)     # Lsun


def load_diskmass(acq=ACQ, verbose=True):
    t6 = {r["UGC"]: r for r in _rows(os.path.join(acq, "dms6_table6_sigma_z.tsv"))}
    t7 = {r["UGC"]: r for r in _rows(os.path.join(acq, "dms7_hR_hz.tsv"))}
    t1 = {r["UGC"]: r for r in
          _rows(os.path.join(acq, "dms6_table1_galaxy_properties.tsv"))}
    t5 = {r["UGC"]: r for r in
          _rows(os.path.join(acq, "dms6_table5_orientation.tsv"))}
    ugcs = sorted(set(t6) & set(t7) & set(t1) & set(t5), key=int)
    gals = []
    for u in ugcs:
        a, b, c, d = t6[u], t7[u], t1[u], t5[u]
        g = DMGalaxy(
            ugc=u, D=_f(c["Dist"]), eD=_f(c["e_Dist"]),
            hR_as=_f(c["h_R_arcsec"]), ehR_as=_f(c["e_h_R_arcsec"]),
            hR_kpc=_f(b["h_R"]), ehR_kpc=_f(b["e_h_R"]),
            hz_kpc=_f(b["h_z"]), ehz_kpc=_f(b["e_h_z"]),
            mu0K=_f(c["mu0_K_i"]), emu0K=_f(c["e_mu0_K_i"]),
            BK=_f(c["B_K"]), MK=_f(d["M_K"]), BD=_f(c["BD_ratio"], 0.0),
            incl=_f(d["i_TF"]), eincl=_f(d["e_i_TF"]),
            Vsini=_f(d["Vc_sini_gas"]), eVsini=_f(d["e_Vc_sini_gas"]),
            Vflat_TF=_f(d["Vflat_TF"]),
            Varot=_f(d["Varot_OIII"]), rs_as=_f(d["rs_OIII"]),
            sLOS0=_f(a["sigma_LOS_0"]), esLOS0=_f(a["e_sigma_LOS_0"]),
            hsLOS_as=_f(a["h_sigma_LOS"]), ehsLOS_as=_f(a["e_h_sigma_LOS"]),
            sz0=_f(a["sigma_z_0"]), esz0=_f(a["e_sigma_z_0"]),
            hsz_as=_f(a["h_sigma_z"]), ehsz_as=_f(a["e_h_sigma_z"]))
        if not np.isfinite(g.rs_as) or g.rs_as <= 0:
            g.rs_as = _f(d["rs_stars"], 5.0)
            g.Varot = _f(d["Varot_stars"], g.Vsini)
        g.hR_m = g.hR_as * g.D / ARCSEC_PER_RAD * 1e3 * KPC
        g.SigmaL0 = 10.0 ** (0.4 * (MSUN_K + 21.572 - g.mu0K))
        g.Ldisk = 2 * np.pi * g.SigmaL0 * (g.hR_m / PC) ** 2
        if g.esLOS0 / g.sLOS0 > CUTS["max_frac_err_sigma"]:
            g.keep, g.drop_reason = False, "e_sigma_LOS_0/sigma_LOS_0 > 0.30"
        elif g.BD > CUTS["max_BD"]:
            g.keep, g.drop_reason = False, "B/D > 0.35"
        gals.append(g)
    if verbose:
        print(f"   DiskMass galaxies joined on UGC : {len(gals)}")
        for g in gals:
            if not g.keep:
                print(f"   dropped UGC {g.ugc:>5}            : {g.drop_reason}")
        print(f"   RETAINED                        : "
              f"{len([x for x in gals if x.keep])}")
    return gals


# ================================================== velocity-ellipsoid algebra
def beta_epicyclic(dlnV_dlnR):
    return np.sqrt(np.clip(0.5 * (1.0 + dlnV_dlnR), 1e-6, None))


def sigma_los_from_z(sigma_z, incl_deg, alpha, beta):
    """sigma_LOS^2 = sigma_R^2 sin^2 i sin^2 th + sigma_th^2 sin^2 i cos^2 th
                     + sigma_z^2 cos^2 i,  azimuthally averaged."""
    i = np.radians(incl_deg)
    c2, s2 = np.cos(i) ** 2, np.sin(i) ** 2
    return sigma_z * np.sqrt(c2 + 0.5 * s2 * (1.0 + beta ** 2) / alpha ** 2)


def sigma_z_from_los(sigma_los, incl_deg, alpha, beta):
    i = np.radians(incl_deg)
    c2, s2 = np.cos(i) ** 2, np.sin(i) ** 2
    return sigma_los / np.sqrt(c2 + 0.5 * s2 * (1.0 + beta ** 2) / alpha ** 2)


def effective_alpha(sigma_los, sigma_z, incl_deg, beta):
    """Invert the published (sigma_LOS_0, sigma_z_0, i) triple for the alpha
    DiskMass actually used.  A diagnostic on the adopted ellipsoid."""
    i = np.radians(incl_deg)
    c2, s2 = np.cos(i) ** 2, np.sin(i) ** 2
    lhs = (sigma_los / sigma_z) ** 2 - c2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(0.5 * s2 * (1.0 + beta ** 2) / lhs)


# ================================================ finite-thickness correction
#  T(x = R/h_R ; s = h_sech/h_R) = g_R(sech^2 disk)/g_R(razor thin), midplane.
#  Scale-free, so one table serves every galaxy.  Computed as a RATIO of two
#  solves on the SAME grid, which cancels the box size and the monopole boundary
#  condition -- the two things a single solve would leave in.
_TCACHE: dict = {}
_TQ = np.array([0.04, 0.10, 0.16, 0.24, 0.32, 0.45])
_TPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "thickness_table.npz")


def _thickness_curve(q, nR=240, nz=160, box=16.0, zbox=4.0):
    hR, Sig0 = 3.0, 300 * MSUN / KPC ** 2 * 1e6
    Mtot = 2 * np.pi * Sig0 * (hR * KPC) ** 2
    out = []
    for hz in (max(q, 1e-9) * hR, 0.004 * hR):
        g = X.Grid(nR, nz, box * hR, zbox * hR)
        with np.errstate(over="ignore", invalid="ignore"):
            rho = np.nan_to_num(
                X.exponential_disk(g.Rc / KPC, g.zc / KPC, Sig0, hR, hz))
        rho *= (Mtot / 2.0) / float(np.sum(rho * g.V))
        Psi, _, _ = X.solve_axi(rho, X.isotropic_A(rho.shape), g,
                                X.monopole_bc(g, Mtot), tol=1e-12, maxiter=20000)
        out.append(X.midplane_vc(Psi, g) ** 2 / np.maximum(g.Rc, 1e-30))
    return g.Rc / (hR * KPC), out[0] / out[1]


def thickness_T(x, q, verbose=False):
    if not _TCACHE:
        if os.path.exists(_TPATH):
            d = np.load(_TPATH)
            if all(f"x{qq:.2f}" in d for qq in _TQ):
                for qq in _TQ:
                    _TCACHE[float(qq)] = (d[f"x{qq:.2f}"], d[f"t{qq:.2f}"])
        if not _TCACHE:
            blob = {}
            for qq in _TQ:
                xs, ts = _thickness_curve(qq)
                _TCACHE[float(qq)] = (xs, ts)
                blob[f"x{qq:.2f}"], blob[f"t{qq:.2f}"] = xs, ts
                if verbose:
                    print(f"      thickness table q={qq:.2f} built")
            np.savez(_TPATH, **blob)
    qc = float(np.clip(q, _TQ[0], _TQ[-1]))
    j = int(np.clip(np.searchsorted(_TQ, qc, side="right") - 1, 0, len(_TQ) - 2))
    q0, q1 = _TQ[j], _TQ[j + 1]
    w = (qc - q0) / (q1 - q0)
    x0, t0 = _TCACHE[float(q0)]
    x1, t1 = _TCACHE[float(q1)]
    return (1 - w) * np.interp(x, x0, t0) + w * np.interp(x, x1, t1)


# ======================================================== baryon surface model
class Baryons:
    """Exponential stellar disk + exponential gas disk.

    Vertical structure: the van der Kruit family at the requested k, with h_z the
    TABULATED exponential scale height.  The 2-D solver, which only ever supplies
    RATIOS (thickness correction, tensor B_R), uses the isothermal sech^2 with
    scale parameter 2 h_z; that choice moves g_R by <1% and nothing else.
    """

    def __init__(self, gal: DMGalaxy, Upsilon, f_gas, hz_kpc, k=1.5,
                 h_gas_over_hR=None, h_zgas_over_hz=None, nR=200, Rmax_hR=5.0,
                 k_gas=2.0):
        F = FID
        h_gas_over_hR = F["h_gas_over_hR"] if h_gas_over_hR is None else h_gas_over_hR
        h_zgas_over_hz = (F["h_zgas_over_hz"] if h_zgas_over_hz is None
                          else h_zgas_over_hz)
        self.gal = gal
        self.hR = gal.hR_m
        self.hz = hz_kpc * KPC                 # exponential scale height
        self.hg = h_gas_over_hR * self.hR
        self.hzg = h_zgas_over_hz * self.hz
        self.k = k
        self.prof = profile_for_k(k)
        self.prof_g = profile_for_k(k_gas)
        self.A_ss, self.A_sg, self.L_s = vertical_weights(
            self.prof, self.hzg / self.hz, self.prof_g)
        self.Ups = Upsilon
        self.Mstar = Upsilon * gal.Ldisk * MSUN
        self.Mgas = f_gas * self.Mstar
        self.Sig_s0 = self.Mstar / (2 * np.pi * self.hR ** 2)
        self.Sig_g0 = self.Mgas / (2 * np.pi * self.hg ** 2)
        self.h_sech = 2.0 * self.hz           # solver-side geometry
        self.h_sech_g = 2.0 * self.hzg
        self.R = np.linspace(0.02, Rmax_hR, nR) * self.hR

    def Sigma_star(self, R=None):
        R = self.R if R is None else R
        return self.Sig_s0 * np.exp(-R / self.hR)

    def Sigma_gas(self, R=None):
        R = self.R if R is None else R
        return self.Sig_g0 * np.exp(-R / self.hg)

    def gR_newton(self, R=None, thick=True):
        """g_R(R,0) = pi G Sigma0 x [I0K0-I1K1](x/2) per component, times the
        solver-measured finite-thickness correction.  Poisson is linear."""
        R = self.R if R is None else R
        x = R / self.hR
        y = np.maximum(x / 2.0, 1e-8)
        g = (np.pi * G * self.Sig_s0 * x * (i0(y) * k0(y) - i1(y) * k1(y)))
        if thick:
            g = g * thickness_T(x, self.h_sech / self.hR)
        if self.Sig_g0 > 0:
            xg = R / self.hg
            yg = np.maximum(xg / 2.0, 1e-8)
            gg = (np.pi * G * self.Sig_g0 * xg
                  * (i0(yg) * k0(yg) - i1(yg) * k1(yg)))
            if thick:
                gg = gg * thickness_T(xg, self.h_sech_g / self.hg)
            g = g + gg
        return g

    def Vc2_newton(self, R=None):
        R = self.R if R is None else R
        return R * self.gR_newton(R)

    def Sigma_below(self, R, u):
        """Sigma(<z) with u = z/h_z, both components on their own profiles."""
        cs = np.interp(u, self.prof.u, self.prof.Cn, left=0.0, right=1.0)
        cg = np.interp(u * self.hz / self.hzg, self.prof_g.u, self.prof_g.Cn,
                       left=0.0, right=1.0)
        return self.Sigma_star(R) * cs + self.Sigma_gas(R) * cg

    def sigma_z2_newton(self, R=None):
        """Closed form of the Newtonian z-Jeans integral.

            sigma_z^2 = 2 pi G h_z [Sigma_* A_ss + Sigma_g A_sg]
                        - L_s h_z^2 (1/R) dVc^2/dR

        with A_ss = k/2 exactly, so the pure stellar limit is the DiskMass
        closure sigma_z^2 = pi G k Sigma h_z by construction.
        """
        R = self.R if R is None else R
        Vc2 = R * self.gR_newton(R)
        s2 = (2 * np.pi * G * self.hz
              * (self.Sigma_star(R) * self.A_ss + self.Sigma_gas(R) * self.A_sg)
              - self.L_s * self.hz ** 2 * np.gradient(Vc2, R) / R)
        return np.maximum(s2, 1e-30)


# ================================================================ gravity laws
def nu_rar(x):
    """RAR interpolation, McGaugh Lelli & Schombert 2016.  x = g_N/a0."""
    x = np.maximum(np.asarray(x, float), 1e-300)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def mu_simple(x):
    return x / (1.0 + x)


def s_gap(r, Mb_kg, eta, a0):
    """Anisotropic-tensor gap profile s(r) = r^2/(r_t(r+r_t)),
    r_t = eta sqrt(G M_b/a0).  Reproduced from mirror_models.py."""
    rt = eta * np.sqrt(G * Mb_kg / a0)
    return r ** 2 / (rt * (r + rt))


@dataclass
class Law:
    name: str
    kind: str                    # 'newton' | 'algebraic' | 'aqual' | 'tensor'
    params: dict = field(default_factory=dict)


LAW_NEWTON = Law("newton", "newton")


def aqual_Kz(KzN, gR, a0):
    """Solve the AQUAL vertical equation for K_z, by bisection.

        mu(|grad Phi|/a0) K_z = K_z(Newton)      mu(x) = x/(1+x)
      =>  (K_z - K_z^N) sqrt(g_R^2 + K_z^2) = K_z^N a0

    The left side is strictly increasing in K_z for K_z >= K_z^N, is negative at
    K_z = K_z^N and positive at K_z = K_z^N + a0, so the root is bracketed
    unconditionally.  The obvious fixed-point iteration
    K <- K_z^N (1 + a0/g) has contraction factor -> 1 in the deep-MOND limit and
    silently fails to converge there; bisection does not.
    """
    lo = KzN.copy()
    hi = KzN + a0
    for _ in range(45):          # 2^-45 relative: below double-precision needs
        mid = 0.5 * (lo + hi)
        F = (mid - KzN) * np.sqrt(gR ** 2 + mid ** 2) - KzN * a0
        lo = np.where(F < 0, mid, lo)
        hi = np.where(F < 0, hi, mid)
    return 0.5 * (lo + hi)


def Kz_grid(law: Law, bar: Baryons, R, u, gR, gRN, Vc2N, muz=None, Bz2d=None):
    """K_z on the (R, z = u h_z) grid from the z-integrated field equation.

    The radial-leakage term is the SAME for every law.  Integrating the field
    equation over z leaves int_0^z (1/R) d_R(R A g_R) dz', and the radial
    reduction of every law here gives A g_R = g_R(Newton) to the accuracy of the
    reduction, so that term equals the Newtonian one and cancels from B_z at
    leading order.  What remains is exact:

        Newton   K_z = K_z^N
        RAR      K_z = nu(|g_N|/a0) K_z^N            (QUMOND, leading order)
        AQUAL    (K_z - K_z^N) sqrt(g_R^2+K_z^2) = K_z^N a0
        tensor   K_z = K_z^N / mu_z                  (2-D solved ratio if given)
    """
    zz = u[None, :] * bar.hz
    Sig_lt = bar.Sigma_below(R[:, None], u[None, :])
    dN = (np.gradient(Vc2N, R) / R)[:, None]
    KzN = np.maximum(2 * np.pi * G * Sig_lt - zz * dN, 1e-30)
    if law.kind == "newton":
        return KzN, KzN
    a0 = law.params.get("a0", A0_CANON)
    if law.kind == "algebraic":
        gtot = np.sqrt(gRN[:, None] ** 2 + KzN ** 2)
        Kz = nu_rar(gtot / a0) * KzN
    elif law.kind == "aqual":
        Kz = aqual_Kz(KzN, gR[:, None] * np.ones_like(KzN), a0)
    elif law.kind == "tensor":
        if Bz2d is not None:
            Kz = KzN * Bz2d[:, None]
        else:
            mz = np.ones_like(R) if muz is None else muz
            Kz = KzN / mz[:, None]
    else:
        raise ValueError(law.kind)
    return np.maximum(Kz, 1e-30), KzN


def sigma_z_of_R(law: Law, bar: Baryons, R=None, gR=None, muz=None, Bz2d=None):
    """z-Jeans:  sigma_z^2(R) = int_0^inf w(u) K_z(R, u h_z) h_z du.

    Returns (sigma_z [m/s], B_z_eff = sigma_z^2/sigma_z_Newton^2).
    """
    R = bar.R if R is None else R
    u, w = bar.prof.u, bar.prof.w
    gRN = bar.gR_newton(R)
    Vc2N = R * gRN
    if gR is None:
        if law.kind == "newton":
            gR = gRN
        elif law.name.startswith("rar"):
            gR = nu_rar(gRN / law.params["a0"]) * gRN
        elif law.name.startswith("aqual"):
            a0 = law.params["a0"]
            gR = 0.5 * (gRN + np.sqrt(gRN ** 2 + 4 * gRN * a0))
        else:
            gR = gRN
    Kz, KzN = Kz_grid(law, bar, R, u, gR, gRN, Vc2N, muz=muz, Bz2d=Bz2d)
    s2 = np.trapezoid(w[None, :] * Kz, u, axis=1) * bar.hz
    s2N = np.trapezoid(w[None, :] * KzN, u, axis=1) * bar.hz
    return np.sqrt(np.maximum(s2, 0.0)), s2 / s2N


# ================================================ aperture and PSF convolution
def smear_kernel(dx_as, fibre_diam_as, psf_fwhm_as):
    from scipy.signal import fftconvolve
    r_fib = 0.5 * fibre_diam_as
    s = psf_fwhm_as / 2.3548
    n = int(np.ceil((r_fib + 3.5 * s) / dx_as))
    ax = np.arange(-n, n + 1) * dx_as
    XX, YY = np.meshgrid(ax, ax, indexing="ij")
    rr = np.hypot(XX, YY)
    k = fftconvolve((rr <= r_fib).astype(float),
                    np.exp(-0.5 * (rr / s) ** 2), mode="same")
    return k / k.sum()


def apply_aperture(gal: DMGalaxy, R_as, sigma_los_as, fibre_diam_as,
                   psf_fwhm_as, dx_as=0.30, box_hR=3.2):
    """Luminosity-weighted second moment inside one fibre.

    DiskMass shift each fibre to the local V_LOS before co-adding, so the
    residual beam smearing is the WITHIN-fibre velocity gradient, which is what
    a fibre-sized kernel reproduces.
    """
    from scipy.signal import fftconvolve
    inc = np.radians(gal.incl)
    half = box_hR * gal.hR_as
    ax = np.arange(-half, half + dx_as, dx_as)
    XX, YY = np.meshgrid(ax, ax, indexing="ij")           # x = major axis
    Rd = np.maximum(np.hypot(XX, YY / max(np.cos(inc), 1e-3)), 1e-6)
    I = np.exp(-Rd / gal.hR_as)
    Vp = (2.0 / np.pi) * gal.Varot * np.arctan(Rd / max(gal.rs_as, 1e-3))
    VL = Vp * (XX / Rd)
    sl = np.interp(Rd, R_as, sigma_los_as,
                   left=sigma_los_as[0], right=sigma_los_as[-1])
    k = smear_kernel(dx_as, fibre_diam_as, psf_fwhm_as)
    m0 = fftconvolve(I, k, mode="same")
    m1 = fftconvolve(I * VL, k, mode="same")
    m2 = fftconvolve(I * (VL ** 2 + sl ** 2), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = m2 / m0 - (m1 / m0) ** 2
    s2 = np.where(np.isfinite(s2) & (s2 > 0), s2, np.nan)
    out = np.full_like(R_as, np.nan, dtype=float)
    edges = np.concatenate(([0.0], 0.5 * (R_as[1:] + R_as[:-1]),
                            [R_as[-1] * 1.5]))
    idx = np.digitize(Rd.ravel(), edges) - 1
    v, wt = s2.ravel(), I.ravel()
    ok = np.isfinite(v)
    for j in range(len(R_as)):
        m = ok & (idx == j)
        if m.sum() > 3:
            out[j] = np.sqrt(np.sum(wt[m] * v[m]) / np.sum(wt[m]))
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.interp(R_as[bad], R_as[~bad], out[~bad])
    return out


# ==================================================== exponential-fit operator
def fit_exponential(R, y, lo, hi, w=None):
    """ln y = ln y0 - R/h over lo <= R <= hi."""
    m = (R >= lo) & (R <= hi) & np.isfinite(y) & (y > 0)
    if m.sum() < 4:
        return np.nan, np.nan, 0.0
    p = np.polyfit(R[m], np.log(y[m]), 1, w=None if w is None else w[m])
    return float(np.exp(p[1])), float(-1.0 / p[0]), float(m.mean())


def fit_exponential_rows(R_as, Y, lo, hi):
    """Vectorised: one exponential fit per row of Y on the shared R_as grid."""
    m = (R_as >= lo) & (R_as <= hi)
    x = R_as[m]
    ly = np.log(np.maximum(Y[:, m], 1e-30))
    n = x.size
    sx, sxx = x.sum(), (x * x).sum()
    sy = ly.sum(axis=1)
    sxy = (ly * x).sum(axis=1)
    den = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return np.exp(a), -1.0 / b


# =============================================================== 2-D reference
def solve_2d(bar: Baryons, mode, eta=None, a0=None, nR=160, nz=96,
             box_hR=14.0, zbox_hR=7.0, tol=1e-11, maxiter=12000):
    """Full axisymmetric solve with axisym.py.  Gates the semi-analytic chain and
    supplies the tensor radial force, which has no exact reduction.

    mode: 'newton' | 'tensor_aniso' | 'tensor_iso'
    """
    hR_kpc = bar.hR / KPC
    g = X.Grid(nR, nz, box_hR * hR_kpc, zbox_hR * hR_kpc)
    with np.errstate(over="ignore", invalid="ignore"):
        rho = np.nan_to_num(X.exponential_disk(g.Rc / KPC, g.zc / KPC,
                                               bar.Sig_s0, hR_kpc,
                                               bar.h_sech / KPC))
        if bar.Sig_g0 > 0:
            rho = rho + np.nan_to_num(
                X.exponential_disk(g.Rc / KPC, g.zc / KPC, bar.Sig_g0,
                                   bar.hg / KPC, bar.h_sech_g / KPC))
    Mtot = bar.Mstar + bar.Mgas
    rho *= (Mtot / 2.0) / float(np.sum(rho * g.V))
    if mode == "newton":
        A = X.isotropic_A(rho.shape)
    else:
        RR = g.Rc[:, None] * np.ones((1, g.nz))
        zz = np.ones((g.nR, 1)) * g.zc[None, :]
        muR = 1.0 / (1.0 + eta * s_gap(np.sqrt(RR ** 2 + zz ** 2), Mtot, eta, a0))
        muz = muR.copy() if mode == "tensor_iso" else np.ones_like(muR)
        A = (muR, muz, np.zeros_like(muR))
    Psi, it, rel = X.solve_axi(rho, A, g, X.monopole_bc(g, Mtot),
                               tol=tol, maxiter=maxiter)
    vc = X.midplane_vc(Psi, g)
    return dict(grid=g, Psi=Psi, R_m=g.Rc, gR=vc ** 2 / np.maximum(g.Rc, 1e-30),
                Kz=np.abs(X.vertical_Kz(Psi, g, iz=1)), rel=rel, iters=it,
                converged=bool(rel < 1e-9))
