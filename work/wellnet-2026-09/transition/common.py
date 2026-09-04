"""The ONE forward framework: S(M, r) across eFEDS, LoCuSS and strong-lens cores.

Everything downstream -- design measurement, hierarchy fit, nulls,
responsiveness, blind evaluation -- calls only what is in here, so the three
surveys cannot drift apart.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("efeds-hsc", "lead01", "closure"):
    q = os.path.join(ROOT, p)
    if q not in sys.path:
        sys.path.insert(0, q)

import pipeline as P                                            # noqa: E402
import lead01 as L                                              # noqa: E402
import closure as C                                             # noqa: E402
import build as B                                               # noqa: E402
import decl                                                     # noqa: E402

MPC, KPC, MSUN, G = P.MPC, P.KPC, P.MSUN, P.G
CLIGHT = P.CLIGHT
ARCSEC = P.ARCSEC
A0 = decl.A0_RAR
M0 = decl.M0_MSUN * MSUN

SURVEYS = ("efeds", "locuss", "sl")


def aperture_R500(cl, mode):
    """The R500 used for BOTH the radius axis and the mass aperture.

    'cat'  external catalogue aperture (Bahar+2022 / Okabe M_WL / MCXC)
    'dyn'  dynamical R500 under the frozen law from the baryons alone
    """
    if mode == "cat":
        return cl.extra.get("R500_cat")
    return cl.R500


def M_at(cl, R):
    return float(np.interp(R, cl.r, cl.M_gas))


# ------------------------------------------------------------------ geometry
def sigma_crit(z_l, z_s):
    D_l = float(P.d_ang(z_l))
    D_s = float(P.d_ang(z_s))
    D_ls = float(P.d_ang12(z_l, z_s))
    return CLIGHT ** 2 / (4.0 * math.pi * G) * D_s / (D_l * D_ls), D_l


def M_2d(r, M3, R, r_trunc=20.0 * MPC):
    """Projected mass inside the cylinder of radius R, EXACTLY.

        M_2D(<R) = M_3D(<R) + Int_R^rt (dM_3D/dr) [1 - sqrt(1 - R^2/r^2)] dr

    because the fraction of a spherical shell at radius r that falls inside
    the cylinder is 1 for r < R and 1 - sqrt(1 - R^2/r^2) for r > R.

    [caught a bug]  The shear pipeline's sigma_from_g builds Sigma_bar by
    integrating Sigma(R') inward from a grid that starts at 1 kpc and assumes
    Sigma ~ const inside it.  Tested against a singular isothermal sphere that
    is wrong by 8.1% at R = 27 kpc, 4.1% at 54 kpc and 2.1% at 108 kpc, and
    the error is FLAT in n_t, n_R and the radial grid density -- the
    programme's own signature for a modelling mismatch rather than a
    quadrature error.  It is negligible for the eFEDS shear (R > 0.29 Mpc,
    R/r_min > 58) but not for strong-lensing cores at 50-250 kpc.  This form
    has no inner boundary term at all.
    The integrand's derivative is singular at r = R (w ~ 1 - sqrt(r-R)), so
    the outer integral is done in t with r = R cosh t, which removes the kink
    exactly: dr = R sinh t dt and sqrt(1 - R^2/r^2) = tanh t.
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    dM = np.gradient(M3, r)
    lr = np.log(r)
    ldM = np.log(np.maximum(dM, 1e-300))
    n_t = 600
    u = np.linspace(0.0, 1.0, n_t)
    out = np.empty(R.size)
    for i, Rv in enumerate(R):
        T = math.acosh(max(r_trunc / Rv, 1.0 + 1e-12))
        t = u * T
        rr = Rv * np.cosh(t)
        dm = np.exp(np.interp(np.log(np.clip(rr, r[0], r[-1])), lr, ldM))
        integ = dm * (1.0 - np.tanh(t)) * Rv * np.sinh(t)
        out[i] = float(np.interp(Rv, r, M3)) + float(np.trapezoid(integ, t))
    return out


def kappa_bar(cl, theta_as, z_s):
    """Mean convergence inside theta for the FROZEN law, Sigma_s = 1.

    kappa_bar = M_2D(<R) / (pi R^2 Sigma_cr),  R = theta D_l.
    """
    scr, D_l = sigma_crit(cl.z, z_s)
    R = np.atleast_1d(theta_as) * ARCSEC * D_l
    g = B.g_rar(cl.g_b)
    M3 = g * cl.r ** 2 / G
    return M_2d(cl.r, M3, R) / (math.pi * R ** 2 * scr)


# ============================================================ the S estimates
class Bundle:
    """All three surveys in the common representation, plus their design."""

    def __init__(self, verbose=True, star_mult=1.0, f_star_efeds=0.0,
                 r500_mode="cat", theta_max=None, sl_agg=True):
        self.r500_mode = r500_mode
        self.theta_max = theta_max
        # Collapse the strong-lens sample to ONE point per cluster.
        # Its within-cluster radial structure is demonstrably artefactual:
        # S = 1/kappa_bar(theta) and ln(r/R500) = ln(theta D_l/R500) share
        # theta with d ln x/d ln theta = 1 exactly, and the measured internal
        # slope comes out POSITIVE (+0.20 +- 0.03) where every other probe
        # says negative.  Left in, that artefact drives the joint beta by 65
        # in -2 ln L while the 3365 eFEDS shear points move it by 12.
        # Aggregating keeps the cluster-level AMPLITUDE, which is the part
        # the Einstein-radius argument actually supports.
        self.sl_agg = sl_agg
        self.ef, self.obs, self.syss, self.ef_cuts = B.build_efeds(verbose)
        self.lo, self.lop, self.lo_cuts = B.build_locuss(verbose)
        self.sl, self.sl_cuts = B.build_sl(verbose, star_mult=star_mult,
                                           theta_max=theta_max)
        self.f_star_efeds = f_star_efeds
        self._efeds_design()
        self._locuss_points()
        self._sl_points()
        if sl_agg:
            self._collapse_sl()

    def _collapse_sl(self):
        import collections
        by = collections.OrderedDict()
        for r in self.sl_rows:
            by.setdefault(r["cid"], []).append(r)
        rows = []
        for cid, v in by.items():
            n = len(v)
            e2 = float(np.mean([q["e_stat"] ** 2 for q in v])) / n
            rows.append(dict(
                cid=cid, sid="ALL", n_img=sum(q["n_img"] for q in v),
                n_systems=n,
                theta_as=float(np.mean([q["theta_as"] for q in v])),
                z_s=float(np.mean([q["z_s"] for q in v])),
                lnS=float(np.mean([q["lnS"] for q in v])),
                S=float(math.exp(np.mean([q["lnS"] for q in v]))),
                lnM=float(np.mean([q["lnM"] for q in v])),
                lnx=float(np.mean([q["lnx"] for q in v])),
                lnkT=float(np.mean([q["lnkT"] for q in v])),
                lng=float(np.mean([q["lng"] for q in v])),
                # measurement error on the CLUSTER MEAN: the propagated
                # image-radius error plus the observed scatter across this
                # cluster's own image systems, which is the empirical size of
                # the monopole approximation for this cluster
                e_stat=math.sqrt(e2 + (np.var([q["lnS"] for q in v], ddof=1)
                                       / n if n > 1 else 0.0)),
                r=float(np.mean([q["r"] for q in v])),
                kappa_bar=float(np.mean([q["kappa_bar"] for q in v])),
                dlnS_dlntheta=float(np.mean([q["dlnS_dlntheta"] for q in v])),
                lnS_sd=float(np.std([q["lnS"] for q in v], ddof=1))
                if n > 1 else 0.0))
        self.sl_rows_full = self.sl_rows
        self.sl_rows = rows

    # ------------------------------------------------------------- eFEDS
    def _efeds_design(self):
        obs = self.obs
        idx = np.arange(len(self.syss))
        self.ef_idx = idx
        self._ef_index = {id(s): j for j, s in enumerate(self.syss)}
        self.F = C.Flat(obs, idx)
        R = self.F.R
        sysi = self.F.sysi
        R500c = np.array([c.extra["R500_cat"] for c in self.ef])
        R500d = np.array([c.R500 for c in self.ef])
        r500 = R500c if self.r500_mode == "cat" else R500d
        # the mass is measured in the SAME aperture as the radius axis
        M = np.array([M_at(c, rr) for c, rr in zip(self.ef, r500)])
        kT = np.array([c.kT for c in self.ef])
        self.ef_M, self.ef_R500c, self.ef_R500d, self.ef_kT = M, R500c, R500d, kT
        self.ef_x = dict(
            lnM=np.log(M[sysi] / M0),
            lnx=np.log(R / r500[sysi]),
            lnkT=np.log(kT[sysi] / 3.0),
        )
        gb = np.concatenate([np.interp(obs.R[k], self.syss[j].r,
                                       self.syss[j].g_b)
                             for j, k in enumerate(idx)])
        self.ef_gb = gb
        self.ef_x["lng"] = np.log(np.maximum(gb, 1e-30) / A0)

    # ------------------------------------------------------------ LoCuSS
    def _locuss_points(self):
        rows = []
        for c, p in zip(self.lo, self.lop):
            r = p["r"]                              # r500 from M_WL
            Mdyn = float(c.M_dyn(r))
            S = p["M_WL"] / Mdyn
            r500 = r if self.r500_mode == "cat" else c.R500
            rows.append(dict(
                cid=c.id, lnS=math.log(S), S=S,
                lnM=math.log(M_at(c, r500) / M0),
                lnx=math.log(r / r500),
                lnkT=math.log(c.kT / 3.0) if np.isfinite(c.kT) else np.nan,
                lng=math.log(max(float(c.g_b_at(r)), 1e-30) / A0),
                e_stat=p["frac_err"], r=r, Mdyn=Mdyn, M_WL=p["M_WL"]))
        self.lo_rows = rows

    # --------------------------------------------------------------- SL
    def _sl_points(self):
        rows = []
        self.sl_dropped = {}
        for c, systems in self.sl:
            r500 = c.extra.get("R500_cat")
            if self.r500_mode == "cat" and not r500:
                self.sl_dropped[c.id] = (
                    "no external catalogue R500: nearest MCXC source is "
                    f"{c.extra['mcxc'].get('nearest')} at "
                    f"{c.extra['mcxc']['sep_arcmin']:.1f} arcmin, beyond the "
                    "declared 3 arcmin match radius.  Kept in the R500_dyn "
                    "variant.")
                continue
            for s in systems:
                th = s["theta_as"]
                kb = float(kappa_bar(c, th, s["z_s"]))
                if not (kb > 0):
                    continue
                S = 1.0 / kb
                # d ln S / d ln theta, for propagating the image-radius spread
                kb2 = float(kappa_bar(c, th * 1.05, s["z_s"]))
                dlnS = -(math.log(kb2) - math.log(kb)) / math.log(1.05)
                sd_th = s["theta_sd"] / max(th, 1e-9) / math.sqrt(s["n_img"])
                D_l = float(P.d_ang(c.z))
                r = th * ARCSEC * D_l
                rr = r500 if (self.r500_mode == "cat" and r500) else c.R500
                rows.append(dict(
                    cid=c.id, sid=s["system_id"], n_img=s["n_img"],
                    theta_as=th, z_s=s["z_s"], lnS=math.log(S), S=S,
                    lnM=math.log(M_at(c, rr) / M0),
                    lnx=math.log(r / rr),
                    lnkT=math.log(c.kT / 3.0) if np.isfinite(c.kT) else np.nan,
                    lng=math.log(max(float(c.g_b_at(r)), 1e-30) / A0),
                    e_stat=abs(dlnS) * sd_th, r=r, kappa_bar=kb,
                    dlnS_dlntheta=float(dlnS)))
        self.sl_rows = rows


# ============================================================ the projections
def project_slip(syss, obs, idx, slipfun, law="rar"):
    """Sigma_len, DeltaSigma_len at the measured radii with an arbitrary slip.

    slipfun(system) -> array on system.r, the SHAPE of Sigma_s(r) including
    any per-system constant.  The global amplitude is left out and applied
    afterwards as a linear factor, which is exact because the Abel projection
    is linear in rho.
    """
    Ss, dSs = [], []
    for j, k in enumerate(idx):
        sm = syss[k]
        g = C.g_law(sm, law) * slipfun(sm)
        S, dS = sm.sigma_profile(g, obs.R[k], 20.0)
        Ss.append(S)
        dSs.append(dS)
    return np.concatenate(Ss), np.concatenate(dSs)


def slip_power(bundle, beta, alpha=0.0):
    """Sigma_s(r) = (M/M0)^alpha (r/R500)^beta, unit amplitude."""
    r500 = bundle.ef_R500c if bundle.r500_mode == "cat" else bundle.ef_R500d
    M = bundle.ef_M

    def f(sm):
        j = bundle._ef_index[id(sm)]
        return (M[j] / M0) ** alpha * (sm.r / r500[j]) ** beta
    return f


def slip_accel(gamma):
    def f(sm):
        return (np.maximum(sm.g_b, 1e-30) / A0) ** gamma
    return f


def slip_transition(bundle, A, xt, p=decl.TRANSITION_P):
    r500 = bundle.ef_R500c if bundle.r500_mode == "cat" else bundle.ef_R500d

    def f(sm):
        j = bundle._ef_index[id(sm)]
        return np.exp(A / (1.0 + (sm.r / (xt * r500[j])) ** p))
    return f
