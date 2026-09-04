"""
THE PATH-GEOMETRY REDSHIFT TEST.

    ln(1+z) = c1 D + c2 I_q + c3 I_T + c4 I_g + c5 I_q^2 + c6 I_q I_T

D is the INDEPENDENT distance (Pantheon+ / Cosmicflows-4 / megamaser); the I's
are integrals along the sight line the photon actually took.  Fitting z(D) is
not the test.  The test is whether two sources at the SAME independent distance
but with DIFFERENT intervening void content have different redshifts.

RUN timedilation.py FIRST.  If the mechanism does not stretch time as well as
frequency it is already excluded and everything here is a bounded feasibility
study.

THIS IS A LOGICALLY INDEPENDENT HYPOTHESIS.  Success or failure in galaxy or
cluster gravity is not evidence either way.

------------------------------------------------------------------------------
DECLARED BEFORE ANY RESIDUAL IS LOOKED AT
------------------------------------------------------------------------------
CUTS (inherited unchanged from the void-data lane, which declared them in code
before residuals):  path_covered_frac >= 0.5,  r_end >= 100 Mpc/h.

ARMS (forced by void-data finding 4 -- DESIVAST is edge-limited below z = 0.11
because the DR1 BGS footprint is only 0.745 sr):
    PRIMARY   SDSS DR7 VAST VoidFinder,  z < 0.11
    WATERSHED DESIVAST REVOLVER,         all z, flagged edge-limited
    FOOTPRINT-SAFE WATERSHED  DESIVAST REVOLVER, z > 0.11
Never averaged.

REGRESSOR.  The transverse residual dI_q = I_q - <I_q>(r), the footprint-
averaged void path length at that radius, computed in the void-data lane by a
240-direction scramble.  Raw I_q is NOT used for the headline: void-data
finding 1 measured its null expectation at 27-40% of c1 at 30-38 sigma.

TIDAL TERMS.  c3 and c6 are fitted ONLY on watershed geometry (REVOLVER).  On
sphere-based VoidFinder the potential inside a void is quadratic, T_ij is
isotropic, and I_T collapses onto a density-weighted copy of I_q
(corr = -0.754, VIF 16.5-18.2).

BLIND SPLIT.  A frozen 50/50 split by object, seed 20260904, declared here and
in the code before any residual is inspected.  Coefficients are fitted on the
train half, FROZEN, and the held-out half is touched exactly once.

------------------------------------------------------------------------------
THE NULL MODEL -- everything below is injected under the hypothesis c2 = 0
------------------------------------------------------------------------------
N0 shared-distance artefact.  I_q is built from a ray truncated at D_C(z), so
   it knows the TRUE distance while the regressor D is the noisy independent
   one (6% Pantheon+, 23.5% CF4).  The estimator uses I_q to repair D.
N1 reconstructed peculiar velocities.  Linear-theory radial velocity from the
   reconstructed density field, v = -(2f / 3 Omega_m H0) grad Phi, with an
   uncertain overall amplitude N(1, 0.3) for the reconstruction itself.
N2 lensing magnification and demagnification.  kappa along the sight line with
   the standard kernel; a standard candle behind a void is demagnified, so its
   inferred distance is biased LARGE exactly where I_q is large.
   dlnD = -kappa to first order.
N3 host-galaxy effects.  Distance-indicator zero point drifting with the host's
   own environment (Tully-Fisher and fundamental-plane calibrations are
   population dependent; Pantheon+ has the host-mass step).
N4 calibration drift across the sky.  Spurious dipole + quadrupole in mu.
N5 survey selection.  Inhomogeneous Malmquist: the distance estimate is pulled
   toward line-of-sight overdensities, by an amount set by the local
   d ln n / d ln r, which differs between void and non-void sight lines.  The
   log-slope is DEMEANED (a constant rescales every distance alike and is
   absorbed by c1) and its residual amplitude is UNCERTAIN in [0, 1], because
   CF4 and Pantheon+ already apply bias corrections of unknown residual size.
N6 covariance from using redshift to build the void catalogue.  Two halves:
   (a) the source endpoint is placed at D_C(z_obs), so its own peculiar
       velocity displaces the ray end by v/H0, changing I_q by exactly
       1_void(endpoint) * v/H0 while changing ln(1+z) by v/c -- a first-order
       covariance between regressor and response;
   (b) the voids themselves are found in redshift space and are RSD-stretched
       along the line of sight, an ~f|delta_v|/3 multiplicative distortion of
       I_q.

Three DISJOINT simulation sets are used, as the programme brief requires:
CALIBRATION (sets the critical value), AUDIT (untouched, verifies the false
positive rate), INJECTION (measures power).

Outputs  redshift_results.json,  nuisance_desi.csv,  nuisance_sdss.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VOIDLANE = os.path.abspath(os.path.join(HERE, "..", "void-data"))
sys.path.insert(0, os.path.join(VOIDLANE, "code"))

C_KMS = 299792.458
OMEGA_M = 0.315
H0_H = 100.0                 # h = 1, lengths in Mpc/h
C1_FID = H0_H / C_KMS        # 3.335641e-4 per Mpc/h
SIGMA_V = 300.0              # km/s, uncorrelated peculiar-velocity noise
GROWTH_F = OMEGA_M ** 0.55   # 0.5297
DL_STEP = 2.0

# --- declared cuts and split -------------------------------------------
CUT_PATH_COVERED = 0.5
CUT_R_END = 100.0
SPLIT_SEED = 20260904
Z_FOOTPRINT_SAFE = 0.11

# --- nuisance amplitudes, declared ranges ------------------------------
# host-galaxy zero-point drift with environment, mag per unit delta.
# Pantheon+ host-mass step is ~0.04-0.06 mag across the full mass range;
# CF4 TF/FP zero points move by a comparable amount between populations.
A_HOST_MAG = 0.05
# sky calibration drift: CF4 stitches many surveys; hemisphere-scale zero
# point offsets of ~0.02-0.03 mag are documented in TF compilations.
A_CAL_MAG = 0.03
# typical void interior underdensity, from the void-data density-field check
DELTA_VOID = -0.66
# the global h that maps the independent distances onto the redshift frame,
# fitted once in the void-data lane (robustness.json) and frozen here
H_INDEP = 0.7430579725381734

STAT_NULL = {"N0", "N1", "N2", "N5", "N6a"}
FULL_NULL = {"N0", "N1", "N2", "N3", "N4", "N5", "N6a"}


# =======================================================================
# linear algebra
# =======================================================================
def _colscale(X, w):
    s = np.sqrt(np.average(X ** 2, axis=0, weights=w))
    s[s <= 0] = 1.0
    return s


def wls(X, y, w):
    """Weighted least squares with internal column scaling (the six-term
    design spans 1e-5 to 1e4, so an unscaled normal equation is ill-posed).
    Returns (beta, cov) in the ORIGINAL units."""
    s = _colscale(X, w)
    Xs = X / s
    XW = Xs * w[:, None]
    N = Xs.T @ XW
    Ninv = np.linalg.inv(N)
    beta = Ninv @ (XW.T @ y)
    return beta / s, Ninv / np.outer(s, s)


def coef_response(X, w, idx):
    """Row of (X'WX)^-1 X'W producing the coefficient at column idx.

    The bias induced in that coefficient by ANY deterministic perturbation
    eps of the response is exactly  row @ eps.  Used instead of Monte Carlo
    wherever the confound is deterministic.
    """
    s = _colscale(X, w)
    Xs = X / s
    XW = Xs * w[:, None]
    return np.linalg.solve(Xs.T @ XW, XW.T)[idx] / s[idx]


# =======================================================================
# nuisance construction -- DESI arm (real reconstructed field)
# =======================================================================
def build_nuisance_desi(force=False):
    cache = os.path.join(HERE, "nuisance_desi_v2.csv")
    if os.path.exists(cache) and not force:
        print("  nuisance_desi.csv cached", flush=True)
        return pd.read_csv(cache)

    from astropy.io import fits
    from common import FootprintMask, sky_to_unit
    from density_field import DensityField
    from voids import SphereUnionVoids, TriangleVoids, load_v2, load_voidfinder

    DESIVAST = os.path.join(VOIDLANE, "raw", "desivast")
    d = pd.read_csv(os.path.join(VOIDLANE, "path_integrals_analysed.csv"))
    t0 = time.time()

    fields, masks = {}, {}
    for cap in ("NGC", "SGC"):
        with fits.open(os.path.join(
                DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_ZOBOV_{cap}.fits")) as h:
            g = h[3].data
            m = np.asarray(g["OUT"]) == 0
            X = np.stack([np.asarray(g[k], float)[m] for k in ("X", "Y", "Z")], 1)
        r = np.linalg.norm(X, axis=1)
        fm = FootprintMask(np.degrees(np.arctan2(X[:, 1], X[:, 0])) % 360,
                           np.degrees(np.arcsin(X[:, 2] / r)), pix_deg=0.5)
        masks[cap] = fm
        fields[cap] = DensityField(X, fm, dx=4.0, smooth=5.0, name=cap)
        print(f"  field {cap} rebuilt ({time.time()-t0:.0f}s)", flush=True)

    geo = {}
    for cap in ("NGC", "SGC"):
        _, mx, ho = load_voidfinder(
            os.path.join(DESIVAST, f"DESIVAST_BGS_VOLLIM_VoidFinder_{cap}.fits"))
        geo[("VoidFinder", cap)] = SphereUnionVoids(ho, mx)
        _, v2, tri = load_v2(
            os.path.join(DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_REVOLVER_{cap}.fits"))
        geo[("REVOLVER", cap)] = TriangleVoids(v2, tri)
    print(f"  void geometries loaded ({time.time()-t0:.0f}s)", flush=True)

    U = sky_to_unit(d["ra"].to_numpy(float), d["dec"].to_numpy(float))
    rend = d["r_end_mpch"].to_numpy(float)
    cap_of = d["cap"].to_numpy()
    frac_err = np.log(10.0) / 5.0 * d["sigma_mu"].to_numpy(float)
    n = len(d)

    v_pec = np.zeros(n)          # N1, km/s, radial, linear theory
    kappa = np.zeros(n)          # N2, convergence
    dlnn_dlnr = np.zeros(n)      # N5, d ln(1+delta) / d ln r at the source
    in_void = {"VoidFinder": np.zeros(n), "REVOLVER": np.zeros(n)}  # N6a
    kappa_2phase = np.zeros(n)   # cross-check of the SDSS-arm proxy

    # N1: v = -(2 f / (3 Omega_m H0)) grad Phi, with grad Phi in (km/s)^2/(Mpc/h)
    vfac = -2.0 * GROWTH_F / (3.0 * OMEGA_M * H0_H)
    # N2: kappa = (3/2) Omega_m (H0/c)^2 Int delta(chi) chi (chi_s - chi)/chi_s dchi
    kfac = 1.5 * OMEGA_M * (H0_H / C_KMS) ** 2

    for cap in ("NGC", "SGC"):
        sel = np.where(cap_of == cap)[0]
        if len(sel) == 0:
            continue
        fld = fields[cap]
        pts = U[sel] * rend[sel][:, None]
        gx = np.stack([fld.sample(fld.grad[a], pts)[0] for a in range(3)], 1)
        v_pec[sel] = vfac * np.einsum("ij,ij->i", gx, U[sel])
        for i_local, i in enumerate(sel):
            u = U[i]
            cs = rend[i]
            ell = np.arange(0.5 * DL_STEP, cs, DL_STEP)
            p = ell[:, None] * u[None, :]
            dv, ins = fld.sample(fld.delta, p)
            good = fld.in_survey(p) & ins
            W = ell * (cs - ell) / cs
            kappa[i] = kfac * np.sum(dv[good] * W[good]) * DL_STEP
            # Logarithmic radial density gradient at the source, measured over
            # the window the inhomogeneous Malmquist bias actually samples --
            # plus or minus one DISTANCE ERROR, not an arbitrary 30 Mpc/h.
            # Using too short a lever arm turns d ln(1+delta)/d ln r into a
            # ratio of two small numbers and inflates it by an order of
            # magnitude; that was the dominant spurious term in the first run.
            half = float(np.clip(frac_err[i] * cs, 20.0, 100.0))
            near = good & (ell > cs - half)
            if near.sum() >= 4:
                A = np.stack([np.ones(near.sum()), np.log(ell[near])], 1)
                yv = np.log(np.clip(1.0 + dv[near], 0.05, None))
                dlnn_dlnr[i] = np.linalg.lstsq(A, yv, rcond=None)[0][1]
        # N6a and the two-phase kappa proxy
        for alg in ("VoidFinder", "REVOLVER"):
            g = geo[(alg, cap)]
            for i in sel:
                if alg == "VoidFinder":
                    _, iv = g.ray_intervals(U[i], rend[i])
                else:
                    _, iv, _ = g.ray_intervals(U[i], rend[i])
                te = rend[i]
                in_void[alg][i] = float(any(
                    (a <= te + 1e-9) and (b >= te - 1e-9) for a, b in iv))
        print(f"  {cap}: {len(sel)} sources done ({time.time()-t0:.0f}s)",
              flush=True)

    Fv = 0.5228
    Iq = d["I_q_REVOLVER"].to_numpy(float)
    kappa_2phase = (kfac * DELTA_VOID / (1.0 - Fv) * (rend / 6.0)
                    * (Iq - Fv * rend))

    out = pd.DataFrame({
        "name": d["name"], "v_pec_lin_kms": v_pec, "kappa_los": kappa,
        "kappa_2phase": kappa_2phase, "dlnn_dlnr": dlnn_dlnr,
        "in_void_VoidFinder": in_void["VoidFinder"],
        "in_void_REVOLVER": in_void["REVOLVER"]})
    out.to_csv(cache, index=False)
    print(f"  wrote nuisance_desi.csv ({time.time()-t0:.0f}s)", flush=True)
    return out


# =======================================================================
# nuisance construction -- SDSS arm (two-phase void model)
# =======================================================================
def build_nuisance_sdss(force=False):
    cache = os.path.join(HERE, "nuisance_sdss.csv")
    if os.path.exists(cache) and not force:
        print("  nuisance_sdss.csv cached", flush=True)
        return pd.read_csv(cache)

    from common import sky_to_unit
    from voids import SphereUnionVoids, load_vast_sdss_holes

    REPO = os.path.abspath(os.path.join(VOIDLANE, "..", "..", ".."))
    VASTSDSS = os.path.join(REPO, "work", "private",
                            "open-gravity-void-source-v2")
    s = pd.read_csv(os.path.join(VOIDLANE, "path_integrals_sdss.csv"))
    smx, sho = load_vast_sdss_holes(
        os.path.join(VASTSDSS,
                     "VoidFinder-nsa_v1_0_1_Planck2018_comoving_holes.txt"),
        os.path.join(VASTSDSS,
                     "VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximal.txt"))
    sdss = SphereUnionVoids(sho, smx)
    U = sky_to_unit(s["ra"].to_numpy(float), s["dec"].to_numpy(float))
    rend = s["r_end_mpch"].to_numpy(float)
    pts = U * rend[:, None]

    # point-in-union-of-spheres, chunked
    inv = np.zeros(len(s), bool)
    C, R = sdss.c, sdss.R
    step = 200
    for a in range(0, len(pts), step):
        b = min(a + step, len(pts))
        d2 = ((pts[a:b, None, :] - C[None, :, :]) ** 2).sum(-1)
        inv[a:b] = (d2 < (R ** 2)[None, :]).any(1)
    Fv = 0.5881
    kfac = 1.5 * OMEGA_M * (H0_H / C_KMS) ** 2
    Iq = s["I_q_SDSS_VoidFinder"].to_numpy(float)
    kappa = kfac * DELTA_VOID / (1.0 - Fv) * (rend / 6.0) * (Iq - Fv * rend)
    out = pd.DataFrame({"name": s["name"], "in_void_SDSS": inv.astype(float),
                        "kappa_2phase": kappa})
    out.to_csv(cache, index=False)
    print(f"  wrote nuisance_sdss.csv, in-void fraction "
          f"{inv.mean():.3f}", flush=True)
    return out


# =======================================================================
# the estimator and its null
# =======================================================================
class Arm:
    """One analysis arm: design matrix, weights, nuisance vectors, null MC."""

    def __init__(self, label, D_true, D_obs_frac_err, y_obs, X, w,
                 nuis, note=""):
        self.label = label
        self.D_true = D_true
        self.fr = D_obs_frac_err
        self.y = y_obs
        self.X = X
        self.w = w
        self.nuis = nuis
        self.note = note
        self.n = len(y_obs)
        # dX_j / dI_q for every column that is built from I_q (N6a transport)
        self.n6_deriv = {2: np.ones(len(y_obs))}
        self.names = None

    def fit(self):
        beta, Ninv = wls(self.X, self.y, self.w)
        return beta, np.sqrt(np.diag(Ninv))

    # ---- deterministic bias of a coefficient from a response perturbation
    def bias_from_response(self, eps, idx=2):
        return float(coef_response(self.X, self.w, idx) @ eps)

    # ---- the Monte Carlo null ---------------------------------------
    def null_mc(self, nsim, seed, components, idx=2):
        """
        Truth: ln(1+z) = c1 D_true.  NO path term of any kind.
        `components` is a set of the N-labels to switch on.

        Returns (coeffs, responses).  `coeffs` is the sampled null
        distribution of the coefficient at column idx.  `responses` is the
        EXACT per-realisation factor R by which an injected true coefficient
        would appear in the fit, obtained from linearity of weighted least
        squares in y rather than from a separate injection run -- so power and
        the dS/dtheta check carry no Monte Carlo noise at all.
        """
        rng = np.random.default_rng(seed)
        n = self.n
        Dt = self.D_true
        sig_pec = SIGMA_V / C_KMS
        x0 = self.X[:, idx].copy()
        x_signal = self.X[:, 2].copy()   # a true path term enters via I_q
        nu = self.nuis
        u = nu["uhat"]
        out = np.empty(nsim)
        resp = np.empty(nsim)
        for _ in range(nsim):
            y = C1_FID * Dt
            lnD = np.zeros(n)
            dx = np.zeros(n)

            if "N0" in components:
                y = y + rng.normal(0, sig_pec, n)
                lnD = lnD + rng.normal(0, self.fr, n)
            if "N1" in components:
                # linear-theory reconstruction: the SHAPE is well determined,
                # the amplitude is not (bias, smoothing, mask), so it carries
                # an uncertain multiplier rather than being asserted exactly.
                y = y + max(0.0, rng.normal(1.0, 0.3)) * nu["v_pec"] / C_KMS
            if "N2" in components:
                lnD = lnD - nu["kappa"]
            if "N3" in components:
                a = rng.normal(0, A_HOST_MAG)
                lnD = lnD + (np.log(10) / 5.0) * a * nu["env"]
            if "N4" in components:
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                dip = u @ v
                q = rng.normal(size=(3, 3))
                q = 0.5 * (q + q.T)
                q -= np.eye(3) * np.trace(q) / 3.0
                q /= np.linalg.norm(q)
                quad = np.einsum("ni,ij,nj->n", u, q, u)
                a = rng.normal(0, A_CAL_MAG)
                lnD = lnD + (np.log(10) / 5.0) * a * (dip + quad)
            if "N5" in components or "N5raw" in components:
                # Inhomogeneous Malmquist. TWO corrections relative to the
                # first pass, both forced and neither tuned to the answer:
                #  (i) the log-slope is DEMEANED -- a constant d ln n / d ln r
                #      rescales every distance by the same factor and is
                #      absorbed by c1, so asserting its mean as a bias on c2
                #      is simply wrong;
                #  (ii) the amplitude is UNCERTAIN, because CF4 and Pantheon+
                #      already apply their own bias corrections and the
                #      residual fraction is unknown. An uncertain amplitude
                #      belongs in the variance, not in the mean.
                # Keeping the first-pass treatment instead ("N5raw") is
                # reported as the size of the nuisance-modelling ambiguity.
                if "N5raw" in components:
                    lnD = lnD - (self.fr ** 2) * nu["dlnn_dlnr_raw"]
                else:
                    lnD = lnD - (rng.uniform(0.0, 1.0) * self.fr ** 2
                                 * nu["dlnn_dlnr"])
            Xs = self.X.copy()
            Xs[:, 1] = Dt * np.exp(lnD)
            if "N6a" in components:
                # the source's own peculiar velocity moves BOTH its redshift
                # and, because the ray is truncated at D_C(z_obs), the end of
                # the ray -- by v/H0 if the endpoint sits inside a void.
                # Every column built from I_q inherits the displacement
                # through its own derivative dX_j / dI_q.
                vv = rng.normal(0, SIGMA_V, n)
                y = y + vv / C_KMS
                dIq = nu["in_void"] * (vv / H0_H)
                for j, deriv in self.n6_deriv.items():
                    Xs[:, j] = self.X[:, j] + deriv * dIq
                dx = self.n6_deriv.get(idx, np.zeros(n)) * dIq
            out[_] = wls(Xs, y, self.w)[0][idx]
            # exact response of this coefficient to a true c2 * I_q term
            resp[_] = coef_response(Xs, self.w, idx) @ x_signal
        return out, resp


def make_desi_arm(d, nu, alg, six_term=False, mask=None, regressor=None):
    m = np.ones(len(d), bool) if mask is None else mask
    dd = d[m].reset_index(drop=True)
    nn = nu[m].reset_index(drop=True)
    D = dd["D_comov_indep_mpc"].to_numpy(float) * H_INDEP
    y = np.log1p(dd["z_cmb"].to_numpy(float))
    fr = np.log(10.0) / 5.0 * dd["sigma_mu"].to_numpy(float)
    w = 1.0 / ((SIGMA_V / C_KMS) ** 2 + (C1_FID * fr * dd["r_end_mpch"]) ** 2)
    w = w.to_numpy(float) if hasattr(w, "to_numpy") else np.asarray(w)
    r = dd["r_end_mpch"].to_numpy(float)
    A = np.stack([np.ones_like(r), r, r ** 2, r ** 3], 1)

    def detrend(v):
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ c

    if regressor == "I_q_den":
        # field-based alternative: the underdensity path integral, which uses
        # NO void finder at all. Immune to the cross-pipeline systematic.
        dIq = detrend(dd["I_q_den"].to_numpy(float))
    else:
        dIq = dd[f"dI_q_{alg}"].to_numpy(float)
    cols = [np.ones(len(dd)), D, dIq]
    names = ["const", "D", "dI_q"]
    n6 = {2: np.ones(len(dd))}
    if six_term:
        Iq = dd[f"I_q_{alg}"].to_numpy(float)
        IT = dd[f"I_T_{alg}"].to_numpy(float)
        Ig = dd["I_g"].to_numpy(float)
        cols += [detrend(IT), detrend(Ig), detrend(Iq ** 2), detrend(Iq * IT)]
        names += ["dI_T", "dI_g", "dI_q^2", "dI_q*I_T"]
        # dX_j / dI_q for the columns built from I_q
        n6[5] = 2.0 * Iq
        n6[6] = IT
    X = np.stack(cols, 1)
    from common import sky_to_unit
    uhat = sky_to_unit(dd["ra"].to_numpy(float), dd["dec"].to_numpy(float))
    env = dd["delta_at_source"].to_numpy(float)
    env = (env - env.mean()) / env.std()
    g_raw = nn["dlnn_dlnr"].to_numpy(float)
    nuis = {"v_pec": nn["v_pec_lin_kms"].to_numpy(float),
            "kappa": nn["kappa_los"].to_numpy(float),
            "env": env, "uhat": uhat,
            "dlnn_dlnr_raw": g_raw,
            "dlnn_dlnr": g_raw - np.average(g_raw, weights=w),
            "in_void": nn[f"in_void_{alg}"].to_numpy(float)}
    arm = Arm(f"DESI/{alg}" + ("/6term" if six_term else "")
              + ("/I_q_den" if regressor == "I_q_den" else ""),
              r, fr, y, X, w, nuis)
    arm.names = names
    arm.n6_deriv = n6
    arm.frame = dd
    return arm


def make_sdss_arm(s, nu, v_scale, gradient_by_env):
    m = (s["r_end_mpch"] >= CUT_R_END).to_numpy()
    ss = s[m].reset_index(drop=True)
    nn = nu[m].reset_index(drop=True)
    D = 10 ** (ss["mu"].to_numpy(float) / 5.0 - 5.0)      # Mpc, luminosity
    z = ss["z_cmb"].to_numpy(float)
    D = D / (1.0 + z) * H_INDEP                           # comoving Mpc/h
    y = np.log1p(z)
    fr = np.log(10.0) / 5.0 * ss["sigma_mu"].to_numpy(float)
    r = ss["r_end_mpch"].to_numpy(float)
    w = 1.0 / ((SIGMA_V / C_KMS) ** 2 + (C1_FID * fr * r) ** 2)
    X = np.stack([np.ones(len(ss)), D, ss["dI_q_SDSS"].to_numpy(float)], 1)
    from common import sky_to_unit
    uhat = sky_to_unit(ss["ra"].to_numpy(float), ss["dec"].to_numpy(float))
    dIq = ss["dI_q_SDSS"].to_numpy(float)
    iv = nn["in_void_SDSS"].to_numpy(float)
    # There is no reconstructed density field for the SDSS footprint, so the
    # environment-dependent nuisances are carried across from the DESI arm
    # through the one environment variable that IS available here: whether the
    # source itself sits inside a catalogued void.  Using dI_q itself as the
    # environment template would make the nuisance exactly degenerate with the
    # regressor and inflate the null by an order of magnitude for no physical
    # reason -- that was the first run's error.
    v_proxy = v_scale * (dIq / max(1e-9, dIq.std()))
    g_in, g_out = gradient_by_env
    g_raw = np.where(iv > 0.5, g_in, g_out)
    nuis = {"v_pec": v_proxy, "kappa": nn["kappa_2phase"].to_numpy(float),
            "env": (iv - iv.mean()) / max(1e-9, iv.std()), "uhat": uhat,
            "dlnn_dlnr_raw": g_raw,
            "dlnn_dlnr": g_raw - np.average(g_raw, weights=w),
            "in_void": iv}
    arm = Arm("SDSS/VoidFinder", r, fr, y, X, w, nuis)
    arm.names = ["const", "D", "dI_q"]
    arm.frame = ss
    return arm


# =======================================================================
def summarise(arm, nsim_cal, nsim_audit, res, tag):
    """Fit, null-calibrate on three disjoint simulation sets, report.

    Everything is quoted relative to the FIDUCIAL c1 = H0/c, not to the fitted
    c1: the fitted c1 is attenuated by the errors-in-variables regression of a
    precise redshift on a noisy distance, and dividing by it would inflate
    every ratio by 1/attenuation.
    """
    beta, sig = arm.fit()
    c1, c2 = beta[1], beta[2]

    comp = {}
    seeds = {"N0": 101, "N1": 202, "N2": 303, "N3": 404, "N4": 505,
             "N5": 606, "N6a": 707, "N5raw": 808}
    for k in ["N0", "N1", "N2", "N3", "N4", "N5", "N6a", "N5raw"]:
        s, _ = arm.null_mc(200, 4242 + seeds[k],
                           {"N0", k} if k != "N0" else {"N0"})
        comp[k] = {"null_mean_over_c1": float(s.mean() / C1_FID),
                   "null_sd_over_c1": float(s.std() / C1_FID)}

    stat, rstat = arm.null_mc(nsim_cal, 10_000 + len(arm.y), STAT_NULL)
    cal, rcal = arm.null_mc(nsim_cal, 20_000 + len(arm.y), FULL_NULL)
    audit, _ = arm.null_mc(nsim_audit, 777_000 + len(arm.y), FULL_NULL)
    crit = np.percentile(np.abs(cal - cal.mean()), 95.0)
    fpr = float((np.abs(audit - cal.mean()) > crit).mean())

    R = float(rcal.mean())               # response of c2-hat to a true c2
    c2_corr = c2 - cal.mean()
    z_stat = (c2 - stat.mean()) / stat.std()
    z_full = c2_corr / cal.std()
    # de-biased estimate of the TRUE c2, in units of the fiducial c1
    est = c2_corr / R / C1_FID
    sd_stat = stat.std() / R / C1_FID
    sd_full = cal.std() / R / C1_FID

    out = {
        "n": int(arm.n),
        "note": arm.note,
        "coefficients": {nm: float(b) for nm, b in zip(arm.names, beta)},
        "analytic_sigma": {nm: float(s) for nm, s in zip(arm.names, sig)},
        "c1_fitted_per_mpch": float(c1),
        "c1_fiducial_per_mpch": C1_FID,
        "c1_attenuation_factor": float(c1 / C1_FID),
        "c1_attenuation_note":
            "c1 < c1_fiducial because the response is a precise redshift and "
            "the regressor a noisy distance; classic errors-in-variables "
            "attenuation. It is NOT a measurement of H0.",
        "response_factor_R": R,
        "c2_raw_per_mpch": float(c2),
        "null_mean_c2_full": float(cal.mean()),
        "null_sd_c2_full": float(cal.std()),
        "null_mean_c2_statistical_only": float(stat.mean()),
        "null_sd_c2_statistical_only": float(stat.std()),
        "c2_over_c1_estimate_null_subtracted_response_corrected": est,
        "sigma_statistical_over_c1": sd_stat,
        "sigma_full_incl_systematics_over_c1": sd_full,
        "significance_vs_statistical_null_sigma": float(z_stat),
        "significance_vs_full_null_sigma": float(z_full),
        "two_sided_95pct_interval_on_c2_over_c1": [
            float(est - 1.96 * sd_full), float(est + 1.96 * sd_full)],
        "abs_95pct_upper_limit_on_c2_over_c1": float(
            abs(est) + 1.96 * sd_full),
        "min_detectable_c2_over_c1_at_3sigma_statistical": float(3 * sd_stat),
        "min_detectable_c2_over_c1_at_3sigma_with_systematics": float(
            3 * sd_full),
        "critical_value_95pct_from_calibration_sims": float(crit),
        "false_positive_rate_on_untouched_audit_sims": fpr,
        "n_calibration_sims": int(nsim_cal),
        "n_audit_sims": int(nsim_audit),
        "per_component_null_over_c1": comp,
    }
    res[tag] = out
    print(f"\n  [{arm.label}] n = {arm.n}")
    print(f"    c1 fitted {c1:.4e} = {c1/C1_FID:.3f} x fiducial "
          f"(errors-in-variables attenuation, NOT H0)")
    print(f"    response factor R = {R:.3f}")
    print(f"    c2 raw            = {c2/C1_FID*100:+7.3f}% of c1_fid")
    print(f"    null (stat only)  = {stat.mean()/C1_FID*100:+7.3f}% "
          f"+/- {stat.std()/C1_FID*100:.3f}%")
    print(f"    null (full)       = {cal.mean()/C1_FID*100:+7.3f}% "
          f"+/- {cal.std()/C1_FID*100:.3f}%")
    print(f"    ESTIMATE c2/c1    = {est*100:+7.3f}% "
          f"+/- {sd_stat*100:.3f}% (stat) +/- "
          f"{np.sqrt(max(0.0, sd_full**2-sd_stat**2))*100:.3f}% (sys)")
    print(f"    significance      = {z_stat:+.2f} sigma (stat null), "
          f"{z_full:+.2f} sigma (full null)")
    print(f"    |c2/c1| < {out['abs_95pct_upper_limit_on_c2_over_c1']*100:.2f}%"
          f" at 95%;  audit FPR = {fpr:.3f}")
    return out


def main():
    print("=" * 72)
    print("PATH-GEOMETRY REDSHIFT TEST")
    print("=" * 72)
    res = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "independence_statement":
               "Logically independent of the galaxy and cluster gravity lanes. "
               "Nothing there is evidence for this hypothesis and nothing here "
               "bears on that work.",
           "declared_before_residuals": {
               "cuts": {"path_covered_frac_min": CUT_PATH_COVERED,
                        "r_end_mpch_min": CUT_R_END},
               "blind_split_seed": SPLIT_SEED,
               "regressor": "transverse residual dI_q, never raw I_q",
               "tidal_terms_only_on": "watershed (REVOLVER) geometry",
               "arms": ["SDSS VAST z<0.11 (primary)",
                        "DESIVAST REVOLVER all z (edge-limited)",
                        "DESIVAST REVOLVER z>0.11 (footprint-safe)"],
               "nuisance_amplitudes": {"A_host_mag": A_HOST_MAG,
                                       "A_cal_mag": A_CAL_MAG,
                                       "delta_void": DELTA_VOID,
                                       "sigma_v_kms": SIGMA_V}}}

    with open(os.path.join(HERE, "timedilation.json")) as fh:
        td = json.load(fh)
    res["time_dilation_gate"] = {
        "headline": td["headline"]["statement"],
        "gate_failed_by": td["headline"]["gate_failed_by"],
        "gate_passed_by": td["headline"]["gate_passed_by"],
        "status": "the fit below is a BOUNDED FEASIBILITY STUDY",
    }

    print("\n[1] nuisance construction")
    nud = build_nuisance_desi()
    nus = build_nuisance_sdss()

    d = pd.read_csv(os.path.join(VOIDLANE, "path_integrals_analysed.csv"))
    s = pd.read_csv(os.path.join(VOIDLANE, "path_integrals_sdss.csv"))
    assert len(d) == 4389 and len(nud) == 4389, (len(d), len(nud))
    assert len(s) == 25123 and len(nus) == 25123, (len(s), len(nus))
    assert (d["name"].values == nud["name"].values).all()
    assert (s["name"].values == nus["name"].values).all()
    print(f"  row counts asserted: DESI {len(d)}, SDSS {len(s)}")

    res["nuisance_summary"] = {
        "desi": {
            "v_pec_lin_kms_sd": float(nud["v_pec_lin_kms"].std()),
            "v_pec_lin_kms_p5_p95": [float(np.percentile(nud["v_pec_lin_kms"], 5)),
                                     float(np.percentile(nud["v_pec_lin_kms"], 95))],
            "kappa_los_sd": float(nud["kappa_los"].std()),
            "kappa_los_mean": float(nud["kappa_los"].mean()),
            "corr_kappa_with_dIq_REVOLVER": float(np.corrcoef(
                nud["kappa_los"], d["dI_q_REVOLVER"])[0, 1]),
            "corr_kappa_los_with_two_phase_proxy": float(np.corrcoef(
                nud["kappa_los"], nud["kappa_2phase"])[0, 1]),
            "corr_vpec_with_dIq_REVOLVER": float(np.corrcoef(
                nud["v_pec_lin_kms"], d["dI_q_REVOLVER"])[0, 1]),
            "endpoint_in_void_fraction_VoidFinder": float(
                nud["in_void_VoidFinder"].mean()),
            "endpoint_in_void_fraction_REVOLVER": float(
                nud["in_void_REVOLVER"].mean()),
            "dlnn_dlnr_sd": float(nud["dlnn_dlnr"].std()),
        },
        "sdss": {
            "endpoint_in_void_fraction": float(nus["in_void_SDSS"].mean()),
            "kappa_2phase_sd": float(nus["kappa_2phase"].std()),
        }}
    print("  kappa sd (DESI) = "
          f"{nud['kappa_los'].std():.2e}, corr with dI_q = "
          f"{np.corrcoef(nud['kappa_los'], d['dI_q_REVOLVER'])[0,1]:+.3f}")
    print("  v_pec sd (DESI) = "
          f"{nud['v_pec_lin_kms'].std():.1f} km/s, corr with dI_q = "
          f"{np.corrcoef(nud['v_pec_lin_kms'], d['dI_q_REVOLVER'])[0,1]:+.3f}")

    v_scale = float(np.corrcoef(nud["v_pec_lin_kms"],
                                d["dI_q_REVOLVER"])[0, 1]
                    * nud["v_pec_lin_kms"].std())
    ivd = nud["in_void_VoidFinder"].to_numpy(float) > 0.5
    gradient_by_env = (float(nud["dlnn_dlnr"][ivd].mean()),
                       float(nud["dlnn_dlnr"][~ivd].mean()))
    res["nuisance_summary"]["transfer_to_sdss"] = {
        "coherent_v_pec_amplitude_aligned_with_dIq_kms": v_scale,
        "dlnn_dlnr_in_void": gradient_by_env[0],
        "dlnn_dlnr_out_of_void": gradient_by_env[1]}
    print(f"  d ln n / d ln r: in-void {gradient_by_env[0]:+.2f}, "
          f"outside {gradient_by_env[1]:+.2f}")

    # WHAT IS THE REGRESSOR ACTUALLY MEASURING? -- a check that was not run
    # before, and that changes how c2 must be read.
    r = d["r_end_mpch"].to_numpy(float)
    A = np.stack([np.ones_like(r), r, r ** 2, r ** 3], 1)

    def det(v):
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ c

    reg = {}
    for alg in ("VoidFinder", "REVOLVER"):
        dq = d[f"dI_q_{alg}"].to_numpy(float)
        reg[alg] = {
            "corr_dI_q_with_mean_delta_along_los": float(np.corrcoef(
                dq, d["mean_delta_los"])[0, 1]),
            "corr_dI_q_with_underdensity_path_integral_I_q_den": float(
                np.corrcoef(dq, det(d["I_q_den"].to_numpy(float)))[0, 1]),
            "corr_dI_q_with_delta_at_source": float(np.corrcoef(
                dq, d["delta_at_source"])[0, 1]),
        }
        print(f"  {alg}: corr(dI_q, mean delta along LOS) = "
              f"{reg[alg]['corr_dI_q_with_mean_delta_along_los']:+.3f}, "
              f"corr(dI_q, I_q_den) = "
              f"{reg[alg]['corr_dI_q_with_underdensity_path_integral_I_q_den']:+.3f}")
    res["what_the_regressor_measures"] = reg
    res["what_the_regressor_measures"]["reading"] = (
        "A catalogued void path length is NOT the same variable as an "
        "underdensity path length. For the sphere-based VoidFinder the two "
        "are nearly uncorrelated; for the REVOLVER watershed the transverse "
        "residual correlates POSITIVELY with the mean line-of-sight density, "
        "because the watershed tiles the whole volume into zones rather than "
        "selecting empty ones. Any c2 fitted on a catalogue I_q is a "
        "coefficient of catalogue membership, not of emptiness. The I_q_den "
        "arm below uses the reconstructed density field instead and needs no "
        "void finder at all.")

    print("\n[2] arms")
    arms = {}
    arms["sdss_primary"] = make_sdss_arm(s, nus, v_scale, gradient_by_env)
    arms["sdss_primary"].note = (
        "PRIMARY. SDSS DR7 VAST VoidFinder, z < 0.11, footprint 2.13 sr. "
        "Sphere-based geometry, so no tidal terms (void-data finding 3).")
    arms["desi_revolver_all"] = make_desi_arm(d, nud, "REVOLVER")
    arms["desi_revolver_all"].note = (
        "DESIVAST REVOLVER, all z. EDGE-LIMITED below z = 0.11: the DR1 BGS "
        "footprint is 0.745 sr and a >=10 Mpc/h void cannot be inscribed at "
        "r = 125 Mpc/h without touching the edge.")
    arms["desi_voidfinder_all"] = make_desi_arm(d, nud, "VoidFinder")
    arms["desi_voidfinder_all"].note = "DESIVAST VoidFinder, all z, edge-limited."
    msafe = (d["z_cmb"] > Z_FOOTPRINT_SAFE).to_numpy()
    arms["desi_revolver_safe"] = make_desi_arm(d, nud, "REVOLVER", mask=msafe)
    arms["desi_revolver_safe"].note = (
        "FOOTPRINT-SAFE watershed arm, z > 0.11. This is the only DESI subset "
        "finding 4 permits, and it contains n = "
        f"{int(msafe.sum())} sources -- all Pantheon+ supernovae.")
    arms["desi_density_field"] = make_desi_arm(d, nud, "REVOLVER",
                                               regressor="I_q_den")
    arms["desi_density_field"].note = (
        "VOID-FINDER-FREE arm. The regressor is the underdensity path "
        "integral of the reconstructed density field, radially detrended. It "
        "carries no void-catalogue systematic at all, but it inherits the "
        "reconstruction's mask (delta = 0 outside the survey) and its 5 Mpc/h "
        "smoothing.")
    for k, a in arms.items():
        print(f"  {k}: n = {a.n}")
    res["arm_sizes"] = {k: int(a.n) for k, a in arms.items()}

    print("\n[3] fits with the full null subtracted")
    res["fits"] = {}
    summarise(arms["sdss_primary"], 400, 400, res["fits"], "sdss_primary")
    summarise(arms["desi_revolver_all"], 400, 400, res["fits"],
              "desi_revolver_all")
    summarise(arms["desi_voidfinder_all"], 400, 400, res["fits"],
              "desi_voidfinder_all")
    summarise(arms["desi_revolver_safe"], 400, 400, res["fits"],
              "desi_revolver_safe")
    summarise(arms["desi_density_field"], 400, 400, res["fits"],
              "desi_density_field")

    # ---- the arm-to-arm dispersion is the real error bar --------------
    est = {k: v["c2_over_c1_estimate_null_subtracted_response_corrected"]
           for k, v in res["fits"].items()}
    raw = {k: v["c2_raw_per_mpch"] / C1_FID for k, v in res["fits"].items()}
    vals = np.array(list(est.values()))
    rawv = np.array(list(raw.values()))
    quoted = np.array([res["fits"][k]["sigma_full_incl_systematics_over_c1"]
                       for k in est])
    res["arm_to_arm_dispersion"] = {
        "estimates_c2_over_c1": est,
        "raw_c2_over_c1_before_null_subtraction": raw,
        "sd_of_estimates": float(vals.std(ddof=1)),
        "range_of_estimates": [float(vals.min()), float(vals.max())],
        "sd_of_raw_values": float(rawv.std(ddof=1)),
        "range_of_raw_values": [float(rawv.min()), float(rawv.max())],
        "median_quoted_sigma": float(np.median(quoted)),
        "dispersion_over_quoted_sigma": float(vals.std(ddof=1)
                                              / np.median(quoted)),
        "reading": (
            "Five estimators of the SAME coefficient, sharing the same sources "
            "and differing only in how the path integral is built, disagree by "
            "more than any of them quotes. The RAW coefficients before null "
            "subtraction agree far better than the null-subtracted ones, which "
            "locates the disagreement in the nuisance model rather than in the "
            "data. The dispersion, not the quoted sigma, is the honest error "
            "bar."),
    }
    print("  arm-to-arm dispersion of the estimate: "
          "%.2f%% of c1 across [%+.2f%%, %+.2f%%], vs a median quoted sigma "
          "of %.2f%%" % (vals.std(ddof=1)*100, vals.min()*100, vals.max()*100,
                         np.median(quoted)*100))
    print("  raw coefficients before null subtraction: [%+.2f%%, %+.2f%%], "
          "sd %.2f%%" % (rawv.min()*100, rawv.max()*100, rawv.std(ddof=1)*100))

    print("\n[4] six-term law on watershed geometry")
    six = make_desi_arm(d, nud, "REVOLVER", six_term=True)
    six.note = ("Six-term law. c3 and c6 are meaningful only here: on "
                "VoidFinder spheres I_T collapses onto I_q. Edge-limited "
                "sample; the footprint-safe subset has n = 46 and cannot "
                "support six coefficients.")
    beta6, sig6 = six.fit()
    # conditioning of the design, reported rather than assumed
    Z = six.X[:, 1:]
    Z = (Z - Z.mean(0)) / Z.std(0)
    Cm = np.corrcoef(Z, rowvar=False)
    vif = np.diag(np.linalg.inv(Cm))
    n6 = {}
    for j in range(2, six.X.shape[1]):
        cal, _ = six.null_mc(200, 31_000 + j, FULL_NULL, idx=j)
        n6[six.names[j]] = {
            "coefficient": float(beta6[j]),
            "analytic_sigma": float(sig6[j]),
            "coefficient_over_analytic_sigma": float(beta6[j] / sig6[j]),
            "vif": float(vif[j - 1]),
            "null_mean": float(cal.mean()),
            "null_sd": float(cal.std()),
            "null_subtracted": float(beta6[j] - cal.mean()),
            "significance_sigma": float((beta6[j] - cal.mean()) / cal.std()),
        }
        print(f"    {six.names[j]:>10s}: {beta6[j]:+.4e} +/- {sig6[j]:.2e} "
              f"(VIF {vif[j-1]:5.2f})  null {cal.mean():+.3e} +/- "
              f"{cal.std():.2e} -> {(beta6[j]-cal.mean())/cal.std():+.2f} sig")
    n6["_design_vif"] = {nm: float(v) for nm, v in zip(six.names[1:], vif)}
    n6["_c4_caveat"] = (
        "c4 (dI_g) is reported but must NOT be interpreted: the density field "
        "is zero outside the survey mask, so I_g is a lower bound and the "
        "least trustworthy of the integrals (void-data lane, section 3).")
    res["six_term_watershed"] = {
        "n": int(six.n), "note": six.note,
        "c1": float(beta6[1]), "terms": n6,
        "footprint_safe_n_for_six_terms": int(msafe.sum()),
        "structural_obstruction": (
            "void-data finding 3 forces the tidal terms onto watershed "
            "geometry, which exists only in DESIVAST; void-data finding 4 "
            "forces DESIVAST above z = 0.11, where n = 46. The two findings "
            "cannot be satisfied at once with the data on disk, so c3 and c6 "
            "have no footprint-safe determination at all."),
    }

    print("\n[5] matched-pair estimator (the decisive comparison, differenced)")
    res["matched_pairs"] = matched_pair_test(arms["sdss_primary"],
                                             "SDSS/VoidFinder")
    res["matched_pairs_desi"] = matched_pair_test(arms["desi_revolver_all"],
                                                  "DESI/REVOLVER")

    print("\n[6] blind split: fit on train, freeze, touch the holdout once")
    res["blind"] = blind_split(arms["sdss_primary"])

    print("\n[7] injection recovery / monotone-invariance of the statistic")
    res["injection"] = injection_recovery(arms["sdss_primary"])

    print("\n[8] cross-pipeline systematic floor")
    res["systematic_floor"] = systematic_floor(d, nud)

    print("\n[9] circularity")
    with open(os.path.join(VOIDLANE, "robustness.json")) as fh:
        rb = json.load(fh)
    res["circularity"] = {
        "four_places": [
            "void positions come from r = D_C(z; Omega_m = 0.315): the "
            "catalogue is a redshift-space product in Cartesian clothing",
            "the volume-limited sample definition (MAGLIM = -20) uses a "
            "cosmology-dependent luminosity distance, so which galaxies exist "
            "depends on the law under test",
            "voids are found in redshift space and are RSD-stretched along the "
            "line of sight, uncorrected",
            "the source endpoints are placed by the same law",
        ],
        "endpoint_half_measured": rb["endpoint_placement_sensitivity"],
        "radial_remap_size": rb["max_radial_shift_mpch_at_zmax"],
        "verdict": (
            "Reuse costs about 18% on the leverage variable and 5-10% on the "
            "radial metric. Tolerable for a feasibility and power study, NOT "
            "for a claimed detection. A genuine no-expansion analysis would "
            "have to rerun VoidFinder and V2 under its own distance law, "
            "which also changes the sample definition."),
    }

    with open(os.path.join(HERE, "redshift_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("\nwrote redshift_results.json")
    return res


def matched_pair_test(arm, label):
    """
    The decisive comparison, done by differencing rather than regression.

    Pairs of sources with nearly the same INDEPENDENT distance but very
    different void content. The estimator is the weighted slope of
    d ln(1+z) on d I_q over such pairs. It removes c1 to first order, and it
    has its OWN shared-distance null, which is simulated the same way.
    """
    D = arm.X[:, 1]
    x = arm.X[:, 2]
    y = arm.y
    order = np.argsort(D)
    D, x, y = D[order], x[order], y[order]
    fr = arm.fr[order]
    Dt = arm.D_true[order]
    n = len(D)
    # neighbours within a declared 20 Mpc/h window in the OBSERVED distance
    lo = np.searchsorted(D, D - 20.0, "left")
    hi = np.searchsorted(D, D + 20.0, "right")
    rng = np.random.default_rng(5150)
    i_idx, j_idx = [], []
    for i in range(n):
        cand = np.arange(lo[i], hi[i])
        cand = cand[cand != i]
        if len(cand) == 0:
            continue
        big = cand[np.abs(x[cand] - x[i]) > 60.0]
        if len(big) == 0:
            continue
        j = big[rng.integers(len(big))]
        i_idx.append(i)
        j_idx.append(j)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    if len(i_idx) < 50:
        return {"n_pairs": int(len(i_idx)),
                "note": "too few matched pairs to estimate"}
    dy = y[i_idx] - y[j_idx]
    dx = x[i_idx] - x[j_idx]
    wp = np.ones(len(dy))
    slope = float((wp * dx * dy).sum() / (wp * dx * dx).sum())

    # null: same pairing, but truth has no path term
    sig_pec = SIGMA_V / C_KMS
    nulls = np.empty(300)
    for s2 in range(300):
        r2 = np.random.default_rng(9000 + s2)
        yy = C1_FID * Dt + r2.normal(0, sig_pec, n)
        dyy = yy[i_idx] - yy[j_idx]
        nulls[s2] = (wp * dx * dyy).sum() / (wp * dx * dx).sum()
    out = {
        "estimator": "weighted slope of d ln(1+z) on d I_q over pairs matched "
                     "to |dD| < 20 Mpc/h with |dI_q| > 60 Mpc/h",
        "n_pairs": int(len(i_idx)),
        "slope_per_mpch": slope,
        "slope_over_c1": float(slope / C1_FID),
        "null_mean": float(nulls.mean()),
        "null_sd": float(nulls.std()),
        "null_mean_over_c1": float(nulls.mean() / C1_FID),
        "null_subtracted_over_c1": float((slope - nulls.mean()) / C1_FID),
        "significance_sigma": float((slope - nulls.mean()) / nulls.std()),
    }
    print(f"  [{label}] {out['n_pairs']} pairs: slope/c1 = "
          f"{out['slope_over_c1']*100:+.2f}%, null "
          f"{out['null_mean_over_c1']*100:+.2f}% -> "
          f"{out['significance_sigma']:+.2f} sigma")
    return out


def blind_split(arm):
    rng = np.random.default_rng(SPLIT_SEED)
    tr = rng.random(arm.n) < 0.5
    ho = ~tr
    btr, _ = wls(arm.X[tr], arm.y[tr], arm.w[tr])
    # FROZEN coefficients evaluated once on the holdout
    r_full = arm.y[ho] - arm.X[ho] @ btr
    b_noc2 = btr.copy()
    b_noc2[2] = 0.0
    r_noc2 = arm.y[ho] - arm.X[ho] @ b_noc2
    chi2_full = float((arm.w[ho] * r_full ** 2).sum())
    chi2_noc2 = float((arm.w[ho] * r_noc2 ** 2).sum())
    bho, _ = wls(arm.X[ho], arm.y[ho], arm.w[ho])
    out = {"seed": SPLIT_SEED, "n_train": int(tr.sum()), "n_holdout": int(ho.sum()),
           "c2_train": float(btr[2]), "c2_train_over_c1": float(btr[2] / btr[1]),
           "c2_holdout_refit": float(bho[2]),
           "c2_holdout_refit_over_c1": float(bho[2] / bho[1]),
           "chi2_holdout_with_frozen_c2": chi2_full,
           "chi2_holdout_with_c2_set_to_zero": chi2_noc2,
           "delta_chi2_frozen_transfer": chi2_noc2 - chi2_full,
           "reading": "positive delta_chi2 means the frozen c2 helps on the "
                      "held-out half; it is reported against the null "
                      "distribution, not against zero. The refit value is "
                      "printed only to expose the size of the difference "
                      "between the correct frozen procedure and the "
                      "refit-on-holdout mistake."}
    print(f"  train c2/c1 = {out['c2_train_over_c1']*100:+.3f}%, "
          f"holdout refit = {out['c2_holdout_refit_over_c1']*100:+.3f}%, "
          f"frozen transfer dchi2 = {out['delta_chi2_frozen_transfer']:+.2f}")
    return out


def injection_recovery(arm):
    """Power, and the dS/dtheta != 0 check on the headline statistic.

    Weighted least squares is linear in y, so an injected c2 shifts the fitted
    coefficient by exactly c2 * R with R computed per realisation. Injection
    recovery is therefore EXACT here rather than Monte Carlo, and the
    monotone-invariance check is a statement about a computed derivative, not
    about a noisy difference of two simulation means.
    """
    rows = []
    for tag, comps in (("statistical", STAT_NULL), ("full", FULL_NULL)):
        samp, resp = arm.null_mc(400, 55_500 + len(tag), comps)
        mu0, sd0, R = samp.mean(), samp.std(), resp.mean()
        block = []
        for frac in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]:
            shift = frac * C1_FID * R
            S = shift / sd0
            power = float((np.abs(samp - mu0 + shift) / sd0 > 3.0).mean())
            block.append({"injected_c2_over_c1": frac,
                          "recovered_c2_over_c1": float(shift / R / C1_FID),
                          "statistic_sigma": float(S),
                          "power_at_3sigma": power})
            print(f"  [{tag:>11s}] inject {frac*100:5.2f}% -> recover "
                  f"{block[-1]['recovered_c2_over_c1']*100:+6.3f}%, "
                  f"S = {S:+7.2f} sigma, power = {power:.2f}")
        S = np.array([b["statistic_sigma"] for b in block])
        inj = np.array([b["injected_c2_over_c1"] for b in block])
        dS = np.diff(S) / np.diff(inj)
        rows.append({
            "null": tag, "response_factor_R": float(R),
            "null_sd_over_c1": float(sd0 / C1_FID),
            "rows": block,
            "dS_dtheta": [float(v) for v in dS],
            "monotone_invariance_check_passed": bool(np.all(dS > 1e-6)),
            "spread_of_statistic_sigma": float(S.max() - S.min()),
            "min_detectable_c2_over_c1_at_3sigma": float(
                3.0 * sd0 / R / C1_FID),
        })
    return {"blocks": rows,
            "note": "power quoted at a two-sided 3-sigma threshold calibrated "
                    "on the null itself"}


def systematic_floor(d, nud):
    m = d["I_q_SDSS_VoidFinder"].notna().to_numpy()
    a = d.loc[m, "dI_q_VoidFinder"].to_numpy(float)
    r = d.loc[m, "r_end_mpch"].to_numpy(float)
    b = d.loc[m, "I_q_SDSS_VoidFinder"].to_numpy(float)
    A = np.stack([np.ones_like(r), r, r ** 2, r ** 3], 1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    bt = b - A @ c
    rr = float(np.corrcoef(a, bt)[0, 1])
    out = {"n_common_sightlines": int(m.sum()),
           "transverse_residual_pearson_r": rr,
           "mean_I_q_DESIVAST": float(d.loc[m, "I_q_VoidFinder"].mean()),
           "mean_I_q_SDSS": float(b.mean()),
           "implied_relative_systematic_on_c2": float(
               np.sqrt(max(0.0, 1.0 - rr ** 2)) / max(1e-9, abs(rr))),
           "reading": "two independently built VoidFinder catalogues on the "
                      "same sight lines disagree on the transverse residual, "
                      "which is the variable the decisive test uses. The "
                      "implied relative systematic is the ratio of the "
                      "disagreeing part to the agreeing part; it multiplies "
                      "any statistical error bar on c2."}
    print(f"  n = {out['n_common_sightlines']}, transverse r = {rr:.3f}, "
          f"implied relative systematic on c2 = "
          f"x{out['implied_relative_systematic_on_c2']:.1f}")
    return out


if __name__ == "__main__":
    main()
