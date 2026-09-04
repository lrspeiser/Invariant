"""The null, with every shared path actually simulated.

WHAT THE NULL HYPOTHESIS IS, and why it is not "S = 1"
------------------------------------------------------
The parameters under test are alpha, beta and gamma -- whether S depends on
mass, on clustercentric radius or on acceleration.  The survey amplitudes are
nuisances.  So the correct null is

    H0' :  ln S = o_k   (one constant per survey, at its H_P value),
           NO dependence on M, r or g

and NOT "S = 1".  Setting S = 1 in the strong-lensing cores would mean the
observed arcs do not exist at all -- the frozen law is subcritical there -- so
a null built that way scatters image radii into a regime the data never
occupy and returns nonsense.  This was found by running it; the numbers it
produced (ln S sd 1.69 against 0.27 in the data) were impossible.

The truth is therefore the DATA'S OWN AMPLITUDE with no structure, and what is
redrawn is the measurement noise on the inputs -- with the whole chain,
INCLUDING the regressors, rebuilt from the redrawn inputs.

WHAT SHARES WHAT (the construction expressions, written out before believing
any correlation):

  eFEDS   S_hat comes from g_+ (DECADE shapes) against a model built from the
          Bahar+2022 Vikhlinin parameters.  ln M and the radius axis are built
          from the SAME density parameters.  Redraw: Vikhlinin parameters from
          their published marginal errors (three variance scalings, because
          Bahar publishes NO covariance and the parameters are strongly
          covariant), plus fresh shape noise.

  LoCuSS  S_hat = M_WL / M_dyn(r500_WL) and r500_WL = (3 M_WL/4 pi 500
          rho_c)^(1/3), so M_WL is in the numerator AND sets the radius at
          which the denominator is evaluated; ln M = ln M_gas(<r500_WL) moves
          with it too.  Redraw: M_WL about its H0' value, and the ACCEPT n_e
          shells.

  SL      S_hat = 1/kappa_bar(theta) and ln(r/R500) = ln(theta D_l/R500), so a
          COHERENT error in the assumed centre moves BOTH, with
          d ln x/d ln theta = 1 exactly and d ln S/d ln theta ~ +1.  That
          manufactures a POSITIVE beta out of nothing.  Redraw: a per-cluster
          centring shift in ln theta (sd 0.10), and the ACCEPT n_e shells.
          The observed image radii themselves are KEPT -- their spread inside
          a cluster is many sources at many radii, not measurement noise.
"""
from __future__ import annotations

import math

import numpy as np

import build as B
import common as K
import pipeline as P

MPC, MSUN, G = P.MPC, P.MSUN, P.G


def perturb_accept(name, rng, scale=1.0):
    """M_gas(<r) on the common grid with the ACCEPT n_e shells redrawn."""
    rm, ne, nee, tx, nrow = B.read_accept(name)
    ne2 = np.maximum(ne + scale * nee * rng.normal(size=ne.size), 1e-12)
    lr, ln = np.log(rm), np.log(ne2)
    k = min(5, len(rm))
    so = float(np.clip(np.polyfit(lr[-k:], ln[-k:], 1)[0], -4.5, -1.2))
    si = float(np.clip(np.polyfit(lr[:min(3, len(rm))],
                                  ln[:min(3, len(rm))], 1)[0], -2.0, 0.0))
    lg = np.log(B.R_GRID)
    out = np.interp(lg, lr, ln)
    out = np.where(lg > lr[-1], ln[-1] + so * (lg - lr[-1]), out)
    out = np.where(lg < lr[0], ln[0] + si * (lg - lr[0]), out)
    rho = B.L.MU_E * B.L.M_P * np.exp(out) * 1e6
    return B.cumtrap(4.0 * math.pi * B.R_GRID ** 2 * rho, B.R_GRID)


def theta_at(cl, z_s, target, th_ref, lo=0.3, hi=400.0, n=90):
    """Radius where kappa_bar = target, and d ln kappa_bar/d ln theta there.

    One vectorised kappa_bar call on a log grid then log-log interpolation:
    kappa_bar is smooth and monotone here, so bisecting would cost 40 extra
    projections per system and buy nothing.
    """
    th = np.geomspace(lo, hi, n)
    kb = np.asarray(K.kappa_bar(cl, th, z_s))
    lk, lt = np.log(np.maximum(kb, 1e-30)), np.log(th)
    d = lk - math.log(target)
    s = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
    if s.size == 0:
        return th_ref, -1.0
    i = int(s[-1])
    t = math.exp(lt[i] + (lt[i + 1] - lt[i]) * d[i] / (d[i] - d[i + 1]))
    slope = (lk[i + 1] - lk[i]) / (lt[i + 1] - lt[i])
    return t, slope


def locuss_null(bd, rng, fs, o_lo, scale=1.0):
    """One H0' realisation of the LoCuSS points, with M_WL shared as it is."""
    S0 = math.exp(o_lo)
    lnS, lnM, lnx, lng, e = [], [], [], [], []
    for c, p in zip(bd.lo, bd.lop):
        Mg = perturb_accept(c.extra["accept_name"], rng, scale)
        c2 = B.Cluster(c.id, "locuss", c.z, B.R_GRID, Mg, c.M_star,
                       kT_keV=c.kT)
        # H0': the TRUE lensing mass is S0 x M_dyn in its own 500-overdensity
        # aperture:  S0 M_dyn(R) = 500 rho_c (4pi/3) R^3.
        g = B.g_rar(c2.g_b)
        Md = g * c2.r ** 2 / G
        tgt = 500.0 * B.rho_c(c.z) * (4.0 * math.pi / 3.0) * c2.r ** 3
        f = S0 * Md - tgt
        s = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
        if s.size == 0:
            R = p["r"]
        else:
            i = int(s[0])
            u = f[i] / (f[i] - f[i + 1])
            R = c2.r[i] + u * (c2.r[i + 1] - c2.r[i])
        M_true = 500.0 * B.rho_c(c.z) * (4.0 * math.pi / 3.0) * R ** 3
        fe = p["frac_err"]
        M_obs = M_true * math.exp(rng.normal() * math.hypot(fe, fs[0]))
        r_obs = (3.0 * M_obs / (4.0 * math.pi * 500.0
                                * B.rho_c(c.z))) ** (1.0 / 3.0)
        lnS.append(math.log(M_obs / c2.M_dyn(r_obs)))
        lnM.append(math.log(K.M_at(c2, r_obs) / K.M0))
        lnx.append(0.0)
        lng.append(math.log(max(float(c2.g_b_at(r_obs)), 1e-30) / K.A0))
        e.append(fe)
    return np.array(lnS), np.column_stack([lnM, lnx, lng]), np.array(e)


def sl_null(bd, rng, fs, o_sl, scale=1.0, sd_centre=0.10):
    """One H0' realisation of the strong-lens points.

    The OBSERVED image radii and source redshifts are kept: the spread of
    theta across image systems inside one cluster is real astrophysics (many
    sources at many radii), not measurement noise, so a null that regenerates
    it would not be a null of the same design.  What is redrawn is
      * the ACCEPT n_e shells, which enter kappa_bar (hence S) and ln M;
      * a per-cluster CENTRING error in ln theta, sd = 0.10, which moves ln S
        and ln(r/R500) COHERENTLY -- that is the shared path that exists here,
        and it manufactures a positive beta out of nothing;
      * the declared within-cluster and cluster-common scatter in ln S.
    """
    lnS, lnM, lnx, lng, e = [], [], [], [], []
    by = {c.id: c for c, _ in bd.sl}
    common = {cid: rng.normal() * fs[2] for cid in by}
    shift = {cid: rng.normal() * sd_centre for cid in by}
    cache = {}
    for row in bd.sl_rows:
        cid = row["cid"]
        if cid not in cache:
            c = by[cid]
            Mg = perturb_accept(c.extra["accept"], rng, scale)
            cache[cid] = B.Cluster(cid, "sl", c.z, B.R_GRID, Mg, c.M_star,
                                   kT_keV=c.kT, extra=dict(c.extra))
        c2 = cache[cid]
        th = row["theta_as"] * math.exp(shift[cid])
        kb = float(K.kappa_bar(c2, th, row["z_s"]))
        # H0': ln S is the survey constant plus scatter, with NO dependence on
        # M, r or g.  The kappa_bar just computed fixes how a centring shift
        # and a gas redraw propagate into the ESTIMATE.
        kb0 = float(K.kappa_bar(c2, row["theta_as"], row["z_s"]))
        drift = -math.log(max(kb, 1e-12)) + math.log(max(kb0, 1e-12))
        r500 = c2.extra["R500_cat"]
        rr = th * P.ARCSEC * float(P.d_ang(c2.z))
        lnS.append(o_sl + drift + common[cid]
                   + rng.normal() * math.hypot(fs[1], row["e_stat"]))
        lnM.append(math.log(K.M_at(c2, r500) / K.M0))
        lnx.append(math.log(rr / r500))
        lng.append(math.log(max(float(c2.g_b_at(rr)), 1e-30) / K.A0))
        e.append(row["e_stat"])
    return np.array(lnS), np.column_stack([lnM, lnx, lng]), np.array(e)
