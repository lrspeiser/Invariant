"""estimators.py -- an INDEPENDENT quadrupole estimator for the shear field.

Run BF's cluster anisotropy detector (``universes/analysis.py::cluster_reduce``)
works like this:

    * subtract a five-bin, nearest-neighbour interpolation of the monopole
      Delta Sigma from the TANGENTIAL ellipticity only;
    * average ``2 * e_t * cos(2 phi)`` and ``2 * e_t * sin(2 phi)`` over every
      source between 0.2 and 2.2 R500, all radii pooled;
    * form q_amp = |c2 + i s2|, a phase pa_q = atan2(s2, c2)/2, and then
      PROJECTIONS q_ext = q_amp cos(2(pa_q - axis_ext)),
      q_bar = q_amp cos(2(pa_q - pa_bar));
    * test |S| against a critical value calibrated on scalar universes.

Nothing in this file reuses any of that.  This estimator

    * uses BOTH the tangential and the CROSS component.  A quadrupole in the
      lensing potential puts an m=2 pattern into e_x as well as e_t, and the
      cross channel is immune to monopole mis-modelling, which is the only
      place a radial error can leak;
    * fits the monopole SIMULTANEOUSLY with the quadrupole as a free constant
      per radial bin, so there is no subtraction step and no interpolation;
    * carries m=4 terms as nuisance, absorbing the reduced-shear leakage of an
      m=2 convergence;
    * returns a complex quadrupole PER RADIAL BIN with its 2x2 covariance,
      from which a noise-DEBIASED squared amplitude and a phase error follow.
      BF's |c2 + i s2| is positive-definite and therefore biased upward by
      exactly the noise variance; the debiased version can go negative, which
      is what a null-centred statistic requires;
    * is weighted by the per-source Sigma_crit, so the recovered amplitude is
      source-redshift independent.

Complex-quadrupole convention
-----------------------------
For a lensing potential perturbation  Psi_2 = psi(R) cos(2(phi - phi0))  the
tangential and cross ellipticities carry

    e_t = T(R) cos(2(phi - phi0)),      e_x = X(R) sin(2(phi - phi0)),

so with the per-bin design  e_t ~ a + c cos2phi + s sin2phi  and
e_x ~ d + p cos2phi + q sin2phi,

    Z_t = c + i s = T e^{2 i phi0},     Z_x = q - i p = X e^{2 i phi0}.

Both share the phase 2 phi0.  The combined inverse-variance estimate Z has
    arg Z = 2 phi0,  |Z| = the quadrupole amplitude in reduced-shear units.
"""
from __future__ import annotations

import numpy as np

# the reference Sigma_crit scaling: every cluster's ellipticities are put on a
# common source plane so amplitudes are comparable between clusters
BINS_R500 = ((0.20, 0.55), (0.55, 1.10), (1.10, 2.20))
MIN_SRC = 60


def tangential_cross(e1, e2, phi):
    """(e_t, e_x) about the declared centre."""
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    et = -(e1 * c2 + e2 * s2)
    ex = (e1 * s2 - e2 * c2)
    return et, ex


def _wls_cov(X, y, w):
    """Weighted least squares; returns (beta, cov) with cov from the residuals."""
    A = X * w[:, None]
    XtX = X.T @ A
    try:
        Xi = np.linalg.inv(XtX + 1e-12 * np.eye(X.shape[1]))
    except np.linalg.LinAlgError:
        return None, None
    beta = Xi @ (A.T @ y)
    r = y - X @ beta
    # sandwich covariance: robust to the fact that the shape-noise variance is
    # not exactly the declared one and varies with position
    meat = (X * (w * r)[:, None]).T @ (X * (w * r)[:, None])
    cov = Xi @ meat @ Xi
    return beta, cov


def cluster_quadrupole(cd, sigma_crit_fn, bins=BINS_R500, shape_sd=0.26):
    """Complex quadrupole per radial bin for one cluster's raw WL catalogue.

    Parameters
    ----------
    cd : the dict emitted by ``universes.corpus.emit_cluster``
    sigma_crit_fn : callable(z_lens, z_source) -> Msun/kpc^2

    Returns a dict with, per bin:
        Z      complex quadrupole (reduced-shear units, common source plane)
        C      2x2 covariance of (Re Z, Im Z)
        Q2     noise-debiased |Z|^2  = |Z|^2 - tr(C)      (can be negative)
        mono   the fitted monopole e_t amplitude in that bin
        n      number of sources used
    """
    R5 = float(cd["R500"])
    x, y = cd["src_x"], cd["src_y"]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    et, ex = tangential_cross(cd["e1"], cd["e2"], phi)

    zs = np.maximum(cd["z_src_phot"], cd["z"] + 0.05)
    Scr = sigma_crit_fn(cd["z"], zs)
    ok = np.isfinite(Scr) & (cd["z_src_phot"] > cd["z"] + 0.10) & (Scr > 0)
    if ok.sum() < 4 * MIN_SRC:
        return None
    Sref = float(np.median(Scr[ok]))
    # rescale onto a common source plane; weights follow
    scale = Scr / Sref
    w_all = 1.0 / (shape_sd ** 2 * np.maximum(scale, 1e-6) ** 2)

    out = {"R500": R5, "bins": [], "Sref": Sref}
    for (a, b) in bins:
        s = ok & (r >= a * R5) & (r < b * R5)
        n = int(s.sum())
        rec = {"lo": a, "hi": b, "n": n}
        if n < MIN_SRC:
            rec.update(Z=0j, C=np.zeros((2, 2)), Q2=0.0, mono=0.0, ok=False)
            out["bins"].append(rec)
            continue
        ph = phi[s]
        ones = np.ones(n)
        # design: monopole + m=2 + m=4 nuisance
        D = np.stack([ones, np.cos(2 * ph), np.sin(2 * ph),
                      np.cos(4 * ph), np.sin(4 * ph)], 1)
        yt = et[s] * scale[s]
        yx = ex[s] * scale[s]
        wv = w_all[s] * scale[s] ** 2          # weight on the rescaled variable
        bt, ct = _wls_cov(D, yt, wv)
        bx, cx = _wls_cov(D, yx, wv)
        if bt is None or bx is None:
            rec.update(Z=0j, C=np.zeros((2, 2)), Q2=0.0, mono=0.0, ok=False)
            out["bins"].append(rec)
            continue
        # Z_t = c + i s ; Z_x = q - i p
        Zt = np.array([bt[1], bt[2]])
        Ct = ct[np.ix_([1, 2], [1, 2])]
        Zx = np.array([bx[2], -bx[1]])
        M = np.array([[0.0, 1.0], [-1.0, 0.0]])
        Cx = M @ cx[np.ix_([1, 2], [1, 2])] @ M.T
        # inverse-variance combination of two independent estimates of the same Z
        try:
            Pt, Px = np.linalg.inv(Ct), np.linalg.inv(Cx)
            C = np.linalg.inv(Pt + Px)
            Zv = C @ (Pt @ Zt + Px @ Zx)
        except np.linalg.LinAlgError:
            C = Ct
            Zv = Zt
        Z = complex(Zv[0], Zv[1])
        rec.update(Z=Z, C=C, Q2=float(abs(Z) ** 2 - np.trace(C)),
                   mono=float(bt[0]), ok=True)
        out["bins"].append(rec)
    return out


def combine_bins(q):
    """Inverse-variance combination of the per-bin Z into one cluster Z."""
    P = np.zeros((2, 2))
    v = np.zeros(2)
    n = 0
    for rec in q["bins"]:
        if not rec["ok"]:
            continue
        try:
            Pi = np.linalg.inv(rec["C"])
        except np.linalg.LinAlgError:
            continue
        P += Pi
        v += Pi @ np.array([rec["Z"].real, rec["Z"].imag])
        n += rec["n"]
    if n == 0 or np.linalg.cond(P) > 1e12:
        return 0j, np.eye(2) * 1e6, 0
    C = np.linalg.inv(P)
    Zv = C @ v
    return complex(Zv[0], Zv[1]), C, n


def project(Z, C, angle_deg):
    """Signed projection of Z on an axis, and its variance.

    Re[ Z e^{-2 i a} ] with a = angle in radians;  Var from C.
    """
    a = np.deg2rad(float(angle_deg))
    u = np.array([np.cos(2 * a), np.sin(2 * a)])
    val = float(u @ np.array([Z.real, Z.imag]))
    var = float(u @ C @ u)
    return val, max(var, 1e-30)


# ======================================================================
# galaxy channel: the m=3 harmonic of the IFU velocity field
# ======================================================================
# An m=2 modulation of the circular speed, v_c(R,phi)^2 = V(R)^2 [1 + q cos 2(phi-psi)],
# appears in the line-of-sight field v = v_c cos(phi) sin(i) as
#
#     sin i V(R) [ cos(phi) + (q/4)( cos(3phi - 2psi) + cos(phi - 2psi) ) ],
#
# so the m=1 part is degenerate with the rotation curve itself and the clean
# directional signature is the m=3 harmonic, whose phase is 2 psi.
#
# Run BF extracted it by fitting a rotation curve in 8 radial bins using only
# spaxels with |cos theta| > 0.32, subtracting it, and taking the UNWEIGHTED
# mean of res*cos3theta over 1-5 R_d with all radii pooled.  This estimator
# instead fits every harmonic SIMULTANEOUSLY per ring by weighted least squares
# -- constant, m=1, m=2 (warp/lopsidedness nuisance) and m=3 -- so no curve is
# ever subtracted, no spaxels are discarded, and the m=3 amplitude comes out
# with a covariance.  It is normalised by the fitted m=1 amplitude in the same
# ring, which makes it the dimensionless q and removes the distance, the
# inclination and the mass scale.
GAL_RINGS = ((1.0, 2.0), (2.0, 3.2), (3.2, 5.0))


def galaxy_m3(gd, rings=GAL_RINGS, min_spax=24):
    """Complex m=3 modulation q e^{2 i psi} of one galaxy's velocity field.

    Returns (W, var) with W = q_hat e^{2 i psi_hat} (complex) and var the
    variance of each component, or None if the field is unusable.
    """
    d = gd["dist_obs"] * 1e3
    ax = gd["ax_arcsec"] * d * (np.pi / 180.0 / 3600.0)          # kpc
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    pa = np.deg2rad(gd["pa_obs"])
    xr = X * np.cos(pa) + Y * np.sin(pa)
    yr = -X * np.sin(pa) + Y * np.cos(pa)
    inc = np.deg2rad(gd["incl_obs"])
    yd = yr / max(np.cos(inc), 1e-3)
    R = np.hypot(xr, yd) + 1e-9
    th = np.arctan2(yd, xr)
    m0 = gd["mask"]
    Rd = gd["Rd_obs"]

    P = np.zeros((2, 2))
    v = np.zeros(2)
    used = 0
    for (a, b) in rings:
        s = m0 & (R >= a * Rd) & (R < b * Rd)
        n = int(s.sum())
        if n < min_spax:
            continue
        t = th[s]
        y = gd["v_map"][s]
        w = 1.0 / np.maximum(gd["v_err"][s], 1.0) ** 2
        D = np.stack([np.ones(n), np.cos(t), np.sin(t), np.cos(2 * t),
                      np.sin(2 * t), np.cos(3 * t), np.sin(3 * t)], 1)
        beta, cov = _wls_cov(D, y, w)
        if beta is None:
            continue
        A1 = float(np.hypot(beta[1], beta[2]))
        if A1 < 8.0:                     # no measurable rotation in this ring
            continue
        Z = 4.0 * np.array([beta[5], beta[6]]) / A1
        C = 16.0 * cov[np.ix_([5, 6], [5, 6])] / A1 ** 2
        try:
            Pi = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            continue
        P += Pi
        v += Pi @ Z
        used += n
    if used == 0 or np.linalg.cond(P) > 1e12:
        return None
    C = np.linalg.inv(P)
    Zv = C @ v
    return complex(Zv[0], Zv[1]), C
