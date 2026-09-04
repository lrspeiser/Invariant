"""THE STATISTIC AND EVERY ANALYSIS CHOICE, DECLARED IN CODE BEFORE ANY VALUE.

Hypothesis (the geometric half of the path-redshift class, the only half that
survives Run AK's supernova time-dilation test):

    an achromatic path redshift  d ln(1+z) = c2 * I_q  applied to CMB photons
    gives            dT/T = -c2 * dI_q
    hence            beta == dT/d(dI_q)   [uK per Mpc/h]
    and              c2/c1 = -beta*1e-6 / (T_CMB * c1_fiducial)
                           = beta * (-1.09989e-3)          with c1 = H0/c.

ESTIMATOR (declared, not chosen after the fact)
-----------------------------------------------
Ordinary least squares of the degraded Planck temperature on a design matrix,
over equal-area HEALPix pixels; `beta` is the coefficient on dI_q.  Three nested
models, all reported, with M2 named the HEADLINE before any number was seen:

    M1  [1, dipole(3), dI_q]                       ISW-unmarginalised
    M2  [1, dipole(3), dI_q, I_phi_in]             HEADLINE -- ISW separated
    M3  M2 + [edge_deg, csc_b, dust, sync]         systematics-hardened

The monopole and the three dipole components are ALWAYS projected out: the CMB
dipole is a kinematic signal of order 3 mK and the footprint is a patch, so an
unremoved dipole leaks into any smooth template.

NULL (the reference distribution)
---------------------------------
Random rotations and reflections of the footprint on the sphere.  The template
values travel with the pixels; the temperatures do not.  A rotation is admitted
only if
    (a) >= 98% of its pixels have Planck common-mask fraction >= 0.9,
    (b) min |b| over its pixels >= 20 deg   (the true footprint's own min is 24),
    (c) overlap with the true footprint <= 5%  -- a rotation that overlaps the
        footprint is not a null, and under the blind guard it is also the only
        way the certificate could have seen the measurement.

Errors are taken from this null, never from the OLS covariance: Run AK's own
lane would have announced 6.1 sigma from analytic errors where its simulated
null gave 1.8.  The analytic number is computed anyway and reported alongside,
because the ratio of the two is the diagnostic.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

C_KMS = 299792.458
C1_FIDUCIAL = 100.0 / C_KMS
T_CMB_K = 2.7255
UK_PER_MPCH_TO_C2C1 = -1e-6 / (T_CMB_K * C1_FIDUCIAL)      # -1.09989e-3

MODELS = {
    "M1_no_isw": ["dI_q"],
    "M2_isw_marginalised": ["dI_q", "I_phi_in"],
    "M3_hardened": ["dI_q", "I_phi_in", "edge_deg", "csc_b", "dust", "sync"],
}
HEADLINE = "M2_isw_marginalised"


def galactic_vectors(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    cb = np.cos(b)
    return np.stack([cb * np.cos(l), cb * np.sin(l), np.sin(b)], axis=-1)


def vectors_to_lb(G):
    l = np.degrees(np.arctan2(G[:, 1], G[:, 0])) % 360.0
    b = np.degrees(np.arcsin(np.clip(G[:, 2], -1, 1)))
    return l, b


def design(templates, cols, G):
    """[1, dipole(3), <cols>] with every column centred (the constant excepted)."""
    n = len(G)
    X = [np.ones(n), G[:, 0], G[:, 1], G[:, 2]]
    for c in cols:
        v = np.asarray(templates[c], float)
        X.append(v - v.mean())
    return np.stack(X, axis=1)


def fit_beta(X, T):
    """OLS; returns (beta on the first non-dipole column, all coefficients, analytic sd)."""
    coef, *_ = np.linalg.lstsq(X, T, rcond=None)
    resid = T - X @ coef
    dof = max(len(T) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    XtXi = np.linalg.pinv(X.T @ X)
    sd = np.sqrt(np.maximum(np.diag(XtXi) * s2, 0.0))
    return float(coef[4]), coef, float(sd[4])


def weights_for_beta(X):
    """The fixed linear functional w with beta = w . T, for analytic wT C w."""
    XtXi = np.linalg.pinv(X.T @ X)
    return (XtXi @ X.T)[4]


def random_rotation(rng, reflect=False):
    """One uniform SO(3) element, optionally composed with a reflection (det -1)."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    if reflect:
        Q = Q @ np.diag([1.0, 1.0, -1.0])             # genuine reflection, det = -1
    return Q


def random_rotations(rng, n, with_reflections=True):
    """n placements; exactly half are reflections when with_reflections is set."""
    return [random_rotation(rng, with_reflections and (i % 2 == 1)) for i in range(n)]


def admissible_rotation(pix, b_rot, mask_frac, true_pix_set,
                        min_unmasked_frac=0.98, min_abs_b=20.0, max_overlap=0.05):
    if np.mean(mask_frac[pix] >= 0.9) < min_unmasked_frac:
        return False, "masked"
    if np.min(np.abs(b_rot)) < min_abs_b:
        return False, "low_b"
    ov = len(true_pix_set.intersection(pix.tolist())) / len(pix)
    if ov > max_overlap:
        return False, "overlaps_footprint"
    return True, "ok"


def cl_theory(path, lmax=1500):
    """Planck PR3 best-fit theory D_l^TT [uK^2] -> C_l [uK^2], l = 0..lmax."""
    raw = np.loadtxt(path, comments="#")
    ell = raw[:, 0].astype(int)
    dl = raw[:, 1]
    cl = np.zeros(lmax + 1)
    m = ell <= lmax
    cl[ell[m]] = 2.0 * np.pi * dl[m] / (ell[m] * (ell[m] + 1.0))
    return cl


def beam_pixwin(lmax, fwhm_deg):
    l = np.arange(lmax + 1)
    sig = np.radians(fwhm_deg) / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-0.5 * l * (l + 1) * sig ** 2)


def corr_table(cl, bl, ntheta=200001):
    """C(theta) = sum_l (2l+1)/(4pi) C_l B_l^2 P_l(cos theta), on a theta grid."""
    x = np.cos(np.linspace(0.0, np.pi, ntheta))
    w = (2 * np.arange(len(cl)) + 1) / (4 * np.pi) * cl * bl ** 2
    p0 = np.ones_like(x)
    p1 = x.copy()
    out = w[0] * p0 + w[1] * p1
    for l in range(2, len(cl)):
        p2 = ((2 * l - 1) * x * p1 - (l - 1) * p0) / l
        out += w[l] * p2
        p0, p1 = p1, p2
    return x, out


def pixel_cov(G, cl, bl, chunk=1500):
    """Pixel-pixel covariance of a Gaussian isotropic field on directions G."""
    xg, cg = corr_table(cl, bl)
    order = np.argsort(xg)
    xg, cg = xg[order], cg[order]
    n = len(G)
    C = np.empty((n, n), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.clip(G[s:e] @ G.T, -1.0, 1.0)
        C[s:e] = np.interp(d, xg, cg)
    return C
