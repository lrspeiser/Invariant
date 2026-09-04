"""D(r) = F - r F' for the symmetric nonlocal kernel, by ANALYTIC
differentiation under the integral sign.

WHY THIS MODULE EXISTS
----------------------
For an exterior potential Phi = -(G M / r) F(r) the inward acceleration is

    g(r) = (G M / r^2) [ F(r) - r F'(r) ]  ==  (G M / r^2) D(r)

and a flat rotation curve needs D proportional to r with D > 0 throughout.
The previous lane measured dlnF/dlnr up to 0.899, i.e. D = 0.101 F, so the
observable is a ~10:1 cancellation inside F.  Every number in that lane was
obtained by finite-differencing Phi (a five-point log stencil, or a cubic
spline through Phi), which differences the SAME cancellation a second time.
This module removes the second differencing entirely.

THE ALGEBRA
-----------
The spherical reduction of the kernel is exact:

    Phi(r) = -(2 pi G / r) Int r' rho(r') I(r,r') dr'
    I(r,r') = Int_{a}^{b} F( qbar(r,r',D) ) dD ,   a = |r-r'| , b = r+r'

Define J(r) = Int r' rho I dr', so Phi = -2 pi G J / r and
F_eff = -r Phi/(G M) = 2 pi J / M.  Then EXACTLY

    D = F_eff - r F_eff' = (2 pi / M) Int r' rho(r') [ I - r dI/dr ] dr'

    dI/dr = F(qbar|_{D=b}) - sgn(r-r') F(qbar|_{D=a})
            + Int_a^b F_q(qbar) dqbar/dr dD

    dqbar/dr = Int_0^1 q'(r_s) r (1-s) / r_s  ds

using r_s(s)^2 = r^2 + s (r'^2 - r^2 - D^2) + (s D)^2, so d(r_s^2)/dr =
2 r (1-s).  On the two boundary segments the path radius is elementary:

    D = a :  r_s = r + s (r' - r)          (radially aligned)
    D = b :  r_s = | r - s (r + r') |      (antipodal, through the origin)

NEWTONIAN CHECK, term by term.  With F = 1, F_q = 0:
  r' > r : I - r dI/dr = 2r - r(1 + 1) = 0
  r' < r : I - r dI/dr = 2r' - r(1 - 1) = 2 r'
so D = (2 pi/M) Int_0^r 2 r'^2 rho dr' = M(<r)/M = 1 for an exterior point.
That identity is checked to round-off in `audit_stability.py`.

A GAUGE FACT WORTH STATING.  F is defined only up to an additive multiple of
r: replacing F -> F + c r shifts Phi by the constant -G M c and leaves D
unchanged (D = F - rF' annihilates c r exactly).  So F itself is not an
observable; D is.  Any statement about "F bounded" must be converted into a
statement about D before it constrains dynamics.

UNITS: kpc, Msun, km/s, as in nonlocal_kernel.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "C:/Users/henry/Documents/Codex/2026-08-21/"
                   "Invariant-main-integration/work/wellnet-2026-09/nonlocal")
import nonlocal_kernel as NK          # noqa: E402

G = NK.G


def _xp(use_gpu):
    if use_gpu:
        import cupy as cp
        return cp
    return np


def _asnumpy(a, xp):
    return xp.asnumpy(a) if xp is not np else np.asarray(a)


# ==========================================================================
#  1.  q PROFILE WITH A DECLARED INTERPOLATION RULE
# ==========================================================================
#  The existing kernel interpolates q log-linearly in ln r (np.interp on the
#  log grid).  That makes q C^0 and dq/dr piecewise CONSTANT -- a staircase.
#  Since D contains dqbar/dr, which is a path integral of dq/dr, the rule is
#  a first-class numerical knob and is exposed here rather than hard-wired.

_RULES = ("loglin", "lin", "pchip", "cubic", "akima")


class QProfile:
    """q(r) with a declared interpolation rule and a consistent derivative.

    `rule`:
      loglin  np.interp in ln r            -- what the existing kernel uses;
                                              q is C^0, dq/dr is a staircase
      lin     np.interp in r               -- same order, different metric
      pchip   monotone cubic in ln r       -- C^1, no overshoot
      cubic   natural cubic spline in ln r -- C^2, can overshoot
      akima   Akima in ln r                -- C^1, overshoot-resistant

    For the smooth rules the interpolant and its exact derivative are
    resampled onto a grid `over`-times finer than the source grid, and looked
    up log-linearly at run time; the resampling error is O(over^-2) relative
    to the source-grid error and is verified negligible in the audit.
    """

    def __init__(self, r, q, rule="loglin", over=32):
        r = np.asarray(r, float)
        q = np.asarray(q, float)
        assert r.ndim == 1 and r.shape == q.shape, "q profile shape"
        assert np.all(np.diff(r) > 0), "q profile must be increasing in r"
        assert rule in _RULES, f"unknown rule {rule}"
        self.rule = rule
        self.r_src, self.q_src = r, q
        lnr = np.log(r)
        if rule == "loglin":
            self.lnr = lnr
            self.q = q
            self.slope = np.diff(q) / np.diff(lnr)      # dq/dlnr, per cell
            self.dqdr = None
        elif rule == "lin":
            n = over * len(r)
            rf = np.linspace(r[0], r[-1], n)
            self.lnr = np.log(rf)
            self.q = np.interp(rf, r, q)
            sl = np.diff(q) / np.diff(r)
            idx = np.clip(np.searchsorted(r, rf, side="right") - 1,
                          0, len(r) - 2)
            self.dqdr = sl[idx]
            self.slope = None
        else:
            from scipy.interpolate import (PchipInterpolator, CubicSpline,
                                           Akima1DInterpolator)
            mk = dict(pchip=PchipInterpolator, cubic=CubicSpline,
                      akima=Akima1DInterpolator)[rule]
            f = mk(lnr, q)
            n = over * len(r)
            lf = np.linspace(lnr[0], lnr[-1], n)
            self.lnr = lf
            self.q = f(lf)
            self.dqdr = f(lf, 1) / np.exp(lf)
            self.slope = None
        self.lo, self.hi = self.lnr[0], self.lnr[-1]

    def to(self, xp):
        """Device-resident arrays for the batched evaluator."""
        d = dict(lnr=xp.asarray(self.lnr), q=xp.asarray(self.q))
        if self.slope is not None:
            d["slope"] = xp.asarray(self.slope)
        else:
            d["dqdr"] = xp.asarray(self.dqdr)
        return d

    @staticmethod
    def eval(d, rs, xp, need_deriv=True):
        """(q, dq/dr) at radii `rs` (any shape) from a `to()` dict."""
        lrs = xp.log(xp.maximum(rs, 1e-300))
        q = xp.interp(lrs, d["lnr"], d["q"])
        if not need_deriv:
            return q, None
        if "slope" in d:
            k = xp.clip(xp.searchsorted(d["lnr"], lrs, side="right") - 1,
                        0, d["slope"].shape[0] - 1)
            dq = d["slope"][k] / xp.maximum(rs, 1e-300)
        else:
            dq = xp.interp(lrs, d["lnr"], d["dqdr"])
        return q, dq


# ==========================================================================
#  2.  THE BATCHED ANALYTIC EVALUATOR
# ==========================================================================

def _global_panels(r_lo, r_hi, r_eval, dlnr_max=0.35, n_gl=8):
    """Composite Gauss-Legendre in ln r' with a panel edge at every field
    radius, so the |r-r'| kink always lands on a panel boundary."""
    edges = np.unique(np.concatenate(
        [[math.log(r_lo)], np.log(np.atleast_1d(r_eval)), [math.log(r_hi)]]))
    full = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:]):
        k = max(1, int(math.ceil((b - a) / dlnr_max)))
        full.extend(np.linspace(a, b, k + 1)[1:])
    full = np.asarray(full)
    xs, ws = NK.gauss_legendre(n_gl)
    a = full[:-1][:, None]
    b = full[1:][:, None]
    u = 0.5 * (b + a) + 0.5 * (b - a) * xs[None, :]
    w = (0.5 * (b - a) * ws[None, :]).ravel()
    rp = np.exp(u.ravel())
    return rp, w * rp


def phi_and_D(fld, r_eval, qprof=None, Fname="F1_poly", alpha=1.0, beta=0.0,
              p=1.0, n_D=32, n_s=12, n_gl=8, dlnr_max=0.35,
              r_lo=None, r_hi=None, use_gpu=False, chunk=6, Mtot=None,
              want_phi=True):
    """Return (F_eff, D) at `r_eval`, D = F_eff - r dF_eff/dr, ANALYTIC.

    `fld` is a nonlocal_kernel.SphericalField; `qprof` a QProfile built from
    (fld.r, fld.q) -- defaults to the log-linear rule, i.e. exactly what
    `spherical_potential_batch` does, so the two are directly comparable.
    """
    xp = _xp(use_gpu)
    Ffun, dFfun = NK.FAMILIES[Fname][0], NK.FAMILIES[Fname][1]
    if qprof is None:
        qprof = QProfile(fld.r, fld.q, "loglin")
    qd = qprof.to(xp)
    r_eval = np.atleast_1d(np.asarray(r_eval, float))
    lo = fld.r[0] if r_lo is None else r_lo
    hi = fld.r[-1] if r_hi is None else r_hi
    M = float(fld.Menc[-1] if Mtot is None else Mtot)

    rp_np, wr_np = _global_panels(lo, hi, r_eval, dlnr_max, n_gl)
    rho_np = fld.rho_at(rp_np)
    xsn, wsn = NK.gauss_legendre(n_D)
    ssn, wssn = NK.gauss_legendre(n_s)

    rp = xp.asarray(rp_np)
    w_src = xp.asarray(wr_np * rp_np * rho_np)          # r' rho dr'
    xs = xp.asarray(xsn); ws = xp.asarray(wsn)
    s_n = xp.asarray(0.5 * (ssn + 1.0)); s_w = xp.asarray(0.5 * wssn)

    outF = np.empty(len(r_eval), float)
    outD = np.empty(len(r_eval), float)
    for k0 in range(0, len(r_eval), chunk):
        rv = xp.asarray(r_eval[k0:k0 + chunk])[:, None]           # (c,1)
        a = xp.abs(rv - rp[None, :])
        b = rv + rp[None, :]
        half = 0.5 * (b - a); mid = 0.5 * (b + a)
        D = mid[:, :, None] + half[:, :, None] * xs[None, None, :]
        wD = half[:, :, None] * ws[None, None, :]
        rr = rv[:, :, None, None]
        rpb = rp[None, :, None, None]
        Db = D[:, :, :, None]
        sb = s_n[None, None, None, :]
        val = rr ** 2 + sb * (rpb ** 2 - rr ** 2 - Db ** 2) + (sb * Db) ** 2
        rs = xp.sqrt(xp.maximum(val, 0.0))
        qs, dqs = QProfile.eval(qd, rs, xp)
        qbar = xp.tensordot(qs, s_w, axes=([3], [0]))             # (c,nr',nD)
        #  dqbar/dr = Int q'(r_s) r (1-s)/r_s ds
        dqbar = xp.tensordot(dqs * (1.0 - sb) / xp.maximum(rs, 1e-300),
                             s_w, axes=([3], [0])) * rv[:, :, None]
        Fv = Ffun(qbar, 0.0, alpha=alpha, beta=beta, p=p)
        Fq = dFfun(qbar, 0.0, alpha=alpha, beta=beta, p=p)
        I = xp.sum(Fv * wD, axis=2)                               # (c,nr')
        #  For p < 1, dF/dqbar diverges as qbar -> 0.  A CLIPPED q makes
        #  qbar = 0 on an open set of (x,x') pairs -- both endpoints and the
        #  whole segment inside the dense region -- and there dqbar/dr is
        #  identically 0 too, so the product is 0 and not inf*0 = nan.  The
        #  guard encodes that; the genuinely divergent case (qbar = 0 with
        #  dqbar != 0) is a measure-zero set of quadrature nodes and is
        #  discussed in the smoothness audit rather than papered over.
        prod = xp.where(qbar > 0.0, Fq * dqbar, 0.0)
        bulk = xp.sum(prod * wD, axis=2)                          # (c,nr')

        #  boundary segments, D = a and D = b
        sa = s_n[None, None, :]
        rs_a = rv[:, :, None] + sa * (rp[None, :, None] - rv[:, :, None])
        rs_b = xp.abs(rv[:, :, None]
                      - sa * (rv[:, :, None] + rp[None, :, None]))
        qa, _ = QProfile.eval(qd, xp.abs(rs_a), xp, need_deriv=False)
        qb, _ = QProfile.eval(qd, rs_b, xp, need_deriv=False)
        qbar_a = xp.tensordot(qa, s_w, axes=([2], [0]))
        qbar_b = xp.tensordot(qb, s_w, axes=([2], [0]))
        Fa = Ffun(qbar_a, 0.0, alpha=alpha, beta=beta, p=p)
        Fb = Ffun(qbar_b, 0.0, alpha=alpha, beta=beta, p=p)
        sgn = xp.sign(rv - rp[None, :])
        dIdr = Fb - sgn * Fa + bulk
        integ = I - rv * dIdr
        Dv = (2.0 * math.pi / M) * xp.sum(w_src[None, :] * integ, axis=1)
        outD[k0:k0 + chunk] = _asnumpy(Dv, xp)
        if want_phi:
            Fe = (2.0 * math.pi / M) * xp.sum(w_src[None, :] * I, axis=1)
            outF[k0:k0 + chunk] = _asnumpy(Fe, xp)
    return (outF if want_phi else None), outD


def D_stencil(fld, r_eval, dlog=2e-3, Mtot=None, **kw):
    """D(r) the OLD way: five-point log stencil on F_eff.  Kept so the two
    routes can be differenced -- that difference is the audit."""
    r_eval = np.atleast_1d(np.asarray(r_eval, float))
    offs = np.array([-2, -1, 0, 1, 2]) * dlog
    coef = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * dlog)
    rr = (r_eval[:, None] * np.exp(offs[None, :])).ravel()
    M = float(fld.Menc[-1] if Mtot is None else Mtot)
    phi = NK.spherical_potential_batch(fld, rr, **kw)
    Fe = (-rr * phi / (G * M)).reshape(len(r_eval), 5)
    dF_dlnr = Fe @ coef
    return Fe[:, 2], Fe[:, 2] - dF_dlnr        # F - r F' = F - dF/dlnr
