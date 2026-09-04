"""Smoothness classes for the clipped void state q.

The programme's `delta` state is

    u(rho_s) = rho_ref/rho_s - 1 ,      q = clip(u, 0, 1-eps)

which is C^0 in position and has a DISCONTINUOUS gradient across the two
clipping surfaces  rho_s = rho_ref  (where q leaves 0) and
rho_s = rho_ref/(2-eps)  (where q reaches 1).  Four replacements are provided,
spanning the smoothness ladder, all of them reducing to the hard clip as the
rounding width w -> 0:

  hard      C^0   q' jumps by  du/dr  at each surface
  quad      C^1   quadratic corner, phi = (u+w)^2/(4w) on |u| <= w
  quintic   C^2   phi = 2w (t^3 - t^4/2), t = (u+w)/(2w)          <-- primary
  softplus  C^inf phi = (u + sqrt(u^2 + w^2))/2

THE POINT OF THE DISTINCTION, and it is the whole Job-2 trade:  `quad` and
`quintic` have COMPACT SUPPORT for the rounding -- they are identically zero
for u <= -w -- so q(Sun) = 0 EXACTLY whenever the local density exceeds
rho_ref/(1-w), which for rho_ref = 1e6 and rho_s(Sun) = 4e7 means any
w <= 0.975.  `softplus` is analytic and therefore never exactly zero: it
leaves q(Sun) = w^2/(4|u_Sun|) + O(w^4), which is what forces a bound on
rho_ref.  So C^2 and "exactly safe in the solar system" are COMPATIBLE; it is
ANALYTICITY that costs the exact zero, not smoothness.

Derivation of the quintic.  Write phi(u) = 2w g(t), t = (u+w)/(2w).  Then
phi' = g'(t) and phi'' = g''(t)/(2w), so C^2 matching to 0 at u = -w and to u
at u = +w needs g(0)=g'(0)=g''(0)=0, g(1)=1/2, g'(1)=1, g''(1)=0.  The unique
quintic with only cubic-and-above terms is g = t^3 - t^4/2 (the t^5
coefficient comes out exactly zero).  phi''' jumps by 6/(2w)^2 at u = -w, so
the function is C^2 and not C^3, which is exactly what was asked for.
"""
from __future__ import annotations

import math

import sys

import numpy as np

sys.path.insert(0, "C:/Users/henry/Documents/Codex/2026-08-21/"
                   "Invariant-main-integration/work/wellnet-2026-09/nonlocal")
import nonlocal_kernel as NK          # noqa: E402

KINDS = ("hard", "quad", "quintic", "softplus")
SMOOTHNESS = dict(hard=0, quad=1, quintic=2, softplus=99)


def _phi(u, w, kind):
    """Rounded max(u, 0)."""
    u = np.asarray(u, float)
    if kind == "hard" or w <= 0:
        return np.maximum(u, 0.0)
    if kind == "softplus":
        return 0.5 * (u + np.sqrt(u * u + w * w))
    t = np.clip((u + w) / (2.0 * w), 0.0, 1.0)
    if kind == "quad":
        g = 0.5 * t ** 2
    elif kind == "quintic":
        g = t ** 3 - 0.5 * t ** 4
    else:
        raise KeyError(kind)
    return np.where(u >= w, u, 2.0 * w * g)


def _dphi(u, w, kind):
    u = np.asarray(u, float)
    if kind == "hard" or w <= 0:
        return (u > 0).astype(float)
    if kind == "softplus":
        return 0.5 * (1.0 + u / np.sqrt(u * u + w * w))
    t = np.clip((u + w) / (2.0 * w), 0.0, 1.0)
    if kind == "quad":
        g1 = t
    else:
        g1 = 3.0 * t ** 2 - 2.0 * t ** 3
    return np.where(u >= w, 1.0, g1)


def _d2phi(u, w, kind):
    u = np.asarray(u, float)
    if kind == "hard" or w <= 0:
        return np.zeros_like(u)
    if kind == "softplus":
        s = np.sqrt(u * u + w * w)
        return 0.5 * w * w / s ** 3
    t = np.clip((u + w) / (2.0 * w), 0.0, 1.0)
    if kind == "quad":
        g2 = np.ones_like(t)
    else:
        g2 = 6.0 * t - 6.0 * t ** 2
    inside = (u > -w) & (u < w)
    return np.where(inside, g2 / (2.0 * w), 0.0)


def q_clip(u, w=0.0, kind="hard", eps=1e-12, deriv=0):
    """q(u) = rounded clip of u onto [0, 1-eps].  `deriv` = 0, 1 or 2 gives
    q, dq/du, d2q/du2."""
    top = 1.0 - eps
    f = (_phi, _dphi, _d2phi)[deriv]
    return f(u, w, kind) - f(u - top, w, kind)


def u_of_rho(rho_s, rho_ref):
    rho_s = np.maximum(np.asarray(rho_s, float), 1e-300)
    return rho_ref / rho_s - 1.0


def build_field_clipped(prof, rho_ref=1e5, w=0.0, kind="hard", L_s=0.0,
                        rho_floor=NK.RHO_BAR_B, label="", eps=1e-12):
    """SphericalField whose q is the (possibly rounded) clipped delta state."""
    r, rho, Mr, rfun, Mfun, Mtot = prof
    rho_f = rho + rho_floor
    rho_s = NK.smooth_spherical(r, rho_f, L_s) if L_s > 0 else rho_f
    q = q_clip(u_of_rho(rho_s, rho_ref), w=w, kind=kind, eps=eps)
    return NK.SphericalField(r=r, rho=rho, q=q, Menc=Mr.copy(),
                             rho_fun=rfun, Menc_fun=Mfun, label=label)


def clip_radii(prof, rho_ref, rho_floor=NK.RHO_BAR_B, L_s=0.0):
    """The two clipping radii: rho_s = rho_ref (q leaves 0) and
    rho_s = rho_ref/2 (q reaches 1)."""
    r, rho, Mr, rfun, Mfun, Mtot = prof
    rho_f = rho + rho_floor
    rho_s = NK.smooth_spherical(r, rho_f, L_s) if L_s > 0 else rho_f
    out = {}
    for tag, target in (("q_leaves_0", rho_ref), ("q_reaches_1",
                                                  0.5 * rho_ref)):
        d = np.log(rho_s) - math.log(target)
        s = np.where(np.diff(np.sign(d)) != 0)[0]
        out[tag] = (float(np.interp(0.0, [d[s[-1]], d[s[-1] + 1]],
                                    [r[s[-1]], r[s[-1] + 1]]))
                    if len(s) else float("nan"))
    return out


#: The Milky-Way vertical model the previous lane used for the solar checks.
#: Bland-Hawthorn & Gerhard 2016 local baryon budget:
#:   Sigma(R0) = 45 Msun/pc^2, h_R = 2.6 kpc, h_z = 0.30 kpc,
#:   midplane rho(R0) = 7.6e7 Msun/kpc^3.  Reproduced here rather than
#:   imported so the smoothness audit is self-contained.
MW = dict(Sigma0=45.0e6 / (2 * 0.30), hR=2.6, hz=0.30, R0=8.2)


def mw_rho_local(R=8.2, z=0.0):
    """Midplane-ish density of the MW disk model, Msun/kpc^3."""
    rho0 = MW["Sigma0"]
    return rho0 * math.exp(-(R - MW["R0"]) / MW["hR"]) / math.cosh(
        z / MW["hz"]) ** 2


def solar_eps(q_sun, gradq_sun, alpha, p, Fname="F1_poly", beta=0.0,
              D_AU=1.0):
    """Fractional inverse-square-law violation at D_AU astronomical units.

    The anomalous term is  G m M F'(qbar) grad_1 qbar / D, a fixed-direction
    force falling as 1/D rather than 1/D^2, so

        eps(D) = |F'(qbar)| |grad q| D / (2 F)

    (the 1/2 is because grad_1 qbar = grad q / 2 on a short segment).  This is
    the previous lane's linearisation, which it checked against the exact path
    average to 0.25 per cent at 1, 10 and 30 AU.
    """
    Ff, dF = NK.FAMILIES[Fname][0], NK.FAMILIES[Fname][1]
    F = float(Ff(q_sun, 0.0, alpha=alpha, beta=beta, p=p))
    Fp = float(dF(q_sun, 0.0, alpha=alpha, beta=beta, p=p))
    D = D_AU * NK.AU_KPC
    return abs(Fp) * abs(gradq_sun) * D / (2.0 * F)
