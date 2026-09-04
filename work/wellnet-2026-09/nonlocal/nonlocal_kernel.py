"""Symmetric nonlocal gravity kernel: the space BETWEEN two masses modulates
their coupling.

    Phi(x) = -G Int [ rho_b(x') / |x - x'| ] F[ qbar(x,x'), Tbar(x,x') ] d^3x'

    qbar(x,x') = Int_0^1 q[ (1-s) x + s x' ] ds                (path average)
    Tbar(x,x') = Int_0^L q T_ij k^i k^j dl / a0                 (path line int)

Both path functionals are manifestly invariant under x <-> x' (reversing the
parameterisation s -> 1-s maps the segment onto itself), so any F built from
them is reciprocal, W(x,x') = W(x',x).  That is verified numerically in
`screen_nonlocal.py` rather than asserted.

RECIPROCITY IS NOT MOMENTUM CONSERVATION.  This module makes the distinction
explicit because it is the sharpest structural result in the lane.  With the
test-particle force f = -m grad Phi and the q field held fixed, a two-body
system feels

    f_1 + f_2 = G m1 m2 F'(qbar) < grad q >_path / D

where < grad q >_path = Int_0^1 grad q(x(s)) ds.  Its component along the
separation is exactly [q(x2) - q(x1)] / D, by the fundamental theorem of
calculus applied along the segment.  So a SYMMETRIC kernel still produces a
net self-force on an isolated pair whenever the two bodies sit at different
values of q -- i.e. for every unequal-mass pair, because the heavier body
digs a deeper density well and therefore sits at lower q.  Symmetry of the
kernel buys reciprocity of the pair term; only TRANSLATION INVARIANCE buys
momentum conservation, and F(qbar) is not translation invariant unless q is
carried along with the bodies.  `pair_forces` returns the residual so it can
be measured.

UNITS throughout: kpc, Msun, km/s.  G = 4.300917270e-6 kpc (km/s)^2 / Msun.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------- constants
G = 4.300917270e-6          # kpc (km/s)^2 / Msun
KPC_M = 3.0856775814913673e19
AU_KPC = 1.495978707e11 / KPC_M          # 4.8481e-9 kpc
A0 = 1.2e-10 * KPC_M / 1.0e6             # 1.2e-10 m/s^2 in (km/s)^2/kpc
H0_h = 0.674
RHO_CRIT = 277.53663 * H0_h ** 2         # Msun/kpc^3
OMEGA_B = 0.0493
OMEGA_M = 0.3153
RHO_BAR_B = OMEGA_B * RHO_CRIT           # 6.21 Msun/kpc^3  (baryons only)
RHO_BAR_M = OMEGA_M * RHO_CRIT           # 39.7 Msun/kpc^3  (LCDM total)


# ==========================================================================
#  1.  KERNEL FAMILIES
# ==========================================================================
#  Every family is F(qbar, Tbar; alpha, beta, p).  `dF` is dF/dqbar, needed
#  for the analytic force and for the momentum-residual identity.

def F_poly(qb, Tb=0.0, alpha=1.0, beta=0.0, p=1.0):
    """F = 1 + alpha qbar^p"""
    return 1.0 + alpha * np.power(qb, p)


def dF_poly(qb, Tb=0.0, alpha=1.0, beta=0.0, p=1.0):
    return alpha * p * np.power(qb, p - 1.0)


def F_exp(qb, Tb=0.0, alpha=1.0, beta=0.0, p=1.0):
    """F = exp(alpha qbar^p)"""
    return np.exp(alpha * np.power(qb, p))


def dF_exp(qb, Tb=0.0, alpha=1.0, beta=0.0, p=1.0):
    return alpha * p * np.power(qb, p - 1.0) * np.exp(alpha * np.power(qb, p))


def F_pade(qb, Tb=0.0, alpha=1.0, beta=1.0, p=1.0):
    """F = 1 + alpha qbar^p / (1 + beta qbar^p)"""
    u = np.power(qb, p)
    return 1.0 + alpha * u / (1.0 + beta * u)


def dF_pade(qb, Tb=0.0, alpha=1.0, beta=1.0, p=1.0):
    u = np.power(qb, p)
    du = p * np.power(qb, p - 1.0)
    return alpha * du / (1.0 + beta * u) ** 2


def F_tidal(qb, Tb=0.0, alpha=1.0, beta=1.0, p=1.0):
    """F = 1 + alpha qbar^p + beta Int q T_ij k^i k^j dl / a0"""
    return 1.0 + alpha * np.power(qb, p) + beta * Tb


def dF_tidal(qb, Tb=0.0, alpha=1.0, beta=1.0, p=1.0):
    return alpha * p * np.power(qb, p - 1.0)


FAMILIES = {
    "F1_poly": (F_poly, dF_poly, "1 + a q^p"),
    "F2_exp": (F_exp, dF_exp, "exp(a q^p)"),
    "F3_pade": (F_pade, dF_pade, "1 + a q^p/(1 + b q^p)"),
    "F4_tidal": (F_tidal, dF_tidal, "1 + a q^p + b Int q T_kk dl / a0"),
}


def F_sup(name, alpha, beta, p):
    """Supremum of F over qbar in [0,1), Tbar >= 0 ignored (Tbar term is
    separately bounded; see `tidal_line_bound`)."""
    if name == "F1_poly":
        return 1.0 + alpha
    if name == "F2_exp":
        return math.exp(alpha)
    if name == "F3_pade":
        return 1.0 + alpha / (1.0 + beta)
    if name == "F4_tidal":
        return 1.0 + alpha          # + beta * sup Tbar, added by caller
    raise KeyError(name)


# ==========================================================================
#  2.  VOID-STATE FIELDS q
# ==========================================================================
#  (i)  q_delta   : q = -delta/(1+delta) clipped to [0,1).  Note the algebra:
#                   -delta/(1+delta) = rho_ref/rho_s - 1, so q = 0 wherever
#                   rho_s >= rho_ref and q -> 1 once rho_s <= rho_ref/2.  It is
#                   a near-step function of the smoothed density.
#  (ii) q_screen  : (1 - L^2 lap) q = S(rho, g), S the programme's Q3 source
#                   1/(1 + (rho/rho_c)^m + (g/a0)^n).
#  (iii) q_smooth : q = 1/(1 + rho_s/rho_ref)  -- the programme's Q1 with m=1.
#                   Smooth everywhere, so it admits analytic gradients; used
#                   for the momentum residual and as the sensitivity control.

def q_from_delta(rho_s, rho_ref=RHO_BAR_B, eps=1e-12):
    """q = -delta/(1+delta) = rho_ref/rho_s - 1, clipped to [0, 1)."""
    rho_s = np.maximum(np.asarray(rho_s, float), 1e-300)
    return np.clip(rho_ref / rho_s - 1.0, 0.0, 1.0 - eps)


def q_from_smooth(rho_s, rho_ref=RHO_BAR_B, m=1.0):
    """q = 1 / (1 + (rho_s/rho_ref)^m).  Smooth, in (0, 1]."""
    rho_s = np.maximum(np.asarray(rho_s, float), 1e-300)
    return 1.0 / (1.0 + (rho_s / rho_ref) ** m)


def q_source_Q3(rho_s, gN, rho_ref=RHO_BAR_B, m=1.0, a0=A0, n=1.0):
    """S(rho, g) = 1/(1 + (rho/rho_c)^m + (g/a0)^n), the programme's Q3."""
    rho_s = np.maximum(np.asarray(rho_s, float), 1e-300)
    return 1.0 / (1.0 + (rho_s / rho_ref) ** m + (np.abs(gN) / a0) ** n)


# ------------------------------------------------- spherical field helpers
def smooth_spherical(r, rho, L, r_out=None):
    """Gaussian smoothing of a spherically symmetric rho(r) with scale L.

    The angular integral of a 3-D Gaussian against a shell is analytic:

        rho_s(r) = 1/(r L sqrt(2 pi)) Int rho(r') r'
                   [ e^{-(r-r')^2/2L^2} - e^{-(r+r')^2/2L^2} ] dr'

    `r` must be a monotonically increasing grid; the quadrature is trapezoid
    in r'.  With L -> 0 the identity rho_s -> rho is recovered.
    """
    r = np.asarray(r, float)
    rho = np.asarray(rho, float)
    if L <= 0:
        return rho.copy()
    rr = r[:, None]
    rp = r[None, :]
    ker = (np.exp(-(rr - rp) ** 2 / (2 * L ** 2))
           - np.exp(-(rr + rp) ** 2 / (2 * L ** 2)))
    integ = rho[None, :] * rp * ker
    out = np.trapezoid(integ, r, axis=1) / (r * L * math.sqrt(2 * math.pi))
    return out


def screen_1d(z, S, L):
    """Solve (1 - L^2 d^2/dz^2) q = S on a uniform line, q -> S at the ends.

    Used for the solar-neighbourhood check, where the steep direction is the
    vertical one and a 1-D profile is the honest reduction.
    """
    from scipy.linalg import solve_banded
    z = np.asarray(z, float)
    S = np.asarray(S, float)
    if L <= 0:
        return S.copy()
    h = z[1] - z[0]
    n = len(z)
    c = (L / h) ** 2
    ab = np.zeros((3, n))
    ab[0, 1:] = -c
    ab[1, :] = 1.0 + 2.0 * c
    ab[2, :-1] = -c
    b = S.copy()
    ab[0, 1] = 0.0
    ab[1, 0] = 1.0
    ab[1, -1] = 1.0
    ab[2, -2] = 0.0
    b[0] = S[0]
    b[-1] = S[-1]
    return solve_banded((1, 1), ab, b)


def smooth_axisym(R, z, rho, L):
    """Gaussian smoothing of an axisymmetric rho(R,z), scale L.

    The azimuthal integral of a 3-D Gaussian gives a modified Bessel function,
    and writing I_0(x) = ive(0,x) e^x combines the exponents into
    exp(-[(R-R')^2 + (z-z')^2] / 2L^2), which never overflows:

        rho_s = (2 pi L^2)^{-3/2} 2 pi Int dR' R' Int dz' rho ive(0, R R'/L^2)
                exp(-[(R-R')^2 + (z-z')^2]/(2 L^2))

    rho is given on the tensor grid (R, z) as rho[i,j].  Returns the same shape.
    """
    from scipy.special import ive
    R = np.asarray(R, float); z = np.asarray(z, float)
    if L <= 0:
        return np.asarray(rho, float).copy()
    pref = 2.0 * math.pi * (2.0 * math.pi * L ** 2) ** -1.5
    KR = ive(0, R[:, None] * R[None, :] / L ** 2) \
        * np.exp(-(R[:, None] - R[None, :]) ** 2 / (2 * L ** 2))
    Kz = np.exp(-(z[:, None] - z[None, :]) ** 2 / (2 * L ** 2))
    wR = np.gradient(R) * R
    wz = np.gradient(z)
    tmp = (rho * wz[None, :]) @ Kz.T                 # (nR, nz)
    out = (KR * wR[None, :]) @ tmp                   # (nR, nz)
    return pref * out


def screen_spherical(r, S, L):
    """Solve (1 - L^2 lap) q = S in spherical symmetry by its Green function.

    G_3(D) = e^{-D/L} / (4 pi L^2 D); the angular integral over a shell is
    analytic via the substitution mu -> D:

        q(r) = 1/(2 L r) Int S(r') r'
               [ e^{-|r-r'|/L} - e^{-(r+r')/L} ] dr'

    The r' grid must extend far enough that S has reached its ambient value,
    otherwise the missing outer shells act as a spurious Dirichlet condition.
    """
    r = np.asarray(r, float)
    S = np.asarray(S, float)
    if L <= 0:
        return S.copy()
    rr = r[:, None]
    rp = r[None, :]
    ker = (np.exp(-np.abs(rr - rp) / L) - np.exp(-(rr + rp) / L))
    integ = S[None, :] * rp * ker
    return np.trapezoid(integ, r, axis=1) / (2.0 * L * r)


# ==========================================================================
#  3.  SPHERICAL NONLOCAL SOLVER
# ==========================================================================
#  The double integral collapses to two dimensions.  Writing the separation
#  D = |x - x'| as the inner variable removes the 1/D singularity exactly:
#
#      Int dmu / D = Int dD / (r r')     because  dmu = -D dD / (r r')
#
#  so
#      Phi(r) = -(2 pi G / r) Int r' rho(r') [ Int_{|r-r'|}^{r+r'} F dD ] dr'
#
#  With F = 1 the inner bracket is 2 min(r, r') and the exact Newtonian
#  result -G M(<r)/r - 4 pi G Int_r^inf r' rho dr' drops out identically.
#  This is checked to round-off in the screen.
#
#  A point on the segment from x (radius r) to x' (radius r', separation D)
#  has radius
#      r_s(s)^2 = r^2 + s (r'^2 - r^2 - D^2) + s^2 D^2
#  which is exact and needs no angles.

_GL_CACHE: dict[int, tuple] = {}


def gauss_legendre(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


@dataclass
class SphericalField:
    """A spherically symmetric baryon distribution plus its q state."""
    r: np.ndarray                 # kpc, increasing, log-spaced
    rho: np.ndarray               # Msun/kpc^3
    q: np.ndarray                 # void state in [0,1)
    Menc: np.ndarray = field(default=None)     # Msun
    gN: np.ndarray = field(default=None)       # (km/s)^2/kpc, positive
    label: str = ""
    rho_fun: object = None        # optional exact callable rho(r)
    Menc_fun: object = None       # optional exact callable M(<r)

    def __post_init__(self):
        if self.Menc is None:
            if self.Menc_fun is not None:
                self.Menc = self.Menc_fun(self.r)
            else:
                integ = 4.0 * math.pi * self.r ** 2 * self.rho
                self.Menc = np.concatenate(
                    [[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1])
                                      * np.diff(self.r))])
        if self.gN is None:
            self.gN = G * self.Menc / np.maximum(self.r, 1e-30) ** 2

    def q_at(self, rs):
        """Interpolate q at radii rs (log-linear, clamped at the ends)."""
        lr = np.log(np.maximum(self.r, 1e-30))
        return np.interp(np.log(np.maximum(rs, 1e-30)), lr, self.q)

    def rho_at(self, rs):
        if self.rho_fun is not None:
            return self.rho_fun(rs)
        lr = np.log(np.maximum(self.r, 1e-30))
        return np.exp(np.interp(np.log(np.maximum(rs, 1e-30)), lr,
                                np.log(np.maximum(self.rho, 1e-300))))

    def Menc_at(self, rs):
        if self.Menc_fun is not None:
            return self.Menc_fun(rs)
        lr = np.log(np.maximum(self.r, 1e-30))
        return np.interp(np.log(np.maximum(rs, 1e-30)), lr, self.Menc)

    def d2Phi_dr2_at(self, rs):
        """Radial-radial Newtonian tidal component Phi'' = -2GM/r^3 + 4 pi G rho."""
        M = self.Menc_at(rs)
        rho = self.rho_at(rs)
        return -2.0 * G * M / np.maximum(rs, 1e-30) ** 3 + 4.0 * math.pi * G * rho

    def dPhi_dr_over_r_at(self, rs):
        M = self.Menc_at(rs)
        return G * M / np.maximum(rs, 1e-30) ** 3


def _path_radii(r, rp, D, s):
    """Radii along the straight segment, shape broadcast of (rp, D, s)."""
    a = rp ** 2 - r ** 2 - D ** 2
    val = r ** 2 + s * a + (s * D) ** 2
    return np.sqrt(np.maximum(val, 0.0))


def _path_cos2(r, rp, D, s, rs):
    """(n_hat . k_hat)^2 along the segment: n_hat radial, k_hat = (x'-x)/D.

    x(s).k_hat = [ x.(x'-x) + s D^2 ] / D and x.(x'-x) = (r'^2 - r^2 - D^2)/2,
    so cos = ( (r'^2 - r^2 - D^2)/2 + s D^2 ) / (D r_s).
    """
    num = 0.5 * (rp ** 2 - r ** 2 - D ** 2) + s * D ** 2
    den = np.maximum(D * rs, 1e-30)
    c = num / den
    return np.clip(c, -1.0, 1.0) ** 2


def _log_panels(lo, hi, split, n_pan, n_gl):
    """Composite Gauss-Legendre nodes/weights in ln r' over [lo,hi], with a
    panel boundary placed exactly at `split` so the |r-r'| kink sits on a
    node.  Returns (r', w) with w already carrying the dr' = r' dlnr factor."""
    xs, ws = gauss_legendre(n_gl)
    edges = np.concatenate([np.linspace(math.log(lo), math.log(split),
                                        n_pan + 1),
                            np.linspace(math.log(split), math.log(hi),
                                        n_pan + 1)[1:]])
    a = edges[:-1][:, None]
    b = edges[1:][:, None]
    u = 0.5 * (b + a) + 0.5 * (b - a) * xs[None, :]
    w = (0.5 * (b - a) * ws[None, :]).ravel()
    rp = np.exp(u.ravel())
    return rp, w * rp


def spherical_potential(fld: SphericalField, r_eval, Fname="F1_poly",
                        alpha=1.0, beta=0.0, p=1.0, n_D=32, n_s=16,
                        n_pan=24, n_gl=8, r_lo=None, r_hi=None,
                        use_tidal=None):
    """Phi(r) for the nonlocal kernel on a spherically symmetric source.

    Composite Gauss-Legendre in ln r' with a panel edge at r' = r, and
    Gauss-Legendre in D on [|r-r'|, r+r'].  Returns Phi in (km/s)^2.
    """
    Ffun = FAMILIES[Fname][0]
    if use_tidal is None:
        use_tidal = (Fname == "F4_tidal") and beta != 0.0
    xs, ws = gauss_legendre(n_D)
    ss, wss = gauss_legendre(n_s)
    s_nodes = 0.5 * (ss + 1.0)
    s_w = 0.5 * wss
    lo = fld.r[0] if r_lo is None else r_lo
    hi = fld.r[-1] if r_hi is None else r_hi

    out = np.empty(len(np.atleast_1d(r_eval)), float)
    for i, r in enumerate(np.atleast_1d(r_eval)):
        rp, wr = _log_panels(lo, hi, min(max(r, lo * 1.0000001),
                                         hi * 0.9999999), n_pan, n_gl)
        rho = fld.rho_at(rp)
        a = np.abs(r - rp)
        b = r + rp
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        D = mid[:, None] + half[:, None] * xs[None, :]        # (nr', nD)
        wD = half[:, None] * ws[None, :]
        rs = _path_radii(r, rp[:, None, None], D[:, :, None],
                         s_nodes[None, None, :])               # (nr',nD,ns)
        qs = fld.q_at(rs)
        qbar = np.tensordot(qs, s_w, axes=([2], [0]))          # (nr', nD)
        if use_tidal:
            c2 = _path_cos2(r, rp[:, None, None], D[:, :, None],
                            s_nodes[None, None, :], rs)
            radial = fld.dPhi_dr_over_r_at(rs)
            Tkk = (fld.d2Phi_dr2_at(rs) - radial) * c2 + radial
            # line integral: dl = D ds, normalised by a0
            Tbar = np.tensordot(qs * Tkk, s_w, axes=([2], [0])) * D / A0
        else:
            Tbar = np.zeros_like(qbar)
        Fv = Ffun(qbar, Tbar, alpha=alpha, beta=beta, p=p)
        inner = np.sum(Fv * wD, axis=1)                        # (nr',)
        out[i] = -(2.0 * math.pi * G / r) * np.sum(wr * rp * rho * inner)
    return out


def _global_panels(r_lo, r_hi, r_eval, dlnr_max=0.35, n_gl=8):
    """One quadrature in ln r' whose panel edges include EVERY evaluation
    radius, so the |r-r'| kink always lands on a panel boundary and a single
    node set serves all field radii.  That is what makes the batched (GPU)
    evaluation possible without losing the exponential convergence."""
    edges = np.unique(np.concatenate(
        [[math.log(r_lo)], np.log(np.atleast_1d(r_eval)), [math.log(r_hi)]]))
    full = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:]):
        k = max(1, int(math.ceil((b - a) / dlnr_max)))
        full.extend(np.linspace(a, b, k + 1)[1:])
    full = np.asarray(full)
    xs, ws = gauss_legendre(n_gl)
    a = full[:-1][:, None]
    b = full[1:][:, None]
    u = 0.5 * (b + a) + 0.5 * (b - a) * xs[None, :]
    w = (0.5 * (b - a) * ws[None, :]).ravel()
    rp = np.exp(u.ravel())
    return rp, w * rp


def spherical_potential_batch(fld: SphericalField, r_eval, Fname="F1_poly",
                              alpha=1.0, beta=0.0, p=1.0, n_D=32, n_s=12,
                              n_gl=8, dlnr_max=0.35, r_lo=None, r_hi=None,
                              use_tidal=None, use_gpu=False, chunk=8):
    """Batched Phi(r_eval).  Identical mathematics to `spherical_potential`,
    but one shared r' quadrature for all field radii (see `_global_panels`)."""
    xp = _xp(use_gpu)
    Ffun = FAMILIES[Fname][0]
    if use_tidal is None:
        use_tidal = (Fname == "F4_tidal") and beta != 0.0
    r_eval = np.atleast_1d(np.asarray(r_eval, float))
    lo = fld.r[0] if r_lo is None else r_lo
    hi = fld.r[-1] if r_hi is None else r_hi
    rp_np, wr_np = _global_panels(lo, hi, r_eval, dlnr_max, n_gl)
    rho_np = fld.rho_at(rp_np)
    xsn, wsn = gauss_legendre(n_D)
    ssn, wssn = gauss_legendre(n_s)

    rp = xp.asarray(rp_np); wr = xp.asarray(wr_np); rho = xp.asarray(rho_np)
    xs = xp.asarray(xsn); ws = xp.asarray(wsn)
    s_nodes = xp.asarray(0.5 * (ssn + 1.0)); s_w = xp.asarray(0.5 * wssn)
    lrq = xp.asarray(np.log(np.maximum(fld.r, 1e-30)))
    qq = xp.asarray(fld.q)
    if use_tidal:
        lrm = lrq
        Mg = xp.asarray(fld.Menc)
        rhog = xp.asarray(np.maximum(fld.rho, 1e-300))

    out = np.empty(len(r_eval), float)
    for k0 in range(0, len(r_eval), chunk):
        rv = xp.asarray(r_eval[k0:k0 + chunk])[:, None]          # (c,1)
        a = xp.abs(rv - rp[None, :])
        b = rv + rp[None, :]
        half = 0.5 * (b - a); mid = 0.5 * (b + a)
        D = mid[:, :, None] + half[:, :, None] * xs[None, None, :]
        wD = half[:, :, None] * ws[None, None, :]
        rr = rv[:, :, None, None]
        rpb = rp[None, :, None, None]
        Db = D[:, :, :, None]
        sb = s_nodes[None, None, None, :]
        val = rr ** 2 + sb * (rpb ** 2 - rr ** 2 - Db ** 2) + (sb * Db) ** 2
        rs = xp.sqrt(xp.maximum(val, 0.0))
        qs = xp.interp(xp.log(xp.maximum(rs, 1e-30)), lrq, qq)
        qbar = xp.tensordot(qs, s_w, axes=([3], [0]))
        if use_tidal:
            lrs = xp.log(xp.maximum(rs, 1e-30))
            Mv = xp.interp(lrs, lrm, Mg)
            rhov = xp.exp(xp.interp(lrs, lrm, xp.log(rhog)))
            rad = G * Mv / xp.maximum(rs, 1e-30) ** 3
            d2 = -2.0 * rad + 4.0 * math.pi * G * rhov
            num = 0.5 * (rpb ** 2 - rr ** 2 - Db ** 2) + sb * Db ** 2
            c2 = xp.clip(num / xp.maximum(Db * rs, 1e-30), -1.0, 1.0) ** 2
            Tkk = (d2 - rad) * c2 + rad
            Tbar = xp.tensordot(qs * Tkk, s_w, axes=([3], [0])) * D / A0
        else:
            Tbar = xp.zeros_like(qbar)
        Fv = Ffun(qbar, Tbar, alpha=alpha, beta=beta, p=p)
        inner = xp.sum(Fv * wD, axis=2)                          # (c, nr')
        res = -(2.0 * math.pi * G / rv[:, 0]) * xp.sum(
            (wr * rp * rho)[None, :] * inner, axis=1)
        out[k0:k0 + chunk] = xp.asnumpy(res) if use_gpu else res
    return out


def spherical_vcirc_spline(fld: SphericalField, r_eval, npts=None, pad=0.25,
                           **kw):
    """v_c^2 = dPhi/dln r from a cubic spline of Phi(ln r).

    Phi is analytic in ln r away from the source edges, so the spline
    derivative is far cheaper than a stencil per radius and is checked
    against the exact Newtonian gradient at alpha = 0 in the screen.
    """
    from scipy.interpolate import CubicSpline
    r_eval = np.atleast_1d(np.asarray(r_eval, float))
    if npts is None:
        npts = max(48, 3 * len(r_eval))
    lo = math.log(r_eval[0]) - pad
    hi = math.log(r_eval[-1]) + pad
    rg = np.exp(np.linspace(lo, hi, npts))
    phi = spherical_potential_batch(fld, rg, **kw)
    cs = CubicSpline(np.log(rg), phi)
    return cs(np.log(r_eval), 1), rg, phi


def spherical_vcirc(fld: SphericalField, r_eval, dlog=2e-3, **kw):
    """v_c^2 = r dPhi/dr by a five-point log-radius stencil on Phi.

    Phi is smooth in ln r, so the stencil is accurate; the screen also checks
    it against the exact Newtonian gradient with alpha = 0.
    """
    r_eval = np.atleast_1d(np.asarray(r_eval, float))
    offs = np.array([-2, -1, 0, 1, 2]) * dlog
    coef = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * dlog)
    rr = (r_eval[:, None] * np.exp(offs[None, :])).ravel()
    phi = spherical_potential(fld, rr, **kw).reshape(len(r_eval), 5)
    dphi_dlnr = phi @ coef
    return dphi_dlnr           # = r dPhi/dr = v_c^2


def spherical_F_effective(fld: SphericalField, r_eval, **kw):
    """The effective F(r) implied by the computed potential: F = -r Phi/(G M_tot).

    For a point mass this is exactly the kernel factor; for an extended source
    it is the quantity that enters the far-field theorem.
    """
    Phi = spherical_potential_batch(fld, r_eval, **kw)
    Mtot = fld.Menc[-1]
    return -np.asarray(r_eval, float) * Phi / (G * Mtot)


# ==========================================================================
#  4.  EXACT FAR-FIELD / POINT-MASS THEORY
# ==========================================================================

def vc2_pointmass(M, r, F, dFdr):
    """v_c^2 = G M (F/r - dF/dr) for Phi = -G M F(r)/r.  Exact."""
    return G * M * (F / r - dFdr)


def required_F(r, v2, M, F_ref=1.0, r_ref=None):
    """The F(r) profile a given rotation curve demands, for a point source.

        d/dr (F/r) = -v^2 / (G M r)   =>   F(r)/r = F(r0)/r0 - Int v^2/(G M r) dr

    Equivalently F = -r Phi/(G M) with Phi the potential of the observed curve.
    `r_ref` defaults to r[0].
    """
    r = np.asarray(r, float)
    v2 = np.asarray(v2, float)
    if r_ref is None:
        r_ref = r[0]
    lnr = np.log(r)
    integ = np.concatenate([[0.0], np.cumsum(
        0.5 * (v2[1:] + v2[:-1]) * np.diff(lnr))])
    return r * (F_ref / r_ref - integ / (G * M))


def flat_window(alpha, F_family="F1_poly", beta=0.0):
    """Widest radial range over which a BOUNDED F can hold v_c exactly flat.

    Solving F - r F' = C r gives F(r) = C r ln(r_*/r), the only profile that
    makes v^2 = G M C constant.  It peaks at r = r_*/e with F = C r_*/e.  With
    1 <= F <= F_max the flat stretch is bounded by the two roots of
    C r ln(r_*/r) = 1, and the widest possible stretch sets the apex on the
    ceiling: C r_* / e = F_max.  Returns (r2/r1 with F allowed to rise then
    fall, r_apex/r1 with F required monotone).
    """
    Fmax = F_sup(F_family, alpha, beta, 1.0)
    if Fmax <= 1.0:
        return 1.0, 1.0
    eps = 1.0 / (math.e * Fmax)          # value of u ln(1/u) at F = 1
    from scipy.optimize import brentq
    f = lambda u: u * math.log(1.0 / u) - eps
    lo = brentq(f, 1e-14, 1.0 / math.e)
    hi = brentq(f, 1.0 / math.e, 1.0 - 1e-14)
    return hi / lo, (1.0 / math.e) / lo


# ==========================================================================
#  5.  TWO-BODY FORCES, RECIPROCITY, MOMENTUM RESIDUAL
# ==========================================================================

@dataclass
class GaussianCloud:
    """A sum of Gaussian blobs: an analytic smoothed density with analytic
    gradient, so the momentum residual can be evaluated without differencing."""
    pos: np.ndarray               # (N,3) kpc
    mass: np.ndarray              # (N,) Msun
    L: float                      # kpc, smoothing scale
    rho_amb: float = 0.0          # Msun/kpc^3, uniform ambient floor

    def rho(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        d2 = np.sum((x[:, None, :] - self.pos[None, :, :]) ** 2, axis=2)
        amp = self.mass / (2 * math.pi * self.L ** 2) ** 1.5
        return np.sum(amp[None, :] * np.exp(-d2 / (2 * self.L ** 2)),
                      axis=1) + self.rho_amb

    def grad_rho(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        dx = x[:, None, :] - self.pos[None, :, :]
        d2 = np.sum(dx ** 2, axis=2)
        amp = self.mass / (2 * math.pi * self.L ** 2) ** 1.5
        w = amp[None, :] * np.exp(-d2 / (2 * self.L ** 2)) / (-self.L ** 2)
        return np.sum(w[:, :, None] * dx, axis=1)


def q_cloud(cloud: GaussianCloud, x, rho_ref=RHO_BAR_B, kind="smooth"):
    rho = cloud.rho(x)
    if kind == "smooth":
        return 1.0 / (1.0 + rho / rho_ref)
    return q_from_delta(rho, rho_ref)


def grad_q_cloud(cloud: GaussianCloud, x, rho_ref=RHO_BAR_B, kind="smooth"):
    rho = cloud.rho(x)
    gr = cloud.grad_rho(x)
    if kind == "smooth":
        fac = -1.0 / (rho_ref * (1.0 + rho / rho_ref) ** 2)
    else:
        inside = (rho_ref / rho - 1.0)
        fac = np.where((inside > 0) & (inside < 1.0),
                       -rho_ref / rho ** 2, 0.0)
    return fac[:, None] * gr


def path_qbar(cloud, x1, x2, n_s=64, rho_ref=RHO_BAR_B, kind="smooth",
              weight=None):
    """qbar and its two endpoint gradients along the segment x1 -> x2.

    `weight` allows a deliberately NON-reciprocal path weighting w(s) with
    Int w = 1; w(s) = 1 is the symmetric path average.  Returns
    (qbar, grad_1 qbar, grad_2 qbar, <grad q>_path).
    """
    ss, ws = gauss_legendre(n_s)
    s = 0.5 * (ss + 1.0)
    w = 0.5 * ws
    if weight is not None:
        wv = weight(s)
        w = w * wv
        w = w / np.sum(w)
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    pts = x1[None, :] + s[:, None] * (x2 - x1)[None, :]
    qv = q_cloud(cloud, pts, rho_ref, kind)
    gv = grad_q_cloud(cloud, pts, rho_ref, kind)
    qbar = float(np.sum(w * qv))
    g1 = np.sum((w * (1.0 - s))[:, None] * gv, axis=0)
    g2 = np.sum((w * s)[:, None] * gv, axis=0)
    gsum = np.sum(w[:, None] * gv, axis=0)
    return qbar, g1, g2, gsum


def pair_forces(cloud, x1, x2, m1, m2, Fname="F1_poly", alpha=1.0, beta=0.0,
                p=1.0, n_s=64, rho_ref=RHO_BAR_B, kind="smooth",
                weight=None, weight_rev=None):
    """Forces on two point masses from the nonlocal kernel, q held fixed.

    f_1 = G m1 m2 [ F (x2-x1)/D^3 + F'(qbar) grad_1 qbar / D ]
    f_2 = G m1 m2 [ F (x1-x2)/D^3 + F'(qbar) grad_2 qbar / D ]

    For a non-reciprocal kernel the two use different path weights, so F is
    evaluated twice.  Returns (f1, f2, diagnostics).
    """
    Ffun, dFfun, _ = FAMILIES[Fname]
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    d = x2 - x1
    D = float(np.linalg.norm(d))
    qb12, g1_12, g2_12, gsum12 = path_qbar(cloud, x1, x2, n_s, rho_ref,
                                           kind, weight)
    if weight_rev is None and weight is None:
        qb21, g1_21, g2_21 = qb12, g1_12, g2_12
    else:
        wr = weight_rev if weight_rev is not None else weight
        qb21, g2_21, g1_21, _ = path_qbar(cloud, x2, x1, n_s, rho_ref,
                                          kind, wr)
    F12 = float(Ffun(qb12, 0.0, alpha=alpha, beta=beta, p=p))
    F21 = float(Ffun(qb21, 0.0, alpha=alpha, beta=beta, p=p))
    dF12 = float(dFfun(qb12, 0.0, alpha=alpha, beta=beta, p=p))
    dF21 = float(dFfun(qb21, 0.0, alpha=alpha, beta=beta, p=p))
    pre = G * m1 * m2
    f1 = pre * (F12 * d / D ** 3 + dF12 * g1_12 / D)
    f2 = pre * (F21 * (-d) / D ** 3 + dF21 * g2_21 / D)
    q1 = float(q_cloud(cloud, x1[None, :], rho_ref, kind)[0])
    q2 = float(q_cloud(cloud, x2[None, :], rho_ref, kind)[0])
    diag = dict(D=D, qbar=qb12, qbar_rev=qb21, F=F12, F_rev=F21,
                dF=dF12, dF_rev=dF21, g1=g1_12, g2=g2_21, q1=q1, q2=q2,
                f_newton=pre / D ** 2,
                grad_q_path=gsum12,
                identity_axial=float((q2 - q1) / D),
                measured_axial=float(np.dot(gsum12, d / D)))
    return f1, f2, diag


def nbody_forces(cloud, pos, mass, Fname="F1_poly", alpha=1.0, beta=0.0,
                 p=1.0, n_s=64, rho_ref=RHO_BAR_B, kind="smooth",
                 weight=None):
    """Total force on every body of an ISOLATED system whose q is sourced by
    those same bodies.  Returns (forces, net, decomposition).

    The net force must vanish for a momentum-conserving theory.  It is split
    into
      'gradient'  : G m_i m_j F'(qbar) <grad q>_path / D   -- present for any
                    reciprocal kernel whenever the two bodies sit at different
                    q, because <grad q>_path . n = [q_j - q_i]/D exactly;
      'asymmetry' : G m_i m_j [F(qbar_ij) - F(qbar_ji)] n_ij / D^2 -- present
                    only when the path weighting is not symmetric.
    """
    pos = np.asarray(pos, float)
    mass = np.asarray(mass, float)
    N = len(mass)
    f = np.zeros((N, 3))
    grad_part = np.zeros(3)
    asym_part = np.zeros(3)
    # A deliberately non-reciprocal kernel applies the SAME w(s) measured from
    # whichever endpoint is named first; passing w(1-s) would silently restore
    # reciprocity, which is exactly the bug this comment exists to prevent.
    wrev = weight
    for i in range(N):
        for j in range(i + 1, N):
            fi, fj, d = pair_forces(cloud, pos[i], pos[j], mass[i], mass[j],
                                    Fname=Fname, alpha=alpha, beta=beta, p=p,
                                    n_s=n_s, rho_ref=rho_ref, kind=kind,
                                    weight=weight, weight_rev=wrev)
            f[i] += fi
            f[j] += fj
            nij = (pos[j] - pos[i]) / d["D"]
            pre = G * mass[i] * mass[j]
            # exact split: f_i + f_j = pre (F_ij - F_ji) n / D^2
            #                        + pre [ F'_ij grad_i qbar + F'_ji grad_j qbar ] / D
            asym_part += pre * (d["F"] - d["F_rev"]) * nij / d["D"] ** 2
            grad_part += pre * (d["dF"] * d["g1"]
                                + d["dF_rev"] * d["g2"]) / d["D"]
    return f, f.sum(axis=0), dict(gradient=grad_part, asymmetry=asym_part)


# ==========================================================================
#  6.  3-D DIRECT SUM AND THE LOW-RANK / FFT ACCELERATION
# ==========================================================================

def _xp(use_gpu):
    if use_gpu:
        import cupy as cp
        return cp
    return np


def grid3d(n, Lbox):
    h = Lbox / n
    ax = (np.arange(n) - n / 2 + 0.5) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return h, ax, X, Y, Z


def direct_potential_3d(pos_f, pos_s, mass_s, qgrid, ax, Fname="F1_poly",
                        alpha=1.0, beta=0.0, p=1.0, n_s=16, soft=0.0,
                        use_gpu=False, chunk=4096, qbar_mode="path",
                        soft_mode="plummer"):
    """Exact O(N_f N_s n_s) evaluation of the nonlocal potential.

    q is supplied as a regular 3-D grid on the axes `ax` (same axis in x,y,z)
    and is trilinearly interpolated along each segment.

    qbar_mode='midpoint' replaces the path average by (q(x) + q(x'))/2, the
    separable surrogate the FFT accelerator uses.  Running both isolates the
    surrogate error from the SVD-truncation error.
    """
    xp = _xp(use_gpu)
    Ffun = FAMILIES[Fname][0]
    ss, ws = gauss_legendre(n_s)
    s = xp.asarray(0.5 * (ss + 1.0))
    w = xp.asarray(0.5 * ws)
    Pf = xp.asarray(pos_f, dtype=xp.float64)
    Ps = xp.asarray(pos_s, dtype=xp.float64)
    Ms = xp.asarray(mass_s, dtype=xp.float64)
    Q = xp.asarray(qgrid, dtype=xp.float64)
    a0 = float(ax[0]); dh = float(ax[1] - ax[0]); n = len(ax)

    def interp(pts):
        u = (pts - a0) / dh
        u = xp.clip(u, 0.0, n - 1.0000001)
        i0 = xp.floor(u).astype(xp.int32)
        t = u - i0
        i1 = i0 + 1
        c = 0.0
        for bx in (0, 1):
            for by in (0, 1):
                for bz in (0, 1):
                    wx = t[..., 0] if bx else 1 - t[..., 0]
                    wy = t[..., 1] if by else 1 - t[..., 1]
                    wz = t[..., 2] if bz else 1 - t[..., 2]
                    ix = i1[..., 0] if bx else i0[..., 0]
                    iy = i1[..., 1] if by else i0[..., 1]
                    iz = i1[..., 2] if bz else i0[..., 2]
                    c = c + wx * wy * wz * Q[ix, iy, iz]
        return c

    out = xp.zeros(Pf.shape[0], dtype=xp.float64)
    qf = interp(Pf) if qbar_mode == "midpoint" else None
    # Auto-chunk on a fixed element budget.  The interpolation allocates about
    # ten temporaries of shape (N_f, chunk, n_s), so a chunk chosen without
    # reference to N_f is how this routine runs out of memory on a large grid.
    budget = 4_000_000 if not use_gpu else 40_000_000
    chunk = max(1, min(chunk, int(budget / max(Pf.shape[0] * n_s, 1))))
    for k in range(0, Ps.shape[0], chunk):
        Pc = Ps[k:k + chunk]
        Mc = Ms[k:k + chunk]
        dvec = Pc[None, :, :] - Pf[:, None, :]
        if soft_mode == "plateau":
            # D = max(|x - x'|, soft): exactly the kernel the FFT convolution
            # tabulates, so the two can be compared without a softening
            # convention masquerading as an acceleration error.
            D = xp.maximum(xp.sqrt(xp.sum(dvec ** 2, axis=2)), soft)
        else:
            D = xp.sqrt(xp.sum(dvec ** 2, axis=2) + soft ** 2)
        if qbar_mode == "midpoint":
            qbar = 0.5 * (qf[:, None] + interp(Pc)[None, :])
        else:
            pts = (Pf[:, None, None, :]
                   + s[None, None, :, None] * dvec[:, :, None, :])
            qv = interp(pts)
            qbar = xp.tensordot(qv, w, axes=([2], [0]))
        Fv = Ffun(qbar, 0.0, alpha=alpha, beta=beta, p=p)
        out += xp.sum(Mc[None, :] * Fv / D, axis=1)
    out = -G * out
    return xp.asnumpy(out) if use_gpu else out


def midpoint_qbar_matrix(q_f, q_s):
    """qbar ~ (q(x) + q(x'))/2 -- the separable (rank-deficient) surrogate."""
    return 0.5 * (q_f[:, None] + q_s[None, :])


def lowrank_factors(Fname, alpha, beta, p, rank, nq=129):
    """SVD of F((u+v)/2) on a q x q grid -> separable factors a_m(u), b_m(v).

    Returns callables (A(u), B(v)) giving arrays of shape (..., rank), such
    that F((u+v)/2) ~ sum_m A_m(u) B_m(v).  With that factorisation each term
    is an ordinary 1/r convolution and the whole potential is R FFTs.
    """
    Ffun = FAMILIES[Fname][0]
    u = np.linspace(0.0, 1.0, nq)
    M = Ffun(0.5 * (u[:, None] + u[None, :]), 0.0, alpha=alpha, beta=beta, p=p)
    U, S, Vt = np.linalg.svd(M)
    R = min(rank, len(S))
    A = U[:, :R] * S[:R][None, :]
    B = Vt[:R, :].T
    rel = float(np.sqrt(np.sum(S[R:] ** 2)) / np.sqrt(np.sum(S ** 2))) \
        if R < len(S) else 0.0

    def Afun(x):
        x = np.asarray(x, float)
        return np.stack([np.interp(x, u, A[:, m]) for m in range(R)], axis=-1)

    def Bfun(x):
        x = np.asarray(x, float)
        return np.stack([np.interp(x, u, B[:, m]) for m in range(R)], axis=-1)

    return Afun, Bfun, S, rel


def fft_convolve_invr(rho, h, soft=None):
    """Zero-padded FFT convolution of rho with 1/|r| on a cubic grid."""
    n = rho.shape[0]
    N = 2 * n
    ax = (np.arange(N) - N // 2) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    if soft is None:
        soft = 0.5 * h
    K = 1.0 / np.maximum(R, soft)
    K = np.fft.ifftshift(K)
    pad = np.zeros((N, N, N))
    pad[:n, :n, :n] = rho
    conv = np.fft.irfftn(np.fft.rfftn(pad) * np.fft.rfftn(K), s=(N, N, N))
    return conv[:n, :n, :n]


def lowrank_potential_3d(rho, qgrid, h, Fname="F1_poly", alpha=1.0, beta=0.0,
                         p=1.0, rank=6):
    """Nonlocal potential via the separable midpoint surrogate + FFT.

    Phi(x) = -G sum_m A_m(q(x)) * [ (B_m(q) rho) conv 1/r ](x) * h^3
    Cost O(R n^3 log n) instead of O(n^6 n_s).
    """
    Afun, Bfun, S, rel = lowrank_factors(Fname, alpha, beta, p, rank)
    A = Afun(qgrid)
    B = Bfun(qgrid)
    acc = np.zeros_like(rho)
    for m in range(A.shape[-1]):
        acc += A[..., m] * fft_convolve_invr(B[..., m] * rho, h)
    return -G * acc * h ** 3, rel
