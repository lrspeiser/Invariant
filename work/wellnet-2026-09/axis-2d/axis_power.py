"""JOB 1 -- THREE SEPARATED POWER SURFACES, one per AXIS PROVENANCE.

Every tensor test in this programme so far has asked "is there some tensor
atom?".  That question conflates three physically distinct hypotheses whose
SPHERICAL LIMITS differ:

    source axis         the anisotropy is built out of the source's own
                        structure.  As the source becomes spherical the
                        response tensor becomes  k_r rhat rhat^T + k_t (I -
                        rhat rhat^T)  and the spherical blindness theorem says
                        k_t is EXACTLY unobservable.  Power MUST collapse.
    external tidal axis the anisotropy is built on a direction the source does
                        not set.  A spherical source still has an anisotropic
                        field, K is not spherically symmetric, and the
                        blindness theorem does NOT apply.  Power must NOT
                        collapse.
    member-well network the anisotropy is set by the distribution of
                        CATALOGUED wells.  Its spherical limit is whatever the
                        well distribution is, and it must additionally survive
                        coarse-graining -- a real coherence scale cannot be
                        set by the cataloguer.

Run AC's near-spherical control conflated them.  Its injected law used the
grammar's `dd` basis, which is dhat dhat^T for a FIXED dhat -- an EXTERNAL
axis.  So its near-spherical row was never covered by the blindness theorem,
and the fact that its power did not collapse was correct behaviour being read
as a failed control.  This module separates the three, gives each its own null
and its own empirical critical value, and shows the source-axis surface
collapsing while the external-axis surface does not.

WHAT IS NEW HERE BESIDES THE SPLIT

1. THE NULL IS A STAGE-0 NULL, not a member of the bank's own grammar.  Every
   null realisation carries, simultaneously:
       * a scalar response drawn from FIVE qualitatively different families
         (alternative interpolating function, potential-depth gate, mean-
         density gate, tidal gate, nonlocal screened fraction) with random
         parameters, none of which is in the atom bank;
       * a TRIAXIAL BARYON DISTRIBUTION AT A RANDOM PROJECTION -- the source
         that generates the data is not the source that built the bank;
       * line-of-sight DEPROJECTION ERROR (the depth is rescaled);
       * MISCENTERING (the mass is offset from the assumed centre);
       * a radial mass-to-light gradient;
       * multiplicative SHEAR-CALIBRATION error and an additive c-term.
   A flexible tensor model can absorb any of these and call it anisotropy,
   which is exactly why they are in the null and not in a caveat.

2. THE OBSERVABLE IS TWO-DIMENSIONAL.  The detector sees the midplane radial
   acceleration AND both components of the projected shear (gamma_1, gamma_2)
   -- not |alpha|, not an azimuthal average.  Azimuthal averaging discards the
   directional information this lane exists to use.

3. THE STATISTIC IS CROSS-FITTED.  The scalar nuisance is fitted on one half
   of the observable points and evaluated on the other, so an anisotropic atom
   cannot win merely because the scalar interpolating function was imperfect
   on the points it was fitted to.

4. THREE DISJOINT SIMULATION SETS.  Calibration sims set the critical value D*
   as the 95th percentile of the null statistic; UNTOUCHED audit sims verify
   the realised false-positive rate at that D*; injection sims measure power.
   "5% by construction" is not a measurement when the percentile was taken on
   the same simulations.

5. AMPLITUDE IS TRANSLATED.  Each injection records the maximum fractional
   acceleration anisotropy max|dg|/g that its response tensor produces, so a
   point on the power surface can be compared with a candidate law's predicted
   amplitude rather than with an abstract knob.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
if LANE not in sys.path:
    sys.path.insert(0, LANE)

from field_grammar import (A0, G, GPU, KPC, MSUN, Poisson, SHAPES,   # noqa
                           div, grad, nu_rar, sphericity, xp)
from field_search import Search, chol_solve                          # noqa
from qumond_degeneracy import independent_atoms                      # noqa

# ------------------------------------------------------------------ geometry
N = 48
L_KPC = 4000.0
RS_KPC = 700.0
MTOT = 3.0e14 * MSUN
R_IN_KPC, R_OUT_KPC = 250.0, 1500.0
SCALES = (0.3, 1.0, 3.0)              # the bank's grid
INJ_SCALES = (0.55, 1.8)              # deliberately BETWEEN the bank's points
PROVENANCE = ("source", "external", "network")
TILT_DEG = (0, 45, 90)

#: the five fixed directions whose (dhat dhat^T - I/3) SPAN every constant
#: symmetric traceless 3x3.  The external axis is not known a priori, so the
#: search must cover all of them -- and the look-elsewhere cost of that choice
#: then goes through the calibration, which is what the standing brief means by
#: "every null realisation must pass through ... choice of axis".
DDIRS = {"dd0": (1.0, 0.0, 0.0),
         "dd45": (1.0, 1.0, 0.0),
         "dd90": (0.0, 1.0, 0.0),
         "ddxz": (1.0, 0.0, 1.0),
         "ddyz": (0.0, 1.0, 1.0)}
#: three orientations of the CATALOGUED well distribution, relative to the
#: smooth source's major axis
NNTILTS = {"nn0": 0.0, "nn45": math.pi / 4, "nn90": math.pi / 2}

BASIS_OF = {"source": ["That"],
            "external": list(DDIRS),
            "network": list(NNTILTS)}
#: the "axis known" variants: the environment map of JOB 3 supplies the
#: external axis, so the search does not have to find it.  Measuring both says
#: how much an independently measured axis is worth.
for _t in TILT_DEG:
    BASIS_OF[f"external_known{_t}"] = [f"dd{_t}"]
    BASIS_OF[f"network_known{_t}"] = [f"nn{_t}"]
ARMS = (list(PROVENANCE)
        + [f"external_known{t}" for t in TILT_DEG]
        + [f"network_known{t}" for t in TILT_DEG])


def to_np(a):
    return a.get() if GPU else np.asarray(a)


# ------------------------------------------------------------------- sources
def grids(n=N, L_kpc=L_KPC):
    L = L_kpc * KPC
    h = L / n
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]
    return h, X, Y, Z


def make_rho(q=0.5, tilt=0.0, depth=1.0, dx=(0.0, 0.0, 0.0), grad_ml=0.0,
             n=N, L_kpc=L_KPC, rs_kpc=RS_KPC, M=MTOT):
    """A triaxial source: prolate along x with axis ratio q in both minor axes.

    q -> 1 is the spherical limit, which is the axis the blindness theorem
    talks about.  `tilt` rotates the source in the SKY PLANE (x, y), which is
    the rotation the projected shear can see.  `depth` rescales the
    line-of-sight extent only -- that is deprojection error.  `dx` offsets the
    mass from the assumed centre -- that is miscentering.  `grad_ml` tilts the
    profile radially, standing in for a mass-to-light gradient.
    """
    h, X, Y, Z = grids(n, L_kpc)
    X = X - dx[0] * KPC
    Y = Y - dx[1] * KPC
    Z = Z - dx[2] * KPC
    if tilt != 0.0:
        c, s = math.cos(tilt), math.sin(tilt)
        X, Y = c * X + s * Y, -s * X + c * Y
    u = xp.sqrt(X ** 2 + (Y / q) ** 2 + (Z / (q * depth)) ** 2)
    rho = xp.exp(-(u / (rs_kpc * KPC)) ** 2)
    if grad_ml != 0.0:
        rho = rho * (1.0 + grad_ml * u / (rs_kpc * KPC))
        rho = xp.maximum(rho, 0.0)
    rho *= M / float(rho.sum() * h ** 3)
    return rho, h


def wells(nw, q=0.5, tilt=0.0, seed=0, rs_kpc=RS_KPC, M=MTOT):
    """A catalogue of discrete wells, the thing a member-well tensor is built on.

    Drawn from an ellipsoid with its OWN axis ratio and orientation, because
    the network hypothesis says the axis is set by the catalogue, not by the
    smooth source.  nw = 1 is the fully coarse-grained limit, which is what the
    coarse-graining gate compares against.
    """
    rng = np.random.default_rng(seed)
    p = rng.normal(0.0, 1.0, (nw, 3)) * (rs_kpc * KPC / math.sqrt(2.0))
    p[:, 1] *= q
    p[:, 2] *= q
    c, s = math.cos(tilt), math.sin(tilt)
    x, y = c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]
    p[:, 0], p[:, 1] = x, y
    return xp.asarray(p), xp.asarray(np.full(nw, M / nw))


def flux_orthogonal(comp, u):
    """Remove the part of a response tensor that only RESCALES the flux.

    THE REPAIR THIS FUNCTION IS.  The first version of this module put the raw
    tensor bases in the detector and the source-axis power surface did NOT
    collapse in the spherical limit -- 0.42 to 0.62 at axis ratio 0.970, flat
    in axis ratio.  The standing brief says that means a bug, and the blindness
    theorem says where it is.

    In spherical symmetry ghat is an eigenvector of any spherically symmetric
    K, so K grad Phi_N = (ghat^T K ghat) grad Phi_N: the tensor acts as a
    SCALAR rescaling with radial profile ghat^T K ghat.  That profile is not in
    the span of the bank's scalar weights, so admitting the tensor atom
    genuinely improves the fit -- with no anisotropy anywhere.  The detector was
    measuring extra scalar flexibility and calling it a tensor.

    The blindness theorem says the transverse eigenvalue is unobservable.  It
    does NOT say the tensor atom is unobservable.  So the atom is redefined as

        K_perp = B - (ghat^T B ghat) I,      ghat^T K_perp ghat = 0

    whose action on grad Phi_N has NO component along grad Phi_N.  It can only
    TURN the flux.  In exact spherical symmetry K_perp grad Phi_N vanishes
    identically, so the source-axis atoms are exactly null there and the power
    surface must collapse -- now by construction rather than by hope.  For a
    fixed EXTERNAL direction ghat is not an eigenvector even for a spherical
    source, so those atoms survive, which is precisely the distinction this
    lane exists to draw.

    It also disposes of the QUMOND degeneracy automatically: for B = ghat
    ghat^T the radial part is 1 and K_perp grad Phi_N = 0 identically, so the
    field-direction projector drops out of the grammar without a special case.
    """
    axx, ayy, azz, axy, axz, ayz = comp
    ux, uy, uz = u
    rad = (axx * ux * ux + ayy * uy * uy + azz * uz * uz
           + 2.0 * (axy * ux * uy + axz * ux * uz + ayz * uy * uz))
    return (axx - rad, ayy - rad, azz - rad, axy, axz, ayz)


# ------------------------------------------------------------ tensor bases
def network_tensor(pw, mw, n=N, L_kpc=L_KPC, soft_kpc=150.0):
    """Sum_a w_a (nhat_a nhat_a^T - I/3) / Sum_a w_a, w_a = M_a / (d_a^2 + e^2).

    The directionless inverse-square well strength of the standing brief's
    variable 3, promoted to a tensor by keeping the direction to each well.
    Opposing wells do NOT cancel in the traceless part, which is the property
    that distinguishes this from a vector external field.
    """
    h, X, Y, Z = grids(n, L_kpc)
    e2 = (soft_kpc * KPC) ** 2
    acc = [xp.zeros((n, n, n)) for _ in range(6)]
    norm = xp.zeros((n, n, n))
    for a in range(pw.shape[0]):
        dx = X - pw[a, 0]
        dy = Y - pw[a, 1]
        dz = Z - pw[a, 2]
        d2 = dx * dx + dy * dy + dz * dz + e2
        w = mw[a] / d2
        nx, ny, nz = dx / xp.sqrt(d2), dy / xp.sqrt(d2), dz / xp.sqrt(d2)
        acc[0] += w * (nx * nx - 1.0 / 3.0)
        acc[1] += w * (ny * ny - 1.0 / 3.0)
        acc[2] += w * (nz * nz - 1.0 / 3.0)
        acc[3] += w * nx * ny
        acc[4] += w * nx * nz
        acc[5] += w * ny * nz
        norm += w
    norm = norm + 1e-300
    return tuple(a / norm for a in acc)


def dd_tensor(dhat, n=N):
    d = np.asarray(dhat, float)
    d = d / np.linalg.norm(d)
    one = xp.ones((n, n, n))
    return (one * (d[0] ** 2 - 1 / 3), one * (d[1] ** 2 - 1 / 3),
            one * (d[2] ** 2 - 1 / 3), one * d[0] * d[1],
            one * d[0] * d[2], one * d[1] * d[2])


# --------------------------------------------------------------- field maths
class Fields:
    """Newtonian field and invariants for one mass distribution."""

    def __init__(self, rho, h, P, L_nl_kpc=300.0):
        self.rho, self.h, self.P = rho, h, P
        self.Phi_N = P.solve(4.0 * math.pi * G * rho)
        gx, gy, gz = grad(self.Phi_N, h)
        self.gvec = (gx, gy, gz)
        self.gmag = xp.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-30
        T = {}
        for i, gi in enumerate((gx, gy, gz)):
            for j in range(i, 3):
                T[(i, j)] = xp.gradient(gi, h, axis=j)
        for (i, j) in list(T):
            if i != j:
                T[(j, i)] = T[(i, j)]
        self.T = T
        t2 = sum(T[(i, j)] ** 2 for i in range(3) for j in range(3))
        self.T0 = float(xp.sqrt(xp.mean(t2))) + 1e-300
        nn = rho.shape[0]
        kk = 2.0 * math.pi * xp.fft.fftfreq(nn, d=h)
        k2 = (kk[:, None, None] ** 2 + kk[None, :, None] ** 2
              + kk[None, None, :] ** 2)
        Lm = L_nl_kpc * KPC
        qf = xp.fft.ifftn(xp.fft.fftn(rho) / (1.0 + (Lm ** 2) * k2)).real
        self.inv = {
            "x_g": self.gmag / A0,
            "x_Phi": xp.abs(self.Phi_N) / (float(xp.max(xp.abs(self.Phi_N)))
                                           + 1e-300),
            "x_rho": rho / (float(xp.mean(rho)) + 1e-300),
            "x_T": xp.sqrt(t2) / self.T0,
            "q_L": qf / (float(xp.mean(xp.abs(qf))) + 1e-300)}
        nrm = xp.sqrt(t2) + 1e-300
        tr = T[(0, 0)] + T[(1, 1)] + T[(2, 2)]
        self.That = ((T[(0, 0)] - tr / 3) / nrm, (T[(1, 1)] - tr / 3) / nrm,
                     (T[(2, 2)] - tr / 3) / nrm, T[(0, 1)] / nrm,
                     T[(0, 2)] / nrm, T[(1, 2)] / nrm)
        self.nu = nu_rar(self.gmag / A0)

    def apply(self, comp=None, weight=None):
        """Psi solving lap Psi = div[ w nu (I + comp) grad Phi_N ]."""
        gx, gy, gz = self.gvec
        w = self.nu if weight is None else self.nu * weight
        if comp is None:
            vx, vy, vz = w * gx, w * gy, w * gz
        else:
            axx, ayy, azz, axy, axz, ayz = comp
            vx = w * ((1.0 + axx) * gx + axy * gy + axz * gz)
            vy = w * (axy * gx + (1.0 + ayy) * gy + ayz * gz)
            vz = w * (axz * gx + ayz * gy + (1.0 + azz) * gz)
        return self.P.solve(div(vx, vy, vz, self.h))

    def aniso_frac(self, comp, mask=None):
        """max fractional acceleration anisotropy produced by K = I + comp.

        For a symmetric perturbation the acceleration along the two extreme
        eigen-directions differs by (l_max - l_min); reported relative to the
        mean so it is directly comparable with a predicted dg/g.  Evaluated
        ONLY inside the shell the observable actually covers -- a maximum taken
        over the whole box would be dominated by the far outskirts, where the
        weight functions are largest and nothing is measured.
        """
        axx, ayy, azz, axy, axz, ayz = comp
        if mask is not None:
            axx, ayy, azz = axx * mask, ayy * mask, azz * mask
            axy, axz, ayz = axy * mask, axz * mask, ayz * mask
        # eigenvalue spread of a symmetric 3x3, via the traceless invariant
        tr = (axx + ayy + azz) / 3.0
        a, b, c = axx - tr, ayy - tr, azz - tr
        j2 = 0.5 * (a * a + b * b + c * c) + axy ** 2 + axz ** 2 + ayz ** 2
        # for a traceless symmetric 3x3, l_max - l_min <= 2 sqrt(j2)
        spread = 2.0 * xp.sqrt(xp.maximum(j2, 0.0))
        return float(xp.max(spread / (1.0 + tr)))


# ------------------------------------------------------------- the observable
class Obs:
    """Midplane radial acceleration plus BOTH components of projected shear.

    gamma_1 = (d_xx - d_yy) S / 2,  gamma_2 = d_xy S,  with S = int Psi dz the
    projected potential.  Keeping gamma_1 and gamma_2 separately rather than
    |alpha| or an azimuthal average is the whole point of this lane: the phase
    of the quadrupole is where the axis provenance lives.
    """

    def __init__(self, n=N, h=None, L_kpc=L_KPC):
        h = (L_kpc * KPC / n) if h is None else h
        self.n, self.h = n, h
        ax = (xp.arange(n) - n / 2 + 0.5) * h
        X = ax[:, None] * xp.ones((1, n))
        Y = xp.ones((n, 1)) * ax[None, :]
        R = xp.sqrt(X ** 2 + Y ** 2)
        self.X, self.Y, self.R = X, Y, R
        self.m = (R > R_IN_KPC * KPC) & (R < R_OUT_KPC * KPC)
        self.npt = int(self.m.sum())
        self.phi = xp.arctan2(Y, X)

    def of(self, Psi):
        n, h = self.n, self.h
        a = xp.gradient(Psi, h, axis=0)[:, :, n // 2]
        b = xp.gradient(Psi, h, axis=1)[:, :, n // 2]
        v = ((self.X * a + self.Y * b) / (self.R + 1e-30))[self.m]
        S = Psi.sum(axis=2) * h
        sx = xp.gradient(S, h, axis=0)
        sy = xp.gradient(S, h, axis=1)
        sxx = xp.gradient(sx, h, axis=0)
        syy = xp.gradient(sy, h, axis=1)
        sxy = xp.gradient(sx, h, axis=1)
        g1 = (0.5 * (sxx - syy))[self.m]
        g2 = (sxy)[self.m]
        return v, g1, g2


# ------------------------------------------------------------------ the bank
class ProvenanceBank:
    """Atoms tagged by AXIS PROVENANCE, so each hypothesis can be tested alone."""

    def __init__(self, q, tilt, dhat, well_tilt, nw=60, verbose=True):
        t0 = time.time()
        self.q, self.tilt = q, tilt
        rho, h = make_rho(q=q, tilt=tilt)
        self.axis_ratio = sphericity(rho, h)
        self.h = h
        self.P = Poisson(N, h)
        self.F = Fields(rho, h, self.P)
        self.O = Obs(h=h)
        self.B = {"I": (xp.ones((N, N, N)), xp.ones((N, N, N)),
                        xp.ones((N, N, N)), xp.zeros((N, N, N)),
                        xp.zeros((N, N, N)), xp.zeros((N, N, N))),
                  "That": self.F.That}
        for name, d in DDIRS.items():
            self.B[name] = dd_tensor(d)
        for name, wt in NNTILTS.items():
            pw, mw = wells(nw, q=q, tilt=wt, seed=7)
            self.B[name] = network_tensor(pw, mw)
        self.pw, self.mw = wells(nw, q=q, tilt=0.0, seed=7)
        # every non-identity basis is made FLUX-ORTHOGONAL, so a tensor atom
        # can only turn the flux, never rescale it -- see flux_orthogonal
        gx, gy, gz = self.F.gvec
        gm = self.F.gmag
        uhat = (gx / gm, gy / gm, gz / gm)
        self.uhat = uhat
        for name in list(self.B):
            if name != "I":
                self.B[name] = flux_orthogonal(self.B[name], uhat)
        h3, X3, Y3, Z3 = grids(N, L_KPC)
        r3 = xp.sqrt(X3 ** 2 + Y3 ** 2 + Z3 ** 2)
        self.shell = ((r3 > R_IN_KPC * KPC) & (r3 < R_OUT_KPC * KPC)).astype(
            xp.float64)
        self.Psi0 = self.F.apply(None)
        v, g1, g2 = self.O.of(self.Psi0)
        self.s = (float(xp.std(v)) + 1e-300, float(xp.std(g1)) + 1e-300,
                  float(xp.std(g2)) + 1e-300)
        self.y0 = self.pack(v, g1, g2)
        atoms, meta = [], []
        for iname, ival in self.F.inv.items():
            for sname, f in SHAPES:
                for sc in SCALES:
                    w = f(ival / sc)
                    w = w - xp.mean(w)
                    sd = float(xp.std(w))
                    if not np.isfinite(sd) or sd < 1e-12:
                        continue
                    w = w / sd
                    for bname, comp in self.B.items():
                        if bname == "I":
                            R = self.F.apply(None, weight=w)
                        else:
                            R = self.F.apply(tuple(w * ci for ci in comp))
                        if not bool(xp.isfinite(R).all()):
                            continue
                        vv, aa, bb = self.O.of(R - self.Psi0)
                        atoms.append(self.pack(vv, aa, bb))
                        meta.append(f"{sname}({iname}/{sc:g}) x {bname}")
        Aall = xp.stack(atoms)
        self.meta_all = meta
        keep = self._dedup(Aall, meta)
        self.A = Aall[xp.asarray(np.array(keep, dtype=np.int64))]
        self.meta = [meta[i] for i in keep]
        self.basis = [m.split(" x ")[-1] for m in self.meta]
        self.cols = {a: [i for i, b in enumerate(self.basis)
                         if b in BASIS_OF[a]] for a in ARMS}
        self.cols["scalar"] = [i for i, b in enumerate(self.basis) if b == "I"]
        if verbose:
            print(f"      bank q={q:.2f} axis ratio {self.axis_ratio:.3f}: "
                  f"{len(meta)} generated -> {len(self.meta)} independent "
                  f"(scalar {len(self.cols['scalar'])}, "
                  + ", ".join(f"{p} {len(self.cols[p])}" for p in PROVENANCE)
                  + f"), {time.time()-t0:.1f}s")

    def pack(self, v, g1, g2):
        return xp.concatenate([v / self.s[0], g1 / self.s[1], g2 / self.s[2]])

    @staticmethod
    def _dedup(A, meta, tol=1e-8):
        Aa = A.reshape(A.shape[0], -1)
        nrm = xp.sqrt((Aa ** 2).sum(axis=1)) + 1e-300
        C = to_np((Aa / nrm[:, None]) @ (Aa / nrm[:, None]).T)
        n = C.shape[0]
        seen = -np.ones(n, dtype=int)
        keep = []
        for i in range(n):
            if seen[i] >= 0:
                continue
            for j in range(i, n):
                if seen[j] < 0 and abs(abs(C[i, j]) - 1.0) < tol:
                    seen[j] = len(keep)
            keep.append(i)
        return keep


# --------------------------------------------------------------- the detector
class Arm:
    """A cross-fitted exhaustive k = 2 search over one column subset."""

    def __init__(self, A, folds):
        self.folds = folds
        self.S = [Search(A[:, f]) for f in folds]
        self.A = A
        n = A.shape[0]
        jj, kk = np.triu_indices(n, k=1)
        self.pairs = (jj, kk)
        self.idx = xp.asarray(np.stack([jj, kk], 1).astype(np.int32))

    def oof_sse(self, y):
        """Out-of-fold sum of squares: fit on one fold, evaluate on the other."""
        tot, npt = 0.0, 0
        for f in (0, 1):
            S = self.S[f]
            S.set_target(y[self.folds[f]])
            Gs = S.G[self.idx[:, :, None], self.idx[:, None, :]]
            vs = S.v[self.idx]
            c, ok = chol_solve(Gs, vs, S.ridge * S.npt)
            sse = (S.yy - 2.0 * (c * vs).sum(1)
                   + ((c[:, None, :] @ Gs)[:, 0] * c).sum(1))
            bad = (~ok) | (~xp.isfinite(sse)) | (sse > S.yy * 1.0000001)
            sse = xp.where(bad, xp.float64(1e30), sse)
            i = int(xp.argmin(sse))
            ii, jj = int(self.pairs[0][i]), int(self.pairs[1][i])
            ci = c[i]
            g = 1 - f
            yr = y[self.folds[g]] - (ci[0] * self.A[ii][self.folds[g]]
                                     + ci[1] * self.A[jj][self.folds[g]])
            tot += float(yr @ yr)
            npt += yr.size
        return tot, npt


class Detector:
    """Cross-fitted statistic D_p for each axis provenance p, on one bank."""

    def __init__(self, bank, seed=0):
        self.bank = bank
        npt = bank.A.shape[1]
        rng = np.random.default_rng(seed)
        pick = rng.random(npt) < 0.5
        f0 = xp.asarray(np.where(pick)[0])
        f1 = xp.asarray(np.where(~pick)[0])
        self.folds = (f0, f1)
        sc = bank.cols["scalar"]
        self.arms = {"scalar": Arm(bank.A[xp.asarray(np.array(sc, np.int64))],
                                   self.folds)}
        for a in ARMS:
            cols = sc + bank.cols[a]
            self.arms[a] = Arm(bank.A[xp.asarray(np.array(cols, np.int64))],
                               self.folds)

    def stat(self, y, arms=None):
        """D_a = RMS_oof(scalar only) - RMS_oof(scalar + arm a)."""
        tgt = y - self.bank.y0
        s0, n0 = self.arms["scalar"].oof_sse(tgt)
        r0 = math.sqrt(s0 / n0)
        out = {}
        for a in (ARMS if arms is None else arms):
            s, n = self.arms[a].oof_sse(tgt)
            out[a] = r0 - math.sqrt(s / n)
        return out


# ------------------------------------------------------- Stage-0 null and signal
SCALAR_FAMILIES = ("nu_simple", "phi_gate", "rho_gate", "tidal_gate", "nonlocal")


def scalar_weight(F, rng):
    """A smooth scalar response NOT in the atom bank's grammar.

    Five qualitatively different families; the null must not be a slightly
    misspecified member of the same parametric family the detector searches.
    """
    fam = SCALAR_FAMILIES[rng.integers(len(SCALAR_FAMILIES))]
    x = F.gmag / A0
    if fam == "nu_simple":
        a = rng.uniform(0.6, 1.8)
        w = (0.5 * (1.0 + xp.sqrt(1.0 + 4.0 / xp.maximum(x, 1e-30) ** a))
             ** (1.0 / a)) / F.nu
    elif fam == "phi_gate":
        b = rng.uniform(0.05, 0.30)
        p = xp.abs(F.Phi_N) / (float(xp.max(xp.abs(F.Phi_N))) + 1e-300)
        w = 10.0 ** (b * xp.log10(xp.maximum(p, 1e-6)))
    elif fam == "rho_gate":
        a, m = rng.uniform(0.1, 0.8), rng.uniform(0.5, 2.5)
        u = F.inv["x_rho"] / rng.uniform(0.2, 3.0)
        w = 1.0 + a / (1.0 + u ** m)
    elif fam == "tidal_gate":
        a, m = rng.uniform(0.1, 0.8), rng.uniform(0.5, 2.0)
        u = F.inv["x_T"] / rng.uniform(0.2, 3.0)
        w = 1.0 + a * u ** m / (1.0 + u ** m)
    else:
        a = rng.uniform(0.1, 0.7)
        w = 1.0 + a * xp.tanh(F.inv["q_L"] / rng.uniform(0.3, 3.0))
    w = xp.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    return xp.clip(w, 0.2, 5.0), fam


def draw(bank, rng, kind, amp, noise, tilt=0.0):
    """One realisation: perturbed baryons + a scalar law (+ optional tensor).

    Returns (observable vector, diagnostics).  `kind` is 'null' or one of the
    three provenances.  The tensor injection uses shape-function scales
    BETWEEN the bank's grid points, so the truth is never exactly in the basis.
    """
    q_true = float(np.clip(bank.q + rng.normal(0, 0.05), 0.30, 0.999))
    t_true = bank.tilt + rng.normal(0, math.radians(12.0))
    depth = float(np.exp(rng.normal(0, 0.10)))
    off = rng.normal(0, 0.02 * L_KPC, 3)
    gml = rng.normal(0, 0.15)
    rho, h = make_rho(q=q_true, tilt=t_true, depth=depth, dx=tuple(off),
                      grad_ml=gml)
    F = Fields(rho, h, bank.P)
    w_sc, fam = scalar_weight(F, rng)
    Psi = F.apply(None, weight=w_sc)
    v, g1, g2 = bank.O.of(Psi)
    y = bank.pack(v, g1, g2)
    afrac = 0.0
    turn = 0.0
    if kind != "null" and amp > 0:
        if kind == "external":
            d = (math.cos(tilt), math.sin(tilt), 0.0)
            comp0 = dd_tensor(d)
        elif kind == "network":
            pw, mw = wells(60, q=q_true, tilt=tilt + t_true, seed=int(
                rng.integers(1 << 30)))
            comp0 = network_tensor(pw, mw)
        else:                                   # source axis
            comp0 = F.That
        acc = None
        for sgn, sc in ((1.0, INJ_SCALES[0]), (-0.6, INJ_SCALES[1])):
            iname = list(F.inv)[rng.integers(len(F.inv))]
            sname, f = SHAPES[rng.integers(len(SHAPES))]
            ww = f(F.inv[iname] / sc)
            ww = ww - xp.mean(ww)
            sd = float(xp.std(ww))
            if sd < 1e-12:
                continue
            ww = sgn * ww / sd
            c = tuple(ww * ci for ci in comp0)
            acc = c if acc is None else tuple(a + b for a, b in zip(acc, c))
        if acc is not None:
            # what FRACTION of this response actually turns the flux?  The
            # rest is a scalar rescaling that the blindness theorem says is
            # unobservable as anisotropy, and it is the reason a source-axis
            # tensor is intrinsically hard to see even on a triaxial source.
            gx0, gy0, gz0 = F.gvec
            gm0 = F.gmag
            axx, ayy, azz, axy, axz, ayz = acc
            vx0 = axx * gx0 + axy * gy0 + axz * gz0
            vy0 = axy * gx0 + ayy * gy0 + ayz * gz0
            vz0 = axz * gx0 + ayz * gy0 + azz * gz0
            rad = (vx0 * gx0 + vy0 * gy0 + vz0 * gz0) / gm0 ** 2
            tx, ty, tz = (vx0 - rad * gx0, vy0 - rad * gy0, vz0 - rad * gz0)
            sh = bank.shell
            num = float(xp.sum(sh * (tx ** 2 + ty ** 2 + tz ** 2)))
            den = float(xp.sum(sh * (vx0 ** 2 + vy0 ** 2 + vz0 ** 2))) + 1e-300
            turn = math.sqrt(num / den)
            Psi_t = F.apply(acc, weight=w_sc)
            vv, aa, bb = bank.O.of(Psi_t - Psi)
            o = bank.pack(vv, aa, bb)
            sd = float(xp.std(o)) + 1e-300
            k = amp * float(xp.std(y)) / sd
            y = y + k * o
            afrac = F.aniso_frac(tuple(k * a for a in acc), bank.shell)
    # shear-calibration error: multiplicative m and additive c on the shear
    ncomp = bank.O.npt
    mcal = 1.0 + rng.normal(0, 0.02)
    y = xp.concatenate([y[:ncomp], mcal * y[ncomp:2 * ncomp],
                        mcal * y[2 * ncomp:]])
    cadd = rng.normal(0, 0.01, 2)
    y = y + xp.concatenate([xp.zeros(ncomp),
                            xp.full(ncomp, float(cadd[0])),
                            xp.full(ncomp, float(cadd[1]))])
    y = y + xp.asarray(rng.normal(0, noise * float(xp.std(y)), y.size))
    return y, dict(family=fam, q_true=q_true, aniso_frac=afrac,
                   turn_fraction=turn)


# ----------------------------------------------------------------------- main
def main():
    print("=" * 78)
    print("AXIS POWER -- three separated power surfaces, one per provenance")
    print("=" * 78)
    NCAL = int(os.environ.get("NCAL", 160))
    NAUD = int(os.environ.get("NAUD", 160))
    NSIG = int(os.environ.get("NSIG", 20))
    QS = [float(x) for x in os.environ.get("QS", "0.50,0.75,0.90,0.98")
          .split(",")]
    NOISE = [float(x) for x in os.environ.get("NOISE", "0.02,0.10,0.30")
             .split(",")]
    AMPS = [float(x) for x in os.environ.get("AMPS", "0.05,0.15,0.35")
            .split(",")]
    TILTS = [0.0, math.pi / 4, math.pi / 2]
    print(f"\n   grid: q {QS}, noise {NOISE}, amp {AMPS}, "
          f"tilt {[round(math.degrees(t)) for t in TILTS]} deg")
    print(f"   sims: {NCAL} calibration + {NAUD} AUDIT + {NSIG} per injection "
          f"cell, all disjoint")
    OUT = os.environ.get("OUT", "axis_power.json")
    rng = np.random.default_rng(int(os.environ.get("SEED", 20260904)))
    t0 = time.time()
    out = {"config": dict(n=N, L_kpc=L_KPC, rs_kpc=RS_KPC,
                          M_msun=MTOT / MSUN, r_in_kpc=R_IN_KPC,
                          r_out_kpc=R_OUT_KPC, n_cal=NCAL, n_aud=NAUD,
                          n_sig=NSIG, qs=QS, noise=NOISE, amps=AMPS,
                          tilts_deg=[math.degrees(t) for t in TILTS],
                          scalar_families=list(SCALAR_FAMILIES),
                          bank_scales=list(SCALES),
                          injection_scales=list(INJ_SCALES)),
           "rows": [], "coarse_grain": {}, "monotonicity": {}}

    # Checkpoint after every (geometry, noise) row.  The first attempt at this
    # grid died silently after eight of fifteen rows and lost all of them; a
    # long GPU run that only writes at the end is a run that can be lost.
    ck = os.path.join(HERE, OUT)
    if os.path.exists(ck):
        try:
            prev = json.load(open(ck, encoding="utf-8"))
            if prev.get("config", {}).get("qs") == QS:
                out["rows"] = prev.get("rows", [])
                out["coarse_grain"] = prev.get("coarse_grain", {})
                print(f"   resuming from {OUT}: {len(out['rows'])} rows done")
        except Exception:                                       # noqa: BLE001
            pass
    done = {(round(r["q"], 4), round(r["noise"], 4)) for r in out["rows"]}

    for q in QS:
        if all((round(q, 4), round(nz, 4)) in done for nz in NOISE):
            continue
        bank = ProvenanceBank(q=q, tilt=0.0, dhat=(1.0, 0.0, 0.0),
                              well_tilt=0.0)
        det = Detector(bank, seed=11)
        for noise in NOISE:
            if (round(q, 4), round(noise, 4)) in done:
                continue
            # ---- calibration sims: set D* for EVERY arm.  The three
            #      provenances are tested simultaneously, so a family-wise
            #      critical value is carried alongside the per-arm one.
            cal = {a: [] for a in ARMS}
            for _ in range(NCAL):
                y, _ = draw(bank, rng, "null", 0.0, noise)
                d = det.stat(y)
                for a in ARMS:
                    cal[a].append(d[a])
            Dstar = {a: float(np.percentile(cal[a], 95)) for a in ARMS}
            # A cross-fitted D can be negative, so the null is not a point mass
            # at zero; but if it nearly is, D* degenerates and the detector is
            # not correctly sized.  That is measured, not assumed away.
            nullpos = {a: float(np.mean(np.array(cal[a]) > 1e-12))
                       for a in ARMS}
            Dstar_fw = {a: float(np.percentile(cal[a], 100 * (1 - 0.05 / 3)))
                        for a in ARMS}
            # ---- audit sims: UNTOUCHED, verify the realised size at D*
            aud = {a: 0 for a in ARMS}
            audfw = {a: 0 for a in ARMS}
            for _ in range(NAUD):
                y, _ = draw(bank, rng, "null", 0.0, noise)
                d = det.stat(y)
                for a in ARMS:
                    aud[a] += int(d[a] > Dstar[a])
                    audfw[a] += int(d[a] > Dstar_fw[a])
            row = dict(q=q, axis_ratio=bank.axis_ratio, noise=noise,
                       D_star=Dstar, D_star_familywise=Dstar_fw,
                       null_median={a: float(np.median(cal[a])) for a in ARMS},
                       null_frac_positive=nullpos,
                       audit_fpr={a: aud[a] / NAUD for a in ARMS},
                       audit_fpr_familywise={a: audfw[a] / NAUD for a in ARMS},
                       power={})
            print(f"\n   q = {q:.2f} (axis ratio {bank.axis_ratio:.3f}), "
                  f"noise {noise:.0%}")
            for a in ARMS:
                print(f"      {a:18s} D* {Dstar[a]:.4e}   audit FPR "
                      f"{aud[a]/NAUD:.3f}  (family-wise {audfw[a]/NAUD:.3f})"
                      f"   null frac > 0: {nullpos[a]:.2f}")
            # ---- injection sims
            for p in PROVENANCE:
                # tilt is the angle between the injected preferred axis and the
                # source's major axis.  For the SOURCE provenance those are the
                # same axis by definition, so that surface has one dimension
                # fewer and is run at tilt = 0 only.
                tl = TILTS if p != "source" else [0.0]
                for tilt in tl:
                    td = round(math.degrees(tilt))
                    known = f"{p}_known{td}" if p != "source" else None
                    for amp in AMPS:
                        hits = 0
                        khits = 0
                        afr, tfr = [], []
                        for _ in range(NSIG):
                            y, dg = draw(bank, rng, p, amp, noise, tilt=tilt)
                            d = det.stat(y)
                            hits += int(d[p] > Dstar[p])
                            if known:
                                khits += int(d[known] > Dstar[known])
                            afr.append(dg["aniso_frac"])
                            tfr.append(dg["turn_fraction"])
                        key = f"{p}|tilt{td}|amp{amp}"
                        row["power"][key] = dict(
                            provenance=p, tilt_deg=math.degrees(tilt),
                            amp=amp, power=hits / NSIG,
                            power_axis_known=(khits / NSIG if known else None),
                            aniso_frac_median=float(np.median(afr)),
                            turn_fraction_median=float(np.median(tfr)))
                    ps = [row["power"][f"{p}|tilt{td}|amp{a}"]["power"]
                          for a in AMPS]
                    pk = [row["power"][f"{p}|tilt{td}|amp{a}"]
                          ["power_axis_known"] for a in AMPS]
                    fr = [round(row["power"][f"{p}|tilt{td}|amp{a}"]
                                ["aniso_frac_median"], 4) for a in AMPS]
                    print(f"      {p:9s} tilt {td:3d} deg   "
                          f"power (axis searched) {ps}")
                    if known:
                        print(f"                            "
                              f"power (axis KNOWN)    {pk}")
                    tf = [round(row["power"][f"{p}|tilt{td}|amp{a}"]
                                ["turn_fraction_median"], 4) for a in AMPS]
                    print(f"                            max dg/g over the "
                          f"measured shell: {fr}")
                    print(f"                            flux-TURNING fraction "
                          f"of the injected response: {tf}")
            out["rows"].append(row)
            with open(ck, "w") as f:
                json.dump(out, f, indent=1)
        # ---- coarse-graining gate, once per geometry
        if q == QS[0] and not out["coarse_grain"]:
            out["coarse_grain"] = coarse_gate(bank)
        # release the previous geometry's device memory before the next bank
        del det, bank
        if GPU:
            xp.get_default_memory_pool().free_all_blocks()
    out["seconds"] = time.time() - t0

    # ---- the headline: does the source-axis surface collapse?
    print("\n" + "=" * 78)
    print("   SPHERICAL-LIMIT CHECK (the theorem is the check, not a hope)")
    for noise in NOISE:
        print(f"\n   noise {noise:.0%}   mean power at amp "
              f"{max(AMPS)}, averaged over tilt")
        print("      axis ratio |  source    external   network")
        for r in out["rows"]:
            if abs(r["noise"] - noise) > 1e-12:
                continue
            vals = []
            for p in PROVENANCE:
                v = [r["power"][k]["power"] for k in r["power"]
                     if r["power"][k]["provenance"] == p
                     and abs(r["power"][k]["amp"] - max(AMPS)) < 1e-12]
                vals.append(float(np.mean(v)))
            print(f"      {r['axis_ratio']:9.3f}  |  " +
                  "   ".join(f"{v:.3f}   " for v in vals))
    out["monotonicity"] = monotonicity(out, AMPS)
    with open(ck, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   elapsed {out['seconds']:.0f}s")
    print(f"   written: {OUT}")


def coarse_gate(bank, ns=(1, 10, 60, 300)):
    """A real coherence scale cannot be set by the cataloguer.

    The identical continuous mass is represented as 1, 10, 60 and 300 wells and
    the resulting network tensor field is compared.  Drift that does not fall
    with N means the network hypothesis is counting catalogue rows, not
    measuring a field.
    """
    print("\n   COARSE-GRAINING GATE on the member-well tensor")
    ref = None
    rows = []
    for nw in ns:
        pw, mw = wells(nw, q=bank.q, tilt=0.0, seed=7)
        comp = flux_orthogonal(network_tensor(pw, mw), bank.uhat)
        Psi = bank.F.apply(comp)
        v, g1, g2 = bank.O.of(Psi - bank.Psi0)
        o = to_np(bank.pack(v, g1, g2))
        if ref is None:
            ref = o
        d = float(np.sqrt(np.mean((o - ref) ** 2))
                  / (np.sqrt(np.mean(ref ** 2)) + 1e-300))
        rows.append(dict(n_wells=nw, drift_vs_1well=d))
        print(f"      {nw:4d} wells   drift vs the 1-well limit = {d:.4f}")
    d300 = rows[-1]["drift_vs_1well"]
    d60 = rows[-2]["drift_vs_1well"]
    conv = abs(d300 - d60) / max(d60, 1e-300)
    print(f"      60 -> 300 wells changes the drift by {conv:.3f}")
    print("      (a drift that keeps growing with N is catalogue-row counting;")
    print("       one that converges is a genuine field with a coherence scale)")
    return dict(rows=rows, relative_change_60_to_300=conv,
                converged=bool(conv < 0.25))


def monotonicity(out, AMPS):
    """dS/dtheta != 0 for the headline statistic, printed as a spread.

    This programme has been bitten by a rank statistic that was bit-identical
    across three decades of the parameter it was meant to measure.  So the
    power surface is checked for actual dependence on each of its axes.
    """
    res = {}
    for p in PROVENANCE:
        by_amp = {a: [] for a in AMPS}
        by_ar = {}
        by_tilt = {}
        for r in out["rows"]:
            for k, c in r["power"].items():
                if c["provenance"] != p:
                    continue
                by_amp[c["amp"]].append(c["power"])
                by_ar.setdefault(round(r["axis_ratio"], 3), []).append(
                    c["power"])
                by_tilt.setdefault(round(c["tilt_deg"]), []).append(c["power"])
        ma = [float(np.mean(by_amp[a])) for a in AMPS]
        mr = {k: float(np.mean(v)) for k, v in sorted(by_ar.items())}
        mt = {k: float(np.mean(v)) for k, v in sorted(by_tilt.items())}
        res[p] = dict(vs_amplitude=dict(zip([str(a) for a in AMPS], ma)),
                      spread_amplitude=max(ma) - min(ma),
                      vs_axis_ratio={str(k): v for k, v in mr.items()},
                      spread_axis_ratio=max(mr.values()) - min(mr.values()),
                      vs_tilt={str(k): v for k, v in mt.items()},
                      spread_tilt=max(mt.values()) - min(mt.values()))
        print(f"\n   monotonicity, {p}:")
        print(f"      vs amplitude   {ma}   spread {max(ma)-min(ma):.3f}")
        print(f"      vs axis ratio  {list(mr.values())}   "
              f"spread {max(mr.values())-min(mr.values()):.3f}")
        print(f"      vs tilt        {list(mt.values())}   "
              f"spread {max(mt.values())-min(mt.values()):.3f}")
    return res


if __name__ == "__main__":
    main()
