"""CHANNEL 4 -- cluster amplitude and radial shape, and the two galaxy screens.

Built on the tensor lane (`../tensor/`), imported and NOT modified: the
synthetic A2029 (real X-COP baryon profile + 300 statistical members), the
closed-form symmetric-3x3 matrix exponential, the GPU field machinery, and
`mechanism.py`'s calibrated spherical reduction.

TWO THINGS THIS LANE REFUSES TO GET WRONG, both of which have already cost the
programme a number:

1. THE SHELL AVERAGE OF A CONDUCTIVITY IS PHYSICS.  k varies by orders of
   magnitude across a shell once |A| is large.  calib.py measured three
   candidate averages against six full nonlinear 3-D solves: worst departure
   arithmetic 46.9%, cell-wise 22.4%, HARMONIC 20.4%, and the arithmetic mean
   is not merely less accurate but qualitatively wrong -- it turns over and
   fakes a saturation at B = 2.1, which is how A_T = -12.8 got reported where
   the truth is -4.7.  The harmonic mean is used here; the arithmetic one is
   computed alongside and stored so the bracket is always visible.

   For the SCALAR competitor a0 -> a0 (1 + A W) there is no conductivity to
   average, so the same calibrated rule is TRANSLATED rather than dropped: in
   the deep-MOND regime the two parametrisations are related exactly by
   k_eq = (1 + A W)^(-2/3), so the harmonic mean of k_eq corresponds to
       (1 + A W)_eff = < (1 + A W)^(2/3) >^(3/2)
   and the arithmetic alternative < 1 + A W > is stored beside it.  The
   cluster sits at g_b/a0 = 0.07-0.18, deep enough for that translation to
   hold to a few per cent, and the galaxy probes -- which are NOT deep MOND --
   are evaluated with the exact mu inversion instead.

2. THE MEMBER GALAXY.  A cluster member sits at |Phi_N| = 1.09e12 m^2/s^2,
   DEEPER than the cluster's own 1 Mpc shell at 7.22e11.  Anything gated on
   potential depth fires hardest inside member galaxies -- precisely where the
   fundamental plane says nothing is happening.  Both galaxy probes are
   evaluated for every candidate and both are hard screens at the declared
   0.040 dex tolerance (the RAR's intrinsic scatter).
"""
from __future__ import annotations

import os
import sys

import numpy as np

TENSOR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
          "work/wellnet-2026-09/tensor")
if TENSOR not in sys.path:
    sys.path.insert(0, TENSOR)

import cluster as CL                                            # noqa: E402
import mechanism as MECH                                        # noqa: E402
import wellnet as W                                             # noqa: E402
from wellnet import KPC, MSUN                                   # noqa: E402

from tw_core import (A0, G, W_of, W_CEIL, mond_invert,           # noqa: E402
                     sym3_from_dir)
from ch_radial import L_NL, M_NL                                # noqa: E402

XP = W.get_xp(True)
RADII = MECH.RADII
GAL_RADII = MECH.GAL_RADII
GAL_TOL_DEX = MECH.GAL_TOL_DEX
BREQ = MECH.BREQ                       # lane-12 measured radial requirement
BFLAT = np.full(len(RADII), 2.0)       # the independent flat X-COP target
SOFT = 1.0 * KPC

#: well-weight settings for the tensor_S structure, taken unchanged from the
#: tensor lane's own surviving corner rather than re-scanned here.
WELL_SETTINGS = [
    dict(tag="plaw_p1q2s2_L1000_x", family="plaw", p=1.0, q=2.0, s=2.0,
         L=1000.0 * KPC, exclude_nearest=True),
    dict(tag="plaw_p1q1s2_L300_x", family="plaw", p=1.0, q=1.0, s=2.0,
         L=300.0 * KPC, exclude_nearest=True),
    dict(tag="plaw_p0q1s2_L300", family="plaw", p=0.0, q=1.0, s=2.0,
         L=300.0 * KPC, exclude_nearest=False),
    dict(tag="expo_p1q2_L1000_x", family="expo", p=1.0, q=2.0, s=1.0,
         L=1000.0 * KPC, exclude_nearest=True),
]


# --------------------------------------------------------------- Newtonian
def newton_full(points, c, xp=np, chunk=1 << 14, wells=None, gas=True):
    """Phi_N, g_N vector, Hessian(Phi_N) and rho_b at arbitrary points.

    Smooth gas: exact for the spherical profile.  Wells: Plummer-softened
    point masses at a single GLOBAL softening of 1 kpc, used consistently for
    Phi, g, the Hessian and rho, so the Poisson identity trace H = 4 pi G rho
    holds to round-off rather than approximately.

    gas=False and an explicit `wells` list is the ISOLATED FIELD GALAXY: it
    does not live in the cluster, so it must not see the cluster's gas.  Run
    AB's contexts() makes exactly the same distinction (use_cluster_phi).
    """
    iso = xp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    Rr = xp.sqrt(xp.sum(points ** 2, axis=1))
    nhat = points / xp.maximum(Rr, 1e-30)[:, None]
    if gas:
        rg = xp.asarray(c["r_gas"])
        Mg = xp.asarray(c["M_gas"])
        dM = xp.asarray(c["dM_gas"])
        tailrev = xp.cumsum((dM / rg)[::-1])[::-1]
        Phi_gas = -G * (Mg / rg + tailrev)
        g_gas = G * Mg / rg ** 2
        rho_gas = xp.gradient(Mg, rg) / (4.0 * np.pi * rg ** 2)
        Rc = xp.clip(Rr, rg[0], rg[-1])
        Phi = xp.interp(Rc, rg, Phi_gas)
        gr = xp.interp(Rc, rg, g_gas)
        rho = xp.interp(Rc, rg, rho_gas)
        gv = -gr[:, None] * nhat
        dgr = xp.interp(Rc, rg, xp.gradient(g_gas, rg))
        nn = xp.stack([nhat[:, 0] ** 2, nhat[:, 1] ** 2, nhat[:, 2] ** 2,
                       nhat[:, 0] * nhat[:, 1], nhat[:, 0] * nhat[:, 2],
                       nhat[:, 1] * nhat[:, 2]], axis=1)
        trm = gr / xp.maximum(Rr, 1e-30)
        H = dgr[:, None] * nn + trm[:, None] * (iso[None, :] - nn)
    else:
        Phi = xp.zeros(points.shape[0])
        rho = xp.zeros(points.shape[0])
        gv = xp.zeros_like(points)
        H = xp.zeros((points.shape[0], 6))
    if wells is None:
        wx, wm = xp.asarray(c["pos"]), xp.asarray(c["Mg"])
    else:
        wx, wm = xp.asarray(wells[0]), xp.asarray(wells[1])
    P = points.shape[0]
    Mnl = xp.zeros(P)
    for i0 in range(0, P, chunk):
        i1 = min(P, i0 + chunk)
        d = points[i0:i1, None, :] - wx[None, :, :]
        d2 = xp.sum(d * d, axis=-1)
        q = xp.sqrt(d2 + SOFT ** 2)
        Phi[i0:i1] -= xp.sum(G * wm[None, :] / q, axis=1)
        gv[i0:i1] -= xp.sum(G * wm[None, :, None] * d / q[:, :, None] ** 3,
                            axis=1)
        rho[i0:i1] += xp.sum(3.0 * wm[None, :] * SOFT ** 2
                             / (4.0 * np.pi * q ** 5), axis=1)
        Mnl[i0:i1] += xp.sum(xp.where(d2 < L_NL ** 2, wm[None, :], 0.0), axis=1)
        c3 = G * wm[None, :] / q ** 3
        e = d / q[:, :, None]
        ee = xp.stack([e[..., 0] ** 2, e[..., 1] ** 2, e[..., 2] ** 2,
                       e[..., 0] * e[..., 1], e[..., 0] * e[..., 2],
                       e[..., 1] * e[..., 2]], axis=-1)
        H[i0:i1] += xp.sum(c3[:, :, None] * (iso[None, None, :]
                                             - 3.0 * ee), axis=1)
        del d, d2, q, c3, e, ee
    # smooth gas inside L_NL of x, local-density approximation
    Mnl = Mnl + rho * (4.0 / 3.0) * np.pi * L_NL ** 3
    gmag = xp.sqrt(xp.sum(gv * gv, axis=1))
    return dict(Phi=Phi, gv=gv, gmag=gmag, H=H, rho=rho, Mnl=Mnl, rhat=nhat)


def _traceless_norm_and_hat(H, xp=np):
    """Traceless part of a sym3 field: Frobenius norm, and That = D/|D|_2."""
    tr3 = (H[:, 0] + H[:, 1] + H[:, 2]) / 3.0
    D = H.copy()
    D[:, 0] -= tr3
    D[:, 1] -= tr3
    D[:, 2] -= tr3
    fro = xp.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2 + D[:, 2] ** 2
                  + 2.0 * (D[:, 3] ** 2 + D[:, 4] ** 2 + D[:, 5] ** 2))
    lam = W.sym3_eigvals(D, xp)
    spec = xp.maximum(xp.max(xp.abs(lam), axis=-1), 1e-300)
    return fro, D / spec[:, None]


class ClusterBench:
    """Cluster shells, an isolated field galaxy, and a cluster MEMBER galaxy."""

    def __init__(self, n=64, seed=20260903, ndir=1500, verbose=False):
        clu, fld, mem = MECH.contexts(n=n, seed=seed, ndir=ndir)
        self.clu, self.fld, self.mem = clu, fld, mem
        self.c = clu["c"]
        self.probes = {}
        self._register("cluster", clu["sub_pts"], clu["sub_rhat"],
                       clu["sub_shells"], RADII)
        self._register("field", fld["pts"], fld["rhat"], fld["masks"],
                       GAL_RADII, r=fld["r"], Menc=fld["Menc"],
                       wells=(fld["wx"], fld["wm"]), gas=False)
        self._register("member", mem["pts"], mem["rhat"], mem["masks"],
                       GAL_RADII, r=mem["r"], Menc=mem["Menc"])
        self.r_prof = clu["r_prof"]
        self.M_prof = clu["M_prof"]
        self._S_cache = {}

    def _register(self, tag, pts, rhat, masks, radii, r=None, Menc=None,
                  wells=None, gas=True):
        N = newton_full(pts, self.c, xp=XP, wells=wells, gas=gas)
        fro, That = _traceless_norm_and_hat(N["H"], XP)
        ghat = N["gv"] / XP.maximum(N["gmag"], 1e-300)[:, None]
        dd = XP.stack([ghat[:, 0] ** 2 - 1 / 3, ghat[:, 1] ** 2 - 1 / 3,
                       ghat[:, 2] ** 2 - 1 / 3, ghat[:, 0] * ghat[:, 1],
                       ghat[:, 0] * ghat[:, 2], ghat[:, 1] * ghat[:, 2]],
                      axis=1)
        inv = dict(one=XP.ones_like(N["gmag"]), gn=N["gmag"] / A0,
                   phi=XP.abs(N["Phi"]), rhobar=XP.maximum(N["rho"], 1e-40),
                   tidal=XP.maximum(fro, 1e-45),
                   qbar=N["Mnl"] / (N["Mnl"] + M_NL))
        self.probes[tag] = dict(pts=pts, rhat=rhat, masks=masks, radii=radii,
                                inv=inv, That=That, dd=dd, r=r, Menc=Menc,
                                gmag=N["gmag"], Phi=N["Phi"])

    # ------------------------------------------------------- well-network S
    def S_of(self, tag, ws):
        key = (tag, ws["tag"])
        if key not in self._S_cache:
            p = self.probes[tag]
            wx = XP.asarray(self.c["pos"])
            wm = XP.asarray(self.c["Mg"])
            if tag == "field":
                wx = XP.asarray(self.fld["wx"])
                wm = XP.asarray(self.fld["wm"])
            self._S_cache[key] = W.S_tensor(
                p["pts"], wx, wm, family=ws["family"], p=ws["p"], q=ws["q"],
                s=ws["s"], L=ws["L"], xp=XP,
                exclude_nearest=ws["exclude_nearest"])
        return self._S_cache[key]

    # ------------------------------------------------------------ the base
    def base_tensor(self, tag, cand):
        """W(I(x)) * Struct(x) as a (P,6) sym3 field, on the GPU."""
        p = self.probes[tag]
        if cand.form == "off" or cand.inv == "one":
            # NO response.  W = 0, not 1: a base-only row must reduce to the
            # plain MOND law with B = 1 everywhere, otherwise the amplitude fit
            # turns it into a global rescaling of a0 that the radial channel
            # (which uses tw_core.W_cand and correctly returns zeros) never
            # sees.  That inconsistency made the three BASE_ rows in the first
            # full run report a member violation of 0.38-0.45 dex for laws that
            # have no environmental response at all.
            Wf = XP.zeros_like(p["inv"]["gn"])
        else:
            I = XP.maximum(p["inv"][cand.inv] / cand.I0, 1e-300)
            if cand.form == "sat":
                u = I ** cand.m
                Wf = u / (1.0 + u)
            elif cand.form == "pow":
                Wf = I ** cand.m
            elif cand.form == "log":
                Wf = XP.log1p(I ** cand.m)
            elif cand.form == "inv":
                Wf = 1.0 / (1.0 + I ** cand.m)
            else:
                raise ValueError(cand.form)
            Wf = XP.minimum(XP.nan_to_num(Wf, nan=0.0, posinf=W_CEIL),
                            W_CEIL)
        if cand.struct == "iso_K":
            St = -XP.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])[None, :] \
                * XP.ones((Wf.shape[0], 1))
        elif cand.struct == "tensor_d":
            St = p["dd"]
        elif cand.struct == "tensor_T":
            St = p["That"]
        elif cand.struct == "tensor_S":
            St = self.S_of(tag, cand.extra["well"])
        elif cand.struct == "scalar_a0":
            St = XP.zeros((Wf.shape[0], 6))     # W only; no tensor is used
        else:
            raise ValueError(cand.struct)
        return Wf[:, None] * St, Wf

    # ----------------------------------------------------------- boosts
    def k_shell(self, tag, cand, amps):
        base, _ = self.base_tensor(tag, cand)
        p = self.probes[tag]
        # the amplitude times the base enters an exp(); clip the exponent at
        # +-60 (a factor 1e26 in k) so a runaway response returns a finite,
        # obviously-absurd number instead of an overflow warning and a nan
        nb = XP.max(XP.abs(base)) * max(abs(float(np.min(amps))),
                                        abs(float(np.max(amps))))
        if float(nb) > 60.0:
            base = base * (60.0 / float(nb))
        return MECH.k_means(base, p["rhat"], amps, p["masks"])

    def B_scalar(self, tag, cand, amps):
        """Exact mu-inversion for the scalar competitor a0 -> a0 (1+A W)."""
        p = self.probes[tag]
        _, Wf = self.base_tensor(tag, cand)
        Wn = W.asnumpy(Wf)
        A = np.asarray(amps, float)
        nm = len(p["masks"])
        Bh = np.empty((len(A), nm))
        Ba = np.empty((len(A), nm))
        for j, msk in enumerate(p["masks"]):
            w = Wn[W.asnumpy(msk)]
            fac = 1.0 + A[:, None] * w[None, :]
            fac = np.maximum(fac, 1e-6)
            f_h = np.mean(fac ** (2.0 / 3.0), axis=1) ** 1.5
            f_a = np.mean(fac, axis=1)
            r, Me = self._probe_rM(tag, j)
            F = G * Me / r ** 2
            g0 = mond_invert(np.array([F]), np.array([1.0]), cand.a0,
                             cand.base)[0]
            for arr, f in ((Bh, f_h), (Ba, f_a)):
                gg = np.array([mond_invert(np.array([F]), np.array([1.0]),
                                           cand.a0 * ff, cand.base)[0]
                               for ff in f])
                arr[:, j] = gg / g0
        return Bh, Ba

    def _probe_rM(self, tag, j):
        if tag == "cluster":
            rk = RADII[j]
            i = min(int(np.searchsorted(self.r_prof, rk * KPC)),
                    len(self.r_prof) - 1)
            return float(self.r_prof[i]), float(self.M_prof[i])
        p = self.probes[tag]
        sl = int(j * len(p["r"]) // len(p["masks"]))
        return float(p["r"][sl]), float(p["Menc"][sl])

    def B_of(self, tag, cand, amps):
        """B(amp, radius) for one probe set: (harmonic, arithmetic)."""
        if cand.struct == "scalar_a0":
            return self.B_scalar(tag, cand, amps)
        kh, ka = self.k_shell(tag, cand, amps)
        out = []
        for km in (kh, ka):
            B = np.empty_like(km)
            for j in range(km.shape[1]):
                r, Me = self._probe_rM(tag, j)
                F = G * Me / r ** 2
                kk = np.maximum(km[:, j], 1e-12)
                B[:, j] = (mond_invert(np.full_like(kk, F), kk, cand.a0,
                                       cand.base)
                           / mond_invert(np.array([F]), np.array([1.0]),
                                         cand.a0, cand.base)[0])
            out.append(B)
        return out[0], out[1]


def evaluate(bench, cand, amps, target="lane12"):
    """Fit the amplitude on the cluster, then report every cluster-side number.

    The amplitude A is the candidate's SECOND and last global constant, and it
    is fitted here -- galaxies cannot see it, because every viable gate is off
    at galaxy depths by construction.  So this channel is IN-SAMPLE and the two
    vertical channels are out-of-sample predictions.  Stated, not buried.
    """
    Bc_h, Bc_a = bench.B_of("cluster", cand, amps)
    tgt = BREQ if target == "lane12" else BFLAT
    lg = np.log10(np.maximum(Bc_h, 1e-12))
    rms = np.sqrt(np.mean((lg - np.log10(tgt)[None, :]) ** 2, axis=1))
    j = int(np.nanargmin(rms))
    Aopt = float(amps[j])
    # the SECOND, independent cluster target: a flat B = 2 across 300-1414 kpc,
    # which is X-COP's nu/nu_RAR = 2.53 for A2029 and does NOT come from a
    # published lensing mass profile.  Reported beside the lane-12 fit so the
    # provenance-caveated target never stands alone.
    rms_f = np.sqrt(np.mean((lg - np.log10(BFLAT)[None, :]) ** 2, axis=1))
    jf = int(np.nanargmin(rms_f))
    cand.A = float(amps[jf])
    Bf_f, _ = bench.B_of("field", cand, [amps[jf]])
    Bm_f, _ = bench.B_of("member", cand, [amps[jf]])
    flat = dict(A_flat=float(amps[jf]), rms_dex_at_A_flat=float(rms_f[jf]),
                B_cluster_flat=[float(x) for x in Bc_h[jf]],
                field_dex_flat=float(np.max(np.abs(np.log10(
                    np.maximum(Bf_f[0], 1e-12))))),
                member_dex_flat=float(np.max(np.abs(np.log10(
                    np.maximum(Bm_f[0], 1e-12))))))
    cand.A = Aopt
    Bf_h, _ = bench.B_of("field", cand, [Aopt])
    Bm_h, _ = bench.B_of("member", cand, [Aopt])
    rms_flat = float(np.sqrt(np.mean((lg[j] - np.log10(BFLAT)) ** 2)))
    rms_l12 = float(np.sqrt(np.mean((lg[j] - np.log10(BREQ)) ** 2)))
    fdex = float(np.max(np.abs(np.log10(np.maximum(Bf_h[0], 1e-12)))))
    mdex = float(np.max(np.abs(np.log10(np.maximum(Bm_h[0], 1e-12)))))
    sp = float(np.nanmax(Bc_h[:, 2]) - np.nanmin(Bc_h[:, 2]))
    return dict(A=Aopt, B_cluster=[float(x) for x in Bc_h[j]],
                B_cluster_arith=[float(x) for x in Bc_a[j]],
                rms_dex_lane12=rms_l12, rms_dex_flat=rms_flat,
                field_dex=fdex, member_dex=mdex,
                B_field=[float(x) for x in Bf_h[0]],
                B_member=[float(x) for x in Bm_h[0]],
                shape=float(Bc_h[j, 3] / max(Bc_h[j, 0], 1e-12)),
                B_1Mpc=float(Bc_h[j, 2]),
                amp_spread_B1Mpc=sp,
                at_amp_grid_edge=bool(j in (0, len(amps) - 1)),
                harm_vs_arith_dex=float(np.max(np.abs(
                    np.log10(np.maximum(Bc_h[j], 1e-12))
                    - np.log10(np.maximum(Bc_a[j], 1e-12))))), **flat)
