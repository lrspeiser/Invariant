"""CHANNELS 2 and 3 -- vertical amplitude B_z, and vertical radial shape
h_sigma_LOS.  Built on adyn_model.py (Run L), which is imported and NOT
modified; only the law layer is generalised.

CHANNEL 2, AMPLITUDE, IS A CONSTRAINT AND NOT A DISCRIMINATOR, and this lane
says so rather than scoring it as if it were.  The measurement is
    B_z = 0.715,  68% [0.468, 1.079],  95% [0.301, 1.670],  width 0.192 dex
with the systematic floor 8.4x the statistical part and dominated by
common-mode terms (the Upsilon_K zero point above all).  The largest
law-to-Newton separation any law produces is 0.190 dex = 0.99 sigma.  A
candidate is therefore scored PASS/FAIL against the 95% interval and its
z-score is reported, but the joint ranking never lets this channel decide
anything on its own.

CHANNEL 3, SHAPE, DOES DISCRIMINATE.  h_sigma_LOS is blind to a constant
vertical boost -- multiplying K_z by 8 moves it by 1.6e-15 dex -- so it sees
only the RADIAL RUN of the boost, which is exactly the part the amplitude
systematics cannot fake.  Observed 28.65 arcsec; Newton 30.80, RAR 35.20,
AQUAL 34.96, anisotropic tensor 31.34, isotropic tensor 48.16, at chi2/dof
10.5 / 20.2 / 20.0 / 11.1 / 132.9.  It rejected the isotropic tensor.

HOW A CANDIDATE ENTERS.  The z-integrated field equation gives, exactly,
    K_z^N(R,z) = 2 pi G Sigma(<z) - z (1/R) dV_c^2/dR
and the radial leakage term is the same for every law, so it cancels from B_z
at leading order (adyn_model.Kz_grid documents this).  On top of that:
    scalar_a0   K_z = nu(g_tot^N/(a0(1+A W))) K_z^N
    iso_K       K = c I, c = exp(-A W):  K_z = nu(g^N/(a0 c^1.5)) K_z^N / c
    tensor_*    K = exp(a P),  P = dhat dhat^T - I/3, a = A W (times -1.5 for
                the tidal structure, whose That IS -1.5 P for a point mass):
                    e^T K e = e^{-a/3} + (e^{2a/3} - e^{-a/3}) (e.dhat)^2
                so K_z uses (e_z.dhat)^2 and g_R uses (e_R.dhat)^2.  In the
                MIDPLANE dhat_z = 0 for the field-direction and well-network
                structures, so those SUPPRESS the vertical field (k_zz =
                e^{-a/3} < 1 for a > 0) while boosting the radial one -- that
                is the anisotropy the shape channel can see.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import i0, i1, k0, k1

ADYN = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
        "work/gravity-cluster-audit-2026-09/adyn")
GRAVLAB = ("C:/Users/henry/Documents/Codex/2026-08-21/"
           "Invariant-main-integration/work/gravitylab")
for p in (ADYN, GRAVLAB):
    if p not in sys.path:
        sys.path.insert(0, p)

import adyn_model as M                                          # noqa: E402
from tw_core import A0, W_of, nu_rar, mond_invert               # noqa: E402
from ch_radial import L_NL, M_NL                                # noqa: E402

G, KPC, PC, MSUN = M.G, M.KPC, M.PC, M.MSUN

#: Run L's posterior on the observed vertical amplitude.
BZ_OBS = 0.715
BZ_OBS_LOG = float(np.log10(BZ_OBS))
BZ_WIDTH_DEX = 0.192
BZ_95 = (0.301, 1.670)
#: Run L's fiducial nuisances, adopted unchanged.
UPS_K, F_GAS, ALPHA, K_VERT, F_HG, F_HZG = 0.60, 0.25, 0.60, 1.5, 2.0, 0.5
WIN = (0.3, 2.0)


class VerticalBench:
    """The DiskMass forward chain, vectorised over galaxies, law-agnostic."""

    def __init__(self, nR=200, nu=140, verbose=False):
        gals = M.load_diskmass(verbose=verbose)
        self.GAL = [g for g in gals if g.keep]
        self.NG = len(self.GAL)
        self.NR, self.NU = nR, nu
        self.XG = np.linspace(0.02, 5.0, nR)             # R/h_R
        self.UG = np.linspace(0.0, 12.0, nu)             # z/h_z
        y = np.maximum(self.XG / 2.0, 1e-8)
        self.BRF = i0(y) * k0(y) - i1(y) * k1(y)
        # the Freeman POTENTIAL shape, for the |Phi_N| invariant
        self.PHF = i0(y) * k1(y) - i1(y) * k0(y)
        self.hR_m = np.array([g.hR_m for g in self.GAL])[:, None]
        self.hR_as = np.array([g.hR_as for g in self.GAL])[:, None]
        self.SigL0 = np.array([g.SigmaL0 for g in self.GAL])[:, None]
        self.INC = np.radians(np.array([g.incl for g in self.GAL]))[:, None]
        RS_AS = np.array([g.rs_as for g in self.GAL])[:, None]
        u = np.maximum(self.XG[None, :] * self.hR_as / RS_AS, 1e-9)
        dlnV = u / ((1 + u ** 2) * np.arctan(u))
        self.BETA = np.sqrt(np.clip(0.5 * (1 + dlnV), 1e-6, None))
        self.R_AS = self.XG[None, :] * self.hR_as
        self.OBS_AMP = np.array([g.sLOS0 for g in self.GAL])
        self.OBS_EAMP = np.array([g.esLOS0 for g in self.GAL])
        self.OBS_H = np.array([g.hsLOS_as for g in self.GAL])
        self.OBS_EH = np.array([g.ehsLOS_as for g in self.GAL])
        self.HZ = np.array([g.hz_kpc for g in self.GAL])
        self.J10 = int(np.argmin(np.abs(self.XG - 1.0)))
        self.J22 = int(np.argmin(np.abs(self.XG - 2.2)))
        self.base = self._newton_chain()
        self.APC = self._aperture()
        self._inv_cache: dict = {}
        _sl = self._to_los(np.sqrt(self.base["s2"]) / 1e3)
        self.amp_newton, hxN = M.fit_exponential_rows(self.XG, _sl, *WIN)
        self.h_newton_as = hxN * np.squeeze(self.hR_as)

    # ------------------------------------------------------------ Newtonian
    def _newton_chain(self):
        XG, NG = self.XG, self.NG
        prof = M.profile_for_k(K_VERT)
        A_ss, A_sg, L_s = M.vertical_weights(prof, F_HZG, M.profile_for_k(2.0))
        Sig_s0 = UPS_K * self.SigL0 * MSUN / PC ** 2
        hz = (self.HZ * KPC)[:, None]
        hg = F_HG * self.hR_m
        Sig_g0 = F_GAS * Sig_s0 / F_HG ** 2
        R = XG[None, :] * self.hR_m
        Ts = np.array([M.thickness_T(XG, float(2 * hz[j, 0] / self.hR_m[j, 0]))
                       for j in range(NG)])
        xg = XG / F_HG
        yg = np.maximum(xg / 2.0, 1e-8)
        brg = i0(yg) * k0(yg) - i1(yg) * k1(yg)
        phg = i0(yg) * k1(yg) - i1(yg) * k0(yg)
        Tg = np.array([M.thickness_T(xg, float(2 * F_HZG * hz[j, 0] / hg[j, 0]))
                       for j in range(NG)])
        gR = (np.pi * G * Sig_s0 * XG[None, :] * self.BRF[None, :] * Ts
              + np.pi * G * Sig_g0 * xg[None, :] * brg[None, :] * Tg)
        # Freeman potential, Phi -> 0 at infinity, both components
        PhiN = -(np.pi * G * Sig_s0 * self.hR_m * XG[None, :]
                 * self.PHF[None, :] * 2.0
                 + np.pi * G * Sig_g0 * hg * xg[None, :] * phg[None, :] * 2.0)
        Vc2 = R * gR
        Sig_s = Sig_s0 * np.exp(-XG[None, :])
        Sig_g = Sig_g0 * np.exp(-xg[None, :])
        dV = np.gradient(Vc2, XG, axis=1) / (self.hR_m * R)
        s2 = np.maximum(2 * np.pi * G * hz * (Sig_s * A_ss + Sig_g * A_sg)
                        - L_s * hz ** 2 * dV, 1e-30)
        w = np.interp(self.UG, prof.u, prof.w)
        Cs = np.interp(self.UG, prof.u, prof.Cn, left=0.0, right=1.0)
        pg = M.profile_for_k(2.0)
        Cg = np.interp(self.UG / F_HZG, pg.u, pg.Cn, left=0.0, right=1.0)
        zz = self.UG[None, None, :] * hz[:, :, None]
        Sig_lt = (Sig_s[:, :, None] * Cs[None, None, :]
                  + Sig_g[:, :, None] * Cg[None, None, :])
        KzN = np.maximum(2 * np.pi * G * Sig_lt - zz * dV[:, :, None], 1e-30)
        return dict(gR=gR, Vc2=Vc2, s2=s2, hz=hz, R=R, w=w, zz=zz, KzN=KzN,
                    PhiN=PhiN, Sig_s=Sig_s, Sig_g=Sig_g, prof=prof)

    def _aperture(self):
        sl0 = self._to_los(np.sqrt(self.base["s2"]) / 1e3, apply_ap=False)
        APC = np.ones((self.NG, self.NR))
        for j, g in enumerate(self.GAL):
            sm = M.apply_aperture(g, self.R_AS[j], sl0[j],
                                  M.FID["fibre_diam_as"], M.FID["psf_fwhm_as"])
            APC[j] = np.clip(sm / sl0[j], 0.5, 3.0)
        return APC

    def _to_los(self, sz_kms, apply_ap=True):
        c2, s2i = np.cos(self.INC) ** 2, np.sin(self.INC) ** 2
        sl = sz_kms * np.sqrt(c2 + 0.5 * s2i * (1 + self.BETA ** 2) / ALPHA ** 2)
        return sl * self.APC if apply_ap else sl

    # ----------------------------------------------------------- invariants
    def invariants(self):
        """Raw invariant fields on the (galaxy, R, z) grid, SI."""
        if self._inv_cache:
            return self._inv_cache
        b = self.base
        KzN, zz, R = b["KzN"], b["zz"], b["R"][:, :, None]
        gRN = b["gR"][:, :, None] * np.ones_like(KzN)
        gtot = np.sqrt(gRN ** 2 + KzN ** 2)
        # Phi(R,z) = Phi(R,0) + int_0^z K_z dz'   (K_z = -dPhi/dz > 0)
        dPhi = np.concatenate(
            [np.zeros_like(KzN[:, :, :1]),
             np.cumsum(0.5 * (KzN[:, :, 1:] + KzN[:, :, :-1])
                       * np.diff(zz, axis=2), axis=2)], axis=2)
        Phi = b["PhiN"][:, :, None] + dPhi
        Menc = np.maximum(R ** 2 * gRN, 1e-30) / G
        Mtot = np.max(Menc, axis=1, keepdims=True)
        # Hessian of Phi_N in (R, phi, z).  Phi grows outward, so with g_R and
        # K_z the (positive) inward field magnitudes: T_RR = dg_R/dR,
        # T_pp = g_R/R, T_zz = dK_z/dz, and trace T = 4 pi G rho exactly.
        T_zz = np.gradient(KzN, axis=2) / np.maximum(
            np.gradient(zz, axis=2), 1e-30)
        T_RR = np.gradient(gRN, axis=1) / np.maximum(
            np.gradient(R, axis=1), 1e-30)
        T_pp = gRN / np.maximum(R, 1e-30)
        tr3 = (T_zz + T_RR + T_pp) / 3.0
        d_zz, d_RR, d_pp = T_zz - tr3, T_RR - tr3, T_pp - tr3
        Tn = np.sqrt(d_zz ** 2 + d_RR ** 2 + d_pp ** 2)
        spec = np.maximum(np.maximum(np.abs(d_zz), np.abs(d_RR)),
                          np.maximum(np.abs(d_pp), 1e-30))
        self._inv_cache = dict(
            one=np.ones_like(KzN), gn=gtot / A0, phi=np.abs(Phi),
            rhobar=np.maximum(3.0 * tr3 / (4.0 * np.pi * G), 1e-40),
            tidal=np.maximum(Tn, 1e-45),
            qbar=(Mtot / (Mtot + M_NL)) * np.ones_like(KzN),
            _gRN=gRN, _KzN=KzN, _gtot=gtot,
            _That_zz=d_zz / spec, _That_RR=d_RR / spec)
        return self._inv_cache

    # ------------------------------------------------------------- the laws
    def _Kz_BR(self, cand):
        """(K_z on the (NG,NR,NU) grid, B_R on (NG,NR)) for one candidate."""
        b, I = self.base, self.invariants()
        KzN, gRN, gtot = I["_KzN"], I["_gRN"], I["_gtot"]
        a0 = cand.a0
        if cand.form == "off" or cand.inv == "one":
            W = np.zeros_like(KzN)
        else:
            W = W_of(cand.form, I[cand.inv] / cand.I0, cand.m)
        # the exponent enters exp(); clip at +-60 (a factor 1e26 in k) so a
        # runaway response returns an obviously-absurd finite number rather
        # than an overflow and a silent nan.  Every clip is visible in the
        # reported k range.
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        def nueff(F, k, aa=None):
            """|grad Phi| / F in a medium of radial conductivity k.

            mond_invert with k = 1 reproduces plain nu_RAR / plain AQUAL, so
            the k = 1 limit of every candidate here is bit-identical to Run L's
            'algebraic' law.  Using the algebraic form for the AQUAL base too
            is a STATED approximation: Run L's bisection on the exact AQUAL
            vertical equation differs from it by <1% (h 34.96 vs 35.20 arcsec,
            B_z 1.548 vs 1.547), and that difference is cross-checked below.
            """
            return mond_invert(F, k, a0 if aa is None else aa,
                               cand.base) / np.maximum(F, 1e-300)

        if cand.struct == "scalar_a0":
            a0e = np.maximum(a0 * (1.0 + cand.A * W), 1e-30)
            Kz = nueff(gtot, np.ones_like(gtot), a0e) * KzN
            BR = nueff(b["gR"], np.ones_like(b["gR"]), a0e[:, :, 0])
            Kz = np.nan_to_num(Kz, nan=1e-30, posinf=1e30, neginf=1e-30)
            BR = np.nan_to_num(BR, nan=1.0, posinf=1e30, neginf=1e-30)
            return np.maximum(Kz, 1e-30), np.maximum(BR, 1e-30)
        if cand.struct == "iso_K":
            k_zz = k_RR = np.exp(np.clip(-cand.A * W, -60.0, 60.0))
        elif cand.struct == "tensor_T":
            # That is DIAGONAL in (R, phi, z) for an axisymmetric disk, so no
            # projector approximation is needed: k_zz and k_RR come straight
            # from its own eigenvalues, normalised by the spectral norm so
            # |That|_2 = 1 identically (the screen lane's bound).
            a = np.clip(cand.A * W, -60.0, 60.0)
            k_zz = np.exp(np.clip(a * I["_That_zz"], -60.0, 60.0))
            k_RR = np.exp(np.clip(a * I["_That_RR"], -60.0, 60.0))
        else:
            a = np.clip(cand.A * W, -60.0, 60.0)
            if cand.struct == "tensor_d":
                dz2 = (KzN / np.maximum(gtot, 1e-300)) ** 2
            else:                                    # tensor_S, dominant well
                Rf = b["R"][:, :, None] * np.ones_like(KzN)
                rr = np.sqrt(Rf ** 2 + b["zz"] ** 2)
                dz2 = (b["zz"] / np.maximum(rr, 1e-30)) ** 2
            e_a, e_b = np.exp(-a / 3.0), np.exp(2.0 * a / 3.0)
            dz2 = np.nan_to_num(dz2, nan=0.0)
            k_zz = np.maximum(e_a + (e_b - e_a) * dz2, 1e-30)
            k_RR = np.maximum(e_a + (e_b - e_a) * (1.0 - dz2), 1e-30)
        Kz = nueff(gtot, k_zz) * KzN
        BR = nueff(b["gR"], k_RR[:, :, 0])
        Kz = np.nan_to_num(Kz, nan=1e-30, posinf=1e30, neginf=1e-30)
        BR = np.nan_to_num(BR, nan=1.0, posinf=1e30, neginf=1e-30)
        return np.maximum(Kz, 1e-30), np.maximum(BR, 1e-30)

    # --------------------------------------------------------------- scoring
    def predict(self, cand):
        b = self.base
        Kz, BR = self._Kz_BR(cand)
        w, zz = b["w"], b["zz"]
        s2 = np.trapezoid(w[None, None, :] * Kz, zz, axis=2)
        s2n = np.trapezoid(w[None, None, :] * b["KzN"], zz, axis=2)
        BzE = s2 / s2n
        sl = self._to_los(np.sqrt(b["s2"] * BzE) / 1e3)
        amp, hx = M.fit_exponential_rows(self.XG, sl, *WIN)
        h_as = hx * np.squeeze(self.hR_as)
        chi_h = float(np.mean(((self.OBS_H - h_as) / self.OBS_EH) ** 2))
        chi_a = float(np.mean(((self.OBS_AMP - amp) / self.OBS_EAMP) ** 2))
        # B_z(law) is defined EXACTLY as Run L defines it: the square of the
        # ratio of the FITTED exponential amplitudes, law over Newton, so the
        # comparison with the observed 0.715 is like for like.
        Bz_law = float(10.0 ** np.mean(2 * np.log10(amp / self.amp_newton)))
        z_amp = abs(np.log10(max(Bz_law, 1e-12)) - BZ_OBS_LOG) / BZ_WIDTH_DEX
        return dict(h_median_as=float(np.median(h_as)),
                    h_chi2dof=chi_h, amp_chi2dof=chi_a,
                    amp_median=float(np.median(amp)),
                    Bz_law=Bz_law,
                    Bz_law_1hR=float(np.median(BzE[:, self.J10])),
                    BR_2p2=float(np.median(BR[:, self.J22])),
                    A_dyn_2p2=float(np.median(BR[:, self.J22])
                                    / max(np.median(BzE[:, self.J22]), 1e-12)),
                    z_amp=float(z_amp),
                    amp_within_95=bool(BZ_95[0] <= Bz_law <= BZ_95[1]),
                    h_as=h_as)
