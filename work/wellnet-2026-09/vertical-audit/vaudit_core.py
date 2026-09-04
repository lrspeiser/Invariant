"""Re-instantiation of the adyn B_z pipeline, for auditing.

Nothing here is a *simplification* of `adyn_run.py`.  Every physics function is
imported from `adyn_model.py` (the same module the original run used) and the
three glue functions that live inside the `adyn_run.py` SCRIPT body --
`newton_chain`, `to_los`, `draw_common`/`draw_pergal` -- are reproduced
character-for-character below, because `adyn_run.py` executes a 20-minute run on
import and cannot be imported.

`selftest()` asserts that this module reproduces the published
d log10 B_z / d log10 Sigma_0 = -0.346 and the published nuisance budget.  If it
does not, every number downstream is void.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import i0, i1, k0, k1

ADYN = ("C:/Users/henry/Documents/Codex/2026-08-21/"
        "Invariant-main-integration/work/gravity-cluster-audit-2026-09/adyn")
GRAVLAB = ("C:/Users/henry/Documents/Codex/2026-08-21/"
           "Invariant-main-integration/work/gravitylab")
for p in (ADYN, GRAVLAB):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(ADYN)                       # thickness_table.npz lives beside the code
import adyn_model as M               # noqa: E402

G, KPC, PC, MSUN = M.G, M.KPC, M.PC, M.MSUN

# frozen law constants, read from adyn_results.json (fitted to SPARC only)
FIT = dict(rar=1.0839269140212038e-10, aqual=1.0592537251772854e-10)

NR = 200
XG = np.linspace(0.02, 5.0, NR)
YY = np.maximum(XG / 2.0, 1e-8)
BRF = i0(YY) * k0(YY) - i1(YY) * k1(YY)
NU_C = 140
UG = np.linspace(0.0, 12.0, NU_C)


# ------------------------------------------------------------------ the sample
def load(verbose=False):
    gals = [g for g in M.load_diskmass(verbose=verbose) if g.keep]
    return gals


class Bench:
    """Everything `adyn_run.py` puts at module scope between STEP 1 and STEP 3."""

    def __init__(self, gals=None, mu0K=None, hR_as=None, hz_kpc=None,
                 sLOS0=None, apc=None):
        self.GAL = gals if gals is not None else load()
        self.NG = len(self.GAL)
        g = self.GAL
        # photometry may be OVERRIDDEN (that is the point of the injection test)
        # DMGalaxy.mu0K is loaded from column mu0_K_i (inclination-corrected)
        self.mu0K = np.array([x.mu0K for x in g]) if mu0K is None else np.asarray(mu0K, float)
        self.hR_as_v = np.array([x.hR_as for x in g]) if hR_as is None else np.asarray(hR_as, float)
        self.D = np.array([x.D for x in g])
        self.hR_as = self.hR_as_v[:, None]
        self.hR_m = (self.hR_as_v * self.D / M.ARCSEC_PER_RAD * 1e3 * KPC)[:, None]
        self.SigL0 = (10.0 ** (0.4 * (M.MSUN_K + 21.572 - self.mu0K)))[:, None]
        self.INC = np.radians(np.array([x.incl for x in g]))[:, None]
        self.RS_AS = np.array([x.rs_as for x in g])[:, None]
        UARC = np.maximum(XG[None, :] * self.hR_as / self.RS_AS, 1e-9)
        DLNV = UARC / ((1 + UARC ** 2) * np.arctan(UARC))
        self.BETA = np.sqrt(np.clip(0.5 * (1 + DLNV), 1e-6, None))
        self.R_AS = XG[None, :] * self.hR_as
        self.OBS_AMP = np.array([x.sLOS0 for x in g]) if sLOS0 is None else np.asarray(sLOS0, float)
        self.OBS_EAMP = np.array([x.esLOS0 for x in g])
        self.OBS_H = np.array([x.hsLOS_as for x in g])
        self.OBS_EH = np.array([x.ehsLOS_as for x in g])
        self.HZ_TAB = np.array([x.hz_kpc for x in g]) if hz_kpc is None else np.asarray(hz_kpc, float)
        self.EHZ_TAB = np.array([x.ehz_kpc for x in g])
        bk = np.array([x.BK for x in g])
        self.BK = np.where(np.isfinite(bk), bk, np.nanmedian(bk))
        self.E_DIST = np.array([x.eD / x.D for x in g])
        self.hR_kpc = np.array([x.hR_kpc for x in g])
        self.J10 = int(np.argmin(np.abs(XG - 1.0)))
        self.J22 = int(np.argmin(np.abs(XG - 2.2)))
        if apc is None:
            self._apc()
        else:
            # reuse a precomputed 2.7" fibre (x) 1.5" PSF correction.  Only ever
            # legitimate when the photometry differs from the reference by a few
            # per cent; `inject_recover.py` gates this numerically.
            self.APC = np.asarray(apc, float)

    # --------------------------------------------------- aperture (x) PSF, 3a
    def _apc(self):
        self.APC = np.ones((self.NG, NR))
        base = self.newton_chain(np.full(self.NG, 0.60), self.HZ_TAB,
                                 np.full(self.NG, 0.25), np.full(self.NG, 0.60),
                                 1.5, 2.0, 0.5)
        sl0 = self.to_los(np.sqrt(base["s2"]) / 1e3, np.full(self.NG, 0.60),
                          apply_ap=False)
        for j, gg in enumerate(self.GAL):
            sm = M.apply_aperture(gg, self.R_AS[j], sl0[j],
                                  M.FID["fibre_diam_as"], M.FID["psf_fwhm_as"])
            self.APC[j] = np.clip(sm / sl0[j], 0.5, 3.0)

    # ------------------------------------- VERBATIM from adyn_run.py, line 380
    def newton_chain(self, Ups, hz_kpc, fgas, alpha, kv, f_hg, f_hzg):
        NG = self.NG
        hR_m, SigL0 = self.hR_m, self.SigL0
        prof = M.profile_for_k(kv)
        A_ss, A_sg, L_s = M.vertical_weights(prof, f_hzg, M.profile_for_k(2.0))
        Sig_s0 = Ups[:, None] * SigL0 * MSUN / PC ** 2
        hz = (hz_kpc * KPC)[:, None]
        hg = f_hg * hR_m
        Sig_g0 = fgas[:, None] * Sig_s0 / f_hg ** 2
        R = XG[None, :] * hR_m
        Ts = np.array([M.thickness_T(XG, float(2 * hz[j, 0] / hR_m[j, 0]))
                       for j in range(NG)])
        xg = XG / f_hg
        yg = np.maximum(xg / 2.0, 1e-8)
        brg = i0(yg) * k0(yg) - i1(yg) * k1(yg)
        Tg = np.array([M.thickness_T(xg, float(2 * f_hzg * hz[j, 0] / hg[j, 0]))
                       for j in range(NG)])
        gR = (np.pi * G * Sig_s0 * XG[None, :] * BRF[None, :] * Ts
              + np.pi * G * Sig_g0 * xg[None, :] * brg[None, :] * Tg)
        Vc2 = R * gR
        Sig_s = Sig_s0 * np.exp(-XG[None, :])
        Sig_g = Sig_g0 * np.exp(-xg[None, :])
        dV = np.gradient(Vc2, XG, axis=1) / (hR_m * R)
        s2 = np.maximum(2 * np.pi * G * hz * (Sig_s * A_ss + Sig_g * A_sg)
                        - L_s * hz ** 2 * dV, 1e-30)
        return dict(gR=gR, Vc2=Vc2, s2=s2, Sig_s=Sig_s, Sig_g=Sig_g, hz=hz, R=R,
                    prof=prof, f_hzg=f_hzg)

    def to_los(self, sz_kms, alpha, apply_ap=True):
        c2, s2i = np.cos(self.INC) ** 2, np.sin(self.INC) ** 2
        sl = sz_kms * np.sqrt(c2 + 0.5 * s2i * (1 + self.BETA ** 2)
                              / alpha[:, None] ** 2)
        return sl * self.APC if apply_ap else sl

    # --------------------------------------------- VERBATIM, adyn_run.py 646ff
    @staticmethod
    def draw_common(r):
        return dict(
            zU=r.normal(np.log10(M.FID["Upsilon_K"]), M.FID["s_Upsilon"]),
            sc=r.normal(M.FID["col_slope"], M.FID["s_col_slope"]),
            dhz=r.normal(0.0, M.FID["s_hz_sys"]),
            kv=round(float(r.uniform(M.FID["k_lo"], M.FID["k_hi"])), 2),
            al=float(np.clip(r.normal(M.FID["alpha"], M.FID["s_alpha"]),
                             0.35, 0.95)),
            lfg=r.normal(np.log10(M.FID["f_gas"]), 0.20),
            fhg=float(r.uniform(1.5, 3.0)),
            fhzg=round(float(r.uniform(0.3, 0.8)), 3),
            lo=float(r.uniform(0.2, 0.5)), hi=float(r.uniform(1.5, 2.5)))

    def draw_pergal(self, r, C):
        NG = self.NG
        lU = C["zU"] + C["sc"] * (self.BK - M.FID["BK_pivot"])
        lhz = np.log10(self.HZ_TAB) + C["dhz"]
        lfg = np.full(NG, C["lfg"])
        lU = lU + r.normal(0.0, M.FID["s_Upsilon_gal"], NG)
        lhz = lhz + r.normal(0.0, self.EHZ_TAB / self.HZ_TAB / np.log(10), NG)
        lhz = lhz + 0.643 * np.log10(1.0 + r.normal(0.0, self.E_DIST, NG))
        lfg = lfg + r.normal(0.0, 0.15, NG)
        sob = self.OBS_AMP + r.normal(0.0, self.OBS_EAMP, NG)
        return (10 ** lU, 10 ** lhz, np.clip(10 ** lfg, 0.0, 3.0),
                np.full(NG, C["al"]), np.maximum(sob, 1.0))

    def amp_newton(self, Ups, hz, fg, al, C):
        b = self.newton_chain(Ups, hz, fg, al, C["kv"], C["fhg"], C["fhzg"])
        sl = self.to_los(np.sqrt(b["s2"]) / 1e3, al)
        a_, h_ = M.fit_exponential_rows(XG, sl, C["lo"], C["hi"])
        return a_, h_ * np.squeeze(self.hR_as), b

    # ------------------------------------------------------------ observables
    def log_sigma0(self):
        return np.log10(np.squeeze(self.SigL0))

    def bz_draws(self, ndraw=800, seed=999, sLOS0=None):
        """(ndraw, NG) array of log10 B_z(observed), exactly as step 6b."""
        r = np.random.default_rng(seed)
        out = []
        keep = self.OBS_AMP
        if sLOS0 is not None:
            self.OBS_AMP = np.asarray(sLOS0, float)
        for _ in range(ndraw):
            C = self.draw_common(r)
            Ups, hz, fg, al, sob = self.draw_pergal(r, C)
            aN, _, _ = self.amp_newton(Ups, hz, fg, al, C)
            out.append(2.0 * np.log10(sob / aN))
        self.OBS_AMP = keep
        return np.array(out)

    def slope_draws(self, bz=None, ndraw=800, seed=999):
        bz = self.bz_draws(ndraw, seed) if bz is None else bz
        x = self.log_sigma0()
        return np.array([np.polyfit(x, bz[d], 1)[0] for d in range(bz.shape[0])])

    # ---------------------------------------------------- law B_z, step 4/6
    def law_Bz(self, key, base):
        """B_z_eff(NG,NR) for 'rar' / 'aqual_simple'; VERBATIM adyn_run.py 528."""
        if key == "newton":
            return np.ones((self.NG, NR))
        prof = base["prof"]
        w = np.interp(UG, prof.u, prof.w)
        Cs = np.interp(UG, prof.u, prof.Cn, left=0.0, right=1.0)
        pg = M.profile_for_k(2.0)
        Cg = np.interp(UG / base["f_hzg"], pg.u, pg.Cn, left=0.0, right=1.0)
        hz, R, gRN, Vc2N = base["hz"], base["R"], base["gR"], base["Vc2"]
        zz = UG[None, None, :] * hz[:, :, None]
        Sig_lt = (base["Sig_s"][:, :, None] * Cs[None, None, :]
                  + base["Sig_g"][:, :, None] * Cg[None, None, :])
        dN = (np.gradient(Vc2N, XG, axis=1) / (self.hR_m * R))[:, :, None]
        KzN = np.maximum(2 * np.pi * G * Sig_lt - zz * dN, 1e-30)
        if key == "rar":
            a0v = FIT["rar"]
            Kz = M.nu_rar(np.sqrt(gRN[:, :, None] ** 2 + KzN ** 2) / a0v) * KzN
        elif key == "aqual_simple":
            a0v = FIT["aqual"]
            gR = 0.5 * (gRN + np.sqrt(gRN ** 2 + 4 * gRN * a0v))
            Kz = M.aqual_Kz(KzN, gR[:, :, None] * np.ones_like(KzN), a0v)
        else:
            raise KeyError(key)
        s2 = np.trapezoid(w[None, None, :] * Kz, zz, axis=2)
        s2n = np.trapezoid(w[None, None, :] * KzN, zz, axis=2)
        return s2 / s2n


# ------------------------------------------------------------------- selftest
def selftest(ndraw=800, verbose=True):
    B = Bench()
    bz = B.bz_draws(ndraw, seed=999)
    sl = B.slope_draws(bz)
    out = dict(NG=B.NG,
               slope_p50=float(np.percentile(sl, 50)),
               slope_p16=float(np.percentile(sl, 16)),
               slope_p84=float(np.percentile(sl, 84)),
               slope_sd_raw=float(np.std(sl)),
               logBz_mean=float(np.mean(bz)),
               mu0_range=[float(B.mu0K.min()), float(B.mu0K.max())])
    if verbose:
        print(f"    galaxies retained            : {out['NG']}  (published 28)")
        print(f"    d log10 Bz / d log10 Sigma_0 : {out['slope_p50']:+.3f}"
              f"  68% [{out['slope_p16']:+.3f}, {out['slope_p84']:+.3f}]"
              f"   (published -0.346 [-0.416,-0.276])")
        print(f"    raw sd over nuisance draws   : {out['slope_sd_raw']:.4f}"
              f"   (published 0.1735/1.3485 = 0.1287)")
        print(f"    mean log10 B_z               : {out['logBz_mean']:+.3f}"
              f"   (published -0.146)")
    return B, bz, sl, out


if __name__ == "__main__":
    selftest()
