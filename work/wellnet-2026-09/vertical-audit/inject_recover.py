"""ITEM 2 -- the shared-denominator audit, by Newtonian injection and recovery.

THE CHARGE
----------
    log10 B_z  =  2 log10 sigma_LOS,0(obs)  -  log10 Sigma_dyn-denominator
and the denominator contains  Sigma_L0 = 10^(0.4(M_K,sun + 21.572 - mu0_K,i))
with a measured logarithmic coefficient of -0.994 (see bz_formula.md).  The
ABSCISSA of the headline regression is log10 Sigma_L0 -- the same number.  A
photometric error therefore moves a galaxy RIGHT in x and DOWN in y, and the
naive null slope is NEGATIVE even when B_z = 1 identically.  A label shuffle
cannot see this, because the covariance is WITHIN a row.

WHAT THIS SCRIPT DOES
---------------------
    (a) give every galaxy a true Newtonian B_z = 1;
    (b) draw latent Sigma_b, sigma_z, h_z, h_R and photometry;
    (c) apply the complete observational covariance -- including the correlated
        h_z-from-h_R Bershady relation and the mu0_K / h_R decomposition
        degeneracy at several assumed correlation strengths;
    (d) reconstruct B_z and Sigma_0 through the REAL pipeline (vaudit_core.Bench
        reproduces adyn_run.py's -0.346 to three decimals);
    (e) report P(slope <= -0.346 | Newton).

It then INVERTS the question: by what factor must the photometric error be
inflated before the artefact alone reproduces the observed slope?

And it checks SIZING: inject a true non-zero slope and confirm the estimator
recovers it without bias, so the null is a null and not a broken forward model.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vaudit_core as V                                       # noqa: E402
import adyn_model as M                                        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OBSERVED_SLOPE = -0.34592111547830706          # adyn_results.json, p50
BERSHADY_EXP = 0.643                           # d log h_z / d log h_R


# ============================================================== truth + noise
class Injector:
    """One Newtonian universe per call to `trial`."""

    def __init__(self, ndraw_inner=120, seed=0, exact_apc=False):
        self.ref = V.Bench()                   # catalogue values = latent truth
        self.exact_apc = exact_apc
        self.T = self.ref                      # the TRUE universe never moves
        self.NG = self.ref.NG
        g = self.ref.GAL
        self.mu0_true = self.ref.mu0K.copy()
        self.hR_true = self.ref.hR_as_v.copy()
        self.hz_true = self.ref.HZ_TAB.copy()
        self.e_mu0 = np.array([x.emu0K for x in g])
        self.e_lhR = np.array([x.ehR_as / x.hR_as for x in g]) / np.log(10)
        self.e_lhz = np.array([x.ehz_kpc / x.hz_kpc for x in g]) / np.log(10)
        self.e_sig = np.array([x.esLOS0 for x in g])
        self.e_inc = np.array([x.eincl for x in g])
        self.inc = np.array([x.incl for x in g])
        self.ndraw_inner = ndraw_inner
        # Bershady scatter ORTHOGONAL to the h_R channel: e_hz contains the
        # h_R propagation already, so remove it rather than double-count.
        self.s_rel = np.sqrt(np.maximum(
            self.e_lhz ** 2 - (BERSHADY_EXP * self.e_lhR) ** 2, 1e-6))

    # ------------------------------------------------------------- one trial
    def trial(self, r, rho_mu_hR=0.0, mu0_infl=1.0, hR_infl=1.0,
              incl_err=False, true_slope=0.0, resample_gal=False):
        """Return the recovered d log10 B_z / d log10 Sigma_0.

        `true_slope` != 0 injects a genuine law  B_z = (Sigma_0/<Sigma_0>)^p
        so the estimator's SIZING can be checked on the same machinery.
        """
        NG = self.NG
        # ---- (b) latent truth: nuisances drawn from the pipeline's own priors
        C_t = V.Bench.draw_common(r)
        lU_t = C_t["zU"] + C_t["sc"] * (self.ref.BK - M.FID["BK_pivot"]) \
            + r.normal(0.0, M.FID["s_Upsilon_gal"], NG)
        lfg_t = C_t["lfg"] + r.normal(0.0, 0.15, NG)
        al_t = np.full(NG, C_t["al"])

        # ---- (c) the observational covariance, generated as a LATENT->OBSERVED
        #          map so the SAME realisation feeds x and y
        d_lhR = r.normal(0.0, self.e_lhR * hR_infl)            # log10 h_R error
        s_mu_free = np.sqrt(np.maximum(
            (0.4 * self.e_mu0 * mu0_infl) ** 2
            - (rho_mu_hR * 2.0 * self.e_lhR * hR_infl) ** 2, 0.0))
        # rho_mu_hR = -1 is the fixed-total-luminosity degeneracy
        #   L = 2 pi Sigma_0 h_R^2  =>  d log Sigma_0 = -2 d log h_R
        d_lSig = (rho_mu_hR * 2.0 * d_lhR) + r.normal(0.0, s_mu_free)
        d_lhz = BERSHADY_EXP * d_lhR + r.normal(0.0, self.s_rel)

        mu0_obs = self.mu0_true - d_lSig / 0.4        # mu0 falls as Sigma rises
        hR_obs = self.hR_true * 10 ** d_lhR
        hz_obs = self.hz_true * 10 ** d_lhz
        inc_obs = self.inc.copy()
        if incl_err:
            inc_obs = np.clip(self.inc + r.normal(0.0, self.e_inc), 5.0, 80.0)

        # ---- (a) TRUE universe: Newton, B_z = 1 exactly (or the injected law)
        T = self.T                             # latent photometry never changes
        bt = T.newton_chain(10 ** lU_t, self.hz_true, np.clip(10 ** lfg_t, 0, 3),
                            al_t, C_t["kv"], C_t["fhg"], C_t["fhzg"])
        boost = 1.0
        if true_slope:
            lx = T.log_sigma0()
            boost = 10 ** (true_slope * (lx - lx.mean()))[:, None]
        sl_t = T.to_los(np.sqrt(bt["s2"] * boost) / 1e3, al_t)
        a_t, _ = M.fit_exponential_rows(V.XG, sl_t, C_t["lo"], C_t["hi"])
        sig_obs = np.maximum(a_t + r.normal(0.0, self.e_sig), 1.0)

        # ---- (d) reconstruct through the REAL pipeline on the OBSERVED inputs
        gal_obs = [self._clone(g, inc_obs[j]) for j, g in enumerate(self.ref.GAL)] \
            if incl_err else self.ref.GAL
        O = V.Bench(gals=gal_obs, mu0K=mu0_obs, hR_as=hR_obs, hz_kpc=hz_obs,
                    sLOS0=sig_obs,
                    apc=None if self.exact_apc else self.ref.APC)
        idx = r.integers(0, NG, NG) if resample_gal else np.arange(NG)
        bz = O.bz_draws(self.ndraw_inner, seed=int(r.integers(1, 2 ** 31)))
        x = O.log_sigma0()
        sl = np.array([np.polyfit(x[idx], bz[d][idx], 1)[0]
                       for d in range(bz.shape[0])])
        return float(np.median(sl))

    @staticmethod
    def _clone(g, incl):
        import copy
        h = copy.copy(g)
        h.incl = float(incl)
        return h


# ==================================================================== driver
def run(ntrial=400, ndraw_inner=120, seed=20260904):
    t0 = time.time()
    inj = Injector(ndraw_inner=ndraw_inner)
    rng = np.random.default_rng(seed)
    res = {}

    # ---- GATE: reusing the reference aperture operator must not move anything.
    # scipy caches an FFT plan per array length, and h_R jitters continuously,
    # so recomputing it 400x exhausts memory; six paired trials settle it.
    ge = Injector(ndraw_inner=ndraw_inner, exact_apc=True)
    a = np.array([inj.trial(np.random.default_rng(1000 + i), rho_mu_hR=-1.0)
                  for i in range(6)])
    b = np.array([ge.trial(np.random.default_rng(1000 + i), rho_mu_hR=-1.0)
                  for i in range(6)])
    res["gate_apc_reuse"] = dict(max_abs_diff=float(np.max(np.abs(a - b))),
                                 slopes_reused=a.tolist(),
                                 slopes_exact=b.tolist())
    print(f"  GATE aperture-operator reuse: max |d slope| = "
          f"{np.max(np.abs(a-b)):.2e} over 6 paired trials "
          f"({'PASS' if np.max(np.abs(a-b)) < 5e-3 else 'FAIL'})")
    del ge

    scen = [
        ("A_independent", dict(rho_mu_hR=0.0)),
        ("B_fixed_Ltot", dict(rho_mu_hR=-1.0)),
        ("C_fixedL_incl", dict(rho_mu_hR=-1.0, incl_err=True)),
        ("D_adversarial_x3", dict(rho_mu_hR=-1.0, mu0_infl=3.0, hR_infl=3.0,
                                  incl_err=True)),
        ("E_galaxy_resample", dict(rho_mu_hR=-1.0, incl_err=True,
                                   resample_gal=True)),
    ]
    for name, kw in scen:
        s = np.array([inj.trial(rng, **kw) for _ in range(ntrial)])
        p = float(np.mean(s <= OBSERVED_SLOPE))
        res[name] = dict(kw={k: (v if not isinstance(v, bool) else int(v))
                             for k, v in kw.items()},
                         n=ntrial, median=float(np.median(s)),
                         mean=float(np.mean(s)), sd=float(np.std(s)),
                         p16=float(np.percentile(s, 16)),
                         p84=float(np.percentile(s, 84)),
                         p2_5=float(np.percentile(s, 2.5)),
                         p97_5=float(np.percentile(s, 97.5)),
                         min=float(s.min()),
                         p_le_observed=p,
                         # exact one-sided 95% upper bound for 0 of n successes
                         p_le_observed_upper95=float(1 - 0.05 ** (1 / ntrial))
                         if p == 0 else p)
        print(f"  {name:<20} null slope {np.median(s):+.4f} "
              f"[{np.percentile(s,2.5):+.4f},{np.percentile(s,97.5):+.4f}]  "
              f"min {s.min():+.4f}   P(<= -0.346) = {p:.4f}   "
              f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------- inversion: how much?
    print("\n  inversion -- photometric-error inflation needed for the artefact")
    infl_scan = {}
    for f in (1, 2, 3, 5, 7, 10, 14):
        s = np.array([inj.trial(rng, rho_mu_hR=-1.0, mu0_infl=f, hR_infl=f)
                      for _ in range(max(ntrial // 4, 60))])
        infl_scan[f] = dict(median=float(np.median(s)), sd=float(np.std(s)),
                            p_le_observed=float(np.mean(s <= OBSERVED_SLOPE)))
        print(f"    e_mu0_K x{f:<3d} (median {0.070*f:.2f} mag) -> null slope "
              f"{np.median(s):+.4f} +- {np.std(s):.4f}   "
              f"P(<= -0.346) = {infl_scan[f]['p_le_observed']:.3f}")
    res["error_inflation_scan"] = infl_scan

    # ------------------------------------------------------- sizing / power
    print("\n  sizing: inject a TRUE slope and recover it (same machinery)")
    siz = {}
    for p_true in (-0.60, -0.346, -0.15, 0.0, +0.30):
        s = np.array([inj.trial(rng, rho_mu_hR=-1.0, true_slope=p_true)
                      for _ in range(max(ntrial // 4, 60))])
        siz[f"{p_true:+.3f}"] = dict(median=float(np.median(s)),
                                     sd=float(np.std(s)),
                                     bias=float(np.median(s) - p_true))
        print(f"    injected {p_true:+.3f} -> recovered {np.median(s):+.4f} "
              f"+- {np.std(s):.4f}   bias {np.median(s)-p_true:+.4f}")
    res["sizing"] = siz
    res["observed_slope"] = OBSERVED_SLOPE
    res["config"] = dict(ntrial=ntrial, ndraw_inner=ndraw_inner, seed=seed,
                         bershady_exp=BERSHADY_EXP)
    res["wall_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    n = 400 if "--fast" not in sys.argv else 60
    d = 120 if "--fast" not in sys.argv else 40
    out = run(ntrial=n, ndraw_inner=d)
    with open(os.path.join(OUT, "injection_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  wrote injection_results.json  ({out['wall_s']:.0f}s)")
