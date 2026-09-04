"""The joint likelihood and the prespecified-hierarchy fitter.

ONE objective across three surveys:

    -2 ln L  =  chi2(eFEDS raw reduced shear)
              + sum_LoCuSS [ (lnS_obs - lnS_mod)^2/s^2 + ln s^2 ]
              + sum_SL     [ (lnS_obs - lnS_mod)^2/s^2 + ln s^2 ]
              + sum_surveys (o_k - mu_k)^2 / sd_k^2          <- external priors

with  ln S_mod = o_k + f_theta(M, r, g).

The eFEDS term is NOT a set of per-point S estimates: the model's Sigma_s(r)
is applied to the 3-D mass and re-projected through the same Abel integral the
shear pipeline uses, so a radially varying S stays exact.  Only the radial
SHAPE of the slip requires a re-projection; every per-system constant factors
out of the projection linearly and is applied afterwards.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

import common as K
import decl

SURVEYS = ("efeds", "locuss", "sl")

# ------------------------------------------------- which shape needs a solve
# 'shape' parameters enter the 3-D slip and force a re-projection of all 496
# eFEDS systems; 'amp' parameters are per-system constants and are free.
MODELS = {
    "H0":   dict(shape=None, amp=[],            free_offsets=False, c=False),
    "H_P":  dict(shape=None, amp=[],            free_offsets=True,  c=False),
    "H_M":  dict(shape=None, amp=["alpha"],     free_offsets=False, c=True),
    "H_R":  dict(shape="beta", amp=[],          free_offsets=False, c=True),
    "H_G":  dict(shape="gamma", amp=[],         free_offsets=False, c=True),
    "H_MR": dict(shape="beta", amp=["alpha"],   free_offsets=False, c=True),
    "H_T":  dict(shape=("A", "lnxt"), amp=[],   free_offsets=False, c=True),
}


class Joint:
    def __init__(self, bundle, surveys=SURVEYS, mass_key="lnM",
                 fixed_scatter=None, cache=None):
        self.b = bundle
        self.surveys = tuple(surveys)
        self.mass_key = mass_key
        # FIXED intrinsic scatters.  Estimated ONCE under H_P -- the model with
        # free per-survey offsets, which by construction removes all
        # between-survey structure and leaves only within-survey scatter --
        # then frozen for every model.  A free variance would otherwise absorb
        # model misfit and turn a bad fit into "lots of scatter"; with 4
        # strong-lens clusters that degeneracy is severe.
        self.fixed_scatter = fixed_scatter
        self._cache = {} if cache is None else cache
        b = bundle
        self.lo_lnS = np.array([r["lnS"] for r in b.lo_rows])
        self.lo_x = np.array([[r[mass_key], r["lnx"], r["lng"]]
                              for r in b.lo_rows])
        self.lo_e = np.array([r["e_stat"] for r in b.lo_rows])
        self.sl_lnS = np.array([r["lnS"] for r in b.sl_rows])
        self.sl_x = np.array([[r[mass_key], r["lnx"], r["lng"]]
                              for r in b.sl_rows])
        self.sl_e = np.array([r["e_stat"] for r in b.sl_rows])
        self.ef_lnM = b.ef_x[mass_key]
        # SL image systems inside one cluster share the gas model, the BCG
        # centre and the monopole approximation, so they are NOT independent.
        # Blocks by cluster; the covariance carries a cluster-common variance
        # alongside the within-cluster one.  Treating 49 systems as 49
        # independent points would over-weight the strong-lensing anchor by
        # roughly sqrt(49/4).
        cids = [r["cid"] for r in b.sl_rows]
        self.sl_blocks = [np.where(np.array(cids) == c)[0]
                          for c in sorted(set(cids))]

    @staticmethod
    def _block_chi2(res, dvar, s_c2, blocks):
        """chi2 + logdet for C = diag(dvar) + s_c2 * 11^T within each block."""
        tot = 0.0
        for b in blocks:
            r = res[b]
            d = dvar[b]
            di = 1.0 / d
            s1 = float(np.sum(di))
            q = float(np.sum(r * r * di))
            u = float(np.sum(r * di))
            den = 1.0 + s_c2 * s1
            tot += q - s_c2 * u * u / den
            tot += float(np.sum(np.log(d))) + math.log(den)
        return tot

    # ------------------------------------------------------- eFEDS projection
    def project(self, model, shape):
        fam = {"H0": "flat", "H_P": "flat", "H_M": "flat", "H_R": "pow",
               "H_MR": "pow", "H_G": "acc", "H_T": "trans"}[model]
        key = (fam, tuple(np.round(np.atleast_1d(shape), 6))
               if shape is not None else None)
        if key in self._cache:
            return self._cache[key]
        b = self.b
        if model in ("H0", "H_P", "H_M"):
            fn = (lambda sm: 1.0)
        elif model in ("H_R", "H_MR"):
            fn = K.slip_power(b, float(np.atleast_1d(shape)[0]))
        elif model == "H_G":
            fn = K.slip_accel(float(np.atleast_1d(shape)[0]))
        elif model == "H_T":
            A, lnxt = float(shape[0]), float(shape[1])
            fn = K.slip_transition(b, A, math.exp(lnxt))
        else:
            raise ValueError(model)
        S, dS = K.project_slip(b.syss, b.obs, b.ef_idx, fn)
        if len(self._cache) > 400:
            self._cache.clear()
        self._cache[key] = (S, dS)
        return S, dS

    # ---------------------------------------------------------- the objective
    def m2lnL(self, model, shape, pars, want_parts=False):
        """pars = [c?, alpha?, o_efeds, o_locuss, o_sl, ln s_lo, ln s_sl]."""
        spec = MODELS[model]
        i = 0
        c = 0.0
        if spec["c"]:
            c = pars[i]
            i += 1
        alpha = 0.0
        if "alpha" in spec["amp"]:
            alpha = pars[i]
            i += 1
        o = dict(zip(SURVEYS, pars[i:i + 3]))
        i += 3
        if self.fixed_scatter is None:
            s_lo, s_sl = math.exp(pars[i]), math.exp(pars[i + 1])
            s_slc = math.exp(pars[i + 2])      # SL cluster-common scatter
        else:
            s_lo, s_sl, s_slc = self.fixed_scatter

        tot, parts = 0.0, {}

        if "efeds" in self.surveys:
            S, dS = self.project(model, shape)
            amp = np.exp(o["efeds"] + c + alpha * self.ef_lnM)
            ch = self.b.F.chi2(S, dS, amp)
            parts["efeds"] = ch
            tot += ch

        if "locuss" in self.surveys:
            beta = 0.0
            gamma = 0.0
            if model in ("H_R", "H_MR"):
                beta = float(np.atleast_1d(shape)[0])
            if model == "H_G":
                gamma = float(np.atleast_1d(shape)[0])
            mod = o["locuss"] + c + alpha * self.lo_x[:, 0] \
                + beta * self.lo_x[:, 1] + gamma * self.lo_x[:, 2]
            if model == "H_T":
                A, xt = float(shape[0]), math.exp(float(shape[1]))
                mod = o["locuss"] + c + A / (
                    1.0 + (np.exp(self.lo_x[:, 1]) / xt) ** decl.TRANSITION_P)
            v = self.lo_e ** 2 + s_lo ** 2
            ch = float(np.sum((self.lo_lnS - mod) ** 2 / v + np.log(v)))
            parts["locuss"] = ch
            tot += ch

        if "sl" in self.surveys:
            beta = 0.0
            gamma = 0.0
            if model in ("H_R", "H_MR"):
                beta = float(np.atleast_1d(shape)[0])
            if model == "H_G":
                gamma = float(np.atleast_1d(shape)[0])
            mod = o["sl"] + c + alpha * self.sl_x[:, 0] \
                + beta * self.sl_x[:, 1] + gamma * self.sl_x[:, 2]
            if model == "H_T":
                A, xt = float(shape[0]), math.exp(float(shape[1]))
                mod = o["sl"] + c + A / (
                    1.0 + (np.exp(self.sl_x[:, 1]) / xt) ** decl.TRANSITION_P)
            v = self.sl_e ** 2 + s_sl ** 2
            ch = self._block_chi2(self.sl_lnS - mod, v, s_slc ** 2,
                                  self.sl_blocks)
            parts["sl"] = ch
            tot += ch

        pri = 0.0
        if not spec["free_offsets"]:
            for k in SURVEYS:
                p = decl.OFFSET_PRIORS[k]
                pri += ((o[k] - p["mean"]) / p["sd"]) ** 2
        parts["prior"] = pri
        tot += pri
        return (tot, parts) if want_parts else tot

    # ------------------------------------------------------------------ fit
    def fit(self, model, shape_grid=None):
        spec = MODELS[model]
        n_amp = (1 if spec["c"] else 0) + len(spec["amp"])
        n_nuis = 3 if self.fixed_scatter is None else 0
        x0 = np.zeros(n_amp + 3 + n_nuis)
        if n_nuis:
            x0[-3:] = math.log(0.25)

        def inner(shape, x0=x0):
            f = lambda p: self.m2lnL(model, shape, p)
            r = minimize(f, x0, method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-5, fatol=1e-5))
            r = minimize(f, r.x, method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
            return r

        if spec["shape"] is None:
            r = inner(None)
            return dict(model=model, shape=None, x=r.x, m2lnL=float(r.fun),
                        k=n_amp)
        best, curve = None, []
        for sh in shape_grid:
            r = inner(sh)
            curve.append((np.atleast_1d(sh).tolist(), float(r.fun)))
            if best is None or r.fun < best[1]:
                best = (sh, float(r.fun), r.x)
        nsh = len(np.atleast_1d(spec["shape"]))
        return dict(model=model, shape=np.atleast_1d(best[0]).tolist(),
                    x=best[2], m2lnL=best[1], k=n_amp + nsh, curve=curve)


def unpack(model, res):
    """Human-readable parameter dict from a fit result."""
    spec = MODELS[model]
    x = res["x"]
    i = 0
    out = {}
    if spec["c"]:
        out["c"] = float(x[i]); i += 1
    if "alpha" in spec["amp"]:
        out["alpha"] = float(x[i]); i += 1
    for j, k in enumerate(SURVEYS):
        out[f"offset_{k}"] = float(x[i + j])
    i += 3
    if len(x) > i:
        out["sigma_int_locuss"] = float(math.exp(x[i]))
        out["sigma_int_sl_within"] = float(math.exp(x[i + 1]))
        out["sigma_int_sl_cluster"] = float(math.exp(x[i + 2]))
    if res.get("shape") is not None:
        names = np.atleast_1d(spec["shape"]).tolist()
        for n, v in zip(names, np.atleast_1d(res["shape"])):
            out[n] = float(v)
    return out
