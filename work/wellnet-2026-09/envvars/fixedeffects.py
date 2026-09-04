"""SYSTEM FIXED EFFECTS: does radial variation WITHIN one object follow the
proposed second variable?

    log nu = alpha_i + f(g_bar, r) + beta X_env          alpha_i per system

Run AD found every WITHIN-class beta zero or negative (+0.090 field galaxies,
-0.319 groups, -0.181 clusters) while only the POOLED sample gave +0.169, and
that a bare class index reproduces 99% of the pooled fit quality.  That is a
textbook Simpson's paradox and it is why the ladder is undecidable.  A
within-object estimator cannot be fooled that way: the per-system intercept
absorbs everything that distinguishes one object from another, including the
class label, the mass, the redshift and every selection effect, and beta is
then identified by RADIAL SHAPE alone.

WHAT PLAYS THE ROLE OF log nu HERE.  The observable is per-cluster, per-bin
tangential reduced shear from DECADE (Run AI), which is a projected, non-local
functional of the 3-D acceleration.  So the estimator is written in the forward
direction:

    g_pred(r) = g_RAR(g_b(r)) x 10^(alpha_i + beta X_env(r))   ->  Sigma, DSigma
             ->  g_t model,  compared with the measured g_t.

10^alpha_i is an exact multiplicative constant on Sigma and DeltaSigma, so the
per-system intercept is profiled in closed form and never enters the search.
f(g_bar, r) is the RAR; the residualised variables (suffix _perp) additionally
project a QUADRATIC in (log g_bar, log r) out of X_env, which is the flexible-
scalar-nuisance version of the same question.

MODELS
    B0   RAR x 10^alpha_i                    the null (per-system amplitude)
    B1   RAR x 10^(alpha_i + beta X)         one extra GLOBAL parameter
    C0   RAR x 10^A                          the within-class null
    C1   RAR x 10^(A + beta X)               within-class, cross-system leverage

BLIND PROTECTION, DECLARED BEFORE ANY RESIDUAL WAS EXAMINED
    systems sorted by eFEDS name, split alternately; EVEN ranks TRAIN, ODD
    ranks HELD OUT.  Identical to Run AI's declared split, reused unchanged.
    beta is fitted on TRAIN, FROZEN, and the held-out set is scored once.  The
    per-system intercepts are object-specific NUISANCE parameters and are
    refitted on the held-out objects -- they are not the hypothesis, and
    without them the within-object model is not defined on a new object.

THE SHARED-QUANTITY NULL IS MANDATORY AND IT FIRES.  Run AI measured
E[beta-hat | H0] = -0.0666 +- 0.0101 for potential depth, driven by noise in
the X-ray density fit alone.  Every variable here gets its own null with the
actual published errors on every density parameter, and every estimate is
quoted against its own null.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import numpy as np

import envvars as EV
from envvars import (MPC, MSUN, G, A0, PHI0, T0, RULES, PRIMARY_RULE,
                     EPS_PRIMARY, R_TRUNC_MPC, SIGMA_Z_ASSUMED)
import pipeline as P
import efeds_hsc as E
import decade_test as D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = {}

# the variables carried through every estimator
ENVVARS = ["x1", "x2", "x3", "x4a", "x4b", "x4d"]
# ... and the two competitors with NO environmental content whatever.  Within
# one object every internally-sourced environmental variable is a monotone
# function of radius, so a bare radius tilt is the null that the within-object
# design actually has to beat -- it is Run AI's M3 moved inside the object.
COMPET = ["xr", "xgb"]
VARS = ENVVARS + COMPET
VAR_LABEL = {
    "x1": "V1  potential depth      log10 DeltaPhi_b/Phi0",
    "x2": "V2  vector external      log10 |g_ext|/a0",
    "x3": "V3  directionless well   log10 W_eps/a0",
    "x4a": "V4a tidal magnitude      log10 |T~|/T0",
    "x4b": "V4b tidal shape          rho/<rho>",
    "x4d": "V4d external tidal       log10 |T~_ext|/T0",
    "xr": "--  COMPETITOR           log10 r/Mpc, a bare radius tilt",
    "xgb": "--  COMPETITOR           log10 g_bar/a0, a bare acceleration tilt",
}
BETA_GRID = np.linspace(-2.0, 2.0, 101)
DELTA_LIN = 0.25                       # +- step for the linearised derivative
CLIP_SD = 6.0                          # declared safeguard, see standardise()
R_IN_MPC = 0.05                        # inner freeze radius, DECLARED
R_OUT_MPC = R_TRUNC_MPC                # outer freeze = the projection cut
# Every variable is standardised, and the (log g_bar, log r) nuisance quadratic
# is fitted, over EXACTLY the radial range that enters the lensing projection.
# Doing it over the narrower 0.2-5 Mpc band where the shear points sit leaves
# the residualised variables uncontrolled between 5 and 8 Mpc, where the
# extrapolated quadratic can be several times its own in-range scatter and the
# projection still integrates the mass.
PROJ_RANGE_MPC = (R_IN_MPC, R_OUT_MPC)


def freeze_outside(X, r):
    """Hold every variable constant outside the resolved radial range.

    The solver grid runs from 5 kpc to 30 Mpc.  eFEDS resolves nothing inside
    about 25 kpc at the median redshift, the shear points start at 0.33 Mpc,
    and the projection is cut at 8 Mpc -- yet |T~| = sqrt(6)(g/r)|1-rho/<rho>|
    DIVERGES as r -> 0 for any cusped fit, so an unfrozen tidal variable lets
    the model multiply the innermost, entirely unconstrained, mass-free grid
    points by 10^(6 beta).  Freezing X at its value at 0.05 Mpc and at 8 Mpc
    removes that channel for every variable identically, including beta = 0.
    """
    i0 = int(np.searchsorted(r, R_IN_MPC * MPC))
    i1 = int(np.searchsorted(r, R_OUT_MPC * MPC))
    Y = X.copy()
    Y[..., :i0] = Y[..., i0][..., None]
    Y[..., i1:] = Y[..., i1 - 1][..., None]
    return Y


def standardise(X, mask):
    """Centre and scale a variable by its own pooled mean and sd.

    A global constant shift in X is exactly a global amplitude shift, and a
    global rescale is exactly a rescale of beta, so NOTHING physical changes.
    What does change is that beta becomes the response in dex of log g per ONE
    STANDARD DEVIATION of the variable -- directly comparable across four
    quantities with completely different units -- and that 10^(beta X) stays
    numerically sane instead of spanning 10^(-4.3 beta) as the raw |g_ext| does.

    Values are then clipped at +-6 sd.  The mean and sd are taken over the
    0.2-5 Mpc fit range, where every shear point lives; the clip therefore only
    touches the innermost and outermost points of the 5 kpc - 30 Mpc solver
    grid, where the enclosed baryonic mass is either zero or beyond the 8 Mpc
    projection cut and contributes nothing to any measured radius.  Without it
    the innermost grid point, where the cumulative M_b starts from exactly
    zero, overflows 10^(beta X).
    """
    v = X[mask]
    v = v[np.isfinite(v)]
    mu, sd = float(np.mean(v)), float(np.std(v))
    sd = sd if sd > 1e-12 else 1.0
    z = (X - mu) / sd
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip(z, -CLIP_SD, CLIP_SD), mu, sd


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def save():
    """Write the results JSON after every section, so that a crash in a later
    section does not cost the earlier ones."""
    with open(os.path.join(HERE, "envvars_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(RES, f, indent=1)


# ------------------------------------------------------------- forward model
def unit_shear(systems, obs, idx, X=None, beta=0.0, n_R=260, n_t=500,
               trunc=R_TRUNC_MPC):
    """Model tangential reduced shear at unit amplitude, for each system.

    Returns a list of arrays A_kj: what the model predicts with 10^alpha = 1.
    The truncation is FIXED for every model including beta = 0, so the
    comparison is like for like.
    """
    out = []
    for k in idx:
        s = systems[k]
        g = s.g_rar(s.g_b)
        if beta != 0.0 and X is not None:
            g = g * 10.0 ** (beta * X[k])
        S, dS, dM = P.sigma_from_g(s.r, g, obs.R[k], trunc, n_R, n_t)
        out.append(D.gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k], 1.0))
    return out


def derivs(systems, obs, idx, XX, A0t, delta=DELTA_LIN, **kw):
    """First and second differences of the forward model with respect to beta."""
    Ap = unit_shear(systems, obs, idx, XX, +delta, **kw)
    Am = unit_shear(systems, obs, idx, XX, -delta, **kw)
    D1 = [(p - m) / (2 * delta) for p, m in zip(Ap, Am)]
    D2 = [(p - 2 * a + m) / delta ** 2
          for p, a, m in zip(Ap, A0t, Am)]
    return D1, D2


def _pack(obs, idx, A):
    """Stack (A, y, sigma) with a per-system slice index."""
    y = np.concatenate([obs.gt[k] for k in idx])
    e = np.concatenate([obs.er[k] for k in idx])
    a = np.concatenate(A)
    n = np.array([len(obs.gt[k]) for k in idx])
    edge = np.concatenate([[0], np.cumsum(n)])
    return a, y, e, edge


def chi2_profiled(a, y, e, edge, per_system):
    """chi^2 after profiling an UNRESTRICTED linear amplitude.

    per_system=True gives one amplitude per object (the fixed-effects null),
    False gives one global amplitude (the within-class null).  The model is
    exactly linear in the amplitude -- 10^alpha multiplies Sigma and DeltaSigma
    identically -- so this projection is exact up to the reduced-shear
    correction, which is checked against the full nonlinear profile below.
    """
    w = 1.0 / e ** 2
    if per_system:
        tot = float(np.sum(y ** 2 * w))
        num = np.add.reduceat(a * y * w, edge[:-1])
        den = np.add.reduceat(a * a * w, edge[:-1])
        good = den > 0
        return tot - float(np.sum(num[good] ** 2 / den[good])), num, den
    num = float(np.sum(a * y * w))
    den = float(np.sum(a * a * w))
    return (float(np.sum(y ** 2 * w)) - num ** 2 / max(den, 1e-300),
            np.array([num]), np.array([den]))


def leverage(a0, d0, e, edge, per_system):
    """Fisher information for beta AFTER the amplitudes are profiled out.

    The per-system intercept absorbs any part of dA/dbeta that is proportional
    to A itself, so the identifiable design vector is D orthogonalised against
    A within each object.  For a variable that is CONSTANT inside an object
    that residual is exactly zero and beta is not identified at all -- which is
    the correct answer for V2, not a numerical failure.
    """
    w = 1.0 / e ** 2
    if per_system:
        num = np.add.reduceat(a0 * d0 * w, edge[:-1])
        den = np.add.reduceat(a0 * a0 * w, edge[:-1])
        k = np.repeat(num / np.maximum(den, 1e-300), np.diff(edge))
    else:
        k = np.sum(a0 * d0 * w) / max(np.sum(a0 * a0 * w), 1e-300)
    dt = d0 - k * a0
    F = float(np.sum(dt ** 2 * w))
    Ftot = float(np.sum(d0 ** 2 * w))
    return F, (F / Ftot if Ftot > 0 else 0.0)


def fit_beta_linear(A0_, D1_, D2_, obs, idx, per_system, grid=BETA_GRID):
    """beta-hat from  A(beta) = A0 + beta D1 + beta^2 D2 / 2.

    D1 and D2 are the central first and second differences of the FULL
    nonlinear forward model at +-DELTA_LIN, so the curvature that matters near
    the minimum is retained rather than assumed away.
    """
    a0, y, e, edge = _pack(obs, idx, A0_)
    d1 = np.concatenate(D1_)
    d2 = np.concatenate(D2_)
    c = np.array([chi2_profiled(a0 + b * d1 + 0.5 * b * b * d2,
                                y, e, edge, per_system)[0] for b in grid])
    F, lev = leverage(a0, d1, e, edge, per_system)
    sig = 1.0 / math.sqrt(F) if F > 0 else float("inf")
    bh = _beta_from_grid(c, grid)
    at_edge = bool(abs(bh) >= grid.max() - 1e-9)
    return bh, sig, c, dict(leverage_frac=float(lev), at_grid_edge=at_edge)


def fit_beta_exact(systems, obs, idx, X, per_system, grid, **kw):
    """beta-hat from the FULL nonlinear forward model on a beta grid."""
    c = []
    for b in grid:
        A = unit_shear(systems, obs, idx, X, b, **kw)
        a, y, e, edge = _pack(obs, idx, A)
        c.append(chi2_profiled(a, y, e, edge, per_system)[0])
    c = np.array(c)
    i = int(np.argmin(c))
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        h = grid[1] - grid[0]
        den = (y0 - 2 * y1 + y2)
        bh = grid[i] + 0.5 * h * (y0 - y2) / den if den > 0 else grid[i]
        sig = math.sqrt(2.0 * h ** 2 / den) if den > 0 else float("inf")
    else:
        bh, sig = float(grid[i]), float("inf")
    return float(bh), float(sig), c


# --------------------------------------------------------- the declared split
def declared_split(obs):
    """Sort by eFEDS name, alternate: EVEN ranks TRAIN, ODD ranks HELD OUT."""
    order = np.argsort([r["id"] for r in obs.sys])
    train = np.array(sorted(order[0::2]))
    test = np.array(sorted(order[1::2]))
    assert len(set(train) & set(test)) == 0
    assert len(train) + len(test) == len(obs)
    return train, test


# ------------------------------------- the shared-quantity null, per variable
def _perturb_recs(recs, rng, z_jitter=True):
    """Redraw every published density parameter and, optionally, z."""
    out = []
    for r in recs:
        q = dict(r)
        q["n0sq"] = max(r["n0sq"] + rng.normal() * r["e_n0sq"], 1e-6)
        q["rs"] = max(r["rs"] + rng.normal() * r["e_rs"], 1e-3 * MPC)
        q["eps"] = r["eps"] + rng.normal() * r["e_eps"]
        q["beta"] = max(r["beta"] + rng.normal() * r["e_beta"], 0.34)
        q["alpha"] = r["alpha"] + rng.normal() * r["e_alpha"]
        q["z_pert"] = max(r["z"] + (rng.normal() * SIGMA_Z_ASSUMED
                                    * (1.0 + r["z"]) if z_jitter else 0.0),
                          1e-3)
        out.append(q)
    return out


def radial_mask(sys_list):
    lr = np.array([np.log10(s.r / MPC) for s in sys_list])
    return (lr >= math.log10(PROJ_RANGE_MPC[0])) & \
           (lr <= math.log10(PROJ_RANGE_MPC[1]))


def _env_of(sys_list, wells, varkeys, eps=EPS_PRIMARY):
    """The requested environmental variables for a list of System objects."""
    out = {k: [] for k in varkeys}
    for s in sys_list:
        gext, Wext, Text = EV.external_fields(s, wells, eps)
        need_self_W = ("x3" in varkeys)
        Wself = EV.v3_self(s, eps) if need_self_W else 0.0
        Tself, q, _, _ = EV.v4_self(s)
        v = dict(x1=lambda: EV.v1_potential_depth(s, PRIMARY_RULE)[0],
                 x2=lambda: np.log10(np.maximum(gext, 1e-300) / A0),
                 x3=lambda: np.log10(np.maximum(Wself + Wext, 1e-300) / A0),
                 x4a=lambda: np.log10(np.maximum(
                     np.sqrt(Tself ** 2 + Text ** 2), 1e-300) / T0),
                 x4b=lambda: q,
                 x4d=lambda: np.log10(np.maximum(Text, 1e-300) / T0),
                 xr=lambda: np.log10(s.r / MPC),
                 xgb=lambda: np.log10(np.maximum(s.g_b, 1e-30) / A0))
        for kk in varkeys:
            out[kk].append(v[kk]())
    return {k: freeze_outside(np.array(vv), sys_list[0].r)
            for k, vv in out.items()}


def _shear_at(sys_list, obs, idx, X, b, n_R, n_t):
    out = []
    for j, k in enumerate(idx):
        s = sys_list[j]
        g = s.g_rar(s.g_b)
        if b != 0.0 and X is not None:
            g = g * 10.0 ** (b * X[j])
        S, dS, _ = P.sigma_from_g(s.r, g, obs.R[k], R_TRUNC_MPC, n_R, n_t)
        out.append(D.gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k], 1.0))
    return out


def _beta_from_grid(c, grid=BETA_GRID):
    i = int(np.argmin(c))
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        h = grid[1] - grid[0]
        den = (y0 - 2 * y1 + y2)
        if den > 0:
            return float(grid[i] + 0.5 * h * (y0 - y2) / den)
    return float(grid[i])


def simulate_null(recs_all, obs, idx, varkeys, n_mc, seed, amp,
                  beta_inject=0.0, inject_key=None, n_R=160, n_t=300,
                  eps=EPS_PRIMARY, label="", do_perp=True, raw=False):
    """E[beta-hat | H0] for every variable and both estimators.

    TRUTH uses the PUBLISHED density parameters; the ANALYSIS uses a redraw of
    every one of them.  The SAME redraw builds the baseline model and the
    environmental variable, which is exactly the shared-quantity channel: on
    this sample Run AI measured -0.0666 +- 0.0101 of pure X-ray fit noise.
    """
    ids = [obs.sys[k]["id"] for k in idx]
    by_id = {r["id"]: r for r in recs_all}
    sys_true = [P.System(by_id[i], 0.0) for i in ids]
    rm = radial_mask(sys_true)
    inj = None
    if beta_inject != 0.0:
        inj = np.array(_env_of(sys_true, EV.Wells(recs_all, 0.0),
                               [inject_key], eps)[inject_key])
        inj = standardise(inj, rm)[0]
    A_true = _shear_at(sys_true, obs, idx, inj, beta_inject, n_R, n_t)

    rng = np.random.default_rng(seed)
    tags = ["raw", "perp"] if do_perp else ["raw"]
    keys = [f"{k}_{t}" for k in varkeys for t in tags]
    acc = {k: {"within_object": [], "within_class": []} for k in keys}
    ee = np.concatenate([obs.er[k] for k in idx])
    nper = np.array([len(obs.gt[k]) for k in idx])
    edge = np.concatenate([[0], np.cumsum(nper)])
    rg = sys_true[0].r
    t0 = time.time()
    for it in range(n_mc):
        y = np.concatenate([amp * a for a in A_true]) \
            + rng.normal(size=ee.size) * ee
        pr = _perturb_recs(recs_all, rng)
        dz = np.array([q["z_pert"] - q["z"] for q in pr])
        wells2 = EV.Wells(pr, 0.0, dz=dz)
        by_id2 = {r["id"]: r for r in pr}
        sys2 = [P.System(by_id2[i], 0.0) for i in ids]
        Xs = _env_of(sys2, wells2, varkeys, eps)
        Xs["lgb"] = np.array([np.log10(np.maximum(s.g_b, 1e-30)) for s in sys2])
        Xs["lr"] = np.array([np.log10(s.r / MPC) for s in sys2])
        A0l = _shear_at(sys2, obs, idx, None, 0.0, n_R, n_t)
        a0 = np.concatenate(A0l)
        # the residualisation is REDONE inside the realisation, exactly as an
        # analyst would have to do it, so the perp null carries the noise in
        # the nuisance fit as well
        if do_perp:
            _, cf = EV.collinearity(Xs, varkeys, rm)
            Xperp = EV.residualise(Xs, varkeys, cf)
        for k in varkeys:
            for tag in tags:
                src = Xs[k] if tag == "raw" else Xperp[k]
                Xk = standardise(freeze_outside(np.array(src), rg), rm)[0]
                Ap = _shear_at(sys2, obs, idx, Xk, +DELTA_LIN, n_R, n_t)
                Am = _shear_at(sys2, obs, idx, Xk, -DELTA_LIN, n_R, n_t)
                d1 = np.concatenate([(p - m) / (2 * DELTA_LIN)
                                     for p, m in zip(Ap, Am)])
                d2 = np.concatenate([(p - 2 * a + m) / DELTA_LIN ** 2
                                     for p, a, m in zip(Ap, A0l, Am)])
                for ps, nm in ((True, "within_object"),
                               (False, "within_class")):
                    c = np.array([chi2_profiled(a0 + b * d1 + 0.5 * b * b * d2,
                                                y, ee, edge, ps)[0]
                                  for b in BETA_GRID])
                    acc[f"{k}_{tag}"][nm].append(_beta_from_grid(c))
        if it == 0:
            print(f"      {label}one realisation {time.time()-t0:.1f} s; "
                  f"{n_mc} -> about {(time.time()-t0)*n_mc/60:.1f} min")
    if raw:
        return acc
    out = {}
    for k in keys:
        out[k] = {}
        for name in ("within_object", "within_class"):
            v = np.array(acc[k][name])
            out[k][name] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                sem=float(v.std(ddof=1) / math.sqrt(len(v))),
                                n=int(len(v)),
                                frac_at_grid_edge=float(np.mean(
                                    np.abs(v) >= BETA_GRID.max() - 1e-9)))
    return out


# --------------------------------------------------------------------- driver
def main(n_mc=60, n_inj=12):
    hdr("0.  DATA, VARIABLES AND THE DECLARED SPLIT")
    recs, obs, systems = EV.load_all(0.0)
    wells = EV.Wells(recs, 0.0)
    V = EV.build_variables(systems, wells, EPS_PRIMARY, PRIMARY_RULE)
    rr = V["lr"]
    mask = (rr >= math.log10(PROJ_RANGE_MPC[0])) & \
           (rr <= math.log10(PROJ_RANGE_MPC[1]))
    V["xr"] = V["lr"].copy()
    V["xgb"] = V["lgb"] - math.log10(A0)
    rgrid = systems[0].r
    for k in VARS:
        V[k] = freeze_outside(V[k], rgrid)
    col, coefs = EV.collinearity(V, ENVVARS, mask)
    Vp = EV.residualise(V, ENVVARS, coefs)
    for k in ENVVARS:
        # the (log g_bar, log r) quadratic is unbounded outside the range it
        # was fitted on, so the residual has to be frozen at the same radii
        Vp[k] = freeze_outside(Vp[k], rgrid)
    for k in COMPET:                 # a competitor has no residualised form
        Vp[k] = V[k].copy()
    scale = {}
    for k in VARS:
        V[k], mu, sd = standardise(V[k], mask)
        Vp[k], mu2, sd2 = standardise(Vp[k], mask)
        nclip = int(np.sum(np.abs(V[k]) >= CLIP_SD - 1e-9))
        nclipp = int(np.sum(np.abs(Vp[k]) >= CLIP_SD - 1e-9))
        scale[k] = dict(mean=mu, sd=sd, perp_mean=mu2, perp_sd=sd2,
                        n_clipped=nclip, n_clipped_perp=nclipp,
                        n_total=int(V[k].size),
                        n_clipped_inside_fit_range=int(
                            np.sum((np.abs(V[k]) >= CLIP_SD - 1e-9) & mask)),
                        n_clipped_perp_inside_fit_range=int(
                            np.sum((np.abs(Vp[k]) >= CLIP_SD - 1e-9) & mask)))
        print(f"   {k:4s} standardised: mean {mu:+9.4f}, sd {sd:7.4f}   "
              f"(residualised sd {sd2:7.4f})   clipped raw {nclip}, "
              f"perp {nclipp}, of {V[k].size}")
    RES["standardisation"] = scale
    train, test = declared_split(obs)
    allk = np.arange(len(obs))
    npts = {n: int(sum(len(obs.gt[k]) for k in i))
            for n, i in (("all", allk), ("train", train), ("test", test))}
    print(f"   declared split: {len(train)} train systems "
          f"({npts['train']} points), {len(test)} held out "
          f"({npts['test']} points)")
    RES["split"] = dict(n_train=int(len(train)), n_test=int(len(test)),
                        points=npts, rule="sorted by eFEDS ID, even=train, "
                                           "odd=held out; identical to Run AI")

    save()
    hdr("1.  LINEARISATION AND AMPLITUDE-PROFILE CHECKS")
    A0_all = unit_shear(systems, obs, allk)
    a, y, e, edge = _pack(obs, allk, A0_all)
    chi_fe, num, den = chi2_profiled(a, y, e, edge, True)
    chi_cl, numc, denc = chi2_profiled(a, y, e, edge, False)
    amp_cl = numc[0] / denc[0]
    amp_fe = num / np.maximum(den, 1e-300)
    print(f"   global amplitude (within-class)  10^A = {amp_cl:.4f}  "
          f"-> A = {math.log10(max(amp_cl,1e-9)):+.4f} dex")
    print(f"   per-system amplitudes: median {np.median(amp_fe):.3f}, "
          f"{np.mean(amp_fe < 0)*100:.1f}% negative "
          f"(expected: the per-object S/N is far below 1)")
    print(f"   chi2  B0 (per-system amp) {chi_fe:.2f} on {npts['all']} points, "
          f"C0 (global amp) {chi_cl:.2f}")
    # exact nonlinear global amplitude, through the full reduced-shear model
    g = np.linspace(-0.5, 1.0, 25)
    cex = []
    for aa in g:
        tot = 0.0
        for j, k in enumerate(allk):
            s = systems[k]
            gg = s.g_rar(s.g_b) * 10.0 ** aa
            S, dS, _ = P.sigma_from_g(s.r, gg, obs.R[k], R_TRUNC_MPC, 160, 300)
            p = D.gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k], 1.0)
            tot += float(np.sum(((p - obs.gt[k]) / obs.er[k]) ** 2))
        cex.append(tot)
    i = int(np.argmin(cex))
    amp_exact = 10.0 ** g[i]
    print(f"   exact nonlinear global amplitude 10^A = {amp_exact:.4f} "
          f"(linear closed form {amp_cl:.4f}); kappa << 1 so the two agree")
    kap_max = 0.0
    for j, k in enumerate(allk):
        s = systems[k]
        S, dS, _ = P.sigma_from_g(s.r, s.g_rar(s.g_b) * amp_cl, obs.R[k],
                                  R_TRUNC_MPC, 160, 300)
        kap_max = max(kap_max, float(np.max(S * obs.scinv[k])))
    print(f"   max convergence kappa anywhere in the sample: {kap_max:.4f}")
    RES["amplitude_checks"] = dict(
        global_amp_linear=float(amp_cl), global_amp_exact=float(amp_exact),
        chi2_B0=float(chi_fe), chi2_C0=float(chi_cl), kappa_max=float(kap_max),
        frac_negative_per_system_amp=float(np.mean(amp_fe < 0)),
        n_points=npts["all"])

    save()
    hdr("2.  THE FOUR VARIABLES, WITHIN-OBJECT AND WITHIN-CLASS, on TRAIN")
    print("   beta is in dex of log g per ONE STANDARD DEVIATION of the "
          "variable.")
    print("   'lev' is the fraction of the variable's Fisher information that "
          "survives\n   the amplitude projection: 0 means beta is not "
          "identified at all.\n")
    A0t = [A0_all[k] for k in train]
    fits = {}
    for key in VARS:
        for tag, XX in (("raw", V[key]), ("perp", Vp[key])):
            t0 = time.time()
            D1, D2 = derivs(systems, obs, train, XX, A0t)
            row = {}
            for ps, name in ((True, "within_object"), (False, "within_class")):
                bh, sg, cc, dg = fit_beta_linear(A0t, D1, D2, obs, train, ps)
                a_, y_, e_, ed_ = _pack(obs, train, A0t)
                c0 = chi2_profiled(a_, y_, e_, ed_, ps)[0]
                dchi = c0 - cc.min()
                row[name] = dict(beta=bh, sigma=sg, dchi2=float(dchi),
                                 dBIC=float(-dchi + math.log(npts["train"])),
                                 **dg)
            fits[(key, tag)] = row
            wo, wc = row["within_object"], row["within_class"]
            print(f"   {key:4s} {tag:4s}  WO beta {wo['beta']:+7.4f} "
                  f"+- {min(wo['sigma'],99.9):6.4f} lev {wo['leverage_frac']:.2e} "
                  f"dchi2 {wo['dchi2']:6.2f}  |  "
                  f"WC beta {wc['beta']:+7.4f} "
                  f"+- {min(wc['sigma'],99.9):6.4f} lev {wc['leverage_frac']:.2e} "
                  f"dchi2 {wc['dchi2']:6.2f}   [{time.time()-t0:.0f}s]")
    RES["fits_train"] = {f"{k}_{t}": v for (k, t), v in fits.items()}

    save()
    hdr("3.  EXACT vs LINEARISED, on the primary variables")
    exact = {}
    for key in ("x1", "x3", "x4a"):
        grid = np.linspace(-1.0, 1.0, 11)
        for ps, name in ((True, "within_object"), (False, "within_class")):
            bh, sg, _ = fit_beta_exact(systems, obs, train, V[key], ps, grid,
                                       n_R=160, n_t=300)
            exact[f"{key}_{name}"] = dict(beta=bh, sigma=sg)
            print(f"   {key:4s} {name:14s}  exact beta {bh:+7.4f} "
                  f"vs linearised {fits[(key,'raw')][name]['beta']:+7.4f}")
    RES["exact_vs_linearised"] = exact

    save()
    hdr("4.  THE SHARED-QUANTITY NULL, one per variable and per estimator")
    print("   Redraw of every published density parameter (n0^2, rs, epsilon,")
    print("   beta, alpha) plus an assumed sigma_z = 0.005(1+z); the SAME")
    print("   redraw builds the baseline model and the variable, the shear is")
    print("   regenerated independently.  Run AI measured -0.0666 +- 0.0101")
    print("   for potential depth on this sample: -6.6 sigma of pure fit noise.")
    parts = sorted(glob.glob(os.path.join(HERE, "null_part_*.json")))
    if parts:
        print(f"   merging {len(parts)} precomputed null slices")
        null = merge_partials(parts)
    else:
        null = simulate_null(recs, obs, train, VARS, n_mc, 20260904, amp_cl)
    RES["null"] = null
    for k in sorted(null):
        for name in ("within_object", "within_class"):
            d = null[k][name]
            print(f"   {k:9s} {name:14s}  E[beta|H0] = {d['mean']:+7.4f} "
                  f"+- {d['sem']:.4f}  (sd {d['sd']:.4f}, n {d['n']}, "
                  f"edge {d['frac_at_grid_edge']:.2f})")

    save()
    hdr("5.  RESPONSIVENESS GATE  d(beta-hat)/d(beta_injected), with the spread")
    resp = {}
    for key in ("x1", "x2", "x3", "x4a", "x4b"):
        rows = {}
        for name in ("within_object", "within_class"):
            rows[name] = []
        for binj in (0.0, 0.3):
            pp = sorted(glob.glob(os.path.join(
                HERE, f"inj_{key}_{int(round(binj*100))}_*.json")))
            if pp:
                s = merge_partials(pp)
            else:
                s = simulate_null(recs, obs, train, [key], n_inj,
                                  20260905 + int(binj * 10), amp_cl,
                                  beta_inject=binj, inject_key=key,
                                  label=f"{key} inj={binj}: ", do_perp=False)
            for name in ("within_object", "within_class"):
                rows[name].append((binj, s[f"{key}_raw"][name]["mean"],
                                   s[f"{key}_raw"][name]["sem"]))
        resp[key] = {}
        for name in ("within_object", "within_class"):
            (b0, m0, e0), (b1, m1, e1) = rows[name]
            slope = (m1 - m0) / (b1 - b0)
            resp[key][name] = dict(
                slope=float(slope),
                slope_err=float(math.hypot(e0, e1) / (b1 - b0)),
                at_0=float(m0), at_injected=float(m1), beta_injected=float(b1))
            print(f"   {key:4s} {name:14s}  d(beta-hat)/d(beta_inj) = "
                  f"{slope:+.4f} +- {resp[key][name]['slope_err']:.4f}   "
                  f"(beta-hat {m0:+.4f} -> {m1:+.4f} for injected 0 -> {b1})")
    RES["responsiveness"] = resp

    save()
    hdr("6.  FROZEN TRANSFER to the held-out half, touched once")
    trans = {}
    A0te = [A0_all[k] for k in test]
    a_, y_, e_, ed_ = _pack(obs, test, A0te)
    base = {ps: chi2_profiled(a_, y_, e_, ed_, ps)[0]
            for ps in (True, False)}
    for key in VARS:
        for tag, XX in (("raw", V[key]), ("perp", Vp[key])):
            row = {}
            for ps, name in ((True, "within_object"), (False, "within_class")):
                b = fits[(key, tag)][name]["beta"]
                A = unit_shear(systems, obs, test, XX, b)
                aa, yy, ee, edd = _pack(obs, test, A)
                c = chi2_profiled(aa, yy, ee, edd, ps)[0]
                row[name] = dict(beta_frozen=float(b),
                                 chi2_null=float(base[ps]), chi2_model=float(c),
                                 dchi2=float(base[ps] - c),
                                 dBIC=float(-(base[ps] - c)
                                            + math.log(npts["test"])))
            trans[f"{key}_{tag}"] = row
            print(f"   {key:4s} {tag:4s}  held-out dchi2  "
                  f"within-object {row['within_object']['dchi2']:+8.3f}  "
                  f"(dBIC {row['within_object']['dBIC']:+7.2f})   "
                  f"within-class {row['within_class']['dchi2']:+8.3f}  "
                  f"(dBIC {row['within_class']['dBIC']:+7.2f})")
    RES["frozen_transfer"] = trans

    save()
    hdr("7.  SENSITIVITY: the five boundary rules and three smoothing scales")
    sens = {}
    for ru in RULES:
        Xr, rrefs = [], []
        for s in systems:
            x, rref = EV.v1_potential_depth(s, ru)
            Xr.append(x)
            rrefs.append(rref / MPC)
        Xr = np.array(Xr)
        # DeltaPhi -> 0 at r_ref by construction, so the projection is cut at
        # 0.8 r_ref for EVERY model including beta = 0, exactly as in Run AI
        tr = min(R_TRUNC_MPC, 0.8 * float(np.median(rrefs)))
        m2 = mask & (V["lr"] < math.log10(tr))
        if m2.sum() < 1000:
            m2 = mask
        Xr = standardise(freeze_outside(Xr, rgrid), m2)[0]
        A0r = unit_shear(systems, obs, train, None, 0.0, trunc=tr)
        D1, D2 = derivs(systems, obs, train, Xr, A0r, trunc=tr)
        row = {"r_trunc_Mpc": float(tr),
               "r_ref_median_Mpc": float(np.median(rrefs))}
        for ps, name in ((True, "within_object"), (False, "within_class")):
            bh, sg, _, dg = fit_beta_linear(A0r, D1, D2, obs, train, ps)
            row[name] = dict(beta=bh, sigma=sg, **dg)
        sens[f"V1_{ru}"] = row
        print(f"   V1 rule {ru:11s} r_ref {np.median(rrefs):5.2f} Mpc, cut "
              f"{tr:4.2f}  WO {row['within_object']['beta']:+7.4f}"
              f"   WC {row['within_class']['beta']:+7.4f}")
    for ep, lab in ((EV.EPS_SENS[0], "20kpc"), (EPS_PRIMARY, "50kpc"),
                    (EV.EPS_SENS[1], "200kpc")):
        Xr = []
        for s in systems:
            gext, Wext, _ = EV.external_fields(s, wells, ep, full_tensor=False)
            Xr.append(np.log10(np.maximum(EV.v3_self(s, ep) + Wext,
                                          1e-300) / A0))
        Xr = standardise(freeze_outside(np.array(Xr), rgrid), mask)[0]
        D1, D2 = derivs(systems, obs, train, Xr, A0t)
        row = {}
        for ps, name in ((True, "within_object"), (False, "within_class")):
            bh, sg, _, dg = fit_beta_linear(A0t, D1, D2, obs, train, ps)
            row[name] = dict(beta=bh, sigma=sg, **dg)
        sens[f"V3_eps{lab}"] = row
        print(f"   V3 eps {lab:8s}                        "
              f"WO {row['within_object']['beta']:+7.4f}"
              f"   WC {row['within_class']['beta']:+7.4f}")
    RES["sensitivity"] = sens

    RES["collinearity"] = col
    RES["variable_labels"] = VAR_LABEL
    with open(os.path.join(HERE, "envvars_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(RES, f, indent=1)
    print("\n   wrote envvars_results.json")


def worker(out_path, n, seed, inject_key=None, beta_inject=0.0):
    """One slice of the Monte Carlo, written as raw per-realisation values.

    The realisations are independent, so the null is split across processes
    and merged by `merge_partials`.  Each slice gets its own seed."""
    recs, obs, systems = EV.load_all(0.0)
    A0_all = unit_shear(systems, obs, np.arange(len(obs)))
    a, y, e, edge = _pack(obs, np.arange(len(obs)), A0_all)
    amp_cl = chi2_profiled(a, y, e, edge, False)[1][0] /         chi2_profiled(a, y, e, edge, False)[2][0]
    train, _ = declared_split(obs)
    keys = VARS if inject_key is None else [inject_key]
    acc = simulate_null(recs, obs, train, keys, n, seed, amp_cl,
                        beta_inject=beta_inject, inject_key=inject_key,
                        do_perp=(inject_key is None), raw=True,
                        label=f"worker seed {seed}: ")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(n=n, seed=seed, inject_key=inject_key,
                       beta_inject=beta_inject, amp=float(amp_cl), acc=acc),
                  f)
    print(f"wrote {out_path}")


def merge_partials(paths):
    acc = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        for k, v in d["acc"].items():
            for nm, vals in v.items():
                acc.setdefault(k, {}).setdefault(nm, []).extend(vals)
    out = {}
    for k, v in acc.items():
        out[k] = {}
        for nm, vals in v.items():
            a = np.array(vals)
            out[k][nm] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                              sem=float(a.std(ddof=1) / math.sqrt(len(a))),
                              n=int(len(a)),
                              frac_at_grid_edge=float(np.mean(
                                  np.abs(a) >= BETA_GRID.max() - 1e-9)))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # worker <out.json> <n> <seed> [inject_key] [beta_inject]
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
               sys.argv[5] if len(sys.argv) > 5 else None,
               float(sys.argv[6]) if len(sys.argv) > 6 else 0.0)
    else:
        n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 60
        n_inj = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        main(n_mc, n_inj)
