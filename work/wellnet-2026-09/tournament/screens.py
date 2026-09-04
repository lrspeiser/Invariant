"""STRUCTURAL SCREENS.  Every one of these is measured, none is asserted.

    asymptotic     d ln g / d ln r at r -> infinity.  The boundedness theorem
                   says a response confined to a bounded range leaves it at
                   -2; a flat rotation curve needs -1.  MEASURED, per
                   candidate, from the exact spherical reduction, so a bounded
                   form is convicted by its own number rather than by the
                   theorem's say-so.
    momentum       F_net = -oint T.n dS - (1/8 pi G) int (d_i K_jk) d_j Psi
                   d_k Psi, on two UNEQUAL masses, minus the Newtonian null on
                   the identical grid.  Reused from the screen lane
                   (fieldsolve.py + screen._identity_force), unmodified.  The
                   brief requires this to be reported for every survivor and a
                   momentum carrier or a variational completion declared.
    posdef         min eigenvalue and condition number of K over every probe.
    coarse_grain   drift of the response when the SAME smooth rho is described
                   by N = 1 ... 10^4 catalogue rows.  A response built from
                   the Poisson-smooth fields cannot see the partition at all;
                   one built from a row list can, and the screen lane's
                   discriminator d ln(drift)/d ln L separates a genuine
                   physical kernel (-3.11) from row-counting (+0.12).
    responsive     dS/dtheta != 0 for every headline statistic, over the
                   scanned range.  The programme has been bitten twice by
                   statistics that were exactly constant in the parameter they
                   were supposed to measure.
"""
from __future__ import annotations

import sys

import numpy as np

ROOT = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
        "work/wellnet-2026-09/")
for _p in (ROOT + "screen", ROOT + "tensor"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import families as FAM                                          # noqa: E402
import fieldsolve as FS                                         # noqa: E402
import screen as SCR                                            # noqa: E402
import wellnet as WN                                            # noqa: E402

from tw_core import (A0, G, KPC, MSUN, W_of, W_sup, mond_invert,   # noqa: E402
                     k_radial_pointwise)
from ch_radial import L_NL, M_NL                                 # noqa: E402


# ---------------------------------------------------------- 1. asymptotics
def asymptotic(cand, M=1e11 * MSUN, r_hi=1e5 * KPC, n=64):
    """d ln g / d ln r for an isolated point mass, at r ~ 1e5 scale radii.

    The response's OWN slope (B(r) = g/g_base) is reported beside the total,
    so a candidate that merely inherits MOND's -1 is not credited with it.
    """
    r = np.logspace(np.log10(1.0 * KPC), np.log10(r_hi), n)
    gN = G * M / r ** 2
    inv = dict(one=np.ones_like(r), gn=gN / A0,
               phi=G * M / r,
               rhobar=np.full_like(r, 1e-40),
               tidal=np.sqrt(6.0) * G * M / r ** 3,
               qbar=np.full_like(r, M / (M + M_NL)))
    if cand.form == "off" or cand.inv == "one":
        Wv = np.zeros_like(r)
    else:
        Wv = W_of(cand.form, inv[cand.inv] / cand.I0, cand.m)
    if cand.struct == "scalar_a0":
        a0e = cand.a0 * (1.0 + cand.A * Wv)
        g = mond_invert(gN, np.ones_like(r), a0e, cand.base)
    else:
        g = mond_invert(gN, k_radial_pointwise(cand, Wv), cand.a0, cand.base)
    g0 = mond_invert(gN, np.ones_like(r), cand.a0, cand.base)
    lg, lr = np.log(np.maximum(g, 1e-300)), np.log(r)
    sl = float((lg[-1] - lg[-5]) / (lr[-1] - lr[-5]))
    lb = np.log(np.maximum(g / g0, 1e-300))
    slb = float((lb[-1] - lb[-5]) / (lr[-1] - lr[-5]))
    return dict(slope_total=sl, slope_response=slb,
                W_sup=W_sup(cand.form),
                W_at_r_hi=float(Wv[-1]), W_at_r_lo=float(Wv[0]),
                flat_curve=bool(abs(sl + 1.0) < 0.05))


# ------------------------------------------------------------- 2. momentum
def _box_fields(box, rho, Mtot):
    """Phi_N, g_N, the Hessian, rho and the nonlocal mass on a screen-lane Box."""
    N = FS.solve_newton(rho, box, Mtot=Mtot)
    P = N["Psi"]
    gx, gy, gz = np.gradient(P, box.h, edge_order=2)
    gN = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    gv = np.stack([gx, gy, gz], -1)
    ghat = gv / np.maximum(gN, 1e-300)[..., None]
    H = np.empty(box.shape + (3, 3))
    for i, gi in enumerate((gx, gy, gz)):
        d = np.gradient(gi, box.h, edge_order=2)
        for j in range(3):
            H[..., i, j] = d[j]
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    tr3 = np.trace(H, axis1=-2, axis2=-1)[..., None, None] / 3.0
    D = H - tr3 * np.eye(3)
    ev = np.linalg.eigvalsh(D)
    spec = np.maximum(np.abs(ev).max(-1), 1e-300)
    That = D / spec[..., None, None]
    fro = np.sqrt((D ** 2).sum((-1, -2)))
    Mnl = np.full(box.shape, Mtot)          # L_NL = 300 kpc >> the 25 kpc pair
    inv = dict(one=np.ones(box.shape), gn=gN / A0, phi=np.abs(P),
               rhobar=np.maximum(rho, 1e-40), tidal=np.maximum(fro, 1e-45),
               qbar=Mnl / (Mnl + M_NL))
    return dict(PhiN=P, gN=gN, ghat=ghat, That=That, inv=inv)


def _K_field(cand, fld, box, wx, wm):
    if cand.form == "off" or cand.inv == "one":
        Wv = np.zeros(box.shape)
    else:
        Wv = W_of(cand.form, fld["inv"][cand.inv] / cand.I0, cand.m)
    a = np.clip(cand.A * Wv, -60.0, 60.0)
    I3 = np.eye(3)
    if cand.struct == "iso_K":
        E = np.exp(-a)[..., None, None] * I3
    elif cand.struct == "tensor_T":
        E = _expm_sym(a[..., None, None] * fld["That"])
    else:
        if cand.struct == "tensor_d":
            d = fld["ghat"]
        else:
            S6 = WN.S_tensor(box.pts, wx, wm, family=cand.extra["well"]["family"],
                             p=cand.extra["well"]["p"], q=cand.extra["well"]["q"],
                             s=cand.extra["well"]["s"], L=cand.extra["well"]["L"],
                             exclude_nearest=cand.extra["well"]["exclude_nearest"])
            Sm = np.stack([np.stack([S6[:, 0], S6[:, 3], S6[:, 4]], -1),
                           np.stack([S6[:, 3], S6[:, 1], S6[:, 5]], -1),
                           np.stack([S6[:, 4], S6[:, 5], S6[:, 2]], -1)], -2)
            return _expm_sym(a.reshape(-1)[:, None, None]
                             * Sm).reshape(box.shape + (3, 3))
        P = np.einsum("...i,...j->...ij", d, d) - I3 / 3.0
        E = _expm_sym(a[..., None, None] * P)
    return E


def _expm_sym(Mx):
    w, V = np.linalg.eigh(Mx)
    return np.einsum("...ij,...j,...kj->...ik", V, np.exp(w), V)


#: The momentum test must be run WHERE THE RESPONSE IS ON.  A galaxy-scale
#: pair sits at |Phi_N| ~ 1e10 m^2/s^2, four orders below a Phi_0 = 1e12 gate,
#: so it measures nothing but the discretisation floor.  The primary
#: configuration is therefore a CLUSTER-scale pair at |Phi_N| ~ 9e11; the
#: galaxy-scale pair is kept as the secondary, where a null IS the expected
#: and correct answer.
CFG_CLUSTER = dict(M1=8e13 * MSUN, M2=2e13 * MSUN, d=500.0 * KPC,
                   sig=60.0 * KPC, Lbox=2400.0)
CFG_GALAXY = dict(M1=8e10 * MSUN, M2=2e10 * MSUN, d=25.0 * KPC,
                  sig=3.0 * KPC, Lbox=120.0)


def momentum(cand, n=40, cfg=None):
    """|F_net|/(G M1 M2/d^2) above the SAME BASE LAW's null on the same grid.

    Newton is not the right null for a MOND-based candidate: AQUAL and QUMOND
    are variational, so their exact net force is zero, but the two-solve
    discretisation leaves a residual of 1-2% of G M1 M2/d^2 (the screen lane
    measures 0.021 for AQUAL and 0.011 for QUMOND and passes them on declared
    variational grounds).  The response's violation is therefore measured
    against the base law with the response switched off, which shares every
    discretisation choice with it.
    """
    cfg = CFG_CLUSTER if cfg is None else cfg
    M1, M2, d, sig = cfg["M1"], cfg["M2"], cfg["d"], cfg["sig"]
    box = FAM.Box(n, cfg["Lbox"])
    c1 = np.array([-d * M2 / (M1 + M2), 0.0, 0.0])
    c2 = np.array([+d * M1 / (M1 + M2), 0.0, 0.0])
    rho = (FAM.gauss_rho(box.pts, M1, sig, c1)
           + FAM.gauss_rho(box.pts, M2, sig, c2)).reshape(box.shape)
    Mtot = M1 + M2
    rho = FAM.normalise_mass(rho, box.vol, Mtot)
    Fref = G * M1 * M2 / d ** 2
    null = FS.solve_newton(rho, box, Mtot=Mtot)
    null_rel = float(np.linalg.norm(FS.net_force(rho, null["Psi"], box)) / Fref)
    form = "rar" if cand.base == "rar" else "simple"
    if cand.base == "newton":
        base_rel = null_rel
    else:
        rb = FS.solve_qumond(rho, box, a0=cand.a0, Mtot=Mtot, form=form)
        base_rel = float(np.linalg.norm(FS.net_force(rho, rb["Psi"], box))
                         / Fref)
    fld = _box_fields(box, rho, Mtot)
    Wmax = 0.0 if (cand.form == "off" or cand.inv == "one") else float(
        np.max(W_of(cand.form, fld["inv"][cand.inv] / cand.I0, cand.m)))
    out = dict(newton_null_rel=null_rel, base_null_rel=base_rel, n=n,
               Lbox=cfg["Lbox"], W_max_on_config=Wmax,
               Phi_max=float(np.abs(fld["PhiN"]).max()))
    if cand.form == "off" or cand.inv == "one" or cand.A == 0.0:
        out.update(law_net_force_rel=base_rel, excess=0.0, kind="base-only",
                   variational=cand.base in ("aqual", "rar", "newton"),
                   note="no position-dependent response; AQUAL/QUMOND are "
                        "variational by construction, so the residual is "
                        "discretisation")
        return out
    if cand.struct == "scalar_a0":
        Wv = W_of(cand.form, fld["inv"][cand.inv] / cand.I0, cand.m)
        A0f = cand.a0 * (1.0 + cand.A * Wv)
        r = FS.solve_qumond(rho, box, a0=cand.a0, Mtot=Mtot, A0_field=A0f,
                            form=form)
        Fl = float(np.linalg.norm(FS.net_force(rho, r["Psi"], box)) / Fref)
        out.update(law_net_force_rel=Fl, excess=max(Fl - base_rel, 0.0),
                   kind="scalar_a0", variational=False,
                   surface_term_rel=None, gradK_term_rel=None)
        return out
    wx = np.stack([c1, c2])
    wm = np.array([M1, M2])
    K = _K_field(cand, fld, box, wx, wm).reshape(-1, 3, 3)
    ev = np.linalg.eigvalsh(K)
    cond = float((ev[:, 2] / np.maximum(ev[:, 0], 1e-300)).max())
    if ev.min() <= 0 or not np.isfinite(cond) or cond > 1e8:
        out.update(law_net_force_rel=float("nan"), excess=float("nan"),
                   kind=cand.struct, K_min_eig=float(ev.min()),
                   K_cond=cond,
                   note="K's condition number exceeds 1e8 on this "
                        "configuration, so there is no bounded solution to "
                        "compare forces with.  That is a statement about how "
                        "extreme the fitted amplitude is, not a solver "
                        "complaint -- the screen lane records the same "
                        "outcome for one pair-channel row.")
        return out
    r = FS.solve_K(rho, K, box, Mtot=Mtot)
    Fl = FS.net_force(rho, r["Psi"], box)
    Fid, Fsurf, Fvol = SCR._identity_force(K, r["Psi"], box)
    law_rel = float(np.linalg.norm(Fl) / Fref)
    out.update(law_net_force_rel=law_rel, excess=max(law_rel - null_rel, 0.0),
               variational=False,
               kind=cand.struct, K_min_eig=float(ev.min()), K_cond=cond,
               identity_net_force_rel=float(np.linalg.norm(Fid) / Fref),
               surface_term_rel=float(np.linalg.norm(Fsurf) / Fref),
               gradK_term_rel=float(np.linalg.norm(Fvol) / Fref),
               identity_agreement=float(np.linalg.norm(Fl - Fid)
                                        / max(np.linalg.norm(Fl), 1e-300)),
               solver_resid=float(r["resid"]))
    return out


# --------------------------------------------------------- 3. coarse graining
def coarse_grain(cand, Ns=(1, 10, 100, 1000, 10000), Mtot=1e14 * MSUN,
                 Rcl=1000.0 * KPC, probe=800.0 * KPC, seed=7):
    """Drift of the radial response when ONE smooth rho is cut into N rows.

    rho is held fixed; only the partition changes.  A response built from the
    Poisson-smooth fields (Phi_N, g_N, rho, the Hessian) never sees the
    partition, so its drift is exactly zero and that is a property of the
    construction, not a numerical accident.  A response built from a row list
    does see it.
    """
    if cand.struct != "tensor_S":
        return dict(drift=[0.0] * len(Ns), N=list(Ns), max_drift=0.0,
                    depends_on_catalogue=False,
                    note="response is a functional of the Poisson-smooth "
                         "fields only; the well list never enters")
    # ONE cloud, drawn once, cut N ways: the N = 10 partition is the first 10
    # rows of the N = 10^4 partition, so the series is a nested refinement of a
    # single mass distribution and not a sequence of independent resamples.
    # Total mass is held fixed at Mtot for every N, so any drift is caused by
    # the partition and by nothing else.
    rng = np.random.default_rng(seed)
    Nmax = max(Ns)
    u_all = rng.random(Nmax) ** (1.0 / 3.0)
    d_all = rng.normal(size=(Nmax, 3))
    d_all /= np.linalg.norm(d_all, axis=1, keepdims=True)
    x = np.array([probe, 0.0, 0.0])[None, :]
    ws = cand.extra["well"]
    out = []
    for N in Ns:
        wx = d_all[:N] * (u_all[:N] * Rcl)[:, None]
        wm = np.full(N, Mtot / N)
        S6 = WN.S_tensor(x, wx, wm, family=ws["family"], p=ws["p"], q=ws["q"],
                         s=ws["s"], L=ws["L"],
                         exclude_nearest=ws["exclude_nearest"])
        out.append(float(S6[0, 0]))
    out = np.asarray(out)
    dr = np.abs(np.diff(out))
    return dict(N=list(Ns), S_rr=[float(v) for v in out],
                drift=[float(v) for v in dr], max_drift=float(dr.max()),
                depends_on_catalogue=True,
                note="S is normalised, so the mass exponent cancels; only the "
                     "GEOMETRY of the partition can move it")


# ----------------------------------------------------------- 4. responsive
def responsive(values, label):
    """dS/dtheta != 0 over the scanned range -- mandatory since Runs K and L."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return dict(label=label, spread=0.0, responsive=False)
    return dict(label=label, spread=float(v.max() - v.min()),
                responsive=bool(v.max() - v.min() > 0.0))
