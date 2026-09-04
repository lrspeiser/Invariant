"""
Shared statistics + the radial definitions.

Statistics are Run AT's, verbatim where possible, so the two halves of the audit
are comparable.  The headline is a SLOPE, not a correlation -- AT.6 showed the
correlation coefficient saturates (an injected slope of -0.25 drives corr to
-0.92), so a correlation near -0.8 carries almost no information about size.
"""
from __future__ import annotations
import math

import numpy as np

import ingest as I

KPC, MPC, MSUN, G, A0 = I.KPC, I.MPC, I.MSUN, I.G, I.A0


# ---------------------------------------------------------------- statistics
def rank(a):
    """tie-corrected ranks.  Run AL found argsort(argsort(v)) breaks ties by
    array position; CLASH has exactly repeated radii (100/200/400/600 kpc), so
    that bug would fire here."""
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sm = np.zeros(len(cnt))
    np.add.at(sm, inv, r)
    return (sm / cnt)[inv]


def pear(u, w):
    u = np.asarray(u, float) - np.mean(u)
    w = np.asarray(w, float) - np.mean(w)
    d = math.sqrt(float(u @ u) * float(w @ w))
    return float(u @ w / d) if d > 0 else 0.0


def spear(u, w):
    return pear(rank(u), rank(w))


def ols_slope(x, y):
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def fe_slope(x, y, groups):
    """within-cluster (fixed-effects) slope: each cluster gets its own level."""
    g = np.asarray(groups)
    names = sorted(set(g.tolist()))
    D = np.column_stack([(g == c).astype(float) for c in names] + [np.asarray(x, float)])
    c, *_ = np.linalg.lstsq(D, np.asarray(y, float), rcond=None)
    return float(c[-1])


def var_decomp(y, groups):
    """total / between-cluster / within-cluster variance of y."""
    y = np.asarray(y, float)
    g = np.asarray(groups)
    names = sorted(set(g.tolist()))
    tot = float(np.var(y, ddof=0))
    mu = np.array([y[g == c].mean() for c in names])
    nn = np.array([(g == c).sum() for c in names], float)
    bet = float(np.sum(nn * (mu - y.mean()) ** 2) / len(y))
    return dict(total=tot, between=bet, within=tot - bet,
                between_fraction=bet / tot, within_fraction=(tot - bet) / tot,
                n_groups=len(names), n_points=len(y))


# ------------------------------------------------------- excess statistics
def excess_y(gb, go):
    """RAR residual, Run AT's statistic:  log10 nu_obs - log10 nu_RAR."""
    return I.rar_residual(gb, go)


def excess_a0(gb, go):
    """log10(a0_eff / a0_canonical) with a0_eff = g_obs^2 / g_bar.
    This is Lane 12's statistic -- the one that produced the record's CLASH
    numbers.  Deep-MOND form; the full RAR inversion agrees to 0.01 dex on the
    paired 100->600 kpc difference (checked in tests.py)."""
    return np.log10(np.asarray(go, float) ** 2 / np.asarray(gb, float) / A0)


# ------------------------------------------------------- radial definitions
def baryonic_mass(r, gb):
    """M_bar(<r) = g_bar r^2 / G, recovered exactly from the published table.
    No total mass anywhere."""
    return np.asarray(gb, float) * np.asarray(r, float) ** 2 / G / MSUN


def _loglog_interp(xs, ys, target, allow_extrap=True):
    """solve ys(x) = target by log-log linear interpolation on a sorted profile.
    Returns (x, extrapolated_flag)."""
    o = np.argsort(xs)
    lx, ly = np.log(np.asarray(xs)[o]), np.log(np.asarray(ys)[o])
    lt = math.log(target)
    if ly[0] > ly[-1]:                      # decreasing
        lx, ly = lx[::-1], ly[::-1]
    if ly[0] <= lt <= ly[-1]:
        return float(math.exp(np.interp(lt, ly, lx))), False
    if not allow_extrap:
        return float("nan"), True
    # power-law continuation off the nearer end
    if lt < ly[0]:
        s = (ly[1] - ly[0]) / (lx[1] - lx[0])
        return float(math.exp(lx[0] + (lt - ly[0]) / s)), True
    s = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    return float(math.exp(lx[-1] + (lt - ly[-1]) / s)), True


def baryon_radii(T, C, f_b=I.F_B):
    """Three baryon-only characteristic radii.  All use ONLY g_bar and r, i.e.
    only M_bar(<r) = g_bar r^2/G.  No total mass, no overdensity of a total
    mass, no lensing.  Each threshold is a GLOBAL constant (constraint 3).

      R_b,gas : mean enclosed baryon density = 500 rho_c(z) f_b   (Run AT's form)
      R_b,M   : M_bar(<R) = M_b*, a single global mass
      R_b,g   : g_bar(R)  = g_b*, a single global acceleration
    """
    nm, r, gb = T["name"], T["r"], T["gb"]
    names = sorted(set(nm.tolist()))
    Mb = baryonic_mass(r, gb)
    Mstar = float(np.median(Mb))                    # global constant
    gstar = float(np.median(gb))                    # global constant
    out = {}
    for c in names:
        m = nm == c
        rr, mm, gg = r[m], Mb[m], gb[m]
        z = C[c]["z"]
        # R_b,gas : M_bar(<R) = (4/3) pi 500 rho_c f_b R^3  ->  solve on rho_bar
        A = (4 / 3) * math.pi * 500 * I.rhoc(z) * f_b
        rho_bar = mm * MSUN / ((4 / 3) * math.pi * rr ** 3)
        Rg, xg = _loglog_interp(rr, rho_bar, A)
        RM, xM = _loglog_interp(rr, mm, Mstar)
        Rgg, xgg = _loglog_interp(rr, gg, gstar)
        out[c] = dict(Rb_gas=Rg, Rb_gas_extrap=xg,
                      Rb_M=RM, Rb_M_extrap=xM,
                      Rb_g=Rgg, Rb_g_extrap=xgg)
    return out, dict(Mb_star_Msun=Mstar, g_b_star=gstar)


def m500_from_tx(kT_keV, z):
    """M500c from an X-ray temperature via a GLOBAL M-T relation.
    Vikhlinin+2009 (ApJ 692, 1033) form:
        M500 E(z) = A (kT/5 keV)^alpha,  A = 3.02e14 h70^-1 Msun, alpha = 1.53
    Every parameter is global; the only per-object input is the measured kT_X,
    which is independent of the lensing.  A scaling-relation PROXY, not a
    per-cluster mass measurement -- labelled as such everywhere it is used."""
    Ez = math.sqrt(I.OM * (1 + z) ** 3 + I.OL)
    return 3.02e14 * (kT_keV / 5.0) ** 1.53 / Ez


def radial_definitions(T, C):
    """Every radial variable, as ARRAYS over the 84 points, plus a per-cluster
    normaliser dict so the null can rebuild them."""
    nm, r = T["name"], T["r"]
    Rb, thr = baryon_radii(T, C)
    norm = {}
    for c in sorted(C):
        norm[c] = dict(
            R500_lens=C[c]["R500_lens"],
            R500_xray=C[c]["R500_xray"],
            R500_TX=I.r_from_mass(m500_from_tx(C[c]["kT"], C[c]["z"]), C[c]["z"]),
            Rb_gas=Rb[c]["Rb_gas"], Rb_M=Rb[c]["Rb_M"], Rb_g=Rb[c]["Rb_g"])
    defs = {"r_physical": np.ones(len(r)) * MPC}     # 1 Mpc: a global constant
    for k in ("R500_lens", "R500_xray", "R500_TX", "Rb_gas", "Rb_M", "Rb_g"):
        defs["r_over_" + k] = np.array([norm[c][k] for c in nm])
    x = {k: np.log10(r / v) for k, v in defs.items()}
    return x, norm, Rb, thr
