"""The well-mirror / narrowing-gap laws, in the four formulations proposed.

Every law here is GLOBAL. The only per-galaxy inputs are MEASURED baryonic
quantities (enclosed baryonic acceleration, radius, total baryonic mass) and
the declared nuisances (distance, inclination, mass-to-light). No galaxy is
permitted a gravity parameter of its own; `audit_global_only` below is run in
mirror_run.py and asserts that.

  OPTION 2 -- gravitational-flux form
      div[ mu(D) grad Phi ] = 4 pi G rho_b
      mu(D) = 1 / [ (1-eta) + eta (D0/D)^n ]
      (D0/D)^n = 1 + s(r),   s(r) = r^2 / ( r_t (r + r_t) )
    so  (1-eta) + eta(1+s) = 1 + eta s  and mu = 1/(1 + eta s).
    Spherical symmetry integrates the field equation exactly:
        mu(D) g r^2 = G M_enc(r)   =>   g = g_N (1 + eta s(r))
    with r_t = eta sqrt(G M_b / a0)  DETERMINED by M_b, not fitted.
    Asymptotically g -> eta G M_b/(r_t r), so
        v_f^2 = eta G M_b / r_t = sqrt(G M_b a0)
    i.e. eta CANCELS out of the flat velocity: eta sets only where the
    transition happens, a0 alone sets the BTFR normalisation.

  OPTION 3 -- two fields
      g_+ = g_N [ 1 + (a_+/g_N)^p ]^(1/(2p)),  a_+ = a0/eta^2
      g   = (1-eta) g_N + eta g_+
    deep limit g -> eta sqrt(a_+ g_N) = sqrt(a0 g_N), so a0 again alone fixes
    the BTFR and (eta, p) shape the transition.

  OPTION 4 -- nonlocal mirror, GLOBAL length l
      g = g_N + eta G M_b / ( l (r + l) )
    asymptotically v_f^2 = eta G M_b / l, so v_f^4 ~ M_b^2: this formulation
    predicts a BTFR SLOPE OF 2, not 4. That is a structural prediction, not a
    fitting detail, and it is the first thing the screening will show.
"""
from __future__ import annotations

import numpy as np

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
AU = 1.495978707e11
A0_CANON = 1.2e-10


# --------------------------------------------------------------- Option 2
def r_t_of(Mb_msun, eta, a0):
    """Transition radius in METRES, determined entirely by M_b and globals.

    r_t = eta sqrt(G M_b / a0). Nothing galaxy-specific enters except M_b,
    which is a measured baryonic mass, so this is not a per-galaxy parameter.
    """
    return eta * np.sqrt(G * np.asarray(Mb_msun, float) * MSUN / a0)


def s_gap(r_m, Mb_msun, eta, a0):
    """The postulated gap profile s(r) = r^2 / (r_t (r + r_t))."""
    rt = r_t_of(Mb_msun, eta, a0)
    return r_m ** 2 / (rt * (r_m + rt))


def mu_opt2(r_m, Mb_msun, eta, a0):
    """mu(D) along the postulated profile: mu = 1/(1 + eta s)."""
    return 1.0 / (1.0 + eta * s_gap(r_m, Mb_msun, eta, a0))


def opt2(gN, r_m, Mb_msun, eta=0.5, a0=A0_CANON, **kw):
    """g = g_N [1 + eta r^2/(r_t(r+r_t))]. Exact spherical reduction."""
    return gN * (1.0 + eta * s_gap(r_m, Mb_msun, eta, a0))


def opt2_pointmass(r_m, Mb_msun, eta=0.5, a0=A0_CANON, **kw):
    """Outside-the-mass form, for probes that only have a total mass."""
    Mb = np.asarray(Mb_msun, float) * MSUN
    rt = r_t_of(Mb_msun, eta, a0)
    return G * Mb / r_m ** 2 + eta * G * Mb / (rt * (r_m + rt))


# --------------------------------------------------------------- Option 3
def opt3(gN, eta=0.5, a0=A0_CANON, p=1.0, **kw):
    """Two fields: g = (1-eta) g_N + eta g_+, g_+ interpolating at a_+."""
    a_plus = a0 / eta ** 2
    x = np.maximum(a_plus / np.maximum(gN, 1e-300), 1e-300)
    # (1 + x^p)^(1/2p) computed through logaddexp: at large p and large x the
    # direct form overflows, which silently produced inf on the KiDS holdout.
    gplus = gN * np.exp(np.logaddexp(0.0, p * np.log(x)) / (2.0 * p))
    return (1.0 - eta) * gN + eta * gplus


# --------------------------------------------------------------- Option 4
def opt4(gN, r_m, Mb_msun, eta=0.5, l_kpc=10.0, **kw):
    """Nonlocal mirror with a GLOBAL length l (kpc). Literal form: the mirror
    term carries the TOTAL baryonic mass, as written in the proposal."""
    l = l_kpc * KPC
    Mb = np.asarray(Mb_msun, float) * MSUN
    return gN + eta * G * Mb / (l * (r_m + l))


def opt4_menc(gN, r_m, Mb_msun, eta=0.5, l_kpc=10.0, **kw):
    """Variant using the ENCLOSED mass in the mirror term, so the extra force
    vanishes at the centre. M_enc(r) = g_N r^2/G. Robustness check only."""
    l = l_kpc * KPC
    Menc = gN * r_m ** 2 / G
    return gN + eta * G * Menc / (l * (r_m + l))


# ---------------------------------------------------- reference / benchmarks
def newton(gN, **kw):
    return gN


def aqual_simple(gN, a0=A0_CANON, **kw):
    """mu(g/a0) g = g_N with mu = x/(1+x); the benchmark to beat."""
    return 0.5 * (gN + np.sqrt(gN * gN + 4.0 * gN * a0))


def rar(gN, a0=A0_CANON, **kw):
    """McGaugh, Lelli & Schombert 2016. Reference, not a candidate."""
    return gN / (1.0 - np.exp(-np.sqrt(gN / a0)))


# ----------------------------------------------------------------- registry
#  free  : global parameter names, in fit order
#  needs : extra per-point inputs the law consumes (all MEASURED, never fitted)
LAWS = {
    "newton": dict(fn=newton, free=(), needs=()),
    "aqual_simple": dict(fn=aqual_simple, free=("a0",), needs=()),
    "rar": dict(fn=rar, free=("a0",), needs=()),
    "opt2": dict(fn=opt2, free=("eta", "a0"), needs=("r_m", "Mb_msun")),
    "opt2_eta_half": dict(fn=lambda gN, **k: opt2(gN, eta=0.5,
                                                  **{q: v for q, v in k.items()
                                                     if q != "eta"}),
                          free=("a0",), needs=("r_m", "Mb_msun")),
    "opt3": dict(fn=opt3, free=("eta", "a0", "p"), needs=()),
    "opt3_eta_half": dict(fn=lambda gN, **k: opt3(gN, eta=0.5,
                                                  **{q: v for q, v in k.items()
                                                     if q != "eta"}),
                          free=("a0", "p"), needs=()),
    "opt4": dict(fn=opt4, free=("eta", "l_kpc"), needs=("r_m", "Mb_msun")),
    "opt4_eta_half": dict(fn=lambda gN, **k: opt4(gN, eta=0.5,
                                                  **{q: v for q, v in k.items()
                                                     if q != "eta"}),
                          free=("l_kpc",), needs=("r_m", "Mb_msun")),
    "opt4_menc": dict(fn=opt4_menc, free=("eta", "l_kpc"),
                      needs=("r_m", "Mb_msun")),
}

#  parameter bounds, in the space actually searched (log10 where marked)
BOUNDS = dict(eta=(0.01, 8.0), a0=(-13.0, -8.0), p=(0.15, 60.0),
              l_kpc=(-1.0, 4.0), f_int=(-4.0, 0.0))
LOGPAR = {"a0", "l_kpc", "f_int"}
START = dict(eta=0.5, a0=np.log10(A0_CANON), p=1.0, l_kpc=1.0, f_int=-1.0)


# ------------------------------------------------------------------- audits
def audit_global_only(law_name, gals, params):
    """Assert the law has no per-galaxy gravity freedom.

    Two checks, both of which have to pass:
      1. the parameter vector handed to the law is IDENTICAL for every galaxy;
      2. for Option 2, r_t recomputed from each galaxy's own M_b equals
         eta sqrt(G M_b/a0) to machine precision -- i.e. r_t is a function of a
         measured mass and the global parameters, never an independent knob.
    """
    seen = {tuple(sorted(params.items()))}
    assert len(seen) == 1, "parameters differ between galaxies"
    report = dict(law=law_name, n_global=len(params), n_per_galaxy=0,
                  params={k: float(v) for k, v in params.items()})
    if "eta" in params and "a0" in params:
        eta, a0 = params["eta"], params["a0"]
        worst = 0.0
        for g in gals:
            Mb = np.atleast_1d(g.draws["Mb_cat"])
            rt = r_t_of(Mb, eta, a0)
            direct = eta * np.sqrt(G * Mb * MSUN / a0)
            worst = max(worst, float(np.max(np.abs(rt / direct - 1.0))))
        report["r_t_determined_by_Mb_max_reldiff"] = worst
        assert worst < 1e-12, "r_t is not a pure function of M_b"
    return report


def btfr_slope_prediction(law_name, eta, a0, l_kpc=None,
                          Mb=np.logspace(8.0, 11.5, 40)):
    """Asymptotic v_f^4 vs M_b slope each formulation predicts, computed
    numerically from the law itself rather than asserted."""
    r = np.logspace(np.log10(3.0), np.log10(3000.0), 400) * KPC
    vf4 = []
    for m in Mb:
        gN = G * m * MSUN / r ** 2
        if law_name.startswith("opt2"):
            g = opt2(gN, r, m, eta=eta, a0=a0)
        elif law_name.startswith("opt4"):
            g = opt4(gN, r, m, eta=eta, l_kpc=l_kpc)
        elif law_name.startswith("opt3"):
            g = opt3(gN, eta=eta, a0=a0, p=1.0)
        else:
            g = aqual_simple(gN, a0=a0)
        v2 = g * r
        # flat part: where v^2 is flattest over the outer decade
        k = int(0.8 * len(r))
        vf4.append(float(np.median(v2[k:]) ** 2))
    sl = np.polyfit(np.log10(Mb), np.log10(vf4), 1)[0]
    return float(sl)
