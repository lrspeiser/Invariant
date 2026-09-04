"""FORMATION LANE, part 1 -- the linear response of each candidate about a
homogeneous baryonic background.

THE QUESTION THIS LANE EXISTS TO ANSWER.  A static equation that reproduces
present-day rotation curves does not show that the law helped matter organise.
Before anyone spends compute on a cosmological simulation, perturb a
homogeneous baryonic background, extract the linear response, and ask whether
the homogeneous state is even a sensible solution.

WHAT IS IMPORTED AND NOT MODIFIED.
    ../tensor/wellnet.py     S_tensor, sym3 algebra, gate_field, well_weights
    ../tournament/tw_core.py W_of, the response forms, mu/nu bases
    ../../gravitylab/solver.py  the face-flux finite-volume discipline; the
                             periodic operator here follows the same
                             conservative form but WRAPS, because a periodic
                             box is the right boundary condition for a
                             cosmological background and an open Dirichlet
                             shell is not.

NO OBSERVATIONAL DATA IS LOADED IN THIS LANE AT ALL.  There is therefore no
blind-protection issue: nothing is fitted, nothing is scored against a
holdout, and no split is consumed.  KiDS and wide binaries are not loaded, not
listed, and not referenced; neither is SPARC, neither are the cluster
catalogues.  Every constant used here is frozen from
../tournament/tournament.json and is quoted with its provenance.

THE FOUR STRUCTURAL FACTS THIS FILE ESTABLISHES.

1. THE FROZEN TIDAL GATE DOES NOT DIVERGE ON A HOMOGENEOUS BACKGROUND.  The
   brief anticipated f ~ (|T|/T_0)^-m blowing up at |T| = 0.  The form
   actually frozen by the tournament is `inv`, W = 1/(1 + I^m), which
   tw_core.W_sup reports as BOUNDED by 1.  At |T| = 0 the gate SATURATES at
   its maximum, W = 1, so a0 -> a0 (1 + A) = 17 a0.  The homogeneous state
   exists; the gate is fully ON there.

2. AND ITS LINEAR RESPONSE IS IDENTICALLY ZERO.  |T| is a norm of a quantity
   that vanishes in the background, so I = |T|/T_0 is O(|delta|) and
   W = 1 - I^2 + O(I^4) with m = 2 even.  dW/d delta = 0 at delta = 0.  The
   gate first acts at SECOND order.  At linear order the candidate is exactly
   AQUAL with a0 -> 17 a0 and nothing else.

3. BUT I IS NOT SMALL, SO THAT EXPANSION IS NOT THE RELEVANT ONE.  The gate's
   argument is I = |T|/T_0 with |T| ~ 4 pi G rhobar delta, and rhobar grows as
   a^-3.  At recombination I ~ 3 for delta ~ 1e-5.  The expansion parameter of
   the gate is I, not delta, and the two are unrelated in size.

4. THE MOND FAMILY HAS NO LINEAR REGIME AT ALL.  In the deep-MOND limit the
   source of the growth equation goes as delta^(1/2), which is NOT Lipschitz at
   delta = 0.  delta = 0 and delta = (C^2/144) t^4 both satisfy the same
   equation with the same initial data.  The homogeneous state is a solution
   but not an isolated one, so "linearise and read off an eigenvalue" is not
   an available operation for any candidate whose base law is AQUAL or QUMOND.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
ROOT = os.path.dirname(LANE)
for _p in (os.path.join(LANE, "tensor"), os.path.join(LANE, "tournament"),
           os.path.join(ROOT, "gravitylab")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wellnet as WN                                            # noqa: E402
from tw_core import W_of, W_sup, nu_rar                          # noqa: E402

G = 6.674e-11
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
MSUN = 1.98892e30

# ------------------------------------------------------------------ cosmology
#: Background expansion.  A MODIFIED POISSON EQUATION DOES NOT DETERMINE THE
#: BACKGROUND -- that needs the relativistic completion nobody has written for
#: these candidates.  Three defensible backgrounds are therefore carried and
#: the spread is reported, with `lcdm_bg_baryon_src` declared PRIMARY because
#: it is the one every published MOND-cosmology calculation uses: standard
#: Friedmann expansion (as the relativistic parents are built to reproduce),
#: with only baryons sourcing the perturbations.
H0_SI = 67.4e3 / MPC          # Planck-like h = 0.674
OMEGA_B = 0.0493
OMEGA_M_LCDM = 0.315
RHO_CRIT0 = 3.0 * H0_SI ** 2 / (8.0 * np.pi * G)
RHO_B0 = OMEGA_B * RHO_CRIT0

BACKGROUNDS = {
    # name: (Omega_m for H(a), Omega_L for H(a), Omega_src for the source)
    "eds_baryon_src": (1.0, 0.0, OMEGA_B),
    "lcdm_bg_baryon_src": (OMEGA_M_LCDM, 1.0 - OMEGA_M_LCDM, OMEGA_B),
    "baryon_only_flat": (OMEGA_B, 1.0 - OMEGA_B, OMEGA_B),
}
PRIMARY_BACKGROUND = "lcdm_bg_baryon_src"


def hubble(a, bg=PRIMARY_BACKGROUND):
    om, ol, _ = BACKGROUNDS[bg]
    return H0_SI * np.sqrt(om / a ** 3 + ol + (1.0 - om - ol) / a ** 2)


def rho_src(a, bg=PRIMARY_BACKGROUND):
    """Mean density of the matter that SOURCES perturbations, at scale a."""
    return BACKGROUNDS[bg][2] * RHO_CRIT0 / a ** 3


# ------------------------------------------------------- the interpolations
def mu_simple(x):
    """AQUAL's mu(x) = x/(1+x); the tournament's `aqual` base is its inverse."""
    x = np.maximum(np.asarray(x, float), 0.0)
    return x / (1.0 + x)


def Lmu_simple(x):
    """dln mu/dln x for mu_simple.  Deep MOND -> 1, Newtonian -> 0."""
    x = np.maximum(np.asarray(x, float), 0.0)
    return 1.0 / (1.0 + x)


def nu_simple(y):
    """g/g_N for AQUAL-simple: exactly tw_core.g_of_gN('aqual')/g_N."""
    y = np.maximum(np.asarray(y, float), 1e-300)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y))


def Lnu_simple(y):
    """dln nu/dln y, analytic.  Deep MOND -> -1/2, Newtonian -> 0."""
    y = np.maximum(np.asarray(y, float), 1e-300)
    s = np.sqrt(1.0 + 4.0 / y)
    return -2.0 / (s * y * (1.0 + s))


def Lnu_rar(y):
    """dln nu/dln y for the McGaugh nu.  Deep MOND -> -1/2."""
    y = np.maximum(np.asarray(y, float), 1e-300)
    r = np.sqrt(y)
    e = np.exp(-r)
    return -r * e / (2.0 * (1.0 - e))


NU = {"aqual": nu_simple, "qumond": nu_rar, "rar": nu_rar,
      "newton": lambda y: np.ones_like(np.asarray(y, float))}
LNU = {"aqual": Lnu_simple, "qumond": Lnu_rar, "rar": Lnu_rar,
       "newton": lambda y: np.zeros_like(np.asarray(y, float))}


# --------------------------------------------------------- frozen candidates
#: Every number below is copied from ../tournament/tournament.json (records[])
#: and ../tournament/focus.json (head_to_head[]).  `src` names the row.
CANDIDATES = {
    "tidal_scalar": dict(
        label="tidal-gated scalar  a0 -> a0(1 + A/(1+(|T|/T0)^2))",
        src="tournament.json  aqual|scalar_a0|tidal|inv|m2|I1e-33",
        base="aqual", struct="scalar_a0", inv="tidal", form="inv",
        m=2.0, I0=1.0e-33, A=16.0, a0=1.00230625e-10, k_params=4),
    "depth_S_p0": dict(
        label="well-network tensor S, p = 0, potential-depth gate",
        src="tournament.json  aqual|tensor_S[plaw_p0q1s2_L300]|phi|sat|m2|I1e+12",
        base="aqual", struct="tensor_S", inv="phi", form="sat",
        m=2.0, I0=1.0e12, A=-25.0, a0=1.0203437500000001e-10, k_params=4,
        well=dict(family="plaw", p=0.0, q=1.0, s=2.0, L=300.0 * KPC,
                  exclude_nearest=False)),
    "depth_S_p1_literal": dict(
        label="well-network tensor S, brief's literal p = 1, no self-exclusion",
        src="focus.json  aqual|tensor_S[plaw_p1q1s2_L300_literal]|sat m=2 Phi0=1e+12",
        base="aqual", struct="tensor_S", inv="phi", form="sat",
        m=2.0, I0=1.0e12, A=-31.0, a0=1.0580375e-10, k_params=4,
        well=dict(family="plaw", p=1.0, q=1.0, s=2.0, L=300.0 * KPC,
                  exclude_nearest=False)),
    "aqual": dict(
        label="AQUAL, variational reference",
        src="tournament.json  BASE_aqual",
        base="aqual", struct="none", inv="one", form="off",
        m=1.0, I0=1.0, A=0.0, a0=1.0580375e-10, k_params=1),
    "qumond": dict(
        label="QUMOND (nu of McGaugh et al.), variational reference",
        src="tournament.json  BASE_rar",
        base="qumond", struct="none", inv="one", form="off",
        m=1.0, I0=1.0, A=0.0, a0=1.0844e-10, k_params=1),
    "newton": dict(
        label="Newton, trivial control",
        src="tournament.json  BASE_newton",
        base="newton", struct="none", inv="one", form="off",
        m=1.0, I0=1.0, A=0.0, a0=3.0e-11, k_params=1),
}
ORDER = ["tidal_scalar", "depth_S_p0", "depth_S_p1_literal",
         "aqual", "qumond", "newton"]


# ============================================================================
# 1.  DOES A HOMOGENEOUS BACKGROUND EXIST, AND WHAT IS THE GATE THERE?
# ============================================================================
def homogeneous_state(name, a=1.0, bg=PRIMARY_BACKGROUND):
    """The candidate's state on an EXACTLY homogeneous baryonic background.

    Returns a dict with the invariant's background value, whether it is even
    defined, the gate value W, and the effective a0 / K.

    The Jeans swindle is used and declared: a modified Poisson equation on an
    infinite homogeneous medium has no translation-invariant solution for the
    ABSOLUTE potential (the standard problem, not special to these
    candidates), so the field variable is the PECULIAR potential with a
    vanishing spatial mean.  That fixes phi's gauge for AQUAL/QUMOND/Newton --
    where nothing depends on phi's value -- but it is exactly the choice a
    potential-depth gate is a function of, which is why depth-gated candidates
    are marked undefined below rather than evaluated.
    """
    c = CANDIDATES[name]
    out = dict(name=name, label=c["label"], src=c["src"], base=c["base"],
               struct=c["struct"], inv=c["inv"], form=c["form"],
               a=a, a0_fitted=c["a0"], A=c["A"], m=c["m"], I0=c["I0"])
    if c["form"] == "off":
        out.update(invariant_defined=True, invariant_bg=None, W_bg=0.0,
                   W_sup=0.0, a0_eff=c["a0"], K_bg="identity",
                   admits_background=True,
                   verdict="base law only; homogeneous state is the Jeans-swindle "
                           "solution phi = 0, exactly as in Newtonian cosmology")
        return out

    out["W_sup"] = W_sup(c["form"])
    if c["inv"] == "tidal":
        # |T| = |traceless Hessian of Phi_N|_F.  On an exactly homogeneous
        # background the Hessian is (4 pi G rhobar/3) delta_ij, whose traceless
        # part is IDENTICALLY ZERO.  The invariant is defined and equals 0.
        Ibg = 0.0
        W = float(W_of(c["form"], Ibg / c["I0"], c["m"]))
        out.update(invariant_defined=True, invariant_bg=Ibg, W_bg=W,
                   a0_eff=c["a0"] * (1.0 + c["A"] * W), K_bg="identity",
                   admits_background=True,
                   gate_diverges=False,
                   verdict="|T| = 0 identically on a homogeneous background.  The "
                           "frozen form is inv, W = 1/(1+I^m), BOUNDED by 1, so the "
                           "gate SATURATES at its maximum rather than diverging: "
                           "W = 1 and a0 -> a0 (1 + A).  The background exists.")
        return out

    if c["inv"] == "phi":
        # |Phi_N| is defined only up to a constant.  On a homogeneous
        # background there is no boundary and no reference point, so the gate's
        # argument is pure gauge.  Three defensible cosmological rules are
        # quoted and they do not agree.
        rules = {
            "jeans_swindle": 0.0,
            "hubble_patch": float(0.5 * hubble(a, bg) ** 2 * (2.998e8 / hubble(a, bg)) ** 2),
            "horizon_mass": float(G * (4.0 * np.pi / 3.0) * rho_src(a, bg)
                                  * (2.998e8 / hubble(a, bg)) ** 2),
        }
        Ws = {k: float(W_of(c["form"], v / c["I0"], c["m"]))
              for k, v in rules.items()}
        out.update(invariant_defined=False, invariant_bg=rules, W_bg=Ws,
                   a0_eff=None, K_bg="undefined",
                   admits_background=False,
                   dex_spread=float(np.log10(max(rules.values()) /
                                             max(min(v for v in rules.values()
                                                     if v > 0), 1e-300))),
                   verdict="|Phi_N| on a homogeneous background is PURE GAUGE.  The "
                           "three defensible cosmological reference rules span the "
                           "whole gate, from W = 0 to W = 1.  Run AH already measured "
                           "the two admissible galaxy rules differing by 0.87 dex "
                           "against a 0.9 dex off/on margin; in cosmology the "
                           "ambiguity is total, not 0.87 dex.")
        return out
    raise ValueError(c["inv"])


def _sym6_fro(S):
    return float(np.sqrt(np.sum(np.asarray(S) ** 2
                                * np.array([1., 1., 1., 2., 2., 2.]))))


def tensor_S_background_defined(nwell_list=(1, 10, 100, 1000, 10000, 100000),
                                nseed=12, seed0=20260904, box=100.0 * MPC,
                                L=30.0 * MPC):
    """Is S defined on a homogeneous background?  Measured, not asserted.

    S is a NORMALISED direction average over a DISCRETE well catalogue.  A
    homogeneous continuum has no wells: numerator and denominator both vanish
    and S = 0/0, regulated only by wellnet's eps_w.  With a finite random
    catalogue S is pure shot noise, |S| ~ N^(-1/2), so the response of a
    homogeneous universe is set by how finely the cataloguer chose to chop it
    up.  wellnet.S_tensor is called unmodified; the exponent is FITTED, not
    assumed, over `nseed` independent realisations per N.
    """
    rows = []
    for N in nwell_list:
        vals = []
        for j in range(nseed):
            rng = np.random.default_rng(seed0 + 1000 * j + N)
            wx = (rng.random((int(N), 3)) - 0.5) * box
            wm = np.full(int(N), 1.0e11 * MSUN)
            S = WN.S_tensor(np.zeros((1, 3)), wx, wm, family="plaw", p=1.0,
                            q=1.0, s=2.0, L=L, M_0=1.0e11 * MSUN,
                            exclude_nearest=False, eps_w=1e-12)
            vals.append(_sym6_fro(S[0]))
        rows.append(dict(N=int(N), S_norm_rms=float(np.sqrt(np.mean(
            np.array(vals) ** 2))), S_norm_sd=float(np.std(vals))))
    Ns = np.array([r["N"] for r in rows], float)
    Sv = np.array([r["S_norm_rms"] for r in rows], float)
    slope = (float(np.polyfit(np.log(Ns), np.log(Sv), 1)[0])
             if Ns.size > 1 else float("nan"))
    return dict(rows=rows, continuum_limit=0.0,
                shot_noise_slope=slope, shot_noise_slope_expected=-0.5,
                max_over_min=float(Sv.max() / max(Sv.min(), 1e-300)),
                eps_w_note="eps_w = 1e-12 so the 0/0 is not masked by the "
                           "regulariser; with wellnet's default 1e-3 the "
                           "empty-catalogue limit returns S = 0 by fiat",
                verdict="S on a homogeneous background is 0/0 in the continuum and "
                        "pure shot noise for any finite catalogue.  The response of "
                        "the homogeneous state is therefore set by the catalogue "
                        "resolution, which the field equation does not specify.")


# ============================================================================
# 2.  THE LINEAR RESPONSE ABOUT A BACKGROUND FIELD -- ANALYTIC
# ============================================================================
def response_tensor(base, gmag, a0, ghat=np.array([0.0, 0.0, 1.0])):
    """The tensor that multiplies grad(delta phi) in the linearised equation.

    AQUAL   K_ij = mu(x) [delta_ij + L_mu(x) ghat_i ghat_j],  x = |g|/a0
    QUMOND  K_ij = nu(y) [delta_ij + L_nu(y) nhat_i nhat_j],  y = |g_N|/a0
            (acting on grad phi_N rather than on grad phi)
    Newton  K = I

    Returned as a full 3x3, together with its eigenvalues.  A NEGATIVE
    eigenvalue would make the field equation non-elliptic and the theory
    ill-posed: that is the first stability question and it is checked
    numerically for every candidate rather than asserted.
    """
    ghat = np.asarray(ghat, float)
    ghat = ghat / np.linalg.norm(ghat)
    P = np.outer(ghat, ghat)
    if base == "newton":
        return np.eye(3), np.ones(3), 0.0
    if base == "aqual":
        x = gmag / a0
        pre, L = float(mu_simple(x)), float(Lmu_simple(x))
    else:
        y = gmag / a0
        pre, L = float(NU[base](y)), float(LNU[base](y))
    K = pre * (np.eye(3) + L * P)
    return K, np.array([pre, pre, pre * (1.0 + L)]), L


def mode_multiplier(base, gmag, a0, costheta):
    """Q(theta): the factor by which the growth source of a small plane-wave
    perturbation is multiplied, relative to Newton, in the presence of a
    UNIFORM background field g.  theta is the angle between k and g.

        AQUAL   Q = 1 / [ mu(x) (1 + L_mu cos^2 theta) ]
        QUMOND  Q = nu(y) (1 + L_nu cos^2 theta)

    Both reduce to nu at theta = 90 deg and to nu/2 in the deep-MOND limit at
    theta = 0, so THEY AGREE AT BOTH EXTREMES AND DIFFER IN BETWEEN.  That
    disagreement is a genuine, quantified difference between the two
    variational references, not a discretisation artefact.
    """
    c2 = np.asarray(costheta, float) ** 2
    if base == "newton":
        return np.ones_like(c2)
    if base == "aqual":
        x = gmag / a0
        return 1.0 / (mu_simple(x) * (1.0 + Lmu_simple(x) * c2))
    y = gmag / a0
    return NU[base](y) * (1.0 + LNU[base](y) * c2)


def geometry_factor(base, y, n):
    """Q for a symmetric collapse in n dimensions, EXACT (curl field vanishes).

        n = 1  slab / pancake      Q = nu (1 + L_nu)
        n = 2  cylinder / filament Q = nu (1 + L_nu/2)
        n = 3  sphere              Q = nu (1 + L_nu/3)

    Derivation: with delta uniform inside, g_N ~ r so y ~ r and
    div g = nu (n + L_nu) g_N / r while 4 pi G rhobar delta = n g_N / r.
    Deep MOND (L_nu = -1/2) gives 1/2, 3/4, 5/6 times nu -- so the MOND
    enhancement is LARGEST for the most isotropic geometry, which is the
    opposite of overproducing pancakes.
    """
    if base == "newton":
        return np.ones_like(np.asarray(y, float))
    return NU[base](y) * (1.0 + LNU[base](y) / float(n))


def ensemble_isotropy(base, gmag, a0, nsamp=200000, seed=20260904):
    """Is the response statistically isotropic where local growth is directional?

    <ghat ghat> = I/3 for an isotropic distribution of background directions,
    so <K> is exactly isotropic with the (1 + L/3) factor -- the same factor
    the SPHERICAL geometry gives.  What survives locally is a quadrupole; its
    Frobenius size relative to the isotropic part is reported, together with a
    Monte-Carlo check that the ensemble mean quadrupole falls as N^(-1/2).
    """
    rng = np.random.default_rng(seed)
    if base == "newton":
        pre, L = 1.0, 0.0
    elif base == "aqual":
        x = gmag / a0
        pre, L = float(mu_simple(x)), float(Lmu_simple(x))
    else:
        y = gmag / a0
        pre, L = float(NU[base](y)), float(LNU[base](y))
    v = rng.normal(size=(nsamp, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    P = np.einsum("ni,nj->nij", v, v)
    Kbar = pre * (np.eye(3) + L * P.mean(axis=0))
    iso_analytic = pre * (1.0 + L / 3.0)
    # local quadrupole size, |L (ghat ghat - I/3)|_F = |L| sqrt(2/3)
    quad_local = abs(L) * math.sqrt(2.0 / 3.0)
    dev = Kbar - np.trace(Kbar) / 3.0 * np.eye(3)
    ns = [1000, 10000, 100000, nsamp]
    decay = []
    for n in ns:
        Pm = P[:n].mean(axis=0)
        d = pre * L * (Pm - np.eye(3) / 3.0)
        decay.append(float(np.linalg.norm(d, "fro")))
    return dict(pre=pre, L=L,
                iso_mean_analytic=iso_analytic,
                iso_mean_mc=float(np.trace(Kbar) / 3.0),
                local_quadrupole_over_isotropic=float(
                    quad_local / max(abs(1.0 + L / 3.0), 1e-300)),
                ensemble_quadrupole_fro=float(np.linalg.norm(dev, "fro")),
                mc_n=ns, mc_quadrupole=decay,
                mc_decay_slope=float(np.polyfit(np.log(ns),
                                                np.log(np.maximum(decay, 1e-300)),
                                                1)[0]) if abs(L) > 0 else None)


# ============================================================================
# 3.  THE TIDAL INVARIANT OF A PERTURBED UNIVERSE
# ============================================================================
#: |T| = |traceless Hessian of Phi_N|_F.  For a statistically isotropic
#: Gaussian density field the five independent components of T are isotropic
#: Gaussian on the traceless-symmetric space with
#:     < T_ij T_ij > = (2/3) (4 pi G rhobar)^2 sigma_delta^2
#: so |T|/(4 pi G rhobar sigma) follows a chi_5 distribution scaled by
#: sqrt(2/15).  This is exact, and it is verified by Monte Carlo below.
TIDAL_C_RMS = math.sqrt(2.0 / 3.0)
#: single-mode / geometry coefficients, |T| = c_n * 4 pi G rhobar delta
TIDAL_C_GEOM = {1: math.sqrt(2.0 / 3.0), 2: 1.0 / math.sqrt(6.0), 3: 0.0}


def tidal_chi5_check(nsamp=400000, seed=20260904):
    """Verify < |T|^2 > = (2/3)(4 pi G rhobar sigma)^2 and the chi_5 shape."""
    rng = np.random.default_rng(seed)
    # draw isotropic khat, unit-variance delta_hat, build H = khat khat delta
    v = rng.normal(size=(nsamp, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    d = rng.normal(size=nsamp)
    H = np.einsum("n,ni,nj->nij", d, v, v)
    tr = np.trace(H, axis1=1, axis2=2)
    T = H - tr[:, None, None] / 3.0 * np.eye(3)[None]
    t2 = np.einsum("nij,nij->n", T, T)
    return dict(mean_T2_over_sigma2=float(t2.mean()),
                analytic=2.0 / 3.0,
                rel_err=float(abs(t2.mean() - 2.0 / 3.0) / (2.0 / 3.0)),
                mean_tr2=float((tr ** 2).mean()), analytic_tr2=1.0)


def tidal_gate_average(A, T0, m, Trms, rule="deep_mond_calibrated",
                       nquad=4001):
    """< 1 + A W > over the chi_5 distribution of |T|, three averaging rules.

    THE SHELL AVERAGE OF A RESPONSE IS PHYSICS, NOT BOOKKEEPING -- the lesson
    that cost this programme A_T = -12.8 where the truth was -4.7.  The
    tournament's own calibrated translation for the scalar structure is that in
    the deep-MOND regime a0 -> a0(1+AW) is equivalent to a conductivity
    k_eq = (1+AW)^(-2/3), whose HARMONIC mean corresponds to

        (1 + A W)_eff = < (1 + A W)^(2/3) >^(3/2)

    That rule is PRIMARY here because it is the one ch_cluster.py already
    calibrated against full 3-D solves.  The arithmetic mean < 1 + A W > and
    the naive "response of the mean" 1 + A W(<|T|>) are returned beside it so
    the bracket is always visible.
    """
    # |T| = Trms * chi_5 / sqrt(5)  (so that <|T|^2> = Trms^2)
    u = np.linspace(1e-6, 12.0, nquad)                # chi_5 variable
    pdf = u ** 4 * np.exp(-u ** 2 / 2.0)
    pdf /= np.trapezoid(pdf, u)
    Tv = Trms * u / math.sqrt(5.0)
    W = W_of("inv", Tv / T0, m)
    R = 1.0 + A * W
    arith = float(np.trapezoid(R * pdf, u))
    calib = float(np.trapezoid(np.maximum(R, 1e-300) ** (2.0 / 3.0) * pdf, u)
                  ** 1.5)
    mean_T = float(np.trapezoid(Tv * pdf, u))
    of_mean = float(1.0 + A * W_of("inv", mean_T / T0, m))
    return dict(primary=calib if rule == "deep_mond_calibrated" else arith,
                deep_mond_calibrated=calib, arithmetic=arith,
                response_of_mean=of_mean, mean_absT=mean_T, rms_absT=Trms,
                bracket_dex=float(abs(np.log10(max(arith, 1e-300) /
                                               max(calib, 1e-300)))))


def sigma_from_cutoff(delta_at_k, k_lo, k_hi, nk=400, slope=1.0):
    """sigma_delta from a mode amplitude law delta_k = delta_at_k (k/k_ref)^slope.

    The deep-MOND attractor gives delta_k ~ k (slope = 1), so the variance is
    UV-dominated: sigma^2 ~ Int k^2 dk k^2 diverges.  The tidal gate's argument
    is therefore SET BY THE SMALLEST SCALE RETAINED, and the theory does not
    say what that is.  This function exists so that dependence can be printed
    rather than hidden.
    """
    k = np.geomspace(k_lo, k_hi, nk)
    # dimensionless variance per log k for a mode amplitude law
    dvar = (delta_at_k * (k / k_lo) ** slope) ** 2
    return float(np.sqrt(np.trapezoid(dvar / k, k)))


# ============================================================================
# 4.  THE WELL-NETWORK TENSOR'S LINEAR RESPONSE -- ANALYTIC AND MEASURED
# ============================================================================
def _j2(z):
    z = np.asarray(z, float)
    small = z < 1e-4
    zs = np.where(small, 1.0, z)
    val = ((3.0 / zs ** 3 - 1.0 / zs) * np.sin(zs)
           - 3.0 * np.cos(zs) / zs ** 2)
    return np.where(small, z ** 2 / 15.0, val)


def S_form_factor(k, family="plaw", p=0.0, q=1.0, s=2.0, L=300.0 * KPC,
                  rmax_over_L=20.0, nr=200000, want_den=False):
    """The exact linear response of S to a plane-wave density perturbation.

    Field point at the origin, well number density n = nbar (1 + delta cos k.z).
    The zeroth order vanishes by isotropy; using Int_-1^1 P_2(mu) cos(z mu) dmu
    = -2 j_2(z), the first order is

        delta S_ij = - delta * Jfac(k) * (zhat_i zhat_j - delta_ij/3)
        Jfac(k)    = Int_0^rmax w r^2 j_2(k r) dr / Int_0^rmax w r^2 dr

    Jfac ~ k^2 <r^4>/(15 <r^2>) as k -> 0 and -> 0 as k -> infinity.

    THE FROZEN KERNEL IS NOT NORMALISABLE.  Both surviving well settings are
    plaw with q = 1, s = 2, so w ~ r^-2 at large r and the DENOMINATOR
    Int w r^2 dr diverges linearly with rmax.  On an unbounded homogeneous
    background -- which is what a cosmological background is -- the normalising
    sum is infinite and S is identically zero, response and all.  On a finite
    catalogue it is finite and proportional to 1/rmax.  rmax is therefore an
    explicit argument here and its effect is measured, not hidden.
    """
    r = np.geomspace(1e-4 * L, rmax_over_L * L, nr)
    if family == "plaw":
        w = (1.0 + (r / L) ** q) ** (-s)
    elif family == "expo":
        w = np.exp(-(r / L) ** q)
    else:
        raise ValueError(family)
    # p enters only through the mass factor, a common constant for an
    # equal-mass population, and cancels from the normalised S.
    z = np.atleast_1d(np.asarray(k, float))[..., None] * r
    num = np.trapezoid(w * r ** 2 * _j2(z), r, axis=-1)
    den = np.trapezoid(w * r ** 2, r)
    return (num / den, den) if want_den else num / den


def S_kernel_normalisability(family="plaw", q=1.0, s=2.0, L=300.0 * KPC,
                             rmaxes=(2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
                                     200.0, 500.0)):
    """Measure how the S normalisation and response scale with catalogue size."""
    kk = np.array([1.0 / L])
    rows = []
    for rm in rmaxes:
        J, den = S_form_factor(kk, family=family, q=q, s=s, L=L,
                               rmax_over_L=rm, want_den=True)
        rows.append(dict(rmax_over_L=float(rm), denominator=float(den),
                         Jfac_at_kL_1=float(J[0])))
    # fit only the large-rmax tail: below rmax ~ 20 L the truncated j_2
    # integral still oscillates and the core r < L still contributes.
    tail = [r for r in rows if r["rmax_over_L"] >= 20.0]
    x = np.log([r["rmax_over_L"] for r in tail])
    ds = float(np.polyfit(x, np.log([r["denominator"] for r in tail]), 1)[0])
    return dict(rows=rows, fitted_over_rmax_over_L_ge=20.0,
                den_slope=ds,
                J_slope=float(np.polyfit(x, np.log([abs(r["Jfac_at_kL_1"])
                                                    for r in tail]), 1)[0]),
                normalisable=bool(abs(ds) < 0.05),
                verdict="w ~ r^-(q s) = r^-2 for both frozen settings, so "
                        "Int w r^2 dr grows linearly with the catalogue radius "
                        "and the normalised S falls as 1/rmax.  On an unbounded "
                        "homogeneous background S = 0 identically.")


def S_form_factor_measured(kL, family="plaw", p=0.0, q=1.0, s=2.0,
                           L=300.0 * KPC, nwell=120000, delta=0.30,
                           rmax_over_L=20.0, seed=20260904, nreal=6):
    """Measure delta S directly with wellnet.S_tensor, imported unmodified.

    Wells are drawn inside a SPHERE of radius rmax about the field point, with
    number density proportional to 1 + delta cos(k z) -- the same truncation
    the analytic quadrature uses, so the two are comparable term by term rather
    than only in shape.  S is evaluated at the centre and averaged over nreal
    independent catalogues to beat the N^(-1/2) shot noise.
    """
    k = kL / L
    rmax = rmax_over_L * L
    vals = []
    for j in range(nreal):
        rng = np.random.default_rng(seed + 7919 * j)
        got = []
        need = nwell
        while need > 0:
            c = (rng.random((4 * nwell, 3)) - 0.5) * 2 * rmax
            c = c[np.sum(c ** 2, axis=1) < rmax ** 2]
            acc = (rng.random(c.shape[0]) * (1.0 + delta)
                   < 1.0 + delta * np.cos(k * c[:, 2]))
            c = c[acc]
            got.append(c)
            need -= c.shape[0]
        wx = np.concatenate(got)[:nwell]
        wm = np.full(nwell, 1.0e11 * MSUN)
        S = WN.S_tensor(np.zeros((1, 3)), wx, wm, family=family, p=p, q=q, s=s,
                        L=L, M_0=1.0e11 * MSUN, exclude_nearest=False,
                        eps_w=1e-12)
        vals.append(float(S[0, 2]))
    Szz = float(np.mean(vals))
    sd = float(np.std(vals) / math.sqrt(nreal))
    Jm = -Szz / ((2.0 / 3.0) * delta)
    Ja = float(S_form_factor(np.array([k]), family=family, p=p, q=q, s=s, L=L,
                             rmax_over_L=rmax_over_L)[0])
    return dict(kL=float(kL), Szz_measured=Szz, Szz_se=sd,
                Jfac_measured=float(Jm), Jfac_analytic=Ja,
                rel_diff=float(abs(Jm - Ja) / max(abs(Ja), 1e-300)),
                n_sigma=float(abs(Szz - (-(2.0 / 3.0) * delta * Ja))
                              / max(sd, 1e-300)),
                nwell=int(nwell), nreal=int(nreal), delta=float(delta),
                rmax_over_L=float(rmax_over_L))


# ============================================================================
# 5.  PERIODIC NONLINEAR FIELD SOLVER
# ============================================================================
class PeriodicBox:
    """div( A grad psi ) = 4 pi G rho on a triply periodic box.

    Face-flux (finite-volume) discretisation, the same conservative form as
    gravitylab/solver.py, but WRAPPING -- which for a periodic cosmological
    background is the correct boundary condition and for an isolated source
    would be the self-contradictory one solver.py's docstring warns about.
    The operator has a constant null space, removed by projecting the mean out
    of both the source and the iterate.
    """

    def __init__(self, n, Lbox):
        self.n, self.L = int(n), float(Lbox)
        self.h = self.L / self.n
        ax = (np.arange(self.n) + 0.5) * self.h
        self.X, self.Y, self.Z = np.meshgrid(ax, ax, ax, indexing="ij")

    # ---------------------------------------------------------- operator
    def grad(self, f):
        h = self.h
        gx = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * h)
        gy = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * h)
        gz = (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2 * h)
        return gx, gy, gz

    def dcentral(self, f, ax):
        return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * self.h)

    def apply(self, psi, A):
        """div(A grad psi), periodic, SYMMETRIC to round-off.

        Diagonal terms use the compact two-point face flux, which is the
        conservative form gravitylab/solver.py uses and is self-adjoint by
        inspection.  The CROSS terms are written as D_i(A_ij D_j psi) with
        CENTRED D_i: since the periodic centred difference is anti-self-adjoint
        and A_ij = A_ji, the cross block is self-adjoint as well.  The
        face-averaged cross form used for an open box is NOT -- it was measured
        here at 1.9e-2 relative asymmetry, which conjugate gradients are not
        entitled to.
        """
        Axx, Ayy, Azz, Axy, Axz, Ayz = A
        h = self.h
        out = np.zeros_like(psi)
        for ax, Ad in ((0, Axx), (1, Ayy), (2, Azz)):
            af = 0.5 * (Ad + np.roll(Ad, -1, ax))       # A at the i+1/2 face
            F = af * (np.roll(psi, -1, ax) - psi) / h
            out += (F - np.roll(F, 1, ax)) / h
        if not (np.all(Axy == 0) and np.all(Axz == 0) and np.all(Ayz == 0)):
            gx, gy, gz = self.grad(psi)
            out += self.dcentral(Axy * gy + Axz * gz, 0)
            out += self.dcentral(Axy * gx + Ayz * gz, 1)
            out += self.dcentral(Axz * gx + Ayz * gy, 2)
        return out

    def solve_linear(self, rho, A, tol=1e-12, maxiter=20000, x0=None):
        """CG with the constant mode projected out."""
        b = 4.0 * np.pi * G * (rho - rho.mean())
        b -= b.mean()
        x = np.zeros_like(b) if x0 is None else (x0 - x0.mean())
        r = b - self.apply(x, A)
        r -= r.mean()
        p = r.copy()
        rs = float(np.sum(r * r))
        bn = float(np.sqrt(np.sum(b * b))) or 1.0
        it, rel = 0, float(np.sqrt(rs)) / bn
        for it in range(1, maxiter + 1):
            if rel < tol:
                break
            Ap = self.apply(p, A)
            Ap -= Ap.mean()
            den = float(np.sum(p * Ap))
            if den == 0.0:
                break
            al = rs / den
            x += al * p
            r -= al * Ap
            rs_new = float(np.sum(r * r))
            rel = float(np.sqrt(rs_new)) / bn
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x - x.mean(), it, rel

    def solve_nonlinear(self, rho, Afun, npicard=120, tol=1e-12, relax=0.5,
                        atol=1e-10):
        """Picard on A = A(grad psi), started from the NEWTONIAN solution.

        Starting from psi = 0 is a trap and it cost this file one gate: for
        AQUAL, mu(0) = 0, so A vanishes identically, the linear solve returns
        psi = 0, the Picard change is 0 and the loop declares convergence at
        the wrong fixed point.  The iteration is therefore seeded with A = I.
        Relaxation is applied to A, not to psi.
        """
        iso = self.iso_tensor(1.0)
        psi, it0, rel0 = self.solve_linear(rho, iso, tol=tol)
        A = Afun(psi)
        hist = [dict(picard=-1, cg_iters=it0, cg_rel=rel0, dpsi=np.nan,
                     note="newtonian seed")]
        for i in range(npicard):
            psi_new, it, rel = self.solve_linear(rho, A, tol=tol, x0=psi)
            den = max(float(np.max(np.abs(psi_new))), 1e-300)
            dpsi = float(np.max(np.abs(psi_new - psi))) / den
            psi = psi_new
            A_new = Afun(psi)
            A = tuple((1.0 - relax) * a + relax * b for a, b in zip(A, A_new))
            hist.append(dict(picard=i, cg_iters=it, cg_rel=rel, dpsi=dpsi))
            if dpsi < atol:
                break
        return psi, Afun(psi), hist

    def iso_tensor(self, v=1.0):
        o = np.full(self.X.shape, float(v))
        z = np.zeros(self.X.shape)
        return (o, o.copy(), o.copy(), z, z.copy(), z.copy())

    # ------------------------------------------------------------ physics
    def net_force(self, rho, psi):
        """Integral of rho * (-grad psi) over the box, per unit total mass.

        Zero for any variational law by periodicity.  Non-zero exactly when
        the response has an explicit spatial dependence that is not a function
        of |grad psi| alone -- which is what "no candidate has a declared
        momentum carrier" means once you put the law in a box.
        """
        gx, gy, gz = self.grad(psi)
        M = float(np.sum(rho)) * self.h ** 3
        F = np.array([-float(np.sum(rho * gx)), -float(np.sum(rho * gy)),
                      -float(np.sum(rho * gz))]) * self.h ** 3
        return F, F / max(M, 1e-300)


def sym3_min_eig(A):
    """Smallest eigenvalue of a (6,)-packed symmetric field, for ellipticity."""
    Axx, Ayy, Azz, Axy, Axz, Ayz = A
    M = np.stack([np.stack([Axx, Axy, Axz], -1),
                  np.stack([Axy, Ayy, Ayz], -1),
                  np.stack([Axz, Ayz, Azz], -1)], -2)
    ev = np.linalg.eigvalsh(M)
    return float(ev.min()), float(ev.max())


def aqual_tensor_field(box, psi, a0_field):
    """A = mu(|grad psi|/a0(x)) I -- AQUAL, with a possibly varying a0."""
    gx, gy, gz = box.grad(psi)
    gm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    mu = mu_simple(gm / a0_field)
    z = np.zeros_like(mu)
    return (mu, mu.copy(), mu.copy(), z, z.copy(), z.copy())


# ============================================================================
# 6.  RESPONSIVENESS GATE
# ============================================================================
def responsiveness(fn, thetas, label=""):
    """Verify numerically that dS/dtheta != 0 over the tested range.

    A statistic that is bit-identical across a decade of its own coupling is
    the failure this programme has now caught four times.  Every headline
    number in this lane passes through here and the SPREAD IS PRINTED.
    """
    th = np.asarray(thetas, float)
    S = np.array([float(fn(t)) for t in th])
    finite = np.isfinite(S)
    spread = (float(np.nanmax(S[finite]) - np.nanmin(S[finite]))
              if finite.any() else float("nan"))
    d = np.gradient(S, th) if th.size > 2 else np.array([np.nan])
    return dict(label=label, theta=th.tolist(), S=S.tolist(),
                spread=spread,
                rel_spread=float(spread / max(abs(np.nanmedian(S)), 1e-300)),
                n_distinct=int(np.unique(np.round(S, 12)).size),
                dS_dtheta_min_abs=float(np.nanmin(np.abs(d))),
                dS_dtheta_max_abs=float(np.nanmax(np.abs(d))),
                responsive=bool(spread > 0 and np.unique(S).size > 1))
