"""The tournament grammar: candidates, invariants, response functions.

    K = exp[ f_0(I) I  +  f_T(I) That  +  f_d(I) dhat dhat^T  +  f_nl(I) B_nl ]

A CANDIDATE is (base MOND law, invariant, response form, exponent, invariant
scale, tensor structure, amplitude).  Two global constants are free -- a0 and
the amplitude A -- and NOTHING is per-object.

WHY THE RESPONSE FORM MATTERS AND IS SCANNED, NOT CHOSEN.
The boundedness theorem (Run AB) says a response confined to a bounded range
can only renormalise G, so d ln g/d ln r stays -2 and rotation curves cannot
flatten.  That is a statement about the response AS A FUNCTION OF RADIUS in a
fixed system.  It does NOT forbid a response that is a bounded function of an
UNBOUNDED invariant, because such a response still separates a cluster from a
galaxy -- it just cannot make the curve flat on its own.  Both are in the
grammar and both are measured:

    form   W(I)                 sup W        can flatten a curve?
    pow    I^m                  infinity     possibly
    log    ln(1+I^m)            infinity     possibly
    sat    I^m/(1+I^m)          1            no, only environmental
    inv    1/(1+I^m)            1            no, only environmental
    off    0                    0            null control

The asymptotic slope is MEASURED for every candidate (screens.asymptotic), it
is never assumed from the form.

THE INVARIANTS, all computable identically in every channel.
    gn      |g_N|/a0                    bounded ratio in the RAR sense
    phi     |Phi_N|/Phi_0               UNBOUNDED; the known working gate
    rhobar  3 M_b(<r)/(4 pi r^3 rho_0)  UNBOUNDED, mean enclosed baryon density
    tidal   |g_N|/(r a0/L_T) ~ |T|/T_0  UNBOUNDED
    qbar    nonlocal mass fraction      BOUNDED in [0,1) -- the theorem's control
    one     1                           no gate

|Phi_N| IS DEFINED ONLY UP TO A CONSTANT.  The boundary rule is declared here,
in advance, and three defensible rules are carried so the spread is visible:
    "inf"  (PRIMARY) Phi -> 0 at infinity, with the baryon distribution
           continued outside the last measured point as a point mass.
    "last" Phi referenced to the last measured radius (a potential DIFFERENCE).
    "half" Phi = -G M_b/max(r, r_half), the crude point-mass rule.
Run Z's warning is that the rule DEFINES the variable rather than conditioning
it, so every phi-gated result is repeated under all three.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
A0 = 1.2e-10

# ------------------------------------------------------------------ grammar
INVARIANTS = ("one", "gn", "phi", "rhobar", "tidal", "qbar")
FORMS = ("off", "sat", "pow", "log", "inv")
#: how the response enters the field equation
STRUCTURES = (
    "scalar_a0",    # a0 -> a0 (1 + A W)                    THE BRIEF'S COMPETITOR
    "iso_K",        # K = exp(-A W) I                       isotropic conductivity
    "tensor_S",     # K = exp(A W S)                        well-network, traceless
    "tensor_d",     # K = exp(A W (dhat dhat^T - I/3))      dhat = ghat_N
    "tensor_T",     # K = exp(A W That)                     normalised tidal
)
#: number of free global constants, for the parsimony rule.  a0 always counts.
NPARAM = {
    "off": 1,           # a0 only -- the base MOND law
}


def n_params(cand) -> int:
    """Free GLOBAL constants of a candidate.  Used by the parsimony rule."""
    if cand.form == "off" or cand.inv == "one":
        return 1                       # a0
    n = 2                              # a0, amplitude A
    if cand.form in ("sat", "pow", "log", "inv"):
        n += 1                         # exponent m
    if cand.inv in ("phi", "rhobar", "tidal", "qbar"):
        n += 1                         # the invariant scale I_0
    return n


@dataclass
class Candidate:
    name: str
    base: str = "rar"           # 'rar' | 'aqual' | 'newton'
    inv: str = "one"
    form: str = "off"
    m: float = 1.0
    I0: float = 1.0             # Phi_0 / rho_0 / T_0 in SI
    struct: str = "scalar_a0"
    A: float = 0.0              # amplitude, FITTED on the cluster channel
    a0: float = A0              # FITTED on SPARC train
    phi_rule: str = "inf"
    extra: dict = field(default_factory=dict)

    def key(self):
        return (f"{self.base}|{self.struct}|{self.inv}|{self.form}"
                f"|m={self.m:g}|I0={self.I0:.4g}|{self.phi_rule}")


# ------------------------------------------------------- response functions
#: Declared numerical ceiling on the response.  A response of 1e6 is already
#: far outside anything observable (it would change g by 10^3 at least), and
#: without a ceiling the pow form with a negative exponent overflows on
#: invariants that legitimately reach 1e-40.  Every clip is counted and
#: reported; no headline number rests on a clipped value.
W_CEIL = 1.0e6
N_CLIPPED = [0]


def W_of(form, I, m):
    """The dimensionless response W(I).  I is already I/I_0."""
    I = np.maximum(np.asarray(I, float), 1e-300)
    if form == "off":
        return np.zeros_like(I)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if form == "sat":
            u = I ** m
            out = u / (1.0 + u)
        elif form == "pow":
            out = I ** m
        elif form == "log":
            out = np.log1p(I ** m)
        elif form == "inv":
            out = 1.0 / (1.0 + I ** m)
        else:
            raise ValueError(form)
    out = np.nan_to_num(out, nan=0.0, posinf=W_CEIL, neginf=0.0)
    if np.any(out > W_CEIL):
        N_CLIPPED[0] += 1
    return np.minimum(out, W_CEIL)


def W_sup(form):
    """sup W over I in [0, inf).  inf means UNBOUNDED."""
    return {"off": 0.0, "sat": 1.0, "inv": 1.0,
            "pow": math.inf, "log": math.inf}[form]


def W_cand(cand, invmap):
    """W(I(x)) for a candidate given a dict of raw invariant fields."""
    if cand.form == "off" or cand.inv == "one":
        return np.zeros_like(np.asarray(invmap["gn"], float))
    return W_of(cand.form, np.asarray(invmap[cand.inv], float) / cand.I0,
                cand.m)


# ------------------------------------------------------------- the MOND base
def nu_rar(x):
    """g/g_N = nu(g_N/a0), McGaugh Lelli Schombert 2016."""
    x = np.maximum(np.asarray(x, float), 1e-300)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def g_of_gN(base, gN, a0):
    """The spherical/algebraic radial law with NO response term."""
    gN = np.asarray(gN, float)
    if base == "newton":
        return gN.copy()
    if base == "rar":
        return nu_rar(gN / a0) * gN
    if base == "aqual":                # mu(g/a0) g = g_N, mu = x/(1+x)
        return 0.5 * (gN + np.sqrt(gN ** 2 + 4.0 * gN * a0))
    raise ValueError(base)


def g_response(cand, gN, W, k_r=None):
    """The full radial law: base MOND, then the candidate's response.

    scalar_a0 : a0 -> a0 (1 + A W)
    everything else enters through the radial eigenvalue k_r of K, solving
        mu(sqrt(k) |Phi'|/a0) k |Phi'| = g_N          (mu = x/(1+x))
    exactly, which for k = 1 reproduces the AQUAL base identically.
    """
    gN = np.asarray(gN, float)
    if cand.struct == "scalar_a0":
        a0e = cand.a0 * (1.0 + cand.A * np.asarray(W, float))
        return g_of_gN(cand.base, gN, a0e)
    if k_r is None:
        k_r = k_radial_pointwise(cand, W)
    return mond_invert(gN, k_r, cand.a0, cand.base)


def mond_invert(F, k, a0, base="aqual"):
    """|Phi'| solving mu(sqrt(k)|Phi'|/a0) k |Phi'| = F, closed form.

    Same algebra as tensor/field.py Mu.invert with kind='simple'.  For
    base='rar' the nu form is used instead, with the same k scaling, so that
    k = 1 reproduces the plain RAR: nu is applied to the k-corrected argument
    and the result divided by k.
    """
    F = np.asarray(F, float)
    k = np.maximum(np.asarray(k, float), 1e-300)
    if base == "newton":
        return F / k
    if base == "aqual":
        beta = F / (np.sqrt(k) * a0)
        X = 0.5 * (beta + np.sqrt(beta * beta + 4.0 * beta))
        return a0 * X / np.sqrt(k)
    if base == "rar":
        # QUMOND-style form  g = nu(F/(k^{1/2} a0)) F / k.
        #
        # Within the family  nu(F/(k^p a0)) F/k  the exponent p is FIXED, not
        # chosen: k = 1 reproduces plain RAR for every p; the Newtonian limit
        # gives g ~ F/k for every p; but deep MOND gives
        #     g ~ sqrt(F a0) k^{p/2 - 1}
        # so matching the AQUAL branch's g ~ sqrt(F a0) k^{-3/4} requires
        # p = 1/2 uniquely.
        #
        # BUG, Run AQ: this read k ** 1.5 (p = 3/2) from the tournament of
        # 2026-09-04 onward, giving deep-MOND g ~ k^{-1/4} while the comment
        # beside it claimed k^{-3/4}.  The error is EXACTLY ZERO at k = 1, so
        # it is invisible for every scalar_a0 candidate (k_radial_pointwise
        # returns 1) and for both Newtonian limits; it bites only the
        # base='rar' half of the k != 1 structures, at 0.15 dex per e-fold of
        # k.  See test_mond_invert_k_scaling below.
        return nu_rar(F / (k ** 0.5 * a0)) * F / k
    raise ValueError(base)


def k_radial_pointwise(cand, W):
    """k_r = rhat^T K rhat for the structures whose eigenvector IS rhat.

    In a spherically reduced geometry dhat = rhat and S = s (rhat rhat^T -
    I/3), so every traceless structure contributes its LARGEST eigenvalue
    along rhat.  For the disk channels the full tensor is used instead
    (ch_vertical), not this reduction.
    """
    W = np.asarray(W, float)
    if cand.struct == "iso_K":
        return np.exp(-cand.A * W)
    if cand.struct in ("tensor_S", "tensor_d", "tensor_T"):
        # traceless structure with rhat as an eigenvector of eigenvalue 2/3
        # (dhat dhat^T - I/3 exactly; S saturates the same bound; That is
        # normalised to |That|_2 < 1 and is scanned at its radial value).
        lam = cand.extra.get("lam_r", 2.0 / 3.0)
        return np.exp(cand.A * W * lam)
    if cand.struct == "scalar_a0":
        return np.ones_like(W)
    raise ValueError(cand.struct)


# ------------------------------------------------------------ sym3 helpers
def sym3_iso():
    """The identity as a (6,) symmetric-3x3 in tensor-lane order."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def sym3_from_dir(d):
    """dhat dhat^T - I/3 for unit vectors d (P,3) -> (P,6), traceless."""
    d = np.asarray(d, float)
    return np.stack([d[:, 0] ** 2 - 1 / 3, d[:, 1] ** 2 - 1 / 3,
                     d[:, 2] ** 2 - 1 / 3, d[:, 0] * d[:, 1],
                     d[:, 0] * d[:, 2], d[:, 1] * d[:, 2]], axis=1)


def sym3_traceless_norm(m):
    """|M|_F for a (.,6) symmetric-3x3 in tensor-lane order."""
    m = np.asarray(m, float)
    return np.sqrt(m[..., 0] ** 2 + m[..., 1] ** 2 + m[..., 2] ** 2
                   + 2 * (m[..., 3] ** 2 + m[..., 4] ** 2 + m[..., 5] ** 2))
