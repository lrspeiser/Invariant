"""compiler.py -- the pre-data admissibility compiler.

    check(candidate) -> {gate: (pass, measured_value, reason)}

Four gates, run BEFORE any GPU time and before any astronomical datum is
touched.  Each one rejects a whole FAMILY of candidate laws on a mathematical
property, not on a fit:

    GATE 1  CONSTANT-K DEGENERACY
            For constant symmetric positive-definite K the substitution
            x' = K^(-1/2) x turns  div[K grad Phi] = 4 pi G rho  into a plain
            Laplacian with a stretched source.  A response that is constant
            over the region a probe samples is therefore not an observable: it
            is a coordinate stretch plus a transformed source, degenerate with
            source ellipticity, inclination, line-of-sight depth, distance and
            baryonic deprojection.  The gate implements the substitution
            explicitly, measures the residual, then requires at least one of:
              (a) K varies spatially over the probe's radial range by more
                  than a stated threshold;
              (b) the preferred axis is fixed by an INDEPENDENTLY MEASURED
                  direction and is misaligned with the probe's radial
                  direction by more than a stated angle;
              (c) two probes with different geometry disagree in a way one
                  global coordinate stretch cannot produce.

    GATE 2  POTENTIAL GAUGE
            Absolute Newtonian potential is undefined until a boundary
            convention is fixed.  Run AH measured the two admissible GLOBAL
            rules differing by 0.87 dex in median galaxy potential depth
            against an off/on gate margin of only 0.9 dex.  Four defensible
            operational rules are implemented (nearest gravitational saddle,
            fixed overdensity boundary, fixed multiple of a baryonic scale
            radius, edge of a reconstructed environmental volume), plus the
            tournament's declared primary, and the spread of the headline
            quantity across them is reported.  A candidate whose VERDICT
            changes across defensible rules is FLAGGED, loudly, not
            eliminated.

    GATE 3  COARSE GRAINING
            One identical continuous galaxy is represented as 1 catalogue
            object, 10 subcomponents and N stellar-mass cells; convergence is
            required below a stated threshold.  Merging neighbouring entries,
            changing the detection threshold, varying deblending, moving mass
            between intracluster light and members, and changing mesh
            resolution are all tested.  Uniform refinement is the WEAK test (a
            mass exponent cancels exactly under it); selective refinement is
            the one with teeth, and there only p = 1 is admissible.  The
            physical-scale-versus-catalogue-row discriminator
            d ln(drift)/d ln L is reimplemented with its sign convention
            stated in full (see the GATE 3 header comment).

    GATE 4  RECIPROCITY AND ACTION
            F(x,x') = F(x',x) is required for nonlocal kernels unless a
            momentum carrier is explicitly declared.  Then the functional
            Jacobian delta Phi(x)/delta rho(y) is tested for symmetry under
            the relevant boundary conditions; an asymmetric Jacobian cannot
            come from an action, so the candidate is nonconservative.  This
            does not prove a relativistic completion exists -- it rejects
            nonconservative candidates immediately.

DATA STATEMENT
--------------
This lane opens NO observational data of any kind.  There is no data-reading
code in it, no network access, and no path outside its own directory is read.
KiDS and the wide binaries are not loaded, listed or referenced; neither is
SPARC, nor any cluster catalogue.  Every number produced here comes from
closed-form synthetic geometries defined in this file.  Because nothing
observational is touched, THERE IS NO BLIND-PROTECTION ISSUE with this lane --
stated explicitly rather than omitted.  `test_compiler.py` asserts it
mechanically by intercepting `open` for the duration of a full run.

Lane: work/wellnet-2026-09/compiler/
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================ 0. constants
G = 6.674e-11
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
AU = 1.495978707e11
MSUN = 1.98892e30
A0 = 1.2e-10

#: Tolerance inherited from the tournament's own H2/H3 galaxy screens: 0.040
#: dex, the RAR's intrinsic scatter.  A difference smaller than this is not an
#: observable in this programme, so it is not allowed to rescue a candidate.
TOL_DEX = 0.040

#: Gate 1 escape (b): the preferred axis must be misaligned with the probe's
#: radial direction by at least this much.  A radially aligned anisotropy has
#: the exact exterior solution Psi = -GM/(k_r r), which is a pure rescaling of
#: G and therefore degenerate with the mass-to-light ratio and the distance.
TOL_MISALIGN_DEG = 10.0

#: Gate 3: convergence threshold on the relative drift of the response when a
#: FIXED continuous mass distribution is re-tabulated.  The screen lane's own
#: `N_safe` criterion.
TOL_COARSE = 1.0e-3

#: Gate 3: admissible mass exponent under SELECTIVE refinement.  The relative
#: weight of a refined object against an unrefined one moves as N^(1-p), so
#: only p = 1 has a limit.  Tolerance on the measured slope.
TOL_WEIGHT_SLOPE = 1.0e-2

#: Gate 4: the round-off floor.  Measured on the Newtonian and AQUAL/QUMOND
#: base laws, which are variational, so anything at or below this "passes at
#: round-off".
TOL_ASYM = 1.0e-9

#: Run AH's measured spread between its two admissible GLOBAL boundary rules,
#: and the off/on margin of the gate it feeds.
RUN_AH_PHI_SPREAD_DEX = 0.87
RUN_AH_GATE_MARGIN_DEX = 0.90

DATA_STATEMENT = (
    "No observational data of any kind is opened by this module. Every number "
    "is produced from closed-form synthetic geometries defined in this file. "
    "KiDS and the wide binaries are never loaded, listed or referenced, and "
    "neither is SPARC nor any cluster catalogue. Nothing here touches "
    "observational data, so there is no blind-protection issue in this lane."
)


# ============================================================ 1. candidates
#: Where a candidate's preferred axis comes from.  This is the split the brief
#: demands: source axis, external tidal axis, catalogue well network.
AXIS_PROVENANCE = {
    "scalar_a0":   ("none",      "no preferred axis: a scalar rescaling of a0"),
    "iso_K":       ("none",      "isotropic K: no preferred axis at all"),
    "tensor_d":    ("source",    "dhat = ghat_N, read off the baryon map"),
    "tensor_T":    ("external",  "the traceless Hessian of the baryonic Phi_N"),
    "tensor_S":    ("catalogue", "set by the well list, i.e. by the cataloguer"),
    "wells":       ("catalogue", "family C: the alignment tensor of a row list"),
    "pairs":       ("catalogue", "family D: the pair tensor of a row list"),
    "tidal_const": ("external",  "family E: constant couplings on the "
                                 "normalised tidal tensor"),
    "depth":       ("none",      "family B: a0 modulated by |Phi|, no axis"),
    "none":        ("none",      "base law, no response"),
}

#: Which invariants are functionals of the Poisson-smooth fields (so a row
#: list never enters), and which are read off a catalogue.
INVARIANT_SOURCE = {
    "one":    "constant",
    "gn":     "smooth",           # |grad Phi_N|
    "phi":    "smooth-nonlocal",  # needs a boundary rule -> GATE 2
    "rhobar": "smooth",           # div g_N / 4 pi G, the Poisson source itself
    "tidal":  "smooth",           # traceless Hessian, a local 2nd derivative
    "qbar":   "smooth-ball",      # M_b within a declared global L_NL of x
}

#: The invariants that carry an undetermined additive constant -- gauge
#: dependent, and so required to be an operational DIFFERENCE.
GAUGE_DEPENDENT_INVARIANTS = {"phi"}

#: Structures whose response IS the tensor, with no invariant to gate on.
CONSTANT_COUPLING_STRUCTS = {"wells", "pairs", "tidal_const"}

#: Structures built from a catalogue row list rather than a smooth field.
CATALOGUE_STRUCTS = {"tensor_S", "wells", "pairs"}

#: The screen lane's family-E regulator: That = T0 / sqrt(eps_T^2 + |T0|^2).
#: Default a0 / (10 kpc), exactly `families._pE`.
EPS_T_DEFAULT = A0 / (10.0 * KPC)

#: WHICH FUNCTIONAL OF rho THE RESPONSE READS.  This is what GATE 4 turns on.
#:
#: In the QUMOND form  lap Psi = div[ K(.) grad Phi_N ]  the law comes from an
#: action with Phi_N still solving Poisson if and only if  K(u) u  is a
#: gradient in u = grad Phi_N.  For K = phi(|u|) I, and for the traceless
#: field-direction structure K = exp(a(|u|)(uhat uhat^T - I/3)) -- for which
#: K u = e^{2a/3} u exactly -- that holds, and the law is just AQUAL/QUMOND
#: with a redefined interpolating function.  Any response that reads Phi_N
#: itself, the Hessian, rho, a nonlocal ball mass, or a catalogue row list is
#: NOT a function of u, so the law as written is not the Euler-Lagrange system
#: of any action in which Phi_N is the Newtonian potential.  A variational
#: completion would have to promote the extra field to a dynamical one, which
#: changes the law.
ADMISSIBLE_SOURCES = {"none", "|grad Phi_N|", "grad Phi_N direction"}

TENSOR_SOURCE = {
    "scalar_a0":   "none",
    "iso_K":       "none",
    "depth":       "none",
    "none":        "none",
    "tensor_d":    "grad Phi_N direction",
    "tensor_T":    "Hessian of Phi_N",
    "tidal_const": "Hessian of Phi_N",
    "tensor_S":    "catalogue row list",
    "wells":       "catalogue row list",
    "pairs":       "catalogue row list",
}
INVARIANT_FUNCTIONAL = {
    "one":    "none",
    "gn":     "|grad Phi_N|",
    "phi":    "Phi_N (nonlocal, gauge-dependent)",
    "rhobar": "rho = lap Phi_N",
    "tidal":  "Hessian of Phi_N",
    "qbar":   "nonlocal ball mass",
}


def response_field(cand: "Candidate") -> Dict[str, Any]:
    """The two functionals the response reads, and whether both are
    admissible for an action in which Phi_N still solves Poisson."""
    if not cand.responds():
        return dict(tensor="none", invariant="none", admissible=True)
    ts = TENSOR_SOURCE.get(cand.struct, "unknown")
    if cand.struct in CONSTANT_COUPLING_STRUCTS:
        inv_s = "none"
    else:
        inv_s = INVARIANT_FUNCTIONAL.get(cand.inv, "unknown")
    if cand.field_source == "rows":
        ts = "catalogue row list"
    return dict(tensor=ts, invariant=inv_s,
                admissible=bool(ts in ADMISSIBLE_SOURCES
                                and inv_s in ADMISSIBLE_SOURCES))


@dataclass
class Candidate:
    """One candidate law, in a grammar wide enough for every family the
    programme has characterised.

    The tournament grammar (Run AH) is (base, struct, inv, form, m, I0, A);
    the screen lane's families A-E add `well`, `pair`, `depth` and
    `tidal_const` parameters.
    """
    name: str = "unnamed"
    base: str = "aqual"          # 'newton' | 'aqual' | 'qumond' | 'rar'
    struct: str = "scalar_a0"    # see AXIS_PROVENANCE
    inv: str = "one"             # see INVARIANT_SOURCE
    form: str = "off"            # 'off'|'sat'|'inv'|'pow'|'log'
    m: float = 1.0
    I0: float = 1.0
    A: float = 0.0
    a0: float = A0
    phi_rule: str = "inf"
    well: Optional[Dict[str, Any]] = None          # family-C / tensor_S weight
    pair: Optional[Dict[str, Any]] = None          # family-D pair weight
    depth: Optional[Dict[str, Any]] = None         # family-B depth parameters
    tidal_const: Optional[Dict[str, Any]] = None   # family-E couplings
    #: is the tensor's source field read off the CATALOGUE ROW LIST or off the
    #: smooth density?  Run AB's named repair for family E is exactly this
    #: switch: "source the tidal tensor from the smooth density rather than
    #: the row list and four screens plus two gates pass automatically".
    field_source: str = "smooth"                   # 'smooth' | 'rows'
    momentum_carrier: str = ""                     # downgrades GATE 4 to FLAG
    pair_kernel: Optional[Callable] = None         # explicit F(x, x')
    note: str = ""

    def signature(self) -> Tuple:
        """The DISCRETE part of the candidate.

        Gates 2, 3 and 4 depend on this and on nothing else, which is what
        makes the compiler a family eliminator rather than a setting screen.
        """
        w = self.well or {}
        p = self.pair or {}
        return (self.base, self.struct, self.inv, self.form,
                float(self.m), self.phi_rule, self.field_source,
                w.get("family"), w.get("p"), w.get("q"), w.get("s"),
                w.get("L"), w.get("exclude_nearest"),
                p.get("p"), p.get("q"), p.get("s"),
                bool(self.momentum_carrier), self.pair_kernel is not None)

    def responds(self) -> bool:
        """Does the candidate have a live response term at all?"""
        if self.struct in CONSTANT_COUPLING_STRUCTS:
            return self.A != 0.0
        return not (self.form == "off" or self.inv == "one" or self.A == 0.0
                    or self.struct == "none")


def _parse_well_tag(tag: str) -> Dict[str, Any]:
    """`plaw_p0q1s2_L300` / `expo_p1q2_L1000_x` -> the weight parameters."""
    import re
    parts = tag.split("_")
    fam = parts[0]
    p = q = s = None
    L = 300.0
    for tok in parts[1:]:
        if tok.startswith("L") and tok[1:].replace(".", "").isdigit():
            L = float(tok[1:])
        else:
            mm = re.fullmatch(r"p([\d.]+)q([\d.]+)(?:s([\d.]+))?", tok)
            if mm:
                p = float(mm.group(1))
                q = float(mm.group(2))
                s = float(mm.group(3)) if mm.group(3) else 1.0
    return dict(family=fam, p=1.0 if p is None else p,
                q=1.0 if q is None else q, s=1.0 if s is None else s,
                L=L * KPC, exclude_nearest=tag.endswith("_x"))


def from_tournament_record(rec: Dict[str, Any]) -> Candidate:
    """Build a Candidate from one row of `tournament/tournament.json`."""
    struct = rec["struct"]
    nm = rec["name"]
    well = None
    if struct == "tensor_S" and "[" in nm:
        well = _parse_well_tag(nm.split("[", 1)[1].split("]", 1)[0])
    return Candidate(name=nm, base=rec["base"], struct=struct,
                     inv=rec["inv"], form=rec["form"], m=float(rec["m"]),
                     I0=float(rec["I0"]), A=float(rec.get("A") or 0.0),
                     a0=float(rec.get("a0") or A0), well=well)


# ============================================================ 2. response
def W_of(form: str, I, m: float, ceil: float = 1.0e6) -> np.ndarray:
    """The dimensionless response W(I), I already divided by I_0.

    Identical algebra to `tournament/tw_core.W_of`, reimplemented here so the
    compiler carries no import dependency on the lane it audits.
    """
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
    out = np.nan_to_num(out, nan=0.0, posinf=ceil, neginf=0.0)
    return np.minimum(out, ceil)


def W_sup(form: str) -> float:
    return {"off": 0.0, "sat": 1.0, "inv": 1.0,
            "pow": math.inf, "log": math.inf}[form]


def response_W(cand: Candidate, inv_fields: Dict[str, np.ndarray]) -> np.ndarray:
    """W(I(x)) for a candidate, given the raw invariant fields."""
    ref = np.asarray(inv_fields["gn"], float)
    if cand.struct in CONSTANT_COUPLING_STRUCTS:
        return np.ones_like(ref)                # the response IS the tensor
    if cand.form == "off" or cand.inv == "one":
        return np.zeros_like(ref)
    return W_of(cand.form, np.asarray(inv_fields[cand.inv], float) / cand.I0,
                cand.m)


def nu_rar(x):
    """RAR nu, evaluated with expm1 so 1 - exp(-sqrt(x)) does not cancel."""
    x = np.maximum(np.asarray(x, float), 1e-300)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return 1.0 / np.maximum(-np.expm1(-np.sqrt(x)), 1e-300)


def g_of_gN(base: str, gN, a0):
    """The algebraic radial law with no response term."""
    gN = np.asarray(gN, float)
    a0 = np.asarray(a0, float)
    if base == "newton":
        return gN.copy()
    if base == "rar":
        return nu_rar(gN / a0) * gN
    if base in ("aqual", "qumond"):        # mu = x/(1+x), closed form
        return 0.5 * (gN + np.sqrt(gN ** 2 + 4.0 * gN * a0))
    raise ValueError(base)


def mond_invert(F, k, a0, base="aqual"):
    """|Phi'| solving the k-corrected MOND relation; k = 1 reproduces the base.

    Same algebra as `tournament/tw_core.mond_invert`.
    """
    F = np.asarray(F, float)
    k = np.maximum(np.asarray(k, float), 1e-300)
    if base == "newton":
        return F / k
    if base in ("aqual", "qumond"):
        beta = F / (np.sqrt(k) * a0)
        X = 0.5 * (beta + np.sqrt(beta * beta + 4.0 * beta))
        return a0 * X / np.sqrt(k)
    if base == "rar":
        return nu_rar(F / (k ** 1.5 * a0)) * F / k
    raise ValueError(base)


def radial_eigen(struct: str, lam, A: float, W) -> np.ndarray:
    """k_r = rhat^T K rhat for the response structures."""
    W = np.asarray(W, float)
    lam = np.asarray(lam, float)
    if struct in ("scalar_a0", "depth", "none"):
        return np.ones_like(W)
    if struct == "iso_K":
        return np.exp(np.clip(-A * W, -700, 700))
    if struct in ("tensor_S", "tensor_d", "tensor_T", "wells", "tidal_const"):
        return np.exp(np.clip(A * W * lam, -700, 700))
    if struct == "pairs":                  # K = exp[-alpha C], C positive
        return np.exp(np.clip(-A * W * lam, -700, 700))
    raise ValueError(struct)


def predict_g(cand: Candidate, inv_fields, gN, lam) -> np.ndarray:
    """The candidate's predicted radial acceleration at a set of probe points.

    This is the spherical reduction every channel of the tournament scores:
    the response enters through the eigenvalue of K along the measurement
    direction.  It is declared, not hidden -- the full tensor solve is the
    tensor lane's job and costs a PDE per candidate, which is exactly what a
    pre-data compiler must avoid.
    """
    W = response_W(cand, inv_fields)
    if cand.struct in ("scalar_a0", "depth"):
        a_eff = np.maximum(cand.a0 * (1.0 + cand.A * W), 1e-14 * cand.a0)
        return g_of_gN(cand.base, gN, a_eff)
    k_r = radial_eigen(cand.struct, lam, cand.A, W)
    return mond_invert(gN, k_r, cand.a0, cand.base)


# ============================================================ 3. geometry
class Plummer:
    """A closed-form spherical baryonic component.

    Everything the gates need -- Phi, g, the full Hessian and rho -- in closed
    form, so no PDE solve and no quadrature error enters the compiler.
    """
    __slots__ = ("M", "a", "c")

    def __init__(self, M, a, centre=(0.0, 0.0, 0.0)):
        self.M = float(M)
        self.a = float(a)
        self.c = np.asarray(centre, float)

    def phi(self, x):
        d = np.asarray(x, float) - self.c
        return -G * self.M / np.sqrt((d * d).sum(-1) + self.a ** 2)

    def g(self, x):
        d = np.asarray(x, float) - self.c
        s2 = (d * d).sum(-1) + self.a ** 2
        return -G * self.M * d / s2[..., None] ** 1.5

    def hess(self, x):
        d = np.asarray(x, float) - self.c
        s2 = (d * d).sum(-1) + self.a ** 2
        pre = G * self.M / s2 ** 1.5
        return pre[..., None, None] * (
            np.eye(3) - 3.0 * d[..., :, None] * d[..., None, :]
            / s2[..., None, None])

    def rho(self, x):
        d = np.asarray(x, float) - self.c
        s2 = (d * d).sum(-1) + self.a ** 2
        return 3.0 * self.M * self.a ** 2 / (4.0 * np.pi * s2 ** 2.5)

    def M_enc(self, r):
        r = np.asarray(r, float)
        return self.M * r ** 3 / (r ** 2 + self.a ** 2) ** 1.5

    def M_in_ball(self, d, L, n=256):
        """Mass inside a ball of radius L whose centre is a distance d from
        this component's centre.  Exact solid-angle overlap, 1-D quadrature.

            f(r', d, L) = 1                                if r' + d <= L
                        = 0                                if |d - r'| >= L
                        = (L^2 - (d-r')^2) / (4 d r')      otherwise
        """
        d = np.atleast_1d(np.asarray(d, float))
        rp = np.geomspace(1e-3 * self.a, 60.0 * self.a, n)
        dM = np.gradient(self.M_enc(rp), rp)
        f = np.clip((L ** 2 - (d[:, None] - rp[None, :]) ** 2)
                    / (4.0 * np.maximum(d[:, None], 1e-30) * rp[None, :]),
                    0.0, 1.0)
        f = np.where(np.abs(d[:, None] - rp[None, :]) >= L, 0.0, f)
        f = np.where(rp[None, :] + d[:, None] <= L, 1.0, f)
        return np.trapezoid(dM[None, :] * f, rp, axis=1)


def fib_dirs(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    rxy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], 1)


def sym_traceless(H: np.ndarray) -> np.ndarray:
    tr = np.trace(H, axis1=-2, axis2=-1)
    return H - tr[..., None, None] * np.eye(3) / 3.0


#: L_NL and M_0 of the nonlocal invariant: the tournament's declared globals.
L_NL = 300.0 * KPC
M_NL = 1.0e12 * MSUN


@dataclass
class Probe:
    """A synthetic probe geometry.  Nothing observational; every field is a
    closed-form evaluation of the declared components."""
    name: str
    centre: np.ndarray
    pts: np.ndarray                  # (P,3)
    rhat: np.ndarray                 # (P,3)
    r: np.ndarray                    # (P,)
    gN: np.ndarray                   # (P,)
    gvec: np.ndarray                 # (P,3)
    dhat: np.ndarray                 # (P,3)
    That: np.ndarray                 # (P,3,3)
    phi_inf: np.ndarray              # (P,)
    inv: Dict[str, np.ndarray]
    wells: Tuple[np.ndarray, np.ndarray]   # the CATALOGUE rows of this geometry
    note: str = ""

    def misalign_deg(self, struct: str) -> float:
        """Median angle between the structure's preferred axis and rhat."""
        if struct == "tensor_d":
            c = np.abs((self.dhat * self.rhat).sum(-1))
        elif struct in ("tensor_T", "tidal_const"):
            _, V = np.linalg.eigh(self.That)
            c = np.abs((V[..., -1] * self.rhat).sum(-1))
        else:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.median(c), -1.0, 1.0))))


def _qbar_of(comps: Sequence[Plummer], pts: np.ndarray) -> np.ndarray:
    Mb = np.zeros(len(pts))
    for c in comps:
        d = np.linalg.norm(pts - c.c, axis=-1)
        Mb += c.M_in_ball(d, L_NL)
    return Mb / (Mb + M_NL)


def make_probe(name: str, comps: Sequence[Plummer], centre, radii,
               wells: Tuple[np.ndarray, np.ndarray], ndir: int = 12,
               note: str = "") -> Probe:
    centre = np.asarray(centre, float)
    dirs = fib_dirs(ndir)
    radii = np.asarray(radii, float)
    pts = (centre[None, None, :] + dirs[None, :, :] * radii[:, None, None]
           ).reshape(-1, 3)
    rvec = pts - centre
    r = np.linalg.norm(rvec, axis=-1)
    rhat = rvec / r[:, None]
    phi = sum(c.phi(pts) for c in comps)
    gv = sum(c.g(pts) for c in comps)
    H = sum(c.hess(pts) for c in comps)
    rho = sum(c.rho(pts) for c in comps)
    gN = np.linalg.norm(gv, axis=-1)
    dhat = gv / np.maximum(gN, 1e-300)[:, None]
    T0 = sym_traceless(H)
    tnorm = np.sqrt((T0 * T0).sum((-1, -2)))
    That = T0 / np.maximum(tnorm, 1e-300)[..., None, None]
    inv = dict(one=np.ones_like(r), gn=gN / A0, phi=np.abs(phi),
               rhobar=np.maximum(rho, 1e-40),
               tidal=np.maximum(tnorm, 1e-45), qbar=_qbar_of(comps, pts))
    return Probe(name=name, centre=centre, pts=pts, rhat=rhat, r=r, gN=gN,
                 gvec=gv, dhat=dhat, That=That, phi_inf=phi, inv=inv,
                 wells=wells, note=note)


# --- declared probe parameters --------------------------------------------
# Chosen so the probes reproduce Run AH's recorded probe table (median |T| and
# median |Phi_N| for a cluster shell, an isolated field galaxy and a cluster
# member galaxy).  They are a caricature -- one gas component, one BCG, 300
# member rows -- and are declared here and nowhere else.
GAL_M = 5.0e10 * MSUN
GAL_A = 3.0 * KPC
CLU_GAS_M = 2.0e14 * MSUN
CLU_GAS_A = 900.0 * KPC
BCG_M = 3.0e12 * MSUN
BCG_A = 20.0 * KPC
N_MEMBERS = 300
MEMBER_D = 700.0 * KPC
NEIGHBOUR_D = 1.0 * MPC
CLU_R = 1000.0 * KPC

_PROBES: Dict[str, Probe] = {}


def _member_catalogue(seed: int = 20260904):
    """300 member galaxies drawn once from a declared Plummer profile."""
    rng = np.random.default_rng(seed)
    u = rng.random(N_MEMBERS) ** (1.0 / 3.0)
    d = rng.normal(size=(N_MEMBERS, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = d * (u * CLU_R)[:, None]
    m = 10 ** rng.uniform(9.0, 11.6, N_MEMBERS) * MSUN
    # the probe galaxy is member 0, placed at the declared radius
    x[0] = np.array([MEMBER_D, 0.0, 0.0])
    m[0] = GAL_M
    return x, m


def probes() -> Dict[str, Probe]:
    """The probe geometries the gates measure on, built once."""
    if _PROBES:
        return _PROBES
    gal = Plummer(GAL_M, GAL_A, (0, 0, 0))
    nbr = Plummer(GAL_M, GAL_A, (NEIGHBOUR_D, 0, 0))
    gas = Plummer(CLU_GAS_M, CLU_GAS_A, (0, 0, 0))
    bcg = Plummer(BCG_M, BCG_A, (0, 0, 0))
    mx, mm = _member_catalogue()
    members = [Plummer(mm[i], GAL_A * (mm[i] / GAL_M) ** 0.3, mx[i])
               for i in range(N_MEMBERS)]

    _PROBES["galaxy_field"] = make_probe(
        "galaxy_field", [gal, nbr], (0, 0, 0),
        np.geomspace(10 * KPC, 30 * KPC, 5),
        wells=(np.array([[0.0, 0, 0], [NEIGHBOUR_D, 0, 0]]),
               np.array([GAL_M, GAL_M])),
        note="isolated 5e10 Msun galaxy with one equal neighbour at 1 Mpc, "
             "sampled 10-30 kpc")
    _PROBES["cluster_shell"] = make_probe(
        "cluster_shell", [gas, bcg] + members, (0, 0, 0),
        np.geomspace(300 * KPC, 1414 * KPC, 5),
        wells=(np.vstack([[[0.0, 0, 0]], mx]),
               np.concatenate([[BCG_M], mm])),
        note="2e14 Msun gas + 3e12 BCG + 300 members, sampled 300-1414 kpc")
    _PROBES["galaxy_member"] = make_probe(
        "galaxy_member", [gas, bcg] + members, (MEMBER_D, 0, 0),
        np.geomspace(10 * KPC, 30 * KPC, 5),
        wells=(np.vstack([[[0.0, 0, 0]], mx]),
               np.concatenate([[BCG_M], mm])),
        note="the same galaxy at 700 kpc inside the cluster, sampled 10-30 "
             "kpc from its OWN centre")
    _PROBES["solar"] = make_probe(
        "solar", [Plummer(MSUN, 1.0e3, (0, 0, 0))], (0, 0, 0),
        np.array([AU]), wells=(np.zeros((1, 3)), np.array([MSUN])), ndir=6,
        note="1 Msun at 1 AU: the regime in which a constant K is exactly a "
             "redefinition of GM")
    return _PROBES


def probe_table() -> Dict[str, Dict[str, Any]]:
    """Median |T| and median |Phi_N| per probe -- Run AH's table, recomputed."""
    out = {}
    for nm, p in probes().items():
        out[nm] = dict(median_tidal=float(np.median(p.inv["tidal"])),
                       median_absPhi=float(np.median(p.inv["phi"])),
                       median_gn_over_a0=float(np.median(p.inv["gn"])),
                       median_qbar=float(np.median(p.inv["qbar"])),
                       n_points=int(len(p.r)),
                       r_kpc=[round(float(v), 3)
                              for v in np.unique(np.round(p.r / KPC, 6))])
    return out


# ---- the alignment / pair tensors, evaluated on a probe's own catalogue ----
def well_weight(Ma, ra, well: Dict[str, Any]) -> np.ndarray:
    """(M/M0)^p times a radial shape.  The family-C / tensor_S weight."""
    M0 = well.get("M0", 1e10 * MSUN)
    u = (np.asarray(Ma, float) / M0) ** well["p"]
    x = np.asarray(ra, float) / well["L"]
    if well.get("family", "plaw") == "plaw":
        return u * (1.0 + x ** well["q"]) ** (-well["s"])
    return u * np.exp(-(x ** well["q"]))


def S_wells(points, wx, wm, well: Dict[str, Any], eps: float = 1e-12,
            r_soft: float = 0.01 * KPC, exclude_nearest: bool = False
            ) -> np.ndarray:
    """S(x) = <w (n n^T - I/3)> / <|w|>, the normalised alignment tensor."""
    points = np.asarray(points, float)
    WX = np.asarray(wx, float)
    WM = np.asarray(wm, float)
    out = np.empty((len(points), 3, 3))
    eye = np.eye(3)[None]
    blk = max(1, int(4e6 // max(len(WX), 1)))
    for s0 in range(0, len(points), blk):
        P = points[s0:s0 + blk]
        d = WX[None, :, :] - P[:, None, :]
        r = np.maximum(np.sqrt((d * d).sum(-1)), r_soft)
        n = d / r[..., None]
        w = well_weight(WM[None, :], r, well)
        if exclude_nearest and WX.shape[0] > 1:
            j = np.argmin(r, axis=1)
            w = w.copy()
            w[np.arange(len(P)), j] = 0.0
        num = np.einsum("pb,pbi,pbj->pij", w, n, n)
        sw = w.sum(1)
        den = np.abs(w).sum(1)
        out[s0:s0 + blk] = ((num - sw[:, None, None] * eye / 3.0)
                            / (eps + den)[:, None, None])
    return out


def C_pairs(points, wx, wm, pair: Dict[str, Any],
            max_pairs: int = 4_000_000) -> np.ndarray:
    """The family-D pair tensor C(x).  Cost is O(P N^2) with no locality."""
    points = np.asarray(points, float)
    wx = np.asarray(wx, float)
    wm = np.asarray(wm, float)
    N = len(wm)
    if N * (N - 1) // 2 > max_pairs:
        raise MemoryError(f"family D needs {N * (N - 1) // 2:,} pairs for "
                          f"N = {N} rows; compiler limit {max_pairs:,}")
    ia, ib = np.triu_indices(N, 1)
    M0 = pair.get("M0", 1e10 * MSUN)
    sig_perp = pair.get("sigma_perp", 2.0 * KPC)
    sig_par = pair.get("sigma_par", 5.0 * KPC)
    d_soft = pair.get("d_soft", 0.02 * KPC)
    xa, xb = wx[ia], wx[ib]
    dv = xb - xa
    dab = np.maximum(np.linalg.norm(dv, axis=-1), d_soft)
    e = dv / dab[:, None]
    x = dab / pair["L"]
    w = (((wm[ia] * wm[ib]) / M0 ** 2) ** pair["p"]) * x ** (-pair["q"]) \
        * np.exp(-(x ** pair["s"]))
    mid = 0.5 * (xa + xb)
    out = np.zeros((len(points), 3, 3))
    blk = max(1, int(4e6 // max(len(ia), 1)))
    for s0 in range(0, len(points), blk):
        P = points[s0:s0 + blk]
        rel = P[:, None, :] - mid[None, :, :]
        dpar = (rel * e[None, :, :]).sum(-1)
        dperp2 = np.maximum((rel * rel).sum(-1) - dpar ** 2, 0.0)
        Wt = (np.exp(-dperp2 / (2 * sig_perp ** 2))
              * np.exp(-dpar ** 2 / (2 * sig_par ** 2)))
        out[s0:s0 + blk] = np.einsum("pk,ki,kj->pij", Wt * w[None, :], e, e)
    return out


_LAM_CACHE: Dict[Tuple, Dict[str, np.ndarray]] = {}


def probe_lambda(cand: Candidate, pname: str) -> np.ndarray:
    """rhat^T (traceless structure) rhat, MEASURED on the probe's catalogue.

    For `tensor_d` and `tensor_T` the axis is a functional of the smooth
    baryonic field and is computed from it.  For `tensor_S`, `wells` and
    `pairs` it is computed from the probe's own catalogue rows, which is the
    point: the axis is set by the cataloguer.
    """
    p = probes()[pname]
    st = cand.struct
    if st == "tensor_d":
        return (p.dhat * p.rhat).sum(-1) ** 2 - 1.0 / 3.0
    if st == "tensor_T":
        return np.einsum("pi,pij,pj->p", p.rhat, p.That, p.rhat)
    if st == "tidal_const":
        # family E's REGULARISED normalisation, exactly `families.tidal_hat`:
        #   That = T0 / sqrt(eps_T^2 + |T0|_F^2)
        # so the response is a smooth function of the tidal magnitude and
        # does NOT saturate at the spectral bound in weak-tide regions.  This
        # is the regulator Run AG found suppressing the anisotropy 97x in
        # galaxy outskirts.
        eps = (cand.tidal_const or {}).get("eps_T", EPS_T_DEFAULT)
        Tn = p.inv["tidal"]
        return (np.einsum("pi,pij,pj->p", p.rhat, p.That, p.rhat)
                * Tn / np.sqrt(eps ** 2 + Tn ** 2))
    if st in ("tensor_S", "wells"):
        w = cand.well or dict(family="plaw", p=1.0, q=2.0, s=1.0, L=10.0 * KPC)
        key = (pname, "S", w.get("family"), w["p"], w["q"], w["s"], w["L"],
               bool(w.get("exclude_nearest")))
        if key not in _LAM_CACHE:
            S = S_wells(p.pts, *p.wells, w,
                        exclude_nearest=bool(w.get("exclude_nearest")))
            _LAM_CACHE[key] = np.einsum("pi,pij,pj->p", p.rhat, S, p.rhat)
        return _LAM_CACHE[key]
    if st == "pairs":
        pr = cand.pair or dict(p=1.0, q=1.0, s=2.0, L=10.0 * KPC)
        key = (pname, "C", pr["p"], pr["q"], pr["s"], pr["L"])
        if key not in _LAM_CACHE:
            C = C_pairs(p.pts, *p.wells, pr)
            _LAM_CACHE[key] = np.einsum("pi,pij,pj->p", p.rhat, C, p.rhat)
        return _LAM_CACHE[key]
    return np.zeros(len(p.r))


# ==================================================== 4. GATE 1 machinery
def invsqrtm_sym(K: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(K)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def constant_K_stretch_demo(K: Optional[np.ndarray] = None,
                            n_src: int = 400, n_fld: int = 300,
                            seed: int = 20260904) -> Dict[str, Any]:
    """THE THEOREM, IMPLEMENTED AND MEASURED.

    For a constant symmetric positive-definite K the exact solution of

        div[K grad Phi] = 4 pi G rho

    for a point mass is  Phi = -GM / (sqrt(det K) sqrt(r^T K^-1 r)).  The claim
    the gate rests on is that this is IDENTICALLY the plain Newtonian potential
    of a STRETCHED source: with

        x' = K^(-1/2) x        m'_a = m_a / sqrt(det K)

    one has  Phi_K(x) = Phi_Newton(x'; {x'_a, m'_a})  for every x.

    This routine builds a genuinely distributed, genuinely triaxial source,
    evaluates BOTH sides on an independent set of field points by two
    different expressions, and reports the residual.  It then quantifies WHAT
    the constant K is degenerate with:
      * sqrt(det K) multiplies GM -- a rescaling of the stellar mass-to-light
        ratio, of the distance, or of G;
      * the eigenvalue ratios are an apparent source axis ratio -- degenerate
        with source ellipticity, inclination and line-of-sight depth.
    """
    rng = np.random.default_rng(seed)
    if K is None:
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        K = Q @ np.diag([1.7, 0.9, 0.55]) @ Q.T
        K = 0.5 * (K + K.T)
    K = np.asarray(K, float)
    w = np.linalg.eigvalsh(K)
    assert w.min() > 0, "K must be positive definite"
    detK = float(np.linalg.det(K))
    Kinv = np.linalg.inv(K)
    Kmh = invsqrtm_sym(K)

    xa = rng.normal(size=(n_src, 3)) * np.array([3.0, 1.5, 0.8]) * KPC
    ma = rng.random(n_src) * 1e8 * MSUN + 1e7 * MSUN
    x = fib_dirs(n_fld) * (10.0 + 40.0 * rng.random(n_fld))[:, None] * KPC

    dx = x[:, None, :] - xa[None, :, :]
    q = np.einsum("fsi,ij,fsj->fs", dx, Kinv, dx)
    phi_K = -(G / math.sqrt(detK)) * (ma[None, :] / np.sqrt(q)).sum(1)

    xp_, xap = x @ Kmh.T, xa @ Kmh.T
    map_ = ma / math.sqrt(detK)
    dxp = xp_[:, None, :] - xap[None, :, :]
    phi_N = -G * (map_[None, :] / np.sqrt((dxp * dxp).sum(-1))).sum(1)

    resid = float(np.max(np.abs(phi_K - phi_N)) / np.max(np.abs(phi_K)))
    sm = np.linalg.eigvalsh(Kmh)
    return dict(
        K_eigenvalues=[float(v) for v in w], det_K=detK,
        residual_rel=resid, n_source=n_src, n_field=n_fld,
        equivalent_source_axis_ratio=float(sm.min() / sm.max()),
        equivalent_log10_ML_offset_dex=float(0.5 * math.log10(detK)),
        Upsilon_star_uncertainty_dex=0.06,
        statement=("Phi_K(x) == Phi_Newton(K^-1/2 x) for the source stretched "
                   "by K^-1/2 and rescaled by 1/sqrt(det K), to "
                   f"{resid:.2e} relative. A constant K is a coordinate "
                   "stretch plus a transformed source and carries no "
                   "observable of its own."))


#: Bounds on the DEGENERATE alternative, when it is required to be physically
#: available rather than merely mathematically present.  A stretch is
#: degenerate with the stellar mass-to-light ratio (the mid-IR route pins
#: Upsilon* to 0.06 dex, so 0.30 dex is generous), and with inclination,
#: line-of-sight depth and baryonic deprojection (a factor 3 in an apparent
#: conductivity is generous).  The gate's VERDICT uses the UNBOUNDED fit,
#: which is the conservative choice -- a candidate escapes only if not even an
#: arbitrary stretch reproduces it -- and the bounded residual is reported
#: alongside so it is visible how much of an escape rests on an implausible
#: stretch.
STRETCH_BOUND_LN_ALPHA = 0.30 * math.log(10.0)
STRETCH_BOUND_LN_K = math.log(3.0)


def _gauss_newton(model, y, n_par: int, n_iter: int = 25, tol: float = 1e-11,
                  ridge: float = 1e-10, bounds=None):
    """Levenberg-free Gauss-Newton with a ridged normal-equation solve.

    A ridge rather than a pseudo-inverse because the two parameters of the
    degenerate model are exactly collinear in the Newtonian regime (there
    g depends only on alpha/k) and only separate across the MOND transition.
    The RESIDUAL is well defined either way -- it is a projection -- but the
    parameters are not, which is itself part of what the gate is saying.
    """
    theta = np.zeros(n_par)
    best = (np.inf, theta.copy())
    for _ in range(n_iter):
        f0 = model(theta)
        if not np.all(np.isfinite(f0)):
            break
        r = y - f0
        rms = float(np.sqrt(np.mean(r ** 2)))
        if rms < best[0]:
            best = (rms, theta.copy())
        J = np.empty((len(r), n_par))
        for j in range(n_par):
            th = theta.copy()
            th[j] += 1e-5
            J[:, j] = (model(th) - f0) / 1e-5
        if not np.all(np.isfinite(J)):
            break
        A = J.T @ J
        A[np.diag_indices_from(A)] += ridge * max(np.trace(A), 1e-300)
        try:
            step = np.linalg.solve(A, J.T @ r)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -5.0, 5.0)
        theta = np.clip(theta + step, -300.0, 300.0)
        if bounds is not None:
            theta = np.clip(theta, bounds[0], bounds[1])
        if np.max(np.abs(step)) < tol:
            break
    f = model(theta)
    if np.all(np.isfinite(f)):
        rms = float(np.sqrt(np.mean((y - f) ** 2)))
        if rms < best[0]:
            best = (rms, theta.copy())
    return best


def _fit_stretch(logg_target, gN, a0: float, base: str):
    """Best 'undistorted law acting on a stretched source' fit.

    The degenerate alternative has exactly two constants: a uniform
    conductivity k (what a constant K contributes) and a source rescaling
    alpha (what stretching the source contributes -- mass-to-light,
    inclination, line-of-sight depth, distance).  Both are GLOBAL over the
    probe by construction, because a coordinate stretch is a global linear map.
    """
    def model(th):
        return np.log10(np.maximum(
            mond_invert(math.exp(th[1]) * gN, math.exp(th[0]), a0, base),
            1e-300))
    free = _gauss_newton(model, logg_target, 2)
    lo = np.array([-STRETCH_BOUND_LN_K, -STRETCH_BOUND_LN_ALPHA])
    bounded = _gauss_newton(model, logg_target, 2, bounds=(lo, -lo))
    return free[0], free[1], bounded[0]


def _fit_stretch_joint(targets, a0: float, base: str):
    """One SHARED conductivity k across probes, one alpha PER probe.

    A coordinate stretch is a single global linear map, so k is common; the
    source mass is a per-object quantity, so alpha is not.  If this fit cannot
    reproduce the candidate on every probe at once, the probes disagree in a
    way no coordinate stretch can produce -- escape (c).
    """
    nP = len(targets)
    gNs = [g for _, g in targets]
    y = np.concatenate([lg for lg, _ in targets])

    def model(th):
        k = math.exp(th[0])
        return np.concatenate([
            np.log10(np.maximum(
                mond_invert(math.exp(th[1 + i]) * gNs[i], k, a0, base),
                1e-300)) for i in range(nP)])
    free = _gauss_newton(model, y, 1 + nP)
    lo = np.array([-STRETCH_BOUND_LN_K]
                  + [-STRETCH_BOUND_LN_ALPHA] * nP)
    bounded = _gauss_newton(model, y, 1 + nP, bounds=(lo, -lo))
    return free[0], free[1], bounded[0]


GATE1_PROBES = ("galaxy_field", "cluster_shell", "galaxy_member")


def gate1(cand: Candidate) -> Tuple[bool, Dict[str, Any], str]:
    """GATE 1 -- constant-K degeneracy."""
    P = probes()
    if not cand.responds():
        return (True, dict(responds=False),
                "base law: K = I identically, so there is no tensor to be "
                "degenerate with a coordinate stretch")

    per_probe, targets = {}, []
    for nm in GATE1_PROBES:
        p = P[nm]
        lam = probe_lambda(cand, nm)
        g = predict_g(cand, p.inv, p.gN, lam)
        lg = np.log10(np.maximum(g, 1e-300))
        rms, th, rms_b = _fit_stretch(lg, p.gN, cand.a0, cand.base)
        W = response_W(cand, p.inv)
        kr = radial_eigen(cand.struct, lam, cand.A, W)
        per_probe[nm] = dict(
            resid_dex=rms, resid_dex_bounded_stretch=rms_b,
            ln_k_fit=float(th[0]), ln_alpha_fit=float(th[1]),
            spread_ln_k_r=float(np.log(max(kr.max(), 1e-300))
                                - np.log(max(kr.min(), 1e-300))),
            W_min=float(W.min()), W_max=float(W.max()),
            median_k_r=float(np.median(kr)),
            median_lambda=float(np.median(lam)))
        targets.append((lg, p.gN))

    joint_rms, _, joint_rms_b = _fit_stretch_joint(targets, cand.a0,
                                                   cand.base)
    prov, prov_why = AXIS_PROVENANCE.get(cand.struct, ("none", ""))
    misal = {nm: P[nm].misalign_deg(cand.struct) for nm in GATE1_PROBES}
    best_misal = max(misal.values()) if misal else 0.0
    max_single = max(v["resid_dex"] for v in per_probe.values())

    esc_a = max_single > TOL_DEX
    esc_b = prov in ("source", "external") and best_misal > TOL_MISALIGN_DEG
    esc_c = joint_rms > TOL_DEX
    passed = bool(esc_a or esc_b or esc_c)
    escapes = [n for n, ok in (("a_spatial_variation", esc_a),
                               ("b_independent_axis", esc_b),
                               ("c_probe_disagreement", esc_c)) if ok]
    val = dict(per_probe=per_probe, joint_resid_dex=joint_rms,
               joint_resid_dex_bounded_stretch=joint_rms_b,
               max_single_probe_resid_dex=max_single,
               max_single_probe_resid_dex_bounded_stretch=max(
                   v["resid_dex_bounded_stretch"]
                   for v in per_probe.values()),
               stretch_bounds=dict(
                   max_log10_source_rescale=STRETCH_BOUND_LN_ALPHA
                   / math.log(10.0),
                   max_conductivity_factor=math.exp(STRETCH_BOUND_LN_K)),
               axis_provenance=prov, axis_provenance_note=prov_why,
               axis_misalignment_deg=misal, escapes=escapes,
               tol_dex=TOL_DEX, tol_misalign_deg=TOL_MISALIGN_DEG)
    if passed:
        why = (f"escapes {', '.join(escapes)}; max single-probe stretch "
               f"residual {max_single:.4f} dex, joint (one global stretch, "
               f"per-object mass) {joint_rms:.4f} dex, axis provenance "
               f"'{prov}' misaligned by up to {best_misal:.1f} deg")
    else:
        why = (f"NO escape: the prediction on every probe is reproduced by an "
               f"undistorted law acting on a stretched source to "
               f"{max_single:.2e} dex (single probe) and {joint_rms:.2e} dex "
               f"(one global stretch across probes), both below the "
               f"{TOL_DEX} dex tolerance; axis provenance '{prov}' "
               f"({prov_why}), misalignment {best_misal:.1f} deg. Degenerate "
               f"with source ellipticity, inclination, line-of-sight depth, "
               f"distance and baryonic deprojection.")
    return passed, val, why


# ==================================================== 5. GATE 2 machinery
#: Cosmic mean BARYON density, declared: Omega_b rho_crit at h = 0.7.
RHO_B_MEAN = 0.0493 * 9.204e-27         # kg m^-3
OVERDENSITY = 200.0
ENV_VOLUME_RADIUS = 5.0 * MPC
SCALE_RADIUS_MULTIPLE = 10.0

R_TRUNC_FLAT = 1.0 * MPC

#: The four rules the brief names, plus the two GLOBAL rules Run AH compared.
PHI_RULES_BRIEF = ("saddle", "overdensity", "scale_radius", "env_volume")
PHI_RULES = PHI_RULES_BRIEF + ("inf", "flat_1Mpc")
PHI_RULE_DOC = {
    "saddle": "|Phi_N(x) - Phi_N(x_saddle)| with x_saddle the nearest "
              "gravitational saddle of the total baryonic potential",
    "overdensity": f"referenced to the radius at which the smoothed baryon "
                   f"density falls to {OVERDENSITY:g} x the cosmic mean "
                   f"baryon density",
    "scale_radius": f"referenced to {SCALE_RADIUS_MULTIPLE:g} x the baryonic "
                    f"scale radius",
    "env_volume": f"referenced to the edge of a reconstructed environmental "
                  f"volume of radius {ENV_VOLUME_RADIUS / MPC:g} Mpc",
    "inf": "Phi -> 0 at infinity with the baryons continued outside the last "
           "measured point as a POINT MASS (the tournament's declared "
           "PRIMARY; Run AH's first global rule)",
    "flat_1Mpc": f"Phi -> 0 at infinity with the baryons continued outside "
                 f"the last measured point as a FLAT ROTATION CURVE truncated "
                 f"at a universal R_trunc = {R_TRUNC_FLAT / MPC:g} Mpc "
                 f"(Run AH's second global rule)",
}


def _saddle_x(m1, a1, m2, a2, D, n=400):
    """The gravitational saddle of two Plummer spheres on the line joining
    them: where the two accelerations cancel."""
    xs = np.linspace(1e-3, 1 - 1e-3, n) * D

    def f(x):
        return (G * m1 * x / (x ** 2 + a1 ** 2) ** 1.5
                - G * m2 * (D - x) / ((D - x) ** 2 + a2 ** 2) ** 1.5)

    v = f(xs)
    idx = np.where(np.diff(np.sign(v)) != 0)[0]
    if len(idx) == 0:
        return 0.5 * D
    x0, x1 = xs[idx[0]], xs[idx[0] + 1]
    s0 = np.sign(v[idx[0]])
    for _ in range(80):
        xm = 0.5 * (x0 + x1)
        if np.sign(f(xm)) == s0:
            x0 = xm
        else:
            x1 = xm
    return 0.5 * (x0 + x1)


def _overdensity_radius(M, a):
    """Radius at which a Plummer sphere's density reaches the threshold."""
    thr = OVERDENSITY * RHO_B_MEAN
    s2 = (3.0 * M * a ** 2 / (4.0 * np.pi * thr)) ** 0.4
    return float(np.sqrt(max(s2 - a ** 2, (0.1 * a) ** 2)))


def phi_depth_population(n: int = 400, seed: int = 20260904
                         ) -> Dict[str, np.ndarray]:
    """A synthetic galaxy population and its potential depth under every rule.

    NOTHING observational: masses are drawn log-uniform over the range the
    programme's galaxy channels cover, sizes from a declared mass-size
    relation, and each galaxy is given one neighbour at a declared separation.
    The point is not the population -- it is the SPREAD BETWEEN RULES, which is
    a property of the rules and not of the sample.
    """
    rng = np.random.default_rng(seed)
    M = 10 ** rng.uniform(9.0, 11.7, n) * MSUN
    a = GAL_A * (M / GAL_M) ** 0.30
    D = 10 ** rng.uniform(np.log10(0.3), np.log10(3.0), n) * MPC
    Mn = 10 ** rng.uniform(9.0, 11.7, n) * MSUN
    an = GAL_A * (Mn / GAL_M) ** 0.30

    def phi_tot(s, i):
        return (-G * M[i] / np.sqrt(s ** 2 + a[i] ** 2)
                - G * Mn[i] / np.sqrt((D[i] - s) ** 2 + an[i] ** 2))

    r_probe = 5.0 * a
    phi_probe = np.array([phi_tot(r_probe[i], i) for i in range(n)])
    xs = np.array([_saddle_x(M[i], a[i], Mn[i], an[i], D[i]) for i in range(n)])
    r200 = np.array([_overdensity_radius(M[i], a[i]) for i in range(n)])
    refs = {
        "saddle": np.array([phi_tot(xs[i], i) for i in range(n)]),
        "overdensity": np.array([phi_tot(r200[i], i) for i in range(n)]),
        "scale_radius": np.array([phi_tot(SCALE_RADIUS_MULTIPLE * a[i], i)
                                  for i in range(n)]),
        "env_volume": np.array([phi_tot(ENV_VOLUME_RADIUS, i)
                                for i in range(n)]),
        "inf": np.zeros(n),
    }
    out = {k: np.abs(phi_probe - v) for k, v in refs.items()}
    # the FLAT continuation: outside r_probe the baryons are continued as a
    # flat rotation curve of speed v_c(r_probe) truncated at R_TRUNC_FLAT,
    # which adds v_c^2 ln(R_trunc / r_probe) to the depth.  This is Run AH's
    # second admissible global rule and the source of its 0.87 dex.
    vc2 = G * M * r_probe ** 2 / (r_probe ** 2 + a ** 2) ** 1.5
    out["flat_1Mpc"] = np.abs(phi_probe) + vc2 * np.log(
        np.maximum(R_TRUNC_FLAT / r_probe, 1.0))
    out["_M_Msun"] = M / MSUN
    out["_r200_kpc"] = r200 / KPC
    out["_saddle_kpc"] = xs / KPC
    return out


_PHI_POP: Dict[str, np.ndarray] = {}


def phi_rule_spread() -> Dict[str, Any]:
    """Median galaxy potential depth under each rule, and the spread in dex."""
    global _PHI_POP
    if not _PHI_POP:
        _PHI_POP = phi_depth_population()
    med = {k: float(np.median(_PHI_POP[k])) for k in PHI_RULES}

    def _spread(keys):
        v = np.array([med[k] for k in keys])
        v = v[v > 0]
        return float(np.log10(v.max() / v.min())) if len(v) > 1 else 0.0

    return dict(median_absPhi_by_rule=med, spread_dex=_spread(PHI_RULES),
                spread_dex_brief_four=_spread(PHI_RULES_BRIEF),
                spread_dex_run_AH_pair=_spread(("inf", "flat_1Mpc")),
                rules=list(PHI_RULES), n_rules=len(PHI_RULES),
                doc=PHI_RULE_DOC,
                run_AH_recorded_spread_dex=RUN_AH_PHI_SPREAD_DEX,
                gate_margin_dex=RUN_AH_GATE_MARGIN_DEX)


def gate2(cand: Candidate) -> Tuple[bool, Dict[str, Any], str]:
    """GATE 2 -- potential gauge.  A LOUD FLAG, never an elimination."""
    if cand.inv not in GAUGE_DEPENDENT_INVARIANTS or not cand.responds():
        note = {"tidal": "|T| is a local second derivative of Phi_N with no "
                         "boundary constant",
                "gn": "|g_N| is a first derivative of Phi_N",
                "rhobar": "rho is the Poisson source itself",
                "qbar": "an enclosed mass fraction on a declared global "
                        "smoothing length",
                "one": "no invariant"}.get(cand.inv, "not potential-based")
        return (True, dict(gauge_dependent=False, invariant=cand.inv),
                f"invariant '{cand.inv}' carries no undetermined additive "
                f"constant: {note}")

    sp = phi_rule_spread()
    med = sp["median_absPhi_by_rule"]
    verdicts, Wgal = {}, {}
    for rule in PHI_RULES:
        I = np.array([med[rule]])
        W = (W_of(cand.form, I / cand.I0, cand.m)
             if cand.form != "off" else np.zeros(1))
        Wgal[rule] = float(W[0])
        verdicts[rule] = bool(abs(cand.A * W[0]) > 1.0e-3)
    flips = len(set(verdicts.values())) > 1
    wv = np.array([Wgal[r] for r in PHI_RULES])
    wp = wv[wv > 0]
    wspread = float(np.log10(wp.max() / wp.min())) if len(wp) > 1 else 0.0

    val = dict(gauge_dependent=True, invariant=cand.inv,
               median_absPhi_by_rule=med, phi_spread_dex=sp["spread_dex"],
               W_by_rule=Wgal, W_spread_dex=wspread,
               gate_fires_in_galaxies_by_rule=verdicts,
               verdict_changes_across_rules=flips,
               declared_primary=cand.phi_rule,
               run_AH_recorded_spread_dex=RUN_AH_PHI_SPREAD_DEX,
               gate_margin_dex=RUN_AH_GATE_MARGIN_DEX)
    why = (f"CONVENTION-DEPENDENT. |Phi_N| is defined only up to a constant. "
           f"Across {len(PHI_RULES)} defensible operational rules the median "
           f"galaxy depth spans {sp['spread_dex']:.2f} dex (Run AH measured "
           f"0.87 dex between its two global rules, against an off/on margin "
           f"of 0.90 dex), and this candidate's own response spans "
           f"{wspread:.2f} dex. "
           + ("THE ON/OFF VERDICT ITSELF CHANGES across defensible rules: "
              "this candidate's galaxy screens are decided by the boundary "
              "convention, not by the law."
              if flips else
              "The on/off verdict is stable across the rules tested, but the "
              "response amplitude is not, so every headline number must be "
              "quoted with the rule attached."))
    return (True, val, why)


# ==================================================== 6. GATE 3 machinery
#
# SIGN CONVENTION -- stated in full, because two lanes measured different
# numbers under different definitions and could not reconcile them.
#
#   drift(N; L) = max over probe points of ||K_N - K_ref||_inf / ||K_ref||_inf
#
# where K_N is the response evaluated on an N-row partition of ONE FIXED
# continuous mass distribution and K_ref is the SAME response evaluated on the
# full 16,384-point quadrature cloud, i.e. the N -> infinity reference.  N is
# held FIXED and L -- THE LAW'S OWN COHERENCE LENGTH, a parameter of the
# candidate -- is swept.  The reported slope is
#
#       d ln(drift) / d ln L      at fixed N,   L a parameter of the LAW.
#
#   NEGATIVE  -> widening the law's own kernel buys accuracy at fixed
#                catalogue resolution: the discreteness is set by a physical
#                length.  Screen-lane reference: -3.11, a genuine kernel.
#   NEAR ZERO -> the drift is set by the distance to the nearest ROW, which no
#                parameter of the law controls.  Screen-lane references:
#                -0.55 for family C, +0.12 for pure row counting.
#
# The tournament lane's `coarse.py` reports a DIFFERENT quantity: the slope of
# |S(N_{k+1}) - S(N_k)| against the mean nearest-neighbour spacing of the
# PARTITION.  There the swept variable belongs to the catalogue, not to the
# law, and refining the partition drives the spacing and the successive step
# down together, so any convergent law gives a POSITIVE slope of about +1.
# That is why it measured +1.0 to +1.5 and could not confirm a match: the two
# numbers are slopes of different functions of different variables.
# `coherence_slope()` computes the screen-lane quantity, `successive_step_
# slope()` the tournament-lane one, on the same candidates, so the difference
# is exhibited rather than argued.

def plummer_cloud(Nq: int = 16384, M: float = GAL_M, a: float = GAL_A,
                  centre=(0, 0, 0), umax: float = 0.98,
                  seed: int = 20260903):
    """A deterministic equal-mass quadrature cloud for a Plummer sphere.

    Equal-mass because a greedy equal-mass partition cannot balance below the
    mass of a single quadrature point; unequal masses make 'refinement'
    silently stop refining, which looks exactly like a converged law.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(Nq) + 0.5) / Nq * umax
    r = a * u ** (1 / 3) / np.sqrt(np.maximum(1 - u ** (2 / 3), 1e-12))
    r = r[rng.permutation(Nq)]
    return fib_dirs(Nq) * r[:, None] + np.asarray(centre, float), \
        np.full(Nq, M / Nq)


def nested_partitions(pts, m, Ns) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Greedy equal-mass partition, snapshotted at every requested N.

    Successive N are genuine REFINEMENTS of one another: cell k of the
    N-partition is a union of cells of the N'-partition for every N' > N.  A
    fresh random resampling at each N would confound refinement with sampling
    noise.
    """
    Ns = sorted(set(int(x) for x in Ns))
    pts = np.asarray(pts, float)
    m = np.asarray(m, float)
    cells, masses = [np.arange(len(m))], [float(m.sum())]
    out, target = {}, list(Ns)
    while target:
        while len(cells) < target[0]:
            k = int(np.argmax(masses))
            if masses[k] <= 0:
                break
            idx = cells[k]
            if len(idx) < 2:
                masses[k] = -1.0
                if all(x <= 0 for x in masses):
                    break
                continue
            P, w = pts[idx], m[idx]
            ax = int(np.argmax(P.max(0) - P.min(0)))
            order = np.argsort(P[:, ax], kind="stable")
            cw = np.cumsum(w[order])
            cut = min(max(int(np.searchsorted(cw, cw[-1] / 2.0)) + 1, 1),
                      len(order) - 1)
            a_, b_ = idx[order[:cut]], idx[order[cut:]]
            cells[k], masses[k] = a_, float(m[a_].sum())
            cells.append(b_)
            masses.append(float(m[b_].sum()))
        N = target.pop(0)
        good = [c for c in cells if len(c)]
        wx = np.array([(pts[c] * m[c, None]).sum(0) / m[c].sum() for c in good])
        wm = np.array([m[c].sum() for c in good])
        assert abs(wm.sum() - m.sum()) / m.sum() < 1e-10, "partition lost mass"
        assert wx.shape == (len(good), 3) and wm.shape == (len(good),)
        out[N] = (wx, wm)
    return out


def smooth_rho(points, wx, wm, L) -> np.ndarray:
    """rho_L(x) = sum_a M_a (2 pi L^2)^-3/2 exp(-|x-x_a|^2 / 2 L^2).

    A law built on this has a GENUINE coherence length: the sum is a quadrature
    of a convolution, so its error is O((l/L)^2).  The positive control for the
    physical branch of the coherence test.
    """
    points = np.asarray(points, float)
    wx = np.asarray(wx, float)
    wm = np.asarray(wm, float)
    out = np.empty(len(points))
    pre = (2 * np.pi * L ** 2) ** -1.5
    blk = max(1, int(4e6 // max(len(wx), 1)))
    for s0 in range(0, len(points), blk):
        P = points[s0:s0 + blk]
        d2 = ((wx[None, :, :] - P[:, None, :]) ** 2).sum(-1)
        out[s0:s0 + blk] = (wm[None, :] * pre * np.exp(-d2 / (2 * L ** 2))
                            ).sum(1)
    return out


def count_wells(points, wx, wm, L) -> np.ndarray:
    """n_wells within L.  Pure catalogue counting -- no mass weighting."""
    points = np.asarray(points, float)
    wx = np.asarray(wx, float)
    out = np.empty(len(points))
    blk = max(1, int(4e6 // max(len(wx), 1)))
    for s0 in range(0, len(points), blk):
        P = points[s0:s0 + blk]
        d2 = ((wx[None, :, :] - P[:, None, :]) ** 2).sum(-1)
        out[s0:s0 + blk] = (d2 < L ** 2).sum(1)
    return out


def _K_wells(pts, wx, wm, L_kpc, sT=0.5, p=1.0, q=2.0, s=1.0, family="plaw"):
    S = S_wells(pts, wx, wm, dict(family=family, p=p, q=q, s=s, L=L_kpc * KPC))
    w, V = np.linalg.eigh(sT * S)
    return np.einsum("pij,pj,pkj->pik", V, np.exp(w), V)


def _K_smooth(pts, wx, wm, L_kpc, rho0=4.0e-22, sc=0.5):
    k = np.exp(np.minimum(sc * smooth_rho(pts, wx, wm, L_kpc * KPC) / rho0, 40))
    return k[:, None, None] * np.eye(3)[None]


def _K_count(pts, wx, wm, L_kpc, N0=50.0, sc=0.02):
    k = np.exp(np.minimum(sc * count_wells(pts, wx, wm, L_kpc * KPC) / N0, 40))
    return k[:, None, None] * np.eye(3)[None]


COHERENCE_LAWS = {
    "X4_smooth_density": _K_smooth,     # genuine kernel,    reference -3.11
    "C1_wells_pow_p1":   _K_wells,      # family C p = 1,    reference -0.55
    "X2_count_wells":    _K_count,      # pure row counting, reference +0.12
}
COHERENCE_REFERENCE = {"X4_smooth_density": -3.11,
                       "C1_wells_pow_p1": -0.55,
                       "X2_count_wells": +0.12}


def _coherence_probe(nr=8, ndir=16):
    rs = np.geomspace(0.5 * KPC, 40.0 * KPC, nr)
    return (fib_dirs(ndir)[None, :, :] * rs[:, None, None]).reshape(-1, 3)


def coherence_slope(law: Callable, Ls_kpc=(2.0, 4.0, 8.0, 16.0, 32.0),
                    Nfix=(100, 1000), Nq: int = 16384,
                    nr: int = 8, ndir: int = 16) -> Dict[str, Any]:
    """d ln(drift) / d ln L at FIXED N, sweeping the LAW's own length.

    Geometry matched to the screen lane: a 5e10 Msun / 3 kpc Plummer cloud of
    `Nq` equal-mass quadrature points; probe points on `nr` log-spaced radii
    from 0.5 to 40 kpc times `ndir` Fibonacci directions.
    """
    qx, qm = plummer_cloud(Nq=Nq)
    pts = _coherence_probe(nr, ndir)
    parts = nested_partitions(qx, qm, Nfix)
    rows = {}
    for L in Ls_kpc:
        ref = law(pts, qx, qm, L)
        nref = max(np.abs(ref).max(), 1e-300)
        rows[float(L)] = {int(N): float(np.abs(law(pts, *parts[N], L) - ref
                                               ).max() / nref) for N in Nfix}
    slopes = {}
    for N in Nfix:
        xs = [math.log(L) for L in Ls_kpc if rows[float(L)][N] > 0]
        ys = [math.log(rows[float(L)][N]) for L in Ls_kpc
              if rows[float(L)][N] > 0]
        slopes[int(N)] = (float(np.polyfit(xs, ys, 1)[0])
                          if len(xs) >= 3 else float("nan"))
    good = [v for v in slopes.values() if np.isfinite(v)]
    return dict(L_kpc=list(Ls_kpc), drift=rows, slope_by_N=slopes,
                mean_slope=float(np.mean(good)) if good else float("nan"),
                convention="d ln(drift)/d ln L at FIXED N; L is the LAW's own "
                           "coherence length; drift is measured against the "
                           "full-cloud reference")


def successive_step_slope(law: Callable, L_kpc: float = 8.0,
                          Ns=(4, 10, 40, 100, 400, 1000, 4000),
                          Nq: int = 16384, nr: int = 8, ndir: int = 16
                          ) -> Dict[str, Any]:
    """The TOURNAMENT lane's quantity, computed here for comparison only.

    d ln |drift between successive N| / d ln (mean nearest-neighbour spacing
    of the PARTITION).  The swept variable belongs to the catalogue, not to
    the law, so a merely convergent response gives a POSITIVE slope.
    """
    qx, qm = plummer_cloud(Nq=Nq)
    pts = _coherence_probe(nr, ndir)
    parts = nested_partitions(qx, qm, Ns)
    vals, spac = [], []
    for N in Ns:
        wx, wm = parts[N]
        vals.append(law(pts, wx, wm, L_kpc))
        if len(wx) > 1:
            d = np.sqrt(((wx[:, None, :] - wx[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            spac.append(float(d.min(1).mean()))
        else:
            spac.append(float("nan"))
    dr = [float(np.abs(vals[i + 1] - vals[i]).max()
                / max(np.abs(vals[i]).max(), 1e-300))
          for i in range(len(vals) - 1)]
    Lnn = np.array(spac[1:])
    ok = (np.array(dr) > 0) & np.isfinite(Lnn)
    slope = (float(np.polyfit(np.log(Lnn[ok]), np.log(np.array(dr)[ok]), 1)[0])
             if ok.sum() >= 2 else float("nan"))
    return dict(N=list(Ns), successive_drift=dr,
                nn_spacing_kpc=[s / KPC for s in spac], slope=slope,
                convention="d ln|K(N_k+1)-K(N_k)| / d ln(nearest-neighbour "
                           "spacing of the PARTITION); the swept variable "
                           "belongs to the catalogue, not to the law")


def uniform_refinement(well: Dict[str, Any], ps=(0.5, 1.0, 2.0),
                       Nq: int = 16384, Ns=(1, 10, 100, 1000, 10000),
                       nr: int = 8, ndir: int = 16, sT: float = 0.5
                       ) -> Dict[str, Any]:
    """The WEAK test.  A mass exponent cancels EXACTLY under uniform
    equal-mass refinement, because (M/N M0)^p is common to the numerator and
    the denominator of a normalised direction average and divides out.
    Reproduced here: p = 0.5, 1, 2 must give the same drift to five figures.
    """
    qx, qm = plummer_cloud(Nq=Nq)
    pts = _coherence_probe(nr, ndir)
    parts = nested_partitions(qx, qm, Ns)
    def Kof(S):
        w_, V = np.linalg.eigh(sT * S)
        return np.einsum("pij,pj,pkj->pik", V, np.exp(w_), V)

    out, outK = {}, {}
    for p in ps:
        w = dict(well)
        w["p"] = p
        ref = S_wells(pts, qx, qm, w)
        nref = max(np.abs(ref).max(), 1e-300)
        Kref = Kof(ref)
        nKref = max(np.abs(Kref).max(), 1e-300)
        out[float(p)] = {}
        outK[float(p)] = {}
        for N in Ns:
            S = S_wells(pts, *parts[N], w)
            out[float(p)][int(N)] = float(np.abs(S - ref).max() / nref)
            outK[float(p)][int(N)] = float(np.abs(Kof(S) - Kref).max() / nKref)
    N1 = min(Ns)
    v = np.array([out[float(p)][N1] for p in ps])
    vK = np.array([outK[float(p)][N1] for p in ps])
    rel = float((v.max() - v.min()) / max(v.mean(), 1e-300))
    relK = float((vK.max() - vK.min()) / max(vK.mean(), 1e-300))
    return dict(drift_by_p=out, drift_K_by_p=outK, p_list=list(ps),
                drift_at_N1_by_p=[float(x) for x in v],
                drift_K_at_N1_by_p=[float(x) for x in vK],
                relative_spread_across_p=rel,
                relative_spread_across_p_K=relK,
                cancels=bool(max(rel, relK) < 1e-5),
                recorded_drift_K=0.28013,
                note="uniform refinement is the WEAK test: the mass exponent "
                     "cancels exactly, so it cannot see p at all. `drift_K` "
                     "is the drift of K = exp(sT S), which is the quantity "
                     "the screen lane reports as 0.28013 to five figures for "
                     "p = 0.5, 1 and 2 alike")


def selective_refinement(ps=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                         Ns=(1, 2, 4, 8, 16, 32, 64, 128),
                         M: float = 6.0e10 * MSUN, sep: float = 40.0 * KPC,
                         probe_off: float = 8.0 * KPC,
                         well: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """The test WITH TEETH.

    Two equal objects `sep` apart; object 1 is split into N rows, object 2 is
    never split, and the weight ratio is read `probe_off` from object 2 where
    nothing physical has changed.  The relative weight of the refined object
    against the unrefined one moves as N^(1-p).  Only p = 1 has a limit.
    """
    well = dict(well or dict(family="plaw", p=1.0, q=2.0, s=1.0, L=10.0 * KPC))
    x1 = np.zeros(3)
    x2 = np.array([sep, 0.0, 0.0])
    probe = x2 + np.array([probe_off, 0.0, 0.0])
    out = {}
    for p in ps:
        w = dict(well)
        w["p"] = p
        ratios = []
        for N in Ns:
            sub = fib_dirs(N) * (0.5 * KPC) + x1
            wx = np.vstack([sub, x2[None, :]])
            wm = np.concatenate([np.full(N, M / N), [M]])
            d = np.maximum(np.linalg.norm(wx - probe, axis=-1), 0.01 * KPC)
            ww = well_weight(wm, d, w)
            ratios.append(float(ww[:N].sum() / ww[N]))
        sl = float(np.polyfit(np.log(Ns), np.log(ratios), 1)[0])
        out[float(p)] = dict(weight_ratio=[float(v) for v in ratios],
                             slope=sl, predicted=1.0 - p,
                             admissible=bool(abs(sl) < TOL_WEIGHT_SLOPE))
    return dict(N=list(Ns), by_p=out,
                admissible_p=[p for p in ps if out[float(p)]["admissible"]],
                note="d ln(W1/W2)/d ln N must be 0, i.e. p = 1; at any other "
                     "exponent the field near an UNTOUCHED object depends on "
                     "how finely a DIFFERENT object happens to be tabulated, "
                     "and the dependence has no limit")


def representation_convergence(well: Dict[str, Any],
                               Ns=(1, 10, 100, 1000, 10000),
                               Nq: int = 16384, nr: int = 8, ndir: int = 16
                               ) -> Dict[str, Any]:
    """One identical continuous galaxy as 1 catalogue object, 10 subcomponents
    and N stellar-mass cells.  Convergence required below TOL_COARSE."""
    qx, qm = plummer_cloud(Nq=Nq)
    pts = _coherence_probe(nr, ndir)
    parts = nested_partitions(qx, qm, Ns)
    ref = S_wells(pts, qx, qm, well)
    nref = max(np.abs(ref).max(), 1e-300)
    drift = {int(N): float(np.abs(S_wells(pts, *parts[N], well) - ref).max()
                           / nref) for N in Ns}
    have = sorted(drift)
    n_safe = next((N for N in have
                   if all(drift[M_] < TOL_COARSE for M_ in have if M_ >= N)),
                  None)
    # The brief's requirement is convergence across ONE catalogue object, TEN
    # subcomponents and N stellar-mass cells -- so the 1-row and 10-row
    # representations must agree with the continuum, not merely the finest
    # one.  A law that needs 10^4 cells per galaxy has a continuum limit but
    # is wrong at the resolution any catalogue actually has.
    at_cat = max(drift[N] for N in have if N <= 10)
    return dict(N=list(Ns), drift=drift, tol=TOL_COARSE, N_safe=n_safe,
                has_continuum_limit=bool(drift[max(have)] < TOL_COARSE),
                drift_at_catalogue_resolution=float(at_cat),
                converged=bool(at_cat < TOL_COARSE),
                drift_1_row=drift[min(have)], drift_10_rows=drift.get(10),
                note="one FIXED continuous galaxy, re-tabulated; rho never "
                     "changes, only the row list. Convergence is required at "
                     "1 and 10 rows as well as at the finest partition")


def tidal_from_rows_drift(eps_T: float = EPS_T_DEFAULT,
                          Ns=(1, 10, 100, 1000, 10000), Nq: int = 16384,
                          nr: int = 8, ndir: int = 16) -> Dict[str, Any]:
    """Family E's tidal tensor, sourced from the ROW LIST instead of rho.

    Run AB classified E1/E2 as CATALOGUE-ARTEFACTUAL and named the repair:
    "source the tidal tensor from the smooth density rather than the row
    list".  This measures the difference the repair makes.  The reference is
    the Hessian of the full quadrature cloud, i.e. the smooth field.
    """
    qx, qm = plummer_cloud(Nq=Nq)
    pts = _coherence_probe(nr, ndir)
    parts = nested_partitions(qx, qm, Ns)

    def That_of(wx, wm, soft=0.05 * KPC):
        d = np.asarray(wx)[None, :, :] - pts[:, None, :]
        r2 = (d * d).sum(-1) + soft ** 2
        pre = G * np.asarray(wm)[None, :] / r2 ** 1.5
        H = (pre[..., None, None]
             * (np.eye(3) - 3.0 * d[..., :, None] * d[..., None, :]
                / r2[..., None, None])).sum(1)
        T0 = sym_traceless(H)
        n = np.sqrt((T0 * T0).sum((-1, -2)))
        return T0 / np.sqrt(eps_T ** 2 + n ** 2)[..., None, None]

    ref = That_of(qx, qm)
    nref = max(np.abs(ref).max(), 1e-300)
    drift = {int(N): float(np.abs(That_of(*parts[N]) - ref).max() / nref)
             for N in Ns}
    have = sorted(drift)
    at_cat = max(drift[N] for N in have if N <= 10)
    steps = [abs(drift[have[i + 1]] - drift[have[i]])
             for i in range(len(have) - 1)]
    return dict(N=list(Ns), drift=drift, tol=TOL_COARSE,
                drift_at_catalogue_resolution=float(at_cat),
                drift_at_finest=float(drift[max(have)]),
                converged=bool(at_cat < TOL_COARSE
                               and drift[max(have)] < TOL_COARSE),
                shrinking=bool(all(steps[i] >= steps[i + 1]
                                   for i in range(len(steps) - 1))),
                note="the tidal tensor read off the ROW LIST; the reference "
                     "is the same tensor read off the full quadrature cloud, "
                     "i.e. off the smooth density")


def catalogue_perturbations(well: Dict[str, Any], N: int = 1000,
                            Nq: int = 16384, nr: int = 6, ndir: int = 12,
                            seed: int = 11) -> Dict[str, Any]:
    """Merging, detection threshold, deblending, ICL reassignment, mesh.

    Each perturbation leaves the underlying continuous mass distribution
    unchanged (or moves mass only between bookkeeping categories) and reports
    the relative change in the response.
    """
    rng = np.random.default_rng(seed)
    qx, qm = plummer_cloud(Nq=Nq)
    pts = _coherence_probe(nr, ndir)
    wx, wm = nested_partitions(qx, qm, [N])[N]
    base = S_wells(pts, wx, wm, well)
    nb = max(np.abs(base).max(), 1e-300)

    def rel(S):
        return float(np.abs(S - base).max() / nb)

    out = {}

    # (i) merge the closest 20% of rows into single rows at their centroid
    d = np.sqrt(((wx[:, None, :] - wx[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d.min(1))
    used, mx, mm = set(), [], []
    for i in order:
        if int(i) in used or len(used) >= 0.4 * len(wx):
            continue
        j = int(np.argmin(d[i]))
        if j in used:
            continue
        used.update({int(i), j})
        M2 = wm[i] + wm[j]
        mx.append((wx[i] * wm[i] + wx[j] * wm[j]) / M2)
        mm.append(M2)
    keep = [i for i in range(len(wx)) if i not in used]
    wx_m = np.vstack([wx[keep]] + ([np.array(mx)] if mx else []))
    wm_m = np.concatenate([wm[keep]] + ([np.array(mm)] if mm else []))
    out["merge_neighbours"] = rel(S_wells(pts, wx_m, wm_m, well))

    # (ii) detection threshold: drop the faintest 30% of rows, two ways
    sel = wm > np.quantile(wm, 0.30)
    out["detect_threshold_discard"] = rel(S_wells(pts, wx[sel], wm[sel], well))
    out["detect_threshold_redistribute"] = rel(S_wells(
        pts, wx[sel], wm[sel] * (wm.sum() / wm[sel].sum()), well))

    # (iii) deblending: split the heaviest 10% of rows into two
    hi = np.argsort(wm)[-max(1, int(0.1 * len(wm))):]
    off = rng.normal(size=(len(hi), 3))
    off = 0.3 * KPC * off / np.linalg.norm(off, axis=1, keepdims=True)
    wx_d = np.vstack([wx, wx[hi] + off])
    wx_d[hi] = wx[hi] - off
    wm_d = np.concatenate([wm, wm[hi] * 0.5])
    wm_d[hi] = wm[hi] * 0.5
    out["deblend_split"] = rel(S_wells(pts, wx_d, wm_d, well))

    # (iv) intracluster light: 20% of every row's mass leaves the CATALOGUE
    #      for a smooth component.  rho is unchanged; only the row list is.
    out["ICL_reassign_20pc"] = rel(S_wells(pts, wx, 0.8 * wm, well))

    # (v) mesh resolution: the same law on a 4x coarser quadrature cloud
    qx2, qm2 = plummer_cloud(Nq=Nq // 4)
    wx2, wm2 = nested_partitions(qx2, qm2, [N])[N]
    out["mesh_resolution_4x"] = rel(S_wells(pts, wx2, wm2, well))

    worst = max(out.values())
    return dict(perturbations=out, worst=float(worst), tol=TOL_COARSE,
                passes=bool(worst < TOL_COARSE), N=N,
                worst_name=max(out, key=out.get))


def pair_tensor_collapse(pair: Dict[str, Any],
                         Ns=(1, 10, 25, 50, 100, 200, 400, 800),
                         alpha: float = 0.3, Nq: int = 32768,
                         nr: int = 6, ndir: int = 12) -> Dict[str, Any]:
    """Family D: does the pair tensor have a continuum limit?

    ||C|| ~ N^(2-2p), so K = exp[-alpha C] collapses exponentially with
    catalogue resolution unless p = 1.  The recorded result is a local log
    slope of 0.0102 / 1.0101 / 0.1667 for (p,q) = (1,1) / (0.5,1) / (1,3)
    against a prediction of 0 / 1 / log-divergent, and lambda_min(K) falling
    from 3.4e-1 to 8.3e-80 as N goes 10 -> 800 at p = 1/2.

    The reported slope is the LOCAL log slope at the largest N, which is what
    "does this converge" means; a slope fitted across the whole range is
    contaminated by the small-N transient.
    """
    qx, qm = plummer_cloud(Nq=Nq)
    parts = nested_partitions(qx, qm, Ns)
    probe = _coherence_probe(nr, ndir)
    norms, lmin = [], []
    for N in Ns:
        C = C_pairs(probe, *parts[N], pair)
        norms.append(float(np.abs(np.linalg.eigvalsh(C)).max()))
        ev = np.linalg.eigvalsh(-alpha * C)
        lmin.append(float(np.exp(np.clip(ev.min(), -745.0, 700.0))))
    live = [(n, v) for n, v in zip(Ns, norms) if v > 0]
    if len(live) >= 3:
        xs = np.log([n for n, _ in live[-3:]])
        ys = np.log([v for _, v in live[-3:]])
        sl = float(np.polyfit(xs, ys, 1)[0])
    else:
        sl = float("nan")
    # representation drift: 1 catalogue row against the finest partition
    ref = norms[-1]
    drift1 = float(abs(norms[0] - ref) / max(ref, 1e-300))
    i10 = Ns.index(10) if 10 in Ns else 1
    drift10 = float(abs(norms[i10] - ref) / max(ref, 1e-300))
    p = pair["p"]
    return dict(N=list(Ns), norm_C=norms, lambda_min_K=lmin,
                slope_lnC_lnN=sl, predicted_slope=2.0 - 2.0 * p,
                lambda_min_first=lmin[1] if len(lmin) > 1 else lmin[0],
                lambda_min_last=lmin[-1],
                lambda_min_decades=float(
                    math.log10(max(lmin[1] if len(lmin) > 1 else lmin[0],
                                   1e-320))
                    - math.log10(max(lmin[-1], 1e-320))),
                drift_1_row=drift1, drift_10_rows=drift10,
                converged=bool(abs(sl) < 0.2 and max(drift1, drift10)
                               < TOL_COARSE),
                has_continuum_limit=bool(abs(sl) < 0.2),
                note="||C|| ~ N^(2-2p); K = exp[-alpha C], so a growing ||C|| "
                     "is an exponential COLLAPSE of the response tensor with "
                     "catalogue resolution -- not slow convergence, no limit "
                     "at all. A single catalogue row has NO pairs, so family "
                     "D has exactly zero effect on an unsubdivided object")


def cluster_M_dyn_representation(sT: float = 0.5, well: Optional[Dict] = None,
                                 M: float = 1.0e14 * MSUN,
                                 a: float = 400.0 * KPC,
                                 r_probe: float = 1000.0 * KPC,
                                 Ns=(1, 10000), Nq: int = 32768,
                                 ndir: int = 24) -> Dict[str, Any]:
    """Family C's inferred dynamical mass for ONE cluster, as a function of
    how many catalogue rows it is written down as.  Recorded: 14%."""
    qx, qm = plummer_cloud(Nq=Nq, M=M, a=a)
    parts = nested_partitions(qx, qm, Ns)
    dirs = fib_dirs(ndir)
    probe = dirs * r_probe
    well = dict(well or dict(family="plaw", p=1.0, q=2.0, s=1.0, L=10.0 * KPC))
    out = {}
    for N in Ns:
        S = S_wells(probe, *parts[N], well)
        lam = np.einsum("pi,pij,pj->p", dirs, S, dirs)
        out[int(N)] = float(np.median(1.0 / np.exp(sT * lam)))
    ks = sorted(out)
    Mnewt = float(M * r_probe ** 3 / (r_probe ** 2 + a ** 2) ** 1.5)
    return dict(M_dyn_over_M_newt_by_N=out,
                M_newt_enclosed_Msun=Mnewt / MSUN,
                M_dyn_by_N_Msun={k: out[k] * Mnewt / MSUN for k in ks},
                fractional_change=float(out[ks[-1]] / out[ks[0]] - 1.0),
                recorded_fractional_change=0.142,
                note="the SAME cluster, the SAME rho, differing only in how "
                     "many catalogue rows it is entered as")


_G3_CACHE: Dict[Tuple, Dict[str, Any]] = {}


def gate3_key(cand: Candidate, cheap: bool) -> Tuple:
    """GATE 3 depends ONLY on how the response reads the mass distribution.

    Not on the base law, not on the invariant, not on the response form, not
    on the amplitude.  Keying the measurement on that alone is what makes the
    gate a family eliminator: 1,560 tensor_S settings in the tournament reduce
    to four weight families.
    """
    w = cand.well or {}
    p = cand.pair or {}
    return ("g3", cand.struct, cand.field_source,
            w.get("family"), w.get("p"), w.get("q"), w.get("s"), w.get("L"),
            w.get("exclude_nearest"),
            p.get("p"), p.get("q"), p.get("s"), p.get("L"),
            (cand.tidal_const or {}).get("eps_T"), cheap)


def gate4_key(cand: Candidate) -> Tuple:
    """GATE 4 depends on the response's FUNCTIONAL FORM, not on its
    amplitude: the Jacobian's asymmetry is zero or non-zero according to which
    functional of rho the response reads.  The amplitude is kept in the key
    anyway, because the reported VALUE depends on it and the compiler must not
    quote a number it did not measure for this setting."""
    return ("g4", cand.base, cand.struct, cand.inv, cand.form, float(cand.m),
            float(cand.I0), float(cand.A), cand.field_source,
            bool(cand.momentum_carrier), cand.pair_kernel is not None,
            (cand.tidal_const or {}).get("eps_T"),
            None if cand.pair is None else tuple(sorted(
                (k, v) for k, v in cand.pair.items())))


def gate3(cand: Candidate, cheap: bool = True
          ) -> Tuple[bool, Dict[str, Any], str]:
    """GATE 3 -- coarse graining."""
    if not cand.responds():
        return (True, dict(catalogue_dependent=False),
                "base law: the response is identically zero, so no row list "
                "can enter it")

    inv_src = INVARIANT_SOURCE.get(cand.inv, "smooth")
    rows = cand.struct in CATALOGUE_STRUCTS or cand.field_source == "rows"
    if not rows:
        return (True, dict(catalogue_dependent=False, invariant_source=inv_src,
                           structure=cand.struct, field_source="smooth"),
                f"the response is a functional of the Poisson-smooth fields "
                f"only (invariant '{cand.inv}' = {inv_src}, structure "
                f"'{cand.struct}'): the well list never enters, so the drift "
                f"under re-tabulation is exactly zero by CONSTRUCTION, not "
                f"numerically small")

    key = gate3_key(cand, cheap)
    if key in _G3_CACHE:
        res = _G3_CACHE[key]
    else:
        res: Dict[str, Any] = dict(structure=cand.struct,
                                   catalogue_dependent=True,
                                   field_source=cand.field_source)
        if cand.struct == "pairs":
            pr = cand.pair or dict(p=1.0, q=1.0, s=2.0, L=10.0 * KPC)
            res["p"] = float(pr["p"])
            res["pair_collapse"] = pair_tensor_collapse(pr)
        elif cand.struct in ("tensor_S", "wells"):
            well = cand.well or dict(family="plaw", p=1.0, q=2.0, s=1.0,
                                     L=10.0 * KPC)
            res["p"] = float(well["p"])
            res["well"] = {k: (v / KPC if k == "L" else v)
                           for k, v in well.items()}
            res["selective_refinement"] = selective_refinement(
                ps=(res["p"],), well=well)
            res["representation"] = representation_convergence(
                well, Ns=(1, 10, 100, 1000) if cheap else
                (1, 10, 100, 1000, 10000),
                Nq=4096 if cheap else 16384,
                nr=5 if cheap else 8, ndir=10 if cheap else 16)
            if not cheap:
                res["perturbations"] = catalogue_perturbations(well)
                res["uniform_refinement"] = uniform_refinement(well)
        else:                      # a smooth structure read off the row list
            res["representation"] = tidal_from_rows_drift(
                (cand.tidal_const or {}).get("eps_T", EPS_T_DEFAULT),
                Ns=(1, 10, 100, 1000) if cheap else
                (1, 10, 100, 1000, 10000),
                Nq=4096 if cheap else 16384,
                nr=5 if cheap else 8, ndir=10 if cheap else 16)
        _G3_CACHE[key] = res

    if cand.struct == "pairs":
        pc = res["pair_collapse"]
        ok = bool(pc["converged"])
        why = (f"family-D pair tensor: ||C|| ~ N^(2-2p) with p = {res['p']:g}, "
               f"measured local log slope {pc['slope_lnC_lnN']:.4f} against "
               f"the predicted {pc['predicted_slope']:.4f}; lambda_min(K) runs "
               f"{pc['lambda_min_first']:.2e} -> {pc['lambda_min_last']:.2e} "
               f"over N = 10 -> {pc['N'][-1]}, "
               f"{pc['lambda_min_decades']:.1f} decades. Representation drift "
               f"{pc['drift_1_row']:.3f} at 1 row and {pc['drift_10_rows']:.3f} "
               f"at 10 rows against a tolerance of {TOL_COARSE:g}: a single "
               f"catalogue row has NO PAIRS, so the law has exactly zero "
               f"effect on an unsubdivided object. "
               + ("A continuum limit exists."
                  if pc["has_continuum_limit"] else
                  "There is NO continuum limit: the response tensor collapses "
                  "exponentially as the catalogue is refined."))
        return ok, res, why

    rep = res["representation"]
    ok = bool(rep["converged"])
    if cand.struct in ("tensor_S", "wells"):
        p = res["p"]
        sel = res["selective_refinement"]["by_p"][float(p)]
        ok = ok and bool(sel["admissible"])
        pmsg = (f"Mass exponent p = {p:g}: under SELECTIVE refinement the "
                f"weight of a refined object against an untouched one moves "
                f"as N^(1-p), measured slope {sel['slope']:.4f} against the "
                f"predicted {sel['predicted']:.4f}"
                + ("; p = 1 is the one admissible exponent and this candidate "
                   "has it. " if sel["admissible"] else
                   f". Only p = 1 is admissible; p = {p:g} makes the field "
                   f"near an UNTOUCHED object depend on how finely a "
                   f"DIFFERENT object happens to be tabulated, with no "
                   f"limit. "))
    else:
        pmsg = ""
    if not cheap and "perturbations" in res:
        pert = res["perturbations"]
        ok = ok and bool(pert["passes"])
        pmsg += (f"Worst catalogue perturbation {pert['worst']:.3e} "
                 f"({pert['worst_name']}) against tolerance "
                 f"{TOL_COARSE:g}. ")
    why = (f"the response is built from a ROW LIST, so a cataloguer's choices "
           f"enter the field equation. {pmsg}"
           f"Re-tabulating ONE fixed continuous galaxy as 1 row, 10 "
           f"subcomponents and up to {max(rep['N'])} mass cells moves the "
           f"response by {rep['drift_at_catalogue_resolution']:.4f} at "
           f"catalogue resolution (1-10 rows) against a tolerance of "
           f"{TOL_COARSE:g}"
           + (f" and {rep['drift'][max(rep['drift'])]:.2e} at the finest "
              f"partition." if "drift" in rep else ".")
           + (" A real coherence scale must be universal and appear in the "
              "field equation; it cannot be set by the cataloguer."
              if not ok else " Convergent at every representation tested."))
    return ok, res, why


# ==================================================== 7. GATE 4 machinery
def _radial_background(n: int = 1400, rmin=0.05 * KPC, rmax=3000.0 * KPC,
                       M: float = 1.0e12 * MSUN, a: float = 20.0 * KPC):
    r = np.geomspace(rmin, rmax, n)
    comp = Plummer(M, a)
    return r, comp


def _invariants_radial(r, comp):
    """The five invariants on the radial grid, all in closed form."""
    F = G * comp.M_enc(r) / r ** 2
    phi = -G * comp.M / np.sqrt(r ** 2 + comp.a ** 2)
    rho = comp.rho(np.stack([r, 0 * r, 0 * r], -1))
    # For a spherical potential |T| = sqrt(2/3)|Phi'' - Phi'/r|
    #                               = sqrt(2/3)|4 pi G rho - 3 F/r|.
    tid = np.sqrt(2.0 / 3.0) * np.abs(4 * np.pi * G * rho - 3.0 * F / r)
    qb = np.full_like(r, comp.M / (comp.M + M_NL))
    return dict(one=np.ones_like(r), gn=F / A0, phi=np.abs(phi),
                rhobar=np.maximum(rho, 1e-40),
                tidal=np.maximum(tid, 1e-45), qbar=qb), F, rho


#: The screen lane's own ceiling on the condition number of K.  Conjugate
#: gradients on a system of condition number kappa cannot reach a relative
#: residual below about kappa * 2e-16 in float64, so kappa > 1e8 forecloses a
#: 1e-11 solve before a single iteration runs.  The tournament records the
#: same outcome for its own extreme amplitudes: "K's condition number exceeds
#: 1e8 on this configuration, so there is no bounded solution to compare
#: forces with."
COND_MAX = 1.0e8


def response_health(cand: Candidate) -> Dict[str, Any]:
    """Is the candidate's field equation numerically solvable at all?

    A fitted amplitude of 300 on a response of order unity drives
    k_r = exp(A W lambda) to 1e80.  That is not a small number in an
    otherwise fine law -- it is a statement that the field equation has no
    bounded solution, and a compiler that silently returned "symmetric
    Jacobian" because the derivative underflowed would be admitting a
    candidate on a numerical accident.
    """
    P = probes()
    conds, finite = [], True
    for nm in GATE1_PROBES:
        p = P[nm]
        lam = probe_lambda(cand, nm)
        W = response_W(cand, p.inv)
        if cand.struct in ("scalar_a0", "depth"):
            k = np.abs(1.0 + cand.A * W)
        else:
            k = radial_eigen(cand.struct, lam, cand.A, W)
        k = np.asarray(k, float)
        conds.append(float(k.max() / max(k.min(), 1e-300)))
        g = predict_g(cand, p.inv, p.gN, lam)
        finite = finite and bool(np.all(np.isfinite(g)) and np.all(g > 0))
    cond = max(conds)
    return dict(k_condition=cond, finite_positive_g=finite,
                cond_max=COND_MAX,
                degenerate=bool(cond > COND_MAX or not finite))


def jacobian_asymmetry(cand: Candidate,
                       rs_kpc=(2.0, 5.0, 12.0, 30.0, 70.0, 160.0, 380.0,
                               900.0)) -> Dict[str, Any]:
    """Is delta Phi(x) / delta rho(y) symmetric?

    A law that comes from an action has a symmetric functional Jacobian: the
    second variation of an action is a Hessian and a Hessian is symmetric.  An
    ASYMMETRIC Jacobian therefore proves that no action produces the law as
    written, so the candidate is nonconservative.  Symmetry does NOT prove a
    relativistic completion exists -- it only fails to reject.

    The test uses spherically symmetric perturbations.  That is a genuine
    one-sided argument: asymmetry on a subspace implies asymmetry.  It is
    computed SEMI-ANALYTICALLY, which matters -- a brute-force
    finite-difference Jacobian on a discretised radial grid is asymmetric at
    O(dr) even for Newton, and that grid artefact would swamp a weak gate.

        Phi(r_i) - Phi(R) = -Int_{r_i}^{R} g(s) ds,   g = Gamma(F(s), I(s))
        dF(s)/dm_j = G H(s - r_j)/s^2                       -> symmetric
        dI(s)/dm_j = invariant-specific, closed form        -> the test

    The F channel contributes -Int_{max(r_i,r_j)}^{R} Gamma_F G/s^2 ds, which
    depends on (i,j) only through max(r_i, r_j) and is EXACTLY symmetric.
    Every asymmetry reported therefore comes from the response's dependence on
    a field that is not |g_N| -- which is exactly Run AH's finding that the
    third-law violation is a property of "the response depends on position",
    not of anisotropy.
    """
    r, comp = _radial_background()
    inv, F, rho = _invariants_radial(r, comp)
    eps_T = (cand.tidal_const or {}).get("eps_T", EPS_T_DEFAULT)
    fam_E = cand.struct == "tidal_const"
    A_use = float(cand.A)

    # A clipped exponential has ZERO derivative, so a response that saturates
    # the float64 exponent range or the declared W ceiling anywhere on the
    # reference background would make the Jacobian come out symmetric by
    # UNDERFLOW rather than by symmetry.  The measurement is therefore made on
    # the largest radial window in which the response is unclipped, and the
    # window is reported.  Truncating the window is legitimate because the
    # quantity measured is Phi(r_i) - Phi(R): R is the window's outer edge.
    lam_c = 1.0 if cand.struct == "iso_K" else (2.0 / 3.0)
    if cand.responds():
        Wg = (np.ones_like(r) if cand.struct in CONSTANT_COUPLING_STRUCTS
              else (np.zeros_like(r) if (cand.form == "off"
                                         or cand.inv == "one")
                    else W_of(cand.form,
                              np.asarray(inv["tidal" if fam_E else cand.inv],
                                         float) / cand.I0, cand.m)))
        if cand.struct in ("scalar_a0", "depth"):
            lnk = np.log(np.maximum(np.abs(1.0 + A_use * Wg), 1e-300))
        else:
            lnk = A_use * Wg * lam_c
        lnk = np.clip(np.nan_to_num(lnk, nan=0.0, posinf=700.0,
                                    neginf=-700.0), -700.0, 700.0)
        # the window in which the field equation is SOLVABLE: k_r may not span
        # more than COND_MAX, the screen lane's own ceiling on the condition
        # number of K, anchored at the median radius.
        anchor = len(r) // 2
        tolln = math.log(COND_MAX)
        lo = anchor
        while lo > 0 and abs(lnk[lo - 1] - lnk[anchor]) < tolln:
            lo -= 1
        hi = anchor
        while hi < len(r) - 1 and abs(lnk[hi + 1] - lnk[anchor]) < tolln:
            hi += 1
        ok_ceiling = Wg < 0.999e6
        while hi > lo and not ok_ceiling[hi]:
            hi -= 1
        while lo < hi and not ok_ceiling[lo]:
            lo += 1
    else:
        lo, hi = 0, len(r) - 1
    if hi - lo < 20:
        return dict(asymmetry=float("inf"), worst_pair=float("inf"), n=0,
                    kind="degenerate", window_kpc=[float("nan")] * 2,
                    note="the response spans more than the solvable condition "
                         "number over essentially the whole reference "
                         "background: there is no window in which the "
                         "Jacobian can be measured")
    r = r[lo:hi + 1]
    inv = {k: v[lo:hi + 1] for k, v in inv.items()}
    F, rho = F[lo:hi + 1], rho[lo:hi + 1]

    rj = np.array(rs_kpc) * KPC
    rj = rj[(rj > r[0] * 1.2) & (rj < r[-1] * 0.8)]
    if len(rj) < 3:
        return dict(asymmetry=float("inf"), worst_pair=float("inf"),
                    n=len(rj), kind="degenerate",
                    window_kpc=[float(r[0] / KPC), float(r[-1] / KPC)],
                    note="fewer than three probe shells fall inside the "
                         "window in which the response is unclipped")
    nj = len(rj)
    idx = np.searchsorted(r, rj)
    lam0 = np.full_like(r, lam_c)

    def g_of(Fv, Iv):
        lam = lam0
        if fam_E:
            # family E's response is the REGULARISED tidal direction, so it is
            # a smooth function of |T| and therefore of the second derivative
            # of Phi_N.  Its coupling channel is the tidal one.
            W = np.ones_like(Fv)
            lam = math.sqrt(2.0 / 3.0) * Iv / np.sqrt(eps_T ** 2 + Iv ** 2)
        elif cand.struct in CONSTANT_COUPLING_STRUCTS:
            W = np.ones_like(Fv)
        elif cand.form == "off" or cand.inv == "one":
            W = np.zeros_like(Fv)
        else:
            W = W_of(cand.form, Iv / cand.I0, cand.m)
        if cand.struct in ("scalar_a0", "depth"):
            a_eff = np.maximum(cand.a0 * (1.0 + A_use * W), 1e-14 * cand.a0)
            return g_of_gN(cand.base, Fv, a_eff)
        return mond_invert(Fv, radial_eigen(cand.struct, lam, A_use, W),
                           cand.a0, cand.base)

    Iv = np.asarray(inv["tidal" if fam_E else cand.inv], float)
    h = 1e-6
    GF = (g_of(F * (1 + h), Iv) - g_of(F * (1 - h), Iv)) / (2 * h * F)
    with np.errstate(invalid="ignore", divide="ignore"):
        GI = np.nan_to_num(
            (g_of(F, Iv * (1 + h)) - g_of(F, Iv * (1 - h)))
            / (2 * h * np.maximum(Iv, 1e-300)))

    dr = np.diff(r)

    def tail(f):
        """Int_{r_k}^{R} f ds for every k."""
        seg = 0.5 * (f[1:] + f[:-1]) * dr
        out = np.zeros_like(f)
        out[:-1] = np.cumsum(seg[::-1])[::-1]
        return out

    TF = tail(GF * G / r ** 2)

    if fam_E:
        kind = "tidal"
    elif cand.struct in ("wells", "pairs"):
        # the response is a functional of the ROW LIST, not of rho at all, so
        # delta/delta rho is not defined; the structural half of GATE 4
        # carries these and the Jacobian half reports the symmetric channel.
        kind = "zero"
    elif cand.form == "off" or cand.inv == "one":
        kind = "zero"
    else:
        kind = {"gn": "maxform", "phi": "phi", "tidal": "tidal",
                "rhobar": "rho", "qbar": "qbar", "one": "zero"}.get(
                    cand.inv, "zero")

    Tmax = tail(GI * G / (r ** 2 * A0))
    Trho3 = None
    if kind == "tidal":
        sg = np.sign(4 * np.pi * G * rho - 3.0 * F / r)
        Trho3 = tail(GI * sg * G / r ** 3)

    J = np.zeros((nj, nj))
    for i in range(nj):
        for j in range(nj):
            k = max(idx[i], idx[j])
            val = -TF[k]
            if kind == "maxform":
                val += -Tmax[k]
            elif kind == "phi":
                val += -tail(GI * (G / np.maximum(r, rj[j])))[idx[i]]
            elif kind == "rho":
                if idx[j] >= idx[i]:
                    val += -GI[idx[j]] / (4 * np.pi * rj[j] ** 2)
            elif kind == "tidal":
                c = math.sqrt(2.0 / 3.0)
                sgn = np.sign(4 * np.pi * G * rho[idx[j]]
                              - 3.0 * F[idx[j]] / rj[j])
                if idx[j] >= idx[i]:
                    val += -c * sgn * GI[idx[j]] * G / rj[j] ** 2
                val += 3.0 * c * Trho3[k]
            elif kind == "qbar":
                s = r
                f_ov = np.clip((L_NL ** 2 - (s - rj[j]) ** 2)
                               / (4.0 * s * rj[j]), 0.0, 1.0)
                f_ov = np.where(np.abs(s - rj[j]) >= L_NL, 0.0, f_ov)
                f_ov = np.where(s + rj[j] <= L_NL, 1.0, f_ov)
                val += -tail(GI * f_ov * M_NL / (comp.M + M_NL) ** 2)[idx[i]]
            J[i, j] = val

    nrm = np.abs(J).max()
    if nrm == 0:
        return dict(asymmetry=0.0, worst_pair=0.0, n=nj, kind=kind,
                    window_kpc=[float(r[0] / KPC), float(r[-1] / KPC)],
                    note="Jacobian identically zero")
    if not np.isfinite(nrm):
        return dict(asymmetry=float("inf"), worst_pair=float("inf"), n=nj,
                    kind="degenerate",
                    window_kpc=[float(r[0] / KPC), float(r[-1] / KPC)],
                    note="the Jacobian overflows float64 on the reference "
                         "background: the response is not a bounded "
                         "perturbation of the base law and the action test "
                         "cannot be run")
    return dict(asymmetry=float(np.linalg.norm(J - J.T)
                                / max(np.linalg.norm(J), 1e-300)),
                worst_pair=float(np.abs(J - J.T).max() / nrm),
                n=nj, kind=kind,
                window_kpc=[float(r[0] / KPC), float(r[-1] / KPC)],
                r_shell_kpc=[float(v / KPC) for v in rj],
                note="delta Phi(r_i)/delta m_j, semi-analytic; the |g_N| "
                     "channel is exactly symmetric by construction, so every "
                     "reported asymmetry is the response's")


def jacobian_asymmetry_fd(cand: Candidate,
                          rs_kpc=(3.0, 8.0, 20.0, 50.0, 120.0, 300.0),
                          width: float = 0.06, frac: float = 1e-3
                          ) -> Dict[str, Any]:
    """Independent brute-force finite-difference Jacobian, for validation.

    Perturbs the background by a smooth Gaussian-in-ln-r shell, re-solves the
    whole radial law and re-integrates the potential.  Shares no code path
    with the semi-analytic route, so agreement between the two is a real
    cross-check rather than an algebraic identity.
    """
    r, comp = _radial_background(n=3000)
    lnr = np.log(r)
    rj = np.array(rs_kpc) * KPC
    lam = np.ones_like(r) if cand.struct == "iso_K" else np.full_like(r, 2 / 3)

    def shell(j):
        u = (lnr - math.log(rj[j])) / width
        w = np.exp(-0.5 * u * u)
        return w / np.trapezoid(w, lnr)          # dM/dlnr, unit total mass

    def cumint(f):
        return np.concatenate([[0.0], np.cumsum(
            0.5 * (f[1:] + f[:-1]) * np.diff(lnr))])

    def phi_of(dm_vec):
        Menc = comp.M_enc(r).copy()
        Mtot = comp.M
        phiN = -G * comp.M / np.sqrt(r ** 2 + comp.a ** 2)
        for j, dm in enumerate(dm_vec):
            if dm == 0.0:
                continue
            pj = shell(j) * dm
            cum = cumint(pj)
            Menc = Menc + cum
            Mtot = Mtot + dm
            # Phi of a shell distribution: -G[M(<r)/r + Int_r^inf dm/t]
            outer = np.trapezoid(pj / r, lnr) - cumint(pj / r)
            phiN = phiN - G * (cum / r + outer)
        F = G * Menc / r ** 2
        rho = (np.gradient(F, r) + 2 * F / r) / (4 * np.pi * G)
        tid = np.sqrt(2.0 / 3.0) * np.abs(4 * np.pi * G * rho - 3.0 * F / r)
        inv = dict(one=np.ones_like(r), gn=F / A0, phi=np.abs(phiN),
                   rhobar=np.maximum(rho, 1e-40),
                   tidal=np.maximum(tid, 1e-45),
                   qbar=np.full_like(r, Mtot / (Mtot + M_NL)))
        g = predict_g(cand, inv, F, lam)
        seg = 0.5 * (g[1:] + g[:-1]) * np.diff(r)
        out = np.zeros_like(r)
        out[:-1] = -np.cumsum(seg[::-1])[::-1]
        return out

    dm = frac * comp.M
    cols = []
    for j in range(len(rj)):
        v = np.zeros(len(rj))
        v[j] = dm
        pp = phi_of(v)
        v[j] = -dm
        cols.append((pp - phi_of(v)) / (2 * dm))
    J = np.stack(cols, 1)[np.searchsorted(r, rj), :]
    return dict(asymmetry=float(np.linalg.norm(J - J.T)
                                / max(np.linalg.norm(J), 1e-300)),
                worst_pair=float(np.abs(J - J.T).max()
                                 / max(np.abs(J).max(), 1e-300)),
                n=len(rj), method="finite difference")


_FD_FLOOR: List[float] = []


def fd_floor() -> float:
    """The finite-difference Jacobian's own resolution.

    Measured on the Newtonian control, which is EXACTLY symmetric in the
    continuum, so whatever the FD route reports for it is the instrument's
    floor.  A smooth Gaussian-in-ln-r perturbing shell of finite width breaks
    the exact max(r_i, r_j) structure at O(width^2), and that -- not the law
    -- is what the floor measures.  Quoting an FD asymmetry without it would
    be quoting a discretisation artefact as physics.
    """
    if not _FD_FLOOR:
        _FD_FLOOR.append(max(
            jacobian_asymmetry_fd(Candidate("newton_floor", base="newton",
                                            struct="none", inv="one",
                                            form="off", A=0.0))["asymmetry"],
            jacobian_asymmetry_fd(Candidate("aqual_floor", base="aqual",
                                            struct="none", inv="one",
                                            form="off", A=0.0))["asymmetry"]))
    return _FD_FLOOR[0]


def kernel_reciprocity(cand: Candidate, n: int = 200,
                       seed: int = 5) -> Dict[str, Any]:
    """F(x, x') == F(x', x) for an explicitly declared pair kernel."""
    if cand.pair_kernel is None and cand.pair is None:
        return dict(applicable=False,
                    note="no explicit pair kernel is declared")
    rng = np.random.default_rng(seed)
    xa = rng.normal(size=(n, 3)) * 20 * KPC
    xb = rng.normal(size=(n, 3)) * 20 * KPC
    if cand.pair_kernel is not None:
        f1 = np.array([cand.pair_kernel(xa[i], xb[i]) for i in range(n)])
        f2 = np.array([cand.pair_kernel(xb[i], xa[i]) for i in range(n)])
    else:
        pr = cand.pair
        M0 = pr.get("M0", 1e10 * MSUN)
        ma = np.full(n, 1e10 * MSUN)
        mb = np.full(n, 3e10 * MSUN)
        d = np.maximum(np.linalg.norm(xa - xb, axis=-1), 0.02 * KPC)
        x = d / pr["L"]

        def wf(m1, m2):
            return (((m1 * m2) / M0 ** 2) ** pr["p"]) * x ** (-pr["q"]) \
                * np.exp(-(x ** pr["s"]))
        f1, f2 = wf(ma, mb), wf(mb, ma)
    rel = float(np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f1)), 1e-300))
    return dict(applicable=True, max_relative_asymmetry=rel,
                reciprocal=bool(rel < 1e-12),
                note="reciprocity of the kernel is NOT the third law: a "
                     "symmetric interaction still leaks momentum when the "
                     "coupling depends on a field that differs across the "
                     "configuration -- Run Y measured reciprocity at 4.1e-16 "
                     "with an 11% momentum leak")


#: Structures whose response is a functional of the ROW LIST.  A row-list
#: functional cannot be written as delta/delta rho at all -- the map from rho
#: to the response is not even defined without the cataloguer's partition -- so
#: no action over rho can produce it, independently of the Jacobian test.
ROWLIST_RESPONSE = {"tensor_S", "wells", "pairs"}


def gate4(cand: Candidate) -> Tuple[bool, Dict[str, Any], str]:
    """GATE 4 -- reciprocity and action."""
    rec = kernel_reciprocity(cand)
    jac = jacobian_asymmetry(cand)
    asym = float(jac["asymmetry"])
    if not np.isfinite(asym):
        asym = float("inf")
    rowlist = cand.struct in ROWLIST_RESPONSE and cand.responds()
    health = response_health(cand) if cand.responds() else dict(
        k_condition=1.0, finite_positive_g=True, cond_max=COND_MAX,
        degenerate=False)
    src = response_field(cand)

    val = dict(jacobian=jac, reciprocity=rec, asymmetry=asym, tol=TOL_ASYM,
               momentum_carrier=cand.momentum_carrier,
               rowlist_response=rowlist, health=health, response_field=src,
               variational=bool(src["admissible"] and asym <= TOL_ASYM
                                and not health["degenerate"]))

    if health["degenerate"]:
        return (False, val,
                f"the field equation has NO BOUNDED SOLUTION at this setting: "
                f"the response drives the condition number of K to "
                f"{health['k_condition']:.3e} against a ceiling of "
                f"{COND_MAX:.0e}"
                + ("" if health["finite_positive_g"] else
                   ", and the predicted acceleration is not finite and "
                   "positive everywhere on the probes")
                + ". An action cannot be tested for a law with no solution, "
                  "and a Jacobian that comes out symmetric here would be a "
                  "numerical underflow, not a symmetry. The tournament "
                  "records the same outcome for its own extreme amplitudes.")
    if rec["applicable"] and not rec["reciprocal"] and not cand.momentum_carrier:
        return (False, val,
                f"pair kernel is NOT reciprocal "
                f"({rec['max_relative_asymmetry']:.2e} relative) and no "
                f"momentum carrier is declared")
    if rowlist and not cand.momentum_carrier:
        return (False, val,
                "the response is a functional of a CATALOGUE ROW LIST, not of "
                "rho, so delta g(x)/delta rho(y) is not even defined without "
                "the cataloguer's partition: no action over rho can produce "
                "this law. No momentum carrier is declared. (The pair kernel "
                "itself is reciprocal where one exists -- reciprocity is not "
                "the third law.)")
    if cand.momentum_carrier and not src["admissible"]:
        return (True, val,
                f"the response reads '{src['tensor']}' / '{src['invariant']}', "
                f"neither of which is a function of grad Phi_N, so the law as "
                f"written is not variational (measured Jacobian asymmetry "
                f"{asym:.3e}); but a momentum carrier is declared "
                f"('{cand.momentum_carrier}'), so the missing momentum has "
                f"somewhere to go. FLAGGED: the carrier must still be shown to "
                f"close the budget.")
    if not src["admissible"]:
        floor = (" The Jacobian instrument returned a value at or below its "
                 "own round-off floor for this setting, so the verdict rests "
                 "on the structural argument rather than on the measurement; "
                 "the instrument has no power below the floor and that is "
                 "reported rather than hidden."
                 if asym <= TOL_ASYM else
                 f" The functional Jacobian delta Phi(r_i)/delta m_j is "
                 f"measured ASYMMETRIC at {asym:.3e} (worst pair "
                 f"{jac['worst_pair']:.3e}), confirming it.")
        return (False, val,
                f"the response reads '{src['tensor']}' (tensor source) and "
                f"'{src['invariant']}' (invariant), and neither is a function "
                f"of grad Phi_N alone. In the QUMOND form the law comes from "
                f"an action with Phi_N still solving Poisson only if K(u)u is "
                f"a gradient in u = grad Phi_N; it is not here, so no action "
                f"produces this law as written and momentum is not conserved. "
                f"No momentum carrier is declared." + floor)
    if asym <= TOL_ASYM:
        return (True, val,
                f"the response reads only '{src['tensor']}' / "
                f"'{src['invariant']}', so K(u)u is a gradient in "
                f"u = grad Phi_N and the law is AQUAL/QUMOND with a redefined "
                f"interpolating function; the functional Jacobian "
                f"delta Phi(x)/delta rho(y) is symmetric to {asym:.2e} "
                f"(round-off floor {TOL_ASYM:.0e}), confirming it. This does "
                f"NOT prove a relativistic completion exists.")
    if cand.momentum_carrier:
        return (True, val,
                f"functional Jacobian is asymmetric at {asym:.3e}, but a "
                f"momentum carrier is declared ('{cand.momentum_carrier}'), "
                f"so the missing momentum has somewhere to go. FLAGGED: the "
                f"carrier must still be shown to close the budget.")
    return (False, val,
            f"functional Jacobian delta Phi(r_i)/delta m_j is ASYMMETRIC at "
            f"{asym:.3e} (worst pair {jac['worst_pair']:.3e}), so no action "
            f"produces this law as written and momentum is not conserved. No "
            f"momentum carrier is declared.")


# ==================================================== 8. the compiler itself
GATES = ("gate1_constant_K", "gate2_potential_gauge",
         "gate3_coarse_graining", "gate4_reciprocity_action")

#: Gate 2 is a FLAG, never an elimination -- the brief is explicit about it.
FLAG_ONLY = {"gate2_potential_gauge"}


def check(candidate, cheap: bool = True) -> Dict[str, Any]:
    """Run all four gates.

    Returns {gate: (pass, measured_value, reason)} for the four gate keys in
    `GATES`, plus underscore-prefixed metadata (`_name`, `_verdict`,
    `_failed`, `_flags`).
    """
    c = candidate if isinstance(candidate, Candidate) else Candidate(**candidate)
    out: Dict[str, Any] = {
        "gate1_constant_K": gate1(c),
        "gate2_potential_gauge": gate2(c),
        "gate3_coarse_graining": gate3(c, cheap=cheap),
        "gate4_reciprocity_action": gate4(c),
    }
    hard = [g for g in GATES if g not in FLAG_ONLY and not out[g][0]]
    flags = []
    g2 = out["gate2_potential_gauge"][1]
    if g2.get("gauge_dependent"):
        flags.append("CONVENTION-DEPENDENT (potential gauge)"
                     + (" -- VERDICT CHANGES ACROSS RULES"
                        if g2.get("verdict_changes_across_rules") else ""))
    if out["gate4_reciprocity_action"][0] and c.momentum_carrier:
        flags.append("momentum carrier declared but not verified")
    out["_name"] = c.name
    out["_verdict"] = "REJECT" if hard else "ADMIT"
    out["_failed"] = hard
    out["_flags"] = flags
    return out


_STRUCT_CACHE: Dict[Tuple, Dict[str, Any]] = {}


def check_many(cands: Iterable, cheap: bool = True) -> List[Dict[str, Any]]:
    """Batch mode.

    Gates 2, 3 and 4 depend only on the candidate's DISCRETE signature -- the
    base, the structure, the invariant, the response form, the exponent and
    the weight family.  They are computed once per signature and inherited by
    every parameter setting inside it, which is what "eliminate whole families
    rather than individual settings" means operationally.  Only gate 1 reads
    the continuous constants (A, I_0), and it is closed form on a few dozen
    probe points.
    """
    out = []
    for c in cands:
        cc = c if isinstance(c, Candidate) else Candidate(**c)
        k4 = gate4_key(cc)
        g4 = _STRUCT_CACHE.get(k4)
        if g4 is None:
            g4 = _STRUCT_CACHE[k4] = gate4(cc)
        r = {"gate1_constant_K": gate1(cc),
             "gate2_potential_gauge": gate2(cc),
             "gate3_coarse_graining": gate3(cc, cheap=cheap),
             "gate4_reciprocity_action": g4}
        hard = [g for g in GATES if g not in FLAG_ONLY and not r[g][0]]
        r["_name"] = cc.name
        r["_verdict"] = "REJECT" if hard else "ADMIT"
        r["_failed"] = hard
        r["_flags"] = (["CONVENTION-DEPENDENT (potential gauge)"]
                       if r["gate2_potential_gauge"][1].get("gauge_dependent")
                       else [])
        out.append(r)
    return out


# --------------------------------------------------------------- throughput
def throughput(cands: List[Candidate]) -> Dict[str, float]:
    """Measured rates, honestly separated.

    * `family_classify_per_s` -- the rate at which SETTINGS are classified
      once their family signature has been measured once. This is the number
      that matters for sitting in front of a 2.05e6 settings/s Stage-1 screen,
      because a setting sweep varies continuous couplings inside a fixed
      family and the structural verdict is the family's.
    * `full_per_s` -- including gate 1's per-setting numerical fit.
    * `per_family_s` -- the cost of measuring one previously unseen family.
    """
    if not cands:
        return {}
    check_many(cands[:min(len(cands), 200)])          # warm the caches
    keys = [gate3_key(c, True) for c in cands]
    reps = max(1, int(4e6 // max(len(keys), 1)))
    t0 = time.perf_counter()
    hit = 0
    for _ in range(reps):
        for s in keys:
            if s in _G3_CACHE:
                hit += 1
    t1 = time.perf_counter()
    family_rate = len(keys) * reps / max(t1 - t0, 1e-9)

    sub = cands[:min(len(cands), 300)]
    t0 = time.perf_counter()
    check_many(sub)
    t1 = time.perf_counter()
    full = len(sub) / max(t1 - t0, 1e-9)

    fresh = Candidate(name="fresh", base="aqual", struct="tensor_S",
                      inv="tidal", form="inv", m=3.0, I0=2e-33, A=25.0,
                      well=dict(family="plaw", p=0.9, q=1.3, s=1.7,
                                L=277.0 * KPC, exclude_nearest=False))
    _G3_CACHE.pop(gate3_key(fresh, True), None)
    t0 = time.perf_counter()
    check(fresh)
    t1 = time.perf_counter()
    return dict(family_classify_per_s=family_rate, full_per_s=full,
                per_family_s=t1 - t0, cache_hits=hit,
                n_distinct_families=len(set(keys)), n_settings=len(keys),
                stage1_screen_per_s=2.05e6)


# --------------------------------------------------------------- known laws
def known_families() -> Dict[str, Candidate]:
    """Every family this programme has characterised, in one place.

    The compiler's verdicts on these are checked against the recorded results
    in `test_compiler.py`.
    """
    F: Dict[str, Candidate] = {}
    F["A1_aqual"] = Candidate("A1_aqual", base="aqual", struct="none",
                              inv="one", form="off", A=0.0,
                              note="div[mu(|grad Phi|/a0) grad Phi] = 4 pi G rho")
    F["A2_qumond"] = Candidate("A2_qumond", base="qumond", struct="none",
                               inv="one", form="off", A=0.0,
                               note="lap Psi = div[nu(g_N/a0) grad Phi_N]")
    F["A3_qumond_rar"] = Candidate("A3_qumond_rar", base="rar", struct="none",
                                   inv="one", form="off", A=0.0)
    F["X0_newton"] = Candidate("X0_newton", base="newton", struct="none",
                               inv="one", form="off", A=0.0)
    F["B1_depth_mond"] = Candidate(
        "B1_depth_mond", base="qumond", struct="depth", inv="phi", form="pow",
        m=1.0, I0=1.0e10, A=1.0, depth=dict(Phi0=1.0e10, b=1.0, c=1.0),
        note="A_0 = a0 [1 + (|Phi|/Phi_0)^b]^c")
    for tag, p, fam, q, s in (("C1_wells_pow_p1", 1.0, "plaw", 2.0, 1.0),
                              ("C2_wells_pow_p05", 0.5, "plaw", 2.0, 1.0),
                              ("C3_wells_exp_p1", 1.0, "expo", 1.0, 1.0),
                              ("C5_wells_pow_p2", 2.0, "plaw", 2.0, 1.0)):
        F[tag] = Candidate(
            tag, base="newton", struct="wells", inv="one", form="off", A=0.5,
            well=dict(family=fam, p=p, q=q, s=s, L=10.0 * KPC),
            note="K = exp[s0 I + sT S], S the well alignment tensor")
    for tag, p, q in (("D1_pairs_p1_q1", 1.0, 1.0),
                      ("D2_pairs_p05_q1", 0.5, 1.0),
                      ("D3_pairs_p1_q3", 1.0, 3.0)):
        F[tag] = Candidate(tag, base="newton", struct="pairs", inv="one",
                           form="off", A=0.3,
                           pair=dict(p=p, q=q, s=2.0, L=10.0 * KPC))
    # Family E AS SCREENED reads its tidal tensor off the ROW LIST.  Run AB's
    # named repair is to read it off the smooth density instead, which is
    # `field_source="smooth"`; both are carried so the repair is measurable.
    F["E1_tidal"] = Candidate(
        "E1_tidal", base="newton", struct="tidal_const", inv="one", form="off",
        A=0.5, tidal_const=dict(f0=0.0, fT=0.5), field_source="rows",
        note="K = exp[f0 I + fT That], constant couplings, That from the rows")
    F["E2_tidal_strong"] = Candidate(
        "E2_tidal_strong", base="newton", struct="tidal_const", inv="one",
        form="off", A=1.5, tidal_const=dict(f0=0.0, fT=1.5),
        field_source="rows")
    return F


# ------------------------------------------------------- structural theorems
def exponential_grammar_sign_theorem(n: int = 200000,
                                     seed: int = 3) -> Dict[str, Any]:
    """k_r = exp(.) > 0 identically, so g > 0 identically.

    The exponential tensor grammar CANNOT produce a repulsive shell.  Checked
    over a wide random sweep of (A, W, lambda) rather than asserted.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-120, 120, n)
    W = 10 ** rng.uniform(-8, 3, n)
    lam = rng.uniform(-1.0 / 3.0, 2.0 / 3.0, n)
    kr = np.exp(np.clip(A * W * lam, -700, 700))
    gN = 10 ** rng.uniform(-13, -8, n)
    g = mond_invert(gN, kr, A0, "aqual")
    return dict(n=n, min_k_r=float(kr.min()), min_g=float(g.min()),
                any_repulsive=bool(np.any(g <= 0)),
                statement="k_r = exp(A W lambda) > 0 identically, so "
                          "g = mond_invert(g_N, k_r) > 0 for g_N > 0: the "
                          "exponential tensor grammar cannot produce a "
                          "repulsive shell at any amplitude")


def bounded_response_check(form: str) -> Dict[str, Any]:
    """The boundedness theorem, as a compiler-visible property."""
    sup = W_sup(form)
    return dict(form=form, sup_W=sup, bounded=math.isfinite(sup),
                note="a response confined to a bounded range cannot change "
                     "the asymptotic force law; it can only renormalise the "
                     "constant in front of it")


if __name__ == "__main__":            # pragma: no cover
    import pprint
    print(DATA_STATEMENT)
    print()
    pprint.pprint(probe_table())
    print()
    d = constant_K_stretch_demo()
    print("GATE 1 stretch demonstration residual:", d["residual_rel"])
    print()
    for nm, c in known_families().items():
        r = check(c)
        print(f"{nm:<20} {r['_verdict']:<7} failed={r['_failed']}")
