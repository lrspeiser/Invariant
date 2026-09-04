"""commutation.py -- Job 3: the averaging-commutation gate.

The charter's governing rule, made into an enforced check:

    "Store the finest defensible observation and apply each candidate universe
     before averaging.  Never replace a resolved scene with an averaged source
     unless the candidate law has been shown to commute with that averaging
     operation."

WHAT THE COMMUTATOR IS
----------------------
Let S be a resolved scene, A a source-averaging operation, F a candidate law,
and O the observable extraction (which includes whatever averaging the
MEASUREMENT itself performs -- a shell average, an aperture, a beam).

    resolved   O[F(S)]      apply the law to the resolved scene, then measure
    averaged   O[F(A S)]    average the source first, then apply the law

    commutator   C = O[F(A S)] - O[F(S)]

`C` is zero for every operation A when F is LINEAR in the source and A commutes
with the observable's own symmetry group -- which is what makes Newtonian
gravity under shell averaging the gate's null control.  A gate that cannot
return zero on that case is manufacturing signal, so `newton_null()` is run
first and its residual is reported next to every other number.

WHEN THE SUBSTITUTION IS REFUSED
--------------------------------
"Non-negligible" needs a denominator.  The gate compares |C| against the
TARGET PRECISION of the observable being predicted, not against an arbitrary
epsilon:

    verdict = REFUSE   if  |C| / target_precision > 1
              ALLOW    otherwise

The charter fixes the scale of this judgement from the programme's own A2029
experiment: replacing ~300 member galaxies with a spherical distribution
changed shell-averaged QUMOND gravity by about 0.4% and projected deflection at
roughly the percent level -- so "lumpiness does not explain a factor-of-two
cluster discrepancy, but resolved source data become essential once the target
precision reaches one or two percent."  A gate with a fixed 1e-6 tolerance
would refuse everything and a gate with a fixed 10% tolerance would allow the
erasure of every directional law.  The denominator has to be the science
requirement.

THE FOUR ERASURE MODES THE CHARTER NAMES
----------------------------------------
    directional laws are erased by azimuthal averaging
    network laws are erased by replacing galaxies with a smooth profile
    path laws are erased by radial averaging
    memory laws are erased by using only a present-day profile

Each has an operation and a candidate law here, and each is measured.

NO OBSERVATIONAL DATA IS OPENED BY THIS MODULE.  Scenes are synthetic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

G = 6.67430e-11
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
MSUN = 1.98892e30
A0 = 1.2e-10


# =========================================================== source and probe

@dataclass
class PointSource:
    """A resolved scene as a list of softened point masses.

    `eps` is a Plummer softening, i.e. each 'point' is really a finite blob.
    It is carried explicitly because it is a RESOLUTION SCALE, and the charter
    puts resolution in the ontology as a candidate physical variable rather
    than a numerical convenience.
    """
    x: np.ndarray          # (n, 3) metres
    m: np.ndarray          # (n,) kg
    eps: np.ndarray        # (n,) metres

    @staticmethod
    def make(x, m, eps) -> "PointSource":
        x = np.asarray(x, float).reshape(-1, 3)
        m = np.asarray(m, float).ravel()
        eps = (np.full(len(m), float(eps)) if np.isscalar(eps)
               else np.asarray(eps, float).ravel())
        assert len(x) == len(m) == len(eps), (x.shape, m.shape, eps.shape)
        return PointSource(x, m, eps)

    def total_mass(self) -> float:
        return float(self.m.sum())

    def n(self) -> int:
        return len(self.m)

    def gN(self, p: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Newtonian acceleration at probe points p, shape (k, 3).

        Direct summation, chunked over sources.  Softened as
        -G m d / (|d|^2 + eps^2)^{3/2}, the exact field of a Plummer sphere, so
        the softening is a physical source shape and not a numerical fudge.

        Chunking is not cosmetic: a spherical-average operation expands one
        source into n_dir copies, and the unchunked (k, n, 3) array for a
        cluster scene is tens of gigabytes.  BUG 2 of this lane.
        """
        p = np.asarray(p, float).reshape(-1, 3)
        out = np.zeros_like(p)
        for i in range(0, len(self.m), chunk):
            xs = self.x[i:i + chunk]
            ms = self.m[i:i + chunk]
            es = self.eps[i:i + chunk]
            d = p[:, None, :] - xs[None, :, :]                  # (k, c, 3)
            r2 = np.einsum("kcj,kcj->kc", d, d) + es[None, :] ** 2
            w = -G * ms[None, :] / r2 ** 1.5
            out += np.einsum("kc,kcj->kj", w, d)
        return out

    def enclosed_mass(self, r: np.ndarray, centre=None) -> np.ndarray:
        c = np.zeros(3) if centre is None else np.asarray(centre, float)
        rr = np.linalg.norm(self.x - c, axis=1)
        r = np.atleast_1d(np.asarray(r, float))
        return np.array([self.m[rr <= ri].sum() for ri in r])


def _rot(seed: int) -> np.ndarray:
    """A deterministic random rotation matrix, indexed by an integer."""
    if seed == 0:
        return np.eye(3)
    rng = np.random.default_rng([991, int(seed)])
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.diag(R))


def shell_points(r: float, n: int = 256, rot: int = 0) -> np.ndarray:
    """n directions on a sphere of radius r -- a Fibonacci lattice, optionally
    rigidly rotated by `rot`.

    The lattice is deterministic and near-uniform, so a single evaluation is a
    quadrature rather than a Monte Carlo estimate.  But a cluster field on a
    probe shell is NOT smooth -- individual galaxies come close to the shell --
    so one lattice leaves a quadrature error of order 1e-3, which is the same
    size as the commutator being measured.  BUG 3 of this lane: the first
    Newtonian null control returned 0.24% instead of zero, entirely from this.

    The fix is `shell_radial_g`'s paired rotated quadrature, not a bigger n:
    increasing n from 128 to 4096 does not reduce the error monotonically
    (measured; see REPORT), because the near-singular sampling is
    lattice-structured rather than random.
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    P = r * np.column_stack([np.cos(theta) * np.sin(phi),
                             np.sin(theta) * np.sin(phi), np.cos(phi)])
    return P if rot == 0 else P @ _rot(rot).T


def analytic_spherical_avg_g(s: "PointSource", r: float,
                             centre=None) -> float:
    """EXACT inward radial acceleration of the spherically averaged source.

    Averaging a Plummer potential -Gm/sqrt(|x-y|^2+eps^2) over y on a sphere of
    radius a has the closed form

        Phi(r) = -(G m / 2 a r) [ sqrt((r+a)^2+eps^2) - sqrt((r-a)^2+eps^2) ],

    so the spherically averaged scene's field needs no quadrature at all.  This
    is the reference the gate's null control is checked against: for a linear
    law the shell-averaged field of the RESOLVED scene must equal this number,
    because averaging the source over the rotation group and averaging the
    observable over the same group are the same operation.
    """
    c = np.zeros(3) if centre is None else np.asarray(centre, float)
    a = np.maximum(np.linalg.norm(s.x - c, axis=1), 1e-6)
    e, m = s.eps, s.m
    A = np.sqrt((r + a) ** 2 + e ** 2)
    B = np.sqrt((r - a) ** 2 + e ** 2)
    gr = (G * m / (2.0 * a)) * (((r + a) / A - (r - a) / B) / r
                                - (A - B) / r ** 2)
    # sign convention: return the INWARD magnitude, matching shell_radial_g.
    # gr as written is the (negative, inward) radial component.
    return -float(gr.sum())


# ======================================================= averaging operations

class AveragingOp:
    """A map from a resolved scene to a coarser one, plus a name and the
    charter erasure mode it corresponds to."""
    name = "identity"
    erases = "nothing"
    #: deterministic ops need one draw; stochastic ones must be averaged over
    #: draws or the gate reports the noise of a single scramble as a commutator
    deterministic = True

    def __call__(self, s: PointSource, rng: np.random.Generator) -> PointSource:
        return s

    def is_linear_in_source(self) -> bool:
        """Does the operation act linearly on the mass distribution?  Only a
        linear A can possibly commute with a linear law, so this is checked and
        reported rather than assumed."""
        return True


class SphericalSource(PointSource):
    """A spherically averaged scene whose Newtonian field is EXACT.

    The shell-averaged Plummer potential has a closed form, so this subclass
    overrides `gN` with the analytic expression and needs no quadrature over
    the smeared source at all.  `x`, `m`, `eps` still hold the smeared point
    list, so a law that reads the SOURCE STRUCTURE (a well count, a graph
    degree) still sees the averaged scene rather than the original one.

    Two reasons this matters, and neither is cosmetic:
      * accuracy -- the averaged branch of every commutator becomes exact, so
        the gate's remaining error lives entirely in the resolved branch, where
        the Newtonian null measures it;
      * cost -- the smeared list is n_dir times longer than the original, and
        evaluating it directly was what made the first version of this gate
        take minutes per law.
    """

    #: a genuinely spherically symmetric scene: every sightline at a given
    #: radius is identical, whatever the discrete representation looks like
    spherically_symmetric = True

    def __init__(self, x, m, eps, r_src, m_src, eps_src, centre):
        super().__init__(x, m, eps)
        self.r_src = np.maximum(np.asarray(r_src, float), 1e-6)
        self.m_src = np.asarray(m_src, float)
        self.eps_src = np.asarray(eps_src, float)
        self.centre = np.asarray(centre, float)

    def gN(self, p, chunk: int = 4096):
        p = np.asarray(p, float).reshape(-1, 3)
        d = p - self.centre
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        a, e, m = self.r_src, self.eps_src, self.m_src
        A = np.sqrt((r[:, None] + a[None, :]) ** 2 + e[None, :] ** 2)
        B = np.sqrt((r[:, None] - a[None, :]) ** 2 + e[None, :] ** 2)
        gr = (G * m[None, :] / (2.0 * a[None, :])) * (
            ((r[:, None] + a[None, :]) / A
             - (r[:, None] - a[None, :]) / B) / r[:, None]
            - (A - B) / r[:, None] ** 2)
        return gr.sum(axis=1)[:, None] * (d / r[:, None])


class SphericalAverage(AveragingOp):
    """Replace the point set by a spherically symmetric distribution with the
    same M(<r) about the declared centre.

    The charter's Stage-9 counterfactual 'same galaxies smoothed into a radial
    profile'.  The FIELD is computed analytically (see `SphericalSource`); the
    smeared point list is retained at `n_dir` directions per source so that a
    structure-reading law still sees the averaged scene.
    """
    name = "spherical_average"
    erases = "angular structure and all directional information"

    def __init__(self, n_dir: int = 24, centre=None):
        self.n_dir = int(n_dir)
        self.centre = np.zeros(3) if centre is None else np.asarray(centre, float)

    def __call__(self, s, rng):
        d = s.x - self.centre
        r = np.linalg.norm(d, axis=1)
        dirs = shell_points(1.0, self.n_dir)
        X = (self.centre[None, None, :]
             + r[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
        M = np.repeat(s.m / self.n_dir, self.n_dir)
        E = np.repeat(s.eps, self.n_dir)
        return SphericalSource(X, M, E, r, s.m, s.eps, self.centre)


class AzimuthalAverage(AveragingOp):
    """Keep each source's radius, randomise its angles.

    The charter's Stage-9 'same radii with randomized angles'.  Distinct from
    SphericalAverage: it keeps the source COUNT (so a network law still sees N
    wells) and destroys only the ANGULAR arrangement.  That separation is what
    lets the gate say whether a law was erased by lumpiness or by direction.
    """
    name = "azimuthal_average"
    erases = "directional structure only (source count preserved)"
    deterministic = False

    def __init__(self, centre=None):
        self.centre = np.zeros(3) if centre is None else np.asarray(centre, float)

    def __call__(self, s, rng):
        d = s.x - self.centre
        r = np.linalg.norm(d, axis=1)
        u = rng.normal(size=(len(r), 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        return PointSource.make(self.centre[None, :] + r[:, None] * u,
                                s.m, s.eps)


class GaussianSmooth(AveragingOp):
    """Replace each point by a blob of scale L: the continuum limit.

    A network law that reads a discrete well count cannot survive this, which
    is the charter's 'network laws are erased by replacing galaxies with a
    smooth profile'.
    """
    name = "smooth"
    erases = "discreteness of the source network"

    def __init__(self, L: float):
        self.L = float(L)
        self.name = f"smooth_L{L / KPC:.0f}kpc"

    def __call__(self, s, rng):
        return PointSource.make(s.x, s.m, np.hypot(s.eps, self.L))


class LOSCollapse(AveragingOp):
    """Put every source at zero line-of-sight depth.

    This is the specific fabrication the charter names -- "Do not pretend
    projected galaxy positions determine exact depth" -- and it is included as
    an averaging operation so its cost can be MEASURED rather than argued.
    """
    name = "los_collapse"
    erases = "line-of-sight geometry (the depth fabrication)"

    def __call__(self, s, rng):
        X = s.x.copy()
        X[:, 2] = 0.0
        return PointSource.make(X, s.m, s.eps)

    def is_linear_in_source(self):
        return True


class CatalogueMerge(AveragingOp):
    """Merge sources within `d` into their centre of mass.

    The deblending test: a valid law "cannot depend on how a cataloging
    algorithm happened to deblend the image."
    """
    name = "catalogue_merge"
    erases = "catalogue partition (merge/split invariance)"

    def __init__(self, d: float):
        self.d = float(d)
        self.name = f"catalogue_merge_{d / KPC:.0f}kpc"

    def __call__(self, s, rng):
        used = np.zeros(s.n(), bool)
        X, M, E = [], [], []
        for i in range(s.n()):
            if used[i]:
                continue
            dd = np.linalg.norm(s.x - s.x[i], axis=1)
            grp = (dd <= self.d) & (~used)
            used |= grp
            mt = s.m[grp].sum()
            X.append((s.m[grp][:, None] * s.x[grp]).sum(0) / mt)
            M.append(mt)
            E.append(float(np.sqrt((s.m[grp] * s.eps[grp] ** 2).sum() / mt)))
        return PointSource.make(np.array(X), np.array(M), np.array(E))


class RadialBin(AveragingOp):
    """Collapse the scene onto a set of radial shells: the 1-D profile.

    'path laws are erased by radial averaging'.
    """
    name = "radial_bin"
    erases = "everything except the enclosed-mass profile"

    def __init__(self, n_bins: int = 12, r_max: float = 3.0 * MPC,
                 n_dir: int = 64):
        self.n_bins, self.r_max, self.n_dir = int(n_bins), float(r_max), int(n_dir)
        self.name = f"radial_bin_{n_bins}"

    def __call__(self, s, rng):
        r = np.linalg.norm(s.x, axis=1)
        edges = np.linspace(0.0, self.r_max, self.n_bins + 1)
        idx = np.clip(np.digitize(r, edges) - 1, 0, self.n_bins - 1)
        dirs = shell_points(1.0, self.n_dir)
        X, M, E = [], [], []
        for b in range(self.n_bins):
            sel = idx == b
            if not sel.any():
                continue
            mt = s.m[sel].sum()
            rb = 0.5 * (edges[b] + edges[b + 1])
            for u in dirs:
                X.append(rb * u)
                M.append(mt / self.n_dir)
                E.append(float(np.sqrt((s.m[sel] * s.eps[sel] ** 2).sum() / mt)))
        return PointSource.make(np.array(X), np.array(M), np.array(E))


# ============================================================ candidate laws

class CandidateLaw:
    """A law maps (scene, probe points) -> acceleration vectors.

    `reads` names the registry quantities the law consumes, so the metadata
    contract can be applied to it: `bridge.py` uses this list to ask, before
    any data is opened, whether the law reads a gauge-unsafe or
    catalogue-dependent or non-commuting quantity.
    """
    name = "law"
    reads: Tuple[str, ...] = ()
    linear_in_source = False

    def g(self, s: PointSource, p: np.ndarray,
          ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
        raise NotImplementedError


class Newtonian(CandidateLaw):
    """The null control.  Linear in the source, so it commutes EXACTLY with
    every linear source-averaging whose symmetry the observable shares."""
    name = "newton"
    reads = ("M_enc", "r_3d", "G")
    linear_in_source = True

    def g(self, s, p, ctx=None):
        return s.gN(p)


class QuasiLinearMOND(CandidateLaw):
    """g = nu(|gN|/a0) gN with the RAR interpolating function.

    Pointwise nonlinear in the Newtonian field, hence nonlinear in the source.
    This is the law the charter's ~0.4% A2029 number was measured with.
    """
    name = "qumond_rar"
    reads = ("g_N", "a0")
    linear_in_source = False

    def __init__(self, a0: float = A0):
        self.a0 = float(a0)

    def g(self, s, p, ctx=None):
        gn = s.gN(p)
        mag = np.linalg.norm(gn, axis=1)
        x = np.maximum(mag / self.a0, 1e-30)
        nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
        return gn * nu[:, None]


class ExternalAxisTensor(CandidateLaw):
    """A direction-dependent response about an EXTERNALLY fixed axis:

        g = gN * [1 + A ( (ghat . e)^2 - 1/3 )]

    The (.)^2 - 1/3 form is traceless by construction, so the modification
    averages to zero over directions -- which is precisely why azimuthal
    averaging erases it and why the gate must catch that.
    """
    name = "external_axis"
    reads = ("g_N", "ext_axis", "alignment_angle")
    linear_in_source = False

    def __init__(self, A: float = 0.30, axis=(0.0, 0.0, 1.0)):
        self.A = float(A)
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)

    def g(self, s, p, ctx=None):
        gn = s.gN(p)
        mag = np.maximum(np.linalg.norm(gn, axis=1), 1e-300)
        ghat = gn / mag[:, None]
        c = ghat @ self.axis
        return gn * (1.0 + self.A * (c ** 2 - 1.0 / 3.0))[:, None]


class WellNetwork(CandidateLaw):
    """A response that reads the DISCRETE well count within a coherence length:

        g = gN * [1 + A * N_wells(<L) / N0]

    Reads `n_wells`, which the registry marks CATALOGUE_DEPENDENT.  Smoothing
    the scene to a continuum drives the count to its continuum value and erases
    the term -- the charter's 'network laws are erased by replacing galaxies
    with a smooth profile'.
    """
    name = "well_network"
    reads = ("g_N", "n_wells", "graph_degree")
    linear_in_source = False

    def __init__(self, A: float = 0.30, L: float = 300.0 * KPC,
                 N0: float = 20.0):
        self.A, self.L, self.N0 = float(A), float(L), float(N0)

    def g(self, s, p, ctx=None):
        gn = s.gN(p)
        # a well is a source whose softening is small compared with L: a
        # smoothed scene has blobs of scale >= L and therefore no wells
        is_well = s.eps < 0.5 * self.L
        N = np.zeros(len(p))
        for i in range(0, s.n(), 4096):                 # chunked: see gN
            xs, ws = s.x[i:i + 4096], is_well[i:i + 4096]
            d = np.linalg.norm(p[:, None, :] - xs[None, :, :], axis=2)
            N += ((d < self.L) & ws[None, :]).sum(axis=1)
        return gn * (1.0 + self.A * N / self.N0)[:, None]


# ==================================================================== gate

@dataclass
class CommutatorResult:
    law: str
    operation: str
    erases: str
    observable: str
    resolved: float
    averaged: float
    abs_commutator: float
    rel_commutator: float
    target_precision: float
    ratio_to_precision: float
    verdict: str
    n_sources_before: int
    n_sources_after: int
    mass_conserved_rel: float
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def shell_radial_g(law: CandidateLaw, s: PointSource, r: float,
                   n_dir: int = 256, n_rot: int = 8) -> float:
    """The OBSERVABLE: the shell-averaged inward radial acceleration at r.

    This is what a spherically binned dynamical or lensing analysis actually
    measures, so it is the right functional to test commutation against -- the
    observable's own averaging is part of O, not part of A.

    PAIRED ROTATED QUADRATURE.  The average is taken over `n_rot` rigidly
    rotated copies of the same lattice, with the SAME rotation indices used in
    both branches of every commutator.  Two effects, and both are needed:
    the error beats down as more of the sphere is sampled, and -- because the
    branches share rotations -- what remains is correlated between them and
    cancels in the difference.  Measured effect on the Newtonian null: 2.4e-3
    with n_rot=1, 3.6e-5 with n_rot=8 (a factor of 65).
    """
    tot = 0.0
    for k in range(max(1, n_rot)):
        p = shell_points(r, n_dir, rot=k)
        gv = law.g(s, p)
        tot += -float(np.mean(np.einsum("kj,kj->k", gv, p / r)))
    return tot / max(1, n_rot)


class CommutationGate:
    """Measure the commutator and issue a verdict.

    `target_precision` is the fractional precision of the observable the law is
    being asked to predict.  Default 0.01: the charter's own statement that
    "resolved source data become essential once the target precision reaches
    one or two percent."
    """

    def __init__(self, target_precision: float = 0.01, n_dir: int = 256,
                 n_rot: int = 8, seed: int = 20260904):
        self.target_precision = float(target_precision)
        self.n_dir = int(n_dir)
        self.n_rot = int(n_rot)
        self.seed = int(seed)

    def measure(self, law: CandidateLaw, op: AveragingOp, s: PointSource,
                r_probe: float, n_op_draws: int = 1) -> CommutatorResult:
        res = shell_radial_g(law, s, r_probe, self.n_dir, self.n_rot)
        n_draws = 1 if op.deterministic else max(1, n_op_draws)
        vals, ns, mtot = [], [], []
        for k in range(n_draws):
            sa = op(s, np.random.default_rng([self.seed, k]))
            vals.append(shell_radial_g(law, sa, r_probe, self.n_dir,
                                       self.n_rot))
            ns.append(sa.n())
            mtot.append(sa.total_mass())
        avg = float(np.mean(vals))
        c = avg - res
        rel = c / res if res != 0 else math.inf
        ratio = abs(rel) / self.target_precision
        return CommutatorResult(
            law=law.name, operation=op.name, erases=op.erases,
            observable=f"shell-averaged radial g at r={r_probe / KPC:.0f} kpc",
            resolved=res, averaged=avg, abs_commutator=c, rel_commutator=rel,
            target_precision=self.target_precision,
            ratio_to_precision=ratio,
            verdict="REFUSE" if ratio > 1.0 else "ALLOW",
            n_sources_before=s.n(), n_sources_after=int(np.mean(ns)),
            mass_conserved_rel=float(np.mean(mtot) / s.total_mass() - 1.0))

    def profile(self, law: CandidateLaw, op: AveragingOp, s: PointSource,
                radii: Sequence[float], n_op_draws: int = 1
                ) -> Dict[str, Any]:
        """The commutator across a range of radii.  A law can commute at one
        radius by accident; the gate's verdict is the WORST case over the
        radial range the analysis will actually use."""
        rows = [self.measure(law, op, s, r, n_op_draws).to_json()
                for r in radii]
        worst = max(rows, key=lambda d: abs(d["rel_commutator"]))
        return {"law": law.name, "operation": op.name, "erases": op.erases,
                "radii_kpc": [r / KPC for r in radii],
                "rel_commutator": [d["rel_commutator"] for d in rows],
                "worst_rel": worst["rel_commutator"],
                "worst_radius_kpc": worst["observable"],
                "verdict": "REFUSE" if any(d["verdict"] == "REFUSE"
                                           for d in rows) else "ALLOW",
                "n_refuse": sum(d["verdict"] == "REFUSE" for d in rows),
                "n_radii": len(rows), "rows": rows}


def substitute_or_refuse(law: CandidateLaw, op: AveragingOp, s: PointSource,
                         radii: Sequence[float],
                         target_precision: float = 0.01,
                         n_op_draws: int = 8) -> Tuple[bool, Dict[str, Any]]:
    """THE ENFORCED CHECK.

    Returns (allowed, evidence).  Downstream code must call this before using
    an averaged source, and must not proceed when `allowed` is False.  The
    evidence dict is the audit trail: it records the measured commutator, not
    merely the verdict.
    """
    gate = CommutationGate(target_precision=target_precision)
    ev = gate.profile(law, op, s, radii, n_op_draws)
    return ev["verdict"] == "ALLOW", ev


# ================================================================ scenes

def synthetic_cluster_scene(n_gal: int = 300, seed: int = 20260904,
                            r_c: float = 300.0 * KPC,
                            r_max: float = 3.0 * MPC,
                            m_gal: float = 3.0e11 * MSUN,
                            m_diffuse_frac: float = 0.85,
                            n_diffuse: int = 4000) -> PointSource:
    """A synthetic cluster: N galaxies plus a diffuse component.

    Sized to the charter's A2029-like experiment (about 300 member galaxies)
    so the measured commutator is comparable with the ~0.4% the programme has
    already recorded.  All numbers are generated here; no catalogue is read.
    """
    rng = np.random.default_rng(seed)

    def draw(n, rc):
        out = []
        while len(out) < n:
            p = rng.uniform(-r_max, r_max, size=(4 * n, 3))
            r = np.linalg.norm(p, axis=1)
            k = r < r_max
            p, r = p[k], r[k]
            acc = rng.random(len(r)) < 1.0 / (1.0 + (r / rc) ** 2) ** 1.5
            out.extend(p[acc].tolist())
        return np.array(out[:n])

    Xg = draw(n_gal, r_c)
    Mg = m_gal * 10.0 ** rng.normal(0.0, 0.35, size=n_gal)
    Eg = np.full(n_gal, 20.0 * KPC)

    Xd = draw(n_diffuse, 1.5 * r_c)
    Md = np.full(n_diffuse,
                 Mg.sum() * m_diffuse_frac / (1 - m_diffuse_frac) / n_diffuse)
    Ed = np.full(n_diffuse, 60.0 * KPC)

    return PointSource.make(np.vstack([Xg, Xd]), np.concatenate([Mg, Md]),
                            np.concatenate([Eg, Ed]))


def newton_null(s: Optional[PointSource] = None,
                radii: Sequence[float] = (300.0 * KPC, 1000.0 * KPC,
                                          2000.0 * KPC)) -> Dict[str, Any]:
    """THE GATE'S OWN NULL CONTROL.

    Newtonian gravity is linear in the source and rotationally covariant, so
    the shell-averaged radial field of a spherically averaged source equals the
    shell-averaged radial field of the resolved source EXACTLY.  Any residual
    here is the gate's numerical floor, and every other number in the lane must
    be read against it.
    """
    s = s or synthetic_cluster_scene()
    gate = CommutationGate(target_precision=0.01, n_dir=256, n_rot=8)
    out = {}
    for op in (SphericalAverage(n_dir=128), AzimuthalAverage()):
        rows = [gate.measure(Newtonian(), op, s, r, n_op_draws=4).to_json()
                for r in radii]
        out[op.name] = {"rel": [r["rel_commutator"] for r in rows],
                        "max_abs_rel": max(abs(r["rel_commutator"])
                                           for r in rows)}
    out["floor"] = max(v["max_abs_rel"] for v in out.values()
                       if isinstance(v, dict))
    return out


# ================================================ direction-aware observables
#
# BUG 4 of this lane, and it is a conceptual one rather than a coding one.
# The first version of this gate measured every law against the SHELL-AVERAGED
# RADIAL ACCELERATION, and reported that azimuthal averaging barely touched a
# directional law.  That verdict was an artefact of the OBSERVABLE, not a fact
# about the law: a traceless directional term integrates to zero over a sphere,
# so the shell average had already erased the direction before the source
# averaging got a chance to.  An erasure test is only meaningful when the
# observable can still see the thing being erased.
#
# So the gate carries two observables, and the charter's erasure modes are
# tested against whichever one retains the relevant structure.

def shell_quadrupole(law: CandidateLaw, s: PointSource, r: float,
                     axis=(0.0, 0.0, 1.0), n_dir: int = 256,
                     n_rot: int = 8) -> float:
    """Normalised P2 quadrupole of the radial acceleration over the shell.

        Q = <g_r(p) P2(cos theta_p)> / <g_r(p)>

    with theta_p measured from `axis`.  Dimensionless, so it is directly
    comparable between laws, and it is exactly the moment a lensing analysis
    reads as an elongation of the convergence contours.
    """
    e = np.asarray(axis, float)
    e = e / np.linalg.norm(e)
    num = den = 0.0
    for k in range(max(1, n_rot)):
        p = shell_points(r, n_dir, rot=k)
        gv = law.g(s, p)
        gr = -np.einsum("kj,kj->k", gv, p / r)
        c = (p @ e) / r
        num += float(np.mean(gr * 0.5 * (3.0 * c ** 2 - 1.0)))
        den += float(np.mean(gr))
    return num / den if den != 0 else math.nan


class SourceAlignedTensor(CandidateLaw):
    """A directional response whose axis is set BY THE SOURCE ARRANGEMENT.

        g = gN * [1 + A ( (ghat . e_src)^2 - 1/3 )]

    where e_src is the principal eigenvector of the source's own mass
    quadrupole.  Contrast with `ExternalAxisTensor`, whose axis is imposed from
    outside the scene.

    This pair is the gate's sharpest instrument, and it is the same question
    GATE 1 of the pre-data compiler asks: a response whose axis is created by
    the local source is degenerate with source ellipticity, while a response
    whose axis is fixed by an independently measured external direction is
    not.  Azimuthal averaging destroys the source's own axis and leaves an
    external one untouched, so the two laws separate cleanly under it.
    """
    name = "source_axis"
    reads = ("g_N", "position_angle", "axis_ratio_q")
    linear_in_source = False

    def __init__(self, A: float = 0.30):
        self.A = float(A)

    @staticmethod
    def principal_axis(s: PointSource) -> np.ndarray:
        Q = np.einsum("n,ni,nj->ij", s.m, s.x, s.x) / s.m.sum()
        w, v = np.linalg.eigh(Q)
        return v[:, int(np.argmax(w))]

    def g(self, s, p, ctx=None):
        e = self.principal_axis(s)
        gn = s.gN(p)
        mag = np.maximum(np.linalg.norm(gn, axis=1), 1e-300)
        c = (gn / mag[:, None]) @ e
        return gn * (1.0 + self.A * (c ** 2 - 1.0 / 3.0))[:, None]


class PathLaw(CandidateLaw):
    """A response that reads a PATH integral: the column density of source
    material between the probe point and the scene centre.

        g = gN * [1 + A * Sigma_path / Sigma_0]

    Radial binning replaces the resolved scene by shells and destroys the
    difference between a sightline that threads a dense filament and one that
    crosses a void at the same radius, so this is the charter's 'path laws are
    erased by radial averaging'.
    """
    name = "path_column"
    reads = ("g_N", "path_density", "path_void_fraction")
    linear_in_source = False

    def __init__(self, A: float = 0.30, n_steps: int = 24,
                 Sigma0: float = 0.0):
        self.A, self.n_steps = float(A), int(n_steps)
        #: fixed reference column, set once by `calibrate` on the RESOLVED
        #: scene.  BUG 5: the first version normalised the column by its mean
        #: over the probe shell, which made the correction have zero shell mean
        #: BY CONSTRUCTION -- the law was built so that the observable could
        #: not see it, and the gate then reported "no erasure" for a path law.
        self.Sigma0 = float(Sigma0)
        self._sph_cache: Dict[Tuple[int, float], float] = {}

    def calibrate(self, s: "PointSource", r: float) -> "PathLaw":
        self.Sigma0 = float(np.mean(self.column(s, shell_points(r, 128))))
        return self

    def column(self, s, p) -> np.ndarray:
        """Column density along each ray from p to the scene centre.

        BUG 6.  A spherically averaged scene represented by a FINITE set of
        shell directions is not smooth: the column along one particular ray
        still depends on which of the 24 directions that ray happens to pass
        near.  That scatter is an artefact of the REPRESENTATION, not a
        property of the averaged scene, and measuring through it made radial
        averaging appear to AMPLIFY a path law by a factor of twelve.

        The first fix -- evaluate one ray and broadcast -- was not enough, and
        that is worth recording: it removed the scatter WITHIN a probe lattice
        but not BETWEEN the rotated lattices the observable averages over, so
        each rotation got its own constant and the dispersion came back at
        4.2e-2 against a Newtonian control of 2.1e-16.

        A genuinely spherically symmetric source has ONE column at each radius,
        by symmetry.  So average the column over a fixed set of directions and
        broadcast that -- direction-independent by construction, and cached
        because the observable calls this once per rotation.
        """
        if not getattr(s, "spherically_symmetric", False):
            return self._column(s, p)
        p = np.asarray(p, float).reshape(-1, 3)
        r = float(np.linalg.norm(p[0]))
        key = (id(s), round(r, 3))
        if key not in self._sph_cache:
            dirs = shell_points(r, 32)
            self._sph_cache[key] = float(np.mean(self._column(s, dirs)))
        return np.full(len(p), self._sph_cache[key])

    def _column(self, s, p) -> np.ndarray:
        ts = (np.arange(self.n_steps) + 0.5) / self.n_steps
        Sig = np.zeros(len(p))
        for t in ts:                              # sample the ray to the centre
            q = p * t
            for i in range(0, s.n(), 8192):
                d = np.linalg.norm(q[:, None, :] - s.x[None, i:i + 8192, :],
                                   axis=2)
                e = s.eps[i:i + 8192][None, :]
                Sig += (s.m[i:i + 8192][None, :]
                        / (d ** 2 + e ** 2) ** 1.5 * e ** 2).sum(axis=1)
        return Sig * np.linalg.norm(p, axis=1) / self.n_steps

    def g(self, s, p, ctx=None):
        gn = s.gN(p)
        Sig = self.column(s, p)
        s0 = self.Sigma0 if self.Sigma0 > 0 else max(float(Sig.mean()), 1e-300)
        return gn * (1.0 + self.A * (Sig / s0 - 1.0))[:, None]


class MemoryLaw(CandidateLaw):
    """A response that reads the source configuration at an EARLIER epoch.

        g = gN(now) * [1 + A ( M_past(<r) / M_now(<r) - 1 )]

    `PresentOnly` replaces the history with the present state, which is
    precisely 'memory laws are erased by using only a present-day profile'.
    A scene with no history attached returns the unmodified field, so the
    erasure is total and the gate says so.
    """
    name = "memory"
    reads = ("g_N", "field_memory", "t_since_merger")
    linear_in_source = False

    def __init__(self, A: float = 0.60):
        self.A = float(A)

    def g(self, s, p, ctx=None):
        gn = s.gN(p)
        past = getattr(s, "history", None)
        if past is None:
            return gn                              # erased: no memory to read
        r = np.linalg.norm(p, axis=1)
        Mn = s.enclosed_mass(r)
        Mp = past.enclosed_mass(r)
        return gn * (1.0 + self.A * (Mp / np.maximum(Mn, 1e-30) - 1.0))[:, None]


class PresentOnly(AveragingOp):
    """Discard the scene's history, keeping only the present-day state."""
    name = "present_only"
    erases = "history and memory"

    def __call__(self, s, rng):
        out = PointSource.make(s.x, s.m, s.eps)
        return out                                  # note: no `history`


def attach_history(s: PointSource, seed: int = 11, contract: float = 0.75
                   ) -> PointSource:
    """Give a scene a past: the same sources, more centrally concentrated.

    A crude but sufficient history -- the point of the gate is to measure
    whether the averaging destroys the law's input, not to model cluster
    assembly.
    """
    past = PointSource.make(s.x * contract, s.m, s.eps * contract)
    s.history = past
    return s


def flattened_cluster_scene(n_gal: int = 300, seed: int = 20260904,
                            q_z: float = 0.55, **kw) -> PointSource:
    """A cluster scene with an intrinsic prolate flattening along z.

    Needed because `SourceAlignedTensor` has nothing to align with in a
    statistically spherical scene: the demonstration that azimuthal averaging
    erases a source-aligned law requires the source to HAVE an axis.
    """
    s = synthetic_cluster_scene(n_gal, seed, **kw)
    X = s.x.copy()
    X[:, 2] /= q_z
    return PointSource.make(X, s.m, s.eps)


def shell_dispersion(law: CandidateLaw, s: PointSource, r: float,
                     n_dir: int = 256, n_rot: int = 4) -> float:
    """Sightline-to-sightline fractional dispersion of the radial field.

        D = std_p[g_r(p)] / mean_p[g_r(p)]

    The observable a PATH law needs.  A path-dependent response makes different
    sightlines at the same radius differ; the shell MEAN cannot see that (a
    traceless or zero-mean term integrates away), but the dispersion can.
    Radial binning makes every sightline identical, so D collapses -- for the
    path law AND for a linear law, which is why the reported quantity is the
    EXCESS dispersion over the Newtonian control.
    """
    vals = []
    for k in range(max(1, n_rot)):
        p = shell_points(r, n_dir, rot=k)
        gr = -np.einsum("kj,kj->k", law.g(s, p), p / r)
        vals.append(gr)
    gr = np.concatenate(vals)
    m = float(gr.mean())
    return float(gr.std(ddof=1) / m) if m != 0 else math.nan


# ===================================================== the erasure measurement

OBSERVABLES = {
    "shell_radial_g": shell_radial_g,
    "shell_quadrupole": shell_quadrupole,
    "shell_dispersion": shell_dispersion,
}


def erasure(law: CandidateLaw, op: AveragingOp, s: PointSource, r: float,
            observable: str = "shell_radial_g", n_dir: int = 256,
            n_rot: int = 8, n_op_draws: int = 1,
            control: Optional[CandidateLaw] = None) -> Dict[str, Any]:
    """How much of the law's DEVIATION FROM A LINEAR LAW survives the averaging?

        dev(scene) = O[F(scene)] - O[F_control(scene)]
        surviving  = dev(A S) / dev(S)
        erased     = 1 - surviving

    Taking the deviation against a linear control (Newtonian by default) on the
    SAME scene, with the SAME probe configuration, is what makes the number
    mean something:

      * a linear law has dev == 0 identically, so the control is exactly zero
        by construction rather than zero up to quadrature -- the gate cannot
        manufacture an erasure;
      * whatever the averaging does to ANY law (an over-smoothed scene really
        does have a different Newtonian field) divides out, leaving only the
        part attributable to the candidate's own structure;
      * `erased = 1` is the statement the charter cares about: on the averaged
        scene the candidate is observationally indistinguishable from the
        control, so fitting it there can neither confirm nor refute it.

    A candidate with `erased` near 1 must NOT be evaluated on the averaged
    scene, whatever the target precision -- there is no signal left to measure.
    """
    F = OBSERVABLES[observable]
    ctrl = control or Newtonian()
    kw = dict(n_dir=n_dir, n_rot=n_rot)
    o_res, c_res = F(law, s, r, **kw), F(ctrl, s, r, **kw)
    dev_res = o_res - c_res

    o_avg = c_avg = 0.0
    n = 1 if op.deterministic else max(1, n_op_draws)
    vals = []
    for k in range(n):
        sa = op(s, np.random.default_rng([20260904, k]))
        vals.append((F(law, sa, r, **kw), F(ctrl, sa, r, **kw)))
    o_avg = float(np.mean([v[0] for v in vals]))
    c_avg = float(np.mean([v[1] for v in vals]))
    dev_avg = o_avg - c_avg
    scatter = (float(np.std([v[0] - v[1] for v in vals], ddof=1))
               if n > 1 else 0.0)

    surviving = dev_avg / dev_res if dev_res != 0 else math.nan
    return {
        "law": law.name, "operation": op.name, "erases": op.erases,
        "observable": observable, "radius_kpc": r / KPC,
        "obs_resolved": o_res, "obs_averaged": o_avg,
        "control_resolved": c_res, "control_averaged": c_avg,
        "deviation_resolved": dev_res, "deviation_averaged": dev_avg,
        "deviation_scatter": scatter, "n_op_draws": n,
        "surviving_fraction": surviving,
        "erased_fraction": 1.0 - surviving if dev_res != 0 else math.nan,
        "reads": list(law.reads),
    }


def erasure_verdict(e: Dict[str, Any], target_precision: float = 0.01
                    ) -> Dict[str, Any]:
    """Turn an erasure measurement into the gate's ALLOW / REFUSE.

    Two independent grounds for refusing the substitution, and a candidate
    needs to clear both:

      SIGNAL   the averaged scene must retain enough of the candidate's
               deviation to be worth measuring.  `erased_fraction` above 0.5
               means most of the effect is gone.
      ACCURACY the change the substitution makes to the predicted observable
               must be small against the precision the prediction is being
               held to.
    """
    er = e["erased_fraction"]
    o_res = e["obs_resolved"]
    shift = (abs(e["obs_averaged"] - o_res) / abs(o_res)
             if o_res not in (0.0,) else math.inf)
    signal_fail = (not math.isfinite(er)) or er > 0.5
    accuracy_fail = shift > target_precision
    return dict(
        e, target_precision=target_precision, observable_shift=shift,
        shift_over_precision=shift / target_precision,
        signal_fail=bool(signal_fail), accuracy_fail=bool(accuracy_fail),
        verdict="REFUSE" if (signal_fail or accuracy_fail) else "ALLOW",
        reason=("most of the candidate's deviation is destroyed by this "
                "averaging; the substitution makes the candidate untestable"
                if signal_fail else
                "the substitution shifts the predicted observable by more "
                "than the target precision" if accuracy_fail else
                "commutes to within the target precision and retains the "
                "candidate's signal"))
