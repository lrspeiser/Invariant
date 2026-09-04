"""controls.py -- the control harness every claim in the well-network programme must pass.

Built BEFORE the discoveries it polices, on purpose. Run J showed the search
machinery manufactures gain on its own: a physics-free twin with radial
structure permuted away inside each galaxy recovered +3.1% of training
improvement at K = 8 and +2.1% at k = 3, while the real winners came out ~4%
WORSE than the RAR on blind galaxies. A number produced without a control of
that kind is not a measurement.

The nine controls, in the order the brief lists them:

  1  residual_null            preserve sigma and object offsets, destroy the
                              proposed spatial signal.  Generalises the SPARC
                              `perm_g` from curves to fields.
  2  position_scramble        preserve every member's mass AND its clustercentric
                              radius, randomise its angular position.  Holds the
                              radial mass profile exactly fixed and destroys the
                              geometry of the well network.
  3  mass_scramble            preserve the geometry exactly, permute the member
                              masses among the positions.
  4  smoothed_source          same radial mass profile, angularly averaged.
                              (= the ensemble mean of control 2; verified.)
  5  synthetic_universes      mocks under five KNOWN laws, then the full
                              discovery pipeline on each.  Must recover the
                              injected family; must NOT invent tensor or
                              nonlocal effects in scalar data.  Reports the
                              false-positive rate for "tensor detected in
                              scalar data".
  6  assert_parameter_responsive
                              dS/dtheta != 0 over the tested range, as a hard
                              raise.  A rank statistic in this programme was
                              bit-identical across three decades of the coupling
                              it was supposed to measure.
  7  check_exchangeability    instruments the pipeline and diffs the operation
                              traces of the true and shuffled arms.  Not a
                              docstring promise: the numpy/scipy entry points are
                              patched and every call is recorded.
  8  shared_denominator_report
                              detect an input common to both axes, measure the
                              induced error correlation, and compute the NULL
                              EXPECTATION of the naive estimator instead of
                              assuming zero.  Plus an errors-in-variables
                              estimator validated as unbiased by simulation.
  9  SplitData / FrozenModel  makes "re-solve the coefficients on the held-out
                              set" structurally inexpressible.

Everything is plain numpy + scipy on the CPU. The controls are meant to be
cheap enough that there is no excuse for skipping them; the expensive GPU
search in work/gravitylab stays where it is and calls into this module.

Conventions
-----------
Every control is a callable  f(source, seed_or_rng, **opts) -> ControlRealisation.
A ControlRealisation carries the realised data, the invariants that were
CHECKED to be preserved (with the numerical residual of the check), and the
quantities that were deliberately destroyed. Nothing is asserted in prose that
is not also asserted in code.

Units: kpc, Msun, km/s. G = 4.300917270e-6 kpc (km/s)^2 / Msun.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import hmac
import math
import os
import secrets
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:
    from scipy import ndimage as _ndimage
except Exception:                                       # pragma: no cover
    _ndimage = None
try:
    from scipy import optimize as _optimize
    from scipy import stats as _stats
except Exception:                                       # pragma: no cover
    _optimize = _stats = None

__all__ = [
    "ControlRealisation", "ObjectPointSource", "FieldSource", "ClusterSource",
    "residual_null", "position_scramble", "mass_scramble", "smoothed_source",
    "assert_parameter_responsive", "ParameterBlindError",
    "check_exchangeability", "ExchangeabilityError", "trace_ops",
    "shared_denominator_report", "eiv_fit", "validate_eiv",
    "SplitData", "FrozenModel", "SealedHoldoutError", "FrozenSealError",
    "synthetic_universe", "run_discovery", "tensor_false_positive_rate",
    "G_KPC", "A0_KPC", "nu_rar",
]

# --------------------------------------------------------------------------
#  constants
# --------------------------------------------------------------------------
G_KPC = 4.300917270e-6            # kpc (km/s)^2 / Msun
KPC_M = 3.0856775814913673e19     # m
#: a0 = 1.2e-10 m/s^2 expressed in (km/s)^2 / kpc
A0_KPC = 1.2e-10 * KPC_M / 1.0e6


def nu_rar(x):
    """RAR / MOND interpolation, g = nu(x) g_N with x = g_N/a0."""
    x = np.maximum(np.asarray(x, float), 1e-30)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


# --------------------------------------------------------------------------
#  fingerprints and the realisation container
# --------------------------------------------------------------------------
def fingerprint(*arrays) -> str:
    """Content hash of a set of arrays: dtype, shape and bytes."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"<none>")
            continue
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _as_rng(seed_or_rng) -> np.random.Generator:
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


@dataclass
class ControlRealisation:
    """One realisation of one control, with its checks already run.

    kind        which control produced it
    data        the control realisation itself (same type as the source)
    invariants  {name: (value_source, value_control, abs_residual)} -- things
                the control CLAIMS to preserve, each verified numerically here
    destroyed   {name: (value_source, value_control)} -- things it claims to
                destroy, with the numbers, so "destroyed" is falsifiable
    meta        seed, options, source fingerprint
    """
    kind: str
    data: Any
    invariants: dict = field(default_factory=dict)
    destroyed: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def check(self, tol: float = 1e-10) -> None:
        """Raise if any claimed invariant moved by more than `tol`."""
        bad = {k: v for k, v in self.invariants.items() if not (v[2] <= tol)}
        if bad:
            lines = [f"      {k}: source {v[0]!r} control {v[1]!r} "
                     f"residual {v[2]:.3e} > tol {tol:.1e}"
                     for k, v in bad.items()]
            raise AssertionError(
                f"control '{self.kind}' broke {len(bad)} invariant(s) it "
                "claims to preserve:\n" + "\n".join(lines))

    def summary(self) -> dict:
        return {"kind": self.kind,
                "invariants": {k: {"source": _j(v[0]), "control": _j(v[1]),
                                   "residual": _j(v[2])}
                               for k, v in self.invariants.items()},
                "destroyed": {k: {"source": _j(v[0]), "control": _j(v[1])}
                              for k, v in self.destroyed.items()},
                "meta": {k: _j(v) for k, v in self.meta.items()}}


def _j(v):
    """JSON-safe scalar."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else str(f)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_j(x) for x in v.ravel()[:16]]
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    return v


# --------------------------------------------------------------------------
#  source models
# --------------------------------------------------------------------------
@dataclass
class ObjectPointSource:
    """Point observations grouped by object. The SPARC-shaped case.

    obj    (n,)  integer object id (galaxy, cluster, ...)
    coord  (n,d) position within the object (r, or x/y/z, or r/theta/phi)
    value  (n,)  the observed quantity the model is compared against
    sigma  (n,)  its measurement uncertainty
    model  (n,)  the baseline prediction residuals are taken against
    """
    obj: np.ndarray
    coord: np.ndarray
    value: np.ndarray
    sigma: np.ndarray
    model: np.ndarray
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.obj = np.asarray(self.obj, np.int64)
        self.coord = np.atleast_2d(np.asarray(self.coord, float))
        if self.coord.shape[0] != self.obj.size and self.coord.shape[1] == self.obj.size:
            self.coord = self.coord.T
        for k in ("value", "sigma", "model"):
            setattr(self, k, np.asarray(getattr(self, k), float))
        n = self.obj.size
        for k in ("value", "sigma", "model"):
            assert getattr(self, k).shape == (n,), f"{k} shape {getattr(self,k).shape} != ({n},)"
        assert self.coord.shape[0] == n, "coord rows != n points"
        assert np.all(self.sigma > 0), "non-positive sigma"

    @property
    def n(self):
        return self.obj.size

    def residual(self):
        return self.value - self.model

    def replace_value(self, v):
        return ObjectPointSource(self.obj, self.coord, v, self.sigma,
                                 self.model, self.name, dict(self.extra))

    def fingerprint(self):
        return fingerprint(self.obj, self.coord, self.value, self.sigma, self.model)


@dataclass
class FieldSource:
    """A resolved three-dimensional map on a regular Cartesian grid.

    rho     (nx,ny,nz) mass density, Msun/kpc^3
    dx      cell size, kpc (isotropic)
    origin  (3,) physical coordinate of index (0,0,0), kpc
    value / sigma / model / mask   optional observation maps on the SAME grid;
            `mask` selects the voxels that carry an observation
    obj     optional (nx,ny,nz) int map of which object each voxel belongs to
    """
    rho: np.ndarray
    dx: float
    origin: np.ndarray = None
    value: np.ndarray = None
    sigma: np.ndarray = None
    model: np.ndarray = None
    mask: np.ndarray = None
    obj: np.ndarray = None
    name: str = ""

    def __post_init__(self):
        self.rho = np.asarray(self.rho, float)
        assert self.rho.ndim == 3, "FieldSource.rho must be 3-D"
        self.dx = float(self.dx)
        if self.origin is None:
            self.origin = -0.5 * self.dx * (np.array(self.rho.shape, float) - 1.0)
        self.origin = np.asarray(self.origin, float)
        for k in ("value", "sigma", "model"):
            v = getattr(self, k)
            if v is not None:
                v = np.asarray(v, float)
                assert v.shape == self.rho.shape, f"{k} shape != rho shape"
                setattr(self, k, v)
        if self.mask is None:
            self.mask = np.ones(self.rho.shape, bool)
        self.mask = np.asarray(self.mask, bool)
        if self.obj is None:
            self.obj = np.zeros(self.rho.shape, np.int64)
        self.obj = np.asarray(self.obj, np.int64)

    @property
    def dV(self):
        return self.dx ** 3

    def coords(self):
        """(3, nx, ny, nz) physical coordinates of the cell centres."""
        ax = [self.origin[i] + self.dx * np.arange(self.rho.shape[i])
              for i in range(3)]
        return np.stack(np.meshgrid(*ax, indexing="ij"))

    def radius(self, centre=(0.0, 0.0, 0.0)):
        c = self.coords()
        centre = np.asarray(centre, float)
        return np.sqrt(sum((c[i] - centre[i]) ** 2 for i in range(3)))

    def total_mass(self):
        return float(self.rho.sum() * self.dV)

    def enclosed_mass(self, edges, centre=(0.0, 0.0, 0.0)):
        r = self.radius(centre).ravel()
        w = self.rho.ravel() * self.dV
        h, _ = np.histogram(r, bins=np.asarray(edges, float), weights=w)
        return np.cumsum(h)

    def shell_anisotropy(self, nbin=None, centre=(0.0, 0.0, 0.0)):
        """RMS fractional departure of rho from its own shell mean.

        EXACTLY zero for any purely radial field, on any grid. The multipole
        amplitudes are not: a cubic lattice has an ell = 4 anisotropy of its
        own, so `multipole(4)` of a perfectly spherical rho is nonzero and is
        measuring the grid, not the source. Use this for the
        resolved-versus-averaged comparison and `multipole` only with that
        caveat attached.
        """
        r = self.radius(centre).ravel()
        if nbin is None:
            nbin = max(8, int(round(r.max() / self.dx)))
        edges = np.linspace(0.0, r.max() * (1 + 1e-9), nbin + 1)
        k = np.clip(np.digitize(r, edges[1:-1]), 0, nbin - 1)
        m = self.rho.ravel()
        tot = np.bincount(k, weights=m, minlength=nbin)
        cnt = np.bincount(k, minlength=nbin).astype(float)
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)[k]
        good = mean > 0
        if not good.any():
            return 0.0
        return float(np.sqrt(np.mean(((m[good] - mean[good]) / mean[good]) ** 2)))

    def multipole(self, ell, centre=(0.0, 0.0, 0.0)):
        """|sum_i m_i r_i^ell P_ell(cos theta_i)| / sum_i m_i r_i^ell -- an
        axis-free amplitude of the ell-th moment about the z axis."""
        c = self.coords()
        centre = np.asarray(centre, float)
        z = c[2] - centre[2]
        r = self.radius(centre)
        m = self.rho * self.dV
        good = r > 0
        ct = np.zeros_like(r)
        ct[good] = z[good] / r[good]
        pl = {0: np.ones_like(ct), 2: 0.5 * (3 * ct ** 2 - 1.0),
              4: (35 * ct ** 4 - 30 * ct ** 2 + 3) / 8.0}[ell]
        num = float((m * r ** ell * pl).sum())
        den = float((m * r ** ell).sum())
        return abs(num) / den if den > 0 else 0.0

    def fingerprint(self):
        return fingerprint(self.rho, np.array([self.dx]), self.origin)


@dataclass
class ClusterSource:
    """Member galaxies of one or more clusters.

    pos     (N,3) or (N,2) member position RELATIVE TO THE CLUSTER CENTRE, Mpc
    mass    (N,)  member mass, Msun
    cid     (N,)  which cluster the member belongs to
    projected  True if `pos` is a sky-projected separation (then a 3-D angular
               scramble would change the observable R_proj and is refused)
    """
    pos: np.ndarray
    mass: np.ndarray
    cid: np.ndarray = None
    mass_err: np.ndarray = None
    projected: bool = False
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pos = np.atleast_2d(np.asarray(self.pos, float))
        self.mass = np.asarray(self.mass, float)
        N = self.mass.size
        assert self.pos.shape[0] == N, "pos rows != n members"
        assert self.pos.shape[1] in (2, 3), "pos must be (N,2) or (N,3)"
        if self.cid is None:
            self.cid = np.zeros(N, np.int64)
        self.cid = np.asarray(self.cid, np.int64)
        if self.mass_err is not None:
            self.mass_err = np.asarray(self.mass_err, float)
        assert np.all(self.mass >= 0), "negative member mass"

    @property
    def N(self):
        return self.mass.size

    @property
    def ndim(self):
        return self.pos.shape[1]

    def radii(self):
        return np.sqrt((self.pos ** 2).sum(axis=1))

    def radial_profile(self, nbin=12):
        """Mass in log-spaced clustercentric shells, per cluster, concatenated.
        This is the quantity control 2 must hold EXACTLY fixed."""
        r = self.radii()
        out = []
        for c in np.unique(self.cid):
            w = self.cid == c
            rr, mm = r[w], self.mass[w]
            pos = rr[rr > 0]
            lo = pos.min() * 0.999 if pos.size else 1e-6
            hi = rr.max() * 1.001 if rr.size else 1.0
            e = np.geomspace(lo, hi, nbin + 1)
            h, _ = np.histogram(rr, bins=e, weights=mm)
            out.append(h)
        return np.concatenate(out) if out else np.zeros(0)

    def network_energy(self):
        """W = sum_{i<j} m_i m_j / |r_i - r_j|, per cluster, summed.

        The simplest statistic that depends on the GEOMETRY of the well network
        at fixed radial mass profile. Controls 2, 3 and 4 must move it; control
        2 must do so while leaving `radial_profile` bit-identical.
        """
        tot = 0.0
        for c in np.unique(self.cid):
            w = np.flatnonzero(self.cid == c)
            if w.size < 2:
                continue
            p, m = self.pos[w], self.mass[w]
            d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            iu = np.triu_indices(w.size, 1)
            dd = np.maximum(d[iu], 1e-6)
            tot += float((m[:, None] * m[None, :])[iu].dot(1.0 / dd))
        return tot

    def replace(self, pos=None, mass=None, mass_err=None):
        return ClusterSource(self.pos if pos is None else pos,
                             self.mass if mass is None else mass,
                             self.cid,
                             self.mass_err if mass_err is None else mass_err,
                             self.projected, self.name, dict(self.extra))

    def fingerprint(self):
        return fingerprint(self.pos, self.mass, self.cid)


# --------------------------------------------------------------------------
#  block permutation, the primitive under controls 1-3
# --------------------------------------------------------------------------
def permute_within_blocks(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Index map that permutes positions inside each block and nowhere else.

    Returns `src` such that x[src] is x permuted within blocks. Blocks of size
    1 are identity, which is the honest behaviour: a control cannot destroy
    structure that a single point does not carry.

    Vectorised: `lexsort((key, block))` orders the indices by block and then by
    a random key, which is a uniform permutation inside each block, and the
    stable argsort of the block labels gives the destination slots. O(n log n)
    with no Python loop, because a control that is slower than the search is a
    control that gets skipped.
    """
    block = np.asarray(block)
    n = block.size
    key = rng.random(n)
    order = np.lexsort((key, block))
    base = np.argsort(block, kind="stable")
    src = np.empty(n, np.int64)
    src[base] = order
    return src


def _block_codes(block):
    """Dense 0..K-1 codes for arbitrary block labels, plus K."""
    _, inv = np.unique(np.asarray(block), return_inverse=True)
    inv = np.asarray(inv).ravel()
    return inv, int(inv.max()) + 1 if inv.size else 0


def _block_mean_expand(v, block):
    """Per-element mean of `v` over its own block."""
    inv, K = _block_codes(block)
    s = np.bincount(inv, weights=np.asarray(v, float), minlength=K)
    c = np.bincount(inv, minlength=K)
    return (s / np.maximum(c, 1))[inv]


def _block_means(v, block):
    """One mean per block, in sorted block-label order."""
    inv, K = _block_codes(block)
    s = np.bincount(inv, weights=np.asarray(v, float), minlength=K)
    c = np.bincount(inv, minlength=K)
    return s / np.maximum(c, 1)


def block_sizes(block) -> dict:
    _, cnt = np.unique(np.asarray(block), return_counts=True)
    return {"n_blocks": int(cnt.size), "min": int(cnt.min()),
            "median": float(np.median(cnt)), "max": int(cnt.max()),
            "n_singleton": int((cnt == 1).sum())}

# ==========================================================================
#  CONTROL 1 -- RESIDUAL NULLS
# ==========================================================================
def _block_from_spec(obj, radius, spec, nshell):
    """Build the permutation blocks from a spec.

    "object"            -> one block per object; destroys ALL spatial structure
                           inside the object, keeps the object-level offset.
                           This is the SPARC `perm_g` of Run J.
    "object+shell"      -> one block per (object, radial shell); keeps the
                           RADIAL profile of the residual exactly and destroys
                           only the ANGULAR structure. This is the field
                           generalisation, and the correct null for a
                           well-network claim, which is a claim about geometry
                           at fixed radial profile.
    "global"            -> one block; destroys the object offsets too. Weaker,
                           kept only because Run J's `perm` used it.
    array               -> explicit block labels.
    """
    if isinstance(spec, np.ndarray):
        return spec
    if spec == "global":
        return np.zeros(obj.size, np.int64)
    if spec == "object":
        return obj.copy()
    if spec == "object+shell":
        if radius is None:
            raise ValueError("block='object+shell' needs a radial coordinate")
        b = np.zeros(obj.size, np.int64)
        for o in np.unique(obj):
            w = np.flatnonzero(obj == o)
            r = radius[w]
            q = np.quantile(r, np.linspace(0, 1, nshell + 1))
            q[0] -= 1e-12
            q = np.maximum.accumulate(q)
            q[1:] = np.where(np.diff(q) <= 0, q[:-1] + 1e-12, q[1:])
            b[w] = np.digitize(r, q[1:-1])
        return obj.astype(np.int64) * (nshell + 1) + b
    raise ValueError(f"unknown block spec {spec!r}")


def _residual_null_core(resid, sigma, block, rng, standardise):
    """Permute residuals inside blocks, preserving sigma AND the block offset."""
    src = permute_within_blocks(block, rng)
    if not standardise:
        return resid[src], src
    mu = _block_mean_expand(resid, block)
    z = (resid - mu) / sigma              # standardise: the uncertainty MAP
    new_eps = z[src] * sigma              # stays attached to its position
    new_eps = new_eps - _block_mean_expand(new_eps, block)   # exact offset
    return mu + new_eps, src


def _within_block_corr(y, x, block):
    """Size-weighted mean Pearson r(y, x) inside blocks -- a scalar summary of
    'the proposed spatial signal' that the null is supposed to destroy."""
    num = den = 0.0
    for bb in np.unique(block):
        w = block == bb
        if w.sum() < 3:
            continue
        yy, xx = y[w], x[w]
        sy, sx = yy.std(), xx.std()
        if sy <= 0 or sx <= 0:
            continue
        r = float(((yy - yy.mean()) * (xx - xx.mean())).mean() / (sy * sx))
        num += r * w.sum()
        den += w.sum()
    return num / den if den else float("nan")


def residual_null(source, seed=0, *, block="object+shell", nshell=4,
                  standardise=True, radius_from=None):
    """CONTROL 1. Preserve the uncertainties and the object offsets; destroy
    the proposed spatial signal.

    Accepts an ObjectPointSource (rotation-curve shaped) or a FieldSource
    (a resolved 3-D map). In both cases the returned realisation has:

      * sigma  bit-identical, at the SAME position -- the measurement error
        map is not shuffled, only the residual is;
      * the mean residual inside every block bit-identical to ~1e-15, so
        distance, inclination and M/L offsets survive: they are real, and they
        are not new physics;
      * the residual's dependence on position inside each block destroyed.

    `standardise=True` permutes sigma-standardised residuals and rescales, so a
    heteroscedastic field does not acquire a fake noise map. `standardise=False`
    reproduces Run J's `perm_g` byte for byte.
    """
    rng = _as_rng(seed)
    if isinstance(source, FieldSource):
        return _residual_null_field(source, rng, block, nshell, standardise)
    if not isinstance(source, ObjectPointSource):
        raise TypeError(f"residual_null needs a source model, got {type(source)}")
    s = source
    r = (np.linalg.norm(s.coord, axis=1) if radius_from is None
         else np.asarray(radius_from, float))
    blk = _block_from_spec(s.obj, r, block, nshell)
    resid = s.residual()
    new_resid, _ = _residual_null_core(resid, s.sigma, blk, rng, standardise)
    out = s.replace_value(s.model + new_resid)
    bm0, bm1 = _block_means(resid, blk), _block_means(new_resid, blk)
    inv = {
        "block_mean_residual": (float(np.abs(bm0).max()), float(np.abs(bm1).max()),
                                float(np.abs(bm0 - bm1).max())),
        "sigma_map": (fingerprint(s.sigma)[:12], fingerprint(out.sigma)[:12],
                      float(np.abs(s.sigma - out.sigma).max())),
        "model": (fingerprint(s.model)[:12], fingerprint(out.model)[:12],
                  float(np.abs(s.model - out.model).max())),
        "coordinates": (fingerprint(s.coord)[:12], fingerprint(out.coord)[:12],
                        float(np.abs(s.coord - out.coord).max())),
        "n_points": (s.n, out.n, abs(s.n - out.n)),
    }
    if not standardise:
        inv["within_block_residual_multiset"] = (
            fingerprint(np.sort(resid[np.argsort(blk, kind="stable")]))[:12],
            fingerprint(np.sort(new_resid[np.argsort(blk, kind="stable")]))[:12],
            float(np.abs(np.sort(resid) - np.sort(new_resid)).max()))
    dest = {"resid_rms": (float(np.sqrt((resid ** 2).mean())),
                          float(np.sqrt((new_resid ** 2).mean())))}
    probes = {"radius": r}
    for c in range(s.coord.shape[1]):
        probes[f"coord{c}"] = s.coord[:, c]
    for pn, pv in probes.items():
        a = _within_block_corr(resid, pv, blk)
        if np.isfinite(a):
            dest[f"within_block_corr_resid_{pn}"] = (
                a, _within_block_corr(new_resid, pv, blk))
    return ControlRealisation("residual_null", out, inv, dest,
                              {"seed": _seed_of(seed), "block": str(block),
                               "nshell": nshell, "standardise": standardise,
                               "blocks": block_sizes(blk),
                               "source_fingerprint": s.fingerprint()[:16]})


def residual_null_batch(source: ObjectPointSource, seed, B: int, *,
                        block="object+shell", nshell=4, standardise=True,
                        radius_from=None):
    """B independent realisations of CONTROL 1, sharing one block structure.

    Returns (values (n, B), ControlRealisation) where the realisation carries
    the WORST-CASE invariant residual over all B draws, not a spot check on
    one of them. The batch path exists because a control loop that costs more
    than the search it polices is a control loop that gets shortened.
    """
    rng = _as_rng(seed)
    s = source
    r = (np.linalg.norm(s.coord, axis=1) if radius_from is None
         else np.asarray(radius_from, float))
    blk = _block_from_spec(s.obj, r, block, nshell)
    inv_c, K = _block_codes(blk)
    cnt = np.bincount(inv_c, minlength=K)

    def bm(v):
        return (np.bincount(inv_c, weights=v, minlength=K)
                / np.maximum(cnt, 1))
    resid = s.residual()
    mu_b = bm(resid)
    mu = mu_b[inv_c]
    z = (resid - mu) / s.sigma
    base = np.argsort(blk, kind="stable")
    out = np.empty((s.n, B))
    worst = 0.0
    for b in range(B):
        key = rng.random(s.n)
        order = np.lexsort((key, blk))
        src = np.empty(s.n, np.int64)
        src[base] = order
        if standardise:
            ne = z[src] * s.sigma
            ne = ne - bm(ne)[inv_c]
            nr = mu + ne
        else:
            nr = resid[src]
        out[:, b] = s.model + nr
        worst = max(worst, float(np.abs(bm(nr) - mu_b).max()))
    rec = ControlRealisation(
        "residual_null[batch]", out,
        {"block_mean_residual": (float(np.abs(mu_b).max()),
                                 float(np.abs(mu_b).max()), worst),
         "sigma_map": ("unchanged", "unchanged", 0.0)},
        {"resid_rms": (float(np.sqrt((resid ** 2).mean())),
                       float(np.sqrt(((out - s.model[:, None]) ** 2).mean())))},
        {"seed": _seed_of(seed), "B": B, "block": str(block),
         "standardise": standardise, "blocks": block_sizes(blk),
         "source_fingerprint": s.fingerprint()[:16]})
    return out, rec


def _residual_null_field(s: FieldSource, rng, block, nshell, standardise):
    if s.value is None or s.sigma is None or s.model is None:
        raise ValueError("FieldSource needs value, sigma and model maps for a "
                         "residual null")
    m = s.mask
    idx = np.flatnonzero(m.ravel())
    obj = s.obj.ravel()[idx]
    r = s.radius().ravel()[idx]
    blk = _block_from_spec(obj, r, block, nshell)
    resid = (s.value - s.model).ravel()[idx]
    sig = s.sigma.ravel()[idx]
    new_resid, _ = _residual_null_core(resid, sig, blk, rng, standardise)
    newval = s.value.copy()
    flat = newval.ravel()
    flat[idx] = s.model.ravel()[idx] + new_resid
    out = FieldSource(s.rho, s.dx, s.origin, newval, s.sigma, s.model,
                      s.mask, s.obj, s.name)
    bm0, bm1 = _block_means(resid, blk), _block_means(new_resid, blk)
    inv = {
        "block_mean_residual": (float(np.abs(bm0).max()), float(np.abs(bm1).max()),
                                float(np.abs(bm0 - bm1).max())),
        "sigma_map": (fingerprint(s.sigma)[:12], fingerprint(out.sigma)[:12], 0.0),
        "rho_map": (fingerprint(s.rho)[:12], fingerprint(out.rho)[:12], 0.0),
        "mask_voxels": (int(m.sum()), int(out.mask.sum()),
                        abs(int(m.sum()) - int(out.mask.sum()))),
    }
    dest = {"within_block_corr_resid_radius":
            (_within_block_corr(resid, r, blk),
             _within_block_corr(new_resid, r, blk)),
            "angular_rms_of_shell_mean_resid":
            (_angular_scatter(resid, r, s.coords().reshape(3, -1)[:, idx]),
             _angular_scatter(new_resid, r, s.coords().reshape(3, -1)[:, idx]))}
    return ControlRealisation("residual_null[field]", out, inv, dest,
                              {"seed": "gen", "block": str(block),
                               "nshell": nshell, "standardise": standardise,
                               "blocks": block_sizes(blk),
                               "source_fingerprint": s.fingerprint()[:16]})


def _angular_scatter(v, r, xyz, nshell=8, nphi=4, ncos=4):
    """RMS across angular sectors of the sector-mean of v, averaged over radial
    shells. Near zero for an angularly featureless residual field.

    Sectors are binned in BOTH azimuth and cos(theta): a quadrupolar pattern
    (which is what an anisotropic law produces) is invisible to azimuth-only
    sectors, and a control that cannot see the structure it destroys is not
    evidence that the structure was destroyed."""
    phi = np.arctan2(xyz[1], xyz[0])
    ct = np.divide(xyz[2], np.maximum(r, 1e-12))
    sec = (np.digitize(phi, np.linspace(-np.pi, np.pi, nphi + 1)[1:-1]) * ncos
           + np.digitize(ct, np.linspace(-1.0, 1.0, ncos + 1)[1:-1]))
    q = np.quantile(r, np.linspace(0, 1, nshell + 1))[1:-1]
    sh = np.digitize(r, q)
    vals = []
    for a in np.unique(sh):
        w = sh == a
        means = [v[w & (sec == t)].mean() for t in np.unique(sec)
                 if (w & (sec == t)).sum() >= 3]
        if len(means) >= 3:
            vals.append(np.std(means))
    return float(np.mean(vals)) if vals else float("nan")


def _seed_of(seed):
    return seed if isinstance(seed, (int, np.integer)) else "generator"


# ==========================================================================
#  CONTROL 2 -- POSITION-SCRAMBLED CLUSTERS
# ==========================================================================
def position_scramble(source: ClusterSource, seed=0, *, footprint=None,
                      max_tries=64):
    """CONTROL 2. Hold every member's mass AND its clustercentric radius
    exactly fixed; randomise its angular position.

    This is the key control for the well-network hypothesis. The radial mass
    profile -- which is what every spherically-averaged cluster measurement
    actually constrains -- is held bit-identical, so any statistic that only
    sees M(<r) CANNOT move. Whatever does move is geometry.

    `footprint(pos)->bool` optionally rejects positions outside the survey
    mask. Members for which no accepted direction was found in `max_tries`
    keep their original position; the count is reported, not hidden, because
    those members are a residual of the real signal left inside the control.
    """
    rng = _as_rng(seed)
    s = source
    if s.projected and s.ndim == 3:
        raise ValueError(
            "positions are flagged `projected` but are 3-D: an isotropic "
            "scramble would change the observed projected radius. Supply "
            "2-D projected positions, or set projected=False for true 3-D.")
    r = s.radii()
    if s.ndim == 3:
        u = rng.normal(size=(s.N, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
    else:
        a = rng.uniform(0, 2 * np.pi, s.N)
        u = np.stack([np.cos(a), np.sin(a)], axis=1)
    new = r[:, None] * u
    n_fallback = 0
    accept_tries = np.ones(s.N, np.int64)
    if footprint is not None:
        ok = np.asarray(footprint(new), bool)
        for t in range(2, max_tries + 1):
            if ok.all():
                break
            bad = ~ok
            if s.ndim == 3:
                v = rng.normal(size=(int(bad.sum()), 3))
                v /= np.linalg.norm(v, axis=1, keepdims=True)
            else:
                a = rng.uniform(0, 2 * np.pi, int(bad.sum()))
                v = np.stack([np.cos(a), np.sin(a)], axis=1)
            cand = r[bad][:, None] * v
            good = np.asarray(footprint(cand), bool)
            sel = np.flatnonzero(bad)
            new[sel[good]] = cand[good]
            accept_tries[sel] = t
            ok[sel[good]] = True
        n_fallback = int((~ok).sum())
        new[~ok] = s.pos[~ok]
    out = s.replace(pos=new)
    pos_r = r > 0
    rel = (float(np.abs(out.radii()[pos_r] / r[pos_r] - 1).max())
           if pos_r.any() else 0.0)
    inv = {
        "member_radii": (float(r.max()), float(out.radii().max()),
                         float(np.abs(r - out.radii()).max())),
        "member_radii_relative": (0.0, 0.0, rel),
        "member_masses": (fingerprint(s.mass)[:12], fingerprint(out.mass)[:12],
                          float(np.abs(s.mass - out.mass).max())),
        "radial_mass_profile": (fingerprint(s.radial_profile())[:12],
                                fingerprint(out.radial_profile())[:12],
                                float(np.abs(s.radial_profile()
                                             - out.radial_profile()).max())),
        "cluster_membership": (fingerprint(s.cid)[:12], fingerprint(out.cid)[:12],
                               float(np.abs(s.cid - out.cid).max())),
    }
    dest = {"network_energy": (s.network_energy(), out.network_energy()),
            "mean_pair_separation": (_mean_pair_sep(s), _mean_pair_sep(out))}
    return ControlRealisation("position_scramble", out, inv, dest,
                              {"seed": _seed_of(seed), "ndim": s.ndim,
                               "projected": s.projected,
                               "footprint": footprint is not None,
                               "n_members": s.N,
                               "n_fallback_kept_original": n_fallback,
                               "mean_footprint_tries": float(accept_tries.mean()),
                               "source_fingerprint": s.fingerprint()[:16],
                               "meanfield_network_energy":
                                   meanfield_network_energy(s)})


def _mean_pair_sep(s: ClusterSource):
    tot, cnt = 0.0, 0
    for c in np.unique(s.cid):
        w = np.flatnonzero(s.cid == c)
        if w.size < 2:
            continue
        p = s.pos[w]
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(w.size, 1)
        tot += float(d[iu].sum()); cnt += iu[0].size
    return tot / cnt if cnt else float("nan")


def meanfield_network_energy(s: ClusterSource):
    """E[W] under control 2, in closed form.

    For independent isotropic directions at fixed radii, <1/|r_i - r_j|> is
    exactly 1/max(r_i, r_j) (only the ell = 0 term of the multipole expansion
    survives the angular average). So the expected network energy of a
    position-scrambled cluster is sum_{i<j} m_i m_j / r_>, with no simulation.
    Control 4 (the angular average) must reproduce this same number, and
    control 2 averaged over realisations must converge to it. That identity is
    the cross-check that controls 2 and 4 are the same null.

    Valid for 3-D positions. For 2-D the angular average of 1/|r_i-r_j| is an
    elliptic integral, not 1/r_>, and this returns NaN rather than a wrong
    number.
    """
    if s.ndim != 3:
        return float("nan")
    r = s.radii()
    tot = 0.0
    for c in np.unique(s.cid):
        w = np.flatnonzero(s.cid == c)
        if w.size < 2:
            continue
        rr, mm = r[w], s.mass[w]
        iu = np.triu_indices(w.size, 1)
        rmax = np.maximum(rr[iu[0]], rr[iu[1]])
        tot += float((mm[iu[0]] * mm[iu[1]] / np.maximum(rmax, 1e-12)).sum())
    return tot


# ==========================================================================
#  CONTROL 3 -- MASS-SCRAMBLED CLUSTERS
# ==========================================================================
def mass_scramble(source: ClusterSource, seed=0, *, within="cluster"):
    """CONTROL 3. Geometry preserved exactly; masses permuted among positions.

    Everything attached to the MASS travels with it (the mass error, and any
    per-member array listed in `source.extra['mass_attached']`); everything
    attached to the POSITION stays put. Getting that wrong is how a control
    quietly stops being a control.
    """
    rng = _as_rng(seed)
    s = source
    blk = s.cid if within == "cluster" else np.zeros(s.N, np.int64)
    src = permute_within_blocks(blk, rng)
    new_mass = s.mass[src]
    new_err = None if s.mass_err is None else s.mass_err[src]
    out = s.replace(mass=new_mass, mass_err=new_err)
    for k in s.extra.get("mass_attached", []):
        out.extra[k] = np.asarray(s.extra[k])[src]
    for k, v in s.extra.items():
        if k not in out.extra:
            out.extra[k] = v
    srt = lambda a, b: float(np.abs(np.sort(a) - np.sort(b)).max())
    inv = {
        "positions": (fingerprint(s.pos)[:12], fingerprint(out.pos)[:12],
                      float(np.abs(s.pos - out.pos).max())),
        "member_radii": (float(s.radii().max()), float(out.radii().max()),
                         float(np.abs(s.radii() - out.radii()).max())),
        "mass_multiset_per_cluster_relative":
            ("sorted", "sorted",
             max([srt(s.mass[s.cid == c], out.mass[out.cid == c])
                  / max(float(s.mass[s.cid == c].max()), 1e-30)
                  for c in np.unique(s.cid)] or [0.0])),
        "total_mass_per_cluster_relative":
            (float(s.mass.sum()), float(out.mass.sum()),
             max([abs(float(s.mass[s.cid == c].sum()
                            - out.mass[out.cid == c].sum()))
                  / max(float(s.mass[s.cid == c].sum()), 1e-30)
                  for c in np.unique(s.cid)] or [0.0])),
    }
    r = s.radii()
    dest = {"network_energy": (s.network_energy(), out.network_energy()),
            "mass_radius_corr": (_corr(np.log(np.maximum(s.mass, 1)), r),
                                 _corr(np.log(np.maximum(out.mass, 1)), r)),
            "radial_mass_profile_l1":
                (0.0, float(np.abs(s.radial_profile()
                                   - out.radial_profile()).sum()))}
    return ControlRealisation("mass_scramble", out, inv, dest,
                              {"seed": _seed_of(seed), "within": within,
                               "blocks": block_sizes(blk),
                               "source_fingerprint": s.fingerprint()[:16]})


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ==========================================================================
#  CONTROL 4 -- SMOOTHED SOURCE
# ==========================================================================
@dataclass
class SmoothedCluster:
    """A spherically symmetric source with the SAME M(<r) as the original."""
    edges: np.ndarray             # (ncl, nbin+1) shell edges, Mpc
    shell_mass: np.ndarray        # (ncl, nbin)
    cids: np.ndarray

    def M_enclosed(self, r, c=0):
        i = int(np.flatnonzero(self.cids == c)[0])
        e, m = self.edges[i], self.shell_mass[i]
        cum = np.r_[0.0, np.cumsum(m)]
        return np.interp(np.asarray(r, float), e, cum)

    def density(self, r, c=0):
        i = int(np.flatnonzero(self.cids == c)[0])
        e, m = self.edges[i], self.shell_mass[i]
        vol = 4.0 / 3.0 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
        rho = m / np.maximum(vol, 1e-30)
        k = np.clip(np.digitize(np.asarray(r, float), e[1:-1]), 0, rho.size - 1)
        return rho[k]

    def network_energy(self, source: ClusterSource):
        """W of the smoothed source, evaluated with the members' own radii:
        sum_{i<j} m_i m_j / r_>. Identical to meanfield_network_energy."""
        return meanfield_network_energy(source)


def smoothed_source(source, seed=None, *, nbin=None, centre=(0.0, 0.0, 0.0)):
    """CONTROL 4. Same radial mass profile, angularly averaged.

    FieldSource  -> FieldSource whose rho is the shell mean. Mass inside every
                    shell, hence M(<r) at every shell edge, is preserved
                    EXACTLY (the grid is regular, so the shell mean is the
                    mass-conserving average). All ell > 0 moments go to the
                    residual set by the finite shell width.
    ClusterSource-> SmoothedCluster: the members' mass redistributed uniformly
                    over their own shells.

    This control has no randomness. It is the ensemble mean of control 2, and
    `meanfield_network_energy` proves that in closed form.
    """
    if isinstance(source, FieldSource):
        return _smoothed_field(source, nbin, centre)
    if isinstance(source, ClusterSource):
        return _smoothed_cluster(source, nbin)
    raise TypeError(f"smoothed_source: unsupported source {type(source)}")


def _smoothed_field(s: FieldSource, nbin, centre):
    r = s.radius(centre)
    if nbin is None:
        nbin = max(8, int(round(r.max() / s.dx)))
    edges = np.linspace(0.0, r.max() * (1 + 1e-9), nbin + 1)
    k = np.clip(np.digitize(r.ravel(), edges[1:-1]), 0, nbin - 1)
    m = s.rho.ravel()
    tot = np.bincount(k, weights=m, minlength=nbin)
    cnt = np.bincount(k, minlength=nbin).astype(float)
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    rho_s = mean[k].reshape(s.rho.shape)
    out = FieldSource(rho_s, s.dx, s.origin, s.value, s.sigma, s.model,
                      s.mask, s.obj, s.name)
    M0, M1 = s.enclosed_mass(edges, centre), out.enclosed_mass(edges, centre)
    rel = float(np.abs(M1 - M0).max() / max(M0[-1], 1e-30))
    inv = {"total_mass": (s.total_mass(), out.total_mass(),
                          abs(s.total_mass() - out.total_mass()) / max(s.total_mass(), 1e-30)),
           "M_enclosed_at_shell_edges": (float(M0[-1]), float(M1[-1]), rel),
           # the brief's standing failure list includes non-monotonic M(r) in
           # lensing deprojection. Here it cannot happen -- M(<r) is a
           # cumulative sum of non-negative shell masses -- and the check is
           # run anyway, because "cannot happen" is what every such bug says.
           "M_enclosed_monotone": (0.0, 0.0,
                                   float(max(0.0, -np.min(np.diff(M1)) /
                                             max(M1[-1], 1e-30)))),
           "monopole": (s.multipole(0, centre), out.multipole(0, centre),
                        abs(s.multipole(0, centre) - out.multipole(0, centre)))}
    dest = {"shell_anisotropy": (s.shell_anisotropy(nbin, centre),
                                 out.shell_anisotropy(nbin, centre)),
            "quadrupole_l2": (s.multipole(2, centre), out.multipole(2, centre)),
            "hexadecapole_l4": (s.multipole(4, centre), out.multipole(4, centre))}
    return ControlRealisation("smoothed_source[field]", out, inv, dest,
                              {"nbin": nbin, "dx": s.dx,
                               "shell_width_over_dx": float(
                                   (edges[1] - edges[0]) / s.dx),
                               # a purely radial field on a CUBIC grid still
                               # has an ell=4 moment: it is the lattice, not
                               # the source. This is that floor, measured.
                               "lattice_l4_floor": out.multipole(4, centre),
                               "lattice_l2_floor": out.multipole(2, centre),
                               "source_fingerprint": s.fingerprint()[:16]})


def _smoothed_cluster(s: ClusterSource, nbin):
    if nbin is None:
        nbin = 12
    r = s.radii()
    cs = np.unique(s.cid)
    E, M = [], []
    for c in cs:
        w = s.cid == c
        rr, mm = r[w], s.mass[w]
        pos = rr[rr > 0]
        lo = (pos.min() * 0.999 if pos.size else 1e-6)
        hi = rr.max() * 1.001
        e = np.r_[0.0, np.geomspace(lo, hi, nbin)]
        h, _ = np.histogram(rr, bins=e, weights=mm)
        E.append(e); M.append(h)
    sm = SmoothedCluster(np.array(E), np.array(M), cs)
    Mtot0 = float(s.mass.sum())
    Mtot1 = float(sm.shell_mass.sum())
    # M(<r) preserved at every shell edge, per cluster
    resid = 0.0
    for i, c in enumerate(cs):
        w = s.cid == c
        for e in sm.edges[i]:
            direct = float(s.mass[w][r[w] <= e].sum())
            resid = max(resid, abs(direct - float(sm.M_enclosed(e, c))))
    mono = 0.0
    for i, c in enumerate(cs):
        cum = np.r_[0.0, np.cumsum(sm.shell_mass[i])]
        mono = max(mono, float(max(0.0, -np.min(np.diff(cum)))
                               / max(cum[-1], 1e-30)))
    inv = {"total_mass": (Mtot0, Mtot1, abs(Mtot0 - Mtot1) / max(Mtot0, 1e-30)),
           "M_enclosed_at_shell_edges": (Mtot0, Mtot1,
                                         resid / max(Mtot0, 1e-30)),
           "M_enclosed_monotone": (0.0, 0.0, mono)}
    dest = {"network_energy": (s.network_energy(),
                               sm.network_energy(s)),
            "angular_information": ("member directions", "none (spherical)")}
    return ControlRealisation("smoothed_source[cluster]", sm, inv, dest,
                              {"nbin": nbin, "n_clusters": int(cs.size),
                               "source_fingerprint": s.fingerprint()[:16]})

# ==========================================================================
#  CONTROL 6 -- PARAMETER-SENSITIVITY TESTS
# ==========================================================================
class ParameterBlindError(AssertionError):
    """A headline statistic did not move when its own parameter moved."""


def assert_parameter_responsive(stat: Callable[[float], float],
                                thetas: Sequence[float], *, name: str,
                                min_rel_spread: float = 1e-6,
                                min_distinct: int = 2,
                                require_monotone: bool = False,
                                raise_on_fail: bool = True,
                                verbose: bool = True) -> dict:
    """CONTROL 6. Verify numerically that dS/dtheta != 0 over the tested range.

    This exists because a rank statistic in this programme was BIT-IDENTICAL
    across three decades of the coupling it was supposed to measure: the test
    was mathematically blind to its own parameter, and reported a clean null
    that carried no information at all. A monotone transform of one axis leaves
    every rank statistic exactly invariant, so "the correlation did not change
    with kappa" was a theorem about ranks, not a fact about gravity.

    It raises. It is not a diagnostic that can be read past: a statistic that
    cannot see its parameter must stop the run.
    """
    th = np.asarray(list(thetas), float)
    S = np.array([float(stat(t)) for t in th])
    finite = np.isfinite(S)
    uniq = np.unique(S[finite])
    scale = max(float(np.nanmedian(np.abs(S[finite]))) if finite.any() else 0.0,
                float(np.nanstd(S[finite])) if finite.any() else 0.0, 1e-300)
    spread = float(np.nanmax(S[finite]) - np.nanmin(S[finite])) if finite.any() else float("nan")
    rel = spread / scale
    d = np.diff(S)
    mono = bool(np.all(d >= -1e-15) or np.all(d <= 1e-15))
    rec = {"name": name, "n_theta": int(th.size),
           "theta_min": float(th.min()), "theta_max": float(th.max()),
           "theta_decades": float(np.log10(th.max() / th.min()))
           if th.min() > 0 else None,
           "S": [float(v) for v in S],
           "n_distinct": int(uniq.size), "spread": spread,
           "relative_spread": rel, "monotone": mono,
           "n_nonfinite": int((~finite).sum()),
           "min_rel_spread_required": min_rel_spread, "passed": True}
    fail = []
    if uniq.size < min_distinct:
        fail.append(f"only {uniq.size} distinct value(s) over "
                    f"{th.size} settings of theta -- the statistic is "
                    f"INVARIANT under its own parameter")
    if not np.isfinite(rel) or rel < min_rel_spread:
        fail.append(f"relative spread {rel:.3e} < {min_rel_spread:.1e}")
    if require_monotone and not mono:
        fail.append("S(theta) is not monotone over the tested range")
    rec["passed"] = not fail
    rec["failures"] = fail
    if verbose:
        print(f"   responsiveness: {name}")
        for t, v in zip(th, S):
            print(f"      theta = {t:<12.6g}  S = {v!r}")
        print(f"      distinct {uniq.size}/{th.size}   spread {spread:.6g}   "
              f"relative {rel:.3e}   {'PASS' if rec['passed'] else 'FAIL'}")
    if fail and raise_on_fail:
        raise ParameterBlindError(
            f"statistic '{name}' is blind to its own parameter over "
            f"[{th.min():g}, {th.max():g}]: " + "; ".join(fail) +
            f"\n   values: {list(S)}")
    return rec


def responsiveness_suite(stats: dict, thetas, **kw) -> dict:
    """Run control 6 over a whole family of headline statistics.

    Collects every failure and raises ONCE at the end naming all of them, so a
    report cannot quietly contain one blind statistic among ten good ones.
    """
    out, bad = {}, []
    for nm, fn in stats.items():
        try:
            out[nm] = assert_parameter_responsive(fn, thetas, name=nm,
                                                  raise_on_fail=False, **kw)
        except Exception as exc:                      # pragma: no cover
            out[nm] = {"name": nm, "passed": False, "failures": [repr(exc)]}
        if not out[nm]["passed"]:
            bad.append(nm)
    if bad:
        raise ParameterBlindError(
            f"{len(bad)} of {len(stats)} headline statistics are blind to "
            f"their own parameter: {bad}")
    return out


# ==========================================================================
#  CONTROL 7 -- EXCHANGEABILITY TESTS
# ==========================================================================
class ExchangeabilityError(AssertionError):
    """The true and the control arm did not go through the same pipeline."""


_TRACE_STACK: list = []

#: Every entry point that can silently make the two arms different. Grouped by
#: the five operation classes the brief names.
def _patch_targets():
    T = []
    for nm in ("interp",):                                    # interpolation
        T.append((np, nm, "interp"))
    for nm in ("convolve", "correlate"):                       # smoothing
        T.append((np, nm, "smooth"))
    for nm in ("where", "clip", "compress", "nan_to_num", "isfinite",
               "isnan", "extract", "putmask"):                 # masking
        T.append((np, nm, "mask"))
    for nm in ("histogram", "histogram2d", "histogramdd", "digitize",
               "searchsorted", "percentile", "quantile", "bincount",
               "average"):                                     # sampling/binning
        T.append((np, nm, "sample"))
    for nm in ("permutation", "shuffle", "normal", "random", "choice",
               "randint"):                                     # randomness
        T.append((np.random, nm, "RANDOM"))
    if _ndimage is not None:
        for nm in ("gaussian_filter", "gaussian_filter1d", "uniform_filter",
                   "median_filter", "convolve", "correlate"):
            T.append((_ndimage, nm, "smooth"))
        for nm in ("map_coordinates", "zoom", "shift", "rotate"):
            T.append((_ndimage, nm, "interp"))
    return [(m, n, c) for m, n, c in T if hasattr(m, n)]


#: bound BEFORE any patching, so the tracer's own bookkeeping never re-enters
#: the tracer. Without this, recording a call to np.isfinite records a call to
#: np.isfinite.
_ISFINITE = np.isfinite


#: arrays at or below this size get a VALUE fingerprint as well as a shape.
#: Smoothing kernels, bin edges, aperture lists and window functions live
#: here, and a control that misses "the two arms were smoothed with different
#: kernels" is not a control. Above it, only the structure is recorded: the
#: data itself is supposed to differ between the arms.
_SMALL_ARRAY = 64


def _argsig(a):
    """Structural signature of one argument. Values only for SMALL arrays."""
    if isinstance(a, np.ndarray):
        nf = int((~_ISFINITE(a)).sum()) if a.dtype.kind == "f" else 0
        if a.size <= _SMALL_ARRAY:
            h = hashlib.sha256(np.ascontiguousarray(
                a).tobytes()).hexdigest()[:12]
            return ("smallarray", a.shape, str(a.dtype), nf, h)
        return ("array", a.shape, str(a.dtype), nf)
    if isinstance(a, (bool, str, type(None), np.bool_)):
        return ("exact", a)
    if isinstance(a, (int, np.integer)):
        return ("exact", int(a))
    if isinstance(a, (float, np.floating)):
        return ("float", float(a))
    if isinstance(a, (list, tuple)):
        return ("seq", len(a), tuple(_argsig(x) for x in a[:8]))
    if isinstance(a, slice):
        return ("exact", repr(a))
    return ("obj", type(a).__name__)


_IN_WRAPPER = [False]


def _wrap(label, cls, fn):
    def inner(*args, **kwargs):
        out = fn(*args, **kwargs)
        if _TRACE_STACK and not _IN_WRAPPER[0]:
            _IN_WRAPPER[0] = True
            try:
                _TRACE_STACK[-1].append({
                    "op": label, "class": cls,
                    "args": tuple(_argsig(a) for a in args),
                    "kwargs": tuple(sorted((k, _argsig(v))
                                           for k, v in kwargs.items())),
                    "out": _argsig(out) if not isinstance(out, tuple)
                           else ("tuple", tuple(_argsig(o) for o in out))})
            finally:
                _IN_WRAPPER[0] = False
        return out
    inner.__name__ = getattr(fn, "__name__", label)
    inner.__doc__ = getattr(fn, "__doc__", None)
    inner._controls_wrapped = True
    return inner


@contextlib.contextmanager
def trace_ops(extra_targets: Iterable = ()):
    """Record every interpolation / smoothing / masking / sampling call made.

    Patches the numpy and scipy entry points for the duration. This is real
    instrumentation, not a convention: a pipeline that quietly smooths one arm
    and not the other shows up in the trace whether or not it says so.

    Known limits, stated rather than papered over:
      * `from numpy import interp` binds the original at import time and is
        invisible to the patch. Call through the module.
      * ndarray METHODS (`a.mean()`, `a.clip()`) and compiled inner loops are
        not traceable this way.
      * User-level aperture and selection steps must be declared with
        `trace_note` or wrapped with `@traced`; that part is cooperative.
    """
    tr: list = []
    _TRACE_STACK.append(tr)
    saved = []
    for mod, nm, cls in list(_patch_targets()) + list(extra_targets):
        orig = getattr(mod, nm)
        if getattr(orig, "_controls_wrapped", False):
            continue
        saved.append((mod, nm, orig))
        setattr(mod, nm, _wrap(f"{getattr(mod,'__name__',mod)}.{nm}", cls, orig))
    try:
        yield tr
    finally:
        for mod, nm, orig in saved:
            setattr(mod, nm, orig)
        _TRACE_STACK.pop()


def trace_note(name: str, **kw):
    """Declare a pipeline step the patcher cannot see (aperture, selection)."""
    if _TRACE_STACK:
        _TRACE_STACK[-1].append({
            "op": f"note.{name}", "class": "note", "args": (),
            "kwargs": tuple(sorted((k, _argsig(v)) for k, v in kw.items())),
            "out": ("exact", None)})


def traced(name=None):
    """Decorator: put a user function into the trace with its argument shapes."""
    def deco(fn):
        lbl = name or fn.__qualname__

        def inner(*args, **kwargs):
            out = fn(*args, **kwargs)
            if _TRACE_STACK:
                _TRACE_STACK[-1].append({
                    "op": f"user.{lbl}", "class": "user",
                    "args": tuple(_argsig(a) for a in args),
                    "kwargs": tuple(sorted((k, _argsig(v)) for k, v in kwargs.items())),
                    "out": _argsig(out) if not isinstance(out, tuple)
                           else ("tuple", tuple(_argsig(o) for o in out))})
            return out
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner
    return deco


def _cmp_sig(a, b):
    """Return ('ok'|'warn'|'error', message)."""
    if a[0] != b[0]:
        return "error", f"kind {a[0]} vs {b[0]}"
    if a[0] in ("array", "smallarray"):
        if a[1] != b[1]:
            return "error", f"shape {a[1]} vs {b[1]}"
        if a[2] != b[2]:
            return "error", f"dtype {a[2]} vs {b[2]}"
        if a[3] != b[3]:
            return "warn", f"non-finite count {a[3]} vs {b[3]}"
        if a[0] == "smallarray" and a[4] != b[4]:
            return "warn", (f"small array of shape {a[1]} differs in VALUE "
                            f"({a[4]} vs {b[4]}) -- a kernel, bin edge set or "
                            "aperture that is not the same in both arms is an "
                            "exchangeability failure; a small data profile is "
                            "not")
        return "ok", ""
    if a[0] == "exact":
        return ("ok", "") if a[1] == b[1] else ("error", f"{a[1]!r} vs {b[1]!r}")
    if a[0] == "float":
        if a[1] == b[1] or (not math.isfinite(a[1]) and not math.isfinite(b[1])):
            return "ok", ""
        return "warn", f"data-derived scalar {a[1]!r} vs {b[1]!r}"
    if a[0] == "seq":
        if a[1] != b[1]:
            return "error", f"length {a[1]} vs {b[1]}"
        worst = "ok"; msgs = []
        for x, y in zip(a[2], b[2]):
            s, m = _cmp_sig(x, y)
            if s == "error":
                worst = "error"; msgs.append(m)
            elif s == "warn" and worst == "ok":
                worst = "warn"; msgs.append(m)
        return worst, "; ".join(msgs)
    if a[0] == "tuple":
        worst = "ok"; msgs = []
        for x, y in zip(a[1], b[1]):
            s, m = _cmp_sig(x, y)
            if s == "error":
                worst = "error"; msgs.append(m)
            elif s == "warn" and worst == "ok":
                worst = "warn"; msgs.append(m)
        return worst, "; ".join(msgs)
    return ("ok", "") if a == b else ("error", f"{a!r} vs {b!r}")


def check_exchangeability(pipeline: Callable, arm_true, arm_control, *,
                          name: str = "pipeline", raise_on_fail: bool = True,
                          strict: bool = False, verbose: bool = True) -> dict:
    """CONTROL 7. Verify the two arms went through EXACTLY the same operations.

    `pipeline(source)` is run once on the real source and once on the control
    realisation, with the numpy/scipy entry points instrumented. The two
    operation traces are then diffed:

      ERROR  a different op, a different order, a different array shape or
             dtype, a different exact-valued argument, or ANY randomness
             inside the pipeline (a random pipeline makes the arms
             incomparable by construction; the control must be realised
             BEFORE the pipeline, not inside it)
      WARN   a different count of non-finite values, or a data-derived float
             argument that differs -- legitimate in an adaptive step, but it
             is exactly how an adaptive cut stops being exchangeable, so it is
             reported with both numbers

    The point of doing it this way is that a docstring saying "both arms use
    the same code" is not checkable and this is.
    """
    with trace_ops() as ta:
        out_true = pipeline(arm_true)
    with trace_ops() as tb:
        out_ctrl = pipeline(arm_control)
    errors, warns = [], []
    rnd = [i for i, e in enumerate(ta) if e["class"] == "RANDOM"]
    rnd += [i for i, e in enumerate(tb) if e["class"] == "RANDOM"]
    if rnd:
        errors.append({"index": int(rnd[0]), "kind": "randomness",
                       "msg": "the pipeline itself draws random numbers; "
                              "realise the control before the pipeline"})
    n = min(len(ta), len(tb))
    for i in range(n):
        a, b = ta[i], tb[i]
        if a["op"] != b["op"]:
            errors.append({"index": i, "kind": "op-sequence",
                           "msg": f"{a['op']} vs {b['op']}"})
            break
        for fieldname in ("args", "kwargs", "out"):
            av, bv = a[fieldname], b[fieldname]
            if fieldname == "kwargs":
                ka = dict(av); kb = dict(bv)
                if set(ka) != set(kb):
                    errors.append({"index": i, "kind": "kwargs",
                                   "msg": f"{a['op']}: {sorted(ka)} vs {sorted(kb)}"})
                    continue
                for k in ka:
                    st, msg = _cmp_sig(ka[k], kb[k])
                    if st == "error":
                        errors.append({"index": i, "kind": f"kwarg:{k}",
                                       "msg": f"{a['op']}: {msg}"})
                    elif st == "warn":
                        warns.append({"index": i, "kind": f"kwarg:{k}",
                                      "msg": f"{a['op']}: {msg}"})
                continue
            if fieldname == "args":
                if len(av) != len(bv):
                    errors.append({"index": i, "kind": "arity",
                                   "msg": f"{a['op']}: {len(av)} vs {len(bv)}"})
                    continue
                for k, (x, y) in enumerate(zip(av, bv)):
                    st, msg = _cmp_sig(x, y)
                    if st == "error":
                        errors.append({"index": i, "kind": f"arg{k}",
                                       "msg": f"{a['op']}: {msg}"})
                    elif st == "warn":
                        warns.append({"index": i, "kind": f"arg{k}",
                                      "msg": f"{a['op']}: {msg}"})
                continue
            st, msg = _cmp_sig(av, bv)
            if st == "error":
                errors.append({"index": i, "kind": "output", "msg": f"{a['op']}: {msg}"})
            elif st == "warn":
                warns.append({"index": i, "kind": "output", "msg": f"{a['op']}: {msg}"})
    if len(ta) != len(tb):
        errors.append({"index": n, "kind": "op-count",
                       "msg": f"true arm made {len(ta)} traced calls, control "
                              f"arm made {len(tb)}"})
    if strict:
        errors = errors + warns
        warns = []
    rec = {"name": name, "n_ops_true": len(ta), "n_ops_control": len(tb),
           "op_classes": _class_counts(ta), "strict": strict,
           "n_errors": len(errors), "n_warnings": len(warns),
           "errors": errors[:20], "warnings": warns[:20],
           "passed": not errors,
           "result_true": _j(out_true) if np.isscalar(out_true) else "<obj>",
           "result_control": _j(out_ctrl) if np.isscalar(out_ctrl) else "<obj>"}
    if verbose:
        print(f"   exchangeability [{name}]: {len(ta)} traced ops on the true "
              f"arm, {len(tb)} on the control arm")
        print(f"      op classes: {rec['op_classes']}")
        print(f"      {len(errors)} error(s), {len(warns)} warning(s)  "
              f"{'PASS' if rec['passed'] else 'FAIL'}")
        for e in errors[:6]:
            print(f"      ERROR at op {e['index']}  [{e['kind']}]  {e['msg']}")
        for w in warns[:4]:
            print(f"      warn  at op {w['index']}  [{w['kind']}]  {w['msg']}")
    if errors and raise_on_fail:
        raise ExchangeabilityError(
            f"pipeline '{name}' treats the two arms differently: "
            + "; ".join(f"[{e['kind']}] {e['msg']}" for e in errors[:5]))
    return rec


def _class_counts(tr):
    out = {}
    for e in tr:
        out[e["class"]] = out.get(e["class"], 0) + 1
    return out


# ==========================================================================
#  CONTROL 9 -- FROZEN-COEFFICIENT ENFORCEMENT
# ==========================================================================
class SealedHoldoutError(RuntimeError):
    """An attempt to reach the held-out data in a way the API does not allow."""


class FrozenSealError(RuntimeError):
    """A FrozenModel's coefficients do not match the seal issued at fit time."""


_SEAL_KEY = secrets.token_bytes(32)     # per-process; seals are not portable


def _seal(coef, train_fp, design_fp):
    h = hmac.new(_SEAL_KEY, digestmod=hashlib.sha256)
    h.update(np.ascontiguousarray(coef, np.float64).tobytes())
    h.update(train_fp.encode()); h.update(design_fp.encode())
    return h.hexdigest()


class FrozenModel:
    """Coefficients solved on the training rows and then frozen.

    There is no setter, no `refit`, and no way to obtain a writable view of
    `coef`: the array is returned read-only and every use re-checks an HMAC
    over the coefficient bytes and the fingerprints of the data it was fitted
    on. A model whose coefficients were touched after freezing cannot be
    evaluated -- it raises rather than reporting a number.
    """
    __slots__ = ("_coef", "_atoms", "_train_fp", "_design_fp", "_seal",
                 "_ridge", "_nbatch")

    def __init__(self, coef, atoms, train_fp, design_fp, ridge):
        c = np.array(coef, np.float64)
        c.setflags(write=False)
        self._coef = c
        self._atoms = tuple(atoms)
        self._train_fp = train_fp
        self._design_fp = design_fp
        self._ridge = float(ridge)
        self._nbatch = 1 if c.ndim == 1 else int(c.shape[1])
        self._seal = _seal(c, train_fp, design_fp)

    @property
    def coef(self):
        return self._coef

    @property
    def atoms(self):
        return self._atoms

    @property
    def nbatch(self):
        return self._nbatch

    def _verify(self):
        if _seal(self._coef, self._train_fp, self._design_fp) != self._seal:
            raise FrozenSealError(
                "the frozen coefficients no longer match the seal issued at "
                "fit time. They were modified after freezing.")

    def predict(self, X):
        self._verify()
        return np.asarray(X, float) @ self._coef

    def __setattr__(self, k, v):
        if k in FrozenModel.__slots__ and hasattr(self, "_seal"):
            raise FrozenSealError(f"FrozenModel.{k} is frozen; refit instead")
        object.__setattr__(self, k, v)

    def __repr__(self):
        return (f"FrozenModel(p={self._coef.shape[0]}, batch={self._nbatch}, "
                f"train={self._train_fp[:8]}, seal={self._seal[:8]})")


class SplitData:
    """CONTROL 9. The only container through which a held-out set is touched.

    The design is that the WRONG THING CANNOT BE EXPRESSED:

      * the held-out rows of the design matrix and of the target are captured
        in a closure at construction and are NOT stored as attributes, so
        there is no `splits.blind_y` to hand to a solver;
      * the only thing that crosses into the held-out set is a FrozenModel,
        and `evaluate` accepts nothing else;
      * a FrozenModel can only be produced by `fit`, which slices the TRAIN
        rows and has no code path that reaches any other row;
      * evaluating a model whose train fingerprint does not match this
        object's raises, so a model fitted on some other split cannot be
        smuggled in;
      * `evaluate` returns scalars. It never returns residuals, predictions or
        targets, so an outer loop cannot reconstruct the held-out target by
        differencing;
      * the held-out set is touch-counted. `max_touches` above 1 requires a
        written reason, which is recorded in the audit log.

    In Run J this bug reported +2.17% where the correct procedure reported
    -3.73%: a 5.9-point swing, from positive to negative, entirely from
    re-solving the coefficients on the blind galaxies.

    What this CANNOT do: Python has no private state, so
    `obj.evaluate.__closure__` still reaches the arrays for anyone determined
    to. It stops the mistake, not the forgery. That distinction is stated
    plainly rather than dressed up.
    """

    def __init__(self, design, y, split, *, group=None, atoms=None,
                 max_touches=1, touch_reason=None, name=""):
        X = np.asarray(design, float)
        Y = np.asarray(y, float)
        split = np.asarray(split)
        assert X.ndim == 2, "design must be (n, p)"
        assert Y.shape[0] == X.shape[0], "y rows != design rows"
        assert split.shape[0] == X.shape[0], "split labels != design rows"
        if max_touches != 1 and not touch_reason:
            raise ValueError(
                "raising max_touches above 1 requires an explicit written "
                "reason; it is recorded in the audit log")
        self.name = name
        self.n, self.p = X.shape
        self.atoms = tuple(atoms) if atoms is not None else tuple(
            f"a{i}" for i in range(self.p))
        self.nbatch = 1 if Y.ndim == 1 else int(Y.shape[1])
        self.group = None if group is None else np.asarray(group)
        self._splits = {s: int((split == s).sum()) for s in np.unique(split)}
        self._max_touches = int(max_touches)
        self._touch_reason = touch_reason
        self._touches = {}
        self.audit = []
        tr = split == "train"
        self._Xtr = X[tr].copy()
        self._Ytr = Y[tr].copy()
        self._train_fp = fingerprint(self._Xtr, self._Ytr)
        self._design_fp = fingerprint(np.array([self.p]), np.array(
            [hash(a) % (1 << 31) for a in self.atoms], np.int64))
        held = {}
        for s in self._splits:
            if s == "train":
                continue
            held[s] = (X[split == s].copy(), Y[split == s].copy())

        def _evaluate(model, on):
            Xh, Yh = held[on]
            r = Yh - model.predict(Xh)
            rms = np.sqrt((r ** 2).mean(axis=0))
            return {"split": on, "n": int(Xh.shape[0]),
                    "rms": float(rms) if np.ndim(rms) == 0 else rms}
        self._evaluate = _evaluate
        self._held_names = tuple(held)
        # `held` stays alive ONLY inside the closure cell of `_evaluate`; it is
        # never bound to self, so there is no `splits.blind_y` to pass to a
        # solver. `X` and `Y` are dropped here so the full arrays do not
        # survive on the frame either.
        del X, Y

    # ------------------------------------------------------------ blocking
    def __getattr__(self, k):
        if k in ("blind", "blind_y", "blind_X", "holdout", "y_blind",
                 "validation", "X", "y", "design"):
            raise SealedHoldoutError(
                f"SplitData has no attribute '{k}'. The held-out rows are not "
                "reachable: fit on train, freeze, then call "
                "evaluate(frozen_model). If you can see the held-out target, "
                "you can fit on it, and this programme has already paid for "
                "that mistake once.")
        raise AttributeError(k)

    # ------------------------------------------------------------ the API
    def fit(self, ridge: float = 0.0) -> FrozenModel:
        """Solve the coefficients on the TRAIN rows and freeze them."""
        X, Y = self._Xtr, self._Ytr
        A = X.T @ X
        if ridge:
            A = A + ridge * X.shape[0] * np.eye(self.p)
        b = X.T @ Y
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        m = FrozenModel(coef, self.atoms, self._train_fp, self._design_fp, ridge)
        self.audit.append({"event": "fit", "ridge": ridge,
                           "n_train": int(X.shape[0]), "p": self.p,
                           "nbatch": self.nbatch, "seal": m._seal[:12]})
        return m

    def train_rms(self, model: FrozenModel):
        self._check(model)
        r = self._Ytr - model.predict(self._Xtr)
        v = np.sqrt((r ** 2).mean(axis=0))
        return float(v) if np.ndim(v) == 0 else v

    def evaluate(self, model: FrozenModel, on: str = "blind"):
        """Score a FROZEN model on a held-out split. Touch-counted."""
        if not isinstance(model, FrozenModel):
            raise SealedHoldoutError(
                "evaluate() accepts a FrozenModel only. Anything that could "
                "carry data -- an array, a design matrix, a fitting function "
                "-- is refused, because that is how coefficients get re-solved "
                "on the held-out set.")
        self._check(model)
        if on not in self._held_names:
            raise SealedHoldoutError(f"no held-out split named {on!r}; have "
                                     f"{list(self._held_names)}")
        used = self._touches.get(on, 0)
        if used >= self._max_touches:
            raise SealedHoldoutError(
                f"held-out split {on!r} has already been touched "
                f"{used} time(s), the declared maximum. A held-out set that is "
                "consulted repeatedly is a validation set with a misleading "
                "name.")
        self._touches[on] = used + 1
        out = self._evaluate(model, on)
        self.audit.append({"event": "evaluate", "split": on,
                           "touch": used + 1, "seal": model._seal[:12],
                           "rms": _j(out["rms"])})
        return out

    def _check(self, model):
        model._verify()
        if model._train_fp != self._train_fp:
            raise SealedHoldoutError(
                "this FrozenModel was fitted on different training data "
                "(fingerprint mismatch). Evaluating it here would compare a "
                "model to a split it never saw.")
        if model._design_fp != self._design_fp:
            raise SealedHoldoutError("atom set mismatch between model and split")

    def report(self):
        return {"name": self.name, "n": self.n, "p": self.p,
                "nbatch": self.nbatch, "splits": self._splits,
                "max_touches": self._max_touches,
                "touch_reason": self._touch_reason,
                "touches": dict(self._touches),
                "train_fingerprint": self._train_fp[:16],
                "audit": self.audit}


def _unguarded_refit_on_holdout(design, y, split, on="blind", ridge=0.0):
    """THE ANTI-PATTERN, implemented once so its size can be measured.

    This is the Run J bug: solve the coefficients using the held-out rows and
    then quote the held-out RMS. It is never called by any control; it exists
    only so `test_controls.py` can put a number on what the guard prevents.
    """
    X = np.asarray(design, float); Y = np.asarray(y, float)
    split = np.asarray(split)
    m = split == on
    Xh, Yh = X[m], Y[m]
    A = Xh.T @ Xh
    if ridge:
        A = A + ridge * Xh.shape[0] * np.eye(X.shape[1])
    coef = np.linalg.lstsq(A, Xh.T @ Yh, rcond=None)[0]
    r = Yh - Xh @ coef
    return float(np.sqrt((r ** 2).mean())) if r.ndim == 1 else \
        np.sqrt((r ** 2).mean(axis=0))

# ==========================================================================
#  CONTROL 8 -- SHARED-DENOMINATOR DETECTOR
# ==========================================================================
_SAFE_ENV = {
    "np": np, "log": np.log, "log10": np.log10, "exp": np.exp,
    "sqrt": np.sqrt, "abs": np.abs, "power": np.power, "pi": np.pi,
    "nu_rar": nu_rar, "G_KPC": G_KPC, "A0_KPC": A0_KPC,
    "minimum": np.minimum, "maximum": np.maximum,
}


def expression_inputs(expr: str, names: Iterable[str]) -> set:
    """Which measured inputs a construction expression actually uses.

    Parsed from the AST, not matched with a substring: `M_WL` and `M_WL_err`
    are different names and `log(M)` does not contain `M` by accident.
    """
    names = set(names)
    tree = ast.parse(expr, mode="eval")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return used & names


def _evaluate_exprs(exprs: dict, values: dict, env: dict = None):
    e = dict(_SAFE_ENV)
    if env:
        e.update(env)
    e.update(values)
    out = {}
    for k, ex in exprs.items():
        out[k] = np.asarray(eval(ex, {"__builtins__": {}}, e), float)  # noqa: S307
    return out


def _draw_inputs(inputs, rng, size=None):
    """One noisy realisation of every measured input, using its own sigma."""
    out = {}
    for k, spec in inputs.items():
        v = np.asarray(spec["value"], float)
        s = np.asarray(spec.get("sigma", 0.0), float) * np.ones_like(v)
        dist = spec.get("dist", "normal")
        if np.all(s == 0):
            out[k] = v.copy() if size is None else np.tile(v, (size, 1))
            continue
        shp = v.shape if size is None else (size,) + v.shape
        z = rng.normal(size=shp)
        if dist == "lognormal":
            rel = s / np.maximum(np.abs(v), 1e-300)
            out[k] = v * np.exp(rel * z - 0.5 * rel ** 2)
        else:
            out[k] = v + s * z
    return out


def shared_denominator_report(inputs: dict, exprs: dict,
                              estimator: Callable[[dict], float], *,
                              null_carrier: str, seed: int = 0,
                              ndraw: int = 4000, nnull: int = 4000,
                              env: dict = None, verbose: bool = True,
                              also_decorrelate_null: bool = True,
                              series_order: Sequence[str] = None,
                              carrier_series: str = None,
                              nboot_eiv: int = 0) -> dict:
    """CONTROL 8. Is a measured input on BOTH axes, and what is the null then?

    inputs   {name: {"value": (n,), "sigma": (n,), "dist": "normal"|"lognormal"}}
    exprs    {series_name: construction expression in terms of the input names}
    estimator(series_dict) -> float, the naive statistic under test
    null_carrier  the input whose association with the rest is destroyed to
                  build H0. Everything else -- including the shared input and
                  its error -- is left exactly as it is, so the null retains
                  the artefact and contains none of the signal.

    Returns the shared inputs, the INDUCED error correlation between the
    series, the null distribution of the naive estimator, and the p-value of
    the observed value AGAINST ITS OWN NULL rather than against zero.

    This is the standing first suspicion of this programme. rho_p = -0.304 was
    retracted because ln E_obs and ln M_WL both contain M_WL: their errors
    correlate at +0.96, the naive partial estimator has expectation about
    -0.12 under a true null, and the observed value therefore sat at p = 0.563
    -- an artefact reported as evidence.
    """
    rng = _as_rng(seed)
    names = list(inputs)
    used = {k: sorted(expression_inputs(ex, names)) for k, ex in exprs.items()}
    keys = list(exprs)
    shared = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            s = sorted(set(used[a]) & set(used[b]))
            if s:
                shared[f"{a}|{b}"] = s
    obs_series = _evaluate_exprs(exprs, {k: np.asarray(v["value"], float)
                                         for k, v in inputs.items()}, env)
    n = len(next(iter(obs_series.values())))
    stat_obs = float(estimator(obs_series))

    # ---- induced error correlation: perturb the inputs, hold the latents
    draws = {k: [] for k in keys}
    for _ in range(ndraw):
        d = _draw_inputs(inputs, rng)
        s = _evaluate_exprs(exprs, d, env)
        for k in keys:
            draws[k].append(s[k])
    D = {k: np.array(v) for k, v in draws.items()}          # (ndraw, n)
    err_corr, err_cov = {}, {}
    for i, a in enumerate(keys):
        for b in keys[i:]:
            ca = D[a] - D[a].mean(axis=0)
            cb = D[b] - D[b].mean(axis=0)
            cov = (ca * cb).mean(axis=0)
            err_cov[f"{a}|{b}"] = cov
            if b != a:
                sa = D[a].std(axis=0); sb = D[b].std(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = cov / np.maximum(sa * sb, 1e-300)
                err_corr[f"{a}|{b}"] = r

    # ---- the null of the NAIVE estimator, with the real error covariance
    lat = {k: np.asarray(v["value"], float) for k, v in inputs.items()}
    null_perm = np.empty(nnull)
    for t in range(nnull):
        L = dict(lat)
        L[null_carrier] = lat[null_carrier][rng.permutation(n)]
        tmp = {k: dict(inputs[k]) for k in inputs}
        for k in tmp:
            tmp[k] = dict(tmp[k]); tmp[k]["value"] = L[k]
        d = _draw_inputs(tmp, rng)
        null_perm[t] = estimator(_evaluate_exprs(exprs, d, env))
    null_dec = None
    if also_decorrelate_null:
        null_dec = _decorrelated_null(inputs, exprs, estimator, null_carrier,
                                      rng, nnull, env)

    # ---- the STRUCTURAL null and the errors-in-variables estimator
    null_str, eiv = None, None
    if series_order is not None and carrier_series is not None:
        ks = list(series_order)
        Cm = error_covariance_from_inputs(inputs, exprs, ks, seed=seed + 1,
                                          ndraw=max(1000, ndraw // 2), env=env)
        Ym = np.column_stack([obs_series[k] for k in ks])
        eiv = eiv_fit(Ym, Cm, nboot=nboot_eiv, seed=seed + 2)
        null_str = structural_null(Ym, Cm, estimator, ks, carrier_series,
                                   nnull=nnull, seed=seed + 3, fit=eiv)

    def _p(nullv):
        nullv = nullv[np.isfinite(nullv)]
        if nullv.size == 0:
            return {}
        ge = float((nullv >= stat_obs).mean())
        le = float((nullv <= stat_obs).mean())
        return {"mean": float(nullv.mean()), "median": float(np.median(nullv)),
                "sd": float(nullv.std(ddof=1)),
                "q025": float(np.quantile(nullv, 0.025)),
                "q975": float(np.quantile(nullv, 0.975)),
                "p_two_sided": float(min(1.0, 2 * min(ge, le))),
                "p_one_sided_le": le, "p_one_sided_ge": ge,
                "p_naive_vs_zero": float(min(1.0, 2 * min(
                    (nullv >= 0).mean(), (nullv <= 0).mean())))}

    rec = {"n_objects": int(n),
           "inputs_used": used, "shared_inputs": shared,
           "has_shared_input": bool(shared),
           "statistic_observed": stat_obs,
           "induced_error_correlation": {
               k: {"mean": float(np.nanmean(v)),
                   "median": float(np.nanmedian(v)),
                   "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
               for k, v in err_corr.items()},
           "null_permutation": _p(null_perm),
           "null_decorrelated": _p(null_dec) if null_dec is not None else None,
           "null_structural": _p(null_str) if null_str is not None else None,
           "eiv": None if eiv is None else {
               "series": list(series_order),
               "beta": [float(v) for v in eiv["beta"]],
               "beta_naive": [float(v) for v in eiv["beta_naive"]],
               "beta_ci95": eiv.get("beta_ci95"),
               "intrinsic_scatter": float(eiv["intrinsic_scatter"]),
               "carrier": carrier_series,
               "carrier_beta": float(eiv["beta"][list(series_order).index(
                   carrier_series) - 1]),
               "carrier_beta_naive": float(eiv["beta_naive"][
                   list(series_order).index(carrier_series) - 1])},
           "null_expectation_is_zero": None,
           "ndraw": ndraw, "nnull": nnull, "null_carrier": null_carrier}
    primary = rec["null_structural"] or rec["null_permutation"]
    rec["primary_null"] = ("structural" if rec["null_structural"]
                           else "permutation")
    m = primary.get("mean", float("nan"))
    sd = primary.get("sd", float("nan"))
    rec["null_expectation_is_zero"] = bool(abs(m) < 2 * sd / math.sqrt(nnull))
    rec["verdict"] = (
        f"NAIVE NULL IS NOT ZERO (E = {m:+.3f}) -- compare the observed value "
        "to this null, not to zero"
        if not rec["null_expectation_is_zero"] else
        "naive null is consistent with zero at this sample size")
    if verbose:
        print(f"   shared-denominator report ({n} objects)")
        for k, v in used.items():
            print(f"      {k:<14} built from {v}")
        if shared:
            for k, v in shared.items():
                print(f"      SHARED INPUT on {k}: {v}")
        else:
            print("      no measured input appears in more than one series")
        for k, v in rec["induced_error_correlation"].items():
            print(f"      induced error correlation {k}: mean "
                  f"{v['mean']:+.3f}  range [{v['min']:+.3f}, {v['max']:+.3f}]")
        print(f"      observed statistic {stat_obs:+.4f}")
        np_ = rec["null_permutation"]
        print(f"      null (permute {null_carrier}): mean {np_['mean']:+.4f}"
              f"  sd {np_['sd']:.4f}  95% [{np_['q025']:+.4f}, {np_['q975']:+.4f}]")
        print(f"      p(observed vs ITS OWN null) = {np_['p_two_sided']:.3f}")
        if null_dec is not None:
            nd = rec["null_decorrelated"]
            print(f"      null (decorrelated latents): mean {nd['mean']:+.4f}"
                  f"   p = {nd['p_two_sided']:.3f}")
        if null_str is not None:
            ns = rec["null_structural"]
            print(f"      null (STRUCTURAL, beta[{carrier_series}] = 0, all "
                  f"other structure kept):")
            print(f"          mean {ns['mean']:+.4f}  sd {ns['sd']:.4f}  "
                  f"95% [{ns['q025']:+.4f}, {ns['q975']:+.4f}]")
            print(f"          p(observed vs THIS null) = "
                  f"{ns['p_two_sided']:.3f}")
            e = rec["eiv"]
            print(f"      EIV slope d{series_order[0]}/d{carrier_series} = "
                  f"{e['carrier_beta']:+.4f}   naive OLS "
                  f"{e['carrier_beta_naive']:+.4f}"
                  + (f"   95% {[round(e['beta_ci95'][0][list(series_order).index(carrier_series)-1], 3), round(e['beta_ci95'][1][list(series_order).index(carrier_series)-1], 3)]}"
                     if e.get("beta_ci95") else ""))
        print(f"      => {rec['verdict']}  (primary null: "
              f"{rec['primary_null']})")
    return rec


def _decorrelated_null(inputs, exprs, estimator, carrier, rng, nnull, env):
    """Second null: latents drawn from a joint model with the carrier's
    cross-covariances set to zero, errors added with the real sigmas."""
    names = list(inputs)
    tf = {k: (np.log if inputs[k].get("dist") == "lognormal" else (lambda z: z))
          for k in names}
    itf = {k: (np.exp if inputs[k].get("dist") == "lognormal" else (lambda z: z))
           for k in names}
    Z = np.column_stack([tf[k](np.asarray(inputs[k]["value"], float))
                         for k in names])
    mu = Z.mean(axis=0)
    S = np.cov(Z, rowvar=False)
    S = np.atleast_2d(S)
    j = names.index(carrier)
    S = S.copy()
    S[j, :] = 0.0; S[:, j] = 0.0
    S[j, j] = float(np.var(Z[:, j], ddof=1))
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-12)
    Lc = V @ np.diag(np.sqrt(w))
    n = Z.shape[0]
    out = np.empty(nnull)
    for t in range(nnull):
        lat = mu + rng.normal(size=(n, len(names))) @ Lc.T
        tmp = {}
        for i, k in enumerate(names):
            tmp[k] = dict(inputs[k])
            tmp[k]["value"] = itf[k](lat[:, i])
        d = _draw_inputs(tmp, rng)
        out[t] = estimator(_evaluate_exprs(exprs, d, env))
    return out


def structural_null(Y, C, estimator, keys, carrier: str, *, nnull=4000,
                    seed=0, fit=None, verbose=False):
    """The null that KEEPS the covariate structure and removes only the link
    under test. This is the one that exposes a shared-denominator artefact.

    Fit the errors-in-variables model to the observed series, set the
    coefficient of the CARRIER to zero, and simulate from the fitted model
    with the real per-object error covariance C_i. The simulated data then
    have:

      * the real latent correlation between the carrier and the control
        variables (in the LoCuSS case, hotter clusters really are more
        massive);
      * the real dependence of the response on the controls;
      * the real, strongly off-diagonal error covariance produced by the
        shared input;
      * and NO dependence of the response on the carrier.

    Permuting the carrier instead destroys the carrier-control association
    too, which removes the very mechanism that biases the naive estimator. A
    permutation null is not automatically the conservative choice, and here it
    is the wrong one: it centres at zero and declares the artefact
    significant.
    """
    rng = _as_rng(seed)
    Y = np.asarray(Y, float); C = np.asarray(C, float)
    n, m = Y.shape
    j = keys.index(carrier)
    if j == 0:
        raise ValueError("the carrier cannot be the response (column 0)")
    fit = fit if fit is not None else eiv_fit(Y, C)
    beta0 = np.array(fit["beta"], float)
    beta0[j - 1] = 0.0
    mu, Sig, s, c0 = fit["mu"], fit["Sigma"], fit["intrinsic_scatter"], fit["intercept"]
    w, V = np.linalg.eigh(np.atleast_2d(Sig))
    Ls = V @ np.diag(np.sqrt(np.maximum(w, 1e-14)))
    Lc = np.linalg.cholesky(C + 1e-12 * np.eye(m)[None])
    out = np.empty(nnull)
    for t in range(nnull):
        xi = mu + rng.normal(size=(n, m - 1)) @ Ls.T
        eta = c0 + xi @ beta0 + rng.normal(0.0, s, n)
        lat = np.column_stack([eta, xi])
        e = np.einsum("nij,nj->ni", Lc, rng.normal(size=(n, m)))
        Yn = lat + e
        out[t] = estimator({k: Yn[:, i] for i, k in enumerate(keys)})
    if verbose:
        print(f"      structural null (beta[{carrier}] = 0): mean "
              f"{out.mean():+.4f}  sd {out.std(ddof=1):.4f}")
    return out


def partial_spearman(a, b, c):
    """rho_p(a, b | c) -- the exact statistic that was retracted."""
    ra, rb, rc = (_rank(a), _rank(b), _rank(c))
    rab = np.corrcoef(ra, rb)[0, 1]
    rac = np.corrcoef(ra, rc)[0, 1]
    rbc = np.corrcoef(rb, rc)[0, 1]
    den = math.sqrt(max(1e-15, (1 - rac ** 2) * (1 - rbc ** 2)))
    return float((rab - rac * rbc) / den)


def _rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(x.size, float)
    r[o] = np.arange(1, x.size + 1, dtype=float)
    return r


# -------------------------------------------------- errors-in-variables
def _eiv_nll(theta, Y, C, m):
    k = m - 1
    c = theta[0]
    beta = theta[1:1 + k]
    lns = theta[1 + k]
    mu = theta[2 + k:2 + 2 * k]
    lt = theta[2 + 2 * k:]
    L = np.zeros((k, k))
    iu = np.tril_indices(k)
    L[iu] = lt
    d = np.diag(L).copy()
    np.fill_diagonal(L, np.exp(np.clip(d, -12, 8)))
    Sig = L @ L.T
    B = np.zeros((m, k))
    B[0] = beta
    B[1:] = np.eye(k)
    V = B @ Sig @ B.T
    V[0, 0] += math.exp(2 * min(lns, 8.0))
    mean = np.empty(m)
    mean[0] = c + float(beta @ mu)
    mean[1:] = mu
    Vi = V[None, :, :] + C
    try:
        Lc = np.linalg.cholesky(Vi)
    except np.linalg.LinAlgError:
        return 1e12
    d0 = (Y - mean)[:, :, None]
    sol = np.linalg.solve(Lc, d0)[:, :, 0]
    return float(0.5 * (2.0 * np.sum(np.log(np.diagonal(Lc, axis1=1, axis2=2)))
                        + np.sum(sol * sol)))


def _eiv_unpack(theta, m):
    """(intercept, beta, sigma_intrinsic, mu, Sigma) from the packed vector."""
    k = m - 1
    c = float(theta[0])
    beta = np.asarray(theta[1:1 + k], float)
    s = math.exp(min(float(theta[1 + k]), 8.0))
    mu = np.asarray(theta[2 + k:2 + 2 * k], float)
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = theta[2 + 2 * k:]
    d = np.diag(L).copy()
    np.fill_diagonal(L, np.exp(np.clip(d, -12, 8)))
    return c, beta, s, mu, L @ L.T


def eiv_fit(Y, C, *, nboot: int = 0, seed: int = 0, polish: bool = True,
            x0=None):
    """Errors-in-variables regression with a PER-OBJECT error covariance.

    Y  (n, m)     observed series; column 0 is the response, 1..m-1 covariates
    C  (n, m, m)  the measurement error covariance of each object's row --
                  NOT assumed diagonal. When one measured input enters two
                  series, the off-diagonal is the whole point.

    Model
        xi_i ~ N(mu, Sigma)                        latent covariates
        eta_i = c + beta . xi_i + eps_i            latent response
        y_i   = (eta_i, xi_i) + e_i,  e_i ~ N(0, C_i)
    so the marginal is Gaussian with covariance B Sigma B' + s^2 e1 e1' + C_i
    and the likelihood is exact. Returns beta (the slopes the naive fit
    attenuates), the intrinsic scatter, and the naive OLS slopes for contrast.
    """
    Y = np.asarray(Y, float); C = np.asarray(C, float)
    n, m = Y.shape
    k = m - 1
    assert C.shape == (n, m, m), f"C must be (n,m,m), got {C.shape}"
    # naive OLS
    X = np.column_stack([np.ones(n), Y[:, 1:]])
    beta_naive = np.linalg.lstsq(X, Y[:, 0], rcond=None)[0][1:]
    # method of moments start
    S = np.cov(Y, rowvar=False)
    V = np.atleast_2d(S) - C.mean(axis=0)
    try:
        beta0 = np.linalg.solve(V[1:, 1:], V[0, 1:])
    except np.linalg.LinAlgError:
        beta0 = beta_naive.copy()
    s20 = max(float(V[0, 0] - beta0 @ V[1:, 1:] @ beta0), 1e-6)
    Sig0 = V[1:, 1:]
    w, Vv = np.linalg.eigh(np.atleast_2d(Sig0))
    Sig0 = Vv @ np.diag(np.maximum(w, 1e-6)) @ Vv.T
    L0 = np.linalg.cholesky(Sig0)
    lt = L0[np.tril_indices(k)].copy()
    di = np.cumsum(np.arange(1, k + 1)) - 1
    lt[di] = np.log(np.maximum(np.diag(L0), 1e-6))
    mu0 = Y[:, 1:].mean(axis=0)
    c0 = float(Y[:, 0].mean() - beta0 @ mu0)
    if x0 is None:
        x0 = np.r_[c0, beta0, 0.5 * math.log(s20), mu0, lt]
    if _optimize is None:                                  # pragma: no cover
        return {"beta": beta0, "beta_naive": beta_naive, "method": "moments"}
    r = _optimize.minimize(_eiv_nll, x0, args=(Y, C, m), method="L-BFGS-B",
                           options=dict(maxiter=2000, maxfun=20000))
    if polish:
        r2 = _optimize.minimize(_eiv_nll, r.x, args=(Y, C, m),
                                method="Nelder-Mead",
                                options=dict(maxiter=6000, maxfev=6000,
                                             xatol=1e-9, fatol=1e-9))
        if r2.fun < r.fun:
            r = _optimize.minimize(_eiv_nll, r2.x, args=(Y, C, m),
                                   method="L-BFGS-B",
                                   options=dict(maxiter=2000, maxfun=20000))
    th = r.x
    c_, beta, s_, mu_, Sig_ = _eiv_unpack(th, m)
    out = {"beta": beta, "beta_naive": beta_naive, "beta_moments": beta0,
           "intercept": c_, "mu": mu_, "Sigma": Sig_, "theta": th,
           "intrinsic_scatter": s_,
           "nll": float(r.fun), "converged": bool(r.success), "method": "mle"}
    if nboot:
        rng = _as_rng(seed)
        bs = []
        for _ in range(nboot):
            i = rng.integers(0, n, n)
            try:
                # warm-started from the full-data MLE: a cold start on n = 40
                # objects wanders off and produces bootstrap intervals ten
                # units wide, which are optimiser noise, not uncertainty
                b = eiv_fit(Y[i], C[i], nboot=0, polish=False, x0=th)["beta"]
            except Exception:                              # pragma: no cover
                continue
            if np.all(np.isfinite(b)) and np.max(np.abs(b)) < 50:
                bs.append(b)
        if bs:
            Bm = np.array(bs)
            out["beta_ci95"] = [np.quantile(Bm, 0.025, axis=0).tolist(),
                                np.quantile(Bm, 0.975, axis=0).tolist()]
            out["beta_boot_sd"] = Bm.std(axis=0, ddof=1).tolist()
            out["nboot_ok"] = len(bs)
    return out


def validate_eiv(*, n: int = 40, betas=(-0.6, -0.3, 0.0, 0.3, 0.6),
                 rho_err: float = 0.96, nsim: int = 200, seed: int = 0,
                 sig_err: float = 0.25, sig_lat: float = 0.5,
                 intrinsic: float = 0.15, verbose: bool = True) -> dict:
    """Is the EIV estimator unbiased across the WHOLE parameter range?

    Simulates the exact pathology: one covariate whose measurement error is
    correlated with the response's error at `rho_err` (the LoCuSS value is
    +0.96, because M_WL is inside both). Reports the bias of the naive OLS
    slope and of the EIV slope at each true beta. An estimator that is
    validated only at beta = 0 is not validated.
    """
    rng = _as_rng(seed)
    rows = []
    for b_true in betas:
        nai, ei = [], []
        for _ in range(nsim):
            xi = rng.normal(0, sig_lat, n)                 # latent covariate
            eta = b_true * xi + rng.normal(0, intrinsic, n)
            Ci = np.array([[sig_err ** 2, rho_err * sig_err * sig_err],
                           [rho_err * sig_err * sig_err, sig_err ** 2]])
            Lc = np.linalg.cholesky(Ci + 1e-12 * np.eye(2))
            e = rng.normal(size=(n, 2)) @ Lc.T
            Y = np.column_stack([eta + e[:, 0], xi + e[:, 1]])
            C = np.repeat(Ci[None], n, axis=0)
            X = np.column_stack([np.ones(n), Y[:, 1]])
            nai.append(np.linalg.lstsq(X, Y[:, 0], rcond=None)[0][1])
            try:
                ei.append(float(eiv_fit(Y, C, polish=False)["beta"][0]))
            except Exception:                              # pragma: no cover
                ei.append(np.nan)
        nai = np.array(nai); ei = np.array(ei[:])
        ok = np.isfinite(ei)
        rows.append({"beta_true": float(b_true),
                     "naive_mean": float(np.mean(nai)),
                     "naive_bias": float(np.mean(nai) - b_true),
                     "eiv_mean": float(np.nanmean(ei)),
                     "eiv_bias": float(np.nanmean(ei) - b_true),
                     "eiv_mcse": float(np.nanstd(ei, ddof=1) / math.sqrt(max(ok.sum(), 1))),
                     "eiv_sd": float(np.nanstd(ei, ddof=1)),
                     "n_ok": int(ok.sum())})
    worst = max(abs(r["eiv_bias"]) for r in rows)
    worst_z = max(abs(r["eiv_bias"]) / max(r["eiv_mcse"], 1e-12) for r in rows)
    nwb = max(abs(r["naive_bias"]) for r in rows)
    rec = {"rows": rows, "rho_err": rho_err, "n": n, "nsim": nsim,
           "worst_abs_eiv_bias": worst, "worst_bias_in_mcse": worst_z,
           "naive_worst_abs_bias": nwb,
           "bias_reduction_factor": float(nwb / max(worst, 1e-12)),
           # "unbiased" is a statement about SIZE, not about a p-value: an MLE
           # at n = 40 has a finite-sample bias, and with enough simulations
           # any nonzero bias is significant. What matters is whether it is
           # small compared with the effects being claimed.
           "unbiased": bool(worst < 0.05),
           "bias_significant_at_this_nsim": bool(worst_z > 3.0)}
    if verbose:
        print(f"   EIV validation: n={n}, error correlation {rho_err:+.2f}, "
              f"{nsim} sims per point")
        print("      beta_true    naive mean (bias)      EIV mean (bias)     "
              "MCSE")
        for r in rows:
            print(f"      {r['beta_true']:+8.3f}   {r['naive_mean']:+8.4f} "
                  f"({r['naive_bias']:+7.4f})   {r['eiv_mean']:+8.4f} "
                  f"({r['eiv_bias']:+7.4f})   {r['eiv_mcse']:.4f}")
        print(f"      worst |EIV bias| {worst:.4f} ({worst_z:.1f} MC standard "
              f"errors)   {'UNBIASED to < 0.05' if rec['unbiased'] else 'BIASED'}")
        print(f"      worst |naive bias| {nwb:.4f}  "
              "(this is the attenuation the naive estimator carries)")
        print(f"      the EIV estimator removes a factor "
              f"{rec['bias_reduction_factor']:.0f} of the bias; a residual "
              f"finite-sample bias of {worst:.3f} remains at n = {n}")
    return rec


def error_covariance_from_inputs(inputs, exprs, keys, *, seed=0, ndraw=3000,
                                 env=None):
    """Per-object (m,m) error covariance of the constructed series.

    Propagates the published input errors through the ACTUAL construction
    expressions by Monte Carlo, so the off-diagonal produced by a shared input
    is measured rather than assumed away.
    """
    rng = _as_rng(seed)
    D = {k: [] for k in keys}
    for _ in range(ndraw):
        d = _draw_inputs(inputs, rng)
        s = _evaluate_exprs(exprs, d, env)
        for k in keys:
            D[k].append(s[k])
    M = np.stack([np.array(D[k]) for k in keys], axis=-1)   # (ndraw, n, m)
    M = M - M.mean(axis=0, keepdims=True)
    n, m = M.shape[1], M.shape[2]
    C = np.einsum("dnm,dnk->nmk", M, M) / M.shape[0]
    return C

# ==========================================================================
#  CONTROL 5 -- SYNTHETIC KNOWN-LAW UNIVERSES
# ==========================================================================
#  Five generators, three of them SCALAR (no direction, no length scale of
#  their own), one genuinely anisotropic, one genuinely nonlocal. The
#  discovery pipeline is then run on each, and two things are checked:
#
#     RECOVERY      the injected family wins out of sample
#     NO INVENTION  a scalar universe does not make the pipeline report a
#                   tensor or a nonlocal effect
#
#  The second number -- the false-positive rate for "tensor effect detected in
#  scalar data" -- is the credibility of any future tensor claim, so it is
#  measured at three levels of discipline rather than asserted at one.
# --------------------------------------------------------------------------
RHO_C = 3.0 * (0.07 ** 2) / (8.0 * np.pi * G_KPC)      # Msun/kpc^3, H0=70

LAW_DEFAULTS = {
    "newton":   {},
    "mond":     {"a0": A0_KPC},
    "gr_dm":    {"M200_norm": 3.0e11, "M200_slope": 0.70, "conc": 10.0},
    "tensor":   {"a0": A0_KPC, "eps_T": 0.20},
    "nonlocal": {"a0": A0_KPC, "L_kpc": 6.0},
}
SCALAR_LAWS = ("newton", "mond", "gr_dm")
LAW_NAMES = tuple(LAW_DEFAULTS)


def _hernquist_M(r, M, a):
    return M * r ** 2 / (r + a) ** 2


def _nfw_M(r, M200, c):
    r200 = (M200 / (200.0 * RHO_C * 4.0 * np.pi / 3.0)) ** (1.0 / 3.0)
    rs = r200 / c
    f = np.log(1.0 + c) - c / (1.0 + c)
    x = r / rs
    return M200 * (np.log(1.0 + x) - x / (1.0 + x)) / f


def _kernel_smoothed_M(r, M, a, L, nq=65):
    """M_eff(<r) = Int M_b(<s) K(s-r; L) ds / Int K -- a genuinely NONLOCAL
    functional of the source: it depends on the profile away from r, so two
    systems with the same M_b(<r) but different shapes differ."""
    t = np.linspace(-5.0, 5.0, nq)
    w = np.exp(-0.5 * t ** 2)[None, :]
    s = r[:, None] + L * t[None, :]
    good = s > 0
    ww = np.where(good, w, 0.0)
    Ms = np.where(good, _hernquist_M(np.maximum(s, 1e-9), M[:, None], a[:, None]), 0.0)
    return (Ms * ww).sum(1) / np.maximum(ww.sum(1), 1e-300)


def law_g(law, params, *, r, cost, M, a, gN):
    """g(r, cos theta) under a named law. Global constants only."""
    p = dict(LAW_DEFAULTS[law]); p.update(params or {})
    if law == "newton":
        return gN.copy(), p
    if law == "mond":
        return nu_rar(gN / p["a0"]) * gN, p
    if law == "gr_dm":
        M200 = p["M200_norm"] * (M / 1.0e10) ** p["M200_slope"]
        Mh = _nfw_M(r, M200, p["conc"])
        return G_KPC * (_hernquist_M(r, M, a) + Mh) / r ** 2, p
    if law == "tensor":
        P2 = 0.5 * (3.0 * cost ** 2 - 1.0)
        return nu_rar(gN / p["a0"]) * gN * (1.0 + p["eps_T"] * P2), p
    if law == "nonlocal":
        Meff = _kernel_smoothed_M(r, M, a, p["L_kpc"])
        geff = G_KPC * Meff / r ** 2
        return nu_rar(geff / p["a0"]) * geff, p
    raise ValueError(f"unknown law {law!r}")


@dataclass
class MockUniverse:
    law: str
    params: dict
    source: ObjectPointSource
    sysid: np.ndarray
    r: np.ndarray
    cost: np.ndarray
    gN: np.ndarray
    Mb: np.ndarray
    a: np.ndarray
    rbin: np.ndarray
    y_true: np.ndarray
    offsets: np.ndarray
    sys_Mb: np.ndarray
    sys_a: np.ndarray

    @property
    def block_angular(self):
        """(system, radius) blocks: the null that destroys ANGULAR structure
        while holding the radial profile of the residual exactly fixed."""
        return self.sysid * 1000 + self.rbin


def synthetic_universe(law: str, seed: int = 0, *, n_sys: int = 60,
                       n_r: int = 10, n_t: int = 5, params: dict = None,
                       sigma_point: tuple = (0.08, 0.14),
                       sigma_offset: float = 0.06,
                       verbose: bool = False) -> ControlRealisation:
    """CONTROL 5a. A mock universe generated under a KNOWN law.

    Systems are Hernquist baryon spheroids with independent mass and scale, so
    "the shape of the source at fixed enclosed mass" is a real second
    direction and a nonlocal law is identifiable. Each system carries an
    orientation axis, and every point is sampled at a known angle to it, so an
    anisotropic law is identifiable too. Noise is heteroscedastic per point
    plus a per-system offset, which is what distance / inclination / M/L
    errors look like and is exactly what control 1 must preserve.
    """
    rng = _as_rng(seed)
    M = 10.0 ** rng.uniform(9.0, 11.0, n_sys)
    a = 10.0 ** rng.uniform(0.0, 0.9, n_sys)
    sysid, rr, ct, rbin = [], [], [], []
    for j in range(n_sys):
        rj = np.geomspace(0.5 * a[j], 25.0 * a[j], n_r)
        cj = np.linspace(0.05, 0.95, n_t)
        R, C = np.meshgrid(rj, cj, indexing="ij")
        B = np.repeat(np.arange(n_r), n_t)
        sysid.append(np.full(R.size, j)); rr.append(R.ravel())
        ct.append(C.ravel()); rbin.append(B)
    sysid = np.concatenate(sysid); rr = np.concatenate(rr)
    ct = np.concatenate(ct); rbin = np.concatenate(rbin)
    Mp, ap = M[sysid], a[sysid]
    gN = G_KPC * _hernquist_M(rr, Mp, ap) / rr ** 2
    g, p = law_g(law, params, r=rr, cost=ct, M=Mp, a=ap, gN=gN)
    y_true = np.log10(g / gN)
    off = rng.normal(0.0, sigma_offset, n_sys)
    sig = rng.uniform(sigma_point[0], sigma_point[1], rr.size)
    y = y_true + off[sysid] + rng.normal(0.0, sig)
    model = np.log10(nu_rar(gN / A0_KPC))            # the RAR baseline
    src = ObjectPointSource(sysid, np.column_stack([rr, ct]), y, sig, model,
                            name=f"mock[{law}]")
    u = MockUniverse(law, p, src, sysid, rr, ct, gN, Mp, ap, rbin, y_true,
                     off, M, a)
    inv = {"n_points": (rr.size, src.n, 0),
           "n_systems": (n_sys, int(np.unique(sysid).size),
                         abs(n_sys - int(np.unique(sysid).size)))}
    dest = {"injected_law": (law, law),
            "signal_rms_dex": (float(np.sqrt((y_true ** 2).mean())), None),
            "angular_signal_rms_dex":
                (float(np.sqrt(((y_true - _block_mean_expand(
                    y_true, sysid * 1000 + rbin)) ** 2).mean())), None)}
    if verbose:
        print(f"   mock[{law}] {n_sys} systems x {n_r} radii x {n_t} angles "
              f"= {rr.size} points   signal "
              f"{dest['signal_rms_dex'][0]:.4f} dex   angular part "
              f"{dest['angular_signal_rms_dex'][0]:.4f} dex")
    return ControlRealisation(f"synthetic_universe[{law}]", u, inv, dest,
                              {"seed": _seed_of(seed), "law": law,
                               "params": _j(p), "n_sys": n_sys, "n_r": n_r,
                               "n_t": n_t, "sigma_offset": sigma_offset})


# ------------------------------------------------------------- atom banks
def _standardise(cols, names):
    X = np.column_stack(cols)
    keep = []
    for i in range(X.shape[1]):
        if names[i] == "1":
            keep.append(i); continue
        s = X[:, i].std()
        if np.isfinite(X[:, i]).all() and s > 1e-12:
            X[:, i] = (X[:, i] - X[:, i].mean()) / s
            keep.append(i)
    return X[:, keep], [names[i] for i in keep]


def build_bank(u: MockUniverse, banks=("scalar", "radial", "gal"),
               L_nl=(2.0, 6.0, 20.0), shape_override=None):
    """Design matrix, tagged by which direction each atom uses.

    scalar   functions of x = g_N/a0 alone -- the RAR's own direction
    radial   functions of r -- the second scalar direction
    gal      system-level scalars
    tensor   functions of cos(theta) to the system axis; ZERO-MEAN over the
             sphere, so a law with no direction has no projection on them
    nonlocal functions of the kernel-smoothed source at fixed PHYSICAL length
    """
    x = np.maximum(u.gN / A0_KPC, 1e-30)
    r = u.r
    cols, names, dims = [], [], []

    def add(nm, dim, v):
        cols.append(np.asarray(v, float)); names.append(nm); dims.append(dim)

    add("1", "int", np.ones_like(x))
    if "scalar" in banks:
        add("log10x", "g", np.log10(x))
        add("inv1px", "g", 1.0 / (1.0 + x))
        add("invsqrtx", "g", 1.0 / np.sqrt(x))
        add("expnegsqrtx", "g", np.exp(-np.sqrt(x)))
        add("tanhlogx", "g", np.tanh(np.log10(x)))
        add("log1px", "g", np.log10(1.0 + x))
    if "radial" in banks:
        add("log10r", "r", np.log10(r))
        add("inv1pr10", "r", 1.0 / (1.0 + r / 10.0))
        add("sqrtr10", "r", np.sqrt(r / 10.0))
    if "gal" in banks:
        add("log10Mb", "gal", np.log10(u.Mb / 1e10))
    if "tensor" in banks:
        c = u.cost
        P2 = 0.5 * (3 * c ** 2 - 1.0)
        P4 = (35 * c ** 4 - 30 * c ** 2 + 3) / 8.0
        add("P2", "tensor", P2)
        add("P4", "tensor", P4)
        add("P2*log10x", "tensor", P2 * np.log10(x))
        add("P2/(1+x)", "tensor", P2 / (1.0 + x))
    if "nonlocal" in banks:
        aa = u.a if shape_override is None else np.asarray(shape_override, float)
        Mloc = np.maximum(_hernquist_M(r, u.Mb, aa), 1.0)
        for L in L_nl:
            Meff = np.maximum(_kernel_smoothed_M(r, u.Mb, aa, L), 1.0)
            add(f"q_nl({L:g}kpc)", "nonlocal", np.log10(Meff / Mloc))
    X, nm = _standardise(cols, names)
    keep = {n: d for n, d in zip(names, dims)}
    return X, nm, [keep[n] for n in nm]


def _split_labels(sysid, frac_train=2 / 3, seed=0):
    """Blind protection: the split is by WHOLE SYSTEM and is drawn from the
    system index alone, so it cannot correlate with any residual."""
    rng = _as_rng(seed)
    s = np.unique(sysid)
    perm = rng.permutation(s.size)
    ntr = int(round(frac_train * s.size))
    lab = np.where(perm < ntr, "train", "blind")
    m = dict(zip(s, lab))
    return np.array([m[i] for i in sysid])


def _frozen_blind_rms(X, Y, split, atoms, *, ridge=1e-8, name=""):
    """One fit on train, freeze, ONE touch of blind. Control 9 all the way."""
    sd = SplitData(X, Y, split, atoms=atoms, name=name)
    mdl = sd.fit(ridge=ridge)
    out = sd.evaluate(mdl, on="blind")
    return out["rms"], sd.train_rms(mdl), sd, mdl


def _gain(rms_base, rms_ext):
    return 100.0 * (np.asarray(rms_base) - np.asarray(rms_ext)) / np.asarray(rms_base)


def detect_structure(u: MockUniverse, *, seed=0, B=99, B_nl=None, alpha=0.05,
                     split_seed=None, tests=("tensor", "nonlocal"),
                     verbose=False) -> dict:
    """Is there a TENSOR or a NONLOCAL effect in this universe?

    Both tests are "does adding this bank of atoms improve the BLIND fit",
    calibrated against a null that destroys exactly the structure being
    claimed and nothing else:

      tensor    control 1 with (system, radius) blocks. The radial profile of
                the residual is held bit-identical and only the assignment of
                residual to ANGLE is permuted. Under H0 the angle carries
                nothing, so the permutation is exact.
      nonlocal  the system-level SHAPE parameter is permuted across systems,
                so the nonlocal atoms keep their marginal distribution and
                lose their link to the system they describe.

    Reports three verdicts of increasing discipline, because the gap between
    them IS the result: `naive` (any improvement), `threshold` (more than 1%),
    and `calibrated` (p <= alpha against the matched null).
    """
    rng = _as_rng(seed)
    B_nl = B if B_nl is None else B_nl
    split = _split_labels(u.sysid, seed=0 if split_seed is None else split_seed)
    y = u.source.value
    Xs, ns, _ = build_bank(u, ("scalar", "radial", "gal"))
    rec = {"law": u.law, "B": B, "B_nl": B_nl, "alpha": alpha,
           "n_train_systems": int(np.unique(u.sysid[split == "train"]).size),
           "n_blind_systems": int(np.unique(u.sysid[split == "blind"]).size)}

    if "tensor" in tests:
        # design fixed, only the target changes -> one batched fit, one touch
        Xt, nt, _ = build_bank(u, ("scalar", "radial", "gal", "tensor"))
        blk = u.block_angular
        Yn, nullrec = residual_null_batch(u.source, rng, B, block=blk,
                                          standardise=True, radius_from=u.r)
        nullrec.check(1e-9)
        Yb = np.column_stack([y, Yn])
        rec["null_invariant_worst"] = nullrec.invariants[
            "block_mean_residual"][2]
        rs_b, rs_tr, _, _ = _frozen_blind_rms(Xs, Yb, split, ns, name="scalar")
        rt_b, _, _, _ = _frozen_blind_rms(Xt, Yb, split, nt,
                                          name="scalar+tensor")
        gT = _gain(rs_b, rt_b)
        pT = float((1 + np.sum(gT[1:] >= gT[0])) / (B + 1))
        rec["rms_blind_scalar"] = float(np.atleast_1d(rs_b)[0])
        rec["rms_train_scalar"] = float(np.atleast_1d(rs_tr)[0])
        rec["tensor"] = {"gain_pct": float(gT[0]),
                         "null_mean_gain_pct": float(gT[1:].mean()),
                         "null_q95_gain_pct": float(np.quantile(gT[1:], 0.95)),
                         "p": pT,
                         "detected_naive": bool(gT[0] > 0),
                         "detected_threshold1pct": bool(gT[0] > 1.0),
                         "detected_calibrated": bool(pT <= alpha)}
        if verbose:
            t = rec["tensor"]
            print(f"      tensor   gain {t['gain_pct']:+6.2f}%  null mean "
                  f"{t['null_mean_gain_pct']:+6.2f}%  q95 "
                  f"{t['null_q95_gain_pct']:+6.2f}%   p={t['p']:.3f}  "
                  f"{'DETECTED' if t['detected_calibrated'] else '-'}")

    if "nonlocal" in tests:
        rs0, _, _, _ = _frozen_blind_rms(Xs, y, split, ns, name="scalar-nl")
        rs0 = float(rs0)
        rec.setdefault("rms_blind_scalar", rs0)
        Xn, nn, _ = build_bank(u, ("scalar", "radial", "gal", "nonlocal"))
        rn_b, _, _, _ = _frozen_blind_rms(Xn, y, split, nn,
                                          name="scalar+nonlocal")
        gNobs = float(_gain(rs0, float(rn_b)))
        gN_list = []
        for b in range(B_nl):
            ap = u.sys_a[rng.permutation(u.sys_a.size)][u.sysid]
            Xnb, nnb, _ = build_bank(u, ("scalar", "radial", "gal", "nonlocal"),
                                     shape_override=ap)
            rb, _, _, _ = _frozen_blind_rms(Xnb, y, split, nnb, name="nl-null")
            gN_list.append(float(_gain(rs0, float(rb))))
        gNn = np.array(gN_list)
        pN = float((1 + np.sum(gNn >= gNobs)) / (B_nl + 1))
        rec["nonlocal"] = {"gain_pct": gNobs,
                           "null_mean_gain_pct": float(gNn.mean()),
                           "null_q95_gain_pct": float(np.quantile(gNn, 0.95)),
                           "p": pN,
                           "detected_naive": bool(gNobs > 0),
                           "detected_threshold1pct": bool(gNobs > 1.0),
                           "detected_calibrated": bool(pN <= alpha)}
        if verbose:
            nl = rec["nonlocal"]
            print(f"      nonlocal gain {nl['gain_pct']:+6.2f}%  null mean "
                  f"{nl['null_mean_gain_pct']:+6.2f}%  q95 "
                  f"{nl['null_q95_gain_pct']:+6.2f}%   p={nl['p']:.3f}  "
                  f"{'DETECTED' if nl['detected_calibrated'] else '-'}")
    return rec


# ------------------------------------------------- parametric family recovery
def _law_pred(law, theta, u: MockUniverse):
    p = dict(LAW_DEFAULTS[law])
    if law == "mond":
        p["a0"] = 10.0 ** theta[0]
    elif law == "gr_dm":
        p["M200_norm"] = 10.0 ** theta[0]; p["M200_slope"] = theta[1]
    elif law == "tensor":
        p["a0"] = 10.0 ** theta[0]; p["eps_T"] = theta[1]
    elif law == "nonlocal":
        p["a0"] = 10.0 ** theta[0]; p["L_kpc"] = 10.0 ** theta[1]
    g, _ = law_g(law, p, r=u.r, cost=u.cost, M=u.Mb, a=u.a, gN=u.gN)
    return np.log10(np.maximum(g, 1e-300) / u.gN), p


_LAW_THETA0 = {"newton": [], "mond": [math.log10(A0_KPC)],
               "gr_dm": [11.3, 0.7], "tensor": [math.log10(A0_KPC), 0.2],
               "nonlocal": [math.log10(A0_KPC), math.log10(6.0)]}
#: free GLOBAL constants each family carries (never per object). The tensor and
#: nonlocal families NEST the MOND family -- eps_T -> 0 and L -> 0 recover it
#: exactly -- so a bare argmin over blind RMS can only ever pick the richer
#: family, by a margin that is pure noise. Parsimony is not a preference here,
#: it is the difference between recovering the law and inventing one.
_LAW_NFREE = {"newton": 0, "mond": 1, "gr_dm": 2, "tensor": 2, "nonlocal": 2}


def recover_family(u: MockUniverse, *, split_seed=0, verbose=False) -> dict:
    """CONTROL 5b. Does the pipeline pick the family that generated the data?

    Every family gets its own GLOBAL constants fitted on the training systems
    (never per object), an intercept and a single amplitude coefficient, and
    is then FROZEN and scored once on the blind systems. The winner is the
    family with the smallest blind RMS.
    """
    split = _split_labels(u.sysid, seed=split_seed)
    tr = split == "train"
    y = u.source.value
    out = {}
    for law in LAW_NAMES:
        th0 = _LAW_THETA0[law]

        def obj(t):
            pred, _ = _law_pred(law, t, u)
            A = np.column_stack([np.ones(tr.sum()), pred[tr]])
            c = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            r = y[tr] - A @ c
            return float(np.sqrt((r ** 2).mean()))
        if th0 and _optimize is not None:
            res = _optimize.minimize(obj, np.array(th0, float),
                                     method="Nelder-Mead",
                                     options=dict(maxiter=400, fatol=1e-8,
                                                  xatol=1e-6))
            th = res.x
        else:
            th = np.array(th0, float)
        pred, p = _law_pred(law, th, u)
        X = np.column_stack([np.ones(u.source.n), pred])
        rms_b, rms_t, sd, mdl = _frozen_blind_rms(
            X, y, split, ("1", f"pred[{law}]"), name=f"family[{law}]")
        out[law] = {"theta": [float(v) for v in th],
                    "params": _j(p), "n_free": _LAW_NFREE[law],
                    "blind_rms": float(rms_b), "train_rms": float(rms_t),
                    "amplitude": float(np.atleast_1d(mdl.coef)[1])}
    n_blind = int((split == "blind").sum())
    argmin = min(out, key=lambda k: out[k]["blind_rms"])
    rbest = out[argmin]["blind_rms"]
    se = rbest / math.sqrt(2.0 * max(n_blind, 1))     # SE of an RMS
    within = [k for k in out if out[k]["blind_rms"] <= rbest + se]
    pars = min(within, key=lambda k: (_LAW_NFREE[k], out[k]["blind_rms"]))
    rec = {"injected": u.law, "recovered": pars,
           "recovered_argmin": argmin,
           "correct": pars == u.law, "correct_argmin": argmin == u.law,
           "one_se": se, "within_one_se": sorted(within),
           "families": out, "n_blind_points": n_blind,
           "margin_dex": float(sorted(v["blind_rms"] for v in out.values())[1]
                               - rbest)}
    if verbose:
        print(f"      family recovery: injected {u.law}, argmin {argmin}, "
              f"parsimonious {pars} ({'OK' if rec['correct'] else 'WRONG'})")
        for k, v in sorted(out.items(), key=lambda kv: kv[1]["blind_rms"]):
            mark = "*" if k in within else " "
            print(f"        {mark}{k:<9} blind {v['blind_rms']:.5f} dex  "
                  f"train {v['train_rms']:.5f}  amp {v['amplitude']:+.3f}  "
                  f"n_free={v['n_free']}")
        print(f"         one-SE band {se:.5f} dex; * = statistically tied "
              f"with the best")
    return rec


def run_discovery(u: MockUniverse, *, seed=0, B=99, B_nl=None, alpha=0.05,
                  verbose=False) -> dict:
    """CONTROL 5. The full pipeline on one mock universe."""
    return {"family": recover_family(u, verbose=verbose),
            "structure": detect_structure(u, seed=seed, B=B, B_nl=B_nl,
                                          alpha=alpha, verbose=verbose)}


def tensor_false_positive_rate(*, n_universes=60, laws=SCALAR_LAWS, B=99,
                               alpha=0.05, seed=0, n_sys=60, n_r=10, n_t=5,
                               n_universes_nl=None, B_nl=49,
                               verbose=True) -> dict:
    """CONTROL 5c. How often does a SCALAR universe produce a tensor claim?

    Three decision rules of increasing discipline are scored on the same
    simulations, because the gap between them is the number that matters:

      naive        the tensor atoms improved the blind RMS at all
      threshold    they improved it by more than 1%
      calibrated   the improvement exceeds its own matched permutation null
                   at alpha

    Whatever the calibrated rate turns out to be IS the false-positive rate a
    future tensor claim has to be read against.
    """
    rng = _as_rng(seed)
    n_nl = n_universes // 3 if n_universes_nl is None else n_universes_nl
    rows, rows_nl = [], []
    for law in laws:
        for i in range(n_universes):
            s = int(rng.integers(0, 2 ** 31 - 1))
            u = synthetic_universe(law, s, n_sys=n_sys, n_r=n_r, n_t=n_t).data
            d = detect_structure(u, seed=s + 1, B=B, alpha=alpha,
                                 tests=("tensor",))
            rows.append({"law": law, "seed": s,
                         "gain_T": d["tensor"]["gain_pct"],
                         "p_T": d["tensor"]["p"],
                         "naive": d["tensor"]["detected_naive"],
                         "thr": d["tensor"]["detected_threshold1pct"],
                         "cal": d["tensor"]["detected_calibrated"]})
            if i < n_nl:
                dn = detect_structure(u, seed=s + 2, B_nl=B_nl, alpha=alpha,
                                      tests=("nonlocal",))
                rows_nl.append({"law": law, "gain_N": dn["nonlocal"]["gain_pct"],
                                "p_N": dn["nonlocal"]["p"],
                                "naive_N": dn["nonlocal"]["detected_naive"],
                                "cal_N": dn["nonlocal"]["detected_calibrated"]})
            if verbose and (i + 1) % 10 == 0:
                print(f"      {law:<8} {i+1}/{n_universes} universes")

    def rate(rws, key, sub=None):
        r = [x[key] for x in rws if sub is None or x["law"] == sub]
        return float(np.mean(r)) if r else float("nan")
    per_law = {l: {"n": sum(1 for x in rows if x["law"] == l),
                   "fpr_naive": rate(rows, "naive", l),
                   "fpr_threshold1pct": rate(rows, "thr", l),
                   "fpr_calibrated": rate(rows, "cal", l),
                   "fpr_nonlocal_calibrated": rate(rows_nl, "cal_N", l),
                   "median_gain_T_pct": float(np.median(
                       [x["gain_T"] for x in rows if x["law"] == l]))}
               for l in laws}
    n = len(rows)
    fc = rate(rows, "cal")
    out = {"n_universes_total": n, "n_universes_nonlocal": len(rows_nl),
           "laws": list(laws), "B": B, "B_nl": B_nl, "alpha": alpha,
           "fpr_naive": rate(rows, "naive"),
           "fpr_threshold1pct": rate(rows, "thr"),
           "fpr_calibrated": fc,
           "fpr_nonlocal_naive": rate(rows_nl, "naive_N"),
           "fpr_nonlocal_calibrated": rate(rows_nl, "cal_N"),
           "fpr_calibrated_se": float(math.sqrt(max(fc * (1 - fc), 0) / n)),
           "median_gain_T_pct": float(np.median([x["gain_T"] for x in rows])),
           "q95_gain_T_pct": float(np.quantile([x["gain_T"] for x in rows], 0.95)),
           "p_uniformity_ks": _ks_uniform([x["p_T"] for x in rows]),
           "per_law": per_law, "rows": rows, "rows_nonlocal": rows_nl}
    if verbose:
        print(f"\n   TENSOR false-positive rate over {n} SCALAR universes "
              f"({', '.join(laws)}):")
        print(f"      naive     (any blind improvement)      "
              f"{100*out['fpr_naive']:6.1f}%")
        print(f"      threshold (> 1% blind improvement)     "
              f"{100*out['fpr_threshold1pct']:6.1f}%")
        print(f"      CALIBRATED (p <= {alpha} vs matched null) "
              f"{100*out['fpr_calibrated']:6.1f}% "
              f"+- {100*out['fpr_calibrated_se']:.1f}%")
        print(f"      median tensor 'gain' on scalar data "
              f"{out['median_gain_T_pct']:+.2f}%   95th pct "
              f"{out['q95_gain_T_pct']:+.2f}%")
        print(f"      KS uniformity of the calibrated p-values: "
              f"D={out['p_uniformity_ks']['D']:.3f}, "
              f"p={out['p_uniformity_ks']['p']:.3f}")
        print(f"   NONLOCAL false-positive rate over {len(rows_nl)} scalar "
              f"universes: naive {100*out['fpr_nonlocal_naive']:.1f}%   "
              f"calibrated {100*out['fpr_nonlocal_calibrated']:.1f}%")
    return out


def _ks_uniform(p):
    p = np.sort(np.asarray(p, float))
    n = p.size
    if n == 0:
        return {"D": float("nan"), "p": float("nan"), "n": 0}
    i = np.arange(1, n + 1)
    D = float(max(np.max(i / n - p), np.max(p - (i - 1) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D
    s = sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam)
            for k in range(1, 100))
    return {"D": D, "p": float(min(1.0, max(0.0, 2 * s))), "n": int(n)}


def known_law_suite(*, seed=0, B=199, B_nl=99, n_sys=60, n_r=10, n_t=5,
                    verbose=True) -> dict:
    """CONTROL 5. One universe per law, both requirements checked on each."""
    rng = _as_rng(seed)
    out = {}
    for law in LAW_NAMES:
        s = int(rng.integers(0, 2 ** 31 - 1))
        if verbose:
            print(f"\n   --- injected law: {law}")
        u = synthetic_universe(law, s, n_sys=n_sys, n_r=n_r, n_t=n_t,
                               verbose=verbose).data
        out[law] = run_discovery(u, seed=s + 1, B=B, B_nl=B_nl,
                                 verbose=verbose)
        out[law]["seed"] = s
    tab = {l: {"recovered": out[l]["family"]["recovered"],
               "correct": out[l]["family"]["correct"],
               "tensor_detected": out[l]["structure"]["tensor"]["detected_calibrated"],
               "nonlocal_detected": out[l]["structure"]["nonlocal"]["detected_calibrated"],
               "p_tensor": out[l]["structure"]["tensor"]["p"],
               "p_nonlocal": out[l]["structure"]["nonlocal"]["p"]}
           for l in out}
    req = {
        "recovery_all_correct": all(tab[l]["correct"] for l in tab),
        "no_tensor_in_scalar": not any(tab[l]["tensor_detected"]
                                       for l in SCALAR_LAWS),
        "no_nonlocal_in_scalar": not any(tab[l]["nonlocal_detected"]
                                         for l in SCALAR_LAWS),
        "tensor_found_in_tensor": tab["tensor"]["tensor_detected"],
        "nonlocal_found_in_nonlocal": tab["nonlocal"]["nonlocal_detected"],
    }
    if verbose:
        print("\n   summary")
        print("      injected    recovered   tensor?  p_T     nonlocal?  p_N")
        for l in LAW_NAMES:
            t = tab[l]
            print(f"      {l:<11} {t['recovered']:<11} "
                  f"{'YES' if t['tensor_detected'] else ' no':<8} "
                  f"{t['p_tensor']:.3f}   "
                  f"{'YES' if t['nonlocal_detected'] else ' no':<9} "
                  f"{t['p_nonlocal']:.3f}")
        for k, v in req.items():
            print(f"      {k:<28} {'PASS' if v else 'FAIL'}")
    return {"per_law": out, "table": tab, "requirements": req}
