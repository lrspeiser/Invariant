"""Candidate field laws A-E, written once, generically, so the screen can run
them.

Every candidate exposes the same three things:

  * ``dim_kernel(xp)``    -- the scalar core of the law, evaluated with either
                            numpy or `dimx`, so dimensional consistency is
                            DECIDED BY EXECUTION rather than by assertion.
  * ``K_at(points, ctx)`` -- the response tensor at arbitrary points, as an
                            array of shape (P, 3, 3). This is the object the
                            covariance, positive-definiteness, permutation and
                            coarse-graining screens act on.
  * ``spec``              -- name, family letter, global parameters with units.

Units are SI throughout (metres, kilograms, seconds); kpc and Msun appear only
where a parameter is naturally quoted that way, and are converted immediately.

A note on what a "well" is. Families C and D are sums over a discrete list of
wells -- rows of a catalogue. The mass distribution is a separate object: the
source density rho that appears on the right of the field equation. The screen
deliberately keeps these two apart, because the whole question in Stage 1b is
whether a law's prediction depends on how a FIXED smooth rho was chopped into
rows. So every coarse-graining test holds rho fixed and varies only the well
list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

import dimx
from dimx import Q, DIMLESS, MASS, LENGTH, ACCEL, POTENTIAL, TIDAL, _dim

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
A0 = 1.2e-10                     # m s^-2
C_LIGHT = 2.99792458e8

# ---------------------------------------------------------------- backend
try:                                                   # pragma: no cover
    import cupy as _cp
    _cp.zeros(1)
    HAVE_GPU = True
except Exception:                                      # pragma: no cover
    _cp = None
    HAVE_GPU = False


def xpof(a):
    return _cp if (HAVE_GPU and _cp is not None and isinstance(a, _cp.ndarray)) else np


def asnumpy(a):
    return _cp.asnumpy(a) if (HAVE_GPU and isinstance(a, _cp.ndarray)) else np.asarray(a)


GPU_FALLBACKS = []


def gpu_guard(fn):
    """Fall back to numpy if the GPU is out of memory.

    This machine runs several lanes at once, so a CuPy allocation failure is a
    scheduling accident, not a property of the candidate.  Falling back keeps
    the numbers identical (float64 either way) and records that it happened,
    rather than letting a transient OOM be recorded as a screen failure.
    """
    import functools

    @functools.wraps(fn)
    def wrap(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:                          # noqa: BLE001
            mod = type(e).__module__ or ""
            if _cp is None or not mod.startswith("cupy"):
                raise
            GPU_FALLBACKS.append(f"{fn.__name__}: {type(e).__name__}: {e}"[:200])
            kw["use_gpu"] = False
            return fn(*a, **kw)
    return wrap


# ============================================================ interpolations
def mu_simple(X, xp=np):
    return X / (1.0 + X)


def mu_standard(X, xp=np):
    return X / xp.sqrt(1.0 + X * X)


def nu_simple(y, xp=np):
    """The QUMOND partner of mu_simple: g = nu(g_N/a0) g_N."""
    return 0.5 * (1.0 + xp.sqrt(1.0 + 4.0 / y))


def nu_rar(y, xp=np):
    """RAR / 'exponential' nu.  g = g_N / (1 - exp(-sqrt(g_N/a0)))."""
    return 1.0 / (1.0 - xp.exp(-xp.sqrt(y)))


# ============================================================ weight kernels
# These are the functions the dimensional screen runs with xp = dimx.  They may
# only use +, -, *, /, **, and xp.exp / xp.sqrt, all of which both numpy and
# dimx provide.

def weight_C(M_a, r_a, gN, prm, xp=np):
    """Family C well weight.  prm['shape'] selects one of the three forms
    written down in the brief."""
    u = (M_a / prm["M0"]) ** prm["p"]
    x = r_a / prm["L"]
    shape = prm.get("shape", "pow")
    if shape == "pow":
        return u * (1.0 + x ** prm["q"]) ** (-prm["s"])
    if shape == "exp":
        return u * xp.exp(-(x ** prm["q"]))
    if shape == "pow_g":
        return u / ((1.0 + (gN / prm["a0"]) ** prm["m"])
                    * (1.0 + x ** prm["q"]) ** prm["s"])
    raise ValueError(shape)


def weight_D(M_a, M_b, d_ab, prm, xp=np):
    """Family D pair weight."""
    u = (M_a * M_b / (prm["M0"] * prm["M0"])) ** prm["p"]
    x = d_ab / prm["L"]
    return u * x ** (-prm["q"]) * xp.exp(-(x ** prm["s"]))


def A0_depth(gN, Phi, prm, xp=np):
    """Family B: the acceleration scale itself depends on the well depth."""
    return prm["a0"] * (1.0 + (abs(Phi) / prm["Phi0"]) ** prm["b"]) ** prm["c"]


# ============================================================ candidate spec
@dataclass
class Candidate:
    name: str
    family: str                      # 'A'..'E'
    kind: str                        # 'scalar_mu'|'qumond'|'depth'|'wells'|'pairs'|'tidal'
    prm: Dict[str, float]
    prm_dims: Dict[str, tuple]
    note: str = ""
    reciprocal_by_construction: bool = False
    momentum_carrier: str = ""       # must be non-empty if not reciprocal

    def copy_with(self, **kw):
        p = dict(self.prm)
        p.update(kw.pop("prm", {}))
        d = dict(self.__dict__)
        d["prm"] = p
        d.update(kw)
        d.pop("__dict__", None)
        return Candidate(**{k: v for k, v in d.items()
                            if k in Candidate.__dataclass_fields__})


# --------------------------------------------------------- default parameters
def _pC(**kw):
    p = dict(M0=1e10 * MSUN, p=1.0, L=10.0 * KPC, q=2.0, s=1.0, m=2.0,
             a0=A0, s0=0.0, sT=0.5, eps=1e-12, shape="pow",
             r_soft=0.01 * KPC)
    p.update(kw)
    return p


_DIMS_C = dict(M0=MASS, p=DIMLESS, L=LENGTH, q=DIMLESS, s=DIMLESS, m=DIMLESS,
               a0=ACCEL, s0=DIMLESS, sT=DIMLESS, eps=DIMLESS, r_soft=LENGTH)


def _pD(**kw):
    p = dict(M0=1e10 * MSUN, p=1.0, L=10.0 * KPC, q=1.0, s=2.0,
             sigma_perp=2.0 * KPC, sigma_par=5.0 * KPC, alpha=0.3,
             d_soft=0.02 * KPC)
    p.update(kw)
    return p


_DIMS_D = dict(M0=MASS, p=DIMLESS, L=LENGTH, q=DIMLESS, s=DIMLESS,
               sigma_perp=LENGTH, sigma_par=LENGTH, alpha=DIMLESS,
               d_soft=LENGTH)


def _pE(**kw):
    p = dict(f0=0.0, fT=0.5, eps_T=A0 / (10.0 * KPC))
    p.update(kw)
    return p


_DIMS_E = dict(f0=DIMLESS, fT=DIMLESS, eps_T=TIDAL)


CANDIDATES: Dict[str, Candidate] = {}


def _reg(c: Candidate):
    CANDIDATES[c.name] = c
    return c


# ---- A -------------------------------------------------------------------
_reg(Candidate("A1_aqual_simple", "A", "scalar_mu",
               dict(a0=A0, form="simple"), dict(a0=ACCEL),
               "div[mu(|grad Phi|/a0) grad Phi] = 4 pi G rho, mu = X/(1+X)",
               reciprocal_by_construction=True))
_reg(Candidate("A2_qumond_simple", "A", "qumond",
               dict(a0=A0, form="simple"), dict(a0=ACCEL),
               "lap Psi = div[nu(|grad Phi_N|/a0) grad Phi_N], nu from mu_simple",
               reciprocal_by_construction=True))
_reg(Candidate("A3_qumond_rar", "A", "qumond",
               dict(a0=A0, form="rar"), dict(a0=ACCEL),
               "QUMOND with the RAR nu", reciprocal_by_construction=True))

# ---- B -------------------------------------------------------------------
_reg(Candidate("B1_depth_mond", "B", "depth",
               dict(a0=A0, Phi0=1.0e10, b=1.0, c=1.0),
               dict(a0=ACCEL, Phi0=POTENTIAL, b=DIMLESS, c=DIMLESS),
               "A_0 = a0 [1 + (|Phi|/Phi_0)^b]^c, then g = nu(g_N/A_0) g_N",
               reciprocal_by_construction=False,
               momentum_carrier=""))
_reg(Candidate("B2_depth_mond_weak", "B", "depth",
               dict(a0=A0, Phi0=1.0e12, b=1.0, c=0.5),
               dict(a0=ACCEL, Phi0=POTENTIAL, b=DIMLESS, c=DIMLESS),
               "same, with a much deeper Phi_0 so the depth term is weak",
               reciprocal_by_construction=False))

# ---- C -------------------------------------------------------------------
_reg(Candidate("C1_wells_pow_p1", "C", "wells", _pC(shape="pow", p=1.0),
               _DIMS_C, "w = (M/M0)^1 [1+(r/L)^q]^-s"))
_reg(Candidate("C2_wells_pow_p05", "C", "wells", _pC(shape="pow", p=0.5),
               _DIMS_C, "w = (M/M0)^0.5 [1+(r/L)^q]^-s"))
_reg(Candidate("C3_wells_exp_p1", "C", "wells", _pC(shape="exp", p=1.0, q=1.0),
               _DIMS_C, "w = (M/M0)^1 exp[-(r/L)^q]"))
_reg(Candidate("C4_wells_gsupp_p1", "C", "wells",
               _pC(shape="pow_g", p=1.0), _DIMS_C,
               "w = (M/M0)^1 / {[1+(gN/a0)^m][1+(r/L)^q]^s}"))
_reg(Candidate("C5_wells_pow_p2", "C", "wells", _pC(shape="pow", p=2.0),
               _DIMS_C, "w = (M/M0)^2 [1+(r/L)^q]^-s"))

# ---- D -------------------------------------------------------------------
_reg(Candidate("D1_pairs_p1_q1", "D", "pairs", _pD(p=1.0, q=1.0), _DIMS_D,
               "w_ab = (Ma Mb/M0^2)^1 (d/L)^-1 exp[-(d/L)^2]"))
_reg(Candidate("D2_pairs_p05_q1", "D", "pairs", _pD(p=0.5, q=1.0), _DIMS_D,
               "w_ab = (Ma Mb/M0^2)^0.5 (d/L)^-1 exp[-(d/L)^2]"))
_reg(Candidate("D3_pairs_p1_q3", "D", "pairs", _pD(p=1.0, q=3.0), _DIMS_D,
               "w_ab with q = 3, the marginal case for the pair integral"))

# ---- E -------------------------------------------------------------------
_reg(Candidate("E1_tidal", "E", "tidal", _pE(), _DIMS_E,
               "K = exp[f0 I + fT That], That the normalised tidal tensor"))
_reg(Candidate("E2_tidal_strong", "E", "tidal", _pE(fT=1.5), _DIMS_E,
               "same with a stronger tidal coupling"))


# ============================================================ geometry
class Box:
    """Cartesian box wrapping ``solver.grids``, with SI coordinates."""

    def __init__(self, n: int, L_kpc: float):
        import solver as S
        self.n = n
        self.L = L_kpc * KPC
        self.h, self.ax, self.X, self.Y, self.Z = S.grids(n, self.L)
        self.shape = self.X.shape
        self.pts = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], 1)
        self.r = np.sqrt(self.X ** 2 + self.Y ** 2 + self.Z ** 2)
        self.vol = self.h ** 3


def plummer_rho(pts, M, a, centre=(0, 0, 0)):
    c = np.asarray(centre, float)
    r2 = np.sum((pts - c) ** 2, axis=-1)
    return 3 * M / (4 * np.pi * a ** 3) * (1 + r2 / a ** 2) ** -2.5


def gauss_rho(pts, M, sig, centre=(0, 0, 0)):
    c = np.asarray(centre, float)
    r2 = np.sum((pts - c) ** 2, axis=-1)
    return M / (2 * np.pi * sig ** 2) ** 1.5 * np.exp(-r2 / (2 * sig ** 2))


def expdisk_rho(pts, M, Rd, hz):
    R = np.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
    z = pts[..., 2]
    Sig0 = M / (2 * np.pi * Rd ** 2)
    return Sig0 * np.exp(-R / Rd) / (2 * hz) / np.cosh(z / hz) ** 2


def normalise_mass(rho, vol, M):
    tot = rho.sum() * vol
    assert tot > 0, "empty density"
    return rho * (M / tot)


# ============================================================ clouds
def _fib_dirs(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    rxy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], 1)


def equal_mass_cloud(kind, Nq, M, seed=20260903, **kw):
    """A deterministic, low-discrepancy cloud of Nq points of EQUAL mass M/Nq.

    This is the reference mass distribution for the coarse-graining test.  It
    has to be equal-mass, because otherwise a greedy equal-mass partition
    cannot balance below the mass of a single quadrature point and the
    resulting 'refinement' silently stops refining -- a failure that looks
    exactly like a converged law.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(Nq) + 0.5) / Nq
    if kind == "plummer":
        a = kw["a"]
        u = u * kw.get("umax", 0.995)
        r = a * u ** (1 / 3) / np.sqrt(np.maximum(1 - u ** (2 / 3), 1e-12))
        r = r[rng.permutation(Nq)]
        x = _fib_dirs(Nq) * r[:, None]
    elif kind == "expdisk":
        Rd, hz = kw["Rd"], kw["hz"]
        t = np.linspace(0, 12.0, 20000)
        cdf = 1.0 - (1.0 + t) * np.exp(-t)
        R = np.interp(u, cdf, t) * Rd
        R = R[rng.permutation(Nq)]
        uz = (np.arange(Nq) + 0.5) / Nq
        z = hz * np.arctanh(np.clip(2 * uz - 1, -0.9999, 0.9999))
        z = z[rng.permutation(Nq)]
        phi = 2 * np.pi * ((np.arange(Nq) * 0.6180339887498949) % 1.0)
        x = np.stack([R * np.cos(phi), R * np.sin(phi), z], 1)
    elif kind == "shell":
        x = _fib_dirs(Nq) * kw["r"]
    else:
        raise ValueError(kind)
    x = x + np.asarray(kw.get("centre", (0.0, 0.0, 0.0)), float)
    m = np.full(Nq, M / Nq)
    return x, m


# ============================================================ partitions
def nested_partitions(pts, m, Ns):
    """One greedy pass producing SNAPSHOTS at every requested N.

    Snapshots at increasing N are genuine refinements of one another: cell k of
    the N-partition is a union of cells of the N'-partition for every N' > N.
    """
    Ns = sorted(set(int(x) for x in Ns))
    pts = np.asarray(pts, float)
    m = np.asarray(m, float)
    cells = [np.arange(len(m))]
    masses = [float(m.sum())]
    out = {}
    target = list(Ns)
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
            cut = int(np.searchsorted(cw, cw[-1] / 2.0)) + 1
            cut = min(max(cut, 1), len(order) - 1)
            a, b = idx[order[:cut]], idx[order[cut:]]
            cells[k], masses[k] = a, float(m[a].sum())
            cells.append(b)
            masses.append(float(m[b].sum()))
        N = target.pop(0)
        good = [c for c in cells if len(c)]
        wx = np.array([(pts[c] * m[c, None]).sum(0) / m[c].sum() for c in good])
        wm = np.array([m[c].sum() for c in good])
        out[N] = (wx, wm)
    return out


def equal_mass_partition(pts, m, N, rng=None):
    """Split a quadrature cloud into N nested cells of near-equal mass.

    Greedy: repeatedly take the heaviest cell and cut it at the mass median of
    its widest axis.  Successive N are therefore genuine REFINEMENTS of one
    another, which is what "partition refinement" means; a fresh random
    resampling at each N would confound refinement with sampling noise.

    Returns (well positions (N,3), well masses (N,)).
    """
    pts = np.asarray(pts, float)
    m = np.asarray(m, float)
    cells = [np.arange(len(m))]
    masses = [float(m.sum())]
    while len(cells) < N:
        k = int(np.argmax(masses))
        idx = cells[k]
        if len(idx) < 2:
            masses[k] = -1.0                      # cannot split further
            if all(x < 0 for x in masses):
                break
            continue
        P = pts[idx]
        w = m[idx]
        ext = P.max(0) - P.min(0)
        ax = int(np.argmax(ext))
        order = np.argsort(P[:, ax], kind="stable")
        cw = np.cumsum(w[order])
        half = cw[-1] / 2.0
        cut = int(np.searchsorted(cw, half)) + 1
        cut = min(max(cut, 1), len(order) - 1)
        a, b = idx[order[:cut]], idx[order[cut:]]
        cells[k] = a
        masses[k] = float(m[a].sum())
        cells.append(b)
        masses.append(float(m[b].sum()))
    wx = np.array([(pts[c] * m[c, None]).sum(0) / m[c].sum() for c in cells])
    wm = np.array([m[c].sum() for c in cells])
    return wx, wm


def check_partition(wx, wm, Mtot, tol=1e-10):
    """Row-count and mass-conservation assertions (silent-extraction guard)."""
    assert wx.ndim == 2 and wx.shape[1] == 3, f"bad well array {wx.shape}"
    assert wm.shape == (wx.shape[0],), f"mass/pos mismatch {wm.shape} {wx.shape}"
    rel = abs(wm.sum() - Mtot) / Mtot
    assert rel < tol, f"partition lost mass: rel error {rel:.3e}"
    assert np.all(np.isfinite(wx)) and np.all(wm > 0)
    return rel


# ============================================================ tensor builders
class SingularK(ValueError):
    """The response tensor collapsed to zero or blew up.

    For family D at p < 1 the pair tensor C grows like N^(2-2p), so
    K = exp[-alpha C] underflows to the zero matrix once the catalogue is fine
    enough.  That is a real property of the candidate and has to be reported as
    such -- not allowed to reach LAPACK as a matrix of zeros or NaNs, where it
    would look like a solver crash.
    """


EXP_CLIP = 700.0            # exp(700) ~ 1e304, the edge of float64


def _sym_expm(Msym, tag=""):
    """Matrix exponential of a stack of symmetric 3x3 matrices, (P,3,3)."""
    xp = xpof(Msym)
    if not bool(xp.all(xp.isfinite(Msym))):
        raise SingularK(f"non-finite argument to exp[.]{' ' + tag if tag else ''}")
    w, V = xp.linalg.eigh(Msym)
    wmin, wmax = float(w.min()), float(w.max())
    if wmin < -EXP_CLIP or wmax > EXP_CLIP:
        raise SingularK(
            f"exp[.] argument spans [{wmin:.4g}, {wmax:.4g}]{' ' + tag if tag else ''}"
            f": K would underflow/overflow float64. The response tensor has "
            f"collapsed, so the field equation has no bounded solution")
    return xp.einsum("pij,pj,pkj->pik", V, xp.exp(w), V)


@gpu_guard
def S_wells(points, wx, wm, prm, gN=None, block=1024, pblock=8192,
            use_gpu=None):
    """Family-C alignment tensor S at ``points`` (P,3).  Returns (P,3,3).

    Chunked over BOTH points and wells: the (P, B, 3) intermediate is the
    memory bottleneck once P is a 48^3 grid and B is 10^4 catalogue rows.
    """
    use_gpu = HAVE_GPU if use_gpu is None else (use_gpu and HAVE_GPU)
    xp = _cp if use_gpu else np
    points = np.asarray(points, float)
    WX = xp.asarray(wx, dtype=xp.float64)
    WM = xp.asarray(wm, dtype=xp.float64)
    gA = None if gN is None else np.asarray(gN, float)
    out = np.empty((len(points), 3, 3))
    eye = xp.eye(3)[None, :, :]
    for s in range(0, len(points), pblock):
        P = xp.asarray(points[s:s + pblock])
        g = None if gA is None else xp.asarray(gA[s:s + pblock])
        num = xp.zeros((P.shape[0], 3, 3))
        den = xp.zeros(P.shape[0])
        sw = xp.zeros(P.shape[0])
        for i in range(0, WX.shape[0], block):
            d = WX[None, i:i + block, :] - P[:, None, :]      # (p,B,3)
            r = xp.sqrt((d * d).sum(-1))
            r = xp.maximum(r, prm["r_soft"])
            n = d / r[..., None]
            w = weight_C(WM[None, i:i + block], r,
                         None if g is None else g[:, None], prm, xp=xp)
            num += xp.einsum("pb,pbi,pbj->pij", w, n, n)
            sw += w.sum(1)
            den += xp.abs(w).sum(1)
        S = (num - sw[:, None, None] * eye / 3.0) \
            / (prm["eps"] + den)[:, None, None]
        out[s:s + pblock] = asnumpy(S) if use_gpu else S
    return out


def S_continuum(points, qx, qm, prm, gN=None, block=4096, use_gpu=None):
    """The N -> infinity target of family C at p = 1: the mass-weighted
    integral of (n n^T - I/3) against rho.  Computed by direct quadrature on
    the same cloud the partitions were cut from, so it is an INDEPENDENT
    reference and not a re-use of any partition."""
    pr = dict(prm)
    pr["p"] = 1.0
    return S_wells(points, qx, qm, pr, gN=gN, block=block, use_gpu=use_gpu)


@gpu_guard
def C_pairs(points, wx, wm, prm, pair_block=100_000, pblock=2048,
            use_gpu=None, max_pairs=40_000_000):
    """Family-D pair-channel tensor C at ``points`` (P,3).  Returns (P,3,3).

    Cost is O(P * N^2).  ``max_pairs`` is a hard guard and the screen REPORTS
    the refusal rather than silently subsampling, because 'this law cannot be
    evaluated on a catalogue of realistic size' is itself a result.
    """
    use_gpu = HAVE_GPU if use_gpu is None else (use_gpu and HAVE_GPU)
    xp = _cp if use_gpu else np
    N = len(wm)
    npair = N * (N - 1) // 2
    if npair > max_pairs:
        raise MemoryError(f"family D needs {npair:,} pairs for N={N} rows; "
                          f"screen limit {max_pairs:,}")
    points = np.asarray(points, float)
    ia, ib = np.triu_indices(N, 1)
    WX = xp.asarray(wx, dtype=xp.float64)
    WM = xp.asarray(wm, dtype=xp.float64)
    out = np.zeros((len(points), 3, 3))
    for s in range(0, len(points), pblock):
        P = xp.asarray(points[s:s + pblock])
        acc = xp.zeros((P.shape[0], 3, 3))
        for i in range(0, npair, pair_block):
            A = xp.asarray(ia[i:i + pair_block])
            B = xp.asarray(ib[i:i + pair_block])
            xa, xb = WX[A], WX[B]
            dv = xb - xa
            dab = xp.maximum(xp.sqrt((dv * dv).sum(-1)), prm["d_soft"])
            e = dv / dab[:, None]
            w = weight_D(WM[A], WM[B], dab, prm, xp=xp)
            mid = 0.5 * (xa + xb)
            rel = P[:, None, :] - mid[None, :, :]              # (p,K,3)
            dpar = (rel * e[None, :, :]).sum(-1)
            dperp2 = xp.maximum((rel * rel).sum(-1) - dpar * dpar, 0.0)
            W = xp.exp(-dperp2 / (2 * prm["sigma_perp"] ** 2)) \
                * xp.exp(-dpar ** 2 / (2 * prm["sigma_par"] ** 2))
            acc += xp.einsum("pk,ki,kj->pij", W * w[None, :], e, e)
        out[s:s + pblock] = asnumpy(acc) if use_gpu else acc
    return out


def K_from_S(S, prm):
    xp = xpof(S)
    M = prm["s0"] * xp.eye(3)[None] + prm["sT"] * S
    return _sym_expm(M, "K = exp[s0 I + sT S]")


def K_from_C(C, prm):
    xp = xpof(C)
    return _sym_expm(-prm["alpha"] * C, "K = exp[-alpha C]")


def K_from_T(That, prm):
    xp = xpof(That)
    M = prm["f0"] * xp.eye(3)[None] + prm["fT"] * That
    return _sym_expm(M, "K = exp[f0 I + fT That]")


def tidal_hat(Phi, h, prm):
    """Normalised traceless tidal tensor from a potential on a (n,n,n) grid.

    Returns (P,3,3) flattened in C order to match ``Box.pts``.
    """
    gx, gy, gz = np.gradient(Phi, h, edge_order=2)
    T = np.empty(Phi.shape + (3, 3))
    for i, gi in enumerate((gx, gy, gz)):
        d = np.gradient(gi, h, edge_order=2)
        for j in range(3):
            T[..., i, j] = d[j]
    T = 0.5 * (T + np.swapaxes(T, -1, -2))
    tr = np.trace(T, axis1=-2, axis2=-1)
    T0 = T - tr[..., None, None] * np.eye(3) / 3.0
    nrm = np.sqrt(prm["eps_T"] ** 2 + (T0 * T0).sum((-1, -2)))
    That = T0 / nrm[..., None, None]
    return That.reshape(-1, 3, 3)


def pack_A(K):
    """(P,3,3) -> the solver's 6-tuple of (n,n,n) component arrays."""
    n = round(len(K) ** (1 / 3))
    assert n ** 3 == len(K), f"{len(K)} points is not a cube"
    sh = (n, n, n)
    return (K[:, 0, 0].reshape(sh), K[:, 1, 1].reshape(sh),
            K[:, 2, 2].reshape(sh), K[:, 0, 1].reshape(sh),
            K[:, 0, 2].reshape(sh), K[:, 1, 2].reshape(sh))


def iso_A(shape, val=1.0):
    o = np.full(shape, float(val))
    z = np.zeros(shape)
    return o, o.copy(), o.copy(), z, z.copy(), z.copy()
