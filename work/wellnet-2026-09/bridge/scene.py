"""scene.py -- the two-concentration scene, and every law's effective density on it.

THE SCENE (declared; nothing observational)

    two Plummer concentrations A (mass M_A, at x = -D/2) and B (M_B, at +D/2),
    common scale radius a, an optional intervening filament of axis density
    rho_f and transverse radius R_f represented as a chain of Plummer beads
    (spacing R_f, so the axial density ripple is below 0.3%), everything
    axisymmetric about the pair axis x.  Fields are tabulated on an (x, R)
    grid and projected to the sky at any inclination of the axis.

THE LAWS, as EFFECTIVE DENSITY rho_eff = lap(Phi)/(4 pi G) (what lensing sees)

    newton   rho_eff = rho
    P        rho_eff = rho + lap(Phi_dir^eps + Phi_3)/(4 pi G)
             Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z),  P = Int dOmega C(n) C(-n)
             Phi_dir^eps(z) = -G eps Int rho(y) v(z,y)/|z-y|  (the endpoint term's
             path modification; its two-body CROSS part is evaluated with each
             source group as a point source -- declared, and bounded in
             `endpoint_cross_check`)
    qumond   rho_eff = rho + rho_ph,  rho_ph = div[(nu-1) grad Phi_N]/(4 pi G)
             in CLOSED FORM from the Plummer Hessians (no finite differences):
             rho_ph = (nu-1) rho - nu'(y) g^T J g /(4 pi G a0 |g|),  J = dg/dx
    tensor   BL's action, as the declared caricature: the QUMOND-form base with
             the simple mu, plus each body's EXACT first-order l=2 response
             (tensor_family.solve_l2) with the axis along the pair line
    cdm      rho_eff = rho + rho_DM: spherical haloes are already inside the
             declared Plummers (the scene's Plummers are each law's total lens
             mass), so the competitor's bridge is a filament of POSITIVE dark
             mass; the triaxial-halo attack lives in `cdm_attack.py`

P(z) is stored PAIRWISE by source group (AA, BB, FF, AB, AF, BF) so the whole
(M_A, M_B, rho_f) ladder is exact bilinear rescaling of six grids, and the
M_A M_B scaling of the observable is then MEASURED on the projected feature
rather than assumed from the algebra (the phi'(rho) factor and the Laplacian
do not rescale).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import guard                                    # noqa: F401  (sys.path)
import compiler as C                            # noqa: E402

G, KPC, MPC, MSUN, A0 = C.G, C.KPC, C.MPC, C.MSUN, C.A0
Plummer = C.Plummer

RHO_STAR_FID = 1.0e-24                 # kg m^-3, BL's fiducial
RHO_MEAN = 0.315 * 9.204e-27           # BL's no-new-scale variant (mean matter)
RHO_B_MEAN = 0.0493 * 9.204e-27        # cosmic mean BARYON density (reference)
EPS_LADDER = (0.3, 0.03, 0.003)
RHO_STAR_LADDER = {"1e-24": RHO_STAR_FID, "rho_mean": RHO_MEAN}

# BL's declared bridge scene (path_family.bridge): two 3e14 Msun Plummers,
# a = 400 kpc, 4 Mpc apart, vacuum between.  Kept as the fiducial so every
# number here can be set beside BL's.
FID = dict(MA=3.0e14 * MSUN, MB=3.0e14 * MSUN, a=400.0 * KPC, D=4.0 * MPC,
           rho_f=0.0, R_f=250.0 * KPC)

_GL_T, _GL_W = np.polynomial.legendre.leggauss(48)
_GL_T = 0.5 * (_GL_T + 1.0)
_GL_W = 0.5 * _GL_W


def fib_dirs(n: int) -> np.ndarray:
    return C.fib_dirs(n)


# ============================================================ scene geometry
def filament_beads(D: float, a: float, rho_f: float, R_f: float
                   ) -> List[Plummer]:
    """A chain of Plummer beads along the axis between the two cores.

    Bead scale radius R_f, spacing Delta = R_f (axial ripple < 0.3%), mass
    m_b = rho_f pi R_f^2 Delta so the AXIS density of the chain is rho_f
    (a chain of Plummers of spacing Delta has axis density m_b/(pi R_f^2
    Delta) exactly in the continuum limit).  The chain spans the segment
    between the two cores' scale radii, |x| <= D/2 - a."""
    if rho_f <= 0.0:
        return []
    L = D - 2.0 * a
    n = max(int(round(L / R_f)), 1)
    delta = L / n
    m_b = rho_f * math.pi * R_f ** 2 * delta
    xs = -0.5 * L + (np.arange(n) + 0.5) * delta
    return [Plummer(m_b, R_f, (float(x), 0.0, 0.0)) for x in xs]


@dataclass
class TwoBody:
    """The declared scene: groups of Plummers, keyed A, B, F."""
    MA: float = FID["MA"]
    MB: float = FID["MB"]
    a: float = FID["a"]
    D: float = FID["D"]
    rho_f: float = FID["rho_f"]
    R_f: float = FID["R_f"]
    groups: Dict[str, List[Plummer]] = field(default_factory=dict)

    def __post_init__(self):
        self.groups = {
            "A": [Plummer(self.MA, self.a, (-0.5 * self.D, 0.0, 0.0))],
            "B": [Plummer(self.MB, self.a, (+0.5 * self.D, 0.0, 0.0))],
            "F": filament_beads(self.D, self.a, self.rho_f, self.R_f),
        }

    @property
    def comps(self) -> List[Plummer]:
        return [c for g in ("A", "B", "F") for c in self.groups[g]]

    def group_mass(self, g: str) -> float:
        return float(sum(c.M for c in self.groups[g]))

    def rho_group(self, g: str, pts) -> np.ndarray:
        pts = np.asarray(pts, float)
        out = np.zeros(pts.shape[:-1])
        for c in self.groups[g]:
            out = out + c.rho(pts)
        return out

    def rho(self, pts, scale: Optional[Dict[str, float]] = None) -> np.ndarray:
        scale = scale or {}
        pts = np.asarray(pts, float)
        out = np.zeros(pts.shape[:-1])
        for g in ("A", "B", "F"):
            if self.groups[g]:
                out = out + scale.get(g, 1.0) * self.rho_group(g, pts)
        return out

    def gN(self, pts, scale: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Newtonian acceleration vector (P,3)."""
        scale = scale or {}
        pts = np.asarray(pts, float)
        out = np.zeros(pts.shape)
        for g in ("A", "B", "F"):
            for c in self.groups[g]:
                out = out + scale.get(g, 1.0) * c.g(pts)
        return out

    def jacN(self, pts, scale: Optional[Dict[str, float]] = None) -> np.ndarray:
        """J_ij = d g_i / d x_j = -Hess(Phi_N), (P,3,3)."""
        scale = scale or {}
        pts = np.asarray(pts, float)
        out = np.zeros(pts.shape + (3,))
        for g in ("A", "B", "F"):
            for c in self.groups[g]:
                out = out - scale.get(g, 1.0) * c.hess(pts)
        return out


# ============================================ closed-form half-line column
def half_column(comp: Plummer, z: np.ndarray, n: np.ndarray) -> np.ndarray:
    """C(z, n) = Int_0^inf rho(z + s n) ds for one Plummer sphere.

    Derived here (independently of synthesis/path_family.py, and tested
    against it and against quadrature in test_bridge.py).  With x = z - c,
    p = x.n and b^2 = |x|^2 - p^2 + a^2,  |z + s n - c|^2 + a^2 = (s+p)^2 + b^2,
    so

        C = (3 M a^2 / 4 pi) Int_p^inf du (u^2 + b^2)^(-5/2)
          = (3 M a^2 / 4 pi) [ 2/(3 b^4) - p (2p^2 + 3b^2) / (3 b^4 (p^2+b^2)^{3/2}) ].

    For p > 30 b the bracket is the difference of two O(1/b^4) numbers, so the
    far tail uses the series 1/(4p^4) - 5 b^2/(12 p^6) + 35 b^4/(64 p^8)
    (relative truncation error < (b/p)^6 ~ 1e-9).  z: (...,3), n: (...,3) unit,
    broadcastable."""
    z = np.asarray(z, float)
    n = np.asarray(n, float)
    x = z - comp.c
    p = (x * n).sum(-1)
    b2 = (x * x).sum(-1) - p * p + comp.a ** 2
    b2 = np.maximum(b2, 1e-12 * comp.a ** 2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exact = (2.0 / (3.0 * b2 * b2)
                 - p * (2.0 * p * p + 3.0 * b2)
                 / (3.0 * b2 * b2 * (p * p + b2) ** 1.5))
        pp = np.maximum(p, 1e-300)
        series = (1.0 / (4.0 * pp ** 4) - 5.0 * b2 / (12.0 * pp ** 6)
                  + 35.0 * b2 * b2 / (64.0 * pp ** 8))
    far = p > 30.0 * np.sqrt(b2)
    I = np.where(far, series, exact)
    return (3.0 * comp.M * comp.a ** 2 / (4.0 * math.pi)) * I


def group_columns(scene: TwoBody, pts: np.ndarray, dirs: np.ndarray
                  ) -> Dict[str, np.ndarray]:
    """C_g(z, n) for every group g: dict of (P, D) arrays."""
    Z = pts[:, None, :]
    N = dirs[None, :, :]
    out = {}
    for g in ("A", "B", "F"):
        acc = np.zeros((len(pts), len(dirs)))
        for comp in scene.groups[g]:
            acc += half_column(comp, Z, N)
        out[g] = acc
    return out


PAIRS = ("AA", "BB", "FF", "AB", "AF", "BF")


def pairwise_P(scene: TwoBody, pts: np.ndarray, n_dir: int = 4000,
               chunk: int = 192) -> Dict[str, np.ndarray]:
    """P_gh(z) = Int dOmega [C_g(n) C_h(-n) + (g<->h if g != h)].

    Sum over pairs gives P(z) = Int dOmega C(n) C(-n) for the full scene; each
    P_gh is bilinear in the group masses, which is what makes the mass and
    filament ladders exact rescalings.  Fibonacci directions with equal
    weights 4 pi / n_dir; convergence in n_dir is measured in
    `P_convergence`, not assumed."""
    dirs = fib_dirs(n_dir)
    w = 4.0 * math.pi / n_dir
    out = {k: np.zeros(len(pts)) for k in PAIRS}
    for s0 in range(0, len(pts), chunk):
        P = pts[s0:s0 + chunk]
        Cp = group_columns(scene, P, dirs)
        Cm = group_columns(scene, P, -dirs)
        for g in ("A", "B", "F"):
            out[g + g][s0:s0 + chunk] = w * (Cp[g] * Cm[g]).sum(1)
        for g, h in (("A", "B"), ("A", "F"), ("B", "F")):
            out[g + h][s0:s0 + chunk] = w * (Cp[g] * Cm[h]
                                             + Cp[h] * Cm[g]).sum(1)
    return out


def P_total(Pgh: Dict[str, np.ndarray], scale: Optional[Dict[str, float]] = None
            ) -> np.ndarray:
    scale = scale or {}
    s = {g: scale.get(g, 1.0) for g in "ABF"}
    return sum(s[k[0]] * s[k[1]] * Pgh[k] for k in PAIRS)


# ==================================================== the vacuum state
def phi_vac(rho, rho_star: float):
    return 1.0 / (1.0 + rho / rho_star)


def dphi_vac(rho, rho_star: float):
    return -(1.0 / rho_star) / (1.0 + rho / rho_star) ** 2


def v_segments(rho_fn: Callable, z: np.ndarray, c: np.ndarray,
               n_gl: int = 48) -> np.ndarray:
    """Vacuum fraction of the segments z_i -> c_i (Gauss-Legendre), given a
    density function on points."""
    t, w = np.polynomial.legendre.leggauss(n_gl)
    t = 0.5 * (t + 1.0)
    w = 0.5 * w
    z = np.asarray(z, float)
    c = np.asarray(c, float)
    pts = z[:, None, :] + t[None, :, None] * (c - z)[:, None, :]
    r = rho_fn(pts.reshape(-1, 3)).reshape(len(z), n_gl)
    return (r * w[None, :]).sum(1)


# ====================================================== the (x, R) grid
@dataclass
class AxiGrid:
    xs: np.ndarray
    Rs: np.ndarray

    @property
    def dx(self) -> float:
        return float(self.xs[1] - self.xs[0])

    @property
    def dR(self) -> float:
        return float(self.Rs[1] - self.Rs[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.xs), len(self.Rs)

    def points(self) -> np.ndarray:
        X, R = np.meshgrid(self.xs, self.Rs, indexing="ij")
        return np.stack([X.ravel(), R.ravel(), np.zeros(X.size)], -1)

    def laplacian(self, F: np.ndarray) -> np.ndarray:
        """Cylindrical Laplacian F_xx + F_RR + F_R/R on the uniform grid,
        second-order central differences; the axis by the even extension
        (F_RR + F_R/R -> 2 F_RR, with F_RR(0) = 2(F(dR) - F(0))/dR^2).
        One-sided at the outer edges (those rows are never read)."""
        dx, dR = self.dx, self.dR
        out = np.zeros_like(F)
        out[1:-1, :] += (F[2:, :] - 2.0 * F[1:-1, :] + F[:-2, :]) / dx ** 2
        out[0, :] += (F[2, :] - 2.0 * F[1, :] + F[0, :]) / dx ** 2
        out[-1, :] += (F[-1, :] - 2.0 * F[-2, :] + F[-3, :]) / dx ** 2
        R = self.Rs[None, :]
        FRR = np.zeros_like(F)
        FRR[:, 1:-1] = (F[:, 2:] - 2.0 * F[:, 1:-1] + F[:, :-2]) / dR ** 2
        FRR[:, -1] = (F[:, -1] - 2.0 * F[:, -2] + F[:, -3]) / dR ** 2
        FR = np.zeros_like(F)
        FR[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * dR)
        FR[:, -1] = (F[:, -1] - F[:, -2]) / dR
        out[:, 1:] += FRR[:, 1:] + FR[:, 1:] / R[:, 1:]
        out[:, 0] += 4.0 * (F[:, 1] - F[:, 0]) / dR ** 2
        return out

    def volume_weights(self) -> np.ndarray:
        """2 pi R dR dx per cell (the axis ring gets pi (dR/2)^2 dx)."""
        w = 2.0 * math.pi * self.Rs * self.dR * self.dx
        w[0] = math.pi * (0.5 * self.dR) ** 2 * self.dx
        return np.broadcast_to(w[None, :], self.shape).copy()

    def integrate(self, F: np.ndarray, mask: Optional[np.ndarray] = None
                  ) -> float:
        w = self.volume_weights()
        if mask is not None:
            w = w * mask
        return float((F * w).sum())


def default_grid(D: float, x_half: float = 1.5, R_max_frac: float = 1.0,
                 dx: float = 40.0 * KPC) -> AxiGrid:
    nx = int(round(2.0 * x_half * D / dx)) + 1
    nR = int(round(R_max_frac * D / dx)) + 1
    xs = np.linspace(-x_half * D, x_half * D, nx)
    Rs = np.linspace(0.0, R_max_frac * D, nR)
    return AxiGrid(xs, Rs)


# ============================================ the P-law fields on the grid
@dataclass
class PathFields:
    """Everything the P law needs on the grid, computed ONCE per scene shape
    and rescaled exactly for the mass / filament ladder."""
    scene: TwoBody
    grid: AxiGrid
    n_dir: int
    Pgh: Dict[str, np.ndarray]                 # (N,) per pair, unscaled
    rho_g: Dict[str, np.ndarray]               # (N,) per group, unscaled
    seconds: float = 0.0

    @classmethod
    def build(cls, scene: TwoBody, grid: AxiGrid, n_dir: int = 4000
              ) -> "PathFields":
        import time
        t0 = time.perf_counter()
        pts = grid.points()
        Pgh = pairwise_P(scene, pts, n_dir)
        rho_g = {g: scene.rho_group(g, pts) if scene.groups[g]
                 else np.zeros(len(pts)) for g in "ABF"}
        return cls(scene, grid, n_dir, Pgh, rho_g,
                   seconds=time.perf_counter() - t0)

    def rho(self, scale=None) -> np.ndarray:
        scale = scale or {}
        return sum(scale.get(g, 1.0) * self.rho_g[g] for g in "ABF")

    def phi3_full(self, eps: float, rho_star: float, scale=None) -> np.ndarray:
        """Phi_3 of the whole scene: -(G eps/2) phi'(rho) P."""
        return -0.5 * G * eps * dphi_vac(self.rho(scale), rho_star) \
            * P_total(self.Pgh, scale)

    def phi3_isolated(self, g: str, eps: float, rho_star: float, scale=None
                      ) -> np.ndarray:
        scale = scale or {}
        s = scale.get(g, 1.0)
        return -0.5 * G * eps * dphi_vac(s * self.rho_g[g], rho_star) \
            * s * s * self.Pgh[g + g]

    def phi3_bridge(self, eps: float, rho_star: float, scale=None
                    ) -> np.ndarray:
        """The carrier term's two-body feature: full minus isolated A minus
        isolated B (the filament, if any, is entirely 'bridge')."""
        return (self.phi3_full(eps, rho_star, scale)
                - self.phi3_isolated("A", eps, rho_star, scale)
                - self.phi3_isolated("B", eps, rho_star, scale))

    def phi_dir_cross(self, eps: float, rho_star: float, scale=None
                      ) -> np.ndarray:
        """The endpoint term's two-body cross part, each source group as a
        point source at its centre (beads individually):

            -G eps sum_i M_i [ v_scene(z, c_i) - v_own(z, c_i) ] / |z - c_i|

        where v_own is the vacuum fraction on the source's OWN isolated
        density (A alone, B alone); for the filament, whose reference is the
        NEWTONIAN filament, the whole eps-part counts as bridge."""
        scale = scale or {}
        pts = self.grid.points()
        sc = self.scene
        rho_all = lambda p: sc.rho(p, scale)                       # noqa: E731
        out = np.zeros(len(pts))
        for g in "AB":
            comp = sc.groups[g][0]
            sg = scale.get(g, 1.0)
            cc = np.repeat(comp.c[None, :], len(pts), 0)
            v_all = phi_vac_mean(rho_all, pts, cc, rho_star)
            v_own = phi_vac_mean(lambda p: sg * comp.rho(p), pts, cc, rho_star)
            d = np.maximum(np.linalg.norm(pts - comp.c, axis=-1), 0.05 * comp.a)
            out += -G * eps * sg * comp.M * (v_all - v_own) / d
        sf = scale.get("F", 1.0)
        for bead in sc.groups["F"]:
            cc = np.repeat(bead.c[None, :], len(pts), 0)
            v_all = phi_vac_mean(rho_all, pts, cc, rho_star)
            d = np.maximum(np.linalg.norm(pts - bead.c, axis=-1), 0.05 * bead.a)
            out += -G * eps * sf * bead.M * v_all / d
        return out


def phi_vac_mean(rho_fn, z, c, rho_star, n_gl: int = 48) -> np.ndarray:
    return v_segments(lambda p: phi_vac(rho_fn(p), rho_star), z, c, n_gl)


# =================================================== the scalar competitors
def nu_rar(y):
    y = np.asarray(y, float)
    with np.errstate(over="ignore", divide="ignore"):
        return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(y, 1e-300))))


def dnu_rar(y):
    y = np.maximum(np.asarray(y, float), 1e-300)
    s = np.sqrt(y)
    e = np.exp(-s)
    return -e / (2.0 * s * (1.0 - e) ** 2)


def nu_simple(y):
    y = np.maximum(np.asarray(y, float), 1e-300)
    return 0.5 + np.sqrt(0.25 + 1.0 / y)


def dnu_simple(y):
    y = np.maximum(np.asarray(y, float), 1e-300)
    return -(1.0 / y ** 2) / (2.0 * np.sqrt(0.25 + 1.0 / y))


NU = {"rar": (nu_rar, dnu_rar), "simple": (nu_simple, dnu_simple)}


def phantom_density(scene: TwoBody, pts: np.ndarray, which: str = "rar",
                    scale=None, a0: float = A0,
                    groups: Optional[Sequence[str]] = None,
                    cell: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """rho_ph = (nu-1) rho - nu'(y) g^T J g / (4 pi G a0 |g|), closed form.

    `groups` restricts the SOURCE to a subset (the isolated references)."""
    nu, dnu = NU[which]
    sub = scene
    if groups is not None:
        sub = TwoBody(scene.MA, scene.MB, scene.a, scene.D, scene.rho_f,
                      scene.R_f)
        for g in "ABF":
            if g not in groups:
                sub.groups[g] = []
    out = _phantom_eval(sub, pts, nu, dnu, scale, a0)
    # The saddle (g_N = 0) is an INTEGRABLE 1/sqrt(distance) singularity of
    # rho_ph.  A grid point that lands on it (or within 1e-5 a0 of it) is
    # replaced by the cell average over a 4x4x4 offset sub-grid of the cell.
    gn = np.linalg.norm(sub.gN(pts, scale), axis=-1)
    bad = (~np.isfinite(out)) | (gn < 1e-5 * a0)
    if np.any(bad) and cell is not None:
        dx, dR = cell
        offs = (np.arange(4) + 0.5) / 4.0 - 0.5
        for i in np.flatnonzero(bad):
            sub_pts = np.array([[pts[i, 0] + ox * dx, pts[i, 1] + oy * dR,
                                 pts[i, 2] + oz * dR]
                                for ox in offs for oy in offs for oz in offs])
            v = _phantom_eval(sub, sub_pts, nu, dnu, scale, a0)
            out[i] = float(np.mean(v[np.isfinite(v)]))
    return out


def _phantom_eval(sub, pts, nu, dnu, scale, a0):
    rho = sub.rho(pts, scale)
    g = sub.gN(pts, scale)
    J = sub.jacN(pts, scale)
    gn = np.linalg.norm(g, axis=-1)
    gn = np.maximum(gn, 1e-300)
    y = gn / a0
    gJg = np.einsum("pi,pij,pj->p", g, J, g)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return (nu(y) - 1.0) * rho - dnu(y) * gJg / (4.0 * math.pi * G * a0 * gn)


# ======================================================= the tensor caricature
def tensor_l2_density(comp: Plummer, pts: np.ndarray, axis: np.ndarray,
                      fE: float = 1.0, n_r: int = 1400) -> np.ndarray:
    """rho_2 = f_E [chi'' + 2 chi'/r - 6 chi/r^2] P2(cos theta) / (4 pi G)
    for one body on its own AQUAL(simple-mu) base, chi from
    tensor_family.solve_l2 (BL's exact first-order l=2 solution), theta from
    `axis`."""
    import tensor_family as T
    a = comp.a
    rg = np.geomspace(1e-3 * a, 3e3 * a, n_r)
    gN = G * comp.M_enc(rg) / rg ** 2
    g0 = C.g_of_gN("aqual", gN, A0)
    x0 = g0 / A0
    mu0 = T.mu(x0)
    a_coef = x0 * (2.0 + x0) / (1.0 + x0) ** 2
    chi, dchi = T.solve_l2(rg, g0, mu0, a_coef, T.h(x0), T.hp(x0), x0)
    d2chi = np.gradient(dchi, rg)
    lap2 = d2chi + 2.0 * dchi / rg - 6.0 * chi / rg ** 2
    # chi ~ c r^2 at the centre, so chi'' + 2 chi'/r - 6 chi/r^2 -> 0 there;
    # the ODE's innermost nodes carry its Dirichlet boundary and are not
    # read (a grid point ON a body centre once picked up the boundary node
    # and carried 1e17 Msun in one cell -- caught by the sphere-integral test)
    r_safe = 0.05 * a
    ok = rg >= r_safe
    d = pts - comp.c
    r = np.linalg.norm(d, axis=-1)
    ct = (d @ axis) / np.maximum(r, 1e-300)
    P2 = 0.5 * (3.0 * ct * ct - 1.0)
    lap_at = np.interp(r, rg[ok], lap2[ok], left=0.0, right=0.0)
    lap_at = np.where(r < r_safe, 0.0, lap_at)
    return fE * lap_at * P2 / (4.0 * math.pi * G)


# ================================================ effective densities, by law
def rho_eff_bridge(law: str, pf: PathFields, eps: float = 0.0,
                   rho_star: float = RHO_STAR_FID, scale=None,
                   fE: float = 0.3, rho_f_dm: float = 0.0,
                   include_endpoint_cross: bool = True) -> Dict[str, np.ndarray]:
    """The BRIDGE effective density of `law` on the grid:

        bridge_law = rho_eff_law[full scene] - rho_eff_law[A alone]
                                             - rho_eff_law[B alone]

    returned as a dict with the baryonic filament ('filament', common to every
    law) and the law-specific remainder ('specific') separated, both (nx, nR).
    """
    grid, sc = pf.grid, pf.scene
    pts = grid.points()
    shp = grid.shape
    scale = scale or {}
    fil = (scale.get("F", 1.0) * pf.rho_g["F"]).reshape(shp)
    if law == "newton":
        spec = np.zeros(shp)
    elif law == "P":
        phi = pf.phi3_bridge(eps, rho_star, scale)
        if include_endpoint_cross:
            phi = phi + pf.phi_dir_cross(eps, rho_star, scale)
        spec = grid.laplacian(phi.reshape(shp)) / (4.0 * math.pi * G)
    elif law in ("qumond", "tensor"):
        which = "rar" if law == "qumond" else "simple"
        cell = (grid.dx, grid.dR)
        full = phantom_density(sc, pts, which, scale, cell=cell)
        isoA = phantom_density(sc, pts, which, scale, groups=("A",), cell=cell)
        isoB = phantom_density(sc, pts, which, scale, groups=("B",), cell=cell)
        spec = (full - isoA - isoB).reshape(shp)
        if law == "tensor":
            axis = np.array([1.0, 0.0, 0.0])
            for g in "AB":
                comp = sc.groups[g][0]
                sg = scale.get(g, 1.0)
                cs = Plummer(sg * comp.M, comp.a, comp.c)
                spec = spec + tensor_l2_density(cs, pts, axis, fE).reshape(shp)
    elif law == "cdm":
        dm = TwoBody(sc.MA, sc.MB, sc.a, sc.D, rho_f_dm, sc.R_f)
        spec = dm.rho_group("F", pts).reshape(shp) if dm.groups["F"] \
            else np.zeros(shp)
    else:
        raise ValueError(law)
    return dict(filament=fil, specific=spec, total=fil + spec)


# ====================================== the isolated-body radial reference
def isolated_body_profile(comp: Plummer, law: str, eps: float = 0.0,
                          rho_star: float = RHO_STAR_FID, n_r: int = 300,
                          n_dir: int = 4000, which: str = "rar") -> Dict[str, np.ndarray]:
    """rho_eff(r) of ONE Plummer under `law` (spherical), on a log grid, for
    the endpoint reference profiles.  For P this includes the body's own
    carrier term (P_AA, closed form) and its own endpoint modification
    (3-D source-centred quadrature of -G eps Int rho v/|r-y|)."""
    a = comp.a
    rg = np.geomspace(2e-2 * a, 40.0 * a, n_r)
    rho = comp.rho(np.stack([rg, 0 * rg, 0 * rg], -1) + comp.c)
    out = dict(r=rg, rho=rho)
    if law == "newton" or law == "cdm":
        out["rho_eff"] = rho
        return out
    if law in ("qumond", "tensor"):
        one = TwoBody(comp.M, comp.M, comp.a, 10.0 * comp.a)
        one.groups = {"A": [Plummer(comp.M, comp.a, (0.0, 0.0, 0.0))],
                      "B": [], "F": []}
        pts = np.stack([rg, 0 * rg, 0 * rg], -1)
        out["rho_eff"] = rho + phantom_density(
            one, pts, "rar" if law == "qumond" else "simple")
        return out
    # ---- P: carrier term along a radius (spherical: any direction)
    c0 = Plummer(comp.M, comp.a, (0.0, 0.0, 0.0))
    pts = np.stack([rg, 0 * rg, 0 * rg], -1)
    dirs = fib_dirs(n_dir)
    P = np.zeros(n_r)
    for i in range(n_r):
        Z = np.repeat(pts[i][None, :], n_dir, 0)
        P[i] = (4.0 * math.pi / n_dir) * (half_column(c0, Z, dirs)
                                          * half_column(c0, Z, -dirs)).sum()
    phi3 = -0.5 * G * dphi_vac(rho, rho_star) * P          # per unit eps
    # ---- endpoint modification, source-centred quadrature (per unit eps)
    phiv = np.array([_phi_v_single(c0, r, rho_star) for r in rg])
    phi1 = phi3 + phiv
    lnr = np.log(rg)
    dphi = np.gradient(phi1, lnr) / rg
    d2phi = np.gradient(dphi, lnr) / rg
    lap = d2phi + 2.0 * dphi / rg
    out["phi_extra_per_eps"] = phi1
    out["rho_extra_per_eps"] = lap / (4.0 * math.pi * G)
    out["rho_eff"] = rho + eps * out["rho_extra_per_eps"]
    return out


def _phi_v_single(comp: Plummer, r: float, rho_star: float, n_t: int = 160,
                  n_th: int = 40, n_seg: int = 24) -> float:
    """Phi_v(r) = -G Int rho(y) v(x,y)/|x-y| d^3y for x on the z-axis at
    radius r from one isolated Plummer at the origin (azimuth exact by
    symmetry)."""
    z = np.array([0.0, 0.0, r])
    ct, wt = np.polynomial.legendre.leggauss(n_th)
    st = np.sqrt(1.0 - ct ** 2)
    t = np.geomspace(1e-3 * comp.a, 400.0 * comp.a, n_t)
    y = np.stack([t[:, None] * st[None, :], np.zeros((n_t, n_th)),
                  t[:, None] * ct[None, :]], -1).reshape(-1, 3)
    rho_y = comp.rho(y)
    d = np.maximum(np.linalg.norm(y - z[None, :], axis=-1), 1e-4 * comp.a)
    v = phi_vac_mean(comp.rho, np.repeat(z[None, :], len(y), 0), y, rho_star,
                     n_gl=n_seg)
    dt = np.gradient(t)
    wgt = ((t ** 2 * dt)[:, None] * wt[None, :] * 2.0 * math.pi).reshape(-1)
    return -G * float(np.sum(wgt * rho_y * v / d))


def abel_project(rg: np.ndarray, rho_eff: np.ndarray, R: np.ndarray
                 ) -> np.ndarray:
    """Sigma(R) = 2 Int_R^inf rho(r) r dr / sqrt(r^2 - R^2), r = R cosh t."""
    t = np.linspace(0.0, 6.0, 240)
    r = R[:, None] * np.cosh(t)[None, :]
    f = np.interp(r.ravel(), rg, rho_eff, left=rho_eff[0], right=0.0
                  ).reshape(r.shape)
    return 2.0 * R * np.trapezoid(f * np.cosh(t)[None, :], t, axis=1)
