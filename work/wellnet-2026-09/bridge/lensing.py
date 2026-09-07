"""lensing.py -- projection to the sky, convergence, shear, and BF's declared
detector model (no survey file is opened; every number is a declared synthetic
value from universes/provenance.DECLARED_NOISE, and the cosmology is the
universes package's own closed-form one).

    Sigma(xi)  = Int rho_eff dl                     (projection at inclination i)
    kappa      = Sigma / Sigma_cr
    gamma      = F^-1[ (k1^2 - k2^2 + 2 i k1 k2) / k^2  kappa^ ]   (zero-padded FFT)

Sky frame: xi1 along the PROJECTED pair axis, xi2 across it; the axis is
inclined by i out of the sky plane (i = 0: axis in the sky).  A 3-D point on
the line of sight at depth l is  xi1 e1 + xi2 e2 + l e3  with the pair axis
x^ = cos i e1 + sin i e3, so  x = xi1 cos i + l sin i  and
R^2 = xi1^2 + xi2^2 + l^2 - x^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

import guard                                            # noqa: F401
import scene as S
from universes import physics as ph                     # noqa: E402
from universes.provenance import DECLARED_NOISE as DN   # noqa: E402

KPC, MPC, MSUN, G = S.KPC, S.MPC, S.MSUN, S.G
MSUN_PER_KPC2_TO_SI = MSUN / KPC ** 2                   # kg m^-2


# ================================================= declared survey geometry
@dataclass
class Survey:
    """BF's declared detector: shape noise 0.26 per component, 20 sources
    per arcmin^2, lens at z = 0.3, sources drawn as BF's corpus draws them."""
    z_l: float = 0.30
    sigma_e: float = DN["wl_shape_noise_per_component"]
    n_arcmin2: float = DN["wl_source_density_arcmin2"]
    m_bias_sigma: float = DN["wl_multiplicative_bias_sigma"]
    c_bias_sigma: float = DN["wl_additive_bias_sigma"]
    seed: int = 20260904

    @property
    def kpc_per_arcmin(self) -> float:
        return float(ph.D_A(self.z_l) * 1e3 * (math.pi / 180.0 / 60.0))

    @property
    def sigma_crit_eff(self) -> float:
        """<1/Sigma_cr>^-1 over BF's source-redshift draw, in kg m^-2."""
        rng = np.random.default_rng(self.seed)
        zs = np.clip(self.z_l + 0.25 + rng.gamma(2.6, 0.28, 20000),
                     self.z_l + 0.08, 3.4)
        inv = 1.0 / (ph.sigma_crit(self.z_l, zs) * MSUN_PER_KPC2_TO_SI)
        return float(1.0 / np.mean(inv))

    def sources_per_pixel(self, pix_m: float) -> float:
        side_arcmin = pix_m / KPC / self.kpc_per_arcmin
        return self.n_arcmin2 * side_arcmin ** 2

    def pixel_noise(self, pix_m: float) -> float:
        """Shape-noise sd per shear component per pixel: sigma_e / sqrt(N)."""
        return self.sigma_e / math.sqrt(self.sources_per_pixel(pix_m))


# =============================================================== the sky grid
@dataclass
class Sky:
    x1: np.ndarray            # (n1,) metres, along the projected axis
    x2: np.ndarray            # (n2,) metres, across

    @property
    def pix(self) -> float:
        return float(self.x1[1] - self.x1[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.x1), len(self.x2)

    def mesh(self):
        return np.meshgrid(self.x1, self.x2, indexing="ij")


def default_sky(D: float, half1: float = 1.5, half2: float = 1.4,
                pix: float = 40.0 * KPC) -> Sky:
    """The field: +-1.5 D along the axis, +-1.4 D across it.  The height is
    what lets the endpoint models be MEASURED (not extrapolated) out to
    1.4 D = 5.6 Mpc from either centre, which covers the region behind the
    other endpoint; only the far corners use the declared r^-2 tail."""
    n1 = int(round(2 * half1 * D / pix)) + 1
    n2 = int(round(2 * half2 * D / pix)) + 1
    return Sky(np.linspace(-half1 * D, half1 * D, n1),
               np.linspace(-half2 * D, half2 * D, n2))


# ================================================================ projection
def project(grid: S.AxiGrid, F: np.ndarray, sky: Sky, incl_deg: float = 0.0,
            n_los: Optional[int] = None) -> np.ndarray:
    """Sigma(xi1, xi2) = Int F(x, R) dl, bilinear interpolation on the
    (x, R) grid, zero outside it."""
    ci, si = math.cos(math.radians(incl_deg)), math.sin(math.radians(incl_deg))
    Lmax = float(grid.Rs[-1]) if abs(si) < 1e-12 else \
        float(max(grid.Rs[-1], abs(grid.xs[-1])) / max(abs(si), 0.3))
    if n_los is None:
        n_los = int(round(2 * Lmax / grid.dR)) + 1
    l = np.linspace(-Lmax, Lmax, n_los)
    X1, X2 = sky.mesh()
    out = np.zeros(sky.shape)
    dl = l[1] - l[0]
    for lv in l:
        x = X1 * ci + lv * si
        R2 = X1 ** 2 + X2 ** 2 + lv ** 2 - x ** 2
        R = np.sqrt(np.maximum(R2, 0.0))
        cx = (x - grid.xs[0]) / grid.dx
        cR = R / grid.dR
        out += map_coordinates(F, [cx.ravel(), cR.ravel()], order=1,
                               mode="constant", cval=0.0).reshape(sky.shape)
    return out * dl


def project_radial(rg: np.ndarray, rho_eff: np.ndarray, sky: Sky,
                   centre_x1: float) -> np.ndarray:
    """Sigma map of a spherical profile centred on the axis at xi1 = centre."""
    X1, X2 = sky.mesh()
    R = np.sqrt((X1 - centre_x1) ** 2 + X2 ** 2).ravel()
    Rb = np.geomspace(max(R.min(), 1e-3 * KPC), R.max() * 1.001, 400)
    Sb = S.abel_project(rg, rho_eff, Rb)
    return np.interp(R, Rb, Sb).reshape(sky.shape)


def plummer_sigma(M: float, a: float, sky: Sky, centre_x1: float) -> np.ndarray:
    X1, X2 = sky.mesh()
    R2 = (X1 - centre_x1) ** 2 + X2 ** 2
    return M * a ** 2 / (math.pi * (R2 + a ** 2) ** 2)


def radial_shear_map(rg: np.ndarray, rho_eff: np.ndarray, sky: Sky,
                     centre_x1: float, sigma_crit: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gamma_1, gamma_2, kappa) of a SPHERICAL profile, analytically:
    gamma_t(R) = [M_2D(<R)/(pi R^2) - Sigma(R)] / Sigma_cr, no FFT, so the
    finite field and its padding never enter the endpoint shear."""
    X1, X2 = sky.mesh()
    R = np.sqrt((X1 - centre_x1) ** 2 + X2 ** 2)
    phi = np.arctan2(X2, X1 - centre_x1)
    Rb = np.geomspace(1e-3 * KPC, R.max() * 1.001, 600)
    Sb = S.abel_project(rg, rho_eff, Rb)
    M2 = np.concatenate(([0.0], np.cumsum(
        0.5 * (Sb[1:] * Rb[1:] + Sb[:-1] * Rb[:-1]) * np.diff(Rb)))) * 2 * math.pi
    gt_b = (M2 / (math.pi * Rb ** 2) - Sb) / sigma_crit
    Rf = np.maximum(R.ravel(), Rb[0])
    gt = np.interp(Rf, Rb, gt_b).reshape(sky.shape)
    kap = np.interp(Rf, Rb, Sb).reshape(sky.shape) / sigma_crit
    return -gt * np.cos(2 * phi), -gt * np.sin(2 * phi), kap


# ============================================================= shear via FFT
def shear_from_kappa(kappa: np.ndarray, pix: float, pad: int = 2
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """gamma_1, gamma_2 from kappa on a uniform grid (Kaiser-Squires forward
    relation), zero-padded by `pad`x to suppress periodic images."""
    n1, n2 = kappa.shape
    N1, N2 = pad * n1, pad * n2
    K = np.zeros((N1, N2))
    K[:n1, :n2] = kappa
    kh = np.fft.fft2(K)
    k1 = np.fft.fftfreq(N1, d=pix)[:, None] * 2 * math.pi
    k2 = np.fft.fftfreq(N2, d=pix)[None, :] * 2 * math.pi
    k2sum = k1 ** 2 + k2 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(k2sum > 0, (k1 ** 2 - k2 ** 2) / k2sum, 0.0)
        f2 = np.where(k2sum > 0, 2.0 * k1 * k2 / k2sum, 0.0)
    g1 = np.real(np.fft.ifft2(kh * f1))[:n1, :n2]
    g2 = np.real(np.fft.ifft2(kh * f2))[:n1, :n2]
    return g1, g2


def kappa_from_shear(g1: np.ndarray, g2: np.ndarray, pix: float, pad: int = 2
                     ) -> np.ndarray:
    """Kaiser-Squires inversion; the k = 0 mode (mass sheet) is set to zero."""
    n1, n2 = g1.shape
    N1, N2 = pad * n1, pad * n2
    G1 = np.zeros((N1, N2))
    G2 = np.zeros((N1, N2))
    G1[:n1, :n2] = g1
    G2[:n1, :n2] = g2
    k1 = np.fft.fftfreq(N1, d=pix)[:, None] * 2 * math.pi
    k2 = np.fft.fftfreq(N2, d=pix)[None, :] * 2 * math.pi
    k2sum = k1 ** 2 + k2 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(k2sum > 0, (k1 ** 2 - k2 ** 2) / k2sum, 0.0)
        f2 = np.where(k2sum > 0, 2.0 * k1 * k2 / k2sum, 0.0)
    kh = np.fft.fft2(G1) * f1 + np.fft.fft2(G2) * f2
    return np.real(np.fft.ifft2(kh))[:n1, :n2]


def tangential_shear(g1, g2, X1, X2, cx: float, cy: float = 0.0):
    """gamma_t, gamma_x about (cx, cy)."""
    phi = np.arctan2(X2 - cy, X1 - cx)
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    gt = -(g1 * c2 + g2 * s2)
    gx = -(-g1 * s2 + g2 * c2)
    return gt, gx


# ============================================================ noise draws
def noise_maps(sky: Sky, survey: Survey, rng: np.random.Generator,
               systematics: bool = True) -> Dict[str, np.ndarray]:
    """One realisation of BF's declared detector on the pixel grid: shape
    noise per component, an additive bias c, and a spatially coherent PSF
    residual of amplitude 2 sigma_c at a random wavevector (exactly BF's
    corpus.emit_cluster).  The multiplicative bias is returned separately,
    because it multiplies the SIGNAL."""
    sd = survey.pixel_noise(sky.pix)
    n1, n2 = sky.shape
    e1 = rng.normal(0.0, sd, (n1, n2))
    e2 = rng.normal(0.0, sd, (n1, n2))
    m = 0.0
    if systematics:
        m = float(rng.normal(0.0, survey.m_bias_sigma))
        c1 = rng.normal(0.0, survey.c_bias_sigma)
        c2 = rng.normal(0.0, survey.c_bias_sigma)
        X1, X2 = sky.mesh()
        rmax = 0.5 * (sky.x1[-1] - sky.x1[0])
        kx, ky = rng.normal(size=2) * 2.0 / rmax
        ca = 2.0 * survey.c_bias_sigma
        e1 = e1 + c1 + ca * np.cos(kx * X1 + ky * X2)
        e2 = e2 + c2 + ca * np.sin(kx * X1 + ky * X2)
    return dict(e1=e1, e2=e2, m_bias=m, pixel_sd=sd)
