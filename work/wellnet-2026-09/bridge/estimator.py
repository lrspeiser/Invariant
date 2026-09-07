"""estimator.py -- the bridge statistic, as an observer would form it.

INPUT   a shear map (gamma_1, gamma_2) on the sky grid, the two endpoint
        centres (known from the baryons), the pair axis.
STEPS   1. ENDPOINT MODEL.  Each endpoint's tangential-shear profile
           gamma_t(r) is measured in the sectors PERPENDICULAR to the pair
           axis (|phi -/+ 90 deg| < 30 deg), where no bridge lives, in log
           radial bins; the axisymmetric model gamma(r, phi) =
           -gamma_t(r) e^{2 i phi} is then subtracted everywhere.  The two
           endpoints are fitted alternately (4 sweeps).  Exact for any law
           whose isolated body is spherical, WITHOUT assuming a profile
           shape -- the P law's and QUMOND's endpoint haloes are not
           Plummer-like and are not asked to be.
        2. RESIDUAL CONVERGENCE by Kaiser-Squires on the residual shear
           (k = 0 mode set to zero: the mass sheet).
        3. STRIP STATISTICS in the bridge frame:
              between   xA + R_ex < xi1 < xB - R_ex
              behind    the same length outside each endpoint
              core      |xi2| < w_core;   wing   w_core < |xi2| < w_wing
           S_core, S_wing, S_bcore, S_bwing (mean residual kappa), the
           transverse profile of the between region, and the derived
              Q  = S_core - S_wing            the compensation contrast
              N  = A_core S_core + A_wing S_wing   net between-strip 'mass'
              DQ = (S_core - S_bcore) - (S_wing - S_bwing)
                                              between minus behind: an m = 2
                                              halo pattern is symmetric under
                                              phi -> phi + 180 deg and cancels
                                              here exactly; a bridge does not.

Every step is LINEAR in the shear map, so the statistics' noise is Gaussian
with a variance that does not depend on the signal; it is measured on an
untouched null half rather than propagated, and the two are compared.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import lensing as L
import scene as S

KPC, MPC, MSUN = S.KPC, S.MPC, S.MSUN


@dataclass
class Geometry:
    xA: float
    xB: float
    R_ex: float = 500.0 * KPC          # exclusion around each centre
    w_core: float = 200.0 * KPC        # core strip half-width (the P core is
    #                                    |xi2| < ~150 kpc: a column PRODUCT)
    w_wing: float = 1600.0 * KPC       # wing strip outer half-width
    w_net: float = 2400.0 * KPC        # wide aperture for the net mass
    sector_half_deg: float = 30.0      # perpendicular sectors for the model
    n_rbins: int = 60
    n_sweeps: int = 4
    n_profile_bins: int = 35

    @property
    def L_between(self) -> float:
        return (self.xB - self.xA) - 2.0 * self.R_ex


def _sector_profile(gt: np.ndarray, r: np.ndarray, phi: np.ndarray,
                    geom: Geometry, r_edges: np.ndarray) -> np.ndarray:
    """Mean gamma_t per log radial bin over the perpendicular sectors."""
    c = np.abs(np.cos(phi)) < math.sin(math.radians(geom.sector_half_deg))
    idx = np.digitize(r, r_edges) - 1
    ok = c & (idx >= 0) & (idx < len(r_edges) - 1)
    sums = np.bincount(idx[ok], weights=gt[ok], minlength=len(r_edges) - 1)
    cnt = np.bincount(idx[ok], minlength=len(r_edges) - 1)
    prof = np.full(len(r_edges) - 1, np.nan)
    m = cnt > 0
    prof[m] = sums[m] / cnt[m]
    # fill empty bins by interpolation in log r
    rc = np.sqrt(r_edges[1:] * r_edges[:-1])
    if not np.all(m):
        prof[~m] = np.interp(np.log(rc[~m]), np.log(rc[m]), prof[m])
    return prof


def endpoint_models(g1: np.ndarray, g2: np.ndarray, sky: L.Sky,
                    geom: Geometry) -> Dict[str, np.ndarray]:
    X1, X2 = sky.mesh()
    ends = {"A": geom.xA, "B": geom.xB}
    polar = {}
    for k, xc in ends.items():
        r = np.sqrt((X1 - xc) ** 2 + X2 ** 2)
        phi = np.arctan2(X2, X1 - xc)
        polar[k] = (r, phi, np.cos(2 * phi), np.sin(2 * phi))
    r_lo = 1.5 * sky.pix
    r_hi = 0.5 * (sky.x2[-1] - sky.x2[0])          # the field's half-height:
    #                                                 the last bin with
    #                                                 perpendicular-sector pixels
    r_edges = np.geomspace(r_lo, r_hi, geom.n_rbins + 1)
    rc = np.sqrt(r_edges[1:] * r_edges[:-1])
    model = {k: (np.zeros_like(g1), np.zeros_like(g2)) for k in ends}
    profiles = {}
    for _ in range(geom.n_sweeps):
        for k in ends:
            other = [o for o in ends if o != k][0]
            r, phi, c2, s2 = polar[k]
            g1r = g1 - model[other][0]
            g2r = g2 - model[other][1]
            gt = -(g1r * c2 + g2r * s2)
            prof = _sector_profile(gt.ravel(), r.ravel(), phi.ravel(),
                                   geom, r_edges)
            # gamma_t(r) at every pixel: log-r interpolation out to r_hi (the
            # field's half-height, so every bin has sector pixels); beyond it
            # -- only the far corners -- a DECLARED point-mass tail, gamma_t
            # ~ r^-2, scaled linearly by the last bin.  (A slope FITTED to
            # log gamma_t of the outer bins made the estimator nonlinear in
            # the data -- caught by test T8 and by detect.py's own check --
            # which would have voided the signal + noise decomposition.)
            gt_pix = np.interp(np.log(np.maximum(r, r_lo)), np.log(rc), prof)
            far = r > rc[-1]
            gt_pix = np.where(far, prof[-1] * (rc[-1] / np.maximum(r, r_lo)) ** 2,
                              gt_pix)
            model[k] = (-gt_pix * c2, -gt_pix * s2)
            profiles[k] = prof
    return dict(g1_model=model["A"][0] + model["B"][0],
                g2_model=model["A"][1] + model["B"][1],
                r_centres=rc, profile_A=profiles["A"], profile_B=profiles["B"])


def region_masks(sky: L.Sky, geom: Geometry) -> Dict[str, np.ndarray]:
    X1, X2 = sky.mesh()
    Lb = geom.L_between
    core = np.abs(X2) < geom.w_core
    wing = (np.abs(X2) >= geom.w_core) & (np.abs(X2) < geom.w_wing)
    between = (X1 > geom.xA + geom.R_ex) & (X1 < geom.xB - geom.R_ex)
    behind = ((X1 < geom.xA - geom.R_ex) & (X1 > geom.xA - geom.R_ex - Lb)) | \
             ((X1 > geom.xB + geom.R_ex) & (X1 < geom.xB + geom.R_ex + Lb))
    return dict(core_between=core & between, wing_between=wing & between,
                core_behind=core & behind, wing_behind=wing & behind,
                between=between & (np.abs(X2) < geom.w_wing),
                net_between=between & (np.abs(X2) < geom.w_net),
                net_behind=behind & (np.abs(X2) < geom.w_net))


def bridge_statistics(g1: np.ndarray, g2: np.ndarray, sky: L.Sky,
                      geom: Geometry, sigma_crit: float = 1.0,
                      return_maps: bool = False,
                      template: Optional[np.ndarray] = None
                      ) -> Dict[str, object]:
    """`template`: an optional noise-free residual-kappa map of the P
    prediction at a reference eps; A_mf = <kres, T>/<T, T> over the wide
    between aperture is then the matched-filter amplitude in units of that
    reference (1.0 = the reference prediction)."""
    em = endpoint_models(g1, g2, sky, geom)
    g1r = g1 - em["g1_model"]
    g2r = g2 - em["g2_model"]
    kres = L.kappa_from_shear(g1r, g2r, sky.pix)
    masks = region_masks(sky, geom)
    apix = sky.pix ** 2
    S_ = {k: float(kres[m].mean()) for k, m in masks.items()}
    A_core = float(masks["core_between"].sum()) * apix
    A_wing = float(masks["wing_between"].sum()) * apix
    A_net = float(masks["net_between"].sum()) * apix
    A_field = float(kres.size) * apix
    Q = S_["core_between"] - S_["wing_between"]
    # net 'mass' in the WIDE between aperture, mass-sheet corrected, minus
    # the same aperture behind (which removes any residual field gradient)
    N = ((S_["net_between"] - S_["net_behind"]) * A_net
         / (1.0 - A_net / A_field))
    DQ = ((S_["core_between"] - S_["core_behind"])
          - (S_["wing_between"] - S_["wing_behind"]))
    A_mf = float("nan")
    if template is not None:
        m = masks["net_between"]
        tt = float((template[m] ** 2).sum())
        A_mf = float((kres[m] * template[m]).sum() / tt) if tt > 0 else float("nan")
    # transverse profile of the between region
    X1, X2 = sky.mesh()
    edges = np.linspace(-geom.w_wing, geom.w_wing, geom.n_profile_bins + 1)
    idx = np.digitize(X2.ravel(), edges) - 1
    ok = masks["between"].ravel() & (idx >= 0) & (idx < geom.n_profile_bins)
    sums = np.bincount(idx[ok], weights=kres.ravel()[ok],
                       minlength=geom.n_profile_bins)
    cnt = np.bincount(idx[ok], minlength=geom.n_profile_bins)
    prof = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    out = dict(S_core=S_["core_between"], S_wing=S_["wing_between"],
               S_bcore=S_["core_behind"], S_bwing=S_["wing_behind"],
               S_net=S_["net_between"], S_bnet=S_["net_behind"],
               Q=Q, N_kappa_m2=N, N_kg=N * sigma_crit,
               N_Msun=N * sigma_crit / MSUN, DQ=DQ, A_mf=A_mf,
               A_core_m2=A_core, A_wing_m2=A_wing, A_net_m2=A_net,
               profile_x2_m=(0.5 * (edges[1:] + edges[:-1])).tolist(),
               profile_kappa=prof.tolist())
    if return_maps:
        out["kappa_res"] = kres
        out["g1_res"] = g1r
        out["g2_res"] = g2r
        out["endpoint_models"] = em
    return out


STAT_KEYS = ("S_core", "S_wing", "S_bcore", "S_bwing", "Q", "N_kappa_m2",
             "DQ", "A_mf")


def stat_vector(res: Dict[str, object]) -> np.ndarray:
    return np.array([res[k] for k in STAT_KEYS], float)


# ============================================ the full-scene shear of a law
def law_kappa_maps(law: str, pf: S.PathFields, sky: L.Sky, sigma_crit: float,
                   eps: float = 0.0, rho_star: float = S.RHO_STAR_FID,
                   scale: Optional[Dict[str, float]] = None, fE: float = 0.3,
                   rho_f_dm: float = 0.0, incl_deg: float = 0.0,
                   iso_cache: Optional[Dict] = None,
                   include_endpoint_cross: bool = True) -> Dict[str, np.ndarray]:
    """kappa of the endpoints (isolated, under the law), of the baryonic
    filament, and of the law-specific bridge, on the sky."""
    scale = scale or {}
    sc = pf.scene
    ci = math.cos(math.radians(incl_deg))
    xA, xB = -0.5 * sc.D * ci, +0.5 * sc.D * ci
    ends = np.zeros(sky.shape)
    g1e = np.zeros(sky.shape)
    g2e = np.zeros(sky.shape)
    for g, xc in (("A", xA), ("B", xB)):
        comp = sc.groups[g][0]
        sg = scale.get(g, 1.0)
        key = (law, g, sg, rho_star)
        if iso_cache is not None and key in iso_cache:
            prof = dict(iso_cache[key])
        else:
            cs = S.Plummer(sg * comp.M, comp.a, (0.0, 0.0, 0.0))
            prof = S.isolated_body_profile(cs, law, 1.0, rho_star)
            if iso_cache is not None:
                iso_cache[key] = dict(prof)
        if law == "P":
            prof["rho_eff"] = prof["rho"] + eps * prof["rho_extra_per_eps"]
        a1, a2, ak = L.radial_shear_map(prof["r"], prof["rho_eff"], sky, xc,
                                        sigma_crit)
        ends += ak
        g1e += a1
        g2e += a2
    br = S.rho_eff_bridge(law, pf, eps, rho_star, scale, fE, rho_f_dm,
                          include_endpoint_cross)
    fil = L.project(pf.grid, br["filament"], sky, incl_deg)
    spec = L.project(pf.grid, br["specific"], sky, incl_deg)
    kb = (fil + spec) / sigma_crit
    b1, b2 = L.shear_from_kappa(kb, sky.pix, pad=3)
    return dict(kappa_ends=ends, kappa_fil=fil / sigma_crit,
                kappa_spec=spec / sigma_crit, kappa_bridge=kb,
                kappa_total=ends + kb,
                g1=g1e + b1, g2=g2e + b2, g1_ends=g1e, g2_ends=g2e,
                g1_bridge=b1, g2_bridge=b2,
                Sigma_spec=spec, Sigma_fil=fil, xA=xA, xB=xB)
