"""run_bridge.py -- Job 2: the bridge observables at detector level.

For the path action at eps in {0.3, 0.03, 0.003} and rho_* in {1e-24, rho_mean}:
the projected effective surface density along and across the axis, the shear
(gamma_1 along the axis, gamma_2, and the tangential shear about the
midpoint), the net enclosed mass, and the observer's strip statistics; then
the three BL properties as measurements with numerical errors -- (a) net mass,
(b) the M_A M_B log-slopes, (c) the rho_f dependence -- and the same
estimator on Newton, the QUMOND/RAR scalar, BL's tensor action and a CDM
filament of positive mass.  Noise enters in detect.py; here the errors are
NUMERICAL (grid halving, direction count) so the properties can be stated as
properties of the law before the detector is switched on.

    python run_bridge.py
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict

import numpy as np

import guard
import scene as S
import lensing as L
import estimator as E
import build_cache as BC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KPC, MPC, MSUN, G = S.KPC, S.MPC, S.MSUN, S.G

EPS_REF = 0.03                       # the matched-filter template's eps
RHO_F_LADDER = (0.0, 1e-27, 3e-27, 1e-26, 3e-26, 1e-25, 3e-25, 1e-24, 3e-24)
MASS_LADDER = (0.5, 2 ** -0.5, 1.0, 2 ** 0.5, 2.0)


def coarse_fields(pf: S.PathFields) -> S.PathFields:
    """The same fields on every other grid node (80 kpc): the Laplacian and
    projection error is estimated from the difference (second order:
    error at 40 kpc ~ difference / 3)."""
    g = S.AxiGrid(pf.grid.xs[::2], pf.grid.Rs[::2])
    shp = pf.grid.shape
    sub = lambda v: v.reshape(shp)[::2, ::2].ravel()          # noqa: E731
    return S.PathFields(pf.scene, g, pf.n_dir,
                        {k: sub(v) for k, v in pf.Pgh.items()},
                        {k: sub(v) for k, v in pf.rho_g.items()})


def transverse(sky: L.Sky, M: np.ndarray, x1: float = 0.0, ys=None) -> Dict:
    j = int(np.argmin(np.abs(sky.x1 - x1)))
    if ys is None:
        ys = np.array([0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200,
                       1500, 2000, 2500, 3000]) * KPC
    return {int(round(y / KPC)): float(np.interp(y, sky.x2, M[j])) for y in ys}


def along(sky: L.Sky, M: np.ndarray, x2: float = 0.0, xs=None) -> Dict:
    k = int(np.argmin(np.abs(sky.x2 - x2)))
    if xs is None:
        xs = np.array([-3000, -2500, -2000, -1750, -1500, -1250, -1000, -500,
                       0, 500, 1000, 1500, 2000, 3000]) * KPC
    return {int(round(x / KPC)): float(np.interp(x, sky.x1, M[:, k])) for x in xs}


def profile_features(sky: L.Sky, Sig: np.ndarray) -> Dict:
    j = int(np.argmin(np.abs(sky.x1)))
    k0 = int(np.argmin(np.abs(sky.x2)))
    prof = Sig[j, k0:]
    ys = sky.x2[k0:]
    core = float(prof[0])
    sgn = np.sign(core)
    cross = np.flatnonzero(np.sign(prof) != sgn)
    y0 = float(ys[cross[0]]) if len(cross) else float("nan")
    wing = prof[cross[0]:] if len(cross) else prof * 0
    ipk = int(np.argmax(np.abs(wing))) if len(wing) else 0
    return dict(Sigma_axis=core,
                zero_crossing_kpc=y0 / KPC,
                wing_peak=float(wing[ipk]) if len(wing) else float("nan"),
                wing_peak_kpc=float(ys[cross[0] + ipk] / KPC) if len(cross) else float("nan"))


def tangential_about_midpoint(sky: L.Sky, g1: np.ndarray, g2: np.ndarray,
                              radii=(0.5, 1.0, 1.5, 2.0, 3.0)) -> Dict:
    X1, X2 = sky.mesh()
    gt, gx = L.tangential_shear(g1, g2, X1, X2, 0.0)
    R = np.sqrt(X1 ** 2 + X2 ** 2)
    out = {}
    for r in radii:
        m = np.abs(R - r * MPC) < 0.75 * sky.pix
        out[f"{r:.1f}_Mpc"] = dict(gamma_t=float(gt[m].mean()),
                                   gamma_x=float(gx[m].mean()))
    return out


def net_masses(pf: S.PathFields, F: np.ndarray, sky: L.Sky, Sig: np.ndarray
               ) -> Dict:
    grid = pf.grid
    w = grid.volume_weights()
    out = {}
    for fx, fR in ((0.75, 0.5), (1.4, 0.95)):
        m = (np.abs(grid.xs)[:, None] < fx * pf.scene.D) & \
            (grid.Rs[None, :] < fR * pf.scene.D)
        tot = float((F * w * m).sum())
        tabs = float((np.abs(F) * w * m).sum())
        out[f"box_x{fx}D_R{fR}D"] = dict(net_Msun=tot / MSUN, abs_Msun=tabs / MSUN,
                                         net_over_abs=tot / max(tabs, 1e-300))
    a = sky.pix ** 2
    out["sky_2D"] = dict(net_Msun=float(Sig.sum() * a / MSUN),
                         abs_Msun=float(np.abs(Sig).sum() * a / MSUN),
                         net_over_abs=float(Sig.sum() / max(np.abs(Sig).sum(), 1e-300)))
    return out


def slope_fit(xs, ys) -> Dict:
    """log-slope with its standard error from the scatter about a power law."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.abs(np.asarray(ys, float)))
    A = np.stack([np.ones_like(lx), lx], 1)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    dof = max(len(lx) - 2, 1)
    cov = float(resid @ resid) / dof * np.linalg.inv(A.T @ A)
    return dict(slope=float(coef[1]), se=float(math.sqrt(max(cov[1, 1], 0.0))),
                n=int(len(lx)), rms_dex=float(np.sqrt(np.mean(resid ** 2)) / math.log(10)))


def main():
    guard.arm()
    t0 = time.perf_counter()
    pf = BC.load("fid")
    pff = BC.load("fid_fil")
    pfc = coarse_fields(pf)
    sc = pf.scene
    D = sc.D
    sky = L.default_sky(D)
    sv = L.Survey()
    scr = sv.sigma_crit_eff
    geom = E.Geometry(xA=-0.5 * D, xB=0.5 * D)
    iso: Dict = {}
    out: Dict = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scene=dict(M_A_Msun=sc.MA / MSUN, M_B_Msun=sc.MB / MSUN, a_kpc=sc.a / KPC,
                   D_Mpc=D / MPC, rho_f=sc.rho_f, R_f_kpc=sc.R_f / KPC,
                   grid=dict(nx=pf.grid.shape[0], nR=pf.grid.shape[1],
                             dx_kpc=pf.grid.dx / KPC, n_dir=pf.n_dir),
                   rho_at_midpoint=float(pf.rho()[int(np.argmin(
                       np.abs(pf.grid.points()[:, 0]) + pf.grid.points()[:, 1]))]),
                   note="BL's declared bridge scene, so every number sits beside BL's"),
        survey=dict(z_l=sv.z_l, kpc_per_arcmin=sv.kpc_per_arcmin,
                    sigma_crit_eff_kg_m2=scr, sigma_e=sv.sigma_e,
                    n_arcmin2=sv.n_arcmin2, pixel_kpc=sky.pix / KPC,
                    sources_per_pixel=sv.sources_per_pixel(sky.pix),
                    pixel_shear_sd=sv.pixel_noise(sky.pix)),
        estimator=dict(R_ex_kpc=geom.R_ex / KPC, w_core_kpc=geom.w_core / KPC,
                       w_wing_kpc=geom.w_wing / KPC, w_net_kpc=geom.w_net / KPC,
                       sector_half_deg=geom.sector_half_deg, eps_ref=EPS_REF,
                       stat_keys=list(E.STAT_KEYS)))

    # ---------------------------------------------------- the template
    mapsT = E.law_kappa_maps("P", pf, sky, scr, eps=EPS_REF, rho_star=S.RHO_STAR_FID,
                             iso_cache=iso)
    resT = E.bridge_statistics(mapsT["g1"], mapsT["g2"], sky, geom, scr,
                               return_maps=True)
    template = resT["kappa_res"]
    np.save(os.path.join(RES, "template_kres.npy"), template)
    out["numerical_floor_newton"] = {}

    # ---------------------------------------------------- 1. the P ladder
    ladder = {}
    for rtag, rs in S.RHO_STAR_LADDER.items():
        for eps in S.EPS_LADDER:
            t1 = time.perf_counter()
            maps = E.law_kappa_maps("P", pf, sky, scr, eps=eps, rho_star=rs,
                                    iso_cache=iso)
            res = E.bridge_statistics(maps["g1"], maps["g2"], sky, geom, scr,
                                      template=template)
            br = S.rho_eff_bridge("P", pf, eps, rs)
            br_nc = S.rho_eff_bridge("P", pf, eps, rs, include_endpoint_cross=False)
            Sig = maps["Sigma_spec"]
            Sig_nc = L.project(pf.grid, br_nc["specific"], sky)
            # coarse-grid error estimate
            brc = S.rho_eff_bridge("P", pfc, eps, rs)
            Sigc = L.project(pfc.grid, brc["specific"], sky)
            j0 = int(np.argmin(np.abs(sky.x1)))
            k0 = int(np.argmin(np.abs(sky.x2)))
            feat = profile_features(sky, Sig)
            ladder[f"{rtag}|eps={eps}"] = dict(
                eps=eps, rho_star=rs, seconds=time.perf_counter() - t1,
                phi3_midpoint_kms2=float(pf.phi3_bridge(eps, rs)[
                    int(np.argmin(np.abs(pf.grid.points()[:, 0])
                                  + pf.grid.points()[:, 1]))] / 1e6),
                Sigma_transverse_midplane=transverse(sky, Sig),
                Sigma_along_axis=along(sky, Sig),
                features=feat,
                Sigma_axis_error_grid=float(abs(Sig[j0, k0] - Sigc[j0, k0]) / 3.0),
                wing_error_grid=float(abs(np.interp(feat["wing_peak_kpc"] * KPC, sky.x2, Sig[j0])
                                          - np.interp(feat["wing_peak_kpc"] * KPC, sky.x2, Sigc[j0])) / 3.0)
                if np.isfinite(feat["wing_peak_kpc"]) else float("nan"),
                endpoint_cross_fraction_axis=float((Sig[j0, k0] - Sig_nc[j0, k0])
                                                   / Sig[j0, k0]),
                endpoint_cross_fraction_wing_1Mpc=float(
                    (np.interp(1 * MPC, sky.x2, Sig[j0]) - np.interp(1 * MPC, sky.x2, Sig_nc[j0]))
                    / np.interp(1 * MPC, sky.x2, Sig[j0])),
                kappa_spec_midpoint=float(maps["kappa_spec"][j0, k0]),
                gamma1_transverse_midplane=transverse(sky, maps["g1_bridge"]),
                gamma2_transverse_midplane=transverse(sky, maps["g2_bridge"]),
                gamma1_along_axis=along(sky, maps["g1_bridge"]),
                gamma_t_about_midpoint=tangential_about_midpoint(
                    sky, maps["g1_bridge"], maps["g2_bridge"]),
                net_mass=net_masses(pf, br["specific"], sky, Sig),
                net_mass_coarse=net_masses(pfc, brc["specific"], sky, Sigc),
                strip_stats={k: float(res[k]) for k in E.STAT_KEYS},
                N_Msun=float(res["N_Msun"]),
                profile_kappa_res=res["profile_kappa"],
                profile_x2_kpc=[v / KPC for v in res["profile_x2_m"]])
            r = ladder[f"{rtag}|eps={eps}"]
            print(f"P rho_*={rtag:<8} eps={eps:<6} Sigma_axis {feat['Sigma_axis']:+.4e} "
                  f"(+-{r['Sigma_axis_error_grid']:.1e}) wing {feat['wing_peak']:+.3e} at "
                  f"{feat['wing_peak_kpc']:.0f} kpc, y0 {feat['zero_crossing_kpc']:.0f} kpc; "
                  f"net/abs 3D {r['net_mass']['box_x1.4D_R0.95D']['net_over_abs']:+.2e}; "
                  f"Q {res['Q']:+.3e} DQ {res['DQ']:+.3e} A_mf {res['A_mf']:+.3f} "
                  f"N {res['N_Msun']:+.2e} Msun  [{r['seconds']:.0f}s]")
    out["P_ladder"] = ladder

    # ------------------------------------- 2. property (b): the mass ladder
    mass = {}
    for rtag, rs in S.RHO_STAR_LADDER.items():
        rows = []
        for sA in MASS_LADDER:
            for mode in ("A_only", "both"):
                sB = sA if mode == "both" else 1.0
                scale = {"A": sA, "B": sB}
                maps = E.law_kappa_maps("P", pf, sky, scr, eps=EPS_REF, rho_star=rs,
                                        scale=scale, iso_cache=iso)
                res = E.bridge_statistics(maps["g1"], maps["g2"], sky, geom, scr,
                                          template=template)
                j0 = int(np.argmin(np.abs(sky.x1)))
                k0 = int(np.argmin(np.abs(sky.x2)))
                rows.append(dict(sA=sA, sB=sB, mode=mode,
                                 Sigma_axis=float(maps["Sigma_spec"][j0, k0]),
                                 Q=float(res["Q"]), A_mf=float(res["A_mf"]),
                                 DQ=float(res["DQ"])))
        fits = {}
        for mode, xkey in (("A_only", "sA"), ("both", "sA")):
            sel = [r for r in rows if r["mode"] == mode]
            xs = [r[xkey] for r in sel]
            fits[mode] = {q: slope_fit(xs, [r[q] for r in sel])
                          for q in ("Sigma_axis", "Q", "A_mf")}
        mass[rtag] = dict(rows=rows, log_slopes=fits,
                          note="'A_only': M_A varied at fixed M_B (slope 1 expected "
                               "if bilinear); 'both': M_A = M_B varied together "
                               "(slope 2 expected)")
        print(f"mass ladder rho_*={rtag}: Sigma_axis slope in M_A "
              f"{fits['A_only']['Sigma_axis']['slope']:.3f} +- {fits['A_only']['Sigma_axis']['se']:.3f}; "
              f"both {fits['both']['Sigma_axis']['slope']:.3f} +- {fits['both']['Sigma_axis']['se']:.3f}; "
              f"A_mf: {fits['A_only']['A_mf']['slope']:.3f}, {fits['both']['A_mf']['slope']:.3f}")
    out["mass_scaling"] = mass

    # ---------------------------------- 3. property (c): the filament ladder
    fil = {}
    skyf = sky
    for rtag, rs in S.RHO_STAR_LADDER.items():
        rows = []
        for rho_f in RHO_F_LADDER:
            sF = rho_f / pff.scene.rho_f
            scale = {"F": sF}
            t1 = time.perf_counter()
            maps = E.law_kappa_maps("P", pff, skyf, scr, eps=EPS_REF, rho_star=rs,
                                    scale=scale, iso_cache=iso)
            res = E.bridge_statistics(maps["g1"], maps["g2"], skyf, geom, scr,
                                      template=template)
            j0 = int(np.argmin(np.abs(skyf.x1)))
            k0 = int(np.argmin(np.abs(skyf.x2)))
            Sig_spec = maps["Sigma_spec"]
            Sig_fil = maps["Sigma_fil"]
            mid = int(np.argmin(np.abs(pff.grid.points()[:, 0]) + pff.grid.points()[:, 1]))
            rho_mid = float(pff.rho(scale)[mid])
            rows.append(dict(
                rho_f=rho_f, rho_f_over_rho_star=rho_f / rs, rho_at_midpoint=rho_mid,
                Sigma_axis_specific=float(Sig_spec[j0, k0]),
                Sigma_axis_filament_newton=float(Sig_fil[j0, k0]),
                Sigma_wing_specific_400kpc=float(np.interp(400 * KPC, skyf.x2, Sig_spec[j0])),
                Q_specific_plus_filament=float(res["Q"]), A_mf=float(res["A_mf"]),
                N_Msun=float(res["N_Msun"]),
                minus_dphi_over_max=float(-S.dphi_vac(rho_mid, rs) * rs),
                seconds=time.perf_counter() - t1))
        # local log-slopes of |Sigma_axis_specific| vs rho_f (adjacent pairs)
        rf = np.array([r["rho_f"] for r in rows[1:]])
        ss = np.array([abs(r["Sigma_axis_specific"]) for r in rows[1:]])
        sn = np.array([abs(r["Sigma_axis_filament_newton"]) for r in rows[1:]])
        slopes = [dict(rho_f_lo=float(rf[i]), rho_f_hi=float(rf[i + 1]),
                       path_specific=float(np.log(ss[i + 1] / ss[i]) / np.log(rf[i + 1] / rf[i])),
                       newton_filament=float(np.log(sn[i + 1] / sn[i]) / np.log(rf[i + 1] / rf[i])))
                  for i in range(len(rf) - 1)]
        fil[rtag] = dict(rows=rows, local_log_slopes=slopes,
                         vacuum_limit_Sigma_axis=rows[0]["Sigma_axis_specific"],
                         note="the law-specific bridge exists at rho_f = 0 (it is a "
                              "two-body effect of the endpoints' columns); the "
                              "Newtonian filament vanishes there and grows with slope 1")
        print(f"filament ladder rho_*={rtag}: Sigma_axis(spec) at rho_f=0 "
              f"{rows[0]['Sigma_axis_specific']:+.3e}; slopes path "
              f"{[round(s['path_specific'], 2) for s in slopes]} vs newton "
              f"{[round(s['newton_filament'], 2) for s in slopes]}")
    out["filament_scaling"] = fil

    # ------------------------------------- 4. competitors on the same estimator
    comp = {}
    cases = [("newton", {}), ("qumond", {}), ("tensor", dict(fE=0.3)),
             ("tensor", dict(fE=1.0)), ("tensor", dict(fE=-0.5)),
             ("cdm", dict(rho_f_dm=1e-27)), ("cdm", dict(rho_f_dm=1e-26)),
             ("cdm", dict(rho_f_dm=1e-25)),
             ("P", dict(eps=0.003, rho_star=S.RHO_STAR_FID)),
             ("P", dict(eps=0.03, rho_star=S.RHO_STAR_FID)),
             ("P", dict(eps=0.003, rho_star=S.RHO_MEAN)),
             ("P", dict(eps=0.3, rho_star=S.RHO_MEAN))]
    for law, kw in cases:
        t1 = time.perf_counter()
        maps = E.law_kappa_maps(law, pf, sky, scr, iso_cache=iso, **kw)
        res = E.bridge_statistics(maps["g1"], maps["g2"], sky, geom, scr,
                                  template=template)
        Sig = maps["Sigma_spec"]
        tag = law + ("" if not kw else "|" + ",".join(f"{k}={v:g}" for k, v in kw.items()))
        comp[tag] = dict(
            law=law, params=kw, seconds=time.perf_counter() - t1,
            Sigma_transverse_midplane=transverse(sky, Sig),
            Sigma_along_axis=along(sky, Sig),
            features=profile_features(sky, Sig) if np.any(Sig) else None,
            net_mass_sky_2D=dict(net_Msun=float(Sig.sum() * sky.pix ** 2 / MSUN),
                                 abs_Msun=float(np.abs(Sig).sum() * sky.pix ** 2 / MSUN)),
            gamma_t_about_midpoint=tangential_about_midpoint(
                sky, maps["g1_bridge"], maps["g2_bridge"]),
            strip_stats={k: float(res[k]) for k in E.STAT_KEYS},
            N_Msun=float(res["N_Msun"]), profile_kappa_res=res["profile_kappa"],
            kappa_ends_max=float(maps["kappa_ends"].max()))
        print(f"{tag:<32} Q {res['Q']:+.3e} DQ {res['DQ']:+.3e} N {res['N_Msun']:+.3e} "
              f"A_mf {res['A_mf']:+.3f} S_core {res['S_core']:+.3e} S_wing {res['S_wing']:+.3e} "
              f"S_bcore {res['S_bcore']:+.3e} [{comp[tag]['seconds']:.0f}s]")
    out["competitors"] = comp

    # ------------------------------------------------ 5. inclination sample
    inc = {}
    for i_deg in (0.0, 30.0, 45.0, 60.0, 75.0):
        maps = E.law_kappa_maps("P", pf, sky, scr, eps=EPS_REF, rho_star=S.RHO_STAR_FID,
                                incl_deg=i_deg, iso_cache=iso)
        g = E.Geometry(xA=maps["xA"], xB=maps["xB"])
        ok = g.L_between > 4 * geom.w_core
        res = E.bridge_statistics(maps["g1"], maps["g2"], sky, g, scr,
                                  template=template) if ok else None
        Sig = maps["Sigma_spec"]
        inc[f"{i_deg:.0f}"] = dict(
            incl_deg=i_deg, projected_D_Mpc=(maps["xB"] - maps["xA"]) / MPC,
            Sigma_transverse_midplane=transverse(sky, Sig),
            features=profile_features(sky, Sig),
            strip_stats={k: float(res[k]) for k in E.STAT_KEYS} if res else None,
            note=None if ok else "between region too short for the strips")
        print(f"inclination {i_deg:2.0f} deg: D_proj {inc[f'{i_deg:.0f}']['projected_D_Mpc']:.2f} Mpc, "
              f"Sigma_axis {inc[f'{i_deg:.0f}']['features']['Sigma_axis']:+.3e}"
              + (f", Q {res['Q']:+.3e} A_mf {res['A_mf']:+.3f}" if res else ""))
    out["inclination"] = inc

    out["direction_convergence"] = dict(
        note="P at six bridge points for n_dir = 1000..16000 (scene.pairwise_P); "
             "the grids use 4000 (fid) and 2000 (others)",
        rows={int(n): S.P_total(S.pairwise_P(sc, np.array(
            [[0, 0, 0], [0, 0.5 * MPC, 0], [1.0 * MPC, 0, 0], [-1.7 * MPC, 0, 0]]), n)).tolist()
            for n in (1000, 2000, 4000, 8000)})
    out["provenance"] = guard.summary()
    out["wall_seconds"] = time.perf_counter() - t0
    with open(os.path.join(RES, "bridge.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote bridge.json in {out['wall_seconds']:.0f}s; provenance:",
          out["provenance"]["assertion"])


if __name__ == "__main__":
    main()
