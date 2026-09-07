"""cdm_attack.py -- Job 3: attack the compensation.

BL's CDM distinction rests on ONE sentence: rho_DM >= 0 cannot make a
net-zero-mass bridge.  Here it is tested where an observer would test it --
in PROJECTION, through the same estimator, after the endpoints' azimuthal
profiles are subtracted -- against the widest positive-mass configuration
the bench can write:

    a DICTIONARY of positive-mass components, each run through the estimator
    on its own:  elliptical (projected-triaxial) Plummer haloes about each
    endpoint with axis ratio q, position angle theta from the pair axis,
    scale a_h and a centre offset delta along the axis (so sums of them are
    haloes with radius-dependent ellipticity, isophote TWISTS and
    lopsidedness), and positive filaments of several widths.

The estimator is linear in the shear, so the residual of any positive
combination is the same combination of residuals, and the question

    "is there a rho_DM >= 0 whose projected bridge signature equals P's?"

is a NON-NEGATIVE least-squares problem (scipy.optimize.nnls) on the masked
residual maps.  The UNCONSTRAINED least squares (masses of either sign) is
the control: if it reproduces the signature and NNLS does not, the obstruction
is positivity itself; if NNLS also reproduces it, the falsifier is weaker than
BL claimed.  Three fit regions (between; between + behind; wide aperture) and
three inclinations of the P signature; a CDM-prior variant restricts the
dictionary to haloes aligned with the pair axis (BK.4: haloes align with
their filament).

    python cdm_attack.py
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List

import numpy as np
from scipy.optimize import nnls

import guard
import scene as S
import lensing as L
import estimator as E
import build_cache as BC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KPC, MPC, MSUN = S.KPC, S.MPC, S.MSUN
EPS_REF = 0.03

Q_GRID = (0.5, 0.7, 0.85, 1.0)
TH_GRID = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
A_GRID = (0.5, 1.0, 2.0, 4.0)                # in units of the scene's a
D_GRID = (-0.5, 0.0, 0.5)                    # offset along the axis, units of a
RF_GRID = (100.0, 200.0, 400.0)              # filament widths, kpc


def elliptical_plummer_sigma(M, a, q, theta_deg, sky: L.Sky, cx, cy=0.0):
    """Projected Plummer stratified on ellipses of axis ratio q, major axis
    at theta from xi1; integrates to M."""
    X1, X2 = sky.mesh()
    t = math.radians(theta_deg)
    u = (X1 - cx) * math.cos(t) + (X2 - cy) * math.sin(t)
    v = -(X1 - cx) * math.sin(t) + (X2 - cy) * math.cos(t)
    m2 = u * u + (v / q) ** 2
    return M * a ** 2 / (math.pi * q * (m2 + a ** 2) ** 2)


def component_shear(kind, params, sky: L.Sky, scr, sc: S.TwoBody):
    """(g1, g2) of one positive-mass component, with the spherical part of a
    halo done analytically and only the compact non-spherical remainder by
    FFT (so the finite field never enters at the level of the signal)."""
    if kind == "halo":
        M, a, q, th, cx, dx = params
        Sig = elliptical_plummer_sigma(M, a, q, th, sky, cx + dx)
        Ssph = L.plummer_sigma(M, a, sky, cx + dx)
        rg = np.geomspace(2e-2 * a, 40.0 * a, 300)
        rho = S.Plummer(M, a).rho(np.stack([rg, 0 * rg, 0 * rg], -1))
        g1s, g2s, _ = L.radial_shear_map(rg, rho, sky, cx + dx, scr)
        d1, d2 = L.shear_from_kappa((Sig - Ssph) / scr, sky.pix, pad=3)
        return g1s + d1, g2s + d2, Sig
    if kind == "filament":
        rho_f, R_f = params
        beads = S.filament_beads(sc.D, sc.a, rho_f, R_f)
        Sig = np.zeros(sky.shape)
        for b in beads:
            Sig += L.plummer_sigma(b.M, b.a, sky, b.c[0])
        g1, g2 = L.shear_from_kappa(Sig / scr, sky.pix, pad=3)
        return g1, g2, Sig
    raise ValueError(kind)


def build_dictionary(sky, scr, sc, geom) -> List[Dict]:
    atoms = []
    for end, cx in (("A", -0.5 * sc.D), ("B", 0.5 * sc.D)):
        for q in Q_GRID:
            for th in (TH_GRID if q < 1.0 else (0.0,)):
                for fa in A_GRID:
                    for fd in D_GRID:
                        if q == 1.0 and fd == 0.0:
                            continue          # a centred sphere: no residual by construction
                        M = 1.0e13 * MSUN     # unit block; the weights are masses / 1e13
                        g1, g2, Sig = component_shear(
                            "halo", (M, fa * sc.a, q, th, cx, fd * sc.a), sky, scr, sc)
                        r = E.bridge_statistics(g1, g2, sky, geom, scr, return_maps=True)
                        atoms.append(dict(kind="halo", end=end, q=q, theta=th, a_over_a=fa,
                                          offset_over_a=fd, mass_unit_Msun=M / MSUN,
                                          kres=r["kappa_res"],
                                          aligned=bool(th <= 30.0 or th >= 150.0 or q == 1.0)))
    for R_f in RF_GRID:
        rho_f = 1.0e-26
        g1, g2, Sig = component_shear("filament", (rho_f, R_f * KPC), sky, scr, sc)
        r = E.bridge_statistics(g1, g2, sky, geom, scr, return_maps=True)
        Mf = float(Sig.sum() * sky.pix ** 2)
        atoms.append(dict(kind="filament", R_f_kpc=R_f, rho_f_unit=rho_f,
                          mass_unit_Msun=Mf / MSUN, kres=r["kappa_res"], aligned=True))
    return atoms


def fit(target: np.ndarray, atoms: List[Dict], mask: np.ndarray, allowed=None):
    idx = [i for i, a in enumerate(atoms) if allowed is None or allowed(a)]
    A = np.stack([atoms[i]["kres"][mask] for i in idx], 1)
    b = target[mask]
    w_nn, rn = nnls(A, b, maxiter=20000)
    w_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    bb = float(b @ b)
    r_nn = float(np.sum((A @ w_nn - b) ** 2))
    r_ls = float(np.sum((A @ w_ls - b) ** 2))
    return dict(idx=idx, w_nnls=w_nn, w_lstsq=w_ls,
                mimic_nnls=1.0 - r_nn / bb, mimic_lstsq=1.0 - r_ls / bb,
                target_power=bb)


def budget_curve(target, atoms, mask, allowed=None, M_scene=6.0e14,
                 budgets=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)) -> Dict:
    """The positive-mass mimic fraction as a function of the ADDED mass
    allowed, relative to the scene's own lens mass (2 x 3e14 Msun): a mimic
    that needs many times the mass of the clusters it is supposed to be is
    not a mimic of this system.  Implemented as NNLS on the system augmented
    with a mass-penalty row sqrt(mu) m_i, mu scanned so that the total added
    mass crosses each budget."""
    idx = [i for i, a in enumerate(atoms) if allowed is None or allowed(a)]
    A = np.stack([atoms[i]["kres"][mask] for i in idx], 1)
    b = target[mask]
    m = np.array([atoms[i]["mass_unit_Msun"] for i in idx])
    bb = float(b @ b)
    scale = float(np.sqrt((A ** 2).sum() / len(idx))) / max(m.mean(), 1e-300)
    rows = []
    for mu in np.geomspace(1e-6, 1e4, 41):
        A2 = np.vstack([A, math.sqrt(mu) * scale * m[None, :]])
        b2 = np.concatenate([b, [0.0]])
        w, _ = nnls(A2, b2, maxiter=20000)
        added = float(w @ m)
        r = float(np.sum((A @ w - b) ** 2))
        rows.append((added / M_scene, 1.0 - r / bb))
    rows.sort()
    out = {}
    for B in budgets:
        ok = [q for a, q in rows if a <= B]
        out[str(B)] = max(ok) if ok else 0.0
    return dict(mimic_by_budget=out, curve=rows,
                note="mimic fraction at added mass <= budget x (2 x 3e14 Msun)")


def describe(w, idx, atoms) -> Dict:
    tot = float(w.sum())
    if tot <= 0:
        return dict(total_weight=0.0)
    halo_mass = sum(w[j] * atoms[i]["mass_unit_Msun"] for j, i in enumerate(idx)
                    if atoms[i]["kind"] == "halo")
    fil_mass = sum(w[j] * atoms[i]["mass_unit_Msun"] for j, i in enumerate(idx)
                   if atoms[i]["kind"] == "filament")
    perp = sum(w[j] for j, i in enumerate(idx) if atoms[i]["kind"] == "halo"
               and 60.0 <= atoms[i]["theta"] <= 120.0 and atoms[i]["q"] < 1.0)
    par = sum(w[j] for j, i in enumerate(idx) if atoms[i]["kind"] == "halo"
              and atoms[i]["aligned"] and atoms[i]["q"] < 1.0)
    off = sum(w[j] for j, i in enumerate(idx) if atoms[i]["kind"] == "halo"
              and atoms[i]["offset_over_a"] != 0.0)
    top = sorted(range(len(w)), key=lambda j: -w[j])[:8]
    return dict(total_weight=tot,
                added_halo_mass_Msun=float(halo_mass),
                added_filament_mass_Msun=float(fil_mass),
                weight_fraction_perpendicular=float(perp / tot),
                weight_fraction_aligned=float(par / tot),
                weight_fraction_offset=float(off / tot),
                top_components=[dict({k: v for k, v in atoms[idx[j]].items() if k != "kres"},
                                     weight=float(w[j])) for j in top if w[j] > 0])


def main():
    guard.arm()
    t0 = time.perf_counter()
    pf = BC.load("fid")
    sc = pf.scene
    D = sc.D
    sky = L.default_sky(D, pix=80.0 * KPC)
    sv = L.Survey()
    scr = sv.sigma_crit_eff
    geom = E.Geometry(xA=-0.5 * D, xB=0.5 * D)
    masks = E.region_masks(sky, geom)
    regions = {
        "between": masks["core_between"] | masks["wing_between"],
        "between+behind": (masks["core_between"] | masks["wing_between"]
                           | masks["core_behind"] | masks["wing_behind"]),
        "wide": masks["net_between"] | masks["net_behind"],
    }
    t1 = time.perf_counter()
    atoms = build_dictionary(sky, scr, sc, geom)
    print(f"dictionary: {len(atoms)} positive-mass components in {time.perf_counter() - t1:.0f}s")

    iso: Dict = {}
    out = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               dictionary=dict(n=len(atoms), q=Q_GRID, theta_deg=TH_GRID,
                               a_over_a=A_GRID, offset_over_a=D_GRID, R_f_kpc=RF_GRID,
                               pixel_kpc=sky.pix / KPC),
               fits={})
    for incl in (0.0, 30.0, 60.0):
        maps = E.law_kappa_maps("P", pf, sky, scr, eps=EPS_REF, rho_star=S.RHO_STAR_FID,
                                incl_deg=incl, iso_cache=iso)
        gP = E.Geometry(xA=maps["xA"], xB=maps["xB"])
        rP = E.bridge_statistics(maps["g1"], maps["g2"], sky, gP, scr, return_maps=True)
        target = rP["kappa_res"]
        if incl > 0:
            # the dictionary is built for the projected centres at +-D/2; at an
            # inclination the endpoints sit at +-D cos(i)/2, so rebuild the atoms
            atoms_i = build_dictionary(sky, scr, S.TwoBody(D=D * math.cos(math.radians(incl)),
                                                          a=sc.a), gP)
            masks_i = E.region_masks(sky, gP)
        else:
            atoms_i, masks_i = atoms, masks
        regs = {"between": masks_i["core_between"] | masks_i["wing_between"],
                "between+behind": (masks_i["core_between"] | masks_i["wing_between"]
                                   | masks_i["core_behind"] | masks_i["wing_behind"]),
                "wide": masks_i["net_between"] | masks_i["net_behind"]}
        for rname, mask in regs.items():
            for prior, allowed in (("any_orientation", None),
                                   ("cdm_prior_aligned", lambda a: a["aligned"])):
                f = fit(target, atoms_i, mask, allowed)
                key = f"incl{incl:.0f}|{rname}|{prior}"
                stats_mimic = None
                # the strip statistics of the best NNLS mimic vs the target
                mimic_map = sum(f["w_nnls"][j] * atoms_i[i]["kres"] for j, i in enumerate(f["idx"]))
                def strips(km):
                    m_ = masks_i
                    S_ = {k: float(km[m_[k]].mean()) for k in ("core_between", "wing_between",
                                                               "core_behind", "wing_behind",
                                                               "net_between", "net_behind")}
                    return dict(Q=S_["core_between"] - S_["wing_between"],
                                DQ=(S_["core_between"] - S_["core_behind"])
                                   - (S_["wing_between"] - S_["wing_behind"]),
                                N_kappa=(S_["net_between"] - S_["net_behind"])
                                        * float(m_["net_between"].sum()) * sky.pix ** 2,
                                S_core=S_["core_between"], S_wing=S_["wing_between"],
                                S_bcore=S_["core_behind"])
                bc = budget_curve(target, atoms_i, mask, allowed)
                out["fits"][key] = dict(
                    incl_deg=incl, region=rname, prior=prior, n_atoms=len(f["idx"]),
                    mimic_fraction_positive_mass=f["mimic_nnls"],
                    mimic_fraction_unconstrained_sign=f["mimic_lstsq"],
                    target_power=f["target_power"],
                    best_positive=describe(f["w_nnls"], f["idx"], atoms_i),
                    mimic_by_mass_budget=bc["mimic_by_budget"],
                    strips_target=strips(target), strips_mimic=strips(mimic_map))
                r = out["fits"][key]
                print(f"{key:<45} mimic: positive {r['mimic_fraction_positive_mass']:.3f}  "
                      f"any-sign {r['mimic_fraction_unconstrained_sign']:.3f}  "
                      f"(added halo mass {r['best_positive'].get('added_halo_mass_Msun', 0):.2e} Msun, "
                      f"perp frac {r['best_positive'].get('weight_fraction_perpendicular', 0):.2f}, "
                      f"offset frac {r['best_positive'].get('weight_fraction_offset', 0):.2f}); "
                      f"Q target {r['strips_target']['Q']:+.2e} mimic {r['strips_mimic']['Q']:+.2e}; "
                      f"DQ {r['strips_target']['DQ']:+.2e} / {r['strips_mimic']['DQ']:+.2e}; "
                      f"budget: " + " ".join(f"{k}:{v:.2f}" for k, v in r["mimic_by_mass_budget"].items()))
    # a single component's best effort, for intuition: which one atom does best?
    best = None
    tgt = None
    maps = E.law_kappa_maps("P", pf, sky, scr, eps=EPS_REF, rho_star=S.RHO_STAR_FID, iso_cache=iso)
    tgt = E.bridge_statistics(maps["g1"], maps["g2"], sky, geom, scr, return_maps=True)["kappa_res"]
    m = regions["between+behind"]
    for a in atoms:
        x = a["kres"][m]
        b = tgt[m]
        w = float(max(x @ b, 0.0) / max(x @ x, 1e-300))
        q = 1.0 - float(np.sum((w * x - b) ** 2) / (b @ b))
        if best is None or q > best[0]:
            best = (q, {k: v for k, v in a.items() if k != "kres"}, w)
    out["best_single_component"] = dict(mimic_fraction=best[0], component=best[1], weight=best[2])
    out["statement"] = (
        "mimic fraction = 1 - ||kappa_res(P) - kappa_res(mimic)||^2 / ||kappa_res(P)||^2 "
        "over the fit region; 'positive' is the NNLS solution over the dictionary "
        "(every component has rho >= 0), 'any-sign' the control with masses of "
        "either sign")
    out["provenance"] = guard.summary()
    out["wall_seconds"] = time.perf_counter() - t0
    with open(os.path.join(RES, "cdm_attack.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote cdm_attack.json in {out['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
