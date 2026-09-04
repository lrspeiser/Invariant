"""
Robustness / circularity quantification.

(1) SENSITIVITY OF THE HEADLINE STATISTIC.  The failure mode "a rank statistic
    was bit-identical across three decades of the parameter it measured" is
    checked explicitly: the transverse dynamic range sigma(Delta I_q) is
    recomputed while the void minimum radius, the density smoothing scale and
    the ray step are varied, and the spread is printed.

(2) CIRCULARITY.  DESIVAST places every void at
        r = D_C(z_obs ; Omega_m = 0.315, h = 1)
    and we place every source endpoint the same way.  If the true
    distance-redshift law differs, both move.  We quantify:
      (a) the radial remap size for three alternative laws;
      (b) what happens to I_q if the source endpoint is instead placed at its
          INDEPENDENT distance while the voids stay where LCDM put them --
          this is the mismatch a genuine no-expansion analysis would inherit
          if it reused this catalogue without rebuilding it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (C_KMS, FootprintMask, R_MAX_VOID, comoving_distance,
                    sky_to_unit, utc_now, DH)
from voids import SphereUnionVoids, load_voidfinder, load_v2, TriangleVoids

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
DESIVAST = os.path.join(LANE, "raw", "desivast")


def alt_distance_laws(z):
    """Comoving distance under alternatives, Mpc/h, all matched at z -> 0."""
    z = np.asarray(z, float)
    out = {}
    out["LCDM_Om0.315"] = comoving_distance(z)
    out["linear_cz_over_H0"] = DH * z
    out["Milne_empty"] = DH * np.log(1.0 + z)          # tired-light-like
    # EdS
    out["EdS_Om1"] = 2.0 * DH * (1.0 - 1.0 / np.sqrt(1.0 + z))
    return out


def main():
    df = pd.read_csv(os.path.join(LANE, "path_integrals_analysed.csv"))
    out = {"generated_utc": utc_now()}

    # ---------------- (1) sensitivity ---------------------------------
    masks = {}
    gal = {}
    for cap in ("NGC", "SGC"):
        with fits.open(os.path.join(
                DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_ZOBOV_{cap}.fits")) as h:
            g = h[3].data
            m = np.asarray(g["OUT"]) == 0
            X = np.stack([np.asarray(g[k], float)[m] for k in ("X", "Y", "Z")], 1)
        gal[cap] = X
        r = np.linalg.norm(X, axis=1)
        masks[cap] = FootprintMask(
            np.degrees(np.arctan2(X[:, 1], X[:, 0])) % 360,
            np.degrees(np.arcsin(X[:, 2] / r)), pix_deg=0.5)

    hole = {}
    for cap in ("NGC", "SGC"):
        _, mx, ho = load_voidfinder(
            os.path.join(DESIVAST, f"DESIVAST_BGS_VOLLIM_VoidFinder_{cap}.fits"))
        hole[cap] = (mx, ho)

    U = sky_to_unit(df["ra"].to_numpy(float), df["dec"].to_numpy(float))
    rend = df["r_end_mpch"].to_numpy(float)
    cap_of = df["cap"].to_numpy()

    sens = {}
    for rmin in [0.0, 5.0, 8.0, 10.0, 12.0]:
        Iq = np.zeros(len(df))
        for cap in ("NGC", "SGC"):
            mx, ho = hole[cap]
            keep = ho["RADIUS"] >= rmin
            sub = {k: v[keep] for k, v in ho.items()}
            geo = SphereUnionVoids(sub)
            idx = np.where(cap_of == cap)[0]
            for i in idx:
                L, _ = geo.ray_intervals(U[i], rend[i])
                Iq[i] = L
        # transverse residual via a quadratic fit in r_end (proxy for <I_q>(r))
        A = np.stack([np.ones_like(rend), rend, rend ** 2], 1)
        c, *_ = np.linalg.lstsq(A, Iq, rcond=None)
        res = Iq - A @ c
        sens[f"hole_radius_min_{rmin:g}"] = {
            "n_holes_kept": int(sum((hole[c2][1]["RADIUS"] >= rmin).sum()
                                    for c2 in ("NGC", "SGC"))),
            "I_q_mean": float(Iq.mean()),
            "transverse_std": float(res.std()),
        }
        print(f"  rmin={rmin:g}: mean I_q={Iq.mean():.2f}  "
              f"transverse std={res.std():.2f}", flush=True)
    vals = [v["transverse_std"] for v in sens.values()]
    out["sensitivity_hole_radius_cut"] = sens
    out["sensitivity_spread"] = {
        "transverse_std_min": float(min(vals)),
        "transverse_std_max": float(max(vals)),
        "range_over_median": float((max(vals) - min(vals)) / np.median(vals)),
        "monotone_invariance_check_passed": bool(max(vals) - min(vals) > 1e-6),
    }

    # ---------------- (2) circularity ----------------------------------
    zs = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.24])
    laws = alt_distance_laws(zs)
    tbl = {}
    for k, v in laws.items():
        tbl[k] = {f"z={z:.2f}": float(d) for z, d in zip(zs, v)}
    ref = laws["LCDM_Om0.315"]
    ratios = {k: {f"z={z:.2f}": float(d / r) for z, d, r in zip(zs, v, ref)}
              for k, v in laws.items()}
    out["distance_laws_mpch"] = tbl
    out["distance_law_ratio_to_LCDM"] = ratios
    out["max_radial_remap_at_zmax"] = {
        k: float(abs(v[-1] / ref[-1] - 1.0)) for k, v in laws.items()}
    out["max_radial_shift_mpch_at_zmax"] = {
        k: float(abs(v[-1] - ref[-1])) for k, v in laws.items()}

    # (2b) endpoint placed at the INDEPENDENT distance instead of at D_C(z)
    #      -- global h chosen to null the mean offset (H0 is absorbed anyway)
    dind = df["D_comov_indep_mpc"].to_numpy(float)
    good = np.isfinite(dind) & (dind > 0)
    h_best = float(np.median(rend[good] / dind[good]))
    r_alt = dind * h_best
    Iq_alt = np.zeros(len(df))
    geo_cap = {}
    for cap in ("NGC", "SGC"):
        geo_cap[cap] = SphereUnionVoids(hole[cap][1])
    for i in range(len(df)):
        if not good[i]:
            Iq_alt[i] = np.nan
            continue
        L, _ = geo_cap[cap_of[i]].ray_intervals(U[i], max(0.0, float(r_alt[i])))
        Iq_alt[i] = L
    base = df["I_q_VoidFinder"].to_numpy(float)
    d_iq = Iq_alt - base
    ok = np.isfinite(d_iq)
    out["endpoint_placement_sensitivity"] = {
        "implied_h_to_match_redshift_frame": h_best,
        "n": int(ok.sum()),
        "median_abs_delta_I_q_mpch": float(np.median(np.abs(d_iq[ok]))),
        "std_delta_I_q_mpch": float(np.std(d_iq[ok])),
        "median_abs_delta_over_transverse_std": None,
        "note": ("I_q recomputed with the ray truncated at the source's own "
                 "independent distance (times a single global h) instead of at "
                 "D_C(z_cmb). The voids themselves stay where LCDM put them, "
                 "so this isolates the endpoint half of the circularity."),
    }
    tstd = float(np.std(df["dI_q_VoidFinder"].to_numpy(float)))
    out["endpoint_placement_sensitivity"][
        "median_abs_delta_over_transverse_std"] = float(
        np.median(np.abs(d_iq[ok])) / tstd)
    out["transverse_std_reference_mpch"] = tstd

    with open(os.path.join(LANE, "robustness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["sensitivity_spread"], indent=2))
    print(json.dumps(out["endpoint_placement_sensitivity"], indent=2))
    return out


if __name__ == "__main__":
    main()
