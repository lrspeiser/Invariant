"""Render the report tables straight from the results JSONs.

Every number in REPORT.md comes through here rather than being transcribed by
hand.
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "envvars_results.json"), encoding="utf-8"))
B = json.load(open(os.path.join(HERE, "envvars_build.json"), encoding="utf-8"))

NAME = {"x1": "V1 potential depth", "x2": "V2 vector g_ext",
        "x3": "V3 directionless W", "x4a": "V4a tidal |T~|",
        "x4b": "V4b tidal shape", "x4d": "V4d external |T~_ext|",
        "xr": "-- radius tilt", "xgb": "-- acceleration tilt"}
ORDER = ["x1", "x2", "x3", "x4a", "x4b", "x4d", "xr", "xgb"]


def z_of(b, null):
    sd = null["sd"]
    return (b - null["mean"]) / sd if sd > 0 else float("nan")


def main_table(tag="raw"):
    print(f"\n### beta, {tag} variable, TRAIN, each against its own "
          f"simulated null\n")
    print("| variable | within-object beta | its null E[b\\|H0] | z vs own null "
          "| lev | within-class beta | its null E[b\\|H0] | z vs own null | lev |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        f = R["fits_train"].get(f"{k}_{tag}")
        n = R["null"].get(k)
        if not f or not n:
            continue
        cells = [NAME[k]]
        for est in ("within_object", "within_class"):
            b = f[est]["beta"]
            nl = n[est]
            edge = "*" if f[est].get("at_grid_edge") else ""
            cells += [f"{b:+.4f}{edge}",
                      f"{nl['mean']:+.4f} +- {nl['sem']:.4f}",
                      f"{z_of(b, nl):+.2f}",
                      f"{f[est]['leverage_frac']:.3f}"]
        print("| " + " | ".join(cells) + " |")
    print("\n`*` = the profile minimum sits at the edge of the "
          f"[{-2.0}, {2.0}] grid, so the value is a bound, not an estimate.  "
          "`lev` is the fraction of the variable's Fisher information for beta "
          "that survives the amplitude projection; 0 means beta is not "
          "identified at all.")


def null_table():
    print("\n### The simulated null, per variable and per estimator\n")
    print("| variable | WO E[b\\|H0] | WO sd | WO 95% detectable \\|b\\| | "
          "WC E[b\\|H0] | WC sd | WC 95% detectable \\|b\\| |")
    print("|---|---|---|---|---|---|---|")
    for k in ORDER:
        n = R["null"].get(k)
        if not n:
            continue
        c = [NAME[k]]
        for est in ("within_object", "within_class"):
            d = n[est]
            c += [f"{d['mean']:+.4f} +- {d['sem']:.4f}", f"{d['sd']:.4f}",
                  f"{1.96*d['sd']:.3f}"]
        print("| " + " | ".join(c) + " |")
    n0 = R["null"]["x1"]["within_object"]["n"]
    print(f"\n{n0} realisations.  Every realisation redraws n0^2, rs, epsilon, "
          "beta, alpha for all 542 catalogued systems from the published "
          "errors, plus an assumed sigma_z = 0.005(1+z), and regenerates the "
          "shear independently around the TRUE model.")


def collinearity_table():
    print("\n### Collinearity with a quadratic in (log g_bar, log r)\n")
    print("| variable | R^2 on the solver grid | residual | R^2 at the "
          "shear-measured radii | residual | within-object range | "
          "between-object sd at 1 Mpc |")
    print("|---|---|---|---|---|---|---|")
    for k in ORDER[:6]:
        c = B["collinearity"][k]
        cs = B["collinearity_at_shear_radii"][k]
        lv = B["leverage"][k]
        print(f"| {NAME[k]} | {c['R2']:.4f} | {c['resid_rms']:.4f} | "
              f"{cs['R2']:.4f} | {cs['resid_rms']:.4f} | "
              f"{lv['median_within_object_range']:.4f} | "
              f"{lv['between_object_sd_at_1Mpc']:.4f} |")


def responsiveness_table():
    print("\n### Responsiveness gate  d(beta-hat)/d(beta_injected)\n")
    print("| variable | estimator | beta-hat at inj 0 | at inj 0.30 | slope | "
          "slope error |")
    print("|---|---|---|---|---|---|")
    for k, v in R["responsiveness"].items():
        for est, d in v.items():
            print(f"| {NAME[k]} | {est} | {d['at_0']:+.4f} | "
                  f"{d['at_injected']:+.4f} | {d['slope']:+.4f} | "
                  f"{d['slope_err']:.4f} |")


def transfer_table(tag="raw"):
    print(f"\n### Frozen transfer to the held-out half, {tag} variable, "
          "touched once\n")
    print("| variable | WO dchi2 | WO dBIC | WC dchi2 | WC dBIC |")
    print("|---|---|---|---|---|")
    for k in ORDER:
        t = R["frozen_transfer"].get(f"{k}_{tag}")
        if not t:
            continue
        print(f"| {NAME[k]} | {t['within_object']['dchi2']:+.3f} | "
              f"{t['within_object']['dBIC']:+.2f} | "
              f"{t['within_class']['dchi2']:+.3f} | "
              f"{t['within_class']['dBIC']:+.2f} |")
    print("\nPositive dchi2 = the frozen model fits the held-out half better "
          "than the same model with beta = 0.  A dBIC below zero would be the "
          "only case in which adding the variable is preferred.")


def sensitivity_table():
    print("\n### Sensitivity: five boundary rules, three smoothing scales\n")
    print("| setting | within-object beta | within-class beta |")
    print("|---|---|---|")
    for k, v in R["sensitivity"].items():
        print(f"| {k} | {v['within_object']['beta']:+.4f} | "
              f"{v['within_class']['beta']:+.4f} |")


def coarse_table():
    print("\n### Coarse-graining, the same continuous mass at N catalogue rows\n")
    print("| system | N rows | drift W (dex) | drift \\|g\\| (dex) | "
          "drift W, selective | drift \\|g\\|, selective |")
    print("|---|---|---|---|---|---|")
    for g in B["coarse_graining"]:
        for N in g["N_grid"]:
            e = g["series"][str(N)]
            print(f"| {g['system']} | {e['n_rows']} | {e['drift_W_dex']:.5f} | "
                  f"{e['drift_g_dex']:.5f} | "
                  f"{e['drift_W_selective_dex']:.5f} | "
                  f"{e['drift_g_selective_dex']:.5f} |")
    print("\n| system | beta_N W | beta_N \\|g\\| | beta_N W sel | "
          "beta_N \\|g\\| sel |")
    print("|---|---|---|---|---|")
    for g in B["coarse_graining"]:
        print(f"| {g['system']} | {g['beta_N_W']:.4f} | {g['beta_N_g']:.4f} | "
              f"{g['beta_N_W_selective']:.4f} | "
              f"{g['beta_N_g_selective']:.4f} |")
    print("\n| system | K pieces per external well | drift W_ext (dex) | "
          "drift \\|g_ext\\| (dex) |")
    print("|---|---|---|---|")
    for g in B["coarse_graining"]:
        for K, e in g["external_refinement"].items():
            print(f"| {g['system']} | {K} | {e['drift_W_ext_dex']:.6f} | "
                  f"{e['drift_g_ext_dex']:.6f} |")


if __name__ == "__main__":
    collinearity_table()
    main_table("raw")
    main_table("perp")
    null_table()
    responsiveness_table()
    transfer_table("raw")
    transfer_table("perp")
    sensitivity_table()
    coarse_table()
