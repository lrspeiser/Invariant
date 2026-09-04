"""JOB 3 -- the measurement.  Runs ONLY after certificate_voidcmb.json is issued.

This is the single place the blind guard is disarmed.  Every analysis choice was
fixed in estimator.py and pathmap.py before this file was ever executed; nothing
here selects a cut, a model or a null after seeing a value.

    python measure.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import certify_voidcmb as CV                                   # noqa: E402
import estimator as E                                          # noqa: E402

RNG = np.random.default_rng(4092026)

# ---- LCDM ISW normalisation for the I_phi_in template, declared with its inputs
OMEGA_M, DELTA_VOID = 0.315, 0.7
H0_INV_MPCH = 1.0 / 2997.92458          # H0/c in (Mpc/h)^-1
F_GROWTH = OMEGA_M ** 0.55
# dT/T = 2 H (f-1) int(phi) dchi, phi_inside = -(1/4) H0^2 Om delta (3R^2 - r^2)
ISW_UK_PER_JPHI = (2.7255e6 * 0.5 * H0_INV_MPCH ** 3 * OMEGA_M * DELTA_VOID
                   * (F_GROWTH - 1.0))


def summarise(est, null, resp, label):
    null = np.asarray(null, float)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (est - mu) / sd
    p = (np.sum(np.abs(null - mu) >= abs(est - mu)) + 1) / (len(null) + 1)
    return dict(label=label, c2_over_c1=float(est),
                c2_over_c1_corrected=float(est / resp),
                null_mean=mu, null_sd=sd, z_null_calibrated=float(z),
                p_two_sided=float(p), n_null=int(len(null)),
                abs_limit_95=float(abs(est) + 1.96 * sd),
                abs_limit_95_corrected=float((abs(est) + 1.96 * sd) / resp),
                responsiveness=float(resp))


def run(ctx, model, n_null=1200, seed=11, inject=None, tag=""):
    bank, tried, dets = CV.rotation_bank(ctx, n_null, seed=seed)
    ev = [ctx.evaluate(R, model, inject=inject, what="null") for R in bank]
    null = np.array([e["c2c1"] for e in ev])
    true = ctx.evaluate(None, model, inject=inject, what="MEASUREMENT")
    out = summarise(true["c2c1"], null, 0.9814, f"{tag}{model}")
    out.update(beta_uK_per_mpch=true["beta"], n_pixels=int(true["n"]),
               n_pixels_null_mean=float(np.mean([e["n"] for e in ev])),
               sd_analytic_ols=float(true["sd_ols"] * abs(E.UK_PER_MPCH_TO_C2C1)),
               null_sd_over_analytic=float(np.std(null, ddof=1)
                                           / (true["sd_ols"] * abs(E.UK_PER_MPCH_TO_C2C1))),
               null_sd_proper=float(null[dets > 0].std(ddof=1)),
               null_sd_reflected=float(null[dets < 0].std(ddof=1)),
               rotations_tried=int(tried))
    return out, true, null


def main():
    t0 = time.time()
    cert = json.loads(io.open(os.path.join(HERE, "certificate_voidcmb.json"),
                              encoding="utf-8").read())
    if not cert.get("issued"):
        print("Stage 4 certificate NOT issued -- refusing to measure.")
        return 3
    print("certificate issued; disarming the blind guard once.", flush=True)

    res = {"certificate_issued": True, "isw_norm_uK_per_Jphi": ISW_UK_PER_JPHI,
           "variants": {}, "headline": None}

    # ---------------- headline: nside 64, 5 deg erosion, PR3 SMICA-noSZ, M2
    ctx = CV.Context()
    ctx.guard.disarm("Stage 4 certificate issued for (geometric path redshift, "
                     "beta = dT/d dI_q); measurement authorised")
    for m in ("M1_no_isw", "M2_isw_marginalised", "M3_hardened"):
        out, true, null = run(ctx, m, n_null=2000, seed=7)
        res["variants"][m] = out
        if m == E.HEADLINE:
            res["headline"] = out
            # the ISW coefficient sits at index 5 in M2's design
            isw_coef = float(true["coef"][5])
            isw_null = None
            res["isw"] = dict(
                coef_uK_per_Jphi=isw_coef,
                lcdm_expected_uK_per_Jphi=float(ISW_UK_PER_JPHI),
                ratio_to_lcdm=float(isw_coef / ISW_UK_PER_JPHI),
                rms_isw_signal_uK=float(abs(isw_coef) * ctx.map_t["I_phi_in"].std()),
                lcdm_rms_isw_uK=float(abs(ISW_UK_PER_JPHI)
                                      * ctx.map_t["I_phi_in"].std()),
                corr_template_with_dIq=float(np.corrcoef(
                    ctx.map_t["dI_q"], ctx.map_t["I_phi_in"])[0, 1]),
                shift_M1_to_M2=float(res["variants"]["M2_isw_marginalised"]["c2_over_c1"]
                                     - res["variants"]["M1_no_isw"]["c2_over_c1"]),
                error_inflation_from_marginalising=float(
                    res["variants"]["M2_isw_marginalised"]["null_sd"]
                    / res["variants"]["M1_no_isw"]["null_sd"]))
        print(f"  {m:22s} c2/c1 = {out['c2_over_c1']:+.5f}  "
              f"({out['z_null_calibrated']:+.2f} sigma, p={out['p_two_sided']:.3f})",
              flush=True)

    # ---------------- variants: every declared robustness axis
    E.MODELS["S_near"] = ["I_q_near", "I_phi_in"]
    E.MODELS["S_far"] = ["I_q_far", "I_phi_in"]
    E.MODELS["S_nonedge"] = ["I_q_nonedge", "I_phi_in"]
    E.MODELS["S_isw_k3"] = ["dI_q", "I_phi_k3"]
    E.MODELS["S_no_dipole_only"] = ["dI_q"]

    for m in ("S_near", "S_far", "S_nonedge", "S_isw_k3"):
        out, _, _ = run(ctx, m, n_null=1000, seed=21)
        res["variants"][m] = out
        print(f"  {m:22s} c2/c1 = {out['c2_over_c1']:+.5f}  "
              f"({out['z_null_calibrated']:+.2f} sigma)", flush=True)

    # second component-separation map, different release
    ctx2 = CV.Context(map_key="smica_pr2")
    ctx2.guard.disarm("variant arm")
    out, _, _ = run(ctx2, E.HEADLINE, n_null=1000, seed=31, tag="PR2_")
    res["variants"]["map_smica_PR2_nside1024"] = out
    print(f"  {'PR2 SMICA':22s} c2/c1 = {out['c2_over_c1']:+.5f}  "
          f"({out['z_null_calibrated']:+.2f} sigma)", flush=True)

    # erosion and resolution
    for pm, lbl in (("pathmap_ns64_er2.npz", "erode_2deg"),
                    ("pathmap_ns64_er8.npz", "erode_8deg"),
                    ("pathmap_ns128_er5.npz", "nside_128")):
        nside = 128 if "ns128" in pm else 64
        c = CV.Context(pathmap=pm, nside=nside)
        c.guard.disarm("variant arm")
        out, _, _ = run(c, E.HEADLINE, n_null=800, seed=41, tag=lbl + "_")
        res["variants"][lbl] = out
        print(f"  {lbl:22s} c2/c1 = {out['c2_over_c1']:+.5f}  "
              f"({out['z_null_calibrated']:+.2f} sigma, n={out['n_pixels']})", flush=True)

    # ---------------- responsiveness, measured on the real sky at the true placement
    resp = {}
    for a in (0.001, 0.002, 0.004, -0.004):
        v = ctx.evaluate(None, E.HEADLINE,
                         inject=("dI_q", CV.BETA_PER_C2C1 * a), what="responsiveness")
        resp[f"{a:+.4f}"] = float(v["c2c1"])
    xs = np.array([float(k) for k in resp])
    ys = np.array([resp[k] for k in resp])
    slope = float(np.polyfit(xs, ys, 1)[0])
    res["responsiveness"] = dict(
        injected_vs_recovered=resp, slope_in_grammar=slope,
        pixelisation_factor=float(cert["checks"]["C1_responsive"]
                                  ["responsiveness_pixelisation"]),
        total=float(slope * cert["checks"]["C1_responsive"]["responsiveness_pixelisation"]),
        out_of_grammar=cert["checks"]["C6_out_of_grammar"]["recovery"])

    # ---------------- the comparison the task asks for
    h = res["headline"]
    sd = h["null_sd"]
    res["comparison_to_AK_bound"] = dict(
        ak_bound_lo=CV.AK_BOUND_LO, ak_bound_hi=CV.AK_BOUND_HI,
        ak_bound_this_map=float(4.0e-5 / ctx.map_t["dI_q"].std() / (100.0 / 299792.458)),
        measured=h["c2_over_c1_corrected"],
        limit_95=h["abs_limit_95_corrected"],
        tighter_than_ak_lo=float(CV.AK_BOUND_LO / h["abs_limit_95_corrected"]),
        tighter_than_ak_hi=float(CV.AK_BOUND_HI / h["abs_limit_95_corrected"]),
        sigma_at_ak_lo=float(CV.AK_BOUND_LO / sd),
        sigma_at_ak_hi=float(CV.AK_BOUND_HI / sd),
        three_sigma_floor=float(3 * sd / 0.9814))
    res["seconds"] = time.time() - t0
    res["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["run_id"] = "BI-voidcmb"
    p = os.path.join(HERE, "voidcmb_results.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(res, indent=1, default=float))
    print(f"\nwrote {p}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
