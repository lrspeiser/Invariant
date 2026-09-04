"""Merge every results file into the single deliverable efeds_hsc_results.json."""
from __future__ import annotations

import datetime as dt
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PARTS = [
    ("decade_raw_shear_test", "decade_results.json"),
    ("hsc_published_stack_test", "hsc_stack_results.json"),
    ("lensing_numerics_gates", "gates.json"),
    ("lensing_amplitude_calibration", "calib_check.json"),
]


def main():
    out = {
        "lane": "work/wellnet-2026-09/efeds-hsc",
        "written_utc": dt.datetime.now(dt.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": {},
    }
    for key, fn in PARTS:
        p = os.path.join(HERE, fn)
        out[key] = json.load(open(p, encoding="utf-8")) \
            if os.path.exists(p) else None
        print(f"   {'ok ' if out[key] else 'MISSING'} {fn}")

    d = out.get("decade_raw_shear_test") or {}
    h = out.get("hsc_published_stack_test") or {}
    mc = (d.get("M_model_comparison") or {}).get("models", {})
    if mc:
        best = min(mc.items(), key=lambda kv: kv[1]["bic"])
        # NULL-CALIBRATED beta.  The Monte Carlo in section D shows that noise
        # in the X-ray density fit alone drives the naive estimator to
        # -0.0666 +- 0.0101 under H0: beta = 0.  Quoting the raw estimate
        # against zero would repeat the mistake that produced the retracted
        # rho_p = -0.304, so the estimate is referred to its own null.
        dn = d.get("D_shared_quantity_null") or {}
        raw = dn.get("linear_estimator_on_data")
        if raw is None:
            raw = (d.get("M_beta") or {}).get("beta", 0.0)
        bias = dn.get("mean", 0.0)
        sd = max(dn.get("sd", 0.0),
                 (d.get("R_responsiveness") or {}).get("sigma_beta", 0.0))
        corr = raw - bias
        out["beta_null_calibrated"] = {
            "beta_raw": raw,
            "null_expectation_H0": bias,
            "null_sd": dn.get("sd"),
            "injection_sd": (d.get("R_responsiveness") or {}).get("sigma_beta"),
            "beta_corrected": corr,
            "sigma_adopted": sd,
            "sigma_from_zero": corr / sd if sd else None,
            "sigma_from_RunR_0.17188": (0.17188 - corr) / sd if sd else None,
            "note": "beta = beta_raw - E[beta_hat | H0].  The uncertainty is "
                    "the larger of the H0 Monte-Carlo scatter and the "
                    "injection-recovery scatter."}
        out["headline"] = {
            "raw_shear_obtainable": True,
            "raw_shear_source": "DECADE metacalibration catalogue, "
                                "delve_dr3.decade_shear, NOIRLab Astro Data "
                                "Lab TAP, unauthenticated",
            "hsc_per_source_shear_obtainable": False,
            "hsc_block": "HTTP 401 on every hsc-release archive route",
            "n_systems_measured": d.get("M_model_comparison", {})
                                   .get("n_train_systems"),
            "tangential_signal_sigma": (d.get("N_nulls") or {}).get("gt_snr"),
            "cross_signal_sigma": (d.get("N_nulls") or {}).get("gx_snr"),
            "beta_potential_depth_raw": (d.get("M_beta") or {}).get("beta"),
            "beta_null_calibrated":
                out.get("beta_null_calibrated", {}).get("beta_corrected"),
            "beta_sigma": out.get("beta_null_calibrated", {})
                             .get("sigma_adopted"),
            "beta_sigma_from_zero":
                out.get("beta_null_calibrated", {}).get("sigma_from_zero"),
            "beta_sigma_from_RunR":
                out.get("beta_null_calibrated", {})
                   .get("sigma_from_RunR_0.17188"),
            "random_point_null_sigma":
                (d.get("X_random_null") or {}).get("gt_sigma"),
            "responsiveness_slope":
                (d.get("R_responsiveness") or {}).get("slope"),
            "dchi2_vs_class_step": (d.get("M_beta") or {})
                                   .get("dchi2_vs_class_step"),
            "best_model_on_BIC": best[0],
            "hsc_within_class_leverage_dex":
                (h.get("A1_leverage_dex") or {}).get("fixed10Mpc"),
            "R2_of_xPhi_on_gbar_and_r":
                (h.get("A2_R2_of_xPhi_on_competitors") or {})
                .get("quadratic in (log g_b, log r)"),
        }
    with open(os.path.join(HERE, "efeds_hsc_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n   wrote efeds_hsc_results.json")
    print(json.dumps(out["headline"], indent=1))


if __name__ == "__main__":
    main()
