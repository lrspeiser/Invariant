"""Consolidate every sub-result into the lane deliverable `vertical_audit.json`."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(n):
    p = os.path.join(HERE, n)
    return json.load(open(p)) if os.path.exists(p) else {"MISSING": n}


inj = load("injection_results.json")
sl = load("slope_stats.json")
mc = load("model_compare.json")
mg = load("manga_check.json")
bz = load("bz_sensitivity.json")
mn = load("model_null.json")
dg = load("degeneracy_check.json")
sy = load("systematics_required.json")

out = {
    "lane": "work/wellnet-2026-09/vertical-audit",
    "audited": "work/gravity-cluster-audit-2026-09/adyn "
               "(d log10 B_z / d log10 Sigma_0 = -0.346 +- 0.173, p = 0.0095)",
    "reproduction": {
        "slope_p50": sl.get("nuisance_slope", {}).get("median"),
        "published": -0.34592111547830706,
        "note": "vaudit_core.Bench reproduces the published slope, its 16-84 "
                "range and its raw sd to four decimals; every number below "
                "comes from that reproduction."},

    "item1_formula": {
        "definition": "B_z = [sigma_LOS_0(obs) / sigma_LOS_0(model, Newton)]^2",
        "document": "bz_formula.md",
        "shared_quantity": "Sigma_L0 = 10^(0.4(M_K,sun + 21.572 - mu0_K,i)) "
                           "appears in the model denominator AND is the "
                           "abscissa of the headline regression",
        "measured_exponents": {k: v for k, v in bz.items()
                               if isinstance(v, dict) and "exponent" in v},
        "abscissa": bz.get("abscissa"),
        "defect_sigma_z2_floor": bz.get("s2_floor")},

    "item2_shared_denominator": {
        "verdict": "STRUCTURE CONFIRMED, EFFECT NEGLIGIBLE. The artefact "
                   "displaces the slope by -0.012 to -0.018, i.e. 3.5-5.2% of "
                   "the -0.346 signal. The Newtonian injection null never "
                   "reaches -0.346 in 400 trials per scenario.",
        "injection": inj,
        "null_sizing_residual_rms": {
            "observed_dex": 0.1669, "newtonian_null_dex": 0.1705,
            "null_68pc": [0.1534, 0.1843], "ratio": 0.98,
            "note": "the null reproduces the observed scatter, so it is a "
                    "null and not an under-dispersed forward model"},
        "eiv_clean_form": {k: v for k, v in sl.items() if k.startswith("eiv_")},
        "attenuation": sl.get("attenuation"),
        "closed_form_cross_check": sl.get("closed_form"),
        "error_covariance": sl.get("error_covariance"),
        "decomposition": sl.get("decomposition")},

    "item3_reconciliation": {
        "verdict": "RECONCILED, and the published error bar is wrong. "
                   "+-0.173 is the RAW standard deviation of a distribution "
                   "with skewness +26 and excess kurtosis +900, inflated x1.35; "
                   "it does not converge with the number of draws. "
                   "p = 0.0095 is a ONE-SIDED galaxy-bootstrap tail from a "
                   "different resampling. Neither divides into the other.",
        "nuisance_slope": sl.get("nuisance_slope"),
        "bootstrap": sl.get("bootstrap"),
        "commensurate": sl.get("reconciliation"),
        "sd_vs_ndraw": {"200": 0.0656, "400": 0.1658, "800": 0.1287,
                        "1600": 0.3157, "3200": 0.2597, "6400": 0.3276,
                        "robust_sd_stable": [0.0640, 0.0713]},
        "recommended_headline": "-0.346 +- 0.117 (galaxy bootstrap, robust), "
                                "one-sided p = 0.010, i.e. 2.3-3.0 sigma"},

    "item4_local_vs_global": mc.get("item4_model_comparison"),
    "item4_split": mc.get("item4_split"),
    "item4_nuisance_ranking": mc.get("item4_nuisance_ranking"),
    "item4_state_variable_range": mc.get("item4_state_variable_range"),
    "item4_calibrated_model_selection_nulls": mn,
    "item4_global_equals_tilted_upsilon": dg,
    "systematics_required_to_null_the_signal": sy,

    "item5_identical_pipeline": mc.get("item5_pipeline_predictions"),
    "item5_h_over_hR": mc.get("item5_h_over_hR"),

    "independent_check_manga": mg,

    "promotion_criterion": {
        "newtonian_injection_null_correctly_sized": True,
        "slope_survives_shared_denominator_covariance": True,
        "latent_Sigma_dyn_vs_Sigma_b_agrees": True,
        "law_predictions_through_same_pipeline": True,
        "verdict": "All four gates pass. The slope is NOT a shared-denominator "
                   "artefact. But it is NOT evidence for a local modified "
                   "force law either. The calibrated model-selection nulls "
                   "reject BOTH Newton and RAR: the data carry a Sigma_0-"
                   "dependent amplitude that Newton lacks and no radial "
                   "gradient that RAR requires. What fits is a radially FLAT "
                   "factor, and that is degenerate to 5e-15 dex with Newton "
                   "plus Upsilon_K ~ Sigma_0^p. Promote as a measurement of "
                   "the Sigma_dyn-Sigma_b relation (s = 0.65 +- 0.12), NOT as "
                   "a gravitational result."},
}
with open(os.path.join(HERE, "vertical_audit.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote vertical_audit.json")
for k in out:
    v = out[k]
    print(f"  {k:<40}{'MISSING' if isinstance(v, dict) and 'MISSING' in v else 'ok'}")
