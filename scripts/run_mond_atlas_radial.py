"""Execute a fixed-physics radial baseline and galaxy-level exploratory residual audit.

This stage uses published force templates. It must never be labeled a full-field
MOND disk calculation, a cube likelihood, or an observed 3D reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mond_atlas_common import PROTOCOL, ROOT, digest, read_json, sparc_inputs, write_csv, write_json

CONVERSION = 1e6 / 3.085677581491367e19  # (km/s)^2/kpc -> m/s^2
OUT = ROOT / "work/gravity-first-principles/mond-atlas-radial-001"


def speeds(radius, gas, disk, bulge, a0=1.2e-10, ml_disk=.5, ml_bulge=.7, distance_scale=1.):
    radius, gas, disk, bulge = [np.asarray(v, dtype=float) for v in (radius, gas, disk, bulge)]
    v2 = (gas * np.abs(gas) + ml_disk*disk**2 + ml_bulge*bulge**2) * distance_scale
    gbar = v2 / (radius*distance_scale) * CONVERSION
    valid = np.isfinite(gbar) & (gbar > 0) & (radius > 0)
    newton, mond = np.full_like(gbar, np.nan), np.full_like(gbar, np.nan)
    newton[valid] = np.sqrt(v2[valid])
    nu = .5 + np.sqrt(.25+a0/gbar[valid])
    mond[valid] = np.sqrt(v2[valid]*nu)
    return newton, mond, gbar, valid


def fixed_folds(names, n=5):
    ordered = sorted(set(names), key=lambda n:hashlib.sha256(("mond-atlas-v1|"+n).encode()).digest())
    mapping = {name:i % n for i,name in enumerate(ordered)}
    return np.array([mapping[name] for name in names])


def ridge_predict(x, y, train, test, penalty):
    center = x[train].mean(axis=0)
    scale = np.maximum(x[train].std(axis=0), 1e-12)
    design = (x-center)/scale
    response_center = y[train].mean()
    a = design[train]
    coef = np.linalg.solve(a.T@a + penalty*np.eye(a.shape[1]), a.T@(y[train]-response_center))
    return response_center + design[test]@coef


def nested_predictions(x, y, fold):
    result = np.full(len(y), np.nan)
    for f in sorted(set(fold)):
        train, test = fold != f, fold == f
        penalties = [.1, 1., 10., 100.]
        losses = []
        for penalty in penalties:
            errors = []
            for j in sorted(set(fold[train])):
                inside, validate = train & (fold != j), train & (fold == j)
                pred = ridge_predict(x, y, inside, validate, penalty)
                errors.extend((pred-y[validate])**2)
            losses.append(np.mean(errors))
        result[test] = ridge_predict(x, y, train, test, penalties[int(np.argmin(losses))])
    return result


def pattern_audit(rows, protocol):
    chosen = [r for r in rows if r["selected"] and r["gas_mass_fraction_proxy"] is not None
              and r["effective_stellar_surface_brightness"] > 0 and r["disk_scale_length"] > 0]
    names = [r["galaxy"] for r in chosen]
    folds = fixed_folds(names)
    y = np.array([r["median_log10_vobs_over_mond"] for r in chosen])
    base = np.array([[r["median_log10_gbar"], r["spread_log10_gbar"]] for r in chosen])
    base_pred = nested_predictions(base, y, folds)
    base_mse = np.mean((base_pred-y)**2)
    rng = np.random.default_rng(protocol["random_seed"])
    results, predictions = [], []
    features = {"gas_mass_fraction":np.array([r["gas_mass_fraction_proxy"] for r in chosen]),
        "effective_stellar_surface_brightness":np.log10([r["effective_stellar_surface_brightness"] for r in chosen]),
        "disk_scale_length":np.log10([r["disk_scale_length"] for r in chosen]),
        "hubble_type":np.array([r["hubble_type"] for r in chosen])}
    for name, feature in features.items():
        full = nested_predictions(np.column_stack((base, feature)), y, folds)
        full_mse = np.mean((full-y)**2)
        gain = base_mse-full_mse
        null = []
        for _ in range(protocol["permutations"]):
            shuffled = rng.permutation(feature)
            pred = nested_predictions(np.column_stack((base, shuffled)), y, folds)
            null.append(base_mse - np.mean((pred-y)**2))
        paired = (base_pred-y)**2 - (full-y)**2
        ci = np.quantile(rng.choice(paired, (4000,len(paired))).mean(axis=1), [.025,.975])
        p = (1+sum(v >= gain for v in null))/(len(null)+1)
        results.append(dict(feature=name, galaxy_count=len(y), baseline_rmse_dex=float(np.sqrt(base_mse)),
            extended_rmse_dex=float(np.sqrt(full_mse)), mse_gain_percent=float(100*gain/base_mse),
            paired_mse_gain_bootstrap95=ci.tolist(), permutation_p=float(p),
            bonferroni_p=min(1.,len(features)*p),
            test_status="exploratory_whole_galaxy_cv_not_causal_or_survey_confirmation"))
        predictions.extend(dict(galaxy=n, feature=name, fold=int(folds[i]), residual_target=float(y[i]),
            base_prediction=float(base_pred[i]), extended_prediction=float(full[i])) for i,n in enumerate(names))
    return results, predictions


def run(output):
    if output.exists():
        raise FileExistsError("Use a new run directory: "+str(output))
    output.mkdir(parents=True)
    config = read_json(PROTOCOL)
    physics, selection = config["physics"], config["radial_selection"]
    curves, metadata, photometry, sources = sparc_inputs()
    point_rows, galaxy_rows = [], []
    for galaxy in curves:
        name = galaxy["name"]
        m = metadata[name]
        x = np.array(galaxy["rows"], dtype=float)
        r, observed, error, gas, disk, bulge = x.T
        newton, mond, gbar, valid = speeds(r, gas, disk, bulge, physics["a0_m_s2"],
                                          physics["stellar_disk_ml"], physics["stellar_bulge_ml"])
        valid &= (observed > 0) & (error > 0) & np.isfinite(x).all(axis=1)
        cuts = []
        if m["quality"] > selection["quality_max"]:
            cuts.append("quality")
        if not selection["inclination_deg"][0] <= m["inclination_deg"] <= selection["inclination_deg"][1]:
            cuts.append("inclination")
        if sum(valid) < selection["minimum_valid_radii"]:
            cuts.append("insufficient_valid_radii")
        selected = not cuts
        scenarios = []
        for mass_factor in selection["mass_sensitivity_factors"]:
            for dsign in [-1,0,1]:
                d = m["distance_mpc"] + dsign*m["distance_error_mpc"]
                if d <= 0:
                    continue
                for isign in [-1,0,1]:
                    inc = m["inclination_deg"] + isign*m["inclination_error_deg"]
                    if not 0 < inc <= 90 or m["inclination_deg"] <= 0:
                        continue
                    _, p, _, _ = speeds(r,gas,disk,bulge,physics["a0_m_s2"],
                        physics["stellar_disk_ml"]*mass_factor,
                        physics["stellar_bulge_ml"]*mass_factor,d/m["distance_mpc"])
                    scenarios.append(p*np.sin(np.deg2rad(inc))/np.sin(np.deg2rad(m["inclination_deg"])))
        samples = np.array(scenarios)
        # Retain invalid force cases explicitly; no replacing NaN with zero mass.
        low = np.array([np.min(col[np.isfinite(col)]) if np.isfinite(col).any() else np.nan for col in samples.T])
        high = np.array([np.max(col[np.isfinite(col)]) if np.isfinite(col).any() else np.nan for col in samples.T])
        for i in range(len(r)):
            clean = lambda value: float(value) if np.isfinite(value) else None
            sd, sb = photometry[name][i]
            point_rows.append(dict(galaxy=name, radial_index=i, radius_kpc=float(r[i]),
                observed_speed_km_s=float(observed[i]), published_error_km_s=float(error[i]),
                newton_speed_km_s=clean(newton[i]), algebraic_mond_speed_km_s=clean(mond[i]),
                gbar_m_s2=clean(gbar[i]), sensitivity_min_km_s=clean(low[i]), sensitivity_max_km_s=clean(high[i]),
                disk_luminosity_surface_density=sd, bulge_luminosity_surface_density=sb,
                row_valid=bool(valid[i]), selected=bool(selected and valid[i]),
                status="scored_descriptively" if selected and valid[i] else ";".join(cuts) if cuts else "invalid_row",
                depth_information="published_mass_template_assumptions",
                full_field_prediction=False, independent_cube_prediction=False))
        v, p, n, acceleration = observed[valid], mond[valid], newton[valid], gbar[valid]
        gasmass = 1.33*m["hi_mass_1e9_msun"]
        starproxy = physics["stellar_disk_ml"]*m["luminosity_1e9_lsun"]
        galaxy_rows.append(dict(galaxy=name, selected=selected, exclusion=";".join(cuts),
            published_radii=len(r), valid_radii=int(sum(valid)),
            newton_rms_fractional_error=float(np.sqrt(np.mean((n/v-1)**2))) if len(v) else None,
            mond_rms_fractional_error=float(np.sqrt(np.mean((p/v-1)**2))) if len(v) else None,
            median_log10_vobs_over_mond=float(np.median(np.log10(v/p))) if len(v) else None,
            median_log10_gbar=float(np.median(np.log10(acceleration))) if len(v) else None,
            spread_log10_gbar=float(np.ptp(np.log10(acceleration))) if len(v) else None,
            sensitivity_fraction_bracketing_observed=float(np.mean((low[valid]<=v)&(v<=high[valid]))) if len(v) else None,
            gas_mass_fraction_proxy=gasmass/(gasmass+starproxy) if gasmass+starproxy>0 else None,
            effective_stellar_surface_brightness=m["effective_sb_lsun_pc2"],
            disk_scale_length=m["rdisk_kpc"], hubble_type=m["hubble_type"],
            quality=m["quality"], inclination_deg=m["inclination_deg"],
            exposure="legacy_and_current_development_reanalysis"))
    patterns, predictions = pattern_audit(galaxy_rows, config["pattern_analysis"])
    write_csv(output/"radial-predictions.csv", point_rows)
    write_csv(output/"galaxy-residuals.csv", galaxy_rows)
    write_csv(output/"pattern-holdout-predictions.csv", predictions)
    write_json(output/"patterns.json", patterns)
    selected = [r for r in galaxy_rows if r["selected"]]
    summary = dict(status="EXECUTED_FIXED_RADIAL_BASELINE_NOT_FULL_3D_ATLAS",
        generated_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(PROTOCOL),
        input_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sources},
        code_hashes={p.name:digest(p) for p in [Path(__file__), ROOT/"scripts/mond_atlas_common.py"]},
        published_galaxies=len(galaxy_rows), published_radii=len(point_rows),
        selected_galaxies=len(selected), selected_radii=sum(r["valid_radii"] for r in selected),
        galaxy_weighted_newton_rms_fractional_error_percent=float(100*np.sqrt(np.mean([r["newton_rms_fractional_error"]**2 for r in selected]))),
        galaxy_weighted_mond_rms_fractional_error_percent=float(100*np.sqrt(np.mean([r["mond_rms_fractional_error"]**2 for r in selected]))),
        median_galaxy_newton_rms_fractional_error_percent=float(100*np.median([r["newton_rms_fractional_error"] for r in selected])),
        median_galaxy_mond_rms_fractional_error_percent=float(100*np.median([r["mond_rms_fractional_error"] for r in selected])),
        galaxies_mond_lower_fractional_error_than_newton=sum(r["mond_rms_fractional_error"] < r["newton_rms_fractional_error"] for r in selected),
        fixed_a0_m_s2=physics["a0_m_s2"], fitted_gravity_parameters=0, dark_halo_terms=0,
        full_field_mond_galaxies=0, validated_independent_cube_predictions=0,
        patterns=patterns,
        limitations=["Algebraic MOND is a radial approximation; these are not full QUMOND/AQUAL disk solutions.",
            "SPARC force templates omit independently mapped molecular/hot/ionized corrections and have geometric assumptions.",
            "The 27-corner envelope is sensitivity only, not a posterior or confidence region; inclination changes only velocity projection, not source geometry.",
            "Errors are descriptive; correlated radius errors and distance/inclination systematics preclude chi-square significance.",
            "Galaxies were used previously in theory development. Cross-validation is not pristine confirmation and does not hold out physical groups or surveys.",
            "Gas fraction is a total-HI plus uniform stellar-M/L proxy, not an independently measured total baryon fraction or spatial density.",
            "Permutation shuffles are an exploratory exchangeability diagnostic; they do not establish causality or control every morphology/distance confounder.",
            "No universal density/coherence correction is selected or tuned by this run."])
    write_json(output/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
