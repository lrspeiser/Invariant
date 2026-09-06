"""Run the frozen theory-only correlated-noise motion benchmark on CPU.

Every run writes fresh assigned report/private directories. No observational
data, package installation, network access, Git operation or prior-file mutation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

sys.dont_write_bytecode = True
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from mond_atlas_motion_controls import Geometry, Instrument, direct_reference_cube, forward_cube
from mond_atlas_motion_covariance import (
    FixedPartition, channel_covariance, fit_motion, forecast_evaluation, innovation_noise,
    numerical_controls, pixel_scales, predictive_pass,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "work/gravity-first-principles/mond-atlas-motion-covariance-001"
PRIVATE = ROOT / "work/private/mond-atlas-motion-covariance-001"
CONFIG = ROOT / "configs/mond_atlas_motion_covariance_v1.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def load_frozen():
    freeze = json.loads((REPORT / "freeze.json").read_text())
    config = json.loads(CONFIG.read_text())
    if (freeze["disposition"] != "THEORY_BENCHMARK_ONLY"
            or config["disposition"] != "THEORY_BENCHMARK_ONLY"
            or config["observational_scoring_allowed"] is not False
            or config["source_dataset"] is not None
            or freeze["implementation_started"] is not False
            or freeze["synthetic_study_generated"] is not False
            or digest(CONFIG) != freeze["config_sha256"]
            or digest(REPORT / "PREFLIGHT.md") != freeze["preflight_sha256"]):
        raise ValueError("frozen theory-only config/preflight mismatch")
    for name, expected in freeze["immutable_dependencies"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("immutable prior dependency changed: "+name)
    return config, json.loads((ROOT / config["prior_motion_config"]).read_text()), freeze


METRICS = (
    "signal_error", "fresh_signal_q", "same_marginal_q", "same_conditional_marginal_q",
    "same_conditional_nll", "same_conditional_q", "fresh_transferred_q", "transferred_signal_error",
)


def extract_metrics(row):
    return {
        "signal_error": row["signal_noiseless_true_marginal"]["q_per_cell"],
        "fresh_signal_q": row["fresh_signal_mean_q_per_cell"],
        "same_marginal_q": row["same_signal_true_marginal"]["q_per_cell"],
        "same_conditional_marginal_q": row["same_conditional_common_true_marginal"]["q_per_cell"],
        "same_conditional_nll": row["same_conditional_assumed_distribution"]["nll_per_cell"],
        "same_conditional_q": row["same_conditional_assumed_distribution"]["q_per_cell"],
        "fresh_transferred_q": row["fresh_transferred_mean_q_per_cell"],
        "transferred_signal_error": row["noiseless_transferred_noise_control"]["q_per_cell"],
    }


def describe(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"values": [], "n_realizations": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
    return {"values": a.tolist(), "n_realizations": len(a), "mean": float(a.mean()),
            "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max())}


def summarize(rows, config):
    """Never count overlapping folds as independent Monte Carlo realizations."""
    output = {"unit": "one noise realization, after arithmetic averaging over both frozen folds",
              "independent_realizations_per_case": config["study"]["realizations_per_case"],
              "cases": [], "paired_differences": []}
    methods = config["study"]["methods"]
    for case in config["study"]["cases"]:
        selected = [r for r in rows if r["case"] == case["name"]]
        item = {"case": case["name"], "methods": {}, "oracle": {}}
        # Require both folds for a realization-level aggregate; missing/failed fits remain listed.
        for method in methods+["oracle"]:
            samples = {}
            failures, flags, param_flags = [], [], []
            for realization in range(config["study"]["realizations_per_case"]):
                folds = [r for r in selected if r["realization"] == realization]
                valid = []
                for row in folds:
                    r = row["oracle"] if method == "oracle" else row["methods"][method]
                    if r.get("status") == "FIT_EXCEPTION":
                        failures.append({"realization": realization, "fold": row["fold"], "error": r["exception"]})
                        continue
                    valid.append(r["evaluation"])
                    if method != "oracle":
                        flags.append(bool(r["predictive_pass"]))
                        param_flags.append(bool(r["parameter_pass"]))
                if len(valid) != len(config["masks"]["fold_ids"]):
                    continue
                samples[realization] = {
                    split: {metric: float(np.mean([extract_metrics(v[split])[metric] for v in valid]))
                            for metric in METRICS}
                    for split in valid[0]
                }
            result = {"per_realization_fold_averages": samples, "fit_exceptions": failures,
                      "predictive_passes_over_fold_fits": int(sum(flags)), "fold_fit_count": len(flags),
                      "parameter_passes_over_fold_fits": int(sum(param_flags)), "metrics": {}}
            for split in ("heldout_channels", "heldout_pixels", "heldout_joint"):
                result["metrics"][split] = {metric: describe([v[split][metric] for v in samples.values()]) for metric in METRICS}
            if method == "oracle":
                item["oracle"] = result
            else:
                item["methods"][method] = result
        output["cases"].append(item)
        for first, second in (("expanded_correct", "circular_correct"),
                              ("expanded_diagonal", "circular_diagonal"),
                              ("circular_correct", "circular_diagonal"),
                              ("expanded_correct", "expanded_diagonal")):
            a = item["methods"][first]["per_realization_fold_averages"]
            b = item["methods"][second]["per_realization_fold_averages"]
            keys = sorted(set(a) & set(b))
            output["paired_differences"].append({
                "case": case["name"], "difference": first+" minus "+second,
                "negative_means_first_has_lower_error_or_nll": True,
                "metrics": {split: {metric: describe([a[r][split][metric]-b[r][split][metric] for r in keys])
                                     for metric in METRICS}
                            for split in ("heldout_channels", "heldout_pixels", "heldout_joint")},
            })
    return output


def plot_results(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = summary["aggregate"]["cases"]
    labels = ["Zero extra amplitudes", "Radial streaming", "Combined effects"]
    methods = summary["config_study_methods"]
    colors = ["#5381a6", "#cf723b", "#88a9be", "#e4ae7c"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained", sharex=True)
    x = np.arange(len(cases))
    for j, (method, color) in enumerate(zip(methods, colors)):
        for ax, metric in zip(axes, ("signal_error", "fresh_signal_q")):
            values = [c["methods"][method]["metrics"]["heldout_channels"][metric] for c in cases]
            means = [v["mean"] if v["mean"] is not None else np.nan for v in values]
            sd = [v["sample_sd"] if v["sample_sd"] is not None else 0 for v in values]
            ax.bar(x+(j-1.5)*0.19, means, 0.18, yerr=sd, capsize=3, color=color, label=method.replace("_", " "))
    axes[0].set_ylabel("Noiseless signal error q / N")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[1].set_ylabel("Independent fresh-noise q / N")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels)
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Correlated-channel synthetic motion comparison\nCommon true marginal covariance; bars show mean +/- sample SD over 4 realizations", fontsize=12)
    fig.savefig(out / "signal-and-fresh-noise.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    names = [("same_marginal_q", "Same noise: signal mean"),
             ("same_conditional_marginal_q", "Same noise: conditional forecast"),
             ("fresh_signal_q", "Fresh noise: signal mean"),
             ("fresh_transferred_q", "Fresh noise: transferred old correction")]
    for j, (metric, label) in enumerate(names):
        values = [c["methods"]["expanded_correct"]["metrics"]["heldout_channels"][metric] for c in cases]
        ax.bar(x+(j-1.5)*0.19, [v["mean"] for v in values], 0.18,
               yerr=[v["sample_sd"] for v in values], capsize=3, label=label,
               color=["#557da1", "#3c8b77", "#c37a36", "#a05767"][j])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Point forecast error in true marginal metric q / N")
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Expanded model with correct covariance: interpolation versus signal\nHeld-out channels at training pixels; old-noise transfer is a negative control", fontsize=11)
    fig.savefig(out / "noise-interpolation-control.png", dpi=170)
    plt.close(fig)


def write_readme(summary, out):
    text = ["# Executed correlated-channel motion benchmark", "", "**THEORY_BENCHMARK_ONLY.** No observed source/covariance or native selection admission.",
            f"Run `{summary['run_id']}`: {summary['statistical_controls']} statistical controls and {summary['forward_controls']} prior forward-law controls passed before study generation.",
            "", "Three synthetic cases, four independent noise realizations per case, two fresh",
            "replicates per realization, two overlapping folds and four fitting methods.",
            "Results below average folds within each realization, then report the mean over",
            "four realizations. Full SD/ranges, paired differences, every fit/start, parameter",
            "errors and forecast distributions are retained in summary.json and fold receipts.", "",
            "| Case | Method | Noiseless signal q/N | Fresh q/N | Same conditional q/N | Prediction passes / fits | Parameter passes / fits |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for case in summary["aggregate"]["cases"]:
        for name, result in case["methods"].items():
            m = result["metrics"]["heldout_channels"]
            fmt = lambda key: "missing" if m[key]["mean"] is None else f"{m[key]['mean']:.5f}"
            text.append(f"| {case['case']} | {name} | {fmt('signal_error')} | {fmt('fresh_signal_q')} | {fmt('same_conditional_q')} | {result['predictive_passes_over_fold_fits']}/{result['fold_fit_count']} | {result['parameter_passes_over_fold_fits']}/{result['fold_fit_count']} |")
    text += ["", "The q/N columns are for held-out channels at training pixels. Signal and fresh",
             "errors use the same true marginal covariance for every method. Conditional q/N",
             "uses each method's assumed Schur covariance, so compare log densities including",
             "log determinants in the full receipt when comparing those distributions.",
             "Prediction pass requires all three held-out subsets, not this column alone.",
             "", "![Signal and fresh noise](signal-and-fresh-noise.png)",
             "", "![Noise interpolation](noise-interpolation-control.png)",
             "", "The same-noise correction conditions only on training residuals. Its application",
             "to fresh noise is explicitly a negative control: independent fresh realizations",
             "have zero covariance with the old training noise. Noiseless signal errors and",
             "fresh-data errors prevent that interpolation benefit being counted as motion recovery.",
             "", "All fitted-mean distributions are plug-in diagnostics. The noise covariance",
             "identity is exact for fixed parameters, but nonlinear fitted-parameter uncertainty",
             "is not integrated. Four realizations do not establish coverage or significance.",
             "All source/instrument parameters outside the frozen fit list are known. Pressure",
             "support, force balance, dynamics, source uncertainty, observed spatial/channel",
             "covariance and response-selected gas masks remain unvalidated or missing.",
             "", "From the Invariant root:", "", "```powershell",
             "python -B scripts/run_mond_atlas_motion_covariance.py",
             "python -B -m unittest discover -s tests -p test_mond_atlas_motion_covariance.py -v",
             "```", "", "Every run creates new assigned directories; existing receipts and arrays are immutable."]
    (out / "README.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def execute(run_id=None, controls_only=False):
    started = time.perf_counter()
    config, prior, freeze = load_frozen()
    run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run ID must be a unique simple directory name")
    out, private = REPORT / run_id, PRIVATE / run_id
    if out.exists() or private.exists():
        raise FileExistsError("refusing to overwrite existing report or private run")
    out.mkdir(parents=True)
    private.mkdir(parents=True)
    code_paths = ["scripts/mond_atlas_motion_covariance.py", "scripts/run_mond_atlas_motion_covariance.py",
                  "tests/test_mond_atlas_motion_covariance.py"]
    initial = {"run_id": run_id, "started_utc": utc(), "disposition": "THEORY_BENCHMARK_ONLY",
               "freeze": freeze, "code_hashes": {p: digest(ROOT / p) for p in code_paths},
               "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                           "scipy": scipy.__version__, "platform": platform.platform(), "device": "CPU", "threads": 1},
               "observational_files_opened": 0, "network_requests": 0, "synthetic_study_started": False}
    save_json(out / "execution-start.json", initial)
    try:
        with threadpool_limits(limits=1):
            instrument = Instrument(**prior["instrument"])
            geometry = Geometry(**prior["geometry"])
            shape = (instrument.nchannel, instrument.npix, instrument.npix)
            cov, sigma_c = channel_covariance(instrument.nchannel, config["noise"])
            scale = pixel_scales(instrument.npix, config["noise"])
            partitions = {f: FixedPartition.build(shape, f) for f in config["masks"]["fold_ids"]}
            design_arrays = {"channel_covariance": cov, "channel_sigma": sigma_c, "pixel_scale": scale,
                             "channel_edges": instrument.edges}
            for fold, partition in partitions.items():
                design_arrays.update({f"fold{fold}_{name}": value for name, value in partition.masks().items()})
            np.savez_compressed(private / "design.npz", **design_arrays)
            save_json(out / "pre-response-design.json", {
                "utc": utc(), "design_array_sha256": digest(private / "design.npz"),
                "design_path": (private / "design.npz").relative_to(ROOT).as_posix(),
                "config_sha256": digest(CONFIG),
                "shape": shape, "fold_counts": {str(f): {name: int(mask.sum()) for name, mask in p.masks().items()} for f, p in partitions.items()},
                "noise_draw_indices": [0]+list(range(1, config["study"]["fresh_replicates_per_realization"]+1)),
            })
            controls = numerical_controls(config, prior)
            controls["completed_utc"] = utc()
            save_json(out / "numerical-controls.json", controls)
            if not controls["all_passed"]:
                save_json(out / "failure.json", {"status": "BENCHMARK_FAILED", "synthetic_study_started": False})
                return 2
            save_json(out / "response-access-gate.json", {
                "utc": utc(), "all_statistical_and_forward_controls_passed": True,
                "controls_sha256": digest(out / "numerical-controls.json"),
                "design_sha256": digest(out / "pre-response-design.json"), "config_sha256": digest(CONFIG),
                "observational_scoring_allowed": False, "synthetic_study_may_begin": not controls_only})
            print(f"{controls['statistical_control_count']} statistical + {controls['prior_forward_control_count']} forward controls passed.", flush=True)
            if controls_only:
                return 0
            rows = []
            for ci, case in enumerate(config["study"]["cases"]):
                true_params = dict(prior["base_parameters"], **case["overrides"])
                truth = direct_reference_cube(true_params, geometry, instrument, *config["study"]["fit"]["truth_quadrature"])
                _, flux = forward_cube(true_params, geometry, instrument, accounting=True)
                truth_file = private / f"truth-{case['name']}.npz"
                np.savez_compressed(truth_file, truth=truth)
                save_json(out / f"truth-{case['name']}.json", {"parameters": true_params, "production_flux_accounting": flux,
                          "truth_path": truth_file.relative_to(ROOT).as_posix(), "truth_sha256": digest(truth_file)})
                for realization in range(config["study"]["realizations_per_case"]):
                    trial_start = time.perf_counter()
                    noises = []
                    seed_vectors = []
                    for draw in range(config["study"]["fresh_replicates_per_realization"]+1):
                        seed = [config["study"]["seed"], ci, realization, draw]
                        seed_vectors.append(seed)
                        noises.append(innovation_noise(shape, config["noise"], np.random.default_rng(np.random.SeedSequence(seed))))
                    data, fresh = truth+noises[0], [truth+n for n in noises[1:]]
                    arrays = {"same_noise": noises[0], "same_data": data,
                              **{f"fresh_noise_{j}": n for j, n in enumerate(noises[1:])},
                              **{f"fresh_data_{j}": d for j, d in enumerate(fresh)}}
                    for fold, partition in partitions.items():
                        row = {"case": case["name"], "case_index": ci, "realization": realization,
                               "fold": fold, "truth_parameters": true_params, "seed_vectors": seed_vectors,
                               "methods": {}, "oracle": {"evaluation": forecast_evaluation(truth, data, truth, fresh, partition, cov, cov, scale)}}
                        for method in config["study"]["methods"]:
                            assumed = np.diag(np.diag(cov)) if method.endswith("diagonal") else cov
                            try:
                                prediction, fit = fit_motion(data, partition, assumed, scale, prior, config, method.startswith("expanded"))
                                fit["evaluation"] = forecast_evaluation(prediction, data, truth, fresh, partition, cov, assumed, scale)
                                fit["parameter_errors"] = {k: float(fit["parameters"][k]-v) for k, v in true_params.items()}
                                fit["parameter_pass"] = all(abs(fit["parameter_errors"][k]) <= tol for k, tol in prior["study"]["parameter_recovery_tolerances"].items())
                                fit["predictive_pass"] = predictive_pass(fit["evaluation"])
                                fit["status"] = "FIT_EXECUTED" if fit["optimizer_success"] else "OPTIMIZER_DID_NOT_CONVERGE"
                                arrays[f"fold{fold}_{method}"] = prediction
                                row["methods"][method] = fit
                            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                                row["methods"][method] = {"status": "FIT_EXCEPTION", "exception": repr(exc)}
                        rows.append(row)
                        save_json(out / f"case-{case['name']}-r{realization}-f{fold}.json", row)
                    packet = private / f"case-{case['name']}-r{realization}.npz"
                    np.savez_compressed(packet, **arrays)
                    save_json(out / f"packet-{case['name']}-r{realization}.json", {"path": packet.relative_to(ROOT).as_posix(), "sha256": digest(packet), "seed_vectors": seed_vectors})
                    print(f"{case['name']} realization {realization+1}/{config['study']['realizations_per_case']}: both folds, four methods; {time.perf_counter()-trial_start:.1f}s", flush=True)
        fits = [m for r in rows for m in r["methods"].values()]
        summary = {**initial, "completed_utc": utc(), "elapsed_s": time.perf_counter()-started,
                   "status": "EXECUTED_THEORY_BENCHMARK_ONLY", "synthetic_study_started": True,
                   "statistical_controls": controls["statistical_control_count"], "forward_controls": controls["prior_forward_control_count"],
                   "config_study_methods": config["study"]["methods"], "fold_realization_receipts": len(rows), "fit_count": len(fits),
                   "fit_exceptions": sum(r["status"] == "FIT_EXCEPTION" for r in fits),
                   "nonconverged_fits": sum(r["status"] == "OPTIMIZER_DID_NOT_CONVERGE" for r in fits),
                   "noise_realizations": len(config["study"]["cases"])*config["study"]["realizations_per_case"],
                   "fresh_noise_realizations": len(config["study"]["cases"])*config["study"]["realizations_per_case"]*config["study"]["fresh_replicates_per_realization"],
                   "aggregate": summarize(rows, config), "missing_closures": prior["missing_closures"],
                   "parameter_uncertainty_integrated": False, "admitted_observed_likelihoods": 0, "gravity_scores": 0}
        save_json(out / "summary.json", summary)
        write_readme(summary, out)
        plot_results(summary, out)
        save_json(out / "artifact-hashes.json", {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(out.iterdir()) if p.is_file()})
        print(f"Completed {out.relative_to(ROOT)}; {summary['fit_count']} fits, {summary['fit_exceptions']} exceptions, {summary['nonconverged_fits']} nonconverged.", flush=True)
        return 0
    except Exception as exc:
        if not (out / "failure.json").exists():
            save_json(out / "failure.json", {"utc": utc(), "status": "EXECUTION_FAILED", "exception": repr(exc),
                                            "retained_case_receipts": [p.name for p in out.glob("case-*.json")]})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(execute(args.run_id, args.controls_only))
