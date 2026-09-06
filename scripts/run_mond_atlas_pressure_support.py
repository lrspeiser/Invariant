"""Run frozen theory-only pressure controls, then a bounded synthetic study.

Creates new numbered runs without overwriting earlier runs. Raw arrays remain
under work/private. No input path for observational data exists.
"""
import os

# Set before importing numerical libraries; also enforce loaded-library limits.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from mond_atlas_pressure_support import (case_balance, case_column, fit_amplitude,
                                         independent_truth, numerical_controls)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "mond-atlas-pressure-support-001"
PUBLIC = ROOT / "work/gravity-first-principles" / PACKAGE
PRIVATE = ROOT / "work/private" / PACKAGE
CONFIG = ROOT / "configs/mond_atlas_pressure_support_v1.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def check_freeze():
    freeze = json.loads((PUBLIC / "freeze.json").read_text(encoding="utf-8"))
    assert sha(CONFIG) == freeze["config_sha256"], "Frozen config changed"
    assert sha(PUBLIC / "PREFLIGHT.md") == freeze["preflight_sha256"], "Preflight changed"
    for path, expected in freeze["prior_immutable_files"].items():
        assert sha(ROOT / path) == expected, f"Prior immutable file changed: {path}"
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["admission"] == "THEORY_BENCHMARK_ONLY"
    return config, freeze


def metrics(prediction, truth, observed, fresh, study):
    train = np.array(study["train_indices"])
    test = np.array(study["heldout_indices"])
    sigma = study["noise_sigma_km_s"]
    return {
        "train_q_per_sample": float(np.mean(((prediction[train]-observed[train])/sigma)**2)),
        "heldout_q_per_sample": float(np.mean(((prediction[test]-observed[test])/sigma)**2)),
        "fresh_heldout_q_per_sample": float(np.mean(((prediction[test]-fresh[test])/sigma)**2)),
        "noiseless_signal_rmse_km_s": float(np.sqrt(np.mean((prediction-truth)**2))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args(argv)
    if not (args.run_id.startswith("run-") and args.run_id[4:].isdigit()):
        parser.error("run-id must be run- followed by digits")
    config, freeze = check_freeze()
    pub, private = PUBLIC / args.run_id, PRIVATE / args.run_id
    if pub.exists() or private.exists():
        raise FileExistsError("Run exists; choose a new run ID to retain all results")
    pub.mkdir(parents=True)
    private.mkdir(parents=True)
    receipt = {"admission": config["admission"], "started_utc": utc(),
               "run_id": args.run_id, "config_sha256": sha(CONFIG),
               "freeze_sha256": sha(PUBLIC / "freeze.json"),
               "implementation_sha256": {relative(ROOT / p): sha(ROOT / p) for p in
                   ["scripts/mond_atlas_pressure_support.py", "scripts/run_mond_atlas_pressure_support.py",
                    "tests/test_mond_atlas_pressure_support.py"]},
               "prior_immutable_files_checked": len(freeze["prior_immutable_files"]),
               "observational_source_or_velocity_files_opened": 0,
               "runtime": {"python": sys.version, "platform": platform.platform(),
                           "numpy": np.__version__, "scipy": scipy.__version__,
                           "matplotlib": matplotlib.__version__, "gpu": False},
               "status": "STARTED"}
    write_json(pub / "start.json", receipt)
    try:
        with threadpool_limits(limits=1):
            receipt["runtime"]["threadpools"] = threadpool_info()
            if any(pool["num_threads"] != 1 for pool in threadpool_info()):
                raise RuntimeError("CPU single-thread admission failed")
            controls = numerical_controls(config)
            write_json(pub / "controls.json", {"completed_utc": utc(), "controls": controls})
            receipt["controls_passed"] = sum(x["passed"] for x in controls)
            receipt["controls_total"] = len(controls)
            if not all(x["passed"] for x in controls):
                receipt["status"] = "BENCHMARK_FAILED"
                raise RuntimeError("Independent controls failed; study response generation forbidden")
            receipt["controls_admitted_utc"] = utc()
            if args.controls_only:
                receipt["status"] = "CONTROLS_ONLY_PASSED"
            else:
                run_study(config, pub, private, receipt)
                receipt["status"] = "THEORY_BENCHMARK_ONLY_COMPLETE" if receipt["failed_fits"] == 0 else "FIT_FAILURES_RETAINED"
        check_freeze()
        receipt["prior_immutable_files_rechecked_after_run"] = len(freeze["prior_immutable_files"])
    except Exception as exc:
        receipt["error"] = repr(exc)
        if receipt["status"] == "STARTED":
            receipt["status"] = "EXECUTION_FAILED"
        write_json(pub / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), "utc": utc()})
    finally:
        receipt["completed_utc"] = utc()
        receipt["private_files"] = {relative(p): sha(p) for p in sorted(private.rglob("*")) if p.is_file()}
        write_json(pub / "receipt.json", receipt)
        public_hashes = {relative(p): sha(p) for p in sorted(pub.rglob("*")) if p.is_file()}
        write_json(pub / "manifest.json", {"public": public_hashes, "private": receipt["private_files"]})
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] in ("CONTROLS_ONLY_PASSED", "THEORY_BENCHMARK_ONLY_COMPLETE") else 1


def run_study(config, pub, private, receipt):
    study = config["study"]
    r = np.linspace(study["radius_min_kpc"], study["radius_max_kpc"], study["radius_count"])
    projection = np.sin(np.deg2rad(study["inclination_deg"]))
    train = np.array(study["train_indices"])
    heldout = np.array(study["heldout_indices"])
    if set(train) & set(heldout) or set(train) | set(heldout) != set(range(len(r))):
        raise ValueError("Frozen radial split must partition the samples")
    np.savez_compressed(private / "design.npz", radius=r, train=train, heldout=heldout)
    results, seed_receipt, cases_summary, failures = [], [], [], []
    receipt["first_study_response_generated_utc"] = utc()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), constrained_layout=True)
    for index, case in enumerate(study["cases"]):
        v2, vc2, support = independent_truth(r, case)
        if np.any(v2 < 0):
            raise ValueError(f"Predeclared admissible case is impossible: {case['id']}")
        truth = study["systemic_km_s"] + projection*np.sqrt(v2)
        column = case_column(r, case)
        np.savez_compressed(private / (case["id"]+"-truth.npz"),
                            radius=r, rotation_squared=v2, circular_squared=vc2,
                            support=support, los_truth=truth, sigma=column.sigma,
                            integrated_pressure=column.integrated_pressure,
                            pressure_gradient=column.pressure_gradient)
        noiseless_fits = {}
        for model in study["fit_models"]:
            fit, pred = fit_amplitude(r, truth, train, case, model, study)
            fit["relative_force_bias"] = fit["amplitude"]/case["amplitude"]-1
            fit["noiseless_signal_rmse_km_s"] = float(np.sqrt(np.mean((pred-truth)**2)))
            noiseless_fits[model] = fit
            np.savez_compressed(private / f"{case['id']}-{model}-noiseless-fit.npz", prediction=pred)
            if not fit["success"]:
                failures.append({"case": case["id"], "model": model, "kind": "noiseless", "fit": fit})
        ax = axes[0, index]
        ax.plot(r, projection*np.sqrt(vc2), color="#8395a7", ls="--", label="Circular speed, projected")
        ax.plot(r, truth, color="#16324f", label="Pressure-balanced truth")
        colors = {"pressure_blind": "#c76d28", "known_pressure": "#008675"}
        for seed_base in study["seeds"]:
            seed = seed_base + index*study["case_seed_stride"]
            fresh_seed = seed+study["fresh_seed_offset"]
            noise = np.random.default_rng(seed).normal(0, study["noise_sigma_km_s"], len(r))
            fresh_noise = np.random.default_rng(fresh_seed).normal(0, study["noise_sigma_km_s"], len(r))
            obs, fresh = truth+noise, truth+fresh_noise
            packet = {"noise": noise, "fresh_noise": fresh_noise, "observed": obs, "fresh": fresh}
            seed_receipt.append({"case": case["id"], "seed": seed, "fresh_seed": fresh_seed})
            for model in study["fit_models"]:
                row = {"case": case["id"], "model": model, "seed": seed, "fresh_seed": fresh_seed}
                try:
                    fit, pred = fit_amplitude(r, obs, train, case, model, study)
                    row.update(fit)
                    row["relative_force_bias"] = fit["amplitude"]/case["amplitude"]-1
                    row.update(metrics(pred, truth, obs, fresh, study))
                    packet[model+"_prediction"] = pred
                    packet[model+"_amplitude"] = np.array(fit["amplitude"])
                    if not fit["success"]:
                        failures.append(row.copy())
                    ax.plot(r, pred, color=colors[model], alpha=0.38, lw=1)
                except Exception as exc:
                    row.update({"success": False, "error": repr(exc)})
                    failures.append(row.copy())
                results.append(row)
            np.savez_compressed(private / f"{case['id']}-seed-{seed}.npz", **packet)
        case_rows = [x for x in results if x["case"] == case["id"]]
        summary = {"case": case["id"], "true_amplitude": case["amplitude"], "noiseless_fits": noiseless_fits,
                   "failed_fit_count": sum(not row["success"] for row in case_rows), "models": {}}
        for j, model in enumerate(study["fit_models"]):
            selected = [x for x in case_rows if x["model"] == model and x["success"]]
            summary["models"][model] = {"successful_count": len(selected)}
            for key in ["relative_force_bias", "noiseless_signal_rmse_km_s", "train_q_per_sample",
                        "heldout_q_per_sample", "fresh_heldout_q_per_sample"]:
                values = [x[key] for x in selected]
                summary["models"][model][key] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)} if values else None
            if selected:
                axes[1, index].scatter(np.full(len(selected), j), [100*x["relative_force_bias"] for x in selected],
                                       color=colors[model], s=35, label=model.replace("_", " "))
        cases_summary.append(summary)
        ax.set_title(case["id"].replace("_", " "))
        ax.set_xlabel("Radius [kpc]")
        ax.set_ylabel("LOS mean speed [km/s]")
        ax.grid(alpha=.18)
        if index == 0:
            ax.legend(fontsize=8)
        axes[1, index].axhline(0, color="#8395a7", ls="--")
        axes[1, index].set_xticks([0, 1], ["Pressure blind", "Known pressure"])
        axes[1, index].set_ylabel("Supplied force amplitude error [%]")
        axes[1, index].grid(axis="y", alpha=.18)
    fig.suptitle("Pressure support: fitting speeds can still give the wrong force\nSynthetic radial mechanics • fixed source and inclination • four noise draws per case", fontsize=13)
    fig.savefig(pub / "pressure-and-force-recovery.png", dpi=160)
    plt.close(fig)
    impossible = case_balance(r, study["impossible_case"])
    np.savez_compressed(private / "impossible-signed-equilibrium.npz", radius=r,
                        rotation_squared=impossible.rotation_squared, feasible=impossible.feasible,
                        support=impossible.support)
    expected_rejections = [{"case": study["impossible_case"]["id"], "status": impossible.status,
                            "invalid_radii": int(np.sum(~impossible.feasible)), "total_radii": len(r),
                            "minimum_rotation_squared": float(impossible.rotation_squared.min()),
                            "action": "Retained signed values; no speed or fit generated"}]
    write_json(pub / "fits.json", {"fits": results, "failures": failures})
    write_json(pub / "summary.json", {"admission": config["admission"], "cases": cases_summary,
                                     "expected_rejections": expected_rejections, "seeds": seed_receipt,
                                     "limitations": config["unresolved"]})
    receipt.update({"noisy_fits": len(results), "noiseless_fits": 2*len(study["cases"]),
                    "failed_fits": len(failures), "noise_arrays": 2*len(seed_receipt),
                    "expected_rejected_cases": len(expected_rejections),
                    "noise_covariance": "known diagonal in radial mean-speed space",
                    "evaluation": "Heldout radii and independently redrawn heldout noise; no claim of population significance"})


if __name__ == "__main__":
    raise SystemExit(main())
