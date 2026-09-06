"""Execute one frozen CPU-only synthetic motion benchmark, retaining every run.

Run from the repository root:
  python scripts/run_mond_atlas_motion_controls.py
Only this milestone's report/private directories are written. No network or Git.
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

# Restrict CPU numerical-library threads in this process, not the user's environment.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from mond_atlas_motion_controls import (
    Geometry, Instrument, PARAMETERS, direct_reference_cube, evaluate_prediction,
    fit_model, fixed_splits, forward_cube, known_noise_sigma, numerical_controls,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mond_atlas_motion_controls_v1.json"
REPORT = ROOT / "work/gravity-first-principles/mond-atlas-motion-controls-001"
PRIVATE = ROOT / "work/private/mond-atlas-motion-controls-001"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def verify_freeze(config_path=CONFIG, report=REPORT):
    freeze = json.loads((report / "freeze.json").read_text(encoding="utf-8-sig"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks = {
        "config_sha256": digest(config_path),
        "preflight_sha256": digest(report / "PREFLIGHT.md"),
        "shared_cube_sha256": digest(ROOT / "scripts/mond_atlas_cube.py"),
    }
    if (any(freeze[k] != v for k, v in checks.items())
            or freeze["disposition"] != "THEORY_BENCHMARK_ONLY"
            or config["disposition"] != "THEORY_BENCHMARK_ONLY"
            or config["observational_scoring_allowed"] is not False
            or config["source_dataset"] is not None
            or freeze["synthetic_response_generated"] is not False
            or freeze["implementation_started"] is not False):
        raise ValueError("frozen theory-only preflight, config or shared primitive changed")
    return config, freeze, checks


def compact_readme(summary):
    rows = [
        "# Executed motion-forward benchmark", "",
        "**THEORY_BENCHMARK_ONLY.** Prescribed thin emitting rings; no observed galaxy",
        "motion, source/covariance admission, gravity score or mass inference.", "",
        f"Run `{summary['run_id']}` completed {summary['completed_utc']} on CPU.",
        f"All {summary['numerical_control_count']} frozen numerical controls passed before synthetic response generation.",
        "",
        "The table reports joint held-out channels AND pixels. q/N uses the supplied",
        "independent Gaussian noise covariance. Truth error/N compares to the noiseless",
        "independent quadrature truth. Recovery criteria were frozen before the run and",
        "require all three held-out subsets to pass. No result is discarded.", "",
        "| Injection | Circular q/N | Expanded q/N | Circular truth error/N | Expanded truth error/N | Predictive recovery | Parameter recovery |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["cases"]:
        c, e = row["fits"]["circular_only"], row["fits"]["expanded"]
        cm, em = c["evaluation"]["heldout_joint"], e["evaluation"]["heldout_joint"]
        rows.append(f"| {row['name']} | {cm['weighted_data_error_per_voxel']:.4f} | {em['weighted_data_error_per_voxel']:.4f} | {cm['weighted_truth_error_per_voxel']:.4f} | {em['weighted_truth_error_per_voxel']:.4f} | {e['predictive_recovery']} | {e['parameter_recovery']} |")
    rows.extend([
        "", "Full channel-only, pixel-only, joint, and training values, both optimizer starts,",
        "parameter errors, bound contacts and sensitivity degeneracies are in summary.json.",
        "One random noise draw per case is an illustration, not a coverage study.", "",
        "## Physical and statistical limits", "",
        "The coordinate convention, equations, primary references, flux accounting,",
        "source restrictions and all thresholds are in ../PREFLIGHT.md and the frozen config.",
        "Spectral channels use the existing Gaussian integration primitive; the beam uses",
        "the existing zero-padded convolution. A beam-support halo preserves outside-field",
        "in-scatter. The declared spatial response is a linear tent, not a top-hat detector.",
        "Independent truth uses different radial/azimuthal quadrature, rotation matrices,",
        "SciPy Gaussian CDF and direct convolution; it shares the declared physical model.",
        "", "The expanded model is conditional on known center, emission radial profile,",
        "total intrinsic flux, asymmetry phase, channel response and beam. The diagonal",
        "covariance is known by construction and does not validate observed gas noise.",
        "The noiseless face-on control is exactly insensitive to planar speed and radial",
        "flow. An unresolved axisymmetric line profile also has an exact rotation/radial",
        "amplitude degeneracy for the shared radial profile. Local Jacobian sensitivities",
        "and parameter errors show remaining confusion even when pixel predictions pass.",
        "", "Gaussian line width does not implement pressure support. No force balance,",
        "continuity/time evolution, finite thickness, optical depth, self absorption,",
        "vertical motions, mass-to-light conversion, gravity or lensing closure is solved.",
        "", "## Reproduce", "",
        "From the Invariant repository with the existing NumPy/SciPy/threadpoolctl environment:",
        "", "```powershell",
        "& 'C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe' scripts/run_mond_atlas_motion_controls.py",
        "& 'C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe' -B -m unittest discover -s tests -p test_mond_atlas_motion_controls.py -v",
        "```", "",
        "Every execution creates a fresh run directory and retains synthetic arrays in",
        "work/private/mond-atlas-motion-controls-001/. Existing receipts are never overwritten.",
    ])
    return "\n".join(rows)+"\n"


def plot_study(summary, out, private):
    """Standalone scientific diagnostic figures; no model selection depends on plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Zero amplitude", "Warp", "Radial flow", "Emission asymmetry", "Combined", "Face-on radial"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, layout="constrained")
    xx = np.arange(len(labels))
    for offset, name, color, label in [(-0.18, "circular_only", "#3973a5", "Circular-only"),
                                       (0.18, "expanded", "#c26a33", "Expanded motion")]:
        metric = [r["fits"][name]["evaluation"]["heldout_joint"] for r in summary["cases"]]
        axes[0].bar(xx+offset, [r["weighted_data_error_per_voxel"] for r in metric], 0.34, color=color, label=label)
        axes[1].bar(xx+offset, [r["weighted_truth_error_per_voxel"] for r in metric], 0.34, color=color)
    axes[0].axhline(1, color="#444444", lw=1, ls="--", label="Supplied noise expectation")
    axes[0].set_ylabel("Data residual q / N")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_ylim(0, 1.9)
    axes[1].set_ylabel("Truth-mean weighted error / N")
    axes[1].set_yscale("log")
    axes[1].set_xticks(xx, labels, fontsize=9)
    axes[1].set_ylim(0.00005, 1)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Synthetic motion benchmark: held-out channels and pixels\nKnown diagonal covariance; one noise realization per case", fontsize=13)
    fig.savefig(out / "heldout-diagnostics.png", dpi=170)
    fig.savefig(out / "heldout-diagnostics.svg")
    plt.close(fig)

    with np.load(private / "combined.npz") as a:
        edges = a["channel_edges_km_s"]
        centers = (edges[:-1]+edges[1:])/2
        channels = [int(np.argmin(abs(centers-v))) for v in (-50, 0, 50)]
        fig, axes = plt.subplots(3, 3, figsize=(9, 8), layout="constrained")
        for col, channel in enumerate(channels):
            vmax = float(a["truth"][channel].max())
            im = axes[0, col].imshow(a["truth"][channel], origin="lower", cmap="magma", vmin=0, vmax=vmax)
            fig.colorbar(im, ax=axes[0, col], shrink=0.8, label="Integrated flux / pixel")
            axes[0, col].set_title(f"{centers[channel]:.0f} km/s channel")
            for row, name in enumerate(("circular_only", "expanded"), start=1):
                res = (a[f"prediction_{name}"][channel]-a["truth"][channel])/a["sigma"][channel]
                im = axes[row, col].imshow(res, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
                fig.colorbar(im, ax=axes[row, col], shrink=0.8, label="(Prediction - truth) / sigma")
            for ax in axes[:, col]:
                ax.set_xticks([])
                ax.set_yticks([])
        for row, label in enumerate(("Independent truth", "Circular-only", "Expanded")):
            axes[row, 0].set_ylabel(label, fontsize=11)
        fig.suptitle("Combined synthetic injection: channel images\nAll pixels displayed; held-out metrics use the frozen disjoint masks", fontsize=13)
        fig.savefig(out / "combined-channel-images.png", dpi=170)
        plt.close(fig)


def execute(run_id=None):
    started = time.perf_counter()
    config, freeze, hashes = verify_freeze()
    run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run ID must be a simple unique directory name")
    out, private = REPORT / run_id, PRIVATE / run_id
    out.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    code_paths = [ROOT / "scripts/mond_atlas_motion_controls.py", Path(__file__).resolve(),
                  ROOT / "tests/test_mond_atlas_motion_controls.py"]
    initial = {
        "started_utc": utc(), "run_id": run_id, "disposition": "THEORY_BENCHMARK_ONLY",
        "frozen_preflight": freeze, "verified_input_hashes": hashes,
        "implementation_hashes": {str(p.relative_to(ROOT)): digest(p) for p in code_paths},
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(), "device": "CPU",
                    "numerical_library_threads": 1},
        "access_accounting": {"observed_source_files_opened": 0, "observed_response_files_opened": 0,
                              "synthetic_response_generated": False, "network_requests": 0},
    }
    save_json(out / "execution-start.json", initial)
    try:
        with threadpool_limits(limits=1):
            controls = numerical_controls(config)
            controls["completed_utc"] = utc()
            save_json(out / "numerical-controls.json", controls)
            if not controls["all_passed"]:
                save_json(out / "failure.json", {"disposition": "BENCHMARK_FAILED", "synthetic_response_generated": False})
                return 2
            save_json(out / "response-access-gate.json", {
                "utc": utc(), "all_required_numerical_gates_passed": True,
                "controls_sha256": digest(out / "numerical-controls.json"),
                "config_sha256": hashes["config_sha256"], "observational_scoring_allowed": False,
                "synthetic_study_may_begin": True,
            })
            print(f"{len(controls['controls'])} numerical controls passed; generating frozen synthetic cases.", flush=True)
            g, ins = Geometry(**config["geometry"]), Instrument(**config["instrument"])
            case_results = []
            seeds = np.random.SeedSequence(config["study"]["seed"]).spawn(len(config["study"]["cases"]))
            for case, seed in zip(config["study"]["cases"], seeds):
                case_start = time.perf_counter()
                truth_parameters = dict(config["base_parameters"], **case["overrides"])
                truth = direct_reference_cube(truth_parameters, g, ins, *config["quadrature"]["truth"])
                sigma = known_noise_sigma(truth.shape, config["study"]["noise_sigma_flux"])
                splits = fixed_splits(truth.shape)
                noise = np.random.default_rng(seed).normal(size=truth.shape)*sigma
                data = truth+noise
                arrays = {"truth": truth, "noise": noise, "data": data, "sigma": sigma,
                          "channel_edges_km_s": ins.edges, **{f"mask_{k}": m for k, m in splits.items()}}
                row = {"name": case["name"], "truth_parameters": truth_parameters,
                       "noise_seed_spawn_key": list(seed.spawn_key), "fits": {},
                       "conditional_source_parameters": config["geometry"],
                       "split_counts": {k: int(m.sum()) for k, m in splits.items()}}
                _, row["production_truth_flux_accounting"] = forward_cube(truth_parameters, g, ins, accounting=True)
                for expanded in (False, True):
                    _, prediction, fit = fit_model(data, sigma, splits["train"], g, ins, config,
                                                   expanded, fixed=case.get("fixed_fit_parameters", {}))
                    fit["evaluation"] = evaluate_prediction(prediction, data, truth, sigma, splits)
                    fit["parameter_errors"] = {k: float(fit["parameters"][k]-truth_parameters[k]) for k in PARAMETERS}
                    fit["parameter_recovery"] = all(abs(fit["parameter_errors"][k]) <= tol
                                                    for k, tol in config["study"]["parameter_recovery_tolerances"].items())
                    fit["predictive_recovery"] = all(
                        metric["weighted_data_error_per_voxel"] <= 1.25
                        and metric["weighted_truth_error_per_voxel"] <= 0.25
                        for name, metric in fit["evaluation"].items() if name != "train")
                    row["fits"][fit["model"]] = fit
                    arrays[f"prediction_{fit['model']}"] = prediction
                row["elapsed_s"] = time.perf_counter()-case_start
                npz_path = private / f"{case['name']}.npz"
                np.savez_compressed(npz_path, **arrays)
                row["synthetic_arrays"] = {"path": str(npz_path.relative_to(ROOT)), "sha256": digest(npz_path)}
                save_json(out / f"case-{case['name']}.json", row)
                case_results.append(row)
                c, e = (row["fits"][k]["evaluation"]["heldout_joint"]["weighted_data_error_per_voxel"]
                        for k in ("circular_only", "expanded"))
                print(f"{case['name']}: held-out q/N circular={c:.4f}, expanded={e:.4f}; {row['elapsed_s']:.1f}s", flush=True)
        summary = {**initial, "completed_utc": utc(), "elapsed_s": time.perf_counter()-started,
                   "status": "EXECUTED_THEORY_BENCHMARK_ONLY", "numerical_control_count": len(controls["controls"]),
                   "numerical_controls_passed": True, "cases": case_results,
                   "admitted_observed_cube_likelihoods": 0, "gravity_scores_produced": 0,
                   "covariance": config["study"]["covariance"], "missing_closures": config["missing_closures"],
                   "access_accounting": {"observed_source_files_opened": 0, "observed_response_files_opened": 0,
                                         "synthetic_response_generated": True, "network_requests": 0}}
        save_json(out / "summary.json", summary)
        (out / "README.md").write_text(compact_readme(summary), encoding="utf-8")
        plot_study(summary, out, private)
        save_json(out / "artifact-hashes.json", {
            str(p.relative_to(ROOT)): digest(p) for p in sorted(out.iterdir()) if p.is_file()
        })
        print(f"Completed {out.relative_to(ROOT)}", flush=True)
        return 0
    except Exception as exc:
        failure = {"utc": utc(), "status": "EXECUTION_FAILED", "exception": repr(exc),
                   "completed_case_files": [p.name for p in out.glob("case-*.json")]}
        if not (out / "failure.json").exists():
            save_json(out / "failure.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Optional unique run directory name; existing runs cannot be replaced.")
    args = parser.parse_args()
    raise SystemExit(execute(args.run_id))
