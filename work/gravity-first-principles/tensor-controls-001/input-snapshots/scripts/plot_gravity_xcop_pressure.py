"""Plot the fixed cluster pressure and historical Cassini scenario comparison."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.output_stem.with_suffix(suffix) for suffix in [".png", ".svg", ".json"]]
    if any(path.exists() for path in paths):
        raise FileExistsError("figure outputs already exist")
    result_path = ROOT/"work/gravity-first-principles/xcop-pressure-002/result.json"
    data = json.loads(result_path.read_bytes())
    receipt = json.loads(result_path.with_name("receipt.json").read_bytes())
    if sha256(result_path.read_bytes()).hexdigest() != receipt["result_sha256"]:
        raise ValueError("pressure receipt hash mismatch")
    previous_path = ROOT/data["config"]["predecessor"]
    if sha256(previous_path.read_bytes()).hexdigest() != data["input_hashes"][data["config"]["predecessor"]]:
        raise ValueError("quadrupole input hash mismatch")
    previous = json.loads(previous_path.read_bytes())
    summary = data["summary"]["nominal_models"]
    baseline = summary["empirical_RAR_a0_1.2e-10"]["equal_cluster_mse_log10_ratio"]**.5
    cassini = previous["config"]["cassini_summary"]
    center = cassini["mean_Q2_s_minus2"]/1e-27
    half_width = cassini["one_sigma_s_minus2"]*cassini["screen_sigma_multiplier"]/1e-27
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    colors, markers = ["#b75b12", "#157c87", "#6845a1"], ["o", "s", "^"]
    plot_rows = []
    for shape, color, marker in zip(data["config"]["shapes"], colors, markers):
        models = sorted([model for model in data["models"] if model.get("shape") == shape], key=lambda m: m["a0"])
        x = [model["a0"]/1e-10 for model in models]
        rmse = [summary[model["id"]]["equal_cluster_mse_log10_ratio"]**.5 for model in models]
        axes[0].plot(x, rmse, color=color, marker=marker, linestyle=":", label=f"m = {shape:g}", markersize=7)
        low, high = [], []
        for model in models:
            solar = next(row for row in data["prior_cassini_summary_screen"] if row["model"] == model["id"])
            q = [row["Q2_s_minus2"]/1e-27 for row in solar["scenarios"]]
            low.append(min(q))
            high.append(max(q))
            plot_rows.append({"model": model["id"], "a0_in_1e10_units": model["a0"]/1e-10,
                              "pressure_log10_RMSE": summary[model["id"]]["equal_cluster_mse_log10_ratio"]**.5,
                              "Q2_scenario_range_in_1e27_units": [min(q), max(q)]})
        axes[1].vlines(x, low, high, color=color, linewidth=2)
        axes[1].scatter(x, low, color=color, marker=marker, s=38)
        axes[1].scatter(x, high, facecolors="white", edgecolors=color, marker=marker, s=38)
        axes[1].plot(x, (np.asarray(low)+high)/2, color=color, linestyle=":", alpha=.6)
    axes[0].axhline(baseline, color="#555555", linestyle="--", linewidth=1.2, label="RAR comparator")
    axes[0].set(title="Cluster pressure: nominal source assumptions", ylabel="Equal-cluster RMS log10 pressure residual (dex)", ylim=(0, .29))
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    axes[1].axhspan(center-half_width, center+half_width, color="#387b4e", alpha=.13)
    axes[1].axhline(center+half_width, color="#387b4e", linewidth=1)
    axes[1].text(1.20, 3, "Historical Cassini screen", ha="center", color="#285f3b", fontsize=10)
    axes[1].set(title="Solar System: same global parameters", ylabel=r"Predicted $Q_2$ ($10^{-27}$ s$^{-2}$)", ylim=(-5, 68))
    for ax in axes:
        ax.set_xlabel(r"Universal $a_0$ ($10^{-10}$ m s$^{-2}$)")
        ax.set_xticks([.5, 1.2, 2])
        ax.set_xlim(.35, 2.15)
        ax.grid(axis="y", color="#d9dce0", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    fig.suptitle("Stronger cluster response conflicts with the tested local screen", fontsize=15, y=.97)
    fig.text(.5, .07, "Nine fixed scalar candidates; eight previously exposed clusters, 30 pressure targets. No galaxy or lensing validation.",
             ha="center", fontsize=9)
    fig.text(.5, .032, "Vertical segments span two assumed Galactic fields, not confidence intervals. Dotted lines only connect sampled settings.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .92), w_pad=2.5)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=180, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    with paths[2].open("x", encoding="utf8", newline="\n") as handle:
        json.dump({"pressure_result_sha256": receipt["result_sha256"], "data": plot_rows,
                   "RAR_pressure_log10_RMSE": baseline, "Cassini_screen_in_1e27_units": [center-half_width, center+half_width],
                   "scientific_status": data["status"]}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
