"""render.py -- the six clusters as images: X-ray brightness and hardness structure.

Left column of each pair is smoothed 0.5-7 keV surface brightness, log scaled.
Right is the hardness ratio H/(S+H), a temperature PROXY, on a scale set
PER CLUSTER by its own 5th-95th percentiles.

The per-cluster scaling is deliberate and is the honest choice. Hardness is not
comparable between these clusters: they sit at redshifts from 0.077 to 0.54, so
the same gas is observed through different amounts of spectral shift, and ACIS's
effective area has degraded over the 24 years spanning these observations. What
IS comparable is the structure inside one cluster -- where its gas is hotter or
cooler than its own surroundings. That is the thing spherical deprojection
removes, and the only thing these panels are being asked to show.

    python render.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))

NICE = {"ABELL_2029": "Abell 2029", "ABELL_0370": "Abell 370",
        "ABELL_2744": "Abell 2744", "MACS_J0717_5p3745": "MACS J0717",
        "ABELL_1063S": "Abell S1063", "MACS_J1149_5p2223": "MACS J1149"}
NOTE = {"ABELL_2029": "relaxed, strong cool core  z=0.077",
        "ABELL_0370": "merger  z=0.375",
        "ABELL_2744": "major merger  z=0.308",
        "MACS_J0717_5p3745": "quadruple merger  z=0.546",
        "ABELL_1063S": "near-relaxed  z=0.348",
        "MACS_J1149_5p2223": "merger  z=0.544"}
ORDER = ["ABELL_2029", "ABELL_0370", "ABELL_2744",
         "MACS_J0717_5p3745", "ABELL_1063S", "MACS_J1149_5p2223"]

HARD_CMAP = LinearSegmentedColormap.from_list(
    "hard", ["#185FA5", "#378ADD", "#85B7EB", "#F1EFE8",
             "#FAC775", "#EF9F27", "#993C1D"])


def main():
    meta = json.load(io.open(os.path.join(HERE, "maps.json"), encoding="utf-8"))["meta"]
    data = json.load(io.open(os.path.join(HERE, "maps_data.json"), encoding="utf-8"))

    fig, axes = plt.subplots(3, 4, figsize=(15.5, 11.6), facecolor="white")
    for k, cl in enumerate(ORDER):
        if cl not in data:
            continue
        m = meta[cl]
        half = m["field_arcmin"]
        ext = [half, -half, -half, half]        # RA increases left
        c = np.array(data[cl]["c"], dtype=float)
        h = np.array(data[cl]["r"], dtype=float)
        h[h == 255] = np.nan

        r, col = divmod(k, 2)
        a1, a2 = axes[r][col * 2], axes[r][col * 2 + 1]

        cc = c.copy()
        lo, hi = np.percentile(cc, 35), np.percentile(cc, 99.99)
        cc = np.arcsinh((cc - lo) / max(hi - lo, 1e-6) * 12.0)
        a1.imshow(cc, origin="lower", extent=ext, cmap="magma",
                  vmin=0, vmax=np.percentile(cc, 99.9))
        a1.set_title("%s\n%s" % (NICE[cl], NOTE[cl]), fontsize=10.5, loc="left")

        im = a2.imshow(h, origin="lower", extent=ext, cmap=HARD_CMAP,
                       vmin=0, vmax=254)
        a2.set_title("hardness asymmetry  +/-%.3f   (blue cooler, red hotter\n"
                     "than the average at the same radius)"
                     % m["residual_limit"], fontsize=9.5, loc="left")

        for ax in (a1, a2):
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.set_yticks([-10, -5, 0, 5, 10])
            ax.tick_params(labelsize=8, colors="#555")
            for s in ax.spines.values():
                s.set_color("#ccc")
            ax.plot(0, 0, "+", color="#00ff88", ms=9, mew=1.4)
        a1.set_ylabel("arcmin", fontsize=8)
        del im

    fig.suptitle(
        "Chandra ACIS: X-ray brightness (left of each pair) and hardness ASYMMETRY (right)\n"
        "3.3M photons, 27 observations. The radial hardness gradient is ACIS vignetting -- +0.124 in every cluster, "
        "relaxed and merging alike -- and has been divided out.\n"
        "What remains is how much hotter or cooler each direction is than the azimuthal average at the same radius.",
        fontsize=11, y=0.988)
    fig.tight_layout(rect=[0, 0.01, 1, 0.955])
    out = os.path.join(HERE, "cluster_xray_maps.png")
    fig.savefig(out, dpi=125, facecolor="white")
    print("wrote", out, "%.1f MB" % (os.path.getsize(out) / 1048576))
    return 0


if __name__ == "__main__":
    sys.exit(main())
