"""fingerprint.py -- per-universe medians of a handful of named observables.

Reads the cached pools written by run_stage5.py; generates nothing new.  The
purpose is to let a reader check that each mock universe looks like a universe
before believing any separation result derived from it.
"""
from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOLDIR = os.path.join(HERE, "results", "pools")

SHOW = {
    "rar_b1": "log10(g_obs/g_bar) at g_bar ~ 1e-11.5 m/s^2 (galaxies)",
    "rar_b3": "log10(g_obs/g_bar) at g_bar ~ 1e-10.5 m/s^2 (galaxies)",
    "outer_slope": "median outer log-slope of the rotation curve",
    "btfr_slope": "d log v_flat / d log M_bar",
    "vert_minus_rad": "log10[(g_z/g_z,bar) / (g_R/g_R,bar)] -- vertical vs radial",
    "wl_b1": "log10(dSigma_obs/dSigma_bar), 0.45-0.9 R500 (clusters, raw shear)",
    "he_b1": "log10(M_hydrostatic/M_bar), 0.4-1.5 R500",
    "dyn_b1": "log10(M_dyn/M_bar), 0.5-2 R500 (member redshifts)",
    "q_amp": "mean cluster shear quadrupole amplitude",
    "sl_frac": "fraction of clusters producing multiple images",
    "sn_dur": "d log(light-curve duration) / d log(1+z_obs)",
    "ep_ld": "log10(M_lens/M_dyn) at matched radii (uncalibrated offset; only changes carry information)",
}


def main():
    out = {"_note": ("medians over each arm's pool of corpora; these are the blind "
                     "pipeline's own outputs, not the generative truth"),
           "_observables": SHOW, "arms": {}}
    for fn in sorted(glob.glob(os.path.join(POOLDIR, "*.pkl"))):
        tag = os.path.basename(fn).rsplit("_", 2)[0]
        if tag.startswith(("scan_", "orc_", "fine_", "eq")):
            continue
        with open(fn, "rb") as f:
            recs = pickle.load(f)
        out["arms"][tag] = {k: float(np.median([r["features"].get(k, np.nan) for r in recs]))
                            for k in SHOW}
        out["arms"][tag]["n_corpora"] = len(recs)
    with open(os.path.join(HERE, "results", "E8_fingerprints.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("arms:", ", ".join(out["arms"]))
    for k in SHOW:
        print(f"{k:18s}", " ".join(f"{out['arms'][a][k]:8.3f}" for a in out["arms"]))


if __name__ == "__main__":
    main()
