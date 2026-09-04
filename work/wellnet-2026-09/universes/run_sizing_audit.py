"""run_sizing_audit.py -- verify the false-positive rate on an UNTOUCHED null set.

E0 reports the realised rate at its own 95th percentile, which is 0.05 by
construction and therefore proves nothing.  The charter asks for three disjoint
simulation sets: calibration to SET the critical value, untouched audit to
VERIFY the rate, injection to measure power.

This pass supplies the second one, two ways:

  by replicate  critical value from the odd A-vs-A replicates, rate measured on
                the even ones (same universes, independent draws)
  by arm        critical value from one half of the ARMS, rate measured on the
                other half -- the harder test, since it asks whether a critical
                value calibrated on one set of worlds transfers to worlds it
                was never fitted on

Reads only the cached pools; generates no new corpora.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from universes import generate as gn          # noqa: E402
from universes import run_stage5 as R5        # noqa: E402
from universes import stats as st             # noqa: E402

RES = os.path.join(HERE, "results")
POOLDIR = os.path.join(RES, "pools")
NREP = 40


def main():
    with open(os.path.join(RES, "channel_map.json")) as f:
        CM = json.load(f)
    R5.KEYS = CM["feature_order"]
    R5.CHAN_IDX = {c: [R5.KEYS.index(k) for k in v] for c, v in CM["channels"].items()}

    arms = {}
    for fn in sorted(os.listdir(POOLDIR)):
        tag = fn.rsplit("_", 2)[0]
        if tag.startswith(("scan_", "orc_", "fine_", "eq")):
            continue
        with open(os.path.join(POOLDIR, fn), "rb") as f:
            arms[tag] = pickle.load(f)
    print("arms:", ", ".join(arms))

    recs = []
    for name, pool in arms.items():
        X, _ = gn.to_matrix(pool, R5.KEYS)
        q = len(X) // 4
        for rep in range(NREP):
            idx = np.random.default_rng(21000 + rep).permutation(len(X))
            r = R5.sepmax(X[idx[:q]], X[idx[q:2 * q]],
                          X[idx[2 * q:3 * q]], X[idx[3 * q:4 * q]], seed=rep)
            recs.append({"arm": name, "rep": rep, "z": r["z_max"],
                         "best": r["best_test"],
                         "per_test": {k: v["z"] for k, v in r["per_test"].items()}})
    z = np.array([r["z"] for r in recs])
    armv = np.array([r["arm"] for r in recs])
    repv = np.array([r["rep"] for r in recs])

    out = {"n_null_tests": len(recs), "n_reps_per_arm": NREP,
           "n_arms": len(arms), "arms": list(arms)}

    # --- split by replicate ------------------------------------------------
    cal, aud = z[repv % 2 == 0], z[repv % 2 == 1]
    for alpha in (0.05, 0.01):
        c = float(np.quantile(cal, 1 - alpha))
        out[f"by_replicate_alpha{alpha}"] = {
            "critical_value_from_calibration_half": c,
            "n_cal": int(len(cal)), "n_audit": int(len(aud)),
            "realised_rate_on_untouched_audit_half":
                st.rate_with_ci(int((aud >= c).sum()), len(aud))}

    # --- split by arm ------------------------------------------------------
    names = list(arms)
    calarms = set(names[::2])
    cal2 = z[np.isin(armv, list(calarms))]
    aud2 = z[~np.isin(armv, list(calarms))]
    for alpha in (0.05, 0.01):
        c = float(np.quantile(cal2, 1 - alpha))
        out[f"by_arm_alpha{alpha}"] = {
            "calibration_arms": sorted(calarms),
            "audit_arms": sorted(set(names) - calarms),
            "critical_value_from_calibration_arms": c,
            "n_cal": int(len(cal2)), "n_audit": int(len(aud2)),
            "realised_rate_on_untouched_audit_arms":
                st.rate_with_ci(int((aud2 >= c).sum()), len(aud2))}

    # --- per-arm null z, to expose any arm whose null is heavier ------------
    out["per_arm_null"] = {}
    for name in names:
        v = z[armv == name]
        out["per_arm_null"][name] = {
            "median": float(np.median(v)), "z95": float(np.quantile(v, 0.95)),
            "max": float(v.max()),
            "tail_ratio_p95_over_median": float(np.quantile(v, 0.95)
                                                / max(np.median(v), 1e-9))}
    out["heaviest_tailed_arm"] = max(out["per_arm_null"],
                                     key=lambda k: out["per_arm_null"][k]["z95"])
    out["note"] = ("the by-arm split is the demanding one: a critical value that "
                   "transfers across worlds it was never calibrated on is the only "
                   "kind that can be applied to a real dataset")
    with open(os.path.join(RES, "E10_sizing_audit.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    for k in ("by_replicate_alpha0.05", "by_arm_alpha0.05",
              "by_replicate_alpha0.01", "by_arm_alpha0.01"):
        d = out[k]
        rk = [x for x in d if x.startswith("realised")][0]
        print(f"{k}: crit "
              f"{[v for kk, v in d.items() if kk.startswith('critical')][0]:.2f} -> "
              f"{d[rk]['rate']:.3f} [{d[rk]['lo']:.3f},{d[rk]['hi']:.3f}] "
              f"(n={d[rk]['n']})")
    print("heaviest-tailed arm:", out["heaviest_tailed_arm"],
          out["per_arm_null"][out["heaviest_tailed_arm"]])


if __name__ == "__main__":
    main()
