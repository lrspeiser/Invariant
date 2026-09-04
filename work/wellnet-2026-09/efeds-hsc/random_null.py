"""Score the random-point null and merge it into decade_results.json.

Kept separate from decade_test.py only so that the null can be scored after the
(slow) random-position acquisition finishes, without re-running the whole fit.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import decade_test as D

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = os.path.join(HERE, "decade_random_shear_profiles.tsv")
    prof = D.load_profiles(p)
    T, X, W, R = [], [], [], []
    for v in prof.values():
        for x in v:
            if x["n"] >= 50 and x["err"] > 0 and np.isfinite(x["gt"]):
                T.append(x["gt"])
                X.append(x["gx"])
                W.append(1.0 / x["err"] ** 2)
                R.append(x["R"])
    T, X, W = np.array(T), np.array(X), np.array(W)
    m = float(np.sum(W * T) / np.sum(W))
    mx = float(np.sum(W * X) / np.sum(W))
    se = float(1.0 / math.sqrt(np.sum(W)))
    print("=" * 78)
    print("RANDOM-POINT NULL")
    print("=" * 78)
    print(f"\n   {len(prof)} random positions, {len(T)} points")
    print(f"   tangential {m:+.5f} +- {se:.5f}   ({m / se:+.2f} sigma)")
    print(f"   cross      {mx:+.5f} +- {se:.5f}   ({mx / se:+.2f} sigma)")
    out = {"n_positions": len(prof), "n_points": int(len(T)),
           "gt": m, "gx": mx, "sigma": se, "gt_sigma": m / se,
           "gx_sigma": mx / se, "passed": bool(abs(m / se) < 3)}
    rp = os.path.join(HERE, "decade_results.json")
    if os.path.exists(rp):
        res = json.load(open(rp, encoding="utf-8"))
        real = (res.get("N_nulls") or {}).get("gt_mean")
        if real:
            print(f"   cluster signal {real:+.5f} = "
                  f"{real / max(abs(m), se):.1f}x the random-point residual")
            out["cluster_signal_over_random"] = real / max(abs(m), se)
        res["X_random_null"] = out
        json.dump(res, open(rp, "w", encoding="utf-8"), indent=1)
        print("   merged into decade_results.json")
    print(f"   -> {'PASS' if out['passed'] else 'FAIL: spurious signal'}")


if __name__ == "__main__":
    main()
