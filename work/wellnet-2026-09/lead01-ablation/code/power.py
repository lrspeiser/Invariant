"""Item 2, addendum: where the uncertainty on the paired difference comes from,
and therefore whether more held-out clusters could ever settle it.

The frozen-coefficient bootstrap resamples only the held-out objects, so its
spread shrinks as 1/sqrt(n_test).  The nested bootstrap also resamples the
training set, and THAT contribution does not shrink with n_test at all.  If the
training contribution alone already exceeds the observed difference, no amount
of extra validation data can separate the two models.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as C
from ablation import paired_bootstrap

RNG = np.random.default_rng(20260906)


def main():
    d = C.load_ladder()
    t = C.system_table(d)
    tr = t["rank"] <= 4
    te = t["rank"] >= 5
    p = os.path.join(C.LANE, "ablation.json")
    res = json.load(open(p))
    pb = res["paired_bootstrap"]["frozen_coefficients"]
    pbn = res["paired_bootstrap"]["nested_refit_on_resampled_training"]
    obs = pb["observed_delta_rms"]
    s_test = pb["boot_sd"]
    s_all = pbn["boot_sd"]
    s_train = math.sqrt(max(s_all ** 2 - s_test ** 2, 0.0))
    n_te = pb["n_test"]
    print("=" * 78)
    print("ITEM 2 addendum -- can more clusters ever settle M1 vs M3?")
    print("=" * 78)
    print(f"   observed dRMS = RMS(M1) - RMS(M3) = {obs:+.5f} dex")
    print(f"   uncertainty from resampling the {n_te} held-out clusters : "
          f"{s_test:.5f} dex   (shrinks as 1/sqrt(n_test))")
    print(f"   uncertainty from the 200-system TRAINING set             : "
          f"{s_train:.5f} dex   (does NOT shrink with n_test)")
    print(f"   total                                                    : "
          f"{s_all:.5f} dex")
    print(f"\n   ceiling on the separation with an INFINITE held-out cluster "
          f"sample:")
    print(f"      z_max = {obs / s_train:.2f} sigma")
    ns = {}
    for z in (2.0, 3.0):
        need = (obs / z) ** 2 - s_train ** 2
        ns[f"n_test_for_{z:.0f}sigma"] = (
            float(n_te * (s_test ** 2) / need) if need > 0 else None)
    print(f"      held-out clusters needed for 2 sigma: "
          f"{ns['n_test_for_2sigma'] if ns['n_test_for_2sigma'] else 'unreachable'}")
    print(f"      held-out clusters needed for 3 sigma: "
          f"{ns['n_test_for_3sigma'] if ns['n_test_for_3sigma'] else 'unreachable'}")

    # how much would the TRAINING set have to grow?
    print("\n   the training contribution scales as 1/sqrt(n_train); the "
          "training\n   set would have to grow by the following factor for a "
          "3 sigma verdict\n   with an infinite validation sample:")
    fac = (3 * s_train / obs) ** 2
    print(f"      x{fac:.1f}  ->  {200 * fac:.0f} training systems "
          f"(currently 200)")

    # empirical: recompute the frozen bootstrap on subsamples of the test set
    print("\n   empirical check of the 1/sqrt(n_test) scaling "
          "(RANDOM subsets, 12 repeats each, coefficients frozen):")
    scal = []
    idx = np.where(te)[0]
    for frac in (0.25, 0.5, 1.0):
        k = max(6, int(round(frac * n_te)))
        sds = []
        for _ in range(12 if k < n_te else 1):
            pick = RNG.choice(idx, k, replace=False)
            sub = np.zeros(len(t["lg"]), bool)
            sub[pick] = True
            b = paired_bootstrap(t, tr, sub, "M1", "M3", nb=2000)
            sds.append(b["boot_sd"])
        m = float(np.mean(sds))
        scal.append((k, m, float(m * math.sqrt(k / n_te))))
        print(f"      n_test = {k:3d}: mean bootstrap sd {m:.5f} dex   "
              f"-> sd x sqrt(n/{n_te}) = {m * math.sqrt(k / n_te):.5f} "
              f"(constant if the scaling holds)")

    out = dict(observed_delta_rms=obs, n_test=n_te,
               sd_from_test_resampling=s_test,
               sd_from_training_resampling=s_train,
               sd_total=s_all,
               z_ceiling_with_infinite_validation=float(obs / s_train),
               n_test_required=ns,
               training_growth_factor_for_3sigma=float(fac),
               training_systems_required_for_3sigma=float(200 * fac),
               subsample_scaling_sd_x_sqrt_n=[[int(a), float(b_), float(c)]
                                  for a, b_, c in scal],
               verdict=("the uncertainty contributed by the 200-system training "
                        "set alone is comparable to the entire observed "
                        "difference, so the two models cannot be separated by "
                        "adding validation clusters"))
    res["paired_bootstrap"]["power"] = out
    json.dump(res, open(p, "w"), indent=2)
    print(f"\nmerged into {p}")


if __name__ == "__main__":
    main()
