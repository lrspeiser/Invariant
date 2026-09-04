"""
How much of Run AT's 29% is THIS BUG?

Run AT ran a forward null in which the true excess has no radial structure at
all, switched every source of measurement noise off, and still recovered
S1 = -0.2067 and S3 = -0.1359, i.e. 28.3% of the observed slope -0.4803.  That
number is the pipeline's own deterministic bias.

This script splits it, inside Run AT's own machinery, with ONE switch:
the temperature grid on which the simulated cluster is "observed".

    clamped   T is observed on the real coarse grid, which stops at
              median r/R500 = 0.91 -- so np.interp clamps beyond it, exactly
              as the bench does.  This is Run AT's configuration.
    covered   T is observed on a grid with the SAME inner bins, extended
              outward with the same log spacing to the last density bin, so
              nothing is ever extrapolated.  Everything else -- the truth, the
              boundary pressure, the R500 inference, the statistics -- is
              byte-for-byte the same code.

The difference between the two is the part of the pipeline bias that is this
bug; what survives in `covered` is everything else (the R500 inference step,
the coarse grid, np.gradient, the pooling).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

AUD = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
       "work/wellnet-2026-09/r500-audit")
if AUD not in sys.path:
    sys.path.insert(0, AUD)

import ingest as I                                                  # noqa: E402
import nullsim as N                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KPC = I.KPC


def extend_coarse(T, pad_factor=1.0):
    """Extend r_coarse outward to the last density bin, same log spacing.

    r_bnd is DELIBERATELY left at its original value so that make_truth's
    pressure anchor -- and therefore the truth itself -- is unchanged.  Only
    the observation grid moves.
    """
    rc = T.r_coarse
    rmax = T.r.max() * 1.000001
    if rc.max() >= rmax:
        return 0
    step = math.log(rc[-1] / rc[-2])
    n = int(math.ceil(math.log(rmax / rc[-1]) / step))
    add = rc[-1] * np.exp(step * np.arange(1, n + 1))
    T.r_coarse = np.concatenate([rc, add])
    T.sig_lnT = np.concatenate([T.sig_lnT, np.full(n, T.sig_lnT[-1])])
    return n


def run_one(covered, seed=0):
    clusters = [I.load_xcop_cluster(nm) for nm in I.XCOP_EXPECTED]
    clusters = [c for c in clusters if c is not None]
    assert len(clusters) == 12, f"loaded {len(clusters)} clusters, expected 12"
    TS = [N.Template(c) for c in clusters]
    added = 0
    if covered:
        for T in TS:
            added += extend_coarse(T)
    cfg0 = dict(N.DEFAULT_CFG, ne_scale=0.0, T_scale=0.0, T_calib=0.0,
                rho_corr=0.0)
    rr = N.one_realisation(TS, np.random.default_rng(seed), cfg0)
    st = N.stats_of(rr)
    st["n_bins_added"] = added
    st["y_minus_ytrue_median"] = float(np.median(rr["y"] - rr["y_true"]))
    st["y_minus_ytrue_rms"] = float(np.std(rr["y"] - rr["y_true"]))
    # R500 recovery: the other half of the publication chain
    st["R500_ratio_median"] = float(np.median(
        [rr["R500_obs_map"][k] / rr["R500_true_map"][k] for k in rr["R500_obs_map"]]))
    return st


def main():
    rec = json.load(open(os.path.join(AUD, "results.json"), encoding="utf-8"))
    at = rec["job1"]["forward_null_noiseless"]
    obs_S3 = rec["job2d"]["clamped_temperature_bug"]["S3_all"]
    obs_S1 = rec["job2d"]["clamped_temperature_bug"]["S1_all"]

    print("=" * 78)
    print("Run AT's noiseless forward null, with and without the clamp")
    print("=" * 78)
    a = run_one(covered=False)
    b = run_one(covered=True)
    print(f"   recorded by Run AT     S1 = {at['S1_hse']:+.4f}   "
          f"S3 = {at['S3_hse']:+.4f}")
    print(f"   reproduced here        S1 = {a['S1_hse']:+.4f}   "
          f"S3 = {a['S3_hse']:+.4f}")
    ok = (abs(a["S1_hse"] - at["S1_hse"]) < 1e-9
          and abs(a["S3_hse"] - at["S3_hse"]) < 1e-9)
    print(f"   bit-identical reproduction: {ok}")
    print(f"\n   clamp switched OFF     S1 = {b['S1_hse']:+.4f}   "
          f"S3 = {b['S3_hse']:+.4f}   ({b['n_bins_added']} T bins added)")
    print(f"   R500 recovery ratio    clamped {a['R500_ratio_median']:.4f}   "
          f"covered {b['R500_ratio_median']:.4f}")

    tot1, tot3 = a["S1_hse"], a["S3_hse"]
    res1, res3 = b["S1_hse"], b["S3_hse"]
    print(f"\n   {'':<28}{'S1 (corr)':>14}{'S3 (slope)':>14}")
    print("   " + "-" * 56)
    print(f"   {'observed on real data':<28}{obs_S1:>+14.4f}{obs_S3:>+14.4f}")
    print(f"   {'total pipeline bias':<28}{tot1:>+14.4f}{tot3:>+14.4f}")
    print(f"   {'  of which THE CLAMP':<28}{tot1-res1:>+14.4f}{tot3-res3:>+14.4f}")
    print(f"   {'  of which everything else':<28}{res1:>+14.4f}{res3:>+14.4f}")
    print("   " + "-" * 56)
    print(f"   {'bias as % of observed':<28}"
          f"{100*tot1/obs_S1:>13.1f}%{100*tot3/obs_S3:>13.1f}%")
    print(f"   {'  clamp share of that':<28}"
          f"{100*(tot1-res1)/tot1:>13.1f}%{100*(tot3-res3)/tot3:>13.1f}%")
    print(f"   {'  clamp as % of observed':<28}"
          f"{100*(tot1-res1)/obs_S1:>13.1f}%{100*(tot3-res3)/obs_S3:>13.1f}%")

    out = dict(recorded_AT=at, reproduced_clamped=a, clamp_removed=b,
               reproduction_exact=bool(ok),
               observed_S1=obs_S1, observed_S3=obs_S3,
               split=dict(
                   total_bias_S1=tot1, total_bias_S3=tot3,
                   clamp_S1=tot1 - res1, clamp_S3=tot3 - res3,
                   other_S1=res1, other_S3=res3,
                   bias_pct_of_observed_S3=100 * tot3 / obs_S3,
                   clamp_pct_of_bias_S3=100 * (tot3 - res3) / tot3,
                   clamp_pct_of_observed_S3=100 * (tot3 - res3) / obs_S3,
                   bias_pct_of_observed_S1=100 * tot1 / obs_S1,
                   clamp_pct_of_bias_S1=100 * (tot1 - res1) / tot1,
                   clamp_pct_of_observed_S1=100 * (tot1 - res1) / obs_S1))
    json.dump(out, open(os.path.join(HERE, "at_null_split.json"), "w",
                        encoding="utf-8"), indent=1, default=float)
    print("\n   wrote at_null_split.json")
    return out


if __name__ == "__main__":
    main()
