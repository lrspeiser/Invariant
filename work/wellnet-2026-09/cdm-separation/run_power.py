"""run_power.py -- SIZE FIRST, then power, then verdicts.

Order is not negotiable, and it is the order Run BF's own sizing audit forced:
its nominal 0.01 realised 0.033, so no verdict here is taken at a nominal tail.

  P0  build calibration and AUDIT pools with disjoint seeds for every arm
  P1  SIZING: critical values from the CALIBRATION half only; realised false
      positive rate measured on the UNTOUCHED audit half, at nominal 0.05 and
      0.01, two-sided and one-sided
  P2  the number that matters: the rate at which each NEW-GRAVITY detector
      fires on the DARK MATTER universe (BF's 0.648 family-wise)
  P3  the converse: the power of each CDM-DISCRIMINATOR on the dark-matter
      universe, sized against a null of the surviving modified-gravity families
  P4  the joint procedure, and its family-wise rate on CDM
  P5  sample-size scaling -- the number of clusters at which each statistic
      reaches 3 sigma
  P6  responsiveness d(estimate)/d(injected) on the tensor amplitude scan
  P7  U10 (systematics only) survival
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import guard                                    # noqa: E402
import worker as W                              # noqa: E402
from universes.stats import rate_with_ci, responsiveness   # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

N_HALF = int(os.environ.get("N_HALF", 800))

# threshold amplitudes: the amplitude at which BF's own scans say the effect
# becomes observable at the family-wise critical value (E9_equivalence_at_threshold)
THRESH = {"U05_tensor_axis": 0.0200293, "U06_wellnet": 0.00718279,
          "U09_path_redshift": 0.00813803}
FID = {"U05_tensor_axis": 0.5, "U06_wellnet": 0.06, "U09_path_redshift": 0.03}

# arm -> (uid, knob, sys_scale, noise_scale)
ARMS = {
    "U01_newton":       ("U01_baryons_newton", None, 1.0, 1.0),
    "U03_mond":         ("U03_mond_scalar", None, 1.0, 1.0),
    "H0_scalar_null":   ("H0_scalar_null", None, 1.0, 1.0),
    "U10_systematics":  ("U10_systematics", None, 3.0, 1.0),
    "U02_cdm":          ("U02_cdm", None, 1.0, 1.0),
    "U02_cdm_3xsys":    ("U02_cdm", None, 3.0, 1.0),
    "U05_thresh":       ("U05_tensor_axis", THRESH["U05_tensor_axis"], 1.0, 1.0),
    "U05_fid":          ("U05_tensor_axis", FID["U05_tensor_axis"], 1.0, 1.0),
    "U05_A1":           ("U05_tensor_axis", 1.0, 1.0, 1.0),
    "U06_thresh":       ("U06_wellnet", THRESH["U06_wellnet"], 1.0, 1.0),
    "U06_fid":          ("U06_wellnet", FID["U06_wellnet"], 1.0, 1.0),
    "U09_thresh":       ("U09_path_redshift", THRESH["U09_path_redshift"], 1.0, 1.0),
    "U09_fid":          ("U09_path_redshift", FID["U09_path_redshift"], 1.0, 1.0),
}

SEED0 = {k: 3_100_000 + 7919 * i for i, k in enumerate(ARMS)}

# the candidate statistics, and what each one's alternative hypothesis IS
STATS = {
    "S_ext":   "new gravity: a quadrupole locked to the external axis",
    "G_ext":   "new gravity: galaxy m=3 locked to the external axis",
    "S_45":    "misspecified-axis control (must be a null detector)",
    "S_bar":   "dark matter: a quadrupole locked to the baryon major axis",
    "S_diff":  "signed contrast, external minus baryon axis",
    "S_morph": "dark matter: quadrupole power rises with baryon ellipticity",
    "S_shape": "radial shape contrast of the quadrupole power",
    "S_ext_raw":  "S_ext without studentising (the form BF's aniso_ext takes)",
    "S_bar_raw":  "S_bar without studentising",
    "S_diff_raw": "S_diff without studentising (BF's aniso_ext_minus_bar)",
    "S_45_raw":   "misspecified-axis control, unstudentised",
}
NEW_GRAVITY = ("S_ext", "G_ext")
CDM_DETECT = ("S_bar", "S_diff", "S_morph", "S_shape")

# the null family for a NEW-GRAVITY detector: exactly Run BF's calibration arms
CAL_NEWGRAV = ("U01_newton", "U03_mond", "H0_scalar_null", "U10_systematics")
# the same without the systematics arm, whose own baryon-aligned quadrupole
# drags the pooled null off zero and inflates every two-sided critical value
CAL_SCALAR = ("U01_newton", "U03_mond", "H0_scalar_null")
# the null family for a CDM detector: the surviving modified-gravity families
CAL_CDM = ("U03_mond", "H0_scalar_null", "U05_thresh", "U05_fid",
           "U06_thresh", "U06_fid", "U09_thresh", "U09_fid")


def build(arm, n, half, log=True, n_clu=12, seed_off=0):
    uid = ARMS[arm]
    base = SEED0[arm] + (0 if half == "cal" else 500_000) + seed_off
    jobs = [(uid, base + i, n_clu) for i in range(n)]
    t0 = time.time()
    out = W.run_batch(jobs)
    if log:
        print(f"  {arm:<18} {half:<4} n={len(out):<5} {time.time() - t0:6.1f}s",
              flush=True)
    return out


def col(pool, key):
    return np.array([r.get(key, 0.0) for r in pool], float)


def crit_two(vals, alpha):
    return float(np.quantile(np.abs(vals[np.isfinite(vals)]), 1 - alpha))


def crit_one(vals, alpha, side=+1):
    v = vals[np.isfinite(vals)] * side
    return float(np.quantile(v, 1 - alpha))


def main():
    led = guard.start()
    t00 = time.time()
    print(f"building pools, N_HALF={N_HALF}", flush=True)
    pools = {}
    for arm in ARMS:
        pools[(arm, "cal")] = build(arm, N_HALF, "cal")
        pools[(arm, "aud")] = build(arm, N_HALF, "aud")

    out = {"n_half": N_HALF, "arms": list(ARMS), "statistics": STATS}

    # ------------------------------------------------------- P1  SIZING
    print("\nP1 sizing on the untouched audit half", flush=True)
    sizing = {}
    for family, arms in (("new_gravity_null", CAL_NEWGRAV),
                         ("scalar_null", CAL_SCALAR), ("cdm_null", CAL_CDM)):
        cal = [r for a in arms for r in pools[(a, "cal")]]
        aud = [r for a in arms for r in pools[(a, "aud")]]
        fam = {}
        for s in STATS:
            vc, va = col(cal, s), col(aud, s)
            row = {}
            for alpha in (0.05, 0.01):
                c2 = crit_two(vc, alpha)
                cp_ = crit_one(vc, alpha, +1)
                cm = crit_one(vc, alpha, -1)
                row[f"nominal_{alpha}"] = dict(
                    crit_two_sided=c2,
                    realised_fpr_two_sided=rate_with_ci(
                        int(np.sum(np.abs(va) >= c2)), len(va)),
                    crit_upper=cp_,
                    realised_fpr_upper=rate_with_ci(
                        int(np.sum(va >= cp_)), len(va)),
                    crit_lower=-cm,
                    realised_fpr_lower=rate_with_ci(
                        int(np.sum(-va >= cm)), len(va)))
            row["null_mean"] = float(np.mean(va))
            row["null_sd"] = float(np.std(va))
            fam[s] = row
        sizing[family] = fam
    out["P1_sizing"] = sizing

    # measured 0.05 critical values, used for every verdict below
    CRIT = {}
    for family, arms in (("new_gravity_null", CAL_NEWGRAV),
                         ("scalar_null", CAL_SCALAR), ("cdm_null", CAL_CDM)):
        cal = [r for a in arms for r in pools[(a, "cal")]]
        CRIT[family] = {s: dict(two=crit_two(col(cal, s), 0.05),
                                up=crit_one(col(cal, s), 0.05, +1),
                                lo=crit_one(col(cal, s), 0.05, -1))
                        for s in STATS}
    out["critical_values"] = CRIT

    # ------------------------------------------ P2 the number that matters
    print("P2 rate on the dark-matter universe", flush=True)
    P2 = {}
    for s in STATS:
        c = CRIT["new_gravity_null"][s]
        row = {}
        for arm in ARMS:
            v = col(pools[(arm, "aud")], s)
            row[arm] = dict(
                two_sided=rate_with_ci(int(np.sum(np.abs(v) >= c["two"])), len(v)),
                upper=rate_with_ci(int(np.sum(v >= c["up"])), len(v)),
                lower=rate_with_ci(int(np.sum(-v >= c["lo"])), len(v)),
                mean=float(np.mean(v)), sd=float(np.std(v)))
        P2[s] = row
    out["P2_rates_vs_newgrav_null"] = P2

    P2b = {}
    for s in STATS:
        c = CRIT["scalar_null"][s]
        row = {}
        for arm in ARMS:
            v = col(pools[(arm, "aud")], s)
            row[arm] = dict(
                two_sided=rate_with_ci(int(np.sum(np.abs(v) >= c["two"])), len(v)),
                upper=rate_with_ci(int(np.sum(v >= c["up"])), len(v)),
                lower=rate_with_ci(int(np.sum(-v >= c["lo"])), len(v)),
                mean=float(np.mean(v)), sd=float(np.std(v)))
        P2b[s] = row
    out["P2_rates_vs_scalar_null"] = P2b

    # family-wise, directly comparable with BF's 0.648
    fam_two, fam_up = {}, {}
    for arm in ARMS:
        hit2 = np.zeros(len(pools[(arm, "aud")]), bool)
        hitu = np.zeros(len(pools[(arm, "aud")]), bool)
        for s in NEW_GRAVITY:
            c = CRIT["new_gravity_null"][s]
            v = col(pools[(arm, "aud")], s)
            hit2 |= np.abs(v) >= c["two"]
            hitu |= v >= c["up"]
        fam_two[arm] = rate_with_ci(int(hit2.sum()), len(hit2))
        fam_up[arm] = rate_with_ci(int(hitu.sum()), len(hitu))
    out["P2_familywise_newgravity"] = {"two_sided": fam_two, "one_sided_upper": fam_up}

    # ---------------------------------------- P3 power of the CDM detectors
    print("P3 power of the CDM discriminators", flush=True)
    P3 = {}
    for s in CDM_DETECT:
        c = CRIT["cdm_null"][s]
        row = {}
        for arm in ARMS:
            v = col(pools[(arm, "aud")], s)
            row[arm] = dict(
                two_sided=rate_with_ci(int(np.sum(np.abs(v) >= c["two"])), len(v)),
                upper=rate_with_ci(int(np.sum(v >= c["up"])), len(v)),
                lower=rate_with_ci(int(np.sum(-v >= c["lo"])), len(v)),
                mean=float(np.mean(v)), sd=float(np.std(v)))
        P3[s] = row
    out["P3_cdm_detector_rates"] = P3

    # ------------------------------------------------ P4 joint procedure
    # declare NEW GRAVITY only if the external-axis statistic fires AND the
    # baryon-axis statistic does not.  The second clause is the CDM veto.
    print("P4 joint procedure", flush=True)
    P4 = {}
    ce = CRIT["new_gravity_null"]["S_ext"]
    cb = CRIT["cdm_null"]["S_bar"]
    for arm in ARMS:
        se = col(pools[(arm, "aud")], "S_ext")
        sb = col(pools[(arm, "aud")], "S_bar")
        ge = col(pools[(arm, "aud")], "G_ext")
        cg = CRIT["new_gravity_null"]["G_ext"]
        fire = (np.abs(se) >= ce["two"]) | (np.abs(ge) >= cg["two"])
        veto = sb >= cb["up"]
        P4[arm] = dict(
            fires_no_veto=rate_with_ci(int(np.sum(fire & ~veto)), len(se)),
            fires=rate_with_ci(int(fire.sum()), len(se)),
            veto_rate=rate_with_ci(int(veto.sum()), len(se)))
    out["P4_joint"] = P4

    # ----------------------------------------------- P5 sample-size scaling
    print("P5 sample-size scaling", flush=True)
    P5 = {}
    for nclu in (3, 6, 12, 18):
        sub = {}
        for arm in ("U02_cdm", "U03_mond", "U05_fid"):
            pool = build(arm, 300, f"n{nclu}", log=False, n_clu=nclu,
                         seed_off=1_000_000 + 40_000 * nclu)
            sub[arm] = {s: dict(mean=float(np.mean(col(pool, s))),
                                sd=float(np.std(col(pool, s)))) for s in STATS}
        P5[str(nclu)] = sub
        print(f"  n_clu={nclu:2d} S_bar(U02) = {sub['U02_cdm']['S_bar']['mean']:8.2f}"
              f" +- {sub['U02_cdm']['S_bar']['sd']:.2f}", flush=True)
    out["P5_sample_size"] = P5

    # ------------------------------------------------ P6 responsiveness
    print("P6 responsiveness on the tensor amplitude scan", flush=True)
    amps = [0.0, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    scan = {}
    for i, a in enumerate(amps):
        pool = W.run_batch([(("U05_tensor_axis", a, 1.0, 1.0),
                             4_400_000 + 1301 * i + j, 12) for j in range(300)])
        scan[str(a)] = {s: dict(mean=float(np.mean(col(pool, s))),
                                sd=float(np.std(col(pool, s)))) for s in STATS}
    out["P6_tensor_scan"] = scan
    out["P6_responsiveness"] = {
        s: responsiveness(amps, [scan[str(a)][s]["mean"] for a in amps])
        for s in STATS}

    for uid, knobs, tag in (
            ("U06_wellnet", [0.0, 0.0072, 0.015, 0.03, 0.06, 0.12], "wellnet"),
            ("U09_path_redshift", [0.0, 0.008, 0.015, 0.03, 0.06, 0.12], "path")):
        sc = {}
        for i, a in enumerate(knobs):
            pool = W.run_batch([((uid, a, 1.0, 1.0),
                                 5_500_000 + 90_001 * (zlib.crc32(tag.encode()) % 97) + 1301 * i + j, 12)
                                for j in range(250)])
            sc[str(a)] = {s: dict(mean=float(np.mean(col(pool, s))),
                                  sd=float(np.std(col(pool, s)))) for s in STATS}
        out[f"P6_{tag}_scan"] = sc
        out[f"P6_{tag}_responsiveness"] = {
            s: responsiveness(knobs, [sc[str(a)][s]["mean"] for a in knobs])
            for s in STATS}

    W.close_pool()
    out["provenance"] = guard.stop()
    out["elapsed_s"] = time.time() - t00
    p = os.path.join(RES, "P_power.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {p}   ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
