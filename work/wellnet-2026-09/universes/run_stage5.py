"""run_stage5.py -- the Stage 5 experiments.

Order matters and is not negotiable:

  E0  SIZE THE TEST FIRST.  A-vs-A separations (identical universe, different
      seeds) give the realised false-positive rate and the multiplicity null
      of the statistic that is actually used.  Nothing is interpreted until
      this passes.
  E1  the pairwise observational equivalence-class map on the fiducial corpus
  E2  the same map test by test
  E3  amplitude scans: at what amplitude does each effect become observable
  E4  the seven Stage 5 identifiability questions, including the dark-matter
      false-positive control
  E5  admissibility gates: coarse-graining, reciprocity, gauge
  E6  the missing-observation oracle for every indistinguishable pair
  E7  every knob translated into a predicted observable amplitude

The test statistic everywhere is the MAX over {whole-corpus discriminant,
every single channel}, with the look-elsewhere cost of that max inside the
critical value.  Calibration draws and audit draws are disjoint everywhere.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from universes import analysis as an          # noqa: E402
from universes import generate as gn          # noqa: E402
from universes import physics as ph           # noqa: E402
from universes import provenance as pv        # noqa: E402
from universes import scenes as sc            # noqa: E402
from universes import stats as st             # noqa: E402
from universes.analysis import CHANNELS       # noqa: E402

RES = os.path.join(HERE, "results")
POOLDIR = os.path.join(RES, "pools")
os.makedirs(POOLDIR, exist_ok=True)

UIDS = list(ph.UNIVERSES)
N_POOL = int(os.environ.get("N_POOL", 480))
N_AMP = int(os.environ.get("N_AMP", 480))
SEED0 = 700000
NPERM = 3000
KEYS = None
CHAN_IDX = None


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"  wrote {name}", flush=True)


def pool(tag, arm, n, seed_base):
    """Draw (or load) a pool of corpora for one arm.  Cached on disk."""
    fn = os.path.join(POOLDIR, f"{tag}_{n}_{seed_base}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    recs = gn.run_batch([(arm, seed_base + i) for i in range(n)])
    with open(fn, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    return recs


def quarters(recs):
    """Calibration and AUDIT matrices, DISJOINT, same size everywhere.

    Quarters, so the A-vs-A sizing (which needs four disjoint groups from one
    universe) and the A-vs-B tests operate at exactly the same n.  A critical
    value measured at one n and applied at another is not a calibration.
    """
    X, _ = gn.to_matrix(recs, KEYS)
    q = len(X) // 4
    return X[:q], X[2 * q:3 * q]


def sepmax(Ac, Bc, Aa, Ba, seed):
    return st.separation_max(Ac, Bc, Aa, Ba, CHAN_IDX, n_perm=NPERM, seed=seed)


# ======================================================================
def main():
    global KEYS, CHAN_IDX
    t00 = time.time()
    gn.get_lib()
    gn.get_pool()
    pv.start_ledger(os.path.dirname(HERE))
    manifest = pv.run_manifest({
        "lane": "work/wellnet-2026-09/universes",
        "n_pool_per_arm": N_POOL, "n_per_amplitude": N_AMP, "n_perm": NPERM,
        "corpus": {"n_gal": gn.N_GAL, "n_clu": gn.N_CLU, "n_sn": gn.N_SN,
                   "lib_gal": gn.N_GAL_LIB, "lib_clu": gn.N_CLU_LIB,
                   "lib_seed": gn.LIB_SEED},
        "universes": ph.UNIVERSES, "fiducial_knobs": ph.FIDUCIAL,
        "statistic": ("max over {whole-corpus shrinkage-LDA discriminant, each "
                      "single channel}; the look-elsewhere cost of the max is "
                      "inside the critical value"),
    })

    arms = {u: (u, None, 3.0 if u == "U10_systematics" else 1.0, 1.0) for u in UIDS}
    arms["H0_scalar_null"] = ("H0_scalar_null", None, 1.0, 1.0)
    arms["U02_cdm_hard"] = ("U02_cdm", None, 3.0, 1.0)

    print(f"[pools] {len(arms)} arms x {N_POOL} draws", flush=True)
    pools = {}
    for i, (name, arm) in enumerate(arms.items()):
        pools[name] = pool(name, arm, N_POOL, SEED0 + 10000 * i)
        print(f"  {name}: {len(pools[name])} draws ({time.time()-t00:.0f}s)", flush=True)

    KEYS = sorted(pools[UIDS[0]][0]["features"])
    lib = gn.get_lib()
    from universes import corpus as cp
    rngp = np.random.default_rng(1)
    Cp = cp.draw_corpus(ph.draw_universe("U03_mond_scalar", rngp), lib, rngp,
                        n_gal=gn.N_GAL, n_clu=gn.N_CLU, n_sn=gn.N_SN)
    chan_of = an.analyse(Cp, split_seed=1)["channels"]
    CHAN_IDX = {c: [i for i, k in enumerate(KEYS) if chan_of.get(k) == c]
                for c in CHANNELS}
    CHAN_IDX = {c: v for c, v in CHAN_IDX.items() if len(v) >= 2}
    save("channel_map.json", {"channels": {c: [KEYS[i] for i in v]
                                           for c, v in CHAN_IDX.items()},
                              "feature_order": KEYS})

    # =================================================================
    # E0  SIZE THE TEST
    # =================================================================
    print("[E0] sizing on A-vs-A nulls", flush=True)
    null_zmax, null_best, per_arm, per_test_null = [], [], {}, {}
    NREP = 20
    for name, recs in pools.items():
        X, _ = gn.to_matrix(recs, KEYS)
        q = len(X) // 4
        zs = []
        for rep in range(NREP):
            idx = np.random.default_rng(9000 + rep).permutation(len(X))
            r = sepmax(X[idx[:q]], X[idx[q:2 * q]],
                       X[idx[2 * q:3 * q]], X[idx[3 * q:4 * q]], seed=rep)
            zs.append(r["z_max"]); null_zmax.append(r["z_max"])
            null_best.append(r["best_test"])
            for k, v in r["per_test"].items():
                per_test_null.setdefault(k, []).append(v["z"])
        per_arm[name] = {"median_zmax": float(np.median(zs)),
                         "max_zmax": float(np.max(zs))}
    nz = np.array(null_zmax)
    z_crit_single = float(np.quantile(nz, 0.95))
    zmax45 = [np.max(np.random.default_rng(k).choice(nz, size=45)) for k in range(6000)]
    z_crit_family = float(np.quantile(zmax45, 0.95))
    fp_single = st.rate_with_ci(int((nz >= z_crit_single).sum()), len(nz))
    e0 = {"n_null_tests": len(nz), "n_reps_per_arm": NREP,
          "statistic": manifest["statistic"],
          "z_crit_single_pair_95pct": z_crit_single,
          "z_crit_family_wise_45_pairs_95pct": z_crit_family,
          "realised_fp_at_the_single_pair_critical_value": fp_single,
          "null_zmax_median": float(np.median(nz)),
          "null_zmax_max": float(nz.max()),
          "which_test_wins_under_H0": {k: int(null_best.count(k))
                                       for k in sorted(set(null_best))},
          "per_test_null": {k: {"z95": float(np.quantile(v, 0.95)),
                                "z_median": float(np.median(v)),
                                "z_max": float(np.max(v)), "n": len(v)}
                            for k, v in per_test_null.items()},
          "per_arm": per_arm,
          "note": ("A-vs-A = the SAME universe, different seeds, independent "
                   "nuisance draws; discriminants fitted on calibration draws "
                   "and scored on disjoint audit draws.")}
    save("E0_sizing.json", e0)
    print(f"  z_crit single {z_crit_single:.2f}  family {z_crit_family:.2f}  "
          f"null median {np.median(nz):.2f}", flush=True)

    # =================================================================
    # E1 / E2
    # =================================================================
    print("[E1] pairwise equivalence map", flush=True)
    pair_rows = []
    for i in range(len(UIDS)):
        for j in range(i + 1, len(UIDS)):
            a, b = UIDS[i], UIDS[j]
            Ac, Aa = quarters(pools[a]); Bc, Ba = quarters(pools[b])
            r = sepmax(Ac, Bc, Aa, Ba, seed=i * 100 + j)
            r.update({"a": a, "b": b})
            pair_rows.append(r)
        print(f"  {UIDS[i]} ({time.time()-t00:.0f}s)", flush=True)

    parent = {u: u for u in UIDS}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for r in pair_rows:
        if r["z_max"] < z_crit_family:
            ra, rb = find(r["a"]), find(r["b"])
            if ra != rb:
                parent[ra] = rb
    classes = {}
    for u in UIDS:
        classes.setdefault(find(u), []).append(u)
    save("E1_equivalence_map.json",
         {"z_crit_family": z_crit_family, "z_crit_single": z_crit_single,
          "pairs": [{k: v for k, v in r.items() if k != "per_test"} for r in pair_rows],
          "equivalence_classes": [sorted(v) for v in classes.values()],
          "indistinguishable_pairs": [
              {"a": r["a"], "b": r["b"], "z_max": r["z_max"],
               "best_test": r["best_test"], "auc_full": r["auc_full"]}
              for r in pair_rows if r["z_max"] < z_crit_family]})
    save("E2_channel_separation.json",
         {"z_crit_family": z_crit_family,
          "per_test_null_z95": {k: v["z95"] for k, v in e0["per_test_null"].items()},
          "rows": [{"a": r["a"], "b": r["b"],
                    **{k: v["z"] for k, v in r["per_test"].items()}}
                   for r in pair_rows]})

    # =================================================================
    # E3  amplitude scans
    # =================================================================
    print("[E3] amplitude scans", flush=True)
    SCAN = {
        "U04_env_scalar": [0.0, 0.15, 0.3, 0.6, 1.2, 2.4],
        "U05_tensor_axis": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        "U06_wellnet": [0.0, 0.015, 0.03, 0.06, 0.12, 0.24],
        "U07_memory": [0.0, 0.05, 0.10, 0.20, 0.40, 0.80],
        "U08_ep_slip": [0.0, 0.025, 0.05, 0.10, 0.20, 0.40],
        "U09_path_redshift": [0.0, 0.0075, 0.015, 0.03, 0.06, 0.12],
    }
    Bc3, Ba3 = quarters(pools["U03_mond_scalar"])
    scan_out = {}
    for k, (uid, amps) in enumerate(SCAN.items()):
        rows = []
        for m, amp in enumerate(amps):
            recs = pool(f"scan_{uid}_{amp:g}", (uid, amp, 1.0, 1.0), N_AMP,
                        SEED0 + 400000 + 20000 * k + 900 * m)
            Ac, Aa = quarters(recs)
            n, na = min(len(Ac), len(Bc3)), min(len(Aa), len(Ba3))
            r = sepmax(Ac[:n], Bc3[:n], Aa[:na], Ba3[:na], seed=31 + k * 10 + m)
            rows.append({"amp": amp, "z": r["z_max"], "best_test": r["best_test"],
                         "z_full": r["z_full"], "auc_full": r["auc_full"],
                         "n": len(recs),
                         "detectors": {d: float(np.mean([x["detectors"][d] for x in recs]))
                                       for d in recs[0]["detectors"]},
                         "axis_R": float(np.mean([x["axis_R"] for x in recs])),
                         "axis_err": float(np.mean([x["axis_err"] for x in recs])),
                         "axis_proj": float(np.mean([x["axis_proj"] for x in recs])),
                         "axis_proj45": float(np.mean([x["axis_proj45"] for x in recs]))})
        aa = [r["amp"] for r in rows]
        scan_out[uid] = {"knob": ph.KNOB[uid],
                         "fiducial": ph.FIDUCIAL[uid][ph.KNOB[uid]], "rows": rows,
                         "z_crit_family": z_crit_family,
                         "z5": st.threshold_amplitude(aa, [r["z"] for r in rows], 5.0),
                         "z3": st.threshold_amplitude(aa, [r["z"] for r in rows], 3.0),
                         "z_crit": st.threshold_amplitude(aa, [r["z"] for r in rows],
                                                          z_crit_family),
                         "responsiveness_z": st.responsiveness(aa, [r["z"] for r in rows])}
        print(f"  {uid}: " + ", ".join(f"{r['amp']:g}:{r['z']:.1f}" for r in rows),
              flush=True)
    save("E3_amplitude_scans.json", scan_out)

    # =================================================================
    # E4  the seven questions
    # =================================================================
    print("[E4] identifiability questions", flush=True)
    q = {}
    H = N_POOL // 2

    def rate(vals, crit):
        v = np.abs(np.asarray(vals, float))
        return st.rate_with_ci(int((v >= crit).sum()), len(v))

    a0h = np.array([r["a0_hat"] for r in pools["U03_mond_scalar"]])
    a0t = np.array([r["a0_true"] for r in pools["U03_mond_scalar"]])
    q["Q1_recover_scalar"] = {
        "estimator": "one-parameter RAR fit to the galaxy channel only",
        "responsiveness_dlog_a0hat_dlog_a0true": st.responsiveness(a0t, a0h),
        "bias_dex": float(np.mean(a0h - a0t)),
        "scatter_dex": float(np.std(a0h - a0t)), "n": int(len(a0h)),
        "on_U02_cdm": {"bias_dex": float(np.mean(
            np.array([r["a0_hat"] for r in pools["U02_cdm"]])
            - np.array([r["a0_true"] for r in pools["U02_cdm"]]))),
            "scatter_dex": float(np.std([r["a0_hat"] for r in pools["U02_cdm"]]))},
        "on_U01_baryons": {"a0_hat_median": float(np.median(
            [r["a0_hat"] for r in pools["U01_baryons_newton"]]))},
    }

    null_arms = ["H0_scalar_null", "U03_mond_scalar", "U04_env_scalar"]
    ng = np.concatenate([[r["detectors"]["gal_aniso"] for r in pools[a]] for a in null_arms])
    nc = np.concatenate([[r["detectors"]["aniso_ext"] for r in pools[a]] for a in null_arms])
    cal_g = st.detector_calibration(ng[::2]); aud_g = ng[1::2]
    cal_c = st.detector_calibration(nc[::2]); aud_c = nc[1::2]
    q["Q2_scalar_vs_anisotropy"] = {
        "null_family": ("7 qualitatively different scalar families including "
                        "surface-density-gated, potential-depth-gated and an "
                        "unbounded smooth random response, PLUS the "
                        "environment-scalar universe"),
        "galaxy_m3_detector": {
            "critical_value_from_calibration": cal_g,
            "audit_false_positive_rate": rate(aud_g, cal_g["crit"]),
            "power_on_U05_tensor": rate([r["detectors"]["gal_aniso"]
                                         for r in pools["U05_tensor_axis"]], cal_g["crit"]),
            "rate_on_U02_cdm": rate([r["detectors"]["gal_aniso"]
                                     for r in pools["U02_cdm"]], cal_g["crit"]),
            "rate_on_U06_wellnet": rate([r["detectors"]["gal_aniso"]
                                         for r in pools["U06_wellnet"]], cal_g["crit"]),
            "rate_on_U10_systematics": rate([r["detectors"]["gal_aniso"]
                                             for r in pools["U10_systematics"]],
                                            cal_g["crit"])},
        "cluster_shear_quadrupole_detector": {
            "critical_value_from_calibration": cal_c,
            "audit_false_positive_rate": rate(aud_c, cal_c["crit"]),
            "power_on_U05_tensor": rate([r["detectors"]["aniso_ext"]
                                         for r in pools["U05_tensor_axis"]], cal_c["crit"]),
            "rate_on_U02_cdm": rate([r["detectors"]["aniso_ext"]
                                     for r in pools["U02_cdm"]], cal_c["crit"]),
            "rate_on_U10_systematics": rate([r["detectors"]["aniso_ext"]
                                             for r in pools["U10_systematics"]],
                                            cal_c["crit"])},
    }

    q["Q3_external_axis"] = {
        "aligned": {u: {"median_err_deg": float(np.median([r["axis_err"] for r in pools[u]])),
                        "concentration_R": float(np.median([r["axis_R"] for r in pools[u]])),
                        "projection": float(np.mean([r["axis_proj"] for r in pools[u]]))}
                    for u in ("U03_mond_scalar", "U05_tensor_axis", "U02_cdm",
                              "U10_systematics", "H0_scalar_null")},
        "misaligned_45deg_control": {u: float(np.mean([r["axis_proj45"] for r in pools[u]]))
                                     for u in ("U03_mond_scalar", "U05_tensor_axis")},
        "amplitude_dependence": {str(r["amp"]): {"median_err_deg": r["axis_err"],
                                                 "R": r["axis_R"], "proj": r["axis_proj"],
                                                 "proj_45deg": r["axis_proj45"]}
                                 for r in scan_out["U05_tensor_axis"]["rows"]},
        "note": ("the concentration R is invariant under a global rotation of the "
                 "assumed axis; the PROJECTION is not, and it is the projection "
                 "that collapses when the axis is misspecified."),
    }

    nn = np.concatenate([[r["detectors"]["network"] for r in pools[a]]
                         for a in ("U03_mond_scalar", "U10_systematics",
                                   "U02_cdm", "H0_scalar_null")])
    cal_n = st.detector_calibration(nn[::2])
    q["Q4_network_vs_ellipticity"] = {
        "detector": ("slope of the shear residual on the member-derived well-strength "
                     "map, MINUS the same slope for an angle-scrambled member catalogue "
                     "(mass and clustercentric radius preserved exactly)"),
        "critical_value": cal_n,
        "audit_false_positive_rate": rate(nn[1::2], cal_n["crit"]),
        "rate_on_U10_systematics_baryon_ellipticity": rate(
            [r["detectors"]["network"] for r in pools["U10_systematics"]], cal_n["crit"]),
        "rate_on_U02_triaxial_halo": rate(
            [r["detectors"]["network"] for r in pools["U02_cdm"]], cal_n["crit"]),
        "power_on_U06_wellnet": rate(
            [r["detectors"]["network"] for r in pools["U06_wellnet"]], cal_n["crit"]),
        "mean_detector_vs_amplitude": {str(r["amp"]): r["detectors"]["network"]
                                       for r in scan_out["U06_wellnet"]["rows"]},
        "responsiveness_detector_vs_B": st.responsiveness(
            [r["amp"] for r in scan_out["U06_wellnet"]["rows"]],
            [r["detectors"]["network"] for r in scan_out["U06_wellnet"]["rows"]]),
    }

    npa = np.concatenate([[r["detectors"]["path"] for r in pools[a]]
                          for a in ("U03_mond_scalar", "U10_systematics", "H0_scalar_null")])
    cal_p = st.detector_calibration(npa[::2])
    q["Q5_path_after_systematics"] = {
        "detector": "slope of the supernova Hubble residual on the path void fraction",
        "critical_value": cal_p,
        "audit_false_positive_rate": rate(npa[1::2], cal_p["crit"]),
        "rate_on_U10_systematics": rate(
            [r["detectors"]["path"] for r in pools["U10_systematics"]], cal_p["crit"]),
        "power_on_U09_path": rate(
            [r["detectors"]["path"] for r in pools["U09_path_redshift"]], cal_p["crit"]),
        "time_dilation_check": {u: float(np.mean([r["features"]["sn_dur"] for r in pools[u]]))
                                for u in ("U03_mond_scalar", "U09_path_redshift")},
        "responsiveness_detector_vs_eps": st.responsiveness(
            [r["amp"] for r in scan_out["U09_path_redshift"]["rows"]],
            [r["detectors"]["path"] for r in scan_out["U09_path_redshift"]["rows"]]),
    }

    DETS = ("aniso_ext", "aniso_ext_minus_bar", "gal_aniso", "network",
            "memory", "ep_slip", "path", "env")
    cal_arms = ("U01_baryons_newton", "U03_mond_scalar", "U10_systematics",
                "H0_scalar_null")
    q6 = {"calibration_arms": list(cal_arms),
          "critical_values_from": "first half of each calibration arm's pool",
          "per_detector": {}}
    fired_cdm = fired_null = fired_hard = None
    for d in DETS:
        cv = np.concatenate([[r["detectors"][d] for r in pools[a][:H]] for a in cal_arms])
        cal = st.detector_calibration(cv)
        aud = np.concatenate([[r["detectors"][d] for r in pools[a][H:]] for a in cal_arms])
        cdm = np.abs([r["detectors"][d] for r in pools["U02_cdm"]])
        cdh = np.abs([r["detectors"][d] for r in pools["U02_cdm_hard"]])
        q6["per_detector"][d] = {
            "critical_value": cal,
            "audit_fp_on_calibration_families": rate(aud, cal["crit"]),
            "fp_on_U02_cdm": rate(cdm, cal["crit"]),
            "fp_on_U02_cdm_with_3x_systematics": rate(cdh, cal["crit"])}
        fa, fn, fh = cdm >= cal["crit"], np.abs(aud) >= cal["crit"], cdh >= cal["crit"]
        fired_cdm = fa if fired_cdm is None else (fired_cdm | fa)
        fired_null = fn if fired_null is None else (fired_null | fn)
        fired_hard = fh if fired_hard is None else (fired_hard | fh)
    q6["family_wise_any_detector"] = {
        "on_calibration_audit": st.rate_with_ci(int(fired_null.sum()), len(fired_null)),
        "on_U02_cdm": st.rate_with_ci(int(fired_cdm.sum()), len(fired_cdm)),
        "on_U02_cdm_with_3x_systematics": st.rate_with_ci(int(fired_hard.sum()),
                                                          len(fired_hard)),
        "note": (f"{len(DETS)} detectors at a nominal 0.05 each; the family-wise rate "
                 "is the programme-level multiplicity the brief warns about.")}
    q["Q6_false_new_gravity_in_dark_matter"] = q6

    q["Q7_amplitude_thresholds"] = {
        u: {"knob": v["knob"], "fiducial": v["fiducial"], "z3": v["z3"], "z5": v["z5"],
            "z_crit_family": v["z_crit"], "responsiveness": v["responsiveness_z"]}
        for u, v in scan_out.items()}
    save("E4_questions.json", q)

    # =================================================================
    # E5  admissibility gates
    # =================================================================
    print("[E5] admissibility gates", flush=True)
    cg = [sc.coarse_grain_check(g.clu, seed=i) for i, g in enumerate(lib.geoms[:8])]
    rc = [sc.reciprocity_check(g.clu) for g in lib.geoms[:8]]
    gauge = {}
    for mult, tag in ((3.0, "3xR500 (declared primary)"), (1.5, "1.5xR500")):
        vals = []
        for gm in lib.geoms:
            c = gm.clu
            rgg = np.geomspace(5.0, mult * c.R500, 200)
            gg = c.gN(rgg)
            cum = np.concatenate(([0.0], np.cumsum(0.5 * (gg[1:] + gg[:-1]) * np.diff(rgg))))
            vals.append(float(np.interp(0.7 * c.R500, rgg, cum[-1] - cum)))
        gauge[tag] = vals
    a = np.log10(gauge["3xR500 (declared primary)"]); b = np.log10(gauge["1.5xR500"])
    e5 = {"coarse_graining": {"per_cluster": cg,
                              "worst_max_rel_change": float(max(c["max_rel_change"] for c in cg)),
                              "threshold": 0.02,
                              "pass": bool(max(c["max_rel_change"] for c in cg) < 0.02)},
          "reciprocity": {"per_cluster": rc,
                          "worst": float(max(r["max_abs_asymmetry_over_maxF"] for r in rc)),
                          "pass": bool(max(r["max_abs_asymmetry_over_maxF"] for r in rc) < 1e-12)},
          "potential_gauge": {"rule_values_dex": {k: np.log10(v).tolist() for k, v in gauge.items()},
                              "mean_offset_dex": float(np.mean(a - b)),
                              "spread_of_offset_dex": float(np.std(a - b)),
                              "rank_correlation_between_rules":
                                  float(np.corrcoef(np.argsort(np.argsort(a)),
                                                    np.argsort(np.argsort(b)))[0, 1]),
                              "note": ("the two admissible boundary rules differ by a "
                                       "near-constant offset and preserve the ordering, so "
                                       "the depth VARIABLE is rule-robust even though its "
                                       "zero point is convention.")}}
    save("E5_gates.json", e5)

    # =================================================================
    # E6  the missing observation
    # =================================================================
    print("[E6] missing-observation oracle", flush=True)
    indist = [r for r in pair_rows if r["z_max"] < z_crit_family]
    z95 = {k: v["z95"] for k, v in e0["per_test_null"].items()}
    N_ORACLE = max(240, N_AMP // 2)
    e6 = {"z_crit_family": z_crit_family, "per_test_null_z95": z95, "pairs": []}
    ORACLES = (("noise x0.25: 16x the effective source density and 4x the "
                "spectroscopic / temperature precision", 1.0, 0.25, 30, 12),
               ("systematics x0.25: shear calibration, photo-z, M/L, inclination, "
                "miscentring and non-thermal pressure all 4x better controlled",
                0.25, 1.0, 30, 12),
               ("survey x1.5: 45 galaxies and 18 clusters per corpus",
                1.0, 1.0, 45, 18))
    for k, r in enumerate(indist):
        a_, b_ = r["a"], r["b"]
        chz = {t: v["z"] for t, v in r["per_test"].items()}
        ex = {t: chz[t] - z95.get(t, 2.0) for t in chz if t != "full"}
        best = max(ex, key=ex.get) if ex else None
        row = {"a": a_, "b": b_, "z_max": r["z_max"], "best_test": r["best_test"],
               "auc_full": r["auc_full"], "test_z": chz,
               "best_channel": best, "best_channel_z": chz.get(best, 0.0),
               "best_channel_null_z95": z95.get(best),
               "best_channel_clears_its_null": bool(ex.get(best, -1) > 0),
               "corpus_multiplier_for_z5_sqrtN":
                   float((5.0 / max(r["z_max"], 1e-3)) ** 2) if r["z_max"] > 0.05 else None,
               "sqrtN_assumption": ("z scales as sqrt(number of independent objects) "
                                    "at fixed per-object precision"),
               "oracles": {}}
        for tag, ss, ns, ng_, nc_ in ORACLES:
            recs = {}
            for u in (a_, b_):
                s2 = ss * (3.0 if u == "U10_systematics" else 1.0)
                gn.N_GAL, gn.N_CLU = ng_, nc_
                recs[u] = pool(f"orc_{u}_{ss}_{ns}_{ng_}_{nc_}", (u, None, s2, ns),
                               N_ORACLE, SEED0 + 900000 + 137 * UIDS.index(u))
                gn.N_GAL, gn.N_CLU = 30, 12
            Ac, Aa = quarters(recs[a_]); Bc, Ba = quarters(recs[b_])
            rr = sepmax(Ac, Bc, Aa, Ba, seed=51 + k)
            row["oracles"][tag] = {"z_max": rr["z_max"], "best_test": rr["best_test"],
                                   "n_cal": len(Ac), "n_aud": len(Aa)}
        e6["pairs"].append(row)
        print(f"  {a_} vs {b_}: z_max={r['z_max']:.2f} best {r['best_test']}", flush=True)
    save("E6_missing_observations.json", e6)

    # =================================================================
    # E7  knob -> observable amplitude
    # =================================================================
    print("[E7] knob -> observable amplitude", flush=True)
    e7 = {}
    for uid, amps in SCAN.items():
        rows = []
        for amp in amps:
            dg, quad = [], []
            for gm in lib.geoms[:10]:
                c = gm.clu; rg = gm.rg
                u3 = ph.draw_universe("U03_mond_scalar", np.random.default_rng(5))
                ux = ph.draw_universe(uid, np.random.default_rng(5), knob=amp)
                F3 = ph.cluster_field(u3, c, rg)
                Fx = ph.cluster_field(ux, c, rg)
                gx = Fx["g_m"].copy()
                if uid == "U06_wellnet":
                    gx = gx + amp * np.gradient(gm.Ex_bar, rg)
                sel = (rg > 0.2 * c.R500) & (rg < 2.0 * c.R500)
                dg.append(float(np.max(np.abs(gx[sel] / F3["g_m"][sel] - 1.0))))
                if Fx["chi"] is not None:
                    Phi = ph.cluster_potential_1d(rg, Fx["g_l"])
                    quad.append(float(np.max(np.abs(Fx["chi"][sel] / Phi[sel]))))
            rows.append({"amp": amp,
                         "max_fractional_acceleration_change_vs_U03": float(np.median(dg)),
                         "max_potential_quadrupole_fraction":
                             float(np.median(quad)) if quad else None})
        e7[uid] = {"knob": ph.KNOB[uid], "rows": rows}
    e7["_note"] = ("median over 10 library clusters of the maximum over 0.2-2.0 R500; "
                   "a null from a detector with no power below the predicted amplitude "
                   "says nothing.")
    save("E7_observable_amplitudes.json", e7)

    gn.close_pool()
    manifest["elapsed_s"] = time.time() - t00
    manifest["file_access"] = pv.stop_ledger()
    save("run_manifest.json", manifest)
    print(f"[done] {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
