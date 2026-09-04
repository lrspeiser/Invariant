"""Items 1 and 2 of the lane.

1  THE ABLATION.  Decompose "never saw a cluster" into
       A  fit galaxies ONLY      -> predict groups AND clusters
       B  fit groups ONLY        -> predict clusters
       C  fit galaxies + groups  -> predict clusters      (the published result)
   Same freeze-then-evaluate-once discipline in every arm.

2  THE PAIRED OBJECT BOOTSTRAP.  Is the potential law meaningfully worse than
   the class step on the held-out clusters, statistically indistinguishable, or
   sometimes better?  Paired difference distribution, not two point estimates.
   Plus: can the two be distinguished AT ALL by construction?
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as C

RNG = np.random.default_rng(20260904)
NB = 20000


def arm_report(t, train, test, tag, note=""):
    """Fit every model on `train`, freeze, evaluate once on `test`."""
    out = dict(tag=tag, note=note,
               n_train=int(train.sum()), n_test=int(test.sum()),
               train_ranks=sorted(set(t["rank"][train].tolist())),
               test_ranks=sorted(set(t["rank"][test].tolist())))
    out["train_lp_range"] = [float(t["lp"][train].min()),
                             float(t["lp"][train].max())]
    out["test_lp_range"] = [float(t["lp"][test].min()),
                            float(t["lp"][test].max())]
    out["train_lg_range"] = [float(t["lg"][train].min()),
                             float(t["lg"][train].max())]
    out["test_lg_range"] = [float(t["lg"][test].min()),
                            float(t["lg"][test].max())]
    out["test_dev_mean"] = float(t["dev"][test].mean())
    out["test_dev_sd"] = float(t["dev"][test].std())
    out["models"] = {}
    res_store = {}
    for m in ("M0", "M1", "M2", "M3"):
        A = C.design(t, m)
        deficient, cond = C.rank_deficient(A[train])
        c, res, rms = C.fit_freeze_eval(t, train, test, m)
        res_store[m] = res
        rec = dict(rms=rms, bias=float(res.mean()), scatter=float(res.std()),
                   coef=[float(x) for x in c],
                   design_rank_deficient_on_train=deficient,
                   smallest_singular_value_ratio=cond)
        if m == "M1":
            rec["beta"] = float(c[3])
            rec["q_implied"] = float(2 * c[3])
        if m == "M2":
            rec["gamma"] = float(c[3])
        if m == "M3":
            rec["step_dex"] = float(c[3])
        out["models"][m] = rec
    # skill relative to the RAR-only baseline in the SAME arm
    r0 = out["models"]["M0"]["rms"]
    for m in ("M1", "M2", "M3"):
        out["models"][m]["skill_vs_M0"] = float(1 - out["models"][m]["rms"] / r0)
    # a constant predictor: the training-set mean deviation, frozen
    cmean = float(t["dev"][train].mean())
    rc = t["dev"][test] - cmean
    out["models"]["Mconst"] = dict(
        rms=float(np.sqrt(np.mean(rc ** 2))), bias=float(rc.mean()),
        scatter=float(rc.std()), coef=[cmean],
        note="frozen training-set mean deviation, zero free structure")
    res_store["Mconst"] = rc
    return out, res_store


def paired_bootstrap(t, train, test, mA, mB, nb=NB, nested=False):
    """Paired object bootstrap of RMS(mA) - RMS(mB) on the held-out set.

    Coefficients FROZEN on `train` (nested=False), or refit on a resampled
    training set each draw (nested=True) so coefficient uncertainty enters too.
    Objects in the held-out set are resampled with replacement, and BOTH models
    are scored on the SAME resampled objects -- that is the pairing.
    """
    itr = np.where(train)[0]
    ite = np.where(test)[0]
    nte = len(ite)
    AA, AB = C.design(t, mA), C.design(t, mB)
    y = t["dev"]

    def frozen(idx_tr):
        ca, *_ = np.linalg.lstsq(AA[idx_tr], y[idx_tr], rcond=None)
        cb, *_ = np.linalg.lstsq(AB[idx_tr], y[idx_tr], rcond=None)
        return ca, cb

    ca0, cb0 = frozen(itr)
    eA0 = y[ite] - AA[ite] @ ca0
    eB0 = y[ite] - AB[ite] @ cb0
    obs = float(np.sqrt(np.mean(eA0 ** 2)) - np.sqrt(np.mean(eB0 ** 2)))

    d = np.empty(nb)
    for k in range(nb):
        pick = RNG.integers(0, nte, nte)
        if nested:
            ptr = itr[RNG.integers(0, len(itr), len(itr))]
            try:
                ca, cb = frozen(ptr)
            except np.linalg.LinAlgError:
                ca, cb = ca0, cb0
        else:
            ca, cb = ca0, cb0
        j = ite[pick]
        ea = y[j] - AA[j] @ ca
        eb = y[j] - AB[j] @ cb
        d[k] = math.sqrt(float(np.mean(ea ** 2))) - math.sqrt(float(np.mean(eb ** 2)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    # per-object paired statistics, coefficients frozen
    dsq = eA0 ** 2 - eB0 ** 2
    dabs = np.abs(eA0) - np.abs(eB0)
    # paired t on squared error
    se = dsq.std(ddof=1) / math.sqrt(nte)
    tstat = float(dsq.mean() / se) if se > 0 else float("nan")
    # exact sign test on |residual|
    nwin = int((dabs < 0).sum())
    nloss = int((dabs > 0).sum())
    ntot = nwin + nloss
    p_sign = 2 * min(
        sum(math.comb(ntot, i) for i in range(nwin + 1)) / 2 ** ntot,
        sum(math.comb(ntot, i) for i in range(nloss + 1)) / 2 ** ntot)
    return dict(
        model_A=mA, model_B=mB, n_test=nte, nested=nested, n_draws=nb,
        rms_A=float(np.sqrt(np.mean(eA0 ** 2))),
        rms_B=float(np.sqrt(np.mean(eB0 ** 2))),
        observed_delta_rms=obs,
        boot_mean=float(d.mean()), boot_sd=float(d.std()),
        ci95=[float(lo), float(hi)],
        p_A_better=float((d < 0).mean()),
        pct=[float(x) for x in np.percentile(d, [1, 5, 25, 50, 75, 95, 99])],
        paired_mean_sq_diff=float(dsq.mean()),
        paired_t_on_squared_error=tstat,
        n_objects_A_better=nwin, n_objects_B_better=nloss,
        sign_test_p=float(min(p_sign, 1.0)),
        mean_abs_resid_A=float(np.abs(eA0).mean()),
        mean_abs_resid_B=float(np.abs(eB0).mean()),
    )


def distinguishability(t, train, test):
    """Can the class step and the potential law be told apart AT ALL, by
    construction?  Compare their frozen predictions object by object."""
    A1, A3 = C.design(t, "M1"), C.design(t, "M3")
    y = t["dev"]
    itr = np.where(train)[0]
    c1, *_ = np.linalg.lstsq(A1[itr], y[itr], rcond=None)
    c3, *_ = np.linalg.lstsq(A3[itr], y[itr], rcond=None)
    p1, p3 = A1 @ c1, A3 @ c3
    dpred = p1 - p3
    out = dict(
        corr_pred_all=float(np.corrcoef(p1, p3)[0, 1]),
        max_abs_pred_diff_all=float(np.abs(dpred).max()),
        rms_pred_diff_all=float(np.sqrt(np.mean(dpred ** 2))),
        rms_pred_diff_test=float(np.sqrt(np.mean(dpred[test] ** 2))),
        max_abs_pred_diff_test=float(np.abs(dpred[test]).max()),
        median_measurement_error_test=float(np.median(t["e_dev"][test])),
        n_test_where_diff_exceeds_1sigma=int(
            (np.abs(dpred[test]) > t["e_dev"][test]).sum()),
        n_test=int(test.sum()),
    )
    # within the held-out clusters, how much does log|Phi_b| actually vary?
    out["test_lp_sd"] = float(t["lp"][test].std())
    out["test_lp_range"] = float(t["lp"][test].max() - t["lp"][test].min())
    out["beta"] = float(c1[3])
    out["M1_prediction_swing_across_test_lp_range"] = float(
        c1[3] * out["test_lp_range"])
    out["M3_prediction_swing_within_test"] = 0.0
    out["step_dex"] = float(c3[3])
    # the object where they disagree most
    j = int(np.argmax(np.abs(dpred)))
    out["most_discrepant_object"] = dict(
        name=str(t["name"][j]), rank=int(t["rank"][j]), lp=float(t["lp"][j]),
        lg=float(t["lg"][j]), pred_M1=float(p1[j]), pred_M3=float(p3[j]),
        diff=float(dpred[j]), observed=float(y[j]),
        measurement_error=float(t["e_dev"][j]))
    # per rung: mean |prediction difference|
    out["by_rank"] = {}
    for k in sorted(set(t["rank"].tolist())):
        m = t["rank"] == k
        out["by_rank"][str(k)] = dict(
            n=int(m.sum()), mean_abs_pred_diff=float(np.abs(dpred[m]).mean()),
            mean_pred_M1=float(p1[m].mean()), mean_pred_M3=float(p3[m].mean()),
            mean_observed=float(y[m].mean()))
    return out


def responsiveness_gate(t, train, test):
    """d(beta)/d(q) must not be zero -- the monotone-invariance failure mode.
    Inject (q/2) log|Phi_b| into the response, refit on the training set, and
    check the estimator moves.  Also check the TRANSFER metric responds."""
    A1 = C.design(t, "M1")
    itr, ite = np.where(train)[0], np.where(test)[0]
    rows = []
    for q in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        yy = t["dev"] + 0.5 * q * t["lp"]
        c, *_ = np.linalg.lstsq(A1[itr], yy[itr], rcond=None)
        rms = float(np.sqrt(np.mean((yy[ite] - A1[ite] @ c) ** 2)))
        rows.append((q, float(c[3]), rms))
    dq = [(rows[i + 1][1] - rows[i][1]) / (rows[i + 1][0] - rows[i][0])
          for i in range(len(rows) - 1)]
    return dict(injection=[[a, b, c_] for a, b, c_ in rows],
                dbeta_dq=[float(x) for x in dq],
                dbeta_dq_min=float(min(dq)), dbeta_dq_max=float(max(dq)),
                beta_spread=float(rows[-1][1] - rows[0][1]),
                transfer_rms_spread=float(max(r[2] for r in rows)
                                          - min(r[2] for r in rows)))


def shared_denominator_null(t, train, test, nmc=4000):
    """The shared-denominator null, SIMULATED with the actual error covariance,
    for the TRANSFER statistic rather than for beta.

    Under H0 the deviation depends on log g_bar only.  A coherent baryonic mass
    error moves log g_bar and log|Phi_b| by +delta and log nu by -delta; a
    distance error moves log g_bar by -2eps and log|Phi_b| by -eps.  So
    log|Phi_b| and the response share an error, and a transfer improvement can
    appear from nothing.  This measures how much.
    """
    d = C.load_ladder()
    win, _, _ = C.window_mask(d)
    # per-system coherent M_b error and distance error, matched to the ladder
    names = t["name"]
    idx = {s: np.where(win & (d["system"] == s))[0] for s in names}
    sysmb = np.array([d["sys_lg_Mb"][idx[s][0]] for s in names])
    egobs = np.array([float(np.median(d["e_lg_gobs"][idx[s]])
                            / math.sqrt(len(idx[s]))) for s in names])
    e_dist = 0.05
    n = len(names)
    A0d = C.design(t, "M0")
    c0, *_ = np.linalg.lstsq(A0d, t["dev"], rcond=None)
    sig = float(np.std(t["dev"] - A0d @ c0))
    out = {m: np.empty(nmc) for m in ("M0", "M1", "M2", "M3")}
    betas = np.empty(nmc)
    for k in range(nmc):
        dl = RNG.normal(0, 1, n) * sysmb
        dd = RNG.normal(0, 1, n) * e_dist
        tt = dict(t)
        tt["lg"] = t["lg"] + dl - 2 * dd
        tt["lp"] = t["lp"] + dl - dd
        tt["lr"] = t["lr"] + dd
        tt["dev"] = (A0d @ c0) + RNG.normal(0, sig, n) - dl \
            + RNG.normal(0, 1, n) * egobs
        for m in ("M0", "M1", "M2", "M3"):
            _, _, rms = C.fit_freeze_eval(tt, train, test, m)
            out[m][k] = rms
        c, *_ = np.linalg.lstsq(C.design(tt, "M1")[train],
                                tt["dev"][train], rcond=None)
        betas[k] = c[3]
    res = dict(n_draws=nmc, intrinsic_scatter_dex=sig,
               beta_null_mean=float(betas.mean()),
               beta_null_sd=float(betas.std()),
               beta_null_ci95=[float(x) for x in np.percentile(betas, [2.5, 97.5])])
    for m in out:
        res["transfer_null_" + m] = dict(
            mean=float(out[m].mean()), sd=float(out[m].std()),
            ci95=[float(x) for x in np.percentile(out[m], [2.5, 97.5])])
    res["transfer_null_delta_M1_minus_M3"] = dict(
        mean=float((out["M1"] - out["M3"]).mean()),
        sd=float((out["M1"] - out["M3"]).std()),
        ci95=[float(x) for x in np.percentile(out["M1"] - out["M3"],
                                              [2.5, 97.5])])
    return res


def main():
    print("=" * 78)
    print("lead01-ablation  --  items 1 and 2")
    print("=" * 78)
    d = C.load_ladder()
    t = C.system_table(d)
    print(f"\nladder sha256 {C.sha256(C.LADDER)}")
    print(f"window log g_bar = [{t['window'][0]:.4f}, {t['window'][1]:.4f}]")
    print(f"{len(t['lg'])} systems; by rung: "
          + ", ".join(f"{k}:{int((t['rank'] == k).sum())}"
                      for k in sorted(set(t["rank"].tolist()))))
    print("\nreproduction gate against the published Run R numbers:")
    gate = C.gate(t)

    gal = t["rank"] == 1
    grp = (t["rank"] >= 2) & (t["rank"] <= 4)
    clu = t["rank"] >= 5

    res = dict(gate=gate, ladder_sha256=C.sha256(C.LADDER),
               window=[float(t["window"][0]), float(t["window"][1])],
               n_systems=len(t["lg"]),
               by_rank={str(k): int((t["rank"] == k).sum())
                        for k in sorted(set(t["rank"].tolist()))},
               n_galaxies=int(gal.sum()), n_groups=int(grp.sum()),
               n_clusters=int(clu.sum()))

    # ---------------------------------------------------------------- item 1
    print("\n" + "=" * 78)
    print("ITEM 1  THE ABLATION")
    print("=" * 78)
    arms = {}
    resid = {}
    spec = [
        ("A_gal_to_grp", gal, grp, "fit galaxies ONLY -> predict GROUPS"),
        ("A_gal_to_clu", gal, clu, "fit galaxies ONLY -> predict CLUSTERS"),
        ("A_gal_to_grpclu", gal, grp | clu,
         "fit galaxies ONLY -> predict groups AND clusters"),
        ("B_grp_to_clu", grp, clu, "fit groups ONLY -> predict CLUSTERS"),
        ("C_galgrp_to_clu", gal | grp, clu,
         "fit galaxies + groups -> predict CLUSTERS  [the published result]"),
    ]
    for tag, tr, te, note in spec:
        a, r_ = arm_report(t, tr, te, tag, note)
        arms[tag] = a
        resid[tag] = r_
        print(f"\n{tag}: {note}")
        print(f"   train {a['n_train']} systems rungs {a['train_ranks']}, "
              f"test {a['n_test']} systems rungs {a['test_ranks']}")
        print(f"   train log|Phi_b| range {a['train_lp_range'][0]:.2f}..."
              f"{a['train_lp_range'][1]:.2f}   test "
              f"{a['test_lp_range'][0]:.2f}...{a['test_lp_range'][1]:.2f}")
        print(f"   held-out deviations: mean {a['test_dev_mean']:+.4f} dex, "
              f"sd {a['test_dev_sd']:.4f} dex")
        print(f"      {'model':<8} {'rms':>8} {'bias':>9} {'scatter':>9} "
              f"{'skill':>8}  extra coefficient")
        for m in ("Mconst", "M0", "M1", "M2", "M3"):
            r = a["models"][m]
            ex = ""
            if m == "M1":
                ex = f"beta = {r['beta']:+.4f}  -> q = {r['q_implied']:+.4f}"
            if m == "M2":
                ex = f"gamma = {r['gamma']:+.4f}"
            if m == "M3":
                ex = f"step = {r['step_dex']:+.4f} dex"
                if r["design_rank_deficient_on_train"]:
                    ex += "   *** NOT ESTIMABLE: the step is constant on the " \
                          "training set ***"
            sk = f"{r.get('skill_vs_M0', float('nan')):+8.3f}" \
                if "skill_vs_M0" in r else "       -"
            print(f"      {m:<8} {r['rms']:8.4f} {r['bias']:+9.4f} "
                  f"{r['scatter']:9.4f} {sk}  {ex}")
    res["ablation"] = arms

    # is the galaxies-only / groups-only transfer significantly better than the
    # RAR alone?  paired bootstrap of M1 against M0 in every arm.
    print("\n   paired bootstrap of M1 against M0 (RAR alone), per arm:")
    m1_vs_m0 = {}
    for tag, tr, te, note in spec:
        b = paired_bootstrap(t, tr, te, "M1", "M0")
        m1_vs_m0[tag] = b
        print(f"      {tag:<18} dRMS = {b['observed_delta_rms']:+.4f} "
              f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
              f"P(M1 better) = {b['p_A_better']:.4f}  "
              f"objects M1 better {b['n_objects_A_better']}/"
              f"{b['n_objects_A_better'] + b['n_objects_B_better']}")
    res["ablation_M1_vs_M0_paired"] = m1_vs_m0

    # how much of C's success is B's?  compare frozen predictions.
    A1 = C.design(t, "M1")
    yv = t["dev"]
    cA, *_ = np.linalg.lstsq(A1[gal], yv[gal], rcond=None)
    cB, *_ = np.linalg.lstsq(A1[grp], yv[grp], rcond=None)
    cC, *_ = np.linalg.lstsq(A1[gal | grp], yv[gal | grp], rcond=None)
    pA, pB, pC = A1[clu] @ cA, A1[clu] @ cB, A1[clu] @ cC
    res["ablation_prediction_agreement_on_clusters"] = dict(
        rms_B_minus_C=float(np.sqrt(np.mean((pB - pC) ** 2))),
        rms_A_minus_C=float(np.sqrt(np.mean((pA - pC) ** 2))),
        mean_B_minus_C=float(np.mean(pB - pC)),
        mean_A_minus_C=float(np.mean(pA - pC)),
        beta_A=float(cA[3]), beta_B=float(cB[3]), beta_C=float(cC[3]),
        note="how close is the groups-only frozen prediction to the published "
             "galaxies+groups one, object by object, on the held-out clusters")
    print("\n   Do galaxies+groups predict anything groups alone did not?")
    print(f"      beta: galaxies-only {cA[3]:+.4f}, groups-only {cB[3]:+.4f}, "
          f"galaxies+groups {cC[3]:+.4f}")
    print(f"      rms |pred_B - pred_C| on the held-out clusters = "
          f"{res['ablation_prediction_agreement_on_clusters']['rms_B_minus_C']:.4f} dex")
    print(f"      rms |pred_A - pred_C| on the held-out clusters = "
          f"{res['ablation_prediction_agreement_on_clusters']['rms_A_minus_C']:.4f} dex")

    # leverage available in each training set
    lev = {}
    for tag, m in (("galaxies", gal), ("groups", grp), ("galaxies+groups", gal | grp),
                   ("clusters", clu)):
        lev[tag] = dict(n=int(m.sum()), lp_sd=float(t["lp"][m].std()),
                        lp_range=float(t["lp"][m].max() - t["lp"][m].min()),
                        lg_sd=float(t["lg"][m].std()),
                        dev_mean=float(t["dev"][m].mean()),
                        dev_sd=float(t["dev"][m].std()))
    res["training_set_leverage"] = lev
    print("\n   leverage in log|Phi_b| available to each training set:")
    for k, v in lev.items():
        print(f"      {k:<18} n={v['n']:4d}  sd {v['lp_sd']:.3f}  range "
              f"{v['lp_range']:.3f} dex   mean deviation "
              f"{v['dev_mean']:+.4f} +- {v['dev_sd']:.4f}")
    # extrapolation factor: how far outside the training lp range do the
    # clusters sit, in units of the training spread
    for tag, m in (("A_gal", gal), ("B_grp", grp), ("C_galgrp", gal | grp)):
        gapmed = float(np.median(t["lp"][clu]) - np.median(t["lp"][m]))
        res["training_set_leverage"][tag + "_median_lp_gap_to_clusters"] = gapmed
        res["training_set_leverage"][tag + "_gap_in_training_sd_units"] = \
            float(gapmed / t["lp"][m].std())
        print(f"      clusters sit {gapmed:+.3f} dex from the median of {tag}"
              f"  = {gapmed / t['lp'][m].std():.2f} training sd")

    # ---------------------------------------------------------------- item 2
    print("\n" + "=" * 78)
    print("ITEM 2  THE PAIRED OBJECT BOOTSTRAP  --  M1 (potential) vs M3 (step)")
    print("=" * 78)
    tr, te = gal | grp, clu
    pb = paired_bootstrap(t, tr, te, "M1", "M3", nested=False)
    pbn = paired_bootstrap(t, tr, te, "M1", "M3", nested=True)
    res["paired_bootstrap"] = dict(frozen_coefficients=pb,
                                   nested_refit_on_resampled_training=pbn)
    for nm, b in (("coefficients frozen", pb),
                  ("nested (training resampled too)", pbn)):
        print(f"\n   {nm}, {b['n_draws']} draws, {b['n_test']} held-out clusters")
        print(f"      RMS  M1 = {b['rms_A']:.4f} dex,  M3 = {b['rms_B']:.4f} dex")
        print(f"      paired dRMS = RMS(M1) - RMS(M3) = "
              f"{b['observed_delta_rms']:+.5f} dex")
        print(f"      bootstrap    {b['boot_mean']:+.5f} +- {b['boot_sd']:.5f}"
              f"   95% [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]")
        print(f"      P(M1 better than M3) = {b['p_A_better']:.4f}")
        print(f"      percentiles 1/5/25/50/75/95/99: "
              + " ".join(f"{x:+.4f}" for x in b["pct"]))
        print(f"      per-object: M1 closer on {b['n_objects_A_better']} of "
              f"{b['n_objects_A_better'] + b['n_objects_B_better']} clusters, "
              f"sign-test p = {b['sign_test_p']:.4f}")
        print(f"      paired t on squared error = "
              f"{b['paired_t_on_squared_error']:+.3f}")

    # every pair of models, paired, for context
    print("\n   all pairwise paired comparisons on the same held-out clusters:")
    pw = {}
    for a in ("M0", "M1", "M2", "M3"):
        for b_ in ("M0", "M1", "M2", "M3"):
            if a >= b_:
                continue
            r = paired_bootstrap(t, tr, te, a, b_, nb=4000)
            pw[f"{a}_vs_{b_}"] = r
            print(f"      {a} vs {b_}: dRMS {r['observed_delta_rms']:+.4f} "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
                  f"P({a} better) = {r['p_A_better']:.3f}")
    res["pairwise"] = pw

    # ---- can they be distinguished at all? -------------------------------
    print("\n   CAN THE TWO BE DISTINGUISHED AT ALL, BY CONSTRUCTION?")
    dist = distinguishability(t, tr, te)
    res["distinguishability"] = dist
    print(f"      correlation of the two frozen predictions over all 252 "
          f"systems: {dist['corr_pred_all']:+.6f}")
    print(f"      rms |pred_M1 - pred_M3| over all systems  = "
          f"{dist['rms_pred_diff_all']:.4f} dex")
    print(f"      rms |pred_M1 - pred_M3| on the held-out clusters = "
          f"{dist['rms_pred_diff_test']:.4f} dex")
    print(f"      max |pred_M1 - pred_M3| on the held-out clusters = "
          f"{dist['max_abs_pred_diff_test']:.4f} dex")
    print(f"      median measurement error on the held-out deviations = "
          f"{dist['median_measurement_error_test']:.4f} dex")
    print(f"      clusters where the two predictions differ by more than that "
          f"object's own error: {dist['n_test_where_diff_exceeds_1sigma']} of "
          f"{dist['n_test']}")
    print(f"      inside the held-out clusters log|Phi_b| spans "
          f"{dist['test_lp_range']:.3f} dex (sd {dist['test_lp_sd']:.3f}), so "
          f"M1 swings {dist['M1_prediction_swing_across_test_lp_range']:.4f} "
          f"dex across them while M3 is flat by construction")
    print("      most discrepant object: "
          + json.dumps(dist["most_discrepant_object"]))
    print("      per rung, mean |pred_M1 - pred_M3|:")
    for k, v in dist["by_rank"].items():
        print(f"         rung {k}: n={v['n']:4d}  {v['mean_abs_pred_diff']:.4f} "
              f"dex   (M1 {v['mean_pred_M1']:+.4f}, M3 {v['mean_pred_M3']:+.4f}, "
              f"observed {v['mean_observed']:+.4f})")

    # ---- discipline gates ------------------------------------------------
    print("\n" + "=" * 78)
    print("DISCIPLINE GATES")
    print("=" * 78)
    rg = responsiveness_gate(t, tr, te)
    res["responsiveness_gate"] = rg
    print("   monotone-invariance / responsiveness gate, d(beta)/d(q):")
    for q, b_, r_ in rg["injection"]:
        print(f"      injected q = {q:.2f}  ->  recovered beta = {b_:+.5f}, "
              f"transfer rms = {r_:.4f}")
    print(f"      d(beta)/d(q) over the range: {rg['dbeta_dq_min']:.4f} ... "
          f"{rg['dbeta_dq_max']:.4f}   (must not be zero)")
    print(f"      beta spread {rg['beta_spread']:.4f}, transfer-rms spread "
          f"{rg['transfer_rms_spread']:.4f}")

    print("\n   shared-denominator null for the TRANSFER statistic, simulated")
    print("   with the actual error covariance (coherent M_b moves log g_bar")
    print("   and log|Phi_b| by +delta and log nu by -delta; distance moves")
    print("   log g_bar by -2eps and log|Phi_b| by -eps):")
    nul = shared_denominator_null(t, tr, te)
    res["shared_denominator_null"] = nul
    print(f"      beta under H0 = {nul['beta_null_mean']:+.4f} +- "
          f"{nul['beta_null_sd']:.4f}  (the naive assumption is 0)")
    for m in ("M0", "M1", "M2", "M3"):
        k = "transfer_null_" + m
        print(f"      transfer rms under H0, {m}: {nul[k]['mean']:.4f} +- "
              f"{nul[k]['sd']:.4f}")
    dd = nul["transfer_null_delta_M1_minus_M3"]
    print(f"      NULL EXPECTATION of dRMS(M1-M3) = {dd['mean']:+.5f} +- "
          f"{dd['sd']:.5f}   95% [{dd['ci95'][0]:+.5f}, {dd['ci95'][1]:+.5f}]")
    z = (pb["observed_delta_rms"] - dd["mean"]) / dd["sd"]
    res["paired_bootstrap"]["dRMS_z_against_shared_denominator_null"] = float(z)
    print(f"      observed dRMS {pb['observed_delta_rms']:+.5f}  ->  z = {z:+.2f}"
          f" against its own null")

    out = os.path.join(C.LANE, "ablation.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
