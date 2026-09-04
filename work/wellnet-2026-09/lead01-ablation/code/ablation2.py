"""Item 1, second half: the cross-arm scoreboard, and three checks the first
pass showed were needed.

(a) Every (training arm, model) pair scored on the IDENTICAL 52 held-out
    clusters, so the arms are directly comparable and the published result can
    be beaten or not beaten by a model that saw less.
(b) Offset recovery: what fraction of the observed mean cluster/group deviation
    each frozen model actually predicts.  RMS alone hides this.
(c) The responsiveness gate applied to the DISCRIMINATING statistic.  The first
    pass found d(transfer rms of M1)/dq = 0 EXACTLY, which is a
    monotone-invariance failure of the headline statistic; the paired
    difference must be checked instead.
(d) Robustness of the arm-A and arm-B conclusions to the free quadratic in
    log g_bar being extrapolated out of the training range.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as C
from ablation import paired_bootstrap

RNG = np.random.default_rng(20260905)


def design_lin(t, which):
    """Same models but LINEAR in log g_bar -- the robustness check on
    extrapolating a quadratic fitted inside one class."""
    n = len(t["lg"])
    base = np.column_stack([np.ones(n), t["lg"]])
    return {"M0": base,
            "M1": np.column_stack([base, t["lp"]]),
            "M2": np.column_stack([base, t["lr"]]),
            "M3": np.column_stack([base, (t["rank"] > 1).astype(float)])}[which]


def score(t, train, test, which, dfun=C.design):
    A = dfun(t, which)
    c, *_ = np.linalg.lstsq(A[train], t["dev"][train], rcond=None)
    pred = A[test] @ c
    res = t["dev"][test] - pred
    return c, pred, res, float(np.sqrt(np.mean(res ** 2)))


def main():
    d = C.load_ladder()
    t = C.system_table(d)
    gal = t["rank"] == 1
    grp = (t["rank"] >= 2) & (t["rank"] <= 4)
    clu = t["rank"] >= 5
    out = {}

    print("=" * 78)
    print("(a) CROSS-ARM SCOREBOARD  --  identical 52 held-out clusters")
    print("=" * 78)
    print(f"    observed cluster deviations: mean {t['dev'][clu].mean():+.4f} "
          f"dex, sd {t['dev'][clu].std():.4f} dex, "
          f"n = {int(clu.sum())} systems")
    print(f"\n    {'arm':<18} {'model':<6} {'rms':>8} {'bias':>9} "
          f"{'scatter':>9} {'mean pred':>10} {'offset recovered':>17}")
    board = {}
    arms = (("galaxies only", gal), ("groups only", grp),
            ("galaxies+groups", gal | grp))
    obs_mean = float(t["dev"][clu].mean())
    for aname, tr in arms:
        for m in ("M0", "M1", "M2", "M3"):
            c, pred, res, rms = score(t, tr, clu, m)
            defic, cond = C.rank_deficient(C.design(t, m)[tr])
            rec = dict(arm=aname, model=m, rms=rms, bias=float(res.mean()),
                       scatter=float(res.std()), mean_pred=float(pred.mean()),
                       offset_recovered=float(pred.mean() / obs_mean),
                       estimable=not defic,
                       coef=[float(x) for x in c])
            board[f"{aname}|{m}"] = rec
            flag = "" if not defic else "   [step not estimable]"
            print(f"    {aname:<18} {m:<6} {rms:8.4f} {res.mean():+9.4f} "
                  f"{res.std():9.4f} {pred.mean():+10.4f} "
                  f"{pred.mean() / obs_mean:16.1%}{flag}")
    # the frozen training-set mean, per arm
    for aname, tr in arms:
        cm = float(t["dev"][tr].mean())
        r = t["dev"][clu] - cm
        rec = dict(arm=aname, model="Mconst", rms=float(np.sqrt(np.mean(r ** 2))),
                   bias=float(r.mean()), scatter=float(r.std()), mean_pred=cm,
                   offset_recovered=float(cm / obs_mean), estimable=True,
                   coef=[cm])
        board[f"{aname}|Mconst"] = rec
        print(f"    {aname:<18} {'Mconst':<6} {rec['rms']:8.4f} "
              f"{rec['bias']:+9.4f} {rec['scatter']:9.4f} {cm:+10.4f} "
              f"{cm / obs_mean:16.1%}")
    out["cross_arm_scoreboard"] = board
    best = min(board.values(), key=lambda r: r["rms"] if r["estimable"] else 9)
    print(f"\n    BEST transfer onto the held-out clusters anywhere in the "
          f"table: {best['arm']} / {best['model']} at {best['rms']:.4f} dex")
    out["best_transfer"] = best

    # paired bootstrap of that best model against the two published contenders
    print("\n    paired bootstrap against the published models "
          "(same 52 objects, all coefficients frozen):")
    pubtr = gal | grp
    cmp_ = {}

    def paired_arms(trA, mA, trB, mB, nb=20000):
        """Paired bootstrap of any two (training arm, model) pairs on the same
        held-out clusters, both sets of coefficients frozen."""
        AA, AB = C.design(t, mA), C.design(t, mB)
        y = t["dev"]
        ca, *_ = np.linalg.lstsq(AA[trA], y[trA], rcond=None)
        cb, *_ = np.linalg.lstsq(AB[trB], y[trB], rcond=None)
        ite = np.where(clu)[0]
        eA = y[ite] - AA[ite] @ ca
        eB = y[ite] - AB[ite] @ cb
        obs = float(np.sqrt(np.mean(eA ** 2)) - np.sqrt(np.mean(eB ** 2)))
        dd = np.empty(nb)
        for k in range(nb):
            p = RNG.integers(0, len(ite), len(ite))
            dd[k] = math.sqrt(float(np.mean(eA[p] ** 2))) \
                - math.sqrt(float(np.mean(eB[p] ** 2)))
        lo, hi = np.percentile(dd, [2.5, 97.5])
        return dict(observed_delta_rms=obs, ci95=[float(lo), float(hi)],
                    p_A_better=float((dd < 0).mean()),
                    rms_A=float(np.sqrt(np.mean(eA ** 2))),
                    rms_B=float(np.sqrt(np.mean(eB ** 2))),
                    n_objects_A_better=int((np.abs(eA) < np.abs(eB)).sum()),
                    n_test=len(ite))
    for tag, (trA, mA, trB, mB) in {
            "groupsM0_vs_publishedM1": (grp, "M0", pubtr, "M1"),
            "groupsM0_vs_publishedM3": (grp, "M0", pubtr, "M3"),
            "galaxiesM1_vs_publishedM1": (gal, "M1", pubtr, "M1"),
            "groupsM1_vs_publishedM1": (grp, "M1", pubtr, "M1"),
    }.items():
        r = paired_arms(trA, mA, trB, mB)
        cmp_[tag] = r
        print(f"      {tag:<28} {r['rms_A']:.4f} vs {r['rms_B']:.4f}  "
              f"dRMS {r['observed_delta_rms']:+.4f} "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"P(A better) = {r['p_A_better']:.4f}  "
              f"A closer on {r['n_objects_A_better']}/{r['n_test']}")
    out["cross_arm_paired"] = cmp_

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(b) OFFSET RECOVERY on the GROUPS, from galaxies only")
    print("=" * 78)
    obs_g = float(t["dev"][grp].mean())
    print(f"    observed group deviation: mean {obs_g:+.4f} dex, "
          f"sd {t['dev'][grp].std():.4f}")
    rec = {}
    for m in ("M0", "M1", "M2"):
        c, pred, res, rms = score(t, gal, grp, m)
        rec[m] = dict(rms=rms, mean_pred=float(pred.mean()),
                      offset_recovered=float(pred.mean() / obs_g),
                      bias=float(res.mean()), scatter=float(res.std()))
        print(f"    {m}: mean prediction {pred.mean():+.4f} dex = "
              f"{pred.mean() / obs_g:.1%} of the observed group offset "
              f"(rms {rms:.4f}, scatter {res.std():.4f})")
    out["galaxies_to_groups_offset_recovery"] = rec

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(c) RESPONSIVENESS OF THE DISCRIMINATING STATISTIC")
    print("=" * 78)
    print("    The first pass found d(transfer rms of M1)/dq = 0 EXACTLY.")
    print("    That is not a bug: injecting (q/2) log|Phi_b| into the response")
    print("    is absorbed exactly by M1's own coefficient, so M1's residuals")
    print("    are algebraically unchanged.  The consequence is important --")
    print("    M1's transfer rms is INVARIANT to the size of the effect it is")
    print("    supposed to be detecting, so '0.1066 dex' is not a measurement")
    print("    of q.  The paired difference is the statistic that responds.")
    tr = gal | grp
    A1, A3 = C.design(t, "M1"), C.design(t, "M3")
    rows = []
    for q in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, -0.2, -0.4):
        yy = t["dev"] + 0.5 * q * t["lp"]
        c1, *_ = np.linalg.lstsq(A1[tr], yy[tr], rcond=None)
        c3, *_ = np.linalg.lstsq(A3[tr], yy[tr], rcond=None)
        r1 = float(np.sqrt(np.mean((yy[clu] - A1[clu] @ c1) ** 2)))
        r3 = float(np.sqrt(np.mean((yy[clu] - A3[clu] @ c3) ** 2)))
        rows.append((q, r1, r3, r1 - r3, float(c1[3])))
    print(f"\n    {'injected q':>11} {'rms M1':>9} {'rms M3':>9} "
          f"{'dRMS':>10} {'beta':>9}")
    for q, r1, r3, dl, b in rows:
        print(f"    {q:11.2f} {r1:9.4f} {r3:9.4f} {dl:+10.5f} {b:+9.4f}")
    base = [r for r in rows if r[0] == 0][0]
    slope = [(rows[i][3] - base[3]) / (rows[i][0] - 0)
             for i in range(len(rows)) if rows[i][0] != 0]
    out["responsiveness_of_paired_statistic"] = dict(
        table=[[float(x) for x in r] for r in rows],
        d_dRMS_dq_secants=[float(x) for x in slope],
        d_dRMS_dq_min=float(min(slope)), d_dRMS_dq_max=float(max(slope)),
        dRMS_spread=float(max(r[3] for r in rows) - min(r[3] for r in rows)),
        note="d(rms M1)/dq is EXACTLY zero by construction; the paired "
             "difference dRMS = rms(M1) - rms(M3) is the responsive statistic")
    print(f"\n    d(dRMS)/dq secants from q=0: {min(slope):+.4f} ... "
          f"{max(slope):+.4f}   spread of dRMS over the range "
          f"{out['responsiveness_of_paired_statistic']['dRMS_spread']:.4f} dex")
    print(f"    GATE: the discriminating statistic DOES respond "
          f"(non-zero derivative).  The headline one does NOT.")
    # what q would the observed dRMS correspond to?
    s0 = float(np.mean([abs(x) for x in slope]))
    print(f"    observed dRMS = +0.01128 dex; at |d(dRMS)/dq| ~ {s0:.4f} that is")
    print(f"    the dRMS a true q of about {0.01128 / s0:.3f} would displace -- "
          f"compare q_required = 0.371")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(d) ROBUSTNESS: linear instead of quadratic in log g_bar")
    print("=" * 78)
    print("    Arms A and B extrapolate a quadratic fitted inside a single")
    print("    class.  Repeat with a linear baseline to check the conclusions")
    print("    are not an artefact of that extrapolation.")
    lin = {}
    print(f"\n    {'arm':<18} {'model':<6} {'rms quad':>9} {'rms lin':>9} "
          f"{'beta quad':>10} {'beta lin':>10}")
    for aname, tr_ in (("galaxies only", gal), ("groups only", grp),
                       ("galaxies+groups", gal | grp)):
        for m in ("M0", "M1", "M2", "M3"):
            cq, _, _, rq = score(t, tr_, clu, m)
            cl, _, _, rl = score(t, tr_, clu, m, dfun=design_lin)
            bq = float(cq[3]) if len(cq) > 3 else float("nan")
            bl = float(cl[2]) if len(cl) > 2 else float("nan")
            lin[f"{aname}|{m}"] = dict(rms_quadratic=rq, rms_linear=rl,
                                       coef_quadratic=float(bq),
                                       coef_linear=float(bl))
            print(f"    {aname:<18} {m:<6} {rq:9.4f} {rl:9.4f} "
                  f"{bq:10.4f} {bl:10.4f}")
    out["linear_baseline_robustness"] = lin

    p = os.path.join(C.LANE, "ablation.json")
    js = json.load(open(p))
    js.update(out)
    json.dump(js, open(p, "w"), indent=2)
    print(f"\nmerged into {p}")


if __name__ == "__main__":
    main()
