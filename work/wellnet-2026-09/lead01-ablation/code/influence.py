"""Item 1, third pass: which part of the training set actually carries the
published transfer, plus two housekeeping checks the brief requires.

(a) Leave-one-rung-out of the arm-C training set.  If dropping the 29 rich
    groups destroys the transfer, the published result is carried by the
    groups; if dropping the 164 galaxies destroys it, by the galaxies.
(b) Non-monotone M_b(<r): the brief lists it as a known failure mode, and the
    baryonic half-mass radius used by the PRIMARY boundary rule depends on it.
(c) Sealed-holdout audit: confirm by string search that no KiDS or wide-binary
    row is anywhere in the inputs this lane reads.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as C
from boundary import build_profiles


def main():
    d = C.load_ladder()
    t = C.system_table(d)
    gal = t["rank"] == 1
    grp = (t["rank"] >= 2) & (t["rank"] <= 4)
    clu = t["rank"] >= 5
    base = gal | grp
    out = {}

    print("=" * 78)
    print("(a) LEAVE-ONE-RUNG-OUT of the published training set")
    print("=" * 78)
    full = {}
    for m in ("M0", "M1", "M3"):
        _, _, r = C.fit_freeze_eval(t, base, clu, m)
        full[m] = r
    print(f"    full training set (rungs 1-4, 200 systems): "
          + ", ".join(f"{m} {full[m]:.4f}" for m in ("M0", "M1", "M3")))
    print(f"\n    {'dropped from training':<26} {'n':>4} "
          f"{'M0':>8} {'M1':>8} {'M3':>8} {'beta':>9} {'step':>8}")
    loo = {}
    for k in sorted(set(t["rank"][base].tolist())):
        keep = base & (t["rank"] != k)
        rec = dict(n_train=int(keep.sum()))
        for m in ("M0", "M1", "M3"):
            c, _, r = C.fit_freeze_eval(t, keep, clu, m)
            rec[m] = r
            if m == "M1":
                rec["beta"] = float(c[3])
            if m == "M3":
                defic, _ = C.rank_deficient(C.design(t, m)[keep])
                rec["step"] = float(c[3])
                rec["step_estimable"] = not defic
        loo[f"rung_{k}"] = rec
        st = f"{rec['step']:+8.4f}" if rec["step_estimable"] else "     n/a"
        print(f"    rung {k} ({int((t['rank'] == k).sum()):3d} systems)"
              f"{'':<8} {rec['n_train']:4d} {rec['M0']:8.4f} {rec['M1']:8.4f} "
              f"{rec['M3']:8.4f} {rec['beta']:+9.4f} {st}")
    # drop ALL groups at once = arm A; drop all galaxies = arm B (already done)
    out["leave_one_rung_out"] = loo
    out["full_training"] = full

    # how much does each rung shift beta?
    bs = [loo[k]["beta"] for k in loo]
    print(f"\n    beta across the leave-one-rung-out refits: "
          f"{min(bs):+.4f} ... {max(bs):+.4f}  (full set {0.17188:+.4f})")
    out["beta_leave_one_rung_out_range"] = [float(min(bs)), float(max(bs))]

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(b) NON-MONOTONE BARYONIC MASS PROFILES")
    print("=" * 78)
    profs, _ = build_profiles(d)
    nonmono, worst = 0, 0.0
    nres = 0
    for s, p in profs.items():
        if p.kind != "resolved":
            continue
        nres += 1
        m = p.Mgrid
        dec = np.minimum.accumulate(m[::-1])[::-1]
        frac = float(np.max((m - dec) / m)) if len(m) > 1 else 0.0
        if np.any(np.diff(m) < 0):
            nonmono += 1
            worst = max(worst, frac)
    print(f"    {nonmono} of {nres} resolved systems have a non-monotone "
          f"M_eff(<r) = g_bar r^2/G")
    print(f"    worst local decrease: {worst:.1%} of the running value")
    print("    Cause (already identified in Run R): these are SPARC DISK rows,")
    print("    where g_bar is the razor-thin-disk field rather than GM(<r)/r^2")
    print("    and the signed V_gas from central HI holes makes V_b^2 r/G fall.")
    print("    Consequence here: the baryonic half-mass radius used by the")
    print("    PRIMARY boundary rule is defined as the FIRST crossing of half")
    print("    the outermost value, which is well defined regardless.")
    # sensitivity: last crossing instead of first
    diff = []
    for s, p in profs.items():
        if p.kind != "resolved":
            continue
        m, r = p.Mgrid, p.r
        tgt = 0.5 * m[-1]
        above = m >= tgt
        # every upward crossing of the half-mass level
        cross = [j for j in range(1, len(m)) if above[j] and not above[j - 1]]
        if above[0]:
            cross = [0] + cross
        if len(cross) > 1:
            diff.append(abs(math.log10(r[cross[-1]] / r[cross[0]])))
    print(f"    systems where a last-crossing definition would differ: "
          f"{len(diff)}; median |dlog r_half| = "
          f"{np.median(diff) if diff else 0:.4f} dex")
    out["non_monotone_Mb"] = dict(
        n_resolved=nres, n_non_monotone=nonmono,
        worst_local_decrease_fraction=worst,
        n_systems_r_half_definition_matters=len(diff),
        median_dlog_r_half_dex=float(np.median(diff)) if diff else 0.0)

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(c) SEALED-HOLDOUT AUDIT")
    print("=" * 78)
    bad = ("kids", "wide_binary", "wide binary", "widebinary", "gaia_wb")
    hits = 0
    for col in ("system", "cls", "source", "probe"):
        for v in set(d[col].tolist()):
            if any(b in str(v).lower() for b in bad):
                print(f"    *** SEALED HOLDOUT STRING FOUND: {col} = {v}")
                hits += 1
    print(f"    scanned every distinct value of system / class / source / probe "
          f"in the ladder: {hits} sealed-holdout matches")
    print(f"    probes present: {sorted(set(d['probe'].tolist()))}")
    print(f"    sources present: {sorted(set(d['source'].tolist()))}")
    out["sealed_holdout_audit"] = dict(
        matches=hits, probes=sorted(set(d["probe"].tolist())),
        sources=sorted(set(d["source"].tolist())))
    assert hits == 0

    p = os.path.join(C.LANE, "ablation.json")
    js = json.load(open(p))
    js["influence"] = out
    json.dump(js, open(p, "w"), indent=2)
    print(f"\nmerged into {p}")


if __name__ == "__main__":
    main()
