"""baseline.py -- every pre-existing compiler verdict, snapshotted BEFORE the
two-body probe is added and compared field by field AFTER (BL's discipline:
the extension must be additive -- 29 prior verdicts unchanged).

    python baseline.py --snapshot     # run on the UNPATCHED compiler
    python baseline.py --compare      # run on the patched one
"""
from __future__ import annotations

import json
import os
import sys
import time

import guard
import compiler as C

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SNAP = os.path.join(RES, "baseline_prepatch.json")


def verdicts() -> dict:
    now = {}

    def rec(tag, r):
        g1 = r["gate1_constant_K"][1]
        now[tag] = dict(verdict=r["_verdict"], failed=r["_failed"],
                        labels=r["_labels"], flags=r["_flags"],
                        primary=r["_taxonomy"]["primary"],
                        defects=[d["code"] for d in r["_taxonomy"]["defects"]],
                        g4_asym=r[C.GATE4][1].get("asymmetry"),
                        g1_max_single=g1.get("max_single_probe_resid_dex"),
                        g1_joint=g1.get("joint_resid_dex"),
                        g1_escapes=g1.get("escapes"))
    for k, c in C.known_families().items():
        rec("KF:" + k, C.check(c))
    for k, c in C.external_axis_elements().items():
        rec("EA:" + k, C.check(c))
    xc = C.run_external_controls(cheap=True)
    for k, v in xc["rows"].items():
        now["XC:" + k] = dict(verdict=v["verdict"], required=v["required"],
                              agrees=v["agrees"], bin=v["taxonomy_bin"],
                              failed=v["failed"])
    now["_external_controls_all_agree"] = bool(xc["all_agree"])
    now["_n_external_controls"] = int(xc["n"])
    now["_n_agree"] = int(xc["n_agree"])
    return now


def main():
    guard.arm()
    t0 = time.perf_counter()
    if "--snapshot" in sys.argv:
        v = verdicts()
        v["_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        v["_probes"] = sorted(C.probes().keys())
        with open(SNAP, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(v, fh, indent=1, default=float)
        print(f"snapshot: {len([k for k in v if not k.startswith('_')])} verdicts, "
              f"probes {v['_probes']}, controls {v['_n_agree']}/{v['_n_external_controls']} "
              f"[{time.perf_counter() - t0:.0f}s]")
        return
    base = json.load(open(SNAP, encoding="utf-8"))
    now = verdicts()
    keys = [k for k in base if not k.startswith("_")]
    diffs = {k: dict(before=base[k], after=now.get(k)) for k in keys if base[k] != now.get(k)}
    out = dict(n_compared=len(keys), n_changed=len(diffs), changed=diffs,
               external_controls_all_agree=now["_external_controls_all_agree"],
               n_external_controls=now["_n_external_controls"],
               n_agree=now["_n_agree"],
               probes_before=base["_probes"], probes_after=sorted(C.probes().keys()),
               statement=("the two-body probe is additive: no pre-existing verdict, "
                          "failed-gate list, taxonomy bin, defect list, escape list "
                          "or measured statistic changed" if not diffs else
                          "PRIOR VERDICTS CHANGED -- see `changed`"),
               wall_seconds=time.perf_counter() - t0)
    with open(os.path.join(RES, "baseline_compare.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"compare: {out['n_compared']} verdicts, {out['n_changed']} changed; "
          f"controls {out['n_agree']}/{out['n_external_controls']}; "
          f"probes {out['probes_after']} [{out['wall_seconds']:.0f}s]")
    if diffs:
        for k, d in diffs.items():
            print("  CHANGED", k, d)


if __name__ == "__main__":
    main()
