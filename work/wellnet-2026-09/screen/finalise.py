"""Post-run repair and stamping pass.

Three things happen here, all of them recorded in screen_results.json so the
file says what was recomputed and when:

 1. C4_wells_gsupp_p1 is re-run.  Its first pass hit a bug in two helper paths
    (the critical-coupling probe and the selective-refinement weight ratio)
    which called the family-C weight with no g_N field, so weight form 3 raised
    instead of returning a verdict.  The bug was in the SCREEN, not the
    candidate, so the candidate is re-run rather than being recorded as a fail.
 2. The family-D reciprocity-versus-alpha sweep is added.  D's default alpha
    makes K - I only a few parts in a thousand, so its reciprocity pass at that
    coupling says nothing about the structure; the sweep shows the violation
    appearing as alpha grows.
 3. SHA-256 of every source file, including the two gravitylab modules this
    lane imports and never modifies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import families as F
import screen as SC

PATH = "screen_results.json"


def main():
    doc = json.loads(Path(PATH).read_text(encoding="utf-8"))

    broken = [nm for nm, r in doc["candidates"].items()
              if any("error" in v for v in r["screens"].values())]
    missing = [nm for nm in SC.ALL if nm not in doc["candidates"]]
    print("candidates with a screen that raised:", broken, flush=True)
    print("candidates missing from the results file:", missing, flush=True)
    for nm in broken + missing:
        r = SC.run_screen(SC.ALL[nm], Nq=32768)
        doc["candidates"][nm] = r
        print(f"  re-ran {nm:24s} {r['verdict']:5s} failed={r['failed']}",
              flush=True)

    def stat_recip(c, wx, wm, pts, cloud):
        return SC.s10_reciprocity(c, n=32, Lbox=120.0)["value"]

    D = F.CANDIDATES["D1_pairs_p1_q1"]
    s = SC.sensitivity(D, "alpha", [0.3, 3.0, 30.0, 100.0, 300.0], stat_recip)
    s["monotone_invariance_guard"] = (
        "PASS: statistic moves with its parameter" if s["responds"]
        else "FAIL: statistic is bit-identical across the swept range")
    doc.setdefault("sensitivities", {})["D_reciprocity_vs_alpha"] = s
    print("D_reciprocity_vs_alpha:", [round(x, 6) for x in s["statistic"]],
          flush=True)

    # Re-apply the coarse-graining classifier to the STORED series.  The
    # criterion was tightened after the run (a drift that sits at 0.44 +- 0.05
    # across four decades of N must not be called convergent just because the
    # fitted exponent is 0.07), and the classification is a function of the
    # recorded numbers alone, so it can be recomputed without re-solving
    # anything.  The physics numbers are untouched.
    reclassified = {}
    for nm, r in doc["candidates"].items():
        s11 = r["screens"].get("S11_coarse_uniform", {})
        if "drift" not in s11:
            continue
        drift = {int(k): v for k, v in s11["drift"].items()}
        step = {int(k): v for k, v in s11.get("step", {}).items()}
        scales = {int(k): v for k, v in s11["partition_scale_kpc"].items()}
        L = SC.ALL[nm].prm.get("L", 10 * SC.KPC) / SC.KPC
        cls, nsafe = SC._classify(drift, step, scales, L,
                                  s11.get("rate_beta_step"))
        if cls != s11.get("classification"):
            reclassified[nm] = [s11.get("classification"), cls]
        s11["classification"], s11["N_safe"] = cls, nsafe
        s11["passed"] = cls in ("partition-independent", "coherence-limited",
                                "convergent-quadrature")
        s13 = SC.s13_coherence(SC.ALL[nm], s11,
                               r["screens"].get("S12_coarse_selective", {}))
        r["screens"]["S13_coherence"] = s13
        r["failed"] = [k for k, v in r["screens"].items()
                       if v.get("passed") is False]
        r["verdict"] = "PASS" if not r["failed"] else "FAIL"
    print("reclassified:", reclassified, flush=True)

    an = doc.setdefault("analyses", {})
    # The first pass measured the well discontinuity between +e and -e, which
    # is identically zero because n n^T is even in n; the real discontinuity is
    # between different LINES of approach and needs the softening pushed below
    # the probe offset.  Recomputed here for every tensor candidate.
    disc = {}
    qx, qm = SC.galaxy_cloud(Nq=32768)
    wx, wm = F.nested_partitions(qx, qm, [100])[100]
    for nm, c in SC.ALL.items():
        if c.kind in SC.TENSOR_KINDS:
            try:
                disc[nm] = SC._well_discontinuity(c, wx, wm)
            except Exception as e:                       # noqa: BLE001
                disc[nm] = {"error": f"{type(e).__name__}: {e}"}
    an["well_discontinuity"] = disc
    for nm, v in disc.items():
        if "error" not in v:
            print(f"  discontinuity {nm:22s} N=100 rows: {v['N_partition']:.4g}"
                  f"   single row: {v['single_well']:.4g}", flush=True)

    an["coherence_scaling"] = SC.analysis_coherence_scaling()
    for k, v in an["coherence_scaling"].items():
        print(f"  coherence scaling {k:22s} slope={v['mean_slope']} "
              f"-> {v['verdict']}", flush=True)
    if "D_response_collapse" not in an:
        an["D_response_collapse"] = SC.analysis_D_collapse()
        print("  added D_response_collapse", flush=True)

    doc["source_hashes"] = SC.source_hashes()
    doc["gpu_fallbacks"] = list(F.GPU_FALLBACKS)
    doc.setdefault("repairs", []).append(dict(
        recomputed_candidates=broken, added_candidates=missing,
        added=["sensitivities.D_reciprocity_vs_alpha", "source_hashes"]))
    Path(PATH).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", PATH, flush=True)


if __name__ == "__main__":
    main()
