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
    print("candidates with a screen that raised:", broken, flush=True)
    for nm in broken:
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

    doc["source_hashes"] = SC.source_hashes()
    doc["gpu_fallbacks"] = list(F.GPU_FALLBACKS)
    doc.setdefault("repairs", []).append(dict(
        recomputed_candidates=broken,
        added=["sensitivities.D_reciprocity_vs_alpha", "source_hashes"]))
    Path(PATH).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", PATH, flush=True)


if __name__ == "__main__":
    main()
