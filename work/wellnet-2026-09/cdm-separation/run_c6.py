"""run_c6.py -- the out-of-grammar injection for the CDM discriminators.

Stage 4's C6 asks whether a law injected from OUTSIDE the inference grammar is
recovered.  For the tensor detectors that is done in ``run_forward.py`` (F4: a
log-Gaussian ring quadrupole on the external axis, a radial family neither the
generator nor the halo model contains).  For the CDM discriminators the
equivalent injection is a BARYON-ALIGNED quadrupole with the same out-of-family
radial profile: nothing about S_bar assumes a halo-shaped radius run, and this
is the test of that claim.

The injection is built without touching ``forward.py``: a ring quadrupole is
emitted on the external axis and then the two axis LABELS are exchanged.  Both
axes are drawn independently and uniformly, so the exchange is exact, and it
leaves the estimator facing a baryon-aligned quadrupole of an out-of-family
radial shape with an unrelated external axis.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import forward as F                                        # noqa: E402
import guard                                               # noqa: E402
import pipeline as PL                                      # noqa: E402
from universes.stats import rate_with_ci, responsiveness   # noqa: E402

RES = os.path.join(HERE, "results")
N = int(os.environ.get("N_C6", 400))


def one(seed, A, swap=True):
    rng = np.random.default_rng(seed)
    cl = F.corpus("tensor", rng, n_clu=12, A_tensor=A, ring=True)
    if swap:
        for c in cl:
            c["pa_bar_obs"], c["axis_ext_obs"] = c["axis_ext_obs"], c["pa_bar_obs"]
    return PL.statistics(PL.cluster_rows(cl, F.sigma_crit))


def main():
    guard.start()
    t0 = time.time()
    Fw = json.load(io.open(os.path.join(RES, "F_forward.json"), encoding="utf-8"))
    crit = {s: Fw["F2_sizing"]["cdm_null"][s]["crit"]
            for s in ("S_bar", "S_diff", "S_morph", "S_shape")}
    amps = [0.0, 0.05, 0.1, 0.2, 0.4]
    out = {"n": N, "amplitudes": amps, "critical_values_from": "F_forward cdm_null",
           "rows": {}}
    for i, A in enumerate(amps):
        pool = [one(12_300_000 + 5011 * i + k, A) for k in range(N)]
        row = {}
        for s, c in crit.items():
            v = np.array([p[s] for p in pool], float)
            row[s] = dict(mean=float(v.mean()), sd=float(v.std()),
                          upper=rate_with_ci(int(np.sum(v >= c["up"])), len(v)),
                          two_sided=rate_with_ci(int(np.sum(np.abs(v) >= c["two"])),
                                                 len(v)))
        out["rows"][str(A)] = row
        print(f"  A={A:<5g} S_bar mean {row['S_bar']['mean']:+7.2f} "
              f"recovery {row['S_bar']['upper']['rate']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    out["responsiveness"] = {
        s: responsiveness(amps, [out["rows"][str(a)][s]["mean"] for a in amps])
        for s in crit}
    # C6 asks whether an out-of-grammar injection is RECOVERED, not whether it
    # is recovered with the sign the in-grammar family would have given.  The
    # ring drives S_diff and S_shape the other way, so the two-sided rate is
    # the honest recovery measure; the one-sided rate is kept beside it.
    out["recovery_at_A0.1"] = {s: out["rows"]["0.1"][s]["two_sided"]["rate"]
                               for s in crit}
    out["recovery_at_A0.1_one_sided"] = {s: out["rows"]["0.1"][s]["upper"]["rate"]
                                         for s in crit}
    out["provenance"] = guard.stop()
    p = os.path.join(RES, "C6_out_of_grammar.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
