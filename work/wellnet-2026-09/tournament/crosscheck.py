"""Cross-check: does this lane's cluster channel reproduce the tensor lane's
own published survivor, number for number?

Tensor lane REPORT.md section 8.3 reports exactly one scanned point that
threads both galaxy needles against the lane-12 shape:

    plaw, p = 0, q = 1, s = 2, L = 300 kpc, no self-exclusion,
    phi gate Phi_0 = 1e12, m = 4,  A_T = -24.7
    -> B = 2.51, 3.22, 2.57, 1.98   RMS 0.099 dex   member violation 0.015 dex

and its ungated headline is A_T = -4.7 for B = 2 at 1 Mpc.  Both are
recomputed here through THIS lane's code path, which differs from the tensor
lane's in the Newtonian potential (Plummer softening used consistently for
Phi, g, the Hessian and rho, against their hard 1 kpc floor) and in nothing
else that matters.  Writes crosscheck.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ch_cluster as CC                                          # noqa: E402
from tw_core import KPC, Candidate                                # noqa: E402

WS = dict(tag="plaw_p0q1s2_L300", family="plaw", p=0.0, q=1.0, s=2.0,
          L=300.0 * KPC, exclude_nearest=False)
PUBLISHED = dict(A_T=-24.7, B=[2.51, 3.22, 2.57, 1.98], rms_dex=0.099,
                 member_dex=0.015, ungated_A_T_for_B2=-4.7)


def main():
    B = CC.ClusterBench(n=64)
    out = dict(published=PUBLISHED, well=WS["tag"])

    # (a) at the tensor lane's own amplitude, not refitted
    c = Candidate("x", base="aqual", a0=1.2e-10, inv="phi", form="sat", m=4.0,
                  I0=1e12, struct="tensor_S", A=-24.7, extra=dict(well=WS))
    Bc, Bca = B.B_of("cluster", c, [-24.7])
    Bf, _ = B.B_of("field", c, [-24.7])
    Bm, _ = B.B_of("member", c, [-24.7])
    rms = float(np.sqrt(np.mean((np.log10(Bc[0]) - np.log10(CC.BREQ)) ** 2)))
    out["at_published_amplitude"] = dict(
        A_T=-24.7, B=[float(x) for x in Bc[0]], rms_dex=rms,
        field_dex=float(np.max(np.abs(np.log10(Bf[0])))),
        member_dex=float(np.max(np.abs(np.log10(Bm[0])))),
        B_arith=[float(x) for x in Bca[0]])

    # (b) refitted on this lane's amplitude grid
    amps = np.linspace(-60.0, 0.0, 241)
    r = CC.evaluate(B, c, amps, target="lane12")
    out["refitted_here"] = {k: v for k, v in r.items()
                            if not isinstance(v, np.ndarray)}

    # (c) the ungated headline: A_T that gives B = 2 at 1 Mpc
    c0 = Candidate("u", base="aqual", a0=1.2e-10, inv="one", form="off",
                   struct="tensor_S", extra=dict(well=WS))
    a2 = np.linspace(-20.0, 0.0, 401)
    Bh, Ba = B.B_of("cluster", c0, a2)
    j = int(np.argmin(np.abs(Bh[:, 2] - 2.0)))
    ja = int(np.argmin(np.abs(Ba[:, 2] - 2.0)))
    out["ungated"] = dict(A_T_for_B2_harmonic=float(a2[j]),
                          A_T_for_B2_arithmetic=float(a2[ja]),
                          B_at_that_amp=[float(x) for x in Bh[j]],
                          arithmetic_saturates=bool(Ba[:, 2].max() < 2.2),
                          arithmetic_max_B=float(Ba[:, 2].max()),
                          harmonic_max_B=float(Bh[:, 2].max()),
                          note="the tensor lane reports the ARITHMETIC mean "
                               "saturating near B = 2.1 and demanding "
                               "A_T = -12.8, against the harmonic mean's "
                               "-4.7; both behaviours are re-measured here")
    with open(os.path.join(HERE, "crosscheck.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
