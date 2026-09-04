"""The member-galaxy screen under a FROZEN global amplitude.

focus.py refits the amplitude A on every member realisation, which conflates
two different things.  A is a GLOBAL constant of the law: it is fitted once and
frozen, and the member realisation is a nuisance we do not know.  The honest
test is therefore

    fit A on realisation 1, FREEZE it, then measure the member violation on
    realisations 2..N,

which is what this does.  Both protocols are reported, because they answer
different questions and they give very different numbers.

The tensor lane's own seed test (seed_robustness.json) held its point fixed at
A_T = -24.7 and measured member 0.031 +- 0.023 dex across five draws.  That is
the frozen protocol on ONE point; this widens it to the structures the
tournament ranks, and adds the refitted protocol beside it.

Writes member_freeze.json.
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

SEEDS = (20260903, 11, 23, 37, 51, 67, 83, 101)
AMPS = np.unique(np.concatenate([
    -np.logspace(3, 1.79, 40), np.linspace(-60.0, 60.0, 241),
    np.logspace(1.79, 3, 40)]))
WELLS = CC.WELL_SETTINGS + [
    dict(tag="plaw_p1q1s2_L300_literal", family="plaw", p=1.0, q=1.0, s=2.0,
         L=300.0 * KPC, exclude_nearest=False)]
CASES = [
    ("scalar_a0", None, "sat", 2.0, 1e12),
    ("scalar_a0", None, "sat", 4.0, 1e12),
    ("iso_K", None, "sat", 2.0, 1e12),
    ("tensor_d", None, "sat", 2.0, 1e12),
    ("tensor_T", None, "sat", 2.0, 1e12),
    ("tensor_S", WELLS[2], "sat", 2.0, 1e12),
    ("tensor_S", WELLS[2], "sat", 4.0, 1e12),
    ("tensor_S", WELLS[0], "sat", 4.0, 1e12),
    ("tensor_S", WELLS[4], "sat", 2.0, 1e12),
]


def main():
    benches = {s: CC.ClusterBench(n=64, seed=s) for s in SEEDS}
    out = []
    for st, ws, form, m, I0 in CASES:
        c = Candidate("x", base="aqual", a0=1.058e-10, inv="phi", form=form,
                      m=m, I0=I0, struct=st,
                      extra=dict(well=ws) if ws else {})
        r0 = CC.evaluate(benches[SEEDS[0]], c, AMPS, target="lane12")
        Afrz = r0["A"]
        froz, refit, rmsf = [], [], []
        for s in SEEDS:
            B = benches[s]
            c.A = Afrz
            Bm, _ = B.B_of("member", c, [Afrz])
            Bc, _ = B.B_of("cluster", c, [Afrz])
            froz.append(float(np.max(np.abs(np.log10(
                np.maximum(Bm[0], 1e-12))))))
            rmsf.append(float(np.sqrt(np.mean(
                (np.log10(np.maximum(Bc[0], 1e-12))
                 - np.log10(CC.BREQ)) ** 2))))
            rr = CC.evaluate(B, c, AMPS, target="lane12")
            refit.append(float(rr["member_dex"]))
            c.A = Afrz
        row = dict(structure=st, well=(ws["tag"] if ws else None),
                   gate=f"phi {form} m={m:g} Phi0={I0:.0e}", A_frozen=Afrz,
                   seeds=list(SEEDS),
                   member_dex_frozen=froz,
                   member_frozen_mean=float(np.mean(froz)),
                   member_frozen_sd=float(np.std(froz, ddof=1)),
                   member_frozen_n_pass=int(sum(v <= 0.040 for v in froz)),
                   cluster_rms_frozen=rmsf,
                   member_dex_refit=refit,
                   member_refit_mean=float(np.mean(refit)),
                   member_refit_sd=float(np.std(refit, ddof=1)),
                   member_refit_n_pass=int(sum(v <= 0.040 for v in refit)),
                   tol=0.040)
        out.append(row)
        print(f"{st:<11}{str(row['well'])[:22]:<23}{row['gate']:<22}"
              f"A={Afrz:>8.2f}  frozen {row['member_frozen_mean']:.3f} +- "
              f"{row['member_frozen_sd']:.3f} ({row['member_frozen_n_pass']}"
              f"/{len(SEEDS)} pass)   refit "
              f"{row['member_refit_mean']:.3f} +- {row['member_refit_sd']:.3f}"
              f" ({row['member_refit_n_pass']}/{len(SEEDS)})", flush=True)
    with open(os.path.join(HERE, "member_freeze.json"), "w",
              newline="\n") as fh:
        json.dump(dict(seeds=list(SEEDS), tol=0.040, rows=out), fh, indent=1,
                  default=float)
    print("wrote member_freeze.json")


if __name__ == "__main__":
    main()
