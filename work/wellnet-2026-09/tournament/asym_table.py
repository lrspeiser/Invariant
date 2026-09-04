"""THE SHARPENED BOUNDEDNESS STATEMENT, as a table rather than an assertion.

Run AB's theorem: a response confined to a bounded range cannot change the
asymptotic force law.  The repair it prescribes is "make the response
unbounded".  This lane measures whether that repair is available INSIDE the
stated grammar, and finds that it is not, for a reason the theorem does not
name:

    every invariant in the grammar DECREASES outward for an isolated source,

so an unbounded f evaluated on a vanishing argument still tends to a constant
(or to zero).  W -> 0 or W -> const leaves the asymptotics untouched -- const
only renormalises G -- and the only way to make W grow is to use a NEGATIVE
exponent, which does change the asymptotics and changes them the wrong way:
the rotation curve rises without bound instead of flattening.

    invariant     behaviour as r -> inf for a point mass
    gn            g_N/a0        ~ r^-2      -> 0
    phi           |Phi_N|/Phi_0 ~ r^-1      -> 0
    rhobar        rho/rho_0     = 0 outside -> 0
    tidal         |T|/T_0       ~ r^-3      -> 0
    qbar          M(<L_NL)/(M+M_0)          -> const, and bounded anyway

Writes asym_table.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import screens as SS                                             # noqa: E402
from tw_core import KPC, MSUN, Candidate, W_sup                   # noqa: E402
from tournament import FORM_M, INV_SCALES                         # noqa: E402


def main():
    rows = []
    for inv, scales in INV_SCALES.items():
        I0 = scales[1]
        for form, m in FORM_M:
            for st, A in (("scalar_a0", 10.0), ("iso_K", 2.0)):
                c = Candidate("x", base="aqual", a0=1.06e-10, inv=inv,
                              form=form, m=m, I0=I0, struct=st, A=A)
                r = SS.asymptotic(c)
                rows.append(dict(inv=inv, I0=I0, form=form, m=m, struct=st,
                                 A=A, W_lo=r["W_at_r_lo"], W_hi=r["W_at_r_hi"],
                                 W_sup=("inf" if not np.isfinite(r["W_sup"])
                                        else r["W_sup"]),
                                 slope_total=r["slope_total"],
                                 slope_response=r["slope_response"],
                                 flat=r["flat_curve"]))
    base = {}
    for b in ("newton", "aqual", "rar"):
        c = Candidate("b", base=b, a0=1.06e-10)
        base[b] = SS.asymptotic(c)["slope_total"]
    grow = [r for r in rows if r["W_hi"] > r["W_lo"] * 1.0001]
    flat = [r for r in rows if r["flat"]]
    out = dict(base_slopes=base, n_rows=len(rows),
               n_with_W_growing_outward=len(grow),
               n_flat_curve=len(flat),
               slope_when_W_grows=sorted({round(r["slope_total"], 4)
                                          for r in grow}),
               rows=rows,
               conclusion=(
                   "Of %d (invariant, form, exponent, structure) combinations, "
                   "%d have a response that GROWS outward, and every one of "
                   "those leaves the asymptotic slope at %s rather than -1.  "
                   "The remaining %d leave it exactly at the base law's value, "
                   "because W tends to 0 or to a constant and a constant only "
                   "renormalises G.  Making f unbounded does not help: the "
                   "obstruction is that every invariant in the grammar decays "
                   "outward, so the ARGUMENT vanishes."
                   % (len(rows), len(grow),
                      sorted({round(r["slope_total"], 3) for r in grow}),
                      len(rows) - len(grow))))
    with open(os.path.join(HERE, "asym_table.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("\nby invariant x form (structure = scalar_a0):")
    print(f"{'inv':<8}{'form':<6}{'m':>6}{'W(10kpc)':>12}{'W(1e5kpc)':>12}"
          f"{'slope':>9}{'flat':>6}")
    for r in rows:
        if r["struct"] != "scalar_a0":
            continue
        print(f"{r['inv']:<8}{r['form']:<6}{r['m']:>6.1f}{r['W_lo']:>12.4g}"
              f"{r['W_hi']:>12.4g}{r['slope_total']:>9.4f}"
              f"{str(r['flat']):>6}")


if __name__ == "__main__":
    main()
