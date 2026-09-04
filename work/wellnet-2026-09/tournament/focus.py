"""Focused follow-ups the main grid cannot answer, written to focus.json.

    1. THE HEAD-TO-HEAD the brief asks for.  The scalar competitor
       a0 -> a0 f(|Phi_N|/Phi_0) against the well-network tensor at the SAME
       gate, channel by channel, so it is visible exactly which channel (if
       any) the anisotropy earns its place on.
    2. WHY the well-network tensor escapes the member-galaxy screen.  S is a
       normalised direction average, so whether the HOST galaxy or the crowd
       of other cluster members dominates it is decided by the weight family's
       mass exponent p and by self-exclusion -- both modelling choices, not
       physics.  The brief's literal formula (p = 1, no self-exclusion) is
       scanned beside the tensor lane's surviving corner.
    3. THE BOUNDARY RULE for |Phi_N|, which Run Z warned DEFINES the variable.
       A galaxy's outer continuation is uncertain by about a decade in |Phi_N|,
       and a potential-depth gate is a function of exactly that number.
    4. Resolution and seed robustness of anything that ends up mattering.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ch_cluster as CC                                          # noqa: E402
import ch_radial as CR                                           # noqa: E402
import ch_vertical as CV                                         # noqa: E402
import screens as SS                                             # noqa: E402
from tw_core import KPC, Candidate                                # noqa: E402
from tournament import AMPS, H_OBS, TOL, joint                    # noqa: E402

#: the brief's LITERAL well formula -- mass weighting on, no self-exclusion --
#: added beside the tensor lane's surviving corner.
WELLS_ALL = CC.WELL_SETTINGS + [
    dict(tag="plaw_p1q1s2_L300_literal", family="plaw", p=1.0, q=1.0, s=2.0,
         L=300.0 * KPC, exclude_nearest=False),
    dict(tag="plaw_p1q2s2_L1000_literal", family="plaw", p=1.0, q=2.0, s=2.0,
         L=1000.0 * KPC, exclude_nearest=False),
]
GATES = [("sat", 2.0, 1e12), ("sat", 4.0, 1e12), ("sat", 2.0, 3e12),
         ("pow", 2.0, 3e12), ("pow", 1.0, 1e12)]


def row(name, cand, RD, INV, VB, CB, fit_a0=True):
    cand.A = 0.0
    if fit_a0:
        CR.fit_a0(cand, RD, INV)
    cl = CC.evaluate(CB, cand, AMPS, target="lane12")
    rms, _ = CR.score(cand, RD, INV)
    vt = VB.predict(cand)
    asym = SS.asymptotic(cand)
    r = dict(name=name, a0=cand.a0, A=cand.A, radial_rms_dex=rms,
             cluster_rms_dex=cl["rms_dex_lane12"],
             cluster_rms_flat_dex=cl["rms_dex_flat"],
             cluster_B=cl["B_cluster"], cluster_B_1Mpc=cl["B_1Mpc"],
             field_dex=cl["field_dex"], member_dex=cl["member_dex"],
             member_dex_flat=cl["member_dex_flat"],
             A_flat=cl["A_flat"], rms_at_A_flat=cl["rms_dex_at_A_flat"],
             harm_vs_arith_dex=cl["harm_vs_arith_dex"],
             at_amp_grid_edge=cl["at_amp_grid_edge"],
             vert_Bz=vt["Bz_law"], vert_h_as=vt["h_median_as"],
             vert_h_chi2dof=vt["h_chi2dof"], vert_A_dyn=vt["A_dyn_2p2"],
             asym_slope=asym["slope_total"], _h_sigma=VB.h_sigma)
    r.update(joint(r))
    r.pop("_h_sigma")
    return r


def main():
    t0 = time.time()
    out = {}
    RD = CR.build(verbose=False)
    INV = CR.invariants(RD, "inf")
    VB = CV.VerticalBench()
    VB.h_sigma = float(1.2533 * np.std(VB.OBS_H, ddof=1) / np.sqrt(VB.NG))
    CB = CC.ClusterBench(n=64)

    # ---- 1 + 2: head-to-head at matched gates, across every well setting
    print("head-to-head at matched gates")
    hh = []
    for form, m, I0 in GATES:
        g = f"{form} m={m:g} Phi0={I0:.0e}"
        for base in ("aqual", "rar"):
            for st in ("scalar_a0", "iso_K", "tensor_d", "tensor_T"):
                c = Candidate("x", base=base, inv="phi", form=form, m=m,
                              I0=I0, struct=st)
                hh.append(dict(gate=g, base=base, structure=st, well=None,
                               **row(f"{base}|{st}|{g}", c, RD, INV, VB, CB)))
            for ws in WELLS_ALL:
                c = Candidate("x", base=base, inv="phi", form=form, m=m,
                              I0=I0, struct="tensor_S", extra=dict(well=ws))
                hh.append(dict(gate=g, base=base, structure="tensor_S",
                               well=ws["tag"],
                               **row(f"{base}|tensor_S[{ws['tag']}]|{g}", c,
                                     RD, INV, VB, CB)))
        print(f"   {g}: {len(hh)} rows  {time.time()-t0:.0f}s")
    out["head_to_head"] = hh

    # ---- why tensor_S escapes the member screen: p and self-exclusion
    esc = []
    for ws in WELLS_ALL:
        sub = [h for h in hh if h["well"] == ws["tag"]]
        esc.append(dict(tag=ws["tag"], p=ws["p"],
                        exclude_nearest=ws["exclude_nearest"],
                        member_dex_min=min(h["member_dex"] for h in sub),
                        member_dex_median=float(np.median(
                            [h["member_dex"] for h in sub]))))
    out["member_escape_vs_weight_family"] = dict(
        rows=esc,
        note="S is normalised, so whether the HOST or the crowd of other "
             "cluster members dominates it is set by p and by self-exclusion. "
             "p = 0 flattens the mass weighting so a 4e11 Msun host counts no "
             "more than a 1e9 Msun dwarf, and the 300-member crowd then "
             "orients S along the CLUSTER radius rather than the host's -- "
             "which is what lets the member violation collapse.  That escape "
             "is a choice inside the weight family, not a property of "
             "anisotropy.")

    # ---- 3: the |Phi_N| boundary rule
    print("boundary-rule sensitivity")
    br = {}
    for rule in CR.PHI_RULES:
        I2 = CR.invariants(RD, rule)
        rows = []
        for form, m, I0 in GATES[:3]:
            c = Candidate("x", base="aqual", inv="phi", form=form, m=m, I0=I0,
                          struct="scalar_a0")
            c.A = 0.0
            CR.fit_a0(c, RD, I2)
            cl = CC.evaluate(CB, c, AMPS, target="lane12")
            rms, _ = CR.score(c, RD, I2)
            rows.append(dict(gate=f"{form} m={m:g} Phi0={I0:.0e}", A=c.A,
                             a0=c.a0, radial_rms_dex=rms,
                             gate_W_max_galaxy=float(np.max(
                                 __import__("tw_core").W_of(
                                     form, I2["phi"][RD["is_train"]] / I0, m))),
                             member_dex=cl["member_dex"]))
        br[rule] = dict(median_phi=float(np.median(I2["phi"][RD["is_train"]])),
                        max_phi=float(np.max(I2["phi"][RD["is_train"]])),
                        rows=rows)
    out["phi_boundary_rule"] = dict(
        primary="inf", rules=br,
        note="'inf' and 'flat' are both GLOBAL prescriptions.  They differ by "
             "0.87 dex in the median galaxy |Phi_N|, and a potential-depth "
             "gate is a function of exactly that number, so the outer "
             "continuation of a galaxy's baryon distribution is a first-order "
             "systematic on the whole mechanism, not a detail.")

    # ---- 4: momentum vs resolution, on the structures that matter
    print("momentum vs resolution")
    mom = []
    for st, A in (("scalar_a0", 30.0), ("iso_K", 3.4), ("tensor_d", -5.25),
                  ("tensor_T", 4.0)):
        for n in (28, 36, 44):
            c = Candidate("x", base="aqual", a0=1.058e-10, inv="phi",
                          form="sat", m=4.0, I0=1e12, struct=st, A=A)
            r = SS.momentum(c, n=n)
            mom.append(dict(structure=st, A=A, n=n,
                            excess=r["excess"], base_null=r["base_null_rel"],
                            newton_null=r["newton_null_rel"],
                            gradK=r.get("gradK_term_rel"),
                            surface=r.get("surface_term_rel")))
            print(f"   {st:<10} n={n} excess {r['excess']:.4f}")
    out["momentum_vs_resolution"] = mom

    # ---- 5: seed robustness of the member screen
    print("member screen vs member realisation")
    seeds = (20260903, 11, 23, 37, 51)
    sr = []
    for st, ws in (("scalar_a0", None), ("tensor_S", WELLS_ALL[2]),
                   ("tensor_S", WELLS_ALL[4])):
        vals, mems = [], []
        for sd in seeds:
            B = CC.ClusterBench(n=64, seed=sd)
            c = Candidate("x", base="aqual", a0=1.058e-10, inv="phi",
                          form="sat", m=2.0, I0=1e12, struct=st,
                          extra=dict(well=ws) if ws else {})
            e = CC.evaluate(B, c, AMPS, target="lane12")
            vals.append(e["rms_dex_lane12"])
            mems.append(e["member_dex"])
            del B
        sr.append(dict(structure=st, well=(ws["tag"] if ws else None),
                       seeds=list(seeds),
                       member_dex=[float(v) for v in mems],
                       member_mean=float(np.mean(mems)),
                       member_sd=float(np.std(mems, ddof=1)),
                       cluster_rms=[float(v) for v in vals],
                       tol=TOL["galaxy_dex"]))
        print(f"   {st:<10} member {np.mean(mems):.4f} +- "
              f"{np.std(mems, ddof=1):.4f}")
    out["member_realisation_scatter"] = sr

    # ---- 6: cluster resolution
    print("cluster grid resolution")
    res = []
    for n in (48, 64, 96):
        B = CC.ClusterBench(n=n)
        c = Candidate("x", base="aqual", a0=1.058e-10, inv="phi", form="sat",
                      m=2.0, I0=1e12, struct="tensor_S",
                      extra=dict(well=WELLS_ALL[2]))
        e = CC.evaluate(B, c, AMPS, target="lane12")
        res.append(dict(n=n, A=e["A"], B=e["B_cluster"],
                        rms=e["rms_dex_lane12"], member=e["member_dex"]))
        print(f"   n={n}  B={np.round(e['B_cluster'],3)}  "
              f"member={e['member_dex']:.4f}")
        del B
    out["cluster_resolution"] = res

    out["seconds"] = time.time() - t0
    with open(os.path.join(HERE, "focus.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote focus.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
