"""Post-run corrections and the checks that need the survivor list.

    1. Re-score the three BASE_ rows with the response genuinely OFF.  The
       first full run fitted an amplitude for them because ch_cluster's W was
       1 rather than 0 when form == 'off', which turned a base-only row into a
       global rescaling of a0 that the radial channel never saw.  Fixed in
       ch_cluster.base_tensor; the three rows are recomputed and patched into
       tournament.json.
    2. Does any SURVIVOR rest on a clipped response, a grid-edge amplitude, or
       a grid-edge a0?  Reported per survivor rather than in aggregate.
    3. The momentum screen for the surviving structure at an amplitude where
       K is not degenerate, since the survivors' own fitted amplitudes drive
       the condition number past 1e8.
    4. The harmonic-vs-arithmetic bracket restricted to the survivors.

Writes finalise.json and patches tournament.json in place.
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
import ch_radial as CR                                           # noqa: E402
import ch_vertical as CV                                         # noqa: E402
import screens as SS                                             # noqa: E402
import tournament as T                                           # noqa: E402
from tw_core import W_of, Candidate                               # noqa: E402


def main():
    RD = CR.build(verbose=False)
    INV = CR.invariants(RD, "inf")
    VB = CV.VerticalBench()
    VB.h_sigma = float(1.2533 * np.std(VB.OBS_H, ddof=1) / np.sqrt(VB.NG))
    CB = CC.ClusterBench(n=64)
    J = json.load(open(os.path.join(HERE, "tournament.json")))
    R = J["records"]
    out = {}

    # ---- 1. the three base rows, response genuinely off
    base = []
    for b in ("newton", "rar", "aqual"):
        c = Candidate(f"BASE_{b}", base=b, inv="one", form="off",
                      struct="scalar_a0")
        r = T.score_one(c, RD, INV, VB, CB)
        base.append(T.strip(r))
        print(f"BASE_{b:<7} a0={r['a0']:.4e} A={r['A']:.3f} "
              f"radial {r['radial_rms_dex']:.4f} B_z {r['vert_Bz']:.3f} "
              f"h {r['vert_h_as']:.2f} chi2 {r['vert_h_chi2dof']:.2f} "
              f"clus {r['cluster_rms_dex']:.3f} B1Mpc {r['cluster_B_1Mpc']:.3f}"
              f" fld {r['field_dex']:.4f} mem {r['member_dex']:.4f} "
              f"J {r['J']:.3f} survives {r['survives']}")
    out["base_rows_corrected"] = base
    byname = {x["name"]: x for x in base}
    for i, rec in enumerate(R):
        if rec["name"] in byname:
            R[i] = byname[rec["name"]]

    # ---- 2. survivor hygiene
    surv = sorted([r for r in R if r["survives"]], key=lambda r: r["J"])
    hyg = []
    for r in surv:
        c = T.rebuild(r)
        Wg = W_of(r["form"], INV[r["inv"]][RD["is_train"]] / r["I0"], r["m"]) \
            if r["form"] != "off" else np.zeros(1)
        pc = CB.probes["cluster"]["inv"][r["inv"]]
        Wc = W_of(r["form"], CC.W.asnumpy(pc) / r["I0"], r["m"])
        hyg.append(dict(name=r["name"], a0=r["a0"], A=r["A"],
                        W_max_galaxy=float(np.max(Wg)),
                        W_max_cluster=float(np.max(Wc)),
                        clipped=bool(max(np.max(Wg), np.max(Wc)) >= 1e6),
                        a0_at_grid_edge=bool(r["a0"] <= 3.05e-11
                                             or r["a0"] >= 3.95e-10),
                        amp_at_grid_edge=bool(r["at_amp_grid_edge"]),
                        gate_fires_in_galaxies=r["gate_fires_in_galaxies"],
                        harm_vs_arith_dex=r["harm_vs_arith_dex"],
                        cluster_rms_flat_dex=r["cluster_rms_flat_dex"],
                        A_flat=r["cluster_A_flat"],
                        rms_at_A_flat=r["cluster_rms_at_A_flat"],
                        member_dex_flat=r["member_dex_flat"]))
    out["survivor_hygiene"] = hyg
    print(f"\nsurvivors {len(surv)}; any clipped: "
          f"{any(h['clipped'] for h in hyg)}; any a0 at grid edge: "
          f"{any(h['a0_at_grid_edge'] for h in hyg)}; any amplitude at grid "
          f"edge: {any(h['amp_at_grid_edge'] for h in hyg)}")
    hv = [h["harm_vs_arith_dex"] for h in hyg]
    print(f"harmonic-vs-arithmetic bracket over survivors: median "
          f"{np.median(hv):.4f}, max {np.max(hv):.4f} dex")
    out["survivor_harm_vs_arith"] = dict(median=float(np.median(hv)),
                                         max=float(np.max(hv)))

    # ---- 3. momentum for the surviving structure at a tractable amplitude
    mom = []
    ws = [w for w in CC.WELL_SETTINGS if w["tag"] == "plaw_p0q1s2_L300"][0]
    for A in (-95.0, -40.0, -20.0, -10.0, -5.0):
        c = Candidate("S", base="aqual", a0=1.041e-10, inv="phi", form="pow",
                      m=2.0, I0=3e12, struct="tensor_S", A=A,
                      extra=dict(well=ws))
        r = SS.momentum(c, n=32)
        mom.append(dict(A=A, excess=r["excess"], base_null=r["base_null_rel"],
                        K_min_eig=r.get("K_min_eig"), K_cond=r.get("K_cond"),
                        gradK=r.get("gradK_term_rel"),
                        surface=r.get("surface_term_rel"),
                        note=r.get("note")))
        print(f"tensor_S A={A:>7.1f}  excess "
              f"{r['excess'] if np.isfinite(r['excess']) else float('nan'):.4f}"
              f"  K_cond {r.get('K_cond')}")
    out["momentum_tensor_S"] = mom

    J["records"] = R
    J["finalised"] = True
    with open(os.path.join(HERE, "tournament.json"), "w", newline="\n") as fh:
        json.dump(J, fh, indent=1, default=float)
    with open(os.path.join(HERE, "finalise.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote finalise.json and patched tournament.json")


if __name__ == "__main__":
    main()
