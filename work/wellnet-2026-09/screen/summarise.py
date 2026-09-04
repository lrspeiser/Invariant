"""Compact console summary of screen_results.json and stage2_results.json.

Reporting aid only -- it computes nothing, it just lays out what the two result
files already contain so the tables in REPORT.md can be transcribed exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCREENS = ["S1_dimensions", "S2_rotation", "S2b_grid_rotation",
           "S3_translation", "S4_gauge_offset", "S5_positive_definite",
           "S6_newtonian_limit", "S7_asymptotics", "S8_gain_bound",
           "S9_permutation", "S10_reciprocity", "S11_coarse_uniform",
           "S11b_coarse_potential", "S12_coarse_selective", "S13_coherence"]
SHORT = {k: k.split("_")[0] for k in SCREENS}


def mark(v):
    p = v.get("passed")
    if p is None:
        return " . "
    return " P " if p else " F "


def num(v, k="value"):
    x = v.get(k)
    if x is None:
        return ""
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)[:12]


def main(a="screen_results.json", b="stage2_results.json"):
    d = json.loads(Path(a).read_text(encoding="utf-8"))
    print("=" * 100)
    print("STAGE 1 verdict matrix   (P pass, F fail, . informational)")
    print("=" * 100)
    hdr = "candidate".ljust(24) + "".join(SHORT[k].rjust(6) for k in SCREENS)
    print(hdr)
    for nm, r in d["candidates"].items():
        row = nm.ljust(24)
        for k in SCREENS:
            row += mark(r["screens"].get(k, {})).rjust(6)
        print(row + "   " + r["verdict"])

    print("\n" + "=" * 100)
    print("KEY NUMBERS")
    print("=" * 100)
    for nm, r in d["candidates"].items():
        s = r["screens"]
        print(f"\n{nm}   [{r['kind']}]  {r['note'][:70]}")
        for k in SCREENS:
            v = s.get(k, {})
            if "error" in v:
                print(f"   {k:24s} ERROR {v['error'][:80]}")
                continue
            extra = ""
            if k == "S6_newtonian_limit" and "anisotropy" in v:
                extra = f" aniso={num(v,'anisotropy')}"
            if k == "S7_asymptotics":
                extra = (f" slope_out={num(v,'slope_large_r')}"
                         f" slope_in={num(v,'slope_small_r')}"
                         f" Kdisc={v.get('K_discontinuity_at_well')}")
            if k == "S8_gain_bound":
                extra = (f" boost20kpc={num(v,'boost_at_20kpc')}"
                         f" boost2e4={num(v,'boost_at_2e4kpc')}"
                         f" needed={num(v,'boost_needed_at_2e4kpc')}"
                         f" unbounded={v.get('unbounded_boost')}")
            if k == "S10_reciprocity":
                extra = (f" law={num(v,'law_net_force_rel')}"
                         f" null={num(v,'newton_null_rel')}"
                         f" identity={num(v,'identity_net_force_rel')}"
                         f" agree={num(v,'identity_agreement')}")
            if k == "S11_coarse_uniform":
                extra = (f" beta={num(v,'rate_beta')}"
                         f" beta_step={num(v,'rate_beta_step')}"
                         f" cls={v.get('classification')}"
                         f" Nsafe={v.get('N_safe')}")
            if k == "S11b_coarse_potential":
                extra = (f" dPhi={num(v,'dPhi_1_to_max')}"
                         f" dPhi_rms={num(v,'dPhi_rms_1_to_max')}"
                         f" dvc={num(v,'dvc_1_to_max')}"
                         f" N={v.get('N_lo')}->{v.get('N_hi')}")
            if k == "S12_coarse_selective":
                extra = (f" wslope={num(v,'weight_ratio_slope')}"
                         f" pred={num(v,'weight_ratio_slope_predicted')}"
                         f" beta_step={num(v,'rate_beta_step')}")
            if k == "S13_coherence":
                extra = (f" cls={v.get('classification')}"
                         f" physical={v.get('physical')}"
                         f" fracAfterL={num(v,'fraction_of_total_drift_after_L')}")
            print(f"   {k:24s} {mark(v)} {num(v):>10s}  tol={v.get('tol')}{extra}")

    print("\n" + "=" * 100)
    print("ANALYSES")
    print("=" * 100)
    print(json.dumps(d.get("analyses", {}), indent=1)[:4000])
    print("\nSWEEPS")
    print(json.dumps(d.get("sweeps", {}), indent=1)[:3000])
    print("\nSENSITIVITIES")
    for k, v in d.get("sensitivities", {}).items():
        print(f"  {k:34s} values={v['values']}")
        print(f"  {'':34s} stat  ={[round(x,6) for x in v['statistic']]}")
        print(f"  {'':34s} spread={v['spread']:.6g}  {v['monotone_invariance_guard']}")

    p = Path(b)
    if p.exists():
        s2 = json.loads(p.read_text(encoding="utf-8"))
        print("\n" + "=" * 100)
        print("STAGE 2")
        print("=" * 100)
        for nm, r in s2.items():
            print(f"\n{nm}  {r['verdict']}  failed={r['failed']}")
            for gk, g in r["geometries"].items():
                if "error" in g:
                    print(f"   {gk:26s} ERROR {g['error'][:90]}")
                    continue
                def f(key, fmt=".4g"):
                    x = g.get(key)
                    return "n/a" if x is None else format(float(x), fmt)

                line = f"   {gk:26s} {'P' if g.get('passed') else 'F'}"
                if gk == "G1_point_mass":
                    line += (f" boost={[round(x,3) for x in g['boost'][::3]]}"
                             f" gridexcess={f('grid_convergence_excess')}"
                             f" Kcond={g.get('K_cond')}")
                if gk == "G2_two_body":
                    line += f" worst_mid_excess={f('value','.3e')}"
                    for sep, v in g.get("per_separation", {}).items():
                        line += (f"\n        d={sep:>5s} kpc mid={v['midpoint_force_rel']:.3e}"
                                 f" null={v['midpoint_force_newton_null']:.3e}"
                                 f" axial_excess={v['max_axial_excess']:.4g}")
                if gk == "G3_exponential_disk":
                    line += (f" boost={[round(x,3) for x in g['boost'][::3]]}"
                             f" nvf={f('newton_vs_freeman_outer')}")
                if gk == "G4_sphere":
                    line += (f" boost={[round(x,3) for x in g['boost'][::3]]}"
                             f" nve={f('newton_vs_exact')}")
                if gk == "G5_disk_plus_external":
                    line += (f" excess_shift={f('excess_shift')}"
                             f" g_ext/a0={f('g_ext_over_a0','.3g')}")
                if gk == "G6G7_cluster_subdivision":
                    md = g.get("implied_Mdyn_at_1Mpc_Msun", {})
                    line += (f" dvc={f('dvc_1_to_max')}"
                             f" Mdyn_ratio={f('Mdyn_ratio_1_to_max','.5g')}"
                             f" rows={g.get('n_rows_used')}"
                             f" Mdyn/1e13={ {k: round(v/1e13,3) for k,v in md.items()} }"
                             f" infeasible={list(g.get('infeasible',{}))}")
                print(line)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
