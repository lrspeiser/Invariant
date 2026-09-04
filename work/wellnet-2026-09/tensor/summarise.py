"""Turn mechanism_map.json into the tables that go into REPORT.md."""
from __future__ import annotations

import json
import sys

import numpy as np

d = json.load(open("mechanism_map.json"))
W = d["wellnet"]
C = d["channels"]
M = d["meta"]
BAR = "-" * 92


def hdr(t):
    print("\n" + t + "\n" + BAR)


print(f"grid {M['grid']}^3, {M['n_members']} members, "
      f"{M['n_pairs']} pairs, shell cells {M['shell_cells']}")
print(f"|Phi_N|: field galaxy 20 kpc {M['PhiN_field_20kpc']:.3e}   "
      f"cluster 1 Mpc {M['PhiN_cluster_1Mpc']:.3e}   "
      f"member galaxy 20 kpc {M['PhiN_member_20kpc']:.3e}")
print(f"target B = {M['target']} at 1 Mpc, band {M['band']}, "
      f"galaxy tolerance {M['gal_tol_dex']} dex")

hdr("WELL-NETWORK: reach and feasibility by gate")
print(f"{'gate':<14}{'rows':>6}{'reach B=2':>11}{'field ok':>10}"
      f"{'+member ok':>12}{'best field dex':>16}{'best member dex':>17}")
for g in sorted({r["gate"] for r in W}):
    rr = [r for r in W if r["gate"] == g]
    rt = [r for r in rr if r["reaches_target"]]
    fd = min([r["field_dex"] for r in rt], default=float("nan"))
    md = min([r["member_dex"] for r in rt], default=float("nan"))
    print(f"{g:<14}{len(rr):>6}{len(rt):>11}"
          f"{sum(1 for r in rr if r['feasible']):>10}"
          f"{sum(1 for r in rr if r['feasible_incl_members']):>12}"
          f"{fd:>16.3f}{md:>17.3f}")

hdr("WELL-NETWORK: reach and feasibility by weight family / self-exclusion")
print(f"{'family':<10}{'excl':>6}{'rows':>6}{'reach':>7}{'field ok':>10}"
      f"{'best field dex':>16}{'max B':>10}")
for f in sorted({r["family"] for r in W}):
    for ex in (False, True):
        rr = [r for r in W if r["family"] == f and r["exclude_nearest"] == ex]
        if not rr:
            continue
        rt = [r for r in rr if r["reaches_target"]]
        print(f"{f:<10}{int(ex):>6}{len(rr):>6}{len(rt):>7}"
              f"{sum(1 for r in rr if r['feasible']):>10}"
              f"{min([r['field_dex'] for r in rt], default=np.nan):>16.3f}"
              f"{max(r['B_max'] for r in rr):>10.2f}")

hdr("WELL-NETWORK: the ten rows with the smallest field-galaxy violation")
print(f"{'fam':<8}{'p':>4}{'q':>4}{'s':>5}{'L':>6}{'ex':>3} {'gate':<12}"
      f"{'A_T':>8}   B(300,500,1000,1414)         "
      f"{'fld dex':>8}{'mem dex':>8}{'shape':>7}")
for r in sorted([r for r in W if r["reaches_target"]],
                key=lambda r: r["field_dex"])[:10]:
    b = " ".join(f"{v:6.3f}" for v in r["B_cl_at_target"])
    print(f"{r['family']:<8}{r['p']:>4.0f}{r['q']:>4.0f}{r['s']:>5.1f}"
          f"{r['L_kpc']:>6.0f}{int(r['exclude_nearest']):>3} {r['gate']:<12}"
          f"{r['amp_for_target']:>8.2f}   {b}   "
          f"{r['field_dex']:>8.3f}{r['member_dex']:>8.3f}{r['shape']:>7.2f}")

hdr("WELL-NETWORK: how much of the boost survives if the lumpiness is "
    "averaged away")
print("   B_cl is <exp(A_T g S_rr)> over the shell; B_smooth is "
      "exp(A_T g <S_rr>).")
print(f"{'fam':<8}{'ex':>3} {'gate':<12}{'A_T':>8}   "
      f"{'B(1000) full':>13}{'B(1000) smooth':>16}{'ratio':>8}")
sel = sorted([r for r in W if r["reaches_target"] and "B_cl_smooth" in r],
             key=lambda r: r["field_dex"])[:8]
for r in sel:
    print(f"{r['family']:<8}{int(r['exclude_nearest']):>3} {r['gate']:<12}"
          f"{r['amp_for_target']:>8.2f}   {r['B_cl_at_target'][2]:>13.3f}"
          f"{r['B_cl_smooth'][2]:>16.3f}"
          f"{r['B_cl_at_target'][2]/r['B_cl_smooth'][2]:>8.3f}")

hdr("PAIR CHANNELS: reach and feasibility by sign and d_par mode")
print(f"{'mode':<7}{'sign':>6}{'rows':>6}{'reach':>7}{'field ok':>10}"
      f"{'+member ok':>12}{'best field dex':>16}{'best member dex':>17}"
      f"{'max B':>12}")
for mode in sorted({r["mode"] for r in C}):
    for sg in (-1, 1):
        rr = [r for r in C if r["mode"] == mode and r["sign"] == sg]
        if not rr:
            continue
        rt = [r for r in rr if r["reaches_target"]]
        print(f"{mode:<7}{sg:>6}{len(rr):>6}{len(rt):>7}"
              f"{sum(1 for r in rr if r['feasible']):>10}"
              f"{sum(1 for r in rr if r['feasible_incl_members']):>12}"
              f"{min([r['field_dex'] for r in rt], default=np.nan):>16.4f}"
              f"{min([r['member_dex'] for r in rt], default=np.nan):>17.3f}"
              f"{max(r['B_max'] for r in rr):>12.1f}")

hdr("PAIR CHANNELS: the ten rows with the flattest cluster boost profile")
print(f"{'mode':<6}{'sp':>5}{'sl':>6}{'q':>3}{'p':>3}{'L':>6}{'sgn':>4}"
      f"{'alpha':>10}   B(300,500,1000,1414)          {'shape':>7}"
      f"{'fld dex':>9}{'mem dex':>9}{'aniso/iso':>10}")
for r in sorted([r for r in C if r["reaches_target"]],
                key=lambda r: abs(np.log(r["shape"])))[:10]:
    b = " ".join(f"{v:7.3f}" for v in r["B_cl_at_target"])
    print(f"{r['mode']:<6}{r['sigma_perp_kpc']:>5.0f}{r['sigma_par_kpc']:>6.0f}"
          f"{r['q']:>3.0f}{r['p']:>3.0f}{r['L_kpc']:>6.0f}{r['sign']:>4}"
          f"{r['amp_for_target']:>10.2e}   {b}   {r['shape']:>7.3f}"
          f"{r['field_dex']:>9.4f}{r['member_dex']:>9.3f}"
          f"{r['aniso_over_iso_median']:>10.3f}")

hdr("PAIR CHANNELS: is the boost the trace of C or its anisotropy?")
print(f"{'mode':<6}{'sp':>5}{'sl':>6}{'sgn':>4}{'alpha':>10}"
      f"{'B(1000) full':>14}{'B(1000) trace only':>20}{'ratio':>8}"
      f"{'aniso/iso':>11}")
for r in sorted([r for r in C if "B_cl_isotropic_part" in r],
                key=lambda r: abs(np.log(r["shape"])))[:8]:
    print(f"{r['mode']:<6}{r['sigma_perp_kpc']:>5.0f}"
          f"{r['sigma_par_kpc']:>6.0f}{r['sign']:>4}"
          f"{r['amp_for_target']:>10.2e}{r['B_cl_at_target'][2]:>14.3f}"
          f"{r['B_cl_isotropic_part'][2]:>20.3f}"
          f"{r['B_cl_at_target'][2]/r['B_cl_isotropic_part'][2]:>8.3f}"
          f"{r['aniso_over_iso_median']:>11.3f}")

hdr("SENSITIVITY: dB over the scanned amplitude range (monotone-invariance "
    "check)")
sw = np.array([r["dB_spread"] for r in W])
sc = np.array([r["dB_spread"] for r in C])
print(f"   well-network  median {np.median(sw):.3f}  min {sw.min():.3f}  "
      f"max {sw.max():.3f}   rows with zero spread: {int((sw == 0).sum())}")
print(f"   pair channels median {np.median(sc):.3f}  min {sc.min():.3f}  "
      f"max {sc.max():.3f}   rows with zero spread: {int((sc == 0).sum())}")

if "resolution_check" in d:
    hdr("RESOLUTION SENSITIVITY of the well-network boost at the amplitudes "
        "used")
    for row in d["resolution_check"]:
        s = row["shape"]
        print(f"   {s['family']} p={s['p']} q={s['q']} s={s['s']} "
              f"L={s['L_kpc']:.0f} kpc excl={int(s['exclude_nearest'])}  "
              f"gate={row['gate']} A_T={row['A_T']}")
        for n, k, b in zip(row["n"], row["k"], row["B"]):
            print(f"      n={n:4d}  k = " + " ".join(f"{v:.4f}" for v in k)
                  + "   B = " + " ".join(f"{v:.3f}" for v in b))
        b0, b1 = np.array(row["B"][-2]), np.array(row["B"][-1])
        print(f"      last refinement changes B by "
              f"{100*np.abs(b1/b0-1).max():.2f}%")

if "headline_3d" in d:
    hdr("FULL 3-D VERIFICATION of the selected points")
    for r in d["headline_3d"]:
        print(f"   {r['cand']}")
        print("      3-D  B = " + " ".join(f"{v:.3f}" for v in r["B_3d"]))
        print("      map  B = " + " ".join(f"{v:.3f}" for v in r["B_map"]))
        print("      3D/map = " + " ".join(f"{v:.3f}" for v in r["ratio"]))
        print("      projected deflection ratio at "
              + " ".join(f"{int(x)}" for x in r["R_proj_kpc"]) + " kpc: "
              + " ".join(f"{v:.3f}" for v in r["B_deflection"]))

hdr("THE THREE REQUIREMENTS APPLIED IN SEQUENCE")
print("   shape = B(1414)/B(300); the measured X-COP excess is flat to ~20%,")
print("   so 0.7 < shape < 1.4 is the loosest defensible profile cut.")
for nm, rows in (("well-network", W), ("pair channels", C)):
    n0 = len(rows)
    r1 = [r for r in rows if r["reaches_target"]]
    r2 = [r for r in r1 if r["field_dex"] < M["gal_tol_dex"]]
    r3 = [r for r in r2 if r["member_dex"] < M["gal_tol_dex"]]
    r4 = [r for r in r3 if 0.7 < r["shape"] < 1.4]
    r5 = [r for r in r4 if all(M["band"][0] <= b <= M["band"][1]
                               for b in r["B_cl_at_target"])]
    print(f"   {nm:<14} scanned {n0:>5} -> reach B=2 {len(r1):>5} -> "
          f"field ok {len(r2):>5} -> member ok {len(r3):>5} -> "
          f"flat profile {len(r4):>5} -> whole B profile in band {len(r5):>5}")
    for r in sorted(r5, key=lambda r: r["field_dex"])[:5]:
        keys = {k: v for k, v in r.items()
                if k in ("family", "p", "q", "s", "L_kpc", "exclude_nearest",
                         "gate", "mode", "sigma_perp_kpc", "sigma_par_kpc",
                         "sign")}
        print(f"        {keys}  amp={r['amp_for_target']:.3g}  B="
              + " ".join(f"{v:.3f}" for v in r["B_cl_at_target"])
              + f"  fld {r['field_dex']:.3f}  mem {r['member_dex']:.3f}")

if "lane12" in d:
    L12 = d["lane12"]
    hdr("AGAINST THE PROGRAMME'S OWN MEASURED RADIAL REQUIREMENT (lane 12)")
    print("   a0 enhancement from lensing alone, three independent samples;")
    print("   deep-MOND g = sqrt(g_N a0) so the required field boost is")
    print("   B = sqrt(a0 enhancement).  Required B at "
          + " ".join(f"{int(x)}" for x in M["radii_kpc"]) + " kpc: "
          + " ".join(f"{v:.3f}" for v in L12["B_required_at_RADII"]))
    print("   " + L12["note"])
    print(f"\n{'tensor':<14}{'best rms(dex)':>15}{'rows < 0.10 dex':>17}"
          f"{'... and both galaxies ok':>26}")
    for nm, rows in (("well-network", W), ("pair channels", C)):
        r1 = [r for r in rows if r["rms_dex_vs_lane12"] < 0.10]
        r2 = [r for r in r1 if r["field_dex_at_shape"] < M["gal_tol_dex"]
              and r["member_dex_at_shape"] < M["gal_tol_dex"]]
        print(f"{nm:<14}{min(r['rms_dex_vs_lane12'] for r in rows):>15.4f}"
              f"{len(r1):>17}{len(r2):>26}")
    print()
    for nm, rows in (("well-network", W), ("pair channels", C)):
        print(f"   best five shape matches, {nm}:")
        for r in sorted(rows, key=lambda r: r["rms_dex_vs_lane12"])[:5]:
            keys = {k: v for k, v in r.items()
                    if k in ("family", "p", "q", "s", "L_kpc",
                             "exclude_nearest", "gate", "mode",
                             "sigma_perp_kpc", "sigma_par_kpc", "sign")}
            print(f"      rms {r['rms_dex_vs_lane12']:.4f} dex  amp="
                  f"{r['amp_for_shape']:.4g}  B="
                  + " ".join(f"{v:.3f}" for v in r["B_cl_at_shape"])
                  + f"  fld {r['field_dex_at_shape']:.3f}"
                  f"  mem {r['member_dex_at_shape']:.3f}")
            print(f"         {keys}")

hdr("SUMMARY")
for k, v in d["summary"].items():
    print(f"   {k:<34} {v}")
