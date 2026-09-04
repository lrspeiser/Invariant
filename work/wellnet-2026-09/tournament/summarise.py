"""tournament.json + gates.json + focus.json -> the tables in REPORT.md."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "tournament.json")))
R = J["records"]
G = json.load(open(os.path.join(HERE, "gates.json")))
F = None
if os.path.exists(os.path.join(HERE, "focus.json")):
    F = json.load(open(os.path.join(HERE, "focus.json")))


def hdr(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def four_channel(rows, title, n=25):
    hdr(title)
    print(f"{'candidate':<50}{'k':>2}{'a0/1e-10':>9}{'A':>9}"
          f"{'radial':>8}{'B_z':>7}{'h_as':>7}{'chi2h':>7}"
          f"{'clus':>7}{'B1Mpc':>7}{'fld':>7}{'mem':>7}{'J':>7}")
    print("-" * 130)
    for r in rows[:n]:
        print(f"{r['name'][:50]:<50}{r['n_params']:>2}"
              f"{r['a0']*1e10:>9.3f}{r['A']:>9.2f}"
              f"{r['radial_rms_dex']:>8.3f}{r['vert_Bz']:>7.3f}"
              f"{r['vert_h_as']:>7.2f}{r['vert_h_chi2dof']:>7.1f}"
              f"{r['cluster_rms_dex']:>7.3f}{r['cluster_B_1Mpc']:>7.2f}"
              f"{r['field_dex']:>7.3f}{r['member_dex']:>7.3f}{r['J']:>7.3f}")


hdr("FUNNEL")
f = J["funnel"]
print(f"candidates {f['candidates']}, evaluated {f['evaluated']}, "
      f"errors {f['errors']}, survivors {f['survivors']}")
print(f"\n{'screen':<20}{'kills alone':>12}{'unique kills':>14}"
      f"{'seq before':>12}{'seq after':>11}")
for k, v in f.items():
    if isinstance(v, dict) and "kills_alone" in v:
        print(f"{k:<20}{v['kills_alone']:>12}{v['unique_kills']:>14}"
              f"{v['sequential']['before']:>12}{v['sequential']['after']:>11}")

hdr("PER-CHANNEL WINNERS (none of them is the joint winner)")
for k, v in f["per_channel_winner"].items():
    print(f"{v['label']:<26}{v['winner'][:50]:<52}{v['value']:>9.4f}"
          f"   survives={v['survives']}  fails={','.join(v['failed'])[:44]}")

surv = [r for r in R if r["survives"]]
surv.sort(key=lambda r: r["J"])
four_channel(surv, f"SURVIVORS OF ALL SEVEN SCREENS ({len(surv)})", 40)

ref = [r for r in R if r["name"].startswith("BASE_")]
four_channel(ref, "REFERENCE ROWS: the base laws with no response")

sc = sorted([r for r in R if r["struct"] == "scalar_a0"],
            key=lambda r: r["J"])
four_channel(sc, "THE SCALAR COMPETITOR  a0 -> a0 (1 + A W)  -- best 12 by J", 12)
scm = sorted([r for r in R if r["struct"] == "scalar_a0"],
             key=lambda r: r["member_dex"])
four_channel(scm, "THE SCALAR COMPETITOR -- best 8 by MEMBER-GALAXY violation", 8)

for st in ("iso_K", "tensor_d", "tensor_T", "tensor_S"):
    rows = sorted([r for r in R if r["struct"] == st], key=lambda r: r["J"])
    four_channel(rows, f"BEST BY J: {st}", 6)

hdr("SELECTION")
print(json.dumps(J["selection"], indent=1)[:2200])

hdr("MOMENTUM AND COARSE GRAINING ON THE SHORTLIST")
print(f"{'candidate':<52}{'F/Fref':>9}{'base null':>11}{'gradK':>9}"
      f"{'surface':>10}{'cg drift':>10}")
for r in R:
    if "momentum" in r:
        m, c = r["momentum"], r.get("coarse_grain", {})
        gk = m.get("gradK_term_rel")
        sf = m.get("surface_term_rel")
        print(f"{r['name'][:52]:<52}"
              f"{m.get('excess', float('nan')):>9.3f}"
              f"{m.get('base_null_rel', float('nan')):>11.4f}"
              f"{(('%.3f' % gk) if gk is not None else '-'):>9}"
              f"{(('%.4f' % sf) if sf is not None else '-'):>10}"
              f"{c.get('max_drift', 0.0):>10.4f}")

hdr("DOES THE VERTICAL CHANNEL SEE ANYTHING THE GATE DOES?")
gal_ok = [r for r in R if r["screens"]["H2_field_galaxy"]["pass"]
          and r["screens"]["H3_member_galaxy"]["pass"]]
base = {r["base"]: r for r in ref}
for b in ("rar", "aqual"):
    sub = [r for r in gal_ok if r["base"] == b]
    if not sub:
        continue
    h = np.array([r["vert_h_as"] for r in sub])
    print(f"base {b}: {len(sub)} candidates passing both galaxy screens; "
          f"h_sigma_LOS spans {h.min():.2f} to {h.max():.2f} arcsec "
          f"(base-only law {base[b]['vert_h_as']:.2f})")

hdr("SHELL AVERAGE, RESPONSIVENESS, W CEILING")
print(json.dumps(G["shell_average_bracket_dex"], indent=1))
for k, v in G["responsiveness"].items():
    print(f"  {k:<20} spread {v['spread']:.6g}  responsive={v['responsive']}")
print("W ceiling hits:", G["W_ceiling_hits"])

hdr("REPRODUCTION OF THE LANES THIS ONE REUSES")
print(json.dumps(G["reproduction_of_prior_lanes"], indent=1))

hdr("|Phi_N| BOUNDARY RULE (gates.json)")
print(json.dumps(G["phi_boundary_rule"], indent=1)[:1800])

if "member_realisation_scatter" in G:
    hdr("MEMBER SCREEN vs MEMBER REALISATION (gates.json)")
    print(json.dumps(G["member_realisation_scatter"], indent=1)[:2000])

if F:
    hdr("FOCUS: head-to-head, scalar vs tensor at matched gates")
    hh = F["head_to_head"]
    print(f"{'gate':<22}{'base':<6}{'structure':<12}{'well':<28}"
          f"{'radial':>8}{'clus':>7}{'B1Mpc':>7}{'fld':>8}{'mem':>8}"
          f"{'h_as':>7}{'B_z':>7}{'J':>7}")
    for h in hh:
        print(f"{h['gate']:<22}{h['base']:<6}{h['structure']:<12}"
              f"{str(h['well'])[:27]:<28}{h['radial_rms_dex']:>8.3f}"
              f"{h['cluster_rms_dex']:>7.3f}{h['cluster_B_1Mpc']:>7.2f}"
              f"{h['field_dex']:>8.4f}{h['member_dex']:>8.4f}"
              f"{h['vert_h_as']:>7.2f}{h['vert_Bz']:>7.3f}{h['J']:>7.3f}")
    hdr("FOCUS: member escape vs weight family")
    print(json.dumps(F["member_escape_vs_weight_family"], indent=1))
    hdr("FOCUS: momentum vs resolution")
    for m in F["momentum_vs_resolution"]:
        print(f"  {m['structure']:<12} A={m['A']:>7.2f} n={m['n']:>3} "
              f"excess {m['excess']:.4f}  base null {m['base_null']:.5f}  "
              f"gradK {m['gradK']}")
    hdr("FOCUS: phi boundary rule")
    print(json.dumps(F["phi_boundary_rule"], indent=1)[:2600])
    hdr("FOCUS: member realisation scatter and cluster resolution")
    print(json.dumps(F["member_realisation_scatter"], indent=1)[:1600])
    print(json.dumps(F["cluster_resolution"], indent=1)[:1200])
