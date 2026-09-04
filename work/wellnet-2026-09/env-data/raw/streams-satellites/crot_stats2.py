"""Per-galaxy counts for the long-slit gas+stars series and the polar-ring HI/CO
catalogues: how many DISTINCT galaxies have BOTH components kinematically
measured."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from crot_parse import parse_vizier
from collections import Counter

D = os.path.dirname(os.path.abspath(__file__))
R = {}


def tabs(short):
    return parse_vizier(os.path.join(D, "crot_%s.raw.tsv" % short))[2]


def gals(t, col):
    names = [c["name"] for c in t["columns"]]
    if col not in names:
        return set()
    i = names.index(col)
    return set(r[i].strip() for r in t["rows"] if r[i].strip())


print("=" * 78)
print("LONG-SLIT GAS + STARS SERIES (Corsini / Pizzella / Vega Beltran / Sarzi)")
print("=" * 78)
SERIES = [
 ("cp_corsini1999",   "gaskin",   "stelkin",  "Galaxy", "Galaxy", "Corsini+ 1999 A&A 342,671"),
 ("cp_vegabeltran2001","table6",  "table5",   "Name",   "Name",   "Vega Beltran+ 2001 A&A 374,394"),
 ("cp_corsini2003",   "table3",   "table4",   "Name",   "Name",   "Corsini+ 2003 A&A 408,873"),
 ("cp_pizzella2004",  "table3",   "table4",   "NGC",    "NGC",    "Pizzella+ 2004 A&A 424,447"),
]
tot_both = 0
for short, gtab, stab, gcol, scol, ref in SERIES:
    ts = tabs(short)
    g = [t for t in ts if gtab in t["name"]]
    s = [t for t in ts if stab in t["name"]]
    if not g or not s:
        print("  %-30s TABLE NOT FOUND (%s / %s)" % (ref, gtab, stab)); continue
    G, S = gals(g[0], gcol), gals(s[0], scol)
    both = G & S
    tot_both += len(both)
    print("  %-34s gas-kin galaxies=%-3d stellar-kin galaxies=%-3d BOTH=%-3d  (%d+%d data points)"
          % (ref, len(G), len(S), len(both), g[0]["nrows"], s[0]["nrows"]))
    R[short] = {"ref": ref, "n_gal_gas_kin": len(G), "n_gal_stellar_kin": len(S),
                "n_gal_BOTH": len(both), "galaxies_both": sorted(both),
                "n_gas_datapoints": g[0]["nrows"], "n_stellar_datapoints": s[0]["nrows"]}

# Corsini 2002 (NGC 2855) and Sarzi 2000 (NGC 4672) are single-galaxy papers
ts = tabs("cp_corsini2002")
print("  %-34s single galaxy NGC 2855: stellar table3=%d rows, gas table4=%d rows  BOTH=1"
      % ("Corsini+ 2002 A&A 382,488", ts[0]["nrows"], ts[1]["nrows"]))
R["cp_corsini2002"] = {"ref": "Corsini+ 2002 (NGC 2855)", "n_gal_BOTH": 1,
                       "n_rows_t3": ts[0]["nrows"], "n_rows_t4": ts[1]["nrows"]}
ts = tabs("cp_sarzi2000")
names = [c["name"] for c in ts[0]["columns"]]
ki = names.index("Kin"); ax = names.index("Axis")
print("  %-34s single galaxy NGC 4672: %d rows; Kin=%s ; Axis=%s  BOTH=1"
      % ("Sarzi+ 2000 A&A 360,439", ts[0]["nrows"],
         dict(Counter(r[ki].strip() for r in ts[0]["rows"])),
         dict(Counter(r[ax].strip() for r in ts[0]["rows"]))))
R["cp_sarzi2000"] = {"ref": "Sarzi+ 2000 (NGC 4672)", "n_gal_BOTH": 1,
                     "n_rows": ts[0]["nrows"],
                     "kin_counts": dict(Counter(r[ki].strip() for r in ts[0]["rows"])),
                     "axis_counts": dict(Counter(r[ax].strip() for r in ts[0]["rows"]))}
tot_both += 2
print("\n  TOTAL distinct galaxies with BOTH gas and stellar long-slit kinematics: %d" % tot_both)
R["longslit_total_both"] = tot_both

print("\n" + "=" * 78)
print("POLAR-RING CATALOGUES: which have RING and/or HOST kinematics")
print("=" * 78)
ts = tabs("prg_hi_vandriel2002")
names = [c["name"] for c in ts[0]["columns"]]
rows = [dict(zip(names, r)) for r in ts[0]["rows"]]
n_vopt = sum(1 for r in rows if r["Vopt"].strip())
n_vhi = sum(1 for r in rows if r["VHI"].strip())
n_w50 = sum(1 for r in rows if r["W50"].strip())
n_both = sum(1 for r in rows if r["Vopt"].strip() and r["VHI"].strip())
print("van Driel+ 2002 (n=%d): Vopt(optical systemic)=%d  VHI=%d  W50(HI linewidth)=%d  both Vopt&VHI=%d"
      % (len(rows), n_vopt, n_vhi, n_w50, n_both))
print("   NOTE: W50/W20 are GLOBAL single-dish HI linewidths - they are an integrated")
print("   kinematic AMPLITUDE for the HI, NOT a resolved ring rotation curve, and they")
print("   do NOT separate ring gas from host gas. Type column:",
      dict(Counter(r["Type"].strip() for r in rows)))
R["vandriel2002_detail"] = {"n": len(rows), "n_Vopt": n_vopt, "n_VHI": n_vhi,
    "n_W50": n_w50, "n_both_Vopt_VHI": n_both,
    "type_counts": dict(Counter(r["Type"].strip() for r in rows)),
    "resolved_ring_rotation_curve": False}

ts = tabs("prg_hi_huchtmeier1997")
names = [c["name"] for c in ts[0]["columns"]]
rows = [dict(zip(names, r)) for r in ts[0]["rows"]]
n_hrv = sum(1 for r in rows if r["HRV"].strip())
n_vhi = sum(1 for r in rows if r["VHI"].strip())
n_dv = sum(1 for r in rows if r["dv20"].strip())
print("Huchtmeier 1997 table1 (n=%d): HRV(optical)=%d  VHI=%d  dv20(HI width)=%d"
      % (len(rows), n_hrv, n_vhi, n_dv))
R["huchtmeier1997_detail"] = {"n": len(rows), "n_HRV": n_hrv, "n_VHI": n_vhi,
    "n_dv20": n_dv, "resolved_ring_rotation_curve": False}

ts = tabs("prg_co_combes2013")
names = [c["name"] for c in ts[0]["columns"]]
rows = [dict(zip(names, r)) for r in ts[0]["rows"]]
conf = [r for r in rows if "C" in dict(zip(names, [x for x in r]))["n_SPRC"]] if False else \
       [dict(zip(names, r)) for r in ts[0]["rows"] if "C" in dict(zip(names, r))["n_SPRC"]]
print("Combes+ 2013 (n=%d): kinematically confirmed polar rings = %d  -> SPRC %s"
      % (len(rows), len(conf), ", ".join(c["SPRC"].strip() for c in conf)))
R["combes2013_detail"] = {"n": len(rows), "n_confirmed": len(conf),
    "confirmed_SPRC": [c["SPRC"].strip() for c in conf]}

with open(os.path.join(D, "crot_component_pair_counts.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=2)
print("\nWROTE crot_component_pair_counts.json")
