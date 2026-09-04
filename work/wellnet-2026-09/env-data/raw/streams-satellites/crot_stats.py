"""Compute, for every acquired catalogue, the numbers the brief asks for:
 (a) how many objects have BOTH components kinematically measured,
 (b) the measured misalignment-angle distribution,
 (c) EXACT counts of the ~90-degree POLAR subsets.
Emits crot_polar_and_counterrotator_summary.json + cleaned TSVs of the
high-value subsets."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest
from crot_parse import parse_vizier
from collections import Counter

D = os.path.dirname(os.path.abspath(__file__))
R = {}


def T(short, table_substr=None):
    """Return (columns, rows-as-dicts) for one table of a raw VizieR file."""
    path = os.path.join(D, "crot_%s.raw.tsv" % short)
    cat, title, tables = parse_vizier(path)
    if table_substr:
        tables = [t for t in tables if table_substr in t["name"]]
    t = tables[0]
    names = [c["name"] for c in t["columns"]]
    rows = [dict(zip(names, r)) for r in t["rows"]]
    return t, names, rows


def fl(x):
    try:
        v = float(str(x).strip())
        return v
    except Exception:
        return None


def fold90(d):
    """Fold a PA difference to 0..90 (undirected axis difference)."""
    d = abs(d) % 180.0
    return 180.0 - d if d > 90.0 else d


def fold180(d):
    """Fold to 0..180 (keeps counter-rotation at ~180 distinct from polar at ~90)."""
    return abs(d) % 360.0 if abs(d) % 360.0 <= 180 else 360.0 - (abs(d) % 360.0)


def hist(vals, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(("%d-%d" % (a, b), sum(1 for v in vals if a <= v < b)))
    return out


# =========================================================== Raimundo 2023 ===
print("=" * 78)
print("RAIMUNDO+ 2023  J/other/NatAs/7.463  (SAMI DR3 stellar vs gas kinematic PAs)")
t, names, rows = T("kinangles_raimundo2023")
print("  total rows: %d" % len(rows))
both = [r for r in rows if fl(r["PAs"]) is not None and fl(r["PAg"]) is not None]
dpa = [r for r in rows if fl(r["DPA"]) is not None]
print("  rows with BOTH PAstellar and PAgas measured : %d" % len(both))
print("  rows with DPA (misalignment angle) measured : %d" % len(dpa))
vals = [fl(r["DPA"]) for r in dpa]
print("  DPA range: %.1f .. %.1f deg" % (min(vals), max(vals)))
print("  DPA histogram (deg):")
for lab, n in hist(vals, list(range(0, 181, 15))):
    print("     %-8s %4d" % (lab, n))
# polar definitions
pol30 = [r for r in dpa if abs(fl(r["DPA"]) - 90) <= 30]
pol20 = [r for r in dpa if abs(fl(r["DPA"]) - 90) <= 20]
pol10 = [r for r in dpa if abs(fl(r["DPA"]) - 90) <= 10]
cr30 = [r for r in dpa if fl(r["DPA"]) >= 150]
al30 = [r for r in dpa if fl(r["DPA"]) <= 30]
print("  ALIGNED    DPA <= 30 deg              : %d" % len(al30))
print("  POLAR      |DPA-90| <= 30 deg (60-120): %d" % len(pol30))
print("  POLAR      |DPA-90| <= 20 deg (70-110): %d" % len(pol20))
print("  POLAR      |DPA-90| <= 10 deg (80-100): %d" % len(pol10))
print("  COUNTER-ROT DPA >= 150 deg            : %d" % len(cr30))
# significance: require DPA larger than its own 3-sigma error
sig = [r for r in dpa if fl(r["e_DPA"]) is not None
       and abs(fl(r["DPA"]) - 90) <= 30 and fl(r["e_DPA"]) <= 30]
print("  POLAR |DPA-90|<=30 AND 3sigma err <= 30 deg : %d" % len(sig))
R["raimundo2023"] = {"catalog": "J/other/NatAs/7.463", "n_rows": len(rows),
    "n_both_components_kinematically_measured": len(both), "n_with_DPA": len(dpa),
    "n_aligned_le30": len(al30), "n_polar_60_120": len(pol30),
    "n_polar_70_110": len(pol20), "n_polar_80_100": len(pol10),
    "n_counterrot_ge150": len(cr30), "n_polar_60_120_err3sig_le30": len(sig),
    "dpa_histogram_15deg": hist(vals, list(range(0, 181, 15)))}
# write the polar subset
out = os.path.join(D, "crot_raimundo2023_POLAR_subset.tsv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(names) + "\n")
    for r in sorted(pol30, key=lambda x: abs(fl(x["DPA"]) - 90)):
        fh.write("\t".join(r.get(n, "") for n in names) + "\n")
print("  WROTE %s (%d rows)" % (os.path.basename(out), len(pol30)))
R["raimundo2023"]["polar_subset_file"] = os.path.basename(out)

# =========================================================== Bevacqua 2022 ===
print("\n" + "=" * 78)
print("BEVACQUA+ 2022  J/MNRAS/511/139  (MaNGA counter-rotating disc candidates)")
t, names, rows = T("manga_crd_bevacqua2022", "table1")
print("  table1 rows (CRD candidates): %d" % len(rows))
dv = [fl(r["DPA"]) for r in rows if fl(r["DPA"]) is not None]
print("  rows with DPA measured: %d   range %.1f .. %.1f deg" % (len(dv), min(dv), max(dv)))
print("  DPA histogram (deg):")
for lab, n in hist(dv, list(range(0, 181, 15))):
    print("     %-8s %4d" % (lab, n))
bpol = [r for r in rows if fl(r["DPA"]) is not None and abs(fl(r["DPA"]) - 90) <= 30]
bpol20 = [r for r in rows if fl(r["DPA"]) is not None and abs(fl(r["DPA"]) - 90) <= 20]
bcr = [r for r in rows if fl(r["DPA"]) is not None and fl(r["DPA"]) >= 150]
print("  POLAR |DPA-90|<=30 : %d" % len(bpol))
print("  POLAR |DPA-90|<=20 : %d" % len(bpol20))
print("  COUNTER-ROT DPA>=150: %d" % len(bcr))
print("  n_DPA note flags:", dict(Counter(r.get("n_DPA", "") for r in rows)))
print("  Pop (gas corotates with young/old disc):", dict(Counter(r.get("Pop", "") for r in rows)))
t2, n2, rows2 = T("manga_crd_bevacqua2022", "tableg18")
print("  parent table g18 rows: %d" % len(rows2))
R["bevacqua2022"] = {"catalog": "J/MNRAS/511/139", "n_CRD_candidates": len(rows),
    "n_parent_g18": len(rows2), "n_with_DPA": len(dv),
    "n_polar_60_120": len(bpol), "n_polar_70_110": len(bpol20),
    "n_counterrot_ge150": len(bcr),
    "dpa_histogram_15deg": hist(dv, list(range(0, 181, 15)))}

# ============================================================ Ristea 2024 ===
print("\n" + "=" * 78)
print("RISTEA+ 2024  J/MNRAS/527/7438  (MaNGA: stellar AND gas rotation curves)")
t, names, rows = T("manga_kincat_ristea2024")
print("  total rows: %d" % len(rows))
for rad in ("1Re", "1.3Re", "2Re"):
    s = "VelST%s" % rad; g = "VelG%s" % rad
    ns = sum(1 for r in rows if fl(r[s]) is not None)
    ng = sum(1 for r in rows if fl(r[g]) is not None)
    nb = sum(1 for r in rows if fl(r[s]) is not None and fl(r[g]) is not None)
    print("   %-6s stellar V: %4d   gas V: %4d   BOTH: %4d" % (rad, ns, ng, nb))
    R.setdefault("ristea2024", {})["n_both_%s" % rad.replace(".", "p")] = nb
    R["ristea2024"]["n_stellar_%s" % rad.replace(".", "p")] = ns
    R["ristea2024"]["n_gas_%s" % rad.replace(".", "p")] = ng
nb1 = sum(1 for r in rows if fl(r["VelST1Re"]) is not None and fl(r["VelG1Re"]) is not None)
R["ristea2024"].update({"catalog": "J/MNRAS/527/7438", "n_rows": len(rows),
    "n_both_components_kinematically_measured_1Re": nb1,
    "has_misalignment_angle_column": False})
print("  NOTE: this catalogue has NO misalignment-angle column - it gives the two")
print("        rotation AMPLITUDES, not the angle between the two spin axes.")
print("  Sample flag:", dict(Counter(r.get("Sample", "") for r in rows)))

# ======================================================== SPRC / polar rings ===
print("\n" + "=" * 78)
print("MOISEEV+ 2011  J/MNRAS/418/244  SDSS-based Polar Ring Catalogue (SPRC)")
t, names, rows = T("sprc_moiseev2011")
print("  total rows: %d" % len(rows))
tc = Counter(r["Type"].strip() for r in rows)
print("  Type breakdown (B=best, G=good, R=related, P=possible face-on ring):", dict(tc))
ncz = sum(1 for r in rows if fl(r["cz"]) is not None)
print("  with heliocentric cz: %d" % ncz)
print("  *** SPRC gives NO kinematics of the ring and NO kinematics of the host. ***")
R["sprc_moiseev2011"] = {"catalog": "J/MNRAS/418/244", "n_rows": len(rows),
    "type_counts": dict(tc), "n_with_cz": ncz,
    "has_ring_kinematics": False, "has_host_kinematics": False}

print("\nCOMBES+ 2013  J/A+A/554/A11  CO observations of polar ring galaxies")
t, names, rows = T("prg_co_combes2013", "table1")
print("  total rows: %d" % len(rows))
conf = [r for r in rows if "C" in r.get("n_SPRC", "")]
print("  kinematically CONFIRMED polar rings (n_SPRC='C'): %d" % len(conf))
print("  SPRC numbers: %s" % ", ".join(r["SPRC"] for r in rows))
print("  confirmed SPRC numbers: %s" % ", ".join(r["SPRC"] for r in conf))
R["combes2013"] = {"catalog": "J/A+A/554/A11", "n_rows": len(rows),
    "n_kinematically_confirmed": len(conf),
    "confirmed_SPRC_ids": [r["SPRC"] for r in conf],
    "all_SPRC_ids": [r["SPRC"] for r in rows]}

print("\nHUCHTMEIER 1997  J/A+A/319/401  HI survey of polar ring galaxies II")
t, names, rows = T("prg_hi_huchtmeier1997", "table1")
print("  table1 rows: %d   cols: %s" % (len(rows), names))
R["huchtmeier1997"] = {"catalog": "J/A+A/319/401", "n_rows_table1": len(rows)}

print("\nVAN DRIEL+ 2002  J/A+A/386/140  HI survey of polar ring galaxies IV")
t, names, rows = T("prg_hi_vandriel2002")
print("  rows: %d" % len(rows))
print("  cols: %s" % names)
R["vandriel2002"] = {"catalog": "J/A+A/386/140", "n_rows": len(rows), "columns": names}

# ============================================================== ATLAS3D II ===
print("\n" + "=" * 78)
print("KRAJNOVIC+ 2011 ATLAS3D II (arXiv:1102.3801) kinematic classification")
p = os.path.join(D, "crot_atlas3d_II_krajnovic2011_kinclass.tsv")
lines = open(p, encoding="utf-8").read().splitlines()
hdr = lines[0].split("\t"); rows = [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]
print("  rows: %d" % len(rows))
kc = Counter(r["KinStruct"] for r in rows); gc = Counter(r["Group"] for r in rows)
n2s = sum(1 for r in rows if "2s" in r["KinStruct"])
nkdc = sum(1 for r in rows if "KDC" in r["KinStruct"])
ncrc = sum(1 for r in rows if "CRC" in r["KinStruct"])
psis = [fl(r["Psi"]) for r in rows if fl(r["Psi"]) is not None]
print("  2-sigma (counter-rotating disc): %d   KDC: %d   CRC: %d   KDC+CRC: %d"
      % (n2s, nkdc, ncrc, nkdc + ncrc))
print("  Psi (STELLAR kin axis vs PHOTOMETRIC axis) >=75 deg: %d ; >=80: %d"
      % (sum(1 for p_ in psis if p_ >= 75), sum(1 for p_ in psis if p_ >= 80)))
R["atlas3d_II"] = {"source": "arXiv:1102.3801", "n_rows": len(rows),
    "n_2sigma_counterrotating_disc": n2s, "n_KDC": nkdc, "n_CRC": ncrc,
    "n_KDC_plus_CRC": nkdc + ncrc, "kinstruct_counts": dict(kc),
    "group_counts": dict(gc),
    "n_Psi_ge75": sum(1 for p_ in psis if p_ >= 75),
    "n_Psi_ge80": sum(1 for p_ in psis if p_ >= 80),
    "psi_is_stellar_vs_photometric_NOT_gas_vs_stars": True}
# 2-sigma + KDC/CRC subset file
out = os.path.join(D, "crot_atlas3d_II_2sigma_KDC_CRC_subset.tsv")
sub = [r for r in rows if ("2s" in r["KinStruct"] or "KDC" in r["KinStruct"]
                           or "CRC" in r["KinStruct"])]
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(hdr) + "\n")
    for r in sub:
        fh.write("\t".join(r[h] for h in hdr) + "\n")
print("  WROTE %s (%d rows: 2sigma + KDC + CRC)" % (os.path.basename(out), len(sub)))
R["atlas3d_II"]["subset_file"] = os.path.basename(out)
R["atlas3d_II"]["n_subset_2sigma_KDC_CRC"] = len(sub)

# ======================================= CALIFA gas PA vs photometric PA ===
print("\n" + "=" * 78)
print("GARCIA-LORENZO+ 2015  J/A+A/573/A59  (CALIFA ionised-gas velocity fields)")
t, names, rows = T("califa_gaskin_garcialorenzo2015")
print("  rows: %d" % len(rows))
n_pa1 = sum(1 for r in rows if fl(r["PA1"]) is not None)
n_rec = sum(1 for r in rows if fl(r["PArec"]) is not None)
n_app = sum(1 for r in rows if fl(r["PAapp"]) is not None)
print("  with morphological PA1: %d ; gas PA_rec: %d ; gas PA_app: %d"
      % (n_pa1, n_rec, n_app))
# gas kinematic PA vs morphological PA -> misalignment
mis = []
for r in rows:
    pm_ = fl(r["PA1"]); pk = fl(r["PArec"])
    if pm_ is None or pk is None:
        continue
    mis.append((r["Name"], fold90(pk - pm_)))
print("  galaxies with BOTH morphological PA and gas kinematic PA: %d" % len(mis))
mv = [m[1] for m in mis]
print("  |PA_gas,kin - PA_phot| folded to 0-90, histogram:")
for lab, n in hist(mv, list(range(0, 91, 15))):
    print("     %-8s %4d" % (lab, n))
print("  >=60 deg: %d   >=75 deg: %d"
      % (sum(1 for v in mv if v >= 60), sum(1 for v in mv if v >= 75)))
R["garcialorenzo2015"] = {"catalog": "J/A+A/573/A59", "n_rows": len(rows),
    "n_with_morph_and_gaskin_PA": len(mis),
    "n_gas_phot_misalign_ge60": sum(1 for v in mv if v >= 60),
    "n_gas_phot_misalign_ge75": sum(1 for v in mv if v >= 75),
    "note": ("PA1 is a MORPHOLOGICAL (photometric) PA, not a stellar KINEMATIC PA, "
             "so this is gas-kinematics-vs-photometry, a weaker constraint than a "
             "gas-vs-stellar-kinematics misalignment.")}

# ============================================================ Gasymov 2025 ===
print("\n" + "=" * 78)
print("GASYMOV+ 2025  J/ApJS/281/19  (MaNGA stellar counter-rotation)")
for tn in ("table2", "table3", "table9"):
    t, names, rows = T("manga_counterrot_gasymov2025", tn)
    print("  %-8s rows: %4d" % (tn, len(rows)))
    if tn in ("table2", "table3"):
        print("      CRConfig:", dict(Counter(r.get("CRConfig", "").strip() for r in rows)))
    R.setdefault("gasymov2025", {})["n_%s" % tn] = len(rows)
R["gasymov2025"]["catalog"] = "J/ApJS/281/19"
R["gasymov2025"]["has_misalignment_angle_column"] = False

with open(os.path.join(D, "crot_polar_and_counterrotator_summary.json"),
          "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=2)
print("\nWROTE crot_polar_and_counterrotator_summary.json")
