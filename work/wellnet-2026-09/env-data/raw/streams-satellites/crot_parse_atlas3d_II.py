"""Transcribe the ATLAS3D II (Krajnovic+ 2011, MNRAS 414, 2923) kinematic
classification table verbatim from the arXiv LaTeX source into a TSV.

Row-count guard (the recorded failure mode: a table split across two table*
environments silently returned 59 of 100 rows). This source has exactly ONE
deluxetable and ONE \\startdata..\\enddata block; we assert the extracted count
against the paper's stated sample size of 260 ATLAS3D early-type galaxies.
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest

D = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(D, "crot_atlas3d_II_krajnovic2011_src", "krajnovic_A3D_kinmis.tex")
EXPECTED_N = 260   # ATLAS3D early-type galaxy sample (Cappellari+ 2011, ATLAS3D I)

txt = open(TEX, encoding="utf-8", errors="replace").read()

n_env = len(re.findall(r"\\begin\{deluxetable\}", txt))
n_start = len(re.findall(r"\\startdata", txt))
n_tabstar = len(re.findall(r"\\begin\{table\*\}", txt))
print("GUARD: deluxetable envs=%d  startdata blocks=%d  table* envs=%d"
      % (n_env, n_start, n_tabstar))
assert n_env == 1 and n_start == 1, "more than one table environment - re-check split tables"

body = txt.split(r"\startdata", 1)[1].split(r"\enddata", 1)[0]
raw_rows = [r for r in body.split("\\\\") if r.strip()]
print("GUARD: raw row fragments after splitting on '\\\\\\\\' = %d" % len(raw_rows))


def pm(cell):
    """'  18.0 $\\pm$   4.1' -> ('18.0','4.1');  plain -> (val,'')"""
    c = cell.strip()
    if "$\\pm$" in c:
        a, b = c.split("$\\pm$", 1)
        return a.strip(), b.strip()
    return c, ""


COLS = [
    ("Name",      "",      "Galaxy name"),
    ("PAphot",    "deg",   "Global photometric position angle of the stellar body"),
    ("e_PAphot",  "deg",   "Uncertainty on PAphot"),
    ("eps",       "",      "Ellipticity (1-b/a) of the stellar body"),
    ("e_eps",     "",      "Uncertainty on eps"),
    ("PAkin",     "deg",   "Global STELLAR kinematic position angle (from the mean stellar velocity map)"),
    ("e_PAkin",   "deg",   "Uncertainty on PAkin"),
    ("Psi",       "deg",   "Kinematic misalignment angle Psi between PAphot and PAkin (0-90)"),
    ("k5k1",      "",      "Mean k5/k1 kinemetry ratio (velocity-map regularity)"),
    ("e_k5k1",    "",      "Uncertainty on k5k1"),
    ("k1max",     "km/s",  "Maximum kinemetric rotation amplitude k1 within the SAURON FoV"),
    ("Morph",     "",      "Morphological feature flag: N none, B bar, R ring, BR bar+ring, S shells, I interaction"),
    ("Dust",      "",      "Dust/blue-nucleus flag: N none, D dust disc, F filaments, B blue nucleus"),
    ("KinStruct", "",      "Kinemetric type/feature: RR/NRR + NF, LV, CRC, KDC, 2s (2-sigma), 2m, KT"),
    ("Group",     "",      "Kinematic group a-f (a no rotation, b NRR featureless, c KDC/CRC, d 2-sigma, e RR, f unclassified)"),
]

rows = []
for r in raw_rows:
    r = r.strip()
    if not r or r.startswith("%"):
        continue
    cells = r.split("&")
    if len(cells) != 11:
        print("  !! unexpected cell count %d: %r" % (len(cells), r[:110]))
        continue
    name = cells[0].strip()
    paph, e_paph = pm(cells[1])
    eps,  e_eps = pm(cells[2])
    pakin, e_pakin = pm(cells[3])
    psi = cells[4].strip()
    k5k1, e_k5k1 = pm(cells[5])
    k1max = cells[6].strip()
    morph = cells[7].strip()
    dust = cells[8].strip()
    kin = cells[9].strip()
    grp = cells[10].strip()
    rows.append([name, paph, e_paph, eps, e_eps, pakin, e_pakin, psi,
                 k5k1, e_k5k1, k1max, morph, dust, kin, grp])

print("EXTRACTED %d rows (expected %d from ATLAS3D I sample size)" % (len(rows), EXPECTED_N))
assert len(rows) == EXPECTED_N, ("ROW COUNT MISMATCH: got %d expected %d"
                                 % (len(rows), EXPECTED_N))
names = [r[0] for r in rows]
assert len(set(names)) == len(names), "duplicate galaxy names"

out = os.path.join(D, "crot_atlas3d_II_krajnovic2011_kinclass.tsv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(c[0] for c in COLS) + "\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")
print("WROTE %s" % out)

# ---- the two subsamples the brief asks for ----
kin = [r[13] for r in rows]
grp = [r[14] for r in rows]
from collections import Counter
print("\nKinStruct counts:", dict(Counter(kin)))
print("Group counts:", dict(Counter(grp)))

n_2s  = sum(1 for k in kin if "2s" in k)
n_kdc = sum(1 for k in kin if "KDC" in k)
n_crc = sum(1 for k in kin if "CRC" in k)
n_grp_c = sum(1 for g in grp if g == "c")
n_grp_d = sum(1 for g in grp if g == "d")
print("\n2-sigma (counter-rotating disc) KinStruct : %d   [group d = %d]" % (n_2s, n_grp_d))
print("KDC                                       : %d" % n_kdc)
print("CRC (counter-rotating core)               : %d" % n_crc)
print("KDC+CRC                                   : %d   [group c = %d]" % (n_kdc + n_crc, n_grp_c))

# Psi distribution (STELLAR kinematics vs PHOTOMETRIC major axis)
psis = []
for r in rows:
    try:
        psis.append(float(r[7]))
    except ValueError:
        pass
psis.sort()
import statistics
print("\nPsi (stellar-kin vs phot major axis) n=%d  min=%.1f med=%.1f max=%.1f"
      % (len(psis), psis[0], statistics.median(psis), psis[-1]))
for lo in (5, 10, 15, 30, 45, 60, 75, 80, 85):
    print("   Psi > %2d deg : %3d" % (lo, sum(1 for p in psis if p > lo)))
print("   Psi >= 75 deg (near-polar STELLAR misalignment): %d"
      % sum(1 for p in psis if p >= 75))

write_manifest(
    out,
    source_url="https://arxiv.org/e-print/1102.3801",
    query=("GET https://arxiv.org/e-print/1102.3801 (ATLAS3D II, Krajnovic et al. 2011, "
           "MNRAS 414, 2923); verbatim transcription of the single deluxetable "
           "'Properties of ATLAS3D galaxies' (lines 1259-1580 of "
           "krajnovic_A3D_kinmis.tex, one \\startdata..\\enddata block)"),
    columns=[{"name": c[0], "unit": c[1], "description": c[2]} for c in COLS],
    row_count=len(rows),
    source_file_within_archive="krajnovic_A3D_kinmis.tex",
    measurement_or_model=(
        "MEASUREMENT. PAphot and eps are photometric measurements (SDSS/INT imaging "
        "moments). PAkin is measured from the SAURON mean STELLAR velocity map by "
        "kinemetry. Psi = |PAphot - PAkin| folded to 0-90 deg is derived arithmetically "
        "from two measurements. k5/k1 and k1max are kinemetric harmonic amplitudes of "
        "the observed velocity field. KinStruct/Group are classifications OF the "
        "measured velocity maps. NO dark-matter halo, Jeans/JAM model, or dynamical "
        "mass-to-light ratio enters any column of this table."),
    note=("ATLAS3D II is NOT in VizieR: Vizier.find_catalogs over the full description "
          "index returns ATLAS3D I, III, IV, VII, XXIII, XXIX, XXX, XXXI but no II. "
          "Transcribed from the arXiv LaTeX source instead. Row count 260 matches the "
          "ATLAS3D early-type galaxy sample size exactly; 260 unique galaxy names. "
          "CAUTION: Psi here compares the STELLAR kinematic axis to the PHOTOMETRIC "
          "axis of the same stellar body - it is NOT a gas-vs-star misalignment."),
    extra={"bibcode": "2011MNRAS.414.2923K",
           "arxiv": "1102.3801",
           "expected_row_count": EXPECTED_N,
           "row_count_check": "PASS (260 == 260)",
           "n_table_environments": n_env,
           "n_startdata_blocks": n_start,
           "subsample_2sigma_counterrotating_disc": n_2s,
           "subsample_KDC": n_kdc,
           "subsample_CRC_counterrotating_core": n_crc,
           "subsample_group_c_KDC_plus_CRC": n_grp_c,
           "subsample_group_d_2sigma": n_grp_d,
           "kinstruct_counts": dict(Counter(kin)),
           "group_counts": dict(Counter(grp))})
