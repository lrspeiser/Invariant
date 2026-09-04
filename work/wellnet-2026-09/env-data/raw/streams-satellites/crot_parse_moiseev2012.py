"""Transcribe Moiseev (2012) 'Inner Polar Rings and Disks: Observed Properties'
Table 1 -- galaxies with an inner polar structure (IPS), with the MEASURED angle
Delta_i between the inner polar structure and the host disc.

*** THIS IS THE BRIEF'S NAMED FAILURE MODE IN THE WILD. *** Table 1 is split
across TWO separate table*/tabular environments (the second carries
\\caption{(continue)} and a \\setcounter{table}{0}). Parsing only the first
tabular silently returns roughly half the rows. We parse BOTH and assert the
total against the paper's own statement: "In total, Table 1 lists 47 galaxies".
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest

D = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(D, "crot_moiseev2012_inner_polar_rings_src", "Moiseev_arxiv.tex")
EXPECTED_N = 47      # "In total, Table 1 lists 47 galaxies" (line 341 of the source)
EXPECTED_DI = 27     # "The angle Delta i was estimated for 27 objects" (line 685)

txt = open(TEX, encoding="utf-8", errors="replace").read()

blocks = re.findall(r"\\begin\{tabular\}\{l\|l\|l\|r\|r\|r\|r\|r\|r\|r\|r\|r\|r\|l\}(.*?)\\end\{tabular\}",
                    txt, re.S)
print("GUARD: found %d tabular blocks with Table-1's column spec "
      "(a single-block parse would have MISSED the rest)" % len(blocks))
assert len(blocks) == 2, "expected Table 1 to be split into exactly 2 blocks"


def strip_tex(c):
    c = c.strip()
    c = re.sub(r"\\cite[a-z]*\{[^}]*\}", "", c)
    c = re.sub(r"\\citet\{[^}]*\}", "", c)
    c = c.replace("\\degr", "").replace("$", "").replace("~", " ")
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    c = c.replace("{", "").replace("}", "")
    return " ".join(c.split())


COLS = [
 ("Name",   "",     "Galaxy name"),
 ("Type",   "",     "Morphological type (RC3/NED)"),
 ("T",      "",     "Numerical morphological T-type"),
 ("D",      "Mpc",  "Distance"),
 ("r_ang",  "arcsec", "Radius of the inner polar structure, angular"),
 ("r_kpc",  "kpc",  "Radius of the inner polar structure, linear"),
 ("PA0",    "deg",  "Position angle of the MAIN (host) galactic disc"),
 ("i0",     "deg",  "Inclination of the MAIN (host) galactic disc"),
 ("PAbar",  "deg",  "Position angle of the bar, where present"),
 ("PA1",    "deg",  "Position angle of the INNER POLAR structure"),
 ("i1",     "deg",  "Inclination of the INNER POLAR structure"),
 ("Delta_i","deg",  "MEASURED angle between the inner polar structure and the "
                    "main galactic disc (the misalignment angle). '*' marks a "
                    "value estimated from projected geometry only."),
 ("Comm",   "",     "Tracer of the inner polar structure: HII=ionised gas, "
                    "CO=molecular gas, HI=neutral hydrogen, s=stars; "
                    "r=rotation resolved, w=warp/weak"),
 ("Ref",    "",     "Literature reference(s)"),
]

rows = []
for bi, blk in enumerate(blocks):
    body = blk.split(r"\hline", 2)[-1]
    for raw in body.split("\\\\"):
        if "&" not in raw:
            continue
        cells = raw.split("&")
        if len(cells) != 14:
            continue
        vals = [strip_tex(c) for c in cells]
        if not vals[0] or vals[0].startswith("Name") or vals[0].startswith("(1)"):
            continue
        if vals[0] in ("", "\\hline"):
            continue
        rows.append(vals)
    print("   block %d -> running total %d rows" % (bi + 1, len(rows)))

print("\nEXTRACTED %d rows (paper states %d)" % (len(rows), EXPECTED_N))
assert len(rows) == EXPECTED_N, "ROW COUNT MISMATCH: got %d expected %d" % (len(rows), EXPECTED_N)
names = [r[0] for r in rows]
assert len(set(names)) == len(names), "duplicate galaxy names: %s" % (
    [n for n in names if names.count(n) > 1],)

out = os.path.join(D, "crot_moiseev2012_inner_polar_structures.tsv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(c[0] for c in COLS) + "\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")
print("WROTE %s" % out)

# --- Delta_i statistics: how many are ~90 deg POLAR ---
def parse_di(s):
    """'90', '90^*', '78-90', '55-70^*', '>90^*', '39, 86', '--    80'
    -> list of floats. NOTE the '^' left behind by TeX superscript stripping:
    forgetting it silently drops every starred (projected-estimate) value."""
    s = s.replace("*", "").replace("^", "").strip()
    if not s or set(s) <= set("- "):
        return []
    s = s.replace(">", "").replace("<", "")
    out = []
    for part in re.split(r"[,\s]+", s):
        part = part.strip()
        if not part or set(part) <= set("-"):
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", part)
        if m:                       # a range: use the midpoint
            out.append((float(m.group(1)) + float(m.group(2))) / 2.0)
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


have_di = [r for r in rows if parse_di(r[11])]
print("\nGalaxies with a measured Delta_i : %d  (paper states %d)"
      % (len(have_di), EXPECTED_DI))
di_max = {r[0]: max(parse_di(r[11])) for r in have_di}
n90_10 = sum(1 for v in di_max.values() if abs(v - 90) <= 10)
n90_15 = sum(1 for v in di_max.values() if abs(v - 90) <= 15)
n90_20 = sum(1 for v in di_max.values() if abs(v - 90) <= 20)
n_ge80 = sum(1 for v in di_max.values() if v >= 80)
print("  Delta_i within 10 deg of 90 (80-100) : %d" % n90_10)
print("  Delta_i within 15 deg of 90 (75-105) : %d" % n90_15)
print("  Delta_i within 20 deg of 90 (70-110) : %d" % n90_20)
print("  Delta_i >= 80 deg                    : %d" % n_ge80)
print("  sorted Delta_i: %s" % sorted(round(v, 1) for v in di_max.values()))

# tracer breakdown: which component is measured
from collections import Counter
tr = Counter()
for r in rows:
    c = r[12]
    for k in ("HII", "CO", "HI", "s"):
        if re.search(r"\b%s\b" % k, c):
            tr[k] += 1
print("\nTracer of the inner polar structure (Comm. column):", dict(tr))
n_star_ips = sum(1 for r in rows if re.search(r"\bs\b", r[12]))
n_gas_ips = sum(1 for r in rows if re.search(r"\b(HII|CO|HI)\b", r[12]))
print("  IPS traced by GAS (HII/CO/HI): %d ;  IPS traced by STARS: %d" % (n_gas_ips, n_star_ips))
n_both_pa = sum(1 for r in rows if r[6] and r[6] != "--" and r[9] and r[9] != "--")
print("  rows with BOTH host PA0 and polar-structure PA1 measured: %d" % n_both_pa)

write_manifest(out,
  source_url="https://arxiv.org/e-print/1204.4437",
  query=("GET https://arxiv.org/e-print/1204.4437 (Moiseev 2012, 'Inner Polar Rings and "
         "Disks: Observed Properties', Astrophysical Bulletin); verbatim transcription "
         "of Table 1 'List of galaxies with inner polar structures', which is SPLIT "
         "ACROSS TWO tabular environments (the second captioned '(continue)')"),
  columns=[{"name": c[0], "unit": c[1], "description": c[2]} for c in COLS],
  row_count=len(rows),
  source_file_within_archive="Moiseev_arxiv.tex",
  measurement_or_model=("MEASUREMENT. PA0/i0 describe the host stellar disc and PA1/i1 "
      "the inner polar structure; both come from observed velocity fields and/or "
      "photometry in the cited literature. Delta_i is the measured angle between the "
      "two planes (deprojected where i0 and i1 are both available; entries marked '*' "
      "are projected-geometry estimates only). NO dark-matter halo, NO Jeans/JAM "
      "model, NO dynamical mass enters this table."),
  note=("*** HIGH VALUE FOR THIS PROGRAMME. *** These are galaxies where an inner "
        "gas or stellar structure orbits in a plane strongly inclined (often ~90 deg) "
        "to the host stellar disc, i.e. two components with DIFFERENT angular-momentum "
        "directions in one baryonic system. Delta_i here is a DEPROJECTED angle between "
        "the two planes where both inclinations are known, which is stronger than the "
        "projected DeltaPA of the IFU misalignment catalogues. "
        "EXTRACTION GUARD: Table 1 is split across TWO tabular environments; parsing "
        "only the first would have silently returned a partial table. Both were parsed "
        "and the total asserted == 47, the paper's own stated count."),
  extra={"arxiv": "1204.4437", "paper": "Moiseev 2012, Astrophysical Bulletin",
         "expected_row_count": EXPECTED_N, "row_count_check": "PASS (47 == 47)",
         "n_tabular_blocks": len(blocks),
         "n_with_measured_Delta_i": len(have_di),
         "paper_stated_n_with_Delta_i": EXPECTED_DI,
         "delta_i_count_reconciliation": (
             "We recover 28 rows with a non-empty Delta_i cell; the paper states 27. "
             "The single extra row is NGC 4233, whose Delta_i cell in the LaTeX source "
             "reads '--    80' - a malformed cell that is simultaneously a dash "
             "(no value) and the number 80. Excluding NGC 4233 gives exactly 27, "
             "matching the paper. Treat NGC 4233's Delta_i as UNRELIABLE. All "
             "Delta_i-based counts below INCLUDE NGC 4233 at 80 deg; subtract 1 from "
             "the >=80 and within-10-deg counts to exclude it."),
         "ambiguous_row": "NGC 4233 (Delta_i cell = '--    80')",
         "n_Delta_i_within_10deg_of_90": n90_10,
         "n_Delta_i_within_15deg_of_90": n90_15,
         "n_Delta_i_within_20deg_of_90": n90_20,
         "n_Delta_i_ge_80": n_ge80,
         "n_IPS_traced_by_gas": n_gas_ips, "n_IPS_traced_by_stars": n_star_ips,
         "n_with_both_PA0_and_PA1": n_both_pa,
         "tracer_counts": dict(tr)})
