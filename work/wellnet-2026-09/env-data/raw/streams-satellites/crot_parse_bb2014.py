"""Transcribe Barrera-Ballesteros et al. (2014, A&A 568, A70) per-galaxy STELLAR
and IONISED-GAS kinematic position angles for non-interacting CALIFA galaxies,
and derive the gas-vs-stars kinematic misalignment.

*** SILENT-EXTRACTION TRAP AGAIN. *** Both tables live in SEPARATE archive
members (tabletex_Skin_prop.txt, tabletex_Gkin_prop.txt) pulled into the main
noInter_final.tex by \\input{}. Parsing only the main .tex returns ZERO tables.
"""
import sys, os, re, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest

D = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(D, "crot_bb2014_califa_kinalign_src")
EXPECTED_N = 80    # "80 non-interacting galaxies"


def clean(c):
    c = c.replace(r"\phantom{-}", "").replace(r"\phantom{0}", "")
    c = re.sub(r"\\phantom\{[^}]*\}", "", c)
    c = c.replace("$\\pm$", "|").replace("$", "")
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    c = c.replace("{", "").replace("}", "")
    return " ".join(c.split())


def parse(fn):
    txt = open(os.path.join(SRC, fn), encoding="utf-8", errors="replace").read()
    n_tab = len(re.findall(r"\\begin\{tabular\}", txt))
    # EACH FILE HOLDS TWO tabular environments (the table is continued onto a
    # second page). Taking only the first \midrule..\bottomrule block returns
    # 51 of 80 rows -- the exact silent-truncation failure this programme has
    # already been bitten by. Concatenate EVERY block.
    blocks = re.findall(r"\\midrule(.*?)\\bottomrule", txt, re.S)
    assert len(blocks) == n_tab, ("found %d midrule..bottomrule blocks but %d tabular "
                                  "environments in %s" % (len(blocks), n_tab, fn))
    body = "\n".join(blocks)
    out = {}
    for raw in body.split("\\\\"):
        if "&" not in raw:
            continue
        cells = [clean(c) for c in raw.split("&")]
        if len(cells) != 11:
            continue
        if not re.fullmatch(r"\d+", cells[0]):
            continue
        def val(x):
            p = x.split("|")[0].strip()
            try:
                return float(p)
            except ValueError:
                return None
        def err(x):
            p = x.split("|")
            if len(p) < 2:
                return None
            try:
                return float(p[1].strip())
            except ValueError:
                return None
        out[cells[0]] = {
            "id": cells[0], "Vsys": val(cells[3]), "rmax": val(cells[4]),
            "rmin": val(cells[5]), "PAmorph": val(cells[6]), "e_PAmorph": err(cells[6]),
            "PAkin_app": val(cells[7]), "e_PAkin_app": err(cells[7]),
            "dPAkin_app": val(cells[8]),
            "PAkin_rec": val(cells[9]), "e_PAkin_rec": err(cells[9]),
            "dPAkin_rec": val(cells[10]),
        }
    return n_tab, out


ns, S = parse("tabletex_Skin_prop.txt")
ng, G = parse("tabletex_Gkin_prop.txt")
print("GUARD: stellar table tabular envs=%d rows=%d ; gas table tabular envs=%d rows=%d"
      % (ns, len(S), ng, len(G)))
print("       (both tables are \\input{} from SEPARATE archive members - parsing only")
print("        noInter_final.tex would have found ZERO tabular environments)")
print("EXTRACTED stellar=%d gas=%d (paper states %d non-interacting galaxies)"
      % (len(S), len(G), EXPECTED_N))
assert len(S) == EXPECTED_N, "stellar row count %d != %d" % (len(S), EXPECTED_N)
assert len(G) == EXPECTED_N, "gas row count %d != %d" % (len(G), EXPECTED_N)
assert set(S) == set(G), "stellar and gas tables cover different galaxies"


def fold180(d):
    d = abs(d) % 360.0
    return 360.0 - d if d > 180.0 else d


def fold90(d):
    d = abs(d) % 180.0
    return 180.0 - d if d > 90.0 else d


COLS = [
 ("CALIFAid", "", "CALIFA catalogue ID"),
 ("PAkin_star_app", "deg", "STELLAR kinematic PA, approaching side"),
 ("e_PAkin_star_app", "deg", "Error on PAkin_star_app"),
 ("PAkin_star_rec", "deg", "STELLAR kinematic PA, receding side"),
 ("e_PAkin_star_rec", "deg", "Error on PAkin_star_rec"),
 ("PAkin_gas_app", "deg", "IONISED-GAS kinematic PA, approaching side"),
 ("e_PAkin_gas_app", "deg", "Error on PAkin_gas_app"),
 ("PAkin_gas_rec", "deg", "IONISED-GAS kinematic PA, receding side"),
 ("e_PAkin_gas_rec", "deg", "Error on PAkin_gas_rec"),
 ("PAmorph_star", "deg", "Morphological (photometric) PA at r_max, stellar table"),
 ("Vsys_star", "km/s", "Systemic velocity from the stellar velocity field"),
 ("Vsys_gas", "km/s", "Systemic velocity from the gas velocity field"),
 ("dPA_gas_star_rec", "deg", "DERIVED: |PAkin_gas_rec - PAkin_star_rec| folded to 0-180 "
                             "(gas-vs-stars kinematic misalignment, receding side)"),
 ("dPA_gas_star_rec_fold90", "deg", "DERIVED: same folded to 0-90 (undirected axis "
                                    "difference; 90 = polar, 0 = co- OR counter-aligned)"),
]

rows, dpa = [], []
for k in sorted(S, key=lambda x: int(x)):
    s, g = S[k], G[k]
    d = d90 = None
    if s["PAkin_rec"] is not None and g["PAkin_rec"] is not None:
        d = fold180(g["PAkin_rec"] - s["PAkin_rec"])
        d90 = fold90(g["PAkin_rec"] - s["PAkin_rec"])
        dpa.append(d)
    def f(x):
        return "" if x is None else ("%.1f" % x)
    rows.append([k, f(s["PAkin_app"]), f(s["e_PAkin_app"]), f(s["PAkin_rec"]),
                 f(s["e_PAkin_rec"]), f(g["PAkin_app"]), f(g["e_PAkin_app"]),
                 f(g["PAkin_rec"]), f(g["e_PAkin_rec"]), f(s["PAmorph"]),
                 f(s["Vsys"]), f(g["Vsys"]), f(d), f(d90)])

out = os.path.join(D, "crot_bb2014_califa_gas_vs_stellar_PAkin.tsv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(c[0] for c in COLS) + "\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")
print("WROTE %s (%d rows)" % (out, len(rows)))

print("\nGalaxies with BOTH stellar and gas kinematic PA (receding side): %d" % len(dpa))
print("  dPA(gas-stars) range %.1f .. %.1f  median %.1f"
      % (min(dpa), max(dpa), statistics.median(dpa)))
edges = list(range(0, 181, 15))
print("  histogram (deg):")
for a, b in zip(edges[:-1], edges[1:]):
    print("     %3d-%3d : %3d" % (a, b, sum(1 for v in dpa if a <= v < b)))
n_pol30 = sum(1 for v in dpa if abs(v - 90) <= 30)
n_pol20 = sum(1 for v in dpa if abs(v - 90) <= 20)
n_pol10 = sum(1 for v in dpa if abs(v - 90) <= 10)
n_cr = sum(1 for v in dpa if v >= 150)
n_al = sum(1 for v in dpa if v <= 30)
print("  ALIGNED   dPA <= 30           : %d" % n_al)
print("  POLAR     |dPA-90| <= 30      : %d" % n_pol30)
print("  POLAR     |dPA-90| <= 20      : %d" % n_pol20)
print("  POLAR     |dPA-90| <= 10      : %d" % n_pol10)
print("  COUNTER   dPA >= 150          : %d" % n_cr)

write_manifest(out,
  source_url="https://arxiv.org/e-print/1405.5222",
  query=("GET https://arxiv.org/e-print/1405.5222 (Barrera-Ballesteros et al. 2014, "
         "A&A 568, A70, 'Kinematic alignment of non-interacting CALIFA galaxies'); "
         "verbatim transcription and JOIN of the two archive members "
         "tabletex_Skin_prop.txt (stellar kinematics) and tabletex_Gkin_prop.txt "
         "(ionised-gas kinematics), matched on CALIFA id"),
  columns=[{"name": c[0], "unit": c[1], "description": c[2]} for c in COLS],
  row_count=len(rows),
  source_file_within_archive="tabletex_Skin_prop.txt + tabletex_Gkin_prop.txt",
  measurement_or_model=("MEASUREMENT. PAkin for stars and for ionised gas are each "
      "fitted directly to the observed CALIFA velocity fields (approaching and receding "
      "sides fitted separately, with errors). PAmorph is photometric. Vsys is measured. "
      "dPA_gas_star is derived arithmetically from two measurements. NO dark-matter "
      "halo, NO Jeans/JAM model, NO dynamical mass-to-light ratio."),
  note=("*** BOTH COMPONENTS KINEMATICALLY MEASURED in all %d galaxies, but ZERO of them "
        "are misaligned. *** Measured dPA(gas-stars) spans only 0.3 to 23.1 deg "
        "(median 4.7): ALL 80 are kinematically ALIGNED, and there are NO polar and NO "
        "counter-rotating systems in this table. That is the paper's own result, not a "
        "defect - the sample is NON-INTERACTING BY SELECTION and the paper is titled "
        "'Kinematic ALIGNMENT of non-interacting CALIFA galaxies'. USE THIS AS THE "
        "ALIGNED CONTROL POPULATION, not as a source of polar systems. Anyone looking "
        "here for CALIFA counter-rotators will find none. "
        "EXTRACTION GUARD: both tables are \\input{} from "
        "separate archive members; the main noInter_final.tex contains ZERO tabular "
        "environments, so a naive single-file parse returns nothing. Row counts asserted "
        "== 80 for both tables and the two id sets verified identical. "
        "CAUTION: dPA here is computed from the RECEDING-side PA of each component; the "
        "approaching-side columns are also provided and give an independent estimate - "
        "their difference is a useful systematic check."
        % len(rows)),
  extra={"arxiv": "1405.5222", "bibcode": "2014A&A...568A..70B",
         "expected_row_count": EXPECTED_N, "row_count_check": "PASS (80 == 80)",
         "n_both_components_measured": len(dpa),
         "n_aligned_le30": n_al, "n_polar_60_120": n_pol30,
         "n_polar_70_110": n_pol20, "n_polar_80_100": n_pol10,
         "n_counterrotating_ge150": n_cr,
         "projected_not_deprojected": True,
         "sample_selection": "non-interacting CALIFA galaxies"})
