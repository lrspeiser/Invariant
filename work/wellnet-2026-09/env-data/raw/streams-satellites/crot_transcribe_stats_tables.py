"""Transcribe the misalignment STATISTICS tables (exact counts by angular range)
from Bryant+ 2019 (SAMI) and Ristea+ 2022 (SAMI) LaTeX sources.

These papers publish NO per-galaxy misalignment catalogue -- only aggregate
counts. That is stated plainly in the manifests and in the section report.
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest

D = os.path.dirname(os.path.abspath(__file__))


def clean(c):
    c = c.strip()
    c = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\substack\{([^}]*)\}", r"\1", c)
    c = c.replace("\\%", "%").replace("$", "").replace("\\pm", "+/-")
    c = c.replace("^{\\rm{o}}", "deg").replace("^{\\circ}", "deg")
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    c = c.replace("{", "").replace("}", "").replace("\\\\", "")
    return " ".join(c.split())


# ---------------------------------------------------- Bryant+ 2019 table 1 ---
src = os.path.join(D, "crot_bryant2019_sami_misalign_src", "MisalignmentSummaryTable")
txt = open(src, encoding="utf-8", errors="replace").read()
rows = []
for line in txt.splitlines():
    if "&" not in line or "multicolumn" in line or "cline" in line:
        continue
    if line.strip().startswith("\\hline") or "Number & Fraction" in line:
        continue
    cells = [clean(c) for c in line.split("&")]
    # layout: range, GAMA_All_N, frac, Field_N, frac, Groups_N, frac,
    #         <spacer>, Cluster_N, Cluster_frac  == 10 fields
    if len(cells) == 10 and cells[0]:
        rows.append(cells)
assert rows, "Bryant summary table parsed to 0 rows - check the & field count"
out_rows = [[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[8], r[9]] for r in rows]
assert len(out_rows) == 7, ("expected 7 PA-offset-range rows, got %d" % len(out_rows))
HDR = ["PA_offset_range", "GAMA_All_N", "GAMA_All_frac", "GAMA_Field_N",
       "GAMA_Field_frac", "GAMA_Groups_N", "GAMA_Groups_frac",
       "Cluster_N", "Cluster_frac"]
out = os.path.join(D, "crot_bryant2019_sami_misalignment_stats.tsv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(HDR) + "\n")
    for r in out_rows:
        fh.write("\t".join(r) + "\n")
print("BRYANT+2019 table:")
for r in [HDR] + out_rows:
    print("   " + " | ".join("%-16s" % c for c in r))

write_manifest(out,
    source_url="https://arxiv.org/e-print/1811.09298",
    query=("GET https://arxiv.org/e-print/1811.09298 (Bryant et al. 2019, MNRAS 483, 458, "
           "'The SAMI Galaxy Survey: stellar and gas misalignments and the origin of gas "
           "in nearby galaxies'); verbatim transcription of the file "
           "'MisalignmentSummaryTable' which the main .tex pulls in via \\input{} inside "
           "a table* environment"),
    columns=[{"name": h, "unit": ("deg" if h == "PA_offset_range" else "count/fraction")}
             for h in HDR],
    row_count=len(out_rows),
    source_file_within_archive="MisalignmentSummaryTable",
    measurement_or_model=("MEASUREMENT. Counts of galaxies binned by the measured "
        "|PA_stellar - PA_gas| offset, where both PAs are fitted to the observed SAMI "
        "H-alpha gas and stellar velocity fields. No DM halo, no Jeans/JAM model."),
    note=("AGGREGATE COUNTS ONLY. Bryant et al. 2019 publishes NO per-galaxy "
          "misalignment catalogue in its arXiv source and has no VizieR catalogue: "
          "the arXiv tarball contains exactly two tables, both statistical summaries. "
          "The individual galaxy IDs and their PA offsets are NOT recoverable from "
          "this source. Denominators: 486 GAMA field/group galaxies and 136 cluster "
          "galaxies have BOTH stellar and gas kinematic PAs successfully fitted "
          "(622 total). The 40-140 deg row is the closest published proxy for the "
          "POLAR population: 24 GAMA + 5 cluster = 29 galaxies."),
    extra={"bibcode": "2019MNRAS.483..458B", "arxiv": "1811.09298",
           "n_both_components_measured_GAMA": 486,
           "n_both_components_measured_clusters": 136,
           "n_both_components_measured_total": 622,
           "n_misaligned_gt30_total": 55 + 15,
           "n_misaligned_40_140_POLAR_BAND_total": 24 + 5,
           "n_counterrotating_gt150_total": 17 + 4,
           "extraction_guard": ("table pulled in via \\input{} from a separate archive "
                                "member - would have been silently missed by parsing "
                                "only the main .tex file")})

# ------------------------------------------------- Bryant+ 2019 morphology ---
src2 = os.path.join(D, "crot_bryant2019_sami_misalign_src", "MislignmentStatsByMophology")
txt2 = open(src2, encoding="utf-8", errors="replace").read()
rows2 = []
for line in txt2.splitlines():
    if "&" not in line:
        continue
    cells = [clean(c) for c in line.split("&")]
    if len(cells) == 10 and cells[0]:
        rows2.append(cells[:9] + [cells[9]])
    elif len(cells) == 9 and cells[0]:
        rows2.append(cells)
HDR2 = ["Description", "E", "E-S0", "S0", "S0-ESpirals", "ESpirals",
        "ESpirals-LSpirals", "LSpirals", "Unknown_or_NA"]
out2 = os.path.join(D, "crot_bryant2019_sami_misalignment_by_morphology.tsv")
with open(out2, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(HDR2) + "\n")
    for r in rows2:
        fh.write("\t".join(r) + "\n")
print("\nBRYANT+2019 by-morphology table: %d rows" % len(rows2))
write_manifest(out2,
    source_url="https://arxiv.org/e-print/1811.09298",
    query=("GET https://arxiv.org/e-print/1811.09298; verbatim transcription of the "
           "archive member 'MislignmentStatsByMophology' (\\input{} inside a table*)"),
    columns=[{"name": h, "unit": "count or fraction"} for h in HDR2],
    row_count=len(rows2),
    source_file_within_archive="MislignmentStatsByMophology",
    measurement_or_model=("MEASUREMENT. Counts binned by measured |PA_stellar-PA_gas| "
        "and by visual morphology. No DM model."),
    note=("NOTE: the header row of this file lists 9 morphology columns but the source "
          "file's first data line is itself the header; column 9 merges 'Unknown' and "
          "'NA'. Aggregate counts only - no per-galaxy IDs."),
    extra={"bibcode": "2019MNRAS.483..458B", "arxiv": "1811.09298"})

# ---------------------------------------------------- Ristea+ 2022 Table A1 ---
tex = open(os.path.join(D, "crot_ristea2022_sami_misalign_drivers_src",
                        "MAIN_paper_mnras_template.tex"),
           encoding="utf-8", errors="replace").read()
blk = tex.split(r"\label{tab:TableA1}", 1)[1].split(r"\end{tabular*}", 1)[0]
rows3 = []
for line in blk.splitlines():
    if "&" not in line or "multicolumn" in line or "multirow" in line:
        continue
    cells = [clean(c) for c in line.split("&")]
    if len(cells) == 6 and cells[0] and not cells[0].startswith("Whole"):
        rows3.append(cells)
HDR3 = ["Misalignment_range_deg", "Whole", "StarForming", "NonStarForming",
        "LateType", "EarlyType"]
out3 = os.path.join(D, "crot_ristea2022_sami_misalignment_fractions.tsv")
with open(out3, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(HDR3) + "\n")
    for r in rows3:
        fh.write("\t".join(r) + "\n")
print("\nRISTEA+2022 Table A1:")
for r in [HDR3] + rows3:
    print("   " + " | ".join("%-26s" % c for c in r))
write_manifest(out3,
    source_url="https://arxiv.org/e-print/2210.01147",
    query=("GET https://arxiv.org/e-print/2210.01147 (Ristea et al. 2022, MNRAS 517, "
           "2677, 'The SAMI Galaxy Survey: physical drivers of stellar-gas kinematic "
           "misalignments in the nearby Universe'); verbatim transcription of Table A1"),
    columns=[{"name": h, "unit": ("deg" if h.startswith("Misalign") else
                                  "percent (N/total)")} for h in HDR3],
    row_count=len(rows3),
    source_file_within_archive="MAIN_paper_mnras_template.tex",
    measurement_or_model=("MEASUREMENT. Fractions of SAMI DR3 galaxies binned by the "
        "measured |DeltaPA_stars-gas| from fits to the observed stellar and ionised-gas "
        "velocity fields. No DM halo, no Jeans/JAM model."),
    note=("AGGREGATE FRACTIONS ONLY - no per-galaxy IDs in the arXiv source. Parent "
          "sample = 1445 SAMI DR3 galaxies with BOTH stellar and ionised-gas kinematics "
          "measured. 169 misaligned (>30 deg); 95 in the 'unstable' 30-150 deg band "
          "(this band CONTAINS the ~90 deg polar systems but the paper does not "
          "separate them); 53 counter-rotating (150-180 deg); 1276 aligned (<30 deg)."),
    extra={"bibcode": "2022MNRAS.517.2677R", "arxiv": "2210.01147",
           "n_parent_both_components_measured": 1445,
           "n_all_misaligned_gt30": 169, "n_unstable_30_150": 95,
           "n_counterrotating_150_180": 53, "n_aligned_lt30": 1276})
print("\ndone")
