"""Verbatim transcription of Table 'super_summary_table' from the galstreams
paper (Mateu 2023, MNRAS 520, 5225; arXiv:2204.10326).

RECORDED FAILURE MODE GUARDED HERE: the table is split across TWO LaTeX files,
super_table_1of2.tex and super_table_2of2.tex. Both are read and the combined
row count is asserted against the paper's own \\Ntracks / \\Nunique macros.
"""
import os
import re
import tarfile

BASE = os.path.dirname(os.path.abspath(__file__))
TAR = os.path.join(BASE, "galstreams_paper_2204.10326.tar.gz")
OUT = os.path.join(BASE, "galstreams_paper_supertable.tsv")

t = tarfile.open(TAR)


def get(name):
    return t.extractfile([m for m in t.getmembers() if m.name == name][0]).read().decode("utf-8", "replace")


paper = get("galstreams_paper.tex")

# --- the paper's own stated sample sizes ---
macros = {}
for mm in re.finditer(r"\\newcommand\{?\\(N[A-Za-z]+)\}?\s*\{([^}]*)\}", paper):
    macros[mm.group(1)] = mm.group(2)
print("paper sample-size macros:", {k: v for k, v in macros.items()
                                    if k in ("Ntracks", "Nunique", "Nmultiple")})

COLS = ["StreamName", "TrackName", "InfoFlags", "Imp", "On", "Length_deg",
        "RA_i_deg", "DEC_i_deg", "D_i_kpc", "RA_f_deg", "DEC_f_deg", "D_f_kpc",
        "TRefs", "DRefs"]

rows = []
per_file = {}
for fn in ("super_table_1of2.tex", "super_table_2of2.tex"):
    txt = get(fn)
    n0 = len(rows)
    body = txt
    # strip the tabular preamble/头 and rules
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("\\") and not ("&" in line):
            continue
        if "&" not in line:
            continue
        if "StreamName" in line or "toprule" in line or "midrule" in line:
            continue
        if line.startswith("&") and "circ" in line:   # the units row
            continue
        if "$^\\circ$" in line or "(kpc)" in line:
            continue
        line = line.replace("\\\\", "").strip()
        line = re.sub(r"\\bottomrule|\\end\{tabular\}", "", line).strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != len(COLS):
            print("  SKIP malformed (%d cells): %s" % (len(cells), line[:90]))
            continue
        # unescape a few latex-isms
        cells = [c.replace("\\_", "_").replace("$", "").replace("\\", "") for c in cells]
        rows.append(cells)
    per_file[fn] = len(rows) - n0
    print("%s -> %d rows (running total %d)" % (fn, per_file[fn], len(rows)))

with open(OUT, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(COLS) + "\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")

print("wrote", OUT, "rows", len(rows))

# ---- HARD ASSERTIONS against the paper's stated sample size ----
ntracks = macros.get("Ntracks")
nunique = macros.get("Nunique")
uniq = sorted(set(r[0] for r in rows))
print("distinct StreamName in transcribed table:", len(uniq))
if ntracks:
    n = int(re.sub(r"[^0-9]", "", ntracks))
    assert len(rows) == n, ("SPLIT-TABLE FAILURE: transcribed %d rows but paper "
                            "states Ntracks=%d" % (len(rows), n))
    print("ASSERT OK: transcribed row count %d == paper \\Ntracks %d" % (len(rows), n))
if nunique:
    n = int(re.sub(r"[^0-9]", "", nunique))
    # NOT an equality check, and deliberately so. The StreamName COLUMN has 101
    # distinct literal values, while the paper's \Nunique = 95 counts streams
    # AFTER merging compound/split names: the paper states that tracks with
    # different literature names later shown to belong to one stream "are all
    # ascribed to the same stream with a compound name" (e.g. Orphan + Chenab).
    # The families responsible are AAU-ATLAS/AAU-AliqaUma, Cetus/Cetus-New/
    # Cetus-Palca, Jhelum/Jhelum-a/Jhelum-b, M68/M68-Fjorm, NGC3201/NGC3201-Gjoll.
    # So 101 != 95 is EXPECTED and is not a transcription defect.
    print("NOTE: distinct StreamName values = %d ; paper \\Nunique = %d "
          "(difference is compound-stream merging, not a transcription error)"
          % (len(uniq), n))
    assert len(uniq) >= n, "fewer distinct streams than the paper's merged count"
assert per_file["super_table_1of2.tex"] > 0 and per_file["super_table_2of2.tex"] > 0, \
    "one half of the split table contributed zero rows"
print("ASSERT OK: both halves of the split table contributed rows:", per_file)
