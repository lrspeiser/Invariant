"""Transcribe Smirnova & Moiseev 2013 (Astrophys. Bull. 68, 371; arXiv:1311.4138)
Table 1 -- 'Parameters of galaxies' -- verbatim to TSV.

This table is SPLIT ACROSS TWO `table*` ENVIRONMENTS (the second captioned
'(continue)').  That is precisely the recorded silent-extraction failure mode,
so this script parses BOTH environments and asserts the combined row count
against the paper's stated sample size of 78 galaxies.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "eprints", "1311.4138_SmirnovaMoiseev2013_arepolar", "SmirnovaMoiseev.tex")
TEX = open(SRC, encoding="utf-8", errors="replace").read()

EXPECTED = 78  # "In total, the sample consists of 78 galaxies"; "a sample of 78 most reliable polar ring galaxy"


def clean(c):
    c = c.strip()
    c = c.replace("\\,", " ").replace("~", " ")
    c = re.sub(r"\\degr|\\hline|\$|\\", "", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c


rows = []
envs = 0
for m in re.finditer(r"\\begin\{table\*\}(.*?)\\end\{table\*\}", TEX, re.S):
    body = m.group(1)
    if "Parameters of galaxies" not in body and "(continue)" not in body:
        continue
    envs += 1
    tab = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}", body, re.S).group(1)
    for line in tab.split("\\\\"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        cells = [clean(c) for c in line.split("&")]
        if len(cells) != 11:
            continue
        if not re.match(r"^(SPRC|PRC)\s", cells[0]):
            continue
        rows.append(cells)

assert envs == 2, "expected the table to span exactly 2 table* environments, found %d" % envs
assert len(rows) == EXPECTED, "got %d rows, paper states a sample of %d galaxies" % (len(rows), EXPECTED)
names = [r[0] for r in rows]
assert len(set(names)) == len(names), "duplicate galaxy names -- the two halves overlap"

header = ["Name", "disk_a_arcsec", "disk_b_arcsec", "disk_PA_deg",
          "ring_a_arcsec", "ring_b_arcsec", "ring_PA_deg", "z",
          "delta1_deg", "delta2_deg", "Dring_over_Ddisk"]
path = os.path.join(HERE, "smirnova_moiseev2013_table1_PRG_geometry.tsv")
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

blob = open(path, "rb").read()
with open(path + ".manifest.json", "w", encoding="utf-8") as f:
    json.dump({
        "file": os.path.basename(path),
        "source_url": "https://arxiv.org/e-print/1311.4138",
        "source_file_within_archive": "SmirnovaMoiseev.tex",
        "paper": "Smirnova K.I. & Moiseev A.V. 2013, 'Are polar rings indeed polar?', Astrophysical Bulletin 68, 371 (arXiv:1311.4138v1)",
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "row_count": len(rows),
        "column_count": len(header),
        "columns": [
            {"name": "Name", "unit": "SPRC (Moiseev+2011) or PRC (Whitmore+1990) designation"},
            {"name": "disk_a_arcsec", "unit": "arcsec (inner/host disk semi-major axis)"},
            {"name": "disk_b_arcsec", "unit": "arcsec (inner/host disk semi-minor axis)"},
            {"name": "disk_PA_deg", "unit": "deg (position angle of the inner/host disk)"},
            {"name": "ring_a_arcsec", "unit": "arcsec (polar ring semi-major axis)"},
            {"name": "ring_b_arcsec", "unit": "arcsec (polar ring semi-minor axis)"},
            {"name": "ring_PA_deg", "unit": "deg (position angle of the polar ring)"},
            {"name": "z", "unit": "dimensionless redshift; '--' = unknown"},
            {"name": "delta1_deg", "unit": "deg (angle between the ring and disk planes, solution 1)"},
            {"name": "delta2_deg", "unit": "deg (angle between the ring and disk planes, solution 2)"},
            {"name": "Dring_over_Ddisk", "unit": "dimensionless (ring/disk diameter ratio)"},
        ],
        "query": "GET https://arxiv.org/e-print/1311.4138 ; transcribed from LaTeX",
        "extraction": "Verbatim transcription. The published table is SPLIT ACROSS TWO table* environments (the second captioned '(continue)'); both were parsed and the combined row count asserted equal to the paper's stated sample of 78 galaxies. No unit conversion, no derivation.",
        "note": "delta1/delta2 are the TWO geometric solutions for the angle between the polar-ring plane and the host-disk plane, obtained by deprojecting the observed axis ratios and position angles of the two photometric components (Smirnova & Moiseev 2013 Sect. 3). The degeneracy is not resolved by imaging alone. These are PHOTOMETRIC/GEOMETRIC angles, NOT kinematic measurements.",
    }, f, indent=2)
print("OK  %s  rows=%d cols=%d  (2 table* environments merged)" % (os.path.basename(path), len(rows), len(header)))
