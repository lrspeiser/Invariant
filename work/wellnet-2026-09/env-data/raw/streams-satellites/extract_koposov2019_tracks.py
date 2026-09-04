"""Verbatim transcription of the Orphan-Chenab MEASUREMENT tables from the
LaTeX source of Koposov et al. 2019, MNRAS 485, 4726 (arXiv:1812.08172).

VizieR holds ONLY the RR Lyrae subset (J/MNRAS/485/4726/table15, 109 rows).
The stream's on-sky track and, crucially, its RADIAL VELOCITY TRACK are NOT in
VizieR; they exist only as tables in the paper. They are transcribed here.

Coordinates phi1/phi2 are in the Orphan stream frame defined in the paper.
"""
import os
import re
import tarfile

BASE = os.path.dirname(os.path.abspath(__file__))
TAR = os.path.join(BASE, "koposov2019_paper_1812.08172.tar.gz")

t = tarfile.open(TAR)
txt = t.extractfile([m for m in t.getmembers() if m.name == "main.tex"][0]).read().decode("utf-8", "replace")

# Split on table environments and keep caption + tabular body together
blocks = []
for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", txt, re.S):
    blocks.append(m.group(1))
print("table environments found:", len(blocks))

SPECS = [
    ("Stream track measurements from \\gaia RGB stars",
     "koposov2019_orphan_track_gaia_rgb",
     ["phi1_deg", "phi2_deg", "sigma_phi2_deg"],
     ["deg", "deg", "deg"],
     "MEASUREMENT. On-sky track of the Orphan stream measured from Gaia RGB star "
     "counts: stream centroid phi2 and its intrinsic width sigma_phi2 in bins of "
     "phi1. No potential assumed."),
    ("Stream track measurements from DECaLS matched filtered stars",
     "koposov2019_orphan_track_decals",
     ["phi1_deg", "phi2_deg", "sigma_phi2_deg"],
     ["deg", "deg", "deg"],
     "MEASUREMENT. On-sky track of the Orphan stream measured from DECaLS "
     "matched-filter star counts. No potential assumed."),
    ("Measurement of Orphan stream velocity track from  SDSS spectroscopic observations",
     "koposov2019_orphan_velocity_track",
     ["phi1_deg", "Vrad_kms", "e_Vrad_kms"],
     ["deg", "km/s", "km/s"],
     "MEASUREMENT. Radial-velocity track of the Orphan stream from SDSS "
     "spectroscopy. The paper states the third column is the velocity "
     "MEASUREMENT UNCERTAINTY, not a velocity dispersion. No potential assumed."),
    ("The location of points defining the natural spline for the   stream track shown on Fig",
     "koposov2019_orphan_selection_spline_track",
     ["phi1_deg", "phi2_deg"], ["deg", "deg"],
     "SELECTION FUNCTION, NOT A MEASUREMENT. These are the knots of a natural "
     "spline used only to SELECT candidate RR Lyrae near the stream. They "
     "describe the authors' selection window, not a measured track."),
    ("The location of points defining the natural spline for the   heliocentric distance track",
     "koposov2019_orphan_selection_spline_distance",
     ["phi1_deg", "dist_kpc"], ["deg", "kpc"],
     "SELECTION FUNCTION, NOT A MEASUREMENT. Spline knots used to select RR Lyrae "
     "by heliocentric distance."),
    ("The location of points defining the spline for the   track of proper motion along the stream",
     "koposov2019_orphan_selection_spline_pm",
     ["phi1_deg", "pm_phi1_masyr"], ["deg", "mas/yr"],
     "SELECTION FUNCTION, NOT A MEASUREMENT. Spline knots used to select RR Lyrae "
     "by proper motion along the stream."),
]

written = []
for cap_key, stem, cols, units, mm in SPECS:
    # normalise whitespace on BOTH sides: the LaTeX source wraps captions across
    # lines with irregular indentation, so a literal substring test misses them.
    def norm(s):
        return re.sub(r"\s+", " ", s)

    blk = None
    key = norm(cap_key)
    for b in blocks:
        if key in norm(b):
            blk = b
            break
    if blk is None:
        print("!! caption not found:", cap_key[:60])
        continue
    tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", blk, re.S)
    body = tab.group(1)
    rows = []
    for line in body.split("\\\\"):
        line = line.strip()
        line = re.sub(r"\\hline|\\\\", "", line).strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != len(cols):
            continue
        # keep only numeric data rows (drops the header and unit rows)
        try:
            [float(c.replace("$", "").replace("\\", "")) for c in cells]
        except ValueError:
            continue
        rows.append(cells)
    out = os.path.join(BASE, "stream_" + stem + ".tsv")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print("%-46s rows=%3d -> %s" % (stem, len(rows), os.path.basename(out)))
    written.append((out, cols, units, len(rows), mm, cap_key))

# manifests
import sys
sys.path.insert(0, BASE)
from _manifest import write_manifest  # noqa: E402

for out, cols, units, n, mm, cap in written:
    assert n > 0, "empty table: " + out
    write_manifest(
        out,
        source_url="https://arxiv.org/e-print/1812.08172",
        source_file_within_archive="main.tex",
        query="extract_koposov2019_tracks.py -- regex transcription of the LaTeX tabular "
              "whose caption contains: " + cap[:70],
        columns=[{"name": c, "unit": u} for c, u in zip(cols, units)],
        row_count=n,
        extraction="Verbatim transcription of the published LaTeX table. No unit conversion, "
                   "no derivation, no cross-table join. Row count asserted non-zero and "
                   "reported; each table lives in a SINGLE table environment in main.tex "
                   "(checked: 8 table environments total, none split).",
        measurement_or_model=mm,
        note="Koposov et al. 2019, MNRAS 485, 4726 (arXiv:1812.08172), Orphan-Chenab stream. "
             "These tables are NOT in the VizieR record J/MNRAS/485/4726, which holds only "
             "the RR Lyrae subset (table15, 109 rows). phi1/phi2 are in the Orphan stream "
             "frame defined in that paper.",
    )
print("\nwrote", len(written), "tables")
