import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from _manifest import write_manifest  # noqa: E402

write_manifest(
    os.path.join(BASE, "ibata2021_paper_2012.05245.tar.gz"),
    source_url="https://arxiv.org/e-print/2012.05245",
    query="HTTP GET https://arxiv.org/e-print/2012.05245",
    columns=[], row_count=None,
    extraction="Unmodified arXiv LaTeX source tarball for Ibata et al. 2021, ApJ 914, 123, "
               "'Charting the Galactic Acceleration Field I'. Read ONLY to establish the "
               "provenance of the VizieR table's Stream label column.",
    measurement_or_model="PAPER SOURCE (documentation).",
    note="Establishes that the per-star 'Stream' label 1-32 in J/ApJ/914/123 has NO published "
         "name mapping: the paper says the label is used to colour the streams in a figure. "
         "Also confirms the paper's own per-stream counts (276 stars for the NGC 6397 stream, "
         "388 for NGC 3201) used here to validate the derived label-to-name mapping. "
         "The orbits drawn in that paper's figures were integrated in the Dehnen & Binney "
         "(1998) potential model #1 -- those orbit curves are MODEL, not measurement.",
)

write_manifest(
    os.path.join(BASE, "koposov2019_paper_1812.08172.tar.gz"),
    source_url="https://arxiv.org/e-print/1812.08172",
    query="HTTP GET https://arxiv.org/e-print/1812.08172",
    columns=[], row_count=None,
    extraction="Unmodified arXiv LaTeX source tarball for Koposov et al. 2019, MNRAS 485, 4726, "
               "'Piercing the Milky Way: an all-sky view of the Orphan stream'. Source of the "
               "six transcribed stream_koposov2019_* tables.",
    measurement_or_model="PAPER SOURCE (documentation).",
    note="main.tex holds 8 table environments; each target table is contained in a SINGLE "
         "environment (verified by verify_koposov_rows.py), so the split-table failure mode "
         "does not apply here. Numeric row counts in the source are 8 / 10 / 8 / 0 / 14 / 12 / "
         "12 / 10 and the transcriptions reproduce them exactly. Table 4 (RR Lyrae) is printed "
         "in truncated form -- its full 109 rows are the VizieR record J/MNRAS/485/4726/table15, "
         "acquired separately.",
)

p = os.path.join(BASE, "stream_ibata2021_streamfinder_members.ReadMe.txt")
write_manifest(
    p,
    source_url="https://cdsarc.cds.unistra.fr/ftp/J/ApJ/914/123/ReadMe",
    query="HTTP GET https://cdsarc.cds.unistra.fr/ftp/J/ApJ/914/123/ReadMe",
    columns=[], row_count=None,
    extraction="Unmodified CDS ReadMe for catalogue J/ApJ/914/123.",
    measurement_or_model="DOCUMENTATION.",
    note="Authoritative column definitions. Key facts recorded from it: HRV uses 1000 km/s as a "
         "NULL SENTINEL and e_HRV uses 10000; dSF is 'Distance to the star estimated by "
         "STREAMFINDER' and is therefore MODEL-DEPENDENT (STREAMFINDER searches for stars "
         "consistent with a common orbit in an ASSUMED Galactic potential); r_HRV=0 means no "
         "velocity measurement and occurs 5275 times out of 5960, leaving 685 stars with a "
         "velocity. The per-reference counts in the ReadMe (15/1/6/44/135/86/155/5/7/6/44/181) "
         "sum to 685 and were verified against the downloaded table exactly.",
)
print("DONE")
