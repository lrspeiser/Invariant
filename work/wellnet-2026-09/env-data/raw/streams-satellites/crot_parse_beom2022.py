"""Transcribe Beom et al. (2022) Table 1: the list of MaNGA EDGE-ON galaxies with
a counter-rotating GASEOUS disc. Small but fully identified (MaNGA-ID + plate-IFU),
so each object can be looked up directly in the MaNGA DAP."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest

D = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(D, "crot_beom2022_manga_counterrot_edgeon_src", "Beom_2022_arXiv.tex")
EXPECTED_N = 10   # "10 edge-on galaxies" with a counter-rotating gaseous disc

txt = open(TEX, encoding="utf-8", errors="replace").read()
blk = txt.split(r"\label{tab:list}", 1)[1].split(r"\end{tabular}", 1)[0]
print("GUARD: 1 tabular block for tab:list; searching for stray continuation blocks...")
print("       total 'The list of the gaseous counter-rotating' captions in source: %d"
      % len(re.findall(r"The list of the gaseous counter-rotating", txt)))

rows = []
for raw in blk.split("\\\\"):
    if "&" not in raw or "multicolumn" in raw:
        continue
    cells = [c.strip() for c in raw.split("&")]
    if len(cells) != 9:
        continue
    c0 = cells[0].replace("\\hline", "").strip()
    if not re.match(r"^\d+-\d+$", c0):     # MaNGA-ID looks like '1-225'
        continue
    cells[0] = c0
    rows.append([re.sub(r"\$[^$]*\$", "", c).replace("\\hline", "").strip() for c in cells])

print("EXTRACTED %d rows (paper: %d edge-on counter-rotating galaxies)" % (len(rows), EXPECTED_N))
assert len(rows) == EXPECTED_N, "ROW COUNT MISMATCH %d != %d" % (len(rows), EXPECTED_N)
assert len(set(r[0] for r in rows)) == len(rows), "duplicate MaNGA IDs"

COLS = [
 ("MaNGAID",   "",              "MaNGA-ID (unique MaNGA galaxy identifier)"),
 ("PlateIFU",  "",              "Plate-IFU designation (look-up key for the MaNGA DAP)"),
 ("SDSSID",    "",              "SDSS object designation"),
 ("Label",     "",              "Letter A-J used in the paper, ordered by stellar mass"),
 ("RAdeg",     "deg",           "Right ascension (J2000)"),
 ("DEdeg",     "deg",           "Declination (J2000)"),
 ("z",         "",              "Redshift"),
 ("Distance",  "Mpc",           "Distance"),
 ("logMstar",  "[Msun]",        "Log stellar mass"),
]
out = os.path.join(D, "crot_beom2022_manga_edgeon_counterrot_gas.tsv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(c[0] for c in COLS) + "\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")
for r in rows:
    print("   %-10s %-12s %-3s z=%-8s D=%-5s logM*=%s" % (r[0], r[1], r[3], r[6], r[7], r[8]))
print("WROTE %s" % out)

write_manifest(out,
  source_url="https://arxiv.org/e-print/2206.00682",
  query=("GET https://arxiv.org/e-print/2206.00682 (Beom et al. 2022, 'SDSS-IV MaNGA: "
         "Characteristics of Edge-on Galaxies with a Counter-rotating Gaseous Disk'); "
         "verbatim transcription of Table 1 'The list of the gaseous counter-rotating "
         "galaxies' (label tab:list)"),
  columns=[{"name": c[0], "unit": c[1], "description": c[2]} for c in COLS],
  row_count=len(rows),
  source_file_within_archive="Beom_2022_arXiv.tex",
  measurement_or_model=("MEASUREMENT for the identification and the astrometry/redshift. "
      "The counter-rotation is established from the observed MaNGA stellar and ionised-"
      "gas velocity fields and position-velocity diagrams. Distance follows from z and a "
      "Hubble constant; logMstar is a stellar-population MODEL quantity. NO dark-matter "
      "halo, NO Jeans/JAM model."),
  note=("EDGE-ON MaNGA galaxies (i ~ 90 deg) whose IONISED GAS counter-rotates with "
        "respect to the stellar disc. Geometry is ANTI-PARALLEL (~180 deg), NOT polar "
        "(~90 deg) - the gas orbits in the SAME plane as the stars but in the opposite "
        "sense. Value for this programme: because the hosts are edge-on, the projection "
        "factor is near unity and both rotation curves are measured with minimal "
        "inclination correction. Every object carries a plate-IFU, so the full velocity "
        "fields can be pulled from the MaNGA DAP."),
  extra={"arxiv": "2206.00682", "paper": "Beom et al. 2022",
         "expected_row_count": EXPECTED_N, "row_count_check": "PASS (10 == 10)",
         "geometry": "anti-parallel (~180 deg) counter-rotation, not polar",
         "both_components_kinematically_measured": True,
         "host_inclination": "edge-on by selection"})
