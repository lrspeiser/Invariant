"""Verbatim transcriptions of small published tables that are not in VizieR.

Every table asserts its row count against a number stated in the paper text.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.join(HERE, "eprints")


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean(c):
    c = c.strip()
    c = c.replace("~", " ").replace("\\,", " ")
    c = re.sub(r"\$\\pm\$|\\pm", "+/-", c)
    c = re.sub(r"\\degr|\\Msun|\\hline|\\\\", "", c)
    c = re.sub(r"[${}\\]", "", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c


def emit(name, header, rows, expected, meta):
    assert len(rows) == expected, "%s: %d rows, expected %d" % (name, len(rows), expected)
    for r in rows:
        assert len(r) == len(header), "%s: row %r has %d cells vs header %d" % (name, r[:2], len(r), len(header))
    p = os.path.join(HERE, name)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    blob = open(p, "rb").read()
    doc = {"file": name, "retrieved_utc": utc(),
           "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
           "row_count": len(rows), "column_count": len(header)}
    doc.update(meta)
    json.dump(doc, open(p + ".manifest.json", "w", encoding="utf-8"), indent=2)
    print("OK  %-52s rows=%-3d cols=%d" % (name, len(rows), len(header)))


# ---------------------------------------------------------------- Iodice 2003
tex = open(os.path.join(EP, "astro-ph_0211281_Iodice2003_PRG_TullyFisher", "iodicee.tex"),
           encoding="utf-8", errors="replace").read()
i = tex.index("\\startdata")
j = tex.index("\\enddata", i)
rows = []
for line in tex[i + 10:j].split("\\\\"):
    line = line.strip()
    if "&" not in line:
        continue
    rows.append([clean(c) for c in line.split("&")])
emit("iodice2003_table1_PRG_TF.tsv",
     ["Object", "PRC", "V_kms", "M_Kn", "M_B", "dV20_kms", "Ref"], rows, 16,
     {"source_url": "https://arxiv.org/e-print/astro-ph/0211281",
      "source_file_within_archive": "iodicee.tex",
      "paper": "Iodice E., Arnaboldi M., Bournaud F., Combes F., Sparke L.S., van Driel W., Capaccioli M. 2003, 'Polar Ring Galaxies and the Tully-Fisher relation: implications for the dark halo shape', ApJ 585, 730 (arXiv:astro-ph/0211281)",
      "columns": [
          {"name": "Object", "unit": "common name"},
          {"name": "PRC", "unit": "Whitmore+1990 Polar Ring Catalogue designation"},
          {"name": "V_kms", "unit": "km/s (heliocentric systemic velocity)"},
          {"name": "M_Kn", "unit": "mag (absolute Kn-band magnitude; blank = not measured)"},
          {"name": "M_B", "unit": "mag (absolute B-band magnitude)"},
          {"name": "dV20_kms", "unit": "km/s (HI linewidth at 20% of the peak line flux density)"},
          {"name": "Ref", "unit": "reference code: vD = van Driel et al. 2000/2002a/2002b and refs therein + Richter et al. 1994; vG = van Gorkom et al. 1987; A = Arnaboldi et al. 1993"},
      ],
      "query": "GET https://arxiv.org/e-print/astro-ph/0211281 ; transcribed from LaTeX deluxetable",
      "extraction": "Verbatim transcription of published Table 1. Row count asserted = 16. No unit conversion, no derivation.",
      "note": "IMPORTANT SCOPE LIMIT: dV20 is a GLOBAL, SPATIALLY UNRESOLVED single-dish HI linewidth. In classic PRGs the HI is associated with the POLAR structure, so dV20 traces the POLAR plane, NOT the host disk. This table therefore contains ONE plane only. The paper's Figure 2 shows, as an arrow per system, the offset in log(dV20) between the host-galaxy equatorial plane and the polar ring, but those host-plane velocities are NOT tabulated anywhere in the paper. M_B for NGC 4650A is from Gallagher et al. 2002; for NGC 2586 and A0136-0801 the B magnitudes relative to the host galaxy are from Reshetnikov et al. 1994. For NGC 660 the velocities are derived from Halpha data (van Driel et al. 1995)."})

# ------------------------------------------------------------- Khoperskov 2014
tex = open(os.path.join(EP, "1404.1247_Khoperskov2014_oblate", "shape_polar.tex"),
           encoding="utf-8", errors="replace").read()
i = tex.index("\\caption{Parameters of PRGs from the photometric data.}")
i = tex.index("\\hline", tex.index("\\begin{tabular}", i))
j = tex.index("\\end{tabular}", i)
rows = []
for line in tex[i:j].split("\\\\"):
    line = re.sub(r"^\s*(\\hline\s*)+", "", line).strip()
    if "&" not in line:
        continue
    cells = [clean(c) for c in line.split("&")]
    if not re.match(r"^(SPRC|NGC)\s*\d", cells[0]):
        continue
    rows.append(cells)
emit("khoperskov2014_table2_SPRC7_NGC4262_photometry.tsv",
     ["Name", "D_Mpc", "i_ring_deg", "i_CG_deg", "delta_deg",
      "col6_1e10Msun", "col7_kpc", "col8_1e10Msun", "col9_kpc", "col10_kpc",
      "col11_kpc", "col12_kpc"], rows, 2,
     {"source_url": "https://arxiv.org/e-print/1404.1247",
      "source_file_within_archive": "shape_polar.tex",
      "paper": "Khoperskov S.A., Moiseev A.V., Khoperskov A.V., Saburova A.S. 2014, 'To be or not to be oblate: the shape of the dark matter halo in polar ring galaxies', MNRAS 441, 2650 (arXiv:1404.1247)",
      "columns": [
          {"name": "Name", "unit": "galaxy"},
          {"name": "D_Mpc", "unit": "Mpc (adopted distance)"},
          {"name": "i_ring_deg", "unit": "deg -- SEE note: the published footnote defines i_ring as 'inclination angle of the central galaxy' and i_CG as 'inclination angle of the polar component', i.e. the footnote definitions are swapped relative to the subscripts"},
          {"name": "i_CG_deg", "unit": "deg -- same caveat"},
          {"name": "delta_deg", "unit": "deg (relative angle between the central galaxy and the polar component; TWO geometric solutions are given, separated by '/')"},
          {"name": "col6_1e10Msun", "unit": "10^10 Msun per the published units row"},
          {"name": "col7_kpc", "unit": "kpc per the published units row"},
          {"name": "col8_1e10Msun", "unit": "10^10 Msun per the published units row"},
          {"name": "col9_kpc", "unit": "kpc per the published units row"},
          {"name": "col10_kpc", "unit": "kpc per the published units row"},
          {"name": "col11_kpc", "unit": "kpc per the published units row"},
          {"name": "col12_kpc", "unit": "kpc per the published units row"},
      ],
      "query": "GET https://arxiv.org/e-print/1404.1247 ; transcribed from LaTeX",
      "extraction": "Verbatim transcription of published Table 2, 2 rows (SPRC-7 and NGC 4262). Values are copied unchanged.",
      "note": "PUBLISHED-TABLE DEFECT, RECORDED RATHER THAN SILENTLY REPAIRED. The header row of the published table carries 11 labels after 'Name' (D, i_ring, i_CG, delta, r_d, M_b, r_b, r_b^max, M_1, R_disc, R_PR) and the units row carries 11 units (Mpc, deg, deg, deg, 1e10 Msun, kpc, 1e10 Msun, kpc, kpc, kpc, kpc). These are mutually inconsistent: r_d would be a mass and M_b a length, and the resulting values are unphysical (bulge scale r_b = 3.4 > bulge size r_b^max = 0.8 for SPRC-7). The footnote list contains a COMMENTED-OUT entry '% M_d --- photometric mass of the central disc', which indicates a disc-mass column label was dropped from the header. The units row is self-consistent and the physically coherent reading is: col6 = disc mass (1e10 Msun), col7 = disc exponential scale length r_d (kpc), col8 = bulge mass M_b (1e10 Msun), col9 = bulge scale r_b (kpc), col10 = bulge size r_b^max (kpc), col11 = outer radius of the central galaxy R_disc (kpc), col12 = outer radius of the polar ring R_PR (kpc) -- under which the polar-ring mass M_1 is absent from the table. Columns are therefore emitted with NEUTRAL names carrying only the published units; do not attach the published labels without checking against the paper.",
      "context": "SPRC-7 and NGC 4262 are the two systems for which this paper derived a dark-halo axis ratio from rotation measured in BOTH planes. SPRC-7: ionized-gas velocity field of the polar disc from scanning Fabry-Perot (Brosch et al. 2010) plus a SCORPIO long-slit stellar rotation curve of the host at slit PA = 150 deg, asymmetric-drift corrected. NGC 4262 (= SPRC-33): WSRT HI of the polar ring (Oosterloo et al. 2010) plus SCORPIO-2 long-slit stellar kinematics of the host at slit PA = 0 and 160 deg. Both rotation curves appear as FIGURES only (paper's Fig. 3); no tabulated V(r)."})
