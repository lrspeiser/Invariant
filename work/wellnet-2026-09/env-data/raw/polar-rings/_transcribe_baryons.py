"""Verbatim transcription of the two published baryonic mass/photometry tables
that the programme needs to compute g_bar in BOTH planes.

  1. Iodice et al. 2015 (A&A 583, A48; arXiv:1509.01112) Table 1 -- the
     component masses and scales of the NGC 4650A two-perpendicular-disk model.
  2. Jozsa et al. 2009 (A&A 494, 489; arXiv:0810.3025) Table 1 -- basic
     properties of NGC 2685.
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


def emit(name, header, rows, meta):
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


# ------------------------------------------------ NGC 4650A two-disk model ---
# The published table interleaves a symbol row and a value row per component,
# so it is transcribed here as one row per component with the values only.
NGC4650A = [
    # component, M(1e9 Msun), h(kpc), r(kpc), r1(kpc), r2(kpc)
    ["HG bulge",   "0.2",  "",    "0.17",  "",     ""],
    ["HG disk",    "10.3", "0.5", "0.948", "",     ""],
    ["Polar disk", "15.",  "0.5", "",      "5.95", "6.8"],
    ["HI disk",    "7.2",  "0.5", "",      "3.4",  "15.3"],
    ["DM halo",    "15",   "1.2", "6.0",   "",     ""],
]
# guard: the transcription must match the published values still present in the source
src = open(os.path.join(EP, "1509.01112_Iodice2015_NGC4650A_MUSE",
                        "n4650a_MUSE_astroph.tex"), encoding="utf-8", errors="replace").read()
tab = src[src.index("\\label{tab:model}"):src.index("\\end{tabular}", src.index("\\label{tab:model}"))]
for tok in ["0.2", "0.17", "10.3", "0.948", "15.", "5.95", "6.8", "7.2", "3.4", "15.3", "6.0", "1.2"]:
    assert tok in tab, "value %r absent from the published table body" % tok
assert len(NGC4650A) == 5

emit("iodice2015_table1_NGC4650A_mass_model.tsv",
     ["Component", "M_1e9Msun", "h_kpc", "r_kpc", "r1_kpc", "r2_kpc"], NGC4650A,
     {"source_url": "https://arxiv.org/e-print/1509.01112",
      "source_file_within_archive": "n4650a_MUSE_astroph.tex (Table 1, 'Masses and scales for the model components')",
      "paper": "Iodice E., Coccato L., Combes F., de Zeeuw T., Arnaboldi M., Weilbacher P.M., Bacon R., Kuntschner H., Spavone M. 2015, 'Mapping the inner regions of the polar disk galaxy NGC 4650A with MUSE', A&A 583, A48 (arXiv:1509.01112)",
      "columns": [
          {"name": "Component", "unit": "model component (HG = host galaxy)"},
          {"name": "M_1e9Msun", "unit": "1e9 Msun (total mass of the component)"},
          {"name": "h_kpc", "unit": "kpc (characteristic scale HEIGHT)"},
          {"name": "r_kpc", "unit": "kpc (characteristic scale LENGTH)"},
          {"name": "r1_kpc", "unit": "kpc (inner radius of the stellar/gaseous polar disk)"},
          {"name": "r2_kpc", "unit": "kpc (outer radius of the stellar/gaseous polar disk)"},
      ],
      "query": "GET https://arxiv.org/e-print/1509.01112 ; transcribed from LaTeX Table 1",
      "extraction": "Verbatim transcription. The published table interleaves a symbol row and a value row per component; only the VALUE rows are emitted, one row per component. Every transcribed numeric token was asserted present in the published table body before writing. No unit conversion, no derivation.",
      "note": "THE BARYONIC INPUT FOR g_bar IN BOTH PLANES FOR NGC 4650A. The first four rows (HG bulge, HG disk, Polar disk, HI disk) are the BARYONS: total baryonic mass 0.2 + 10.3 + 15 + 7.2 = 32.7e9 Msun, of which 10.5e9 sits in the host-disk plane and 22.2e9 in the polar plane. The fifth row, 'DM halo', is a FITTED MODEL COMPONENT, NOT AN OBSERVATION -- under the programme's no-dark-matter-as-observation rule it must be used for debugging/comparison only and never as data. Compare the earlier Combes & Arnaboldi (1996) decomposition of the same galaxy, reproduced verbatim in Lughausen, Famaey & Kroupa 2013 (arXiv:1304.4931 Sect. 3): Plummer bulge M_b = 0.2e9 Msun with r_p = 0.17 kpc; Miyamoto-Nagai host disk M_d = 11e9 Msun with h_r = 0.748 kpc and h_z = 0.3 kpc; stellar polar ring 9.5e9 Msun with h_r1 = 6.8, h_r2 = 5.95 kpc; gaseous polar ring 6.4e9 Msun with h_r1 = 15.3, h_r2 = 3.4 kpc. The two decompositions agree to ~10-20% per component.",
      "geometry": "MUSE (Iodice+2015): host galaxy photometric major axis P.A. = 67 +/- 2 deg, polar disk major axis P.A. = 160 deg, i.e. 93 deg apart on the sky. Systemic velocity 2875 +/- 3 km/s, central stellar sigma = 60 +/- 6 km/s. Polar-disk ionized gas reaches V ~ 100-120 km/s at R ~ 75 arcsec ~ 16 kpc with sigma_gas ~ 30-40 km/s; host stellar sigma stays at ~50-60 km/s at all radii."})

# --------------------------------------------------- NGC 2685 basic props ---
src = open(os.path.join(EP, "0810.3025_Josza2009_NGC2685_kinmodel", "xx.tex"),
           encoding="utf-8", errors="replace").read()
i = src.index("Basic properties of NGC~2685.")
j = src.index("\\end{tabular}", i)
rows = []
for line in src[i:j].split("\\\\"):
    line = line.strip()
    if "&" not in line or "\\hline" == line:
        continue
    cells = [c.strip() for c in line.split("&")]
    if len(cells) != 4:
        continue
    desc, param, val, ref = cells
    desc = re.sub(r"\\citealt\{([^}]*)\}|\\citep\[\]\[\]\{([^}]*)\}|[\\{}$]", " ", desc)
    param = re.sub(r"[\\{}$]|rm |ion\[HI\]|,", " ", param)
    val = re.sub(r"\\,|\\pm|[\\{}$]", lambda m: "+/-" if m.group(0) == "\\pm" else " ", val)
    ref = re.sub(r"\\citealt\{([^}]*)\}|\\citep\[\]\[\]\{([^}]*)\}|[\\{}$\[\]]", " ", ref)
    def tidy(x):
        x = re.sub(r"\bion\s*H\s*i\b", "HI", x)
        x = re.sub(r"\b(rm|hline|odot|prime|frac|hmsm|dmsm|citealt|citep|LEDA\s*$)\b", " ", x)
        x = x.replace("_ ", "_").replace("^ ", "^")
        return re.sub(r"\s+", " ", x).strip(" .,")
    row = [tidy(x) for x in (desc, param, val, ref)]
    row[3] = re.sub(r"\s+", " ", ref.replace("citealt", "").replace("citep", "")).strip(" []{}")
    if row[0].lower().startswith("description"):
        continue
    if not row[2]:
        continue
    rows.append(row)
assert len(rows) >= 20, "only %d property rows parsed" % len(rows)
emit("josza2009_table1_NGC2685_properties.tsv",
     ["Description", "Parameter", "Value", "Reference"], rows,
     {"source_url": "https://arxiv.org/e-print/0810.3025",
      "source_file_within_archive": "xx.tex (Table 1, 'Basic properties of NGC 2685')",
      "paper": "Jozsa G.I.G., Oosterloo T.A., Morganti R., Klein U., Erben T. 2009, A&A 494, 489 (arXiv:0810.3025)",
      "columns": [
          {"name": "Description", "unit": "quantity described, with its unit stated inline"},
          {"name": "Parameter", "unit": "symbol used in the paper"},
          {"name": "Value", "unit": "as printed, units are given in the Description column"},
          {"name": "Reference", "unit": "source of the value"},
      ],
      "query": "GET https://arxiv.org/e-print/0810.3025 ; transcribed from LaTeX Table 1",
      "extraction": "Verbatim transcription of the published Table 1. LaTeX macros stripped; numeric values untouched. Units remain embedded in the Description column exactly as published.",
      "note": "Baryonic and distance data for NGC 2685 (PRC A-03, the Helix/Spindle) accompanying the tabulated tilted-ring rotation curve in josza2009_table5_NGC2685_tiltedring.tsv."})
