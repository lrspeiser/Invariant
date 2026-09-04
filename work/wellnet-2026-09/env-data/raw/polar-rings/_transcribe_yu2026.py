"""Transcribe Yu et al. 2026 (arXiv:2601.22222) Tables 1 and 2 verbatim to TSV.

Guard against the recorded failure mode (a LaTeX table split across two
environments silently returning a subset): the paper states the sample is
"40 kinematically confirmed PRGs", so Table 1 MUST yield exactly 40 rows or
this script raises.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "eprints", "2601.22222_2026_PRG_HI_observations", "PRG_TFR_FAST.tex")
TEX = open(SRC, encoding="utf-8", errors="replace").read()


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean(cell):
    c = cell.strip()
    c = c.replace("~", " ")
    c = re.sub(r"\$\\pm\$", "+/-", c)
    c = re.sub(r"\\pm", "+/-", c)
    c = c.replace("$\\leq$", "<=").replace("\\leq", "<=")
    c = c.replace("$", "").replace("\\", "")
    c = re.sub(r"\s+", " ", c).strip()
    return c


def grab(startpat, endpat):
    i = TEX.index(startpat)
    j = TEX.index(endpat, i)
    body = TEX[i + len(startpat):j]
    rows = []
    for line in body.split("\\\\"):
        line = line.strip()
        if not line:
            continue
        # drop comment-only lines and pure-comment prefixes
        line = "\n".join(l for l in line.split("\n") if not l.strip().startswith("%"))
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if "&" not in line:
            continue
        rows.append([clean(c) for c in line.split("&")])
    return rows


def write(name, header, rows, note, columns, expected):
    assert len(rows) == expected, "%s: got %d rows, expected %d" % (name, len(rows), expected)
    ncol = len(header)
    for r in rows:
        assert len(r) == ncol, "%s: row %r has %d cells, header has %d" % (name, r[:3], len(r), ncol)
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    blob = open(path, "rb").read()
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "file": name,
            "source_url": "https://arxiv.org/e-print/2601.22222",
            "source_file_within_archive": "PRG_TFR_FAST.tex",
            "paper": "Yu, Zheng et al. 2026, 'Insights into the Physical Nature of Polar Ring Galaxies from HI Observations', ApJ 999, 199 (arXiv:2601.22222v1)",
            "retrieved_utc": utcnow(),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "row_count": len(rows),
            "column_count": ncol,
            "columns": columns,
            "query": "GET https://arxiv.org/e-print/2601.22222 ; transcribed from LaTeX deluxetable",
            "extraction": "Verbatim transcription of the published LaTeX table. No unit conversion, no derivation, no cross-table join. LaTeX markup stripped ($, \\pm -> +/-, \\leq -> <=). Row count asserted against the paper's stated sample size.",
            "note": note,
        }, f, indent=2)
    print("OK  %-44s rows=%-4d cols=%d" % (name, len(rows), ncol))


# ---- Table 1: 40 kinematically confirmed PRGs -------------------------------
t1 = grab("\\label{tab:basic}", "\\enddata")
# the label appears AFTER the table; grab by \startdata instead
i = TEX.index("\\tablecaption{A comprehensive database of 40 kinematically confirmed PRGs.}")
i = TEX.index("\\startdata", i)
j = TEX.index("\\enddata", i)
body = TEX[i + len("\\startdata"):j]
rows1 = []
for line in body.split("\\\\"):
    line = "\n".join(l for l in line.split("\n") if not l.strip().startswith("%")).strip()
    if "&" not in line:
        continue
    rows1.append([clean(c) for c in line.split("&")])

write("yu2026_table1_confirmed_PRGs.tsv",
      ["PRG", "Name", "RAdeg", "DEdeg", "z", "KinConfRefs", "FAST_Mode", "FAST_Time_s"],
      rows1,
      "Yu et al. 2026 Table 1. THE definitive census of kinematically confirmed polar-ring "
      "galaxies: 40 systems. Column KinConfRefs are the paper's numeric reference codes for the "
      "KINEMATIC confirmation: 1=Whitmore+1990 AJ 100,1489; 2=Whitmore+1987 ApJ 314,439; "
      "3=Schiminovich+2013 AJ 145,34; 4=Hagen-Thorn+1997 A&A 319,430; 5=Merkulova+2008 AstL 34,542; "
      "6=Cox+1995 NYASA 751,27; 7=Reshetnikov+2001 MNRAS 322,689; 8=Reshetnikov+1994 A&A 291,57; "
      "9=Reshetnikov+2006 A&A 446,447; 10=Arnaboldi+1993 A&A 267,21; 11=Arnaboldi+1994 ASPC 54,437; "
      "12=Reshetnikov+2002 A&A 383,390; 13=Reshetnikov+2005 A&A 431,503; 14=van Driel+1995 AJ 109,942; "
      "15=Merkulova+2009 AstL 35,587; 16=Schiminovich+1994 ApJ 423,L101; 17=Egorov & Moiseev 2019 "
      "MNRAS 486,4186; 18=Moiseev+2014 ASPC 486,71; 19=Brosch+2010 MNRAS 401,2067; "
      "20=Moiseev+2011 MNRAS 418,244; 21=Bettoni+2010 A&A 519,A72; 22=Merkulova+2013 arXiv:1302.1339; "
      "23=Moiseev 2008 AstBu 63,201. FAST_Mode/FAST_Time_s blank = no new FAST observation.",
      [{"name": "PRG", "unit": "catalogue designation (PRC = Whitmore+1990, SPRC = Moiseev+2011)"},
       {"name": "Name", "unit": "common name"},
       {"name": "RAdeg", "unit": "deg (J2000, from NED)"},
       {"name": "DEdeg", "unit": "deg (J2000, from NED)"},
       {"name": "z", "unit": "dimensionless (optical redshift from NED)"},
       {"name": "KinConfRefs", "unit": "reference codes for the kinematic confirmation"},
       {"name": "FAST_Mode", "unit": "FAST observing mode"},
       {"name": "FAST_Time_s", "unit": "s"}],
      expected=40)

# ---- Table 2: HI + derived properties --------------------------------------
i = TEX.index("\\tablecaption{Properties of Polar Ring Galaxies}")
i = TEX.index("\\startdata", i)
j = TEX.index("\\enddata", i)
body = TEX[i + len("\\startdata"):j]
rows2 = []
for line in body.split("\\\\"):
    line = "\n".join(l for l in line.split("\n") if not l.strip().startswith("%")).strip()
    if "&" not in line:
        continue
    rows2.append([clean(c) for c in line.split("&")])

write("yu2026_table2_HI_properties.tsv",
      ["Galaxy", "D_L_Mpc", "V_c", "F_HI", "V85", "A_F", "K", "SNR", "sigma_mJy",
       "logM_HI", "incl_deg", "V_rot", "V_rot_i90", "HI_Ref", "NUV_r", "logM_star", "logM_bary"],
      rows2,
      "Yu et al. 2026 Table 2. CRITICAL CAVEAT: these are SINGLE-DISH (FAST / ALFALFA / HIPASS / "
      "Nancay) GLOBAL, SPATIALLY UNRESOLVED HI profiles. V_rot is derived from the global line "
      "width V85 de-projected with the OPTICAL inclination of the HOST; it does NOT separate the "
      "host-disk plane from the polar-ring plane, and the paper says so explicitly: 'PRGs do not "
      "follow a tight TFR or bTFR if the HI resides primarily in the host galaxy. But the scatter "
      "decreases significantly if we assume the gas is mainly distributed in the polar ring. "
      "Spatially resolved HI observations are essential to disentangle the gas distribution and "
      "kinematics in PRGs.' Treat V_rot here as a one-plane, geometry-ambiguous quantity. "
      "HI_Ref: 1=this work (FAST); 2=Masters+2014; 3=Haynes+2018 (ALFALFA); 4=Koribalski+2004 "
      "(HIPASS BGC); 5=Stark+2021; 6=Richter+1994; 7=van Driel+2002; 8=Springob+2005. "
      "logM_bary = log(M_star + 1.33 M_HI).",
      [{"name": "Galaxy", "unit": "catalogue designation"},
       {"name": "D_L_Mpc", "unit": "Mpc (Planck 2018 flat LCDM, H0=67.4)"},
       {"name": "V_c", "unit": "km/s (HI flux-weighted central velocity)"},
       {"name": "F_HI", "unit": "Jy km/s (total global HI flux)"},
       {"name": "V85", "unit": "km/s (width enclosing 85% of total flux)"},
       {"name": "A_F", "unit": "dimensionless (HI profile asymmetry, >=1)"},
       {"name": "K", "unit": "dimensionless (HI profile shape)"},
       {"name": "SNR", "unit": "dimensionless"},
       {"name": "sigma_mJy", "unit": "mJy (noise at ~6.4 km/s channel)"},
       {"name": "logM_HI", "unit": "log10(Msun); '<=' marks a 3sigma upper limit"},
       {"name": "incl_deg", "unit": "deg (OPTICAL inclination of the host)"},
       {"name": "V_rot", "unit": "km/s (V85/2 de-projected by the optical inclination)"},
       {"name": "V_rot_i90", "unit": "km/s (V85/2 assuming edge-on)"},
       {"name": "HI_Ref", "unit": "reference code for the HI data"},
       {"name": "NUV_r", "unit": "mag (NSA catalogue)"},
       {"name": "logM_star", "unit": "log10(Msun) (NSA, Chabrier IMF, BC03)"},
       {"name": "logM_bary", "unit": "log10(Msun)"}],
      expected=33)
