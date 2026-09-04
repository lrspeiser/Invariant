"""Transcribe Jozsa, Oosterloo, Morganti, Klein & Erben 2009 (A&A 494, 489;
arXiv:0810.3025) Table 5 -- the radially dependent best-fit TiRiFiC tilted-ring
parameters for NGC 2685 (the Helix / Spindle) -- verbatim to TSV.

This is the only fully TABULATED rotation curve in this lane that also carries
the ORIENTATION (inclination, position angle and 3-D spin normal) of every
ring, i.e. V_rot sampled as the orbit plane swings through ~125 deg of position
angle within one baryonic system.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "eprints", "0810.3025_Josza2009_NGC2685_kinmodel", "xx.tex")
TEX = open(SRC, encoding="utf-8", errors="replace").read()

i = TEX.index("\\label{Tab_5}")
i = TEX.index("\\hline", TEX.index("\\begin{tabular}", i))
j = TEX.index("\\end{tabular}", i)

rows = []
for line in TEX[i:j].split("\\\\"):
    line = re.sub(r"^\s*(\\hline\s*)+", "", line).strip()
    if "&" not in line:
        continue
    cells = []
    for c in line.split("&"):
        c = c.strip()
        c = re.sub(r"\$|\\", "", c)
        cells.append(c.strip())
    if not re.fullmatch(r"-?\d+(\.\d+)?", cells[0]):
        continue
    rows.append(cells)

assert len(rows) == 21, "expected 21 tilted rings, got %d" % len(rows)
for r in rows:
    assert len(r) == 21, "row r_p=%s has %d cells, expected 21" % (r[0], len(r))
radii = [float(r[0]) for r in rows]
assert radii == sorted(radii), "radii not monotonic: %r" % radii
assert radii[0] == 0.0 and radii[-1] == 420.0

header = ["r_arcsec", "r_kpc", "e_r_kpc", "I_tot_faceon", "e_I_tot_faceon",
          "N_HI_faceon", "e_N_HI_faceon", "sigma_faceon", "e_sigma_faceon",
          "V_rot", "e_V_rot", "incl_deg", "e_incl_deg", "PA_deg", "e_PA_deg",
          "n_W", "e_n_W", "n_N", "e_n_N", "n_LOS", "e_n_LOS"]
path = os.path.join(HERE, "josza2009_table5_NGC2685_tiltedring.tsv")
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

blob = open(path, "rb").read()
json.dump({
    "file": os.path.basename(path),
    "source_url": "https://arxiv.org/e-print/0810.3025",
    "source_file_within_archive": "xx.tex (Table 5, 'Radially dependent best-fit parameters')",
    "paper": "Jozsa G.I.G., Oosterloo T.A., Morganti R., Klein U., Erben T. 2009, 'Kinematic modelling of disk galaxies. III. The warped Spindle NGC 2685', A&A 494, 489 (arXiv:0810.3025)",
    "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sha256": hashlib.sha256(blob).hexdigest(),
    "bytes": len(blob),
    "row_count": len(rows),
    "column_count": len(header),
    "columns": [
        {"name": "r_arcsec", "unit": "arcsec (tilted-ring radius, projected)"},
        {"name": "r_kpc", "unit": "kpc"},
        {"name": "e_r_kpc", "unit": "kpc"},
        {"name": "I_tot_faceon", "unit": "Jy km/s (face-on surface brightness)"},
        {"name": "e_I_tot_faceon", "unit": "Jy km/s"},
        {"name": "N_HI_faceon", "unit": "1e19 atoms/cm^2 (face-on HI column density)"},
        {"name": "e_N_HI_faceon", "unit": "1e19 atoms/cm^2"},
        {"name": "sigma_faceon", "unit": "Msun/pc^2 (face-on HI surface density; NOT helium-corrected)"},
        {"name": "e_sigma_faceon", "unit": "Msun/pc^2"},
        {"name": "V_rot", "unit": "km/s (DEPROJECTED rotation velocity of the ring)"},
        {"name": "e_V_rot", "unit": "km/s"},
        {"name": "incl_deg", "unit": "deg (ring inclination)"},
        {"name": "e_incl_deg", "unit": "deg"},
        {"name": "PA_deg", "unit": "deg (ring position angle)"},
        {"name": "e_PA_deg", "unit": "deg"},
        {"name": "n_W", "unit": "dimensionless (spin normal vector component towards W)"},
        {"name": "e_n_W", "unit": "dimensionless"},
        {"name": "n_N", "unit": "dimensionless (spin normal vector component towards N)"},
        {"name": "e_n_N", "unit": "dimensionless"},
        {"name": "n_LOS", "unit": "dimensionless (spin normal vector component towards the observer)"},
        {"name": "e_n_LOS", "unit": "dimensionless"},
    ],
    "query": "GET https://arxiv.org/e-print/0810.3025 ; transcribed from LaTeX Table 5",
    "extraction": "Verbatim transcription of published Table 5. 21 rings asserted, 21 cells per row asserted, monotonic radius asserted, endpoints asserted (0 and 420 arcsec). No unit conversion, no derivation.",
    "note": "PUBLISHED-CAPTION DEFECT, RECORDED NOT REPAIRED: the caption enumerates items (1)-(23) and lists '(16) Inclination (deg)' a second time, but the tabular header and every data row carry exactly 21 fields. The column names emitted here follow the HEADER ROW of the table, which is self-consistent: r_p, r_t, dr_t, I_tot_f, dI, N_HI, dN, sigma, dsigma, V_rot, dV_rot, i, di, pa, dpa, n_W, dn_W, n_N, dn_N, n_LOS, dn_LOS. WHAT THIS IS: a TiRiFiC 3-D tilted-ring fit to WSRT HI of NGC 2685 (PRC A-03, the Helix/Spindle). The position angle swings from 245 deg at r=0.88 kpc through 205 deg at 2.65 kpc to ~120 deg beyond 7 kpc, and the inclination from 35 to 70 deg, so V_rot is sampled across ~125 deg of orbit-plane orientation WITHIN ONE GALAXY. IMPORTANT CAVEAT: this paper's conclusion is that NGC 2685 is NOT a classical polar ring but an extremely warped, kinematically COHERENT single HI disk, inclined ~70 deg to the main lenticular body in the inner parts and becoming coplanar with it at large radii. It is therefore a continuous-warp system, not two independent orthogonal rotators; treat it as a direction-scan of one disk, not as two-plane kinematics in the PRG sense.",
}, open(path + ".manifest.json", "w", encoding="utf-8"), indent=2)
print("OK  %s  rows=%d cols=%d" % (os.path.basename(path), len(rows), len(header)))
