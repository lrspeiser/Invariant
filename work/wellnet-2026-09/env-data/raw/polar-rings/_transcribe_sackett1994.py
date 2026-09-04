"""Transcribe Sackett, Rix, Jarvis & Freeman 1994 (ApJ 436, 629; arXiv:astro-ph/9406015)
Table 2 -- 'Disk Major Axis Kinematics for NGC 4650A' -- verbatim to TSV.

The arXiv e-print SOURCE for this paper is dvips PostScript, not LaTeX, so the
table is transcribed from the arXiv PDF via pdftotext.  pdftotext emits the
three columns of this table as three separate runs (R block, V block, sigma
block), so they are re-zipped here and the three block lengths are asserted
equal.  Nothing is interpolated: if the blocks disagreed in length the script
would fail rather than silently drop rows.
"""
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "pdfs", "astro-ph_9406015_Sackett1994_NGC4650A_flathalo.pdf")

page = subprocess.run(["pdftotext", "-f", "7", "-l", "7", PDF, "-"],
                      capture_output=True, text=True, check=True).stdout
lines = [l.strip() for l in page.split("\n") if l.strip()]

i = next(k for k, l in enumerate(lines) if l.startswith("Table 2:"))
assert "Disk Major Axis Kinematics for NGC 4650A" in lines[i], lines[i]

# R block: standalone numbers; V block: "value error" pairs -- pdftotext
# alternates them.  Sigma block: one long trailing run after 'km s 1]'.
R, V, EV = [], [], []
for l in lines[i:]:
    if re.fullmatch(r"-?\d+\.\d+", l):
        R.append(l)
    elif re.fullmatch(r"-?\d+\.\d+ \d+\.\d+", l):
        a, b = l.split()
        V.append(a)
        EV.append(b)

sig_line = next(l for l in lines[i:] if l.startswith("km s 1]"))
nums = re.findall(r"\d+\.\d+", sig_line)
assert len(nums) % 2 == 0, "sigma block has an odd number of values"
SIG = nums[0::2]
ESIG = nums[1::2]

assert len(R) == len(V) == len(SIG), \
    "block lengths disagree: R=%d V=%d sigma=%d" % (len(R), len(V), len(SIG))
N = len(R)
assert N == 23, "expected 23 tabulated radii, got %d" % N
# monotonic radius check -- guards against a scrambled column
vals = [float(x) for x in R]
assert vals == sorted(vals), "radii are not monotonically increasing: %r" % vals

rows = list(zip(R, V, EV, SIG, ESIG))
path = os.path.join(HERE, "sackett1994_table2_NGC4650A_disk_kinematics.tsv")
header = ["R_arcsec", "V_kms", "e_V_kms", "sigma_kms", "e_sigma_kms"]
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

blob = open(path, "rb").read()
json.dump({
    "file": os.path.basename(path),
    "source_url": "https://arxiv.org/pdf/astro-ph/9406015",
    "source_file_within_archive": "page 7 of the arXiv PDF (the e-print SOURCE is dvips PostScript, not LaTeX)",
    "paper": "Sackett P.D., Rix H.-W., Jarvis B.J. & Freeman K.C. 1994, 'The Flattened Dark Halo of Polar Ring Galaxy NGC 4650A: A Conspiracy of Shapes?', ApJ 436, 629 (arXiv:astro-ph/9406015)",
    "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sha256": hashlib.sha256(blob).hexdigest(),
    "bytes": len(blob),
    "row_count": N,
    "column_count": len(header),
    "columns": [
        {"name": "R_arcsec", "unit": "arcsec (galactocentric radius along the CENTRAL DISK major axis, PA~63 deg; negative = approaching/NE side)"},
        {"name": "V_kms", "unit": "km/s (STELLAR line-of-sight velocity relative to systemic; NOT inclination-corrected, NOT asymmetric-drift corrected)"},
        {"name": "e_V_kms", "unit": "km/s (1-sigma)"},
        {"name": "sigma_kms", "unit": "km/s (stellar line-of-sight velocity dispersion, with 19 km/s already subtracted in quadrature for pixel interpolation + template mismatch)"},
        {"name": "e_sigma_kms", "unit": "km/s (1-sigma)"},
    ],
    "query": "GET https://arxiv.org/pdf/astro-ph/9406015 ; pdftotext -f 7 -l 7",
    "extraction": "Verbatim transcription of published Table 2. pdftotext emits the three table columns as three separate text runs; they were re-zipped and the block lengths asserted equal (23 each) plus a monotonic-radius check. No unit conversion, no derivation, no inclination correction applied.",
    "note": "THIS IS THE HOST-DISK-PLANE ROTATION CURVE OF NGC 4650A (stellar absorption lines, long slit at the central disk major axis). The POLAR-plane counterpart is NOT tabulated in this paper: Sect. 4.1 states the models were fitted to Halpha velocities (from Nicholson 1989's Fabry-Perot field) outward of 10 arcsec and required to match 'the deprojected HI speed of 105-125 km/s at 90 arcsec', with a 'measured polar HI linewidth of 240 km/s'. Radii from -17.33 to -23.93 arcsec on the NE side are absent because a foreground star contaminated those spectra; the +-20-30 arcsec data were combined into the single outermost bin on each side. Adopted distance: v_LG = 2625 km/s with H0 = 75 km/s/Mpc (=> ~35 Mpc); I-band m_I = 11.43. Model disk: M_D = 6.0e9 Msun, h = 4.7 arcsec, i = 68 deg (thin disk) or M_D = 5.25e9 Msun, h = 4.4 arcsec, i = 78 deg (thick disk); ring mass M_R = 10e9 / 9.7e9 Msun.",
}, open(path + ".manifest.json", "w", encoding="utf-8"), indent=2)
print("OK  %s  rows=%d cols=%d" % (os.path.basename(path), N, len(header)))
