"""Verbatim transcription of the Milky Way circular-velocity curve from
Eilers, Hogg, Rix & Ness 2019, ApJ 871, 120 (arXiv:1810.09466).

WHY THIS IS HERE: this lane's whole purpose is to pair an OUT-OF-PLANE tracer
with an IN-PLANE rotation curve in the SAME system. For the Milky Way the
out-of-plane tracers are the stellar streams; this is the matching in-plane
measurement. Neither Eilers+2019 nor Mroz+2019 has a CDS/VizieR catalogue
(both ReadMe URLs return HTTP 404), so the table is taken from the LaTeX source.
"""
import os
import re
import sys
import tarfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from _manifest import write_manifest  # noqa: E402

BS = chr(92)
t = tarfile.open(os.path.join(BASE, "eilers2019_paper_1810.09466.tar.gz"))
txt = t.extractfile([m for m in t.getmembers() if m.name == "main.tex"][0]).read().decode("utf-8", "replace")

envs = re.findall(BS + BS + r"begin\{(deluxetable\*?|table\*?)\}(.*?)" + BS + BS + r"end\{\1\}",
                  txt, re.S)
print("table environments:", len(envs))
for i, (kind, body) in enumerate(envs):
    cap = re.search(r"caption\{(.{0,140})", body, re.S)
    print("  %d) %-14s %s" % (i, kind, (cap.group(1)[:120].replace("\n", " ") if cap else "")))

# The circular-velocity table: rows of  R  v_c  -err  +err
target = None
for kind, body in envs:
    if re.search(r"circular velocity|rotation curve|v_?c", body, re.I):
        if re.search(r"\d+\.\d+\s*&\s*\d+\.\d+", body):
            target = body
            break
if target is None:
    print("!! no numeric circular-velocity table found in the source")
    sys.exit(1)

rows = []
for line in target.split(BS * 2):
    line = re.sub(BS + BS + r"(hline|tableline|startdata|enddata|nodata)", " ", line)
    line = line.strip()
    if not line or "&" not in line:
        continue
    cells = [re.sub(r"[$" + BS + r"{}]", "", c).strip() for c in line.split("&")]
    try:
        vals = [float(c) for c in cells]
    except ValueError:
        continue
    rows.append([("%g" % v) for v in vals])

ncol = max(len(r) for r in rows)
rows = [r for r in rows if len(r) == ncol]
print("\nnumeric rows recovered: %d  (ncol=%d)" % (len(rows), ncol))
for r in rows[:4]:
    print("   ", r)
print("   ...")
for r in rows[-2:]:
    print("   ", r)

COLS = {4: ["R_kpc", "vc_kms", "e_vc_minus_kms", "e_vc_plus_kms"],
        3: ["R_kpc", "vc_kms", "e_vc_kms"],
        5: ["R_kpc", "vc_kms", "e_vc_minus_kms", "e_vc_plus_kms", "e_vc_sys_kms"]}
cols = COLS.get(ncol, ["col%d" % (i + 1) for i in range(ncol)])
units = ["kpc"] + ["km/s"] * (ncol - 1)

out = os.path.join(BASE, "stream_eilers2019_MW_rotation_curve.tsv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")
print("wrote", out)

# The paper states the curve spans 5-25 kpc; assert the transcription agrees.
rmin = min(float(r[0]) for r in rows)
rmax = max(float(r[0]) for r in rows)
print("R range: %.2f - %.2f kpc" % (rmin, rmax))
assert 4.0 <= rmin <= 6.5, "R_min %.2f outside the paper's stated 5 kpc start" % rmin
assert 23.0 <= rmax <= 26.5, "R_max %.2f outside the paper's stated 25 kpc end" % rmax
print("ASSERT OK: radial range matches the paper's stated 5-25 kpc coverage")

write_manifest(
    out,
    source_url="https://arxiv.org/e-print/1810.09466",
    source_file_within_archive="main.tex",
    query="extract_eilers2019_mw_rc.py -- transcription of the circular-velocity table",
    columns=[{"name": c, "unit": u} for c, u in zip(cols, units)],
    row_count=len(rows),
    extraction="Verbatim transcription of the published LaTeX table. No unit conversion, no "
               "derivation, no cross-table join. Radial range asserted against the paper's "
               "stated 5-25 kpc coverage.",
    measurement_or_model=(
        "MEASUREMENT of the IN-PLANE circular velocity curve. Derived from Gaia DR2 + APOGEE "
        "luminous red giants via Jeans modelling of an axisymmetric disc; it assumes "
        "axisymmetry and dynamical equilibrium of the DISC TRACERS, but assumes NO dark-matter "
        "halo and no parametric mass model. The published v_c(R) is an inferred kinematic "
        "quantity, not a raw observable -- the raw observables are the stars' positions, "
        "distances, proper motions and line-of-sight velocities."),
    note="Eilers, Hogg, Rix & Ness 2019, ApJ 871, 120: 'The Circular Velocity Curve of the "
         "Milky Way from 5 to 25 kpc'. NOT AVAILABLE IN VIZIER/CDS: "
         "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/871/120/ReadMe returns HTTP 404, as does the "
         "equivalent for Mroz+2019 (J/ApJ/870/L10). This is the IN-PLANE counterpart that pairs "
         "with the Milky Way stellar streams, which are the OUT-OF-PLANE tracers of the same "
         "system.",
)
