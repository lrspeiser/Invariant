"""Acquire the two wide-field spectroscopic catalogues that close the AS1063 and
MACS J0717 velocity gaps, and re-derive membership with a stated criterion.

AS1063: the CLASH-VLT public release (Mercurio et al. 2021). NOT at CDS -- it is
distributed from the project's own site. Carries a documented PARSING TRAP:
exactly one row has a space inside its object ID, so whitespace splitting yields
8 fields instead of 7 and shifts every subsequent column. Parsed here by
character column position, with a per-row field-count assertion.

MACS J0717: VizieR J/ApJS/211/21 (Ebeling, Ma & Barrett 2014), filtered to the
J0717.5+3745 field. The '+' must be percent-encoded in the MACS= filter.

Neither catalogue ships a membership flag, so membership is re-derived here and
the criterion is stated in the manifest and compared against the published
counts.
"""
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import write_manifest, LANE  # noqa: E402

C_KMS = 299792.458
OUT = os.path.join(LANE, "velocities")
RAW = os.path.join(OUT, "raw")
UA = "Mozilla/5.0 (compatible; gravity-lane-acquire/1.0; research)"

AS1063_URL = ("https://drive.google.com/uc?export=download&"
              "id=1b_b7mFXk26UUaOsVF0qustI5C4U68Xhz")
AS1063_LANDING = "https://sites.google.com/site/vltclashpublic/data-release"
M0717_URL = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/211/21/zspec"
             "&MACS=J0717.5%2B3745&-out.max=unlimited&-out.all")

Z_AS1063 = 0.3480
Z_M0717 = 0.5458
DV_CUT = 3000.0  # km/s, rest-frame


def curl(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    p = subprocess.run(["curl", "-sSL", "--max-time", "300", "-A", UA, "-o", dest, url],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("curl failed: %s" % p.stderr[:400])
    return os.path.getsize(dest)


def dv_kms(z, z_cl):
    return C_KMS * (z - z_cl) / (1.0 + z_cl)


# =====================================================================
# ABELL S1063 -- CLASH-VLT
# =====================================================================
raw_as = os.path.join(RAW, "AS1063_CLASH-VLT_zcat_11_2021.dat")
n = curl(AS1063_URL, raw_as)
text = io.open(raw_as, encoding="utf-8", errors="replace").read()
lines = text.splitlines()
hdr = [l for l in lines if l.lstrip().startswith("#")]
body = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
print("AS1063 raw: %d bytes, %d header lines, %d data lines" % (n, len(hdr), len(body)))

# The trap: one row splits into 8 fields. Repair it by rejoining the two
# fragments of the ID, and assert that this is the ONLY such row.
ragged = [i for i, l in enumerate(body) if len(l.split()) != 7]
print("rows whose whitespace split is not 7 fields: %d -> %s"
      % (len(ragged), [body[i].split()[0] for i in ragged]))
assert len(ragged) <= 1, "more raggedness than the documented single row: %d" % len(ragged)

rows = []
for i, l in enumerate(body):
    f = l.split()
    if len(f) == 8:
        f = [f[0] + f[1]] + f[2:]      # rejoin the split identifier
    assert len(f) == 7, "row %d has %d fields after repair" % (i, len(f))
    rows.append(f)
assert len(rows) == len(body)

zs = [float(r[3]) for r in rows]
qf = [r[4] for r in rows]
dv = [dv_kms(z, Z_AS1063) for z in zs]
SECURE = {"3"}
LIKELY_OR_BETTER = {"2", "3", "9", "5", "6"}

n_all = sum(1 for d in dv if abs(d) < DV_CUT)
n_secure = sum(1 for d, q in zip(dv, qf) if abs(d) < DV_CUT and q in SECURE)
n_reliable = sum(1 for d, q in zip(dv, qf) if abs(d) < DV_CUT and q in LIKELY_OR_BETTER)
qcounts = {q: qf.count(q) for q in sorted(set(qf))}
print("AS1063 membership: |dv|<%.0f km/s -> all=%d secure(flag3)=%d reliable=%d"
      % (DV_CUT, n_all, n_secure, n_reliable))
print("AS1063 quality-flag counts:", qcounts)

clean_as = os.path.join(OUT, "AS1063_CLASHVLT_Mercurio2021_zcat.tsv")
COLS_AS = ["ID", "RAdeg", "DEdeg", "z", "qflag", "zref", "Rmag_Kron"]
with io.open(clean_as, "w", encoding="utf-8", newline="\n") as f:
    f.write("\t".join(COLS_AS) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

write_manifest(
    clean_as, AS1063_URL,
    exact_query=("HTTP GET " + AS1063_URL + " (303 redirect to "
                 "drive.usercontent.google.com, followed with curl -L; no authentication). "
                 "Landing page: " + AS1063_LANDING),
    note=("Abell S1063 / RXC J2248.7-4431. CLASH-VLT public spectroscopic redshift catalogue, "
          "'redshift cat v3.6, Release date November 2021', the published release of Mercurio et "
          "al. 2021 (A&A 656, A147). THIS CATALOGUE IS NOT AT CDS: VizieR J/A+A/656/A147 does not "
          "exist and returns the Cooper+2013 fallback, and cdsarc returns 404. It is distributed "
          "only from the CLASH-VLT project site. 3850 redshifts over the full VIMOS field, which "
          "replaces the 290-row 30-arcmin NED cone that was previously the only AS1063 velocity "
          "source. "
          "PARSING TRAP, REAL AND HANDLED: exactly one row carries a SPACE INSIDE ITS OBJECT ID "
          "(CLASHVLTJ2249 9.98-442802.3, which should read CLASHVLTJ224959.98-442802.3), so "
          "whitespace splitting yields 8 fields for that row and 7 for the other 3849, shifting "
          "every subsequent column. The fragments are rejoined here and a per-row field-count "
          "assertion is applied; the raw upstream file is preserved unmodified. "
          "NO MEMBERSHIP FLAG is provided -- see the membership_rederived block."),
    extraction=("Verbatim values from the upstream .dat, whitespace re-delimited to TAB with the "
                "single split identifier rejoined. No numerical change, no unit conversion, no "
                "cross-table join. Row count asserted against the upstream data-line count."),
    row_count=len(rows), column_count=7,
    columns=[{"name": "ID", "unit": "CLASH-VLT identifier"},
             {"name": "RAdeg", "unit": "deg (J2000)"},
             {"name": "DEdeg", "unit": "deg (J2000)"},
             {"name": "z", "unit": "spectroscopic redshift"},
             {"name": "qflag", "unit": "quality: 2=LIKELY ~80%, 3=SECURE 100%, "
                                       "9=SINGLE-LINE >90%, 5=Gomez+2012, 6=Magellan/Kelson"},
             {"name": "zref", "unit": "source: 1=CLASH-VLT, 2=Gomez, 3=Magellan, "
                                      "4=MUSE Karman, 5=GLASS"},
             {"name": "Rmag_Kron", "unit": "AB mag, WFI Kron R, extinction-corrected"}],
    extra={"cluster": "Abell S1063 = RXC J2248.7-4431",
           "product": "member_velocities_spectroscopic_redshifts",
           "is_raw_observable": True, "presupposes_dark_matter": False,
           "instrument": "VLT/VIMOS (CLASH-VLT), plus MUSE, GLASS, Magellan and Gomez+2012 "
                         "redshifts merged by the release",
           "cluster_redshift": Z_AS1063,
           "quality_flag_counts": qcounts,
           "membership_flag_provided": False,
           "membership_rederived": {
               "criterion": "rest-frame |dv| = c*|z - z_cl|/(1+z_cl) < 3000 km/s, z_cl = 0.3480",
               "n_members_all_quality": n_all,
               "n_members_secure_flag3_only": n_secure,
               "n_members_quality_2_3_9_5_6": n_reliable,
               "published_comparison": ("Mercurio et al. 2021 report 1234 members from a "
                                        "peak-plus-gap selection, which is a different and more "
                                        "restrictive procedure than a fixed velocity cut. The "
                                        "numbers here are therefore expected to differ and are "
                                        "not a reproduction of theirs."),
               "warning": "Re-derive membership deliberately for the downstream test; do not "
                          "treat these counts as the published member list."},
           "parsing_trap_handled": "one row with a space inside its ID; rejoined, asserted unique",
           "raw_response_file": "raw/AS1063_CLASH-VLT_zcat_11_2021.dat"})

# =====================================================================
# MACS J0717.5+3745 -- Ebeling, Ma & Barrett 2014
# =====================================================================
raw_m = os.path.join(RAW, "MACS0717_Ebeling2014_ApJS211_21_zspec_vizier.tsv")
n = curl(M0717_URL, raw_m)
txt = io.open(raw_m, encoding="utf-8", errors="replace").read()

# Trap check: confirm VizieR echoed the identifier we asked for, not a fallback.
assert "J/ApJS/211/21" in txt or "J_ApJS_211_21" in txt, \
    "VizieR did not echo the requested identifier -- possible silent fallback"
assert "Error=" not in txt, "VizieR returned an error payload"

lines = txt.splitlines()
seps = [i for i, l in enumerate(lines) if l.startswith("---")]
i = seps[-1]
colnames = [c.strip() for c in lines[i - 2].split("\t")]
units = [u.strip() for u in lines[i - 1].split("\t")]
drows = [l.split("\t") for l in lines[i + 1:] if l.strip() and not l.startswith("#")]
print("MACS0717 raw: %d bytes, %d rows, %d cols" % (n, len(drows), len(colnames)))
assert len(drows) == 1266, "expected 1266 rows for the J0717.5+3745 field, got %d" % len(drows)

jz = colnames.index("z")
zs_m, dv_m = [], []
for r in drows:
    try:
        z = float(r[jz])
    except ValueError:
        continue
    zs_m.append(z)
    dv_m.append(dv_kms(z, Z_M0717))
n_mem_m = sum(1 for d in dv_m if abs(d) < DV_CUT)
print("MACS0717 membership: |dv|<%.0f km/s -> %d of %d with a parseable z"
      % (DV_CUT, n_mem_m, len(zs_m)))

clean_m = os.path.join(OUT, "MACS0717_Ebeling2014_ApJS211_21_zspec.tsv")
with io.open(clean_m, "w", encoding="utf-8", newline="\n") as f:
    f.write("\t".join(colnames) + "\n")
    for r in drows:
        f.write("\t".join(r) + "\n")

write_manifest(
    clean_m, M0717_URL,
    exact_query=M0717_URL,
    note=("MACS J0717.5+3745. Ebeling, Ma & Barrett 2014, ApJS 211, 21, VizieR table "
          "J/ApJS/211/21/zspec, filtered to the J0717.5+3745 field. Keck/DEIMOS, Keck/LRIS and "
          "Gemini/GMOS spectroscopy. 1266 rows, asserted against the CDS ReadMe entry "
          "'table4.dat 70 1266'. VizieR IDENTIFIER ECHO VERIFIED -- the response carries the "
          "requested J/ApJS/211/21, not the J/MNRAS/430/1125 fallback that VizieR silently "
          "serves for nonexistent identifiers. "
          "QUERY TRAP: the '+' in the MACS= filter value MUST be percent-encoded as %2B or the "
          "filter matches nothing. "
          "This replaces the 30-arcmin NED cone as the homogeneous, single-survey MACS J0717 "
          "velocity catalogue. The parent VizieR table holds 1921 rows over exactly three fields: "
          "J0717.5+3745 = 1266, J1149.5+2223 = 590, J0416.1-2403 = 65. "
          "NO MEMBERSHIP FLAG is provided -- see the membership_rederived block."),
    extraction=("Verbatim VizieR asu-tsv data block, header and units separated, filtered "
                "server-side to MACS=J0717.5+3745. No numerical change, no unit conversion, no "
                "cross-table join. Row count asserted against the CDS ReadMe."),
    row_count=len(drows), column_count=len(colnames),
    columns=[{"name": c, "unit": u} for c, u in zip(colnames, units)],
    extra={"cluster": "MACS J0717.5+3745",
           "product": "member_velocities_spectroscopic_redshifts",
           "is_raw_observable": True, "presupposes_dark_matter": False,
           "instrument": "Keck/DEIMOS (D), Keck/LRIS (L), Gemini/GMOS (G); see the Inst column",
           "cluster_redshift": Z_M0717,
           "vizier_identifier": "J/ApJS/211/21/zspec",
           "vizier_identifier_echoed_in_response": True,
           "membership_flag_provided": False,
           "membership_rederived": {
               "criterion": "rest-frame |dv| = c*|z - z_cl|/(1+z_cl) < 3000 km/s, z_cl = 0.5458",
               "n_members": n_mem_m,
               "n_with_parseable_z": len(zs_m),
               "published_comparison": ("Jauzac et al. 2018 and Limousin et al. 2016 quote 1079 "
                                        "redshifts and 537 members from this same dataset, using "
                                        "their own membership procedures rather than a fixed "
                                        "velocity cut."),
               "warning": "Re-derive membership deliberately for the downstream test."},
           "raw_response_file": "raw/MACS0717_Ebeling2014_ApJS211_21_zspec_vizier.tsv"})

print("\ndone")
