# -*- coding: utf-8 -*-
"""Product 2 (BCG) -- HFF-DeepSpace bright-cluster-galaxy photometry (Shipley+2018)."""
import os, re, json, hashlib, datetime, math

ROOT = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
NOW  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def manifest(path, source_url, exact_query, row_count, columns, extraction, note,
             raw_response_file, extra=None):
    m = {"file": os.path.basename(path), "source_url": source_url, "exact_query": exact_query,
         "retrieved_utc": NOW, "sha256": sha256(path), "bytes": os.path.getsize(path),
         "row_count": row_count, "column_count": len(columns), "columns": columns,
         "extraction": extraction, "note": note, "raw_response_file": raw_response_file}
    if extra: m.update(extra)
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print("  manifest ->", os.path.basename(path) + ".manifest.json")

def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")
    print("WROTE", os.path.basename(path), "rows=", len(rows))

def parse_vizier_tsv(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    lines = [l.rstrip("\r") for l in txt.split("\n")]
    if any("Error=Table or Catalog not found" in l or "does not exist in catalog" in l for l in lines):
        raise RuntimeError("VizieR NOT FOUND response: " + path)
    body = [l for l in lines if not l.startswith("#")]
    di = None
    for i, l in enumerate(body):
        if l.startswith("---") and "\t" in l:
            di = i; break
    assert di is not None, "no dashes separator"
    header = body[di - 2].split("\t")
    units = body[di - 1].split("\t")
    rows = [l.split("\t") for l in body[di + 1:] if l.strip() and not l.startswith("#")]
    rows = [r for r in rows if len(r) >= len(header) - 2]
    return header, units, rows

print("=" * 70); print("PRODUCT 2 -- Shipley+2018 HFF-DeepSpace bCG photometry   ", NOW); print("=" * 70)

raw = os.path.join(ROOT, "bcg", "raw", "shipley2018_clugal_bCG_raw.tsv")
hdr, units, rows = parse_vizier_tsv(raw)
assert len(rows) == 391, "expected 391 UseKs==2 rows, got %d" % len(rows)
CL = hdr.index("Cl")
counts = {}
for r in rows:
    counts[r[CL].strip()] = counts.get(r[CL].strip(), 0) + 1
print("  per-cluster bCG counts:", counts)
assert set(counts) == {"A1063-clu", "A2744-clu", "A370-clu", "M0416-clu", "M0717-clu", "M1149-clu"}, counts

out = os.path.join(ROOT, "bcg", "shipley2018_hff_bCG_photometry.tsv")
write_tsv(out, hdr, rows)
cols = [{"name": h, "unit": (units[i].strip() if i < len(units) else "")} for i, h in enumerate(hdr)]
SHIPLEY_NOTE = (
    "*** PRODUCT 2 (BCG photometry) for the six HFF clusters: Abell 2744, MACSJ0416.1-2403, MACSJ0717.5+3745, "
    "MACSJ1149.5+2223, Abell S1063 (A1063-clu), Abell 370. Abell 2029 is NOT an HFF cluster and is absent. "
    "Shipley et al. 2018, ApJS 235, 14 (HFF-DeepSpace v3.9). "
    "SELECTION: rows with UseKs==2, which the catalogue ReadMe defines as 'The source is a selected \"bright\" "
    "cluster galaxy (bCG) for modeling (see Section 3.1.1)'. This is the authors' own bCG modelling set -- it is "
    "the set of BRIGHT CLUSTER GALAXIES, typically ~35-90 per cluster, NOT a single BCG per cluster. Use "
    "shipley2018_hff_BCG_brightest_per_cluster.tsv for a single brightest-galaxy-per-cluster extract. "
    "BANDS: up to 17 filters, HST/ACS + HST/WFC3 F225W...F160W, plus VLT-HAWKI/Keck-MOSFIRE Ks and Spitzer/IRAC "
    "ch1-ch4. Fluxes are total f_nu with ZP(AB)=25, i.e. m_AB = -2.5*log10(flux) + 25 (ReadMe note G1); fluxes are "
    "already corrected for the zeropoint correction and for Galactic extinction. "
    "*** CRITICAL CAVEAT -- THE bCG + ICL LIGHT IS MODELLED AND SUBTRACTED. *** Shipley et al. build models of the "
    "combined bCG+ICL light and subtract them before detecting and measuring sources, so the photometry of ALL "
    "OTHER (non-bCG) catalogue sources in these fields is measured on bCG+ICL-SUBTRACTED images. Per-band flag "
    "f_F*** == 4 means 'BCGs, not modeled out' (ReadMe note G2) and marks bands where that subtraction was not "
    "applied. Consequently this catalogue must NOT be used to measure the ICL: the ICL has been removed by "
    "construction. It is the BCG/bright-member photometry source only. "
    "NO LIGHT PROFILE: this is aperture/total photometry only -- no Sersic index, no effective radius, no "
    "axis ratio, no position angle, no mu(r). HFF-DeepSpace publishes the bCG+ICL model IMAGES, but no tabulated "
    "structural parameters, and the VizieR deposit (tables table1, clugal, pargal) contains NO stellar-mass table, "
    "so no BCG stellar masses are available from this source. "
    "MASS-MODEL DEPENDENCE: none for the photometry itself.")
manifest(out, "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "-source=J/ApJS/235/14/clugal&UseKs==2&-out=**&-out.max=unlimited",
    len(rows), cols,
    "VizieR asu-tsv constrained query on catalogue J/ApJS/235/14 table 'clugal' (44465 rows total) selecting the "
    "391 rows flagged UseKs==2 (bCG modelling set). Per-cluster counts asserted to cover exactly the six HFF "
    "cluster fields. Response verified to contain real rows and no 'Table or Catalog not found' error.",
    SHIPLEY_NOTE, "bcg/raw/shipley2018_clugal_bCG_raw.tsv",
    {"catalogue_id_echoed": "J/ApJS/235/14/clugal", "vizier_table_total_rows": 44465,
     "vizier_readme": "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/235/14/ReadMe"})

# ---------------------------------------------------------------
# Derived: brightest bCG per cluster near the Montes&Trujillo 2018 centre
# ---------------------------------------------------------------
print("\n[7b] Derived: brightest bCG per cluster")
CENTRES = {  # from bcg/montes_trujillo2018_hff_centres_sblimits.tsv (MT2018 Table 1): centre + cluster z
    "A2744-clu": ("Abell 2744",       "00:14:21.2", "-30:23:50.1", 0.308),
    "M0416-clu": ("MACSJ0416.1-2403", "04:16:08.9", "-24:04:28.7", 0.396),
    "M0717-clu": ("MACSJ0717.5+3745", "07:17:34.0", "+37:44:49.0", 0.545),
    "M1149-clu": ("MACSJ1149.5+2223", "11:49:36.3", "+22:23:58.1", 0.543),
    "A1063-clu": ("Abell S1063",      "22:48:44.4", "-44:31:48.5", 0.348),
    "A370-clu":  ("Abell 370",        "02:39:52.9", "-01:34:36.5", 0.375),
}
DZ = 0.02   # |zspec - z_cluster| tolerance for spectroscopic membership
def hms2deg(s):
    h, m, sec = [float(x) for x in s.split(":")]
    return 15.0 * (h + m / 60.0 + sec / 3600.0)
def dms2deg(s):
    sign = -1.0 if s.strip().startswith("-") else 1.0
    d, m, sec = [float(x) for x in s.strip().lstrip("+-").split(":")]
    return sign * (d + m / 60.0 + sec / 3600.0)
def sep_arcsec(ra1, de1, ra2, de2):
    r1, d1, r2, d2 = map(math.radians, (ra1, de1, ra2, de2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 3600.0

iRA, iDE, iID, iz = hdr.index("RAJ2000"), hdr.index("DEJ2000"), hdr.index("ID"), hdr.index("zspec")
iF160, ieF160 = hdr.index("FF160W"), hdr.index("e_FF160W")
iF814, ieF814 = hdr.index("FF814W"), hdr.index("e_FF814W")
def fnum(s):
    try: return float(s)
    except Exception: return None
def ab(f):
    return round(-2.5 * math.log10(f) + 25.0, 4) if (f and f > 0) else None

SEARCH_R = 60.0  # arcsec
out_rows = []
rejected = []
for cl, (full, ra_s, de_s, zcl) in CENTRES.items():
    cra, cde = hms2deg(ra_s), dms2deg(de_s)
    cand = []
    for r in rows:
        if r[CL].strip() != cl: continue
        ra, de = fnum(r[iRA]), fnum(r[iDE])
        if ra is None or de is None: continue
        d = sep_arcsec(ra, de, cra, cde)
        if d > SEARCH_R: continue
        f160 = fnum(r[iF160])
        if f160 is None or f160 <= 0: continue
        cand.append((f160, d, r))
    assert cand, "no bCG candidate within %.0f arcsec of %s centre" % (SEARCH_R, full)
    cand.sort(key=lambda t: -t[0])          # brightest F160W flux first
    n_all = len(cand)
    # REJECT spectroscopically confirmed non-members (foreground/background interlopers).
    keep = []
    for f160, d, r in cand:
        z = fnum(r[iz])
        if z is not None and z > 0 and abs(z - zcl) > DZ:
            if len(rejected) < 40:
                rejected.append((full, r[iID], z, zcl, ab(f160), d))
            continue
        keep.append((f160, d, r))
    assert keep, "all candidates rejected as non-members for %s" % full
    f160, d, r = keep[0]
    z = fnum(r[iz])
    zflag = ("spec-confirmed member" if (z is not None and z > 0)
             else "no zspec (membership unverified)")
    out_rows.append([full, cl, r[iID], r[iRA], r[iDE], r[iz], zcl, zflag,
                     round(d, 2), n_all, n_all - len(keep),
                     r[iF160], r[ieF160], ab(fnum(r[iF160])),
                     r[iF814], r[ieF814], ab(fnum(r[iF814]))])
    print("   %-18s ID=%-6s sep=%5.1f\" mF160W=%.3f  [%d cand, %d rejected as non-members]  %s"
          % (full, r[iID], d, ab(f160), n_all, n_all - len(keep), zflag))
if rejected:
    print("   -- spectroscopic non-members rejected (brightest few):")
    for full, i, z, zcl, mag, d in rejected[:6]:
        print("      %-18s ID=%-6s zspec=%.4f vs z_cl=%.3f  mF160W=%.2f sep=%.1f\"" % (full, i, z, zcl, mag, d))

hdrb = ["Cluster", "Cl_field", "HFFDS_ID", "RAdeg_J2000", "DEdeg_J2000", "zspec", "z_cluster",
        "membership_flag", "sep_from_MT2018_centre_arcsec", "n_bCG_candidates_within_60arcsec",
        "n_rejected_spec_nonmembers",
        "FF160W_fnu_ZP25", "e_FF160W", "mAB_F160W", "FF814W_fnu_ZP25", "e_FF814W", "mAB_F814W"]
out = os.path.join(ROOT, "bcg", "shipley2018_hff_BCG_brightest_per_cluster.tsv")
write_tsv(out, hdrb, out_rows)
manifest(out, "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "-source=J/ApJS/235/14/clugal&UseKs==2&-out=**&-out.max=unlimited  [then DERIVED: brightest F160W source "
    "within 60 arcsec of the Montes & Trujillo 2018 Table 1 cluster centre that is not a spectroscopically "
    "confirmed non-member, per cluster field]",
    len(out_rows),
    [{"name": "Cluster", "unit": ""}, {"name": "Cl_field", "unit": ""}, {"name": "HFFDS_ID", "unit": ""},
     {"name": "RAdeg_J2000", "unit": "deg"}, {"name": "DEdeg_J2000", "unit": "deg"},
     {"name": "zspec", "unit": ""}, {"name": "z_cluster", "unit": ""},
     {"name": "membership_flag", "unit": ""},
     {"name": "sep_from_MT2018_centre_arcsec", "unit": "arcsec"},
     {"name": "n_bCG_candidates_within_60arcsec", "unit": ""},
     {"name": "n_rejected_spec_nonmembers", "unit": ""},
     {"name": "FF160W_fnu_ZP25", "unit": "f_nu, ZP(AB)=25"}, {"name": "e_FF160W", "unit": "f_nu, ZP(AB)=25"},
     {"name": "mAB_F160W", "unit": "AB mag"},
     {"name": "FF814W_fnu_ZP25", "unit": "f_nu, ZP(AB)=25"}, {"name": "e_FF814W", "unit": "f_nu, ZP(AB)=25"},
     {"name": "mAB_F814W", "unit": "AB mag"}],
    "*** DERIVED PRODUCT, NOT A PUBLISHED BCG LIST. *** Built from shipley2018_hff_bCG_photometry.tsv by taking, "
    "in each of the six HFF cluster fields, the UseKs==2 (bCG-flagged) source with the largest total F160W flux "
    "lying within 60 arcsec of the Montes & Trujillo 2018 Table 1 cluster centre, AFTER discarding any candidate "
    "whose catalogue zspec differs from the cluster redshift by more than 0.02 (a spectroscopically confirmed "
    "foreground/background interloper). "
    "*** THIS MEMBERSHIP CUT IS LOAD-BEARING: without it the brightest bCG-flagged source within 60 arcsec of the "
    "MACSJ0416.1-2403 centre is HFF-DeepSpace ID 20004 at zspec = 0.1137, a bright FOREGROUND galaxy, not the "
    "cluster BCG at z = 0.396. Montes & Trujillo 2018 independently flag this object, stating that for M0416, "
    "M0717 and M1149 'the presence of a bright foreground source close to the cluster made us mask a part of the "
    "image to avoid contamination of the ICL'. *** "
    "AB magnitudes computed from the catalogue fluxes as m_AB = -2.5*log10(f) + 25 per ReadMe note G1. "
    "The full selection rule is recorded here so it can be reproduced or overridden; "
    "'n_bCG_candidates_within_60arcsec', 'n_rejected_spec_nonmembers', 'membership_flag' and "
    "'sep_from_MT2018_centre_arcsec' are given so every choice can be audited. Rows whose membership_flag reads "
    "'no zspec (membership unverified)' have NO spectroscopic redshift in the catalogue and were accepted on "
    "brightness and position alone.",
    "Brightest bright-cluster-galaxy per HFF cluster field. "
    "*** CAVEAT: Abell 2744, MACSJ0416, MACSJ0717 and MACSJ1149 are MULTI-BCG / merging systems -- Montes & "
    "Trujillo 2018 refer throughout to 'the BCG(s)' and DeMaio et al. 2018 exclude MACS0717 outright for having "
    "'no clear central BCG'. For those four clusters a single brightest galaxy is NOT an adequate description of "
    "the central galaxy population; consult the full 391-row bCG file. *** "
    "BAND: HST/WFC3-IR F160W and HST/ACS F814W total fluxes, ZP(AB)=25, Galactic-extinction corrected. "
    "The same bCG+ICL subtraction caveat as the parent file applies: this catalogue removes the ICL by "
    "construction and cannot be used to measure it. No light profile, no stellar mass, no Sersic parameters.",
    "bcg/raw/shipley2018_clugal_bCG_raw.tsv")

print("\nPART 3 COMPLETE")
