# -*- coding: utf-8 -*-
"""Product 2 (BCG light profiles) -- Donzelli+2011 and Lauer+2014, incl. Abell 2029."""
import os, re, json, hashlib, datetime

ROOT = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
NOW  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

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
    lines = [l.rstrip("\r") for l in open(path, encoding="utf-8", errors="replace").read().split("\n")]
    bad = [l for l in lines if "Error=Table or Catalog not found" in l or "does not exist in catalog" in l]
    if bad: raise RuntimeError("VizieR NOT FOUND: %s :: %s" % (path, bad[0]))
    body = [l for l in lines if not l.startswith("#")]
    di = next(i for i, l in enumerate(body) if l.startswith("---") and "\t" in l)
    header, units = body[di - 2].split("\t"), body[di - 1].split("\t")
    rows = [l.split("\t") for l in body[di + 1:] if l.strip() and not l.startswith("#")]
    return header, units, [r for r in rows if len(r) >= len(header) - 2]

print("=" * 70); print("PRODUCT 2 -- BCG light profiles (Donzelli+2011, Lauer+2014)  ", NOW); print("=" * 70)

# ---------------------------------------------------------------
# 8. DONZELLI, MURIEL & MADRID 2011, ApJS 195, 15  (J/ApJS/195/15/table2)
# ---------------------------------------------------------------
print("\n[8] Donzelli+2011 BCG luminosity profiles (Rc band)")
raw = os.path.join(ROOT, "bcg", "raw", "donzelli2011_table2_raw.tsv")
hdr, units, rows = parse_vizier_tsv(raw)
assert len(rows) == 430, "Donzelli2011 expected 430 BCGs (ReadMe: table2.dat 430 records; abstract: 'a total of 430 brightest cluster galaxies'), got %d" % len(rows)
out = os.path.join(ROOT, "bcg", "donzelli2011_bcg_profiles.tsv")
write_tsv(out, hdr, rows)
DESC = {
 "Name": "Abell cluster name (ANNNN)",
 "mue": "Effective surface magnitude of the Sersic component (Rc band)",
 "re": "Effective radius of the Sersic component",
 "n": "Sersic index n",
 "mu0": "Central Rc surface magnitude of the OUTER EXPONENTIAL component (blank = single-component BCG)",
 "r0": "Scale length of the outer exponential component (blank = single-component BCG)",
 "MAGS": "Absolute Rc magnitude of the Sersic component",
 "MAGexp": "Absolute Rc magnitude of the exponential component (0/blank = none fitted)",
 "MAGtot": "Total Rc absolute magnitude of the BCG",
 "S/e": "Sersic-to-exponential component light ratio (blank = single-component BCG)",
 "alpha": "d log(Lm)/d log(rm), the log-slope of the metric curve of growth at r=14.5 kpc",
 "elli": "Inner ellipticity, measured at 10 arcsec (axis ratio q = 1 - elli)",
 "ello": "Outer ellipticity, measured at ~23-24 mag/arcsec2 (axis ratio q = 1 - ello)",
 "PAi": "Inner isophote position angle, N through E (deg)",
 "PAo": "Outer isophote position angle, N through E (deg)",
 "MAGM": "Metric absolute Rc magnitude within a circular aperture of radius 14.5 kpc",
}
cols = [{"name": h, "unit": (units[i].strip() if i < len(units) else "")} for i, h in enumerate(hdr)]
for c in cols:
    if c["name"] in DESC: c["description"] = DESC[c["name"]]
DONZ_NOTE = (
 "*** PRODUCT 2 (BCG light profile). Donzelli, Muriel & Madrid 2011, ApJS 195, 15. "
 "INCLUDES ABELL 2029 (Name='A2029'). Also covers 430 Abell-cluster BCGs generally; the six HFF clusters are "
 "NOT in it (they are not in the low-z Abell BCG sample). "
 "BAND: Cousins Rc, ground-based CCD imaging. "
 "SB LIMIT: 24.5 Rc mag/arcsec2 (stated in the paper abstract) -- much shallower than Kluge+2020 (30 g' "
 "mag/arcsec2), so Donzelli's outer profile is far less well constrained than Kluge's for the same galaxy. "
 "PROFILE PARAMETERS SUPPLIED: Sersic index n, effective radius re (kpc), effective surface brightness mue, "
 "inner and outer ellipticity (axis ratio q = 1 - e), inner and outer isophote position angles, total absolute "
 "magnitude, and a metric magnitude inside r = 14.5 kpc. This is a fitted mu(r) description, not a tabulated mu(r). "
 "ICL/BCG SEPARATION METHOD: for the 205 of 430 BCGs (~48%) that need two components, the fit is an inner Sersic "
 "PLUS AN OUTER EXPONENTIAL, and the outer exponential is the extended/envelope (ICL-like) component; 'S/e' is the "
 "Sersic-to-exponential light ratio. Kluge et al. 2021 tabulate this same decomposition as the literature "
 "'S+Exp' ICL method giving f_ICL = 40 +/- 14 per cent. "
 "*** FOR ABELL 2029 SPECIFICALLY: mu0, r0, MAGexp and S/e are ALL BLANK, i.e. A2029/IC 1101 is classified as a "
 "SINGLE-COMPONENT BCG and NO exponential/envelope component was fitted. Therefore this catalogue yields NO ICL "
 "decomposition and NO ICL fraction for A2029 -- only the single-Sersic BCG+envelope description "
 "(mue=26.30 Rc mag/arcsec2, re=439.41 kpc, n=5.78, MAGtot=-27.40, elli=0.26, ello=0.52, PAi=20.5, PAo=26.6, "
 "MAGM=-23.58). This independently reproduces the Kluge+2020 result that A2029 is a single-Sersic system. *** "
 "MASS-MODEL DEPENDENCE: none -- purely photometric.")
manifest(out, "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "-source=J/ApJS/195/15/table2&-out=**&-out.max=unlimited",
    len(rows), cols,
    "VizieR asu-tsv download of J/ApJS/195/15 table2. Row count asserted ==430 against BOTH the ReadMe file "
    "summary (table2.dat, 430 records) and the paper abstract ('a total of 430 brightest cluster galaxies'). "
    "Response verified to contain real rows and no 'Table or Catalog not found' error.",
    DONZ_NOTE, "bcg/raw/donzelli2011_table2_raw.tsv",
    {"catalogue_id_echoed": "J/ApJS/195/15/table2",
     "vizier_readme": "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/195/15/ReadMe"})

iN = hdr.index("Name")
a = [r for r in rows if r[iN].strip() == "A2029"]
assert len(a) == 1, "expected 1 A2029 row in Donzelli, got %d" % len(a)
out2 = os.path.join(ROOT, "bcg", "donzelli2011_bcg_A2029.tsv")
write_tsv(out2, hdr, a)
manifest(out2, "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "-source=J/ApJS/195/15/table2&-out=**&-out.max=unlimited  [then filtered Name=='A2029']",
    1, cols, "Single-row extract (Name=='A2029') from donzelli2011_bcg_profiles.tsv.",
    DONZ_NOTE, "bcg/raw/donzelli2011_table2_raw.tsv")

# ---------------------------------------------------------------
# 9. LAUER et al. 2014, ApJ 797, 82  (J/ApJ/797/82/bcg)
# ---------------------------------------------------------------
print("\n[9] Lauer+2014 BCG metric photometry")
raw = os.path.join(ROOT, "bcg", "raw", "lauer2014_bcg_raw.tsv")
hdr, units, rows = parse_vizier_tsv(raw)
assert len(rows) == 433, "Lauer2014 expected 433 BCGs (abstract: 'photometry and spectroscopy of 433 z<=0.08 BCGs'), got %d" % len(rows)
out = os.path.join(ROOT, "bcg", "lauer2014_bcg_metric_photometry.tsv")
write_tsv(out, hdr, rows)
DESC2 = {
 "Abell": "Abell cluster number", "RAJ2000": "BCG right ascension (J2000)",
 "DEJ2000": "BCG declination (J2000)", "VBCG": "BCG heliocentric velocity",
 "VCl": "Cluster mean velocity", "SigCl": "Cluster velocity dispersion",
 "Ng": "Number of cluster galaxies used", "AB": "Galactic absorption A_B",
 "Alpha": "Log-slope of the BCG photometric curve of growth at the metric radius r_m",
 "Lm": "Absolute metric luminosity within r_m (aperture radius 14.3 kpc)",
 "Lm-2rm": "Absolute metric luminosity within 2 r_m (28.6 kpc)",
 "Lm-4rm": "Absolute metric luminosity within 4 r_m (57.2 kpc)",
 "Sigma": "BCG central stellar velocity dispersion",
 "rx": "Radial offset of the BCG from the X-ray cluster centre",
 "Xref": "X-ray source: R=ROSAT, C=Chandra", "Notes": "BCG common name",
}
cols = [{"name": h, "unit": (units[i].strip() if i < len(units) else "")} for i, h in enumerate(hdr)]
for c in cols:
    if c["name"] in DESC2: c["description"] = DESC2[c["name"]]
LAUER_NOTE = (
 "*** PRODUCT 2 (BCG aperture photometry / coarse curve of growth). Lauer, Postman, Strauss, Graves & Chisari "
 "2014, ApJ 797, 82. Full-sky survey of 433 Abell-cluster BCGs at z<=0.08. "
 "INCLUDES ABELL 2029 (Abell=2029), whose BCG is identified in the 'Notes' column as I1101 = IC 1101, at "
 "RA=227.7337 deg, Dec=+05.7444 deg, with an X-ray centre offset rx = 1 kpc. "
 "The six HFF clusters are NOT in this sample (all are far above z=0.08). "
 "PROFILE INFORMATION: three nested CIRCULAR APERTURE absolute magnitudes -- Lm within the metric radius "
 "r_m = 14.3 kpc, Lm-2rm within 2 r_m, and Lm-4rm within 4 r_m -- plus Alpha, the logarithmic slope "
 "d log(L_m)/d log(r_m) of the curve of growth at r_m. For A2029: Alpha=0.935, Lm=-23.731, Lm-2rm=-24.385, "
 "Lm-4rm=-24.924 mag. This is a 3-point curve of growth, NOT a tabulated mu(r) and NOT a Sersic fit. "
 "*** BAND CAVEAT: the CDS ReadMe does NOT state the photometric band for Lm. This is the Postman & Lauer BCG "
 "survey, whose photometry is Kron-Cousins R; treat the band as R but VERIFY against the paper before using the "
 "magnitudes on an absolute scale. *** "
 "ICL/BCG SEPARATION METHOD: NONE. These are fixed-aperture magnitudes of the BCG with no ICL decomposition; "
 "the outer aperture (57.2 kpc) necessarily contains some ICL. Not an ICL measurement. "
 "MASS-MODEL DEPENDENCE: none for the photometry. SigCl and Sigma are dynamical measurements, not mass models.")
manifest(out, "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "-source=J/ApJ/797/82/bcg&-out=**&-out.max=unlimited",
    len(rows), cols,
    "VizieR asu-tsv download of J/ApJ/797/82 table 'bcg' (the CDS table name is 'bcg', NOT 'table7' -- querying "
    "'table7' returns '#INFO Error=Table table7 does not exist in catalog'). Row count asserted ==433 against the "
    "paper abstract ('photometry and spectroscopy of 433 z<=0.08 brightest cluster galaxies').",
    LAUER_NOTE, "bcg/raw/lauer2014_bcg_raw.tsv",
    {"catalogue_id_echoed": "J/ApJ/797/82/bcg",
     "vizier_readme": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/797/82/ReadMe"})

iA = hdr.index("Abell")
a = [r for r in rows if r[iA].strip() == "2029"]
assert len(a) == 1, "expected 1 A2029 row in Lauer, got %d" % len(a)
out2 = os.path.join(ROOT, "bcg", "lauer2014_bcg_A2029_IC1101.tsv")
write_tsv(out2, hdr, a)
manifest(out2, "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "-source=J/ApJ/797/82/bcg&-out=**&-out.max=unlimited  [then filtered Abell=='2029']",
    1, cols, "Single-row extract (Abell=='2029') from lauer2014_bcg_metric_photometry.tsv.",
    LAUER_NOTE, "bcg/raw/lauer2014_bcg_raw.tsv")

print("\nPART 4 COMPLETE")
