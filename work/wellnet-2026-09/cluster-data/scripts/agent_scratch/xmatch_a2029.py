# -*- coding: utf-8 -*-
"""Positional cross-match of the A2029 spectroscopic members (Sohn+2019, ApJ 871,129,
membership flag Mm='Y') against the Simard+2011 pure-Sersic structural catalogue
retrieved as a 110 arcmin cone about the cluster centre.

Writes a joined CSV that keeps EVERY member row, matched or not, so that unmatched
members are visible rather than silently dropped.
"""
import os, csv, json, math, hashlib, datetime

MEM = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data\members"
MATCH_RADIUS_ARCSEC = 1.5


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_tsv(path):
    L = open(path, errors="replace").read().split("\n")
    b = [l for l in L if not l.startswith("#")]
    while b and b[0].strip() == "":
        b.pop(0)
    hdr = b[0].split("\t")
    rows = [l.split("\t") for l in b[3:] if l.strip() != ""]
    return hdr, rows


# --- Simard pure-Sersic cone ---
sh, sr = read_tsv(os.path.join(MEM, "A2029field_Simard2011_ApJS196_11_table3_pureSersic_cone110arcmin.raw.tsv"))
sRA, sDE = sh.index("_RA"), sh.index("_DE")
KEEP = ["objID", "z", "Sp", "Scale", "gg2d", "rg2d", "Rhlr", "Rchl_r", "Rhlg", "Rchl_g", "e", "e_e", "phi", "e_phi", "ng", "e_ng"]
kidx = [sh.index(k) for k in KEEP]
simard = []
for r in sr:
    try:
        simard.append((float(r[sRA]), float(r[sDE]), r))
    except ValueError:
        pass
print("Simard cone rows usable: %d / %d" % (len(simard), len(sr)))

# bin by declination for a cheap search
BIN = 0.02
buckets = {}
for i, (ra, de, r) in enumerate(simard):
    buckets.setdefault(int(de / BIN), []).append(i)


def match(ra, de):
    best, bd = None, 1e9
    b0 = int(de / BIN)
    for b in (b0 - 1, b0, b0 + 1):
        for i in buckets.get(b, []):
            ra2, de2, _ = simard[i]
            d = math.hypot((ra - ra2) * math.cos(math.radians(de)), de - de2) * 3600.0
            if d < bd:
                bd, best = d, i
    return (best, bd) if bd <= MATCH_RADIUS_ARCSEC else (None, bd)


# --- Sohn+2019 ApJ 871,129 redshift catalogue, Mm flag ---
mh, mr = read_tsv(os.path.join(MEM, "A2029_Sohn2019_ApJ871_129_table1_redshift_catalog.raw.tsv"))
iRA, iDE, iMm, iZ, iR, iOB = (mh.index("RAJ2000"), mh.index("DEJ2000"), mh.index("Mm"),
                              mh.index("z"), mh.index("rmag"), mh.index("objID"))

out = os.path.join(MEM, "A2029_members_Sohn2019_x_Simard2011_structural.csv")
HDR = (["sohn_objID", "RAJ2000", "DEJ2000", "z", "rmag", "member_Mm",
        "simard_matched", "match_sep_arcsec", "objID_agrees"] + ["simard_" + k for k in KEEP])
nm = {"Y": 0, "N": 0}
matched = {"Y": 0, "N": 0}
seps = []
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(HDR)
    for r in mr:
        try:
            ra, de = float(r[iRA]), float(r[iDE])
        except ValueError:
            continue
        mm = r[iMm].strip()
        nm[mm] = nm.get(mm, 0) + 1
        i, d = match(ra, de)
        row = [r[iOB].strip(), r[iRA].strip(), r[iDE].strip(), r[iZ].strip(), r[iR].strip(), mm]
        if i is None:
            row += ["0", "%.3f" % d if d < 1e8 else "", ""] + [""] * len(KEEP)
        else:
            matched[mm] = matched.get(mm, 0) + 1
            seps.append(d)
            src = simard[i][2]
            agree = "1" if src[sh.index("objID")].strip() == r[iOB].strip() else "0"
            row += ["1", "%.3f" % d, agree] + [src[j].strip() for j in kidx]
        w.writerow(row)

print("\nSohn+2019 ApJ871,129 catalogue: members(Mm=Y)=%d  non-members(Mm=N)=%d" % (nm.get("Y", 0), nm.get("N", 0)))
print("Matched to Simard within %.1f arcsec:  members=%d (%.1f%%)  non-members=%d (%.1f%%)"
      % (MATCH_RADIUS_ARCSEC, matched.get("Y", 0), 100.0 * matched.get("Y", 0) / max(nm.get("Y", 1), 1),
         matched.get("N", 0), 100.0 * matched.get("N", 0) / max(nm.get("N", 1), 1)))
print("UNMATCHED members = %d (%.1f%%)  -- retained in the output file with blank structural columns"
      % (nm.get("Y", 0) - matched.get("Y", 0),
         100.0 * (nm.get("Y", 0) - matched.get("Y", 0)) / max(nm.get("Y", 1), 1)))
if seps:
    seps.sort()
    print("match separations: median=%.3f  90th pct=%.3f  max=%.3f arcsec" % (seps[len(seps) // 2], seps[int(0.9 * len(seps))], seps[-1]))

nrows = sum(1 for _ in open(out, encoding="utf-8")) - 1
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
man = {
    "file": os.path.basename(out),
    "source_url": "DERIVED: positional join of two retrieved files (no new network fetch)",
    "exact_query": ("Left join, keeping every row of A2029_Sohn2019_ApJ871_129_table1_redshift_catalog.raw.tsv, onto "
                    "A2029field_Simard2011_ApJS196_11_table3_pureSersic_cone110arcmin.raw.tsv by nearest neighbour on "
                    "(RAJ2000, DEJ2000) vs (_RA, _DE) with a %.1f arcsec maximum separation." % MATCH_RADIUS_ARCSEC),
    "retrieved_utc": utcnow(),
    "sha256": h,
    "bytes": os.path.getsize(out),
    "row_count": nrows,
    "column_count": len(HDR),
    "columns": [{"name": c, "unit": ""} for c in HDR],
    "extraction": ("Nearest-neighbour positional cross-match, flat-sky separation with a cos(Dec) factor on the RA "
                   "difference, match radius %.1f arcsec. EVERY Sohn row is retained: simard_matched=0 rows carry blank "
                   "structural columns rather than being dropped. Statistics at build time: "
                   "%d members (Mm=Y) of which %d matched (%.1f per cent), %d non-members (Mm=N) of which %d matched; "
                   "median match separation %.3f arcsec, maximum %.3f arcsec. "
                   "objID_agrees compares the SDSS objID of the two catalogues and is 0 for essentially all rows because "
                   "Simard+2011 carries SDSS DR7 objIDs while Sohn+2019 carries later-release objIDs -- the POSITIONAL "
                   "match, not the objID, is the join key."
                   % (MATCH_RADIUS_ARCSEC, nm.get("Y", 0), matched.get("Y", 0),
                      100.0 * matched.get("Y", 0) / max(nm.get("Y", 1), 1), nm.get("N", 0), matched.get("N", 0),
                      seps[len(seps) // 2] if seps else -1, seps[-1] if seps else -1)),
    "note": ("ABELL 2029 member galaxies with STRUCTURAL PARAMETERS attached. Left table: Sohn et al. 2019 ApJ 871, 129 "
             "Table 1 (RA/Dec, redshift, SDSS r magnitude, and the caustic MEMBERSHIP FLAG Mm = Y/N). Right table: "
             "Simard et al. 2011 ApJS 196, 11 Table 3, the PURE SERSIC GIM2D decompositions of SDSS DR7, cone-searched "
             "within 110 arcmin of the cluster centre. "
             "MEASURED (Sohn): RA, Dec, spectroscopic redshift, r magnitude, membership flag. "
             "MEASURED (Simard, from 2D GIM2D surface-brightness fits to SDSS g and r images): "
             "simard_ng = Sersic index n with error; simard_e = ellipticity (axis ratio q = 1 - e) with error; "
             "simard_phi = position angle (deg) with error; simard_gg2d / simard_rg2d = pure-Sersic model g and r "
             "magnitudes (-> g-r colour). "
             "SEMI-MODEL-DERIVED: simard_Rhlr / simard_Rhlg (semi-major-axis half-light radius) and simard_Rchl_r / "
             "simard_Rchl_g (CIRCULARISED half-light radius) are quoted in kpc -- the fitted angular size is measured but "
             "the conversion to kpc uses the redshift and an assumed cosmology (simard_Scale is the kpc/arcsec factor). "
             "CAVEAT: Simard+2011 fits are ground-based SDSS imaging at roughly 1.4 arcsec seeing, so at z=0.078 the "
             "structural parameters are far coarser than the HST-based Granata+2026 fits used for the HFF clusters; the "
             "two are NOT on the same measurement footing and should not be pooled without care."),
    "raw_response_file": "A2029field_Simard2011_ApJS196_11_table3_pureSersic_cone110arcmin.raw.tsv",
    "inputs": ["A2029_Sohn2019_ApJ871_129_table1_redshift_catalog.raw.tsv",
               "A2029field_Simard2011_ApJS196_11_table3_pureSersic_cone110arcmin.raw.tsv"],
}
json.dump(man, open(out + ".manifest.json", "w", encoding="utf-8"), indent=2)
print("\nwrote", os.path.basename(out), nrows, "rows")
