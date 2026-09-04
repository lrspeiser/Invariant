# -*- coding: utf-8 -*-
import requests, os, json, hashlib, datetime, re
from astropy.io import fits

BASE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
MEM = os.path.join(BASE, "members")
ROOT = "https://archive.stsci.edu/hlsps/buffalo"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


FIELDS = [("abell2744", "A2744", "Abell 2744"),
          ("abell370", "A370", "Abell 370"),
          ("abells1063", "AS1063", "Abell S1063 / RXC J2248.7-4431"),
          ("macs0416", "MACS0416", "MACS J0416.1-2403"),
          ("macs0717", "MACS0717", "MACS J0717.5+3745"),
          ("macs1149", "MACS1149", "MACS J1149.5+2223")]

NOTE = ("Pagul et al., BUFFALO (Beyond Ultra-deep Frontier Fields And Legacy Observations) HLSP v2.0 photometric catalogue "
        "for {full}, over the WIDER BUFFALO footprint (about 4x the HFF area). Detection on an IR-weighted stack with "
        "SExtractor 2.25.0; all images PSF-matched to F160W; intracluster light and bright cluster galaxies subtracted with "
        "GALFIT / GALAPAGOS-M before photometry. "
        "MEASURED: ALPHA_J2000_DET / DELTA_J2000_DET (RA/Dec in deg from the stacked detection image), isophotal fluxes and "
        "errors FLUX_ISO_*/FLUXERR_ISO_* in uJy for up to 15 bands (F275W F336W F435W F475W F606W F625W F814W F105W F110W "
        "F125W F140W F160W, Ks, IRAC1, IRAC2) -> magnitudes and colours (fluxes are MW-extinction corrected; the LePhare "
        "photometric offsets listed in the readme are NOT applied); ZSPEC = compiled SPECTROSCOPIC redshift where available "
        "with quality flag ZSPEC_Q, provenance ZSPEC_REF (BUF = Lagattuta et al. 2019 MUSE, GLS = GLASS compilation, NED) and "
        "ZSPEC_SEP (match separation, arcsec). "
        "MODEL-DERIVED: ZCHI2 (LePhare photometric redshift from minimum chi2 -- the release recommends this one), CHI2_RED, "
        "ZPDF with 68 per cent limits ZPDF_LOW/ZPDF_UPP, ZSECOND, EXT_LAW, E_BV. "
        "Missing bands are set to -99.9. NO membership flag, NO stellar mass, NO Sersic parameters (no n, no Re, no axis "
        "ratio, no position angle) -- this catalogue is photometry plus redshift only.")

res = {}
for d, short, full in FIELDS:
    url = "%s/%s/catalogs/pagul-v2.0/hlsp_buffalo_hst_ir-weighted_%s_multi_v2.0_catalog.fits" % (ROOT, d, d)
    out = os.path.join(MEM, "%s_BUFFALO_Pagul_v2.0_catalog.fits" % short)
    r = requests.get(url, timeout=900)
    if r.status_code != 200:
        print("  %-10s HTTP %d  NOT FOUND" % (short, r.status_code))
        res[short] = "NOT_FOUND(HTTP%d)" % r.status_code
        continue
    with open(out, "wb") as f:
        f.write(r.content)
    with fits.open(out) as hd:
        t = hd[1]
        nrow = t.header["NAXIS2"]
        cols = [{"name": c.name, "unit": (c.unit or "")} for c in t.columns]
    man = {
        "file": os.path.basename(out),
        "source_url": url,
        "exact_query": "HTTP GET %s (whole file, no query parameters)" % url,
        "retrieved_utc": utcnow(),
        "sha256": sha256f(out),
        "bytes": os.path.getsize(out),
        "row_count": int(nrow),
        "column_count": len(cols),
        "columns": cols,
        "extraction": ("Binary FITS file downloaded verbatim and stored unmodified. row_count read from NAXIS2 of HDU 1 and "
                       "column_count/columns from that HDU's column definitions using astropy.io.fits; no rows were parsed, "
                       "filtered or rewritten."),
        "note": NOTE.format(full=full),
        "raw_response_file": os.path.basename(out) + "  (this file IS the verbatim upstream bytes)",
        "readme_url": "%s/%s/catalogs/pagul-v2.0/hlsp_buffalo_hst_ir-weighted_%s_multi_v2.0_readme.txt" % (ROOT, d, d),
    }
    json.dump(man, open(out + ".manifest.json", "w", encoding="utf-8"), indent=2)
    print("  %-10s rows=%-7d cols=%-4d bytes=%d" % (short, nrow, len(cols), os.path.getsize(out)))
    res[short] = (int(nrow), len(cols))
print(json.dumps(res, indent=1))
